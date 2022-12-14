from pathlib import Path
import time
from typing import List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torch.optim import Optimizer
from deepspeed.ops.adam import DeepSpeedCPUAdam
from deepspeed.ops.adam import FusedAdam

from torchinfo import summary
from transformers import DataCollatorForLanguageModeling

import trlTrainer
from trlTrainer import TRLTrainer

from datastructures import DictBuffer
from datastructures import LineBuffer
from datastructures import DecisionDataset, DecisionDatasetCollator

from agents import DecisionAgent
from games import VectorPlayer
from games import GameReward, WinReward, LivingReward, InvalidReward, LetterReward

from core import (logprobs_from_logits,
                  entropy_from_logits,
                  flatten_dict,
                  stats_to_np,
                  stats_to_cpu,
                  stack_stat_dicts,
                  pad_mask,
                  getKW)


class DecisionTuner(TRLTrainer):
    default_params = {
        "alg_name": "decision",
        "lr": 1.41e-5,
        "reference": False,
        # KL Calcuated per forward batch importance corrected exact gradients
        "adap_kl_ctrl": False,
        "init_kl_coef": 0.1,
        "target": 6,
        "horizon": 10000,
        # KL added to rewards at start of Reject Epochs
        "adap_kl_ctrl_rew": False,
        "init_kl_coef_rew": 0.0,
        "target_rew": 6,
        "horizon_rew": 10000,
        # end KL
        "batch_size": 256,
        "forward_batch_size": 16,
        "epochs_per_game": 1,
        "game_gamma": 0.8,
        "few_shot": 0,
        "prepend": "return",
        # compat with ppo
        'vf_coef': 0
    }

    def __init__(self, model_name, **params):
        super().__init__(model_name, **params)
        self.trainer_buffer = None
        self.agent_buffer = DictBuffer(self.params["batch_size"])

        gameRew = GameReward(value=1, num_agents=self.params["num_agents"])
        gameRew = WinReward(value=2, num_agents=self.params["num_agents"], parentReward=gameRew)
        gameRew = InvalidReward(value=-0.5, num_agents=self.params["num_agents"], parentReward=gameRew)

        letterRew = LetterReward(value=1, num_agents=self.params["num_agents"], letters=('e', 'E'))

        self.playerKWArgs = getKW(exTurns=0.0, rewardFunc=letterRew)
        self.agentKWArgs = getKW(useUnfinished=True, GAMMA=self.params["game_gamma"], MEMORY_LEN=self.params["few_shot"])

    def getDevice(self):
        return self.device

    def __call__(self, input_ids, **kwargs):
        return self.forward(input_ids, **kwargs)

    def setup(self, stage=None):
        super().setup(stage=stage)
        if self.params["single_game"]:
            # agent = NLPAgent(buffer, humanTurns=0)
            self.agent = DecisionAgent(self.agent_buffer, num_agents=self.params["num_agents"], rank=self.trainer.global_rank,
                                   world_size=self.trainer.world_size, **self.agentKWArgs)
            print("Training")
            self.agent.train()  # Tell the agent it should update its parameters.
            # player = Player(agent, "./games/tw-rewardsDense_goalDetailed.z8", verbose=False)  # Dense rewards game.
            self.player = VectorPlayer(self.agent, "./games/tw-rewardsDense_goalDetailed.z8", verbose=False,
                                  num_agents=self.params["num_agents"],
                                  rank=self.trainer.global_rank, world_size=self.trainer.world_size, **self.playerKWArgs)

        else:
            self.agent = DecisionAgent(self.agent_buffer, num_agents=self.params["num_agents"], rank=self.trainer.global_rank,
                                   world_size=self.trainer.world_size, **self.agentKWArgs)
            print("Training on 100 games")
            self.agent.train()  # Tell the agent it should update its parameters.
            self.player = VectorPlayer(self.agent, "./training_games/", verbose=False, num_agents=self.params["num_agents"],
                                  rank=self.trainer.global_rank, world_size=self.trainer.world_size, **self.playerKWArgs)  # Each game will be seen 5 times.
    
    def on_test_epoch_end(self):
        self.saveStats(filename="stats/test.pt")
    
    def on_train_epoch_end(self):
        for ctl in [self.kl_ctl, self.kl_ctl_rew]:
            if self.trainer.is_global_zero:
                gathered_kl = [None for i in range(self.trainer.world_size)]
            else:
                gathered_kl = None
            torch.distributed.gather_object(ctl.kl_list, object_gather_list=gathered_kl, dst=0)
            ctl.kl_list = []
            if self.trainer.is_global_zero:
                for kl in gathered_kl:
                    ctl.kl_list.extend(kl)

        if self.current_epoch % self.params['log_freq'] == 0:
            if self.trainer.is_global_zero:
                self.saveStats()
        
        # stats are computed on process 1, update kl ctls on all processes
        kl_values = [self.kl_ctl.value, self.kl_ctl_rew.value]
        torch.distributed.broadcast_object_list(kl_values, src=0)
        if not self.trainer.is_global_zero:
            self.kl_ctl.updateValue(kl_values[0])
            self.kl_ctl_rew.updateValue(kl_values[1])
    
    def saveStats(self, filename=None):
        if filename is None:
            filename = f"stats/{self.alg_name}_epoch_{self.current_epoch}-step_{self.global_step}.pt"
            
        data = self.trainer_buffer.sample(self.params['batch_size'])
        scores, ret_cross, _ = zip(*data)

        timing = dict()
        timing[f'time/{self.alg_name}/optimize_step'] = time.time() - self.epoch_time
        timing[f'time/{self.alg_name}/game_time'] = self.game_time

        timing['time/filesystem/save_model'] = self.saveModelTime
        timing['time/filesystem/save_stats'] = self.saveStatTime

        t = time.time()
        train_stats = stack_stat_dicts(self.all_stats)
        self.all_stats = []

        # print("kl list ", len(self.kl_ctl_rew.kl_list))
        stats = self.record_step_stats(scores=scores, train_stats=train_stats)
        stats = stats_to_np(stats)
        timing[f'time/{self.alg_name}/calc_stats'] = time.time() - t

        self.kl_ctl.update(stats['objective/kl'], self.params['log_freq'] * self.params['batch_size'])
        self.kl_ctl_rew.update(stats['objective/kl_rew'],
                                       self.params['log_freq'] * self.params['batch_size'] // self.params['epochs_per_game'])

        flatparams = flatten_dict(self.params, prefix="params/")
        flatcfg = flatten_dict(self.model.config.to_diff_dict(), prefix="config/")
        # timing[f'time/{self.alg_name}/total'] = time.time() - t0
        stats.update(timing)
        stats.update(flatparams)
        stats.update(flatcfg)
        # print(stats)
        t = time.time()
        torch.save(stats, filename)
        self.saveStatTime = time.time() - t

    def runGame(self):
        self.trainer_buffer.clear()
        self.agent_buffer.clear()

        # self is passing the model to do forward passes with
        self.player.runGame(self, self.params['batch_size'])
        self.agent.fillBuffer()
        torch.distributed.barrier()

        # may be short since not using unfinished
        while len(self.agent_buffer) < self.params['batch_size']:
            self.player.runGame(self, max(2, self.params['batch_size'] // 4))
            self.agent.fillBuffer()


        self.kl_ctl_rew.kl_list = []
        # scores, queries, responses, _values_next, ret_cross, _adv_cross, _old_values, _old_logprobs = self.agent_buffer.sample(
        #     self.params['batch_size'])

        exp_dict = self.agent_buffer.sample(self.params['batch_size'])
        
        scores = exp_dict["reward"]
        queries = exp_dict["prompt_tens"]
        responses = exp_dict["action_tens"]
        ret_cross = exp_dict["ret_cross"]

        # first part of original step, gets old logprobs and ref logprobs
        prepend = self.params["prepend"]
        for i in range(len(scores)):
            val = 0
            if prepend == "score":
                val = scores[i]
            elif prepend == "return":
                val = ret_cross[i]
            text = f"This play through has a {prepend} of {val}\n{queries[i]}{responses[i]}"
            input_ids = self.tokenizer(text, add_special_tokens=True, return_tensors="pt")['input_ids']
            self.trainer_buffer.append(scores, ret_cross, input_ids[0])

    def configure_optimizers(self) -> List[Optimizer]:
        """ Initialize Adam optimizer"""
        # self.optimizer = Adam(model.parameters(), lr=self.params['lr'])
        # optimizer = Adam(list(self.model.parameters()) + list(self.valueHead.parameters()), lr=self.params['lr'])
        if self.deepspeed_offload:
            optimizer = DeepSpeedCPUAdam(self.model.parameters(), lr=self.params['lr'])
        else:
            optimizer = FusedAdam(self.model.parameters(), lr=self.params['lr'])

        return [optimizer]

    def __dataloader(self) -> DataLoader:
        self.data_collator = DataCollatorForLanguageModeling(self.tokenizer, mlm=False)
        self.trainer_buffer = LineBuffer(self.params['batch_size'])
        dataset = DecisionDataset(self.trainer_buffer, self.params['batch_size'])
        dataloader = DataLoader(dataset=dataset,
                                batch_size=self.params['forward_batch_size'],
                                collate_fn=DecisionDatasetCollator(text_collator=self.data_collator)
                                )
        return dataloader

    def train_dataloader(self) -> DataLoader:
        """Get train loader"""
        return self.__dataloader()

    def test_dataloader(self) -> DataLoader:
        return self.__dataloader()

    # values variable for compatability
    def forward(self, input_ids, use_cache=False, past_key_values=None, outputVals=False, outputRef=False, attention_mask=None, outputLogits=True):
        output = {}
        if outputLogits or outputVals:
            if past_key_values is None:
                lmOut = self.model(input_ids, output_hidden_states=outputVals, use_cache=use_cache, attention_mask=attention_mask)
            else:
                lmOut = self.model(input_ids, output_hidden_states=outputVals, use_cache=use_cache,
                                   past_key_values=past_key_values, attention_mask=attention_mask)
            # print(dir(lmOut))
            if outputLogits:
                logits = lmOut.logits
                output["logits"] = logits

            if use_cache:
                cache = lmOut.past_key_values
                output["cache"] = cache

        if outputRef:
            with torch.no_grad():
                output["ref_logits"] = None

        if outputVals:
            v = torch.zeros(list(logits.shape[0:2])+[1])
            output["values"] = v

        return output

    def batched_forward_pass(self, queries, responses, outputLogits=True, outputVals=True, outputRef=True):
        """Calculate model outputs in multiple batches."""
        bs = self.params['batch_size']
        fbs = self.params['forward_batch_size']
        all_logprobs = []
        all_ref_logprobs = []
        all_values = []

        output = {}
        
        if self.trainer.global_rank == 0:
            print("batched forward pass ", flush=True)

        for i in range(int(bs / fbs)):
            query_batch = queries[i * fbs:(i + 1) * fbs]
            response_batch = responses[i * fbs:(i + 1) * fbs]
            model_input = self.data_collator([torch.cat([q, r]) for q, r in zip(query_batch, response_batch)])
            input_ids = model_input["input_ids"]
            attention_mask = pad_mask(input_ids, self.tokenizer.pad_token_id)
            # print("forward pass mask ", attention_mask)

            if self.trainer.global_rank == 0:
                print("\r", i, "/", int(bs / fbs), sep="", end="", flush=True)

            with torch.no_grad():
                input_ids = input_ids.to(self.device)
                lmout = self.forward(input_ids, outputVals=outputVals, outputRef=outputRef, outputLogits=outputLogits, attention_mask=attention_mask)

                if outputLogits:
                    logits = lmout["logits"]
                    logprobs = logprobs_from_logits(logits[:, :-1, :], input_ids[:, 1:])
                if outputRef:
                    ref_logits = lmout["ref_logits"]
                    ref_logprobs = logprobs_from_logits(ref_logits[:, :-1, :], input_ids[:, 1:])
                if outputVals:
                    v = lmout["values"]

            for j in range(fbs):
                # both logits and values are shifted 1 left from the input
                # left pad
                gen_len = len(response_batch[j])
                if outputVals:
                    all_values.append(v[j, -(gen_len + 1):-1])
                # logits already shifted
                if outputLogits:
                    all_logprobs.append(logprobs[j, -gen_len:])
                if outputRef:
                    all_ref_logprobs.append(ref_logprobs[j, -gen_len:])

        rem = bs % fbs
        if rem != 0:
            if self.trainer.global_rank == 0:
                print("remainder batch ", rem, sep="", end="", flush=True)
            query_batch = queries[-rem:]
            response_batch = responses[-rem:]
            input_ids = self.data_collator([torch.cat([q, r]) for q, r in zip(query_batch, response_batch)])[
                "input_ids"]
            attention_mask = pad_mask(input_ids, self.tokenizer.pad_token_id)

            with torch.no_grad():
                input_ids = input_ids.to(self.device)
                lmout = self.forward(input_ids, outputVals=outputVals, outputRef=outputRef, outputLogits=outputLogits,
                                     attention_mask=attention_mask)

                if outputLogits:
                    logits = lmout["logits"]
                    logprobs = logprobs_from_logits(logits[:, :-1, :], input_ids[:, 1:])
                if outputRef:
                    ref_logits = lmout["ref_logits"]
                    ref_logprobs = logprobs_from_logits(ref_logits[:, :-1, :], input_ids[:, 1:])
                if outputVals:
                    v = lmout["values"]

            for j in range(rem):
                # both logits and values are shifted 1 left from the input
                # remove left pad
                gen_len = len(response_batch[j])
                if outputVals:
                    all_values.append(v[j, -(gen_len + 1):-1])
                # logits already shifted
                if outputLogits:
                    all_logprobs.append(logprobs[j, -gen_len:])
                if outputRef:
                    all_ref_logprobs.append(ref_logprobs[j, -gen_len:])

        output["logprobs"] = all_logprobs
        output["values"] = all_values
        output["ref_logprobs"] = all_ref_logprobs
        return output

    # data is list of strings
    def training_step(self, batch, nb_batch):
        scores, ret_cross, input_ids = batch['input_ids']
        # fbs = input_ids.shape[0]

        input_mask = pad_mask(input_ids, self.tokenizer.pad_token_id)

        lmout = self.forward(input_ids, outputVals=False, outputRef=False, attention_mask=input_mask)
        logits = lmout["logits"]
        logits_shifted = logits[:, :-1]
        targets = input_ids[:, 1:]
        # target_mask = input_mask[:, 1:]

        logprob = logprobs_from_logits(logits, targets)

        ce_loss_batch = torch.tensor(0, device=self.device, dtype=logprob.dtype)
        train_stats = []

        ce_loss_batch = F.cross_entropy(torch.transpose(logits_shifted, 1,2), targets, ignore_index=self.tokenizer.pad_token_id)
        loss = ce_loss_batch

        stats = dict(
            loss=dict(total=loss)
        )
        train_stats.append(stats_to_cpu(flatten_dict(stats)))

        if self.trainer.is_global_zero:
            gathered_stats = [None for i in range(self.trainer.world_size)]
        else:
            gathered_stats = None
        torch.distributed.gather_object(train_stats, object_gather_list=gathered_stats, dst=0)
        if self.trainer.is_global_zero:
            for stats in gathered_stats:
                self.all_stats.extend(stats)

        return ce_loss_batch


def train(model_name, single_game=False):

    UPDATE_FREQUENCY = 64
    FORWARD_BATCH = 8
    LOG_FREQUENCY = 1
    NUM_AGENTS = 16

    trainer = trlTrainer.getTrainer()

    print("rank out of world :", trainer.global_rank, " ", trainer.world_size)
    UPDATE_FREQUENCY = max(UPDATE_FREQUENCY // trainer.world_size, 2)
    FORWARD_BATCH = max(FORWARD_BATCH // trainer.world_size, 1)
    NUM_AGENTS = max(NUM_AGENTS // trainer.world_size, 1)


    params = {'batch_size': UPDATE_FREQUENCY,
             'log_freq': LOG_FREQUENCY,
             "forward_batch_size": FORWARD_BATCH,
              "num_agents": NUM_AGENTS,
              "single_game": single_game}
    decision_tuner = DecisionTuner(model_name, **params)

    trainer.fit(decision_tuner)


if __name__ == "__main__":
    pl.seed_everything(2061630618)
    # seed_everything(42)

    MODEL_NAME = 'gpt2'
    # MODEL_NAME = 'gpt2-medium'
    # MODEL_NAME = 'EleutherAI/gpt-j-6B'
    # MODEL_NAME = 'EleutherAI/gpt-neo-1.3B'
    # MODEL_NAME = "EleutherAI/gpt-neox-20b"
    SINGLE_GAME = False

    Path("stats").mkdir(parents=True, exist_ok=True)
    Path("checkpoints").mkdir(parents=True, exist_ok=True)

    train(model_name=MODEL_NAME, single_game=SINGLE_GAME)
