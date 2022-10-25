# sepparates value head from the model so that it can be a drop in transformer where output_hidden_states is an option
from pathlib import Path
import numpy as np
import torch.nn.functional as F
from torch.optim import Adam
import torch
import collections
import time
import random
from argparse import Namespace

from agents import VectorNLPAgent
from datastructures import RejectDataset
from datastructures import RejectionBuffer
from datastructures import ReplayBuffer
from datastructures import LineBuffer
from datastructures import RejectDatasetCollator

import pytorch_lightning as pl
from pytorch_lightning import seed_everything
import argparse
from typing import Tuple, List
import torch.optim as optim
from torch.optim import Optimizer
from deepspeed.ops.adam import DeepSpeedCPUAdam
from deepspeed.ops.adam import FusedAdam
from torch.utils.data import DataLoader
from torchinfo import summary

from trlTrainer import TRLTrainer
from games import VectorPlayer, getEnvs

from transformers import DataCollatorForLanguageModeling
from games import GameReward, WinReward, LivingReward, InvalidReward, LetterReward

from core import (logprobs_from_logits,
                  whiten,
                  clip_by_value,
                  entropy_from_logits,
                  flatten_dict,
                  average_torch_dicts,
                  stats_to_np,
                  stats_to_cpu,
                  stack_dicts,
                  add_suffix,
                  WANDB_PADDING,
                  pad_mask,
                  getKW)


class RejectionTuner(TRLTrainer):
    default_params = {
        "alg_name": "reject",
        "lr": 1.41e-5,
        "reference": True,
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
        "epochs_per_game": 4,
        "clear_buffer_each_game": True,
        "train_generation": False,
        "game_gamma": 0.8,
        "few_shot": 1,
        # compat with ppo
        'vf_coef': 0
    }

    def __init__(self, model_name, **params):
        super().__init__(model_name, **params)
        self.trainer_buffer = None
        self.rejectRatio = 0
        self.agent_buffer = ReplayBuffer(self.params["game_batch_size"])

        self.playerKWArgs = getKW(exTurns=0.25)
        self.agentKWArgs = getKW(useUnfinished=True, GAMMA=self.params["game_gamma"], MEMORY_LEN=self.params["few_shot"])

    def getDevice(self):
        return self.device

    def __call__(self, input_ids, **kwargs):
        return self.forward(input_ids, **kwargs)
    
    def __dataloader(self) -> DataLoader:
        self.data_collator = DataCollatorForLanguageModeling(self.tokenizer, mlm=False)
        self.trainer_buffer = RejectionBuffer(min=False, rank=self.trainer.global_rank,
                                              world_size=self.trainer.world_size)
        dataset = RejectDataset(self.trainer_buffer, sample_size=self.params['batch_size'], rank=self.trainer.global_rank, world_size=self.trainer.world_size)
        dataloader = DataLoader(dataset=dataset,
                                batch_size=self.params['forward_batch_size'],
                                collate_fn=RejectDatasetCollator(text_collator=self.data_collator)
                                )
        return dataloader

    def train_dataloader(self) -> DataLoader:
        """Get train loader"""
        return self.__dataloader()

    def configure_optimizers(self) -> List[Optimizer]:
        """ Initialize Adam optimizer"""
        # self.optimizer = Adam(model.parameters(), lr=self.params['lr'])
        # optimizer = Adam(list(self.model.parameters()) + list(self.valueHead.parameters()), lr=self.params['lr'])
        if self.deepspeed_offload:
            optimizer = DeepSpeedCPUAdam(self.model.parameters(), lr=self.params['lr'])
        else:
            optimizer = FusedAdam(self.model.parameters(), lr=self.params['lr'])

        return [optimizer]

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
                ref_logits = self.ref_model(input_ids).logits
                output["ref_logits"] = ref_logits

        if outputVals:
            v = torch.zeros(list(logits.shape[0:2])+[1])
            output["values"] = v

        return output

    def batched_forward_pass(self, queries, responses, outputLogits=True, outputVals=True, outputRef=True):
        """Calculate model outputs in multiple batches."""
        bs = self.params['game_batch_size']
        fbs = self.params['forward_batch_size']
        all_logprobs = []
        all_ref_logprobs = []
        all_values = []

        output = {}

        for i in range(int(bs / fbs)):
            query_batch = queries[i * fbs:(i + 1) * fbs]
            response_batch = responses[i * fbs:(i + 1) * fbs]
            model_input = self.data_collator([torch.cat([q, r]) for q, r in zip(query_batch, response_batch)])
            input_ids = model_input["input_ids"]
            attention_mask = pad_mask(input_ids, self.tokenizer.pad_token_id)
            # print("forward pass mask ", attention_mask)

            with torch.no_grad():
                # # logits, _, v = self.model(input_ids)
                # lmOut = self.model(input_ids, output_hidden_states=True)
                # # print(dir(lmOut))
                # logits, hidden_state = lmOut.logits, lmOut.hidden_states[-1]
                # v = self.valueHead(hidden_state)
                # # ref_logits, _, _ = self.ref_model(input_ids)
                # ref_logits = self.ref_model(input_ids).logits
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
                # right pad
                # start = len(query_batch[j]) - 1
                # end = len(query_batch[j]) + len(response_batch[j]) - 1
                # all_values.append(v[j, start:end])
                # all_logprobs.append(logprobs[j, start:end])
                # all_ref_logprobs.append(ref_logprobs[j, start:end])
                # left pad
                gen_len = len(response_batch[j])
                if outputVals:
                    all_values.append(v[j, -(gen_len+1):-1])
                # logits already shifted
                if outputLogits:
                    all_logprobs.append(logprobs[j, -gen_len:])
                if outputRef:
                    all_ref_logprobs.append(ref_logprobs[j, -gen_len:])

        rem = bs % fbs
        if rem != 0:
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
                # right pad
                # start = len(query_batch[j]) - 1
                # end = len(query_batch[j]) + len(response_batch[j]) - 1
                # all_values.append(v[j, start:end])
                # all_logprobs.append(logprobs[j, start:end])
                # all_ref_logprobs.append(ref_logprobs[j, start:end])
                # left pad
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

        if self.trainer.is_global_zero:
            if self.current_epoch % self.params['save_freq'] == 0:
                t = time.time()
                self.model.save_pretrained(f"checkpoints/{self.alg_name}_model_epoch_{self.current_epoch}")
                self.saveModelTime = time.time() - t

            if self.current_epoch % self.params['log_freq'] == 0:
                data, scores_kl = self.trainer_buffer.sample(self.params['batch_size'])
                scores, _, _, values_next, ret_cross, adv_cross, logprobs, ref_logprobs, values, rewards, non_score_reward = zip(*data)

                timing = dict()
                timing[f'time/{self.alg_name}/optimize_step'] = time.time() - self.epoch_time
                timing[f'time/{self.alg_name}/game_time'] = self.game_time

                timing['time/filesystem/save_model'] = self.saveModelTime
                timing['time/filesystem/save_stats'] = self.saveStatTime

                t = time.time()
                train_stats = stack_dicts(self.all_stats)
                self.all_stats = []

                train_stats['rejectRatio'] = self.rejectRatio
                train_stats['policy/ratio'] = torch.flatten(train_stats['policy/ratio']).unsqueeze(0)

                # print("kl list ", len(self.kl_ctl_rew.kl_list))
                stats = self.record_step_stats(scores=scores, logprobs=logprobs, ref_logprobs=ref_logprobs,
                                               non_score_reward=non_score_reward, train_stats=train_stats)
                stats = stats_to_np(stats)
                timing[f'time/{self.alg_name}/calc_stats'] = time.time() - t

                self.kl_ctl.update(stats['objective/kl'], self.params['log_freq'] * self.params['batch_size'])
                self.kl_ctl_rew.update(stats['objective/kl_rew'],
                                       self.params['log_freq'] * self.params['batch_size'] // self.params[
                                           f'epochs_per_game'])

                # timing[f'time/{self.alg_name}/total'] = time.time() - t0
                stats.update(timing)
                # print(stats)
                t = time.time()
                torch.save(stats, f"stats/{self.alg_name}_epoch_{self.current_epoch}-step_{self.global_step}.pt")
                self.saveStatTime = time.time() - t

        # stats are computed on process 1, update kl ctls on all processes
        kl_values = [self.kl_ctl.value, self.kl_ctl_rew.value]
        torch.distributed.broadcast_object_list(kl_values, src=0)
        if not self.trainer.is_global_zero:
            self.kl_ctl.updateValue(kl_values[0])
            self.kl_ctl_rew.updateValue(kl_values[1])

    def runGame(self):
        if self.params["clear_buffer_each_game"]:
            self.trainer_buffer.clear()
        self.agent_buffer.clear()

        # self is passing the model to do forward passes with
        self.player.runGame(self, self.params['game_batch_size'])
        self.agent.fillBuffer()
        self.kl_ctl_rew.kl_list = []
        scores, queries, responses, values_next, ret_cross, adv_cross, values, logprobs = self.agent_buffer.sample(
            self.params['game_batch_size'])

        # first part of original step, gets old logprobs and ref logprobs
        bs = self.params['batch_size']

        timing = dict()
        t0 = time.time()

        response_lengths = [len(r) for r in responses]

        t = time.time()
        # logprobs, ref_logprobs, values = self.batched_forward_pass(queries, responses)
        ref_logprobs = \
        self.batched_forward_pass(queries, responses, outputLogits=False, outputVals=False, outputRef=True)[
            "ref_logprobs"]

        timing[f'time/{self.alg_name}/forward_pass'] = time.time() - t

        # print("run game")
        # print(values)

        t = time.time()
        rewards, non_score_reward = self.compute_rewards(scores, logprobs, ref_logprobs)
        timing[f'time/{self.alg_name}/compute_rewards'] = time.time() - t

        scores_kl = []
        for i in range(self.params['game_batch_size']):
            query = queries[i]
            response = responses[i]
            # use returns discounted across multiple transitions

            score_kl = ret_cross[i].to(self.device)
            kl_rew = torch.sum(non_score_reward[i]).to(self.device)
            score_kl += kl_rew
            scores_kl.append(score_kl)
            # use only current rewards
            # scores_batch = scores[i]

        for lineItem, score_kl in zip(zip(scores, queries, responses, values_next, ret_cross, adv_cross, logprobs, ref_logprobs,
                            values, rewards, non_score_reward), scores_kl):
            self.trainer_buffer.append(lineItem, score_kl)

        start_n = len(self.trainer_buffer)
        self.trainer_buffer.reject(self.params["batch_size"] * self.trainer.world_size, threshType="top n")
        reject_n = len(self.trainer_buffer)
        if self.trainer.is_global_zero:
            print("rejection ", start_n, " ", reject_n)
            self.rejectRatio = torch.tensor(reject_n / start_n)

    # data is list of strings
    def training_step(self, batch, nb_batch):
        scores, queries, responses, model_input, lengths, values_next, ret_cross, adv_cross, old_logprobs, ref_logprobs, values, rewards, non_score_reward, reject_scores = batch
        fbs = scores.shape[0]

        loss_total = None
        input_ids = model_input["input_ids"]
        input_mask = pad_mask(input_ids, self.tokenizer.pad_token_id)
        query_ids = queries["input_ids"]
        query_mask = pad_mask(query_ids, self.tokenizer.pad_token_id)
        response_ids = responses["input_ids"]
        response_mask = pad_mask(response_ids, self.tokenizer.pad_token_id)

        lmout = self.forward(input_ids, outputVals=False, outputRef=False, attention_mask=input_mask)
        logits = lmout["logits"]
        logits_shifted = logits[:, :-1]
        targets = input_ids[:, 1:]
        target_mask = input_mask[:, 1:]

        logprob = logprobs_from_logits(logits, targets)

        ce_loss_batch = torch.tensor(0, device=self.device, dtype=logprob.dtype)
        kl_loss_batch = torch.tensor(0, device=self.device, dtype=logprob.dtype)
        train_stats = []

        if not self.params["train_generation"]:
            # print(logits_shifted.shape, targets.shape)
            # inputs batch x classes x seq dim
            # targets batch x seq dim
            ce_loss_batch = F.cross_entropy(torch.transpose(logits_shifted, 1,2), targets, ignore_index=self.tokenizer.pad_token_id)
        ce_loss = ce_loss_batch

        for i in range(fbs):
            querry_len = lengths[i][0]
            gen_len = lengths[i][1]
            total_len = lengths[i][2]

            targ = targets[i:i+1, -gen_len:]
            logit = logits_shifted[i:i+1, -gen_len:]
            logp = logprob[i:i+1, -gen_len:]

            ref_logp = ref_logprobs[i:i+1, :gen_len]
            old_logp = old_logprobs[i:i+1, :gen_len]

            if self.params["train_generation"]:
                ce_loss = F.cross_entropy(torch.transpose(logit, 1,2), targ, ignore_index=self.tokenizer.pad_token_id)
                ce_loss_batch += ce_loss

            kl = logp - ref_logp
            ratio = torch.exp(logp - old_logp)
            kl = kl * ratio
            # for stats and adaptive update
            self.kl_ctl.kl_list.append(kl.detach().to("cpu"))
            # mean across tokens
            kl_loss = torch.mean(kl)
            kl_loss_batch += kl_loss

            entropy = torch.mean(entropy_from_logits(logit))
            approxkl = .5 * torch.mean((logp - old_logp) ** 2)
            policykl = torch.mean(logp - old_logp)

            loss = ce_loss + self.kl_ctl.value * kl_loss

            stats = dict(
                loss=dict(policy=ce_loss, kl=kl_loss, total=loss),
                policy=dict(entropy=entropy, approxkl=approxkl, policykl=policykl, ratio=ratio),
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

        kl_loss_batch /= fbs
        if self.params["train_generation"]:
            ce_loss_batch /= fbs

        return ce_loss_batch + self.kl_ctl.value * kl_loss_batch


def train(model_name, single_game=False):
    from time import time

    UPDATE_FREQUENCY = 64
    FORWARD_BATCH = 4
    LOG_FREQUENCY = 1
    NUM_AGENTS = 4

    trainer = trlTrainer.getTrainer()

    print("rank out of world :", trainer.global_rank, " ", trainer.world_size)
    UPDATE_FREQUENCY = max(UPDATE_FREQUENCY // trainer.world_size, 2)
    FORWARD_BATCH = max(FORWARD_BATCH // trainer.world_size, 1)
    NUM_AGENTS = max(NUM_AGENTS // trainer.world_size, 1)

    
    params = {'batch_size': UPDATE_FREQUENCY // 2,
              'game_batch_size': UPDATE_FREQUENCY,
             'save_freq': SAVE_FREQUENCY,
             'log_freq': LOG_FREQUENCY,
             "forward_batch_size": FORWARD_BATCH,
              "num_agents": NUM_AGENTS,
              "single_game": single_game}
    reject_tuner = RejectionTuner(model_name, **params)


    trainer.fit(reject_tuner)


if __name__ == "__main__":
    import argparse

    seed_everything(42)

    # model_name = 'gpt2'
    # model_name = 'EleutherAI/gpt-j-6B'
    model_name = 'EleutherAI/gpt-neo-1.3B'
    # model_name = "EleutherAI/gpt-neox-20b"
    single_game = False

    Path("stats").mkdir(parents=True, exist_ok=True)
    Path("checkpoints").mkdir(parents=True, exist_ok=True)

    train(model_name, single_game=single_game)