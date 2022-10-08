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

from agents import VectorNLPAgent, NLPAgent
from datastructures import RejectDataset
from datastructures import RejectionBuffer
from datastructures import ReplayBuffer
from datastructures import LineBuffer
from datastructures import RejectDatasetCollator

import pytorch_lightning as pl
import argparse
from collections import OrderedDict, deque
from typing import Tuple, List
import torch.optim as optim
from torch.optim import Optimizer
from deepspeed.ops.adam import DeepSpeedCPUAdam
from torch.utils.data import DataLoader
from torchinfo import summary

from trlTrainer import TRLTrainer
from valueHead import ValueHead
from games import VectorPlayer, getEnvs, Player

from transformers import DataCollatorForLanguageModeling

from core import (logprobs_from_logits,
                  whiten,
                  clip_by_value,
                  entropy_from_logits,
                  flatten_dict,
                  average_torch_dicts,
                  stats_to_np,
                  stack_dicts,
                  add_suffix,
                  WANDB_PADDING,
                  pad_mask)

def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

class AdaptiveKLController:
    """
    Adaptive KL controller described in the paper:
    https://arxiv.org/pdf/1909.08593.pdf
    """

    def __init__(self, init_kl_coef, target, horizon):
        self.value = init_kl_coef
        self.target = target
        self.horizon = horizon
        self.kl_list = []

    def update(self, current, n_steps):
        target = self.target
        proportional_error = np.clip(current / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


class FixedKLController:
    """Fixed KL controller."""

    def __init__(self, kl_coef):
        self.value = kl_coef
        self.kl_list = []

    def update(self, current, n_steps):
        pass


class RejectionTuner(TRLTrainer):
    default_params = {
        "alg_name": "reject",
        "lr": 1.41e-5,
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
        # compat with ppo
        'vf_coef': 0
    }

    def __init__(self, model_name, player, buffer, agent, **params):
        super().__init__(model_name, player, buffer, agent, **params)
        self.trainer_buffer = RejectionBuffer(min=False)
        self.rejectRatio = 0

    def getDevice(self):
        return self.device

    def __call__(self, input_ids, **kwargs):
        return self.forward(input_ids, **kwargs)
    
    def __dataloader(self) -> DataLoader:
        dataset = RejectDataset(self.trainer_buffer, self.params['batch_size'])
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
        optimizer = DeepSpeedCPUAdam(self.model.parameters(), lr=self.params['lr'])
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
        if self.current_epoch % self.params['save_freq'] == 0:
            t = time.time()
            self.model.save_pretrained(f"checkpoints/{self.alg_name}_model_epoch_{self.current_epoch}")
            self.saveModelTime = time.time() - t

        if self.current_epoch % self.params['log_freq'] == 0:
            data, scores_kl = self.trainer_buffer.sample(self.params['batch_size'])
            scores, _, _, values_next, ret_cross, adv_cross, logprobs, ref_logprobs, values, rewards, non_score_reward = zip(*data)

            timing = dict()
            timing[f'time/{self.alg_name}/optimize_step'] = time.time() - self.epoch_time

            timing['time/filesystem/save_model'] = self.saveModelTime
            timing['time/filesystem/save_stats'] = self.saveStatTime

            t = time.time()
            train_stats = stack_dicts(self.all_stats)

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

    def runGame(self):
        if self.params["clear_buffer_each_game"]:
            self.trainer_buffer.clear()

        # self is passing the model to do forward passes with
        self.player.runGame(self, self.params['game_batch_size'])
        self.agent.fillBuffer()
        self.kl_ctl_rew.kl_list = []
        scores, queries, responses, values_next, ret_cross, adv_cross, values, logprobs = self.agent_buffer.sample(
            self.params['game_batch_size'])

        # first part of original step, gets old logprobs and ref logprobs
        bs = self.params['batch_size']
        assert self.params['game_batch_size'] == len(queries), f"Batch size ({self.params['game_batch_size']}) does not match number of examples ({len(queries)})"

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
        self.trainer_buffer.reject(self.params["batch_size"], threshType="top n")
        reject_n = len(self.trainer_buffer)
        print("rejection ", start_n, " ", reject_n)
        self.rejectRatio = torch.tensor(reject_n / start_n)

    def compute_rewards(self, scores, logprobs, ref_logprobs):
        """Compute per token rewards from scores and KL-penalty."""
        rewards, non_score_rewards = [], []
        for score, logprob, ref_logprob in zip(scores, logprobs, ref_logprobs):
            logprob = logprob.to(self.device)
            ref_logprob = ref_logprob.to(self.device)
            kl = logprob - ref_logprob
            # for stats and adaptive update
            self.kl_ctl_rew.kl_list.append(kl.detach().to("cpu"))
            non_score_reward = -self.kl_ctl_rew.value * kl
            non_score_rewards.append(non_score_reward)
            reward = non_score_reward.clone()
            reward[-1] += score
            rewards.append(reward)
        return rewards, non_score_rewards

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
            train_stats.append(flatten_dict(stats))

        self.all_stats.extend(train_stats)

        kl_loss_batch /= fbs
        if self.params["train_generation"]:
            ce_loss /= fbs

        return ce_loss_batch + self.kl_ctl.value * kl_loss_batch


def train(model_name, single_game=False, NUM_AGENTS=1):
    from time import time

    UPDATE_FREQUENCY = 16
    FORWARD_BATCH = 2
    LOG_FREQUENCY = 1
    SAVE_FREQUENCY = 16
    NUM_AGENTS = 4

    buffer = ReplayBuffer(UPDATE_FREQUENCY)

    if single_game:
        agent = NLPAgent(buffer, humanTurns=0)
        print("Training")
        agent.train()  # Tell the agent it should update its parameters.
        player = Player(agent, "./games/tw-rewardsDense_goalDetailed.z8", verbose=False)  # Dense rewards game.

    else:
        agent = VectorNLPAgent(buffer, num_agents=NUM_AGENTS)
        print("Training on 100 games")
        agent.train()  # Tell the agent it should update its parameters.
        player = VectorPlayer(agent, "./training_games/", verbose=False, num_agents=NUM_AGENTS,
                              exTurns=0.25)  # Each game will be seen 5 times.

    params = {'batch_size': UPDATE_FREQUENCY // 2,
              'game_batch_size': UPDATE_FREQUENCY,
             'save_freq': SAVE_FREQUENCY,
             'log_freq': LOG_FREQUENCY}
    reject_tuner = RejectionTuner(model_name, player, buffer, agent, **params)

    from pytorch_lightning.strategies.deepspeed import DeepSpeedStrategy
    # trainer = pl.Trainer(
    #     logger=False,
    #     accelerator='gpu', devices=1,
    #     max_epochs=500,
    #     precision=16,
    #     strategy=DeepSpeedStrategy(
    #         stage=3,
    #         offload_optimizer=True,
    #         offload_parameters=False,
    #         ),
    #     )
    trainer = pl.Trainer(
        enable_checkpointing=False,
        logger=False,
        accelerator='gpu', devices=1,
        max_epochs=500,
        precision=16,
        strategy=DeepSpeedStrategy(
            config="ds_config_zero3_light.json"
        ),
    )
    trainer.fit(reject_tuner)


if __name__ == "__main__":
    import argparse

    getEnvs()
    print("generated envs")

    # model_name = 'gpt2-xl'
    # model_name = 'EleutherAI/gpt-j-6B'
    model_name = 'EleutherAI/gpt-neo-1.3B'
    single_game = False

    Path("stats").mkdir(parents=True, exist_ok=True)
    Path("checkpoints").mkdir(parents=True, exist_ok=True)

    train(model_name, single_game=single_game)