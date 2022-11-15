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

from datastructures import RLDataset
from datastructures import ReplayBuffer
from datastructures import LineDataset
from datastructures import LineBuffer
from datastructures import RLDatasetCollator

import pytorch_lightning as pl
from pytorch_lightning import seed_everything
import argparse
from collections import OrderedDict, deque
from typing import Tuple, List
import torch.optim as optim
from torch.optim import Optimizer
from deepspeed.ops.adam import DeepSpeedCPUAdam
from deepspeed.ops.adam import FusedAdam
from torch.utils.data import DataLoader
from torchinfo import summary

from valueHead import ValueHead
from games import VectorPlayer, getEnvs
from agents import VectorNLPAgent

from transformers import DataCollatorForLanguageModeling

import trlTrainer
from trlTrainer import TRLTrainer
from games import GameReward, WinReward, LivingReward, InvalidReward, LetterReward

from core import (logprobs_from_logits,
                  whiten, whitenBatch,
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


class PPOTrainer(TRLTrainer):
    """
    The PPO_trainer uses Proximal Policy Optimization to optimise language models.
    """

    default_params = {
        "alg_name": "ppo",
        # "lr": 1.41e-5,
        "lr": 0.7e-5,
        "reference": True,
        # KL Calcuated per forward batch importance corrected exact gradients
        "adap_kl_ctrl": False,
        "init_kl_coef": 0.0,
        "target": 6,
        "horizon": 10000,
        # KL added to rewards at start of PPO Epochs
        "adap_kl_ctrl_rew": False,
        "init_kl_coef_rew": 0.1,
        "target_rew": 6,
        "horizon_rew": 10000,
        # end KL
        "gamma": 1,
        "lam": 0.95,
        "cliprange": .2,
        "cliprange_value": .2,
        "vf_coef": .1,
        "batch_size": 256,
        "forward_batch_size": 16,
        "epochs_per_game": 4,
        "game_gamma": 0.8,
        "few_shot": 0,
        "ent_coef" : 0.0, # Entropy coefficient
    }

    def __init__(self, model_name=None, **params):
        super().__init__(model_name=model_name, **params)

        self.trainer_buffer = None

        self.agent_buffer = ReplayBuffer(self.params["batch_size"])

        gameRew = GameReward(value=1, num_agents=self.params["num_agents"])
        gameRew = WinReward(value=100, num_agents=self.params["num_agents"], parentReward=gameRew)
        gameRew = InvalidReward(value=-.5, num_agents=self.params["num_agents"], parentReward=gameRew)

        letterRew = LetterReward(value=1, num_agents=self.params["num_agents"], letters=('e', 'E'))

        invalidRew = InvalidReward(value=-1, num_agents=self.params["num_agents"], parentReward=None)

        self.playerKWArgs = getKW(exTurns=0.33, rewardFunc=gameRew)
        self.agentKWArgs = getKW(useUnfinished=True, GAMMA=self.params["game_gamma"],
                                 MEMORY_LEN=self.params["few_shot"])

        # print("kl value ", self.kl_ctl.value)
        # print("kl rew value ", self.kl_ctl_rew.value)

        # print(self.playerKWArgs)
        # print(self.agentKWArgs)

    # def setup(self, stage=None):
    #     super().setup(stage=stage)
    def configure_sharded_model(self):
        if not hasattr(self, "valueHead"):
            self.valueHead = ValueHead(self.model_name)

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
        scores, _, _, values_next, ret_cross, adv_cross, logprobs, ref_logprobs, values, rewards, non_score_reward = zip(
            *data)

        timing = dict()
        timing[f'time/{self.alg_name}/optimize_step'] = time.time() - self.epoch_time
        timing[f'time/{self.alg_name}/game_time'] = self.game_time

        timing['time/filesystem/save_model'] = self.saveModelTime
        timing['time/filesystem/save_stats'] = self.saveStatTime

        t = time.time()
        train_stats = stack_dicts(self.all_stats)
        self.all_stats = []

        # reshape advantages/ratios such that they are not averaged.
        train_stats['policy/advantages'] = torch.flatten(train_stats['policy/advantages']).unsqueeze(0)
        train_stats['policy/advantages'] = torch.nan_to_num(train_stats['policy/advantages'], WANDB_PADDING)
        train_stats['policy/ratio'] = torch.flatten(train_stats['policy/ratio']).unsqueeze(0)

        # print("kl list ", len(self.kl_ctl_rew.kl_list))
        stats = self.record_step_stats(scores=scores, logprobs=logprobs, ref_logprobs=ref_logprobs,
                                       non_score_reward=non_score_reward, train_stats=train_stats)
        stats[f'{self.alg_name}/val/var_explained'] = 1 - stats[f'{self.alg_name}/val/error'] / stats[
            f'{self.alg_name}/returns/var']

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
        torch.save(stats, filename)
        self.saveStatTime = time.time() - t

    def runGame(self):
        self.trainer_buffer.clear()
        self.agent_buffer.clear()
        # self is passing the model to do forward passes with
        self.player.runGame(self, self.params['batch_size'])
        self.agent.fillBuffer()

        # print("rank ", self.trainer.global_rank, " arrived at rungame barrier")
        torch.distributed.barrier()

        self.kl_ctl_rew.kl_list = []
        scores, queries, responses, values_next, ret_cross, adv_cross, values, logprobs = self.agent_buffer.sample(
            self.params['batch_size'])

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
        # print("rank ", self.trainer.global_rank, " finished ref logprobs")

        timing[f'time/{self.alg_name}/forward_pass'] = time.time() - t

        # print("run game")
        # print(values)

        t = time.time()
        rewards, non_score_reward = self.compute_rewards(scores, logprobs, ref_logprobs)
        timing[f'time/{self.alg_name}/compute_rewards'] = time.time() - t
        for lineItem in zip(scores, queries, responses, values_next, ret_cross, adv_cross, logprobs, ref_logprobs,
                            values, rewards, non_score_reward):
            self.trainer_buffer.append(lineItem)
        # print("rank ", self.trainer.global_rank, " finished train buffer")

    def configure_optimizers(self) -> List[Optimizer]:
        """ Initialize Adam optimizer"""
        # self.optimizer = Adam(model.parameters(), lr=self.params['lr'])
        # optimizer = Adam(list(self.model.parameters()) + list(self.valueHead.parameters()), lr=self.params['lr'])
        paramList = list(self.model.parameters()) + list(self.valueHead.parameters())
        if True:  # use normal deepspeed opt
            if self.deepspeed_offload:
                optimizer = DeepSpeedCPUAdam(paramList, lr=self.params['lr'])
            else:
                optimizer = FusedAdam(paramList, lr=self.params['lr'])
        else:
            # doesn't appear to work with deepspeed distributed
            from nero import Nero
            optimizer = Nero(paramList, lr=self.params['lr'])

        return [optimizer]

    def __dataloader(self) -> DataLoader:
        self.data_collator = DataCollatorForLanguageModeling(self.tokenizer, mlm=False)
        self.trainer_buffer = LineBuffer(self.params['batch_size'])
        dataset = RLDataset(self.trainer_buffer, self.params['batch_size'],
                            rank=self.trainer.global_rank, world_size=self.trainer.world_size)
        dataloader = DataLoader(dataset=dataset,
                                batch_size=self.params['forward_batch_size'],
                                collate_fn=RLDatasetCollator(text_collator=self.data_collator)
                                )
        return dataloader

    def train_dataloader(self) -> DataLoader:
        """Get train loader"""
        return self.__dataloader()

    def test_dataloader(self) -> DataLoader:
        return self.__dataloader()

    def forward(self, input_ids, use_cache=False, past_key_values=None, outputVals=False, outputRef=False,
                attention_mask=None, outputLogits=True):
        # print(f"forward on rank {self.trainer.global_rank} with cache {use_cache} with past key {past_key_values is not None}  vals {outputVals} reference {outputRef} mask {attention_mask is not None}  logit {outputLogits}")
        output = {}
        if outputLogits or outputVals:
            if past_key_values is None:
                lmOut = self.model(input_ids, output_hidden_states=outputVals, use_cache=use_cache,
                                   attention_mask=attention_mask)
            else:
                lmOut = self.model(input_ids, output_hidden_states=outputVals, use_cache=use_cache,
                                   past_key_values=past_key_values, attention_mask=attention_mask)
            # print(f"forward on rank {self.trainer.global_rank} finished hugging face model")
            # print(dir(lmOut))
            if outputLogits:
                logits = lmOut.logits
                output["logits"] = logits

            if use_cache:
                cache = lmOut.past_key_values
                output["cache"] = cache

            if outputVals:
                hidden_state = lmOut.hidden_states[-1]
                v = self.valueHead(hidden_state)
                output["values"] = v

        if outputRef:
            with torch.no_grad():
                ref_logits = self.ref_model(input_ids).logits
                output["ref_logits"] = ref_logits

        # print(f"forward on rank {self.trainer.global_rank} finished all")
        return output

    def batched_forward_pass(self, queries, responses, outputLogits=True, outputVals=True, outputRef=True):
        """Calculate model outputs in multiple batches."""
        bs = self.params['batch_size']
        fbs = self.params['forward_batch_size']
        all_logprobs = []
        all_ref_logprobs = []
        all_values = []

        output = {}
        
        # print("\n" + "batched forward on rank " + str(self.trainer.global_rank))
        if self.trainer.global_rank == 0:
            print("batched forward pass ", flush=True)

        for i in range(int(bs / fbs)):
            # print("rank ", self.trainer.global_rank, " ref batch ", i)
            query_batch = queries[i * fbs:(i + 1) * fbs]
            response_batch = responses[i * fbs:(i + 1) * fbs]
            model_input = self.data_collator([torch.cat([q, r]) for q, r in zip(query_batch, response_batch)])
            input_ids = model_input["input_ids"]
            attention_mask = pad_mask(input_ids, self.tokenizer.pad_token_id)
            # print("forward pass mask ", attention_mask)
            
            if self.trainer.global_rank == 0:
                print("\r", i, "/", int(bs / fbs), sep="", end="", flush=True)

            with torch.no_grad():
                # # logits, _, v = self.model(input_ids)
                # lmOut = self.model(input_ids, output_hidden_states=True)
                # # print(dir(lmOut))
                # logits, hidden_state = lmOut.logits, lmOut.hidden_states[-1]
                # v = self.valueHead(hidden_state)
                # # ref_logits, _, _ = self.ref_model(input_ids)
                # ref_logits = self.ref_model(input_ids).logits
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
                    v = lmout.v

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
            # print("rank ", self.trainer.global_rank, " final ref batch ", rem)
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
                    v = lmout.v

            for j in range(rem):
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

        output["logprobs"] = all_logprobs
        output["values"] = all_values
        output["ref_logprobs"] = all_ref_logprobs
        return output

    def test_step(self, batch, nb_batch):
        with torch.no_grad():
            scores, queries, responses, model_input, lengths, values_next, ret_cross, adv_cross, logprobs, ref_logprobs, values, rewards, non_score_reward = batch
            fbs = scores.shape[0]

            # reccomended by torch when zero3 config
            torch.cuda.empty_cache()
            t = time.time()
            timing = dict()

            train_stats, loss = self.train_minibatch(logprobs, values,
                                                     rewards, queries,
                                                     responses,
                                                     model_input, lengths,
                                                     values_next=values_next, ref_logprobs=ref_logprobs)

            if self.trainer.is_global_zero:
                gathered_stats = [None for i in range(self.trainer.world_size)]
            else:
                gathered_stats = None
            torch.distributed.gather_object(train_stats, object_gather_list=gathered_stats, dst=0)
            if self.trainer.is_global_zero:
                for stats in gathered_stats:
                    self.all_stats.extend(stats)

    def training_step(self, batch, nb_batch):
        scores, queries, responses, model_input, lengths, values_next, ret_cross, adv_cross, logprobs, ref_logprobs, values, rewards, non_score_reward = batch
        fbs = scores.shape[0]

        # reccomended by torch when zero3 config
        torch.cuda.empty_cache()
        t = time.time()
        timing = dict()

        train_stats, loss = self.train_minibatch(logprobs, values,
                                                 rewards, queries,
                                                 responses,
                                                 model_input, lengths,
                                                 values_next=values_next, ref_logprobs=ref_logprobs)

        if self.trainer.is_global_zero:
            gathered_stats = [None for i in range(self.trainer.world_size)]
        else:
            gathered_stats = None
        torch.distributed.gather_object(train_stats, object_gather_list=gathered_stats, dst=0)
        if self.trainer.is_global_zero:
            for stats in gathered_stats:
                self.all_stats.extend(stats)

        return loss

    def train_minibatch(self, old_logprobs, values, rewards, query, response, model_input, lengths, values_next=(0.0,),
                        ref_logprobs=None):
        """Train one PPO minibatch"""
        loss_total = None
        input_ids = model_input["input_ids"]
        input_mask = pad_mask(input_ids, self.tokenizer.pad_token_id)
        query_ids = query["input_ids"]
        query_mask = pad_mask(query_ids, self.tokenizer.pad_token_id)
        response_ids = response["input_ids"]
        response_mask = pad_mask(response_ids, self.tokenizer.pad_token_id)

        lmout = self.forward(input_ids, outputVals=True, outputRef=False, attention_mask=input_mask)
        logits, vpred = lmout["logits"], lmout["values"]
        train_stats = []
        returnsList = []
        advantagesList = []
        for i in range(logits.shape[0]):
            # keep batch dim
            returns, advantages = self.computeAdvantage(logits[i:i + 1], vpred[i:i + 1], old_logprobs[i:i + 1],
                                                        values[i:i + 1], rewards[i:i + 1], query_ids[i:i + 1],
                                                        response_ids[i:i + 1], input_ids[i:i + 1], lengths[i],
                                                        values_next=values_next[i:i + 1],
                                                        ref_logprobs=ref_logprobs[i:i + 1])
            returnsList.append(returns)
            advantagesList.append(advantages)

        # causes a deadlock at the beginning of loss() on multi gpu. Unsure how
        # gets a combined mean and var of every element in batch across all ranks
        # print("whiten batch rank ", self.trainer.global_rank, flush=True)
        # whitenBatch(advantagesList, rank=self.trainer.global_rank, world_size=self.trainer.world_size)
        # print("whiten batch finished rank ", self.trainer.global_rank, flush=True)

        for i in range(logits.shape[0]):
            # keep batch dim
            # print("loss i ", i, "rank ", self.trainer.global_rank, flush=True)
            loss, stat = self.loss(logits[i:i + 1], vpred[i:i + 1], old_logprobs[i:i + 1],
                                                      values[i:i + 1], rewards[i:i + 1], query_ids[i:i + 1],
                                                      response_ids[i:i + 1], input_ids[i:i + 1], lengths[i],
                                                      returnsList[i], advantagesList[i],
                                                      values_next=values_next[i:i + 1],
                                                      ref_logprobs=ref_logprobs[i:i + 1])
            
            # loss = loss_p + loss_v + kl_loss
            # loss = loss_p + loss_v
            # if self.kl_ctl.value != 0.0:
            #     loss = loss + kl_loss
                
            train_stats.append(stat)

            if loss_total is None:
                loss_total = loss
            else:
                loss_total += loss
        
        # print("train step finished rank ", self.trainer.global_rank, flush=True)
        return train_stats, loss

    def computeAdvantage(self, logits, vpred, old_logprobs, values, rewards, query, response, input_ids, lengths,
                         values_next=0.0,
                         ref_logprobs=None):
        lastgaelam = 0
        advantages_reversed = []
        # gen_len = response.shape[1]
        querry_len = lengths[0]
        gen_len = lengths[1]
        total_len = lengths[2]

        # remove left pad
        values = values[:, -gen_len:]
        rewards = rewards[:, -gen_len:]
        old_logprobs = old_logprobs[:, -gen_len:]
        ref_logprobs = ref_logprobs[:, -gen_len:]

        for t in reversed(range(gen_len)):
            nextvalues = values[:, t + 1] if t < gen_len - 1 else self.params["game_gamma"] * values_next
            delta = rewards[:, t] + self.params['gamma'] * nextvalues - values[:, t]
            lastgaelam = delta + self.params['gamma'] * self.params['lam'] * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1]).transpose(0, 1)

        returns = advantages + values
        # whiten as a batch instead
        advantages = whiten(advantages)
        advantages = advantages.detach()

        return returns, advantages

    def loss(self, logits, vpred, old_logprobs, values, rewards, query, response, input_ids, lengths, returns,
             advantages, values_next=0.0,
             ref_logprobs=None):
        """Calculate policy and value losses."""
        # lastgaelam = 0
        # advantages_reversed = []
        # gen_len = response.shape[1]
        querry_len = lengths[0]
        gen_len = lengths[1]
        total_len = lengths[2]

        # removes left pad
        values = values[:, -gen_len:]
        rewards = rewards[:, -gen_len:]
        old_logprobs = old_logprobs[:, -gen_len:]
        ref_logprobs = ref_logprobs[:, -gen_len:]

        #         for t in reversed(range(gen_len)):
        #             nextvalues = values[:, t + 1] if t < gen_len - 1 else self.params["game_gamma"] * values_next
        #             delta = rewards[:, t] + self.params['gamma'] * nextvalues - values[:, t]
        #             lastgaelam = delta + self.params['gamma'] * self.params['lam'] * lastgaelam
        #             advantages_reversed.append(lastgaelam)
        #         advantages = torch.stack(advantages_reversed[::-1]).transpose(0, 1)

        #         returns = advantages + values
        #         advantages = whiten(advantages)
        advantages = advantages.detach()

        # computed batched before this method called
        # logits, vpred = self.forward(model_input, outputVals=True)

        logprob = logprobs_from_logits(logits[:, :-1, :], input_ids[:, 1:])

        # only the generation part of the values/logprobs is needed
        # both logits and values are shifted 1 left from the input
        # remove left pad
        # logits were already shifted
        logprob, vpred = logprob[:, -gen_len:], vpred[:, -(gen_len + 1):-1]
        
        # print("vf loss rank ", self.trainer.global_rank, flush=True)
        vpredclipped = clip_by_value(vpred,
                                     values - self.params["cliprange_value"],
                                     values + self.params["cliprange_value"])

        vf_losses1 = (vpred - returns) ** 2
        vf_losses2 = (vpredclipped - returns) ** 2
        vf_loss = .5 * torch.mean(torch.max(vf_losses1, vf_losses2))
        vf_clipfrac = torch.mean(torch.gt(vf_losses2, vf_losses1).double())

        # print("pg loss rank ", self.trainer.global_rank, flush=True)
        ratio = torch.exp(logprob - old_logprobs)

        pg_losses = -advantages * ratio
        pg_losses2 = -advantages * torch.clamp(ratio,
                                               1.0 - self.params['cliprange'],
                                               1.0 + self.params['cliprange'])

        pg_loss = torch.mean(torch.max(pg_losses, pg_losses2))
        pg_clipfrac = torch.mean(torch.gt(pg_losses2, pg_losses).double())
        
        # print("kl loss rank ", self.trainer.global_rank, flush=True)
        # directly backprop through KL instead of a KL reward penalty
        kl = logprob - ref_logprobs
        # importence sampling correction KL (P || R) sampled from Q
        # backprop through ratio and kl
        kl = kl * ratio
        # for stats and adaptive update
        self.kl_ctl.kl_list.append(kl.detach().to("cpu"))
        # mean across tokens
        kl_loss = torch.mean(kl)
        
        
        # print("total loss rank ", self.trainer.global_rank, flush=True)
        loss = pg_loss + self.params['vf_coef'] * vf_loss
        if self.kl_ctl.value != 0.0:
            # print("add kl loss rank ", self.trainer.global_rank, flush=True)
            loss = loss + self.kl_ctl.value * kl_loss

        # print("loss stats rank ", self.trainer.global_rank)
        entropy = torch.mean(entropy_from_logits(logits))
        approxkl = .5 * torch.mean((logprob - old_logprobs) ** 2)
        policykl = torch.mean(logprob - old_logprobs)
        return_mean, return_var = torch.mean(returns), torch.var(returns)
        value_mean, value_var = torch.mean(values), torch.var(values)

        ent_loss = -torch.mean(entropy)
        if self.params["ent_coef"] != 0.0:
            loss += self.params["ent_coef"] * ent_loss

        stats = dict(
            loss=dict(policy=pg_loss, value=vf_loss, kl=kl_loss, ent=ent_loss, total=loss),
            policy=dict(entropy=entropy, approxkl=approxkl, policykl=policykl, clipfrac=pg_clipfrac,
                        advantages=advantages, advantages_mean=torch.mean(advantages), ratio=ratio),
            returns=dict(mean=return_mean, var=return_var),
            val=dict(vpred=torch.mean(vpred), error=torch.mean((vpred - returns) ** 2),
                     clipfrac=vf_clipfrac, mean=value_mean, var=value_var),
        )
        # print("loss return rank ", self.trainer.global_rank, flush=True)
        return loss, stats_to_cpu(flatten_dict(stats))


def train(model_name=None, single_game=True):
    from time import time

    UPDATE_FREQUENCY = 64
    FORWARD_BATCH = 8
    LOG_FREQUENCY = 1
    NUM_AGENTS = 8
    PPO_EPOCHS = 1

    trainer = trlTrainer.getTrainer()
    # print("rank out of world :", trainer.global_rank, " " , trainer.world_size)
    UPDATE_FREQUENCY = max(UPDATE_FREQUENCY // trainer.world_size, 1)
    FORWARD_BATCH = max(FORWARD_BATCH // trainer.world_size, 1)
    NUM_AGENTS = max(NUM_AGENTS // trainer.world_size, 1)

    if trainer.is_global_zero:
        print("Params per thread: update freq ", UPDATE_FREQUENCY, " forward batch ", FORWARD_BATCH, " num agents ",
              NUM_AGENTS)

    ppo_config = {'batch_size': UPDATE_FREQUENCY, 'forward_batch_size': FORWARD_BATCH, "log_freq": LOG_FREQUENCY,
                  "num_agents": NUM_AGENTS, "single_game": single_game, "epochs_per_game":PPO_EPOCHS}
    ppo_trainer = PPOTrainer(model_name=model_name, **ppo_config)

    trainer.fit(ppo_trainer)


if __name__ == "__main__":
    import argparse

    seed_everything(2061630618)
    # seed_everything(42)

    # model_name = 'gpt2'
    # model_name = 'gpt2-medium'
    # model_name = 'EleutherAI/gpt-j-6B'
    # model_name = 'EleutherAI/gpt-neo-1.3B'
    model_name = "EleutherAI/gpt-neox-20b"
    single_game = False

    Path("stats").mkdir(parents=True, exist_ok=True)
    Path("checkpoints").mkdir(parents=True, exist_ok=True)

    train(model_name=model_name, single_game=single_game)