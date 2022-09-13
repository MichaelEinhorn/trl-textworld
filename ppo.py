# sepparates value head from the model so that it can be a drop in transformer where output_hidden_states is an option

import numpy as np
import torch.nn.functional as F
from torch.optim import Adam
import torch
import collections
import time
import random

from datastructures import RLDataset
from datastructures import ReplayBuffer
from datastructures import LineDataset
from datastructures import LineBuffer
from datastructures import RLDatasetCollator

import pytorch_lightning as pl
import argparse
from collections import OrderedDict, deque
from typing import Tuple, List
import torch.optim as optim
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchinfo import summary

from valueHead import ValueHead
from games import Player, getEnvs

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
                  WANDB_PADDING)


# using deepspeed pytorch-lightning


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


class PPOTrainer(pl.LightningModule):
    """
    The PPO_trainer uses Proximal Policy Optimization to optimise language models.
    """

    default_params = {
        "lr": 1.41e-5,
        # KL Calcuated per forward batch importance corrected exact gradients
        "adap_kl_ctrl": False,
        "init_kl_coef": 0,
        "target": 6,
        "horizon": 10000,
        # KL added to rewards at start of PPO Epochs
        "adap_kl_ctrl_rew": False,
        "init_kl_coef_rew": 0,
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
        "ppo_epochs": 4,
    }

    def __init__(self, model_name, player, buffer, agent, **ppo_params):
        super().__init__()
        """
        Initialize PPOTrainer.

        Args:
            model (torch.model): Hugging Face transformer GPT2 model with value head
            ref_model (torch.model): Hugging Face transformer GPT2 refrence model used for KL penalty
            tokenizer (tokenizer): Hugging Face tokenizer
            ppo_params (dict or None): PPO parameters for training. Can include following keys:
                'lr' (float): Adam learning rate, default: 1.41e-5
                'batch_size' (int): Number of samples per optimisation step, default: 256
                'forward_batch_size' (int): Number of samples forward passed through model at a time, default: 16
                'ppo_epochs' (int): Number of optimisation epochs per batch of samples, default: 4
                'gamma' (float)): Gamma parameter for advantage calculation, default: 1.
                'lam' (float): Lambda parameter for advantage calcualation, default: 0.95
                'cliprange_value' (float): Range for clipping values in loss calculation, default: 0.2
                'cliprange' (float): Range for clipping in PPO policy gradient loss, default: 0.2
                'vf_coef' (float): Scaling factor for value loss, default: 0.1
                'adap_kl_ctrl' (bool): Use adaptive KL control, otherwise linear, default: True
                'init_kl_coef' (float): Initial KL penalty coefficient (used for adaptive and linear control), default: 0.2
                'target' (float): Target KL value for adaptive KL control, default: 6.0
                'horizon' (float): Horizon for adaptive KL control, default: 10000

        """

        self.ppo_params = self.default_params
        self.ppo_params.update(ppo_params)

        self.agent_buffer = buffer
        self.ppo_buffer = LineBuffer(self.ppo_params['batch_size'])
        self.player = player

        # gpt2 and gpt2-xl
        if 'gpt2' in model_name:
            from transformers import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel
            self.model = GPT2LMHeadModel.from_pretrained(model_name)
            self.ref_model = GPT2LMHeadModel.from_pretrained(model_name)
            self.tokenizer = GPT2Tokenizer.from_pretrained(model_name, padding_side='left')
            self.tokenizer.pad_token = self.tokenizer.unk_token
        elif 'gpt-j' in model_name:
            from transformers import GPT2Tokenizer, GPTJForCausalLM
            self.model = GPTJForCausalLM.from_pretrained(model_name)
            self.ref_model = GPTJForCausalLM.from_pretrained(model_name)
            self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2', padding_side='left')
            self.tokenizer.pad_token = self.tokenizer.unk_token
        elif 'gpt-neo' in model_name:
            from transformers import GPTNeoForCausalLM, GPT2Tokenizer
            self.model = GPTNeoForCausalLM.from_pretrained(model_name)
            self.ref_model = GPTNeoForCausalLM.from_pretrained(model_name)
            self.tokenizer = GPT2Tokenizer.from_pretrained(model_name, padding_side='left')
            self.tokenizer.pad_token = self.tokenizer.unk_token

        print(self.model.config.torch_dtype)
        summary(self.model)

        self.config = self.model.config
        # copied from gpt2.py, not sure what this does
        self.config.num_labels = 1
        self.valueHead = ValueHead(self.config)

        # make value head same precision as model
        # if self.config.torch_dtype == torch.float16:
        #     self.valueHead = self.valueHead.half()

        self.agent = agent

        # print(self.valueHead)

        self.data_collator = DataCollatorForLanguageModeling(self.tokenizer, mlm=False)

        if self.ppo_params['adap_kl_ctrl']:
            self.kl_ctl = AdaptiveKLController(self.ppo_params['init_kl_coef'],
                                               self.ppo_params['target'],
                                               self.ppo_params['horizon'])
        else:
            self.kl_ctl = FixedKLController(self.ppo_params['init_kl_coef'])

        if self.ppo_params['adap_kl_ctrl_rew']:
            self.kl_ctl_rew = AdaptiveKLController(self.ppo_params['init_kl_coef_rew'],
                                               self.ppo_params['target_rew'],
                                               self.ppo_params['horizon_rew'])
        else:
            self.kl_ctl_rew = FixedKLController(self.ppo_params['init_kl_coef_rew'])

    def getDevice(self):
        return self.device

    # def on_train_start(self):
    #     self.runGame()
    def on_train_epoch_start(self):
        # train on the same data ppo epochs times before generating a new set
        if self.current_epoch % self.ppo_params['ppo_epochs'] == 0:
            self.runGame()

    def runGame(self):
        # self is passing the model to do forward passes with
        self.player.runGame(self, self.ppo_params['batch_size'])
        self.agent.fillBuffer()
        scores, queries, responses, values_next, ret_cross, adv_cross = self.agent_buffer.sample(
            self.ppo_params['batch_size'])

        # first part of original step, gets old logprobs and ref logprobs
        bs = self.ppo_params['batch_size']
        assert bs == len(queries), f"Batch size ({bs}) does not match number of examples ({len(queries)})"

        timing = dict()
        t0 = time.time()

        response_lengths = [len(r) for r in responses]

        t = time.time()
        logprobs, ref_logprobs, values = self.batched_forward_pass(queries, responses)
        timing['time/ppo/forward_pass'] = time.time() - t

        # print("run game")
        # print(values)

        t = time.time()
        rewards, non_score_reward = self.compute_rewards(scores, logprobs, ref_logprobs)
        timing['time/ppo/compute_rewards'] = time.time() - t
        for lineItem in zip(scores, queries, responses, values_next, ret_cross, adv_cross, logprobs, ref_logprobs,
                            values, rewards, non_score_reward):
            self.ppo_buffer.append(lineItem)

    def configure_optimizers(self) -> List[Optimizer]:
        """ Initialize Adam optimizer"""
        # self.optimizer = Adam(model.parameters(), lr=self.ppo_params['lr'])
        optimizer = Adam(list(self.model.parameters()) + list(self.valueHead.parameters()), lr=self.ppo_params['lr'])
        return [optimizer]

    def __dataloader(self) -> DataLoader:
        dataset = RLDataset(self.ppo_buffer, self.ppo_params['batch_size'])
        dataloader = DataLoader(dataset=dataset,
                                batch_size=self.ppo_params['forward_batch_size'],
                                collate_fn=RLDatasetCollator(text_collator=self.data_collator)
                                )
        return dataloader

    def train_dataloader(self) -> DataLoader:
        """Get train loader"""
        return self.__dataloader()

    def forward(self, input_ids, use_cache=False, past_key_values=None, outputVals=False, outputRef=False, attention_mask=None):
        if past_key_values is None:
            lmOut = self.model(input_ids, output_hidden_states=outputVals, use_cache=use_cache, attention_mask=attention_mask)
        else:
            lmOut = self.model(input_ids, output_hidden_states=outputVals, use_cache=use_cache,
                               past_key_values=past_key_values, attention_mask=attention_mask)
        # print(dir(lmOut))
        logits = lmOut.logits

        output = [logits]

        if use_cache:
            cache = lmOut.past_key_values
            output.append(cache)

        if outputRef:
            with torch.no_grad():
                ref_logits = self.ref_model(input_ids).logits
                output.append(ref_logits)

        if outputVals:
            hidden_state = lmOut.hidden_states[-1]
            v = self.valueHead(hidden_state)
            output.append(v)
        # ref_logits, _, _ = self.ref_model(input_ids)

        return tuple(output)

    def training_step(self, batch, nb_batch):
        # rew, prompt[0], action[0], values, ret_cross, adv_cross
        scores, queries, responses, model_input, lengths, values_next, ret_cross, adv_cross, logprobs, ref_logprobs, values, rewards, non_score_reward = batch

        #         print("train step")
        #         print(values)

        fbs = scores.shape[0]
        """
        Run a PPO optimisation step.

        args:
            queries (List): List of tensors containing the encoded queries, shape [query_length]
            responses (List): List of tensors containing the encoded responses, shape [response_length]
            scores (List): tensor containing the scores, shape [batch_size]

        returns:
            train_stats (dict): a summary of the training statistics
        """

        # print(queries[0].shape)

        t = time.time()
        timing = dict()
        all_stats = []
        # moved to start train epoch, loops through dataset ppo_epochs times before generating a new set
        # for _ in range(self.ppo_params['ppo_epochs']):
        train_stats, loss = self.train_minibatch(logprobs, values,
                                                 rewards, queries,
                                                 responses,
                                                 model_input, lengths,
                                                 values_next=values_next, ref_logprobs=ref_logprobs)
        all_stats.append(train_stats)
        timing['time/ppo/optimize_step'] = time.time() - t

        t = time.time()
        train_stats = stack_dicts(all_stats)

        train_stats['epoch/batch'] = (self.epoch, nb_batch)

        # reshape advantages/ratios such that they are not averaged.
        train_stats['policy/advantages'] = torch.flatten(train_stats['policy/advantages']).unsqueeze(0)
        train_stats['policy/advantages'] = torch.nan_to_num(train_stats['policy/advantages'], WANDB_PADDING)
        train_stats['policy/ratio'] = torch.flatten(train_stats['policy/ratio']).unsqueeze(0)

        stats = self.record_step_stats(scores=scores, logprobs=logprobs, ref_logprobs=ref_logprobs,
                                       non_score_reward=non_score_reward, train_stats=train_stats,
                                       kl_coef=self.kl_ctl.value)
        stats = stats_to_np(stats)
        timing['time/ppo/calc_stats'] = time.time() - t

        self.kl_ctl.update(stats['objective/kl'], fbs)
        self.kl_ctl_rew.update(stats['objective/kl_rew'], fbs)

        # timing['time/ppo/total'] = time.time() - t0
        stats.update(timing)
        print(stats)

        return loss

    #     def step(self, queries, responses, scores):
    #         """
    #         Run a PPO optimisation step.

    #         args:
    #             queries (List): List of tensors containing the encoded queries, shape [query_length]
    #             responses (List): List of tensors containing the encoded responses, shape [response_length]
    #             scores (List): tensor containing the scores, shape [batch_size]

    #         returns:
    #             train_stats (dict): a summary of the training statistics
    #         """

    #         # print(queries[0].shape)
    #         bs = self.ppo_params['batch_size']
    #         assert bs == len(queries), f"Batch size ({bs}) does not match number of examples ({len(queries)})"

    #         timing = dict()
    #         t0 = time.time()

    #         response_lengths = [len(r) for r in responses]

    #         t = time.time()
    #         logprobs, ref_logprobs, values = self.batched_forward_pass(queries, responses)
    #         timing['time/ppo/forward_pass'] = time.time() - t

    #         t = time.time()
    #         rewards, non_score_reward = self.compute_rewards(scores, logprobs, ref_logprobs)
    #         timing['time/ppo/compute_rewards'] = time.time() - t

    #         t = time.time()
    #         all_stats = []
    #         total_loss = None
    #         idxs = list(range(bs))
    #         for _ in range(self.ppo_params['ppo_epochs']):
    #             random.shuffle(idxs)
    #             for i in range(bs):
    #                 idx = idxs[i]
    #                 train_stats, loss = self.train_minibatch(logprobs[idx].unsqueeze(0), values[idx].unsqueeze(0),
    #                                                    rewards[idx].unsqueeze(0), queries[idx].unsqueeze(0),
    #                                                    responses[idx].unsqueeze(0),
    #                                                    torch.cat([queries[idx], responses[idx]]).unsqueeze(0))
    #                 all_stats.append(train_stats)
    #                 if total_loss is None:
    #                     total_loss = loss
    #                 else:
    #                     total_loss += loss
    #         timing['time/ppo/optimize_step'] = time.time() - t

    #         t = time.time()
    #         train_stats = stack_dicts(all_stats)

    #         # reshape advantages/ratios such that they are not averaged.
    #         train_stats['policy/advantages'] = torch.flatten(train_stats['policy/advantages']).unsqueeze(0)
    #         train_stats['policy/advantages'] = torch.nan_to_num(train_stats['policy/advantages'], WANDB_PADDING)
    #         train_stats['policy/ratio'] = torch.flatten(train_stats['policy/ratio']).unsqueeze(0)

    #         stats = self.record_step_stats(scores=scores, logprobs=logprobs, ref_logprobs=ref_logprobs,
    #                                        non_score_reward=non_score_reward, train_stats=train_stats,
    #                                        kl_coef=self.kl_ctl.value)
    #         stats = stats_to_np(stats)
    #         timing['time/ppo/calc_stats'] = time.time() - t

    #         self.kl_ctl.update(stats['objective/kl'], self.ppo_params['batch_size'])

    #         timing['time/ppo/total'] = time.time() - t0
    #         stats.update(timing)
    #         return total_loss, stats

    def batched_forward_pass(self, queries, responses):
        """Calculate model outputs in multiple batches."""
        bs = self.ppo_params['batch_size']
        fbs = self.ppo_params['forward_batch_size']
        all_logprobs = []
        all_ref_logprobs = []
        all_values = []

        for i in range(int(bs / fbs)):
            query_batch = queries[i * fbs:(i + 1) * fbs]
            response_batch = responses[i * fbs:(i + 1) * fbs]
            input_ids = self.data_collator([torch.cat([q, r]) for q, r in zip(query_batch, response_batch)])[
                "input_ids"]

            with torch.no_grad():
                # # logits, _, v = self.model(input_ids)
                # lmOut = self.model(input_ids, output_hidden_states=True)
                # # print(dir(lmOut))
                # logits, hidden_state = lmOut.logits, lmOut.hidden_states[-1]
                # v = self.valueHead(hidden_state)
                # # ref_logits, _, _ = self.ref_model(input_ids)
                # ref_logits = self.ref_model(input_ids).logits
                input_ids = input_ids.to(self.device)
                logits, ref_logits, v = self.forward(input_ids, outputVals=True, outputRef=True)

                logprobs = logprobs_from_logits(logits[:, :-1, :], input_ids[:, 1:])
                ref_logprobs = logprobs_from_logits(ref_logits[:, :-1, :], input_ids[:, 1:])

            for j in range(fbs):
                # both logits and values are shifted 1 left from the input
                start = len(query_batch[j]) - 1
                end = len(query_batch[j]) + len(response_batch[j]) - 1
                all_values.append(v[j, start:end])
                all_logprobs.append(logprobs[j, start:end])
                all_ref_logprobs.append(ref_logprobs[j, start:end])

        rem = bs % fbs
        if rem != 0:
            query_batch = queries[-rem:]
            response_batch = responses[-rem:]
            input_ids = self.data_collator([torch.cat([q, r]) for q, r in zip(query_batch, response_batch)])[
                "input_ids"]

            with torch.no_grad():
                input_ids = input_ids.to(self.device)
                logits, ref_logits, v = self.forward(input_ids, outputVals=True, outputRef=True)

                logprobs = logprobs_from_logits(logits[:, :-1, :], input_ids[:, 1:])
                ref_logprobs = logprobs_from_logits(ref_logits[:, :-1, :], input_ids[:, 1:])

            for j in range(rem):
                # both logits and values are shifted 1 left from the input
                start = len(query_batch[j]) - 1
                end = len(query_batch[j]) + len(response_batch[j]) - 1
                all_values.append(v[j, start:end])
                all_logprobs.append(logprobs[j, start:end])
                all_ref_logprobs.append(ref_logprobs[j, start:end])

        return all_logprobs, all_ref_logprobs, all_values

    def train_minibatch(self, logprobs, values, rewards, query, response, model_input, lengths, values_next=(0.0,),
                        ref_logprobs=None):
        """Train one PPO minibatch"""
        loss_total = None
        input_ids = model_input["input_ids"]
        query_ids = query["input_ids"]
        response_ids = response["input_ids"]
        logits, vpred = self.forward(input_ids, outputVals=True)
        for i in range(logits.shape[0]):
            # keep batch dim
            loss_p, loss_v, kl_loss, train_stats = self.loss(logits[i:i + 1], vpred[i:i + 1], logprobs[i:i + 1],
                                                             values[i:i + 1], rewards[i:i + 1], query_ids[i:i + 1],
                                                             response_ids[i:i + 1], input_ids[i:i + 1], lengths[i],
                                                             values_next=values_next[i:i + 1],
                                                             ref_logprobs=ref_logprobs[i:i + 1])
            loss = loss_p + loss_v + kl_loss

            if loss_total is None:
                loss_total = loss
            else:
                loss_total += loss
        # self.optimizer.zero_grad()
        # loss.backward()
        # self.optimizer.step()
        return train_stats, loss

    def compute_rewards(self, scores, logprobs, ref_logprobs):
        """Compute per token rewards from scores and KL-penalty."""
        rewards, non_score_rewards = [], []
        for score, logprob, ref_logprob in zip(scores, logprobs, ref_logprobs):
            kl = logprob - ref_logprob
            # for stats and adaptive update
            self.kl_ctl_rew.kl_list.append(kl)
            non_score_reward = -self.kl_ctl_rew.value * kl
            non_score_rewards.append(non_score_reward)
            reward = non_score_reward.clone()
            reward[-1] += score
            rewards.append(reward)
        return rewards, non_score_rewards

    def loss(self, logits, vpred, old_logprobs, values, rewards, query, response, input_ids, lengths, values_next=0.0,
             ref_logprobs=None):
        """Calculate policy and value losses."""
        lastgaelam = 0
        advantages_reversed = []
        # gen_len = response.shape[1]
        querry_len = lengths[0]
        gen_len = lengths[1]
        total_len = lengths[2]

        values = values[:, :gen_len]
        rewards = rewards[:, :gen_len]
        old_logprobs = old_logprobs[:, :gen_len]
        ref_logprobs = ref_logprobs[:, :gen_len]

        for t in reversed(range(gen_len)):
            nextvalues = values[:, t + 1] if t < gen_len - 1 else values_next
            delta = rewards[:, t] + self.ppo_params['gamma'] * (nextvalues - values[:, t])
            lastgaelam = delta + self.ppo_params['gamma'] * self.ppo_params['lam'] * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1]).transpose(0, 1)

        returns = advantages + values
        advantages = whiten(advantages)
        advantages = advantages.detach()

        # computed batched before this method called
        # logits, vpred = self.forward(model_input, outputVals=True)

        logprob = logprobs_from_logits(logits[:, :-1, :], input_ids[:, 1:])

        # only the generation part of the values/logprobs is needed
        # both logits and values are shifted 1 left from the input
        start = querry_len - 1
        end = querry_len + gen_len - 1
        logprob, vpred = logprob[:, start:end], vpred[:, start:end]
        # logprob, vpred = logprob[:, total_len-gen_len:total_len], vpred[:, total_len-gen_len - 1:total_len-1]

        vpredclipped = clip_by_value(vpred,
                                     values - self.ppo_params["cliprange_value"],
                                     values + self.ppo_params["cliprange_value"])

        vf_losses1 = (vpred - returns) ** 2
        vf_losses2 = (vpredclipped - returns) ** 2
        vf_loss = .5 * torch.mean(torch.max(vf_losses1, vf_losses2))
        vf_clipfrac = torch.mean(torch.gt(vf_losses2, vf_losses1).double())

        ratio = torch.exp(logprob - old_logprobs)

        pg_losses = -advantages * ratio
        pg_losses2 = -advantages * torch.clamp(ratio,
                                               1.0 - self.ppo_params['cliprange'],
                                               1.0 + self.ppo_params['cliprange'])

        pg_loss = torch.mean(torch.max(pg_losses, pg_losses2))
        pg_clipfrac = torch.mean(torch.gt(pg_losses2, pg_losses).double())

        # directly backprop through KL instead of a KL reward penalty
        kl = logprob - ref_logprobs
        # importence sampling correction KL (P || R) sampled from Q
        # backprop through ratio and kl
        kl = kl * ratio
        # for stats and adaptive update
        self.kl_ctl.kl_list.append(kl)
        # mean across tokens
        kl_loss = torch.mean(kl)

        loss = pg_loss + self.ppo_params['vf_coef'] * vf_loss + self.kl_ctl.value * kl_loss

        entropy = torch.mean(entropy_from_logits(logits))
        approxkl = .5 * torch.mean((logprob - old_logprobs) ** 2)
        policykl = torch.mean(logprob - old_logprobs)
        return_mean, return_var = torch.mean(returns), torch.var(returns)
        value_mean, value_var = torch.mean(values), torch.var(values)

        stats = dict(
            loss=dict(policy=pg_loss, value=vf_loss, kl=kl_loss, total=loss),
            policy=dict(entropy=entropy, approxkl=approxkl, policykl=policykl, clipfrac=pg_clipfrac,
                        advantages=advantages, advantages_mean=torch.mean(advantages), ratio=ratio),
            returns=dict(mean=return_mean, var=return_var),
            val=dict(vpred=torch.mean(vpred), error=torch.mean((vpred - returns) ** 2),
                     clipfrac=vf_clipfrac, mean=value_mean, var=value_var),
        )
        return pg_loss, self.ppo_params['vf_coef'] * vf_loss, self.kl_ctl.value * kl_loss, flatten_dict(stats)

    def record_step_stats(self, **data):
        """Record training step statistics."""
        # kl_list = [logprobs - ref_logprobs for logprobs, ref_logprobs in zip(data['logprobs'], data['ref_logprobs'])]
        kl_list = self.kl_ctl.kl_list
        mean_kl = torch.mean(torch.stack([torch.sum(kl) for kl in kl_list]))
        kl_list_rew = self.kl_ctl.kl_list_rew
        mean_kl_rew = torch.mean(torch.stack([torch.sum(kl) for kl in kl_list_rew]))

        mean_entropy = torch.mean(torch.stack([torch.sum(-log_probs) for log_probs in data['logprobs']]))
        mean_non_score_reward = torch.mean(
            torch.stack([torch.sum(non_score_reward) for non_score_reward in data['non_score_reward']]))
        stats = {
            'objective/kl': mean_kl,
            'objective/kl_dist': kl_list,
            'objective/kl_rew': mean_kl_rew,
            'objective/kl_dist_rew': kl_list_rew,
            # too big, makes file messy
            # 'objective/logprobs': data['logprobs'],
            # 'objective/ref_logprobs': data['ref_logprobs'],
            'objective/kl_coef': self.kl_ctl.value,
            'objective/kl_coef_rew': self.kl_ctl_rew.value,
            'objective/entropy': mean_entropy,
            'ppo/mean_non_score_reward': mean_non_score_reward,
        }

        for k, v in data['train_stats'].items():
            stats[f'ppo/{k}'] = torch.mean(v, axis=0)
        stats['ppo/val/var_explained'] = 1 - stats['ppo/val/error'] / stats['ppo/returns/var']
        return stats


def train(model_name, single_game=True):
    from agents import NLPAgent
    from time import time

    UPDATE_FREQUENCY = 16
    LOG_FREQUENCY = 16

    buffer = ReplayBuffer(UPDATE_FREQUENCY)
    agent = NLPAgent(buffer, humanTurns=0)

    if single_game:
        print("Training")
        agent.train()  # Tell the agent it should update its parameters.
        player = Player(agent, "./games/tw-rewardsDense_goalDetailed.z8", verbose=False)  # Dense rewards game.

    else:
        print("Training on 100 games")
        agent.train()  # Tell the agent it should update its parameters.
        player = Player(agent, "./training_games/", verbose=False)  # Each game will be seen 5 times.

    ppo_config = {'batch_size': UPDATE_FREQUENCY, 'forward_batch_size': 8}
    ppo_trainer = PPOTrainer(model_name, player, buffer, agent, **ppo_config)

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
        logger=False,
        accelerator='gpu', devices=1,
        max_epochs=500,
        precision=16,
        strategy=DeepSpeedStrategy(
            config="ds_config_zero2_light.json"
        ),
    )

    trainer.fit(ppo_trainer)


if __name__ == "__main__":
    import argparse

    getEnvs()
    print("generated envs")

    # model_name = 'gpt2-xl'
    # model_name = 'EleutherAI/gpt-j-6B'
    model_name = 'EleutherAI/gpt-neo-1.3B'
    single_game = False

    train(model_name, single_game)
