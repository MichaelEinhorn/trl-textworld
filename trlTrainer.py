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
import argparse
from collections import OrderedDict, deque
from typing import Tuple, List
import torch.optim as optim
from torch.optim import Optimizer
from deepspeed.ops.adam import DeepSpeedCPUAdam
from torch.utils.data import DataLoader
from torchinfo import summary

from valueHead import ValueHead
from games import VectorPlayer, getEnvs

from transformers import DataCollatorForLanguageModeling
from pytorch_lightning.strategies.deepspeed import DeepSpeedStrategy

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


class TRLTrainer(pl.LightningModule):
    def __init__(self, model_name, player, buffer, agent, **params):
        super().__init__()

        self.params = self.default_params
        self.params.update(params)
        self.alg_name = self.params["alg_name"]

        self.agent_buffer = buffer

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

        self.agent = agent

        self.data_collator = DataCollatorForLanguageModeling(self.tokenizer, mlm=False)

        if self.params['adap_kl_ctrl']:
            self.kl_ctl = AdaptiveKLController(self.params['init_kl_coef'],
                                               self.params['target'],
                                               self.params['horizon'])
        else:
            self.kl_ctl = FixedKLController(self.params['init_kl_coef'])

        if self.params['adap_kl_ctrl_rew']:
            self.kl_ctl_rew = AdaptiveKLController(self.params['init_kl_coef_rew'],
                                               self.params['target_rew'],
                                               self.params['horizon_rew'])
        else:
            self.kl_ctl_rew = FixedKLController(self.params['init_kl_coef_rew'])
            
        self.all_stats = []
        
        self.saveModelTime = 0
        self.saveStatTime = 0

    def getDevice(self):
        return self.device

    @property
    def deepspeed_offload(self) -> bool:
        strategy = self.trainer.strategy
        if isinstance(strategy, DeepSpeedStrategy):
            config = strategy.config['zero_optimization']
            return config.get('offload_optimizer') or config.get('offload_param')
        return False

    # def on_train_start(self):
    #     self.runGame()
    def on_train_epoch_start(self):
        # train on the same data epochs per game times before generating a new set
        if self.current_epoch % self.params['epochs_per_game'] == 0:
            self.runGame()
            
        self.epoch_time = time.time()

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

    def forward(self, input_ids, use_cache=False, past_key_values=None, outputVals=False, outputRef=False, attention_mask=None, outputLogits=True):
        return None

    def batched_forward_pass(self, queries, responses, outputLogits=True, outputVals=True, outputRef=True):
        return None

    def record_step_stats(self, **data):
        """Record training step statistics."""
        # kl_list = [logprobs - ref_logprobs for logprobs, ref_logprobs in zip(data['logprobs'], data['ref_logprobs'])]
        kl_list = self.kl_ctl.kl_list
        mean_kl = torch.mean(torch.stack([torch.sum(kl) for kl in kl_list]))
        kl_list_rew = self.kl_ctl_rew.kl_list
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
            'objective/vf_coef': self.params['vf_coef'],
            'objective/entropy': mean_entropy,
            f'{self.alg_name}/mean_non_score_reward': mean_non_score_reward,
        }
                   
        self.kl_ctl.kl_list = []
        

        for k, v in data['train_stats'].items():
            # print(k, v)
            stats[f'{self.alg_name}/{k}'] = torch.mean(v, axis=0)
        return stats