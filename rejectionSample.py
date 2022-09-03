import datasets
from datasets import load_dataset
import math
import os
import sys
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional

import transformers
from transformers import Trainer
from transformers import TrainingArguments
from transformers import HfArgumentParser

import torch
import numpy as np

# using deepspeed huggingface trainer integreation

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

    def update(self, current, n_steps):
        target = self.target
        proportional_error = np.clip(current / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


class FixedKLController:
    """Fixed KL controller."""

    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current, n_steps):
        pass


class RejectionTuner:
    def __init__(self, model_name, player, agent, **reject_params):
        self.player = player

        # gpt2 and gpt2-xl
        if 'gpt2' in model_name:
            from transformers import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel
            self.model = GPT2LMHeadModel.from_pretrained(model_name)
            self.ref_model = GPT2LMHeadModel.from_pretrained(model_name)
            self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        elif model_name == 'gptj':
            from transformers import GPT2Tokenizer, GPTJForCausalLM
            if False:
                self.model = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", revision="float16",
                                                             torch_dtype=torch.float16, low_cpu_mem_usage=True)
                self.ref_model = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", revision="float16",
                                                                 torch_dtype=torch.float16, low_cpu_mem_usage=True)
            else:
                self.model = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B")
                self.ref_model = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B")
            self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

        print(self.model.config.torch_dtype)
        summary(self.model)

        self.config = self.model.config

        self.agent = agent

        # print(self.valueHead)

        self.data_collator = DataCollatorForLanguageModeling(self.tokenizer, mlm=False)

        if self.reject_params['adap_kl_ctrl']:
            self.kl_ctl = AdaptiveKLController(self.reject_params['init_kl_coef'],
                                               self.reject_params['target'],
                                               self.reject_params['horizon'])
        else:
            self.kl_ctl = FixedKLController(self.reject_params['init_kl_coef'])

    def step(self, data):
        trainer = Trainer(model=self.model, TrainingArguments=train_args, train_dataset=data, tokenizer=self.tokenizer)
        trainer.train()

if __name__ == "__main__":
    model_name = 'gpt2-xl'
    # model_name = 'gptj'
    low_ram = True
    single_game = True

    train_args = HfArgumentParser.parse_args_into_dataclasses(TrainingArguments)

    reject_tuner = RejectionTuner(model_name, train_args, player, agent)