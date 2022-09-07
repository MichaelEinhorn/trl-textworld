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

from datastructures import RejectionBuffer, RejectDataset

import torch
import numpy as np

from games import Player, getEnvs

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

from transformers import TrainerCallback
class CallBackRouter(TrainerCallback):
    def __init__(self, obj):
        self.obj = obj

    def on_epoch_begin(self, args, state, control, **kwargs):
        if hasattr(self.obj, 'on_epoch_begin'):
            self.obj.on_epoch_begin(args, state, control, **kwargs)


class RejectionTuner:
    default_params = {
        "lr": 1.41e-5,
        "adap_kl_ctrl": True,
        "init_kl_coef": 0.2,
        "target": 6,
        "horizon": 10000,
        "batch_size": 256,
        "epochs_per_game": 4,
    }

    

    def __init__(self, model_name, player, agent, buffer, train_args, **reject_params):

        self.reject_params = self.default_params
        self.reject_params.update(reject_params)
        # calls on epoch begin method
        self.callBackRouter = CallBackRouter(self)

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
        self.agent_buffer = buffer

        self.train_args = train_args
        self.reject_buffer = RejectionBuffer(min=False)
        self.train_ds = RejectDataset(self.reject_buffer, self.reject_params['batch_size'])

        self.data_collator = DataCollatorForLanguageModeling(self.tokenizer, mlm=False)

        if self.reject_params['adap_kl_ctrl']:
            self.kl_ctl = AdaptiveKLController(self.reject_params['init_kl_coef'],
                                               self.reject_params['target'],
                                               self.reject_params['horizon'])
        else:
            self.kl_ctl = FixedKLController(self.reject_params['init_kl_coef'])
    
    
    def __call__(self, input_ids, use_cache=False, past_key_values=None, outputVals=False, outputRef=False):
        return self.forward(input_ids, use_cache, past_key_values, outputVals, outputRef)
    # values variable for compatability
    def forward(self, input_ids, use_cache=False, past_key_values=None, outputVals=False, outputRef=False):
        if past_key_values is None:
            lmOut = self.model(input_ids, output_hidden_states=False, use_cache=use_cache)
        else:
            lmOut = self.model(input_ids, output_hidden_states=False, use_cache=use_cache, past_key_values=past_key_values)
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
            v = torch.tensor(0)
            output.append(v)
        # ref_logits, _, _ = self.ref_model(input_ids)
        
        return tuple(output)

    # data is list of strings
    def step(self):
        trainer = Trainer(model=self.model, TrainingArguments=self.train_args, train_dataset=self.train_ds, tokenizer=self.tokenizer)
        trainer.train()

    def on_epoch_begin(self, args, state, control, **kwargs):
        epoch = int(state.epoch)
        # fills reject dataset's buffer with a new set of experiences
        if epoch % self.reject_params["epochs_per_game"] == 0:
            self.runGame()

    def runGame(self):
        # self is passing the model to do forward passes with
        self.player.runGame(self, self.reject_params['batch_size'])
        self.agent.fillBuffer()
        scores, queries, responses, values_old, ret_cross, adv_cross = self.agent_buffer.sample(
            self.reject_params['batch_size'])

        # first part of original step, gets old logprobs and ref logprobs
        bs = self.reject_params['batch_size']

        # removes old experiences and only train on new ones. Remove to train on best of old and new
        # self.reject_buffer.clear()
        self.reject_buffer.append()

        for i in range(int(bs)):
            query = queries[i]
            response = responses[i]
            # use returns discounted across multiple transitions, they will be again discounted across tokens in a transition
            score = ret_cross[i]
            # use only current rewards
            # scores_batch = scores[i]

            input_ids = torch.cat([query, response])["input_ids"]

            self.reject_buffer.append(input_ids, score)

        self.reject_buffer.reject(0.5, threshType="frac")


def train(model_name, train_args, single_game=True):
    from agents import NLPAgent
    from time import time

    LOG_FREQUENCY = 10
    UPDATE_FREQUENCY = 10

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

    reject_params = {'batch_size': UPDATE_FREQUENCY}
    reject_tuner = RejectionTuner(model_name, train_args, player, agent, buffer, train_args, reject_params)

if __name__ == "__main__":
    model_name = 'gpt2-xl'
    # model_name = 'gptj'
    single_game = True
    getEnvs()


    train_args = HfArgumentParser.parse_args_into_dataclasses(TrainingArguments)



    train(train_args)