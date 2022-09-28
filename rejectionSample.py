import math
import os
import sys
from argparse import Namespace
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional

import transformers
from transformers import Trainer
from transformers import TrainingArguments
from transformers import HfArgumentParser
from torchinfo import summary

from transformers import DataCollatorForLanguageModeling
from datastructures import RejectionBuffer, RejectDataset, ReplayBuffer

import torch
import numpy as np
from agents import NLPAgent, VectorNLPAgent

from games import Player, VectorPlayer, getEnvs

from core import pad_mask, logprobs_from_logits

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
        "adap_kl_ctrl": False,
        "init_kl_coef": 0.2,
        "target": 6,
        "horizon": 10000,
        "adap_kl_ctrl_rew": False,
        "init_kl_coef_rew": 0.2,
        "target_rew": 6,
        "horizon_rew": 10000,
        "batch_size": 256,
        "epochs_per_game": 4,
    }

    

    def __init__(self, model_name, player, agent, buffer, train_args=None):

        self.reject_params = self.default_params
        # self.reject_params.update(reject_params)
        # calls on epoch begin method
        self.callBackRouter = CallBackRouter(self)

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
        self.agent_buffer = buffer

        self.train_args = train_args
        self.reject_buffer = RejectionBuffer(min=False)
        self.train_ds = RejectDataset(self.reject_buffer, self.reject_params['batch_size'])

        self.data_collator = DataCollatorForLanguageModeling(self.tokenizer, mlm=False)
        self.current_epoch = 0

        if self.reject_params['adap_kl_ctrl']:
            self.kl_ctl = AdaptiveKLController(self.reject_params['init_kl_coef'],
                                               self.reject_params['target'],
                                               self.reject_params['horizon'])
        else:
            self.kl_ctl = FixedKLController(self.reject_params['init_kl_coef'])

        if self.reject_params['adap_kl_ctrl_rew']:
            self.kl_ctl_rew = AdaptiveKLController(self.reject_params['init_kl_coef_rew'],
                                               self.reject_params['target_rew'],
                                               self.reject_params['horizon_rew'])
        else:
            self.kl_ctl_rew = FixedKLController(self.reject_params['init_kl_coef_rew'])

    def getDevice(self):
        return self.model.device

    def __call__(self, input_ids, **kwargs):
        return self.forward(input_ids, **kwargs)

    # values variable for compatability
    def forward(self, input_ids, use_cache=False, past_key_values=None, outputVals=False, outputRef=False, attention_mask=None, outputLogits=True):
        output = Namespace()
        if outputLogits or outputVals:
            if past_key_values is None:
                lmOut = self.model(input_ids, output_hidden_states=outputVals, use_cache=use_cache, attention_mask=attention_mask)
            else:
                lmOut = self.model(input_ids, output_hidden_states=outputVals, use_cache=use_cache,
                                   past_key_values=past_key_values, attention_mask=attention_mask)
            # print(dir(lmOut))
            if outputLogits:
                logits = lmOut.logits
                output.logits = logits

            if use_cache:
                cache = lmOut.past_key_values
                output.cache = cache

        if outputRef:
            with torch.no_grad():
                ref_logits = self.ref_model(input_ids).logits
                output.ref_logits = ref_logits

        if outputVals:
            v = torch.tensor([[[0]]])
            output.values = v

        return output

    def batched_forward_pass(self, queries, responses, outputLogits=True, outputVals=True, outputRef=True):
        """Calculate model outputs in multiple batches."""
        bs = self.ppo_params['batch_size']
        fbs = self.ppo_params['forward_batch_size']
        all_logprobs = []
        all_ref_logprobs = []
        all_values = []

        output = Namespace()

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
                    logits = lmout.logits
                    logprobs = logprobs_from_logits(logits[:, :-1, :], input_ids[:, 1:])
                if outputRef:
                    ref_logits = lmout.ref_logits
                    ref_logprobs = logprobs_from_logits(ref_logits[:, :-1, :], input_ids[:, 1:])
                if outputVals:
                    v = lmout.v

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
                    logits = lmout.logits
                    logprobs = logprobs_from_logits(logits[:, :-1, :], input_ids[:, 1:])
                if outputRef:
                    ref_logits = lmout.ref_logits
                    ref_logprobs = logprobs_from_logits(ref_logits[:, :-1, :], input_ids[:, 1:])
                if outputVals:
                    v = lmout.v

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

        output.logprobs = all_logprobs
        output.values = all_values
        output.ref_logprobs = all_ref_logprobs
        return output

    # data is list of strings
    def startTraining(self):
        trainer = Trainer(model=self.model, args=self.train_args, train_dataset=self.train_ds, tokenizer=self.tokenizer, callbacks=[self.callBackRouter])
        trainer.train()

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.current_epoch = int(state.epoch)
        # fills reject dataset's buffer with a new set of experiences
        if self.current_epoch % self.reject_params["epochs_per_game"] == 0:
            self.runGame()
        print("epoch ", self.current_epoch)

    def runGame(self):
        # self is passing the model to do forward passes with
        self.player.runGame(self, self.reject_params['batch_size'])
        self.agent.fillBuffer()
        scores, queries, responses, values_next, ret_cross, adv_cross, values, logprobs = self.agent_buffer.sample(
            self.reject_params['batch_size'])

        # first part of original step, gets old logprobs and ref logprobs
        bs = self.reject_params['batch_size']

        ref_logprobs = self.batched_forward_pass(queries, responses, outputLogits=False, outputVals=False,
                                                 outputRef=True).ref_logprobs

        # removes old experiences and only train on new ones. Remove to train on best of old and new
        # self.reject_buffer.clear()

        rewards, non_score_rewards = self.compute_rewards(scores, logprobs, ref_logprobs)

        for i in range(int(bs)):
            query = queries[i]
            response = responses[i]
            # use returns discounted across multiple transitions

            score = ret_cross[i]
            kl_rew = torch.sum(non_score_rewards)
            score += kl_rew
            # use only current rewards
            # scores_batch = scores[i]

            input_ids = torch.cat([query, response])["input_ids"]

            self.reject_buffer.append(input_ids, score)

        self.reject_buffer.reject(0.5, threshType="frac")


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


def train(model_name, train_args, single_game=True, NUM_AGENTS=1):
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
        agent = VectorNLPAgent(buffer, num_agents=NUM_AGENTS)
        print("Training on 100 games")
        agent.train()  # Tell the agent it should update its parameters.
        player = VectorPlayer(agent, "./training_games/", verbose=False, num_agents=NUM_AGENTS,
                              exTurns=0.25)  # Each game will be seen 5 times.

    reject_params = {'batch_size': UPDATE_FREQUENCY}
    #         def __init__(self, model_name, player, agent, buffer, train_args, **reject_params):
    reject_tuner = RejectionTuner(model_name, player, agent, buffer, train_args=train_args)
    reject_tuner.startTraining()

if __name__ == "__main__":
    model_name = 'gpt2-xl'
    # model_name = 'gptj'
    single_game = False
    num_agents = 4
    getEnvs()

    print(TrainingArguments)
    parser = HfArgumentParser(TrainingArguments)
    train_args, unknown_args = parser.parse_args_into_dataclasses(return_remaining_strings=True)
    print(train_args)
    print(unknown_args)
    train(model_name, train_args, single_game=single_game, NUM_AGENTS=num_agents)
    
# deepspeed --num_gpus=1 rejectionSample.py --deepspeed ds_config_zero2.json --per_device_train_batch_size 1 --output_dir output_dir --overwrite_output_dir --fp16 --do_train --max_train_samples 500 --num_train_epochs 1