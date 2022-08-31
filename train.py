import torch
import os
from glob import glob
import gym
import textworld.gym

import numpy as np

import pytorch_lightning as pl
import argparse
from collections import OrderedDict, deque
from typing import Tuple, List
import torch.optim as optim
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from ppoValHead import PPOTrainer
from torchinfo import summary
from datastructures import ReplayBuffer

import deepspeed


def getEnvs(download=True):
    if download:
        os.system("wget https://aka.ms/textworld/notebooks/data.zip")
        os.system("unzip -nq data.zip && rm -f data.zip")
    else:
        # Same as !make_games.sh
        os.system(
            "tw-make tw-simple --rewards dense    --goal detailed --seed 18 --test --silent -f --output games/tw-rewardsDense_goalDetailed.z8")
        os.system(
            "tw-make tw-simple --rewards balanced --goal detailed --seed 18 --test --silent -f --output games/tw-rewardsBalanced_goalDetailed.z8")
        os.system(
            "tw-make tw-simple --rewards sparse   --goal detailed --seed 18 --test --silent -f --output games/tw-rewardsSparse_goalDetailed.z8")
        os.system(
            "tw-make tw-simple --rewards dense    --goal brief    --seed 18 --test --silent -f --output games/tw-rewardsDense_goalBrief.z8")
        os.system(
            "tw-make tw-simple --rewards balanced --goal brief    --seed 18 --test --silent -f --output games/tw-rewardsBalanced_goalBrief.z8")
        os.system(
            "tw-make tw-simple --rewards sparse   --goal brief    --seed 18 --test --silent -f --output games/tw-rewardsSparse_goalBrief.z8")
        os.system(
            "tw-make tw-simple --rewards sparse   --goal none     --seed 18 --test --silent -f --output games/tw-rewardsSparse_goalNone.z8")


class Player:
    def __init__(self, agent, path, max_step=100, verbose=True):
        torch.manual_seed(20211021)  # For reproducibility when using action sampling.

        self.infos_to_request = agent.infos_to_request
        self.infos_to_request.max_score = True  # Needed to normalize the scores.

        self.path = path
        self.verbose = verbose

        self.gamefiles = [path]
        if os.path.isdir(path):
            self.gamefiles = glob(os.path.join(path, "*.z8"))

        print(self.gamefiles)
        self.env_id = textworld.gym.register_games(self.gamefiles,
                                                   request_infos=self.infos_to_request,
                                                   max_episode_steps=max_step)
        print(self.env_id)
        self.env = gym.make(self.env_id)  # Create a Gym environment to play the text game.
        if verbose:
            if os.path.isdir(path):
                print(os.path.dirname(path), end="")
            else:
                print(os.path.basename(path), end="")

        # Collect some statistics: nb_steps, final reward.
        self.avg_moves, self.avg_scores, self.avg_norm_scores = [], [], []

        self.no_episode = 0
        self.done = True

    def resetEnv(self):
        self.obs, self.infos = self.env.reset()  # Start new episode.

        self.score = 0
        self.done = False
        self.nb_moves = 0

    def runGame(self, steps=10):
        print("running game for " + str(steps))
        for i in range(steps):
            if self.done:
                self.resetEnv()

            command = self.agent.act(self.obs, self.score, self.done, self.infos)
            self.obs, self.score, self.done, self.infos = self.env.step(command)
            if hasattr(self.agent, 'reportScore'):
                self.agent.reportScore(self.score, self.done, self.infos)
            self.nb_moves += 1

            if self.done:
                self.no_episode += 1
                self.agent.act(self.obs, self.score, self.done, self.infos)  # Let the agent know the game is done.

                if self.verbose:
                    print(".", end="")
                self.avg_moves.append(self.nb_moves)
                self.avg_scores.append(self.score)
                self.avg_norm_scores.append(self.score / self.infos["max_score"])

    def close(self):
        self.env.close()
        if self.verbose:
            if os.path.isdir(self.path):
                msg = "  \tavg. steps: {:5.1f}; avg. normalized score: {:4.1f} / {}."
                print(msg.format(np.mean(self.avg_moves), np.mean(self.avg_norm_scores), 1))
            else:
                msg = "  \tavg. steps: {:5.1f}; avg. score: {:4.1f} / {}."
                print(msg.format(np.mean(self.avg_moves), np.mean(self.avg_scores), self.infos["max_score"]))


def train(model_name, low_ram=True, single_game=True):
    from agents import NLPAgent
    from time import time

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # gpt2 and gpt2-xl
    if 'gpt2' in model_name:
        from transformers import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel
        model = GPT2LMHeadModel.from_pretrained(model_name)
        model_ref = GPT2LMHeadModel.from_pretrained(model_name)
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    elif model_name == 'gptj':
        from transformers import GPT2Tokenizer, GPTJForCausalLM
        if low_ram:
            model = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", revision="float16",
                                                    torch_dtype=torch.float16, low_cpu_mem_usage=True)
            model_ref = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", revision="float16",
                                                        torch_dtype=torch.float16, low_cpu_mem_usage=True)
        else:
            model = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B")
            model_ref = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B")
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    print(model.config.torch_dtype)

    UPDATE_FREQUENCY = 10
    LOG_FREQUENCY = 10

    buffer = ReplayBuffer(UPDATE_FREQUENCY)
    agent = NLPAgent(model, tokenizer, buffer, humanTurns=0)
    summary(model_ref)

    model = model.to(device)
    model_ref = model_ref.to(device)

    if single_game:
        print("Training")
        agent.train()  # Tell the agent it should update its parameters.
        player = Player(agent, "./games/tw-rewardsDense_goalDetailed.z8", verbose=False)  # Dense rewards game.

    else:
        print("Training on 100 games")
        agent.train()  # Tell the agent it should update its parameters.
        player = Player(agent, "./training_games/", verbose=False)  # Each game will be seen 5 times.

    if model_ref is not None:
        # initialize trainer
        ppo_config = {'batch_size': UPDATE_FREQUENCY, 'forward_batch_size': 1}
        ppo_trainer = PPOTrainer(model, model_ref, tokenizer, player, buffer, **ppo_config)
        valueHead = ppo_trainer.valueHead
        agent.valueHead = valueHead

    from pytorch_lightning.strategies.deepspeed import DeepSpeedStrategy
    trainer = pl.Trainer(
        logger=False,
        accelerator='gpu', devices=1,
        max_epochs=500,
        precision=16,
        strategy=DeepSpeedStrategy(
            zero_optimization=True,
            stage=3))

    trainer.fit(ppo_trainer)


if __name__ == "__main__":
    import argparse

    getEnvs()
    print("generated envs")

    model_name = 'gpt2-xl'
    # model_name = 'gptj'
    low_ram = True
    single_game = True

    train(model_name, low_ram, single_game)
