import torch
import os
from glob import glob
import gym
import textworld.gym

import numpy as np

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

        self.agent = agent
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

    def runGame(self, lightmodel, steps=10):
        print("running game for " + str(steps))
        for i in range(steps):
            if self.done:
                self.resetEnv()
                print("reset env")

            command = self.agent.act(self.obs, self.score, self.done, self.infos, lightmodel)
            self.obs, self.score, self.done, self.infos = self.env.step(command)
            if hasattr(self.agent, 'reportScore'):
                self.agent.reportScore(self.score, self.done, self.infos)
            self.nb_moves += 1

            if self.done:
                self.no_episode += 1

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

class VectorPlayer:
    def __init__(self, agent, path, max_step=100, verbose=True, num_agents=1, **kwargs):
        torch.manual_seed(20211021)  # For reproducibility when using action sampling.

        self.exTurns = None
        if "exTurns" in kwargs:
            self.exTurns = kwargs["exTurns"]

        self.agent = agent
        self.num_agents = num_agents
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
                                                   max_episode_steps=max_step, batch_size=num_agents,
                                                   auto_reset=True)
        print(self.env_id)
        self.env = gym.make(self.env_id)  # Create a Gym environment to play the text game.
        if verbose:
            if os.path.isdir(path):
                print(os.path.dirname(path), end="")
            else:
                print(os.path.basename(path), end="")

        # Collect some statistics: nb_steps, final reward.
        self.avg_moves, self.avg_scores, self.avg_norm_scores = [], [], []
        self.nb_moves = 0
        self.no_episode = 0
        self.done = True

    def resetEnv(self):
        self.obs, self.infos = self.env.reset()  # Start new episode.

        self.score = [0 for i in range(self.num_agents)]
        self.done = [False for i in range(self.num_agents)]
        self.nb_moves = 0

    def runGame(self, lightmodel, steps=10):
        print("running game for " + str(steps))
        self.resetEnv()

        exTurnSampler = None
        if self.exTurns is not None:
            exTurnSampler = torch.distributions.bernoulli.Bernoulli(probs=1 - self.exTurns)
        
        while steps > 0:
            stepsCompleted = self.num_agents

            if self.exTurns is not None:
                ex = exTurnSampler.sample()
                if ex == 1:
                    stepsCompleted = 0
                command = self.agent.act(self.obs, self.score, self.done, self.infos, lightmodel, exTurn=ex)
            else:
                command = self.agent.act(self.obs, self.score, self.done, self.infos, lightmodel)

            self.obs, self.score, self.done, self.infos = self.env.step(command)
            if hasattr(self.agent, 'reportScore'):
                self.agent.reportScore(self.score, self.done, self.infos)
            self.nb_moves += 1

            if True in self.done:
                self.no_episode += 1

                if self.verbose:
                    print(".", end="")
                self.avg_scores.append(self.score)
                self.avg_norm_scores.append(self.score / self.infos["max_score"])

            steps -= stepsCompleted

    def close(self):
        self.env.close()
        if self.verbose:
            if os.path.isdir(self.path):
                msg = "  \tavg. steps: {:5.1f}; avg. normalized score: {:4.1f} / {}."
                print(msg.format(self.nb_moves/self.no_episode, np.mean(self.avg_norm_scores), 1))
            else:
                msg = "  \tavg. steps: {:5.1f}; avg. score: {:4.1f} / {}."
                print(msg.format(self.nb_moves/self.no_episode, np.mean(self.avg_scores), self.infos["max_score"]))