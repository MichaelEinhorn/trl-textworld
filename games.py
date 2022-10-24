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


# Reward is the gain/loss in score.
class GameReward:
    def __init__(self, value=1, num_agents=1):
        self.value=value
        self.num_agents=num_agents
        self.last_score = [0 for i in range(self.num_agents)]
        
    def reward(self, score, actionList, done, infos):
        rew = [0 for i in range(self.num_agents)]
        for i in range(self.num_agents):
            rew[i] = self.value * (score[i] - self.last_score[i])
            self.last_score[i] = score[i]
            # Will be starting a new episode. Reset the last score.
            if done[i]:
                self.last_score[i] = 0
        return rew

    def reset(self):
        self.last_score = [0 for i in range(self.num_agents)]

# adds value on win, subtracts on lose
class WinReward:
    def __init__(self, value=100, parentReward=None, num_agents=1):
        self.value = value
        self.parentReward = parentReward
        self.num_agents=num_agents

    def reward(self, score, actionList, done, infos):
        rew = [0 for i in range(self.num_agents)]
        if self.parentReward is not None:
            rew = self.parentReward.reward(score, actionList, done, infos)

        for i in range(self.num_agents):
            if infos["won"][i]:
                rew += self.value
            if infos["lost"][i]:
                rew -= self.value
        return rew

    def reset(self):
        if self.parentReward is not None:
            self.parentReward.reset()

class LivingReward:
    def __init__(self, value=-0.1, parentReward=None, num_agents=1):
        self.value = value
        self.parentReward = parentReward
        self.num_agents=num_agents

    def reward(self, score, actionList, done, infos):
        rew = [0 for i in range(self.num_agents)]
        if self.parentReward is not None:
            rew = self.parentReward.reward(score, actionList, done, infos)

        for i in range(self.num_agents):
            rew += self.value
        return rew

    def reset(self):
        if self.parentReward is not None:
            self.parentReward.reset()

# negatively rewards invalid actions
class InvalidReward:
    def __init__(self, value=-1, parentReward=None, num_agents=1):
        self.value = value
        self.parentReward = parentReward
        self.lastActionInfos = [None for i in range(num_agents)]
        self.num_agents=num_agents
        
    def reward(self, score, actionList, done, infos):
        rew = [0 for i in range(self.num_agents)]
        if self.parentReward is not None:
            rew = self.parentReward.reward(score, actionList, done, infos)
                 
        for i in range(self.num_agents):
            # test for invalid action by either a none or an unchanged action
            # print(infos["last_action"][i])
            # print(self.lastActionInfos[i])
            # print(infos["last_action"][i] == self.lastActionInfos[i])
            # print(infos["last_action"][i] is None or infos["last_action"][i] == self.lastActionInfos[i])
            if infos["last_action"][i] is None or infos["last_action"][i] == self.lastActionInfos[i]:
                rew[i] += self.value
            # print(rew[i])
        self.lastActionInfos = infos["last_action"]
        return rew

    def reset(self):
        self.lastActionInfos = [None for i in range(self.num_agents)]
        if self.parentReward is not None:
            self.parentReward.reset()

# number of a set of letters within the action
class LetterReward:
    def __init__(self, value=1, parentReward=None, num_agents=1, letters=('e', 'E')):
        self.value = value
        self.parentReward = parentReward
        self.num_agents=num_agents
        self.letters = letters
        
    def reward(self, score, actionList, done, infos):
        rew = [0 for i in range(self.num_agents)]
        if self.parentReward is not None:
            rew = self.parentReward.reward(score, actionList, done, infos)
        
        for i in range(self.num_agents):
            count = 0
            for letter in self.letters:
                count += actionList[i].count(letter)
            rew[i] += self.value * count
        return rew

    def reset(self):
        if self.parentReward is not None:
            self.parentReward.reset()
        

class VectorPlayer:
    def __init__(self, agent, path, rewardFunc=None, max_step=100, verbose=True, num_agents=1, rank=0, world_size=1, **kwargs):
        # torch.manual_seed(20211021 + rank)  # For reproducibility when using action sampling.
        self.rewardFunc = rewardFunc
        if self.rewardFunc is None:
            self.rewardFunc = GameReward(num_agents=num_agents)

        self.exTurns = None
        if "exTurns" in kwargs:
            self.exTurns = kwargs["exTurns"]
        self.decisionTrans = False
        if "decisionTrans" in kwargs:
            self.decisionTrans = kwargs["decisionTrans"]

        self.agent = agent
        self.num_agents = num_agents
        self.infos_to_request = agent.infos_to_request
        self.infos_to_request.max_score = True  # Needed to normalize the scores.

        self.path = path
        self.verbose = verbose

        self.gamefiles = [path]
        if os.path.isdir(path):
            self.gamefiles = glob(os.path.join(path, "*.z8"))

        # split games between processes
        self.gamefiles = [self.gamefiles[i] for i in range(rank, len(self.gamefiles), world_size)]

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

        self.rank = rank
        self.world_size = world_size

        self.obs, self.score, self.done, self.infos = None, None, None, None
        
        self.resetEnv()

    def resetEnv(self):
        self.obs, self.infos = self.env.reset()  # Start new episode.

        self.score = [0 for i in range(self.num_agents)]
        self.done = [False for i in range(self.num_agents)]
        self.nb_moves = 0
        self.rewardFunc.reset()

    def runGame(self, lightmodel, steps=10):
        if self.rank == 0:
            print("\n" + "running game for " + str(steps) + " steps on rank " + str(self.rank))
        exTurnSampler = None
        if self.exTurns is not None:
            exTurnSampler = torch.distributions.bernoulli.Bernoulli(probs=self.exTurns)
        total_steps = steps
        while steps > 0:
            if self.rank == 0:
                print("\r", total_steps - steps, "/", total_steps, sep="", end="", flush=True)
            # print(total_steps - steps, "/", total_steps, " on rank ", self.rank, sep="", flush=True)
            # torch.distributed.barrier()
            stepsCompleted = self.num_agents

            if self.exTurns is not None:
                ex = exTurnSampler.sample()
                if ex == 1:
                    stepsCompleted = 0
                command = self.agent.act(self.obs, self.score, self.done, self.infos, lightmodel, exTurn=ex)
            elif self.decisionTrans:
                command = self.agent.act(self.obs, self.score, self.done, self.infos, lightmodel, decisionTrans=self.decisionTrans)
            else:
                command = self.agent.act(self.obs, self.score, self.done, self.infos, lightmodel)

            self.obs, self.score, self.done, self.infos = self.env.step(command)
            
            if True in self.done:
                self.no_episode += 1

                if self.verbose:
                    print(".", end="")
                self.avg_scores.append(self.score)
                # self.avg_norm_scores.append(self.score / self.infos["max_score"])

            steps -= stepsCompleted
            
            # if steps <= 0:
            if hasattr(self.agent, 'report'):
                rew = self.rewardFunc.reward(self.score, command, self.done, self.infos)
                self.agent.report(rew, self.score, self.done, self.infos)

            self.nb_moves += 1
        if self.rank == 0:
            print("\r", total_steps - steps, "/", total_steps, sep="", flush=True)
        # print(total_steps - steps, "/", total_steps, " on rank ", self.rank, sep="", flush=True)
        torch.distributed.barrier()

    def close(self):
        self.env.close()
        if self.verbose:
            if os.path.isdir(self.path):
                msg = "  \tavg. steps: {:5.1f}; avg. normalized score: {:4.1f} / {}."
                print(msg.format(self.nb_moves/self.no_episode, np.mean(self.avg_norm_scores), 1))
            else:
                msg = "  \tavg. steps: {:5.1f}; avg. score: {:4.1f} / {}."
                print(msg.format(self.nb_moves/self.no_episode, np.mean(self.avg_scores), self.infos["max_score"]))


# human player
if __name__ == "__main__":
    from agents import HumanAgent
    getEnvs()

    gameRew = GameReward(value=1, num_agents=1)
    gameRew = WinReward(value=100, num_agents=1, parentReward=gameRew)
    gameRew = InvalidReward(value=-1, num_agents=1, parentReward=gameRew)

    letterRew = LetterReward(value=1, num_agents=1, letters=('e', 'E'))

    agent_ = HumanAgent(num_agents=1, MEMORY_LEN=1)
    player_ = VectorPlayer(agent_, "./training_games/", verbose=False, num_agents=1,
                 rank=0, world_size=1, rewardFunc=gameRew)  # Each game will be seen 5 times.
    player_.runGame(None, steps=100)
