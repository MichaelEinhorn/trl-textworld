import numpy as np

import re
from typing import List, Mapping, Any, Optional
from collections import defaultdict
from datastructures import RollingBuffer

import textworld
import textworld.gym
from textworld import EnvInfos

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from transformers import top_k_top_p_filtering
from torch.nn import Identity

class RandomAgent(textworld.gym.Agent):
    """ Agent that randomly selects a command from the admissible ones. """

    def __init__(self, seed=1234):
        self.seed = seed
        self.rng = np.random.RandomState(self.seed)

    @property
    def infos_to_request(self) -> textworld.EnvInfos:
        return textworld.EnvInfos(admissible_commands=True)

    def act(self, obs: str, score: int, done: bool, infos: Mapping[str, Any]) -> str:
        return self.rng.choice(infos["admissible_commands"])


class HumanAgent(textworld.gym.Agent):
    """ Agent that randomly selects a command from the admissible ones. """

    def __init__(self, seed=1234):
        self.seed = seed
        self.rng = np.random.RandomState(self.seed)

    @property
    def infos_to_request(self) -> textworld.EnvInfos:
        return textworld.EnvInfos(admissible_commands=True)

    def act(self, obs: str, score: int, done: bool, infos: Mapping[str, Any]) -> str:
        print(obs)
        return input()


class NLPAgent:
    """ Hugging Face Transformer Agent """

    GAMMA = 0.9
    MEMORY_LEN = 3

    def __init__(self, buffer, humanTurns=0) -> None:
        self._initialized = False
        self._epsiode_has_started = False

        self.memory = RollingBuffer(self.MEMORY_LEN)
        self.transitionBuffer = buffer

        self.humanTurns = humanTurns
        self.humanTurnsRem = self.humanTurns

        self.mode = "test"
        self.transitions = []
        self.clearTextWorldArt = True

    def train(self):
        self.mode = "train"
        self.stats = {"max": defaultdict(list), "mean": defaultdict(list)}
        self.transitions = []
        # self.model.reset_hidden(1)
        self.last_score = 0
        self.no_train_step = 0
        self.clearTextWorldArt = True

        self.memory.clear()

    def test(self):
        self.clearTextWorldArt = True
        self.mode = "test"
        # self.model.reset_hidden(1)

    @property
    def infos_to_request(self) -> EnvInfos:
        return EnvInfos(description=True, inventory=True, admissible_commands=True,
                        won=True, lost=True)

    def _discount_rewards(self, last_values=None):
        returns, advantages = [], []
        if last_values is None:
            # not sure if this makes sense for when there is no next state
            # _, _, _, R = self.transitions[-1]
            R = 0
        else:
            R = last_values.data
        for t in reversed(range(len(self.transitions))):
            rewards, _, _, values = self.transitions[t]
            R = rewards + self.GAMMA * R
            adv = R - values
            returns.append(R)
            advantages.append(adv)

        return returns[::-1], advantages[::-1]

    def fillBuffer(self):
        # get discounted returns and advantages across multiple actions. Currently not used
        returns, advantages = self._discount_rewards()
        
        output = []

        for t in reversed(range(len(self.transitions))):
            rew, prompt, action, values = self.transitions[t]
            ret_cross = returns[t]
            adv_cross = advantages[t]
            # returns and advantages across multiple actions
            # ppo trainer computes returns and advantages across tokens within an action
            # not sure if these are usefull
            exper = (rew, prompt[0], action[0], values, ret_cross, adv_cross)
            self.transitionBuffer.append(exper)

        self.transitions = []

    # fill in results from action and train if time
    def reportScore(self, score, done, infos):
        if self.mode == "train":
            reward = score - self.last_score  # Reward is the gain/loss in score.
            self.last_score = score
            if infos["won"]:
                reward += 100
            if infos["lost"]:
                reward -= 100

            self.transitions[-1][0] = reward  # Update reward information. Was initialized as none

            self.no_train_step += 1

            self.stats["max"]["score"].append(score)

    def act(self, obs: str, score: int, done: bool, infos: Mapping[str, Any], lightmodel) -> Optional[str]:
        if self.clearTextWorldArt:
            self.clearTextWorldArt = False
            if "Welcome to TextWorld!" in obs:
                obs = obs[obs.index("Welcome to TextWorld!"):]
            elif "$$$$$$$" in obs:
                obs = obs[obs.rindex("$$$$$$$"):]

        # Build agent's observation: feedback + look + inventory.
        pastStates = ""
        for mem in self.memory:
            pastStates = pastStates + mem + "\n"
        admissible_commands_str = "options: "
        for adm_cmd in infos["admissible_commands"]:
            admissible_commands_str += adm_cmd + ", "
        input_ = "{}\n{}\n{}\n{}\nYou".format(obs, infos["description"], infos["inventory"], admissible_commands_str)
        prompt = pastStates + input_

        # grabs value of last token in action
        values = 0
        # convert text to tensor
        input_ids = lightmodel.tokenizer.encode(prompt, add_special_tokens=True, return_tensors="pt")
        print("prompt tokens: ", input_ids.shape)
        print(input_)

        if self.humanTurnsRem > 0:
            action = input()
            self.memory.append(input_ + action)
            self.humanTurnsRem -= 1
            return action

        new_tokens = 0
        next_token = None

        cache = None

        while new_tokens == 0 or (new_tokens < 20 and "\n" not in lightmodel.tokenizer.decode(next_token)
                                  and next_token != lightmodel.tokenizer.eos_token):
            # run model
            with torch.no_grad():
                # get logits, only get last value
                input_ids = input_ids.to(lightmodel.device)
                if cache is None:
                    logits, cache, values = lightmodel(input_ids, use_cache=True, outputVals=True)
                else:
                    logits, cache, values = lightmodel(input_ids[:, -1:], outputVals=True, use_cache=True, past_key_values=cache)
                
                next_token_logits = logits[:, -1, :]
                next_token_logits = top_k_top_p_filtering(next_token_logits, top_k=0, top_p=1)
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
                input_ids = torch.cat([input_ids, next_token.unsqueeze(-1)], dim=-1)

                new_tokens += 1

        action_tens = input_ids[:, -new_tokens:]
        prompt_tens = input_ids[:, :-new_tokens]

        action = lightmodel.tokenizer.decode(action_tens[0, :])

        print("action")
        print(action)

        # only grab last token
        values = values[0, -1, 0]
        print("last token value in action")
        print(values)

        self.memory.append(input_ + action)

        if self.mode == "train" and not done:
            self.transitions.append([None, prompt_tens.to(torch.device("cpu")), action_tens.to(torch.device("cpu")), values.to(torch.device("cpu"))])  # Reward will be set on the next call

        if done:
            self.last_score = 0  # Will be starting a new episode. Reset the last score.
            self.memory.clear()
            self.clearTextWorldArt = True
            self.humanTurnsRem = self.humanTurns

        return action
