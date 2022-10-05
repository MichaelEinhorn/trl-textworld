import numpy as np
from pathlib import Path
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

from core import logprobs_from_logits

import string
import re

def clean_str(s):
    allowed_chars = " " + string.ascii_letters + string.digits
    s = re.sub(f'[^{allowed_chars}]', '', s)
    return s

def printFile(s, i, epoch):
        with open(f"trajectories/epoch_{epoch}_agent_{i}.txt", "a") as myfile:
            myfile.write(s + "\n")

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

    GAMMA = 0.5
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

        # PPO trainer uses next values to get a value across transitions
        self.returnNextValues = True
        
        self.testCountLetters = ('e', 'E') # None
        
        self.rewValStat = []

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
            # dont discount between episodes
            rewards, _, _, values, done, _, _ = self.transitions[t]
            if done:
                R = 0
            R = rewards + self.GAMMA * R
            adv = R - values
            returns.append(R)
            advantages.append(adv)

        return returns[::-1], advantages[::-1]

    def fillBuffer(self):
        np.savetxt("rewVals.csv", self.rewValStat, delimiter=',')
        # get discounted returns and advantages across multiple actions. Currently not used
        returns, advantages = self._discount_rewards()

        for t in reversed(range(len(self.transitions))):
            rew, prompt, action, next_value, done, value, logprob = self.transitions[t]
            ret_cross = returns[t]
            adv_cross = advantages[t]
            # returns and advantages across multiple actions
            # ppo trainer computes returns and advantages across tokens within an action
            # not sure if these are usefull
            exper = (rew, prompt[0], action[0], next_value, ret_cross, adv_cross, value[0], logprob[0])
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
            
            if self.testCountLetters is None:
                self.transitions[-1][0] = reward  # Update reward information. Was initialized as none

            self.no_train_step += 1

            self.stats["max"]["score"].append(score)

            # notification of done does not have a new state
            if done:
                self.last_score = 0  # Will be starting a new episode. Reset the last score.
                self.memory.clear()
                self.clearTextWorldArt = True
                self.humanTurnsRem = self.humanTurns
                # mark last transition of episode
                if len(self.transitions) != 0:
                    self.transitions[-1][4] = True

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
        if self.testCountLetters is not None:
            prompt = "hello"
            print(prompt)
        input_ids = lightmodel.tokenizer.encode(prompt, add_special_tokens=True, return_tensors="pt")
        if self.testCountLetters is None:
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
        
        logprobList = []
        valuesList = []
        while new_tokens == 0 or (new_tokens < 20 and "\n" not in lightmodel.tokenizer.decode(next_token)
                                  and next_token != lightmodel.tokenizer.eos_token):
            # run model
            with torch.no_grad():
                # get logits, only get last value
                input_ids = input_ids.to(lightmodel.getDevice())
                # input_ids = input_ids.to(lightmodel.model.device)
                if cache is None:
                    lmout = lightmodel(input_ids, use_cache=True, outputVals=True)
                else:
                    lmout = lightmodel(input_ids[:, -1:], outputVals=True, use_cache=True,
                                                       past_key_values=cache)

                logits, cache, values = lmout["logits"], lmout["cache"], lmout["values"]
                
                
                
                
                next_token_logits = logits[:, -1, :]
                next_token_logits = top_k_top_p_filtering(next_token_logits, top_k=0, top_p=1)
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
                input_ids = torch.cat([input_ids, next_token.unsqueeze(-1)], dim=-1)
                
                logprob = logprobs_from_logits(logits[:, -1:, :], next_token.unsqueeze(-1))
                logprobList.append(logprob)
                valuesList.append(values[:, -1, :])

                new_tokens += 1
                
        # print(logprobList)
        # print(valuesList)
        logprob = torch.stack(logprobList, dim=1)
        logprob = logprob.squeeze(2)
        print("logprob ", logprob.shape)
        values = torch.stack(valuesList, dim=1)

        # doesn't need shifting since input ids is already 1 longer than logits
        

        action_tens = input_ids[:, -new_tokens:]
        prompt_tens = input_ids[:, :-new_tokens]

        action = lightmodel.tokenizer.decode(action_tens[0, :])

        # print("action")
        print(action)

        # only grab last token
        # doesn't need shifting since input ids is already 1 longer than values
        # val = values[:, -new_tokens:]
        first_value = val[0, 0, 0]
        print("first value in action", first_value)

        self.memory.append(input_ + action)

        if self.mode == "train" and not done:
            reward = None
            if self.testCountLetters is not None:
                count = 0
                for letter in self.testCountLetters:
                    count += action.count(letter)
                # count /= len(action)
                reward = torch.tensor(count, dtype=val.dtype)
                print("reward")
                print(reward)
                self.rewValStat.append([lightmodel.current_epoch, reward, first_value.detach().cpu().numpy()])

            if not self.returnNextValues:
                self.transitions.append(
                    [reward, prompt_tens.to(torch.device("cpu")), action_tens.to(torch.device("cpu")),
                     first_value.to(torch.device("cpu")), False, self.GAMMA * val.to("cpu"), logprob.to("cpu")])
                # additional discount for cross experience values
            else:
                #                                       # not done on last step
                if len(self.transitions) != 0 and not self.transitions[-1][4]:
                    self.transitions[-1][3] = first_value.to(torch.device("cpu"))
                self.transitions.append(
                    [reward, prompt_tens.to(torch.device("cpu")), action_tens.to(torch.device("cpu")),
                     torch.tensor(0, dtype=val.dtype), False, val.to("cpu"), logprob.to("cpu")])

        # removes non ascii chars and \
        action = clean_str(action)
        # print(action)
        return action


class VectorNLPAgent:
    """ Hugging Face Transformer Agent """

    GAMMA = 0.5
    MEMORY_LEN = 2

    def __init__(self, buffer, num_agents=1) -> None:
        self._initialized = False
        self._epsiode_has_started = False
        self.num_agents = num_agents

        self.memory = [RollingBuffer(self.MEMORY_LEN) for i in range(self.num_agents)]
        self.transitionBuffer = buffer

        self.mode = "test"
        self.transitions = [[] for i in range(self.num_agents)]
        self.clearTextWorldArt = [True for i in range(self.num_agents)]

        # PPO trainer uses next values to get a value across transitions
        self.returnNextValues = True

        self.testCountLetters = ('e', 'E')  # None

        self.rewValStat = []
        
        Path("trajectories").mkdir(parents=True, exist_ok=True)

    def train(self):
        self.mode = "train"
        self.stats = {"max": defaultdict(list), "mean": defaultdict(list)}
        self.transitions = [[] for i in range(self.num_agents)]
        # self.model.reset_hidden(1)
        self.last_score = [0 for i in range(self.num_agents)]
        self.no_train_step = [0 for i in range(self.num_agents)]
        self.clearTextWorldArt = [True for i in range(self.num_agents)]

        for mem in self.memory:
            mem.clear()

    def test(self):
        self.clearTextWorldArt = [True for i in range(self.num_agents)]
        self.mode = "test"

    @property
    def infos_to_request(self) -> EnvInfos:
        return EnvInfos(description=True, inventory=True, admissible_commands=True,
                        won=True, lost=True)

    def _discount_rewards(self):
        returnsList, advantagesList = [], []
        for i in range(self.num_agents):
            trans = self.transitions[i]
            R = 0
            returns, advantages = [], []
            for t in reversed(range(len(trans))):
                # dont discount between episodes
                rewards, _, _, values, done, _, _ = trans[t]
                if done:
                    R = 0
                R = rewards + self.GAMMA * R
                adv = R - values
                returns.append(R)
                advantages.append(adv)

            returnsList.append(returns[::-1])
            advantagesList.append(advantages[::-1])
        return returnsList, advantagesList

    def fillBuffer(self):
        np.savetxt("rewVals.csv", self.rewValStat, delimiter=',')
        # get discounted returns and advantages across multiple actions. Currently not used
        returnsList, advantagesList = self._discount_rewards()
        for i in range(self.num_agents):
            trans = self.transitions[i]
            returns = returnsList[i]
            advantages = advantagesList[i]
            for t in reversed(range(len(trans))):
                rew, prompt, action, next_value, done, value, logprob = trans[t]
                ret_cross = returns[t]
                adv_cross = advantages[t]
                # returns and advantages across multiple actions
                # ppo trainer computes returns and advantages across tokens within an action
                # not sure if these are usefull
                exper = (rew, prompt[0], action[0], next_value, ret_cross, adv_cross, value[0], logprob[0])
                self.transitionBuffer.append(exper)

        self.transitions = [[] for i in range(self.num_agents)]

    # fill in results from action and train if time
    def reportScore(self, score, done, infos):
        for i in range(self.num_agents):
            if self.mode == "train":
                reward = score[i] - self.last_score[i]  # Reward is the gain/loss in score.
                self.last_score[i] = score[i]
                if infos["won"][i]:
                    reward += 100
                if infos["lost"][i]:
                    reward -= 100

                if self.testCountLetters is None:
                    self.transitions[i][-1][0] = reward  # Update reward information. Was initialized as none

                self.no_train_step[i] += 1

                self.stats["max"]["score"].append(score)

                # notification of done does not have a new state
                if done[i]:
                    self.last_score[i] = 0  # Will be starting a new episode. Reset the last score.
                    self.memory[i].clear()
                    self.clearTextWorldArt[i] = True
                    # mark last transition of episode
                    if len(self.transitions[i]) != 0:
                        self.transitions[i][-1][4] = True
                    
    def act(self, observation, score, done, infos, lightmodel, **kwargs):
        promptList = []
        inputList = []
        epoch = lightmodel.current_epoch

        # infos is dict of lists
        for i in range(self.num_agents):
            obs = observation[i]
            if self.clearTextWorldArt[i]:
                self.clearTextWorldArt[i] = False
                if "Welcome to TextWorld!" in obs:
                    obs = obs[obs.index("Welcome to TextWorld!"):]
                elif "$$$$$$$" in obs:
                    obs = obs[obs.rindex("$$$$$$$"):]

            # Build agent's observation: feedback + look + inventory.
            pastStates = ""
            for mem in self.memory[i]:
                pastStates = pastStates + mem + "\n"
            admissible_commands_str = "actions: "
            for adm_cmd in infos["admissible_commands"][i]:
                admissible_commands_str += adm_cmd + ", "
            input_ = "{}\n{}\n{}\n{}\nWhat do you do?\n".format(obs, infos["description"][i], infos["inventory"][i], admissible_commands_str)
            prompt = pastStates + input_

            # convert text to tensor

            if self.testCountLetters is not None:
                # prompt = "hello"
                # if i == 0:
                #     print(prompt)
                printFile(input_, i, epoch)

            if self.testCountLetters is None:
                # if i == 0:
                #     # print("prompt tokens: ", input_ids.shape)
                #     print(input_)
                printFile(input_, i, epoch)

            promptList.append(prompt)
            inputList.append(input_)

        # does a random admissable action and adds to memory. Does not create a transition in experience buffer
        actionList = []
        if "exTurn" in kwargs:
            if kwargs["exTurn"] == 1:
                for i in range(self.num_agents):
                    commands = infos["admissible_commands"][i]
                    idx = np.random.choice(len(commands))
                    action = commands[idx]
                    input_ = inputList[i]
                    self.memory[i].append(input_ + action)
                    printFile("example turn", i, epoch)
                    printFile(action, i, epoch)
                    actionList.append(action)
                return actionList

        model_input = lightmodel.tokenizer(promptList, add_special_tokens=True, return_tensors="pt", padding=True, return_attention_mask=True)
        input_ids = model_input["input_ids"]
        attention_mask = model_input["attention_mask"]
        new_tokens = 0
        next_token = None

        values = 0

        cache = None
        # 0 is done, 1 is continuing
        finished = torch.ones((self.num_agents,), device=lightmodel.getDevice())
        maxLen = 20
        genLengths = [maxLen for i in range(self.num_agents)]

        logprobList = []
        valuesList = []
        while new_tokens == 0 or (new_tokens < maxLen and torch.sum(finished) > 0):
            # run model
            with torch.no_grad():
                # get logits, only get last value
                input_ids = input_ids.to(lightmodel.getDevice())
                attention_mask = attention_mask.to(lightmodel.getDevice())
                # input_ids = input_ids.to(lightmodel.model.device)
                if cache is None:
                    lmout = lightmodel(input_ids, use_cache=True, outputVals=True, attention_mask=attention_mask)
                else:
                    # print("cache ", input_ids[:, -1:].shape, len(cache), attention_mask.shape)
                    lmout = lightmodel(input_ids[:, -1:], outputVals=True, use_cache=True,
                                                       past_key_values=cache, attention_mask=attention_mask)

                logits, cache, values = lmout["logits"], lmout["cache"], lmout["values"]

                next_token_logits = logits[:, -1, :]
                next_token_logits = top_k_top_p_filtering(next_token_logits, top_k=0, top_p=1)
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
                input_ids = torch.cat([input_ids, next_token.unsqueeze(-1)], dim=-1)
                
                # keep last token in att
                # breaks indexing if this is after the finished check
                attention_mask = torch.cat([attention_mask, finished.unsqueeze(-1)], dim=-1)

                for i in range(self.num_agents):
                    # only if not finished
                    if finished[i] == 1 and ("\n" in lightmodel.tokenizer.decode(next_token[i]) or next_token[i] == lightmodel.tokenizer.eos_token):
                        finished[i] = 0
                        genLengths[i] = new_tokens + 1

                logprob = logprobs_from_logits(logits[:, -1:, :], next_token.unsqueeze(-1))
                logprobList.append(logprob)
                valuesList.append(values[:, -1, :])

                new_tokens += 1
                
        # print("att shape ", input_ids.shape, attention_mask.shape)
        # print(logprobList)
        # print(valuesList)
        logprob = torch.stack(logprobList, dim=1)
        logprob = logprob.squeeze(2)
        values = torch.stack(valuesList, dim=1)
        
        # print("val logp ", values.shape, logprob.shape)


        for i in range(self.num_agents):
            inp = input_ids[i:i+1]
            att = attention_mask[i]
            

            # first and last nonzero index
            start = torch.arange(att.shape[0], 0, -1, device=lightmodel.getDevice())
            start = torch.argmax(start * att)
            end = torch.arange(att.shape[0], device=lightmodel.getDevice())
            end = torch.argmax(end * att)
            # print("att ", i, att, start, end, genLengths[i])

            inp = inp[:, start:end+1]
            action_tens = inp[:, -genLengths[i]:]
            prompt_tens = inp[:, :-genLengths[i]]
            # print("inp act prompt shape ", inp.shape, action_tens.shape, prompt_tens.shape, i)

            action = lightmodel.tokenizer.decode(action_tens[0, :])
            input_ = inputList[i]

            # doesn't need shifting since input ids is already 1 longer than logprob
            logp = logprob[i:i+1, :genLengths[i]]

            # if i == 0:
            #     print("action")
            #     print(action)
            printFile(clean_str(action), i, epoch)

            # doesn't need shifting since input ids is already 1 longer than values
            val = values[i:i+1, :genLengths[i]]
            # print(values.shape, val.shape, i)
            first_value = val[0, 0, 0]
            # if i == 0:
            #     print("first value in action", first_value)
            #     # print(value)
            printFile("first value in action " + str(first_value), i, epoch)
            # only grab last token
            # value = values[i, genLengths[i] - 1, 0]

            self.memory[i].append(input_ + action)

            if self.mode == "train":
                reward = None  # Reward will be set on the next call

                if self.testCountLetters is not None:
                    count = 0
                    for letter in self.testCountLetters:
                        count += action.count(letter)
                    # count /= len(action)
                    reward = torch.tensor(count, dtype=first_value.dtype)
                    # if i == 0:
                    #     print("reward", reward)
                    printFile("reward " + str(reward), i, epoch)
                    self.rewValStat.append([lightmodel.current_epoch, reward, first_value.detach().cpu().numpy()])

                if not self.returnNextValues:
                    self.transitions[i].append(
                        [reward, prompt_tens.to(torch.device("cpu")), action_tens.to(torch.device("cpu")),
                         first_value.to(torch.device("cpu")), False, self.GAMMA * val.to("cpu"), logp.to("cpu")])
                    # additional discount for cross experience values
                else:
                    #                                       # not done on last step
                    if len(self.transitions[i]) != 0 and not self.transitions[i][-1][4]:
                        self.transitions[i][-1][3] = first_value.to(torch.device("cpu"))
                    self.transitions[i].append(
                        [reward, prompt_tens.to(torch.device("cpu")), action_tens.to(torch.device("cpu")),
                         torch.tensor(0, dtype=val.dtype), False, val.to("cpu"), logp.to("cpu")])

            # removes non ascii chars and \
            action = clean_str(action)
            actionList.append(action)
        return actionList
