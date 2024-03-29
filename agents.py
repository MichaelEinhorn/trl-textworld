import numpy as np
from pathlib import Path
from typing import List, Mapping, Any, Optional
from collections import defaultdict
from collections import deque

import textworld
import textworld.gym
from textworld import EnvInfos

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from transformers import top_k_top_p_filtering
from torch.nn import Identity

from core import logprobs_from_logits, qidx_from_qs

import string
import re

import random


def clean_str(s):
    allowed_chars = " " + string.ascii_letters + string.digits + ".?'-"
    s = re.sub(f"[^{allowed_chars}]", "", s)
    # remove endoftext potentially causing EOFError
    s = s.replace("endoftext", "")
    return s


# does not match escape chars like \n
def hasLettersOrNum(s):
    allowed_chars = string.ascii_letters + string.digits
    return bool(re.search(f'[{allowed_chars}]', s))


def printFile(s, i, epoch, rank, num_agents):
    with open(f"trajectories/epoch_{epoch}_agent_{i + rank * num_agents}.txt", "a") as myfile:
        myfile.write(s + "\n")


class RandomAgent(textworld.gym.Agent):
    """ Agent that randomly selects a command from the admissible ones. """

    def __init__(self, seed=2061630618, num_agents=1, **kwargs):
        self.seed = seed
        self.rng = np.random.RandomState(self.seed)
        self.num_agents=num_agents

    def infos_to_request(self) -> textworld.EnvInfos:
        return textworld.EnvInfos(admissible_commands=True)

    def act(self, obs: str, score: int, done: bool, infos: Mapping[str, Any]):
        actionList = []
        for i in range(self.num_agents):
            actionList.append(self.rng.choice(infos["admissible_commands"][i]))
        return actionList


class HumanAgent(textworld.gym.Agent):
    """ Agent that randomly selects a command from the admissible ones. """

    def __init__(self, seed=2061630618, num_agents=1, MEMORY_LEN=1, **kwargs):
        self.seed = seed
        self.rng = np.random.RandomState(self.seed)
        self.num_agents=num_agents

        self.MEMORY_LEN = MEMORY_LEN
        self.memory = Memory(MEMORY_LEN=MEMORY_LEN, num_agents=num_agents)
        self.lastActionInfos = [None for i in range(self.num_agents)]

        # textworld is adding an extra turn at the end of the game
        self.gameResettingExtraTurn = [False for i in range(self.num_agents)]

    @property
    def infos_to_request(self) -> EnvInfos:
        return EnvInfos(description=True, inventory=True, admissible_commands=True,
                        won=True, lost=True, command_templates=True,
                        entities=True, feedback=True, game=True, intermediate_reward=True,
                        last_action=True, last_command=True, location=True, moves=True, objective=True, policy_commands=True, score=True, verbs=True)
    
    def report(self, rewards, score, done, infos, **kwargs):
        print("report --------------------------------------------------------------\n\n\n")
        print(rewards, infos["won"], infos["lost"], done)
        for i in range(self.num_agents):
            if self.gameResettingExtraTurn[i]:
                self.gameResettingExtraTurn[i] = False
                continue
            print("invalid ", i, " ", infos["last_action"][i] is None or infos["last_action"][i] == self.lastActionInfos[i])
            if done[i]:
                print("done clearing mem")
                self.memory.clear(i)
        self.lastActionInfos = infos["last_action"]

    def act(self, observation, score, done, infos, lightmodel=None, **kwargs):
        promptList = []
        inputList = []
        actionList = []
        # infos is dict of lists
        for i in range(self.num_agents):
            if infos["won"][i] or infos["lost"][i]:
                actionList.append("placeholder")
                self.gameResettingExtraTurn[i] = True
                continue

            obs = observation[i]
            # Build agent's observation: feedback + look + inventory.
            prompt, input_ = self.memory.getFormattedPrompt(i, obs, infos)

            print("info --------------------------------------------------------------")
            for k,v in infos.items():
                print(k, v)
            print("prompt --------------------------------------------------------------")
            print(prompt)

            promptList.append(prompt)
            inputList.append(input_)

            action = input()
            actionList.append(action)
            self.memory.append(i, input_, action)
        return actionList


# retains memory and formats prompts for the Agent
class Memory:
    def __init__(self, MEMORY_LEN=1, num_agents=1, objective=True):
        self.MEMORY_LEN = MEMORY_LEN
        self.num_agents = num_agents
        self.objective = objective
        self.memory = [deque(maxlen=self.MEMORY_LEN) for i in range(self.num_agents)]

    def clear(self, i):
        self.memory[i].clear()

    # removes examine actions
    def filterAdmCmd(self, cmd_list, i, infos, examine=False, shuffle=False):
        out = []
        for cmd in cmd_list:
            if not "examine" in cmd or examine:
                out.append(cmd)
        if shuffle:
            random.shuffle(out)
        return out

    def clearExcessPunctuation(self, obs):
        # clear textworld opening art
        seq = "Welcome to TextWorld!"
        if seq in obs:
            obs = obs[obs.rindex(seq) + len(seq):]
        seq = "$$$$$$$"
        if seq in obs:
            obs = obs[obs.rindex(seq) + len(seq):]

        # removes headlines such as -= Bedroom =- Less weird punctuation is probably better for the model
        obs = re.sub(r"-=.*?=-", "", obs)

        # clear double new lines
        while "\n\n" in obs:
            obs = obs.replace("\n\n", "\n")

        # clear newlines and spaces at the top
        if obs.startswith("\n") or obs.startswith(" "):
            idx = re.search("[^\n ]", obs).start()
            obs = obs[idx:]

        return obs

    # obs is just the observation for the current index
    # infos is a dict with every index
    def getFormattedPrompt(self, i, obs, infos):
        obs = self.clearExcessPunctuation(obs)

        pastStates = ""
        for mem in self.memory[i]:
            pastStates = pastStates + mem + "\n"
        admissible_commands_str = "Your possible actions are "

        adm_cmd_list = self.filterAdmCmd(infos["admissible_commands"][i], i, infos, examine=False, shuffle=False)

        for cmd_idx in range(len(adm_cmd_list) - 1):
            adm_cmd = adm_cmd_list[cmd_idx]
            admissible_commands_str += adm_cmd + ", "
        adm_cmd = adm_cmd_list[len(adm_cmd_list) - 1]
        admissible_commands_str += "or " + adm_cmd + "."
        # infos["description"][i]
        inventoryStr = infos["inventory"][i]
        if "nothing" in inventoryStr:
            inventoryStr = ""
        else:
            inventoryStr = inventoryStr + " "
        # input_ = "{}{}{} You can only choose a single action. You take the action to ".format(obs, inventoryStr,
        #                                                    admissible_commands_str)
        input_ = f"{obs}{inventoryStr}{admissible_commands_str} You choose the action to "
        prompt = pastStates + input_

        obj = self.clearExcessPunctuation(infos["objective"][i])
        if self.objective and obj not in prompt:
            prompt = obj + "\n" + prompt

        return prompt, input_

    def append(self, i, input_, action):
        if not input_.endswith(" ") and not action.startswith(" "):
           self.memory[i].append(input_ + " " + action)
        else:
            self.memory[i].append(input_ + action)


class VectorNLPAgent():
    """ Hugging Face Transformer Agent """

    def __init__(self, buffer, num_agents=1, rank=0, world_size=1, useUnfinished=True, GAMMA=0.8, MEMORY_LEN=1, **kwargs) -> None:
        self._initialized = False
        self._epsiode_has_started = False
        self.num_agents = num_agents

        self.GAMMA = GAMMA
        self.MEMORY_LEN = MEMORY_LEN

        self.memory = Memory(MEMORY_LEN=MEMORY_LEN, num_agents=num_agents)
        self.transitionBuffer = buffer

        self.mode = "test"
        self.transitions = [[] for i in range(self.num_agents)]
        # self.clearTextWorldArt = [True for i in range(self.num_agents)]

        self.useUnfinished = useUnfinished

        self.rewValStat = []

        Path("trajectories").mkdir(parents=True, exist_ok=True)

        self.rank = rank
        self.world_size = world_size
        self.epoch = 0
        
        self.rng = None
        # init later to get device
        # self.rng = torch.Generator()
        # self.rng.manual_seed(self.rng.initial_seed() + rank)

        # textworld is adding an extra turn at the end of the game
        self.gameResettingExtraTurn = [False for i in range(self.num_agents)]

    def train(self):
        self.mode = "train"
        self.stats = {"max": defaultdict(list), "mean": defaultdict(list)}
        self.transitions = [[] for i in range(self.num_agents)]
        # self.model.reset_hidden(1)
        self.no_train_step = [0 for i in range(self.num_agents)]
        # self.clearTextWorldArt = [True for i in range(self.num_agents)]

        for i in range(self.num_agents):
            self.memory.clear(i)

    def test(self):
        # self.clearTextWorldArt = [True for i in range(self.num_agents)]
        self.mode = "test"

    @property
    def infos_to_request(self) -> EnvInfos:
        return EnvInfos(description=True, inventory=True, admissible_commands=True,
                        won=True, lost=True, last_action=True, objective=True)

    @torch.no_grad()
    def _discount_rewards(self):
        returnsList, advantagesList, finishedList = [], [], []
        for i in range(self.num_agents):
            trans = self.transitions[i]
            R = 0
            returns, advantages, finished = [], [], []
            fin = False
            for t in reversed(range(len(trans))):
                # dont discount between episodes
                # rewards, _, _, values, done, _, _ = trans[t]
                trans_dict = trans[t]
                reward = trans_dict["reward"]
                done = trans_dict["done"]
                first_value = trans_dict["value"][0]
                if done:
                    R = 0
                    fin=True
                R = reward + self.GAMMA * R
                adv = R - first_value
                returns.append(R)
                advantages.append(adv)
                finished.append(fin)

            returnsList.append(returns[::-1])
            advantagesList.append(advantages[::-1])
            finishedList.append(finished[::-1])
        return returnsList, advantagesList, finishedList

    @torch.no_grad()
    def fillBuffer(self, **kwargs):
        # get discounted returns and advantages across multiple actions. Currently not used
        returnsList, advantagesList, finishedList = self._discount_rewards()
        unfinCount = [0 for i in range(self.num_agents)]
        for i in range(self.num_agents):
            trans = self.transitions[i]
            returns = returnsList[i]
            advantages = advantagesList[i]
            finished = finishedList[i]
            for t in reversed(range(len(trans))):
                trans_dict = trans[t]
                # rew, prompt, action, next_value, done, value, logprob = trans[t]
                trans_dict["ret_cross"] = returns[t]
                trans_dict["adv_cross"] = advantages[t]
                fin = finished[t]
                # returns and advantages across multiple actions
                # ppo trainer computes returns and advantages across tokens within an action
                # not sure if these are usefull
                # exper = (rew, prompt[0], action[0], next_value, ret_cross, adv_cross, value[0], logprob[0])
                if fin or self.useUnfinished:
                    self.transitionBuffer.append(trans_dict)
                else:
                    unfinCount[i] += 1
        if self.useUnfinished:
            self.transitions = [[] for i in range(self.num_agents)]
        else:
            self.transitions = [self.transitions[i][-unfinCount[i]:] for i in range(self.num_agents)]

    # fill in results from action and train if time
    @torch.no_grad()
    def report(self, rewards, score, done, infos, **kwargs):
        exTurn = False
        if "exTurn" in kwargs:
            if kwargs["exTurn"] == 1:
                exTurn = True

        for i in range(self.num_agents):
            if self.mode == "train":
                if self.gameResettingExtraTurn[i]:
                    self.gameResettingExtraTurn[i] = False
                    continue

                printFile(f"reward: {rewards[i]}", i, self.epoch, self.rank, self.num_agents)

                if not exTurn:
                    self.transitions[i][-1]["reward"] = rewards[i]  # Update reward information. Was initialized as none

                if not exTurn:
                    self.no_train_step[i] += 1

                self.stats["max"]["score"].append(score)

                # notification of done does not have a new state
                if done[i]:
                    self.memory.clear(i)
                    # self.clearTextWorldArt[i] = True
                    # mark last transition of episode
                    # if not exTurn:
                    if len(self.transitions[i]) != 0:
                        self.transitions[i][-1]["done"] = True

    @torch.no_grad()
    def act(self, observation, score, done, infos, lightmodel, **kwargs):
        promptList = []
        inputList = []
        epoch = lightmodel.current_epoch
        self.epoch = epoch
        
        if self.rng is None:
            self.rng = torch.Generator(device=lightmodel.device)
            self.rng.manual_seed(2061630618 + self.rank)

        # infos is dict of lists
        for i in range(self.num_agents):
            if infos["won"][i] or infos["lost"][i]:
                printFile("placeholder for dummy turn", i, epoch, self.rank, self.num_agents)
                promptList.append("placeholder")
                inputList.append("placeholder")
                self.gameResettingExtraTurn[i] = True
                continue

            obs = observation[i]

            # Build agent's observation: feedback + look + inventory.
            prompt, input_ = self.memory.getFormattedPrompt(i, obs, infos)

            printFile(input_, i, epoch, self.rank, self.num_agents)

            promptList.append(prompt)
            inputList.append(input_)

        # does a random admissable action and adds to memory. Does not create a transition in experience buffer
        actionList = []
        if "exTurn" in kwargs:
            # print(kwargs["exTurn"])
            if kwargs["exTurn"] == 1:
                for i in range(self.num_agents):
                    if infos["won"][i] or infos["lost"][i]:
                        actionList.append("placeholder")
                        continue

                    commands = infos["admissible_commands"][i]
                    commands = self.memory.filterAdmCmd(commands, i, infos, examine=False)
                    idx = np.random.choice(len(commands))
                    action = commands[idx] + "."
                    input_ = inputList[i]

                    self.memory.append(i, input_, action)

                    printFile("example turn", i, epoch, self.rank, self.num_agents)
                    printFile(action, i, epoch, self.rank, self.num_agents)
                    actionList.append(action)
                return actionList

        model_input = lightmodel.tokenizer(promptList, add_special_tokens=True, return_tensors="pt", padding=True,
                                           return_attention_mask=True)
        input_ids = model_input["input_ids"]
        attention_mask = model_input["attention_mask"]
        new_tokens = 0
        next_token = None

        values = 0

        cache = None
        # 0 is done, 1 is continuing
        finished = torch.ones((self.num_agents,), device=lightmodel.getDevice())
        # if the model has outputted a letter or number
        # will not stop until after this has been flipped
        startedLetter = torch.zeros((self.num_agents,), device=lightmodel.getDevice())
        maxLen = 20
        genLengths = [maxLen for i in range(self.num_agents)]

        logprobList = []
        valuesList = []
        # printFile("start generation loop", 0, epoch, self.rank, self.num_agents)
        # while new_tokens == 0 or (new_tokens < maxLen and torch.sum(finished) > 0):
        # generate all 20 tokens, because gpus need to call forward pass in sync for deepspeed
        while new_tokens == 0 or (new_tokens < maxLen):
            # run model
            with torch.no_grad():
                # get logits, only get last value
                input_ids = input_ids.to(lightmodel.getDevice())
                attention_mask = attention_mask.to(lightmodel.getDevice())
                # input_ids = input_ids.to(lightmodel.model.device)
                # printFile("new tokens " + str(new_tokens), 0, epoch, self.rank, self.num_agents)
                if cache is None:
                    lmout = lightmodel(input_ids, use_cache=True, outputVals=True, attention_mask=attention_mask)
                else:
                    # print("cache ", input_ids[:, -1:].shape, len(cache), attention_mask.shape)
                    lmout = lightmodel(input_ids[:, -1:], outputVals=True, use_cache=True,
                                       past_key_values=cache, attention_mask=attention_mask)
                # printFile("finished forward", 0, epoch, self.rank, self.num_agents)

                logits, cache, values = lmout["logits"], lmout["cache"], lmout["values"]

                next_token_logits = logits[:, -1, :]
                # next_token_logits = top_k_top_p_filtering(next_token_logits, top_k=0, top_p=1)
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1, generator=self.rng).squeeze(1)
                input_ids = torch.cat([input_ids, next_token.unsqueeze(-1)], dim=-1)

                # keep last token in att
                # breaks indexing if this is after the finished check
                attention_mask = torch.cat([attention_mask, finished.unsqueeze(-1)], dim=-1)

                for i in range(self.num_agents):
                    # only if not finished
                    if finished[i] == 1:
                        decodedToken = lightmodel.tokenizer.decode(next_token[i])
                        if startedLetter[i] == 0 and hasLettersOrNum(decodedToken):
                            startedLetter[i] = 1

                        stopCond = False
                        stopCond = stopCond or decodedToken in "\n.,?!:;"
                        stopCond = stopCond or next_token[i] == lightmodel.tokenizer.eos_token

                        stopCond = stopCond and startedLetter[i] == 1

                        if stopCond:
                            finished[i] = 0
                            if next_token[i] == lightmodel.tokenizer.eos_token or "\n" in decodedToken:
                                genLengths[i] = new_tokens
                            else:
                                genLengths[i] = new_tokens + 1
                            

                logprob = logprobs_from_logits(logits[:, -1:, :], next_token.unsqueeze(-1))
                logprobList.append(logprob)
                valuesList.append(values[:, -1, :])

                new_tokens += 1

                # printFile("next token " + str(next_token), 0, epoch, self.rank, self.num_agents)

        # print("att shape ", input_ids.shape, attention_mask.shape)
        # print(logprobList)
        # print(valuesList)
        logprob = torch.stack(logprobList, dim=1)
        logprob = logprob.squeeze(2)
        values = torch.stack(valuesList, dim=1)

        # print("val logp ", values.shape, logprob.shape)

        for i in range(self.num_agents):
            if infos["won"][i] or infos["lost"][i]:
                actionList.append("placeholder")
                continue
            # printFile("post process", i, epoch, self.rank, self.num_agents)

            inp = input_ids[i:i + 1]
            att = attention_mask[i]

            # first and last nonzero index
            start = torch.arange(att.shape[0], 0, -1, device=lightmodel.getDevice())
            start = torch.argmax(start * att)
            end = torch.arange(att.shape[0], device=lightmodel.getDevice())
            end = torch.argmax(end * att)
            # print("att ", i, att, start, end, genLengths[i])
            # printFile("genlen, start, end", i, epoch)
            # printFile(str(genLengths[i]) + ", " + str(start) + ", "  + str(end), i, epoch)
            
            # remove left and right pad
            inp = inp[:, start:end + 1]
            # split action and prompt
            action_tens = inp[:, -genLengths[i]:]
            prompt_tens = inp[:, :-genLengths[i]]
            # print("inp act prompt shape ", inp.shape, action_tens.shape, prompt_tens.shape, i)

            action = lightmodel.tokenizer.decode(action_tens[0, :])
            input_ = inputList[i]

            # doesn't need shifting since input ids is already 1 longer than logprob
            logp = logprob[i:i + 1, :genLengths[i]]

            # if i == 0:
            #     print("action")
            #     print(action)
            printFile(clean_str(action), i, epoch, self.rank, self.num_agents)
            if action != clean_str(action):
                printFile("uncleaned action: " + action, i, epoch, self.rank, self.num_agents)

            # doesn't need shifting since input ids is already 1 longer than values
            val = values[i:i + 1, :genLengths[i]]
            # print(values.shape, val.shape, i)
            
            # if i == 0:
            #     print("first value in action", first_value)
            #     # print(value)
            # printFile("first value in action " + str(first_value.item()), i, epoch, self.rank, self.num_agents)
            printFile("token values " + str(val.tolist()), i, epoch, self.rank, self.num_agents)
            printFile("action probability " + str(torch.exp(torch.sum(logp)).item()), i, epoch, self.rank, self.num_agents)
            # only grab last token
            # value = values[i, genLengths[i] - 1, 0]
            clean_action = clean_str(action)

            self.memory.append(i, input_, clean_action)

            # Reward will be set on report
            # fill next value spot in transitions
            first_value = val[0, 0, 0]
            average_value = torch.mean(val)
            # previous transition
            if len(self.transitions[i]) != 0 and not self.transitions[i][-1]["done"]:
                self.transitions[i][-1]["first_value_next"] = first_value.to(torch.device("cpu"))
                self.transitions[i][-1]["average_value_next"] = average_value.to(torch.device("cpu"))
                
            self.addTransition(i, prompt_tens=prompt_tens, action_tens=action_tens, logp=logp, val=val)

            # removes non ascii chars and \
            actionList.append(clean_action)
        return actionList

    @torch.no_grad()
    def addTransition(self, i, reward=None, prompt_tens=None, action_tens=None, done=False, val=torch.tensor([[[0]]]), logp=torch.tensor([[[0]]])):
        self.transitions[i].append(
                {"reward": reward, "prompt_tens":prompt_tens.to(torch.device("cpu"))[0], "action_tens":action_tens.to(torch.device("cpu"))[0],
                "first_value_next":torch.tensor(0, dtype=val.dtype), "average_value_next":torch.tensor(0, dtype=val.dtype),
                "done":done, "value":val.to("cpu")[0], "logp":logp.to("cpu")[0]})


class DecisionAgent(VectorNLPAgent):
    def __init__(self, buffer, num_agents=1, rank=0, world_size=1, useUnfinished=True, GAMMA=0.8, MEMORY_LEN=1, **kwargs) -> None:
        super().__init__(buffer, num_agents=num_agents, rank=rank, world_size=world_size, useUnfinishe=useUnfinished, GAMMA=GAMMA, MEMORY_LEN=MEMORY_LEN, **kwargs)

    @torch.no_grad()
    def act(self, observation, score, done, infos, lightmodel, **kwargs):
        promptList = []
        inputList = []
        epoch = lightmodel.current_epoch
        self.epoch = epoch
        
        if self.rng is None:
            self.rng = torch.Generator(device=lightmodel.device)
            self.rng.manual_seed(2061630618 + self.rank)

        # infos is dict of lists
        for i in range(self.num_agents):
            if infos["won"][i] or infos["lost"][i]:
                printFile("placeholder for dummy turn", i, epoch, self.rank, self.num_agents)
                promptList.append("placeholder")
                inputList.append("placeholder")
                self.gameResettingExtraTurn[i] = True
                continue

            obs = observation[i]

            # Build agent's observation: feedback + look + inventory.
            prompt, input_ = self.memory.getFormattedPrompt(i, obs, infos)

            printFile(input_, i, epoch, self.rank, self.num_agents)

            promptList.append(prompt)
            inputList.append(input_)

        actionList = []
        for i in range(self.num_agents):
            if infos["won"][i] or infos["lost"][i]:
                actionList.append("placeholder")
                continue

            commands = infos["admissible_commands"][i]
            commands = self.memory.filterAdmCmd(commands, i, infos, examine=False)
            idx = np.random.choice(len(commands))
            action = commands[idx] + "."
            input_ = inputList[i]

            self.memory.append(i, input_, action)

            printFile(action, i, epoch, self.rank, self.num_agents)
            actionList.append(action)

        for i in range(self.num_agents):
            if self.mode == "train":
                action = actionList[i]
                prompt = promptList[i]
                # Reward will be set on the next call
                self.addTransition(i, prompt_tens=prompt, action_tens=action)

                return actionList

    @torch.no_grad()
    def addTransition(self, i, reward=None, prompt_tens=None, action_tens=None, done=False, val=[0], logp=[[[0]]]):
        self.transitions[i].append(
                {"reward": reward, "prompt_tens":prompt_tens, "action_tens":action_tens,
                "done":done, "value":val, "logp":logp})

class VectorNLPAgentQ(VectorNLPAgent):
    def __init__(self, buffer, num_agents=1, rank=0, world_size=1, useUnfinished=True, GAMMA=0.8, MEMORY_LEN=1, **kwargs) -> None:
        super().__init__(buffer, num_agents, rank, world_size, useUnfinished, GAMMA, MEMORY_LEN, **kwargs)

    @torch.no_grad()
    def act(self, observation, score, done, infos, lightmodel, **kwargs):
        promptList = []
        inputList = []
        epoch = lightmodel.current_epoch
        self.epoch = epoch
        
        if self.rng is None:
            self.rng = torch.Generator(device=lightmodel.device)
            self.rng.manual_seed(2061630618 + self.rank)

        # infos is dict of lists
        for i in range(self.num_agents):
            if infos["won"][i] or infos["lost"][i]:
                printFile("placeholder for dummy turn", i, epoch, self.rank, self.num_agents)
                promptList.append("placeholder")
                inputList.append("placeholder")
                self.gameResettingExtraTurn[i] = True
                continue

            obs = observation[i]

            # Build agent's observation: feedback + look + inventory.
            prompt, input_ = self.memory.getFormattedPrompt(i, obs, infos)

            printFile(input_, i, epoch, self.rank, self.num_agents)

            promptList.append(prompt)
            inputList.append(input_)

        # does a random admissable action and adds to memory. Does not create a transition in experience buffer
        actionList = []
        if "exTurn" in kwargs:
            # print(kwargs["exTurn"])
            if kwargs["exTurn"] == 1:
                for i in range(self.num_agents):
                    if infos["won"][i] or infos["lost"][i]:
                        actionList.append("placeholder")
                        continue

                    commands = infos["admissible_commands"][i]
                    commands = self.memory.filterAdmCmd(commands, i, infos, examine=False)
                    idx = np.random.choice(len(commands))
                    action = commands[idx] + "."
                    input_ = inputList[i]

                    self.memory.append(i, input_, action)

                    printFile("example turn", i, epoch, self.rank, self.num_agents)
                    printFile(action, i, epoch, self.rank, self.num_agents)
                    actionList.append(action)
                return actionList

        model_input = lightmodel.tokenizer(promptList, add_special_tokens=True, return_tensors="pt", padding=True,
                                           return_attention_mask=True)
        input_ids = model_input["input_ids"]
        attention_mask = model_input["attention_mask"]
        new_tokens = 0
        next_token = None

        values = 0

        cache = None
        # 0 is done, 1 is continuing
        finished = torch.ones((self.num_agents,), device=lightmodel.getDevice())
        # if the model has outputted a letter or number
        # will not stop until after this has been flipped
        startedLetter = torch.zeros((self.num_agents,), device=lightmodel.getDevice())
        maxLen = 20
        genLengths = [maxLen for i in range(self.num_agents)]

        logprobList = []
        valuesList = []
        qidxList = []
        # printFile("start generation loop", 0, epoch, self.rank, self.num_agents)
        # while new_tokens == 0 or (new_tokens < maxLen and torch.sum(finished) > 0):
        # generate all 20 tokens, because gpus need to call forward pass in sync for deepspeed
        while new_tokens == 0 or (new_tokens < maxLen):
            # run model
            with torch.no_grad():
                # get logits, only get last value
                input_ids = input_ids.to(lightmodel.getDevice())
                attention_mask = attention_mask.to(lightmodel.getDevice())
                # input_ids = input_ids.to(lightmodel.model.device)
                # printFile("new tokens " + str(new_tokens), 0, epoch, self.rank, self.num_agents)
                if cache is None:
                    lmout = lightmodel(input_ids, use_cache=True, outputVals=True, outputQs=True, attention_mask=attention_mask)
                else:
                    # print("cache ", input_ids[:, -1:].shape, len(cache), attention_mask.shape)
                    lmout = lightmodel(input_ids[:, -1:], outputVals=True, outputQs=True, use_cache=True,
                                       past_key_values=cache, attention_mask=attention_mask)
                # printFile("finished forward", 0, epoch, self.rank, self.num_agents)

                logits, cache, values = lmout["logits"], lmout["cache"], lmout["values"]
                qs = lmout["qs"]

                next_token_logits = logits[:, -1, :]
                # next_token_logits = top_k_top_p_filtering(next_token_logits, top_k=0, top_p=1)
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1, generator=self.rng).squeeze(1)
                input_ids = torch.cat([input_ids, next_token.unsqueeze(-1)], dim=-1)

                # keep last token in att
                # breaks indexing if this is after the finished check
                attention_mask = torch.cat([attention_mask, finished.unsqueeze(-1)], dim=-1)

                for i in range(self.num_agents):
                    # only if not finished
                    if finished[i] == 1:
                        decodedToken = lightmodel.tokenizer.decode(next_token[i])
                        if startedLetter[i] == 0 and hasLettersOrNum(decodedToken):
                            startedLetter[i] = 1

                        stopCond = False
                        stopCond = stopCond or decodedToken in "\n.,?!:;"
                        stopCond = stopCond or next_token[i] == lightmodel.tokenizer.eos_token

                        stopCond = stopCond and startedLetter[i] == 1

                        if stopCond:
                            finished[i] = 0
                            if next_token[i] == lightmodel.tokenizer.eos_token or "\n" in decodedToken:
                                genLengths[i] = new_tokens
                            else:
                                genLengths[i] = new_tokens + 1
                            

                logprob = logprobs_from_logits(logits[:, -1:, :], next_token.unsqueeze(-1))
                logprobList.append(logprob)
                valuesList.append(values[:, -1, :])
                qidx = [qidx_from_qs(q[:, -1:, :], next_token.unsqueeze(-1)) for q in qs]
                qidx = reduce(torch.minimum, qidx)
                qidxList.append(qidx)

                new_tokens += 1

                # printFile("next token " + str(next_token), 0, epoch, self.rank, self.num_agents)

        # print("att shape ", input_ids.shape, attention_mask.shape)
        # print(logprobList)
        # print(valuesList)
        logprob = torch.stack(logprobList, dim=1)
        logprob = logprob.squeeze(2)
        qidx = torch.stack(qidxList, dim=1)
        qidx = qidx.squeeze(2).detach()
        values = torch.stack(valuesList, dim=1)

        # print("val logp ", values.shape, logprob.shape)

        for i in range(self.num_agents):
            if infos["won"][i] or infos["lost"][i]:
                actionList.append("placeholder")
                continue
            # printFile("post process", i, epoch, self.rank, self.num_agents)

            inp = input_ids[i:i + 1]
            att = attention_mask[i]

            # first and last nonzero index
            start = torch.arange(att.shape[0], 0, -1, device=lightmodel.getDevice())
            start = torch.argmax(start * att)
            end = torch.arange(att.shape[0], device=lightmodel.getDevice())
            end = torch.argmax(end * att)
            # print("att ", i, att, start, end, genLengths[i])
            # printFile("genlen, start, end", i, epoch)
            # printFile(str(genLengths[i]) + ", " + str(start) + ", "  + str(end), i, epoch)
            
            # remove left and right pad
            inp = inp[:, start:end + 1]
            # split action and prompt
            action_tens = inp[:, -genLengths[i]:]
            prompt_tens = inp[:, :-genLengths[i]]
            # print("inp act prompt shape ", inp.shape, action_tens.shape, prompt_tens.shape, i)

            action = lightmodel.tokenizer.decode(action_tens[0, :])
            input_ = inputList[i]

            # doesn't need shifting since input ids is already 1 longer than logprob
            logp = logprob[i:i + 1, :genLengths[i]]

            # if i == 0:
            #     print("action")
            #     print(action)
            printFile(clean_str(action), i, epoch, self.rank, self.num_agents)
            if action != clean_str(action):
                printFile("uncleaned action: " + action, i, epoch, self.rank, self.num_agents)

            # doesn't need shifting since input ids is already 1 longer than values
            val = values[i:i + 1, :genLengths[i]]
            q_val = qidx[i:i + 1, :genLengths[i]]
            # print(values.shape, val.shape, i)
            
            # if i == 0:
            #     print("first value in action", first_value)
            #     # print(value)
            # printFile("first value in action " + str(first_value.item()), i, epoch, self.rank, self.num_agents)
            printFile("token values " + str(val.tolist()), i, epoch, self.rank, self.num_agents)
            printFile("token qs " + str(q_val.tolist()), i, epoch, self.rank, self.num_agents)
            printFile("action probability " + str(torch.exp(torch.sum(logp)).item()), i, epoch, self.rank, self.num_agents)
            # only grab last token
            # value = values[i, genLengths[i] - 1, 0]
            clean_action = clean_str(action)

            self.memory.append(i, input_, clean_action)

            # Reward will be set on report
            # fill next value spot in transitions
            first_value = val[0, 0, 0]
            average_value = torch.mean(val)
            first_q = q_val[0, 0]
            average_q = torch.mean(q_val)
            # previous transition
            if len(self.transitions[i]) != 0 and not self.transitions[i][-1]["done"]:
                self.transitions[i][-1]["first_value_next"] = first_value.to(torch.device("cpu"))
                self.transitions[i][-1]["average_value_next"] = average_value.to(torch.device("cpu"))
                self.transitions[i][-1]["first_q_next"] = first_q.to(torch.device("cpu"))
                self.transitions[i][-1]["average_q_next"] = average_q.to(torch.device("cpu"))
            
            self.addTransition(i, prompt_tens=prompt_tens, action_tens=action_tens, logp=logp, val=val, q=q_val)

            # removes non ascii chars and \
            actionList.append(clean_action)
        return actionList

    @torch.no_grad()
    def addTransition(self, i, reward=None, prompt_tens=None, action_tens=None, done=False, val=torch.tensor([[[0]]]), logp=torch.tensor([[[0]]]), q=None):
        self.transitions[i].append(
                {"reward": reward, "prompt_tens":prompt_tens.to(torch.device("cpu"))[0], "action_tens":action_tens.to(torch.device("cpu"))[0],
                "first_value_next":torch.tensor(0, dtype=val.dtype), "average_value_next":torch.tensor(0, dtype=val.dtype),
                "first_q_next":torch.tensor(0, dtype=val.dtype), "average_q_next":torch.tensor(0, dtype=val.dtype),
                "done":done, "value":val.to("cpu")[0], "logp":logp.to("cpu")[0], "q":q.to("cpu")[0]})