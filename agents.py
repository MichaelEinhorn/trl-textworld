import numpy as np

import re
from typing import List, Mapping, Any, Optional
from collections import defaultdict
from datastructures import buffer

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
    UPDATE_FREQUENCY = 10
    LOG_FREQUENCY = 10
    GAMMA = 0.9
    MEMORY_LEN = 3
    
    def __init__(self, model, model_ref, tokenizer, humanTurns=0) -> None:
        self._initialized = False
        self._epsiode_has_started = False
       
        self.memory = buffer(self.MEMORY_LEN)

        self.model = model
        self.model_ref = model_ref
        self.tokenizer = tokenizer

        self.humanTurns = humanTurns
        self.humanTurnsRem = self.humanTurns
        
        if model_ref is not None:
          # initialize trainer
          ppo_config = {'batch_size': self.UPDATE_FREQUENCY, 'forward_batch_size': 1}
          self.ppo_trainer = PPOTrainer(self.model, self.model_ref, self.tokenizer, **ppo_config)
          self.valueHead = self.ppo_trainer.valueHead

        if device == "cuda":
          self.model.cuda()
          self.tokenizer.cuda()
          
        self.mode = "test"

        self.clearTextWorldArt = True
    
    def train(self):
        if self.model_ref is None:
          raise NotImplementedError
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
            _, _, _, R = self.transitions[-1]
        else:
            R = last_values.data
        for t in reversed(range(len(self.transitions))):
            rewards, _, _, values = self.transitions[t]
            R = rewards + self.GAMMA * R
            adv = R - values
            returns.append(R)
            advantages.append(adv)
            
        return returns[::-1], advantages[::-1]
    
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
            if self.no_train_step % self.UPDATE_FREQUENCY == 0:
                # get discounted returns and advantages across multiple actions. Currently not used
                returns, advantages = self._discount_rewards()

                query = []
                response = []
                rewardList = []

                for t in reversed(range(len(self.transitions))):
                  rew, prompt, action, values = self.transitions[t]

                  query.append(prompt[0])
                  response.append(action[0])
                  rewardList.append(rew)

                train_stats = self.ppo_trainer.step(query, response, rewardList)

                if self.no_train_step % self.LOG_FREQUENCY == 0:
                  print(train_stats)

                self.transitions = []
        
    def act(self, obs: str, score: int, done: bool, infos: Mapping[str, Any]) -> Optional[str]:
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
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=True, return_tensors="pt").to(device)
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
        
        while new_tokens == 0 or (new_tokens < 20 and "\n" not in self.tokenizer.decode(next_token) 
                                  and next_token != self.tokenizer.eos_token):
          # run model
          with torch.no_grad():
                # get logits, only get last value
                if cache is None:
                    lmOut = self.model(input_ids, output_hidden_states=True, use_cache=True)
                else:
                    lmOut = self.model(input_ids[:,-1:], output_hidden_states=True, use_cache=True, past_key_values=cache)
                # print(dir(lmOut))
                logits, hidden_state, cache = lmOut.logits, lmOut.hidden_states[-1], lmOut.past_key_values
                
                # hidden_state = hidden_state[-1]
                values = self.valueHead(hidden_state)
                
                next_token_logits = logits[:, -1, :]
                next_token_logits = top_k_top_p_filtering(next_token_logits, top_k=0, top_p=1)
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
                input_ids = torch.cat([input_ids, next_token.unsqueeze(-1)], dim=-1)
                
                new_tokens += 1

        del cache
        del hidden_state
        
        action_tens = input_ids[:,-new_tokens:].to("cpu")
        prompt_tens = input_ids[:,:-new_tokens].to("cpu")
        
        del input_ids

        action = self.tokenizer.decode(action_tens[0,:])

        print("action")
        print(action)
        
        # only grab last token
        values = values[0,-1,0]
        print("last token value in action")
        print(values)

        self.memory.append(input_ + action)
        
        if self.mode == "train" and not done:
            self.transitions.append([None, prompt_tens, action_tens, values])  # Reward will be set on the next call
            
        if done:
            self.last_score = 0  # Will be starting a new episode. Reset the last score.
            self.memory.clear()
            self.clearTextWorldArt = True
            self.humanTurnsRem = self.humanTurns
            
        return action
    