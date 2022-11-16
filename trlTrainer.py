# sepparates value head from the model so that it can be a drop in transformer where output_hidden_states is an option
from pathlib import Path
import numpy as np
import torch.nn.functional as F
from torch.optim import Adam
import torch
import collections
import time
import random
from argparse import Namespace

from datastructures import RLDataset
from datastructures import ReplayBuffer
from datastructures import LineDataset
from datastructures import LineBuffer
from datastructures import RLDatasetCollator

import pytorch_lightning as pl
import argparse
from collections import OrderedDict, deque
from typing import Tuple, List
import torch.optim as optim
from torch.optim import Optimizer
from deepspeed.ops.adam import DeepSpeedCPUAdam
from torch.utils.data import DataLoader
from torchinfo import summary

from valueHead import ValueHead
from games import VectorPlayer, getEnvs
from agents import VectorNLPAgent

from transformers import DataCollatorForLanguageModeling
from pytorch_lightning.strategies.deepspeed import DeepSpeedStrategy
from transformers.deepspeed import HfDeepSpeedConfig
# from lightning_transformers.utilities.deepspeed import enable_transformers_pretrained_deepspeed_sharding

# from lightning_transformers.task.nlp.language_modeling import LanguageModelingTransformer

from core import (logprobs_from_logits,
                  whiten,
                  clip_by_value,
                  entropy_from_logits,
                  flatten_dict,
                  average_torch_dicts,
                  stats_to_np,
                  stack_dicts,
                  add_suffix,
                  WANDB_PADDING,
                  pad_mask)


# using deepspeed pytorch-lightning


class AdaptiveKLController:
    """
    Adaptive KL controller described in the paper:
    https://arxiv.org/pdf/1909.08593.pdf
    """

    def __init__(self, init_kl_coef, target, horizon):
        self.value = init_kl_coef
        self.target = target
        self.horizon = horizon
        self.kl_list = []

    def update(self, current, n_steps):
        target = self.target
        proportional_error = np.clip(current / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult

    def updateValue(self, v):
        self.value = v


class FixedKLController:
    """Fixed KL controller."""

    def __init__(self, kl_coef):
        self.value = kl_coef
        self.kl_list = []

    def update(self, current, n_steps):
        pass

    def updateValue(self, v):
        pass


class TRLTrainer(pl.LightningModule):
    def __init__(self, model_name=None, **params):
        super().__init__()
        self.save_hyperparameters()

        self.params = self.default_params
        self.params.update(params)
        self.alg_name = self.params["alg_name"]
        self.model_name = model_name

        if self.params['adap_kl_ctrl']:
            self.kl_ctl = AdaptiveKLController(self.params['init_kl_coef'],
                                               self.params['target'],
                                               self.params['horizon'])
        else:
            self.kl_ctl = FixedKLController(self.params['init_kl_coef'])

        if self.params['adap_kl_ctrl_rew']:
            self.kl_ctl_rew = AdaptiveKLController(self.params['init_kl_coef_rew'],
                                               self.params['target_rew'],
                                               self.params['horizon_rew'])
        else:
            self.kl_ctl_rew = FixedKLController(self.params['init_kl_coef_rew'])
            
        self.all_stats = []
        
        self.saveModelTime = 0
        self.saveStatTime = 0
        self.epoch_time = 0
        self.game_time = 0

    def getDevice(self):
        return self.device

    @property
    def deepspeed_offload(self) -> bool:
        strategy = self.trainer.strategy
        if isinstance(strategy, DeepSpeedStrategy):
            config = strategy.config['zero_optimization']
            return config.get('offload_optimizer') # or config.get('offload_param')
        return False

    @property
    def deepspeed_stage(self):
        strategy = self.trainer.strategy
        if isinstance(strategy, DeepSpeedStrategy):
            config = strategy.config['zero_optimization']
            return config.get('stage')
        return None

    # appears to have no effect on deepspeed checkpoint size
    # def on_save_checkpoint(self, checkpoint):
    #     keyList = list(checkpoint['state_dict'].keys())
    #     for k in keyList:
    #         if "ref_model" in k:
    #             del checkpoint['state_dict'][k]
        # print(checkpoint.keys())
        # print(checkpoint['state_dict'].keys())

    def setup(self, stage=None):
        self.dschf = HfDeepSpeedConfig(self.trainer.strategy.config)
        # enable_transformers_pretrained_deepspeed_sharding(self)
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model_name = self.model_name
        if not hasattr(self, "model"):
            # gpt2 and gpt2-xl
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
        if not hasattr(self, "ref_model"):
            if self.params["reference"]:
                self.ref_model = AutoModelForCausalLM.from_pretrained(model_name)
        if not hasattr(self, "tokenizer"):
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
            self.tokenizer.pad_token = self.tokenizer.unk_token


        # print(self.model.config.torch_dtype)
        # summary(self.model)
        
        if self.trainer.is_global_zero:
            getEnvs()
            print("generated envs")
        torch.distributed.barrier()
        
        if self.params["single_game"]:
            # agent = NLPAgent(buffer, humanTurns=0)
            self.agent = VectorNLPAgent(self.agent_buffer, num_agents=self.params["num_agents"], rank=self.trainer.global_rank,
                                   world_size=self.trainer.world_size, **self.agentKWArgs)
            print("Training")
            self.agent.train()  # Tell the agent it should update its parameters.
            # player = Player(agent, "./games/tw-rewardsDense_goalDetailed.z8", verbose=False)  # Dense rewards game.
            self.player = VectorPlayer(self.agent, "./games/tw-rewardsDense_goalDetailed.z8", verbose=False,
                                  num_agents=self.params["num_agents"],
                                  rank=self.trainer.global_rank, world_size=self.trainer.world_size, **self.playerKWArgs)

        else:
            self.agent = VectorNLPAgent(self.agent_buffer, num_agents=self.params["num_agents"], rank=self.trainer.global_rank,
                                   world_size=self.trainer.world_size, **self.agentKWArgs)
            print("Training on 100 games")
            self.agent.train()  # Tell the agent it should update its parameters.
            self.player = VectorPlayer(self.agent, "./training_games/", verbose=False, num_agents=self.params["num_agents"],
                                  rank=self.trainer.global_rank, world_size=self.trainer.world_size, **self.playerKWArgs)  # Each game will be seen 5 times.
    
    def on_test_epoch_start(self):
        # train on the same data epochs per game times before generating a new set
        game_time = time.time()
        self.runGame()
        self.game_time = time.time() - game_time
        # print("game time ", self.game_time)

        self.epoch_time = time.time()
        # print("rank ", self.trainer.global_rank, " arrived at epoch barrier")
        torch.distributed.barrier()
        
    def on_train_epoch_start(self):
        # train on the same data epochs per game times before generating a new set
        if self.current_epoch % self.params['epochs_per_game'] == 0:
            game_time = time.time()
            self.runGame()
            self.game_time = time.time() - game_time
            # print("game time ", self.game_time)

        self.epoch_time = time.time()
        # print("rank ", self.trainer.global_rank, " arrived at epoch barrier")
        torch.distributed.barrier()

    def compute_rewards(self, scores, logprobs, ref_logprobs):
        """Compute per token rewards from scores and KL-penalty."""
        rewards, non_score_rewards = [], []
        with torch.no_grad():
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

    def forward(self, input_ids, use_cache=False, past_key_values=None, outputVals=False, outputRef=False, attention_mask=None, outputLogits=True):
        return None

    def batched_forward_pass(self, queries, responses, outputLogits=True, outputVals=True, outputRef=True):
        return None

    def record_step_stats(self, **data):
        """Record training step statistics."""
        # kl_list = [logprobs - ref_logprobs for logprobs, ref_logprobs in zip(data['logprobs'], data['ref_logprobs'])]
        kl_list = self.kl_ctl.kl_list
        mean_kl = torch.mean(torch.stack([torch.sum(kl) for kl in kl_list]))
        kl_list_rew = self.kl_ctl_rew.kl_list
        mean_kl_rew = torch.mean(torch.stack([torch.sum(kl) for kl in kl_list_rew]))

        mean_entropy = torch.mean(torch.stack([torch.sum(-log_probs) for log_probs in data['logprobs']]))
        mean_non_score_reward = torch.mean(
            torch.stack([torch.sum(non_score_reward).to("cpu") for non_score_reward in data['non_score_reward']]))
        stats = {
            'objective/kl': mean_kl,
            'objective/kl_dist': kl_list,
            'objective/kl_rew': mean_kl_rew,
            'objective/kl_dist_rew': kl_list_rew,
            # too big, makes file messy
            # 'objective/logprobs': data['logprobs'],
            # 'objective/ref_logprobs': data['ref_logprobs'],
            'objective/kl_coef': self.kl_ctl.value,
            'objective/kl_coef_rew': self.kl_ctl_rew.value,
            'objective/vf_coef': self.params['vf_coef'],
            'objective/entropy': mean_entropy,
            f'{self.alg_name}/mean_non_score_reward': mean_non_score_reward,
        }
                   
        self.kl_ctl.kl_list = []
        

        for k, v in data['train_stats'].items():
            # print(k, v)
            stats[f'{self.alg_name}/{k}'] = torch.mean(v, axis=0)
        return stats
    
def getTrainer(**kwargs):
    SAVE_FREQUENCY = 4
    from pytorch_lightning.strategies.deepspeed import DeepSpeedStrategy
    from pytorch_lightning.callbacks import ModelCheckpoint
    checkpoint_callback = ModelCheckpoint(dirpath="checkpoints/", filename="ppo-{epoch:02d}",
                                          every_n_epochs=SAVE_FREQUENCY, save_weights_only=True)
    trainer = pl.Trainer(
        enable_checkpointing=True,
        logger=False,
        accelerator='gpu', devices=8,
        max_epochs=500,
        precision=16,
        strategy=DeepSpeedStrategy(
            stage=3,
            offload_optimizer=True,
            offload_parameters=True
        ),
        callbacks=[checkpoint_callback]
    )
    return trainer
    