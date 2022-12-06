from time import time
import pytorch_lightning as pl
from pathlib import Path
from torchinfo import summary
from pytorch_lightning.strategies.deepspeed import DeepSpeedStrategy
from pytorch_lightning.callbacks import ModelCheckpoint

from pytorch_lightning.utilities.deepspeed import (
        convert_zero_checkpoint_to_fp32_state_dict
    )

from ppo import PPOTrainer
from rejectionSample import RejectionTuner
from decisionTrans import DecisionTuner

if __name__ == "__main__":
    import argparse
    
    pl.seed_everything(42)

    model_name = 'gpt2'
    # model_name = 'EleutherAI/gpt-j-6B'
    # model_name = 'EleutherAI/gpt-neo-1.3B'
    # model_name = "EleutherAI/gpt-neox-20b"
    single_game = False
    
    Path("stats").mkdir(parents=True, exist_ok=True)
    Path("checkpoints").mkdir(parents=True, exist_ok=True)
    
    trainer = pl.Trainer(
        enable_checkpointing=True,
        logger=False,
        accelerator='gpu', devices=1,
        max_epochs=4,
        precision=16,
        strategy=DeepSpeedStrategy(
            stage=3,
            offload_optimizer=True,
            offload_parameters=True 
        )
    )
    # Lightning deepspeed has saved a directory instead of a file
    save_path = "checkpoints/ppo-epoch=02.ckpt" 
    # output_path = "checkpoints/ppo.pt" 
    # convert_zero_checkpoint_to_fp32_state_dict(save_path, output_path)
    UPDATE_FREQUENCY = 64
    FORWARD_BATCH = 8
    LOG_FREQUENCY = 1
    SAVE_FREQUENCY = 1
    NUM_AGENTS = 8
    
    UPDATE_FREQUENCY = max(UPDATE_FREQUENCY // trainer.world_size, 1)
    FORWARD_BATCH = max(FORWARD_BATCH // trainer.world_size, 1)
    NUM_AGENTS = max(NUM_AGENTS // trainer.world_size, 1)
    
    ppo_config = {'batch_size': UPDATE_FREQUENCY, 'forward_batch_size': FORWARD_BATCH, "log_freq": LOG_FREQUENCY, "num_agents": NUM_AGENTS, "single_game": single_game}
    ppo_trainer = PPOTrainer(model_name=model_name, **ppo_config)
    
    trainer.test(model=ppo_trainer, ckpt_path=save_path)
    # summary(ppo_trainer.model)

    # trainer.test(ppo_trainer)