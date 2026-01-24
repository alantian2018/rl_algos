import sys
import os
#sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import gymnasium
import torch
import torch.nn as nn
from torch.distributions import Categorical
import draccus
from dataclasses import dataclass, replace
from functools import partial
from datetime import datetime
import termcolor
import numpy as np


from ppo import PPOConfig, PPO, CarRacingActor, CarRacingCritic

def make_carracing_env(render_mode=None):
    env = gymnasium.make("CarRacing-v3", continuous=False, render_mode=render_mode)
    return env


@dataclass 
class CarRacingConfig(PPOConfig):
    obs_dim: tuple = (96, 96, 3)
    act_dim: int = 5
    actor_hidden_size: int = 128
    critic_hidden_size: int = 128
    
    T: int = 2048
    gamma: float = 0.99
    gae_lambda: float = 0.95
    epsilon: float = 0.1
    actor_lr: float = 3e-4
    critic_lr: float = 1e-3
    minibatch_size: int = 256
    epochs_per_batch: int = 4
    entropy_coefficient: float = 0.05
    
    total_gradient_steps: int = 500_000
    device: str = "cuda" if torch.cuda.is_available() \
                else 'mps' if torch.backends.mps.is_available() \
                else "cpu"
    
    wandb_project: str = "ppo-carracing"
    wandb_run_name: str = None
    video_log_freq: int = 2_000
    
    save_dir: str = "checkpoints/carracing"
    save_freq: int = None
    
    load_from_existing_checkpoint: bool = False
    load_checkpoint_path: str = None


@draccus.wrap()
def main(config: CarRacingConfig):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config = replace(config, save_dir=f"{config.save_dir}/{timestamp}")

    env = make_carracing_env()

    in_channels = config.obs_dim[2]
    height = config.obs_dim[0]
    width = config.obs_dim[1]

    actor = CarRacingActor(
        in_channels=in_channels,
        height=height,
        width=width,
        act_dim=config.act_dim,
        hidden_size=config.actor_hidden_size,
    )
    critic = CarRacingCritic(
        in_channels=in_channels,
        height=height,
        width=width,
        hidden_size=config.critic_hidden_size,
    )
  
    if config.load_from_existing_checkpoint:
        actor.load_state_dict(torch.load(config.load_checkpoint_path)['actor_state_dict'])
        critic.load_state_dict(torch.load(config.load_checkpoint_path)['critic_state_dict'])
        print(termcolor.colored(f"Loaded checkpoint from {config.load_checkpoint_path}", 'green'))
    
    ppo = PPO(config, env, actor, critic, make_env=make_carracing_env)
    ppo.run_batch(config.total_gradient_steps)


if __name__ == "__main__":
    main()
