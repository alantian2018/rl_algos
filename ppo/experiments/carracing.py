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


from ppo import PPO, CNNActor, CNNCritic, PPOConfig

def make_carracing_env(render_mode=None):
    env = gymnasium.make("CarRacing-v3", continuous=False, render_mode=render_mode)
    return env


@dataclass 
class CarRacingConfig(PPOConfig):
    obs_dim: tuple = (96, 96, 3)
    act_dim: int = 5
    actor_hidden_size: int = 64
    critic_hidden_size: int = 64
    
    T: int = 1024
    gamma: float = 0.99
    gae_lambda: float = 0.95
    epsilon: float = 0.1
    actor_lr: float = 3e-4
    critic_lr: float = 1e-3
    minibatch_size: int = 256
    epochs_per_batch: int = 4
    entropy_coefficient: float = 0.01
    
    total_gradient_steps: int = 500_000
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    wandb_project: str = "ppo-carracing"
    wandb_run_name: str = None
    video_log_freq: int = 5000
    
    save_dir: str = "checkpoints/carracing"
    save_freq: int = 10_000
    
    load_from_existing_checkpoint: bool = False
    load_checkpoint_path: str = None


@draccus.wrap()
def main(config: CarRacingConfig):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config = replace(config, save_dir=f"{config.save_dir}/{timestamp}")

    print(termcolor.colored("="*100, 'green'))
    print(termcolor.colored("Config: ", 'green'))   
    for key, value in config.__dict__.items():
        print(termcolor.colored(f"  {key}: {value}", 'green'))
    print(termcolor.colored("="*100, 'green'))

    env = make_carracing_env()

    in_channels = config.obs_dim[2]
    height = config.obs_dim[0]
    width = config.obs_dim[1]

    actor = CNNActor(
        in_channels=in_channels,
        height=height,
        width=width,
        act_dim=config.act_dim,
        hidden_size=config.actor_hidden_size,
    )
    critic = CNNCritic(
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
