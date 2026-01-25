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
    exp_name: str = "carracing"

    obs_dim: tuple = (96,96,3)
    act_dim: int = 5

    actor_hidden_size: int = 128
    critic_hidden_size: int = 128  
    # lower epsilon  
    epsilon: float = 0.1

    entropy_coefficient: float = 0.05
    entropy_decay: bool = True
    
    total_gradient_steps: int = 500_000
    
    wandb_entity: str = 'apcsc'

    video_log_freq: int = 2_000
    
    save_freq: int = 10_000

    device: str = 'cpu'


@draccus.wrap()
def main(config: CarRacingConfig):
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
     
    ppo = PPO(config, env, actor, critic, make_env=make_carracing_env)
    ppo.run_batch(config.total_gradient_steps)


if __name__ == "__main__":
    main()
