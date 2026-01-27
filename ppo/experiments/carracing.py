import sys
import os
import gymnasium
import torch
import torch.nn as nn
from torch.distributions import Categorical
import draccus
from dataclasses import dataclass, replace
import numpy as np


from ppo import PPOConfig, PPO, CarRacingActor, CarRacingCritic


def make_carracing_env(render_mode=None, normalize=True):
    env = gymnasium.make("CarRacing-v3", continuous=False, render_mode=render_mode,)
    if not normalize:
        return env

    # Normalize uint8 pixel observations to float32 in range (0, 1)
    class _NormalizeObsWrapper(gymnasium.Wrapper):
        def _normalize(self, obs):
            obs = obs[:84,:,:].copy()
            if isinstance(obs, np.ndarray) and obs.dtype == np.uint8:
                return (obs.astype(np.float32) / 255.0)
            
            return obs

        def reset(self, **kwargs):
            obs, info = self.env.reset(**kwargs)
            return self._normalize(obs), info

        def step(self, action):
            obs, reward, terminated, truncated, info = self.env.step(action)
            return self._normalize(obs), reward, terminated, truncated, info
        
        

    return _NormalizeObsWrapper(env)


@dataclass 
class CarRacingConfig(PPOConfig):
    exp_name: str = "carracing"

    obs_dim: tuple = (96,96,3)
    act_dim: int = 5

    actor_hidden_size: int = 128
    critic_hidden_size: int = 128  

    entropy_coefficient: float = 0.05
    entropy_decay: bool = True
    entropy_decay_steps: bool = 5000
    minibatch_size: int = 64
    T: int = 2048
    epsilon: int = 0.1
    
    total_gradient_steps: int = 500_000
    frame_stack: int = 4
    
    wandb_entity: str = 'apcsc'

    video_log_freq: int = 2_000
    
    save_freq: int = 10_000

    device: str = 'cpu'
  #  path_to_checkpoint: str = 'ppo/checkpoints/carracing/20260126_175202/checkpoint_9999.pt'


@draccus.wrap()
def main(config: CarRacingConfig):
    env = make_carracing_env(normalize=True)
    obs, _ = env.reset()

    in_channels = config.obs_dim[2] * config.frame_stack
    height = obs.shape[0]
    width =  obs.shape[1]
   
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
