import gymnasium
import draccus
from dataclasses import dataclass
import torch
import sys
import os

 
from sac import SAC, SACConfig


def make_pendulum_env(render_mode=None):
    return gymnasium.make("Pendulum-v1", render_mode=render_mode)


@dataclass 
class PendulumSACConfig(SACConfig):
    state_dim: int = 3
    action_dim: int = 1
    hidden_dim: int = 64
    gamma: float = 0.99
    tau: float = 0.2
    alpha: float = 0.0
    batch_size: int = 256
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    gradient_step_ratio: int = 1
    collect_rollout_steps: int = 64
    before_training_steps: int = 1000
    replay_buffer_capacity: int = 1_000_000
    total_train_steps: int = 50_000
    action_low: float = -2.0
    action_high: float = 2.0
    is_continuous: bool = True
    
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Wandb
    wandb_project: str = "sac-pendulum"
    wandb_run_name: str = None
    video_log_freq: int = 1000
    log_freq: int = 100


@draccus.wrap()
def main(config: PendulumSACConfig):
    env = make_pendulum_env()
    sac = SAC(config, env, make_env=make_pendulum_env)
    sac.train(total_train_steps=config.total_train_steps)


if __name__ == "__main__":
    main()
