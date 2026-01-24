import gymnasium
import draccus
from dataclasses import dataclass
import torch
import sys
import os
 
from sac import SAC, SACConfig


class CartPoleWrapper(gymnasium.Wrapper):
    def step(self, action):
        discrete_action = int(action[0] > 0.5)
        return self.env.step(discrete_action)


def make_cartpole_env(render_mode=None):
    env = gymnasium.make("CartPole-v1", render_mode=render_mode)
    return CartPoleWrapper(env)


@dataclass 
class CartPoleSACConfig(SACConfig):
    state_dim: int = 4
    action_dim: int = 1
    hidden_dim: int = 10
    gamma: float = 0.99
    tau: float = 0.005
    alpha: float = 0.
    batch_size: int = 256
    actor_lr: float = 3e-4
    critic_lr: float = 3e-5
    gradient_step_ratio: int = 1
    collect_rollout_steps: int = 200
    replay_buffer_capacity: int = 100_000
    total_train_steps: int = 50_000

    # for discrete, low = 0, high = len(discrete)
    action_low: float = 0.0
    action_high: float = 2.0
    is_continuous: bool = False
    
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    wandb_project: str = "sac-cartpole"
    wandb_run_name: str = None
    video_log_freq: int = 5000
    log_freq: int = 100


@draccus.wrap()
def main(config: CartPoleSACConfig):
    env = make_cartpole_env()
    sac = SAC(config, env, make_env=make_cartpole_env)
    sac.train(total_train_steps=config.total_train_steps)


if __name__ == "__main__":
    main()

