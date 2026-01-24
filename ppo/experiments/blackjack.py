import gymnasium
import draccus
from dataclasses import dataclass
import torch
from ppo import PPO, PPOConfig, Actor, Critic
import numpy as np
class BlackjackWrapper(gymnasium.ObservationWrapper):
    def observation(self, obs):
        return np.array(obs, dtype=np.float32)

def make_blackjack_env(render_mode=None):
    env = gymnasium.make("Blackjack-v1", render_mode=render_mode)
    return BlackjackWrapper(env)


@dataclass 
class BlackjackConfig(PPOConfig):
    """Blackjack-specific config."""
    obs_dim: int = 3
    act_dim: int = 2
    actor_hidden_size: int = 64
    critic_hidden_size: int = 64
    total_gradient_steps: int = 1_000_000
    wandb_project: str = "ppo-blackjack"
    wandb_run_name: str = None
    video_log_freq: int = total_gradient_steps // 20


@draccus.wrap()
def main(config: BlackjackConfig):
    env = make_blackjack_env()
    actor = Actor(config.obs_dim, config.act_dim, config.actor_hidden_size)
    critic = Critic(config.obs_dim, config.critic_hidden_size)  
    ppo = PPO(config, env, actor, critic, make_env=make_blackjack_env)
    ppo.run_batch(total_gradient_steps=config.total_gradient_steps)


if __name__ == "__main__":
    main()

