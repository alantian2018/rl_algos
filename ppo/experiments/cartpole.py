import gymnasium
import draccus
from dataclasses import dataclass
import torch
from ppo import PPO, PPOConfig
from ppo import Actor, Critic

def make_cartpole_env(render_mode=None):
    return gymnasium.make("CartPole-v1", render_mode=render_mode)


@dataclass 
class CartPoleConfig(PPOConfig):
    """CartPole-specific config."""
    obs_dim: int = 4
    act_dim: int = 2
    actor_hidden_size: int = 32
    critic_hidden_size: int = 32
    
    # Training
    total_gradient_steps: int = 200_000
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Wandb
    wandb_project: str = "ppo-cartpole"
    wandb_run_name: str = None
    video_log_freq: int = 5000


@draccus.wrap()
def main(config: CartPoleConfig):
    env = make_cartpole_env()
    actor = Actor(config.obs_dim, config.act_dim, config.actor_hidden_size)
    critic = Critic(config.obs_dim, config.critic_hidden_size)  
    ppo = PPO(config, env, actor, critic, make_env=make_cartpole_env)
    ppo.run_batch(total_gradient_steps=config.total_gradient_steps)


if __name__ == "__main__":
    main()

