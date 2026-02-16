import gymnasium
import draccus
from dataclasses import dataclass
import torch
from ppo import PPO, PPOConfig
from ppo import Actor, Critic


def make_cartpole_env(render_mode=None):
    return gymnasium.make("CartPole-v1", render_mode=render_mode)


# PLZ DO NOT FORGET TYPE HINTS
@dataclass
class CartPoleConfig(PPOConfig):
    """CartPole-specific config."""

    exp_name: str = "cartpole"
    obs_dim: int = 4
    act_dim: int = 2
    actor_hidden_size: int = 32
    critic_hidden_size: int = 32
    device: str = "cpu"
    frame_stack: int = 4
    # Training
    total_gradient_steps: int = 50_000
    video_log_freq: int = 5000

    save_freq: int = 5000


@draccus.wrap()
def main(config: CartPoleConfig):

    env = make_cartpole_env()

    actor = Actor(
        config.obs_dim * config.frame_stack, config.act_dim, config.actor_hidden_size
    )
    critic = Critic(config.obs_dim * config.frame_stack, config.critic_hidden_size)
    ppo = PPO(config, env, actor, critic, make_env=make_cartpole_env)
    ppo.run_batch(total_gradient_steps=config.total_gradient_steps)


if __name__ == "__main__":
    main()
