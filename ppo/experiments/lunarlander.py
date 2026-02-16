import gymnasium
import draccus
from dataclasses import dataclass
from ppo import PPO, PPOConfig
from ppo import Actor, Critic


def make_lunarlander_env(render_mode=None):
    return gymnasium.make("LunarLander-v3", render_mode=render_mode)


@dataclass
class LunarLanderConfig(PPOConfig):
    """LunarLander-specific config."""

    obs_dim: int = 8
    act_dim: int = 4
    actor_hidden_size: int = 64
    critic_hidden_size: int = 64

    # Training
    total_gradient_steps: int = 1_000_000

    video_log_freq: int = total_gradient_steps // 20


@draccus.wrap()
def main(config: LunarLanderConfig):
    env = make_lunarlander_env()
    actor = Actor(config.obs_dim, config.act_dim, config.actor_hidden_size)
    critic = Critic(config.obs_dim, config.critic_hidden_size)
    ppo = PPO(config, env, actor, critic, make_env=make_lunarlander_env)
    ppo.run_batch(total_gradient_steps=config.total_gradient_steps)


if __name__ == "__main__":
    main()
