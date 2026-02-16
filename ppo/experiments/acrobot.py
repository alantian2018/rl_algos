import gymnasium
import draccus
from dataclasses import dataclass
from ppo import PPO, PPOConfig, Actor, Critic


def make_acrobot_env(render_mode=None):
    return gymnasium.make("Acrobot-v1", render_mode=render_mode)


@dataclass
class AcrobotConfig(PPOConfig):
    """Acrobot-specific config."""

    exp_name: str = "acrobot"
    obs_dim: int = 6
    act_dim: int = 3
    actor_hidden_size: int = 64
    critic_hidden_size: int = 64

    # Training
    total_gradient_steps: int = 200_000

    video_log_freq: int = total_gradient_steps // 20


@draccus.wrap()
def main(config: AcrobotConfig):
    env = make_acrobot_env()
    actor = Actor(config.obs_dim, config.act_dim, config.actor_hidden_size)
    critic = Critic(config.obs_dim, config.critic_hidden_size)
    ppo = PPO(config, env, actor, critic, make_env=make_acrobot_env)
    ppo.run_batch(total_gradient_steps=config.total_gradient_steps)


if __name__ == "__main__":
    main()
