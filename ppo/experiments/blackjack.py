import gymnasium
import draccus
from dataclasses import dataclass
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

    exp_name: str = "blackjack"
    obs_dim: int = 3
    act_dim: int = 2
    actor_hidden_size: int = 64
    critic_hidden_size: int = 64
    total_gradient_steps: int = 1_000_000
    video_log_freq: int = total_gradient_steps // 20


@draccus.wrap()
def main(config: BlackjackConfig):
    env = make_blackjack_env()

    ppo = PPO(config, env, make_env=make_blackjack_env)
    ppo.run_batch(total_gradient_steps=config.total_gradient_steps)


if __name__ == "__main__":
    main()
