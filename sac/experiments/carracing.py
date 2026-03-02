import gymnasium
import numpy as np
import draccus
from dataclasses import dataclass, field

from sac import SAC, SACConfig, Qfunction, CNNEncoder, Policy

from common import NormalizeObsWrapper


class ThrottleWrapper(gymnasium.ActionWrapper):
    """Maps 2D action [steering, throttle] to 3D [steering, gas, brake]."""

    def __init__(self, env):
        super().__init__(env)

    def action(self, action):
        steering, throttle = action
        if throttle > 0:
            gas, brake = throttle, 0.0
        else:
            gas, brake = 0.0, -throttle

        return np.array([steering, gas, brake], dtype=np.float32)


class NormalizeRewardWrapper(gymnasium.RewardWrapper):
    def __init__(self, env, scale=1 / 90):
        super().__init__(env)
        self.scale = scale

    def reward(self, reward):
        return reward * self.scale


def make_carracing_env(render_mode=None, normalize=True):
    env = gymnasium.make(
        "CarRacing-v3",
        continuous=True,
        render_mode=render_mode,
    )
    if not normalize:
        return env

    return NormalizeRewardWrapper(NormalizeObsWrapper(ThrottleWrapper(env)))


@dataclass
class CarRacing(SACConfig):
    exp_name: str = "carracing"
    state_dim: tuple = field(default=(96, 96, 3))

    action_dim: int = 2
    action_low: float = (-1.0, -1.0)
    action_high: float = (1.0, 1.0)

    hidden_dim: int = 128
    autotune_entropy: bool = True
    batch_size: int = 32
    gradient_step_ratio: int = 1
    collect_rollout_steps: int = 32
    before_training_steps: int = 10

    frame_stack: int = 4

    replay_buffer_capacity: int = 10_000
    total_train_steps: int = 500_000

    video_log_freq: int = 250
    save_freq: int = 5_000
    log_freq: int = 5

    wandb_entity: str = None


@draccus.wrap()
def main(config: CarRacing):
    env = make_carracing_env()

    height, width, in_channels = config.state_dim

    policy_encoder = CNNEncoder(
        in_channels=in_channels * config.frame_stack,
        height=height,
        width=width,
        hidden_dim=config.hidden_dim,
    )
    qf1_encoder = CNNEncoder(
        in_channels=in_channels * config.frame_stack,
        height=height,
        width=width,
        hidden_dim=config.hidden_dim,
    )
    qf2_encoder = CNNEncoder(
        in_channels=in_channels * config.frame_stack,
        height=height,
        width=width,
        hidden_dim=config.hidden_dim,
    )

    policy = Policy(
        encoder=policy_encoder,
        action_dim=config.action_dim,
        action_low=config.action_low,
        action_high=config.action_high,
    )
    qf1 = Qfunction(qf1_encoder, config.action_dim, config.hidden_dim)
    qf2 = Qfunction(qf2_encoder, config.action_dim, config.hidden_dim)

    sac = SAC(config, policy, qf1, qf2, env, make_env=make_carracing_env)
    sac.train(total_train_steps=config.total_train_steps)


if __name__ == "__main__":
    main()
