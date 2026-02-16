import gymnasium
import draccus
from dataclasses import dataclass, field

from sac import SAC, SACConfig, Qfunction, CNNEncoder, Policy

from common import NormalizeObsWrapper


# TODO WIP NEED TO ADD CUSTOM POLICY AND Q NETWORKS HERE
def make_carracing_env(render_mode=None, normalize=True):
    env = gymnasium.make(
        "CarRacing-v3",
        continuous=True,
        render_mode=render_mode,
    )
    if not normalize:
        return env

    return NormalizeObsWrapper(env)


@dataclass
class CarRacing(SACConfig):
    exp_name: str = "carracing"
    state_dim: tuple = field(default=(96, 96, 3))
    action_dim: int = 3
    action_low: float = -1.0
    action_high: float = 1.0
    hidden_dim: int = 128
    autotune_entropy: bool = True
    batch_size: int = 64
    gradient_step_ratio: int = 3
    collect_rollout_steps: int = 64
    before_training_steps: int = 500
    replay_buffer_capacity: int = 10_000_000
    total_train_steps: int = 50_000
    video_log_freq: int = 2_000
    save_freq: int = 10_000
    log_freq: int = 100
    device: str = "cpu"


@draccus.wrap()
def main(config: CarRacing):
    env = make_carracing_env()
    height, width, in_channels = config.state_dim

    policy_encoder = CNNEncoder(
        in_channels=in_channels,
        height=height,
        width=width,
        hidden_dim=config.hidden_dim,
    )
    qf1_encoder = CNNEncoder(
        in_channels=in_channels,
        height=height,
        width=width,
        hidden_dim=config.hidden_dim,
    )
    qf2_encoder = CNNEncoder(
        in_channels=in_channels,
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
