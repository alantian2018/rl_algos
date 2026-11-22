import gymnasium
import draccus
from dataclasses import dataclass
from sac import Policy, Qfunction, MLPEncoder
from sac import SAC, SACConfig


def make_pendulum_env(render_mode=None):
    env = gymnasium.make(
        "InvertedPendulum-v5", reset_noise_scale=0.1, render_mode=render_mode
    )
    return env


@dataclass
class PendulumSACConfig(SACConfig):
    exp_name: str = "inverted_pendulum"
    state_dim: int = 4
    action_dim: int = 1
    action_low: tuple = (-3.0,)
    action_high: tuple = (3.0,)

    gradient_step_ratio: int = 100
    collect_rollout_steps: int = 128
    batch_size: int = 64
    before_training_steps: int = 200
    replay_buffer_capacity: int = 1_000_000
    total_train_steps: int = 75_000

    video_log_freq: int = 1000
    log_freq: int = 5
    save_freq: int = 25_000


@draccus.wrap()
def main(config: PendulumSACConfig):
    env = make_pendulum_env()

    policy_encoder = MLPEncoder(
        state_dim=config.state_dim, hidden_dim=config.hidden_dim
    )
    qf1_encoder = MLPEncoder(state_dim=config.state_dim, hidden_dim=config.hidden_dim)
    qf2_encoder = MLPEncoder(state_dim=config.state_dim, hidden_dim=config.hidden_dim)

    policy = Policy(
        encoder=policy_encoder,
        action_dim=config.action_dim,
        action_low=config.action_low,
        action_high=config.action_high,
    )
    qf1 = Qfunction(
        encoder=qf1_encoder, action_dim=config.action_dim, hidden_dim=config.hidden_dim
    )
    qf2 = Qfunction(
        encoder=qf2_encoder, action_dim=config.action_dim, hidden_dim=config.hidden_dim
    )

    sac = SAC(config, policy, qf1, qf2, env, make_env=make_pendulum_env)

    sac.train(total_train_steps=config.total_train_steps)


if __name__ == "__main__":
    main()
