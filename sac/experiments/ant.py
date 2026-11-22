import gymnasium
import draccus
from dataclasses import dataclass
from sac import Policy, Qfunction, MLPEncoder
from sac import SAC, SACConfig
 


def make_ant_env(render_mode=None):
    env = gymnasium.make("Ant-v5", render_mode=render_mode)
    return env


@dataclass
class AntSACConfig(SACConfig):
    exp_name: str = "ant"
    state_dim: int = 105
    action_dim: int = 8
    action_low: tuple = (-1.0,) * 8
    action_high: tuple = (1.0,) * 8
    frame_stack: int = 3

    gradient_step_ratio: int = 256
    collect_rollout_steps: int = 256
    before_training_steps: int = 20_000
    batch_size: int = 256
    replay_buffer_capacity: int = 1_000_000
    total_train_steps: int = 1_000_000

    video_log_freq: int = 10_000
    log_freq: int = 5
    save_freq: int = 25_000
    path_to_checkpoint: str = "sac/checkpoints/ant/20260301_214822/checkpoint_301056.pt"


@draccus.wrap()
def main(config: AntSACConfig):
    env = make_ant_env()

    policy_encoder = MLPEncoder(
        state_dim=config.state_dim * config.frame_stack, hidden_dim=config.hidden_dim
    )
    qf1_encoder = MLPEncoder(state_dim=config.state_dim * config.frame_stack, hidden_dim=config.hidden_dim)
    qf2_encoder = MLPEncoder(state_dim=config.state_dim * config.frame_stack, hidden_dim=config.hidden_dim)

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

    sac = SAC(config, policy, qf1, qf2, env, make_env=make_ant_env)

    sac.train(total_train_steps=config.total_train_steps)


if __name__ == "__main__":
    main()
