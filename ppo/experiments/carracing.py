import gymnasium
import draccus
from dataclasses import dataclass


from ppo import PPOConfig, PPO, ImageActor, ImageCritic
from common import NormalizeObsWrapper


def make_carracing_env(render_mode=None, normalize=True):
    env = gymnasium.make(
        "CarRacing-v3",
        continuous=False,
        render_mode=render_mode,
    )
    if not normalize:
        return env

    return NormalizeObsWrapper(env)


@dataclass
class CarRacingConfig(PPOConfig):
    exp_name: str = "carracing"

    obs_dim: tuple = (96, 96, 3)
    act_dim: int = 5

    actor_hidden_size: int = 128
    critic_hidden_size: int = 128

    entropy_coefficient: float = 0.01
    entropy_decay: bool = False
    entropy_decay_steps: bool = 100_000
    minibatch_size: int = 64
    T: int = 2048
    epsilon: int = 0.1

    total_gradient_steps: int = 500_000
    frame_stack: int = 4

    wandb_entity: str = "apcsc"

    video_log_freq: int = 2_000

    save_freq: int = 10_000

    device: str = "cpu"
    path_to_checkpoint: str = (
        "ppo/checkpoints/carracing/20260127_173026/checkpoint_9999.pt"
    )


@draccus.wrap()
def main(config: CarRacingConfig):
    env = make_carracing_env(normalize=True)
    obs, _ = env.reset()

    in_channels = config.obs_dim[2] * config.frame_stack
    height = obs.shape[0]
    width = obs.shape[1]

    actor = ImageActor(
        in_channels=in_channels,
        height=height,
        width=width,
        act_dim=config.act_dim,
        hidden_size=config.actor_hidden_size,
    )
    critic = ImageCritic(
        in_channels=in_channels,
        height=height,
        width=width,
        hidden_size=config.critic_hidden_size,
    )

    ppo = PPO(config, env, actor, critic, make_env=make_carracing_env)
    ppo.run_batch(config.total_gradient_steps)


if __name__ == "__main__":
    main()
