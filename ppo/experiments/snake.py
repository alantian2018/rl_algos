from env.snake.snake import SnakeEnv
from ppo import PPOConfig, SnakeActor, SnakeCritic, PPO
from dataclasses import dataclass
from functools import partial
import draccus


@dataclass
class SnakeConfig(PPOConfig):
    """Snake-specific config."""

    exp_name: str = "snake"

    act_dim: int = 4
    grid_height: int = 10
    grid_width: int = 10
    max_steps: int = 2000
    in_channels: int = 3

    actor_hidden_size: int = 64
    critic_hidden_size: int = 64
    total_gradient_steps: int = 1_000_000

    video_log_freq: int = 1_000

    save_freq: int = 20_000
    device: str = "cpu"


def make_snake_env(grid_height, grid_width, max_steps, render_mode=None):
    return SnakeEnv(grid_height, grid_width, max_steps, render_mode)


@draccus.wrap()
def main(config: SnakeConfig):
    # Add timestamp to save_dir

    env = make_snake_env(config.grid_height, config.grid_width, config.max_steps)

    actor = SnakeActor(
        in_channels=config.in_channels,
        height=config.grid_height,
        width=config.grid_width,
        act_dim=config.act_dim,
        hidden_size=config.actor_hidden_size,
    )
    critic = SnakeCritic(
        in_channels=config.in_channels,
        height=config.grid_height,
        width=config.grid_width,
        hidden_size=config.critic_hidden_size,
    )

    # Bind grid params so make_env only needs render_mode
    make_env = partial(
        make_snake_env, config.grid_height, config.grid_width, config.max_steps
    )
    ppo = PPO(config, env, actor, critic, make_env=make_env)
    ppo.run_batch(config.total_gradient_steps)


if __name__ == "__main__":
    main()
