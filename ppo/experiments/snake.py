from env.snake import SnakeEnv
from ppo import PPOConfig, CNNActor, CNNCritic, PPO
from dataclasses import dataclass, replace
from functools import partial
from datetime import datetime
import draccus
import torch
import termcolor

@dataclass 
class SnakeConfig(PPOConfig):
    """Snake-specific config."""
    
    grid_height: int = 10
    grid_width: int = 10
    max_steps: int = 2000
    in_channels: int = 3
    obs_dim: tuple[int, int, int] = (10, 10, 3)  # for compatibility with base class
    act_dim: int = 4
    actor_hidden_size: int = 64
    critic_hidden_size: int = 64
    total_gradient_steps: int = 1_000_000
    device: str = "cpu"
    wandb_project: str = "ppo-snake"
    wandb_run_name: str = None
    video_log_freq: int = 1_000
    gamma: float = 0.99

    save_dir: str = "checkpoints/snake"
    save_freq: int = 20_000

    load_from_existing_checkpoint: bool = False
    load_checkpoint_path: str = None


def make_snake_env(grid_height, grid_width, max_steps, render_mode=None):
    return SnakeEnv(grid_height, grid_width, max_steps, render_mode)


@draccus.wrap()
def main(config: SnakeConfig):
    # Add timestamp to save_dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config = replace(config, save_dir=f"{config.save_dir}/{timestamp}")
    
    env = make_snake_env(config.grid_height, config.grid_width, config.max_steps)
     
    actor = CNNActor(
        in_channels=config.in_channels,
        height=config.grid_height,
        width=config.grid_width,
        act_dim=config.act_dim,
        hidden_size=config.actor_hidden_size,
    )
    critic = CNNCritic(
        in_channels=config.in_channels,
        height=config.grid_height,
        width=config.grid_width,
        hidden_size=config.critic_hidden_size,
    )
    if config.load_from_existing_checkpoint:
        actor.load_state_dict(torch.load(config.load_checkpoint_path)['actor_state_dict'])
        critic.load_state_dict(torch.load(config.load_checkpoint_path)['critic_state_dict'])
        print(termcolor.colored(f"Loaded checkpoint from {config.load_checkpoint_path}", 'green'))

    
    # Bind grid params so make_env only needs render_mode
    make_env = partial(make_snake_env, config.grid_height, config.grid_width, config.max_steps)
    ppo = PPO(config, env, actor, critic, make_env=make_env)
    ppo.run_batch(config.total_gradient_steps)

if __name__ == "__main__":
    main()