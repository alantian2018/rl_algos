import gymnasium
import draccus
from dataclasses import dataclass, field
import torch
from ppo import PPO, PPOConfig
from ppo import Actor, Critic
import numpy as np
from functools import partial
from env import ElevatorEnv_v1


def make_elevator_env(
    num_elevators,
    num_floors,
    max_steps,
    max_people,
    spawn_rate,
    render_mode="rgb_array",
):
    return ElevatorEnv_v1(
        num_elevators, num_floors, max_steps, max_people, spawn_rate, render_mode
    )


# PLZ DO NOT FORGET TYPE HINTS
@dataclass
class ElevatorConfig(PPOConfig):
    exp_name: str = "elevator"
    log_keys: list[str] = field(default_factory=lambda: ["mean_elevator_waiting_time", "max_elevator_waiting_time", "min_elevator_waiting_time"])
    num_elevators: int = 2
    num_floors: int = 10
    max_steps: int = 300
    max_people: int = 100
    spawn_rate: float = 0.02

    actor_hidden_size: int = 64
    critic_hidden_size: int = 64
    frame_stack: int = 4
    # Training
    total_gradient_steps: int = 50_000
    video_log_freq: int = 5000

    save_freq: int = 5000


@draccus.wrap()
def main(config: ElevatorConfig):
    
    config.wandb_project = f"ppo-elevator-{config.num_elevators}"
    config.save_dir = f"ppo/checkpoints/elevator/{config.num_elevators}"
    config.act_shape = config.num_elevators

    env = make_elevator_env(
        config.num_elevators,
        config.num_floors,
        config.max_steps,
        config.max_people,
        config.spawn_rate,
    )

    obs, _ = env.reset()
    obs_dim = obs.shape[0]
    actor = Actor(
        obs_dim * config.frame_stack,
        act_dim = 3,
        hidden_size=config.actor_hidden_size,
        act_shape=config.num_elevators,
    )
    critic = Critic(obs_dim * config.frame_stack, config.critic_hidden_size)

    make_env_fn = partial(
        make_elevator_env,
        config.num_elevators,
        config.num_floors,
        config.max_steps,
        config.max_people,
        config.spawn_rate,        
    )
    ppo = PPO(config, env, actor, critic, make_env=make_env_fn)
    ppo.run_batch(total_gradient_steps=config.total_gradient_steps)


if __name__ == "__main__":
    main()
