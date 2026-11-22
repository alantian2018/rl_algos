import os
from dataclasses import dataclass
from functools import partial

import cv2
import draccus
import gymnasium
import numpy as np

from dreamer.dreamer.config import DreamerConfig
from dreamer.experiments.training import run_experiment


class PixelCarRacing(gymnasium.Wrapper):
    def __init__(self, image_size=(64, 64), action_repeat=2, reward_scale=0.1):
        if action_repeat < 1:
            raise ValueError("action_repeat must be positive")
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        super().__init__(
            gymnasium.make("CarRacing-v3", continuous=True, render_mode="rgb_array")
        )
        self.image_size = image_size
        self.action_repeat = action_repeat
        self.reward_scale = reward_scale
        self.observation_space = gymnasium.spaces.Box(
            0, 255, (*image_size, 3), dtype=np.uint8
        )

    def pixels(self, obs):
        return cv2.resize(obs, self.image_size[::-1], interpolation=cv2.INTER_AREA)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self.pixels(obs), info

    def step(self, action):
        raw_reward = 0.0
        for _ in range(self.action_repeat):
            obs, reward, terminated, truncated, info = self.env.step(action)
            raw_reward += reward
            if terminated or truncated:
                break
        info = dict(info, raw_reward=raw_reward)
        return (
            self.pixels(obs),
            raw_reward * self.reward_scale,
            terminated,
            truncated,
            info,
        )


@dataclass
class CarRacingConfig(DreamerConfig):
    # NOTE: uh this is a bit of a misnomer its actually number of env steps not train steps
    total_train_steps: int = 2_000_000
    reconstruction_log_freq: int = 2_000
    eval_every: int = 5_000
    eval_episodes: int = 3
    video_log_freq: int = 10_000
    # how many start states to sample from the replay buffer
    batch_size: int = 32
    # time horizon of how long each of those samples are
    sequence_length: int = 64
    device: str = "mps"
    save_freq: int = 10_000
    use_wandb: bool = True
    # wandb_entity : str =
    # wandb_project : str =

    exp_name: str = "carracing_pixels"
    obs_dim: tuple = (64, 64)
    obs_channels: int = 3
    action_dim: int = 3
    action_low: tuple = (-1.0, 0.0, 0.0)
    action_high: tuple = (1.0, 1.0, 1.0)
    deterministic_dim: int = 200
    stochastic_dim: int = 30
    latent_obs_size: int = 256
    hidden_dim: int = 256
    before_training_steps: int = 5_000
    train_steps_per_episode: int = 100
    replay_buffer_capacity: int = 100_000
    save_replay_buffer: bool = True
    model_lr: float = 6e-4
    actor_lr: float = 8e-5
    critic_lr: float = 8e-5
    imagination_horizon: int = 15
    reconstruction_weight: float = 1.0
    edge_weight: float = 1.0
    reward_weight: float = 1.0
    continuation_weight: float = 10.0
    kl_weight: float = 0.1
    free_nats: float = 3.0
    log_freq: int = 1
    video_fps: int = 25
    action_repeat: int = 1
    reward_scale: float = 0.1
    num_threads: int = 4


@draccus.wrap()
def main(config: CarRacingConfig):
    make_env = partial(
        PixelCarRacing, config.obs_dim, config.action_repeat, config.reward_scale
    )
    run_experiment(config, make_env)


if __name__ == "__main__":
    main()
