import os
from dataclasses import dataclass
from functools import partial

import cv2
import draccus
import gymnasium
import numpy as np

from dreamer.dreamer.config import DreamerConfig
from dreamer.experiments.training import evaluate, run_experiment


class PixelPendulum(gymnasium.Wrapper):
    def __init__(self, image_size=(64, 64), action_repeat=1, reward_scale=0.1):
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        super().__init__(gymnasium.make("Pendulum-v1", render_mode="rgb_array"))
        self.image_size = image_size
        self.action_repeat = action_repeat
        self.reward_scale = reward_scale
        self.observation_space = gymnasium.spaces.Box(
            0, 255, (*image_size, 1), dtype=np.uint8
        )

    def pixels(self):
        pendulum = self.env.unwrapped
        last_u = pendulum.last_u
        try:
            pendulum.last_u = None
            frame = self.env.render()
        finally:
            pendulum.last_u = last_u
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, self.image_size[::-1], interpolation=cv2.INTER_AREA)
        return frame[..., None]

    def reset(self, **kwargs):
        _, info = self.env.reset(**kwargs)
        return self.pixels(), info

    def step(self, action):
        raw_reward = 0.0
        for _ in range(self.action_repeat):
            _, reward, terminated, truncated, info = self.env.step(action)
            raw_reward += reward
            if terminated or truncated:
                break
        info = dict(info, raw_reward=raw_reward)
        return (
            self.pixels(),
            raw_reward * self.reward_scale,
            terminated,
            truncated,
            info,
        )


@dataclass
class PendulumConfig(DreamerConfig):
    exp_name: str = "pendulum_pixels"
    obs_dim: tuple = (64, 64)
    obs_channels: int = 1
    action_dim: int = 1
    action_low: tuple = (-2.0,)
    action_high: tuple = (2.0,)
    deterministic_dim: int = 128
    stochastic_dim: int = 16
    latent_obs_size: int = 128
    hidden_dim: int = 128
    batch_size: int = 50
    sequence_length: int = 32
    before_training_steps: int = 5_000
    train_steps_per_episode: int = 50
    replay_buffer_capacity: int = 200_000
    save_replay_buffer: bool = True
    model_lr: float = 6e-4
    actor_lr: float = 8e-5
    critic_lr: float = 8e-5
    imagination_horizon: int = 15

    free_nats: float = 3.0
    reconstruction_weight: float = 1.0
    edge_weight: float = 1.0
    reward_weight: float = 1.0
    continuation_weight: float = 10.0
    kl_weight: float = 0.1

    device: str = "mps"
    use_wandb: bool = True
    save_freq: int = 10_000
    log_freq: int = 1

    # this is kinda a misnomer, this is number of environment interactions. should probably change but ehhhhhhhhh
    total_train_steps: int = 2_000_000
    reconstruction_log_freq: int = 500
    eval_every: int = 500
    eval_episodes: int = 5
    video_log_freq: int = 1000
    video_fps: int = 10
    action_repeat: int = 1
    reward_scale: float = 0.1

    num_threads: int = 4


@draccus.wrap()
def main(config: PendulumConfig):
    make_env = partial(
        PixelPendulum, config.obs_dim, config.action_repeat, config.reward_scale
    )
    run_experiment(config, make_env)


if __name__ == "__main__":
    main()
