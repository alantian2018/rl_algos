from dataclasses import dataclass
from math import isfinite

from common.config import GlobalConfig


@dataclass(kw_only=True)
class DreamerConfig(GlobalConfig):
    # Current observation handler accepts channels-first images.
    obs_dim: tuple[int, int]  # (height, width)
    obs_channels: int
    action_dim: int
    action_low: tuple[float, ...]
    action_high: tuple[float, ...]
    squash_actions: bool = True  # tanh, then scale to [action_low, action_high]

    deterministic_dim: int = 200
    stochastic_dim: int = 30
    latent_obs_size: int = 1024
    hidden_dim: int = 400

    model_lr: float = 6e-4
    actor_lr: float = 8e-5
    critic_lr: float = 8e-5
    grad_clip: float = 50.0
    entropy_coefficient: float = 3e-4
    return_norm_rate: float = 0.01

    reconstruction_weight: float = 1.0
    edge_weight: float = 0.0
    reward_weight: float = 1.0
    continuation_weight: float = 10.0
    kl_weight: float = 1.0
    free_nats: float = 3.0

    batch_size: int = 50
    sequence_length: int = 50  # transitions per replay window
    replay_buffer_capacity: int = 100_000  # environment transitions
    before_training_steps: int = 5_000
    train_steps_per_episode: int = 100  # updates after collecting a complete episode
    save_replay_buffer: bool = False
    log_freq: int = 10  # gradient updates
    reconstruction_log_freq: int | None = 2_000
    reconstruction_frames: int = 8

    gamma: float = 0.99
    lambda_: float = 0.95
    imagination_horizon: int = 15

    seed: int | None = None
    eval_seed: int | None = None

    def __post_init__(self):
        if not isfinite(self.edge_weight) or self.edge_weight < 0:
            raise ValueError("edge_weight must be finite and nonnegative")
        if self.action_dim < 1:
            raise ValueError("action_dim must be positive")
        if (
            len(self.action_low) != self.action_dim
            or len(self.action_high) != self.action_dim
        ):
            raise ValueError("action bounds must have length action_dim")
        for low, high in zip(self.action_low, self.action_high):
            if not low < high:
                raise ValueError("action_low must be less than action_high")
            if self.squash_actions and not (isfinite(low) and isfinite(high)):
                raise ValueError("squash_actions requires finite action bounds")
        if self.batch_size < 1 or self.sequence_length < 1:
            raise ValueError("batch_size and sequence_length must be positive")
        if self.replay_buffer_capacity < self.sequence_length:
            raise ValueError("replay capacity must hold at least one sequence")
        if self.before_training_steps < 0 or self.train_steps_per_episode < 0:
            raise ValueError("training step counts must be nonnegative")
        if self.log_freq < 1 or self.imagination_horizon < 1:
            raise ValueError("log_freq and imagination_horizon must be positive")
        if self.reconstruction_frames < 1 or (
            self.reconstruction_log_freq is not None
            and self.reconstruction_log_freq < 1
        ):
            raise ValueError(
                "reconstruction_frames and enabled reconstruction_log_freq must be positive"
            )

        if self.save_dir is None:
            self.save_dir = f"dreamer/checkpoints/{self.exp_name}"
        if self.wandb_project is None and self.use_wandb:
            self.wandb_project = f"dreamer-{self.exp_name}"
