from dataclasses import dataclass
from typing import Optional
from common import GlobalConfig
from datetime import datetime


@dataclass(kw_only=True)
class PPOConfig(GlobalConfig):
    obs_dim: Optional[int | tuple] = None
    act_dim: Optional[int] = None
    act_shape: Optional[int] = 1
    actor_hidden_size: int = 128
    critic_hidden_size: int = 128

    T: int = 2048
    gamma: float = 0.99
    gae_lambda: float = 0.95
    epsilon: float = 0.2
    actor_lr: float = 3e-4
    critic_lr: float = 1e-3
    minibatch_size: int = 64
    epochs_per_batch: int = 4
    entropy_coefficient: float = 0.01
    entropy_decay: bool = True
    entropy_decay_steps: Optional[int] = None

    # prefer ppo-specific checkpoint folder when save_dir is not provided
    save_dir: Optional[str] = None

    def __post_init__(self):

        if self.save_dir is None:
            self.save_dir = f"ppo/checkpoints/{self.exp_name}"

        if self.wandb_project is None and self.use_wandb:
            self.wandb_project = f"ppo-{self.exp_name}"
