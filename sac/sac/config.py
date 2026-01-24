from dataclasses import dataclass
from typing import Optional

@dataclass
class SACConfig:
    state_dim: int
    action_dim: int
    hidden_dim: int = 64
    gamma: float = 0.99
    tau: float = 0.005
    alpha: float = 0.2
    batch_size: int = 256
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    update_every: int = 1
    num_updates: int = 10
    replay_buffer_capacity: int = 1_000_000
    before_training_steps: int = 1000
    gradient_step_ratio: int = 1 # num of gradient steps per rollout step
    collect_rollout_steps: int = 1000
    device: str = "cpu"
    action_low: float = 9999
    action_high: float = 9999
    is_continuous: bool = False
    # Wandb
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None
    video_log_freq: Optional[int] = None
    log_freq: int = 10  # Log every N training steps
