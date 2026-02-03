from dataclasses import dataclass
from typing import Optional
from common import GlobalConfig
@dataclass(kw_only=True)
class SACConfig(GlobalConfig):
    state_dim: int
    action_dim: int
    hidden_dim: int = 64

    # discount factor
    gamma: float = 0.99
    # weight update factor
    tau: float = 0.005

    # entropy stuff
    alpha: float = 0.2
    autotune_entropy: float = True
    target_entropy: Optional[int] = None
    alpha_lr: int = 3e-4

    # learning hyperparams
    batch_size: int = 256
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4

 
 
    replay_buffer_capacity: int = 1_000_000
    before_training_steps: int = 1000
    gradient_step_ratio: int = 1 # num of gradient steps per rollout step
    collect_rollout_steps: int = 1000
    device: str = "cpu"
    action_low: float = 9999
    action_high: float = 9999
    is_continuous: bool = True
    save_dir: Optional[str] = None
    

    def __post_init__(self):
       
        if self.save_dir is None:
            self.save_dir = f'sac/checkpoints/{self.exp_name}'

        if self.wandb_project is None and self.use_wandb:
            self.wandb_project = f'sac-{self.exp_name}'



