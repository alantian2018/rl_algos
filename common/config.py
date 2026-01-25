from dataclasses import dataclass
import torch
from typing import Optional

@dataclass(kw_only=True)
class GlobalConfig:
    exp_name: str 
    device: str = "cuda" if torch.cuda.is_available() \
                else 'mps' if torch.backends.mps.is_available() \
                else "cpu"
    frame_stack: int = 1

    # Wandb
    use_wandb: bool = True
    wandb_entity: Optional[str] = None
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None
    video_log_freq: Optional[int] = None
    
    # Checkpointing
    save_freq: Optional[int] = None  # Save checkpoint every N steps (None = disabled)
    save_dir: str = None   # Directory to save checkpoints

    # loads path_to_checkpoint if set
    path_to_checkpoint: Optional[str] = None
