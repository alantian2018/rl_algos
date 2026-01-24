import torch
import numpy as np
import wandb
import gymnasium
import os
from pathlib import Path
from dataclasses import asdict
from typing import Optional, Callable, Dict, Any


def save_checkpoint(
    path: str,
    actor: torch.nn.Module,
    critic: torch.nn.Module,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    step: int,
    **kwargs,
):
    """Save a training checkpoint."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'step': step,
        'actor_state_dict': actor.state_dict(),
        'critic_state_dict': critic.state_dict(),
        'actor_optimizer_state_dict': actor_optimizer.state_dict(),
        'critic_optimizer_state_dict': critic_optimizer.state_dict(),
        **kwargs,
    }, path)
    print(f"Saved checkpoint to {path}")


def load_checkpoint(
    path: str,
    actor: torch.nn.Module,
    critic: torch.nn.Module,
    actor_optimizer: Optional[torch.optim.Optimizer] = None,
    critic_optimizer: Optional[torch.optim.Optimizer] = None,
) -> Dict[str, Any]:
    """Load a training checkpoint. Returns the checkpoint dict."""
    checkpoint = torch.load(path, weights_only=False)
    actor.load_state_dict(checkpoint['actor_state_dict'])
    critic.load_state_dict(checkpoint['critic_state_dict'])
    if actor_optimizer is not None:
        actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
    if critic_optimizer is not None:
        critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
    print(f"Loaded checkpoint from {path} (step {checkpoint['step']})")
    return checkpoint


class Logger:
    """Handles all wandb logging and video recording."""
    
    def __init__(
        self,
        config: Any,
        make_env: Optional[Callable[..., gymnasium.Env]] = None,
    ):
        self.config = config
        self.make_env = make_env
        self.use_wandb = getattr(config, 'wandb_project', None) is not None
        self.video_log_freq = getattr(config, 'video_log_freq', None)
        self.last_video_step = -1
        
        if self.use_wandb:
            wandb.init(
                project=config.wandb_project,
                name=config.wandb_run_name,
                config=asdict(config),
            )
    
    def log(self, metrics: Dict[str, Any], step: int):
        """Log metrics to wandb."""
        if self.use_wandb:
            wandb.log(metrics, step=step)
    
    def log_rollout(self, episode_returns: list, step: int):
        """Log rollout episode statistics."""
        if self.use_wandb and episode_returns:
            self.log({
                "rollout/episode_return_mean": sum(episode_returns) / len(episode_returns),
                "rollout/episode_return_max": max(episode_returns),
                "rollout/num_episodes": len(episode_returns),
            }, step=step)
    
    def log_training(self, actor_loss: float, critic_loss: float, entropy: float, step: int):
        """Log training metrics."""
        if self.use_wandb:
            self.log({
                "train/actor_loss": actor_loss,
                "train/critic_loss": critic_loss,
                "train/entropy": entropy,
            }, step=step)
    
    def maybe_record_video(self, actor: torch.nn.Module, step: int, device: str):
        """Record video if it's time to do so."""
        if not self.use_wandb or self.video_log_freq is None or self.make_env is None:
            return
        
        if step == 0 or step - self.last_video_step >= self.video_log_freq:
            self._record_video(actor, step, device)
            self.last_video_step = step
    
    def _record_video(self, actor: torch.nn.Module, step: int, device: str):
        """Record an evaluation episode and log video to wandb."""
        eval_env = self.make_env(render_mode="rgb_array")
        frames = []
        obs, _ = eval_env.reset()
        obs = torch.tensor(obs, dtype=torch.float32).to(device)
        
        done = False
        episode_return = 0.0
        while not done:
            frame = eval_env.render()
            frames.append(frame)
            
            with torch.no_grad():
                distribution = actor(obs)
                action = distribution.sample()
            
            obs, reward, terminated, truncated, _ = eval_env.step(action[0].cpu().numpy())
            obs = torch.tensor(obs, dtype=torch.float32).to(device)
            episode_return += reward
            done = terminated or truncated
        
        eval_env.close()
        
        # Convert to video format: (T, H, W, C) -> (T, C, H, W)
        video = np.stack(frames).transpose(0, 3, 1, 2)
        self.log({
            "eval/video": wandb.Video(video, fps=30, format="mp4"),
            "eval/episode_return": episode_return,
        }, step=step)
    
    def finish(self):
        """Finish wandb run."""
        if self.use_wandb:
            wandb.finish()
