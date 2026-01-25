import torch
import numpy as np
import wandb
import gymnasium
from pathlib import Path
from dataclasses import asdict
from typing import Optional, Callable, Dict, Any
from torch import nn
import termcolor
from collections import deque
import signal, pdb
signal.signal(signal.SIGUSR1, lambda s,f: pdb.set_trace())

class Logger:
    """Handles all wandb logging and video recording, as well as saving/loading checkpointing"""
    
    def __init__(
        self,
        config: Any,
        make_env: Optional[Callable[..., gymnasium.Env]] = None,
        save_dir = None
    ):
        self.config = config
        self.make_env = make_env
        self.use_wandb = self.config.use_wandb
        self.video_log_freq = getattr(config, 'video_log_freq', None)
        self.last_video_step = -1
        self.last_saved = -1
        self.save_dir = save_dir
        
        if self.use_wandb:
            wandb.init(
                entity=config.wandb_entity,
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
    
    def log_training(self, training_info: dict, step: int):
        """Log training metrics."""

        dict_to_log = {}
        for key, item in training_info.items():
            dict_to_log[f'train/{key}']= item
        if self.use_wandb:
            self.log(dict_to_log, step=step)

    def save_checkpoint(
        self,
        networks: Dict[str, nn.Module],
        step: Any,
        **kwargs,
    ):
        """Save a training checkpoint."""

        path = f"{self.save_dir}/checkpoint_{step}.pt"
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        if step not in networks:
            networks['step'] = step
        if self.use_wandb:
            networks['wandb_id'] = wandb.run.id

        dict_to_save = {}
        
        for key, item in networks.items():
            if hasattr(item, 'state_dict'):
                dict_to_save[f'{key}_state_dict'] = item.state_dict()
            else:
                dict_to_save[key] = item

        torch.save(dict_to_save, path)
        print(termcolor.colored(f"Saved checkpoint to {path}", 'green'))

    def maybe_save_checkpoint(self, step: int, networks: Dict):
        """Save checkpoint if it's time to do so."""
        save_freq = self.config.save_freq
        if save_freq is None:
            return
        if step - self.last_saved >= self.config.save_freq:
            self.last_saved = step
            self.save_checkpoint( networks=networks, step=step)

    def load_checkpoint(
        self,
        path: str,
        networks: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Load a training checkpoint. Returns the checkpoint dict."""
        checkpoint = torch.load(path, weights_only=False)
        for key, item in networks.items():
            if hasattr(item, 'state_dict'):
                try:
                    item.load_state_dict(checkpoint[f'{key}_state_dict'])
                except:
                    print(termcolor.colored(f'WARNING! Failed to load {key}', 'yellow'))
            else:
                networks[key] = checkpoint[key]

        print(termcolor.colored( f"Loaded checkpoint from {path} (step {checkpoint['step']})", 'green'))
        return checkpoint

    def maybe_record_video(self, actor: torch.nn.Module, step: int, device: str):
        """Record video if it's time to do so."""
        if not self.use_wandb or self.video_log_freq is None or self.make_env is None:
            return
        
        if step == 0 or step - self.last_video_step >= self.video_log_freq:
            with torch.no_grad():
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
        frame_stack = FrameStack(frames = self.config.frame_stack)
        frame_stack.add_to_frame_stack(obs)
        while not done:

            frame = eval_env.render()
            frames.append(frame)
            
            with torch.no_grad():
                distribution = actor(frame_stack.get_frames())
                action = distribution.sample()
            
            obs, reward, terminated, truncated, _ = eval_env.step(action.squeeze(0).cpu().numpy())
            obs = torch.tensor(obs, dtype=torch.float32).to(device)
            frame_stack.add_to_frame_stack(obs)
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



class FrameStack:
    def __init__(self, frames: int, raw_obs_dim: Optional[tuple] = None):
        self.max_frames = frames
        self.frame_stack = deque(maxlen=frames)
        self.raw_obs_dim = raw_obs_dim

    def add_to_frame_stack(self, data: torch.Tensor):
        if self.raw_obs_dim:
            assert data.shape == self.raw_obs_dim
        
        self.frame_stack.append(data.detach().clone())
        while len(self.frame_stack) < self.max_frames:
            self.frame_stack.append(data.detach().clone())
       
    def get_frames(self): 
        return torch.concatenate(list(self.frame_stack), dim=-1)

        

