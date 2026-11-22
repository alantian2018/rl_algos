import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from functools import partial

import cv2
import imageio.v2 as imageio
import draccus
import gymnasium
import numpy as np
import torch
import wandb

from dreamer.dreamer.algorithm import Dreamer
from dreamer.dreamer.config import DreamerConfig
from dreamer.dreamer.visualization import reconstruct_replay


class PixelPendulum(gymnasium.Wrapper):
    def __init__(self, image_size=(64, 64), action_repeat=2, reward_scale=0.1):
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        super().__init__(gymnasium.make("Pendulum-v1", render_mode="rgb_array"))
        self.image_size = image_size
        self.action_repeat = action_repeat
        self.reward_scale = reward_scale
        self.observation_space = gymnasium.spaces.Box(0, 255, (*image_size, 1), dtype=np.uint8)

    def pixels(self):
        frame = self.env.render()
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
        return self.pixels(), raw_reward * self.reward_scale, terminated, truncated, info


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
    batch_size: int = 8
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
    device: str = "mps"
    use_wandb: bool = True
    save_freq: int = 10_000
    log_freq: int = 1

    # this is kinda a misnomer, this is number of environment interactions. should probably change but ehhhhhhhhh
    total_train_steps: int = 2_000_000
    eval_every: int = 200
    eval_episodes: int = 5
    video_log_freq: int = 600
    video_fps: int = 10
    action_repeat: int = 2
    reward_scale: float = 0.1
    num_threads: int = 4


@torch.no_grad()
def evaluate(agent, config, video_path=None):
    return agent.evaluate(
        config.eval_episodes, reward_key="raw_reward",
        video_path=video_path, video_fps=config.video_fps,
    )


@draccus.wrap()
def main(config: PendulumConfig):
    torch.set_num_threads(config.num_threads)
    make_env = partial(PixelPendulum, config.obs_dim, config.action_repeat, config.reward_scale)
    env = make_env()
    agent = Dreamer(config, env, make_env=make_env)
    run_dir = Path(agent.logger.save_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(asdict(config), indent=2))
    initial_video = run_dir / "videos" / f"eval_{agent.env_steps}.mp4" if config.video_log_freq is not None else None
    initial_scores = evaluate(agent, config, video_path=initial_video)
    initial_mean = float(np.mean(initial_scores))
    results = {"evaluations": [{
        "step": agent.env_steps, "returns": initial_scores, "mean": initial_mean,
    }]}
    (run_dir / "results.json").write_text(json.dumps(results, indent=2))
    original_log = agent.logger.log
    initial_metrics = {"eval/raw_return_mean": initial_mean}
    if initial_video is not None and agent.logger.use_wandb:
        initial_metrics["eval/video"] = wandb.Video(str(initial_video), fps=config.video_fps, format="mp4")
    original_log(initial_metrics, step=agent.env_steps)
    next_eval = agent.env_steps + config.eval_every
    next_video = agent.env_steps + config.video_log_freq if config.video_log_freq is not None else None

    next_reconstruction = agent.env_steps + config.reconstruction_log_freq if config.reconstruction_log_freq is not None else None

    def log(metrics, step):
        nonlocal next_eval, next_video, next_reconstruction
        with (run_dir / "training.jsonl").open("a") as stream:
            stream.write(json.dumps({"step": step, **metrics}) + "\n")
        metrics = dict(metrics)
        final_step = step >= config.total_train_steps
        if step >= next_eval or final_step:
            video_path = None
            if next_video is not None and (step >= next_video or final_step):
                video_path = run_dir / "videos" / f"eval_{step}.mp4"
                next_video = step + config.video_log_freq
            scores = evaluate(agent, config, video_path=video_path)
            results["evaluations"].append({"step": step, "returns": scores, "mean": float(np.mean(scores))})
            (run_dir / "results.json").write_text(json.dumps(results, indent=2))
          
            next_eval = step + config.eval_every
            metrics["eval/raw_return_mean"] = float(np.mean(scores))
            if video_path is not None and agent.logger.use_wandb:
                metrics["eval/video"] = wandb.Video(str(video_path), fps=config.video_fps, format="mp4")
        if next_reconstruction is not None and agent.gradient_steps > 0 and (step >= next_reconstruction or final_step):
            grid = reconstruct_replay(agent, config.reconstruction_frames)
            if grid is not None:
                image_path = run_dir / "reconstructions" / f"reconstruction_{step}.png"
                image_path.parent.mkdir(parents=True, exist_ok=True)
                imageio.imwrite(image_path, grid)
                if agent.logger.use_wandb:
                    metrics["model/reconstructions"] = wandb.Image(
                        str(image_path), caption="Top: replay frames. Bottom: posterior reconstructions conditioned on observations."
                    )
                next_reconstruction = step + config.reconstruction_log_freq
        original_log(metrics, step)

    agent.logger.log = log
    try:
        agent.train(config.total_train_steps)
        if not results["evaluations"] or results["evaluations"][-1]["step"] != agent.env_steps:
            scores = evaluate(agent, config)
            results["evaluations"].append({"step": agent.env_steps, "returns": scores, "mean": float(np.mean(scores))})
        (run_dir / "results.json").write_text(json.dumps(results, indent=2))
        print("RESULTS", str(run_dir / "results.json"), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
