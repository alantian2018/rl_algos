import json
from dataclasses import asdict
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
import wandb

from dreamer.dreamer.algorithm import Dreamer
from dreamer.dreamer.visualization import reconstruct_replay


@torch.no_grad()
def evaluate(agent, config, video_path=None):
    return agent.evaluate(
        config.eval_episodes,
        seed=config.eval_seed,
        reward_key="raw_reward",
        video_path=video_path,
        video_fps=config.video_fps,
    )


def run_experiment(config, make_env):
    torch.set_num_threads(config.num_threads)
    env = make_env()
    agent = Dreamer(config, env, make_env=make_env)
    run_dir = Path(agent.logger.save_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(asdict(config), indent=2))
    initial_video = (
        run_dir / "videos" / f"eval_{agent.env_steps}.mp4"
        if config.video_log_freq is not None
        else None
    )
    initial_scores = evaluate(agent, config, video_path=initial_video)
    initial_mean = float(np.mean(initial_scores))
    results = {
        "evaluations": [
            {
                "step": agent.env_steps,
                "returns": initial_scores,
                "mean": initial_mean,
            }
        ]
    }
    (run_dir / "results.json").write_text(json.dumps(results, indent=2))
    original_log = agent.logger.log
    initial_metrics = {"eval/raw_return_mean": initial_mean}
    if initial_video is not None and agent.logger.use_wandb:
        initial_metrics["eval/video"] = wandb.Video(
            str(initial_video), fps=config.video_fps, format="mp4"
        )
    original_log(initial_metrics, step=agent.env_steps)
    next_eval = agent.env_steps + config.eval_every
    next_video = (
        agent.env_steps + config.video_log_freq
        if config.video_log_freq is not None
        else None
    )

    next_reconstruction = (
        agent.env_steps + config.reconstruction_log_freq
        if config.reconstruction_log_freq is not None
        else None
    )

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
            results["evaluations"].append(
                {"step": step, "returns": scores, "mean": float(np.mean(scores))}
            )
            (run_dir / "results.json").write_text(json.dumps(results, indent=2))

            next_eval = step + config.eval_every
            metrics["eval/raw_return_mean"] = float(np.mean(scores))
            if video_path is not None and agent.logger.use_wandb:
                metrics["eval/video"] = wandb.Video(
                    str(video_path), fps=config.video_fps, format="mp4"
                )
        if (
            next_reconstruction is not None
            and agent.gradient_steps > 0
            and (step >= next_reconstruction or final_step)
        ):
            grid = reconstruct_replay(agent, config.reconstruction_frames)
            if grid is not None:
                image_path = run_dir / "reconstructions" / f"reconstruction_{step}.png"
                image_path.parent.mkdir(parents=True, exist_ok=True)
                imageio.imwrite(image_path, grid)
                if agent.logger.use_wandb:
                    metrics["model/reconstructions"] = wandb.Image(
                        str(image_path),
                        caption="Top: replay frames. Bottom: posterior reconstructions conditioned on observations.",
                    )
                next_reconstruction = step + config.reconstruction_log_freq
        original_log(metrics, step)

    agent.logger.log = log
    try:
        agent.train(config.total_train_steps)
        if (
            not results["evaluations"]
            or results["evaluations"][-1]["step"] != agent.env_steps
        ):
            scores = evaluate(agent, config)
            results["evaluations"].append(
                {
                    "step": agent.env_steps,
                    "returns": scores,
                    "mean": float(np.mean(scores)),
                }
            )
        (run_dir / "results.json").write_text(json.dumps(results, indent=2))
        print("RESULTS", str(run_dir / "results.json"), flush=True)
    finally:
        env.close()
