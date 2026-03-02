import gymnasium
import numpy as np
from stable_baselines3 import SAC

from stable_baselines3.common.vec_env import (
    SubprocVecEnv,
    DummyVecEnv,
    VecFrameStack,
    VecTransposeImage,
    VecVideoRecorder,
    VecMonitor,
)
from stable_baselines3.common.callbacks import EvalCallback
from wandb.integration.sb3 import WandbCallback
import wandb
import torch


def make_env():
    def _init():
        env = gymnasium.make("CarRacing-v3", continuous=True, render_mode="rgb_array")
        return env

    return _init


if __name__ == "__main__":
    run = wandb.init(
        project="sac_carracing_gymnasium",
        sync_tensorboard=True,
        monitor_gym=True,
        save_code=True,
    )

    num_envs = 4
    env = SubprocVecEnv([make_env() for _ in range(num_envs)])
    env = VecFrameStack(env, n_stack=4)
    env = VecTransposeImage(env)
    env = VecMonitor(env)

    eval_env = DummyVecEnv([make_env()])
    eval_env = VecFrameStack(eval_env, n_stack=4)
    eval_env = VecTransposeImage(eval_env)
    eval_env = VecMonitor(eval_env)
    eval_env = VecVideoRecorder(
        eval_env,
        f"videos/{run.id}",
        record_video_trigger=lambda x: x % 5000 == 0,
        video_length=1000,
    )

    model = SAC(
        "CnnPolicy",
        env,
        buffer_size=20_000,
        learning_starts=1000,
        batch_size=256,
        optimize_memory_usage=False,
        tensorboard_log=f"runs/{run.id}",
        device=(
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        ),
    )

    eval_callback = EvalCallback(
        eval_env,
        n_eval_episodes=5,
        eval_freq=5000,
        best_model_save_path=f"models/{run.id}/best",
        log_path=f"models/{run.id}/eval",
        deterministic=True,
    )

    model.learn(
        total_timesteps=500_000,
        callback=[
            WandbCallback(
                model_save_path=f"models/{run.id}",
                verbose=2,
            ),
            eval_callback,
        ],
        log_interval=1,
        progress_bar=True,
    )
    model.save("sac_carracing")
    run.finish()
