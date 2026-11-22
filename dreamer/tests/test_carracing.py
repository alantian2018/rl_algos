import json

import cv2
import gymnasium
import numpy as np
import pytest

from dreamer.experiments import carracing
from dreamer.experiments.carracing import CarRacingConfig, PixelCarRacing


@pytest.mark.parametrize("action_repeat", [1, 2, 4])
def test_carracing_pixels_rewards_and_time_limit(action_repeat):
    env = PixelCarRacing(action_repeat=action_repeat)
    reference = gymnasium.make("CarRacing-v3", continuous=True, render_mode="rgb_array")
    env.env._max_episode_steps = 3
    reference._max_episode_steps = 3
    try:
        obs, _ = env.reset(seed=4)
        raw_obs, _ = reference.reset(seed=4)
        assert obs.shape == (64, 64, 3)
        assert obs.dtype == np.uint8
        np.testing.assert_array_equal(
            obs, cv2.resize(raw_obs, (64, 64), interpolation=cv2.INTER_AREA)
        )
        action = np.array([0.1, 0.5, 0.0], dtype=np.float32)
        done = False
        while not done:
            expected_reward = 0.0
            for _ in range(action_repeat):
                raw_obs, reward, expected_terminated, expected_truncated, _ = (
                    reference.step(action)
                )
                expected_reward += reward
                if expected_terminated or expected_truncated:
                    break
            obs, reward, terminated, truncated, info = env.step(action)
            assert env.observation_space.contains(obs)
            np.testing.assert_array_equal(
                obs, cv2.resize(raw_obs, (64, 64), interpolation=cv2.INTER_AREA)
            )
            np.testing.assert_allclose(info["raw_reward"], expected_reward)
            np.testing.assert_allclose(reward, expected_reward * 0.1)
            assert (terminated, truncated) == (expected_terminated, expected_truncated)
            done = terminated or truncated
        assert env.render().shape[-1] == 3
    finally:
        env.close()
        reference.close()


def test_carracing_defaults_match_environment():
    config = CarRacingConfig()
    env = PixelCarRacing()
    try:
        assert config.obs_channels == 3 and config.frame_stack == 1
        np.testing.assert_array_equal(config.action_low, env.action_space.low)
        np.testing.assert_array_equal(config.action_high, env.action_space.high)
        assert config.path_to_checkpoint is None
        assert config.save_dir == "dreamer/checkpoints/carracing_pixels"
    finally:
        env.close()


def test_carracing_experiment_trains_and_logs_media(tmp_path, monkeypatch):
    original = PixelCarRacing

    def short_env(*args, **kwargs):
        env = original(*args, **kwargs)
        env.env._max_episode_steps = 6
        return env

    monkeypatch.setattr(carracing, "PixelCarRacing", short_env)
    config = CarRacingConfig(
        device="cpu",
        use_wandb=False,
        save_dir=str(tmp_path),
        save_freq=None,
        total_train_steps=1,
        action_repeat=2,
        before_training_steps=0,
        train_steps_per_episode=1,
        sequence_length=3,
        batch_size=1,
        imagination_horizon=2,
        deterministic_dim=16,
        stochastic_dim=8,
        latent_obs_size=12,
        hidden_dim=32,
        eval_episodes=1,
        video_log_freq=1,
        reconstruction_log_freq=1,
        num_threads=1,
    )
    carracing.main.__wrapped__(config)
    rows = [
        json.loads(line)
        for line in next(tmp_path.rglob("training.jsonl")).read_text().splitlines()
    ]
    assert rows[-1]["train/gradient_steps"] == 1
    assert "train/actor_loss" in rows[-1]
    assert "train/edge_loss" in rows[-1]
    assert len(list(tmp_path.rglob("reconstruction_3.png"))) == 1
    assert len(list(tmp_path.rglob("eval_*.mp4"))) == 2
    results = json.loads(next(tmp_path.rglob("results.json")).read_text())
    assert results["evaluations"][-1]["step"] == 3
