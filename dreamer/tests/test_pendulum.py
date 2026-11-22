import gymnasium
import numpy as np
import pytest
import torch

from dreamer.experiments.pendulum import PendulumConfig, PixelPendulum
from dreamer.dreamer.worldmodel import WorldModel


def test_pixel_observations_hide_torque_arrow_but_render_keeps_it():
    env = PixelPendulum()
    try:
        env.reset(seed=4)
        obs, *_ = env.step(np.array([2.0], dtype=np.float32))
        last_u = env.unwrapped.last_u
        with_arrow = env.render()
        env.unwrapped.last_u = None
        without_arrow = env.render()
        np.testing.assert_array_equal(obs, env.pixels())
        env.unwrapped.last_u = last_u
        assert np.any(with_arrow != without_arrow)
        np.testing.assert_array_equal(env.pixels(), obs)
        assert env.unwrapped.last_u == last_u
        np.testing.assert_array_equal(env.render(), with_arrow)
    finally:
        env.close()


def test_pixel_observations_restore_torque_after_render_failure(monkeypatch):
    env = PixelPendulum()
    try:
        env.reset(seed=4)
        env.step(np.array([2.0], dtype=np.float32))
        last_u = env.unwrapped.last_u

        def fail_render():
            assert env.unwrapped.last_u is None
            raise RuntimeError("render failed")

        monkeypatch.setattr(env.env, "render", fail_render)
        with pytest.raises(RuntimeError, match="render failed"):
            env.pixels()
        assert env.unwrapped.last_u == last_u
    finally:
        env.close()


def test_pixel_pendulum_returns_images_and_preserves_transition_rewards():
    env = PixelPendulum(action_repeat=2, reward_scale=0.1)
    reference = gymnasium.make("Pendulum-v1")
    try:
        obs, _ = env.reset(seed=4)
        reference.reset(seed=4)
        assert obs.shape == (64, 64, 1)
        assert obs.dtype == np.uint8
        assert env.observation_space.contains(obs)
        action = np.array([0.5], dtype=np.float32)
        expected_reward = 0.0
        for _ in range(2):
            _, reward, _, _, _ = reference.step(action)
            expected_reward += reward
        obs, reward, terminated, truncated, info = env.step(action)
        assert env.observation_space.contains(obs)
        assert not terminated and not truncated
        np.testing.assert_allclose(reward, expected_reward * 0.1)
        np.testing.assert_allclose(info["raw_reward"], expected_reward)
        for _ in range(99):
            obs, _, terminated, truncated, _ = env.step(action)
        assert truncated and not terminated
        assert env.observation_space.contains(obs)
    finally:
        env.close()
        reference.close()


def test_pendulum_defaults_match_pixel_environment():
    config = PendulumConfig()
    assert config.obs_dim == (64, 64)
    assert config.obs_channels == 1
    assert config.action_low == (-2.0,)
    assert config.action_high == (2.0,)
    assert config.frame_stack == 1


def test_world_model_applies_configured_weights_and_free_nats():
    model = WorldModel(16, 8, 1, (64, 64), 1, 12, 32, (0, 0, 0, 2), 3)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    obs = torch.zeros(1, 2, 1, 64, 64)
    flags = torch.zeros(1, 2, 1, dtype=torch.bool)
    loss, metrics, _ = model.loss(obs, torch.zeros(1, 2, 1), torch.zeros(1, 2, 1), flags, flags, obs)
    torch.testing.assert_close(metrics["raw_kl"], torch.tensor(0.0))
    torch.testing.assert_close(metrics["kl_loss"], torch.tensor(3.0))
    torch.testing.assert_close(loss, torch.tensor(6.0))


def test_pixel_experiment_saves_reconstruction_images(tmp_path):
    import json
    from dreamer.experiments.pendulum import main

    config = PendulumConfig(
        device="cpu", use_wandb=False, save_dir=str(tmp_path), save_freq=None,
        total_train_steps=1, before_training_steps=0, train_steps_per_episode=1,
        sequence_length=3, batch_size=1, imagination_horizon=2,
        deterministic_dim=16, stochastic_dim=8, latent_obs_size=12, hidden_dim=32,
        eval_episodes=1, video_log_freq=None, reconstruction_log_freq=1,
        num_threads=1,
    )
    main.__wrapped__(config)
    assert len(list(tmp_path.rglob("reconstruction_100.png"))) == 1
    results = json.loads(next(tmp_path.rglob("results.json")).read_text())
    assert "random" not in results
    assert results["evaluations"][-1]["step"] == 100
