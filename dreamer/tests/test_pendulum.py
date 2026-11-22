import gymnasium
import numpy as np
import torch

from dreamer.experiments.pendulum import PendulumConfig, PixelPendulum
from dreamer.dreamer.worldmodel import WorldModel


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
