from types import SimpleNamespace

import numpy as np
import pytest
import torch

from dreamer.dreamer.replaybuffer import ReplayBuffer
from dreamer.dreamer.visualization import make_reconstruction_grid, reconstruct_replay
from dreamer.dreamer.worldmodel import WorldModel


@pytest.mark.parametrize("channels", [1, 3])
def test_grid_places_targets_above_reconstructions(channels):
    targets = torch.stack(
        [torch.zeros(channels, 2, 3), torch.full((channels, 2, 3), 0.25)]
    )
    predictions = torch.stack(
        [torch.full((channels, 2, 3), 2.0), torch.full((channels, 2, 3), 0.5)]
    )
    grid = make_reconstruction_grid(targets, predictions)
    assert grid.shape == ((4, 6) if channels == 1 else (4, 6, 3))
    assert grid.dtype == np.uint8
    assert (grid[:2, :3] == 0).all()
    assert (grid[:2, 3:] == 64).all()
    assert (grid[2:, :3] == 255).all()
    assert (grid[2:, 3:] == 128).all()


def test_replay_reconstruction_uses_next_frames_and_preserves_training_state():
    model = WorldModel(16, 8, 1, (64, 64), 1, 12, 32)
    replay = ReplayBuffer(seed=1)
    agent = SimpleNamespace(
        world_model=model,
        replay_buffer=replay,
        config=SimpleNamespace(sequence_length=3),
    )
    assert reconstruct_replay(agent) is None
    images = np.stack(
        [np.full((1, 64, 64), value, dtype=np.uint8) for value in (0, 50, 100, 150)]
    )
    replay.add_episode(
        images,
        np.zeros((3, 1)),
        np.zeros(3),
        np.array([False, False, True]),
        np.zeros(3, dtype=bool),
    )
    rng = torch.get_rng_state().clone()
    replay_rng = repr(replay.rng.bit_generator.state)
    grid = reconstruct_replay(agent, num_frames=2)
    assert grid.shape == (128, 128)
    assert (grid[:64, :64] == 50).all()
    assert (grid[:64, 64:] == 150).all()
    torch.testing.assert_close(torch.get_rng_state(), rng)
    assert repr(replay.rng.bit_generator.state) == replay_rng
    assert model.training
    assert all(parameter.grad is None for parameter in model.parameters())
