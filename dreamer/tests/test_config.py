from dataclasses import asdict, dataclass

import pytest

from dreamer.dreamer.config import DreamerConfig


def config(**overrides):
    fields = dict(
        exp_name="carracing", obs_dim=(96, 96), obs_channels=3,
        action_dim=2, action_low=(-1.0, 0.0), action_high=(1.0, 1.0),
    )
    return DreamerConfig(**(fields | overrides))


def test_config_defaults_and_serialization():
    result = config()
    assert result.save_dir == "dreamer/checkpoints/carracing"
    assert result.wandb_project == "dreamer-carracing"
    assert asdict(result)["squash_actions"] is True
    assert result.continuation_weight == 10.0


def test_explicit_logging_and_checkpoint_settings_are_preserved():
    result = config(save_dir="custom", wandb_project="comparison")
    assert result.save_dir == "custom"
    assert result.wandb_project == "comparison"
    assert config(use_wandb=False).wandb_project is None


def test_experiment_dataclass_can_override_defaults():
    @dataclass
    class Experiment(DreamerConfig):
        exp_name: str = "test"
        obs_dim: tuple = (64, 64)
        obs_channels: int = 3
        action_dim: int = 1
        action_low: tuple = (-2.0,)
        action_high: tuple = (2.0,)
        batch_size: int = 4

    assert Experiment().batch_size == 4
    assert Experiment().wandb_project == "dreamer-test"


@pytest.mark.parametrize("overrides", [
    {"action_low": (-1.0,)},
    {"action_high": (-1.0, 1.0)},
    {"action_low": (float("-inf"), 0.0)},
    {"sequence_length": 0},
    {"replay_buffer_capacity": 2},
])
def test_invalid_settings_are_rejected(overrides):
    with pytest.raises(ValueError):
        config(**overrides)


def test_unbounded_actions_can_disable_squashing():
    result = config(
        squash_actions=False,
        action_low=(float("-inf"), float("-inf")),
        action_high=(float("inf"), float("inf")),
    )
    assert not result.squash_actions
