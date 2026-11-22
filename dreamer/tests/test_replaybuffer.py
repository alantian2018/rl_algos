import numpy as np
import pytest

from dreamer.dreamer.replaybuffer import ReplayBuffer


def episode(length=3, truncated=False):
    return {
        "observations": np.arange((length + 1) * 2, dtype=np.uint8).reshape(
            length + 1, 2
        ),
        "actions": np.arange(length, dtype=np.float32).reshape(length, 1),
        "rewards": np.arange(length, dtype=np.float32) + 10,
        "terminated": np.array([False] * (length - 1) + [not truncated]),
        "truncated": np.array([False] * (length - 1) + [truncated]),
    }


@pytest.mark.parametrize("truncated", [False, True])
def test_episode_preserves_final_observation_reward_and_end_flags(truncated):
    replay = ReplayBuffer()
    data = episode(truncated=truncated)
    replay.add_episode(**data)
    stored = replay.episodes[0]
    for key, value in data.items():
        np.testing.assert_array_equal(stored[key], value)
    assert replay.size == 3
    assert len(stored["observations"]) == 4
    assert stored["rewards"][-1] == 12
    assert stored["observations"].dtype == np.uint8
    assert stored["terminated"][-1] == (not truncated)
    assert stored["truncated"][-1] == truncated


def test_storage_does_not_alias_collectors_arrays():
    replay = ReplayBuffer()
    data = episode()
    replay.add_episode(**data)
    for value in data.values():
        value[...] = 0
    assert replay.episodes[0]["observations"][-1, -1] == 7
    assert replay.episodes[0]["rewards"][-1] == 12
    assert replay.episodes[0]["terminated"][-1]


def test_capacity_evicts_whole_oldest_episodes():
    replay = ReplayBuffer(capacity=5)
    replay.add_episode(**episode(2))
    replay.add_episode(**episode(3))
    assert replay.size == 5
    replay.add_episode(**episode(4, truncated=True))
    assert replay.size == 4
    assert len(replay.episodes) == 1
    assert replay.episodes[0]["truncated"][-1]


@pytest.mark.parametrize(
    "problem",
    [
        "missing_final_obs",
        "early_done",
        "unfinished",
        "reward_shape",
        "action_shape",
        "oversized",
    ],
)
def test_invalid_episode_does_not_modify_replay(problem):
    replay = ReplayBuffer(capacity=5)
    replay.add_episode(**episode(2))
    data = episode(6 if problem == "oversized" else 3)
    if problem == "missing_final_obs":
        data["observations"] = data["observations"][:-1]
    elif problem == "early_done":
        data["terminated"][0] = True
    elif problem == "unfinished":
        data["terminated"][-1] = False
    elif problem == "reward_shape":
        data["rewards"] = data["rewards"][:, None]
    elif problem == "action_shape":
        data["actions"] = data["actions"][:, 0]
    with pytest.raises(ValueError):
        replay.add_episode(**data)
    assert replay.size == 2
    assert len(replay.episodes) == 1


def test_one_transition_episode_is_valid():
    replay = ReplayBuffer(capacity=1)
    replay.add_episode(**episode(1))
    assert replay.size == 1
    assert len(replay.episodes[0]["observations"]) == 2


@pytest.mark.parametrize("truncated", [False, True])
def test_full_episode_sample_aligns_terminal_targets(truncated):
    replay = ReplayBuffer(seed=4)
    data = episode(3, truncated)
    replay.add_episode(**data)
    batch = replay.sample(2, 3)
    assert batch["observations"].shape == (2, 3, 2)
    assert batch["actions"].shape == (2, 3, 1)
    for row in range(2):
        np.testing.assert_array_equal(
            batch["observations"][row], data["observations"][:-1]
        )
        np.testing.assert_array_equal(batch["actions"][row], data["actions"])
        for key in ("rewards", "terminated", "truncated"):
            assert batch[key].shape == (2, 3, 1)
            np.testing.assert_array_equal(batch[key][row, :, 0], data[key])
        np.testing.assert_array_equal(
            batch["next_observations"][row], data["observations"][1:]
        )
    assert "dones" not in batch
    assert "prediction_mask" not in batch


def test_windows_are_contiguous_and_never_cross_episodes():
    replay = ReplayBuffer(seed=10)
    for offset, length in ((0, 2), (50, 5)):
        data = episode(length)
        data["observations"] += offset
        data["actions"] += offset
        data["rewards"] += offset
        replay.add_episode(**data)
    batch = replay.sample(100, 2)
    starts = batch["observations"][:, 0, 0]
    assert (starts == 0).any()
    assert (starts >= 50).any()
    for row, start in enumerate(starts):
        offset = 0 if start < 50 else 50
        transition = (int(start) - offset) // 2
        np.testing.assert_array_equal(np.diff(batch["observations"][row, :, 0]), [2])
        np.testing.assert_array_equal(
            batch["actions"][row, :, 0], np.arange(transition, transition + 2) + offset
        )
        np.testing.assert_array_equal(
            batch["rewards"][row, :, 0],
            np.arange(transition, transition + 2) + offset + 10,
        )


def test_sampling_skips_short_episodes_and_returns_independent_arrays():
    replay = ReplayBuffer(seed=1)
    replay.add_episode(**episode(1))
    data = episode(3)
    replay.add_episode(**data)
    batch = replay.sample(1, 3)
    np.testing.assert_array_equal(batch["observations"][0], data["observations"][:-1])
    for value in batch.values():
        value[...] = 0
    np.testing.assert_array_equal(
        replay.episodes[1]["observations"], data["observations"]
    )
    np.testing.assert_array_equal(replay.episodes[1]["rewards"], data["rewards"])


@pytest.mark.parametrize("batch_size, length", [(0, 2), (1, 0), (1, 5)])
def test_invalid_or_unavailable_sample_is_rejected(batch_size, length):
    replay = ReplayBuffer()
    replay.add_episode(**episode(2))
    with pytest.raises(ValueError):
        replay.sample(batch_size, length)


def test_empty_replay_sample_is_rejected():
    with pytest.raises(ValueError, match="no episode"):
        ReplayBuffer().sample(1, 2)
