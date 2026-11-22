from collections import deque

import numpy as np


class ReplayBuffer:
    """Store complete episodes; capacity and size count environment transitions."""

    def __init__(self, capacity=100_000, seed=None):
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self.episodes = deque()
        self.size = 0
        self.rng = np.random.default_rng(seed)

    def add_episode(self, observations, actions, rewards, terminated, truncated):
        """Store T transitions and T+1 observations, including the final one.

        actions[t] takes observations[t] to observations[t+1], yielding rewards[t],
        terminated[t], and truncated[t]. Observations must not include the next
        episode's reset observation. Scalar transition fields have shape [T];
        actions have shape [T, action_dim]. Oldest whole episodes are evicted
        when capacity is exceeded. An episode larger than capacity is rejected.
        """
        episode = {
            "observations": np.array(observations, copy=True),
            "actions": np.array(actions, dtype=np.float32, copy=True),
            "rewards": np.array(rewards, dtype=np.float32, copy=True),
            "terminated": np.array(terminated, dtype=np.bool_, copy=True),
            "truncated": np.array(truncated, dtype=np.bool_, copy=True),
        }
        actions = episode["actions"]
        if actions.ndim != 2 or len(actions) == 0:
            raise ValueError("actions must have shape [T, action_dim] with T > 0")
        length = len(actions)
        if length > self.capacity:
            raise ValueError("episode exceeds replay capacity")
        if episode["observations"].ndim < 2 or len(episode["observations"]) != length + 1:
            raise ValueError("observations must have shape [T+1, *obs_shape]")
        for key in ("rewards", "terminated", "truncated"):
            if episode[key].shape != (length,):
                raise ValueError(f"{key} must have shape [T]")
        episode_ends = episode["terminated"] | episode["truncated"]
        if episode_ends[:-1].any() or not episode_ends[-1]:
            raise ValueError("only the final transition may end a complete episode")

        while self.size + length > self.capacity:
            self.size -= len(self.episodes.popleft()["actions"])
        self.episodes.append(episode)
        self.size += length

    def sample(self, batch_size, sequence_length):
    
        if batch_size < 1 or sequence_length < 1:
            raise ValueError("batch_size must be positive and sequence_length positive")
        episodes = list(self.episodes)
        counts = np.array([
            max(0, len(item["actions"]) - sequence_length + 1)
            for item in episodes
        ], dtype=np.int64)
        cumulative = counts.cumsum()
        if len(cumulative) == 0 or cumulative[-1] == 0:
            raise ValueError("no episode is long enough for sequence_length")

        batch = {key: [] for key in (
            "observations", "actions", "rewards", "next_observations", "terminated", "truncated"
        )}
        for window in self.rng.integers(cumulative[-1], size=batch_size):
            index = int(np.searchsorted(cumulative, window, side="right"))
            start = int(window - (cumulative[index - 1] if index else 0))
            stop = start + sequence_length
            item = episodes[index]
            batch["observations"].append(item["observations"][start:stop])
            batch["next_observations"].append(item["observations"][start + 1:stop + 1])
            batch["actions"].append(item["actions"][start:stop])
            for key in ("rewards", "terminated", "truncated"):
                batch[key].append(item[key][start:stop, None])

        batch = {key: np.stack(values) for key, values in batch.items()}
        return batch
