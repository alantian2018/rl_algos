from collections import deque
import numpy as np
from dataclasses import dataclass
import torch

@dataclass
class Step:
    obs: np.ndarray | torch.Tensor | list
    action: np.ndarray 
    reward: float
    next_obs: np.ndarray | torch.Tensor | list
    terminated: bool
    truncated: bool

class ReplayBuffer:
    def __init__(self, capacity: int = 1_000_000):
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity

    def add(self, step: Step): 
        self.buffer.append(step)

    def sample(self, size: int):
        if len(self.buffer) < size:
            raise ValueError('Please add more data')
            
        indices = np.random.choice( range(len(self.buffer)), size=size, replace=False)
        obs_out = np.array([self.buffer[i].obs for i in indices])
        action_out = np.array([self.buffer[i].action for i in indices])
        reward_out = np.array([self.buffer[i].reward for i in indices])
        next_obs_out = np.array([self.buffer[i].next_obs for i in indices])
        terminated_out = np.array([self.buffer[i].terminated for i in indices])
        truncated_out = np.array([self.buffer[i].truncated for i in indices])

        return (
            obs_out, 
            action_out, 
            reward_out, 
            next_obs_out, 
            terminated_out, 
            truncated_out,
            indices
        )

    def __len__(self):
        return len(self.buffer)
