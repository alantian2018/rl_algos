import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.nn import Sequential, Conv2d, ReLU

MAX_LOG_STD = 2
MIN_LOG_STD = -20


class BaseAction(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        action_dim: int,
        action_low: list[int],
        action_high: list[int],
    ):
        super().__init__()
        self.mu = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.log_std = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

        self.register_buffer(
            "action_low", torch.tensor(action_low, dtype=torch.float32)
        )
        self.register_buffer(
            "action_high", torch.tensor(action_high, dtype=torch.float32)
        )
        self.action_dim = action_dim

    def forward(self, x):
        mu = self.mu(x)
        log_std = torch.tanh(self.log_std(x))
        log_std = MIN_LOG_STD + 0.5 * (MAX_LOG_STD - MIN_LOG_STD) * (log_std + 1)
        std = torch.exp(log_std)
        return Normal(mu, std)

    def get_action(self, x):
        dist = self.forward(x)
        u = dist.rsample()

        action = torch.tanh(u)
        action_scaled = (action + 1) / 2 * (
            self.action_high - self.action_low
        ) + self.action_low
        log_probs = dist.log_prob(u)
        log_probs -= torch.log(1 - action.pow(2) + 1e-6)
        log_probs -= torch.log((self.action_high - self.action_low) / 2.0)
        log_probs = log_probs.sum(dim=-1, keepdim=True)
        return action_scaled, log_probs


class Encoder(nn.Module):
    def __init__(self, output_dim: int):
        super().__init__()
        self.output_dim = output_dim

    def forward(self, x):
        raise NotImplementedError


class MLPEncoder(Encoder):
    def __init__(self, state_dim: int, hidden_dim: int):
        super().__init__(output_dim=hidden_dim)
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, state):
        return self.encoder(state)


class CNNEncoder(Encoder):
    def __init__(self, in_channels: int, height: int, width: int, hidden_dim: int = 64):
        super().__init__(output_dim=hidden_dim)
        self.encoder = nn.Sequential(
            Conv2d(in_channels, 32, kernel_size=8, stride=4),
            ReLU(),
            Conv2d(32, 64, kernel_size=4, stride=2),
            ReLU(),
            Conv2d(64, 64, kernel_size=3, stride=1),
            ReLU(),
            nn.Flatten(),
        )

        # Calculate flattened size
        with torch.no_grad():
            sample = torch.zeros(1, in_channels, height, width)
            flattened_size = self.encoder(sample).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(flattened_size, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, state):
        if state.ndim == 3:
            state = state.unsqueeze(0)
        state = state.permute(0, 3, 1, 2)
        x = self.encoder(state)
        return self.fc(x)


class Policy(nn.Module):
    def __init__(
        self,
        encoder,
        action_dim=None,
        action_head=None,
        action_low=None,
        action_high=None,
    ):
        super().__init__()
        self.encoder = encoder

        assert action_dim is not None or action_head is not None

        if action_head is None:
            self.action_head = BaseAction(
                hidden_dim=encoder.output_dim,
                action_dim=action_dim,
                action_low=action_low,
                action_high=action_high,
            )
        else:
            self.action_head = action_head

    def forward(self, state):
        embeddings = self.encoder(state)
        return self.action_head(embeddings)

    def get_action(self, state):
        embeddings = self.encoder(state)
        return self.action_head.get_action(embeddings)


class Qfunction(nn.Module):
    def __init__(self, encoder: Encoder, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.encoder = encoder
        self.q_head = nn.Sequential(
            nn.Linear(encoder.output_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor):
        state_emb = self.encoder(state)
        return self.q_head(torch.cat([state_emb, action], dim=-1))
