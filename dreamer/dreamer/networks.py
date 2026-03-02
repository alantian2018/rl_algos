# encoder/decoder
# dynamics
# recurrent
# prior, posterior
# state = [h_t, z_t]

# actor/critic
import torch.nn as nn
import torch
from dataclasses import dataclass
from torch.nn import functional as F


class DreamerState:
    def __init__(self, deterministic: torch.Tensor, stochastic: torch.Tensor):
        self.deterministic = deterministic
        self.stochastic = stochastic

    def get_state(self):
        return torch.cat([self.deterministic, self.stochastic], dim=-1)


class Critic(nn.Module):
    def __init__(self, deterministic_dim, stochastic_dim, hidden_size):
        super().__init__()
        nn.Sequential(
            nn.Linear(deterministic_dim + stochastic_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, dreamer_state: DreamerState):
        state = torch.cat(
            [dreamer_state.deterministic, dreamer_state.stochastic], dim=-1
        )
        return self.net(state)


class Actor(nn.Module):
    def __init__(self, deterministic_dim, stochastic_dim, hidden_size, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(deterministic_dim + stochastic_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
        )

    def forward(self, dreamer_state: DreamerState):
        state = torch.cat(
            [dreamer_state.deterministic, dreamer_state.stochastic], dim=-1
        )
        return self.net(state)


class DeterministicDynamics(nn.Module):
    def __init__(self, deterministic_dim, stochastic_dim, action_dim):
        super().__init__()
        self.gru = nn.GRU(
            input_size=stochastic_dim + action_dim,
            hidden_size=deterministic_dim,
            batch_first=True,
        )

    def forward(self, state: DreamerState, action):
        gru_input = torch.cat([state.stochastic, action], dim=-1).unsqueeze(1)
        h_prev = state.deterministic.unsqueeze(0)  # (1,B,H)

        _, h_next = self.gru(gru_input, h_prev)
        return h_next.squeeze(0)


class FeatureExtractor(nn.Module):
    def __init__(self, deterministic_dim, hidden_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(deterministic_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

    def forward(self, h_t):
        return self.net(h_t)


class Prior(nn.Module):
    def __init__(self, deterministic_dim, stochastic_dim, hidden_size):
        super().__init__()
        # s_t-1 + a_t-1 -> z_t
        self.feature_extractor = FeatureExtractor(deterministic_dim, hidden_size)
        # z_t -> mu, stdev
        self.mu_net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, stochastic_dim),
        )
        self.stdev_net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, stochastic_dim),
        )

    def forward(self, h_t):
        feature = self.feature_extractor(h_t)
        mu = self.mu_net(feature)
        stdev = self.stdev_net(feature)
        stdev = torch.clamp(F.softplus(stdev) + 0.1, max=10.0)
        return torch.distributions.Normal(mu, stdev)


class Posterior(nn.Module):
    def __init__(self, deterministic_dim, stochastic_dim, hidden_size, latent_dim):
        super().__init__()
        # s_t -> z_t
        self.feature_extractor = FeatureExtractor(deterministic_dim, hidden_size)
        # z_t -> mu, stdev
        self.mu_net = nn.Sequential(
            nn.Linear(hidden_size + latent_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, stochastic_dim),
        )
        self.stdev_net = nn.Sequential(
            nn.Linear(hidden_size + latent_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, stochastic_dim),
        )

    def forward(self, h_t, latent_obs_t):
        feature = self.feature_extractor(h_t)
        mu = self.mu_net(torch.cat([feature, latent_obs_t], dim=-1))
        stdev = self.stdev_net(torch.cat([feature, latent_obs_t], dim=-1))
        stdev = torch.clamp(F.softplus(stdev) + 0.1, max=10.0)
        return torch.distributions.Normal(mu, stdev)


class RewardDecoder(nn.Module):
    def __init__(self, deterministic_dim, stochastic_dim, hidden_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(deterministic_dim + stochastic_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, dreamer_state_t: DreamerState):
        state = dreamer_state_t.get_state()
        return self.net(state)


# encoder / decoder
