from .networks import DreamerState
import torch.nn as nn
import torch
import torch.nn.functional as F


# seq model
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


# "dynamics predictor"
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


# "encoder"
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


class RSSM(nn.Module):
    def __init__(
        self, deterministic_dim, stochastic_dim, action_dim, hidden_size, latent_dim
    ):
        super().__init__()
        self.sequential_model = DeterministicDynamics(
            deterministic_dim, stochastic_dim, action_dim
        )
        self.prior = Prior(deterministic_dim, stochastic_dim, hidden_size)
        self.posterior = Posterior(
            deterministic_dim, stochastic_dim, hidden_size, latent_dim
        )

    def observe(self, prev_state: DreamerState, prev_action, latent_obs):
        h = self.sequential_model(prev_state, prev_action)

        z_prior = self.prior(h)
        z_post = self.posterior(h, latent_obs)

        z = z_post.rsample()
        state = DreamerState(h, z)
        return state, z_prior, z_post

    def imagine(self, prev_state: DreamerState, prev_action):
        h = self.sequential_model(prev_state, prev_action)
        z_prior = self.prior(h)
        z = z_prior.rsample()

        state = DreamerState(h, z)
        return state, z_prior
