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

    def flatten(self):
        return DreamerState(
            self.deterministic.reshape(-1, self.deterministic.shape[-1]),
            self.stochastic.reshape(-1, self.stochastic.shape[-1]),
        )


class Critic(nn.Module):
    def __init__(self, deterministic_dim, stochastic_dim, hidden_size):
        super().__init__()
        self.net = nn.Sequential(
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
    def __init__(
        self,
        deterministic_dim,
        stochastic_dim,
        hidden_size,
        action_dim,
        action_low=None,
        action_high=None,
        squash_actions=True,
    ):
        super().__init__()
        self.squash_actions = squash_actions
        low = (
            torch.full((action_dim,), -1.0)
            if action_low is None
            else torch.tensor(action_low, dtype=torch.float32)
        )
        high = (
            torch.full((action_dim,), 1.0)
            if action_high is None
            else torch.tensor(action_high, dtype=torch.float32)
        )
        if low.shape != (action_dim,) or high.shape != (action_dim,):
            raise ValueError("action bounds must have shape [action_dim]")
        if not (low < high).all():
            raise ValueError("action_low must be less than action_high")
        if squash_actions and not (
            torch.isfinite(low).all() and torch.isfinite(high).all()
        ):
            raise ValueError("squash_actions requires finite action bounds")
        self.register_buffer("action_low", low)
        self.register_buffer("action_high", high)
        self.net = nn.Sequential(
            nn.Linear(deterministic_dim + stochastic_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(hidden_size, action_dim)
        self.std_head = nn.Linear(hidden_size, action_dim)

    def forward(self, dreamer_state: DreamerState):
        state = dreamer_state.get_state()

        features = self.net(state)
        mu = self.mu_head(features)
        std = 0.1 + 0.9 * torch.sigmoid(self.std_head(features))
        return torch.distributions.Normal(mu, std)

    def get_action(self, dreamer_state: DreamerState, deterministic=False):
        distribution = self(dreamer_state)
        raw_action = distribution.loc if deterministic else distribution.rsample()
        log_prob = distribution.log_prob(raw_action)
        action = raw_action
        if self.squash_actions:
            scale = (self.action_high - self.action_low) / 2
            action = self.action_low + (torch.tanh(raw_action) + 1) * scale
            log_jacobian = 2 * (
                torch.log(raw_action.new_tensor(2.0))
                - raw_action
                - F.softplus(-2 * raw_action)
            )
            log_prob = log_prob - log_jacobian - torch.log(scale)
        info = {
            "mu": distribution.loc.detach(),
            "std": distribution.scale.detach(),
            "raw_action": raw_action.detach(),
        }
        return action, log_prob.sum(dim=-1, keepdim=True), info

    def get_log_prob(self, dreamer_state: DreamerState, actions, raw_actions=None):
        distribution = self(dreamer_state)
        if raw_actions is None:
            raw_actions = actions.detach()
            if self.squash_actions:
                scale = (self.action_high - self.action_low) / 2
                normalized = (raw_actions - self.action_low) / scale - 1
                epsilon = torch.finfo(normalized.dtype).eps
                raw_actions = torch.atanh(normalized.clamp(-1 + epsilon, 1 - epsilon))
        else:
            raw_actions = raw_actions.detach()
        if self.squash_actions:
            scale = (self.action_high - self.action_low) / 2
            log_jacobian = 2 * (
                torch.log(raw_actions.new_tensor(2.0))
                - raw_actions
                - F.softplus(-2 * raw_actions)
            )
            log_prob = (
                distribution.log_prob(raw_actions) - log_jacobian - torch.log(scale)
            )
        else:
            log_prob = distribution.log_prob(raw_actions)
        return log_prob.sum(dim=-1, keepdim=True)


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


class ContinueDecoder(nn.Module):
    def __init__(self, deterministic_dim, stochastic_dim, hidden_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(deterministic_dim + stochastic_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, state: DreamerState):
        logits = self.net(state.get_state())
        return torch.distributions.Bernoulli(logits=logits)


# encoder / decoder
class ObservationEncoder(nn.Module):
    def __init__(self, input_size, input_channels, latent_obs_dim, hidden_dim):
        super().__init__()
        # assume its a 2d image for now...
        self.nets = nn.Sequential(
            nn.Conv2d(input_channels, 32, 5, 3),
            nn.ReLU(),
            nn.Conv2d(32, 64, 5, 2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        dummy = torch.zeros(1, input_channels, *input_size)

        with torch.no_grad():
            h1 = self.nets[:2](dummy)
            h2 = self.nets[2:4](h1)
            h3 = self.nets[4:-1](h2)

        self.h1_hw = h1.shape[-2:]
        self.conv_shape = h3.shape[1:]
        self.flat_size = h3[0].numel()

        print(f"flat_size for encoder is {self.flat_size}")
        self.proj = nn.Sequential(
            nn.Linear(self.flat_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_features=latent_obs_dim),
        )

    def forward(self, obs):
        flat = self.nets(obs)
        return self.proj(flat)


class ObservationDecoder(nn.Module):
    def __init__(
        self, encoder, input_size, input_channels, hidden_state_size, hidden_dim
    ):
        super().__init__()

        self.conv_shape = encoder.conv_shape
        self.h1_hw = encoder.h1_hw
        self.input_size = input_size
        self.input_channels = input_channels

        self.proj = nn.Sequential(
            nn.Linear(hidden_state_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, encoder.flat_size),
            nn.ReLU(),
        )

        self.deconv0 = nn.ConvTranspose2d(64, 64, 3, 1, padding=1)
        self.deconv1 = nn.ConvTranspose2d(64, 64, 3, 1)
        self.deconv2 = nn.ConvTranspose2d(64, 32, 5, 2)
        self.deconv3 = nn.ConvTranspose2d(32, input_channels, 5, 3)

    def forward(self, z):
        B = z.shape[0]

        x = self.proj(z)
        x = x.reshape(B, *self.conv_shape)

        x = F.relu(self.deconv0(x))
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x, output_size=(B, 32, *self.h1_hw)))
        x = self.deconv3(x, output_size=(B, self.input_channels, *self.input_size))

        return x


class ObservationHandler(nn.Module):
    def __init__(
        self, input_size, input_channels, latent_obs_dim, hidden_state_size, hidden_dim
    ):
        super().__init__()
        self.encoder = ObservationEncoder(
            input_size, input_channels, latent_obs_dim, hidden_dim
        )
        self.decoder = ObservationDecoder(
            self.encoder, input_size, input_channels, hidden_state_size, hidden_dim
        )

    def encode(self, obs):
        return self.encoder(obs)

    def decode(self, state: DreamerState):
        return self.decoder(state.get_state())


# posterior -> what it should have predicted with observatoin
# prior -> what it did predict without observation, should be learned to be the same as posterior
if __name__ == "__main__":
    pass
