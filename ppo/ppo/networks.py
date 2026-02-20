import torch
from torch.nn import Module, Linear, ReLU, Sequential, Conv2d, Flatten
from torch.distributions import Categorical


class Actor(Module):
    """Actor network -> policy pi(a|s)"""

    def __init__(
        self, obs_dim: int, act_dim: int, hidden_size: int, act_shape: int = 1
    ):
        super().__init__()
        self.net = Sequential(
            Linear(obs_dim, hidden_size),
            ReLU(),
            Linear(hidden_size, hidden_size),
            ReLU(),
            Linear(hidden_size, act_shape * act_dim),
        )
        self.act_shape = act_shape
        self.act_dim = act_dim

    def forward(self, obs: torch.Tensor) -> Categorical:
        if obs.dim == 1:
            obs = obs.unsqueeze(0)
        logits = self.net(obs)
        if self.act_shape > 1:
            logits = logits.view(*logits.shape[:-1], self.act_shape, self.act_dim)
        return Categorical(logits=logits)


class Critic(Module):
    """Critic network -> value function V(s)"""

    def __init__(self, obs_dim: int, hidden_size: int):
        super().__init__()
        self.net = Sequential(
            Linear(obs_dim, hidden_size),
            ReLU(),
            Linear(hidden_size, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.dim == 1:
            obs = obs.unsqueeze(0)

        return self.net(obs)


class SnakeActor(Module):
    """CNN Actor network -> policy pi(a|s) for image observations."""

    def __init__(
        self,
        in_channels: int,
        height: int,
        width: int,
        act_dim: int,
        hidden_size: int,
        act_shape: int = 1,
    ):
        super().__init__()
        self.conv = Sequential(
            Conv2d(in_channels, hidden_size, kernel_size=3, stride=1, padding=1),
            ReLU(),
            Conv2d(hidden_size, hidden_size, kernel_size=3, stride=1, padding=1),
            ReLU(),
        )
        # After conv with same padding, spatial dims preserved
        self.fc = Sequential(
            Flatten(),
            Linear(hidden_size * height * width, hidden_size),
            ReLU(),
            Linear(hidden_size, act_dim * act_shape),
        )
        self.act_shape = act_shape
        self.act_dim = act_dim

    def forward(self, obs: torch.Tensor) -> Categorical:
        # obs: (batch, H, W, C) -> (batch, C, H, W)
        if obs.dim() == 3:
            obs = obs.unsqueeze(0)  # add batch dim
        obs = obs.permute(0, 3, 1, 2)  # HWC -> CHW
        x = self.conv(obs)
        logits = self.fc(x)
        if self.act_shape > 1:
            logits = logits.view(*logits.shape[:-1], self.act_shape, self.act_dim)
        return Categorical(logits=logits)


class SnakeCritic(Module):
    """CNN Critic network -> value function V(s) for image observations."""

    def __init__(self, in_channels: int, height: int, width: int, hidden_size: int):
        super().__init__()
        self.conv = Sequential(
            Conv2d(in_channels, hidden_size, kernel_size=3, stride=1, padding=1),
            ReLU(),
            Conv2d(hidden_size, hidden_size, kernel_size=3, stride=1, padding=1),
            ReLU(),
        )
        self.fc = Sequential(
            Flatten(),
            Linear(hidden_size * height * width, hidden_size),
            ReLU(),
            Linear(hidden_size, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # obs: (batch, H, W, C) -> (batch, C, H, W)
        if obs.dim() == 3:
            obs = obs.unsqueeze(0)
        obs = obs.permute(0, 3, 1, 2)
        x = self.conv(obs)
        return self.fc(x)


class ImageActor(Module):
    """Downsampling CNN tailored for 96x96x3 CarRacing observations."""

    def __init__(
        self,
        in_channels: int,
        height: int,
        width: int,
        act_dim: int,
        hidden_size: int,
        act_shape: int = 1,
    ):
        super().__init__()
        # aggressive downsampling: kernel/stride choices inspired by Atari-style nets
        self.conv = Sequential(
            Conv2d(in_channels, 32, kernel_size=8, stride=4),
            ReLU(),
            Conv2d(32, 64, kernel_size=4, stride=2),
            ReLU(),
            Conv2d(64, 64, kernel_size=3, stride=1),
            ReLU(),
        )

        def _conv_out_dim(size, k, s, p=0):
            return (size + 2 * p - k) // s + 1

        h = _conv_out_dim(height, 8, 4)
        h = _conv_out_dim(h, 4, 2)
        h = _conv_out_dim(h, 3, 1)
        w = _conv_out_dim(width, 8, 4)
        w = _conv_out_dim(w, 4, 2)
        w = _conv_out_dim(w, 3, 1)

        flattened = 64 * h * w

        self.fc = Sequential(
            Flatten(),
            Linear(flattened, hidden_size),
            ReLU(),
            Linear(hidden_size, act_dim * act_shape),
        )
        self.act_shape = act_shape
        self.act_dim = act_dim

    def forward(self, obs: torch.Tensor) -> Categorical:
        if obs.dim() == 3:
            obs = obs.unsqueeze(0)
        obs = obs.permute(0, 3, 1, 2)  # HWC -> CHW
        x = self.conv(obs)
        logits = self.fc(x)
        if self.act_shape > 1:
            logits = logits.view(*logits.shape[:-1], self.act_shape, self.act_dim)
        return Categorical(logits=logits)


class ImageCritic(Module):
    """Downsampling CNN critic for 96x96x3 CarRacing observations."""

    def __init__(self, in_channels: int, height: int, width: int, hidden_size: int):
        super().__init__()
        self.conv = Sequential(
            Conv2d(in_channels, 32, kernel_size=8, stride=4),
            ReLU(),
            Conv2d(32, 64, kernel_size=4, stride=2),
            ReLU(),
            Conv2d(64, 64, kernel_size=3, stride=1),
            ReLU(),
        )

        def _conv_out_dim(size, k, s, p=0):
            return (size + 2 * p - k) // s + 1

        h = _conv_out_dim(height, 8, 4)
        h = _conv_out_dim(h, 4, 2)
        h = _conv_out_dim(h, 3, 1)
        w = _conv_out_dim(width, 8, 4)
        w = _conv_out_dim(w, 4, 2)
        w = _conv_out_dim(w, 3, 1)

        flattened = 64 * h * w

        self.fc = Sequential(
            Flatten(),
            Linear(flattened, hidden_size),
            ReLU(),
            Linear(hidden_size, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.dim() == 3:
            obs = obs.unsqueeze(0)
        obs = obs.permute(0, 3, 1, 2)
        x = self.conv(obs)
        return self.fc(x)
