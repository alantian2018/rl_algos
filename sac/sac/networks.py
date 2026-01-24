import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal

class Qfunction(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.action_dim = action_dim

        self.model = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor):
        return self.model(torch.cat([state, action], dim=-1))
    
    

MAX_LOG_STD = 2
MIN_LOG_STD = -20

class Policy(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, action_low = 0, action_high = 1, hidden_dim: int = 64, ):
        super().__init__()
        self.extract_embeddings = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(), )
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
        self.action_low = action_low
        self.action_high = action_high
        self.action_dim = action_dim
     

    def forward(self, state: torch.Tensor):
        x = self.extract_embeddings(state)
        mu = self.mu(x)
        log_std = torch.tanh(self.log_std(x))
        log_std = MIN_LOG_STD + 1/2 * (MAX_LOG_STD - MIN_LOG_STD) * (log_std+1)
        std = torch.exp(log_std)
        return Normal(mu, std)

    def get_action(self, state: torch.Tensor):
        dist = self.forward(state)
        u = dist.rsample()
        action = torch.tanh(u)
        action_scaled = (action + 1) / 2 * (self.action_high - self.action_low) + self.action_low

        log_probs = dist.log_prob(u)
        log_probs -= torch.log(1 - action.pow(2) + 1e-6)
        
        log_probs = torch.reshape(log_probs, (-1, self.action_dim))
        
        return action_scaled, log_probs.sum(dim=1, keepdim=True)
        