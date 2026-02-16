"""
Generalized Advantage Estimation (GAE)

Estimates the advantage function A(s,a) = Q(s,a) - V(s)
where Q(s,a) is not directly available but V(s) is available via critic network.

Q(s,a) can be broken into r(s,a) + gamma * V(s_t+1). Then:
A(s,a) = Q(s,a) - V(s) = r(s,a) + gamma * V(s_t+1) - V(s)
"""

import torch

"""
think of delta as how surprised our critic is at timestep t. 
ie, the goal of the critic is to discover advantageous moves.
we quantify this via delta for one timestep.

Consider 
    a perfect critic, then delta = 0 for all timesteps.
        Then, we don't need to propogate a training signal to actor since we are already perfect.  
        Theoretically, this is degenerate since actor could be really bad,
        but in practice, the actor will always discover acions that lead to better rewards than expected. 
    a random critic, then delta will be positive or negative sometimes
        we want to penalize the critic for being wrong and push actor towards the better 
        than expected states (via higher advantage)
"""


def delta(
    reward: float, current_value: torch.Tensor, next_value: torch.Tensor, gamma: float
) -> torch.Tensor:
    """Temporal difference error."""
    return reward + gamma * next_value - current_value


"""""
But how do we know which moves are actually giving us the advantage? 
Clearly it isn't just the immediate move in most cases, but a sequence of moves.

We need to somehow assign credit future timesteps into current calculation.

GAE formula:
A_t = delta_t + gamma * lambda * delta_t+1 + gamma^2 * lambda^2 * delta_t+2 + ...
    = delta_t + gamma * lambda * A_t+1

    Remember: delta = where the critic V(s_t) discovers advantage
        
Where:
    delta = how wrong the critic V(s_t) is at timestep t
    gamma = discount factor for future terms similar to V
    lambda = do i trust my critic (bias) or trajectory (variance)
        low lambda: I only trust my critic's immediate estimate.
                    Unfortunately, if the critic is erroneous,
                    (ie your move is bad but critic gives you high advantage),
                    the actor will blindly trust it. 
        lambda = 1: trust future observed rewards. You look at actual rewards from trajectory in
                    future timesteps, which smoothes out catastrophic critic estimates. 
                    However, the trajectory may achieve high advantage
                    not because of your move, but because of blind luck.
        lambda tells us we need to trust our critic (tries to filter out the luck),
               but also trust the future trajectory (tells us if our move is actually bad).
"""


def gae(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    values: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> torch.Tensor:
    """Compute Generalized Advantage Estimation."""
    T = len(rewards)
    advantages = torch.zeros_like(rewards)

    for t in range(T - 1, -1, -1):
        if t == T - 1 or dones[t]:
            # No future value for last timestep or terminal states
            advantages[t] = delta(
                reward=rewards[t],
                current_value=values[t],
                next_value=0,
                gamma=gamma,
            )
        else:
            current_delta = delta(
                reward=rewards[t],
                current_value=values[t],
                next_value=values[t + 1],
                gamma=gamma,
            )
            # A_t = delta_t + gamma * lambda * A_t+1
            advantages[t] = current_delta + gamma * gae_lambda * advantages[t + 1]

    return advantages
