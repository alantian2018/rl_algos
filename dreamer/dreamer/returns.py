import torch


def discount_weights(continuations: torch.Tensor, gamma: float) -> torch.Tensor:
    discounts = gamma * continuations
    return torch.cumprod(torch.cat([
        torch.ones_like(discounts[:, :1]), discounts[:, :-1]
    ], dim=1), dim=1)


def lambda_return(
    rewards: torch.Tensor,
    continuations: torch.Tensor,
    values: torch.Tensor,
    gamma: float,
    lambda_: float,
) -> torch.Tensor:
    T = rewards.shape[1]
    returns = []
    next_return = values[:, -1]

    for t in range(T - 1, -1, -1):
        discount = gamma * continuations[:, t]
        next_return = rewards[:, t] + discount * (
            (1 - lambda_) * values[:, t + 1] + lambda_ * next_return
        )
        returns.append(next_return)

    return torch.stack(returns[::-1], dim=1)
