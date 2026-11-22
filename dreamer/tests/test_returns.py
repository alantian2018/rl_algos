import pytest
import torch

from dreamer.dreamer.returns import lambda_return


def test_single_transition_keeps_final_bootstrap():
    returns = lambda_return(
        torch.tensor([[[2.0]]]),
        torch.tensor([[[0.25]]]),
        torch.tensor([[[1.0], [4.0]]]),
        0.5,
        0.95,
    )
    torch.testing.assert_close(returns, torch.tensor([[[2.5]]]))


def test_continuation_discounts_bootstrap_and_recursion():
    returns = lambda_return(
        torch.tensor([[[1.0], [3.0]]]),
        torch.tensor([[[1.0], [0.5]]]),
        torch.tensor([[[1.0], [2.0], [4.0]]]),
        0.9,
        0.8,
    )
    torch.testing.assert_close(returns, torch.tensor([[[4.816], [4.8]]]))


def test_zero_continuation_stops_future_rewards_and_values():
    returns = lambda_return(
        torch.tensor([[[2.0], [100.0]], [[3.0], [200.0]]]),
        torch.tensor([[[0.0], [1.0]], [[0.0], [1.0]]]),
        torch.tensor([[[1.0], [20.0], [30.0]], [[1.0], [40.0], [50.0]]]),
        0.99,
        0.95,
    )
    torch.testing.assert_close(returns[:, 0], torch.tensor([[2.0], [3.0]]))


def test_lambda_one_matches_discounted_bootstrapped_returns():
    rewards = torch.tensor([[[1.0], [2.0]]])
    values = torch.tensor([[[7.0], [8.0], [12.0]]])
    returns = lambda_return(rewards, torch.tensor([[[1.0], [0.5]]]), values, 0.5, 1.0)
    torch.testing.assert_close(returns, torch.tensor([[[3.5], [5.0]]]))


def test_lambda_zero_matches_one_step_targets():
    rewards = torch.tensor([[[1.0], [2.0]]])
    values = torch.tensor([[[7.0], [8.0], [12.0]]])
    returns = lambda_return(rewards, torch.tensor([[[1.0], [0.5]]]), values, 0.5, 0.0)
    torch.testing.assert_close(returns, torch.tensor([[[5.0], [5.0]]]))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_return_gradients_reach_rewards_continuations_and_bootstrap(dtype):
    rewards = torch.tensor([[[1.0], [2.0]]], dtype=dtype, requires_grad=True)
    continuations = torch.tensor([[[0.8], [0.4]]], dtype=dtype, requires_grad=True)
    values = torch.tensor([[[7.0], [8.0], [12.0]]], dtype=dtype, requires_grad=True)
    returns = lambda_return(rewards, continuations, values, 0.5, 1.0)
    assert returns.dtype == dtype
    returns[:, 0].sum().backward()
    torch.testing.assert_close(
        rewards.grad, torch.tensor([[[1.0], [0.4]]], dtype=dtype)
    )
    torch.testing.assert_close(
        values.grad, torch.tensor([[[0.0], [0.0], [0.08]]], dtype=dtype)
    )
    torch.testing.assert_close(
        continuations.grad, torch.tensor([[[2.2], [2.4]]], dtype=dtype)
    )
