import pytest
import torch
from torch.distributions import AffineTransform, TanhTransform, TransformedDistribution

from dreamer.dreamer.networks import Actor, DreamerState


@pytest.mark.parametrize("shape", [(1,), (8,), (2, 3)])
def test_squashed_actions_respect_each_dimensions_bounds(shape):
    actor = Actor(16, 8, 32, 2, (-2.0, 0.0), (2.0, 1.0))
    state = DreamerState(torch.randn(*shape, 16), torch.randn(*shape, 8))
    actions, log_prob, _ = actor.get_action(state)
    assert actions.shape == (*shape, 2)
    assert log_prob.shape == (*shape, 1)
    assert (actions >= actor.action_low).all()
    assert (actions <= actor.action_high).all()


def test_log_probability_matches_torch_transformed_distribution():
    actor = Actor(16, 8, 32, 2, (-2.0, 0.0), (2.0, 1.0))
    state = DreamerState(torch.randn(8, 16), torch.randn(8, 8))
    actions, log_prob, _ = actor.get_action(state)
    scale = (actor.action_high - actor.action_low) / 2
    expected = TransformedDistribution(
        actor(state),
        [
            TanhTransform(),
            AffineTransform(actor.action_low + scale, scale),
        ],
    )
    torch.testing.assert_close(
        log_prob, expected.log_prob(actions).sum(-1, keepdim=True)
    )


def test_unsquashed_sampling_matches_base_normal():
    actor = Actor(16, 8, 32, 2, squash_actions=False)
    state = DreamerState(torch.randn(2, 16), torch.randn(2, 8))
    distribution = actor(state)
    torch.manual_seed(2)
    expected = distribution.rsample()
    torch.manual_seed(2)
    action, log_prob, _ = actor.get_action(state)
    torch.testing.assert_close(action, expected)
    torch.testing.assert_close(
        log_prob, distribution.log_prob(expected).sum(-1, keepdim=True)
    )


@pytest.mark.parametrize("squash", [False, True])
def test_action_sampling_preserves_gradients_and_dtype(squash):
    actor = Actor(16, 8, 32, 2, squash_actions=squash).double()
    h = torch.randn(2, 16, dtype=torch.float64, requires_grad=True)
    z = torch.randn(2, 8, dtype=torch.float64, requires_grad=True)
    action, log_prob, info = actor.get_action(DreamerState(h, z))
    assert action.dtype == log_prob.dtype == actor.action_low.dtype == torch.float64
    assert not info["mu"].requires_grad
    assert not info["std"].requires_grad
    (action.square().mean() + log_prob.mean()).backward()
    for parameter in list(actor.parameters()) + [h, z]:
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_deterministic_zero_mean_maps_to_action_midpoint():
    actor = Actor(16, 8, 32, 2, (-2.0, 0.0), (2.0, 1.0))
    with torch.no_grad():
        actor.mu_head.weight.zero_()
        actor.mu_head.bias.zero_()
    state = DreamerState(torch.zeros(1, 16), torch.zeros(1, 8))
    action, _, _ = actor.get_action(state, deterministic=True)
    torch.testing.assert_close(action, torch.tensor([[0.0, 0.5]]))


def test_saturated_actions_have_finite_log_probabilities_and_gradients():
    actor = Actor(16, 8, 32, 2)
    with torch.no_grad():
        actor.mu_head.weight.zero_()
        actor.mu_head.bias.copy_(torch.tensor([-100.0, 100.0]))
    state = DreamerState(torch.zeros(1, 16), torch.zeros(1, 8))
    action, log_prob, _ = actor.get_action(state, deterministic=True)
    torch.testing.assert_close(action, torch.tensor([[-1.0, 1.0]]))
    assert torch.isfinite(log_prob).all()
    log_prob.sum().backward()
    assert torch.isfinite(actor.mu_head.bias.grad).all()


@pytest.mark.parametrize("squash", [False, True])
def test_fixed_action_log_probability_matches_sampling(squash):
    actor = Actor(16, 8, 32, 2, (-2.0, 0.0), (2.0, 1.0), squash)
    states = DreamerState(torch.randn(2, 3, 16), torch.randn(2, 3, 8))
    with torch.no_grad():
        actions, expected, _ = actor.get_action(states)
    log_probs = actor.get_log_prob(states, actions)
    torch.testing.assert_close(log_probs, expected, atol=1e-4, rtol=1e-4)
    log_probs.mean().backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in actor.parameters()
    )


def test_reinforce_log_prob_treats_sampled_action_as_fixed():
    actor = Actor(16, 8, 32, 1, squash_actions=False)
    with torch.no_grad():
        for p in actor.parameters():
            p.zero_()
    states = DreamerState(torch.zeros(1, 16), torch.zeros(1, 8))
    action = torch.ones(1, 1, requires_grad=True)
    log_prob = actor.get_log_prob(states, action)
    (-log_prob.mean()).backward()
    assert actor.mu_head.bias.grad < 0
    assert action.grad is None


def test_saved_raw_actions_preserve_saturated_log_probs_and_score_gradients():
    actor = Actor(16, 8, 32, 2, (-2.0, 0.0), (2.0, 1.0))
    with torch.no_grad():
        actor.mu_head.weight.zero_()
        actor.mu_head.bias.copy_(torch.tensor([-100.0, 100.0]))
    states = DreamerState(torch.zeros(8, 16), torch.zeros(8, 8))
    with torch.no_grad():
        actions, expected, info = actor.get_action(states)
    torch.testing.assert_close(
        actions, actor.action_low.new_tensor([-2.0, 1.0]).expand_as(actions)
    )
    raw_actions = info["raw_action"].clone().requires_grad_()
    log_probs = actor.get_log_prob(states, actions, raw_actions=raw_actions)
    torch.testing.assert_close(log_probs, expected)
    distribution = actor(states)
    expected_gradient = (
        (raw_actions.detach() - distribution.loc.detach())
        / distribution.scale.detach().square()
    ).sum(0)
    log_probs.sum().backward()
    torch.testing.assert_close(actor.mu_head.bias.grad, expected_gradient)
    assert raw_actions.grad is None
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in actor.parameters()
    )


def test_actor_std_is_bounded_without_changing_mean():
    actor = Actor(16, 8, 32, 2)
    with torch.no_grad():
        actor.mu_head.weight.zero_()
        actor.mu_head.bias.copy_(torch.tensor([-4.0, 4.0]))
        actor.std_head.weight.zero_()
        actor.std_head.bias.copy_(torch.tensor([-100.0, 100.0]))
    states = DreamerState(torch.zeros(1, 16), torch.zeros(1, 8))
    distribution = actor(states)
    assert (distribution.scale >= 0.1).all()
    assert (distribution.scale <= 1.0).all()
    torch.testing.assert_close(distribution.loc, torch.tensor([[-4.0, 4.0]]))


def test_actor_std_keeps_gradient_outside_bounds_of_a_hard_clamp():
    actor = Actor(16, 8, 32, 1)
    with torch.no_grad():
        actor.std_head.weight.zero_()
        actor.std_head.bias.fill_(3.0)
    states = DreamerState(torch.zeros(1, 16), torch.zeros(1, 8))
    distribution = actor(states)
    distribution.scale.sum().backward()
    assert 0.1 < distribution.scale.item() < 1.0
    assert actor.std_head.bias.grad.item() > 0
