import pytest
import torch
from torch.distributions import Bernoulli, Normal, kl_divergence

from dreamer.dreamer.networks import Actor, Critic, DreamerState
from dreamer.dreamer.worldmodel import WorldModel, sobel_edges


@pytest.fixture
def model():
    torch.manual_seed(7)
    return WorldModel(16, 8, 2, (64, 64), 3, 12, 32)


@pytest.mark.parametrize("channels", [1, 3])
def test_sobel_detects_direction_and_preserves_constant_images(channels):
    constant = torch.ones(2, 3, channels, 8, 8)
    torch.testing.assert_close(
        sobel_edges(constant), torch.zeros(2, 3, 2 * channels, 8, 8)
    )
    ramp = torch.arange(8, dtype=torch.float32).expand_as(constant)
    edges = sobel_edges(ramp).reshape(2, 3, channels, 2, 8, 8)
    torch.testing.assert_close(
        edges[..., 0, :, 1:-1], torch.ones_like(edges[..., 0, :, 1:-1])
    )
    torch.testing.assert_close(
        edges[..., 1, :, :], torch.zeros_like(edges[..., 1, :, :])
    )


def test_edge_loss_is_weighted_and_trains_encoder_and_decoder(model):
    model.loss_weights = (0, 0, 0, 0)
    model.edge_weight = 2.0
    obs = torch.rand(1, 2, 3, 64, 64)
    flags = torch.zeros(1, 2, 1, dtype=torch.bool)
    loss, metrics, _ = model.loss(
        obs,
        torch.zeros(1, 2, 2),
        torch.zeros(1, 2, 1),
        flags,
        flags,
        obs,
    )
    assert metrics["edge_loss"] > 0
    torch.testing.assert_close(loss.detach(), 2 * metrics["edge_loss"])
    loss.backward()
    for module in (model.obs_handler.encoder, model.obs_handler.decoder, model.rssm):
        assert any(
            p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters()
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_initial_state_matches_model_dtype_and_device(model, dtype):
    model.to(dtype=dtype)
    first = model.initial_state(2)
    second = model.initial_state(2)
    assert first.deterministic.shape == (2, 16)
    assert first.stochastic.shape == (2, 8)
    for tensor in (first.deterministic, first.stochastic):
        assert tensor.dtype == dtype
        assert tensor.device == next(model.parameters()).device
        assert not tensor.requires_grad
        assert torch.count_nonzero(tensor) == 0
    first.deterministic.fill_(1)
    first.stochastic.fill_(1)
    assert torch.count_nonzero(second.get_state()) == 0


def test_observation_changes_posterior_but_not_prior(model):
    state = model.initial_state(2)
    actions = torch.randn(2, 2)
    obs = torch.randn(2, 3, 64, 64)
    first, prior, posterior = model.observe_step(obs, state, actions)
    second, other_prior, other_posterior = model.observe_step(obs + 5, state, actions)
    assert isinstance(prior, Normal)
    assert isinstance(posterior, Normal)
    assert posterior.loc.shape == (2, 8)
    assert (posterior.scale > 0).all()
    torch.testing.assert_close(first.deterministic, second.deterministic)
    torch.testing.assert_close(prior.loc, other_prior.loc)
    torch.testing.assert_close(prior.scale, other_prior.scale)
    assert not torch.allclose(posterior.loc, other_posterior.loc)


def test_imagination_uses_prior_and_is_differentiable(model):
    previous = model.initial_state(2)
    actions = torch.randn(2, 2, requires_grad=True)
    torch.manual_seed(11)
    state, prior = model.rssm.imagine(previous, actions)
    torch.manual_seed(11)
    torch.testing.assert_close(state.stochastic, prior.rsample())
    assert state.deterministic.shape == (2, 16)
    assert state.stochastic.shape == (2, 8)
    state.get_state().square().mean().backward()
    assert torch.isfinite(actions.grad).all()
    assert actions.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.rssm.posterior.parameters())


@pytest.mark.parametrize("batch_size, length", [(1, 1), (2, 4)])
def test_sequence_matches_repeated_observe_step(model, batch_size, length):
    obs = torch.randn(batch_size, length, 3, 64, 64)
    next_obs = torch.randn_like(obs)
    actions = torch.randn(batch_size, length, 2)
    terminated = torch.zeros(batch_size, length, dtype=torch.bool)
    truncated = torch.zeros_like(terminated)
    initial = DreamerState(torch.randn(batch_size, 16), torch.randn(batch_size, 8))
    torch.manual_seed(19)
    states, priors, posteriors = model.observe_sequence(
        obs, actions, terminated, truncated, next_obs, initial
    )
    assert states.deterministic.shape == (batch_size, length, 16)
    assert states.stochastic.shape == (batch_size, length, 8)
    assert priors.loc.shape == posteriors.loc.shape == (batch_size, length, 8)

    torch.manual_seed(19)
    state, _, _ = model.observe_step(obs[:, 0], initial, torch.zeros(batch_size, 2))
    for t in range(length):
        state, prior, posterior = model.observe_step(
            next_obs[:, t], state, actions[:, t]
        )
        torch.testing.assert_close(states.deterministic[:, t], state.deterministic)
        torch.testing.assert_close(states.stochastic[:, t], state.stochastic)
        torch.testing.assert_close(priors.loc[:, t], prior.loc)
        torch.testing.assert_close(priors.scale[:, t], prior.scale)
        torch.testing.assert_close(posteriors.loc[:, t], posterior.loc)
        torch.testing.assert_close(posteriors.scale[:, t], posterior.scale)


def test_done_resets_following_observation_only_for_finished_rows(model):
    obs = torch.randn(2, 3, 3, 64, 64)
    actions = torch.randn(2, 3, 2)
    terminated = torch.tensor([[True, False, False], [False, False, False]])
    truncated = torch.tensor([[False, False, False], [False, True, False]])
    next_obs = torch.randn_like(obs)
    changed_next_obs = next_obs.clone()
    changed_next_obs[:, 0] += 10
    changed_obs = obs.clone()
    changed_actions = actions.clone()
    changed_obs[:, 0] += 10
    changed_actions[:, 0] += 10

    torch.manual_seed(23)
    original, _, _ = model.observe_sequence(
        obs, actions, terminated, truncated, next_obs
    )
    torch.manual_seed(23)
    changed, _, _ = model.observe_sequence(
        changed_obs, changed_actions, terminated, truncated, changed_next_obs
    )

    # Termination at t=0 resets row 0 at t=1, not at t=0.
    assert not torch.allclose(original.get_state()[0, 0], changed.get_state()[0, 0])
    torch.testing.assert_close(original.get_state()[0, 1:], changed.get_state()[0, 1:])
    # Row 1 retains history at t=1 and forgets it after truncation at t=1.
    assert not torch.allclose(original.get_state()[1, 1], changed.get_state()[1, 1])
    torch.testing.assert_close(original.get_state()[1, 2], changed.get_state()[1, 2])


def test_final_done_preserves_terminal_observation_history(model):
    observations = torch.randn(1, 4, 3, 64, 64)
    obs, next_obs = observations[:, :-1], observations[:, 1:]
    actions = torch.randn(1, 3, 2)
    terminated = torch.tensor([[False, False, True]])
    truncated = torch.zeros_like(terminated)
    torch.manual_seed(5)
    terminal, _, _ = model.observe_sequence(
        obs, actions, terminated, truncated, next_obs
    )
    torch.manual_seed(5)
    continuing, _, _ = model.observe_sequence(
        obs, actions, torch.zeros_like(terminated), truncated, next_obs
    )
    torch.testing.assert_close(terminal.get_state(), continuing.get_state())


@pytest.mark.parametrize("ending", ["neither", "terminated", "truncated"])
def test_recurrent_gradients_stop_at_episode_boundary(model, ending):
    obs = torch.randn(1, 3, 3, 64, 64, requires_grad=True)
    actions = torch.randn(1, 3, 2, requires_grad=True)
    terminated = torch.tensor([[ending == "terminated", False, False]])
    truncated = torch.tensor([[ending == "truncated", False, False]])
    next_obs = torch.randn_like(obs)
    obs_before, actions_before = obs.detach().clone(), actions.detach().clone()
    states, _, _ = model.observe_sequence(obs, actions, terminated, truncated, next_obs)
    states.get_state()[:, -1].square().mean().backward()
    assert torch.isfinite(obs.grad).all()
    assert torch.isfinite(actions.grad).all()
    if ending != "neither":
        assert obs.grad[:, 0].count_nonzero() == 0
        assert actions.grad[:, 0].count_nonzero() == 0
        assert actions.grad[:, 1].abs().sum() > 0
    else:
        assert obs.grad[:, 0].abs().sum() > 0
        assert actions.grad[:, 0].abs().sum() > 0
    assert actions.grad[:, -1].abs().sum() > 0
    torch.testing.assert_close(obs, obs_before)
    torch.testing.assert_close(actions, actions_before)


@pytest.mark.parametrize("image_size, channels", [((64, 64), 3), ((96, 96), 12)])
@pytest.mark.parametrize("leading_shape", [(1,), (2, 3)])
def test_predictions_preserve_shapes_and_frame_order(
    image_size, channels, leading_shape
):
    model = WorldModel(16, 8, 2, image_size, channels, 12, 32)
    states = DreamerState(
        torch.randn(*leading_shape, 16), torch.randn(*leading_shape, 8)
    )
    result = model.predict(states)
    assert result["reconstructions"].shape == (*leading_shape, channels, *image_size)
    assert result["rewards"].shape == (*leading_shape, 1)
    assert isinstance(result["continuations"], Bernoulli)
    assert result["continuations"].batch_shape == (*leading_shape, 1)

    index = (0,) if len(leading_shape) == 1 else (1, 2)
    single = DreamerState(
        states.deterministic[index].unsqueeze(0),
        states.stochastic[index].unsqueeze(0),
    )
    torch.testing.assert_close(
        result["reconstructions"][index], model.obs_handler.decode(single)[0]
    )


def test_prediction_can_skip_image_decoder(model, monkeypatch):
    states = model.initial_state(2)
    expected = model.predict(states)

    def unexpected_decode(*args):
        pytest.fail("Image decoder ran despite decode_obs=False")

    monkeypatch.setattr(model.obs_handler, "decode", unexpected_decode)
    result = model.predict(states, False)
    assert result["reconstructions"] is None
    torch.testing.assert_close(result["rewards"], expected["rewards"])
    torch.testing.assert_close(
        result["continuations"].probs, expected["continuations"].probs
    )


def test_world_model_predictions_and_kl_backpropagate_to_all_components(model):
    obs = torch.randn(2, 3, 3, 64, 64)
    actions = torch.randn(2, 3, 2)
    next_obs = torch.randn_like(obs)
    states, priors, posteriors = model.observe_sequence(
        obs,
        actions,
        torch.zeros(2, 3, dtype=torch.bool),
        torch.zeros(2, 3, dtype=torch.bool),
        next_obs,
    )
    predictions = model.predict(states)
    # Exercise existing components together, without defining the future training loss.
    loss = (predictions["reconstructions"] - next_obs).square().mean()
    loss = loss + predictions["rewards"].square().mean()
    loss = loss - predictions["continuations"].log_prob(torch.ones(2, 3, 1)).mean()
    loss = loss + kl_divergence(posteriors, priors).sum(-1).mean()
    loss.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name


@pytest.mark.parametrize("leading_shape", [(1,), (2, 3)])
def test_actor_and_critic_accept_latent_states(leading_shape):
    states = DreamerState(
        torch.randn(*leading_shape, 16), torch.randn(*leading_shape, 8)
    )
    actor = Actor(16, 8, 32, 2)
    critic = Critic(16, 8, 32)
    distribution = actor(states)
    assert isinstance(distribution, Normal)
    assert distribution.loc.shape == (*leading_shape, 2)
    assert (distribution.scale > 0).all()
    assert critic(states).shape == (*leading_shape, 1)
    (distribution.rsample().square().mean() + critic(states).square().mean()).backward()
    for parameter in list(actor.parameters()) + list(critic.parameters()):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


@pytest.mark.parametrize("trailing_dimension", [True])
def test_loss_accepts_scalar_targets_and_backpropagates(model, trailing_dimension):
    obs = torch.randn(2, 3, 3, 64, 64)
    actions = torch.randn(2, 3, 2)
    truncated = torch.zeros(2, 3, 1, dtype=torch.bool)
    rewards = torch.randn(2, 3)
    terminated = torch.zeros(2, 3, dtype=torch.bool)
    if trailing_dimension:
        rewards = rewards.unsqueeze(-1)
        terminated = terminated.unsqueeze(-1)
    initial = model.initial_state(2)
    total, metrics, states = model.loss(
        obs,
        actions,
        rewards,
        terminated,
        truncated,
        torch.randn_like(obs),
        initial_state=initial,
    )
    assert total.shape == ()
    assert torch.isfinite(total)
    expected_total = (
        metrics["reconstruction_loss"]
        + metrics["reward_loss"]
        + 10 * metrics["continuation_loss"]
        + metrics["kl_loss"]
    )
    torch.testing.assert_close(total.detach(), expected_total)
    assert all(not metric.requires_grad for metric in metrics.values())
    assert states.deterministic.shape == (2, 3, 16)
    total.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name


@pytest.mark.parametrize("sequence_length", [1, 3])
def test_representation_metrics_measure_time_variation(model, sequence_length):
    obs = torch.randn(2, sequence_length, 3, 64, 64)
    flags = torch.zeros(2, sequence_length, 1, dtype=torch.bool)
    _, metrics, states = model.loss(
        obs,
        torch.zeros(2, sequence_length, 2),
        torch.zeros(2, sequence_length, 1),
        flags,
        flags,
        obs,
    )
    torch.testing.assert_close(
        metrics["stochastic_state_variation"],
        states.stochastic.detach().std(dim=1, correction=0).mean(),
    )
    torch.testing.assert_close(
        metrics["observation_variation"],
        obs.std(dim=1, correction=0).mean(),
    )
    assert all(
        torch.isfinite(value) and not value.requires_grad for value in metrics.values()
    )
    assert metrics["prior_std"] > 0
    assert metrics["posterior_std"] > 0
    if sequence_length == 1:
        assert all(
            value == 0 for key, value in metrics.items() if key.endswith("variation")
        )


def test_prediction_mask_must_match_batch_and_time(model):
    with pytest.raises(ValueError, match="prediction_mask"):
        model.loss(
            torch.randn(1, 3, 3, 64, 64),
            torch.zeros(1, 3, 2),
            torch.zeros(1, 3),
            torch.zeros(1, 3, 1, dtype=torch.bool),
            torch.zeros(1, 3, 1, dtype=torch.bool),
            next_obs=torch.randn(1, 3, 3, 64, 64),
            prediction_mask=torch.ones(1, 1),
        )


def test_step_env_predicts_from_next_state_without_decoding_images(model, monkeypatch):
    def unexpected_decode(*args):
        pytest.fail("Latent stepping must not decode images")

    monkeypatch.setattr(model.obs_handler, "decode", unexpected_decode)
    state = model.initial_state(2)
    actions = torch.randn(2, 2, requires_grad=True)
    next_state, rewards, continuations = model.step_env(state, actions)
    assert next_state.deterministic.shape == (2, 16)
    assert next_state.stochastic.shape == (2, 8)
    assert rewards.shape == continuations.shape == (2, 1)
    assert ((continuations >= 0) & (continuations <= 1)).all()
    torch.testing.assert_close(rewards, model.reward_decoder(next_state))
    torch.testing.assert_close(continuations, model.continue_decoder(next_state).probs)
    (rewards.sum() + continuations.sum()).backward()
    assert torch.isfinite(actions.grad).all()
    assert actions.grad.abs().sum() > 0
