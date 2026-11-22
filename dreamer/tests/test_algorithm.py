import gymnasium
from dataclasses import replace
import numpy as np
import pytest
import torch

from dreamer.dreamer.algorithm import Dreamer
from dreamer.dreamer.config import DreamerConfig
from dreamer.dreamer.networks import Actor, Critic, DreamerState
from dreamer.dreamer.worldmodel import WorldModel


class ImageEnv(gymnasium.Env):
    observation_space = gymnasium.spaces.Box(0, 255, (64, 64, 3), np.uint8)
    action_space = gymnasium.spaces.Box(
        np.array([-2.0, 0.0], dtype=np.float32),
        np.array([2.0, 1.0], dtype=np.float32),
    )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(self.observation_space.shape, dtype=np.uint8), {}


@pytest.fixture
def config():
    return DreamerConfig(
        exp_name="test",
        obs_dim=(64, 64),
        obs_channels=3,
        action_dim=2,
        action_low=(-2.0, 0.0),
        action_high=(2.0, 1.0),
        deterministic_dim=16,
        stochastic_dim=8,
        hidden_dim=32,
        latent_obs_size=12,
        device="cpu",
        use_wandb=False,
    )


def test_initialization_builds_separate_optimizers_and_configured_actor(config):
    algorithm = Dreamer(config, ImageEnv())
    assert algorithm.env_steps == algorithm.gradient_steps == 0
    assert algorithm.replay_buffer.size == 0
    for module, optimizer, lr in (
        (algorithm.world_model, algorithm.model_optimizer, config.model_lr),
        (algorithm.actor, algorithm.actor_optimizer, config.actor_lr),
        (algorithm.critic, algorithm.critic_optimizer, config.critic_lr),
    ):
        assert {id(p) for p in module.parameters()} == {
            id(p) for group in optimizer.param_groups for p in group["params"]
        }
        assert optimizer.param_groups[0]["lr"] == lr
        assert all(p.device.type == "cpu" for p in module.parameters())
    state = algorithm.world_model.initial_state(1)
    action, _, _ = algorithm.actor.get_action(state)
    assert algorithm.env.action_space.contains(action.detach().numpy()[0])
    torch.testing.assert_close(
        algorithm.actor.action_low, torch.tensor(config.action_low)
    )
    torch.testing.assert_close(
        algorithm.actor.action_high, torch.tensor(config.action_high)
    )


def test_initialization_preserves_custom_networks(config):
    world_model = WorldModel(16, 8, 2, (64, 64), 3, 12, 32)
    actor = Actor(16, 8, 32, 2, squash_actions=False)
    critic = Critic(16, 8, 32)
    algorithm = Dreamer(config, ImageEnv(), world_model, actor, critic)
    assert algorithm.world_model is world_model
    assert algorithm.actor is actor
    assert algorithm.critic is critic


def test_initialization_rejects_discrete_environment(config):
    env = ImageEnv()
    env.action_space = gymnasium.spaces.Discrete(2)
    with pytest.raises(ValueError, match="continuous Box"):
        Dreamer(config, env)


class ShortImageEnv(ImageEnv):
    def __init__(self, truncated=False, channels_first=False, floating=False):
        self.truncate = truncated
        self.floating = floating
        shape = (3, 64, 64) if channels_first else (64, 64, 3)
        self.image = np.zeros(shape, dtype=np.float32 if floating else np.uint8)
        self.actions = []

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        self.actions = []
        self.image.fill(0)
        return self.image, {}

    def step(self, action):
        assert self.action_space.contains(action)
        self.actions.append(action.copy())
        self.t += 1
        self.image.fill(self.t / 10 if self.floating else self.t * 50)
        done = self.t == 3
        return (
            self.image,
            float(self.t),
            done and not self.truncate,
            done and self.truncate,
            {},
        )


@pytest.mark.parametrize("truncated", [False, True])
@pytest.mark.parametrize("channels_first", [False, True])
def test_collection_preserves_observations_actions_and_final_transition(
    config, truncated, channels_first
):
    config.before_training_steps = 0
    env = ShortImageEnv(truncated, channels_first)
    algorithm = Dreamer(config, env)
    stats = algorithm.collect_episode()
    stored = algorithm.replay_buffer.episodes[0]
    assert stats == {"episode_return": 6.0, "episode_length": 3}
    assert algorithm.env_steps == algorithm.replay_buffer.size == 3
    assert stored["observations"].shape == (4, 3, 64, 64)
    np.testing.assert_array_equal(stored["observations"][:, 0, 0, 0], [0, 50, 100, 150])
    np.testing.assert_array_equal(stored["actions"], env.actions)
    np.testing.assert_array_equal(stored["rewards"], [1, 2, 3])
    assert stored["terminated"][-1] == (not truncated)
    assert stored["truncated"][-1] == truncated
    assert all(p.grad is None for p in algorithm.world_model.parameters())
    assert all(p.grad is None for p in algorithm.actor.parameters())
    algorithm.collect_episode()
    assert algorithm.env_steps == 6
    assert algorithm.episode_length == 3


@pytest.mark.parametrize("floating", [False, True])
def test_collection_preprocesses_images_and_tracks_previous_actions(
    config, monkeypatch, floating
):
    # First action is random; subsequent actions use the actor.
    config.before_training_steps = 1
    env = ShortImageEnv(floating=floating)
    algorithm = Dreamer(config, env)
    inputs, previous_actions = [], []
    original_observe = algorithm.world_model.observe_step

    def observe(obs, state, action):
        assert not torch.is_grad_enabled()
        inputs.append(obs.clone())
        previous_actions.append(action.clone())
        return original_observe(obs, state, action)

    actor_calls = []
    original_get_action = algorithm.actor.get_action

    def get_action(state):
        actor_calls.append(state)
        return original_get_action(state)

    monkeypatch.setattr(algorithm.world_model, "observe_step", observe)
    monkeypatch.setattr(algorithm.actor, "get_action", get_action)
    algorithm.collect_episode()
    assert len(actor_calls) == 2
    assert previous_actions[0].count_nonzero() == 0
    for t in (1, 2):
        torch.testing.assert_close(
            previous_actions[t][0], torch.tensor(env.actions[t - 1])
        )
        expected = t / 10 if floating else t * 50 / 255
        torch.testing.assert_close(inputs[t], torch.full((1, 3, 64, 64), expected))


def test_warmup_collection_does_not_call_actor(config, monkeypatch):
    algorithm = Dreamer(config, ShortImageEnv())

    def unexpected_action(*args):
        pytest.fail("Actor should not be used during random warmup")

    monkeypatch.setattr(algorithm.actor, "get_action", unexpected_action)
    algorithm.collect_episode()
    assert algorithm.replay_buffer.size == 3


def test_collected_standard_transitions_feed_world_model_loss(config):
    algorithm = Dreamer(config, ShortImageEnv())
    algorithm.collect_episode()
    batch = algorithm.replay_buffer.sample(1, 3)
    obs = torch.tensor(batch["observations"], dtype=torch.float32) / 255
    next_obs = torch.tensor(batch["next_observations"], dtype=torch.float32) / 255
    torch.testing.assert_close(obs[:, 1:], next_obs[:, :-1])
    total, metrics, states = algorithm.world_model.loss(
        obs=obs,
        actions=torch.tensor(batch["actions"]),
        rewards=torch.tensor(batch["rewards"]),
        terminated=torch.tensor(batch["terminated"]),
        truncated=torch.tensor(batch["truncated"]),
        next_obs=next_obs,
    )
    assert states.deterministic.shape == (1, 3, 16)
    predicted = algorithm.world_model.predict(states)
    expected_reconstruction = (
        0.5
        * (predicted["reconstructions"] - next_obs).square().sum((-3, -2, -1)).mean()
    )
    torch.testing.assert_close(
        metrics["reconstruction_loss"], expected_reconstruction.detach()
    )
    total.backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in algorithm.world_model.parameters()
    )


@pytest.mark.parametrize("floating", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_world_model_update_preprocesses_replay_and_changes_only_model(
    config, monkeypatch, floating, dtype
):
    config.sequence_length = 3
    config.grad_clip = 0.1
    algorithm = Dreamer(config, ShortImageEnv(floating=floating))
    algorithm.world_model.to(dtype=dtype)
    algorithm.collect_episode()
    before = {
        name: p.detach().clone() for name, p in algorithm.world_model.named_parameters()
    }
    actor_before = {
        name: p.detach().clone() for name, p in algorithm.actor.named_parameters()
    }
    original_loss = algorithm.world_model.loss

    def loss(obs, actions, rewards, terminated, truncated, next_obs):
        for value in (obs, actions, rewards, next_obs):
            assert value.dtype == dtype
            assert value.device == next(algorithm.world_model.parameters()).device
        assert terminated.dtype == truncated.dtype == torch.bool
        expected = torch.tensor(
            [0, 0.1, 0.2] if floating else [0, 50 / 255, 100 / 255], dtype=dtype
        )
        torch.testing.assert_close(obs[0, :, 0, 0, 0], expected)
        expected_next = torch.tensor(
            [0.1, 0.2, 0.3] if floating else [50 / 255, 100 / 255, 150 / 255],
            dtype=dtype,
        )
        torch.testing.assert_close(next_obs[0, :, 0, 0, 0], expected_next)
        return original_loss(obs, actions, rewards, terminated, truncated, next_obs)

    monkeypatch.setattr(algorithm.world_model, "loss", loss)
    metrics, states = algorithm.update_world_model(1)
    assert algorithm.gradient_steps == 1
    assert not states.deterministic.requires_grad
    assert not states.stochastic.requires_grad
    assert all(
        not value.requires_grad and torch.isfinite(value) for value in metrics.values()
    )
    assert any(
        not torch.equal(before[name], p)
        for name, p in algorithm.world_model.named_parameters()
    )
    for name, parameter in algorithm.actor.named_parameters():
        torch.testing.assert_close(parameter, actor_before[name])
        assert parameter.grad is None
    assert all(p.grad is None for p in algorithm.critic.parameters())
    norm = torch.stack(
        [p.grad.norm() for p in algorithm.world_model.parameters()]
    ).norm()
    assert norm <= config.grad_clip + 1e-6
    # A second update must build a fresh graph and clear prior gradients.
    algorithm.update_world_model(1)
    assert algorithm.gradient_steps == 2


@pytest.mark.parametrize("horizon", [1, 4])
def test_imagination_preserves_start_states_and_advances_trajectory(
    config, monkeypatch, horizon
):
    algorithm = Dreamer(config, ShortImageEnv())
    states = DreamerState(torch.randn(2, 3, 16), torch.randn(2, 3, 8))
    seen_states, seen_actions = [], []
    original_step = algorithm.world_model.step_env

    def step(state, action):
        seen_states.append(state)
        seen_actions.append(action)
        return original_step(state, action)

    def unexpected_env_step(*args):
        pytest.fail("Imagination must not call Gymnasium")

    monkeypatch.setattr(algorithm.world_model, "step_env", step)
    monkeypatch.setattr(algorithm.env, "step", unexpected_env_step)
    trajectory, actions, rewards, continuations, values, raw_actions = (
        algorithm.imagine_rollout(states, horizon)
    )
    assert trajectory.deterministic.shape == (6, horizon + 1, 16)
    assert trajectory.stochastic.shape == (6, horizon + 1, 8)
    assert actions.shape == (6, horizon, 2)
    assert values.shape == (6, horizon + 1, 1)
    assert rewards.shape == continuations.shape == (6, horizon, 1)
    assert raw_actions.shape == actions.shape
    scale = (algorithm.actor.action_high - algorithm.actor.action_low) / 2
    torch.testing.assert_close(
        actions, algorithm.actor.action_low + (raw_actions.tanh() + 1) * scale
    )
    torch.testing.assert_close(
        trajectory.get_state()[:, 0], states.flatten().get_state()
    )
    for t in range(horizon):
        torch.testing.assert_close(
            seen_states[t].get_state(), trajectory.get_state()[:, t]
        )
        torch.testing.assert_close(seen_actions[t], actions[:, t])
    assert algorithm.env_steps == algorithm.gradient_steps == 0


def test_reinforce_imagination_has_no_gradients_or_parameter_flag_changes(config):
    algorithm = Dreamer(config, ShortImageEnv())
    states = DreamerState(
        torch.randn(2, 16, requires_grad=True), torch.randn(2, 8, requires_grad=True)
    )
    parameters = list(algorithm.world_model.parameters()) + list(
        algorithm.critic.parameters()
    )
    parameters[0].requires_grad_(False)
    original_flags = [p.requires_grad for p in parameters]
    trajectory, actions, rewards, continuations, values, raw_actions = (
        algorithm.imagine_rollout(states, 3)
    )
    assert [p.requires_grad for p in parameters] == original_flags
    assert not trajectory.deterministic.requires_grad
    assert not trajectory.stochastic.requires_grad
    assert all(
        not tensor.requires_grad
        for tensor in (actions, rewards, continuations, values, raw_actions)
    )


def test_reinforce_update_changes_actor_and_critic_without_model_gradients(config):
    algorithm = Dreamer(config, ShortImageEnv())
    states = DreamerState(torch.randn(2, 3, 16), torch.randn(2, 3, 8))
    rollout = algorithm.imagine_rollout(states, 3)
    model_before = {
        name: p.detach().clone() for name, p in algorithm.world_model.named_parameters()
    }
    actor_before = {
        name: p.detach().clone() for name, p in algorithm.actor.named_parameters()
    }
    critic_before = {
        name: p.detach().clone() for name, p in algorithm.critic.named_parameters()
    }
    metrics = algorithm.update_actor_critic(*rollout)
    assert all(
        not value.requires_grad and torch.isfinite(value) for value in metrics.values()
    )
    for name, parameter in algorithm.world_model.named_parameters():
        torch.testing.assert_close(parameter, model_before[name])
        assert parameter.grad is None
    for module, before in (
        (algorithm.actor, actor_before),
        (algorithm.critic, critic_before),
    ):
        assert any(
            not torch.equal(before[name], p) for name, p in module.named_parameters()
        )
        assert all(
            p.grad is not None and torch.isfinite(p.grad).all()
            for p in module.parameters()
        )
    algorithm.update_actor_critic(*algorithm.imagine_rollout(states, 3))
    assert torch.isfinite(algorithm.return_quantiles).all()


def test_positive_advantage_increases_selected_actions_likelihood(config):
    config.entropy_coefficient = 0
    config.squash_actions = False
    algorithm = Dreamer(config, ShortImageEnv())
    with torch.no_grad():
        for p in algorithm.actor.parameters():
            p.zero_()
    states = DreamerState(torch.zeros(1, 2, 16), torch.zeros(1, 2, 8))
    actions = torch.ones(1, 1, 2)
    rewards = torch.ones(1, 1, 1)
    continuations = torch.zeros(1, 1, 1)
    values = torch.zeros(1, 2, 1)
    actor_states = DreamerState(states.deterministic[:, :-1], states.stochastic[:, :-1])
    before = algorithm.actor.get_log_prob(actor_states, actions).detach()
    algorithm.update_actor_critic(
        states, actions, rewards, continuations, values, actions
    )
    after = algorithm.actor.get_log_prob(actor_states, actions).detach()
    assert (after > before).all()


def test_saturated_policy_update_uses_original_gaussian_samples(config, monkeypatch):
    config.entropy_coefficient = 0
    algorithm = Dreamer(config, ShortImageEnv())
    with torch.no_grad():
        algorithm.actor.mu_head.weight.zero_()
        algorithm.actor.mu_head.bias.copy_(torch.tensor([-100.0, 100.0]))
    states = DreamerState(torch.zeros(2, 16), torch.zeros(2, 8))
    rollout = algorithm.imagine_rollout(states, 3)
    original = algorithm.actor.get_log_prob
    seen = []

    def get_log_prob(state, actions, raw_actions=None):
        assert raw_actions is not None
        torch.testing.assert_close(raw_actions, rollout[-1])
        assert raw_actions.abs().min() > 90
        seen.append(True)
        return original(state, actions, raw_actions=raw_actions)

    monkeypatch.setattr(algorithm.actor, "get_log_prob", get_log_prob)
    metrics = algorithm.update_actor_critic(*rollout)
    assert seen
    assert all(torch.isfinite(value) for value in metrics.values())
    assert metrics["actor_grad_norm"] < 100


def test_return_normalization_tracks_smoothed_percentiles(config):
    config.return_norm_rate = 0.25
    algorithm = Dreamer(config, ShortImageEnv())
    states = DreamerState(torch.zeros(2, 2, 16), torch.zeros(2, 2, 8))
    actions = torch.zeros(2, 1, 2)
    rewards = torch.tensor([[[0.0]], [[100.0]]])
    continuation = torch.zeros(2, 1, 1)
    values = torch.zeros(2, 2, 1)
    metrics = algorithm.update_actor_critic(
        states, actions, rewards, continuation, values, torch.zeros_like(actions)
    )
    torch.testing.assert_close(algorithm.return_quantiles, torch.tensor([5.0, 95.0]))
    torch.testing.assert_close(metrics["return_scale"], torch.tensor(90.0))
    algorithm.update_actor_critic(
        states, actions, rewards * 2, continuation, values, torch.zeros_like(actions)
    )
    torch.testing.assert_close(algorithm.return_quantiles, torch.tensor([6.25, 118.75]))


@pytest.mark.parametrize("truncated, expected_count", [(False, 2), (True, 3)])
def test_imagination_starts_exclude_termination_but_keep_truncation(
    config, monkeypatch, truncated, expected_count
):
    config.sequence_length = 3
    algorithm = Dreamer(config, ShortImageEnv(truncated=truncated))
    algorithm.collect_episode()
    original_loss = algorithm.world_model.loss
    inferred = []

    def loss(*args):
        result = original_loss(*args)
        inferred.append(result[2])
        return result

    monkeypatch.setattr(algorithm.world_model, "loss", loss)
    _, starts = algorithm.update_world_model(1)
    assert starts.deterministic.shape == (expected_count, 16)
    assert starts.stochastic.shape == (expected_count, 8)
    torch.testing.assert_close(
        starts.deterministic, inferred[0].deterministic[0, :expected_count]
    )
    assert not starts.deterministic.requires_grad


def test_training_loop_warmup_updates_and_checkpoint_resume(
    config, tmp_path, monkeypatch
):
    config.sequence_length = 3
    config.batch_size = 1
    config.before_training_steps = 4
    config.train_steps_per_episode = 2
    config.imagination_horizon = 2
    config.log_freq = 1
    config.save_freq = 3
    config.save_replay_buffer = True
    config.save_dir = str(tmp_path)
    algorithm = Dreamer(config, ShortImageEnv())
    logged, finished, saved_steps = [], [], []
    monkeypatch.setattr(
        algorithm.logger, "log", lambda metrics, step: logged.append((step, metrics))
    )
    monkeypatch.setattr(algorithm.logger, "finish", lambda: finished.append(True))
    original_save = algorithm.logger.save_checkpoint

    def save_checkpoint(networks, step):
        saved_steps.append(step)
        return original_save(networks, step)

    monkeypatch.setattr(algorithm.logger, "save_checkpoint", save_checkpoint)
    before = {
        name: p.detach().clone() for name, p in algorithm.actor.named_parameters()
    }
    algorithm.train(6)
    assert algorithm.env_steps == 6
    assert algorithm.gradient_steps == 2
    assert algorithm.replay_buffer.size == 6
    assert [step for step, _ in logged] == [3, 6]
    assert "train/actor_loss" not in logged[0][1]
    assert "train/actor_loss" in logged[1][1]
    assert finished == [True]
    assert saved_steps == [3, 6]
    assert any(
        not torch.equal(before[name], p)
        for name, p in algorithm.actor.named_parameters()
    )

    checkpoint = next(tmp_path.rglob("checkpoint_6.pt"))
    restored = Dreamer(
        replace(config, path_to_checkpoint=str(checkpoint)), ShortImageEnv()
    )
    assert restored.env_steps == 6
    assert restored.gradient_steps == 2
    assert restored.replay_buffer.size == 6
    torch.testing.assert_close(restored.return_quantiles, algorithm.return_quantiles)
    for name in ("world_model", "actor", "critic"):
        for actual, expected in zip(
            getattr(restored, name).parameters(), getattr(algorithm, name).parameters()
        ):
            torch.testing.assert_close(actual, expected)
    for name in ("model_optimizer", "actor_optimizer", "critic_optimizer"):
        assert getattr(restored, name).state_dict()["state"]
    restored.train(9)
    assert restored.env_steps == 9
    assert restored.gradient_steps == 4


def test_training_waits_for_long_enough_episode(config):
    config.sequence_length = 4
    config.before_training_steps = 0
    config.train_steps_per_episode = 1
    algorithm = Dreamer(config, ShortImageEnv())
    algorithm.train(4)
    assert algorithm.env_steps == 6
    assert algorithm.gradient_steps == 0


def test_training_skips_behavior_when_all_starts_are_terminal(config, monkeypatch):
    class OneStepEnv(ShortImageEnv):
        def step(self, action):
            obs, reward, _, _, info = super().step(action)
            return obs, reward, True, False, info

    config.sequence_length = 1
    config.batch_size = 1
    config.before_training_steps = 0
    config.train_steps_per_episode = 1
    algorithm = Dreamer(config, OneStepEnv())

    def unexpected_rollout(*args):
        pytest.fail("Terminal states must not start imagined rollouts")

    monkeypatch.setattr(algorithm, "imagine_rollout", unexpected_rollout)
    algorithm.train(1)
    assert algorithm.env_steps == algorithm.gradient_steps == 1
    assert algorithm.return_quantiles is None
    assert all(p.grad is None for p in algorithm.actor.parameters())


def test_evaluation_uses_separate_env_and_preserves_training_state(config):
    created, closed = [], []

    def make_env():
        env = ShortImageEnv()
        env.close = lambda: closed.append(True)
        created.append(env)
        return env

    algorithm = Dreamer(config, ShortImageEnv(), make_env=make_env)
    algorithm.collect_episode()
    algorithm.actor.eval()
    rng = torch.get_rng_state().clone()
    returns = algorithm.evaluate(2, seed=4)
    torch.testing.assert_close(torch.get_rng_state(), rng)
    assert returns == [6.0, 6.0]
    assert algorithm.env_steps == 3
    assert algorithm.gradient_steps == 0
    assert algorithm.replay_buffer.size == 3
    assert algorithm.world_model.training
    assert not algorithm.actor.training
    assert closed == [True]
    assert created[0] is not algorithm.env
    assert all(p.grad is None for p in algorithm.world_model.parameters())
    first_actions = np.stack(created[0].actions)
    algorithm.evaluate(2, seed=4)
    np.testing.assert_array_equal(first_actions, np.stack(created[1].actions))


def test_evaluation_can_report_unscaled_reward_from_info(config):
    class ScaledEnv(ShortImageEnv):
        def step(self, action):
            obs, reward, terminated, truncated, info = super().step(action)
            return (
                obs,
                reward * 0.1,
                terminated,
                truncated,
                dict(info, raw_reward=reward),
            )

    algorithm = Dreamer(config, ShortImageEnv(), make_env=ScaledEnv)
    assert algorithm.evaluate(1, reward_key="raw_reward") == [6.0]
    np.testing.assert_allclose(algorithm.evaluate(1), [0.6])


def test_resume_without_saved_replay_refills_before_updating(
    config, tmp_path, monkeypatch
):
    config.sequence_length = 3
    config.batch_size = 1
    config.before_training_steps = 6
    config.train_steps_per_episode = 1
    config.imagination_horizon = 2
    config.save_freq = 3
    config.save_dir = str(tmp_path)
    algorithm = Dreamer(config, ShortImageEnv())
    for _ in range(3):
        algorithm.collect_episode()
    algorithm.save_ckpt(9)
    checkpoint = next(tmp_path.rglob("checkpoint_9.pt"))
    restored = Dreamer(
        replace(config, path_to_checkpoint=str(checkpoint), save_replay_buffer=True),
        ShortImageEnv(),
    )
    assert restored.replay_buffer.size == 0
    original_get_action = restored.actor.get_action

    def get_action(*args, **kwargs):
        assert restored.replay_buffer.size >= 6
        return original_get_action(*args, **kwargs)

    monkeypatch.setattr(restored.actor, "get_action", get_action)
    restored.train(12)
    assert restored.replay_buffer.size == 3
    assert restored.gradient_steps == 0
    restored.train(15)
    assert restored.replay_buffer.size == 6
    assert restored.gradient_steps == 1


def test_video_evaluation_records_first_episode_and_final_frame(config, tmp_path):
    import imageio.v2 as imageio

    class RenderEnv(ShortImageEnv):
        def render(self):
            return np.full((32, 32, 3), self.t * 30, dtype=np.uint8)

    algorithm = Dreamer(config, ShortImageEnv(), make_env=RenderEnv)
    path = tmp_path / "videos" / "eval.mp4"
    returns = algorithm.evaluate(2, video_path=path, video_fps=7)
    assert returns == [6.0, 6.0]
    assert path.is_file()
    reader = imageio.get_reader(path)
    try:
        assert reader.count_frames() == 4
        assert reader.get_meta_data()["fps"] == 7
        assert reader.get_data(0).mean() < 2
        assert reader.get_data(3).mean() > 85
    finally:
        reader.close()
