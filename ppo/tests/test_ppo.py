import torch
import pytest
import numpy as np
import gymnasium
from functools import partial
from dataclasses import dataclass

from ppo import PPO, PPOConfig, Actor, Critic, gae

# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------


@dataclass
class SingleDiscreteConfig(PPOConfig):
    exp_name: str = "test_single"
    obs_dim: int = 4
    act_dim: int = 2
    act_shape: int = 1
    actor_hidden_size: int = 16
    critic_hidden_size: int = 16
    T: int = 64
    minibatch_size: int = 32
    epochs_per_batch: int = 2
    frame_stack: int = 1
    device: str = "cpu"
    use_wandb: bool = False
    save_freq: None = None
    save_dir: None = None


@dataclass
class MultiDiscreteConfig(PPOConfig):
    exp_name: str = "test_multi"
    obs_dim: int = 8
    act_dim: int = 2
    act_shape: int = 2
    actor_hidden_size: int = 16
    critic_hidden_size: int = 16
    T: int = 64
    minibatch_size: int = 32
    epochs_per_batch: int = 2
    frame_stack: int = 1
    device: str = "cpu"
    use_wandb: bool = False
    save_freq: None = None
    save_dir: None = None


class MultiDiscreteCartpole(gymnasium.Env):
    """Two CartPoles whose actions are independent (act_shape=2)."""

    def __init__(self, render_mode="rgb_array"):
        self.envs = [
            gymnasium.make("CartPole-v1", render_mode=render_mode) for _ in range(2)
        ]
        self.render_mode = render_mode

    def step(self, action):
        obs, reward, terminated, truncated = [], 0, False, False
        for c, env in enumerate(self.envs):
            o, r, te, tr, _ = env.step(int(action[c]))
            obs.append(o)
            reward += r
            terminated = te or terminated
            truncated = tr or truncated
        return np.concatenate(obs), reward / len(self.envs), terminated, truncated, {}

    def render(self):
        images = [env.render() for env in self.envs]
        return np.hstack(images)

    def reset(self, **kwargs):
        obs = []
        for env in self.envs:
            o, _ = env.reset(**kwargs)
            obs.append(o)
        return np.concatenate(obs), {}

    def close(self):
        for env in self.envs:
            env.close()


def make_cartpole(render_mode=None):
    return gymnasium.make("CartPole-v1", render_mode=render_mode)


def make_multi_cartpole(render_mode=None):
    return MultiDiscreteCartpole(render_mode=render_mode or "rgb_array")


# ---------------------------------------------------------------------------
# 1. Network shape tests
# ---------------------------------------------------------------------------


class TestActorShapes:
    def test_single_discrete_unbatched(self):
        actor = Actor(obs_dim=4, act_dim=2, hidden_size=16, act_shape=1)
        obs = torch.randn(1, 4)
        dist = actor(obs)
        action = dist.sample()
        assert action.shape == (1,), f"Expected (1,), got {action.shape}"
        lp = dist.log_prob(action)
        assert lp.shape == (1,), f"Expected (1,), got {lp.shape}"

    def test_single_discrete_batched(self):
        actor = Actor(obs_dim=4, act_dim=2, hidden_size=16, act_shape=1)
        obs = torch.randn(32, 4)
        dist = actor(obs)
        action = dist.sample()
        assert action.shape == (32,)
        lp = dist.log_prob(action)
        assert lp.shape == (32,)

    def test_multi_discrete_unbatched(self):
        actor = Actor(obs_dim=8, act_dim=2, hidden_size=16, act_shape=2)
        obs = torch.randn(1, 8)
        dist = actor(obs)
        action = dist.sample()
        assert action.shape == (1, 2), f"Expected (1, 2), got {action.shape}"
        lp = dist.log_prob(action)
        assert lp.shape == (1, 2)

    def test_multi_discrete_batched(self):
        actor = Actor(obs_dim=8, act_dim=2, hidden_size=16, act_shape=2)
        obs = torch.randn(32, 8)
        dist = actor(obs)
        action = dist.sample()
        assert action.shape == (32, 2)
        lp = dist.log_prob(action)
        assert lp.shape == (32, 2)
        # joint log-prob after summing over act_shape dim
        joint_lp = lp.sum(-1)
        assert joint_lp.shape == (32,)

    def test_multi_discrete_actions_in_range(self):
        actor = Actor(obs_dim=8, act_dim=5, hidden_size=16, act_shape=3)
        obs = torch.randn(64, 8)
        dist = actor(obs)
        action = dist.sample()
        assert action.shape == (64, 3)
        assert (action >= 0).all() and (action < 5).all()


class TestCriticShapes:
    def test_unbatched(self):
        critic = Critic(obs_dim=4, hidden_size=16)
        obs = torch.randn(1, 4)
        val = critic(obs)
        assert val.shape == (1, 1)

    def test_batched(self):
        critic = Critic(obs_dim=4, hidden_size=16)
        obs = torch.randn(32, 4)
        val = critic(obs)
        assert val.shape == (32, 1)


# ---------------------------------------------------------------------------
# 2. GAE tests
# ---------------------------------------------------------------------------


class TestGAE:
    def test_single_step(self):
        """With T=1, GAE = delta = r + gamma*0 - V(s) (terminal)."""
        rewards = torch.tensor([1.0])
        dones = torch.tensor([1.0])
        values = torch.tensor([0.5])
        adv = gae(rewards, dones, values, gamma=0.99, gae_lambda=0.95)
        expected = 1.0 + 0.99 * 0 - 0.5  # delta for terminal state
        assert torch.allclose(adv, torch.tensor([expected]))

    def test_two_steps_no_done(self):
        rewards = torch.tensor([1.0, 2.0])
        dones = torch.tensor([0.0, 0.0])
        values = torch.tensor([0.5, 0.8])
        gamma, lam = 0.99, 0.95

        # t=1 (last step): delta_1 = r1 + gamma*0 - V1  (last timestep)
        delta_1 = 2.0 + gamma * 0 - 0.8
        a_1 = delta_1

        # t=0: delta_0 = r0 + gamma*V1 - V0
        delta_0 = 1.0 + gamma * 0.8 - 0.5
        a_0 = delta_0 + gamma * lam * a_1

        adv = gae(rewards, dones, values, gamma, lam)
        expected = torch.tensor([a_0, a_1])
        assert torch.allclose(adv, expected, atol=1e-6)

    def test_done_resets_advantage(self):
        """A done at t=0 should prevent bootstrapping from t=1."""
        rewards = torch.tensor([1.0, 2.0])
        dones = torch.tensor([1.0, 0.0])
        values = torch.tensor([0.5, 0.8])
        gamma, lam = 0.99, 0.95

        delta_1 = 2.0 + gamma * 0 - 0.8
        a_1 = delta_1
        # t=0 is terminal: delta_0 = r0 + gamma*0 - V0, no future A
        delta_0 = 1.0 + gamma * 0 - 0.5
        a_0 = delta_0

        adv = gae(rewards, dones, values, gamma, lam)
        expected = torch.tensor([a_0, a_1])
        assert torch.allclose(adv, expected, atol=1e-6)

    def test_all_zeros(self):
        rewards = torch.zeros(5)
        dones = torch.zeros(5)
        values = torch.zeros(5)
        adv = gae(rewards, dones, values, gamma=0.99, gae_lambda=0.95)
        assert torch.allclose(adv, torch.zeros(5))


# ---------------------------------------------------------------------------
# 3. PPO loss function tests
# ---------------------------------------------------------------------------


class TestPPOLosses:
    """Test actor and critic losses in isolation (no env needed)."""

    def _make_ppo(self):
        config = SingleDiscreteConfig()
        env = make_cartpole()
        ppo = PPO(config, env, actor=None, critic=None, make_env=make_cartpole)
        return ppo

    def test_actor_loss_no_clip(self):
        ppo = self._make_ppo()
        advantages = torch.tensor([1.0, -1.0, 0.5])
        old_lp = torch.tensor([-0.5, -0.8, -0.3])
        new_lp = old_lp.clone()  # identical policy -> ratio=1, no clipping
        loss = ppo._actor_loss(advantages, old_lp, new_lp, epsilon=0.2)
        # ratio=1 everywhere, loss = -mean(adv * 1) = -mean([1, -1, 0.5]) = -0.1667
        expected = -advantages.mean()
        assert torch.allclose(loss, expected, atol=1e-5)

    def test_actor_loss_clips_large_ratio(self):
        ppo = self._make_ppo()
        advantages = torch.ones(4)
        old_lp = torch.zeros(4)
        new_lp = torch.full((4,), 1.0)  # ratio = e^1 ≈ 2.718
        loss = ppo._actor_loss(advantages, old_lp, new_lp, epsilon=0.2)
        # ratio ≈ 2.718 > 1.2, clipped_ratio = 1.2
        # min(adv * ratio, adv * clipped_ratio) = min(2.718, 1.2) = 1.2
        # loss = -1.2
        assert torch.allclose(loss, torch.tensor(-1.2), atol=1e-3)

    def test_critic_loss_zero_when_perfect(self):
        ppo = self._make_ppo()
        targets = torch.tensor([1.0, 2.0, 3.0])
        loss = ppo._critic_loss(targets, targets.clone())
        assert loss.item() == 0.0

    def test_critic_loss_positive(self):
        ppo = self._make_ppo()
        targets = torch.tensor([1.0, 2.0, 3.0])
        preds = torch.zeros(3)
        loss = ppo._critic_loss(targets, preds)
        assert loss.item() > 0.0

    def test_entropy_coefficient_no_decay(self):
        ppo = self._make_ppo()
        ppo.config.entropy_decay = False
        ppo.entropy_coefficient = 0.05
        assert ppo._get_entropy_coefficient(step=500, total_grad_steps=1000) == 0.05

    def test_entropy_coefficient_decays(self):
        ppo = self._make_ppo()
        ppo.config.entropy_decay = True
        ppo.entropy_coefficient = 0.1
        coef_early = ppo._get_entropy_coefficient(step=0, total_grad_steps=1000)
        coef_late = ppo._get_entropy_coefficient(step=900, total_grad_steps=1000)
        assert coef_early > coef_late


# ---------------------------------------------------------------------------
# 4. Log-prob / entropy shape tests through PPO internals
# ---------------------------------------------------------------------------


class TestLogProbShapes:
    """Verify that _get_log_prob_and_entropy returns correct shapes."""

    def test_single_discrete(self):
        config = SingleDiscreteConfig()
        env = make_cartpole()
        ppo = PPO(config, env, actor=None, critic=None, make_env=make_cartpole)

        obs = torch.randn(32, config.obs_dim)
        actions = torch.randint(0, config.act_dim, (32,))
        lp, ent = ppo._get_log_prob_and_entropy(obs, actions)
        assert lp.shape == (32,), f"Expected (32,), got {lp.shape}"
        assert ent.shape == (32,), f"Expected (32,), got {ent.shape}"

    def test_multi_discrete(self):
        config = MultiDiscreteConfig()
        env = MultiDiscreteCartpole()
        ppo = PPO(config, env, actor=None, critic=None, make_env=make_multi_cartpole)

        obs = torch.randn(32, config.obs_dim)
        actions = torch.randint(0, config.act_dim, (32, config.act_shape))
        lp, ent = ppo._get_log_prob_and_entropy(obs, actions)
        assert lp.shape == (32,), f"Expected (32,), got {lp.shape}"
        # entropy is (batch, act_shape) — not summed, just averaged downstream
        assert ent.shape == (32, config.act_shape), f"Expected (32, 2), got {ent.shape}"


# ---------------------------------------------------------------------------
# 5. Integration: run a few gradient steps without crashing
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_single_discrete_training_runs(self):
        config = SingleDiscreteConfig()
        env = make_cartpole()
        ppo = PPO(config, env, actor=None, critic=None, make_env=make_cartpole)
        ppo.run_batch(total_gradient_steps=4)

    def test_multi_discrete_training_runs(self):
        config = MultiDiscreteConfig()
        env = MultiDiscreteCartpole()
        ppo = PPO(config, env, actor=None, critic=None, make_env=make_multi_cartpole)
        ppo.run_batch(total_gradient_steps=4)

    def test_sample_batch_shapes_single(self):
        config = SingleDiscreteConfig()
        env = make_cartpole()
        ppo = PPO(config, env, actor=None, critic=None, make_env=make_cartpole)
        obs, actions, reward, done, log_probs, _ = ppo._sample_batch()
        T = config.T
        assert obs.shape == (T, config.obs_dim)
        assert actions.shape == (T,)
        assert reward.shape == (T,)
        assert done.shape == (T,)
        assert log_probs.shape == (T,)

    def test_sample_batch_shapes_multi(self):
        config = MultiDiscreteConfig()
        env = MultiDiscreteCartpole()
        ppo = PPO(config, env, actor=None, critic=None, make_env=make_multi_cartpole)
        obs, actions, reward, done, log_probs, _ = ppo._sample_batch()
        T = config.T
        assert obs.shape == (T, config.obs_dim)
        assert actions.shape == (T, config.act_shape)
        assert reward.shape == (T,)
        assert done.shape == (T,)
        assert log_probs.shape == (T,)

    def test_critic_value_matches_batch(self):
        """Critic output squeezed to (T,) should match reward/done shape."""
        config = SingleDiscreteConfig()
        env = make_cartpole()
        ppo = PPO(config, env, actor=None, critic=None, make_env=make_cartpole)
        obs, *_ = ppo._sample_batch()
        value = ppo.critic(obs).squeeze(-1)
        assert value.shape == (config.T,)
