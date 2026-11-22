from typing import Callable, Optional

import gymnasium
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm

from common.base_algorithm import BaseAlgorithm
from .config import DreamerConfig
from .networks import Actor, Critic, DreamerState
from .replaybuffer import ReplayBuffer
from .worldmodel import WorldModel
from .returns import lambda_return, discount_weights
from .evaluation import evaluate_policy
from .utils import prepare_observation, update_return_scale


class Dreamer(BaseAlgorithm):
    def __init__(
        self,
        config: DreamerConfig,
        env: gymnasium.Env,
        world_model: Optional[WorldModel] = None,
        actor: Optional[Actor] = None,
        critic: Optional[Critic] = None,
        make_env: Optional[Callable[..., gymnasium.Env]] = None,
    ):
        if not isinstance(env.action_space, gymnasium.spaces.Box):
            raise ValueError(
                "The current Dreamer actor requires continuous Box actions"
            )
        if env.action_space.shape != (config.action_dim,):
            raise ValueError("Environment action shape must match action_dim")
        if config.frame_stack != 1:
            raise ValueError(
                "Use frame_stack=1; supply any stacked channels through the environment"
            )
        super().__init__(config, env, make_env)
        self.make_env = make_env

        if config.seed:
            torch.manual_seed(config.seed)
            env.action_space.seed(config.seed)

        if world_model is None:
            world_model = WorldModel(
                config.deterministic_dim,
                config.stochastic_dim,
                config.action_dim,
                config.obs_dim,
                config.obs_channels,
                config.latent_obs_size,
                config.hidden_dim,
                (
                    config.reconstruction_weight,
                    config.reward_weight,
                    config.continuation_weight,
                    config.kl_weight,
                ),
                config.free_nats,
                edge_weight=config.edge_weight,
            )
        if actor is None:
            actor = Actor(
                config.deterministic_dim,
                config.stochastic_dim,
                config.hidden_dim,
                config.action_dim,
                config.action_low,
                config.action_high,
                config.squash_actions,
            )
        if critic is None:
            critic = Critic(
                config.deterministic_dim, config.stochastic_dim, config.hidden_dim
            )

        self.world_model = world_model.to(self.device)
        self.actor = actor.to(self.device)
        self.critic = critic.to(self.device)
        self.model_optimizer = Adam(self.world_model.parameters(), lr=config.model_lr)
        self.actor_optimizer = Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=config.critic_lr)
        self.replay_buffer = ReplayBuffer(config.replay_buffer_capacity, config.seed)
        self.env_steps = 0
        self.gradient_steps = 0
        self.return_quantiles = None
        loaded = self.load_ckpt_if_needed()
        if loaded:
            self.env_steps = loaded["env_steps"]
            self.gradient_steps = loaded["gradient_steps"]
            self.logger.last_saved = self.env_steps
            self.return_quantiles = loaded["return_quantiles"]
            if self.return_quantiles is not None:
                self.return_quantiles = self.return_quantiles.to(self.device)
            if config.save_replay_buffer:
                self.replay_buffer = loaded["replay_buffer"]

    @torch.no_grad()
    def collect_episode(self, progress=None):
        """Collect a complete episode, preserving its final observation in replay.

        Accepts HWC or CHW images (and HW grayscale). Replay stores copied CHW
        images in their original dtype. uint8 inputs are divided by 255 only
        when passed to the model; floating inputs are assumed preprocessed.
        """
        obs, _ = self.env.reset(seed=self.config.seed if self.env_steps == 0 else None)
        state = self.world_model.initial_state(1)
        previous_action = state.deterministic.new_zeros(1, self.config.action_dim)
        observations, actions, rewards, terminated_flags, truncated_flags = (
            [],
            [],
            [],
            [],
            [],
        )
        self.episode_return = 0.0
        self.episode_length = 0
        done = False

        while True:
            image, obs_tensor = prepare_observation(
                obs, self.config, next(self.world_model.parameters()).dtype
            )
            observations.append(image)
            if done:
                break

            state, _, _ = self.world_model.observe_step(
                obs_tensor, state, previous_action
            )
            replay_warmup = min(
                self.config.before_training_steps, self.config.replay_buffer_capacity
            )
            if (
                self.env_steps < self.config.before_training_steps
                or self.replay_buffer.size + self.episode_length < replay_warmup
            ):
                action = np.array(self.env.action_space.sample(), copy=True)
            else:
                action_tensor, _, _ = self.actor.get_action(state)
                action = (
                    action_tensor[0]
                    .cpu()
                    .numpy()
                    .astype(self.env.action_space.dtype, copy=True)
                )

            # Keep our own copy in case the environment reuses or mutates arrays.
            actions.append(action.copy())
            obs, reward, terminated, truncated, _ = self.env.step(action)
            rewards.append(reward)
            terminated_flags.append(terminated)
            truncated_flags.append(truncated)
            previous_action = torch.as_tensor(
                actions[-1], device=self.device, dtype=state.deterministic.dtype
            ).unsqueeze(0)
            self.env_steps += 1
            if progress is not None:
                progress.update(1)
            self.episode_length += 1
            self.episode_return += float(reward)
            done = terminated or truncated

        self.replay_buffer.add_episode(
            observations, actions, rewards, terminated_flags, truncated_flags
        )
        return {
            "episode_return": self.episode_return,
            "episode_length": self.episode_length,
        }

    def evaluate(
        self,
        num_episodes=5,
        seed=10_000,
        deterministic=True,
        reward_key=None,
        video_path=None,
        video_fps=30,
    ):
        return evaluate_policy(
            self, num_episodes, seed, deterministic, reward_key, video_path, video_fps
        )

    def update_world_model(self, batch_size):
        data = self.replay_buffer.sample(batch_size, self.config.sequence_length)
        dtype = next(self.world_model.parameters()).dtype
        obs = torch.as_tensor(data["observations"], device=self.device, dtype=dtype)
        actions = torch.as_tensor(data["actions"], device=self.device, dtype=dtype)
        rewards = torch.as_tensor(data["rewards"], device=self.device, dtype=dtype)
        terminated = torch.as_tensor(
            data["terminated"], device=self.device, dtype=torch.bool
        )
        truncated = torch.as_tensor(
            data["truncated"], device=self.device, dtype=torch.bool
        )
        next_obs = torch.as_tensor(
            data["next_observations"], device=self.device, dtype=dtype
        )
        if data["observations"].dtype == np.uint8:
            obs = obs / 255.0
        if data["next_observations"].dtype == np.uint8:
            next_obs = next_obs / 255.0

        self.model_optimizer.zero_grad()
        loss, metrics, states = self.world_model.loss(
            obs,
            actions,
            rewards,
            terminated,
            truncated,
            next_obs,
        )
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.world_model.parameters(), self.config.grad_clip
        )
        self.model_optimizer.step()
        self.gradient_steps += 1
        metrics["world_model_loss"] = loss.detach()
        metrics["model_grad_norm"] = grad_norm.detach()
        valid_starts = ~terminated[..., 0]
        states = DreamerState(
            states.deterministic.detach()[valid_starts],
            states.stochastic.detach()[valid_starts],
        )
        return metrics, states

    @torch.no_grad()
    def imagine_rollout(self, states: DreamerState, imagine_horizon):
        if imagine_horizon < 1:
            raise ValueError("imagine_horizon must be positive")

        state = states.flatten()
        deterministic = [state.deterministic]
        stochastic = [state.stochastic]

        actions, rewards, continuations, raw_actions = [], [], [], []

        values = [self.critic(state)]
        for _ in range(imagine_horizon):
            action, _, info = self.actor.get_action(state)
            next_state, reward, continuation = self.world_model.step_env(state, action)
            next_value = self.critic(next_state)

            deterministic.append(next_state.deterministic)
            stochastic.append(next_state.stochastic)
            actions.append(action)
            rewards.append(reward)
            continuations.append(continuation)
            values.append(next_value)
            raw_actions.append(info["raw_action"])
            state = next_state

        imagined_states = DreamerState(
            torch.stack(deterministic, dim=1),
            torch.stack(stochastic, dim=1),
        )
        actions = torch.stack(actions, dim=1)
        rewards = torch.stack(rewards, dim=1)
        continuations = torch.stack(continuations, dim=1)
        values = torch.stack(values, dim=1)
        raw_actions = torch.stack(raw_actions, dim=1)
        return imagined_states, actions, rewards, continuations, values, raw_actions

    def update_actor_critic(
        self, imagined_states, actions, rewards, continuations, values, raw_actions
    ):

        with torch.no_grad():
            returns = lambda_return(
                rewards, continuations, values, self.config.gamma, self.config.lambda_
            )
            return_scale, self.return_quantiles = update_return_scale(
                returns, self.return_quantiles, self.config.return_norm_rate
            )
            advantages = (returns - values[:, :-1]) / return_scale
            weights = discount_weights(continuations, self.config.gamma)

        states = DreamerState(
            imagined_states.deterministic[:, :-1].detach(),
            imagined_states.stochastic[:, :-1].detach(),
        )
        log_probs = self.actor.get_log_prob(states, actions, raw_actions=raw_actions)
        entropy_actions, entropy_log_probs, actor_info = self.actor.get_action(states)
        entropy = -entropy_log_probs
        actor_loss = -(
            weights
            * (log_probs * advantages + self.config.entropy_coefficient * entropy)
        ).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_grad_norm = torch.nn.utils.clip_grad_norm_(
            self.actor.parameters(), self.config.grad_clip
        )
        self.actor_optimizer.step()

        predicted_values = self.critic(states)
        critic_loss = (
            weights * F.mse_loss(predicted_values, returns, reduction="none")
        ).mean()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_grad_norm = torch.nn.utils.clip_grad_norm_(
            self.critic.parameters(), self.config.grad_clip
        )
        self.critic_optimizer.step()

        metrics = {
            "actor_loss": actor_loss.detach(),
            "critic_loss": critic_loss.detach(),
            "actor_entropy": entropy.detach().mean(),
            "return_mean": returns.mean(),
            "return_scale": return_scale,
            "actor_grad_norm": actor_grad_norm.detach(),
            "critic_grad_norm": critic_grad_norm.detach(),
            "actor_mean_abs": actor_info["mu"].abs().mean(),
            "actor_mean": actor_info["mu"].mean(),
            "actor_std": actor_info["std"].mean(),
            "action_mean": actions.detach().mean(),
            "action_std": actions.detach().std(unbiased=False),
        }
        if self.actor.squash_actions:
            normalized_actions = (
                2
                * (entropy_actions.detach() - self.actor.action_low)
                / (self.actor.action_high - self.actor.action_low)
                - 1
            )
            metrics["actor_saturation_fraction"] = (
                (normalized_actions.abs() > 0.99).float().mean()
            )
        return metrics

    def _get_networks(self):
        networks = {
            "world_model": self.world_model,
            "actor": self.actor,
            "critic": self.critic,
            "model_optimizer": self.model_optimizer,
            "actor_optimizer": self.actor_optimizer,
            "critic_optimizer": self.critic_optimizer,
            "env_steps": self.env_steps,
            "gradient_steps": self.gradient_steps,
            "return_quantiles": self.return_quantiles,
        }
        if self.config.save_replay_buffer:
            networks["replay_buffer"] = self.replay_buffer
        return networks

    def train(self, total_train_steps):
        if total_train_steps < 1:
            raise ValueError("total_train_steps must be positive")
        last_logged_update = self.gradient_steps
        progress = tqdm(
            total=total_train_steps,
            initial=self.env_steps,
            desc="Training",
            unit="steps",
            mininterval=0,
            miniters=1,
        )
        try:
            while self.env_steps < total_train_steps:
                progress.set_description("Collecting episode")
                episode = self.collect_episode(progress)
                training_metrics = {}
                can_sample = any(
                    len(episode_data["actions"]) >= self.config.sequence_length
                    for episode_data in self.replay_buffer.episodes
                )
                replay_ready = self.replay_buffer.size >= min(
                    self.config.before_training_steps,
                    self.config.replay_buffer_capacity,
                )
                if (
                    self.env_steps >= self.config.before_training_steps
                    and replay_ready
                    and can_sample
                ):
                    for update in range(self.config.train_steps_per_episode):
                        update_label = (
                            f"{update + 1}/{self.config.train_steps_per_episode}"
                        )
                        progress.set_description(f"Dreaming... [{update_label}]")
                        metrics, states = self.update_world_model(
                            self.config.batch_size
                        )
                        if states.deterministic.shape[0] > 0:

                            rollout = self.imagine_rollout(
                                states, self.config.imagination_horizon
                            )
                            behavior_metrics = self.update_actor_critic(*rollout)
                            metrics.update(behavior_metrics)
                        for key, value in metrics.items():
                            training_metrics.setdefault(key, []).append(value.item())

                metrics_to_log = {
                    "rollout/episode_return": episode["episode_return"],
                    "rollout/episode_length": episode["episode_length"],
                    "replay/transitions": self.replay_buffer.size,
                    "train/gradient_steps": self.gradient_steps,
                }
                if (
                    training_metrics
                    and self.gradient_steps - last_logged_update >= self.config.log_freq
                ):
                    for key, values in training_metrics.items():
                        metrics_to_log[f"train/{key}"] = sum(values) / len(values)
                    last_logged_update = self.gradient_steps
                progress.set_description("Logging / evaluation")
                self.logger.log(metrics_to_log, step=self.env_steps)
                progress.set_description("Saving checkpoint")
                self.logger.maybe_save_checkpoint(
                    step=self.env_steps, networks=self._get_networks()
                )

            if (
                self.config.save_freq is not None
                and self.logger.last_saved != self.env_steps
            ):
                self.save_ckpt(self.env_steps)
                self.logger.last_saved = self.env_steps
            progress.set_description("Finished")
        finally:
            progress.close()
            self.logger.finish()
