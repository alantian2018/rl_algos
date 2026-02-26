from dataclasses import dataclass, asdict
from typing import Optional, Callable


from common.base_algorithm import BaseAlgorithm
from .networks import Policy, Qfunction
from .replaybuffer import ReplayBuffer
import torch.optim as optim
import gymnasium as gym
from tqdm import tqdm
import numpy as np
import torch
from termcolor import colored
import torch.nn.functional as F
from .config import SACConfig
from copy import deepcopy


class SAC(BaseAlgorithm):
    def __init__(
        self,
        config: SACConfig,
        policy: Policy,
        qf1: Qfunction,
        qf2: Qfunction,
        env: gym.Env,
        make_env: Optional[Callable[..., gym.Env]] = None,
    ):
        super().__init__(config=config, env=env, make_env=make_env)

        self.policy = policy.to(self.device)
        self.qf1 = qf1.to(self.device)
        self.qf2 = qf2.to(self.device)

        self.target_qf1 = deepcopy(qf1).to(self.device)
        self.target_qf2 = deepcopy(qf2).to(self.device)
        # just to be safe load state dict
        self.target_qf1.load_state_dict(self.qf1.state_dict())
        self.target_qf2.load_state_dict(self.qf2.state_dict())

        self.optimizer_policy = optim.Adam(self.policy.parameters(), lr=config.actor_lr)
        self.optimizer_qf1 = optim.Adam(self.qf1.parameters(), lr=config.critic_lr)
        self.optimizer_qf2 = optim.Adam(self.qf2.parameters(), lr=config.critic_lr)

        self.replay_buffer = ReplayBuffer(
            capacity=config.replay_buffer_capacity,
            obs_dim=self.obs_dim,
            action_dim=config.action_dim,
        )

        self.old_obs, _ = self.env.reset()

        if config.autotune_entropy:
            if config.target_entropy is None:
                print(
                    colored(f"Using {-config.action_dim} as target entropy", "yellow")
                )
                self.target_entropy = -config.action_dim
            else:
                self.target_entropy = config.target_entropy
            self.log_alpha = torch.tensor(0.0, requires_grad=True, device=self.device)
            self.log_alpha_optimizer = optim.Adam([self.log_alpha], lr=config.alpha_lr)
            self.alpha = torch.exp(self.log_alpha).item()

        else:
            assert config.alpha is not None
            self.alpha = config.alpha

        # Episode tracking
        self.episode_return = 0.0
        self.episode_returns = []
        loaded = self.load_ckpt_if_needed()
        if loaded and "replay_buffer" in loaded:
            self.replay_buffer = loaded["replay_buffer"]
            print(
                colored(
                    f"Loaded replay buffer of length {len(self.replay_buffer)}", "green"
                )
            )

    def _maybe_autotune_entropy(self, obs):
        if self.config.autotune_entropy:
            with torch.no_grad():
                _, log_probs = self.policy.get_action(obs)
            alpha_loss = (
                -self.log_alpha.exp() * (log_probs + self.target_entropy)
            ).mean()
            self.log_alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.log_alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().item()

    def _get_networks(self):
        networks = {
            "qf1": self.qf1,
            "qf2": self.qf2,
            "qf1_optimizer": self.optimizer_qf1,
            "qf2_optimizer": self.optimizer_qf2,
            "target_qf1": self.target_qf1,
            "target_qf2": self.target_qf2,
            "policy": self.policy,
            "policy_optimizer": self.optimizer_policy,
            "replay_buffer": self.replay_buffer,
        }
        if self.config.autotune_entropy:
            networks["log_alpha"] = self.log_alpha
            networks["log_alpha_optimizer"] = self.log_alpha_optimizer

        return networks

    def collect_rollout(self):
        with torch.no_grad():
            for _ in range(self.config.collect_rollout_steps):
                obs_tensor = torch.tensor(self.old_obs, dtype=torch.float32).to(
                    self.device
                )
                action, _ = self.policy.get_action(obs_tensor)
 

                try:
                    next_obs, reward, terminated, truncated, _ = self.env.step(
                        action.detach().cpu().numpy()
                    )
                except:
                    next_obs, reward, terminated, truncated, _ = self.env.step(
                        action.squeeze(0).detach().cpu().numpy()
                    )

                self.replay_buffer.add(
                    obs=self.old_obs.copy(),
                    action=action.detach().cpu().numpy(),
                    reward=reward,
                    next_obs=next_obs,
                    terminated=terminated,
                    truncated=truncated,
                )
                self.old_obs = next_obs

                # Track episode return
                self.episode_return += reward

                if terminated or truncated:
                    self.episode_returns.append(self.episode_return)
                    self.episode_return = 0.0
                    self.old_obs, _ = self.env.reset()

    def calculate_q_target(
        self,
        next_obs: np.ndarray,
        reward: np.ndarray,
        terminated: np.ndarray,
        truncated: np.ndarray,
    ):
        with torch.no_grad():
            dones = terminated  # | truncated

            next_obs_tensor = torch.tensor(next_obs, dtype=torch.float32).to(
                self.device
            )
            reward_tensor = torch.tensor(reward, dtype=torch.float32).to(self.device)
            dones_tensor = torch.tensor(dones, dtype=torch.float32).to(self.device)

            sampled_actions, log_probs = self.policy.get_action(next_obs_tensor)

            q1 = self.target_qf1(next_obs_tensor, sampled_actions)
            q2 = self.target_qf2(next_obs_tensor, sampled_actions)

            soft_v = torch.min(q1, q2) - self.alpha * log_probs

            return reward_tensor + self.config.gamma * (1 - dones_tensor) * soft_v

    def calculate_policy_target(self, obs_tensor: np.ndarray):

        actions, log_probs = self.policy.get_action(obs_tensor)

        q1 = self.qf1(obs_tensor, actions)
        q2 = self.qf2(obs_tensor, actions)

        return torch.min(q1, q2) - (self.alpha * log_probs)

    def update_qf(self, qf1_loss, qf2_loss):
        self.optimizer_qf1.zero_grad()
        self.optimizer_qf2.zero_grad()
        l = qf1_loss + qf2_loss
        l.backward()
        self.optimizer_qf1.step()
        self.optimizer_qf2.step()

    def update_targets(self):
        tau = self.config.tau
        with torch.no_grad():
            for target_param, param in zip(
                self.target_qf1.parameters(), self.qf1.parameters()
            ):
                target_param.data.copy_(
                    tau * param.data + (1 - tau) * target_param.data
                )
            for target_param, param in zip(
                self.target_qf2.parameters(), self.qf2.parameters()
            ):
                target_param.data.copy_(
                    tau * param.data + (1 - tau) * target_param.data
                )

    def train(self, total_train_steps: int):

        assert len(self.config.action_low) == len(self.config.action_high) == self.config.action_dim

        for (i,j) in zip (self.config.action_low, self.config.action_high):
            assert (
                i < j
            ), "action_low must be less than action_high"

        while len(self.replay_buffer) < self.config.before_training_steps:
            self.collect_rollout()

        for step in tqdm(range(total_train_steps), desc="Training"):
            # Maybe record video
            self.logger.maybe_record_video(self.policy, step, self.device)

            # always collect rollout
            self.episode_returns = []
            self.collect_rollout()

            # Log rollout stats
            self.logger.log_rollout(self.episode_returns, step=step)

            # if replay buffer is not full, collect more rollouts
            while len(self.replay_buffer) < self.config.batch_size:
                self.collect_rollout()

            qf1_loss_total = 0.0
            qf2_loss_total = 0.0
            policy_loss_total = 0.0

            for _ in range(self.config.gradient_step_ratio):
                (
                    obs,
                    action,
                    reward,
                    next_obs,
                    terminated,
                    truncated,
                ) = self.replay_buffer.sample(self.config.batch_size)

                obs_tensor = torch.tensor(obs, dtype=torch.float32).to(self.device)
                action_tensor = torch.tensor(action, dtype=torch.float32).to(
                    self.device
                )

                q_target = self.calculate_q_target(
                    next_obs, reward, terminated, truncated
                ).detach()

                qf1_loss = F.mse_loss(self.qf1(obs_tensor, action_tensor), q_target)
                qf2_loss = F.mse_loss(self.qf2(obs_tensor, action_tensor), q_target)
                self.update_qf(qf1_loss, qf2_loss)

                policy_target = self.calculate_policy_target(obs_tensor)
                # we need to push towards the right way ie minimize the loss here. no mse since we need to move
                # towards this direction, we don't want penalty for being too low.
                policy_loss = -policy_target.mean()
                self.optimizer_policy.zero_grad()
                policy_loss.backward()
                self.optimizer_policy.step()

                self._maybe_autotune_entropy(obs_tensor)
                self.update_targets()

                qf1_loss_total += qf1_loss.item()
                qf2_loss_total += qf2_loss.item()
                policy_loss_total += policy_loss.item()

            # Log training metrics
            if step % self.config.log_freq == 0:
                self.logger.log_training(
                    {
                        "qf1_loss": qf1_loss_total / self.config.gradient_step_ratio,
                        "qf2_loss": qf2_loss_total / self.config.gradient_step_ratio,
                        "policy_loss": policy_loss_total
                        / self.config.gradient_step_ratio,
                        "alpha": self.alpha,
                    },
                    step=step,
                )
            self.logger.maybe_save_checkpoint(step=step, networks=self._get_networks())
        self.logger.save_checkpoint(
            step=total_train_steps, networks=self._get_networks()
        )
        self.logger.finish()
