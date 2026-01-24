from dataclasses import dataclass, asdict
from typing import Optional, Callable
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sac.networks import Policy, Qfunction
from sac.replaybuffer import ReplayBuffer, Step
import torch.optim as optim
import gymnasium as gym
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from utils import SACLogger 
from config import SACConfig

class SAC:
    def __init__(
        self, 
        config: SACConfig, 
        env: gym.Env,
        make_env: Optional[Callable[..., gym.Env]] = None,
    ):
        self.config = config
        self.device = config.device
        
        self.policy = Policy(config.state_dim, config.action_dim, config.action_low, config.action_high, config.hidden_dim).to(self.device)

        self.qf1 = Qfunction(config.state_dim, config.action_dim, config.hidden_dim).to(self.device)
        self.qf2 = Qfunction(config.state_dim, config.action_dim, config.hidden_dim).to(self.device)

        self.target_qf1 = Qfunction(config.state_dim, config.action_dim, config.hidden_dim).to(self.device)
        self.target_qf2 = Qfunction(config.state_dim, config.action_dim, config.hidden_dim).to(self.device)
        self.target_qf1.load_state_dict(self.qf1.state_dict())
        self.target_qf2.load_state_dict(self.qf2.state_dict())


        self.optimizer_policy = optim.Adam(self.policy.parameters(), lr=config.actor_lr)
        self.optimizer_qf1 = optim.Adam(self.qf1.parameters(), lr=config.critic_lr)
        self.optimizer_qf2 = optim.Adam(self.qf2.parameters(), lr=config.critic_lr)

        self.env = env
        self.replay_buffer = ReplayBuffer(capacity=config.replay_buffer_capacity)

        self.old_obs, _ = self.env.reset()
        
        # Episode tracking
        self.episode_return = 0.0
        self.episode_returns = []
        
        # Logger
        self.logger = SACLogger(config, make_env=make_env)
        
        
    def collect_rollout(self):
       
        for _ in range(self.config.collect_rollout_steps):
            obs_tensor = torch.tensor(self.old_obs, dtype=torch.float32).to(self.device)
            action, _ = self.policy.get_action(obs_tensor)
            if not self.config.is_continuous:
                action = torch.floor(action).to(torch.int32)

            next_obs, reward, terminated, truncated, _ = self.env.step(action.detach().cpu().numpy())                
            step = Step(self.old_obs.copy(), 
                        action.detach().cpu().numpy(),
                        reward,
                        next_obs.copy(),
                        terminated,
                        truncated
                    )
            
            self.replay_buffer.add(step)
            self.old_obs = next_obs
            
            # Track episode return
            self.episode_return += reward

            if terminated or truncated:
                self.episode_returns.append(self.episode_return)
                self.episode_return = 0.0
                self.old_obs, _ = self.env.reset()

    def calculate_q_target(self, next_obs: np.ndarray, reward: np.ndarray, terminated: np.ndarray, truncated: np.ndarray):
        with torch.no_grad():
            dones = terminated | truncated
            
            next_obs_tensor = torch.tensor(next_obs, dtype=torch.float32).to(self.device)
            reward_tensor = torch.tensor(reward, dtype=torch.float32).to(self.device).unsqueeze(-1)
            dones_tensor = torch.tensor(dones, dtype=torch.float32).to(self.device).unsqueeze(-1)

            sampled_actions, log_probs = self.policy.get_action(next_obs_tensor)
        
            q1 = self.target_qf1(next_obs_tensor, sampled_actions) 
            q2 = self.target_qf2(next_obs_tensor, sampled_actions) 

            soft_v = torch.min(q1, q2) - self.config.alpha * log_probs
        
            return reward_tensor + self.config.gamma * (1 - dones_tensor) * soft_v
    
    def calculate_policy_target(self, obs: np.ndarray):
        obs_tensor = torch.tensor(obs, dtype=torch.float32).to(self.device)
        
        actions, log_probs = self.policy.get_action(obs_tensor)
      
        q1 = self.qf1(obs_tensor, actions)
        q2 = self.qf2(obs_tensor, actions)
       
        return torch.min(q1, q2) - (self.config.alpha * log_probs)

    def update_qf(self, qf1_loss, qf2_loss):
        self.optimizer_qf1.zero_grad()
        self.optimizer_qf2.zero_grad()
        qf1_loss.backward()
        qf2_loss.backward()
        self.optimizer_qf1.step()
        self.optimizer_qf2.step()

    def update_targets(self):
        tau = self.config.tau
        with torch.no_grad():
            for target_param, param in zip(self.target_qf1.parameters(), self.qf1.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
            for target_param, param in zip(self.target_qf2.parameters(), self.qf2.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

   
    
    def train(self, total_train_steps: int):
        assert self.config.action_low < self.config.action_high, "action_low must be less than action_high"

        if not self.config.is_continuous:
            assert self.config.action_low == 0 and self.config.action_dim == 1, "action_low must be 0 and action dim must be 1"
            self.config.action_high -= 1e-6
            print('using torch.floor for action')
        
        while len(self.replay_buffer) < self.config.before_training_steps:
            self.collect_rollout()
        
        for step in tqdm(range(total_train_steps), desc='Training'):
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
                obs, action, reward, next_obs, terminated, truncated, _ = self.replay_buffer.sample(self.config.batch_size)
                
                obs_tensor = torch.tensor(obs, dtype=torch.float32).to(self.device)
                action_tensor = torch.tensor(action, dtype=torch.float32).to(self.device)
                
                q_target = self.calculate_q_target(next_obs, reward, terminated, truncated).detach()

                qf1_loss = F.mse_loss(self.qf1(obs_tensor, action_tensor), q_target)
                qf2_loss = F.mse_loss(self.qf2(obs_tensor, action_tensor), q_target)
                self.update_qf(qf1_loss, qf2_loss)

                policy_target = self.calculate_policy_target(obs)
                # we need to push towards the right way ie minimize the loss here. no mse since we need to move
                # towards this direction, we don't want penalty for being too low.
                policy_loss = -policy_target.mean()
                self.optimizer_policy.zero_grad()
                policy_loss.backward()
                self.optimizer_policy.step()

                self.update_targets()
                
                qf1_loss_total += qf1_loss.item()
                qf2_loss_total += qf2_loss.item()
                policy_loss_total += policy_loss.item()
            
            # Log training metrics
            if step % self.config.log_freq == 0:
                self.logger.log_training(
                    qf1_loss=qf1_loss_total / self.config.gradient_step_ratio,
                    qf2_loss=qf2_loss_total / self.config.gradient_step_ratio,
                    policy_loss=policy_loss_total / self.config.gradient_step_ratio,
                    step=step,
                )
        
        self.logger.finish()
