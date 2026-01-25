import torch
import torch.nn.functional as F
from torch.optim import Adam
import gymnasium
from tqdm import tqdm
from typing import Optional, Callable

from .config import PPOConfig
from .networks import Actor, Critic
from .gae import gae
import termcolor
from common import BaseAlgorithm

class PPO(BaseAlgorithm):
    def __init__(self, config: PPOConfig,
     env: gymnasium.Env,
     actor: None | Actor,
     critic: None | Critic,
     make_env: Optional[Callable[..., gymnasium.Env]] = None,):
        
        super().__init__(config, env, make_env=make_env)

        if actor is None:
            assert isinstance(config.obs_dim, int), \
                termcolor.colored("the default actor only works for linear input dimensions. Please bring your own actor", 'yellow')
            self.actor = Actor(config.obs_dim, config.act_dim, config.actor_hidden_size)
        else:
            assert actor is not None
            self.actor = actor
        if critic is None:
            assert isinstance(config.obs_dim, int), \
                termcolor.colored("the default critic only works for linear input dimensions. Please bring your own critic", 'yellow')
            self.critic = Critic(config.obs_dim, config.critic_hidden_size)
        else:
            assert critic is not None
            self.critic = critic

    
        self.actor = actor.to(config.device)
        self.critic = critic.to(config.device)
        self.actor_optimizer = Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=config.critic_lr)
        self.T = config.T
        self.entropy_coefficient = config.entropy_coefficient
       
        self.minibatch_size = config.minibatch_size

        self.cur_obs = None

        self.load_ckpt_if_needed()

        

    def _calculate_advantages(self, rewards: torch.Tensor, dones: torch.Tensor, values: torch.Tensor, gamma: float, gae_lambda: float):
        return gae(rewards, dones, values, gamma, gae_lambda)
    
    def _sample_batch(self):
        """Sample a batch of data from the environment. (T timesteps)"""
        
        with torch.no_grad():
            

            obs = torch.zeros((self.T,) + self.obs_dim, device=self.config.device)
            action = torch.zeros(self.T, device=self.config.device)
            reward = torch.zeros(self.T, device=self.config.device)
            done = torch.zeros(self.T, device=self.config.device)
            log_probs = torch.zeros(self.T, device=self.config.device)
            
            if self.cur_obs is None:
                self.cur_obs, _ = self.env.reset()
                self.cur_obs = torch.tensor(self.cur_obs, dtype=torch.float32, device=self.config.device)

            episode_returns = []
            for t in range(self.T):
                distribution = self.actor(self.cur_obs)
                actions = distribution.sample()
             
                log_probs_ = distribution.log_prob(actions)
                next_obs, rewards, terminated, truncated, _ = self.env.step(actions.squeeze(0).cpu().numpy())
                
                next_obs = torch.tensor(next_obs, dtype=torch.float32, device=self.config.device)

                obs[t] = self.cur_obs
                action[t] = actions
                reward[t] = rewards
                done[t] = terminated or truncated
                log_probs[t] = log_probs_
                self.cur_obs = next_obs
                
                # Track episode stats
                self.episode_return += rewards
                self.episode_length += 1

                if done[t]:
                    episode_returns.append(self.episode_return)
                    self.episode_return = 0.0
                    self.episode_length = 0
                    self.cur_obs, _ = self.env.reset()
                    self.cur_obs = torch.tensor(self.cur_obs, dtype=torch.float32, device=self.config.device)

                    
            return obs, action, reward, done, log_probs.detach(), episode_returns

    def _get_log_prob_and_entropy(self, obs, actions):
        """Get the log probability and entropy of the actions, needed to check distributional shift"""
        distribution = self.actor(obs)
        log_probs = distribution.log_prob(actions)
        entropy = distribution.entropy()
        return log_probs, entropy

    def _actor_loss(self, advantages: torch.Tensor, old_log_probs: torch.Tensor, new_log_probs: torch.Tensor, epsilon: float = 0.2):
        """
        ratio = pi_new(a|s) / pi_old(a|s) -> how much our policy changed. 
        advantage = how much better our new policy is compared to the old policy

        By taking a product of ratio * advantage, you reward for distributions that are more likely to 
        pick high advantage actions.
        """
        ratio = torch.exp(new_log_probs - old_log_probs) 
        # clip the ratio to prevent extreme changes in policy.
        clipped_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
        surrogate = torch.min(advantages * ratio, advantages * clipped_ratio)
        # negate because we want to maximize the expected ratio * advantage, so we minimize the loss.
        return -surrogate.mean()

    def _critic_loss(self, v_target: torch.Tensor, values: torch.Tensor):
        """MSE on current critic values, and v_target = advantages + values (e.g. Q func)"""
        return F.mse_loss(v_target, values)
            
    def _get_networks(self):
        return {
            'actor': self.actor,
            'critic': self.critic,
            'actor_optimizer': self.actor_optimizer,
            'critic_optimizer': self.critic_optimizer
        }

    def _get_entropy_coefficient(self, step, total_grad_steps):
        if not self.config.entropy_decay:
            return self.entropy_coefficient
        else:
            return self.entropy_coefficient * (1 - (step / total_grad_steps))

    
    def run_batch(self, total_gradient_steps: int):
        t = 0
        pbar = tqdm(total=total_gradient_steps)
        
        while t < total_gradient_steps:
            # Record video if needed
            self.logger.maybe_record_video(self.actor, t, self.config.device)
            
            obs, actions, reward, done, old_log_probs, episode_returns = self._sample_batch()      

            
            # Log episode returns
            self.logger.log_rollout(episode_returns, step=t)

            """use values as a proxy for Advantage function. We glue this down and treat it as an oracle."""
            value = self.critic(obs).squeeze(-1).detach()
            advantages = self._calculate_advantages(reward, done, value, self.config.gamma, self.config.gae_lambda)
            v_target = (advantages + value).detach()
            
            # run multiple epochs over the same batch of data
            for epoch in range(self.config.epochs_per_batch):
                # split each batch of data into minibatches for stability.
                for i in range(0, self.T, self.minibatch_size):
                    batch_obs = obs[i:i+self.minibatch_size]
                    batch_actions = actions[i:i+self.minibatch_size]
                    batch_v_target = v_target[i:i+self.minibatch_size]
                    batch_log_probs = old_log_probs[i:i+self.minibatch_size]
                    batch_advantages = advantages[i:i+self.minibatch_size]
                    normalized_batch_advantages = (batch_advantages - batch_advantages.mean()) / (batch_advantages.std() + 1e-8)

                    """
                    great, now we have a new actor policy after the 2nd minibatch
                    we need the log probabilities of commiting those actions. 
                    If you look at the loss, you see we want to maximize the advantage,
                    so we need to maximize the probability of commiting beneifical actions with high actions.

                    We compare to old distribution to ensure distributional shift is small/clipped (for stability).
                    """
                    ecoef=self._get_entropy_coefficient(step=t, total_grad_steps=total_gradient_steps)
                    new_log_probs, entropy = self._get_log_prob_and_entropy(batch_obs, batch_actions)
                    actor_loss = self._actor_loss(
                        advantages=normalized_batch_advantages, 
                        old_log_probs=batch_log_probs, 
                        new_log_probs=new_log_probs, 
                        epsilon=self.config.epsilon
                        # you want higher entropy for exploration, otherwise you'll be too greedy.
                    ) - entropy.mean() * ecoef

                    """
                    Now the actor is done, we need to recalibrate the critic to the new values.
                    Again, we treat our advantages as an oracle.
                    A = Q - V
                    V_target = A + V = Q - V + V -> V_target is a proxy for Q

                    Our dream would be to estimate Q directly but thats hard,
                    so we estimate for V because a high Q must have a high V.

                    Then our actor, using critic V, will try to get the best action to maximize Q.
                    """
                    new_critic_value = self.critic(batch_obs).squeeze(-1)
                    critic_loss = self._critic_loss(v_target=batch_v_target, values=new_critic_value)
                    
                    self.actor_optimizer.zero_grad()
                    self.critic_optimizer.zero_grad()
                    actor_loss.backward()
                    critic_loss.backward()
                    self.actor_optimizer.step()
                    self.critic_optimizer.step()
                    
                    # Log training metrics
                    self.logger.log_training(
                        {
                            'actor_loss': actor_loss.item(),
                            'critic_loss': critic_loss.item(),
                            'entropy' : entropy.mean().item(),
                            'entropy_coef': ecoef,
                        },
                        step=t
                    )
                    # Save checkpoint if needed
                    self.logger.maybe_save_checkpoint(t, self._get_networks())
             
                    
                    pbar.update(1)
                    t += 1
                    if t == total_gradient_steps:
                        break
                if t == total_gradient_steps:
                    break
        
        pbar.close()
        
        self.save_ckpt(step=t)
        
        self.logger.finish()

