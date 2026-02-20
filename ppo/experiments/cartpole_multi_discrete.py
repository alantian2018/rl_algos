import gymnasium
import draccus
from dataclasses import dataclass
import torch
from ppo import PPO, PPOConfig
from ppo import Actor, Critic
import numpy as np
from functools import partial
# for testing my multi discrete implementation of ppo....


class CartpoleMultiDiscrete(gymnasium.Env):
    def __init__(self, num_envs = 2, render_mode = 'rgb_array'):
        self.envs = [
            gymnasium.make("CartPole-v1", render_mode=render_mode) for _ in range(num_envs)
        ]
        self.render_mode = render_mode

    def step(self, action):
        obs = []
        reward = 0
        terminated = False
        truncated = False
      
        for c,env in enumerate(self.envs):
             
            o, r, te, tr, _  = env.step(action[c])
            obs.append(o)
            reward +=r 
            terminated = te or terminated
            truncated = tr or truncated
        return np.concatenate(obs), reward / len(self.envs), terminated, truncated, {}
    
    def render(self, render_mode = None):
        if render_mode is None:
            render_mode = self.render_mode
        assert render_mode is not None
        # concatenate all
        images = [env.render() for env in self.envs]
        return np.hstack(images)
    
    def reset(self):
        obs = []
        infos = []

        for i, env in enumerate(self.envs):
            o, info = env.reset()
            obs.append(o)
            infos.append(info)

        return np.concatenate(obs), {}
        
    def close(self):
        for env in self.envs:
            env.close()




def make_cartpole_env(num_envs, render_mode=None):
    return CartpoleMultiDiscrete(num_envs, render_mode)


# PLZ DO NOT FORGET TYPE HINTS
@dataclass
class CartPoleConfig(PPOConfig):
    """CartPole-specific config."""

    exp_name: str = "cartpole_multi_discrete"
    obs_dim: int = 4 * 2
    act_dim: int = 2
    act_shape: int = 2
    actor_hidden_size: int = 32
    critic_hidden_size: int = 32
    frame_stack: int = 4
    # Training
    total_gradient_steps: int = 50_000
    video_log_freq: int = 5000

    save_freq: int = 5000


@draccus.wrap()
def main(config: CartPoleConfig):

    env = make_cartpole_env(num_envs=config.act_shape)

    actor = Actor(
        config.obs_dim * config.frame_stack, config.act_dim, config.actor_hidden_size, act_shape = config.act_shape
    )
    critic = Critic(config.obs_dim * config.frame_stack, config.critic_hidden_size)

    make_env_fn = partial(make_cartpole_env, num_envs = config.act_shape)
    ppo = PPO(config, env, actor, critic, make_env=make_env_fn)
    ppo.run_batch(total_gradient_steps=config.total_gradient_steps)


if __name__ == "__main__":
    main()
