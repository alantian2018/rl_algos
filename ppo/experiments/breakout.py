import gymnasium
import draccus
from dataclasses import dataclass
import numpy as np
import ale_py
from common import NormalizeObsWrapper
from ppo import PPOConfig, PPO, ImageActor, ImageCritic


def make_breakout_env(render_mode=None, normalize=True, obs_type="grayscale"):
    env = gymnasium.make("ALE/Breakout-v5", obs_type=obs_type, render_mode=render_mode,)
    if not normalize:
        return env
    return NormalizeObsWrapper(env)


@dataclass 
class BreakoutConfig(PPOConfig):
    exp_name: str = "breakout"

    obs_dim: tuple = (210,160,1)
    act_dim: int = 4

    actor_hidden_size: int = 128
    critic_hidden_size: int = 128  

    entropy_coefficient: float = 0.05
    entropy_decay: bool = True
    entropy_decay_steps: bool = 100_000
    minibatch_size: int = 64
    T: int = 2048
    epsilon: int = 0.1
    
    total_gradient_steps: int = 500_000
    frame_stack: int = 4
    

    video_log_freq: int = 2_000
    
    save_freq: int = 10_000

    device: str = 'cpu'

    path_to_checkpoint: str = "/Users/alantian/work/rl/ppo/checkpoints/breakout/20260128_152730/checkpoint_239999.pt"


@draccus.wrap()
def main(config: BreakoutConfig):
    env = make_breakout_env()
   
    obs, _ = env.reset()

    in_channels = config.obs_dim[2] * config.frame_stack
    height = obs.shape[0]
    width =  obs.shape[1]
   
    actor = ImageActor(
        in_channels=in_channels,
        height=height,
        width=width,
        act_dim=config.act_dim,
        hidden_size=config.actor_hidden_size,
    )
    critic = ImageCritic(
        in_channels=in_channels,
        height=height,
        width=width,
        hidden_size=config.critic_hidden_size,
    )
   
    ppo = PPO(config, env, actor, critic, make_env=make_breakout_env)
    ppo.run_batch(config.total_gradient_steps)


if __name__ == "__main__":
    main()
