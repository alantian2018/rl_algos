from typing import Callable, Optional
import gymnasium 
from .config import GlobalConfig
from datetime import datetime
from .utils import Logger
import termcolor 


class BaseAlgorithm:
    def __init__(
            self,
            config: GlobalConfig, 
            env: gymnasium.Env,
            make_env: Optional[Callable[..., gymnasium.Env]] = None,
    ):
        self.config = config
        self.env = env

        save_dir = None

        if config.save_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir = f"{config.save_dir}/{timestamp}"


        dummy_obs, _ = env.reset()
        self.raw_obs_dim = dummy_obs.shape
        self.obs_dim = self.raw_obs_dim[:-1] + (self.raw_obs_dim[-1] * self.config.frame_stack,)
        self.logger = Logger(config, make_env=make_env, save_dir=save_dir)
       

        print(termcolor.colored("="*100, 'green'))
        print(termcolor.colored("Config: ", 'green'))   
        for key, value in self.config.__dict__.items():
            print(termcolor.colored(f"  {key}: {value}", 'green'))
        print(termcolor.colored(f'Saving to: {save_dir}', 'green'))
        print(termcolor.colored("="*100, 'green'))
    
        self.episode_return = 0.0
        self.episode_length = 0

    def load_ckpt_if_needed(self):
        if self.config.path_to_checkpoint:
            path = self.config.path_to_checkpoint
            self.logger.load_checkpoint(path = path,
                                        networks=self._get_networks())
            print(termcolor.colored(f'loaded ckpt from {path}', 'green'))
    
    def _get_networks(self):
        raise NotImplementedError
    
    def save_ckpt(self, step: int):
        # Save final checkpoint
        if self.config.save_freq is not None:
            self.logger.save_checkpoint(
                networks=self._get_networks(),
                step=step
            )