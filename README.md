# RL

My personal implementations of reinforcement learning algorithms.

---

<table align="center">
  <tr>
    <td align="center">
      <img src="ppo/gifs/snake.gif" width="200"><br>
      <sub>Snake</sub>
    </td>
    <td align="center">
      <img src="ppo/gifs/carracing.gif" width="300"><br>
      <sub>Car Racing</sub>
    </td>
  </tr>
</table>


### Installation
```
conda create -y -n "rl" python=3.10
conda activate rl
pip install -r requirements.txt
```

Then to run:

```
python -m ppo.experiments.[INSERT_EXP_NAME]

e.g.
python -m ppo.experiments.carracing
```
## Algorithms

### PPO (Proximal Policy Optimization)
> See [ppo/ppo/algorithm.py](ppo/ppo/algorithm.py) for the PPO implementation and [ppo/ppo/gae.py](ppo/ppo/gae.py) for Generalized Advantage Estimation.

### SAC (Soft Actor-Critic) ***[WIP]***
> See [sac/sac.py](sac/sac.py) for the SAC implementation

---


## Results on `Wandb`!

### PPO
- [Car Racing](https://wandb.ai/alantian2018/ppo-carracing)
- [Snake](https://wandb.ai/alantian2018/ppo-snake/runs/j1jym4xk)
- [CartPole](https://wandb.ai/alantian2018/ppo-cartpole?nw=nwuseralantian2018)
- [LunarLander](https://wandb.ai/alantian2018/ppo-lunarlander?nw=nwuseralantian2018)
- [Acrobot](https://wandb.ai/alantian2018/ppo-acrobot?nw=nwuseralantian2018)

### SAC
- [Pendulum](https://wandb.ai/alantian2018/sac-cartpole?nw=nwuseralantian2018)

---
Some of the boilerplate (`eg configs, wandb logging, utils, etc`) were handled by opus 4.5 :)
