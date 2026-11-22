import torch
import imageio.v2 as imageio
from pathlib import Path

from .utils import prepare_observation


@torch.no_grad()
def evaluate_policy(agent, num_episodes=5, seed=None, deterministic=True, reward_key=None,
                    video_path=None, video_fps=30):
    if agent.make_env is None:
        raise ValueError("Evaluation requires a make_env factory")
    if num_episodes < 1:
        raise ValueError("num_episodes must be positive")
    env = agent.make_env()
    if env is agent.env:
        raise ValueError("make_env must create a separate evaluation environment")
    model_training, actor_training = agent.world_model.training, agent.actor.training
    device = torch.device(agent.device)
    cuda_devices = [device.index or 0] if device.type == "cuda" else []
    mps_rng = torch.mps.get_rng_state() if device.type == "mps" else None
    episode_returns = []
    writer = None
    try:
        if video_path is not None:
            video_path = Path(video_path)
            video_path.parent.mkdir(parents=True, exist_ok=True)
            writer = imageio.get_writer(str(video_path), fps=video_fps, macro_block_size=1)
        agent.world_model.eval()
        agent.actor.eval()
        with torch.random.fork_rng(devices=cuda_devices):
            for episode in range(num_episodes):
                if seed is not None:
                    torch.manual_seed(seed + episode)
                    obs, _ = env.reset(seed=seed + episode)
                else:
                    obs, _ = env.reset()

                state = agent.world_model.initial_state(1)
                previous_action = state.deterministic.new_zeros(1, agent.config.action_dim)
                total = 0.0
                done = False
                while not done:
                    if writer is not None:
                        frame = env.render()
                        if frame is None:
                            raise ValueError("Video evaluation requires an environment rendering RGB arrays")
                        writer.append_data(frame)
                    _, obs_tensor = prepare_observation(obs, agent.config, next(agent.world_model.parameters()).dtype)
                    state, _, _ = agent.world_model.observe_step(obs_tensor, state, previous_action)
                    action, _, _ = agent.actor.get_action(state, deterministic=deterministic)
                    action_array = action[0].cpu().numpy().astype(env.action_space.dtype, copy=True)
                    obs, reward, terminated, truncated, info = env.step(action_array)
                    previous_action = action
                    total += float(info[reward_key] if reward_key is not None else reward)
                    done = terminated or truncated
                episode_returns.append(total)
                if writer is not None:
                    writer.append_data(env.render())
                    writer.close()
                    writer = None
    finally:
        if writer is not None:
            writer.close()
        agent.world_model.train(model_training)
        agent.actor.train(actor_training)
        if mps_rng is not None:
            torch.mps.set_rng_state(mps_rng)
        env.close()
    return episode_returns
