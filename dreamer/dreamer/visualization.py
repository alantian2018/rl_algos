import numpy as np
import torch

from .networks import DreamerState


def make_reconstruction_grid(targets, reconstructions):
    actual_row = torch.cat(list(targets.detach().float().cpu()), dim=-1)
    reconstructed_row = torch.cat(list(reconstructions.detach().float().cpu()), dim=-1)
    grid = torch.cat((actual_row, reconstructed_row), dim=-2).clamp(0, 1)
    grid = (grid.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    return grid[..., 0] if grid.shape[-1] == 1 else grid


@torch.no_grad()
def reconstruct_replay(agent, num_frames=8):
    if num_frames < 1:
        raise ValueError("num_frames must be positive")
    if not agent.replay_buffer.episodes:
        return None
    episode = agent.replay_buffer.episodes[-1]
    length = min(agent.config.sequence_length, len(episode["actions"]))
    model = agent.world_model
    parameter = next(model.parameters())
    images = torch.as_tensor(
        episode["observations"][: length + 1],
        device=parameter.device,
        dtype=parameter.dtype,
    ).unsqueeze(0)
    if episode["observations"].dtype == np.uint8:
        images = images / 255.0
    actions = torch.as_tensor(
        episode["actions"][:length], device=parameter.device, dtype=parameter.dtype
    ).unsqueeze(0)
    terminated = torch.as_tensor(
        episode["terminated"][:length], device=parameter.device
    ).unsqueeze(0)
    truncated = torch.as_tensor(
        episode["truncated"][:length], device=parameter.device
    ).unsqueeze(0)
    indices = torch.linspace(
        0, length - 1, min(num_frames, length), device=parameter.device
    ).long()
    training = model.training
    cuda_devices = (
        [parameter.device.index or 0] if parameter.device.type == "cuda" else []
    )
    mps_rng = torch.mps.get_rng_state() if parameter.device.type == "mps" else None
    try:
        model.eval()
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(0)
            states, _, _ = model.observe_sequence(
                images[:, :-1], actions, terminated, truncated, images[:, 1:]
            )
            selected = DreamerState(
                states.deterministic[:, indices], states.stochastic[:, indices]
            )
            predictions = model.predict(selected)["reconstructions"]
            return make_reconstruction_grid(images[0, indices + 1], predictions[0])
    finally:
        model.train(training)
        if mps_rng is not None:
            torch.mps.set_rng_state(mps_rng)
