import numpy as np
import torch


def prepare_observation(obs, config, dtype):
    image = np.asarray(obs)
    height, width = config.obs_dim
    channels = config.obs_channels
    if image.shape == (height, width) and channels == 1:
        image = image[None]
    elif image.shape == (height, width, channels):
        image = image.transpose(2, 0, 1)
    if image.shape != (channels, height, width):
        raise ValueError("Observation shape does not match configured image dimensions")
    image = np.array(image, copy=True)
    tensor = torch.as_tensor(image, device=config.device, dtype=dtype).unsqueeze(0)
    if image.dtype == np.uint8:
        tensor = tensor / 255.0
    return image, tensor


@torch.no_grad()
def update_return_scale(returns, quantiles, rate):
    current = torch.quantile(
        returns.float(), returns.new_tensor([0.05, 0.95], dtype=torch.float32)
    ).to(returns)
    if quantiles is None:
        quantiles = current
    else:
        quantiles.lerp_(current, rate)
    scale = (quantiles[1] - quantiles[0]).clamp_min(1)
    return scale, quantiles
