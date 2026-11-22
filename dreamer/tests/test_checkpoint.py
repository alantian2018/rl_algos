import pytest
import torch

from common.config import GlobalConfig
from common.utils import Logger


def test_checkpoint_saving_preserves_callers_mapping_and_loads_weights(tmp_path):
    config = GlobalConfig(exp_name="checkpoint", use_wandb=False, device="cpu")
    logger = Logger(config, save_dir=str(tmp_path))
    model = torch.nn.Linear(2, 1)
    original = model.weight.detach().clone()
    networks = {"model": model}
    logger.save_checkpoint(networks, step=10)
    assert list(networks) == ["model"]
    checkpoint = tmp_path / "checkpoint_10.pt"
    with torch.no_grad():
        model.weight.zero_()
    logger.load_checkpoint(str(checkpoint), networks)
    torch.testing.assert_close(model.weight, original)
    assert torch.load(checkpoint, weights_only=False)["step"] == 10


def test_failed_checkpoint_write_keeps_existing_checkpoint(tmp_path, monkeypatch):
    config = GlobalConfig(exp_name="checkpoint", use_wandb=False, device="cpu")
    logger = Logger(config, save_dir=str(tmp_path))
    networks = {"model": torch.nn.Linear(2, 1)}
    logger.save_checkpoint(networks, step=10)
    checkpoint = tmp_path / "checkpoint_10.pt"
    previous = checkpoint.read_bytes()

    def failed_save(data, stream):
        stream.write(b"partial")
        raise RuntimeError("write interrupted")

    monkeypatch.setattr("common.utils.torch.save", failed_save)
    with pytest.raises(RuntimeError, match="write interrupted"):
        logger.save_checkpoint(networks, step=10)
    assert checkpoint.read_bytes() == previous
    assert not list(tmp_path.glob("*.tmp"))
