import pytest
import torch

from dreamer.dreamer.networks import DreamerState


@pytest.mark.parametrize("shape", [(1, 1), (2, 3), (6,)])
def test_flatten_preserves_features_order_and_original_state(shape):
    count = 1
    for dimension in shape:
        count *= dimension
    h = torch.arange(count * 4).reshape(*shape, 4)
    z = torch.arange(count * 2).reshape(*shape, 2)
    state = DreamerState(h, z)
    flat = state.flatten()
    assert flat is not state
    assert flat.deterministic.shape == (count, 4)
    assert flat.stochastic.shape == (count, 2)
    assert state.deterministic.shape == (*shape, 4)
    assert state.stochastic.shape == (*shape, 2)
    torch.testing.assert_close(flat.deterministic[0], torch.arange(4))
    torch.testing.assert_close(flat.deterministic[-1], torch.arange((count - 1) * 4, count * 4))
    torch.testing.assert_close(flat.stochastic[-1], torch.arange((count - 1) * 2, count * 2))


def test_flatten_handles_noncontiguous_states():
    h = torch.arange(24).reshape(2, 3, 4).transpose(0, 1)
    z = torch.arange(12).reshape(2, 3, 2).transpose(0, 1)
    flat = DreamerState(h, z).flatten()
    assert flat.deterministic.shape == (6, 4)
    assert flat.stochastic.shape == (6, 2)
    torch.testing.assert_close(flat.deterministic[1], h[0, 1])
    torch.testing.assert_close(flat.stochastic[1], z[0, 1])


def test_flatten_preserves_gradient_flow_and_dtype():
    h = torch.randn(2, 3, 4, dtype=torch.float64, requires_grad=True)
    z = torch.randn(2, 3, 2, dtype=torch.float64, requires_grad=True)
    flat = DreamerState(h, z).flatten()
    assert flat.deterministic.dtype == flat.stochastic.dtype == torch.float64
    assert flat.deterministic.device == h.device
    (2 * flat.deterministic.sum() + 3 * flat.stochastic.sum()).backward()
    torch.testing.assert_close(h.grad, torch.full_like(h, 2))
    torch.testing.assert_close(z.grad, torch.full_like(z, 3))
