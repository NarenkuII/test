import pytest

torch = pytest.importorskip("torch")

from signlang import model


def test_model_forward() -> None:
    net = model.KeypointModel()
    dummy = torch.randn(2, 3, 512, 512)
    out = net(dummy)
    assert out.shape == (2, 64, 128, 128)
