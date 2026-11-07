"""int8 quantisation on a small MLP, with both backends (needs torch)."""
import pytest

torch = pytest.importorskip("torch")

from medvqa.compress import model_size_mb, quantize_int8  # noqa: E402


def mlp():
    torch.manual_seed(0)
    return torch.nn.Sequential(torch.nn.Linear(256, 512), torch.nn.ReLU(), torch.nn.Linear(512, 64))


@pytest.mark.parametrize("backend", ["torch.ao", "torchao"])
def test_int8_is_smaller_and_close(backend):
    if backend == "torchao":
        pytest.importorskip("torchao")
    ref = mlp().eval()
    x = torch.randn(8, 256)
    with torch.no_grad():
        y_ref = ref(x)
    q = quantize_int8(mlp(), backend=backend)
    with torch.no_grad():
        y_q = q(x)
    assert model_size_mb(q) < 0.5 * model_size_mb(ref)
    rel_err = ((y_q - y_ref).norm() / y_ref.norm()).item()
    assert rel_err < 0.05, rel_err


def test_unknown_backend():
    with pytest.raises(ValueError):
        quantize_int8(mlp(), backend="nope")
