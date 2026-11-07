"""Making the model small and fast enough for a CPU.

Two steps:

1. Merge LoRA into the base weights (W' = W + B·A·alpha/r). The adapter
   disappears and inference costs exactly what the base model costs.
2. Dynamic int8 quantisation of every nn.Linear: weights are stored as 8-bit
   integers with a scale per output channel (about 4x smaller than float32),
   activations are quantised on the fly. On CPUs with int8 instructions
   (AVX-512 VNNI, ARM dot-product) matrix multiplies get faster too. No
   calibration data is needed, which is why it is the first thing to try.

Two implementations:
* "torchao"  - PyTorch's current quantisation library (pip install torchao)
* "torch.ao" - the older built-in API, deprecated in recent PyTorch releases
"auto" uses torchao when it is installed.

Next steps for real edge devices (not implemented here): export to ONNX and
run with ONNX Runtime, or convert to GGUF and run with llama.cpp.
"""
from __future__ import annotations

import io
import warnings

import torch
from torch import nn


def _torchao_available() -> bool:
    try:
        import torchao  # noqa: F401
        return True
    except ImportError:
        return False


def quantize_int8(model: nn.Module, backend: str = "auto") -> nn.Module:
    model = model.to("cpu").float().eval()
    if backend == "auto":
        backend = "torchao" if _torchao_available() else "torch.ao"
    if backend == "torchao":
        from torchao.quantization import Int8DynamicActivationInt8WeightConfig, quantize_

        quantize_(model, Int8DynamicActivationInt8WeightConfig())  # in place
        return model
    if backend == "torch.ao":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from torch.ao.quantization import quantize_dynamic

            return quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
    raise ValueError(f"unknown backend {backend!r} (auto, torchao, torch.ao)")


def model_size_mb(model: nn.Module) -> float:
    """Size of the serialised weights (what you would ship to a device)."""
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.tell() / 1e6
