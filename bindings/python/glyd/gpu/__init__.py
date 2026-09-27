"""Glyd on the GPU: a bf16 model's weights held compressed in GPU memory,
bit for bit, and multiplied from there (pip install "glyd[gpu]").

    model = glyd.from_pretrained("Qwen/Qwen3-8B")      # packed as it loads, ready for generate()
    model = glyd.compress(model)                        # a model already loaded, in place
    glyd.save_pretrained(model, "qwen3-8b-glyd")        # glyd-v1: loads without packing again
    print(glyd.fit("Qwen/Qwen3-32B", gpu="48GB"))       # bf16 against Glyd on one GPU

fit needs only the standard library; the rest needs PyTorch,
transformers, safetensors and huggingface_hub, imported at first use, and
the kernels' library (libglyd_gpu_cudaN.so; GLYD_GPU_LIB names another).
Under the Business Source License 1.1 (LICENSE-glyd-gpu), as the rest of
Glyd's GPU code.
"""
import importlib
from .fit import Fit, fit  # noqa: F401

_LAZY = {"from_pretrained": "hf", "compress": "model", "save_pretrained": "format"}


def __getattr__(name):
    if name not in _LAZY:
        raise AttributeError(f"module 'glyd.gpu' has no attribute {name!r}")
    try:
        return getattr(importlib.import_module("." + _LAZY[name], __name__), name)
    except ImportError as e:
        raise ImportError(f'glyd.{name} needs PyTorch and transformers: pip install "glyd[gpu]" ({e})') from e
