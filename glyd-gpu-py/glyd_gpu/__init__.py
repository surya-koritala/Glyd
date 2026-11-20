"""Glyd on the GPU: a bf16 model's weights held compressed in GPU memory,
bit for bit, and multiplied from there (pip install "glyd[gpu]").

    model = glyd_gpu.from_pretrained("Qwen/Qwen3.5-9B")  # packed as it loads, ready for generate()
    model = glyd_gpu.compress(model)                      # a model already loaded, in place
    glyd_gpu.save_pretrained(model, "qwen3.5-9b-glyd")    # glyd-v4: loads without packing again (tables=True: glyd-v5, smaller)
    print(glyd_gpu.fit("Qwen/Qwen3.5-27B", gpu="48GB"))   # bf16 against Glyd on one GPU

glyd.from_pretrained, glyd.save_pretrained and glyd.fit are these, through the glyd package (pip install "glyd[gpu]").

fit needs only the standard library; the rest needs PyTorch,
transformers, safetensors and huggingface_hub, imported at first use, and
the kernels' library (libglyd_gpu_cudaN.so; GLYD_GPU_LIB names another).
Under the Apache License 2.0 (LICENSE), as the rest of
Glyd.
"""
import importlib

__version__ = "0.33.0"  # the glyd package's: glyd[gpu] pins the same one
from .fit import Fit, fit  # noqa: E402,F401

_LAZY = {"from_pretrained": "hf", "compress": "model", "save_pretrained": "format"}


def __getattr__(name):
    if name not in _LAZY:
        raise AttributeError(f"module 'glyd_gpu' has no attribute {name!r}")
    try:
        return getattr(importlib.import_module("." + _LAZY[name], __name__), name)
    except ImportError as e:
        raise ImportError(f'glyd_gpu.{name} needs PyTorch and transformers: pip install "glyd[gpu]" ({e})') from e
