"""Glyd's GPU half is the glyd-gpu package (pip install "glyd[gpu]", or "glyd[vllm]" for the vLLM plugin and glyd run): glyd.gpu is its
public API, re-exported: from_pretrained, save_pretrained, compress, fit and Fit."""
import importlib

__all__ = ["Fit", "compress", "fit", "from_pretrained", "save_pretrained"]
MISSING = 'the GPU half of glyd is the glyd-gpu package, which is not installed here: pip install "glyd[gpu]" (Linux, an NVIDIA GPU)'


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(f"module 'glyd.gpu' has no attribute {name!r}")
    try:
        gpu = importlib.import_module("glyd_gpu")
    except ImportError as e:
        raise ImportError(f"glyd.{name}: {MISSING} ({e})") from e
    return getattr(gpu, name)
