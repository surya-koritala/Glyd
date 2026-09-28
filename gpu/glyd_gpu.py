"""Glyd weights on the GPU: bf16 tensors held compressed in VRAM and
decoded on the GPU, bit for bit (glyd_gpu.cu has the layout).

    p = pack(w)          # w: a bf16 CUDA tensor
    w2 = unpack(p)       # the same bits
    p.bits_per_weight()

The Python side lives in the glyd package (bindings/python/glyd/gpu:
kernels.py, and _lib.py over the prebuilt library); this module is it for
the scripts here, the package taken from this checkout, with the kernels
from the prebuilt library where one is found, else built here by
PyTorch's JIT (needs nvcc).
"""
import os
import sys

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bindings", "python")))
import torch
from torch.utils.cpp_extension import load
from glyd.gpu import _lib, kernels
from glyd.gpu.kernels import *  # noqa: F401,F403 (pack, unpack, Mma, mma_gemm ... as before)
from glyd.gpu.kernels import _chunks, _hist, _none, _sign_mantissa  # noqa: F401 (the scripts' use)


def _arch_flags():
    """Code for this machine's GPU (Hopper as sm_90a, for its warpgroup
    instructions); GLYD_GPU_ARCH=sm_89,sm_90a builds for several (sm_90
    there as sm_90a too: without it the TMA kernel is a trap)."""
    archs = os.environ.get("GLYD_GPU_ARCH")
    if not archs:
        major, minor = torch.cuda.get_device_capability()
        archs = f"sm_{major}{minor}" + ("a" if major == 9 else "")
    flags = []
    for a in archs.split(","):
        a = "sm_90a" if a == "sm_90" else a
        flags += ["-gencode", f"arch=compute_{a[3:]},code={a}"]
    return flags


def _jit():
    """glyd_gpu.cu built here for this GPU by PyTorch's extension builder
    (needs nvcc): the pybind module."""
    return load(
        name="glyd_gpu",
        sources=[os.path.join(os.path.dirname(os.path.abspath(__file__)), "glyd_gpu.cu")],
        extra_cuda_cflags=["-O3"] + _arch_flags() + (["-Xptxas", "-v"] if os.environ.get("GLYD_GPU_PTXAS") else []),
        verbose=bool(os.environ.get("GLYD_GPU_PTXAS")),
    )


def _prebuilt():
    """The prebuilt library (build_lib.sh): $GLYD_GPU_LIB, else
    libglyd_gpu_cudaN.so next to this file for PyTorch's CUDA N; None if
    neither."""
    if os.environ.get("GLYD_GPU_LIB"):
        return os.environ["GLYD_GPU_LIB"]
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"libglyd_gpu_cuda{(torch.version.cuda or '0').split('.')[0]}.so")
    return path if os.path.exists(path) else None


# The kernels: from the prebuilt library through _lib.py (its C API, the
# pybind module's functions) where it is found, else built here; the
# package's functions call them through kernels._ext.
if _prebuilt():
    _lib.load(_prebuilt())
    _ext = _lib
else:
    _ext = _jit()
kernels._ext = _ext


if __name__ == "__main__":
    # The many-token products against the fp32 product: mma_gemm_mid (Ampere on), mma_gemm_wg (Hopper), and a
    # prompt's mma_gemm_big (Ampere and Ada, both layouts: on GeForce Ada its consumers of half a row block, blocks of
    # 128 tokens and of 256, to 1100). Odd row blocks, units shared by blocks, exceptions few and many (past a stage's
    # copy: read from global memory), 1-600 tokens, bias; the same every run.
    import torch.nn.functional as F
    assert torch.cuda.get_device_capability()[0] >= 8, "Ampere or later"
    hopper = torch.cuda.get_device_capability() == (9, 0)
    products = [("mma_gemm_mid", mma_gemm_mid)] + ([("mma_gemm_wg", mma_gemm_wg)] if hopper else [("mma_gemm_big", mma_gemm_big)])
    torch.manual_seed(0)
    for O, K, wild in [(64, 64, 0), (192, 128, 0), (128, 4096, 0), (1024, 2048, 0), (5120, 1024, 0.001), (192, 4096, 0.1), (3072, 5120, 0.02), (17408, 1024, 0.01)]:
        w = torch.randn(O, K, device="cuda") * 0.02
        m = torch.rand(O, K, device="cuda") < wild  # this share of weights at exponents far from the commonest 15
        w[m] = torch.randn(int(m.sum()), device="cuda") * torch.exp2(torch.randint(-40, 20, (int(m.sum()),), device="cuda").float())
        w = w.to(torch.bfloat16)
        q = pack_mma12(w)
        assert torch.equal(mma_unpack(q).view(torch.int16), w.view(torch.int16))
        bias = torch.randn(O, device="cuda").to(torch.bfloat16)
        for name, prod in products:
            big = name == "mma_gemm_big"
            for p in (q, pack_mma(w)) if big else (q,):
                for M in [1, 7, 16, 17, 32, 33, 64, 65, 100, 128, 129, 256, 257, 600] + ([400, 1100] if big else []):
                    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
                    for b in (None, bias):
                        ref = F.linear(x.float(), w.float(), None if b is None else b.float())
                        y = prod(p, x, b)
                        err = ((y.float() - ref).abs().max() / ref.abs().max()).item()
                        assert err < 1e-2 and torch.equal(y, prod(p, x, b)), (name, type(p).__name__, O, K, wild, M, err)
            print(f"{name} {O}x{K}, {int(q.exc_base[-1])} exceptions: 1-{1100 if big else 600} tokens within 1e-2, the same every run")
