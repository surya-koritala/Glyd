"""The tiered layout's product for 1-16 tokens (the step kernel) on the GPU:

  * against an fp32 product (within 1e-2 of the largest output), with and without a bias, for 1-16 tokens;
  * the same bits every run;
  * the same bits when the flag of every step, of every 2nd to every 64th, or of a share of them chosen at random, is set, which sends those steps through the general
    decode: the short form of the decode is the general one, step by step, on every matrix here (a heavy tail, steps over the short
    form's limits, a list of literals at the window's edge, NaNs and infinities among the weights);
  * the 8-warp kernel (GLYD_GPU_STEP=0) within 1e-2;
  * units shared by blocks (one row block over many blocks) and many row blocks over few steps.

    python check_step.py [--quick]      (the library as glyd_gpu_dev.py finds it: $GLYD_GPU_LIB, else the one next to it)"""
import copy
import os
import sys
import time

import numpy as np
import torch

import glyd_gpu_dev as g

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "glyd-gpu-py", "tests"))
from test_gpu import mma_matrices  # (the matrices of the packers' tests: a heavy tail, steps at the short form's limits, NaNs, infinities)

dev, bf = "cuda", torch.bfloat16
quick = "--quick" in sys.argv
MS = (1, 2, 3, 4, 5, 8, 9, 12, 16)
checks = {"fp32": 0, "flags": 0, "rerun": 0, "nw": 0}


def bits(t):
    return t.contiguous().view(torch.int16)


def with_flags(p, every):
    """A copy of pack p whose steps' flag (bit 31 of block_base) is set on every `every`-th step; a float: on that share of them, chosen by a fixed seed."""
    q = copy.copy(p)
    bb = p.block_base.clone()
    sel = torch.zeros(bb.numel(), dtype=torch.bool, device=bb.device)
    if isinstance(every, float):
        g = torch.Generator(device="cpu").manual_seed(5)
        sel[:-1] = (torch.rand(bb.numel() - 1, generator=g) < every).to(bb.device)
    else:
        sel[: bb.numel() - 1 : every] = True
    bb[sel] = bb[sel] | torch.iinfo(torch.int32).min
    q.block_base = bb
    return q


def weights(n, wild=0.0):
    w = torch.randn(n, device=dev) * 0.02
    m = torch.rand(n, device=dev) < wild
    w[m] = torch.randn(int(m.sum()), device=dev) * torch.exp2(torch.randint(-40, 20, (int(m.sum()),), device=dev).float())
    return w.to(bf)


def near(y, ref, what):
    big = ref.abs().max().item()
    err = (y.double() - ref).abs().max().item()
    assert err <= 1e-2 * big + 1e-6, (what, err, big)


def check(name, p, u, tame, Ms=MS):
    O, K = p.shape
    wf = u.view(bf).double()
    x = torch.randn(16, K, dtype=torch.float32, device=dev).to(bf)
    bias = (torch.randn(O, dtype=torch.float32, device=dev) * 0.1).to(bf)
    every = (1, 2, 3, 5, 7, 31, 32, 33, 64, 0.01, 0.2, 0.5) if not quick else (1, 5, 0.2)
    flagged = [(e, with_flags(p, e)) for e in every]
    for M in Ms:
        for b in (None, bias):
            y = g.mma_gemm(p, x[:M], b)
            if tame:
                near(y, x[:M].double() @ wf.t() + (b.double() if b is not None else 0), (name, M, "bias" if b is not None else "none"))
                checks["fp32"] += 1
            assert torch.equal(bits(y), bits(g.mma_gemm(p, x[:M], b))), (name, M, "the same bits every run")
            checks["rerun"] += 1
            for e, q in flagged:
                assert torch.equal(bits(y), bits(g.mma_gemm(q, x[:M], b))), (name, M, f"flag on every {e}th step: the general decode's product is the short form's")
                checks["flags"] += 1
    # the knobs: blocks of 2 and of 4 warps (1-2 tokens), the 8-warp kernel
    if tame and not quick:
        for M in (1, 2, 8, 13):
            ref = x[:M].double() @ wf.t()
            for env, val in (("GLYD_GPU_STEP", "0"),):
                os.environ[env] = val
                try:
                    y = g.mma_gemm(p, x[:M])
                    near(y, ref, (name, M, env, val))
                    assert torch.equal(bits(y), bits(g.mma_gemm(p, x[:M]))), (name, M, env, val, "the same bits every run")
                finally:
                    del os.environ[env]
                checks["nw"] += 1


t0 = time.time()
assert g.lib() is not None, "no library found: run build_lib.sh, or give its path in GLYD_GPU_LIB"
print("library", g.library(), flush=True)

# the adversarial matrices of the packer's tests (golden.py)
for name, w in mma_matrices():
    o, k = w.shape
    u = torch.from_numpy(w.astype(np.int16)).to(dev)
    p = g.pack_mma(u.view(bf))
    wf = u.view(bf).double()
    tame = bool(torch.isfinite(wf).all()) and wf.abs().max().item() < 1e12
    check(name, p, u, tame)
    print(f"{name:44s} {o}x{k}: {p.flagged()} of {o * k // 1024} steps flagged, ok", flush=True)

# random matrices: a trained matrix's spread, with `wild` of the weights anywhere (steps over the short form's limits where it is high)
torch.manual_seed(1)
shapes = [(64, 16, 0.0), (64, 4096, 0.0), (64, 8192, 0.01), (192, 4096, 0.0), (448, 208, 0.002), (4096, 256, 0.0), (1024, 1024, 0.05), (2048, 4096, 0.0), (4096, 4096, 0.0), (4096, 4096, 0.003), (12288, 512, 0.001),
          (192, 80, 0.0), (320, 112, 0.001), (128, 528, 0.0), (64, 2064, 0.0), (256, 4112, 0.002), (64, 16 * 129 * 4, 0.0)]
if quick:
    shapes = shapes[:5]
for o, k, wild in shapes:
    w = weights(o * k, wild).view(o, k)
    p = g.pack_mma(w)
    check(f"random {o}x{k} wild {wild}", p, w.view(torch.int16), True, Ms=MS if o * k <= 4096 * 1024 else (1, 2, 8, 16))
    print(f"random {o}x{k}, wild {wild}: {p.flagged()} of {o * k // 1024} steps flagged, ok", flush=True)

print(f"all ok: {sum(checks.values())} checks {checks}, {time.time() - t0:.0f} s")
