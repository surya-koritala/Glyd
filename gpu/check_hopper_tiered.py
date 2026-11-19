"""The tiered layout's route WG on Hopper (tiered_tma_kernel: 1-64 tokens, K a multiple of 64) on this GPU:

  * the same bits as the 12-bit layout's TMA kernel on the same matrix (glyd_gpu_mma12_gemm_x3: two warpgroups, a block an
    SM, the same split of the work and the same products in the same order; equal outputs say every weight was decoded
    exact), 1-64 tokens, with and without a bias, for the shapes of two warpgroups (GLYD_GPU_HT_V 0 and 4);
  * against fp32 (within 1e-2 of the largest output) where the matrix is tame;
  * the same bits with the flag set on every 2nd to 64th step and on a random share (those steps by the general decode),
    and with the blocks' bytes read from global memory (GLYD_GPU_HT_BB=0, dev builds);
  * the same bits every run; each of the kernel's other shapes (GLYD_GPU_HT_V 1, 2, 3, 5) against fp32 and run twice;
  * the routes, the new route asked for (GLYD_GPU_HT_MAX=64, dev builds; off by default until measured): the tiered layout's
    WG from 1 to 64 tokens where K is a multiple of 64, past 64 decoded.

    python check_hopper_tiered.py [--quick]      (the library as glyd_gpu_dev.py finds it: $GLYD_GPU_LIB, else the one next to it)"""
import copy
import os
import sys
import time

import numpy as np
import torch

os.environ.setdefault("GLYD_GPU_HT_MAX", "64")  # (the route WG for the tiered layout's steps: read at each route asked; before the first)
import glyd_gpu_dev as g  # noqa: E402
from glyd_gpu import _lib  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "glyd-gpu-py", "tests"))
from test_gpu import mma_matrices  # noqa: E402  (the matrices of the packers' tests: a heavy tail, steps at the short form's limits, NaNs, infinities)

dev, bf = "cuda", torch.bfloat16
quick = "--quick" in sys.argv
MS = (1, 2, 3, 8, 9, 15, 16, 17, 24, 31, 32, 33, 48, 63, 64) if not quick else (1, 16, 17, 33, 64)
SAME = ("0", "4")  # the shapes of two warpgroups a block, one block an SM: the 12-bit TMA kernel's split
OTHER = ("1", "2", "3", "5")
checks = {"same as 12-bit": 0, "fp32": 0, "flags": 0, "global": 0, "rerun": 0, "shapes": 0, "routes": 0}


def bits(t):
    return t.contiguous().view(torch.int16)


def with_flags(p, every):
    """A copy of pack p whose steps' flag (bit 31 of block_base) is set on every `every`-th step; a float: on that share of them (check_step.py's)."""
    q = copy.copy(p)
    bb = p.block_base.clone()
    sel = torch.zeros(bb.numel(), dtype=torch.bool, device=bb.device)
    if isinstance(every, float):
        r = torch.Generator(device="cpu").manual_seed(5)
        sel[:-1] = (torch.rand(bb.numel() - 1, generator=r) < every).to(bb.device)
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


def env(**kv):
    class E:
        def __enter__(self):
            self.old = {k: os.environ.get(k) for k in kv}
            os.environ.update({k: str(v) for k, v in kv.items()})

        def __exit__(self, *a):
            for k, v in self.old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
            _lib._need.cache_clear()

    _lib._need.cache_clear()  # (a workspace's size depends on the shape)
    return E()


def wg(p, x, b=None):
    return g.mma_linear(p, x, b, way=g.X3)


def check(name, w, tame):
    O, K = w.shape
    p, q = g.pack_mma(w), g.pack_mma12(w)
    wf = w.double()
    x = torch.randn(64, K, dtype=torch.float32, device=dev).to(bf)
    bias = (torch.randn(O, dtype=torch.float32, device=dev) * 0.1).to(bf)
    every = (2, 3, 5, 7, 31, 32, 33, 64, 0.01, 0.2, 0.5) if not quick else (2, 33, 0.2)
    flagged = [(e, with_flags(p, e)) for e in every]
    for M in MS:
        for b in (None, bias):
            ref = x[:M].double() @ wf.t() + (b.double() if b is not None else 0)
            y12 = g.mma_gemm_x3(q, x[:M], b)
            for v in SAME:
                with env(GLYD_GPU_HT_V=v):
                    y = wg(p, x[:M], b)
                    assert torch.equal(bits(y), bits(y12)), (name, M, v, "the 12-bit TMA kernel's bits")
                    checks["same as 12-bit"] += 1
                    assert torch.equal(bits(y), bits(wg(p, x[:M], b))), (name, M, v, "the same bits every run")
                    checks["rerun"] += 1
            y = wg(p, x[:M], b)
            if tame:
                near(y, ref, (name, M, "fp32"))
                checks["fp32"] += 1
            for e, f in flagged:
                assert torch.equal(bits(y), bits(wg(f, x[:M], b))), (name, M, f"flag on every {e}th step: the general decode's product is the short form's")
                checks["flags"] += 1
            with env(GLYD_GPU_HT_BB=0):
                assert torch.equal(bits(y), bits(wg(p, x[:M], b))), (name, M, "the blocks' bytes from global memory")
                checks["global"] += 1
            if tame and b is None and M in (1, 16, 17, 33, 64):
                for v in OTHER:
                    with env(GLYD_GPU_HT_V=v):
                        z = wg(p, x[:M])
                        near(z, ref, (name, M, v, "fp32"))
                        assert torch.equal(bits(z), bits(wg(p, x[:M]))), (name, M, v, "the same bits every run")
                        checks["shapes"] += 1
    code = _lib.gpu()
    for M in (1, 16, 17, 64, 65):
        want = g.X3 if M <= 64 else g.DECODE
        assert g.way(p, code, M)[0] == want, (name, M, g.way(p, code, M), "the route")
        checks["routes"] += 1
    return p


t0 = time.time()
assert g.lib() is not None, "no library found: run build_lib.sh, or give its path in GLYD_GPU_LIB"
assert torch.cuda.get_device_capability() == (9, 0), "a Hopper GPU's check"
print("library", g.library(), flush=True)

for name, w in mma_matrices():  # the packers' tests' matrices, K a multiple of 64
    o, k = w.shape
    if k % 64:
        continue
    u = torch.from_numpy(w.astype(np.int16)).to(dev).view(bf)
    wf = u.double()
    tame = bool(torch.isfinite(wf).all()) and wf.abs().max().item() < 1e12
    p = check(name, u, tame)
    print(f"{name:44s} {o}x{k}: {p.flagged()} of {o * k // 1024} steps flagged, ok", flush=True)

torch.manual_seed(1)
shapes = [(64, 64, 0.0), (128, 4096, 0.0), (192, 4096, 0.0), (320, 1024, 0.001), (1024, 1024, 0.05), (2048, 4096, 0.1), (4096, 4096, 0.0), (4096, 4096, 0.003), (12288, 512, 0.001), (64, 64 * 129, 0.0)]
if quick:
    shapes = shapes[:4]
for o, k, wild in shapes:
    w = weights(o * k, wild).view(o, k)
    p = check(f"random {o}x{k} wild {wild}", w, True)
    print(f"random {o}x{k}, wild {wild}: {p.flagged()} of {o * k // 1024} steps flagged, ok", flush=True)

print(f"all ok: {sum(checks.values())} checks {checks}, {time.time() - t0:.0f} s")
