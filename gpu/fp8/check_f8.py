"""The lossless FP8 kernels of glyd_gpu.cu (f8_pack, f8_unpack, f8_unpack_experts) against fp8ref.py, bit for bit, on a GPU.

    GLYD_GPU_LIB=/path/libglyd_gpu_cuda13.so PYTHONPATH=glyd-gpu-py python gpu/fp8/check_f8.py [--samples DIR] [--no-bw]
    python gpu/fp8/check_f8.py --emu [--quick]          the same cases on the CPU, thread for thread (emu.py: no GPU, no torch)

Every case of glyd-gpu-py/tests/test_fp8ref.py is run with the GPU's pack and unpack in the reference's place: the GPU's pack must give
the bytes of fp8ref.pack (core, heads, var, all of them), and the GPU's unpack of those bytes the matrix, bit for bit: all 256 byte
patterns, NaNs and both zeros, the shapes that are not multiples of a patch, plain patches beside coded ones, the real samples
(gpu/fp8/fetch_samples.py DIR; --samples DIR) and the rest of that file's cases. Then what the reference has no case for: unpack into
rows apart and pack from rows apart (a row's distance of 16 bytes and not, K and O that are not multiples of 16 or of a patch; the bytes
between rows must not change), unpack of a list of a stack's matrices (ids repeated, out of range, in any order), CUDA graphs that hold
the unpack and the list's unpack and replay them with other ids (GPU only), and a line of the unpack's bandwidth and the pack's time for
a 7168 x 18432 matrix (GPU only). f8_linear (the product from the pack) is checked against an exact float64 product: within bf16's rounding of
the output and fp32 sums (lin_close), not bit for bit (its sums are its own). Exit status 1 on any failure.
    compute-sanitizer --tool memcheck|racecheck python gpu/fp8/check_f8.py --no-bw --only case_lin_quick     (a few products, for the sanitizer)"""
import argparse
import collections
import importlib.util
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(HERE, "..", "..", "glyd-gpu-py")


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


fp8ref = load("fp8ref", os.path.join(PKG, "glyd_gpu", "fp8ref.py"))
T = load("test_fp8ref", os.path.join(PKG, "tests", "test_fp8ref.py"))
weights = T.weights


class Backend:
    """pack(w) -> (core, heads, var) numpy arrays; unpack(core, heads, var, shape, ld) -> the matrix; experts(...) -> [n, O, K]."""

    name = ""


class Gpu(Backend):
    def __init__(self):
        import torch

        sys.path.insert(0, PKG)
        from glyd_gpu import kernels

        self.torch, self.g = torch, kernels
        assert torch.cuda.is_available(), "no CUDA GPU"
        kernels.lib()  # (the library loaded now: GLYD_GPU_LIB, else the one beside the package)
        from glyd_gpu import _lib

        self.lib_need = _lib._need
        self.name = f"GPU {torch.cuda.get_device_name()}"

    def pack(self, w, ld=None):
        torch = self.torch
        O, K = w.shape
        x = torch.from_numpy(np.ascontiguousarray(w)).cuda()
        if ld and ld != K:  # rows ld bytes apart: a view of a wider matrix whose other bytes are not the matrix's
            big = torch.full((O, ld), 0x5A, dtype=torch.uint8, device="cuda")
            big[:, :K] = x
            x = big[:, :K]
        p = self.g.f8_pack(x)
        torch.cuda.synchronize()
        return p.core.cpu().numpy(), p.heads.cpu().numpy().view(np.uint32), p.var.cpu().numpy().view(np.uint32)

    def packed(self, core, heads, var, shape):
        torch = self.torch
        t = lambda a, d: torch.from_numpy(np.ascontiguousarray(a).view(d)).cuda()
        return self.g.F8Packed(shape, t(core, np.uint8), t(heads, np.int32), t(var, np.int32))

    def unpack(self, core, heads, var, shape, ld=None):
        torch = self.torch
        O, K = shape
        ld = ld or K
        p = self.packed(core, heads, var, shape)
        out = torch.full((O, ld), 0xA5, dtype=torch.uint8, device="cuda")
        self.g.f8_unpack_into(p, out[:, :K])
        torch.cuda.synchronize()
        m = out.cpu().numpy()
        assert (m[:, K:] == 0xA5).all(), "the unpack wrote between the rows"
        return m[:, :K].copy()

    def experts(self, core, heads, var, E, O, K, ids, ld=None):
        torch = self.torch
        ld = ld or K
        p = self.packed(core, heads, var, (E * O, K))
        i = torch.tensor(ids, dtype=torch.int32, device="cuda")
        out = torch.full((len(ids), O, ld), 0xA5, dtype=torch.uint8, device="cuda")
        self.g.f8_unpack_experts(p, E, i, out[:, :, :K] if ld != K else out)
        torch.cuda.synchronize()
        return out.cpu().numpy()[:, :, :K].copy()


    def linear(self, core, heads, var, N, K, kind, scale, x, plan=None):
        """f8_linear on the pack's bytes: x uint16 (bf16 bits) [M, K] -> y uint16 [M, N] (the library's plan; plan: (ntl, ctas, ahead, dense, a8): a grid of ctas
        CTAs (0: the library's), ahead, dense and a8 (None: the library's), a development library's GLYD_F8_LIN_GRID, GLYD_F8_LIN_AHEAD, GLYD_F8_LIN_DENSE and
        GLYD_F8_LIN_A8; ntl the library's)."""
        torch = self.torch
        p = self.packed(core, heads, var, (N, K))
        sc = torch.from_numpy(np.ascontiguousarray(scale)).cuda()
        xt = torch.from_numpy(np.ascontiguousarray(x).view(np.int16)).cuda().view(torch.bfloat16)
        env = {}
        if plan and plan[1]:
            env["GLYD_F8_LIN_GRID"] = str(plan[1])
        if plan and len(plan) > 2 and plan[2] is not None:
            env["GLYD_F8_LIN_AHEAD"] = str(plan[2])
        if plan and len(plan) > 3 and plan[3] is not None:
            env["GLYD_F8_LIN_DENSE"] = str(plan[3])
        if plan and len(plan) > 4 and plan[4] is not None:
            env["GLYD_F8_LIN_A8"] = str(plan[4])
        os.environ.update(env)
        self.lib_need.cache_clear()  # (the workspace's size is the plan's)
        try:
            y = self.g.f8_linear(p, KINDS[kind], sc.reshape(()) if kind == 1 else sc, xt)
            torch.cuda.synchronize()
        finally:
            for k in env:
                del os.environ[k]
            self.lib_need.cache_clear()
        return y.view(torch.int16).cpu().numpy().view(np.uint16)


class Cpu(Backend):
    def __init__(self):
        from emu import Emu

        self.e = Emu()
        self.name = "the CPU emulation (emu.py)"

    def pack(self, w, ld=None):
        if ld and ld != w.shape[1]:
            big = np.full((w.shape[0], ld), 0x5A, np.uint8)
            big[:, : w.shape[1]] = w
            w = big[:, : w.shape[1]]
        return self.e.pack(w)

    def unpack(self, core, heads, var, shape, ld=None):
        return self.e.unpack(core, heads, var, shape, ld)

    def experts(self, core, heads, var, E, O, K, ids, ld=None):
        return self.e.experts(core, heads, var, E, O, K, ids, ld)

    def linear(self, core, heads, var, N, K, kind, scale, x, plan=None):
        pl = list(plan or lin_plan(N, K, x.shape[0], 4))
        if len(pl) > 2 and pl[2] is None:
            pl[2] = 0
        if len(pl) > 3 and pl[3] is None:
            pl[3] = 0
        if len(pl) > 4 and pl[4] is None:
            pl[4] = 0
        return self.e.linear(core, heads, var, N, K, kind, scale, x, *pl)


def same(name, got, ref):
    """got (core, heads, var) are the reference's, array by array: else the first patch that differs."""
    for a, b, what in zip(got, (ref.core, ref.heads, ref.var), ("core", "heads", "var")):
        if a.shape != b.shape or not np.array_equal(a, b):
            where = ""
            if a.shape == b.shape:
                bad = np.nonzero((a != b).reshape(len(a), -1).any(1))[0] if what != "var" else np.nonzero(a != b)[0]
                where = f", first {what} index {bad[:5].tolist()} of {len(bad)}"
            raise AssertionError(f"{name}: the GPU's {what} is not the reference's (shapes {a.shape} {b.shape}){where}")


def install(b):
    """test_fp8ref's reference module with its pack and unpack replaced by the backend's: pack gives the reference's bytes (checked) and unpack reads them."""
    ref_pack, f8 = T.f8.pack, T.f8

    def pack(w):
        ref = ref_pack(w)  # (the reference first: it refuses what is not a matrix of bytes)
        same(f"pack {w.shape}", b.pack(np.asarray(w)), ref)
        return ref

    def unpack(p):
        return b.unpack(p.core, p.heads, p.var, p.shape)

    f8.pack, f8.unpack = pack, unpack
    return ref_pack


# ---- the fused product (f8_linear) against an exact product of the same dequantized weights ----
KINDS = ("block", "tensor", "mx", "ds32", "channel")
_B = np.arange(256)
E4M3 = np.where((_B & 0x78) == 0, (_B & 7) / 8 * 2.0**-6, (1 + (_B & 7) / 8) * 2.0 ** (((_B >> 3) & 15) - 7.0)) * np.where(_B & 0x80, -1, 1)  # (0x7F, 0xFF: NaN, not used here)


def lin_plan(N, K, M, ctas):
    """glyd_gpu.cu's plan for a GPU that holds ctas CTAs at once: (8-token tiles, CTAs: at most one a row group's patch)."""
    n = 1 if M <= 8 else 2 if M <= 16 else 4 if M <= 32 else 8
    return n, min(ctas, lin_work(N, K))


def lin_work(N, K):
    """The product's work: the row groups' (128 rows) patches."""
    return -(-(-(-N // 64) * 4) // 8) * -(-K // 128)


def bf16_bits(a):
    u = np.asarray(a, np.float32).view(np.uint32).astype(np.uint64)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def bf16_value(b):
    return (np.asarray(b).astype(np.uint32) << 16).view(np.float32).astype(np.float64)


def dequant(w8, kind, sc):
    """W as the checkpoint means it: e4m3 values times their scales (float64)."""
    N, K = w8.shape
    n, k = np.arange(N)[:, None], np.arange(K)[None, :]
    s = (sc[n // 128, k // 128] if kind == 0 else sc.reshape(-1)[0] if kind == 1 else 2.0 ** (sc[n, k // 32].astype(np.float64) - 127) if kind == 2
         else 2.0 ** (sc[n // 32, k // 32].astype(np.float64) - 127) if kind == 3 else sc.reshape(-1)[n].astype(np.float64))
    return E4M3[w8] * s


def lin_scales(kind, N, K, rng):
    if kind == 0:
        return (rng.random((-(-N // 128), -(-K // 128))) * 2e-3 + 1e-3).astype(np.float32)
    if kind == 1:
        return np.array([2.5e-3], np.float32)
    if kind == 4:
        return (rng.random(N) * 2e-3 + 1e-3).astype(np.float32)
    rows = N if kind == 2 else -(-N // 32)
    return rng.integers(110, 130, (rows, -(-K // 32))).astype(np.uint8)


def lin_weights(N, K, rng, repo=None):
    """e4m3 bytes (no NaN): a real sample's 64 x 128 patches tiled (repo, where the samples are), else a spread like a checkpoint's."""
    smp, m = os.environ.get("GLYD_FP8_SAMPLES"), None
    if repo and smp and os.path.exists(os.path.join(smp, "manifest.json")):
        import json

        m = ([e for e in json.load(open(os.path.join(smp, "manifest.json"))) if e["repo"] == repo] or [None])[0]  # (a repo not fetched: the spread)
    if m is not None:
        a = np.load(os.path.join(smp, m["file"]))
        r0, c0 = rng.integers(0, max(1, a.shape[0] - N + 1)), rng.integers(0, max(1, a.shape[1] - K + 1))
        w = np.resize(a[r0 : r0 + N, c0 : c0 + K], (N, K)) if a.shape[0] >= N and a.shape[1] >= K else np.resize(a, (N, K))
    else:
        ex = np.clip(np.rint(11 - rng.exponential(2.0, (N, K))), 0, 15).astype(np.int64)
        w = ((rng.integers(0, 2, (N, K)) << 7) | (ex << 3) | rng.integers(0, 8, (N, K))).astype(np.uint8)
    w = w.copy()
    w[(w & 0x7F) == 0x7F] -= 1
    return w


def lin_dense(N, K, rng):
    """Patches still coded but with many values past level 2 and many literals (a lane's unit with more than 8 that reach level 3, ranks 13-15 often)."""
    p = np.array([22, 18, 15, 9, 8, 8, 3.2, 3, 2.8, 2.6, 2.4, 2.2, 2, 1, 0.6, 0.4])
    ex = rng.permutation(16)[rng.choice(16, (N, K), p=p / p.sum())]
    w = ((rng.integers(0, 2, (N, K)) << 7) | (ex << 3) | rng.integers(0, 8, (N, K))).astype(np.uint8)
    w[(w & 0x7F) == 0x7F] -= 1
    return w


def lin_crafted(N, K, rng, every=3):
    """Coded patches (N, K multiples of a patch) built for the dense pass's limits: each row block's stream has 768 values past level 1 (the most the
    pass takes), every `every`-th of them past level 2 (3: 8 of a lane's 24, the most a lane takes, so the pass runs, and its variable part, some 70
    literals in, is over VB words: read where it is in var; 2: 12 of 24, so f8_granule decodes it)."""
    st = np.zeros(2048, np.int64)  # a row block's stream: 0 level 1, 1 level 2, 2 past it
    sp = np.flatnonzero(np.isin(np.arange(2048) % 8, (0, 3, 5)))
    st[sp] = 1
    st[sp[every - 1 :: every]] = 2
    st = np.tile(st, 4)

    def counts(total, n):  # n counts adding to total, each above the next
        c = (total - n * (n - 1) // 2) // n + np.arange(n)[::-1]
        c[0] += total - c.sum()
        return c

    out = np.zeros((N, K), np.uint8)
    for r0 in range(0, N, 64):
        for c0 in range(0, K, 128):
            rank, at = np.zeros(8192, np.int64), 0
            for lvl, n in ((0, 3), (1, 3), (2, 10)):  # ranks 0-2, 3-5, 6-15 (13-15: the literals)
                sel = st == lvl
                rank[sel] = rng.permutation(np.repeat(np.arange(at, at + n), counts(int(sel.sum()), n)))
                at += n
            v = (rng.integers(0, 2, 8192) << 7) | (rng.permutation(16)[rank] << 3) | rng.integers(0, 8, 8192)
            patch = np.empty(8192, np.uint8)
            patch[fp8ref.PERM] = v
            out[r0 : r0 + 64, c0 : c0 + 128] = patch.reshape(64, 128)
    out[(out & 0x7F) == 0x7F] -= 1  # (no NaN: the mantissa, not the exponent, changes)
    return out


VB, EB, ES = 80, 768, 24  # glyd_gpu.cu's VB, EB and ES


def lin_branches(p, N, K):
    """The row blocks of a pack (those holding a row of the matrix) by the path f8_linear's decode takes, as glyd_gpu.cu's f8_geo, f8_granule and
    f8_granule_dense decide it: var (a coded row block whose variable part is over VB words: read where it is in var), spills (over EB values past
    level 1: the dense pass leaves it to f8_granule), marks (a lane of the dense pass with over 8 of its ES values past level 2: the same), and of
    those the dense pass decodes, dlits (with literals) and dvar (its variable part read where it is in var)."""
    c, KT = collections.Counter(), -(-K // 128)
    for pi, h in enumerate(p.heads.astype(np.int64)):
        for mb in range(4):
            if (pi // KT) * 64 + 16 * mb >= N or h[3] >> 31:
                continue
            n1, n2, n3 = h[3] & 0x3FFF, (h[3] >> 14) & 0x3FFF, h[2] >> 20
            x, y, z, w = h[4:8]
            o2, o3, o4, e2, e3, e4 = ((0, 0, 0, x & 0xFFFF, y >> 16, w & 0x3FF) if mb == 0 else
                                      (x & 0xFFFF, y >> 16, w & 0x3FF, x >> 16, z & 0xFFFF, (w >> 10) & 0x3FF) if mb == 1 else
                                      (x >> 16, z & 0xFFFF, (w >> 10) & 0x3FF, y & 0xFFFF, z >> 16, w >> 20) if mb == 2 else
                                      (y & 0xFFFF, z >> 16, w >> 20, n1, n2, n3))
            a2, a3, a4 = o2 >> 4, (3 * o3) >> 5, o4 >> 3
            v3 = ((e2 + 15) >> 4) - a2 + 2
            v4 = v3 + ((3 * e3 + 31) >> 5) - a3 + 2
            big, n1b = int(v4 + ((e4 + 7) >> 3) - a4 > VB), int(e2 - o2)
            c["var"] += big
            if n1b > EB:
                c["spills"] += 1
                continue
            j = o2 + np.arange(n1b)  # the row block's level-2 digits (the patch's run at var[h[0]]), ES a lane of the pass
            d = (p.var[h[0] + (j >> 4)].astype(np.int64) >> (2 * (j & 15))) & 3
            if n1b and np.add.reduceat((d == 3).astype(np.int64), np.arange(0, n1b, ES)).max() > 8:
                c["marks"] += 1
                continue
            c["dlits"] += int(e4 != o4)
            c["dvar"] += big
    return c


def e4m3_rne(v):
    """float32 values as e4m3 bytes: nearest of the 127 magnitudes, ties to the even byte, past 448 to 448 (cvt.rn.satfinite's)."""
    mag = np.abs(E4M3[:127])
    a = np.abs(v.astype(np.float64))
    hi = np.clip(np.searchsorted(mag, a), 1, 126)
    lo = hi - 1
    dlo, dhi = a - mag[lo], mag[hi] - a
    b = np.where(dlo < dhi, lo, np.where(dhi < dlo, hi, np.where(lo % 2 == 0, lo, hi)))
    b = np.where(a >= 448, 126, b)
    return (b | np.where(np.signbit(v), 0x80, 0)).astype(np.uint8)


def x_a8(x):
    """X (bf16 bits [M, K]) as the W8A8 product quantizes it: a token's 128 columns times 448 over their largest magnitude (fp32), as e4m3, and the scale
    back (the largest over 448, fp32): the values the product multiplies, float64."""
    M, K = x.shape
    KT = -(-K // 128)
    f = np.zeros((M, KT * 128), np.float32)
    f[:, :K] = bf16_value(x).astype(np.float32)
    f = f.reshape(M, KT, 128)
    mx = np.abs(f).max(-1, keepdims=True)
    inv = np.where(mx > 0, np.float32(448) / np.where(mx > 0, mx, 1), np.float32(0)).astype(np.float32)
    q = E4M3[e4m3_rne((f * inv).astype(np.float32))]
    return (q * (mx / np.float32(448)).astype(np.float32)).reshape(M, KT * 128)[:, :K]


def lin_close(name, y, w8, kind, sc, x, a8=False):
    """y (bf16 bits) is the exact product to within bf16's rounding of y and fp32 sums: |y - ref| <= 2^-8 |ref| + 2^-14 (|x| |W|^T); a8: of X as the W8A8
    product quantizes it."""
    W, X = dequant(w8, kind, sc), (x_a8(x) if a8 else bf16_value(x))
    ref, mag = X @ W.T, np.abs(X) @ np.abs(W).T
    err = np.abs(bf16_value(y) - ref)
    bad = err > 2.0**-8 * np.abs(ref) + 2.0**-14 * mag
    if bad.any():
        i = np.argwhere(bad)[0]
        raise AssertionError(f"{name}: {int(bad.sum())} of {bad.size} outputs off, first at {tuple(i)}: {bf16_value(y)[tuple(i)]} for {ref[tuple(i)]} (scale of the sums {mag[tuple(i)]:.3g})")
    return float((err / (mag + 1e-300)).max())


def case_linear(b, rng=np.random.default_rng(31)):
    """f8_linear for every kind of scales, 1 to 64 tokens, shapes that are not multiples of a patch, plain patches beside coded ones (the first patch, and
    the last, whose variable part starts at a word that is not 16-byte aligned in most cases: at least one such is required), dense patches (lin_dense,
    lin_crafted) at 5 to 64 tokens, every plan (a step ahead and the spills' pass together too), each fallback of the decode met at 17-64 tokens (lin_branches:
    each count above 0), and (emulation) the grids that cut row groups' K between CTAs; on a GPU also the real samples' bytes, gemma's 704 columns (a
    patch and a half's tail) with a row's scales, the same bits from two runs (K split between CTAs too), activations 2^-40 to 2^20 of the usual
    (lin_small), and CUDA graphs replayed on other inputs.
    W8A8 (a development library's GLYD_F8_LIN_A8=1, never a default) on compute capability 12.x, the only GPUs where the library takes it."""
    emu = isinstance(b, Cpu)
    e4 = emu or b.torch.cuda.get_device_capability()[0] == 12  # (W8A8: glyd_gpu.cu's f8_lin_a8)
    S, D, C3, C2 = "spread", "dense", 3, 2  # (the weights: lin_weights, lin_dense, lin_crafted's every)
    shapes = [(64, 128, 1, S), (130, 300, 13, S), (192, 384, 8, S), (320, 256, 17, S), (256, 512, 33, S), (64, 1024, 64, S), (96, 4096, 2, S), (128, 256, 5, D),
              (128, 256, 33, D), (128, 384, 64, D), (256, 256, 24, C2), (128, 512, 64, C3)]
    if not emu:
        shapes += [(4096, 4096, 1, D), (1344, 4096, 8, D), (6144, 256, 32, D), (512, 6144, 64, D), (5120, 2304, 3, D)]
        shapes += [(2816, 704, M, S, (4,)) for M in (1, 8, 33, 64)]  # (gemma-4's expert_down: K a patch and a half past 5)
    worst = worst8 = 0.0
    odd = 0  # (plain patches whose variable part starts at a word not 16-byte aligned)
    seen = collections.Counter()  # (lin_branches of the matrices of 17-64 tokens)
    nmat = 0
    for i, (N, K, M, gen, *kinds) in enumerate(shapes):
        for kind in (kinds[0] if kinds else range(5)):
            nmat += 1
            repo = ("zai-org/GLM-5.3-Flash", "mistralai/Mistral-Small-4-119B-2603", "MiniMaxAI/MiniMax-M3-MXFP8", "deepseek-ai/DeepSeek-V4.1-Flash",
                    "RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic")[kind]
            w8 = (lin_weights(N, K, rng, repo if not emu else None) if gen == S else lin_dense(N, K, rng) if gen == D else lin_crafted(N, K, rng, gen))
            if i % 2:  # plain patches (random bytes) among coded ones: the first and the last
                for r0, c0 in ((0, 0), ((N - 1) // 64 * 64, (K - 1) // 128 * 128)):
                    blk = w8[r0 : r0 + 64, c0 : c0 + 128]
                    blk[...] = rng.integers(0, 256, blk.shape).astype(np.uint8)
                w8[(w8 & 0x7F) == 0x7F] -= 1
            sc = lin_scales(kind, N, K, rng)
            p = fp8ref.pack(w8)
            odd += int((((p.heads[:, 3] >> 31) == 1) & (p.heads[:, 0] % 4 != 0)).sum())
            x = bf16_bits(rng.standard_normal((M, K)).astype(np.float32) * (1 + 3 * rng.random((M, 1))))
            W, n8 = lin_work(N, K), 1 if M <= 8 else 2 if M <= 16 else 4 if M <= 32 else 8
            if n8 >= 4:
                seen += lin_branches(p, N, K)
            plans = [None, (n8, 1), (n8, W), (8 if emu else n8, min(7, W))]  # (the library's; one CTA; a patch a CTA; 7: parts of row groups' K)
            if n8 >= 4:  # (17-64 tokens: the decode a step ahead or not, and a step ahead with the spills' pass)
                plans += [(n8, 0 if not emu else min(5, W), 0), (n8, min(3, W), 1), (n8, 0 if not emu else min(4, W), 1, 1), (n8, W, 1, 1)]
            plans += [(n8, 0 if not emu else min(4, W), None, 1), (n8, min(5, W), 0 if n8 >= 4 else None, 1), (n8, W, None, 1)]  # (the spills' pass first)
            for plan in plans:
                y = b.linear(p.core, p.heads, p.var, N, K, kind, sc, x, plan)
                worst = max(worst, lin_close(f"{N}x{K} M={M} {KINDS[kind]} plan {plan}", y, w8, kind, sc, x))
            if not emu and i == 0:
                assert np.array_equal(y, b.linear(p.core, p.heads, p.var, N, K, kind, sc, x)), "two runs differ"
            if e4 and kind in (0, 1, 4):  # W8A8: the plans of the grid, X as e4m3
                for ctas in [0 if not emu else min(4, W), 1, W, min(7, W)]:
                    plan = (n8, ctas, 0, 0, 1)
                    y = b.linear(p.core, p.heads, p.var, N, K, kind, sc, x, plan)
                    worst8 = max(worst8, lin_close(f"{N}x{K} M={M} {KINDS[kind]} plan {plan}", y, w8, kind, sc, x, a8=True))
                if not emu and i == 0:
                    assert np.array_equal(y, b.linear(p.core, p.heads, p.var, N, K, kind, sc, x, plan)), "two runs differ (W8A8)"
    assert odd, "no plain patch at a word that is not 16-byte aligned among the cases: another seed"
    miss = [k for k in ("var", "spills", "marks", "dlits", "dvar") if not seen[k]]
    assert not miss, f"no row block of 17-64 tokens takes the decode's {', '.join(miss)} path ({dict(seen)})"
    print(f"  f8_linear: {nmat} matrices within bf16's rounding of the exact products (every plan); the worst error {worst:.2e} of the sum of |x w|; "
          f"{odd} plain patches at words not 16-byte aligned; row blocks of 17-64 tokens by path: {dict(seen)}")
    if e4:
        print(f"  f8_linear W8A8: the block, tensor and channel matrices within bf16's rounding of the exact products of X as e4m3; the worst error {worst8:.2e}")
    if not emu:
        lin_repeat(b, rng)
        lin_small(b, rng)
        for M in (4, 33, 64):
            lin_graph(b, rng, M)
        if e4:
            os.environ["GLYD_F8_LIN_A8"] = "1"
            b.lib_need.cache_clear()
            try:
                for M in (4, 33, 64):
                    lin_graph(b, rng, M, a8=True)
            finally:
                del os.environ["GLYD_F8_LIN_A8"]
                b.lib_need.cache_clear()


def case_lin_quick(b, rng=np.random.default_rng(37)):
    """(GPU) A few f8_linear products for compute-sanitizer's memcheck and racecheck (python check_f8.py --only case_lin_quick under it): every kind, 1, 33 and
    64 tokens, plain patches at the first and the last patch (variable part at a word not 16-byte aligned), K split between 7 CTAs, the library's plan
    and the development plans (a step ahead, the spills' pass first, both; W8A8 on compute capability 12.x); dense patches too (lin_dense, lin_crafted)."""
    e4 = b.torch.cuda.get_device_capability()[0] == 12
    odd = 0
    for N, K, M, gen in ((130, 300, 1, None), (320, 1024, 33, None), (512, 8192, 64, None), (128, 384, 33, 0), (256, 256, 24, 2), (128, 512, 64, 3)):
        for kind in range(5):
            w8 = lin_weights(N, K, rng) if gen is None else lin_dense(N, K, rng) if gen == 0 else lin_crafted(N, K, rng, gen)
            for r0, c0 in ((0, 0), ((N - 1) // 64 * 64, (K - 1) // 128 * 128)):
                blk = w8[r0 : r0 + 64, c0 : c0 + 128]
                blk[...] = rng.integers(0, 256, blk.shape).astype(np.uint8)
            w8[(w8 & 0x7F) == 0x7F] -= 1
            sc, p = lin_scales(kind, N, K, rng), fp8ref.pack(w8)
            odd += int((((p.heads[:, 3] >> 31) == 1) & (p.heads[:, 0] % 4 != 0)).sum())
            x = bf16_bits(rng.standard_normal((M, K)).astype(np.float32))
            n8 = 1 if M <= 8 else 2 if M <= 16 else 4 if M <= 32 else 8
            plans = [None, (n8, 7), (n8, 0, None, 1)] + ([(n8, 0, 1), (n8, 0, 1, 1)] if n8 >= 4 else []) + ([(n8, 7, 0, 0, 1)] if e4 and kind in (0, 1, 4) else [])
            for plan in plans:
                y = b.linear(p.core, p.heads, p.var, N, K, kind, sc, x, plan)
                lin_close(f"quick {N}x{K} M={M} {KINDS[kind]} plan {plan}", y, w8, kind, sc, x, a8=bool(plan and len(plan) > 4 and plan[4]))
    assert odd, "no plain patch at a word that is not 16-byte aligned: another seed"


def lin_repeat(b, rng):
    """The same bits from two runs where K is split between CTAs (their partial sums added by the last): the library's plan and grids of 7 and 37 CTAs."""
    for N, K, M in ((512, 8192, 4), (512, 8192, 33), (1024, 6144, 64)):
        w8, sc = lin_weights(N, K, rng), lin_scales(0, N, K, rng)
        p = fp8ref.pack(w8)
        x = bf16_bits(rng.standard_normal((M, K)).astype(np.float32))
        n8 = 1 if M <= 8 else 2 if M <= 16 else 4 if M <= 32 else 8
        for plan in (None, (n8, 7), (n8, 37)):
            y = b.linear(p.core, p.heads, p.var, N, K, 0, sc, x, plan)
            lin_close(f"repeat {N}x{K} M={M} plan {plan}", y, w8, 0, sc, x)
            for _ in range(2):
                assert np.array_equal(y, b.linear(p.core, p.heads, p.var, N, K, 0, sc, x, plan)), f"two runs differ: {N}x{K} M={M} plan {plan}"


def lin_small(b, rng):
    """Activations 2^-40, 2^-20, 2^-12 and 2^20 of the usual, every kind: the kernel's bf16 weights are 2^-120 times the e4m3 values, so X goes in times
    2^64 (at 2^-120 alone, x w fell below fp32's normal range, 2^-126, from about 2^-16 of the usual: 1% off at 2^-20 on an RTX 5090, 8df2d686)."""
    for f in (2.0**-40, 2.0**-20, 2.0**-12, 2.0**20):
        for kind in range(5):
            N, K, M = 256, 1024, 8
            w8, sc = lin_weights(N, K, rng), lin_scales(kind, N, K, rng)
            p = fp8ref.pack(w8)
            x = bf16_bits(rng.standard_normal((M, K)).astype(np.float32) * np.float32(f))
            y = b.linear(p.core, p.heads, p.var, N, K, kind, sc, x)
            lin_close(f"x times {f:g}, {KINDS[kind]}", y, w8, kind, sc, x)


def lin_graph(b, rng, M, a8=False):
    """A CUDA graph that holds f8_linear of M tokens (K split over CTAs: its counters and partial sums too; a8: X's quantization too), replayed on other
    inputs in the same memory; its capture allocates no more than an empty capture does (the workspace is kept a stream: made by a call on the
    capture's stream first, as a caller must)."""
    torch, g = b.torch, b.g
    N, K = 512, 8192
    w8 = lin_weights(N, K, rng)
    sc = lin_scales(0, N, K, rng)
    ref = fp8ref.pack(w8)
    p = b.packed(ref.core, ref.heads, ref.var, (N, K))
    s = torch.from_numpy(sc).cuda()
    x = torch.zeros(M, K, dtype=torch.bfloat16, device="cuda")
    y = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
    st = torch.cuda.Stream()
    with torch.cuda.stream(st):
        g.f8_linear(p, "block", s, x, y)
    torch.cuda.synchronize()

    def capture(work):  # (a graph of work on stream st, and the bytes its capture allocated)
        gr, before = torch.cuda.CUDAGraph(), torch.cuda.memory_allocated()
        with torch.cuda.graph(gr, stream=st):
            work()
        return gr, torch.cuda.memory_allocated() - before

    _, torch_own = capture(lambda: None)
    graph, grown = capture(lambda: g.f8_linear(p, "block", s, x, y))
    assert grown <= torch_own, f"f8_linear's capture (M={M}{', a8' if a8 else ''}) allocated {grown} bytes, an empty one {torch_own}"
    for _ in range(3):
        xs = bf16_bits(rng.standard_normal((M, K)).astype(np.float32))
        x.copy_(torch.from_numpy(xs.view(np.int16)).cuda().view(torch.bfloat16))
        graph.replay()
        torch.cuda.synchronize()
        lin_close("graph replay", y.view(torch.int16).cpu().numpy().view(np.uint16), w8, 0, sc, xs, a8=a8)
    del graph
    torch.cuda.synchronize()


def case_dense(b):
    """Pack and unpack, bit for bit, of coded patches with many values past level 2 (lanes' units with more than 8 of them) and many literals."""
    for seed in range(8):
        w = lin_dense(256, 512, np.random.default_rng(seed))
        ref = fp8ref.pack(w)
        assert ref.plain() == 0
        same(f"pack dense {seed}", b.pack(w), ref)
        assert np.array_equal(b.unpack(ref.core, ref.heads, ref.var, w.shape), w), seed


def case_strides(b):
    """Unpack into rows apart and pack from rows apart."""
    rng = np.random.default_rng(11)
    for O, K, ld in ((64, 128, 128), (64, 128, 144), (70, 200, 200), (70, 200, 203), (130, 129, 160), (1, 300, 300), (5, 7, 33), (65, 17, 32)):
        w = weights(O, K, seed=int(rng.integers(1000)))
        if O > 1:
            w[rng.integers(0, O)] = rng.integers(0, 256, K, dtype=np.uint8)  # (a row of anything)
        ref = fp8ref.pack(w)
        same(f"strided pack {(O, K, ld)}", b.pack(w, ld=ld), ref)
        got = b.unpack(ref.core, ref.heads, ref.var, (O, K), ld=ld)
        assert np.array_equal(got, w), (O, K, ld)


def case_experts(b, rng=np.random.default_rng(12)):
    """A stack of E matrices packed as one [E O, K]: any of them, in any order and repeated, the others and the ids outside [0, E) left alone."""
    for E, O, K, ld in ((4, 64, 128, 128), (5, 128, 300, 300), (3, 192, 256, 272), (6, 64, 130, 144), (2, 256, 1024, 1024)):
        mats = [weights(O, K, seed=int(rng.integers(1000))) for _ in range(E)]
        stack = np.concatenate(mats)
        ref = fp8ref.pack(stack)
        same(f"stack {(E, O, K)}", b.pack(stack), ref)
        ids = [E - 1, 0, 2 % E, 0, -1, E, 1 % E, 2**31 - 1, -(2**31)]
        out = b.experts(ref.core, ref.heads, ref.var, E, O, K, ids, ld=ld)
        for i, e in enumerate(ids):
            if 0 <= e < E:
                assert np.array_equal(out[i], mats[e]), (E, O, K, i, e)
            else:
                assert (out[i] == 0xA5).all(), f"id {e}: its place was written"


def case_graph(b):
    """CUDA graphs that hold the unpack and the list's unpack: captured once, replayed with the ids and the pack's bytes rewritten in place. Their capture
    may allocate what an empty capture does and no more (torch 2.13 allocates 1,024 bytes a capture of its own: its generator's state)."""
    torch, g = b.torch, b.g
    E, O, K = 4, 128, 384
    mats = [weights(O, K, seed=20 + i) for i in range(E)]
    ref = fp8ref.pack(np.concatenate(mats))
    p = b.packed(ref.core, ref.heads, ref.var, (E * O, K))
    ids = torch.tensor([0, 1, 2, 3], dtype=torch.int32, device="cuda")
    out = torch.zeros(4, O, K, dtype=torch.uint8, device="cuda")
    whole = torch.zeros(E * O, K, dtype=torch.uint8, device="cuda")
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):  # (a warm-up first, as a capture needs)
        g.f8_unpack_experts(p, E, ids, out)
        g.f8_unpack_into(p, whole)
    torch.cuda.synchronize()

    def capture(work):  # a graph of work, and the bytes its capture allocated
        gr, before = torch.cuda.CUDAGraph(), torch.cuda.memory_allocated()
        with torch.cuda.graph(gr):
            work()
        return gr, torch.cuda.memory_allocated() - before

    empty, torch_own = capture(lambda: None)  # (torch's own per capture: its generator's state)
    graph, grown = capture(lambda: (g.f8_unpack_experts(p, E, ids, out), g.f8_unpack_into(p, whole)))
    assert grown <= torch_own, f"a capture allocated ({grown} bytes, an empty one {torch_own})"
    for order in ([3, 2, 1, 0], [1, 1, 0, -1], [2, 4, 3, 0]):
        ids.copy_(torch.tensor(order, dtype=torch.int32))
        out.fill_(0xA5)
        graph.replay()
        torch.cuda.synchronize()
        got = out.cpu().numpy()
        for i, e in enumerate(order):
            assert np.array_equal(got[i], mats[e]) if 0 <= e < E else (got[i] == 0xA5).all(), (order, i)
        assert np.array_equal(whole.cpu().numpy(), np.concatenate(mats))
        whole.zero_()
    # a pack of the same matrices in another order (as many words): the replay decodes what is in the same memory now
    other = [mats[i] for i in (2, 0, 3, 1)]
    ref2 = fp8ref.pack(np.concatenate(other))
    assert ref2.core.shape == ref.core.shape and ref2.var.shape == ref.var.shape
    p.core.copy_(torch.from_numpy(ref2.core)), p.heads.copy_(torch.from_numpy(ref2.heads.view(np.int32))), p.var.copy_(torch.from_numpy(ref2.var.view(np.int32)))
    ids.copy_(torch.tensor([3, 2, 1, 0], dtype=torch.int32))
    out.fill_(0xA5)
    graph.replay()
    torch.cuda.synchronize()
    assert all(np.array_equal(out[i].cpu().numpy(), other[e]) for i, e in enumerate([3, 2, 1, 0]))
    assert np.array_equal(whole.cpu().numpy(), np.concatenate(other))


def bandwidth(b):
    """The unpack's bandwidth and the pack's time for a 7168 x 18432 matrix (block-scaled Gaussian weights, as test_fp8ref's)."""
    torch, g = b.torch, b.g
    O, K = 7168, 18432
    x = torch.randn(O, K, device="cuda")
    top = x.abs().reshape(O // 128, 128, K // 128, 128).amax((1, 3))
    x = (x.reshape(O // 128, 128, K // 128, 128) * (448 / top)[:, None, :, None]).reshape(O, K).clamp_(-448, 448)
    w = x.to(g.F8_E4M3)
    del x, top
    torch.cuda.synchronize()
    times = []
    for _ in range(4):
        t = time.perf_counter()
        p = g.f8_pack(w)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t)
    out = torch.empty_like(w)
    g.f8_unpack_into(p, out)
    assert torch.equal(out.view(torch.uint8), w.view(torch.uint8)), "the 7168 x 18432 matrix does not come back"
    plain = p.plain()
    n = 200
    for _ in range(10):
        g.f8_unpack_into(p, out)
    a, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    runs = []
    for _ in range(5):
        a.record()
        for _ in range(n):
            g.f8_unpack_into(p, out)
        e.record()
        e.synchronize()
        runs.append(a.elapsed_time(e) / n)
    ms = min(runs)
    gbs = O * K / ms / 1e6
    print(f"bandwidth: {O} x {K} e4m3 on {torch.cuda.get_device_name()}: unpack {ms * 1e3:.1f} us = {gbs:.0f} GB/s of e4m3 written ({(O * K + p.nbytes()) / ms / 1e6:.0f} GB/s with the packed bytes read; "
          f"runs of {n}: {', '.join(f'{r * 1e3:.1f}' for r in runs)} us); pack {min(times) * 1e3:.1f} ms best of {len(times)} ({sorted(times)[len(times) // 2] * 1e3:.1f} ms median, "
          f"the size query's wait and the allocations in); {p.bits_per_value():.3f} bits a value, {plain} plain patches")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--emu", action="store_true", help="the CPU emulation in place of the GPU")
    ap.add_argument("--quick", action="store_true", help="(emu) leave out the 4096 x 7168 matrix and the fuzz")
    ap.add_argument("--samples", help="the directory gpu/fp8/fetch_samples.py wrote (default: GLYD_FP8_SAMPLES, else ~/.cache/glyd-fp8-samples)")
    ap.add_argument("--no-bw", action="store_true", help="no bandwidth line")
    ap.add_argument("--only", help="the cases whose names hold this text")
    a = ap.parse_args()
    if a.samples:
        os.environ["GLYD_FP8_SAMPLES"] = a.samples
    sys.path.insert(0, HERE)
    b = Cpu() if a.emu else Gpu()
    print(f"check_f8 on {b.name}")
    install(b)
    cases = [(n, f) for n, f in vars(T).items() if n.startswith("test_") and callable(f)]
    if a.quick:
        cases = [(n, f) for n, f in cases if n not in ("test_fuzz",)]
    cases += [("case_strides", case_strides), ("case_experts", case_experts), ("case_dense", case_dense), ("case_linear", case_linear)]
    if not a.emu:
        cases += [("case_graph", case_graph), ("case_lin_quick", case_lin_quick)]
    bad = 0
    for name, fn in cases:
        if a.only and a.only not in name:
            continue
        t = time.time()
        try:
            if name.startswith("case_"):
                fn(b)
            else:
                fn()
            print(f"{name} ok ({time.time() - t:.1f} s)", flush=True)
        except Exception as ex:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            print(f"{name} FAILED: {ex!r}", flush=True)
            bad += 1
    if not a.emu and not a.no_bw and not a.only:
        bandwidth(b)
    print("ALL PASSED (pack and unpack bit for bit; f8_linear within bf16's rounding of the exact products)" if not bad else f"{bad} FAILED")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
