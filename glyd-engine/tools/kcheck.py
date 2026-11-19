"""The engine's own kernels against vLLM's operations, on random inputs, bit for bit.

    python kcheck.py ENGINE_BINARY WORKDIR [ROWS]

Inputs are written to WORKDIR/in, `glyd-engine kcheck` runs the kernels (outputs in WORKDIR/out), and each output is compared with vLLM 0.30's
native definitions (vllm/ir/ops/layernorm.py, SiluAndMul.forward_native, RotaryEmbedding.forward_static, the cos/sin cache of
RotaryEmbeddingBase), run eagerly (a rounding to bf16 after every operation) and compiled by torch.compile (inductor: fused, fp32 inside),
which is how vLLM's default mode runs them. For each: the share of elements that are the same bits.
"""
import os
import subprocess
import sys

import numpy as np
import torch
import torch.nn.functional as F

try:
    import vllm._C  # noqa: F401  (vLLM's CUDA operations: torch.ops._C.*)
except Exception:
    pass

EPS = 1e-6
THETA = 1e6


def put(path, t):
    t.contiguous().view(torch.int16).cpu().numpy().tofile(path)


def get(path, shape, dev):
    return torch.from_numpy(np.fromfile(path, dtype=np.int16)).view(torch.bfloat16).view(*shape).to(dev)


def share(a, b):
    same = (a.view(torch.int16) == b.view(torch.int16)).float().mean().item()
    d = (a.float() - b.float()).abs()
    return f"{100 * same:7.3f}% equal bits, largest difference {d.max().item():.4g}"


def rms_native(x, w, eps):  # vllm/ir/ops/layernorm.py: rms_norm
    xf = x.to(torch.float32)
    var = xf.pow(2).mean(dim=-1, keepdim=True)
    xf = xf * torch.rsqrt(var + eps)
    return (xf.to(w.dtype) * w).to(x.dtype)


def add_rms_native(x, r, w, eps):  # fused_add_rms_norm
    xf = x.to(torch.float32) + r.to(torch.float32)
    res = xf.to(x.dtype)
    var = xf.pow(2).mean(dim=-1, keepdim=True)
    xf = xf * torch.rsqrt(var + eps)
    return (xf.to(w.dtype) * w).to(x.dtype), res


def silu_native(gu):
    d = gu.shape[-1] // 2
    return F.silu(gu[..., :d]) * gu[..., d:]


def cos_sin_cache(dim, max_pos, dev):  # RotaryEmbeddingBase._compute_cos_sin_cache, then .to(bf16)
    inv = 1.0 / (THETA ** (torch.arange(0, dim, 2, dtype=torch.float, device=dev) / dim))
    t = torch.arange(max_pos, dtype=torch.float, device=dev)
    freqs = torch.einsum("i,j -> ij", t, inv)
    return torch.cat((freqs.cos(), freqs.sin()), dim=-1)


def rope_apply(x, cos, sin):  # ApplyRotaryEmb.forward_static, neox
    cos = cos.unsqueeze(-2).to(x.dtype)
    sin = sin.unsqueeze(-2).to(x.dtype)
    x1, x2 = torch.chunk(x, 2, dim=-1)
    return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)


def qk_native(qkv, qw, kw, cache, pos, nq, nk, eps):  # Qwen3Attention: q_norm, k_norm per head, then the rotary embedding
    T = qkv.shape[0]
    q, k = qkv[:, : nq * 128], qkv[:, nq * 128 : (nq + nk) * 128]
    qn = rms_native(q.reshape(T, nq, 128), qw, eps)
    kn = rms_native(k.reshape(T, nk, 128), kw, eps)
    cos, sin = cache.index_select(0, pos).chunk(2, dim=-1)
    return rope_apply(qn, cos, sin).reshape(T, nq * 128), rope_apply(kn, cos, sin).reshape(T, nk * 128)


def main(engine, work, rows=4096):
    rows = int(rows)
    dev = "cuda"
    torch.manual_seed(0)
    ind, outd = os.path.join(work, "in"), os.path.join(work, "out")
    os.makedirs(ind, exist_ok=True)
    K, I, nq, nk, P = 4096, 12288, 32, 8, 4096
    bf = torch.bfloat16
    x = (torch.randn(rows, K, device=dev) * 1.5).to(bf)
    y = (torch.randn(rows, K, device=dev) * 0.7).to(bf)
    w = (torch.randn(K, device=dev) * 0.3 + 1.0).to(bf)
    gu = (torch.randn(rows, 2 * I, device=dev) * 2.0).to(bf)
    qkv = (torch.randn(rows, (nq + 2 * nk) * 128, device=dev) * 2.0).to(bf)
    qw = (torch.randn(128, device=dev) * 0.3 + 1.0).to(bf)
    kw = (torch.randn(128, device=dev) * 0.3 + 1.0).to(bf)
    pos = torch.randint(0, P, (rows,), device=dev, dtype=torch.int32)
    cache = cos_sin_cache(128, P, dev).to(bf)
    for n, t in (("norm_x", x), ("norm_y", y), ("norm_w", w), ("silu_gu", gu), ("rope_qkv", qkv), ("rope_qw", qw), ("rope_kw", kw), ("rope_cos", cache[:, :64]), ("rope_sin", cache[:, 64:])):
        put(os.path.join(ind, n + ".bin"), t)
    pos.cpu().numpy().tofile(os.path.join(ind, "rope_pos.bin"))
    open(os.path.join(ind, "norm.txt"), "w").write(f"{rows} {K} {EPS}")
    open(os.path.join(ind, "rope.txt"), "w").write(f"{rows} {nq} {nk} {P} {EPS} {THETA}")
    open(os.path.join(ind, "silu.txt"), "w").write(f"{rows} {I}")
    subprocess.run([engine, "kcheck", ind, outd], check=True)

    comp_rms, comp_add, comp_silu, comp_qk = (torch.compile(f) for f in (rms_native, add_rms_native, silu_native, qk_native))
    o = lambda n, s: get(os.path.join(outd, n), s, dev)
    print(f"rmsnorm ({rows} rows of {K}):")
    mine = o("norm_out.bin", (rows, K))
    print("  against native, eager:        ", share(mine, rms_native(x, w, EPS)))
    print("  against native, compiled:     ", share(mine, comp_rms(x, w, EPS)))
    try:
        out = torch.empty_like(x)
        torch.ops._C.rms_norm(out, x, w, EPS)
        print("  against vLLM's CUDA op:       ", share(mine, out))
    except Exception as e:
        print("  (vLLM's CUDA op not available:", type(e).__name__, ")")
    print(f"rmsnorm with the residual add:")
    ref_e = add_rms_native(x, y, w, EPS)
    ref_c = comp_add(x, y, w, EPS)
    mx, mo = o("add_x.bin", (rows, K)), o("add_out.bin", (rows, K))
    print("  residual, against eager:      ", share(mx, ref_e[1]), "| compiled:", share(mx, ref_c[1]))
    print("  normalized, against eager:    ", share(mo, ref_e[0]))
    print("  normalized, against compiled: ", share(mo, ref_c[0]))
    try:
        xx, rr = x.clone(), y.clone()
        torch.ops._C.fused_add_rms_norm(xx, rr, w, EPS)  # (in place: xx the normalized, rr the new residual)
        print("  normalized, against the CUDA op:", share(mo, xx), "| residual:", share(mx, rr))
    except Exception as e:
        print("  (vLLM's CUDA op not available:", type(e).__name__, ")")
    print("SiLU product:")
    ms = o("silu_out.bin", (rows, I))
    print("  against native, eager:        ", share(ms, silu_native(gu)))
    print("  against native, compiled:     ", share(ms, comp_silu(gu)))
    try:
        out = torch.empty(rows, I, dtype=bf, device=dev)
        torch.ops._C.silu_and_mul(out, gu)
        print("  against vLLM's CUDA op:       ", share(ms, out))
    except Exception as e:
        print("  (vLLM's CUDA op not available:", type(e).__name__, ")")
    print("query/key norm and rotary:")
    mq, mk = o("rope_q.bin", (rows, nq * 128)), o("rope_k.bin", (rows, nk * 128))
    pl = pos.long()
    eq, ek = qk_native(qkv, qw, kw, cache, pl, nq, nk, EPS)
    cq, ck = comp_qk(qkv, qw, kw, cache, pl, nq, nk, EPS)
    print("  query, against eager:         ", share(mq, eq), "| compiled:", share(mq, cq))
    print("  key,   against eager:         ", share(mk, ek), "| compiled:", share(mk, ck))
    # the engine's own cos/sin table against torch's (on the GPU, as vLLM makes it, and on the CPU, as HF does)
    mc = torch.from_numpy(np.fromfile(os.path.join(outd, "my_cos.bin"), dtype=np.int16)).view(bf).view(P, 64).to(dev)
    ms_ = torch.from_numpy(np.fromfile(os.path.join(outd, "my_sin.bin"), dtype=np.int16)).view(bf).view(P, 64).to(dev)
    cpu = cos_sin_cache(128, P, "cpu").to(bf).to(dev)
    print("rotary tables (the engine's, fp32 as torch then bf16):")
    print("  cos against torch on the GPU: ", share(mc, cache[:, :64]), "| on the CPU:", share(mc, cpu[:, :64]))
    print("  sin against torch on the GPU: ", share(ms_, cache[:, 64:]), "| on the CPU:", share(ms_, cpu[:, 64:]))


if __name__ == "__main__":
    main(*sys.argv[1:4])
