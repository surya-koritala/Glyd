"""Several tokens at once: the fused fast-format GEMM, the batched product
and the mma layout's tensor-core product against decoding the matrix to
bf16 then PyTorch's matmul, and against bf16 itself, on one layer's
matrices of a model; kernel GPU time from the profiler.

    python gemm.py MODEL_DIR 1,2,4,8,16,32"""
import sys, json, os, torch
import torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity
from safetensors import safe_open
import glyd_gpu as g

d = sys.argv[1]
Ms = [int(m) for m in sys.argv[2].split(",")]
index = json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"]
names = [n for n in index if n.startswith("model.layers.0.") and n.endswith("proj.weight")]


def gpu_us(f):
    for _ in range(2):
        f()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(10):
            f()
        torch.cuda.synchronize()
    return sum(e.device_time_total for e in prof.key_averages() if e.device_type.name == "CUDA") / 10


for name in names:
    with safe_open(os.path.join(d, index[name]), "pt", device="cuda") as f:
        w = f.get_tensor(name)
    p = g.pack_fast(w)
    q = g.pack_mma(w)
    assert torch.equal(g.mma_unpack(q).view(torch.int16), w.view(torch.int16)), f"{name}: mma layout not exact"
    q12 = g.pack_mma12(w)
    assert torch.equal(g.mma_unpack(q12).view(torch.int16), w.view(torch.int16)), f"{name}: mma12 layout not exact"
    scratch = torch.empty(p.n, dtype=torch.bfloat16, device="cuda")
    row = []
    for M in Ms:
        x = torch.randn(M, w.shape[1], dtype=torch.bfloat16, device="cuda")
        ref = F.linear(x, w).float()
        err = ((g.fast_gemm(p, x).float() - ref).abs().max() / ref.abs().max()).item()
        assert err < 1e-2, f"{name} M={M}: {err}"
        tb = gpu_us(lambda: F.linear(x, w))
        td = gpu_us(lambda: F.linear(x, g.fast_unpack(p, scratch)))
        tf = gpu_us(lambda: g.fast_gemm(p, x))
        tv = ""
        if M in (2, 4, 8, 16):
            yv = g.fast_bgemv(p, x).float()
            ev = ((yv - ref).abs().max() / ref.abs().max()).item()
            assert ev < 1e-2, f"{name} bgemv M={M}: {ev}"
            tv = f" | bgemv {gpu_us(lambda: g.fast_bgemv(p, x)):6.0f}"
        if M <= 64:
            em = ((g.mma_gemm(q, x).float() - ref).abs().max() / ref.abs().max()).item()
            assert em < 1e-2, f"{name} mma M={M}: {em}"
            tv += f" | mma {gpu_us(lambda: g.mma_gemm(q, x)):6.0f}"
            e12 = ((g.mma_gemm(q12, x).float() - ref).abs().max() / ref.abs().max()).item()
            assert e12 < 1e-2 and torch.equal(g.mma_gemm(q12, x), g.mma_gemm(q12, x)), f"{name} mma12 M={M}: {e12}"
            tv += f" | mma12 {gpu_us(lambda: g.mma_gemm(q12, x)):6.0f}"
        if M >= 32:
            eb = ((g.mma_gemm_big(q, x, variant=1).float() - ref).abs().max() / ref.abs().max()).item()
            assert eb < 1e-2, f"{name} mma big M={M}: {eb}"
            tv += f" | big128 {gpu_us(lambda: g.mma_gemm_big(q, x, variant=1)):6.0f}"
            e12 = ((g.mma_gemm_big(q12, x, variant=1).float() - ref).abs().max() / ref.abs().max()).item()
            assert e12 < 1e-2, f"{name} mma12 big M={M}: {e12}"
            tv += f" | big128/12 {gpu_us(lambda: g.mma_gemm_big(q12, x, variant=1)):6.0f}"
            if M >= 256:
                e2 = ((g.mma_gemm_big(q, x, variant=2).float() - ref).abs().max() / ref.abs().max()).item()
                assert e2 < 1e-2, f"{name} mma big256 M={M}: {e2}"
                tv += f" | big256 {gpu_us(lambda: g.mma_gemm_big(q, x, variant=2)):6.0f}"
                tv += f" | big256/12 {gpu_us(lambda: g.mma_gemm_big(q12, x, variant=2)):6.0f}"
        row.append(f"M={M}: bf16 {tb:6.0f} | decode+mm {td:6.0f} | fused {tf:6.0f}{tv} us")
    print(f"{name.split('.')[-2]:>10} {str(tuple(w.shape)):>14} fast {p.bits_per_weight():.2f} mma {q.bits_per_weight():.2f} mma12 {q12.bits_per_weight():.2f} bits  " + "  ".join(row))
    del w, p, q, scratch
    torch.cuda.empty_cache()
