"""Several tokens at once: the fused fast-format GEMM against decoding the
matrix to bf16 then PyTorch's matmul, and against bf16 itself, on one
layer's matrices of a model; kernel GPU time from the profiler."""
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
        row.append(f"M={M}: bf16 {tb:6.0f} | decode+mm {td:6.0f} | fused {tf:6.0f} us")
    print(f"{name.split('.')[-2]:>10} {str(tuple(w.shape)):>14}  " + "  ".join(row))
    del w, p, scratch
    torch.cuda.empty_cache()
