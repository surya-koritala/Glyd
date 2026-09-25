"""GPU time of each kernel (profiler), fused matvec against bf16, by matrix."""
import sys, torch
import torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity
from safetensors.torch import load_file
import glyd_gpu as g

ts = load_file(sys.argv[1], device="cuda")
names = sys.argv[2:]
for name in names:
    w = ts[name]
    p = g.pack(w)
    x = torch.randn(w.shape[1], dtype=torch.bfloat16, device="cuda")
    for _ in range(5):
        g.gemv(p, x); F.linear(x, w)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(20):
            g.gemv(p, x)
            F.linear(x, w)
        torch.cuda.synchronize()
    for e in prof.key_averages():
        if e.device_type.name == "CUDA" and e.count >= 20:
            print(f"{name.split('.')[-2]:>12} {tuple(w.shape)} {e.key[:50]:50s} {e.device_time / 1:8.1f} us avg")
