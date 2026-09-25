"""Fused matrix-vector against bf16 on one layer's matrices of a model
(and its output layer): kernel GPU times from the profiler."""
import sys, json, os, torch
import torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity
from safetensors import safe_open
import glyd_gpu as g

d = sys.argv[1]
index = json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"]
names = [n for n in index if n.startswith("model.layers.0.") and n.endswith("proj.weight")] + ["lm_head.weight"]
for name in names:
    with safe_open(os.path.join(d, index[name]), "pt", device="cuda") as f:
        w = f.get_tensor(name)
    x = torch.randn(w.shape[1], dtype=torch.bfloat16, device="cuda")
    fp, hp = g.pack_fast(w), g.pack(w)
    ref = F.linear(x, w).float()
    for label, y in (("fast", g.fast_gemv(fp, x)), ("huffman", g.gemv(hp, x))):
        err = ((y.float() - ref).abs().max() / ref.abs().max()).item()
        assert err < 1e-2, f"{name} {label}: relative error {err}"
    runs = {"bf16": lambda: F.linear(x, w), "fast": lambda: g.fast_gemv(fp, x), "huffman": lambda: g.gemv(hp, x)}
    for f in runs.values():
        f()
    torch.cuda.synchronize()
    times = {}
    for k, f in runs.items():
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(20):
                f()
            torch.cuda.synchronize()
        times[k] = sum(e.device_time_total for e in prof.key_averages() if e.device_type.name == "CUDA") / 20
    b = w.numel() * 2
    print(f"{name.split('.')[-2]:>10} {str(tuple(w.shape)):>14}: bf16 {times['bf16']:7.1f} us {b/times['bf16']/1e3:4.0f} GB/s | fast {times['fast']:7.1f} us {fp.nbytes()/times['fast']/1e3:4.0f} GB/s {times['bf16']/times['fast']:.2f}x | huffman {times['huffman']:7.1f} us {hp.nbytes()/times['huffman']/1e3:4.0f} GB/s {times['bf16']/times['huffman']:.2f}x")
    del w, fp, hp
    torch.cuda.empty_cache()
