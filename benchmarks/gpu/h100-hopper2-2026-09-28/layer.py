"""Per layer on Hopper: a model's layer L products (q,k,v and gate,up merged, as e2e.py --merge runs them), the 12-bit
layout from the real weights, against cuBLAS bf16 in the same process: each call timed alone after an L2 flush (CUDA
events, median of 9 after 2), as the ceiling job's glyd_layer.py. Routes: cuBLAS (F.linear), wg (mma_gemm_wg: the
library's kernel for M), dec (mma_unpack into a scratch buffer, then F.linear). wg checked against the fp32 product.
    GLYD_GPU_LIB=lib.so python layer.py MODEL_DIR [--layer 10] [--M 129,256,...] [--routes cuBLAS,wg,dec]
(GLYD_GPU_DIR: the gpu/ of the tree to run, default this checkout's; CHECK=0: no check against fp32, for builds that
skip a part.)
"""
import argparse, json, os, sys, torch
import torch.nn.functional as F
from safetensors import safe_open

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--layer", type=int, default=10)
ap.add_argument("--M", default="129,256,512,1024,2048,4096")
ap.add_argument("--routes", default="cuBLAS,wg,dec")
ap.add_argument("--only", default="")
ap.add_argument("--reps", type=int, default=9)
args = ap.parse_args()
sys.path.insert(0, os.environ.get("GLYD_GPU_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "gpu")))
import glyd_gpu as g

d = args.model
index = json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"]


def get(name):
    with safe_open(os.path.join(d, index[name]), "pt", device="cuda") as f:
        return f.get_tensor(name)


pre = f"model.layers.{args.layer}."
W = {
    "qkv": torch.cat([get(pre + f"self_attn.{n}_proj.weight") for n in "qkv"]),
    "o": get(pre + "self_attn.o_proj.weight"),
    "gate_up": torch.cat([get(pre + f"mlp.{n}_proj.weight") for n in ("gate", "up")]),
    "down": get(pre + "mlp.down_proj.weight"),
}
if args.only:
    W = {k: v for k, v in W.items() if k in args.only.split(",")}
flush = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")


def timed(f):
    ts = []
    for i in range(args.reps + 2):
        flush.zero_()
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        f()
        b.record()
        b.synchronize()
        if i >= 2:
            ts.append(a.elapsed_time(b) * 1000)
    ts.sort()
    return ts[len(ts) // 2]


name = os.path.basename(d.rstrip("/"))
P = {k: g.pack_mma12(w) for k, w in W.items()}
scratch = torch.empty(max(w.numel() for w in W.values()), dtype=torch.bfloat16, device="cuda")
routes = args.routes.split(",")
print(f"{name} layer {args.layer}, 12-bit, {torch.cuda.get_device_name()}, {os.environ.get('GLYD_GPU_LIB', 'JIT')}: "
      + ", ".join(f"{k} {tuple(w.shape)} {int(P[k].exc_base[-1])} exc" for k, w in W.items()), flush=True)
for M in [int(m) for m in args.M.split(",")]:
    tot = {}
    for k, w in W.items():
        torch.manual_seed(M)
        x = torch.randn(M, w.shape[1], dtype=torch.bfloat16, device="cuda")
        p = P[k]
        fn = {"cuBLAS": lambda: F.linear(x, w), "wg": lambda: g.mma_gemm_wg(p, x), "dec": lambda: F.linear(x, g.mma_unpack(p, scratch))}
        if "wg" in routes and os.environ.get("CHECK", "1") == "1":
            ref = F.linear(x.float(), w.float())
            y = g.mma_gemm_wg(p, x)
            err = ((y.float() - ref).abs().max() / ref.abs().max()).item()
            assert err < 1e-2 and torch.equal(y, g.mma_gemm_wg(p, x)), (k, M, err)
        t = {r: timed(fn[r]) for r in routes}
        for r, v in t.items():
            tot[r] = tot.get(r, 0.0) + v
        c = t.get("cuBLAS", 1.0)
        print(f"{name} {k} {w.shape[0]}x{w.shape[1]} M={M}: " + " | ".join(f"{r} {v:.1f} us {v / c:.3f}x" for r, v in t.items()), flush=True)
    c = tot.get("cuBLAS", 1.0)
    print(f"{name} layer M={M}: " + " | ".join(f"{r} {v:.1f} us {v / c:.3f}x" for r, v in tot.items()), flush=True)
