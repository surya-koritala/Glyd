"""A model's layer products through one library, against cuBLAS bf16 in the same process: q,k,v and gate,up merged (as
e2e.py --merge runs them), the 12-bit layout from layer L's real weights, each call timed alone after an L2 flush (CUDA
events, median of REPS after 2), the kernel's output checked against the fp32 product (1e-2) and the same every run.
One line a product and M: TAG MODEL PRODUCT OxK M=m: cuBLAS <us> us | KERNEL <us> us <ratio>x
    GLYD_GPU_DIR=tree/gpu GLYD_GPU_LIB=lib.so python cu12_layer.py MODEL_DIR --tag T [--only o,qkv] [--M 129,256]
KERNEL: wg (mma_gemm_wg, Hopper) or mid (mma_gemm_mid, for a smoke test off Hopper).
"""
import argparse, json, os, sys, torch
import torch.nn.functional as F
from safetensors import safe_open

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--tag", required=True)
ap.add_argument("--layer", type=int, default=10)
ap.add_argument("--M", default="129,256,512,1024")
ap.add_argument("--only", default="")
ap.add_argument("--kernel", default="wg", choices=["wg", "mid"])
ap.add_argument("--reps", type=int, default=9)
ap.add_argument("--name", default="")
args = ap.parse_args()
sys.path.insert(0, os.environ["GLYD_GPU_DIR"])
import glyd_gpu as g

d = args.model
index = json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"]


def get(name):
    with safe_open(os.path.join(d, index[name]), "pt", device="cuda") as f:
        return f.get_tensor(name)


pre = f"model.layers.{args.layer}."
make = {
    "qkv": lambda: torch.cat([get(pre + f"self_attn.{n}_proj.weight") for n in "qkv"]),
    "o": lambda: get(pre + "self_attn.o_proj.weight"),
    "gate_up": lambda: torch.cat([get(pre + f"mlp.{n}_proj.weight") for n in ("gate", "up")]),
    "down": lambda: get(pre + "mlp.down_proj.weight"),
}
only = args.only.split(",") if args.only else list(make)
kern = g.mma_gemm_wg if args.kernel == "wg" else g.mma_gemm_mid
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


name = args.name or os.path.basename(d.rstrip("/"))
for k in only:
    w = make[k]()
    p = g.pack_mma12(w)
    for M in [int(m) for m in args.M.split(",")]:
        torch.manual_seed(M)
        x = torch.randn(M, w.shape[1], dtype=torch.bfloat16, device="cuda")
        ref = F.linear(x.float(), w.float())
        y = kern(p, x)
        err = ((y.float() - ref).abs().max() / ref.abs().max()).item()
        ok = err < 1e-2 and torch.equal(y, kern(p, x))
        c, t = timed(lambda: F.linear(x, w)), timed(lambda: kern(p, x))
        print(f"{args.tag} {name} {k} {w.shape[0]}x{w.shape[1]} M={M}: cuBLAS {c:.1f} us | {args.kernel} {t:.1f} us {t / c:.3f}x"
              + ("" if ok else f" | CHECK FAILED (error {err:.2e})"), flush=True)
    del w, p
    torch.cuda.empty_cache()
