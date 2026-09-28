"""One layer's products (q,k,v and gate,up merged, as e2e --merge runs them) at M tokens: cuBLAS bf16 against
mma_gemm_big variants, L2 flushed before each call, CUDA events, median of N. Real weights of layer L.

    python mb.py MODEL_DIR [--layer 10] [--M 256,512,1024,2048,4096] [--fmt mma,mma12] [--variants 0,1,2] [--reps 7]

Variants as the library in GLYD_GPU_LIB takes them: 0 its choice of blocks, 1 blocks of 128 tokens by two row blocks,
2 of 256 by one (the logs' other numbers came from scratch builds: gpu/README.md, prompts on GeForce Ada).
"""
import argparse, glob, json, os, sys, torch
import torch.nn.functional as F
from safetensors import safe_open

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--layer", type=int, default=10)
ap.add_argument("--M", default="256,512,1024,2048,4096")
ap.add_argument("--fmt", default="mma,mma12")
ap.add_argument("--variants", default="0")
ap.add_argument("--reps", type=int, default=7)
ap.add_argument("--check", action="store_true", help="each variant's output against fp32 (and the same every run)")
ap.add_argument("--only", default="", help="products to time, e.g. gate_up,down")
ap.add_argument("--interleave", action="store_true", help="each repetition times cuBLAS and every variant in turn (drift shared), medians")
args = ap.parse_args()
sys.path.insert(0, os.environ.get("GLYD_GPU_DIR", "."))
import glyd_gpu as g

d = args.model
if not os.path.exists(os.path.join(d, "config.json")):
    d = glob.glob(os.path.expanduser(f"~/p1/hf/hub/models--Qwen--{d}/snapshots/*"))[0]
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


def timed(f, reps):
    ts = []
    for i in range(reps + 2):
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


Ms = [int(m) for m in args.M.split(",")]
variants = [int(v) for v in args.variants.split(",")]
name = os.path.basename(d.rstrip("/")) if "snapshots" not in d else d.split("models--Qwen--")[1].split("/")[0]
print(f"{name} layer {args.layer}: " + ", ".join(f"{k} {tuple(w.shape)}" for k, w in W.items()), flush=True)
for fmt in args.fmt.split(","):
    P = {k: (g.pack_mma(w) if fmt == "mma" else g.pack_mma12(w)) for k, w in W.items()}
    if fmt == "mma12":
        print("  exceptions a step: " + ", ".join(f"{k} {int(p.exc_base[-1]) / (p.n / 1024):.3f} (max {int((p.exc_base[1:] - p.exc_base[:-1]).max())})" for k, p in P.items()))
    for M in Ms:
        tot = {"cublas": 0.0}
        row = {}
        for k, w in W.items():
            x = torch.randn(M, w.shape[1], dtype=torch.bfloat16, device="cuda")
            if args.interleave:
                fs = [lambda: F.linear(x, w)] + [(lambda v=v: g.mma_gemm_big(P[k], x, None, v)) for v in variants]
                ts = [[] for _ in fs]
                for i in range(args.reps + 1):
                    for f_, tl in zip(fs, ts):
                        tl.append(timed(f_, 1))
                meds = [sorted(tl[1:])[len(tl[1:]) // 2] for tl in ts]
                row[k] = meds
                tot["cublas"] += meds[0]
                for v, t in zip(variants, meds[1:]):
                    tot[v] = tot.get(v, 0.0) + t
                continue
            t = timed(lambda: F.linear(x, w), args.reps)
            row[k] = [t]
            tot["cublas"] += t
            if args.check:
                ref = F.linear(x.float(), w.float())
            for v in variants:
                t = timed(lambda: g.mma_gemm_big(P[k], x, None, v), args.reps)
                row[k].append(t)
                tot[v] = tot.get(v, 0.0) + t
                if args.check:
                    y = g.mma_gemm_big(P[k], x, None, v)
                    err = ((y.float() - ref).abs().max() / ref.abs().max()).item()
                    same = torch.equal(y, g.mma_gemm_big(P[k], x, None, v))
                    assert err < 1e-2 and same, (fmt, k, M, v, err, same)
        cells = " | ".join(f"{k} " + "/".join(f"{t:.0f}" for t in ts) for k, ts in row.items())
        c = tot["cublas"]
        print(f"  {fmt:5} M={M:5}: layer cuBLAS {c:7.0f} us, " + ", ".join(f"v{v} {tot[v]:7.0f} ({tot[v] / c:.3f}x)" for v in variants) + f"  [{cells}]", flush=True)
