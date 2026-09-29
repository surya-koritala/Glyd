"""A prompt's routes per layer on this GPU, against cuBLAS bf16 in the same process: layer L's four products (q,k,v and
gate,up merged, as e2e.py --merge runs them) of a Qwen3 checkpoint, packed in the layout asked for.

- Each product alone, after an L2 flush (CUDA events, median of REPS after 2): cuBLAS; the fused prompt kernel
  (mma_gemm_big) as the library picks its blocks (v0) and each variant (v1: 128 tokens by two row blocks, v2: 256 by
  one, v3: 256 by two, 12-bit); the matrix decoded, then cuBLAS (dec+cuBLAS) and the decode alone (dec); and the
  route GLinear takes on this GPU today (path).
- A pass of CHAIN layers (the four products each, the same packs, no flush), as a prompt runs them, by route:
  cuBLAS; fused (GLinear, every prompt fused); decoded (each matrix decoded, then cuBLAS, on the current stream);
  ahead (each matrix decoded beside the products before it, model.Ahead, as on GeForce Ada; its first pass records
  the order). The pass's time, its ratio to cuBLAS's, and the GPU's SM clock and power (nvidia-smi every 100 ms,
  medians over the timed passes).
- --mid: steps of 17-64 tokens, a pass each: mma_gemm (a step's kernel) against mma_gemm_mid (GLinear.mid), the
  12-bit layout.

    GLYD_GPU_LIB=lib.so python route_layer.py SNAPSHOT_DIR [--layout mma12] [--M 128,...] [--mid 17,...] [--chain 12]

One line per product and M, then per route and M; a header line with the GPU, the library and the layout."""
import argparse, json, os, subprocess, sys, time, torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--layer", type=int, default=10)
ap.add_argument("--layout", default="mma12", choices=["mma12", "mma"])
ap.add_argument("--M", default="128,256,384,512,640,768,1024,1536,2048,3072,4096")
ap.add_argument("--mid", default="", help="step sizes (tokens) for mma_gemm against mma_gemm_mid, e.g. 17,24,32,48,64")
ap.add_argument("--chain", type=int, default=12, help="layers a pass")
ap.add_argument("--reps", type=int, default=9)
ap.add_argument("--passes", type=int, default=5)
ap.add_argument("--tag", default="", help="a label for every line (the library's)")
args = ap.parse_args()
import glyd_gpu as g  # noqa: E402 (gpu/ on sys.path: the job runs from there)
from glyd.gpu import model as gm  # noqa: E402

d = args.model
index = json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"]


def get(name):
    with safe_open(os.path.join(d, index[name]), "pt", device="cuda") as f:
        return f.get_tensor(name)


pre = f"model.layers.{args.layer}."
W = {"qkv": torch.cat([get(pre + f"self_attn.{n}_proj.weight") for n in "qkv"]), "o": get(pre + "self_attn.o_proj.weight"),
     "gate_up": torch.cat([get(pre + f"mlp.{n}_proj.weight") for n in ("gate", "up")]), "down": get(pre + "mlp.down_proj.weight")}
twelve = args.layout == "mma12"
P = {k: (g.pack_mma12 if twelve else g.pack_mma)(w) for k, w in W.items()}
flush = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
scratch = torch.empty(max(w.numel() for w in W.values()), dtype=torch.bfloat16, device="cuda")
tag = (args.tag + " ") if args.tag else ""
never = 1 << 62


def timed(f, reps=args.reps, flushed=True):
    ts = []
    for i in range(reps + 2):
        if flushed:
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


class Smi:
    """nvidia-smi's SM clock (MHz) and power (W) every 100 ms while it is open: medians."""

    def __enter__(self):
        self.p = subprocess.Popen(["nvidia-smi", "-i", str(torch.cuda.current_device()), "--query-gpu=clocks.sm,power.draw", "--format=csv,noheader,nounits", "-lms", "100"], stdout=subprocess.PIPE, text=True)
        return self

    def __exit__(self, *e):
        time.sleep(0.25)
        self.p.terminate()
        rows = [l.split(",") for l in self.p.communicate()[0].splitlines() if l.count(",") == 1]
        vals = [(float(a), float(b)) for a, b in rows if a.strip().replace(".", "").isdigit() and b.strip().replace(".", "").isdigit()]
        med = lambda v: sorted(v)[len(v) // 2] if v else float("nan")
        self.clock, self.power = med([c for c, _ in vals]), med([p for _, p in vals])


def lins(route):
    """CHAIN layers' GLinears over the four packs, routed: fused (no prompt decoded), decoded (from 65 tokens),
    ahead (decoded ahead from 65), mid (steps by mma_gemm_mid), step (steps by mma_gemm); the scratch buffer set."""
    out = []
    for _ in range(args.chain):
        layer = []
        for k in ("qkv", "o", "gate_up", "down"):
            lin = gm.GLinear(P[k], None)
            lin.dec = 65 if route == "decoded" else never
            lin.ahead = 65 if route == "ahead" else never
            if route in ("mid", "step"):
                lin.mid = route == "mid"
            lin.step = lin._step()
            layer.append(lin)
        out.append(layer)
    gm.Ahead.reset(torch.device("cuda", torch.cuda.current_device()))
    gm.Scratch.buf.clear()
    gm.set_scratch(nn.ModuleList([l for layer in out for l in layer]), False)
    return out


def chain(route, M):
    """A pass's time (us) by route at M tokens, and the clock and power while it ran."""
    xs = {k: torch.randn(M, w.shape[1], dtype=torch.bfloat16, device="cuda") for k, w in W.items()}
    if route == "cuBLAS":
        run = lambda: [F.linear(xs[k], W[k]) for _ in range(args.chain) for k in W]
    else:
        ls = lins(route)
        run = lambda: [lin(xs[k]) for layer in ls for k, lin in zip(("qkv", "o", "gate_up", "down"), layer)]
    with torch.no_grad():
        run()  # (ahead: the order recorded)
        run()
        torch.cuda.synchronize()
        with Smi() as s:
            t = timed(run, reps=args.passes, flushed=False)
    return t, s


name = next((p.split("--", 1)[1].replace("--", "/") for p in d.split(os.sep) if p.startswith("models--")), os.path.basename(d.rstrip("/")))  # (a Hub cache's snapshot: the repo)
print(f"{tag}{name} layer {args.layer}, {args.layout}, {torch.cuda.get_device_name()} (sm_{''.join(map(str, torch.cuda.get_device_capability()))}), "
      f"kernels from {os.environ.get('GLYD_GPU_LIB', 'the JIT build')}; " + ", ".join(f"{k} {tuple(w.shape)}" for k, w in W.items()), flush=True)
with torch.no_grad():
    for M in [int(m) for m in args.M.split(",") if m]:
        for k, w in W.items():
            x, p = torch.randn(M, w.shape[1], dtype=torch.bfloat16, device="cuda"), P[k]
            routes = {"cuBLAS": lambda: F.linear(x, w), "v0": lambda: g.mma_gemm_big(p, x)}
            for v in ((1, 2, 3) if twelve else (1, 2)):
                routes[f"v{v}"] = lambda v=v: g.mma_gemm_big(p, x, None, v)
            routes["dec+cuBLAS"] = lambda: F.linear(x, g.mma_unpack(p, scratch))
            routes["dec"] = lambda: g.mma_unpack(p, scratch)
            t = {r: timed(f) for r, f in routes.items()}
            kern = gm.GLinear(p, None).kernel(M)
            path = "dec+cuBLAS" if kern is None else "v0" if kern is g.mma_gemm_big else kern.__name__
            c = t["cuBLAS"]
            print(f"{tag}{k} {w.shape[0]}x{w.shape[1]} M={M}: cuBLAS {c:.1f} us | path ({path}) {t.get(path, float('nan')) / c:.3f}x | "
                  + " | ".join(f"{r} {v:.1f} us {v / c:.3f}x" for r, v in t.items() if r != "cuBLAS"), flush=True)
        res = {r: chain(r, M) for r in ("cuBLAS", "fused", "decoded", "ahead")}
        c = res["cuBLAS"][0]
        print(f"{tag}pass of {args.chain} layers M={M}: " + " | ".join(f"{r} {t / 1000:.2f} ms {t / c:.3f}x ({s.clock:.0f} MHz, {s.power:.0f} W)" for r, (t, s) in res.items()), flush=True)
    for M in [int(m) for m in args.mid.split(",") if m]:
        if not twelve:
            break
        res = {r: chain(r, M) for r in ("cuBLAS", "step", "mid")}
        c = res["cuBLAS"][0]
        print(f"{tag}steps, pass of {args.chain} layers M={M}: " + " | ".join(f"{r} {t / 1000:.3f} ms {t / c:.3f}x ({s.clock:.0f} MHz, {s.power:.0f} W)" for r, (t, s) in res.items()), flush=True)
