"""The route SPLIT (option 2) per layer: layer L's products (q, k, v and gate, up merged, as e2e.py --merge runs them)
called through GLinear in a prompt's order, --copies layers of them (Linears of their own over the same packs, so that
the order recorded spans them all, as a model's does), each pass timed whole (CUDA events on the current stream; the
median of --reps, the modes in turn, after 2 passes each untimed, the first of which records the order): the route
SPLIT as the library routes M on this GPU, today's route (v0.25.0's: model.Split off on this device), and bf16
(F.linear on the weights, the same inputs). The route SPLIT's outputs within 1e-2 of fp32 (the first copy's) and the
same bits pass to pass. Real weights of layer L of MODEL (a directory, or a Hub repo in the cache).

    GLYD_GPU_LIB=LIB python layer.py MODEL [--layer 10] [--M 769,1024,2048,4096,8192] [--copies 8] [--reps 5]
"""
import argparse, glob, json, os, statistics, sys
import torch
import torch.nn.functional as F
from safetensors import safe_open

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "gpu"))
import glyd_gpu as g  # noqa: E402
from glyd.gpu import model as gm  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--layer", type=int, default=10)
ap.add_argument("--M", default="769,1024,2048,4096,8192")
ap.add_argument("--copies", type=int, default=8)
ap.add_argument("--reps", type=int, default=5)
args = ap.parse_args()

d = args.model
if not os.path.exists(os.path.join(d, "config.json")):
    d = glob.glob(os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub", f"models--{d.replace('/', '--')}", "snapshots", "*"))[0]
idx = os.path.join(d, "model.safetensors.index.json")
index = json.load(open(idx))["weight_map"] if os.path.exists(idx) else None


def get(name):
    with safe_open(os.path.join(d, index[name] if index else "model.safetensors"), "pt", device="cuda") as f:
        return f.get_tensor(name)


pre = f"model.layers.{args.layer}."
W = {
    "qkv": torch.cat([get(pre + f"self_attn.{n}_proj.weight") for n in "qkv"]),
    "o": get(pre + "self_attn.o_proj.weight"),
    "gate_up": torch.cat([get(pre + f"mlp.{n}_proj.weight") for n in ("gate", "up")]),
    "down": get(pre + "mlp.down_proj.weight"),
}
names = list(W)
P = {k: g.pack_mma12(w) for k, w in W.items()}
for k, w in W.items():
    assert torch.equal(g.mma_unpack(P[k]).view(torch.int16), w.view(torch.int16)), k
lins = [[gm.GLinear(P[k], None) for k in names] for _ in range(args.copies)]
gm.set_scratch(torch.nn.ModuleList([m for c in lins for m in c]), False)
dev = torch.device("cuda", torch.cuda.current_device())
gpu = lins[0][0].gpu
print(f"{torch.cuda.get_device_name()} (code {gpu}, {torch.cuda.get_device_properties(dev).multi_processor_count} SMs), layer {args.layer} of {os.path.basename(args.model.rstrip('/'))}: "
      + ", ".join(f"{k} {list(W[k].shape)}" for k in names) + f"; {args.copies} copies a pass; CUDA_DEVICE_MAX_CONNECTIONS={os.environ.get('CUDA_DEVICE_MAX_CONNECTIONS', '(unset)')}", flush=True)
ROUTE = {getattr(g, n): n for n in ("DECODE", "GEMM", "MID", "WG", "BIG", "AHEAD", "SPLIT")}

split = None  # this device's model.Split once a pass by the route has made it (False: it cannot run), kept while off


def pass_(mode, X):
    """A pass over the copies' products in order (mode: split, today or bf16): the first copy's outputs. The device's
    Split kept from pass to pass (a pass without it made anew would record the order again)."""
    global split
    gm.Split.of.pop(dev, None)
    if mode == "today":
        gm.Split.of[dev] = False
    elif split is not None:
        gm.Split.of[dev] = split
    out = []
    for c in range(args.copies):
        for i, k in enumerate(names):
            y = F.linear(X[k], W[k]) if mode == "bf16" else lins[c][i](X[k])
            if c == 0:
                out.append(y)
    if mode == "split":
        split = gm.Split.of.get(dev, split)
    return out


for M in [int(m) for m in args.M.split(",")]:
    torch.manual_seed(M)
    X = {k: torch.randn(M, w.shape[1], dtype=torch.bfloat16, device=dev) for k, w in W.items()}
    modes = ("split", "today", "bf16")
    ts, outs = {m: [] for m in modes}, {}
    for m in modes:  # untimed: the first records the order, the second follows it
        pass_(m, X)
        outs[m] = pass_(m, X)
    torch.cuda.synchronize()
    for _ in range(args.reps):
        for m in modes:
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            ys = pass_(m, X)
            b.record()
            b.synchronize()
            ts[m].append(a.elapsed_time(b))
            if m == "split":
                assert all(torch.equal(p.view(torch.int16), q.view(torch.int16)) for p, q in zip(ys, outs[m])), f"M={M}: the route SPLIT's outputs differ pass to pass"
    worst = 0.0
    for k, y in zip(names, outs["split"]):
        ref = F.linear(X[k].float(), W[k].float())
        worst = max(worst, ((y.float() - ref).abs().max() / ref.abs().max()).item())
    assert worst < 1e-2, f"M={M}: {worst} off fp32"
    gm.Split.of.pop(dev, None)
    if split is not None:
        gm.Split.of[dev] = split
    routes = ", ".join(f"{k} {ROUTE[lins[0][i].route(M)[0]]}" + (f" ({g.split_sms(P[k], gpu | g.WITH_SPLIT, M)} SMs)" if lins[0][i].route(M)[0] == g.SPLIT else "") for i, k in enumerate(names))
    t = {m: statistics.median(v) / args.copies for m, v in ts.items()}
    print(f"M={M}: a layer bf16 {t['bf16']:.3f} ms, today's route {t['today']:.3f} ms ({t['today'] / t['bf16']:.3f}x bf16), "
          f"SPLIT {t['split']:.3f} ms ({t['split'] / t['bf16']:.3f}x bf16, {t['split'] / t['today']:.3f}x today's); routes: {routes}; "
          f"the ring {'ran' if isinstance(split, gm.Split) else 'could not run' if split is False else 'not made'}; {worst:.1e} off fp32, the same bits pass to pass; "
          f"SPLIT's passes {min(ts['split']) / args.copies:.3f}-{max(ts['split']) / args.copies:.3f} ms", flush=True)
    del X, outs, ys
