"""The decode kernel (mma_unpack: a matrix decoded whole, for cuBLAS and exact mode) of main's 12-bit layout and library,
v0.25.0's (split byte) and this tree's load orders (GLYD_DEC_ORDER 0-3, Nib::load_decode; 0 v0.25.0's kernel), in one
process, on layer L's matrices (q, k, v and gate, up merged, as layer.py). First every variant's decode of every matrix
checked bit for bit against the weights; then each call timed alone after an L2 flush (CUDA events), a repetition
timing every variant in turn (the order rotated each time), the median of N. whole: a warp a step (mma_unpack's
default); ahead: 2 warps an SM, each taking every so many steps (as model.Ahead decodes a 12-bit prompt's matrices
ahead of their products: GeForce Ada's and an A10's).

    PYTHONPATH=FIX/bindings/python GLYD_GPU_LIB=FIX_LIB python dec_time.py MAIN_TREE MAIN_LIB REL_LIB MODEL [--layer 10] [--reps 21] [--kinds whole,ahead]"""
import argparse, ctypes, glob, importlib.util, json, os, sys
import torch
from safetensors import safe_open

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "splitbyte-2026-09-29"))
import both
from both import g, new

ap = argparse.ArgumentParser()
ap.add_argument("main_tree")
ap.add_argument("main_lib")
ap.add_argument("rel_lib")
ap.add_argument("model")
ap.add_argument("--layer", type=int, default=10)
ap.add_argument("--reps", type=int, default=21)
ap.add_argument("--kinds", default="whole,ahead")
ap.add_argument("--orders", default="0,1,2,3")
ap.add_argument("--final")
args = ap.parse_args()
old, old_pack = both.load(args.main_tree, args.main_lib)


def copy(name, path, api=None):  # another library through a copy of _lib (api: its C API, where not this tree's)
    spec = importlib.util.spec_from_file_location(name, new.__file__)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    m.API_VERSION = api or m.API_VERSION
    m.load(path)
    return m


rel = copy("rel_lib", args.rel_lib)  # v0.25.0's

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
PO = {k: old_pack(w) for k, w in W.items()}
PN = {k: g.pack_mma12(w) for k, w in W.items()}
V = [("main", old, PO, None), ("v0.25.0", rel, PN, None)] + [(f"order {o}", new, PN, o) for o in args.orders.split(",") if o]
# v0.25.1's C API is 5: 73b9560's 4 renumbered for the release (the same functions, arguments and 12-bit bytes).
V += [("v0.25.1", copy("final_lib", args.final, ctypes.CDLL(args.final).glyd_gpu_api_version()), PN, None)] if args.final else []
sms = torch.cuda.get_device_properties(0).multi_processor_count
KINDS = {"whole": 0, "ahead": 2 * sms}
kinds = [k for k in args.kinds.split(",") if k]
out = {k: torch.empty(w.numel(), dtype=torch.bfloat16, device="cuda") for k, w in W.items()}


def run(v, k, warps):
    _, lib, P, o = v
    if o is not None:
        os.environ["GLYD_DEC_ORDER"] = o  # read by this tree's library at each call
    g._ext = lib
    g.mma_unpack(P[k], out[k], 0, None, warps)


name = os.path.basename(os.path.dirname(os.path.dirname(d))).split("--")[-1] if "snapshots" in d else os.path.basename(d.rstrip("/"))
print(f"{name} layer {args.layer} on {torch.cuda.get_device_name()} ({sms} SMs): " + ", ".join(f"{k} {tuple(w.shape)}" for k, w in W.items())
      + "; exceptions a step, main / split byte: " + ", ".join(f"{k} {int(PO[k].exc_base[-1]) / (w.numel() / 1024):.3f} / {int(PN[k].exc_base[-1]) / (w.numel() / 1024):.3f}" for k, w in W.items()), flush=True)
bad = []
for v in V:  # each variant's own output (a pattern written first), the weights' bits; also loads each kernel before timing
    for k, w in W.items():
        for kind in kinds:
            out[k].view(torch.int16).fill_(-1)
            run(v, k, KINDS[kind])
            if not torch.equal(out[k].view(torch.int16), w.reshape(-1).view(torch.int16)):
                bad.append(f"{v[0]} {kind} {k}")
g._ext = new
print("bits: " + (f"DIFFER: {', '.join(bad)}" if bad else f"every variant's decode of every matrix ({', '.join(kinds)}) the weights' bits"), flush=True)
flush = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")


def once(f):
    flush.zero_()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    f()
    b.record()
    b.synchronize()
    return a.elapsed_time(b) * 1000


T = {}
for r in range(args.reps):
    order = V[r % len(V):] + V[: r % len(V)]
    for k in W:
        for kind in kinds:
            for v in order:
                T.setdefault((v[0], k, kind), []).append(once(lambda: run(v, k, KINDS[kind])))
g._ext = new
med = {key: sorted(t)[len(t) // 2] for key, t in T.items()}
moved = {k: PN[k].nbytes() + w.numel() * 2 for k, w in W.items()}  # split byte's pack read, the matrix written
for kind in kinds:
    what = f"{kind} ({KINDS[kind]} warps)" if KINDS[kind] else kind
    for k in list(W) + ["layer"]:
        t = {v[0]: sum(med[v[0], j, kind] for j in W) if k == "layer" else med[v[0], k, kind] for v in V}
        gbs = (sum(moved.values()) if k == "layer" else moved[k]) / 1e3
        cells = [f"main {t['main']:.1f} us"] + [f"{n} {t[n]:.1f} us ({t[n] / t['main']:.3f} main" + (f", {t[n] / t['v0.25.0']:.3f} v0.25.0)" if n != "v0.25.0" else ")") for n in t if n != "main"]
        print(f"  {what} {k}: " + " | ".join(cells) + f"  [GB/s: " + " ".join(f"{gbs / t[n]:.0f}" for n in t) + "]", flush=True)
print(f"median of {args.reps}; GB/s: split byte's pack read and the bf16 matrix written a microsecond, main's pack as large")
sys.exit(1 if bad else 0)
