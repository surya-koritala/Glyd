"""One layer's products (q, k, v and gate, up merged, as e2e.py --merge runs them) and decodes, main's 12-bit layout
and library against split byte and this tree's (both.py: one process, the same weights and inputs), each call timed
alone after an L2 flush (CUDA events), a repetition timing every one in turn, the median of N. By M: the library's
route for M on this GPU (mma_linear, route -1: a Linear's one-call path), the kernels apart (step kernel to 64
tokens, mid kernel 17-128, the prompt kernel past 64), and the whole matrix decoded (mma_unpack, as for cuBLAS and
decode ahead). Real weights of layer L.

    PYTHONPATH=TREE/bindings/python GLYD_GPU_LIB=TREE_LIB python layer.py MAIN_TREE MAIN_LIB MODEL [--layer 10] [--M 1,8,32,256,1024] [--reps 15]"""
import argparse, glob, json, os
import torch
from safetensors import safe_open
import both
from both import g, new

ap = argparse.ArgumentParser()
ap.add_argument("main_tree")
ap.add_argument("main_lib")
ap.add_argument("model")
ap.add_argument("--layer", type=int, default=10)
ap.add_argument("--M", default="1,8,32,256,1024")
ap.add_argument("--reps", type=int, default=15)
args = ap.parse_args()
old, old_pack = both.load(args.main_tree, args.main_lib)

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
P = {k: (old_pack(w), g.pack_mma12(w)) for k, w in W.items()}
for k, w in W.items():  # the same bits either way before anything is timed
    assert torch.equal(g.mma_unpack(P[k][1]).view(torch.int16), w.view(torch.int16))
    a, b = both.pair(old, lambda p: g.mma_unpack(p), *P[k])
    assert torch.equal(a.view(torch.int16), b.view(torch.int16))
flush = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")


def once(f):
    flush.zero_()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    f()
    b.record()
    b.synchronize()
    return a.elapsed_time(b) * 1000


def call(lib, f):
    def run():
        g._ext = lib
        f()
    return run


gpu = new.gpu()
name = os.path.basename(os.path.dirname(os.path.dirname(d))).split("--")[-1] if "snapshots" in d else os.path.basename(d.rstrip("/"))
print(f"{name} layer {args.layer} on {torch.cuda.get_device_name()}: " + ", ".join(f"{k} {tuple(w.shape)}" for k, w in W.items())
      + "; exceptions a step, 12-bit / split byte: " + ", ".join(f"{k} {int(P[k][0].exc_base[-1]) / (w.numel() / 1024):.3f} / {int(P[k][1].exc_base[-1]) / (w.numel() / 1024):.3f}" for k, w in W.items()), flush=True)
routes = {g.DECODE: "decode", g.GEMM: "step", g.MID: "mid", g.WG: "wg", g.BIG: "prompt", g.AHEAD: "ahead"}
for M in [int(m) for m in args.M.split(",")] + [0]:
    kinds = ["decode"] if M == 0 else ["route"] + (["step"] if M <= 64 else []) + (["mid"] if 17 <= M <= 128 else []) + (["prompt"] if M > 64 else [])
    cols = [f"{k} {side}" for k in kinds for side in ("12-bit", "split byte")]
    tot = {c: 0.0 for c in cols}
    cells, taken = [], set()
    for k, w in W.items():
        x = torch.randn(max(M, 1), w.shape[1], dtype=torch.bfloat16, device="cuda")
        taken.add(routes.get(g.route(P[k][1], gpu, M)[0], "?") if M else "decode")
        fs = {}
        for side, lib, p in (("12-bit", old, P[k][0]), ("split byte", new, P[k][1])):
            for kind in kinds:
                f = {"route": lambda p=p: g.mma_linear(p, x, None, -1), "step": lambda p=p: g.mma_gemm(p, x), "mid": lambda p=p: g.mma_gemm_mid(p, x),
                     "prompt": lambda p=p: g.mma_gemm_big(p, x), "decode": lambda p=p: g.mma_unpack(p)}[kind]
                fs[f"{kind} {side}"] = call(lib, f)
        for f in fs.values():  # warm
            f()
        ts = {c: [] for c in cols}
        for i in range(args.reps):
            for c in cols:
                ts[c].append(once(fs[c]))
        med = {c: sorted(v)[len(v) // 2] for c, v in ts.items()}
        for c in cols:
            tot[c] += med[c]
        cells.append(f"{k} " + "/".join(f"{med[c]:.1f}" for c in cols))
    g._ext = new
    print(f"  {'decode' if M == 0 else f'M={M:5}'} ({'/'.join(sorted(taken))}): " + " | ".join(f"{kind} {tot[kind + ' 12-bit']:8.1f} us, split byte {tot[kind + ' split byte']:8.1f} ({tot[kind + ' split byte'] / tot[kind + ' 12-bit']:.3f}x)" for kind in kinds)
          + f"  [{' | '.join(cells)}; us: {'/'.join(cols)}]", flush=True)
