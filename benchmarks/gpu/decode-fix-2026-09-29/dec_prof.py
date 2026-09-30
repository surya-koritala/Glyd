"""For ncu: one matrix of layer L (gate and up merged) decoded once by each variant, main's library, v0.25.0's and this
tree's load orders (GLYD_DEC_ORDER), whole (a warp a step) and then, with --kinds whole,ahead, ahead (2 warps an SM);
no other decode launched. The launches in the order printed.

    ncu -k regex:mma_unpack_kernel ... python dec_prof.py MAIN_TREE MAIN_LIB REL_LIB MODEL [--matrix gate_up] [--kinds whole]"""
import argparse, glob, importlib.util, json, os, sys
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
ap.add_argument("--matrix", default="gate_up")
ap.add_argument("--kinds", default="whole")
ap.add_argument("--orders", default="0,1,2,3")
args = ap.parse_args()
old, old_pack = both.load(args.main_tree, args.main_lib)
spec = importlib.util.spec_from_file_location("rel_lib", new.__file__)
rel = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rel)
rel.load(args.rel_lib)
d = args.model
idx = os.path.join(d, "model.safetensors.index.json")
index = json.load(open(idx))["weight_map"]


def get(name):
    with safe_open(os.path.join(d, index[name]), "pt", device="cuda") as f:
        return f.get_tensor(name)


pre = f"model.layers.{args.layer}."
names = {"qkv": [f"self_attn.{n}_proj" for n in "qkv"], "o": ["self_attn.o_proj"], "gate_up": ["mlp.gate_proj", "mlp.up_proj"], "down": ["mlp.down_proj"]}[args.matrix]
w = torch.cat([get(pre + n + ".weight") for n in names])
po, pn = old_pack(w), g.pack_mma12(w)
out = torch.empty(w.numel(), dtype=torch.bfloat16, device="cuda")
sms = torch.cuda.get_device_properties(0).multi_processor_count
n = 0
for kind in args.kinds.split(","):
    warps = 2 * sms if kind == "ahead" else 0
    for v, lib, p, o in [("main", old, po, None), ("v0.25.0", rel, pn, None)] + [(f"order {o}", new, pn, o) for o in args.orders.split(",") if o]:
        if o is not None:
            os.environ["GLYD_DEC_ORDER"] = o
        g._ext = lib
        g.mma_unpack(p, out, 0, None, warps)
        torch.cuda.synchronize()
        ok = torch.equal(out.view(torch.int16), w.reshape(-1).view(torch.int16))
        print(f"launch {n}: {v} {kind} ({warps} warps), {args.matrix} {tuple(w.shape)}, the weights' bits: {ok}", flush=True)
        n += 1
