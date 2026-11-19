"""The engine's KV tables (glyd-engine kvcal, safetensors) against the plugin's (python -m glyd_gpu kvcal, npz) for the same model and positions.

    python cal_compare.py ENGINE.safetensors PLUGIN.npz
"""
import json
import re
import struct
import sys

import numpy as np


def read_safetensors(path):
    b = open(path, "rb").read()
    n = struct.unpack("<Q", b[:8])[0]
    h = json.loads(b[8 : 8 + n])
    out = {}
    for k, v in h.items():
        if k == "__metadata__":
            continue
        dt = {"U8": np.uint8, "I16": np.int16, "I32": np.int32}[v["dtype"]]
        s, e = v["data_offsets"]
        out[k] = np.frombuffer(b[8 + n + s : 8 + n + e], dtype=dt).reshape(v["shape"])
    return out, h.get("__metadata__", {})


def main(eng, plug):
    e, em = read_safetensors(eng)
    z = np.load(plug, allow_pickle=False)
    names = [str(x) for x in z["layers"]]
    order = sorted(range(len(names)), key=lambda i: int(re.search(r"layers\.(\d+)\.", names[i]).group(1)))
    pk, pv = z["k"][order], z["v"][order]
    pm = json.loads(str(z["meta"]))
    emeta = json.loads(em["meta"])
    for w, a, b in (("k", e["k"], pk), ("v", e["v"], pv)):
        eq = a == b
        d = np.abs(a.astype(int) - b.astype(int))
        print(f"windows {w}: {a.shape}, equal in {eq.mean() * 100:.3f}% ({int((~eq).sum())} of {eq.size} differ), differ by {d[~eq].mean() if (~eq).any() else 0:.2f} on average, at most {d.max()}; layers with a difference: {sorted(set(np.nonzero((~eq).any(axis=(1, 2)))[0].tolist()))}")
    if "exc" in e and "exc" in z.files:
        a, b = e["exc"], z["exc"]
        print(f"exceptions per token and layer {a.shape} against {b.shape}: equal in {(a == b).mean() * 100:.3f}%, mean difference {np.abs(a.astype(int) - b.astype(int)).mean():.4f}, largest {np.abs(a.astype(int) - b.astype(int)).max()}")
    for key in ("positions", "refused", "layers", "heads", "channels"):
        print(f"meta {key}: engine {emeta.get(key)}, plugin {pm.get(key)}")
    print(f"meta held_out_rate: engine {emeta['held_out_rate']}, plugin {pm['held_out_rate']}")
    pe, pp = emeta["store"], pm["store"]
    for key in ("swept", "bay", "capacity", "worst_chunks_per_layer_leaf", "worst_token", "median_chunks_per_layer_leaf", "hard", "hard_text", "hard_ids", "hard_worst_leaf_need", "second_layers"):
        print(f"store {key}: engine {pe.get(key)}, plugin {pp.get(key)}")
    print(f"store hard_text_ids equal: {pe.get('hard_text_ids') == pp.get('hard_text_ids')}; hard_ids_sample equal: {pe.get('hard_ids_sample') == pp.get('hard_ids_sample')}")
    print(f"pages: engine {emeta['pages']}\n       plugin {pm['pages']}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
