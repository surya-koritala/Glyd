"""The plugin's KV tables file (npz, `python -m glyd_gpu.kvcal`) as the engine's sidecar: kv-tables.safetensors with k and v (uint8
[layers, kv heads, channels], layers in order) and the tables' meta (JSON) in the header. Run where numpy is.

    python npz_to_tables.py ~/.cache/glyd/kv/kvtab-XXXX.npz /path/to/packed-model-dir/kv-tables.safetensors
"""
import json
import re
import struct
import sys

import numpy as np


def main(src, dst):
    z = np.load(src, allow_pickle=False)
    names = [str(n) for n in z["layers"]]
    order = sorted(range(len(names)), key=lambda i: int(re.search(r"layers\.(\d+)\.", names[i]).group(1)))
    k, v = np.ascontiguousarray(z["k"][order]), np.ascontiguousarray(z["v"][order])
    assert k.dtype == np.uint8 and v.dtype == np.uint8 and k.shape == v.shape and k.ndim == 3, (k.dtype, k.shape, v.shape)
    meta = json.loads(str(z["meta"]))
    for key in ("per_kind", "per_tensor", "by_depth"):  # (the calibration's own measures: not the engine's)
        meta.pop(key, None)
    meta["store"] = {a: b for a, b in meta.get("store", {}).items() if a != "top"}
    kb, vb = k.tobytes(), v.tobytes()
    header = {
        "__metadata__": {"format": "glyd-kv-tables-v0", "meta": json.dumps(meta, sort_keys=True)},
        "k": {"dtype": "U8", "shape": list(k.shape), "data_offsets": [0, len(kb)]},
        "v": {"dtype": "U8", "shape": list(v.shape), "data_offsets": [len(kb), len(kb) + len(vb)]},
    }
    h = json.dumps(header, separators=(",", ":"))
    h += " " * (-(8 + len(h)) % 8)
    with open(dst, "wb") as f:
        f.write(struct.pack("<Q", len(h)))
        f.write(h.encode())
        f.write(kb)
        f.write(vb)
    print(f"{dst}: {k.shape[0]} layers, {k.shape[1]} heads, {k.shape[2]} channels, positions {meta.get('positions')}, model {meta.get('model')}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
