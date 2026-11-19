"""Matrices of a Hugging Face checkpoint into one safetensors file, each read by its byte range (no shard downloaded whole) and those
given together joined along their rows, as vLLM joins q, k, v and gate, up.

    python fetch_tensors.py REPO OUT.safetensors NAME=TENSOR[+TENSOR...] ...
    python fetch_tensors.py Qwen/Qwen3.5-9B q35.safetensors gate_up=model.language_model.layers.3.mlp.gate_proj.weight+model.language_model.layers.3.mlp.up_proj.weight

A public repo (no token is sent). Each tensor's bytes are checked against its header's size."""
import json
import struct
import sys
import time

import requests
import torch
from safetensors.torch import save_file

DT = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}


def get(url, lo, hi):
    """Bytes lo..hi (inclusive) of url, with retries."""
    for i in range(5):
        try:
            r = requests.get(url, headers={"Range": f"bytes={lo}-{hi}"}, timeout=120)
            if r.status_code == 206 and len(r.content) == hi - lo + 1:
                return r.content
            err = f"HTTP {r.status_code}, {len(r.content)} bytes"
        except requests.RequestException as e:
            err = repr(e)
        time.sleep(2 + 3 * i)
    raise RuntimeError(f"{url} bytes {lo}-{hi}: {err}")


def main():
    repo, out, specs = sys.argv[1], sys.argv[2], sys.argv[3:]
    base = f"https://huggingface.co/{repo}/resolve/main/"
    r = requests.get(base + "model.safetensors.index.json", timeout=60)
    wm = r.json()["weight_map"] if r.status_code == 200 else None  # (none: one file, model.safetensors)
    heads = {}

    def tensor(name):
        f = wm[name] if wm else "model.safetensors"
        if f not in heads:
            n = struct.unpack("<Q", get(base + f, 0, 7))[0]
            heads[f] = (8 + n, json.loads(get(base + f, 8, 8 + n - 1)))
        start, h = heads[f]
        m = h[name]
        a, b = m["data_offsets"]
        t = torch.frombuffer(bytearray(get(base + f, start + a, start + b - 1)), dtype=DT[m["dtype"]]).reshape(m["shape"])
        return t

    got = {}
    for s in specs:
        key, _, names = s.partition("=")
        parts = [tensor(n) for n in names.split("+")]
        got[key] = torch.cat(parts) if len(parts) > 1 else parts[0]
        print(f"{repo} {key}: {list(got[key].shape)} {got[key].dtype} from {len(parts)} tensor(s)", flush=True)
    save_file({k: v.contiguous() for k, v in got.items()}, out)


if __name__ == "__main__":
    main()
