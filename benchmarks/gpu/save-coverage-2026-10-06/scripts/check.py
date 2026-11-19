"""A glyd save (or, with no glyd.json, a bf16 checkpoint packed as it loads) checked and measured: loaded as saved with verify (every packed tensor decoded and its sha256 checked
against glyd.json, every tensor saved as it is by its own), one greedy chat answer, the weights' GPU memory, and the
save's bytes against the bf16 bytes of the same tensors and of the whole source checkpoint. One JSON line to OUT.

    python check.py SOURCE_DIR SAVE_DIR OUT.json
"""
import glob
import json
import math
import os
import sys
import time

import torch
from transformers import AutoTokenizer

import glyd_gpu as gg

PROMPT = "In one sentence: what does a lossless compressor guarantee?"


def header(path):
    with open(path, "rb") as f:
        return json.loads(f.read(int.from_bytes(f.read(8), "little")))


def tensors(d):
    """{name: (dtype, shape, bytes)} of every tensor in d's safetensors files."""
    out = {}
    for p in glob.glob(os.path.join(d, "*.safetensors")):
        for k, t in header(p).items():
            if k != "__metadata__":
                out[k] = (t["dtype"], t["shape"], t["data_offsets"][1] - t["data_offsets"][0])
    return out


def groups(names):
    """Tensor counts by their name's first two parts (model.language_model, model.visual, mtp.layers ...)."""
    g = {}
    for n in names:
        k = ".".join(n.split(".")[:2])
        g[k] = g.get(k, 0) + 1
    return dict(sorted(g.items()))


def main(src, save, out):
    r = {"save": save}
    t0 = time.time()
    model = gg.from_pretrained(save, device="cuda:0", layout="mma", verify=True, compile=False)
    q = model.config.quantization_config
    torch.cuda.synchronize()
    r.update(verified=q.verified, hashed=q.hashed, unhashed=q.unhashed, load_s=round(time.time() - t0, 1),
             gpu_bytes=torch.cuda.memory_allocated(), gpu_peak_bytes=torch.cuda.max_memory_allocated())
    print(f"{save}: {q.verified} tensors decode to glyd.json's sha256, {q.hashed} saved as they are match theirs", flush=True)

    tok = AutoTokenizer.from_pretrained(save)
    enc = tok.apply_chat_template([{"role": "user", "content": PROMPT}], add_generation_prompt=True, return_tensors="pt", return_dict=True, enable_thinking=False).to("cuda:0")
    t0 = time.time()
    ids = model.generate(**enc, max_new_tokens=64, do_sample=False)
    r.update(prompt=PROMPT, answer=tok.decode(ids[0, enc["input_ids"].shape[1]:], skip_special_tokens=True), gen_s=round(time.time() - t0, 1))
    print("answer:", r["answer"], flush=True)

    r["glyd_gpu"] = os.path.dirname(gg.__file__)
    if not os.path.exists(os.path.join(save, "glyd.json")):  # a bf16 checkpoint: no save to measure
        with open(out, "w") as f:
            json.dump(r, f, indent=1)
        print(json.dumps({k: r[k] for k in ("gpu_bytes", "gpu_peak_bytes")}), flush=True)
        return
    m = json.load(open(os.path.join(save, "glyd.json")))
    st = tensors(save)
    packed = [t for e in m["packs"].values() for t in e["tensors"]]
    r["save_bytes"] = sum(os.path.getsize(p) for p in glob.glob(os.path.join(save, "*.safetensors")))
    r["same_tensors_bf16_bytes"] = sum(math.prod(t["shape"]) * 2 for t in packed) + sum(st[k][2] for k in m["tensors"])
    r["packs"], r["packed_tensors"], r["as_is_tensors"] = len(m["packs"]), len(packed), len(m["tensors"])
    r["as_is_bytes"] = sum(st[k][2] for k in m["tensors"])
    src_t = tensors(src)
    r["source_bytes"] = sum(os.path.getsize(p) for p in glob.glob(os.path.join(src, "*.safetensors")))
    r["source_tensors"], r["source_groups"] = len(src_t), groups(src_t)
    r["save_groups"] = groups([t["name"] for t in packed] + list(m["tensors"]))
    r["save_vs_same_bf16"] = round(1 - r["save_bytes"] / r["same_tensors_bf16_bytes"], 4)
    r["format"], r["glyd"], r["source"] = m["format"], m["glyd"], m["source"]
    with open(out, "w") as f:
        json.dump(r, f, indent=1)
    print(json.dumps({k: r[k] for k in ("save_bytes", "same_tensors_bf16_bytes", "source_bytes", "save_vs_same_bf16", "gpu_bytes")}), flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:4])
