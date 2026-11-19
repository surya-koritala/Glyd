#!/usr/bin/env python3
"""The routed experts' inputs on the calibration documents, from the bf16 reference (research/qwen-data/qwen_ref.py, layer at a time, the GPU
lock per layer): for every decoder layer the MoE input of each token (bf16), its 10 experts and their routing weights, one file per layer and batch.
  QWEN_SNAP=.../flash-next python calib_capture.py --tokens calib.json --out DIR      (DIR/L{l:02d}_b{b}.pt: {"x", "idx", "w"})"""
import argparse, json, os, sys, time
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "qwen-data"))
import qwen_ref as R

ap = argparse.ArgumentParser(); ap.add_argument("--tokens", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()
docs = json.load(open(a.tokens))["docs"]
B = int(os.environ.get("QWEN_BATCH", "8"))
os.makedirs(a.out, exist_ok=True)
ck = R.Ckpt(R.SNAP)
model, tc = R.build(ck)
state = {"b": 0}
for l, layer in enumerate(model.layers):
    def grab(m, args, l=l):
        x, idx, w = args[0], args[1], args[2]
        torch.save({"x": x.detach().to(torch.bfloat16).cpu(), "idx": idx.detach().to(torch.int16).cpu(), "w": w.detach().float().cpu()}, f"{a.out}/L{l:02d}_b{state['b']}.pt")
    layer.mlp.experts.register_forward_pre_hook(grab)
t0 = time.time()
with torch.no_grad():
    for b in range(0, len(docs), B):
        state["b"] = b // B
        model(input_ids=torch.tensor(docs[b:b + B], dtype=torch.long).to(R.DEV), use_cache=False)
        print(f"batch {b // B}: {min(b + B, len(docs))}/{len(docs)} documents, {time.time() - t0:.0f} s", flush=True)
json.dump({"docs": len(docs), "tokens": sum(map(len, docs)), "batches": (len(docs) + B - 1) // B, "seconds": round(time.time() - t0)}, open(f"{a.out}/meta.json", "w"))
