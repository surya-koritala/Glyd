#!/usr/bin/env python3
"""The bf16 reference logits for KL divergence (kld_compare.py), Gemma 4 dense, whole model on one GPU (transformers, eager attention, as
research/gemma4-engine/ref_gemma4.py): each chunk of the tokens file read at once, the logits at positions n/2 .. n-2 (they predict tokens
n/2+1 .. n-1) written as float32 rows [chunks * (n/2 - 1), vocab] to OUT, and OUT.json with the vocab and rows per chunk.
  python kld_ref_g4.py MODEL_DIR chunks.json OUT.f32"""
import json, sys, time
import numpy as np, torch
from transformers import AutoModelForCausalLM

model_dir, tokens, out = sys.argv[1:4]
docs = json.load(open(tokens))["docs"]
t0 = time.time()
# GPU_GIB=N: the model split between the GPU (N GiB) and the computer's memory (CPU_GIB), run on the GPU (a 62 GB model on a 40 GB card)
import os
gpu = os.environ.get("GPU_GIB")
kw = dict(device_map="auto", max_memory={0: f"{gpu}GiB", "cpu": f"{os.environ.get('CPU_GIB', '150')}GiB"}) if gpu else dict(device_map="cuda")
# DTYPE=fp32, ATTN=sdpa: the other references (the float32 one the bf16 ones are measured against; transformers' other attention)
dt = torch.float32 if os.environ.get("DTYPE") == "fp32" else torch.bfloat16
model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=dt, attn_implementation=os.environ.get("ATTN", "eager"), **kw).eval()
print(f"loaded {type(model).__name__} in {time.time() - t0:.0f} s", flush=True)
V, per = None, None
with open(out, "wb") as f:
    for c, d in enumerate(docs):
        n = len(d)
        with torch.no_grad():
            lg = model(torch.tensor([d], device="cuda")).logits[0, n // 2:n - 1].float().cpu().numpy()
        V, per = lg.shape[1], lg.shape[0]
        f.write(lg.astype("<f4").tobytes())
        print(f"chunk {c + 1}/{len(docs)} ({time.time() - t0:.0f} s)", flush=True)
json.dump({"vocab": V, "rows_per_chunk": per, "model": model_dir, "chunks": len(docs), "transformers": __import__("transformers").__version__, "torch": torch.__version__}, open(out + ".json", "w"))
