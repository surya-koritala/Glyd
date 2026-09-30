"""Where a vLLM step's GPU time goes, bf16 against Glyd, on this GPU: the model (default Qwen/Qwen3-8B) as vLLM serves it
(torch.compile, CUDA graphs), steps driven through the engine one at a time, each window under torch's profiler:
- decode steps of B sequences, B in BATCHES (20 steps each, after their prompts' and 5 more);
- prompt steps of M tokens, M in PROMPTS (3 each, a request of M tokens alone).
For each: the kernels' GPU time a step by kind (Glyd's products and decodes, cuBLAS/CUTLASS GEMMs, attention, the
rest), the linear layers' share of it (Glyd's and the GEMMs'), the step's wall time, and its costliest kernels.
    VLLM_ENABLE_V1_MULTIPROCESSING=0 python profile_steps.py MODE OUT.json [MODEL]    (MODE: bf16 or glyd)
Env: BATCHES ("1 8 32 64 128 256"), PROMPTS ("512 2048 8192"), UTIL (0.9). The model's length and a step's tokens
are made to take the longest prompt whole; the JSON is written after each window, so what ran is kept."""
import json
import os
import re
import statistics
import sys
import time

import torch
from vllm import LLM, SamplingParams

mode, out = sys.argv[1], sys.argv[2]
model = sys.argv[3] if len(sys.argv) > 3 else "Qwen/Qwen3-8B"
GLYD = re.compile(r"mma12?_|mma_gemm|mma_unpack|mma_moe|finish_kernel|moe_route|moe_sum")
GEMM = re.compile(r"gemm|nvjet|xmma|cutlass|cublas|sm90_|sm80_|sm89_|sm86_", re.I)
ATTN = re.compile(r"flash|fmha|attention|attn", re.I)


def kind(name):
    return "glyd" if GLYD.search(name) else "gemm" if GEMM.search(name) else "attention" if ATTN.search(name) else "other"


BATCHES = [int(b) for b in os.environ.get("BATCHES", "1 8 32 64 128 256").split()]
PROMPTS = [int(m) for m in os.environ.get("PROMPTS", "512 2048 8192").split()]
longest = max(PROMPTS + [4096])  # (a prompt of M tokens and its one token: one step of M)
llm = LLM(model=model, quantization="glyd" if mode == "glyd" else None, dtype="bfloat16", gpu_memory_utilization=float(os.environ.get("UTIL", "0.9")), max_model_len=longest + 64, max_num_batched_tokens=max(longest, 8192), seed=0, enable_prefix_caching=False)
eng, n = llm.llm_engine, [0]


def add(tokens, max_tokens):
    n[0] += 1
    rid = f"r{n[0]}"
    eng.add_request(rid, {"prompt_token_ids": [(n[0] * 7919 + j * 104729) % 150000 + 100 for j in range(tokens)]}, SamplingParams(temperature=0, max_tokens=max_tokens, ignore_eos=True))
    return rid


def window(steps):
    """steps engine steps under the profiler: {kind: GPU ms a step}, wall ms a step, the costliest kernels."""
    torch.cuda.synchronize()
    t = time.perf_counter()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(steps):
            eng.step()
        torch.cuda.synchronize()
    wall = (time.perf_counter() - t) * 1e3 / steps
    by, top = {"glyd": 0.0, "gemm": 0.0, "attention": 0.0, "other": 0.0}, {}
    for e in prof.events():
        if e.device_type == torch.autograd.DeviceType.CUDA:
            ms = e.device_time_total / 1e3 if hasattr(e, "device_time_total") else e.cuda_time_total / 1e3
            by[kind(e.name)] += ms / steps
            top[e.name[:90]] = top.get(e.name[:90], 0.0) + ms / steps
    return by, wall, sorted(top.items(), key=lambda x: -x[1])[:6]


res = {"mode": mode, "model": model, "gpu": torch.cuda.get_device_name(), "vllm_glyd": (llm.llm_engine.vllm_config.additional_config or {}).get("glyd"), "decode": [], "prompt": []}
for B in BATCHES:
    ids = [add(32, 64) for _ in range(B)]
    for _ in range(6):  # their prompts, then 5 decode steps
        eng.step()
    by, wall, top = window(20)
    eng.abort_request(ids)
    while eng.has_unfinished_requests():
        eng.step()
    res["decode"].append({"M": B, "gpu_ms": by, "wall_ms": wall, "top": top})
    json.dump(res, open(out, "w"), indent=1)
    print(f"{mode} decode B={B}: {by} wall {wall:.2f} ms", flush=True)
for M in PROMPTS:
    runs = []
    for _ in range(3):
        add(M, 1)
        by, wall, top = window(1)
        while eng.has_unfinished_requests():
            eng.step()
        runs.append((sum(by.values()), by, wall, top))
    tot, by, wall, top = sorted(runs, key=lambda r: r[0])[1]  # the median by GPU time
    res["prompt"].append({"M": M, "gpu_ms": by, "wall_ms": wall, "top": top})
    json.dump(res, open(out, "w"), indent=1)
    print(f"{mode} prompt M={M}: {by} wall {wall:.2f} ms", flush=True)
print(f"| Step | M | GPU ms | linear (Glyd + GEMMs) | Glyd | GEMMs | attention | other | wall ms |")
print("| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
for k in ("decode", "prompt"):
    for r in res[k]:
        b, t = r["gpu_ms"], sum(r["gpu_ms"].values())
        lin = b["glyd"] + b["gemm"]
        print(f"| {k} | {r['M']} | {t:.2f} | {lin:.2f} ({lin / t * 100:.0f}%) | {b['glyd']:.2f} | {b['gemm']:.2f} | {b['attention']:.2f} | {b['other']:.2f} | {r['wall_ms']:.2f} |")
