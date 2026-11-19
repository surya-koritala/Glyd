"""Where a decode step's time goes, vLLM's own KV cache against kv lossless (eager, bf16 weights, prefix caching and chunked prefill off, as kv_tps.py runs them): B requests of L tokens in all
(a prompt of L - 128 random tokens, then decode steps), the steps driven through the engine one at a time. Per decode step, from a window of STEPS steps without the profiler (the wall time)
and one under torch's profiler (the kernels' GPU time by kind, and what the GPU sat idle for): attention (FlashAttention's kernels, or the plugin's write, decode and combine kernels) and
the KV write, the GEMMs, the rest; and the costliest kernels.

    VLLM_ENABLE_V1_MULTIPROCESSING=0 python kv_step_profile.py MODE OUT.json [--shapes 32x1024,4x8192] [--steps 20] [--tables FILE] [--graphs]     (MODE: stock or kv)

--graphs: vLLM's default mode (compiled, FULL decode graphs) in place of eager: a step's kernels are then a graph's (the profiler lists them as it does eager's); the wall time has no launches in it."""
import json
import os
import re
import sys
import time

import torch
from vllm import LLM, SamplingParams

mode, out = sys.argv[1], sys.argv[2]
a = sys.argv[3:]
shapes = [tuple(int(x) for x in s.split("x")) for s in (a[a.index("--shapes") + 1] if "--shapes" in a else "32x1024,4x8192").split(",")]
steps = int(a[a.index("--steps") + 1]) if "--steps" in a else 20
if "--tables" in a:
    os.environ["GLYD_KV_TABLES"] = a[a.index("--tables") + 1]
model = "Qwen/Qwen3-8B"
kw = dict(model=model, dtype="bfloat16", enforce_eager="--graphs" not in a, max_model_len=8704, gpu_memory_utilization=0.96, seed=0, max_num_seqs=256, max_num_batched_tokens=8704, enable_prefix_caching=False,
          enable_chunked_prefill=False)
if mode == "kv":
    kw.update(quantization="glyd", additional_config={"glyd": {"fraction": 0, "kv": "lossless"}})
llm = LLM(**kw)
eng = llm.llm_engine
vocab = eng.model_config.get_vocab_size() - 200
KIND = [("kv write", re.compile(r"reshape_and_cache|write_kernel", re.I)), ("attention", re.compile(r"flash|fmha|splitkv|attn_kernel|combine_kernel", re.I)), ("gemm", re.compile(r"gemm|nvjet|xmma|cutlass|cublas|sm8\d_", re.I))]  # (vLLM's own write, reshape_and_cache_flash, is a write: it matched "flash" first until 2026-10-02, so a stock step's attention held its write)


def kind(name):
    return next((k for k, r in KIND if r.search(name)), "other")


n = [0]


def add(tokens, max_tokens):
    n[0] += 1
    rid = f"r{n[0]}"
    eng.add_request(rid, {"prompt_token_ids": [(n[0] * 7919 + j * 104729) % vocab + 100 for j in range(tokens)]}, SamplingParams(temperature=0, max_tokens=max_tokens, ignore_eos=True))
    return rid


res = {"mode": mode, "graphs": "--graphs" in a, "gpu": torch.cuda.get_device_name(), "kv_blocks": eng.vllm_config.cache_config.num_gpu_blocks, "rows": []}
for B, L in shapes:
    ids = [add(L - 128, 400) for _ in range(B)]
    for _ in range(B + 3):  # (the prompts, at most one a step, then a few decode steps)
        eng.step()
    for _ in range(5):
        eng.step()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(steps):
        eng.step()
    torch.cuda.synchronize()
    wall = (time.perf_counter() - t) * 1e3 / steps
    torch.cuda.synchronize()
    t = time.perf_counter()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(steps):
            eng.step()
        torch.cuda.synchronize()
    pwall = (time.perf_counter() - t) * 1e3 / steps
    by, top = {}, {}
    for e in prof.events():
        if e.device_type == torch.autograd.DeviceType.CUDA:
            ms = (e.device_time_total if hasattr(e, "device_time_total") else e.cuda_time_total) / 1e3
            k = kind(e.name)
            by[k] = by.get(k, 0.0) + ms / steps
            top[e.name[:100]] = top.get(e.name[:100], 0.0) + ms / steps
    busy = sum(by.values())
    row = {"B": B, "L": L, "wall_ms": round(wall, 3), "profiled_wall_ms": round(pwall, 3), "gpu_busy_ms": round(busy, 3), "gpu_idle_ms_in_profiled": round(pwall - busy, 3), "by_kind_ms": {k: round(v, 3) for k, v in by.items()},
           "top": [(k, round(v, 3)) for k, v in sorted(top.items(), key=lambda x: -x[1])[:10]]}
    res["rows"].append(row)
    print(json.dumps(row), flush=True)
    eng.abort_request(ids)
    while eng.has_unfinished_requests():
        eng.step()
    json.dump(res, open(out, "w"), indent=1)
