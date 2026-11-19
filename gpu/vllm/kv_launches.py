"""The kernels a decode step launches a layer for attention, vLLM's own KV cache against kv lossless, in vLLM's default mode (compiled, FULL decode graphs) or eager: B requests of L tokens, the
steps driven through the engine one at a time; under torch's profiler the kernels of STEPS decode steps, counted by kind (the plugin's write, attention and combine kernels; FlashAttention's
and its cache write; everything else) and divided by the steps and the layers. A graph's kernels are launched by its replay: the profiler lists them one by one.

    VLLM_ENABLE_V1_MULTIPROCESSING=0 python kv_launches.py MODE OUT.json [--eager] [--rows 8] [--len 1024] [--steps 8] [--tables FILE]     (MODE: stock or kv)"""
import json
import os
import re
import sys

import torch
from vllm import LLM, SamplingParams

mode, out = sys.argv[1], sys.argv[2]
a = sys.argv[3:]
B = int(a[a.index("--rows") + 1]) if "--rows" in a else 8
L = int(a[a.index("--len") + 1]) if "--len" in a else 1024
steps = int(a[a.index("--steps") + 1]) if "--steps" in a else 8
if "--tables" in a:
    os.environ["GLYD_KV_TABLES"] = a[a.index("--tables") + 1]
kw = dict(model="Qwen/Qwen3-8B", dtype="bfloat16", enforce_eager="--eager" in a, max_model_len=4096, gpu_memory_utilization=0.85, seed=0, enable_prefix_caching=False, enable_chunked_prefill=False)
if mode == "kv":
    kw.update(quantization="glyd", additional_config={"glyd": {"fraction": 0, "kv": "lossless"}})
llm = LLM(**kw)
eng = llm.llm_engine
layers = eng.model_config.hf_text_config.num_hidden_layers
vocab = eng.model_config.get_vocab_size() - 200
KINDS = [("plugin write", re.compile(r"kvc.*write_kernel")), ("plugin attention", re.compile(r"kvc.*attn_kernel")), ("plugin combine", re.compile(r"kvc.*combine_kernel")),
         ("flash cache write", re.compile(r"reshape_and_cache", re.I)), ("flash attention (its split-KV kernel and its merge)", re.compile(r"flash|fmha|splitkv|mha_", re.I))]
for i in range(B):
    eng.add_request(f"r{i}", {"prompt_token_ids": [(i * 7919 + j * 104729) % vocab + 100 for j in range(L)]}, SamplingParams(temperature=0, max_tokens=64, ignore_eos=True))
for _ in range(B + 4):  # (the prompts, at most one a step, then decode steps: past the graphs' first replays)
    eng.step()
torch.cuda.synchronize()
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
    for _ in range(steps):
        eng.step()
    torch.cuda.synchronize()
count, total = {}, 0
for e in prof.events():
    if e.device_type == torch.autograd.DeviceType.CUDA:
        total += 1
        k = next((n for n, r in KINDS if r.search(e.name)), None)
        if k:
            count[k] = count.get(k, 0) + 1
res = {"mode": mode, "eager": "--eager" in a, "cudagraph_mode": str(eng.vllm_config.compilation_config.cudagraph_mode), "rows": B, "tokens": L, "layers": layers, "steps": steps,
       "kernels_a_step": round(total / steps, 1), "kernels_a_step_a_layer_by_kind": {k: round(v / steps / layers, 3) for k, v in count.items()}, "attention_launches_a_layer": round(sum(count.values()) / steps / layers, 3)}
print(json.dumps(res), flush=True)
json.dump(res, open(out, "w"), indent=1)
