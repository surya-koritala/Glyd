"""What inductor's deterministic mode costs: one vLLM (argv: glyd|bf16, default|deterministic, OUT.json; Qwen3-8B, CUDA
graphs on, an empty compile cache): decode tokens/s at 1, 8 and 32 sequences (256 tokens each, the second of two runs),
and a prompt pass of 8 prompts of 1,024 tokens (max_tokens 1, not graphed), the median of 3."""
import json
import statistics
import sys
import time

from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

mode, how, out = sys.argv[1:4]
cc = {"inductor_compile_config": {"deterministic": True, "combo_kernels": True, "benchmark_combo_kernel": False}} if how == "deterministic" else {}
t0 = time.perf_counter()
llm = LLM(model="Qwen/Qwen3-8B", quantization="glyd" if mode == "glyd" else None, dtype="bfloat16", gpu_memory_utilization=0.85, max_model_len=4096, seed=0, enable_prefix_caching=False, compilation_config=cc)
res = {"mode": mode, "how": how, "start_s": round(time.perf_counter() - t0, 1)}
sp = SamplingParams(temperature=0, max_tokens=256, ignore_eos=True)
for n in (1, 8, 32):
    llm.generate(["The history of data compression began"] * n, sp, use_tqdm=False)
    t = time.perf_counter()
    llm.generate(["The history of data compression began"] * n, sp, use_tqdm=False)
    res[f"tokens_per_s_{n}"] = round(n * 256 / (time.perf_counter() - t), 1)
prompts = [TokensPrompt(prompt_token_ids=[(i * 7919 + j * 104729) % 150000 + 100 for j in range(1024)]) for i in range(8)]
one = SamplingParams(temperature=0, max_tokens=1)
llm.generate(prompts, one, use_tqdm=False)
times = []
for _ in range(3):
    t = time.perf_counter()
    llm.generate(prompts, one, use_tqdm=False)
    times.append(time.perf_counter() - t)
res["prompts_8x1024_s"] = round(statistics.median(times), 3)
json.dump(res, open(out, "w"))
print(json.dumps(res), flush=True)
