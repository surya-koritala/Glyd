# M1 spike: vLLM's LLM with --quantization glyd (the plugin from its entry point) against bf16 on the same prompts,
# greedy: tokens, the KV cache's size, and time, with CUDA graphs (vLLM's default: FULL_AND_PIECEWISE) or eager.
#   python spike.py MODEL bf16|glyd [--eager] [--layout mma|mma12] OUT.json
import json, os, sys, time
import torch

model, mode, out = sys.argv[1], sys.argv[2], sys.argv[-1]
eager = "--eager" in sys.argv
if "--layout" in sys.argv:
    os.environ["GLYD_LAYOUT"] = sys.argv[sys.argv.index("--layout") + 1]
from vllm import LLM, SamplingParams

prompts = [
    "The history of data compression began",
    "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
    "Q: A train leaves at 3:40 pm and the trip takes 2 hours 35 minutes. When does it arrive?\nA:",
    "The capital of Australia is",
    "Translate to French: The weather is lovely today, so we will walk to the market.",
    "In quantum mechanics, the uncertainty principle states that",
    "SELECT name, COUNT(*) FROM orders JOIN customers ON",
    "Once upon a time, in a village at the edge of a great forest,",
]
t0 = time.perf_counter()
llm = LLM(model=model, quantization=None if mode == "bf16" else "glyd", dtype="bfloat16", gpu_memory_utilization=0.85,
          max_model_len=4096, enforce_eager=eager, seed=0)
load = time.perf_counter() - t0
cfg = llm.llm_engine.vllm_config
res = {"model": model, "mode": mode, "eager": eager, "layout": os.environ.get("GLYD_LAYOUT"), "load_s": round(load, 1),
       "num_gpu_blocks": cfg.cache_config.num_gpu_blocks, "block_size": cfg.cache_config.block_size,
       "cudagraph_mode": str(cfg.compilation_config.cudagraph_mode), "torch": torch.__version__}
greedy = SamplingParams(temperature=0, max_tokens=64, logprobs=1)
outs = llm.generate(prompts, greedy)
res["tokens"] = [list(o.outputs[0].token_ids) for o in outs]
res["logprobs"] = [[float(next(iter(d.values())).logprob) for d in o.outputs[0].logprobs] for o in outs]
res["text0"] = outs[0].outputs[0].text[:200]
# one sequence: decode tokens/s (256 tokens), then 8 and 32 sequences
for n in (1, 8, 32):
    sp = SamplingParams(temperature=0, max_tokens=256, ignore_eos=True)
    llm.generate(prompts[:1] * n, sp)  # warm
    t = time.perf_counter()
    llm.generate(prompts[:1] * n, sp)
    res[f"tokens_per_s_{n}"] = round(n * 256 / (time.perf_counter() - t), 1)
res["max_allocated_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2) if torch.cuda.is_initialized() else None
json.dump(res, open(out, "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k not in ("tokens", "logprobs")}, indent=1))
