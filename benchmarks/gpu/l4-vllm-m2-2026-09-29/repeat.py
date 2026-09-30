"""One vLLM, the check's 8 prompts (64 greedy tokens, logprobs) generated three times in the same process, prefix
caching off: the same bits each time? argv: glyd|bf16, compiled|eager, out.json."""
import json
import sys

sys.path.insert(0, sys.argv[4])
from check_vllm import PROMPTS  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402

mode, how, out = sys.argv[1], sys.argv[2], sys.argv[3]
llm = LLM(model="Qwen/Qwen3-1.7B", quantization="glyd" if mode == "glyd" else None, dtype="bfloat16", gpu_memory_utilization=0.85, max_model_len=4096, seed=0, enable_prefix_caching=False, enforce_eager=how == "eager")
sp = SamplingParams(temperature=0, max_tokens=64, logprobs=0)
runs = []
for _ in range(3):
    outs = llm.generate(PROMPTS, sp, use_tqdm=False)
    runs.append([[list(o.outputs[0].token_ids), [d[t].logprob for d, t in zip(o.outputs[0].logprobs, o.outputs[0].token_ids)]] for o in outs])
same = [sum(a == b for a, b in zip(runs[0], r)) for r in runs[1:]]
print(f"{mode} {how}, one process: the second and third generate against the first, prompts bit for bit {same[0]} and {same[1]} of 8", flush=True)
json.dump({"mode": mode, "how": how, "same": same, "runs": runs}, open(out, "w"))
