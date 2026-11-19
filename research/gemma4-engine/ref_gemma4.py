"""The bf16 reference for the Gemma 4 engine: transformers' own forward pass (text only, CPU, eager attention), on fixed prompts.
For each prompt: the token ids (the chat template, thinking off), a greedy continuation, and for every position of prompt + continuation
the top-8 next tokens with their log-probabilities (after the final soft-cap, as sampling sees them). The engine's check reads this.
    python ref_gemma4.py MODEL_DIR OUT.json [N_NEW]"""
import json, os, sys, time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_dir, out = sys.argv[1], sys.argv[2]
n_new = int(sys.argv[3]) if len(sys.argv) > 3 else 32
PROMPTS = [
    "What is the capital of France? Answer in one word.",
    "Write a Python function that returns the n-th Fibonacci number.",
    "Explain in two sentences why the sky is blue.",
    "List three prime numbers greater than 100.",
    "Translate to German: The weather is nice today.",
    # a long one, past the 1,024-token sliding window
    "Summarise this in one sentence: " + " ".join(f"Item {i}: the shop on street {i % 37} sells {['apples', 'bread', 'lamps', 'books'][i % 4]}." for i in range(260)),
]
torch.set_num_threads(30)
tok = AutoTokenizer.from_pretrained(model_dir)
t0 = time.time()
# GPU_GIB=N: the model split between the GPU (N GiB) and the CPU's memory (CPU_GIB, default 38), run on the GPU (a model larger than the
# machine's memory: 31B's 62 GB on a 24 GB card and 50 GB of RAM); the same eager attention, the products in cuBLAS's order.
gpu = os.environ.get("GPU_GIB")
kw = dict(device_map="auto", max_memory={0: f"{gpu}GiB", "cpu": f"{os.environ.get('CPU_GIB', '38')}GiB"}) if gpu else {}
model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, attn_implementation="eager", **kw)
dev = "cuda:0" if gpu else "cpu"
model.eval()
print(f"loaded {type(model).__name__} in {time.time() - t0:.0f} s", flush=True)
res = {"model": model_dir, "transformers": __import__("transformers").__version__, "torch": torch.__version__, "prompts": []}
for p in PROMPTS:
    text = tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ids = tok(text, add_special_tokens=False, return_tensors="pt").input_ids.to(dev)
    t0 = time.time()
    with torch.no_grad():
        gen = model.generate(ids, max_new_tokens=n_new, do_sample=False)
        full = gen[0]
        logits = model(full[None]).logits[0].float()
    lp = torch.log_softmax(logits, -1)
    top = torch.topk(lp, 8, -1)
    r = {"prompt": p[:80], "ids": ids[0].tolist(), "continuation": full[ids.shape[1]:].tolist(),
         "text": tok.decode(full[ids.shape[1]:]), "top_ids": top.indices.tolist(), "top_lp": [[round(x, 5) for x in row] for row in top.values.tolist()]}
    res["prompts"].append(r)
    print(f"{len(r['ids'])} prompt tokens, {time.time() - t0:.0f} s: {r['text'][:80]!r}", flush=True)
    json.dump(res, open(out, "w"))
print("done")
