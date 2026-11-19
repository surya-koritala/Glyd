"""The reference of the engine's correctness proof: vLLM 0.30's greedy tokens and logprobs for 16 prompts (the 8 of gpu/vllm/check_vllm.py and 8 more)
(64 tokens each), in the default mode (compiled, FULL decode graphs), as one batch and one prompt at a time (the control: vLLM against itself).

    python ref_vllm.py MODEL OUT.json MODE [TEXTFILE]   MODE: bf16 | glyd1 (Glyd's weights, vLLM's cache) | kvglyd (Glyd's weights, the lossless cache)
                                                        TEXTFILE: its first 1030, 2100 and 3500 tokens are three more, long prompts

OUT.json also holds the prompts' token ids, which `glyd-engine check` reads ({"prompts": ...}).
REF_BATCHED=N sets vLLM's tokens a step (max_num_batched_tokens; default 8192): the control for the long prompts, whose prefill it chunks.
REF_KW='{"language_model_only": true, ...}': more of vLLM's arguments (JSON), over the defaults (a 2026 model with a vision tower on a 24 GB GPU).
"""
import json
import os
import sys
import time

PROMPTS = [
    "The history of data compression began",
    'def fibonacci(n):\n    """Return the n-th Fibonacci number."""\n',
    "Q: A train leaves at 3:40 pm and the trip takes 2 hours 35 minutes. When does it arrive?\nA:",
    "The capital of Australia is",
    "Translate to French: The weather is lovely today, so we will walk to the market.",
    "In quantum mechanics, the uncertainty principle states that",
    "SELECT name, COUNT(*) FROM orders JOIN customers ON",
    "Once upon a time, in a village at the edge of a great forest,",
    "Explain in two sentences why the sky is blue.",
    "import numpy as np\n\ndef softmax(x):\n",
    "The three primary colors of light are",
    "List the planets of the solar system in order from the sun:",
    "Water boils at 100 degrees Celsius at sea level because",
    "A haiku about autumn leaves:\n",
    "The derivative of sin(x) * x^2 is",
    "Dear team, the release planned for Friday",
]
TOKENS = 64
LONG = (1030, 2100, 3500)
SCORE = 1536


def main(model, out, mode):
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if mode == "glyd1":
        os.environ.update(GLYD_FRACTION="1", GLYD_KV="off")
    elif mode == "kvglyd":
        os.environ.update(GLYD_FRACTION="1", GLYD_KV="lossless")
    elif mode != "bf16":
        raise SystemExit(f"mode {mode}")
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    kw = dict(model=model, dtype="bfloat16", max_model_len=4096, gpu_memory_utilization=0.9, seed=0)
    if mode != "bf16":
        kw["quantization"] = "glyd"
    if os.environ.get("REF_BATCHED"):
        kw["max_num_batched_tokens"] = int(os.environ["REF_BATCHED"])
    kw.update(json.loads(os.environ.get("REF_KW", "{}")))
    t0 = time.time()
    llm = LLM(**kw)
    load_s = time.time() - t0
    tok = llm.get_tokenizer()
    ids = [tok.encode(p) for p in PROMPTS]
    sp = SamplingParams(temperature=0, max_tokens=TOKENS, logprobs=0)

    def run(batches):
        outs = [o for b in batches for o in llm.generate([TokensPrompt(prompt_token_ids=i) for i in b], sp, use_tqdm=False)]
        return {
            "tokens": [list(o.outputs[0].token_ids) for o in outs],
            "logprobs": [[d[t].logprob for d, t in zip(o.outputs[0].logprobs, o.outputs[0].token_ids)] for o in outs],
        }

    r = {"model": model, "mode": mode, "prompts": ids, "load_s": round(load_s, 1), "batch": run([ids]), "single": run([[i] for i in ids])}
    if len(sys.argv) > 4:  # long prompts (the first N tokens of a text file, N in LONG): chunked prefills, many pages, one at a time
        text = open(sys.argv[4]).read()
        t = tok.encode(text)
        long_ids = [t[:n] for n in LONG if len(t) >= n]
        r["long_prompts"] = long_ids
        r["long"] = run([[i] for i in long_ids])
        sc = t[:SCORE]  # teacher-forced: vLLM's prompt logprobs over the text's first SCORE tokens, and its largest-logit token at each position
        r["score_ids"] = sc
        o = llm.generate([TokensPrompt(prompt_token_ids=sc)], SamplingParams(temperature=0, max_tokens=1, prompt_logprobs=1), use_tqdm=False)[0]
        pl = o.prompt_logprobs[1:]
        r["score_lp"] = [d[x].logprob for d, x in zip(pl, sc[1:])]
        r["score_top1"] = [next(k for k, v in d.items() if v.rank == 1) for d in pl]
    with open(out, "w") as f:
        json.dump(r, f)
    print(f"{out}: {mode}, {len(ids)} prompts, load {load_s:.0f} s", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:4])
