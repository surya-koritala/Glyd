"""A long context through the lossless KV cache, against vLLM's own cache with the same weights: one prompt of N tokens (this repository's documents, repeated where they are fewer), every prompt token's logprob, the first generated
tokens, the time of the prompt and of a decode step at that depth.

    VLLM_ENABLE_V1_MULTIPROCESSING=0 python kv_long.py stock|kv OUT.json [--tokens 36000] [--new 64] [--model Qwen/Qwen3-8B] [--tables FILE] [--fraction 1] [--max-model-len 40960]
    python kv_long.py compare STOCK.json KV.json

Each run is a vLLM of its own (eager, prompts in steps of 1,024 tokens, prefix caching off, a KV cache of the same number of blocks in both: room is left for the logits of the prompt's logprobs); `stock` is `kv: off`
and `kv` is `kv: lossless` with the model's weights packed as `glyd run` serves it (--fraction 1), so that the two differ in the KV cache only. `compare` says how many prompt logprobs are bit for bit, whether the first token's logprob is, how many of the generated tokens are the same and the decode step's time."""
import json
import os
import sys
import time

DOCS = ("README.md", "CHANGELOG.md", "gpu/README.md", "gpu/vllm/README.md", "docs/spec.md", "docs/engineering-notes.md")


def prompt_ids(tok, n):
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, "..", ".."))
    text = "\n\n---\n\n".join(open(os.path.join(root, d), encoding="utf-8").read() for d in DOCS if os.path.exists(os.path.join(root, d)))
    ids = tok.encode(text)
    if len(ids) < n:  # (the documents again, as many times as it takes)
        ids = ids * -(-n // len(ids))
    return ids[:n]


def run(mode, out, a):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    model = a["model"]
    if a.get("tables"):
        os.environ["GLYD_KV_TABLES"] = a["tables"]
    n, new = a["tokens"], a["new"]
    ids = prompt_ids(AutoTokenizer.from_pretrained(model), n)
    mml = a.get("max_model_len") or n + new + 16
    kw = dict(model=model, dtype="bfloat16", enforce_eager=True, gpu_memory_utilization=0.92, seed=0, max_model_len=mml, max_num_seqs=2, max_num_batched_tokens=1024, enable_prefix_caching=False,
              enable_chunked_prefill=True, num_gpu_blocks_override=(n + new + 64) // 16 + 80, quantization="glyd", additional_config={"glyd": {"fraction": a["fraction"], "kv": "lossless" if mode == "kv" else "off"}})
    t0 = time.perf_counter()
    llm = LLM(**kw)
    res = {"mode": mode, "model": model, "tokens": n, "new": new, "max_model_len": mml, "load_s": round(time.perf_counter() - t0, 1)}
    p = TokensPrompt(prompt_token_ids=ids)
    o = llm.generate([p], SamplingParams(max_tokens=new, temperature=0.0, logprobs=0, prompt_logprobs=0, ignore_eos=True), use_tqdm=False)[0]
    res["prompt_logprobs"] = [None if d is None else d[ids[i]].logprob for i, d in enumerate(o.prompt_logprobs)]
    res["tokens_out"] = list(o.outputs[0].token_ids)
    res["logprobs_out"] = [d[t].logprob for d, t in zip(o.outputs[0].logprobs, o.outputs[0].token_ids)]
    # the time of the prompt, and of a decode step at that depth (the prompt alone, then with `new` more tokens)
    t = {}
    for k, m in (("prompt", 1), ("prompt_and_decode", 1 + new)):
        best = None
        for _ in range(2):
            t1 = time.perf_counter()
            llm.generate([p], SamplingParams(max_tokens=m, temperature=0.0, ignore_eos=True), use_tqdm=False)
            d = time.perf_counter() - t1
            best = d if best is None else min(best, d)
        t[k] = best
    res["prompt_s"] = round(t["prompt"], 3)
    res["decode_ms_a_token"] = round((t["prompt_and_decode"] - t["prompt"]) / new * 1000, 3)
    json.dump(res, open(out, "w"))
    print(json.dumps({k: v for k, v in res.items() if k not in ("prompt_logprobs", "tokens_out", "logprobs_out")}), flush=True)


def compare(sp, kp):
    s, k = json.load(open(sp)), json.load(open(kp))
    sl, kl = s["prompt_logprobs"][1:], k["prompt_logprobs"][1:]
    same = sum(x == y for x, y in zip(sl, kl))
    far = max((abs(x - y) for x, y in zip(sl, kl) if x != y), default=0.0)
    gen = sum(x == y for x, y in zip(s["tokens_out"], k["tokens_out"]))
    first = s["logprobs_out"][0] == k["logprobs_out"][0]
    agree = 0
    for x, y in zip(s["tokens_out"], k["tokens_out"]):
        if x != y:
            break
        agree += 1
    print(f"{k['tokens']:,}-token prompt: {same:,} of {len(sl):,} prompt logprobs bit for bit vLLM's (the largest difference {far:.3g}); the first generated token's logprob {'bit for bit' if first else 'not'}; "
          f"{gen} of {len(s['tokens_out'])} generated tokens the same ({agree} from the start)")
    print(f"prompt {s['prompt_s']} s with vLLM's cache and {k['prompt_s']} s with kv; a decode step at that depth {s['decode_ms_a_token']} ms and {k['decode_ms_a_token']} ms ({s['decode_ms_a_token'] / k['decode_ms_a_token']:.3f}x)")


if __name__ == "__main__":
    argv = sys.argv[1:]
    if argv[0] == "compare":
        compare(argv[1], argv[2])
    else:
        opt = lambda f, d, c=str: c(argv[argv.index(f) + 1]) if f in argv else d
        run(argv[0], argv[1], {"tokens": opt("--tokens", 36000, int), "new": opt("--new", 64, int), "model": opt("--model", "Qwen/Qwen3-8B"), "tables": opt("--tables", None),
                               "fraction": opt("--fraction", 1.0, float), "max_model_len": opt("--max-model-len", 0, int)})
