"""One user's prompt time on vLLM 0.30 in its default mode (compiled, FULL decode graphs), the shapes of the engine's `bench --prompt`: a prompt of N random token ids (the engine's generator),
one token generated, prefix caching off; the time of the generation is the prompt's, the median of REPS after a warm-up. Several N in one load.

    python ref_prefill.py MODEL OUT.json MODE [N,N,...=1024,3500] [REPS=7]      MODE: bf16 | glyd1 | kvglyd
"""
import json
import os
import statistics
import sys
import time

from ref_speed import random_prompt


def main(model, out, mode, ns="1024,3500", reps=7):
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

    reps = int(reps)
    kw = dict(model=model, dtype="bfloat16", max_model_len=4096, gpu_memory_utilization=0.9, seed=0, enable_prefix_caching=False)
    if mode != "bf16":
        kw["quantization"] = "glyd"
    t0 = time.time()
    llm = LLM(**kw)
    load_s = time.time() - t0
    vocab = llm.llm_engine.model_config.hf_config.vocab_size
    one = SamplingParams(temperature=0, max_tokens=1, ignore_eos=True)
    r = {"model": model, "mode": mode, "load_s": round(load_s, 1), "prompts": {}}
    for n in [int(x) for x in ns.split(",")]:
        prompt = [TokensPrompt(prompt_token_ids=random_prompt(n, vocab))]
        llm.generate(prompt, one, use_tqdm=False)  # (warm-up)
        pre = []
        for _ in range(reps):
            t = time.time()
            llm.generate(prompt, one, use_tqdm=False)
            pre.append((time.time() - t) * 1e3)
        r["prompts"][str(n)] = {"prefill_ms": [round(x, 1) for x in pre], "prefill_ms_median": round(statistics.median(pre), 1)}
    with open(out, "w") as f:
        json.dump(r, f)
    print(json.dumps(r), flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:6])
