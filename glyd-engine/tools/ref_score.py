"""vLLM's teacher-forced logprobs over the first SCORE tokens of a text at a given tokens a step (chunked prefill where the step is smaller than the text):
how much its own answer moves with the chunking, the control for the engine's.

    python ref_score.py MODEL MODE MAX_NUM_BATCHED_TOKENS TEXTFILE OUT.json      MODE: bf16 | glyd1 | kvglyd
"""
import json
import os
import sys

SCORE = 1536


def main(model, mode, batched, textfile, out):
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    if mode == "glyd1":
        os.environ.update(GLYD_FRACTION="1", GLYD_KV="off")
    elif mode == "kvglyd":
        os.environ.update(GLYD_FRACTION="1", GLYD_KV="lossless")
    elif mode != "bf16":
        raise SystemExit(f"mode {mode}")
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    kw = dict(model=model, dtype="bfloat16", max_model_len=4096, gpu_memory_utilization=0.9, seed=0, max_num_batched_tokens=int(batched), enable_prefix_caching=False)
    if mode != "bf16":
        kw["quantization"] = "glyd"
    llm = LLM(**kw)
    tok = llm.get_tokenizer()
    ids = tok.encode(open(textfile).read())[:SCORE]
    o = llm.generate([TokensPrompt(prompt_token_ids=ids)], SamplingParams(temperature=0, max_tokens=1, prompt_logprobs=1), use_tqdm=False)[0]
    pl = o.prompt_logprobs[1:]
    r = {"model": model, "mode": mode, "max_num_batched_tokens": int(batched), "score_ids": ids, "score_lp": [d[x].logprob for d, x in zip(pl, ids[1:])], "score_top1": [next(k for k, v in d.items() if v.rank == 1) for d in pl]}
    json.dump(r, open(out, "w"))
    print(out, "position 1500:", r["score_lp"][1500], flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:6])
