"""One user's decode speed on vLLM 0.30 in its default mode (compiled, FULL decode graphs), the shape of the engine's `bench`: a prompt of N
random token ids (the same generator as the engine's), then NEW tokens, prefix caching off; the time per decoded token is
(a generation of NEW tokens - a generation of 1) / (NEW - 1), the median of REPS.

    python ref_speed.py MODEL OUT.json MODE [N=1024] [NEW=256] [REPS=5]      MODE: bf16 | glyd1 | kvglyd
REF_KW='{"language_model_only": true, ...}': more of vLLM's arguments (JSON), over the defaults. OUT.json also has the GPU memory torch holds after the runs (vLLM's
weights, cache and buffers in this process: VLLM_ENABLE_V1_MULTIPROCESSING=0) and vLLM's own count of its weights' bytes.
"""
import json
import os
import statistics
import sys
import time


def random_prompt(n, vocab):  # (report.rs's random_prompt)
    s, out = 0x9E3779B97F4A7C15, []
    m = (1 << 64) - 1
    for _ in range(n):
        s ^= (s << 13) & m
        s ^= s >> 7
        s ^= (s << 17) & m
        out.append(s % vocab)
    return out


def main(model, out, mode, n=1024, new=256, reps=5):
    n, new, reps = int(n), int(new), int(reps)
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

    kw = dict(model=model, dtype="bfloat16", max_model_len=4096, gpu_memory_utilization=0.9, seed=0, enable_prefix_caching=False)
    if mode != "bf16":
        kw["quantization"] = "glyd"
    kw.update(json.loads(os.environ.get("REF_KW", "{}")))
    t0 = time.time()
    llm = LLM(**kw)
    load_s = time.time() - t0
    vocab = llm.llm_engine.model_config.hf_text_config.vocab_size  # (a model with a vision tower keeps it in its text config)
    prompt = [TokensPrompt(prompt_token_ids=random_prompt(n, vocab))]
    full = SamplingParams(temperature=0, max_tokens=new, ignore_eos=True)
    one = SamplingParams(temperature=0, max_tokens=1, ignore_eos=True)
    llm.generate(prompt, full, use_tqdm=False)  # (warm-up)
    ms, pre = [], []
    for _ in range(reps):
        t = time.time()
        llm.generate(prompt, one, use_tqdm=False)
        tp = time.time() - t
        t = time.time()
        llm.generate(prompt, full, use_tqdm=False)
        tt = time.time() - t
        pre.append(tp * 1e3)
        ms.append((tt - tp) / (new - 1) * 1e3)
    r = {"model": model, "mode": mode, "prompt_tokens": n, "new_tokens": new, "load_s": round(load_s, 1), "decode_ms": [round(x, 3) for x in ms], "decode_ms_median": round(statistics.median(ms), 3),
         "decode_tokens_per_s": round(1e3 / statistics.median(ms), 2), "prefill_ms": [round(x, 1) for x in pre], "prefill_ms_median": round(statistics.median(pre), 1)}
    import torch

    r["torch_reserved_gib"] = round(torch.cuda.memory_reserved() / 2**30, 3)
    r["torch_max_allocated_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
    try:  # (vLLM's count of the weights' bytes, where this version keeps it there)
        r["model_memory_gib"] = round(llm.llm_engine.model_executor.driver_worker.worker.model_runner.model_memory_usage / 2**30, 3)
    except Exception:
        pass
    with open(out, "w") as f:
        json.dump(r, f)
    print(json.dumps(r), flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:7])
