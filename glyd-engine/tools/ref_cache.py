"""The KV cache vLLM 0.30 holds at 0.9 of the GPU for a model, at a given tokens a step: the engine's memory proof compares at both its own chunk and vLLM's default.

    python ref_cache.py MODEL MODE MAX_NUM_BATCHED_TOKENS [MAX_MODEL_LEN=4096]       MODE: bf16 | glyd1 | kvglyd      (prints one JSON line)
"""
import json
import os
import sys


def main(model, mode, batched, mml=4096):
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    if mode == "glyd1":
        os.environ.update(GLYD_FRACTION="1", GLYD_KV="off")
    elif mode == "kvglyd":
        os.environ.update(GLYD_FRACTION="1", GLYD_KV="lossless")
    elif mode != "bf16":
        raise SystemExit(f"mode {mode}")
    from vllm import LLM

    kw = dict(model=model, dtype="bfloat16", max_model_len=int(mml), gpu_memory_utilization=0.9, seed=0, max_num_batched_tokens=int(batched), enable_prefix_caching=False)
    if mode != "bf16":
        kw["quantization"] = "glyd"
    llm = LLM(**kw)
    cc = llm.llm_engine.vllm_config.cache_config
    print(json.dumps({"model": model, "mode": mode, "max_num_batched_tokens": int(batched), "max_model_len": int(mml), "blocks": cc.num_gpu_blocks, "block_size": cc.block_size, "kv_tokens": cc.num_gpu_blocks * cc.block_size}), flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:5])
