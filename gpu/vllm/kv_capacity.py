"""The lossless KV cache's capacity against vLLM's own: the "GPU KV cache size" each logs (tokens) at the flags the design's figure was logged at (Qwen3-8B, eager, bf16 weights,
max-model-len 8704, 8704 tokens a step, prefix caching and chunked prefill off, 0.9), for stock and for kv lossless in each configuration ARENA:POOL (a page's own arena of ARENA
entries, four bytes an entry, its header in steps of 128 bytes, or `header` for what the header leaves; the escape pool's share of the cache, percent), each in a vLLM of its own.

    python kv_capacity.py [--out DIR] [--tables FILE] [--configs header:1.0,header:0.5,191:0] [MODEL]"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_vllm import run  # noqa: E402


def main():
    a = sys.argv[1:]
    out = a[a.index("--out") + 1] if "--out" in a else "kv_capacity_results"
    tables = a[a.index("--tables") + 1] if "--tables" in a else None
    configs = (a[a.index("--configs") + 1] if "--configs" in a else "header:1.0,191:0").split(",")
    model = next((x for i, x in enumerate(a) if not x.startswith("--") and (i == 0 or a[i - 1] not in ("--out", "--tables", "--configs"))), "Qwen/Qwen3-8B")
    os.makedirs(out, exist_ok=True)
    env = {"GLYD_KV_TABLES": tables} if tables else {}
    s = run({"model": model, "plain": True, "eager": True, "r1": True}, out, "stock", {})
    base = s["blocks"] * s["block_size"]
    print(f"stock: KV cache {base:,} tokens ({s['blocks']:,} blocks of {s['block_size']})", flush=True)
    rows = [{"bay": None, "tokens": base}]
    for c in configs:
        bay, store = c.split(":")
        e = dict(env, GLYD_KV_RESERVE_1=store)
        if bay != "header":
            e["GLYD_KV_SPARE"] = bay
        k = run({"model": model, "kv": True, "eager": True, "r1": True}, out, f"kv-{bay}-{store}", e)
        t = k["blocks"] * k["block_size"]
        rows.append({"bay": bay, "store_percent": float(store), "tokens": t, "blocks": k["blocks"]})
        print(f"kv lossless, arena {bay}, escape pool {store}% of the cache: KV cache {t:,} tokens ({k['blocks']:,} blocks): {t / base:.4f}x stock's", flush=True)
    json.dump({"model": model, "stock_blocks": s["blocks"], "rows": rows}, open(os.path.join(out, "capacity.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
