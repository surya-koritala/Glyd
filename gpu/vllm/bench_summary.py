"""bench_serve.sh's results as a table: each request rate, bf16 against Glyd: requests/s, output tokens/s, time to
first token (TTFT, mean and p99), time per output token (TPOT, mean and p99), and each mode's KV cache.

    python bench_summary.py RESULTS_DIR"""
import glob
import json
import os
import re
import sys

R = sys.argv[1]
modes = [m for m in ("bf16", "glyd") if glob.glob(os.path.join(R, f"{m}-rate*.json"))]
rates = sorted({re.search(r"rate(.+)\.json$", p).group(1) for p in glob.glob(os.path.join(R, "*-rate*.json"))}, key=lambda r: float("inf") if r == "inf" else float(r))
for m in modes:
    kv = open(os.path.join(R, f"kv-{m}.txt")).read() if os.path.exists(os.path.join(R, f"kv-{m}.txt")) else ""
    tokens = re.search(r"GPU KV cache size: ([\d,]+) tokens", kv)
    conc = re.search(r"Maximum concurrency for ([\d,]+) tokens per request: ([\d.]+)x", kv)
    load = re.search(r"Model loading took ([\d.]+) GiB", kv)
    print(f"{m}: weights {load.group(1) if load else '?'} GiB, KV cache {tokens.group(1) if tokens else '?'} tokens, max concurrency {conc.group(2) + 'x at ' + conc.group(1) + ' tokens' if conc else '?'}")
print()
print("| Rate (req/s) | Mode | Requests/s | Output tokens/s | TTFT mean / p99 (ms) | TPOT mean / p99 (ms) | Completed |")
print("| :--- | :--- | ---: | ---: | ---: | ---: | ---: |")
for rate in rates:
    for m in modes:
        p = os.path.join(R, f"{m}-rate{rate}.json")
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        f = lambda k: d.get(k, float("nan"))
        print(f"| {rate} | {m} | {f('request_throughput'):.2f} | {f('output_throughput'):.1f} | {f('mean_ttft_ms'):.0f} / {f('p99_ttft_ms'):.0f} | {f('mean_tpot_ms'):.1f} / {f('p99_tpot_ms'):.1f} | {d.get('completed', '?')} |")
