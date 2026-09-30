"""bench_serve.sh's results as a table: each request rate, bf16 against Glyd: requests/s, output tokens/s, time to
first token (TTFT, mean and p99), time per output token (TPOT, mean and p99), the GPU's median SM clock and its
hottest (nvidia-smi's each second, where logged), and each mode's KV cache.

    python bench_summary.py RESULTS_DIR"""
import glob
import json
import os
import re
import sys

R = sys.argv[1]
modes = [m for m in ("bf16", "glyd") if glob.glob(os.path.join(R, f"{m}-rate*.json"))]
rates = sorted({re.search(r"rate(.+)\.json$", p).group(1) for p in glob.glob(os.path.join(R, "*-rate*.json"))}, key=lambda r: float("inf") if r == "inf" else float(r))
def kv(m, suffix=""):
    """A server's weights, KV cache and max concurrency from its log's lines (kv-MODE[-cold].txt), or None."""
    p = os.path.join(R, f"kv-{m}{suffix}.txt")
    if not os.path.exists(p):
        return None
    t = open(p).read()
    tokens = re.search(r"GPU KV cache size: ([\d,]+) tokens", t)
    conc = re.search(r"Maximum concurrency for ([\d,]+) tokens per request: ([\d.]+)x", t)
    load = re.search(r"Model loading took ([\d.]+) GiB", t)
    return f"weights {load.group(1) if load else '?'} GiB, KV cache {tokens.group(1) if tokens else '?'} tokens, max concurrency {conc.group(2) + 'x at ' + conc.group(1) + ' tokens' if conc else '?'}"


for m in modes:
    cold = kv(m, "-cold")
    print(f"{m}: {kv(m) or 'no server'}" + (f" (started cold, on an empty compile cache: {cold})" if cold else ""))
print()
print("| Rate (req/s) | Mode | Requests/s | Output tokens/s | TTFT mean / p99 (ms) | TPOT mean / p99 (ms) | Completed | SM clock, temperature |")
print("| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |")


def smi(p):
    """nvidia-smi's samples (timestamp, temperature, SM clock, power): the median clock (MHz) and the hottest (C)."""
    try:
        rows = [r.split(", ") for r in open(p).read().splitlines() if r.count(",") == 3]
        clocks, temps = sorted(float(r[2]) for r in rows), [float(r[1]) for r in rows]
        return f"{clocks[len(clocks) // 2]:.0f} MHz, {max(temps):.0f} C"
    except (OSError, ValueError, IndexError):
        return ""


for rate in rates:
    for m in modes:
        p = os.path.join(R, f"{m}-rate{rate}.json")
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        f = lambda k: d.get(k, float("nan"))
        gpu = smi(os.path.join(R, f"smi-{m}-rate{rate}.csv"))
        print(f"| {rate} | {m} | {f('request_throughput'):.2f} | {f('output_throughput'):.1f} | {f('mean_ttft_ms'):.0f} / {f('p99_ttft_ms'):.0f} | {f('mean_tpot_ms'):.1f} / {f('p99_tpot_ms'):.1f} | {d.get('completed', '?')} | {gpu} |")
