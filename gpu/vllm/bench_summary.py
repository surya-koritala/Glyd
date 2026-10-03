"""bench_serve.sh's results as a table: each request rate, bf16 against Glyd: requests/s, output tokens/s, time to
first token (TTFT, mean and p99), time per output token (TPOT, mean and p99), the GPU's median SM clock while it
worked and its hottest (nvidia-smi's each second, where logged), and each mode's KV cache. Where modes glyd@F ran
(Glyd packing the fraction F of the layers), a second table of each rate's weights, KV cache, requests/s, TTFT and TPOT
against bf16's. Each server's highest prefix cache hit rate is printed (its log's, every 10 s); one above 1% flags the
run NOT COMPARABLE, since the random prompts are new to a server and a hit is a prefill that mode did not do.

    python bench_summary.py RESULTS_DIR"""
import glob
import json
import os
import re
import sys

R = sys.argv[1]
modes = sorted({os.path.basename(p).rsplit("-rate", 1)[0] for p in glob.glob(os.path.join(R, "*-rate*.json"))}, key=lambda m: (m != "bf16", m != "glyd", float(m.split("@")[1]) if "@" in m else 0.0))  # (bf16, glyd, glyd@F by F)
rates = sorted({re.search(r"rate(.+)\.json$", p).group(1) for p in glob.glob(os.path.join(R, "*-rate*.json"))}, key=lambda r: float("inf") if r == "inf" else float(r))
def kvnum(m, suffix=""):
    """A server's weights (GiB), KV cache (tokens) and max concurrency (its tokens, the times) from its log's lines
    (kv-MODE[-cold].txt), or None."""
    p = os.path.join(R, f"kv-{m}{suffix}.txt")
    if not os.path.exists(p):
        return None
    t = open(p).read()
    tokens = re.search(r"GPU KV cache size: ([\d,]+) tokens", t)
    conc = re.search(r"Maximum concurrency for ([\d,]+) tokens per request: ([\d.]+)x", t)
    load = re.search(r"Model loading took ([\d.]+) GiB", t)
    return (float(load.group(1)) if load else None, int(tokens.group(1).replace(",", "")) if tokens else None, (conc.group(1), conc.group(2)) if conc else None)


def kv(m, suffix=""):
    """kvnum as a line, or None."""
    k = kvnum(m, suffix)
    if k is None:
        return None
    load, tokens, conc = k
    return f"weights {load if load is not None else '?'} GiB, KV cache {f'{tokens:,}' if tokens is not None else '?'} tokens, max concurrency {conc[1] + 'x at ' + conc[0] + ' tokens' if conc else '?'}"


def hit(m):
    """The highest "Prefix cache hit rate" (%) in mode m's server log (serve-MODE.txt), or None."""
    try:
        rates = re.findall(r"Prefix cache hit rate: ([\d.]+)%", open(os.path.join(R, f"serve-{m}.txt"), errors="replace").read())
    except OSError:
        return None
    return max(map(float, rates)) if rates else None


for m in modes:
    cold = kv(m, "-cold")
    print(f"{m}: {kv(m) or 'no server'}" + (f" (started cold, on an empty compile cache: {cold})" if cold else ""))
hits = {m: hit(m) for m in modes}
print("Prefix cache hit rate, at most (the servers' logs): " + ", ".join(f"{m} {'n/a' if h is None else f'{h:.1f}%'}" for m, h in hits.items()))
over = [f"{m} {h:.1f}%" for m, h in hits.items() if h is not None and h > 1]
warn = f"NOT COMPARABLE: the prefix cache skipped the prefill of repeated prompts (hit rate above 1%: {', '.join(over)})" if over else ""
if warn:
    print(warn)
print()
print("| Rate (req/s) | Mode | Requests/s | Output tokens/s | TTFT mean / p99 (ms) | TPOT mean / p99 (ms) | Completed | SM clock, temperature |")
print("| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |")


def loaded(rows):
    """nvidia-smi's samples (timestamp, temperature, SM clock, power) while the GPU worked: above the midpoint of the
    least and the most power drawn, where the most is over 1.5x the least (the idle seconds before the requests came,
    between them and after, left out), else all of them."""
    w = [float(r[3]) for r in rows]
    lo, hi = min(w), max(w)
    return [r for r, x in zip(rows, w) if hi <= 1.5 * lo or x > (lo + hi) / 2]


def smi(p):
    """The median SM clock (MHz) over the GPU's loaded samples, and the hottest (C)."""
    try:
        rows = [r.split(", ") for r in open(p).read().splitlines() if r.count(",") == 3]
        clocks = sorted(float(r[2]) for r in loaded(rows))
        return f"{clocks[len(clocks) // 2]:.0f} MHz, {max(float(r[1]) for r in rows):.0f} C"
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


def against_bf16():
    """Where modes glyd@F ran: each rate's modes' weights, KV cache, requests/s, TTFT and TPOT (means), each beside
    bf16's as a ratio (for the times, less is better)."""
    if "bf16" not in modes or not any("@" in m for m in modes):
        return
    print()
    print("| Rate (req/s) | Mode | Weights (GiB) | KV cache (tokens) | Requests/s | TTFT mean (ms) | TPOT mean (ms) |")
    print("| :--- | :--- | ---: | ---: | ---: | ---: | ---: |")
    ref_kv = kvnum("bf16")
    for rate in rates:
        base = None
        for m in modes:
            p = os.path.join(R, f"{m}-rate{rate}.json")
            if not os.path.exists(p):
                continue
            d = json.load(open(p))
            if m == "bf16":
                base = d
            k = kvnum(m)
            x = lambda v, r, fmt: "" if v is None else fmt.format(v) + (f" ({v / r:.2f}x)" if r and m != "bf16" else "")
            print(f"| {rate} | {m} | {x(k and k[0], ref_kv and ref_kv[0], '{:.2f}')} | {x(k and k[1], ref_kv and ref_kv[1], '{:,}')} | "
                  + " | ".join(x(d.get(key), base and base.get(key), fmt) for key, fmt in (("request_throughput", "{:.2f}"), ("mean_ttft_ms", "{:,.0f}"), ("mean_tpot_ms", "{:.1f}"))) + " |")


against_bf16()
if warn:
    print()
    print(warn)
