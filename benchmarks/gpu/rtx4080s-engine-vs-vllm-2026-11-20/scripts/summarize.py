#!/usr/bin/env python3
"""The tables of the README from the load files of run.sh.

    python3 summarize.py ENGINE_GIB VLLM_GIB engine1.json engine2.json -- vllm1.json vllm2.json

Per load the median over the files given for each engine; the hardware ceiling is BW / weight bytes a token (736 GB/s for the RTX 4080 SUPER; the weights the server says it holds,
in GiB), and a decode rate is put as a percentage of it (the KV cache's reads, which grow with the prompt, are not counted in the ceiling).
"""
import json
import statistics
import sys

BW = 736e9
eg, vg = float(sys.argv[1]) * 2**30, float(sys.argv[2]) * 2**30
i = sys.argv.index("--")
E = [json.load(open(p)) for p in sys.argv[3:i]]
V = [json.load(open(p)) for p in sys.argv[i + 1:]]
ce, cv = BW / eg, BW / vg


def cell(runs, k, key):
    xs = [c[key] for r in runs for c in r["cells"] if (c["prompt"], c["conc"]) == k and c.get(key) is not None]
    return statistics.median(xs) if xs else None


def spread(runs, k, key):
    xs = [c[key] for r in runs for c in r["cells"] if (c["prompt"], c["conc"]) == k and c.get(key) is not None]
    return (max(xs) - min(xs)) / statistics.median(xs) * 100 if len(xs) > 1 else 0


print(f"ceiling: engine {eg / 2**30:.2f} GiB of weights = {ce:.1f} tokens/s; vLLM {vg / 2**30:.2f} GiB = {cv:.1f} tokens/s\n")
print("| load | decode tok/s a user: engine (% of ceiling) | vLLM (%) | engine / vLLM | total tok/s: engine | vLLM | first token s: engine | vLLM | engine / vLLM |")
print("| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
order = {"short": 0, "2k": 1, "4k": 2}
keys = sorted({(c["prompt"], c["conc"]) for c in E[0]["cells"]}, key=lambda k: (order[k[0]], k[1]))
for k in keys:
    de, dv = cell(E, k, "decode_tps_per_user"), cell(V, k, "decode_tps_per_user")
    te, tv = cell(E, k, "agg_tps"), cell(V, k, "agg_tps")
    fe, fv = cell(E, k, "ttft_s"), cell(V, k, "ttft_s")
    pt = cell(E, k, "prompt_tokens")
    name = {"short": "short (~35 tokens)", "2k": f"~{round(pt, -2):.0f}-token", "4k": f"~{round(pt, -2):.0f}-token"}[k[0]] + f", {k[1]} at once"
    print(f"| {name} | {de:.1f} ({de / ce * 100:.0f}%) | {dv:.1f} ({dv / cv * 100:.0f}%) | {de / dv:.2f}x | {te:.1f} | {tv:.1f} | {fe:.3f} | {fv:.3f} | {fe / fv:.2f}x |")
print(f"\npeak GPU memory (compute processes, nvidia-smi): engine {[r['peak_mib'] for r in E]} MiB, vLLM {[r['peak_mib'] for r in V]} MiB")
print("run to run spread of decode/user (max - min over median): engine " + ", ".join(f"{spread(E, k, 'decode_tps_per_user'):.1f}%" for k in keys) + "; vLLM " + ", ".join(f"{spread(V, k, 'decode_tps_per_user'):.1f}%" for k in keys))
