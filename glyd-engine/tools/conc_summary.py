#!/usr/bin/env python3
"""A table of the passes engine_conc_bench.sh made: python conc_summary.py DIR [DIR...] (each DIR one run, with MODE-cN.json, hit-MODE-cN.txt)."""
import json
import os
import re
import sys

rows = []
for d in sys.argv[1:]:
    for f in sorted(os.listdir(d)):
        m = re.fullmatch(r"(engine|kvglyd|bf16)-c(\d+)\.json", f)
        if not m:
            continue
        j = json.load(open(os.path.join(d, f)))
        hf = os.path.join(d, f"hit-{m[1]}-c{m[2]}.txt")
        hit = float(open(hf).read().strip() or 0) if os.path.exists(hf) else float("nan")
        rows.append((os.path.basename(d.rstrip("/")), m[1], int(m[2]), j, hit))
print(f"{'run':<12}{'server':<8}{'users':>5}{'done':>6}{'tok/s':>8}{'per user':>9}{'TTFT ms':>9}{'p99':>8}{'TPOT ms':>9}{'p99':>8}{'hit %':>7}")
for run, mode, c, j, hit in sorted(rows, key=lambda r: (r[2], r[1], r[0])):
    tp = j["mean_tpot_ms"]
    flag = "  NOT COMPARABLE" if hit > 1 else ""
    print(f"{run:<12}{mode:<8}{c:>5}{j['completed']:>6}{j['output_throughput']:>8.1f}{1000 / tp:>9.1f}{j['mean_ttft_ms']:>9.0f}{j['p99_ttft_ms']:>8.0f}{tp:>9.1f}{j['p99_tpot_ms']:>8.1f}{hit:>7.2f}{flag}")
