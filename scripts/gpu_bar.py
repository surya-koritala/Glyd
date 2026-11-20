#!/usr/bin/env python3
"""The speed bar of glyd-engine/src/machine.rs (GPUS), applied to two measurement files made by benchmarks/gpu/*/scripts/load.py (or run.sh): the engine's and vLLM's.

    python3 scripts/gpu_bar.py ENGINE.json VLLM.json "NVIDIA GeForce RTX 4080 SUPER" 8.9 [FAMILY] [--stale-ok]
    python3 scripts/gpu_bar.py e1.json,e2.json,e3.json,e4.json,e5.json v1.json,...,v5.json "NVIDIA GeForce RTX 4080 SUPER" 8.9 Qwen3   # the means of the runs, and their spread
    python3 scripts/gpu_bar.py --summarize e1.json,e2.json,... [--model ID --released YYYY-MM-DD --family NAME] > engine.json
        # the runs' means and spread in a small file of its own (what a benchmarks/ directory keeps; gpu_bar.py takes it in place of the runs), with the model it was measured on, when that model was released
        # (the Hub's createdAt) and its family (machine.rs: QWEN3, ...)

A (GPU, family) passes where, at every load measured (every prompt length and number of users), the engine's decode tokens/s a user and in all, and its time to first token, are at least vLLM's
(within NOISE, 3%, the spread of two runs of one engine), and its peak GPU memory is lower. A passing one prints its line for GPUS, `("RTX 4080 SUPER", (8, 9), &["Qwen3"]),` (a GPU already there gets the family
added to its line); a failing one prints each load it failed, by how much, and no line. The model measured must have been released within the last 6 months (the owner's rule: the file says when; --stale-ok
passes an older one, and says so, for a family that has no newer model). Exit 0 passes, 1 fails the bar, 2 a model too old. docs/adding-a-gpu.md.
"""
import datetime
import json
import re
import statistics
import sys

NOISE = 0.03


def summarize(runs, meta=None):
    """The runs' means and spreads as one small file's dictionary."""
    info, cs = cells_of(runs)
    return {"summary_of_runs": info["n"], **(meta or {}), "peak_mib": info["peak_mib"], "cells": [{"prompt": k[0], "conc": k[1], **c} for k, c in cs.items()]}


def cells(paths):
    runs = [json.load(open(p)) for p in paths.split(",")]
    if len(runs) == 1 and "summary_of_runs" in runs[0]:
        d = runs[0]
        return {"peak_mib": d["peak_mib"], "n": d["summary_of_runs"], **{k: d[k] for k in ("model", "released", "family") if k in d}}, {(c["prompt"], c["conc"]): c for c in d["cells"]}
    return cells_of(runs)


def cells_of(runs):
    """Per load, the mean over the runs of each figure (and its standard deviation, as a fraction of the mean, under key + "_sd"); the peak memory the largest."""
    out = {}
    for c in runs[0]["cells"]:
        k = (c["prompt"], c["conc"])
        cell = {"errors": sum(r_c.get("errors") or 0 for r in runs for r_c in r["cells"] if (r_c["prompt"], r_c["conc"]) == k)}
        for key in ("decode_tps_per_user", "agg_tps", "ttft_s"):
            xs = [r_c[key] for r in runs for r_c in r["cells"] if (r_c["prompt"], r_c["conc"]) == k and r_c.get(key) is not None]
            if xs:
                cell[key] = statistics.mean(xs)
                cell[key + "_sd"] = (statistics.stdev(xs) / cell[key]) if len(xs) > 1 else 0.0
        out[k] = cell
    return {"peak_mib": max(r["peak_mib"] for r in runs), "n": len(runs)}, out


def word(name):
    # the words of the driver's name for the GPU that tell it from its neighbours (machine.rs matches them as whole words):
    # "NVIDIA GeForce RTX 4080 SUPER" -> "RTX 4080 SUPER", "NVIDIA L4" -> "L4", "NVIDIA A100-SXM4-80GB" -> "A100"
    n = re.sub(r"^NVIDIA\s+(GeForce\s+)?", "", name).strip()
    if re.match(r"RTX\s+\d", n):
        return n
    m = re.match(r"([A-Za-z]+\d+)", n)
    return m.group(1) if m else n


def main():
    args = sys.argv[1:]
    stale_ok = "--stale-ok" in args
    args = [x for x in args if x != "--stale-ok"]
    if args and args[0] == "--summarize":
        meta, rest = {}, args[1:]
        while len(rest) > 1 and rest[-2] in ("--model", "--released", "--family"):
            meta[rest[-2][2:]] = rest[-1]
            rest = rest[:-2]
        json.dump(summarize([json.load(open(p)) for p in rest[0].split(",")], meta), sys.stdout, indent=None, separators=(",", ":"))
        return
    if len(args) not in (4, 5):
        sys.exit(__doc__)
    (e, ec), (v, vc) = cells(args[0]), cells(args[1])
    name, cc = args[2], args[3]
    family = args[4] if len(args) == 5 else e.get("family", "Qwen3")
    if e.get("model"):
        print(f"engine measured on {e['model']} (family {family}), released {e.get('released', 'unknown')}")
    old = False
    if e.get("released"):
        age = (datetime.date.today() - datetime.date.fromisoformat(e["released"])).days
        old = age > 183
        if old:
            print(f"NOTE: {e['model']} was released {age} days ago, more than 6 months: the owner's rule wants a newer model" + (" (--stale-ok: kept, for a family with no newer one)" if stale_ok else ""))
    elif not stale_ok:
        print("NOTE: the file does not say what model it is or when it was released (gpu_bar.py --summarize ... --model ID --released DATE)")
    fails = []
    print(f"{e['n']} runs of the engine, {v['n']} of vLLM: means (the engine's standard deviation as a percent of its mean in brackets)")
    print(f"{'load':<10}{'':<4}{'decode/user':>30}{'decode total':>30}{'first token (s)':>30}")
    for k in sorted(set(ec) & set(vc), key=lambda k: (["short", "2k", "4k"].index(k[0]) if k[0] in ("short", "2k", "4k") else 9, k[1])):
        a, b = ec[k], vc[k]
        row = f"{k[0]:<6} x{k[1]:<5}"
        for key, higher in (("decode_tps_per_user", True), ("agg_tps", True), ("ttft_s", False)):
            x, y = a.get(key), b.get(key)
            if x is None or y is None:
                row += f"{'n/a':>30}"
                fails.append(f"{k[0]} x{k[1]}: {key} missing")
                continue
            ratio = x / y if higher else y / x  # (above 1: the engine is better)
            ok = ratio >= 1 - NOISE
            row += f"{x:>10.3f} /{y:>8.3f} {ratio:5.2f} ({a.get(key + '_sd', 0) * 100:.1f}%)"
            if not ok:
                fails.append(f"{k[0]} x{k[1]}: {key} {x:.3f} against vLLM's {y:.3f} ({(1 - ratio) * 100:.0f}% behind)")
        if a.get("errors") or b.get("errors"):
            fails.append(f"{k[0]} x{k[1]}: requests failed (engine {a.get('errors')}, vLLM {b.get('errors')})")
        print(row)
    pe, pv = e.get("peak_mib"), v.get("peak_mib")
    print(f"peak GPU memory: engine {pe} MiB, vLLM {pv} MiB")
    if pe is None or pv is None or pe >= pv:
        fails.append(f"memory: engine {pe} MiB is not less than vLLM's {pv} MiB")
    if fails:
        print("\nFAILS the bar:\n  " + "\n  ".join(fails))
        sys.exit(1)
    if old and not stale_ok:
        print("\nPasses the bar, but on a model older than 6 months: no line.")
        sys.exit(2)
    major, minor = cc.split(".")
    print(f'\nPASSES the bar for {family} models. For GPUS in glyd-engine/src/machine.rs (a GPU already there: add the family to its line):\n    ("{word(name)}", ({major}, {minor}), &["{family}"]),')


main()
