#!/usr/bin/env python3
"""The speed bar of glyd-engine/src/machine.rs (GPUS), applied to two load files made by benchmarks/gpu/*/scripts/load.py (or run.sh): the engine's and vLLM's.

    python3 scripts/gpu_bar.py ENGINE.json VLLM.json "NVIDIA GeForce RTX 4080 SUPER" 8.9

A GPU passes where, at every load measured (every prompt length and number of users), the engine's decode tokens/s a user and in all, and its time to first token, are at least
vLLM's (within NOISE, 3%, the spread of two runs of one engine), and its peak GPU memory is lower. A passing GPU prints its line for GPUS; a failing one prints each load it
failed, by how much, and no line. Exit 0 passes, 1 fails. docs/adding-a-gpu.md.
"""
import json
import re
import sys

NOISE = 0.03


def cells(path):
    d = json.load(open(path))
    return d, {(c["prompt"], c["conc"]): c for c in d["cells"]}


def word(name):
    # the words of the driver's name for the GPU that tell it from its neighbours (machine.rs matches them as whole words):
    # "NVIDIA GeForce RTX 4080 SUPER" -> "RTX 4080 SUPER", "NVIDIA L4" -> "L4", "NVIDIA A100-SXM4-80GB" -> "A100"
    n = re.sub(r"^NVIDIA\s+(GeForce\s+)?", "", name).strip()
    if re.match(r"RTX\s+\d", n):
        return n
    m = re.match(r"([A-Za-z]+\d+)", n)
    return m.group(1) if m else n


def main():
    if len(sys.argv) != 5:
        sys.exit(__doc__)
    (e, ec), (v, vc) = cells(sys.argv[1]), cells(sys.argv[2])
    name, cc = sys.argv[3], sys.argv[4]
    fails = []
    print(f"{'load':<10}{'':<4}{'decode/user':>22}{'decode total':>22}{'first token (s)':>22}")
    for k in sorted(set(ec) & set(vc), key=lambda k: (["short", "2k", "4k"].index(k[0]) if k[0] in ("short", "2k", "4k") else 9, k[1])):
        a, b = ec[k], vc[k]
        row = f"{k[0]:<6} x{k[1]:<5}"
        for key, higher in (("decode_tps_per_user", True), ("agg_tps", True), ("ttft_s", False)):
            x, y = a.get(key), b.get(key)
            if x is None or y is None:
                row += f"{'n/a':>22}"
                fails.append(f"{k[0]} x{k[1]}: {key} missing")
                continue
            ratio = x / y if higher else y / x  # (above 1: the engine is better)
            ok = ratio >= 1 - NOISE
            row += f"{x:>10.3f} /{y:>8.3f} {ratio:5.2f}"
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
    major, minor = cc.split(".")
    print(f'\nPASSES the bar. For GPUS in glyd-engine/src/machine.rs:\n    ("{word(name)}", ({major}, {minor})),')


main()
