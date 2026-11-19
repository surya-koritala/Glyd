"""ht_job.sh's results as tables (tiered-speed-2026-10-06's summarize.py, its prompts left out): per model, the layer's products by
tokens a step (each layout's route on this GPU, bf16's cuBLAS), summed as a step through the model's layers takes them, the tiered
layout's shapes (GLYD_GPU_HT_V) and its routes before (gemm); memory against instructions (from the L2 cache against from memory, clk a
step); vllm serve; the kernels' SASS.   python summarize.py RESULTS_DIR"""
import csv
import glob
import json
import os
import statistics
import sys

R = sys.argv[1]
# a step through the model: each matrix's count (layers of each kind), as vLLM runs them (q, k, v and gate, up joined)
MIX = {
    "Qwen3.5-9B": {"gate_up": 32, "down": 32, "qkv": 8, "o": 8, "gdn_in": 24, "gdn_out": 24, "lm_head": 1},
    "Qwen3.5-27B": {"gate_up": 64, "down": 64, "qkv": 16, "o": 16, "gdn_in": 48, "gdn_out": 48},
    "gemma-4-12B-it": {"gate_up": 48, "down": 48, "qkv": 40, "o": 40, "qk_global": 8, "o_global": 8},
    "gemma-4-31B-it": {"gate_up": 60, "down": 60, "qkv": 50, "o": 50, "qk_global": 10, "o_global": 10},
}
SMALL, PROMPTS = (1, 8, 16, 32, 64), ()


def load(m):
    rows = []
    for f in (f"{R}/layouts-{m}.json", f"{R}/layouts-head-{m}.json"):
        if os.path.exists(f):
            rows += json.load(open(f))["rows"]
    return rows


def clock():
    """The median SM clock while the GPU was busy (utilization 50% and over) in the bench, MHz."""
    try:
        v = [float(r[1]) for r in csv.reader(open(f"{R}/smi-bench.csv")) if len(r) > 5 and float(r[5]) >= 50]
        return statistics.median(v) if v else 0
    except (OSError, ValueError):
        return 0


def t(rows, name, M, layout, route="default", variant="base"):
    for r in rows:
        if r["name"] == name and r["M"] == M and r["layout"] == layout and r["route"] == route and r.get("variant", "base") == variant:
            return r
    return None


def main():
    print(open(f"{R}/gpu-short.txt").read().strip() if os.path.exists(f"{R}/gpu-short.txt") else "")
    for f in ("env.txt", "COMMIT"):
        if os.path.exists(f"{R}/{f}"):
            print(open(f"{R}/{f}").read().strip())
    mhz = clock()
    print(f"median SM clock under load: {mhz:.0f} MHz\n")
    if os.path.exists(f"{R}/steps.txt"):
        print("######## steps\n" + open(f"{R}/steps.txt").read())
    for m, mix in MIX.items():
        rows = load(m)
        if not rows:
            continue
        g = rows[0]
        print(f"######## {m}: GPU time of a step's products, us (each layout by its route on this GPU), and tiered over 12-bit")
        names = [n for n in mix if any(r["name"] == n for r in rows)]
        print(f"{'matrix':10s} {'O x K':>13s} {'bits t/12':>11s} {'flag%':>6s} | " + " | ".join(f"{M:>4d}: mma / mma12 / bf16 = t/12" for M in SMALL))
        for n in names:
            r0 = next(r for r in rows if r["name"] == n)
            b = {l: next((r["bits"] for r in rows if r["name"] == n and r["layout"] == l), 0) for l in ("mma", "mma12")}
            fl = next((r["flagged"] for r in rows if r["name"] == n and r["layout"] == "mma"), 0)
            cells = []
            for M in SMALL:
                a, c = t(rows, n, M, "mma"), t(rows, n, M, "mma12")
                cells.append(f"{a['us']:6.1f} / {c['us']:6.1f} / {a['bf16_us']:6.1f} = {a['us'] / c['us']:4.2f}" if a and c else " " * 33)
            print(f"{n:10s} {r0['O']:6d}x{r0['K']:<6d} {b['mma']:5.2f}/{b['mma12']:5.2f} {100 * fl:6.2f} | " + " | ".join(cells))
        print(f"{'a step':10s} {'(layers)':>13s} {'':11s} {'':6s} | ", end="")
        cells = []
        for M in SMALL:
            s = {l: sum(mix[n] * (t(rows, n, M, l) or {}).get("us", float("nan")) for n in names) for l in ("mma", "mma12")}
            sb = sum(mix[n] * (t(rows, n, M, "mma") or {}).get("bf16_us", float("nan")) for n in names)
            cells.append(f"{s['mma']:6.0f} / {s['mma12']:6.0f} / {sb:6.0f} = {s['mma'] / s['mma12']:4.2f}")
        print(" | ".join(cells) + "\n")
        vs = [v for v in dict.fromkeys(r.get("variant", "base") for r in rows) if v != "base"]
        if vs:
            Ms = sorted({r["M"] for r in rows})
            print("the tiered layout's variants (its default route under each one's variables), a step through the layers over the 12-bit layout's; per matrix, the variant over 12-bit")
            for v in ["base"] + vs:
                cells = []
                for M in Ms:
                    a = sum(mix[n] * (t(rows, n, M, "mma", variant=v) or {}).get("us", float("nan")) for n in names if n != "lm_head")
                    b = sum(mix[n] * (t(rows, n, M, "mma12") or {}).get("us", float("nan")) for n in names if n != "lm_head")
                    cells.append(f"{M}:{a / b:5.3f}")
                print(f"  {v:>8s}: " + " ".join(cells))
                for n in names:
                    cc = []
                    for M in Ms:
                        a, b = t(rows, n, M, "mma", variant=v), t(rows, n, M, "mma12")
                        cc.append(f"{M}:{a['us'] / b['us']:5.3f}" if a and b else f"{M}:  -  ")
                    print(f"    {n:10s} " + " ".join(cc))
            print()
        Ms = sorted({r["M"] for r in rows if r["M"] <= 64})
        print("the tiered layout by its routes before this one (gemm: the step kernel to 16 tokens, mma_gemm past them), a step through the layers over the 12-bit layout's")
        cells = []
        for M in Ms:
            a = sum(mix[n] * (t(rows, n, M, "mma", "gemm") or {}).get("us", float("nan")) for n in names if n != "lm_head")
            b = sum(mix[n] * (t(rows, n, M, "mma12") or {}).get("us", float("nan")) for n in names if n != "lm_head")
            cells.append(f"{M}:{a / b:5.3f}")
        print("      gemm: " + " ".join(cells) + "\n")
        if PROMPTS:
            print(f"prompts, a pass through the layers' products, ms: tiered by its route (default) and by each, 12-bit by its route, bf16")
        for M in PROMPTS:  # (none in this job)
            vals = {}
            for l, rt in (("mma", "default"), ("mma", "big"), ("mma", "decode"), ("mma12", "default"), ("mma12", "big"), ("mma12", "decode")):
                vals[(l, rt)] = sum(mix[n] * (t(rows, n, M, l, rt) or {}).get("us", float("nan")) for n in names if n != "lm_head") / 1e3
            sb = sum(mix[n] * (t(rows, n, M, "mma") or {}).get("bf16_us", float("nan")) for n in names if n != "lm_head") / 1e3
            print(f"  {M:5d} tokens: mma {vals[('mma', 'default')]:7.2f} (big {vals[('mma', 'big')]:7.2f}, decode {vals[('mma', 'decode')]:7.2f}) | mma12 {vals[('mma12', 'default')]:7.2f} (big {vals[('mma12', 'big')]:7.2f}, decode {vals[('mma12', 'decode')]:7.2f}) | bf16 {sb:7.2f}"
                  f" | tiered's best / 12-bit's route {min(vals[('mma', 'big')], vals[('mma', 'decode')]) / vals[('mma12', 'default')]:4.2f}")
        print()
        print(f"memory or instructions: clk a step of 1,024 weights at {mhz or g['mhz']:.0f} MHz (from memory; from the L2 cache where it fits), the packs' read rate GB/s, bf16's at 1 token")
        for n in names:
            parts = []
            for l in ("mma", "mma12"):
                for M in (1, 16, 64):
                    r = t(rows, n, M, l)
                    if not r:
                        continue
                    k = (mhz or r["mhz"]) / r["mhz"]
                    hot = f"/{r['hot_us'] * r['clk_step'] / r['us'] * k:4.0f}" if "hot_us" in r else ""
                    parts.append(f"{l} {M}: {r['clk_step'] * k:4.0f}{hot}")
            r1 = t(rows, n, 1, "mma")
            if r1:
                rd = lambda l: (t(rows, n, 1, l) or {}).get("us", float("nan"))
                gb = lambda l, bits: r1["O"] * r1["K"] * bits / 8 / (rd(l) * 1e3)
                bm = next((r["bits"] for r in rows if r["name"] == n and r["layout"] == "mma"), 0)
                b12 = next((r["bits"] for r in rows if r["name"] == n and r["layout"] == "mma12"), 0)
                parts.append(f"GB/s 1 token: mma {gb('mma', bm):5.0f}, mma12 {gb('mma12', b12):5.0f}, bf16 {r1['O'] * r1['K'] * 2 / (r1['bf16_us'] * 1e3):5.0f}")
            print(f"  {n:10s} " + "; ".join(parts))
        print()
    for l in ("mma", "mma12"):
        f = f"{R}/moe-{l}.json"
        if os.path.exists(f):
            d = json.load(open(f))
            print(f"######## experts (Qwen3.5-35B-A3B layer 3, {d['E']} experts, top {d['k']}), {l}: a layer's products, us (route + gate/up + down), bf16 = vLLM's fused_experts")
            for r in d["rows"]:
                print(f"  T={r['T']:5d}: all {r.get('all_us', float('nan')):8.1f}  gate+up {r.get('gate_up_us', float('nan')):8.1f}  down {r.get('down_us', float('nan')):8.1f}  bf16 {r.get('bf16_us', float('nan')):8.1f}")
            print()
    modes = [m for m in ("bf16", "mma12", "mma", "mma12b", "mmab") if glob.glob(f"{R}/serve/{m}-u*.json")]
    if modes:
        print("######## vllm serve Qwen3.5-9B (text only), 1,024 tokens in and 256 out: output tokens/s, mean time to first token ms, mean time a token ms")
        res = {(m, u): json.load(open(f"{R}/serve/{m}-u{u}.json")) for m in modes for u in (1, 8, 16, 32, 64) if os.path.exists(f"{R}/serve/{m}-u{u}.json")}
        print(f"{'users':>6s} | " + " | ".join(f"{m:>26s}" for m in modes) + " | tiered over 12-bit (tok/s, TTFT, TPOT): each pass of each, then both passes' means")
        r3 = lambda a, b: f"{a['output_throughput'] / b['output_throughput']:5.3f} {a['mean_ttft_ms'] / b['mean_ttft_ms']:5.3f} {a['mean_tpot_ms'] / b['mean_tpot_ms']:5.3f}"
        for u in (1, 8, 16, 32, 64):
            cells = [f"{res[(m, u)]['output_throughput']:8.1f} {res[(m, u)]['mean_ttft_ms']:8.1f} {res[(m, u)]['mean_tpot_ms']:7.2f}" if (m, u) in res else " " * 26 for m in modes]
            parts = [f"{t}/{w} {r3(res[(t, u)], res[(w, u)])}" for t, w in (("mma", "mma12"), ("mmab", "mma12b"), ("mma12b", "mma12"), ("mmab", "mma")) if (t, u) in res and (w, u) in res]
            ts, ws = [res[(m, u)] for m in ("mma", "mmab") if (m, u) in res], [res[(m, u)] for m in ("mma12", "mma12b") if (m, u) in res]
            if ts and ws:
                mean = lambda xs, k: sum(x[k] for x in xs) / len(xs)
                parts.append("means " + " ".join(f"{mean(ts, k) / mean(ws, k):5.3f}" for k in ("output_throughput", "mean_ttft_ms", "mean_tpot_ms")))
            print(f"{u:6d} | " + " | ".join(cells) + " | " + "; ".join(parts))
        print()
    if os.path.exists(f"{R}/sass.txt"):
        print("######## SASS (sass.txt): each kernel's whole count and its loops")
        for line in open(f"{R}/sass.txt"):
            if line.startswith("########") or "instructions" in line:
                print("  " + line.rstrip()[:200])


if __name__ == "__main__":
    main()
