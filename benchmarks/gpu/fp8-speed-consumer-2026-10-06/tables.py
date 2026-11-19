"""The tables of README.md from a run's bench JSON (gpu/fp8/bench_f8.py --out): python tables.py BENCH.json [BENCH_B.json] > tables.md
Times are microseconds a product, each runner's weights on copies (200 MB, or 3 L2s from the bench of 2026-10-06's review on), one CUDA graph a
runner, best of 5. "Per-shape sum": a model's distinct FP8 Linear shapes at a tensor-parallel size, each once (not weighted by layer counts, so not a
decode step's time), a runner's times added over vLLM's added; "dense" leaves out the expert shapes (vLLM runs routed experts through its fused MoE
kernels, not the dense FP8 Linear kernel they are timed against here). Left out of every sum and count, and marked in the per-shape table: a shape
whose copies held less than 1.5 times the GPU's L2 (the older runs' cap of 64 copies: some of its weights may have come from L2), and a block-scaled
shape vLLM refuses at its tensor-parallel split (N or K not a multiple of 128); and on compute capability 10.x/12.x, a Qwen3.6/3.8 shape timed against
DeepGEMM, which stock vLLM 0.30 turns off for their model types there (qwen3_5_text, qwen3_5_moe_text: its _DEEPGEMM_BLACKWELL_EXCLUDED_MODEL_TYPES)
and serves with CUTLASS's block-scaled kernel (bench_f8.py before the review's fix stood in model type qwen3 for every model)."""
import collections, json, math, sys

A = json.load(open(sys.argv[1]))
B = json.load(open(sys.argv[2])) if len(sys.argv) > 2 else None
key = lambda r: (r["model"], r["tp"], r["N"], r["K"], r["M"])
L2 = A.get("l2") or {"NVIDIA GeForce RTX 4090": 72 << 20, "NVIDIA GeForce RTX 5090": 96 << 20}[A["gpu"]]
why_out = lambda r: ("not stock's kernel" if A["cc"][0] in (10, 12) and r["model"].startswith(("Qwen3.6", "Qwen3.8")) and (r.get("vllm_kernel") or "").startswith("DeepGemm") else
                     "L2" if r.get("copies", 0) * r["N"] * r["K"] < 1.5 * L2 else
                     "vLLM refuses" if r.get("kind") == "block" and r["tp"] > 1 and (r["N"] % 128 or r["K"] % 128) else "")
every = [r for r in A["rows"] if r.get("M", -1) > 0]
rows = [r for r in every if not why_out(r)]
bidx = {key(r): r for r in B["rows"] if r.get("M", -1) > 0} if B else {}
specs = sorted({s for r in rows for s in r.get("variants", {})})
gm = lambda v: math.exp(sum(map(math.log, v)) / len(v)) if v else float("nan")
f = lambda v: "-" if v is None else f"{v:.1f}"
runners = [("ours", lambda r: r.get("ours_us"))] + [(s, lambda r, s=s: r.get("variants", {}).get(s)) for s in specs]
if B:
    runners += [(f"B {s or '(W8A8)'}", lambda r, s=s: (lambda b: (b.get("variants", {}).get(s) if s else b.get("ours_us")) if b else None)(bidx.get(key(r))))
                for s in [None] + sorted({s for r in B["rows"] for s in r.get("variants", {})})]
runners += [("decode + vLLM's kernel", lambda r: r.get("route_us"))]
models = list(dict.fromkeys(r["model"] for r in rows))

print(f"# {A['gpu']} (sm_{A['cc'][0]}{A['cc'][1]}, L2 {L2 >> 20} MiB)\n")
print("vLLM's kernels: " + "; ".join(sorted({f"{r['model']}: {r['vllm_kernel']}" for r in rows if r.get("vllm_kernel")})) + "\n")
if A.get("served"):
    print("Stock vLLM's own selection on each checkpoint (bench_f8.py --served): " + "; ".join(f"{m}: {', '.join(v['kernels']) or 'no line'}" for m, v in A["served"].items()) + "\n")
out = collections.Counter(why_out(r) for r in every if why_out(r))
print(f"Shapes x tokens left out of the sums: {sum(out.values())} of {len(every)} ({', '.join(f'{k}: {v}' for k, v in out.items()) or 'none'})\n")
mis = sorted({r["err"] for r in every if "a served run selects" in (r.get("err") or "")})
if mis:
    print("No vLLM time (not the kernel stock vLLM serves the checkpoint with): " + "; ".join(mis) + "\n")
print("## Per-shape sums: a runner's times added over vLLM's added (each distinct shape once; dense: the expert shapes left out)\n")
print("| model | tp | shapes | runner | M=1 | M=8 | M=32 | M=64 |\n| :--- | ---: | :--- | :--- | ---: | ---: | ---: | ---: |")
for m in models:
    for tp in sorted({r["tp"] for r in rows if r["model"] == m}):
        for which in ("all", "dense"):
            if which == "dense" and not any(r["role"].startswith("expert") for r in rows if r["model"] == m):
                continue
            for name, get in runners:
                cells = []
                for M in (1, 8, 32, 64):
                    rs = [r for r in rows if r["model"] == m and r["tp"] == tp and r["M"] == M and r.get("vllm_us") and (which == "all" or not r["role"].startswith("expert"))]
                    if rs and all(get(r) for r in rs):
                        cells.append(f"{sum(get(r) for r in rs) / sum(r['vllm_us'] for r in rs):.2f}")
                    else:
                        cells.append("-")
                if any(c != "-" for c in cells):
                    print(f"| {m} | {tp} | {which} | {name} | " + " | ".join(cells) + " |")
print()
print("## Per shape: geometric mean of a runner's time over vLLM's, all models (in brackets: shapes at most vLLM's; then the range)\n")
print("| runner | M=1 | M=8 | M=32 | M=64 |\n| :--- | ---: | ---: | ---: | ---: |")
for name, get in runners:
    cells = []
    for M in (1, 8, 32, 64):
        v = [get(r) / r["vllm_us"] for r in rows if r["M"] == M and r.get("vllm_us") and get(r)]
        cells.append(f"{gm(v):.2f} ({sum(x <= 1 for x in v)}/{len(v)}; {min(v):.2f}-{max(v):.2f})" if v else "-")
    print(f"| {name} | " + " | ".join(cells) + " |")
print()
print("## Per shape: ours over the decode route (in brackets: shapes where the route is faster; then the range)\n")
print("| kind | M=1 | M=8 | M=32 | M=64 |\n| :--- | ---: | ---: | ---: | ---: |")
for kind in sorted({r["kind"] for r in rows}):
    cells = []
    for M in (1, 8, 32, 64):
        v = [r["ours_us"] / r["route_us"] for r in rows if r["kind"] == kind and r["M"] == M and r.get("ours_us") and r.get("route_us")]
        cells.append(f"{gm(v):.2f} ({sum(x > 1 for x in v)}/{len(v)}; {min(v):.2f}-{max(v):.2f})" if v else "-")
    print(f"| {kind} | " + " | ".join(cells) + " |")
print()
print("## Every shape (microseconds)\n")
hdr = ["vLLM"] + [n for n, _ in runners] + ["decode alone", "copies (MB)", "left out"]
print("| model | tensor | tp | N x K | M | " + " | ".join(hdr) + " |\n| :--- | :--- | ---: | :--- | ---: | " + " | ".join("---:" for _ in hdr) + " |")
for r in every:
    cells = [f(r.get("vllm_us"))] + [f(get(r)) for _, get in runners] + [f(r.get("unpack_us")), f"{r.get('copies', 0) * r['N'] * r['K'] / 1e6:.0f}", why_out(r)]
    print(f"| {r['model']} | {r['role']} | {r['tp']} | {r['N']} x {r['K']} | {r['M']} | " + " | ".join(cells) + " |")
print()
if A.get("numerics"):
    ns = [k[:-4] for k in sorted({k for r in A["numerics"] for k in r if k.endswith("_rms") and k.startswith("ours")})]
    print("## Numerics: error against an exact float64 product of the dequantized weights (max: of the sum of |x w|; rms: of the outputs' rms)\n")
    print("| model | tensor | M | vLLM kernel | vLLM max | vLLM rms | " + " | ".join(f"{n} max | {n} rms" for n in ns) + " |")
    print("| :--- | :--- | ---: | :--- | ---: | ---: | " + " | ".join("---: | ---:" for _ in ns) + " |")
    for r in A["numerics"]:
        e = lambda k: "(not stock's)" if k.startswith("vllm") and why_out(dict(r, tp=1, copies=1 << 40)) else f"{r[k]:.1e}" if k in r else "-"
        print(f"| {r['model']} | {r['role']} | {r['M']} | {r['vllm_kernel'][:40]} | {e('vllm_max')} | {e('vllm_rms')} | " + " | ".join(f"{e(n + '_max')} | {e(n + '_rms')}" for n in ns) + " |")
