"""thr-MODEL-{fused,ahead}N.txt and thr2-... (e2e.py --prefill, the 12-bit layout, GLYD_AHEAD_MIN past every length
or 1024): one pass (ms) fused and decoded ahead, each the mean of its runs (their range), bf16 the median; then, for
each threshold T (decoded ahead from T tokens, fused below), how much slower than the faster of the two each model's
prompts are."""
import glob, os, re, statistics, sys
logs = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
d = {}
for f in glob.glob(f"{logs}/thr*-*.txt"):
    model, mode = re.match(r".*/thr2?-(.+)-(fused|ahead)\d\.txt", f).groups()
    for l in open(f):
        if "prefill:" in l:
            who = "bf16" if l.startswith("bf16") else mode
            for n, p in re.findall(r"(\d+) tokens ([\d.]+) ms", l):
                d.setdefault((model, who, int(n)), []).append(float(p))
models = sorted({k[0] for k in d})
Ns = sorted({k[2] for k in d})
best = {}
for model in models:
    print(f"{model}: " + " | ".join(str(n) for n in Ns))
    for who in ("bf16", "fused", "ahead"):
        v = [d.get((model, who, n)) for n in Ns]
        if any(v):
            print(f"  {who:5} " + " | ".join("-" if not x else f"{statistics.median(x):.1f}" if who == "bf16" else f"{statistics.mean(x):.1f} [{min(x):.1f}-{max(x):.1f}]" for x in v))
    f, a = [statistics.mean(d[(model, "fused", n)]) for n in Ns], [statistics.mean(d[(model, "ahead", n)]) for n in Ns]
    print("  fused against ahead " + " | ".join(f"{100 * (x / y - 1):+.1f}%" for x, y in zip(f, a)))
    best[model] = (f, a)
print("decoded ahead from T tokens: the most and the mean a prompt is slower than the faster route, by model")
for T in Ns + [Ns[-1] + 1]:
    cells = []
    for model in models:
        f, a = best[model]
        loss = [100 * ((y if n >= T else x) / min(x, y) - 1) for n, x, y in zip(Ns, f, a)]
        cells.append(f"{model} {max(loss):.1f} / {statistics.mean(loss):.1f}%")
    print(f"  T {T:5}: " + ", ".join(cells))
