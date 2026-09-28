"""e2e-{main,fin}{1,2}-MODEL-FMT.txt: one pass and the first token (ms) at each length, bf16 (the runs' median) and
Glyd main / branch (the mean of each tree's two runs, with its spread), and generation tokens/s."""
import glob, re, statistics, sys
logs = sys.argv[1] if len(sys.argv) > 1 else "/home/surya-koritala/p6prefill/logs"
data = {}
for f in glob.glob(f"{logs}/e2e-*-*.txt"):
    m = re.match(r".*/e2e-(main|fin)(\d)-(.+)-(mma12|mma)\.txt", f)
    if not m:
        continue
    tree, rep, model, fmt = m.groups()
    for l in open(f):
        who = "bf16" if l.startswith("bf16") else "glyd" if l.startswith("glyd") else None
        if who and "prefill:" in l:
            for n, p, t in re.findall(r"(\d+) tokens ([\d.]+) ms \(\d+ tokens/s\), first token ([\d.]+) ms", l):
                data.setdefault((model, fmt, who if who == "bf16" else tree, "pass", int(n)), []).append(float(p))
                data.setdefault((model, fmt, who if who == "bf16" else tree, "ttft", int(n)), []).append(float(t))
        m2 = re.match(r"(bf16|glyd)[^:]*: batch (\d+): ([\d.]+) tokens/s", l)
        if m2:
            data.setdefault((model, fmt, "bf16" if m2.group(1) == "bf16" else tree, "gen", int(m2.group(2))), []).append(float(m2.group(3)))
models = sorted({k[0] for k in data})
for model in models:
    for fmt in ("mma", "mma12"):
        for what in ("pass", "ttft", "gen"):
            Ns = sorted({k[4] for k in data if k[0] == model and k[1] == fmt and k[3] == what})
            if not Ns:
                continue
            print(f"{model} {fmt} {what}: " + " | ".join(str(n) for n in Ns))
            for who in ("bf16", "main", "fin"):
                cells = []
                for n in Ns:
                    v = data.get((model, fmt, who, what, n), [])
                    if not v:
                        cells.append("-")
                        continue
                    mean = statistics.mean(v)
                    ref = statistics.median(data.get((model, fmt, "bf16", what, n), [mean]))
                    rel = "" if who == "bf16" else f" ({100 * (mean / ref - 1):+.1f}%)"
                    cells.append(f"{mean:.1f}{rel}" + (f" [{min(v):.1f}-{max(v):.1f}]" if len(v) > 1 and who != "bf16" else ""))
                print(f"  {who:5} " + " | ".join(cells))
