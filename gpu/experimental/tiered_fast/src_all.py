import sys, src_show
steps = float(sys.argv[2]); N = int(sys.argv[3]) if len(sys.argv) > 3 else 40
d = src_show.parse(sys.argv[1])
rows = []
for f, lines in d.items():
    for ln, (n, s) in lines.items():
        if n: rows.append((n, f.split("/")[-1], ln, s))
tot = sum(r[0] for r in rows if r[1] in ("harness.cu", "glyd_gpu.cu", "v2.cuh"))
print("executed per step, by file:", {f.split("/")[-1]: round(sum(n for n, _ in lines.values()) / steps, 1) for f, lines in d.items()})
for n, f, ln, s in sorted(rows, reverse=True)[:N]:
    print(f"{n / steps:7.1f}  {f}:{ln}  {s[:140]}")
