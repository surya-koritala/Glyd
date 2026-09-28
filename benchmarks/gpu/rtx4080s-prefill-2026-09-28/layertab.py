"""layer-{main,fin}N-MODEL.txt (or PREFIX-..., e.g. step; mb.py, variant 0 through each tree's library): a layer's time
over cuBLAS's in the same run, each tree's runs averaged (their range), per model, layout and length.
    python layertab.py [LOGS] [PREFIX]"""
import glob, os, re, statistics, sys
logs = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
pre = sys.argv[2] if len(sys.argv) > 2 else "layer"
d = {}
for f in glob.glob(f"{logs}/{pre}-*-*.txt"):
    m = re.match(rf".*/{pre}-(main|fin)\d-(.+)\.txt", f)
    tree, model = m.groups()
    for l in open(f):
        m2 = re.match(r"\s+(mma12|mma)\s+M=\s*(\d+): layer cuBLAS\s+([\d.]+) us, v0\s+([\d.]+) \(([\d.]+)x\)", l)
        if m2:
            fmt, M, c, t, r = m2.groups()
            d.setdefault((model, fmt, int(M), tree), []).append(float(t) / float(c))
for model in sorted({k[0] for k in d}):
    for fmt in ("mma12", "mma"):
        Ms = sorted({k[2] for k in d if k[0] == model and k[1] == fmt})
        if not Ms:
            continue
        print(f"{model} {fmt}: " + " | ".join(str(M) for M in Ms))
        for tree in ("main", "fin"):
            cells = []
            for M in Ms:
                v = d.get((model, fmt, M, tree), [])
                cells.append(f"{statistics.mean(v):.3f}" + (f" [{min(v):.3f}-{max(v):.3f}]" if len(v) > 1 else "") if v else "-")
            print(f"  {tree:4} " + " | ".join(cells))
        gains = [statistics.mean(d[(model, fmt, M, 'fin')]) / statistics.mean(d[(model, fmt, M, 'main')]) - 1 for M in Ms if (model, fmt, M, 'fin') in d and (model, fmt, M, 'main') in d]
        print("  branch against main: " + " | ".join(f"{100 * g:+.1f}%" for g in gains))
