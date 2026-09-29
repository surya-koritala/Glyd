"""The routes' results in one table each: end to end (e2e-ROUTE.txt: a prompt's pass, Glyd against bf16 in the same
process, by length) and per layer (layer.txt: a pass of layers by route, against cuBLAS, with the SM clock and power;
steps by mma_gemm against mma_gemm_mid). The best route at each length is marked *.
    python summary.py RESULTS_DIR"""
import glob, os, re, sys

R = sys.argv[1]
e2e = {}
for f in sorted(glob.glob(os.path.join(R, "e2e-*.txt"))):
    route = os.path.basename(f)[4:-4]
    text = open(f).read()
    lines = {w: re.findall(r"(\d+) tokens ([\d.]+) ms \(", l) for w, l in re.findall(r"^(bf16|glyd \w+) prefill: (.*)$", text, re.M)}
    bf = {int(n): float(t) for n, t in lines.get("bf16", [])}
    gl = {int(n): float(t) for k, v in lines.items() if k.startswith("glyd") for n, t in v}
    if bf and gl:
        e2e[route] = {n: (gl[n], bf[n]) for n in gl if n in bf}
    smi = os.path.join(R, f"smi-e2e-{route}.csv")
    if os.path.exists(smi):  # the run's busy samples: SM clock and power
        rows = [l.split(", ") for l in open(smi).read().splitlines()[1:]]
        busy = [(float(r[1].split()[0]), float(r[3].split()[0])) for r in rows if len(r) > 5 and r[5].split()[0].isdigit() and int(r[5].split()[0]) >= 90 and r[1].split()[0].replace(".", "").isdigit()]
        if busy and route in e2e:
            e2e[route]["smi"] = (sorted(c for c, _ in busy)[len(busy) // 2], sorted(p for _, p in busy)[len(busy) // 2])
if e2e:
    lengths = sorted({n for r in e2e.values() for n in r if n != "smi"})
    print("End to end, a prompt's pass: Glyd ms (over bf16's in the same run)")
    print("| route | " + " | ".join(str(n) for n in lengths) + " | SM clock, power (busy) |")
    print("| :--- | " + " | ".join("---:" for _ in lengths) + " | ---: |")
    best = {n: min((r[n][0] / r[n][1], k) for k, r in e2e.items() if n in r)[1] for n in lengths}
    for k, r in e2e.items():
        cells = [f"{r[n][0]:.1f} ({100 * (r[n][0] / r[n][1] - 1):+.1f}%){'*' if best[n] == k else ''}" if n in r else "" for n in lengths]
        s = r.get("smi")
        print(f"| {k} | " + " | ".join(cells) + f" | {f'{s[0]:.0f} MHz, {s[1]:.0f} W' if s else ''} |")
    print()
layer = os.path.join(R, "layer.txt")
if os.path.exists(layer):
    for tag in sorted({l.split()[0] for l in open(layer) if " pass of " in l}):
        rows = [l for l in open(layer) if l.startswith(tag + " ") and " pass of " in l]
        for kind in ("pass of", "steps, pass of"):
            got = [(int(re.search(r"M=(\d+)", l).group(1)), re.findall(r"(\w+) ([\d.]+) ms ([\d.]+)x \((\S+) MHz, (\S+) W\)", l)) for l in rows if l.split(" ", 1)[1].startswith(kind)]
            if not got:
                continue
            routes = [r for r, *_ in got[0][1]]
            print(f"Per layer, {tag}, {kind} layers: time over cuBLAS's (SM clock MHz, power W)")
            print("| M | " + " | ".join(routes) + " |")
            print("| ---: | " + " | ".join("---:" for _ in routes) + " |")
            for M, res in got:
                b = min((float(x), r) for r, _, x, _, _ in res if r != "cuBLAS")[1]
                print(f"| {M} | " + " | ".join(f"{float(x):.3f}{'*' if r == b else ''} ({c}, {p})" for r, _, x, c, p in res) + " |")
            print()
loop = os.path.join(R, "loop.txt")
if os.path.exists(loop):
    got = {}
    for l in open(loop):
        r = re.search(r"(static|eager) batch (\d+): (cache \d+|prompt \d+).*: ([\d.]+) ms a step", l)
        if r:
            got.setdefault((int(r.group(2)), r.group(3)), {})[r.group(1)] = float(r.group(4))
    print("generate()'s step, ms: compiled (static cache) against eager, by the cache's length (a 16-token prompt, 64 tokens of max_new_tokens' allowed) or the prompt")
    print("| batch | length | compiled | eager | compiled's gain |")
    print("| ---: | :--- | ---: | ---: | ---: |")
    for (b, what), v in sorted(got.items(), key=lambda kv: (kv[0][0], kv[0][1].split()[0], int(kv[0][1].split()[1]))):
        if "static" in v and "eager" in v:
            print(f"| {b} | {what} | {v['static']:.2f} | {v['eager']:.2f} | {100 * (v['eager'] / v['static'] - 1):+.0f}% |")
