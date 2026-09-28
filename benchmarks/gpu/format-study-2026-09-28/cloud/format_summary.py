"""gpu-format's confirming run, summarized from its results directory (python format_summary.py R): the machine, the
steps done, the self-check, each layer run (the split byte against the 12-bit layout in the same kernel, and both
against cuBLAS; the mean of the runs), the full-model check, and format-report-1.md's go criteria:
- H100 (9.0): the split byte at most 0.95x the 12-bit layout's layer time at 128-512 tokens, and at most 1.00x at 1-64
  (main's route: the step kernel to 16 tokens, the TMA kernel from 17);
- A100 (8.0): at most 1.00x at main's routes timed (the step kernel to 16 tokens, the prompt kernel from 129; its own
  17-128 kernel is not in the prototype: the others are shown, not judged);
- all products the same bits as the 12-bit layout's, every tensor bit for bit.
Written by format_job.sh after every step, so a partial run still reads."""
import glob, os, re, sys

R = sys.argv[1]


def read(name):
    p = os.path.join(R, name)
    return open(p, errors="replace").read() if os.path.exists(p) else ""


out = []
mach = read("machine.txt")
q = re.search(r"^name, compute_cap[^\n]*\n([^\n]+)", mach, re.M)
cc = q.group(1).split(",")[1].strip() if q else "?"
tv = re.search(r"^torch \S+ CUDA \S+", mach, re.M)
nv = re.search(r"release (\S+), (V\S+)", mach)
src = re.search(r"^sources: .*", mach, re.M)
out.append(f"machine: {q.group(1).strip() if q else '?'}")
out.append(f"         {tv.group(0) if tv else '?'}; nvcc {nv.group(1) + ' ' + nv.group(2) if nv else '?'}; {src.group(0) if src else 'sources: ?'}")
out.append("steps: " + "; ".join(re.sub(r"^\S+ \(\+(\d+) s\) ", r"[\1 s] ", l) for l in read("steps.txt").splitlines()))
sc = read("selfcheck.txt")
oks = sc.count("round trip and products ok")
ex = re.findall(r"^exit (\d+)", sc, re.M)
out.append(f"self-check: {oks} matrices ok, exit {ex[-1] if ex else '?'}" + ("" if oks or not sc else f" ({sc.strip().splitlines()[-1][:150]})"))

# layer runs: {model: {M: {route: [(t12, tsb)], "cublas": [us]}}}
runs = {}
for f in sorted(glob.glob(os.path.join(R, "layer-*-run*.txt"))):
    model = re.match(r"layer-(.+)-run\d+\.txt", os.path.basename(f)).group(1)
    for line in open(f, errors="replace"):
        m = re.match(r"\s+M=\s*(\d+): layer cuBLAS\s+([\d.]+) us \| (.*?)\s+\[", line)
        if not m:
            continue
        M, cb = int(m.group(1)), float(m.group(2))
        t = {k + v: float(x) for k, v, x, _ in re.findall(r"(\w+?)(12|sb)\s+([\d.]+) \(([\d.]+)x\)", m.group(3))}
        d = runs.setdefault(model, {}).setdefault(M, {"cublas": []})
        d["cublas"].append(cb)
        for r in {k[:-2] for k in t}:
            if r + "12" in t and r + "sb" in t:
                d.setdefault(r, []).append((t[r + "12"], t[r + "sb"]))


def mean(v):
    return sum(v) / len(v)


def main_route(M):  # the kernel main takes for the 12-bit layout at M tokens on this GPU (of those timed)
    if cc == "9.0":
        return "step" if M <= 16 else "wg"
    if cc == "8.9":
        return "step" if M <= 16 else "mid" if M <= 64 else "big"
    return "step" if M <= 64 else "big"


worst = {"128-512": [], "1-64": [], "all": [], "a100": []}
for model, byM in runs.items():
    out.append(f"{model}: M, cuBLAS us, then each kernel: 12-bit and split byte against cuBLAS, split / 12-bit (the runs' mean; n runs)")
    for M in sorted(byM):
        d = byM[M]
        cells = []
        for r in (k for k in d if k != "cublas"):
            pairs = d[r]
            cb = mean(d["cublas"])
            s = mean([b / a for a, b in pairs])
            cells.append(f"{r} {mean([a for a, _ in pairs]) / cb:.3f} / {mean([b for _, b in pairs]) / cb:.3f} = {s:.3f} ({len(pairs)})")
            worst["all"].append((s, model, M, r))
            if r == main_route(M) and M <= 64:
                worst["1-64"].append((s, model, M, r))
            if r == main_route(M) and 128 <= M <= 512:
                worst["128-512"].append((s, model, M, r))
            if (r == "step" and M <= 16) or (r == "big" and M >= 129):
                worst["a100"].append((s, model, M, r))
        out.append(f"  M={M:5d} {mean(d['cublas']):8.1f} | " + " | ".join(cells))

checks = sorted(glob.glob(os.path.join(R, "check-*.txt")))
same = diff = 0
bits, prods = [], []
for f in checks:
    t = open(f, errors="replace").read()
    same += t.count("same bits")
    diff += t.count("DIFFERS")
    bits += re.findall(r"^\S+: \d+ tensors, [\d.]+ B weights, every one bit for bit.*$", t, re.M)
    prods += re.findall(r"^\S+: layer \d+'s products (all ok|\d+ failed)", t, re.M)
    if "Traceback" in t or "Error" in t:
        out.append(f"check {os.path.basename(f)}: FAILED: " + [l for l in t.splitlines() if l.strip()][-1][:200])
for b in bits:
    out.append("check: " + b)
if checks:
    out.append(f"check: {same} products the 12-bit layout's bits, {diff} differ; within 1e-2 of fp32 and repeatable: {', '.join(prods) or '?'}")


def verdict(name, items, limit):
    if not items:
        return f"{name}: not measured"
    s, model, M, r = max(items)
    return f"{name}: {'PASS' if s <= limit else 'FAIL'} (worst {s:.3f}, {model} M={M} {r}; limit {limit:.2f}; {len(items)} points)"


out.append("go criteria (format-report-1.md section 4):")
if cc == "9.0":
    out.append("  " + verdict("H100, 128-512 tokens (TMA kernel)", worst["128-512"], 0.95))
    out.append("  " + verdict("H100, 1-64 tokens (main's route)", worst["1-64"], 1.00))
elif cc == "8.0":
    out.append("  " + verdict("A100, main's routes (1-16 tokens, 129-768)", worst["a100"], 1.00))
    rest = [x for x in worst["all"] if x not in worst["a100"]]
    if rest:
        out.append(f"  (not judged: the other kernels timed, worst {max(rest)[0]:.3f}: {max(rest)[1]} M={max(rest)[2]} {max(rest)[3]})")
else:
    out.append(f"  (compute capability {cc}: no criterion; the worst split / 12-bit, every M and kernel: "
               + (f"{max(worst['all'])[0]:.3f})" if worst["all"] else "none yet)"))
out.append("  products: " + ("not checked yet" if not checks else "PASS (every tensor bit for bit, every product the same bits)" if bits and same and not diff and prods and all(x == "all ok" for x in prods) else "FAIL"))
print("\n".join(out))
