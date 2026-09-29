"""h100_cu12.sh's summary from what is in RESULTS so far (a cap still leaves partial answers):
(a) each library's toolchain, ptxas's registers, spills and C7515 a kernel, the SASS's serialized wgmma, its products
against fp32 and bit for bit against the others, and its layer times against cuBLAS; (b) the whole-tiles rule against
library 4; (c) the e2e check.
    python cu12_summary.py RESULTS            (the summary on stdout)
    python cu12_summary.py RESULTS --winner   (the library for the e2e check)
"""
import glob, json, os, re, statistics, sys

R = sys.argv[1]
LIBS = ["cu12-base", "cu12-fix", "cu13-base", "cu13-fix", "cu13-fixn1"]
KERNELS = [("wgp256", "mma12_wgp_kernelILi256ELi2E"), ("wgp192", "mma12_wgp_kernelILi192ELi2E")] + [
    (f"tma{n}", f"mma12_tma_kernelILi{n}ELi2E") for n in (16, 32, 64, 96, 112, 128)]


def read(p):
    try:
        return open(p).read()
    except OSError:
        return ""


def ptxas(log):
    """{kernel: (registers, spill bytes, C7515)} from a build's -Xptxas -v output."""
    out, c7515 = {}, set(re.findall(r"C7515\).*?function '(\S+?)'", log))
    for short, key in KERNELS:
        m = re.search(r"Compiling entry function '(_Z\d+" + key + r"\S*)' for 'sm_90a'\n(.*?)ptxas info\s+: Used (\d+) registers", log, re.S)
        if m:
            sp = re.findall(r"(\d+) bytes spill stores, (\d+) bytes spill loads", m.group(2))
            out[short] = (int(m.group(3)), sum(int(a) + int(b) for a, b in sp), any(f.startswith(m.group(1)[:60]) or m.group(1).startswith(f) for f in c7515))
    return out


def layer(path):
    """{(tag, model, product, M): (cuBLAS us, kernel us)}"""
    out = {}
    for line in read(path).splitlines():
        m = re.match(r"(\S+) (\S+) (\w+) \d+x\d+ M=(\d+): cuBLAS ([\d.]+) us \| \w+ ([\d.]+) us", line)
        if m:
            out[(m.group(1), m.group(2), m.group(3), int(m.group(4)))] = (float(m.group(5)), float(m.group(6)))
    return out


def med(v):
    return statistics.median(v) if v else None


def by_lib(runs):
    """{lib: {(model, product, M): (median cuBLAS us, median kernel us)}} over rounds (tags LIB-rN)."""
    acc = {}
    for (tag, model, prod, M), (c, t) in runs.items():
        lib = tag.rsplit("-r", 1)[0]
        acc.setdefault(lib, {}).setdefault((model, prod, M), []).append((c, t))
    return {lib: {k: (med([c for c, _ in v]), med([t for _, t in v])) for k, v in d.items()} for lib, d in acc.items()}


def winner():
    b = by_lib(layer(f"{R}/layer-b.txt"))
    a = by_lib(layer(f"{R}/layer-a.txt"))
    pick = "cu13-fix"
    common = set(b.get("cu13-fix", {})) & set(b.get("cu13-fixn1", {}))
    if common and sum(b["cu13-fixn1"][k][1] for k in common) < sum(b["cu13-fix"][k][1] for k in common):
        pick = "cu13-fixn1"
    common = set(a.get("cu13-fix", {})) & set(a.get("cu13-base", {}))
    if common and sum(a["cu13-fix"][k][1] for k in common) > 1.01 * sum(a["cu13-base"][k][1] for k in common):
        pick = "cu13-base"  # (the fix costs CUDA 13 more than 1%)
    return pick


if len(sys.argv) > 2 and sys.argv[2] == "--winner":
    print(winner())
    sys.exit(0)

print(read(f"{R}/machine-short.txt").strip())
print(read(f"{R}/toolchains.txt").strip())
print()
print("(a) Builds (sm_90a; ptxas -v): registers / spill bytes / C7515 (wgmma serialized) a kernel; SASS: HGMMAs each")
print("    followed by a wait for all (WARPGROUP.DEPBAR.LE gsb0, 0x0) out of all, in mma12_wgp_kernel<256> / mma12_tma_kernel<64>")
for lib in LIBS:
    log = read(f"{R}/build-{lib}.txt")
    if not log:
        continue
    ok = os.path.exists(f"{R}/lib-{lib}.ok")
    p = ptxas(log)
    cells = " ".join(f"{k} {p[k][0]}/{p[k][1]}/{'C7515' if p[k][2] else '-'}" for k, _ in KERNELS if k in p)
    sass = read(f"{R}/sass-{lib}.txt").strip()
    print(f"  {lib}: {'built' if ok else 'BUILD FAILED'} | {cells} | {sass}")
print()
print("(a) Hopper products (cu12_check.py: the self-test's matrices, 1-2100 tokens, bias and none): within 1e-2 of fp32")
print("    and the same every run; then bit for bit against cu13-base")
checks = {}
for lib in LIBS:
    try:
        checks[lib] = json.load(open(f"{R}/check-{lib}.json"))
    except (OSError, ValueError):
        continue
ref = checks.get("cu13-base")
for lib, c in checks.items():
    line = f"  {lib}: {c['product']} {len(c['sha'])} products, {len(c['bad'])} off"
    if ref and lib != "cu13-base":
        keys = set(c["sha"]) & set(ref["sha"])
        diff = sorted(k for k in keys if c["sha"][k] != ref["sha"][k])
        small = [k for k in diff if int(re.search(r"M=(\d+)", k).group(1)) <= 128]
        line += f"; against cu13-base: {len(keys) - len(diff)} of {len(keys)} identical"
        if diff:
            line += f" ({len(small)} of the differing at 128 tokens or fewer; e.g. {diff[0]})"
    print(line)
print()
a = by_lib(layer(f"{R}/layer-a.txt"))
if a:
    print("(a) Per product, kernel time / cuBLAS's (median over rounds; M 17-128 the TMA kernel, 129-1024 mma12_wgp_kernel)")
    models = list(dict.fromkeys(k[0] for d in a.values() for k in d))  # (in the order run)
    keys = sorted({k for d in a.values() for k in d}, key=lambda k: (models.index(k[0]), ["qkv", "o", "gate_up", "down"].index(k[1]) if k[1] in ("qkv", "o", "gate_up", "down") else 9, k[2]))
    libs = [l for l in LIBS if l in a]
    print("  " + "model product M: cuBLAS us | " + " | ".join(libs))
    for k in keys:
        c = [a[l][k][0] for l in libs if k in a[l]]
        print(f"  {k[0]} {k[1]} M={k[2]}: {med(c):.1f} | " + " | ".join(f"{a[l][k][1] / a[l][k][0]:.3f}x" if k in a[l] else "-" for l in libs))
    print("  Sums over each model's products above (time / cuBLAS's)")
    for model in models:
        for M in sorted({k[2] for k in keys if k[0] == model}):
            cells = []
            for l in libs:
                ks = [k for k in a[l] if k[0] == model and k[2] == M]
                cells.append(f"{sum(a[l][k][1] for k in ks) / sum(a[l][k][0] for k in ks):.3f}x" if ks else "-")
            print(f"  {model} M={M}: " + " | ".join(cells))
    for lo, hi in (("cu12-base", "cu12-fix"), ("cu13-base", "cu13-fix"), ("cu13-fix", "cu12-fix")):
        if lo in a and hi in a:
            ks = sorted(set(a[lo]) & set(a[hi]))
            r = [a[hi][k][1] / a[lo][k][1] for k in ks]
            print(f"  {hi} / {lo}, kernel time a product: median {med(r):.3f}, {min(r):.3f}-{max(r):.3f} over {len(r)}")
    print()
b = by_lib(layer(f"{R}/layer-b.txt"))
if b:
    print("(b) The whole-tiles rule: cu13-fixn1 against cu13-fix (library 4), kernel time / cuBLAS's (median over rounds)")
    ks = sorted(set(b.get("cu13-fix", {})) | set(b.get("cu13-fixn1", {})))
    for k in ks:
        f, n = b.get("cu13-fix", {}).get(k), b.get("cu13-fixn1", {}).get(k)
        cell = lambda v: f"{v[1] / v[0]:.3f}x" if v else "-"
        print(f"  {k[0]} {k[1]} M={k[2]}: fix {cell(f)} | fix + rule {cell(n)}" + (f" | rule/fix {n[1] / f[1]:.3f}" if f and n else ""))
    if "cu13-fix" in b and "cu13-fixn1" in b:
        common = set(b["cu13-fix"]) & set(b["cu13-fixn1"])
        print(f"  all: rule/fix {sum(b['cu13-fixn1'][k][1] for k in common) / sum(b['cu13-fix'][k][1] for k in common):.3f} over {len(common)} products")
    print()
e2e = read(f"{R}/e2e.txt") + read(f"{R}/e2e-exact.txt")
if e2e:
    print(f"(c) e2e with the winning library, {read(f'{R}/winner.txt').strip()} (the model: steps below)")
    for line in e2e.splitlines():
        if re.search(r"prefill:|fused: batch|exact: batch|bit-identical|identical to bf16", line):
            print("  " + line.strip()[:300])
    print()
print(read(f"{R}/steps.txt").strip())
