"""h100_val.sh's summary from what is in RESULTS so far: (a) each library's toolchain, ptxas's registers, spills and
C7515 a kernel, the SASS's serialized wgmma; (b) the self-test, and the mma_gemm_wg outputs bit for bit against each
other and against the h100f run's, where the committed rule takes base's split or fixn1's; (c) the products against
cuBLAS, marked where the committed split differs from fixn1's; (d) e2e; (e) 479139a's library on (8960, 128).
    python val_summary.py RESULTS
"""
import json, os, re, statistics, sys

R = sys.argv[1]
HERE = os.path.dirname(os.path.abspath(__file__))
LIBS = ["cu13", "cu12"]
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


def split(O, K, M, most, rule):
    """mma12_wgp_run's (nc, R, D, ncs) for a product past 128 tokens: rule committed (the cap on 479139a's), fixn1 (the
    h100f run's fifth library) or base (3add84a)."""
    NT = 192 if (M + 191) // 192 == (M + 255) // 256 else 256
    T, S = ((O // 64 + 1) // 2 + 1) // 2 * ((M + NT - 1) // NT), K // 64
    nc = max(1, min(most, T * S // 8))
    if rule != "base" and T < nc and 6 * (nc // T * T) >= 5 * nc:
        nc = nc // T * T
    r, d = T % nc, T - T % nc
    if rule == "committed":
        w = r * min(3, nc // r) if r else 0
        ncs = min(r * S, nc if not d else (w if 6 * w >= 5 * nc else min(nc, 3 * r)))
    elif rule == "fixn1":
        ncs = 0 if not r else (r * min(3, nc // r) if d else nc)
    else:
        ncs = min(nc, 3 * r) if d else nc
    return nc, r, d, ncs


def layer(path):
    """{(tag, model, product, O, K, M): (cuBLAS us, kernel us, check failed)}"""
    out = {}
    for line in read(path).splitlines():
        m = re.match(r"(\S+) (\S+) (\w+) (\d+)x(\d+) M=(\d+): cuBLAS ([\d.]+) us \| \w+ ([\d.]+) us", line)
        if m:
            out[(m.group(1), m.group(2), m.group(3), int(m.group(4)), int(m.group(5)), int(m.group(6)))] = (float(m.group(7)), float(m.group(8)), "CHECK FAILED" in line)
    return out


sms = int(read(f"{R}/sms.txt").strip() or 0)
most = sms // 2  # clusters of two, a block an SM (66 on an H100 SXM, as the h100f run's outputs showed: 22 of 22 splits)
print(read(f"{R}/machine-short.txt").strip())
print(read(f"{R}/toolchains.txt").strip())
print()
print("(a) Builds (sm_90a; ptxas -v): registers / spill bytes / C7515 (wgmma serialized) a kernel; SASS: HGMMAs each")
print("    followed by a wait for all (WARPGROUP.DEPBAR.LE gsb0, 0x0) out of all, in mma12_wgp_kernel<256> / mma12_tma_kernel<64>")
for lib in LIBS + ["old"]:
    log = read(f"{R}/build-{lib}.txt")
    if log:
        p = ptxas(log)
        cells = " ".join(f"{k} {p[k][0]}/{p[k][1]}/{'C7515' if p[k][2] else '-'}" for k, _ in KERNELS if k in p)
        print(f"  {lib}: {'built' if os.path.exists(f'{R}/lib-{lib}.ok') else 'BUILD FAILED'} | {cells} | {read(f'{R}/sass-{lib}.txt').strip()}")
print()
checks = {}
for lib in LIBS:
    st = read(f"{R}/selftest-{lib}.txt")
    if not st:
        continue
    if not checks:
        print("(b) The self-test (glyd_gpu.py) through each library; its mma_gemm_wg outputs (val_check.py: the self-test's")
        print("    matrices, 1-2100 tokens, bias and none) within 1e-2 of fp32, the same every run, and bit for bit")
    ok = len(re.findall(r"within 1e-2, the same every run", st))
    fail = [l for l in st.splitlines() if "AssertionError" in l or "Error" in l][:2]
    print(f"  {lib} self-test: {ok} product lines passed, {st.strip().splitlines()[-1]}" + (f" | {' | '.join(fail)}" if fail else ""))
    try:
        checks[lib] = json.load(open(f"{R}/check-{lib}.json"))
        c = checks[lib]
        print(f"  {lib} check: {c['product']} {len(c['sha'])} products, {len(c['bad'])} off" + (f": {c['bad'][:3]}" if c["bad"] else ""))
    except (OSError, ValueError):
        pass
if "cu13" in checks and "cu12" in checks:
    a, b = checks["cu13"]["sha"], checks["cu12"]["sha"]
    same = sum(a[k] == b[k] for k in a if k in b)
    print(f"  cu12 against cu13: {same} of {len(set(a) & set(b))} identical")
prev = {n: json.load(open(f"{HERE}/prev-check-cu13-{n}.json"))["sha"] for n in ("base", "fixn1") if os.path.exists(f"{HERE}/prev-check-cu13-{n}.json")}
if "cu13" in checks and len(prev) == 2 and sms == 132 and checks["cu13"]["product"] == "mma_gemm_wg":
    # The h100f run (an H100 SXM, the same inputs): each product's expected bits are base's where the committed rule
    # takes base's split (or no split: 128 tokens or fewer), fixn1's where it takes fixn1's.
    got, tally, off, new = checks["cu13"]["sha"], {}, [], 0
    for key, sha in got.items():
        m = re.match(r"(\d+)x(\d+) M=(\d+)", key)
        O, K, M = map(int, m.groups())
        if key not in prev["base"]:
            new += 1  # (the matrix the h100f run did not have)
            continue
        c = split(O, K, M, most, "committed")
        src = "base" if M <= 128 or c == split(O, K, M, most, "base") else "fixn1" if c == split(O, K, M, most, "fixn1") else None
        if src is None:
            tally.setdefault("a split neither had", [0, 0])[1] += 1
            continue
        tally.setdefault(src, [0, 0])[0] += prev[src][key] == sha
        tally[src][1] += 1
        if prev[src][key] != sha:
            off.append(key)
    print("  cu13 against the h100f run's outputs where the committed rule takes base's split (or none) or fixn1's: "
          + "; ".join(f"{s}'s {h} of {n} identical" for s, (h, n) in sorted(tally.items())) + (f" (off: {off[:4]})" if off else "")
          + f"; the new matrix's {new} were not in that run")
if checks:
    print()
runs = layer(f"{R}/layer.txt")
if runs:
    acc = {}
    for (tag, model, prod, O, K, M), (c, t, bad) in runs.items():
        acc.setdefault((model, prod, O, K, M), {}).setdefault(tag.rsplit("-r", 1)[0], []).append((c, t, bad))
    order = ["qkv", "o", "gate_up", "down"]
    print(f"(c) Per product, kernel time / cuBLAS's (median of rounds); split past 128 tokens: nc / R / D / ncs on {most} clusters,")
    print("    * where the committed rule's differs from fixn1's (the h100f run's fifth library), + where from 3add84a's")
    print("  model product M: cuBLAS us | " + " | ".join(LIBS) + " | split")
    models = list(dict.fromkeys(k[0] for k in acc))
    for k in sorted(acc, key=lambda k: (models.index(k[0]), order.index(k[1]) if k[1] in order else 9, k[4])):
        model, prod, O, K, M = k
        cb = statistics.median(c for v in acc[k].values() for c, _, _ in v)
        cells = []
        for lib in LIBS:
            v = acc[k].get(lib)
            cells.append(f"{statistics.median(t for _, t, _ in v) / statistics.median(c for c, _, _ in v):.3f}x" + (" CHECK FAILED" if any(b for _, _, b in v) else "") if v else "-")
        sp = ""
        if M > 128:
            c, f, b = split(O, K, M, most, "committed"), split(O, K, M, most, "fixn1"), split(O, K, M, most, "base")
            sp = "/".join(map(str, c)) + (" *" if c != f else "") + (" +" if c != b else "")
        print(f"  {model} {prod} M={M}: {cb:.1f} | " + " | ".join(cells) + f" | {sp}")
    print("  Sums over each model's products above (time / cuBLAS's)")
    for model in models:
        for M in sorted({k[4] for k in acc if k[0] == model}):
            cells = []
            for lib in LIBS:
                ks = [k for k in acc if k[0] == model and k[4] == M and lib in acc[k]]
                t = sum(statistics.median(x for _, x, _ in acc[k][lib]) for k in ks)
                c = sum(statistics.median(x for x, _, _ in acc[k][lib]) for k in ks)
                cells.append(f"{t / c:.3f}x" if ks else "-")
            print(f"  {model} M={M}: " + " | ".join(cells))
    print()
for lib in LIBS:
    e2e = read(f"{R}/e2e-{lib}.txt") + read(f"{R}/e2e-exact-{lib}.txt")
    if e2e:
        print(f"(d) e2e, the {lib} library:")
        for line in e2e.splitlines():
            if re.search(r"prefill:|fused: batch|exact: batch|bit-identical|identical to bf16|^exit|Error", line):
                print("  " + line.strip()[:300])
        print()
e = read(f"{R}/e-479139a.txt")
if e:
    print("(e) 479139a's library (no cap), (8960, 128):")
    print("\n".join("  " + l for l in e.strip().splitlines()[-6:]))
    print()
print(read(f"{R}/steps.txt").strip())
