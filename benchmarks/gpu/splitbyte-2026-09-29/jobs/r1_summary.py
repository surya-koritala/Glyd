"""r1_job.sh's results in one page: python r1_summary.py RESULTS_DIR. First line: CHECKS PASS or FAIL (every step run
exited 0; xcheck's bits the same; e2e12's lines the same as main's; the Rust saves byte-identical to Python's; this
device's routes as expected); then the decoder test's own first line (results/dec, its PASS or FAIL), each step's exit
with the check's last words, and layer.py's ratios (split byte's time over main's 12-bit layout's, by the library's
route and by kernel; the attribution's, attr-*, beside them). sb_summary.py's page, with the release's steps and the
decoder test."""
import glob, os, re, sys

R = sys.argv[1]


def lines(name, root=R):
    try:
        return open(os.path.join(root, name), errors="replace").read().splitlines()
    except OSError:
        return []


def last(ls, pat):
    return next((l for l in reversed(ls) if re.search(pat, l)), None)


out, bad = [], []
steps = {}
for f in sorted(glob.glob(os.path.join(R, "*.txt"))):
    n = os.path.basename(f)[:-4]
    ls = lines(n + ".txt")
    m = ls and re.match(r"exit (-?\d+)$", ls[-1])
    if m:
        steps[n] = int(m.group(1))
        if steps[n]:
            bad.append(f"{n} exit {steps[n]}")
for n in steps:
    ls = lines(n + ".txt")
    if n.startswith("xcheck") and not last(ls, r"^all the same bits"):
        bad.append(f"{n}: no 'all the same bits'")
    if n == "rust-pack" and not last(ls, r"^Rust: all passed"):
        bad.append("rust-pack: no 'Rust: all passed'")
    if n == "routes" and not last(ls, r"^routes: as expected"):
        bad.append("routes: not as expected")
cmp = lines("e2e12-compare.txt")
if any("DIFFER" in l for l in cmp):
    bad.append("e2e12: " + "; ".join(l for l in cmp if "DIFFER" in l))
out.append(f"CHECKS {'PASS' if steps and not bad else 'FAIL'}: {len(steps)} steps run" + (f"; {'; '.join(bad)}" if bad else ", each exit 0"))
out += lines("machine-short.txt")[:1] + lines("env.txt")[:3]
dec = lines("summary.txt", os.path.join(R, "dec"))
if dec or os.path.isdir(os.path.join(R, "dec")):
    out.append(f"decoder test (option 2, results/dec): {dec[0] if dec else 'no summary yet'}")
out.append("")
words = {"selftest": r"every bf16 bit pattern|Error|error", "check_capi": r"calls compared|Error|error", "test_gpu": r"ok$|FAIL|Error",
         "check_api-dense": r"check_api: all passed|Error|assert", "check_api-moe": r"check_api: all passed|Error|assert",
         "routes": r"^routes:|Error", "rust-tests": r"test result|error|panicked", "rust-pack": r"Rust: all passed|FAIL", "attr": r"no attribution"}
for n, e in steps.items():
    ls = lines(n + ".txt")
    pat = words.get(n, r"all the same bits|Error|error|assert")
    w = last(ls[:-1], pat) if n.startswith(("selftest", "check_", "test_gpu", "xcheck", "routes", "rust", "attr")) else None
    out.append(f"{n}: exit {e}" + (f" | {w.strip()[:160]}" if w else ""))
    if n == "rust-pack":
        out += ["  " + l for l in ls if "byte-identical" in l or "DIFFERS" in l]
    if n == "routes":
        out += ["  " + l for l in ls if l.startswith(("  tiered", "  12-bit")) or "code" in l][:4]
out += [""] + cmp
rows = []
for f in sorted(glob.glob(os.path.join(R, "layer-*-run*.txt"))) + sorted(glob.glob(os.path.join(R, "attr-*.txt"))):
    b = os.path.basename(f)
    model, run = re.match(r"layer-(.+)-run(\d+)\.txt", b).groups() if b.startswith("layer-") else (b[:-4], "-")
    for l in lines(b):
        m = re.match(r"\s+(decode|M=\s*(\d+)) \(([^)]*)\): (.*?)\s+\[", l)
        if m:
            ratios = re.findall(r"(\w+)\s+[\d.]+ us, split byte\s+[\d.]+ \(([\d.]+)x\)", m.group(4))
            rows.append((model, run, m.group(2) or "decode", m.group(3), ratios))
if rows:
    out += ["", "layer.py: split byte's time over main's 12-bit layout's (model, run, M, the route taken: by route | by kernel); attr-loop:",
            "main-once's over main's (the loop an entry a pass alone), attr-splitbyte: the branch's over main-once's (split byte; since aa7a6fc its loop main's)"]
    for model, run, M, route, ratios in rows:
        out.append(f"  {model} run {run} {M:>6} ({route}): " + " | ".join(f"{k} {r}" for k, r in ratios))
    rr = [float(r) for model, run, M, route, ratios in rows if run != "-" for k, r in ratios if k == "route"]
    if rr:
        out.append(f"  by the library's route: {min(rr):.3f}-{max(rr):.3f} over {len(rr)} rows")
out += ["", "steps:"] + ["  " + l for l in lines("steps.txt")]
print("\n".join(out))
