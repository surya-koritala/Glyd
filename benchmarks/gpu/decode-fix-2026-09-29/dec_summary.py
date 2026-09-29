"""h100_dec.sh's results in one page: python dec_summary.py RESULTS_DIR. First line: BITS PASS or FAIL (every step
run exited 0; the self-test and xcheck.py with each load order; dec_time.py's checks; dec_e2e.py's sha256 lines the
same in every variant and tree); then the decode kernel's time a layer by variant (dec_time.py: main's, v0.25.0's and
the load orders', each over main's), end to end (dec_e2e.py), the Nsight Compute profile's main lines, and the steps."""
import csv, glob, os, re, sys

R = sys.argv[1]


def lines(name):
    try:
        return open(os.path.join(R, name), errors="replace").read().splitlines()
    except OSError:
        return []


bad, steps = [], {}
for f in sorted(glob.glob(os.path.join(R, "*.txt"))):
    n = os.path.basename(f)[:-4]
    ls = lines(n + ".txt")
    m = ls and re.match(r"exit (-?\d+)$", ls[-1])
    if m:
        steps[n] = int(m.group(1))
        if steps[n] and not n.startswith("prof"):
            bad.append(f"{n} exit {steps[n]}")
        if n.startswith("selftest") and not any("every bf16 bit pattern" in l for l in ls):
            bad.append(f"{n}: no 'every bf16 bit pattern'")
        if n.startswith("xcheck") and not any(l.startswith("all the same bits") for l in ls):
            bad.append(f"{n}: no 'all the same bits'")
        if n.startswith("time-") and not any(l.startswith("bits: every variant") for l in ls):
            bad.append(f"{n}: " + next((l for l in ls if l.startswith("bits:")), "no bits line"))
sha = {}
for f in sorted(glob.glob(os.path.join(R, "e2e-*.txt"))):
    tree = os.path.basename(f)[4:-4]
    for l in lines(os.path.basename(f)):
        m = re.match(r"(exact|fused) (as built|order \d): (.*) sha256 (\w+)", l)
        if m:
            sha.setdefault((m.group(1), m.group(3)), {})[f"{tree} {m.group(2)}"] = m.group(4)
differ = [f"{k[0]} {k[1]}: " + ", ".join(f"{v} {h}" for v, h in d.items()) for k, d in sha.items() if len(set(d.values())) > 1]
bad += [f"e2e sha256 differ: {x}" for x in differ]
out = [f"BITS {'PASS' if steps and not bad else 'FAIL'}: {len(steps)} steps run" + (f"; {'; '.join(bad)}" if bad else ", each exit 0")
       + (f"; e2e: {sum(len(d) for d in sha.values())} sha256 lines, {len(sha)} outputs, the same in every variant and tree" if sha and not differ else "")]
out += lines("machine-short.txt")[:1] + lines("env.txt")[:2]
out += ["", "The decode a layer (dec_time.py; median of its repetitions), us, each over main's; whole: a warp a step (for cuBLAS,",
        "exact mode), ahead: 2 warps an SM (decode ahead). v0.25.0: its library; order N: this tree's, GLYD_DEC_ORDER=N (0 v0.25.0's kernel)."]
for f in sorted(glob.glob(os.path.join(R, "time-*.txt"))):
    b = os.path.basename(f)
    model, run = re.match(r"time-(.+)-run(\d+)\.txt", b).groups()
    for l in lines(b):
        m = re.match(r"\s+(whole|ahead)( \(\d+ warps\))? layer: (.*?)\s+\[GB/s: (.*)\]", l)
        if m:
            cells = re.findall(r"(main|v0\.25\.0|order \d) ([\d.]+) us(?: \(([\d.]+) main)?", m.group(3))
            out.append(f"  {model} run {run} {m.group(1)}: " + " | ".join(f"{v} {t}" + (f" ({r})" if r else "") for v, t, r in cells)
                       + f"  [GB/s {m.group(4)}]")
out += ["", "End to end (dec_e2e.py), each over main's: exact mode's steps (ms a token) and prompts (ms)"]
e2e = {}
for f in sorted(glob.glob(os.path.join(R, "e2e-*.txt"))):
    tree = os.path.basename(f)[4:-4]
    for l in lines(os.path.basename(f)):
        m = re.match(r"(exact|fused) (as built|order \d): (steps|prompt \d+ tokens) ([\d.]+) ms", l)
        if m:
            e2e.setdefault((m.group(1), m.group(3)), {})[tree if m.group(2) == "as built" else m.group(2)] = float(m.group(4))
rank = lambda v: ["main", "main2", "v0.25.0"].index(v) if v in ("main", "main2", "v0.25.0") else 3 + int(v[-1]) if v.startswith("order") else 9
for (mode, what), d in e2e.items():
    base = d.get("main")
    out.append(f"  {mode} {what}: " + " | ".join(f"{v} {d[v]:.3f}" + (f" ({d[v] / base:.3f})" if base and v != "main" else "") for v in sorted(d, key=rank)))
raw = os.path.join(R, "ncu", "raw.csv")
prof = lines("prof.txt")
rows = list(csv.reader(open(raw, errors="replace"))) if os.path.exists(raw) else []
head = next((i for i, r in enumerate(rows) if "ID" in r and "Kernel Name" in r), None)
if head is not None and rows[head + 2:]:
    cols, units, data = rows[head], rows[head + 1], rows[head + 2:]
    who = [re.sub(r"^launch \d+: ", "", l).split(",")[0] for l in prof if l.startswith("launch ")]
    get = lambda r, c: r[cols.index(c)] if c in cols and cols.index(c) < len(r) else ""
    out += ["", "Nsight Compute (dec_prof.py: one matrix, each variant once): time, DRAM throughput, bytes read and written,",
            "achieved occupancy, registers, and the most warp-cycles an issue by stall reason"]
    for i, r in enumerate(data):
        stalls = sorted(((float(get(r, c) or 0), re.sub(r"smsp__average_warps_issue_stalled_(\w+?)_per_issue_active\.ratio", r"\1", c))
                         for c in cols if re.match(r"smsp__average_warps_issue_stalled_\w+_per_issue_active\.ratio", c)), reverse=True)
        unit = lambda c: units[cols.index(c)] if c in cols and cols.index(c) < len(units) else ""
        cells = [f"{what} {get(r, c)} {unit(c)}".rstrip() for what, c in (
            ("time", "gpu__time_duration.sum"), ("DRAM", "dram__throughput.avg.pct_of_peak_sustained_elapsed"),
            ("read", "dram__bytes_read.sum"), ("written", "dram__bytes_write.sum"),
            ("occupancy", "sm__warps_active.avg.pct_of_peak_sustained_active"), ("registers", "launch__registers_per_thread")) if get(r, c)]
        out.append(f"  {who[i] if i < len(who) else '?'}: " + ", ".join(cells) + ("; stalls: " + ", ".join(f"{n} {v:.2f}" for v, n in stalls[:5]) if stalls else ""))
elif prof:
    err = next((l for l in prof if "ERR_NVGPUCTRPERM" in l or "rror" in l), prof[-1])
    out += ["", f"Nsight Compute: no profile ({err.strip()[:200]})"]
elif os.path.isdir(os.path.join(R, "ncu")) or any("ncu:" in l for l in lines("steps.txt")):
    out += ["", "Nsight Compute: " + next((l.split("ncu", 1)[1] for l in lines("steps.txt") if " ncu" in l), " no profile")]
out += ["", "steps:"] + ["  " + l for l in lines("steps.txt")]
print("\n".join(out))
