"""o3_job.sh's results in one page: python o3_summary.py RESULTS_DIR. First line: CHECKS PASS or FAIL (every step run
exited 0; check_capi's ring ran; the split tests and the stress passed; the ring ran in every e2e.py run). Then the
GPU, and whether SPLIT is the shipping route (a GH200) or forced on (a measurement, not a route); e2e.py's forward
pass and first token per model and length (bf16; v0.25.1's routes and SPLIT, the medians of the rounds; SPLIT over
v0.25.1 each round); layer.py per layer; the SM clock and power over each step and over each e2e.py phase (SPLIT,
v0.25.1); each step's exit. Last, a DECIDES line per model and length: SPLIT's forward pass over v0.25.1's, the median
of the rounds' ratios (each round's two phases run back to back); SPLIT stays where that is 0.980 or less (it beats
v0.25.1 by at least 2%) and is dropped where it is not."""
import datetime, glob, os, re, statistics, sys

R = sys.argv[1]
STAYS = 0.98  # SPLIT's time over v0.25.1's at or below which it stays


def lines(name):
    try:
        return open(os.path.join(R, name), errors="replace").read().splitlines()
    except OSError:
        return []


steps, bad = {}, []
for f in sorted(glob.glob(os.path.join(R, "*.txt")), key=os.path.getmtime):
    n = os.path.basename(f)[:-4]
    ls = lines(n + ".txt")
    m = ls and re.match(r"exit (-?\d+)$", ls[-1])
    if m:
        steps[n] = int(m.group(1))
        if steps[n]:
            bad.append(f"{n} exit {steps[n]}" + (f" ({ls[-2].strip()[:140]})" if len(ls) > 1 else ""))
capi = lines("check_capi.txt")
ring = next((l for l in capi if l.startswith(("the route SPLIT: ", "the route SPLIT cannot run"))), None)
if "check_capi" in steps and (ring is None or "cannot run" in ring):
    bad.append("check_capi: " + (ring or "no line on the route SPLIT"))
for n in ("check_capi", "test_split", "split_stress"):
    if n not in steps:
        bad.append(f"{n}: not run")
stress = next((l for l in lines("split_stress.txt") if l.startswith("split_stress:")), None)
route = (lines("route.txt") or ["SPLIT: (no route.txt)"])[0]
forced = "forced" in route
expect = lines("expect.txt")  # the models (their names) and the lengths the job asked for
models = expect[0].split() if expect else []
lengths = [int(x) for x in expect[1].split(",")] if len(expect) > 1 else []

# e2e.py: "<label> prefill: N tokens T ms (X tokens/s), first token F ms, ...; from A to B", label bf16, "glyd mma12"
# (SPLIT) or "glyd mma12 without SPLIT" (v0.25.1's routes), in the order run; "the route SPLIT's ring: cuda:0 ran"
e2e, windows, rings = {}, [], {}  # {model: {side: [{n: (ms, first token ms)}, ...]}}, [(model, side, a, b)], {model: line}
for f in sorted(glob.glob(os.path.join(R, "e2e-*.txt"))):
    model = os.path.basename(f)[4:-4]
    models += [model] if model not in models else []
    for l in lines(os.path.basename(f)):
        m = re.match(r"(.*) prefill: (.*?)(?:; from (\S+ \S+) to (\S+ \S+))?$", l)
        if m:
            side = "bf16" if m.group(1) == "bf16" else "v0251" if m.group(1).endswith("without SPLIT") else "split"
            e2e.setdefault(model, {}).setdefault(side, []).append({int(n): (float(t), float(ft)) for n, t, ft in re.findall(r"(\d+) tokens ([\d.]+) ms \(\d+ tokens/s\), first token ([\d.]+) ms", m.group(2))})
            if m.group(3):
                windows.append((model, side, m.group(3), m.group(4)))
        if l.startswith("the route SPLIT's ring: "):
            rings[model] = l[len("the route SPLIT's ring: "):]
    if model in e2e and "ran" not in rings.get(model, ""):
        bad.append(f"e2e-{model}: the ring {rings.get(model, 'state not logged')}")
layer = {}
for f in sorted(glob.glob(os.path.join(R, "layer-*.txt"))):
    name = os.path.basename(f)[6:-4]
    for l in lines(os.path.basename(f)):
        m = re.match(r"M=(\d+): a layer bf16 ([\d.]+) ms, today's route ([\d.]+) ms .*?SPLIT ([\d.]+) ms .*?routes: (.*?); the ring (.*?);", l)
        if m:
            layer.setdefault(name, {})[int(m.group(1))] = dict(bf16=float(m.group(2)), v0251=float(m.group(3)), split=float(m.group(4)), routes=m.group(5), ring=m.group(6))


def rounds(model, n):
    """[(SPLIT ms, v0.25.1 ms, SPLIT first token, v0.25.1 first token)] a round: the k-th phase each way (back to back)."""
    s, v = e2e.get(model, {}).get("split", []), e2e.get(model, {}).get("v0251", [])
    return [(a[n][0], b[n][0], a[n][1], b[n][1]) for a, b in zip(s, v) if n in a and n in b]


med = statistics.median
out = [f"CHECKS {'PASS' if not bad else 'FAIL'}: {len(steps)} steps run" + (f"; {'; '.join(bad)}" if bad else ", each exit 0")]
out += (lines("machine-short.txt") or [route])[:1] + lines("env.txt")[:3]
if ring:
    out.append(ring)
if stress:
    out.append(f"the stress: {stress}")
out += [f"e2e {m}, the route SPLIT's ring: {r}" for m, r in rings.items()]
out.append("")
for model in models:
    by = e2e.get(model, {})
    if not by:
        continue
    nr = min(len(by.get("split", [])), len(by.get("v0251", [])))
    out.append(f"e2e.py {model} (a forward pass ms, the first token ms): bf16 | v0.25.1's routes | SPLIT, the medians of {nr} rounds | SPLIT/v0.25.1, the median of the rounds' ratios (each round's)")
    for n in sorted({n for runs in by.values() for r in runs for n in r}):
        rs = rounds(model, n)
        b = by.get("bf16", [{}])[0].get(n)
        cells = [f"bf16 {b[0]:.1f} ({b[1]:.1f})" if b else "bf16 -"]
        if rs:
            s_ms, v_ms = med(r[0] for r in rs), med(r[1] for r in rs)
            ratio, ft = med(r[0] / r[1] for r in rs), med(r[2] / r[3] for r in rs)
            cells += [f"v0.25.1 {v_ms:.1f} ({med(r[3] for r in rs):.1f})", f"SPLIT {s_ms:.1f} ({med(r[2] for r in rs):.1f})",
                      f"SPLIT/v0.25.1 {ratio:.3f} (first token {ft:.3f}; rounds {', '.join(f'{r[0] / r[1]:.3f}' for r in rs)})"]
            if b:
                cells.append(f"SPLIT/bf16 {s_ms / b[0]:.3f}, v0.25.1/bf16 {v_ms / b[0]:.3f}")
        out.append(f"  {n:>5}: " + " | ".join(cells))
for name, by in layer.items():
    out.append(f"layer.py {name}, layer 10 (ms): bf16 | v0.25.1's route | SPLIT")
    for M, v in sorted(by.items()):
        out.append(f"  {M:>5}: {v['bf16']:.3f} | {v['v0251']:.3f} ({v['v0251'] / v['bf16']:.3f}x) | {v['split']:.3f} ({v['split'] / v['bf16']:.3f}x bf16, {v['split'] / v['v0251']:.3f}x v0.25.1's); {v['routes']}; the ring {v['ring']}")
out += ["  " + l.strip() for l in lines("split-kernel-ptxas.txt")]

# clocks and power: smi.csv's samples within each step's window and each e2e.py phase's
fields = (lines("smi-fields.txt") or ["timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu"])[0].split(",")
samples = []
for l in lines("smi.csv"):
    v = [x.strip() for x in l.split(",")]
    if len(v) != len(fields):
        continue
    try:
        samples.append((datetime.datetime.strptime(v[0], "%Y/%m/%d %H:%M:%S.%f"), float(v[1]), float(v[3]), float(v[4]), int(v[5], 16) if len(v) > 5 else 0))
    except ValueError:
        pass


def within(a, b):
    a, b = (datetime.datetime.strptime(x, "%Y/%m/%d %H:%M:%S") for x in (a, b))
    return [s for s in samples if a <= s[0] <= b + datetime.timedelta(seconds=1)]


def clocks(w):
    sm, pw = [s[1] for s in w], [s[2] for s in w]
    return (f"{statistics.mean(sm):.0f} ({min(sm):.0f}-{max(sm):.0f}) MHz, {statistics.mean(pw):.0f} ({max(pw):.0f}) W, {max(s[3] for s in w):.0f} C, "
            f"{100 * sum(1 for s in w if s[4] & 0x4) / len(w):.0f}% / {100 * sum(1 for s in w if s[4] & 0x60) / len(w):.0f}%")


if samples:
    out.append("")
    out.append(f"clocks and power ({len(samples)} samples): SM MHz mean (min-max), W mean (max), temperature max, samples power-capped (0x4) / thermal (0x20, 0x40)")
    for l in lines("windows.txt"):
        n, a, b = l.split("\t")
        w = within(a, b)
        if w:
            out.append(f"  step {n}: {clocks(w)}")
    for model in models:
        for side, what in (("split", "SPLIT"), ("v0251", "v0.25.1"), ("bf16", "bf16")):
            w = [s for mm, sd, a, b in windows if mm == model and sd == side for s in within(a, b)]
            if w:
                out.append(f"  e2e {model}, the {what} phases: {clocks(w)}")
out.append("")
out += [f"{n}: exit {e}" for n, e in steps.items()]
out += ["steps: " + l for l in lines("steps.txt")[-3:]]
out.append("")

# DECIDES, a line per model and length (the lengths asked, else those run)
tag = f" [{route[len('SPLIT '):] if route.startswith('SPLIT ') else route}]" if forced else ""
for model in models:
    for n in lengths or sorted({n for runs in e2e.get(model, {}).values() for r in runs for n in r}):
        rs, st = rounds(model, n), rings.get(model)
        why = "not run" if model not in e2e else "the ring's state not logged" if not st else f"the ring {st}" if "ran" not in st else None if rs else "no round of both ways"
        if why:
            out.append(f"DECIDES {model} {n} tokens: no numbers ({why}){tag}")
            continue
        ratio = med(r[0] / r[1] for r in rs)
        gain = 100 * (1 - ratio)
        verdict = f"SPLIT stays ({gain:.1f}% faster, at least 2%)" if ratio <= STAYS else f"SPLIT dropped ({gain:.1f}% faster, under 2%)" if gain > 0 else f"SPLIT dropped ({-gain:.1f}% slower)"
        out.append(f"DECIDES {model} {n} tokens: SPLIT/v0.25.1 {ratio:.3f}, a forward pass, the median of {len(rs)} rounds ({', '.join(f'{r[0] / r[1]:.3f}' for r in rs)}; "
                   f"SPLIT {med(r[0] for r in rs):.1f} ms, v0.25.1 {med(r[1] for r in rs):.1f} ms; first token {med(r[2] / r[3] for r in rs):.3f}): {verdict}{tag}")
print("\n".join(out))
