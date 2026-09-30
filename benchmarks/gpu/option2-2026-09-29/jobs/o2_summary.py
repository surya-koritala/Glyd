"""o2_job.sh's results in one page: python o2_summary.py RESULTS_DIR. First line: CHECKS PASS or FAIL (every step run
exited 0; check_capi's ring ran; the split tests passed); then what the numbers decide (e2e.py's forward pass, the route
SPLIT over today's route, each model and length; the last session's scheduling and the SM sweep beside it; the
breakdown's GPU time by kind, with SPLIT against without; layer.py per layer); then the tables, the SM clock and power
over each step (smi.csv), and each step's exit."""
import datetime, glob, os, re, statistics, sys

R = sys.argv[1]


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
        if steps[n] and n != "split_stress_old":  # (the stress on the tree before the fix: its failures the check's proof)
            bad.append(f"{n} exit {steps[n]}" + (f" ({ls[-2].strip()[:140]})" if len(ls) > 1 else ""))
capi = lines("check_capi.txt")
ring = next((l for l in capi if l.startswith(("the route SPLIT: ", "the route SPLIT cannot run"))), None)
if "check_capi" in steps and (ring is None or "cannot run" in ring):
    bad.append("check_capi: " + (ring or "no line on the route SPLIT"))
for n in ("check_capi", "test_split", "split_stress"):
    if n not in steps:
        bad.append(f"{n}: not run")
stress = {n: next((l for l in lines(n + ".txt") if l.startswith("split_stress:")), None) for n in ("split_stress", "split_stress_old")}
ran = len(steps) - ("split_stress_old" in steps)
out = [f"CHECKS {'PASS' if not bad else 'FAIL'}: {ran} steps run" + (f"; {'; '.join(bad)}" if bad else ", each exit 0") + (" (and the stress on the tree before the fix)" if "split_stress_old" in steps else "")]
out += lines("machine-short.txt")[:1] + lines("env.txt")[:3]
if ring:
    out.append(ring)
if stress["split_stress"]:
    out.append(f"the stress, this tree: {stress['split_stress']}")
if stress["split_stress_old"] or "split_stress_old" in steps:
    out.append(f"the stress, the tree before the fix: {stress['split_stress_old'] or 'no summary line'} (it should fail there)")
    out += ["  " + l.strip() for l in lines("split_stress_old.txt") if "the recording" in l][:3]

# e2e.py: "<label> prefill: N tokens T ms (X tokens/s), first token F ms, ..." and "<label> breakdown, N tokens, MODE: ..."
e2e, brk = {}, {}  # {(run, model): {n: {side: (ms, first token ms)}}}, {(run, model): {(n, mode): {...}}}
for f in sorted(glob.glob(os.path.join(R, "e2e-*.txt"))):
    base = os.path.basename(f)[4:-4]
    m = re.match(r"(slots3|sms\d+)-(.*)$", base)
    key = (m.group(1), m.group(2)) if m else ("", base)
    for l in lines(os.path.basename(f)):
        m = re.match(r"(.*) prefill: (.*)$", l)
        if m:
            side = "bf16" if m.group(1) == "bf16" else "today" if m.group(1).endswith("without SPLIT") else "split"
            for n, t, ft in re.findall(r"(\d+) tokens ([\d.]+) ms \(\d+ tokens/s\), first token ([\d.]+) ms", m.group(2)):
                e2e.setdefault(key, {}).setdefault(int(n), {})[side] = (float(t), float(ft))
        m = re.match(r".* breakdown, (\d+) tokens, (SPLIT|without SPLIT): a pass ([\d.]+) ms, the host issues it in ([\d.]+); GPU span ([\d.]+) ms, idle ([\d.]+); "
                     r"kernels: GEMMs ([\d.]+) ms, the decode ([\d.]+) \(beside GEMMs ([\d.]+), beside the rest ([\d.]+), alone ([\d.]+)\), attention ([\d.]+), the rest ([\d.]+)", l)
        if m:
            v = [float(x) for x in m.groups()[2:]]
            brk.setdefault(key, {})[int(m.group(1)), m.group(2)] = dict(zip(("pass", "host", "span", "idle", "gemm", "dec", "dec_gemm", "dec_rest", "dec_alone", "attn", "rest"), v))
layer = {}
for f in sorted(glob.glob(os.path.join(R, "layer*.txt"))):
    name = os.path.basename(f)[:-4]
    for l in lines(os.path.basename(f)):
        m = re.match(r"M=(\d+): a layer bf16 ([\d.]+) ms, today's route ([\d.]+) ms .*?SPLIT ([\d.]+) ms .*?routes: (.*?); the ring (.*?);", l)
        if m:
            layer.setdefault(name, {})[int(m.group(1))] = dict(bf16=float(m.group(2)), today=float(m.group(3)), split=float(m.group(4)), routes=m.group(5), ring=m.group(6))


def verdict(r):
    return "faster" if r < 0.99 else "slower" if r > 1.01 else "even"


dec = []
for (run, model), by in e2e.items():
    if run:
        continue
    parts = [f"{n} {s['split'][0] / s['today'][0]:.3f}x ({verdict(s['split'][0] / s['today'][0])})" for n, s in sorted(by.items()) if "split" in s and "today" in s]
    if parts:
        dec.append(f"e2e {model}, a forward pass by the route SPLIT over today's: " + ", ".join(parts))
    b = [f"{n} {s['split'][0] / s['bf16'][0]:.3f}x" for n, s in sorted(by.items()) if "split" in s and "bf16" in s]
    if b:
        dec.append(f"e2e {model}, SPLIT over bf16: " + ", ".join(b))
for (run, model), by in e2e.items():
    main = e2e.get(("", model), {})
    if run == "slots3":
        parts = [f"{n} {s['split'][0]:.1f} ms against the gated {main[n]['split'][0]:.1f}" for n, s in sorted(by.items()) if "split" in s and "split" in main.get(n, {})]
        dec.append(f"e2e {model}, the last session's scheduling (3 slots): " + ", ".join(parts))
    elif run.startswith("sms"):
        parts = [f"{n} {s['split'][0] / main[n]['split'][0]:.3f}x" for n, s in sorted(by.items()) if "split" in s and "split" in main.get(n, {})]
        dec.append(f"e2e {model}, the decode on {run[3:]} SMs over the route's: " + ", ".join(parts))
for (run, model), by in brk.items():
    for (n, mode), v in sorted(by.items()):
        if mode == "SPLIT":
            w = by.get((n, "without SPLIT"))
            rest = f", the rest {v['rest'] + v['attn']:.1f} ms against {w['rest'] + w['attn']:.1f} without" if w else ""
            dec.append(f"breakdown {model}{' (' + run + ')' if run else ''} {n}: decode beside the rest {v['dec_rest']:.1f} ms of {v['dec']:.1f}{rest}, idle {v['idle']:.1f}, host {v['host']:.1f} of {v['pass']:.1f}")
for name, by in layer.items():
    dec.append(f"per layer {name[6:]}, SPLIT over today's: " + ", ".join(f"{M} {v['split'] / v['today']:.3f}x" for M, v in sorted(by.items())))
out.append("DECIDES: " + (" | ".join(dec) if dec else "no numbers yet"))
out.append("")

for (run, model), by in e2e.items():
    out.append(f"e2e.py {model}{' ' + run if run else ''} (a forward pass ms; the first token ms), bf16 | today's route | the route SPLIT:")
    for n, s in sorted(by.items()):
        b, t, sp = s.get("bf16"), s.get("today"), s.get("split")
        cell = lambda v, name: f"{name} {v[0]:.1f} ({v[1]:.1f})" if v else f"{name} -"
        rel = ""
        if t and sp:
            rel = f" | SPLIT/today {sp[0] / t[0]:.3f} (first token {sp[1] / t[1]:.3f})" + (f", SPLIT/bf16 {sp[0] / b[0]:.3f}, today/bf16 {t[0] / b[0]:.3f}" if b else "")
        out.append(f"  {n:>5}: {cell(b, 'bf16')}, {cell(t, 'today')}, {cell(sp, 'SPLIT')}{rel}")
for (run, model), by in brk.items():
    out.append(f"breakdown {model}{' ' + run if run else ''} (ms): a pass, the host's issue; GPU span, idle; GEMMs; the decode (beside GEMMs / the rest / alone); attention; the rest")
    for (n, mode), v in sorted(by.items()):
        out.append(f"  {n:>5} {mode:>13}: {v['pass']:.1f}, {v['host']:.1f}; {v['span']:.1f}, {v['idle']:.1f}; {v['gemm']:.1f}; {v['dec']:.1f} ({v['dec_gemm']:.1f} / {v['dec_rest']:.1f} / {v['dec_alone']:.1f}); {v['attn']:.1f}; {v['rest']:.1f}")
for name, by in layer.items():
    out.append(f"{name} (a layer, ms): bf16 | today's route | the route SPLIT")
    for M, v in sorted(by.items()):
        out.append(f"  {M:>5}: {v['bf16']:.3f} | {v['today']:.3f} ({v['today'] / v['bf16']:.3f}x) | {v['split']:.3f} ({v['split'] / v['bf16']:.3f}x bf16, {v['split'] / v['today']:.3f}x today's); {v['routes']}; the ring {v['ring']}")
out += ["  " + l for l in lines("split-kernel-ptxas.txt")]

# clocks and power by step: smi.csv's samples within each step's window
fields = (lines("smi-fields.txt") or ["timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu"])[0].split(",")
samples = []
for l in lines("smi.csv"):
    v = [x.strip() for x in l.split(",")]
    if len(v) != len(fields):
        continue
    try:
        t = datetime.datetime.strptime(v[0], "%Y/%m/%d %H:%M:%S.%f")
        samples.append((t, float(v[1]), float(v[3]), float(v[4]), int(v[5], 16) if len(v) > 5 else 0))
    except ValueError:
        pass
if samples:
    out.append("")
    out.append(f"clocks and power ({len(samples)} samples): step: SM MHz mean (min-max), W mean (max), temperature max, samples power-capped (0x4) / thermal (0x20, 0x40)")
    for l in lines("windows.txt"):
        n, a, b = l.split("\t")
        a, b = (datetime.datetime.strptime(x, "%Y/%m/%d %H:%M:%S") for x in (a, b))
        w = [s for s in samples if a <= s[0] <= b + datetime.timedelta(seconds=1)]
        if w:
            sm, pw = [s[1] for s in w], [s[2] for s in w]
            cap = sum(1 for s in w if s[4] & 0x4) / len(w)
            hot = sum(1 for s in w if s[4] & 0x60) / len(w)
            out.append(f"  {n}: {statistics.mean(sm):.0f} ({min(sm):.0f}-{max(sm):.0f}) MHz, {statistics.mean(pw):.0f} ({max(pw):.0f}) W, {max(s[3] for s in w):.0f} C, {100 * cap:.0f}% / {100 * hot:.0f}%")
out.append("")
out += [f"{n}: exit {e}" for n, e in steps.items()]
out += ["steps: " + l for l in lines("steps.txt")[-3:]]
print("\n".join(out))
