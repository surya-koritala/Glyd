"""resp_job.sh's results in one page: python resp_summary.py RESULTS_DIR [--json OUT]. Each model's table, bf16 against
Glyd (its default, compiled where fast_generate takes the call, and exact=True): time to first token at each prompt
length, tokens a second at each batch, the two mixes (time to first token and total), each the median of its repeats
(the warm-up apart); the load (GB, seconds, layout), which calls ran compiled, the first calls (the compile), and
whether each mode's greedy tokens are bf16's; the GPU's clocks and power while each mode ran (smi-*.csv). --json: every
result in one file."""
import csv, glob, json, os, statistics, sys

R = sys.argv[1]
res = {}
for f in sorted(glob.glob(os.path.join(R, "*.json"))):
    try:
        r = json.load(open(f))
    except ValueError:
        continue
    if "mode" in r and "model" in r:
        res.setdefault(os.path.basename(r["model"].rstrip("/")) if "snapshots" not in r["model"] else r["model"].split("models--")[-1].split("/")[0].split("--")[-1], {})[r["mode"]] = r
MODES = ["bf16", "glyd", "exact"]


def smi(model, mode):
    try:
        rows = [r for r in csv.reader(open(os.path.join(R, f"smi-{model}-{mode}.csv"))) if len(r) >= 5]
    except OSError:
        return ""
    num = lambda i: [float(r[i].split()[0]) for r in rows if r[i].split()[0].replace(".", "", 1).isdigit()]
    sm, pw = num(1), num(3)
    busy = [s for s, u in zip(sm, num(5) or [100] * len(sm)) if u > 50] or sm
    return f"SM {statistics.median(busy):.0f} MHz, {statistics.median(pw):.0f} W (medians while busy)" if busy and pw else ""


out, first = [], None
for model, modes in res.items():
    first = first or next(iter(modes.values()))
    head = [m for m in MODES if m in modes]
    lab = {"bf16": "bf16", "glyd": "Glyd (default)", "exact": "Glyd exact"}
    out += [f"## {model}", ""]
    for m in head:
        L = modes[m]["load"]
        if not L.get("fits"):
            out.append(f"- {lab[m]}: does not fit ({L.get('error', '')[:120]})")
        else:
            out.append(f"- {lab[m]}: {L['gb']} GB on the GPU after the load ({L['seconds']} s)" + (f", layout {L['layout']}" if L.get("layout") else "")
                       + (f", generate() compiled to {L['compiled_cap']} positions" if L.get("compiled_cap") else "") + (f"; {smi(model, m)}" if smi(model, m) else ""))
    cfgs = []
    for m in head:
        for c in modes[m].get("configs", []):
            if c["name"] not in cfgs:
                cfgs.append(c["name"])
    rows = []
    get = lambda m, n: next((c for c in modes[m].get("configs", []) if c["name"] == n), {})
    for n in cfgs:
        c0 = next(get(m, n) for m in head if get(m, n))
        what = {"ttft": f"time to first token, {c0['prompt']}-token prompt", "rate": f"tokens/s, {c0['batch']} sequence{'s' if c0['batch'] > 1 else ''} ({c0['new']} new)",
                "mix": f"{n.split()[1]}: {c0['prompt']}-token prompt, {c0['new']}-token reply"}[n.split()[0]]
        cells = []
        for m in head:
            c = get(m, n)
            if not c or c.get("skipped"):
                cells.append(f"({c.get('skipped', 'not run')[:40]})" if c else "")
                continue
            if n.startswith("rate"):
                v, b = c.get("tokens_per_s"), get("bf16", n).get("tokens_per_s")
                cells.append(f"{v:.1f}" + (f" ({v / b:.2f}x)" if b and m != "bf16" else "") + (" c" if c.get("path") == "compiled" else "") if v else "")
            elif n.startswith("ttft"):
                v, b = c.get("ttft"), get("bf16", n).get("ttft")
                cells.append(f"{v * 1e3:.0f} ms" + (f" ({v / b:.2f}x)" if b and m != "bf16" else "") + (" c" if c.get("path") == "compiled" else "") if v else "")
            else:
                v, t = c.get("ttft"), c.get("total")
                cells.append(f"{v * 1e3:.0f} ms / {t:.2f} s" + (" c" if c.get("path") == "compiled" else "") if v else "")
        rows.append((what, cells))
    out += ["", "| | " + " | ".join(lab[m] for m in head) + " |", "| :-- |" + " --: |" * len(head)]
    out += [f"| {w} | " + " | ".join(cells) + " |" for w, cells in rows]
    warm = []
    for m in head:
        cs = [c for c in modes[m].get("configs", []) if "warmup" in c]
        if cs:
            warm.append(f"{lab[m]} {cs[0]['warmup']['total']:.1f} s ({cs[0]['name']}; median repeat {cs[0].get('total', float('nan')):.2f} s)")
    same = []
    for m in head:
        if m != "bf16" and "bf16" in modes:
            both = [(c, get("bf16", c["name"])) for c in modes[m].get("configs", []) if c.get("sha") and get("bf16", c["name"]).get("sha")]
            if both:
                same.append(f"{lab[m]} {sum(a['sha'] == b['sha'] for a, b in both)} of {len(both)}")
    out += ["", "c: that call ran compiled (fast_generate). The first call of each mode, the compile's in the default mode: " + "; ".join(warm) + "."]
    if same:
        out.append("Greedy tokens (the first sequence's) the same as bf16's, configurations: " + "; ".join(same) + ".")
    out.append("")
if first:
    out = [f"# How fast it responds: {first['gpu']} ({first['gpu_memory_gb']} GB; {first['smi']}), {first['cpu']} ({first['cpus']} CPUs, {first['host']})",
           f"torch {first['torch']} (CUDA {first['cuda']}), transformers {first['transformers']}, glyd {next((r.get('glyd') for m in res.values() for r in m.values() if r.get('glyd')), '?')};"
           + " greedy decoding, every reply forced to its length; median of repeats, each configuration's first call apart", ""] + out
print("\n".join(out))
if "--json" in sys.argv:
    json.dump(res, open(sys.argv[sys.argv.index("--json") + 1], "w"), indent=1)
