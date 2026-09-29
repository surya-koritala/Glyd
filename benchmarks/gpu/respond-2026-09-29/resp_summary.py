"""resp_job.sh's results in one page: python resp_summary.py RESULTS_DIR [--json OUT]. Per model: the time to first
token at each prompt length, the tokens a second at each batch and the two mixes, bf16 eager and compiled against
Glyd's default and exact, each the median of its repeats (the warm-up apart), and Glyd's default over each bf16; the
load (GB, seconds, layout), which calls ran compiled (c), the first calls (the compile), whether each mode's greedy
tokens are bf16's; the GPU's clocks and power while each mode ran (smi-*.csv). --json: every result in one file."""
import csv, glob, json, os, statistics, sys

R = sys.argv[1]
res = {}
for f in sorted(glob.glob(os.path.join(R, "*.json"))):
    try:
        r = json.load(open(f))
    except ValueError:
        continue
    if "mode" in r and "model" in r:
        m = r["model"].rstrip("/")
        res.setdefault(m.split("models--")[-1].split("/")[0].split("--")[-1] if "models--" in m else os.path.basename(m), {})[r["mode"]] = r
MODES = ["bf16", "bf16c", "glyd", "exact"]
LAB = {"bf16": "bf16 eager", "bf16c": "bf16 compiled", "glyd": "Glyd (default)", "exact": "Glyd exact"}


def smi(model, mode):
    try:
        rows = [r for r in csv.reader(open(os.path.join(R, f"smi-{model}-{mode}.csv"))) if len(r) >= 6]
    except OSError:
        return ""
    num = lambda i: [float(r[i].split()[0]) for r in rows if r[i].split() and r[i].split()[0].replace(".", "", 1).isdigit()]
    sm, pw, ut = num(1), num(3), num(5)
    busy = [(s, p) for s, p, u in zip(sm, pw, ut) if u > 50] or list(zip(sm, pw))
    return f"SM {statistics.median(s for s, _ in busy):.0f} MHz, {statistics.median(p for _, p in busy):.0f} W (medians while busy)" if busy else ""


out, first = [], None
for model, modes in res.items():
    first = first or next(iter(modes.values()))
    head = [m for m in MODES if m in modes]
    get = lambda m, n: next((c for c in modes.get(m, {}).get("configs", []) if c["name"] == n), {})
    out += [f"## {model}", ""]
    for m in head:
        L = modes[m]["load"]
        s = smi(model, m)
        out.append(f"- {LAB[m]}: does not fit ({L.get('error', '')[:120]})" if not L.get("fits") else
                   f"- {LAB[m]}: {L['gb']} GB on the GPU after the load ({L['seconds']} s)" + (f", layout {L['layout']}" if L.get("layout") else "")
                   + (f", generate() compiled to {L['compiled_cap']} positions" if L.get("compiled_cap") else "") + (f"; {s}" if s else ""))
    names = list(dict.fromkeys(c["name"] for m in head for c in modes[m].get("configs", [])))
    ratio = [b for b in ("bf16", "bf16c") if b in head and "glyd" in head]

    def cell(c, key, fmt):
        if not c:
            return ""
        if c.get("skipped"):
            return f"({c['skipped'][:30]})"
        v = c.get(key)
        return (fmt(v) + (" c" if c.get("path") == "compiled" else "")) if v is not None else ""

    def rel(n, key, b):
        g, x = get("glyd", n).get(key), get(b, n).get(key)
        return f"{g / x:.2f}x" if g and x else ""

    tt = [n for n in names if n.startswith("ttft")]
    if tt:
        out += ["", "Time to first token, ms (Glyd's over bf16's: under 1 is sooner)", "",
                "| prompt | " + " | ".join(LAB[m] for m in head) + "".join(f" | Glyd / {LAB[b]}" for b in ratio) + " |", "| ---: |" + " ---: |" * (len(head) + len(ratio))]
        out += [f"| {get(head[0], n).get('prompt') or n.split()[1]} | " + " | ".join(cell(get(m, n), "ttft", lambda v: f"{v * 1e3:.0f}") for m in head)
                + "".join(f" | {rel(n, 'ttft', b)}" for b in ratio) + " |" for n in tt]
    rt = [n for n in names if n.startswith("rate")]
    if rt:
        new = get(head[0], rt[0]).get("new")
        out += ["", f"Tokens a second after the first ({new} new, a 128-token prompt; Glyd's over bf16's: over 1 is faster)", "",
                "| sequences | " + " | ".join(LAB[m] for m in head) + "".join(f" | Glyd / {LAB[b]}" for b in ratio) + " |", "| ---: |" + " ---: |" * (len(head) + len(ratio))]
        out += [f"| {n.split()[1]} | " + " | ".join(cell(get(m, n), "tokens_per_s", lambda v: f"{v:.1f}") for m in head)
                + "".join(f" | {rel(n, 'tokens_per_s', b)}" for b in ratio) + " |" for n in rt]
    mx = [n for n in names if n.startswith("mix")]
    if mx:
        out += ["", "The mixes: time to first token, ms / total, s (Glyd's total over bf16's: under 1 is sooner)", "",
                "| mix | " + " | ".join(LAB[m] for m in head) + "".join(f" | Glyd / {LAB[b]}" for b in ratio) + " |", "| :-- |" + " ---: |" * (len(head) + len(ratio))]
        for n in mx:
            c0 = get(head[0], n)
            cells = []
            for m in head:
                c = get(m, n)
                cells.append(f"({c['skipped'][:30]})" if c.get("skipped") else f"{c['ttft'] * 1e3:.0f} / {c['total']:.2f}" + (" c" if c.get("path") == "compiled" else "") if c.get("total") else "")
            out.append(f"| {n.split()[1]}: {c0.get('prompt')} + {c0.get('new')} | " + " | ".join(cells) + "".join(f" | {rel(n, 'total', b)}" for b in ratio) + " |")
    warm = [f"{LAB[m]} {cs[0]['warmup']['total']:.1f} s" for m in head for cs in [[c for c in modes[m].get("configs", []) if "warmup" in c]] if cs]
    same = []
    for m in head:
        if m != "bf16" and "bf16" in modes:
            both = [(c, get("bf16", c["name"])) for c in modes[m].get("configs", []) if c.get("sha") and get("bf16", c["name"]).get("sha")]
            if both:
                same.append(f"{LAB[m]} {sum(a['sha'] == b['sha'] for a, b in both)} of {len(both)}")
    out += ["", "c: the call ran compiled (fast_generate). Each process's first call (excluded above; the compile's in a compiled mode): " + "; ".join(warm) + "."]
    if same:
        out.append("Greedy tokens (the first sequence's) the same as bf16 eager's, configurations: " + "; ".join(same) + ".")
    out.append("")
if first:
    out = [f"# How fast it responds: {first['gpu']} ({first['gpu_memory_gb']} GB; {first['smi']}), {first['cpu']} ({first['cpus']} CPUs, {first['host']})",
           f"torch {first['torch']} (CUDA {first['cuda']}), transformers {first['transformers']}, glyd {next((r.get('glyd') for m in res.values() for r in m.values() if r.get('glyd')), '?')};"
           + " greedy decoding, every reply forced to its length; median of repeats, each configuration's first call apart", ""] + out
print("\n".join(out))
if "--json" in sys.argv:
    json.dump(res, open(sys.argv[sys.argv.index("--json") + 1], "w"), indent=1)
