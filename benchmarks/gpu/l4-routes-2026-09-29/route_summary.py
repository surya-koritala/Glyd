"""route_e2e.py's results as tables: python route_summary.py RESULTS_DIR. Per model and layout, a prompt's pass in ms by
length: bf16's, then each Glyd route's and its ratio to bf16's (* the fastest route), each with the GPU's SM clock and
power over its window (medians of the smi-*.csv samples while the GPU was busy)."""
import csv, datetime, glob, json, os, statistics, sys

R = sys.argv[1]
res = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(R, "route-*.json")))]
samples = []
for f in glob.glob(os.path.join(R, "smi-*.csv")):
    for r in csv.reader(open(f)):
        try:
            t = datetime.datetime.strptime(r[0].strip(), "%Y/%m/%d %H:%M:%S.%f").replace(tzinfo=datetime.timezone.utc).timestamp()
            samples.append((t, float(r[1].split()[0]), float(r[3].split()[0]), float(r[5].split()[0])))
        except (ValueError, IndexError):
            pass


def smi(w):
    s = [x for x in samples if w[0] <= x[0] <= w[1] and x[3] > 50] or [x for x in samples if w[0] <= x[0] <= w[1]]
    return f"{statistics.median(x[1] for x in s):.0f} MHz {statistics.median(x[2] for x in s):.0f} W" if s else "?"


name = lambda r: r["model"].split("models--")[-1].split("/")[0].split("--")[-1] if "models--" in r["model"] else os.path.basename(r["model"].split(" (")[0].rstrip("/"))
out = []
for model in dict.fromkeys(name(r) for r in res):
    b = {x["tokens"]: x for r in res if name(r) == model and r["mode"] == "bf16" for x in r["rows"] if "ms" in x}
    for r in [r for r in res if name(r) == model and r["mode"] == "glyd"]:
        routes = list(dict.fromkeys(x["route"] for x in r["rows"]))
        out += [f"## {model}, {'tiered' if r['layout'] == 'mma' else '12-bit'} ({r['layout']}), {r['gpu']} (code {r.get('gpu_code')}), glyd {r.get('glyd')}", "",
                "A prompt's pass, ms (over bf16's; SM clock and power, medians over the window); * the fastest route", "",
                "| tokens | bf16 | " + " | ".join(routes) + " |", "| ---: | ---: |" + " ---: |" * len(routes)]
        for L in sorted({x["tokens"] for x in r["rows"]}):
            row = {x["route"]: x for x in r["rows"] if x["tokens"] == L}
            best = min((x["ms"], k) for k, x in row.items() if "ms" in x)[1] if any("ms" in x for x in row.values()) else None
            cells = []
            for k in routes:
                x = row.get(k, {})
                if "ms" not in x:
                    cells.append(x.get("error", "")[:30])
                    continue
                rel = f" ({(x['ms'] / b[L]['ms'] - 1) * 100:+.1f}%)" if L in b else ""
                cells.append(f"{x['ms']:.1f}{rel}{'*' if k == best else ''} [{smi(x['window'])}]")
            bc = f"{b[L]['ms']:.1f} [{smi(b[L]['window'])}]" if L in b else ""
            out.append(f"| {L} | {bc} | " + " | ".join(cells) + " |")
        out.append("")
print("\n".join(out))
