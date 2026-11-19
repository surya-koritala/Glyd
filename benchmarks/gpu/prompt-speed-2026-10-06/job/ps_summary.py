"""summary.txt of a ps_job.sh run: python3 ps_summary.py RESULTS (standard library only)."""
import glob
import json
import os
import re
import sys

R = sys.argv[1]


def read(path, default=""):
    try:
        with open(path) as f:
            return f.read()
    except OSError:
        return default


print(read(os.path.join(R, "gpu-short.txt")).strip())
print(read(os.path.join(R, "env.txt")).strip())
print("tree", read(os.path.join(R, "COMMIT")).strip())
print("\n######## steps")
print(read(os.path.join(R, "steps.txt")).rstrip())

for path in sorted(glob.glob(os.path.join(R, "pass-*.json"))):
    d = json.load(open(path))
    rows = d["rows"]
    print(f"\n######## {d['model']}: a pass through the layers' products ({d['products']} products, {d['weights'] / 1e9:.2f} G weights), ms, and over bf16's (timed just before)")
    ways = []
    for r in rows:
        if r["way"] != "bf16" and r["way"] not in ways:
            ways.append(r["way"])
    for layout in sorted({r["layout"] for r in rows if r["layout"] != "-"}):
        print(f"{layout:>5s}  tokens |  bf16 ms | " + " | ".join(f"{w:>15s}" for w in ways) + " | fastest")
        for M in sorted({r["M"] for r in rows}):
            got = [r for r in rows if r["layout"] == layout and r["M"] == M]
            b = next((r["ms"] for r in got if r["way"] == "bf16"), None)
            if b is None:
                continue
            cells, best = [], None
            for w in ways:
                r = next((r for r in got if r["way"] == w), None)
                if r is None:
                    cells.append(f"{'-':>15s}")
                    continue
                flag = "" if (r["same"] is not False and (r["err"] is None or r["err"] < 1.5e-2)) else "!"
                cells.append(f"{r['ms']:8.2f} {r['ms'] / b:5.3f}{flag}")
                if best is None or r["ms"] < best[1]:
                    best = (w, r["ms"])
            print(f"{'':>5s} {M:7d} | {b:8.2f} | " + " | ".join(cells) + (f" | {best[0]} {best[1] / b:.3f}" if best else ""))
    fails = [r for r in rows if r["same"] is False or (r["err"] is not None and r["err"] >= 1.5e-2)]
    print(f"checks: {len(rows) - len(fails)} of {len(rows)} rows pass (decode and ahead: bf16's bits; glyd and glinear: within 1.5e-2 of fp32)"
          + (f"; FAILED {[(r['layout'], r['way'], r['M']) for r in fails]}" if fails else ""))
    txt = read(path[:-5] + ".txt")
    for line in txt.splitlines():
        if "FAILED" in line or "bits a weight" in line:
            print("  " + line)

modes = sorted({os.path.basename(p)[len("serve-") : -4] for p in glob.glob(os.path.join(R, "serve", "serve-*.txt"))}, key=lambda m: ("bf16", "glyd", "ahead").index(m) if m in ("bf16", "glyd", "ahead") else 9)
if modes:
    print("\n######## vllm serve Qwen/Qwen3.5-9B, one user: time to first token (vllm bench serve, median and p99 of the prompts), the same flags each")
    base = {}
    for m in modes:
        log = read(os.path.join(R, "serve", f"serve-{m}.txt"))
        load = re.findall(r"Model loading took ([\d.]+) GiB", log)
        kv = re.findall(r"GPU KV cache size: ([\d,]+) tokens", log)
        cells = []
        for name in ("p1k", "p8k"):
            try:
                j = json.load(open(os.path.join(R, "serve", f"{m}-{name}.json")))
                t50, t99 = j.get("median_ttft_ms"), j.get("p99_ttft_ms")
                if m == "bf16":
                    base[name] = t50
                vs = f" = {t50 / base[name]:.3f}x bf16" if m != "bf16" and base.get(name) else ""
                cells.append(f"{name}: TTFT {t50:7.1f} ms (p99 {t99:7.1f}){vs}, {j.get('completed')} prompts, {j.get('total_input_tokens')} tokens in")
            except (OSError, ValueError, TypeError, KeyError):
                cells.append(f"{name}: -")
        g = read(os.path.join(R, "serve", f"greedy-{m}.json"))
        try:
            text = json.loads(g)["choices"][0]["text"]
        except (ValueError, KeyError, IndexError, TypeError):
            text = None
        if m == "bf16":
            base["text"] = text
        same = "" if m == "bf16" or text is None else (", greedy text = bf16's" if text == base.get("text") else ", greedy text differs from bf16's")
        print(f"{m:>6s}: weights {load[-1] if load else '-'} GiB, KV cache {kv[-1] if kv else '-'} tokens{same}")
        for c in cells:
            print(f"        {c}")
