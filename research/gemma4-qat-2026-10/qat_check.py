"""Are a QAT checkpoint's bf16 weights int4 x group-32 scales, exactly? One 256-row slice of each tensor kind (first, middle, last layer),
through lattice_int4.run: exact groups, bit-exact round trip, bits a value.   python -I qat_check.py REPO OUT.json"""
import json, re, sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import hf, lattice_int4 as L

repo, out = sys.argv[1], sys.argv[2]
ix = hf.index(repo)
text = {k: v for k, v in ix.items() if k.startswith("model.language_model.") and v["dtype"] in ("BF16", "F16") and len(v["shape"]) >= 2}
kinds = {}
for k in text:
    kinds.setdefault(re.sub(r"\.layers\.\d+\.", ".layers.N.", k), []).append(k)
res = []
for kind, names in sorted(kinds.items()):
    names = sorted(names, key=lambda n: int(re.search(r"\.layers\.(\d+)\.", n).group(1)) if ".layers." in n else 0)
    picks = sorted({names[0], names[len(names) // 2], names[-1]})
    for n in picks:
        t = text[n]
        sh = t["shape"]
        rows = (0, 1) if len(sh) == 3 else (0, min(256, sh[0]))
        w = hf.fetch(repo, n, t, rows=rows)
        w = w.reshape(-1, sh[-1])[:256]
        if w.shape[1] % L.G:
            res.append({"label": n, "skipped": f"columns {w.shape[1]} not a multiple of {L.G}"}); continue
        t0 = time.time(); r = L.run(w, n); r["seconds"] = round(time.time() - t0, 1)
        res.append(r)
        print(f"{n}: exact {r['exact_groups']}/{r['groups']} roundtrip {r['bit_exact_roundtrip']} {r['bits_per_value']} bits (bf16 order-0 {r['plugin_order0_of_bf16_codes']})", flush=True)
        json.dump(res, open(out, "w"), indent=1)
