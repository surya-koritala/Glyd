"""API pre-screen: dtype mix, architecture and quantization of each repo (2.5 s apart)."""
import json, os, sys, time, urllib.request
out = {}
path = sys.argv[2]
if os.path.exists(path):
    out = json.load(open(path))
for r in [l.strip() for l in open(sys.argv[1]) if l.strip()]:
    if r in out:
        continue
    for t in range(5):
        try:
            d = json.loads(urllib.request.urlopen(urllib.request.Request(f"https://huggingface.co/api/models/{r}", headers={"User-Agent": "glyd-inventory/1"}), timeout=60).read())
            c = d.get("config") or {}
            out[r] = {"created": (d.get("createdAt") or "")[:10], "params": (d.get("safetensors") or {}).get("parameters"), "arch": c.get("architectures"),
                      "quant": (c.get("quantization_config") or {}).get("quant_method"), "gated": d.get("gated"), "shards": len([s for s in d.get("siblings", []) if s["rfilename"].endswith(".safetensors")])}
            break
        except Exception as e:
            out[r] = {"err": str(e)}
            time.sleep(20 * (t + 1))
    json.dump(out, open(path, "w"), indent=0)
    print(r, out[r], flush=True)
    time.sleep(2.5)
