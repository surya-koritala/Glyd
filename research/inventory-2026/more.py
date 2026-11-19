"""From screen.json: the bf16 repos to read (more_fetch.txt) and why the others are not inventoried (more_excluded.json)."""
import json, os, re
REG = open(os.path.join(os.environ.get("VLLM_SRC") or "../vllm030/vllm-0.30.0", "vllm/model_executor/models/registry.py")).read()
s = json.load(open("screen.json"))
have = {f[:-5].replace("__", "/") for f in os.listdir("headers")}
fetch, ex = [], {}
for r, v in s.items():
    if r in have:
        continue
    if "err" in v:
        ex[r] = "Hugging Face API error: " + v["err"][:80]
        continue
    p = v.get("params") or {}
    tot = sum(p.values()) or 1
    arch = (v.get("arch") or ["?"])[0]
    if v.get("quant") or p.get("BF16", 0) / tot < 0.9:
        ex[r] = f"not a bf16 checkpoint ({v.get('quant') or ', '.join(f'{k} {n/1e9:.1f}B' for k, n in p.items())})"
    elif not v.get("created", "").startswith("2026"):
        ex[r] = f"created {v.get('created')}"
    elif v.get("gated"):
        ex[r] = f"gated on Hugging Face ({sum(p.values())/1e9:.1f}B params, {arch}): headers need an accepted license and a token"
    elif f'"{arch}"' not in REG:
        ex[r] = f"{arch}: not in vLLM 0.30.0's registry (vLLM does not serve it; the plugin does not apply)"
    else:
        fetch.append((v.get("shards", 0), r))
fetch.sort()
open("more_fetch.txt", "w").write("\n".join(r for _, r in fetch) + "\n")
json.dump(ex, open("more_excluded.json", "w"), indent=1)
print(len(fetch), "to fetch,", sum(n for n, _ in fetch), "shards;", len(ex), "excluded")
