"""Truth (safetensors headers) vs preflight's estimate, per 2026 bf16 model."""
import glob, json, math, os, re, sys
sys.path.insert(0, sys.argv[1] if len(sys.argv) > 1 else "/Users/suryakoritala/glyd-tr-runfam/glyd-gpu-py")
import importlib; from glyd_gpu import preflight as pf; fit = importlib.import_module("glyd_gpu.fit")
SZ = {"BF16": 2, "F16": 2, "F32": 4, "F8_E4M3": 1, "U8": 1, "I8": 1, "I64": 8, "I32": 4, "F8_E8M0": 1, "U32": 4}
TOWER = re.compile(r"vision|visual|audio|image|multi_modal|mm_proj|embed_vision|embed_audio|speech|projector|vit\b|merger")
MTP = re.compile(r"(^|\.)mtp(\.|_)|nextn|\.mtp_")
TABLE = re.compile(r"(embed_tokens|embed_tokens_per_layer|lm_head|word_embeddings|wte|embedding|embeddings)\.weight$")
def truth(path):
    d = json.load(open(path))
    t = {n: v for sh in d["shards"].values() for n, v in sh.items() if n != "__metadata__"}
    c = dict(lin=0, tables=0, tower=0, mtp=0, rest=0, total=0, nonbf16=0)
    for n, (dt, sh) in t.items():
        b = math.prod(sh) * SZ.get(dt, 2)
        c["total"] += b
        if dt not in ("BF16", "F16"):
            c["nonbf16"] += b
        if MTP.search(n): c["mtp"] += b
        elif TOWER.search(n): c["tower"] += b
        elif TABLE.search(n) and len(sh) == 2: c["tables"] += b
        elif (len(sh) == 2 or (len(sh) == 3 and "expert" in n)) and "norm" not in n and not ROUTER.search(n) and dt in ("BF16", "F16"): c["lin"] += b
        else: c["rest"] += b
    return d, c
ROUTER = re.compile(r"router|\.gate(\.weight)?$|_gate\.weight$|expert_bias|e_score|A_log|position")
G = 1e9
rows = []
for p in sorted(glob.glob(sys.argv[2] + "/*.json")):
    d, c = truth(p)
    cfg = d.get("config")
    if not cfg or not c["total"] or not str(d.get("created") or "").startswith("2026"): continue  # (2026 releases only)
    m = pf.model_of(d["repo"], cfg, bf16=c["total"])
    lin, oth, moe = pf.linear_bytes(cfg.get("text_config", cfg))
    try: kv = fit.kv_bytes(cfg, 1)
    except Exception as e: kv = -1
    if c["total"] > float(sys.argv[3] if len(sys.argv) > 3 else 1e18): continue
    print(f"{d['repo'][:44]:<45}{c['total']/G:7.2f} | lin {c['lin']/G:7.2f} est {m.lin/G:7.2f} {(m.lin-c['lin'])/max(c['lin'],1):+6.1%} | tab {c['tables']/G:5.2f} est {m.tables/G:5.2f} | tower {c['tower']/G:4.2f} mtp {c['mtp']/G:4.2f} rest {c['rest']/G:4.2f} | kv/tok {kv}")
