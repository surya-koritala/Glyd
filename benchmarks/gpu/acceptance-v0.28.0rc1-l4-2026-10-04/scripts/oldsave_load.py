"""With glyd-gpu 0.28.0rc1: the glyd-v1 save of 0.27.0 loaded with verify=True, its greedy answers against 0.27.0's; a glyd-v4 save of the same model made here (and of the old save, which must be the same bytes), and that one loaded with verify=True."""
import hashlib, json, os, shutil, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oldsave_common import MODEL, answers, same, versions
import torch
import glyd
from transformers import AutoTokenizer

out = sys.argv[1]
base = json.load(open(f"{out}/answers-027.json"))
print("versions", versions(), "torch", torch.__version__, torch.cuda.get_device_name(0), "| 0.27.0's answers by", base["versions"], flush=True)
tok = AutoTokenizer.from_pretrained(MODEL)
good = True


def qinfo(m):
    qc = m.config.quantization_config
    return {k: getattr(qc, k, None) for k in ("layout", "verify", "verified", "hashed", "unhashed")}


def compare(label, got):
    global good
    for want_name in ("memory", "loaded"):
        res = same(base[want_name], got)
        good &= all(res)
        print(f"{label} against 0.27.0's ({want_name}): {sum(res)} of {len(res)} answers token for token the same", flush=True)
    for a, b in zip(base["loaded"], got):
        print(f"    {a['prompt']!r} -> 0.27.0: {a['text']!r} | now: {b['text']!r}", flush=True)


# 1. the old save, loaded with verify=True
old = f"{out}/save027"
print("old save's format:", json.load(open(f"{old}/glyd.json"))["format"], flush=True)
m = glyd.from_pretrained(old, verify=True)
print("0.28.0rc1 loads the 0.27.0 save with verify=True:", qinfo(m), flush=True)
compare("0.27.0 save loaded by 0.28.0rc1", answers(m, tok))

# 2. a glyd-v4 save: of the model from the Hub, and of the old save as loaded here (packed again as it loaded)
v4, v4old = f"{out}/save028-v4", f"{out}/save028-v4-from-old"
for p in (v4, v4old):
    shutil.rmtree(p, ignore_errors=True)
fresh = glyd.from_pretrained(MODEL, layout="mma")
glyd.save_pretrained(fresh, v4, layout="mma")
glyd.save_pretrained(m, v4old, layout="mma")
for p in (v4, v4old):
    man = json.load(open(f"{p}/glyd.json"))
    print(f"saved: {p}: format {man['format']}, glyd {man['glyd']}, layout {man['layout']}, {len(man['packs'])} packs", flush=True)
    good &= man["format"] == "glyd-v4"


def shards(p):
    return {f: hashlib.sha256(open(f"{p}/{f}", "rb").read()).hexdigest() for f in sorted(os.listdir(p)) if f.endswith(".safetensors")}


s1, s2 = shards(v4), shards(v4old)
print("the glyd-v4 save of the Hub's model and of the old save as loaded: the shards are", "the same bytes" if s1 == s2 else "DIFFERENT", {k: v[:12] for k, v in s1.items()}, flush=True)
del fresh, m
torch.cuda.empty_cache()

# 3. the glyd-v4 save loaded with verify=True
m4 = glyd.from_pretrained(v4, verify=True)
print("0.28.0rc1 loads its own glyd-v4 save with verify=True:", qinfo(m4), flush=True)
compare("glyd-v4 save loaded by 0.28.0rc1", answers(m4, tok))
print("RESULT", "PASS" if good else "FAIL", flush=True)
sys.exit(0 if good else 1)
