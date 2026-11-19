"""With glyd-gpu 0.27.0: Qwen3-0.6B packed in the smallest layout (mma), its greedy answers, saved (glyd-v1), the save loaded again with verify=True and the same answers."""
import json, os, shutil, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oldsave_common import MODEL, answers, same, versions
import torch
import glyd
from transformers import AutoTokenizer

out = sys.argv[1]
print("versions", versions(), "torch", torch.__version__, torch.cuda.get_device_name(0), flush=True)
tok = AutoTokenizer.from_pretrained(MODEL)
m = glyd.from_pretrained(MODEL, layout="mma")
mem = answers(m, tok)
path = f"{out}/save027"
shutil.rmtree(path, ignore_errors=True)
glyd.save_pretrained(m, path, layout="mma")
man = json.load(open(f"{path}/glyd.json"))
print(f"saved: {path}: format {man['format']}, glyd {man['glyd']}, layout {man['layout']}, {len(man['packs'])} packs, files {sorted(os.listdir(path))}", flush=True)
del m
torch.cuda.empty_cache()
m2 = glyd.from_pretrained(path, verify=True)
qc = m2.config.quantization_config
print("0.27.0 loads its own save with verify=True:", {k: getattr(qc, k, None) for k in ("layout", "verify", "verified", "hashed", "unhashed")}, flush=True)
loaded = answers(m2, tok)
json.dump({"versions": versions(), "memory": mem, "loaded": loaded}, open(f"{out}/answers-027.json", "w"), indent=1)
for a, b, ok in zip(mem, loaded, same(mem, loaded)):
    print(f"0.27.0, packed in memory against its own save loaded: {'same' if ok else 'DIFFERENT'}: {a['prompt']!r} -> {a['text']!r} ({len(a['tokens'])} tokens)", flush=True)
sys.exit(0 if all(same(mem, loaded)) else 1)
