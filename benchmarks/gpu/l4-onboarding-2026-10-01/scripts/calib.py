"""calib.py LOG...: vLLM's own memory numbers in a glyd server log against what preflight.py predicted for its settings."""
import os, re, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "bindings", "python"))
from glyd.gpu import preflight as pf
GiB = 2**30
cfgs = {r: c for rungs in pf.LADDERS.values() for r, c in rungs}
for path in sys.argv[1:]:
    text = open(path, errors="replace").read()
    first = text.splitlines()[0]
    repo = re.search(r"vllm\.entrypoints\.cli\.main serve (\S+)", first)
    repo = repo.group(1) if repo else None
    w = re.search(r"Model loading took ([\d.]+) GiB", text)
    kv = re.search(r"Available KV cache memory: ([\d.]+) GiB", text)
    tok = re.search(r"GPU KV cache size: ([\d,]+) tokens", text)
    mem = re.search(r"Free memory on device (?:\S+ )?\(([\d.]+)/([\d.]+) GiB\) on startup\. Desired GPU memory utilization is \(([\d.]+), ([\d.]+) GiB\)\. Actual usage is ([\d.]+) GiB for consumed memory \(weights \+ non-torch\), ([\d.]+) GiB for peak activation, and ([\d.]+) GiB for CUDAGraph", text)
    layout = re.search(r"GLYD_LAYOUT=(\w+)", first)
    ctx = re.search(r"--max-model-len (\d+)", first)
    print(path.split("/")[-1], "|", repo, "| layout", layout and layout.group(1), "| ctx", ctx and ctx.group(1), "| eager" if "--enforce-eager" in first else "| compiled")
    if not (w and kv and tok and mem and repo in cfgs):
        print("   (no complete memory lines or an unknown model)")
        continue
    free, total, util, budget, used, act, graph = (float(x) for x in mem.groups())
    m = pf.model_of(repo, cfgs[repo])
    lay = layout.group(1) if layout else "mma"
    pw = pf.weights_on_gpu(m, lay) / GiB
    pk = max(0.0, (util * total - pw - pf.NON_KV / GiB) * pf.KV_FIT) * GiB // m.kv_token
    actual_nonkv = budget - float(w.group(1)) - float(kv.group(1))
    print(f"   weights: vLLM {w.group(1)} GiB, predicted {pw:.2f} ({pw - float(w.group(1)):+.2f})")
    print(f"   non-KV beyond the weights: vLLM {actual_nonkv:.2f} GiB (consumed {used} = weights + {used - float(w.group(1)):.2f} non-torch; activation {act}; graphs {graph}), constant {pf.NON_KV / GiB:.2f}")
    print(f"   KV tokens: vLLM {tok.group(1)} at budget {budget} GiB (util {util} of {total}, free at start {free}); predicted {int(pk):,} ({int(pk) - int(tok.group(1).replace(',', '')):+,})")
