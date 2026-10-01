"""summ.py LOG...: what vLLM logged of memory and time in each glyd server log (weights, KV cache, budget, non-KV, start time, warnings)."""
import re, sys
from datetime import datetime

def stamp(line):
    m = re.search(r"(\d\d)-(\d\d) (\d\d):(\d\d):(\d\d)", line)
    return datetime(2026, int(m[1]), int(m[2]), int(m[3]), int(m[4]), int(m[5])) if m else None

for path in sys.argv[1:]:
    text = open(path, errors="replace").read()
    lines = text.splitlines()
    first = lines[0] if lines else ""
    g = lambda pat, flags=0: (re.search(pat, text, flags) or [None, None])[1]
    flags = re.findall(r"--(max-model-len|gpu-memory-utilization) (\S+)", first)
    enforce = "--enforce-eager" in first
    w = g(r"Model loading took ([\d.]+) GiB")
    load_s = g(r"Model loading took [\d.]+ GiB memory and ([\d.]+) seconds")
    kv = g(r"Available KV cache memory: ([\d.]+) GiB")
    toks = g(r"GPU KV cache size: ([\d,]+) tokens")
    mem = re.search(r"Free memory on device (?:\S+ )?\(([\d.]+)/([\d.]+) GiB\) on startup\. Desired GPU memory utilization is \(([\d.]+), ([\d.]+) GiB\)\. Actual usage is ([\d.]+) GiB for consumed memory \(weights \+ non-torch\), ([\d.]+) GiB for peak activation, and ([\d.]+) GiB for CUDAGraph", text)
    t0 = next((stamp(l) for l in lines if "api_utils" in l and stamp(l)), None)
    i1 = next((i for i, l in enumerate(lines) if "Application startup complete" in l), None)
    t1 = next((stamp(l) for l in reversed(lines[:i1]) if stamp(l)), None) if i1 else None  # (uvicorn's own line has no stamp: the one before it)
    print(path.split("/")[-1])
    print("  flags:", dict(flags), "eager" if enforce else "compiled", "| startup:", f"{(t1 - t0).seconds} s" if t0 and t1 else "no startup")
    print("  weights (vLLM 'Model loading took'):", w, "GiB in", load_s, "s; KV:", kv, "GiB =", toks, "tokens")
    if mem:
        free, total, util, budget, used, act, graph = (float(x) for x in mem.groups())
        nonkv = float(budget) - float(w or 0) - float(kv or 0)
        print(f"  free at start {free} of {total} GiB; budget {util} = {budget} GiB; consumed {used} (non-torch {used - float(w or 0):.2f}), activation {act}, graphs {graph}; non-KV beyond the weights {nonkv:.2f} GiB")
    print("  warnings: allocator OOM", len(re.findall(r"memory allocation failed with OOM", text)), "| expandable mapping", len(re.findall(r"expandable_segments: memory mapping failed", text)), "| tracebacks", text.count("Traceback"), "| ERROR lines", len(re.findall(r"\bERROR\b", text)))
