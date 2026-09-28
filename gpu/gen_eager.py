# generate() tokens/s as a user runs it: bf16 (transformers: eager) or glyd.from_pretrained (compiled: a static cache,
# CUDA graphs; GLYD_COMPILE=0 eager), or (COMPILE=1) either with cache_implementation="static", as transformers
# compiles it; the first generate() (its compile and capture) timed apart. argv: MODEL bf16|glyd [batch] [tokens]
import os, sys, time, torch
import glyd
from transformers import AutoModelForCausalLM, AutoTokenizer

name, which = sys.argv[1], sys.argv[2]
B = int(sys.argv[3]) if len(sys.argv) > 3 else 1
N = int(sys.argv[4]) if len(sys.argv) > 4 else 128
tok = AutoTokenizer.from_pretrained(name)
ids = tok("The history of data compression began", return_tensors="pt").input_ids.cuda().repeat(B, 1)
m = glyd.from_pretrained(name) if which == "glyd" else AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, device_map={"": "cuda:0"})
kw = dict(cache_implementation="static") if os.environ.get("COMPILE") else {}
with torch.no_grad():
    t = time.perf_counter()
    m.generate(ids, max_new_tokens=N, do_sample=False, **kw)  # (the static cache the timed runs' size)
    torch.cuda.synchronize()
    print(f"first generate(): {time.perf_counter() - t:.1f} s", flush=True)
    for _ in range(int(os.environ.get("REPS", 2))):
        torch.cuda.synchronize()
        t = time.perf_counter()
        m.generate(ids, max_new_tokens=N, min_new_tokens=N, do_sample=False, **kw)
        torch.cuda.synchronize()
        t = time.perf_counter() - t
        print(f"{which}{' compiled' if '_compiled_call' in m.__dict__ else ''} {name} batch {B}: {B * N / t:.1f} tokens/s ({t / N * 1000:.2f} ms a step)", flush=True)
if os.environ.get("THREADS"):
    import threading
    print("python threads:", threading.enumerate())
    print("os threads:", [open(f"/proc/self/task/{t}/comm").read().strip() for t in os.listdir("/proc/self/task")])
