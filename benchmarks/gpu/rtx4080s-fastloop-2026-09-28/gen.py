# generate() as a user calls it, in a fresh process: the load, the first generate() (the warm-up: compile and capture
# where compiled), then REPS timed runs. argv: MODEL bf16|glyd BATCH TOKENS. MODE: plain (model.generate as is) or
# static (cache_implementation="static": transformers' compiled loop, today's way to ask for it). KW: extra
# from_pretrained keywords for glyd as a Python dict literal (e.g. "{'compile': False}").
import ast, os, sys, time, torch
import glyd
from torch._dynamo.utils import counters
from transformers import AutoModelForCausalLM, AutoTokenizer

name, which, B, N = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
mode = os.environ.get("MODE", "plain")
kw = dict(cache_implementation="static") if mode == "static" else {}
tok = AutoTokenizer.from_pretrained(name)
ids = tok("The history of data compression began", return_tensors="pt").input_ids.cuda().repeat(B, 1)
t = time.perf_counter()
if which == "glyd":
    m = glyd.from_pretrained(name, **ast.literal_eval(os.environ.get("KW", "{}")))
else:
    m = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, device_map={"": "cuda:0"})
torch.cuda.synchronize()
load = time.perf_counter() - t
label = f"{which} {name.split('/')[-1]} batch {B} {mode}{' ' + os.environ['KW'] if os.environ.get('KW') else ''}{' GLYD_COMPILE=' + os.environ['GLYD_COMPILE'] if 'GLYD_COMPILE' in os.environ else ''}"
with torch.no_grad():
    t = time.perf_counter()
    m.generate(ids, max_new_tokens=N, min_new_tokens=N, do_sample=False, **kw)
    torch.cuda.synchronize()
    warm = time.perf_counter() - t
    rates = []
    for _ in range(int(os.environ.get("REPS", 3))):
        torch.cuda.synchronize()
        t = time.perf_counter()
        out = m.generate(ids, max_new_tokens=N, min_new_tokens=N, do_sample=False, **kw)
        torch.cuda.synchronize()
        rates.append(B * N / (time.perf_counter() - t))
compiled = bool({"_compiled_call", "glyd_compiled"} & m.__dict__.keys())
rates.sort()
print(f"{label}: load {load:.1f} s, first generate() {warm:.1f} s, then {rates[len(rates) // 2]:.1f} tokens/s (median of {len(rates)}: {', '.join(f'{r:.1f}' for r in rates)}), "
      f"{'compiled' if compiled else 'eager'}, graphs {counters['stats']['unique_graphs']}, peak {torch.cuda.max_memory_allocated() / 1e9:.2f} GB", flush=True)
print("   text:", tok.decode(out[0, ids.shape[1]:ids.shape[1] + 24]).replace("\n", " "), flush=True)
