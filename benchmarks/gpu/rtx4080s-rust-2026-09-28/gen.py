# generate() tokens/s, eager (each Linear's product its one-call path), for one model and layout: argv MODEL LAYOUT
# BATCHES TOKENS. Prints "LAYOUT batch B: tokens/s", the best of REPS runs (3), after a warm-up.
import os, sys, time, torch
import glyd
from transformers import AutoTokenizer

name, layout, batches, N = sys.argv[1], sys.argv[2], [int(b) for b in sys.argv[3].split(",")], int(sys.argv[4])
tok = AutoTokenizer.from_pretrained(name)
m = glyd.from_pretrained(name, layout=layout)
with torch.no_grad():
    for B in batches:
        ids = tok("The history of data compression began", return_tensors="pt").input_ids.cuda().repeat(B, 1)
        m.generate(ids, max_new_tokens=8, do_sample=False)
        best = 0
        for _ in range(int(os.environ.get("REPS", 3))):
            torch.cuda.synchronize()
            t = time.perf_counter()
            m.generate(ids, max_new_tokens=N, min_new_tokens=N, do_sample=False)
            torch.cuda.synchronize()
            best = max(best, B * N / (time.perf_counter() - t))
        print(f"{layout} batch {B}: {best:.1f} tokens/s", flush=True)
