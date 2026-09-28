# Models one after another in one process (each let go of before the next), plain generate(): does each compile, or
# does torch._dynamo's recompile limit leave the later ones eager? argv MODEL N
import sys, time, warnings, torch
import glyd
from torch._dynamo.utils import counters
from transformers import AutoTokenizer
name, n = sys.argv[1], int(sys.argv[2])
ids = AutoTokenizer.from_pretrained(name)("The history of data compression began", return_tensors="pt").input_ids.cuda()
for i in range(n):
    m = glyd.from_pretrained(name, compile=i < n - 1)  # (the last eager, for reference)
    with warnings.catch_warnings(record=True) as w, torch.no_grad():
        warnings.simplefilter("always")
        m.generate(ids, max_new_tokens=16, do_sample=False)
        torch.cuda.synchronize()
        t = time.perf_counter()
        m.generate(ids, max_new_tokens=64, min_new_tokens=64, do_sample=False)
        torch.cuda.synchronize()
        t = time.perf_counter() - t
    del m
    torch.cuda.empty_cache()
    print(f"model {i}: {64 / t:.1f} tokens/s, graphs {counters['stats']['unique_graphs']}, after del {torch.cuda.memory_allocated() / 1e9:.2f} GB allocated, {torch.cuda.memory_reserved() / 1e9:.2f} reserved", flush=True)
import torch._dynamo
print("recompile_limit after:", torch._dynamo.config.recompile_limit, flush=True)
