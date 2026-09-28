# A chat's turns: generate() with the context growing each call, the fast loop as installed (first call compiles, a new
# length recompiles or records a graph); each call's time and the graphs dynamo made. argv MODEL
import sys, time, torch
import glyd
from torch._dynamo.utils import counters
m = glyd.from_pretrained(sys.argv[1])
g = torch.Generator().manual_seed(0)
x = torch.randint(100, 20000, (1, 40), generator=g).cuda()
for turn in range(8):
    torch.cuda.synchronize()
    t = time.perf_counter()
    with torch.no_grad():
        out = m.generate(x, attention_mask=torch.ones_like(x), max_new_tokens=64, min_new_tokens=64, do_sample=False, pad_token_id=0)
    torch.cuda.synchronize()
    t = time.perf_counter() - t
    print(f"turn {turn}: context {x.shape[1]} + 64: {t:.2f} s, graphs {counters['stats']['unique_graphs']}, cache {getattr(m, '_previous_max_cache_length', None)}", flush=True)
    x = torch.cat([out, torch.randint(100, 20000, (1, 40), generator=g).cuda()], 1)
