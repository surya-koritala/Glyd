# A step's time against the cache's length, compiled (transformers' static cache, as fast_generate asks for it) and
# eager (the dynamic cache), in a fresh process each: argv MODEL MODE(static|eager) BATCH. Two cases:
# - long prompts (P tokens, then 64 generated): the cache as long as the context;
# - a short prompt with a generous max_new_tokens (the static cache that long from the first step), stopped after 64.
import sys, time, torch
import glyd
from transformers import StoppingCriteria

name, mode, B = sys.argv[1], sys.argv[2], int(sys.argv[3])
m = glyd.from_pretrained(name, compile=False)
kw = dict(cache_implementation="static") if mode == "static" else {}
N = 64


class After(StoppingCriteria):
    def __init__(self, n):
        self.n = n

    def __call__(self, input_ids, scores, **k):
        return torch.full((input_ids.shape[0],), input_ids.shape[1] >= self.n, dtype=torch.bool, device=input_ids.device)


def gen(x, **k):
    m.__dict__.pop("_previous_max_cache_length", None)  # (the static cache this call's length, not the longest yet)
    torch.cuda.synchronize()
    t = time.perf_counter()
    with torch.no_grad():
        m.generate(x, attention_mask=torch.ones_like(x), do_sample=False, pad_token_id=0, **kw, **k)
    torch.cuda.synchronize()
    return time.perf_counter() - t


g = torch.Generator().manual_seed(0)
for P in (16, 512, 1024, 2048, 4096):
    x = torch.randint(100, 20000, (B, P), generator=g).cuda()
    gen(x, max_new_tokens=N, min_new_tokens=N)  # (compiles for this length where static)
    t1 = min(gen(x, max_new_tokens=1) for _ in range(2))
    tn = min(gen(x, max_new_tokens=N, min_new_tokens=N) for _ in range(2))
    print(f"{name.split('/')[-1]} {mode} batch {B}: prompt {P}, {N} new: {(tn - t1) / (N - 1) * 1000:.2f} ms a step ({B * (N - 1) / (tn - t1):.1f} tokens/s)", flush=True)
x = torch.randint(100, 20000, (B, 16), generator=g).cuda()
for L in (256, 1024, 2048, 4096):
    gen(x, max_new_tokens=L - 16, stopping_criteria=[After(16 + N)])
    t1 = min(gen(x, max_new_tokens=L - 16, stopping_criteria=[After(17)]) for _ in range(2))
    tn = min(gen(x, max_new_tokens=L - 16, stopping_criteria=[After(16 + N)]) for _ in range(2))
    print(f"{name.split('/')[-1]} {mode} batch {B}: prompt 16, max_new_tokens {L - 16}, stopped at {N}: {(tn - t1) / (N - 1) * 1000:.2f} ms a step ({B * (N - 1) / (tn - t1):.1f} tokens/s)", flush=True)
