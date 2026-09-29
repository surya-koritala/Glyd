"""generate()'s step, compiled (transformers' static cache: the fast loop, glyd.from_pretrained's default) against
eager (compile=False: the dynamic cache), by the static cache's length: a step's attention reads the static cache
whole, masked, so its time grows with the length asked for (max_new_tokens), not with the tokens in it. A prompt of
16 tokens, max_new_tokens making the cache LEN long, stopped after 64 tokens (as an answer that ends before the
length allowed); then (one sequence) prompts of P tokens and 64 new (the cache as long as the context). The step's
ms: (64 tokens' time - 1 token's) / 63, the best of 2 after a warm-up (which compiles).
    python loop.py MODEL static|eager BATCH"""
import sys, time, torch
import glyd
from transformers import StoppingCriteria

name, mode, B = sys.argv[1], sys.argv[2], int(sys.argv[3])
m = glyd.from_pretrained(name, compile=False)  # the static cache asked for here, whatever the length
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
x = torch.randint(100, 20000, (B, 16), generator=g).cuda()
for L in (256, 1024, 2048, 4096):
    gen(x, max_new_tokens=L - 16, stopping_criteria=[After(16 + N)])
    t1 = min(gen(x, max_new_tokens=L - 16, stopping_criteria=[After(17)]) for _ in range(2))
    tn = min(gen(x, max_new_tokens=L - 16, stopping_criteria=[After(16 + N)]) for _ in range(2))
    print(f"{name.split('/')[-1]} {mode} batch {B}: cache {L} (max_new_tokens {L - 16}), stopped at {N}: {(tn - t1) / (N - 1) * 1000:.2f} ms a step", flush=True)
for P in ((512, 2048) if B == 1 else ()):
    x = torch.randint(100, 20000, (B, P), generator=g).cuda()
    gen(x, max_new_tokens=N, min_new_tokens=N)
    t1 = min(gen(x, max_new_tokens=1) for _ in range(2))
    tn = min(gen(x, max_new_tokens=N, min_new_tokens=N) for _ in range(2))
    print(f"{name.split('/')[-1]} {mode} batch {B}: prompt {P}, {N} new: {(tn - t1) / (N - 1) * 1000:.2f} ms a step", flush=True)
