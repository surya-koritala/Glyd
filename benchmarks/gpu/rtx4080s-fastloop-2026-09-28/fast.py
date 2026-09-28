# The fast loop's behaviours on one model: argv MODEL [TESTS...]
import os, sys, time, warnings, torch
import glyd
from glyd.gpu import model as gm
from torch._dynamo.utils import counters
from transformers import AutoTokenizer, CompileConfig, DynamicCache

name = sys.argv[1]
tests = set(sys.argv[2:]) or {"plain", "exact", "own", "budget", "fail", "shapes", "prefill", "memory"}
tok = AutoTokenizer.from_pretrained(name)
ids = tok("The history of data compression began", return_tensors="pt").input_ids.cuda()
compiled = lambda m: "glyd_compiled" in m.__dict__


def gen(m, x=ids, **kw):
    torch.cuda.synchronize()
    t = time.perf_counter()
    with torch.no_grad():
        out = m.generate(x, **{"do_sample": False, **kw})
    torch.cuda.synchronize()
    return out, time.perf_counter() - t


def forget(m):
    torch._dynamo.reset()
    m.__dict__.pop("glyd_compiled", None)
    m.__dict__.pop("_previous_max_cache_length", None)


m = glyd.from_pretrained(name)
print(name, "fast loop installed:", "glyd_fast" in m.__dict__)
if "plain" in tests:
    a, t = gen(m, max_new_tokens=32, min_new_tokens=32)
    print(f"plain: compiled {compiled(m)}, first {t:.1f} s")
    b, _ = gen(m, max_new_tokens=32, min_new_tokens=32, cache_implementation="static")
    print("   explicit static: tokens as plain's:", torch.equal(a, b))
    e = glyd.from_pretrained(name, compile=False)
    c, _ = gen(e, max_new_tokens=32, min_new_tokens=32)
    print("   compile=False: eager", not compiled(e), "; tokens as plain's", (a[0] == c[0]).long().cumprod(0).sum().item() - ids.shape[1], "of 32")
    del e
    forget(m)
if "own" in tests:
    counters.clear()
    for kw in (dict(past_key_values=DynamicCache(config=m.config)), dict(num_beams=2), dict(use_cache=False), dict(cache_implementation="dynamic"), dict(output_hidden_states=True, return_dict_in_generate=True), dict(prompt_lookup_num_tokens=3)):
        o, t = gen(m, max_new_tokens=8, **kw)
        print(f"own {list(kw)}: ok in {t:.1f} s, compiled {compiled(m)}")
        forget(m)
    from transformers import GenerationConfig
    with torch.no_grad():
        t = time.perf_counter()
        m.generate(ids, generation_config=GenerationConfig(max_new_tokens=8, do_sample=False))
        t = time.perf_counter() - t
    print(f"generation_config: ok in {t:.1f} s, compiled {compiled(m)}")
    forget(m)
    o, t = gen(m, max_new_tokens=8, do_sample=True, top_p=0.9)
    print(f"sampling: ok in {t:.1f} s, compiled {compiled(m)}")
    forget(m)
if "budget" in tests:
    n = m.glyd_fast - ids.shape[1]
    for new in (n, n + 1):
        o, t = gen(m, max_new_tokens=new, max_time=1.0)  # (stops after a second)
        print(f"budget: max_new_tokens {new} (the model's cap {m.glyd_fast}): compiled {compiled(m)}, {t:.1f} s")
        forget(m)
if "fail" in tests:
    def broken(gm_, inputs):
        raise RuntimeError("a backend that fails")
    ref, _ = gen(glyd.from_pretrained(name, compile=False), max_new_tokens=16, min_new_tokens=16)
    f = glyd.from_pretrained(name)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        o, t = gen(f, max_new_tokens=16, min_new_tokens=16, compile_config=CompileConfig(backend=broken))
        o2, _ = gen(f, max_new_tokens=16, min_new_tokens=16)
    ours = [str(x.message) for x in w if "glyd" in str(x.message)]
    print(f"fail: warnings {ours}; tokens as eager's {torch.equal(o, ref)} and again {torch.equal(o2, ref)}; eager after: {f.__dict__.get('glyd_eager')}, static cache on the next call: {compiled(f)}")
    del f
    torch._dynamo.reset()
if "shapes" in tests:
    counters.clear()
    for B, new in ((1, 64), (1, 64), (1, 128), (1, 256), (1, 100), (2, 64), (4, 64), (8, 64), (1, 64)):
        x = ids.repeat(B, 1)
        o, t = gen(m, x, max_new_tokens=new, min_new_tokens=new)
        print(f"shapes: batch {B} new {new}: {t:.2f} s ({B * new / t:.0f} tokens/s), graphs {counters['stats']['unique_graphs']}, compiled {compiled(m)}")
    forget(m)
if "prefill" in tests:
    for n in (512, 2048, 4096):
        x = torch.randint(0, 1000, (1, n), device="cuda", generator=torch.Generator(device="cuda").manual_seed(n))
        for kw in ({}, dict(cache_implementation="dynamic")):
            ts = []
            for _ in range(4):
                o, t = gen(m, x, max_new_tokens=1, attention_mask=torch.ones_like(x), **kw)
                ts.append(t)
            print(f"prefill {n}: first token {min(ts[1:]) * 1000:.1f} ms ({'plain' if not kw else 'dynamic'})")
if "memory" in tests:
    for kw in ({}, dict(cache_implementation="dynamic")):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        for B in (1, 8):
            gen(m, ids.repeat(B, 1), max_new_tokens=128, min_new_tokens=128, **kw)
        print(f"memory: peak {(torch.cuda.max_memory_allocated() - base) / 1e9:.3f} GB over the model ({'plain' if not kw else 'dynamic'}), reserved {torch.cuda.memory_reserved() / 1e9:.2f} GB")
if "exact" in tests:
    x = glyd.from_pretrained(name, exact=True)
    print("exact: fast loop installed:", "glyd_fast" in x.__dict__)
    o, t = gen(x, max_new_tokens=8)
    print(f"exact: compiled {compiled(x)}")
    del x
print("fast.py done")
