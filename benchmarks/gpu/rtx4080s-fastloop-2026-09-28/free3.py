# The fast loop with its compiled forward an unbound torch.compile(type(model).__call__) called with the model, against
# transformers' torch.compile(model.__call__) (bound: the model refers to it): the same tokens, the same speed, and the
# model freed at del? argv MODEL bound|unbound
import gc, sys, time, weakref, torch
import glyd
from glyd.gpu import model as gm
from transformers import AutoTokenizer
name, how = sys.argv[1], sys.argv[2]
ids = AutoTokenizer.from_pretrained(name)("The history of data compression began", return_tensors="pt").input_ids.cuda()
mem = lambda: (torch.cuda.synchronize(), torch.cuda.memory_allocated() / 1e9)[1]


def unbound(self, own, compile_config=None):
    cfg = compile_config or self._default_compile_config()
    c = self.__dict__.get("glyd_compiled")
    if c is None or c[0] != cfg:
        c = self.glyd_compiled = (cfg, torch.compile(type(self).__call__, **cfg.to_dict()))
    f = c[1]
    return lambda *a, **k: f(self, *a, **k)


if how == "unbound":
    gm._compiled_call = unbound
base = mem()
m = glyd.from_pretrained(name)
with torch.no_grad():
    for i in range(3):
        torch.cuda.synchronize()
        t = time.perf_counter()
        out = m.generate(ids, max_new_tokens=128, min_new_tokens=128, do_sample=False)
        torch.cuda.synchronize()
        t = time.perf_counter() - t
        print(f"{how}: generate {i}: {t:.2f} s ({128 / t:.1f} tokens/s); tokens {out[0, -128:].tolist()[:12]}", flush=True)
alive = weakref.ref(m)
held = mem() - base
del m, out
torch.cuda.empty_cache()
a = mem() - base
print(f"{how}: held {held:.2f} GB; after del {a:.2f} GB, the model {'alive' if alive() is not None else 'freed'}", flush=True)
gc.collect()
torch.cuda.empty_cache()
print(f"{how}: after gc.collect() {mem() - base:.2f} GB, the model {'alive' if alive() is not None else 'freed'}", flush=True)
