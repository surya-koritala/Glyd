# The coordinator's rulings, checked: (3) families whose static cache fails in transformers 5.17 run eager from the
# start, no failing first call, where the config makes them fail, and compile where it does not; a compile error re-runs
# eager with one warning; any other error (out of memory, in the chain or not) is the call's, and the model still
# compiles after. (1) torch._dynamo's recompile_limit untouched outside Glyd's calls.
import os, sys, tempfile, warnings, torch
sys.path.insert(0, os.path.expanduser("~/p8fastloop/bindings/python"))
import test_gpu as t
import glyd
import torch._dynamo
from glyd.gpu import model as gm
from transformers import AutoModelForCausalLM, CompileConfig

print("recompile_limit at start:", torch._dynamo.config.recompile_limit, flush=True)


def gen(m, x, **kw):
    with warnings.catch_warnings(record=True) as w, torch.no_grad():
        warnings.simplefilter("always")
        out = m.generate(x, attention_mask=torch.ones_like(x), max_new_tokens=8, min_new_tokens=8, do_sample=False, pad_token_id=0, **kw)[0, x.shape[1]:]
    return out, [str(v.message)[:160] for v in w if "glyd" in str(v.message)]


for kind, heads in (("deepseek_v2", 2), ("deepseek_v2", 4), ("deepseek_v3", 2), ("deepseek_v3", 4), ("kimi_linear", 2), ("kimi_linear", 4), ("axk1", 2), ("axk1", 4), ("llama4_text", None)):
    over = dict(t.FAMILIES[kind])
    if heads:
        over["num_key_value_heads"] = heads
    model, auto, cfg, ids, kw = t.tiny_model(torch, kind, over)
    with tempfile.TemporaryDirectory() as d:
        model.to(torch.bfloat16).save_pretrained(d)
        del model
        x = ids[:1, :8]
        g = glyd.from_pretrained(d)
        out, said = gen(g, x)
        b = AutoModelForCausalLM.from_pretrained(d, dtype=torch.bfloat16, device_map={"": "cuda:0"})
        try:
            ref, _ = gen(b, x, cache_implementation="static")
            ref = f"bf16 static works, Glyd's compiled tokens as its {(out == ref).long().cumprod(0).sum().item()} of 8"
        except Exception as e:
            ref = f"bf16 static fails ({type(e).__name__})"
        print(f"{kind} (kv heads {heads}, heads {cfg.get_text_config().num_attention_heads}): set up {'glyd_fast' in g.__dict__}, compiled {'glyd_compiled' in g.__dict__}, warnings {said}; {ref}", flush=True)
        del g, b
        torch._dynamo.reset()

name = "Qwen/Qwen3-0.6B"
from transformers import AutoTokenizer
ids = AutoTokenizer.from_pretrained(name)("The history of data compression began", return_tensors="pt").input_ids.cuda()
m = glyd.from_pretrained(name)


def oom(graph, inputs, **kw):
    raise torch.cuda.OutOfMemoryError("CUDA out of memory (a test)")


for what, kw in (("out of memory while compiling", dict(compile_config=CompileConfig(backend=oom))), ("a call's own error", dict(max_new_tokens=-1))):
    try:
        gen(m, ids, **kw)
        print(what + ": no error?!", flush=True)
    except Exception as e:
        print(f"{what}: {type(e).__name__} raised to the caller; the model eager after: {bool(m.__dict__.get('glyd_eager'))}", flush=True)
    torch._dynamo.reset()
    m.__dict__.pop("glyd_compiled", None)
out, said = gen(m, ids)
print(f"then a plain call: compiled {'glyd_compiled' in m.__dict__}, warnings {said}", flush=True)
torch._dynamo.reset()
m.__dict__.pop("glyd_compiled", None)


def broken(graph, inputs, **kw):
    raise RuntimeError("a backend that fails")


out, said = gen(m, ids, compile_config=CompileConfig(backend=broken))
print(f"a compile error: warnings {said}; the model eager after: {bool(m.__dict__.get('glyd_eager'))}", flush=True)
print("recompile_limit at the end:", torch._dynamo.config.recompile_limit, flush=True)
