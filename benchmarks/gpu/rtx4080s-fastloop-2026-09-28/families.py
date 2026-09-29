# generate() on tiny random models of every mixture-of-experts family test_gpu.py packs (and a few dense ones), loaded
# by glyd.from_pretrained as a user loads them: the fast loop (compiled by default) against compile=False, 8 greedy
# tokens each: taken or not, a warning, an error, the tokens the same. argv: [kinds...]
import os, sys, tempfile, time, traceback, warnings, torch
sys.path.insert(0, os.path.expanduser("~/p8fastloop/bindings/python"))
import test_gpu as t
import glyd
from transformers import AutoModelForCausalLM

DENSE = {"llama": {}, "mistral": {}, "qwen2": {}, "qwen3": {}, "gemma2": {}, "gemma3_text": {}, "phi3": {}, "olmo2": {}, "granite": {}, "cohere2": {}, "smollm3": {}, "glm4": {}}
kinds = sys.argv[1:] or [k for k, o in t.FAMILIES.items() if not o.get("compress")] + list(DENSE)
for kind in kinds:
    over = t.FAMILIES.get(kind, DENSE.get(kind, {}))
    line = f"{kind}: "
    try:
        model, auto, cfg, ids, kw = t.tiny_model(torch, kind, over)
        if kw or auto is not AutoModelForCausalLM:
            print(line + f"skipped ({auto.__name__}{', decoder inputs' if kw else ''})", flush=True)
            continue
        with tempfile.TemporaryDirectory() as d:
            model.to(torch.bfloat16).save_pretrained(d)
            del model
            x = ids[:1, :8]
            res = {}
            for c in (False, True):
                g = glyd.from_pretrained(d, compile=c)
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    s = time.perf_counter()
                    with torch.no_grad():
                        out = g.generate(x, attention_mask=torch.ones_like(x), max_new_tokens=8, min_new_tokens=8, do_sample=False, pad_token_id=0)
                    s = time.perf_counter() - s
                ours = [str(m.message) for m in w if "glyd" in str(m.message)]
                res[c] = (out[0, 8:], "glyd_compiled" in g.__dict__, ours, s)
                del g
                torch._dynamo.reset()
        (a, _, _, _), (b, comp, ours, s) = res[False], res[True]
        line += f"{'compiled' if comp and not ours else 'eager'} in {s:.1f} s; tokens as eager's {(a == b).long().cumprod(0).sum().item()} of 8" + (f"; warned: {ours}" if ours else "")
    except Exception as e:
        line += f"ERROR {type(e).__name__}: {str(e).splitlines()[0][:200] if str(e) else ''} | " + traceback.format_exc().splitlines()[-3].strip()[:160]
    print(line, flush=True)
    torch.cuda.empty_cache()
print("families done")
