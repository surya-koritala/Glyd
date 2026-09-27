"""glyd.from_pretrained on each model, as a user's install runs it, against the
same model in bf16 and in fp32 (the reference for the true value):

- the default (fused products): over PROMPTS, the logits at every position
  against fp32's: the mean |difference| for Glyd and for bf16, and how often
  each one's top token is fp32's;
- exact=True: logits bit-identical to bf16's at every position;
- 32 greedy tokens (the text), the load time and peak memory, and
  save_pretrained then from_pretrained(path, verify=True) giving the same
  logits.

bf16 and fp32 run on the GPU where they fit it, else on the CPU (fp32) or
not at all (bf16: its checks are skipped and the line says so).

    python check_models.py MODEL [MODEL ...]
"""
import gc
import os
import sys
import tempfile
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PACKAGE = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bindings", "python"))
if os.path.isdir(os.path.join(PACKAGE, "glyd")):
    sys.path.insert(0, PACKAGE)
import glyd  # noqa: E402

PROMPTS = [
    "The history of data compression began",
    "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
    "Q: A train leaves at 3:40 pm and the trip takes 2 hours 35 minutes. When does it arrive?\nA:",
    "The capital of Australia is",
    "Translate to French: The weather is lovely today, so we will walk to the market.",
    "In quantum mechanics, the uncertainty principle states that",
    "SELECT name, COUNT(*) FROM orders JOIN customers ON",
    "Once upon a time, in a village at the edge of a great forest,",
]
FREE = 2 << 30  # bytes kept free on the GPU for activations and the runtime


def params_bytes(name, bytes_each):
    from accelerate import init_empty_weights
    from transformers import AutoConfig
    with init_empty_weights():
        m = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(name))
    return sum(p.numel() for p in m.parameters()) * bytes_each


def logits(model, device):
    """Every position's logits for every prompt, as fp32 on the CPU."""
    out = []
    with torch.no_grad():
        for ids in IDS:
            out.append(model(ids.to(device)).logits[0].float().cpu())
    return out


def load(name, dtype, device):
    return AutoModelForCausalLM.from_pretrained(name, dtype=dtype, device_map={"": device})


def flush():
    """What a model held, back to the GPU once its last reference is gone."""
    gc.collect()
    torch.cuda.empty_cache()


def compare(a, ref):
    """mean |a - ref| over every logit, and the share of positions whose top token is ref's."""
    d = sum((x - r).abs().sum().item() for x, r in zip(a, ref)) / sum(r.numel() for r in ref)
    top = sum((x.argmax(-1) == r.argmax(-1)).sum().item() for x, r in zip(a, ref)) / sum(r.shape[0] for r in ref)
    return d, top


def same(a, b):
    return all(torch.equal(x.view(torch.int32), y.view(torch.int32)) for x, y in zip(a, b))


cap = torch.cuda.get_device_properties(0).total_memory
for name in sys.argv[1:]:
    tok = AutoTokenizer.from_pretrained(name)
    IDS = [tok(p, return_tensors="pt").input_ids for p in PROMPTS]
    positions = sum(i.shape[1] for i in IDS)
    print(f"== {name}: {len(PROMPTS)} prompts, {positions} positions")

    dev32 = "cuda:0" if params_bytes(name, 4) + FREE < cap else "cpu"
    m = load(name, torch.float32, dev32)
    ref = logits(m, dev32)
    del m
    flush()

    bf = None
    if params_bytes(name, 2) + FREE < cap:
        m = load(name, torch.bfloat16, "cuda:0")
        bf = logits(m, "cuda:0")
        ids = IDS[0].cuda()
        out_bf = m.generate(ids, max_new_tokens=32, min_new_tokens=32, do_sample=False)[0, ids.shape[1]:]
        del m
        flush()

    torch.cuda.reset_peak_memory_stats()
    t = time.perf_counter()
    g = glyd.from_pretrained(name)
    t = time.perf_counter() - t
    peak = torch.cuda.max_memory_allocated() / 1e9
    gl = logits(g, "cuda:0")
    ids = IDS[0].cuda()
    out = g.generate(ids, max_new_tokens=32, min_new_tokens=32, do_sample=False)[0, ids.shape[1]:]
    d_g, top_g = compare(gl, ref)
    line = f"   glyd {g.config.quantization_config.layout}: loaded in {t:.1f} s, peak {peak:.2f} GB; against fp32 ({dev32}): mean |diff| {d_g:.4f}, top token fp32's {100 * top_g:.2f}%"
    if bf is not None:
        d_b, top_b = compare(bf, ref)
        line += f"; bf16's {d_b:.4f}, {100 * top_b:.2f}%; tokens as bf16's: {(out == out_bf).long().cumprod(0).sum().item()} of 32"
    else:
        line += "; bf16 does not fit this GPU"
    print(line)
    print("   text:", tok.decode(out).replace("\n", " "))

    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as d:
        glyd.save_pretrained(g, d)
        size = sum(os.path.getsize(os.path.join(d, f)) for f in os.listdir(d) if f.endswith(".safetensors"))
        del g
        flush()
        r = glyd.from_pretrained(d, verify=True)
        assert same(logits(r, "cuda:0"), gl), "the saved model's logits"
        print(f"   saved {size / 1e9:.2f} GB; reloaded with {r.config.quantization_config.verified} tensors verified, logits as before")
        del r
        flush()

    if bf is not None:
        x = glyd.from_pretrained(name, exact=True)
        assert same(logits(x, "cuda:0"), bf), "exact=True: bf16's logits"
        print(f"   exact=True: logits bit-identical to bf16's at all {positions} positions")
        del x
        flush()
print("check_models: all passed")
