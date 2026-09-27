"""glyd's Python API end to end on the GPU, as a user of the package runs
it (the kernels from the prebuilt library: $GLYD_GPU_LIB, else the one
beside glyd/gpu/kernels.py), against each model in bf16:

- from_pretrained (fused, merged, best_layout's layout): 32 greedy tokens
  compared with bf16's as e2e.py compares them; the load's time, and its
  peak memory against the packed model's bytes and the largest tensor's;
- glyd.gpu.compress on the model loaded in bf16: the same packs, so the
  same logits and tokens bit for bit;
- exact=True: logits bit for bit bf16's, the 32 tokens bf16's;
- compiled, as transformers compiles generate() (a static cache, the
  forward under CUDA graphs), fullgraph, fused and exact: no graph break,
  no graph left to run uncaptured; the tokens against eager's;
- first, with no model loaded yet: the first model compiled with
  exact=True (its CUDA graph decodes into the scratch buffer, and keeps
  its address), the second loaded (a bigger buffer takes its place), the
  first's graph replayed: both models' logits as before;
- save_pretrained, then from_pretrained(path, verify=True): every tensor's
  sha256 against glyd.json, the logits and tokens of the model saved;
  exact=True from the saved packs (merged groups split): bf16's logits;
  the 12-bit layout from the tiered packs (transcoded): the logits of the
  12-bit layout packed from bf16;
- python -m glyd.gpu verify and fit.
A mixture of experts' model (its Experts modules packed, run by "glyd")
takes every check but saving (glyd-v1 holds no packed experts yet), and
two of its own: torch.compile(model.forward, mode="reduce-overhead",
fullgraph=True) called with gradients on, and copy.deepcopy refused.

    python check_api.py [MODEL ...]      (default: Qwen/Qwen3-0.6B Qwen/Qwen3-1.7B)

From this checkout it runs the package beside it (bindings/python); a copy
of it run elsewhere runs the glyd installed (a wheel, its libraries in it)."""
import copy
import os
import subprocess
import sys
import tempfile
import time

PACKAGE = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bindings", "python"))
if os.path.isdir(os.path.join(PACKAGE, "glyd")):
    sys.path.insert(0, PACKAGE)
import torch
from torch._dynamo.utils import counters
from transformers import AutoModelForCausalLM, AutoTokenizer, CompileConfig
import glyd
import glyd.gpu
from glyd.gpu import moe
from glyd.gpu.model import GEmbedding, GLinear, Scratch

TOKENS = 32
PROMPT = "The history of data compression began"


def loaded(f):
    """f()'s model, its load time (s), the GPU memory it peaked at and holds (GB, over what was held before)."""
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    t = time.perf_counter()
    m = f()
    torch.cuda.synchronize()
    return m, time.perf_counter() - t, (torch.cuda.max_memory_allocated() - base) / 1e9, (torch.cuda.memory_allocated() - base) / 1e9


def run(model, ids):
    """The prompt's last logits and TOKENS greedy tokens."""
    with torch.no_grad():
        logits = model(ids, logits_to_keep=1).logits
        out = model.generate(ids, max_new_tokens=TOKENS, min_new_tokens=TOKENS, do_sample=False)
    return logits, out[0, ids.shape[1] :]


def compiled(model, ids):
    """TOKENS greedy tokens from generate() compiled, fullgraph, with no graph break and no CUDA graph skipped;
    the compiled code and its graphs let go of after."""
    counters.clear()
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=TOKENS, min_new_tokens=TOKENS, do_sample=False, cache_implementation="static", compile_config=CompileConfig(fullgraph=True))
    assert not counters["graph_break"] and not counters["inductor"]["cudagraph_skips"], (dict(counters["graph_break"]), counters["inductor"]["cudagraph_skips"])
    torch._dynamo.reset()
    model.__dict__.pop("_compiled_call", None)
    return out[0, ids.shape[1] :]


def same(a, b):
    """Tokens identical from the start, as e2e.py counts them."""
    return (a == b).long().cumprod(0).sum().item()


def exact(a, b):
    return torch.equal(a.view(torch.int16), b.view(torch.int16))


def packed_bytes(model):
    """The model's weights as held: its packs (a mixture of experts' too), and every tensor not packed (biases
    included)."""
    packs = {id(m.p): m.p for m in model.modules() if isinstance(m, (GLinear, GEmbedding))}
    rest = sum(p.numel() * p.element_size() for p in model.parameters())
    return sum(p.nbytes() for p in packs.values()) + moe.nbytes(model) + rest + sum(m.bias.numel() * 2 for m in model.modules() if isinstance(m, GLinear) and m.bias is not None)


NAMES = sys.argv[1:] or ["Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B"]
if len(NAMES) > 1:
    with torch.no_grad():
        a_ids = AutoTokenizer.from_pretrained(NAMES[0])(PROMPT, return_tensors="pt").input_ids.cuda()
        b_ids = AutoTokenizer.from_pretrained(NAMES[1])(PROMPT, return_tensors="pt").input_ids.cuda()
        a = glyd.from_pretrained(NAMES[0], exact=True)
        f = torch.compile(a.forward, mode="reduce-overhead", fullgraph=True)
        before = [f(a_ids, use_cache=False).logits.clone() for _ in range(3)][-1]  # warm-up, capture, replay
        b = glyd.from_pretrained(NAMES[1], exact=True)
        b_before = b(b_ids, use_cache=False).logits
        after = f(a_ids, use_cache=False).logits.clone()
        torch.cuda.synchronize()
        assert exact(before, after) and exact(b_before, b(b_ids, use_cache=False).logits), "a CUDA graph replayed after a bigger scratch buffer took its place"
    print(f"{NAMES[0]} exact compiled, {NAMES[1]} loaded after: the first's CUDA graph replayed, both models' logits as before")
    del a, b, f
    torch._dynamo.reset()
    Scratch.buf.clear()  # the rest as in a process with no model loaded before (its memory lines count the buffer)
    torch.cuda.empty_cache()

for name in NAMES:
    tok = AutoTokenizer.from_pretrained(name)
    ids = tok(PROMPT, return_tensors="pt").input_ids.cuda()

    ref, t, peak, held = loaded(lambda: AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, device_map={"": "cuda:0"}))
    largest = max(p.numel() * p.element_size() for p in ref.parameters())
    logits_a, out_a = run(ref, ids)
    print(f"{name}: bf16 loaded in {t:.1f} s, peak {peak:.2f} GB, holds {held:.2f} GB, largest tensor {largest / 1e9:.2f} GB")
    del ref
    torch.cuda.empty_cache()

    m, t, peak, held = loaded(lambda: glyd.from_pretrained(name))
    q, size = m.config.quantization_config, packed_bytes(m)
    logits_b, out_b = run(m, ids)
    # The peak: the packed model, the embedding in bf16 until the end, and the packers' own scratch.
    print(f"{name}: glyd {q.layout} fused loaded in {t:.1f} s, peak {peak:.2f} GB: packed weights {size / 1e9:.2f} GB + largest tensor {largest / 1e9:.2f} GB + {peak - (size + largest) / 1e9:.2f} GB; holds {held:.2f} GB")
    print(f"   generated tokens identical to bf16: {same(out_a, out_b)} of {TOKENS}; logits bit-identical: {exact(logits_a, logits_b)}")
    print("   text:", tok.decode(out_b).replace("\n", " "))

    c = glyd.gpu.compress(AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16))
    logits_c, out_c = run(c, ids)
    assert exact(logits_b, logits_c) and torch.equal(out_b, out_c) and packed_bytes(c) == size, "glyd.gpu.compress packs as from_pretrained does"
    print(f"   glyd.gpu.compress: the same {size / 1e9:.2f} GB, logits and tokens bit for bit")
    print(f"   compiled (a static cache, CUDA graphs, fullgraph): tokens as eager's: {same(out_b, compiled(m, ids))} of {TOKENS}")
    del c
    torch.cuda.empty_cache()

    x, t, peak, held = loaded(lambda: glyd.from_pretrained(name, exact=True))
    logits_x, out_x = run(x, ids)
    assert exact(logits_a, logits_x) and torch.equal(out_a, out_x), "exact=True: bf16's logits and tokens"
    print(f"{name}: glyd exact loaded in {t:.1f} s, peak {peak:.2f} GB, holds {held:.2f} GB; logits bit-identical: True; generated tokens identical to bf16: {TOKENS} of {TOKENS}")
    print(f"   compiled: tokens as bf16's eager: {same(out_a, compiled(x, ids))} of {TOKENS}")
    del x
    torch.cuda.empty_cache()

    if moe.nbytes(m):
        # torch.compile(forward) as a user calls it, gradients on (a tied output layer stays nn.Linear: the logits
        # need them): its warm-up, capture and replay; then a copy, refused
        counters.clear()
        f = torch.compile(m.forward, mode="reduce-overhead", fullgraph=True)
        for _ in range(3):
            shape = f(ids).logits.shape  # the outputs let go of before the next call (they wait for a backward)
        assert shape[:2] == ids.shape and not counters["graph_break"] and not counters["inductor"]["cudagraph_skips"], (dict(counters["graph_break"]), counters["inductor"]["cudagraph_skips"])
        del f
        torch._dynamo.reset()
        try:
            copy.deepcopy(m)
            raise AssertionError("a model with packed experts copied")
        except TypeError:
            pass
        print("   torch.compile(forward, reduce-overhead, fullgraph), gradients on: no graph break, no CUDA graph skipped; copy.deepcopy refused")
        print(f"{name}: not saved: glyd-v1 holds no packed experts yet")
        del m
        torch.cuda.empty_cache()
        continue
    with tempfile.TemporaryDirectory() as d:
        t = time.perf_counter()
        glyd.save_pretrained(m, d)
        on_disk = sum(os.path.getsize(os.path.join(d, f)) for f in os.listdir(d) if f.endswith(".safetensors"))
        print(f"{name}: saved in {time.perf_counter() - t:.1f} s: {on_disk / 1e9:.2f} GB of safetensors; {', '.join(sorted(os.listdir(d)))}")
        del m
        torch.cuda.empty_cache()
        for verify in (False, True):
            r, t, peak, held = loaded(lambda: glyd.from_pretrained(d, verify=verify))
            logits_r, out_r = run(r, ids)
            assert exact(logits_r, logits_b) and torch.equal(out_r, out_b), f"reloaded (verify={verify}): the saved model's logits and tokens"
            print(f"   from_pretrained(path, verify={verify}): {r.config.quantization_config.verified} tensors verified, loaded in {t:.1f} s, peak {peak:.2f} GB, holds {held:.2f} GB; logits and tokens as the saved model's")
            del r
            torch.cuda.empty_cache()
        r, t, peak, held = loaded(lambda: glyd.from_pretrained(d, exact=True))
        logits_r, out_r = run(r, ids)
        assert exact(logits_r, logits_a) and torch.equal(out_r, out_a), "exact from the saved packs: bf16's logits"
        print(f"   from_pretrained(path, exact=True): merged groups split, loaded in {t:.1f} s; logits bit-identical to bf16: True")
        del r
        torch.cuda.empty_cache()
        other = "mma12" if q.layout == "mma" else "mma"
        r = glyd.from_pretrained(d, layout=other)
        s = glyd.from_pretrained(name, layout=other)
        (logits_r, out_r), (logits_s, out_s) = run(r, ids), run(s, ids)
        assert exact(logits_r, logits_s) and torch.equal(out_r, out_s) and packed_bytes(r) == packed_bytes(s), f"{other} from the saved packs: as packed from bf16"
        print(f"   from_pretrained(path, layout={other!r}): the packs {'transcoded' if other == 'mma12' else 'as saved'}, logits and tokens as {other} packed from bf16")
        del r, s
        torch.cuda.empty_cache()
        env = dict(os.environ, PYTHONPATH=os.path.dirname(os.path.dirname(glyd.__file__)))  # the glyd this run imported
        for cmd in (["verify", d], ["fit", name, "--gpu", "16GB"]):
            out = subprocess.run([sys.executable, "-m", "glyd.gpu", *cmd], env=env, capture_output=True, text=True)
            assert out.returncode == 0, out.stderr
            print("   python -m glyd.gpu", cmd[0] + ":", out.stdout.strip())
print(f"check_api: all passed ({glyd.__file__})")
