"""glyd's Python API end to end on the GPU, as a user of the package runs
it (the kernels from the prebuilt library: $GLYD_GPU_LIB, else the one
beside glyd/gpu/kernels.py), against each model in bf16:

- from_pretrained (fused, merged, best_layout's layout): 32 greedy tokens
  compared with bf16's as e2e.py compares them; the load's time, and its
  peak memory against the packed model's bytes and the largest tensor's;
- generate() as a user calls it: compiled from PyTorch 2.13
  (model.fast_generate: a static cache, CUDA graphs; below 2.13 eager, as
  in 0.23, and the fast loop's cases skipped; TOKENIZERS_PARALLELISM as it
  was after it; the model pickled after it, but with packed experts, which
  refuse a copy); with compile=False eager, the same logits; not with
  GLYD_COMPILE=0; a call with a cache of its own, several beams or a
  static cache past the model's cap (glyd_fast) as transformers runs it;
  disable_compile and return_dict_in_generate eager; as another model's
  assistant, its tokens as with an eager one; out of memory while
  compiling raised, and the next call compiled; a call that fails
  compiled: one warning, run again eager (a skip_prompt TextStreamer's
  text eager's; sampled from a seed, its tokens and text a seeded eager
  run's), and eager from there on; two fresh threads in turn, the second's
  cache longer, their tokens as this thread's; continuing from
  return_dict_in_generate's cache as eager (in a process of its own);
- glyd.gpu.compress on the model loaded in bf16: the same packs, so the
  same logits and tokens bit for bit;
- exact=True: logits bit for bit bf16's, the 32 tokens bf16's;
- a prompt of 2100 tokens, fused and exact, three times (the first records
  the order its matrices are decoded ahead in, model.Ahead, where the GPU
  takes that path; the others follow it): the logits as with each matrix
  decoded on the current stream, bit for bit; exact's bf16's;
- compiled, as transformers compiles generate() (a static cache, the
  forward under CUDA graphs), fullgraph, fused and exact: no graph break,
  no graph left to run uncaptured; the tokens against plain generate()'s
  (exact: bf16's eager ones);
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
takes every check, and two of its own: torch.compile(model.forward,
mode="reduce-overhead", fullgraph=True) called with gradients on, and
copy.deepcopy refused.

    python check_api.py [MODEL ...]      (default: Qwen/Qwen3-0.6B Qwen/Qwen3-1.7B)

From this checkout it runs the package beside it (bindings/python); a copy
of it run elsewhere runs the glyd installed (a wheel, its libraries in it)."""
import copy
import gc
import os
import pickle
import subprocess
import sys
import tempfile
import threading
import time
import warnings

PACKAGE = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bindings", "python"))
if os.path.isdir(os.path.join(PACKAGE, "glyd")):
    sys.path.insert(0, PACKAGE)
import torch
from torch._dynamo.utils import counters
from transformers import AutoModelForCausalLM, AutoTokenizer, CompileConfig, DynamicCache, TextStreamer
import glyd
import glyd.gpu
from glyd.gpu import moe
from glyd.gpu import model as gm
from glyd.gpu.model import GEmbedding, GLinear, Scratch

TOKENS = 32
PROMPT = "The history of data compression began"
FAST = torch.__version__ >= "2.13"  # generate() compiled by default (fast_generate); below 2.13 eager, as in 0.23


def loaded(f):
    """f()'s model, its load time (s), the GPU memory it peaked at and holds (GB, over what was held before: models let
    go of freed first, a compiled one's at a garbage collection)."""
    gc.collect()
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
    model.__dict__.pop("_compiled_call", None)  # (transformers' compiled forward; the fast loop's is in gm._COMPILED)
    gm._COMPILED.pop(model, None)
    return out[0, ids.shape[1] :]


class Recorded(TextStreamer):
    """A TextStreamer (skip_prompt) that keeps its text."""

    def __init__(self, tok):
        super().__init__(tok, skip_prompt=True)
        self.text = ""

    def on_finalized_text(self, text, stream_end=False):
        self.text += text


def fast_loop(m, e, tok, ids, out_e):
    """generate()'s fast loop on e, loaded with compile=False (eager; out_e its TOKENS tokens): GLYD_COMPILE=0 leaves it
    eager; set up (fast_generate), the calls it leaves as they come run eager (a cache of the call's own, two beams, a
    static cache past its cap, glyd_fast, disable_compile, return_dict_in_generate: none makes a static cache), and so
    does a call whose gate's helper (transformers' _prepare_generation_config) fails, its tokens eager's; as m's
    assistant, m's tokens as with it eager (transformers crops the cache an assistant hands back); a call that runs out
    of memory compiling raises it, and the next call compiles; a call that fails compiled (a backend that fails): one
    warning, the call run again eager, and the next eager, their tokens eager's, and a skip_prompt TextStreamer's text
    through it eager's; the same sampled from a seed: its tokens and text as a seeded eager run's."""
    compiled = lambda: e in gm._COMPILED
    static = lambda: "_previous_max_cache_length" in e.__dict__  # (transformers' _prepare_static_cache sets it)
    with torch.no_grad():
        assisted = m.generate(ids, assistant_model=e, max_new_tokens=TOKENS, min_new_tokens=TOKENS, do_sample=False)
        streamed = Recorded(tok)
        e.generate(ids, streamer=streamed, max_new_tokens=TOKENS, min_new_tokens=TOKENS, do_sample=False)
        torch.manual_seed(0)
        drawn = Recorded(tok)
        sampled = e.generate(ids, streamer=drawn, max_new_tokens=TOKENS, min_new_tokens=TOKENS, do_sample=True)
    os.environ["GLYD_COMPILE"] = "0"
    try:
        assert gm.fast_generate(e) is e and "glyd_fast" not in e.__dict__, "GLYD_COMPILE=0: generate() eager"
    finally:
        del os.environ["GLYD_COMPILE"]
    gm.fast_generate(e)
    with torch.no_grad():
        for kw in (dict(past_key_values=DynamicCache(config=e.config)), dict(num_beams=2), dict(max_new_tokens=e.glyd_fast, max_time=0.5), dict(max_cache_len=e.glyd_fast + 1), dict(disable_compile=True), dict(return_dict_in_generate=True)):
            e.generate(ids, **dict(dict(max_new_tokens=8, do_sample=False), **kw))
            assert not compiled() and not static(), ("a call the fast loop leaves as it came, compiled or with a static cache", list(kw))
        out = m.generate(ids, assistant_model=e, max_new_tokens=TOKENS, min_new_tokens=TOKENS, do_sample=False)
        assert torch.equal(out, assisted) and not compiled() and not static(), "as another model's assistant: its tokens as with the assistant eager"
        # transformers' private helper the gate reads (the merged generation config) failing: the call eager, as it came
        cls, calls = type(e), []
        own = cls._prepare_generation_config

        def fails_once(model, *a, **k):
            calls.append(1)
            if len(calls) == 1:
                raise RuntimeError("a helper that fails")
            return own(model, *a, **k)

        had = "_prepare_generation_config" in vars(cls)
        cls._prepare_generation_config = fails_once
        try:
            out = e.generate(ids, max_new_tokens=TOKENS, min_new_tokens=TOKENS, do_sample=False)
        finally:
            if had:
                cls._prepare_generation_config = own
            else:
                del cls._prepare_generation_config
        assert len(calls) == 2 and not compiled() and not static() and torch.equal(out[0, ids.shape[1] :], out_e), "the gate's helper failing: the call eager"

        def broken(graph, inputs, **kwargs):
            raise RuntimeError("a backend that fails")

        def oom(graph, inputs, **kwargs):
            raise torch.cuda.OutOfMemoryError("CUDA out of memory (check_api's)")

        try:
            e.generate(ids, max_new_tokens=8, do_sample=False, compile_config=CompileConfig(backend=oom))
            raise AssertionError("out of memory compiling: raised to the caller")
        except torch._dynamo.exc.BackendCompilerFailed:
            pass
        torch._dynamo.reset()
        gm._COMPILED.pop(e)
        e.generate(ids, max_new_tokens=8, do_sample=False)
        assert not e.__dict__.get("glyd_eager") and compiled(), "out of memory compiling: the next call compiles"
        torch._dynamo.reset()
        gm._COMPILED.pop(e)
        failed = Recorded(tok)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            out = e.generate(ids, streamer=failed, max_new_tokens=TOKENS, min_new_tokens=TOKENS, do_sample=False, compile_config=CompileConfig(backend=broken))
            gm._COMPILED.pop(e, None)
            again = e.generate(ids, max_new_tokens=TOKENS, min_new_tokens=TOKENS, do_sample=False)
        said = [str(x.message) for x in w if "compiled failed" in str(x.message)]
        assert len(said) == 1 and e.glyd_eager and not compiled() and torch.equal(out[0, ids.shape[1] :], out_e) and torch.equal(again[0, ids.shape[1] :], out_e), ("a call that fails compiled", said)
        assert failed.text == streamed.text, ("a streamer through a call that fails compiled: the eager run's text", failed.text, streamed.text)
        # sampled, from a seed: the call run again from the random state it found, so it draws what the attempt drew
        e.__dict__.pop("glyd_eager")
        torch._dynamo.reset()
        torch.manual_seed(0)
        failed = Recorded(tok)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            out = e.generate(ids, streamer=failed, max_new_tokens=TOKENS, min_new_tokens=TOKENS, do_sample=True, compile_config=CompileConfig(backend=broken))
        gm._COMPILED.pop(e, None)
        said = [str(x.message) for x in w if "compiled failed" in str(x.message)]
        assert len(said) == 1 and e.glyd_eager and torch.equal(out, sampled) and failed.text == drawn.text, ("a sampled call that fails compiled, from a seed: a seeded eager run's tokens and text", said, failed.text, drawn.text)
    torch._dynamo.reset()


def threads(m, ids):
    """m's generate() compiled from two fresh threads in turn, the second's prompt longer (its cache a new length:
    compiled again and a CUDA graph recorded in that thread): each call's tokens as the same call's in this thread."""
    got, prompts = {}, (ids, torch.cat([ids, ids], 1))

    def work(k, x):
        try:
            with torch.no_grad():
                got[k] = m.generate(x, max_new_tokens=8, min_new_tokens=8, do_sample=False)
        except Exception as e:  # (to this thread's assert)
            got[k] = e

    for k, x in enumerate(prompts):
        t = threading.Thread(target=work, args=(k, x))
        t.start()
        t.join()
    with torch.no_grad():
        for k, x in enumerate(prompts):
            assert isinstance(got[k], torch.Tensor) and torch.equal(got[k], m.generate(x, max_new_tokens=8, min_new_tokens=8, do_sample=False)), ("generate() in a fresh thread", k, got[k])


CONTINUED = """
import sys, torch, glyd
from transformers import AutoTokenizer
name = sys.argv[1]
tok = AutoTokenizer.from_pretrained(name)
ids = tok("The history of data compression began", return_tensors="pt").input_ids.cuda()
more = tok(" and then", return_tensors="pt", add_special_tokens=False).input_ids.cuda()
got = []
for compile in (True, False):
    m = glyd.from_pretrained(name, compile=compile)
    with torch.no_grad():
        out = m.generate(ids, max_new_tokens=16, min_new_tokens=16, do_sample=False, return_dict_in_generate=True)
        x = torch.cat([out.sequences, more], 1)
        got.append((type(out.past_key_values).__name__, m.generate(x, past_key_values=out.past_key_values, max_new_tokens=16, min_new_tokens=16, do_sample=False)))
    del m
assert got[0][0] == "DynamicCache" and torch.equal(got[0][1], got[1][1]), (got[0][0], got[1][0])
print("continued as eager: 16 tokens past the returned cache")
"""


def ahead(model, ids):
    """A long prompt's last logits: three times (the first records the order its matrices are decoded ahead in,
    the others follow it), then with no decode ahead (each matrix decoded on the current stream: the fused kernel,
    which takes a product Ahead does not, off too); the last two and that one bit for bit."""
    with torch.no_grad():
        runs = [model(ids, logits_to_keep=1).logits for _ in range(3)]
        get, gm.Ahead.get = gm.Ahead.get, staticmethod(lambda d: None)
        fused = [m for m in model.modules() if isinstance(m, GLinear) and m.fused]
        for m in fused:
            m.fused = False
        try:
            off = model(ids, logits_to_keep=1).logits
        finally:
            gm.Ahead.get = get
            for m in fused:
                m.fused = True
    assert exact(runs[1], off) and exact(runs[2], off), "a prompt decoded ahead: its logits as each matrix decoded on the current stream"
    return off


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
    long = torch.randint(0, ref.config.vocab_size, (1, 2100), generator=torch.Generator().manual_seed(0)).cuda()
    with torch.no_grad():
        long_a = ref(long, logits_to_keep=1).logits
    print(f"{name}: bf16 loaded in {t:.1f} s, peak {peak:.2f} GB, holds {held:.2f} GB, largest tensor {largest / 1e9:.2f} GB")
    del ref
    torch.cuda.empty_cache()

    m, t, peak, held = loaded(lambda: glyd.from_pretrained(name))
    q, size = m.config.quantization_config, packed_bytes(m)
    parallel = os.environ.get("TOKENIZERS_PARALLELISM")
    logits_b, out_b = run(m, ids)
    assert (m in gm._COMPILED) == FAST, "plain generate(): compiled (the fast loop) from PyTorch 2.13, else eager"
    assert os.environ.get("TOKENIZERS_PARALLELISM") == parallel, "TOKENIZERS_PARALLELISM after a compiled generate(): as it was"
    if not moe.nbytes(m):  # (a model with packed experts refuses a copy: below)
        pickle.dumps(m)  # after a compiled generate(), nothing unpicklable on the model
    # The peak: the packed model, the embedding in bf16 until the end, and the packers' own scratch.
    print(f"{name}: glyd {q.layout} fused loaded in {t:.1f} s, peak {peak:.2f} GB: packed weights {size / 1e9:.2f} GB + largest tensor {largest / 1e9:.2f} GB + {peak - (size + largest) / 1e9:.2f} GB; holds {held:.2f} GB")
    print(f"   generated tokens identical to bf16: {same(out_a, out_b)} of {TOKENS}; logits bit-identical: {exact(logits_a, logits_b)}")
    print("   text:", tok.decode(out_b).replace("\n", " "))
    e = glyd.from_pretrained(name, compile=False)
    logits_e, out_e = run(e, ids)
    assert e not in gm._COMPILED and "_compiled_call" not in e.__dict__ and exact(logits_e, logits_b), "compile=False: generate() eager, the same packs"
    if not FAST:
        print(f"   PyTorch {torch.__version__} (before 2.13): generate() eager, as in 0.23")
    else:
        print(f"   generate(): compiled by default (a static cache, CUDA graphs), eager with compile=False: tokens as eager's {same(out_e, out_b)} of {TOKENS}")
        fast_loop(m, e, tok, ids, out_e)
        print(f"   GLYD_COMPILE=0 eager; a cache of the call's own, two beams, a static cache past {e.glyd_fast} positions (max_new_tokens or max_cache_len), disable_compile, return_dict_in_generate eager; as another model's assistant, its tokens as eager; out of memory compiling raised, the next call compiled; a backend that fails: one warning, the call run again eager, and the next, their tokens eager's, a streamer's text eager's; sampled from a seed, its tokens and text a seeded eager run's")
    del e
    torch.cuda.empty_cache()
    threads(m, ids)
    print("   generate() compiled from two fresh threads in turn, the second's cache longer: tokens as this thread's")
    if name == NAMES[0]:  # continuing from a returned cache (a static one's overrun would end the CUDA context: a process of its own)
        env = dict(os.environ, PYTHONPATH=os.path.dirname(os.path.dirname(glyd.__file__)))
        r = subprocess.run([sys.executable, "-c", CONTINUED, name], env=env, capture_output=True, text=True, timeout=900)
        assert r.returncode == 0, r.stderr[-3000:]
        print("   return_dict_in_generate: " + r.stdout.strip().splitlines()[-1])
    ahead(m, long)
    a = gm.Ahead.of.get(torch.device("cuda", 0))
    print(f"   a prompt of {long.shape[1]} tokens: " + (f"{len(a.chain)} products decoded ahead, logits as decoded on the current stream, bit for bit" if a and a.chain else "not decoded ahead on this GPU"))

    c = glyd.gpu.compress(AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16))
    logits_c, out_c = run(c, ids)
    assert exact(logits_b, logits_c) and torch.equal(out_b, out_c) and packed_bytes(c) == size, "glyd.gpu.compress packs as from_pretrained does"
    print(f"   glyd.gpu.compress: the same {size / 1e9:.2f} GB, logits and tokens bit for bit")
    print(f"   compiled (a static cache, CUDA graphs, fullgraph): tokens as plain generate()'s: {same(out_b, compiled(m, ids))} of {TOKENS}")
    del c
    torch.cuda.empty_cache()

    x, t, peak, held = loaded(lambda: glyd.from_pretrained(name, exact=True))
    logits_x, out_x = run(x, ids)
    assert exact(logits_a, logits_x) and torch.equal(out_a, out_x), "exact=True: bf16's logits and tokens"
    assert exact(ahead(x, long), long_a), "exact=True: a long prompt's logits bf16's"
    print(f"{name}: glyd exact loaded in {t:.1f} s, peak {peak:.2f} GB, holds {held:.2f} GB; logits bit-identical: True (and a prompt of {long.shape[1]} tokens'); generated tokens identical to bf16: {TOKENS} of {TOKENS}")
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
