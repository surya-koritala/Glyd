"""End to end: a Hugging Face causal LM generating with its weights held
compressed in VRAM (glyd_gpu), against the same model in bf16.

    python e2e.py MODEL_DIR --format fast|huffman|mma [--fused | --exact] [--baseline] [--tokens N]

Decoded path (default): each matrix decoded into a scratch buffer (in
row blocks past 128M weights), then PyTorch's own matmul. --exact: every
matrix decoded whole, into one scratch buffer the size of the largest,
and multiplied by F.linear on the input as it came, as the bf16 model's
nn.Linear does: logits and tokens bit-identical to bf16's. --fused:
one-token steps multiply straight from the packed weights (decoded in
registers, never written out); their sums are in another order than
cuBLAS's, as between any two GEMM kernels, so late tokens may differ.
The bf16 model is never held on the GPU: every Linear is packed from
the CPU copy, one at a time. --compile: generate() as transformers
compiles it (a static cache, the forward under torch.compile's CUDA
graphs), for bf16 and Glyd alike.
"""
import argparse, math, time, torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import glyd_gpu as g
from glyd.gpu import moe
from glyd.gpu.model import GEmbedding, GLinear, Scratch, decoder, merge_linears, pack_modules, plain, set_scratch

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--format", default="fast", choices=["fast", "huffman", "mma", "mma12", "auto"], help="mma: the Linears in the mma layout (the embedding in fast), up to 64 tokens a step multiplied straight from it; mma12: its 12-bit layout (a lighter decode); auto: the one for this GPU (glyd_gpu.best_layout)")
ap.add_argument("--fused", action="store_true")
ap.add_argument("--exact", action="store_true", help="every Linear's matrix decoded whole into one scratch buffer, then F.linear as the bf16 model's: its logits bit for bit (overrides --fused)")
ap.add_argument("--baseline", action="store_true")
ap.add_argument("--tokens", type=int, default=128)
ap.add_argument("--prefill", type=str, default="", help="prompt lengths to time one forward pass and generate() to its first token at, e.g. 128,512,2048")
ap.add_argument("--gemm-max", type=int, default=64, help="fused: steps of up to this many tokens multiply straight from the packed weights (fast format)")
ap.add_argument("--batch", type=str, default="1", help="generate for this many copies of the prompt at once (comma list: each measured)")
ap.add_argument("--gpus", type=int, default=1, help="spread the layers over this many GPUs (bf16: accelerate's device map; glyd: layers balanced by packed size)")
ap.add_argument("--ppl", default="", help="a text file: perplexity over windows of --ppl-window tokens (a forward pass each; 12800 tokens from its 10th MB), and how often the next-token choice is bf16's")
ap.add_argument("--ppl-window", type=int, default=64)
ap.add_argument("--kv", type=str, default="", help="prompt lengths (tokens of enwik8, needs --ppl) to generate --tokens after with the KV cache compressed (gpu/kv.py) against the plain cache: the same tokens, its bytes, the time; with --batch's first size")
ap.add_argument("--smi", default="", help="once a model is on its GPUs: nvidia-smi's report into PREFIX-bf16.txt / PREFIX-glyd.txt")
ap.add_argument("--mmlu", type=int, default=0, help="MMLU (cais/mmlu, test split, a fixed shuffle): accuracy over this many questions, 0-shot, by the answer letter's logit")
ap.add_argument("--profile", type=int, default=0, help="GPU time by kernel over this many generated tokens, against the wall clock, at each --batch size: a step after the prompt, and the prompt's; run once every timing of the run is taken (a profiler session leaves CUPTI's callbacks on, every launch after it slower), bf16's on the model loaded again")
ap.add_argument("--from-pretrained", action="store_true", help="Glyd as glyd.from_pretrained loads it, the package's path (--format mma, mma12 or auto; --merge; --exact; one GPU), in place of this script's packing")
ap.add_argument("--prompts", action="store_true", help="a batch of different prompts (left-padded), not copies of one: a mixture of experts routes each to its own experts")
ap.add_argument("--merge", action="store_true", help="the Linears that take the same input (q, k, v; gate, up) as one product each, for bf16 and Glyd alike, as serving engines run them")
ap.add_argument("--gpu-mem", type=float, default=0, help="GiB a GPU may hold of bf16 weights (the baseline's device map); default: all but 2 GiB")
ap.add_argument("--compile", action="store_true", help="generate() compiled as transformers compiles it: a static cache, the forward under torch.compile (reduce-overhead: CUDA graphs); each batch's warm-up, of --tokens, compiles and captures")
args = ap.parse_args()

tok = AutoTokenizer.from_pretrained(args.model)
prompt = "The history of data compression began"
ids = tok(prompt, return_tensors="pt").input_ids.cuda()
PROMPTS = [prompt, "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n", "Q: A train leaves at 3:40 pm and the trip takes 2 hours 35 minutes. When does it arrive?\nA:", "The capital of Australia is",
           "Translate to French: The weather is lovely today, so we will walk to the market.", "In quantum mechanics, the uncertainty principle states that", "SELECT name, COUNT(*) FROM orders JOIN customers ON", "Once upon a time, in a village at the edge of a great forest,"]


def batch_of(b):
    """generate()'s inputs for b sequences: b copies of the prompt, or (--prompts) b of PROMPTS in turn, left-padded."""
    if not args.prompts:
        return {"inputs": ids.repeat(b, 1)}
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return dict(tok([PROMPTS[i % len(PROMPTS)] for i in range(b)], return_tensors="pt", padding=True).to("cuda"), pad_token_id=tok.pad_token_id)


def prefill(model, label):
    """One forward pass over a prompt of each length: the prompt's tokens a second; and generate() to its first
    token (the time to first token); each timed after two untimed (Glyd's first prompt long enough to decode its
    matrices ahead records their order, model.Ahead). The prompt: token ids drawn below the model's vocabulary, the
    same for bf16 and Glyd (a mixture of experts routes each its own way)."""
    out = []
    for n in [int(x) for x in args.prefill.split(",") if x]:
        x = torch.randint(0, vocab, (1, n), generator=torch.Generator().manual_seed(n)).cuda()
        first = dict(attention_mask=torch.ones_like(x), max_new_tokens=1, do_sample=False, pad_token_id=tok.eos_token_id)
        with torch.no_grad():
            ts = []
            for f in (lambda: model(x, logits_to_keep=1), lambda: model.generate(x, **first)):
                f()
                f()
                torch.cuda.synchronize()
                t = time.perf_counter()
                for _ in range(3):
                    f()
                torch.cuda.synchronize()
                ts.append((time.perf_counter() - t) / 3)
        out.append(f"{n} tokens {ts[0] * 1e3:.1f} ms ({n / ts[0]:.0f} tokens/s), first token {ts[1] * 1e3:.1f} ms")
    if out:
        print(f"{label} prefill: " + ", ".join(out))


def profile(model, label):
    """Where a generated token's time goes: the GPU's kernels, and the rest."""
    if not args.profile:
        return
    from torch.profiler import profile as prof_, ProfilerActivity
    n = args.profile

    def run(x, k):  # GPU time by kernel (us) and wall time (s) of generating k tokens
        with prof_(activities=[ProfilerActivity.CUDA]) as pr:
            t = time.perf_counter()
            model.generate(**x, max_new_tokens=k, min_new_tokens=k, do_sample=False)
            torch.cuda.synchronize()
            wall = time.perf_counter() - t
        return {e.key: e.device_time_total for e in pr.key_averages() if e.device_type.name == "CUDA"}, wall

    for b in [int(v) for v in args.batch.split(",")]:
        x = batch_of(b)
        with torch.no_grad():
            model.generate(**x, max_new_tokens=4, do_sample=False)
            torch.cuda.synchronize()
            one, _ = run(x, 1)  # the prompt and a token
            all_, wall = run(x, n + 1)
        # A step after the prompt: n + 1 tokens' time less 1 token's, over n.
        step = {k: (v - one.get(k, 0)) / n for k, v in all_.items()}
        busy, prompt = sum(step.values()) / 1000, sum(one.values()) / 1000
        print(f"{label} profile, batch {b}: {wall / (n + 1) * 1000:.2f} ms a step, GPU busy {busy:.2f} ms a step after the prompt ({b / busy * 1000:.0f} tokens/s of GPU time), {prompt:.2f} ms for the prompt and a token")
        for k, v in sorted(step.items(), key=lambda kv: -kv[1])[:6]:
            print(f"   {v / 1000:7.3f} ms  {k[:90]}")


def perplexity(model, label):
    if not args.ppl:
        return None
    text = open(args.ppl, "rb").read()[10_000_000:10_400_000].decode("utf-8", "ignore")
    n = 12800 // args.ppl_window
    windows = tok(text, return_tensors="pt").input_ids[0][: n * args.ppl_window].view(n, args.ppl_window).cuda()
    nll, top = 0.0, []
    with torch.no_grad():
        for w in windows:
            lg = model(w[None]).logits[0, :-1].float()
            nll += F.cross_entropy(lg, w[1:], reduction="sum").item()
            top.append(lg.argmax(-1))
    top = torch.stack(top)
    print(f"{label} perplexity: {math.exp(nll / top.numel()):.4f} ({top.numel()} tokens)")
    return top


def smi(what):
    if args.smi:
        import subprocess
        torch.cuda.synchronize()
        report = subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout
        open(f"{args.smi}-{what}.txt", "w").write(report)


def kv_check(model, label):
    if not args.kv:
        return
    from kv import GlydKVCache, use_fused_attention
    use_fused_attention(model)  # SDPA's, save for a fused cache's one-token steps
    text = open(args.ppl, "rb").read()[10_000_000:12_000_000].decode("utf-8", "ignore")
    b = int(args.batch.split(",")[0])
    for T in [int(x) for x in args.kv.split(",")]:
        x = tok(text, return_tensors="pt").input_ids[:, :T].repeat(b, 1).cuda()
        res = {}
        for name in ("plain", "packed", "fused"):
            make = lambda: None if name == "plain" else GlydKVCache(model.config, fused=name == "fused")
            with torch.no_grad():
                model.generate(x[:, :64], max_new_tokens=2, do_sample=False, past_key_values=make())  # warm-up
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                model.generate(x, max_new_tokens=1, do_sample=False, past_key_values=make())  # the prompt alone
                torch.cuda.synchronize()
                t1 = time.perf_counter() - t1
                for i in range(torch.cuda.device_count()):
                    torch.cuda.reset_peak_memory_stats(i)
                t = time.perf_counter()
                o = model.generate(x, max_new_tokens=args.tokens, min_new_tokens=args.tokens, do_sample=False, past_key_values=make(), return_dict_in_generate=True)
                torch.cuda.synchronize()
                t = time.perf_counter() - t
            peak = sum(torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())) / 1e9
            kv = o.past_key_values
            size = kv.nbytes() if name != "plain" else sum(l.keys.numel() * 2 + l.values.numel() * 2 for l in kv.layers)
            res[name] = (o.sequences[:, T:], (t - t1) / (args.tokens - 1), peak, size)
            del o, kv  # the next run's peak without this one's cache
            torch.cuda.empty_cache()
        # Quality through the fused steps: the text's next 256 tokens fed one at a time after the prompt.
        y = tok(text, return_tensors="pt").input_ids[:, T : T + 257].cuda()
        q = {}
        for name in ("plain", "fused"):
            with torch.no_grad():
                out = model(x[:1], past_key_values=GlydKVCache(model.config, fused=True) if name == "fused" else None, use_cache=True, logits_to_keep=1)
                kv, nll, top = out.past_key_values, 0.0, []
                for i in range(256):
                    out = model(y[:, i : i + 1], past_key_values=kv, use_cache=True)
                    lg = out.logits[0, -1].float()
                    nll += F.cross_entropy(lg[None], y[0, i + 1 : i + 2]).item()
                    top.append(int(lg.argmax()))
            q[name] = (math.exp(nll / 256), top)
            del out, kv
        agree = sum(a == b for a, b in zip(q["plain"][1], q["fused"][1])) / 256 * 100
        print(f"{label} KV cache, {T}-token prompt, 256 steps fed: perplexity plain {q['plain'][0]:.4f}, fused {q['fused'][0]:.4f}; next token as plain's {agree:.2f}%")
        sa, ta, pa, ka = res["plain"]
        line = f"{label} KV cache, {T}-token prompt, batch {b}, {args.tokens} new tokens: plain {ka / 1e9:.3f} GB, {1000 * ta:.1f} ms a step, peak {pa:.2f} GB"
        for name in ("packed", "fused"):
            sb, tb, pb, kb = res[name]
            same = (sa == sb).all(0).long().cumprod(0).sum().item()
            line += f"; {name} {kb / 1e9:.3f} GB ({100 * kb / ka:.1f}%), {1000 * tb:.1f} ms a step, peak {pb:.2f} GB, tokens as plain's: {same} of {args.tokens}"
        print(line)


def mmlu(model, label):
    if not args.mmlu:
        return None
    from datasets import load_dataset
    qs = load_dataset("cais/mmlu", "all", split="test").shuffle(seed=0).select(range(args.mmlu))
    letters = [tok(f" {c}", add_special_tokens=False).input_ids[-1] for c in "ABCD"]
    picks, right = [], 0
    with torch.no_grad():
        for q in qs:
            prompt = f"The following is a multiple choice question about {q['subject'].replace('_', ' ')}.\n\n{q['question']}\n"
            prompt += "".join(f"{c}. {a}\n" for c, a in zip("ABCD", q["choices"])) + "Answer:"
            x = tok(prompt, return_tensors="pt").input_ids.cuda()
            pick = int(model(x, logits_to_keep=1).logits[0, -1, letters].argmax())
            picks.append(pick)
            right += pick == q["answer"]
    print(f"{label} MMLU: {100 * right / len(picks):.2f}% of {len(picks)} questions (0-shot)")
    return torch.tensor(picks)


def measure(model, label):
    torch.cuda.synchronize()
    out = None
    kw = dict(cache_implementation="static") if args.compile else {}
    label += " compiled" if args.compile else ""
    with torch.no_grad():
        logits = model(ids, logits_to_keep=1).logits
        for b in [int(x) for x in args.batch.split(",")]:
            batch = batch_of(b)
            model.generate(**batch, max_new_tokens=4, do_sample=False)  # warm-up, eager (the JIT build makes a kernel's done counters at its first call: never in a CUDA graph's memory pool)
            if args.compile:
                model.generate(**batch, max_new_tokens=args.tokens, do_sample=False, **kw)  # compiles and captures, the static cache the timed run's size
            torch.cuda.synchronize()
            for i in range(torch.cuda.device_count()):
                torch.cuda.reset_peak_memory_stats(i)
            t = time.perf_counter()
            o = model.generate(**batch, max_new_tokens=args.tokens, min_new_tokens=args.tokens, do_sample=False, **kw)
            torch.cuda.synchronize()
            t = time.perf_counter() - t
            peaks = [torch.cuda.max_memory_allocated(i) / 1e9 for i in range(torch.cuda.device_count())]
            used = [f"{x:.1f}" for x in peaks if x > 0.05]
            print(f"{label}: batch {b}: {b * args.tokens / t:.1f} tokens/s ({args.tokens / t:.1f} a sequence), peak VRAM {sum(peaks):.2f} GB" + (f" ({' + '.join(used)} GB on {len(used)} GPUs)" if len(used) > 1 else ""))
            out = o[:1] if out is None else out
    return logits, out


def load():
    """The model in bf16 on the host, its Linears merged with --merge."""
    try:
        m = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).eval()
    except ValueError:  # a checkpoint transformers loads only with its vision tower (Muse Glimmer)
        from transformers import AutoModelForImageTextToText
        m = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16).eval()
    if args.merge:
        print(f"merged: {merge_linears(m)} groups of Linears as one product each")
    return m


def on_gpus(m):
    """bf16 on --gpus GPUs, as --baseline runs it (several: accelerate's device map)."""
    if args.gpus > 1:
        from accelerate import dispatch_model, infer_auto_device_map
        cap = args.gpu_mem or torch.cuda.get_device_properties(0).total_memory / 2**30 - 2
        dm = infer_auto_device_map(m, max_memory={i: f"{cap}GiB" for i in range(args.gpus)}, no_split_module_classes=m._no_split_modules)
        dispatch_model(m, device_map=dm)
    else:
        m.cuda()


# A profiler session leaves CUPTI's callbacks on for the process, every CUDA launch after it slower (on an RTX 4080
# SUPER some 20% of Glyd's tokens/s, 9% of bf16's: more launches a second), so every timing of the run, bf16's and
# Glyd's alike, is taken before the first profile; the profiles run last.
model = load()
weights_bf16 = sum(p.numel() * p.element_size() for p in model.parameters())
vocab = model.get_input_embeddings().weight.shape[0]  # prefill's token ids are drawn below it
if args.baseline:
    on_gpus(model)
    smi("bf16")
    logits_a, out_a = measure(model, f"bf16 (weights {weights_bf16 / 1e9:.2f} GB)")
    prefill(model, "bf16")
    top_a = perplexity(model, "bf16")
    mmlu_a = mmlu(model, "bf16")
    if args.gpus > 1:
        from accelerate.hooks import remove_hook_from_module
        remove_hook_from_module(model, recurse=True)
    if args.compile:  # the bf16 graphs and their memory
        torch._dynamo.reset()
        model.__dict__.pop("_compiled_call", None)
    model.cpu()
    torch.cuda.empty_cache()

# Where every decoder layer's weights go: contiguous runs of layers, balanced
# by their bytes, over args.gpus GPUs; the embedding on the first, the final
# norm and the output layer on the last.
if args.format == "auto":
    experts = moe.packable_bytes(model)  # a mixture of experts' too
    lin_bytes = sum(m.weight.numel() * 2 for m in model.modules() if plain(m) and m.weight.shape[0] % 64 == 0 and m.weight.shape[1] % 16 == 0) + experts
    args.format, why = g.best_layout(lin_bytes, weights_bf16 - lin_bytes, args.gpus, moe=experts > 0)
    print(f"auto: {args.format}, {why}")
t0 = time.perf_counter()
if args.from_pretrained:  # the package's path: the checkpoint loaded again, packed as it arrives
    import gc
    import glyd
    assert args.format in ("mma", "mma12") and args.gpus == 1, "--from-pretrained: the mma layouts, one GPU"
    del model
    gc.collect()
    model = glyd.from_pretrained(args.model, layout=args.format, exact=args.exact, merge=args.merge, compile=args.compile).eval()  # (eager but with --compile)
    packed = {id(m.p): m.p for m in model.modules() if isinstance(m, (GLinear, GEmbedding))}
else:
    layer_bytes = [sum(p.numel() for p in l.parameters()) for l in decoder(model).layers]
    per_gpu, acc, gpu_of = sum(layer_bytes) / args.gpus, 0, []
    for b in layer_bytes:
        gpu_of.append(min(args.gpus - 1, int(acc // per_gpu)))
        acc += b
    layer_of = {id(m): gpu_of[i] for i, l in enumerate(decoder(model).layers) for m in l.modules()}  # (no name holding the layers: freed with the model)
    last = args.gpus - 1

    def pack(w, linear):
        if args.format in ("mma", "mma12") and linear and w.shape[0] % 64 == 0 and w.shape[1] % 16 == 0:
            return g.pack_mma12(w) if args.format == "mma12" else g.pack_mma(w)
        return (g.pack if args.format == "huffman" else g.pack_fast)(w)

    with torch.no_grad():
        packed = pack_modules(model, pack, lambda m: torch.device("cuda", layer_of.get(id(m), 0 if isinstance(m, nn.Embedding) else last)), fused=args.fused, exact=args.exact, gemm_max=args.gemm_max)
        if args.format in ("mma", "mma12"):  # a mixture of experts: each layer's experts packed as one matrix (moe.py)
            moe.compress(model, args.format, lambda m: torch.device("cuda", layer_of.get(id(m), 0)), exact=args.exact)
        if args.gpus > 1:
            from accelerate import dispatch_model
            # the decoder's layers where they were packed, its final norm on the last GPU, everything else
            # it holds (embeddings, rotary tables) and any vision tower beside it on the first
            pre = next(n for n, m in model.named_modules() if m is decoder(model))
            dm = {"lm_head": last}
            for n, _ in decoder(model).named_children():
                if n == "layers":
                    dm.update({f"{pre}.layers.{i}": d for i, d in enumerate(gpu_of)})
                else:
                    dm[f"{pre}.{n}"] = last if n == "norm" else 0
            if pre != "model":
                dm.update({f"model.{n}": 0 for n, _ in model.model.named_children() if f"model.{n}" != pre})
            dispatch_model(model, device_map=dm)
        else:
            model.cuda()
        set_scratch(model, args.exact)
torch.cuda.empty_cache()
packed_bytes = sum(p.nbytes() for p in packed.values()) + moe.nbytes(model)
other = sum(p.numel() * p.element_size() for p in model.parameters()) + sum(m.bias.numel() * 2 for m in model.modules() if isinstance(m, GLinear) and m.bias is not None)
in_use = sum(torch.cuda.memory_allocated(i) for i in range(torch.cuda.device_count()))
scratch = sum(b.numel() * 2 for b in Scratch.buf.values())
mode = " exact" if args.exact else " fused" if args.fused else ""
print(f"{args.format}{mode}{' (from_pretrained)' if args.from_pretrained else ''}: packed in {time.perf_counter() - t0:.0f} s; weights {(packed_bytes + other) / 1e9:.2f} GB against {weights_bf16 / 1e9:.2f} GB bf16 ({100 * (packed_bytes + other) / weights_bf16:.1f}%), scratch {scratch / 1e9:.2f} GB, VRAM in use {in_use / 1e9:.2f} GB on {args.gpus} GPU{'s' if args.gpus > 1 else ''}")
smi("glyd")
logits_b, out_b = measure(model, f"glyd {args.format}{mode}")
prefill(model, f"glyd {args.format}")
top_b = perplexity(model, f"glyd {args.format}")
mmlu_b = mmlu(model, f"glyd {args.format}")
kv_check(model, f"glyd {args.format}")
profile(model, f"glyd {args.format}")
if args.baseline and top_b is not None:
    print(f"next-token choice as bf16's: {(top_a.cuda() == top_b).float().mean().item() * 100:.2f}%")
if args.baseline and mmlu_b is not None:
    print(f"MMLU answer as bf16's: {(mmlu_a == mmlu_b).float().mean().item() * 100:.2f}%")
if args.baseline:
    print("logits bit-identical:", torch.equal(logits_a.cuda().view(torch.int16), logits_b.view(torch.int16)))
    same = (out_a.cuda() == out_b).all(0).long().cumprod(0).sum().item() - ids.shape[1]
    print(f"generated tokens identical to bf16: {same} of {args.tokens}")
print("text:", tok.decode(out_b[0][ids.shape[1]:ids.shape[1] + 40]).replace("\n", " "))
if args.baseline and args.profile:  # bf16's profile last, on the model loaded again
    import gc
    if args.compile:  # Glyd's graphs and their memory
        torch._dynamo.reset()
        model.__dict__.pop("_compiled_call", None)
    del model, packed
    Scratch.buf.clear()
    Scratch.replaced.clear()
    gc.collect()
    torch.cuda.empty_cache()
    print(f"glyd's model freed: {torch.cuda.memory_allocated() / 1e9:.2f} GB left in use")
    model = load()
    on_gpus(model)
    profile(model, "bf16")
