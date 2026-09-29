"""How fast a model responds through generate(): its time to first token and its tokens a second, in one of three
modes, one mode a process (run each in a fresh one):

  bf16   transformers' own model in bf16 (AutoModelForCausalLM, dtype bf16), its generate() as it runs by default:
         eager, a dynamic cache;
  glyd   glyd.from_pretrained(MODEL) as it loads by default (layout "auto"; generate() compiled, a static cache and
         CUDA graphs, for a call whose cache holds at most 2048 positions in all, 1280 on a GeForce card; else eager);
  exact  glyd.from_pretrained(MODEL, exact=True): every product the matrix decoded whole, then F.linear (eager).

Greedy decoding; every call's new tokens forced to its count (min_new_tokens = max_new_tokens: no early end of
sequence); the same prompts in every mode (a fixed text's first N tokens, B copies for B sequences). Each call is
timed by a streamer, a put a token after the prompt's (transformers syncs each step on a GPU): its time to first token
(TTFT: from the call to the first new token's put), its tokens a second after that ((new - 1) x B over the first
put to the last), and its total. Per configuration, in this order (a compiled mode's static cache only grows, so the
shorter first): TTFT at each of --prompts (--ttft-new tokens after each), the chat mix, tokens a second at each of
--batches (--new tokens after a --rate-prompt-token prompt), the long-document mix. Each configuration's first call is
a warm-up (the first compile, a CUDA graph's capture, cuBLAS's plans), kept apart; then the median of its repeats.
A configuration past the model's context, out of memory or past --deadline is recorded so, and the rest go on; the
JSON is rewritten after each one.

    python respond.py MODEL --mode bf16|glyd|exact --out RESULT.json [--prompts 128,512,2048,8192] [--batches 1,8,32]
        [--new 256] [--mixes chat:200:300,long:2000:200] [--reps-ttft 5] [--reps-rate 3] [--reps-mix 3] [--deadline EPOCH]
"""
import argparse, hashlib, json, os, platform, statistics, subprocess, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.streamers import BaseStreamer

TEXT = ("The history of data compression begins long before computers. Telegraph operators shortened common words to "
        "save time on the wire, and Morse gave the most frequent letters the shortest codes. Shannon later showed that "
        "the average length of any code is bounded below by the entropy of its source, and Huffman found the optimal "
        "prefix code for a known distribution. Arithmetic coding and its descendants approached that bound more closely "
        "still, while dictionary methods such as those of Lempel and Ziv replaced repeated strings with references to "
        "earlier text. Today the same ideas shrink images, audio, genomes and the weights of neural networks, where the "
        "exponents of floating-point numbers are far more predictable than their mantissas. ")

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--mode", required=True, choices=["bf16", "glyd", "exact"])
ap.add_argument("--out", required=True)
ap.add_argument("--prompts", default="128,512,2048,8192")
ap.add_argument("--ttft-new", type=int, default=16)
ap.add_argument("--batches", default="1,8,32")
ap.add_argument("--rate-prompt", type=int, default=128)
ap.add_argument("--new", type=int, default=256)
ap.add_argument("--mixes", default="chat:200:300,long:2000:200")
ap.add_argument("--reps-ttft", type=int, default=5)
ap.add_argument("--reps-rate", type=int, default=3)
ap.add_argument("--reps-mix", type=int, default=3)
ap.add_argument("--deadline", type=float, default=0, help="seconds since the epoch: no configuration starts past it, and repeats are cut to fit")
args = ap.parse_args()


def smi(q):
    try:
        return subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader"], capture_output=True, text=True, timeout=30).stdout.strip().splitlines()[0]
    except Exception as e:
        return f"? ({e})"


cpu = next((l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")), platform.processor()) if os.path.exists("/proc/cpuinfo") else platform.processor()
R = {"model": args.model, "mode": args.mode, "greedy": True, "gpu": torch.cuda.get_device_name(), "capability": ".".join(map(str, torch.cuda.get_device_capability())),
     "gpu_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2), "smi": smi("driver_version,clocks.max.sm,clocks.max.mem,power.limit,power.default_limit"),
     "torch": torch.__version__, "cuda": torch.version.cuda, "transformers": __import__("transformers").__version__, "cpu": cpu, "cpus": os.cpu_count(),
     "host": platform.machine(), "args": vars(args), "configs": []}


def save():
    with open(args.out + ".tmp", "w") as f:
        json.dump(R, f, indent=1)
    os.replace(args.out + ".tmp", args.out)


tok = AutoTokenizer.from_pretrained(args.model)
torch.cuda.reset_peak_memory_stats()
t0 = time.perf_counter()
try:
    if args.mode == "bf16":
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda:0").eval()
    else:
        import glyd
        R["glyd"] = glyd.__version__
        model = glyd.from_pretrained(args.model, exact=args.mode == "exact").eval()
except torch.cuda.OutOfMemoryError as e:
    R["load"] = {"fits": False, "error": f"{type(e).__name__}: {str(e)[:300]}", "seconds": round(time.perf_counter() - t0, 1)}
    save()
    print(f"{args.mode}: does not fit: {R['load']['error']}", flush=True)
    raise SystemExit(0)
torch.cuda.synchronize()
q = getattr(model.config, "quantization_config", None)
R["load"] = {"fits": True, "seconds": round(time.perf_counter() - t0, 1), "gb": round(torch.cuda.memory_allocated() / 1e9, 2), "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
             "layout": getattr(q, "layout", None), "compiled_cap": getattr(model, "glyd_fast", None)}
print(f"{args.mode}: loaded in {R['load']['seconds']} s, {R['load']['gb']} GB (layout {R['load']['layout']}, compiled calls to {R['load']['compiled_cap']} positions)", flush=True)
path = {"compiled": None}
if args.mode == "glyd":  # which calls fast_generate compiles: its _fast's answer, recorded
    import glyd.gpu.model as gm
    own_fast = gm._fast

    def spy(*a, **k):
        b = own_fast(*a, **k)
        path["compiled"] = b is not None
        return b

    gm._fast = spy
ctx = getattr(model.config.get_text_config(), "max_position_embeddings", None)
need = max([int(x) for x in args.prompts.split(",") if x] + [args.rate_prompt] + [int(m.split(":")[1]) for m in args.mixes.split(",") if m])
ids = tok(TEXT * (need // 100 + 2), return_tensors="pt").input_ids[0]
assert ids.numel() >= need, (ids.numel(), need)


class Clock(BaseStreamer):
    """generate()'s puts, timed: the prompt's, then one a step."""

    def __init__(self):
        self.t = []

    def put(self, value):
        self.t.append(time.perf_counter())

    def end(self):
        pass


def call(L, B, n):
    x = ids[:L].repeat(B, 1).cuda()
    clock = Clock()
    path["compiled"] = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(x, attention_mask=torch.ones_like(x), max_new_tokens=n, min_new_tokens=n, do_sample=False, streamer=clock, pad_token_id=tok.eos_token_id)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    assert out.shape[1] == L + n and len(clock.t) == n + 1, (out.shape, len(clock.t))
    r = {"ttft": clock.t[1] - t0, "total": t1 - t0, "path": "eager" if args.mode != "glyd" else "compiled" if path["compiled"] else "eager"}
    if n > 1:
        r["tokens_per_s"] = (n - 1) * B / (clock.t[-1] - clock.t[1])
    r["sha"] = hashlib.sha256(out[0, L:].cpu().numpy().tobytes()).hexdigest()[:16]
    return r


configs = [(f"ttft {L}", L, 1, args.ttft_new, args.reps_ttft) for L in sorted(int(x) for x in args.prompts.split(",") if x)]
mixes = [m.split(":") for m in args.mixes.split(",") if m]
configs += [(f"mix {m[0]}", int(m[1]), 1, int(m[2]), args.reps_mix) for m in mixes[:1]]
configs += [(f"rate {B}", args.rate_prompt, B, args.new, args.reps_rate) for B in (int(x) for x in args.batches.split(",") if x)]
configs += [(f"mix {m[0]}", int(m[1]), 1, int(m[2]), args.reps_mix) for m in mixes[1:]]
for name, L, B, n, reps in configs:
    c = {"name": name, "prompt": L, "batch": B, "new": n}
    R["configs"].append(c)
    if ctx and L + n > ctx:
        c["skipped"] = f"past the model's context ({ctx})"
    elif args.deadline and time.time() >= args.deadline:
        c["skipped"] = "past the deadline"
    else:
        try:
            torch.cuda.reset_peak_memory_stats()
            c["warmup"] = call(L, B, n)
            left = (args.deadline - time.time()) if args.deadline else float("inf")
            c["reps"] = max(0, min(reps, int(left // max(c["warmup"]["total"], 1e-3))))
            runs = [call(L, B, n) for _ in range(c["reps"])]
            c["runs"] = runs
            c["peak_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
            for k in ("ttft", "total", "tokens_per_s"):
                if runs and k in runs[0]:
                    c[k] = statistics.median(r[k] for r in runs)
            c["path"] = runs[0]["path"] if runs else c["warmup"]["path"]
            c["sha"] = c["warmup"]["sha"]
            c["same_tokens"] = all(r["sha"] == c["sha"] for r in runs)
        except torch.cuda.OutOfMemoryError as e:
            c["skipped"] = f"out of memory: {str(e)[:200]}"
            torch.cuda.empty_cache()
    save()
    got = " ".join(f"{k} {c[k]:.4g}" for k in ("ttft", "total", "tokens_per_s") if k in c)
    print(f"{args.mode} {name} ({L} + {n} tokens, {B} a batch): {c.get('skipped') or got} [warm-up {c['warmup']['total']:.2f} s, {c.get('path')}, {c.get('reps')} repeats]" if "warmup" in c else f"{args.mode} {name}: {c.get('skipped')}", flush=True)
R["done"] = True
save()
