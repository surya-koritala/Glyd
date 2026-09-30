"""Whether generate()'s numbers repeat on this GPU, call by call. Two identical greedy generate() calls in one process
(A, then B: respond.py's prompt, its first --prompt tokens, --batch copies), every F.linear and
scaled_dot_product_attention call hashed on the GPU in order: its inputs' and output's words (and at the prompt's
forward each linear's weight), its pointers mod 256, strides and stream. The first call where B's output differs from
A's, and whether its inputs did, names the operation that does not repeat. A third call (C: the prompt and one step)
runs under the profiler, each call labelled: its CUDA kernels. bf16: transformers' own model; exact:
glyd.from_pretrained(MODEL, exact=True), whose products are F.linear on each matrix decoded whole. --attn holds
scaled_dot_product_attention to one backend; --det sets torch.use_deterministic_algorithms and
CUBLAS_WORKSPACE_CONFIG=:4096:8; --blas, PyTorch's preferred BLAS library (cublas or cublaslt).

    python diag.py MODEL --mode bf16|exact --out RUN.json [--batch 32] [--prompt 128] [--new 32] [--attn math|efficient|flash|cudnn] [--det] [--blas cublas|cublaslt]
    python diag.py --compare X.json Y.json    X's run A against Y's: the first call whose output differs, and the
                                              linear calls whose inputs are the same and outputs not"""
import argparse, bisect, contextlib, hashlib, json, os, sys, tempfile

ap = argparse.ArgumentParser()
ap.add_argument("model", nargs="?")
ap.add_argument("--mode", choices=["bf16", "exact"])
ap.add_argument("--out")
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--prompt", type=int, default=128)
ap.add_argument("--new", type=int, default=32)
ap.add_argument("--attn", choices=["math", "efficient", "flash", "cudnn"])
ap.add_argument("--det", action="store_true")
ap.add_argument("--blas", choices=["cublas", "cublaslt"])
ap.add_argument("--compare", nargs=2, metavar=("X", "Y"))
args = ap.parse_args()


def first_diff(a, b):
    """Two runs' calls: the first whose output differs (with whether its inputs were the same), else None."""
    for i, (x, y) in enumerate(zip(a, b)):
        if x["kind"] != y["kind"] or x["out"] != y["out"]:
            return {"i": i, "step": x["step"], "kind": x["kind"], "kinds_same": x["kind"] == y["kind"], "inputs_same": x["in"] == y["in"]}
    return None if len(a) == len(b) else {"i": min(len(a), len(b)), "why": f"{len(a)} calls against {len(b)}"}


def where(calls, i):
    """Call i's place: its step, its index within the step, and (a linear) which of the step's linears it is."""
    c = calls[i]
    first = next(j for j in range(i + 1) if calls[j]["step"] == c["step"])
    return {"step": c["step"], "in_step": i - first, **{k: c[k] for k in ("kind", "x", "w", "q", "mask", "stream") if k in c}}


if args.compare:
    X, Y = (json.load(open(p)) for p in args.compare)
    a, b = X["A"], Y["A"]
    d = first_diff(a, b)
    same_in = [(x, y) for x, y in zip(a, b) if x["kind"] == "linear" == y["kind"] and x["in"] == y["in"]]
    bad = [i for i, (x, y) in enumerate(zip(a, b)) if x["kind"] == "linear" == y["kind"] and x["in"] == y["in"] and x["out"] != y["out"]]
    print(f"{args.compare[0]} ({X['mode']}) run A against {args.compare[1]} ({Y['mode']}) run A: batch {X['batch']}, attention {X['attn'] or 'default'}{', deterministic' if X['det'] else ''}")
    print(f"  tokens (every sequence): {'the same' if X['tokens']['A_all'] == Y['tokens']['A_all'] else 'differ'}; calls {len(a)} and {len(b)}")
    print(f"  linear calls with the same inputs: {len(same_in)}, of them with another output: {len(bad)}" + (f" (the first: call {bad[0]}, {where(a, bad[0])})" if bad else ""))
    ws = [(x.get("weight"), y.get("weight")) for x, y in zip(a, b) if x["kind"] == "linear" == y["kind"] and x["step"] == 0 == y["step"]]
    print(f"  the prompt's linear calls' weights: {sum(p == q for p, q in ws)} of {len(ws)} the same words")
    if d is None:
        print("  every call's output the same")
    else:
        print(f"  the first call whose output differs: {d}")
        for name, R, calls in ((X["mode"], X, a), (Y["mode"], Y, b)):
            w = where(calls, d["i"])
            k = R["kernels"].get(str(min(w["step"], 1)), [])
            print(f"    {name}: {w}; kernels (the profiled step {min(w['step'], 1)}): {k[w['in_step']] if w['in_step'] < len(k) else '?'}")
    sys.exit(0)

if args.det:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # before the first cuBLAS handle
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

if args.det:
    torch.use_deterministic_algorithms(True, warn_only=True)
if args.blas:
    torch.backends.cuda.preferred_blas_library(args.blas)
TEXT = ("The history of data compression begins long before computers. Telegraph operators shortened common words to "
        "save time on the wire, and Morse gave the most frequent letters the shortest codes. Shannon later showed that "
        "the average length of any code is bounded below by the entropy of its source, and Huffman found the optimal "
        "prefix code for a known distribution. Arithmetic coding and its descendants approached that bound more closely "
        "still, while dictionary methods such as those of Lempel and Ziv replaced repeated strings with references to "
        "earlier text. Today the same ideas shrink images, audio, genomes and the weights of neural networks, where the "
        "exponents of floating-point numbers are far more predictable than their mantissas. ")  # respond.py's
tok = AutoTokenizer.from_pretrained(args.model)
if args.mode == "bf16":
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda:0").eval()
else:
    import glyd
    model = glyd.from_pretrained(args.model, exact=True, compile=False).eval()
ids = tok(TEXT * (args.prompt // 100 + 2), return_tensors="pt").input_ids[0][: args.prompt].repeat(args.batch, 1).cuda()
if args.attn:  # a context for each generate() (sdpa_kernel's enters once)
    from torch.nn.attention import SDPBackend, sdpa_kernel
    attn = lambda: sdpa_kernel({"math": SDPBackend.MATH, "efficient": SDPBackend.EFFICIENT_ATTENTION, "flash": SDPBackend.FLASH_ATTENTION, "cudnn": SDPBackend.CUDNN_ATTENTION}[args.attn])
else:
    attn = contextlib.nullcontext


def hsh(t):
    """A tensor's words (int16 for 2-byte elements, int32 for 4, else each as an int16): their sum and their sum
    weighted by position (mod 65521, plus 1), int64 on its device, 2^24 words at a time."""
    v = t.detach().contiguous().view(-1)
    v = v.view(torch.int16) if v.element_size() == 2 else v.view(torch.int32) if v.element_size() == 4 else v.to(torch.int16)
    s = torch.zeros(2, dtype=torch.int64, device=v.device)
    for i in range(0, v.numel(), 1 << 24):
        c = v[i : i + (1 << 24)].to(torch.int64)
        s += torch.stack([c.sum(), (c * (torch.arange(i, i + c.numel(), device=v.device) % 65521 + 1)).sum()])
    return s


rec, step, prof = None, [0], [False]  # the run's calls (None: not recording), its forward, whether labelled for the profiler


def meta(t):
    return [list(t.shape), list(t.stride()), t.data_ptr() % 256]


def recorded(kind, fn, info, ins):
    n = len(rec) if rec is not None else 0
    if prof[0]:
        with torch.profiler.record_function(f"diag#{n}"):
            out = fn()
    else:
        out = fn()
    if rec is not None:
        e = dict(kind=kind, step=step[0], stream=torch.cuda.current_stream().cuda_stream, **info)
        rec.append((e, [hsh(x) for x in ins] + [hsh(out)] + ([hsh(info.pop("_w"))] if "_w" in info else [])))
        e.pop("_w", None)
    return out


own_linear, own_sdpa = F.linear, F.scaled_dot_product_attention


def linear(x, w, b=None):
    info = {"x": meta(x), "w": meta(w)}
    if step[0] == 0:
        info["_w"] = w  # the weight's words at the prompt's forward (the decoded matrix's, in exact)
    return recorded("linear", lambda: own_linear(x, w, b), info, [x] + ([b] if b is not None else []))


def sdpa(*a, **kw):
    q, k, v = (a[i] if i < len(a) else kw[n] for i, n in enumerate(("query", "key", "value")))
    m = kw.get("attn_mask", a[3] if len(a) > 3 else None)
    info = {"q": meta(q), "mask": None if m is None else [list(m.shape), str(m.dtype)], "is_causal": kw.get("is_causal", False)}
    return recorded("sdpa", lambda: own_sdpa(*a, **kw), info, [q, k, v] + ([m] if m is not None else []))


F.linear = torch.nn.functional.linear = linear
F.scaled_dot_product_attention = torch.nn.functional.scaled_dot_product_attention = sdpa
model.register_forward_pre_hook(lambda mod, inp: step.__setitem__(0, step[0] + 1))


def generate(n):
    global rec
    rec, step[0] = [], -1
    with torch.no_grad(), attn():
        out = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=n, min_new_tokens=n, do_sample=False, pad_token_id=tok.eos_token_id)
    torch.cuda.synchronize()
    flat, at, calls = torch.cat([x for _, h in rec for x in h]).cpu().tolist(), 0, []
    for e, h in rec:
        v, at = flat[at : at + 2 * len(h)], at + 2 * len(h)
        n_in = (len(v) - 2 - (2 if e["step"] == 0 and e["kind"] == "linear" else 0)) // 2
        e["in"], e["out"] = v[: 2 * n_in], v[2 * n_in : 2 * n_in + 2]
        if len(v) > 2 * n_in + 2:
            e["weight"] = v[2 * n_in + 2 :]
        calls.append(e)
    rec = None
    new = out[:, ids.shape[1] :].cpu()
    return calls, {"row0": hashlib.sha256(new[0].numpy().tobytes()).hexdigest()[:16], "all": hashlib.sha256(new.numpy().tobytes()).hexdigest()[:16], "rows_same": bool((new == new[0]).all())}


R = {"model": args.model, "mode": args.mode, "batch": args.batch, "prompt": args.prompt, "new": args.new, "attn": args.attn, "det": args.det, "blas_asked": args.blas,
     "gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
     "blas": str(torch.backends.cuda.preferred_blas_library()),
     "sdpa_enabled": {b: getattr(torch.backends.cuda, f"{b}_sdp_enabled")() for b in ("flash", "mem_efficient", "math", "cudnn")}}
A, ta = generate(args.new)
B, tb = generate(args.new)
d = first_diff(A, B)
R.update(tokens={"A": ta["row0"], "B": tb["row0"], "A_all": ta["all"], "B_all": tb["all"], "rows_same": [ta["rows_same"], tb["rows_same"]]}, first_diff_ab=d)
# C: the prompt and a step under the profiler, each call labelled: its kernels, by step and index within the step
prof[0] = True
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as p:
    C, _ = generate(2)
prof[0] = False


with tempfile.NamedTemporaryFile(suffix=".json") as f:  # each kernel to the labelled call whose time held its launch
    p.export_chrome_trace(f.name)
    tr = json.load(open(f.name))["traceEvents"]
spans = sorted((e["ts"], e["ts"] + e.get("dur", 0), int(e["name"][5:])) for e in tr if e.get("ph") == "X" and str(e.get("name", "")).startswith("diag#"))
launch = {e["args"]["correlation"]: e["ts"] for e in tr if e.get("cat") in ("cuda_runtime", "cuda_driver") and "correlation" in e.get("args", {})}
lab = {}
for e in tr:
    t = launch.get(e.get("args", {}).get("correlation")) if e.get("cat") == "kernel" else None
    j = bisect.bisect_right([s0 for s0, _, _ in spans], t) - 1 if t is not None else -1
    if j >= 0 and spans[j][0] <= t <= spans[j][1]:
        lab.setdefault(spans[j][2], []).append(e["name"])
R["kernels"] = {}
for i, c in enumerate(C):
    R["kernels"].setdefault(str(c["step"]), []).append(lab.get(i, []))
R["A"] = A
R["B"] = [{"kind": c["kind"], "step": c["step"], "in": c["in"], "out": c["out"]} for c in B]
json.dump(R, open(args.out, "w"))
same_in = sum(1 for x, y in zip(A, B) if x["in"] == y["in"] and x["out"] != y["out"])
line = (f"{args.mode}, batch {args.batch}, {args.prompt} + {args.new} tokens, attention {args.attn or 'default'}{', deterministic' if args.det else ''} "
        f"({R['gpu']}, torch {R['torch']}, SDPA {R['sdpa_enabled']}): tokens A {ta['row0']} B {tb['row0']} ({'the same' if ta['all'] == tb['all'] else 'DIFFER'}; "
        f"every sequence the same in each: {ta['rows_same']}, {tb['rows_same']}); {len(A)} calls, {same_in} with the same inputs and another output; ")
if d is None:
    line += "every call's output the same"
else:
    w = where(A, d["i"])
    k = R["kernels"].get(str(min(w["step"], 1)), [])
    line += f"the first call whose output differs: {d} {w}; its kernels: {k[w['in_step']] if w['in_step'] < len(k) else '?'}"
print(line)
