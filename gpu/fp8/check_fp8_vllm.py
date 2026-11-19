"""vLLM's own FP8 against --quantization glyd on an FP8 model directory (a slice made by make_slice.py, or a whole checkpoint), each in a vLLM of
its own: the FP8 Linears and experts held packed (the lossless FP8 codec; GLYD_F8_CODEC=identity: the bytes as they are) and decoded before every
product, which vLLM's FP8 kernel runs as before. What it checks, on the GPU it runs on:

- exact: 4 prompts (the last of about 1,900 tokens, so that the big products run too) in one batch, greedy: their 48 tokens, the log probability of
  each, and the top 20 log probabilities of every prompt position (prompt_logprobs) bit for bit those of stock vLLM, in `eager`
  (--enforce-eager), in `graph` (vLLM's default: torch.compile, CUDA graphs FULL_AND_PIECEWISE; inductor's deterministic mode in both runs,
  without which compiled logits are not always the same from one run to the next, stock's too), and in `graphonly` (CUDA graphs, no
  inductor), where a mismatch tells the compile apart from the capture, `graphq` (compiled, vLLM's quantization op for the activations, which a block-scaled checkpoint has by default and a per-tensor one does not), `graphpw` (compiled, piecewise CUDA graphs only) and `compile` (compiled,
  no CUDA graphs: GLM's KDA attention cannot be captured by either, in stock vLLM 0.30 as in Glyd); stock against itself again in `graph`: where it
  is not reproducible the comparison says INCONCLUSIVE, not PASS;
- the same kernel: the lines vLLM logs for its FP8 kernels' selection are the same in both;
- weights: the GiB vLLM logs for them ("Model loading took") in both, against stock's less what the packs save plus the scratches (the
  Linears' one and the experts' one), the Linears and experts' layers packed (apply_model), the KV cache's size and the concurrency at the
  longest context (--max-model-len, 8192 by default) as vLLM logs them;
- speed: output tokens a second at 1, 8 and 32 users (1,024-token prompts, 256 tokens out each; stock at 1 and 32 alone with --stock-brief), in the
  runs above (their compile settings: inductor's deterministic mode, --pin's autotuners, `graphq`'s flags; not vLLM's defaults); the greedy tokens of
  each batch size's first batch the same in both, or else the same as Glyd's again with stock's KV cache blocks (vLLM's scheduler admits requests by the
  blocks free: a hybrid model's every request takes a block for each group of its linear-attention layers, so a larger cache makes other batches);
- memory: torch's peak during the exact check's generate (its prompt log probabilities included, which vLLM's memory profile does not count), in both;
- --poison: Glyd once more (exact, the last mode, no speed) with every pack decoded back at load and compared with the bytes it was made from
  (GLYD_F8_VERIFY: every Linear, every expert), the Linears' scratch overwritten after every product and the experts buffer e4m3 NaN before every
  call (GLYD_F8_POISON: a read outside the op, of an expert no token is routed to, or across streams would move the logits): bit for bit stock's;
- --chat: the prompts in the model's own chat template (an instruction-tuned model's continuation of a bare prompt can be degenerate);
  --fused N: Glyd once more, exact off, with f8_linear for steps of up to N tokens (GLYD_F8_FUSED_MAX, a development build's), its speed alone.
  --stock-may-fail: a stock that does not start (the checkpoint does not fit at this memory) is an INFO line, not a FAIL: Glyd's run says
  whether it fits packed. `--modes serve --only stock`: stock alone at vLLM's own compile defaults (nothing set; the modes above set inductor's
  deterministic mode, `graphq` the fusions off), the speed reference.

    python check_fp8_vllm.py MODEL_DIR [--out DIR] [--modes eager,graph,graphonly] [--language-model-only] [--max-model-len N] [--no-bench] [--only stock|glyd] [--pin] [--no-again]
                             [--util F] [--max-num-seqs N] [--stock-brief] [--stock-may-fail] [--fused N] [--poison] [--chat]
    python check_fp8_vllm.py MODEL_DIR --diag [--stocks N] [--ops N] [--pin] [--out DIR] [--language-model-only]
    python check_fp8_vllm.py MODEL_DIR --seeds 0,1,2,3 [--x-bits JSON] [--out DIR] [--language-model-only]      (the same, one hash seed a pair of runs, the autotuners pinned unless --tune; Glyd again and with its scratch poisoned)
        (where `graph` differs: stock N times against Glyd N times, compiled, each on a sweep of single prompts' lengths and with the Triton autotuners' choices compared)

PASS / FAIL / INCONCLUSIVE / INFO lines, and a JSON (--out/check_fp8_vllm.json, every run's own JSON and log beside it; the summary keeps each run's
outputs but its prompt log probabilities, and the SHA-256 of its tokens, token and prompt log probabilities, so that which runs were equal can be read
without the runs' own JSONs); the exit code is 1 where anything FAILed. Needs vLLM 0.30 and glyd_gpu (this tree's on PYTHONPATH, or the wheel)."""
import json
import math
import os
import re
import statistics
import subprocess
import sys
import time
import types

TOKENS = 48
TOPK = 20
PROMPTS = [
    "The history of data compression began",
    'def fibonacci(n):\n    """Return the n-th Fibonacci number."""\n',
    "Q: A train leaves at 3:40 pm and the trip takes 2 hours 35 minutes. When does it arrive?\nA:",
]
LONG_TEXT = " ".join(f"Item {i} of the long list: the value is {i * 37 % 101} and the next is {(i * 53 + 7) % 211}." for i in range(80))
DETERMINISTIC = {"inductor_compile_config": {"deterministic": True, "combo_kernels": True, "benchmark_combo_kernel": False}}
MODES = {  # mode -> LLM's own arguments
    "eager": {"enforce_eager": True},
    "graph": {"compilation_config": DETERMINISTIC},
    "graphonly": {"compilation_config": {"mode": 0, "cudagraph_mode": "FULL"}},
    "graphpw": {"compilation_config": {**DETERMINISTIC, "cudagraph_mode": "PIECEWISE"}},
    "graphq": {"compilation_config": {**DETERMINISTIC, "custom_ops": ["+quant_fp8"], "pass_config": {"fuse_norm_quant": False, "fuse_act_quant": False}}},  # (compiled with vLLM's activation quantization op and without its norm and activation fusions, in both runs: else a per-tensor or per-channel Linear's quantization is fused into stock's graph and runs apart in Glyd's op: vllm_fp8.compiled_exact)
    "compile": {"compilation_config": {**DETERMINISTIC, "cudagraph_mode": "NONE"}},  # (compiled, no CUDA graphs: for a model whose attention cannot be captured at all, GLM's KDA)
    "serve": {},  # (vLLM's own defaults, as `vllm serve` compiles: for speed, with exact checked in another mode)
}
SWEEP = (16, 24, 32, 48, 64, 128, 256, 512, 1024, 1860)  # --diag: single prompts of these many tokens
PIN = {"F8_PIN_AUTOTUNE": "1", "PYTHONHASHSEED": "0"}  # --pin: Triton's autotuners keep their first config (the child), and Python hashes the same in every process (a set's order, which the order of what torch.compile does)
_SPEC = {}  # the child's spec, for probe (a function the engine runs: it carries no closure)
BENCH = {"users1": (1, 1024, 256), "users8": (8, 1024, 256), "users32": (32, 1024, 256)}  # name: (requests at once, prompt tokens, tokens out)
BRIEF = ("users1", "users32")  # stock's with --stock-brief


def probe(model):
    """In the engine's process (LLM.apply_model): what is packed, what the GPU holds."""
    import torch
    from glyd_gpu import vllm_fp8

    n, packed, whole, scratch = vllm_fp8.stats(model) if _SPEC.get("quantization") else (0, 0, 0, 0)
    experts = sum(1 for m in model.modules() if getattr(m, "glyd_f8_moe", None))
    fused = sum(1 for r in vllm_fp8._REG.values() if r.fused is not None) if _SPEC.get("quantization") else 0
    verified = list(getattr(vllm_fp8, "VERIFIED", None) or [])  # (GLYD_F8_VERIFY: packs decoded back bit for bit at load, and their bytes)
    kinds = {}
    for m in model.modules():
        qm = getattr(m, "quant_method", None)
        if qm is not None:
            inner = getattr(qm, "inner", qm)
            kinds[type(inner).__name__] = kinds.get(type(inner).__name__, 0) + 1
    w = [(n_, tuple(p.shape), str(p.dtype)) for n_, p in model.named_parameters() if n_.endswith("layers.0.mlp.down_proj.weight") or n_.endswith("layers.0.mlp.gate_up_proj.weight")]
    import gc

    sizes, seen = {}, set()  # (the GPU storages Python reaches, by size: what the two runs hold apart from the KV cache)
    for o in gc.get_objects():
        try:
            if isinstance(o, torch.Tensor) and o.is_cuda and o.untyped_storage().data_ptr() not in seen:
                seen.add(o.untyped_storage().data_ptr())
                sizes[o.untyped_storage().nbytes()] = sizes.get(o.untyped_storage().nbytes(), 0) + 1
        except Exception:  # (a tensor without storage: fake, meta)
            pass
    census = {"storages": len(seen), "bytes": sum(k * v for k, v in sizes.items()), "top": sorted(((k * v, k, v) for k, v in sizes.items()), reverse=True)[:20]}
    return {"census": census, "packed": n, "experts_layers": experts, "fused": fused, "verified": verified, "packed_bytes": packed, "weight_bytes": whole, "scratch_bytes": scratch, "methods": kinds, "sample_weights": w, "allocated": torch.cuda.memory_allocated()}


def pin_autotune():
    """Every Triton autotuner (vLLM's gated delta net and KDA kernels among them) keeps its first config: it picks its config by timing, in each process
    on its own, and another config is another tile and another order of accumulation, so two runs of the same model can differ in a long prompt's logits
    (stock's own too) without anything having changed. The same in both runs of a comparison."""
    import triton.runtime.autotuner as at

    init = at.Autotuner.__init__

    def pinned(self, *a, **k):
        init(self, *a, **k)
        self.configs = self.configs[:1]

    at.Autotuner.__init__ = pinned


def child(spec):
    """One vLLM (spec: model, quantization, mode, language_model_only, max_model_len, bench): the exactness probes, the weights' accounting, the speed."""
    if os.environ.get("F8_PIN_AUTOTUNE") == "1":
        pin_autotune()
    import torch
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    kw = dict(model=spec["model"], dtype="bfloat16", seed=0, max_model_len=spec["max_model_len"], gpu_memory_utilization=spec.get("util", 0.85), max_num_seqs=spec.get("max_num_seqs", 64), enable_prefix_caching=False)
    if spec.get("served_name"):
        kw["served_model_name"] = spec["served_name"]
    if spec.get("blocks"):  # (another run's KV cache blocks: the same batches as it, whatever memory this run's weights leave)
        kw["num_gpu_blocks_override"] = spec["blocks"]
    kw.update(MODES[spec["mode"]])
    if spec.get("quantization"):
        kw["quantization"] = spec["quantization"]
    if spec.get("language_model_only"):
        kw["language_model_only"] = True
    if spec.get("additional_config"):
        kw["additional_config"] = spec["additional_config"]
    _SPEC.update(spec)
    out = {"spec": spec, "gpu": torch.cuda.get_device_name(), "capability": list(torch.cuda.get_device_capability())}
    llm = LLM(**kw)
    cfg = llm.llm_engine.vllm_config
    out["quant_config"] = type(cfg.quant_config).__name__ if cfg.quant_config is not None else None
    out["quantization"] = cfg.model_config.quantization
    out["cudagraph_mode"], out["compile_mode"] = str(cfg.compilation_config.cudagraph_mode), str(cfg.compilation_config.mode)
    out["additional_config"] = cfg.additional_config if isinstance(cfg.additional_config, dict) else None
    out["blocks"], out["block_size"] = cfg.cache_config.num_gpu_blocks, cfg.cache_config.block_size

    try:
        out["probe"] = llm.apply_model(probe)[0]
    except Exception as e:  # (a vLLM that does not run a function in the model: the log's figures stand)
        out["probe"] = {"error": f"{type(e).__name__}: {e}"}
    tok = llm.get_tokenizer()
    prompts = PROMPTS + [LONG_TEXT]
    prompts = prompts if not spec.get("no_exact") else prompts[:1]
    sp = SamplingParams(temperature=0, max_tokens=TOKENS, logprobs=0, prompt_logprobs=TOPK)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    out["allocated_before_generate"] = torch.cuda.memory_allocated()  # (the engine runs in this process: VLLM_ENABLE_V1_MULTIPROCESSING=0)
    if spec.get("chat"):  # (the model's own chat template, as vLLM applies it: a user turn each)
        res = llm.chat([[{"role": "user", "content": p}] for p in prompts], sp, use_tqdm=False)
    else:
        res = llm.generate(prompts, sp, use_tqdm=False)
    out["peak_generate"] = torch.cuda.max_memory_allocated()
    out["prompt_tokens"] = [len(o.prompt_token_ids) for o in res]
    out["tokens"] = [list(o.outputs[0].token_ids) for o in res]
    out["logprobs"] = [[d[t].logprob for d, t in zip(o.outputs[0].logprobs, o.outputs[0].token_ids)] for o in res]
    out["prompt_logprobs"] = [[None if d is None else sorted([int(t), lp.logprob] for t, lp in d.items()) for d in o.prompt_logprobs] for o in res]
    out["text"] = [o.outputs[0].text for o in res]
    if spec.get("sweep"):  # (single prompts of growing length: where a difference between two runs starts)
        ids = tok.encode(LONG_TEXT)
        out["sweep"] = {}
        for n in SWEEP:
            o = llm.generate([TokensPrompt(prompt_token_ids=ids[:n])], SamplingParams(temperature=0, max_tokens=1, prompt_logprobs=5), use_tqdm=False)[0]
            out["sweep"][str(n)] = [None if d is None else sorted([int(t), lp.logprob] for t, lp in d.items()) for d in o.prompt_logprobs]
    if spec.get("bench"):
        import random

        def batch(n, L, seed):
            r = random.Random(seed)
            return [TokensPrompt(prompt_token_ids=[r.randrange(1000, 20000) for _ in range(L)]) for _ in range(n)]

        llm.generate(batch(32, 128, 1), SamplingParams(temperature=0, max_tokens=8, ignore_eos=True), use_tqdm=False)  # (warm-up: what a batch of this size compiles and captures)
        bench = {}
        for name, (n, L, T) in BENCH.items():
            if spec.get("brief") and name not in BRIEF:
                continue
            ts, toks = [], None
            for rep in range(3):
                b = batch(n, L, 100 + rep)
                t0 = time.perf_counter()
                o = llm.generate(b, SamplingParams(temperature=0, max_tokens=T, ignore_eos=True), use_tqdm=False)
                ts.append(time.perf_counter() - t0)
                toks = toks or [list(x.outputs[0].token_ids) for x in o]  # (the first batch's greedy tokens: compared between the runs)
            s = statistics.median(ts)
            bench[name] = {"seconds": s, "tokens_per_s": n * (L if T == 1 else T) / s, "runs": ts, "users": n, "prompt": L, "out": T, "tokens": toks}
        out["bench"] = bench
    return out


def run(spec, outdir, name, env=None):
    """spec in a vLLM of its own (this script, --child), its JSON, and the figures of its log (the weights, the KV cache, the FP8 kernels)."""
    path = os.path.join(outdir, name + ".json")
    env = dict(os.environ, GLYD_FP8="1", VLLM_ENABLE_V1_MULTIPROCESSING="0", TOKENIZERS_PARALLELISM="false", VLLM_ALLOW_INSECURE_SERIALIZATION="1", TRITON_PRINT_AUTOTUNING="1", **(env or {}))  # (LLM.apply_model sends its function to the engine; Triton says which config each autotuner picked)
    env.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")  # (no nvcc needed for the sampler's kernels: the same in both runs)
    if os.path.exists(path):
        os.remove(path)
    with open(os.path.join(outdir, name + ".log"), "w") as log:
        try:
            r = subprocess.run([sys.executable, os.path.abspath(__file__), "--child", json.dumps(spec), path], env=env, stdout=log, stderr=subprocess.STDOUT, timeout=int(os.environ.get("F8_RUN_TIMEOUT", 3600)))
        except subprocess.TimeoutExpired:
            r = types.SimpleNamespace(returncode="timeout")
    with open(os.path.join(outdir, name + ".log"), errors="replace") as f:
        text = f.read()
    if r.returncode or not os.path.exists(path):
        tail = "\n".join(text.strip().splitlines()[-25:])
        return {"error": f"exit {r.returncode}", "log_tail": tail}
    with open(path) as f:
        res = json.load(f)
    m = re.search(r"Model loading took ([\d.]+) GiB", text)
    res["weights_gib"] = float(m.group(1)) if m else None
    m = re.search(r"GPU KV cache size: ([\d,]+) tokens", text)
    res["kv_tokens"] = int(m.group(1).replace(",", "")) if m else None
    m = re.search(r"Available KV cache memory: ([\d.]+) GiB", text)
    res["kv_gib"] = float(m.group(1)) if m else None
    m = re.search(r"Maximum concurrency for ([\d,]+) tokens per request: ([\d.]+)x", text)
    res["concurrency"] = (int(m.group(1).replace(",", "")), float(m.group(2))) if m else None
    res["autotune"] = sorted({(m.group(1), m.group(2).strip().replace("\n", " "), m.group(3).strip().replace("\n", " ")) for m in re.finditer(r"Triton autotuning for function (\S+),\s*with key as (.*?),\s*finished after [^\n]*\s*best config selected: (.*?);", text, re.S)})
    res["kernel_lines"] = sorted({re.sub(r"^.*?\] ", "", ln.strip()) for ln in text.splitlines() if re.search(r"Selected \S+ for|DeepGEMM|deep_gemm|Marlin|marlin|FlashInfer.*(fp8|FP8)|Using .*(Fp8|FP8)", ln) and "WARNING" not in ln and "warmup" not in ln.lower() and "it/s" not in ln})[:20]  # (not a progress bar: DeepGEMM's warmup, built by the first server, is quick in the second)
    return res


def eq(a, b):
    if isinstance(a, float) and isinstance(b, float):
        return a == b or (math.isnan(a) and math.isnan(b))
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(eq(x, y) for x, y in zip(a, b))
    return a == b


def first_diff(a, b):
    """Where two runs' outputs first differ, and by how much (largest |difference| of the log probabilities)."""
    for p, (ta, tb) in enumerate(zip(a["tokens"], b["tokens"])):
        if ta != tb:
            i = next(i for i, (x, y) in enumerate(zip(ta, tb)) if x != y) if ta[: len(tb)] != tb[: len(ta)] else min(len(ta), len(tb))
            return f"prompt {p}: the tokens differ from the {i}th"
    worst, where = 0.0, None
    for key in ("logprobs", "prompt_logprobs"):
        for p, (la, lb) in enumerate(zip(a[key], b[key])):
            if key == "logprobs":
                for i, (x, y) in enumerate(zip(la, lb)):
                    if not eq(x, y) and abs(x - y) >= worst:
                        worst, where = abs(x - y), f"prompt {p}, token {i}, {key}"
            else:
                for i, (da, db) in enumerate(zip(la, lb)):
                    if da is None or db is None:
                        continue
                    if not eq(da, db):
                        d = max([abs(x[1] - y[1]) for x, y in zip(da, db) if x[0] == y[0] and not eq(x[1], y[1])] or [float("inf")])
                        if d >= worst:
                            worst, where = d, f"prompt {p}, position {i}, {key}"
    return f"{where}: largest |difference| {worst:.3g}" if where else "no difference found"


def same(a, b):
    return all(eq(a[k], b[k]) for k in OUTPUTS)


OUTPUTS = ("tokens", "logprobs", "prompt_logprobs")  # (what exact compares)


def digest(r):
    """The SHA-256 of each of a run's outputs as JSON (the same values, bit for bit, give the same digest)."""
    import hashlib

    return {k: hashlib.sha256(json.dumps(r[k]).encode()).hexdigest() for k in OUTPUTS if k in r}


def bench_differ(a, b):
    """The speed batches (by name) whose first batch's greedy tokens two runs both have and differ in."""
    ba, bb = a.get("bench") or {}, b.get("bench") or {}
    return [k for k in ba if k in bb and ba[k].get("tokens") is not None and bb[k].get("tokens") is not None and ba[k]["tokens"] != bb[k]["tokens"]]


def sweep_diff(a, b):
    """Per length of the sweep: the first position of the single prompt where two runs' log probabilities differ (None: none)."""
    out = {}
    for n in SWEEP:
        la, lb = a["sweep"][str(n)], b["sweep"][str(n)]
        out[n] = next((i for i, (x, y) in enumerate(zip(la, lb)) if not eq(x, y)), None)
    return out


def autotune_diff(a, b):
    """The Triton autotuners (function, key) whose chosen config differs between two runs, and how many of them both ran."""
    ca, cb = {(f, k): c for f, k, c in a.get("autotune", [])}, {(f, k): c for f, k, c in b.get("autotune", [])}
    both = ca.keys() & cb.keys()
    return sorted((f, k, ca[(f, k)], cb[(f, k)]) for f, k in both if ca[(f, k)] != cb[(f, k)]), len(both)


def diag(model, outdir, args):
    """Where `graph` differs: stock compiled (--stocks times, each with a compile cache of its own) against Glyd's (--ops times), each on the four prompts
    and on the sweep of single prompts, and which Triton autotuner configs each run took; --pin: the same with every autotuner's first config and one hash seed."""
    opt = lambda name, default=None: args[args.index(name) + 1] if name in args else default
    spec0 = {"model": model, "max_model_len": 4096, "language_model_only": "--language-model-only" in args, "bench": False, "sweep": True, "mode": "graph"}
    base = PIN if "--pin" in args else {}
    plan = [(f"stock-{i}", None) for i in range(1, int(opt("--stocks", 3)) + 1)] + [(f"glyd-{i}", "glyd") for i in range(1, int(opt("--ops", 3)) + 1)]
    runs = {}
    for name, quantization in plan:
        runs[name] = run({**spec0, "quantization": quantization}, outdir, name, {**base, "VLLM_CACHE_ROOT": os.path.join(outdir, f"cache-{name}")})
        print(f"  {name}: {'ERROR ' + runs[name]['error'] if 'error' in runs[name] else 'ok'}", flush=True)
    ref = runs[plan[0][0]]
    if "error" in ref:
        print("FAIL: the first stock run did not run:\n" + ref["log_tail"])
        sys.exit(1)
    pin = "pinned: " if base else "not pinned: "
    for name, r in runs.items():
        if r is ref:
            continue
        if "error" in r:
            print(f"INFO: diag[{name}]: did not run ({r['error']}): {r['log_tail'][-300:]}")
            continue
        sd, (ad, an) = sweep_diff(ref, r), autotune_diff(ref, r)
        print(f"INFO: diag[{pin}{name} against {plan[0][0]}]: the four prompts {'the same bits' if same(ref, r) else 'DIFFER: ' + first_diff(ref, r)}; the sweep (the first position that differs, by prompt length): "
              + ", ".join(f"{n}: {'-' if p is None else p}" for n, p in sd.items()) + f"; autotuners: {len(ad)} of {an} took another config" + (f" ({ad[0][0]} {ad[0][1]}: {ad[0][2]} / {ad[0][3]})" if ad else ""), flush=True)
    with open(os.path.join(outdir, "diag.json"), "w") as f:
        json.dump({"model": model, "pinned": bool(base), "runs": {k: {kk: vv for kk, vv in v.items() if kk not in ("prompt_logprobs", "sweep")} for k, v in runs.items()}, "sweep_diff": {k: sweep_diff(ref, v) for k, v in runs.items() if v is not ref and "error" not in v}}, f, indent=1)


def seed_sweep(model, outdir, args):
    """Whether compiled Glyd and compiled stock can differ: for each PYTHONHASHSEED of --seeds, stock and Glyd (graph mode: vLLM's default compile and CUDA graphs, a fresh compile
    cache each, the autotuners pinned unless --tune), and, on the first seed, Glyd again (does it equal itself) and Glyd with its scratch overwritten after every product
    (GLYD_F8_POISON: a read of the scratch outside the op, or an overlap with another stream, would move the logits). Each comparison is said as soon as both its runs are
    done (a run cut short by the clock leaves the ones before it). --x-bits: the outputs (a run's JSON) of a run that was not stock's, to say which of the runs equal it."""
    opt = lambda name, default=None: args[args.index(name) + 1] if name in args else default
    seeds = [int(x) for x in opt("--seeds").split(",")]
    x = None
    if opt("--x-bits"):
        with open(opt("--x-bits")) as f:
            x = json.load(f)
    spec0 = {"model": model, "max_model_len": 4096, "language_model_only": "--language-model-only" in args, "bench": False, "mode": "graph"}
    s0 = seeds[0]
    plan = [("stock", None, s0, {}), ("glyd", "glyd", s0, {}), ("glyd-poison", "glyd", s0, {"GLYD_F8_POISON": "1"}), ("glyd-again", "glyd", s0, {})]
    plan = [(f"{n}-s{s0}" if n in ("stock", "glyd") else f"glyd-s{s0}-{n[5:]}", q, sd, e) for n, q, sd, e in plan]
    plan += [(f"{v}-s{sd}", q, sd, {}) for sd in seeds[1:] for v, q in (("stock", None), ("glyd", "glyd"))]
    tune = {} if "--tune" in args else {"F8_PIN_AUTOTUNE": "1"}  # (Triton's autotuners keep their first config unless --tune: four unpinned comparisons found their choices change no bit, and each autotune is a minute of every run)
    checks = [(f"stock-s{sd}", f"glyd-s{sd}", f"seed {sd}: Glyd {{}} stock") for sd in seeds] + [
        (f"glyd-s{s0}", f"glyd-s{s0}-again", f"Glyd twice on seed {s0} (two fresh compiles): {{}}"), (f"glyd-s{s0}", f"glyd-s{s0}-poison", "Glyd with its scratch overwritten after every product: {}")]
    runs, said, fails = {}, set(), 0
    for name, quantization, sd, env in plan:
        runs[name] = run({**spec0, "quantization": quantization}, outdir, name, {**tune, "PYTHONHASHSEED": str(sd), "VLLM_CACHE_ROOT": os.path.join(outdir, f"cache-{name}"), **env})
        r = runs[name]
        print(f"  {name}: {'ERROR ' + r['error'] if 'error' in r else 'ok'}" + ("" if x is None or "error" in r else f"; equals the reference run's outputs (--x-bits): {'YES' if same(r, x) else 'no'}"), flush=True)
        if "error" in r:
            fails += 1
            print(f"FAIL: {name} did not run: {r['error']}\n{r['log_tail'][-600:]}", flush=True)
        for i, (u, v, text) in enumerate(checks):
            if i not in said and u in runs and v in runs and "error" not in runs[u] and "error" not in runs[v]:
                said.add(i)
                ok = same(runs[u], runs[v])
                fails += not ok
                print(f"{'PASS' if ok else 'FAIL'}: " + text.format("equals" if text.startswith("seed") and ok else "differs from" if text.startswith("seed") else "the same bits" if ok else first_diff(runs[u], runs[v])) + ("" if ok or not text.startswith("seed") else f": {first_diff(runs[u], runs[v])}"), flush=True)
    ok = {k: v for k, v in runs.items() if "error" not in v}
    outcomes, label = [], {}  # the distinct outputs, in order of first sight: a letter each
    for k, v in ok.items():
        for i, o in enumerate(outcomes):
            if same(v, o):
                label[k] = chr(65 + i)
                break
        else:
            outcomes.append(v)
            label[k] = chr(64 + len(outcomes))
    print("INFO: outcomes by run (one letter a distinct output): " + ", ".join(f"{k}={label[k]}" for k in ok))
    if x is not None:
        print("INFO: equal to the reference run's outputs (--x-bits): " + (", ".join(k for k, v in ok.items() if same(v, x)) or "none"))
    print(f"INFO: stock gave {len({label[k] for k in ok if k.startswith('stock-s')})} distinct output(s) over {len([k for k in ok if k.startswith('stock-s')])} runs; Glyd {len({label[k] for k in ok if k.startswith('glyd-s')})} over {len([k for k in ok if k.startswith('glyd-s')])}")
    with open(os.path.join(outdir, "seeds.json"), "w") as f:
        json.dump({"model": model, "seeds": seeds, "labels": label, "equal_to_x": None if x is None else [k for k, v in ok.items() if same(v, x)], "runs": {k: {kk: vv for kk, vv in v.items() if kk not in ("prompt_logprobs", "sweep")} for k, v in runs.items()}}, f, indent=1)
    sys.exit(1 if fails else 0)


def main():
    args = sys.argv[1:]
    if args and args[0] == "--child":
        res = child(json.loads(args[1]))
        with open(args[2], "w") as f:
            json.dump(res, f)
        return
    opt = lambda name, default=None: args[args.index(name) + 1] if name in args else default
    model = next((a for i, a in enumerate(args) if not a.startswith("--") and (i == 0 or args[i - 1] not in ("--out", "--modes", "--max-model-len", "--only", "--seeds", "--x-bits", "--stocks", "--ops", "--util", "--fused", "--max-num-seqs"))), None)
    if model is None:
        raise SystemExit(__doc__)
    outdir = opt("--out", "./check_fp8_vllm_results")
    os.makedirs(outdir, exist_ok=True)
    modes = opt("--modes", "eager,graph,graphonly").split(",")
    only = opt("--only")
    if "--diag" in args:
        return diag(model, outdir, args)
    if "--seeds" in args:
        return seed_sweep(model, outdir, args)
    pin = PIN if "--pin" in args else {}
    spec0 = {"model": model, "max_model_len": int(opt("--max-model-len", 8192)), "language_model_only": "--language-model-only" in args, "bench": "--no-bench" not in args, "util": float(opt("--util", 0.9)), "max_num_seqs": int(opt("--max-num-seqs", 64)), "chat": "--chat" in args}  # (graphonly: no bench, its start is the check)
    may_fail = "--stock-may-fail" in args
    runs, fails = {}, 0

    def line(kind, text):
        nonlocal fails
        fails += kind == "FAIL"
        print(f"{kind}: {text}", flush=True)

    def why(r):  # (the reason a run did not start, from its log's end)
        t = r.get("log_tail", "")
        m = re.findall(r"(?:ValueError|RuntimeError|OutOfMemoryError|torch\.OutOfMemoryError)[^\n]*", t)
        return (m[-1] if m else t.strip().splitlines()[-1] if t.strip() else r.get("error", ""))[:300]

    def one(name, spec, env):
        t0 = time.time()
        runs[name] = r = run(spec, outdir, name, env)
        print(f"  {name}: {'ERROR ' + r['error'] if 'error' in r else 'ok'} in {time.time() - t0:.0f} s" + (f"; weights {r.get('weights_gib')} GiB, KV {r.get('kv_tokens')} tokens, concurrency {r.get('concurrency')}" if "error" not in r else ""), flush=True)
        return r

    for mode in modes:
        for variant in ("stock", "glyd") if only is None else (only,):
            name = f"{variant}-{mode}"
            spec = {**spec0, "mode": mode, "bench": spec0["bench"] and mode != "graphonly", "quantization": "glyd" if variant == "glyd" else None}
            if variant == "glyd":
                spec["additional_config"] = {"glyd": {"exact": True}}  # (exact: every product the decode and vLLM's own kernel)
            elif "--stock-brief" in args:
                spec["brief"] = True
            r = one(name, spec, pin)
            if "error" in r:
                if variant == "stock" and may_fail:
                    line("INFO", f"{name} did not start: {why(r)}")
                else:
                    line("FAIL", f"{name} did not run ({r['error']}); the log's end:\n" + r["log_tail"])
        s, g = runs.get(f"stock-{mode}"), runs.get(f"glyd-{mode}")
        if mode in ("graph", "graphq", "graphpw", "compile") and only is None and "--no-again" not in args and s and g and "error" not in s and "error" not in g and not same(s, g):
            name = f"stock-{mode}-again"  # (stock again, an empty compile cache: whether stock's own compiled logits are reproducible, which a difference needs to be told)
            runs[name] = run({**spec0, "mode": mode, "quantization": None, "bench": False}, outdir, name, {**pin, "VLLM_CACHE_ROOT": os.path.join(outdir, "cache-again")})
        if only is None and s and g and "error" not in s and "error" not in g and bench_differ(s, g) and s.get("blocks"):
            name = f"glyd-{mode}-sameblocks"  # (Glyd again with stock's KV cache blocks: vLLM's scheduler admits requests by the blocks free, so a larger cache makes other batches)
            one(name, {**spec0, "mode": mode, "quantization": "glyd", "additional_config": {"glyd": {"exact": True}}, "brief": True, "blocks": s["blocks"]}, pin)
    if "--poison" in args and only is None:
        mode = modes[-1]
        name = f"glyd-{mode}-poison"
        r = one(name, {**spec0, "mode": mode, "quantization": "glyd", "bench": False, "additional_config": {"glyd": {"exact": True}}}, {**pin, "GLYD_F8_POISON": "1", "GLYD_F8_VERIFY": "1"})
        if "error" in r:
            line("FAIL", f"{name} did not run ({r['error']}); the log's end:\n" + r["log_tail"])
    if opt("--fused"):
        mode = modes[-1]
        name = f"glyd-{mode}-fused{opt('--fused')}"
        r = one(name, {**spec0, "mode": mode, "quantization": "glyd", "no_exact": True}, {**pin, "GLYD_F8_FUSED_MAX": opt("--fused")})
        if "error" in r:
            line("FAIL", f"{name} did not run ({r['error']}); the log's end:\n" + r["log_tail"])
    gpu = next((r["gpu"] for r in runs.values() if "gpu" in r), "?")
    print(f"GPU {gpu}")
    for mode in modes:
        s, g = runs.get(f"stock-{mode}"), runs.get(f"glyd-{mode}")
        if not g or "error" in g:
            continue
        p = g.get("probe", {})
        if p.get("packed", 0) > 0:
            line("PASS", f"packed[{mode}]: {p['packed']} layers packed ({p.get('experts_layers', 0)} of them experts' layers): {p['packed_bytes'] / 2**30:.2f} GiB held for {p['weight_bytes'] / 2**30:.2f} GiB of weights as vLLM's kernels are fed them "
                 f"({100 * (1 - p['packed_bytes'] / p['weight_bytes']):.1f}% less, {8 * p['packed_bytes'] / p['weight_bytes']:.3f} bits a value), scratches {p['scratch_bytes'] / 2**30:.2f} GiB; methods {p['methods']}; f8_linear for few tokens: {p.get('fused', 0)} Linears")
        else:
            line("FAIL", f"packed[{mode}]: nothing was packed ({p})")
        line("INFO", f"glyd[{mode}]: weights {g.get('weights_gib')} GiB, KV cache {g.get('kv_tokens')} tokens ({g.get('kv_gib')} GiB), concurrency {g.get('concurrency')}")
        if not s or "error" in s:
            line("INFO", f"stock[{mode}] did not start{': ' + why(s) if s else ''}: no comparison; Glyd started, so this checkpoint fits here only packed" if s else f"stock[{mode}] not run")
            continue
        again = runs.get(f"stock-{mode}-again")
        noise = again is not None and "error" not in again and not same(s, again)
        n = sum(len(x) for x in s["tokens"]) + sum(len(x) for x in s["logprobs"]) + sum(len(d) for p_ in s["prompt_logprobs"] for d in p_ if d)
        if same(s, g):
            line("PASS", f"exact[{mode}]: tokens, token log probabilities and the top {TOPK} prompt log probabilities of 4 prompts ({n} values) bit for bit stock vLLM's")
        elif noise:
            line("INCONCLUSIVE", f"exact[{mode}]: Glyd differs from stock ({first_diff(s, g)}), and stock differs from stock again ({first_diff(s, again)}): stock's compiled logits are not reproducible here")
        else:
            line("FAIL", f"exact[{mode}]: Glyd differs from stock: {first_diff(s, g)}" + ("" if again is None else "; stock again equals stock"))
        ks, kg = s.get("kernel_lines", []), g.get("kernel_lines", [])
        if ks == kg:
            line("PASS", f"kernel[{mode}]: the FP8 kernel lines vLLM logs are the same in both ({len(ks)}: {'; '.join(ks)[:300]})")
        else:
            line("FAIL", f"kernel[{mode}]: stock logs {ks}, Glyd {kg}")
        if s.get("weights_gib") is not None and g.get("weights_gib") is not None and p.get("scratch_bytes") is not None:
            want = s["weights_gib"] - (p["weight_bytes"] - p["packed_bytes"]) / 2**30 + p["scratch_bytes"] / 2**30
            ok = abs(g["weights_gib"] - want) <= 0.25  # (vLLM logs two decimals; the caching allocator's blocks)
            line("PASS" if ok else "FAIL", f"weights[{mode}]: vLLM logs {s['weights_gib']} GiB stock, {g['weights_gib']} GiB Glyd ({g['weights_gib'] - s['weights_gib']:+.2f} GiB, {100 * (1 - g['weights_gib'] / s['weights_gib']):.1f}% less); "
                 f"expected {want:.2f} (stock less the packs' saving {(p['weight_bytes'] - p['packed_bytes']) / 2**30:.2f}, plus the scratches {p['scratch_bytes'] / 2**30:.2f})")
            line("INFO", f"KV cache[{mode}]: stock {s.get('kv_tokens')} tokens ({s.get('kv_gib')} GiB, concurrency {s.get('concurrency')}), Glyd {g.get('kv_tokens')} tokens ({g.get('kv_gib')} GiB, concurrency {g.get('concurrency')})")
        if s.get("peak_generate") and g.get("peak_generate"):
            line("INFO", f"memory[{mode}]: torch's peak during the exact check's generate (prompt log probabilities included): stock {s['peak_generate'] / 2**30:.2f} GiB, {(s['peak_generate'] - s['allocated_before_generate']) / 2**30:.2f} over its start; "
                 f"Glyd {g['peak_generate'] / 2**30:.2f} GiB, {(g['peak_generate'] - g['allocated_before_generate']) / 2**30:.2f} over its start")
        for k in (g.get("bench") or {}):
            ts, tg = (s.get("bench") or {}).get(k, {}).get("tokens"), g["bench"][k].get("tokens")
            if ts is None or tg is None:
                continue
            n_, L_, T_ = BENCH[k]
            sb = runs.get(f"glyd-{mode}-sameblocks") or {}
            tb = ((sb.get("bench") or {}).get(k) or {}).get("tokens")
            kind = "PASS" if ts == tg or tb == ts else "FAIL"
            where = [f"request {i} from token {next(j for j, (x, y) in enumerate(zip(a_, b_)) if x != y)}" for i, (a_, b_) in enumerate(zip(ts, tg)) if a_ != b_]
            line(kind, f"exact-bench[{mode}][{k}]: the greedy tokens of the first batch ({n_} requests of {L_} tokens, {T_} out each) {'the same' if ts == tg else 'differ: ' + ', '.join(where)} in both"
                 + ("" if ts == tg else f"; Glyd with stock's {s.get('blocks')} KV cache blocks (vLLM admits requests by the blocks free): " + ("did not run" if tb is None else "the same" if tb == ts else "they differ")))
        if "bench" in s and "bench" in g:
            line("INFO", f"speed[{mode}] (output tokens a second, Glyd / stock): " + ", ".join(f"{k} {g['bench'][k]['tokens_per_s']:.1f} / {s['bench'][k]['tokens_per_s']:.1f} = {g['bench'][k]['tokens_per_s'] / s['bench'][k]['tokens_per_s']:.2f}" if k in s["bench"] else f"{k} {g['bench'][k]['tokens_per_s']:.1f} / -" for k in g["bench"]))
    for mode in modes[-1:]:
        s, z = runs.get(f"stock-{mode}"), runs.get(f"glyd-{mode}-poison")
        if z and "error" not in z:
            v = z.get("probe", {}).get("verified") or [0, 0]
            line("PASS" if v[0] > 0 else "FAIL", f"verify[{mode}]: {v[0]} packs decoded back at load bit for bit the bytes they were made from ({v[1] / 2**30:.2f} GiB; every Linear and every expert packed)")
            if s and "error" not in s:
                line("PASS" if same(s, z) else "FAIL", f"poison[{mode}]: Glyd with the Linears' scratch overwritten after every product and the experts buffer e4m3 NaN before every call: "
                     + ("tokens and log probabilities bit for bit stock's" if same(s, z) else first_diff(s, z)))
    for name, r in runs.items():
        if "fused" in name and "error" not in r and "bench" in r:
            line("INFO", f"speed[{name}] (output tokens a second): " + ", ".join(f"{k} {v['tokens_per_s']:.1f}" for k, v in r["bench"].items()) + f"; f8_linear for {r.get('probe', {}).get('fused', 0)} Linears")
    with open(os.path.join(outdir, "check_fp8_vllm.json"), "w") as f:
        json.dump({"model": model, "gpu": gpu, "modes": modes, "fails": fails, "runs": {k: {**{kk: vv for kk, vv in v.items() if kk not in ("prompt_logprobs",)}, "sha256": digest(v)} for k, v in runs.items()}}, f, indent=1)
    print(f"{'FAIL' if fails else 'PASS'}: {fails} failed; {outdir}/check_fp8_vllm.json")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
