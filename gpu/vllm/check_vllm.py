"""Glyd's vLLM plugin (glyd.gpu.vllm_plugin, --quantization glyd) against
vLLM's own bf16, on the models given (default Qwen/Qwen3-1.7B), each vLLM in
a process of its own:

- bf16 with CUDA graphs (vLLM's default), eager, and its prompts one at a
  time: the noise floor, how far vLLM's own bf16 moves with its schedule;
- glyd fused, tiered and 12-bit, CUDA graphs: 8 prompts' greedy tokens and
  their logprobs against bf16's; bf16's own continuation (1,536 tokens) fed
  back (prompt_logprobs): top-1 agreement with bf16 at least 97%, and the
  mean |logprob difference| of its tokens within twice bf16's noise floor;
- glyd exact, eager: tokens, logprobs and prompt_logprobs bf16 eager's bit
  for bit; exact under torch.compile refused, with why, but with inductor's
  deterministic mode (DETERMINISTIC), where compiled bf16's bit for bit
  (the same mode, CUDA graphs on); fused Glyd compiled in that mode the same
  bits from one run to the next (the second loading the first's graphs);
- per layer (LLM.apply_model): every packed layer's product against
  F.linear on its decoded matrix at 1-4096 tokens, within 1e-2 and the same
  bits every run; with GLYD_VERIFY=1 every pack checked against its
  weights, bit for bit, as it was made;
- a glyd save (glyd.save_pretrained from glyd.from_pretrained, both
  layouts; --saves): loaded as saved, its tokens and logprobs those of the
  bf16 checkpoint packed at load in that layout, bit for bit; in the other
  layout (decoded and packed again) likewise (eager, which gives the same
  bits in every process: compiled, Glyd did not, even on one compile
  cache);
- the compile cache: the layouts and modes in turn on one VLLM_CACHE_ROOT,
  each in a cache of its own (vLLM's additional_config: three caches for
  bf16 and the two layouts); each layout again on it, its own graphs loaded
  and its logits against bf16's as before; bf16 compiled again on an empty
  cache, against its first compile (the compiled graph's own variation).

    python check_vllm.py [--saves] [--quick] [--out DIR] [MODEL ...]

--quick leaves out the two runs that only describe bf16's own noise (its
prompts one at a time, and compiled again on an empty cache): every check
stays.

Needs vLLM (glyd[vllm]) and the library (GLYD_GPU_LIB); --saves makes its
saves with glyd[gpu]'s transformers path (GLYD_SAVE_PYTHON: a Python that
has it, where vLLM's venv lacks accelerate). Its results, a JSON file and a
log a run, in --out (default ./check_vllm_results)."""
import json
import os
import shutil
import subprocess
import sys
import tempfile

TOKENS = 64
LONG = 1536  # bf16's continuation fed back (prompt_logprobs)
PROMPTS = [
    "The history of data compression began",
    'def fibonacci(n):\n    """Return the n-th Fibonacci number."""\n',
    "Q: A train leaves at 3:40 pm and the trip takes 2 hours 35 minutes. When does it arrive?\nA:",
    "The capital of Australia is",
    "Translate to French: The weather is lovely today, so we will walk to the market.",
    "In quantum mechanics, the uncertainty principle states that",
    "SELECT name, COUNT(*) FROM orders JOIN customers ON",
    "Once upon a time, in a village at the edge of a great forest,",
]
MS = (1, 7, 16, 17, 33, 64, 65, 128, 129, 512, 513, 1024, 4096)
AGREE, FLOOR = 0.97, 2.0  # fused: top-1 agreement at least; mean |logprob difference| at most this times bf16's floor
DETERMINISTIC = {"inductor_compile_config": {"deterministic": True, "combo_kernels": True, "benchmark_combo_kernel": False}}  # vLLM's compilation_config


def child(spec):
    """One vLLM instance (spec: model, quantization, eager, compilation_config, one_by_one, long ids, layers): its
    tokens, logprobs and prompt_logprobs, the KV cache's size, and (glyd) its layers' products against F.linear."""
    import torch
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    out = {"spec": spec}
    try:
        llm = LLM(model=spec["model"], quantization=spec.get("quantization"), dtype="bfloat16", gpu_memory_utilization=0.85, max_model_len=4096, enforce_eager=spec.get("eager", False), seed=0, **({"compilation_config": spec["compilation_config"]} if spec.get("compilation_config") else {}))
    except Exception as e:  # (exact under compile: refused)
        out["error"] = f"{type(e).__name__}: {e}"
        return out
    cfg = llm.llm_engine.vllm_config
    out["blocks"], out["block_size"] = cfg.cache_config.num_gpu_blocks, cfg.cache_config.block_size
    out["glyd"] = cfg.additional_config.get("glyd") if isinstance(cfg.additional_config, dict) else None
    greedy = SamplingParams(temperature=0, max_tokens=TOKENS, logprobs=0)
    batches = [[p] for p in PROMPTS] if spec.get("one_by_one") else [PROMPTS]
    outs = [o for b in batches for o in llm.generate(b, greedy, use_tqdm=False)]
    out["tokens"] = [list(o.outputs[0].token_ids) for o in outs]
    out["logprobs"] = [[d[t].logprob for d, t in zip(o.outputs[0].logprobs, o.outputs[0].token_ids)] for o in outs]
    if spec.get("make_long"):  # bf16: its continuation of the first prompt, fed back by every run
        o = llm.generate([PROMPTS[0]], SamplingParams(temperature=0, max_tokens=LONG, ignore_eos=True), use_tqdm=False)[0]
        out["long_ids"] = spec["long_ids"] = list(o.prompt_token_ids) + list(o.outputs[0].token_ids)
    if spec.get("long_ids"):
        ids = spec["long_ids"]
        o = llm.generate([TokensPrompt(prompt_token_ids=ids)], SamplingParams(temperature=0, max_tokens=1, prompt_logprobs=1), use_tqdm=False)[0]
        pl = o.prompt_logprobs[1:]
        out["long_logprobs"] = [d[t].logprob for d, t in zip(pl, ids[1:])]
        out["long_top1"] = [d[t].rank == 1 for d, t in zip(pl, ids[1:])]
    if spec.get("layers"):
        out["layers"] = llm.apply_model(_layers)[0]
    return out


def _layers(model):
    """Every packed layer's product (glyd::vllm_linear, fused) against F.linear on its decoded matrix, at MS tokens:
    the largest relative error, and whether a second call gave the same bits; the layers packed, and verified."""
    import torch
    import torch.nn.functional as F
    from glyd.gpu import kernels as g

    worst, same, n, verified = 0.0, True, 0, 0
    torch.manual_seed(0)
    for m in model.modules():
        if getattr(m, "glyd_words", None) is None:
            continue
        n += 1
        verified += bool(m.glyd_verified)
        O, K = m.glyd_out, m.weight.shape[1]
        p = g.Mma12((O, K), m.glyd_data, m.glyd_a, m.glyd_b, m.glyd_words[0] & 0xFF) if len(m.glyd_words) == 4 else g.Mma((O, K), m.glyd_data, m.glyd_a, m.glyd_b, m.glyd_words)
        w = g.mma_unpack(p).float()
        for M in MS if n <= 4 else (1, 64, 513):  # (every M on the first layers, three on the rest)
            x = torch.randn(M, K, dtype=torch.bfloat16, device=w.device)
            y = torch.ops.glyd.vllm_linear(x, m.glyd_data, m.glyd_a, m.glyd_b, m.glyd_words, m.bias, O, False)
            ref = F.linear(x.float(), w, None if m.bias is None else m.bias.float())
            worst = max(worst, ((y.float() - ref).abs().max() / ref.abs().max()).item())
            same &= torch.equal(y, torch.ops.glyd.vllm_linear(x, m.glyd_data, m.glyd_a, m.glyd_b, m.glyd_words, m.bias, O, False))
    return {"packed": n, "verified": verified, "worst_rel_error": worst, "same_bits": same, **_host(model)}


def _host(model):
    """The host's time a product, microseconds (the calls queued, not waited for): the op (fused, a prompt's 1024 tokens
    and a step's 8) against F.linear on the same shapes, the first packed layer's."""
    import time
    import torch
    import torch.nn.functional as F

    m = next(m for m in model.modules() if getattr(m, "glyd_words", None) is not None)
    O, K = m.glyd_out, m.weight.shape[1]
    w = torch.randn(O, K, dtype=torch.bfloat16, device=m.glyd_data.device)
    out = {}
    for M in (8, 1024):
        x = torch.randn(M, K, dtype=torch.bfloat16, device=w.device)
        for name, f in (("op", lambda: torch.ops.glyd.vllm_linear(x, m.glyd_data, m.glyd_a, m.glyd_b, m.glyd_words, m.bias, O, False)), ("linear", lambda: F.linear(x, w, m.bias))):
            for _ in range(5):
                f()
            torch.cuda.synchronize()
            t = time.perf_counter()
            for _ in range(50):
                f()
            out[f"host_us_{name}_{M}"] = round((time.perf_counter() - t) / 50 * 1e6, 1)
            torch.cuda.synchronize()
    return out


def run(spec, out_dir, name, env=None):
    """spec in a vLLM of its own (this script, --child), its JSON in out_dir/name.json."""
    path = os.path.join(out_dir, name + ".json")
    e = dict(os.environ, VLLM_ENABLE_V1_MULTIPROCESSING="0", **(env or {}))
    with open(os.path.join(out_dir, name + ".log"), "w") as log:
        r = subprocess.run([sys.executable, os.path.abspath(__file__), "--child", json.dumps(spec), path], env=e, stdout=log, stderr=subprocess.STDOUT, timeout=1800)
    if r.returncode or not os.path.exists(path):
        raise RuntimeError(f"{name}: exit {r.returncode}; see {name}.log")
    with open(path) as f:
        return json.load(f)


def same_prefix(a, b):
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]:
        n += 1
    return n


def compare(a, b):
    """Tokens the same from the start (each prompt), prompts bit for bit (tokens and logprobs), the mean |logprob
    difference| over the shared prefixes; on the fed-back continuation: bit for bit, top-1 agreement, mean |diff|."""
    pre = [same_prefix(x, y) for x, y in zip(a["tokens"], b["tokens"])]
    bits = sum(x == y and lx == ly for x, y, lx, ly in zip(a["tokens"], b["tokens"], a["logprobs"], b["logprobs"]))
    d = [abs(u - v) for x, y, lx, ly in zip(a["tokens"], b["tokens"], a["logprobs"], b["logprobs"]) for u, v in list(zip(lx, ly))[: same_prefix(x, y)]]
    r = {"prefix": pre, "bit_identical": bits, "mean_abs_dlogprob": sum(d) / max(1, len(d))}
    if "long_logprobs" in a and "long_logprobs" in b:
        la, lb = a["long_logprobs"], b["long_logprobs"]
        r["long_bit_identical"] = la == lb
        r["long_mean_abs_dlogprob"] = sum(abs(u - v) for u, v in zip(la, lb)) / len(la)
        r["long_top1"] = sum(a["long_top1"]) / len(a["long_top1"])
    return r


def main():
    args = sys.argv[1:]
    saves, quick = "--saves" in args, "--quick" in args
    out_dir = args[args.index("--out") + 1] if "--out" in args else "check_vllm_results"
    models = [a for i, a in enumerate(args) if not a.startswith("--") and (i == 0 or args[i - 1] != "--out")] or ["Qwen/Qwen3-1.7B"]
    os.makedirs(out_dir, exist_ok=True)
    report, failed = [], []

    def check(ok, what):
        report.append(("PASS " if ok else "FAIL ") + what)
        print(report[-1], flush=True)
        if not ok:
            failed.append(what)

    for model in models:
        tag = model.split("/")[-1]
        cache = tempfile.mkdtemp(prefix="vllm-cache-")  # one compile cache for every run of the model (its layouts and modes in turn)
        env = {"VLLM_CACHE_ROOT": cache}
        R = lambda name, spec, extra=None: run(dict(spec, model=spec.get("model", model)), out_dir, f"{tag}-{name}", dict(env, **(extra or {})))
        bf16 = R("bf16", {"make_long": True})
        long = {"long_ids": bf16["long_ids"]}
        eager = R("bf16-eager", dict(long, eager=True))
        floor = compare(eager, bf16)
        noise = ""
        if not quick:
            one = R("bf16-one-by-one", {"one_by_one": True})
            again = run(dict(long, model=model), out_dir, f"{tag}-bf16-recompiled", {"VLLM_CACHE_ROOT": tempfile.mkdtemp(prefix="vllm-cache-")})
            noise = f"; one by one against batched {compare(one, bf16)}; compiled again on an empty cache {compare(again, bf16)}"
        print(f"{tag}: bf16 KV cache {bf16['blocks'] * bf16['block_size']} tokens; its noise: graphs against eager {floor}{noise}", flush=True)
        for layout in ("mma", "mma12"):
            r = R(f"glyd-{layout}", dict(long, quantization="glyd", layers=True), {"GLYD_LAYOUT": layout, "GLYD_VERIFY": "1"})
            c = compare(r, bf16)
            print(f"{tag} glyd {layout}: KV cache {r['blocks'] * r['block_size']} tokens (bf16 {bf16['blocks'] * bf16['block_size']}); against bf16 {c}; layers {r['layers']}; additional_config {r['glyd']}", flush=True)
            check(r["layers"]["verified"] == r["layers"]["packed"] > 0, f"{tag} {layout}: every pack ({r['layers']['packed']}) decoded to its weights bit for bit")
            check(r["layers"]["worst_rel_error"] < 1e-2 and r["layers"]["same_bits"], f"{tag} {layout}: each layer's product within 1e-2 of F.linear ({r['layers']['worst_rel_error']:.2e}) and the same bits every run")
            check(c["long_top1"] >= AGREE, f"{tag} {layout}: top-1 agreement with bf16 on its continuation {c['long_top1']:.4f} (at least {AGREE}; bf16 eager's {floor['long_top1']:.4f})")
            check(c["long_mean_abs_dlogprob"] <= FLOOR * max(floor["long_mean_abs_dlogprob"], 1e-6), f"{tag} {layout}: mean |logprob difference| {c['long_mean_abs_dlogprob']:.2e}, at most {FLOOR}x bf16 eager's against its graphs ({floor['long_mean_abs_dlogprob']:.2e})")
        for layout in ("mma", "mma12"):  # again on the shared cache, the same options: its own graphs loaded, not another's
            again = R(f"glyd-{layout}-again", dict(long, quantization="glyd"), {"GLYD_LAYOUT": layout, "GLYD_VERIFY": "1"})
            loaded = "Directly load AOT compilation" in open(os.path.join(out_dir, f"{tag}-glyd-{layout}-again.log")).read()
            c = compare(again, bf16)
            check(loaded and c["long_top1"] >= AGREE and c["long_mean_abs_dlogprob"] <= FLOOR * max(floor["long_mean_abs_dlogprob"], 1e-6), f"{tag} {layout}: again on the shared compile cache, its graphs loaded ({loaded}), against bf16 as before (top-1 {c['long_top1']:.4f}, mean |diff| {c['long_mean_abs_dlogprob']:.2e})")
        aot = os.path.join(cache, "torch_compile_cache", "torch_aot_compile")
        dirs = sorted(os.listdir(aot)) if os.path.isdir(aot) else []
        check(len(dirs) == 3, f"{tag}: a compile cache for each of bf16, glyd tiered and glyd 12-bit on the one VLLM_CACHE_ROOT ({len(dirs)}), none reused by another")
        x = R("glyd-exact-eager", dict(long, quantization="glyd", eager=True), {"GLYD_EXACT": "1"})
        c = compare(x, eager)
        check(c["bit_identical"] == len(PROMPTS) and c["long_bit_identical"], f"{tag} exact, eager: bf16 eager's tokens, logprobs and prompt_logprobs bit for bit ({c['bit_identical']} of {len(PROMPTS)}; continuation {c['long_bit_identical']})")
        xc = R("glyd-exact-compiled", dict(long, quantization="glyd"), {"GLYD_EXACT": "1"})
        check("error" in xc and "enforce-eager" in xc["error"], f"{tag} exact under torch.compile: refused ({xc.get('error', 'not refused')[:120]})")
        fresh = lambda: {"VLLM_CACHE_ROOT": tempfile.mkdtemp(prefix="vllm-cache-")}  # (each run a compile of its own)
        det = dict(long, model=model, compilation_config=DETERMINISTIC)
        bd = run(det, out_dir, f"{tag}-bf16-det", fresh())
        xd = run(dict(det, quantization="glyd"), out_dir, f"{tag}-glyd-exact-det", dict(fresh(), GLYD_EXACT="1"))
        c = compare(xd, bd)
        check(c["bit_identical"] == len(PROMPTS) and c["long_bit_identical"], f"{tag} exact under torch.compile, inductor deterministic: compiled bf16's tokens, logprobs and prompt_logprobs (the same mode) bit for bit ({c['bit_identical']} of {len(PROMPTS)}; continuation {c['long_bit_identical']})")
        gc = fresh()
        ga, gb = (run(dict(det, quantization="glyd"), out_dir, f"{tag}-glyd-det-{n}", gc) for n in "ab")
        c = compare(ga, gb)
        check(c["bit_identical"] == len(PROMPTS) and c["long_bit_identical"], f"{tag} fused, compiled, inductor deterministic: the same bits from one run to the next, the second on the first's graphs ({c['bit_identical']} of {len(PROMPTS)}; continuation {c['long_bit_identical']}; layout {ga['glyd']['layout']})")
        if saves:
            for layout in ("mma", "mma12"):
                d = tempfile.mkdtemp(prefix=f"glyd-save-{tag}-{layout}-")
                # (GLYD_SAVE_PYTHON: a Python with glyd[gpu]'s transformers path, where vLLM's venv lacks accelerate)
                subprocess.run([os.environ.get("GLYD_SAVE_PYTHON", sys.executable), "-c", "import sys, glyd; m = glyd.from_pretrained(sys.argv[1], layout=sys.argv[3], compile=False); glyd.save_pretrained(m, sys.argv[2], layout=sys.argv[3])", model, d, layout], check=True, timeout=1800)
                # eager: bit for bit from one process to the next (compiled, Glyd is not, even on one compile cache)
                ref = run(dict(long, model=model, quantization="glyd", eager=True), out_dir, f"{tag}-glyd-{layout}-ref", {"VLLM_CACHE_ROOT": cache, "GLYD_LAYOUT": layout})
                for as_layout in (layout, "mma12" if layout == "mma" else "mma"):
                    s = R(f"save-{layout}-as-{as_layout}", dict(long, model=d, quantization="glyd", eager=True), {"GLYD_LAYOUT": as_layout, "GLYD_VERIFY": "1"})
                    want = ref if as_layout == layout else run(dict(long, model=model, quantization="glyd", eager=True), out_dir, f"{tag}-glyd-{as_layout}-ref", {"VLLM_CACHE_ROOT": cache, "GLYD_LAYOUT": as_layout})
                    check(s["tokens"] == want["tokens"] and s["logprobs"] == want["logprobs"] and s["long_logprobs"] == want["long_logprobs"], f"{tag} save ({layout}) loaded as {as_layout}: the bf16 checkpoint packed at load in {as_layout}, bit for bit")
                shutil.rmtree(d, ignore_errors=True)
        shutil.rmtree(cache, ignore_errors=True)
    with open(os.path.join(out_dir, "report.txt"), "w") as f:
        f.write("\n".join(report) + "\n")
    print(f"check_vllm: {'all passed' if not failed else f'{len(failed)} failed'} ({len(report)} checks)")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    if sys.argv[1:2] == ["--child"]:
        result = child(json.loads(sys.argv[2]))
        with open(sys.argv[3], "w") as f:
            json.dump(result, f)
    else:
        main()
