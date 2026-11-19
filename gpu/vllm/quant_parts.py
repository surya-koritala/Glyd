"""A quantized checkpoint served by stock vLLM and by vLLM with --quantization glyd (vllm_quant.py: the checkpoint's own quantized layers, its
16-bit parts packed), on one GPU, each run a vLLM of its own (offline, in-process):

- stock eager and Glyd exact eager (exact, verify: every pack decoded as it is made and compared with its weights bit for bit): 8 prompts'
  greedy tokens with their top-5 logprobs, a text's prompt_logprobs (top 5 at every position) and, for a model that takes images, one
  image's chat: equal bit for bit, or not (the first difference named, and stock eager run again against itself); Triton's autotuned
  choices kept on disk (TRITON_CACHE_AUTOTUNING=1), so every run takes the first run's;
- stock and Glyd (Glyd's default mode, fused, as a user runs it: compiled, CUDA graphs, no verify), each started twice in the order stock, glyd,
  glyd-b, stock-b: the weights vLLM logs ("Model loading took"), the KV cache (tokens, GiB, and the concurrency vLLM gives at --max-model-len),
  output tokens a second of 1 and 32 requests (PROMPT random tokens in, NEW out, ignore_eos; the median of the REPS generate calls of both
  starts), and the 8 prompts' greedy tokens against stock's (fused products are not stock's bits: agreement);
- Glyd's inventory (LLM.apply_model): the options in effect (the model's own GlydConfig), every layer packed (its bf16 bytes and its pack's),
  every 16-bit tensor of 1 M values or more left as it is (by module class and method), the weights' bytes on the GPU; four packed Linears'
  products against F.linear on their decoded matrix.

    python quant_parts.py MODEL --out DIR [--util 0.9] [--mml 8192] [--dtype auto] [--max-num-seqs N] [--layout auto] [--steps stock-eager,glyd-exact,stock,glyd,glyd-b,stock-b] [--no-image]

Results: DIR/<step>.json and .log a run, DIR/summary.json (the comparison) and one line a check on stdout. Needs vLLM 0.30 and glyd_gpu
(PYTHONPATH) with its library (GLYD_GPU_LIB). Runs on a GPU pod, never on a laptop."""
import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time

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
LONG = "\n\n".join(PROMPTS * 6)  # (about 600 tokens: prompt_logprobs at every position)
TOKENS, TOP = 64, 5
STEPS = ("stock-eager", "glyd-exact", "stock", "glyd", "glyd-b", "stock-b")  # (-b: the same run started again, the compiled ones in the order A B B A)


def child(spec):
    import torch
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    out = {"spec": spec}
    kw = dict(model=spec["model"], dtype=spec["dtype"], seed=0, max_model_len=spec["mml"], gpu_memory_utilization=spec["util"], enforce_eager=spec["eager"])
    if spec.get("mns"):  # (a hybrid model's compiled run needs a Mamba cache block a sequence: vLLM's 256 do not fit every GPU)
        kw["max_num_seqs"] = spec["mns"]
    if spec["glyd"] is not None:  # (as vLLM's entry point does where glyd-gpu is installed; one process: VLLM_ENABLE_V1_MULTIPROCESSING=0)
        from glyd_gpu import vllm_plugin

        vllm_plugin.register()
        kw.update(quantization="glyd", additional_config={"glyd": spec["glyd"]})
    t = time.time()
    try:
        llm = LLM(**kw)
    except Exception as e:  # (it does not start: recorded, with vLLM's words)
        out["error"] = f"{type(e).__name__}: {str(e)[:2000]}"
        return out
    out["start_s"] = round(time.time() - t, 1)
    cfg = llm.llm_engine.vllm_config
    out["quantization"], out["quant_config"] = cfg.model_config.quantization, type(cfg.quant_config).__name__
    out["glyd_key"] = cfg.additional_config.get("glyd") if isinstance(cfg.additional_config, dict) else None
    out["blocks"], out["block_size"] = cfg.cache_config.num_gpu_blocks, cfg.cache_config.block_size
    out["gpu"] = torch.cuda.get_device_name()
    if spec["glyd"] is not None:
        out["inventory"] = llm.apply_model(_inventory)[0]
    if spec["texts"]:
        sp = SamplingParams(temperature=0, max_tokens=TOKENS, logprobs=TOP)
        outs = llm.generate(PROMPTS, sp, use_tqdm=False)
        out["tokens"] = [list(o.outputs[0].token_ids) for o in outs]
        out["logprobs"] = [_lp(o.outputs[0].logprobs) for o in outs]
        o = llm.generate([LONG], SamplingParams(temperature=0, max_tokens=1, prompt_logprobs=TOP), use_tqdm=False)[0]
        out["long_ids"], out["long_logprobs"] = list(o.prompt_token_ids), _lp(o.prompt_logprobs[1:])
        if spec["image"]:
            try:
                o = llm.chat([{"role": "user", "content": [{"type": "image_url", "image_url": {"url": _image()}}, {"type": "text", "text": "Describe this picture in one sentence."}]}],
                             SamplingParams(temperature=0, max_tokens=32, logprobs=TOP), use_tqdm=False)[0]
                out["image_tokens"], out["image_logprobs"] = list(o.outputs[0].token_ids), _lp(o.outputs[0].logprobs)
            except Exception as e:
                out["image_error"] = f"{type(e).__name__}: {str(e)[:500]}"
    if spec["tps"]:
        g = torch.Generator().manual_seed(1)
        vocab = min(getattr(cfg.model_config.hf_text_config, "vocab_size", 32000), 150000)
        mk = lambda n: [TokensPrompt(prompt_token_ids=torch.randint(1000, vocab, (spec["prompt"],), generator=g).tolist()) for _ in range(n)]
        llm.generate(mk(2), SamplingParams(temperature=0, max_tokens=8, ignore_eos=True), use_tqdm=False)  # (warm)
        out["tps"] = {}
        for n in spec["users"]:
            runs = []
            for _ in range(spec["reps"]):
                ps = mk(n)  # (prompts of their own each time: the prefix cache serves none of them)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                res = llm.generate(ps, SamplingParams(temperature=0, max_tokens=spec["new"], ignore_eos=True), use_tqdm=False)
                torch.cuda.synchronize()
                runs.append(sum(len(r.outputs[0].token_ids) for r in res) / (time.perf_counter() - t0))
            out["tps"][str(n)] = {"median": round(statistics.median(runs), 2), "runs": [round(r, 2) for r in runs]}
        smi = os.popen("nvidia-smi --query-gpu=clocks.sm,temperature.gpu,power.draw,memory.used --format=csv,noheader,nounits").read().strip()
        out["smi_after"] = smi
    return out


def _lp(seq):
    """vLLM's logprobs of each position (a dict of token id -> Logprob, or None) as [[id, logprob], ...] in rank order: floats kept exactly (JSON's repr round-trips)."""
    return [None if d is None else [[t, d[t].logprob] for t in sorted(d, key=lambda t: (d[t].rank, t))] for d in seq]


def _image():
    """A 448 x 448 PNG made here (shapes and a gradient), as a data URL: the same bytes in every run."""
    import base64
    import io

    import numpy as np
    from PIL import Image, ImageDraw

    y, x = np.mgrid[0:448, 0:448]
    a = np.stack([(x * 255 // 447), (y * 255 // 447), ((x + y) * 255 // 894)], -1).astype(np.uint8)
    im = Image.fromarray(a)
    d = ImageDraw.Draw(im)
    d.ellipse((60, 60, 200, 200), fill=(220, 30, 30))
    d.rectangle((250, 240, 400, 400), fill=(30, 30, 220))
    d.polygon([(224, 40), (300, 180), (148, 180)], fill=(250, 220, 0))
    b = io.BytesIO()
    im.save(b, format="PNG")
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()


def _inventory(model):
    """What Glyd packed in this model and what it left 16-bit: per layer packed its bf16 bytes and its pack's (a tied table once), the 16-bit
    tensors of 1 M values or more not packed (by module class and quantization method), the bytes of every tensor on the GPU; four packed Linears'
    products (glyd::vllm_linear) against F.linear on the decoded matrix, at 1, 7 and 64 tokens."""
    import torch
    import torch.nn.functional as F
    from glyd_gpu import kernels as g

    seen, packed, left, total = set(), [], {}, 0
    for t in list(model.parameters()) + list(model.buffers()):
        if t.is_cuda and t.untyped_storage().data_ptr() not in seen and t.numel():
            seen.add(t.untyped_storage().data_ptr())
            total += t.untyped_storage().nbytes()
    from glyd_gpu import vllm_plugin as vp

    opts = next((g.opts for m in model.modules() if (g := vp._glyd_config(getattr(m, "quant_config", None))) is not None), None)
    seen_packs, checked, worst = set(), 0, 0.0
    for name, m in model.named_modules():
        if getattr(m, "glyd_words", None) is not None and hasattr(m, "glyd_data"):
            K = m.weight.shape[1] if m.weight.dim() == 2 else m.embedding_dim
            ptr = m.glyd_data.data_ptr()
            nbytes = 0 if ptr in seen_packs else sum(x.numel() * x.element_size() for x in (m.glyd_data, m.glyd_a, m.glyd_b))
            seen_packs.add(ptr)
            packed.append({"name": name, "class": type(m).__name__, "rows": int(m.glyd_out), "cols": int(K), "bf16_bytes": 0 if not nbytes else int(m.glyd_out) * int(K) * 2,
                           "pack_bytes": int(nbytes), "verified": bool(getattr(m, "glyd_verified", False)), "words": list(m.glyd_words)})
            if checked < 4 and "Embedding" not in type(m).__name__ and "LMHead" not in type(m).__name__:
                checked += 1
                O = int(m.glyd_out)
                p = g.Mma12((O, K), m.glyd_data, m.glyd_a, m.glyd_b, m.glyd_words[0] & 0xFF) if len(m.glyd_words) == 4 else g.Mma((O, K), m.glyd_data, m.glyd_a, m.glyd_b, m.glyd_words)
                w = g.mma_unpack(p).float()
                for M in (1, 7, 64):
                    x = torch.randn(M, K, dtype=torch.bfloat16, device=w.device)
                    y = torch.ops.glyd.vllm_linear(x, m.glyd_data, m.glyd_a, m.glyd_b, m.glyd_words, getattr(m, "bias", None), O, False)
                    ref = F.linear(x.float(), w, None if getattr(m, "bias", None) is None else m.bias.float())
                    worst = max(worst, ((y.float() - ref).abs().max() / ref.abs().max()).item())
            continue
        for pn, t in m.named_parameters(recurse=False):
            if t.dtype in (torch.bfloat16, torch.float16) and t.numel() >= 1 << 20:
                qm = type(getattr(m, "quant_method", None)).__name__
                key = f"{type(m).__name__} / {qm} / {pn} / {t.dtype}".replace("torch.", "")
                e = left.setdefault(key, {"tensors": 0, "bytes": 0, "example": f"{name}.{pn}"})
                e["tensors"] += 1
                e["bytes"] += t.numel() * t.element_size()
    pb = sum(p["bf16_bytes"] for p in packed)
    pk = sum(p["pack_bytes"] for p in packed)
    return {"opts": opts, "packed": packed, "packed_count": len(packed), "verified_count": sum(p["verified"] for p in packed), "packed_bf16_bytes": pb, "packed_pack_bytes": pk,
            "saved_bytes": pb - pk, "left_16bit": dict(sorted(left.items(), key=lambda kv: -kv[1]["bytes"])), "gpu_tensor_bytes": total, "products_checked": checked,
            "products_worst_rel_error": worst}


def run(spec, out_dir, name):
    """spec in a vLLM of its own (this script, --child): its JSON, with what vLLM logged of the weights and the KV cache."""
    path = os.path.join(out_dir, name + ".json")
    e = dict(os.environ, VLLM_ENABLE_V1_MULTIPROCESSING="0")
    # Triton's autotuner picks a kernel's variant by timing it in each process (the Gated DeltaNet kernels of Qwen3.5/3.6/3.8): two runs of stock
    # eager can then differ in their bits. Its choices are kept on disk and taken again by every later run, stock's and Glyd's alike.
    e.setdefault("TRITON_CACHE_AUTOTUNING", "1")
    with open(os.path.join(out_dir, name + ".log"), "w") as log:
        try:
            rc = subprocess.run([sys.executable, os.path.abspath(__file__), "--child", json.dumps(spec), path], env=e, stdout=log, stderr=subprocess.STDOUT, timeout=5400).returncode
        except subprocess.TimeoutExpired:
            rc = "timeout (5400 s)"
    res = json.load(open(path)) if os.path.exists(path) else {"error": f"exit {rc}, no result (see {name}.log)"}
    text = open(os.path.join(out_dir, name + ".log"), errors="replace").read()
    for key, pat, f in (("weights_gib", r"Model loading took ([\d.]+) GiB", float), ("kv_tokens", r"GPU KV cache size: ([\d,]+) tokens", lambda s: int(s.replace(",", ""))),
                        ("kv_gib", r"Available KV cache memory: ([\d.]+) GiB", float), ("concurrency", r"Maximum concurrency for [\d,]+ tokens per request: ([\d.]+)x", float)):
        m = re.findall(pat, text)
        if m:
            res[key] = f(m[-1])
    if "error" in res:
        tail = [l for l in text.splitlines() if re.search(r"Error|error|refus|glyd:", l)][-6:]
        res["log_tail"] = tail
    with open(path, "w") as fh:
        json.dump(res, fh)
    return res


def first_diff(a, b, path=""):
    """Where two results first differ (None: equal bit for bit)."""
    if a == b:
        return None
    if isinstance(a, list) and isinstance(b, list):
        for i, (x, y) in enumerate(zip(a, b)):
            if x != y:
                return first_diff(x, y, f"{path}[{i}]")
        return f"{path}: lengths {len(a)} and {len(b)}"
    return f"{path}: {str(a)[:80]} != {str(b)[:80]}"


def agreement(a, b):
    """Greedy tokens: the share of positions equal before the first difference of each prompt (and its mean)."""
    shares = []
    for x, y in zip(a, b):
        n = 0
        while n < min(len(x), len(y)) and x[n] == y[n]:
            n += 1
        shares.append(n / max(1, len(x)))
    return round(sum(shares) / max(1, len(shares)), 4)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        res = child(json.loads(sys.argv[2]))
        with open(sys.argv[3], "w") as f:
            json.dump(res, f)
        return
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--out", required=True)
    ap.add_argument("--util", type=float, default=0.9)
    ap.add_argument("--mml", type=int, default=8192)
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--max-num-seqs", type=int, help="vLLM's max_num_seqs for every run, stock's and Glyd's (default: vLLM's)")
    ap.add_argument("--layout", default="auto")
    ap.add_argument("--steps", default=",".join(STEPS))
    ap.add_argument("--prompt", type=int, default=512)
    ap.add_argument("--new", type=int, default=256)
    ap.add_argument("--users", default="1,32")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--no-image", action="store_true")
    ap.add_argument("--fallback-layout", help="where Glyd's compiled run does not start, run it again in this layout (its results as glyd's)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    base = dict(model=a.model, dtype=a.dtype, mml=a.mml, mns=a.max_num_seqs, util=a.util, prompt=a.prompt, new=a.new, users=[int(u) for u in a.users.split(",")], reps=a.reps, image=not a.no_image)
    glyd = {"layout": a.layout}
    specs = {"stock-eager": dict(base, eager=True, glyd=None, texts=True, tps=False), "glyd-exact": dict(base, eager=True, glyd=dict(glyd, exact=True, verify=True), texts=True, tps=False),
             "stock": dict(base, eager=False, glyd=None, texts=True, tps=True), "glyd": dict(base, eager=False, glyd=dict(glyd), texts=True, tps=True)}
    specs["stock-b"], specs["glyd-b"] = specs["stock"], specs["glyd"]
    res, summary = {}, {"model": a.model, "args": vars(a)}
    for step in a.steps.split(","):
        t = time.time()
        res[step] = r = run(specs[step], a.out, step)
        print(f"{step}: {'FAILED to start: ' + r['error'][:300] if 'error' in r else 'ok'} ({time.time() - t:.0f} s){' weights ' + str(r.get('weights_gib')) + ' GiB' if 'weights_gib' in r else ''}"
              f"{', KV ' + format(r['kv_tokens'], ',') + ' tokens' if 'kv_tokens' in r else ''}{', tok/s ' + json.dumps({k: v['median'] for k, v in r['tps'].items()}) if 'tps' in r else ''}", flush=True)
    if a.fallback_layout and "error" in res.get("glyd", {}):
        r = run(dict(specs["glyd"], glyd={"layout": a.fallback_layout}), a.out, "glyd-" + a.fallback_layout)
        summary["glyd_fallback"] = {"layout": a.fallback_layout, "first_error": res["glyd"]["error"][:500]}
        print(f"glyd-{a.fallback_layout}: {'FAILED to start: ' + r['error'][:300] if 'error' in r else 'ok'}{' weights ' + str(r.get('weights_gib')) + ' GiB' if 'weights_gib' in r else ''}", flush=True)
        res["glyd"] = r
    for k in ("stock", "glyd"):  # (both starts' generate calls: one median)
        a, b = res.get(k), res.get(k + "-b")
        if a and b and "tps" in a and "tps" in b:
            for n in a["tps"]:
                runs = a["tps"][n]["runs"] + b["tps"][n]["runs"]
                a["tps"][n] = {"median": round(statistics.median(runs), 2), "runs": runs}
    s, x = res.get("stock-eager"), res.get("glyd-exact")
    if s and x and "error" not in s and "error" not in x:
        checks = {k: first_diff(s.get(k), x.get(k)) for k in ("tokens", "logprobs", "long_ids", "long_logprobs", "image_tokens", "image_logprobs")}
        summary["exact"] = {"equal": all(v is None for v in checks.values()), "diffs": checks, "image": "image_tokens" in s, "image_error": s.get("image_error") or x.get("image_error"),
                            "verified": x["inventory"]["verified_count"], "packed": x["inventory"]["packed_count"]}
        if not summary["exact"]["equal"]:  # (is stock eager the same from one run to the next? where it is not, the difference is not Glyd's to tell)
            s2 = run(specs["stock-eager"], a.out, "stock-eager-2")
            summary["exact"]["stock_self_diffs"] = {k: first_diff(s.get(k), s2.get(k)) for k in checks} if "error" not in s2 else s2["error"][:300]
        print(f"EXACT {'PASS' if summary['exact']['equal'] else 'FAIL'}: Glyd exact eager against stock eager, tokens and top-{TOP} logprobs of 8 prompts, prompt_logprobs of {len(s['long_ids'])} tokens"
              f"{', one image chat' if summary['exact']['image'] else ' (no image: ' + str(summary['exact']['image_error'])[:120] + ')'}; {x['inventory']['verified_count']} of {x['inventory']['packed_count']} packs verified bit for bit"
              f"{'' if summary['exact']['equal'] else '; differs at ' + json.dumps({k: v for k, v in checks.items() if v})[:400] + '; stock eager against itself: ' + json.dumps(summary['exact']['stock_self_diffs'])[:400]}", flush=True)
    s, g_ = res.get("stock"), res.get("glyd")
    if g_ and "error" not in g_:
        inv = g_["inventory"]
        summary["glyd"] = {k: g_.get(k) for k in ("weights_gib", "kv_tokens", "kv_gib", "concurrency", "tps", "quantization", "quant_config", "glyd_key", "start_s")}
        summary["inventory"] = {k: inv.get(k) for k in ("opts", "packed_count", "verified_count", "packed_bf16_bytes", "packed_pack_bytes", "saved_bytes", "gpu_tensor_bytes", "products_checked", "products_worst_rel_error")}
        summary["inventory"]["left_16bit_top"] = dict(list(inv["left_16bit"].items())[:8])
        print(f"PACKED: {inv['packed_count']} layers ({inv['verified_count']} verified bit for bit as packed), {inv['packed_bf16_bytes'] / 2**30:.2f} GiB bf16 held in {inv['packed_pack_bytes'] / 2**30:.2f} GiB ({inv['saved_bytes'] / 2**30:.2f} GiB saved); "
              f"products of {inv['products_checked']} against F.linear: worst relative error {inv['products_worst_rel_error']:.2e}; largest 16-bit left: "
              + "; ".join(f"{k} {v['bytes'] / 2**30:.2f} GiB" for k, v in list(inv["left_16bit"].items())[:3]), flush=True)
    elif g_:
        summary["glyd"] = {k: g_.get(k) for k in ("error", "log_tail")}
    if s:
        summary["stock"] = {k: s.get(k) for k in ("weights_gib", "kv_tokens", "kv_gib", "concurrency", "tps", "quantization", "quant_config", "start_s", "error", "log_tail")}
    if s and g_ and "error" not in s and "error" not in g_:
        summary["fused_agreement"] = agreement(s["tokens"], g_["tokens"])
        line = [f"weights {s.get('weights_gib')} -> {g_.get('weights_gib')} GiB", f"KV tokens {s.get('kv_tokens')} -> {g_.get('kv_tokens')}"]
        for n in s.get("tps", {}):
            line.append(f"{n} user{'s' if n != '1' else ''} {s['tps'][n]['median']} -> {g_['tps'][n]['median']} tok/s ({g_['tps'][n]['median'] / s['tps'][n]['median']:.3f}x)")
        print("STOCK -> GLYD: " + "; ".join(line) + f"; greedy agreement {summary['fused_agreement']}", flush=True)
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)


if __name__ == "__main__":
    main()
