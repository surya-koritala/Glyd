"""The report's measured table from the pod records in records/: python3 table.py [measured.json]

Two kinds of record, one directory a checkpoint:
- quant_parts.py (the 16-bit parts build, PR #42): stock.json, glyd.json (or <dir>-glyd/glyd.json where Glyd ran apart at 1, 8, 32, 64 users)
  are the fused runs (compiled, CUDA graphs: the memory and tok/s cells; their outputs are not stock's bits, so their greedy agreement with
  stock is a column), glyd-exact.json the exact eager run (its weights are a column; its speed was not measured), summary.json the exact
  check (stock eager against Glyd exact eager) and, where the fused runs were in the same directory, the fused agreement;
- check_fp8_vllm.py (the FP8 codes build, PR #37): check_fp8_vllm.json (runs stock-<mode>, glyd-<mode>, their bench; the glyd run is exact
  mode, so the cells are of the exact configuration) and check.txt (PASS/FAIL lines; "N failed" last).
CELLS below names each (GPU, build, directory[, the exact check's directory]) and FACTS each checkpoint's format and its ceiling from
research/quant-2026 (whole checkpoint at a fixed rate / at the best contexts measured, of the stored bytes).
A verdict that is not bit for bit is written FAIL, with where Glyd first differs (and stock's second run, where stock differs from
itself: still a FAIL, not attributed). ALSO adds a check of the same checkpoint and GPU in another kernel configuration, NOTES what a row's
records say beyond its summary.json. A tok/s median carries its runs' range; "within noise" where stock's and Glyd's ranges overlap.
A tok/s cell is marked "†" (KV-bound) where the load's requests (users x (prompt + output tokens)) exceed that run's KV cache.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
USERS = ("1", "8", "32", "64")
P16, F8 = "16-bit parts", "FP8 codes"
CELLS = [  # (GPU, build, record directory under records/, [the exact check's directory where it was run apart])
    # RTX 4090 (24 GB, Ada)
    ("RTX 4090", P16, "rtx4090-qrp/RedHatAI_Qwen3.5-9B-FP8-dynamic"),
    ("RTX 4090", F8, "rtx4090-qrp/RedHatAI_Qwen3.5-9B-FP8-dynamic-f8"),
    ("RTX 4090", P16, "rtx4090-qbm/RedHatAI_Qwen3.5-9B-quantized.w8a8", None, "rtx4090-qrp/RedHatAI_Qwen3.5-9B-quantized.w8a8"),
    ("RTX 4090", P16, "rtx4090-qbm/google_gemma-4-12B-it-qat-w4a16-ct", None, "rtx4090-qrp/google_gemma-4-12B-it-qat-w4a16-ct"),
    ("RTX 4090", P16, "rtx4090-qrp/QuantTrio_Qwen3.5-9B-AWQ"),
    ("RTX 4090", P16, "rtx4090-qbm/cyankiwi_Qwen3.6-27B-AWQ-INT4"),
    # RTX 5090 (32 GB, Blackwell)
    ("RTX 5090", P16, "rtx5090-qbm-run4/nvidia_Qwen3.8-27B-NVFP4", "rtx5090-qbm-run4/nvidia_Qwen3.8-27B-NVFP4-recheck"),
    ("RTX 5090", P16, "rtx5090-qbm-run4/nvidia_NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4"),
    ("RTX 5090", P16, "rtx5090-qbm-run5/nvidia_Qwen3.6-35B-A3B-NVFP4"),
    ("RTX 5090", P16, "rtx5090-qbm-run4/QuantTrio_Qwen3.6-35B-A3B-AWQ"),
    ("RTX 5090", P16, "rtx5090-qbm-run4/RedHatAI_Qwen3.5-9B-quantized.w8a8"),
    ("RTX 5090", F8, "rtx5090-qf8-run2/RedHatAI__gemma-4-26B-A4B-it-FP8-dynamic"),
    ("RTX 5090", F8, "rtx5090-qf8-run2/Qwen__Qwen3.8-27B-FP8"),
    # L40S (48 GB, Ada)
    ("L40S", P16, "l40s-qbp-run2/Qwen_Qwen3.8-27B-FP8", "l40s-qbp-run1/Qwen_Qwen3.8-27B-FP8"),
    ("L40S", F8, "l40s-qf8-run5/Qwen__Qwen3.8-27B-FP8"),
    ("L40S", P16, "l40s-qbp-run2/Qwen_Qwen3.6-35B-A3B-FP8", "l40s-qbp-run1/Qwen_Qwen3.6-35B-A3B-FP8"),
    ("L40S", F8, "l40s-qf8-run5/Qwen__Qwen3.6-35B-A3B-FP8"),
    ("L40S", P16, "l40s-qbp-run1/RedHatAI_gemma-4-26B-A4B-it-FP8-dynamic"),
    ("L40S", F8, "l40s-qf8-run5/RedHatAI__gemma-4-26B-A4B-it-FP8-dynamic"),
    ("L40S", F8, "l40s-qf8-run5/RedHatAI__NVIDIA-Nemotron-3.5-Lightning-30B-A3B-FP8"),
    ("L40S", P16, "l40s-qbp-run2a/nvidia_Gemma-4-31B-IT-NVFP4", "l40s-qbp-run1/nvidia_Gemma-4-31B-IT-NVFP4"),
    ("L40S", P16, "l40s-qbp-run2/nvidia_Qwen3.6-35B-A3B-NVFP4"),
    ("L40S", P16, "l40s-qbp-run1/google_gemma-4-12B-it-qat-w4a16-ct"),
    ("L40S", P16, "l40s-qbp-run1/cyankiwi_Qwen3.6-27B-AWQ-INT4"),
    ("L40S", P16, "l40s-qbp-run1/Qwen_Qwen3.5-35B-A3B-GPTQ-Int4"),
]
# Exact checks of the same checkpoint and GPU in another kernel configuration, after the row's own verdict: record: (directory, the configuration).
ALSO = {"rtx5090-qbm-run4/nvidia_Qwen3.8-27B-NVFP4": (
    "rtx5090-qbm-run5/nvidia_Qwen3.8-27B-NVFP4", "with vLLM's CUTLASS NVFP4 GEMM (FlashInfer's NVFP4 kernels off by VLLM_DISABLED_KERNELS, stock and Glyd; speed not measured in it)")}
NOTES = {  # what the row's records say beyond its summary.json
    "rtx5090-qbm-run4/nvidia_Qwen3.8-27B-NVFP4": "with FlashInfer's NVFP4 GEMM (the configuration benchmarked), 1 of 2 Glyd exact runs differs: the run's first one (its stock eager "
    "failed to start) equals the re-check's stock eager in every field; the re-check's own: "}
SETTINGS = {P16: "16-bit parts (512 in, 256 out, median of 2; exact: eager, top-5)", F8: "FP8 codes (1,024 in, 256 out, median of 3; exact: compiled, top-20)"}
FACTS = {  # checkpoint: (format, ceiling: fixed rate / best contexts, of the stored bytes; research/quant-2026, 2026-10-06)
    "RedHatAI/Qwen3.5-9B-FP8-dynamic": ("FP8, a scale a row", "not sampled; FP8 codes 16.6% / 20.1% of their bytes, 16-bit parts 32.6%"),
    "RedHatAI/Qwen3.5-9B-quantized.w8a8": ("INT8 W8A8", "24.2% / 26.0%"),
    "google/gemma-4-12B-it-qat-w4a16-ct": ("INT4 QAT (compressed-tensors)", "17.0% / 23.6%"),
    "QuantTrio/Qwen3.5-9B-AWQ": ("INT4 AWQ", "not sampled; AWQ codes 0-1%, scales 1%, 16-bit parts 32.6%"),
    "cyankiwi/Qwen3.6-27B-AWQ-INT4": ("INT4 AWQ (compressed-tensors), fp16 parts", "7.8% / 16.3%"),
    "nvidia/Qwen3.8-27B-NVFP4": ("NVFP4 + FP8 attention", "14.4% / 16.7%"),
    "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4": ("NVFP4 + FP8 mixer", "10.4% / 13.1%"),
    "nvidia/Qwen3.6-35B-A3B-NVFP4": ("NVFP4 + FP8 attention", "10.2% / 12.1%"),
    "nvidia/Gemma-4-31B-IT-NVFP4": ("NVFP4, bf16 attention", "23.2% / 24.5%"),
    "QuantTrio/Qwen3.6-35B-A3B-AWQ": ("INT4 AWQ", "13.4% / 20.1%"),
    "Qwen/Qwen3.5-35B-A3B-GPTQ-Int4": ("INT4 GPTQ", "17.6% / 20.7%"),
    "Qwen/Qwen3.8-27B-FP8": ("FP8, block scales", "19.4% / 21.7%"),
    "Qwen/Qwen3.6-35B-A3B-FP8": ("FP8, block scales", "17.5% / 19.9%"),
    "RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic": ("FP8, a scale a row", "15.2% / 21.0%"),
    "RedHatAI/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-FP8": ("FP8, a scale a tensor", "not sampled; FP8 codes 16.6% / 20.1% of their bytes"),
}


def load(p):
    try:
        return json.load(open(p))
    except (OSError, ValueError):
        return {}


def norm(r, req=None):
    """One vLLM run as {weights_gib, kv_tokens, users (at the run's context), context, tps {n: tok/s}, req (tokens a request), error}."""
    if not r:
        return {}
    tps = {n: v.get("median") for n, v in (r.get("tps") or {}).items()}
    tps.update({k[5:]: v.get("tokens_per_s") for k, v in (r.get("bench") or {}).items() if k.startswith("users")})
    spread = {n: (min(v["runs"]), max(v["runs"])) for n, v in (r.get("tps") or {}).items() if v.get("runs")}
    spread.update({k[5:]: (v["users"] * v["out"] / max(v["runs"]), v["users"] * v["out"] / min(v["runs"]))  # runs are seconds
                   for k, v in (r.get("bench") or {}).items() if k.startswith("users") and v.get("runs")})
    c = r.get("concurrency")
    sp = r.get("spec") or {}
    ctx = c[0] if isinstance(c, list) else sp.get("mml") or sp.get("max_model_len")
    out = {"weights_gib": r.get("weights_gib"), "context": ctx, "tps": tps, "spread": spread, "error": r.get("error"),
           "req": req or (sp.get("prompt", 0) + sp.get("new", 0)) or None}
    if not r.get("error"):
        out.update(kv_tokens=r.get("kv_tokens"), users=c[1] if isinstance(c, list) else c)
    return out


def packed(g):
    inv = g.get("inventory") or {}
    if not inv:
        return "-"
    return f"{inv.get('packed_count')} layers, {inv['packed_bf16_bytes'] / 2**30:.2f} GiB bf16 -> {inv['packed_pack_bytes'] / 2**30:.2f} GiB"


def agreement(a, b):
    """quant_parts.py's: greedy tokens, the share of positions equal before the first difference of each prompt (and its mean)."""
    shares = []
    for x, y in zip(a, b):
        n = 0
        while n < min(len(x), len(y)) and x[n] == y[n]:
            n += 1
        shares.append(n / max(1, len(x)))
    return round(sum(shares) / max(1, len(shares)), 4)


def fused(st, g, summary):
    """The fused run against stock: its greedy agreement (summary.json's, else from the two runs' tokens) and the packs it verified as packed."""
    inv = g.get("inventory") or {}
    if not inv:
        return "-"
    a = summary.get("fused_agreement")
    if a is None and st.get("tokens") and g.get("tokens"):
        a = agreement(st["tokens"], g["tokens"])
    return f"greedy agreement {a:.2f}; {inv.get('verified_count')} of {inv.get('packed_count')} packs verified as packed" if a is not None else "-"


def first(diffs):
    """The first differing field of quant_parts.py's diffs, as 'field index' (tokens before logprobs)."""
    k = next((k for k in ("tokens", "long_ids", "image_tokens", "logprobs", "long_logprobs", "image_logprobs") if (diffs or {}).get(k)), None)
    return f"{k} {diffs[k].split(':')[0]}" if k else "?"


def record(d, exact_dir=None, stock_dir=None):
    """(model, stock, glyd, exact verdict, exact-mode weights GiB, the fused run against stock, what was packed) of one checkpoint's record directory."""
    f8 = load(os.path.join(d, "check_fp8_vllm.json"))
    if f8:
        runs = f8.get("runs") or {}
        st = next((v for k, v in runs.items() if k.startswith("stock") and "poison" not in k), {})
        g = next((v for k, v in runs.items() if k.startswith("glyd") and "poison" not in k and "fused" not in k), {})
        txt = open(os.path.join(d, "check.txt"), errors="replace").read() if os.path.exists(os.path.join(d, "check.txt")) else ""
        fails = [ln for ln in txt.splitlines() if ln.startswith("FAIL:") and " failed;" not in ln]
        if "PASS: exact" in txt and not fails:
            exact = "bit for bit: logits (tokens, their log probabilities, top-20 prompt log probabilities of 4 prompts) in the compiled run; the first batch's greedy tokens at 1 and 32 users"
        elif "PASS: exact" in txt:
            at = ", ".join(ln.split("[users")[1].split("]")[0] + " users" for ln in fails if "exact-bench" in ln) or "a check"
            exact = f"FAIL at {at} (exact-bench: the first batch's greedy tokens differ; the 4 prompts' logits bit for bit; the same-blocks rerun that tells vLLM's block admission from Glyd is not in this record)"
        elif "FAIL: exact" in txt:
            exact = "DIFFERS"
        elif "INCONCLUSIVE" in txt:
            exact = "inconclusive: stock vLLM is not reproducible against itself here"
        else:
            exact = "not run"
        m = next((ln for ln in txt.splitlines() if ln.startswith("PASS: packed")), "")
        pk = m.split(": ", 2)[-1].split(" of weights")[0] if m else "-"
        model = (f8.get("model") or os.path.basename(d)).replace("__", "/")
        model = next((k for k in FACTS if k.split("/")[-1] in model), model)
        gx = norm(g, 1024 + 256)
        xw = f"{gx['weights_gib']:.2f} (same run: exact mode)" if gx.get("weights_gib") and not gx.get("error") else "-"
        return model, norm(st, 1024 + 256), gx, exact, xw, "-", pk
    st = load(os.path.join(stock_dir or d, "stock.json"))
    g = load(os.path.join(d + "-glyd", "glyd.json")) or load(os.path.join(d, "glyd.json"))
    summary = load(os.path.join(d + "-glyd", "summary.json")) if os.path.isdir(d + "-glyd") else load(os.path.join(d, "summary.json"))
    ex = load(os.path.join(exact_dir or d, "summary.json")).get("exact") or {}
    x = load(os.path.join(exact_dir or d, "glyd-exact.json"))
    model = (st.get("spec") or g.get("spec") or {}).get("model") or os.path.basename(d).replace("_", "/", 1)
    if not ex:
        exact = "not run"
    elif ex.get("equal"):
        exact = f"bit for bit: 8 prompts' greedy tokens and top-5 logprobs, a 700-token prompt's logprobs{', an image chat' if ex.get('image') else ''} (stock eager vs Glyd exact eager; {ex.get('verified')} of {ex.get('packed')} packs verified against their weights)"
    else:
        exact = "FAIL: Glyd exact eager differs from stock eager (first at " + first(ex.get("diffs")) + ")"
        if isinstance(ex.get("stock_self_diffs"), dict) and any(ex["stock_self_diffs"].values()):
            exact += "; stock eager also differs from itself here (first at " + first(ex["stock_self_diffs"]) + "), so not passed and not attributed (layer-level check pending)"
    w0, xw = (st or {}).get("weights_gib"), x.get("weights_gib")
    xcell = f"{xw:.2f} ({(w0 - xw) / w0 * 100:.1f}%)" if xw and w0 and not x.get("error") and not st.get("error") else "-"
    return model, norm(st), norm(g), exact, xcell, fused(st, g, summary), packed(g)


def cell(st, g, n):
    """tok/s at n users, stock / Glyd (Glyd's share); a side is marked † where n requests exceed its KV cache (KV-bound: queueing)."""
    a, b = st.get("tps", {}).get(n), g.get("tps", {}).get(n)
    if a is None and b is None:
        return "-"

    def f(r, x):
        if x is None:
            return "-"
        kv, req = r.get("kv_tokens"), r.get("req")
        lo, hi = r.get("spread", {}).get(n, (x, x))
        return f"{x:,.0f}" + (f" [{lo:,.0f}-{hi:,.0f}]" if hi - lo >= 0.5 else "") + ("†" if kv and req and int(n) * req > kv else "")
    if not (a and b):
        return f"{f(st, a)} / {f(g, b)}"
    (sl, sh), (gl, gh) = st["spread"].get(n, (a, a)), g["spread"].get(n, (b, b))
    noise = gl <= sh and sl <= gh  # the runs' ranges overlap: no verdict either way
    return f"{f(st, a)} / {f(g, b)} ({b / a:.2f}x{', within noise' if noise else ''})"


def users(r):
    if not r.get("users") or not r.get("kv_tokens"):
        return "-"
    return f"{r['users']:.1f} ({r['kv_tokens']:,} tokens)" + ("" if r.get("context") in (None, 8192) else f" at {r['context']:,}")


def rows():
    for c in CELLS:
        gpu, build, d = c[0], c[1], os.path.join(HERE, "records", c[2])
        ex = os.path.join(HERE, "records", c[3]) if len(c) > 3 and c[3] else None
        sd = os.path.join(HERE, "records", c[4]) if len(c) > 4 and c[4] and os.path.isdir(os.path.join(HERE, "records", c[4])) else None
        if not os.path.isdir(d):
            continue
        model, st, g, exact, xw, fz, pk = record(d, ex, sd)
        exact = NOTES.get(c[2], "") + exact
        if c[2] in ALSO:
            also, how = ALSO[c[2]]
            exact += f"; {how}: " + record(os.path.join(HERE, "records", also), None, os.path.join(HERE, "records", also))[3] + f" (`records/{also}`)"
        commit = " / ".join(dict.fromkeys(open(os.path.join(os.path.dirname(x), "COMMIT")).read()[:8] for x in (d, ex) if x))
        w0, w1 = st.get("weights_gib"), g.get("weights_gib")
        mem = f"{w0:.2f} / {w1:.2f}" if w0 and w1 else f"{w0 or '-'} / {w1 or '-'}"
        notes = [f"{who} did not start" for who, r in (("stock", st), ("Glyd", g)) if r.get("error")]
        saved = f"{w0 - w1:.2f} ({(w0 - w1) / w0 * 100:.1f}%)" if w0 and w1 and not notes else "-"
        fmt, ceil = FACTS.get(model, ("?", "?"))
        rec = f"`records/{c[2]}`" + (f" (exact check: `records/{c[3]}`)" if ex else "")
        yield {"model": model, "gpu": gpu, "build": build, "format": fmt, "stock": st, "glyd": g, "exact": exact, "exact_weights": xw, "fused": fz, "packed": pk,
               "ceiling": ceil, "record": c[2], "exact_record": c[3] if ex else None,
               "commit": commit, "cells": [model, gpu, SETTINGS[build], fmt, mem + ("; " + "; ".join(notes) if notes else ""), saved, xw, f"{users(st)} / {users(g)}"]
               + [cell(st, g, n) for n in USERS] + [exact, fz, pk, ceil, commit, rec]}


HDR = ["Checkpoint", "GPU", "Glyd build", "Format", "Weights GiB, stock / Glyd", "Saved GiB (%)", "Exact-mode weights GiB (saved %)",
       "Users at 8K (KV tokens), stock / Glyd"] + [f"tok/s {n} user{'s' if n != '1' else ''}, stock / Glyd" for n in USERS] + [
       "Bit for bit", "The fused run against stock", "What Glyd packs", "Ceiling (quant-2026): fixed rate / best contexts", "Commit (run / exact check)", "Record"]

if __name__ == "__main__":
    out = list(rows())
    print("| " + " | ".join(HDR) + " |")
    print("| " + " | ".join([":---"] * 4 + ["---:"] * 8 + [":---"] * 6) + " |")
    for r in out:
        print("| " + " | ".join(str(x) for x in r["cells"]) + " |")
    if len(sys.argv) > 1:
        json.dump([{k: v for k, v in r.items() if k != "cells"} for r in out], open(sys.argv[1], "w"), indent=1)
