"""The results report: per 2026 model and GPU, Glyd as measured against bf16 as ESTIMATED (bf16 is not run).

    python3 tools/report/report.py --runs DIR [DIR ...] [--headers DIR] [--out README.md] [--json measured.json]
    python3 tools/report/report.py --selftest

Reader: every vLLM server log under the run directories (a segment from one "non-default args" line to the next): the model,
Glyd or bf16 (--quantization glyd), the settings, "Model loading took", "Available KV cache memory", "GPU KV cache size" and
its "Maximum concurrency", Glyd's layout line; and the speed beside it: `vllm bench serve` results (c<N>.json, *-rateinf.json:
output tokens/s at N users) and speed1.py results (one user). The GPU from the nearest gpu.txt / machine.txt.

bf16 estimator, for the settings of a Glyd run:
- weights: the bytes vLLM 0.30.0 loads in bf16 (research/inventory-2026/inventory.json, "loaded_by_vllm_gb") + vLLM's own
  buffers (0.13 GiB: Qwen3.5-9B's measured 17.66 GiB less its 17.53 checkpoint GiB), split over tensor-parallel GPUs;
- KV room: Glyd's measured KV memory - (bf16's weights - Glyd's measured weights), at Glyd's util;
- users at a context C: the run's logged "Maximum concurrency" at its max-len L, times cost(L) / cost(C), one request's KV cost
  (a hybrid's in vLLM's blocks: ceil(len / block) + its recurrent state's blocks; else fit.kv_bytes, sliding windows counted);
  bf16's = Glyd's times bf16's KV room / Glyd's KV memory;
- output tokens/s at N users: a step reads the weights once (a MoE's experts as N tokens touch them), each user's KV and
  recurrent state (taken as float32), or saturates (an empirical term); each request's prompt is compute; N is capped by the
  users the KV room runs at once and --max-num-seqs. Four constants are fitted on every bf16 point measured on a 2026 model;
  the report prints the in-sample error and the held-out error (each run predicted without it), the band to use.
Where the same pod measured bf16, the report shows that measurement instead of the estimate.
Estimates are for internal reports, never published numbers.
"""
import argparse
import glob
import json
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "glyd-gpu-py"))
from glyd_gpu.fit import kv_bytes  # noqa: E402  (standard library only)

GiB = 2**30
BUFFERS = 0.13 * GiB  # vLLM's own buffers beside the checkpoint's bytes (Qwen3.5-9B: 17.66 measured, 17.53 checkpoint GiB)
INVENTORY = os.path.join(ROOT, "research", "inventory-2026", "inventory.json")
FACTS = os.path.join(ROOT, "research", "report-2026-10-06", "facts.json")
# GPU: (memory bandwidth GB/s, dense bf16 tensor TFLOPS with fp32 accumulation), the makers' data sheets. Longest name first.
GPUS = {
    "H100 80GB HBM3": (3350, 989), "H100 NVL": (3900, 835), "H100 PCIe": (2000, 756), "GH200": (4000, 989), "H200": (4800, 989),
    "A100-SXM4-80GB": (2039, 312), "A100 80GB PCIe": (1935, 312), "A100-SXM4-40GB": (1555, 312), "A100-PCIE-40GB": (1555, 312),
    "L40S": (864, 362), "L4": (300, 121), "A10": (600, 125), "RTX 5090": (1792, 209.5), "RTX 4090": (1008, 165.2),
    "RTX 4080 SUPER": (736, 104.4), "RTX 6000 Ada": (960, 364.2), "RTX A6000": (768, 154.8),
}
SHORT = {"H100 80GB HBM3": "H100 SXM", "A100-SXM4-80GB": "A100 80GB", "A100-SXM4-40GB": "A100 40GB"}

# ---------------------------------------------------------------------------------------------------------------------------- facts

TOWER = re.compile(r"(^|\.)(visual|vision\w*|vpm|apm|audio\w*|sound\w*|tts|resampler|mm_projector|multi_modal_projector|merger|"
                   r"embed_vision|embed_audio|vit_\w+|mlp1|linear_proj|image_newline|spatial_projectors|layerwise_projectors)(\.|$)")
MTP = re.compile(r"(^|\.)(mtp|mtp_layers|nextn\w*)(\.|$)")
EMBED = re.compile(r"(embed_tokens|word_embeddings|\.embed\.|wte|embeddings\.weight|ngram_embeddings\.embedders)")
KV_KEYS = ("num_hidden_layers", "layer_types", "kv_lora_rank", "qk_rope_head_dim", "num_key_value_heads", "num_attention_heads",
           "head_dim", "hidden_size", "sliding_window", "sliding_window_pattern", "full_attention_interval")  # what fit.kv_bytes reads
SIZE = {"BF16": 2, "F16": 2, "F32": 4, "F8_E4M3": 1, "F8_E5M2": 1, "U8": 1, "I8": 1, "I32": 4, "I64": 8, "F64": 8}


def facts_of(header, inv):
    """What decoding reads of a checkpoint: dense bytes (every step), routed experts' bytes, active parameters, KV and state."""
    cfg = header["config"]
    t = cfg.get("text_config", cfg)
    tied = cfg.get("tie_word_embeddings", t.get("tie_word_embeddings", False))
    dense = experts = tower = embed = params_dense = params_exp = 0
    has_head = False
    for shard in header["shards"].values():
        for name, (dtype, shape) in shard.items():
            n = math.prod(shape)
            b = n * SIZE.get(dtype, 2)
            if MTP.search(name):
                continue
            if TOWER.search(name):
                tower += b
            elif EMBED.search(name):
                embed += b
            elif ".experts." in name or name.endswith(".experts") or re.search(r"\.experts\.\w+$", name):
                if "shared_expert" not in name:
                    experts += b
                    params_exp += n
                    continue
                dense += b
                params_dense += n
            else:
                has_head |= "lm_head" in name or "unembed" in name
                dense += b
                params_dense += n
    if tied and not has_head:  # the embedding is the output layer too: read every step
        dense += embed
        params_dense += embed // 2
    E = t.get("num_experts") or t.get("n_routed_experts") or t.get("num_local_experts") or 0
    k = t.get("num_experts_per_tok") or t.get("moe_topk") or t.get("top_k") or 0
    kinds = t.get("layer_types") or []
    state = 0  # recurrent state a user, bytes (float32), read and written each step
    if t.get("linear_num_value_heads"):
        state = kinds.count("linear_attention") * t["linear_num_value_heads"] * t["linear_key_head_dim"] * t["linear_value_head_dim"] * 4
    elif t.get("mamba_num_heads"):
        n_m = (t.get("hybrid_override_pattern") or "").count("M") or kinds.count("mamba")
        state = n_m * t["mamba_num_heads"] * t["mamba_head_dim"] * t.get("ssm_state_size", 128) * 4
    loaded = inv["loaded_by_vllm_gb"] * 1e9
    try:
        kv = kv_bytes(cfg, 1)
    except (KeyError, TypeError, ZeroDivisionError):  # a config fit.kv_bytes does not read: only the measured bytes a token then
        kv = None
    return {"loaded": loaded, "tower": tower, "dense": dense, "experts": experts, "E": E, "k": k,
            "params": params_dense + (params_exp * k / E if E else params_exp), "kv_tok": kv, "state": state,
            "kv_cfg": {k: t[k] for k in KV_KEYS if k in t},
            "arch": inv["arch"], "format": "bf16", "tiered_pct": inv["projected_tiered_saving_pct"],
            "b12_pct": inv["projected_12bit_saving_pct"], "b12_dc_pct": inv.get("projected_12bit_saving_on_a100_h100_pct")}


def build_facts(headers, out=FACTS):
    inv = {m["model"]: m for m in json.load(open(INVENTORY))["models"]}
    facts = {}
    for f in sorted(glob.glob(os.path.join(headers, "*.json"))):
        h = json.load(open(f))
        if h.get("repo") in inv and h.get("shards"):
            facts[h["repo"]] = facts_of(h, inv[h["repo"]])
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(facts, open(out, "w"), indent=1, sort_keys=True)
    return facts

# --------------------------------------------------------------------------------------------------------------------------- reader


def gpu_of(path, stop):
    """The GPU (name, count) from the nearest gpu.txt / machine.txt at or above path."""
    d = path if os.path.isdir(path) else os.path.dirname(path)
    while True:
        for f in ("gpu.txt", "machine.txt"):
            p = os.path.join(d, f)
            if os.path.exists(p):
                lines = [l for l in open(p, errors="replace") if "NVIDIA" in l or "RTX" in l]
                for name in sorted(GPUS, key=len, reverse=True):
                    hits = [l for l in lines if name in l]
                    if hits:
                        m = re.search(r"(\d+) x ", hits[0])
                        return name, int(m.group(1)) if m else len(hits) if f == "gpu.txt" else 1
        if os.path.abspath(d) in (os.path.abspath(stop), "/") or not d:
            return None, 0
        d = os.path.dirname(d)


VERSION = re.compile(r"glyd(?:-gpu)?\s*(?:==|\s)\s*(\d+\.\d+)\.\d+(\+\w+)?")


def version_of(path, stop):
    """Glyd's version of a run ("0.29", or "0.29+runfam" for a branch build): a v0NN directory on its path, else the nearest
    env.txt / versions.txt / install.out / job.log that names it; None: not known (a development build)."""
    m = re.search(r"/v0(\d)(\d)[/-]", path)
    if m:
        return f"0.{m.group(1)}{m.group(2)}"
    d = os.path.dirname(path)
    while True:
        found = [x.group(1) + (x.group(2) or "") for f in ("env.txt", "versions.txt", "install.out", "job.log")
                 if os.path.exists(os.path.join(d, f)) for x in VERSION.finditer(open(os.path.join(d, f), errors="replace").read())]
        if found:
            return max(found, key=len)  # a branch build's own version over the release it starts from
        if os.path.abspath(d) in (os.path.abspath(stop), "/") or not d:
            return None
        d = os.path.dirname(d)


def num(s):
    return float(s.replace(",", ""))


def model_name(model, served=None):
    """Org/Name: a Hub cache path (.../models--Org--Name/snapshots/...) mapped back, else the served name for a local path."""
    m = re.search(r"models--([^/]+)--([^/]+)", model or "")
    if m:
        return f"{m.group(1)}/{m.group(2)}"
    return served if (model or "").startswith("/") and served else model


def parse_log(text):
    """Each vLLM server in a log: one dict per 'non-default args' segment."""
    out = []
    for seg in re.split(r"(?=^.*non-default args: )", text, flags=re.M)[1:]:
        m = re.search(r"non-default args: (\{.*\})", seg)
        args = {}
        for key in ("model", "quantization", "max_model_len", "max_num_seqs", "enforce_eager", "tensor_parallel_size",
                    "gpu_memory_utilization", "language_model_only"):
            v = re.search(rf"'{key}': ('([^']*)'|[\w.]+)", m.group(1))
            if v:
                args[key] = v.group(2) if v.group(2) is not None else v.group(1)
        served = re.search(r"'served_model_name': \['([^']*)'", m.group(1))
        util = re.findall(r"gpu-memory-utilization=([\d.]+)", seg)  # glyd serve's setting, which non-default args leave out
        r = {"model": model_name(args.get("model"), served and served.group(1)),
             "server": "glyd" if args.get("quantization") == "glyd" else "bf16" if not args.get("quantization") else args["quantization"],
             "max_model_len": int(args["max_model_len"]) if "max_model_len" in args else None,
             "max_num_seqs": int(args["max_num_seqs"]) if "max_num_seqs" in args else None,
             "eager": args.get("enforce_eager") == "True", "tp": int(args.get("tensor_parallel_size", 1)),
             "util": float(util[0]) if util else float(args.get("gpu_memory_utilization", 0.9)),
             "lm_only": args.get("language_model_only") == "True"}
        w = re.findall(r"Model loading took ([\d.]+) GiB", seg)
        kvm = re.findall(r"Available KV cache memory: ([\d.]+) GiB", seg)
        kv = re.findall(r"GPU KV cache size: ([\d,]+) tokens", seg)
        conc = re.findall(r"Maximum concurrency for ([\d,]+) tokens per request: ([\d.]+)x", seg)
        bs = re.findall(r"Setting attention block size to (\d+) tokens", seg)
        lay = re.findall(r"glyd: (mma12|mma|tiered|12-bit)\w* layout", seg)
        r.update(weights=float(w[-1]) * GiB if w else None, kv_mem=float(kvm[-1]) * GiB if kvm else None,
                 kv_tokens=int(num(kv[-1])) if kv else None, block=int(bs[-1]) if bs else None,
                 conc_len=int(num(conc[-1][0])) if conc else None, conc=float(conc[-1][1]) if conc else None,
                 layout={"mma12": "12-bit", "mma": "tiered"}.get(lay[-1], lay[-1]) if lay else None)
        if r["model"]:
            out.append(r)
    return out


def bench_of(path):
    """(users, output tokens/s, prompt tokens, output tokens, model, TTFT p50 ms) of one bench result; None if not closed-loop."""
    try:
        d = json.load(open(path))
    except (ValueError, OSError):
        return None
    if "decode_tok_s_median" in d:  # speed1.py: one user, streamed
        P, O = d.get("prompt_tokens", 1024), d.get("new", 256)
        return 1, O / (d["ttft_ms_median"] / 1e3 + (O - 1) / d["decode_tok_s_median"]), P, O, d.get("model"), d["ttft_ms_median"]
    if "output_throughput" not in d or d.get("request_rate") not in ("inf", None) or not d.get("completed"):
        return None
    n = d["num_prompts"]
    return (d.get("max_concurrency") or n, d["output_throughput"], d["total_input_tokens"] / n, d["total_output_tokens"] / n,
            d.get("model_id"), d.get("median_ttft_ms"))


def rel(p, root):
    """A run's source: repository-relative, else relative to the folder that holds the results root (ceiling/results-x/...)."""
    a = os.path.abspath(p)
    return os.path.relpath(a, ROOT) if a.startswith(ROOT + os.sep) else os.path.relpath(a, os.path.dirname(os.path.abspath(root)))


def read_runs(roots):
    """Every server found under the roots, one dict each: memory from its log, speed from the bench results beside it."""
    runs, seen = [], set()
    for root in roots:
        start = len(runs)
        files = [os.path.join(dp, f) for dp, _, fs in os.walk(root) for f in fs]
        benchdirs = {}
        for p in files:
            if re.search(r"(^c\d+|rateinf|^speed-.*)\.json$", os.path.basename(p)):
                b = bench_of(p)
                if b:
                    benchdirs.setdefault(os.path.dirname(p), []).append((p, b))
        logs = {}
        for p in files:
            if not re.search(r"\.(log|txt|err|out)$", p) or os.path.getsize(p) > 50e6:
                continue
            text = open(p, errors="replace").read()
            if "non-default args" not in text or "Model loading took" not in text:
                continue
            for r in parse_log(text):
                r["src"], r["mtime"] = p, os.path.getmtime(p)
                logs.setdefault(p, []).append(r)
        for d, bl in benchdirs.items():
            speed1 = [(p, b) for p, b in bl if os.path.basename(p).startswith("speed-")]
            serve = [(p, b) for p, b in bl if (p, b) not in speed1]
            for p, b in speed1:  # one user's speed alone: Glyd or bf16 from the file's name
                nm = os.path.basename(p)
                gpu, n = gpu_of(p, root)
                key = (b[4], gpu, nm, round(b[1], 3))
                if key in seen:  # a record's copy of a run seen already
                    continue
                seen.add(key)
                runs.append({"model": model_name(b[4]), "server": "bf16" if nm.startswith("speed-vllm") else "glyd", "gpu": gpu,
                             "ngpu": n, "eager": "compiled" not in nm and not nm.startswith("speed-vllm"), "tp": 1, "src": p,
                             "speed": {1: b[1]}, "ttft": {1: b[5]}, "speed1": True, "P": b[2], "O": b[3], "weights": None,
                             "kv_tokens": None, "version": version_of(p, root)})
            if serve:  # the server these were measured on: the last one that started in its directory tree
                mine = [r for lp, rs in logs.items() if lp.startswith(d + os.sep) for r in rs if r["kv_tokens"]]
                if not mine:
                    continue
                r = max(mine, key=lambda r: r["mtime"])
                r["speed"] = {b[0]: b[1] for _, b in serve}
                r["ttft"] = {b[0]: b[5] for _, b in serve}
                r["P"], r["O"] = serve[0][1][2], serve[0][1][3]
                r["benchdir"] = rel(d, root)
        for p, rs in logs.items():
            for r in rs:
                if not r.get("kv_tokens"):
                    if r.get("weights"):  # started, no KV line: still a weights point
                        r["gpu"], r["ngpu"] = gpu_of(p, root)
                        key = (r["model"], r["gpu"], r["server"], r["tp"], r["weights"], None, "null")
                        if key not in seen:
                            seen.add(key)
                            runs.append(r)
                    continue
                r["gpu"], r["ngpu"] = gpu_of(p, root)
                key = (r["model"], r["gpu"], r["server"], r["tp"], r["weights"], r["kv_tokens"], json.dumps(r.get("speed"), sort_keys=True))
                if key not in seen:
                    seen.add(key)
                    runs.append(r)
        for r in runs[start:]:  # this root's runs: Glyd's version, the source relative
            if r["server"] == "glyd" and "version" not in r:
                r["version"] = version_of(r["src"], root)
            r["src"] = rel(r["src"], root)
            r.pop("mtime", None)
    return [r for r in runs if r.get("gpu")]

# ------------------------------------------------------------------------------------------------------------------------ estimator

CAL = {"eb": 0.8, "ef": 0.6, "ep": 0.6, "c0": 0.002}  # fitted by calibrate(); these are the starting values


def bf16_weights(f, tp=1, lm_only=False):
    return (f["loaded"] - (f["tower"] if lm_only else 0)) / tp + BUFFERS


def kv_room_bf16(run, f):
    """bf16's KV memory at a Glyd run's settings: Glyd's measured KV memory less bf16's extra weight bytes (negative: none)."""
    return run["kv_mem"] - (bf16_weights(f, run["tp"], run.get("lm_only")) - run["weights"])


def state_blocks(f):
    """A hybrid model's KV blocks a request holds for its recurrent state, by vLLM 0.30's accounting: two a group of state
    layers, the groups as many layers as the attention layers (checked against the logged "Maximum concurrency" of
    Qwen3.5-9B, Qwen3.8-27B and Qwen3.5-35B-A3B at 4,096, 8,192, 189,440 and 262,144 tokens: 6 on all). None: not known."""
    kinds = (f.get("kv_cfg") or {}).get("layer_types") or []
    lin = kinds.count("linear_attention") + kinds.count("mamba")
    att = len(kinds) - lin
    return 2 * math.ceil(lin / att) if lin and att else None


def cost(f, X, block, running=False):
    """One request of X tokens in KV units: blocks for a hybrid (attention blocks + its state blocks; running: one state
    block a group, what the scheduler holds while it runs), else bytes (fit.kv_bytes: sliding windows counted)."""
    if f["state"]:
        m = state_blocks(f)
        return math.ceil(X / block) + (m // 2 if running else m) if m and block else None
    return kv_bytes(f["kv_cfg"], X) if f.get("kv_cfg") and f["kv_tok"] else X


def users_at(run, f, C, running=False):
    """(users of C tokens each that the run's KV memory holds, whether that is vLLM's own logged count): the run's logged
    "Maximum concurrency" at its max-len, converted to C by the cost of one request at each length."""
    if run.get("conc"):
        conc, L = run["conc"], run["conc_len"]
    elif run.get("kv_tokens") and run.get("max_model_len"):
        conc, L = run["kv_tokens"] / run["max_model_len"], run["max_model_len"]
    else:
        return None, False
    if L == C and not running:
        return conc, True
    a, b = cost(f, L, run.get("block")), cost(f, C, run.get("block"), running)
    return (conc * a / b, False) if a and b else (None, False)


def users_bf16(run, f, C, running=False):
    """bf16's users at C on a Glyd run's settings: Glyd's users scaled by the KV memory bf16 would have."""
    u = users_at(run, f, C, running)[0]
    if u is None or not run.get("kv_mem") or not run.get("weights"):
        return None
    return max(0.0, u * kv_room_bf16(run, f) / run["kv_mem"])


def step(f, gpu, B, ctx, tp, cal):
    bw, fl = GPUS[gpu][0] * 1e9 * cal["eb"] * tp, GPUS[gpu][1] * 1e12 * cal["ef"] * tp
    touched = 1 - (1 - f["k"] / f["E"]) ** B if f["E"] else 1
    kv = kv_bytes(f["kv_cfg"], int(ctx)) if f.get("kv_cfg") and f["kv_tok"] else (f["kv_tok"] or 0) * ctx  # sliding windows counted
    read = f["dense"] + f["experts"] * touched + B * (kv / tp + 2 * f["state"] / tp)
    # ef: an empirical saturation term (prefill interference, scheduling) more than the tensor cores' compute
    return cal["c0"] + max(read / bw, 2 * B * f["params"] / fl)


def tok_s(f, gpu, B, P, O, cap, max_seqs, tp=1, cal=None):
    """bf16's output tokens/s with B users of P-token prompts and O-token answers, at most `cap` running at once (the KV
    room's users at P+O; None: no limit); None where no request fits."""
    cal = cal or CAL
    beff = min(B, max_seqs or 256, int(cap) if cap is not None else B)
    if beff < 1:
        return None
    fl = GPUS[gpu][1] * 1e12 * cal["ep"] * tp
    pre = max(2 * P * f["params"] / fl, (f["dense"] + f["experts"]) / (GPUS[gpu][0] * 1e9 * cal["eb"] * tp))
    return beff * O / (beff * pre + O * step(f, gpu, beff, P + O / 2, tp, cal))


def points(runs, facts):
    """The bf16 speeds measured on 2026 models: (run, users, measured tok/s)."""
    return [(r, u, v) for r in runs if r["server"] == "bf16" and r.get("speed") and r["model"] in facts and r["gpu"] in GPUS
            for u, v in sorted(r["speed"].items())]


def predict(r, u, facts, cal):
    f = facts[r["model"]]
    P, O = r.get("P", 1024), r.get("O", 256)
    cap = users_at(r, f, P + O, running=True)[0]
    return tok_s(f, r["gpu"], u, P, O, cap, r.get("max_num_seqs"), r.get("tp", 1), cal)


def calibrate(runs, facts):
    """The four constants by a coarse-to-fine grid on the squared log error over every bf16 point; returns them and the errors."""
    pts = points(runs, facts)
    if not pts:
        return dict(CAL), []

    def loss(c):
        s = 0
        for r, u, v in pts:
            p = predict(r, u, facts, c)
            s += math.log(p / v) ** 2 if p else 9
        return s
    best = dict(CAL)
    grid = {"eb": (0.3, 1.0), "ef": (0.02, 1.0), "ep": (0.1, 1.0), "c0": (0.0, 0.02)}
    for _ in range(6):
        for k, (lo, hi) in grid.items():
            cands = [lo + (hi - lo) * i / 20 for i in range(21)]
            best[k] = min(cands, key=lambda x: loss({**best, k: x}))
        grid = {k: (max(lo, best[k] - (hi - lo) / 8), min(hi, best[k] + (hi - lo) / 8)) for k, (lo, hi) in grid.items()}
    errs = [(r, u, v, predict(r, u, facts, best)) for r, u, v in pts]
    return best, errs


def held_out(runs, facts):
    """Each bf16 run's points predicted by the constants fitted on the other runs alone: the error to expect where bf16 was
    not run."""
    out = []
    mine = []
    for r, _, _ in points(runs, facts):
        if not any(r is x for x in mine):
            mine.append(r)
    for r0 in mine:
        cal, _ = calibrate([r for r in runs if r is not r0], facts)
        out += [(r0, u, v, predict(r0, u, facts, cal)) for u, v in sorted(r0["speed"].items())]
    return out


def kind(f):
    return "MoE" if f["E"] else "hybrid" if f["state"] else "dense"

# --------------------------------------------------------------------------------------------------------------------------- report


def gib(b):
    return f"{b / GiB:.2f}" if b is not None else "-"


def where(src):
    """A record's directory, or a pod's results directory that is not a record yet."""
    if src.startswith("benchmarks/gpu/"):
        return "`" + "/".join(src.split("/")[:3]) + "`"
    m = re.search(r"(results-[^/]+)", src)
    return f"pod results `{m.group(1)}` (not a record yet)" if m else src


def gname(g, n):
    s = SHORT.get(g, g)
    return f"{n} x {s}" if n and n > 1 else s


def settings(r):
    return ("eager" if r["eager"] else "compiled") + (f", max-len {r['max_model_len']:,}" if r.get("max_model_len") else "") + \
        (f", {r['max_num_seqs']} seqs" if r.get("max_num_seqs") else "") + (f", util {r['util']:.2f}" if r.get("util") else "") + \
        (" (speed1.py)" if r.get("speed1") else "") + \
        (f", glyd {r['version']}" if r["server"] == "glyd" and r.get("version") not in (None, "0.29") else "")


def same(a, b):
    return (a["model"], a["gpu"], a["tp"]) == (b["model"], b["gpu"], b["tp"]) and where(a["src"]) == where(b["src"])


def err_line(errs):
    e = [abs(p / v - 1) * 100 for _, _, v, p in errs if p]
    return (f"mean {sum(e) / len(e):.0f}%, worst {max(e):.0f}% ({len(e)} points)" if e else "no points") + \
        (f", {len(errs) - len(e)} with no estimate" if len(e) < len(errs) else "")


def rng(xs, fmt="{:.1f}%"):
    return (fmt.format(min(xs)) + ("" if min(xs) == max(xs) else " to " + fmt.format(max(xs)))) if xs else "-"


def report(runs, facts, cal, errs, held, title_date):
    L = [f"# Glyd against bf16 on 2026 models: memory, users, tokens/s ({title_date})\n",
         "bf16 is ESTIMATED from Glyd's own measurements (tools/report/report.py), not run for this report; where an earlier run on the"
         " same pod measured bf16, the measured number is shown instead and marked so. Internal: never published.\n"]
    glyd = [r for r in runs if r["server"] == "glyd" and r["model"] in facts and (r.get("version") or "").startswith("0.29")]
    bf = [r for r in runs if r["server"] == "bf16" and r["model"] in facts]
    groups = {}
    for r in glyd:
        if not (r.get("weights") and r.get("kv_mem") or r.get("speed")):
            continue
        groups.setdefault((r["model"], r["gpu"], r["tp"]), []).append(r)
    rows = []
    for (model, gpu, tp), rs in sorted(groups.items()):
        mem = [r for r in rs if r.get("weights") and r.get("kv_mem")]
        sp = [r for r in rs if r.get("speed")]
        m = max(mem, key=lambda r: (bool(r.get("speed")), not r["eager"], r["kv_tokens"] or 0)) if mem else None
        u = max(mem, key=lambda r: (r.get("conc_len") == 8192 and any(same(b, r) and b.get("conc_len") == 8192 for b in bf),
                                    r.get("conc_len") == 8192, r is m)) if mem else None
        s = max(sp, key=lambda r: (len(r["speed"]), r.get("weights") is not None, not r["eager"])) if sp else None
        rows.append((model, gpu, tp, facts[model], m, u, s))
    L.append("## Measured: Glyd (measured) against bf16 (estimated, or measured where marked)\n")
    L.append("Weights: as vLLM loads them per GPU (\"Model loading took\"); bf16 estimated from the checkpoint. Users at 8K: requests of"
             " 8,192 tokens the KV cache holds at once by vLLM's own accounting (a hybrid model's recurrent state counted as vLLM"
             " reserves it): **meas.** = vLLM's logged \"Maximum concurrency\" of a server started at max-len 8,192; **≈** = converted"
             " from a server started at another max-len (by one request's KV cost at each length; not a measurement); bf16 **est.** ="
             " Glyd's users scaled by the KV memory bf16's larger weights leave, at Glyd's settings (glyd serve's util 0.92: about"
             " 0.5 GiB more than vLLM's default 0.90 on a 24 GB GPU, 1.6 GiB on an H100). Tokens/s: output tokens a second of `vllm"
             " bench serve` (about 1,000 tokens in, 256 out) at N users at once; (speed1.py) = one streamed request instead,"
             " O / (TTFT + (O-1) / decode rate). The estimate's error where bf16 was not run is the held-out error below.\n")
    L.append("| Model | GPU | Format / Glyd layout | Weights GiB: Glyd meas. / bf16 est. | Saved GiB (%) | Users at 8K: Glyd / bf16 "
             "| Tok/s 1 user: Glyd meas. / bf16 (Glyd / bf16) | Tok/s 8 users: Glyd meas. / bf16 | Tok/s 32 users: Glyd meas. / bf16 "
             "| Tok/s 64 users: Glyd meas. / bf16 | Settings | Record |")
    L.append("| :-- | :-- | :-- | --: | --: | --: | --: | --: | --: | --: | :-- | :-- |")
    out = []
    for model, gpu, tp, f, m, ur, s in rows:
        rec = {"model": model, "gpu": gpu, "gpus": tp, "format": f["format"], "kind": kind(f)}
        base = m or s
        bw = bf16_weights(f, tp, base.get("lm_only"))
        cells = [model.split("/")[-1], gname(gpu, tp), f"{f['format']} / {(m or {}).get('layout') or '?'}"]
        if m:
            saved = bw - m["weights"]
            cells += [f"{gib(m['weights'])} / {gib(bw)}", f"{gib(saved)} ({100 * saved / bw:.1f}%)"]
            rec.update(layout=m.get("layout"), glyd_weights_gib=m["weights"] / GiB, bf16_weights_gib_est=bw / GiB,
                       saved_gib=saved / GiB, saved_pct=100 * saved / bw, memory_src=m["src"])
            g, direct = users_at(ur, f, 8192)
            bm = sorted((b for b in bf if same(b, ur) and b.get("conc")), key=lambda b: b["conc_len"] != 8192)
            conv = users_at(bm[0], f, 8192)[0] if bm else None
            if bm and bm[0]["conc_len"] == 8192:
                bu, bk = bm[0]["conc"], "meas."
            elif conv is not None:
                bu, bk = conv, f"≈ from bf16 meas. at {bm[0]['conc_len']:,}"
            else:
                bu, bk = users_bf16(ur, f, 8192), "est."
            if g is None:
                cells.append("-")
            else:
                gc = f"{g:.1f} meas." if direct else f"≈{g:.1f}"
                if bu is None:
                    bc = "-"
                elif kv_room_bf16(ur, f) <= 0:
                    bc = "does not fit"
                elif bu < 1:
                    bc = f"{bu:.1f} {bk} (no 8K request fits)"
                else:
                    bc = f"{bu:.1f} {bk}"
                    lu = users_bf16(ur, f, ur["conc_len"]) if bk == "est." and ur.get("conc_len") else None
                    if lu is not None and lu < 1:
                        bc += f" (no start at {ur['conc_len']:,}-token max-len)"
                cells.append(f"{gc} / {bc}")
            rec.update(users_8k={"glyd": g, "glyd_measured": direct, "bf16": bu, "bf16_kind": bk}, users_src=ur["src"])
        else:
            cells += [f"- / {gib(bw)}", "-", "-"]
        if s:
            bms = [b for b in bf if b.get("speed") and same(b, s)]
            P, O = s.get("P", 1024), s.get("O", 256)
            cap = users_bf16(s, f, P + O, running=True) if s.get("kv_mem") else None
            sp = {}
            for u in (1, 8, 32, 64):
                g = s["speed"].get(u)
                meas = [b["speed"][u] for b in bms if u in b["speed"]]
                if g is None:
                    cells.append("-")
                    continue
                if meas:
                    b, k = meas[0], "meas."
                else:
                    b, k = tok_s(f, gpu, u, P, O, cap, s.get("max_num_seqs"), tp, cal), "est."
                sp[u] = (g, b, k)
                cells.append(f"{g:,.0f} / does not fit" if b is None else f"{g:,.0f} / {b:,.0f} {k} ({g / b:.2f}x)")
            rec["tok_s"] = {u: {"glyd": g, "bf16": b, "bf16_kind": k} for u, (g, b, k) in sp.items()}
            rec["speed_src"] = s["src"]
        else:
            cells += ["-"] * 4
        cells += [settings(s or m), "; ".join(sorted({where(x["src"]) for x in (m, ur, s) if x}))]
        L.append("| " + " | ".join(cells) + " |")
        out.append(rec)
    L.append("")
    pairs = pairs_of(runs, bf)
    L += reading(out, [r[6] for r in rows if r[6]], bf, facts, held, pairs)
    L.append("## Projected: models not measured yet (memory only, from the inventory)\n")
    moe_h100 = [r for r in out if r["kind"] == "MoE" and r["gpu"].startswith("H100") and "saved_pct" in r]
    L.append("PROJECTED, not measured: loaded GiB = what vLLM loads in bf16; Glyd = the inventory's projected saving of the checkpoint"
             " with today's plugin (tiered on Ada; 12-bit on the other GPUs; on an A100/H100 a MoE's attention stays bf16) plus"
             " vLLM's buffers." + "".join(f" Caveat: {r['model'].split('/')[-1]}, the MoE measured on an H100, saved {r['saved_pct']:.1f}%"
                                          f" there against {facts[r['model']].get('b12_dc_pct') or 0:.1f}% projected, so the"
                                          " A100/H100 column may be high for MoEs." for r in moe_h100[:1]) + "\n")
    L.append("| Model | Loaded GiB (bf16) | Glyd GiB, tiered (Ada) | saved % | Glyd GiB, 12-bit (5090) | saved % | 12-bit on A100/H100 % |")
    L.append("| :-- | --: | --: | --: | --: | --: | --: |")
    measured = {r[0] for r in rows}
    for model, f in sorted(facts.items()):
        if model in measured:
            continue
        lo = f["loaded"]
        L.append(f"| {model} | {gib(lo + BUFFERS)} | {gib(lo * (1 - f['tiered_pct'] / 100) + BUFFERS)} | {f['tiered_pct']:.1f}% | "
                 f"{gib(lo * (1 - f['b12_pct'] / 100) + BUFFERS)} | {f['b12_pct']:.1f}% | {f.get('b12_dc_pct') or f['b12_pct']:.1f}% |")
    L.append("")
    L += accuracy(bf, glyd, facts, cal, errs, held, pairs)
    return "\n".join(L) + "\n", out


def pairs_of(runs, bf):
    """Each bf16 run with a logged concurrency against the one Glyd 0.29 run on its pod (same util; the same max-len first):
    (Glyd, bf16, same settings)."""
    pairs = []
    for b in bf:
        if not b.get("kv_mem") or not b.get("conc"):
            continue
        gs = [g for g in runs if g["server"] == "glyd" and (g.get("version") or "").startswith("0.29") and g.get("kv_mem") and g.get("weights") and same(g, b)
              and g["util"] == b["util"]]
        if gs:
            g = max(gs, key=lambda g: (g["version"], g.get("max_model_len") == b.get("max_model_len"), g["kv_mem"]))
            pairs.append((g, b, all(g.get(k) == b.get(k) for k in ("max_model_len", "max_num_seqs", "eager", "lm_only"))))
    return pairs


def pair_err(g, b, facts):
    e = users_bf16(g, facts[g["model"]], b["conc_len"])
    return e, (e / b["conc"] - 1) * 100 if e is not None else None


def reading(out, speeds, bf, facts, held, pairs):
    """The summary under the table, from the rows."""
    L = ["## Reading the table\n"]
    by = {}
    for r in out:
        if "saved_pct" in r:
            by.setdefault(r.get("layout") or "?", []).append(r)
    allp = [r["saved_pct"] for rs in by.values() for r in rs]
    L.append(f"- **Memory**: Glyd saves {rng(allp)} of the weights on the {len(allp)} measured rows: " + "; ".join(
        f"{lay} layout {rng([r['saved_pct'] for r in rs])} ({', '.join(sorted({r['model'].split('/')[-1] + ' on ' + gname(r['gpu'], r['gpus']) for r in rs}))})"
        for lay, rs in sorted(by.items())) + ".")
    ups = [r for r in out if r.get("users_8k", {}).get("glyd") and r["users_8k"].get("bf16")]
    if ups:
        L.append("- **Users at 8K**: Glyd holds " + "; ".join(
            f"{r['users_8k']['glyd']:.1f} against {r['users_8k']['bf16']:.1f} {r['users_8k']['bf16_kind']} ({r['model'].split('/')[-1]},"
            f" {gname(r['gpu'], r['gpus'])})" for r in ups) + ". Rows marked ≈ are converted from another max-len, not measured.")
    for ok, what in ((True, "the same settings"), (False, "other settings (max-len, seqs, eager)")):
        e = [pair_err(g, b, facts)[1] for g, b, s in pairs if s == ok and pair_err(g, b, facts)[1] is not None]
        if e:
            L.append(f"- **bf16 users estimate against bf16 measured**, Glyd and bf16 at {what}: {rng(e, '{:+.1f}%')} ({len(e)} pairs)." +
                     ("" if ok else " A negative error means Glyd's run kept more memory outside the KV cache than bf16's did,"
                      " so bf16 est. users at such settings are low and Glyd's lead in users high by as much."))
    meas = []
    for g in speeds:
        for b in bf:
            if g.get("speed") and b.get("speed") and same(g, b):
                us = sorted(set(g["speed"]) & set(b["speed"]))
                if us:
                    meas.append((g, b, us))
    for g, b, us in meas:
        rat = [(u, g["speed"][u] / b["speed"][u]) for u in us]
        lo = min(rat, key=lambda x: x[1])
        slow = [u for u in us if (g.get("ttft") or {}).get(u) and (b.get("ttft") or {}).get(u) and g["ttft"][u] > b["ttft"][u]]
        L.append(f"- **Tokens/s, bf16 measured on the same pod** ({g['model'].split('/')[-1]}, {gname(g['gpu'], g['tp'])}, {where(g['src'])}):"
                 f" Glyd / bf16 {rng([x for _, x in rat], '{:.2f}x')} over {len(us)} loads ({', '.join(map(str, us))} users; lowest at"
                 f" {lo[0]})" + (f"; Glyd's time to first token is slower at {', '.join(map(str, slow))} users" if slow else "") + ".")
    if held:
        e = [abs(p / v - 1) * 100 for _, _, v, p in held if p]
        L.append(f"- **Tokens/s against the estimate**: the estimate's held-out error is mean {sum(e) / len(e):.0f}%, worst {max(e):.0f}%"
                 " (by GPU and kind below), so a Glyd / bf16 est. ratio within that band of 1.00x is not a difference. No speed claim"
                 " rests on an estimate.")
    L.append("")
    return L


def accuracy(bf, glyd, facts, cal, errs, held, pairs):
    L = ["## How good is the bf16 estimate\n"]
    L.append(f"Constants fitted on {len(errs)} bf16 points measured on 2026 models ({len({id(r) for r, *_ in errs})} runs): bandwidth"
             f" efficiency {cal['eb']:.2f}, decode saturation {cal['ef']:.2f} (an empirical term, not a compute crossover), prompt compute"
             f" {cal['ep']:.2f}, a step's fixed time {cal['c0'] * 1e3:.2f} ms. Recurrent state is taken as float32, read and written each"
             " step (not confirmed from vLLM 0.30).\n")
    if errs:
        L.append(f"- In-sample (fitted and checked on the same points): {err_line(errs)}.")
    if held:
        L.append(f"- **Held-out** (each run predicted by constants fitted without it; the error band to use): {err_line(held)}.")
        for name, key in (("GPU", lambda r: gname(r["gpu"], r["tp"])), ("kind", lambda r: kind(facts[r["model"]]))):
            grp = {}
            for x in held:
                grp.setdefault(key(x[0]), []).append(x)
            L.append(f"  - by {name}: " + "; ".join(
                f"{k} {err_line(v)} from {len({id(r) for r, *_ in v})} run(s)" for k, v in sorted(grp.items())) + ".")
        have = {r["gpu"] for r, *_ in held}
        mgpus = sorted({gname(r["gpu"], r["tp"]) for r in glyd if r.get("speed") and r["gpu"] not in have})
        moe = len({id(r) for r, *_ in held if kind(facts[r["model"]]) == "MoE"})
        L.append(f"  - Extrapolated: no bf16 speed was measured on {', '.join(mgpus) or 'none'}; MoE estimates rest on {moe} MoE run(s).")
    if errs:
        L.append("\nEvery bf16 point, with Glyd where the same pod measured it (TTFT p50 ms, Glyd / bf16):\n")
        L.append("| Model | GPU | Users | bf16 meas. tok/s | bf16 est. in-sample | held-out | Glyd meas. | Glyd / bf16 meas. | TTFT ms Glyd / bf16 | Source |")
        L.append("| :-- | :-- | --: | --: | --: | --: | --: | --: | --: | :-- |")
        ho = {(id(r), u): p for r, u, _, p in held}
        for r, u, v, p in errs:
            g = [x for x in sorted(glyd, key=lambda x: x.get("eager", False)) if x.get("speed") and u in x["speed"] and same(x, r)]
            h = ho.get((id(r), u))
            gt, bt = (g[0].get("ttft") or {}).get(u) if g else None, (r.get("ttft") or {}).get(u)
            L.append(f"| {r['model'].split('/')[-1]} | {gname(r['gpu'], r['tp'])} | {u} | {v:,.0f} | " +
                     (f"{p:,.0f} ({(p / v - 1) * 100:+.0f}%)" if p else "-") + " | " + (f"{h:,.0f} ({(h / v - 1) * 100:+.0f}%)" if h else "-") +
                     " | " + (f"{g[0]['speed'][u]:,.0f} | {g[0]['speed'][u] / v:.2f}x" if g else "- | -") + " | " +
                     (f"{gt:,.0f} / {bt:,.0f}" if gt and bt else "-") + f" | {where(r['src'])} |")
    if pairs:
        L.append("\nbf16 users at the bf16 run's own max-len (vLLM's logged \"Maximum concurrency\"), estimated from the same pod's Glyd run;"
                 " settings that differ between the two are listed (other max-len: converted by one request's KV cost; eager or"
                 " not, seqs: other memory beside the KV cache):\n")
        L.append("| Model | GPU | Glyd run (version, settings) | bf16 settings | bf16 meas. users | bf16 est. | error | Source |")
        L.append("| :-- | :-- | :-- | :-- | --: | --: | --: | :-- |")
        for g, b, _ in pairs:
            e, err = pair_err(g, b, facts)
            L.append(f"| {g['model'].split('/')[-1]} | {gname(g['gpu'], g['tp'])} | {g.get('version') or '?'}, {settings(g)} | {settings(b)} | "
                     f"{b['conc']:.2f} at {b['conc_len']:,} | " + (f"{e:.2f} | {err:+.1f}%" if e is not None else "- | -") +
                     f" | {where(g['src'])} |")
        spread = {}
        for b in bf:
            if b.get("kv_mem"):
                spread.setdefault((b["model"], b["gpu"], b["tp"], where(b["src"]), settings(b)), []).append(b["kv_mem"])
        noise = [(k, v) for k, v in spread.items() if len(v) > 1]
        for k, v in noise:
            L.append(f"\nNoise floor: {len(v)} bf16 servers of {k[0].split('/')[-1]} on {gname(k[1], k[2])} at identical settings ({k[4]},"
                     f" {k[3]}) had {', '.join(gib(x) for x in v)} GiB of KV memory: {(max(v) / min(v) - 1) * 100:.0f}% apart from run to run.")
    mem = [(r, bf16_weights(facts[r["model"]], r["tp"], r.get("lm_only"))) for r in bf if r.get("weights")]
    if mem:
        L.append("\nbf16 weights, estimated against measured (vLLM's \"Model loading took\"):\n")
        L.append("| Model | GPU | measured GiB | estimated GiB | error | Source |")
        L.append("| :-- | :-- | --: | --: | --: | :-- |")
        done = set()
        for r, e in mem:
            k = (r["model"], r["gpu"], r["tp"], r["weights"], r.get("lm_only"))
            if k in done:
                continue
            done.add(k)
            L.append(f"| {r['model'].split('/')[-1]} | {gname(r['gpu'], r['tp'])} | {gib(r['weights'])} | {gib(e)} | "
                     f"{(e / r['weights'] - 1) * 100:+.1f}% | {where(r['src'])} |")
    return L


def selftest():
    f = {"loaded": 18.83e9, "tower": 0.9e9, "dense": 17.0e9, "experts": 0, "E": 0, "k": 0, "params": 8.5e9, "kv_tok": 32768,
         "state": 0, "format": "bf16"}
    run = {"weights": 12.68 * GiB, "kv_mem": 10 * GiB, "kv_tokens": 100000, "max_model_len": 10000, "tp": 1}
    room = kv_room_bf16(run, f)
    assert abs(room / GiB - (10 - (18.83e9 + BUFFERS - 12.68 * GiB) / GiB)) < 1e-9
    assert abs(users_bf16(run, f, 10000) - 10 * room / run["kv_mem"]) < 1e-9 and users_at(run, f, 5000)[0] == 20
    assert users_bf16({**run, "kv_mem": 1 * GiB}, f, 10000) == 0  # bf16 does not fit
    # a hybrid (Qwen3.8-27B on an H100, as logged): 1.77x at 262,144 tokens, 784-token blocks, 6 state blocks -> 35.5 at 8,192
    h = {**f, "state": 1, "kv_cfg": {"layer_types": ["linear_attention"] * 48 + ["full_attention"] * 16}}
    hr = {"conc": 1.77, "conc_len": 262144, "block": 784}
    assert state_blocks(h) == 6 and abs(users_at(hr, h, 8192)[0] - 1.77 * 341 / 17) < 1e-9
    assert users_at({**hr, "conc_len": 8192}, h, 8192) == (1.77, True) and users_at({"conc": 1, "conc_len": 8192}, h, 4096) == (None, False)
    one = tok_s(f, "RTX 4090", 1, 1000, 256, None, 256)
    many = tok_s(f, "RTX 4090", 64, 1000, 256, None, 256)
    assert 0 < one < many and abs(tok_s(f, "RTX 4090", 64, 1000, 256, 8.5, 256) - tok_s(f, "RTX 4090", 8, 1000, 256, None, 256)) < 1e-9
    assert tok_s(f, "RTX 4090", 1, 1000, 256, 0.5, 256) is None
    log = ("x non-default args: {'model': 'Qwen/Qwen3.5-9B', 'max_model_len': 8192, 'quantization': 'glyd', 'enforce_eager': True}\n"
           "gpu-memory-utilization=0.9200\nglyd: mma12 layout\nModel loading took 13.89 GiB memory\nAvailable KV cache memory: 9.5 GiB\n"
           "GPU KV cache size: 389,307 tokens\nMaximum concurrency for 8,192 tokens per request: 47.52x\n"
           "y non-default args: {'model': '/root/hf/hub/models--Qwen--Qwen3.5-9B/snapshots/abc', 'tensor_parallel_size': 4}\n"
           "Model loading took 17.66 GiB memory\n")
    a, b = parse_log(log)
    assert (a["server"], a["layout"], a["kv_tokens"], a["max_model_len"], a["eager"], a["util"]) == ("glyd", "12-bit", 389307, 8192, True, 0.92)
    assert (a["conc"], a["conc_len"]) == (47.52, 8192)
    assert (b["model"], b["server"], b["tp"], b["weights"], b["kv_tokens"], b["util"]) == ("Qwen/Qwen3.5-9B", "bf16", 4, 17.66 * GiB, None, 0.9)
    assert version_of("/x/results-rpq38/serve/v028/serve-glyd.txt", "/x") == "0.28"
    print("selftest ok")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", nargs="*", default=[], help="run directories (records first: a copy found later is skipped)")
    ap.add_argument("--headers", help="checkpoint headers (research/inventory-2026's fetch): rebuild facts.json from them")
    ap.add_argument("--out", help="the report (Markdown)")
    ap.add_argument("--json", help="the reader's table and the report's numbers")
    ap.add_argument("--date", default="2026-10-06")
    ap.add_argument("--notes", help="a Markdown file appended to the report")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    facts = build_facts(a.headers) if a.headers else json.load(open(FACTS))
    runs = read_runs(a.runs)
    for name in sorted({r["model"] for r in runs if r["model"] not in facts}):
        print(f"skipped (not a 2026 model in facts.json): {name}", file=sys.stderr)
    runs = [r for r in runs if r["model"] in facts]  # 2026 models only
    for r in runs:
        if r["server"] == "glyd" and not (r.get("version") or "").startswith("0.29"):
            print(f"kept out of the main table (glyd {r['version']}): {r['model']} {r['src']}", file=sys.stderr)
    cal, errs = calibrate(runs, facts)
    held = held_out(runs, facts)
    text, rows = report(runs, facts, cal, errs, held, a.date)
    if a.notes:
        text += "\n" + open(a.notes).read()
    if a.out:
        open(a.out, "w").write(text)
    else:
        print(text)
    if a.json:
        ho = {(id(r), u): p for r, u, _, p in held}
        json.dump({"calibration": cal, "rows": rows, "runs": runs,
                   "bf16_points": [{"model": r["model"], "gpu": r["gpu"], "users": u, "measured": v, "estimated_in_sample": p,
                                    "estimated_held_out": h, "src": r["src"]}
                                   for (r, u, v, p), h in ((e, ho.get((id(e[0]), e[1]))) for e in errs)]}, open(a.json, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
