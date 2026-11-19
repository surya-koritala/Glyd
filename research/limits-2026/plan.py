"""What measure.py reads: for each checkpoint its census (every tensor's bytes by format, part and kind, from the headers alone) and
its samples (seed 0), each with the matching tensor it is compared with, into PLAN_DIR/<org>__<name>.json with the header entries
those reads need. Headers only (hdr.py); no weights, no numpy.

    python plan.py HEADERS_DIR PLAN_DIR

Samples: whole stored rows, about 12 MB a sample, over the (format, kind) groups by bytes (every kind holding 0.1% of the
checkpoint's bytes gets one), a tensor of a group drawn by its bytes. The matching tensor of a sample: for an expert, the
neighbouring expert of the same layer; otherwise the same tensor of the nearest earlier layer that has it (same shape, dtype and
format); the same rows of it.
"""
import collections, json, os, random, re, sys

D = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(D, "..", "quant-2026"))
import quant  # research/quant-2026 (tensors, classify, census, category; no numpy needed for these)
import gguf

BF16 = ["Qwen/Qwen3.5-9B", "Qwen/Qwen3.5-27B", "Qwen/Qwen3.8-27B", "google/gemma-4-12B-it", "google/gemma-4-31B-it",
        "ibm-granite/granite-4.2-8b", "meta-models/Muse-Glimmer-30B", "zai-org/GLM-5.3-BF16",
        "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16", "MiniMaxAI/MiniMax-M3"]
QUANT = ["Qwen/Qwen3.8-27B-FP8", "Qwen/Qwen3.6-35B-A3B-FP8", "mistralai/Mistral-Small-4-119B-2603",
         "RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic", "MiniMaxAI/MiniMax-M3-MXFP8", "nvidia/Qwen3.8-27B-NVFP4",
         "nvidia/Gemma-4-31B-IT-NVFP4", "nvidia/Qwen3.6-35B-A3B-NVFP4", "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4",
         "moonshotai/Kimi-K3", "amd/MiniMax-M3-MXFP4", "deepseek-ai/DeepSeek-V4.1-Flash", "Qwen/Qwen3.5-35B-A3B-GPTQ-Int4",
         "QuantTrio/Qwen3.6-35B-A3B-AWQ", "RedHatAI/Qwen3.8-27B-INT4", "cyankiwi/Qwen3.6-27B-AWQ-INT4",
         "google/gemma-4-12B-it-qat-w4a16-ct", "RedHatAI/Qwen3.5-9B-quantized.w8a8"]
GGUF = ["unsloth/Qwen3.5-9B-GGUF:Qwen3.5-9B-Q4_K_M.gguf", "unsloth/Qwen3.5-9B-GGUF:Qwen3.5-9B-Q8_0.gguf"]
N_BF16, N_QUANT, N_QBF16, N_GGUF = 10, 9, 1, 10  # 10 samples a checkpoint (a quantized one: 9 quantized + 1 of its 16-bit parts)
MIN_SHARE = 0.001
LAYER = re.compile(r"(layers\.|blk\.|\.h\.|blocks?\.)(\d+)\.")
EXPERT = re.compile(r"(experts?\.)(\d+)\.")


def spread(sizes, quota):
    """(group -> bytes) -> (group -> samples): one for every group with MIN_SHARE of the bytes, the rest by bytes."""
    tot = sum(sizes.values()) or 1
    order = sorted(sizes, key=lambda g: -sizes[g])
    take = {g: 1 for g in order if sizes[g] / tot >= MIN_SHARE}
    if len(take) > quota:
        take = {g: 1 for g in order[:quota]}
    while sum(take.values()) < quota:
        g = max(order, key=lambda g: sizes[g] / tot * quota - take.get(g, 0))
        take[g] = take.get(g, 0) + 1
    return order, take


def pick(sizes, rnd):
    """An index drawn with probability by size: a sample stands for bytes, not for tensor names."""
    x = rnd.random() * sum(sizes)
    for i, b in enumerate(sizes):
        x -= b
        if x < 0:
            return i
    return len(sizes) - 1


def interleave(picks):
    """Samples by group, biggest group first -> one list taking each group's first, then each group's second, ..."""
    out, i = [], 0
    while any(len(p) > i for p in picks):
        out += [p[i] for p in picks if len(p) > i]
        i += 1
    return out


def rows_of(R, per_row, rnd, align):
    n = max(align, quant.TARGET // per_row // align * align)
    n = min(n, R // align * align) if R >= align else R
    r0 = rnd.randrange(0, R - n + 1) // align * align if R > n else 0
    return r0, r0 + n


def matching_st(name, ts, cls, e):
    t = ts[name]
    if e is not None and t["shape"][0] > 1:
        return [name, e - 1 if e > 0 else 1]
    m = EXPERT.search(name)
    cands = []
    if m:
        i = int(m.group(2))
        cands = [name[:m.start(2)] + str(j) + name[m.end(2):] for j in (i - 1, i + 1)]
    else:
        m = LAYER.search(name)
        if m:
            L = int(m.group(2))
            cands = [name[:m.start(2)] + str(j) + name[m.end(2):] for j in range(L - 1, max(-1, L - 9), -1)] + \
                    [name[:m.start(2)] + str(L + 1) + name[m.end(2):]]
    for c in cands:
        if c in ts and ts[c]["shape"] == t["shape"] and ts[c]["dtype"] == t["dtype"] and cls[c][0] == cls[name][0]:
            return [c, e]
    return None


def plan_st(hdr, quota, fmts, rnd, quota_bf16=0):
    ts = quant.tensors(hdr)
    qm = quant.qmethod(hdr["config"])
    cls = quant.classify(ts, qm)
    out = []
    for fs, q in ((fmts, quota), (("bf16",), quota_bf16)):
        if not q:
            continue
        groups = collections.defaultdict(list)
        for k, (f, p, w) in sorted(cls.items()):
            if p in ("codes", "bf16") and f in fs and len(ts[k]["shape"]) in (2, 3):
                groups[(f, quant.category(k))].append(k)
        sizes = {g: sum(ts[k]["end"] - ts[k]["start"] for k in v) for g, v in groups.items()}
        order, take = spread(sizes, q)
        picks = []
        for g in order:
            names, mine = list(groups[g]), []
            for _ in range(min(take.get(g, 0), len(names))):
                name = names.pop(pick([ts[k]["end"] - ts[k]["start"] for k in names], rnd))
                t = ts[name]
                R, C, E = quant.stored_rows(t)
                r0, r1 = rows_of(R, C * quant.DT[t["dtype"]], rnd, quant.ROWS)
                e = rnd.randrange(E) if E else None
                mine.append({"fmt": g[0], "cat": g[1], "name": name, "r0": r0, "r1": r1, "e": e, "prev": matching_st(name, ts, cls, e)})
            picks.append(mine)
        out += interleave(picks)
    stems = {quant.stem(s["name"])[0] for s in out} | {quant.stem(s["prev"][0])[0] for s in out if s["prev"]}
    keep = {k for k in ts if quant.stem(k)[0] in stems}
    files = {}
    for fn, f in hdr["files"].items():
        h = {k: v for k, v in f["header"].items() if k in keep}
        if h:
            files[fn] = {"hlen": f["hlen"], "header": h}
    cfg = hdr["config"]
    small = {"quantization_config": cfg.get("quantization_config"),
             "text_config": {"quantization_config": (cfg.get("text_config") or {}).get("quantization_config")}}
    cen = quant.census(ts, cls)
    return {"kind": "st", "qm": qm, "census": cen, "stored": sum(v["bytes"] for v in cen.values()),
            "hdr": {"config": small, "files": files}, "samples": out}


def gguf_cat(n):
    if n.startswith("token_embd"):
        return "embed"
    if n.startswith("output."):
        return "lm_head"
    if "_exps" in n:
        return "expert"
    if any(a in n for a in ("attn", "ssm")):
        return "attn"
    if "ffn" in n:
        return "mlp"
    return "other"


def plan_gguf(h, quota, rnd):
    QT = {"Q4_K", "Q5_K", "Q6_K", "Q8_0"}
    tens = {n: (dims, t, off) for n, dims, t, off in h["tensors"]}
    cen = collections.Counter()
    cnt = collections.Counter()
    groups = collections.defaultdict(list)
    for n, (dims, t, off) in sorted(tens.items()):
        tn = gguf.TYPES[t][0]
        b = gguf.nbytes(dims, t)
        key = f"{tn}:{'blocks' if tn in QT else 'other'}:{gguf_cat(n)}"
        cen[key] += b
        cnt[key] += 1
        if tn in QT and len(dims) == 2:
            groups[(tn, gguf_cat(n))].append(n)
    sizes = {g: sum(gguf.nbytes(tens[n][0], tens[n][1]) for n in v) for g, v in groups.items()}
    order, take = spread(sizes, quota)
    picks = []
    for g in order:
        names, mine = list(groups[g]), []
        for _ in range(min(take.get(g, 0), len(names))):
            name = names.pop(pick([gguf.nbytes(tens[k][0], tens[k][1]) for k in names], rnd))
            (cols, R), t, off = tens[name]
            rb = gguf.nbytes([cols], t)
            r0, r1 = rows_of(R, rb, rnd, 64)
            prev = None
            m = LAYER.search(name)
            if m:
                L = int(m.group(2))
                for j in list(range(L - 1, max(-1, L - 9), -1)) + [L + 1]:
                    c = name[:m.start(2)] + str(j) + name[m.end(2):]
                    if c in tens and tens[c][0] == tens[name][0] and tens[c][1] == t:
                        prev = {"name": c, "start": h["data_start"] + tens[c][2] + r0 * rb}
                        break
            mine.append({"fmt": g[0], "cat": g[1], "name": name, "type": t, "cols": cols, "r0": r0, "r1": r1, "row_bytes": rb,
                         "start": h["data_start"] + off + r0 * rb, "prev": prev})
        picks.append(mine)
    census = {k: {"tensors": cnt[k], "bytes": v} for k, v in sorted(cen.items())}
    return {"kind": "gguf", "file": h["file"], "census": census, "stored": sum(cen.values()), "samples": interleave(picks)}


def main():
    hd, pd = sys.argv[1], sys.argv[2]
    os.makedirs(pd, exist_ok=True)
    for spec in BF16 + QUANT + GGUF:
        repo, _, fn = spec.partition(":")
        tag = repo.replace("/", "__") + (("__" + fn) if fn else "")
        h = json.load(open(os.path.join(hd, tag + ".json")))
        rnd = random.Random(0)
        if fn:
            p = plan_gguf(h, N_GGUF, rnd)
        elif spec in BF16:
            p = plan_st(h, N_BF16, ("bf16",), rnd)
        else:
            p = plan_st(h, N_QUANT, quant.QUANT, rnd, N_QBF16)
        for i, s in enumerate(p["samples"]):
            s["id"] = i
        p["repo"], p["spec"], p["tag"] = repo, spec, tag
        json.dump(p, open(os.path.join(pd, tag + ".json"), "w"))
        cats = collections.Counter((s["fmt"], s["cat"]) for s in p["samples"])
        print(f"{spec}: {p['stored'] / 1e9:.1f} GB, {len(p['samples'])} samples, {sum(1 for s in p['samples'] if s['prev'])} with a match; "
              f"{dict(cats)}", flush=True)


if __name__ == "__main__":
    main()
