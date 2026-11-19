"""The README's tables from measure.py's records (JSON only, no numpy):

    python report.py OUT_DIR RESULTS_DIR        (OUT_DIR: phase1/*.json and phase2.jsonl from the pod)

Writes RESULTS_DIR/<tag>.json (a checkpoint: census, every sample with phase 2 merged in, the whole-checkpoint projection),
RESULTS_DIR/summary.json and RESULTS_DIR/tables.md.

Bits a weight of a sample: every part (codes, scales, zero points, block minimums) at the bits each estimate gives it, over the
sample's weights. Two kinds of number, kept apart: the estimates H0 (order 0, a table a tensor, plug-in with the Miller-Madow
correction) and static (the best single context, plug-in, tables free: a lower bound for a coder with that context), and the code
lengths cm (the context-mixing model's spend over the first cm_values of the part, learned as it goes, nothing sent ahead: what a
coder with that model would reach; the better of two mixer rates where both ran; None where no CM ran), fixed
(research/quant-2026's fixed-rate layout, computed from the sample's histogram, not built; for FP8 codes the format as built,
round trip exact) and built (the layout estimate a tensor: the tiered code over bf16's exponent without its tile heads, fp8ref for
FP8 codes, the rest as stored; capped at the stored width). The measured as-built numbers (tiered 10.72-10.81, 12-bit 12.04,
fp8ref 6.72 on GPUs) are in the README, not here. A whole checkpoint: each (format, part, kind) of its census at the ratio its own
samples of that format and part measured (bf16 by kind where that kind was sampled); a part with no sample as stored. Samples pool
with equal weight (10 a checkpoint whatever its size); whole checkpoints are byte-weighted.
"""
import collections, glob, json, math, os, sys

EST = ("stored", "H0", "static", "cm", "fixed", "built")
LABEL = {
    "Qwen/Qwen3.8-27B-FP8": "FP8, 128x128 block scales", "Qwen/Qwen3.6-35B-A3B-FP8": "FP8, 128x128 block scales",
    "mistralai/Mistral-Small-4-119B-2603": "FP8, a scale a tensor", "RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic": "FP8, a scale a row",
    "MiniMaxAI/MiniMax-M3-MXFP8": "MXFP8", "Qwen/Qwen3.5-35B-A3B-GPTQ-Int4": "INT4 GPTQ", "QuantTrio/Qwen3.6-35B-A3B-AWQ": "INT4 AWQ",
    "cyankiwi/Qwen3.6-27B-AWQ-INT4": "INT4 AWQ", "RedHatAI/Qwen3.8-27B-INT4": "INT4 compressed-tensors",
    "google/gemma-4-12B-it-qat-w4a16-ct": "INT4 compressed-tensors", "RedHatAI/Qwen3.5-9B-quantized.w8a8": "INT8 W8A8",
    "moonshotai/Kimi-K3": "MXFP4", "amd/MiniMax-M3-MXFP4": "MXFP4", "deepseek-ai/DeepSeek-V4.1-Flash": "MXFP4",
    "MiniMaxAI/MiniMax-M3": "bf16 holding FP8 values",  # its attention, MLP and experts: about 6.5 bits order-0 (FP8 upcast)
}
ORDER = ["bf16", "bf16 holding FP8 values", "FP8, 128x128 block scales", "FP8, a scale a tensor", "FP8, a scale a row", "MXFP8", "NVFP4", "MXFP4", "INT4 GPTQ",
         "INT4 AWQ", "INT4 compressed-tensors", "INT8 W8A8", "GGUF Q4_K_M", "GGUF Q8_0", "FP8 parts of NVFP4 and MXFP4 checkpoints",
         "16-bit parts of quantized checkpoints"]
QPARTS = "16-bit parts of quantized checkpoints"
FP8PARTS = "FP8 parts of NVFP4 and MXFP4 checkpoints"  # the FP8 attention, MLP and embeddings sampled inside FP4 checkpoints
POSITION = ("row", "column", "tile 64x128", "block scale", "zero point")
NEIGHBOUR = ("left", "above", "left and above")
COMPS = ("xz -9e", "zstd -22 --long", "zpaq -m5")
FIELDS = {"bf16": [("exponent", 8), ("sign", 1), ("mantissa", 7)], "f16": [("exponent", 5), ("sign", 1), ("mantissa", 10)],
          "e4m3": [("exponent", 4), ("sign", 1), ("mantissa", 3)], "e2m1": [("magnitude", 3), ("sign", 1)]}


def label(r):
    if r["kind"] == "gguf":
        return "GGUF " + r["spec"].rsplit("-", 1)[-1].replace(".gguf", "")
    if r["repo"] in LABEL:
        return LABEL[r["repo"]]
    return "NVFP4" if "NVFP4" in r["repo"] else "bf16"


def main_fmt(lab):
    if lab.startswith("FP8") or lab == "MXFP8":
        return "fp8"
    if lab.startswith("INT4"):
        return "int4"
    if lab.startswith("GGUF"):
        return None  # every block type of the file
    if lab.startswith("bf16"):
        return "bf16"
    return {"NVFP4": "nvfp4", "MXFP4": "mxfp4", "INT8 W8A8": "int8"}[lab]


def main_part(s):
    return "weights" if "weights" in s.get("parts", {}) else "codes"


def part_bits(p, est, name):
    """Bits a value of a part under an estimate."""
    q = p.get("quant", {})
    width = p.get("width", 16)
    if est == "stored":
        return width
    static = p.get("best_static", q.get("ceiling", width))
    if est == "H0":
        return p.get("H0", q.get("H0", width))
    if est == "static":
        return static
    cms = [v for v in (p.get("cm"), p.get("cm_lr2")) if v is not None]
    if est == "cm":
        return min(cms) if cms else None  # None: no CM ran on this part (a side stream below 64 values, or a partial phase 2)
    if est == "fixed":
        return min(q.get("fixed", width), width)
    if est == "built":
        if (p.get("enc") in ("bf16", "f16") and name == "weights") or (p.get("enc") == "e4m3" and name == "codes"):
            return min(q.get("fixed", width), width)  # the tiered code over the exponent / fp8ref; never above the stored width
        return width
    raise KeyError(est)


def part_n(p):
    return p.get("n") or p.get("quant", {}).get("n") or 0


def sample_bits(s, est):
    """Bits a weight of a sample under an estimate; for 'stored' the bytes read."""
    if est == "stored":
        return 8 * s["stored_bytes"] / s["n"]
    bits = [(part_n(p), part_bits(p, est, name)) for name, p in s["parts"].items()]
    if any(b is None for n, b in bits if n):
        return None
    tot = sum(n * b for n, b in bits)
    other = 8 * s["stored_bytes"] - sum(part_n(p) * p.get("width", 16) for p in s["parts"].values())
    return (tot + max(0, other)) / s["n"]  # bytes read that are no part measured (a per-tensor scale) as stored


def merge(od):
    recs = {}
    for f in glob.glob(os.path.join(od, "phase1", "*.json")):
        r = json.load(open(f))
        recs[r["tag"]] = r
    p2 = collections.defaultdict(list)
    p = os.path.join(od, "phase2.jsonl")
    if os.path.exists(p):
        for line in open(p):
            x = json.loads(line)
            p2[(x["tag"], x["id"])].append(x)
    for r in recs.values():
        for s in r["samples"]:
            for x in p2.get((r["tag"], s["id"]), []):
                if "error" in x:
                    s.setdefault("phase2_errors", []).append({x["task"]: x["error"]})
                    continue
                if "comp" in x:
                    s["comp"] = x["comp"]
                for name, v in x.get("parts", {}).items():
                    s.setdefault("parts", {}).setdefault(name, {}).update(v)
            z = s.get("parts", {}).pop("zeros", None)  # records before the rename: the zero points' layout under a second name
            if z and "zero points" in s["parts"]:
                s["parts"]["zero points"].setdefault("quant", z.get("quant", {}))
    return recs


def wavg(pairs):
    pairs = [(n, b) for n, b in pairs if b is not None and n]
    return sum(n * b for n, b in pairs) / sum(n for n, _ in pairs) if pairs else None


def pooled(samples, est):
    return wavg([(s["n"], sample_bits(s, est)) for s in samples])


def part_ratio(samples, part, est):
    """A part's bits over its stored bits, pooled over samples (by values)."""
    num = den = 0.0
    for s in samples:
        p = s.get("parts", {}).get(part)
        if not p or not part_n(p) or part_bits(p, est, part) is None:
            continue
        num += part_n(p) * part_bits(p, est, part)
        den += part_n(p) * p.get("width", 16)
    return num / den if den else None


def project(r, bf16_pool):
    """The whole checkpoint under each estimate: bytes, from the census and the samples' ratios; and how many samples each
    (format, kind) group's projection rests on."""
    ok = [s for s in r["samples"] if "error" not in s]
    out = {"stored": r["stored"], "samples_by_group": dict(collections.Counter(f"{s['fmt']} {s['cat']}" for s in ok))}
    for est in EST:
        size = 0.0
        for key, v in r["census"].items():
            f, part, cat = key.split(":")
            ratio = None
            if r["kind"] == "gguf":
                ss = [s for s in ok if s["fmt"] == f]
                if part == "blocks" and ss:
                    ratio = pooled(ss, est) / pooled(ss, "stored")
            elif f == "bf16":
                ss = [s for s in ok if s["fmt"] == "bf16" and s["cat"] == cat] or [s for s in ok if s["fmt"] == "bf16"] or bf16_pool
                ratio = part_ratio(ss, "weights", est)
            elif part in ("codes", "scales", "zeros"):
                ss = [s for s in ok if s["fmt"] == f]
                ratio = part_ratio(ss, {"codes": "codes", "scales": "scales", "zeros": "zero points"}[part], est)
            size += v["bytes"] * (ratio if ratio is not None else 1.0)
        out[est] = {"bytes": size, "saved": 1 - size / r["stored"]}
    return out


def f2(x):
    return "-" if x is None else f"{x:.2f}"


def f3(x):
    return "-" if x is None else f"{x:.3f}"


def pct(x):
    return "-" if x is None else f"{100 * x:.1f}%"


def gb(x):
    return f"{x / 1e9:,.1f}"


def main():
    od, rd = sys.argv[1], sys.argv[2]
    os.makedirs(rd, exist_ok=True)
    recs = merge(od)
    for r in recs.values():
        r["label"] = label(r)
    bf16_pool = [s for r in recs.values() if r["label"] == "bf16" for s in r["samples"] if "error" not in s]
    by = collections.defaultdict(list)
    for r in recs.values():
        r["projection"] = project(r, bf16_pool)
        by[r["label"]].append(r)
        json.dump(r, open(os.path.join(rd, r["tag"] + ".json"), "w"), indent=1)
    md, summary = [], {}

    def samples_of(lab):
        if lab == QPARTS:
            rs = [r for r in recs.values() if not r["label"].startswith("bf16") and r["kind"] == "st"]
            return rs, [s for r in rs for s in r["samples"] if s["fmt"] == "bf16" and "error" not in s]
        if lab == FP8PARTS:
            rs = [r for r in recs.values() if r["label"] in ("NVFP4", "MXFP4")]
            return rs, [s for r in rs for s in r["samples"] if s["fmt"] == "fp8" and "error" not in s]
        rs = by.get(lab, [])
        mf = main_fmt(lab) if rs else None
        return rs, [s for r in rs for s in r["samples"] if "error" not in s and (mf is None or s["fmt"] == mf)]

    def med(xs):
        xs = sorted(xs)
        return xs[len(xs) // 2] if xs else None

    def rng(xs):
        return f"{100 * min(xs):.1f}-{100 * max(xs):.1f}%" if len(xs) > 1 else pct(xs[0])

    # 1. by format: bits a weight of the format's own samples (its codes, scales and zero points; no 16-bit parts)
    md += ["| Format | Checkpoints | Samples (with a CM) | Stored | Order 0 | Best static context | CM code length | "
           "Layout estimate a tensor | Fixed-rate layout | Best general compressor (its samples: stored / static / CM) | "
           "Room on its own bytes: static / CM | Order 0 over samples: min / median / max |",
           "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- | ---: | ---: |"]
    whole_rows = []
    for lab in ORDER:
        rs, ss = samples_of(lab)
        if not ss:
            continue
        b = {est: pooled(ss, est) for est in EST}
        ncm = sum(1 for s in ss if sample_bits(s, "cm") is not None)
        cs = [s for s in ss if s.get("comp")]
        comp = {}
        for c in COMPS:
            xs = [(s["n"], 8 * s["comp"][c]["bytes"] / s["n"]) for s in cs if s["comp"].get(c, {}).get("bytes")]
            if len(xs) == len(cs) and xs:
                comp[c] = wavg(xs)
        cst, cfl, ccm = (pooled(cs, e) for e in ("stored", "static", "cm")) if cs else (None, None, None)
        h0s = [s["parts"][main_part(s)]["H0"] for s in ss]
        room_own = {e: 1 - b[e] / b["stored"] if b[e] else None for e in ("static", "cm")}
        whole = {}
        if lab not in (QPARTS, FP8PARTS):
            tot = sum(r["stored"] for r in rs)
            whole = {est: 1 - sum(r["projection"][est]["bytes"] for r in rs) / tot for est in EST}
            per = {est: [r["projection"][est]["saved"] for r in rs] for est in EST}
        errs = sum(len(s.get("phase2_errors", [])) for s in ss)
        names = [r["repo"] if r["kind"] == "st" else r["spec"] for r in rs]
        summary[lab] = {"checkpoints": names, "samples": len(ss),
                        "samples_with_cm": ncm, "phase2_errors": errs, "weights": sum(s["n"] for s in ss), "bits": b,
                        "H0_min_median_max": [min(h0s), med(h0s), max(h0s)], "room_own_bytes": room_own,
                        "compressor_samples": len(cs), "compressors": comp, "compressor_samples_stored": cst,
                        "compressor_samples_static": cfl, "compressor_samples_cm": ccm, "whole_saved": whole,
                        "whole_saved_per_checkpoint": {est: dict(zip(names, v)) for est, v in per.items()} if whole else {}}
        bc = min(comp.items(), key=lambda t: t[1]) if comp else None
        cstr = f"{bc[1]:.3f} ({bc[0]}; {f3(cst)} / {f3(cfl)} / {f3(ccm)})" if bc else "-"
        md.append(f"| {lab} | {len(rs)} | {len(ss)} ({ncm}) | {f3(b['stored'])} | {f3(b['H0'])} | **{f3(b['static'])}** | {f3(b['cm'])} | "
                  f"{f3(b['built'])} | {f3(b['fixed'])} | {cstr} | {pct(room_own['static'])} / {pct(room_own['cm'])} | "
                  f"{f3(min(h0s))} / {f3(med(h0s))} / {f3(max(h0s))} |")
        if whole:
            whole_rows.append(f"| {lab} | {pct(whole['static'])} ({rng(per['static'])}) | {pct(whole['cm'])} ({rng(per['cm'])}) | "
                              f"{pct(whole['built'])} ({rng(per['built'])}) | {pct(whole['fixed'])} | "
                              f"**{100 * (whole['static'] - whole['built']):.1f} / {100 * (whole['cm'] - whole['built']):.1f} pts** |")
    # 1b. whole checkpoints by format (byte-weighted: a format's biggest checkpoint dominates; the per-checkpoint range beside)
    md += ["", "| Format | Whole checkpoints: off at the static floor (range) | off at the CM (range) | off at the layout estimate "
           "(range; quantized checkpoints: their 16-bit and FP8 parts only) | off at the fixed-rate layouts | room left: static / CM |",
           "| :--- | ---: | ---: | ---: | ---: | ---: |"] + whole_rows
    # 2. the ladder: what each context buys, on the weights or codes (the samples the CM ladder ran on)
    md += ["", "| Format | Samples | Stored | Order 0 | Fields apart | Best position context | Best neighbour context | Matching tensor | "
           "Best single context | CM: order 0 | + position, scale | + neighbours | + matching tensor | CM's share of the values |",
           "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    ladder = {}
    for lab in ORDER:
        rs, ss = samples_of(lab)
        ss = [s for s in ss if s["parts"].get(main_part(s), {}).get("ladder")]
        if not ss:
            continue
        P = [(s["n"], s["parts"][main_part(s)]) for s in ss]

        def col(fn):
            return wavg([(n, fn(p)) for n, p in P])

        def bestof(p, names):
            v = [p["ctx"][k] for k in names if k in p.get("ctx", {})]
            return min(v) if v else p["H0"]
        row = {"stored": col(lambda p: p["width"]), "order 0": col(lambda p: p["H0"]),
               "fields apart": col(lambda p: p.get("fields", {}).get("fields apart")),
               "position": col(lambda p: bestof(p, POSITION)), "neighbour": col(lambda p: bestof(p, NEIGHBOUR)),
               "matching": col(lambda p: p["ctx"].get("matching tensor")), "static": col(lambda p: p["best_static"]),
               "cm order 0": col(lambda p: p["ladder"]["order 0"]), "cm position": col(lambda p: p["ladder"]["+ position and scale"]),
               "cm neighbours": col(lambda p: p["ladder"]["+ neighbours"]),
               "cm all": col(lambda p: min(v for v in (p.get("cm"), p.get("cm_lr2")) if v is not None)),
               "cm share": col(lambda p: p.get("cm_values", 0) / p["n"])}
        row["samples"] = len(ss)
        ladder[lab] = row
        md.append(f"| {lab} | {len(ss)} | " + " | ".join(f3(row[k]) for k in row if k != "samples") + " |")
    # 3. where the floor's bits go: the CM's bits by field (floats)
    md += ["", "| Format | Samples | Field | Stored bits | CM bits (all contexts) | Order-0 entropy of the field alone |", "| :--- | ---: | :--- | ---: | ---: | ---: |"]
    for lab in ORDER:
        rs, ss = samples_of(lab)
        ss = [s for s in ss if s["parts"].get(main_part(s), {}).get("cm_per_bit") and s["parts"][main_part(s)]["enc"] in FIELDS]
        if not ss:
            continue
        enc = ss[0]["parts"][main_part(ss[0])]["enc"]
        i = 0
        for fname, w in FIELDS[enc]:
            cmb = wavg([(s["n"], sum(s["parts"][main_part(s)]["cm_per_bit"][i:i + w])) for s in ss])
            h = wavg([(s["n"], s["parts"][main_part(s)]["fields"].get(fname if fname != "magnitude" else "exponent")) for s in ss])
            if fname == "magnitude":
                h = None
            md.append(f"| {lab} | {len(ss)} | {fname} | {w} | {f3(cmb)} | {f3(h)} |")
            i += w
    # 4. whole checkpoints
    md += ["", "| Checkpoint | Format | Stored GB | Samples by group | Layout estimate | Fixed-rate layouts | Order 0 | Static floor | CM | "
           "Room left: static / CM (points) |", "| :--- | :--- | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for lab in ORDER:
        for r in sorted(by.get(lab, []), key=lambda r: -r["stored"]):
            p = r["projection"]
            name = r["repo"] if r["kind"] == "st" else r["spec"].replace(":", " ")
            groups = ", ".join(f"{k} {v}" for k, v in sorted(p["samples_by_group"].items(), key=lambda t: -t[1]))
            md.append(f"| {name} | {lab} | {gb(r['stored'])} | {groups} | {pct(p['built']['saved'])} | {pct(p['fixed']['saved'])} | "
                      f"{pct(p['H0']['saved'])} | {pct(p['static']['saved'])} | {pct(p['cm']['saved'])} | "
                      f"{100 * (p['static']['saved'] - p['built']['saved']):.1f} / {100 * (p['cm']['saved'] - p['built']['saved']):.1f} |")
    # 5. general compressors against the floor, the same samples
    md += ["", "| Format | Samples | Stored | xz -9e | zstd -22 --long | zpaq -m5 | Static floor | CM |", "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for lab, v in summary.items():
        if v["compressor_samples"]:
            md.append(f"| {lab} | {v['compressor_samples']} | {f3(v['compressor_samples_stored'])} | " +
                      " | ".join(f3(v["compressors"].get(c)) for c in COMPS) + f" | {f3(v['compressor_samples_static'])} | {f3(v['compressor_samples_cm'])} |")
    # 6. one table for many tensors: the key's order-0 entropy with a table a tensor, a table a kind (all layers), one table
    md += ["", "| Format | A table a tensor | A table a kind | One table a checkpoint |", "| :--- | ---: | ---: | ---: |"]
    for lab in ORDER:
        rs, _ = samples_of(lab)
        if lab in (QPARTS, FP8PARTS) or not rs:
            continue
        mf = main_fmt(lab)
        t1, t2, t3, N = 0.0, 0.0, 0.0, 0
        for r in rs:
            ss = [s for s in r["samples"] if "error" not in s and (mf is None or s["fmt"] == mf)]
            if not ss:
                continue
            P = [s["parts"][main_part(s)] for s in ss]
            if len({len(p["key_hist"]) for p in P}) > 1:
                continue

            def Hh(h):
                n = sum(h)
                return -sum(c / n * math.log2(c / n) for c in h if c)
            for p in P:
                t1 += p["n"] * (Hh(p["key_hist"]) + p["H_value_given_key"])
            kinds = collections.defaultdict(list)
            for s, p in zip(ss, P):
                kinds[s["cat"]].append(p)
            for ps in kinds.values():
                h = [sum(x) for x in zip(*[p["key_hist"] for p in ps])]
                t2 += sum(p["n"] for p in ps) * Hh(h) + sum(p["n"] * p["H_value_given_key"] for p in ps)
            h = [sum(x) for x in zip(*[p["key_hist"] for p in P])]
            t3 += sum(p["n"] for p in P) * Hh(h) + sum(p["n"] * p["H_value_given_key"] for p in P)
            N += sum(p["n"] for p in P)
        if N:
            md.append(f"| {lab} | {t1 / N:.3f} | {t2 / N:.3f} | {t3 / N:.3f} |")
    json.dump({"formats": summary, "ladder": ladder}, open(os.path.join(rd, "summary.json"), "w"), indent=1)
    open(os.path.join(rd, "tables.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
