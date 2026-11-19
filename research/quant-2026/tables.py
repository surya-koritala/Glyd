"""The README's tables from results/: python tables.py [results]   (JSON only; no numpy)."""
import collections, glob, json, os, sys

D = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
HUB = json.load(open(os.path.join(D, "hub.json")))  # repo -> created, downloads_30d, downloads_all_time, as_of (hdr.py)
RES = {json.load(open(p))["repo"]: json.load(open(p)) for p in sorted(glob.glob(os.path.join(D, "*__*.json")))}
FMT = {"nvfp4": "NVFP4", "mxfp4": "MXFP4", "fp8": "FP8", "int4": "INT4", "int8": "INT8", "bf16": "16-bit", "other": "other"}
WIDTH = {"nvfp4": 4, "mxfp4": 4, "int4": 4, "fp8": 8, "int8": 8}


def main_format(r):
    """The quantized format holding the most bytes."""
    b = collections.Counter()
    for k, v in r["census"].items():
        f, p, _ = k.split(":")
        if f in WIDTH:
            b[f] += v["bytes"]
    return b.most_common(1)[0][0]


def kind(r):
    f = main_format(r)
    if f == "fp8":
        blocks = {tuple(s["scale_block"]) for s in r["samples"] if s.get("format") == "fp8" and "scale_block" in s}
        if any(b[0] == 1 and b[1] > 32 for b in blocks):
            return "FP8, a scale a row"
        if any(b[0] > 1 and b[1] > 1 for b in blocks):
            return "FP8, block scales" if any(b[1] <= 128 for b in blocks) else "FP8, a scale a tensor"
        return "FP8, a scale a tensor"
    return {"nvfp4": "NVFP4", "mxfp4": "MXFP4", "int4": "INT4 " + r["quant_method"].upper().replace("COMPRESSED-TENSORS", "(compressed-tensors)"),
            "int8": "INT8 W8A8"}[f]


def gb(x):
    return f"{x / 1e9:,.1f}"


def pct(x):
    return f"{100 * x:.1f}%"


def share(r, pred):
    tot = sum(v["bytes"] for v in r["census"].values())
    return sum(v["bytes"] for k, v in r["census"].items() if pred(*k.split(":"))) / tot


def checkpoints():
    out = ["| Checkpoint | Format | Downloads (all time) | Stored GB | Quantized codes / scales / 16-bit | Fixed-rate layouts | Order-0 floor | Best contexts | GB saved (fixed rate) |",
           "| :--- | :--- | ---: | ---: | :--- | ---: | ---: | ---: | ---: |"]
    rows = sorted(RES.values(), key=lambda r: (kind(r), -HUB[r["repo"]]["downloads_all_time"]))
    for r in rows:
        p = r["projection"]
        c = share(r, lambda f, pp, c: f in WIDTH and pp == "codes")
        s = share(r, lambda f, pp, c: f in WIDTH and pp in ("scales", "zeros"))
        b = share(r, lambda f, pp, c: f == "bf16")
        out.append(f"| [{r['repo']}](https://huggingface.co/{r['repo']}) | {kind(r)} | {HUB[r['repo']]['downloads_all_time']:,} | {gb(p['stored'])} | "
                   f"{pct(c)} / {pct(s)} / {pct(b)} | {pct(p['fixed']['saved'])} | {pct(p['H0']['saved'])} | {pct(p['ceiling']['saved'])} | "
                   f"{gb(p['stored'] - p['fixed']['bytes'])} |")
    return "\n".join(out)


def pooled(fmt, part, repos=None, keep=lambda s: True):
    """bits a value (codes) or a scale, pooled over every sample of every repo (weighted by values)."""
    acc = collections.defaultdict(float)
    for r in RES.values():
        if repos and r["repo"] not in repos:
            continue
        for s in r["samples"]:
            x = s.get(part)
            if s.get("format") != fmt or not x or not keep(s):
                continue
            acc["n"] += x["n"]
            for k in ("H0", "ceiling", "fixed"):
                acc[k] += x["n"] * x[k]
            if part == "scales":
                acc["w"] += x["n"] * x["width"]
    if not acc["n"]:
        return None
    return {k: v / acc["n"] for k, v in acc.items() if k != "n"} | {"n": acc["n"]}


def by_format():
    out = ["| Format (repos) | Part | Stored bits | Order-0 floor | Best contexts | Fixed-rate layout | Values sampled |",
           "| :--- | :--- | ---: | ---: | ---: | ---: | ---: |"]
    groups = collections.defaultdict(list)
    for r in RES.values():
        groups[main_format(r)].append(r["repo"])
    for fmt in ("nvfp4", "mxfp4", "fp8", "int4", "int8"):
        repos = [r for r in RES if any(s.get("format") == fmt for s in RES[r]["samples"])]
        widths = sorted({s["scales"]["width"] for r in repos for s in RES[r]["samples"] if s.get("format") == fmt and "scales" in s})
        for part, sw in [("codes", None)] + [("scales", w) for w in widths]:  # scales of different widths (FP8: E8M0 and BF16) apart
            x = pooled(fmt, part, repos, lambda s: sw is None or s["scales"]["width"] == sw)
            if not x:
                continue
            w = WIDTH[fmt] if part == "codes" else sw
            name = part if len(widths) < 2 or sw is None else f"{part} ({sw}-bit)"
            out.append(f"| {FMT[fmt]} ({len(repos)}) | {name} | {w} | {x['H0']:.3f} | {x['ceiling']:.3f} | {x['fixed']:.3f} | {x['n'] / 1e6:,.1f} M |" if x['n'] >= 1e5 else f"| {FMT[fmt]} ({len(repos)}) | {name} | {w} | {x['H0']:.3f} | {x['ceiling']:.3f} | {x['fixed']:.3f} | {x['n'] / 1e3:,.0f} k |")
    x = pooled("bf16", "bf16")
    out.append(f"| 16-bit parts ({len(RES)}) | weights | 16 | {x['H0']:.3f} | {x['ceiling']:.3f} | {x['fixed']:.3f} | {x['n'] / 1e6:,.0f} M |")
    return "\n".join(out)


def parts():
    """Where each checkpoint's fixed-rate saving comes from: the quantized parts vs the 16-bit ones, in GB."""
    out = ["| Checkpoint | Saved by coding the quantized codes | ... their scales and zero points | ... the 16-bit parts | Total |",
           "| :--- | ---: | ---: | ---: | ---: |"]
    for r in sorted(RES.values(), key=lambda r: (kind(r), -HUB[r["repo"]]["downloads_all_time"])):
        pp = r["projection"]["fixed"]["parts"]
        g = collections.Counter()
        for k, v in pp.items():
            f, p = k.split(":")
            key = "codes" if f in WIDTH and p == "codes" else "scales" if f in WIDTH else "bf16" if f == "bf16" else "other"
            g[key] += v["bytes"] * (1 - v["ratio"])
        tot = r["projection"]["stored"]
        out.append(f"| {r['repo']} | {gb(g['codes'])} ({pct(g['codes'] / tot)}) | {gb(g['scales'])} ({pct(g['scales'] / tot)}) | "
                   f"{gb(g['bf16'])} ({pct(g['bf16'] / tot)}) | {gb(sum(g.values()))} ({pct(sum(g.values()) / tot)}) |")
    return "\n".join(out)


def ranking():
    """Each format's own parts (codes, scales, zero points; the 16-bit parts as one more row) over every checkpoint that has them:
    the share of those bytes a fixed-rate layout and the best contexts save, and the GB that is over the 17 checkpoints."""
    acc = collections.defaultdict(collections.Counter)
    for r in RES.values():
        fx, ce = r["projection"]["fixed"]["parts"], r["projection"]["ceiling"]["parts"]
        seen = set()
        for k, v in fx.items():
            f, p = k.split(":")
            if f not in WIDTH and f != "bf16":
                continue
            a = acc[f]
            a["bytes"] += v["bytes"]
            a["fix"] += v["bytes"] * (1 - v["ratio"])
            a["ceil"] += v["bytes"] * (1 - ce[k]["ratio"])
            if p == "codes" or f == "bf16":
                a["codes_fix"] += v["bytes"] * (1 - v["ratio"])
            if f not in seen:
                seen.add(f)
                a["repos"] += 1
                a["downloads"] += HUB[r["repo"]]["downloads_all_time"]
    out = ["| Format | Checkpoints with it | Their downloads (all time) | Its bytes in them | Saved at a fixed rate | of which the codes | Best contexts |",
           "| :--- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for f, a in sorted(acc.items(), key=lambda kv: -kv[1]["fix"]):
        out.append(f"| {FMT[f]} | {a['repos']} | {a['downloads']:,} | {gb(a['bytes'])} GB | {gb(a['fix'])} GB ({pct(a['fix'] / a['bytes'])}) | "
                   f"{gb(a['codes_fix'])} GB | {gb(a['ceil'])} GB ({pct(a['ceil'] / a['bytes'])}) |")
    return "\n".join(out)


def sixteen():
    """Plan item 1 by layout: the bf16 parts at the tiered code measured on each category's own bf16 samples (else the repo's), the
    mma layout without its tile heads, Ada's pick; at the 12-bit layout's 25% (gpu/README.md), the pick on A10, A100, H100 and
    Blackwell; the fp16 parts apart (no Glyd layout)."""
    out = ["| Checkpoint | bf16 parts GB | Tiered (Ada) GB saved | 12-bit (A10, A100, H100, Blackwell) GB saved | fp16 parts GB (no layout) |",
           "| :--- | ---: | ---: | ---: | ---: |"]
    t = collections.Counter()
    bf16 = lambda s: s.get("dtype") == "BF16"
    for r in sorted(RES.values(), key=lambda r: (kind(r), -HUB[r["repo"]]["downloads_all_time"])):
        f16 = r.get("f16_bytes", {})
        bf = tier = 0.0
        for k, v in r["census"].items():
            f, p, c = k.split(":")
            if f != "bf16":
                continue
            b = v["bytes"] - f16.get(c, 0)
            x = pooled("bf16", "bf16", [r["repo"]], lambda s: bf16(s) and s["category"] == c) or pooled("bf16", "bf16", [r["repo"]], bf16)
            bf += b
            tier += b * (1 - x["fixed"] / 16)
        tot, f = r["projection"]["stored"], sum(f16.values())
        t["bf"] += bf; t["tier"] += tier; t["f16"] += f
        out.append(f"| {r['repo']} | {gb(bf)} | {gb(tier)} ({pct(tier / tot)}) | {gb(bf / 4)} ({pct(bf / 4 / tot)}) | {gb(f)} |")
    out.append(f"| all {len(RES)} | {gb(t['bf'])} | {gb(t['tier'])} ({pct(t['tier'] / t['bf'])} of the bf16 parts) | "
               f"{gb(t['bf'] / 4)} (25.0% of them) | {gb(t['f16'])} |")
    return "\n".join(out)


def counts():
    return ", ".join(f"{r['repo']} {len(r['samples'])}" for r in RES.values() if len(r["samples"]) != 14) or "every repo 14"


def fp8_built():
    out = []
    for repo in ("Qwen/Qwen3.8-27B-FP8", "mistralai/Mistral-Small-4-119B-2603", "Qwen/Qwen3.6-35B-A3B-FP8", "RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic",
                 "deepseek-ai/DeepSeek-V4.1-Flash"):
        x = pooled("fp8", "codes", [repo])
        if x:
            out.append(f"| {repo} | {x['fixed']:.3f} | {x['H0']:.3f} | {x['ceiling']:.3f} |")
    return "| Repo | fp8ref as built | Order-0 | Best contexts |\n| :--- | ---: | ---: | ---: |\n" + "\n".join(out)


def per_repo():
    out = ["| Checkpoint | Format | Codes: order-0 / best contexts / fixed rate (bits) | Scales: stored / order-0 / fixed rate (bits) | Zero points: order-0 / fixed |",
           "| :--- | :--- | :--- | :--- | :--- |"]
    for r in sorted(RES.values(), key=lambda r: (kind(r), -HUB[r["repo"]]["downloads_all_time"])):
        for f in sorted({s["format"] for s in r["samples"] if s.get("format") in WIDTH}):
            c, sc, z = pooled(f, "codes", [r["repo"]]), pooled(f, "scales", [r["repo"]]), pooled(f, "zeros", [r["repo"]])
            out.append(f"| {r['repo']} | {FMT[f]} | {c['H0']:.3f} / {c['ceiling']:.3f} / {c['fixed']:.3f} | "
                       + (f"{sc['w']:.0f} / {sc['H0']:.2f} / {sc['fixed']:.2f}" if sc else "one a tensor") + " | "
                       + (f"{z['H0']:.2f} / {z['fixed']:.2f}" if z else "") + " |")
    return "\n".join(out)


if __name__ == "__main__":
    for title, f in (("checkpoints", checkpoints), ("by format", by_format), ("per repo", per_repo), ("parts", parts), ("ranking", ranking), ("fp8 built", fp8_built),
                     ("16-bit parts by layout", sixteen), ("sample counts other than 14", counts)):
        print(f"\n### {title}\n\n{f()}")
