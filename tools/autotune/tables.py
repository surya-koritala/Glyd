"""The sweep's winners (sweep.py's JSON, one a GPU) as route tables, and gpu/route_tables.h from them.

For each GPU class, layout, size class of a matrix (its weights, O K: BINS) and tokens a step measured, every config's time summed over
the class's matrices; the fastest config the library can be told to take (a route, the prompt kernel's plan; not a development
variable) replaces today's route there only where it took at least GAIN less time over the class and no matrix of the class took more
than WORST of today's time. A row of the table spans the measured token counts in a run that chose the same config (the counts between
two of them too, never past the last); everywhere else the library's own route stays. Kept as they are: the routes that decode a
prompt ahead (AHEAD: glyd.gpu overlaps that decode with the products before it, which the sweep does not), and the tiered layout's
products of up to 64 tokens on compute capability 12 and on Hopper (another change measures those).

    python tables.py SWEEP.json|LAYOUTS_DIR [...] --record DIR [--header ../../gpu/route_tables.h]   (writes DIR/tables.json, DIR/summary.txt, the header)
    python tables.py --render DIR/tables.json [--header ...]                            (the header again from a record)
    (DIR/sweep-CODE.json: each GPU's input, as a sweep: `python tables.py DIR/sweep-*.json --record DIR` derives the record again)"""
import argparse
import json
import math
import os
import sys

BINS = [0, 4_000_000, 12_000_000, 24_000_000, 40_000_000, 64_000_000, 128_000_000, 384_000_000, 1 << 62]
GAIN, WORST = 1.03, 1.01
ROUTES = {"decode": 0, "gemm": 1, "mid": 2, "wg": 3, "big": 4}  # (glyd_gpu.h's GLYD_GPU_ROUTE_*)
NAMES = {v: k for k, v in ROUTES.items()} | {5: "ahead", 6: "split"}
AHEAD = 5
# a step through a model: each matrix's count (layers of each kind; q, k, v and gate, up joined, as vLLM runs them)
MIX = {
    "Qwen3.5-9B": {"gate_up": 32, "down": 32, "qkv": 8, "o": 8, "gdn_in": 24, "gdn_out": 24, "head-Qwen3.5-9B/lm_head": 1},
    "Qwen3.5-27B": {"gate_up": 64, "down": 64, "qkv": 16, "o": 16, "gdn_in": 48, "gdn_out": 48},
    "Qwen3.8-27B": {"gate_up": 64, "down": 64, "qkv": 16, "o": 16, "gdn_in": 48, "gdn_out": 48},
    "Qwen3.5-35B-A3B": {"qkv": 10, "o": 10, "gdn_in": 30, "gdn_out": 30, "shared_gate_up": 40, "shared_down": 40},
    "gemma-4-12B-it": {"gate_up": 48, "down": 48, "qkv": 40, "o": 40, "qk_global": 8, "o_global": 8},
    "gemma-4-31B-it": {"gate_up": 60, "down": 60, "qkv": 50, "o": 50, "qk_global": 10, "o_global": 10},
    "granite-4.2-8b": {"gate_up": 40, "down": 40, "qkv": 40, "o": 40},
}
STEPS, PROMPTS = (1, 8, 32, 64), (512, 2048)


def size_bin(w):
    return next(i for i in range(len(BINS) - 1) if BINS[i] <= w < BINS[i + 1])


def others(code, layout, M):
    """The points another change measures (the tiered layout's products of up to 64 tokens on compute capability 12 and Hopper)."""
    return layout == "mma" and M <= 64 and code % 1000 in (120, 90)


def cfg(r):
    return (r["route"], r["variant"], tuple(sorted(r["env"].items())))


def label(c):
    route, variant, env = c
    return route + (f"/{variant}" if route == "big" else "") + "".join(f" {k}={v}" for k, v in env)


def shippable(c):
    return c[0] in ROUTES and not c[2]


def from_layouts(d):
    """gpu/layouts.py's runs (DIR/layouts-*.json: today's route, gemm, big with the prompt kernel's own plan, decode; the rows of no
    development variable) as a sweep."""
    sweep = {"rows": [], "failed": []}
    for f in sorted(os.listdir(d)):
        if f.startswith("layouts-") and f.endswith(".json"):
            j = json.load(open(os.path.join(d, f)))
            sweep.update(gpu=j["gpu"], code=j["code"], cc=j["cc"])
            for r in j["rows"]:
                if r.get("variant", "base") == "base":
                    route = r["route"]
                    sweep["rows"].append({"model": f[8:-5], "name": r["name"], "O": r["O"], "K": r["K"], "M": r["M"], "layout": r["layout"], "way": r["way"], "head": "head-" in f,
                                          "route": route, "variant": 0 if route != "default" else -1, "env": {}, "default": route == "default", "us": r["us"], "err": r.get("err")})
    return sweep


def decide(sweep):
    """Per layout and size class, by tokens: today's time, the best config's and the best shippable one's (summed over the class), and
    the row shipped where it qualifies; then the rows of the table (runs of measured counts with the same shipped config)."""
    code = sweep["code"]
    by = {}  # (layout, bin, M) -> {matrix: {config: us}}, and today's way
    ways = {}
    for r in sweep["rows"]:
        k = (r["layout"], size_bin(r["O"] * r["K"]), r["M"])
        m = f"{r['model']}/{r['name']}"
        by.setdefault(k, {}).setdefault(m, {})["today" if r["default"] else cfg(r)] = r["us"]
        if r["default"]:
            ways.setdefault(k, {})[m] = r["way"]
    points = []
    for (layout, b, M), mats in sorted(by.items()):
        mats = {m: t for m, t in mats.items() if "today" in t}
        if not mats:
            continue
        today = math.fsum(t["today"] for t in mats.values())  # (fsum: the same bits on every Python; sum() of floats changed in 3.12)
        common = set.intersection(*(set(t) for t in mats.values())) - {"today"}
        total = {c: math.fsum(t[c] for t in mats.values()) for c in common}
        best = min(total, key=total.get) if total else None
        ship = min((c for c in total if shippable(c)), key=total.get, default=None)
        wset = set(ways[(layout, b, M)].values())
        why = None
        if not total:
            why = "no other route measured here"
        elif ship is None:
            why = "no config the library can be told to take"
        elif today / total[ship] < GAIN:
            why = f"under {GAIN:.2f}x"
        elif any(t[ship] > WORST * t["today"] for t in mats.values()):
            why = f"a matrix over {WORST:.2f}x today's"
        elif AHEAD in wset:
            why = "today's route decodes ahead (kept)"
        elif others(code, layout, M):
            why = "measured by another change (kept)"
        points.append({"layout": layout, "bin": b, "M": M, "matrices": sorted(mats), "ways": sorted(wset), "today_us": today,
                       "best": label(best) if best else None, "best_us": total.get(best), "ship": label(ship) if ship else None, "ship_us": total.get(ship),
                       "gain": today / total[ship] if ship else None, "shipped": why is None, "why": why, "_ship": ship})
    rows = []
    for layout in ("mma", "mma12"):
        for b in range(len(BINS) - 1):
            run = None
            for p in sorted((p for p in points if p["layout"] == layout and p["bin"] == b), key=lambda p: p["M"]):
                c = p["_ship"] if p["shipped"] else None
                if run and c == run["c"]:
                    run["mhi"] = p["M"]
                    continue
                if run and run["c"]:
                    rows.append(run)
                run = {"c": c, "mlo": p["M"], "mhi": p["M"], "twelve": int(layout == "mma12"), "bin": b}
            if run and run["c"]:
                rows.append(run)
    table = [{"gpu": code, "twelve": r["twelve"], "wlo": BINS[r["bin"]], "whi": BINS[r["bin"] + 1], "mlo": r["mlo"], "mhi": r["mhi"],
              "route": ROUTES[r["c"][0]], "variant": r["c"][1], "config": label(r["c"])} for r in rows]
    for p in points:
        del p["_ship"]
    return points, table


def lookup(table, gpu, twelve, O, K, M):
    """The header's tuned_route in Python (the tests compare the two): (route, variant), or (-1, 0)."""
    w = O * K
    for t in table:
        if t["gpu"] == gpu and t["twelve"] == twelve and t["wlo"] <= w < t["whi"] and t["mlo"] <= M <= t["mhi"]:
            gemm, k64 = t["route"] == 1, K % 64 == 0
            if (gemm and (M > 64 or O % 64 or K % 16)) or (t["route"] > 1 and not k64) or (t["route"] in (2, 3) and not twelve):
                return -1, 0
            return t["route"], t["variant"]
    return -1, 0


HEADER = """// The routes measured faster than the library's own on a GPU's class, by layout, a matrix's size and tokens a step (generated by
// tools/autotune/tables.py from benchmarks/gpu/{record}/tables.json; do not edit). A row: the GPU's code (glyd_gpu_gpu's, without
// GLYD_GPU_WITH_SPLIT), the layout (1: 12-bit), the matrix's weights O K in [wlo, whi), tokens in [mlo, mhi], the route
// (GLYD_GPU_ROUTE_*) and the prompt kernel's plan there (0: its own choice).
#pragma once
#include <stdint.h>

static const int64_t TUNED_ROUTES[][8] = {{
{rows}    {{-1, 0, 0, 0, 0, 0, 0, 0}},  // (the end: no GPU's code)
}};

// The row's route for M tokens of W [O, K] on the GPU of this code and layout, its plan in *variant (where given), or -1: no row, or
// the row's route does not take this matrix (GEMM: to 64 tokens, O a multiple of 64 and K of 16; the others: K a multiple of 64; MID
// and WG: the 12-bit layout's alone).
static inline int tuned_route(bool twelve, int64_t gpu, int64_t O, int64_t K, int64_t M, int* variant) {{
    const int64_t w = O * K;
    for (const auto& t : TUNED_ROUTES)
        if (t[0] == gpu && t[1] == (int64_t)twelve && w >= t[2] && w < t[3] && M >= t[4] && M <= t[5]) {{
            if (t[6] == 1 ? (M > 64 || O % 64 || K % 16) : (t[6] > 1 && K % 64) || ((t[6] == 2 || t[6] == 3) && !twelve)) return -1;
            if (variant) *variant = (int)t[7];
            return (int)t[6];
        }}
    return -1;
}}
"""


def render(rec):
    rows = "".join(f"    {{{t['gpu']}, {t['twelve']}, {t['wlo']}, {t['whi']}, {t['mlo']}, {t['mhi']}, {t['route']}, {t['variant']}}},  // {t['config']}\n"
                   for g in rec["gpus"] for t in g["table"])
    return HEADER.format(record=rec["record"], rows=rows)


def step_us(sweep, layout, M, table):
    """Per model: the products of a step of M tokens through its layers, us, by today's routes and by the table's (a matrix the table
    does not change: today's time), with the matrices of the model the sweep has."""
    t = {}
    for r in sweep["rows"]:
        if r["layout"] == layout and r["M"] == M:
            t.setdefault(f"{r['model']}/{r['name']}", {})["today" if r["default"] else cfg(r)] = r
    out = {}
    for model, mix in MIX.items():
        before = after = 0.0
        for name, n in mix.items():
            key = name if "/" in name else f"{model}/{name}"
            if key not in t or "today" not in t[key]:
                break
            today = t[key]["today"]
            route, variant = lookup(table, sweep["code"], int(layout == "mma12"), today["O"], today["K"], M)
            c = None if route < 0 else (NAMES[route], variant, ())
            before += n * today["us"]
            after += n * (t[key][c]["us"] if c in t[key] else today["us"])
        else:
            out[model] = (before, after)
    return out


def summary(rec, sweeps):
    lines = []
    for g, sw in zip(rec["gpus"], sweeps):
        lines.append(f"######## {g['name']} (code {g['code']}): {len(sw['rows'])} timings, {len(sw['failed'])} failed checks; table rows {len(g['table'])}")
        for t in g["table"]:
            lines.append(f"  {'12-bit' if t['twelve'] else 'tiered'} weights [{t['wlo'] / 1e6:g}M, {t['whi'] / 1e6:g}M) tokens {t['mlo']}-{t['mhi']}: {t['config']}")
        lines.append("  points where a config was faster than today's by GAIN but not shipped:")
        for p in g["points"]:
            if not p["shipped"] and p["gain"] and p["gain"] >= GAIN:
                lines.append(f"    {p['layout']:5s} bin {p['bin']} M={p['M']:5d}: {p['ship']} {p['gain']:.3f}x ({p['why']})")
        for layout in ("mma", "mma12"):
            lines.append(f"  {'tiered' if layout == 'mma' else '12-bit'}: a step's products through the layers, us, today -> with the table (after / before)")
            cols = {M: step_us(sw, layout, M, g["table"]) for M in STEPS + PROMPTS}
            for model in MIX:
                cells = [f"{M}: {cols[M][model][0]:8.0f} -> {cols[M][model][1]:8.0f} ({cols[M][model][1] / cols[M][model][0]:.3f})" for M in STEPS + PROMPTS if model in cols[M]]
                if cells:
                    lines.append(f"    {model:16s} " + " | ".join(cells))
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sweeps", nargs="*")
    ap.add_argument("--record", default="")
    ap.add_argument("--render", default="")
    ap.add_argument("--header", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "gpu", "route_tables.h"))
    a = ap.parse_args()
    if a.render:
        open(a.header, "w").write(render(json.load(open(a.render))))
        return
    sweeps = [from_layouts(f) if os.path.isdir(f) else json.load(open(f)) for f in a.sweeps]
    rec = {"record": os.path.basename(os.path.normpath(a.record)), "gain": GAIN, "worst": WORST, "bins": BINS, "gpus": []}
    for sw in sweeps:
        points, table = decide(sw)
        rec["gpus"].append({"name": sw["gpu"], "code": sw["code"], "cc": sw["cc"], "points": points, "table": table})
    os.makedirs(a.record, exist_ok=True)
    for sw in sweeps:  # (the record's inputs: every config's time, from which it is derived again)
        json.dump(sw, open(os.path.join(a.record, f"sweep-{sw['code']}.json"), "w"))
    json.dump(rec, open(os.path.join(a.record, "tables.json"), "w"), indent=1)
    open(os.path.join(a.record, "summary.txt"), "w").write(summary(rec, sweeps))
    open(a.header, "w").write(render(rec))
    sys.stdout.write(summary(rec, sweeps))


if __name__ == "__main__":
    main()
