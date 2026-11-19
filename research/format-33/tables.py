"""The README's tables from f33.py's results.   python tables.py RESULTS_DIR"""
import collections, glob, json, os, sys

recs = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(sys.argv[1], "*.json")))]
bad = [r for r in recs if "error" in r]
recs = [r for r in recs if "error" not in r]
N = sum(r["n"] for r in recs)
short = lambda repo: repo.split("/")[1]


def pooled(get):
    v = [(get(r), r["n"]) for r in recs]
    if any(x is None for x, _ in v):
        return None
    return sum(x * n for x, n in v) / N, max(x for x, _ in v), sum(1 for x, _ in v if x <= 10.72)


def g(sym, scope, cfg):
    return lambda r: r["layouts"][sym].get(scope, {}).get(cfg)


print(f"{len(recs)} samples, {N / 1e6:.0f} M weights, {len(bad)} errors")
for r in bad:
    print("  error:", r["sample"]["repo"], r["sample"]["name"], r["error"])

# every (symbol, table, config) present in every sample: pooled bits, worst sample, samples <= 10.72
rows = []
for sym in recs[0]["layouts"]:
    for scope in ("tensor", "dstep", "drow", "dcol", "step"):
        cfgs = set.intersection(*[set(r["layouts"][sym].get(scope, {})) for r in recs])
        for c in cfgs:
            p = pooled(g(sym, scope, c))
            rows.append((p[0], p[1], p[2], sym, scope, c))
rows.sort()
print("\n## Fixed-rate candidates, best 40 by pooled bits (bits a weight: pooled, worst sample, samples at <= 10.72 of", len(recs), ")")
print("| Code | Table | Tiers | Pooled | Worst | <= 10.72 |\n| :-- | :-- | :-- | --: | --: | --: |")
for p, w, k, sym, scope, c in rows[:40]:
    print(f"| {sym} | {scope} | {c} | {p:.3f} | {w:.3f} | {k} |")
print("\n## Each code's best (pooled), tensor table and best table")
best = {}
for p, w, k, sym, scope, c in rows:
    best.setdefault((sym, scope), (p, w, k, c))
for (sym, scope), (p, w, k, c) in sorted(best.items(), key=lambda x: x[1][0]):
    print(f"| {sym} | {scope} | {c} | {p:.3f} | {w:.3f} | {k} |")

tiered = pooled(g("E", "tensor", "2-2-2d"))
print(f"\ntiered as built (E 2-2-2d tensor, the model's count): pooled {tiered[0]:.3f}, worst {tiered[1]:.3f}, {tiered[2]} samples <= 10.72")
# The model of results/sizes counts 2 tier-3 bits a literal more than the library stores (it gives a literal a digit in every tier;
# the library gives it a placeholder tier-2 digit and no tier-3 digit): the tiered layout's exact bits are the model's less
# 2 x P(rank >= 10), which matches f33_pack.py's packs to 0.0001 (results/decode/pack.txt). For the 4-tier codes the model's
# surplus (2 + w4 bits a literal past rank 12 or 16) is 0.0003-0.007 on the five packed tensors: their rows are upper bounds.
exact = lambda r: r["layouts"]["E"]["tensor"]["2-2-2d"] - 2 * (1 - sum(r["floors"]["exp_top"][:10]))
te = pooled(exact)
print(f"tiered as built, exact rule: pooled {te[0]:.3f}, worst {te[1]:.3f}, {te[2]} samples <= 10.72")

# floors pooled
F = collections.OrderedDict()
for k in ("H16", "best_context"):
    F[k] = sum(r["floors"][k] * r["n"] for r in recs) / N
for sym in recs[0]["layouts"]:
    F[f"H({sym})+raw"] = sum(r["layouts"][sym]["H"] * r["n"] for r in recs) / N
    for a in ("bits", "states_16_steps", "states_a_step"):
        F[f"rANS({sym}) {a}"] = sum(r["layouts"][sym]["ans"][a] * r["n"] for r in recs) / N
for k in ("H_E", "H_SM_given_E", "H_S_given_E", "H_m6_given_E", "H_M_given_E", "H_E_given_left", "H_E_given_row", "H_E_given_col", "H_E_given_left_row"):
    F[k] = sum(r["floors"][k] * r["n"] for r in recs) / N
print("\n## Floors and entropy-coded references, pooled")
for k, v in F.items():
    print(f"| {k} | {v:.3f} | {100 * (1 - v / 16):.1f}% |")

# per model and per kind: tiered, the best fixed candidate (top row), floors
bp, bw, bk, bs, bsc, bc = rows[0]
print(f"\n## By model (tiered; best candidate {bs} {bsc} {bc}; H16; best context)")
by = collections.defaultdict(list)
for r in recs:
    by[short(r["sample"]["repo"])].append(r)
    by["kind:" + r["sample"]["kind"]].append(r)
for k, rs in by.items():
    n = sum(r["n"] for r in rs)
    f = lambda get: sum(get(r) * r["n"] for r in rs) / n
    print(f"| {k} | {len(rs)} | {f(exact):.3f} | {max(exact(r) for r in rs):.3f} | {f(g(bs, bsc, bc)):.3f} | {max(g(bs, bsc, bc)(r) for r in rs):.3f} | {f(lambda r: r['floors']['H16']):.3f} | {f(lambda r: r['floors']['best_context']):.3f} |")

print("\n## Every sample")
for r in recs:
    s = r["sample"]
    L = r["layouts"]
    fl = r["floors"]
    print(f"| {short(s['repo'])} | {s['name']} | {s['kind']} | {s['shape'][0]}x{s['shape'][1]} | {exact(r):.3f} | {g(bs, bsc, bc)(r):.3f} | "
          f"{fl['H16']:.3f} | {fl['best_context']:.3f} | {fl['H_E']:.3f} | {fl['exp_top'][0]:.3f} {fl['exp_top'][1]:.3f} {fl['exp_top'][2]:.3f} | {fl['exp_max']} |")

# Whole models: each kind's mean bits weighted by the kind's bf16 matrix bytes in the model (the inventory's headers), when given.
if len(sys.argv) > 2:
    from plan import kind
    print("\n## Whole models, by kind's bytes (tiered exact rule | E 2-2-2-2d | E 2-2-2-3d | H16 | best context: bits, saving)")
    cols = [exact, g("E", "tensor", "2-2-2-2d"), g("E", "tensor", "2-2-2-3d"), lambda r: r["floors"]["H16"], lambda r: r["floors"]["best_context"]]
    for repo in sorted({r["sample"]["repo"] for r in recs}):
        h = json.load(open(os.path.join(sys.argv[2], repo.replace("/", "__") + ".json")))
        by = collections.Counter()
        for ts in h["shards"].values():
            for n, (dt, shape) in ts.items():
                if dt == "BF16" and len(shape) == 2 and shape[0] >= 64 and shape[1] % 16 == 0:
                    by[kind(n)] += shape[0] * shape[1]
        mine = [r for r in recs if r["sample"]["repo"] == repo]
        tot = sum(by.values())
        out = []
        for get in cols:
            b = 0
            for kd, nb in by.items():
                s = [r for r in mine if r["sample"]["kind"] == kd] or mine
                b += nb * sum(get(r) * r["n"] for r in s) / sum(r["n"] for r in s)
            out.append(b / tot)
        print(f"| {short(repo)} | " + " | ".join(f"{x:.3f} ({100 * (1 - x / 16):.2f}%)" for x in out) + " |")
