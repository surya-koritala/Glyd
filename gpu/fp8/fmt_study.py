"""Which fixed-rate lossless code for e4m3 weights, by measurement: bits a value on the samples fetch_samples.py read, with every
tile header, escape and raw tile counted.

    python fmt_study.py DATA_DIR [OUT.md]      (prints the tables; writes DATA_DIR/study.json and OUT.md)

An e4m3 byte is sign (1 bit), exponent (4 bits, bias 7), mantissa (3 bits). Every code here keeps sign and mantissa as they are (4 bits a
value) and codes the exponent, which is where the room is (its entropy is 2.5-2.7 bits of the 4); so a code is its exponent code:

  (a) window(k)       a k-bit code a value: k-bit offsets from the tile's window of 2^k - 1 exponents, the last code an escape whose
                      exponent follows as 4 raw bits.
  (b) tiers(w1, ..)   the tiered code of the BF16 layout (glyd_gpu.kernels.pack_mma) at e4m3's 4-bit exponent: tier 1 is a w1-bit digit
                      a value, 2^w1 - 1 exponents (the commonest of the tile) as digits and the last digit an escape; an escape has
                      a digit of tier 2, of w2 bits, and so on; what is past the last tier is a literal, 4 raw bits (or, as
                      "tiers(.., =f)", the final tier's digits are f bits and hold what is left of the 16: no literal).
  (c) the ranking (b) and the window (a) take their exponents from: the tile's own, the 128 rows' it is in, or the tensor's.
  (d) the floors: order-0 entropy of the byte, and of sign and exponent with the 3 mantissa bits as they are.

A tile has a 32-bit pointer to its escapes and literals (fixed-rate core + a variable part, as the BF16 layout). The ranking of a tile is
4 bits an exponent it names (its direct exponents; all 16 for "=f"). A tile whose coded size is more than raw (8 bits a value + the
pointer), is stored raw."""
import importlib.util, json, os, sys

import numpy as np

PTR = 32  # bits a tile: its pointer into the variable part (its flag for raw in a bit of it)
TILES = ((16, 128), (64, 128), (128, 128))
SCOPES = ("tile", "row128", "tensor")


def entropy(counts):
    p = counts[counts > 0].astype(np.float64)
    p /= p.sum()
    return float(-(p * np.log2(p)).sum())


def tile_hist(a, th, tw):
    """Exponent counts of every tile: [row tiles, column tiles, 16]."""
    R, K = a.shape
    assert R % th == 0 and K % tw == 0, (a.shape, th, tw)
    e = ((a >> 3) & 15).astype(np.int32).reshape(R // th, th, K // tw, tw).transpose(0, 2, 1, 3)
    RT, KT = e.shape[:2]
    idx = (np.arange(RT * KT, dtype=np.int32).reshape(RT, KT, 1, 1) * 16 + e).ravel()
    return np.bincount(idx, minlength=RT * KT * 16).reshape(RT, KT, 16)


def scoped(H, th, scope):
    """H [RT, KT, 16] -> the counts its ranking is made from, [RT, KT, 16] (the tile's own, its 128 rows', or the tensor's)."""
    RT, KT, _ = H.shape
    if scope == "tile":
        return H
    if scope == "row128":
        g = max(1, 128 // th)
        G = H.reshape(RT // g, g, KT, 16).sum((1, 2))
        return np.broadcast_to(np.repeat(G, g, axis=0)[:, None, :], H.shape)
    return np.broadcast_to(H.sum((0, 1)), H.shape)


def ranked(H, ref):
    """H's counts in the order of ref's ranking (commonest first): [tiles, 16] (stable: ties by exponent, high first)."""
    order = np.argsort(-ref[..., ::-1].reshape(-1, 16), axis=1, kind="stable")  # reversed exponents: a tie takes the higher exponent
    order = 15 - order
    return np.take_along_axis(H.reshape(-1, 16), order, axis=1)


def lengths(widths, final):
    """Code length (bits of the exponent code) by rank, the direct count, and the nibbles of ranking a tile names."""
    L, acc = [], 0
    for w in widths:
        acc += w
        L += [acc] * (2 ** w - 1)
    if final:  # the final tier's digits are `final` bits and hold the rest: no literal
        L += [acc + final] * (16 - len(L))
        return L[:16], 16
    L += [acc + 4] * (16 - len(L))  # a literal: the exponent, 4 raw bits
    return L[:16], sum(2 ** w - 1 for w in widths)


def cost_tiers(Hr, n, widths, final, scope, groups):
    """Total bits of the tiles (Hr: counts by rank [T, 16]) under a tiered code, and the share of raw tiles."""
    L, nibbles = lengths(widths, final)
    coded = Hr @ np.array(L, dtype=np.int64) + 4 * n + PTR
    if scope == "tile":
        coded = coded + 4 * nibbles
    raw = 8 * n + PTR
    use_raw = coded > raw
    total = int(np.where(use_raw, raw, coded).sum())
    if scope == "row128":
        total += 4 * nibbles * groups
    elif scope == "tensor":
        total += 4 * nibbles
    return total, float(use_raw.mean())


def cost_window(H, n, th, k, scope, groups):
    """Total bits under the window code (window of 2^k - 1 exponents in a tile, 4 bits for its base)."""
    W = 2 ** k - 1
    ref = scoped(H, th, scope).reshape(-1, 16)
    cs = np.concatenate([np.zeros((len(ref), 1), dtype=np.int64), np.cumsum(ref, 1)], 1)
    cover = cs[:, W:] - cs[:, :-W]  # [T, 16 - W + 1] the count in the window starting at b
    base = cover.argmax(1)
    Hf = H.reshape(-1, 16)
    ch = np.concatenate([np.zeros((len(Hf), 1), dtype=np.int64), np.cumsum(Hf, 1)], 1)
    inside = ch[np.arange(len(Hf)), base + W] - ch[np.arange(len(Hf)), base]
    esc = n - inside
    coded = n * (4 + k) + 4 * esc + PTR + (4 if scope == "tile" else 0)
    raw = 8 * n + PTR
    use_raw = coded > raw
    total = int(np.where(use_raw, raw, coded).sum())
    if scope == "row128":
        total += 4 * groups
    elif scope == "tensor":
        total += 4
    return total, float(use_raw.mean())


CODES = [(w, None) for a in (1, 2, 3) for w in [(a,)] + [(a, b) for b in (1, 2, 3)] + [(a, b, c) for b in (1, 2, 3) for c in (1, 2, 3)]]
CODES += [((2, 2, 2, 2), None), ((2,), 3), ((2,), 4), ((3,), 3), ((3,), 4), ((2, 2), 3), ((2, 2), 4), ((2, 3), 3), ((2, 3), 4), ((2, 2, 2), 2), ((2, 2, 2), 3)]
CODES = [c for c in CODES if sum(2 ** w - 1 for w in c[0]) < 16 and (not c[1] or 16 - sum(2 ** w - 1 for w in c[0]) <= 2 ** c[1])]


def name(widths, final):
    return "tiers(" + ",".join(map(str, widths)) + (f",={final}" if final else "") + ")"


def built():
    """glyd_gpu/fp8ref.py (numpy only), loaded by path."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "glyd-gpu-py", "glyd_gpu", "fp8ref.py")
    spec = importlib.util.spec_from_file_location("fp8ref", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def study_sample(a):
    """{(tile, scope, code name): (bits, raw share)} and the floors, for one sample [R, K]."""
    out = {}
    nvals = a.size
    p = built().pack(a)  # the format as built (heads, parts in words, the shape): its bytes
    out["built"] = p.nbytes() * 8.0
    out["floor"] = entropy(np.bincount(a.ravel(), minlength=256)) * nvals
    out["split"] = (entropy(np.bincount((a.ravel() >> 3), minlength=32)) + 3) * nvals
    for th, tw in TILES:
        H = tile_hist(a, th, tw)
        n = th * tw
        ngroup = {"tile": 0, "row128": a.shape[0] // 128, "tensor": 1}  # headers outside the tiles: one a 128 rows, or one
        # tile-adaptive floor: the exponent's entropy inside each tile, sign and mantissa as they are
        Hf = H.reshape(-1, 16).astype(np.float64)
        p = Hf / n
        ent = -(np.where(p > 0, p * np.log2(np.where(p > 0, p, 1)), 0)).sum(1)
        out[(th, tw, "adaptive floor")] = float((ent * n).sum() + 4 * nvals)
        for scope in SCOPES:
            ref = scoped(H, th, scope)
            Hr = ranked(H, ref)
            for widths, final in CODES:
                out[(th, tw, scope, name(widths, final))] = cost_tiers(Hr, n, widths, final, scope, ngroup[scope])
            for k in (2, 3):
                out[(th, tw, scope, f"window({k})")] = cost_window(H, n, th, k, scope, ngroup[scope])
    return out


def key(k):
    return k if isinstance(k, str) else "/".join([f"{k[0]}x{k[1]}"] + list(k[2:]))


def main():
    """Bits a value of every code on every repo's samples; written to DATA_DIR/study.json."""
    data_dir = sys.argv[1]
    manifest = json.load(open(os.path.join(data_dir, "manifest.json")))
    results = []
    for m in manifest:
        a = np.load(os.path.join(data_dir, m["file"]))
        results.append((m, a.size, study_sample(a)))
        print(m["file"], flush=True, file=sys.stderr)
    keys = list(results[0][2])
    bits = lambda v: v[0] if isinstance(v, tuple) else v
    table = {}
    for repo in sorted({m["repo"] for m, _, _ in results}) + ["all"]:
        rs = [(n, r) for m, n, r in results if repo in ("all", m["repo"])]
        tot = sum(n for n, _ in rs)
        table[repo] = {key(k): sum(bits(r[k]) for _, r in rs) / tot for k in keys}
        table[repo].update({"raw tiles/" + key(k): sum(r[k][1] * n for n, r in rs) / tot for k in keys if isinstance(results[0][2][k], tuple)})
    json.dump(table, open(os.path.join(data_dir, "study.json"), "w"), indent=1)
    return table


def tables(table):
    """The study's tables, in Markdown: bits a value (every head, escape and raw tile counted), by repo and over all the samples."""
    repos = [r for r in table if r != "all"] + ["all"]
    short = lambda r: r.split("/")[1] if "/" in r else r
    row = lambda name, keys: f"| {name} | " + " | ".join(f"{table[r][k]:.3f}" for r in repos for k in keys) + " |"
    head = lambda cols: "| | " + " | ".join(f"{short(r)}" for r in repos) + " |\n| :--- |" + " ---: |" * len(repos)
    md = []

    def block(title, rows):
        md.append(f"\n**{title}**\n\n" + head(None))
        for name, k in rows:
            md.append(f"| {name} | " + " | ".join(f"{table[r][k]:.3f}" for r in repos) + " |")

    block("(d) the floors", [("order-0 floor of the byte", "floor"), ("sign + exponent coded, 3 mantissa bits as they are", "split"),
                              ("exponent's entropy inside each 64x128 tile (+ 4 bits)", "64x128/adaptive floor")])
    t = "64x128/tile/"
    block("(a), (b) the candidates, 64x128 tiles with their own ranking", [
        ("(a) window(2): 3 exponents + escape", t + "window(2)"), ("(a) window(3): 7 exponents + escape", t + "window(3)"),
        ("(b) tiers(2,2): 3 + 3, then a literal", t + "tiers(2,2)"), ("(b) tiers(2,3)", t + "tiers(2,3)"), ("(b) tiers(1,2,2)", t + "tiers(1,2,2)"), ("(b) tiers(2,1,2)", t + "tiers(2,1,2)"),
        ("(b) tiers(2,2,2): 3 + 3 + 3, then a literal", t + "tiers(2,2,2)"), ("(b) tiers(2,2,2,=3)", t + "tiers(2,2,2,=3)"), ("(b) tiers(2,2,2,2)", t + "tiers(2,2,2,2)")])
    own = {"tile": "the tile's own", "row128": "its 128 rows'", "tensor": "the tensor's"}
    block("(c) where the ranking comes from, tiers(2,2,2), by tile size", [
        (f"{ts} tiles, ranking: {own[sc]}", f"{ts}/{sc}/tiers(2,2,2)") for ts in ("16x128", "64x128", "128x128") for sc in own])
    block("(c) the same for window(2) (its base: the tile's own, its 128 rows', the tensor's)", [(f"{ts} tiles, window: {own[sc]}", f"{ts}/{sc}/window(2)") for ts in ("64x128",) for sc in own])
    block("the chosen format as built (glyd_gpu/fp8ref.py: 64x128 patches, levels 2, 2, 3 and a literal (layout 2), the ranking of the patch; every head, part and padding byte counted)", [("fp8ref.pack(...).nbytes() * 8 / values", "built")])
    md.append("")
    return "\n".join(md)


if __name__ == "__main__":
    table = main()
    allv = table["all"]
    n = sum(1 for _ in table) - 1
    out = tables(table)
    print(out)
    for th, tw in TILES:
        print(f"\ntile {th}x{tw}: all samples, bits a value (best 8 of {len(CODES) * 3 + 6})")
        rows = sorted(((k, v) for k, v in allv.items() if k.startswith(f"{th}x{tw}/") and "adaptive" not in k and not k.startswith("raw")), key=lambda kv: kv[1])
        for k, v in rows[:8]:
            print(f"  {k.split('/', 1)[1]:32s} {v:.4f}  raw tiles {allv['raw tiles/' + k] * 100:.2f}%")
    if len(sys.argv) > 2:
        open(sys.argv[2], "w").write(out)
