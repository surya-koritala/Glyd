"""format-33: lossless bf16 codes against the 10.72-bit bar (33.0% off), on real tensors of 2026 models read by HTTP range.

    python f33.py samples.json OUT_DIR        (one OUT_DIR/<n>.json a sample; PROCS processes)
    python f33.py --selftest

Every size is exact: the bytes a step (64 rows x 16 columns, 1024 weights, the tiered layout's unit) of each candidate layout takes,
every digit, word, list entry, table and header counted (rounded up as the layout stores them), plus the step's 4-byte block offset.
Known surplus (as gpu/experimental/tiered_fast/v2bits.py counts it): a literal is given a digit in every tier past the first; the
library gives it a placeholder tier-2 digit and no tier-3 digit. For the tiered layout the exact count is this one less 2 x P(rank >=
10) (tables.py applies it; it matches f33_pack.py's real packs to 0.0001); for the 4-tier codes the surplus is 0.0003-0.007 bits on
the five tensors f33_pack.py packed, so their numbers are upper bounds.
The order inside a step does not change any of these counts. Floors are entropies of the sample (Miller-Madow corrected).

Codes (a "symbol" is part of the bf16 word; the rest of the word is stored raw beside it):
  E     the exponent (8 bits); raw: sign + 7 mantissa bits          (the tiered layout's symbol)
  Em6   exponent + top mantissa bit (9 bits); raw: sign + 6 bits    (exponent-mantissa joint)
  Em65  exponent + top 2 mantissa bits (10 bits); raw: sign + 5
  SE    sign + exponent (9 bits); raw: 7 mantissa bits               (sign-exponent joint)
  Eh    the exponent's top 7 bits (the 12-bit layout's high byte without the sign); raw: sign, exponent bit 0, mantissa (9 bits)
A digit code ("tiers"): tier 1 a w1-bit digit for every weight (fixed part); its 2^w1 - 1 codes are the commonest symbols and the
last goes on to tier 2 (w2-bit digits, kept only for tier 1's escapes), and so on; the last tier either dense (all 2^w codes are
symbols) or with an escape to a literal (10-bit place among the step's tier-1 escapes + the symbol's bits). Tier 2's digits are
stored as 32-bit words, later tiers and literals as bytes (the tiered layout's arrangement: E with widths 2,2,2 dense is exactly the
tiered layout as built, v2b in gpu/experimental/tiered_fast/v2bits.py).
Tables (which symbols the tiers hold, in rank order):
  tensor   one table a tensor (by the tensor's counts)
  dstep    the tensor's table, the exponent shifted by a 3-bit offset chosen a step (header 3 bits a step)
  drow     the same offset a 64-row block (shared by its steps; header 3 bits a row block)
  dcol     the same offset a 16-column strip
  step     the step's own ranking of the tensor's 16 commonest symbols (header 4 bits a tabled symbol a step)
Entropy-coded references (not fixed-rate): a static rANS over each symbol (12-bit frequencies) + raw, with 32 lanes' 32-bit states
written once a step or once every 16 steps.
"""
import concurrent.futures as cf, itertools, json, math, os, struct, sys, time, urllib.error, urllib.request

import numpy as np

HF = "https://huggingface.co"
LN2 = math.log(2)

SYMS = {  # name: (symbol from the 16-bit word, symbol bits, raw bits, exponent shift inside the symbol)
    "E": (lambda w: (w >> 7) & 0xFF, 8, 8, 0),
    "Em6": (lambda w: (w >> 6) & 0x1FF, 9, 7, 1),
    "Em65": (lambda w: (w >> 5) & 0x3FF, 10, 6, 2),
    "SE": (lambda w: w >> 7, 9, 7, 0),
    "Eh": (lambda w: (w >> 8) & 0x7F, 7, 9, None),
}
RMAX = 128  # ranks past 127 are all literals


# ---------------- reading (HTTP range, no token) ----------------

def get(url, rng, tries=8):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={rng[0]}-{rng[1]}", "User-Agent": "glyd-f33/1"})
            with urllib.request.urlopen(req, timeout=180) as r:
                data = r.read()
            if len(data) != rng[1] - rng[0] + 1:
                raise IOError(f"short read {len(data)}")
            return data
        except urllib.error.HTTPError as e:
            if e.code in (401, 403, 404) or i == tries - 1:
                raise
            time.sleep(min(60, 3 * 2 ** i))
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(min(60, 3 * 2 ** i))


def header(repo, shard):
    url = f"{HF}/{repo}/resolve/main/{shard}"
    b = get(url, (0, 262143))
    n = struct.unpack("<Q", b[:8])[0]
    if n + 8 > len(b):
        b += get(url, (len(b), n + 7))
    return n, json.loads(b[8:8 + n])


def read(s, hdr):
    n, h = hdr
    t = h[s["name"]]
    assert t["dtype"] == "BF16" and t["shape"] == s["shape"], (t, s)
    R, C = s["shape"]
    a = 8 + n + t["data_offsets"][0] + s["r0"] * C * 2
    b = 8 + n + t["data_offsets"][0] + s["r1"] * C * 2 - 1
    raw = get(f"{HF}/{s['repo']}/resolve/main/{s['shard']}", (a, b))
    return np.frombuffer(raw, dtype="<u2").reshape(s["r1"] - s["r0"], C)


# ---------------- entropies ----------------

def H(c):
    c = np.asarray(c, dtype=np.float64).ravel()
    n = c.sum()
    p = c[c > 0] / n
    return float(-(p * np.log2(p)).sum() + ((c > 0).sum() - 1) / (2 * n * LN2))


def Hc(ctx, y, ny):
    """H(y | ctx), ctx and y non-negative ints, y < ny."""
    j = ctx.astype(np.int64) * ny + y
    u, c = np.unique(j, return_counts=True)
    cc = np.bincount(np.searchsorted(np.unique(u // ny), u // ny), weights=c)
    n = c.sum()
    return float((-(c * np.log2(c)).sum() + (cc * np.log2(cc)).sum()) / n + (len(u) - len(cc)) / (2 * n * LN2))


# ---------------- layouts ----------------

def steps(W):
    R, C = W.shape
    return W.reshape(R // 64, 64, C // 16, 16).transpose(0, 2, 1, 3).reshape(-1, 1024)


def configs():
    """(widths, last_dense): up to 4 tiers of 1-4 bits, at most RMAX - 1 tabled symbols."""
    out = []
    for k in range(1, 5):
        for ws in itertools.product((1, 2, 3, 4), repeat=k):
            for dense in (True, False):
                cap = sum((1 << w) - 1 for w in ws[:-1]) + (1 << ws[-1]) - (0 if dense else 1)
                if cap < RMAX and not (k == 1 and ws[0] < 2):
                    out.append((ws, dense))
    return out


CONFIGS = configs()


def cuts(ws, dense):
    """rank thresholds: c[i] = first rank tier i + 2 holds (escapes of tier i + 1 are ranks >= c[i]); last: first literal rank"""
    c, t = [], 0
    for i, w in enumerate(ws):
        t += (1 << w) - (0 if (dense and i == len(ws) - 1) else 1)
        c.append(t)
    return c


def step_bytes(G, ws, dense, sb, raw):
    """G[:, r] = weights of rank >= r in each step -> bytes of each step (fixed part, block, block offset)."""
    c = cuts(ws, dense)
    b = 1024 * (ws[0] + raw) // 8 + 4
    for i in range(1, len(ws)):
        n = G[:, c[i - 1]].astype(np.int64)
        b = b + (4 * ((ws[i] * n + 31) // 32) if i == 1 else (ws[i] * n + 7) // 8)
    L = G[:, c[-1]].astype(np.int64)
    return b + ((10 + sb) * L + 7) // 8


def ranks_of(counts):
    order = np.argsort(-counts, kind="stable")
    r = np.full(counts.size, RMAX - 1, dtype=np.int64)
    k = min(RMAX - 1, int((counts > 0).sum()))
    r[order[:k]] = np.arange(k)
    return r, order


def suffix(hist):
    return hist[:, ::-1].cumsum(1)[:, ::-1]


def G_of(sym_steps, rank):
    ns = sym_steps.shape[0]
    idx = (np.arange(ns, dtype=np.int64)[:, None] * RMAX + rank[sym_steps]).ravel()
    h = np.bincount(idx, minlength=ns * RMAX).reshape(ns, RMAX)
    return np.concatenate([suffix(h), np.zeros((ns, 1), h.dtype)], 1)


def shifted(w, name, d):
    """the symbol of w with its exponent lowered by d (clipped), for the offset tables"""
    f, sb, raw, sh = SYMS[name]
    e = ((w >> 7) & 0xFF).astype(np.int64) - d
    e = np.clip(e, 0, 255)
    if name == "SE":
        return ((w >> 15).astype(np.int64) << 8) | e
    return (e << sh) | (f(w).astype(np.int64) & ((1 << sh) - 1))


DELTAS = range(-3, 5)


def measure_layouts(W):
    S = steps(W)
    ns = S.shape[0]
    RB, KS = W.shape[0] // 64, W.shape[1] // 16
    n = S.size
    res = {}
    for name, (f, sb, raw, sh) in SYMS.items():
        sym = f(S).astype(np.int64)
        counts = np.bincount(sym.ravel(), minlength=1 << sb)
        rank, order = ranks_of(counts)
        G = G_of(sym, rank)
        tab_bits = 8 * 2 * min(RMAX, int((counts > 0).sum()))  # the tensor's table (2 bytes a symbol), counted once a sample
        best = {}
        for ws, dense in CONFIGS:
            b = step_bytes(G, ws, dense, sb, raw)
            best[(ws, dense)] = (8 * b.sum() + tab_bits) / n
        top = sorted(best, key=best.get)[:24]
        # the tiered layout as built: E, 2-2-2 dense, always evaluated
        must = [((2, 2, 2), True), ((2, 2), True), ((2, 3), True), ((3, 2), True), ((2, 2, 2, 2), True), ((3, 2, 2), True), ((2, 2, 3), True)]
        keep = list(dict.fromkeys(must + top))
        out = {"tensor": {f"{'-'.join(map(str, ws))}{'d' if dense else 'l'}": best[(ws, dense)] for ws, dense in keep}}
        # offsets (exponent shift) a step, a row block, a column strip; the step's own ranking
        if sh is not None:
            per = []
            for d in DELTAS:
                Gd = G_of(shifted(S, name, d), rank)
                per.append(np.stack([step_bytes(Gd, ws, dense, sb, raw) for ws, dense in keep], 1))
            P = np.stack(per, 0)  # [delta, step, config]
            out["dstep"] = {}
            out["drow"] = {}
            out["dcol"] = {}
            Pr = P.reshape(len(DELTAS), RB, KS, len(keep))
            for j, (ws, dense) in enumerate(keep):
                k = f"{'-'.join(map(str, ws))}{'d' if dense else 'l'}"
                out["dstep"][k] = float((8 * P[:, :, j].min(0).sum() + 3 * ns + tab_bits) / n)
                out["drow"][k] = float((8 * Pr[:, :, :, j].sum(2).min(0).sum() + 3 * RB + tab_bits) / n)
                out["dcol"][k] = float((8 * Pr[:, :, :, j].sum(1).min(0).sum() + 3 * KS + tab_bits) / n)
        # the step's own ranking of the tensor's 16 commonest symbols
        top16 = order[:16]
        pos = np.full(1 << sb, 16, dtype=np.int64)
        pos[top16] = np.arange(16)
        idx = (np.arange(ns, dtype=np.int64)[:, None] * 17 + pos[sym]).ravel()
        h = np.bincount(idx, minlength=ns * 17).reshape(ns, 17)
        outside = h[:, 16:17]
        hs = -np.sort(-h[:, :16], 1)
        Gs = np.zeros((ns, RMAX + 1), dtype=np.int64)
        Gs[:, :17] = suffix(np.concatenate([hs, np.zeros((ns, 1), hs.dtype)], 1))
        Gs[:, :17] += outside
        Gs[:, 17:] = outside
        out["step"] = {}
        for ws, dense in keep:
            k = f"{'-'.join(map(str, ws))}{'d' if dense else 'l'}"
            tabled = min(16, cuts(ws, dense)[-1])
            out["step"][k] = float((8 * step_bytes(Gs, ws, dense, sb, raw).sum() + 4 * tabled * ns + tab_bits) / n)
        # entropy-coded reference: static rANS, 12-bit frequencies (each seen symbol at least 1/4096)
        q = np.where(counts > 0, np.maximum(1, np.round(counts / counts.sum() * 4096)), 0)
        while q.sum() != 4096:  # fix the total on the largest
            q[np.argmax(q)] += 4096 - q.sum()
        ans = float((counts[counts > 0] * -np.log2(q[counts > 0] / 4096)).sum() / n + raw)
        out["ans"] = {"bits": ans, "states_a_step": ans + 32 * 32 / 1024, "states_16_steps": ans + 32 * 32 / 16384}
        out["H"] = H(counts) + raw
        res[name] = out
    return res


def floors(W):
    w = W.ravel()
    e = ((w >> 7) & 0xFF).astype(np.int64)
    sm = (((w >> 15) << 7) | (w & 0x7F)).astype(np.int64)
    word = np.bincount(w, minlength=65536)
    he = H(np.bincount(e, minlength=256))
    h_sm_e = Hc(e, sm, 256)
    out = {"H16": H(word), "H_E": he, "H_SM_given_E": h_sm_e, "H_S_given_E": Hc(e, (w >> 15).astype(np.int64), 2),
           "H_m6_given_E": Hc(e, ((w >> 6) & 1).astype(np.int64), 2), "H_M_given_E": Hc(e, (w & 0x7F).astype(np.int64), 128)}
    # contexts: the exponent given its left neighbour's, given its row's / column's mean exponent (rounded; side information free)
    E = ((W >> 7) & 0xFF).astype(np.int64)
    out["H_E_given_left"] = Hc(E[:, :-1].ravel(), E[:, 1:].ravel(), 256)
    rowm = np.repeat(np.round(E.mean(1) * 4).astype(np.int64), E.shape[1])
    colm = np.tile(np.round(E.mean(0) * 4).astype(np.int64), E.shape[0])
    out["H_E_given_row"] = Hc(rowm - rowm.min(), E.ravel(), 256)
    out["H_E_given_col"] = Hc(colm - colm.min(), E.ravel(), 256)
    out["H_E_given_left_row"] = Hc((E[:, :-1] * 4096 + (np.round(E.mean(1) * 4).astype(np.int64)[:, None] - int(np.round(E.mean(1) * 4).min()))).ravel(), E[:, 1:].ravel(), 256)
    out["best_context"] = min(out["H_E_given_left"], out["H_E_given_row"], out["H_E_given_col"], out["H_E_given_left_row"]) + h_sm_e
    ex = np.bincount(e, minlength=256)
    p = np.sort(ex)[::-1] / ex.sum()
    out["exp_top"] = [round(float(x), 5) for x in p[:12]]
    out["exp_max"] = int(e.max())
    return out


def measure(job):
    s, hdr = job
    t0 = time.time()
    try:
        W = read(s, hdr)
        r = {"sample": s, "n": int(W.size), "floors": floors(W), "layouts": measure_layouts(W)}
        r["tiered"] = r["layouts"]["E"]["tensor"]["2-2-2d"]
    except Exception as e:  # recorded, never hidden
        import traceback
        r = {"sample": s, "error": f"{type(e).__name__}: {e}", "tb": traceback.format_exc()}
    r["seconds"] = round(time.time() - t0, 1)
    return r


def main():
    samples = json.load(open(sys.argv[1]))
    od = sys.argv[2]
    os.makedirs(od, exist_ok=True)
    hdrs = {}
    for s in samples:  # every shard's header once, 1 s apart
        k = (s["repo"], s["shard"])
        if k not in hdrs:
            time.sleep(1)
            hdrs[k] = header(*k)
    jobs = [(s, hdrs[(s["repo"], s["shard"])]) for s in samples]
    with cf.ProcessPoolExecutor(int(os.environ.get("PROCS", 16))) as ex:
        for i, r in enumerate(ex.map(measure, jobs)):
            json.dump(r, open(os.path.join(od, f"{i:03d}.json"), "w"), indent=0)
            s = r["sample"]
            print(f"{i:3d} {s['repo']:30s} {s['name'][-48:]:48s} " + (f"tiered {r['tiered']:.3f} H16 {r['floors']['H16']:.3f}" if "error" not in r else r["error"]) + f" {r['seconds']} s", flush=True)


def selftest():
    rng = np.random.default_rng(0)
    # a Gaussian-like bf16 matrix
    x = (rng.standard_normal((256, 512)) * 0.02).astype(np.float32)
    W = (x.view(np.uint32) >> 16).astype(np.uint16)
    L = measure_layouts(W)
    # the tiered layout by v2bits' formula
    S = steps(W)
    e = ((S >> 7) & 0xFF).astype(np.int64)
    h = np.bincount(e.ravel(), minlength=256)
    order = np.argsort(-h, kind="stable")
    rank = np.full(256, 99)
    rank[order[:10]] = np.arange(10)
    r = rank[e]
    t1, t2, Lc = (r >= 3).sum(1), (r >= 6).sum(1), (r >= 10).sum(1)  # (v2bits' count: a literal with a tier-3 digit)
    v2b = 1280 + ((2 * t2 + 7) >> 3) + ((18 * Lc + 7) >> 3) + 4 * ((t1 + 15) >> 4)
    ref = 8 * v2b.sum() / W.size + 0.03125
    tabled = 16 * min(RMAX, int((h > 0).sum())) / W.size
    assert abs(L["E"]["tensor"]["2-2-2d"] - tabled - ref) < 1e-9, (L["E"]["tensor"]["2-2-2d"], ref)
    # a step of one symbol: tier 1 only
    W1 = np.full((64, 16), 0x3F80, dtype=np.uint16)
    assert abs(measure_layouts(W1)["E"]["tensor"]["2-2-2d"] - (10 + 32 / 1024 + 16 / 1024)) < 1e-9
    # entropies
    assert abs(H(np.ones(16)) - 4 - 15 / (2 * 16 * LN2)) < 1e-12
    y = np.arange(4000) % 4
    assert abs(Hc(np.zeros(4000, int), y, 4) - 2 - 3 / (2 * 4000 * LN2)) < 1e-9
    assert Hc(y, y, 4) < 1e-9
    F = floors(W)
    assert abs(F["H16"] - (F["H_E"] + F["H_SM_given_E"])) < 0.1, F
    print("selftest ok", L["E"]["tensor"]["2-2-2d"], F["H16"])


if __name__ == "__main__":
    if sys.argv[1:] == ["--selftest"]:
        selftest()
    else:
        main()
