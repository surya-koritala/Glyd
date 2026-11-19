"""CPU packers for the decode-cost check (f33_dec.cu): the tiered layout as built (v2b: tiers 2-2-2, tier 3 dense, a literal list
of 18-bit entries; byte for byte the library's format), its four-tier variant T4 (tier 3 with an escape to a dense tier 4 of 2-bit
digits (4 symbols, "t4a" = f33's E 2-2-2-2d) or 3-bit digits (8 symbols, "t4b" = E 2-2-2-3d)), and a static rANS of the exponent
over strips of 16 steps (the entropy-coded reference: 32 lanes' interleaved states, 12-bit frequencies, 16-bit renormalization).
    python f33_pack.py samples.json IDX[,IDX...] OUT_DIR      (also writes a synthetic tensor of random bits: every slow path)
Each tensor T: T.w (its uint16 rows), T.dims, and per layout T.<fmt>.* (see f33_dec.cu's loader); prints each pack's bits a weight.
T4 a step's block: [tier-3 digits of the tier-2 escapes (2 bits)][from the next byte: tier-4 digits of the tier-3 escapes
(2 or 3 bits)][from the next byte: the literal list][the tier-2 digit words, from the end]; the fixed part is the library's."""
import json, os, sys

import numpy as np


def order_steps(W):
    """W [O, K] -> [steps, 1024] in the library's order: step rb * KS + ks, lane 4g + t, weight 4n + j (j = 2 jh + jl) is
    W[64 rb + 8n + g][16 ks + 8 jh + 2t + jl]"""
    O, K = W.shape
    return W.reshape(O // 64, 8, 8, K // 16, 2, 4, 2).transpose(0, 3, 2, 5, 1, 4, 6).reshape(-1, 1024)


def fixed(v, d1):
    """the library's fixed part of each step: tier-1 digit words [2][32 lanes], then the bytes (mantissa << 1 | the pair partner's sign)"""
    S = v.shape[0]
    d = d1.reshape(S, 32, 2, 16).astype(np.uint32)
    words = (d << (2 * np.arange(16, dtype=np.uint32))).sum(-1, dtype=np.uint32)
    out = np.empty((S, 1280), np.uint8)
    out[:, :256] = np.ascontiguousarray(words.transpose(0, 2, 1)).view(np.uint8).reshape(S, 256)
    pr = v.reshape(S, 512, 2).astype(np.uint32)
    byts = ((pr & 0x7F) << 1) | ((pr >> 15) & 1)[:, :, ::-1]
    out[:, 256:] = byts.astype(np.uint8).reshape(S, 32, 2, 16).transpose(0, 2, 1, 3).reshape(S, 1024)
    return out


def put(buf, bitpos, val):
    """OR non-overlapping fields (LSB first) into a byte buffer (int64 array)"""
    bitpos = np.asarray(bitpos, np.int64)
    val = np.asarray(val, np.int64) << (bitpos & 7)
    b = bitpos >> 3
    for k in range(4):
        np.add.at(buf, b + k, (val >> (8 * k)) & 0xFF)


def pack_tiered(W, w4):
    """w4 = 0: the library's v2b; 2 or 3: T4 with a dense tier 4 of w4-bit digits. -> files dict, bits a weight"""
    v = order_steps(W)
    S = v.shape[0]
    e = ((v >> 7) & 0xFF).astype(np.int64)
    order = np.argsort(-np.bincount(e.ravel(), minlength=256), kind="stable")
    nsym = 10 if w4 == 0 else 9 + (1 << w4)
    rank = np.full(256, 255, np.int64)
    rank[order[:nsym]] = np.arange(nsym)
    r = rank[e]
    lit = r >= nsym
    data = fixed(v, np.minimum(r, 3))
    m1 = r >= 3
    t1 = m1.sum(1)
    k1 = np.cumsum(m1, 1) - m1
    d2 = np.where(lit, 0, np.minimum(r - 3, 3))
    m2 = (r >= 6) & ~lit
    d3 = (r - 6) if w4 == 0 else np.minimum(r - 6, 3)
    m3 = ((r >= 9) & ~lit) if w4 else np.zeros_like(m2)
    t2, t3, nl = m2.sum(1), m3.sum(1), lit.sum(1)
    r0 = (2 * t2 + 7) >> 3
    r1 = r0 + ((w4 * t3 + 7) >> 3)
    size = r1 + ((18 * nl + 7) >> 3) + 4 * ((t1 + 15) >> 4)
    start = 128 + np.cumsum(size) - size
    end = start + size
    total = 128 + int(size.sum()) + 256
    buf = np.zeros(total + 8, np.int64)
    put(buf, (8 * end[:, None] - 32 * ((k1 >> 4) + 1) + 2 * (k1 & 15))[m1], d2[m1])
    k2 = np.cumsum(m2, 1) - m2
    put(buf, (8 * start[:, None] + 2 * k2)[m2], d3[m2])
    if w4:
        k3 = np.cumsum(m3, 1) - m3
        put(buf, (8 * (start + r0)[:, None] + w4 * k3)[m3], (r - 9)[m3])
    j = np.cumsum(lit, 1) - lit
    put(buf, (8 * (start + r1)[:, None] + 18 * j)[lit], (k1 | e << 10)[lit])
    assert buf.max() < 256
    blocks = buf[:total].astype(np.uint8)
    bb = np.concatenate([start, end[-1:]]).astype(np.int32)
    sym = [int(x) for x in order[:nsym]] + [0] * 8
    word = lambda a, b, c, d: a | b << 8 | c << 16 | d << 24
    tiers = [word(sym[0], sym[1], sym[2], 2), word(sym[3], sym[4], sym[5], 0)]
    tiers += [word(*sym[6:10])] if w4 == 0 else [word(sym[6], sym[7], sym[8], 0), word(*sym[9:13]), word(*sym[13:17])]
    bits = 8 * (1280 * S + int(size.sum()) + 4 * S) / v.size
    return {"data": data.ravel(), "blocks": blocks, "bb": bb, "tiers": np.array(tiers, np.uint32)}, bits


def pack_rans(W, SS=16):
    """static rANS of the exponent, 32 interleaved lane states a strip of SS steps (decode order: step, weight i, lane)"""
    v = order_steps(W)
    NS = v.shape[0]
    e = ((v >> 7) & 0xFF).astype(np.int64)
    c = np.bincount(e.ravel(), minlength=256)
    q = np.where(c > 0, np.maximum(1, np.round(c / c.sum() * 4096)), 0).astype(np.int64)
    while q.sum() != 4096:
        q[np.argmax(q)] += 4096 - q.sum()
    cum = np.cumsum(q) - q
    table = np.zeros(4096, np.uint32)
    for s in np.nonzero(q)[0]:
        table[cum[s]:cum[s] + q[s]] = s | (q[s] - 1) << 8 | cum[s] << 20
    raw = fixed(v, np.zeros_like(e))[:, 256:]
    nst = -(-NS // SS)
    pad = np.full((nst * SS - NS, 1024), int(np.argmax(c)), np.int64)
    syms = np.concatenate([e, pad]).reshape(nst, SS, 32, 32).transpose(0, 1, 3, 2).reshape(nst, SS * 32, 32)  # [strip, t, lane]
    x = np.full((nst, 32), 1 << 16, np.int64)
    M, Wd = [], []
    for t in range(SS * 32 - 1, -1, -1):
        for lane in range(31, -1, -1):
            s = syms[:, t, lane]
            f = q[s]
            xl = x[:, lane]
            m = xl >= (f << 20)
            M.append(m)
            Wd.append(xl & 0xFFFF)
            xl = np.where(m, xl >> 16, xl)
            x[:, lane] = ((xl // f) << 12) + (xl % f) + cum[s]
    M, Wd = np.stack(M), np.stack(Wd)
    streams = [Wd[M[:, i], i][::-1] for i in range(nst)]
    off = np.concatenate([[0], np.cumsum([len(s) for s in streams])]).astype(np.int64)
    stream = np.concatenate(streams + [np.zeros(64, np.int64)]).astype(np.uint16)
    bits = (8 * raw.size + 16 * int(off[-1]) + 32 * 32 * nst + 64 * nst) / v.size
    return {"raw": raw.ravel(), "states": x.astype(np.uint32).ravel(), "stream": stream, "off": off, "table": table,
            "ss": np.array([SS], np.int32)}, bits


def dump(prefix, W, log):
    O, K = W.shape
    W.astype("<u2").tofile(prefix + ".w")
    open(prefix + ".dims", "w").write(f"{O} {K}")
    for name, (files, bits) in (("v2b", pack_tiered(W, 0)), ("t4a", pack_tiered(W, 2)), ("t4b", pack_tiered(W, 3)), ("rans", pack_rans(W))):
        for k, a in files.items():
            a.tofile(f"{prefix}.{name}.{k}")
        log.write(f"{os.path.basename(prefix)} {name} {bits:.4f}\n")
        print(os.path.basename(prefix), name, f"{bits:.4f} bits a weight", flush=True)


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import f33
    samples = json.load(open(sys.argv[1]))
    od = sys.argv[3]
    os.makedirs(od, exist_ok=True)
    log = open(os.path.join(od, "bits.txt"), "w")
    names = []
    for i in [int(x) for x in sys.argv[2].split(",")]:
        s = samples[i]
        W = f33.read(s, f33.header(s["repo"], s["shard"]))
        dump(os.path.join(od, f"s{i:03d}"), np.ascontiguousarray(W), log)
        names.append(f"s{i:03d}")
    rng = np.random.default_rng(1)
    dump(os.path.join(od, "random"), rng.integers(0, 1 << 16, (256, 1024), dtype=np.uint16), log)  # every slow path
    names.append("random")
    open(os.path.join(od, "list.txt"), "w").write("\n".join(names) + "\n")
