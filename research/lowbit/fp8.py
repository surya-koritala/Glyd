"""FP8 (e4m3: sign, 4 exponent bits, 3 mantissa bits) weights with one scale a 128x128 block (DeepSeek-V3,
Qwen3-FP8): the histograms the README's entropies come from.

    python fp8.py FILE.safetensors OUT.json [--procs P]

Every F8_E4M3 tensor with a `<name>_scale_inv` beside it, counted by kind (role:projection). Counts are of the
stored bytes; nothing is lossy. A code byte is s eeee mmm (bit 7 the sign, bits 3-6 the exponent field).
"""
import json, multiprocessing as mp, re, sys, time
import numpy as np
from st import ST
from ent import H, Hc, huffman_bits, fast_bits, tiered_bits

CH = 2048  # rows a chunk (a multiple of 128)


def bc(a, n):
    return np.bincount(a.ravel(), minlength=n).astype(np.int64)


def scale_f32(S):
    S = np.asarray(S)
    if S.dtype == np.uint16:  # bf16 bits
        return (S.astype(np.uint32) << 16).view(np.float32)
    return S.astype(np.float32)


def add(o, k, v):
    o[k] = o[k] + v if k in o else v


def counts(W, S):
    """Histograms of one matrix W [R, C] (uint8 codes) with block scales S [ceil(R/128), ceil(C/128)]."""
    R, C = W.shape
    Sf = scale_f32(S)
    Rb, Cb = Sf.shape
    assert Rb == -(-R // 128) and Cb == -(-C // 128), (W.shape, Sf.shape)
    lg = np.log2(np.maximum(Sf, 1e-30))
    sab = (np.clip(np.floor(lg), -24, 7) + 24).astype(np.int32)  # absolute scale bucket (octave), 32
    sre = (np.clip(np.round(2 * (lg - np.median(lg))), -8, 8) + 8).astype(np.int32)  # relative to the matrix's median, half octaves, 17
    bj = np.arange(C) // 128
    cols = np.arange(C, dtype=np.int32)
    colh = np.zeros(C * 16, np.int64)
    o = {}
    loc = {k: 0 for k in ("el", "eu", "elu", "scab", "scre")}  # this matrix's own, for the per-matrix conditional entropies
    hrow_sum = 0.0
    for r0 in range(0, R, CH):
        r1 = min(R, r0 + CH)
        w = np.asarray(W[r0:r1])
        e = ((w >> 3) & 15).astype(np.int32)
        h = r1 - r0
        add(o, "code", bc(w, 256))
        loc["el"] = loc["el"] + bc(e[:, :-1] * 16 + e[:, 1:], 256)  # (left exponent, this)
        if r0 == 0:
            cur, up = e[1:], e[:-1]
        else:
            cur, up = e, ((np.asarray(W[r0 - 1:r1 - 1]) >> 3) & 15).astype(np.int32)
        loc["eu"] = loc["eu"] + bc(up * 16 + cur, 256)  # (exponent above, this)
        loc["elu"] = loc["elu"] + bc(cur[:, :-1] * 256 + up[:, 1:] * 16 + cur[:, 1:], 4096)  # (left, above, this)
        rows = np.arange(r0, r1)
        bi = rows // 128
        loc["scab"] = loc["scab"] + bc(sab[bi][:, bj] * 16 + e, 32 * 16)  # (absolute scale bucket, exponent)
        loc["scre"] = loc["scre"] + bc(sre[bi][:, bj] * 16 + e, 17 * 16)  # (scale relative to the matrix's median, exponent)
        add(o, "col128", bc((cols % 128)[None, :] * 16 + e, 128 * 16))  # (column in block, exponent)
        add(o, "row128", bc((rows % 128).astype(np.int32)[:, None] * 16 + e, 128 * 16))
        # the block's largest exponent as side information
        starts_r = np.arange(0, h, 128)
        me = np.maximum.reduceat(np.maximum.reduceat(e, starts_r, axis=0), np.arange(0, C, 128), axis=1)  # [blocks, Cb]
        add(o, "bmx_hist", bc(me, 16))
        add(o, "bmx", bc(np.repeat(np.repeat(me, 128, axis=0)[:h], 128, axis=1)[:, :C] * 16 + e, 256))
        # per-column table (a table for each column of the matrix) and per-row table
        colh += bc(cols[None, :] * 16 + e, C * 16)
        rh = bc(np.arange(h, dtype=np.int32)[:, None] * 16 + e, h * 16).reshape(h, 16).astype(np.float64)
        n = rh.sum(1)
        p = rh / n[:, None]
        hr = -(np.where(p > 0, p * np.log2(np.where(p > 0, p, 1)), 0)).sum(1) + ((rh > 0).sum(1) - 1) / (2 * n * np.log(2))
        hrow_sum += float((n * hr).sum())
    for k, v in loc.items():  # pooled counts, and this matrix's own conditional entropy (a table a matrix)
        add(o, k, v)
    code = o["code"]
    n = int(code.sum())
    o["perm_ctx"] = np.array([n] + [n * Hc(loc[k].reshape(sh)) for k, sh in (("el", (16, 16)), ("eu", (16, 16)), ("elu", (256, 16)),
                                                                           ("scab", (32, 16)), ("scre", (17, 16)))])
    ex = code.reshape(2, 16, 8).sum((0, 2))  # exponent field histogram
    o["perm_exp"] = np.array([n, n * H(ex)])
    o["perm_code"] = np.array([n, n * H(code)])
    o["colcond"] = np.array([n, n * Hc(colh.reshape(C, 16))])  # H(exp | column), a table a column (Miller-Madow corrected)
    o["rowcond"] = np.array([n, hrow_sum])
    # the exponent coded three ways with a table of the matrix's own: Huffman, 3 bits + escape, tiers of 2-bit digits
    fb, fe = fast_bits(ex, 4, 3)
    tb2, tr2 = tiered_bits(ex, 4, 2)
    tb3, tr3 = tiered_bits(ex, 4, 3)
    o["lay"] = np.array([n, n * huffman_bits(ex), n * fb, n * fe, n * tb2, n * tr2, n * tb3, n * tr3,
                         n * ex[:8].sum() / n, n * ex[:12].sum() / n])  # n, huff, fast3 bits, fast3 escapes, tier2, tier2 raw, tier3, tier3 raw, e<8, e<12
    o["exp"] = ex.astype(np.int64)
    return o


def role(name):
    if "shared_experts" in name:
        r = "shared"
    elif ".mlp.experts." in name:
        r = "routed"
    elif "self_attn" in name:
        r = "attention"
    elif ".mlp." in name:
        r = "dense_mlp"
    else:
        r = "other"
    return r + ":" + name.split(".")[-2]


def work(args):
    path, names = args
    f = ST(path)
    acc = {}
    for name in names:
        o = counts(f.get(name), f.get(name + "_scale_inv"))
        a = acc.setdefault(role(name), {})
        for k, v in o.items():
            add(a, k, v)
        a["matrices"] = a.get("matrices", 0) + 1
    return acc


if __name__ == "__main__":
    path, out = sys.argv[1], sys.argv[2]
    procs = int(sys.argv[sys.argv.index("--procs") + 1]) if "--procs" in sys.argv else 6
    f = ST(path)
    names = [k for k in f.keys() if f.info(k)[0] == "F8_E4M3" and k + "_scale_inv" in f.hdr]
    names.sort(key=lambda k: -int(np.prod(f.info(k)[1])))  # biggest first for balance
    t0 = time.time()
    with mp.Pool(procs) as p:
        parts = p.map(work, [(path, names[i::procs * 3]) for i in range(procs * 3)])
    tot = {}
    for acc in parts:
        for kind, a in acc.items():
            t = tot.setdefault(kind, {})
            for k, v in a.items():
                add(t, k, v)
    res = {kind: {k: v.tolist() if hasattr(v, "tolist") else v for k, v in a.items()} for kind, a in tot.items()}
    res["_meta"] = {"file": path.split("/")[-1], "matrices": len(names), "seconds": time.time() - t0}
    json.dump(res, open(out, "w"))
    print(f"{len(names)} matrices in {time.time() - t0:.0f} s -> {out}")
