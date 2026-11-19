"""Lossless room in the quantized checkpoints of 2026: codes, scales and bf16 parts, measured on real tensors read by HTTP range.

    python quant.py HEADERS_DIR OUT_DIR [REPO ...]      (HEADERS_DIR/<org>__<name>.json from hdr.py: config + every shard's header)

Per repo: a census of every tensor's bytes by format and part (codes, scales, zeros, bf16, other), from the headers alone; then
10 quantized weight tensors (attention, MLP, experts; seed 0) and 4 bf16 ones, each a slice of whole stored rows (a multiple of
128, about 12 MB), read by HTTP range with no token. For each sample, from the stored bits (nothing lossy):
  codes   order-0 entropy (a table a tensor), Huffman, the best of a few contexts (left, above, the block's scale, the group's zero
          point, a block summary sent as side information), and a fixed-rate layout (decoded with a few integer operations a value);
  scales  order-0, left/above contexts, and a fixed-rate layout (a k-bit window a row or a k-bit index into the tensor's commonest,
          an escape to the value raw; low mantissa bits kept raw for 16- and 32-bit scales);
  bf16    the whole word's order-0 entropy and the tiered code (3 tiers of 2-bit digits over the exponent).
Writes OUT_DIR/<org>__<name>.json: the census, the samples' numbers, and the whole checkpoint projected (each part's bytes at its
measured ratio; a part with no sample stays as stored). FP8 codes also go through fp8ref.pack (the lossless FP8 format as built)
and back through unpack, bit for bit.
"""
import collections, concurrent.futures as cf, glob, json, math, os, random, re, sys, time, traceback, urllib.error, urllib.request

try:
    import numpy as np
    from ent import H, Hc, fast_bits, huffman_bits, tiered_bits
except ImportError:  # --plan reads the headers only
    np = None

# fp8ref.py (the lossless FP8 format's numpy reference, glyd-gpu-py/glyd_gpu/): beside this file or in the tree. The results are of
# its layout 2 (fp8-lossless@42e6999f, blob aafce648; the same file on private main since #15); code_stats checks it is that layout.
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "glyd-gpu-py", "glyd_gpu"))
FP8REF = "glyd-gpu-py/glyd_gpu/fp8ref.py layout 2 (fp8-lossless@42e6999f, blob aafce648, the same file on private main since #15)"

HF = "https://huggingface.co"
DT = {"F64": 8, "F32": 4, "BF16": 2, "F16": 2, "F8_E4M3": 1, "F8_E5M2": 1, "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1,
      "U16": 2, "U32": 4, "F8_E8M0": 1}
UINT = {1: "u1", 2: "<u2", 4: "<u4", 8: "<u8"}
TARGET, ROWS = 12 << 20, 128
QUANT = ("nvfp4", "mxfp4", "fp8", "int4", "int8")
PTR = 32 / 8192  # a 32-bit pointer a 64 x 128 tile, for the fixed-rate layouts that have a variable part
LN2 = math.log(2)


# ---------------- reading ----------------

def get(url, rng, tries=8):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={rng[0]}-{rng[1]}", "User-Agent": "glyd-quant/1"})
            with urllib.request.urlopen(req, timeout=180) as r:
                data = r.read()
            if len(data) != rng[1] - rng[0] + 1:
                raise IOError(f"short read {len(data)}")
            return data
        except urllib.error.HTTPError as e:
            if e.code in (401, 403, 404):
                raise
            time.sleep(min(60, 3 * 2 ** i))
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(min(60, 3 * 2 ** i))


def tensors(hdr):
    """name -> dict(dtype, shape, file, start, end): absolute byte range in its shard."""
    out = {}
    files = hdr["files"]
    if any(not fn.startswith("consolidated") for fn in files):  # a repo with both layouts (Mistral): the Hugging Face one only
        files = {fn: f for fn, f in files.items() if not fn.startswith("consolidated")}
    for fn, f in files.items():
        for k, v in f["header"].items():
            if k == "__metadata__":
                continue
            a, b = v["data_offsets"]
            out[k] = {"dtype": v["dtype"], "shape": v["shape"], "file": fn, "start": 8 + f["hlen"] + a, "end": 8 + f["hlen"] + b}
    return out


def read_rows(t, r0, r1, expert=None):
    """Rows [r0, r1) of a stored tensor (of expert `expert` of a 3-D one) as raw unsigned words of its dtype's width.
    A 0-D tensor is one row of one value, a 1-D one rows of one value."""
    sh = t["shape"]
    es = DT[t["dtype"]]
    rows, cols = (sh[-2], sh[-1]) if len(sh) >= 2 else ((sh[0], 1) if sh else (1, 1))
    lead = (expert or 0) * rows if len(sh) == 3 else 0
    r1 = min(r1, rows)
    a = t["start"] + (lead + r0) * cols * es
    b = t["start"] + (lead + r1) * cols * es - 1
    assert b < t["end"], (t, r0, r1, expert)
    raw = get(f"{HF}/{REPO[0]}/resolve/main/{t['file']}", (a, b))
    return np.frombuffer(raw, dtype=UINT[es]).reshape(r1 - r0, cols)


REPO = [None]


# ---------------- what each tensor is ----------------

SUFFIXES = (".weight_packed", ".weight_scale_inv", ".weight_scale_2", ".weight_scale", ".weight_zero_point", ".weight_shape",
            ".weight_global_scale", ".input_global_scale", ".input_scale", ".k_scale", ".v_scale", ".qweight", ".qzeros", ".scales",
            ".g_idx", ".scale_inv", ".scale", ".weight")


def stem(name):
    for s in SUFFIXES:
        if name.endswith(s):
            return name[: -len(s)], s
    for s in ("_activation_scale", "_scale_inv", "_blocks", "_scales", "_scale"):
        if name.endswith(s):
            return name[: -len(s)], s
    return name, ""


def category(n):
    n = n.lower()
    if n.startswith("mtp") or ".mtp." in n or "nextn" in n:
        return "mtp"
    if any(v in n for v in ("visual", "vision", "mm_projector", "embed_vision", "audio", "multi_modal")):
        return "vision"
    if "lm_head" in n or "unembed" in n:
        return "lm_head"
    if "embed" in n:  # engram.embed (DeepSeek's lookup tables) too; its engram.wkv projections are not embeddings
        return "embed"
    if "shared_expert" in n:
        return "mlp"
    if re.search(r"(^|\.)experts?(\.|_|$)", n) or ".experts." in n:
        return "expert"
    if any(a in n for a in ("attn", "attention", "mixer.")):
        return "attn"
    if any(m in n for m in ("mlp", "ffn", "feed_forward")):
        return "mlp"
    return "other"


def classify(ts, qmethod):
    """name -> (format, part, weight name it belongs to). part: codes, scales, zeros, bf16, other."""
    by_stem = collections.defaultdict(dict)
    for k in ts:
        s, suf = stem(k)
        by_stem[s][suf] = k
    out = {}
    for s, parts in by_stem.items():
        w = parts.get(".weight") or parts.get(".weight_packed") or parts.get(".qweight") or parts.get("")
        fmt = None
        if w is not None:
            t = ts[w]
            dt, sh = t["dtype"], t["shape"]
            sc = parts.get(".weight_scale") or parts.get(".weight_scale_inv") or parts.get(".scales") or parts.get(".scale") or parts.get(".scale_inv") or parts.get("_scale_inv")
            st = ts[sc] if sc else None
            if dt in ("F8_E4M3",) and len(sh) >= 2:
                fmt = "fp8"
            elif w.endswith(".qweight") and dt == "I32":
                fmt = "int4"
            elif w.endswith(".weight_packed") and dt in ("I32", "U8", "I8") and st is not None and st["dtype"] in ("BF16", "F16", "F32"):
                fmt = "int4"
            elif dt in ("U8", "I8") and st is not None and len(sh) >= 2 and len(st["shape"]) >= 2:
                if st["dtype"] == "F8_E4M3" and st["shape"][-1] * 16 == 2 * sh[-1]:
                    fmt = "nvfp4"
                elif st["dtype"] in ("U8", "F8_E8M0") and st["shape"][-1] * 32 == 2 * sh[-1]:
                    fmt = "mxfp4"
                elif dt == "I8" and st["shape"][-1] == 1:
                    fmt = "int8"
            elif dt in ("BF16", "F16") and len(sh) >= 2 and math.prod(sh) >= 65536:
                fmt = "bf16"
        for suf, k in parts.items():
            if fmt in QUANT:
                if k == w:
                    out[k] = (fmt, "codes", w)
                elif suf in (".weight_scale", ".weight_scale_inv", ".scales", ".scale", ".scale_inv", "_scale_inv") and math.prod(ts[k]["shape"]) > 1:
                    out[k] = (fmt, "scales", w)
                elif suf in (".weight_zero_point", ".qzeros"):
                    out[k] = (fmt, "zeros", w)
                else:
                    out[k] = ("other", "other", None)
            elif fmt == "bf16" and k == w:
                out[k] = ("bf16", "bf16", w)
            else:
                out[k] = ("other", "other", None)
    return out


def census(ts, cls):
    tot = collections.defaultdict(lambda: [0, 0])
    for k, t in ts.items():
        f, p, _ = cls[k]
        key = f"{f}:{p}:{category(k)}"
        tot[key][0] += 1
        tot[key][1] += t["end"] - t["start"]
    return {k: {"tensors": v[0], "bytes": v[1]} for k, v in sorted(tot.items())}


def f16_bytes(ts, cls):
    """category -> bytes of the 16-bit parts stored as fp16 (not bf16)."""
    out = collections.Counter()
    for k, t in ts.items():
        if cls[k][0] == "bf16" and t["dtype"] == "F16":
            out[category(k)] += t["end"] - t["start"]
    return dict(out)


# ---------------- choosing samples ----------------

def stored_rows(t):
    sh = t["shape"]
    return sh[-2], sh[-1], (sh[0] if len(sh) == 3 else None)


def choose(ts, cls, quota, fmts, rnd):
    """[(format, category, weight name, r0, r1, expert)] : `quota` samples over the (format, category) groups by bytes."""
    groups = collections.defaultdict(list)
    for k, (f, p, w) in sorted(cls.items()):
        if p in ("codes", "bf16") and f in fmts and len(ts[k]["shape"]) in (2, 3):
            groups[(f, category(k))].append(k)
    size = {g: sum(ts[k]["end"] - ts[k]["start"] for k in v) for g, v in groups.items()}
    tot = sum(size.values()) or 1
    order = sorted(size, key=lambda g: -size[g])
    take = {g: 1 for g in order if size[g] / tot >= 0.03}
    if len(take) > quota:
        take = {g: 1 for g in order[:quota]}
    while sum(take.values()) < quota and order:
        g = max(order, key=lambda g: size[g] / tot * quota - take.get(g, 0))
        take[g] = take.get(g, 0) + 1
    out = []
    for g in order:
        names = list(groups[g])
        for _ in range(min(take.get(g, 0), len(names))):
            name = names.pop(rnd.randrange(len(names)))
            t = ts[name]
            R, C, E = stored_rows(t)
            per_row = C * DT[t["dtype"]]
            n = max(ROWS, TARGET // per_row // ROWS * ROWS)
            n = min(n, R // ROWS * ROWS) if R >= ROWS else R
            r0 = rnd.randrange(0, R - n + 1) // ROWS * ROWS if R > n else 0
            out.append((g[0], g[1], name, r0, r0 + n, rnd.randrange(E) if E else None))
    return out


# ---------------- unpacking to codes [out rows, in columns] + scales with their block ----------------

def nibbles(u8):
    """uint8 [r, c] -> uint8 [r, 2c], the low nibble first."""
    q = np.empty((u8.shape[0], u8.shape[1] * 2), np.uint8)
    q[:, 0::2] = u8 & 15
    q[:, 1::2] = u8 >> 4
    return q


def int32_nibbles(w32, order=None):
    """uint32 [r, c] -> uint8 [r, 8c]: value k at bits 4k; with `order`, the value at bits 4k goes to place order[k] of its 8."""
    v = ((w32[:, :, None] >> (4 * np.arange(8, dtype=np.uint32))[None, None, :]) & 15).astype(np.uint8)
    if order is not None:
        o = np.empty_like(v)
        o[:, :, order] = v
        v = o
    return v.reshape(w32.shape[0], -1)


AWQ_ORDER = [0, 2, 4, 6, 1, 3, 5, 7]


def scale_rows(st, R_full, r0, r1, expert):
    """The rows of scale tensor st that cover weight rows [r0, r1), and the block height (a scalar: one block of all rows)."""
    sh = st["shape"]
    if not sh or int(np.prod(sh)) == 1:
        return read_rows(st, 0, 1), R_full
    Rs = sh[-2] if len(sh) >= 2 else sh[0]
    bh = -(-R_full // Rs)
    return read_rows(st, r0 // bh, (r1 - 1) // bh + 1, expert), bh


def load(ts, cls, qmethod, s):
    """A sample -> dict(Q codes uint8 [R, C], nb bits, S raw scale words [Rs, Cs] or None, bh, bw, sw scale width, sdt, Z or None)."""
    fmt, cat, name, r0, r1, e = s
    t = ts[name]
    parts = {cls[k][1]: k for k in ts if cls[k][2] == name and k != name}
    W = read_rows(t, r0, r1, e)
    R_full = t["shape"][-2]
    st = ts[parts["scales"]] if "scales" in parts else None
    Z = None
    if fmt == "bf16":
        return {"W16": W}
    if fmt in ("nvfp4", "mxfp4"):
        Q, nb = nibbles(W.view(np.uint8) if W.dtype != np.uint8 else W), 4
        S, bh = scale_rows(st, R_full, r0, r1, e)
    elif fmt in ("fp8", "int8"):
        Q, nb = W.view(np.uint8), 8
        S, bh = scale_rows(st, R_full, r0, r1, e) if st is not None else (None, 1)
    elif fmt == "int4" and name.endswith(".qweight"):
        # gptq: qweight [C/8, R], storage rows are 8 input channels each; awq: qweight [C, R/8], storage rows are input channels
        if qmethod == "awq":
            Qt = int32_nibbles(W.view(np.uint32), AWQ_ORDER)  # [in rows, R]
            in0, in1 = r0, r0 + Qt.shape[0]
        else:
            v = int32_nibbles(W.view(np.uint32))  # [rows, R*8]: value k of word (i, r) at [i, 8r + k]
            rows, R8 = W.shape[0], W.shape[1]
            Qt = v.reshape(rows, R8, 8).transpose(0, 2, 1).reshape(rows * 8, R8)  # input channel 8i + k, output r
            in0, in1 = 8 * r0, 8 * r0 + Qt.shape[0]
        Q, nb = np.ascontiguousarray(Qt.T), 4
        C_in = (t["shape"][0] * (8 if qmethod != "awq" else 1))
        ssh = st["shape"]
        g = C_in // ssh[0]
        S = np.ascontiguousarray(read_rows(st, in0 // g, (in1 - 1) // g + 1).T)
        bh = 1
        if "zeros" in parts:
            zt = ts[parts["zeros"]]
            zw = read_rows(zt, in0 // g, (in1 - 1) // g + 1).view(np.uint32)
            Z = np.ascontiguousarray(int32_nibbles(zw, AWQ_ORDER if qmethod == "awq" else None).T)
        bw = g
        sw = DT[st["dtype"]]
        return {"Q": Q, "nb": nb, "S": S, "bh": bh, "bw": bw, "sw": sw, "sdt": st["dtype"], "Z": Z, "zbh": 1, "zbw": g}
    elif fmt == "int4":  # compressed-tensors pack-quantized: weight_packed int32 [R, C/8], value k at bits 4k
        Q, nb = int32_nibbles(W.view(np.uint32)) if W.dtype == np.uint32 else nibbles(W.view(np.uint8)), 4
        S, bh = scale_rows(st, R_full, r0, r1, e)
        if "zeros" in parts:
            zt = ts[parts["zeros"]]
            zr = zt["shape"][-2]
            if zr * 8 == R_full and zt["dtype"] == "I32":  # packed 8 rows a word, along the rows
                zw = read_rows(zt, r0 // 8, -(-r1 // 8), e).view(np.uint32)
                Z = ((zw[:, None, :] >> (4 * np.arange(8, dtype=np.uint32))[None, :, None]) & 15).astype(np.uint8).reshape(-1, zw.shape[1])
            else:
                Z = read_rows(zt, r0, r1, e).astype(np.uint8)
    else:
        raise ValueError(fmt)
    R, C = Q.shape
    bw = -(-C // S.shape[1]) if S is not None else C
    if S is not None and S.shape[0] * bh < R:
        bh = -(-R // S.shape[0])
    return {"Q": Q, "nb": nb, "S": S, "bh": bh, "bw": bw, "sw": DT[st["dtype"]] if st else 0, "sdt": st["dtype"] if st else None, "Z": Z,
            "zbh": 1, "zbw": bw}


# ---------------- measuring ----------------

def bc(a, n):
    return np.bincount(a.ravel(), minlength=n).astype(np.int64)


def expand(S, bh, bw, R, C):
    return np.repeat(np.repeat(S, bh, axis=0)[:R], bw, axis=1)[:, :C]


def scale_bucket(S, sdt):
    """A small context from a scale: the byte for 8-bit scales, the exponent field for 16- and 32-bit ones."""
    S = S.astype(np.int64)
    if sdt in ("BF16",):
        return (S >> 7) & 255, 256
    if sdt == "F16":
        return (S >> 10) & 31, 32
    if sdt == "F32":
        return (S >> 23) & 255, 256
    return S & 255, 256


MIN_PER_CELL = 10  # a context's entropy counts only with at least this many values a nonzero cell of its joint histogram


def Hc_dense(joint):
    """H(y | x) of a dense joint histogram [x, y], or None when it has too few values a cell to be estimated."""
    j = np.asarray(joint)
    return Hc(j) if j.sum() >= MIN_PER_CELL * (j > 0).sum() else None


def Hc_sparse(ctx, y):
    """H(y | ctx) for integer arrays of any range (plug-in + Miller-Madow), or None when too sparse (see MIN_PER_CELL)."""
    _, c = np.unique(ctx.ravel(), return_inverse=True)
    _, v = np.unique(y.ravel(), return_inverse=True)
    n = v.size
    _, j = np.unique(c.astype(np.int64) * (int(v.max()) + 1) + v, return_counts=True)
    m = np.bincount(c)
    m = m[m > 0]
    if n < MIN_PER_CELL * len(j):
        return None
    return H(j) - H(m) + (len(j) - len(m)) / (2 * n * LN2)


def H_sparse(y):
    _, c = np.unique(y.ravel(), return_counts=True)
    return H(c) + (len(c) - 1) / (2 * c.sum() * LN2)


def fixed_codes(h, nb):
    """Fixed-rate layouts for k-bit codes with a table of the tensor's own: as stored, k-bit window into the commonest + escape,
    tiers of 2- or 3-bit digits; returns (bits a value, name)."""
    cands = [(float(nb), "as stored")]
    for k in range(1, nb):
        b, _ = fast_bits(h, nb, k)
        cands.append((b + PTR, f"{k} bits + escape"))
    for d in (1, 2, 3):
        for t in (1, 2, 3, 4):
            if (2 ** d - 1) * t < 2 ** nb:
                b, _ = tiered_bits(h, nb, t, d)
                cands.append((b + PTR, f"tiers {t}x{d}-bit"))
    return min(cands)


def int8_float_like(q):
    """INT8 as sign, a magnitude class (bit length of |x|, 0..8) coded by tiers, the bits below the leading one raw."""
    x = q.view(np.int8).astype(np.int16).ravel()
    m = np.abs(x)
    cl = np.where(m == 0, 0, np.floor(np.log2(np.maximum(m, 1))).astype(np.int16) + 1)  # 0..8
    h = bc(cl.astype(np.int64), 9)
    raw = float(np.maximum(cl - 1, 0).mean())
    tb = min(tiered_bits(h, 4, t, d)[0] for d in (1, 2) for t in (1, 2, 3) if (2 ** d - 1) * t < 9)
    return 1 + raw + tb + PTR, float(H(h)) + 1 + raw


def code_stats(d):
    Q, nb = d["Q"], d["nb"]
    A = 1 << nb
    q = Q.astype(np.int32)
    R, C = q.shape
    n = q.size
    h = bc(q, A)
    o = {"n": n, "H0": H(h), "huffman": huffman_bits(h)}
    ctx = {}
    ctx["left"] = Hc_dense(bc(q[:, :-1] * A + q[:, 1:], A * A).reshape(A, A))
    if R > 1:
        ctx["above"] = Hc_dense(bc(q[:-1] * A + q[1:], A * A).reshape(A, A))
    if d.get("S") is not None and d["S"].size > 1:
        sb, ns = scale_bucket(d["S"], d["sdt"])
        e = expand(sb, d["bh"], d["bw"], R, C)
        ctx["scale"] = Hc_dense(bc(e * A + q, ns * A).reshape(ns, A))
    if d.get("Z") is not None:
        e = expand(d["Z"].astype(np.int32), d["zbh"], d["zbw"], R, C)
        ctx["zero"] = Hc_dense(bc(e * A + q, 16 * A).reshape(16, A))
    if nb == 4 and d.get("S") is not None and d["bw"] in (16, 32) and C % d["bw"] == 0:
        # a block summary sent as side information: the block's largest magnitude (fp4: the low 3 bits; int4: |code - 8|)
        mag = (q & 7) if d["fmt"] in ("nvfp4", "mxfp4") else np.abs(q - 8)
        bm = mag.reshape(R, C // d["bw"], d["bw"]).max(2)
        e = np.repeat(bm, d["bw"], axis=1)
        nbm = int(bm.max()) + 1
        hb = Hc_dense(bc(e * A + q, nbm * A).reshape(nbm, A))
        ctx["block max (side info counted)"] = None if hb is None else hb + H(bc(bm, nbm)) / d["bw"]
    ctx = {k: v for k, v in ctx.items() if v is not None}
    o["contexts"] = ctx
    o["ceiling"] = min([o["H0"]] + list(ctx.values()))
    if nb == 8 and d["fmt"] == "fp8":
        import fp8ref
        assert getattr(fp8ref, "HEAD_WORDS", None) == 8 and fp8ref.GUARD == 8, "fp8ref is not layout 2: " + FP8REF
        p = fp8ref.pack(Q)
        assert np.array_equal(fp8ref.unpack(p), Q), "fp8ref round trip"
        o["fixed"], o["fixed_name"] = p.bits_per_value(), "fp8ref (as built), round trip exact"
        ex = bc(q >> 3 & 15, 16)
        o["H_sign_exp_plus_mantissa"] = H(bc(q >> 3, 32)) + 3
    elif nb == 8:
        b, h_cls = int8_float_like(Q)
        hi = bc(q >> 4, 16)
        tb = min(tiered_bits(hi, 4, t, dd)[0] for dd in (1, 2, 3) for t in (1, 2, 3) if (2 ** dd - 1) * t < 16) + 4 + PTR
        o["fixed"], o["fixed_name"] = min((b, "sign + class by tiers + low bits raw"), (tb, "high nibble by tiers + low nibble raw"))
        o["H_class_floor"] = h_cls
    else:
        o["fixed"], o["fixed_name"] = fixed_codes(h, nb)
    return o


def window_rows(hi, W):
    """Per row of hi [r, c] (ints), the most values inside a window of W consecutive values: summed over rows."""
    r, c = hi.shape
    s = np.sort(hi, axis=1).astype(np.int64)
    off = (np.arange(r, dtype=np.int64) * (int(s.max()) - int(s.min()) + W + 2))[:, None]
    g = (s - s.min() + off).ravel()
    cnt = np.searchsorted(g, g + W - 1, side="right") - np.arange(g.size)
    return int(cnt.reshape(r, c).max(1).sum())


def scale_stats(S, sdt, sw):
    v = S.astype(np.int64)
    n = v.size
    w = 8 * sw
    m = {"BF16": 7, "F16": 10, "F32": 23}.get(sdt, 0)
    hi = v >> m  # 8-bit scales whole; 16- and 32-bit ones: sign and exponent modelled, the m mantissa bits counted as raw bits
    o = {"n": n, "width": w, "raw_low_bits": m, "H0": H_sparse(hi) + m}
    ctx = {}
    if v.shape[1] > 1:
        ctx["left"] = Hc_sparse(hi[:, :-1], hi[:, 1:])
    if v.shape[0] > 1:
        ctx["above"] = Hc_sparse(hi[:-1], hi[1:])
    ctx = {k: x + m for k, x in ctx.items() if x is not None}
    o["contexts"] = ctx
    o["ceiling"] = min([o["H0"]] + list(ctx.values()))
    cands = [(float(w), "as stored")]
    _, hh = np.unique(hi, return_counts=True)
    for k in range(1, 8):
        b, _ = fast_bits(hh, w - m, k)
        cands.append((b + m, f"{k}-bit index into the tensor's commonest + escape" + (f", {m} low bits raw" if m else "")))
        if v.shape[1] >= 16:
            inside = window_rows(hi, 2 ** k - 1)
            esc = 1 - inside / n
            cands.append((k + esc * (w - m) + m + (w - m) / v.shape[1], f"{k}-bit window a row + escape" + (f", {m} low bits raw" if m else "")))
    o["fixed"], o["fixed_name"] = min(cands)
    return o


def zero_stats(Z):
    h = bc(Z.astype(np.int64), 16)
    b, name = fixed_codes(h, 4)
    return {"n": int(Z.size), "H0": H(h), "ceiling": H(h), "fixed": min(b, 4.0), "fixed_name": name}


def bf16_stats(W16, dtype):
    """A 16-bit float tensor (bf16: 8 exponent bits, fp16: 5): the word's order-0 entropy and the tiered code over the exponent."""
    w = W16.ravel()
    word = bc(w.astype(np.int64), 65536)
    n = int(word.sum())
    eb = 5 if dtype == "F16" else 8
    ex = word.reshape(2, 2 ** eb, 2 ** (15 - eb)).sum((0, 2))
    tb, tr = tiered_bits(ex, eb, 3)
    tb += 16 - eb
    h0 = H(word) + ((word > 0).sum() - 1) / (2 * n * LN2)
    return {"n": n, "H0": h0, "ceiling": h0, "H_exp": H(ex), "fixed": tb, "fixed_name": "tiers 3x2-bit over the exponent"}


def measure(args):
    repo, ts, cls, qm, s = args
    REPO[0] = repo
    t0 = time.time()
    try:
        d = load(ts, cls, qm, s)
        rec = {"format": s[0], "category": s[1], "tensor": s[2], "rows": [s[3], s[4]], "expert": s[5], "shape": list(ts[s[2]]["shape"]),
               "dtype": ts[s[2]]["dtype"]}
        if s[0] == "bf16":
            rec["bf16"] = bf16_stats(d["W16"], ts[s[2]]["dtype"])
        else:
            d["fmt"] = s[0]
            rec["codes"] = code_stats(d)
            rec["codes_shape"] = list(d["Q"].shape)
            if d.get("S") is not None and d["S"].size >= 64:
                rec["scales"] = scale_stats(d["S"], d["sdt"], d["sw"])
                rec["scale_block"] = [int(d["bh"]), int(d["bw"])]
                rec["scale_dtype"] = d["sdt"]
            if d.get("Z") is not None:
                rec["zeros"] = zero_stats(d["Z"])
        rec["seconds"] = round(time.time() - t0, 1)
        return rec
    except Exception as e:
        return {"format": s[0], "category": s[1], "tensor": s[2], "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-1500:]}


def qmethod(cfg):
    qc = cfg.get("quantization_config") or (cfg.get("text_config") or {}).get("quantization_config") or {}
    return (qc.get("quant_method") or qc.get("format") or "").lower()


# ---------------- the whole checkpoint ----------------

def pooled(recs, fmt, part, cat=None):
    """bits a stored bit for each layout, pooled over the samples of format fmt (of category cat, if given; weighted by values)."""
    rows = []
    for r in recs:
        if r.get("format") != fmt or "error" in r or (cat and category(r["tensor"]) != cat):
            continue
        x = r.get(part if part != "bf16" else "bf16")
        if not x:
            continue
        w = {"bf16": 16, "zeros": 4}.get(part) or (x["width"] if part == "scales" else (4 if fmt in ("nvfp4", "mxfp4", "int4") else 8))
        rows.append((x["n"], w, x))
    if not rows:
        return None
    N = sum(n * w for n, w, _ in rows)
    return {k: sum(n * x[k] for n, w, x in rows) / N for k in ("H0", "ceiling", "fixed")} | {"samples": len(rows)}


def project(cen, recs):
    """Each (format, part, category) of the census at the ratio its own samples measured; a group with none at the ratio of the
    repo's samples of that format and part (pooled by values); a part with no sample at all (norms, biases, ...) as stored."""
    total = sum(v["bytes"] for v in cen.values())
    out = {"stored": total}
    for lay in ("fixed", "H0", "ceiling"):
        size, parts = 0.0, collections.defaultdict(lambda: {"bytes": 0, "size": 0.0, "unsampled_bytes": 0})
        for key, v in cen.items():
            f, p, c = key.split(":")
            r = (pooled(recs, f, p, c) or pooled(recs, f, p)) if p in ("codes", "scales", "zeros", "bf16") else None
            ratio = r[lay] if r else 1.0
            a = parts[f"{f}:{p}"]
            a["bytes"] += v["bytes"]
            a["size"] += v["bytes"] * ratio
            if r and not pooled(recs, f, p, c):
                a["unsampled_bytes"] += v["bytes"]
            size += v["bytes"] * ratio
        parts = {k: {"bytes": a["bytes"], "ratio": a["size"] / a["bytes"] if a["bytes"] else 1.0, "unsampled_bytes": a["unsampled_bytes"]}
                 for k, a in parts.items()}
        out[lay] = {"bytes": size, "saved": 1 - size / total, "parts": parts}
    return out


def reproject(hdir, rdir):
    """The census and the projection again, from the headers and the samples' numbers already measured (no tensor is read):
    python quant.py --reproject HEADERS_DIR RESULTS_DIR."""
    for path in sorted(glob.glob(os.path.join(rdir, "*__*.json"))):
        res = json.load(open(path))
        hdr = json.load(open(os.path.join(hdir, os.path.basename(path))))
        ts = tensors(hdr)
        cls = classify(ts, qmethod(hdr["config"]))
        cen = census(ts, cls)
        old, new = collections.Counter(), collections.Counter()
        for c, src in ((old, res["census"]), (new, cen)):
            for k, v in src.items():
                c[k.rsplit(":", 1)[0]] += v["bytes"]
        assert old == new, (path, "the headers differ from the run's", old - new, new - old)
        for r in res["samples"]:
            r["category"] = category(r["tensor"])
        was = res["projection"]["fixed"]["saved"]
        res |= {"census": cen, "f16_bytes": f16_bytes(ts, cls), "projection": project(cen, res["samples"]), "fp8ref": FP8REF}
        json.dump(res, open(path, "w"), indent=1)
        print(f"{res['repo']}: fixed rate {100 * was:.1f}% -> {100 * res['projection']['fixed']['saved']:.1f}%")


PLAN = "--plan" in sys.argv


def main():
    args = [a for a in sys.argv[1:] if a != "--plan"]
    hdir, odir = args[0], args[1]
    os.makedirs(odir, exist_ok=True)
    repos = args[2:] or [l.strip() for l in open(os.path.join(hdir, "..", "repos.txt")) if l.strip()]
    procs = int(os.environ.get("PROCS", os.cpu_count() or 4))
    for repo in repos:
        path = os.path.join(hdir, repo.replace("/", "__") + ".json")
        out = os.path.join(odir, repo.replace("/", "__") + ".json")
        if os.path.exists(out) and not PLAN:
            continue
        t0 = time.time()
        hdr = json.load(open(path))
        ts = tensors(hdr)
        qm = qmethod(hdr["config"])
        cls = classify(ts, qm)
        cen = census(ts, cls)
        rnd = random.Random(0)
        samples = choose(ts, cls, 10, QUANT, rnd) + choose(ts, cls, 4, ("bf16",), rnd)
        if PLAN:  # the census and the samples only: no tensor is read
            tot = sum(v["bytes"] for v in cen.values())
            print(f"== {repo} ({qm}) {tot / 1e9:.2f} GB")
            for k, v in cen.items():
                if v["bytes"] / tot > 0.002:
                    print(f"   {100 * v['bytes'] / tot:5.1f}%  {k}")
            for s in samples:
                print("   sample", s[:2], s[2], s[3:], ts[s[2]]["shape"], {cls[k][1]: ts[k]["shape"] for k in cls if cls[k][2] == s[2] and k != s[2]})
            continue
        with cf.ProcessPoolExecutor(min(procs, len(samples))) as ex:
            jobs = []
            for s in samples:
                mine = [k for k in cls if k == s[2] or cls[k][2] == s[2]]
                jobs.append((repo, {k: ts[k] for k in mine}, {k: cls[k] for k in mine}, qm, s))
            recs = list(ex.map(measure, jobs))
        res = {"repo": repo, "quant_method": qm, "fp8ref": FP8REF, "census": cen, "f16_bytes": f16_bytes(ts, cls), "samples": recs, "projection": project(cen, recs), "seconds": round(time.time() - t0)}
        json.dump(res, open(out, "w"), indent=1)
        p = res["projection"]
        errs = sum(1 for r in recs if "error" in r)
        print(f"{repo}: {p['stored'] / 1e9:.2f} GB; saved fixed {100 * p['fixed']['saved']:.1f}%, order-0 {100 * p['H0']['saved']:.1f}%, "
              f"ceiling {100 * p['ceiling']['saved']:.1f}%; {len(recs)} samples, {errs} errors, {res['seconds']} s", flush=True)


def selftest():
    """The unpacking and the window count against hand-packed data."""
    rng = np.random.default_rng(0)
    q = rng.integers(0, 16, (4, 64), dtype=np.uint8)
    assert np.array_equal(nibbles((q[:, 0::2] | (q[:, 1::2] << 4)).astype(np.uint8)), q)
    for order in (None, AWQ_ORDER):
        w = np.zeros((4, 8), np.uint32)
        for k in range(8):
            w |= q[:, (order[k] if order else k)::8].astype(np.uint32) << np.uint32(4 * k)
        assert np.array_equal(int32_nibbles(w, order), q), order
    assert window_rows(np.array([[1, 2, 3, 10], [5, 5, 6, 9]]), 3) == 6
    assert abs(H(np.ones(16)) - 4) < 1e-12 and abs(Hc_sparse(np.zeros(1000, int), np.arange(1000) % 4) - 2) < 0.01
    assert int8_float_like(np.zeros((128, 256), np.uint8))[0] < 3
    print("selftest ok")


if __name__ == "__main__":
    if sys.argv[1:2] == ["--reproject"]:
        reproject(*sys.argv[2:4])
    else:
        selftest() if sys.argv[1:] == ["--selftest"] else main()
