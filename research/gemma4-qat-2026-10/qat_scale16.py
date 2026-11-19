"""Gemma 4 31B QAT, for the exact 4-bit layout: per group of 32, codes k and a scale s with bf16(k * s) (fp32 product, exact) equal to the
weights bit for bit. Reports the significand bits the scales need, the codes' range with s > 0 (does any group need +8?), negative zeros,
and each tensor's scale exponents.  Slices fetched by HTTP range from the Hub (anonymous), cached beside this file.
    python -I qat_scale16.py [ROWS]"""
import json, os, struct, sys, urllib.request
import numpy as np

REPO = "google/gemma-4-31B-it-qat-q4_0-unquantized"
HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "qat-slices")
ROWS = int(sys.argv[1]) if len(sys.argv) > 1 else 512
os.makedirs(CACHE, exist_ok=True)

def get(url, a=None, b=None):
    rq = urllib.request.Request(url, headers={"Range": f"bytes={a}-{b}"} if a is not None else {})
    with urllib.request.urlopen(rq) as r:
        return r.read()

base = f"https://huggingface.co/{REPO}/resolve/main/"
ix = json.loads(get(base + "model.safetensors.index.json"))["weight_map"]
hdrs = {}

def tensor(name):
    path = os.path.join(CACHE, name.replace("/", "_") + f".{ROWS}.npy")
    if os.path.exists(path):
        return np.load(path)
    f = ix[name]
    if f not in hdrs:
        n = struct.unpack("<Q", get(base + f, 0, 7))[0]
        hdrs[f] = (json.loads(get(base + f, 8, 8 + n - 1)), 8 + n)
    h, off = hdrs[f]
    t = h[name]; R, C = t["shape"]; a0, _ = t["data_offsets"]
    rows = min(ROWS, R)
    w = np.frombuffer(get(base + f, off + a0, off + a0 + rows * C * 2 - 1), np.uint16).reshape(rows, C)
    np.save(path, w)
    return w

def f32(w): return (w.astype(np.uint32) << 16).view(np.float32)
def bf16(x):
    b = np.asarray(x, np.float32).view(np.uint32)
    return ((b + 0x7FFF + ((b >> 16) & 1)) >> 16).astype(np.uint16)

def analyse(name):
    w = tensor(name).reshape(-1, 32)
    v = f32(w).astype(np.float64)
    m = np.abs(v).max(1, keepdims=True)
    G = len(w)
    need = np.full(G, 99); need[m[:, 0] == 0] = 0
    expo = np.zeros(G, np.int64); kmin = np.zeros(G, np.int64); kmax = np.zeros(G, np.int64)
    negzero = int(((w == 0x8000).any(1)).sum())
    for den in (8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 7.5, 6.5):
        todo = need == 99
        if not todo.any(): break
        vv, mm, ww = v[todo], m[todo], w[todo]
        k = np.rint(vv / (mm / den))
        ok = ~((k == 0) & (vv != 0)).any(1) & (np.abs(k).max(1) <= 8)
        # the exact bf16 rounding cell of each value; a power of two's cell toward zero is half as wide
        ulp = np.exp2(np.floor(np.log2(np.where(vv == 0, 1, np.abs(vv)))) - 7)
        down = np.where((ww & 0x7F) == 0, 0.25, 0.5) * ulp
        lo = np.where(vv > 0, vv - down, vv - 0.5 * ulp); hi = np.where(vv > 0, vv + 0.5 * ulp, vv + down)
        kz = np.where(k == 0, np.nan, k)
        a = np.where(kz > 0, lo / kz, hi / kz); b = np.where(kz > 0, hi / kz, lo / kz)
        with np.errstate(invalid="ignore", all="ignore"):
            s_lo, s_hi = np.nanmax(a, 1), np.nanmin(b, 1)
        ok &= s_lo <= s_hi
        s_lo, s_hi = np.where(ok, s_lo, 1.0), np.where(ok, s_hi, 1.0)
        got = np.full(len(vv), 99); sc = np.zeros(len(vv)); e = np.floor(np.log2(s_hi))
        for p in range(1, 25):
            for eo in (0, -1):
                q = np.exp2(e + eo - (p - 1))
                for d in (0, 1, -1):
                    c = (np.rint(0.5 * (s_lo + s_hi) / q) + d) * q
                    fits = ok & (got == 99) & (c >= s_lo) & (c <= s_hi) & (c >= np.exp2(e + eo)) & (c < np.exp2(e + eo + 1))
                    good = fits & (bf16(k * c[:, None]) == ww).all(1)
                    got[good] = p; sc[good] = c[good]
        idx = np.nonzero(todo)[0]; hit = got < 99
        need[idx[hit]] = got[hit]
        expo[idx[hit]] = np.floor(np.log2(sc[hit])).astype(np.int64)
        kmin[idx[hit]] = k[hit].min(1); kmax[idx[hit]] = k[hit].max(1)
    fit = (need > 0) & (need < 99)
    return {"groups": G, "unfit": int((need == 99).sum()), "zero": int((need == 0).sum()), "neg_zero_groups": negzero,
            "bits": np.bincount(np.minimum(need, 30), minlength=31), "plus8": int((fit & (kmax == 8)).sum()),
            "plus8_and_minus8": int((fit & (kmax == 8) & (kmin == -8)).sum()),
            "exp": [int(expo[fit].min()), int(expo[fit].max())] if fit.any() else None}

if __name__ == "__main__":
    names = sorted(n for n in ix if n.startswith("model.language_model.") and n.endswith(".weight") and "norm" not in n
                   and (".layers.0." in n or ".layers.5." in n or ".layers.30." in n or ".layers.59." in n or "embed" in n))
    tot = np.zeros(31, np.int64); T = {"groups": 0, "unfit": 0, "plus8": 0, "plus8_and_minus8": 0, "neg_zero_groups": 0}
    for n in names:
        r = analyse(n)
        for k_ in T: T[k_] += r[k_]
        tot += r["bits"]
        bits = {p: int(c) for p, c in enumerate(r["bits"]) if c and p < 30}
        print(n.replace("model.language_model.", ""), {k_: r[k_] for k_ in ("groups", "unfit", "plus8", "plus8_and_minus8", "neg_zero_groups", "exp")}, "max bits", max(bits) if bits else 0, flush=True)
    cum = np.cumsum(tot[:30]) / T["groups"]
    print("TOTAL", T, "| share fitting with <= p significand bits:", {p: round(float(cum[p]), 6) for p in range(10, 17)})
