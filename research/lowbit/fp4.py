"""MXFP4 weights (E2M1 codes, two a byte, one E8M0 scale a block of 32): the histograms every entropy in the
README comes from.

    python fp4.py FILE.safetensors OUT.json [--sample N] [--procs P]

Reads the expert matrices of a K3-style file (`...experts.E.wN.weight_packed` + `.weight_scale`) or a gpt-oss one
(`...mlp.experts.{gate_up,down}_proj_{blocks,scales}`), counts per matrix, and sums the counts by matrix kind.
Nothing here is lossy: the counts are of the stored bytes themselves.

A code is 4 bits: sign (bit 3) and a magnitude index (bits 0-2) into 0, .5, 1, 1.5, 2, 3, 4, 6.
The stored byte holds two codes, the first element (even column) in the low nibble.
"""
import json, multiprocessing as mp, re, sys, time
import numpy as np
from st import ST
from ent import H

FP4 = np.array([0, .5, 1, 1.5, 2, 3, 4, 6], dtype=np.float32)
SQ = FP4 ** 2


def bc(a, n):
    return np.bincount(a.ravel(), minlength=n).astype(np.int64)


def counts(packed, scales):
    """Histograms of one matrix: packed [R, C/2] uint8, scales [R, C/32] uint8."""
    R, C2 = packed.shape
    C = 2 * C2
    J = C // 32
    assert scales.shape == (R, J), (packed.shape, scales.shape)
    codes = np.empty((R, C), np.uint8)
    codes[:, 0::2] = packed & 15
    codes[:, 1::2] = packed >> 4
    cb = codes.reshape(R, J, 32)
    o = {}
    # codes on their own, as stored byte pairs, and by neighbours
    o["code"] = bc(codes, 16)
    o["byte"] = bc(packed, 256)  # (even, odd): index even + 16 * odd
    prev = cb[:, :, :-1].astype(np.int32)
    cur = cb[:, :, 1:].astype(np.int32)
    o["adj"] = bc(prev * 16 + cur, 256)  # (previous, this) inside a block
    o["adj2"] = bc(cb[:, :, :-2].astype(np.int32) * 256 + prev[:, :, 1:] * 16 + cur[:, :, 1:], 4096)  # (i-2, i-1, i)
    o["up"] = bc(codes[:-1].astype(np.int32) * 16 + codes[1:], 256)  # (same column, row above; this)
    o["pos"] = bc(np.arange(32, dtype=np.int32)[None, None, :] * 16 + cb, 512)  # (position in block, code)
    # codes given their block's scale
    sc = scales.astype(np.int32)
    o["sc"] = bc(sc[:, :, None] * 16 + cb, 4096)
    # codes given a summary of their own block, sent as side information (its histogram is in "*_hist")
    mag = cb & 7
    bmax = mag.max(2).astype(np.int32)
    nzl = np.minimum((mag > 0).sum(2) // 4, 7).astype(np.int32)
    sq = SQ[mag].sum(2)
    sql = np.clip(np.floor(np.log2(np.maximum(sq, 1) / 16) * 1.25), 0, 7).astype(np.int32)
    for name, m, n in (("bmax", bmax, 8), ("bnz", nzl, 8), ("bsq", sql, 8), ("bms", bmax * 8 + sql, 64)):
        o[name] = bc(m[:, :, None] * 16 + cb, n * 16)
        o[name + "_hist"] = bc(m, n)
    # the scales: on their own, and by neighbours; relative to the row's median, and to row and column medians
    o["s"] = bc(scales, 256)
    o["sl"] = bc(scales[:, :-1].astype(np.int32) * 256 + scales[:, 1:], 65536)  # (left, this)
    o["su"] = bc(scales[:-1].astype(np.int32) * 256 + scales[1:], 65536)  # (above, this)
    rm = np.floor(np.median(scales, axis=1)).astype(np.int32)
    rd = sc - rm[:, None]
    rdc = np.clip(rd, -8, 7) + 8
    o["srd"] = bc(rdc, 16)  # offset from the row's median, clipped to -8..7
    o["srd_out"] = np.array([(np.abs(rd + 0.5) > 8).sum()], dtype=np.int64)  # of them outside the clip
    o["srd_l"] = bc(rdc[:, :-1] * 16 + rdc[:, 1:], 256)  # (left offset, this offset)
    o["srd_u"] = bc(rdc[:-1] * 16 + rdc[1:], 256)  # (above offset, this offset)
    o["srd_lu"] = bc(rdc[1:, :-1] * 256 + rdc[:-1, 1:] * 16 + rdc[1:, 1:], 4096)  # (left, above, this), row above's own offset
    cm = np.floor(np.median(rd, axis=0)).astype(np.int32)
    rd2 = np.clip(rd - cm[None, :], -8, 7) + 8
    o["srd2"] = bc(rd2, 16)  # offset from row and column medians
    # best window of w consecutive scales a row, for w = 1..16 : scales inside it (a base a row; escapes are the rest)
    rh = np.bincount((np.arange(R, dtype=np.int64)[:, None] * 16 + rdc).ravel(), minlength=R * 16).reshape(R, 16)
    cs = np.concatenate([np.zeros((R, 1), np.int64), np.cumsum(rh, axis=1)], axis=1)
    win = [(cs[:, w:] - cs[:, :17 - w]).max(1).sum() for w in range(1, 17)]
    o["win_row"] = np.array(win, dtype=np.int64)  # scales inside the best row window of width w (clip counted outside: srd_out)
    th = rh.sum(0)
    tcs = np.concatenate([[0], np.cumsum(th)])
    o["win_mat"] = np.array([(tcs[w:] - tcs[:17 - w]).max() for w in range(1, 17)], dtype=np.int64)
    # per-matrix entropies (a table of its own) as weighted sums: n, sum n*H(code), ns, sum ns*H(scale)
    o["perm"] = np.array([codes.size, codes.size * H(o["code"]), scales.size, scales.size * H(o["s"])], dtype=np.float64)
    return o


def kinds_k3(f):
    for k in f.keys():
        if k.endswith(".weight_packed") and ".experts." in k:
            m = re.search(r"layers\.(\d+)\.block_sparse_moe\.experts\.(\d+)\.(w\d)\.weight_packed", k)
            yield (m.group(3), int(m.group(1)), int(m.group(2)), k, k[:-len("packed")] + "scale")


def kinds_gptoss(f):
    for k in f.keys():
        if k.endswith("_blocks"):
            m = re.search(r"layers\.(\d+)\.mlp\.experts\.(gate_up|down)_proj_blocks", k)
            E = f.info(k)[1][0]
            for e in range(E):
                yield (m.group(2), int(m.group(1)), e, (k, e), (k[:-len("blocks")] + "scales", e))


def get(f, name):
    if isinstance(name, tuple):
        a = f.get(name[0])[name[1]]
        return a.reshape(a.shape[0], -1)
    return f.get(name)


def work(args):
    path, items = args
    f = ST(path)
    acc, n = {}, 0
    for kind, layer, e, pk, sk in items:
        o = counts(np.asarray(get(f, pk)), np.asarray(get(f, sk)))
        a = acc.setdefault(kind, {})
        for key, v in o.items():
            a[key] = a[key] + v if key in a else v
        a["matrices"] = a.get("matrices", 0) + 1
    return acc


def merge(parts):
    tot = {}
    for acc in parts:
        for kind, a in acc.items():
            t = tot.setdefault(kind, {})
            for key, v in a.items():
                t[key] = t[key] + v if key in t else v
    return tot


if __name__ == "__main__":
    path, out = sys.argv[1], sys.argv[2]
    sample = int(sys.argv[sys.argv.index("--sample") + 1]) if "--sample" in sys.argv else 0
    procs = int(sys.argv[sys.argv.index("--procs") + 1]) if "--procs" in sys.argv else 6
    f = ST(path)
    items = list(kinds_k3(f)) or list(kinds_gptoss(f))
    rng = np.random.default_rng(0)
    if sample:  # a seeded sample of `sample` experts of each kind of each layer
        keep = {}
        for it in items:
            keep.setdefault((it[0], it[1]), []).append(it)
        items = []
        for v in keep.values():
            items += [v[i] for i in sorted(rng.choice(len(v), size=min(sample, len(v)), replace=False))]
    t0 = time.time()
    chunks = [(path, items[i::procs * 4]) for i in range(procs * 4)]
    with mp.Pool(procs) as p:
        parts = p.map(work, chunks)
    tot = merge(parts)
    res = {kind: {k: v.tolist() if hasattr(v, "tolist") else v for k, v in a.items()} for kind, a in tot.items()}
    res["_meta"] = {"file": path.split("/")[-1], "matrices": len(items), "seconds": time.time() - t0, "sample": sample}
    json.dump(res, open(out, "w"))
    print(f"{len(items)} matrices in {time.time() - t0:.0f} s -> {out}")
