"""Fixtures for the Rust port of the plugin's KV calibration (src/kvcal.rs): inputs and what glyd_gpu.kvcal (the plugin's Python) makes of them. Run where torch and glyd_gpu are.

    python kvcal_fixtures.py ../tests/fixtures/kvcal.json
"""
import json
import sys

import numpy as np
import torch

from glyd_gpu import kvcal


def main(out):
    rng = np.random.default_rng(7)
    fx = {}

    # the window of 8 exponents holding the most values: a clear winner, ties (the first), and random counts
    hist = torch.from_numpy(rng.integers(0, 50, size=(3, 4, 128))).to(torch.int64)
    hist[0, 0, :] = 0
    hist[0, 0, 10:18] = 100
    hist[1, 2, :] = 5
    hist[2, 1, :] = 0
    hist[2, 1, 120:128] = 9
    hist[2, 3, :] = 0
    fx["best_base"] = {"hist": hist.tolist(), "base": kvcal.best_base(hist).tolist()}

    # the exponents' histogram of bf16 bits, the values outside a window, and a page's needs
    T, H, D = 48, 2, 6
    bits = torch.from_numpy(rng.integers(0, 1 << 16, size=(T, H, D)).astype(np.uint16).view(np.int16))
    e7 = ((bits.to(torch.int32) >> 8) & 127)
    keep = torch.from_numpy(rng.integers(0, 4, size=(T, H, D))) > 0  # most values in a narrow band of exponents
    band = torch.from_numpy(rng.integers(60, 68, size=(T, H, D)).astype(np.int32))
    mix = torch.where(keep, band, e7)
    bits = ((bits.to(torch.int32) & ~(127 << 8)) | (mix << 8)).to(torch.int16)
    base = kvcal.best_base(kvcal.histogram(bits)).to(torch.uint8)
    ek, ev = kvcal.outside(bits, base), kvcal.outside(bits.flip(0), base)
    tot, beyond = kvcal.leaf_needs(ek, ev)
    fx["exp"] = {"bits": bits.view(torch.int16).tolist(), "shape": [T, H, D], "hist": kvcal.histogram(bits).tolist(), "base": base.tolist(),
                 "outside": ek.to(torch.int32).tolist(), "outside_v": ev.to(torch.int32).tolist(), "page_total": tot.tolist(), "page_beyond": beyond.tolist(),
                 "sweep_needs": kvcal.sweep_needs(ek, ev).tolist(), "sweep_exc": kvcal.sweep_exc(ek, ev).tolist()}

    # the sweep's figures per token and layer -> chunks, layers over, the most needed; the pool's meta
    need = torch.from_numpy(rng.integers(0, 1200, size=(60, 6))).to(torch.int32)
    need[7, :] = 5000
    need[9, 2] = 4000
    try:
        from glyd_gpu.vllm_kv import ND, free_bay
    except ImportError:  # (no vLLM where this runs: the plugin's own arithmetic, glyd_gpu/vllm_kv.py's free_bay)
        ND = 16
        free_bay = lambda H: ((16 * H + 4 * (1 + ND) + 127) // 128 * 128 - 16 * H) // 4 - 1 - ND
    bay = free_bay(8)
    capacity = bay + ND * kvcal.CH
    chunks, over, maxneed = kvcal.sweep_stats(need, bay, capacity)
    meta = kvcal.store_meta(chunks, over, maxneed, bay, capacity, 60, lambda i: i % 3 != 0, lambda i: f"<{i}>")
    fx["sweep"] = {"need": need.tolist(), "bay": bay, "capacity": capacity, "chunks": chunks.tolist(), "over": over.tolist(), "maxneed": maxneed.tolist(), "store": meta["store"]}

    # a quantile as the meta takes it
    x = torch.from_numpy(rng.integers(0, 100, size=137)).to(torch.int64)
    fx["quantile"] = {"x": x.tolist(), "q": {str(p): int(torch.quantile(x.float(), p).item()) for p in (0.99, 0.999)}}

    # the sequences of a calibration: documents as synthetic ids
    docs = []
    for kind, name, n in (("chat", "c1", 700), ("chat", "c2", 300), ("code", "k1", 1500), ("docs", "d1", 900), ("prose", "p1", 2200), ("prose", "p2", 1700), ("table", "t1", 400)):
        docs.append((kind, name, [int(v) for v in rng.integers(0, 1000, size=n)]))
    fx["sequences"] = {"docs": [[k, n, ids] for k, n, ids in docs],
                       "by_length": {str(L): [[ids, [[kd, a, b] for kd, a, b in segs]] for ids, segs in kvcal.sequences(docs, L)] for L in (1000, 4096)}}
    fx["constants"] = {"window": kvcal.WINDOW, "refused": kvcal.REFUSED, "inline": kvcal.INLINE, "ch": kvcal.CH, "chunk": kvcal.CHUNK, "lap": kvcal.LAP, "version": kvcal.VERSION, "free_bay_8": free_bay(8), "nd": ND}
    fx["corpus"] = {"digest": kvcal.corpus_digest(), "sections": [[k, n, len(b), b[:40], b[-40:]] for k, n, b in kvcal.corpus()]}
    json.dump(fx, open(out, "w"), separators=(",", ":"))
    print(out, sum(len(json.dumps(v)) for v in fx.values()), "bytes of fixtures")


if __name__ == "__main__":
    main(sys.argv[1])
