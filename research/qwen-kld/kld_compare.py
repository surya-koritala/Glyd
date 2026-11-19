#!/usr/bin/env python3
"""KL divergence of ours vs the bf16 reference, the numbers llama.cpp's perplexity tool prints with --kl-divergence: mean KLD with its
standard error, median, 99% and 99.9% KLD, PPL of each and their ratio, top-1 agreement ("same top p"). Over the vocabulary both
share (the reference's rows; extra padded logits of ours are dropped).
With --top K, the KL over the reference's K most likely tokens, both renormalized to that set (KL as a top-K log records it).
  python kld_compare.py chunks.json ref.f32 ours.f32 [--top K]
(kld_engine.py --ref takes `chunk` and `summary` from here to score each chunk as the engine makes it.)"""
import argparse, json
import numpy as np


def logsoftmax(x):
    x = x.astype(np.float64)
    m = x.max(axis=1, keepdims=True)
    return x - (m + np.log(np.exp(x - m).sum(axis=1, keepdims=True)))


def chunk(ref_rows, our_rows, targets, top=None):
    """A chunk's rows (the reference's, ours cut to its vocabulary): each position's KL (over the reference's `top` tokens, renormalized,
    or the whole vocabulary), the NLL of each side, and how many top-1 picks agree."""
    lp, lq = logsoftmax(ref_rows), logsoftmax(our_rows)
    if top:
        k = np.argpartition(-lp, top - 1, axis=1)[:, :top]
        tp, tq = np.take_along_axis(lp, k, 1), np.take_along_axis(lq, k, 1)
        tp, tq = tp - np.logaddexp.reduce(tp, axis=1, keepdims=True), tq - np.logaddexp.reduce(tq, axis=1, keepdims=True)
        kl = (np.exp(tp) * (tp - tq)).sum(axis=1)
    else:
        kl = (np.exp(lp) * (lp - lq)).sum(axis=1)
    n = len(targets)
    return kl, -lp[np.arange(n), targets], -lq[np.arange(n), targets], int((lp.argmax(1) == lq.argmax(1)).sum())


def summary(parts, ctx, top=None):
    """The numbers over the chunks' `chunk` results."""
    kl, nr, nq = (np.concatenate([p[i] for p in parts]) for i in range(3))
    N = len(kl)
    r = {"positions": N, "chunks": len(parts), "ctx": ctx,
         "mean_kld": float(kl.mean()), "mean_kld_se": float(kl.std(ddof=1) / np.sqrt(N)), "median_kld": float(np.median(kl)),
         "kld_99": float(np.percentile(kl, 99)), "kld_99_9": float(np.percentile(kl, 99.9)), "max_kld": float(kl.max()),
         "chunk_mean_kld": [float(p[0].mean()) for p in parts], "top": top, "ppl_ref": float(np.exp(nr.mean())), "ppl_ours": float(np.exp(nq.mean())),
         "same_top1": sum(p[3] for p in parts) / N}
    r["ppl_ratio"] = r["ppl_ours"] / r["ppl_ref"]
    r["ln_ppl_ratio"] = float(nq.mean() - nr.mean())
    return r


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("tokens"); ap.add_argument("ref"); ap.add_argument("ours"); ap.add_argument("--top", type=int)
    a = ap.parse_args()
    meta = json.load(open(a.ref + ".json"))
    V, per = meta["vocab"], meta["rows_per_chunk"]
    docs = json.load(open(a.tokens))["docs"]
    ref = np.memmap(a.ref, dtype="<f4", mode="r").reshape(-1, V)
    Vo = np.memmap(a.ours, dtype="<f4", mode="r").size // ref.shape[0]
    ours = np.memmap(a.ours, dtype="<f4", mode="r").reshape(-1, Vo)
    assert ours.shape[0] == ref.shape[0] == len(docs) * per, (ours.shape, ref.shape, len(docs), per)
    parts = [chunk(ref[c * per:(c + 1) * per], ours[c * per:(c + 1) * per, :V], np.array(d[len(d) - per:]), a.top) for c, d in enumerate(docs)]
    print(json.dumps(summary(parts, len(docs[0]), a.top), indent=1))
