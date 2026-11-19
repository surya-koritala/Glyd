#!/usr/bin/env python3
"""KL divergence of ours vs the bf16 reference, the numbers llama.cpp's perplexity tool prints with --kl-divergence: mean KLD with its
standard error, median, 99% and 99.9% KLD, PPL of each and their ratio, top-1 agreement ("same top p"). Over the vocabulary both
share (the reference's rows; extra padded logits of ours are dropped).
  python kld_compare.py chunks.json ref.f32 ours.f32 [--vocab V]"""
import argparse, json
import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument("tokens"); ap.add_argument("ref"); ap.add_argument("ours")
a = ap.parse_args()
meta = json.load(open(a.ref + ".json"))
V, per = meta["vocab"], meta["rows_per_chunk"]
docs = json.load(open(a.tokens))["docs"]
ref = np.memmap(a.ref, dtype="<f4", mode="r").reshape(-1, V)
Vo = np.memmap(a.ours, dtype="<f4", mode="r").size // ref.shape[0]
ours = np.memmap(a.ours, dtype="<f4", mode="r").reshape(-1, Vo)
assert ours.shape[0] == ref.shape[0] == len(docs) * per, (ours.shape, ref.shape, len(docs), per)


def logsoftmax(x):
    x = x.astype(np.float64)
    m = x.max(axis=1, keepdims=True)
    return x - (m + np.log(np.exp(x - m).sum(axis=1, keepdims=True)))


kl, nr, nq, same = [], [], [], 0
for c, d in enumerate(docs):
    n = len(d)
    targets = np.array(d[n // 2 + 1:n])
    lp, lq = logsoftmax(ref[c * per:(c + 1) * per]), logsoftmax(ours[c * per:(c + 1) * per, :V])
    kl.append((np.exp(lp) * (lp - lq)).sum(axis=1))
    nr.append(-lp[np.arange(per), targets]); nq.append(-lq[np.arange(per), targets])
    same += int((lp.argmax(1) == lq.argmax(1)).sum())
kl, nr, nq = np.concatenate(kl), np.concatenate(nr), np.concatenate(nq)
N = len(kl)
r = {"positions": N, "chunks": len(docs), "ctx": len(docs[0]),
     "mean_kld": float(kl.mean()), "mean_kld_se": float(kl.std(ddof=1) / np.sqrt(N)), "median_kld": float(np.median(kl)),
     "kld_99": float(np.percentile(kl, 99)), "kld_99_9": float(np.percentile(kl, 99.9)), "max_kld": float(kl.max()),
     "ppl_ref": float(np.exp(nr.mean())), "ppl_ours": float(np.exp(nq.mean())), "same_top1": same / N}
r["ppl_ratio"] = r["ppl_ours"] / r["ppl_ref"]
r["ln_ppl_ratio"] = float(nq.mean() - nr.mean())
print(json.dumps(r, indent=1))
