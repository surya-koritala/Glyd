#!/usr/bin/env python3
"""Calibration documents for the calibrated expert pack (none from wikitext-2's test split, which the KLD is measured on): wikitext-2-raw-v1
TRAIN (Hugging Face Salesforce/wikitext), the engine's Rust sources and the repo's markdown, cut into DOCS documents of LEN tokens.
  python calib_tokens.py TRAIN_PARQUET SRC_TREE TOKENIZER_JSON OUT.json [--len 2048] [--docs 32]"""
import argparse, glob, json, random
import pyarrow.parquet as pq
from tokenizers import Tokenizer

ap = argparse.ArgumentParser()
ap.add_argument("train"); ap.add_argument("src"); ap.add_argument("tokenizer"); ap.add_argument("out")
ap.add_argument("--len", type=int, default=2048); ap.add_argument("--docs", type=int, default=32)
a = ap.parse_args()
tok = Tokenizer.from_file(a.tokenizer)
enc = lambda t: tok.encode(t, add_special_tokens=False).ids
cut = lambda ids: [ids[i:i + a.len] for i in range(0, len(ids) - a.len + 1, a.len)]
kinds = {"wiki_train": cut(enc("".join(pq.read_table(a.train).column("text").to_pylist()[:40000]))),
         "rust": cut(enc("\n\n".join(open(f, errors="ignore").read() for f in sorted(glob.glob(f"{a.src}/glyd-engine/src/*.rs"))))),
         "markdown": cut(enc("\n\n".join(open(f, errors="ignore").read() for f in sorted(glob.glob(f"{a.src}/**/*.md", recursive=True)))))}
random.seed(0)
share = {"wiki_train": a.docs * 3 // 8, "rust": a.docs * 3 // 8}
share["markdown"] = a.docs - sum(share.values())
docs = []
for k, d in kinds.items():
    print(k, len(d), "documents available, taking", min(share[k], len(d)))
    docs += random.sample(d, min(share[k], len(d)))
json.dump({"docs": docs, "len": a.len, "kinds": {k: share[k] for k in kinds}}, open(a.out, "w"))
print(len(docs), "documents of", a.len, "tokens")
