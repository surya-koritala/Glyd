#!/usr/bin/env python3
"""The token chunks for KL divergence vs bf16, the way llama.cpp's perplexity tool cuts them: wikitext-2-raw-v1 test (Hugging Face
Salesforce/wikitext, test-00000-of-00001.parquet, rows joined as they are), tokenized with the model's tokenizer (no special tokens added),
cut into consecutive chunks of CTX tokens from the start; the first N chunks.
  python kld_tokens.py PARQUET TOKENIZER_JSON OUT.json [--ctx 512] [--chunks 64]"""
import argparse, hashlib, json
import pyarrow.parquet as pq
from tokenizers import Tokenizer

ap = argparse.ArgumentParser()
ap.add_argument("parquet"); ap.add_argument("tokenizer"); ap.add_argument("out")
ap.add_argument("--ctx", type=int, default=512); ap.add_argument("--chunks", type=int, default=64)
a = ap.parse_args()
text = "".join(pq.read_table(a.parquet).column("text").to_pylist())
ids = Tokenizer.from_file(a.tokenizer).encode(text, add_special_tokens=False).ids
docs = [ids[i * a.ctx:(i + 1) * a.ctx] for i in range(min(a.chunks, len(ids) // a.ctx))]
json.dump({"docs": docs, "ctx": a.ctx, "corpus": "wikitext-2-raw-v1 test", "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
           "corpus_tokens": len(ids)}, open(a.out, "w"))
print(f"{len(text)} chars, {len(ids)} tokens, {len(docs)} chunks of {a.ctx}")
