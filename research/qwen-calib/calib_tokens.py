#!/usr/bin/env python3
"""Calibration documents for the calibrated expert pack (none from wikitext-2's test split, which the KLD is measured on): wikitext-2-raw-v1
TRAIN (Hugging Face Salesforce/wikitext), the engine's Rust sources and the repo's markdown, cut into DOCS documents of LEN tokens.
  python calib_tokens.py TRAIN_PARQUET SRC_TREE TOKENIZER_JSON OUT.json [--len 2048] [--docs 32] [--chat CHATS.jsonl --snap SNAP]
--chat: conversations (a line each: its kind and messages, with tools where it called them) in the model's chat template, each kind's
cut into documents; they take all but a quarter of the documents (as many as they make), wikitext, Rust and markdown the rest (3:3:2):
calib64k's wikitext-heavy mix made calibrated packs worse off it (2026-10-18: long set 0.235 vs 0.190).
Never an eval text (wikitext-2 test, the long set, Strata's quality set)."""
import argparse, glob, json, random
import pyarrow.parquet as pq
from tokenizers import Tokenizer

ap = argparse.ArgumentParser()
ap.add_argument("train"); ap.add_argument("src"); ap.add_argument("tokenizer"); ap.add_argument("out")
ap.add_argument("--len", type=int, default=2048); ap.add_argument("--docs", type=int, default=32)
ap.add_argument("--chat"); ap.add_argument("--snap")
a = ap.parse_args()
tok = Tokenizer.from_file(a.tokenizer)
enc = lambda t: tok.encode(t, add_special_tokens=False).ids
cut = lambda ids: [ids[i:i + a.len] for i in range(0, len(ids) - a.len + 1, a.len)]
kinds = {"wiki_train": cut(enc("".join(pq.read_table(a.train).column("text").to_pylist()[:40000]))),
         "rust": cut(enc("\n\n".join(open(f, errors="ignore").read() for f in sorted(glob.glob(f"{a.src}/glyd-engine/src/*.rs"))))),
         "markdown": cut(enc("\n\n".join(open(f, errors="ignore").read() for f in sorted(glob.glob(f"{a.src}/**/*.md", recursive=True)))))}
random.seed(0)
rest = a.docs
if a.chat:
    tt, by = None, {}
    for line in open(a.chat):
        c = json.loads(line)
        # (a line's "text" when it was rendered already, else its messages in the chat template: transformers, which this venv may lack)
        if "text" not in c and tt is None:
            from transformers import AutoTokenizer
            tt = AutoTokenizer.from_pretrained(a.snap)
        text = c.get("text") or tt.apply_chat_template(c["messages"], tools=c.get("tools"), tokenize=False, enable_thinking=False)
        by.setdefault("chat_" + c["kind"], []).append(text)
    chat = {k: cut(enc("".join(v))) for k, v in by.items()}
    room = a.docs - a.docs // 4
    while sum(len(v) for v in chat.values()) > room:
        max(chat.values(), key=len).pop()
    kinds = {**chat, **kinds}
    rest = a.docs - sum(len(v) for v in chat.values())
share = {k: len(v) for k, v in kinds.items() if k.startswith("chat_")}
share.update({"wiki_train": rest * 3 // 8, "rust": rest * 3 // 8})
share["markdown"] = rest - share["wiki_train"] - share["rust"]
docs = []
for k, d in kinds.items():
    print(k, len(d), "documents available, taking", min(share[k], len(d)))
    docs += random.sample(d, min(share[k], len(d)))
json.dump({"docs": docs, "len": a.len, "kinds": {k: share[k] for k in kinds}}, open(a.out, "w"))
print(len(docs), "documents of", a.len, "tokens")
