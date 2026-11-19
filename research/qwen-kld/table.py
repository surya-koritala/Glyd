#!/usr/bin/env python3
"""The step table from kld_compare.py / kld_engine.py outputs named s1_{NAME}_{wiki,long}.json (+ .f32.json for the decode speed):
  python table.py DIR NAME[=label] ...      one row a configuration, wikitext 32x512 and ref_long 11x512."""
import json, os, sys

d, names = sys.argv[1], sys.argv[2:]
print("| change | wiki mean KLD (± SE) | median | 99.9% | PPL ratio | top-1 | ref_long mean KLD | decode tok/s (one prompt: depends on the text written, not comparable across rows) |")
print("|---|--:|--:|--:|--:|--:|--:|--:|")
for nl in names:
    n, _, label = nl.partition("=")
    def get(t, ext=".json"):
        p = f"{d}/s1_{n}_{t}{ext}"
        return json.load(open(p)) if os.path.exists(p) and os.path.getsize(p) else None
    w, lg, sp = get("wiki"), get("long"), get("wiki", ".f32.json")
    if not w:
        print(f"| {label or n} | (not measured) |"); continue
    print(f"| {label or n} | {w['mean_kld']:.3f} ± {w['mean_kld_se']:.3f} | {w['median_kld']:.4f} | {w['kld_99_9']:.2f} | {w['ppl_ratio']:.3f} | "
          f"{100 * w['same_top1']:.1f}% | {lg['mean_kld']:.3f} | {sp.get('decode_tok_s', '?') if sp else '?'} |" if lg else "")
