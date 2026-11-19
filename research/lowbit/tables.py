"""The README's tables, printed from results/*.json (every number in them is read from there).

    python tables.py
"""
import glob, json, os
import numpy as np

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results") + "/"
J = lambda f: json.load(open(f if os.path.isabs(f) else R + f))


def fp4(files, key="ALL"):
    S = [J(f)[key] for f in sorted(glob.glob(R + files))]
    return S, lambda k: (float(np.mean([s[k] for s in S])), float(min(s[k] for s in S)), float(max(s[k] for s in S)))


def fmt(t, d=3):
    m, lo, hi = t
    return f"{m:.{d}f} ({lo:.{d}f}-{hi:.{d}f})"


print("### MXFP4: bits a weight, mean over the sampled layers (range)\n")
print("| | stored | codes H0 | scales H0 | ceiling: order 0, a table a matrix | ceiling: best contexts | scales 2 bits, nibbles raw | scales 2 bits, nibbles Huffman by scale |")
print("| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
for name, pat in (("Kimi K3 experts (layers 1, 12, 47, 90)", "fp4_k3_s*.json"), ("gpt-oss-120b experts (4 shards)", "fp4_gptoss_s*.json")):
    S, g = fp4(pat)
    print(f"| {name} | 4.25 | {fmt(g('H_code'))} | {fmt(g('H_scale'))} | {fmt(g('ceiling_order0_per_matrix'))} | {fmt(g('ceiling_contexts'))} | {fmt(g('layout_scales_only_bits'))} | {fmt(g('layout_scales_huffman_by_scale_bits'))} |")

print("\n### MXFP4: what conditioning gives (bits a weight of the 4-bit code; scales a scale)\n")
print("| | H(code) | given previous | given previous 2 | given the row above | given position in block | given the block's scale | block summary sent (max code), net | H(scale) | given left | given above | given row median, left, above |")
print("| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
for name, pat in (("Kimi K3", "fp4_k3_s*.json"), ("gpt-oss-120b", "fp4_gptoss_s*.json")):
    S, g = fp4(pat)
    print(f"| {name} | {g('H_code')[0]:.3f} | {g('H_code|prev')[0]:.3f} | {g('H_code|prev2')[0]:.3f} | {g('H_code|above')[0]:.3f} | {g('H_code|position')[0]:.3f} | {g('H_code|scale')[0]:.3f} | {g('R_block_bmax')[0]:.3f} | {g('H_scale')[0]:.3f} | {g('H_scale|left')[0]:.3f} | {g('H_scale|above')[0]:.3f} | {g('H_scale_rowrel|left,above')[0]:.3f} |")

print("\n### MXFP4: the 2-bit scale layout's escapes and the Huffman nibble code\n")
for name, pat in (("Kimi K3", "fp4_k3_s*.json"), ("gpt-oss-120b", "fp4_gptoss_s*.json")):
    S, g = fp4(pat)
    print(f"- {name}: code histogram (% of codes, magnitude index 0..7, sign +): {np.mean([s['code_hist_pct'][:8] for s in S], axis=0).round(2).tolist()}; Huffman pooled {g('code_huffman_bits')[0]:.3f}, by scale value {g('code_huffman_by_scale_bits')[0]:.3f}; scales with k=1 (1 offset + escape): {g('scale_k1_row_bits_per_scale')[0]:.2f} bits ({g('scale_k1_row_escape_pct')[0]:.1f}% escapes), k=2: {g('scale_k2_row_bits_per_scale')[0]:.3f} bits ({g('scale_k2_row_escape_pct')[0]:.3f}% escapes)")

print("\n### FP8 e4m3: bits a weight of the 8-bit code (block scales: 0.002 bits a weight, not coded)\n")
print("| | H0 pooled | H0, a table a matrix | ceiling, best contexts | exponent Huffman | exponent in tiers of 2-bit digits | exponent 3 bits + escape | escapes / raw share |")
print("| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
rows = [("DeepSeek-V3 routed experts, layers 3 / 9-10 / 31 / 55-56", [J(f)["ROLE routed"] for f in sorted(glob.glob(R + "fp8_ds3_s*.json"))]),
        ("DeepSeek-V3 attention + shared + dense MLP", None), ("Qwen3-8B-FP8 attention + MLP (all layers)", [J("fp8_qwen3_8b.json")["ALL"]])]
for name, S in rows:
    if S is None:
        S = [J(f)[r] for f in sorted(glob.glob(R + "fp8_ds3_s*.json")) for r in ("ROLE attention", "ROLE shared", "ROLE dense_mlp") if r in J(f)]
    m = lambda k: float(np.mean([s[k] for s in S]))
    lo = lambda k: float(min(s[k] for s in S))
    hi = lambda k: float(max(s[k] for s in S))
    print(f"| {name} | {m('H_code'):.3f} | {m('H_code_per_matrix'):.3f} | {m('ceiling_contexts'):.3f} | {m('layout_huffman_exp_bits'):.3f} | {m('layout_tier3_bits'):.3f} | {m('layout_fast3_bits'):.3f} | {m('layout_fast3_escape_pct'):.1f}% / {m('layout_tier3_raw_pct'):.1f}% |")
print("\nPer sample (DeepSeek-V3 routed experts): ")
for f in sorted(glob.glob(R + "fp8_ds3_s*.json")):
    s = J(f)["ROLE routed"]
    print(f"- {os.path.basename(f)}: {s['matrices']} matrices, {s['weights'] / 1e9:.2f} G weights: H0 per matrix {s['H_code_per_matrix']:.3f}, ceiling {s['ceiling_contexts']:.3f}, Huffman {s['layout_huffman_exp_bits']:.3f}, tiered {s['layout_tier3_bits']:.3f}, fast3 {s['layout_fast3_bits']:.3f} (escapes {s['layout_fast3_escape_pct']:.1f}%), top-8 window exceptions {s['window8_top_exceptions_pct']:.1f}%")

print("\n### FP8: what conditioning gives (bits of the exponent field; sign + mantissa given the exponent: ~3.94)\n")
print("| | H(exp) pooled | a table a matrix | given left | given above | given left, above | given scale octave | given column in block | a table a column | a table a row |")
print("| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
for name, S in rows[:1] + rows[2:]:
    m = lambda k: float(np.mean([s[k] for s in S]))
    print(f"| {name} | {m('H_exp'):.3f} | {m('H_exp_per_matrix'):.3f} | {m('H_exp|left (a table a matrix)'):.3f} | {m('H_exp|above (a table a matrix)'):.3f} | {m('H_exp|left,above (a table a matrix)'):.3f} | {m('H_exp|scale_octave (a table a matrix)'):.3f} | {m('H_exp|column_in_block'):.3f} | {m('H_exp|column(table a column)'):.3f} | {m('H_exp|row(table a row)'):.3f} |")

print("\n### Sizes and fits\n")
fit = J("fit.json")
for m, title in (("kimi_k3", "Kimi K3"), ("gpt_oss_120b", "gpt-oss-120b"), ("deepseek_v3", "DeepSeek-V3"), ("qwen3_8b_fp8", "Qwen3-8B-FP8")):
    r = fit[m]
    print(f"\n**{title}**, stored {r['total_stored_GB']:.1f} GB" + (f", KV a 128k sequence {r['kv_a_128k_sequence_GB']:.2f} GB" if "kv_a_128k_sequence_GB" in r else ""))
    if "fit" in r:
        nodes = [k for k in r["fit"][0] if isinstance(r["fit"][0][k], dict)]
        print("\n| layout | weights, GB | off | " + " | ".join(f"{n}: headroom, GB / after the reserve (128k contexts)" for n in nodes) + " |")
        print("| :--- | ---: | ---: | " + " | ".join("---:" for _ in nodes) + " |")
        for row in r["fit"]:
            print(f"| {row['layout']} | {row['weights_GB']:.1f} | {100 * (1 - row['weights_GB'] / r['total_stored_GB']):.1f}% | " + " | ".join(f"{row[n]['headroom_GB']:.1f} / {row[n]['free_after_reserve_GB']:.1f} ({row[n]['seqs_128k']:.1f})" for n in nodes) + " |")
    else:
        print("\n| layout | GB | off |\n| :--- | ---: | ---: |")
        for k, v in r["sizes_GB"].items():
            print(f"| {k} | {v:.1f} | {100 * (1 - v / r['total_stored_GB']):.1f}% |")

print("\n### Generic compressors on raw bytes (share of stored size)\n")
print("| tensor | zstd -19 | xz -6 |\n| :--- | ---: | ---: |")
for r in J("generic.json"):
    print(f"| {r['tensor']} | {r['zstd19_pct']:.1f}% | {r['xz6_pct']:.1f}% |")

print("\n### bf16 parts (measured order-0 floor, bits a weight)\n")
for pat in ("bf16_k3_s*.json", "bf16_gptoss_s*.json", "bf16_ds3_s*.json", "bf16_qwen_s*.json"):
    acc = {}
    for f in sorted(glob.glob(R + pat)):
        for role, a in J(f).items():
            if role == "_meta" or not a["big_n"]:
                continue
            n, s = acc.get(role, (0, 0.0))
            acc[role] = (n + a["big_n"], s + a["sums"]["H_word"])
    print(f"- {pat.split('_')[1]}: " + ", ".join(f"{r} {s / n:.2f} ({n / 1e6:.0f} M weights)" for r, (n, s) in sorted(acc.items())))
