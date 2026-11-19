"""Entropies and layout sizes (bits a weight) from fp8.py's counts.

    python report_fp8.py OUT.json [OUT.json ...] [--json SUMMARY.json]

Several files are summed (shards of one model). Bits a weight are of the 8-bit code; the block scales (32 bits a
128x128 block: 0.002 bits a weight) are on top of the stored size and not coded.
"""
import json, sys
import numpy as np
from ent import H, Hc
from report_fp4 import load, pool


def summarize(a):
    code = np.asarray(a["code"], dtype=np.float64)
    N = code.sum()
    ex = np.asarray(a["exp"], dtype=np.float64)
    s = {"weights": int(N), "matrices": int(a["matrices"])}
    s["exp_hist_pct"] = (100 * ex / N).round(3).tolist()
    s["H_code"] = H(code)
    s["H_code_per_matrix"] = float(a["perm_code"][1] / a["perm_code"][0])
    s["H_exp"] = H(ex)
    s["H_exp_per_matrix"] = float(a["perm_exp"][1] / a["perm_exp"][0])
    s["H_sign"] = H(code.reshape(2, 128).sum(1))
    s["H_mant"] = H(code.reshape(2, 16, 8).sum((0, 1)))
    s["H_sign_mant_given_exp"] = s["H_code"] - s["H_exp"]
    s["H_exp|left"] = Hc(np.asarray(a["el"]).reshape(16, 16))
    s["H_exp|above"] = Hc(np.asarray(a["eu"]).reshape(16, 16))
    s["H_exp|left,above"] = Hc(np.asarray(a["elu"]).reshape(256, 16))
    s["H_exp|scale_octave"] = Hc(np.asarray(a["scab"]).reshape(32, 16))
    s["H_exp|scale_rel_median"] = Hc(np.asarray(a["scre"]).reshape(17, 16))
    s["H_exp|column_in_block"] = Hc(np.asarray(a["col128"]).reshape(128, 16))
    s["H_exp|row_in_block"] = Hc(np.asarray(a["row128"]).reshape(128, 16))
    bh = np.asarray(a["bmx_hist"], dtype=np.float64)
    s["H_exp|block_max_exp"] = Hc(np.asarray(a["bmx"]).reshape(16, 16))
    s["block_max_exp_side_bits_per_weight"] = H(bh) / 16384
    s["block_max_exp_hist_pct"] = (100 * bh / bh.sum()).round(2).tolist()
    pc = np.asarray(a["perm_ctx"], dtype=np.float64)
    for i, nm in enumerate(("left", "above", "left,above", "scale_octave", "scale_rel_median")):
        s[f"H_exp|{nm} (a table a matrix)"] = float(pc[i + 1] / pc[0])
    s["H_sign_mant_given_exp_per_matrix"] = float((a["perm_code"][1] - a["perm_exp"][1]) / a["perm_code"][0])
    s["H_exp|column(table a column)"] = float(a["colcond"][1] / a["colcond"][0])
    s["H_exp|row(table a row)"] = float(a["rowcond"][1] / a["rowcond"][0])
    # contexts with a table of the matrix's own (16 counts a matrix or a few thousand: nothing a weight), except the per-column and per-row
    # tables, which cost a table a column or a row (not counted here: an upper bound on what they could give)
    ctxs = {k: s[k] for k in ("H_exp|left (a table a matrix)", "H_exp|above (a table a matrix)", "H_exp|left,above (a table a matrix)",
                              "H_exp|scale_octave (a table a matrix)", "H_exp|scale_rel_median (a table a matrix)")}
    ctxs["H_exp (a table a matrix)"] = s["H_exp_per_matrix"]
    best = min(ctxs, key=ctxs.get)
    s["best_context"] = best
    s["ceiling_order0_pooled"] = s["H_code"]
    s["ceiling_order0_per_matrix"] = s["H_code_per_matrix"]
    s["ceiling_contexts"] = s["H_sign_mant_given_exp_per_matrix"] + ctxs[best]
    s["ceiling_with_row_tables_uncosted"] = s["H_sign_mant_given_exp_per_matrix"] + min(ctxs[best], s["H_exp|row(table a row)"], s["H_exp|column(table a column)"])
    L = np.asarray(a["lay"], dtype=np.float64)
    n = L[0]
    s["layout_huffman_exp_bits"] = 4 + L[1] / n  # sign and mantissa raw, exponent Huffman (a table a matrix)
    s["layout_fast3_bits"] = 4 + L[2] / n
    s["layout_fast3_escape_pct"] = 100 * L[3] / n
    s["layout_tier2_bits"] = 4 + L[4] / n
    s["layout_tier2_raw_pct"] = 100 * L[5] / n
    s["layout_tier3_bits"] = 4 + L[6] / n
    s["layout_tier3_raw_pct"] = 100 * L[7] / n
    s["window8_top_exceptions_pct"] = 100 * L[8] / n  # exponent field < 8: outside the top window of 8
    s["window4_top_exceptions_pct"] = 100 * L[9] / n  # exponent field < 12
    return s


def show(name, s):
    pct = lambda b: f"{b:.3f} ({100 * (1 - b / 8):.1f}% off 8)"
    print(f"== {name}: {s['weights'] / 1e9:.3f} G weights in {s['matrices']} matrices")
    print("   exponent field histogram % (0..15):", s["exp_hist_pct"])
    print("   block max exponent % (0..15):", s["block_max_exp_hist_pct"])
    for k in ("H_code", "H_code_per_matrix", "H_exp", "H_exp_per_matrix", "H_sign", "H_mant", "H_sign_mant_given_exp", "H_exp|left", "H_exp|above",
              "H_exp|left,above", "H_exp|scale_octave", "H_exp|scale_rel_median", "H_exp|column_in_block", "H_exp|row_in_block",
              "H_exp|block_max_exp", "block_max_exp_side_bits_per_weight", "H_exp|left (a table a matrix)", "H_exp|above (a table a matrix)",
              "H_exp|left,above (a table a matrix)", "H_exp|scale_octave (a table a matrix)", "H_exp|scale_rel_median (a table a matrix)",
              "H_sign_mant_given_exp_per_matrix", "H_exp|column(table a column)", "H_exp|row(table a row)"):
        print(f"   {k:36s} {s[k]:.4f}")
    print(f"   best context: {s['best_context']}")
    print(f"   ceiling, order 0 pooled      {pct(s['ceiling_order0_pooled'])}")
    print(f"   ceiling, order 0 per matrix  {pct(s['ceiling_order0_per_matrix'])}")
    print(f"   ceiling, best contexts       {pct(s['ceiling_contexts'])}   (a table a matrix; with per-row/column tables, uncosted: {pct(s['ceiling_with_row_tables_uncosted'])})")
    print(f"   layout: exponent Huffman           {pct(s['layout_huffman_exp_bits'])}")
    print(f"   layout: exponent 3 bits + escape   {pct(s['layout_fast3_bits'])}  escapes {s['layout_fast3_escape_pct']:.2f}%")
    print(f"   layout: tiers of 2-bit digits (2)  {pct(s['layout_tier2_bits'])}  raw {s['layout_tier2_raw_pct']:.2f}%")
    print(f"   layout: tiers of 2-bit digits (3)  {pct(s['layout_tier3_bits'])}  raw {s['layout_tier3_raw_pct']:.2f}%")
    print(f"   top window of 8 exponents: exceptions {s['window8_top_exceptions_pct']:.2f}%; of 4: {s['window4_top_exceptions_pct']:.2f}%")


if __name__ == "__main__":
    args = [x for x in sys.argv[1:] if not x.startswith("--")]
    out = sys.argv[sys.argv.index("--json") + 1] if "--json" in sys.argv else None
    paths = [p for p in args if p.endswith(".json") and p != out]
    tot = load(paths)
    res = {}
    roles = {}
    for kind in sorted(tot):
        res[kind] = summarize(tot[kind])
        r = kind.split(":")[0]
        roles.setdefault(r, {})
        for k, v in tot[kind].items():
            roles[r][k] = roles[r][k] + v if k in roles[r] else v
    for r in sorted(roles):
        res["ROLE " + r] = summarize(roles[r])
        show("ROLE " + r, res["ROLE " + r])
    res["ALL"] = summarize(pool(tot))
    show("ALL fp8 pooled", res["ALL"])
    if out:
        json.dump(res, open(out, "w"), indent=1)
