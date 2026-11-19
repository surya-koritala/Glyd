"""Entropies and layout sizes (bits a weight) from fp4.py's counts.

    python report_fp4.py OUT.json [OUT.json ...] [--json SUMMARY.json]

Several files are summed (shards of one model); each is also printed alone.
"""
import json, sys
import numpy as np
from ent import H, Hc, huffman_bits, fast_bits, tiered_bits


def load(paths):
    tot = {}
    for p in paths:
        d = json.load(open(p))
        d.pop("_meta", None)
        for kind, a in d.items():
            t = tot.setdefault(kind, {})
            for k, v in a.items():
                v = np.array(v) if isinstance(v, list) else v
                t[k] = t[k] + v if k in t else v
    return tot


def pool(tot, kinds=None):
    r = {}
    for kind, a in tot.items():
        if kinds and kind not in kinds:
            continue
        for k, v in a.items():
            r[k] = r[k] + v if k in r else v
    return r


def summarize(a):
    code = np.asarray(a["code"], dtype=np.float64)
    N = code.sum()
    Ns = np.asarray(a["s"]).sum()
    s = {"weights": int(N), "scales": int(Ns), "matrices": int(a["matrices"])}
    s["code_hist_pct"] = (100 * code / N).round(3).tolist()
    s["H_code"] = H(code)
    s["H_code_per_matrix"] = float(a["perm"][1] / a["perm"][0])
    s["H_sign"] = H(code.reshape(2, 8).sum(1))
    s["H_mag_given_sign"] = s["H_code"] - s["H_sign"]
    s["H_code|prev"] = Hc(np.asarray(a["adj"]).reshape(16, 16))
    s["H_code|prev2"] = Hc(np.asarray(a["adj2"]).reshape(256, 16))
    s["H_code|above"] = Hc(np.asarray(a["up"]).reshape(16, 16))
    s["H_code|position"] = Hc(np.asarray(a["pos"]).reshape(32, 16))
    s["H_code|scale"] = Hc(np.asarray(a["sc"]).reshape(256, 16))
    s["H_pair/2"] = H(a["byte"]) / 2
    for m, n in (("bmax", 8), ("bnz", 8), ("bsq", 8), ("bms", 64)):  # block summary as side information
        hist = np.asarray(a[m + "_hist"], dtype=np.float64)
        s[f"R_block_{m}"] = Hc(np.asarray(a[m]).reshape(n, 16)) + H(hist) / 32  # bits a weight, side info included
        s[f"H_{m}_side_bits_per_block"] = H(hist)
    # scales
    s["H_scale"] = H(a["s"])
    s["H_scale_per_matrix"] = float(a["perm"][3] / a["perm"][2])
    s["H_scale|left"] = Hc(np.asarray(a["sl"]).reshape(256, 256))
    s["H_scale|above"] = Hc(np.asarray(a["su"]).reshape(256, 256))
    s["H_scale_rowrel"] = H(a["srd"])
    s["scale_outside_clip_pct"] = 100 * float(a["srd_out"][0]) / Ns
    s["H_scale_rowrel|left"] = Hc(np.asarray(a["srd_l"]).reshape(16, 16))
    s["H_scale_rowrel|above"] = Hc(np.asarray(a["srd_u"]).reshape(16, 16))
    s["H_scale_rowrel|left,above"] = Hc(np.asarray(a["srd_lu"]).reshape(256, 16))
    s["H_scale_rowcolrel"] = H(a["srd2"])
    # layouts for the scales: a k-bit code, 2^k - 1 offsets from the row's base and an escape (8 raw bits in a side list)
    win = np.asarray(a["win_row"], dtype=np.float64)
    wm = np.asarray(a["win_mat"], dtype=np.float64)
    for k in (1, 2, 3, 4):
        w = 2 ** k - 1
        esc = 1 - win[w - 1] / Ns
        escm = 1 - wm[w - 1] / Ns
        s[f"scale_k{k}_row_escape_pct"] = 100 * esc
        s[f"scale_k{k}_row_bits_per_scale"] = k + 8 * esc
        s[f"scale_k{k}_matrix_escape_pct"] = 100 * escm
        s[f"scale_k{k}_matrix_bits_per_scale"] = k + 8 * escm
    # codes: Huffman on the pooled 16-symbol table, and with a table for each scale value (the scale is decoded first)
    s["code_huffman_bits"] = huffman_bits(code)
    sc = np.asarray(a["sc"], dtype=np.float64).reshape(256, 16)
    s["H_code|scale_value_check"] = sum(sc[v].sum() * H(sc[v]) for v in range(256) if sc[v].sum() > 0) / N
    s["code_huffman_by_scale_bits"] = sum(sc[v].sum() * huffman_bits(sc[v]) for v in range(256) if sc[v].sum() > 0) / N
    # bits a weight, all in
    raw = 4 + 8 / 32
    s["bits_stored"] = raw
    s["ceiling_order0_pooled"] = s["H_code"] + s["H_scale"] / 32
    s["ceiling_order0_per_matrix"] = s["H_code_per_matrix"] + s["H_scale_per_matrix"] / 32
    s["ceiling_contexts"] = min(s["H_code|prev"], s["H_code|prev2"], s["H_code|above"], s["H_code|position"], s["H_code|scale"],
                                s["R_block_bmax"], s["R_block_bnz"], s["R_block_bsq"], s["R_block_bms"]) + \
        min(s["H_scale|left"], s["H_scale|above"], s["H_scale_rowrel|left,above"], s["H_scale_rowcolrel"]) / 32
    # a row's base scale is a byte a row (8 / columns a weight; 3,584 or 3,072 columns in K3, 2,880 in gpt-oss: 0.003 bits)
    base = 8 / 3200
    s["layout_scales_only_bits"] = 4 + s["scale_k2_row_bits_per_scale"] / 32 + base
    s["layout_scales_huffman_bits"] = s["code_huffman_bits"] + s["scale_k2_row_bits_per_scale"] / 32 + base
    s["layout_scales_huffman_by_scale_bits"] = s["code_huffman_by_scale_bits"] + s["scale_k2_row_bits_per_scale"] / 32 + base
    return s


def show(name, s):
    raw = s["bits_stored"]
    pct = lambda b: f"{b:.3f} ({100 * (1 - b / raw):+.1f}%)".replace("+", "-").replace("--", "+") if False else f"{b:.3f} ({100 * (1 - b / raw):.1f}% off)"
    print(f"== {name}: {s['weights'] / 1e9:.3f} G weights in {s['matrices']} matrices, stored {raw} bits a weight")
    print("   code histogram % (sign x magnitude idx 0..7):", s["code_hist_pct"][:8], s["code_hist_pct"][8:])
    for k in ("H_code", "H_code_per_matrix", "H_sign", "H_mag_given_sign", "H_code|prev", "H_code|prev2", "H_code|above", "H_code|position", "H_code|scale", "H_pair/2",
              "R_block_bmax", "R_block_bnz", "R_block_bsq", "R_block_bms", "code_huffman_bits"):
        print(f"   {k:28s} {s[k]:.4f}")
    for k in ("H_scale", "H_scale_per_matrix", "H_scale|left", "H_scale|above", "H_scale_rowrel", "H_scale_rowrel|left", "H_scale_rowrel|above",
              "H_scale_rowrel|left,above", "H_scale_rowcolrel", "scale_outside_clip_pct"):
        print(f"   {k:28s} {s[k]:.4f}")
    for k in (1, 2, 3, 4):
        print(f"   scales k={k}: row base {s[f'scale_k{k}_row_bits_per_scale']:.3f} bits ({s[f'scale_k{k}_row_escape_pct']:.2f}% escapes)   matrix base {s[f'scale_k{k}_matrix_bits_per_scale']:.3f} bits ({s[f'scale_k{k}_matrix_escape_pct']:.2f}%)")
    print(f"   ceiling, order 0 pooled     {pct(s['ceiling_order0_pooled'])}")
    print(f"   ceiling, order 0 per matrix {pct(s['ceiling_order0_per_matrix'])}")
    print(f"   ceiling, best contexts      {pct(s['ceiling_contexts'])}")
    print(f"   layout: codes raw + scales k=2 row base   {pct(s['layout_scales_only_bits'])}")
    print(f"   layout: codes Huffman + scales k=2        {pct(s['layout_scales_huffman_bits'])}")
    print(f"   layout: codes Huffman by scale + scales k=2 {pct(s['layout_scales_huffman_by_scale_bits'])}   (Huffman by scale value {s['code_huffman_by_scale_bits']:.4f}, entropy by scale value {s['H_code|scale_value_check']:.4f})")


if __name__ == "__main__":
    args = [x for x in sys.argv[1:] if not x.startswith("--")]
    paths = [x for x in args if x.endswith(".json") and "summary" not in x]
    out = sys.argv[sys.argv.index("--json") + 1] if "--json" in sys.argv else None
    if out:
        paths = [p for p in paths if p != out]
    res = {}
    tot = load(paths)
    for kind in sorted(tot):
        res[kind] = summarize(tot[kind])
        show(kind, res[kind])
    res["ALL"] = summarize(pool(tot))
    show("ALL kinds pooled", res["ALL"])
    if out:
        json.dump(res, open(out, "w"), indent=1)
