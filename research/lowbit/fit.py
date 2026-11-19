"""The size and fit arithmetic: what each checkpoint weighs under each layout, and whether it fits a node.

    python fit.py            # reads results/*.json, writes results/fit.json and prints the tables

Inputs are the measured summaries (results/fp4_*.json, fp8_*.json, bf16_*.json) and the checkpoints' header censuses
(census_*.json). Assumptions that are not measurements are at the top, in CAPS.
"""
import glob, json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "results") + "/"
GB = 1e9
OVERHEAD_VARIABLE = 0.05  # bits a weight, assumed: stream offsets / padding of an entropy-coded layout (not measured)
BF16_PRACTICAL_EXTRA = 0.2  # bits: Glyd's tiered bf16 code (10.80) over the per-tensor floor (~10.6) in gpu/README.md
BF16_UNMEASURED = 10.55  # bits: embeddings measured here (Qwen3-8B 10.56/10.53, DeepSeek-V3 10.51)
RESERVE_PER_GPU = 8.0  # GB a GPU for CUDA context, NCCL, activations, graphs (assumed)
NODES = {"8xB200 (192 GB, 1,536 GB)": 8 * 192, "8xB200 (DGX/HGX, 180 GB, 1,440 GB)": 8 * 180, "8xH200 (141 GB)": 8 * 141, "8xH100 (80 GB)": 8 * 80}


def J(name):
    return json.load(open(name if os.path.isabs(name) else R + name))


def census(name):
    c = J(name)
    return {(k["role"], k["payload"]): (k["elements"], k["bytes"]) for k in c["kinds"]}, c["total_bytes"]


def bf16_floor(files):
    """Measured order-0 floor (bits a weight, H of the 16-bit word a tensor) by role, pooled over files."""
    acc = {}
    for f in files:
        for role, a in J(f).items():
            if role == "_meta" or not a["big_n"]:
                continue
            n, s = acc.get(role, (0, 0.0))
            acc[role] = (n + a["big_n"], s + a["sums"]["H_word"])
    return {r: s / n for r, (n, s) in acc.items()}


def bf16_bytes(cen, floor, extra):
    """Bytes of the bf16 tensors of a census after a bf16 code at floor + extra bits a weight."""
    out = 0.0
    for (role, payload), (el, by) in cen.items():
        if payload == "bf16":
            out += el * (floor.get(role, BF16_UNMEASURED) + extra) / 8
    return out


def other_bytes(cen):
    """Everything that is neither a bf16 tensor nor a coded payload: left as stored."""
    return sum(by for (role, payload), (el, by) in cen.items() if payload not in ("bf16", "fp4_packed", "fp8", "scale"))


def kv_room(total_gb, cap_gb, ngpu, kv_seq_gb):
    free = cap_gb - total_gb - RESERVE_PER_GPU * ngpu
    return free, max(0.0, free / kv_seq_gb)


def fit_table(name, layouts, kv_seq_gb, nodes):
    rows = []
    for lname, gb in layouts.items():
        row = {"layout": lname, "weights_GB": round(gb, 1)}
        for nn, cap in nodes.items():
            free, seqs = kv_room(gb, cap, 8 if nn.startswith("8x") else 1, kv_seq_gb)
            row[nn] = {"headroom_GB": round(cap - gb, 1), "free_after_reserve_GB": round(free, 1), "seqs_128k": round(seqs, 1), "fits_weights_reserve_and_one_128k": bool(free >= kv_seq_gb)}
        rows.append(row)
    return rows


res = {"assumptions": {"overhead_variable_bits": OVERHEAD_VARIABLE, "bf16_practical_extra_bits": BF16_PRACTICAL_EXTRA,
                       "bf16_unmeasured_bits": BF16_UNMEASURED, "reserve_per_gpu_GB": RESERVE_PER_GPU}}

# ---------------------------------------------------------------- Kimi K3
cen, total = census("census_moonshotai_Kimi-K3.json")
files = sorted(glob.glob(R + "fp4_k3_s*.json"))
S = [J(os.path.basename(f))["ALL"] for f in files]
mean = lambda k: float(np.mean([s[k] for s in S]))
nw = 2 * cen[("routed_experts", "fp4_packed")][1]
exp_stored = cen[("routed_experts", "fp4_packed")][1] + cen[("routed_experts", "scale")][1]
bits = {"stored": 4.25, "ceiling_order0_per_matrix": mean("ceiling_order0_per_matrix"), "ceiling_contexts": mean("ceiling_contexts"),
        "scales_2bit_nibbles_raw": mean("layout_scales_only_bits"), "scales_2bit_nibbles_huffman_by_scale": mean("layout_scales_huffman_by_scale_bits") + OVERHEAD_VARIABLE}
bf = bf16_floor(sorted(glob.glob(R + "bf16_k3_s*.json")))
others = other_bytes(cen)
lay = {}
for nm, b, extra in (("as stored", 4.25, None), ("scales in 2 bits, nibbles raw (fixed rate)", bits["scales_2bit_nibbles_raw"], BF16_PRACTICAL_EXTRA),
                     ("scales in 2 bits, nibbles Huffman by scale (+0.05 b offsets)", bits["scales_2bit_nibbles_huffman_by_scale"], BF16_PRACTICAL_EXTRA),
                     ("ceiling, order 0 per matrix", bits["ceiling_order0_per_matrix"], 0.0), ("ceiling, best contexts", bits["ceiling_contexts"], 0.0)):
    if extra is None:
        lay[nm] = total / GB
    else:
        lay[nm] = (nw * b / 8 + bf16_bytes(cen, bf, extra) + others) / GB
# KV a 128k sequence: MLA layers (24) latent cache 512 + 64 values a token in bf16, KDA layers (69) a 96 x 128 x 128 fp32 state
kv_k3 = (24 * (512 + 64) * 2 * 131072 + 69 * 96 * 128 * 128 * 4) / GB
res["kimi_k3"] = {"layers_sampled": [os.path.basename(f) for f in files], "expert_weights_G": nw / 1e9, "experts_stored_GB": exp_stored / GB,
                  "total_stored_GB": total / GB, "bits_a_weight": bits, "bf16_floor_by_role": bf, "kv_a_128k_sequence_GB": kv_k3,
                  "sizes_GB": lay, "fit": fit_table("k3", lay, kv_k3, {k: v for k, v in NODES.items() if "B200" in k or "H200" in k})}

# ---------------------------------------------------------------- gpt-oss-120b
cen, total = census("census_openai_gpt-oss-120b.json")
files = sorted(glob.glob(R + "fp4_gptoss_s*.json"))
S = [J(os.path.basename(f))["ALL"] for f in files]
nw = 2 * cen[("routed_experts", "fp4_packed")][1]
bits = {"stored": 4.25, "ceiling_order0_per_matrix": mean("ceiling_order0_per_matrix"), "ceiling_contexts": mean("ceiling_contexts"),
        "scales_2bit_nibbles_raw": mean("layout_scales_only_bits"), "scales_2bit_nibbles_huffman_by_scale": mean("layout_scales_huffman_by_scale_bits") + OVERHEAD_VARIABLE}
bf = bf16_floor(sorted(glob.glob(R + "bf16_gptoss_s*.json")))
others = other_bytes(cen)
lay = {}
for nm, b, extra in (("as stored", 4.25, None), ("scales in 2 bits, nibbles raw (fixed rate)", bits["scales_2bit_nibbles_raw"], BF16_PRACTICAL_EXTRA),
                     ("scales in 2 bits, nibbles Huffman by scale (+0.05 b offsets)", bits["scales_2bit_nibbles_huffman_by_scale"], BF16_PRACTICAL_EXTRA),
                     ("ceiling, order 0 per matrix", bits["ceiling_order0_per_matrix"], 0.0), ("ceiling, best contexts", bits["ceiling_contexts"], 0.0)):
    lay[nm] = total / GB if extra is None else (nw * b / 8 + bf16_bytes(cen, bf, extra) + others) / GB
# KV a 128k sequence: 18 full-attention layers, 8 KV heads x 64, bf16; the 18 sliding layers hold 128 tokens
kv_g = (18 * 2 * 8 * 64 * 2 * 131072 + 18 * 2 * 8 * 64 * 2 * 128) / GB
res["gpt_oss_120b"] = {"layers_sampled": [os.path.basename(f) for f in files], "expert_weights_G": nw / 1e9, "total_stored_GB": total / GB, "bits_a_weight": bits,
                       "bf16_floor_by_role": bf, "kv_a_128k_sequence_GB": kv_g, "sizes_GB": lay,
                       "fit": fit_table("gptoss", lay, kv_g, {"1x80GB (H100/A100)": 80, "1x141GB (H200)": 141})}

# ---------------------------------------------------------------- DeepSeek-V3 (FP8, 128x128 block scales)
cen, total = census("census_deepseek-ai_DeepSeek-V3.json")
sh = {3: "fp8_ds3_s00001.json", 9.5: "fp8_ds3_s00021.json", 31: "fp8_ds3_s00079.json", 55.5: "fp8_ds3_s00147.json"}  # layers each sample stands for
layers = np.arange(3, 62)
KEYS = {"ceiling_contexts": "ceiling_contexts", "ceiling_order0_per_matrix": "ceiling_order0_per_matrix", "huffman": "layout_huffman_exp_bits",
        "tiered3": "layout_tier3_bits", "fast3": "layout_fast3_bits"}


def routed(key, how):
    pts = np.array([J(f)["ROLE routed"][KEYS[key]] for f in sh.values()])
    x = np.array(list(sh.keys()))
    return float(np.interp(layers, x, pts).mean()) if how == "interp" else float(pts.mean())


pooled = J("fp8_ds3_s00001.json")  # the other roles: the four samples pooled
roles = {}
for r in ("attention", "dense_mlp", "shared"):
    roles[r] = {k: float(np.mean([J(f)["ROLE " + r][KEYS[k]] for f in sh.values() if ("ROLE " + r) in J(f)])) for k in KEYS}
role_of = {"attention": "attention", "dense_mlp": "dense_mlp", "shared_experts": "shared"}
bf = bf16_floor(sorted(glob.glob(R + "bf16_ds3_s*.json")))
others = other_bytes(cen) + sum(by for (role, payload), (el, by) in cen.items() if payload == "scale")
fp8_el = {k: v[0] for k, v in cen.items() if k[1] == "fp8"}
lay = {"as stored": total / GB}
detail = {}
for how in ("interp", "equal"):
    for lk, label in (("fast3", "exponent 3 bits + escape"), ("tiered3", "exponent in tiers of 2-bit digits"), ("huffman", "exponent Huffman"), ("ceiling_order0_per_matrix", "ceiling, order 0 per matrix"), ("ceiling_contexts", "ceiling, best contexts")):
        b = {"routed_experts": routed(lk, how), **{r: roles[rr][lk] for r, rr in role_of.items()}}
        extra = 0.0 if lk.startswith("ceiling") else BF16_PRACTICAL_EXTRA
        var = OVERHEAD_VARIABLE if lk in ("tiered3", "huffman") else 0.0
        fp8 = sum(el * (b[role] + var) / 8 for (role, _), el in fp8_el.items())
        lay[f"{label} [{how}]"] = (fp8 + bf16_bytes(cen, bf, extra) + others) / GB
        detail[f"{label} [{how}]"] = {"routed_bits": b["routed_experts"] + var}
kv_v3 = (61 * (512 + 64) * 2 * 131072) / GB
res["deepseek_v3"] = {"total_stored_GB": total / GB, "bits_a_weight_routed": detail, "bf16_floor_by_role": bf, "kv_a_128k_sequence_GB": kv_v3, "sizes_GB": lay,
                      "fit": fit_table("v3", lay, kv_v3, {k: v for k, v in NODES.items() if "H100" in k or "H200" in k})}

# ---------------------------------------------------------------- Qwen3-8B-FP8 (whole)
cen, total = census("census_Qwen_Qwen3-8B-FP8.json")
q = J("fp8_qwen3_8b.json")
bf = bf16_floor(sorted(glob.glob(R + "bf16_qwen_s*.json")))
others = other_bytes(cen) + sum(by for (role, payload), (el, by) in cen.items() if payload == "scale")
lay = {"as stored": total / GB}
for lk, label in (("fast3", "exponent 3 bits + escape"), ("tiered3", "exponent in tiers of 2-bit digits"), ("huffman", "exponent Huffman"), ("ceiling_order0_per_matrix", "ceiling, order 0 per matrix"), ("ceiling_contexts", "ceiling, best contexts")):
    fp8 = 0.0
    for (role, payload), (el, by) in cen.items():
        if payload == "fp8":
            rr = "ROLE attention" if role == "attention" else "ROLE dense_mlp"
            fp8 += el * (q[rr][KEYS[lk]] + (OVERHEAD_VARIABLE if lk in ("tiered3", "huffman") else 0)) / 8
    lay[label] = (fp8 + bf16_bytes(cen, bf, 0.0 if lk.startswith("ceiling") else BF16_PRACTICAL_EXTRA) + others) / GB
res["qwen3_8b_fp8"] = {"total_stored_GB": total / GB, "sizes_GB": lay}

json.dump(res, open(R + "fit.json", "w"), indent=1)
for m in ("kimi_k3", "gpt_oss_120b", "deepseek_v3", "qwen3_8b_fp8"):
    r = res[m]
    print(f"\n== {m}: stored {r['total_stored_GB']:.1f} GB" + (f", KV a 128k sequence {r['kv_a_128k_sequence_GB']:.2f} GB" if "kv_a_128k_sequence_GB" in r else ""))
    for k, v in r["sizes_GB"].items():
        print(f"   {k:70s} {v:9.1f} GB  ({100 * (1 - v / r['total_stored_GB']):5.1f}% off)")
    for row in r.get("fit", []):
        print("   fit:", row["layout"][:60].ljust(60), {k: (v["free_after_reserve_GB"], v["seqs_128k"]) for k, v in row.items() if isinstance(v, dict)})
