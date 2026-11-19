"""inventory.json from headers/*.json and classify.py: per model the bytes packed / unpacked by class / not loaded, the projected
savings (labelled projections), and across models the unpacked classes ranked by bytes and by the models that have them."""
import glob, json, os, re
from collections import Counter, defaultdict
import classify as C

HERE = os.path.dirname(os.path.abspath(__file__))
GB = 1e9

# Canonical fix classes: each unpacked label -> (fix class, what fixes it)
CANON = [
    (r"^embedding \(built without", "embedding built without the quant config"),
    (r"^LM head \(built without", "LM head built without the quant config"),
    (r"^LM head \(head_dtype", "LM head kept in float32 (head_dtype)"),
    (r"^embedding \(a plain parameter", "embedding as a plain parameter (Inkling)"),
    (r"n-gram \(PLE\) embedding table", "n-gram (PLE) embedding table (Qwen4Exp: refuses the glyd config)"),
    (r"n-gram embedding table \(built", "n-gram embedding table built without the quant config (LongCat-Flash-Lite)"),
    (r"n-gram projections", "n-gram projections (plain parameter)"),
    (r"MoE experts \(activation relu2", "MoE experts: relu2 without a gate (Nemotron-H)"),
    (r"MoE experts \(activation gelu", "MoE experts: GELU-tanh (Gemma 4)"),
    (r"MoE experts \(activation swiglustep", "MoE experts: swiglustep (Step 3.5/3.7 clamped layers)"),
    (r"MoE experts \(activation swigluoai", "MoE experts: swigluoai (MiniMax-M3)"),
    (r"MoE experts \(", "MoE experts: other unsupported"),
    (r"shared \(sink\) experts", "shared sink experts as plain parameters (Inkling)"),
    (r"Mamba mixer Linears", "Mamba (v1) mixer Linears built without the quant config (Jamba)"),
    (r"^(vision|audio): tower (built without|not given)", "vision/audio tower not given the quant config"),
    (r"^(vision|audio): MLP not given", "vision/audio tower not given the quant config"),
    (r"^(vision|audio): layer built without", "vision/audio projector / patch Linears built without the quant config"),
    (r"^(vision|audio): Linear shape", "Linear shape: rows not a multiple of 64 (vision MLP 4304)"),
    (r"^Linear shape", "Linear shape: rows not a multiple of 64 (decoder)"),
    (r"^layer built without", "decoder Linears built without the quant config (hyper-connections, misc)"),
    (r"small Linears", "small Linears (< 1M weights)"),
    (r"MoE routers", "MoE routers (quant_config=None, fp32 gates)"),
    (r"conv|patch embed", "conv weights / patch embeddings"),
    (r"plain layer", "plain nn.Linear / nn.Embedding in towers"),
    (r"not bf16", "not bf16 (F32 tensors)"),
    (r"norms", "norms"),
    (r"biases", "biases"),
    (r"small params", "small params (A_log, dt_bias, D, scales)"),
]


def canon(label):
    for p, c in CANON:
        if re.search(p, label):
            return c
    return label


# Models in the lists that are not inventoried here, and why
EXCLUDED = {
    "moonshotai/Kimi-K3": "not a bf16 checkpoint: experts MXFP4 (U8, 2.72T values), 57B bf16 params (attention, embeddings)",
    "moonshotai/Kimi-K2.5": "not a bf16 checkpoint: experts INT4 (I32-packed, 1.01T), 12B bf16 params",
    "moonshotai/Kimi-K2.6": "not a bf16 checkpoint: experts INT4 (I32-packed), as K2.5",
    "moonshotai/Kimi-K2.7-Code": "not a bf16 checkpoint: experts INT4 (I32-packed), as K2.5",
    "swiss-ai/Apertus-v1.5-8B": "gated on Hugging Face: headers need an accepted license and a token (8.6B bf16 params + 294M F32; ApertusForCausalLM, as Apertus-v1.1-4B: everything packable)",
    "swiss-ai/Apertus-v1.5-70B": "gated on Hugging Face (71.7B bf16 params + 294M F32; ApertusForCausalLM)",
    "zai-org/GLM-5.1": "same architecture and parameter count as zai-org/GLM-5 (753,864,119,552 bf16): GLM-5's row applies (282 shards not re-read)",
    "zai-org/GLM-5.2": "same architecture and parameter count as zai-org/GLM-5.3-BF16 (753,329,921,024 bf16): GLM-5.3-BF16's row applies",
    "Motif-Technologies/Motif-3": "MotifForCausalLM: listed in vLLM 0.30.0's registry as removed (since 0.10.2): vLLM does not serve it",
    "ornith-ai/Ornith-1.5-397B": "a Qwen3.5-397B-A17B fine-tune (third party): Qwen3.5-397B-A17B's row applies",
    "ibm-granite/granite-4.0-3b-vision": "a LoRA layout (base_model.model.*, lora_A/lora_B over GraniteMoeHybrid): not a plain bf16 checkpoint",
    "google/gemma-4-E2B, -E4B, -12B, -26B-A4B, -31B (base)": "the same tensors as the -it rows",
    "google/gemma-4-*-it-assistant": "speculative drafts (vLLM keeps a draft bf16 unless --speculative-config asks for glyd)",
}
if os.path.exists(os.path.join(HERE, "more_excluded.json")):  # (the wider 2026 list: screened by the Hugging Face API)
    for _k, _v in json.load(open(os.path.join(HERE, "more_excluded.json"))).items():
        EXCLUDED.setdefault(_k, _v)
DERIVED = {"ornith-ai/Ornith-1.5-9B": "Qwen3.5-9B fine-tune (third party)", "ornith-ai/Ornith-1.5-35B-A3B": "Qwen3.5-35B-A3B fine-tune (third party)",
           "XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B": "Qwen3.5-9B architecture (distilled)", "Kwaipilot/KAT-Coder-V2.5-Dev": "Qwen3.5-MoE architecture"}


def main():
    models, ranking_b, ranking_n, fine_b, fine_n = [], Counter(), Counter(), Counter(), Counter()
    caveats = []
    for p in sorted(glob.glob(os.path.join(HERE, "headers", "*.json"))):
        rec = json.load(open(p))
        if rec["repo"] in EXCLUDED or not rec.get("shards"):
            continue
        rows, arch = C.classify(rec)
        if rows is None:
            print("NO FAMILY", rec["repo"], arch)
            continue
        s = C.summarize(rows, arch, rec)
        by = Counter()
        for lab, b in s["unpacked"].items():
            by[canon(lab)] += b
        loaded = s["loaded_bytes"]
        m = {
            "model": s["repo"], "arch": arch, "created": s["created"], "license": rec.get("license"),
            "derived": DERIVED.get(s["repo"]),
            "vllm_rules": ("auto from " + C.FAM[arch]["auto"] + " (embedding/head quant config, mtp; towers not hand-checked)") if C.FAM.get(arch, {}).get("auto") else "hand-read from vLLM 0.30.0's model file",
            "total_gb": round(s["total_bytes"] / GB, 3), "loaded_by_vllm_gb": round(loaded / GB, 3),
            "packed_gb": round(s["packed_bytes"] / GB, 3), "unpacked_gb": round(s["unpacked_bytes"] / GB, 3),
            "not_loaded_gb": {k: round(v / GB, 3) for k, v in s["not_loaded"].items()},
            "unpacked_by_class_gb": {k: round(v / GB, 4) for k, v in by.most_common() if v >= 0.0005 * GB},
            "unpacked_detail_gb": {k: round(v / GB, 4) for k, v in s["unpacked"].items() if v >= 0.0005 * GB},
            "towers_packed_gb": {k: round(v / GB, 3) for k, v in s["tower_packed"].items()},
            "projected_tiered_saving_pct": s["projected_tiered_saving_pct"],
            "projected_12bit_saving_pct": s["projected_12bit_saving_pct"],
            "projected_if_all_packed_tiered_pct": 32.5, "projected_if_all_packed_12bit_pct": 24.75,
            "gain_if_class_packed_tiered_pts": {k: round(100 * v * (1 - 10.80 / 16) / loaded, 2) for k, v in by.most_common() if 100 * v * (1 - 10.80 / 16) / loaded >= 0.01},
        }
        moe = any(st == "packed" and C.EXPERT.search(k) or (F_ := C.F_of(rec)).get("expert_re") and re.search(F_["expert_re"], k) for k, dt, sh, b, st, cls, where in rows)
        if moe:
            att = sum(b for k, dt, sh, b, st, cls, where in rows if st == "packed" and where == "decoder" and re.search(r"attn|attention", k))
            m["moe_attention_linears_bf16_on_a100_h100_gb"] = round(att / GB, 3)
            m["projected_12bit_saving_on_a100_h100_pct"] = round(100 * (s["packed_bytes"] - att) * (1 - 12.04 / 16) / loaded, 2)
        m["unpacked_classes"] = "; ".join(f"{k} {v:.2f} GB" for k, v in m["unpacked_by_class_gb"].items() if v >= 0.01)
        if s.get("blocker"):
            m["blocker"] = s["blocker"]
            m["today_saving_pct"] = 0.0
        if s["packed_with_caveat"]:
            m["packed_with_caveat_gb"] = {k: round(v / GB, 3) for k, v in s["packed_with_caveat"].items()}
            caveats.append(s["repo"])
        models.append(m)
        if m["derived"]:
            continue  # (a fine-tune repeats its base's classes: counted once)
        for k, v in by.items():
            ranking_b[k] += v
            ranking_n[k] += 1
    models.sort(key=lambda m: (m["arch"], m["model"]))
    ranking = [{"cls": k, "total_gb": round(v / GB, 3), "models": ranking_n[k]} for k, v in ranking_b.most_common()]
    ranking_by_models = sorted(ranking, key=lambda r: (-r["models"], -r["total_gb"]))
    out = {
        "what": "Per 2026 bf16 open model: the weights the glyd vLLM plugin (private/main glyd-gpu-py/glyd_gpu/vllm_plugin.py) packs under vLLM 0.30.0 and the ones it leaves unpacked, with bytes. Desk inventory from safetensors headers read by HTTP range and vLLM 0.30.0's model source; nothing run.",
        "date": "2026-10-06",
        "method": {
            "bytes": "checkpoint bytes (GB = 1e9) from each safetensors header; 'loaded by vLLM' drops what vLLM 0.30.0's model file maps to None or skips (MTP / nextn layers, Gemma 4's KV-shared k/v)",
            "packed": "GlydConfig.get_quant_method: a LinearBase built with the quant config of >= 1<<20 weights (vLLM's merged layer: q/k/v, gate/up, in_proj_qkv/z, b/a, MLA q_a/kv_a summed), rows % 64 == 0, columns % 16 == 0, bf16; RoutedExperts: SiLU, no bias, no expert parallel, H % 64, 2I % 128, I % 16; VocabParallelEmbedding / ParallelLMHead (exact types) built with the quant config: dim % 64, >= 4096 rows, head_dtype bf16. Layout rule on an RTX 4090/5090 (on an A100/H100 a MoE model's attention Linears stay bf16: dense_stays)",
            "projections": "PROJECTED, not measured: saving = packed bytes x (1 - bits/16) / bytes vLLM loads; tiered 10.80 bits a weight, 12-bit 12.04. Ceilings with everything packed: 32.5% / 24.75%",
            "check": "Qwen3.5-9B: 17.53 GiB loaded (18.83 GB), 15.36 GiB of it packed -> 12.54 GiB tiered + 0.13 GiB vLLM's own buffers (bf16's measured 17.66 GiB less 17.53 checkpoint GiB) = 12.67 GiB against 12.68 GiB measured (RTX 4090, v0.29, 2026-10-05); 12-bit: 13.86 against 13.89 GiB measured (RTX 5090). The alternatives (embedding packed: 12.05; vision tower unpacked: 12.86) do not match.",
        },
        "models": models,
        "ranking": ranking,
        "ranking_by_models": ranking_by_models,
        "ranking_note": "unpacked classes over the models inventoried (fine-tunes of a listed base counted once), by GB and by number of models with >= 0.5 MB of the class",
        "excluded": EXCLUDED,
        "packed_with_caveat": caveats,
    }
    json.dump(out, open(os.path.join(HERE, "inventory.json"), "w"), indent=1)
    for m in models:
        print(f"{m['model']:52s} {m['loaded_by_vllm_gb']:9.1f} GB  packed {m['packed_gb']:9.1f}  tiered {m['projected_tiered_saving_pct']:5.2f}%  12b {m['projected_12bit_saving_pct']:5.2f}%  {'BLOCKED' if m.get('blocker') else ''}")
    print()
    for r in ranking:
        print(f"{r['total_gb']:10.2f} GB  {r['models']:3d} models  {r['cls']}")


if __name__ == "__main__":
    main()


def readme():
    d = json.load(open(os.path.join(HERE, "inventory.json")))
    L = []
    L.append("# 2026 bf16 models under the glyd vLLM plugin: what is packed, what is not\n")
    L.append("Desk inventory, %s. For every 2026 open bf16 model read, each checkpoint tensor is classified as the plugin at private/main "
             "(`glyd-gpu-py/glyd_gpu/vllm_plugin.py`) packs it under vLLM 0.30.0: packed, left unpacked (why), or not loaded by vLLM. "
             "Nothing was run: safetensors headers were read by HTTP range and vLLM 0.30.0's model files (the sdist) were read for "
             "what each architecture builds with the quant config. The tensor-level data is `inventory.json`; the scripts that made it "
             "are beside it.\n" % d["date"])
    L.append("**Method.** " + d["method"]["packed"] + ".\n")
    L.append("**Projections, not measurements.** " + d["method"]["projections"] + ".\n")
    L.append("**Check against a measurement.** " + d["method"]["check"] + "\n")
    L.append("## Findings\n")
    L.append(open(os.path.join(HERE, "findings.md")).read().strip() + "\n")
    L.append("## Per model\n")
    L.append("GB = 1e9 bytes, as in the checkpoint. Loaded = what vLLM 0.30.0 loads (MTP and nextn layers are not). Tiered / 12-bit: "
             "projected whole-model saving over the loaded bytes. A row marked * uses vLLM rules read automatically from the model file "
             "(embedding and head only); † a fine-tune or same-architecture derivative.\n")
    L.append("| Model | Arch | Loaded GB | Packed GB | Unpacked GB | Unpacked, by class (GB) | Tiered | 12-bit |")
    L.append("| :-- | :-- | --: | --: | --: | :-- | --: | --: |")
    for m in sorted(d["models"], key=lambda m: m["model"].lower()):
        cls = "; ".join(f"{k} {v:.2f}" for k, v in m["unpacked_by_class_gb"].items() if v >= 0.01) or "-"
        mark = ("*" if m["vllm_rules"].startswith("auto") else "") + ("†" if m.get("derived") else "")
        t = f"{m['projected_tiered_saving_pct']:.1f}%"
        b = f"{m['projected_12bit_saving_pct']:.1f}%"
        if m.get("blocker"):
            t, b = f"0 (does not load; {t} once it does)", f"0 ({b})"
        L.append(f"| {m['model']}{mark} | {m['arch']} | {m['loaded_by_vllm_gb']:,.1f} | {m['packed_gb']:,.1f} | {m['unpacked_gb']:,.2f} | {cls} | {t} | {b} |")
    L.append("\n## Unpacked classes, ranked\n")
    L.append(d["ranking_note"] + ".\n")
    L.append("| Class | GB | Models |")
    L.append("| :-- | --: | --: |")
    for r in d["ranking"]:
        if r["total_gb"] >= 0.005:
            L.append(f"| {r['cls']} | {r['total_gb']:,.2f} | {r['models']} |")
    L.append("\nThe same classes by how many models have them (the classes of 5 or more models):\n")
    L.append("| Class | Models | GB |")
    L.append("| :-- | --: | --: |")
    for r in sorted(d["ranking"], key=lambda r: (-r["models"], -r["total_gb"])):
        if r["models"] >= 5:
            L.append(f"| {r['cls']} | {r['models']} | {r['total_gb']:,.2f} |")
    L.append("")
    L.append(open(os.path.join(HERE, "fixplan.md")).read().strip() + "\n")
    L.append("## Not inventoried\n")
    for k, v in d["excluded"].items():
        L.append(f"- {k}: {v}")
    L.append("\n## Files\n")
    L.append("- `inventory.json`: per model the bytes (total, loaded, packed, unpacked by class and in detail, not loaded), the towers' packed bytes, the projections, the gain per class, the vLLM rules used; the ranking; the exclusions.")
    L.append("- `fetch.py` (headers by HTTP range), `classify.py` (the plugin's rules and the per-architecture vLLM facts, each with the file it was read from), `autofam.py` (the automatic rules for architectures not read by hand), `report.py` (inventory.json and this file), `fixplan.md`, `findings.md`.")
    L.append("- Re-run: `pip download vllm==0.30.0 --no-deps --no-binary :all:` and unpack it (`VLLM_SRC` names the folder), `python3 fetch.py repos_all.txt` (headers into headers/, about 2 s a shard; Hugging Face rate-limits bursts), `python3 screen.py LIST screen.json && python3 more.py` for the wider list, then `python3 report.py && README=1 python3 report.py`.")
    open(os.path.join(HERE, "README.md"), "w").write("\n".join(L) + "\n")


if __name__ == "__main__" and os.environ.get("README"):
    readme()
