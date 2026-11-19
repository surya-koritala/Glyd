"""Every tensor of each fetched checkpoint classified as the glyd vLLM plugin (private/main) packs it under vLLM 0.30.0:
packed today, not packed (why), or not loaded by vLLM. Rules: vllm_plugin.py's GlydConfig.get_quant_method (a LinearBase built
with the quant config: >= 1 << 20 weights, rows % 64, columns % 16, bf16; RoutedExperts: SiLU, no bias, H % 64, 2I % 128,
I % 16; VocabParallelEmbedding / ParallelLMHead built with the quant config: dim % 64, >= 4096 rows), and per architecture
what vLLM 0.30.0's model file builds with the quant config (FAM below, read from vllm/model_executor/models/*.py)."""
import glob, json, os, re, sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
SIZE = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8, "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1,
        "F8_E4M3": 1, "F8_E5M2": 1, "F8_E8M0": 1, "F4": 0.5}
BITS = {"tiered": 10.80, "12bit": 12.04}
SMALL = 1 << 20
MERGE = {"q_proj": "qkv", "k_proj": "qkv", "v_proj": "qkv", "gate_proj": "gate_up", "up_proj": "gate_up",
         "in_proj_qkv": "qkvz", "in_proj_z": "qkvz", "in_proj_b": "ba", "in_proj_a": "ba", "w1": "w13", "w3": "w13"}
MLP4304 = "MLP not given the quant config by vLLM (intermediate not a multiple of 64)"
EXPERT = re.compile(r"\.experts\.(?:(\d+)\.)?(gate_up_proj|gate_proj|up_proj|down_proj|w1|w2|w3|w13|linear_fc1|linear_fc2|gate_up|down)(?:\.weight)?$")

# Per architecture, from vLLM 0.30.0's model files:
#   drop: checkpoint names vLLM does not load (its WeightsMapper maps them to None, or load_weights skips them)
#   towers: (pattern, kind, qc) a vision/audio tower; qc: its Linears are LinearBase built with the quant config
#   tower_plain: tower tensors that are nn.Linear / nn.Embedding / conv (never quant config)
#   embed_qc / head_qc: VocabParallelEmbedding / ParallelLMHead built with the quant config
#   noqc: decoder tensors in a LinearBase built with quant_config=None (or a plain nn.Linear / nn.Parameter matrix)
#   act: the experts' activation; ebias: experts with biases
QWEN35 = dict(drop=[r"^mtp\."], towers=[(r"(^|\.)visual\.", "vision", True)], tower_plain=[r"visual\.pos_embed", r"patch_embed"],
              embed_qc=False, head_qc=True, noqc=[r"\.conv1d\."], act="silu")
FAM = {
    "Qwen3_5ForConditionalGeneration": QWEN35,
    "Qwen3_5MoeForConditionalGeneration": QWEN35,
    "Qwen3_5MoeForCausalLM": dict(QWEN35, towers=[]),
    # qwen3_next.py: embed_tokens and lm_head built without quant_config; mtp. mapped to None
    "Qwen3NextForCausalLM": dict(drop=[r"^mtp\."], embed_qc=False, head_qc=False, noqc=[r"\.conv1d\."], act="silu"),
    # vllm/models/qwen4_exp: embed_tokens and lm_head without quant_config; the hyper-connection Linears quant_config=None;
    # the PLE n-gram table's method (Qwen4ExpPLEEmbeddingMethod.from_quant_config) raises NotImplementedError for any
    # quant config but FP8 / ModelOpt: the model does not load under --quantization glyd at all.
    "Qwen4ExpForConditionalGeneration": dict(
        QWEN35, head_qc=False, noqc=[r"\.conv1d\.", r"hyper_connection", r"block_inject"],
        merge={"key_proj": "kv", "value_proj": "kv"},
        special=[(r"ngram_embedding\.shard_\d+\.weight$", "n-gram (PLE) embedding table")],
        blocker="vLLM's Qwen4Exp PLE embedding raises NotImplementedError for a quant config other than FP8/ModelOpt: the model does not load with --quantization glyd"),
}


def nbytes(dt, sh):
    n = 1
    for s in sh:
        n *= s
    return int(n * SIZE[dt])


def numel(sh):
    n = 1
    for s in sh:
        n *= s
    return n


def tcfg(c):
    t = c.get("text_config") or c.get("llm_config") or c.get("language_config") or c
    return t


def classify(rec):
    c = rec.get("config") or {}
    arch = (c.get("architectures") or ["?"])[0]
    F = FAM.get(arch)
    tc = tcfg(c)
    if F is None:
        import autofam
        F = autofam.family(arch, (tc.get("architectures") or [None])[0] if tc is not c else None)
        if F is None:
            return None, arch
        FAM[arch] = F
    if "fn" in F:
        F = dict(F, **F["fn"](c, tc))
    tensors = {k: v for t in rec["shards"].values() for k, v in t.items()}
    # merged groups (vLLM's qkv_proj, gate_up_proj, ...): rows summed, by parent module
    groups = defaultdict(lambda: [0, 0])
    for k, (dt, sh) in tensors.items():
        if len(sh) == 2 and k.endswith(".weight"):
            parent, leaf = k[: -len(".weight")].rsplit(".", 1) if "." in k[: -len(".weight")] else ("", k[: -len(".weight")])
            g = MERGE.get(leaf)
            for a, b in F.get("merge", {}).items():
                if leaf == a:
                    g = b
            if g:
                groups[(parent, g)][0] += sh[0]
                groups[(parent, g)][1] = sh[1]
    if "prep" in F:
        F = dict(F, **F["prep"](tensors, c, tc))
    out = []
    for k, (dt, sh) in tensors.items():
        b = nbytes(dt, sh)
        out.append((k, dt, sh, b) + decide(k, dt, sh, F, tc, groups))
    return out, arch


def decide(k, dt, sh, F, tc, groups):
    """(state, cls, where): state packed / unpacked / not_loaded."""
    for p in F.get("drop", []):
        if re.search(p, k):
            return "not_loaded", F.get("drop_cls", {}).get(p, "mtp" if "mtp" in p else "not loaded"), "-"
    if F.get("drop_extra_layers"):
        m = re.search(r"(?:^|\.)layers\.(\d+)\.", k)
        if m and int(m.group(1)) >= tc.get("num_hidden_layers", 1 << 30):
            return "not_loaded", "mtp", "-"
    if "drop_fn" in F:
        why = F["drop_fn"](k, tc)
        if why:
            return "not_loaded", why, "-"
    where, tower_qc = "decoder", True
    for p, kind, qc in F.get("towers", []):
        if re.search(p, k):
            where, tower_qc = kind, qc
    pre = "" if where == "decoder" else where + ": "
    if F.get("mtp_where") and re.search(F["mtp_where"], k):
        where = "mtp"
    nd = len(sh)
    leaf = k.rsplit(".", 1)[-1]
    for p, cls in F.get("special", []):
        if re.search(p, k):
            return ("packed" if cls == "packed" else "unpacked"), (cls if cls == "packed" else pre + cls), where
    if nd == 2 and re.search(F.get("router", r"\.(gate|router|router\.layer|gating|router\.gate)\.weight$"), k) and not re.search(F.get("expert_re") or "^$", k):
        return "unpacked", pre + "MoE routers", where
    if dt not in ("BF16",):
        if nd <= 1:
            pass
        else:
            return "unpacked", pre + f"not bf16 ({dt})", where
    if nd <= 1:
        if re.search(r"norm|ln_|layernorm|\.ln\d*\.", k, re.I) and leaf in ("weight", "bias"):
            return "unpacked", pre + "norms", where
        if leaf == "bias":
            return "unpacked", pre + "biases", where
        return "unpacked", pre + "small params (A_log, dt_bias, D, scales)", where
    if where != "decoder":
        for p in F.get("tower_plain", []):
            if re.search(p, k):
                return "unpacked", pre + ("conv / patch embed" if nd > 2 or "patch" in k or "conv" in k else "plain layer (nn.Linear / nn.Embedding)"), where
    # embedding tables
    if where == "decoder" and re.search(F.get("embed", r"(embed_tokens|word_embeddings|wte|tok_embeddings)\.weight$"), k):
        if not F.get("embed_qc"):
            return "unpacked", "embedding (built without the quant config)", where
        if sh[1] % 64 or sh[0] < 4096:
            return "unpacked", "embedding (shape)", where
        return "packed", "packed", where
    if where == "decoder" and F.get("ple") and re.search(F["ple"], k):
        if not F.get("ple_qc"):
            return "unpacked", "per-layer embedding (built without the quant config)", where
        if sh[1] % 64 or sh[0] < 4096:
            return "unpacked", "per-layer embedding (shape)", where
        return "packed", "packed", where
    if where == "decoder" and re.search(F.get("head", r"(^|\.)lm_head\.weight$"), k):
        if not F.get("head_qc"):
            return "unpacked", F.get("head_label", "LM head (built without the quant config)"), where
        if sh[1] % 64:
            return "unpacked", "LM head (shape)", where
        return "packed", "packed", where
    m = re.search(F["expert_re"], k) if F.get("expert_re") else EXPERT.search(k)
    if m and where != "vision" and where != "audio" and not re.search(r"shared_expert", k):
        return experts(k, sh, F, tc, m)
    if re.search(r"conv1d\.weight$|\.conv\.weight$", k) or nd > 2:
        return "unpacked", pre + "conv weights", where
    if re.search(F.get("router", r"\.(gate|router|router\.layer|gating|router\.gate)\.weight$"), k) and nd == 2:
        return "unpacked", pre + "MoE routers", where
    if F.get("noqc_fn"):
        lab = F["noqc_fn"](k)
        if lab:
            return "unpacked", pre + lab, where
    for p in F.get("noqc", []):
        p, lab = p if isinstance(p, tuple) else (p, "layer built without the quant config")
        if re.search(p, k):
            return "unpacked", pre + lab, where
    if where != "decoder" and not tower_qc:
        return "unpacked", pre + F.get("tower_label", "tower built without the quant config"), where
    # a LinearBase built with the quant config
    parent, lf = (k[: -len(".weight")].rsplit(".", 1) + [""])[:2] if k.endswith(".weight") else (k, "")
    g = MERGE.get(lf)
    for a, bb in F.get("merge", {}).items():
        if lf == a:
            g = bb
    O, K = groups[(parent, g)] if g and (parent, g) in groups else (sh[0], sh[1])
    if F.get("transposed") and re.search(F["transposed"], k):
        O, K = K, O
    if O * K < SMALL:
        return "unpacked", pre + "small Linears (< 1M weights)", where
    if O % 64 or K % 16:
        return "unpacked", pre + "Linear shape (rows % 64 or columns % 16)", where
    return "packed", "packed", where


def experts(k, sh, F, tc, m):
    act = F.get("act", "silu")
    if F.get("layer_act"):
        act = F["layer_act"](k, tc) or act
    if act != "silu":
        return "unpacked", f"MoE experts (activation {act}, not SiLU)", "decoder"
    if F.get("ebias"):
        return "unpacked", "MoE experts (biases)", "decoder"
    leaf = m.group(m.lastindex) if m.lastindex else ""
    a, b = sh[-2], sh[-1]  # per expert [out, in]
    if re.search(r"down|w2|fc2", leaf):
        H, I = a, b
    elif re.search(r"gate_up|w13|fc1", leaf) or (len(sh) == 3 and a == 2 * (tc.get("moe_intermediate_size") or -1)):
        H, I = b, a // 2
    else:
        H, I = b, a
    if H % 64 or (2 * I) % 128 or I % 16:
        return "unpacked", f"MoE experts (hidden {H} / intermediate {I} sizes)", "decoder"
    if F.get("clamp"):
        return "packed", "packed (clamped SwiGLU: the grouped products omit the clamp)", "decoder"
    return "packed", "packed", "decoder"


def summarize(rows, arch, rec):
    tot = sum(r[3] for r in rows)
    loaded = sum(r[3] for r in rows if r[4] != "not_loaded")
    packed = sum(r[3] for r in rows if r[4] == "packed")
    un = Counter()
    nl = Counter()
    tower_packed = Counter()
    warn = Counter()
    for k, dt, sh, b, st, cls, where in rows:
        if st == "unpacked":
            un[cls] += b
        elif st == "not_loaded":
            nl[cls] += b
        elif cls != "packed":
            warn[cls] += b
        if st == "packed" and where != "decoder":
            tower_packed[where] += b
    s = {"repo": rec["repo"], "arch": arch, "created": (rec.get("created") or "")[:10], "tensors": len(rows),
         "total_bytes": tot, "loaded_bytes": loaded, "packed_bytes": packed, "unpacked_bytes": loaded - packed,
         "unpacked": dict(un.most_common()), "not_loaded": dict(nl.most_common()), "tower_packed": dict(tower_packed), "packed_with_caveat": dict(warn)}
    for name, bits in BITS.items():
        s[f"projected_{name}_saving_pct"] = round(100 * packed * (1 - bits / 16) / loaded, 2) if loaded else 0
        s[f"projected_{name}_all_packed_pct"] = round(100 * (1 - bits / 16), 2)
    if F_of(rec).get("blocker"):
        s["blocker"] = F_of(rec)["blocker"]
    return s


def F_of(rec):
    return FAM.get(((rec.get("config") or {}).get("architectures") or ["?"])[0], {})


def _gemma4(c, tc):
    vc = c.get("vision_config") or {}
    q = bool(vc) and vc.get("hidden_size", 1) % 64 == 0 and vc.get("intermediate_size", 1) % 64 == 0
    L, shared = tc.get("num_hidden_layers", 0), tc.get("num_kv_shared_layers") or 0

    def drop_fn(k, tc):  # gemma4.py load_weights: a KV-shared layer has a plain q_proj; its checkpoint k/v are skipped
        m = re.search(r"language_model\.layers\.(\d+)\.self_attn\.(k_proj|v_proj)\.weight$", k)
        return "KV-shared layers' k/v (skipped by vLLM)" if m and shared and int(m.group(1)) >= L - shared else None
    t = [(r"(^|\.)(vision_tower|embed_vision)\.", "vision", q), (r"(^|\.)(audio_tower|embed_audio)\.", "audio", q)]
    return dict(towers=t, drop_fn=drop_fn, tower_label="tower not given the quant config by vLLM (vision intermediate %s not a multiple of 64)" % vc.get("intermediate_size"))


GEMMA4 = dict(fn=_gemma4, embed_qc=True, head_qc=True, ple=r"embed_tokens_per_layer\.weight$", ple_qc=True, act="gelu_tanh",
              tower_plain=[r"position_embedding_table", r"pos_embedding"], router=r"\.router\.proj\.weight$")
# gemma4_unified.py: no SigLIP / audio towers; vision_embedder.patch_dense and the embedders' projections built with the quant config
GEMMA4U = dict(GEMMA4, fn=lambda c, tc: dict(_gemma4(c, tc), towers=[(r"(^|\.)(vision_embedder|embed_vision)\.", "vision", True), (r"(^|\.)embed_audio\.", "audio", True)]))
FAM.update({"Gemma4ForConditionalGeneration": GEMMA4, "Gemma4UnifiedForConditionalGeneration": GEMMA4U})

FAM.update({
    # granite.py / granitemoeshared.py: embed_tokens and lm_head with the quant config; the router a ReplicatedLinear quant_config=None
    "GraniteForCausalLM": dict(embed_qc=True, head_qc=True),
    "GraniteSWAForCausalLM": dict(embed_qc=True, head_qc=True),
    "GraniteMoeSWAForCausalLM": dict(embed_qc=True, head_qc=True, router=r"\.router(\.layer)?\.weight$"),
    # muse_glimmer.py: embed_tokens without quant config; the vision encoder's QKV/o built without it, its MLP, adapter and
    # projection plain nn.Linear; lm_head with it
    "MuseGlimmerForConditionalGeneration": dict(embed_qc=False, head_qc=True,
        towers=[(r"(^|\.)(vision_tower|vision_adapter|vision_projection|perception_emb_norm)\.?", "vision", False)],
        tower_plain=[r"patch_embedder"]),
})

FAM.update({
    # granite4_vision.py: SigLIP tower with the quant config, but its fc1/fc2 only where 64 divides them (siglip.py: 4304 does
    # not); the Q-Former projectors with it, their out_linear a plain nn.Linear; the LM is GraniteModel (embedding with it)
    "Granite4VisionForConditionalGeneration": dict(embed_qc=True, head_qc=True,
        towers=[(r"(^|\.)(vision_tower|layerwise_projectors|spatial_projectors|image_newline)\.?", "vision", True)],
        fn=lambda c, tc: dict(noqc=([(r"vision_model\.encoder\.layers\.\d+\.mlp\.fc[12]\.weight$", MLP4304)] if (c.get("vision_config") or {}).get("intermediate_size", 0) % 64 else []) + [r"out_linear\.weight$"]),
        tower_plain=[r"embeddings\.(patch_embedding|position_embedding)", r"\.query$", r"image_positions"]),
})

FAM.update({
    # lfm2.py / lfm2_moe.py: embed_tokens without the quant config (tied head: lm_head ties to it, so it stays too); the short
    # conv's conv weight a conv; in_proj / out_proj with it; MoE experts SiLU (w1, w3, w2), the router with it but small
    "Lfm2ForCausalLM": dict(embed_qc=False, head_qc=True),
    "Lfm2MoeForCausalLM": dict(embed_qc=False, head_qc=True, router=r"feed_forward\.gate\.weight$"),
    # lfm2_vl.py: SigLIP2 (lfm2_siglip2.py) with the quant config (fc1 4304 rows fail rows % 64), its patch_embedding and the
    # projector's linear_1 / linear_2 ReplicatedLinear without it
    "Lfm2VlForConditionalGeneration": dict(embed_qc=False, head_qc=True,
        towers=[(r"(^|\.)(vision_tower|multi_modal_projector)\.", "vision", True)],
        noqc=[r"multi_modal_projector\.linear_[12]\.weight$", r"patch_embedding\.weight$"], tower_plain=[r"position_embedding"]),
})

FAM.update({
    # mellum.py on qwen3_moe.py: embed_tokens and lm_head with the quant config; the router with it but small; experts SiLU
    "MellumForCausalLM": dict(embed_qc=True, head_qc=True),
    # exaone4_5.py: Qwen2.5-VL-style tower with the quant config (qkv, proj, gated MLP, merger), mtp. mapped to None, the LM exaone4.py
    # (embed_tokens and lm_head with it)
    "Exaone4_5_ForConditionalGeneration": dict(embed_qc=True, head_qc=True, drop=[r"^mtp\."],
        towers=[(r"(^|\.)visual\.", "vision", True)], tower_plain=[r"patch_embed"]),
})


def _step_act(k, tc):
    m = re.search(r"layers\.(\d+)\.", k)
    lim = tc.get("swiglu_limits") or []
    return "swiglustep" if m and int(m.group(1)) < len(lim) and lim[int(m.group(1))] not in (None, 0) else None


FAM.update({
    # hy_v3.py: embed_tokens without the quant config, lm_head with it (bf16 head: HYV3Config sets no head_dtype); the router a
    # GateLinear (quant_config=None, fp32); the nextn layer (layers.80) skipped by load_weights
    "HYV3ForCausalLM": dict(embed_qc=False, head_qc=True, drop_extra_layers=True, router=r"\.router\.gate\.weight$|\.gate\.weight$"),
    # vllm/models/hy_v4: embed_tokens without it; lm_head with it but head_dtype float32 (enable_lm_head_fp32): the plugin keeps
    # it; MLA q_a + kv_a fused; experts SiLU with swiglu_limit 10 (FusedMoEConfig.swiglu_limit, activation still SILU);
    # model.mtp_layers and layers past num_hidden_layers dropped; hyper-connection fns fp32 (ReplicatedLinear)
    "HYV4ForCausalLM": dict(embed_qc=False, head_qc=False, head_label="LM head (head_dtype float32: kept as vLLM runs it)",
        drop=[r"(^|\.)mtp_layers\."], drop_extra_layers=True, merge={"q_a_proj": "qkv_a", "kv_a_proj_with_mqa": "qkv_a"},
        noqc=[r"wk_weights_proj", r"hc_"], clamp=True),
    # step3p5.py (Step-3.5/3.7 text): embed_tokens without it, lm_head with it; experts stored [E, ...] under .moe.; the
    # router quant_config=None; layers with a swiglu limit run activation swiglustep; nextn layers skipped
    # step3p7.py / step_vl.py: PerceptionEncoder and vit_large_projector with the quant config; conv downsamplers Conv2dLayer
    "Step3p5ForCausalLM": dict(embed_qc=False, head_qc=True, drop_extra_layers=True,
        expert_re=r"\.moe\.(gate_proj|up_proj|down_proj)\.weight$", router=r"\.moe\.gate\.weight$", layer_act=_step_act),
    # jamba.py: embed_tokens and lm_head without the quant config; mamba_mixer.py (Mamba v1) builds conv1d, in_proj, x_proj,
    # dt_proj and out_proj without it; experts SiLU; the router quant_config=None
    "JambaForCausalLM": dict(embed_qc=False, head_qc=False, router=r"\.router\.weight$",
        noqc=[(r"\.mamba\.(in_proj|x_proj|dt_proj|out_proj)\.weight$", "Mamba mixer Linears built without the quant config")]),
    # olmo_hybrid.py: embed_tokens without it, lm_head with it; the GDN q/k/v/g fused (in_proj_qkvg), a/b small, conv1d
    "OlmoHybridForCausalLM": dict(embed_qc=False, head_qc=True),
})
FAM["Step3p7ForConditionalGeneration"] = dict(FAM["Step3p5ForCausalLM"], towers=[(r"(^|\.)(vision_model|vit_large_projector)", "vision", True)],
    tower_plain=[r"vit_downsampler", r"conv1", r"positional_embedding", r"class_embedding"])

FAM.update({
    # nemotron_h.py: embeddings without the quant config, lm_head with it; mtp mapped to None; Mamba2 mixer (mamba_mixer2.py)
    # in_proj / out_proj with it, conv1d without; dense MLP up/down (relu2) with it; routed experts relu2 without a gate
    # (activation_without_mul: not SiLU); the router GateLinear; latent projections with it
    "NemotronHForCausalLM": dict(embed_qc=False, head_qc=True, drop=[r"^mtp\."], embed=r"backbone\.embeddings\.weight$",
        act="relu2 without a gate", router=r"\.mixer\.gate\.weight$"),
})

# nano_nemotron_vl.py: RADIO vision (RadioModel built without the quant config), mlp1 plain nn.Linear, Parakeet sound encoder
# without it; the LM nemotron_h.py
FAM["NemotronH_Nano_Omni_Reasoning_V3"] = dict(FAM["NemotronHForCausalLM"], drop=[r"(^|\.)mtp\."], embed=r"backbone\.embeddings\.weight$",
    towers=[(r"^(vision_model|mlp1)\.", "vision", False), (r"^(sound_encoder|sound_projection)\.", "audio", False)])

FAM.update({
    # cohere2_moe.py: embed_tokens without the quant config (tied: the head is that table); the router quant_config=None
    "Cohere2MoeForCausalLM": dict(embed_qc=False, head_qc=True),
    # step_vl.py: PerceptionEncoder + vit_large_projector with the quant config; the LM Qwen3 (qwen3.py: embedding with it)
    "StepVLForConditionalGeneration": dict(embed_qc=True, head_qc=True,
        towers=[(r"(^|\.)(vision_model|vit_large_projector)", "vision", True)],
        tower_plain=[r"vit_downsampler", r"conv1", r"positional_embedding", r"class_embedding"]),
})

# cohere_compass.py: Qwen3-VL-style tower with the quant config (qkv, proj, MLP, merger, deepstack mergers); embed_tokens
# without it (tied head: stays too)
FAM["CohereCompassForConditionalGeneration"] = dict(embed_qc=False, head_qc=True, towers=[(r"(^|\.)visual\.", "vision", True)],
    tower_plain=[r"visual\.pos_embed", r"patch_embed"])

FAM.update({
    # vllm/models/inkling: the embedding an InklingReplicatedEmbedding (a plain nn.Parameter, not a VocabParallelEmbedding); the
    # head (unembed) a ParallelLMHead with the quant config; q/k/v/r fused (qkvr) with it; routed experts SiLU through FusedMoE;
    # the shared "sink" experts plain bf16 parameters (InklingSinkExperts); vision / audio towers plain nn.Linear; model.mtp. dropped
    "InklingForConditionalGeneration": dict(embed_qc=False, head_qc=True, head=r"llm\.unembed\.weight$", drop=[r"^model\.mtp\."],
        expert_re=r"\.experts\.(w13|w2)_weight$", merge={"wq_du": "qkvr", "wk_dv": "qkvr", "wv_dv": "qkvr", "wr_du": "qkvr"},
        special=[(r"llm\.embed\.weight$", "embedding (a plain parameter, not a VocabParallelEmbedding)"),
                 (r"shared_experts\.shared_w(13|2)_weight$", "shared (sink) experts held as plain parameters")],
        towers=[(r"^model\.visual\.", "vision", False), (r"^model\.audio\.", "audio", False)],
        noqc=[r"rel_logits_proj"]),
    # llama.py: embed_tokens and lm_head with the quant config
    "LlamaForCausalLM": dict(embed_qc=True, head_qc=True),
    # apertus.py: embed_tokens and lm_head with it (xIELU activation params are small)
    "ApertusForCausalLM": dict(embed_qc=True, head_qc=True),
})

MLA = {"q_a_proj": "qkv_a", "kv_a_proj_with_mqa": "qkv_a"}  # fused_qkv_a_proj where q_lora_rank is set
FAM.update({
    # bailing_moe_v3.py: word_embeddings without the quant config, lm_head with it; KDA q/k/v (qkv_proj), b, f, g, o with it,
    # the q/k/v conv1d without; MLA layers q_a + kv_a fused; experts SiLU; the gate its own module (fp32 router)
    "BailingMoeV3ForCausalLM": dict(embed_qc=False, head_qc=True, merge=MLA, drop_extra_layers=True),
    # glm4_moe_lite.py (DeepSeek-V2 style MLA): embed_tokens and lm_head with the quant config; the nextn layer (layers.47:
    # its own embed_tokens, shared_head, eh_proj) skipped; the router a GateLinear
    "Glm4MoeLiteForCausalLM": dict(embed_qc=True, head_qc=True, merge=MLA, drop_extra_layers=True),
})

FAM.update({
    # vllm/models/deepseek_v32 (GLM-5.x, CUDA): make_input_embedding (a VocabParallelEmbedding with the quant config), lm_head with
    # it; MLA q_a + kv_a fused (DeepSeekV2FusedQkvAProjLinear); the indexer's wq_b with it, wk_weights_proj without; experts SiLU
    # (DeepseekV2MoE), the router a GateLinear; nextn layers loaded by the MTP model only (skipped here)
    "GlmMoeDsaForCausalLM": dict(embed_qc=True, head_qc=True, merge=MLA, drop_extra_layers=True, noqc=[r"wk_weights_proj", r"indexer\.(wk|weights_proj)\.weight$"]),
    # cohere2_vision.py: SigLIP with the quant config (fc1/fc2 only where 64 divides them), the projector (linear_1 / linear_2)
    # built without it; the LM from text_config.architectures (Cohere2Moe: embedding without it, tied head)
    "Cohere2VisionForConditionalGeneration": dict(embed_qc=False, head_qc=True,
        towers=[(r"(^|\.)(vision_tower|multi_modal_projector)\.", "vision", True)],
        fn=lambda c, tc: dict(noqc=([(r"vision_model\.encoder\.layers\.\d+\.mlp\.fc[12]\.weight$", MLP4304)] if (c.get("vision_config") or {}).get("intermediate_size", 0) % 64 else []) + [r"multi_modal_projector\.linear_[12]\.weight$"]),
        tower_plain=[r"embeddings\.(patch_embedding|position_embedding)"]),
})


def _glm5next_prep(tensors, c, tc):
    mla = {m.group(1) for k in tensors for m in [re.search(r"layers\.(\d+)\.self_attn\.kv_b_proj", k)] if m}

    def noqc_fn(k):  # glm5next: the MLA layers' attention is built with quant_config=None ("MLA projections are BF16 in checkpoint")
        m = re.search(r"layers\.(\d+)\.self_attn\.", k)
        return "MLA attention built with quant_config=None" if m and m.group(1) in mla and k.endswith("weight") else None
    return dict(noqc_fn=noqc_fn)


KDA = {"q_proj": "kda", "k_proj": "kda", "v_proj": "kda", "b_proj": "kda", "f_a_proj": "kda", "g_a_proj": "kda"}
# vllm/models/glm5next: embed_tokens without the quant config, lm_head with it; KDA in_proj_qkvbfg_a (q, k, v, b, f_a, g_a fused),
# f_b, g_b, o with it, the conv1d without; MLA layers' attention quant_config=None; experts SiLU + swiglu_limit (clamp); the
# vision tower (multimodal.py) with it; hc_*_fn plain parameters; the nextn layer skipped
FAM["Glm5NextForConditionalGeneration"] = dict(embed_qc=False, head_qc=True, merge=KDA, drop_extra_layers=True, clamp=True,
    prep=_glm5next_prep, special=[(r"hc_(attn|ffn)_fn$", "hyper-connection parameters (plain)")], towers=[(r"(^|\.)visual\.", "vision", True)], tower_plain=[r"visual\.downsample", r"patch_embed", r"pos_embed"])

# bailing_moe_v3_vl.py: Qwen3-VL-style tower and the linear_proj projector with the quant config (unless FP8 ignores them); the
# LM bailing_moe_v3.py
FAM["BailingMoeV3VLForConditionalGeneration"] = dict(FAM["BailingMoeV3ForCausalLM"],
    towers=[(r"(^|\.)(visual|linear_proj)\.", "vision", True)], tower_plain=[r"visual\.pos_embed", r"patch_embed"])

# exaone_moe.py: embed_tokens and lm_head with the quant config; mtp. mapped to None; experts SiLU; the router quant_config=None
FAM["ExaoneMoeForCausalLM"] = dict(embed_qc=True, head_qc=True, drop=[r"^mtp\."])

# vllm/models/minimax_m3: embed_tokens and lm_head with the quant config; experts activation swigluoai_uninterleave (not SiLU);
# the router a GateLinear (fp32); "mtp." skipped by load_weights; the vision tower (common/vision_tower.py) with it, its
# patch embedding an nn.Conv3d
FAM["MiniMaxM3SparseForConditionalGeneration"] = dict(embed_qc=True, head_qc=True, drop=[r"(^|\.)mtp\."], act="swigluoai_uninterleave",
    towers=[(r"(^|\.)(vision_tower|multi_modal_projector|patch_merge_mlp)\.", "vision", True)], tower_plain=[r"patch_embedding", r"pos_emb"])

# nemotron_h.py serves the Puzzle variant too (relu2 experts without a gate)
FAM["NemotronHPuzzleForCausalLM"] = FAM["NemotronHForCausalLM"]
# diffusion_gemma.py: Gemma4Model (embedding with the quant config, experts GELU-tanh) and gemma4_mm's tower rule
FAM["DiffusionGemmaForBlockDiffusion"] = GEMMA4
# longcat_flash_ngram.py: the n-gram embedder (oe_embedder) a VocabParallelEmbedding without the quant config; embed_tokens
# without it too; lm_head with it; model.mtp. mapped to None; experts SiLU
FAM["LongcatFlashNgramForCausalLM"] = dict(embed_qc=False, head_qc=True, drop=[r"^model\.mtp\."],
    special=[(r"ngram_embeddings\.embedders\.\d+\.weight$", "n-gram embedding table (built without the quant config)"),
             (r"ngram_embeddings\.post_projs\.", "n-gram projections (a plain stacked parameter)")])


if __name__ == "__main__":
    files = sys.argv[1:] or sorted(glob.glob(os.path.join(HERE, "headers", "*.json")))
    for p in files:
        rec = json.load(open(p))
        if "shards" not in rec or not rec["shards"]:
            print(rec["repo"], "no shards", rec.get("err"))
            continue
        rows, arch = classify(rec)
        if rows is None:
            print(rec["repo"], "NO FAMILY", arch)
            continue
        s = summarize(rows, arch, rec)
        if s.get("blocker"):
            print("   BLOCKER:", s["blocker"])
        print(f"{s['repo']:50s} tot {s['total_bytes']/2**30:8.2f} GiB loaded {s['loaded_bytes']/2**30:8.2f} packed {s['packed_bytes']/2**30:8.2f}"
              f"  tiered {s['projected_tiered_saving_pct']:5.2f}%  12b {s['projected_12bit_saving_pct']:5.2f}%")
        for c, b in s["unpacked"].items():
            print(f"      {b/2**30:8.3f} GiB  {c}")
        for c, b in s["packed_with_caveat"].items():
            print(f"      {b/2**30:8.3f} GiB  (PACKED, caveat) {c}")
        for c, b in s["not_loaded"].items():
            print(f"      {b/2**30:8.3f} GiB  (not loaded) {c}")

