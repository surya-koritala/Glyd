"""Every tensor of the five 2026 FP8 checkpoints: its kind, its dtype, its bytes, the vLLM 0.30 module that owns it, the quant method
vLLM gives it, and Glyd's planned status: packed (FP8 Linear wrapper | FP8 MoE, Task 4 | BF16 path) or kept as vLLM's, with the reason.
The release gate "whole models, every tensor packed or kept with a stated reason" is read from here.

    python inventory.py [--cache DIR] [--md INVENTORY.md] [REPO...]    # no REPO: all five; headers only, read by HTTP range (public repos, no token)
    python inventory.py --selftest

A kind is a tensor's name with its layer and expert numbers taken out (a multi-token-prediction head's tensors are a kind of their own).
The rules (RULES below) were written from the vLLM 0.30.0 model files each checkpoint loads through (the sources in the notes of each
model): first rule that matches "kind|dtype" wins. A kind no rule matches stops the run (no silent skips); the rows add up to the
checkpoint's bytes (checked). Expected saving: SAVE_FP8 of the packed FP8 bytes and SAVE_BF16 of the packed BF16 bytes (the measured
averages, gpu/fp8/README.md and the BF16 path's results); scales, norms, FP4 and everything kept save nothing.
"""
import argparse, collections, concurrent.futures as cf, json, math, os, re, struct, sys, time, urllib.error, urllib.request

HF = "https://huggingface.co"
SAVE_FP8, SAVE_BF16 = 0.164, 0.327
SMALL = 1 << 20  # weights: a Linear under this stays vLLM's in the wrapper (SMALL_LINEAR of vllm_plugin.py, vllm_fp8.py)


# ---- the checkpoint's headers (get and tensors as fmt_measure.py / fetch_samples.py; stdlib only, so this runs without numpy) ----

def get(url, rng=None, tries=8):
    """The bytes of url (rng: an inclusive byte range), retried."""
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={rng[0]}-{rng[1]}"} if rng else {})
            with urllib.request.urlopen(req, timeout=120) as r:
                data = r.read()
            if rng and len(data) != rng[1] - rng[0] + 1:
                raise IOError(f"short read {len(data)} of {rng[1] - rng[0] + 1}")
            return data
        except urllib.error.HTTPError as e:
            if e.code in (401, 403, 404):
                raise
            time.sleep(min(30, 2 ** i))
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(min(30, 2 ** i))


def tensors(repo, index="model.safetensors.index.json"):
    """Every tensor of the checkpoint: name -> (dtype, shape, shard, absolute start, end)."""
    try:
        shards = sorted(set(json.loads(get(f"{HF}/{repo}/resolve/main/{index}"))["weight_map"].values()))
    except urllib.error.HTTPError:
        shards = ["model.safetensors"]

    def head(fn):
        n = struct.unpack("<Q", get(f"{HF}/{repo}/resolve/main/{fn}", (0, 7)))[0]
        return fn, n, json.loads(get(f"{HF}/{repo}/resolve/main/{fn}", (8, 8 + n - 1)))

    out = {}
    with cf.ThreadPoolExecutor(12) as ex:
        for fn, n, hdr in ex.map(head, shards):
            for k, v in hdr.items():
                if k != "__metadata__":
                    a, b = v["data_offsets"]
                    out[k] = (v["dtype"], v["shape"], fn, 8 + n + a, 8 + n + b)
    return out


def headers(repo, index, cache):
    f = os.path.join(cache, repo.replace("/", "__") + "." + index.split(".")[0] + ".json") if cache else None
    if f and os.path.exists(f):
        return json.load(open(f))
    ts = tensors(repo, index)
    if f:
        os.makedirs(cache, exist_ok=True)
        json.dump(ts, open(f, "w"))
    return ts


# ---- kinds ----

SCALE = re.compile(r"(\.weight_scale_inv|\.weight_scale|\.activation_scale|\.input_scale|\.qscale_weight|\.qscale_act|\.scale|_scale_inv|_activation_scale)$")
MTP_TAG = "[mtp] "


def kind(name, mtp=False):
    """The name with its layer and expert numbers taken out (a number between dots, or at the end)."""
    k = re.sub(r"\.\d+(?=\.|$)", ".N", name)
    return MTP_TAG + k if mtp else k


# ---- statuses ----

P8L, P8M, PBF, KEEP = "P8L", "P8M", "PBF", "KEEP"
STATUS = {P8L: "packed (FP8 Linear wrapper)", P8M: "packed (FP8 MoE, Task 4)", PBF: "packed (BF16 path)"}
TODAY = {P8L: "yes (Task 3)", P8M: "no: Task 4", PBF: "no: not wired (the FP8 config hands them back untouched)", KEEP: "-"}

# reasons a tensor is kept as vLLM's (the text of the status; one string a reason, so the totals group by it)
R_SCALE = "scales untouched by design"
R_NORM = "norm, bias or small vector"
R_PARAM = "small nn.Parameter, not a Linear"
R_CONV = "convolution or lookup table, not a Linear"
R_MTP = "MTP head: vLLM's main model skips it (loaded only for speculative decoding)"
R_NOHOOK = "built with quant_config=None: no quant-method hook"
R_FP4 = "FP4 experts: out of scope (3-6% room, no kernel)"
R_M10 = "kv_b_proj stays vLLM's (review M10: vLLM reads its weight after load)"
R_BMM = "bmm layer (grouped o-proj, is_bmm): vLLM reads its weight as a batch of matrices; the wrapper skips bmm layers"
R_ENGRAM = "engram table: ParallelEngramEmbedding (rows gathered by hash id, in host memory through UVA), not a Linear, no quant method"
R_DEQ = "the model dequantizes it to BF16 at load and builds its Linear with quant_config=None: vLLM holds it in BF16 with no hook"
R_VISION = "vision tower built without a quant_config: no hook"
R_SMALL = "Linear under the 1M-weight threshold"
R_ROUTER = "router gate: GateLinear is built with quant_config=None"

F8 = r"\|F8_E4M3$"
BF = r"\|BF16$"
ANY = r"\|"


def rules(*rows):
    return [(re.compile(p), mod, qm, st, note) for p, mod, qm, st, note in rows]


# ---- Qwen/Qwen3.8-27B-FP8: qwen3_5.py (Qwen3_5ForConditionalGeneration), qwen3_next.py, mamba/gdn/qwen_gdn_linear_attn.py, qwen3_vl.py ----
Q_FP8 = "Fp8LinearMethod, block 128x128 (Ada: Marlin W8A16; Hopper: FlashInfer or DeepGEMM)"
Q_UNQ = "UnquantizedLinearMethod"
QWEN = rules(
    (r"^\[mtp\] ", "mtp.* (Qwen3_5 MTP class)", "-", KEEP, R_MTP),
    (r"layers\.N\.mlp\.(gate|up)_proj\.weight" + F8, "MergedColumnParallelLinear (gate_up_proj)", Q_FP8, P8L, ""),
    (r"layers\.N\.mlp\.down_proj\.weight" + F8, "RowParallelLinear (down_proj)", Q_FP8, P8L, ""),
    (r"linear_attn\.in_proj_(qkv|z)\.weight" + F8, "MergedColumnParallelLinear (in_proj_qkvz)", Q_FP8, P8L, "gated delta net"),
    (r"linear_attn\.out_proj\.weight" + F8, "RowParallelLinear (out_proj)", Q_FP8, P8L, "gated delta net"),
    (r"self_attn\.q_proj\.weight" + F8, "QKVParallelLinear (qkv_proj)", Q_FP8, P8L, "q_proj holds the output gate too"),
    (r"self_attn\.(k|v)_proj\.weight" + F8, "QKVParallelLinear (qkv_proj)", Q_FP8, P8L, ""),
    (r"self_attn\.o_proj\.weight" + F8, "RowParallelLinear (o_proj)", Q_FP8, P8L, ""),
    (r"^lm_head\.weight" + BF, "ParallelLMHead (quant_config passed)", "UnquantizedEmbeddingMethod (Fp8Config returns None for it)", PBF, ""),
    (r"^model\.language_model\.embed_tokens\.weight" + BF, "VocabParallelEmbedding (qwen3_5.py builds it without quant_config)", "UnquantizedEmbeddingMethod", KEEP, R_NOHOOK),
    (r"linear_attn\.in_proj_(a|b)\.weight" + BF, "MergedColumnParallelLinear (in_proj_ba)", Q_UNQ + " (in modules_to_not_convert)", KEEP, R_SMALL + " (0.25M weights each, 0.5M merged)"),
    (r"linear_attn\.conv1d\.weight", "ColumnParallelLinear (conv1d, no quant_config)", Q_UNQ, KEEP, R_CONV),
    (r"linear_attn\.(A_log|dt_bias)", "nn.Parameter", "-", KEEP, R_PARAM),
    (r"model\.visual\.blocks\.N\.mlp\.linear_fc1\.weight" + BF, "ColumnParallelLinear (vision MLP; quant_config passed)", Q_UNQ + " (model.visual in modules_to_not_convert)", PBF, "vision tower; not loaded with --language-model-only"),
    (r"model\.visual\.blocks\.N\.mlp\.linear_fc2\.weight" + BF, "RowParallelLinear (vision MLP; quant_config passed)", Q_UNQ + " (model.visual in modules_to_not_convert)", PBF, "vision tower"),
    (r"model\.visual\.blocks\.N\.attn\.qkv\.weight" + BF, "QKVParallelLinear (vision attention; quant_config passed)", Q_UNQ + " (model.visual in modules_to_not_convert)", PBF, "vision tower"),
    (r"model\.visual\.blocks\.N\.attn\.proj\.weight" + BF, "RowParallelLinear (vision attention; quant_config passed)", Q_UNQ + " (model.visual in modules_to_not_convert)", PBF, "vision tower"),
    (r"model\.visual\.merger\.linear_fc[12]\.weight" + BF, "ColumnParallelLinear / RowParallelLinear (vision merger; quant_config passed)", Q_UNQ + " (model.visual in modules_to_not_convert)", PBF, "vision tower"),
    (r"model\.visual\.pos_embed\.weight", "nn.Embedding (vision position table, 2304 rows)", "-", KEEP, R_CONV),
    (r"model\.visual\.patch_embed\.proj\.weight", "Conv3dLayer (patch embedding)", "-", KEEP, R_CONV),
    (r"(norm|bias)", "RMSNorm / LayerNorm weight or Linear bias", "-", KEEP, R_NORM),
)

# ---- zai-org/GLM-5.3-Flash: vllm/models/glm5next/nvidia/{model,attention,kda}.py (Glm5NextForConditionalGeneration), mtp.py ----
G_FP8 = "Fp8LinearMethod, block 128x128 (Ada: Marlin W8A16; Hopper: FlashInfer or DeepGEMM)"
G_MOE = "Fp8MoEMethod, block 128x128 (Triton; FlashInfer-CUTLASS with expert parallel on Hopper)"
G_UNQ = "UnquantizedLinearMethod (quant_config=None)"
GLM = rules(
    (r"^\[mtp\] ", "layer 45 (Glm5NextMTP class; the main model skips layers >= num_hidden_layers)", "-", KEEP, R_MTP),
    (r"mlp\.experts\.N\.(gate|up)_proj\.weight" + F8, "RoutedExperts (w13_weight, gate+up of every expert)", G_MOE, P8M, ""),
    (r"mlp\.experts\.N\.down_proj\.weight" + F8, "RoutedExperts (w2_weight)", G_MOE, P8M, ""),
    (r"mlp\.shared_experts\.(gate|up)_proj\.weight" + F8, "Glm5NextMLP: MergedColumnParallelLinear (gate_up_proj)", G_FP8, P8L, ""),
    (r"mlp\.shared_experts\.down_proj\.weight" + F8, "Glm5NextMLP: RowParallelLinear (down_proj)", G_FP8, P8L, ""),
    (r"mlp\.(gate|up)_proj\.weight" + F8, "Glm5NextMLP: MergedColumnParallelLinear (gate_up_proj); the 3 dense layers", G_FP8, P8L, "dense layers 0-2"),
    (r"mlp\.down_proj\.weight" + F8, "Glm5NextMLP: RowParallelLinear (down_proj); the 3 dense layers", G_FP8, P8L, "dense layers 0-2"),
    (r"self_attn\.o_proj\.weight" + F8, "MLA: RowParallelLinear (o_proj)", G_UNQ + "; the loader's _try_load_fp8_attn_proj turns FP8 + scale into BF16", KEEP, R_DEQ),
    (r"self_attn\.q_b_proj\.weight" + F8, "MLA: ColumnParallelLinear (q_b_proj)", G_UNQ + "; the loader's _try_load_fp8_attn_proj turns FP8 + scale into BF16", KEEP, R_DEQ),
    (r"self_attn\.(q_a_proj|kv_a_proj_with_mqa)\.weight" + F8, "MLA: DeepSeekV2FusedQkvAProjLinear (fused_qkv_a_proj: q_a, kv_a)", G_UNQ + "; the loader's _try_load_fp8_attn_proj turns FP8 + scale into BF16", KEEP, R_DEQ),
    (r"self_attn\.kv_b_proj\.weight" + BF, "MLA: ColumnParallelLinear (kv_b_proj)", G_UNQ, KEEP, R_NOHOOK + "; BF16 in the checkpoint; M10 anyway"),
    (r"self_attn\.indexer\.wq_b\.weight" + BF, "MLA indexer: ReplicatedLinear (wq_b)", G_UNQ, KEEP, R_NOHOOK),
    (r"self_attn\.indexer\.(wk|weights_proj)\.weight" + BF, "MLA indexer: MergedColumnParallelLinear (wk_weights_proj)", G_UNQ, KEEP, R_NOHOOK),
    (r"self_attn\.indexer\.index_kpool_compress_(gate|ape)", "nn.Parameter (indexer k-pool compression)", "-", KEEP, R_PARAM),
    (r"self_attn\.(q|k|v|b|f_a|g_a)_proj\.weight" + BF, "KDA: MergedColumnParallelLinear (in_proj_qkvbfg_a: q, k, v, b, f_a, g_a)", G_UNQ + " (Glm5NextLinearAttention sets vllm_config.quant_config=None)", KEEP, R_NOHOOK + " (KDA layers)"),
    (r"self_attn\.o_proj\.weight" + BF, "KDA: RowParallelLinear (o_proj)", G_UNQ, KEEP, R_NOHOOK + " (KDA layers)"),
    (r"self_attn\.(f_b|g_b)_proj\.weight" + BF, "KDA: ColumnParallelLinear (f_b_proj, g_b_proj)", G_UNQ, KEEP, R_NOHOOK + " (KDA layers)"),
    (r"self_attn\.(q|k|v)_conv1d\.weight", "KDA: ColumnParallelLinear (conv1d; vLLM holds it as F32)", "-", KEEP, R_CONV),
    (r"self_attn\.(dt_bias|A_log)", "KDA: nn.Parameter", "-", KEEP, R_PARAM),
    (r"mlp\.gate\.weight" + BF, "GateLinear (router; quant_config=None)", "UnquantizedLinearMethod", KEEP, R_ROUTER),
    (r"mlp\.gate\.e_score_correction_bias", "nn.Parameter (router bias)", "-", KEEP, R_PARAM),
    (r"layers\.N\.hc_(attn|ffn)_(fn|base|scale)", "nn.Parameter (hyper-connection mixing)", "-", KEEP, R_PARAM),
    (r"^lm_head\.weight" + BF, "ParallelLMHead (quant_config passed)", "UnquantizedEmbeddingMethod (Fp8Config returns None for it)", PBF, ""),
    (r"^model\.language_model\.embed_tokens\.weight" + BF, "VocabParallelEmbedding (model.py builds it without quant_config)", "UnquantizedEmbeddingMethod", KEEP, R_NOHOOK),
    (r"model\.visual\.(patch_embed\.proj|downsample)\.weight", "Glm5NextVisionTransformer (convolution)", "-", KEEP, R_CONV),
    (r"(norm|bias|layernorm)", "RMSNorm / LayerNorm weight or Linear bias", "-", KEEP, R_NORM),
    (r"model\.visual\.[^|]*\.weight" + BF, "Glm5NextVisionTransformer (built with quant_config=None, explicit)", G_UNQ, KEEP, R_VISION + "; not loaded with --language-model-only"),
)

# ---- MiniMaxAI/MiniMax-M3-MXFP8: vllm/models/minimax_m3/nvidia/model.py (MiniMaxM3SparseForConditionalGeneration), common/vision_tower.py ----
M_FP8 = "ModelOptLinearMethod, MXFP8 1x32 (Marlin W8A16 on Ada and Hopper)"
M_MOE = "ModelOptMxFp8FusedMoE (Marlin W8A16 on Ada and Hopper)"
M_UNQ = "UnquantizedLinearMethod (excluded in ignored_layers)"
MINIMAX = rules(
    (r"block_sparse_moe\.experts\.N\.(w1|w3)\.weight" + F8, "RoutedExperts (w13_weight)", M_MOE, P8M, ""),
    (r"block_sparse_moe\.experts\.N\.w2\.weight" + F8, "RoutedExperts (w2_weight)", M_MOE, P8M, ""),
    (r"block_sparse_moe\.shared_experts\.(gate|up)_proj\.weight" + F8, "MiniMaxM3MLP: MergedColumnParallelLinear (gate_up_proj)", M_FP8, P8L, ""),
    (r"block_sparse_moe\.shared_experts\.down_proj\.weight" + F8, "MiniMaxM3MLP: RowParallelLinear (down_proj)", M_FP8, P8L, ""),
    (r"layers\.N\.mlp\.(gate|up)_proj\.weight" + F8, "MiniMaxM3MLP: MergedColumnParallelLinear (gate_up_proj); the 3 dense layers", M_FP8, P8L, "dense layers 0-2"),
    (r"layers\.N\.mlp\.down_proj\.weight" + F8, "MiniMaxM3MLP: RowParallelLinear (down_proj); the 3 dense layers", M_FP8, P8L, "dense layers 0-2"),
    (r"self_attn\.(q|k|v|index_q)_proj\.weight" + F8, "QKVParallelLinear (qkv_proj; the 57 sparse layers' MinimaxM3QKVParallelLinearWithIndexer also holds index_q, index_k)", M_FP8, P8L, "a custom QKVParallelLinear subclass: untested"),
    (r"self_attn\.index_k_proj\.weight" + F8, "MinimaxM3QKVParallelLinearWithIndexer (qkv_proj)", M_FP8, P8L, "128x6144 = 0.8M weights alone, merged into qkv_proj (a custom subclass: untested)"),
    (r"self_attn\.o_proj\.weight" + F8, "RowParallelLinear (o_proj)", M_FP8, P8L, ""),
    (r"block_sparse_moe\.gate\.weight", "GateLinear (router; F32; quant_config=None)", "UnquantizedLinearMethod", KEEP, R_ROUTER),
    (r"block_sparse_moe\.e_score_correction_bias", "nn.Parameter (router bias)", "-", KEEP, R_PARAM),
    (r"language_model\.lm_head\.weight" + BF, "ParallelLMHead (quant_config passed)", "UnquantizedLinearMethod (lm_head in ignored_layers)", PBF, ""),
    (r"language_model\.model\.embed_tokens\.weight" + BF, "VocabParallelEmbedding (quant_config passed)", "UnquantizedEmbeddingMethod", PBF, ""),
    (r"vision_tower\.vision_model\.encoder\.layers\.N\.self_attn\.(q|k|v)_proj\.weight" + BF, "QKVParallelLinear (vision attention; quant_config passed)", M_UNQ, PBF, "vision tower; not loaded with --language-model-only"),
    (r"vision_tower\.vision_model\.encoder\.layers\.N\.self_attn\.out_proj\.weight" + BF, "RowParallelLinear (vision attention; quant_config passed)", M_UNQ, PBF, "vision tower"),
    (r"vision_tower\.vision_model\.encoder\.layers\.N\.mlp\.fc[12]\.weight" + BF, "ColumnParallelLinear / RowParallelLinear (vision MLP; quant_config passed)", M_UNQ, PBF, "vision tower"),
    (r"^(patch_merge_mlp|multi_modal_projector)\.linear_[12]\.weight" + BF, "ColumnParallelLinear / RowParallelLinear (vision projector; quant_config passed)", M_UNQ, PBF, "vision tower"),
    (r"patch_embedding\.weight", "nn.Conv3d (patch embedding, F32)", "-", KEEP, R_CONV),
    (r"(norm|bias|layrnorm)", "GemmaRMSNorm / LayerNorm weight or Linear bias", "-", KEEP, R_NORM),
)

# ---- mistralai/Mistral-Small-4-119B-2603: the consolidated shards (what vLLM loads by default: params.json -> mistral config format,
# MistralLarge3ForCausalLM = DeepseekV3ForCausalLM in model_executor/models/deepseek_v2.py; vision: pixtral.py PixtralForConditionalGeneration) ----
S_FP8 = "Fp8LinearMethod, per-tensor, static activation (Cutlass on Ada and Hopper; fused shards requantized to the max scale at load)"
S_MOE = "Fp8MoEMethod, per-tensor, static activation (Triton on Ada; FlashInfer-CUTLASS or Triton on Hopper)"
MISTRAL = rules(
    (r"layers\.N\.experts\.N\.(w1|w3)\.weight" + F8, "RoutedExperts (w13_weight)", S_MOE, P8M, ""),
    (r"layers\.N\.experts\.N\.w2\.weight" + F8, "RoutedExperts (w2_weight)", S_MOE, P8M, ""),
    (r"layers\.N\.shared_experts\.(w1|w3)\.weight" + F8, "DeepseekV2MLP: MergedColumnParallelLinear (gate_up_proj)", S_FP8, P8L, ""),
    (r"layers\.N\.shared_experts\.w2\.weight" + F8, "DeepseekV2MLP: RowParallelLinear (down_proj)", S_FP8, P8L, ""),
    (r"attention\.(wq_a|wkv_a_with_mqa)\.weight" + F8, "DeepSeekV2FusedQkvAProjLinear (fused_qkv_a_proj: q_a, kv_a)", S_FP8, P8L, ""),
    (r"attention\.wq_b\.weight" + F8, "ColumnParallelLinear (q_b_proj)", S_FP8, P8L, ""),
    (r"attention\.wo\.weight" + F8, "RowParallelLinear (o_proj)", S_FP8, P8L, ""),
    (r"attention\.wkv_b\.weight" + F8, "ColumnParallelLinear (kv_b_proj)", S_FP8, KEEP, R_M10),
    (r"layers\.N\.gate\.weight" + BF, "GateLinear (router; quant_config=None)", "UnquantizedLinearMethod", KEEP, R_ROUTER),
    (r"^output\.weight" + BF, "ParallelLMHead (quant_config passed)", "UnquantizedEmbeddingMethod (Fp8Config returns None for it)", PBF, ""),
    (r"^tok_embeddings\.weight" + BF, "VocabParallelEmbedding (quant_config passed)", "UnquantizedEmbeddingMethod", PBF, ""),
    (r"^vision_encoder\.transformer\.layers\.N\.", "Pixtral VisionTransformer (built without quant_config)", "UnquantizedLinearMethod", KEEP, R_VISION),
    (r"^(vision_language_adapter|patch_merger)\.", "ReplicatedLinear (Pixtral adapter, built without quant_config)", "UnquantizedLinearMethod", KEEP, R_VISION),
    (r"patch_conv\.weight", "Conv2d (patch embedding)", "-", KEEP, R_CONV),
    (r"(norm|ln_pre)", "RMSNorm weight", "-", KEEP, R_NORM),
)

# ---- deepseek-ai/DeepSeek-V4.1-Flash: vllm/models/deepseek_v41/{quant_config,attention,nvidia/model,nvidia/engram,common/engram,nvidia/vl_model}.py ----
D_FP8 = "ModelOptLinearMethod, MXFP8 (e8m0 scales per 32x32; Marlin W8A16 on Ada and Hopper)"
DEEPSEEK = rules(
    (r"^\[mtp\] ", "mtp.* (DSpark/MTP draft classes; the main model maps mtp. to None)", "-", KEEP, R_MTP),
    (r"engram\.embed\.weight" + F8, "ParallelEngramEmbedding (engram.embed_tokens), FP8 rows [384M x 256], 2 layers", "none (an nn.Module of its own; scales are e8m0 per 32)", KEEP, R_ENGRAM),
    (r"engram\.wkv\.weight" + F8, "ReplicatedLinear (engram.wkv)", D_FP8, P8L, ""),
    (r"engram\.(q|k)_weight", "nn.Parameter (engram)", "-", KEEP, R_PARAM),
    (r"attn\.(wq_a|wkv)\.weight" + F8, "MergedColumnParallelLinear (fused_wqa_wkv)", D_FP8, P8L, ""),
    (r"attn\.wq_b\.weight" + F8, "ColumnParallelLinear (wq_b)", D_FP8, P8L, ""),
    (r"attn\.wo_a\.weight" + F8, "ColumnParallelLinear (wo_a, is_bmm)", "ModelOptLinearMethod, MXFP8 + is_bmm: DeepGemmMxfp8BmmLinearKernel on Blackwell only, else EmulationMxfp8LinearKernel (BF16 at load)", KEEP, R_BMM),
    (r"attn\.wo_b\.weight" + F8, "RowParallelLinear (wo_b)", D_FP8, P8L, ""),
    (r"attn\.indexer\.wq_b\.weight" + F8, "ReplicatedLinear (indexer.wq_b)", D_FP8, P8L, "8 layers"),
    (r"attn\.indexer\.(wk|weights_proj)\.weight" + BF, "ReplicatedLinear (indexer; quant_config=None)", "UnquantizedLinearMethod", KEEP, R_NOHOOK),
    (r"attn\.compressor\.(wkv|wgate)\.weight" + BF, "MergedColumnParallelLinear (compressor.fused_wkv_wgate; quant_config=None)", "UnquantizedLinearMethod", KEEP, R_NOHOOK),
    (r"attn\.attn_sink", "nn.Parameter", "-", KEEP, R_PARAM),
    (r"layers\.N\.hc_(attn|ffn)_(fn|base|scale)", "nn.Parameter (hyper-connection mixing)", "-", KEEP, R_PARAM),
    (r"ffn\.gate\.weight" + BF, "GateLinear (router; quant_config=None)", "UnquantizedLinearMethod", KEEP, R_ROUTER),
    (r"ffn\.gate\.(bias|bias_vl)", "nn.Parameter (router bias)", "-", KEEP, R_PARAM),
    (r"ffn\.experts\.N\.w[123]\.weight", "RoutedExperts (FP4 e2m1, two to a byte)", "Mxfp4MoEMethod", KEEP, R_FP4),
    (r"ffn\.shared_experts\.(w1|w3)\.weight" + F8, "DeepseekV4MLP: MergedColumnParallelLinear (gate_up_proj)", D_FP8, P8L, ""),
    (r"ffn\.shared_experts\.w2\.weight" + F8, "DeepseekV4MLP: RowParallelLinear (down_proj)", D_FP8, P8L, ""),
    (r"^embed\.weight" + BF, "VocabParallelEmbedding (embed_tokens; quant_config passed)", "UnquantizedEmbeddingMethod", PBF, ""),
    (r"^head\.weight" + BF, "ParallelLMHead (deepseek_v41 builds it without quant_config)", "UnquantizedEmbeddingMethod", KEEP, R_NOHOOK),
    (r"^(vision|aligner)\.[^|]*\.weight" + BF, "DeepseekV4ViT / DeepseekV4Aligner (built without quant_config; the text backbone drops them)", "-", KEEP, R_VISION + "; stubbed out with an image limit of 0"),
    (r"^(vision|aligner|image_)", "vision tower / aligner / image tokens", "-", KEEP, R_NORM + " / " + R_VISION),
    (r"(norm)", "RMSNorm weight", "-", KEEP, R_NORM),
)

MODELS = {
    "zai-org/GLM-5.3-Flash": dict(
        index="model.safetensors.index.json", rules=GLM,
        mtp=lambda n: n.startswith("model.language_model.layers.") and int(n.split(".")[3]) >= 45,
        notes=["Glm5NextForConditionalGeneration, block FP8 128x128 (fp32 scales), 45 layers (34 KDA, 11 MLA; layers 0-2 dense, 3-44 MoE of 288 experts) + 1 MTP layer 45.",
               "vLLM's GLM model dequantizes the 4 FP8 MLA projections to BF16 at load and builds the MLA and KDA attention without a quant_config: neither Glyd path can reach them.",
               "Task 3 ran this model's 3 dense layers on Ada (exact); the experts need Task 4."]),
    "Qwen/Qwen3.8-27B-FP8": dict(
        index="model.safetensors.index.json", rules=QWEN, mtp=lambda n: n.startswith("mtp."),
        notes=["Qwen3_5ForConditionalGeneration, block FP8 128x128 (BF16 scale tensors in the file), 64 layers (48 gated delta net, 16 attention), dense MLPs.",
               "Task 3 ran 4 of its layers on Ada (exact, eager and graph); Hopper and the whole model are not run."]),
    "MiniMaxAI/MiniMax-M3-MXFP8": dict(
        index="model.safetensors.index.json", rules=MINIMAX, mtp=lambda n: False,
        notes=["MiniMaxM3SparseForConditionalGeneration, MXFP8 (e8m0 scale per 1x32), 60 layers (3 dense, 57 MoE of 128 experts; 57 sparse-attention layers with the lightning indexer).",
               "Not run on a GPU: detected and wrapped by the no-GPU tests only.",
               "On an H100 or A100 the BF16 path leaves the attention Linears of a MoE model to vLLM (dense_stays): the vision tower's attention (0.42 GB) is packed on Ada only."]),
    "mistralai/Mistral-Small-4-119B-2603": dict(
        index="consolidated.safetensors.index.json", rules=MISTRAL, mtp=lambda n: False,
        notes=["Inventoried from the consolidated shards: the repo has params.json, so vLLM 0.30 reads the Mistral config format (-> MistralLarge3ForCausalLM, DeepseekV3 with MLA, + Pixtral vision) and loads consolidated*.safetensors by default. "
               "The repo's HF-format shards (model-0000x-of-00003) hold the same 120.92 GB with the experts fused (gate_up_proj [128, 4096, 4096]); vLLM 0.30's registry has no Mistral4 text model for them.",
               "Per-tensor FP8 (static activation scale), 36 layers, all MoE (128 experts, 1 shared). Task 3: per-tensor ran on Ada with a dynamic-activation slice; static is untested, and the default compile is not bit exact for per-tensor.",
               "Task 4 risk: on Hopper vLLM may pick FlashInfer-CUTLASS for the static per-tensor experts, which cannot take a compact, remapped expert set (the Triton, Marlin and DeepGEMM backends can); Task 4 has to pin the backend there."]),
    "deepseek-ai/DeepSeek-V4.1-Flash": dict(
        index="model.safetensors.index.json", rules=DEEPSEEK, mtp=lambda n: n.startswith("mtp."),
        notes=["DeepseekV41ForCausalLM, FP8 [32,32] e8m0 linears + FP4 routed experts (384 per layer, 40 layers) + 2 engram layers + 3 MTP layers.",
               "Not run on a GPU: detected and wrapped by the no-GPU tests only."]),
}


# ---- the inventory ----

def inventory(repo, ts):
    """Rows [kind, dtype, count, bytes, min weights, module, quant method, status code, note], sorted by bytes, and the keys no rule matches.
    The rows add up to the checkpoint's bytes (asserted when every kind matched)."""
    m = MODELS[repo]
    groups = {}
    for name, (dt, shape, _, a, b) in ts.items():
        g = groups.setdefault((kind(name, m["mtp"](name)), dt), [0, 0, 1 << 62])
        g[0] += 1
        g[1] += b - a
        g[2] = min(g[2], math.prod(shape) if shape else 1)
    rows, bad, owner = [], [], {}
    for (k, dt), (n, nbytes, mn) in groups.items():
        if SCALE.search(k):
            continue
        r = next((r for r in m["rules"] if r[0].search(f"{k}|{dt}")), None)
        if r is None:
            bad.append(f"{k}|{dt}")
            continue
        rows.append([k, dt, n, nbytes, mn, r[1], r[2], r[3], r[4]])
        if k not in owner or owner[k][3] < nbytes:
            owner[k] = rows[-1]
    for (k, dt), (n, nbytes, mn) in groups.items():
        sc = SCALE.search(k)
        if not sc:
            continue
        w = next((owner[c] for c in (k[: sc.start()] + ".weight", k[: sc.start()]) if c in owner), None)  # a scale belongs to its weight's layer
        if w is None:
            bad.append(f"{k}|{dt}")
            continue
        rows.append([k, dt, n, nbytes, mn, "scale of: " + w[5], "-", KEEP, R_SCALE])
    if not bad:
        assert sum(r[3] for r in rows) == sum(b - a for *_, a, b in ts.values())
    rows.sort(key=lambda r: -r[3])
    return rows, bad


def status_text(r):
    return f"kept as vLLM's: {r[8]}" if r[7] == KEEP else STATUS[r[7]] + (f"; {r[8]}" if r[8] else "")


def totals(rows):
    t = collections.Counter()
    why = collections.Counter()
    for r in rows:
        t[r[7]] += r[3]
        if r[7] == KEEP:
            why[r[8].split(" (")[0].split(";")[0]] += r[3]
    total = sum(t.values())
    fp8, bf = t[P8L] + t[P8M], t[PBF]
    saved = SAVE_FP8 * fp8 + SAVE_BF16 * bf
    today = SAVE_FP8 * t[P8L]
    return dict(total=total, p8l=t[P8L], p8m=t[P8M], fp8=fp8, bf16=bf, kept=t[KEEP], saved=saved, pct=100 * saved / total, today=today, today_pct=100 * today / total, why=why)


def gb(x):
    return f"{x / 1e9:.3f} GB" if x >= 1e6 else f"{x / 1e6:.3f} MB"


def markdown(results):
    out = ["# FP8 checkpoints: every tensor, packed or kept",
           "",
           "Private (glyd-gpu-private). Generated by `python gpu/fp8/inventory.py --md gpu/fp8/INVENTORY.md` from the safetensors headers of the five public repos (HTTP range reads) and",
           "the vLLM 0.30.0 model files each checkpoint loads through. Tracks the release gate of `glyd-engine/docs/design/plans/2026-10-05-fp8-lossless.md` (\"whole models, every",
           "tensor packed or kept with a stated reason\"). GB are 10^9 bytes of the checkpoint's files.",
           "",
           f"Expected saving = {SAVE_FP8:.1%} of the packed FP8 bytes + {SAVE_BF16:.1%} of the packed BF16 bytes (the measured averages), as a share of the checkpoint's bytes; scales, norms, FP4, and everything kept save nothing.",
           "\"Today\" is what the Task 3 wrapper (`vllm_fp8.py`, identity codec on the FP8 Linears of 1M weights or more, bmm layers excluded) covers on its branch: the FP8 Linears only; Task 4 (MoE) and the BF16 path inside the FP8 config are not built.",
           "Open items of the Task 3 review (`$S/fp8sdd/task-3-review.md`) that these tensors depend on: M2 (the MoE shared experts' side stream and the one shared scratch are not told apart: set `VLLM_DISABLE_SHARED_EXPERTS_STREAM` or refuse),",
           "M5 (the codec is fed what vLLM's kernel is fed: Marlin's tile-permuted int32 on Ada for block FP8 and every MXFP8, so the gain on those bytes is unmeasured), M10 (kv_b_proj), and Hopper (no logit produced yet).",
           "",
           "## Totals",
           "",
           "| Model | Total GB | Packed FP8 GB (Linear / MoE) | Packed BF16 GB | Kept GB | Expected saving GB | Expected saving | Today (Task 3 only) |",
           "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for repo, (rows, bad, t) in results.items():
        out.append(f"| {repo} | {t['total'] / 1e9:.1f} | {t['fp8'] / 1e9:.1f} ({t['p8l'] / 1e9:.1f} / {t['p8m'] / 1e9:.1f}) | {t['bf16'] / 1e9:.1f} | {t['kept'] / 1e9:.1f} | {t['saved'] / 1e9:.1f} | {t['pct']:.1f}% | {t['today_pct']:.2f}% |")
    for repo, (rows, bad, t) in results.items():
        out += ["", f"## {repo}", ""]
        out += [f"- {n}" for n in MODELS[repo]["notes"]]
        out += ["", f"Total {t['total'] / 1e9:.1f} GB: packed FP8 {t['fp8'] / 1e9:.1f} (Linears {t['p8l'] / 1e9:.1f}, MoE {t['p8m'] / 1e9:.1f}), packed BF16 {t['bf16'] / 1e9:.1f}, kept {t['kept'] / 1e9:.1f}; "
                f"expected saving {t['saved'] / 1e9:.1f} GB = {t['pct']:.1f}% ({t['today_pct']:.2f}% with only what Task 3 covers today).",
                "", "Kept, by reason:", "", "| Reason | Size |", "| :--- | ---: |"]
        out += [f"| {w} | {gb(b)} |" for w, b in t["why"].most_common()]
        out += ["", "| Kind (layer and expert numbers = N) | dtype | count | Size | vLLM 0.30 module | quant method vLLM gives it | Glyd status | Today |", "| :--- | :--- | ---: | ---: | :--- | :--- | :--- | :--- |"]
        for r in rows:
            out.append(f"| `{r[0]}` | {r[1]} | {r[2]} | {gb(r[3])} | {r[5]} | {r[6]} | {status_text(r)} | {TODAY[r[7]]} |")
    return "\n".join(out) + "\n"


def selftest():
    assert kind("model.language_model.layers.12.mlp.experts.255.up_proj.weight") == "model.language_model.layers.N.mlp.experts.N.up_proj.weight"
    assert kind("mtp.0.ffn.gate.bias", True) == "[mtp] mtp.N.ffn.gate.bias" and kind("lm_head.weight") == "lm_head.weight"
    assert SCALE.search("layers.N.attn.wo_a.scale") and SCALE.search("experts.gate_up_proj_scale_inv") and not SCALE.search("layers.N.hc_attn_scale")
    # a synthetic checkpoint: rows add up, a scale takes its weight's owner, an unknown kind is reported
    ts = {"model.language_model.layers.0.mlp.down_proj.weight": ("F8_E4M3", [4, 8], "f", 0, 32),
          "model.language_model.layers.0.mlp.down_proj.weight_scale_inv": ("BF16", [1, 1], "f", 32, 34),
          "model.language_model.layers.1.mlp.down_proj.weight": ("F8_E4M3", [4, 8], "f", 34, 66),
          "lm_head.weight": ("BF16", [2, 2], "f", 66, 74), "mtp.fc.weight": ("BF16", [2, 2], "f", 74, 82), "surprise.tensor": ("F32", [1], "f", 82, 86)}
    rows, bad = inventory("Qwen/Qwen3.8-27B-FP8", ts)
    assert bad == ["surprise.tensor|F32"], bad
    assert sum(r[3] for r in rows) == 82 and {r[0]: r[7] for r in rows}["model.language_model.layers.N.mlp.down_proj.weight"] == P8L
    sc = next(r for r in rows if r[8] == R_SCALE)
    assert sc[5].startswith("scale of: RowParallelLinear") and next(r for r in rows if r[0].startswith("[mtp]"))[7] == KEEP
    print("inventory selftest ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repos", nargs="*")
    ap.add_argument("--cache", help="a folder where the headers are kept (a rerun reads them from it)")
    ap.add_argument("--md", help="write the tables here")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    results, failed = collections.OrderedDict(), 0
    for repo in a.repos or list(MODELS):
        rows, bad = inventory(repo, headers(repo, MODELS[repo]["index"], a.cache))
        if bad:
            failed += 1
            print(f"{repo}: {len(bad)} kinds no rule matches:\n  " + "\n  ".join(bad), file=sys.stderr)
        t = totals(rows)
        results[repo] = (rows, bad, t)
        print(f"{repo}: {t['total'] / 1e9:.1f} GB total, packed FP8 {t['fp8'] / 1e9:.1f} (Linears {t['p8l'] / 1e9:.1f}, MoE {t['p8m'] / 1e9:.1f}), "
              f"packed BF16 {t['bf16'] / 1e9:.1f}, kept {t['kept'] / 1e9:.1f}, expected saving {t['saved'] / 1e9:.1f} GB = {t['pct']:.1f}% (Task 3 alone {t['today_pct']:.2f}%)")
        for w, b in t["why"].most_common(6):
            print(f"    kept {b / 1e9:8.3f} GB  {w}")
    if failed:
        sys.exit(1)
    if a.md:
        open(a.md, "w").write(markdown(results))
        print("wrote", a.md)


if __name__ == "__main__":
    main()
