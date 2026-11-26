//! Checkpoint tensor names by naming convention (R3a, #136): where each convention keeps a model's parts, as data. The description builder (`describe`) names a
//! block's roles through these tables and spells no tensor name itself; `Described::check` holds a weight map to a checkpoint's safetensors index.
//!
//! A table lists a part's roles in order, each with its tensor under the layer's path: `{E}` is an expert (the role is then each expert's matrices, as
//! `Source::Experts`), and `a|b` matrices stacked by rows (`Source::Rows`, or each expert's rows). A builder keeps the roles its kind has (`roles`'s `keep`).
//! The conventions: Hugging Face's Llama-style names (`HF`, with Qwen's, Gemma's and the Kimi-linear and Muse parts beside them), DeepSeek's two (the V3-style
//! names transformers gives the latent attention and its experts, which GLM-5.3 and Kimi-K3 keep; the release's own of DeepSeek-V4), and Nemotron-H's
//! (`backbone.`).

use super::describe::Source;
use std::collections::BTreeSet;
use std::path::Path;

/// A part's roles: (role, tensor under the layer's path).
pub type Roles = &'static [(&'static str, &'static str)];

/// Where a convention keeps the model's own tensors. `{p}` is the text model's prefix.
pub struct Convention {
    pub name: &'static str,
    /// A layer's path, before its number.
    pub layers: &'static str,
    pub embed: &'static str,
    pub final_norm: &'static str,
    pub head: &'static str,
    /// The model's roles besides the embedding, the final norm and the head (under `{p}`).
    pub extra: Roles,
    /// Parts of a checkpoint that the language model does not run (vision and audio towers, their projectors, the draft layers): prefixes, or a
    /// suffix written after `*`.
    pub unread: &'static [&'static str],
}

impl Convention {
    /// Layer `l`'s path.
    pub fn layer(&self, prefix: &str, l: usize) -> String {
        format!("{}{l}", self.layers.replace("{p}", prefix))
    }
    pub fn embed(&self, prefix: &str) -> String {
        self.embed.replace("{p}", prefix)
    }
    pub fn final_norm(&self, prefix: &str) -> String {
        self.final_norm.replace("{p}", prefix)
    }
    pub fn head(&self, prefix: &str) -> String {
        self.head.replace("{p}", prefix)
    }
    pub fn extra(&self, prefix: &str) -> Vec<(String, Source)> {
        roles(self.extra, prefix.trim_end_matches('.'), 0, |_| true)
    }
    pub fn unread(&self) -> Vec<String> {
        self.unread.iter().map(|s| s.to_string()).collect()
    }
}

const TOWERS: &[&str] = &["model.visual.", "mtp."];

/// Hugging Face's Llama-style names (the plain dense decoders', and the base of Qwen3.5's, Gemma 4's, GLM's and Muse's).
pub const HF: Convention = Convention {
    name: "Hugging Face (Llama-style)",
    layers: "{p}layers.",
    embed: "{p}embed_tokens.weight",
    final_norm: "{p}norm.weight",
    head: "lm_head.weight",
    extra: &[],
    unread: TOWERS,
};

/// Gemma 4's: Hugging Face's, with its towers.
pub const GEMMA: Convention = Convention {
    name: "Gemma",
    unread: &["model.embed_audio.", "model.embed_vision.", "model.vision_embedder.", "model.vision_tower.", "model.audio_tower."],
    ..HF
};

/// Kimi-K3's (the `kimi_linear` text model under `language_model.`), with the depth mix's output roles.
pub const KIMI: Convention = Convention {
    name: "Kimi linear",
    head: "language_model.lm_head.weight",
    extra: &[("out_mix_norm", "output_attn_res_norm.weight"), ("out_mix_proj", "output_attn_res_proj.weight")],
    unread: &["vision_tower.", "mm_projector."],
    ..HF
};

/// DeepSeek-V4 as released: no prefix, the weighted hyper-head's roles.
pub const DEEPSEEK: Convention = Convention {
    name: "DeepSeek (release)",
    layers: "layers.",
    embed: "embed.weight",
    final_norm: "norm.weight",
    head: "head.weight",
    extra: &[("out_hc_fn", "hc_head_fn"), ("out_hc_base", "hc_head_base"), ("out_hc_scale", "hc_head_scale")],
    unread: &["mtp."],
};

/// DeepSeek-V4.1 as released (DeepSeek's names, as V4's): no hyper-head (lagged mHC collapses with the last block's weights); the draft layers, the
/// vision tower, its aligner and image markers, and the image tokens' choice bias of each router not read by the text model.
pub const DEEPSEEK_V41: Convention = Convention {
    name: "DeepSeek-V4.1 (release)",
    layers: "layers.",
    embed: "embed.weight",
    final_norm: "norm.weight",
    head: "head.weight",
    extra: &[],
    unread: &["mtp.", "vision.", "aligner.", "image_", "*.ffn.gate.bias_vl"],
};

/// DeepSeek-V4 as transformers saves it.
pub const DEEPSEEK_HF: Convention = Convention {
    name: "DeepSeek (transformers)",
    layers: "model.layers.",
    embed: "model.embed_tokens.weight",
    final_norm: "model.norm.weight",
    extra: &[("out_hc_fn", "model.hc_head.hc_fn"), ("out_hc_base", "model.hc_head.hc_base"), ("out_hc_scale", "model.hc_head.hc_scale")],
    ..DEEPSEEK
};

/// Qwen3.8-Flash-Next's (Condor's model): Hugging Face's, with no final norm of its own (its final mixer's per-stream norm is the final
/// norm, the mixer's bottleneck the model's own roles).
pub const QWEN4X: Convention = Convention {
    name: "Qwen3.8-Flash-Next",
    final_norm: "{p}hyper_connection_mixer.hc_norm.weight",
    extra: &[("out_mix_down", "hyper_connection_mixer.input_mix_weight_down.weight"), ("out_mix_up", "hyper_connection_mixer.input_mix_weight_up.weight")],
    ..HF
};

/// Nemotron-H's (`backbone.`).
pub const NEMOTRON: Convention = Convention {
    name: "Nemotron-H",
    layers: "backbone.layers.",
    embed: "backbone.embeddings.weight",
    final_norm: "backbone.norm_f.weight",
    head: "lm_head.weight",
    extra: &[],
    unread: &["mtp."],
};

/// The parts' tables.
pub mod part {
    use super::Roles;

    // ---- Hugging Face, Llama-style
    /// Attention: its input norm, the q/k/v/o products and their biases, the q and k norms.
    pub const ATTN: Roles = &[
        ("norm_in", "input_layernorm.weight"),
        ("q", "self_attn.q_proj.weight"),
        ("q_bias", "self_attn.q_proj.bias"),
        ("k", "self_attn.k_proj.weight"),
        ("k_bias", "self_attn.k_proj.bias"),
        ("v", "self_attn.v_proj.weight"),
        ("v_bias", "self_attn.v_proj.bias"),
        ("o", "self_attn.o_proj.weight"),
        ("q_norm", "self_attn.q_norm.weight"),
        ("k_norm", "self_attn.k_norm.weight"),
    ];
    /// The gated MLP after the attention's norm.
    pub const MLP: Roles =
        &[("norm_in", "post_attention_layernorm.weight"), ("gate", "mlp.gate_proj.weight"), ("up", "mlp.up_proj.weight"), ("down", "mlp.down_proj.weight")];

    // ---- Qwen3.5 / 3.8
    /// Gated DeltaNet.
    pub const QWEN_LINEAR: Roles = &[
        ("norm_in", "input_layernorm.weight"),
        ("qkv", "linear_attn.in_proj_qkv.weight"),
        ("gate", "linear_attn.in_proj_z.weight"),
        ("beta", "linear_attn.in_proj_b.weight"),
        ("decay", "linear_attn.in_proj_a.weight"),
        ("conv", "linear_attn.conv1d.weight"),
        ("a_log", "linear_attn.A_log"),
        ("dt_bias", "linear_attn.dt_bias"),
        ("norm", "linear_attn.norm.weight"),
        ("o", "linear_attn.out_proj.weight"),
    ];
    /// The softmax-routed experts with a gated shared expert.
    pub const QWEN_MOE: Roles = &[
        ("norm_in", "post_attention_layernorm.weight"),
        ("router", "mlp.gate.weight"),
        ("experts_gate_up", "mlp.experts.{E}.gate_proj.weight|mlp.experts.{E}.up_proj.weight"),
        ("experts_down", "mlp.experts.{E}.down_proj.weight"),
        ("shared_gate", "mlp.shared_expert.gate_proj.weight"),
        ("shared_up", "mlp.shared_expert.up_proj.weight"),
        ("shared_down", "mlp.shared_expert.down_proj.weight"),
        ("shared_router", "mlp.shared_expert_gate.weight"),
    ];

    /// The same with each layer's routed experts stacked in two tensors, as transformers holds them and Qwen3.8-2.4T-A95B is released: `experts_gate_up`
    /// [experts, 2 inter, in] (the gate's rows first), `experts_down` [experts, out, inter].
    pub const QWEN_MOE_STACKED: Roles = &[
        ("norm_in", "post_attention_layernorm.weight"),
        ("router", "mlp.gate.weight"),
        ("experts_gate_up", "mlp.experts.gate_up_proj"),
        ("experts_down", "mlp.experts.down_proj"),
        ("shared_gate", "mlp.shared_expert.gate_proj.weight"),
        ("shared_up", "mlp.shared_expert.up_proj.weight"),
        ("shared_down", "mlp.shared_expert.down_proj.weight"),
        ("shared_router", "mlp.shared_expert_gate.weight"),
    ];

    // ---- Qwen3.8-Flash-Next (Condor): the gated residuals around each block, the indexer, the experts stacked, the n-gram embedding
    pub const QX_GR_ATTN: Roles = &[
        ("norm_in", "attn_hyper_connection.hc_norm.weight"),
        ("mix_down", "attn_hyper_connection.input_mix_weight_down.weight"),
        ("mix_up", "attn_hyper_connection.input_mix_weight_up.weight"),
        ("inject", "attn_hyper_connection.block_inject_weight.weight"),
    ];
    pub const QX_GR_MLP: Roles = &[
        ("norm_in", "mlp_hyper_connection.hc_norm.weight"),
        ("mix_down", "mlp_hyper_connection.input_mix_weight_down.weight"),
        ("mix_up", "mlp_hyper_connection.input_mix_weight_up.weight"),
        ("inject", "mlp_hyper_connection.block_inject_weight.weight"),
    ];
    pub const QX_INDEXER: Roles = &[
        ("index_qk", "self_attn.indexer.index_qk_proj.weight"),
        ("index_q_norm", "self_attn.indexer.q_layernorm.weight"),
        ("index_k_norm", "self_attn.indexer.k_layernorm.weight"),
    ];
    pub const QX_MOE: Roles = &[
        ("router", "mlp.gate.weight"),
        ("experts_gate_up", "mlp.experts.gate_up_proj"),
        ("experts_down", "mlp.experts.down_proj"),
        ("shared_gate", "mlp.shared_expert.gate_proj.weight"),
        ("shared_up", "mlp.shared_expert.up_proj.weight"),
        ("shared_down", "mlp.shared_expert.down_proj.weight"),
        ("shared_router", "mlp.shared_expert_gate.weight"),
    ];
    /// The draft head's own roles (`kinds::MtpInput::Streams`) under `mtp`: its front's two norms and two products, its final mixer.
    pub const QX_MTP: Roles = &[
        ("norm_embed", "pre_fc_norm_embedding.weight"),
        ("norm_hidden", "pre_fc_norm_hidden.weight"),
        ("fc_embedding", "fc_embedding.weight"),
        ("fc_hidden", "fc_hidden.weight"),
        ("final_norm", "hyper_connection_mixer.hc_norm.weight"),
        ("out_mix_down", "hyper_connection_mixer.input_mix_weight_down.weight"),
        ("out_mix_up", "hyper_connection_mixer.input_mix_weight_up.weight"),
    ];
    /// A draft head's own roles in a layer after the model's (`kinds::MtpInput::Joined`, DeepSeek-V3's layout that GLM-5.3 keeps: `model.layers.{L}.` from
    /// L = the model's layers): the next token's embedding's norm and the model's hidden state's, their joined product, the head's final norm.
    pub const MTP_LAYER: Roles =
        &[("eh_proj", "eh_proj.weight"), ("norm_embed", "enorm.weight"), ("norm_hidden", "hnorm.weight"), ("final_norm", "shared_head.norm.weight")];
    /// The n-gram embedding's block; its table's role `ngram` is the shards of `QX_NGRAM_SHARD` stacked (`shards`).
    pub const QX_NGRAM: Roles = &[
        ("norm_in", "ple.norm_query.weight"),
        ("key", "ple.key_proj.weight"),
        ("value", "ple.value_proj.weight"),
        ("norm_key", "ple.norm_key.weight"),
        ("norm_conv", "ple.norm_conv.weight"),
        ("conv", "ple.conv1d.weight"),
        ("ngram_mult", "ple.ple_embedding.layer_multipliers"),
        ("ngram_sizes", "ple.ple_embedding.ngram_heads_vocab_sizes"),
        ("ngram_offsets", "ple.ple_embedding.ngram_heads_offsets"),
    ];
    pub const QX_NGRAM_SHARD: &str = "ple.ple_embedding.ngram_embedding.shard_{S}.weight";

    // ---- the Qwen-VL vision tower (`kinds::VisionDesc`), under `model.visual`
    /// The tower's own roles: the patch product, the position table, the merger.
    pub const VISION: Roles = &[
        ("patch", "patch_embed.proj.weight"),
        ("patch_bias", "patch_embed.proj.bias"),
        ("pos", "pos_embed.weight"),
        ("merge_norm", "merger.norm.weight"),
        ("merge_norm_bias", "merger.norm.bias"),
        ("merge_up", "merger.linear_fc1.weight"),
        ("merge_up_bias", "merger.linear_fc1.bias"),
        ("merge_down", "merger.linear_fc2.weight"),
        ("merge_down_bias", "merger.linear_fc2.bias"),
    ];
    /// A block of the tower, under `model.visual.blocks.{l}`.
    pub const VISION_BLOCK: Roles = &[
        ("norm1", "norm1.weight"),
        ("norm1_bias", "norm1.bias"),
        ("qkv", "attn.qkv.weight"),
        ("qkv_bias", "attn.qkv.bias"),
        ("o", "attn.proj.weight"),
        ("o_bias", "attn.proj.bias"),
        ("norm2", "norm2.weight"),
        ("norm2_bias", "norm2.bias"),
        ("up", "mlp.linear_fc1.weight"),
        ("up_bias", "mlp.linear_fc1.bias"),
        ("down", "mlp.linear_fc2.weight"),
        ("down_bias", "mlp.linear_fc2.bias"),
    ];
    /// Where the tower's tensors are.
    pub const VISION_AT: &str = "model.visual";

    // ---- Gemma 4: sandwich norms and the layer's scalar
    pub const GEMMA_ATTN_OUT: Roles = &[("norm_out", "post_attention_layernorm.weight")];
    /// A layer's per-layer input branch (Gemma 4's E2B, E4B), and the model's tables of the inputs.
    pub const GEMMA_PER_LAYER: Roles = &[
        ("gate", "per_layer_input_gate.weight"),
        ("proj", "per_layer_projection.weight"),
        ("norm_out", "post_per_layer_input_norm.weight"),
        ("res_scale", "layer_scalar"),
    ];
    pub const GEMMA_PER_LAYER_IN: Roles =
        &[("ple_embed", "embed_tokens_per_layer.weight"), ("ple_proj", "per_layer_model_projection.weight"), ("ple_norm", "per_layer_projection_norm.weight")];
    pub const GEMMA_MLP: Roles = &[
        ("norm_in", "pre_feedforward_layernorm.weight"),
        ("gate", "mlp.gate_proj.weight"),
        ("up", "mlp.up_proj.weight"),
        ("down", "mlp.down_proj.weight"),
        ("norm_out", "post_feedforward_layernorm.weight"),
        ("res_scale", "layer_scalar"),
    ];

    // ---- DeepSeek, V3-style (GLM-5.3, Kimi-K3's latent attention)
    /// The latent attention: q in a low rank (or one product), the key-value latent, Kimi's output gate, o.
    pub const MLA: Roles = &[
        ("q_a", "self_attn.q_a_proj.weight"),
        ("q_a_norm", "self_attn.q_a_layernorm.weight"),
        ("q_b", "self_attn.q_b_proj.weight"),
        ("q", "self_attn.q_proj.weight"),
        ("kv_a", "self_attn.kv_a_proj_with_mqa.weight"),
        ("kv_a_norm", "self_attn.kv_a_layernorm.weight"),
        ("kv_b", "self_attn.kv_b_proj.weight"),
        ("g", "self_attn.g_proj.weight"),
        ("o", "self_attn.o_proj.weight"),
    ];
    /// The sparse attention's indexer (GLM-5.3), with the pooled keys' tensors of GLM-5.3-Flash's.
    pub const MLA_INDEXER: Roles = &[
        ("indexer_wq_b", "self_attn.indexer.wq_b.weight"),
        ("indexer_wk", "self_attn.indexer.wk.weight"),
        ("indexer_k_norm", "self_attn.indexer.k_norm.weight"),
        ("indexer_k_norm_bias", "self_attn.indexer.k_norm.bias"),
        ("indexer_weights", "self_attn.indexer.weights_proj.weight"),
        ("indexer_pool_ape", "self_attn.indexer.index_kpool_compress_ape"),
        ("indexer_pool_gate", "self_attn.indexer.index_kpool_compress_gate"),
    ];
    /// Sigmoid-routed experts with a choice bias and shared experts.
    pub const MOE: Roles = &[
        ("router", "mlp.gate.weight"),
        ("choice_bias", "mlp.gate.e_score_correction_bias"),
        ("experts_gate_up", "mlp.experts.{E}.gate_proj.weight|mlp.experts.{E}.up_proj.weight"),
        ("experts_down", "mlp.experts.{E}.down_proj.weight"),
        ("shared_gate", "mlp.shared_experts.gate_proj.weight"),
        ("shared_up", "mlp.shared_experts.up_proj.weight"),
        ("shared_down", "mlp.shared_experts.down_proj.weight"),
    ];
    /// The mHC streams around a block (GLM-5.3-Flash, DeepSeek-V4).
    pub const HC_ATTN: Roles = &[("hc_fn", "hc_attn_fn"), ("hc_base", "hc_attn_base"), ("hc_scale", "hc_attn_scale")];
    pub const HC_FFN: Roles = &[("hc_fn", "hc_ffn_fn"), ("hc_base", "hc_ffn_base"), ("hc_scale", "hc_ffn_scale")];

    // ---- Kimi linear (Kimi-K3, GLM-5.3-Flash): Kimi delta attention, the depth mix, the latent experts
    /// Kimi delta attention: q, k, v, beta, the forget gate's low rank, the output gate (full rank, or GLM's low rank), o, the three convolutions, A_log,
    /// dt_bias and the gated norm.
    pub const KDA: Roles = &[
        ("q", "self_attn.q_proj.weight"),
        ("k", "self_attn.k_proj.weight"),
        ("v", "self_attn.v_proj.weight"),
        ("beta", "self_attn.b_proj.weight"),
        ("decay_a", "self_attn.f_a_proj.weight"),
        ("decay_b", "self_attn.f_b_proj.weight"),
        ("gate", "self_attn.g_proj.weight"),
        ("gate_a", "self_attn.g_a_proj.weight"),
        ("gate_b", "self_attn.g_b_proj.weight"),
        ("o", "self_attn.o_proj.weight"),
        ("conv", "self_attn.q_conv1d.weight|self_attn.k_conv1d.weight|self_attn.v_conv1d.weight"),
        ("a_log", "self_attn.A_log"),
        ("dt_bias", "self_attn.dt_bias"),
        ("norm", "self_attn.o_norm.weight"),
    ];
    pub const NORM_IN: Roles = &[("norm_in", "input_layernorm.weight")];
    pub const NORM_POST: Roles = &[("norm_in", "post_attention_layernorm.weight")];
    pub const KIMI_ATTN_MIX: Roles =
        &[("norm_in", "input_layernorm.weight"), ("mix_norm", "self_attention_res_norm.weight"), ("mix_proj", "self_attention_res_proj.weight")];
    pub const KIMI_FFN_MIX: Roles = &[("norm_in", "post_attention_layernorm.weight"), ("mix_norm", "mlp_res_norm.weight"), ("mix_proj", "mlp_res_proj.weight")];
    /// The gated MLP's products alone (its norm the layer's own: GLM's, Kimi's).
    pub const MLP_BODY: Roles = &[("gate", "mlp.gate_proj.weight"), ("up", "mlp.up_proj.weight"), ("down", "mlp.down_proj.weight")];
    pub const KIMI_MOE: Roles = &[
        ("router", "block_sparse_moe.gate.weight"),
        ("choice_bias", "block_sparse_moe.gate.e_score_correction_bias"),
        ("latent_down", "block_sparse_moe.routed_expert_down_proj.weight"),
        ("latent_up", "block_sparse_moe.routed_expert_up_proj.weight"),
        ("latent_norm_out", "block_sparse_moe.routed_expert_norm.weight"),
        ("experts_gate_up", "block_sparse_moe.experts.{E}.w1.weight|block_sparse_moe.experts.{E}.w3.weight"),
        ("experts_down", "block_sparse_moe.experts.{E}.w2.weight"),
        ("shared_gate", "block_sparse_moe.shared_experts.gate_proj.weight"),
        ("shared_up", "block_sparse_moe.shared_experts.up_proj.weight"),
        ("shared_down", "block_sparse_moe.shared_experts.down_proj.weight"),
    ];

    // ---- Muse Spark: gated residuals, the gated attention output, the latent experts stacked
    pub const MUSE_ATTN: Roles = &[
        ("norm_in", "input_layernorm.weight"),
        ("res_gate", "post_attention_residual_gate.gate"),
        ("q", "self_attn.q_proj.weight"),
        ("k", "self_attn.k_proj.weight"),
        ("v", "self_attn.v_proj.weight"),
        ("g", "self_attn.gate_proj.weight"),
        ("o", "self_attn.o_proj.weight"),
    ];
    pub const MUSE_MOE: Roles = &[
        ("norm_in", "pre_feedforward_layernorm.weight"),
        ("norm_out", "post_feedforward_layernorm.weight"),
        ("res_gate", "post_feedforward_residual_gate.gate"),
        ("router", "mlp.gate.weight"),
        ("choice_bias", "mlp.gate.e_score_correction_bias"),
        ("latent_down", "mlp.pre_expert_proj.weight"),
        ("latent_norm_in", "mlp.pre_expert_norm.weight"),
        ("experts_gate_up", "mlp.experts.gate_up_proj"),
        ("experts_down", "mlp.experts.down_proj"),
        ("expert_norm", "mlp.experts.post_expert_norm.weight"),
        ("latent_up", "mlp.post_expert_proj.weight"),
    ];

    // ---- DeepSeek-V4 (the release's names; transformers' differ in the key-value norm alone, `attn.norm`)
    pub const V4_ATTN: Roles = &[
        ("norm_in", "attn_norm.weight"),
        ("q_a", "attn.wq_a.weight"),
        ("q_b", "attn.wq_b.weight"),
        ("kv", "attn.wkv.weight"),
        ("o_a", "attn.wo_a.weight"),
        ("o_b", "attn.wo_b.weight"),
        ("q_a_norm", "attn.q_norm.weight"),
    ];
    pub const V4_KV_NORM: Roles = &[("kv_norm", "attn.kv_norm.weight")];
    pub const V4_KV_NORM_HF: Roles = &[("kv_norm", "attn.norm.weight")];
    pub const V4_SINKS: Roles = &[("sinks", "attn.attn_sink")];
    /// The compressor of a compressing layer, and the indexer's (its own compressor, its queries and weights) of an overlapping one.
    pub const V4_COMP: Roles = &[
        ("comp_kv", "attn.compressor.wkv.weight"),
        ("comp_gate", "attn.compressor.wgate.weight"),
        ("comp_ape", "attn.compressor.ape"),
        ("comp_norm", "attn.compressor.norm.weight"),
    ];
    pub const V4_INDEXER: Roles = &[
        ("idx_kv", "attn.indexer.compressor.wkv.weight"),
        ("idx_gate", "attn.indexer.compressor.wgate.weight"),
        ("idx_ape", "attn.indexer.compressor.ape"),
        ("idx_norm", "attn.indexer.compressor.norm.weight"),
        ("idx_q_b", "attn.indexer.wq_b.weight"),
        ("idx_weights", "attn.indexer.weights_proj.weight"),
    ];
    /// DeepSeek-V4.1's compressor (ratio 1: no gate; no window bias) and its indexer: the queries and weights on an index source, the key product and
    /// its norm on one that is a kv source too.
    pub const V41_COMP: Roles =
        &[("comp_kv", "attn.compressor.wkv.weight"), ("comp_gate", "attn.compressor.wgate.weight"), ("comp_norm", "attn.compressor.norm.weight")];
    pub const V41_INDEXER: Roles = &[
        ("idx_q_b", "attn.indexer.wq_b.weight"),
        ("idx_weights", "attn.indexer.weights_proj.weight"),
        ("idx_k", "attn.indexer.wk.weight"),
        ("idx_k_norm", "attn.indexer.k_norm.weight"),
    ];
    /// DeepSeek-V4.1's Engram: the n-gram tables, the key and value product (keys first), the gate's two weights (the streams' norm's, the keys').
    pub const ENGRAM: Roles = &[("ngram", "engram.embed.weight"), ("kv", "engram.wkv.weight"), ("norm_in", "engram.q_weight"), ("norm_key", "engram.k_weight")];
    pub const V4_MOE: Roles = &[
        ("norm_in", "ffn_norm.weight"),
        ("router", "ffn.gate.weight"),
        ("hash", "ffn.gate.tid2eid"),
        ("choice_bias", "ffn.gate.bias"),
        ("experts_gate_up", "ffn.experts.{E}.w1.weight|ffn.experts.{E}.w3.weight"),
        ("experts_down", "ffn.experts.{E}.w2.weight"),
        ("shared_gate", "ffn.shared_experts.w1.weight"),
        ("shared_up", "ffn.shared_experts.w3.weight"),
        ("shared_down", "ffn.shared_experts.w2.weight"),
    ];

    // ---- Nemotron-H: a layer is one block, `norm` then the mixer
    pub const NH_MAMBA: Roles = &[
        ("norm_in", "norm.weight"),
        ("in", "mixer.in_proj.weight"),
        ("conv", "mixer.conv1d.weight"),
        ("conv_bias", "mixer.conv1d.bias"),
        ("a_log", "mixer.A_log"),
        ("d", "mixer.D"),
        ("dt_bias", "mixer.dt_bias"),
        ("norm", "mixer.norm.weight"),
        ("o", "mixer.out_proj.weight"),
    ];
    pub const NH_ATTN: Roles =
        &[("norm_in", "norm.weight"), ("q", "mixer.q_proj.weight"), ("k", "mixer.k_proj.weight"), ("v", "mixer.v_proj.weight"), ("o", "mixer.o_proj.weight")];
    pub const NH_MOE: Roles = &[
        ("norm_in", "norm.weight"),
        ("router", "mixer.gate.weight"),
        ("choice_bias", "mixer.gate.e_score_correction_bias"),
        ("latent_down", "mixer.fc1_latent_proj.weight"),
        ("latent_up", "mixer.fc2_latent_proj.weight"),
        ("shared_up", "mixer.shared_experts.up_proj.weight"),
        ("shared_down", "mixer.shared_experts.down_proj.weight"),
        ("experts_up", "mixer.experts.{E}.up_proj.weight"),
        ("experts_down", "mixer.experts.{E}.down_proj.weight"),
    ];
}

/// The roles of `table` that `keep` keeps, named under the path `at` (`experts` of them where a role is each expert's).
pub fn roles(table: Roles, at: &str, experts: usize, keep: impl Fn(&str) -> bool) -> Vec<(String, Source)> {
    let name = |m: &str| if at.is_empty() { m.to_string() } else { format!("{at}.{m}") };
    table
        .iter()
        .filter(|(r, _)| keep(r))
        .map(|&(r, pat)| {
            let src = if pat.contains("{E}") {
                Source::Experts((0..experts).map(|e| pat.split('|').map(|m| name(&m.replace("{E}", &e.to_string()))).collect()).collect())
            } else if pat.contains('|') {
                Source::Rows(pat.split('|').map(name).collect())
            } else {
                Source::Tensor(name(pat))
            };
            (r.to_string(), src)
        })
        .collect()
}

/// One tensor saved in `n` shards along its rows (`{S}` in `pat` the shard's number), under the path `at`: the shards stacked.
pub fn shards(pat: &str, at: &str, n: usize) -> Source {
    Source::Rows((0..n).map(|s| format!("{at}.{}", pat.replace("{S}", &s.to_string()))).collect())
}

/// Every role of `table`.
pub fn all(_: &str) -> bool {
    true
}

/// The tensor names a checkpoint directory holds: its safetensors index's weight map (`model.safetensors.index.json`), else its one file's header.
pub fn index_names(dir: &Path) -> crate::ffi::Result<BTreeSet<String>> {
    let io = |e: std::io::Error| crate::ffi::Error(format!("{}: {e}", dir.display()));
    let idx = dir.join("model.safetensors.index.json");
    if idx.exists() {
        let v: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&idx).map_err(io)?).map_err(|e| crate::ffi::Error(format!("{}: {e}", idx.display())))?;
        let map = v.get("weight_map").and_then(|m| m.as_object()).ok_or_else(|| crate::ffi::Error(format!("{}: no weight_map", idx.display())))?;
        return Ok(map.keys().cloned().collect());
    }
    let ck = glyd_gpu::safetensors::Checkpoint::open(dir).map_err(io)?;
    Ok(ck.names().map(str::to_string).collect())
}
