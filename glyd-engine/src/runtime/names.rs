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

/// Muse Spark's: Hugging Face's, with its audio and vision towers, the vision adapter and its projection, which the text model does not run.
pub const MUSE: Convention =
    Convention { name: "Muse Spark", unread: &["model.audio_tower.", "model.vision_tower.", "model.vision_adapter.", "model.vision_projection."], ..HF };

/// MiMo V2's: Hugging Face's, with its audio encoder, the speech codes' embeddings and the vision tower, which the text model does not run. Its draft
/// layers (`model.mtp.layers.`) are read as draft heads.
pub const MIMO: Convention = Convention { name: "MiMo V2", unread: &["audio_encoder.", "speech_embeddings.", "visual."], ..HF };

/// MiniMax M3's (its vision-language wrapper `minimax_m3_vl`): the text model under `language_model.model.`, its head beside it at `language_model.lm_head`;
/// the vision tower and its two projectors into the text model's width, which the text model does not run; and an expert matrix's `input_global_scale`,
/// the scale of a static 4-bit quantization of the matrix's input that the release's quantization does not use (its activations are quantized by their own
/// size, `activation_scheme` "dynamic") and that the weights' reconstruction does not read.
pub const MINIMAX: Convention = Convention {
    name: "MiniMax M3",
    layers: "language_model.model.layers.",
    embed: "language_model.model.embed_tokens.weight",
    final_norm: "language_model.model.norm.weight",
    head: "language_model.lm_head.weight",
    extra: &[],
    unread: &["vision_tower.", "multi_modal_projector.", "patch_merge_mlp.", "*.input_global_scale"],
};

/// LFM's: Hugging Face's, the norm after the last layer under the name `embedding_norm` (it norms no embedding: the head reads it), and no towers.
pub const LFM: Convention = Convention { name: "LFM", final_norm: "{p}embedding_norm.weight", unread: &[], ..HF };

/// Ling's (the `bailing_hybrid` text model under `model.`, in its vision-language wrapper too): Hugging Face's, the embedding under the name
/// `word_embeddings`; the vision tower and its projector into the text model's width, which the text model does not run. Its draft layers follow the
/// model's under the same names (`model.layers.{L}.`) and are read as draft heads.
pub const LING: Convention = Convention { name: "Ling", embed: "{p}word_embeddings.weight", unread: &["model.visual.", "linear_proj."], ..HF };

/// ERNIE 5.5's: Hugging Face's layout under `model.`, with the hyper-head's roles (`mhc_contract`: its matrix stored transposed) and the logits'
/// correction's two (`model.lm_head.multimax_*`, beside the head `lm_head.weight`). Its drafter is under `model.mtp.` and is read (`part::ERNIE_DRAFT`).
pub const ERNIE: Convention = Convention {
    name: "ERNIE",
    extra: &[
        ("out_hc_fn", "^mhc_contract.hc_head_fn"),
        ("out_hc_base", "mhc_contract.hc_head_base"),
        ("out_hc_scale", "mhc_contract.hc_head_scale"),
        ("logit_ranges", "lm_head.multimax_ranges"),
        ("logit_ts", "lm_head.multimax_ts"),
    ],
    unread: &[],
    ..HF
};

/// Hunyuan's: Hugging Face's layout under `model.`, with the hyper-head's roles (`model.hc_head`). Its drafter (`model.dfly_layers.`) and its vision
/// tower with the adapter into the text model's width (`vit.`) are parts the description leaves: the text model runs without either.
pub const HUNYUAN: Convention = Convention {
    name: "Hunyuan",
    extra: &[("out_hc_fn", "hc_head.hc_head_fn"), ("out_hc_base", "hc_head.hc_head_base"), ("out_hc_scale", "hc_head.hc_head_scale")],
    unread: &["vit.", "model.dfly_layers."],
    ..HF
};

/// Mistral's own names, as its consolidated checkpoints hold them (Mistral Large's release, whose config's `transformers_weights` points transformers at
/// that checkpoint): no prefix, the embedding `tok_embeddings`, the head `output`; the vision encoder, its norm, patch merger and adapter into the text
/// model's width, and the draft model (`mtp_dspark.`), which the text model does not run.
pub const MISTRAL: Convention = Convention {
    name: "Mistral (consolidated)",
    layers: "layers.",
    embed: "tok_embeddings.weight",
    final_norm: "norm.weight",
    head: "output.weight",
    extra: &[],
    unread: &["vision_encoder.", "pre_mm_projector_norm.", "patch_merger.", "vision_language_adapter.", "mtp_dspark."],
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

/// Qwen3.8-Flash-Next's (Ibis's model): Hugging Face's, with no final norm of its own (its final mixer's per-stream norm is the final
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

    // ---- Granite Swash: Llama-style names, and a learned sink a head beside the attention's matrices
    pub const GRANITE_SINKS: Roles = &[("sinks", "self_attn.sinks")];

    // ---- MiMo V2: the query, key and value products one matrix, a sink a head where a layer sort has them, a tensor an expert's matrix
    /// Attention with its three products fused (`attention_projection_layout` "fused_qkv"): `qkv` is cut into the kind's `q`, `k` and `v` by the builder
    /// (`describe::mimo_v2`: the matrix holds them a shard at a time, not q then k then v).
    pub const MIMO_ATTN: Roles = &[
        ("norm_in", "input_layernorm.weight"),
        ("qkv", "self_attn.qkv_proj.weight"),
        ("o", "self_attn.o_proj.weight"),
        ("sinks", "self_attn.attention_sink_bias"),
    ];
    /// The same with a matrix a product (`attention_projection_layout` "split" or unsaid).
    pub const MIMO_ATTN_SPLIT: Roles = &[
        ("norm_in", "input_layernorm.weight"),
        ("q", "self_attn.q_proj.weight"),
        ("k", "self_attn.k_proj.weight"),
        ("v", "self_attn.v_proj.weight"),
        ("o", "self_attn.o_proj.weight"),
        ("sinks", "self_attn.attention_sink_bias"),
    ];
    /// Sigmoid-routed experts with a choice bias and no shared expert.
    pub const MIMO_MOE: Roles = &[
        ("norm_in", "post_attention_layernorm.weight"),
        ("router", "mlp.gate.weight"),
        ("choice_bias", "mlp.gate.e_score_correction_bias"),
        ("experts_gate_up", "mlp.experts.{E}.gate_proj.weight|mlp.experts.{E}.up_proj.weight"),
        ("experts_down", "mlp.experts.{E}.down_proj.weight"),
    ];
    /// A draft layer's MLP (under `model.mtp.layers.{j}`): the dense one, its norm under the draft layer's name.
    pub const MIMO_MTP_MLP: Roles =
        &[("norm_in", "pre_mlp_layernorm.weight"), ("gate", "mlp.gate_proj.weight"), ("up", "mlp.up_proj.weight"), ("down", "mlp.down_proj.weight")];
    /// A draft head's own roles (`kinds::MtpInput::Joined`).
    pub const MIMO_MTP: Roles =
        &[("eh_proj", "eh_proj.weight"), ("norm_embed", "enorm.weight"), ("norm_hidden", "hnorm.weight"), ("final_norm", "final_layernorm.weight")];

    // ---- MiniMax M3: Llama-style attention with the block indexer's products and norms beside it; Mixtral's names for the experts
    /// Attention with q and k norms a head and, on a sparse layer, the indexer's queries, its one key and their norms.
    pub const MINIMAX_ATTN: Roles = &[
        ("norm_in", "input_layernorm.weight"),
        ("q", "self_attn.q_proj.weight"),
        ("k", "self_attn.k_proj.weight"),
        ("v", "self_attn.v_proj.weight"),
        ("o", "self_attn.o_proj.weight"),
        ("q_norm", "self_attn.q_norm.weight"),
        ("k_norm", "self_attn.k_norm.weight"),
        ("index_q", "self_attn.index_q_proj.weight"),
        ("index_k", "self_attn.index_k_proj.weight"),
        ("index_q_norm", "self_attn.index_q_norm.weight"),
        ("index_k_norm", "self_attn.index_k_norm.weight"),
    ];
    /// Sigmoid-routed experts (`w1` the gate, `w3` the up, `w2` the down) with a choice bias beside the router and one shared expert.
    pub const MINIMAX_MOE: Roles = &[
        ("norm_in", "post_attention_layernorm.weight"),
        ("router", "block_sparse_moe.gate.weight"),
        ("choice_bias", "block_sparse_moe.e_score_correction_bias"),
        ("experts_gate_up", "block_sparse_moe.experts.{E}.w1.weight|block_sparse_moe.experts.{E}.w3.weight"),
        ("experts_down", "block_sparse_moe.experts.{E}.w2.weight"),
        ("shared_gate", "block_sparse_moe.shared_experts.gate_proj.weight"),
        ("shared_up", "block_sparse_moe.shared_experts.up_proj.weight"),
        ("shared_down", "block_sparse_moe.shared_experts.down_proj.weight"),
    ];

    // ---- Domyn Edge: Llama-style parts under its own names, a norm after each part as well as before it
    /// Attention between its two norms; the q norm has no weight.
    pub const DOMYN_ATTN: Roles = &[
        ("norm_in", "pre_attention_norm.weight"),
        ("q", "attention.q_proj.weight"),
        ("k", "attention.k_proj.weight"),
        ("v", "attention.v_proj.weight"),
        ("o", "attention.o_proj.weight"),
        ("k_norm", "attention.k_norm.weight"),
        ("norm_out", "post_attention_norm.weight"),
    ];
    /// The gated MLP between its two norms.
    pub const DOMYN_MLP: Roles = &[
        ("norm_in", "pre_feedforward_norm.weight"),
        ("gate", "feedforward.gate_proj.weight"),
        ("up", "feedforward.up_proj.weight"),
        ("down", "feedforward.down_proj.weight"),
        ("norm_out", "post_feedforward_norm.weight"),
    ];

    // ---- LFM: a layer's mixer (a gated short convolution or an attention) and its MLP, each between two norms where the config has the second
    /// The gated short convolution (`kinds::ShortConv`), its biases where the config has them.
    pub const LFM_CONV: Roles = &[
        ("norm_in", "operator_norm.weight"),
        ("in", "conv.in_proj.weight"),
        ("in_bias", "conv.in_proj.bias"),
        ("conv", "conv.conv.weight"),
        ("conv_bias", "conv.conv.bias"),
        ("o", "conv.out_proj.weight"),
        ("o_bias", "conv.out_proj.bias"),
        ("norm_out", "post_operator_norm.weight"),
    ];
    /// Attention with normed queries and keys and the gate of one value a head (`g` [heads, hidden]).
    pub const LFM_ATTN: Roles = &[
        ("norm_in", "operator_norm.weight"),
        ("q", "self_attn.q_proj.weight"),
        ("k", "self_attn.k_proj.weight"),
        ("v", "self_attn.v_proj.weight"),
        ("o", "self_attn.o_proj.weight"),
        ("q_norm", "self_attn.q_norm.weight"),
        ("k_norm", "self_attn.k_norm.weight"),
        ("g", "self_attn.gate_proj.weight"),
        ("norm_out", "post_operator_norm.weight"),
    ];
    /// The gated MLP: `w1` the gate, `w3` the up projection, `w2` the down.
    pub const LFM_MLP: Roles = &[
        ("norm_in", "ffn_norm.weight"),
        ("gate", "feed_forward.w1.weight"),
        ("up", "feed_forward.w3.weight"),
        ("down", "feed_forward.w2.weight"),
        ("norm_out", "post_ffn_norm.weight"),
    ];

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

    // ---- Qwen3.8-Flash-Next (Ibis): the gated residuals around each block, the indexer, the experts stacked, the n-gram embedding
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
    /// Sigmoid-routed experts with a choice bias and shared experts (the shared experts before the routed ones, as a lossy save packs them: src/pack.rs `walk`).
    pub const MOE: Roles = &[
        ("router", "mlp.gate.weight"),
        ("choice_bias", "mlp.gate.e_score_correction_bias"),
        ("shared_gate", "mlp.shared_experts.gate_proj.weight"),
        ("shared_up", "mlp.shared_experts.up_proj.weight"),
        ("shared_down", "mlp.shared_experts.down_proj.weight"),
        ("experts_gate_up", "mlp.experts.{E}.gate_proj.weight|mlp.experts.{E}.up_proj.weight"),
        ("experts_down", "mlp.experts.{E}.down_proj.weight"),
    ];
    /// The mHC streams around a block (GLM-5.3-Flash, DeepSeek-V4).
    pub const HC_ATTN: Roles = &[("hc_fn", "hc_attn_fn"), ("hc_base", "hc_attn_base"), ("hc_scale", "hc_attn_scale")];
    pub const HC_FFN: Roles = &[("hc_fn", "hc_ffn_fn"), ("hc_base", "hc_ffn_base"), ("hc_scale", "hc_ffn_scale")];

    // ---- Kimi linear (Kimi-K3, GLM-5.3-Flash): Kimi delta attention, the depth mix, the latent experts
    /// Kimi delta attention: q, k, v, o, the forget gate's low rank, the output gate (full rank, or GLM's low rank), beta, the three convolutions, A_log,
    /// dt_bias and the gated norm (its matrices in the order a lossy save packs them: src/pack.rs `walk`).
    pub const KDA: Roles = &[
        ("q", "self_attn.q_proj.weight"),
        ("k", "self_attn.k_proj.weight"),
        ("v", "self_attn.v_proj.weight"),
        ("o", "self_attn.o_proj.weight"),
        ("decay_a", "self_attn.f_a_proj.weight"),
        ("decay_b", "self_attn.f_b_proj.weight"),
        ("gate", "self_attn.g_proj.weight"),
        ("gate_a", "self_attn.g_a_proj.weight"),
        ("gate_b", "self_attn.g_b_proj.weight"),
        ("beta", "self_attn.b_proj.weight"),
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

    // ---- Ling: a layer's mixer under `attention` (the Kimi delta attention or the latent attention), the experts' choice bias `expert_bias`
    /// Kimi delta attention with every product at its full rank: q, k, v, o, the forget gate `f_proj`, the output gate `g_proj`, beta, the three
    /// convolutions, A_log, dt_bias and the gated norm.
    pub const LING_KDA: Roles = &[
        ("q", "attention.q_proj.weight"),
        ("k", "attention.k_proj.weight"),
        ("v", "attention.v_proj.weight"),
        ("o", "attention.o_proj.weight"),
        ("decay", "attention.f_proj.weight"),
        ("gate", "attention.g_proj.weight"),
        ("beta", "attention.b_proj.weight"),
        ("conv", "attention.q_conv1d.weight|attention.k_conv1d.weight|attention.v_conv1d.weight"),
        ("a_log", "attention.A_log"),
        ("dt_bias", "attention.dt_bias"),
        ("norm", "attention.o_norm.weight"),
    ];
    /// The latent attention: q as one product (or in a low rank), the key-value latent, the output gate (`g` [heads, hidden] where it is a value a
    /// head), and the output product under the name `dense`.
    pub const LING_MLA: Roles = &[
        ("q_a", "attention.q_a_proj.weight"),
        ("q_a_norm", "attention.q_a_layernorm.weight"),
        ("q_b", "attention.q_b_proj.weight"),
        ("q", "attention.q_proj.weight"),
        ("kv_a", "attention.kv_a_proj_with_mqa.weight"),
        ("kv_a_norm", "attention.kv_a_layernorm.weight"),
        ("kv_b", "attention.kv_b_proj.weight"),
        ("g", "attention.g_proj.weight"),
        ("o", "attention.dense.weight"),
    ];
    /// Sigmoid-routed experts with a choice bias and one shared MLP (the shared one before the routed ones, as `MOE`).
    pub const LING_MOE: Roles = &[
        ("router", "mlp.gate.weight"),
        ("choice_bias", "mlp.gate.expert_bias"),
        ("shared_gate", "mlp.shared_experts.gate_proj.weight"),
        ("shared_up", "mlp.shared_experts.up_proj.weight"),
        ("shared_down", "mlp.shared_experts.down_proj.weight"),
        ("experts_gate_up", "mlp.experts.{E}.gate_proj.weight|mlp.experts.{E}.up_proj.weight"),
        ("experts_down", "mlp.experts.{E}.down_proj.weight"),
    ];
    /// A draft head's own roles in a layer after the model's (`kinds::MtpInput::Joined`).
    pub const LING_MTP: Roles =
        &[("eh_proj", "eh_proj.weight"), ("norm_embed", "enorm.weight"), ("norm_hidden", "hnorm.weight"), ("final_norm", "final_layernorm.weight")];

    // ---- Mistral Large (Mistral's names): the latent attention with its queries' scale by position, the indexer on the layer's input, the experts
    /// The latent attention: the queries one product (`wq`), the key-value latent (absent on a layer that shares another's), the queries' scale by
    /// position (`ssmax_scaling.s`, `.b`, a value a head each), the output product.
    pub const MISTRAL_MLA: Roles = &[
        ("norm_in", "attention_norm.weight"),
        ("q", "attention.wq.weight"),
        ("kv_a", "attention.wkv_a_with_mqa.weight"),
        ("kv_a_norm", "attention.kv_a_norm.weight"),
        ("kv_b", "attention.wkv_b.weight"),
        ("o", "attention.wo.weight"),
        ("q_scale_s", "attention.ssmax_scaling.s"),
        ("q_scale_b", "attention.ssmax_scaling.b"),
    ];
    /// The indexer of a layer that has its own: the LayerNorm of the layer's input its queries read, their product and their scale by position, the keys'
    /// product and LayerNorm, the heads' weights.
    pub const MISTRAL_INDEXER: Roles = &[
        ("indexer_q_norm", "attention.indexer.q_input_norm.weight"),
        ("indexer_q_norm_bias", "attention.indexer.q_input_norm.bias"),
        ("indexer_wq_b", "attention.indexer.wq_b.weight"),
        ("indexer_scale_s", "attention.indexer.ssmax_scaling.s"),
        ("indexer_scale_b", "attention.indexer.ssmax_scaling.b"),
        ("indexer_wk", "attention.indexer.wk.weight"),
        ("indexer_k_norm", "attention.indexer.k_norm.weight"),
        ("indexer_k_norm_bias", "attention.indexer.k_norm.bias"),
        ("indexer_weights", "attention.indexer.weights_proj.weight"),
    ];
    /// The learned factors of the rotary's frequencies: one tensor of the model, read by every layer that rotates.
    pub const MISTRAL_ROPE_ALPHA: &str = "lerope.alpha";
    /// The gated MLP: `w1` the gate, `w3` the up projection, `w2` the down.
    pub const MISTRAL_MLP: Roles =
        &[("norm_in", "ffn_norm.weight"), ("gate", "feed_forward.w1.weight"), ("up", "feed_forward.w3.weight"), ("down", "feed_forward.w2.weight")];
    /// Softmax-routed experts with one shared MLP, a tensor an expert's matrix (the shared one before the routed ones, as `MOE`).
    pub const MISTRAL_MOE: Roles = &[
        ("norm_in", "ffn_norm.weight"),
        ("router", "gate.weight"),
        ("shared_gate", "shared_experts.w1.weight"),
        ("shared_up", "shared_experts.w3.weight"),
        ("shared_down", "shared_experts.w2.weight"),
        ("experts_gate_up", "experts.{E}.w1.weight|experts.{E}.w3.weight"),
        ("experts_down", "experts.{E}.w2.weight"),
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

    // ---- ERNIE 5.5: a layer's attention under `self_attn` (the compressed attention's products under names of its own, the latent attention's as
    // DeepSeek-V3's), its parts that are no product under `self_attn.core_attention`; the experts under `block_sparse_moe` as Kimi's. `^`: a matrix stored
    // transposed; `|`: tensors stacked, or scalars joined
    /// The compressed attention: the query latent and its norm, the queries, the one key-value product and its norm, the grouped low-rank output
    /// (`linear_o_group_proj`, a tensor with no `.weight`) and the output product, the output gate of the query latent, the heads' mix, the sinks.
    pub const ERNIE_COMP_ATTN: Roles = &[
        ("norm_in", "input_layernorm.weight"),
        ("q_a", "self_attn.linear_q_down_proj.weight"),
        ("q_a_norm", "self_attn.q_layernorm.weight"),
        ("q_b", "self_attn.linear_q_up_proj.weight"),
        ("kv", "self_attn.linear_kv_proj.weight"),
        ("kv_norm", "self_attn.kv_layernorm.weight"),
        ("o_a", "self_attn.linear_o_group_proj"),
        ("o_b", "self_attn.o_proj.weight"),
        ("g", "self_attn.gate_proj.weight"),
        ("mix_u", "self_attn.vha_postmix_U"),
        ("mix_v", "self_attn.vha_postmix_V"),
        ("sinks", "self_attn.core_attention.attn_sink"),
    ];
    /// A compressing layer's compressor.
    pub const ERNIE_COMP: Roles = &[
        ("comp_kv", "self_attn.core_attention.compressor.linear_wkv.weight"),
        ("comp_gate", "self_attn.core_attention.compressor.linear_wgate.weight"),
        ("comp_ape", "self_attn.core_attention.compressor.ape"),
        ("comp_norm", "self_attn.core_attention.compressor.norm.weight"),
    ];
    /// The latent attention: its products, the output gate of the query latent, the heads' mix, the sinks (`softmax_offset`), and its indexer's tensors.
    pub const ERNIE_MLA: Roles = &[
        ("norm_in", "input_layernorm.weight"),
        ("q_a", "self_attn.q_a_proj.weight"),
        ("q_a_norm", "self_attn.q_a_layernorm.weight"),
        ("q_b", "self_attn.q_b_proj.weight"),
        ("kv_a", "self_attn.kv_a_proj_with_mqa.weight"),
        ("kv_a_norm", "self_attn.kv_a_layernorm.weight"),
        ("kv_b", "self_attn.kv_b_proj.weight"),
        ("o", "self_attn.o_proj.weight"),
        ("g", "self_attn.gate_proj.weight"),
        ("mix_u", "self_attn.vha_postmix_U"),
        ("mix_v", "self_attn.vha_postmix_V"),
        ("sinks", "self_attn.core_attention.softmax_offset"),
        ("indexer_wq_b", "self_attn.core_attention.indexer.wq_b.weight"),
        ("indexer_wk", "self_attn.core_attention.indexer.wk.weight"),
        ("indexer_k_norm", "self_attn.core_attention.indexer.k_norm.weight"),
        ("indexer_k_norm_bias", "self_attn.core_attention.indexer.k_norm.bias"),
        ("indexer_weights", "self_attn.core_attention.indexer.weights_proj.weight"),
    ];
    /// The residual streams around a block: the mixing matrix stored transposed, its bias, and its three scales as scalars apart (the input's, the
    /// output's, the streams' mixing's, in `hc_scale`'s order).
    pub const ERNIE_HC_ATTN: Roles = &[
        ("hc_fn", "^self_attention_hyper_connection.mapping_proj.weight"),
        ("hc_base", "self_attention_hyper_connection.bias"),
        ("hc_scale", "self_attention_hyper_connection.alpha_pre|self_attention_hyper_connection.alpha_post|self_attention_hyper_connection.alpha_res"),
    ];
    pub const ERNIE_HC_FFN: Roles = &[
        ("hc_fn", "^mlp_hyper_connection.mapping_proj.weight"),
        ("hc_base", "mlp_hyper_connection.bias"),
        ("hc_scale", "mlp_hyper_connection.alpha_pre|mlp_hyper_connection.alpha_post|mlp_hyper_connection.alpha_res"),
    ];
    /// The experts in a latent: the two routers' rows stacked, the choice bias, the learned scale an expert, the latent's two projections and its norm,
    /// the experts (w1 the gate, w3 the up, w2 the down) and the shared expert named as they are.
    pub const ERNIE_MOE: Roles = &[
        ("norm_in", "post_attention_layernorm.weight"),
        ("router", "block_sparse_moe.gate.weight|block_sparse_moe.gate.weight_1"),
        ("choice_bias", "block_sparse_moe.e_score_correction_bias"),
        ("expert_scale", "block_sparse_moe.gate.routed_scaling_factor_param"),
        ("latent_down", "block_sparse_moe.fc1_latent_proj.weight"),
        ("latent_up", "block_sparse_moe.fc2_latent_proj.weight"),
        ("latent_norm_out", "block_sparse_moe.latent_norm.weight"),
        ("shared_gate", "block_sparse_moe.shared_experts.w1.weight"),
        ("shared_up", "block_sparse_moe.shared_experts.w3.weight"),
        ("shared_down", "block_sparse_moe.shared_experts.w2.weight"),
        ("experts_gate_up", "block_sparse_moe.experts.{E}.w1.weight|block_sparse_moe.experts.{E}.w3.weight"),
        ("experts_down", "block_sparse_moe.experts.{E}.w2.weight"),
    ];
    /// The drafter's own roles under `model.mtp`: main_x's product and norm, the Markov head's two factors (the embedding stored transposed), the
    /// confidence head, the final norm and the drafter's hyper-head.
    pub const ERNIE_DRAFT: Roles = &[
        ("main_proj", "feature_projection.weight"),
        ("main_norm", "feature_norm.weight"),
        ("markov_embed", "^markov_head.markov_w1.weight"),
        ("markov_head", "markov_head.markov_w2.weight"),
        ("confidence", "confidence_head.projection.weight"),
        ("final_norm", "norm.weight"),
        ("out_hc_fn", "^mhc_contract.hc_head_fn"),
        ("out_hc_base", "mhc_contract.hc_head_base"),
        ("out_hc_scale", "mhc_contract.hc_head_scale"),
    ];

    // ---- Hunyuan: DeepSeek-V3's names for the latent attention, with its gate, sinks and the norm after it; the compressed indexer under
    // `self_attn.compressed_indexer`; the streams' weights under `hc_attn_layer.hc_pre` and `hc_mlp_layer.hc_pre`; the experts stacked a layer; the
    // n-gram memory under `engram_layer`
    /// The latent attention: its products, the output gate of the block's input, the sinks, and the norm of its output (`norm_out`, where the layer has it).
    pub const HY_MLA: Roles = &[
        ("norm_in", "input_layernorm.weight"),
        ("q_a", "self_attn.q_a_proj.weight"),
        ("q_a_norm", "self_attn.q_a_layernorm.weight"),
        ("q_b", "self_attn.q_b_proj.weight"),
        ("kv_a", "self_attn.kv_a_proj_with_mqa.weight"),
        ("kv_a_norm", "self_attn.kv_a_layernorm.weight"),
        ("kv_b", "self_attn.kv_b_proj.weight"),
        ("g", "self_attn.linear_gate.weight"),
        ("o", "self_attn.o_proj.weight"),
        ("sinks", "self_attn.learnable_sink_param"),
        ("norm_out", "self_attn.post_proj_norm.weight"),
    ];
    /// The compressed indexer (`IndexForm::Grouped`): a token's key and its norm, a group's value and gate products of the key, its position term and
    /// its norm, the queries' product of the query latent, the heads' weights.
    pub const HY_INDEXER: Roles = &[
        ("indexer_wk", "self_attn.compressed_indexer.token_key_projection.weight"),
        ("indexer_k_norm", "self_attn.compressed_indexer.token_key_norm.weight"),
        ("indexer_pool_value", "self_attn.compressed_indexer.key_value_projection.weight"),
        ("indexer_pool_gate", "self_attn.compressed_indexer.key_gate_projection.weight"),
        ("indexer_pool_ape", "self_attn.compressed_indexer.intra_group_ape"),
        ("indexer_pool_norm", "self_attn.compressed_indexer.key_norm.weight"),
        ("indexer_wq_b", "self_attn.compressed_indexer.query_projection.weight"),
        ("indexer_weights", "self_attn.compressed_indexer.head_weight_projection.weight"),
    ];
    /// The streams that do not mix, around a block (`StreamsP::apart`).
    pub const HY_HC_ATTN: Roles =
        &[("hc_fn", "hc_attn_layer.hc_pre.hc_fn"), ("hc_base", "hc_attn_layer.hc_pre.hc_base"), ("hc_scale", "hc_attn_layer.hc_pre.hc_scale")];
    pub const HY_HC_FFN: Roles =
        &[("hc_fn", "hc_mlp_layer.hc_pre.hc_fn"), ("hc_base", "hc_mlp_layer.hc_pre.hc_base"), ("hc_scale", "hc_mlp_layer.hc_pre.hc_scale")];
    /// The dense MLP with the norm of its output.
    pub const HY_MLP: Roles = &[
        ("norm_in", "post_attention_layernorm.weight"),
        ("gate", "mlp.gate_proj.weight"),
        ("up", "mlp.up_proj.weight"),
        ("down", "mlp.down_proj.weight"),
        ("norm_out", "mlp.post_fc2_norm.weight"),
    ];
    /// The experts in a latent: the router and its choice bias, the latent's two projections, the shared expert on the full vector, a layer's routed
    /// experts stacked in two tensors (`experts_gate_up` [experts, 2 inter, latent], the gate's rows first; `experts_down` [experts, latent, inter]),
    /// and the norm of the sum.
    pub const HY_MOE: Roles = &[
        ("norm_in", "post_attention_layernorm.weight"),
        ("router", "mlp.gate.weight"),
        ("choice_bias", "mlp.gate.e_score_correction_bias"),
        ("latent_down", "mlp.routed_expert_down_proj.weight"),
        ("latent_up", "mlp.routed_expert_up_proj.weight"),
        ("shared_gate", "mlp.shared_experts.gate_proj.weight"),
        ("shared_up", "mlp.shared_experts.up_proj.weight"),
        ("shared_down", "mlp.shared_experts.down_proj.weight"),
        ("experts_gate_up", "mlp.experts.gate_up_proj"),
        ("experts_down", "mlp.experts.down_proj"),
        ("norm_out", "mlp.combine_output_norm.weight"),
    ];
    /// The n-gram memory (`NgramForm::EngramNormed`): its tables as one tensor, the compressed vocabulary, the value's weight. The roles that are a
    /// tensor a table or a stream are the patterns below, `{S}` the table's or the stream's number (`hy_memory`).
    pub const HY_MEMORY: Roles = &[
        ("ngram", "engram_layer.multi_head_embedding.embedding.weight"),
        ("ngram_map", "engram_layer.lookup_table"),
        ("value_gamma", "engram_layer.unified_pre_gamma"),
    ];
    /// (role, a tensor a table) and (role, a tensor a stream, the tensor after the streams'): the rows' norms; the streams' norm, the keys' norm and the
    /// streams' scales; the keys' products then the value's, and their biases.
    pub const HY_MEMORY_TABLES: Roles = &[("ngram_norm", "engram_layer.lookup_per_head_norms.{S}.weight")];
    pub const HY_MEMORY_STREAMS: &[(&str, &str, &str)] = &[
        ("norm_in", "engram_layer.norm_query.{S}.weight", ""),
        ("norm_key", "engram_layer.norm_key.{S}.weight", ""),
        ("stream_scale", "engram_layer.unified_per_hc_scales.{S}", ""),
        ("kv", "engram_layer.key_projs.{S}.weight", "engram_layer.value_proj.weight"),
        ("kv_bias", "engram_layer.key_projs.{S}.bias", "engram_layer.value_proj.bias"),
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

/// The roles of `table` that `keep` keeps, named under the path `at` (`experts` of them where a role is each expert's). A pattern says its source's
/// form: `{E}`, a tensor an expert; `a|b`, tensors stacked by rows (vectors joined); `^a`, a matrix stored transposed; else one tensor.
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
            } else if let Some(turned) = pat.strip_prefix('^') {
                Source::Transposed(name(turned))
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

/// Hunyuan's n-gram memory's roles under the layer's path `at`, for `tables` tables and `streams` streams (`part::HY_MEMORY` and its two patterns): a
/// role that is a tensor a table or a stream is those joined in order, the value's after the streams' where the role has one.
pub fn hy_memory(at: &str, tables: usize, streams: usize) -> Vec<(String, Source)> {
    let mut r = roles(part::HY_MEMORY, at, 0, all);
    r.extend(part::HY_MEMORY_TABLES.iter().map(|&(role, pat)| (role.to_string(), shards(pat, at, tables))));
    for &(role, pat, last) in part::HY_MEMORY_STREAMS {
        let Source::Rows(mut v) = shards(pat, at, streams) else { unreachable!("shards are rows") };
        if !last.is_empty() {
            v.push(format!("{at}.{last}"));
        }
        r.push((role.to_string(), Source::Rows(v)));
    }
    r
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
