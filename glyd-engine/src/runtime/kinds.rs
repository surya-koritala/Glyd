//! Layer kinds as data (`docs/design/runtime.md` §3). A model description is a list of blocks; a block is a pre-norm, a body (a mixer
//! or a feed-forward), an optional post-norm and a residual rule. Every family-specific part is a kind with parameters, not a pipeline.
//!
//! Weights are named by their role in the kind (`q`, `k`, `o`, `conv`, ...), not by the checkpoint's names; the description builder
//! (R3a) maps one to the other. Each kind's roles and shapes are listed on its type. A product role `x` may also be given as a
//! low-rank pair `x_a` [rank, in] and `x_b` [out, rank], applied as `x_b (x_a v)`.

/// Where a value is rounded to bf16 inside a kind, for references that hold bf16 tensors between operations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rounding {
    /// Everything in f32.
    F32,
    /// The kind's result rounded once.
    Once,
    /// RMSNorm: the normed value rounded, then times the weight and rounded again (KimiRMSNorm, `research/kimi-k3.md` §2.1). Residual streams
    /// (`StreamsP`): transformers' bf16 tensors between operations, the collapse's norm so too.
    Twice,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NormKind {
    /// x / sqrt(mean(x^2) + eps)
    Rms,
    /// (x - mean) / sqrt(var + eps), plus a bias (role `bias`) when the weight is `Plain`
    Layer,
    /// x / sqrt(sum(x^2) + eps)
    L2,
    /// No norm: the value itself (a branch that reads the residual as it is, as the per-layer input's does).
    Identity,
}

/// How a norm's weight (role `w`, [dim]) scales the normed value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NormWeight {
    /// No weight (Gemma 4's v norm, Muse's q/k norm).
    None,
    /// y = norm(x) w
    Plain,
    /// y = norm(x) (1 + w): Qwen3.5/3.8, Muse's centred norm.
    PlusOne,
    /// y = norm(x (1 + w)): the weight scales the input (Muse's post-expert and post-feed-forward norms).
    InputPlusOne,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Norm {
    pub kind: NormKind,
    pub weight: NormWeight,
    pub eps: f32,
    pub rounding: Rounding,
}

impl Norm {
    pub const fn rms(eps: f32) -> Norm {
        Norm { kind: NormKind::Rms, weight: NormWeight::Plain, eps, rounding: Rounding::F32 }
    }
    pub const fn rms_plus_one(eps: f32) -> Norm {
        Norm { kind: NormKind::Rms, weight: NormWeight::PlusOne, eps, rounding: Rounding::F32 }
    }
    pub const fn rms_bare(eps: f32) -> Norm {
        Norm { kind: NormKind::Rms, weight: NormWeight::None, eps, rounding: Rounding::F32 }
    }
}

/// How the rotated dimensions are paired.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RopeStyle {
    /// (i, i + dim/2): transformers' `rotate_half` (Llama, Qwen, Gemma, Muse).
    Half,
    /// (2i, 2i + 1): GPT-J, DeepSeek-V4's rotary.
    Interleaved,
}

/// Rotary scaling (transformers' `rope_type`).
#[derive(Clone, Debug, PartialEq)]
pub enum RopeScaling {
    None,
    Linear {
        factor: f64,
    },
    DynamicNtk {
        factor: f64,
        max_positions: u32,
    },
    Yarn {
        factor: f64,
        original_max: u32,
        beta_fast: f64,
        beta_slow: f64,
        mscale: Option<f64>,
        mscale_all_dim: Option<f64>,
        attention_factor: Option<f64>,
        truncate: bool,
    },
    Llama3 {
        factor: f64,
        low_freq_factor: f64,
        high_freq_factor: f64,
        original_max: u32,
    },
    LongRope {
        short: Vec<f64>,
        long: Vec<f64>,
        original_max: u32,
        max_positions: u32,
        attention_factor: Option<f64>,
    },
}

impl RopeScaling {
    /// transformers' `rope_type` of the scaling ("default" for none).
    pub fn name(&self) -> &'static str {
        match self {
            RopeScaling::None => "default",
            RopeScaling::Linear { .. } => "linear",
            RopeScaling::DynamicNtk { .. } => "dynamic",
            RopeScaling::Yarn { .. } => "yarn",
            RopeScaling::Llama3 { .. } => "llama3",
            RopeScaling::LongRope { .. } => "longrope",
        }
    }
}

/// Rotary position embedding on the first `dim` dimensions of a head (the rest pass through).
///
/// `dim` is the rotated span; `freqs` of its `dim / 2` angles have a frequency (the rest are 0, which leaves their pairs as they are:
/// Gemma 4's proportional rope on global layers rotates the whole head with a quarter of the angles); the exponent's denominator is
/// `freq_dim` (`dim` for plain and partial rope, the head dim for proportional rope).
#[derive(Clone, Debug, PartialEq)]
pub struct Rope {
    pub theta: f64,
    pub dim: usize,
    pub freqs: usize,
    pub freq_dim: usize,
    pub style: RopeStyle,
    pub scaling: RopeScaling,
    /// M-RoPE: the angles split into sections for temporal, height and width positions (Qwen3.8 vision); text positions use one. The sections
    /// interleave (`mrope_axis`): a config whose sections follow one another (no `mrope_interleaved`) is refused where its positions differ,
    /// with a vision tower.
    pub mrope: Option<[usize; 3]>,
}

impl Rope {
    /// The position an M-RoPE angle `i` turns by: 0 the temporal one, 1 the row's, 2 the column's. The sections interleave as transformers'
    /// `apply_interleaved_mrope` lays them (Qwen3-VL's, Qwen3.5's and Qwen 4's `mrope_interleaved`): angle i takes the row where i mod 3 is 1 and
    /// i < 3 sections[1], the column where i mod 3 is 2 and i < 3 sections[2], else the temporal position. 0 for every angle of a rope with no sections.
    pub fn mrope_axis(&self, i: usize) -> usize {
        match self.mrope {
            Some([_, h, _]) if i % 3 == 1 && i < 3 * h => 1,
            Some([_, _, w]) if i % 3 == 2 && i < 3 * w => 2,
            _ => 0,
        }
    }

    /// The transformers `rope_type` this rope is where it is not the plain rotary of its `dim` (frequencies over the rotated span, no scaling): "proportional" where the
    /// exponents' denominator is the head's, else the scaling's; None for the plain rotary (the one a layer kind's tables hold).
    pub fn not_plain(&self) -> Option<&'static str> {
        if self.freq_dim != self.dim {
            Some("proportional")
        } else if self.scaling != RopeScaling::None {
            Some(self.scaling.name())
        } else {
            None
        }
    }

    /// Plain rope over `dim` dimensions of each head (partial when `dim` is less than the head dim).
    pub fn plain(theta: f64, dim: usize) -> Rope {
        Rope { theta, dim, freqs: dim / 2, freq_dim: dim, style: RopeStyle::Half, scaling: RopeScaling::None, mrope: None }
    }
    /// transformers' `proportional` rope: the whole head rotates; `fraction` of its angles have a frequency, with exponents over the head dim.
    pub fn proportional(theta: f64, head_dim: usize, fraction: f64) -> Rope {
        Rope {
            theta,
            dim: head_dim,
            freqs: (fraction * head_dim as f64 / 2.0) as usize,
            freq_dim: head_dim,
            style: RopeStyle::Half,
            scaling: RopeScaling::None,
            mrope: None,
        }
    }
}

/// The activation of an output gate or a gated MLP.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Act {
    Sigmoid,
    Silu,
    GeluTanh,
    Gelu,
    Relu2,
    /// Kimi-K3's situ, a gated pair's act(g, u) = beta tanh(g / beta) sigmoid(g) linear tanh(u / linear); as a single activation, its
    /// gate factor.
    Situ {
        beta: f32,
        linear: f32,
    },
    /// The SwiGLU of gpt-oss, the gate's sigmoid sharpened and the up shifted: a gated pair's act(g, u) = g sigmoid(alpha g) (u + 1); as a single
    /// activation, its gate factor g sigmoid(alpha g) (alpha 1: SiLU). With an MLP's `Clamp::Gate(l)` the gate is min(g, l) and the up clamp(u, -l, l)
    /// before both, so the shifted up is within 1 - l and 1 + l (MiniMax M3's `hidden_act` "swigluoai" with `swiglu_alpha` and `swiglu_limit`). In `Ref`.
    SwigluOai {
        alpha: f32,
    },
}

impl Act {
    /// The refusal of the gpt-oss SwiGLU: the CPU reference computes it (`reference::gated`), the executor's activations have no sharpened sigmoid and no
    /// shifted up.
    pub const SWIGLU_OAI_NOT_RUN: &'static str =
        "the SwiGLU whose gate is g sigmoid(alpha g) and whose up is shifted by 1 (swigluoai), which the executor does not run yet";

    /// The name transformers' configs give the activation (`hidden_act`, `output_gate_type`).
    pub fn name(&self) -> &'static str {
        match self {
            Act::Sigmoid => "sigmoid",
            Act::Silu => "silu",
            Act::GeluTanh => "gelu_pytorch_tanh",
            Act::Gelu => "gelu",
            Act::Relu2 => "relu2",
            Act::Situ { .. } => "situ",
            Act::SwigluOai { .. } => "swigluoai",
        }
    }
}

/// Where an attention output gate's values come from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GateFrom {
    /// Its own product, role `g` [heads * v_head_dim, hidden] (Muse, Kimi-K3's MLA).
    Own,
    /// Interleaved with the queries in `q`: per head, head_dim query rows then head_dim gate rows (Qwen3.5/3.8 `q_proj`).
    InQuery,
    /// Its own product of one value a head, role `g` [heads, hidden]: every channel of a head's output times the head's one gate (LFM's
    /// `self_attn.gate_proj` on plain attention, Ling's `attention.g_proj` on the latent attention: its "head_wise" gate).
    Head,
}

/// An attention output gate: `o = norm(o_head) * act(gate)` per head before the output product.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OutGate {
    pub act: Act,
    pub from: GateFrom,
    /// A per-head norm of the output before the gate (Muse: weightless RMS).
    pub norm: Option<Norm>,
}

/// A sparse-attention indexer: per query, a score sum_h w_h relu(q_h . k) of each past key (one key a token, a pooled key a block of
/// `block` tokens, or a compressed entry), the best kept. Ibis's block indexer, GLM's pooled-key indexer and DeepSeek-V4's
/// compressed-entry indexer are this kind with parameters. In `Ref`: a latent attention's (`reference::mla`, GLM-5.3-Flash),
/// DeepSeek-V4's (`reference::comp`) and Ibis's on a GQA attention (`reference::attention`, `IndexForm::Mean`).
#[derive(Clone, PartialEq)]
pub struct Indexer {
    pub heads: usize,
    pub head_dim: usize,
    /// What a query keeps. A latent attention: `top_k` tokens, as the `top_k / block` best complete pools of `block` tokens and the tokens
    /// of the incomplete last pool (GLM-5.3-Flash's `index_topk`; `block` 1: the best `top_k` tokens; Ibis's `indexer_budget` in pools of
    /// `indexer_compress_ratio`). DeepSeek-V4: `top_k` compressed entries (`block` is its ratio).
    pub top_k: usize,
    /// Tokens a pooled key covers (GLM-5.3-Flash's `index_kpool`; 1: a key per token).
    pub block: usize,
    /// The layer reuses the previous indexer layer's choice and has no indexer weights (GLM's `indexer_types` "shared").
    pub shared: bool,
    pub form: IndexForm,
    /// The gated form's rotary: the first `rope` channels of each query head and of each token's key rotated to their positions by the attention's rotary
    /// (the full-size GLM's indexer, its `qk_rope_head_dim`); 0: none (GLM-5.3-Flash, Kimi-K3.1, DeepSeek-V4). The mean form's
    /// queries and pools are rotated as `IndexForm::Mean` says, whatever this is.
    pub rope: usize,
    /// The gated form's queries from the block's input and not from the attention's query latent: the input through this norm (roles `indexer_q_norm`
    /// [hidden], and a LayerNorm's bias `indexer_q_norm_bias`), then `indexer_wq_b` [heads head_dim, hidden] (Mistral Large's, whose attention has no query
    /// latent). None: the query latent's (GLM's, Kimi's).
    pub q_in: Option<Norm>,
    /// The gated form's queries scaled by their position before their rotary: head h's values times `indexer_scale_s`[h] ln(pos + 1) + `indexer_scale_b`[h]
    /// (roles [heads]; pos the query's position, the first token's 0), as `Mla::pos_scale` scales the attention's (Mistral Large's).
    pub pos_scale: bool,
}

/// The derived form, with `form` said only where it is not the gated form with LayerNorm-ed keys (`IndexForm::LAYER_KEYS`), `rope` only where it
/// rotates, `q_in` only where the queries are the block's input's and `pos_scale` only where they are scaled: those indexers' descriptions read as they
/// did before the fields (the digests of tests/fixtures/descriptions.json are of this text).
impl std::fmt::Debug for Indexer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut s = f.debug_struct("Indexer");
        s.field("heads", &self.heads).field("head_dim", &self.head_dim).field("top_k", &self.top_k).field("block", &self.block).field("shared", &self.shared);
        if self.form != IndexForm::LAYER_KEYS {
            s.field("form", &self.form);
        }
        if self.rope != 0 {
            s.field("rope", &self.rope);
        }
        if let Some(n) = &self.q_in {
            s.field("q_in", n);
        }
        if self.pos_scale {
            s.field("pos_scale", &self.pos_scale);
        }
        s.finish()
    }
}

/// How an indexer makes its queries, its pooled keys and its scores.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum IndexForm {
    /// Queries and a key a token from their own products, each token's key normed by `key` (role `indexer_k_norm`, and with a LayerNorm its
    /// bias `indexer_k_norm_bias`); a pool's key the softmax(gate + ape)-weighted sum of its tokens' keys; the heads' scores weighed by a learned
    /// product of the block's input. The key's norm: a LayerNorm at 1e-6 (GLM-5.3-Flash and the full-size GLM: `LAYER_KEYS`), an RMS norm at the
    /// model's eps (Kimi-K3.1's `dsa_config.index_k_norm_type` rmsnorm, its LayerNorm otherwise; DeepSeek-V4's compressed entries, normed by
    /// their compressor's `idx_norm`).
    Gated { key: Norm },
    /// Ibis's Qwen Sparse Attention (Qwen3.8-Flash-Next): the `heads` queries and ONE key from one product of the block's input (role
    /// `index_qk` [(heads + 1) head_dim, hidden]); each query head normed by `norm` (role `index_q_norm` [head_dim]) and rotated as the
    /// attention's queries (its rope, on the first `dim` channels of the head); the raw keys cached, a pool's key the mean of its tokens'
    /// raw keys, normed by `norm` (role `index_k_norm`) and rotated at the pool's first position; a pool's score sum_h relu(q_h . k) /
    /// sqrt(head_dim), the heads unweighted.
    Mean { norm: Norm },
}

impl IndexForm {
    /// The gated form whose keys are LayerNorm-ed with a bias at 1e-6 (GLM-5.3-Flash's and the full-size GLM's).
    pub const LAYER_KEYS: IndexForm = IndexForm::Gated { key: Norm { kind: NormKind::Layer, weight: NormWeight::Plain, eps: 1e-6, rounding: Rounding::F32 } };
}

/// Where an attention rounds to bf16 inside: the arithmetic of the implementation a family's engine follows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttnRounding {
    /// Everything in f32 (transformers in float32).
    F32,
    /// FlashAttention's: each product's output, the normed and rotated queries and keys and the values bf16 (as a bf16 cache holds them),
    /// the scores and their softmax f32, the probabilities rounded to bf16 before they weigh the values, the output rounded, and again after
    /// its norm and its gate (vLLM's; the Qwen3.5 layout, Nemotron-H).
    Flash,
    /// transformers' eager attention in bf16: as `Flash`, the rope in bf16 arithmetic (cos and sin bf16, each product and their sum
    /// rounded), and each score the bf16 of its dot product, times the scale and rounded again (and so through a soft-cap) (Gemma 4).
    Eager,
    /// transformers in bf16 around an attention whose scores and softmax are f32 (SDPA): the queries times the scale before the rope,
    /// rounded; the rope in bf16 arithmetic; the probabilities f32; the output rounded, its norm rounded, the gate's activation rounded and
    /// its product rounded (Muse Spark).
    Sdpa,
}

/// MHA, GQA, MQA attention.
///
/// Roles: `q` [heads * head_dim (x2 with an `InQuery` gate), hidden], `k` [kv_heads * head_dim, hidden], `v` [kv_heads * v_head_dim,
/// hidden] (absent with `k_eq_v`), `o` [hidden, heads * v_head_dim]; `q_bias`, `k_bias`, `v_bias` with `bias`; `q_norm`, `k_norm`
/// [head_dim] where those norms have a weight; `g` with an `Own` gate ([heads, hidden] with a `Head` one); `sinks` [heads] with `sinks`. With
/// `kv_from` no `k`, `v`, `k_norm`.
#[derive(Clone, PartialEq)]
pub struct Attn {
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub v_head_dim: usize,
    /// Sliding window: a query at position p attends to positions p - window + 1 ..= p.
    pub window: Option<usize>,
    /// A learned logit per head that takes part in the softmax and attends to nothing (gpt-oss, DeepSeek-V4).
    pub sinks: bool,
    /// None: no position embedding (NoPE).
    pub rope: Option<Rope>,
    pub q_norm: Option<Norm>,
    pub k_norm: Option<Norm>,
    pub v_norm: Option<Norm>,
    pub gate: Option<OutGate>,
    /// Scores soft-capped as cap * tanh(s / cap).
    pub softcap: Option<f32>,
    pub bias: bool,
    /// The scores' factor (1 / sqrt(head_dim) usually; Gemma 4: 1; Muse: its qk scale / sqrt(head_dim)).
    pub scale: f32,
    /// Gemma 4's global layers: the values are the keys' projection (before the k norm and rope), through the v norm.
    pub k_eq_v: bool,
    pub indexer: Option<Indexer>,
    pub rounding: AttnRounding,
    /// KV sharing (Gemma 4's `num_kv_shared_layers`): the queries attend to the keys and values that block `kv_from`'s attention (an earlier
    /// block of the same heads, window and rope) holds for the sequence; this attention has no keys or values of its own and no cache.
    pub kv_from: Option<usize>,
    /// The values' factor: each value times it after its product, before it is cached and weighed (MiMo V2's `attention_value_scale`); 1 elsewhere.
    pub v_scale: f32,
}

/// The derived form, with `kv_from` said only where it is set and `v_scale` only where it is not 1: the other attentions' descriptions read as
/// they did before the fields (the digests of tests/fixtures/descriptions.json are of this text).
impl std::fmt::Debug for Attn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut s = f.debug_struct("Attn");
        s.field("heads", &self.heads).field("kv_heads", &self.kv_heads).field("head_dim", &self.head_dim).field("v_head_dim", &self.v_head_dim);
        s.field("window", &self.window).field("sinks", &self.sinks).field("rope", &self.rope).field("q_norm", &self.q_norm).field("k_norm", &self.k_norm);
        s.field("v_norm", &self.v_norm).field("gate", &self.gate).field("softcap", &self.softcap).field("bias", &self.bias).field("scale", &self.scale);
        s.field("k_eq_v", &self.k_eq_v).field("indexer", &self.indexer).field("rounding", &self.rounding);
        if let Some(j) = self.kv_from {
            s.field("kv_from", &j);
        }
        if self.v_scale != 1.0 {
            s.field("v_scale", &self.v_scale);
        }
        s.finish()
    }
}

/// Multi-head latent attention (DeepSeek-V2/V3 style, Kimi-K3, GLM-5.3).
///
/// Roles: `q` [heads * (nope + rope_dim), hidden] or with `q_lora` `q_a` [q_lora, hidden], `q_a_norm` [q_lora], `q_b` [heads * (nope +
/// rope_dim), q_lora]; `kv_a` [kv_lora + rope_dim, hidden], `kv_a_norm` [kv_lora], `kv_b` [heads * (nope + v_dim), kv_lora], `o` [hidden,
/// heads * v_dim]; `g` [heads * v_dim, hidden] with a gate ([heads, hidden] with a `Head` one). The cache holds per token the normed latent and the
/// shared key part (kv_lora + rope_dim values); `absorbed` names how a backend computes it (the same function).
///
/// Mistral Large's forms of it: `pos_scale` (`q_scale_s`, `q_scale_b` [heads]), `kv_from` (no `kv_a`, no `kv_a_norm`, no cache of latents of its own) and
/// `rope_learned` (`rope_alpha` [rope_dim / 2]).
#[derive(Clone, PartialEq)]
pub struct Mla {
    pub heads: usize,
    pub q_lora: Option<usize>,
    pub kv_lora: usize,
    pub nope: usize,
    pub rope_dim: usize,
    pub v_dim: usize,
    /// None: the rope part is not rotated (Kimi's NoPE MLA).
    pub rope: Option<Rope>,
    pub q_norm: Norm,
    pub kv_norm: Norm,
    pub gate: Option<OutGate>,
    pub scale: f32,
    pub absorbed: bool,
    pub indexer: Option<Indexer>,
    /// The queries scaled by their position ("scalable softmax"): each query head h, after its rotary, times `q_scale_s`[h] ln(pos + 1) + `q_scale_b`[h],
    /// pos the query's own position (the first token's 0, whose factor is the bias alone); the keys are not scaled, so every score of the query, and the
    /// sharpness of its softmax, moves with the logarithm of where the query stands, a factor a head. No state: a prompt's rows and a step's are the same.
    pub pos_scale: bool,
    /// Latent sharing (Mistral Large's `kv_source_layers`): the queries attend to the latents block `kv_from`'s latent attention (an earlier block of the
    /// same latent width, shared key part and rotary) holds for the sequence, each token's normed latent and its shared key part as that block made and
    /// turned them; this attention expands them by its OWN `kv_b`, has no `kv_a` and no `kv_a_norm`, and caches no latents. An indexer of its own
    /// keeps its own keys.
    pub kv_from: Option<usize>,
    /// Learned rotary frequencies (Mistral Large's LeRoPE): angle i's inverse frequency, as the rotary's scaling makes it, times exp(`rope_alpha`[i])
    /// (role [rope_dim / 2], one tensor of the model that every rotating layer reads; the indexer's rotary is the attention's and turns by them too).
    pub rope_learned: bool,
}

/// The derived form, with `pos_scale`, `kv_from` and `rope_learned` said only where they are set: the other latent attentions' descriptions read as they
/// did before the fields (the digests of tests/fixtures/descriptions.json are of this text).
impl std::fmt::Debug for Mla {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut s = f.debug_struct("Mla");
        s.field("heads", &self.heads).field("q_lora", &self.q_lora).field("kv_lora", &self.kv_lora).field("nope", &self.nope);
        s.field("rope_dim", &self.rope_dim).field("v_dim", &self.v_dim).field("rope", &self.rope).field("q_norm", &self.q_norm);
        s.field("kv_norm", &self.kv_norm).field("gate", &self.gate).field("scale", &self.scale).field("absorbed", &self.absorbed);
        s.field("indexer", &self.indexer);
        if self.pos_scale {
            s.field("pos_scale", &self.pos_scale);
        }
        if let Some(j) = self.kv_from {
            s.field("kv_from", &j);
        }
        if self.rope_learned {
            s.field("rope_learned", &self.rope_learned);
        }
        s.finish()
    }
}

impl Mla {
    /// What of a latent attention the executor has no operation for yet (the CPU reference computes each), in words; None where it has none of them.
    pub fn not_run(&self) -> Option<&'static str> {
        if self.pos_scale {
            Some(Mla::POS_SCALE_NOT_RUN)
        } else if self.kv_from.is_some() {
            Some(Mla::KV_FROM_NOT_RUN)
        } else if self.rope_learned {
            Some(Mla::ROPE_LEARNED_NOT_RUN)
        } else if self.indexer.as_ref().is_some_and(|x| x.q_in.is_some() || x.pos_scale) {
            Some(Mla::INDEX_INPUT_NOT_RUN)
        } else {
            None
        }
    }

    pub const POS_SCALE_NOT_RUN: &'static str = "a latent attention whose queries are scaled by their position, which the executor does not run yet";
    pub const KV_FROM_NOT_RUN: &'static str = "a latent attention that attends to another layer's latents, which the executor does not run yet";
    pub const ROPE_LEARNED_NOT_RUN: &'static str = "a latent attention whose rotary's frequencies are learned, which the executor does not run yet";
    pub const INDEX_INPUT_NOT_RUN: &'static str =
        "a latent attention's indexer whose queries are the block's input's, scaled by their position, which the executor does not run yet";
}

/// DeepSeek-V4's attention: one KV head of `head_dim` whose keys are its values (K = V), `heads` query heads, rope on the trailing
/// `rope_dim` of each head (interleaved pairs), learned sinks, a sliding window, and on compressing layers the compressed entries of every
/// closed window of `ratio` tokens: each entry the per-channel softmax-weighted sum of its tokens' projections (`overlap`: the previous
/// window's tokens' first half of the channels with this window's second half, over 2 `ratio` tokens), normed and rotated at the window's
/// first position. With an indexer a query sees only the `top_k` entries it scores highest. The output is rotated back at the query's
/// position, then goes through `o_groups` grouped low-rank products (`o_a`) and `o_b`.
///
/// Roles: `q_a` [q_lora, hidden], `q_a_norm` [q_lora], `q_b` [heads head_dim, q_lora], `kv` [head_dim, hidden], `kv_norm` [head_dim],
/// `o_a` [o_groups o_lora, heads head_dim / o_groups], `o_b` [hidden, o_groups o_lora], `sinks` [heads]; with a ratio `comp_kv`,
/// `comp_gate` [(1 + overlap) head_dim, hidden], `comp_ape` [ratio, (1 + overlap) head_dim], `comp_norm` [head_dim]; with an indexer
/// the same four as `idx_kv`, `idx_gate`, `idx_ape`, `idx_norm` at the indexer's head dim, `idx_q_b` [heads head_dim, q_lora] and
/// `idx_weights` [heads, hidden]. The query's per-head norm has no weight.
///
/// DeepSeek-V4.1's form of it (CSA2): ratio 1 (each token its own entry, no gate) or 2 (windows of 2, no overlap), no position bias (`ape`
/// false: no `comp_ape`), no per-head query norm (`q_head_norm` false), and the entries, the indexer's keys and its picks shared across
/// layers (`share`). A query reads entry j only once its window has closed: j < (pos + 1) / ratio.
#[derive(Clone, PartialEq)]
pub struct Comp {
    pub heads: usize,
    pub head_dim: usize,
    pub q_lora: usize,
    pub o_groups: usize,
    pub o_lora: usize,
    pub window: usize,
    /// 0: the window alone.
    pub ratio: usize,
    pub overlap: bool,
    /// Rope of the window's queries and keys (DeepSeek-V4's "main" on window-only layers, "compress" with YaRN on compressing ones),
    /// over the trailing `dim` of each head; the compressed entries and the indexer use it too.
    pub rope: Rope,
    pub indexer: Option<Indexer>,
    pub norm_eps: f32,
    /// The compressed windows' learned position bias (`comp_ape`, `idx_ape`): DeepSeek-V4's; V4.1's have none.
    pub ape: bool,
    /// The query's per-head weightless RMS norm: DeepSeek-V4's; V4.1 has none.
    pub q_head_norm: bool,
    /// V4.1's sharing across layers; None: every part the layer's own (V4).
    pub share: Option<CompShare>,
}

/// Where a V4.1 compressed attention's parts come from: None, the layer's own; Some(b), block b's of the description (the most recent source
/// at or below the layer).
///
/// The entries: a kv source's compressor (`comp_kv`, `comp_gate` with ratio 2, `comp_norm`: `kv_source_layer_ids`), each entry normed, then
/// rotated at its window's first position; and, where the source has the indexer's key product, the indexer's keys: `idx_k` [index head_dim,
/// head_dim] of each entry before its rotation, normed by the indexer's key norm (`idx_k_norm`), rotated as the entry. The picks: an index
/// source's (`index_source_layer_ids`), whose queries are its own (`idx_q_b` of the q latent, rotated at the query's position; `idx_weights`):
/// score(j) = sum_h w_h relu(q_h . k_j) / sqrt(index head_dim) / sqrt(heads) over the visible entries, the best `top_k` kept in order; a layer
/// that is no index source reads that source's picks of the same position. `candidates`: the candidate source's block, inside whose published
/// blocks an index source after it scores (the rest masked before its top-k).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CompShare {
    pub entries: Option<usize>,
    pub picks: Option<usize>,
    pub candidates: Option<usize>,
    /// On the candidate source (`candidate_source_layer_id`): each query's visible scores max-pooled over blocks of `block` entries, the block
    /// of its newest visible entry kept whatever its score, the best `top` blocks published (`candidate_block_size`, `candidate_topk_blocks`).
    pub publish: Option<(usize, usize)>,
}

/// The derived form, with V4.1's fields said only where they differ from DeepSeek-V4's: V4's descriptions read as they did before them (the
/// digests of tests/fixtures/descriptions.json are of this text).
impl std::fmt::Debug for Comp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut s = f.debug_struct("Comp");
        s.field("heads", &self.heads)
            .field("head_dim", &self.head_dim)
            .field("q_lora", &self.q_lora)
            .field("o_groups", &self.o_groups)
            .field("o_lora", &self.o_lora)
            .field("window", &self.window)
            .field("ratio", &self.ratio)
            .field("overlap", &self.overlap)
            .field("rope", &self.rope)
            .field("indexer", &self.indexer)
            .field("norm_eps", &self.norm_eps);
        if !self.ape || !self.q_head_norm || self.share.is_some() {
            s.field("ape", &self.ape).field("q_head_norm", &self.q_head_norm).field("share", &self.share);
        }
        s.finish()
    }
}

/// Which delta rule a linear-attention layer runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LinearKind {
    /// Gated DeltaNet (Qwen3.5/3.8, Ibis): one decay per value head.
    GatedDeltaNet,
    /// Kimi delta attention (Kimi-K3, GLM-5.3): one decay per key channel.
    KimiDelta,
}

/// The decay's form, g = log of the factor the state is multiplied by each token.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Decay {
    /// g = -exp(A_log) softplus(a + dt_bias)
    Softplus,
    /// g = bound sigmoid(exp(A_log) (a + dt_bias)), bound < 0 (Kimi-K3's and GLM-5.3's `gate_lower_bound`)
    Bounded(f32),
}

/// Gated DeltaNet and Kimi delta attention.
///
/// Roles: `qkv` [2 k_heads k_dim + v_heads v_dim, hidden] (queries, keys, values; or `q`, `k`, `v` apart), `conv` [that width, conv]
/// (each channel's taps, oldest first), `beta` [v_heads, hidden], `decay` [v_heads (GDN) or v_heads k_dim (KDA), hidden] (often low-rank),
/// `a_log` [v_heads], `dt_bias` [v_heads or v_heads k_dim], `gate` [v_heads v_dim, hidden] (often low-rank), `norm` [v_dim], `o`
/// [hidden, v_heads v_dim]. The state is [v_heads, k_dim, v_dim] per sequence, plus the convolution's last `conv - 1` inputs.
#[derive(Clone, Debug, PartialEq)]
pub struct Linear {
    pub kind: LinearKind,
    pub k_heads: usize,
    pub v_heads: usize,
    pub k_dim: usize,
    pub v_dim: usize,
    pub conv: usize,
    pub decay: Decay,
    /// The output's gate after the per-head norm: SiLU (GDN) or sigmoid (KDA).
    pub gate_act: Act,
    pub norm_eps: f32,
    /// f32 on every family today.
    pub state_f32: bool,
}

/// Mamba-2 (Nemotron-H, Granite 4).
///
/// The inner width is `heads head_dim`; the convolution runs over the scan's inputs and the `groups` groups of B and C, `inner + 2
/// groups state` channels. Roles: `in` [inner + (inner + 2 groups state) + heads, hidden] (the rows of z, then the convolution's
/// channels x, B, C, then dt, a value a head), `conv` [inner + 2 groups state, conv] (each channel's taps, oldest first) and
/// `conv_bias` [that width], `a_log`, `d`, `dt_bias` [heads], `norm` [inner] (the gated norm's weight: RMS over each of `groups`
/// groups of the channels of y silu(z)), `o` [hidden, inner]. The state is [heads, head_dim, state] f32 per sequence, plus the
/// convolution's last `conv - 1` inputs.
#[derive(Clone, Debug, PartialEq)]
pub struct Mamba2 {
    pub heads: usize,
    pub head_dim: usize,
    pub state: usize,
    pub groups: usize,
    pub conv: usize,
    /// The tokens of a chunk of the reference's chunked scan (the same function at any chunk; rounding differs).
    pub chunk: usize,
    /// The gated norm's eps.
    pub norm_eps: f32,
}

/// The gated short convolution (LFM's `conv` layers): one product of the block's input gives three vectors of the hidden width, B, C and x, in that
/// order; u = B x, channel by channel; each channel of u through its own causal convolution of `taps` taps over the sequence (a token reads its own u
/// and the `taps` - 1 before it, zeros before the sequence's first; no activation); y = C conv(u); then the output product. No state but the
/// convolution's.
///
/// Roles: `in` [3 hidden, hidden] (the rows of B, then C, then x), `conv` [hidden, taps] (each channel's taps, oldest first), `o` [hidden, hidden]; with
/// `bias`, `in_bias` [3 hidden], `conv_bias` [hidden] and `o_bias` [hidden]. The state is the last `taps` - 1 values of u (the gated product, not the
/// input) a channel, per sequence.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ShortConv {
    pub taps: usize,
    pub bias: bool,
}

impl ShortConv {
    /// The values a sequence's state holds in a layer of `hidden` channels: the last `taps` - 1 gated products of each.
    pub fn state(&self, hidden: usize) -> usize {
        hidden * self.taps.saturating_sub(1)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Mixer {
    Attention(Attn),
    Mla(Mla),
    Compressed(Comp),
    Linear(Linear),
    Mamba2(Mamba2),
    ShortConv(ShortConv),
}

/// A dense MLP: `down(act(gate x) * (up x))` when gated, `down(act(up x))` otherwise.
///
/// Roles `gate` and `up` [inter, in] (or `gate_up` [2 inter, in], the gate's rows first), `down` [out, inter], and their `_bias` with
/// `bias`. In a MoE the same roles prefixed `shared_` are the shared expert.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Mlp {
    pub act: Act,
    pub gated: bool,
    pub bias: bool,
    /// SwiGLU limits: the gate held to the limit above and the up both ways (`Clamp`: where the gate is held).
    pub clamp: Option<Clamp>,
}

impl Mlp {
    /// What of an MLP the executor has no operation for yet (the CPU reference computes it), in words; None where it has none of them.
    pub fn not_run(&self) -> Option<&'static str> {
        matches!(self.act, Act::SwigluOai { .. }).then_some(Act::SWIGLU_OAI_NOT_RUN)
    }
}

/// A gated MLP's SwiGLU limit l: the up is clamp(up, -l, l) in both forms; the gate is held to l before its activation or after it, which are two
/// functions (silu(l) is under l).
#[derive(Clone, Copy, PartialEq)]
pub enum Clamp {
    /// act(min(gate, l)): the gate before its activation (DeepSeek-V4's `swiglu_limit`, GLM-5.3-Flash's).
    Gate(f32),
    /// min(act(gate), l): the activated gate (Ling's `expert_swiglu_limit_list` and `share_expert_swiglu_limit_list`).
    Activated(f32),
}

impl Clamp {
    /// The limit, whichever value of the gate it holds.
    pub fn limit(&self) -> f32 {
        match *self {
            Clamp::Gate(l) | Clamp::Activated(l) => l,
        }
    }

    /// The limit of the form the executor's kernels compute (the gate before its activation); the other form is refused by its name.
    pub fn before(c: Option<Clamp>) -> crate::ffi::Result<Option<f32>> {
        match c {
            None => Ok(None),
            Some(Clamp::Gate(l)) => Ok(Some(l)),
            Some(Clamp::Activated(l)) => crate::ffi::err(format!("{}: {l}", Clamp::ACTIVATED_NOT_RUN)),
        }
    }

    /// The refusal of a SwiGLU whose limit holds the activated gate: the CPU reference computes it, the executor's kernels hold the gate before its activation.
    pub const ACTIVATED_NOT_RUN: &'static str = "a SwiGLU limit on the activated gate (min(silu(gate), limit)), which the executor does not run yet";
}

/// The limit alone where the gate is held before its activation: descriptions of that form read as they did when the limit was a number (the digests of
/// tests/fixtures/descriptions.json are of this text).
impl std::fmt::Debug for Clamp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Clamp::Gate(l) => write!(f, "{l:?}"),
            Clamp::Activated(l) => write!(f, "Activated({l:?})"),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RouterScore {
    Softmax,
    Sigmoid,
    /// sqrt(softplus(x)) (DeepSeek-V4)
    SqrtSoftplus,
}

/// How a MoE layer picks a token's experts and weighs them. Roles `router` [experts, hidden]; `choice_bias` [experts] with `bias`;
/// `hash` [vocab, top_k] (expert ids as values) with `hash`.
#[derive(Clone, Debug, PartialEq)]
pub struct Router {
    pub score: RouterScore,
    /// The logits' factor before the score (Muse: its output multiplier).
    pub logit_scale: f32,
    /// A per-expert bias added to the scores for the choice only (sigmoid and sqrt-softplus routers).
    pub bias: bool,
    /// (groups, groups kept): a group's score is the sum of its two best (biased) scores; experts of the other groups are not chosen.
    pub groups: Option<(usize, usize)>,
    pub top_k: usize,
    /// The chosen scores divided by their sum (+ 1e-20).
    pub renorm: bool,
    pub scale: f32,
    /// Hash routing (DeepSeek-V4's first layers): the token id picks the experts from the table, the scores still weigh them.
    pub hash: bool,
}

/// A latent MoE: experts run in a smaller space between two projections. Roles `latent_down` [dim, hidden], `latent_up` [hidden, dim],
/// and the norms' weights `latent_norm_in`, `expert_norm`, `latent_norm_out` [dim].
#[derive(Clone, Debug, PartialEq)]
pub struct Latent {
    pub dim: usize,
    /// On the projected input (Muse's pre-expert norm).
    pub norm_in: Option<Norm>,
    /// On each expert's output before it is weighed (Muse's post-expert norm).
    pub expert_norm: Option<Norm>,
    /// On the weighed sum before the up projection (Kimi-K3).
    pub norm_out: Option<Norm>,
}

/// A mixture of experts. Roles: the router's; `experts_gate_up` [experts, 2 inter, in] (or `experts_up` [experts, inter, in] when the
/// expert MLP is not gated), `experts_down` [experts, out, inter]; the shared expert's `shared_*`; with `shared_gate`, `shared_router`
/// [1, hidden] (the shared output times sigmoid of it: Qwen3.5/3.8 MoE, Ibis).
#[derive(Clone, Debug, PartialEq)]
pub struct Moe {
    pub router: Router,
    pub experts: usize,
    pub expert: Mlp,
    pub shared: Option<Mlp>,
    pub shared_gate: bool,
    pub latent: Option<Latent>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Ffn {
    Mlp(Mlp),
    Moe(Moe),
}

/// A hashed n-gram embedding added to the residual streams (Ibis's per-layer embedding, Qwen3.8-Flash-Next's `ple`).
///
/// Per position, for each order n of 2..=`order`, the token and the n - 1 tokens before it inside its segment (a segment ends at the
/// token `eos`; a token before the segment's start, or before the sequence, reads as `eos`) hashed: mixed = XOR over p < n of
/// (token_p mult_p), the multipliers from splitmix64 of `seed` + 10007 `index` (`index`: the layer's place among the model's n-gram
/// layers; `vocab` bounds them so no product overflows); each of the order's `heads` tables takes mixed mod its size at its offset (table g
/// of the model's: the (g + 1)-th prime past `base` - 1; the offsets cumulative). The tables' rows concatenated are e [dim]. Then, per
/// stream s of the block's normed input q (the block's `norm_in`, each stream normed on its own): k_s = `norm_key`(`key` e)_s, v =
/// `value` e, gate_s = <k_s, q_s> / sqrt(hidden), made sign(gate_s) sqrt(max(|gate_s|, 1e-6)), g_s = sigmoid(gate_s) v; the output is g +
/// silu(conv(`norm_conv`(g))), a depthwise causal convolution over the streams' channels of `conv` taps `dilation` positions apart; the
/// residual adds it to the streams. Roles: `ngram` [rows, dim / ((order - 1) heads)] (the checkpoint's shards stacked), `key` [streams
/// hidden, dim], `value` [hidden, dim], `norm_key` and `norm_conv` [streams hidden] (their kind `norm`, a stream at a time), `conv`
/// [streams hidden, conv]; and the hash's own int64 tensors, what the parameters make: `ngram_mult` [order], `ngram_sizes`,
/// `ngram_offsets` [(order - 1) heads]. A sequence's state: its last `order` - 1 tokens and the convolution's last (conv - 1) dilation
/// inputs. `form` says where a family's differs (DeepSeek-V4.1's Engram: `NgramForm::Engram`); `table` where the tables are held.
#[derive(Clone, Copy, PartialEq)]
pub struct Ngram {
    pub order: usize,
    pub heads: usize,
    pub dim: usize,
    pub base: u64,
    pub vocab: usize,
    pub seed: u64,
    pub index: usize,
    pub eos: u32,
    pub streams: usize,
    pub conv: usize,
    pub dilation: usize,
    pub norm: Norm,
    pub form: NgramForm,
    pub table: TableAt,
}

/// Where an embedding kind's table, read by rows, is held (`Embed::table`, `Ngram::table`): on the GPU, or on the host: the checkpoint's files mapped, a
/// row read alone where the system's page cache does not hold it (no read-ahead: a prompt's rows are at random), each pass's rows gathered by the host and
/// copied up with it, nothing of the table on the GPU. Any description may hold a table there that no GPU holds beside its model (Qwen3.8-Flash-Next's
/// n-gram tables, 95 GiB; its embedding with them). An embedding that is the head too stays on the GPU, which multiplies by it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TableAt {
    #[default]
    Gpu,
    Host,
}

/// Where an n-gram embedding's hash and output differ, by mechanism.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NgramForm {
    /// Segments ended by `eos`, multipliers from splitmix64, a convolution after the gate: as `Ngram` says (Qwen3.8-Flash-Next's).
    Segmented,
    /// DeepSeek-V4.1's Engram: each token first mapped to a compressed vocabulary of `vocab` ids (tokens that normalise alike, one id: a table
    /// the tokenizer makes, the loader's role `ngram_map` [the model's vocabulary], not a checkpoint tensor), and the hash over those ids; the
    /// multipliers numpy's `default_rng(seed)` draws below (2^63 - 1) / vocab / 2, times 2 plus 1 (`seed`: 10007 times the layer's number); no
    /// segments: a position before the sequence reads as `eos` (its pad id) and the end token is a token like another; no convolution (`conv`
    /// 0): the output is g. The key and the value may be one product, role `kv` [(streams + 1) hidden, dim], the keys' rows first.
    Engram,
}

/// The derived form, with `form` said only where it is not `Segmented` and `table` only where it is not `Gpu`: descriptions of that form read as they did
/// before the fields (the digests of tests/fixtures/descriptions.json are of this text).
impl std::fmt::Debug for Ngram {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut s = f.debug_struct("Ngram");
        s.field("order", &self.order)
            .field("heads", &self.heads)
            .field("dim", &self.dim)
            .field("base", &self.base)
            .field("vocab", &self.vocab)
            .field("seed", &self.seed)
            .field("index", &self.index)
            .field("eos", &self.eos)
            .field("streams", &self.streams)
            .field("conv", &self.conv)
            .field("dilation", &self.dilation)
            .field("norm", &self.norm);
        if self.form != NgramForm::Segmented {
            s.field("form", &self.form);
        }
        if self.table != TableAt::Gpu {
            s.field("table", &self.table);
        }
        s.finish()
    }
}

/// Per-layer inputs (Gemma 4's E2B and E4B, `hidden_size_per_layer_input`): beside its residual each token carries `layers` vectors of `dim` values, made at
/// the embedding from the token and its scaled embedding e: the token's part `ple_embed`[token] (role [vocab, layers dim]) times sqrt(dim); the context's
/// `ple_proj` e (role [layers dim, hidden]) times hidden^-0.5, each layer's `dim` values normed by `norm` (weight `ple_norm` [dim]); the input their sum
/// times 2^-0.5. A token past `vocab` has no token's part (the context's alone). Roles of the model (`Described::extra`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PerLayerIn {
    pub dim: usize,
    pub layers: usize,
    pub vocab: usize,
    pub norm: Norm,
}

/// The gated branch of a per-layer input: o = `proj` (act(`gate` x) input[`layer`]), x the block's input (its `norm_in` `NormKind::Identity`: the residual
/// as it is). Roles `gate` [dim, hidden], `proj` [hidden, dim].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PerLayer {
    pub layer: usize,
    pub act: Act,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Body {
    Mixer(Mixer),
    Ffn(Ffn),
    Ngram(Ngram),
    PerLayer(PerLayer),
}

impl Body {
    /// The products a block of this kind multiplies by, each the roles whose matrices it stacks by rows into one (q, k and v; gate and up; Gated DeltaNet's
    /// queries, keys and values with its output gate z, and its b with its a), as the executor's lowerings read them: a role in none is no matrix of the
    /// kind (a norm, a convolution, a router, an indexer's tensors). A block has the roles of its own (a layer that reads another's keys and values has no
    /// k or v; a shared expert without a gate no `shared_gate`): a product is those of its roles it has. A block of experts' routed experts (its
    /// `experts_*` roles) are each expert's matrices besides these. None for a kind with no products of this form (the compressed attention, the n-gram
    /// embedding, the per-layer inputs). What a lossy save packs (src/pack.rs `walk`).
    pub fn products(&self) -> Option<&'static [&'static [&'static str]]> {
        Some(match self {
            Body::Mixer(Mixer::Attention(_)) => &[&["q", "k", "v"], &["o"]],
            Body::Mixer(Mixer::Linear(l)) if l.kind == LinearKind::GatedDeltaNet => &[&["qkv", "gate"], &["beta", "decay"], &["o"]],
            Body::Mixer(Mixer::Linear(_)) => {
                &[&["q", "k", "v"], &["o"], &["decay"], &["decay_a"], &["decay_b"], &["gate"], &["gate_a"], &["gate_b"], &["beta"]]
            }
            Body::Mixer(Mixer::Mla(_)) => &[&["q_a"], &["q_b"], &["q"], &["kv_a"], &["kv_b"], &["g"], &["o"]],
            Body::Mixer(Mixer::Mamba2(_) | Mixer::ShortConv(_)) => &[&["in"], &["o"]],
            Body::Ffn(Ffn::Mlp(_)) => &[&["gate", "up"], &["down"]],
            Body::Ffn(Ffn::Moe(_)) => &[&["latent_down"], &["latent_up"], &["shared_gate", "shared_up"], &["shared_down"]],
            Body::Mixer(Mixer::Compressed(_)) | Body::Ngram(_) | Body::PerLayer(_) => return None,
        })
    }
}

/// Muse's gated residual X = a X + b O, the factors from role `res_gate` [hidden] and the temperature.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GateP {
    pub temperature: f32,
}

/// n residual streams mixed per block: mHC (GLM-5.3, DeepSeek-V4). The block's input is a weighted sum of the streams; its output is
/// spread back over them with the Sinkhorn-projected mixing of the streams. Roles `hc_fn` [(2 + n) n, n hidden], `hc_base` [(2 + n) n],
/// `hc_scale` [3]. Ibis's gated residuals are the other rule of several streams (`GatedStreamsP`).
#[derive(Clone, Copy, PartialEq)]
pub struct StreamsP {
    pub streams: usize,
    pub sinkhorn_iters: usize,
    /// hc_eps
    pub eps: f32,
    /// The weightless RMS norm of the flattened streams before the mixing weights.
    pub norm_eps: f32,
    /// Where the mixing rounds: `Once`, the release's fp32 mixing projection and update, each result rounded once, the collapse's norm
    /// the weight times the normed value rounded once (DeepSeek-V4); `Twice`, transformers' bf16 tensors between operations: a bf16
    /// projection, post and comb rounded before they weigh, each term of the update rounded, the collapse's norm rounding the normed
    /// value before its weight (GLM-5.3-Flash). The final collapse and norm (`Collapse::Weighted`, `Collapse::Mean`) round as the blocks'.
    pub rounding: Rounding,
    /// Lagged mHC (DeepSeek-V4.1): the block's input is the streams weighed by the previous lagged block's input weights (`pre`), the first
    /// block's by one-hot weights on stream 0 (the streams being copies of the embedding, its embedding unscaled); the weights the block's own
    /// projection makes go to the next block (the model's last ones to the head: `Collapse::LastPre`). Its output's weights and the streams'
    /// mixing are its own, as mHC's.
    pub lagged: bool,
}

/// The derived form, with `lagged` said only where it is set: the other blocks' descriptions read as before the field (the digests of
/// tests/fixtures/descriptions.json are of this text).
impl std::fmt::Debug for StreamsP {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut s = f.debug_struct("StreamsP");
        s.field("streams", &self.streams)
            .field("sinkhorn_iters", &self.sinkhorn_iters)
            .field("eps", &self.eps)
            .field("norm_eps", &self.norm_eps)
            .field("rounding", &self.rounding);
        if self.lagged {
            s.field("lagged", &self.lagged);
        }
        s.finish()
    }
}

/// Kimi-K3's depth mix: a block's input is a softmax mix of the sources so far (the embedding and each completed group of layers'
/// sum) and the current group's partial sum, scored by the query `mix_norm` * `mix_proj` (roles, [hidden]) against each source's RMS
/// norm; the block's output adds to the partial sum. The executor keeps the sources: a block that `starts` a group first closes the
/// partial sum into a source.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DepthMixP {
    pub starts: bool,
}

/// Ibis's gated residuals (Qwen3.8-Flash-Next's `GatedResidual`): `streams` copies of the hidden width, flattened. The block's `norm_in`
/// (role [streams hidden], a weight a stream) norms each stream on its own (n), and there is no other pre-norm; the block's input is the
/// mean over the streams of m_s n_s, m = sigmoid(`mix_up` silu(`mix_down` n / streams)) a value per stream and channel; after the block
/// each stream s adds w_s o, w = 2 sigmoid(`inject` n / streams). No mixing of the streams with each other (mHC's has). Roles `mix_down`
/// [lowrank, streams hidden], `mix_up` [streams hidden, lowrank], `inject` [streams, streams hidden].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GatedStreamsP {
    pub streams: usize,
    pub lowrank: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Residual {
    Add,
    /// Granite's residual multiplier.
    AddScaled(f32),
    Gated(GateP),
    Streams(StreamsP),
    GatedStreams(GatedStreamsP),
    DepthMix(DepthMixP),
    /// x = (x + o) s with the scalar of role `res_scale` [1]: gemma-4's per-layer scalar after the feed-forward block.
    AddThenScale,
}

impl Residual {
    /// The roles the residual reads around its block's body (the block's norms `norm_in` and `norm_out` beside them).
    pub fn roles(&self) -> &'static [&'static str] {
        match self {
            Residual::Add | Residual::AddScaled(_) => &[],
            Residual::Gated(_) => &["res_gate"],
            Residual::Streams(_) => &["hc_fn", "hc_base", "hc_scale"],
            Residual::GatedStreams(_) => &["mix_down", "mix_up", "inject"],
            Residual::DepthMix(_) => &["mix_norm", "mix_proj"],
            Residual::AddThenScale => &["res_scale"],
        }
    }
}

/// Norm roles in a block: `norm_in`, `norm_out`.
#[derive(Clone, Debug, PartialEq)]
pub struct Block {
    pub norm_in: Norm,
    pub body: Body,
    /// Sandwich (Gemma 4, Muse) or post-norm (OLMo 2).
    pub norm_out: Option<Norm>,
    pub residual: Residual,
}

#[derive(Clone, Copy, PartialEq)]
pub struct Embed {
    /// Gemma: x sqrt(hidden).
    pub scale: f32,
    /// A norm of the embedding (Muse: weightless RMS).
    pub norm: Option<Norm>,
    /// Residual streams the embedding is copied into (mHC: 4); 1 for a plain residual.
    pub streams: usize,
    /// The per-layer inputs made beside the residual (Gemma 4's E2B, E4B).
    pub per_layer: Option<PerLayerIn>,
    /// Where its table is held (`TableAt`).
    pub table: TableAt,
}

/// The derived form, with `per_layer` said only where it is set and `table` only where it is not `Gpu`: the other descriptions read as they did before the
/// fields (the digests of tests/fixtures/descriptions.json are of this text).
impl std::fmt::Debug for Embed {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut s = f.debug_struct("Embed");
        s.field("scale", &self.scale).field("norm", &self.norm).field("streams", &self.streams);
        if let Some(p) = &self.per_layer {
            s.field("per_layer", p);
        }
        if self.table != TableAt::Gpu {
            s.field("table", &self.table);
        }
        s.finish()
    }
}

#[derive(Clone, Copy, PartialEq)]
pub struct Head {
    pub tied: bool,
    pub softcap: Option<f32>,
    /// The logits' factor before the soft-cap (Muse's output multiplier).
    pub scale: f32,
    /// How the residual becomes the final norm's input.
    pub collapse: Collapse,
    /// Where the logits round to bf16: `F32` nowhere; `Once` the product's output alone; `Twice` a bf16 tensor between operations, as
    /// transformers' head of a bf16 model makes them: the product's output, then the scale's product and each of the soft-cap's three operations
    /// (the division, the tanh, the product) rounded (Gemma 4 as released: models/gemma4/modeling_gemma4.py takes `lm_head`'s bf16 output through
    /// `/ cap`, `tanh`, `* cap`; vLLM's head in the model's type does the same).
    pub rounding: Rounding,
}

/// The derived form, with `rounding` said only where the logits round: the other heads' descriptions read as they did before the field (the
/// digests of tests/fixtures/descriptions.json are of this text).
impl std::fmt::Debug for Head {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut s = f.debug_struct("Head");
        s.field("tied", &self.tied).field("softcap", &self.softcap).field("scale", &self.scale).field("collapse", &self.collapse);
        if self.rounding != Rounding::F32 {
            s.field("rounding", &self.rounding);
        }
        s.finish()
    }
}

/// The residual stream(s) into the final norm.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Collapse {
    /// The one stream.
    None,
    /// The streams' mean (GLM-5.3-Flash).
    Mean,
    /// The depth mix of every source with the model's roles `out_mix_norm`, `out_mix_proj` (Kimi-K3).
    DepthMix,
    /// DeepSeek-V4's hyper-head: a weighted sum of the streams, the weights sigmoid(`out_hc_fn` rms(streams) `out_hc_scale` +
    /// `out_hc_base`) + eps (roles [n, n hidden], [1], [n]).
    Weighted { eps: f32, norm_eps: f32 },
    /// Ibis's final mixer: the gated residuals' input mix (`GatedStreamsP`) with the model's roles `out_mix_down`, `out_mix_up`, each stream
    /// normed by the final norm (its weight [n hidden], a stream's each) before the mix, and no norm after it.
    GatedMix,
    /// Lagged mHC's (DeepSeek-V4.1, no hyper-head): the streams weighed by the last block's own input weights (`StreamsP::lagged`).
    LastPre,
}

/// A draft head: a few blocks after a projection of [embedding of the next token, the model's last hidden state] (R6); or DSpark's form
/// (`dspark`), whose blocks draft a block of tokens in one pass; `input` says how the head takes the model's state and the next token.
#[derive(Clone, PartialEq)]
pub struct MtpHead {
    pub blocks: Vec<Block>,
    pub norm_embed: Norm,
    pub norm_hidden: Norm,
    pub final_norm: Norm,
    pub dspark: Option<DSpark>,
    pub input: MtpInput,
}

/// How a draft head (`MtpHead`) takes the model's state and the next token's embedding.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MtpInput {
    /// `eh_proj` [hidden, 2 hidden] of [norm_embed(embedding), norm_hidden(hidden)] (the embedding first), the model's final normed hidden state the
    /// head continues; its blocks on one residual stream, then `final_norm` into the model's head (DeepSeek-V3's form).
    Joined,
    /// The model's last residual streams (n of the hidden width, before its collapse): norm_hidden over the n hidden values as one vector, each
    /// stream's part by `fc_hidden` [hidden, hidden], `fc_embedding` [hidden, hidden] of norm_embed(embedding) added to every stream; the head's
    /// blocks on those streams (the model's residual rule), its streams after them the next draft's input; then the model's collapse with the
    /// head's own roles (`Collapse::GatedMix`: `final_norm` on each stream, `out_mix_down`, `out_mix_up`) into the model's head (Qwen3.8-Flash-Next's
    /// `residual_linear_shared` head: vLLM models/qwen4_exp/nvidia/mtp.py, SGLang models/qwen4_exp_mtp.py).
    Streams,
}

/// The derived form, with `input` said only where it is not `Joined`: the other heads' descriptions read as they did before the field (the digests
/// of tests/fixtures/descriptions.json are of this text).
impl std::fmt::Debug for MtpHead {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut s = f.debug_struct("MtpHead");
        s.field("blocks", &self.blocks)
            .field("norm_embed", &self.norm_embed)
            .field("norm_hidden", &self.norm_hidden)
            .field("final_norm", &self.final_norm)
            .field("dspark", &self.dspark);
        if self.input != MtpInput::Joined {
            s.field("input", &self.input);
        }
        s.finish()
    }
}

/// DSpark (DeepSeek-V4's 0731 release, DeepSeek-V4.1): `block` tokens drafted in one pass, the anchor (the last token the target chose) then
/// `block` - 1 `noise` tokens, each position predicting the next. The context: main_x = `main_norm`(`main_proj` [hidden, targets hidden] of the
/// target's attention inputs at the layers `targets`, concatenated) at each position before the anchor; each draft layer's keys (= values) of
/// the context are its own `kv` product of main_x, normed and rotated at the position. The head's blocks are the model's decoder layers with
/// the window alone (`Comp` of ratio 0), their queries attending to the context keys in the window and to every token of the block (not
/// causal inside it); then the streams' collapse (the model's: a hyper-head of the head's own `out_hc_*` roles, or lagged mHC's last weights),
/// `final_norm`, the model's head: the base logits. Sequentially, position j's logits add `markov_head` [vocab, rank] `markov_embed`[prev]
/// ([vocab, rank]: the Markov embedding of the token before it, the anchor at j = 0) and the token is chosen; with `confidence`,
/// sigmoid(`confidence` [1, hidden + rank] [pre-norm hidden, Markov embedding]) estimates its acceptance.
#[derive(Clone, Debug, PartialEq)]
pub struct DSpark {
    pub block: usize,
    pub noise: u32,
    pub targets: Vec<usize>,
    pub markov_rank: usize,
    pub confidence: bool,
}

/// A vision tower: the Qwen-VL vision transformer (Qwen3.8-Flash-Next's `Qwen4ExpVisionModel`). An image's patches of `channels` x `frames` x `patch` x `patch`
/// pixels projected to `hidden` (role `patch`, a bias), plus the learned position table of `table` x `table` rows (role `pos`) resampled bilinearly (corners
/// aligned) to the image's grid of patches; `blocks` pre-norm blocks (LayerNorms with a bias, eps `eps`) of full attention over the image's patches (`heads` of
/// hidden / heads, q, k and v one product with a bias, a 2-D axial rotary: a quarter of a head's channels turning by the patch's row, a quarter by its column,
/// twice over (`rotate_half`), frequencies theta^(-4j / head) for `theta`) and of an MLP (`inter` neurons, `act`, biases); then the merger: a LayerNorm, each
/// `merge` x `merge` block of patches as one row, two products with a bias and a GELU (erf) between, into `out` (the text model's width): the image's tokens,
/// `image_token` in the text, one a row, in their blocks' order. Its tensors are the description's `vision` roles (`names::part::VISION`, `VISION_BLOCK`).
/// In the text its tokens take M-RoPE positions (the attention's `Rope::mrope`): see `vision::positions`.
#[derive(Clone, Debug, PartialEq)]
pub struct VisionDesc {
    pub blocks: usize,
    pub hidden: usize,
    pub heads: usize,
    pub inter: usize,
    pub act: Act,
    pub channels: usize,
    pub patch: usize,
    pub frames: usize,
    pub merge: usize,
    pub out: usize,
    pub table: usize,
    pub theta: f64,
    pub eps: f32,
    pub image_token: u32,
}

impl VisionDesc {
    pub fn head_dim(&self) -> usize {
        self.hidden / self.heads
    }

    /// A patch's pixels, as the patch product reads them: channels x frames x patch x patch.
    pub fn patch_dim(&self) -> usize {
        self.channels * self.frames * self.patch * self.patch
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ModelDesc {
    pub hidden: usize,
    pub vocab: usize,
    pub embed: Embed,
    pub blocks: Vec<Block>,
    pub final_norm: Norm,
    pub head: Head,
    pub mtp: Vec<MtpHead>,
    pub vision: Option<VisionDesc>,
    /// The default for kinds that do not set their own.
    pub rope: Option<Rope>,
}

impl ModelDesc {
    /// Every rounding point a description names in f32: the attentions' (`AttnRounding::F32`), the residual streams', the norms' (the blocks', the final
    /// one, the latent attentions' q and key-value latents', a latent block of experts'), the head's (`Head::rounding`): the description as transformers
    /// runs it in float32, which the fixtures of `tests/runtime` were recorded in.
    pub fn f32_arithmetic(&mut self) {
        self.final_norm.rounding = Rounding::F32;
        for b in self.blocks.iter_mut().chain(self.mtp.iter_mut().flat_map(|m| m.blocks.iter_mut())) {
            b.norm_in.rounding = Rounding::F32;
            if let Some(n) = b.norm_out.as_mut() {
                n.rounding = Rounding::F32;
            }
            match &mut b.body {
                Body::Mixer(Mixer::Attention(a)) => a.rounding = AttnRounding::F32,
                Body::Mixer(Mixer::Mla(m)) => (m.q_norm.rounding, m.kv_norm.rounding) = (Rounding::F32, Rounding::F32),
                // (a latent block's norms: Kimi-K3's on the weighed sum, Muse Spark's on the projected input and each expert's output)
                Body::Ffn(Ffn::Moe(Moe { latent: Some(l), .. })) => {
                    l.norm_in.iter_mut().chain(&mut l.expert_norm).chain(&mut l.norm_out).for_each(|n| n.rounding = Rounding::F32)
                }
                _ => {}
            }
            if let Residual::Streams(s) = &mut b.residual {
                s.rounding = Rounding::F32;
            }
        }
        for m in self.mtp.iter_mut() {
            (m.norm_embed.rounding, m.norm_hidden.rounding, m.final_norm.rounding) = (Rounding::F32, Rounding::F32, Rounding::F32);
        }
        self.head.rounding = Rounding::F32;
    }
}
