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
    /// M-RoPE: the angles split into sections for temporal, height and width positions (Qwen3.8 vision); text positions use one.
    pub mrope: Option<[usize; 3]>,
}

impl Rope {
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
}

/// Where an attention output gate's values come from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GateFrom {
    /// Its own product, role `g` [heads * v_head_dim, hidden] (Muse, Kimi-K3's MLA).
    Own,
    /// Interleaved with the queries in `q`: per head, head_dim query rows then head_dim gate rows (Qwen3.5/3.8 `q_proj`).
    InQuery,
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
/// `block` tokens, or a compressed entry), the best kept. Condor's block indexer, GLM's pooled-key indexer and DeepSeek-V4's
/// compressed-entry indexer are this kind with parameters. In `Ref`: a latent attention's (`reference::mla`, GLM-5.3-Flash),
/// DeepSeek-V4's (`reference::comp`) and Condor's on a GQA attention (`reference::attention`, `IndexForm::Mean`).
#[derive(Clone, PartialEq)]
pub struct Indexer {
    pub heads: usize,
    pub head_dim: usize,
    /// What a query keeps. A latent attention: `top_k` tokens, as the `top_k / block` best complete pools of `block` tokens and the tokens
    /// of the incomplete last pool (GLM-5.3-Flash's `index_topk`; `block` 1: the best `top_k` tokens; Condor's `indexer_budget` in pools of
    /// `indexer_compress_ratio`). DeepSeek-V4: `top_k` compressed entries (`block` is its ratio).
    pub top_k: usize,
    /// Tokens a pooled key covers (GLM-5.3-Flash's `index_kpool`; 1: a key per token).
    pub block: usize,
    /// The layer reuses the previous indexer layer's choice and has no indexer weights (GLM's `indexer_types` "shared").
    pub shared: bool,
    pub form: IndexForm,
}

/// The derived form, with `form` said only where it is not the gated form with LayerNorm-ed keys (`IndexForm::LAYER_KEYS`): those indexers'
/// descriptions read as they did before the field (the digests of tests/fixtures/descriptions.json are of this text).
impl std::fmt::Debug for Indexer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut s = f.debug_struct("Indexer");
        s.field("heads", &self.heads).field("head_dim", &self.head_dim).field("top_k", &self.top_k).field("block", &self.block).field("shared", &self.shared);
        if self.form != IndexForm::LAYER_KEYS {
            s.field("form", &self.form);
        }
        s.finish()
    }
}

/// How an indexer makes its queries, its pooled keys and its scores.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum IndexForm {
    /// Queries and a key a token from their own products, each token's key normed by `key` (role `indexer_k_norm`, and with a LayerNorm its
    /// bias `indexer_k_norm_bias`); a pool's key the softmax(gate + ape)-weighted sum of its tokens' keys; the heads' scores weighed by a learned
    /// product of the block's input. The key's norm: a LayerNorm at 1e-6 (GLM-5.3-Flash, GLM-5, DeepSeek-V3.2: `LAYER_KEYS`), an RMS norm at the
    /// model's eps (Kimi-K3.1's `dsa_config.index_k_norm_type` rmsnorm, its LayerNorm otherwise; DeepSeek-V4's compressed entries, normed by
    /// their compressor's `idx_norm`).
    Gated { key: Norm },
    /// Condor's Qwen Sparse Attention (Qwen3.8-Flash-Next): the `heads` queries and ONE key from one product of the block's input (role
    /// `index_qk` [(heads + 1) head_dim, hidden]); each query head normed by `norm` (role `index_q_norm` [head_dim]) and rotated as the
    /// attention's queries (its rope, on the first `dim` channels of the head); the raw keys cached, a pool's key the mean of its tokens'
    /// raw keys, normed by `norm` (role `index_k_norm`) and rotated at the pool's first position; a pool's score sum_h relu(q_h . k) /
    /// sqrt(head_dim), the heads unweighted.
    Mean { norm: Norm },
}

impl IndexForm {
    /// The gated form whose keys are LayerNorm-ed with a bias at 1e-6 (GLM-5.3-Flash's, GLM-5's, DeepSeek-V3.2's).
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
/// [head_dim] where those norms have a weight; `g` with an `Own` gate; `sinks` [heads] with `sinks`.
#[derive(Clone, Debug, PartialEq)]
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
}

/// Multi-head latent attention (DeepSeek-V2/V3 style, Kimi-K3, GLM-5.3).
///
/// Roles: `q` [heads * (nope + rope_dim), hidden] or with `q_lora` `q_a` [q_lora, hidden], `q_a_norm` [q_lora], `q_b` [heads * (nope +
/// rope_dim), q_lora]; `kv_a` [kv_lora + rope_dim, hidden], `kv_a_norm` [kv_lora], `kv_b` [heads * (nope + v_dim), kv_lora], `o` [hidden,
/// heads * v_dim]; `g` [heads * v_dim, hidden] with a gate. The cache holds per token the normed latent and the shared key part
/// (kv_lora + rope_dim values); `absorbed` names how a backend computes it (the same function).
#[derive(Clone, Debug, PartialEq)]
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
    /// Gated DeltaNet (Qwen3.5/3.8, Condor): one decay per value head.
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

#[derive(Clone, Debug, PartialEq)]
pub enum Mixer {
    Attention(Attn),
    Mla(Mla),
    Compressed(Comp),
    Linear(Linear),
    Mamba2(Mamba2),
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
    /// SwiGLU limits: the gate clamped above and the up both ways (DeepSeek-V4).
    pub clamp: Option<f32>,
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
/// [1, hidden] (the shared output times sigmoid of it: Qwen3.5/3.8 MoE, Condor).
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

/// A hashed n-gram embedding added to the residual streams (Condor's per-layer embedding, Qwen3.8-Flash-Next's `ple`).
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
/// inputs. `form` says where a family's differs (DeepSeek-V4.1's Engram: `NgramForm::Engram`).
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
}

/// Where an n-gram embedding's hash and output differ by family.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NgramForm {
    /// Condor's (Qwen3.8-Flash-Next), as `Ngram` says.
    Condor,
    /// DeepSeek-V4.1's Engram: each token first mapped to a compressed vocabulary of `vocab` ids (tokens that normalise alike, one id: a table
    /// the tokenizer makes, the loader's role `ngram_map` [the model's vocabulary], not a checkpoint tensor), and the hash over those ids; the
    /// multipliers numpy's `default_rng(seed)` draws below (2^63 - 1) / vocab / 2, times 2 plus 1 (`seed`: 10007 times the layer's number); no
    /// segments: a position before the sequence reads as `eos` (its pad id) and the end token is a token like another; no convolution (`conv`
    /// 0): the output is g. The key and the value may be one product, role `kv` [(streams + 1) hidden, dim], the keys' rows first.
    Engram,
}

/// The derived form, with `form` said only where it is not Condor's: Condor's descriptions read as they did before the field (the digests of
/// tests/fixtures/descriptions.json are of this text).
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
        if self.form != NgramForm::Condor {
            s.field("form", &self.form);
        }
        s.finish()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Body {
    Mixer(Mixer),
    Ffn(Ffn),
    Ngram(Ngram),
}

/// Muse's gated residual X = a X + b O, the factors from role `res_gate` [hidden] and the temperature.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GateP {
    pub temperature: f32,
}

/// n residual streams mixed per block: mHC (GLM-5.3, DeepSeek-V4). The block's input is a weighted sum of the streams; its output is
/// spread back over them with the Sinkhorn-projected mixing of the streams. Roles `hc_fn` [(2 + n) n, n hidden], `hc_base` [(2 + n) n],
/// `hc_scale` [3]. Condor's gated residuals are the other rule of several streams (`GatedStreamsP`).
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

/// Condor's gated residuals (Qwen3.8-Flash-Next's `GatedResidual`): `streams` copies of the hidden width, flattened. The block's `norm_in`
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

/// Norm roles in a block: `norm_in`, `norm_out`.
#[derive(Clone, Debug, PartialEq)]
pub struct Block {
    pub norm_in: Norm,
    pub body: Body,
    /// Sandwich (Gemma 4, Muse) or post-norm (OLMo 2).
    pub norm_out: Option<Norm>,
    pub residual: Residual,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Embed {
    /// Gemma: x sqrt(hidden).
    pub scale: f32,
    /// A norm of the embedding (Muse: weightless RMS).
    pub norm: Option<Norm>,
    /// Residual streams the embedding is copied into (mHC: 4); 1 for a plain residual.
    pub streams: usize,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Head {
    pub tied: bool,
    pub softcap: Option<f32>,
    /// The logits' factor before the soft-cap (Muse's output multiplier).
    pub scale: f32,
    /// How the residual becomes the final norm's input.
    pub collapse: Collapse,
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
    /// Condor's final mixer: the gated residuals' input mix (`GatedStreamsP`) with the model's roles `out_mix_down`, `out_mix_up`, each stream
    /// normed by the final norm (its weight [n hidden], a stream's each) before the mix, and no norm after it.
    GatedMix,
    /// Lagged mHC's (DeepSeek-V4.1, no hyper-head): the streams weighed by the last block's own input weights (`StreamsP::lagged`).
    LastPre,
}

/// A draft head: a few blocks after a projection of [embedding of the next token, the model's last hidden state] (R6); or DSpark's form
/// (`dspark`), whose blocks draft a block of tokens in one pass.
#[derive(Clone, Debug, PartialEq)]
pub struct MtpHead {
    pub blocks: Vec<Block>,
    pub norm_embed: Norm,
    pub norm_hidden: Norm,
    pub final_norm: Norm,
    pub dspark: Option<DSpark>,
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

#[derive(Clone, Debug, PartialEq)]
pub struct VisionDesc {
    pub blocks: usize,
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
    /// Every attention's arithmetic in f32 (`AttnRounding::F32`): the description as transformers runs it in float32, which the fixtures of
    /// `tests/runtime` were recorded in.
    pub fn f32_attention(&mut self) {
        for b in self.blocks.iter_mut().chain(self.mtp.iter_mut().flat_map(|m| m.blocks.iter_mut())) {
            if let Body::Mixer(Mixer::Attention(a)) = &mut b.body {
                a.rounding = AttnRounding::F32;
            }
        }
    }
}
