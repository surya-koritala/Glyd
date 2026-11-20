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
    /// RMSNorm only: the normed value rounded, then times the weight and rounded again (KimiRMSNorm, `research/kimi-k3.md` §2.1).
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

/// A sparse-attention indexer: picks `top_k` of the past tokens (or blocks of `block` tokens) per query. Condor's block indexer, GLM's
/// pooled-key indexer and DeepSeek-V4's compressed-entry indexer are this kind with parameters. Not in `Ref` yet.
#[derive(Clone, Debug, PartialEq)]
pub struct Indexer {
    pub heads: usize,
    pub head_dim: usize,
    pub top_k: usize,
    pub block: usize,
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

/// DeepSeek-V4's attention: one latent KV head, a sliding window plus per-layer compressed entries. Not in `Ref` yet.
#[derive(Clone, Debug, PartialEq)]
pub struct Comp {
    pub heads: usize,
    pub head_dim: usize,
    pub window: usize,
    pub ratio: usize,
    pub rope: Option<Rope>,
    pub indexer: Option<Indexer>,
    pub sinks: bool,
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

/// Mamba-2 (Nemotron-H, Granite 4). Not in `Ref` yet.
#[derive(Clone, Debug, PartialEq)]
pub struct Mamba2 {
    pub heads: usize,
    pub head_dim: usize,
    pub state: usize,
    pub groups: usize,
    pub conv: usize,
    pub chunk: usize,
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

#[derive(Clone, Debug, PartialEq)]
pub enum Body {
    Mixer(Mixer),
    Ffn(Ffn),
}

/// Muse's gated residual X = a X + b O, the factors from role `res_gate` [hidden] and the temperature.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GateP {
    pub temperature: f32,
}

/// n residual streams mixed per block (mHC with Sinkhorn on GLM-5.3 and DeepSeek-V4; Condor's gated residuals).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StreamsP {
    pub streams: usize,
    pub sinkhorn_iters: usize,
}

/// Kimi-K3's depth mix: a block's input is a softmax over the embedding and the completed blocks of `block` layers.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DepthMixP {
    pub block: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Residual {
    Add,
    /// Granite's residual multiplier.
    AddScaled(f32),
    Gated(GateP),
    Streams(StreamsP),
    DepthMix(DepthMixP),
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
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Head {
    pub tied: bool,
    pub softcap: Option<f32>,
    pub scale: f32,
}

/// A draft head: a few blocks after a projection of [embedding of the next token, the model's last hidden state] (R6).
#[derive(Clone, Debug, PartialEq)]
pub struct MtpHead {
    pub blocks: Vec<Block>,
    pub norm_embed: Norm,
    pub norm_hidden: Norm,
    pub final_norm: Norm,
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
