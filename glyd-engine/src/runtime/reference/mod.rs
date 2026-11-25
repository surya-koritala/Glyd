//! `Ref`, the CPU f32 reference backend (`docs/design/runtime.md` §3, decision 6): one plain function per kind, written for clarity,
//! the formula of the model's reference code. Products sum in f64 and round to f32; everything else is f32, rounded to bf16 only where a
//! kind's parameters name a rounding point (`kinds::Rounding` of a norm, `kinds::AttnRounding` of an attention).
//!
//! Tolerances, per kind, against what each test compares it to:
//!
//! | kind | against | tolerance |
//! | :--- | :--- | :--- |
//! | every kind | transformers in float32 on the same weights (`tests/runtime/*.json`) | `HF_TOL` (5e-6 of the output's scale) |
//! | attention, MLA, linear | the per-family references, which round to bf16 where vLLM or transformers hold bf16 | `BF16_TOL` (2e-2 of the output's scale) |
//! | each kind below | the engine's CUDA kernels (`glyd-engine kindcheck`, `dynamic/kindcheck.rs`) | the table below |
//!
//! The CUDA kernels against `Ref`, kind by kind (`kindcheck`: every layer of the random model of each recent family's logits case, at
//! the lossless level; a body on the GPU's own input rows, the residual's operations on the GPU's own body outputs; a prompt of 48 rows
//! and 4 decode steps, each value the largest difference over the reference's largest magnitude on a pass: the prompt's rows together, a
//! step's row alone). The bound is `kindcheck::bound`; the worst is the largest an RTX 4080 SUPER measured (2026-11-23) in three runs: the
//! default, a prompt of 60 and 6 steps, and Kimi-K3 and GLM-5.3-Flash at a prompt of 20 and 40 steps (no difference grew with the position).
//! A lossless format must first match its bf16 kernel bit for bit; the kernels hold bf16 activations, so every row has bf16's 2^-9
//! of rounding at each point where a value is stored.
//!
//! | kind (families measured) | bound | why it is not 0 | worst |
//! | :--- | :--- | :--- | :--- |
//! | attention: GQA and MQA, windows, K = V, full, partial, proportional or no rope, an in-query or own gate after a norm, heads of 64 to 512 (Gemma 4, the Qwen3.5 layout, Nemotron-H, Muse Spark) | 2e-2 | `Ref` rounds where the family's engine does (`AttnRounding`: Gemma 4's eager bf16 scores and rope, Muse's SDPA, FlashAttention's bf16 probabilities); left: fp32 sums in another order than f64's, the GPU's exp2 and online softmax, a decode step's keys summed in parts | 5.4e-3 |
//! | latent attention: Kimi-K3's gated NoPE MLA, GLM-5.3-Flash's with the pooled-key indexer | 2e-2 | the GPU keeps the latents, the absorbed queries and the pooled keys bf16, `Ref` f32; an indexer's choice nearer than 1e-2 to another (`kindcheck::NEAR`) can go the other way, and such a row past the bound is named, not compared (one row in each run, its margin 2.3e-3) | 1.1e-2 |
//! | compressed attention: DeepSeek-V4's window, ratio 16, ratio 4 with the indexer | 2e-2 | the window's keys and the compressed entries bf16 on the GPU; the indexer's near ties as above (2 rows of 260, 3 of 330; margins 1.2e-4 to 2.2e-3) | 7.3e-3 |
//! | Gated DeltaNet (the Qwen3.5 layout) | 3e-2 | the state fp32 on both; q, k, v and the gates bf16 on the GPU; the prompt's chunked (WY) form against `Ref`'s token recurrence; the per-head norm after the recurrence divides by the output's RMS, so where a head's sum cancels its error is magnified | 8.6e-3 |
//! | Kimi delta attention (Kimi-K3, GLM-5.3-Flash) | 3e-2 | as Gated DeltaNet (Kimi-K3's first layer, a step's row alone: single rows to 2.1e-2, most 3-7e-3, none growing with the position) | 2.1e-2 |
//! | Mamba-2 (Nemotron-H) | 3e-2 | as Gated DeltaNet, the prompt's chunked scan | 5.6e-3 |
//! | MLP: SwiGLU, its limits, gelu-tanh, situ (Gemma 4, the Qwen3.5 layout, Kimi-K3, GLM-5.3-Flash) | 1e-2 | gate and up rounded before the activation, the activation's product rounded, down's sums in fp32 | 6.1e-3 |
//! | MoE: sigmoid routers with a choice bias, sqrt-softplus and hash routers, shared and latent experts, expert norms (Nemotron-H, Muse Spark, Kimi-K3, GLM-5.3-Flash, DeepSeek-V4) | 2e-2 | each expert's activations bf16, the weighted sum's order; a router's choice near a tie as above (none past the bound) | 9.9e-3 |
//! | residual and norms: RMS (w, 1 + w, input-scaled, Kimi's two roundings), sandwich norms and the layer scalar, the gated residual, mHC's streams, the depth mix, each family's final norm and collapse (all seven) | 2e-2 | the residual bf16 after each add, the normed value rounded, mHC's Sinkhorn and the depth mix's softmax in fp32 on the GPU's bf16 inputs | 1.6e-2 |
//!
//! Not compared on the GPU: the embedding's scale and norm and the head's soft-cap (each family's logits fixture holds them,
//! `logitscheck`); the Qwen3.5 layout's mixture of experts (not lowered to the executor yet); MTP heads (their blocks are the kinds above);
//! Condor's kinds (Qwen3.8-Flash-Next: the gated residuals and the final mixer, the indexer on GQA, the n-gram embedding), which run on
//! Condor's own engine and not on the executor yet: the whole tiny model on `Ref` against that engine's logits fixture instead
//! (`describe_tests::condors_model_on_the_reference_against_condors_engine`: KL mean 8.5e-6, max 5.0e-5).

pub mod attention;
pub mod comp;
pub mod ffn;
pub mod linear;
pub mod mamba;
pub mod mla;
pub mod model;
pub mod mtp;
pub mod ngram;
pub mod norm;
pub mod rope;

#[cfg(test)]
mod hf_tests;
#[cfg(all(test, feature = "dev"))]
mod ref_tests;
#[cfg(any(test, feature = "dev"))]
pub(crate) mod testing;

use super::backend::{
    Arenas, Backend, Captured, Collective, DeviceId, DeviceInfo, Format, HostTensor, KindRef, LayerDesc, LayerImpl, Placement, StepCx, StepShape, Support,
    WeightSource,
};
use super::kinds::{Act, Block, Body, Ffn, Mixer, Residual};
use crate::ffi::{err, Error, Result};
use std::collections::HashMap;

/// Against transformers in float32: the largest difference over the largest output magnitude.
pub const HF_TOL: f32 = 5e-6;
/// Against a reference that rounds to bf16 between operations.
pub const BF16_TOL: f32 = 2e-2;

/// `x` rounded to bf16 (to nearest, ties to even), as an f32.
pub fn bf16(x: f32) -> f32 {
    if x.is_nan() {
        return x;
    }
    let b = x.to_bits();
    f32::from_bits(b.wrapping_add(0x7fff + ((b >> 16) & 1)) & 0xffff_0000)
}

pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

pub fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}

/// torch's softplus (beta 1, threshold 20: above it, x).
pub fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else {
        x.exp().ln_1p()
    }
}

/// An activation of `kinds::Act`.
pub fn act(a: Act, x: f32) -> f32 {
    match a {
        Act::Sigmoid => sigmoid(x),
        Act::Silu => silu(x),
        Act::GeluTanh => 0.5 * x * (1.0 + ((2.0 / std::f32::consts::PI).sqrt() * (x + 0.044715 * x * x * x)).tanh()),
        Act::Gelu => (0.5 * x as f64 * (1.0 + erf(x as f64 / std::f64::consts::SQRT_2))) as f32,
        Act::Relu2 => {
            let r = x.max(0.0);
            r * r
        }
        Act::Situ { beta, .. } => beta * (x / beta).tanh() * sigmoid(x),
    }
}

/// A gated pair: act(g) u, with situ's cap on the up (u -> linear tanh(u / linear)).
pub fn gated(a: Act, g: f32, u: f32) -> f32 {
    match a {
        Act::Situ { linear, .. } => act(a, g) * linear * (u / linear).tanh(),
        _ => act(a, g) * u,
    }
}

/// The error function, to within 1.2e-7 (Numerical Recipes' erfc by Chebyshev fit).
pub fn erf(x: f64) -> f64 {
    let z = x.abs();
    let t = 1.0 / (1.0 + 0.5 * z);
    let p = [-1.26551223, 1.00002368, 0.37409196, 0.09678418, -0.18628806, 0.27886807, -1.13520398, 1.48851587, -0.82215223, 0.17087277];
    let poly = p.iter().rev().fold(0.0, |acc, &c| acc * t + c);
    let erfc = t * (-z * z + poly).exp();
    if x >= 0.0 {
        1.0 - erfc
    } else {
        erfc - 1.0
    }
}

/// W [rows, cols] times x, each row's sum in f64, rounded to f32.
pub fn matvec(w: &[f32], cols: usize, x: &[f32]) -> Vec<f32> {
    debug_assert_eq!(x.len(), cols);
    w.chunks(cols).map(|r| r.iter().zip(x).map(|(&a, &b)| a as f64 * b as f64).sum::<f64>() as f32).collect()
}

/// x / sqrt(sum(x^2) + eps), in place.
pub fn l2norm(x: &mut [f32], eps: f32) {
    let r = 1.0 / (x.iter().map(|v| v * v).sum::<f32>() + eps).sqrt();
    x.iter_mut().for_each(|v| *v *= r);
}

/// A dot product in f64, rounded to f32.
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(&x, &y)| x as f64 * y as f64).sum::<f64>() as f32
}

/// In place softmax over `s`.
pub fn softmax(s: &mut [f32]) {
    let m = s.iter().fold(f32::NEG_INFINITY, |m, &x| m.max(x));
    let mut z = 0f64;
    for v in s.iter_mut() {
        *v = (*v - m).exp();
        z += *v as f64;
    }
    for v in s.iter_mut() {
        *v = (*v as f64 / z) as f32;
    }
}

/// A layer's weights by role (`kinds`), as f32.
#[derive(Clone, Debug, Default)]
pub struct Params {
    pub t: HashMap<String, HostTensor>,
}

impl Params {
    pub fn new() -> Params {
        Params::default()
    }

    pub fn put(&mut self, role: &str, shape: &[usize], data: Vec<f32>) {
        assert_eq!(shape.iter().product::<usize>(), data.len(), "{role}");
        self.t.insert(role.to_string(), HostTensor { shape: shape.to_vec(), data });
    }

    pub fn has(&self, role: &str) -> bool {
        self.t.contains_key(role)
    }

    pub fn opt(&self, role: &str) -> Option<&[f32]> {
        self.t.get(role).map(|t| &t.data[..])
    }

    pub fn vec(&self, role: &str) -> Result<&[f32]> {
        self.opt(role).ok_or_else(|| Error(format!("no weight for role {role}")))
    }

    /// Whether `role` is a product here, dense or low-rank.
    pub fn has_prod(&self, role: &str) -> bool {
        self.has(role) || (self.has(&format!("{role}_a")) && self.has(&format!("{role}_b")))
    }

    /// The product `role` on x (dense [out, in], or `role_b (role_a x)`), plus `role_bias` where there is one.
    pub fn prod(&self, role: &str, x: &[f32]) -> Result<Vec<f32>> {
        let mut y = if let Some(w) = self.t.get(role) {
            if w.shape.len() != 2 || w.shape[1] != x.len() {
                return err(format!("{role}: shape {:?} on an input of {}", w.shape, x.len()));
            }
            matvec(&w.data, x.len(), x)
        } else {
            let (a, b) = (self.t.get(&format!("{role}_a")), self.t.get(&format!("{role}_b")));
            match (a, b) {
                (Some(a), Some(b)) => {
                    let mid = matvec(&a.data, x.len(), x);
                    matvec(&b.data, mid.len(), &mid)
                }
                _ => return err(format!("no weight for role {role}")),
            }
        };
        if let Some(b) = self.opt(&format!("{role}_bias")) {
            y.iter_mut().zip(b).for_each(|(v, &b)| *v += b);
        }
        Ok(y)
    }
}

impl WeightSource for Params {
    fn format(&self, role: &str) -> Option<Format> {
        self.has(role).then_some(Format::F32)
    }
    fn f32(&self, role: &str) -> Option<HostTensor> {
        self.t.get(role).cloned()
    }
}

/// Every role a block may use, with the low-rank and bias forms of the products; `build` takes the ones the source has.
fn roles(b: &Block) -> Vec<String> {
    let prods: &[&str] = match &b.body {
        Body::Mixer(Mixer::Attention(_)) => &["q", "k", "v", "o", "g"],
        Body::Mixer(Mixer::Mla(_)) => &["q", "q_a", "q_b", "kv_a", "kv_b", "o", "g", "indexer_wq_b", "indexer_wk", "indexer_weights", "indexer_pool_gate"],
        Body::Mixer(Mixer::Linear(_)) => &["qkv", "q", "k", "v", "beta", "decay", "gate", "o"],
        Body::Mixer(Mixer::Compressed(_)) => &["q_a", "q_b", "kv", "o_a", "o_b", "comp_kv", "comp_gate", "idx_kv", "idx_gate", "idx_q_b", "idx_weights"],
        Body::Mixer(Mixer::Mamba2(_)) => &["in", "o"],
        Body::Ngram(_) => &["key", "value"],
        Body::PerLayer(_) => &["gate", "proj"],
        Body::Ffn(_) => &[
            "gate",
            "up",
            "gate_up",
            "down",
            "router",
            "shared_gate",
            "shared_up",
            "shared_gate_up",
            "shared_down",
            "shared_router",
            "latent_down",
            "latent_up",
        ],
    };
    let vecs = [
        "q_norm",
        "k_norm",
        "v_norm",
        "q_a_norm",
        "kv_a_norm",
        "sinks",
        "conv",
        "conv_bias",
        "a_log",
        "d",
        "dt_bias",
        "norm",
        "norm_in",
        "norm_in_bias",
        "norm_out",
        "norm_out_bias",
        "res_gate",
        "indexer_k_norm",
        "indexer_k_norm_bias",
        "indexer_pool_ape",
        "index_qk",
        "index_q_norm",
        "index_k_norm",
        "mix_down",
        "mix_up",
        "inject",
        "ngram",
        "norm_key",
        "norm_conv",
    ];
    let mut r: Vec<String> = Vec::new();
    for p in prods {
        for s in ["", "_a", "_b", "_bias"] {
            r.push(format!("{p}{s}"));
        }
    }
    r.extend(vecs.iter().map(|s| s.to_string()));
    r
}

/// A residual gate's two factors per channel (Muse): w = g / T, b = sigmoid(w), a = sqrt(max(sigmoid(-w) (1 + b), 1e-3)).
pub fn residual_gate(g: &[f32], temperature: f32) -> (Vec<f32>, Vec<f32>) {
    g.iter()
        .map(|&g| {
            let w = g / temperature;
            let b = sigmoid(w);
            ((sigmoid(-w) * (1.0 + b)).max(1e-3).sqrt(), b)
        })
        .unzip()
}

/// `n` on each group of `group` values of `x` (a stream at a time), with each group's part of the weight `w`.
pub fn norm_groups(n: &crate::runtime::kinds::Norm, x: &[f32], w: Option<&[f32]>, group: usize) -> Vec<f32> {
    x.chunks(group).enumerate().flat_map(|(i, c)| norm::norm(n, c, w.map(|w| &w[i * group..][..group]), None)).collect()
}

/// Condor's gated residual on one token's `n` streams `xt` [n hidden]: each stream normed by `norm` (its weight `w`) into nx; the mix m =
/// sigmoid(up silu(down nx / n)); the block's input the mean over the streams of m_s nx_s; and with `inject`, the streams' weights of the
/// block's output 2 sigmoid(inject nx / n) (the final mixer has none).
pub fn gated_mix(
    n: usize,
    norm: &crate::runtime::kinds::Norm,
    w: &[f32],
    down: &[f32],
    up: &[f32],
    inject: Option<&[f32]>,
    xt: &[f32],
) -> Result<(Vec<f32>, Vec<f32>)> {
    let (width, d) = (xt.len(), xt.len() / n.max(1));
    let rank = down.len() / width.max(1);
    if n * d != width || w.len() != width || rank * width != down.len() || up.len() != down.len() || inject.is_some_and(|i| i.len() != n * width) {
        return err(format!(
            "gated residual: weights of {}, {}, {} and {:?} values for {n} streams of {d}",
            w.len(),
            down.len(),
            up.len(),
            inject.map(<[f32]>::len)
        ));
    }
    let nx = norm_groups(norm, xt, Some(w), d);
    let mid: Vec<f32> = matvec(down, width, &nx).into_iter().map(|v| silu(v / n as f32)).collect();
    let m: Vec<f32> = matvec(up, rank, &mid).into_iter().map(sigmoid).collect();
    let input = (0..d).map(|i| ((0..n).map(|s| m[s * d + i] as f64 * nx[s * d + i] as f64).sum::<f64>() / n as f64) as f32).collect();
    let inj = inject.map_or(Vec::new(), |i| matvec(i, width, &nx).into_iter().map(|v| 2.0 * sigmoid(v / n as f32)).collect());
    Ok((input, inj))
}

/// mHC's weights for one token: the input's weights over the streams (pre), the output's (post) and the streams' mixing comb[j][k]
/// (from stream j into stream k), from the streams x [n, hidden] (transformers' `DeepseekV4HyperConnection`).
pub fn hc_weights(s: &crate::runtime::kinds::StreamsP, p: &Params, x: &[f32]) -> Result<(Vec<f32>, Vec<f32>, Vec<Vec<f32>>)> {
    let n = s.streams;
    let r = 1.0 / (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32 + s.norm_eps).sqrt();
    let xn: Vec<f32> = x.iter().map(|v| v * r).collect();
    let (f, base, scale) = (p.vec("hc_fn")?, p.vec("hc_base")?, p.vec("hc_scale")?);
    if f.len() != (2 + n) * n * x.len() || base.len() != (2 + n) * n || scale.len() != 3 {
        return err(format!("mHC weights of {}, {} and {} values for {n} streams of {}", f.len(), base.len(), scale.len(), x.len() / n));
    }
    let w = matvec(f, x.len(), &xn);
    let pre: Vec<f32> = (0..n).map(|i| sigmoid(w[i] * scale[0] + base[i]) + s.eps).collect();
    let post: Vec<f32> = (0..n).map(|i| 2.0 * sigmoid(w[n + i] * scale[1] + base[n + i])).collect();
    let mut comb: Vec<Vec<f32>> = (0..n)
        .map(|j| {
            let mut row: Vec<f32> = (0..n).map(|k| w[2 * n + j * n + k] * scale[2] + base[2 * n + j * n + k]).collect();
            softmax(&mut row);
            row.iter().map(|v| v + s.eps).collect()
        })
        .collect();
    let cols = |c: &mut Vec<Vec<f32>>| {
        for k in 0..n {
            let z: f32 = (0..n).map(|j| c[j][k]).sum::<f32>() + s.eps;
            (0..n).for_each(|j| c[j][k] /= z);
        }
    };
    cols(&mut comb);
    for _ in 1..s.sinkhorn_iters {
        for row in comb.iter_mut() {
            let z: f32 = row.iter().sum::<f32>() + s.eps;
            row.iter_mut().for_each(|v| *v /= z);
        }
        cols(&mut comb);
    }
    Ok((pre, post, comb))
}

/// Kimi-K3's depth mix: each source (the embedding, the completed blocks' outputs, the partial sum) scored by the query
/// w_norm * w_proj against the source normed by its RMS (eps 1e-5), the softmax of the scores weighing the sources' sum. A block's
/// `DepthMix` residual needs the sources across blocks, which the executor (R4a) keeps; this is the function it applies.
pub fn depth_mix(sources: &[&[f32]], w_norm: &[f32], w_proj: &[f32]) -> Vec<f32> {
    let d = w_norm.len();
    let mut s: Vec<f32> = sources
        .iter()
        .map(|v| {
            let r = 1.0 / (v.iter().map(|x| x * x).sum::<f32>() / d as f32 + 1e-5).sqrt();
            (0..d).map(|i| (w_norm[i] * w_proj[i]) as f64 * v[i] as f64).sum::<f64>() as f32 * r
        })
        .collect();
    softmax(&mut s);
    (0..d).map(|i| sources.iter().zip(&s).map(|(v, &a)| a as f64 * v[i] as f64).sum::<f64>() as f32).collect()
}

/// mHC's weights of one token's block output: the output's over the streams, and the streams' mixing.
pub type Hc = (Vec<f32>, Vec<Vec<f32>>);

/// A sequence's state in one layer.
pub enum SeqState {
    Attention(attention::KvCache),
    Mla(mla::LatentCache),
    Linear(linear::LinearState),
    Comp(comp::CompState),
    Mamba(mamba::MambaState),
    Ngram(ngram::NgramState),
}

/// One block on the reference backend: its weights, and each sequence's cache or state by slot.
pub struct RefLayer {
    pub block: Block,
    pub hidden: usize,
    pub p: Params,
    pub seqs: HashMap<u32, SeqState>,
    /// A MoE block's choices, each token's in turn, where a pass counts them (`hb_usage::count_on_ref`).
    pub routes: Option<Vec<usize>>,
    /// How near the last token's choice (a router's experts, an indexer's picks) was to another (`ffn::margin`); infinite where the body chose
    /// nothing.
    pub margin: f32,
}

impl RefLayer {
    pub fn new(block: Block, hidden: usize, p: Params) -> RefLayer {
        RefLayer { block, hidden, p, seqs: HashMap::new(), routes: None, margin: f32::INFINITY }
    }

    /// The body on the normed inputs of one token (the sequence's state advanced).
    pub fn body(&mut self, seq: u32, pos: u32, token: u32, h: &[f32]) -> Result<Vec<f32>> {
        self.body_with(None, seq, pos, token, h)
    }

    /// The sequence's attention cache in this block, where it has one.
    pub fn kv_cache(&self, seq: u32) -> Option<&attention::KvCache> {
        match self.seqs.get(&seq) {
            Some(SeqState::Attention(c)) => Some(c),
            _ => None,
        }
    }

    /// `body`, with the block whose keys and values an attention shares (`Attn::kv_from`): `store`, its sequence's cache read, never written.
    pub fn body_with(&mut self, store: Option<&RefLayer>, seq: u32, pos: u32, token: u32, h: &[f32]) -> Result<Vec<f32>> {
        let p = &self.p;
        self.margin = f32::INFINITY;
        match &self.block.body {
            Body::Mixer(Mixer::Attention(a)) => {
                let shared = match (a.kv_from, store) {
                    (None, _) => None,
                    (Some(_), Some(s)) => Some(s.kv_cache(seq).ok_or_else(|| Error(format!("sequence {seq}: no keys and values in the block it shares")))?),
                    (Some(j), None) => return err(format!("an attention that shares block {j}'s keys and values runs with that block (`body_with`)")),
                };
                let st = self.seqs.entry(seq).or_insert_with(|| SeqState::Attention(attention::KvCache::default()));
                let SeqState::Attention(c) = st else { return err("a sequence's state is not an attention cache") };
                let y = attention::step_with(a, p, c, shared, h, pos);
                self.margin = c.margin;
                y
            }
            Body::Mixer(Mixer::Mla(m)) => {
                let st = self.seqs.entry(seq).or_insert_with(|| SeqState::Mla(mla::LatentCache::default()));
                let SeqState::Mla(c) = st else { return err("a sequence's state is not a latent cache") };
                let y = mla::step(m, p, c, h, pos);
                self.margin = c.margin;
                y
            }
            Body::Mixer(Mixer::Linear(l)) => {
                let st = self.seqs.entry(seq).or_insert_with(|| SeqState::Linear(linear::LinearState::new(l)));
                let SeqState::Linear(s) = st else { return err("a sequence's state is not a linear state") };
                linear::step(l, p, s, h)
            }
            Body::Mixer(Mixer::Compressed(c)) => {
                let st = self.seqs.entry(seq).or_insert_with(|| SeqState::Comp(comp::CompState::default()));
                let SeqState::Comp(s) = st else { return err("a sequence's state is not a compressed attention's") };
                let y = comp::step(c, p, s, h, pos);
                self.margin = s.margin;
                y
            }
            Body::Mixer(Mixer::Mamba2(m)) => {
                let st = self.seqs.entry(seq).or_insert_with(|| SeqState::Mamba(mamba::MambaState::new(m)));
                let SeqState::Mamba(s) = st else { return err("a sequence's state is not a Mamba-2 state") };
                mamba::step(m, p, s, h)
            }
            Body::PerLayer(pl) => err(format!("the per-layer input {}'s branch reads its token's input: it runs in the model (`RefModel::body`)", pl.layer)),
            Body::Ngram(n) => {
                let st = self.seqs.entry(seq).or_insert_with(|| SeqState::Ngram(ngram::NgramState::new(n, h.len())));
                let SeqState::Ngram(s) = st else { return err("a sequence's state is not an n-gram embedding's") };
                ngram::step(n, p, s, h, token)
            }
            Body::Ffn(Ffn::Mlp(m)) => ffn::mlp(m, p, "", h),
            Body::Ffn(Ffn::Moe(m)) => {
                let (y, ids, gap) = ffn::moe_routed(m, p, h, token)?;
                if let Some(r) = self.routes.as_mut() {
                    r.extend(ids);
                }
                self.margin = gap;
                Ok(y)
            }
        }
    }

    /// A token's residual width: the streams' (mHC's, the gated residuals', an n-gram embedding's), else the hidden width.
    pub fn width(&self) -> usize {
        self.hidden
            * match (&self.block.residual, &self.block.body) {
                (Residual::Streams(s), _) => s.streams,
                (Residual::GatedStreams(g), _) => g.streams,
                (_, Body::Ngram(n)) => n.streams,
                _ => 1,
            }
    }

    /// The block's input from one token's residual `xt` (its streams, where the rule has several): mHC's weighted sum of the streams, with the
    /// weights its output needs (`post`); the gated residuals' mix of the normed streams (the block's norm taken here), with the streams' weights
    /// of its output; else the residual itself.
    pub fn pre(&self, xt: &[f32]) -> Result<(Vec<f32>, Option<Hc>)> {
        let d = self.hidden;
        Ok(match &self.block.residual {
            Residual::Streams(s) if s.lagged => return err("a lagged mHC block runs in the model's executor (its input's weights are the previous block's)"),
            Residual::Streams(s) => {
                let (pre, post, comb) = hc_weights(s, &self.p, xt)?;
                let input: Vec<f32> = (0..d).map(|i| (0..s.streams).map(|j| pre[j] * xt[j * d + i]).sum()).collect();
                (input, Some((post, comb)))
            }
            Residual::GatedStreams(g) => {
                let p = &self.p;
                let (input, inj) =
                    gated_mix(g.streams, &self.block.norm_in, p.vec("norm_in")?, p.vec("mix_down")?, p.vec("mix_up")?, Some(p.vec("inject")?), xt)?;
                (input, Some((inj, Vec::new())))
            }
            _ => (xt.to_vec(), None),
        })
    }

    /// The residual rule: the block's output `o` (after its post-norm) into the token's residual `xt`, with what `pre` gave for mHC.
    pub fn post(&self, xt: &mut [f32], o: &[f32], hc: Option<Hc>) -> Result<()> {
        let d = self.hidden;
        match &self.block.residual {
            Residual::Add => xt.iter_mut().zip(o).for_each(|(x, &o)| *x += o),
            Residual::AddScaled(s) => xt.iter_mut().zip(o).for_each(|(x, &o)| *x += s * o),
            Residual::AddThenScale => {
                let s = self.p.vec("res_scale")?[0];
                xt.iter_mut().zip(o).for_each(|(x, &o)| *x = (*x + o) * s);
            }
            Residual::Gated(g) => {
                let (a, b) = residual_gate(self.p.vec("res_gate")?, g.temperature);
                for (i, x) in xt.iter_mut().enumerate() {
                    *x = a[i] * *x + b[i] * o[i];
                }
            }
            Residual::Streams(s) => {
                let (post, comb) = hc.ok_or_else(|| Error("mHC's residual without the weights of its input".into()))?;
                let old = xt.to_vec();
                for k in 0..s.streams {
                    for i in 0..d {
                        xt[k * d + i] = post[k] * o[i] + (0..s.streams).map(|j| comb[j][k] * old[j * d + i]).sum::<f32>();
                    }
                }
            }
            Residual::GatedStreams(g) => {
                let (inj, _) = hc.ok_or_else(|| Error("a gated residual without the weights of its input".into()))?;
                for (s, w) in inj.iter().enumerate().take(g.streams) {
                    xt[s * d..][..d].iter_mut().zip(o).for_each(|(x, &o)| *x += w * o);
                }
            }
            Residual::DepthMix(_) => return err("a depth-mix block runs in the model's executor (its sources span blocks)"),
        }
        Ok(())
    }

    /// The block on the tokens of `x` ([tokens, hidden]), in order: pre-norm, body, post-norm, residual. `tokens` (the ids) may be empty
    /// for a block that does not read them.
    pub fn forward(&mut self, positions: &[u32], seqs: &[u32], tokens: &[u32], x: &mut [f32]) -> Result<()> {
        self.forward_with(None, positions, seqs, tokens, x)
    }

    /// `forward`, with the block whose keys and values the attention shares (`body_with`).
    pub fn forward_with(&mut self, store: Option<&RefLayer>, positions: &[u32], seqs: &[u32], tokens: &[u32], x: &mut [f32]) -> Result<()> {
        if seqs.len() != positions.len() {
            return err(format!("{} sequence slots for {} tokens", seqs.len(), positions.len()));
        }
        let width = self.width();
        if x.len() != positions.len() * width {
            return err(format!("a step of {} values for {} tokens of width {width}", x.len(), positions.len()));
        }
        for (t, xt) in x.chunks_mut(width).enumerate() {
            let (input, hc) = self.pre(xt)?;
            let o = self.output_with(store, seqs[t], positions[t], tokens.get(t).copied().unwrap_or(0), &input)?;
            self.post(xt, &o, hc)?;
        }
        Ok(())
    }

    /// The block's output for one token's input (before the residual rule): pre-norm, body, post-norm.
    pub fn output(&mut self, seq: u32, pos: u32, token: u32, input: &[f32]) -> Result<Vec<f32>> {
        self.output_with(None, seq, pos, token, input)
    }

    pub fn output_with(&mut self, store: Option<&RefLayer>, seq: u32, pos: u32, token: u32, input: &[f32]) -> Result<Vec<f32>> {
        let h = self.norm_in(input);
        let o = self.body_with(store, seq, pos, token, &h)?;
        Ok(self.norm_out(o))
    }

    /// The block whose keys and values this one's attention shares (`Attn::kv_from`).
    pub fn kv_from(&self) -> Option<usize> {
        match &self.block.body {
            Body::Mixer(Mixer::Attention(a)) => a.kv_from,
            _ => None,
        }
    }

    /// The block's pre-norm of its input: each stream on its own where the input is the streams (an n-gram embedding's); none where the gated
    /// residuals took it (`pre`).
    pub fn norm_in(&self, input: &[f32]) -> Vec<f32> {
        match (&self.block.residual, &self.block.body) {
            (Residual::GatedStreams(_), _) => input.to_vec(),
            (_, Body::Ngram(_)) => norm_groups(&self.block.norm_in, input, self.p.opt("norm_in"), self.hidden),
            _ => norm::norm(&self.block.norm_in, input, self.p.opt("norm_in"), self.p.opt("norm_in_bias")),
        }
    }

    /// The block's post-norm of its body's output, where it has one.
    pub fn norm_out(&self, o: Vec<f32>) -> Vec<f32> {
        match self.block.norm_out {
            Some(n) => norm::norm(&n, &o, self.p.opt("norm_out"), self.p.opt("norm_out_bias")),
            None => o,
        }
    }
}

impl LayerImpl<Ref> for RefLayer {
    fn workspace(&self, _: &StepShape) -> u64 {
        0
    }
    fn capturable(&self) -> bool {
        false
    }
    fn run(&mut self, cx: &mut StepCx<Ref>, x: &mut Vec<f32>) -> Result<()> {
        self.forward(cx.positions, cx.seqs, cx.tokens, x)
    }
}

/// The CPU f32 reference backend.
pub struct Ref {
    devices: Vec<DeviceInfo>,
}

impl Default for Ref {
    fn default() -> Self {
        Ref { devices: vec![DeviceInfo { vendor: "cpu".into(), arch: std::env::consts::ARCH.into(), memory: 0, unified: true }] }
    }
}

impl Backend for Ref {
    type Queue = ();
    type Fence = ();
    type Tensor = Vec<f32>;

    fn devices(&self) -> &[DeviceInfo] {
        &self.devices
    }

    /// Every kind, in any format a source decodes to f32, at the speed of plain Rust (DeepSeek-V4's compressed attention too: `comp.rs`; a latent
    /// attention's indexer: `mla.rs`; Condor's on GQA: `attention.rs`; its n-gram embedding: `ngram.rs`). The match names each kind, so that a kind added
    /// to `KindRef` is decided here. (A gated-form indexer on plain attention, and GLM's `shared` indexer that reuses the previous layer's picks, are not in
    /// the reference: such a layer says so where it would pick.)
    fn supports(&self, kind: &KindRef, _: &Format) -> Support {
        match kind {
            KindRef::Attention
            | KindRef::SlidingAttention
            | KindRef::SparseAttention
            | KindRef::Mla
            | KindRef::Compressed
            | KindRef::GatedDeltaNet
            | KindRef::KimiDelta
            | KindRef::Mamba2
            | KindRef::Mlp
            | KindRef::Moe
            | KindRef::Ngram
            | KindRef::PerLayer => Support::Slower("the f32 reference".into()),
        }
    }

    fn build(&self, layer: &LayerDesc, w: &dyn WeightSource, _: &Placement, _: &mut Arenas<Self>) -> Result<Box<dyn LayerImpl<Self>>> {
        let b = layer.block;
        if let Support::No(why) = self.supports(&KindRef::of(b), &Format::F32) {
            return err(format!("layer {}: {why}", layer.index));
        }
        let rope = match &b.body {
            Body::Mixer(Mixer::Attention(a)) => a.rope.as_ref(),
            Body::Mixer(Mixer::Mla(m)) => m.rope.as_ref(),
            _ => None,
        };
        // (M-RoPE: the reference's positions are text positions, the same on every axis, which is plain rope)
        let _ = rope;
        let mut p = Params::new();
        for r in roles(b) {
            if let Some(t) = w.f32(&r) {
                p.t.insert(r, t);
            }
        }
        Ok(Box::new(RefLayer::new(b.clone(), layer.hidden, p)))
    }

    fn queue(&self, _: DeviceId) -> Result<()> {
        Ok(())
    }

    fn capture(&self, _: &(), _: &mut dyn FnMut() -> Result<()>) -> Result<Option<Captured>> {
        Ok(None)
    }

    fn collective(&self) -> Option<&dyn Collective> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::testing::{assert_close, Fixture};
    use super::*;
    use crate::runtime::kinds::{Decay, Linear, LinearKind, Norm};

    /// Condor's gated residual by hand on one token of two streams of 4 (a rank of 3): each stream normed with its own part of the weight, the
    /// bottleneck's mix, the mean over the streams, the streams' weights of the output; a block between gated residuals takes no other norm and
    /// adds w_s o to stream s; the final mixer (`Collapse::GatedMix`) is the mix without the weights.
    #[test]
    fn a_gated_residual_by_hand() {
        use super::testing::fill;
        use crate::runtime::kinds::{GatedStreamsP, Mlp};
        let (n, d, r) = (2usize, 4usize, 3usize);
        let mut p = Params::new();
        p.put("norm_in", &[n * d], fill("gr.norm", n * d, -0.3, 0.3));
        p.put("mix_down", &[r, n * d], fill("gr.down", r * n * d, -0.5, 0.5));
        p.put("mix_up", &[n * d, r], fill("gr.up", n * d * r, -0.5, 0.5));
        p.put("inject", &[n, n * d], fill("gr.inject", n * n * d, -0.5, 0.5));
        for (role, rows, cols) in [("gate", 6, d), ("up", 6, d), ("down", d, 6)] {
            p.put(role, &[rows, cols], fill(&format!("gr.{role}"), rows * cols, -0.5, 0.5));
        }
        let x = fill("gr.x", n * d, -2.0, 2.0);
        let nm = Norm::rms_plus_one(1e-6);
        // by hand
        let w = p.vec("norm_in").unwrap();
        let nx: Vec<f32> = (0..n)
            .flat_map(|s| {
                let c = &x[s * d..][..d];
                let rr = 1.0 / (c.iter().map(|v| v * v).sum::<f32>() / d as f32 + 1e-6).sqrt();
                (0..d).map(move |i| c[i] * rr * (1.0 + w[s * d + i]))
            })
            .collect();
        let mid: Vec<f32> = matvec(p.vec("mix_down").unwrap(), n * d, &nx).iter().map(|v| silu(v / 2.0)).collect();
        let m: Vec<f32> = matvec(p.vec("mix_up").unwrap(), r, &mid).iter().map(|&v| sigmoid(v)).collect();
        let input: Vec<f32> = (0..d).map(|i| (m[i] * nx[i] + m[d + i] * nx[d + i]) / 2.0).collect();
        let inj: Vec<f32> = matvec(p.vec("inject").unwrap(), n * d, &nx).iter().map(|v| 2.0 * sigmoid(v / 2.0)).collect();
        let (gi, gw) = gated_mix(n, &nm, w, p.vec("mix_down").unwrap(), p.vec("mix_up").unwrap(), Some(p.vec("inject").unwrap()), &x).unwrap();
        assert_close(&gi, &input, 1e-6, "the block's input");
        assert_close(&gw, &inj, 1e-6, "the streams' weights");
        // the block: the MLP on the mix (no second norm), its output into each stream by its weight
        let mlp = Mlp { act: Act::Silu, gated: true, bias: false, clamp: None };
        let block =
            Block { norm_in: nm, body: Body::Ffn(Ffn::Mlp(mlp)), norm_out: None, residual: Residual::GatedStreams(GatedStreamsP { streams: n, lowrank: r }) };
        let mut layer = RefLayer::new(block, d, p.clone());
        let mut y = x.clone();
        layer.forward(&[0], &[0], &[], &mut y).unwrap();
        let o = ffn::mlp(&mlp, &p, "", &input).unwrap();
        let want: Vec<f32> = (0..n * d).map(|j| x[j] + inj[j / d] * o[j % d]).collect();
        assert_close(&y, &want, 1e-6, "the streams after the block");
        // the final mixer: the mix alone
        assert_eq!(gated_mix(n, &nm, w, p.vec("mix_down").unwrap(), p.vec("mix_up").unwrap(), None, &x).unwrap(), (gi, Vec::new()));
    }

    /// A block built through the backend seam: two sequences interleaved in one flat batch give what each gives alone, and the backend
    /// refuses a kind it has no reference for.
    #[test]
    fn a_block_through_the_seam() {
        let f = Fixture::load("qwen38_gdn");
        let l = Linear {
            kind: LinearKind::GatedDeltaNet,
            k_heads: 2,
            v_heads: 4,
            k_dim: 16,
            v_dim: 16,
            conv: 4,
            decay: Decay::Softplus,
            gate_act: Act::Silu,
            norm_eps: 1e-6,
            state_f32: true,
        };
        let block = Block { norm_in: Norm::rms_plus_one(1e-6), body: Body::Mixer(Mixer::Linear(l)), norm_out: None, residual: Residual::Add };
        let mut p = Params::new();
        for (role, name) in [
            ("qkv", "in_proj_qkv.weight"),
            ("beta", "in_proj_b.weight"),
            ("decay", "in_proj_a.weight"),
            ("gate", "in_proj_z.weight"),
            ("a_log", "A_log"),
            ("dt_bias", "dt_bias"),
            ("norm", "norm.weight"),
            ("o", "out_proj.weight"),
        ] {
            let (shape, v) = f.tensor(name);
            p.put(role, &shape, v);
        }
        let (_, conv) = f.tensor("conv1d.weight");
        p.put("conv", &[128, 4], conv);
        p.put("norm_in", &[64], vec![0.1; 64]);
        let r = Ref::default();
        let desc = LayerDesc { index: 0, block: &block, hidden: 64 };
        let build = || r.build(&desc, &p, &Placement::default(), &mut Arenas::default()).unwrap();
        let x = f.input();
        let (a, b) = (&x[..3 * 64], &x[3 * 64..6 * 64]);
        let alone = |x: &[f32]| {
            let mut layer = build();
            let mut y = x.to_vec();
            layer.run(&mut StepCx { queue: &mut (), positions: &[0, 1, 2], seqs: &[0, 0, 0], tokens: &[] }, &mut y).unwrap();
            y
        };
        let (ya, yb) = (alone(a), alone(b));
        let mut layer = build();
        let mut y: Vec<f32> = [&a[..64], &b[..64], &a[64..128], &b[64..128], &a[128..], &b[128..]].concat();
        layer.run(&mut StepCx { queue: &mut (), positions: &[0, 0, 1, 1, 2, 2], seqs: &[7, 9, 7, 9, 7, 9], tokens: &[] }, &mut y).unwrap();
        let (mut ga, mut gb) = (Vec::new(), Vec::new());
        for (i, ch) in y.chunks(64).enumerate() {
            if i % 2 == 0 {
                ga.extend_from_slice(ch)
            } else {
                gb.extend_from_slice(ch)
            }
        }
        assert_close(&ga, &ya, 0.0, "sequence a");
        assert_close(&gb, &yb, 0.0, "sequence b");
        // (DeepSeek-V4's compressed attention through the seam too, as `RefModel` runs it)
        let f = Fixture::load("dsv4_model");
        let d = crate::runtime::describe::describe(&f.j["config"]).unwrap();
        let comp = d.desc.blocks.iter().find(|b| matches!(b.body, Body::Mixer(Mixer::Compressed(_)))).expect("a compressed attention block");
        assert!(matches!(r.supports(&KindRef::of(comp), &Format::F32), Support::Slower(_)));
        assert!(r.build(&LayerDesc { index: 3, block: comp, hidden: 64 }, &p, &Placement::default(), &mut Arenas::default()).is_ok());
    }
}
