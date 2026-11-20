//! `Ref`, the CPU f32 reference backend (`docs/design/runtime.md` §3, decision 6): one plain function per kind, written for clarity,
//! the formula of the model's reference code. Products sum in f64 and round to f32; everything else is f32, rounded to bf16 only where a
//! kind's parameters name a rounding point (`kinds::Rounding`).
//!
//! Tolerances, per kind, against what each test compares it to:
//!
//! | kind | against | tolerance |
//! | :--- | :--- | :--- |
//! | every kind | transformers in float32 on the same weights (`tests/runtime/*.json`) | `HF_TOL` (5e-6 of the output's scale) |
//! | attention, MLA, linear | the per-family references, which round to bf16 where vLLM or transformers hold bf16 | `BF16_TOL` (2e-2 of the output's scale) |
//!
//! A backend's kernels are checked against `Ref` with their own tolerance (bf16 accumulation order differs); a lossless format must first
//! match its bf16 kernel bit for bit.

pub mod attention;
pub mod linear;
pub mod mla;
pub mod norm;
pub mod rope;

#[cfg(test)]
mod hf_tests;
#[cfg(all(test, feature = "dev"))]
mod ref_tests;
#[cfg(test)]
pub(crate) mod testing;

use super::backend::{
    Arenas, Backend, Captured, Collective, DeviceId, DeviceInfo, Format, HostTensor, KindRef, LayerDesc, LayerImpl, Placement, StepCx, StepShape, Support,
    WeightSource,
};
use super::kinds::{Act, Block, Body, Mixer, Residual, RopeScaling};
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
        Body::Mixer(Mixer::Mla(_)) => &["q", "q_a", "q_b", "kv_a", "kv_b", "o", "g"],
        Body::Mixer(Mixer::Linear(_)) => &["qkv", "q", "k", "v", "beta", "decay", "gate", "o"],
        _ => &[],
    };
    let vecs = [
        "q_norm",
        "k_norm",
        "v_norm",
        "q_a_norm",
        "kv_a_norm",
        "sinks",
        "conv",
        "a_log",
        "dt_bias",
        "norm",
        "norm_in",
        "norm_in_bias",
        "norm_out",
        "norm_out_bias",
        "res_gate",
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

/// A sequence's state in one layer.
pub enum SeqState {
    Attention(attention::KvCache),
    Mla(mla::LatentCache),
    Linear(linear::LinearState),
}

/// One block on the reference backend: its weights, and each sequence's cache or state by slot.
pub struct RefLayer {
    pub block: Block,
    pub hidden: usize,
    pub p: Params,
    pub seqs: HashMap<u32, SeqState>,
}

impl RefLayer {
    pub fn new(block: Block, hidden: usize, p: Params) -> RefLayer {
        RefLayer { block, hidden, p, seqs: HashMap::new() }
    }

    /// The body on the normed inputs of one token (the sequence's state advanced).
    fn body(&mut self, seq: u32, pos: u32, h: &[f32]) -> Result<Vec<f32>> {
        let p = &self.p;
        match &self.block.body {
            Body::Mixer(Mixer::Attention(a)) => {
                let st = self.seqs.entry(seq).or_insert_with(|| SeqState::Attention(attention::KvCache::default()));
                let SeqState::Attention(c) = st else { return err("a sequence's state is not an attention cache") };
                attention::step(a, p, c, h, pos)
            }
            Body::Mixer(Mixer::Mla(m)) => {
                let st = self.seqs.entry(seq).or_insert_with(|| SeqState::Mla(mla::LatentCache::default()));
                let SeqState::Mla(c) = st else { return err("a sequence's state is not a latent cache") };
                mla::step(m, p, c, h, pos)
            }
            Body::Mixer(Mixer::Linear(l)) => {
                let st = self.seqs.entry(seq).or_insert_with(|| SeqState::Linear(linear::LinearState::new(l)));
                let SeqState::Linear(s) = st else { return err("a sequence's state is not a linear state") };
                linear::step(l, p, s, h)
            }
            _ => err("the reference has no such kind yet"),
        }
    }

    /// The block on the tokens of `x` ([tokens, hidden]), in order: pre-norm, body, post-norm, residual.
    pub fn forward(&mut self, positions: &[u32], seqs: &[u32], x: &mut [f32]) -> Result<()> {
        let d = self.hidden;
        if x.len() != positions.len() * d || seqs.len() != positions.len() {
            return err(format!("a step of {} values for {} tokens of width {d}", x.len(), positions.len()));
        }
        for (t, xt) in x.chunks_mut(d).enumerate() {
            let h = norm::norm(&self.block.norm_in, xt, self.p.opt("norm_in"), self.p.opt("norm_in_bias"));
            let mut o = self.body(seqs[t], positions[t], &h)?;
            if let Some(n) = self.block.norm_out {
                o = norm::norm(&n, &o, self.p.opt("norm_out"), self.p.opt("norm_out_bias"));
            }
            match &self.block.residual {
                Residual::Add => xt.iter_mut().zip(&o).for_each(|(x, &o)| *x += o),
                Residual::AddScaled(s) => xt.iter_mut().zip(&o).for_each(|(x, &o)| *x += s * o),
                Residual::Gated(g) => {
                    let (a, b) = residual_gate(self.p.vec("res_gate")?, g.temperature);
                    for (i, x) in xt.iter_mut().enumerate() {
                        *x = a[i] * *x + b[i] * o[i];
                    }
                }
                r => return err(format!("the reference has no residual rule {r:?} yet")),
            }
        }
        Ok(())
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
        self.forward(cx.positions, cx.seqs, x)
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

    /// Every implemented kind in any format a source decodes to f32, at the speed of plain Rust.
    fn supports(&self, kind: &KindRef, _: &Format) -> Support {
        match kind {
            KindRef::Attention | KindRef::SlidingAttention | KindRef::Mla | KindRef::GatedDeltaNet | KindRef::KimiDelta => {
                Support::Slower("the f32 reference".into())
            }
            k => Support::No(format!("no reference for {k:?} yet")),
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
        if rope.is_some_and(|r| r.scaling != RopeScaling::None) {
            return err(format!("layer {}: rope scaling is not in the reference yet", layer.index));
        }
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
        let mamba = Block {
            body: Body::Mixer(Mixer::Mamba2(crate::runtime::kinds::Mamba2 { heads: 1, head_dim: 1, state: 1, groups: 1, conv: 4, chunk: 1 })),
            ..block.clone()
        };
        assert!(r.build(&LayerDesc { index: 3, block: &mamba, hidden: 64 }, &p, &Placement::default(), &mut Arenas::default()).is_err());
    }
}
