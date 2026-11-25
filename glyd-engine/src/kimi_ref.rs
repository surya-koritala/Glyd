//! The CPU reference of Kimi-K3 (src/kimi.rs; the math: research/kimi-k3.md), one sequence a token at a time, from the checkpoint's weights decoded exactly (the
//! experts' MXFP4 codes times their E8M0 scales): fp32 inside, the products' sums in f64, rounded to bf16 where the release holds a bf16 tensor. The delta attention
//! runs its recurrent form (the release's decode path), the latent attention its absorbed form (as the engine runs it; the release expands the keys and values). The GPU's
//! kernels are checked against it; `random` writes a small model of the release's format to check them on.

use crate::ffi::{err, Error, Result};
use crate::glm::GlmKind;
use crate::kimi::ConfigKimi;
use glyd_gpu::safetensors::Checkpoint;
use std::path::Path;

#[cfg(test)]
thread_local! {
    /// A test's switch: every bf16 rounding of this reference off on this thread, so that its arithmetic is held term by term to the kinds' reference in
    /// fp32 (`logits::tests::kimi_references_compute_the_same_function`).
    pub(crate) static PLAIN: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// The bf16 value of `x`, where the release holds a bf16 tensor (`x` itself on a thread of a test that set `PLAIN`).
fn bf16(x: f32) -> f32 {
    #[cfg(test)]
    if PLAIN.with(|p| p.get()) {
        return x;
    }
    crate::qwen35_ref::bf16(x)
}

fn io<T>(r: std::io::Result<T>) -> Result<T> {
    r.map_err(|e| Error(e.to_string()))
}

pub use crate::formats::mxfp4;

/// A matrix's values [rows, cols].
struct Mat {
    cols: usize,
    w: Vec<f32>,
    /// An MXFP4 matrix's codes and scales as the checkpoint has them, a row decoded where it is used (an expert's f32 values would take 8 times its bytes: 118 GB a layer).
    fp4: Option<(Vec<u8>, Vec<u8>)>,
}

impl Mat {
    /// W x, the sums in f64, each output rounded to bf16 (a bf16 product's output).
    fn mul(&self, x: &[f32]) -> Vec<f32> {
        assert_eq!(x.len(), self.cols);
        let dot = |r: &[f32]| bf16(r.iter().zip(x).map(|(&a, &b)| a as f64 * b as f64).sum::<f64>() as f32);
        match &self.fp4 {
            None => self.w.chunks(self.cols).map(dot).collect(),
            Some((q, e)) => q
                .chunks(self.cols / 2)
                .zip(e.chunks(self.cols / 32))
                .map(|(q, e)| dot(&mxfp4(q, e, 1, self.cols).expect("a row of the matrix's shape")))
                .collect(),
        }
    }
}

/// A tensor's values as f32 (bf16 or fp32; an expert's MXFP4 matrix by its `weight_packed` and `weight_scale`).
fn tensor(ck: &Checkpoint, name: &str) -> Result<(Vec<u64>, Vec<f32>)> {
    let read = |n: &str| -> Result<(String, Vec<u64>, Vec<u8>)> {
        let (_, t) = ck.get(n).ok_or_else(|| Error(format!("no tensor {n}")))?;
        Ok((t.dtype.clone(), t.shape.clone(), io(ck.read(n))?))
    };
    if let Some(base) = name.strip_suffix(".weight").filter(|b| ck.get(&format!("{b}.weight_packed")).is_some()) {
        let (_, sh, q) = read(&format!("{base}.weight_packed"))?;
        let (_, _, s) = read(&format!("{base}.weight_scale"))?;
        let (rows, cols) = (sh[0] as usize, sh[1] as usize * 2);
        return Ok((vec![rows as u64, cols as u64], mxfp4(&q, &s, rows, cols)?));
    }
    let (dtype, sh, b) = read(name)?;
    let v = match dtype.as_str() {
        "BF16" => b.chunks(2).map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16)).collect(),
        "F32" => b.chunks(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
        d => return err(format!("{name}: a tensor of {d}")),
    };
    Ok((sh, v))
}

fn mat(ck: &Checkpoint, name: &str) -> Result<Mat> {
    if ck.get(&format!("{name}.weight_packed")).is_some() {
        let read =
            |n: String| -> Result<(Vec<u64>, Vec<u8>)> { Ok((ck.get(&n).ok_or_else(|| Error(format!("no tensor {n}")))?.1.shape.clone(), io(ck.read(&n))?)) };
        let ((sh, q), (_, e)) = (read(format!("{name}.weight_packed"))?, read(format!("{name}.weight_scale"))?);
        let (rows, cols) = (sh[0] as usize, sh[1] as usize * 2);
        if cols % 32 != 0 || q.len() != rows * cols / 2 || e.len() != rows * cols / 32 {
            return err(format!("{name}: an MXFP4 matrix of {rows} x {cols}: {} bytes and {} scales", q.len(), e.len()));
        }
        return Ok(Mat { cols, w: Vec::new(), fp4: Some((q, e)) });
    }
    let (sh, w) = tensor(ck, &format!("{name}.weight"))?;
    if sh.len() != 2 {
        return err(format!("{name}: {sh:?}, not a matrix"));
    }
    Ok(Mat { cols: sh[1] as usize, w, fp4: None })
}

fn vector(ck: &Checkpoint, name: &str) -> Result<Vec<f32>> {
    Ok(tensor(ck, name)?.1)
}

/// KimiRMSNorm: normed in fp32 and rounded to bf16, then times the weight and rounded again.
fn rms(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let r = 1.0 / ((x.iter().map(|&v| v * v).sum::<f32>() / x.len() as f32 + eps).sqrt());
    x.iter().zip(w).map(|(&v, &g)| bf16(g * bf16(v * r))).collect()
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// situ(a, b) = 4 tanh(a / 4) sigmoid(a) x 25 tanh(b / 25) (the config's two caps), in fp32, rounded to bf16.
pub fn situ(a: f32, b: f32, (beta, linear): (f32, f32)) -> f32 {
    bf16(beta * (a / beta).tanh() * sigmoid(a) * linear * (b / linear).tanh())
}

/// The depth mix of `src` (each a bf16 vector of the model's width): a query u = norm weight x proj weight, each source scored u . v / rms(v) (eps 1e-5), the
/// softmax over the sources weighting their sum, rounded to bf16.
fn mix(src: &[&[f32]], w_n: &[f32], w_p: &[f32]) -> Vec<f32> {
    let h = w_n.len();
    let s: Vec<f64> = src
        .iter()
        .map(|v| {
            let r = 1.0 / ((v.iter().map(|&x| x * x).sum::<f32>() / h as f32 + 1e-5).sqrt());
            (0..h).map(|d| (w_n[d] * w_p[d]) as f64 * v[d] as f64).sum::<f64>() * r as f64
        })
        .collect();
    let m = s.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let e: Vec<f64> = s.iter().map(|&x| (x - m).exp()).collect();
    let z: f64 = e.iter().sum();
    (0..h).map(|d| bf16(src.iter().zip(&e).map(|(v, &a)| a / z * v[d] as f64).sum::<f64>() as f32)).collect()
}

struct Kda {
    q: Mat,
    k: Mat,
    v: Mat,
    conv: [Vec<f32>; 3],
    f_a: Mat,
    f_b: Mat,
    b: Mat,
    g: Mat,
    o: Mat,
    a_log: Vec<f32>,
    dt_bias: Vec<f32>,
    o_norm: Vec<f32>,
    /// The convolutions' last inputs (q, k, v; `conv_kernel - 1` each), the state [heads, dim (keys), dim (values)].
    tail: [Vec<Vec<f32>>; 3],
    s: Vec<f32>,
}

struct Mla {
    q_a: Mat,
    q_norm: Vec<f32>,
    q_b: Mat,
    kv_a: Mat,
    kv_norm: Vec<f32>,
    kv_b: Mat,
    g: Mat,
    o: Mat,
    /// Each past position's latent, normed, and the key part every head shares [kv_rank + rope].
    keys: Vec<Vec<f32>>,
}

enum Mixer {
    Kda(Box<Kda>),
    Mla(Box<Mla>),
}

struct Moe {
    gate: Mat,
    bias: Vec<f32>,
    down: Mat,
    up: Mat,
    norm: Vec<f32>,
    experts: Vec<[Mat; 3]>,
    shared: [Mat; 3],
}

enum Ffn {
    Dense([Mat; 3]),
    Moe(Box<Moe>),
}

struct Layer {
    in_norm: Vec<f32>,
    post_norm: Vec<f32>,
    attn_res: (Vec<f32>, Vec<f32>),
    mlp_res: (Vec<f32>, Vec<f32>),
    mixer: Mixer,
    ffn: Ffn,
}

pub struct RefKimi {
    pub c: ConfigKimi,
    /// The step's closest expert choice: the smallest gap, over its layers, between the k-th choice's score (plus bias) and the next one's.
    pub margin: std::cell::Cell<f32>,
    /// Where set, each step's state after each layer: the partial sum and the completed blocks (`kimilayers`).
    pub trace: Option<Vec<(Vec<f32>, Vec<Vec<f32>>)>>,
    /// Each mixture-of-experts layer's experts (in the model's order) for the tokens to come, each token's in turn, taken in the place of its router's own
    /// choice (their weights its router's scores): a comparison's, which runs another reference on the same choices. Empty: the router's.
    pub take: std::cell::RefCell<Vec<std::collections::VecDeque<Vec<usize>>>>,
    embed: Vec<f32>,
    head: Mat,
    norm: Vec<f32>,
    out_res: (Vec<f32>, Vec<f32>),
    layers: Vec<Layer>,
}

impl RefKimi {
    pub fn load(dir: &Path) -> Result<RefKimi> {
        let c = ConfigKimi::load(dir)?;
        let ck = io(Checkpoint::open(dir))?;
        let p = c.prefix;
        let mlp = |q: &str, names: [&str; 3]| -> Result<[Mat; 3]> {
            Ok([mat(&ck, &format!("{q}.{}", names[0]))?, mat(&ck, &format!("{q}.{}", names[1]))?, mat(&ck, &format!("{q}.{}", names[2]))?])
        };
        let mut layers = Vec::new();
        for l in 0..c.layers {
            let q = format!("{p}layers.{l}");
            let a = format!("{q}.self_attn");
            let m = |n: &str| mat(&ck, &format!("{a}.{n}"));
            let mixer = match c.kinds[l] {
                GlmKind::Kda => {
                    let conv = |n: &str| vector(&ck, &format!("{a}.{n}_conv1d.weight"));
                    let w = c.kda.heads * c.kda.dim;
                    let tail = || vec![vec![0f32; w]; c.kda.conv_kernel - 1];
                    Mixer::Kda(Box::new(Kda {
                        q: m("q_proj")?,
                        k: m("k_proj")?,
                        v: m("v_proj")?,
                        conv: [conv("q")?, conv("k")?, conv("v")?],
                        f_a: m("f_a_proj")?,
                        f_b: m("f_b_proj")?,
                        b: m("b_proj")?,
                        g: m("g_proj")?,
                        o: m("o_proj")?,
                        a_log: vector(&ck, &format!("{a}.A_log"))?,
                        dt_bias: vector(&ck, &format!("{a}.dt_bias"))?,
                        o_norm: vector(&ck, &format!("{a}.o_norm.weight"))?,
                        tail: [tail(), tail(), tail()],
                        s: vec![0f32; c.kda.heads * c.kda.dim * c.kda.dim],
                    }))
                }
                GlmKind::Mla => Mixer::Mla(Box::new(Mla {
                    q_a: m("q_a_proj")?,
                    q_norm: vector(&ck, &format!("{a}.q_a_layernorm.weight"))?,
                    q_b: m("q_b_proj")?,
                    kv_a: m("kv_a_proj_with_mqa")?,
                    kv_norm: vector(&ck, &format!("{a}.kv_a_layernorm.weight"))?,
                    kv_b: m("kv_b_proj")?,
                    g: m("g_proj")?,
                    o: m("o_proj")?,
                    keys: Vec::new(),
                })),
            };
            let ffn = if c.moe[l] {
                let f = format!("{q}.block_sparse_moe");
                let experts = (0..c.experts.experts).map(|e| mlp(&format!("{f}.experts.{e}"), ["w1", "w3", "w2"])).collect::<Result<_>>()?;
                Ffn::Moe(Box::new(Moe {
                    gate: mat(&ck, &format!("{f}.gate"))?,
                    bias: vector(&ck, &format!("{f}.gate.e_score_correction_bias"))?,
                    down: mat(&ck, &format!("{f}.routed_expert_down_proj"))?,
                    up: mat(&ck, &format!("{f}.routed_expert_up_proj"))?,
                    norm: vector(&ck, &format!("{f}.routed_expert_norm.weight"))?,
                    experts,
                    shared: mlp(&format!("{f}.shared_experts"), ["gate_proj", "up_proj", "down_proj"])?,
                }))
            } else {
                Ffn::Dense(mlp(&format!("{q}.mlp"), ["gate_proj", "up_proj", "down_proj"])?)
            };
            let res = |n: &str| -> Result<(Vec<f32>, Vec<f32>)> {
                Ok((vector(&ck, &format!("{q}.{n}_res_norm.weight"))?, vector(&ck, &format!("{q}.{n}_res_proj.weight"))?))
            };
            layers.push(Layer {
                in_norm: vector(&ck, &format!("{q}.input_layernorm.weight"))?,
                post_norm: vector(&ck, &format!("{q}.post_attention_layernorm.weight"))?,
                attn_res: res("self_attention")?,
                mlp_res: res("mlp")?,
                mixer,
                ffn,
            });
        }
        Ok(RefKimi {
            embed: vector(&ck, &format!("{p}embed_tokens.weight"))?,
            head: mat(&ck, "language_model.lm_head")?,
            norm: vector(&ck, &format!("{p}norm.weight"))?,
            out_res: (vector(&ck, &format!("{p}output_attn_res_norm.weight"))?, vector(&ck, &format!("{p}output_attn_res_proj.weight"))?),
            layers,
            c,
            margin: std::cell::Cell::new(f32::INFINITY),
            trace: None,
            take: Default::default(),
        })
    }

    /// Kimi delta attention on `x` (the layer's normed input): the recurrent form, its state and the convolutions' inputs carried to the next token.
    fn kda(c: &ConfigKimi, k: &mut Kda, x: &[f32]) -> Vec<f32> {
        let (nh, dd, taps) = (c.kda.heads, c.kda.dim, c.kda.conv_kernel);
        let w = nh * dd;
        let raw = [k.q.mul(x), k.k.mul(x), k.v.mul(x)];
        // the causal convolutions (the current input times the last tap), silu, bf16
        let mut qkv: Vec<Vec<f32>> = Vec::new();
        for (j, r) in raw.iter().enumerate() {
            let y: Vec<f32> = (0..w)
                .map(|ch| {
                    let mut s = k.conv[j][ch * taps + taps - 1] * r[ch];
                    for (i, past) in k.tail[j].iter().enumerate() {
                        s += k.conv[j][ch * taps + i] * past[ch];
                    }
                    bf16(s * sigmoid(s))
                })
                .collect();
            k.tail[j].remove(0);
            k.tail[j].push(r.clone());
            qkv.push(y);
        }
        let graw = k.f_b.mul(&k.f_a.mul(x));
        let beta: Vec<f32> = k.b.mul(x).iter().map(|&b| sigmoid(b)).collect();
        let gate = k.g.mul(x);
        let scale = (dd as f32).powf(-0.5);
        let mut out = vec![0f32; w];
        for h in 0..nh {
            let norm = |v: &[f32]| -> Vec<f32> {
                let n = (v.iter().map(|&a| a * a).sum::<f32>() + 1e-6).sqrt();
                v.iter().map(|&a| a / n).collect()
            };
            let (q, kk, v) = (norm(&qkv[0][h * dd..][..dd]), norm(&qkv[1][h * dd..][..dd]), &qkv[2][h * dd..][..dd]);
            let s = &mut k.s[h * dd * dd..][..dd * dd];
            for i in 0..dd {
                let g = c.kda.bound * sigmoid(k.a_log[h].exp() * (graw[h * dd + i] + k.dt_bias[h * dd + i]));
                let a = g.exp();
                for j in 0..dd {
                    s[i * dd + j] *= a;
                }
            }
            // u = beta (v - S^T k); S += k u^T; o = scale S^T q
            let u: Vec<f32> = (0..dd).map(|j| beta[h] * (v[j] - (0..dd).map(|i| s[i * dd + j] * kk[i]).sum::<f32>())).collect();
            for i in 0..dd {
                for j in 0..dd {
                    s[i * dd + j] += kk[i] * u[j];
                }
            }
            let o: Vec<f32> = (0..dd).map(|j| bf16(scale * (0..dd).map(|i| s[i * dd + j] * q[i]).sum::<f32>())).collect();
            // the gated norm: per head over its channels, the shared weight, sigmoid of the gate, one rounding
            let r = 1.0 / ((o.iter().map(|&a| a * a).sum::<f32>() / dd as f32 + 1e-5).sqrt());
            for j in 0..dd {
                out[h * dd + j] = bf16(k.o_norm[j] * o[j] * r * sigmoid(gate[h * dd + j]));
            }
        }
        k.o.mul(&out)
    }

    /// The latent attention on `x`, in the absorbed form the engine runs (the same math as the release's, its roundings where the engine has them): the query from its
    /// latent; the token's latent c (normed) and the key part every head shares (kr, not rotated) kept; each head's query through its key rows of kv_b_proj
    /// (q' = Wk^T q_nope, fp32), its scores q' . c + q_rope . kr times 1 / sqrt(nope + rope) over every position so far, the weighted sum of the latents (fp32)
    /// through its value rows (rounded once); the output gated by sigmoid(g x), then o.
    fn mla(c: &ConfigKimi, m: &mut Mla, x: &[f32]) -> Vec<f32> {
        let d = &c.mla;
        let (nh, qw, kr) = (d.heads, d.nope + d.rope, d.kv_rank);
        let q = m.q_b.mul(&rms(&m.q_a.mul(x), &m.q_norm, 1e-6));
        let ckr = m.kv_a.mul(x);
        let mut lat = rms(&ckr[..kr], &m.kv_norm, 1e-6);
        lat.extend_from_slice(&ckr[kr..]);
        m.keys.push(lat);
        let w = &m.kv_b.w;
        let sc = 1.0 / (qw as f64).sqrt();
        let gate = m.g.mul(x);
        let mut o = vec![0f32; nh * d.v];
        for h in 0..nh {
            let qn = &q[h * qw..][..d.nope];
            let qr = &q[h * qw + d.nope..][..d.rope];
            let row = |r: usize| &w[(h * (d.nope + d.v) + r) * kr..][..kr];
            let qa: Vec<f64> = (0..kr).map(|j| (0..d.nope).map(|dd| qn[dd] as f64 * row(dd)[j] as f64).sum::<f64>() as f32 as f64).collect();
            let s: Vec<f64> = m
                .keys
                .iter()
                .map(|k| ((0..kr).map(|j| qa[j] * k[j] as f64).sum::<f64>() + (0..d.rope).map(|j| qr[j] as f64 * k[kr + j] as f64).sum::<f64>()) * sc)
                .collect();
            let mx = s.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let e: Vec<f64> = s.iter().map(|&x| (x - mx).exp()).collect();
            let z: f64 = e.iter().sum();
            let olat: Vec<f64> = (0..kr).map(|j| (m.keys.iter().zip(&e).map(|(k, &p)| p * k[j] as f64).sum::<f64>() / z) as f32 as f64).collect();
            for v in 0..d.v {
                let val = bf16(row(d.nope + v).iter().zip(&olat).map(|(&a, &b)| a as f64 * b).sum::<f64>() as f32);
                o[h * d.v + v] = bf16(val * bf16(sigmoid(gate[h * d.v + v])));
            }
        }
        m.o.mul(&o)
    }

    /// An MLP's down(situ(gate x, up x)).
    fn mlp(&self, w: &[Mat; 3], x: &[f32]) -> Vec<f32> {
        let (g, u) = (w[0].mul(x), w[1].mul(x));
        w[2].mul(&g.iter().zip(&u).map(|(&a, &b)| situ(a, b, self.c.situ)).collect::<Vec<_>>())
    }

    /// The `i`-th mixture-of-experts layer (in the model's order) on its normed input.
    fn moe(&self, i: usize, m: &Moe, x: &[f32]) -> Vec<f32> {
        let e = &self.c.experts;
        let s: Vec<f32> = m.gate.w.chunks(m.gate.cols).map(|r| sigmoid(r.iter().zip(x).map(|(&a, &b)| a as f64 * b as f64).sum::<f64>() as f32)).collect();
        let mut idx: Vec<usize> = (0..e.experts).collect();
        // (the bias chooses, the score weighs; equal choices: the lower index first)
        idx.sort_by(|&a, &b| (s[b] + m.bias[b]).partial_cmp(&(s[a] + m.bias[a])).unwrap().then(a.cmp(&b)));
        if e.top_k < e.experts {
            let gap = (s[idx[e.top_k - 1]] + m.bias[idx[e.top_k - 1]]) - (s[idx[e.top_k]] + m.bias[idx[e.top_k]]);
            self.margin.set(self.margin.get().min(gap));
        }
        idx.truncate(e.top_k);
        if let Some(t) = self.take.borrow_mut().get_mut(i).and_then(|q| q.pop_front()).filter(|t| !t.is_empty()) {
            idx = t;
        }
        let sum: f32 = idx.iter().map(|&i| s[i]).sum::<f32>() + 1e-20;
        let z = m.down.mul(x);
        let mut y = vec![0f64; e.latent];
        for &i in &idx {
            let w = if e.renorm { s[i] / sum } else { s[i] } * e.scale;
            for (acc, v) in y.iter_mut().zip(self.mlp(&m.experts[i], &z)) {
                *acc += w as f64 * v as f64;
            }
        }
        let y: Vec<f32> = y.iter().map(|&v| bf16(v as f32)).collect();
        let y = m.up.mul(&rms(&y, &m.norm, self.c.eps));
        let shared = self.mlp(&m.shared, x);
        y.iter().zip(&shared).map(|(&a, &b)| bf16(a + b)).collect()
    }

    /// One token through the model: its logits (bf16 values as f32).
    pub fn step(&mut self, token: u32) -> Vec<f32> {
        self.margin.set(f32::INFINITY);
        if let Some(t) = &mut self.trace {
            t.clear();
        }
        let h = self.c.hidden;
        let mut p: Option<Vec<f32>> = Some(self.embed[token as usize * h..][..h].to_vec());
        let mut blocks: Vec<Vec<f32>> = Vec::new();
        let add = |p: Option<Vec<f32>>, a: Vec<f32>| -> Vec<f32> {
            match p {
                None => a,
                Some(p) => p.iter().zip(&a).map(|(&x, &y)| bf16(x + y)).collect(),
            }
        };
        for l in 0..self.layers.len() {
            let pv = p.clone().expect("a partial sum before the attention");
            let x = {
                let ly = &self.layers[l];
                if blocks.is_empty() {
                    pv.clone()
                } else {
                    mix(&blocks.iter().map(|b| &b[..]).chain([&pv[..]]).collect::<Vec<_>>(), &ly.attn_res.0, &ly.attn_res.1)
                }
            };
            if l % self.c.res_block == 0 {
                blocks.push(pv);
                p = None;
            }
            let xn = rms(&x, &self.layers[l].in_norm, self.c.eps);
            let c = &self.c;
            let a = match &mut self.layers[l].mixer {
                Mixer::Kda(k) => RefKimi::kda(c, k, &xn),
                Mixer::Mla(m) => RefKimi::mla(c, m, &xn),
            };
            let ly = &self.layers[l];
            let pv = add(p.take(), a);
            let y = mix(&blocks.iter().map(|b| &b[..]).chain([&pv[..]]).collect::<Vec<_>>(), &ly.mlp_res.0, &ly.mlp_res.1);
            let yn = rms(&y, &ly.post_norm, self.c.eps);
            let m = match &ly.ffn {
                Ffn::Dense(w) => self.mlp(w, &yn),
                Ffn::Moe(m) => self.moe(self.c.moe[..l].iter().filter(|&&x| x).count(), m, &yn),
            };
            p = Some(add(Some(pv), m));
            if let Some(t) = &mut self.trace {
                t.push((p.clone().unwrap_or_default(), blocks.clone()));
            }
        }
        let pv = p.expect("the last block's partial sum");
        let hh = mix(&blocks.iter().map(|b| &b[..]).chain([&pv[..]]).collect::<Vec<_>>(), &self.out_res.0, &self.out_res.1);
        self.head.mul(&rms(&hh, &self.norm, self.c.eps))
    }
}

#[cfg(test)]
impl RefKimi {
    /// Layer `l`'s mixer on its normed input, its state carried (the runtime's kind tests, `runtime::reference::ref_tests`).
    pub(crate) fn mixer_of(&mut self, l: usize, x: &[f32]) -> Vec<f32> {
        let c = &self.c;
        match &mut self.layers[l].mixer {
            Mixer::Kda(k) => RefKimi::kda(c, k, x),
            Mixer::Mla(m) => RefKimi::mla(c, m, x),
        }
    }

    /// The depth mix of `src` with a norm and a projection weight.
    pub(crate) fn mix_of(src: &[&[f32]], w_n: &[f32], w_p: &[f32]) -> Vec<f32> {
        mix(src, w_n, w_p)
    }

    /// Layer `l`'s MLP or MoE on its normed input.
    pub(crate) fn ffn_of(&self, l: usize, x: &[f32]) -> Vec<f32> {
        match &self.layers[l].ffn {
            Ffn::Dense(w) => self.mlp(w, x),
            Ffn::Moe(m) => self.moe(self.c.moe[..l].iter().filter(|&&x| x).count(), m, x),
        }
    }
}

/// A small Kimi-K3 of the release's format in `out` (random weights; the experts MXFP4 with E8M0 scales, its config's `quantization_config` the release's
/// saying so, the rest bf16, A_log and dt_bias fp32): 8 layers in
/// blocks of 3 (so that the depth mix sees up to four sources), the latent attention at layers 3 and 7 (1-based 4 and 8), the dense MLP in layer 0, 16 experts of
/// 4 a token on a latent of 128; the delta and latent attention at the release's head widths (128; nope 128, rope 64, v 128).
///
/// `dense`: the layers before which are dense MLPs (1 as released; 8: no experts); `mla`: the latent attention layers (false: every layer delta attention), to check
/// the parts alone; `real`: Some(experts) for the release's widths instead (`random_real`).
pub fn random(out: &Path, dense: usize, mla: bool, layers: usize) -> Result<()> {
    random_with(out, dense, mla, layers, None, false, RANDOM_SEED)
}

/// The seed of the random models' draws.
pub const RANDOM_SEED: u64 = 0x2026_1021_0000_0001;

/// The case's shape (`random` of 8 layers, or `random_dsa`'s where `dsa`) drawn from `seed` (not 0): its noise floor's other draws.
pub fn random_seeded(out: &Path, dsa: bool, seed: u64) -> Result<()> {
    random_with(out, 1, true, 8, None, dsa, seed)
}

/// `random` of 8 layers with the newer config's sparse attention (`dsa_config`): on its two latent attention layers an indexer of a key a token (4 heads of 64,
/// the keys RMS-normed, the best 16 tokens attended to), its tensors as the release names them. The family's second shape.
pub fn random_dsa(out: &Path) -> Result<()> {
    random_with(out, 1, true, 8, None, true, RANDOM_SEED)
}

/// `random` at the release's widths (a width of 7,168, 96 delta heads of 128, experts of 3,072 neurons on a latent of 3,584, 16 of `experts` a token, 2 shared), every layer
/// delta attention, the first a small dense MLP as released and the others experts, a vocabulary of 4,096 (its embedding small): a model as large as the GPU's memory or larger
/// (4 layers, 3 of 320 experts: 18 GB) to run through the expert cache. Its router is not the small model's decisive one: the scores spread and a bias of 0.01 at most, so that
/// the choices spread over the experts as a trained router's do over a batch.
pub fn random_real(out: &Path, layers: usize, experts: usize) -> Result<()> {
    random_with(out, 1, false, layers.min(8), Some(experts), false, RANDOM_SEED)
}

fn random_with(out: &Path, dense: usize, mla: bool, layers: usize, real: Option<usize>, dsa: bool, seed: u64) -> Result<()> {
    let text = r#"{"architectures": ["KimiK3ForConditionalGeneration"], "model_type": "kimi_k3", "bos_token_id": 1, "eos_token_id": 2,
        "text_config": {"model_type": "kimi_linear", "hidden_act": "situ", "activation_situ_beta": 4.0, "activation_situ_linear_beta": 25.0, "attn_res_block_size": 3, "bos_token_id": 1,
        "eos_token_id": 2, "first_k_dense_replace": 1, "hidden_size": 256, "intermediate_size": 512, "kv_lora_rank": 128, "latent_moe_use_norm": true,
        "linear_attn_config": {"full_attn_layers": [4, 8], "gate_lower_bound": -5.0, "head_dim": 128, "kda_layers": [1, 2, 3, 5, 6, 7], "num_heads": 2, "short_conv_kernel_size": 4, "use_full_rank_gate": true},
        "max_position_embeddings": 4096, "mla_use_nope": true, "mla_use_output_gate": true, "moe_intermediate_size": 256, "moe_layer_freq": 1, "moe_renormalize": true,
        "moe_router_activation_func": "sigmoid", "num_attention_heads": 2, "num_expert_group": 1, "num_experts": 16, "num_experts_per_token": 4, "num_hidden_layers": 8,
        "num_key_value_heads": 2, "num_shared_experts": 1, "q_lora_rank": 128, "qk_nope_head_dim": 128, "qk_rope_head_dim": 64, "rms_norm_eps": 1e-05, "routed_expert_hidden_size": 128,
        "quantization_config": {"config_groups": {"group_0": {"format": "mxfp4-pack-quantized", "targets": ["Linear"], "weights": {"group_size": 32, "num_bits": 4,
        "strategy": "group", "symmetric": true, "type": "float"}}}, "format": "mxfp4-pack-quantized", "ignore": ["re:.*self_attn.*", "re:.*shared_experts.*",
        "re:.*mlp\\.(gate|up|gate_up|down)_proj.*", "re:.*lm_head.*"], "quant_method": "compressed-tensors"},
        "routed_scaling_factor": 1.0, "topk_group": 1, "topk_method": "noaux_tc", "v_head_dim": 128, "vocab_size": 4096}}"#;
    let text = text.replace(r#""first_k_dense_replace": 1"#, &format!(r#""first_k_dense_replace": {dense}"#));
    let text = if mla {
        text
    } else {
        text.replace(r#""full_attn_layers": [4, 8]"#, r#""full_attn_layers": []"#)
            .replace(r#""kda_layers": [1, 2, 3, 5, 6, 7]"#, r#""kda_layers": [1, 2, 3, 4, 5, 6, 7, 8]"#)
    };
    // (fewer layers: delta attention alone, the first `layers` of the list)
    let text = if layers < 8 {
        let kda: Vec<String> = (1..=layers).map(|n| n.to_string()).collect();
        text.replace(r#""num_hidden_layers": 8"#, &format!(r#""num_hidden_layers": {layers}"#))
            .replace(r#""full_attn_layers": [4, 8]"#, r#""full_attn_layers": []"#)
            .replace(r#""full_attn_layers": []"#, r#""full_attn_layers": []"#)
            .replace(r#""kda_layers": [1, 2, 3, 5, 6, 7]"#, &format!(r#""kda_layers": [{}]"#, kda.join(", ")))
            .replace(r#""kda_layers": [1, 2, 3, 4, 5, 6, 7, 8]"#, &format!(r#""kda_layers": [{}]"#, kda.join(", ")))
    } else {
        text
    };
    let text = match real {
        Some(e) => text
            .replace(r#""hidden_size": 256"#, r#""hidden_size": 7168"#)
            .replace(r#""num_heads": 2"#, r#""num_heads": 96"#)
            .replace(r#""moe_intermediate_size": 256"#, r#""moe_intermediate_size": 3072"#)
            .replace(r#""routed_expert_hidden_size": 128"#, r#""routed_expert_hidden_size": 3584"#)
            .replace(r#""num_experts": 16"#, &format!(r#""num_experts": {e}"#))
            .replace(r#""num_experts_per_token": 4"#, r#""num_experts_per_token": 16"#)
            .replace(r#""num_shared_experts": 1"#, r#""num_shared_experts": 2"#),
        None => text,
    };
    let text = if dsa {
        text.replace(
            r#""vocab_size": 4096}}"#,
            r#""vocab_size": 4096, "dsa_config": {"dsa_layers": [4, 8], "index_head_dim": 64, "index_k_norm_type": "rmsnorm", "index_num_heads": 4,
        "index_rope_dim": 0, "index_skip_topk_offset": 1, "index_topk": 16, "index_topk_freq": 1, "indexer_use_fp8": false}}}"#,
        )
    } else {
        text
    };
    let text = text.as_str();
    let c = ConfigKimi::parse(text, "random", None)?;
    if dsa != c.dsa.is_some() {
        return err("the random model's sparse attention is not in its config");
    }
    let mut r = seed;
    let mut uni = move |a: f32, b: f32| {
        r ^= r << 13;
        r ^= r >> 7;
        r ^= r << 17;
        a + (b - a) * ((r >> 40) as f32 / (1u64 << 24) as f32)
    };
    let bfw = |x: f32| {
        let b = x.to_bits();
        ((b + 0x7fff + ((b >> 16) & 1)) >> 16) as u16
    };
    let mut t: Vec<(String, &str, Vec<u64>, Vec<u8>)> = Vec::new();
    let bf = |t: &mut Vec<(String, &str, Vec<u64>, Vec<u8>)>, name: String, shape: Vec<u64>, lo: f32, hi: f32, uni: &mut dyn FnMut(f32, f32) -> f32| {
        let n: u64 = shape.iter().product();
        t.push((name, "BF16", shape, (0..n).flat_map(|_| bfw(uni(lo, hi)).to_le_bytes()).collect()));
    };
    let (h, p) = (c.hidden as u64, c.prefix);
    let lin = |rows: usize, cols: usize| (vec![rows as u64, cols as u64], (3.0 / cols as f32).sqrt());
    let m = |t: &mut Vec<(String, &str, Vec<u64>, Vec<u8>)>, name: String, rows: usize, cols: usize, uni: &mut dyn FnMut(f32, f32) -> f32| {
        let (sh, a) = lin(rows, cols);
        bf(t, format!("{name}.weight"), sh, -a, a, uni);
    };
    // (the small model's draw is held to its noise floor, `logits::tests::noise_floor_against_fp32`: the CPU reference with its rounding points against itself
    // in fp32 on the same choices, at most a tenth of the kind's bound, on this draw and three other seeds. Three scales make it so; before them the floor was
    // KL mean 1.5e-5, max 6.0e-5 (the second shape's 3.2e-4, 2.8e-3), 1e-4 to 1.8e-3 on other seeds; after them 2.3e-6 to 4.2e-6, max 1.5e-5:
    //  - `mix`: the depth mix's projections U(-0.05, 0.05), not U(-0.5, 0.5), whose scores' RMS of about 4.6 made a softmax so sharp that one rounding of a
    //    source moved a block's input by up to half of it;
    //  - `values`: the delta attention's value products 32 times a product's draw: at once the head outputs' mean square was below the gated norm's eps (1e-5)
    //    in 94% of the heads and steps, so the norm scaled instead of norming, as a real model's does not;
    //  - `decay`: its dt_bias U(-4, -2), not U(-1, 1), the state kept over tokens: at U(-1, 1) it forgot a token in about a step, so a head's output was one
    //    or two terms whose query-key products, near 0 for random queries and keys, set its direction, and a rounding of the input swung it by tens of
    //    percent. The release's model at random would show the same (none of the three is an identity: the mix still weighs its sources apart, the
    //    norm still norms, the decay still differs by channel from 0.3 to near 1).)
    let (mix, values, decay) = if real.is_none() { (0.05, 32.0, -3.0) } else { (0.5, 1.0, 0.0) };
    bf(&mut t, format!("{p}embed_tokens.weight"), vec![c.vocab as u64, h], -1.0, 1.0, &mut uni);
    bf(&mut t, "language_model.lm_head.weight".into(), vec![c.vocab as u64, h], -0.05, 0.05, &mut uni);
    for n in ["norm", "output_attn_res_norm"] {
        bf(&mut t, format!("{p}{n}.weight"), vec![h], 0.9, 1.1, &mut uni);
    }
    bf(&mut t, format!("{p}output_attn_res_proj.weight"), vec![1, h], -mix, mix, &mut uni);
    let (w, dd) = (c.kda.heads * c.kda.dim, c.kda.dim);
    for l in 0..c.layers {
        let q = format!("{p}layers.{l}");
        for n in ["input_layernorm", "post_attention_layernorm", "self_attention_res_norm", "mlp_res_norm"] {
            bf(&mut t, format!("{q}.{n}.weight"), vec![h], 0.9, 1.1, &mut uni);
        }
        for n in ["self_attention_res_proj", "mlp_res_proj"] {
            bf(&mut t, format!("{q}.{n}.weight"), vec![1, h], -mix, mix, &mut uni);
        }
        let a = format!("{q}.self_attn");
        match c.kinds[l] {
            GlmKind::Kda => {
                for n in ["q_proj", "k_proj", "v_proj", "g_proj"] {
                    let (sh, r) = lin(w, c.hidden);
                    let r = if n == "v_proj" { r * values } else { r };
                    bf(&mut t, format!("{a}.{n}.weight"), sh, -r, r, &mut uni);
                }
                // (the convolutions' and the gated norm's weights fp32, values no bf16 holds, as Kimi-K3 releases them)
                for n in ["q", "k", "v"] {
                    let len = w * c.kda.conv_kernel;
                    t.push((
                        format!("{a}.{n}_conv1d.weight"),
                        "F32",
                        vec![w as u64, 1, c.kda.conv_kernel as u64],
                        (0..len).flat_map(|_| uni(-0.5, 0.5).to_le_bytes()).collect(),
                    ));
                }
                m(&mut t, format!("{a}.f_a_proj"), dd, c.hidden, &mut uni);
                m(&mut t, format!("{a}.f_b_proj"), w, dd, &mut uni);
                m(&mut t, format!("{a}.b_proj"), c.kda.heads, c.hidden, &mut uni);
                m(&mut t, format!("{a}.o_proj"), c.hidden, w, &mut uni);
                t.push((format!("{a}.o_norm.weight"), "F32", vec![dd as u64], (0..dd).flat_map(|_| uni(0.9, 1.1).to_le_bytes()).collect()));
                for (n, len, lo, hi) in [("A_log", c.kda.heads, -1.0, 1.0), ("dt_bias", w, decay - 1.0, decay + 1.0)] {
                    t.push((format!("{a}.{n}"), "F32", vec![len as u64], (0..len).flat_map(|_| uni(lo, hi).to_le_bytes()).collect()));
                }
            }
            GlmKind::Mla => {
                let d = &c.mla;
                m(&mut t, format!("{a}.q_a_proj"), d.q_rank, c.hidden, &mut uni);
                bf(&mut t, format!("{a}.q_a_layernorm.weight"), vec![d.q_rank as u64], 0.9, 1.1, &mut uni);
                m(&mut t, format!("{a}.q_b_proj"), d.heads * (d.nope + d.rope), d.q_rank, &mut uni);
                m(&mut t, format!("{a}.kv_a_proj_with_mqa"), d.kv_rank + d.rope, c.hidden, &mut uni);
                bf(&mut t, format!("{a}.kv_a_layernorm.weight"), vec![d.kv_rank as u64], 0.9, 1.1, &mut uni);
                m(&mut t, format!("{a}.kv_b_proj"), d.heads * (d.nope + d.v), d.kv_rank, &mut uni);
                m(&mut t, format!("{a}.g_proj"), d.heads * d.v, c.hidden, &mut uni);
                m(&mut t, format!("{a}.o_proj"), c.hidden, d.heads * d.v, &mut uni);
                // (the sparse attention's indexer on the layers the config names: the queries' product of the q latent, the keys' of the layer's input, the keys'
                // norm, the head weights' product)
                if let Some(x) = c.dsa.as_ref().filter(|x| x.layers.contains(&l)) {
                    m(&mut t, format!("{a}.indexer.wq_b"), x.heads * x.dim, d.q_rank, &mut uni);
                    m(&mut t, format!("{a}.indexer.wk"), x.dim, c.hidden, &mut uni);
                    bf(&mut t, format!("{a}.indexer.k_norm.weight"), vec![x.dim as u64], 0.9, 1.1, &mut uni);
                    if x.layer_norm {
                        bf(&mut t, format!("{a}.indexer.k_norm.bias"), vec![x.dim as u64], -0.1, 0.1, &mut uni);
                    }
                    m(&mut t, format!("{a}.indexer.weights_proj"), x.heads, c.hidden, &mut uni);
                }
            }
        }
        let e = &c.experts;
        if !c.moe[l] {
            for (n, rows, cols) in [("gate_proj", c.inter, c.hidden), ("up_proj", c.inter, c.hidden), ("down_proj", c.hidden, c.inter)] {
                m(&mut t, format!("{q}.mlp.{n}"), rows, cols, &mut uni);
            }
            continue;
        }
        let f = format!("{q}.block_sparse_moe");
        // (a decisive router, so that the engine's rounding does not flip a choice: weights twenty times a product's, the scores near 0 or 1, and among those near 1 the bias
        // picks, its values 0.05 apart in a random order)
        let (sh, ga) = lin(e.experts, c.hidden);
        let g = if real.is_some() { ga } else { 20.0 * ga };
        bf(&mut t, format!("{f}.gate.weight"), sh, -g, g, &mut uni);
        let mut order: Vec<usize> = (0..e.experts).collect();
        for i in (1..order.len()).rev() {
            order.swap(i, (uni(0.0, (i + 1) as f32) as usize).min(i));
        }
        let step = if real.is_some() { 0.01 / e.experts as f32 } else { 0.05 };
        t.push((
            format!("{f}.gate.e_score_correction_bias"),
            "F32",
            vec![e.experts as u64],
            order.iter().flat_map(|&k| (k as f32 * step).to_le_bytes()).collect(),
        ));
        m(&mut t, format!("{f}.routed_expert_down_proj"), e.latent, c.hidden, &mut uni);
        m(&mut t, format!("{f}.routed_expert_up_proj"), c.hidden, e.latent, &mut uni);
        bf(&mut t, format!("{f}.routed_expert_norm.weight"), vec![e.latent as u64], 0.9, 1.1, &mut uni);
        let si = e.shared * e.inter;
        for (n, rows, cols) in [("gate_proj", si, c.hidden), ("up_proj", si, c.hidden), ("down_proj", c.hidden, si)] {
            m(&mut t, format!("{f}.shared_experts.{n}"), rows, cols, &mut uni);
        }
        for x in 0..e.experts {
            for (wn, rows, cols) in [("w1", e.inter, e.latent), ("w3", e.inter, e.latent), ("w2", e.latent, e.inter)] {
                // random E2M1 codes; each group's scale near sqrt(3 / cols) / 3
                let packed: Vec<u8> = (0..rows * cols / 2).map(|_| uni(0.0, 256.0) as u8).collect();
                let e0 = (((3.0 / cols as f32).sqrt() / 3.0).log2().round() as i32 + 127) as f32;
                let scales: Vec<u8> = (0..rows * cols / 32).map(|_| (e0 + uni(-1.0, 1.0).round()) as u8).collect();
                t.push((format!("{f}.experts.{x}.{wn}.weight_packed"), "U8", vec![rows as u64, cols as u64 / 2], packed));
                t.push((format!("{f}.experts.{x}.{wn}.weight_scale"), "U8", vec![rows as u64, cols as u64 / 32], scales));
            }
        }
    }
    io(std::fs::create_dir_all(out))?;
    io(std::fs::write(out.join("config.json"), text))?;
    let refs: Vec<(&str, &str, Vec<u64>, &[u8])> = t.iter().map(|(n, d, s, b)| (n.as_str(), *d, s.clone(), &b[..])).collect();
    crate::pack::write_safetensors(&out.join("model.safetensors"), &[("format", "pt".to_string())], &refs)?;
    println!("{}: a random Kimi-K3 model, {} tensors, {:.0} MB", out.display(), t.len(), t.iter().map(|t| t.3.len()).sum::<usize>() as f64 / 1e6);
    Ok(())
}

/// `kimiref DIR [--tokens N]`: the reference's continuation of a fixed list of token ids then its own greedy picks, its top logit a step.
pub fn cmd(a: &crate::args::Args) -> Result<()> {
    let dir = Path::new(a.need(1, "DIR")?);
    let n: usize = a.num("tokens", 32)?;
    let mut m = RefKimi::load(dir)?;
    let mut tok = 1u32;
    let t0 = std::time::Instant::now();
    for i in 0..n {
        let lg = m.step(tok);
        let (best, v) = lg.iter().enumerate().fold((0, f32::NEG_INFINITY), |b, (i, &v)| if v > b.1 { (i, v) } else { b });
        println!("{i:4} token {tok:5} -> {best:5} ({v:.4})");
        tok = if i < n / 2 { (i as u32 * 37 + 11) % m.c.vocab as u32 } else { best as u32 };
    }
    println!("{n} tokens in {:.1} s", t0.elapsed().as_secs_f64());
    Ok(())
}

/// `kimicheck DIR [--tokens N] [--ctx C] [--chunk K] [--no-graph] [--split CUTS --devices GPUS]`: the dynamic engine against this reference on the same tokens (a fixed
/// list, then the reference's own choices): each decode step's best token and the largest logit difference, then the whole sequence again as one prompt in chunks.
pub fn check(a: &crate::args::Args) -> Result<()> {
    let dir = Path::new(a.need(1, "DIR")?);
    let n: usize = a.num("tokens", 64)?;
    let o = crate::dynamic::Opts { ctx: a.num("ctx", 1024)?, chunk: a.num("chunk", 64)?, graph: !a.has("no-graph"), device: 0, layout: a.layout()? };
    let mut r = RefKimi::load(dir)?;
    let mut e = match crate::dynamic::split::tp_args(a)? {
        Some(gpus) => crate::dynamic::split::open_tp(dir, &o, &gpus)?,
        None => {
            let (cuts, devices) = crate::dynamic::split::split_args(a)?;
            crate::dynamic::split::open(dir, &o, &cuts, &devices)?
        }
    };
    println!("{}", e.describe());
    let best = |l: &[f32]| l.iter().enumerate().fold((0usize, f32::NEG_INFINITY), |b, (i, &v)| if v > b.1 { (i, v) } else { b });
    let (mut toks, mut worst, mut agree, mut tok, mut margins) = (Vec::new(), 0f32, 0usize, 1u32, Vec::new());
    for i in 0..n {
        toks.push(tok);
        let lr = r.step(tok);
        if i == 0 {
            e.prefill(&[tok])?;
        } else {
            e.step(tok)?;
        }
        let lg = e.logits()?;
        let (br, bg) = (best(&lr), best(&lg));
        let diff = lr.iter().zip(&lg).fold(0f32, |m, (a, b)| if (a - b).is_nan() { f32::INFINITY } else { m.max((a - b).abs()) });
        worst = worst.max(diff);
        agree += (br.0 == bg.0) as usize;
        let mg = r.margin.get();
        if i == 0 {
            // (where the engine's logits are not all numbers: how many are NaN, and the rows where they start)
            let bad: Vec<usize> = lg.iter().enumerate().filter(|(_, v)| !v.is_finite()).map(|(k, _)| k).collect();
            if !bad.is_empty() {
                println!("engine logits: {} of {} not finite, from {:?}", bad.len(), lg.len(), &bad[..bad.len().min(4)]);
            }
        }
        if i < 6 || i % 16 == 15 || br.0 != bg.0 || diff > 0.3 {
            println!(
                "{i:4} token {tok:5}: reference {:5} ({:.4}), engine {:5} ({:.4}), largest difference {diff:.4}; closest expert choice {mg:.5}",
                br.0, br.1, bg.0, bg.1
            );
        }
        margins.push((diff, mg));
        tok = if i < n / 2 { (i as u32 * 37 + 11) % r.c.vocab as u32 } else { br.0 as u32 };
    }
    println!("decode: {agree} of {n} steps choose the reference's token; the largest logit difference {worst:.4}");
    // (are the steps that differ the ones with a near tie among the experts?)
    let (big, small): (Vec<&(f32, f32)>, Vec<&(f32, f32)>) = margins.iter().partition(|m| m.0 > 0.3);
    let median = |mut v: Vec<f32>| -> f32 {
        v.sort_by(f32::total_cmp);
        v.get(v.len() / 2).copied().unwrap_or(f32::NAN)
    };
    println!(
        "steps differing past 0.3: {} (median closest choice {:.5}); the others {} ({:.5})",
        big.len(),
        median(big.iter().map(|m| m.1).collect()),
        small.len(),
        median(small.iter().map(|m| m.1).collect())
    );
    e.reset()?;
    e.prefill(&toks)?;
    let lg = e.logits()?;
    let mut r2 = RefKimi::load(dir)?;
    let mut lr = Vec::new();
    for &t in &toks {
        lr = r2.step(t);
    }
    let diff = lr.iter().zip(&lg).fold(0f32, |m, (a, b)| if (a - b).is_nan() { f32::INFINITY } else { m.max((a - b).abs()) });
    println!("prompt of {} in chunks of {}: reference {} engine {}, largest logit difference {diff:.4}", toks.len(), o.chunk, best(&lr).0, best(&lg).0);
    Ok(())
}

/// `kimilayers DIR [--tokens N]`: the dynamic engine's first L layers (a stage: layers 0 to L, no head) against this reference's, for each L, on the first N tokens of
/// kimicheck's list: the partial sum (X) and the completed blocks (Blk) after layer L - 1, the largest difference and the largest value.
pub fn layers(a: &crate::args::Args) -> Result<()> {
    let dir = Path::new(a.need(1, "DIR")?);
    let n: usize = a.num("tokens", 4)?;
    let c = ConfigKimi::load(dir)?;
    let o = crate::dynamic::Opts { ctx: 256, chunk: 16, graph: false, device: 0, layout: a.layout()? };
    let toks: Vec<u32> = (0..n).map(|i| if i == 0 { 1 } else { ((i as u32 - 1) * 37 + 11) % c.vocab as u32 }).collect();
    let mut r = RefKimi::load(dir)?;
    r.trace = Some(Vec::new());
    let mut cpu = Vec::new();
    for &t in &toks {
        r.step(t);
        cpu.push(r.trace.clone().unwrap_or_default());
    }
    let (h, blocks) = (c.hidden, c.layers.div_ceil(c.res_block));
    let cmp = |what: &str, g: &[f32], w: &[f32]| {
        let diff = g.iter().zip(w).fold(0f32, |m, (a, b)| if (a - b).is_nan() { f32::INFINITY } else { m.max((a - b).abs()) });
        let big = w.iter().fold(0f32, |m, v| m.max(v.abs()));
        format!("{what} {diff:.4} (of {big:.3})")
    };
    for l in 1..=c.layers {
        // (--last: the model's last stage, its head too, where l is all of them; with GLYD_STAGE_DUMP its boundary read as well)
        let last = a.has("last") && l == c.layers;
        let st = crate::dynamic::Stage { layers: (0, l), first: true, last };
        // (GLYD_KIMI_PLACES: the expert cache with that many places a layer, as kimicheck's)
        let mut e = crate::dynamic::kimi::load_with_places(dir, &o, st, crate::tune!("GLYD_KIMI_PLACES").and_then(|v| v.parse().ok()))?;
        for (i, &t) in toks.iter().enumerate() {
            let mut io = Vec::new();
            e.stage_rows(&[t], i, &mut io, last as usize)?;
            if last {
                let lg = e.logits()?;
                let hh: Vec<f32> = io[io.len() - h..].iter().map(|&w| f32::from_bits((w as u32) << 16)).collect();
                let want = r.head.mul(&hh);
                println!(
                    "token {i}: {} of {} logits not finite; H {} not finite, largest {:.3}; the head on it {:?} against {:?}",
                    lg.iter().filter(|v| !v.is_finite()).count(),
                    lg.len(),
                    hh.iter().filter(|v| !v.is_finite()).count(),
                    hh.iter().fold(0f32, |m, v| m.max(v.abs())),
                    &want[..3],
                    &lg[..3]
                );
            }
            // (io: X then Blk of the row)
            let f: Vec<f32> = io.iter().map(|&w| f32::from_bits((w as u32) << 16)).collect();
            let (p, tr) = (&cpu[i][l - 1], &f);
            let mut line = format!("layers 0..{l}, token {i}: {}", cmp("X", &tr[..h], &p.0));
            for (b, want) in p.1.iter().enumerate().take(blocks) {
                line.push_str(&format!(", {}", cmp(&format!("block {b}"), &tr[h + b * h..h + (b + 1) * h], want)));
            }
            println!("{line}");
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mxfp4_is_its_codes_times_their_scales() {
        // (row 0: codes 1 and 15 in the first byte (0.5, -6), then zeros; scale 2^(128 - 127) = 2)
        let mut packed = vec![0u8; 16];
        packed[0] = 0xF1;
        let w = mxfp4(&packed, &[128], 1, 32).unwrap();
        assert_eq!((w[0], w[1], w[2]), (1.0, -12.0, 0.0));
        assert!(mxfp4(&packed, &[128], 1, 64).is_err());
    }

    #[test]
    fn situ_caps_the_gate_and_the_up() {
        let c = (4.0, 25.0);
        // small values: about silu(a) x b; large: the caps 4 x 25 x sigmoid(a)
        assert!((situ(0.5, 2.0, c) - bf16(0.5 * sigmoid(0.5) * 2.0)).abs() < 0.02);
        assert!((situ(100.0, 1000.0, c) - 100.0).abs() < 0.5);
        assert_eq!(situ(0.0, 5.0, c), 0.0);
    }

    #[test]
    fn the_mix_of_one_source_is_it_and_of_equal_scores_their_mean() {
        let v = [1.0f32, -2.0, 3.0, 0.5];
        assert_eq!(mix(&[&v], &[1.0; 4], &[0.3; 4]), v.to_vec());
        let (a, b) = ([1.0f32, 1.0, 1.0, 1.0], [3.0f32, 3.0, 3.0, 3.0]);
        // (the same direction, so the same score: the mean)
        assert_eq!(mix(&[&a, &b], &[1.0; 4], &[1.0; 4]), vec![2.0; 4]);
    }

    /// The random model loads and steps: finite logits, the same for the same tokens from a fresh load, and its state carried (a token's logits depend on the ones before).
    #[test]
    fn the_random_model_steps() {
        let dir = std::env::temp_dir().join(format!("glyd-kimi-ref-{}", std::process::id()));
        random(&dir, 1, true, 8).unwrap();
        let run = |toks: &[u32]| -> Vec<f32> {
            let mut m = RefKimi::load(&dir).unwrap();
            toks.iter().map(|&t| m.step(t)).last().unwrap()
        };
        let a = run(&[1, 5, 9, 200, 7]);
        assert!(a.iter().all(|v| v.is_finite()) && a.iter().any(|&v| v != 0.0));
        assert_eq!(a, run(&[1, 5, 9, 200, 7]));
        assert_ne!(a, run(&[1, 6, 9, 200, 7]));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
