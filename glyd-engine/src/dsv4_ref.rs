//! The CPU reference of DeepSeek-V4 (src/deepseek_v4.rs), one sequence a token at a time, from the checkpoint's weights decoded exactly: fp32 inside (the products' sums in f64), rounded to bf16 where the
//! release's inference/model.py holds a bf16 tensor, with its simulated quantizations (the cache's FP8 channels, the indexer's FP4 and its Hadamard rotation) and, where `act_fp8`, its activations quantized to
//! FP8 before each FP8 or FP4 product as its kernels do (a block of 128 to a power-of-two scale). Its attention takes the softmax's largest score once, where the release's kernel keeps a running one over blocks of
//! 64 (the same sums but where the bf16 rounding of the weights falls). Equal indexer scores: the earlier entry first. The GPU's kernels are checked against it; `random` writes a small model of the release's
//! shape and format to check them on. A development build's only.

use crate::deepseek_v4::{fp4_runs, fp8_bf16, ConfigV4};
use crate::dev::Buf;
use crate::ffi::{err, Error, Result};
use crate::qwen35_ref::bf16;
use glyd_gpu::safetensors::Checkpoint;
use std::path::Path;

fn io<T>(r: std::io::Result<T>) -> Result<T> {
    r.map_err(|e| Error(e.to_string()))
}

/// A matrix's values [rows, cols].
struct Mat {
    cols: usize,
    w: Vec<f32>,
}

impl Mat {
    /// W x, the sums in f64, each output rounded to f32.
    fn mul(&self, x: &[f32]) -> Vec<f32> {
        assert_eq!(x.len(), self.cols);
        self.w.chunks(self.cols).map(|r| r.iter().zip(x).map(|(&a, &b)| a as f64 * b as f64).sum::<f64>() as f32).collect()
    }

    /// Rows a..b times x.
    fn mul_rows(&self, a: usize, b: usize, x: &[f32]) -> Vec<f32> {
        self.w[a * self.cols..b * self.cols].chunks(self.cols).map(|r| r.iter().zip(x).map(|(&a, &b)| a as f64 * b as f64).sum::<f64>() as f32).collect()
    }
}

fn words(v: &[u16]) -> Vec<f32> {
    v.iter().map(|&b| f32::from_bits((b as u32) << 16)).collect()
}

fn shape(ck: &Checkpoint, name: &str) -> Result<(String, Vec<u64>)> {
    let (_, t) = ck.get(name).ok_or_else(|| Error(format!("no tensor {name}")))?;
    Ok((t.dtype.clone(), t.shape.clone()))
}

/// A tensor's values as f32, whatever its type (bf16, fp32, an FP8 matrix with its block scales, an FP4 matrix with its run scales).
fn tensor(ck: &Checkpoint, name: &str) -> Result<(Vec<u64>, Vec<f32>)> {
    let (dtype, sh) = shape(ck, name)?;
    let b = io(ck.read(name))?;
    let v = match dtype.as_str() {
        "BF16" => words(&b.chunks(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect::<Vec<_>>()),
        "F32" => b.chunks(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
        "F8_E4M3" | "I8" => {
            let (rows, cols) = (sh[0] as usize, sh[1] as usize);
            let sname = name.replace(".weight", ".scale");
            let s = io(ck.read(&sname))?;
            if dtype == "I8" {
                let (code, scale) = fp4_runs(&b, &s, rows, cols * 2)?;
                let sc = words(&scale);
                return Ok((vec![rows as u64, cols as u64 * 2], code.iter().enumerate().map(|(i, &c)| c as f32 * sc[i / 32]).collect()));
            }
            words(&fp8_bf16(&b, &s, rows, cols)?)
        }
        d => return err(format!("{name}: a tensor of {d}")),
    };
    Ok((sh, v))
}

/// A tensor decoded as the reference reads it (FP8 and FP4 with their scales), for the runtime's tests.
#[cfg(test)]
pub(crate) fn tensor_of(ck: &Checkpoint, name: &str) -> Result<(Vec<u64>, Vec<f32>)> {
    tensor(ck, name)
}

fn mat(ck: &Checkpoint, name: &str) -> Result<Mat> {
    let (sh, w) = tensor(ck, name)?;
    if sh.len() != 2 {
        return err(format!("{name}: {sh:?}, not a matrix"));
    }
    Ok(Mat { cols: sh[1] as usize, w })
}

fn vector(ck: &Checkpoint, name: &str) -> Result<Vec<f32>> {
    Ok(tensor(ck, name)?.1)
}

/// model.py's RMSNorm: fp32, the weight times the normed value rounded once.
fn rms(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let r = 1.0 / ((x.iter().map(|&v| v * v).sum::<f32>() / x.len() as f32 + eps).sqrt());
    x.iter().zip(w).map(|(&v, &g)| bf16(g * (v * r))).collect()
}

/// kernel.py's power-of-two scale: 2^ceil(log2(amax / max)), from the f32 bits.
fn pow2_scale(amax: f32, inv_max: f32) -> f32 {
    let b = (amax * inv_max).to_bits();
    let e = ((b >> 23) & 0xFF) as i32 - 127 + ((b & 0x7F_FFFF) != 0) as i32;
    f32::from_bits(((e + 127) as u32) << 23)
}

/// v (at most 448 in size) rounded to E4M3, to nearest and ties to even.
pub fn e4m3_round(v: f32) -> f32 {
    let a = v.abs();
    let e = (((a.to_bits() >> 23) & 0xFF) as i32 - 127).max(-6);
    let q = f32::from_bits(((e - 3 + 127) as u32) << 23);
    ((a / q).round_ties_even() * q).copysign(v)
}

/// v (at most 6 in size) rounded to E2M1, to nearest and ties to even.
pub fn e2m1_round(v: f32) -> f32 {
    let a = v.abs();
    let q = if a < 2.0 {
        0.5
    } else if a < 4.0 {
        1.0
    } else {
        2.0
    };
    ((a / q).round_ties_even() * q).copysign(v)
}

/// kernel.py's act_quant in place (FP8, blocks of `block`, power-of-two scales): each value rounded to E4M3 at its block's scale, back as bf16.
pub fn fp8_sim(x: &mut [f32], block: usize) {
    for b in x.chunks_mut(block) {
        let amax = b.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-4);
        let s = pow2_scale(amax, 1.0 / 448.0);
        for v in b {
            *v = bf16(e4m3_round((*v / s).clamp(-448.0, 448.0)) * s);
        }
    }
}

/// kernel.py's fp4_act_quant in place (E2M1, blocks of 32, power-of-two scales).
pub fn fp4_sim(x: &mut [f32]) {
    for b in x.chunks_mut(32) {
        let amax = b.iter().fold(0f32, |m, v| m.max(v.abs())).max(6.0 * f32::from_bits(1 << 23));
        let s = pow2_scale(amax, 1.0 / 6.0);
        for v in b {
            *v = bf16(e2m1_round((*v / s).clamp(-6.0, 6.0)) * s);
        }
    }
}

/// The Walsh-Hadamard transform in natural order times n^-1/2 (rotate_activation), fp32 inside, rounded to bf16.
pub fn hadamard(x: &mut [f32]) {
    let n = x.len();
    let mut h = 1;
    while h < n {
        for i in (0..n).step_by(2 * h) {
            for j in i..i + h {
                let (a, b) = (x[j], x[j + h]);
                x[j] = a + b;
                x[j + h] = a - b;
            }
        }
        h *= 2;
    }
    let s = (n as f32).powf(-0.5);
    for v in x {
        *v = bf16(*v * s);
    }
}

/// The rotary on x's channel pairs (2i, 2i + 1) at position `pos` (apply_rotary_emb; `inverse`: by the conjugate), fp32 inside, rounded to bf16. The angle pos x f in fp32, its cosine and sine in f64.
fn rope(x: &mut [f32], f: &[f32], pos: usize, inverse: bool) {
    for (i, &fr) in f.iter().enumerate() {
        let ang = pos as f32 * fr;
        let (sn, cs) = (ang as f64).sin_cos();
        let (c, s) = (cs as f32, if inverse { -(sn as f32) } else { sn as f32 });
        let (a, b) = (x[2 * i], x[2 * i + 1]);
        x[2 * i] = bf16(a * c - b * s);
        x[2 * i + 1] = bf16(a * s + b * c);
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// A block's hyper-connection weights: the projection [24 or 4, 4D] (fp32), its base and its scales.
struct Hc {
    f: Mat,
    base: Vec<f32>,
    scale: Vec<f32>,
}

impl Hc {
    fn load(ck: &Checkpoint, p: &str) -> Result<Hc> {
        Ok(Hc { f: mat(ck, &format!("{p}_fn"))?, base: vector(ck, &format!("{p}_base"))?, scale: vector(ck, &format!("{p}_scale"))? })
    }

    /// The projection of the streams normed without a weight.
    fn mixes(&self, x: &[f32], eps: f32) -> Vec<f32> {
        let r = 1.0 / ((x.iter().map(|&v| v * v).sum::<f32>() / x.len() as f32 + eps).sqrt());
        self.f.mul(x).iter().map(|&m| m * r).collect()
    }

    /// hc_pre: the collapse of the streams into a layer's input (bf16), and the spreading weights and the mixing matrix (Sinkhorn-Knopp) for hc_post.
    fn pre(&self, x: &[f32], d: usize, eps: f32, hc_eps: f32, iters: usize) -> (Vec<f32>, [f32; 4], [[f32; 4]; 4]) {
        let m = self.mixes(x, eps);
        let (s, b) = (&self.scale, &self.base);
        let pre: Vec<f32> = (0..4).map(|j| sigmoid(m[j] * s[0] + b[j]) + hc_eps).collect();
        let mut post = [0f32; 4];
        for j in 0..4 {
            post[j] = 2.0 * sigmoid(m[4 + j] * s[1] + b[4 + j]);
        }
        let mut c = [[0f32; 4]; 4];
        for j in 0..4 {
            let z: Vec<f32> = (0..4).map(|k| m[8 + 4 * j + k] * s[2] + b[8 + 4 * j + k]).collect();
            let mx = z.iter().fold(f32::NEG_INFINITY, |a, &v| a.max(v));
            let e: Vec<f32> = z.iter().map(|&v| (v - mx).exp()).collect();
            let sum: f32 = e.iter().sum();
            for k in 0..4 {
                c[j][k] = e[k] / sum + hc_eps;
            }
        }
        let cols = |c: &mut [[f32; 4]; 4]| {
            for k in 0..4 {
                let s: f32 = (0..4).map(|j| c[j][k]).sum();
                for row in c.iter_mut() {
                    row[k] /= s + hc_eps;
                }
            }
        };
        cols(&mut c);
        for _ in 1..iters {
            for row in c.iter_mut() {
                let s: f32 = row.iter().sum();
                for v in row.iter_mut() {
                    *v /= s + hc_eps;
                }
            }
            cols(&mut c);
        }
        let y = (0..d).map(|i| bf16((0..4).map(|j| pre[j] * x[j * d + i]).sum())).collect();
        (y, post, c)
    }

    /// hc_head: the collapse by sigmoid weights alone.
    fn head(&self, x: &[f32], d: usize, eps: f32, hc_eps: f32) -> Vec<f32> {
        let m = self.mixes(x, eps);
        let pre: Vec<f32> = (0..4).map(|j| sigmoid(m[j] * self.scale[0] + self.base[j]) + hc_eps).collect();
        (0..d).map(|i| bf16((0..4).map(|j| pre[j] * x[j * d + i]).sum())).collect()
    }
}

/// hc_post: each stream k from the block's output o and the streams before: post_k o + sum_j comb[j][k] x_j, rounded to bf16.
fn hc_post(x: &mut [f32], o: &[f32], post: &[f32; 4], c: &[[f32; 4]; 4]) {
    let d = o.len();
    let old = x.to_vec();
    for k in 0..4 {
        for i in 0..d {
            x[k * d + i] = bf16(post[k] * o[i] + (0..4).map(|j| c[j][k] * old[j * d + i]).sum::<f32>());
        }
    }
}

/// A compressor: every `ratio` tokens pooled by a softmax over learned scores into one entry of `d` (ratio 4: each pool reaches over the previous window too, the first half of the projections' columns for it).
struct Comp {
    wkv: Mat,
    wgate: Mat,
    ape: Vec<f32>,
    norm: Vec<f32>,
    d: usize,
    ratio: usize,
    overlap: bool,
    rotate: bool,
    /// [coff x ratio, coff x d]: the previous window's rows (where overlapping) then the current window's
    kv: Vec<f32>,
    score: Vec<f32>,
    cache: Vec<Vec<f32>>,
}

impl Comp {
    fn load(ck: &Checkpoint, p: &str, d: usize, ratio: usize, rotate: bool) -> Result<Comp> {
        let coff = 1 + (ratio == 4) as usize;
        let n = coff * ratio * coff * d;
        Ok(Comp {
            wkv: mat(ck, &format!("{p}.wkv.weight"))?,
            wgate: mat(ck, &format!("{p}.wgate.weight"))?,
            ape: vector(ck, &format!("{p}.ape"))?,
            norm: vector(ck, &format!("{p}.norm.weight"))?,
            d,
            ratio,
            overlap: ratio == 4,
            rotate,
            kv: vec![0.0; n],
            score: vec![f32::NEG_INFINITY; n],
            cache: Vec::new(),
        })
    }

    /// Token `pos`'s input x (the attention's input, bf16): where it completes a window, the window's entry into the cache.
    fn step(&mut self, x: &[f32], pos: usize, f: &[f32], rd: usize, eps: f32, sim: bool) {
        let (r, d) = (self.ratio, self.d);
        let w = (1 + self.overlap as usize) * d;
        let kv = self.wkv.mul(x);
        let sc: Vec<f32> = self.wgate.mul(x).iter().zip(&self.ape[(pos % r) * w..][..w]).map(|(a, b)| a + b).collect();
        let row = if self.overlap { r + pos % r } else { pos % r };
        self.kv[row * w..][..w].copy_from_slice(&kv);
        self.score[row * w..][..w].copy_from_slice(&sc);
        if (pos + 1) % r != 0 {
            return;
        }
        // the rows pooled for each channel c: overlapping, the previous window's first half then the current's second half
        let rows: Vec<(usize, usize)> =
            if self.overlap { (0..r).map(|j| (j, 0)).chain((r..2 * r).map(|j| (j, d))).collect() } else { (0..r).map(|j| (j, 0)).collect() };
        let mut out: Vec<f32> = (0..d)
            .map(|c| {
                let s: Vec<f32> = rows.iter().map(|&(j, o)| self.score[j * w + o + c]).collect();
                let mx = s.iter().fold(f32::NEG_INFINITY, |a, &v| a.max(v));
                let e: Vec<f32> = s.iter().map(|&v| (v - mx).exp()).collect();
                let sum: f32 = e.iter().sum();
                rows.iter().zip(&e).map(|(&(j, o), &ev)| self.kv[j * w + o + c] * (ev / sum)).sum::<f32>()
            })
            .collect();
        if self.overlap {
            self.kv.copy_within(r * w..2 * r * w, 0);
            self.score.copy_within(r * w..2 * r * w, 0);
        }
        for v in &mut out {
            *v = bf16(*v);
        }
        let mut out = rms(&out, &self.norm, eps);
        rope(&mut out[d - rd..], f, pos + 1 - r, false);
        if self.rotate {
            hadamard(&mut out);
            if sim {
                fp4_sim(&mut out);
            }
        } else if sim {
            fp8_sim(&mut out[..d - rd], 64);
        }
        self.cache.push(out);
    }
}

struct Indexer {
    wq_b: Mat,
    wproj: Mat,
    comp: Comp,
}

struct Attn {
    wq_a: Mat,
    q_norm: Vec<f32>,
    wq_b: Mat,
    wkv: Mat,
    kv_norm: Vec<f32>,
    wo_a: Mat,
    wo_b: Mat,
    sink: Vec<f32>,
    comp: Option<Comp>,
    idx: Option<Indexer>,
    freqs: Vec<f32>,
    /// the window's keys, by position modulo the window
    win: Vec<Vec<f32>>,
}

struct Moe {
    gate: Mat,
    bias: Option<Vec<f32>>,
    tid2eid: Option<Vec<i64>>,
    experts: Vec<[Mat; 3]>,
    shared: [Mat; 3],
}

struct Layer {
    hc_attn: Hc,
    hc_ffn: Hc,
    attn_norm: Vec<f32>,
    ffn_norm: Vec<f32>,
    attn: Attn,
    moe: Moe,
}

pub struct RefV4 {
    pub c: ConfigV4,
    embed: Mat,
    head: Mat,
    norm: Vec<f32>,
    hc_head: Hc,
    layers: Vec<Layer>,
    pub pos: usize,
    act_fp8: bool,
    /// the caches' and the indexer's simulated quantizations (off with the activations': a check of the arithmetic without the roundings)
    sim: bool,
    /// the closest choice of the last step: the smallest gap, over its layers, between the last expert chosen and the first left out (in the scores that rank them), and between the last entry the
    /// indexer picked and the first it left (relative to the scores' size)
    pub margin: std::cell::Cell<(f32, f32)>,
    /// each layer's attention input (normed) and output, its experts' input (normed) and output, and the streams after it, of the last step, where kept
    pub trace: Option<Vec<[Vec<f32>; 5]>>,
}

impl RefV4 {
    pub fn load(dir: &Path, act_fp8: bool) -> Result<RefV4> {
        let c = ConfigV4::load(dir)?;
        let ck = io(Checkpoint::open(dir))?;
        let mut layers = Vec::new();
        for l in 0..c.layers {
            let p = format!("layers.{l}");
            let a = format!("{p}.attn");
            let ratio = c.ratios[l];
            let comp = if ratio > 0 { Some(Comp::load(&ck, &format!("{a}.compressor"), c.head_dim, ratio, false)?) } else { None };
            let idx = if ratio == 4 {
                Some(Indexer {
                    wq_b: mat(&ck, &format!("{a}.indexer.wq_b.weight"))?,
                    wproj: mat(&ck, &format!("{a}.indexer.weights_proj.weight"))?,
                    comp: Comp::load(&ck, &format!("{a}.indexer.compressor"), c.idx_dim, ratio, true)?,
                })
            } else {
                None
            };
            let attn = Attn {
                wq_a: mat(&ck, &format!("{a}.wq_a.weight"))?,
                q_norm: vector(&ck, &format!("{a}.q_norm.weight"))?,
                wq_b: mat(&ck, &format!("{a}.wq_b.weight"))?,
                wkv: mat(&ck, &format!("{a}.wkv.weight"))?,
                kv_norm: vector(&ck, &format!("{a}.kv_norm.weight"))?,
                wo_a: mat(&ck, &format!("{a}.wo_a.weight"))?,
                wo_b: mat(&ck, &format!("{a}.wo_b.weight"))?,
                sink: vector(&ck, &format!("{a}.attn_sink"))?,
                comp,
                idx,
                freqs: c.freqs(l),
                win: vec![Vec::new(); c.window],
            };
            let f = format!("{p}.ffn");
            let ex = |e: &str| -> Result<[Mat; 3]> {
                Ok([mat(&ck, &format!("{e}.w1.weight"))?, mat(&ck, &format!("{e}.w2.weight"))?, mat(&ck, &format!("{e}.w3.weight"))?])
            };
            let hash = l < c.hash_layers;
            let tid2eid = if hash {
                let b = io(ck.read(&format!("{f}.gate.tid2eid")))?;
                Some(b.chunks(8).map(|x| i64::from_le_bytes(x.try_into().unwrap())).collect())
            } else {
                None
            };
            let moe = Moe {
                gate: mat(&ck, &format!("{f}.gate.weight"))?,
                bias: if hash { None } else { Some(vector(&ck, &format!("{f}.gate.bias"))?) },
                tid2eid,
                experts: (0..c.experts).map(|e| ex(&format!("{f}.experts.{e}"))).collect::<Result<_>>()?,
                shared: ex(&format!("{f}.shared_experts"))?,
            };
            layers.push(Layer {
                hc_attn: Hc::load(&ck, &format!("{p}.hc_attn"))?,
                hc_ffn: Hc::load(&ck, &format!("{p}.hc_ffn"))?,
                attn_norm: vector(&ck, &format!("{p}.attn_norm.weight"))?,
                ffn_norm: vector(&ck, &format!("{p}.ffn_norm.weight"))?,
                attn,
                moe,
            });
        }
        Ok(RefV4 {
            embed: mat(&ck, "embed.weight")?,
            head: mat(&ck, "head.weight")?,
            norm: vector(&ck, "norm.weight")?,
            hc_head: Hc::load(&ck, "hc_head")?,
            layers,
            c,
            pos: 0,
            act_fp8,
            sim: act_fp8,
            trace: None,
            margin: std::cell::Cell::new((f32::INFINITY, f32::INFINITY)),
        })
    }

    /// A linear of an FP8 or FP4 matrix: its input quantized to FP8 first where `act_fp8`; the output rounded to bf16.
    fn q_lin(&self, w: &Mat, x: &[f32]) -> Vec<f32> {
        let mut x = x.to_vec();
        if self.act_fp8 {
            fp8_sim(&mut x, 128);
        }
        w.mul(&x).into_iter().map(bf16).collect()
    }

    fn attention(&mut self, l: usize, x: &[f32]) -> Vec<f32> {
        let (c, pos) = (&self.c, self.pos);
        let (nh, hd, rd, eps) = (c.heads, c.head_dim, c.rope_dim, c.eps);
        let a = &self.layers[l].attn;
        let qr = rms(&self.q_lin(&a.wq_a, x), &a.q_norm, eps);
        let mut q = self.q_lin(&a.wq_b, &qr);
        for h in q.chunks_mut(hd) {
            // (model.py norms q in bf16 arithmetic: the squares, their mean, the root and the product each rounded)
            let mean = bf16(h.iter().map(|&v| bf16(v * v)).sum::<f32>() / hd as f32);
            let r = bf16(1.0 / bf16(mean + eps).sqrt());
            for v in h.iter_mut() {
                *v = bf16(*v * r);
            }
            rope(&mut h[hd - rd..], &a.freqs, pos, false);
        }
        let mut kv = rms(&self.q_lin(&a.wkv, x), &a.kv_norm, eps);
        rope(&mut kv[hd - rd..], &a.freqs, pos, false);
        if self.sim {
            fp8_sim(&mut kv[..hd - rd], 64);
        }
        // the compressed entries this query reads
        let (win, ratio) = (c.window, c.ratios[l]);
        let mut picked: Vec<usize> = Vec::new();
        if ratio == 4 {
            let (act_fp8, sim) = (self.act_fp8, self.sim);
            let a = &mut self.layers[l].attn;
            let ix = a.idx.as_mut().unwrap();
            let mut qi = qr.clone();
            if act_fp8 {
                fp8_sim(&mut qi, 128);
            }
            let mut qi: Vec<f32> = ix.wq_b.mul(&qi).into_iter().map(bf16).collect();
            for h in qi.chunks_mut(c.idx_dim) {
                rope(&mut h[c.idx_dim - rd..], &a.freqs, pos, false);
                hadamard(h);
                if sim {
                    fp4_sim(h);
                }
            }
            ix.comp.step(x, pos, &a.freqs, rd, eps, sim);
            let k = (c.idx_dim as f32).powf(-0.5) * (c.idx_heads as f32).powf(-0.5);
            let wts: Vec<f32> = ix.wproj.mul(x).into_iter().map(|v| bf16(bf16(v) * k)).collect();
            let n = (pos + 1) / ratio;
            let mut sc: Vec<(f32, usize)> = (0..n)
                .map(|t| {
                    let kt = &ix.comp.cache[t];
                    let s: f32 = qi
                        .chunks(c.idx_dim)
                        .zip(&wts)
                        .map(|(qh, &w)| bf16(bf16(qh.iter().zip(kt).map(|(&a, &b)| a as f64 * b as f64).sum::<f64>() as f32).max(0.0) * w))
                        .sum();
                    (bf16(s), t)
                })
                .collect();
            sc.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
            if sc.len() > c.idx_topk {
                let (m, g) = (self.margin.get(), (sc[c.idx_topk - 1].0 - sc[c.idx_topk].0) / sc[0].0.abs().max(1e-6));
                self.margin.set((m.0, m.1.min(g)));
            }
            picked = sc.iter().take(c.idx_topk).map(|&(_, t)| t).collect();
        } else if ratio > 0 {
            picked = (0..(pos + 1) / ratio).collect();
        }
        let sim = self.sim;
        let a = &mut self.layers[l].attn;
        a.win[pos % win] = kv;
        if let Some(cp) = a.comp.as_mut() {
            cp.step(x, pos, &a.freqs, rd, eps, sim);
        }
        let a = &self.layers[l].attn;
        let mut keys: Vec<&[f32]> = (pos.saturating_sub(win - 1)..=pos).map(|p| &a.win[p % win][..]).collect();
        if let Some(cp) = &a.comp {
            keys.extend(picked.iter().map(|&t| &cp.cache[t][..]));
        }
        let scale = (hd as f32).powf(-0.5);
        let mut o = vec![0f32; nh * hd];
        for h in 0..nh {
            let qh = &q[h * hd..][..hd];
            let s: Vec<f32> = keys.iter().map(|k| (qh.iter().zip(k.iter()).map(|(&a, &b)| a as f64 * b as f64).sum::<f64>() as f32) * scale).collect();
            let mx = s.iter().fold(f32::NEG_INFINITY, |m, &v| m.max(v));
            let e: Vec<f32> = s.iter().map(|&v| (v - mx).exp()).collect();
            let den = e.iter().sum::<f32>() + (a.sink[h] - mx).exp();
            let oh = &mut o[h * hd..][..hd];
            for (k, &ev) in keys.iter().zip(&e) {
                let p = bf16(ev);
                for (ov, &kv) in oh.iter_mut().zip(k.iter()) {
                    *ov += p * kv;
                }
            }
            for v in oh.iter_mut() {
                *v = bf16(*v / den);
            }
            rope(&mut oh[hd - rd..], &a.freqs, pos, true);
        }
        // the grouped low rank: group g's heads by its rows of wo_a (bf16), then wo_b
        let (gw, ol) = (nh * hd / c.o_groups, c.o_lora);
        let mid: Vec<f32> = (0..c.o_groups).flat_map(|g| a.wo_a.mul_rows(g * ol, (g + 1) * ol, &o[g * gw..][..gw]).into_iter().map(bf16)).collect();
        self.q_lin(&a.wo_b, &mid)
    }

    fn moe(&self, l: usize, x: &[f32], token: u32) -> Vec<f32> {
        let (c, m) = (&self.c, &self.layers[l].moe);
        let scores: Vec<f32> = m.gate.mul(x).iter().map(|&v| (if v > 20.0 { v } else { v.exp().ln_1p() }).sqrt()).collect();
        let idx: Vec<usize> = match (&m.tid2eid, &m.bias) {
            (Some(t), _) => t[token as usize * c.top_k..][..c.top_k].iter().map(|&e| e as usize).collect(),
            (None, Some(b)) => {
                let mut o: Vec<usize> = (0..c.experts).collect();
                o.sort_by(|&i, &j| (scores[j] + b[j]).total_cmp(&(scores[i] + b[i])).then(i.cmp(&j)));
                let m = self.margin.get();
                let g = (scores[o[c.top_k - 1]] + b[o[c.top_k - 1]]) - (scores[o[c.top_k]] + b[o[c.top_k]]);
                self.margin.set((m.0.min(g), m.1));
                o.truncate(c.top_k);
                o
            }
            (None, None) => unreachable!(),
        };
        let sum: f32 = idx.iter().map(|&e| scores[e]).sum();
        let lim = c.swiglu_limit;
        let expert = |w: &[Mat; 3], wt: Option<f32>| -> Vec<f32> {
            let g = self.q_lin(&w[0], x);
            let u = self.q_lin(&w[2], x);
            let a: Vec<f32> = g
                .iter()
                .zip(&u)
                .map(|(&g, &u)| {
                    let (g, u) = if lim > 0.0 { (g.min(lim), u.clamp(-lim, lim)) } else { (g, u) };
                    let v = g / (1.0 + (-g).exp()) * u;
                    bf16(wt.map_or(v, |w| w * v))
                })
                .collect();
            self.q_lin(&w[1], &a)
        };
        let mut y = vec![0f32; c.hidden];
        for &e in &idx {
            let w = scores[e] / sum * c.route_scale;
            for (yv, v) in y.iter_mut().zip(expert(&m.experts[e], Some(w))) {
                *yv += v;
            }
        }
        y.iter().zip(expert(&m.shared, None)).map(|(&a, b)| bf16(a + b)).collect()
    }

    /// One token through the model at the next position: its logits (fp32).
    pub fn step(&mut self, token: u32) -> Vec<f32> {
        let (d, eps, he, it) = (self.c.hidden, self.c.eps, self.c.hc_eps, self.c.hc_iters);
        let e = &self.embed.w[token as usize * d..][..d];
        let mut x: Vec<f32> = (0..4).flat_map(|_| e.iter().copied()).collect();
        let mut trace = self.trace.as_ref().map(|_| Vec::new());
        self.margin.set((f32::INFINITY, f32::INFINITY));
        for l in 0..self.layers.len() {
            let (h, post, comb) = self.layers[l].hc_attn.pre(&x, d, eps, he, it);
            let ha = rms(&h, &self.layers[l].attn_norm, eps);
            let oa = self.attention(l, &ha);
            hc_post(&mut x, &oa, &post, &comb);
            let (h, post, comb) = self.layers[l].hc_ffn.pre(&x, d, eps, he, it);
            let h = rms(&h, &self.layers[l].ffn_norm, eps);
            let o = self.moe(l, &h, token);
            hc_post(&mut x, &o, &post, &comb);
            if let Some(t) = trace.as_mut() {
                t.push([ha, oa, h, o, x.clone()]);
            }
        }
        self.trace = trace;
        let y = rms(&self.hc_head.head(&x, d, eps, he), &self.norm, eps);
        self.pos += 1;
        self.head.mul(&y)
    }
}

/// E4M3's byte of a value it holds exactly.
fn e4m3_byte(v: f32) -> u8 {
    let (s, a) = (((v.to_bits() >> 31) as u8) << 7, v.abs());
    if a == 0.0 {
        return s;
    }
    let e = ((a.to_bits() >> 23) & 0xFF) as i32 - 127;
    if e < -6 {
        return s | (a / f32::from_bits(118 << 23)) as u8;
    }
    let m = ((a / f32::from_bits(((e + 127) as u32) << 23) - 1.0) * 8.0) as u8;
    s | (((e + 7) as u8) << 3) | m
}

/// A small DeepSeek-V4 of the release's shape and formats in `out` (FP8 with E8M0 block scales, FP4 experts with E8M0 run scales, bf16 and fp32 as it holds them; random weights): five layers (the window
/// alone, overlapping pools of 4 with an indexer, pools of 16, and again), the first routed by token, a window of 32 and an indexer picking 8, so that a few hundred tokens reach every path.
/// `real`: (layers, experts) at the release's widths (DeepSeek-V4-Flash's heads, latents, indexer, windows and compressors, the experts' sizes and 6 a token, its vocabulary; the layers' compression
/// ratios its first ones'), for timing a step on one GPU.
pub fn random(out: &Path, real: Option<(usize, usize)>) -> Result<()> {
    let real_text = real.map(|(layers, experts)| {
        let ratios: Vec<String> = (0..=layers).map(|l| if l == layers || l == 0 { "0".to_string() } else if l % 2 == 1 { "4".to_string() } else { "128".to_string() }).collect();
        format!(
            r#"{{"architectures": ["DeepseekV4ForCausalLM"], "model_type": "deepseek_v4", "hidden_size": 4096, "num_hidden_layers": {layers}, "vocab_size": 129280, "rms_norm_eps": 1e-6, "max_position_embeddings": 1048576,
        "bos_token_id": 0, "eos_token_id": 1, "num_attention_heads": 64, "num_key_value_heads": 1, "head_dim": 512, "qk_rope_head_dim": 64, "q_lora_rank": 1024, "o_lora_rank": 1024, "o_groups": 8,
        "sliding_window": 128, "compress_ratios": [{}], "index_n_heads": 64, "index_head_dim": 128, "index_topk": 512, "n_routed_experts": {experts}, "num_experts_per_tok": 6,
        "moe_intermediate_size": 2048, "n_shared_experts": 1, "num_hash_layers": 1, "routed_scaling_factor": 1.5, "swiglu_limit": 10.0, "scoring_func": "sqrtsoftplus", "topk_method": "noaux_tc",
        "norm_topk_prob": true, "hc_mult": 4, "hc_sinkhorn_iters": 20, "hc_eps": 1e-6, "rope_theta": 10000, "compress_rope_theta": 160000, "hidden_act": "silu", "tie_word_embeddings": false,
        "rope_scaling": {{"beta_fast": 32, "beta_slow": 1, "factor": 16, "original_max_position_embeddings": 65536, "type": "yarn"}}, "num_nextn_predict_layers": 0, "expert_dtype": "fp4",
        "quantization_config": {{"activation_scheme": "dynamic", "fmt": "e4m3", "quant_method": "fp8", "scale_fmt": "ue8m0", "weight_block_size": [128, 128]}}}}"#,
            ratios.join(", ")
        )
    });
    let small_text = r#"{"architectures": ["DeepseekV4ForCausalLM"], "model_type": "deepseek_v4", "hidden_size": 1024, "num_hidden_layers": 5, "vocab_size": 4096, "rms_norm_eps": 1e-6, "max_position_embeddings": 65536,
        "bos_token_id": 0, "eos_token_id": 1, "num_attention_heads": 16, "num_key_value_heads": 1, "head_dim": 512, "qk_rope_head_dim": 64, "q_lora_rank": 256, "o_lora_rank": 256, "o_groups": 4,
        "sliding_window": 32, "compress_ratios": [0, 4, 16, 4, 0, 0], "index_n_heads": 16, "index_head_dim": 128, "index_topk": 8, "n_routed_experts": 16, "num_experts_per_tok": 4,
        "moe_intermediate_size": 256, "n_shared_experts": 1, "num_hash_layers": 1, "routed_scaling_factor": 1.5, "swiglu_limit": 10.0, "scoring_func": "sqrtsoftplus", "topk_method": "noaux_tc",
        "norm_topk_prob": true, "hc_mult": 4, "hc_sinkhorn_iters": 20, "hc_eps": 1e-6, "rope_theta": 10000, "compress_rope_theta": 160000, "hidden_act": "silu", "tie_word_embeddings": false,
        "rope_scaling": {"beta_fast": 32, "beta_slow": 1, "factor": 16, "original_max_position_embeddings": 65536, "type": "yarn"}, "num_nextn_predict_layers": 0, "expert_dtype": "fp4",
        "quantization_config": {"activation_scheme": "dynamic", "fmt": "e4m3", "quant_method": "fp8", "scale_fmt": "ue8m0", "weight_block_size": [128, 128]}}"#;
    let text = real_text.as_deref().unwrap_or(small_text);
    let c = ConfigV4::parse(text, "random", None)?;
    let mut r = 0x2026_1018_0000_0001u64;
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
    let f32s = |v: Vec<f32>| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
    let bf16s = |v: Vec<f32>| v.iter().flat_map(|&x| bfw(x).to_le_bytes()).collect::<Vec<u8>>();
    // an FP8 matrix of weights about U(-a, a), a = sqrt(3 / cols): each 128 x 128 block to its power-of-two scale
    let fp8 = |t: &mut Vec<(String, &str, Vec<u64>, Vec<u8>)>, name: &str, rows: usize, cols: usize, uni: &mut dyn FnMut(f32, f32) -> f32| {
        let a = (3.0 / cols as f32).sqrt();
        let w: Vec<f32> = (0..rows * cols).map(|_| uni(-a, a)).collect();
        let (gr, gc) = (rows.div_ceil(128), cols.div_ceil(128));
        let mut q = vec![0u8; rows * cols];
        let mut s = vec![0u8; gr * gc];
        for bi in 0..gr {
            for bj in 0..gc {
                let mut amax = 0f32;
                for i in bi * 128..((bi + 1) * 128).min(rows) {
                    for j in bj * 128..((bj + 1) * 128).min(cols) {
                        amax = amax.max(w[i * cols + j].abs());
                    }
                }
                let sc = pow2_scale(amax.max(1e-4), 1.0 / 448.0);
                s[bi * gc + bj] = (sc.to_bits() >> 23) as u8;
                for i in bi * 128..((bi + 1) * 128).min(rows) {
                    for j in bj * 128..((bj + 1) * 128).min(cols) {
                        q[i * cols + j] = e4m3_byte(e4m3_round((w[i * cols + j] / sc).clamp(-448.0, 448.0)));
                    }
                }
            }
        }
        t.push((format!("{name}.weight"), "F8_E4M3", vec![rows as u64, cols as u64], q));
        t.push((format!("{name}.scale"), "F8_E8M0", vec![gr as u64, gc as u64], s));
    };
    let (d, hd, nh) = (c.hidden, c.head_dim, c.heads);
    let mut tensors_bf16: Vec<(String, Vec<u64>, f32, f32)> = vec![
        ("embed.weight".into(), vec![c.vocab as u64, d as u64], -1.0, 1.0),
        ("head.weight".into(), vec![c.vocab as u64, d as u64], -0.05, 0.05),
        ("norm.weight".into(), vec![d as u64], 0.9, 1.1),
    ];
    let mut tensors_f32: Vec<(String, Vec<u64>, f32, f32)> = vec![
        ("hc_head_fn".into(), vec![4, 4 * d as u64], -0.02, 0.02),
        ("hc_head_base".into(), vec![4], -0.5, 0.5),
        ("hc_head_scale".into(), vec![1], 0.5, 1.5),
    ];
    for l in 0..c.layers {
        let p = format!("layers.{l}");
        for b in ["attn", "ffn"] {
            tensors_f32.push((format!("{p}.hc_{b}_fn"), vec![24, 4 * d as u64], -0.02, 0.02));
            tensors_f32.push((format!("{p}.hc_{b}_base"), vec![24], -0.5, 0.5));
            tensors_f32.push((format!("{p}.hc_{b}_scale"), vec![3], 0.5, 1.5));
            tensors_bf16.push((format!("{p}.{b}_norm.weight"), vec![d as u64], 0.9, 1.1));
        }
        let a = format!("{p}.attn");
        fp8(&mut t, &format!("{a}.wq_a"), c.q_lora, d, &mut uni);
        fp8(&mut t, &format!("{a}.wq_b"), nh * hd, c.q_lora, &mut uni);
        fp8(&mut t, &format!("{a}.wkv"), hd, d, &mut uni);
        fp8(&mut t, &format!("{a}.wo_a"), c.o_groups * c.o_lora, nh * hd / c.o_groups, &mut uni);
        fp8(&mut t, &format!("{a}.wo_b"), d, c.o_groups * c.o_lora, &mut uni);
        tensors_bf16.push((format!("{a}.q_norm.weight"), vec![c.q_lora as u64], 0.9, 1.1));
        tensors_bf16.push((format!("{a}.kv_norm.weight"), vec![hd as u64], 0.9, 1.1));
        tensors_f32.push((format!("{a}.attn_sink"), vec![nh as u64], -1.0, 1.0));
        let ratio = c.ratios[l];
        let mut comp = |p: String, dd: usize| {
            let coff = 1 + (ratio == 4) as u64;
            let a = (3.0 / d as f32).sqrt();
            tensors_bf16.push((format!("{p}.wkv.weight"), vec![coff * dd as u64, d as u64], -a, a));
            tensors_bf16.push((format!("{p}.wgate.weight"), vec![coff * dd as u64, d as u64], -a, a));
            tensors_bf16.push((format!("{p}.norm.weight"), vec![dd as u64], 0.9, 1.1));
            tensors_f32.push((format!("{p}.ape"), vec![ratio as u64, coff * dd as u64], -0.5, 0.5));
        };
        if ratio > 0 {
            comp(format!("{a}.compressor"), hd);
        }
        if ratio == 4 {
            comp(format!("{a}.indexer.compressor"), c.idx_dim);
            fp8(&mut t, &format!("{a}.indexer.wq_b"), c.idx_heads * c.idx_dim, c.q_lora, &mut uni);
            let s = (3.0 / d as f32).sqrt();
            tensors_bf16.push((format!("{a}.indexer.weights_proj.weight"), vec![c.idx_heads as u64, d as u64], -s, s));
        }
        let f = format!("{p}.ffn");
        let s = (3.0 / d as f32).sqrt();
        tensors_bf16.push((format!("{f}.gate.weight"), vec![c.experts as u64, d as u64], -s, s));
        if l < c.hash_layers {
            let ids: Vec<u8> = (0..c.vocab)
                .flat_map(|_| {
                    let mut pick: Vec<i64> = Vec::new();
                    while pick.len() < c.top_k {
                        let e = (uni(0.0, c.experts as f32) as i64).min(c.experts as i64 - 1);
                        if !pick.contains(&e) {
                            pick.push(e);
                        }
                    }
                    pick.into_iter().flat_map(i64::to_le_bytes).collect::<Vec<u8>>()
                })
                .collect();
            t.push((format!("{f}.gate.tid2eid"), "I64", vec![c.vocab as u64, c.top_k as u64], ids));
        } else {
            tensors_f32.push((format!("{f}.gate.bias"), vec![c.experts as u64], -0.1, 0.1));
        }
        fp8(&mut t, &format!("{f}.shared_experts.w1"), c.inter, d, &mut uni);
        fp8(&mut t, &format!("{f}.shared_experts.w3"), c.inter, d, &mut uni);
        fp8(&mut t, &format!("{f}.shared_experts.w2"), d, c.inter, &mut uni);
        for e in 0..c.experts {
            for (w, rows, cols) in [("w1", c.inter, d), ("w3", c.inter, d), ("w2", d, c.inter)] {
                // random E2M1 codes; each run's scale near sqrt(3 / cols) / 3
                let q: Vec<u8> = (0..rows * cols / 2).map(|_| uni(0.0, 256.0) as u8).collect();
                let e0 = (((3.0 / cols as f32).sqrt() / 3.0).log2().round() as i32 + 127) as f32;
                let s: Vec<u8> = (0..rows * cols / 32).map(|_| (e0 + uni(-1.0, 1.0).round()) as u8).collect();
                t.push((format!("{f}.experts.{e}.{w}.weight"), "I8", vec![rows as u64, cols as u64 / 2], q));
                t.push((format!("{f}.experts.{e}.{w}.scale"), "F8_E8M0", vec![rows as u64, cols as u64 / 32], s));
            }
        }
    }
    for (name, shape, lo, hi) in tensors_bf16 {
        let n: u64 = shape.iter().product();
        let v = (0..n).map(|_| uni(lo, hi)).collect();
        t.push((name, "BF16", shape, bf16s(v)));
    }
    for (name, shape, lo, hi) in tensors_f32 {
        let n: u64 = shape.iter().product();
        let v = (0..n).map(|_| uni(lo, hi)).collect();
        t.push((name, "F32", shape, f32s(v)));
    }
    io(std::fs::create_dir_all(out))?;
    io(std::fs::write(out.join("config.json"), text))?;
    let refs: Vec<(&str, &str, Vec<u64>, &[u8])> = t.iter().map(|(n, d, s, b)| (n.as_str(), *d, s.clone(), &b[..])).collect();
    crate::pack::write_safetensors(&out.join("model.safetensors"), &[("format", "pt".to_string())], &refs)?;
    println!("{}: a random DeepSeek-V4 model, {} tensors, {:.0} MB", out.display(), t.len(), t.iter().map(|t| t.3.len()).sum::<usize>() as f64 / 1e6);
    Ok(())
}

/// `dsv4ref DIR [--tokens N] [--plain]`: the reference's greedy continuation of a fixed prompt (token ids), its top logits a step; `--plain` without the activations' FP8.
pub fn cmd(a: &crate::args::Args) -> Result<()> {
    let dir = Path::new(a.need(1, "DIR")?);
    let n: usize = a.num("tokens", 64)?;
    let mut m = RefV4::load(dir, !a.has("plain"))?;
    let mut tok = 2u32;
    let t0 = std::time::Instant::now();
    for i in 0..n {
        let lg = m.step(tok);
        let (best, v) = lg.iter().enumerate().fold((0, f32::NEG_INFINITY), |b, (i, &v)| if v > b.1 { (i, v) } else { b });
        println!("{i:4} token {tok:5} -> {best:5} ({v:.4})");
        tok = if i < 48 { (i as u32 * 37 + 11) % m.c.vocab as u32 } else { best as u32 };
    }
    println!("{n} tokens in {:.1} s", t0.elapsed().as_secs_f64());
    Ok(())
}

/// The engine on one GPU, or split into stages (`--split CUT,..` before those layers, `--devices 0,1,..`): DeepSeek-V4's, GLM-5.3-Flash's or Muse Spark's, by the directory's config (GLM's experts at `--layout`'s level,
/// penguin's lossless where none is given).
fn engine(dir: &Path, o: &crate::dynamic::Opts, a: &crate::args::Args) -> Result<crate::dynamic::split::Run> {
    if let Some(gpus) = crate::dynamic::split::tp_args(a)? {
        return crate::dynamic::split::open_tp(dir, o, &gpus);
    }
    let (cuts, devices) = crate::dynamic::split::split_args(a)?;
    crate::dynamic::split::open(dir, o, &cuts, &devices)
}

/// `dsv4prompt DIR [--tokens N] [--chunks 32,64,128,256] [--plain]` (with GLYD_V4_SIM=0 for --plain, as dsv4check): the reference once over a fixed list of N
/// tokens (dsv4check's), then the engine's prompt pass of the same tokens in chunks of each size: the largest difference of the last logits and both best tokens.
pub fn prompt(a: &crate::args::Args) -> Result<()> {
    let dir = Path::new(a.need(1, "DIR")?);
    let n: usize = a.num("tokens", 200)?;
    let chunks: Vec<usize> = a
        .flag("chunks")
        .unwrap_or("32,64,128,256")
        .split(',')
        .map(|x| x.trim().parse::<usize>().map_err(|_| Error(format!("--chunks {x}: not a number"))))
        .collect::<Result<_>>()?;
    let mut r = RefV4::load(dir, !a.has("plain"))?;
    let toks: Vec<u32> = (0..n).map(|i| if i == 0 { 2 } else { ((i as u32 - 1) * 37 + 11) % r.c.vocab as u32 }).collect();
    let mut lr = Vec::new();
    for &t in &toks {
        lr = r.step(t);
    }
    let best = |l: &[f32]| l.iter().enumerate().fold((0usize, f32::NEG_INFINITY), |b, (i, &v)| if v > b.1 { (i, v) } else { b });
    for &ch in &chunks {
        let o = crate::dynamic::Opts { ctx: a.num("ctx", 1024)?, chunk: ch, graph: false, device: 0, layout: a.layout()? };
        let mut e = engine(dir, &o, a)?;
        e.prefill(&toks)?;
        let lg = e.logits()?;
        let diff = lr.iter().zip(&lg).fold(0f32, |m, (x, y)| if (x - y).is_nan() { f32::INFINITY } else { m.max((x - y).abs()) });
        println!("a prompt of {n} in chunks of {ch}: reference {} engine {}, largest logit difference {diff:.4}", best(&lr).0, best(&lg).0);
    }
    Ok(())
}

/// `dsv4prof DIR [--prompt P] [--tokens N] [--ctx C]`: where a decode step's time goes, on one GPU without a graph: each stage's milliseconds a step (the engine's marks), then the kernels' launches and time
/// over the same steps; and the step with a graph (a second load).
pub fn prof(a: &crate::args::Args) -> Result<()> {
    let dir = Path::new(a.need(1, "DIR")?);
    let (np, n): (usize, usize) = (a.num("prompt", 256)?, a.num("tokens", 32)?);
    let ctx = a.num("ctx", 2048)?;
    let toks: Vec<u32> = (0..np).map(|i| (i as u32 * 37 + 11) % 4096).collect();
    for graph in [false, true] {
        let o = crate::dynamic::Opts { ctx, chunk: 64, graph, device: 0, layout: a.layout()? };
        let mut e = engine(dir, &o, a)?;
        if !graph {
            println!("{}", e.describe());
        }
        e.prefill(&toks)?;
        let crate::dynamic::split::Run::One(d) = &mut e else { return err("dsv4prof runs on one GPU") };
        let mut tok = 7u32;
        for _ in 0..3 {
            d.step(tok)?;
            tok = (tok * 31 + 5) % 4096;
        }
        if graph {
            let t0 = std::time::Instant::now();
            for _ in 0..n {
                d.step(tok)?;
                tok = (tok * 31 + 5) % 4096;
            }
            d.logits()?;
            println!("with a graph: {:.3} ms a step ({n} steps after {np} tokens)", t0.elapsed().as_secs_f64() * 1e3 / n as f64);
            break;
        }
        d.prof = Some(std::cell::RefCell::new(Vec::new()));
        d.prof_take()?;
        crate::dev::kprof_start();
        crate::trace::start(true);
        let t0 = std::time::Instant::now();
        for _ in 0..n {
            d.step(tok)?;
            tok = (tok * 31 + 5) % 4096;
        }
        d.logits()?;
        let wall = t0.elapsed().as_secs_f64() * 1e3 / n as f64;
        let ks = crate::dev::kprof_stop();
        // (the trace's markers inside the experts' operations, each the time from the marker before it)
        let mut parts: Vec<(&'static str, f64)> = Vec::new();
        for (what, us) in crate::trace::stages(&crate::trace::stop()) {
            if what.starts_with("moe: ") && what != "moe: start" {
                match parts.iter_mut().find(|x| x.0 == what) {
                    Some(x) => x.1 += us,
                    None => parts.push((what, us)),
                }
            }
        }
        let stages = d.prof_take()?;
        let total: f64 = stages.iter().map(|x| x.1).sum::<f64>() / n as f64;
        println!("without a graph: {wall:.3} ms a step; the stages {total:.3} ms:");
        let mut st = stages;
        st.sort_by(|x, y| y.1.total_cmp(&x.1));
        for (what, ms) in &st {
            println!("  {:7.3} ms {:5.1}%  {what}", ms / n as f64, 100.0 * ms / n as f64 / total);
        }
        if !parts.is_empty() {
            println!("the experts' steps, a step:");
            for (what, us) in &parts {
                println!("  {:7.3} ms  {what}", us / 1e3 / n as f64);
            }
        }
        let (launches, kus): (u64, f64) = ks.iter().fold((0, 0.0), |s, k| (s.0 + k.1, s.1 + k.2));
        println!("kernels: {:.0} launches a step, {:.3} ms of kernel time a step; the most:", launches as f64 / n as f64, kus / 1e3 / n as f64);
        for (name, c, us) in ks.iter().take(14) {
            println!("  {:7.3} ms {:6.1} a step  {name}", us / 1e3 / n as f64, *c as f64 / n as f64);
        }
    }
    Ok(())
}

/// `dsv4check DIR [--tokens N] [--ctx C] [--chunk K] [--no-graph] [--layout L] [--plain]` (--plain with GLYD_V4_SIM=0: both without the release's quantizations): the dynamic engine against this reference on the same tokens (a fixed prompt, then the reference's own choices): each decode
/// step's logits (the reference's best and the engine's, the largest difference, the difference at the reference's best), then the whole sequence again as one prompt in chunks (its last logits).
pub fn check(a: &crate::args::Args) -> Result<()> {
    let dir = Path::new(a.need(1, "DIR")?);
    let n: usize = a.num("tokens", 96)?;
    let o = crate::dynamic::Opts { ctx: a.num("ctx", 1024)?, chunk: a.num("chunk", 64)?, graph: !a.has("no-graph"), device: 0, layout: a.layout()? };
    let plain = a.has("plain");
    let mut r = RefV4::load(dir, !plain)?;
    let mut e = engine(dir, &o, a)?;
    println!("{}", e.describe());
    let best = |l: &[f32]| l.iter().enumerate().fold((0usize, f32::NEG_INFINITY), |b, (i, &v)| if v > b.1 { (i, v) } else { b });
    let (mut toks, mut worst, mut agree, mut margins) = (Vec::new(), 0f32, 0usize, Vec::new());
    let mut tok = 2u32;
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
        let diff = lr.iter().zip(&lg).fold(0f32, |m, (a, b)| m.max((a - b).abs()));
        worst = worst.max(diff);
        agree += (br.0 == bg.0) as usize;
        let (me, mi) = r.margin.get();
        if i < 8 || i % 16 == 15 || br.0 != bg.0 || diff > 0.3 {
            println!("{i:4} token {tok:5}: reference {:5} ({:.4}), engine {:5} ({:.4}), largest difference {diff:.4}, at the reference's best {:.4}; closest choices: experts {me:.5}, indexer {mi:.5}", br.0, br.1, bg.0, bg.1, (lg[br.0] - br.1).abs());
        }
        margins.push((diff, me, mi));
        tok = if i < 48 { (i as u32 * 37 + 11) % r.c.vocab as u32 } else { br.0 as u32 };
    }
    println!("decode: {agree} of {n} steps choose the reference's token; the largest logit difference {worst:.4}");
    // the steps with a difference past 0.3 against their closest expert choice: are they the near ties?
    let (big, small): (Vec<&(f32, f32, f32)>, Vec<&(f32, f32, f32)>) = margins.iter().partition(|m| m.0 > 0.3);
    let median = |v: &mut Vec<f32>| -> f32 {
        v.sort_by(f32::total_cmp);
        v.get(v.len() / 2).copied().unwrap_or(f32::NAN)
    };
    println!(
        "steps differing past 0.3: {} (their closest expert choice's median gap {:.5}); the others: {} (median {:.5})",
        big.len(),
        median(&mut big.iter().map(|m| m.1).collect::<Vec<f32>>()),
        small.len(),
        median(&mut small.iter().map(|m| m.1).collect::<Vec<f32>>())
    );
    e.reset()?;
    e.prefill(&toks)?;
    let lg = e.logits()?;
    let mut r2 = RefV4::load(dir, !plain)?;
    let mut lr = Vec::new();
    for &t in &toks {
        lr = r2.step(t);
    }
    let diff = lr.iter().zip(&lg).fold(0f32, |m, (a, b)| m.max((a - b).abs()));
    println!("prompt of {} in chunks of {}: reference {} engine {}, largest logit difference {diff:.4}", toks.len(), o.chunk, best(&lr).0, best(&lg).0);
    Ok(())
}

/// `dsv4layers DIR [--tokens N]`: the dynamic engine's first L layers (a stage of the model: layers 0 to L, no head) against this reference's, for each L, on the first N tokens of dsv4check's prompt: the
/// experts' normed input (H), their output (O) and the streams (S) of layer L - 1 after each token, the largest difference and the largest value.
pub fn layers(a: &crate::args::Args) -> Result<()> {
    let dir = Path::new(a.need(1, "DIR")?);
    let n: usize = a.num("tokens", 4)?;
    let c = ConfigV4::load(dir)?;
    let o = crate::dynamic::Opts { ctx: 256, chunk: 16, graph: false, device: 0, layout: a.layout()? };
    let toks: Vec<u32> = (0..n).map(|i| if i == 0 { 2 } else { ((i as u32 - 1) * 37 + 11) % c.vocab as u32 }).collect();
    let mut r = RefV4::load(dir, !a.has("plain"))?;
    r.trace = Some(Vec::new());
    let mut cpu = Vec::new();
    for &t in &toks {
        r.step(t);
        cpu.push(r.trace.clone().unwrap());
    }
    let d = c.hidden;
    let cmp = |what: &str, g: &[f32], w: &[f32]| {
        let diff = g.iter().zip(w).fold(0f32, |m, (a, b)| m.max((a - b).abs()));
        let big = w.iter().fold(0f32, |m, v| m.max(v.abs()));
        format!("{what} {diff:.4} (of {big:.3})")
    };
    for l in 1..=c.layers {
        let st = crate::dynamic::Stage { layers: (0, l), first: true, last: false };
        let mut e = crate::dynamic::deepseek_v4::load_stage(dir, &o, st)?;
        for (i, &t) in toks.iter().enumerate() {
            let mut io = Vec::new();
            e.stage_rows(&[t], i, &mut io, 0)?;
            // (io: the slots X, H, D, O, S of the row, those the model has)
            let f: Vec<f32> = io.iter().map(|&w| f32::from_bits((w as u32) << 16)).collect();
            let (h, ob, sv) = (&f[d..2 * d], &f[2 * d..3 * d], &f[3 * d..7 * d]);
            let tr = &cpu[i][l - 1];
            println!("layers 0..{l}, token {i}: {}, {}, {}", cmp("H", h, &tr[2]), cmp("O", ob, &tr[3]), cmp("S", sv, &tr[4]));
        }
    }
    Ok(())
}

/// `f8moecheck [--experts E] [--rows N] [--cols K] [--k K] [--tokens T] [--peaked]`: the library's FP8 experts' product (`F8Mat::moe`) on random E4M3 experts (--peaked: exponents peaked as trained
/// weights' are, so that layout H holds most patches compressed; else every byte as likely, every patch plain) and routing against the exact product on the host (each
/// pair's row the sum over its 128-column blocks of x q times the block's scale, in f64), for 1 to T tokens, gathered (X a token's) and not (X a pair's own): the largest difference in bf16 steps of the row's
/// largest value, and the time of a step of one token.
pub fn f8moe_check(a: &crate::args::Args) -> Result<()> {
    let (e, n, kc, k, tmax): (usize, usize, usize, usize, usize) =
        (a.num("experts", 32)?, a.num("rows", 512)?, a.num("cols", 384)?, a.num("k", 4)?, a.num("tokens", 64)?);
    let dev = crate::dev::Dev::new(0)?;
    let lib = crate::ffi::Lib::find()?;
    let mut r = 0x2026_1018_f8e4_0001u64;
    let mut uni = |lo: f32, hi: f32| {
        r ^= r << 13;
        r ^= r >> 7;
        r ^= r << 17;
        lo + (hi - lo) * ((r >> 40) as f32 / (1u64 << 24) as f32)
    };
    // the experts' bytes (finite E4M3), their scales, x
    let peaked = a.has("peaked");
    let q: Vec<u8> = (0..e * n * kc)
        .map(|_| {
            if peaked {
                // (exponent 11 - j, j geometric: the 7 commonest hold all but ~0.8%, as GLM's do)
                let j = (-uni(1e-7, 1.0).log2()).floor().min(11.0) as u32;
                (((uni(0.0, 2.0) as u32 & 1) << 7) | ((11 - j) << 3) | (uni(0.0, 8.0) as u32 & 7)) as u8
            } else {
                let b = (uni(0.0, 256.0) as u32 & 0xFF) as u8;
                if b & 0x7F == 0x7F {
                    b ^ 1
                } else {
                    b
                }
            }
        })
        .collect();
    let (gr, gc) = (n.div_ceil(128), kc.div_ceil(128));
    let sc: Vec<f32> = (0..e * gr * gc).map(|_| uni(0.5, 2.0) / 256.0).collect();
    let f = crate::f8::F8Mat::pack(&dev, &lib, &q, &sc, e * n, kc, true)?;
    let ws = Buf::zeroed(f.moe_workspace(&lib, e, tmax, k)?.max(16))?;
    let done = Buf::zeroed(e.min(tmax * k) * n.div_ceil(128) * 4 + 16)?;
    let s = dev.stream;
    let val = |b: u8| glyd_gpu::fp8::E4M3[b as usize] as f64;
    let mut worst = 0f64;
    for t in [1usize, 2, 7, 33, tmax] {
        for gather in [true, false] {
            let p = t * k;
            // distinct experts a token
            let mut ids = vec![0i64; p];
            for i in 0..t {
                let mut got: Vec<i64> = Vec::new();
                while got.len() < k {
                    let x = (uni(0.0, e as f32) as i64).min(e as i64 - 1);
                    if !got.contains(&x) {
                        got.push(x);
                    }
                }
                ids[i * k..i * k + k].copy_from_slice(&got);
            }
            let rows = if gather { t } else { p };
            let x: Vec<u16> = (0..rows * kc).map(|_| (bf16(uni(-1.0, 1.0)).to_bits() >> 16) as u16).collect();
            let (xb, idb, plan, y) = (Buf::new(x.len() * 2)?, Buf::new(p * 8)?, Buf::new((2 + 2 * e + p) * 4)?, Buf::zeroed(p * n * 2)?);
            xb.write(0, &x)?;
            idb.write(0, &ids)?;
            // SAFETY: device buffers of the sizes the calls take.
            unsafe {
                lib.moe_way(idb.ptr, p as i64, e as i64, plan.ptr, s)?;
                f.moe(&lib, e, xb.ptr, plan.ptr, t, k, gather, y.ptr, ws.ptr, ws.bytes, done.ptr, s)?;
            }
            dev.sync()?;
            let mut pl = vec![0i32; 2 + 2 * e + p];
            plan.read(0, &mut pl)?;
            let mut got = vec![0u16; p * n];
            y.read(0, &mut got)?;
            let order = &pl[2 + 2 * e..];
            for (row, &pair) in order.iter().enumerate() {
                let (ex, xr) = (ids[pair as usize] as usize, if gather { pair as usize / k } else { row });
                let xs = &x[xr * kc..xr * kc + kc];
                let want: Vec<f64> = (0..n)
                    .map(|o| {
                        let w = &q[(ex * n + o) * kc..(ex * n + o + 1) * kc];
                        (0..gc)
                            .map(|b| {
                                (b * 128..((b + 1) * 128).min(kc)).map(|c| f32::from_bits((xs[c] as u32) << 16) as f64 * val(w[c])).sum::<f64>()
                                    * sc[(ex * gr + o / 128) * gc + b] as f64
                            })
                            .sum()
                    })
                    .collect();
                let big = want.iter().fold(1e-30f64, |m, v| m.max(v.abs()));
                for o in 0..n {
                    let g = f32::from_bits((got[row * n + o] as u32) << 16) as f64;
                    worst = worst.max((g - want[o]).abs() / (big / 128.0));
                }
            }
        }
    }
    println!("{e} experts of {n} x {kc}, {k} a token, 1 to {tmax} tokens: the largest difference {worst:.3} bf16 steps of a row's largest value");
    // a step's time (one token)
    let (xb, idb, plan, y) = (Buf::zeroed(kc * 2)?, Buf::new(k * 8)?, Buf::new((2 + 2 * e + k) * 4)?, Buf::zeroed(k * n * 2)?);
    idb.write(0, &(0..k as i64).collect::<Vec<i64>>())?;
    let tm = crate::dev::Timer::new()?;
    // SAFETY: as above.
    let us = unsafe {
        lib.moe_way(idb.ptr, k as i64, e as i64, plan.ptr, s)?;
        for _ in 0..20 {
            f.moe(&lib, e, xb.ptr, plan.ptr, 1, k, true, y.ptr, ws.ptr, ws.bytes, done.ptr, s)?;
        }
        tm.start(s)?;
        for _ in 0..200 {
            f.moe(&lib, e, xb.ptr, plan.ptr, 1, k, true, y.ptr, ws.ptr, ws.bytes, done.ptr, s)?;
        }
        tm.stop(s)? * 1000.0 / 200.0
    };
    let mb = (k * n * kc) as f64 * (f.bytes as f64 / (e * n * kc) as f64) / 1e6;
    println!("one token: {us:.1} us for {mb:.1} MB of its experts' weights ({:.0} GB/s)", mb / 1e3 / (us as f64 / 1e6));
    Ok(())
}

/// `fp4stat SHARD [--max N]`: what an MX FP4 checkpoint's experts carry (the first N matrices of a shard): their codes' and magnitudes' histograms and order-0 entropies, a magnitude's entropy given its
/// left neighbour's and given its run's largest, the scales' entropy and that of each scale less its row's first: the bits a weight a lossless code of each kind could reach, against the release's 4.25.
pub fn fp4_stat(a: &crate::args::Args) -> Result<()> {
    let path = Path::new(a.need(1, "SHARD")?);
    let max: usize = a.num("max", 48)?;
    let ck = Checkpoint::open(path.parent().unwrap_or(Path::new("."))).map_err(|e| Error(e.to_string()))?;
    let h = |c: &[u64]| -> f64 {
        let n: u64 = c.iter().sum();
        c.iter()
            .filter(|&&x| x > 0)
            .map(|&x| {
                let p = x as f64 / n as f64;
                -p * p.log2()
            })
            .sum()
    };
    let (mut code, mut mag) = ([0u64; 16], [0u64; 8]);
    let mut nb = vec![[0u64; 8]; 8]; // [left][this] magnitudes
    let mut bymax = vec![[0u64; 8]; 8]; // [run's largest][this]
    let (mut sc, mut dsc) = (vec![0u64; 256], vec![0u64; 512]);
    let mut names: Vec<String> = ck
        .names()
        .filter(|n| n.contains(".experts.") && n.ends_with(".weight") && ck.get(&n.replace(".weight", ".scale")).is_some())
        .map(String::from)
        .collect();
    names.sort();
    let (mut weights, mut mats) = (0u64, 0usize);
    for n in names.iter().take(max) {
        let (_, t) = ck.get(n).ok_or_else(|| Error(format!("no {n}")))?;
        let (rows, cols) = (t.shape[0] as usize, t.shape[1] as usize * 2);
        let (q, e) = (ck.read(n).map_err(|x| Error(x.to_string()))?, ck.read(&n.replace(".weight", ".scale")).map_err(|x| Error(x.to_string()))?);
        if q.len() * 2 != rows * cols || e.len() * 32 != rows * cols {
            continue;
        }
        mats += 1;
        for r in 0..rows {
            let row = &q[r * cols / 2..(r + 1) * cols / 2];
            let es = &e[r * cols / 32..(r + 1) * cols / 32];
            for (j, &x) in es.iter().enumerate() {
                sc[x as usize] += 1;
                dsc[(x as i32 - es[0] as i32 + 256) as usize] += 1;
                let run: Vec<u8> = row[j * 16..j * 16 + 16].iter().flat_map(|&b| [b & 15, b >> 4]).collect();
                let big = run.iter().map(|c| c & 7).max().unwrap_or(0) as usize;
                for (i, &c) in run.iter().enumerate() {
                    code[c as usize] += 1;
                    mag[(c & 7) as usize] += 1;
                    bymax[big][(c & 7) as usize] += 1;
                    if i > 0 {
                        nb[(run[i - 1] & 7) as usize][(c & 7) as usize] += 1;
                    }
                }
            }
        }
        weights += (rows * cols) as u64;
    }
    let cond = |m: &Vec<[u64; 8]>| -> f64 {
        let tot: u64 = m.iter().flat_map(|r| r.iter()).sum();
        m.iter()
            .map(|r| {
                let n: u64 = r.iter().sum();
                n as f64 / tot as f64 * h(r)
            })
            .sum()
    };
    let total: u64 = mag.iter().sum();
    println!("{mats} matrices, {:.2}G weights", weights as f64 / 1e9);
    println!(
        "magnitudes 0..7 (0, .5, 1, 1.5, 2, 3, 4, 6): {}",
        mag.iter().map(|&x| format!("{:.2}%", 100.0 * x as f64 / total as f64)).collect::<Vec<_>>().join(" ")
    );
    let pos: u64 = code[1..8].iter().sum();
    let neg: u64 = code[9..16].iter().sum();
    println!(
        "signs: {:.3}% negative of the nonzero; zeros: {:.3}% of them -0",
        100.0 * neg as f64 / (pos + neg) as f64,
        100.0 * code[8] as f64 / (code[0] + code[8]).max(1) as f64
    );
    let (hc, hm) = (h(&code), h(&mag));
    println!("H(code) {hc:.3} bits; H(magnitude) {hm:.3}; given its left neighbour {:.3}; given its run's largest {:.3}", cond(&nb), cond(&bymax));
    let scale_bits = h(&sc) / 32.0;
    println!(
        "scales: H {:.3} bits a run ({:.4} a weight), less the row's first {:.3} ({:.4} a weight); seen {}..{}",
        h(&sc),
        scale_bits,
        h(&dsc),
        h(&dsc) / 32.0,
        sc.iter().position(|&x| x > 0).unwrap_or(0),
        255 - sc.iter().rev().position(|&x| x > 0).unwrap_or(0)
    );
    println!(
        "bits a weight: release 4.250; order-0 codes + scales {:.3}; magnitudes given the run's largest + signs (every code, -0 too) + scales {:.3}",
        hc + scale_bits,
        cond(&bymax) + 1.0 + h(&sc) / 32.0
    );
    // the scales' spread: in a matrix, a stack of a layer's experts, a row, a block of 64 rows
    let (mut lay_lo, mut lay_hi, mut mat_w, mut row_w, mut blk_w) = (255u8, 0u8, [0u64; 9], [0u64; 9], [0u64; 9]);
    for n in names.iter().take(max) {
        let (_, t) = ck.get(n).ok_or_else(|| Error(format!("no {n}")))?;
        let (rows, cols) = (t.shape[0] as usize, t.shape[1] as usize * 2);
        let e = ck.read(&n.replace(".weight", ".scale")).map_err(|x| Error(x.to_string()))?;
        if e.len() * 32 != rows * cols {
            continue;
        }
        let w = |s: &[u8]| (s.iter().max().unwrap() - s.iter().min().unwrap()).min(8) as usize;
        lay_lo = lay_lo.min(*e.iter().min().unwrap());
        lay_hi = lay_hi.max(*e.iter().max().unwrap());
        mat_w[w(&e)] += 1;
        let rl = cols / 32;
        for r in 0..rows {
            row_w[w(&e[r * rl..(r + 1) * rl])] += 1;
        }
        for b in 0..rows / 64 {
            blk_w[w(&e[b * 64 * rl..(b + 1) * 64 * rl])] += 1;
        }
    }
    println!("scale spread (largest less smallest, 0..8+): matrices {mat_w:?}; rows {row_w:?}; 64-row blocks {blk_w:?}; all {lay_lo}..{lay_hi}");
    Ok(())
}

/// `f8stat DIR [--max N]`: an FP8 checkpoint's E4M3 matrices (the first N of a directory): the bytes' order-0 entropy, and given the byte to the left, against the library's FP8 format's bits a weight
/// (packed on the GPU, as the engine holds them).
pub fn f8_stat(a: &crate::args::Args) -> Result<()> {
    let dir = Path::new(a.need(1, "DIR")?);
    let max: usize = a.num("max", 16)?;
    let ck = Checkpoint::open(dir).map_err(|e| Error(e.to_string()))?;
    let dev = crate::dev::Dev::new(0)?;
    let lib = crate::ffi::Lib::find()?;
    let h = |c: &[u64]| -> f64 {
        let n: u64 = c.iter().sum();
        c.iter()
            .filter(|&&x| x > 0)
            .map(|&x| {
                let p = x as f64 / n as f64;
                -p * p.log2()
            })
            .sum()
    };
    let mut names: Vec<String> = ck.names().filter(|n| ck.get(n).is_some_and(|(_, t)| t.dtype == "F8_E4M3" && t.shape.len() == 2)).map(String::from).collect();
    names.sort();
    let (mut all, mut nb) = (vec![0u64; 256], vec![vec![0u64; 256]; 256]);
    let (mut f8_bytes, mut weights) = (0usize, 0usize);
    for n in names.iter().take(max) {
        let (_, t) = ck.get(n).ok_or_else(|| Error(format!("no {n}")))?;
        let (rows, cols) = (t.shape[0] as usize, t.shape[1] as usize);
        let q = ck.read(n).map_err(|x| Error(x.to_string()))?;
        for r in 0..rows {
            let row = &q[r * cols..(r + 1) * cols];
            for (i, &b) in row.iter().enumerate() {
                all[b as usize] += 1;
                if i > 0 {
                    nb[row[i - 1] as usize][b as usize] += 1;
                }
            }
        }
        let sc = vec![1f32; rows.div_ceil(128) * cols.div_ceil(128)];
        let m = crate::f8::F8Mat::pack(&dev, &lib, &q, &sc, rows, cols, false)?;
        f8_bytes += m.bytes - sc.len() * 4;
        weights += rows * cols;
        println!(
            "{n} [{rows}, {cols}]: H {:.3} bits, the FP8 format {:.3}",
            h(&{
                let mut c = vec![0u64; 256];
                q.iter().for_each(|&b| c[b as usize] += 1);
                c
            }),
            (m.bytes - sc.len() * 4) as f64 * 8.0 / (rows * cols) as f64
        );
    }
    let tot: u64 = nb.iter().flat_map(|r| r.iter()).sum();
    let cond: f64 = nb
        .iter()
        .map(|r| {
            let n: u64 = r.iter().sum();
            n as f64 / tot as f64 * h(r)
        })
        .sum();
    println!(
        "{} matrices, {:.2}G weights: H(byte) {:.3} bits, given its left neighbour {cond:.3}; the FP8 format {:.3} bits a weight (the release 8)",
        names.len().min(max),
        weights as f64 / 1e9,
        h(&all),
        f8_bytes as f64 * 8.0 / weights as f64
    );
    Ok(())
}

/// `dsv4pack DIR [--layer L]`: one layer of a real checkpoint (a directory with the shard that holds it) decoded and packed as the engine loads it: each FP8 matrix to bf16 (its non-finite weights
/// counted) and packed lossless, the experts' FP4 to five-bit runs stacked and packed (their codes' spread), with each one's time and bits a weight.
pub fn pack_check(a: &crate::args::Args) -> Result<()> {
    use crate::weights::{pack_words, Layout};
    let dir = Path::new(a.need(1, "DIR")?);
    let l: usize = a.num("layer", 2)?;
    let c = ConfigV4::load(dir)?;
    let ck = io(Checkpoint::open(dir))?;
    let (p, a_) = (format!("layers.{l}"), format!("layers.{l}.attn"));
    let mut names: Vec<String> = ["wq_a", "wq_b", "wkv", "wo_a", "wo_b"].iter().map(|n| format!("{a_}.{n}")).collect();
    if c.ratios[l] == 4 {
        names.push(format!("{a_}.indexer.wq_b"));
    }
    names.extend(["w1", "w3", "w2"].iter().map(|n| format!("{p}.ffn.shared_experts.{n}")));
    let (mut dense_w, mut dense_b) = (0usize, 0usize);
    // (--gpu: each matrix's product timed in both forms on GPU 0)
    let gpu = if a.has("gpu") {
        let dev = crate::dev::Dev::new(0)?;
        let lib = crate::ffi::Lib::find()?;
        let big = names
            .iter()
            .map(|n| shape(&ck, &format!("{n}.weight")).map(|(_, s)| s[0].max(s[1]) as usize))
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .max()
            .unwrap_or(1);
        let mut need = 0usize;
        for n in &names {
            let (_, sh) = shape(&ck, &format!("{n}.weight"))?;
            need = need
                .max(lib.f8_linear_workspace(sh[0] as i64, sh[1] as i64, 1)?)
                .max(lib.linear_workspace(false, sh[0] as i64, sh[1] as i64, 1)?)
                .max(lib.linear_workspace(true, sh[0] as i64, sh[1] as i64, 1)?);
        }
        let (x, y) = (Buf::zeroed(big * 2)?, Buf::zeroed(big * 2)?);
        let (ws, done) = (Buf::zeroed(need.max(16))?, Buf::zeroed(4 << 18)?);
        Some((dev, lib, x, y, ws, done))
    } else {
        None
    };
    for n in &names {
        let (_, sh) = shape(&ck, &format!("{n}.weight"))?;
        let (rows, cols) = (sh[0] as usize, sh[1] as usize);
        let t0 = std::time::Instant::now();
        let w = fp8_bf16(&io(ck.read(&format!("{n}.weight")))?, &io(ck.read(&format!("{n}.scale")))?, rows, cols)?;
        let bad = w.iter().filter(|&&x| x & 0x7F80 == 0x7F80).count();
        let t1 = t0.elapsed().as_secs_f64();
        let pk = pack_words(&w, rows, cols, Layout::Tiered, n)?;
        let bytes = pk.data.len() + pk.b1.len() + pk.b2.len();
        println!(
            "{n}: {rows} x {cols}, {bad} not finite, decoded in {t1:.2} s, packed lossless in {:.2} s: {:.2} bits a weight",
            t0.elapsed().as_secs_f64() - t1,
            bytes as f64 * 8.0 / (rows * cols) as f64
        );
        if let Some(g) = &gpu {
            // (a token's product of each form, 200 times, by events: the time, the bytes it reads, the rate)
            let (dev, lib, x, y, ws, done) = g;
            let q = io(ck.read(&format!("{n}.weight")))?;
            let sc: Vec<f32> = io(ck.read(&format!("{n}.scale")))?.iter().map(|&b| crate::deepseek_v4::e8m0(b)).collect();
            // (copies enough that a round of them is past the L2 cache (256 MB), taken in turn: each product reads its matrix from memory, as a token's pass does)
            let copies = (256 << 20) / bytes.max(1) + 2;
            let fs: Vec<crate::f8::F8Mat> = (0..copies).map(|_| crate::f8::F8Mat::pack(dev, lib, &q, &sc, rows, cols, false)).collect::<Result<_>>()?;
            let pbs: Vec<crate::weights::PackBuf> =
                (0..copies).map(|_| crate::weights::upload_packed(pack_words(&w, rows, cols, Layout::Tiered, n)?)).collect::<Result<_>>()?;
            let time = |go: &dyn Fn(usize) -> Result<()>| -> Result<f64> {
                for i in 0..copies {
                    go(i)?;
                }
                let (a, b) = (crate::dev::Ev::new()?, crate::dev::Ev::new()?);
                a.record(dev.stream)?;
                for i in 0..200 {
                    go(i % copies)?;
                }
                b.record(dev.stream)?;
                dev.sync()?;
                Ok(b.since(&a)? as f64 / 200.0)
            };
            // SAFETY: live device buffers sized for the largest matrix; one stream.
            let tf = time(&|i| unsafe { fs[i].linear(lib, x.ptr, 1, y.ptr, ws.ptr, ws.bytes, done.ptr, dev.stream) })?;
            let tp = time(&|i| unsafe { lib.linear(&pbs[i].pack, x.ptr, 1, y.ptr, ws.ptr, ws.bytes, done.ptr, dev.stream) })?;
            let (fb, pbb) = (fs[0].bytes, pbs[0].bytes);
            println!("    a token's product from memory ({copies} copies): FP8 format {:.1} us ({fb} bytes, {:.0} GB/s), lossless bf16 pack {:.1} us ({pbb} bytes, {:.0} GB/s)", tf * 1e3, fb as f64 / tf / 1e6, tp * 1e3, pbb as f64 / tp / 1e6);
            continue;
        }
        dense_w += rows * cols;
        dense_b += bytes;
    }
    if gpu.is_some() {
        return Ok(());
    }
    let (e, i, d) = (c.experts, c.inter, c.hidden);
    let mut hist = [0usize; 32];
    let (mut smin, mut smax) = (u16::MAX, 0u16);
    let mut exp_b = 0usize;
    for (what, mats, rows, cols) in [("first", vec!["w1", "w3"], i, d), ("second", vec!["w2"], d, i)] {
        let t0 = std::time::Instant::now();
        let (mut codes, mut scales) = (Vec::new(), Vec::new());
        for x in 0..e {
            for m in &mats {
                let q = format!("{p}.ffn.experts.{x}.{m}");
                let (cd, sc) = fp4_runs(&io(ck.read(&format!("{q}.weight")))?, &io(ck.read(&format!("{q}.scale")))?, rows, cols)?;
                for &v in &cd {
                    hist[(v + 16) as usize] += 1;
                }
                for &v in &sc {
                    smin = smin.min(v);
                    smax = smax.max(v);
                }
                codes.extend(cd);
                scales.extend(sc);
            }
        }
        let t1 = t0.elapsed().as_secs_f64();
        let r = mats.len() * rows;
        let pk = glyd_gpu::pack::pack_int5_runs_checked(&codes, &scales, e * r, cols)
            .ok_or_else(|| Error(format!("layer {l}: the experts' {what} matrices are not five-bit runs")))?;
        let bytes = pk.data.len() + pk.blocks.len();
        exp_b += bytes;
        println!(
            "experts' {what} matrices: {e} x {r} x {cols}, read and decoded in {t1:.1} s, packed and checked in {:.1} s: {:.2} bits a weight",
            t0.elapsed().as_secs_f64() - t1,
            bytes as f64 * 8.0 / (e * r * cols) as f64
        );
    }
    let tot: usize = hist.iter().sum();
    let spread: Vec<String> = (0..32).filter(|&k| hist[k] > 0).map(|k| format!("{}:{:.1}%", k as i32 - 16, 100.0 * hist[k] as f64 / tot as f64)).collect();
    println!("expert codes: {}; run scales from {:#06x} to {:#06x} (bf16)", spread.join(" "), smin, smax);
    let layer = dense_b + exp_b;
    println!(
        "layer {l}: {:.2} GB packed ({:.2} dense at {:.2} bits, {:.2} experts); the model's {} layers about {:.0} GB with the embedding and head",
        layer as f64 / 1e9,
        dense_b as f64 / 1e9,
        dense_b as f64 * 8.0 / dense_w as f64,
        exp_b as f64 / 1e9,
        c.layers,
        (layer * c.layers) as f64 / 1e9 + 2.0 * (c.vocab * d * 2) as f64 / 1e9
    );
    Ok(())
}

/// `dsv4gen DIR (--prompt TEXT [--tokens N] | --score FILE [--out FILE]) [--split CUTS --devices GPUS] [--ctx C] [--chunk K] [--layout L] [--no-graph]`: the real model (DeepSeek-V4, GLM-5.3-Flash or Muse Spark) on the engine. A prompt
/// (raw text, the tokenizer's start token added): its greedy continuation, the prompt's time and the decode's per token. A file's text: each token's log-probability given the ones before (a token at a
/// time), their mean and the perplexity, and with --out one line a token (`id logprob`) for a comparison with another implementation's.
pub fn gen(a: &crate::args::Args) -> Result<()> {
    let dir = Path::new(a.need(1, "DIR")?);
    let o = crate::dynamic::Opts {
        ctx: a.num("ctx", 4096)?,
        chunk: a.num("chunk", 256)?,
        graph: !a.has("no-graph"),
        device: a.flag("devices").and_then(|d| d.split(',').next()).and_then(|d| d.parse().ok()).unwrap_or(0),
        layout: a.layout()?,
    };
    let tok = crate::tok::Tok::load(dir)?;
    // (the tokenizer adds no start token: a raw text begins with the config's)
    let bos = match std::fs::read_to_string(dir.join("config.json")) {
        Ok(t) if crate::muse::is_muse(&t) => crate::muse::ConfigMuse::load(dir).ok().and_then(|c| c.bos),
        _ => ConfigV4::load(dir).ok().and_then(|c| c.bos),
    };
    let encode = |t: &str| -> Result<Vec<u32>> {
        let ids = tok.encode_with(t, true)?;
        Ok(match bos {
            Some(b) if ids.first() != Some(&b) => std::iter::once(b).chain(ids).collect(),
            _ => ids,
        })
    };
    if a.has("encode") {
        // (the tokenizer alone: the prompt's ids and their text back)
        let ids = encode(a.flag("prompt").unwrap_or(""))?;
        println!("{ids:?}\n{:?}", tok.decode(&ids)?);
        return Ok(());
    }
    let t0 = std::time::Instant::now();
    let mut e = engine(dir, &o, a)?;
    eprintln!("loaded in {:.0} s: {}", t0.elapsed().as_secs_f64(), e.describe());
    let best = |l: &[f32]| l.iter().enumerate().fold((0usize, f32::NEG_INFINITY), |b, (i, &v)| if v > b.1 { (i, v) } else { b }).0 as u32;
    if let Some(f) = a.flag("score") {
        let text = std::fs::read_to_string(f).map_err(|x| Error(format!("{f}: {x}")))?;
        let ids = encode(&text)?;
        let n = ids.len().min(o.ctx - 1);
        let mut out = String::new();
        let (mut sum, t1) = (0f64, std::time::Instant::now());
        e.prefill(&ids[..1])?;
        for k in 1..n {
            let l = e.logits()?;
            let m = l.iter().fold(f32::NEG_INFINITY, |a, &v| a.max(v)) as f64;
            let lse = m + l.iter().map(|&v| (v as f64 - m).exp()).sum::<f64>().ln();
            let lp = l[ids[k] as usize] as f64 - lse;
            sum += lp;
            out.push_str(&format!("{} {lp:.6}\n", ids[k]));
            e.step(ids[k])?;
        }
        println!(
            "{} tokens of {f}: mean log-probability {:.4}, perplexity {:.3} ({:.1} ms a token)",
            n - 1,
            sum / (n - 1) as f64,
            (-sum / (n - 1) as f64).exp(),
            t1.elapsed().as_secs_f64() * 1e3 / (n - 1) as f64
        );
        if let Some(of) = a.flag("out") {
            std::fs::write(of, out).map_err(|x| Error(format!("{of}: {x}")))?;
        }
        return Ok(());
    }
    let prompt = a.flag("prompt").ok_or_else(|| Error("dsv4gen DIR --prompt TEXT, or --score FILE".into()))?;
    let n: usize = a.num("tokens", 64)?;
    let ids = encode(prompt)?;
    let t1 = std::time::Instant::now();
    e.prefill(&ids)?;
    let tp = t1.elapsed().as_secs_f64();
    let mut outs = vec![best(&e.logits()?)];
    let t2 = std::time::Instant::now();
    while outs.len() < n {
        e.step(*outs.last().unwrap())?;
        outs.push(best(&e.logits()?));
    }
    let td = t2.elapsed().as_secs_f64();
    println!("{prompt}{}", tok.decode(&outs)?);
    println!(
        "prompt of {} tokens in {:.0} ms; {} tokens at {:.2} ms each ({:.1} a second)",
        ids.len(),
        tp * 1e3,
        n - 1,
        td * 1e3 / (n - 1).max(1) as f64,
        (n - 1) as f64 / td
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundings_are_the_formats() {
        // E4M3: every representable value is a fixed point; ties go to the even neighbour
        for b in 0..=0x7Eu8 {
            let v = glyd_gpu::fp8::E4M3[b as usize];
            assert_eq!(e4m3_round(v), v);
            assert_eq!(e4m3_byte(v), b);
        }
        assert_eq!(e4m3_round(1.0625), 1.0);
        assert_eq!(e4m3_round(1.1875), 1.25);
        assert_eq!(e4m3_round(2f32.powi(-10)), 0.0);
        assert_eq!(e4m3_round(3.0 * 2f32.powi(-10)), 2f32.powi(-8));
        for (v, r) in [(0.25, 0.0), (0.75, 1.0), (1.25, 1.0), (1.75, 2.0), (2.5, 2.0), (3.5, 4.0), (5.0, 4.0), (5.5, 6.0), (-0.3, -0.5)] {
            assert_eq!(e2m1_round(v), r, "{v}");
        }
        assert_eq!(pow2_scale(448.0, 1.0 / 448.0), 1.0);
        assert_eq!(pow2_scale(449.0, 1.0 / 448.0), 2.0);
        let mut x = vec![1.0, 1.0, 1.0, 1.0];
        hadamard(&mut x);
        assert_eq!(x, vec![2.0, 0.0, 0.0, 0.0]);
    }
}
