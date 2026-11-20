//! Linear attention with the gated delta rule (`kinds::Linear`): Gated DeltaNet (one decay per value head) and Kimi delta attention (one
//! decay per key channel). Per token: a causal depthwise convolution and SiLU over the queries, keys and values; q and k l2-normed (eps
//! 1e-6 inside the root), q times 1/sqrt(k_dim); the state decays by exp(g), takes k (beta (v - S^T k))^T, and the output S^T q is normed
//! per value head (RMS, the shared weight) and gated.

use super::{act, l2norm, sigmoid, softplus, Params};
use crate::ffi::{err, Result};
use crate::runtime::kinds::{Decay, Linear, LinearKind};

/// A sequence's state in one layer: each channel's last `conv - 1` inputs (oldest first) and the state [v_heads, k_dim, v_dim].
pub struct LinearState {
    pub conv: Vec<f32>,
    pub s: Vec<f32>,
}

pub fn conv_dim(l: &Linear) -> usize {
    2 * l.k_heads * l.k_dim + l.v_heads * l.v_dim
}

impl LinearState {
    pub fn new(l: &Linear) -> LinearState {
        LinearState { conv: vec![0.0; conv_dim(l) * (l.conv - 1)], s: vec![0.0; l.v_heads * l.k_dim * l.v_dim] }
    }
}

/// The convolution of one token: channel c's taps `w[c * kernel ..]`, oldest first; its history `state[c * (kernel - 1) ..]` takes x[c].
pub fn conv_step(w: &[f32], state: &mut [f32], x: &[f32], kernel: usize) -> Vec<f32> {
    let h = kernel - 1;
    x.iter()
        .enumerate()
        .map(|(c, &xc)| {
            let st = &mut state[c * h..(c + 1) * h];
            let acc = st.iter().enumerate().map(|(j, &s)| w[c * kernel + j] * s).sum::<f32>() + w[c * kernel + h] * xc;
            if h > 0 {
                st.rotate_left(1);
                st[h - 1] = xc;
            }
            acc * sigmoid(acc)
        })
        .collect()
}

/// The core between the products: the conv's input `qkv`, the raw `beta`, `decay` and `gate` products in, the per-head normed and gated
/// output (v_heads x v_dim, the input of `o`) out.
pub fn core(l: &Linear, p: &Params, st: &mut LinearState, qkv: &[f32], beta: &[f32], decay: &[f32], gate: &[f32]) -> Result<Vec<f32>> {
    let (nk, nv, dk, dv) = (l.k_heads, l.v_heads, l.k_dim, l.v_dim);
    let per_channel = l.kind == LinearKind::KimiDelta;
    if nv % nk != 0 || qkv.len() != conv_dim(l) || beta.len() != nv || gate.len() != nv * dv || decay.len() != if per_channel { nv * dk } else { nv } {
        return err(format!(
            "linear attention: products of {}, {}, {}, {} values for {nk}/{nv} heads of {dk}/{dv}",
            qkv.len(),
            beta.len(),
            decay.len(),
            gate.len()
        ));
    }
    let x = conv_step(p.vec("conv")?, &mut st.conv, qkv, l.conv);
    let (a_log, dt_bias, w) = (p.vec("a_log")?, p.vec("dt_bias")?, p.vec("norm")?);
    let (kd, r, scale) = (nk * dk, nv / nk, 1.0 / (dk as f32).sqrt());
    let mut out = Vec::with_capacity(nv * dv);
    for h in 0..nv {
        let kh = h / r;
        let mut q = x[kh * dk..][..dk].to_vec();
        let mut k = x[kd + kh * dk..][..dk].to_vec();
        let v = &x[2 * kd + h * dv..][..dv];
        l2norm(&mut q, 1e-6);
        l2norm(&mut k, 1e-6);
        q.iter_mut().for_each(|e| *e *= scale);
        let b = sigmoid(beta[h]);
        let g = |i: usize| -> f32 {
            let (a, dt) = if per_channel { (decay[h * dk + i], dt_bias[h * dk + i]) } else { (decay[h], dt_bias[h]) };
            match l.decay {
                Decay::Softplus => -a_log[h].exp() * softplus(a + dt),
                Decay::Bounded(bound) => bound * sigmoid(a_log[h].exp() * (a + dt)),
            }
        };
        let s = &mut st.s[h * dk * dv..][..dk * dv];
        for i in 0..dk {
            let f = g(i).exp();
            s[i * dv..][..dv].iter_mut().for_each(|e| *e *= f);
        }
        let u: Vec<f32> = (0..dv).map(|j| b * (v[j] - (0..dk).map(|i| s[i * dv + j] * k[i]).sum::<f32>())).collect();
        for i in 0..dk {
            for j in 0..dv {
                s[i * dv + j] += k[i] * u[j];
            }
        }
        let o: Vec<f32> = (0..dv).map(|j| (0..dk).map(|i| s[i * dv + j] * q[i]).sum()).collect();
        let rr = 1.0 / (o.iter().map(|x| x * x).sum::<f32>() / dv as f32 + l.norm_eps).sqrt();
        out.extend((0..dv).map(|j| o[j] * rr * w[j] * act(l.gate_act, gate[h * dv + j])));
    }
    Ok(out)
}

/// One token through the layer: the products, the core, then `o`.
pub fn step(l: &Linear, p: &Params, st: &mut LinearState, h: &[f32]) -> Result<Vec<f32>> {
    let qkv = if p.has_prod("qkv") {
        p.prod("qkv", h)?
    } else {
        let mut v = p.prod("q", h)?;
        v.extend(p.prod("k", h)?);
        v.extend(p.prod("v", h)?);
        v
    };
    let o = core(l, p, st, &qkv, &p.prod("beta", h)?, &p.prod("decay", h)?, &p.prod("gate", h)?)?;
    p.prod("o", &o)
}
