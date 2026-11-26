//! Linear attention with the gated delta rule (`kinds::Linear`): Gated DeltaNet (one decay per value head) and Kimi delta attention (one
//! decay per key channel). Per token: a causal depthwise convolution and SiLU over the queries, keys and values; q and k l2-normed (eps
//! 1e-6 inside the root), q times 1/sqrt(k_dim); the state decays by exp(g), takes k (beta (v - S^T k))^T, and the output S^T q is normed
//! per value head (RMS, the shared weight) and gated.

use super::Libm;
use super::{act, l2norm, sigmoid, softplus, Params};
use crate::ffi::{err, Result};
use crate::runtime::kinds::{Decay, Linear, LinearKind};

/// A sequence's state in one layer: each channel's last `conv - 1` inputs (oldest first) and the state [v_heads, k_dim, v_dim].
pub struct LinearState {
    pub conv: Vec<f32>,
    pub s: Vec<f32>,
    /// Each value head's open chunk, where a test runs the chunked form (`chunked`).
    #[cfg(test)]
    pub open: Vec<chunked::Open>,
}

pub fn conv_dim(l: &Linear) -> usize {
    2 * l.k_heads * l.k_dim + l.v_heads * l.v_dim
}

impl LinearState {
    pub fn new(l: &Linear) -> LinearState {
        LinearState {
            conv: vec![0.0; conv_dim(l) * (l.conv - 1)],
            s: vec![0.0; l.v_heads * l.k_dim * l.v_dim],
            #[cfg(test)]
            open: (0..l.v_heads).map(|_| chunked::Open::default()).collect(),
        }
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
        #[cfg(test)]
        if chunked::FORM.get() == Some(chunked::Rounded::Stored) {
            q.iter_mut().chain(k.iter_mut()).for_each(|e| *e = super::bf16(*e));
        }
        q.iter_mut().for_each(|e| *e *= scale);
        let b = sigmoid(beta[h]);
        let g = |i: usize| -> f32 {
            let (a, dt) = if per_channel { (decay[h * dk + i], dt_bias[h * dk + i]) } else { (decay[h], dt_bias[h]) };
            match l.decay {
                Decay::Softplus => -a_log[h].lexp() * softplus(a + dt),
                Decay::Bounded(bound) => bound * sigmoid(a_log[h].lexp() * (a + dt)),
            }
        };
        // (the per-head norm and the gate of the rule's output)
        let tail = |o: &[f32]| -> Vec<f32> {
            let rr = 1.0 / (o.iter().map(|x| x * x).sum::<f32>() / dv as f32 + l.norm_eps).sqrt();
            (0..dv).map(|j| o[j] * rr * w[j] * act(l.gate_act, gate[h * dv + j])).collect()
        };
        let s = &mut st.s[h * dk * dv..][..dk * dv];
        #[cfg(test)]
        if let Some(form) = chunked::FORM.get() {
            if per_channel {
                return err("the chunked form of the delta rule is written for one decay a head");
            }
            if st.open[h].tokens() == chunked::CHUNK {
                st.open[h].close(s, dk, dv);
            }
            out.extend(tail(&st.open[h].token(form, s, &q, &k, v, b, g(0), dk, dv)));
            continue;
        }
        for i in 0..dk {
            let f = g(i).lexp();
            s[i * dv..][..dv].iter_mut().for_each(|e| *e *= f);
        }
        let u: Vec<f32> = (0..dv).map(|j| b * (v[j] - (0..dk).map(|i| s[i * dv + j] * k[i]).sum::<f32>())).collect();
        for i in 0..dk {
            for j in 0..dv {
                s[i * dv + j] += k[i] * u[j];
            }
        }
        let o: Vec<f32> = (0..dv).map(|j| (0..dk).map(|i| s[i * dv + j] * q[i]).sum()).collect();
        out.extend(tail(&o));
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

/// The delta rule's chunked form, for a test to set beside the recurrence of `core`: the form the upstream prompt kernels run (a chunk of up to 64
/// tokens at once from the state it starts with), with or without their roundings to the inputs' type, bf16. For a chunk's tokens i, with G the
/// running sum of the log decays g: the matrix A = (I + L)^-1, L[i][j] = beta_i (k_i . k_j) e^(G_i - G_j) below the diagonal; u = A (beta v) and
/// w = A (beta e^G k); the new values v' = u - w S0; the output (q_i e^(G_i)) S0 + sum over j <= i of (q_i . k_j) e^(G_i - G_j) v'_j; and the state
/// after the chunk S0 e^(G_n) + sum over j of k_j e^(G_n - G_j) (x) v'_j. Each token's row is made as the token comes (a row reads only the rows
/// before it), so a pass is `core` a token at a time with `FORM` set, and `end` at its end.
#[cfg(test)]
pub mod chunked {
    use super::super::{bf16, Libm};
    use super::{Linear, LinearState};
    use std::cell::Cell;

    /// A chunk's tokens at most (the upstream kernels').
    pub const CHUNK: usize = 64;

    /// Which of the chunked form's values are rounded to bf16.
    #[derive(Clone, Copy, Debug, PartialEq)]
    pub enum Rounded {
        /// None: the recurrence's sums in the chunk's order, fp32.
        None,
        /// The solved matrix A, the rows u and w it gives, and the state a chunk starts from as the chunk reads it.
        Four,
        /// Every value the upstream chunk kernels keep or read in the inputs' type: `Four`'s, the normed q and k, the operands beta v and
        /// beta e^G k, and the new values v'.
        Stored,
    }

    thread_local! {
        /// Where Some: `core` runs each token as the next one of its head's open chunk.
        pub static FORM: Cell<Option<Rounded>> = const { Cell::new(None) };
    }

    /// A value head's open chunk: each token's running log decay, key, operands, row of A below the diagonal (fp32, as the solve reads it) and
    /// new value.
    #[derive(Default)]
    pub struct Open {
        g: Vec<f32>,
        k: Vec<Vec<f32>>,
        kb: Vec<Vec<f32>>,
        vb: Vec<Vec<f32>>,
        a: Vec<Vec<f32>>,
        vnew: Vec<Vec<f32>>,
    }

    fn dot(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }

    impl Open {
        /// The chunk's tokens so far.
        pub fn tokens(&self) -> usize {
            self.g.len()
        }

        /// The next token of the chunk: its output S^T q. `s0` is the state the chunk started from ([k_dim, v_dim]); q is scaled, q and k normed.
        #[allow(clippy::too_many_arguments)]
        pub fn token(&mut self, form: Rounded, s0: &[f32], q: &[f32], k: &[f32], v: &[f32], beta: f32, g: f32, dk: usize, dv: usize) -> Vec<f32> {
            let (four, stored) = (form != Rounded::None, form == Rounded::Stored);
            let r = |x: f32, on: bool| if on { bf16(x) } else { x };
            let i = self.tokens();
            let gi = self.g.last().copied().unwrap_or(0.0) + g;
            // row i of A below the diagonal: -L's, then the rows before it folded in (forward substitution)
            let l: Vec<f32> = (0..i).map(|j| -(beta * dot(k, &self.k[j]) * (gi - self.g[j]).lexp())).collect();
            let mut row = l.clone();
            for j in 0..i {
                for m in j + 1..i {
                    row[j] += l[m] * self.a[m][j];
                }
            }
            let a: Vec<f32> = row.iter().map(|&x| r(x, four)).collect();
            let vb: Vec<f32> = v.iter().map(|&x| r(beta * x, stored)).collect();
            let kb: Vec<f32> = k.iter().map(|&x| r(beta * gi.lexp() * x, stored)).collect();
            let (mut u, mut w) = (vb.clone(), kb.clone());
            for j in 0..i {
                u.iter_mut().zip(&self.vb[j]).for_each(|(x, y)| *x += a[j] * y);
                w.iter_mut().zip(&self.kb[j]).for_each(|(x, y)| *x += a[j] * y);
            }
            u.iter_mut().chain(w.iter_mut()).for_each(|x| *x = r(*x, four));
            let h0 = |a: usize, b: usize| r(s0[a * dv + b], four);
            let vnew: Vec<f32> = (0..dv).map(|b| r(u[b] - (0..dk).map(|a| w[a] * h0(a, b)).sum::<f32>(), stored)).collect();
            let mut o: Vec<f32> = (0..dv).map(|b| (0..dk).map(|a| q[a] * gi.lexp() * h0(a, b)).sum()).collect();
            for j in 0..=i {
                let (kj, vj, gj) = if j == i { (k, &vnew, gi) } else { (&self.k[j][..], &self.vnew[j], self.g[j]) };
                let f = dot(q, kj) * (gi - gj).lexp();
                o.iter_mut().zip(vj).for_each(|(x, y)| *x += f * y);
            }
            self.g.push(gi);
            self.k.push(k.to_vec());
            self.kb.push(kb);
            self.vb.push(vb);
            self.a.push(row);
            self.vnew.push(vnew);
            o
        }

        /// The chunk closed: `s` (the state it started from) becomes the state after its tokens, and the chunk is empty.
        pub fn close(&mut self, s: &mut [f32], dk: usize, dv: usize) {
            let Some(&gn) = self.g.last() else { return };
            s.iter_mut().for_each(|e| *e *= gn.lexp());
            for ((k, v), g) in self.k.iter().zip(&self.vnew).zip(&self.g) {
                let f = (gn - g).lexp();
                for a in 0..dk {
                    for b in 0..dv {
                        s[a * dv + b] += k[a] * f * v[b];
                    }
                }
            }
            *self = Open::default();
        }
    }

    /// A pass's end: every head's open chunk closed into the state.
    pub fn end(l: &Linear, st: &mut LinearState) {
        let (dk, dv) = (l.k_dim, l.v_dim);
        for (h, c) in st.open.iter_mut().enumerate() {
            c.close(&mut st.s[h * dk * dv..][..dk * dv], dk, dv);
        }
    }

    /// The chunked form in fp32 is the recurrence: on random inputs, a chunk and a half of tokens through both, the outputs and the states after
    /// them agree to fp32's sums (the tolerance of the reference against transformers in float32, `HF_TOL`, with room for 96 tokens' sums).
    #[test]
    fn the_chunked_form_is_the_recurrence() {
        use crate::runtime::kinds::{Act, Decay, LinearKind};
        use crate::runtime::reference::testing::fill;
        use crate::runtime::reference::Params;
        let l = Linear {
            kind: LinearKind::GatedDeltaNet,
            k_heads: 2,
            v_heads: 4,
            k_dim: 16,
            v_dim: 8,
            conv: 4,
            decay: Decay::Softplus,
            gate_act: Act::Silu,
            norm_eps: 1e-6,
            state_f32: true,
        };
        let mut p = Params::new();
        let cd = super::conv_dim(&l);
        p.put("conv", &[cd, l.conv], fill("chunk.conv", cd * l.conv, -0.5, 0.5));
        p.put("a_log", &[l.v_heads], fill("chunk.a_log", l.v_heads, -1.0, 0.5));
        p.put("dt_bias", &[l.v_heads], fill("chunk.dt", l.v_heads, -0.5, 0.5));
        p.put("norm", &[l.v_dim], fill("chunk.norm", l.v_dim, 0.5, 1.5));
        let n = CHUNK + CHUNK / 2;
        let run = |form: Option<Rounded>| {
            let mut st = LinearState::new(&l);
            let mut out = Vec::new();
            FORM.set(form);
            for t in 0..n {
                let x = |what: &str, len: usize| fill(&format!("chunk.{what}.{t}"), len, -1.0, 1.0);
                out.extend(
                    super::core(&l, &p, &mut st, &x("qkv", cd), &x("beta", l.v_heads), &x("decay", l.v_heads), &x("gate", l.v_heads * l.v_dim)).unwrap(),
                );
            }
            FORM.set(None);
            end(&l, &mut st);
            (out, st.s)
        };
        let (want, got) = (run(None), run(Some(Rounded::None)));
        let rel = |a: &[f32], b: &[f32]| a.iter().zip(b).fold(0f32, |m, (x, y)| m.max((x - y).abs())) / b.iter().fold(0f32, |m, y| m.max(y.abs()));
        let tol = 100.0 * crate::runtime::reference::HF_TOL;
        assert!(rel(&got.0, &want.0) < tol && rel(&got.1, &want.1) < tol, "outputs {:e}, states {:e}", rel(&got.0, &want.0), rel(&got.1, &want.1));
        // (and the rounded forms are other values, near it)
        let four = run(Some(Rounded::Four));
        assert!(four.0 != got.0 && rel(&four.0, &want.0) < crate::runtime::reference::BF16_TOL, "{:e}", rel(&four.0, &want.0));
    }
}
