//! The CPU reference of Qwen3.5's linear-attention layer (Gated DeltaNet), one token at a time, as vLLM 0.30.0 (the engine's correctness
//! reference) computes a decode step: fp32 throughout, rounded to bf16 only where vLLM stores a tensor in bf16. The convolution sums,
//! applies SiLU and rounds once (`causal_conv1d_update`); q and k are l2-normed in fp32 and not rounded (`fused_recurrent_gated_delta_rule`
//! with its in-kernel l2 norm; vLLM's prompt path and transformers round them to bf16); beta
//! is rounded, g is not (`fused_gdn_gating`); the head's output is rounded; the gated norm takes the weight as vLLM holds it (bf16) and
//! rounds once (`RMSNormGated`). The GPU kernels are checked against it. A development build's only.

/// `x` rounded to bf16 (to nearest, ties to even), as an f32.
pub fn bf16(x: f32) -> f32 {
    if x.is_nan() {
        return x;
    }
    let b = x.to_bits();
    f32::from_bits(b.wrapping_add(0x7fff + ((b >> 16) & 1)) & 0xffff_0000)
}

pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// torch's softplus (beta 1, threshold 20: above it, x).
pub fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else {
        x.exp().ln_1p()
    }
}

/// One step of the causal depthwise convolution, then SiLU: channel c's weights are `w[c * kernel ..]`, the oldest tap first; `state[c * (kernel - 1) ..]` holds its last
/// `kernel - 1` inputs, the oldest first, and takes `x[c]` in. bf16 in and out, as the checkpoint's weights and vLLM's state are.
pub fn conv_step(w: &[f32], state: &mut [f32], x: &[f32], kernel: usize) -> Vec<f32> {
    let h = kernel - 1;
    x.iter()
        .enumerate()
        .map(|(c, &xc)| {
            let st = &mut state[c * h..(c + 1) * h];
            let mut acc = 0.0f32;
            for (j, &s) in st.iter().enumerate() {
                acc += w[c * kernel + j] * s;
            }
            acc += w[c * kernel + h] * xc;
            if h > 0 {
                st.rotate_left(1);
                st[h - 1] = xc;
            }
            bf16(silu(acc))
        })
        .collect()
}

/// `x * rsqrt(sum(x^2) + 1e-6)`, in place, in fp32 (not rounded: vLLM's decode step norms in its kernel).
pub fn l2norm(x: &mut [f32]) {
    let r = 1.0 / (x.iter().map(|v| v * v).sum::<f32>() + 1e-6).sqrt();
    x.iter_mut().for_each(|v| *v *= r);
}

/// One token of the gated delta rule for one value head: the state `s[i * dv + j]` (S[i][j], i over the key's dims, j over the value's) decays by exp(g), takes
/// `k (beta (v - S^T k))^T`, and the output is `S^T q`.
pub fn delta_step(s: &mut [f32], q: &[f32], k: &[f32], v: &[f32], g: f32, beta: f32) -> Vec<f32> {
    let (dk, dv) = (k.len(), v.len());
    assert!(s.len() == dk * dv && q.len() == dk);
    let decay = g.exp();
    s.iter_mut().for_each(|x| *x *= decay);
    let delta: Vec<f32> = (0..dv).map(|j| (v[j] - (0..dk).map(|i| s[i * dv + j] * k[i]).sum::<f32>()) * beta).collect();
    for i in 0..dk {
        for j in 0..dv {
            s[i * dv + j] += k[i] * delta[j];
        }
    }
    (0..dv).map(|j| (0..dk).map(|i| s[i * dv + j] * q[i]).sum()).collect()
}

/// One value head's output (bf16) normed over its values in fp32, times the weight (the checkpoint's fp32 rounded to bf16, as vLLM loads it;
/// not centered on zero) and silu(z), rounded to bf16 once at the end: vLLM's `RMSNormGated`, `bf16(norm(o) * w * silu(z))`.
pub fn gated_norm(o: &[f32], z: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let r = 1.0 / (o.iter().map(|x| x * x).sum::<f32>() / o.len() as f32 + eps).sqrt();
    o.iter().zip(z).zip(w).map(|((&x, &zz), &ww)| bf16(x * r * bf16(ww) * silu(zz))).collect()
}

/// A linear-attention layer's heads: `nk` key heads of `dk`, `nv` value heads of `dv` (value head h reads key head h / (nv / nk)), the convolution's kernel.
#[derive(Clone, Copy, Debug)]
pub struct Dims {
    pub nk: usize,
    pub nv: usize,
    pub dk: usize,
    pub dv: usize,
    pub kernel: usize,
}

impl Dims {
    pub fn conv_dim(&self) -> usize {
        2 * self.nk * self.dk + self.nv * self.dv
    }
}

/// A layer's own small tensors: the convolution's weights `[conv_dim * kernel]`, `A_log` and `dt_bias` `[nv]`, the gated norm's weight `[dv]`.
pub struct LinearParams {
    pub conv: Vec<f32>,
    pub a_log: Vec<f32>,
    pub dt_bias: Vec<f32>,
    pub norm: Vec<f32>,
    pub eps: f32,
}

/// A sequence's state in one layer: the convolution's last inputs and the value heads' states.
pub struct LinearState {
    pub conv: Vec<f32>,
    pub s: Vec<f32>,
}

impl LinearState {
    pub fn new(d: &Dims) -> LinearState {
        LinearState { conv: vec![0.0; d.conv_dim() * (d.kernel - 1)], s: vec![0.0; d.nv * d.dk * d.dv] }
    }
}

/// One token through a linear-attention layer between its products: `qkv` (in_proj_qkv's output: queries, keys, values), `z`, `b`, `a` (the other products') in, out_proj's
/// input out (`nv * dv`, bf16 values).
pub fn mixer_step(d: &Dims, p: &LinearParams, st: &mut LinearState, qkv: &[f32], z: &[f32], b: &[f32], a: &[f32]) -> Vec<f32> {
    let x = conv_step(&p.conv, &mut st.conv, qkv, d.kernel);
    let (kd, r, scale) = (d.nk * d.dk, d.nv / d.nk, 1.0 / (d.dk as f32).sqrt());
    let mut out = Vec::with_capacity(d.nv * d.dv);
    for h in 0..d.nv {
        let kh = h / r;
        let mut q = x[kh * d.dk..(kh + 1) * d.dk].to_vec();
        let mut k = x[kd + kh * d.dk..kd + (kh + 1) * d.dk].to_vec();
        let v = &x[2 * kd + h * d.dv..2 * kd + (h + 1) * d.dv];
        l2norm(&mut q);
        l2norm(&mut k);
        q.iter_mut().for_each(|e| *e *= scale);
        let beta = bf16(sigmoid(b[h]));
        let g = -p.a_log[h].exp() * softplus(a[h] + p.dt_bias[h]);
        let o: Vec<f32> = delta_step(&mut st.s[h * d.dk * d.dv..(h + 1) * d.dk * d.dv], &q, &k, v, g, beta).into_iter().map(bf16).collect();
        out.extend(gated_norm(&o, &z[h * d.dv..(h + 1) * d.dv], &p.norm, p.eps));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(a.len(), b.len());
        for (i, (x, y)) in a.iter().zip(b).enumerate() {
            assert!((x - y).abs() <= tol * (1.0 + y.abs()), "at {i}: {x} against {y}");
        }
    }

    #[test]
    fn the_scalars() {
        assert_eq!((bf16(1.0), bf16(1.0 + 1.0 / 512.0), bf16(1.0 + 3.0 / 512.0)), (1.0, 1.0, 1.0 + 4.0 / 512.0));
        // a tie goes to the even neighbour
        assert_eq!((bf16(1.0 + 2.0 / 512.0), bf16(1.0 + 6.0 / 512.0)), (1.0, 1.0 + 8.0 / 512.0));
        assert_eq!(softplus(30.0), 30.0);
        assert!((softplus(0.0) - std::f32::consts::LN_2).abs() < 1e-7);
        assert!((silu(1.0) - 1.0 / (1.0 + (-1.0f32).exp())).abs() < 1e-7 && sigmoid(0.0) == 0.5);
        let mut x = [3.0, 4.0];
        l2norm(&mut x);
        close(&x, &[0.6, 0.8], 1e-6);
        // the l2 norm stays fp32: 1/sqrt(3) is not a bf16 value
        let mut x = [1.0; 3];
        l2norm(&mut x);
        assert!(x[0] != bf16(x[0]) && (x[0] - 3f32.sqrt().recip()).abs() < 1e-6);
    }

    /// A key of length one, written with beta 1 and no decay: reading with that key gives the value; a second write to the key replaces the value (the delta rule), it is not added.
    #[test]
    fn write_then_read() {
        let k = [0.5f32; 4];
        let mut s = vec![0.0; 16];
        assert_eq!(delta_step(&mut s, &k, &k, &[1.0, 2.0, 3.0, 4.0], 0.0, 1.0), [1.0, 2.0, 3.0, 4.0]);
        assert_eq!(delta_step(&mut s, &k, &k, &[-1.0, 0.0, 5.0, 2.0], 0.0, 1.0), [-1.0, 0.0, 5.0, 2.0]);
        // a key at right angles to it reads nothing
        assert_eq!(delta_step(&mut s.clone(), &[0.5, -0.5, 0.5, -0.5], &k, &[0.0; 4], 0.0, 0.0), [0.0; 4]);
        // beta 0 writes nothing; exp(g) = 1/2 halves what is read
        close(&delta_step(&mut s, &k, &k, &[9.0; 4], 0.5f32.ln(), 0.0), &[-0.5, 0.0, 2.5, 1.0], 1e-6);
        // a zero state and a zero value give zero, whatever the query and key
        assert_eq!(delta_step(&mut [0.0; 6], &[1.0, 2.0], &[3.0, -1.0], &[0.0; 3], -0.3, 0.7), [0.0; 3]);
    }

    /// The step taken token by token is the whole causal convolution of the sequence with zero history (inputs and weights of a few bits: every sum is exact).
    #[test]
    fn the_conv_step_is_the_causal_convolution() {
        let (c, kernel, t) = (5, 4, 9);
        let val = |i: usize| ((i * 7 + 3) % 9) as f32 / 4.0 - 1.0;
        let w: Vec<f32> = (0..c * kernel).map(|i| val(i * 3 + 1)).collect();
        let xs: Vec<Vec<f32>> = (0..t).map(|n| (0..c).map(|ch| val(n * 5 + ch * 2)).collect()).collect();
        let mut state = vec![0.0; c * (kernel - 1)];
        for n in 0..t {
            let got = conv_step(&w, &mut state, &xs[n], kernel);
            let want: Vec<f32> = (0..c)
                .map(|ch| {
                    let acc: f32 = (0..kernel).filter(|&j| n + j + 1 >= kernel).map(|j| w[ch * kernel + j] * xs[n + j + 1 - kernel][ch]).sum();
                    bf16(silu(acc))
                })
                .collect();
            assert_eq!(got, want, "token {n}");
        }
    }

    #[test]
    fn the_gated_norm() {
        let o = [1.0, -1.0, 1.0, -1.0];
        close(&gated_norm(&o, &[30.0; 4], &[2.0; 4], 1e-6), &[60.0, -60.0, 60.0, -60.0], 1e-2);
        // the weight is not centered on zero: a zero weight gives zero
        assert_eq!(gated_norm(&o, &[1.0; 4], &[0.0; 4], 1e-6), [0.0; 4]);
        // vLLM's order: the weight rounded to bf16, norm * w * silu(z) in fp32, one rounding (no rounding of the normed value before w)
        let (o, z, w) = ([0.3f32, -1.7, 2.2, 0.05], [0.9f32, -0.4, 1.3, 2.5], [1.0f32 + 1.0 / 1024.0, 0.77, 1.33, 0.91]);
        let r = 1.0 / (o.iter().map(|x| x * x).sum::<f32>() / 4.0 + 1e-6).sqrt();
        let want: Vec<f32> = (0..4).map(|i| bf16(o[i] * r * bf16(w[i]) * silu(z[i]))).collect();
        assert_eq!(gated_norm(&o, &z, &w, 1e-6), want);
        assert_eq!(bf16(w[0]), 1.0);
    }

    /// One token from a zero state: S = k (beta v)^T, so value head h's output is beta v (k . q) with the queries and keys of key head h / r, l2-normed, the query scaled by
    /// 1/sqrt(dk); with the convolution's newest tap 1 and the others anything, the convolution of a first token is silu(x).
    #[test]
    fn the_mixer_on_a_first_token() {
        let d = Dims { nk: 2, nv: 4, dk: 4, dv: 2, kernel: 4 };
        let n = d.conv_dim();
        let conv: Vec<f32> = (0..n * 4).map(|i| if i % 4 == 3 { 1.0 } else { 0.25 * (i % 3) as f32 }).collect();
        let p = LinearParams { conv, a_log: vec![0.1, -0.2, 0.3, 0.0], dt_bias: vec![0.5; 4], norm: vec![1.0, 0.5], eps: 1e-6 };
        let qkv: Vec<f32> = (0..n).map(|i| ((i * 5 + 1) % 11) as f32 / 8.0 - 0.6).collect();
        let (z, b, a) = (vec![0.3, -0.7, 1.1, 0.2, 2.0, -1.0, 0.5, 0.9], vec![0.2, -1.0, 0.0, 3.0], vec![0.1, 0.2, -0.3, 0.4]);
        let mut st = LinearState::new(&d);
        let got = mixer_step(&d, &p, &mut st, &qkv, &z, &b, &a);
        let x: Vec<f32> = qkv.iter().map(|&v| bf16(silu(v))).collect();
        let normed = |s: &[f32]| {
            let r = 1.0 / (s.iter().map(|v| v * v).sum::<f32>() + 1e-6).sqrt();
            s.iter().map(|v| v * r).collect::<Vec<f32>>()
        };
        let mut want = Vec::new();
        for h in 0..4 {
            let kh = h / 2;
            let (q, k) = (normed(&x[kh * 4..kh * 4 + 4]), normed(&x[8 + kh * 4..8 + kh * 4 + 4]));
            let qk: f32 = q.iter().zip(&k).map(|(a, b)| a * b).sum::<f32>() / 2.0;
            let beta = bf16(sigmoid(b[h]));
            let o: Vec<f32> = (0..2).map(|j| bf16(beta * x[16 + h * 2 + j] * qk)).collect();
            want.extend(gated_norm(&o, &z[h * 2..h * 2 + 2], &p.norm, 1e-6));
        }
        close(&got, &want, 2e-2);
        // the state took the token: a second token with no decay reads it back through the same key
        assert!(st.s.iter().any(|&v| v != 0.0) && st.conv[(n - 1) * 3 + 2] == qkv[n - 1]);
    }

    /// The decay: g = -exp(A_log) * softplus(a + dt_bias), so a larger a forgets more of the first token.
    #[test]
    fn the_decay_follows_a() {
        let d = Dims { nk: 1, nv: 1, dk: 2, dv: 2, kernel: 1 };
        let p = LinearParams { conv: vec![1.0; 6], a_log: vec![0.0], dt_bias: vec![0.0], norm: vec![1.0; 2], eps: 1e-6 };
        let first = [0.6, 0.8, 0.6, 0.8, 1.0, -1.0];
        let read = |a: f32| {
            let mut st = LinearState::new(&d);
            mixer_step(&d, &p, &mut st, &first, &[0.0; 2], &[30.0], &[0.0]);
            mixer_step(&d, &p, &mut st, &[0.6, 0.8, 0.6, 0.8, 0.0, 0.0], &[0.0; 2], &[-30.0], &[a]);
            st.s.iter().map(|v| v.abs()).sum::<f32>()
        };
        assert!(read(-5.0) > read(0.0) && read(0.0) > read(5.0));
    }
}
