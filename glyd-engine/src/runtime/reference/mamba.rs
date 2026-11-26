//! Mamba-2 (`kinds::Mamba2`), a token at a time. The input product's rows are z (the gate, the inner width), the convolution's channels (the scan's input x, the inner
//! width, then the groups' B and C, a state's width each) and dt, a value a head. A causal depthwise convolution with a bias, then SiLU, over those channels. The selective
//! state update of each head: its state [head_dim, state] times exp(dt A), plus dt x B; y = state C + d x, where A = -exp(a_log), dt = softplus(dt + dt_bias) with no lower
//! limit (vLLM's and the engine's; transformers' chunked path clamps it to `time_step_min`, its step does not), and a head reads the B and C of its group (heads / groups
//! heads a group, in order). The gated norm: y silu(z), RMS over each group's channels (the inner width / groups), times the weight. Then the output product.

use super::Libm;
use super::{sigmoid, softplus, Params};
use crate::ffi::{err, Result};
use crate::runtime::kinds::Mamba2;

/// A sequence's state in one layer: each convolution channel's last `conv - 1` inputs (oldest first) and the scan's state [heads, head_dim, state].
pub struct MambaState {
    pub conv: Vec<f32>,
    pub s: Vec<f32>,
}

/// The convolution's channels: the scan's input and the groups' B and C.
pub fn conv_dim(m: &Mamba2) -> usize {
    m.heads * m.head_dim + 2 * m.groups * m.state
}

impl MambaState {
    pub fn new(m: &Mamba2) -> MambaState {
        MambaState { conv: vec![0.0; conv_dim(m) * (m.conv - 1)], s: vec![0.0; m.heads * m.head_dim * m.state] }
    }
}

/// One token through the layer: the input product, the convolution, the scan's step, the gated norm, the output product.
pub fn step(m: &Mamba2, p: &Params, st: &mut MambaState, h: &[f32]) -> Result<Vec<f32>> {
    let (nh, pd, ns, g, k) = (m.heads, m.head_dim, m.state, m.groups, m.conv);
    let (inner, cd) = (nh * pd, conv_dim(m));
    let mix = p.prod("in", h)?;
    let (w, b) = (p.vec("conv")?, p.vec("conv_bias")?);
    let (a_log, d, dt_bias, nw) = (p.vec("a_log")?, p.vec("d")?, p.vec("dt_bias")?, p.vec("norm")?);
    if g == 0 || k == 0 || nh % g != 0 || inner % g != 0 || mix.len() != inner + cd + nh || w.len() != cd * k || b.len() != cd {
        return err(format!(
            "Mamba-2: {} input rows, a convolution of {} and {} values, for {nh} heads of {pd}, {g} groups, a state of {ns}, {k} taps",
            mix.len(),
            w.len(),
            b.len()
        ));
    }
    if a_log.len() != nh || d.len() != nh || dt_bias.len() != nh || nw.len() != inner {
        return err(format!(
            "Mamba-2: a_log, d, dt_bias and norm of {}, {}, {} and {} values for {nh} heads of {pd}",
            a_log.len(),
            d.len(),
            dt_bias.len(),
            nw.len()
        ));
    }
    let (z, xbc, dtr) = (&mix[..inner], &mix[inner..inner + cd], &mix[inner + cd..]);
    // the convolution: channel c's taps w[c k ..], oldest first; its history takes the input
    let hist = k - 1;
    let cv: Vec<f32> = (0..cd)
        .map(|c| {
            let s = &mut st.conv[c * hist..(c + 1) * hist];
            let acc = b[c] + s.iter().enumerate().map(|(j, &v)| w[c * k + j] * v).sum::<f32>() + w[c * k + hist] * xbc[c];
            if hist > 0 {
                s.rotate_left(1);
                s[hist - 1] = xbc[c];
            }
            acc * sigmoid(acc)
        })
        .collect();
    let (xs, bb, cc) = (&cv[..inner], &cv[inner..inner + g * ns], &cv[inner + g * ns..]);
    // the scan's step: a head's channels each keep a state of `ns`
    let rep = nh / g;
    let mut y = vec![0f32; inner];
    for hh in 0..nh {
        let (bg, cg) = (&bb[(hh / rep) * ns..][..ns], &cc[(hh / rep) * ns..][..ns]);
        let dt = softplus(dtr[hh] + dt_bias[hh]);
        let da = (-a_log[hh].lexp() * dt).lexp();
        for i in 0..pd {
            let x = xs[hh * pd + i];
            let s = &mut st.s[(hh * pd + i) * ns..][..ns];
            let mut acc = 0f32;
            for n in 0..ns {
                s[n] = s[n] * da + dt * x * bg[n];
                acc += s[n] * cg[n];
            }
            y[hh * pd + i] = acc + d[hh] * x;
        }
    }
    // the gated norm, a group of channels at a time
    let per = inner / g;
    let mut gn = vec![0f32; inner];
    for grp in 0..g {
        let v: Vec<f32> = (grp * per..(grp + 1) * per).map(|c| y[c] * z[c] * sigmoid(z[c])).collect();
        let r = 1.0 / (v.iter().map(|x| x * x).sum::<f32>() / per as f32 + m.norm_eps).sqrt();
        for (j, x) in v.iter().enumerate() {
            gn[grp * per + j] = x * r * nw[grp * per + j];
        }
    }
    p.prod("o", &gn)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layer() -> (Mamba2, Params) {
        let m = Mamba2 { heads: 4, head_dim: 2, state: 3, groups: 2, conv: 3, chunk: 8, norm_eps: 1e-5 };
        let (hidden, inner, cd) = (5, 8, 8 + 12);
        let vals = |n: usize, seed: u32| -> Vec<f32> {
            (0..n as u32).map(|i| ((i.wrapping_mul(2_654_435_761).wrapping_add(seed * 97) >> 8) % 2001) as f32 / 1000.0 - 1.0).collect()
        };
        let mut p = Params::new();
        p.put("in", &[inner + cd + 4, hidden], vals((inner + cd + 4) * hidden, 1));
        p.put("conv", &[cd, 3], vals(cd * 3, 2));
        p.put("conv_bias", &[cd], vals(cd, 3));
        p.put("a_log", &[4], vals(4, 4));
        p.put("d", &[4], vals(4, 5));
        p.put("dt_bias", &[4], vals(4, 6));
        p.put("norm", &[inner], vals(inner, 7));
        p.put("o", &[hidden, inner], vals(hidden * inner, 8));
        (m, p)
    }

    /// The step against the definition written out another way for a sequence of tokens: the convolution as a sum over the last inputs, the state as the closed sum over
    /// the tokens so far (each token's dt x B, decayed by every later token's exp(dt A)), the gated norm by its groups.
    #[test]
    fn the_step_is_the_definition() {
        let (m, p) = layer();
        let (nh, pd, ns, g, k, inner, cd) = (4, 2, 3, 2, 3, 8, 20);
        let xs: Vec<Vec<f32>> = (0..6u32).map(|t| (0..5u32).map(|i| ((t * 31 + i * 17) % 13) as f32 / 6.0 - 1.0).collect()).collect();
        let mut st = MambaState::new(&m);
        let got: Vec<Vec<f32>> = xs.iter().map(|x| step(&m, &p, &mut st, x).unwrap()).collect();
        // the definition, over the whole sequence
        let mixes: Vec<Vec<f32>> = xs.iter().map(|x| p.prod("in", x).unwrap()).collect();
        let (w, b) = (p.vec("conv").unwrap(), p.vec("conv_bias").unwrap());
        let silu = |v: f32| v / (1.0 + (-v).lexp());
        let conv = |t: usize, c: usize| -> f32 {
            let mut acc = b[c];
            for j in 0..k {
                // (tap j reads the input k - 1 - j tokens back; before the first token, zeros)
                if let Some(u) = (t + j + 1).checked_sub(k) {
                    acc += w[c * k + j] * mixes[u][inner + c];
                }
            }
            silu(acc)
        };
        let (a_log, d, dt_bias, nw) = (p.vec("a_log").unwrap(), p.vec("d").unwrap(), p.vec("dt_bias").unwrap(), p.vec("norm").unwrap());
        let dt = |t: usize, h: usize| (1.0 + (mixes[t][inner + cd + h] + dt_bias[h]).lexp()).lln();
        for (t, want_from) in got.iter().enumerate() {
            let mut y = vec![0f32; inner];
            for h in 0..nh {
                let grp = h / (nh / g);
                for i in 0..pd {
                    let mut acc = d[h] * conv(t, h * pd + i);
                    for n in 0..ns {
                        // the state's entry n: every token u <= t gave dt x B, decayed since
                        let mut s = 0f32;
                        for u in 0..=t {
                            let decay: f32 = (u + 1..=t).map(|v| -a_log[h].lexp() * dt(v, h)).sum();
                            s += decay.lexp() * dt(u, h) * conv(u, h * pd + i) * conv(u, inner + grp * ns + n);
                        }
                        acc += s * conv(t, inner + g * ns + grp * ns + n);
                    }
                    y[h * pd + i] = acc;
                }
            }
            let per = inner / g;
            let mut gn = vec![0f32; inner];
            for grp in 0..g {
                let v: Vec<f32> = (grp * per..(grp + 1) * per).map(|c| y[c] * silu(mixes[t][c])).collect();
                let rms = (v.iter().map(|x| x * x).sum::<f32>() / per as f32 + 1e-5).sqrt();
                for j in 0..per {
                    gn[grp * per + j] = v[j] / rms * nw[grp * per + j];
                }
            }
            let want = p.prod("o", &gn).unwrap();
            for (a, b) in want_from.iter().zip(&want) {
                assert!((a - b).abs() <= 2e-4 * (1.0 + b.abs()), "token {t}: {a} against {b}");
            }
        }
        // a sequence's state is its own: a second sequence from nothing gives the first token's output again
        let mut other = MambaState::new(&m);
        assert_eq!(step(&m, &p, &mut other, &xs[0]).unwrap(), got[0]);
    }

    #[test]
    fn shapes_are_checked() {
        let (m, p) = layer();
        let mut st = MambaState::new(&m);
        assert!(step(&m, &p, &mut st, &[0.0; 4]).is_err());
        let odd = Mamba2 { groups: 3, ..m.clone() };
        assert!(step(&odd, &p, &mut MambaState::new(&odd), &[0.0; 5]).unwrap_err().msg.contains("Mamba-2"));
    }
}
