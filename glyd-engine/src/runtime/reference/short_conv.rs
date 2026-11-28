//! The gated short convolution (`kinds::ShortConv`, LFM's `conv` layers). The input product's rows are B, C and x, the hidden width each, in that order; u = B x,
//! channel by channel; each channel of u through its own causal convolution over the sequence (a token reads its own u and the `taps` - 1 before it, zeros before
//! the sequence's first token; a bias where the kind has biases; no activation); y = C conv(u); then the output product. A sequence keeps, a channel, its last
//! `taps` - 1 values of u: the gated product, not the layer's input.
//!
//! Two forms of the one function: a token at a time against the state (`step`), and a run of tokens at once (`prompt`), each channel's line the state then the
//! run's u, a token's output the taps over its window of the line. Each sum takes the bias, then the taps oldest first, in f32: the two forms are the same bits.

use super::Params;
use crate::ffi::{err, Result};
use crate::runtime::kinds::ShortConv;

/// A sequence's state in one layer: each channel's last `taps` - 1 values of u (oldest first), zeros before the sequence's first token.
pub struct ConvState {
    pub u: Vec<f32>,
}

impl ConvState {
    pub fn new(c: &ShortConv, hidden: usize) -> ConvState {
        ConvState { u: vec![0.0; c.state(hidden)] }
    }
}

/// The kind's weights for a layer of `d` channels, their sizes checked: the taps [d, taps] and the convolution's bias.
fn weights<'a>(c: &ShortConv, p: &'a Params, st: &ConvState, d: usize) -> Result<(&'a [f32], Option<&'a [f32]>)> {
    let w = p.vec("conv")?;
    let b = if c.bias { Some(p.vec("conv_bias")?) } else { None };
    if c.taps == 0 || w.len() != d * c.taps || b.is_some_and(|b| b.len() != d) || st.u.len() != c.state(d) {
        return err(format!(
            "short convolution: a convolution of {} values, a bias of {:?} and a state of {} for {d} channels of {} taps",
            w.len(),
            b.map(<[f32]>::len),
            st.u.len(),
            c.taps
        ));
    }
    Ok((w, b))
}

/// The input product of one token's normed input `h`: (u = B x, C).
fn gates(p: &Params, h: &[f32]) -> Result<(Vec<f32>, Vec<f32>)> {
    let d = h.len();
    let mix = p.prod("in", h)?;
    if mix.len() != 3 * d {
        return err(format!("short convolution: an input product of {} rows, where B, C and x are {d} each", mix.len()));
    }
    Ok(((0..d).map(|i| mix[i] * mix[2 * d + i]).collect(), mix[d..2 * d].to_vec()))
}

/// One token through the layer: the input product, the convolution over the state and this token's u (which the state then takes), the output gate, the output
/// product.
pub fn step(c: &ShortConv, p: &Params, st: &mut ConvState, h: &[f32]) -> Result<Vec<f32>> {
    let (d, k) = (h.len(), c.taps);
    let (w, b) = weights(c, p, st, d)?;
    let (u, cg) = gates(p, h)?;
    let hist = k - 1;
    let y: Vec<f32> = (0..d)
        .map(|ch| {
            // (channel ch's taps w[ch k ..], oldest first; its history takes this token's u)
            let s = &mut st.u[ch * hist..(ch + 1) * hist];
            let mut acc = b.map_or(0.0, |b| b[ch]);
            for (j, &v) in s.iter().enumerate() {
                acc += w[ch * k + j] * v;
            }
            acc += w[ch * k + hist] * u[ch];
            if hist > 0 {
                s.rotate_left(1);
                s[hist - 1] = u[ch];
            }
            cg[ch] * acc
        })
        .collect();
    p.prod("o", &y)
}

/// A run of tokens through the layer at once (`hs` [tokens, hidden], in order; the outputs so too): each channel's line is the state's values then the run's u,
/// token t's convolution the taps over the line's values t .. t + taps; the state then holds the line's last `taps` - 1 values. A sequence's prompt from nothing,
/// or its next part from the state the part before left.
pub fn prompt(c: &ShortConv, p: &Params, st: &mut ConvState, hs: &[f32], hidden: usize) -> Result<Vec<f32>> {
    let (d, k) = (hidden, c.taps);
    if d == 0 || !hs.len().is_multiple_of(d) {
        return err(format!("short convolution: a run of {} values for tokens of {d}", hs.len()));
    }
    let (w, b) = weights(c, p, st, d)?;
    let (n, hist) = (hs.len() / d, k - 1);
    let mut cgs = Vec::with_capacity(n);
    // (line[ch]: the state's `hist` values, then each token's u)
    let mut line = vec![0f32; d * (hist + n)];
    for ch in 0..d {
        line[ch * (hist + n)..][..hist].copy_from_slice(&st.u[ch * hist..(ch + 1) * hist]);
    }
    for (t, h) in hs.chunks(d).enumerate() {
        let (u, cg) = gates(p, h)?;
        for ch in 0..d {
            line[ch * (hist + n) + hist + t] = u[ch];
        }
        cgs.push(cg);
    }
    let mut out = Vec::with_capacity(n * d);
    for (t, cg) in cgs.iter().enumerate() {
        let y: Vec<f32> = (0..d)
            .map(|ch| {
                let mut acc = b.map_or(0.0, |b| b[ch]);
                for (j, &v) in line[ch * (hist + n) + t..][..k].iter().enumerate() {
                    acc += w[ch * k + j] * v;
                }
                cg[ch] * acc
            })
            .collect();
        out.extend(p.prod("o", &y)?);
    }
    for ch in 0..d {
        st.u[ch * hist..(ch + 1) * hist].copy_from_slice(&line[ch * (hist + n) + n..][..hist]);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::reference::testing::fill;

    const D: usize = 6;

    fn layer(taps: usize, bias: bool) -> (ShortConv, Params) {
        let mut p = Params::new();
        p.put("in", &[3 * D, D], fill("sc.in", 3 * D * D, -0.5, 0.5));
        p.put("conv", &[D, 1, taps], fill("sc.conv", D * taps, -0.5, 0.5));
        p.put("o", &[D, D], fill("sc.o", D * D, -0.5, 0.5));
        if bias {
            p.put("in_bias", &[3 * D], fill("sc.in_bias", 3 * D, -0.2, 0.2));
            p.put("conv_bias", &[D], fill("sc.conv_bias", D, -0.2, 0.2));
            p.put("o_bias", &[D], fill("sc.o_bias", D, -0.2, 0.2));
        }
        (ShortConv { taps, bias }, p)
    }

    fn steps(c: &ShortConv, p: &Params, x: &[f32]) -> Vec<f32> {
        let mut st = ConvState::new(c, D);
        x.chunks(D).flat_map(|h| step(c, p, &mut st, h).unwrap()).collect()
    }

    /// The step against the definition written out another way, in f64: token t's output is o (C_t * (bias + sum over j of w_j u_(t - (taps - 1) + j))) with u = B x of
    /// each token and nothing before the first; with and without biases, at 1, 4 and 7 taps (1: no state, the layer a gated product of the token alone).
    #[test]
    fn the_step_is_the_definition() {
        let tokens = 11;
        let x = fill("sc.input", tokens * D, -1.0, 1.0);
        for (taps, bias) in [(7, false), (7, true), (4, false), (1, false)] {
            let (c, p) = layer(taps, bias);
            assert_eq!(ConvState::new(&c, D).u.len(), D * (taps - 1));
            let got = steps(&c, &p, &x);
            let mixes: Vec<Vec<f32>> = x.chunks(D).map(|h| p.prod("in", h).unwrap()).collect();
            let (w, b) = (p.vec("conv").unwrap(), p.opt("conv_bias"));
            for t in 0..tokens {
                let y: Vec<f32> = (0..D)
                    .map(|ch| {
                        let mut acc = b.map_or(0.0, |b| b[ch] as f64);
                        for j in 0..taps {
                            // (tap j reads the token taps - 1 - j back)
                            if let Some(s) = (t + j + 1).checked_sub(taps) {
                                acc += w[ch * taps + j] as f64 * (mixes[s][ch] as f64 * mixes[s][2 * D + ch] as f64);
                            }
                        }
                        (mixes[t][D + ch] as f64 * acc) as f32
                    })
                    .collect();
                let want = p.prod("o", &y).unwrap();
                for (a, b) in got[t * D..][..D].iter().zip(&want) {
                    assert!((a - b).abs() <= 1e-5 * (1.0 + b.abs()), "{taps} taps, token {t}: {a} against {b}");
                }
            }
        }
    }

    /// A prompt at once, the same prompt a token at a time, and the prompt in parts of any length (a part shorter than the state among them) are the same
    /// bits, and leave the same state.
    #[test]
    fn a_prompt_is_its_steps_to_the_bit() {
        let tokens = 13;
        let x = fill("sc.input", tokens * D, -1.0, 1.0);
        for (taps, bias) in [(7, false), (7, true), (2, false), (1, true)] {
            let (c, p) = layer(taps, bias);
            let mut by_step = ConvState::new(&c, D);
            let want: Vec<f32> = x.chunks(D).flat_map(|h| step(&c, &p, &mut by_step, h).unwrap()).collect();
            let mut whole = ConvState::new(&c, D);
            assert_eq!(prompt(&c, &p, &mut whole, &x, D).unwrap(), want, "{taps} taps");
            assert_eq!(whole.u, by_step.u, "{taps} taps: the state");
            for cuts in [vec![1, 12], vec![5, 2, 6], vec![9, 1, 1, 2]] {
                let (mut st, mut got, mut at) = (ConvState::new(&c, D), Vec::new(), 0);
                for n in cuts {
                    got.extend(prompt(&c, &p, &mut st, &x[at * D..(at + n) * D], D).unwrap());
                    at += n;
                }
                assert_eq!((got, &st.u), (want.clone(), &by_step.u), "{taps} taps, in parts");
            }
        }
    }

    /// Causal, with a memory of exactly `taps` - 1 tokens: changing token s changes no earlier row, changes rows s ..= s + taps - 1, and no row after them; a
    /// second sequence starts from nothing.
    #[test]
    fn causal_with_a_memory_of_the_taps() {
        let (tokens, taps, s) = (16, 7, 4);
        let (c, p) = layer(taps, false);
        let x = fill("sc.input", tokens * D, -1.0, 1.0);
        let mut other = x.clone();
        other[s * D..(s + 1) * D].iter_mut().for_each(|v| *v = 0.5 - *v);
        let (a, b) = (steps(&c, &p, &x), steps(&c, &p, &other));
        for t in 0..tokens {
            let same = a[t * D..][..D] == b[t * D..][..D];
            assert_eq!(same, t < s || t > s + taps - 1, "token {t}");
        }
        assert_eq!(steps(&c, &p, &x[..D]), a[..D]);
    }

    #[test]
    fn shapes_are_checked() {
        let (c, p) = layer(7, false);
        assert!(step(&c, &p, &mut ConvState::new(&c, D), &[0.0; 4]).is_err());
        let wide = ShortConv { taps: 5, ..c };
        assert!(step(&wide, &p, &mut ConvState::new(&wide, D), &[0.0; D]).unwrap_err().msg.contains("short convolution"));
        assert!(step(&ShortConv { bias: true, ..c }, &p, &mut ConvState::new(&c, D), &[0.0; D]).is_err());
        assert!(prompt(&c, &p, &mut ConvState::new(&c, D), &[0.0; D + 1], D).is_err());
    }
}
