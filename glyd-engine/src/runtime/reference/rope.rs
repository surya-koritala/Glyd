//! Rotary position embedding, as transformers computes it: inverse frequencies in f32, angles pos x inv_freq in f32, cos and sin times
//! the scaling's attention factor.

use crate::runtime::kinds::{Rope, RopeScaling, RopeStyle};

/// The `dim / 2` inverse frequencies (the angles past `freqs` are 0) and the attention factor that multiplies cos and sin, for a
/// sequence of `seq_len` tokens (only dynamic NTK and longrope depend on it; the reference passes position + 1, the decode step's view).
/// transformers' `ROPE_INIT_FUNCTIONS`, in f64 where it uses Python floats.
pub fn inv_freq_at(r: &Rope, seq_len: u32) -> (Vec<f32>, f32) {
    let (dim, fd) = (r.dim, r.freq_dim as f64);
    let pos_freq = |i: usize, base: f64| base.powf((2 * i) as f64 / fd);
    let plain = |base: f64| -> Vec<f64> { (0..dim / 2).map(|i| if i < r.freqs { 1.0 / pos_freq(i, base) } else { 0.0 }).collect() };
    let (f, af): (Vec<f64>, f64) = match &r.scaling {
        RopeScaling::None => (plain(r.theta), 1.0),
        RopeScaling::Linear { factor } => (plain(r.theta).into_iter().map(|v| v / factor).collect(), 1.0),
        RopeScaling::DynamicNtk { factor, max_positions } => {
            let (seq, max) = (seq_len.max(*max_positions) as f64, *max_positions as f64);
            let base = r.theta * ((factor * seq / max) - (factor - 1.0)).powf(fd / (fd - 2.0));
            (plain(base), 1.0)
        }
        RopeScaling::Yarn { factor, original_max, beta_fast, beta_slow, mscale, mscale_all_dim, attention_factor, truncate } => {
            let get = |s: f64, m: f64| if s <= 1.0 { 1.0 } else { 0.1 * m * s.ln() + 1.0 };
            let af = attention_factor.unwrap_or(match (mscale, mscale_all_dim) {
                (Some(m), Some(a)) if *m != 0.0 && *a != 0.0 => get(*factor, *m) / get(*factor, *a),
                _ => get(*factor, 1.0),
            });
            let corr = |rot: f64| (fd * (*original_max as f64 / (rot * 2.0 * std::f64::consts::PI)).ln()) / (2.0 * r.theta.ln());
            let (mut lo, mut hi) = (corr(*beta_fast), corr(*beta_slow));
            if *truncate {
                (lo, hi) = (lo.floor(), hi.ceil());
            }
            let (lo, mut hi) = (lo.max(0.0), hi.min(fd - 1.0));
            if lo == hi {
                hi += 0.001;
            }
            let f = (0..dim / 2)
                .map(|i| {
                    let pf = pos_freq(i, r.theta);
                    let extra = 1.0 - ((i as f64 - lo) / (hi - lo)).clamp(0.0, 1.0);
                    (1.0 / (factor * pf)) * (1.0 - extra) + (1.0 / pf) * extra
                })
                .collect();
            (f, af)
        }
        RopeScaling::Llama3 { factor, low_freq_factor, high_freq_factor, original_max } => {
            let old = *original_max as f64;
            let (low_wl, high_wl) = (old / low_freq_factor, old / high_freq_factor);
            let f = plain(r.theta)
                .into_iter()
                .map(|inv| {
                    let wl = 2.0 * std::f64::consts::PI / inv;
                    if wl < high_wl {
                        inv
                    } else if wl > low_wl {
                        inv / factor
                    } else {
                        let smooth = (old / wl - low_freq_factor) / (high_freq_factor - low_freq_factor);
                        (1.0 - smooth) * inv / factor + smooth * inv
                    }
                })
                .collect();
            (f, 1.0)
        }
        RopeScaling::LongRope { short, long, original_max, max_positions, attention_factor } => {
            let factor = *max_positions as f64 / *original_max as f64;
            let af = attention_factor.unwrap_or(if factor <= 1.0 { 1.0 } else { (1.0 + factor.ln() / (*original_max as f64).ln()).sqrt() });
            let ext = if seq_len > *original_max { long } else { short };
            ((0..dim / 2).map(|i| 1.0 / (ext[i] * pos_freq(i, r.theta))).collect(), af)
        }
    };
    (f.into_iter().map(|v| v as f32).collect(), af as f32)
}

/// The frequencies at a sequence length that changes none of them.
pub fn inv_freq(r: &Rope) -> (Vec<f32>, f32) {
    inv_freq_at(r, 0)
}

/// Rotates one head `x` (its first `r.dim` values) to position `pos`.
pub fn apply(r: &Rope, inv: &[f32], af: f32, x: &mut [f32], pos: u32) {
    let half = r.dim / 2;
    let cs: Vec<(f32, f32)> = inv
        .iter()
        .map(|&f| {
            let a = pos as f32 * f;
            (((a as f64).cos() as f32) * af, ((a as f64).sin() as f32) * af)
        })
        .collect();
    match r.style {
        RopeStyle::Half => {
            for i in 0..half {
                let (c, s) = cs[i];
                let (a, b) = (x[i], x[i + half]);
                x[i] = a * c - b * s;
                x[i + half] = b * c + a * s;
            }
        }
        RopeStyle::Interleaved => {
            for i in 0..half {
                let (c, s) = cs[i];
                let (a, b) = (x[2 * i], x[2 * i + 1]);
                x[2 * i] = a * c - b * s;
                x[2 * i + 1] = b * c + a * s;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Position 0 is the identity; a rotation keeps each pair's length; the score of a rotated query and key depends only on the
    /// positions' difference.
    #[test]
    fn rotations() {
        for r in [Rope::plain(10_000.0, 8), Rope::proportional(1e6, 8, 0.5), Rope { style: RopeStyle::Interleaved, ..Rope::plain(100.0, 8) }] {
            let (inv, af) = inv_freq(&r);
            let x: Vec<f32> = (0..10).map(|i| (i as f32 * 0.7).sin()).collect();
            let mut y = x.clone();
            apply(&r, &inv, af, &mut y, 0);
            assert_eq!(x, y);
            apply(&r, &inv, af, &mut y, 37);
            assert_eq!(&x[8..], &y[8..]);
            let len = |v: &[f32]| v.iter().map(|a| a * a).sum::<f32>();
            assert!((len(&x[..8]) - len(&y[..8])).abs() < 1e-5);
            let score = |pq: u32, pk: u32| {
                let (mut q, mut k) = (x.clone(), x.iter().rev().cloned().collect::<Vec<_>>());
                apply(&r, &inv, af, &mut q, pq);
                apply(&r, &inv, af, &mut k, pk);
                q.iter().zip(&k).map(|(a, b)| a * b).sum::<f32>()
            };
            assert!((score(5, 2) - score(103, 100)).abs() < 1e-4);
        }
        // proportional rope: a quarter of the angles turn, the pairs of the rest stay
        let r = Rope::proportional(1e6, 16, 0.25);
        let (inv, _) = inv_freq(&r);
        assert_eq!((inv.len(), inv.iter().filter(|&&f| f != 0.0).count()), (8, 2));
        assert_eq!(inv[1], 1.0 / 1e6f32.powf(2.0 / 16.0));
    }
}
