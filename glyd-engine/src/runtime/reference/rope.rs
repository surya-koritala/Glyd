//! Rotary position embedding, as transformers computes it: inverse frequencies in f32, angles pos x inv_freq in f32, cos and sin times
//! the scaling's attention factor.

use crate::runtime::kinds::{Rope, RopeScaling, RopeStyle};

/// The `dim / 2` inverse frequencies (the angles past `freqs` are 0) and the attention factor.
pub fn inv_freq(r: &Rope) -> (Vec<f32>, f32) {
    let base = |i: usize| -> f32 { 1.0 / (r.theta as f32).powf((2 * i) as f32 / r.freq_dim as f32) };
    let mut f: Vec<f32> = (0..r.dim / 2).map(|i| if i < r.freqs { base(i) } else { 0.0 }).collect();
    let af = match &r.scaling {
        RopeScaling::None => 1.0,
        RopeScaling::Linear { factor } => {
            f.iter_mut().for_each(|v| *v /= *factor as f32);
            1.0
        }
        _ => unimplemented!("rope scaling {:?} is not in the reference yet", r.scaling),
    };
    (f, af)
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
