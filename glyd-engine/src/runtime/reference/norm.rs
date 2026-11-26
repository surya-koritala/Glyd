//! Norms: RMSNorm, LayerNorm and the l2 norm, with the weight forms of `kinds::NormWeight` and the rounding points of `kinds::Rounding`.

use super::bf16;
use crate::runtime::kinds::{Norm, NormKind, NormWeight, Rounding};

/// `n` on `x` with its weight `w` (and LayerNorm's bias `b`) where the norm has them.
pub fn norm(n: &Norm, x: &[f32], w: Option<&[f32]>, b: Option<&[f32]>) -> Vec<f32> {
    let dim = x.len() as f32;
    let factor = |i: usize| -> f32 {
        let w = w.map_or(0.0, |w| w[i]);
        match n.weight {
            NormWeight::None => 1.0,
            NormWeight::Plain => w,
            NormWeight::PlusOne | NormWeight::InputPlusOne => 1.0 + w,
        }
    };
    let input: Vec<f32> = if n.weight == NormWeight::InputPlusOne { x.iter().enumerate().map(|(i, &v)| v * factor(i)).collect() } else { x.to_vec() };
    let (mean, r) = match n.kind {
        NormKind::Rms => (0.0, 1.0 / (input.iter().map(|&v| v * v).sum::<f32>() / dim + n.eps).sqrt()),
        NormKind::L2 => (0.0, 1.0 / (input.iter().map(|&v| v * v).sum::<f32>() + n.eps).sqrt()),
        NormKind::Identity => (0.0, 1.0),
        NormKind::Layer => {
            let m = input.iter().sum::<f32>() / dim;
            (m, 1.0 / (input.iter().map(|&v| (v - m) * (v - m)).sum::<f32>() / dim + n.eps).sqrt())
        }
    };
    input
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            let normed = (v - mean) * r;
            let f = if n.weight == NormWeight::InputPlusOne { 1.0 } else { factor(i) };
            let y = match n.rounding {
                Rounding::F32 | Rounding::Once => normed * f,
                Rounding::Twice => bf16(normed) * f,
            };
            let y = y + b.map_or(0.0, |b| b[i]);
            if n.rounding == Rounding::F32 {
                y
            } else {
                bf16(y)
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::reference::Libm;

    #[test]
    fn the_forms() {
        let x = [1.0f32, -2.0, 3.0, -4.0];
        let r = 1.0 / (30.0f32 / 4.0 + 1e-6).sqrt();
        let w = [0.5f32, 1.0, -1.0, 2.0];
        let got = norm(&Norm::rms(1e-6), &x, Some(&w), None);
        assert!(got.iter().zip(x.iter().zip(&w)).all(|(&g, (&x, &w))| (g - x * r * w).abs() < 1e-6));
        let got = norm(&Norm::rms_plus_one(1e-6), &x, Some(&w), None);
        assert!(got.iter().zip(x.iter().zip(&w)).all(|(&g, (&x, &w))| (g - x * r * (1.0 + w)).abs() < 1e-6));
        assert_eq!(norm(&Norm::rms_bare(1e-6), &x, None, None), x.iter().map(|&v| v * r).collect::<Vec<_>>());
        // the input-scaled form: a weight of 0 is the bare norm, and the norm of a scaled input does not depend on a common scale
        let n = Norm { weight: NormWeight::InputPlusOne, ..Norm::rms(0.0) };
        assert_eq!(norm(&n, &x, Some(&[0.0; 4]), None), norm(&Norm::rms_bare(0.0), &x, None, None));
        let a = norm(&n, &x, Some(&[1.0; 4]), None);
        assert!(a.iter().zip(norm(&Norm::rms_bare(0.0), &x, None, None)).all(|(a, b)| (a - b).abs() < 1e-6));
        // LayerNorm: zero mean, unit variance, then w and b
        let n = Norm { kind: NormKind::Layer, ..Norm::rms(0.0) };
        let y = norm(&n, &x, Some(&[1.0; 4]), Some(&[0.5; 4]));
        let m: f32 = y.iter().sum::<f32>() / 4.0;
        assert!((m - 0.5).abs() < 1e-6 && (y.iter().map(|v| (v - 0.5) * (v - 0.5)).sum::<f32>() / 4.0 - 1.0).abs() < 1e-5);
        // the l2 norm
        let n = Norm { kind: NormKind::L2, ..Norm::rms_bare(0.0) };
        assert_eq!(norm(&n, &[3.0, 4.0], None, None), vec![0.6, 0.8]);
    }

    /// Kimi's two roundings differ from one: bf16(w bf16(x r)) against bf16(w x r).
    #[test]
    fn the_rounding_points() {
        let x: Vec<f32> = (0..32).map(|i| (i as f32 * 0.37).lsin() * 3.0).collect();
        let w: Vec<f32> = (0..32).map(|i| 1.0 + i as f32 * 0.013).collect();
        let r = 1.0 / (x.iter().map(|v| v * v).sum::<f32>() / 32.0 + 1e-5).sqrt();
        let twice = Norm { rounding: Rounding::Twice, ..Norm::rms(1e-5) };
        let once = Norm { rounding: Rounding::Once, ..Norm::rms(1e-5) };
        assert_eq!(norm(&twice, &x, Some(&w), None), (0..32).map(|i| bf16(w[i] * bf16(x[i] * r))).collect::<Vec<_>>());
        assert_eq!(norm(&once, &x, Some(&w), None), (0..32).map(|i| bf16(x[i] * r * w[i])).collect::<Vec<_>>());
        assert_ne!(norm(&twice, &x, Some(&w), None), norm(&once, &x, Some(&w), None));
    }
}
