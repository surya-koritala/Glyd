//! Multi-head latent attention (`kinds::Mla`), in the expanded form: each cached latent through `kv_b` gives a head's key (nope part) and
//! value; the rope part of the key is one per token, shared by the heads. The absorbed form a backend runs is the same function. With an
//! indexer a query attends only to the tokens it picks (GLM-5.3-Flash: pooled keys; `picks`).

use super::{act, dot, norm::norm, rope, softmax, Params};
use crate::ffi::{err, Result};
use crate::runtime::kinds::{IndexForm, Indexer, Mla, NormKind};

/// A sequence's latents in one layer: per token the normed latent (kv_lora values) then the shared key part (rope_dim, rotated); and where
/// the layer has an indexer of its own, per token its normed key and its pool gate's scores; and the last query's pick `margin`.
#[derive(Default)]
pub struct LatentCache {
    pub pos: Vec<u32>,
    pub lat: Vec<Vec<f32>>,
    pub ikey: Vec<Vec<f32>>,
    pub igate: Vec<Vec<f32>>,
    pub margin: f32,
}

/// A gated indexer's key norm (`IndexForm::Gated`'s `key`) on the key product `k`: its weight, and a LayerNorm's bias.
fn key_norm(ix: &Indexer, p: &Params, k: &[f32]) -> Result<Vec<f32>> {
    let IndexForm::Gated { key } = ix.form else { return err(format!("a latent attention's indexer of the gated form, not {:?}", ix.form)) };
    let bias = if key.kind == NormKind::Layer { Some(p.vec("indexer_k_norm_bias")?) } else { None };
    Ok(norm(&key, k, Some(p.vec("indexer_k_norm")?), bias))
}

/// Which of the cache entries `seen` (in order, the sequence's first token first) the query attends to. While the complete pools of `block`
/// tokens number no more than the `top_k / block` it keeps, every one; else the pools of the best scores, each pool's key the per-channel
/// softmax(gate + ape) weighted sum of its tokens' normed keys, a pool's score sum_h w_h relu(q_h . key / sqrt(head_dim)) with w the
/// weights' product of the block's input over sqrt(heads) and q the indexer's product of the normed query residual `qr`; then the tokens
/// of the incomplete last pool (transformers' `Glm5NextTextIndexer`, `index_kpool_always_select_tail`). Equal scores keep the earlier pool.
/// With the picks, how near the choice was to another (`ffn::margin` of the pools' scores).
pub fn picks(ix: &Indexer, p: &Params, c: &LatentCache, seen: &[usize], qr: Option<&[f32]>, h: &[f32]) -> Result<(Vec<usize>, f32)> {
    let (b, dim) = (ix.block.max(1), ix.head_dim);
    let (pools, keep) = (seen.len() / b, ix.top_k / b);
    if pools <= keep {
        return Ok((seen.to_vec(), f32::INFINITY));
    }
    if ix.shared || c.ikey.len() != c.lat.len() {
        return err("MLA: a shared indexer reuses the previous layer's picks, which the reference does not hold");
    }
    let qr = qr.ok_or_else(|| crate::ffi::Error("MLA: an indexer reads the query's low rank, and this layer has none".into()))?;
    let q = p.prod("indexer_wq_b", qr)?;
    let w: Vec<f32> = p.prod("indexer_weights", h)?.iter().map(|x| x * (ix.heads as f32).powf(-0.5)).collect();
    if q.len() != ix.heads * dim || w.len() != ix.heads {
        return err(format!("MLA: an indexer of {} query values and {} weights for {} heads of {dim}", q.len(), w.len(), ix.heads));
    }
    let ape = if b > 1 { p.vec("indexer_pool_ape")? } else { &[] };
    let sc = (dim as f32).powf(-0.5);
    let score: Vec<f32> = (0..pools)
        .map(|j| {
            let toks = &seen[j * b..(j + 1) * b];
            let key: Vec<f32> = (0..dim)
                .map(|ch| {
                    if b == 1 {
                        return c.ikey[toks[0]][ch];
                    }
                    let mut g: Vec<f32> = toks.iter().enumerate().map(|(i, &t)| c.igate[t][ch] + ape[i * dim + ch]).collect();
                    softmax(&mut g);
                    toks.iter().zip(&g).map(|(&t, &a)| a as f64 * c.ikey[t][ch] as f64).sum::<f64>() as f32
                })
                .collect();
            (0..ix.heads).map(|hh| w[hh] * (dot(&q[hh * dim..][..dim], &key) * sc).max(0.0)).sum()
        })
        .collect();
    let mut ord: Vec<usize> = (0..pools).collect();
    ord.sort_by(|&x, &y| score[y].total_cmp(&score[x]).then(x.cmp(&y)));
    ord.truncate(keep);
    ord.sort_unstable();
    let mut out: Vec<usize> = ord.iter().flat_map(|&j| seen[j * b..(j + 1) * b].iter().copied()).collect();
    out.extend_from_slice(&seen[pools * b..]);
    Ok((out, super::ffn::margin(&score, keep)))
}

pub fn step(m: &Mla, p: &Params, c: &mut LatentCache, h: &[f32], pos: u32) -> Result<Vec<f32>> {
    let (nh, kr, nope, rd, vd) = (m.heads, m.kv_lora, m.nope, m.rope_dim, m.v_dim);
    let qw = nope + rd;
    let (q, qr) = match m.q_lora {
        Some(_) => {
            let qr = norm(&m.q_norm, &p.prod("q_a", h)?, p.opt("q_a_norm"), None);
            (p.prod("q_b", &qr)?, Some(qr))
        }
        None => (p.prod("q", h)?, None),
    };
    let ckr = p.prod("kv_a", h)?;
    if q.len() != nh * qw || ckr.len() != kr + rd {
        return err(format!("MLA: {} query values for {nh} x {qw}, {} latent values for {kr} + {rd}", q.len(), ckr.len()));
    }
    let rot = m.rope.as_ref().map(|r| (r, rope::inv_freq_at(r, pos + 1)));
    let mut lat = norm(&m.kv_norm, &ckr[..kr], p.opt("kv_a_norm"), None);
    let mut kr_part = ckr[kr..].to_vec();
    if let Some((r, (inv, af))) = &rot {
        rope::apply(r, inv, *af, &mut kr_part, pos);
    }
    lat.extend(kr_part);
    c.pos.push(pos);
    c.lat.push(lat);
    // (the indexer's key and pool gate of this token, where the layer has an indexer of its own)
    if let Some(ix) = m.indexer.as_ref().filter(|_| p.has("indexer_wk")) {
        c.ikey.push(key_norm(ix, p, &p.prod("indexer_wk", h)?)?);
        c.igate.push(if ix.block > 1 { p.prod("indexer_pool_gate", h)? } else { Vec::new() });
    }
    let kv_b = p.vec("kv_b")?;
    if kv_b.len() != nh * (nope + vd) * kr {
        return err(format!("MLA: kv_b of {} values for {nh} x ({nope} + {vd}) x {kr}", kv_b.len()));
    }
    let gate = match m.gate {
        Some(_) => Some(p.prod("g", h)?),
        None => None,
    };
    let seen: Vec<usize> = (0..c.pos.len()).filter(|&j| c.pos[j] <= pos).collect();
    let (seen, margin) = match &m.indexer {
        Some(ix) => picks(ix, p, c, &seen, qr.as_deref(), h)?,
        None => (seen, f32::INFINITY),
    };
    c.margin = margin;
    let mut out = Vec::with_capacity(nh * vd);
    for head in 0..nh {
        let rows = |r: usize| &kv_b[(head * (nope + vd) + r) * kr..][..kr];
        let mut qh = q[head * qw..][..qw].to_vec();
        if let Some((r, (inv, af))) = &rot {
            rope::apply(r, inv, *af, &mut qh[nope..], pos);
        }
        let mut s: Vec<f32> = seen
            .iter()
            .map(|&j| {
                let l = &c.lat[j];
                let k_nope: Vec<f32> = (0..nope).map(|d| dot(rows(d), &l[..kr])).collect();
                (dot(&qh[..nope], &k_nope) + dot(&qh[nope..], &l[kr..])) * m.scale
            })
            .collect();
        softmax(&mut s);
        let mut o = vec![0f64; vd];
        for (&j, &pj) in seen.iter().zip(&s) {
            for (d, ov) in o.iter_mut().enumerate() {
                *ov += pj as f64 * dot(rows(nope + d), &c.lat[j][..kr]) as f64;
            }
        }
        let mut o: Vec<f32> = o.into_iter().map(|v| v as f32).collect();
        if let (Some(g), Some(gv)) = (m.gate, &gate) {
            if let Some(n) = g.norm {
                o = norm(&n, &o, None, None);
            }
            o.iter_mut().zip(&gv[head * vd..][..vd]).for_each(|(x, &z)| *x *= act(g.act, z));
        }
        out.extend(o);
    }
    p.prod("o", &out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::kinds::Norm;
    use crate::runtime::reference::testing::{fill, rel};

    const D: usize = 32;

    /// A GLM-5.3-Flash-shaped latent attention (NoPE, q low rank) with an indexer of `heads` heads of 16 keeping `top_k` tokens in pools of `block`.
    fn layer(top_k: usize, block: usize) -> (Mla, Params) {
        let ix = Indexer { heads: 2, head_dim: 16, top_k, block, shared: false, form: crate::runtime::kinds::IndexForm::LAYER_KEYS };
        let m = Mla {
            heads: 2,
            q_lora: Some(24),
            kv_lora: 16,
            nope: 16,
            rope_dim: 0,
            v_dim: 16,
            rope: None,
            q_norm: Norm::rms(1e-6),
            kv_norm: Norm::rms(1e-6),
            gate: None,
            scale: 0.25,
            absorbed: true,
            indexer: Some(ix),
        };
        let mut p = Params::new();
        for (role, shape, lo, hi) in [
            ("q_a", [24, D], -0.3, 0.3),
            ("q_b", [32, 24], -0.3, 0.3),
            ("kv_a", [16, D], -0.3, 0.3),
            ("kv_b", [64, 16], -0.3, 0.3),
            ("o", [D, 32], -0.3, 0.3),
            ("indexer_wq_b", [32, 24], -0.5, 0.5),
            ("indexer_wk", [16, D], -0.5, 0.5),
            ("indexer_weights", [2, D], -0.5, 0.5),
            ("indexer_pool_gate", [16, D], -0.5, 0.5),
            ("indexer_pool_ape", [block, 16], -0.5, 0.5),
        ] {
            p.put(role, &shape, fill(&format!("mla.{role}"), shape[0] * shape[1], lo, hi));
        }
        for (role, n, lo, hi) in
            [("q_a_norm", 24, 0.5, 1.5), ("kv_a_norm", 16, 0.5, 1.5), ("indexer_k_norm", 16, 0.5, 1.5), ("indexer_k_norm_bias", 16, -0.1, 0.1)]
        {
            p.put(role, &[n], fill(&format!("mla.{role}"), n, lo, hi));
        }
        (m, p)
    }

    fn inputs(n: usize) -> Vec<Vec<f32>> {
        (0..n).map(|t| fill(&format!("mla.h{t}"), D, -1.0, 1.0)).collect()
    }

    /// While the context holds no more complete pools than the indexer keeps, every token is attended to: the layer is the one without an
    /// indexer, token for token; with more it is not.
    #[test]
    fn an_indexer_keeps_everything_until_its_pools_run_out() {
        let (m, p) = layer(8, 4);
        let plain = Mla { indexer: None, ..m.clone() };
        let (mut a, mut b) = (LatentCache::default(), LatentCache::default());
        for (t, h) in inputs(16).iter().enumerate() {
            let (x, y) = (step(&m, &p, &mut a, h, t as u32).unwrap(), step(&plain, &p, &mut b, h, t as u32).unwrap());
            // (2 pools kept: 11 tokens are 2 pools and a tail of 3; 12 are 3 pools)
            if t < 11 {
                assert_eq!(x, y, "token {t}");
            } else {
                assert!(rel(&x, &y) > 1e-4, "token {t}: the indexer chose everything");
            }
        }
    }

    /// The gated form's key norm is the kind's: RMS-normed keys (Kimi-K3.1's) read no bias and pick as their keys score, other picks than the
    /// LayerNorm-ed keys'; until the pools run out the layer is the one without an indexer either way.
    #[test]
    fn the_keys_are_normed_as_the_form_says() {
        let (m, mut p) = layer(8, 1);
        let rms = Mla { indexer: m.indexer.clone().map(|ix| Indexer { form: IndexForm::Gated { key: Norm::rms(1e-6) }, ..ix }), ..m.clone() };
        let h = &inputs(1)[0];
        let k = p.prod("indexer_wk", h).unwrap();
        assert_eq!(key_norm(rms.indexer.as_ref().unwrap(), &p, &k).unwrap(), norm(&Norm::rms(1e-6), &k, p.opt("indexer_k_norm"), None));
        let layer_keys = key_norm(m.indexer.as_ref().unwrap(), &p, &k).unwrap();
        assert!(rel(&layer_keys, &norm(&Norm::rms(1e-6), &k, p.opt("indexer_k_norm"), None)) > 1e-3);
        p.t.remove("indexer_k_norm_bias");
        assert!(key_norm(m.indexer.as_ref().unwrap(), &p, &k).is_err(), "a LayerNorm's key reads its bias");
        let plain = Mla { indexer: None, ..m.clone() };
        let (mut a, mut b) = (LatentCache::default(), LatentCache::default());
        for (t, h) in inputs(8).iter().enumerate() {
            let (x, y) = (step(&rms, &p, &mut a, h, t as u32).unwrap(), step(&plain, &p, &mut b, h, t as u32).unwrap());
            assert_eq!(x, y, "token {t}: 8 tokens kept");
        }
    }

    /// The picks by hand: each complete pool's key and score written out for one query, the best two pools' tokens and the tail; and the
    /// layer's output is the attention over those tokens alone.
    #[test]
    fn the_picks_are_the_best_pools_and_the_tail() {
        let (m, p) = layer(8, 4);
        let ix = m.indexer.clone().unwrap();
        let hs = inputs(23);
        let mut c = LatentCache::default();
        for (t, h) in hs.iter().enumerate().take(22) {
            step(&m, &p, &mut c, h, t as u32).unwrap();
        }
        let h = &hs[22];
        let qr = norm(&m.q_norm, &p.prod("q_a", h).unwrap(), p.opt("q_a_norm"), None);
        // (the cache as step leaves it for the 23rd token, which joins before the choice)
        let mut c2 = LatentCache { pos: c.pos.clone(), lat: c.lat.clone(), ikey: c.ikey.clone(), igate: c.igate.clone(), margin: 0.0 };
        let k = key_norm(&ix, &p, &p.prod("indexer_wk", h).unwrap()).unwrap();
        c2.ikey.push(k);
        c2.igate.push(p.prod("indexer_pool_gate", h).unwrap());
        c2.lat.push(Vec::new());
        let seen: Vec<usize> = (0..23).collect();
        let (got, margin) = picks(&ix, &p, &c2, &seen, Some(&qr), h).unwrap();
        assert!(margin > 0.0 && margin.is_finite());
        // by hand: 5 complete pools of 4, scores, the best two in order, the tail 20, 21, 22
        let q = p.prod("indexer_wq_b", &qr).unwrap();
        let w = p.prod("indexer_weights", h).unwrap();
        let (ape, sc) = (p.vec("indexer_pool_ape").unwrap(), 0.25f64);
        let score: Vec<f64> = (0..5)
            .map(|j| {
                let key: Vec<f64> = (0..16)
                    .map(|ch| {
                        let g: Vec<f64> = (0..4).map(|i| (c2.igate[4 * j + i][ch] + ape[i * 16 + ch]) as f64).collect();
                        let z: f64 = g.iter().map(|x| x.exp()).sum();
                        (0..4).map(|i| g[i].exp() / z * c2.ikey[4 * j + i][ch] as f64).sum()
                    })
                    .collect();
                (0..2).map(|hh| w[hh] as f64 / 2f64.sqrt() * ((0..16).map(|d| q[hh * 16 + d] as f64 * key[d]).sum::<f64>() * sc).max(0.0)).sum()
            })
            .collect();
        let mut best: Vec<usize> = (0..5).collect();
        best.sort_by(|&a, &b| score[b].total_cmp(&score[a]));
        assert!(score[best[1]] - score[best[2]] > 1e-4, "the second and third pools are too near to tell: {score:?}");
        let mut want: Vec<usize> = best[..2].to_vec();
        want.sort_unstable();
        let mut want: Vec<usize> = want.iter().flat_map(|&j| 4 * j..4 * j + 4).collect();
        want.extend([20, 21, 22]);
        assert_eq!(got, want);
        // the output: the plain attention over those tokens alone (their latents in a cache of their own, at their positions)
        let mut c3 = LatentCache::default();
        let got_out = step(&m, &p, &mut c, h, 22).unwrap();
        let plain = Mla { indexer: None, ..m.clone() };
        let mut out = Vec::new();
        for &t in &want {
            out = step(&plain, &p, &mut c3, &hs[t], t as u32).unwrap();
        }
        assert!(rel(&got_out, &out) < 1e-6, "the attention over the picks");
    }

    /// One key a token (`block` 1, DeepSeek-V3.2's form): the best `top_k` tokens, no tail.
    #[test]
    fn a_key_a_token() {
        let (m, mut p) = layer(5, 1);
        p.t.remove("indexer_pool_gate");
        p.t.remove("indexer_pool_ape");
        let mut c = LatentCache::default();
        let hs = inputs(9);
        for (t, h) in hs.iter().enumerate() {
            step(&m, &p, &mut c, h, t as u32).unwrap();
        }
        let qr = norm(&m.q_norm, &p.prod("q_a", &hs[8]).unwrap(), p.opt("q_a_norm"), None);
        let got = picks(m.indexer.as_ref().unwrap(), &p, &c, &(0..9).collect::<Vec<_>>(), Some(&qr), &hs[8]).unwrap().0;
        assert_eq!(got.len(), 5);
        assert!(got.windows(2).all(|w| w[0] < w[1]));
    }
}
