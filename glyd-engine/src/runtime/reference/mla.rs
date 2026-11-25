//! Multi-head latent attention (`kinds::Mla`), in the expanded form: each cached latent through `kv_b` gives a head's key (nope part) and
//! value; the rope part of the key is one per token, shared by the heads. The absorbed form a backend runs is the same function. With an
//! indexer a query attends only to the tokens it picks (GLM-5.3-Flash: pooled keys; `picks`).
//!
//! Not checked against transformers: the indexer's rotary (`Indexer::rope`) and a `shared` indexer's lent picks. Both are read from transformers'
//! `GlmMoeDsaIndexer` and `GlmMoeDsaAttention` and checked by hand in this file's tests alone. The family's two recorded fixtures
//! (tests/runtime/glm53_mla.json and glm53_model.json; tests/runtime/record.py's cases of those names) keep every token, their `index_topk` of 64
//! being past their prompt of 7, and name every layer's indexer `full`: the reference agrees with transformers on them whatever it does with either.
//! A check takes those two cases recorded again with an `index_topk` below the prompt's 7 tokens (4, say), and the model's with
//! `indexer_types` of a `shared` layer after a `full` one; record.py runs where torch and a transformers that has `glm_moe_dsa` are installed (a CPU
//! is enough), and `hf_tests` then holds this reference to what it wrote.

use super::{act, dot, norm::norm, rope, softmax, Params};
use crate::ffi::{err, Result};
use crate::runtime::kinds::{IndexForm, Indexer, Mla, NormKind, Rope};

/// A sequence's latents in one layer: per token the normed latent (kv_lora values) then the shared key part (rope_dim, rotated); and where
/// the layer has an indexer of its own, per token its normed key and its pool gate's scores; and the last query's pick `margin`. `picks`:
/// the entries the last query attended to, where the layer has an indexer; `lent`: for a layer whose indexer is `shared`, the picks of the
/// latent attention before it for the query about to run (the model lends them: `RefModel::share_into`), taken by that query.
#[derive(Default)]
pub struct LatentCache {
    pub pos: Vec<u32>,
    pub lat: Vec<Vec<f32>>,
    pub ikey: Vec<Vec<f32>>,
    pub igate: Vec<Vec<f32>>,
    pub margin: f32,
    pub picks: Vec<usize>,
    pub lent: Option<Vec<usize>>,
}

/// A gated indexer's key norm (`IndexForm::Gated`'s `key`) on the key product `k`: its weight, and a LayerNorm's bias.
fn key_norm(ix: &Indexer, p: &Params, k: &[f32]) -> Result<Vec<f32>> {
    let IndexForm::Gated { key } = ix.form else { return err(format!("a latent attention's indexer of the gated form, not {:?}", ix.form)) };
    let bias = if key.kind == NormKind::Layer { Some(p.vec("indexer_k_norm_bias")?) } else { None };
    Ok(norm(&key, k, Some(p.vec("indexer_k_norm")?), bias))
}

/// The indexer's rotary (`Indexer::rope`): the first `rope` channels of a token's key or of a query head `x` turned to `pos` by the attention's rotary
/// `rot` (its tables at this position); nothing with no rotary of the indexer's.
fn index_turn(ix: &Indexer, rot: &Option<(&Rope, (Vec<f32>, f32))>, x: &mut [f32], pos: u32) -> Result<()> {
    if ix.rope == 0 {
        return Ok(());
    }
    match rot {
        Some((r, (inv, af))) if r.dim == ix.rope && x.len() >= r.dim => {
            rope::apply(r, inv, *af, x, pos);
            Ok(())
        }
        _ => err(format!(
            "MLA: an indexer whose first {} channels are rotated, under an attention's rotary of {:?} channels",
            ix.rope,
            rot.as_ref().map(|r| r.0.dim)
        )),
    }
}

/// Which of the cache entries `seen` (in order, the sequence's first token first) the query attends to. While the complete pools of `block`
/// tokens number no more than the `top_k / block` it keeps, every one; else the pools of the best scores, each pool's key the per-channel
/// softmax(gate + ape) weighted sum of its tokens' normed keys, a pool's score sum_h w_h relu(q_h . key / sqrt(head_dim)) with w the
/// weights' product of the block's input over sqrt(heads) and q the indexer's product of the normed query residual `qr`; then the tokens
/// of the incomplete last pool (transformers' `Glm5NextTextIndexer`, `index_kpool_always_select_tail`). Equal scores keep the earlier pool.
/// With the picks, how near the choice was to another (`ffn::margin` of the pools' scores). `turn`: the indexer's rotary of a query head at
/// the query's position (`index_turn`; the keys in the cache were turned at theirs).
pub fn picks(
    ix: &Indexer,
    p: &Params,
    c: &LatentCache,
    seen: &[usize],
    qr: Option<&[f32]>,
    h: &[f32],
    turn: &dyn Fn(&mut [f32]) -> Result<()>,
) -> Result<(Vec<usize>, f32)> {
    let (b, dim) = (ix.block.max(1), ix.head_dim);
    let (pools, keep) = (seen.len() / b, ix.top_k / b);
    if pools <= keep {
        return Ok((seen.to_vec(), f32::INFINITY));
    }
    if ix.shared || c.ikey.len() != c.lat.len() {
        return err("MLA: a shared indexer reuses the previous layer's picks, which the reference does not hold");
    }
    let qr = qr.ok_or_else(|| crate::ffi::Error("MLA: an indexer reads the query's low rank, and this layer has none".into()))?;
    let mut q = p.prod("indexer_wq_b", qr)?;
    let w: Vec<f32> = p.prod("indexer_weights", h)?.iter().map(|x| x * (ix.heads as f32).powf(-0.5)).collect();
    if q.len() != ix.heads * dim || w.len() != ix.heads {
        return err(format!("MLA: an indexer of {} query values and {} weights for {} heads of {dim}", q.len(), w.len(), ix.heads));
    }
    for head in q.chunks_mut(dim) {
        turn(head)?;
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
        let mut key = key_norm(ix, p, &p.prod("indexer_wk", h)?)?;
        index_turn(ix, &rot, &mut key, pos)?;
        c.ikey.push(key);
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
        // (a shared indexer runs none: the query attends to what the latent attention before it picked for it, transformers' `prev_topk_indices`)
        Some(ix) if ix.shared => match c.lent.take() {
            Some(lent) => (lent, f32::INFINITY),
            None => return err("MLA: a shared indexer attends to the picks of the latent attention before it, and none were lent to it"),
        },
        Some(ix) => picks(ix, p, c, &seen, qr.as_deref(), h, &|x: &mut [f32]| index_turn(ix, &rot, x, pos))?,
        None => (seen, f32::INFINITY),
    };
    c.margin = margin;
    if m.indexer.is_some() {
        c.picks = seen.clone();
    }
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
        let ix = Indexer { heads: 2, head_dim: 16, top_k, block, shared: false, form: crate::runtime::kinds::IndexForm::LAYER_KEYS, rope: 0 };
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
        let mut c2 = LatentCache { pos: c.pos.clone(), lat: c.lat.clone(), ikey: c.ikey.clone(), igate: c.igate.clone(), ..Default::default() };
        let k = key_norm(&ix, &p, &p.prod("indexer_wk", h).unwrap()).unwrap();
        c2.ikey.push(k);
        c2.igate.push(p.prod("indexer_pool_gate", h).unwrap());
        c2.lat.push(Vec::new());
        let seen: Vec<usize> = (0..23).collect();
        let (got, margin) = picks(&ix, &p, &c2, &seen, Some(&qr), h, &|_| Ok(())).unwrap();
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

    /// The indexer's rotary (the full-size GLM's: transformers' `GlmMoeDsaIndexer`): the first `rope` channels of each query head and
    /// of each token's key turned by the attention's rotary to their positions before the scores. By hand for one query; other picks than the same
    /// keys unturned; and refused where the attention has no rotary of that many channels.
    #[test]
    fn the_indexers_rotary_turns_its_queries_and_keys() {
        let (m0, mut p) = layer(4, 1);
        let r = Rope { style: crate::runtime::kinds::RopeStyle::Interleaved, ..Rope::plain(1e4, 8) };
        // (a head's query is 16 + 8 wide, a token's latent 16 with the shared key part's 8)
        p.put("q_b", &[48, 24], fill("mla.q_b.turned", 48 * 24, -0.3, 0.3));
        p.put("kv_a", &[24, D], fill("mla.kv_a.turned", 24 * D, -0.3, 0.3));
        let ix = Indexer { rope: 8, ..m0.indexer.clone().unwrap() };
        let m = Mla { rope_dim: 8, rope: Some(r.clone()), scale: 24f32.powf(-0.5), indexer: Some(ix.clone()), ..m0.clone() };
        let unturned = Mla { indexer: Some(Indexer { rope: 0, ..ix.clone() }), ..m.clone() };
        let hs = inputs(12);
        let (mut a, mut b) = (LatentCache::default(), LatentCache::default());
        let mut differ = false;
        for (t, h) in hs.iter().enumerate() {
            let (x, y) = (step(&m, &p, &mut a, h, t as u32).unwrap(), step(&unturned, &p, &mut b, h, t as u32).unwrap());
            differ |= t >= 4 && rel(&x, &y) > 1e-4;
        }
        assert!(differ, "the rotary changed no pick");
        // by hand, the last query (position 11): each key LayerNorm-ed then turned at its position, the query's heads at 11
        let (pos, h) = (11u32, &hs[11]);
        let qr = norm(&m.q_norm, &p.prod("q_a", h).unwrap(), p.opt("q_a_norm"), None);
        let mut q = p.prod("indexer_wq_b", &qr).unwrap();
        let (inv, af) = rope::inv_freq_at(&r, pos + 1);
        for head in q.chunks_mut(16) {
            rope::apply(&r, &inv, af, head, pos);
        }
        let w: Vec<f32> = p.prod("indexer_weights", h).unwrap().iter().map(|x| x / 2f32.sqrt()).collect();
        let score: Vec<f32> = (0..12)
            .map(|t| {
                let mut k = key_norm(&ix, &p, &p.prod("indexer_wk", &hs[t]).unwrap()).unwrap();
                let (inv, af) = rope::inv_freq_at(&r, t as u32 + 1);
                rope::apply(&r, &inv, af, &mut k, t as u32);
                assert_eq!(k, a.ikey[t], "the cache's key of token {t} is the turned one");
                (0..2).map(|hh| w[hh] * (dot(&q[hh * 16..][..16], &k) * 0.25).max(0.0)).sum()
            })
            .collect();
        let mut want: Vec<usize> = (0..12).collect();
        want.sort_by(|&x, &y| score[y].total_cmp(&score[x]).then(x.cmp(&y)));
        want.truncate(4);
        want.sort_unstable();
        let turn = |x: &mut [f32]| {
            rope::apply(&r, &inv, af, x, pos);
            Ok(())
        };
        let got = picks(&ix, &p, &a, &(0..12).collect::<Vec<_>>(), Some(&qr), h, &turn).unwrap().0;
        assert_eq!(got, want);
        assert!(index_turn(&ix, &None, &mut [0f32; 16], 0).is_err(), "an indexer's rotary with no rotary of the attention's");
        assert!(index_turn(&Indexer { rope: 4, ..ix.clone() }, &Some((&r, (inv.clone(), af))), &mut [0f32; 16], 0).is_err(), "of other channels");
    }

    /// A layer whose indexer is `shared` (the full-size GLM's `indexer_types`) runs no indexer: its query attends to the picks it is lent, the
    /// latent attention's before it. With that layer's own weights and its picks it is that layer, token for token; with none lent it says so.
    #[test]
    fn a_shared_indexer_attends_to_the_picks_it_is_lent() {
        let (m, p) = layer(4, 1);
        let shared = Mla { indexer: m.indexer.clone().map(|ix| Indexer { shared: true, ..ix }), ..m.clone() };
        // (a shared layer has no indexer weights)
        let mut ps = Params::new();
        for role in ["q_a", "q_b", "kv_a", "kv_b", "o", "q_a_norm", "kv_a_norm"] {
            ps.t.insert(role.to_string(), p.t[role].clone());
        }
        let (mut a, mut b) = (LatentCache::default(), LatentCache::default());
        let mut chose = false;
        for (t, h) in inputs(12).iter().enumerate() {
            let x = step(&m, &p, &mut a, h, t as u32).unwrap();
            chose |= a.picks.len() < t + 1;
            assert_eq!(a.picks.len(), (t + 1).min(4), "token {t}");
            b.lent = Some(a.picks.clone());
            let y = step(&shared, &ps, &mut b, h, t as u32).unwrap();
            assert_eq!((x, b.picks.clone(), b.lent.is_none(), b.ikey.len()), (y, a.picks.clone(), true, 0), "token {t}");
        }
        assert!(chose, "the indexer kept every token");
        let e = step(&shared, &ps, &mut b, &inputs(13)[12], 12).unwrap_err().msg;
        assert!(e.contains("none were lent"), "{e}");
    }

    /// One key a token (`block` 1, the form Kimi-K3.1's `dsa_config` has): the best `top_k` tokens, no tail.
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
        let ix = m.indexer.as_ref().unwrap();
        let got = picks(ix, &p, &c, &(0..9).collect::<Vec<_>>(), Some(&qr), &hs[8], &|_| Ok(())).unwrap().0;
        // by hand: each token's own key, its score sum_h w_h / sqrt(heads) relu(q_h . key / sqrt(head_dim)), and the five best in the tokens' order
        let q = p.prod("indexer_wq_b", &qr).unwrap();
        let w = p.prod("indexer_weights", &hs[8]).unwrap();
        let (heads, dim) = (ix.heads, ix.head_dim);
        assert_eq!((c.ikey.len(), q.len(), w.len()), (9, heads * dim, heads));
        let score: Vec<f64> = (0..9)
            .map(|t| {
                (0..heads)
                    .map(|hh| {
                        let qk: f64 = (0..dim).map(|d| q[hh * dim + d] as f64 * c.ikey[t][d] as f64).sum();
                        w[hh] as f64 / (heads as f64).sqrt() * (qk / (dim as f64).sqrt()).max(0.0)
                    })
                    .sum()
            })
            .collect();
        let mut best: Vec<usize> = (0..9).collect();
        best.sort_by(|&a, &b| score[b].total_cmp(&score[a]));
        assert!(score[best[4]] - score[best[5]] > 1e-4, "the fifth and sixth tokens are too near to tell: {score:?}");
        let mut want = best[..5].to_vec();
        want.sort_unstable();
        assert_eq!(got, want, "{score:?}");
    }
}
