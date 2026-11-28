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
//!
//! A query's picks can be given (`LatentCache::take`): a check then holds an engine to this reference on the engine's own picks, where a near tie
//! taken the other way would move a row by more than arithmetic does. Each query's record (`Picked`) has the indexer's own picks and the pools'
//! scores, by which two choices' distance is measured (`Picked::margin_between`).

use super::Libm;
use super::{act, dot, norm::norm, rope, softmax, Params};
use crate::ffi::{err, Result};
use crate::runtime::kinds::{IndexForm, Indexer, Mla, NormKind, Rope};

/// A sequence's latents in one layer: per token the normed latent (kv_lora values) then the shared key part (rope_dim, rotated); and where
/// the layer has an indexer of its own, per token its normed key and its pool gate's scores; and the last query's pick `margin`. `picks`:
/// the entries the last query attended to, where the layer has an indexer, and `made` each query's of the model's current call, in order;
/// `lent`: for a layer whose indexer is `shared`, the picks of the latent attention before it for the queries about to run, each query's in
/// turn (the model lends them a call's: `RefModel::share_into`; a check, one query's), taken by those queries. `take`: the
/// picks of the queries to come, each query's in turn, in the place of the indexer's own (a check's: another engine's picks, the entries each
/// query attends to; an empty list: the indexer's own); `picked`: each query's record in turn (`Picked`), where a check reads them.
#[derive(Default)]
pub struct LatentCache {
    pub pos: Vec<u32>,
    pub lat: Vec<Vec<f32>>,
    pub ikey: Vec<Vec<f32>>,
    pub igate: Vec<Vec<f32>>,
    pub margin: f32,
    pub picks: Vec<usize>,
    pub made: Vec<Vec<usize>>,
    pub lent: std::collections::VecDeque<Vec<usize>>,
    pub take: std::collections::VecDeque<Vec<usize>>,
    pub picked: Option<Vec<Picked>>,
}

/// One query's choice by an indexer of its layer's own: the entries it attended to (`picks`: the indexer's own, unless a check gave others), the
/// indexer's own (`own`), how near that choice was to another (`gap`: `ffn::margin` of the pools' scores; infinite where every entry was kept),
/// each complete pool's score (none where every entry was kept), and the tokens a pool (`block`). A sequence's entries are its tokens from the
/// first, so pool j is entries j block .. (j + 1) block.
#[derive(Clone, Debug)]
pub struct Picked {
    pub picks: Vec<usize>,
    pub own: Vec<usize>,
    pub gap: f32,
    pub score: Vec<f32>,
    pub block: usize,
}

impl Picked {
    /// How far apart the indexer's own choice and `other` (entries of the same query) are by the pools' scores: the largest score among the
    /// pools only the own choice holds less the smallest among those only `other` holds, over the scores' largest magnitude (`ffn::margin`'s
    /// scale); the same entries, 0. NaN where `other` is not a choice the indexer could make (pools not whole, another count of them, another
    /// open pool): a difference no tie explains.
    pub fn margin_between(&self, other: &[usize]) -> f32 {
        if other == self.own {
            return 0.0;
        }
        let (b, n) = (self.block.max(1), self.score.len());
        let pools = |v: &[usize]| -> Option<Vec<usize>> {
            let mut count = vec![0usize; n];
            for &e in v.iter().filter(|&&e| e < n * b) {
                count[e / b] += 1;
            }
            count.iter().all(|&c| c == 0 || c == b).then(|| (0..n).filter(|&j| count[j] == b).collect())
        };
        let open = |v: &[usize]| v.iter().copied().filter(|&e| e >= n * b).collect::<Vec<usize>>();
        let (Some(mine), Some(theirs)) = (pools(&self.own), pools(other)) else { return f32::NAN };
        if mine.len() != theirs.len() || open(&self.own) != open(other) {
            return f32::NAN;
        }
        let top = mine.iter().filter(|j| !theirs.contains(j)).map(|&j| self.score[j]).fold(f32::NEG_INFINITY, f32::max);
        let low = theirs.iter().filter(|j| !mine.contains(j)).map(|&j| self.score[j]).fold(f32::INFINITY, f32::min);
        (top - low) / self.score.iter().fold(0f32, |m, x| m.max(x.abs())).max(1e-30)
    }
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
/// The picks with how near the choice was to another and the pools' scores (`Picked`). `turn`: the indexer's rotary of a query head at
/// the query's position (`index_turn`; the keys in the cache were turned at theirs).
pub fn picks(
    ix: &Indexer,
    p: &Params,
    c: &LatentCache,
    seen: &[usize],
    qr: Option<&[f32]>,
    h: &[f32],
    turn: &dyn Fn(&mut [f32]) -> Result<()>,
) -> Result<Picked> {
    let (b, dim) = (ix.block.max(1), ix.head_dim);
    let (pools, keep) = (seen.len() / b, ix.top_k / b);
    if pools <= keep {
        return Ok(Picked { picks: seen.to_vec(), own: seen.to_vec(), gap: f32::INFINITY, score: Vec::new(), block: b });
    }
    if ix.shared || c.ikey.len() != c.lat.len() {
        return err("MLA: a shared indexer reuses the previous layer's picks, which the reference does not hold");
    }
    let qr = qr.ok_or_else(|| crate::ffi::Error("MLA: an indexer reads the query's low rank, and this layer has none".into()))?;
    let mut q = p.prod("indexer_wq_b", qr)?;
    let w = p.prod("indexer_weights", h)?;
    if q.len() != ix.heads * dim || w.len() != ix.heads {
        return err(format!("MLA: an indexer of {} query values and {} weights for {} heads of {dim}", q.len(), w.len(), ix.heads));
    }
    for head in q.chunks_mut(dim) {
        turn(head)?;
    }
    let ape = if b > 1 { p.vec("indexer_pool_ape")? } else { &[] };
    let keys: Vec<Vec<f32>> = (0..pools)
        .map(|j| {
            let toks = &seen[j * b..(j + 1) * b];
            (0..dim)
                .map(|ch| {
                    if b == 1 {
                        return c.ikey[toks[0]][ch];
                    }
                    let mut g: Vec<f32> = toks.iter().enumerate().map(|(i, &t)| c.igate[t][ch] + ape[i * dim + ch]).collect();
                    softmax(&mut g);
                    toks.iter().zip(&g).map(|(&t, &a)| a as f64 * c.ikey[t][ch] as f64).sum::<f64>() as f32
                })
                .collect()
        })
        .collect();
    Ok(choose(ix, &q, &w, &keys, seen))
}

/// The indexer's choice on its inputs: each query head's values `q` [heads dim] (turned where the indexer has a rotary), the head weights' product `w`
/// [heads] (times heads^-0.5 here), and each complete entry's key `keys` (a pool's as pooled, a token's own where an entry is a token), among the
/// entries `seen` (in order; pool j is entries j block .. (j + 1) block, the open pool's after the complete ones). The best `top_k / block` pools of
/// score sum_h w_h relu(q_h . key / sqrt(head_dim)), equal scores the earlier first, in order, then the open pool's tokens (`Picked`: its margin and
/// every pool's score). `picks` runs it on this reference's own inputs; a check runs it on another engine's (`dynamic::PickTrace`).
pub fn choose(ix: &Indexer, q: &[f32], w: &[f32], keys: &[Vec<f32>], seen: &[usize]) -> Picked {
    choose_among(ix, q, w, keys, seen, None)
}

/// `choose`, an entry a pool (`ix.block` 1), among the entries `inside` alone where given (a candidate source's published blocks, `kinds::CompShare`:
/// the rest scored -inf and never picked); the best `top_k` of them, in order.
pub fn choose_among(ix: &Indexer, q: &[f32], w: &[f32], keys: &[Vec<f32>], seen: &[usize], inside: Option<&[usize]>) -> Picked {
    let Some(inside) = inside else { return choose_pools(ix, q, w, keys, seen) };
    let dim = ix.head_dim;
    let w: Vec<f32> = w.iter().map(|x| x * (ix.heads as f32).lpowf(-0.5)).collect();
    let sc = (dim as f32).lpowf(-0.5);
    let score: Vec<f32> = keys
        .iter()
        .enumerate()
        .map(|(j, key)| {
            if inside.binary_search(&j).is_ok() {
                (0..ix.heads).map(|hh| w[hh] * (dot(&q[hh * dim..][..dim], key) * sc).max(0.0)).sum()
            } else {
                f32::NEG_INFINITY
            }
        })
        .collect();
    let mut ord: Vec<usize> = inside.iter().copied().filter(|&j| j < keys.len()).collect();
    ord.sort_by(|&x, &y| score[y].total_cmp(&score[x]).then(x.cmp(&y)));
    ord.truncate(ix.top_k);
    ord.sort_unstable();
    Picked { own: ord.clone(), picks: ord, gap: super::ffn::margin(&score, ix.top_k), score, block: 1 }
}

fn choose_pools(ix: &Indexer, q: &[f32], w: &[f32], keys: &[Vec<f32>], seen: &[usize]) -> Picked {
    let (b, dim) = (ix.block.max(1), ix.head_dim);
    let (pools, keep) = (keys.len(), ix.top_k / b);
    if pools <= keep {
        return Picked { picks: seen.to_vec(), own: seen.to_vec(), gap: f32::INFINITY, score: Vec::new(), block: b };
    }
    let w: Vec<f32> = w.iter().map(|x| x * (ix.heads as f32).lpowf(-0.5)).collect();
    let sc = (dim as f32).lpowf(-0.5);
    let score: Vec<f32> = keys.iter().map(|key| (0..ix.heads).map(|hh| w[hh] * (dot(&q[hh * dim..][..dim], key) * sc).max(0.0)).sum()).collect();
    let mut ord: Vec<usize> = (0..pools).collect();
    ord.sort_by(|&x, &y| score[y].total_cmp(&score[x]).then(x.cmp(&y)));
    ord.truncate(keep);
    ord.sort_unstable();
    let mut out: Vec<usize> = ord.iter().flat_map(|&j| seen[j * b..(j + 1) * b].iter().copied()).collect();
    out.extend_from_slice(&seen[pools * b..]);
    Picked { own: out.clone(), picks: out, gap: super::ffn::margin(&score, keep), score, block: b }
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
        Some(g) if g.from == crate::runtime::kinds::GateFrom::Head => return err("MLA: an output gate of one value a head is plain attention's (`Attn`)"),
        Some(_) => Some(p.prod("g", h)?),
        None => None,
    };
    let seen: Vec<usize> = (0..c.pos.len()).filter(|&j| c.pos[j] <= pos).collect();
    let (seen, margin) = match &m.indexer {
        // (a shared indexer runs none: the query attends to what the latent attention before it picked for it, transformers' `prev_topk_indices`)
        Some(ix) if ix.shared => match c.lent.pop_front() {
            Some(lent) => (lent, f32::INFINITY),
            None => return err("MLA: a shared indexer attends to the picks of the latent attention before it, and none were lent to it"),
        },
        Some(ix) => {
            let mut got = picks(ix, p, c, &seen, qr.as_deref(), h, &|x: &mut [f32]| index_turn(ix, &rot, x, pos))?;
            // (a check's picks for this query in the place of the indexer's own: entries the query sees, each once, in order)
            if let Some(t) = c.take.pop_front().filter(|t| !t.is_empty()) {
                if t.windows(2).any(|w| w[0] >= w[1]) || t.iter().any(|e| seen.binary_search(e).is_err()) {
                    return err(format!("MLA: picks to take of {t:?}: entries the query at {pos} sees, each once, in order"));
                }
                got.picks = t;
            }
            let r = (got.picks.clone(), got.gap);
            if let Some(v) = c.picked.as_mut() {
                v.push(got);
            }
            r
        }
        None => (seen, f32::INFINITY),
    };
    c.margin = margin;
    if m.indexer.is_some() {
        c.picks = seen.clone();
        c.made.push(seen.clone());
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
        let Picked { picks: got, gap: margin, .. } = picks(&ix, &p, &c2, &seen, Some(&qr), h, &|_| Ok(())).unwrap();
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
                        let z: f64 = g.iter().map(|x| x.lexp()).sum();
                        (0..4).map(|i| g[i].lexp() / z * c2.ikey[4 * j + i][ch] as f64).sum()
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

    /// Picks given in the indexer's place (`LatentCache::take`: a check's, another engine's): the indexer's own taken are its own run, bit for bit; another
    /// choice is the attention over those tokens, its record keeping the indexer's own choice, and two choices' distance is by the pools' scores (a pool
    /// swapped for the next best: the choice's own margin); what the indexer could not choose has none; a list that is not entries the query sees, each
    /// once and in order, is refused.
    #[test]
    fn picks_taken_in_the_indexers_place() {
        let (m, p) = layer(8, 4);
        let hs = inputs(23);
        let mut own = LatentCache { picked: Some(Vec::new()), ..Default::default() };
        let outs: Vec<Vec<f32>> = hs.iter().enumerate().map(|(t, h)| step(&m, &p, &mut own, h, t as u32).unwrap()).collect();
        let rec = own.picked.take().unwrap();
        assert_eq!(rec.len(), 23);
        let mut same = LatentCache { take: rec.iter().map(|r| r.own.clone()).collect(), ..Default::default() };
        for (t, h) in hs.iter().enumerate() {
            assert_eq!(step(&m, &p, &mut same, h, t as u32).unwrap(), outs[t], "token {t}");
        }
        // (the last query: its weakest pool kept swapped for the best one left out)
        let last = rec.last().unwrap();
        let (b, n, sc) = (last.block, last.score.len(), &last.score);
        let mut kept: Vec<usize> = (0..n).filter(|&j| last.own.contains(&(j * b))).collect();
        let low = *kept.iter().min_by(|&&x, &&y| sc[x].total_cmp(&sc[y])).unwrap();
        let best = (0..n).filter(|j| !kept.contains(j)).max_by(|&x, &y| sc[x].total_cmp(&sc[y])).unwrap();
        kept.retain(|&j| j != low);
        kept.push(best);
        kept.sort_unstable();
        let mut other: Vec<usize> = kept.iter().flat_map(|&j| j * b..j * b + b).collect();
        other.extend(n * b..23);
        assert_eq!(last.margin_between(&other), (sc[low] - sc[best]) / sc.iter().fold(0f32, |a, x| a.max(x.abs())));
        assert_eq!((last.margin_between(&other), last.margin_between(&last.own)), (last.gap, 0.0));
        assert!(last.margin_between(&other[1..]).is_nan() && last.margin_between(&other[..other.len() - 1]).is_nan(), "a pool not whole, the open pool cut");
        let mut c =
            LatentCache { take: rec[..22].iter().map(|r| r.own.clone()).chain([other.clone()]).collect(), picked: Some(Vec::new()), ..Default::default() };
        let mut got = Vec::new();
        for (t, h) in hs.iter().enumerate() {
            got = step(&m, &p, &mut c, h, t as u32).unwrap();
        }
        let r = c.picked.unwrap().pop().unwrap();
        assert_eq!((&r.picks, &r.own), (&other, &last.own));
        let (plain, mut alone, mut want) = (Mla { indexer: None, ..m.clone() }, LatentCache::default(), Vec::new());
        for &t in &other {
            want = step(&plain, &p, &mut alone, &hs[t], t as u32).unwrap();
        }
        assert!(rel(&got, &want) < 1e-6, "the attention over the picks taken");
        for bad in [vec![0, 23], vec![1, 0], vec![2, 2]] {
            let mut c = LatentCache { take: [bad.clone()].into(), ..Default::default() };
            assert!(step(&m, &p, &mut c, &hs[3], 3).is_err(), "{bad:?}");
        }
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
        let m = Mla { rope_dim: 8, rope: Some(r.clone()), scale: 24f32.lpowf(-0.5), indexer: Some(ix.clone()), ..m0.clone() };
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
        let got = picks(&ix, &p, &a, &(0..12).collect::<Vec<_>>(), Some(&qr), h, &turn).unwrap().picks;
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
            b.lent.push_back(a.picks.clone());
            let y = step(&shared, &ps, &mut b, h, t as u32).unwrap();
            assert_eq!((x, b.picks.clone(), b.lent.is_empty(), b.ikey.len()), (y, a.picks.clone(), true, 0), "token {t}");
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
        let got = picks(ix, &p, &c, &(0..9).collect::<Vec<_>>(), Some(&qr), &hs[8], &|_| Ok(())).unwrap().picks;
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
