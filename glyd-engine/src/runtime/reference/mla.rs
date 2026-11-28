//! Multi-head latent attention (`kinds::Mla`), in the expanded form: each cached latent through `kv_b` gives a head's key (nope part) and
//! value; the rope part of the key is one per token, shared by the heads. The absorbed form a backend runs is the same function. With an
//! indexer a query attends only to the tokens it picks (GLM-5.3-Flash: pooled keys; `picks`).
//!
//! Mistral Large's forms (`Mla::pos_scale`, `Mla::kv_from`, `Mla::rope_learned`, `Indexer::q_in`, `Indexer::pos_scale`) are written from transformers'
//! `MistralLarge4TextAttention`, `MistralLarge4TextIndexer`, `MistralLarge4TextSoftmaxScaling` and `MistralLarge4TextRotaryEmbedding` and from vLLM's and
//! llama.cpp's readings of the same release; they are held by this file's tests by hand, not yet by a fixture recorded from transformers.
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
            // (the indexer's own pairing of its channels, where it is not the attention's: the same angles)
            match ix.rope_style.filter(|st| *st != r.style) {
                Some(style) => rope::apply(&Rope { style, ..(*r).clone() }, inv, *af, x, pos),
                None => rope::apply(r, inv, *af, x, pos),
            }
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
/// The picks with how near the choice was to another and the pools' scores (`Picked`). `qr` is what the indexer's query product reads: the query's
/// low rank, or with `Indexer::q_in` the block's input through that norm. `turn`: what a query head (its number, its values) goes through at the
/// query's position before the scores: its scale by position where the indexer has one, then the indexer's rotary (`index_turn`; the keys in the
/// cache were turned at theirs).
pub fn picks(
    ix: &Indexer,
    p: &Params,
    c: &LatentCache,
    seen: &[usize],
    qr: Option<&[f32]>,
    h: &[f32],
    turn: &dyn Fn(usize, &mut [f32]) -> Result<()>,
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
    for (hh, head) in q.chunks_mut(dim).enumerate() {
        turn(hh, head)?;
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

/// A position's factor of each head where queries are scaled by position (`Mla::pos_scale`, `Indexer::pos_scale`): s[h] ln(pos + 1) + b[h], in f32
/// (transformers' `MistralLarge4TextSoftmaxScaling`: the logarithm of the position plus one as a float, times the head's scale, plus its bias).
pub fn pos_factors(s: &[f32], b: &[f32], pos: u32) -> Vec<f32> {
    let ln = ((pos + 1) as f32).lln();
    s.iter().zip(b).map(|(&s, &b)| s * ln + b).collect()
}

/// The factors of `heads` heads from the roles `s` and `b`, each [heads].
fn factors_of(p: &Params, s: &str, b: &str, heads: usize, pos: u32) -> Result<Vec<f32>> {
    let (sv, bv) = (p.vec(s)?, p.vec(b)?);
    if sv.len() != heads || bv.len() != heads {
        return err(format!("MLA: {s} of {} values and {b} of {} for {heads} heads", sv.len(), bv.len()));
    }
    Ok(pos_factors(sv, bv, pos))
}

pub fn step(m: &Mla, p: &Params, c: &mut LatentCache, h: &[f32], pos: u32) -> Result<Vec<f32>> {
    step_with(m, p, c, None, h, pos)
}

/// `step`, with the latents of the block the layer shares where it does (`Mla::kv_from`): `shared`, that block's cache for the sequence, which holds this
/// token already (the block ran before this one; in a prompt's call it holds the later tokens too, which this query does not see); read, never written.
pub fn step_with(m: &Mla, p: &Params, c: &mut LatentCache, shared: Option<&LatentCache>, h: &[f32], pos: u32) -> Result<Vec<f32>> {
    let (nh, kr, nope, rd, vd) = (m.heads, m.kv_lora, m.nope, m.rope_dim, m.v_dim);
    let qw = nope + rd;
    let (q, qr) = match m.q_lora {
        Some(_) => {
            let qr = norm(&m.q_norm, &p.prod("q_a", h)?, p.opt("q_a_norm"), None);
            (p.prod("q_b", &qr)?, Some(qr))
        }
        None => (p.prod("q", h)?, None),
    };
    if q.len() != nh * qw {
        return err(format!("MLA: {} query values for {nh} x {qw}", q.len()));
    }
    // (the rotary's tables at this position; learned frequencies: each angle's times exp(alpha), in f32 as transformers' rotary embedding has them)
    let rot = match &m.rope {
        Some(r) => {
            let (mut inv, af) = rope::inv_freq_at(r, pos + 1);
            if m.rope_learned {
                let alpha = p.vec("rope_alpha")?;
                if alpha.len() != inv.len() {
                    return err(format!("MLA: rope_alpha of {} values for a rotary of {} angles", alpha.len(), inv.len()));
                }
                inv.iter_mut().zip(alpha).for_each(|(f, &a)| *f *= a.lexp());
            }
            Some((r, (inv, af)))
        }
        None if m.rope_learned => return err("MLA: learned rotary frequencies on a layer with no rotary"),
        None => None,
    };
    match (m.kv_from, shared) {
        (None, _) => {
            let ckr = p.prod("kv_a", h)?;
            if ckr.len() != kr + rd {
                return err(format!("MLA: {} latent values for {kr} + {rd}", ckr.len()));
            }
            let mut lat = norm(&m.kv_norm, &ckr[..kr], p.opt("kv_a_norm"), None);
            let mut kr_part = ckr[kr..].to_vec();
            if let Some((r, (inv, af))) = &rot {
                rope::apply(r, inv, *af, &mut kr_part, pos);
            }
            lat.extend(kr_part);
            c.pos.push(pos);
            c.lat.push(lat);
        }
        // (a layer that shares another's latents makes none: its cache keeps the positions, beside its indexer's keys)
        (Some(_), Some(s)) => {
            c.pos.push(pos);
            c.lat.push(Vec::new());
            // (the lender ran first: it holds this layer's tokens, in order, and in a prompt's call the ones after this one too)
            if !s.pos.starts_with(&c.pos) || s.lat.iter().any(|l| l.len() != kr + rd) {
                return err(format!(
                    "MLA: a layer that shares another's latents at position {pos}: that block holds {} tokens of {:?} values, this one {} of {kr} + {rd}",
                    s.pos.len(),
                    s.lat.last().map(Vec::len),
                    c.pos.len()
                ));
            }
        }
        (Some(j), None) => return err(format!("MLA: a latent attention that shares block {j}'s latents runs with that block (`step_with`)")),
    }
    let lats: &[Vec<f32>] = match (m.kv_from, shared) {
        (Some(_), Some(s)) => &s.lat,
        _ => &[],
    };
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
    // (the gate's product: a value a channel of each head's output, or one a head, `GateFrom::Head`)
    let per_head = m.gate.is_some_and(|g| g.from == crate::runtime::kinds::GateFrom::Head);
    let gate = match (m.gate, &qr) {
        // (of the normed query latent, a value a channel of the heads' outputs: `GateFrom::Latent`)
        (Some(g), Some(qr)) if g.from == crate::runtime::kinds::GateFrom::Latent => Some(p.prod("g", qr)?),
        (Some(g), None) if g.from == crate::runtime::kinds::GateFrom::Latent => {
            return err("MLA: an output gate from the query latent, and this layer's queries are one product (no latent)")
        }
        (Some(_), _) => Some(p.prod("g", h)?),
        (None, _) => None,
    };
    if let Some(gv) = gate.as_ref().filter(|gv| gv.len() != if per_head { nh } else { nh * vd }) {
        return err(format!("MLA: a gate of {} values for {nh} heads{}", gv.len(), if per_head { "" } else { " of their values' width" }));
    }
    let seen: Vec<usize> = (0..c.pos.len()).filter(|&j| c.pos[j] <= pos).collect();
    let (seen, margin) = match &m.indexer {
        // (a shared indexer runs none: the query attends to what the latent attention before it picked for it, transformers' `prev_topk_indices`)
        Some(ix) if ix.shared => match c.lent.pop_front() {
            Some(lent) => (lent, f32::INFINITY),
            None => return err("MLA: a shared indexer attends to the picks of the latent attention before it, and none were lent to it"),
        },
        Some(ix) => {
            // (a window: the query's last tokens are read whatever is picked, and the indexer picks among the ones before them)
            let last = m.window.map_or(0, |w| w.min(seen.len()));
            let (before, recent) = seen.split_at(seen.len() - last);
            // (the indexer's queries: of the block's input through their norm where the kind says so; each head scaled by position, then turned)
            let q_in = match ix.q_in {
                Some(n) => {
                    let bias = if n.kind == NormKind::Layer { Some(p.vec("indexer_q_norm_bias")?) } else { None };
                    Some(norm(&n, h, Some(p.vec("indexer_q_norm")?), bias))
                }
                None => None,
            };
            let fac = if ix.pos_scale { Some(factors_of(p, "indexer_scale_s", "indexer_scale_b", ix.heads, pos)?) } else { None };
            let turn = |hh: usize, x: &mut [f32]| -> Result<()> {
                if let Some(f) = &fac {
                    x.iter_mut().for_each(|v| *v *= f[hh]);
                }
                index_turn(ix, &rot, x, pos)
            };
            let mut got = picks(ix, p, c, before, q_in.as_deref().or(qr.as_deref()), h, &turn)?;
            // (a check's picks for this query in the place of the indexer's own: entries the indexer may pick, each once, in order)
            if let Some(t) = c.take.pop_front().filter(|t| !t.is_empty()) {
                if t.windows(2).any(|w| w[0] >= w[1]) || t.iter().any(|e| before.binary_search(e).is_err()) {
                    return err(format!("MLA: picks to take of {t:?}: entries the query at {pos} may pick, each once, in order"));
                }
                got.picks = t;
            }
            let r = (got.picks.iter().chain(recent).copied().collect(), got.gap);
            if let Some(v) = c.picked.as_mut() {
                v.push(got);
            }
            r
        }
        // (a window and no indexer: the query's last tokens alone, itself the last of them)
        None => (seen[seen.len() - m.window.map_or(seen.len(), |w| w.min(seen.len()))..].to_vec(), f32::INFINITY),
    };
    c.margin = margin;
    if m.indexer.is_some() {
        c.picks = seen.clone();
        c.made.push(seen.clone());
    }
    if m.window.is_some() && m.indexer.as_ref().is_some_and(|ix| ix.shared || ix.block > 1) {
        return err("MLA: a window of the last tokens beside the picks is of a layer whose own indexer has a key a token");
    }
    let sinks = if m.sinks { Some(p.vec("sinks")?) } else { None };
    if sinks.is_some_and(|s| s.len() != nh) {
        return err(format!("MLA: sinks of {} values for {nh} heads", sinks.map_or(0, <[f32]>::len)));
    }
    let fac = if m.pos_scale { Some(factors_of(p, "q_scale_s", "q_scale_b", nh, pos)?) } else { None };
    let lat_of = |j: usize| if m.kv_from.is_some() { &lats[j] } else { &c.lat[j] };
    let mut out = Vec::with_capacity(nh * vd);
    for head in 0..nh {
        let rows = |r: usize| &kv_b[(head * (nope + vd) + r) * kr..][..kr];
        let mut qh = q[head * qw..][..qw].to_vec();
        if let Some((r, (inv, af))) = &rot {
            rope::apply(r, inv, *af, &mut qh[nope..], pos);
        }
        // (the query's scale by position, after its rotary: the whole head, its part without positions and its rotated part)
        if let Some(f) = &fac {
            qh.iter_mut().for_each(|v| *v *= f[head]);
        }
        let mut s: Vec<f32> = seen
            .iter()
            .map(|&j| {
                let l = lat_of(j);
                let k_nope: Vec<f32> = (0..nope).map(|d| dot(rows(d), &l[..kr])).collect();
                (dot(&qh[..nope], &k_nope) + dot(&qh[nope..], &l[kr..])) * m.scale
            })
            .collect();
        // (a sink: one more logit in the softmax, the head's value as it is, which weighs nothing)
        s.extend(sinks.map(|k| k[head]));
        softmax(&mut s);
        let mut o = vec![0f64; vd];
        for (&j, &pj) in seen.iter().zip(&s) {
            for (d, ov) in o.iter_mut().enumerate() {
                *ov += pj as f64 * dot(rows(nope + d), &lat_of(j)[..kr]) as f64;
            }
        }
        out.extend(o.into_iter().map(|v| v as f32));
    }
    // (the heads' outputs mixed with each other where the kind says, before the gate)
    if m.head_mix > 0 {
        super::head_mix(p.vec("mix_u")?, p.vec("mix_v")?, nh, m.head_mix, &mut out)?;
    }
    if let (Some(g), Some(gv)) = (m.gate, &gate) {
        for (head, o) in out.chunks_mut(vd).enumerate() {
            if let Some(n) = g.norm {
                let y = norm(&n, o, None, None);
                o.copy_from_slice(&y);
            }
            if per_head {
                let z = act(g.act, gv[head]);
                o.iter_mut().for_each(|x| *x *= z);
            } else {
                o.iter_mut().zip(&gv[head * vd..][..vd]).for_each(|(x, &z)| *x *= act(g.act, z));
            }
        }
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
        let ix = Indexer {
            heads: 2,
            head_dim: 16,
            top_k,
            block,
            shared: false,
            form: crate::runtime::kinds::IndexForm::LAYER_KEYS,
            rope: 0,
            q_in: None,
            pos_scale: false,
            rope_style: None,
        };
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
            head_mix: 0,
            sinks: false,
            window: None,
            pos_scale: false,
            kv_from: None,
            rope_learned: false,
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
        let Picked { picks: got, gap: margin, .. } = picks(&ix, &p, &c2, &seen, Some(&qr), h, &|_, _| Ok(())).unwrap();
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
        let turn = |_: usize, x: &mut [f32]| {
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
        let got = picks(ix, &p, &c, &(0..9).collect::<Vec<_>>(), Some(&qr), &hs[8], &|_, _| Ok(())).unwrap().picks;
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

    /// A gate of one value a head (`GateFrom::Head`, Ling's "head_wise" `g_proj`) on a latent attention with one query product and an interleaved rotary on
    /// the shared key part: each head's output times sigmoid of the head's row of `g` on the layer's input. It is, to the bit, the gate of a value a
    /// channel whose rows repeat each head's row over the head's values; it is not the layer without a gate; and a head whose row is zero is halved.
    #[test]
    fn a_gate_a_head_is_the_gate_a_channel_with_a_heads_rows_alike() {
        use crate::runtime::kinds::{Act, GateFrom, OutGate, Rope, RopeStyle};
        let (nh, nope, rd, vd, kr) = (2usize, 8usize, 4usize, 6usize, 12usize);
        let gate = |from| Some(OutGate { act: Act::Sigmoid, from, norm: None });
        let m = Mla {
            heads: nh,
            q_lora: None,
            kv_lora: kr,
            nope,
            rope_dim: rd,
            v_dim: vd,
            rope: Some(Rope { style: RopeStyle::Interleaved, ..Rope::plain(6e6, rd) }),
            q_norm: Norm::rms(1e-6),
            kv_norm: Norm::rms(1e-6),
            gate: gate(GateFrom::Head),
            scale: ((nope + rd) as f32).powf(-0.5),
            absorbed: true,
            indexer: None,
            head_mix: 0,
            sinks: false,
            window: None,
            pos_scale: false,
            kv_from: None,
            rope_learned: false,
        };
        let mut p = Params::new();
        for (role, shape) in [("q", [nh * (nope + rd), D]), ("kv_a", [kr + rd, D]), ("kv_b", [nh * (nope + vd), kr]), ("o", [D, nh * vd]), ("g", [nh, D])] {
            p.put(role, &shape, fill(&format!("mla.head.{role}"), shape[0] * shape[1], -0.4, 0.4));
        }
        p.put("kv_a_norm", &[kr], fill("mla.head.kv_a_norm", kr, 0.5, 1.5));
        let g = p.vec("g").unwrap().to_vec();
        let run = |m: &Mla, p: &Params| -> Vec<Vec<f32>> {
            let mut c = LatentCache::default();
            inputs(6).iter().enumerate().map(|(t, h)| step(m, p, &mut c, h, t as u32).unwrap()).collect()
        };
        let by_head = run(&m, &p);
        assert!(by_head.iter().flatten().all(|x| x.is_finite()));
        // (the same gate a channel: head h's row, once a value of the head)
        let mut wide = p.clone();
        let rows: Vec<f32> = (0..nh).flat_map(|h| g[h * D..(h + 1) * D].repeat(vd)).collect();
        wide.put("g", &[nh * vd, D], rows);
        assert_eq!(by_head, run(&Mla { gate: gate(GateFrom::Own), ..m.clone() }, &wide), "a gate a head is a head's rows alike");
        assert!(rel(&by_head[5], &run(&Mla { gate: None, ..m.clone() }, &p)[5]) > 1e-3, "the gate changes the answer");
        // (a gate of the other width is refused, not read short)
        let mut c = LatentCache::default();
        assert!(
            step(&m, &wide, &mut c, &inputs(1)[0], 0).is_err() && step(&Mla { gate: gate(GateFrom::Own), ..m.clone() }, &p, &mut c, &inputs(1)[0], 0).is_err()
        );
        // (a head whose row of `g` is zero has the gate sigmoid(0), a half, exactly: with the other head's columns of `o` zeroed, the layer is the gate-less
        // one at half, to the bit)
        let mut one = p.clone();
        let mut o = p.vec("o").unwrap().to_vec();
        for r in 0..D {
            o[r * nh * vd + vd..(r + 1) * nh * vd].iter_mut().for_each(|x| *x = 0.0);
        }
        one.put("o", &[D, nh * vd], o);
        one.put("g", &[nh, D], (0..nh * D).map(|i| if i < D { 0.0 } else { g[i] }).collect());
        let half: Vec<Vec<f32>> = run(&Mla { gate: None, ..m.clone() }, &one).into_iter().map(|row| row.into_iter().map(|x| 0.5 * x).collect()).collect();
        assert_eq!(run(&m, &one), half, "a head's gate of a half halves its output");
    }

    /// A latent attention of 2 heads (queries in a low rank of 10, no rotary), its weights drawn under `tag`, its output product the identity: a step's
    /// output is then the heads' outputs after the kind's mix and gate. `g` reads the query latent.
    fn latent_gated(tag: &str) -> (Mla, Params) {
        use crate::runtime::kinds::{Act, GateFrom, OutGate};
        let (nh, nope, vd, kr, ql) = (2usize, 8usize, 6usize, 12usize, 10usize);
        let m = Mla {
            heads: nh,
            q_lora: Some(ql),
            kv_lora: kr,
            nope,
            rope_dim: 0,
            v_dim: vd,
            rope: None,
            q_norm: Norm::rms(1e-6),
            kv_norm: Norm::rms(1e-6),
            gate: Some(OutGate { act: Act::Sigmoid, from: GateFrom::Latent, norm: None }),
            scale: (nope as f32).powf(-0.5),
            absorbed: true,
            indexer: None,
            head_mix: 0,
            sinks: false,
            window: None,
            pos_scale: false,
            kv_from: None,
            rope_learned: false,
        };
        let mut p = Params::new();
        for (role, shape) in [("q_a", [ql, D]), ("q_b", [nh * nope, ql]), ("kv_a", [kr, D]), ("kv_b", [nh * (nope + vd), kr]), ("g", [nh * vd, ql])] {
            p.put(role, &shape, fill(&format!("{tag}.{role}"), shape[0] * shape[1], -0.4, 0.4));
        }
        p.put("q_a_norm", &[ql], fill(&format!("{tag}.q_a_norm"), ql, 0.5, 1.5));
        p.put("kv_a_norm", &[kr], fill(&format!("{tag}.kv_a_norm"), kr, 0.5, 1.5));
        let w = nh * vd;
        p.put("o", &[w, w], (0..w * w).map(|i| (i / w == i % w) as u32 as f32).collect());
        (m, p)
    }

    fn steps(m: &Mla, p: &Params, n: usize) -> Vec<Vec<f32>> {
        let mut c = LatentCache::default();
        inputs(n).iter().enumerate().map(|(t, h)| step(m, p, &mut c, h, t as u32).unwrap()).collect()
    }

    /// A gate from the query latent (`GateFrom::Latent`, ERNIE's `gate_proj`): each channel of the heads' outputs times sigmoid of `g`'s product of the
    /// normed query latent, by hand from the gate-less layer's heads. It reads the latent, not the block's input: the same `g` under a gate of the
    /// block's input is refused by its width, and a layer whose queries are one product has no latent to gate by. A `g` of zero halves, to the bit.
    #[test]
    fn a_gate_from_the_query_latent_by_hand() {
        use crate::runtime::kinds::{GateFrom, OutGate};
        use crate::runtime::reference::sigmoid;
        let (m, p) = latent_gated("mla.latent");
        let bare = Mla { gate: None, ..m.clone() };
        let (heads, gated) = (steps(&bare, &p, 6), steps(&m, &p, 6));
        for (t, h) in inputs(6).iter().enumerate() {
            let qr = norm(&m.q_norm, &p.prod("q_a", h).unwrap(), p.opt("q_a_norm"), None);
            let gv = p.prod("g", &qr).unwrap();
            let want: Vec<f32> = heads[t].iter().zip(&gv).map(|(o, &z)| o * sigmoid(z)).collect();
            assert_eq!(gated[t], want, "token {t}");
        }
        assert!(rel(&gated[5], &heads[5]) > 1e-3, "the gate changes the answer");
        let mut c = LatentCache::default();
        let own = Mla { gate: m.gate.map(|g| OutGate { from: GateFrom::Own, ..g }), ..m.clone() };
        assert!(step(&own, &p, &mut c, &inputs(1)[0], 0).unwrap_err().msg.contains("g: shape"));
        let mut one = p.clone();
        one.put("q", &[16, D], fill("mla.latent.q", 16 * D, -0.4, 0.4));
        let mut c = LatentCache::default();
        assert!(step(&Mla { q_lora: None, ..m.clone() }, &one, &mut c, &inputs(1)[0], 0).unwrap_err().msg.contains("no latent"));
        let mut zero = p.clone();
        zero.put("g", &[12, 10], vec![0.0; 120]);
        let half: Vec<Vec<f32>> = heads.iter().map(|r| r.iter().map(|x| 0.5 * x).collect()).collect();
        assert_eq!(steps(&m, &zero, 6), half);
    }

    /// The low-rank mix of the heads' outputs on a latent attention (`Mla::head_mix`, ERNIE's): the heads' outputs of the layer without it, mixed
    /// (`head_mix`: held by hand in its own test), and where the layer gates, mixed first and gated after, which is another answer than the other
    /// order. A V of zero is the layer without the mix, to the bit.
    #[test]
    fn the_heads_outputs_are_mixed_before_the_gate() {
        use crate::runtime::reference::{head_mix, sigmoid};
        let (gated, mut p) = latent_gated("mla.mix");
        let (u, v) = (fill("mla.mix.u", 2 * 3, -0.6, 0.6), fill("mla.mix.v", 2 * 3, -0.6, 0.6));
        p.put("mix_u", &[2, 3], u.clone());
        p.put("mix_v", &[2, 3], v.clone());
        let bare = Mla { gate: None, ..gated.clone() };
        let heads = steps(&bare, &p, 6);
        let mixed = steps(&Mla { head_mix: 3, ..bare.clone() }, &p, 6);
        let both = steps(&Mla { head_mix: 3, ..gated.clone() }, &p, 6);
        let gate_only = steps(&gated, &p, 6);
        for (t, h) in inputs(6).iter().enumerate() {
            let mut want = heads[t].clone();
            head_mix(&u, &v, 2, 3, &mut want).unwrap();
            assert_eq!(mixed[t], want, "token {t}");
            let qr = norm(&gated.q_norm, &p.prod("q_a", h).unwrap(), p.opt("q_a_norm"), None);
            let gv = p.prod("g", &qr).unwrap();
            let after: Vec<f32> = want.iter().zip(&gv).map(|(o, &z)| o * sigmoid(z)).collect();
            assert_eq!(both[t], after, "token {t}: mixed, then gated");
            let mut other = gate_only[t].clone();
            head_mix(&u, &v, 2, 3, &mut other).unwrap();
            assert!(rel(&both[t], &other) > 1e-4, "token {t}: gated, then mixed is another answer");
        }
        let mut zero = p.clone();
        zero.put("mix_v", &[2, 3], vec![0.0; 6]);
        assert_eq!(steps(&Mla { head_mix: 3, ..gated.clone() }, &zero, 6), gate_only);
        let mut c = LatentCache::default();
        assert!(step(&Mla { head_mix: 2, ..gated.clone() }, &p, &mut c, &inputs(1)[0], 0).is_err(), "U and V of another rank");
        assert!(!format!("{gated:?}").contains("head_mix") && format!("{:?}", Mla { head_mix: 3, ..gated }).ends_with("indexer: None, head_mix: 3 }"));
    }

    /// Sinks on a latent attention (`Mla::sinks`, ERNIE's `softmax_offset`): a head's sink is one more logit of its softmax, the value as it is (not
    /// scaled as the scores are), and weighs nothing. The last of four tokens written out a second time, key by key; a sink far below the scores is
    /// the layer without sinks, to the bit; a sink far above them leaves nothing of the output.
    #[test]
    fn a_latent_attentions_sinks_join_the_softmax_and_weigh_nothing() {
        let (gated, mut p) = latent_gated("mla.sink");
        let (nh, nope, vd, kr) = (2usize, 8usize, 6usize, 12usize);
        let bare = Mla { gate: None, ..gated };
        let m = Mla { sinks: true, ..bare.clone() };
        let sinks = vec![0.3f32, -0.7];
        p.put("sinks", &[nh], sinks.clone());
        let (hs, got) = (inputs(4), steps(&m, &p, 4));
        let lats: Vec<Vec<f32>> = hs.iter().map(|h| norm(&m.kv_norm, &p.prod("kv_a", h).unwrap(), p.opt("kv_a_norm"), None)).collect();
        let q = p.prod("q_b", &norm(&m.q_norm, &p.prod("q_a", &hs[3]).unwrap(), p.opt("q_a_norm"), None)).unwrap();
        let kv_b = p.vec("kv_b").unwrap();
        let row = |head: usize, r: usize| &kv_b[(head * (nope + vd) + r) * kr..][..kr];
        for head in 0..nh {
            let mut s: Vec<f32> = lats
                .iter()
                .map(|l| {
                    let key: Vec<f32> = (0..nope).map(|d| dot(row(head, d), l)).collect();
                    dot(&q[head * nope..][..nope], &key) * m.scale
                })
                .collect();
            s.push(sinks[head]);
            softmax(&mut s);
            assert!(s[4] > 0.05, "the sink takes a share: {}", s[4]);
            for d in 0..vd {
                let want = lats.iter().zip(&s).map(|(l, &a)| a as f64 * dot(row(head, nope + d), l) as f64).sum::<f64>() as f32;
                assert_eq!(got[3][head * vd + d], want, "head {head}, value {d}");
            }
        }
        let plain = steps(&bare, &p, 4);
        assert!(rel(&got[3], &plain[3]) > 1e-2);
        let mut far = p.clone();
        far.put("sinks", &[nh], vec![-1e30; nh]);
        assert_eq!(steps(&m, &far, 4), plain, "a sink of no weight");
        far.put("sinks", &[nh], vec![60.0; nh]);
        assert!(steps(&m, &far, 4)[3].iter().zip(&plain[3]).all(|(a, b)| a.abs() < 1e-12 + b.abs() * 1e-12));
        far.put("sinks", &[3], vec![0.0; 3]);
        let mut c = LatentCache::default();
        assert!(step(&m, &far, &mut c, &hs[0], 0).is_err() && step(&m, &plain_params(&p), &mut c, &hs[0], 0).is_err());
        assert!(!format!("{bare:?}").contains("sinks") && format!("{m:?}").ends_with("indexer: None, sinks: true }"));
    }

    /// `p` without its sinks.
    fn plain_params(p: &Params) -> Params {
        let mut q = p.clone();
        q.t.remove("sinks");
        q
    }

    /// A window beside the indexer's picks (`Mla::window`, ERNIE's latent layers): a query attends to its last `window` tokens whatever is picked, and
    /// the indexer keeps its `top_k` among the tokens before them. With an indexer keeping 2 and a window of 3, over nine tokens: each query's entries
    /// are the indexer's picks, all before the window, then the window's; the picks are the best two by the indexer's own scores of the tokens before the
    /// window; the output is, to the bit, the layer without a window made to take those entries; and it is another answer than the layer without the
    /// window. A token at a time is all there is here: the cache is the state. A window with no indexer of the layer's own is refused.
    #[test]
    fn a_window_of_the_last_tokens_is_read_beside_the_picks() {
        let (plain, p) = layer(2, 1);
        let m = Mla { window: Some(3), ..plain.clone() };
        let hs = inputs(9);
        let (mut a, mut b, mut c) = (LatentCache { picked: Some(Vec::new()), ..Default::default() }, LatentCache::default(), LatentCache::default());
        let mut differ = false;
        for (t, h) in hs.iter().enumerate() {
            let got = step(&m, &p, &mut a, h, t as u32).unwrap();
            let (recent, before) = ((t + 1).min(3), (t + 1).saturating_sub(3));
            let entries = a.picks.clone();
            assert_eq!(entries.len(), before.min(2) + recent, "token {t}: {entries:?}");
            assert_eq!(entries[entries.len() - recent..], (t + 1 - recent..=t).collect::<Vec<_>>()[..], "token {t}: the window");
            let rec = a.picked.as_ref().unwrap().last().unwrap().clone();
            assert_eq!(rec.picks[..], entries[..entries.len() - recent], "token {t}: the picks before the window");
            assert!(rec.picks.iter().all(|&e| e < before), "token {t}: {:?}", rec.picks);
            if before > 2 {
                // (the indexer's own scores of the tokens before the window: one a token, the best two kept)
                assert_eq!(rec.score.len(), before, "token {t}");
                let mut ord: Vec<usize> = (0..before).collect();
                ord.sort_by(|&x, &y| rec.score[y].total_cmp(&rec.score[x]).then(x.cmp(&y)));
                ord.truncate(2);
                ord.sort_unstable();
                assert_eq!(rec.picks, ord, "token {t}");
            }
            // (the layer without a window, taking those entries)
            b.take.push_back(entries.clone());
            assert_eq!(step(&plain, &p, &mut b, h, t as u32).unwrap(), got, "token {t}");
            differ |= rel(&step(&plain, &p, &mut c, h, t as u32).unwrap(), &got) > 1e-4;
        }
        assert!(differ, "the window changed no row");
        // (beside the picks of an indexer of the layer's own with a key a token: a pooled or a shared one with a window is refused; with no indexer the
        // window is all the query reads, which `a_window_alone_is_the_last_tokens` holds)
        let mut d = LatentCache::default();
        let pooled = Mla { indexer: m.indexer.clone().map(|ix| Indexer { block: 2, ..ix }), ..m.clone() };
        assert!(step(&pooled, &p, &mut d, &hs[0], 0).unwrap_err().msg.contains("a key a token"));
        assert!(!format!("{plain:?}").contains("window") && format!("{m:?}").ends_with("window: 3 }"));
    }

    /// A window with no indexer (`Mla::window`, Hunyuan's sliding layers): a query at p attends to p - window + 1 ..= p and to nothing before them.
    /// On a layer with a rotary in interleaved pairs and sinks, a window of 5 over twelve tokens: each query's output is, to the bit, the layer without a
    /// window run from an empty cache on those tokens alone at their own positions (the sequence's first tokens while it is shorter than the window); it
    /// is the unwindowed layer's until the window is full and another answer after; and a window of one token reads the query itself.
    #[test]
    fn a_window_alone_is_the_last_tokens() {
        use crate::runtime::kinds::RopeStyle;
        let (m0, mut p) = layer(4, 1);
        p.put("q_b", &[48, 24], fill("mla.q_b.window", 48 * 24, -0.3, 0.3));
        p.put("kv_a", &[24, D], fill("mla.kv_a.window", 24 * D, -0.3, 0.3));
        p.put("sinks", &[2], vec![0.4, -0.2]);
        let r = Rope { style: RopeStyle::Interleaved, ..Rope::plain(1e4, 8) };
        let plain = Mla { rope_dim: 8, rope: Some(r), scale: 24f32.lpowf(-0.5), indexer: None, sinks: true, ..m0 };
        let m = Mla { window: Some(5), ..plain.clone() };
        let hs = inputs(12);
        let (mut a, mut all) = (LatentCache::default(), LatentCache::default());
        for (t, h) in hs.iter().enumerate() {
            let got = step(&m, &p, &mut a, h, t as u32).unwrap();
            let first = (t + 1).saturating_sub(5);
            let mut alone = LatentCache::default();
            let want = (first..=t).map(|j| step(&plain, &p, &mut alone, &hs[j], j as u32).unwrap()).last().unwrap();
            assert_eq!(got, want, "token {t}: tokens {first} to {t} alone");
            let whole = step(&plain, &p, &mut all, h, t as u32).unwrap();
            assert_eq!(got == whole, t < 5, "token {t}: the window is every token until it is full");
        }
        assert_eq!(a.lat.len(), 12, "the cache keeps every token: the window is the query's");
        let mut one = LatentCache::default();
        for (t, h) in hs.iter().enumerate().take(3) {
            let mut alone = LatentCache::default();
            assert_eq!(step(&Mla { window: Some(1), ..plain.clone() }, &p, &mut one, h, t as u32).unwrap(), step(&plain, &p, &mut alone, h, t as u32).unwrap());
        }
        assert!(format!("{m:?}").ends_with("indexer: None, sinks: true, window: 5 }"));
    }

    /// An indexer's rotary that pairs its channels its own way (`Indexer::rope_style`, ERNIE's: halves beside the attention's interleaved pairs): each
    /// token's key in the cache is its LayerNorm-ed key with channel i and channel i + rope / 2 turned by the angle of the attention's i-th
    /// frequency at the token's position, by hand; the attention's own shared key part is turned as before; and the same style as the attention's,
    /// said or not, is one indexer.
    #[test]
    fn an_indexers_rotary_pairs_its_channels_its_own_way() {
        use crate::runtime::kinds::RopeStyle;
        let (m0, mut p) = layer(4, 1);
        let r = Rope { style: RopeStyle::Interleaved, ..Rope::plain(1e4, 8) };
        p.put("q_b", &[48, 24], fill("mla.q_b.turned", 48 * 24, -0.3, 0.3));
        p.put("kv_a", &[24, D], fill("mla.kv_a.turned", 24 * D, -0.3, 0.3));
        let ix = |style| Indexer { rope: 8, rope_style: style, ..m0.indexer.clone().unwrap() };
        let of = |style| Mla { rope_dim: 8, rope: Some(r.clone()), scale: 24f32.lpowf(-0.5), indexer: Some(ix(style)), ..m0.clone() };
        let hs = inputs(12);
        let run = |m: &Mla| {
            let mut c = LatentCache::default();
            let rows: Vec<Vec<f32>> = hs.iter().enumerate().map(|(t, h)| step(m, &p, &mut c, h, t as u32).unwrap()).collect();
            (rows, c)
        };
        let ((halves, a), (pairs, b), (said, _)) = (run(&of(Some(RopeStyle::Half))), run(&of(None)), run(&of(Some(RopeStyle::Interleaved))));
        for (t, h) in hs.iter().enumerate() {
            let k = key_norm(&ix(None), &p, &p.prod("indexer_wk", h).unwrap()).unwrap();
            let (inv, _) = rope::inv_freq_at(&r, t as u32 + 1);
            let mut want = k.clone();
            for i in 0..4 {
                let angle = t as f32 * inv[i];
                let (cos, sin) = ((angle as f64).lcos() as f32, (angle as f64).lsin() as f32);
                (want[i], want[i + 4]) = (k[i] * cos - k[i + 4] * sin, k[i + 4] * cos + k[i] * sin);
            }
            assert_eq!(a.ikey[t], want, "token {t}'s key, turned in halves");
            assert_eq!(a.ikey[t] == b.ikey[t], t == 0, "token {t}: another pairing is another key past position 0");
        }
        assert_eq!(a.lat, b.lat, "the attention's own rotary is as it was");
        assert_eq!(pairs, said);
        assert!(halves.iter().zip(&pairs).skip(4).any(|(x, y)| rel(x, y) > 1e-4), "the pairing changed no pick");
        assert!(!format!("{:?}", ix(None)).contains("rope_style") && format!("{:?}", ix(Some(RopeStyle::Half))).ends_with("rope: 8, rope_style: Half }"));
    }

    /// A latent attention whose queries are one product, with a rotary on the shared key part (Mistral Large's shape, small): `heads` heads of 8 + 4.
    fn direct(rotary: bool) -> (Mla, Params) {
        use crate::runtime::kinds::RopeStyle;
        let (nh, nope, rd, vd, kr) = (2usize, if rotary { 8usize } else { 12 }, if rotary { 4usize } else { 0 }, 6usize, 12usize);
        let m = Mla {
            heads: nh,
            q_lora: None,
            kv_lora: kr,
            nope,
            rope_dim: rd,
            v_dim: vd,
            rope: rotary.then(|| Rope { style: RopeStyle::Interleaved, ..Rope::plain(1e4, rd) }),
            q_norm: Norm::rms(1e-5),
            kv_norm: Norm::rms(1e-5),
            gate: None,
            scale: 12f32.lpowf(-0.5),
            absorbed: true,
            indexer: None,
            pos_scale: false,
            kv_from: None,
            rope_learned: false,
            head_mix: 0,
            sinks: false,
            window: None,
        };
        let mut p = Params::new();
        for (role, shape) in [("q", [nh * 12, D]), ("kv_a", [kr + rd, D]), ("kv_b", [nh * (nope + vd), kr]), ("o", [D, nh * vd])] {
            p.put(role, &shape, fill(&format!("mla.direct.{role}"), shape[0] * shape[1], -0.4, 0.4));
        }
        p.put("kv_a_norm", &[kr], fill("mla.direct.kv_a_norm", kr, 0.5, 1.5));
        (m, p)
    }

    fn run(m: &Mla, p: &Params, n: usize) -> Vec<Vec<f32>> {
        let mut c = LatentCache::default();
        inputs(n).iter().enumerate().map(|(t, h)| step(m, p, &mut c, h, t as u32).unwrap()).collect()
    }

    /// The queries' scale by position (`Mla::pos_scale`, Mistral Large's): head h's query times s_h ln(pos + 1) + b_h after its rotary. The factors by
    /// hand; a scale of 0 and a bias of 1 are the layer without it, to the bit; the first token's factor is the bias alone; and since the scores are
    /// the query's dot products, the row at position p with one s and one b for every head is the row of the layer without the scale whose scores'
    /// factor is `scale` (s ln(p + 1) + b): the softmax sharpens with the logarithm of the position, as the formula says. A scale a head is each head's
    /// own; tensors of another width are refused.
    #[test]
    fn the_queries_scale_with_the_logarithm_of_their_position() {
        assert_eq!(pos_factors(&[0.5, -0.25], &[2.0, 1.0], 0), [2.0, 1.0], "position 0: ln 1 = 0, the bias alone");
        let f = pos_factors(&[0.5, -0.25], &[2.0, 1.0], 6);
        assert_eq!(f, [0.5 * 7f32.lln() + 2.0, -0.25 * 7f32.lln() + 1.0]);
        assert!((f[0] - 2.972_955).abs() < 1e-5 && (f[1] - 0.513_522_5).abs() < 1e-5, "{f:?}");
        for rotary in [true, false] {
            let (plain, mut p) = direct(rotary);
            let scaled = Mla { pos_scale: true, ..plain.clone() };
            let want = run(&plain, &p, 9);
            assert!(step(&scaled, &p, &mut LatentCache::default(), &inputs(1)[0], 0).is_err(), "the scale reads its two tensors");
            (p.put("q_scale_s", &[2], vec![0.0, 0.0]), p.put("q_scale_b", &[2], vec![1.0, 1.0]));
            assert_eq!(run(&scaled, &p, 9), want, "a factor of one");
            // (one s and one b for both heads: at each position the layer whose scores' factor is the scale times the position's)
            let (sv, bv) = (0.75f32, 0.5f32);
            (p.put("q_scale_s", &[2], vec![sv, sv]), p.put("q_scale_b", &[2], vec![bv, bv]));
            let got = run(&scaled, &p, 9);
            for (t, row) in got.iter().enumerate() {
                let fac = sv * ((t + 1) as f32).lln() + bv;
                let at = run(&Mla { scale: plain.scale * fac, ..plain.clone() }, &p, t + 1);
                assert!(rel(row, &at[t]) < 2e-5, "position {t}: the scores' factor {fac}");
                assert!(t < 2 || rel(row, &want[t]) > 1e-4, "position {t}: the scale changed nothing");
            }
            // (position 0 attends to itself alone whatever the factor: the row is the unscaled one)
            assert!(rel(&got[0], &want[0]) < 1e-6);
            // (a head's own scale: the first head as before, the second with a factor of one, is neither layer)
            (p.put("q_scale_s", &[2], vec![sv, 0.0]), p.put("q_scale_b", &[2], vec![bv, 1.0]));
            let mixed = run(&scaled, &p, 9);
            assert!(rel(&mixed[8], &got[8]) > 1e-4 && rel(&mixed[8], &want[8]) > 1e-4);
            p.put("q_scale_s", &[3], vec![0.0; 3]);
            assert!(step(&scaled, &p, &mut LatentCache::default(), &inputs(1)[0], 0).is_err(), "a scale of three values for two heads");
        }
    }

    /// Learned rotary frequencies (`Mla::rope_learned`, Mistral Large's LeRoPE): each angle's inverse frequency times exp(alpha). An alpha of zero
    /// is the rotary as it is, to the bit; an alpha of ln 2 for every angle doubles every frequency, which is the plain layer at twice each
    /// position; another alpha an angle is another answer; and it is refused with no rotary, or with an alpha of another length.
    #[test]
    fn learned_frequencies_scale_the_rotarys_angles() {
        let (plain, mut p) = direct(true);
        let learned = Mla { rope_learned: true, ..plain.clone() };
        let want = run(&plain, &p, 8);
        assert!(step(&learned, &p, &mut LatentCache::default(), &inputs(1)[0], 0).is_err(), "the learned frequencies read their tensor");
        p.put("rope_alpha", &[2], vec![0.0, 0.0]);
        assert_eq!(run(&learned, &p, 8), want, "exp(0) = 1");
        p.put("rope_alpha", &[2], vec![2f32.lln(), 2f32.lln()]);
        let got = run(&learned, &p, 8);
        let mut c = LatentCache::default();
        for (t, h) in inputs(8).iter().enumerate() {
            let twice = step(&plain, &p, &mut c, h, 2 * t as u32).unwrap();
            assert!(rel(&got[t], &twice) < 1e-5, "position {t} at doubled frequencies is position {}", 2 * t);
        }
        assert!(rel(&got[7], &want[7]) > 1e-4, "the frequencies changed nothing");
        p.put("rope_alpha", &[2], vec![0.3, -0.2]);
        let other = run(&learned, &p, 8);
        assert!(rel(&other[7], &want[7]) > 1e-4 && rel(&other[7], &got[7]) > 1e-4);
        p.put("rope_alpha", &[3], vec![0.0; 3]);
        assert!(step(&learned, &p, &mut LatentCache::default(), &inputs(1)[0], 0).is_err(), "three factors for two angles");
        let (nope, pn) = direct(false);
        assert!(step(&Mla { rope_learned: true, ..nope }, &pn, &mut LatentCache::default(), &inputs(1)[0], 0).is_err(), "no rotary to learn");
    }

    /// Latent sharing (`Mla::kv_from`, Mistral Large's `kv_source_layers`): the layer makes no latents and attends to another block's, expanded by its
    /// own `kv_b`. With the lender run on the same inputs it is, to the bit, the layer that makes those latents itself (the lender's `kv_a` and norm,
    /// its own queries, `kv_b` and output product), a prompt's rows and a step's alike; it reads no `kv_a` of its own; and it is refused without the
    /// lender's cache, or with one that holds other tokens than its own.
    #[test]
    fn a_layer_attends_to_the_latents_another_makes() {
        for rotary in [true, false] {
            let (lender, pl) = direct(rotary);
            let sharer = Mla { kv_from: Some(0), ..lender.clone() };
            // (the sharer's own matrices: other queries, another expansion and output product; no kv_a, no norm of it)
            let mut ps = Params::new();
            for (role, shape) in [("q", [24, D]), ("kv_b", [2 * (lender.nope + 6), 12]), ("o", [D, 12])] {
                ps.put(role, &shape, fill(&format!("mla.sharer.{role}"), shape[0] * shape[1], -0.4, 0.4));
            }
            // (the same layer making the lender's latents itself)
            let mut own = ps.clone();
            for role in ["kv_a", "kv_a_norm"] {
                own.t.insert(role.to_string(), pl.t[role].clone());
            }
            let want = run(&lender, &own, 7);
            let (mut cl, mut cs) = (LatentCache::default(), LatentCache::default());
            for (t, h) in inputs(7).iter().enumerate() {
                step(&lender, &pl, &mut cl, h, t as u32).unwrap();
                let got = step_with(&sharer, &ps, &mut cs, Some(&cl), h, t as u32).unwrap();
                assert_eq!(got, want[t], "token {t}");
                assert!(cs.lat[t].is_empty() && cs.pos == cl.pos, "the sharer caches positions, no latents");
            }
            assert!(rel(&want[6], &run(&lender, &pl, 7)[6]) > 1e-4, "the sharer's own matrices are read");
            let h = &inputs(1)[0];
            let none = step(&sharer, &ps, &mut LatentCache::default(), h, 0).unwrap_err().msg;
            assert!(none.contains("shares block 0's latents"), "{none}");
            // (a lender that has not seen the token, and one whose tokens are not this layer's)
            assert!(step_with(&sharer, &ps, &mut LatentCache::default(), Some(&LatentCache::default()), h, 0).is_err());
            assert!(step_with(&sharer, &ps, &mut LatentCache::default(), Some(&cl), h, 3).is_err());
            // (a lender that holds the tokens after this one too, as in a prompt's call: the row is the same)
            let mut prompt = LatentCache::default();
            for (t, h) in inputs(7).iter().enumerate() {
                assert_eq!(step_with(&sharer, &ps, &mut prompt, Some(&cl), h, t as u32).unwrap(), want[t], "token {t}, the lender ahead");
            }
        }
    }

    /// An indexer whose queries are the block's input's (`Indexer::q_in`, Mistral Large's): the input through a LayerNorm with a bias, then the query
    /// product, each head times its scale by position (`Indexer::pos_scale`), then the indexer's rotary; a key a token, LayerNorm-ed and turned at its
    /// position. The picks by hand for the last query; they are not the unscaled queries' on some query, nor the query latent's; and the attention is
    /// over the picks alone.
    #[test]
    fn an_indexer_on_the_blocks_input_scales_its_queries_by_position() {
        use crate::runtime::kinds::{NormWeight, Rounding};
        let (m0, mut p) = direct(true);
        let ln = Norm { kind: NormKind::Layer, weight: NormWeight::Plain, eps: 1e-6, rounding: Rounding::F32 };
        let ix = Indexer {
            heads: 2,
            head_dim: 8,
            top_k: 4,
            block: 1,
            shared: false,
            form: IndexForm::LAYER_KEYS,
            rope: 4,
            q_in: Some(ln),
            pos_scale: true,
            rope_style: None,
        };
        let m = Mla { indexer: Some(ix.clone()), ..m0.clone() };
        for (role, shape) in [("indexer_wq_b", [16, D]), ("indexer_wk", [8, D]), ("indexer_weights", [2, D])] {
            p.put(role, &shape, fill(&format!("mla.in.{role}"), shape[0] * shape[1], -0.5, 0.5));
        }
        for (role, n, lo, hi) in
            [("indexer_q_norm", D, 0.5, 1.5), ("indexer_q_norm_bias", D, -0.1, 0.1), ("indexer_k_norm", 8, 0.5, 1.5), ("indexer_k_norm_bias", 8, -0.1, 0.1)]
        {
            p.put(role, &[n], fill(&format!("mla.in.{role}"), n, lo, hi));
        }
        // (a head whose factor grows with the position and one whose factor falls below zero from the second token on: its scores' sign turns, and
        // with it what the relu keeps)
        (p.put("indexer_scale_s", &[2], vec![1.5, -1.0]), p.put("indexer_scale_b", &[2], vec![0.25, 0.5]));
        let hs = inputs(12);
        let mut c = LatentCache { picked: Some(Vec::new()), ..Default::default() };
        let outs: Vec<Vec<f32>> = hs.iter().enumerate().map(|(t, h)| step(&m, &p, &mut c, h, t as u32).unwrap()).collect();
        let rec = c.picked.take().unwrap();
        assert!(rec.iter().enumerate().all(|(t, r)| r.picks.len() == (t + 1).min(4)));
        // by hand, the last query (position 11)
        let r = m.rope.clone().unwrap();
        let (pos, h) = (11u32, &hs[11]);
        let turn = |x: &mut [f32], at: u32| {
            let (inv, af) = rope::inv_freq_at(&r, at + 1);
            rope::apply(&r, &inv, af, x, at);
        };
        let mut q = p.prod("indexer_wq_b", &norm(&ln, h, p.opt("indexer_q_norm"), p.opt("indexer_q_norm_bias"))).unwrap();
        let (sv, bv) = (p.vec("indexer_scale_s").unwrap(), p.vec("indexer_scale_b").unwrap());
        for (hh, head) in q.chunks_mut(8).enumerate() {
            let f = sv[hh] * 12f32.lln() + bv[hh];
            head.iter_mut().for_each(|v| *v *= f);
            turn(head, pos);
        }
        let w: Vec<f32> = p.prod("indexer_weights", h).unwrap().iter().map(|x| x / 2f32.sqrt()).collect();
        let score: Vec<f32> = (0..12)
            .map(|t| {
                let mut k = key_norm(&ix, &p, &p.prod("indexer_wk", &hs[t]).unwrap()).unwrap();
                turn(&mut k, t as u32);
                assert_eq!(k, c.ikey[t], "the cache's key of token {t}");
                (0..2).map(|hh| w[hh] * (dot(&q[hh * 8..][..8], &k) * 8f32.lpowf(-0.5)).max(0.0)).sum()
            })
            .collect();
        let mut want: Vec<usize> = (0..12).collect();
        want.sort_by(|&x, &y| score[y].total_cmp(&score[x]).then(x.cmp(&y)));
        want.truncate(4);
        want.sort_unstable();
        assert_eq!((&rec[11].picks, &rec[11].score), (&want, &score));
        // (the attention over the picks alone: for a query that picked itself among fewer than all, the layer without an indexer on those tokens at
        // their positions, the query's last)
        let t = (4..12).rev().find(|&t| rec[t].picks.last() == Some(&t)).expect("no query picked itself");
        let mut alone = LatentCache::default();
        let mut out = Vec::new();
        for &j in &rec[t].picks {
            out = step(&m0, &p, &mut alone, &hs[j], j as u32).unwrap();
        }
        assert!(rel(&outs[t], &out) < 1e-6, "query {t}: the attention over the picks");
        // (the scale is read: without it some query picks otherwise; and so is the queries' source)
        let picks_of = |m: &Mla, p: &Params| -> Vec<Vec<usize>> {
            let mut c = LatentCache::default();
            hs.iter().enumerate().map(|(t, h)| step(m, p, &mut c, h, t as u32).map(|_| c.picks.clone())).collect::<Result<_>>().unwrap()
        };
        let with = |f: &dyn Fn(&mut Indexer)| {
            let mut x = ix.clone();
            f(&mut x);
            Mla { indexer: Some(x), ..m0.clone() }
        };
        let own = picks_of(&m, &p);
        assert_ne!(own, picks_of(&with(&|x| x.pos_scale = false), &p), "the queries' scale by position");
        let mut one = p.clone();
        (one.put("indexer_scale_s", &[2], vec![0.0; 2]), one.put("indexer_scale_b", &[2], vec![1.0; 2]));
        assert_eq!(picks_of(&m, &one), picks_of(&with(&|x| x.pos_scale = false), &p), "a factor of one");
        let e = step(
            &with(&|x| x.q_in = None),
            &p,
            &mut LatentCache { pos: (0..9).collect(), lat: vec![vec![0.0; 16]; 9], ikey: vec![vec![0.0; 8]; 9], ..Default::default() },
            h,
            9,
        )
        .unwrap_err()
        .msg;
        assert!(e.contains("reads the query's low rank"), "{e}");
    }

    /// The two families' forms of the latent attention together: an indexer whose queries are the block's input's, scaled by position, turning in its own
    /// pairing (halves under the attention's interleaved rotary), on a layer with sinks that reads a window of its last three tokens beside the four
    /// picks. Each query's entries are picks before the window, then the window; the layer is, to the bit, the layer without a window made to take
    /// those entries; and sinks far below the scores are the layer without sinks.
    #[test]
    fn a_window_and_sinks_beside_an_indexer_on_the_blocks_input() {
        use crate::runtime::kinds::{NormWeight, RopeStyle, Rounding};
        let (m0, mut p) = direct(true);
        let ln = Norm { kind: NormKind::Layer, weight: NormWeight::Plain, eps: 1e-6, rounding: Rounding::F32 };
        let ix = Indexer {
            heads: 2,
            head_dim: 8,
            top_k: 4,
            block: 1,
            shared: false,
            form: IndexForm::LAYER_KEYS,
            rope: 4,
            q_in: Some(ln),
            pos_scale: true,
            rope_style: Some(RopeStyle::Half),
        };
        for (role, shape) in [("indexer_wq_b", [16, D]), ("indexer_wk", [8, D]), ("indexer_weights", [2, D])] {
            p.put(role, &shape, fill(&format!("mla.both.{role}"), shape[0] * shape[1], -0.5, 0.5));
        }
        for (role, n, lo, hi) in
            [("indexer_q_norm", D, 0.5, 1.5), ("indexer_q_norm_bias", D, -0.1, 0.1), ("indexer_k_norm", 8, 0.5, 1.5), ("indexer_k_norm_bias", 8, -0.1, 0.1)]
        {
            p.put(role, &[n], fill(&format!("mla.both.{role}"), n, lo, hi));
        }
        (p.put("indexer_scale_s", &[2], vec![1.5, -1.0]), p.put("indexer_scale_b", &[2], vec![0.25, 0.5]));
        p.put("sinks", &[2], vec![0.3, -0.7]);
        let plain = Mla { indexer: Some(ix), sinks: true, ..m0.clone() };
        let m = Mla { window: Some(3), ..plain.clone() };
        let hs = inputs(12);
        let (mut a, mut b, mut c) = (LatentCache::default(), LatentCache::default(), LatentCache::default());
        let (mut rows, mut differ) = (Vec::new(), false);
        for (t, h) in hs.iter().enumerate() {
            let got = step(&m, &p, &mut a, h, t as u32).unwrap();
            let (recent, before) = ((t + 1).min(3), (t + 1).saturating_sub(3));
            let entries = a.picks.clone();
            assert_eq!(entries.len(), before.min(4) + recent, "token {t}: {entries:?}");
            assert!(entries[..entries.len() - recent].iter().all(|&e| e < before), "token {t}: {entries:?}");
            assert_eq!(entries[entries.len() - recent..], (t + 1 - recent..=t).collect::<Vec<_>>()[..], "token {t}: the window");
            b.take.push_back(entries);
            assert_eq!(step(&plain, &p, &mut b, h, t as u32).unwrap(), got, "token {t}: the layer without a window, taking those entries");
            differ |= rel(&step(&plain, &p, &mut c, h, t as u32).unwrap(), &got) > 1e-4;
            rows.push(got);
        }
        assert!(differ, "the window changed no row");
        let mut far = p.clone();
        far.put("sinks", &[2], vec![-1e30; 2]);
        let run = |m: &Mla, p: &Params| -> Vec<Vec<f32>> {
            let mut c = LatentCache::default();
            hs.iter().enumerate().map(|(t, h)| step(m, p, &mut c, h, t as u32).unwrap()).collect()
        };
        assert_eq!(run(&m, &far), run(&Mla { sinks: false, ..m.clone() }, &p), "sinks far below the scores are no sinks");
        assert_ne!(run(&m, &far), rows);
    }
}
