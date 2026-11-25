//! MHA, GQA and MQA attention (`kinds::Attn`): full or sliding window, optional q/k/v norms, rope (full, partial, proportional or none),
//! learned sinks, a score soft-cap, an output gate, and the rounding points of `kinds::AttnRounding`. Query head h reads KV head
//! h / (heads / kv_heads). One token at a time against the sequence's cache; a prompt is its tokens in order. With Condor's indexer
//! (`IndexForm::Mean`) a query attends only to the tokens it picks (`mean_picks`).

use super::{act, bf16, dot, norm::norm, rope, softmax, Params};
use crate::ffi::{err, Result};
use crate::runtime::kinds::{Attn, AttnRounding, GateFrom, IndexForm, Indexer, Norm, Rope};

/// A sequence's keys and values in one layer, each after its norm and rope, with their positions; with an indexer, each token's raw
/// indexer key, and how near the last query's picks were to another (`ffn::margin`).
#[derive(Default)]
pub struct KvCache {
    pub pos: Vec<u32>,
    pub k: Vec<Vec<f32>>,
    pub v: Vec<Vec<f32>>,
    pub ikey: Vec<Vec<f32>>,
    pub margin: f32,
}

/// A rope and its angles at a step (`rope::inv_freq_at`).
type Rot<'a> = Option<(&'a Rope, (Vec<f32>, f32))>;

/// Which of the cache entries `seen` (in order, the sequence's first token first) a query attends to with Condor's indexer: while the complete
/// pools of `block` tokens number no more than the `top_k / block` it keeps, every one; else the pools of the best scores, each pool's key
/// the mean of its tokens' raw keys (rounded where the attention rounds), normed by `knorm` (role `index_k_norm`) and rotated at the
/// pool's first position, a pool's score sum_h relu(q_h . key) / sqrt(head_dim) with `q` the normed and rotated index queries; then the
/// tokens of the incomplete last pool (transformers' `Qwen4ExpTextQSAIndexer`). Equal scores keep the earlier pool. With the picks, how near
/// the choice was to another.
pub fn mean_picks(
    ix: &Indexer,
    knorm: &Norm,
    p: &Params,
    c: &KvCache,
    seen: &[usize],
    q: &[f32],
    rot: &Rot,
    r16: &dyn Fn(f32) -> f32,
    rope16: bool,
) -> Result<(Vec<usize>, f32)> {
    let (b, dim) = (ix.block.max(1), ix.head_dim);
    let (pools, keep) = (seen.len() / b, ix.top_k / b);
    if pools <= keep {
        return Ok((seen.to_vec(), f32::INFINITY));
    }
    if c.ikey.len() != c.pos.len() || q.len() != ix.heads * dim {
        return err(format!(
            "attention: an indexer of {} query values for {} heads of {dim}, {} keys for {} tokens",
            q.len(),
            ix.heads,
            c.ikey.len(),
            c.pos.len()
        ));
    }
    let wk = p.vec("index_k_norm")?;
    let score: Vec<f32> = (0..pools)
        .map(|j| {
            let toks = &seen[j * b..(j + 1) * b];
            let mean: Vec<f32> = (0..dim).map(|ch| r16((toks.iter().map(|&t| c.ikey[t][ch] as f64).sum::<f64>() / b as f64) as f32)).collect();
            let mut key: Vec<f32> = norm(knorm, &mean, Some(wk), None).into_iter().map(r16).collect();
            if let Some((r, (inv, af))) = rot {
                turn(r, inv, *af, &mut key, c.pos[toks[0]], r16, rope16);
            }
            (0..ix.heads).map(|h| dot(&q[h * dim..][..dim], &key).max(0.0)).sum::<f32>() / (dim as f32).sqrt()
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

/// The rope on `x` at `pos`: in bf16 arithmetic where `rope16`, else in f32 then rounded by `r16`.
fn turn(r: &Rope, inv: &[f32], af: f32, x: &mut [f32], pos: u32, r16: &dyn Fn(f32) -> f32, rope16: bool) {
    if rope16 {
        rope::apply_bf16(r, inv, af, x, pos);
    } else {
        rope::apply(r, inv, af, x, pos);
        x.iter_mut().for_each(|h| *h = r16(*h));
    }
}

/// One token's attention output (before nothing: `o` is applied) for the normed input `h` at `pos`; its key and value join the cache.
pub fn step(a: &Attn, p: &Params, c: &mut KvCache, h: &[f32], pos: u32) -> Result<Vec<f32>> {
    let (nh, nk, hd, vd) = (a.heads, a.kv_heads, a.head_dim, a.v_head_dim);
    if nh % nk != 0 || (a.k_eq_v && vd != hd) {
        return err(format!("attention: {nh} heads over {nk} KV heads, head dims {hd} and {vd}"));
    }
    // (a bf16 engine's activation: rounded where the kind names its rounding points, as it is otherwise; the variants' other points below)
    use AttnRounding::*;
    let r16 = |v: f32| if a.rounding == F32 { v } else { bf16(v) };
    let (rope16, score16, probs16, prescale) =
        (matches!(a.rounding, Eager | Sdpa), a.rounding == Eager, matches!(a.rounding, Flash | Eager), a.rounding == Sdpa);
    let prod = |role: &str| -> Result<Vec<f32>> { Ok(p.prod(role, h)?.into_iter().map(r16).collect()) };
    let raw_q = prod("q")?;
    let in_q = a.gate.is_some_and(|g| g.from == GateFrom::InQuery);
    let (mut q, gate) = if in_q {
        let (mut q, mut g) = (Vec::with_capacity(nh * hd), Vec::with_capacity(nh * hd));
        for ch in raw_q.chunks(2 * hd) {
            q.extend_from_slice(&ch[..hd]);
            g.extend_from_slice(&ch[hd..]);
        }
        (q, Some(g))
    } else {
        let g = match a.gate {
            Some(_) => Some(prod("g")?),
            None => None,
        };
        (raw_q, g)
    };
    let raw_k = prod("k")?;
    let mut v = if a.k_eq_v { raw_k.clone() } else { prod("v")? };
    let mut k = raw_k;
    if q.len() != nh * hd || k.len() != nk * hd || v.len() != nk * vd {
        return err(format!("attention: products of {}, {} and {} values for {nh} x {hd}, {nk} x {hd}, {nk} x {vd}", q.len(), k.len(), v.len()));
    }
    let rot = a.rope.as_ref().map(|r| (r, rope::inv_freq_at(r, pos + 1)));
    let prep = |x: &mut [f32], n: &Option<crate::runtime::kinds::Norm>, role: &str, d: usize, rotate: bool, by: f32| {
        for head in x.chunks_mut(d) {
            if let Some(n) = n {
                let y = norm(n, head, p.opt(role), None);
                head.iter_mut().zip(y).for_each(|(h, y)| *h = r16(y));
            }
            // (SDPA: the queries times the scale before the rope, rounded)
            if by != 1.0 {
                head.iter_mut().for_each(|h| *h = r16(*h * by));
            }
            if let (true, Some((r, (inv, af)))) = (rotate, &rot) {
                // (eager and SDPA: the rope in bf16 arithmetic; else in f32, then rounded once)
                if rope16 {
                    rope::apply_bf16(r, inv, *af, head, pos);
                } else {
                    rope::apply(r, inv, *af, head, pos);
                    head.iter_mut().for_each(|h| *h = r16(*h));
                }
            }
        }
    };
    let scale = if prescale { 1.0 } else { a.scale };
    prep(&mut q, &a.q_norm, "q_norm", hd, true, if prescale { a.scale } else { 1.0 });
    prep(&mut k, &a.k_norm, "k_norm", hd, true, 1.0);
    prep(&mut v, &a.v_norm, "v_norm", vd, false, 1.0);
    // (the indexer's queries, normed and rotated as the attention's, and this token's raw key, from one product)
    let iq = match &a.indexer {
        None => None,
        Some(ix) => {
            let IndexForm::Mean { norm: inorm } = ix.form else {
                return err("attention: an indexer of the gated form (GLM-5.3-Flash's, DeepSeek-V4's) is a latent or compressed attention's; on GQA the reference has Condor's");
            };
            let qk = prod("index_qk")?;
            let dim = ix.head_dim;
            if qk.len() != (ix.heads + 1) * dim {
                return err(format!("attention: an index product of {} values for {} queries and a key of {dim}", qk.len(), ix.heads));
            }
            let mut q = qk[..ix.heads * dim].to_vec();
            for head in q.chunks_mut(dim) {
                let y = norm(&inorm, head, Some(p.vec("index_q_norm")?), None);
                head.iter_mut().zip(y).for_each(|(h, y)| *h = r16(y));
                if let Some((r, (inv, af))) = &rot {
                    turn(r, inv, *af, head, pos, &r16, rope16);
                }
            }
            c.ikey.push(qk[ix.heads * dim..].to_vec());
            Some((ix, inorm, q))
        }
    };
    c.pos.push(pos);
    c.k.push(k);
    c.v.push(v);
    let seen: Vec<usize> = (0..c.pos.len()).filter(|&j| c.pos[j] <= pos && a.window.is_none_or(|w| pos - c.pos[j] < w as u32)).collect();
    let seen = match iq {
        Some((ix, inorm, q)) => {
            let (s, m) = mean_picks(ix, &inorm, p, c, &seen, &q, &rot, &r16, rope16)?;
            c.margin = m;
            s
        }
        None => {
            c.margin = f32::INFINITY;
            seen
        }
    };
    let sinks = if a.sinks { Some(p.vec("sinks")?) } else { None };
    // (eager: the score the bf16 of the product, then of its scale, then of the cap)
    let r_score = |v: f32| if score16 { bf16(v) } else { v };
    let r_prob = |v: f32| if probs16 { bf16(v) } else { v };
    let mut out = Vec::with_capacity(nh * vd);
    for head in 0..nh {
        let kv = head / (nh / nk);
        let qh = &q[head * hd..][..hd];
        let mut s: Vec<f32> = seen
            .iter()
            .map(|&j| {
                let x = r_score(r_score(dot(qh, &c.k[j][kv * hd..][..hd])) * scale);
                match a.softcap {
                    Some(cap) => r_score(cap * (x / cap).tanh()),
                    None => x,
                }
            })
            .collect();
        if let Some(sk) = sinks {
            s.push(sk[head]);
        }
        softmax(&mut s);
        let mut o = vec![0f64; vd];
        for (&j, &pj) in seen.iter().zip(&s) {
            for (ov, &vv) in o.iter_mut().zip(&c.v[j][kv * vd..][..vd]) {
                *ov += r_prob(pj) as f64 * vv as f64;
            }
        }
        let mut o: Vec<f32> = o.into_iter().map(|v| r16(v as f32)).collect();
        if let (Some(g), Some(gv)) = (a.gate, &gate) {
            if let Some(n) = g.norm {
                o = norm(&n, &o, None, None).into_iter().map(r16).collect();
            }
            // (SDPA: the gate's activation rounded too)
            let ga = |z: f32| if prescale { r16(act(g.act, z)) } else { act(g.act, z) };
            o.iter_mut().zip(&gv[head * vd..][..vd]).for_each(|(x, &z)| *x = r16(*x * ga(z)));
        }
        out.extend(o);
    }
    p.prod("o", &out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::kinds::{OutGate, Rope};
    use crate::runtime::reference::bf16;
    use crate::runtime::reference::testing::{fill, rel};

    /// An attention of `nh` heads over `nk` KV heads of `hd` with roles from the generator (q/k norms with weights, a weightless v norm).
    fn layer(nh: usize, nk: usize, hd: usize, hidden: usize) -> (Attn, Params) {
        let a = Attn {
            heads: nh,
            kv_heads: nk,
            head_dim: hd,
            v_head_dim: hd,
            window: None,
            sinks: false,
            rope: Some(Rope::plain(1e4, hd)),
            q_norm: Some(Norm::rms(1e-6)),
            k_norm: Some(Norm::rms(1e-6)),
            v_norm: Some(Norm::rms_bare(1e-6)),
            gate: None,
            softcap: None,
            bias: false,
            scale: (hd as f32).powf(-0.5),
            k_eq_v: false,
            indexer: None,
            rounding: AttnRounding::F32,
        };
        let mut p = Params::new();
        let bfv = |v: Vec<f32>| v.into_iter().map(bf16).collect::<Vec<_>>();
        for (role, rows, cols) in [("q", nh * hd, hidden), ("k", nk * hd, hidden), ("v", nk * hd, hidden), ("o", hidden, nh * hd)] {
            p.put(role, &[rows, cols], bfv(fill(&format!("attn.{role}"), rows * cols, -0.1, 0.1)));
        }
        p.put("q_norm", &[hd], bfv(fill("attn.q_norm", hd, 0.5, 1.5)));
        p.put("k_norm", &[hd], bfv(fill("attn.k_norm", hd, 0.5, 1.5)));
        (a, p)
    }

    fn run(a: &Attn, p: &Params, hidden: usize, tokens: usize) -> Vec<f32> {
        let mut c = KvCache::default();
        let x = fill("attn.input", tokens * hidden, -1.0, 1.0);
        (0..tokens).flat_map(|t| step(a, p, &mut c, &x[t * hidden..][..hidden], t as u32).unwrap()).collect()
    }

    /// The rounding points move the output by what bf16 explains and no more (about one bf16 place of the output's scale, at heads of 128,
    /// 256 and 512), and Eager's rounded scores move it again; F32 is the function the transformers fixtures check.
    #[test]
    fn the_rounding_points_are_bf16s_size() {
        let (hidden, tokens) = (64, 12);
        for (nh, nk, hd) in [(4, 2, 128), (2, 1, 256), (2, 1, 512)] {
            let (a, p) = layer(nh, nk, hd, hidden);
            let f32s = run(&a, &p, hidden, tokens);
            let flash = run(&Attn { rounding: AttnRounding::Flash, ..a.clone() }, &p, hidden, tokens);
            let eager = run(&Attn { rounding: AttnRounding::Eager, ..a.clone() }, &p, hidden, tokens);
            let sdpa = run(&Attn { rounding: AttnRounding::Sdpa, ..a.clone() }, &p, hidden, tokens);
            let rs = rel(&sdpa, &f32s);
            assert!(rs > 0.0 && rs < 1e-2 && sdpa != flash && sdpa != eager, "head {hd}: sdpa {rs:.2e}");
            let (rf, re) = (rel(&flash, &f32s), rel(&eager, &f32s));
            // (measured: 2.8-3.6e-3 and 2.9-3.9e-3)
            assert!(rf > 0.0 && rf < 1e-2, "head {hd}: flash {rf:.2e}");
            assert!(re > 0.0 && re < 1e-2 && eager != flash, "head {hd}: eager {re:.2e}");
        }
    }

    /// Eager rounds each score: on one query against two keys the probabilities are those of the two bf16 scores, by hand.
    #[test]
    fn an_eager_score_is_its_bf16() {
        let (hd, hidden) = (128, 4);
        let mut a = layer(1, 1, hd, hidden).0;
        (a.rope, a.q_norm, a.k_norm, a.v_norm, a.scale) = (None, None, None, None, 1.0);
        a.rounding = AttnRounding::Eager;
        let mut p = Params::new();
        // (q = h0 e, k = h1 e over the head: their product hd h0 h1, made to fall between two bf16 values; v = h2 e, o the identity on dim 0)
        let e = |c: usize| (0..hd).flat_map(|_| (0..hidden).map(move |j| (j == c) as u32 as f32)).collect::<Vec<f32>>();
        p.put("q", &[hd, hidden], e(0));
        p.put("k", &[hd, hidden], e(1));
        p.put("v", &[hd, hidden], e(2));
        p.put("o", &[1, hd], (0..hd).map(|i| (i == 0) as u32 as f32).collect());
        let mut c = KvCache::default();
        let hs = [[0.75f32, 0.4, 1.0, 0.0], [0.75, 0.38, 3.0, 0.0]];
        step(&a, &p, &mut c, &hs[0], 0).unwrap();
        let got = step(&a, &p, &mut c, &hs[1], 1).unwrap()[0];
        // (the query of the second token against both keys, each product of bf16 values, then rounded)
        let exact: Vec<f32> = hs.iter().map(|h| (hd as f64 * 0.75 * bf16(h[1]) as f64) as f32).collect();
        let mut s: Vec<f32> = exact.iter().map(|&x| bf16(x)).collect();
        assert!(exact.iter().zip(&s).all(|(x, s)| x != s), "the scores {exact:?} are bf16 values");
        crate::runtime::reference::softmax(&mut s);
        let want = bf16((bf16(s[0]) as f64 * 1.0 + bf16(s[1]) as f64 * 3.0) as f32);
        assert_eq!(got, want);
        // (and not the f32 scores' output)
        let mut f = exact.clone();
        crate::runtime::reference::softmax(&mut f);
        assert_ne!(want, bf16((bf16(f[0]) as f64 + bf16(f[1]) as f64 * 3.0) as f32));
    }

    /// An attention of 2 query heads over one KV head of 16 (a rotary of 8) with Condor's indexer of 2 heads of 16 keeping 4 tokens in pools of
    /// 2, its roles from the generator.
    fn qsa() -> (Attn, Params) {
        let (mut a, mut p) = layer(2, 1, 16, 32);
        let n = Norm::rms_plus_one(1e-6);
        (a.rope, a.q_norm, a.k_norm, a.v_norm) = (Some(Rope::plain(1e4, 8)), Some(n), Some(n), None);
        a.indexer = Some(Indexer { heads: 2, head_dim: 16, top_k: 4, block: 2, shared: false, form: IndexForm::Mean { norm: n }, rope: 0 });
        p.put("index_qk", &[48, 32], fill("qsa.qk", 48 * 32, -0.5, 0.5));
        p.put("index_q_norm", &[16], fill("qsa.qn", 16, -0.3, 0.3));
        p.put("index_k_norm", &[16], fill("qsa.kn", 16, -0.3, 0.3));
        (a, p)
    }

    /// While a query sees no more complete pools than the indexer keeps (2 pools: 5 tokens, the fifth a tail), it attends to every token: the
    /// attention without the indexer, to the bit; past that it does not.
    #[test]
    fn condors_indexer_keeps_everything_until_its_pools_run_out() {
        let (a, p) = qsa();
        let plain = Attn { indexer: None, ..a.clone() };
        let (x, y) = (run(&a, &p, 32, 9), run(&plain, &p, 32, 9));
        for t in 0..9 {
            let (g, w) = (&x[t * 32..][..32], &y[t * 32..][..32]);
            if t < 5 {
                assert_eq!(g, w, "token {t}");
            } else {
                assert!(rel(g, w) > 1e-4, "token {t}: the indexer chose everything");
            }
        }
    }

    /// The picks by hand for the ninth token: its queries normed and rotated, the four complete pools' keys (the mean of the raw keys, normed,
    /// rotated at the pool's first position), their scores, the best two pools' tokens and the tail token 8.
    #[test]
    fn condors_indexer_picks_by_hand() {
        let (a, p) = qsa();
        let mut c = KvCache::default();
        let xs = fill("attn.input", 9 * 32, -1.0, 1.0);
        for t in 0..9 {
            step(&a, &p, &mut c, &xs[t * 32..][..32], t as u32).unwrap();
        }
        let (n, r) = (Norm::rms_plus_one(1e-6), a.rope.clone().unwrap());
        let (inv, af) = rope::inv_freq_at(&r, 9);
        let qk = p.prod("index_qk", &xs[8 * 32..]).unwrap();
        let q: Vec<f32> = qk[..32]
            .chunks(16)
            .flat_map(|h| {
                let mut h = norm(&n, h, p.opt("index_q_norm"), None);
                rope::apply(&r, &inv, af, &mut h, 8);
                h
            })
            .collect();
        let score: Vec<f32> = (0..4)
            .map(|j| {
                let mean: Vec<f32> = (0..16).map(|ch| (c.ikey[2 * j][ch] + c.ikey[2 * j + 1][ch]) / 2.0).collect();
                let mut k = norm(&n, &mean, p.opt("index_k_norm"), None);
                rope::apply(&r, &inv, af, &mut k, 2 * j as u32);
                (dot(&q[..16], &k).max(0.0) + dot(&q[16..], &k).max(0.0)) / 4.0
            })
            .collect();
        let mut best: Vec<usize> = (0..4).collect();
        best.sort_by(|&x, &y| score[y].total_cmp(&score[x]));
        assert!(score[best[1]] - score[best[2]] > 1e-4, "the second and third pools are too near to tell: {score:?}");
        let mut want: Vec<usize> = best[..2].iter().flat_map(|&j| [2 * j, 2 * j + 1]).collect();
        want.sort_unstable();
        want.push(8);
        let got = mean_picks(a.indexer.as_ref().unwrap(), &n, &p, &c, &(0..9).collect::<Vec<_>>(), &q, &Some((&r, (inv, af))), &|v| v, false).unwrap().0;
        assert_eq!(got, want);
        assert!(c.margin.is_finite() && c.margin > 0.0);
    }

    /// The output gate's form, its norm and the rounding after each: Muse's own gate with a weightless norm, and Qwen3.5's in-query gate.
    #[test]
    fn the_gate_after_the_rounded_output() {
        let (hidden, tokens) = (64, 6);
        let (mut a, mut p) = layer(2, 1, 128, hidden);
        a.rounding = AttnRounding::Flash;
        a.gate = Some(OutGate { act: crate::runtime::kinds::Act::Sigmoid, from: GateFrom::Own, norm: Some(Norm::rms_bare(1e-6)) });
        p.put("g", &[256, hidden], fill("attn.g", 256 * hidden, -0.1, 0.1).into_iter().map(bf16).collect());
        let gated = run(&a, &p, hidden, tokens);
        // (the gate moves the output, and the output is a bf16 product of bf16 values: o's input rounded at every point)
        assert!(rel(&gated, &run(&Attn { gate: None, ..a.clone() }, &p, hidden, tokens)) > 0.1);
        let mut q = a.clone();
        q.gate = Some(OutGate { act: crate::runtime::kinds::Act::Sigmoid, from: GateFrom::InQuery, norm: None });
        let mut pq = p.clone();
        // (q's rows per head: its 128 query rows, then the 128 rows the own gate had)
        let (qw, gw) = (p.vec("q").unwrap().to_vec(), p.vec("g").unwrap().to_vec());
        let rows: Vec<f32> = (0..2).flat_map(|h| [&qw[h * 128 * hidden..][..128 * hidden], &gw[h * 128 * hidden..][..128 * hidden]].concat()).collect();
        pq.put("q", &[512, hidden], rows);
        let in_q = run(&q, &pq, hidden, tokens);
        let own_no_norm = run(&Attn { gate: Some(OutGate { norm: None, ..a.gate.unwrap() }), ..a.clone() }, &p, hidden, tokens);
        assert_eq!(in_q, own_no_norm, "an in-query gate is the own gate of the same rows");
    }
}
