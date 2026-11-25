//! DeepSeek-V4's attention (`kinds::Comp`), as transformers' `DeepseekV4Attention` computes it: one K = V head over a sliding window and
//! the compressed entries of closed windows (HCA, or CSA with overlapping windows and an indexer's choice), learned sinks, the output
//! rotated back at the query's position and projected through grouped low-rank products.

use super::{dot, norm::norm, rope, softmax, Params};
use crate::ffi::{err, Result};
use crate::runtime::kinds::{Comp, CompShare, IndexForm, Indexer, Norm, Rope};
use std::collections::BTreeMap;

/// One compressor's tokens (each its projection and its gate plus the position bias) and the entries of its closed windows, normed and
/// rotated, and as normed before the rotation (`raw`: V4.1's indexer keys are made of these).
#[derive(Clone, Default)]
pub struct Pool {
    rows: Vec<(Vec<f32>, Vec<f32>)>,
    pub entries: Vec<Vec<f32>>,
    pub raw: Vec<Vec<f32>>,
}

/// A sequence's state in one layer: the window's keys by position, the compressor's and the indexer's pools; and V4.1's shared parts (`kinds::CompShare`):
/// the indexer's keys of the entries, each query position's picks and published candidate blocks, which a layer that reads another's has from it.
#[derive(Default)]
pub struct CompState {
    pub win: Vec<(u32, Vec<f32>)>,
    pub comp: Pool,
    pub idx: Pool,
    /// The last query's indexer choice: how near it was to another (`ffn::margin`).
    pub margin: f32,
    pub ikeys: Vec<Vec<f32>>,
    pub picks: BTreeMap<u32, Vec<usize>>,
    pub cands: BTreeMap<u32, (usize, Vec<usize>)>,
}

/// `x` rotated by `pos` (or back by it) on its trailing `r.dim` values.
fn turn(r: &Rope, x: &mut [f32], pos: u32, back: bool) {
    let (inv, af) = rope::inv_freq(r);
    let n = x.len();
    let t = &mut x[n - r.dim..];
    if back {
        let neg: Vec<f32> = inv.iter().map(|f| -f).collect();
        rope::apply(r, &neg, af, t, pos);
    } else {
        rope::apply(r, &inv, af, t, pos);
    }
}

impl Pool {
    /// Token `pos`'s input x into the pool `pre` (`comp` or `idx`) of width `d`: where it closes a window, that window's entry.
    #[allow(clippy::too_many_arguments)]
    fn step(&mut self, c: &Comp, p: &Params, pre: &str, d: usize, x: &[f32], pos: u32) -> Result<()> {
        let (r, ov) = (c.ratio, c.overlap);
        let w = (1 + ov as usize) * d;
        let kv = p.prod(&format!("{pre}_kv"), x)?;
        // (a window of one token has no gate: its softmax is 1; V4.1's windows no position bias)
        let mut g = if r == 1 { vec![0.0; w] } else { p.prod(&format!("{pre}_gate"), x)? };
        if c.ape {
            let ape = p.vec(&format!("{pre}_ape"))?;
            g.iter_mut().zip(&ape[(pos as usize % r) * w..][..w]).for_each(|(a, b)| *a += b);
        }
        if kv.len() != w || g.len() != w {
            return err(format!("DeepSeek-V4 {pre}: projections of {} and {} values for {w}", kv.len(), g.len()));
        }
        self.rows.push((kv, g));
        if (pos as usize + 1) % r != 0 {
            return Ok(());
        }
        let i = (pos as usize + 1) / r - 1;
        let mut entry = vec![0f32; d];
        for (ch, e) in entry.iter_mut().enumerate() {
            // (value, gate) of each token the channel pools
            let mut cand: Vec<(f32, f32)> = Vec::new();
            if ov && i > 0 {
                cand.extend(self.rows[(i - 1) * r..i * r].iter().map(|(k, g)| (k[ch], g[ch])));
            }
            let o = if ov { d } else { 0 };
            cand.extend(self.rows[i * r..(i + 1) * r].iter().map(|(k, g)| (k[o + ch], g[o + ch])));
            let mut s: Vec<f32> = cand.iter().map(|c| c.1).collect();
            softmax(&mut s);
            *e = cand.iter().zip(&s).map(|(c, &a)| c.0 * a).sum();
        }
        // (the indexer's entries normed as its gated form's keys are, `idx_norm`; the attention's by `comp_norm`)
        let n = match (&c.indexer, pre) {
            (Some(Indexer { form: IndexForm::Gated { key }, .. }), "idx") => *key,
            _ => Norm::rms(c.norm_eps),
        };
        let mut entry = norm(&n, &entry, Some(p.vec(&format!("{pre}_norm"))?), None);
        self.raw.push(entry.clone());
        turn(&c.rope, &mut entry, (i * r) as u32, false);
        self.entries.push(entry);
        Ok(())
    }
}

/// A V4.1 index source's picks for the query at `pos` (its `n` visible entries, their keys `st.ikeys`): the scores, masked outside the candidate
/// source's blocks where it reads them; where this layer is the candidate source, its blocks published; the best `top_k` kept, in order. Both are
/// kept in the state by position for the layers that read them.
#[allow(clippy::too_many_arguments)]
fn shared_picks(c: &Comp, ix: &Indexer, sh: &CompShare, p: &Params, st: &mut CompState, qr: &[f32], x: &[f32], pos: u32, n: usize) -> Result<Vec<usize>> {
    if st.ikeys.len() < n {
        return err(format!("CSA2: {} indexer keys for {n} entries", st.ikeys.len()));
    }
    // (before the first window closes there is nothing to pick, or to publish)
    if n == 0 {
        st.picks.insert(pos, Vec::new());
        return Ok(Vec::new());
    }
    let mut qi = p.prod("idx_q_b", qr)?;
    for h in qi.chunks_mut(ix.head_dim) {
        turn(&c.rope, h, pos, false);
    }
    let wts: Vec<f32> = p.prod("idx_weights", x)?.iter().map(|w| w * (ix.heads as f32).powf(-0.5)).collect();
    let sc = (ix.head_dim as f32).powf(-0.5);
    let score: Vec<f32> =
        st.ikeys[..n].iter().map(|k| (0..ix.heads).map(|h| dot(&qi[h * ix.head_dim..][..ix.head_dim], k).max(0.0) * sc * wts[h]).sum()).collect();
    // (the candidate source: each block's best score, the newest visible entry's block first, then the best others; published with the block's size)
    if let Some((b, top)) = sh.publish {
        let nb = n.div_ceil(b);
        let best: Vec<f32> = (0..nb).map(|k| score[k * b..((k + 1) * b).min(n)].iter().copied().fold(f32::NEG_INFINITY, f32::max)).collect();
        let newest = (n - 1) / b;
        let mut ord: Vec<usize> = (0..nb).filter(|&k| k != newest).collect();
        ord.sort_by(|&a, &z| best[z].total_cmp(&best[a]).then(a.cmp(&z)));
        let mut blocks: Vec<usize> = std::iter::once(newest).chain(ord).take(top).collect();
        blocks.sort();
        st.cands.insert(pos, (b, blocks));
    }
    // (an index source after it: the entries outside the published blocks are not candidates)
    let inside: Box<dyn Fn(usize) -> bool> = match (sh.candidates, sh.publish) {
        (Some(src), None) => {
            let (b, blocks) =
                st.cands.get(&pos).cloned().ok_or_else(|| crate::ffi::Error(format!("CSA2: no candidate blocks of position {pos} from block {src}")))?;
            Box::new(move |j| blocks.binary_search(&(j / b)).is_ok())
        }
        _ => Box::new(|_| true),
    };
    let mut ord: Vec<usize> = (0..n).filter(|&j| inside(j)).collect();
    ord.sort_by(|&a, &b| score[b].total_cmp(&score[a]).then(a.cmp(&b)));
    ord.truncate(ix.top_k);
    st.margin = super::ffn::margin(&score, ix.top_k);
    ord.sort();
    st.picks.insert(pos, ord.clone());
    Ok(ord)
}

/// The q latent of `x` (the block's normed input) and its query heads at `pos`, each normed (where the kind says) and rotated.
pub fn queries(c: &Comp, p: &Params, x: &[f32], pos: u32) -> Result<(Vec<f32>, Vec<f32>)> {
    let (nh, hd) = (c.heads, c.head_dim);
    let qr = norm(&Norm::rms(c.norm_eps), &p.prod("q_a", x)?, Some(p.vec("q_a_norm")?), None);
    let mut q = p.prod("q_b", &qr)?;
    if q.len() != nh * hd || hd % c.o_groups != 0 && (nh * hd) % c.o_groups != 0 {
        return err(format!("DeepSeek-V4 attention: {} query values for {nh} heads of {hd}", q.len()));
    }
    for h in q.chunks_mut(hd) {
        if c.q_head_norm {
            let y = norm(&Norm::rms_bare(c.norm_eps), h, None, None);
            h.copy_from_slice(&y);
        }
        turn(&c.rope, h, pos, false);
    }
    Ok((qr, q))
}

/// The key (= value) of `x` at `pos`: its `kv` product normed and rotated.
pub fn key(c: &Comp, p: &Params, x: &[f32], pos: u32) -> Result<Vec<f32>> {
    let mut kv = norm(&Norm::rms(c.norm_eps), &p.prod("kv", x)?, Some(p.vec("kv_norm")?), None);
    turn(&c.rope, &mut kv, pos, false);
    Ok(kv)
}

/// The queries `q` at `pos` over `keys` (each its value too) with the sinks: each head's output rotated back at `pos`, then the grouped low-rank
/// products.
pub fn attend(c: &Comp, p: &Params, q: &[f32], keys: &[&[f32]], pos: u32) -> Result<Vec<f32>> {
    let (nh, hd) = (c.heads, c.head_dim);
    let sinks = p.vec("sinks")?;
    let mut out = Vec::with_capacity(nh * hd);
    for h in 0..nh {
        let qh = &q[h * hd..][..hd];
        let mut s: Vec<f32> = keys.iter().map(|k| dot(qh, k) * (hd as f32).powf(-0.5)).collect();
        s.push(sinks[h]);
        softmax(&mut s);
        let mut o = vec![0f32; hd];
        for (k, &a) in keys.iter().zip(&s) {
            o.iter_mut().zip(k.iter()).for_each(|(v, &kv)| *v += a * kv);
        }
        turn(&c.rope, &mut o, pos, true);
        out.extend(o);
    }
    // the grouped low-rank output: group g's heads through its rows of o_a
    let (oa, gs) = (p.vec("o_a")?, nh * hd / c.o_groups);
    if oa.len() != c.o_groups * c.o_lora * gs {
        return err(format!("DeepSeek-V4 attention: o_a of {} values for {} groups of {} to {}", oa.len(), c.o_groups, gs, c.o_lora));
    }
    let mut y = Vec::with_capacity(c.o_groups * c.o_lora);
    for g in 0..c.o_groups {
        let xg = &out[g * gs..][..gs];
        y.extend((0..c.o_lora).map(|r| dot(&oa[(g * c.o_lora + r) * gs..][..gs], xg)));
    }
    p.prod("o_b", &y)
}

/// One token's attention output for its input `x` (the block's normed input) at `pos`.
pub fn step(c: &Comp, p: &Params, st: &mut CompState, x: &[f32], pos: u32) -> Result<Vec<f32>> {
    st.margin = f32::INFINITY;
    let (qr, q) = queries(c, p, x, pos)?;
    st.win.push((pos, key(c, p, x, pos)?));
    // the compressed entries this query may read
    let mut chosen = Vec::new();
    if c.ratio > 0 {
        if c.share.is_none_or(|s| s.entries.is_none()) {
            st.comp.step(c, p, "comp", c.head_dim, x, pos)?;
        }
        // (V4.1: the indexer's keys of the new entries, where this source has their product)
        if let (Some(ix), true) = (&c.indexer, c.share.is_some() && p.has("idx_k")) {
            let IndexForm::Gated { key } = ix.form else { return err("a compressed attention's indexer of the gated form") };
            for j in st.ikeys.len()..st.comp.raw.len() {
                let mut k = norm(&key, &p.prod("idx_k", &st.comp.raw[j])?, Some(p.vec("idx_k_norm")?), None);
                turn(&c.rope, &mut k, (j * c.ratio) as u32, false);
                st.ikeys.push(k);
            }
        }
        // (an entry is read once its window has closed: what another layer's compressor made for later positions is not)
        let n = st.comp.entries.len().min((pos as usize + 1) / c.ratio);
        chosen = match (&c.indexer, &c.share) {
            (None, _) => (0..n).collect(),
            (Some(_), Some(sh)) if sh.picks.is_some() => {
                st.picks.get(&pos).cloned().ok_or_else(|| crate::ffi::Error(format!("CSA2: no picks of position {pos} from block {:?}", sh.picks)))?
            }
            (Some(ix), Some(sh)) => shared_picks(c, ix, sh, p, st, &qr, x, pos, n)?,
            (Some(ix), None) => {
                st.idx.step(c, p, "idx", ix.head_dim, x, pos)?;
                let mut qi = p.prod("idx_q_b", &qr)?;
                for h in qi.chunks_mut(ix.head_dim) {
                    turn(&c.rope, h, pos, false);
                }
                let wts: Vec<f32> = p.prod("idx_weights", x)?.iter().map(|w| w * (ix.heads as f32).powf(-0.5)).collect();
                let sc = (ix.head_dim as f32).powf(-0.5);
                let score: Vec<f32> =
                    st.idx.entries.iter().map(|k| (0..ix.heads).map(|h| dot(&qi[h * ix.head_dim..][..ix.head_dim], k).max(0.0) * sc * wts[h]).sum()).collect();
                let mut ord: Vec<usize> = (0..score.len()).collect();
                ord.sort_by(|&a, &b| score[b].total_cmp(&score[a]).then(a.cmp(&b)));
                ord.truncate(ix.top_k);
                st.margin = super::ffn::margin(&score, ix.top_k);
                ord.sort();
                ord
            }
        };
    }
    let mut keys: Vec<&[f32]> = st.win.iter().filter(|(pj, _)| *pj <= pos && pos - pj < c.window as u32).map(|(_, k)| &k[..]).collect();
    keys.extend(chosen.iter().map(|&e| &st.comp.entries[e][..]));
    attend(c, p, &q, &keys, pos)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::kinds::{Block, Body, Embed, Head, Mixer, ModelDesc, Residual, RopeScaling, RopeStyle};
    use crate::runtime::reference::{model::RefModel, testing::fill, RefLayer, SeqState};

    const D: usize = 16;
    const HD: usize = 8;

    fn rope() -> Rope {
        Rope { theta: 10000.0, dim: 4, freqs: 2, freq_dim: 4, style: RopeStyle::Interleaved, scaling: RopeScaling::None, mrope: None }
    }

    /// A V4.1 compressed attention of 2 heads of 8 (K = V), compressing by `ratio`, with an indexer of 2 heads of 8 keeping 2 entries where
    /// `indexer`; its parts' sources `share`.
    fn kind(ratio: usize, indexer: bool, share: CompShare) -> Comp {
        Comp {
            heads: 2,
            head_dim: HD,
            q_lora: 12,
            o_groups: 2,
            o_lora: 4,
            window: 3,
            ratio,
            overlap: false,
            rope: rope(),
            indexer: indexer.then_some(Indexer {
                heads: 2,
                head_dim: HD,
                top_k: 2,
                block: ratio,
                shared: false,
                form: IndexForm::Gated { key: Norm::rms(1e-6) },
            }),
            norm_eps: 1e-6,
            ape: false,
            q_head_norm: false,
            share: Some(share),
        }
    }

    /// The weights of block `k`: the attention's, a compressor's where it makes the entries (with the indexer's key product where `keys`), the
    /// indexer's queries where it picks.
    fn params(k: usize, c: &Comp, keys: bool) -> Params {
        let mut p = Params::new();
        let sh = c.share.unwrap();
        let mut put = |role: &str, shape: [usize; 2], lo: f32, hi: f32| p.put(role, &shape, fill(&format!("csa2.{k}.{role}"), shape[0] * shape[1], lo, hi));
        for (role, shape) in [("q_a", [12, D]), ("q_b", [2 * HD, 12]), ("kv", [HD, D]), ("o_a", [8, HD]), ("o_b", [D, 8])] {
            put(role, shape, -0.4, 0.4);
        }
        put("q_a_norm", [1, 12], 0.5, 1.5);
        put("kv_norm", [1, HD], 0.5, 1.5);
        put("sinks", [1, 2], -0.5, 0.5);
        if sh.entries.is_none() {
            put("comp_kv", [HD, D], -0.4, 0.4);
            put("comp_norm", [1, HD], 0.5, 1.5);
            if c.ratio > 1 {
                put("comp_gate", [HD, D], -0.4, 0.4);
            }
            if keys {
                put("idx_k", [HD, HD], -0.5, 0.5);
                put("idx_k_norm", [1, HD], 0.5, 1.5);
            }
        }
        if c.indexer.is_some() && sh.picks.is_none() {
            put("idx_q_b", [2 * HD, 12], -0.5, 0.5);
            put("idx_weights", [2, D], -0.5, 0.5);
        }
        p
    }

    /// A ratio-1 layer's entries are each token's normed compressor product rotated at its position (no gate, no bias); a ratio-2 layer's,
    /// each complete pair's per-channel softmax over its two gates weighing their products, normed, rotated at the pair's first position;
    /// an open pair makes none.
    #[test]
    fn csa2s_entries_by_hand() {
        let own = CompShare { entries: None, picks: None, candidates: None, publish: None };
        let xs: Vec<Vec<f32>> = (0..5).map(|t| fill(&format!("csa2.x{t}"), D, -1.0, 1.0)).collect();
        for ratio in [1, 2] {
            let c = kind(ratio, false, own);
            let p = params(9, &c, false);
            let mut st = CompState::default();
            for (t, x) in xs.iter().enumerate() {
                step(&c, &p, &mut st, x, t as u32).unwrap();
            }
            assert_eq!(st.comp.entries.len(), 5 / ratio, "ratio {ratio}");
            for (j, e) in st.comp.entries.iter().enumerate() {
                let toks = &xs[j * ratio..(j + 1) * ratio];
                let kv: Vec<Vec<f32>> = toks.iter().map(|x| p.prod("comp_kv", x).unwrap()).collect();
                let mut want: Vec<f32> = if ratio == 1 {
                    kv[0].clone()
                } else {
                    let g: Vec<Vec<f32>> = toks.iter().map(|x| p.prod("comp_gate", x).unwrap()).collect();
                    (0..HD)
                        .map(|ch| {
                            let mut s = vec![g[0][ch], g[1][ch]];
                            softmax(&mut s);
                            s[0] * kv[0][ch] + s[1] * kv[1][ch]
                        })
                        .collect()
                };
                want = norm(&Norm::rms(1e-6), &want, p.opt("comp_norm"), None);
                turn(&rope(), &mut want, (j * ratio) as u32, false);
                assert_eq!(e, &want, "ratio {ratio}, entry {j}");
            }
        }
    }

    fn block(c: Comp) -> Block {
        Block { norm_in: Norm::rms(1e-6), body: Body::Mixer(Mixer::Compressed(c)), norm_out: None, residual: Residual::Add }
    }

    /// Sharing across layers (V4.1): block 0 makes the entries, their indexer keys, the picks and the candidate blocks (blocks of 2 entries, 1
    /// kept); block 1 reads its entries and picks; block 2 reads its entries and keys and picks for itself inside block 0's candidate blocks. Each
    /// query's picks are visible entries; block 1's are block 0's position by position; block 2's lie in the published blocks, which hold the
    /// newest visible entry's block; and a block reading the picks alone is refused.
    #[test]
    fn csa2s_parts_are_shared_across_layers() {
        let shares = [
            CompShare { entries: None, picks: None, candidates: None, publish: Some((2, 1)) },
            CompShare { entries: Some(0), picks: Some(0), candidates: None, publish: None },
            CompShare { entries: Some(0), picks: None, candidates: Some(0), publish: None },
        ];
        let layers: Vec<RefLayer> = shares
            .iter()
            .enumerate()
            .map(|(k, &sh)| {
                let c = kind(2, true, sh);
                RefLayer::new(block(c.clone()), D, params(k, &c, k == 0))
            })
            .collect();
        let desc = ModelDesc {
            hidden: D,
            vocab: 32,
            embed: Embed { scale: 1.0, norm: None, streams: 1, per_layer: None },
            blocks: layers.iter().map(|l| l.block.clone()).collect(),
            final_norm: Norm::rms(1e-6),
            head: Head { tied: false, softcap: None, scale: 1.0, collapse: crate::runtime::kinds::Collapse::None },
            mtp: Vec::new(),
            vision: None,
            rope: None,
        };
        let mut m = RefModel {
            desc,
            embed: fill("csa2.embed", 32 * D, -1.0, 1.0),
            final_norm: fill("csa2.fn", D, 0.5, 1.5),
            head: fill("csa2.head", 32 * D, -0.5, 0.5),
            layers,
            p: Params::new(),
        };
        let tokens: Vec<u32> = (0..14).map(|t| (t * 7 + 3) % 32).collect();
        let logits = m.forward(&tokens, 0).unwrap();
        assert!(logits.iter().flatten().all(|v| v.is_finite()));
        let st = |b: usize| match m.layers[b].seqs.get(&0) {
            Some(SeqState::Comp(c)) => c,
            _ => panic!("block {b}"),
        };
        for pos in 0..14u32 {
            let visible = (pos as usize + 1) / 2;
            let (p0, p2) = (&st(0).picks[&pos], &st(2).picks[&pos]);
            assert!(p0.iter().chain(p2).all(|&j| j < visible) && p0.len() == visible.min(2), "{pos}: {p0:?} {p2:?}");
            assert_eq!(&st(1).picks[&pos], p0, "{pos}");
            if visible > 0 {
                let (b, blocks) = &st(0).cands[&pos];
                assert!(blocks.contains(&((visible - 1) / b)) && blocks.len() == 1, "{pos}: {blocks:?}");
                assert!(p2.iter().all(|j| blocks.contains(&(j / b))), "{pos}: {p2:?} outside {blocks:?}");
            }
        }
        // (block 0's own entries, read by blocks 1 and 2 as theirs)
        assert_eq!(st(1).comp.entries, st(0).comp.entries);
        let mut alone = RefLayer::new(block(kind(2, true, shares[1])), D, params(1, &kind(2, true, shares[1]), false));
        assert!(alone.body(0, 3, 0, &fill("csa2.h", D, -1.0, 1.0)).is_err(), "picks of another block, which a layer alone has not");
    }
}
