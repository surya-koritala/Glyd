//! DeepSeek-V4's attention (`kinds::Comp`), as transformers' `DeepseekV4Attention` computes it: one K = V head over a sliding window and
//! the compressed entries of closed windows (HCA, or CSA with overlapping windows and an indexer's choice), learned sinks, the output
//! rotated back at the query's position and projected through grouped low-rank products.

use super::{dot, norm::norm, rope, softmax, Params};
use crate::ffi::{err, Result};
use crate::runtime::kinds::{Comp, Norm, Rope};

/// One compressor's tokens (each its projection and its gate plus the position bias) and the entries of its closed windows.
#[derive(Default)]
pub struct Pool {
    rows: Vec<(Vec<f32>, Vec<f32>)>,
    pub entries: Vec<Vec<f32>>,
}

/// A sequence's state in one layer: the window's keys by position, the compressor's and the indexer's pools.
#[derive(Default)]
pub struct CompState {
    pub win: Vec<(u32, Vec<f32>)>,
    pub comp: Pool,
    pub idx: Pool,
    /// The last query's indexer choice: how near it was to another (`ffn::margin`).
    pub margin: f32,
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
        let ape = p.vec(&format!("{pre}_ape"))?;
        let g: Vec<f32> = p.prod(&format!("{pre}_gate"), x)?.iter().zip(&ape[(pos as usize % r) * w..][..w]).map(|(a, b)| a + b).collect();
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
        let mut entry = norm(&Norm::rms(c.norm_eps), &entry, Some(p.vec(&format!("{pre}_norm"))?), None);
        turn(&c.rope, &mut entry, (i * r) as u32, false);
        self.entries.push(entry);
        Ok(())
    }
}

/// One token's attention output for its input `x` (the block's normed input) at `pos`.
pub fn step(c: &Comp, p: &Params, st: &mut CompState, x: &[f32], pos: u32) -> Result<Vec<f32>> {
    st.margin = f32::INFINITY;
    let (nh, hd) = (c.heads, c.head_dim);
    let qr = norm(&Norm::rms(c.norm_eps), &p.prod("q_a", x)?, Some(p.vec("q_a_norm")?), None);
    let mut q = p.prod("q_b", &qr)?;
    if q.len() != nh * hd || hd % c.o_groups != 0 && (nh * hd) % c.o_groups != 0 {
        return err(format!("DeepSeek-V4 attention: {} query values for {nh} heads of {hd}", q.len()));
    }
    for h in q.chunks_mut(hd) {
        let y = norm(&Norm::rms_bare(c.norm_eps), h, None, None);
        h.copy_from_slice(&y);
        turn(&c.rope, h, pos, false);
    }
    let mut kv = norm(&Norm::rms(c.norm_eps), &p.prod("kv", x)?, Some(p.vec("kv_norm")?), None);
    turn(&c.rope, &mut kv, pos, false);
    st.win.push((pos, kv));
    // the compressed entries this query may read
    let mut keys: Vec<&[f32]> = st.win.iter().filter(|(pj, _)| *pj <= pos && pos - pj < c.window as u32).map(|(_, k)| &k[..]).collect();
    if c.ratio > 0 {
        st.comp.step(c, p, "comp", hd, x, pos)?;
        let n = st.comp.entries.len();
        let chosen: Vec<usize> = match &c.indexer {
            None => (0..n).collect(),
            Some(ix) => {
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
        keys.extend(chosen.iter().map(|&e| &st.comp.entries[e][..]));
    }
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
