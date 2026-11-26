//! The compressed attention layer kind (`runtime::kinds::Comp`: DeepSeek-V4's) as the executor's operations, each a mechanism: the queries (`Op::CompQ`), the window's
//! key-value row (`Op::CompKv`), the compressed keys and values (`Op::Compress`, the attention's and the indexer's), the indexer's queries (`Op::IdxQ`) and pick
//! (`Op::IdxPick`, every indexer's: src/dynamic/mla.rs), the attention over the window and the compressed entries (`Op::CompAttn`), and the output's grouped low rank
//! (`Op::OGroupLora`, `Op::OGroupLoraStack`). The arithmetic is kernels/dsv4.cu's (the release's), its buffers `V4Run`'s. The window and the compressed entries are rings
//! (`Ring::keys`, `Ring::entries`), read and written through the sequence's block tables where they are paged.

use super::mla::{CandAt, PickAt};
use super::{Dyn, IdxKeys, Op, Part, Slot};
use crate::ffi::{err, Error, Result};
use crate::gpu_ops::GpuOps;
use crate::launch;

impl Dyn {
    /// One of the compressed attention's operations of layer `l` on n rows at positions `pos` (its indexer's picks kept where a check traces each row's choices,
    /// `trace_row_picked`).
    pub(super) unsafe fn op_comp(&self, p: &Part, l: usize, op: &Op, n: usize, pos: u64) -> Result<()> {
        let v = self.v4.as_ref().ok_or_else(|| Error("a compressed attention's operation in a model loaded without its buffers".into()))?;
        let (s, nu, eps, hid) = (self.dev.stream, n as u32, p.d.eps, p.d.hidden as i32);
        // (in a pass over several sequences: the first row of the sequence an operation runs on now, and the pass's rows, which the attention's output is laid out by)
        let (row0, rows) = match self.at.get() {
            Some(k) => (self.parts[k].row0, self.parts.iter().map(|q| q.row0 + q.rows).max().unwrap_or(n)),
            None => (0, n),
        };
        let k = &v.k;
        let rope = |r: usize| (self.ropes[r].0.ptr, self.ropes[r].1.ptr);
        match *op {
            Op::CompQ { rope: r, heads, norm } => {
                let (c, sn) = rope(r);
                launch!(k.q_prep, [nu, heads as u32, 1], [128, 1, 1], s; p.slot(Slot::Qo), c, sn, pos, eps, norm as i32)?;
                self.mark("attention: q norm, rotary")
            }
            Op::CompKv { norm, rope: r, ring } => {
                let (c, sn) = rope(r);
                let kv = p.slot(Slot::Kv) + self.rows_before(p.d.width(Slot::Kv));
                launch!(k.kv_prep, [nu, 1, 1], [128, 1, 1], s; kv, p.small[norm].ptr, c, sn, pos, self.cache(ring).0, self.ring_cap[ring] as i32, self.table(ring), 0i64, eps, v.dims.sim as i32)?;
                self.mark("attention: kv norm, rotary, cache")
            }
            Op::Compress { w, ape, norm, d, ratio, overlap, rope: r, cache, state, indexer, raw } => {
                let wd = if overlap { 2 * d } else { d };
                let raw = raw.map_or(0, |x| self.cache(x).0);
                let (c, sn) = rope(r);
                let (h, st0, st1) = (p.slot(Slot::H) + self.rows_before(p.d.hidden), self.state(state), self.state(state + 1));
                let (tab, cache) = (self.table(cache), self.cache(cache).0);
                if n == 1 && hid % 256 == 0 && hid <= 8192 {
                    launch!(k.lin32_1, [((2 * wd) as u32).div_ceil(8), 1, 1], [256, 1, 1], s; h, p.small[w].ptr, p.slot(Slot::Cp), hid, (2 * wd) as i32)?;
                } else {
                    launch!(k.lin32, [((2 * wd) as u32).div_ceil(8), nu.div_ceil(8), 1], [256, 1, 1], s; h, p.small[w].ptr, p.slot(Slot::Cp), n as i32, hid, (2 * wd) as i32)?;
                }
                // (a verify pass: the rows' products kept for `rewind`, which pools the kept ones again from the open windows as they were before the pass)
                if self.verifying.get() && std::ptr::eq(p, &self.big) {
                    let kept = self.verify.as_ref().and_then(|v| v.comp.get(state)).and_then(Option::as_ref);
                    let kept = kept.ok_or_else(|| Error(format!("a compressor's state {state}: no room kept for a verify pass")))?;
                    crate::dev::copy_d2d_async(kept.ptr, p.slot(Slot::Cp), n * 2 * wd * 4, s)?;
                }
                launch!(k.comp_pool, [(d as u32).div_ceil(128), 1, 1], [128, 1, 1], s; p.slot(Slot::Cp), p.small[ape].ptr, st0, st1, v.pooled.ptr, pos, n as i32, ratio as i32, d as i32, overlap as i32)?;
                launch!(k.comp_out, [(n / ratio + 1) as u32, 1, 1], [128, 1, 1], s; v.pooled.ptr, p.small[norm].ptr, c, sn, pos, n as i32, cache, tab, 0i64, ratio as i32, d as i32, indexer as i32, eps, v.dims.sim as i32, raw)?;
                self.mark(if indexer { "indexer: compressor" } else { "attention: compressor" })
            }
            Op::IdxKeys { w, norm, rope: r, ratio, raw, keys } => {
                let (c, sn) = rope(r);
                // (the raw entries and the keys are rings of the entries' group: one block table)
                launch!(k.idx_keys, [(n / ratio + 1) as u32, 1, 1], [128, 1, 1], s; self.cache(raw).0, self.table(keys), 0i64, p.small[w].ptr, p.small[norm].ptr, c, sn, pos, n as i32, self.cache(keys).0, ratio as i32, eps, v.dims.sim as i32)?;
                self.mark("indexer: keys")
            }
            Op::IdxQ { rope: r, heads } => {
                let (c, sn) = rope(r);
                launch!(k.idx_q, [nu, heads as u32, 1], [128, 1, 1], s; p.slot(Slot::Iq), c, sn, pos, v.dims.sim as i32)?;
                self.mark("indexer: q")
            }
            Op::IdxPick(ref x) => {
                let IdxKeys::Entries(cache) = x.keys else { return err("a compressed attention's indexer whose keys are not its compressor's entries") };
                let (iq, iw, entries, tab) =
                    (p.slot(Slot::Iq) + self.rows_before(x.heads * 128), p.slot(Slot::Iw) + self.rows_before(x.heads), self.cache(cache).0, self.table(cache));
                for r0 in (0..n).step_by(v.ra) {
                    let at = PickAt {
                        q: iq + (r0 * x.heads * 128 * 2) as u64,
                        w: iw + (r0 * x.heads * 2) as u64,
                        keys: entries,
                        stride: 128,
                        tab,
                        ts: 0,
                        pos: pos + (r0 * 4) as u64,
                        rows: v.ra.min(n - r0),
                        scores: v.isc.ptr,
                        emax: v.emax,
                        sel: v.sel.ptr + ((row0 + r0) * x.top_k * 4) as u64,
                        sk: x.top_k,
                        cnt: v.cnt.ptr + ((row0 + r0) * 4) as u64,
                        cand: CandAt {
                            blocks: v.cand.ptr + ((row0 + r0) * v.ctop * 4) as u64,
                            nblocks: v.ncand.ptr + ((row0 + r0) * 4) as u64,
                            bsc: v.bsc.ptr,
                            bmax: v.bmax,
                            nent: v.nent.ptr,
                        },
                    };
                    self.idx_pick(x, &at)?;
                    // (the scores' room holds this run of rows alone: each run's picks kept before the next)
                    self.trace_row_picked(p, l, x, &at, (n, r0), false)?;
                }
                self.mark("indexer: picks")
            }
            Op::CompAttn { ring, cache, ratio, picked, sink, rope: r, heads, hpg } => {
                let (c, sn) = rope(r);
                let (kc, cap, ktab, window) = (self.cache(ring).0, self.ring_cap[ring] as i32, self.table(ring), self.ring_of(p, ring).window as i32);
                let (cache, ctab) = cache.map_or((0, 0), |x| (self.cache(x).0, self.table(x)));
                let topk = v.dims.topk;
                let scale = (512f32).powf(-0.5);
                // (a block draft head's pass: every row's window to the pass's last row, whose keys the pass wrote first)
                let last = if self.ahead.get() {
                    if n > v.ra || (n - 1) + v.dims.window > v.lmax {
                        return err(format!("a block of {n} rows that see each other, where the attention takes {} rows and {} keys", v.ra, v.lmax));
                    }
                    pos + ((n - 1) * 4) as u64
                } else {
                    0
                };
                for r0 in (0..n).step_by(v.ra) {
                    let rn = v.ra.min(n - r0) as u32;
                    let (pr, sel, cnt) = (
                        pos + (r0 * 4) as u64,
                        if picked { v.sel.ptr + ((row0 + r0) * topk * 4) as u64 } else { 0 },
                        if picked { v.cnt.ptr + ((row0 + r0) * 4) as u64 } else { 0 },
                    );
                    let q = p.slot(Slot::Qo) + self.rows_before(heads * 512) + (r0 * heads * 512 * 2) as u64;
                    launch!(k.attn_scores, [rn, (v.lmax as u32).div_ceil(32), (heads as u32).div_ceil(16)], [256, 1, 1], s; q, kc, cap, ktab, 0i64, cache, ctab, 0i64, sel, cnt, topk as i32, pr, v.sc.ptr, heads as i32, v.lmax as i32, window, ratio as i32, scale, last)?;
                    launch!(k.attn_soft, [rn, (heads as u32).div_ceil(8), 1], [256, 1, 1], s; v.sc.ptr, p.small[sink].ptr, v.den.ptr, pr, cnt, heads as i32, v.lmax as i32, window, ratio as i32, last)?;
                    // (kernels/dsv4.cu V4_AQ: 8 warps a head, a head a block)
                    launch!(k.attn_out, [rn, heads as u32, 1], [256, 1, 1], s; v.sc.ptr, v.den.ptr, kc, cap, ktab, 0i64, cache, ctab, 0i64, sel, cnt, topk as i32, pr, c, sn, p.slot(Slot::Att), rows as i32, heads as i32, hpg as i32, v.lmax as i32, window, ratio as i32, (row0 + r0) as i32, last)?;
                }
                self.mark("attention: attention")
            }
            Op::OGroupLora { w, in_f8, groups, hpg, ol, y, fp8 } => {
                for g in 0..groups {
                    let (xg, yg) = (p.slot(Slot::Att) + (g * n * hpg * 512 * 2) as u64, p.slot(Slot::Mid) + (g * n * ol * 2) as u64);
                    if in_f8 {
                        p.f8[w + g].linear(&self.lib, xg, n, yg, self.ws.ptr, self.ws.bytes, self.done.ptr, s)?;
                    } else {
                        self.lin(&p.mats[w + g], xg, n, yg)?;
                    }
                }
                launch!(k.ogather, [nu, ((groups * ol / 128) as u32).div_ceil(4), 1], [128, 1, 1], s; p.slot(Slot::Mid), p.slot(y), n as i32, groups as i32, ol as i32, (fp8 && v.dims.sim) as i32)?;
                self.mark("attention: o low rank")
            }
            Op::OGroupLoraStack { w, groups, ol, y, fp8 } => {
                // (the pairs (row t, group g) the groups' plan sorts: group g's rows of Att, then of Mid, one after another, as the per-group products read and write them; a prompt's chunk of
                // more than the product's 64 rows in parts, each part's rows of each group copied to ox, its products into oy, and back into each group's rows of Mid)
                let m = &p.f8[w];
                let (att, mid) = (p.slot(Slot::Att), p.slot(Slot::Mid));
                if n <= crate::f8::MOE_TOKENS {
                    self.lib.moe_way(v.gids.ptr, (n * groups) as i64, groups as i64, v.gplan.ptr, s)?;
                    m.moe(&self.lib, groups, att, v.gplan.ptr, n, groups, false, mid, self.ws.ptr, self.ws.bytes, self.done.ptr, s)?;
                } else {
                    for t0 in (0..n).step_by(crate::f8::MOE_TOKENS) {
                        let tn = (n - t0).min(crate::f8::MOE_TOKENS);
                        for g in 0..groups {
                            crate::dev::copy_d2d_async(v.ox.ptr + (g * tn * m.k * 2) as u64, att + ((g * n + t0) * m.k * 2) as u64, tn * m.k * 2, s)?;
                        }
                        self.lib.moe_way(v.gids.ptr, (tn * groups) as i64, groups as i64, v.gplan.ptr, s)?;
                        m.moe(&self.lib, groups, v.ox.ptr, v.gplan.ptr, tn, groups, false, v.oy.ptr, self.ws.ptr, self.ws.bytes, self.done.ptr, s)?;
                        for g in 0..groups {
                            crate::dev::copy_d2d_async(mid + ((g * n + t0) * ol * 2) as u64, v.oy.ptr + (g * tn * ol * 2) as u64, tn * ol * 2, s)?;
                        }
                    }
                }
                launch!(k.ogather, [nu, ((groups * ol / 128) as u32).div_ceil(4), 1], [128, 1, 1], s; p.slot(Slot::Mid), p.slot(y), n as i32, groups as i32, ol as i32, (fp8 && v.dims.sim) as i32)?;
                self.mark("attention: o low rank")
            }
            _ => unreachable!(),
        }
    }

    /// One of the compressed attention's operations that reads or writes a sequence's rings or states (`rows::takes_lead`) on the pass's first `rows` rows of its `n`, each
    /// a decode row of its own sequence (`Dyn::lead`): one launch of each kernel for all of them, each row through its own block tables and states (`rows::RowTabs`), at
    /// positions `pos`; each row's kernels and sums are those of its own sequence's pass on its one token (a graph's padding rows, at position -1, read and write no ring or
    /// state). The compressors' product of one row is `v4_lin32_1`, of several `v4_lin32`: the same sums in the same order.
    pub(super) unsafe fn comp_lead(&self, p: &Part, l: usize, op: &Op, (rows, n): (usize, usize), pos: u64) -> Result<()> {
        let v = self.v4.as_ref().ok_or_else(|| Error("a compressed attention's operation in a model loaded without its buffers".into()))?;
        let (s, ru, eps, hid) = (self.dev.stream, rows as u32, p.d.eps, p.d.hidden as i32);
        let k = &v.k;
        let rope = |r: usize| (self.ropes[r].0.ptr, self.ropes[r].1.ptr);
        match *op {
            Op::CompKv { norm, rope: r, ring } => {
                let (c, sn) = rope(r);
                let (kc, tab, ts) = self.lead_ring(ring);
                launch!(k.kv_prep, [ru, 1, 1], [128, 1, 1], s; p.slot(Slot::Kv), p.small[norm].ptr, c, sn, pos, kc, self.ring_cap[ring] as i32, tab, ts as i64, eps, v.dims.sim as i32)?;
                self.mark("attention: kv norm, rotary, cache")
            }
            Op::Compress { w, ape, norm, d, ratio, overlap, rope: r, cache, state, indexer, raw } => {
                let wd = if overlap { 2 * d } else { d };
                let (c, sn) = rope(r);
                let (h, cp) = (p.slot(Slot::H), p.slot(Slot::Cp));
                if rows == 1 && hid % 256 == 0 && hid <= 8192 {
                    launch!(k.lin32_1, [((2 * wd) as u32).div_ceil(8), 1, 1], [256, 1, 1], s; h, p.small[w].ptr, cp, hid, (2 * wd) as i32)?;
                } else {
                    launch!(k.lin32, [((2 * wd) as u32).div_ceil(8), ru.div_ceil(8), 1], [256, 1, 1], s; h, p.small[w].ptr, cp, rows as i32, hid, (2 * wd) as i32)?;
                }
                // (each row's pooling state, the state table's two entries from `state`; each row's entry, where it completes a window, at its row of `pooled`)
                let (sts, ss) = self.lead_states(state);
                launch!(k.comp_pool_rows, [(d as u32).div_ceil(128), ru, 1], [128, 1, 1], s; cp, p.small[ape].ptr, sts, ss as i32, v.pooled.ptr, pos, ratio as i32, d as i32, overlap as i32)?;
                let (cache, tab, ts) = self.lead_ring(cache);
                let raw = raw.map_or(0, |x| self.lead_ring(x).0);
                launch!(k.comp_out, [1, ru, 1], [128, 1, 1], s; v.pooled.ptr, p.small[norm].ptr, c, sn, pos, 1i32, cache, tab, ts as i64, ratio as i32, d as i32, indexer as i32, eps, v.dims.sim as i32, raw)?;
                self.mark(if indexer { "indexer: compressor" } else { "attention: compressor" })
            }
            Op::IdxKeys { w, norm, rope: r, ratio, raw, keys } => {
                let (c, sn) = rope(r);
                // (the raw entries and the keys are rings of the entries' group: one block table a row)
                let (keys, tab, ts) = self.lead_ring(keys);
                launch!(k.idx_keys, [1, ru, 1], [128, 1, 1], s; self.lead_ring(raw).0, tab, ts as i64, p.small[w].ptr, p.small[norm].ptr, c, sn, pos, 1i32, keys, ratio as i32, eps, v.dims.sim as i32)?;
                self.mark("indexer: keys")
            }
            Op::IdxPick(ref x) => {
                let IdxKeys::Entries(cache) = x.keys else { return err("a compressed attention's indexer whose keys are not its compressor's entries") };
                let (entries, tab, ts) = self.lead_ring(cache);
                for r0 in (0..rows).step_by(v.ra) {
                    let at = PickAt {
                        q: p.slot(Slot::Iq) + (r0 * x.heads * 128 * 2) as u64,
                        w: p.slot(Slot::Iw) + (r0 * x.heads * 2) as u64,
                        keys: entries,
                        stride: 128,
                        tab: super::rows::tab_at(tab, ts, r0),
                        ts,
                        pos: pos + (r0 * 4) as u64,
                        rows: v.ra.min(rows - r0),
                        scores: v.isc.ptr,
                        emax: v.emax,
                        sel: v.sel.ptr + (r0 * x.top_k * 4) as u64,
                        sk: x.top_k,
                        cnt: v.cnt.ptr + (r0 * 4) as u64,
                        cand: CandAt {
                            blocks: v.cand.ptr + (r0 * v.ctop * 4) as u64,
                            nblocks: v.ncand.ptr + (r0 * 4) as u64,
                            bsc: v.bsc.ptr,
                            bmax: v.bmax,
                            nent: v.nent.ptr,
                        },
                    };
                    self.idx_pick(x, &at)?;
                    self.trace_row_picked(p, l, x, &at, (rows, r0), false)?;
                }
                self.mark("indexer: picks")
            }
            Op::CompAttn { ring, cache, ratio, picked, sink, rope: r, heads, hpg } => {
                let (c, sn) = rope(r);
                let (kc, ktab, kts) = self.lead_ring(ring);
                let (cap, window) = (self.ring_cap[ring] as i32, p.d.rings[ring].window as i32);
                let (cache, ctab, cts) = cache.map_or((0, 0, 0), |x| self.lead_ring(x));
                let topk = v.dims.topk;
                let scale = (512f32).powf(-0.5);
                for r0 in (0..rows).step_by(v.ra) {
                    let rn = v.ra.min(rows - r0) as u32;
                    let (pr, sel, cnt) = (
                        pos + (r0 * 4) as u64,
                        if picked { v.sel.ptr + (r0 * topk * 4) as u64 } else { 0 },
                        if picked { v.cnt.ptr + (r0 * 4) as u64 } else { 0 },
                    );
                    let q = p.slot(Slot::Qo) + (r0 * heads * 512 * 2) as u64;
                    let (kt, ct) = (super::rows::tab_at(ktab, kts, r0), super::rows::tab_at(ctab, cts, r0));
                    launch!(k.attn_scores, [rn, (v.lmax as u32).div_ceil(32), (heads as u32).div_ceil(16)], [256, 1, 1], s; q, kc, cap, kt, kts as i64, cache, ct, cts as i64, sel, cnt, topk as i32, pr, v.sc.ptr, heads as i32, v.lmax as i32, window, ratio as i32, scale, 0u64)?;
                    launch!(k.attn_soft, [rn, (heads as u32).div_ceil(8), 1], [256, 1, 1], s; v.sc.ptr, p.small[sink].ptr, v.den.ptr, pr, cnt, heads as i32, v.lmax as i32, window, ratio as i32, 0u64)?;
                    // (the output laid out by group over the pass's n rows, as the grouped low rank reads it; rows that decode see no row beside them: no `last`)
                    launch!(k.attn_out, [rn, heads as u32, 1], [256, 1, 1], s; v.sc.ptr, v.den.ptr, kc, cap, kt, kts as i64, cache, ct, cts as i64, sel, cnt, topk as i32, pr, c, sn, p.slot(Slot::Att), n as i32, heads as i32, hpg as i32, v.lmax as i32, window, ratio as i32, r0 as i32, 0u64)?;
                }
                self.mark("attention: attention")
            }
            _ => err(format!("{op:?}: not one of the compressed attention's operations on a sequence's rings or states")),
        }
    }

    /// `Dyn::trace_picks`' record of a compressed attention's indexer pick (`Op::IdxPick` over its compressor's entries: DeepSeek-V4's) of layer `l` on n rows of
    /// one sequence whose rings are unpaged (another pass is an error): each row's picks, the entries it reads in their order, as the pick left them (`V4Run`'s
    /// sel and cnt); its queries and head weights' product as the scores read them (Iq after the rotary and, where the release's quantizations are simulated, the
    /// Hadamard rotation onto FP4's grid; Iw before `wscale`); the keys of the entries the pass's last row sees, in order. An entry is a pool of its own here
    /// (`mla::PickTrace`'s): a row's `pos` is the last entry it sees, so that it picks among entries 0 ..= pos (0 for a row that sees none, which picks none).
    pub(super) fn trace_comp_pick(&self, p: &Part, l: usize, op: &Op, n: usize, pos: u64) -> Result<()> {
        let Op::IdxPick(ref x) = *op else { return Ok(()) };
        let IdxKeys::Entries(cache) = x.keys else { return Ok(()) };
        if !std::ptr::eq(p, &self.big) || self.pick_trace.borrow().is_none() {
            return Ok(());
        }
        let v = self.v4.as_ref().ok_or_else(|| Error("a compressed attention's operation in a model loaded without its buffers".into()))?;
        if n > v.sel.bytes / (x.top_k.max(1) * 4) || !self.parts.is_empty() || self.table(cache) != 0 {
            return err(format!("layer {l}: picks traced on a pass of {n} rows of one sequence, its rings unpaged ({} sequences)", self.parts.len().max(1)));
        }
        let (tk, ih, ratio) = (x.top_k, x.heads, x.ratio);
        let (mut at, mut cnt, mut q, mut w) = (vec![0i32; n], vec![0i32; n], vec![0u16; n * ih * 128], vec![0u16; n * ih]);
        self.dev.sync()?;
        // SAFETY: the n positions the operation read, the rows' queries and head weights (slots of the part), into host memory of their sizes; synchronized after.
        unsafe {
            crate::dev::copy_d2h_async(at.as_mut_ptr() as *mut _, pos, n * 4, self.dev.stream)?;
            crate::dev::copy_d2h_async(q.as_mut_ptr() as *mut _, p.slot(Slot::Iq), q.len() * 2, self.dev.stream)?;
            crate::dev::copy_d2h_async(w.as_mut_ptr() as *mut _, p.slot(Slot::Iw), w.len() * 2, self.dev.stream)?;
        }
        self.dev.sync()?;
        if at.iter().any(|&x| x < 0) || at.windows(2).any(|w| w[1] != w[0] + 1) {
            return err(format!("layer {l}: picks traced on rows at positions {at:?}, where a pass's rows are one sequence's in order"));
        }
        v.cnt.read(0, &mut cnt)?;
        let mut picks = Vec::with_capacity(n);
        for (t, &c) in cnt.iter().enumerate() {
            if c < 0 || c as usize > tk {
                return err(format!("layer {l} row {t}: {c} picks of at most {tk}"));
            }
            let mut got = vec![0i32; c as usize];
            v.sel.read(t * tk * 4, &mut got)?;
            if got.iter().any(|&e| e < 0 || e as usize >= (at[t] as usize + 1) / ratio) {
                return err(format!("layer {l} row {t} at {}: picks {got:?}, past the entries it sees", at[t]));
            }
            picks.push(got.into_iter().map(|e| e as usize).collect());
        }
        // (each entry's key in the ring, a room of its own: entry e at e modulo its capacity)
        let entries = (at[n - 1] as usize + 1) / ratio;
        let (base, cap) = (self.cache(cache).0, self.ring_cap[cache]);
        let mut keys = Vec::with_capacity(entries);
        for e in 0..entries {
            let mut k = vec![0u16; 128];
            self.dev.sync()?;
            // SAFETY: one entry's 128 bf16 values in the ring, into host memory of that size; synchronized after.
            unsafe { crate::dev::copy_d2h_async(k.as_mut_ptr() as *mut _, base + ((e % cap) * 128 * 2) as u64, 256, self.dev.stream)? };
            self.dev.sync()?;
            keys.push(k);
        }
        // (a pick inside a candidate source's blocks: each row's entries in the blocks published for it, `V4Run`'s cand and ncand)
        let inside: Vec<Option<Vec<usize>>> = match x.cand {
            Some(super::mla::Cand::Inside { block, .. }) => {
                let (mut nb, mut blocks) = (vec![0i32; n], vec![0i32; n * v.ctop]);
                v.ncand.read(0, &mut nb)?;
                v.cand.read(0, &mut blocks)?;
                (0..n)
                    .map(|t| {
                        let seen = (at[t] as usize + 1) / ratio;
                        let c = (nb[t].max(0) as usize).min(v.ctop);
                        let mut b: Vec<usize> = blocks[t * v.ctop..][..c].iter().map(|&x| x.max(0) as usize).collect();
                        b.sort_unstable();
                        b.dedup();
                        Some(b.into_iter().flat_map(|k| k * block..((k + 1) * block).min(seen)).collect())
                    })
                    .collect()
            }
            _ => vec![None; n],
        };
        let f = |v: &[u16]| v.iter().map(|&h| f32::from_bits((h as u32) << 16)).collect::<Vec<f32>>();
        let rec = super::mla::PickTrace {
            layer: l,
            pos: at.iter().map(|&x| ((x as usize + 1) / ratio).saturating_sub(1)).collect(),
            picks,
            q: q.chunks(ih * 128).map(f).collect(),
            w: w.chunks(ih).map(f).collect(),
            keys: keys.iter().map(|k| f(k)).collect(),
            inside,
        };
        if let Some(t) = self.pick_trace.borrow_mut().as_mut() {
            t.push(rec);
        }
        Ok(())
    }
}
