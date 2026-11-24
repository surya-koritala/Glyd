//! The compressed attention layer kind (`runtime::kinds::Comp`: DeepSeek-V4's) as the executor's operations, each a mechanism: the queries (`Op::CompQ`), the window's
//! key-value row (`Op::CompKv`), the compressed keys and values (`Op::Compress`, the attention's and the indexer's), the indexer's queries (`Op::IdxQ`) and pick
//! (`Op::IdxPick`, every indexer's: src/dynamic/mla.rs), the attention over the window and the compressed entries (`Op::CompAttn`), and the output's grouped low rank
//! (`Op::OGroupLora`, `Op::OGroupLoraStack`). The arithmetic is kernels/dsv4.cu's (the release's), its buffers `V4Run`'s.

use super::mla::PickAt;
use super::{Dyn, IdxKeys, Op, Part, Slot};
use crate::ffi::{err, Error, Result};
use crate::gpu_ops::GpuOps;
use crate::launch;

impl Dyn {
    /// One of the compressed attention's operations on n rows at positions `pos`.
    pub(super) unsafe fn op_comp(&self, p: &Part, op: &Op, n: usize, pos: u64) -> Result<()> {
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
            Op::CompQ { rope: r, heads } => {
                let (c, sn) = rope(r);
                launch!(k.q_prep, [nu, heads as u32, 1], [128, 1, 1], s; p.slot(Slot::Qo), c, sn, pos, eps)?;
                self.mark("attention: q norm, rotary")
            }
            Op::CompKv { norm, rope: r, ring } => {
                let (c, sn) = rope(r);
                let kv = p.slot(Slot::Kv) + self.rows_before(p.d.width(Slot::Kv));
                launch!(k.kv_prep, [nu, 1, 1], [128, 1, 1], s; kv, p.small[norm].ptr, c, sn, pos, self.cache(ring).0, self.ring_cap[ring] as i32, eps, v.dims.sim as i32)?;
                self.mark("attention: kv norm, rotary, cache")
            }
            Op::Compress { w, ape, norm, d, ratio, rope: r, cache, state, indexer } => {
                let wd = if ratio == 4 { 2 * d } else { d };
                let (c, sn) = rope(r);
                let (h, cache, st0, st1) = (p.slot(Slot::H) + self.rows_before(p.d.hidden), self.state(cache), self.state(state), self.state(state + 1));
                if n == 1 && hid % 256 == 0 && hid <= 8192 {
                    launch!(k.lin32_1, [((2 * wd) as u32).div_ceil(8), 1, 1], [256, 1, 1], s; h, p.small[w].ptr, p.slot(Slot::Cp), hid, (2 * wd) as i32)?;
                } else {
                    launch!(k.lin32, [((2 * wd) as u32).div_ceil(8), nu.div_ceil(8), 1], [256, 1, 1], s; h, p.small[w].ptr, p.slot(Slot::Cp), n as i32, hid, (2 * wd) as i32)?;
                }
                launch!(k.comp_pool, [(d as u32).div_ceil(128), 1, 1], [128, 1, 1], s; p.slot(Slot::Cp), p.small[ape].ptr, st0, st1, v.pooled.ptr, pos, n as i32, ratio as i32, d as i32, (ratio == 4) as i32)?;
                launch!(k.comp_out, [(n / ratio + 1) as u32, 1, 1], [128, 1, 1], s; v.pooled.ptr, p.small[norm].ptr, c, sn, pos, n as i32, cache, ratio as i32, d as i32, indexer as i32, eps, v.dims.sim as i32)?;
                self.mark(if indexer { "indexer: compressor" } else { "attention: compressor" })
            }
            Op::IdxQ { rope: r, heads } => {
                let (c, sn) = rope(r);
                launch!(k.idx_q, [nu, heads as u32, 1], [128, 1, 1], s; p.slot(Slot::Iq), c, sn, pos, v.dims.sim as i32)?;
                self.mark("indexer: q")
            }
            Op::IdxPick(ref x) => {
                let IdxKeys::State(cache) = x.keys else { return err("a compressed attention's indexer whose keys are not its compressor's entries") };
                let (iq, iw, entries) = (p.slot(Slot::Iq) + self.rows_before(x.heads * 128), p.slot(Slot::Iw) + self.rows_before(x.heads), self.state(cache));
                for r0 in (0..n).step_by(v.ra) {
                    let at = PickAt {
                        q: iq + (r0 * x.heads * 128 * 2) as u64,
                        w: iw + (r0 * x.heads * 2) as u64,
                        keys: entries,
                        stride: 128,
                        tab: 0,
                        pos: pos + (r0 * 4) as u64,
                        rows: v.ra.min(n - r0),
                        scores: v.isc.ptr,
                        emax: v.emax,
                        sel: v.sel.ptr + ((row0 + r0) * x.top_k * 4) as u64,
                        sk: x.top_k,
                        cnt: v.cnt.ptr + ((row0 + r0) * 4) as u64,
                    };
                    self.idx_pick(x, &at)?;
                }
                self.mark("indexer: picks")
            }
            Op::CompAttn { ring, cache, ratio, sink, rope: r, heads, hpg } => {
                let (c, sn) = rope(r);
                let (kc, cap, window) = (self.cache(ring).0, self.ring_cap[ring] as i32, p.d.rings[ring].window as i32);
                let cache = cache.map_or(0, |x| self.state(x));
                let (topk, picked) = (v.dims.topk, ratio == 4);
                let scale = (512f32).powf(-0.5);
                for r0 in (0..n).step_by(v.ra) {
                    let rn = v.ra.min(n - r0) as u32;
                    let (pr, sel, cnt) = (
                        pos + (r0 * 4) as u64,
                        if picked { v.sel.ptr + ((row0 + r0) * topk * 4) as u64 } else { 0 },
                        if picked { v.cnt.ptr + ((row0 + r0) * 4) as u64 } else { 0 },
                    );
                    let q = p.slot(Slot::Qo) + self.rows_before(heads * 512) + (r0 * heads * 512 * 2) as u64;
                    launch!(k.attn_scores, [rn, (v.lmax as u32).div_ceil(32), (heads as u32).div_ceil(16)], [256, 1, 1], s; q, kc, cap, cache, sel, cnt, topk as i32, pr, v.sc.ptr, heads as i32, v.lmax as i32, window, ratio as i32, scale)?;
                    launch!(k.attn_soft, [rn, (heads as u32).div_ceil(8), 1], [256, 1, 1], s; v.sc.ptr, p.small[sink].ptr, v.den.ptr, pr, cnt, heads as i32, v.lmax as i32, window, ratio as i32)?;
                    // (kernels/dsv4.cu V4_AQ: 8 warps a head, a head a block)
                    launch!(k.attn_out, [rn, heads as u32, 1], [256, 1, 1], s; v.sc.ptr, v.den.ptr, kc, cap, cache, sel, cnt, topk as i32, pr, c, sn, p.slot(Slot::Att), rows as i32, heads as i32, hpg as i32, v.lmax as i32, window, ratio as i32, (row0 + r0) as i32)?;
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
}
