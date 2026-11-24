//! The executor's keys and values in 12-bit lossless pages (R4f #332, R4b #141). A ring whose layer has tables (src/dynamic/kv12.rs) holds its
//! group's blocks as pages of the GPU library (glyd_gpu_kv_*), every bf16 value in 12 bits and read back bit for bit. A step's new keys and values
//! go from a staging buffer of its rows into their pages; a decode row's attention reads the pages (`kv_decode_from`: a sliding layer's window
//! from its first token); a prompt chunk's reads them where they are (`kv_prefill`, the kernel the bf16 cache's prompt attention is too). A verify pass's rows and
//! the warp kernels read them unpacked into a bf16 ring (`kv12_stage`).
use super::kv12::{shapes, KvShape, Tables};
use super::{Dyn, Op, Part};
use crate::dev::Buf;
use crate::ffi::{err, KvGeom, Lib, Result};

/// A page's entries in its own arena, past its heads' inline ones.
pub const ARENA: i64 = 15;
/// The places of each of a status's two rings (void and raw pages).
pub(super) const RING: i64 = 4096;

/// A status's words `kv12_lost` reads: the counters, then the void pages' ring.
pub(super) const VOID_WORDS: usize = 16 + RING as usize;
/// The values of a chunk of the escape pool.
const CHUNK: f64 = 64.0;
/// The decode attention's query heads to a kv head, at most.
const MAX_G: usize = 8;
/// The decode attention's least CTAs a sequence (glyd_gpu_kv_floor_set). Measured on an RTX 4080 SUPER, gemma-4-12B, a decode step of one user at
/// 1,024 tokens: the GPU's own (80 SMs, none) 12.92 ms, 8: 12.79, 16: 12.63, 32: 12.55 (bf16 pages 12.53); at 8,192 tokens 13.28 and 13.06 with 32.
/// A global layer's one kv head gave the library's plan 8 CTAs of the card's 80 SMs.
pub(super) const DECODE_FLOOR: i32 = 32;

/// A ring's pages: its geometry and pool, its tables, the layer number its pages carry in the shared pools, its group (block numbering), its
/// shape, and its decode's split (query heads to a kv head over `parts` calls' sequences, `g` each).
pub(super) struct RingPages {
    pub geom: KvGeom,
    pub pool: Buf,
    pub kbase: Buf,
    pub vbase: Buf,
    pub layer: i64,
    pub group: usize,
    pub shape: KvShape,
    pub parts: usize,
    pub g: usize,
    /// The bf16 ring a prompt chunk's pages are unpacked into: blocks of it (block b at b modulo this).
    pub stage_blocks: usize,
    /// Its decode's kind (`Lossless::kinds`: its window and its calls' sequences a row), whose lengths and first tokens a pass makes once.
    pub kind: usize,
}

/// The 12-bit pages of every ring that has them, the pools they share, and the step's buffers: the rows' keys and values before they are written,
/// their slots, lengths and first tokens, the unpacked ring and the decode's workspace.
pub(super) struct Lossless {
    pub rings: Vec<Option<RingPages>>,
    _escape: Buf,
    _raw: Vec<Buf>,
    pub status: Vec<Buf>,
    /// Each group's void pages named so far (its status's void ring's head as last read).
    pub seen: Vec<u32>,
    pub rows_k: Buf,
    pub rows_v: Buf,
    /// A pass's rows' slots in each group's blocks, and for each kind of decode (window, sequences a row) the rows' lengths and first tokens: made
    /// once a pass (`kv12_prepare`), every layer's write and decode reading its part's rows.
    pub slots: Vec<Buf>,
    pub kinds: Vec<(usize, usize)>,
    pub lens: Vec<Buf>,
    pub first: Vec<Buf>,
    pub stage_k: Buf,
    pub stage_v: Buf,
    pub ws: Buf,
    pub splits: i64,
    /// Device bytes of the pages, the pools and the step's buffers.
    #[cfg_attr(not(feature = "dev"), allow(dead_code))]
    pub bytes: usize,
}

/// A ring's layer and query heads: the layer whose attention writes it, and its query heads.
fn writer(d: &super::ModelDesc, ring: usize) -> Option<(usize, usize)> {
    d.layers.iter().enumerate().find_map(|(l, ops)| {
        ops.iter().find_map(|op| match op {
            Op::Attn(a) if a.writes && a.ring == ring => Some((l, a.nq)),
            _ => None,
        })
    })
}

/// The share of a page's values outside their windows its escape room is made for, at the least. Measured on gemma-4-12B: 0.035% in the worst layer
/// on the calibration's first 1,024 positions, 0.31% at 8,192 tokens (later positions are farther from the windows); room for 0.5% costs a sliding
/// page five chunks, 1.3% of its bytes.
const ROOM_SHARE: f64 = 0.005;

/// The escape chunks a page of a layer is given in the pool: twice what its pages took on the tables' prompts on average (`Tables::chunks`), and
/// at least what ROOM_SHARE of its values outside their windows takes (the pool runs dry where the text is not the calibration's: later positions,
/// other languages).
fn chunks_per_page(s: &KvShape, t: &Tables, l: usize) -> f64 {
    let values = 16.0 * (s.heads * s.channels) as f64 * 2.0;
    let inline = 3.0 * (s.heads * s.channels / 128) as f64 + ARENA as f64;
    let room = ((values * ROOM_SHARE - inline).max(0.0) / CHUNK).ceil();
    (t.chunks.get(l).map_or(0.0, |c| c.0) * 2.0).max(room).max(0.125)
}

impl Lossless {
    /// A ring's bytes a token in 12-bit pages (its keys and values, with its share of the escape pool), where its layer has a table and the
    /// decode attention takes its shape; None: the ring stays bf16.
    pub fn token_bytes(lib: &Lib, d: &super::ModelDesc, t: &Tables, ring: usize) -> Option<f64> {
        let (l, nq) = writer(d, ring)?;
        let s = shapes(d)[l]?;
        let lt = t.layers.get(l)?.as_ref()?;
        if (lt.heads, lt.channels) != (s.heads, s.channels) || s.one || nq % s.heads != 0 {
            return None;
        }
        let g = nq / s.heads;
        if g > MAX_G && (s.heads != 1 || g % MAX_G != 0) {
            return None; // (more query heads to a kv head than the decode takes, split only where they are contiguous: one kv head)
        }
        let page = lib.kv_page_bytes(s.heads as i64, s.channels as i64, ARENA).ok()? as f64;
        let chunk = lib.kv_escape_bytes(1).ok()? as f64;
        Some((page + chunks_per_page(&s, t, l) * chunk) / 16.0)
    }

    /// The pages of the rings `token_bytes` takes: `blocks[g]` blocks in group g (block 0 too, which no sequence holds), `ring_cap` the rings in use,
    /// `rows` the most rows a step writes, `decode_rows` the most a decode attention reads, `ctx` and `chunk` the executor's.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        lib: &Lib,
        d: &super::ModelDesc,
        t: &Tables,
        group: &[usize],
        blocks: &[usize],
        ring_cap: &[usize],
        rows: usize,
        decode_rows: usize,
        ctx: usize,
        chunk: usize,
        splits: i64,
    ) -> Result<Lossless> {
        let lay = shapes(d);
        let groups = blocks.len();
        let (mut chunks, mut rings, mut heads_raw) = (0f64, Vec::new(), Vec::<(i64, i64)>::new());
        for r in 0..d.rings.len() {
            if ring_cap[r] == 0 || Lossless::token_bytes(lib, d, t, r).is_none() {
                rings.push(None);
                continue;
            }
            let (l, nq) = writer(d, r).expect("token_bytes found it");
            let s = lay[l].expect("token_bytes found it");
            let n = blocks[group[r]] + 1;
            chunks += n as f64 * chunks_per_page(&s, t, l);
            let hp = (s.heads * s.channels / 128) as i64;
            match heads_raw.iter_mut().find(|(h, _)| *h == hp) {
                Some((_, k)) => *k += n as i64,
                None => heads_raw.push((hp, n as i64)),
            }
            rings.push(Some((l, nq, s, n)));
        }
        if rings.iter().all(Option::is_none) {
            return err("no ring of this model has 12-bit tables it can use");
        }
        // (a development build's GLYD_KV12_POOL=CHUNKS,SLABS: pools that small, to make pages void on purpose: usercheck --kv12)
        let squeeze: Option<(i64, i64)> = crate::tune!("GLYD_KV12_POOL").and_then(|v| {
            let (c, s) = v.split_once(',')?;
            Some((c.trim().parse().ok()?, s.trim().parse().ok()?))
        });
        let nchunks = squeeze.map_or((chunks.ceil() as i64).max(1024), |s| s.0.max(1));
        if std::env::var_os("GLYD_KV_FLOOR").is_none() {
            lib.kv_floor_set(DECODE_FLOOR)?;
        }
        let escape = Buf::zeroed(lib.kv_escape_bytes(nchunks)? as usize)?;
        // (a raw slab for one page in a hundred of each head count, at least 64)
        let mut raws = Vec::new();
        for &(hp, pages) in &heads_raw {
            let slabs = squeeze.map_or((pages / 100).max(64), |s| s.1);
            raws.push((hp, slabs, Buf::zeroed(lib.kv_raw_bytes(slabs, hp)? as usize)?));
        }
        let mut bytes = escape.bytes + raws.iter().map(|r| r.2.bytes).sum::<usize>();
        let (mut out, mut widest, mut most_g, mut kinds) = (Vec::new(), 0usize, 1usize, Vec::new());
        for (r, x) in rings.into_iter().enumerate() {
            let Some((l, nq, s, n)) = x else {
                out.push(None);
                continue;
            };
            let lt = t.layers[l].as_ref().expect("token_bytes found it");
            let (h, dd) = (s.heads as i64, s.channels as i64);
            let page = lib.kv_page_bytes(h, dd, ARENA)?;
            let hp = h * dd / 128;
            let (_, slabs, raw) = raws.iter().find(|x| x.0 == hp).expect("sized above");
            let geom = KvGeom { heads: h, dim: dd, arena: ARENA, page_stride: page, escape: escape.ptr, nchunks, raw: raw.ptr, nslabs: *slabs };
            let pool = Buf::zeroed(n * page as usize)?;
            let (kbase, vbase) = (Buf::new(lt.k.len())?, Buf::new(lt.v.len())?);
            // (a development build's GLYD_KV12_SHIFT=N: every base moved N, so that nearly every value is outside its window and a page goes raw:
            // with GLYD_KV12_POOL's few slabs, pages void on purpose)
            let shift: u8 = crate::tune!("GLYD_KV12_SHIFT").and_then(|v| v.parse().ok()).unwrap_or(0);
            let moved = |b: &[u8]| b.iter().map(|&x| (x + shift) % 121).collect::<Vec<u8>>();
            kbase.write(0, &moved(&lt.k))?;
            vbase.write(0, &moved(&lt.v))?;
            bytes += pool.bytes + kbase.bytes + vbase.bytes;
            let g = nq / s.heads;
            let parts = g.div_ceil(MAX_G);
            most_g = most_g.max(g / parts);
            widest = widest.max(s.heads * s.channels);
            let stage_blocks = if s.window > 0 { (s.window + chunk).div_ceil(16) + 2 } else { ctx.div_ceil(16) + 1 };
            let kind = match kinds.iter().position(|&k| k == (s.window, parts)) {
                Some(i) => i,
                None => {
                    kinds.push((s.window, parts));
                    kinds.len() - 1
                }
            };
            out.push(Some(RingPages { geom, pool, kbase, vbase, layer: l as i64, group: group[r], shape: s, parts, g: g / parts, stage_blocks, kind }));
        }
        let parts = out.iter().flatten().map(|x| x.parts).max().unwrap_or(1);
        let stage = out.iter().flatten().map(|x| x.stage_blocks * 16 * x.shape.heads * x.shape.channels * 2).max().unwrap_or(0);
        let (rows_k, rows_v) = (Buf::new(rows * widest * 2)?, Buf::new(rows * widest * 2)?);
        let slots: Vec<Buf> = (0..groups).map(|_| Buf::new(8 * rows)).collect::<Result<_>>()?;
        let lens: Vec<Buf> = kinds.iter().map(|k| Buf::new(4 * rows * k.1)).collect::<Result<_>>()?;
        let first: Vec<Buf> = kinds.iter().map(|k| Buf::new(4 * rows * k.1)).collect::<Result<_>>()?;
        let (stage_k, stage_v) = (Buf::new(stage)?, Buf::new(stage)?);
        // the decode's plan: the same splits for every call (a sequence's bits depend on them), the bf16 cache's decode's (attn::LibDecode)
        let heads = out.iter().flatten().map(|x| x.shape.heads).max().unwrap_or(1) as i64;
        let _ = ctx;
        let dmax = out.iter().flatten().map(|x| x.shape.channels).max().unwrap_or(128) as i64;
        let ws_bytes = lib.kv_decode_workspace((decode_rows * parts) as i64, heads, most_g as i64, dmax, splits)?;
        let ws = Buf::zeroed(ws_bytes)?;
        // (no lost bitmap: the void ring names a page each time it is made void, which a block reused by another sequence may be again)
        let status: Vec<Buf> = (0..groups).map(|_| Buf::zeroed(4 * (16 + 2 * RING as usize))).collect::<Result<_>>()?;
        bytes += rows_k.bytes * 2 + stage_k.bytes * 2 + ws.bytes + slots.iter().chain(&lens).chain(&first).chain(&status).map(|b| b.bytes).sum::<usize>();
        Ok(Lossless {
            rings: out,
            _escape: escape,
            _raw: raws.into_iter().map(|r| r.2).collect(),
            seen: vec![0; status.len()],
            status,
            rows_k,
            rows_v,
            slots,
            kinds,
            lens,
            first,
            stage_k,
            stage_v,
            ws,
            splits,
            bytes,
        })
    }

    /// Each group's status words (glyd_gpu.h's kv_write): [0] the exceptions with no place, [4]/[5] the escape chunks taken now and the most at once, [8]/[9] the raw slabs,
    /// [10] the pages made raw, [11] the pages made void.
    pub fn status(&self) -> Result<Vec<[u32; 16]>> {
        self.status
            .iter()
            .map(|b| {
                let mut w = [0u32; 16];
                b.read(0, &mut w)?;
                Ok(w)
            })
            .collect()
    }
}

impl Dyn {
    /// Ring `ring`'s 12-bit pages, where the big model's (part `p`) ring has them.
    pub(super) fn pages12(&self, p: &Part, ring: usize) -> Option<&RingPages> {
        self.lossless.as_ref().filter(|_| std::ptr::eq(p, &self.big)).and_then(|x| x.rings[ring].as_ref())
    }

    /// A pass's 12-bit arguments, once for every layer (`n` rows at `pos`, their parts where it is over several sequences): each part's rows' slots in
    /// each group's blocks (its table there), and each kind of decode's lengths and first tokens.
    pub(super) unsafe fn kv12_prepare(&self, n: usize, pos: (u64, u64)) -> Result<()> {
        let lx = self.lossless.as_ref().expect("pages");
        let s = self.dev.stream;
        // (a ring of each group, whose table is the group's)
        let of_group: Vec<Option<usize>> = (0..lx.slots.len()).map(|g| lx.rings.iter().position(|r| r.as_ref().is_some_and(|r| r.group == g))).collect();
        let parts: Vec<(Option<usize>, usize, usize)> =
            if self.parts.is_empty() { vec![(None, 0, n)] } else { self.parts.iter().enumerate().map(|(k, p)| (Some(k), p.row0, p.rows)).collect() };
        for (k, row0, rows) in parts {
            self.at.set(k);
            let off = 4 * row0 as u64;
            for (g, r) in of_group.iter().enumerate() {
                if let Some(r) = *r {
                    self.kern.kv12_rows(s, pos.0 + off, rows, self.table(r), 0, 1, lx.slots[g].ptr + 8 * row0 as u64, 0, 0)?;
                }
            }
            for (i, &(window, sp)) in lx.kinds.iter().enumerate() {
                let at = 4 * (row0 * sp) as u64;
                self.kern.kv12_rows(s, pos.1 + off, rows, 0, window, sp, 0, lx.lens[i].ptr + at, lx.first[i].ptr + at)?;
            }
        }
        self.at.set(None);
        Ok(())
    }

    /// The first row of the part an operation runs on now (0 outside a pass over several sequences).
    pub(super) fn row0(&self) -> usize {
        self.at.get().map_or(0, |k| self.parts[k].row0)
    }

    /// The step's n rows of ring `ring` (its staging rows, the preparation's output) into their pages, at the slots the pass made (`kv12_prepare`).
    pub(super) unsafe fn kv12_write(&self, ring: usize, n: usize) -> Result<()> {
        let lx = self.lossless.as_ref().expect("a ring with pages");
        let rp = lx.rings[ring].as_ref().expect("a ring with pages");
        let s = self.dev.stream;
        let w = (rp.shape.heads * rp.shape.channels) as i64;
        let g = rp.group;
        self.lib.kv_write(
            &rp.geom,
            rp.pool.ptr,
            rp.kbase.ptr,
            rp.vbase.ptr,
            rp.layer,
            lx.rows_k.ptr,
            w,
            lx.rows_v.ptr,
            w,
            lx.slots[g].ptr + 8 * self.row0() as u64,
            n as i64,
            lx.status[g].ptr,
            0,
            0,
            RING,
            s,
        )
    }

    /// The decode attention of n rows (queries `q`, out `att`, [n, nq d] each) over ring `ring`'s pages, at the lengths and first tokens the pass made.
    pub(super) unsafe fn kv12_decode(&self, ring: usize, n: usize, nq: usize, q: u64, att: u64) -> Result<()> {
        let lx = self.lossless.as_ref().expect("a ring with pages");
        let rp = lx.rings[ring].as_ref().expect("a ring with pages");
        let s = self.dev.stream;
        let at = 4 * (self.row0() * rp.parts) as u64;
        let stride = (nq * rp.shape.channels / rp.parts) as i64;
        // (every row a sequence of the decode's, its parts of query heads more; all of them the same sequence's pages: its table, rows 0 apart)
        self.lib.kv_decode_from(
            &rp.geom,
            rp.g as i64,
            rp.pool.ptr,
            rp.kbase.ptr,
            rp.vbase.ptr,
            q,
            stride,
            self.table(ring),
            0,
            lx.lens[rp.kind].ptr + at,
            lx.first[rp.kind].ptr + at,
            rp.shape.round,
            (n * rp.parts) as i64,
            lx.splits,
            rp.shape.scale,
            att,
            stride,
            lx.ws.ptr,
            lx.ws.bytes,
            s,
        )
    }

    /// A prompt chunk's attention over ring `ring`'s 12-bit pages (glyd_gpu_kv_prefill, after the step's `kv12_write`): the n rows at `start` (the pass's) .. of the
    /// part an operation runs on now, queries `q` and out `out` [n, nq d]; the keys before `start` read from the pages through the sequence's table, the chunk's own from
    /// the staging rows the preparation wrote; every query head of a kv head in one call (no parts). The bf16 cache's prompt attention is the same kernel
    /// (glyd_gpu_kv_prefill_bf16): the same bits.
    pub(super) unsafe fn kv12_prefill(&self, a: &super::AttnOp, n: usize, q: u64, out: u64) -> Result<()> {
        let lx = self.lossless.as_ref().expect("a ring with pages");
        let rp = lx.rings[a.ring].as_ref().expect("a ring with pages");
        let w = rp.shape.heads * rp.shape.channels;
        let at = (self.row0() * w * 2) as u64;
        let rows = Some((lx.rows_k.ptr + at, w as i64, lx.rows_v.ptr + at, w as i64));
        let qs = (a.nq * a.d) as i64;
        self.lib.kv_prefill(
            &rp.geom,
            (a.nq / rp.shape.heads) as i64,
            rp.pool.ptr,
            rp.kbase.ptr,
            rp.vbase.ptr,
            self.table(a.ring),
            q,
            qs,
            rows,
            n as i64,
            self.start.get() as i64,
            rp.shape.window as i64,
            rp.shape.scale,
            rp.shape.round,
            a.softcap as f64,
            0,
            out,
            qs,
            self.dev.stream,
        )
    }

    /// A prompt chunk's keys and values for the attention kind's prompt kernel: the pages of its tokens' blocks (from its window's first, `start` and n rows on) unpacked into
    /// the bf16 ring, block b at b modulo the ring's blocks; returns the ring's keys, values and capacity in tokens (tokens at position modulo it, no table).
    pub(super) unsafe fn kv12_stage(&self, ring: usize, start: usize, n: usize) -> Result<(u64, u64, usize)> {
        let lx = self.lossless.as_ref().expect("a ring with pages");
        let rp = lx.rings[ring].as_ref().expect("a ring with pages");
        let w = rp.shape.window;
        let lo = if w > 0 { (start + 1).saturating_sub(w) } else { 0 };
        let (b0, b1, sb) = (lo / 16, (start + n).div_ceil(16), rp.stage_blocks);
        if b1 - b0 > sb {
            return err(format!("{} blocks of a chunk, where the ring unpacked into holds {sb}", b1 - b0));
        }
        let row = (16 * rp.shape.heads * rp.shape.channels * 2) as u64;
        let mut b = b0;
        while b < b1 {
            let at = b % sb;
            let k = (b1 - b).min(sb - at);
            self.lib.kv_unpack(
                &rp.geom,
                rp.pool.ptr,
                rp.kbase.ptr,
                rp.vbase.ptr,
                self.table(ring) + 4 * b as u64,
                k as i64,
                lx.stage_k.ptr + at as u64 * row,
                lx.stage_v.ptr + at as u64 * row,
                self.dev.stream,
            )?;
            b += k;
        }
        Ok((lx.stage_k.ptr, lx.stage_v.ptr, sb * 16))
    }
}

impl Dyn {
    /// The sequences whose blocks had a page made void since the last call (their pools had no room: what the page holds is not what was written), as
    /// the scheduler's lost blocks of a slot step (`Lost::void` names slots); every sequence where the void ring overflowed. The scheduler takes their
    /// step back and runs them again from their tokens, alone where they are hit again (src/sched.rs `take_back`).
    pub(super) fn kv12_lost(&mut self) -> Result<crate::kv::Lost> {
        let mut lost = crate::kv::Lost::default();
        let (Some(lx), Some(pg)) = (self.lossless.as_mut(), self.paged.as_ref()) else { return Ok(lost) };
        let mut blocks: Vec<(usize, u32)> = Vec::new();
        // (a launched step's status as its lane copied it after the step, where it is that step that landed; else as the GPU holds it now)
        let copied = self.status_from.take();
        for (g, b) in lx.status.iter().enumerate() {
            let mut w = vec![0u32; VOID_WORDS];
            match copied {
                // SAFETY: the lane's pinned copy, VOID_WORDS a group, written by copies its step's event (waited for) followed.
                Some(p) => w.copy_from_slice(unsafe { std::slice::from_raw_parts(p.add(g * VOID_WORDS), VOID_WORDS) }),
                None => b.read(0, &mut w)?,
            }
            let (head, seen) = (w[12], lx.seen[g]);
            let n = head.wrapping_sub(seen);
            lost.overflow |= n as i64 > RING;
            for j in 0..n.min(RING as u32) {
                let e = w[16 + (seen.wrapping_add(j) % RING as u32) as usize];
                if e != 0 {
                    blocks.push((g, e - 1));
                }
            }
            lx.seen[g] = head;
        }
        for slot in 0..self.seqs.len() {
            if blocks.iter().any(|&(g, b)| pg.group_table(slot, g).contains(&b)) {
                lost.void.push(slot as i32);
            }
        }
        Ok(lost)
    }

    /// After a step: an error where a page was made void (its pools had no room: what it holds is not what was written), so that no reply goes on
    /// from it. (Until admission reserves the room a sequence may need and pauses one where the spare pool runs out, R4d.)
    pub(super) fn kv12_check(&self) -> Result<()> {
        let Some(lx) = &self.lossless else { return Ok(()) };
        for (g, w) in lx.status()?.iter().enumerate() {
            if w[11] > 0 {
                return err(format!("the 12-bit KV pages overflowed their pools: {} pages of group {g} made void ({} exceptions found no place)", w[11], w[0]));
            }
        }
        Ok(())
    }

    /// The 12-bit KV tables the next paged layout (`serve_rows`) uses for the rings they fit; None: bf16 pages.
    pub fn set_kv12(&mut self, t: Option<Tables>) {
        self.kv12 = t;
    }

    /// Each ring's pages that hold escape chunks, read from their headers (a page's directory words past its heads' headers and its used word): the
    /// pages, the chunks, and those of block 0 (which a table names where a sequence holds no block: what writes there is no sequence's).
    #[cfg(feature = "dev")]
    fn kv12_chunks_held(&self) -> Result<Vec<(usize, usize, usize, usize)>> {
        let Some(lx) = &self.lossless else { return Ok(Vec::new()) };
        let mut out = Vec::new();
        for (r, rp) in lx.rings.iter().enumerate() {
            let Some(rp) = rp else { continue };
            let (hp, n) = ((rp.shape.heads * rp.shape.channels / 128) as usize, rp.pool.bytes / rp.geom.page_stride as usize);
            let (mut pages, mut chunks, mut zero) = (0, 0, 0);
            for b in 0..n {
                let mut dir = [0u32; 17];
                rp.pool.read(b * rp.geom.page_stride as usize + 16 * hp, &mut dir)?;
                let held = dir[1..].iter().filter(|&&d| d != 0).count();
                if held > 0 {
                    (pages, chunks) = (pages + 1, chunks + held);
                    zero += if b == 0 { held } else { 0 };
                }
            }
            out.push((r, pages, chunks, zero));
        }
        Ok(out)
    }

    /// The selected sequence's keys and values of ring `ring` read back from its 12-bit pages at its tokens before n, as `kv_rows` reads the bf16 ring.
    #[cfg(feature = "dev")]
    unsafe fn kv12_back(&self, ring: usize, n: usize) -> Result<(Vec<u16>, Vec<u16>, usize)> {
        let rp = self.lossless.as_ref().and_then(|x| x.rings[ring].as_ref()).expect("a ring with pages");
        let lx = self.lossless.as_ref().expect("a ring with pages");
        let (w, keep) = (rp.shape.heads * rp.shape.channels, if rp.shape.window > 0 { rp.shape.window } else { n });
        let first = n.saturating_sub(keep);
        let (_, _, cap) = self.kv12_stage(ring, n - 1, 1)?;
        self.dev.sync()?;
        let (mut sk, mut sv) = (vec![0u16; cap * w], vec![0u16; cap * w]);
        lx.stage_k.read(0, &mut sk)?;
        lx.stage_v.read(0, &mut sv)?;
        let (mut k, mut v) = (Vec::with_capacity((n - first) * w), Vec::with_capacity((n - first) * w));
        for p in first..n {
            let at = (p % cap) * w;
            k.extend_from_slice(&sk[at..at + w]);
            v.extend_from_slice(&sv[at..at + w]);
        }
        Ok((k, v, first))
    }
}

/// `glyd-engine kv12check --dir DIR [--case gemma4] [--level L] [--positions N] [--prompts K] [--prompt-tokens P] [--new T]`: a model the executor runs, in bf16 pages and
/// in 12-bit pages of tables made from K sequences of N corpus tokens: each ring's keys and values read back from the 12-bit pages against the bf16 ones (bit for bit), the
/// greedy replies of P prompt tokens (T new) of each (the prompt's own pass reads the same values: its first token is the same; decode reads the pages with the library's
/// attention), the pools' status, and the bytes a token takes.
#[cfg(feature = "dev")]
pub fn check(a: &crate::args::Args) -> Result<()> {
    use crate::conv::Gen;
    use std::path::Path;
    let (case, level) = (a.flag("case").unwrap_or("gemma4"), a.flag("level").unwrap_or("penguin"));
    let random = a.flag("dir").is_none();
    let (positions, prompts) = (a.num("positions", if random { 512 } else { 1024usize })?, a.num("prompts", 4usize)?);
    let (plen, new) = (a.num("prompt-tokens", if random { 700 } else { 1024usize })?, a.num("new", 32usize)?);
    let (mut e, dir, mut seqs) = match a.flag("dir") {
        Some(d) => {
            let dir = Path::new(d);
            let o = super::Opts { ctx: positions.max(plen + new) + 64, chunk: 256, graph: true, device: 0, layout: crate::logits::level_layout(level)? };
            let e = crate::logits::load_dyn(case, dir, &o, level)?;
            let docs = crate::calib::corpus_docs(&crate::tok::Tok::load(dir)?, None)?;
            let seqs: Vec<Vec<u32>> = crate::kvcal::sequences(&docs, positions.max(plen)).into_iter().map(|(s, _)| s).filter(|s| s.len() >= plen).collect();
            (e, dir.display().to_string(), seqs)
        }
        None => {
            // (the case's random model, in the logits cases' context, so the prompts are shorter; the tokens of a fixed xorshift)
            let e = crate::logits::random_dyn(case, level, "kv12check")?;
            let (vocab, mut x) = (e.desc().vocab as u64, 0x9E37_79B9_7F4A_7C15u64);
            let mut next = || {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                (x % vocab) as u32
            };
            let seqs = (0..=prompts).map(|_| (0..positions.max(plen)).map(|_| next()).collect()).collect();
            (e, format!("a random {case} model"), seqs)
        }
    };
    if seqs.len() < 2 {
        return err("the corpus makes too few sequences");
    }
    let prompt: Vec<u32> = seqs.pop().expect("two at least")[..plen].to_vec();
    seqs.truncate(prompts);
    let reply = crate::sched::Reply {
        params: crate::sampler::Params { temperature: 0.0, ..Default::default() },
        max_new: new,
        ignore_eos: true,
        logprobs: None,
        stop_ids: Vec::new(),
    };
    let t0 = std::time::Instant::now();
    let tables = super::kv12::calibrate(&mut e, &seqs.iter().map(|s| s[..positions.min(s.len())].to_vec()).collect::<Vec<_>>(), &dir)?;
    println!("tables of {} sequences in {:.1} s", seqs.len(), t0.elapsed().as_secs_f64());
    let (mean, most) = tables.chunks.iter().fold((0f64, 0u32), |(m, x), c| (m.max(c.0), x.max(c.1)));
    println!("escape chunks a page on the tables' prompts: {mean:.2} on average in the layer that takes most, {most} at most in a page (17: raw)");
    let lay = super::kv12::shapes(&e.big.d);
    let mut rings: Vec<(usize, super::kv12::KvShape)> = lay.iter().flatten().map(|s| (s.ring, *s)).collect();
    rings.dedup_by_key(|x| x.0);
    // the rings as the executor loads them (no pages: position p at p modulo a ring's places), the reference both paged runs are held to
    let (plain_want, _) = e.reply(&prompt, &reply, &mut |_| true)?;
    let plain_n = e.n;
    let mut plain = Vec::new();
    for (_, s) in &rings {
        plain.push(e.kv_rows(s, plain_n)?);
    }
    // bf16 pages (--prof: a decode step's stages timed in its graph, here and on the 12-bit pages)
    let prof = a.has("prof");
    let stages = |e: &Dyn| -> Result<Vec<(&'static str, f64)>> {
        if prof {
            e.prof_take()
        } else {
            Ok(Vec::new())
        }
    };
    e.set_kv12(None);
    e.serve_rows(1)?;
    if prof {
        e.prof_in_graph()?;
    }
    let (want, _) = e.reply(&prompt, &reply, &mut |_| true)?;
    let bf_stages = stages(&e)?;
    let n = e.n;
    let mut bf = Vec::new();
    for (_, s) in &rings {
        bf.push(e.kv_rows(s, n)?);
    }
    println!(
        "{} bf16 pages against the plain rings: {} of {} tokens the same reply",
        if plain.iter().zip(&bf).all(|(a, b)| a == b) { "ok  " } else { "DIFF" },
        plain_want.iter().zip(&want).take_while(|(a, b)| a == b).count(),
        want.len()
    );
    // (from here the 12-bit pages are held to the plain rings)
    let bf = plain;
    let want = plain_want;
    // (the tables against the bf16 replies' own keys and values: each layer's share outside its windows, as the pages will find it)
    let (mut most, mut sum) = (0f64, 0f64);
    let layer_of = |r: usize| lay.iter().position(|x| x.is_some_and(|x| x.ring == r));
    for (((_, s), (k, v, _)), lt) in rings.iter().zip(&bf).zip(rings.iter().map(|(r, _)| layer_of(*r).and_then(|l| tables.layers[l].as_ref()))) {
        let mut h = super::kv12::Hist::new(s.heads, s.channels, false);
        h.add(k, v);
        if let Some(lt) = lt {
            let o = h.outside(lt);
            (most, sum) = (most.max(o), sum + o);
        }
    }
    let worst_cal = tables.outside.iter().cloned().fold(0f64, f64::max);
    println!(
        "the reply's values outside the tables' windows: {:.3}% a layer on average, {:.3}% at most (the calibration's most: {:.3}%)",
        100.0 * sum / rings.len() as f64,
        100.0 * most,
        100.0 * worst_cal
    );
    // 12-bit pages
    e.set_kv12(Some(tables));
    e.serve_rows(1)?;
    let Some(lx) = e.lossless.as_ref() else { return err("no ring took the 12-bit pages") };
    let taken = lx.rings.iter().filter(|r| r.is_some()).count();
    if prof {
        e.prof_in_graph()?;
    }
    let (got, _) = e.reply(&prompt, &reply, &mut |_| true)?;
    if prof {
        let tw = stages(&e)?;
        let total = |v: &[(&str, f64)]| v.iter().map(|x| x.1).sum::<f64>();
        println!("a decode step's stages (ms, its graph's last replay): bf16 pages {:.3} in all, 12-bit pages {:.3}", total(&bf_stages), total(&tw));
        let mut names: Vec<&str> = bf_stages.iter().chain(&tw).map(|x| x.0).collect();
        names.dedup();
        let mut seen = std::collections::HashSet::new();
        for n in names.into_iter().filter(|n| seen.insert(*n)) {
            let (b, t) = (bf_stages.iter().find(|x| x.0 == n).map_or(0.0, |x| x.1), tw.iter().find(|x| x.0 == n).map_or(0.0, |x| x.1));
            println!("  {n:<36} {b:>8.3} {t:>8.3}");
        }
    }
    if e.n != n {
        return err(format!("the 12-bit reply left {} tokens in the cache, the bf16 one {n}", e.n));
    }
    // (the prompt's tokens are held to the plain rings bit for bit: the prompt's pass reads the values unpacked from the pages with the plain kernel; a
    // decode row's attention is the library's, whose sums are its own, so the values the decode's later layers write are compared, not required)
    let (mut differ, mut values, mut later, mut later_same) = (0usize, 0usize, 0usize, 0usize);
    for ((r, s), (k, v, f)) in rings.iter().zip(&bf) {
        if e.lossless.as_ref().is_some_and(|x| x.rings[*r].is_some()) {
            // SAFETY: the ring's pages and the stage are this executor's; the stream is synchronized in it
            let (k2, v2, f2) = unsafe { e.kv12_back(*r, n)? };
            if f2 != *f || k2.len() != k.len() {
                return err(format!("ring {r}: read back from token {f2} ({} values), the plain ring from {f} ({})", k2.len(), k.len()));
            }
            let w = s.heads * s.channels;
            for (t, p) in (*f..n).enumerate() {
                let (a, b) = (t * w, (t + 1) * w);
                let bad = k[a..b].iter().zip(&k2[a..b]).filter(|(x, y)| x != y).count() + v[a..b].iter().zip(&v2[a..b]).filter(|(x, y)| x != y).count();
                if p < prompt.len() {
                    (differ, values) = (differ + bad, values + 2 * w);
                } else {
                    (later, later_same) = (later + 1, later_same + (bad == 0) as usize);
                }
            }
        }
    }
    let same = want.iter().zip(&got).take_while(|(x, y)| x == y).count();
    println!("{taken} of {} rings in 12-bit pages; {n} tokens in the cache", rings.len());
    println!("{} read back: {differ} of {values} values of the prompt's tokens differ from the plain rings'", if differ == 0 { "ok  " } else { "FAIL" });
    println!(
        "{} greedy replies: the first {same} of {new} tokens the same (the prompt's token {}); the decode's tokens' values the plain ones in {later_same} of {later} ring-tokens",
        if want.first() == got.first() { "ok  " } else { "FAIL" },
        if want.first() == got.first() { "the same" } else { "differs" }
    );
    let held = e.kv12_chunks_held()?;
    let (pages, chunks, zero) = held.iter().fold((0, 0, 0), |a, h| (a.0 + h.1, a.1 + h.2, a.2 + h.3));
    let tables_said: f64 = held
        .iter()
        .map(|h| e.kv12.as_ref().and_then(|t| lay.iter().position(|s| s.is_some_and(|s| s.ring == h.0)).and_then(|l| t.chunks.get(l))).map_or(0.0, |c| c.0))
        .sum();
    println!(
        "escape chunks held: {chunks} in {pages} pages ({zero} in block 0, no sequence's); the tables counted {tables_said:.2} a page a ring on their prompts"
    );
    let lx = e.lossless.as_ref().expect("taken above");
    for (g, w) in lx.status()?.iter().enumerate() {
        println!(
            "group {g}: {} exceptions with no place, escape chunks {} now (most {}), raw slabs {} (most {}), pages made raw {}, void {}",
            w[0], w[4], w[5], w[8], w[9], w[10], w[11]
        );
    }
    let bf16: usize = rings.iter().map(|(_, s)| 2 * s.heads * s.channels * 2).sum();
    let twelve: f64 = rings
        .iter()
        .map(|(r, s)| e.kv12.as_ref().and_then(|t| Lossless::token_bytes(&e.lib, &e.big.d, t, *r)).unwrap_or((2 * s.heads * s.channels * 2) as f64))
        .sum();
    println!(
        "a token in every ring: {bf16} bytes in bf16, {twelve:.0} in 12-bit pages ({:.1}%); the pages' memory {:.2} GiB",
        100.0 * twelve / bf16 as f64,
        lx.bytes as f64 / (1u64 << 30) as f64
    );
    if differ != 0 || want.first() != got.first() || lx.status()?.iter().any(|w| w[11] != 0) {
        return err("the 12-bit pages do not hold what the bf16 ones do");
    }
    Ok(())
}
