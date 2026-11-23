//! The executor's keys and values in 12-bit lossless pages (R4f #332, R4b #141). A ring whose layer has tables (src/dynamic/kv12.rs) holds its
//! group's blocks as pages of the GPU library (glyd_gpu_kv_*), every bf16 value in 12 bits and read back bit for bit. A step's new keys and values
//! go from a staging buffer of its rows into their pages; a decode row's attention reads the pages (`kv_decode_from`: a sliding layer's window
//! from its first token); a prompt chunk's pages are unpacked into a bf16 ring and the family's own attention kernel runs on it.
use super::kv12::{shapes, KvShape, Tables};
use super::{Dyn, Op, Part};
use crate::dev::Buf;
use crate::ffi::{err, KvGeom, Lib, Result};

/// A page's entries in its own arena, past its heads' inline ones.
pub const ARENA: i64 = 15;
/// The places of each of a status's two rings (void and raw pages).
const RING: i64 = 4096;
/// The values of a chunk of the escape pool.
const CHUNK: f64 = 64.0;
/// The decode attention's query heads to a kv head, at most.
const MAX_G: usize = 8;

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
}

/// The 12-bit pages of every ring that has them, the pools they share, and the step's buffers: the rows' keys and values before they are written,
/// their slots, lengths and first tokens, the unpacked ring and the decode's workspace.
pub(super) struct Lossless {
    pub rings: Vec<Option<RingPages>>,
    _escape: Buf,
    _raw: Vec<Buf>,
    pub status: Vec<Buf>,
    pub lost: Vec<Buf>,
    pub nlost: Vec<i64>,
    pub rows_k: Buf,
    pub rows_v: Buf,
    pub slots: Buf,
    pub lens: Buf,
    pub first: Buf,
    pub stage_k: Buf,
    pub stage_v: Buf,
    pub ws: Buf,
    pub splits: i64,
    /// Device bytes of the pages, the pools and the step's buffers.
    #[cfg_attr(not(feature = "dev"), allow(dead_code))]
    pub bytes: usize,
}

/// A ring's layer and query heads: the layer whose preparation writes it, and that operation's query heads.
fn writer(d: &super::ModelDesc, ring: usize) -> Option<(usize, usize)> {
    d.layers
        .iter()
        .enumerate()
        .find_map(|(l, ops)| ops.iter().find_map(|op| matches!(*op, Op::QkvPrep { ring: r, .. } if r == ring).then_some(l)).map(|l| (l, nq_of(&ops[..], ring))))
}

fn nq_of(ops: &[Op], ring: usize) -> usize {
    ops.iter().find_map(|op| if let Op::QkvPrep { nq, ring: r, .. } = *op { (r == ring).then_some(nq) } else { None }).unwrap_or(0)
}

/// The escape chunks a page of a layer is given in the pool: twice what its pages took on the tables' prompts on average (`Tables::chunks`; where
/// they were not counted, from the share of its values outside their windows as if spread evenly), at least one in eight pages.
fn chunks_per_page(s: &KvShape, t: &Tables, l: usize) -> f64 {
    let mean = t.chunks.get(l).map(|c| c.0).unwrap_or_else(|| {
        let values = 16.0 * (s.heads * s.channels) as f64 * 2.0;
        let inline = 3.0 * (s.heads * s.channels / 128) as f64 + ARENA as f64;
        (values * t.outside.get(l).copied().unwrap_or(0.0) - inline).max(0.0) / CHUNK
    });
    (mean * 2.0).max(0.125)
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
        let nchunks = (chunks.ceil() as i64).max(1024);
        let escape = Buf::zeroed(lib.kv_escape_bytes(nchunks)? as usize)?;
        // (a raw slab for one page in a hundred of each head count, at least 64)
        let mut raws = Vec::new();
        for &(hp, pages) in &heads_raw {
            let slabs = (pages / 100).max(64);
            raws.push((hp, slabs, Buf::zeroed(lib.kv_raw_bytes(slabs, hp)? as usize)?));
        }
        let mut bytes = escape.bytes + raws.iter().map(|r| r.2.bytes).sum::<usize>();
        let (mut out, mut widest, mut most_g) = (Vec::new(), 0usize, 1usize);
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
            kbase.write(0, &lt.k)?;
            vbase.write(0, &lt.v)?;
            bytes += pool.bytes + kbase.bytes + vbase.bytes;
            let g = nq / s.heads;
            let parts = g.div_ceil(MAX_G);
            most_g = most_g.max(g / parts);
            widest = widest.max(s.heads * s.channels);
            let stage_blocks = if s.window > 0 { (s.window + chunk).div_ceil(16) + 2 } else { ctx.div_ceil(16) + 1 };
            out.push(Some(RingPages { geom, pool, kbase, vbase, layer: l as i64, group: group[r], shape: s, parts, g: g / parts, stage_blocks }));
        }
        let parts = out.iter().flatten().map(|x| x.parts).max().unwrap_or(1);
        let stage = out.iter().flatten().map(|x| x.stage_blocks * 16 * x.shape.heads * x.shape.channels * 2).max().unwrap_or(0);
        let (rows_k, rows_v) = (Buf::new(rows * widest * 2)?, Buf::new(rows * widest * 2)?);
        let (slots, lens, first) = (Buf::new(8 * rows)?, Buf::new(4 * rows * parts)?, Buf::new(4 * rows * parts)?);
        let (stage_k, stage_v) = (Buf::new(stage)?, Buf::new(stage)?);
        // the decode's plan: the same splits for every call (a sequence's bits depend on them), from one sequence of the longest context
        let heads = out.iter().flatten().map(|x| x.shape.heads).max().unwrap_or(1) as i64;
        let splits = lib.kv_decode_splits(1, heads, ctx.div_ceil(16) as i64)?;
        let dmax = out.iter().flatten().map(|x| x.shape.channels).max().unwrap_or(128) as i64;
        let ws_bytes = lib.kv_decode_workspace((decode_rows * parts) as i64, heads, most_g as i64, dmax, splits)?;
        let ws = Buf::zeroed(ws_bytes)?;
        let (mut status, mut lost, mut nlost) = (Vec::new(), Vec::new(), Vec::new());
        for &b in blocks.iter().take(groups) {
            status.push(Buf::zeroed(4 * (16 + 2 * RING as usize))?);
            let nl = 2 * (b as i64 + 1);
            lost.push(Buf::zeroed(4 * (nl as usize).div_ceil(32).max(1))?);
            nlost.push(nl);
        }
        bytes += rows_k.bytes * 2 + slots.bytes + lens.bytes * 2 + stage_k.bytes * 2 + ws.bytes + status.iter().chain(&lost).map(|b| b.bytes).sum::<usize>();
        Ok(Lossless {
            rings: out,
            _escape: escape,
            _raw: raws.into_iter().map(|r| r.2).collect(),
            status,
            lost,
            nlost,
            rows_k,
            rows_v,
            slots,
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
    #[cfg(feature = "dev")]
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

    /// The step's n rows of ring `ring` (its staging rows, the preparation's output) into their pages, at the positions `pos`.
    pub(super) unsafe fn kv12_write(&self, ring: usize, n: usize, pos: u64) -> Result<()> {
        let lx = self.lossless.as_ref().expect("a ring with pages");
        let rp = lx.rings[ring].as_ref().expect("a ring with pages");
        let s = self.dev.stream;
        self.kern.kv12_rows(s, pos, n, self.table(ring), 0, 1, lx.slots.ptr, lx.lens.ptr, lx.first.ptr)?;
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
            lx.slots.ptr,
            n as i64,
            lx.status[g].ptr,
            lx.lost[g].ptr,
            lx.nlost[g],
            RING,
            s,
        )
    }

    /// The decode attention of n rows (queries `q`, out `att`, [n, nq d] each) over ring `ring`'s pages at the positions `pos` (a sliding layer's window from its first token).
    pub(super) unsafe fn kv12_decode(&self, ring: usize, n: usize, pos: u64, nq: usize, q: u64, att: u64) -> Result<()> {
        let lx = self.lossless.as_ref().expect("a ring with pages");
        let rp = lx.rings[ring].as_ref().expect("a ring with pages");
        let s = self.dev.stream;
        self.kern.kv12_rows(s, pos, n, 0, rp.shape.window, rp.parts, lx.slots.ptr, lx.lens.ptr, lx.first.ptr)?;
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
            lx.lens.ptr,
            lx.first.ptr,
            true,
            (n * rp.parts) as i64,
            lx.splits,
            1.0,
            att,
            stride,
            lx.ws.ptr,
            lx.ws.bytes,
            s,
        )
    }

    /// A prompt chunk's keys and values for the family's attention kernel: the pages of its tokens' blocks (from its window's first, `start` and n rows on) unpacked into
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
    /// The 12-bit KV tables the next paged layout (`serve_rows`) uses for the rings they fit; None: bf16 pages.
    pub fn set_kv12(&mut self, t: Option<Tables>) {
        self.kv12 = t;
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
    let (positions, prompts) = (a.num("positions", if random { 512 } else { 2048usize })?, a.num("prompts", 4usize)?);
    let (plen, new) = (a.num("prompt-tokens", if random { 700 } else { 1500usize })?, a.num("new", 32usize)?);
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
    // bf16 pages
    e.set_kv12(None);
    e.serve_rows(1)?;
    let (want, _) = e.reply(&prompt, &reply, &mut |_| true)?;
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
    let (got, _) = e.reply(&prompt, &reply, &mut |_| true)?;
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
