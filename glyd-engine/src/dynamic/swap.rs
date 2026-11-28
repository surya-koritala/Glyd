//! Preemption by swapping (R4d #328; docs/design/runtime.md section 5, "Preemption"): a sequence the scheduler sends back for room has its cache copied to the host's swap
//! area and back, rather than being run again from its tokens (`ModelStep::swap_out`, `swap_in`). What is copied is all a step of it reads, for every family the same way:
//! in each layer group the blocks of its tokens its table holds (a sliding group's from its window on, a shared start's from its floor), each ring's keys and values of
//! them (a ring of keys alone, a latent cache's or compressed entries', its keys); its recurrent states (the delta rule's and the convolutions', Mamba's); and their copy at its mark where it has one. A ring in bf16 pages is copied as its bytes.
//! A ring in 12-bit pages is copied as its pages are, in their 12 bits: each page's bytes, the escape chunks it holds and its raw slab, in an image of the page
//! (`Lib::kv_save`), and loaded into new pages with chunks and slabs lent to their new blocks (`Lib::kv_load`, the directory naming them): the values come back bit for
//! bit, and the link carries a page's bytes and what it holds, not 16 bits a value. A page that finds no chunk or slab when it is loaded is void, as one written is.
//!
//! The copies run on a stream of their own, their host end pinned memory mapped into the device's addresses: one launch of `swap_copy` moves every bf16 block however
//! scattered its pools are, and the library's unpack and write read and write the host's memory themselves, with no staging buffer on the GPU, whose memory is what ran
//! out. Out: the copy runs beside the next steps, in two parts: the first `head` blocks of each group (head: the sequences, the most blocks a decode step takes of a
//! group), then the rest and the states. The blocks given back keep the part that reads them as a fence (`Pages::release_fenced`) and are taken after every free block
//! that has none, the sequence's first entries first; a step that takes one waits for its part alone (`Dyn::fence_wait`). The slot's states are emptied when the slot
//! is used again, after the copy (`Dyn::empty`). Back: the copy runs beside the other sequences' steps, into blocks no one else holds, and the sequence steps once it is
//! done (`ModelStep::swap_ready`). (A development build's GLYD_ENGINE_SWAP_HOLD=1: the next step waits for the whole copy out; usercheck compares the two.)
//!
//! The area is pinned as it is needed, in segments, up to what the plan gives it (`plan::swap_area`: twice the GPU cache it backs, within what the computer has free
//! after the expert store's host tier). A sequence the area has no room for is run again from its tokens, as before.

use super::pages::BLOCK;
use super::Dyn;
use crate::dev::{copy_d2h_async, copy_h2d_async, Ev, Pinned, Side};
use crate::ffi::{Error, Result, Stream};
use std::collections::{HashMap, VecDeque};

/// The bytes a segment of the area is pinned in (more where one sequence's copy is larger), and the alignment of each part of a copy in it (the copy kernel's 16-byte words).
const SEGMENT: usize = 256 << 20;
const ALIGN: usize = 256;

/// The free ranges of the area's segments, (segment, offset, bytes) in order: first fit, and a range given back joined to its neighbours.
#[derive(Default)]
struct Ranges(Vec<(usize, usize, usize)>);

impl Ranges {
    fn take(&mut self, len: usize) -> Option<(usize, usize)> {
        let i = self.0.iter().position(|r| r.2 >= len)?;
        let (seg, off, have) = self.0[i];
        if have == len {
            self.0.remove(i);
        } else {
            self.0[i] = (seg, off + len, have - len);
        }
        Some((seg, off))
    }

    fn give(&mut self, seg: usize, off: usize, len: usize) {
        let i = self.0.partition_point(|r| (r.0, r.1) < (seg, off));
        self.0.insert(i, (seg, off, len));
        if i + 1 < self.0.len() && self.0[i + 1].0 == seg && off + len == self.0[i + 1].1 {
            self.0[i].2 += self.0.remove(i + 1).2;
        }
        if i > 0 && self.0[i - 1].0 == seg && self.0[i - 1].1 + self.0[i - 1].2 == off {
            self.0[i - 1].2 += self.0.remove(i).2;
        }
    }
}

/// Where the parts of a sequence's copy are, as offsets from its start: each group's blocks (int32, the library's save and load read them), the copy kernel's items, each
/// ring's keys and values (its group's entries in order, `per` bytes a block each; a ring in 12-bit pages: its pages' images, `per` bytes each, and the bytes each took, uint32),
/// each state, and each state at the mark (none where there is none). `moved`: the bytes over the link one way (a 12-bit ring's pages' own: what their chunks and slabs add
/// is known once they are saved, `exact`); `fixed`: those of the bf16 rings and the states.
struct Layout {
    blocks: Vec<usize>,
    /// each group's blocks the sequence alone holds with no name (int32: their 12-bit pages are cleared after their part of the copy), the first part's first
    rel: Vec<usize>,
    items: usize,
    nitems: usize,
    rings: Vec<Option<(usize, usize, usize)>>,
    states: Vec<usize>,
    marks: Vec<usize>,
    total: usize,
    moved: usize,
    fixed: usize,
}

/// A sequence swapped out: its range in the area, its tokens, the table entries and floors it held (`Pages::held_entries`), its mark, its layout, and events before and after
/// the copy out.
struct Image {
    at: (usize, usize, usize),
    n: usize,
    entries: Vec<Vec<usize>>,
    floor: Vec<usize>,
    mark: usize,
    lay: Layout,
    began: Ev,
    out: Ev,
}

/// A copy back in flight: the slot, its range in the area, events before and after it, and its bytes.
struct Loading {
    slot: usize,
    at: (usize, usize, usize),
    began: Ev,
    done: Ev,
    moved: usize,
}

/// The swap area and its copies. (The stream is dropped first: its drop waits for the copies, which read and write the segments.)
pub(super) struct Swap {
    side: Side,
    segs: Vec<Pinned>,
    ranges: Ranges,
    /// The most bytes the segments may take (`plan::swap_area`).
    cap: usize,
    images: HashMap<u64, Image>,
    next: u64,
    loading: Vec<Loading>,
    /// The parts of the copies out not yet seen done, in the order they run: each one's fence (2 ticket, then 2 ticket + 1) and the event after it.
    outs: VecDeque<(u64, Ev)>,
    /// The slots swapped out whose states are not emptied yet (the copy's second part reads them), and that part's fence.
    unsettled: Vec<(usize, u64)>,
    /// Bytes copied either way and their seconds on the copy stream: the link's rate.
    bytes: f64,
    secs: f64,
    /// Bytes copied out, and back (`Dyn::swap_moved`).
    moved: (usize, usize),
}

impl Swap {
    pub(super) fn new(cap: usize) -> Result<Swap> {
        Ok(Swap {
            side: Side::new()?,
            segs: Vec::new(),
            ranges: Ranges::default(),
            cap,
            images: HashMap::new(),
            next: 0,
            loading: Vec::new(),
            outs: VecDeque::new(),
            unsettled: Vec::new(),
            bytes: 0.0,
            secs: 0.0,
            moved: (0, 0),
        })
    }

    /// Stream `s` made to wait for the part of a copy out whose fence is `f`, and so for the parts before it (one stream runs them in order), where it is not done yet.
    fn wait_fence(&mut self, f: u64, s: Stream) -> Result<()> {
        while let Some((_, ev)) = self.outs.front() {
            if !ev.done()? {
                break;
            }
            self.outs.pop_front();
        }
        match self.outs.iter().find(|(x, _)| *x == f) {
            Some((_, ev)) => ev.wait(s),
            None => Ok(()),
        }
    }

    /// The link's rate in bytes a second: measured on the copies so far, the planner's figure before any (`plan::LINK_GBS`).
    fn rate(&self) -> f64 {
        if self.secs > 0.0 {
            self.bytes / self.secs
        } else {
            crate::plan::LINK_GBS * 1e9
        }
    }

    /// A copy of `bytes` bytes from `began` to `done` counted in the rate (both done).
    fn count(&mut self, bytes: usize, began: &Ev, done: &Ev) {
        if let Ok(ms) = done.since(began) {
            if ms > 0.0 {
                (self.bytes, self.secs) = (self.bytes + bytes as f64, self.secs + ms as f64 * 1e-3);
            }
        }
    }

    /// A range of `len` bytes in the area: a free one, else a new segment where the plan's size and the computer's free memory leave room for it. None: no room.
    fn alloc(&mut self, len: usize) -> Option<(usize, usize, usize)> {
        let len = len.div_ceil(ALIGN) * ALIGN;
        if let Some((seg, off)) = self.ranges.take(len) {
            return Some((seg, off, len));
        }
        let made: usize = self.segs.iter().map(|s| s.bytes).sum();
        let size = SEGMENT.max(len).min(self.cap.saturating_sub(made));
        let free = crate::offload::host_memory().map_or(u64::MAX, |h| h.saturating_sub(crate::plan::HOST_MARGIN));
        if size < len || size as u64 > free {
            return None;
        }
        let seg = Pinned::mapped(size).ok()?;
        self.segs.push(seg);
        self.ranges.give(self.segs.len() - 1, 0, size);
        self.ranges.take(len).map(|(seg, off)| (seg, off, len))
    }

    /// The host address of `off` bytes into the range `at`.
    fn host(&self, at: (usize, usize, usize), off: usize) -> u64 {
        self.segs[at.0].ptr as u64 + (at.1 + off) as u64
    }

    /// The range `at`'s bytes as `T`s from `off` on, `n` of them, to write a copy's lists in.
    fn slice<T: crate::dev::Plain>(&mut self, at: (usize, usize, usize), off: usize, n: usize) -> &mut [T] {
        let all = self.segs[at.0].as_slice_mut::<u8>();
        let bytes = &mut all[at.1 + off..at.1 + off + n * std::mem::size_of::<T>()];
        assert!((bytes.as_ptr() as usize).is_multiple_of(std::mem::align_of::<T>()));
        // SAFETY: the bytes are aligned for T (the area's parts are ALIGN apart, its segments page-aligned) and hold n of them; T takes any bytes.
        unsafe { std::slice::from_raw_parts_mut(bytes.as_mut_ptr() as *mut T, n) }
    }

    /// The copy back into `slot` waited for and its range given back, where one is in flight (`release` of a slot being copied back).
    fn settle(&mut self, slot: usize) -> Result<()> {
        if let Some(i) = self.loading.iter().position(|l| l.slot == slot) {
            let l = self.loading.remove(i);
            l.done.wait_host()?;
            self.ranges.give(l.at.0, l.at.1, l.at.2);
        }
        Ok(())
    }

    /// Its stream waited for: every copy out and back issued has landed.
    pub(super) fn sync(&self) -> Result<()> {
        self.side.sync()
    }
}

impl Drop for Swap {
    fn drop(&mut self) {
        // (the copies in flight read and write the segments: done before the segments go)
        let _ = self.side.sync();
    }
}

impl Dyn {
    /// The swap area for these sequences, of `cap` bytes at most (`serve_rows`, once the pools are made); none where the keys and values are not paged.
    pub(super) fn swap_room(&mut self, cap: usize) -> Result<()> {
        self.swap = match self.paged.is_some() && cap > 0 {
            true => Some(Swap::new(cap)?),
            false => None,
        };
        Ok(())
    }

    /// The GPU cache the swap area backs: the pools of keys and values (bf16 and 12-bit pages) and every sequence's states.
    pub(super) fn swap_backs(&self) -> usize {
        let pages: usize = self.lossless.as_ref().map_or(0, |lx| lx.rings.iter().flatten().map(|r| r.pool.bytes).sum());
        let states: usize = self.states.iter().map(|b| b.bytes).sum();
        self.kc.bytes + self.vc.bytes + pages + states * self.seqs.len()
    }

    /// Whether a sequence of this engine can be swapped: paged keys and values (every ring of it, DeepSeek-V4's compressed entries among them), and an area.
    fn swaps(&self) -> bool {
        self.swap.is_some() && self.paged.is_some()
    }

    /// Where the parts of a copy of `counts[g]` blocks of each group go (`Layout`), with the states at a mark where `mark`.
    fn swap_layout(&self, counts: &[usize], mark: bool) -> Layout {
        let pg = self.paged.as_ref().expect("paged");
        let lx = self.lossless.as_ref();
        let twelve = |r: usize| lx.and_then(|x| x.rings[r].as_ref());
        let rings = &self.big.d.rings;
        let up = |x: usize| x.div_ceil(ALIGN) * ALIGN;
        let mut at = 0;
        let mut take = |bytes: usize| {
            let a = at;
            at += up(bytes);
            a
        };
        let blocks = counts.iter().map(|&c| take(4 * c)).collect();
        let rel = counts.iter().map(|&c| take(4 * c)).collect();
        let nitems: usize =
            (0..rings.len()).filter(|&r| self.ring_cap[r] > 0 && twelve(r).is_none()).map(|r| (1 + rings[r].values as usize) * counts[pg.group[r]]).sum();
        let items = take(24 * nitems);
        let (mut moved, mut fixed) = (0, 0);
        let rings = (0..rings.len())
            .map(|r| {
                if self.ring_cap[r] == 0 {
                    return None;
                }
                let c = counts[pg.group[r]];
                if let Some(x) = twelve(r) {
                    // (its pages' images, and the bytes each took)
                    moved += c * x.geom.page_stride as usize;
                    return Some((take(x.image * c), take(4 * c), x.image));
                }
                let per = BLOCK * rings[r].nk * rings[r].d * 2;
                // (a ring of keys alone: no values)
                let v = if rings[r].values { per * c } else { 0 };
                fixed += per * c + v;
                Some((take(per * c), take(v), per))
            })
            .collect();
        let states: Vec<usize> = self.states.iter().map(|b| take(b.bytes)).collect();
        let marks: Vec<usize> = if mark { self.states.iter().map(|b| take(b.bytes)).collect() } else { Vec::new() };
        fixed += self.states.iter().map(|b| b.bytes).sum::<usize>() * (1 + mark as usize);
        Layout { blocks, rel, items, nitems, rings, states, marks, total: at, moved: moved + fixed, fixed }
    }

    /// The states at sequence `slot`'s mark, where it has one: the tokens there (0: none).
    fn mark_of(&self, slot: usize) -> usize {
        if self.snaps.len() > slot {
            self.snap_n.get(slot).copied().unwrap_or(0)
        } else {
            0
        }
    }

    /// `ModelStep::swap_cost`: the bytes a swap of the sequence in `slot` moves one way, and the link's rate.
    pub(super) fn swap_cost_of(&self, slot: usize) -> Option<(usize, f64)> {
        let n = self.seq_n(slot);
        if !self.swaps() || n == 0 {
            return None;
        }
        let (entries, _) = self.paged.as_ref()?.pages.held_entries(slot, n);
        let counts: Vec<usize> = entries.iter().map(Vec::len).collect();
        Some((self.swap_layout(&counts, self.mark_of(slot) > 0).moved, self.swap.as_ref()?.rate()))
    }

    /// `ModelStep::swap_out`.
    pub(super) fn swap_out_of(&mut self, slot: usize) -> Result<Option<u64>> {
        let n = self.seq_n(slot);
        if !self.swaps() || n == 0 {
            return Ok(None);
        }
        let mut sw = self.swap.take().expect("swaps");
        let r = self.swap_out_with(&mut sw, slot, n);
        self.swap = Some(sw);
        if matches!(r, Ok(Some(_))) {
            // (what this step's growths promised it of the 12-bit pools goes with it)
            self.kv12_unpromise(slot);
        }
        r
    }

    fn swap_out_with(&mut self, sw: &mut Swap, slot: usize, n: usize) -> Result<Option<u64>> {
        let (entries, floor) = self.paged.as_ref().expect("paged").pages.held_entries(slot, n);
        let counts: Vec<usize> = entries.iter().map(Vec::len).collect();
        let mark = self.mark_of(slot);
        let lay = self.swap_layout(&counts, mark > 0);
        let Some(at) = sw.alloc(lay.total) else { return Ok(None) };
        // (the steps that wrote the sequence are done: the copy reads what they left, and its tables are written below)
        self.dev.sync()?;
        let s = sw.side.stream;
        let pg = self.paged.as_ref().expect("paged");
        for (g, e) in entries.iter().enumerate() {
            let tab = pg.pages.table(slot, g);
            let blocks: Vec<i32> = e.iter().map(|&b| tab[b] as i32).collect();
            sw.slice::<i32>(at, lay.blocks[g], blocks.len()).copy_from_slice(&blocks);
        }
        // (the first part: the first `head` blocks of each group, those a decode step of every sequence takes first)
        let head = self.seqs.len().max(1);
        // (12-bit pages: those of the blocks it alone holds with no name are cleared after their part is saved, so that their chunks and slabs go back to the pools
        // for the sequences that go on: R4b's deterministic capacity; by part, the first's first)
        let mut rel = vec![(0usize, 0usize); entries.len()];
        if self.lossless.is_some() {
            for (g, sole) in pg.pages.sole_unnamed(slot).into_iter().enumerate() {
                let (a, b): (Vec<_>, Vec<_>) = sole.into_iter().partition(|(e, _)| entries[g].binary_search(e).is_ok_and(|j| j < head));
                let list: Vec<i32> = a.iter().chain(&b).map(|x| x.1 as i32).collect();
                sw.slice::<i32>(at, lay.rel[g], list.len()).copy_from_slice(&list);
                rel[g] = (a.len(), b.len());
            }
        }
        let (items, first) = self.swap_items(sw, at, &lay, &entries, slot, true, head);
        sw.slice::<u64>(at, lay.items, items.len()).copy_from_slice(&items);
        let (began, mid, end, out) = (Ev::new()?, Ev::untimed()?, Ev::untimed()?, Ev::new()?);
        began.record(s)?;
        // SAFETY: the items list the blocks of the pools and the area's range for them (the layout's, `per` bytes each, 16-byte multiples and aligned); the range is the
        // area's (mapped, pinned), live until the image is given back, which waits for `out`.
        unsafe {
            for (part, (i0, i1)) in [(0, first), (first, lay.nitems)].into_iter().enumerate() {
                if i1 > i0 {
                    self.kern.swap_copy(s, sw.host(at, lay.items + 24 * i0), i1 - i0, ((i1 - i0) as u32).min(4 * self.dev.sms.max(1) as u32))?;
                }
                if let Some(lx) = &self.lossless {
                    for (r, rp) in lx.rings.iter().enumerate() {
                        let (Some(rp), Some((im, mv, per))) = (rp, lay.rings[r]) else { continue };
                        let (g, h) = (rp.group, head.min(counts[rp.group]));
                        let (j0, j1) = if part == 0 { (0, h) } else { (h, counts[g]) };
                        if j1 > j0 {
                            let (pages, out, moved) = (sw.host(at, lay.blocks[g] + 4 * j0), sw.host(at, im + j0 * per), sw.host(at, mv + 4 * j0));
                            self.lib.kv_save(&rp.geom, rp.pool.ptr, rp.blocks as i64, pages, (j1 - j0) as i64, rp.layer, out, moved, s)?;
                        }
                        let (r0, rn) = if part == 0 { (0, rel[g].0) } else { (rel[g].0, rel[g].1) };
                        if rn > 0 {
                            self.lib.kv_release(
                                &rp.geom,
                                rp.pool.ptr,
                                rp.blocks as i64,
                                sw.host(at, lay.rel[g] + 4 * r0),
                                rn as i64,
                                rp.layer,
                                lx.status[g].ptr,
                                s,
                            )?;
                        }
                    }
                }
                if part == 0 {
                    mid.record(s)?;
                }
            }
            let states = self.part_of(slot, 0, 0, 0)?.states;
            for (i, &src) in states.iter().enumerate() {
                copy_d2h_async(sw.host(at, lay.states[i]) as *mut _, src, self.states[i].bytes, s)?;
            }
            for (i, &off) in lay.marks.iter().enumerate() {
                copy_d2h_async(sw.host(at, off) as *mut _, self.snaps[slot][i].ptr, self.states[i].bytes, s)?;
            }
        }
        end.record(s)?;
        out.record(s)?;
        let ticket = sw.next + 1;
        let (f0, f1) = (2 * ticket, 2 * ticket + 1);
        if crate::tune!("GLYD_ENGINE_SWAP_HOLD").is_some_and(|v| v == "1") {
            // (the next step waits for the whole copy)
            end.wait(self.dev.stream)?;
        }
        sw.outs.extend([(f0, mid), (f1, end)]);
        // the slot: empty, its states filled when it is used again, after the copy (`empty`); its blocks given back with the part that reads them as their fence
        self.set_seq_n(slot, 0);
        if let Some(m) = self.snap_n.get_mut(slot) {
            *m = 0;
        }
        sw.unsettled.push((slot, f1));
        self.paged.as_mut().expect("paged").pages.release_fenced(slot, |g, b| match entries[g].binary_search(&b) {
            Ok(j) if j < head => f0,
            Ok(_) => f1,
            Err(_) => 0,
        });
        self.upload_tables(slot)?;
        sw.next = ticket;
        sw.images.insert(ticket, Image { at, n, entries, floor, mark, lay, began, out });
        Ok(Some(ticket))
    }

    /// The copy kernel's items for sequence `slot`'s bf16 rings: each block of `entries` (the slot's blocks now) to or from its place in the range `at`, keys then values;
    /// those of the first `head` entries of each group first, and how many those are.
    fn swap_items(&self, sw: &Swap, at: (usize, usize, usize), lay: &Layout, entries: &[Vec<usize>], slot: usize, out: bool, head: usize) -> (Vec<u64>, usize) {
        let (first, rest) =
            (self.swap_items_of(sw, at, lay, entries, slot, out, 0..head), self.swap_items_of(sw, at, lay, entries, slot, out, head..usize::MAX));
        let n = first.len() / 3;
        (first.into_iter().chain(rest).collect(), n)
    }

    /// `swap_items` of the entries `part` of each group (by their place in its list).
    #[allow(clippy::too_many_arguments)]
    fn swap_items_of(
        &self,
        sw: &Swap,
        at: (usize, usize, usize),
        lay: &Layout,
        entries: &[Vec<usize>],
        slot: usize,
        out: bool,
        part: std::ops::Range<usize>,
    ) -> Vec<u64> {
        let pg = self.paged.as_ref().expect("paged");
        let mut items = Vec::with_capacity(3 * lay.nitems);
        for (r, ring) in lay.rings.iter().enumerate() {
            let Some((k, v, per)) = *ring else { continue };
            if self.lossless.as_ref().is_some_and(|x| x.rings[r].is_some()) {
                continue;
            }
            let g = pg.group[r];
            let tab = pg.pages.table(slot, g);
            let (ko, vo) = self.ring_off[r];
            let values = self.big.d.rings[r].values.then_some((self.vc.ptr + vo, v));
            for (base, host) in std::iter::once((self.kc.ptr + ko, k)).chain(values) {
                // (an entry with no block, which a copy back's growth gave back, names block 0, that no step reads: its bytes go there, as many items as the layout counted)
                for (j, &b) in entries[g].iter().enumerate().take(part.end).skip(part.start) {
                    let dev = base + (tab[b] as usize * per) as u64;
                    let h = sw.host(at, host + j * per);
                    items.extend(if out { [dev, h, per as u64] } else { [h, dev, per as u64] });
                }
            }
        }
        items
    }

    /// `ModelStep::swap_in`: the copy of `ticket` back into the empty slot `slot`, with room for its first `upto` tokens; None where the pools have no room now.
    pub(super) fn swap_in_of(&mut self, slot: usize, ticket: u64, upto: usize) -> Result<Option<usize>> {
        let Some(mut sw) = self.swap.take() else { return Err(Error(format!("ticket {ticket}: this engine swaps nothing"))) };
        let r = self.swap_in_with(&mut sw, slot, ticket, upto);
        self.swap = Some(sw);
        r
    }

    fn swap_in_with(&mut self, sw: &mut Swap, slot: usize, ticket: u64, upto: usize) -> Result<Option<usize>> {
        let Some(img) = sw.images.get(&ticket) else { return Err(Error(format!("ticket {ticket}: no sequence is swapped by it"))) };
        if self.seq_n(slot) != 0 {
            return Err(Error(format!("a sequence copied back into slot {slot}, which holds {} tokens", self.seq_n(slot))));
        }
        // (its copy out wrote the images read below and the lists this one writes over: done first)
        img.out.wait_host()?;
        // (12-bit pages: a slab for each raw page and a chunk for each a page holds, beside what the step's growths have promised, the pools read with nothing in flight
        // on this stream (`pages::Ledger`); else not now)
        let (at, rings, entries) = (img.at, img.lay.rings.clone(), img.entries.clone());
        let load = self.load_need(sw, at, &rings, &entries);
        let Some(img) = sw.images.get(&ticket) else { return Err(Error(format!("ticket {ticket}: no sequence is swapped by it"))) };
        if let Some((slabs, chunks)) = load.as_ref() {
            let mut ledger = std::mem::take(&mut self.lossless.as_mut().expect("12-bit pages").ledger);
            let fits = ledger.copy_back(&mut super::kvpages::Gpu(self, Some(&*sw)), slot, slabs, *chunks);
            self.lossless.as_mut().expect("12-bit pages").ledger = ledger;
            if !fits? {
                return Ok(None);
            }
        }
        let pg = self.paged.as_mut().expect("paged");
        if !pg.pages.restore(slot, &img.entries, &img.floor) {
            return Ok(None);
        }
        // (its room on to `upto` as `ModelStep::grow` takes a prompt's: from what its next pass reads. `upto` is the whole prompt's where it is still read in passes, and
        // the blocks of `upto`'s own window would give back the window before its `n` tokens: the entries copied out, whose pages the copy back writes)
        if !pg.pages.grow_from(slot, img.n, upto.max(img.n), self.chunk) {
            pg.pages.release(slot);
            return Ok(None);
        }
        // (blocks a copy out still reads: this copy runs after it on the same stream, and the engine's steps write them only after this one, `swap_ready`)
        pg.pages.take_fence();
        // (and the slot's states, where it was swapped out itself and not used since: this copy writes them, after that copy)
        sw.unsettled.retain(|u| u.0 != slot);
        let img = sw.images.remove(&ticket).expect("found above");
        let moved = Self::exact(self.lossless.as_ref(), sw, img.at, &img.lay, &img.entries);
        sw.count(moved, &img.began, &img.out);
        sw.moved.0 += moved;
        let (at, lay) = (img.at, &img.lay);
        let pg = self.paged.as_ref().expect("paged");
        for (g, e) in img.entries.iter().enumerate() {
            let tab = pg.pages.table(slot, g);
            // (an entry the growth above gave back, which no pass reads again: no block, and its page is not loaded. Block 0 is every such entry's: pages loaded
            // into it together each gave its slab back, once in the pool's bitmap and once each in its count, and the pool then had fewer slabs than its count said)
            let blocks: Vec<i32> = e.iter().map(|&b| if tab[b] == 0 { -1 } else { tab[b] as i32 }).collect();
            sw.slice::<i32>(at, lay.blocks[g], blocks.len()).copy_from_slice(&blocks);
        }
        let (items, _) = self.swap_items(sw, at, lay, &img.entries, slot, false, usize::MAX);
        sw.slice::<u64>(at, lay.items, items.len()).copy_from_slice(&items);
        let s = sw.side.stream;
        // (the engine's stream may still empty this slot's states, or step others: what it queued before is done first)
        let after = Ev::untimed()?;
        after.record(self.dev.stream)?;
        after.wait(s)?;
        let (began, done) = (Ev::new()?, Ev::new()?);
        began.record(s)?;
        // SAFETY: as `swap_out_with`, the other way; the blocks are the slot's own (`restore`), which no step reads or writes until `swap_ready`.
        unsafe {
            if lay.nitems > 0 {
                self.kern.swap_copy(s, sw.host(at, lay.items), lay.nitems, (lay.nitems as u32).min(4 * self.dev.sms.max(1) as u32))?;
            }
            if let Some(lx) = &self.lossless {
                for (r, rp) in lx.rings.iter().enumerate() {
                    let (Some(rp), Some((im, _, _))) = (rp, lay.rings[r]) else { continue };
                    let (g, n) = (rp.group, img.entries[rp.group].len());
                    if n > 0 {
                        let (images, dst) = (sw.host(at, im), sw.host(at, lay.blocks[g]));
                        self.lib.kv_load(
                            &rp.geom,
                            images,
                            dst,
                            n as i64,
                            rp.pool.ptr,
                            rp.blocks as i64,
                            rp.layer,
                            lx.status[g].ptr,
                            super::kvpages::RING,
                            s,
                        )?;
                    }
                }
            }
            let states = self.part_of(slot, 0, 0, 0)?.states;
            for (i, &dst) in states.iter().enumerate() {
                copy_h2d_async(dst, sw.host(at, lay.states[i]) as *const _, self.states[i].bytes, s)?;
            }
            for (i, &off) in lay.marks.iter().enumerate() {
                copy_h2d_async(self.snaps[slot][i].ptr, sw.host(at, off) as *const _, self.states[i].bytes, s)?;
            }
        }
        done.record(s)?;
        self.set_seq_n(slot, img.n);
        if self.snap_n.len() > slot {
            self.snap_n[slot] = img.mark;
        }
        self.upload_tables(slot)?;
        sw.moved.1 += moved;
        sw.loading.push(Loading { slot, at, began, done, moved });
        Ok(Some(img.n))
    }

    /// `ModelStep::swap_ready`: whether the copy back into `slot` is done (with `wait`, once it is); the engine's stream then waits for it, and its range is given back.
    pub(super) fn swap_ready_of(&mut self, slot: usize, wait: bool) -> Result<bool> {
        let Some(sw) = self.swap.as_mut() else { return Ok(true) };
        let Some(i) = sw.loading.iter().position(|l| l.slot == slot) else { return Ok(true) };
        if wait {
            sw.loading[i].done.wait_host()?;
        } else if !sw.loading[i].done.done()? {
            return Ok(false);
        }
        let l = sw.loading.remove(i);
        l.done.wait(self.dev.stream)?;
        sw.count(l.moved, &l.began, &l.done);
        sw.ranges.give(l.at.0, l.at.1, l.at.2);
        Ok(true)
    }

    /// `ModelStep::swap_drop`: the copy of `ticket` let go (its copy out finished first).
    pub(super) fn swap_drop_of(&mut self, ticket: u64) {
        let Some(sw) = self.swap.as_mut() else { return };
        if let Some(img) = sw.images.remove(&ticket) {
            let _ = img.out.wait_host();
            sw.moved.0 += Self::exact(self.lossless.as_ref(), sw, img.at, &img.lay, &img.entries);
            sw.ranges.give(img.at.0, img.at.1, img.at.2);
        }
    }

    /// The bytes a copy out moved one way, once it is done: the bf16 rings' and the states', and what each 12-bit page's image took (its bytes, its chunks, its slab).
    fn exact(lx: Option<&super::kvpages::Lossless>, sw: &mut Swap, at: (usize, usize, usize), lay: &Layout, entries: &[Vec<usize>]) -> usize {
        let mut m = lay.fixed;
        for (r, rp) in lx.iter().flat_map(|x| x.rings.iter().enumerate()) {
            if let (Some(rp), Some((_, mv, _))) = (rp, lay.rings[r]) {
                m += sw.slice::<u32>(at, mv, entries[rp.group].len()).iter().map(|&b| b as usize).sum::<usize>();
            }
        }
        m
    }

    /// A slot about to be emptied: its copy back, where one is in flight, finished first.
    pub(super) fn swap_settle(&mut self, slot: usize) -> Result<()> {
        if let Some(sw) = self.swap.as_mut() {
            sw.settle(slot)?;
        }
        Ok(())
    }

    /// The slabs (of each raw pool) and the chunks a copy back's 12-bit pages take as they load, read from their images: a slab for each raw page, a chunk for each
    /// its directory names. None: no 12-bit ring.
    fn load_need(&self, sw: &mut Swap, at: (usize, usize, usize), rings: &[Option<(usize, usize, usize)>], entries: &[Vec<usize>]) -> Option<(Vec<i64>, i64)> {
        let lx = self.lossless.as_ref()?;
        let (mut slabs, mut chunks) = (vec![0i64; lx.raw_pools.len()], 0i64);
        for (r, rp) in lx.rings.iter().enumerate() {
            let (Some(rp), Some((im, _, per)), Some(c)) = (rp, rings[r], lx.ring_raw[r]) else { continue };
            let hp = rp.shape.heads * rp.shape.channels / 128;
            for j in 0..entries[rp.group].len() {
                let w = sw.slice::<u32>(at, im + j * per + 16 * hp, 17);
                if w[0] & (1 << 30) != 0 {
                    continue;
                }
                if w[0] & (1 << 31) != 0 {
                    slabs[c] += 1;
                } else {
                    chunks += w[1..].iter().filter(|&&d| d != 0).count() as i64;
                }
            }
        }
        Some((slabs, chunks))
    }

    /// The engine's stream made to wait for the copies out that still read the blocks taken since the last call (`Pages::take_fence`): before the step that writes them.
    pub(super) fn fence_wait(&mut self) -> Result<()> {
        let Some(pg) = self.paged.as_mut() else { return Ok(()) };
        let f = pg.pages.take_fence();
        match self.swap.as_mut() {
            Some(sw) if f > 0 => sw.wait_fence(f, self.dev.stream),
            _ => Ok(()),
        }
    }

    /// The engine's stream made to wait for the copy out of slot `slot`'s states, where it was swapped out and its states are not emptied since (`empty` fills them
    /// after it); whether it was.
    pub(super) fn unsettled_wait(&mut self, slot: usize) -> Result<bool> {
        let Some(sw) = self.swap.as_mut() else { return Ok(false) };
        let Some(i) = sw.unsettled.iter().position(|u| u.0 == slot) else { return Ok(false) };
        let (_, f) = sw.unsettled.remove(i);
        sw.wait_fence(f, self.dev.stream)?;
        Ok(true)
    }

    /// A slot swapped out and used again (a prompt, a start found or shared): emptied first, after the copy out read its states.
    pub(super) fn settle_slot(&mut self, slot: usize) -> Result<()> {
        if self.swap.as_ref().is_some_and(|sw| sw.unsettled.iter().any(|u| u.0 == slot)) {
            self.empty(slot)?;
        }
        Ok(())
    }

    /// The swap's copies done (its stream waited for): what they give back to the pools is counted.
    pub(super) fn swap_sync(&self) -> Result<()> {
        self.swap.as_ref().map_or(Ok(()), Swap::sync)
    }

    /// The bytes swapped out and back so far.
    #[cfg(feature = "dev")]
    pub(crate) fn swap_moved(&self) -> (usize, usize) {
        self.swap.as_ref().map_or((0, 0), |sw| sw.moved)
    }
}

#[cfg(test)]
mod tests {
    use super::Ranges;

    /// The area's ranges: first fit, a range given back joined to its free neighbours in its segment (never across segments), and every byte had again.
    #[test]
    fn the_areas_ranges_are_taken_first_fit_and_joined_when_given_back() {
        let mut r = Ranges::default();
        r.give(0, 0, 1000);
        r.give(1, 0, 500);
        let a = r.take(300).unwrap();
        let b = r.take(300).unwrap();
        let c = r.take(400).unwrap();
        assert_eq!((a, b, c), ((0, 0), (0, 300), (0, 600)));
        assert_eq!(r.take(600), None);
        assert_eq!(r.take(500), Some((1, 0)));
        r.give(0, 300, 300);
        r.give(0, 0, 300);
        assert_eq!(r.0, vec![(0, 0, 600)]);
        r.give(1, 0, 500);
        r.give(0, 600, 400);
        assert_eq!(r.0, vec![(0, 0, 1000), (1, 0, 500)], "joined within a segment, not across");
        assert_eq!(r.take(1000), Some((0, 0)));
        assert_eq!(r.0, vec![(1, 0, 500)]);
    }
}
