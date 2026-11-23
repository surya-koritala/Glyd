//! Preemption by swapping (R4d #328; docs/design/runtime.md section 5, "Preemption"): a sequence the scheduler sends back for room has its cache copied to the host's swap
//! area and back, rather than being run again from its tokens (`ModelStep::swap_out`, `swap_in`). What is copied is all a step of it reads, for every family the same way:
//! in each layer group the blocks of its tokens its table holds (a sliding group's from its window on, a shared start's from its floor), each ring's keys and values of
//! them (a ring of keys alone, a latent cache's, its keys); its recurrent states (the delta rule's and the convolutions', Mamba's); and their copy at its mark where it has one. A ring in bf16 pages is copied as its bytes.
//! A ring in 12-bit pages is copied as the bf16 values the library unpacks from them, its escape chunks and raw slabs read with them, and written back into new pages by
//! the library's own write, which takes new chunks and slabs where they need them: the values come back bit for bit, but the link carries 16 bits a value, not 12. (A
//! page moved as its own bytes names its chunks of the escape pool by number, lent to its old block by owner: the library has no call that lends them to another.)
//!
//! The copies run on a stream of their own, their host end pinned memory mapped into the device's addresses: one launch of `swap_copy` moves every bf16 block however
//! scattered its pools are, and the library's unpack and write read and write the host's memory themselves, with no staging buffer on the GPU, whose memory is what ran
//! out. Out: the engine's stream waits for the copy before its next step, which may write the blocks given back. Back: the copy runs beside the other sequences' steps,
//! into blocks no one else holds, and the sequence steps once it is done (`ModelStep::swap_ready`).
//!
//! The area is pinned as it is needed, in segments, up to what the plan gives it (`plan::swap_area`: twice the GPU cache it backs, within what the computer has free
//! after the expert store's host tier). A sequence the area has no room for is run again from its tokens, as before.

use super::pages::BLOCK;
use super::Dyn;
use crate::dev::{copy_d2h_async, copy_h2d_async, Ev, Pinned, Side};
use crate::ffi::{Error, Result};
use std::collections::HashMap;

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

/// Where the parts of a sequence's copy are, as offsets from its start: each group's blocks (int32, the library's unpack and write read them) and its tokens' slots (int64, the
/// library's write; groups with a ring in 12-bit pages), the copy kernel's items, each ring's keys and values (its group's entries in order, `per` bytes a block each), each
/// state, and each state at the mark (none where there is none). `moved`: the bytes over the link one way.
struct Layout {
    blocks: Vec<usize>,
    slots: Vec<Option<usize>>,
    items: usize,
    nitems: usize,
    rings: Vec<Option<(usize, usize, usize)>>,
    states: Vec<usize>,
    marks: Vec<usize>,
    total: usize,
    moved: usize,
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
    /// Bytes copied either way and their seconds on the copy stream: the link's rate.
    bytes: f64,
    secs: f64,
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
            bytes: 0.0,
            secs: 0.0,
        })
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

    /// Whether a sequence of this engine can be swapped: paged keys and values, an area, and nothing of a sequence's that a swap does not copy (DeepSeek-V4's compressed
    /// entries).
    fn swaps(&self) -> bool {
        self.swap.is_some() && self.paged.is_some() && self.v4.is_none()
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
        let slots = (0..counts.len()).map(|g| (0..rings.len()).any(|r| twelve(r).is_some_and(|x| x.group == g)).then(|| take(8 * BLOCK * counts[g]))).collect();
        let nitems: usize =
            (0..rings.len()).filter(|&r| self.ring_cap[r] > 0 && twelve(r).is_none()).map(|r| (1 + rings[r].values as usize) * counts[pg.group[r]]).sum();
        let items = take(24 * nitems);
        let mut moved = 0;
        let rings = (0..rings.len())
            .map(|r| {
                if self.ring_cap[r] == 0 {
                    return None;
                }
                let per = match twelve(r) {
                    Some(x) => BLOCK * x.shape.heads * x.shape.channels * 2,
                    None => BLOCK * rings[r].nk * rings[r].d * 2,
                };
                let c = counts[pg.group[r]];
                // (a ring of keys alone: no values)
                let v = if rings[r].values || twelve(r).is_some() { per * c } else { 0 };
                moved += per * c + v;
                Some((take(per * c), take(v), per))
            })
            .collect();
        let states: Vec<usize> = self.states.iter().map(|b| take(b.bytes)).collect();
        let marks: Vec<usize> = if mark { self.states.iter().map(|b| take(b.bytes)).collect() } else { Vec::new() };
        moved += self.states.iter().map(|b| b.bytes).sum::<usize>() * (1 + mark as usize);
        Layout { blocks, slots, items, nitems, rings, states, marks, total: at, moved }
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
        r
    }

    fn swap_out_with(&mut self, sw: &mut Swap, slot: usize, n: usize) -> Result<Option<u64>> {
        let (entries, floor) = self.paged.as_ref().expect("paged").pages.held_entries(slot, n);
        let counts: Vec<usize> = entries.iter().map(Vec::len).collect();
        let mark = self.mark_of(slot);
        let lay = self.swap_layout(&counts, mark > 0);
        let Some(at) = sw.alloc(lay.total) else { return Ok(None) };
        // (the steps that wrote the sequence are done: the copy reads what they left)
        self.dev.sync()?;
        let s = sw.side.stream;
        let pg = self.paged.as_ref().expect("paged");
        for (g, e) in entries.iter().enumerate() {
            let tab = pg.pages.table(slot, g);
            let blocks: Vec<i32> = e.iter().map(|&b| tab[b] as i32).collect();
            sw.slice::<i32>(at, lay.blocks[g], blocks.len()).copy_from_slice(&blocks);
        }
        let items = self.swap_items(sw, at, &lay, &entries, slot, true);
        sw.slice::<u64>(at, lay.items, items.len()).copy_from_slice(&items);
        let (began, out) = (Ev::new()?, Ev::new()?);
        began.record(s)?;
        // SAFETY: the items list the blocks of the pools and the area's range for them (the layout's, `per` bytes each, 16-byte multiples and aligned); the range is the
        // area's (mapped, pinned), live until the image is given back, which waits for `out`.
        unsafe {
            if lay.nitems > 0 {
                self.kern.swap_copy(s, sw.host(at, lay.items), lay.nitems, (lay.nitems as u32).min(4 * self.dev.sms.max(1) as u32))?;
            }
            if let Some(lx) = &self.lossless {
                for (r, rp) in lx.rings.iter().enumerate() {
                    let (Some(rp), Some((k, v, _))) = (rp, lay.rings[r]) else { continue };
                    let g = rp.group;
                    if counts[g] > 0 {
                        self.lib.kv_unpack(
                            &rp.geom,
                            rp.pool.ptr,
                            rp.kbase.ptr,
                            rp.vbase.ptr,
                            sw.host(at, lay.blocks[g]),
                            counts[g] as i64,
                            sw.host(at, k),
                            sw.host(at, v),
                            s,
                        )?;
                    }
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
        out.record(s)?;
        // (the next step may write the blocks given back, and the slot's states: after the copy)
        out.wait(self.dev.stream)?;
        self.empty(slot)?;
        self.paged.as_mut().expect("paged").pages.release(slot);
        self.upload_tables(slot)?;
        sw.next += 1;
        sw.images.insert(sw.next, Image { at, n, entries, floor, mark, lay, began, out });
        Ok(Some(sw.next))
    }

    /// The copy kernel's items for sequence `slot`'s bf16 rings: each block of `entries` (the slot's blocks now) to or from its place in the range `at`, keys then values.
    fn swap_items(&self, sw: &Swap, at: (usize, usize, usize), lay: &Layout, entries: &[Vec<usize>], slot: usize, out: bool) -> Vec<u64> {
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
                for (j, &b) in entries[g].iter().enumerate() {
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
        let pg = self.paged.as_mut().expect("paged");
        if !pg.pages.restore(slot, &img.entries, &img.floor) {
            return Ok(None);
        }
        if !pg.pages.grow(slot, upto.max(img.n)) {
            pg.pages.release(slot);
            return Ok(None);
        }
        let img = sw.images.remove(&ticket).expect("found above");
        // (its copy out read the lists this one writes over: done first, and counted)
        img.out.wait_host()?;
        sw.count(img.lay.moved, &img.began, &img.out);
        let (at, lay) = (img.at, &img.lay);
        let pg = self.paged.as_ref().expect("paged");
        for (g, e) in img.entries.iter().enumerate() {
            let tab = pg.pages.table(slot, g);
            let blocks: Vec<i32> = e.iter().map(|&b| tab[b] as i32).collect();
            sw.slice::<i32>(at, lay.blocks[g], blocks.len()).copy_from_slice(&blocks);
            if let Some(off) = lay.slots[g] {
                // (a 12-bit page's tokens past the sequence's last are not written: its page holds what they were, anything)
                let slots: Vec<i64> = e
                    .iter()
                    .flat_map(|&b| (0..BLOCK).map(move |t| (b * BLOCK + t, tab[b] as i64 * BLOCK as i64 + t as i64)))
                    .map(|(p, s)| if p < img.n { s } else { -1 })
                    .collect();
                sw.slice::<i64>(at, off, slots.len()).copy_from_slice(&slots);
            }
        }
        let items = self.swap_items(sw, at, lay, &img.entries, slot, false);
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
                    let (Some(rp), Some((k, v, _))) = (rp, lay.rings[r]) else { continue };
                    let g = rp.group;
                    let (n, Some(slots)) = (img.entries[g].len(), lay.slots[g]) else { continue };
                    if n > 0 {
                        let w = (rp.shape.heads * rp.shape.channels) as i64;
                        let ring = super::kvpages::RING;
                        let (kh, vh, sl) = (sw.host(at, k), sw.host(at, v), sw.host(at, slots));
                        self.lib.kv_write(
                            &rp.geom,
                            rp.pool.ptr,
                            rp.kbase.ptr,
                            rp.vbase.ptr,
                            rp.layer,
                            kh,
                            w,
                            vh,
                            w,
                            sl,
                            (n * BLOCK) as i64,
                            lx.status[g].ptr,
                            0,
                            0,
                            ring,
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
        sw.loading.push(Loading { slot, at, began, done, moved: lay.moved });
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
            sw.ranges.give(img.at.0, img.at.1, img.at.2);
        }
    }

    /// A slot about to be emptied: its copy back, where one is in flight, finished first.
    pub(super) fn swap_settle(&mut self, slot: usize) -> Result<()> {
        match self.swap.as_mut() {
            Some(sw) => sw.settle(slot),
            None => Ok(()),
        }
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
