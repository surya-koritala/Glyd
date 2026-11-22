//! Paged keys and values for the description executor (R4b, #141; docs/design/runtime.md "Paged blocks for every attention kind"): a sequence's cache is the blocks it holds, not a room of
//! the whole context. A block is BLOCK tokens. Layers of one kind (a layer group: the sliding layers of a window, or the full ones) share one pool of blocks and one block table a sequence:
//! the table's entry b is the pool's block that holds the sequence's tokens b BLOCK .. b BLOCK + BLOCK - 1, in every layer of the group (each ring has its own bytes of the block). A
//! sliding group keeps only the blocks a step can still read, the window and a chunk behind the newest token; older ones go back to the pool.
//!
//! A block may be held by more than one sequence (`share`: a prompt that begins as another sequence does names that sequence's full blocks and runs only the rest). A block
//! counts its holders and goes back to its pool when the last gives it up. A shared block is never written: a sequence writes from its own length on, a shared block is full
//! and below the lengths of all who hold it, and a sequence cut back (`shrink`) into a block that others hold is cut to that block's start instead.
//!
//! Block 0 of every pool is never given out: every entry a sequence does not hold names it, so a write the engine makes there (a graph's capture runs a step at position 0 of a sequence
//! that holds nothing) lands where no one reads. The kernels read a position through the table (`g4_place` in kernels/gemma4.cu); this file is the host's side: which blocks a sequence
//! holds, and which entries changed since the tables were last copied to the GPU.

/// Tokens a block.
pub const BLOCK: usize = 16;

/// One layer group's pool.
struct Pool {
    /// Tokens behind the newest a step may read (the window and a chunk); None: every token (full attention).
    keep: Option<usize>,
    /// Blocks not held, the lowest last (so the first given out are the lowest).
    free: Vec<u32>,
    blocks: u32,
    /// The sequences that hold each block (their table entries that name it).
    refs: Vec<u32>,
}

impl Pool {
    /// A block for one holder.
    fn take(&mut self) -> Option<u32> {
        let b = self.free.pop()?;
        self.refs[b as usize] = 1;
        Some(b)
    }
    /// One holder gives block `b` up: free again when it was the last.
    fn give(&mut self, b: u32) {
        let r = &mut self.refs[b as usize];
        debug_assert!(*r > 0, "block {b} given up twice");
        *r = r.saturating_sub(1);
        if *r == 0 {
            self.free.push(b);
        }
    }
}

/// The blocks every sequence holds in every group.
pub struct Pages {
    pools: Vec<Pool>,
    /// [slot][group]: the block of each of the sequence's table entries (0: none).
    tabs: Vec<Vec<Vec<u32>>>,
    /// [slot][group]: the entries changed since `take_dirty`, as a range.
    dirty: Vec<Vec<Option<(usize, usize)>>>,
    /// [slot][group]: the first entry the sequence ever held. 0 for one that began at its first token; a sequence that shared a start holds a sliding group's blocks from
    /// the window before that start on, and the entries before were never its own: `grow` gives it no block there (an empty block is not the tokens' keys and values, and a
    /// later cut back to before the start's end would read it as if it were).
    floor: Vec<Vec<usize>>,
}

impl Pages {
    /// Pools of `blocks[g]` blocks (block 0 among them) for the groups, each keeping `keep[g]` tokens behind (None: all), for `slots` sequences of at most `ctx` tokens.
    pub fn new(blocks: &[u32], keep: &[Option<usize>], slots: usize, ctx: usize) -> Pages {
        let pools = blocks
            .iter()
            .zip(keep)
            .map(|(&b, &k)| Pool { keep: k, free: (1..b.max(1)).rev().collect(), blocks: b.max(1), refs: vec![0; b.max(1) as usize] })
            .collect::<Vec<_>>();
        let n = ctx.div_ceil(BLOCK);
        let tabs = (0..slots).map(|_| (0..pools.len()).map(|_| vec![0u32; n]).collect()).collect();
        let dirty = (0..slots).map(|_| vec![None; pools.len()]).collect();
        let floor = vec![vec![0; pools.len()]; slots];
        Pages { pools, tabs, dirty, floor }
    }

    pub fn groups(&self) -> usize {
        self.pools.len()
    }

    /// Blocks free in group `g`, and all of its blocks but block 0.
    pub fn free(&self, g: usize) -> (usize, usize) {
        (self.pools[g].free.len(), self.pools[g].blocks as usize - 1)
    }

    /// The most tokens one sequence can hold with nothing else holding blocks: a full group's blocks (but block 0) a BLOCK each; a sliding group no limit where it holds a window and a chunk
    /// (and the blocks they straddle), else what its blocks hold less one. None: no group limits it.
    pub fn capacity(&self) -> Option<usize> {
        self.pools
            .iter()
            .filter_map(|p| {
                let usable = p.blocks as usize - 1;
                match p.keep {
                    None => Some(usable * BLOCK),
                    Some(k) if usable >= k.div_ceil(BLOCK) + 1 => None,
                    Some(_) => Some(usable.saturating_sub(1) * BLOCK),
                }
            })
            .min()
    }

    /// The table entries sequence `slot` needs to hold its first `upto` tokens in group `g`: from the first a step may still read to the last.
    fn span(&self, g: usize, upto: usize) -> (usize, usize) {
        let hi = upto.div_ceil(BLOCK);
        let lo = match self.pools[g].keep {
            Some(k) => upto.saturating_sub(k) / BLOCK,
            None => 0,
        };
        (lo.min(hi), hi)
    }

    /// `span`, and within it the first entry the sequence is given a block for: none before the first it ever held (`floor`).
    fn span_of(&self, slot: usize, g: usize, upto: usize) -> (usize, usize, usize) {
        let (lo, hi) = self.span(g, upto);
        (lo, lo.max(self.floor[slot][g]).min(hi), hi)
    }

    /// Sequence `slot` given the blocks for its first `upto` tokens in every group, and its sliding groups' blocks a step can no longer read given back; false where some group has too
    /// few free (then nothing changes).
    pub fn grow(&mut self, slot: usize, upto: usize) -> bool {
        let max = self.tabs[slot][0].len();
        if upto.div_ceil(BLOCK) > max {
            return false;
        }
        for g in 0..self.pools.len() {
            let (lo, from, hi) = self.span_of(slot, g, upto);
            let need = (from..hi).filter(|&b| self.tabs[slot][g][b] == 0).count();
            // (blocks this call gives back come first: those it alone holds)
            let back = (0..lo).filter(|&b| self.tabs[slot][g][b] != 0 && self.pools[g].refs[self.tabs[slot][g][b] as usize] == 1).count();
            if need > self.pools[g].free.len() + back {
                return false;
            }
        }
        for g in 0..self.pools.len() {
            let (lo, from, hi) = self.span_of(slot, g, upto);
            for b in 0..lo {
                let blk = std::mem::take(&mut self.tabs[slot][g][b]);
                if blk != 0 {
                    self.pools[g].give(blk);
                    self.mark(slot, g, b);
                }
            }
            for b in from..hi {
                if self.tabs[slot][g][b] == 0 {
                    self.tabs[slot][g][b] = self.pools[g].take().expect("counted above");
                    self.mark(slot, g, b);
                }
            }
        }
        true
    }

    /// Every block of sequence `slot` back in its pool.
    pub fn release(&mut self, slot: usize) {
        for g in 0..self.pools.len() {
            self.floor[slot][g] = 0;
            for b in 0..self.tabs[slot][g].len() {
                let blk = std::mem::take(&mut self.tabs[slot][g][b]);
                if blk != 0 {
                    self.pools[g].give(blk);
                    self.mark(slot, g, b);
                }
            }
        }
    }

    /// The first tokens of a sequence a step at `upto` reads in group `g`, as a table entry: a sliding group's window behind `upto` (what `keep` holds less `slack`, a chunk),
    /// a full group's first.
    fn reach(&self, g: usize, upto: usize, slack: usize) -> usize {
        match self.pools[g].keep {
            Some(k) => upto.saturating_sub(k.saturating_sub(slack)) / BLOCK,
            None => 0,
        }
    }

    /// What sequence `slot` would hold cut back to go on within its first `upto` tokens (`shrink`, which makes the cut): `upto`, or the start of the block `upto` is in
    /// where others hold that block too (it would be written from `upto` on). 0 where a group no longer holds what a step there reads: a sliding group gives up the
    /// blocks behind its window and `slack` tokens (a chunk: what `keep` holds beyond the window) as the sequence grows, so one that grew more than `slack` past the place
    /// cannot be cut there.
    pub fn cut_len(&self, slot: usize, upto: usize, slack: usize) -> usize {
        let part = upto / BLOCK;
        let shared = !upto.is_multiple_of(BLOCK) && (0..self.pools.len()).any(|g| self.pools[g].refs[self.tabs[slot][g][part] as usize] > 1);
        let upto = if shared { part * BLOCK } else { upto };
        let hi = upto.div_ceil(BLOCK);
        if (0..self.pools.len()).any(|g| (self.reach(g, upto, slack).min(hi)..hi).any(|b| self.tabs[slot][g][b] == 0)) {
            return 0;
        }
        upto
    }

    /// Sequence `slot` cut back to go on within its first `upto` tokens, as one that had grown no further: the tokens it then holds (`cut_len`), and the blocks past
    /// them given up. 0, and nothing changes, where it cannot be cut there.
    pub fn shrink(&mut self, slot: usize, upto: usize, slack: usize) -> usize {
        let upto = self.cut_len(slot, upto, slack);
        if upto == 0 {
            return 0;
        }
        let hi = upto.div_ceil(BLOCK);
        for g in 0..self.pools.len() {
            for b in hi..self.tabs[slot][g].len() {
                let blk = std::mem::take(&mut self.tabs[slot][g][b]);
                if blk != 0 {
                    self.pools[g].give(blk);
                    self.mark(slot, g, b);
                }
            }
        }
        upto
    }

    /// Whether sequence `slot` holds every block a step at `upto` reads of its first `upto` tokens (a sliding group's from its window on; `slack` as `shrink`'s).
    pub fn holds(&self, slot: usize, upto: usize, slack: usize) -> bool {
        let hi = upto.div_ceil(BLOCK);
        (0..self.pools.len()).all(|g| (self.reach(g, upto, slack).min(hi)..hi).all(|b| self.tabs[slot][g][b] != 0))
    }

    /// The full blocks of the first `upto` tokens of sequence `from` named by sequence `to` as well, which must hold nothing: the tokens `to` then holds (a whole number of
    /// blocks), each of those blocks with one holder more. 0, and nothing changes, where `upto` is less than a block or `from` no longer holds what a step there reads (a
    /// sliding group's blocks behind its window; `slack` as `shrink`'s). Neither writes what the other reads: `to` goes on from the end of those blocks, `from` from its own
    /// length, which is past them.
    pub fn share(&mut self, from: usize, to: usize, upto: usize, slack: usize) -> usize {
        let nb = upto / BLOCK;
        if nb == 0 || from == to || self.tabs[to].iter().any(|t| t.iter().any(|&b| b != 0)) {
            return 0;
        }
        for g in 0..self.pools.len() {
            if (self.reach(g, nb * BLOCK, slack).min(nb)..nb).any(|b| self.tabs[from][g][b] == 0) {
                return 0;
            }
        }
        for g in 0..self.pools.len() {
            self.floor[to][g] = self.reach(g, nb * BLOCK, slack).min(nb);
            for b in self.floor[to][g]..nb {
                let blk = self.tabs[from][g][b];
                self.pools[g].refs[blk as usize] += 1;
                self.tabs[to][g][b] = blk;
                self.mark(to, g, b);
            }
        }
        nb * BLOCK
    }

    fn mark(&mut self, slot: usize, g: usize, b: usize) {
        let d = &mut self.dirty[slot][g];
        *d = Some(d.map_or((b, b + 1), |(lo, hi)| (lo.min(b), hi.max(b + 1))));
    }

    /// Sequence `slot`'s table in group `g`: a block a BLOCK of its tokens (0: none).
    pub fn table(&self, slot: usize, g: usize) -> &[u32] {
        &self.tabs[slot][g]
    }

    /// The entries of sequence `slot`'s table in group `g` changed since the last call (to copy to the GPU), and none from now.
    pub fn take_dirty(&mut self, slot: usize, g: usize) -> Option<(usize, usize)> {
        self.dirty[slot][g].take()
    }

    /// Blocks held in group `g` (one held by several sequences once).
    #[cfg(test)]
    fn held(&self, g: usize) -> usize {
        self.pools[g].refs.iter().filter(|&&r| r > 0).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A full group holds every block up to the newest token; a sliding one only the window and a chunk behind it, the older blocks back in the pool; release gives everything back.
    #[test]
    fn a_sequence_holds_the_blocks_a_step_can_read() {
        let mut p = Pages::new(&[200, 200], &[None, Some(64 + 32)], 2, 1024);
        assert!(p.grow(0, 40));
        assert_eq!((p.held(0), p.held(1)), (3, 3));
        assert!(p.grow(0, 500));
        // full: 32 blocks; sliding: tokens 404 .. 499, blocks 25 .. 31
        assert_eq!((p.held(0), p.held(1)), (32, 7));
        assert!(p.table(0, 1)[..25].iter().all(|&b| b == 0) && p.table(0, 1)[25..32].iter().all(|&b| b != 0));
        // every held block is distinct and none is block 0
        let mut all: Vec<u32> = p.table(0, 0).iter().copied().filter(|&b| b != 0).collect();
        all.sort_unstable();
        all.dedup();
        assert_eq!(all.len(), 32);
        p.release(0);
        assert_eq!((p.held(0), p.held(1), p.free(0).0, p.free(1).0), (0, 0, 199, 199));
    }

    /// A sequence cut back holds what one that grew no further holds where a step can read, and gives the rest back; it cannot be cut to where a sliding group has given blocks back.
    #[test]
    fn a_sequence_is_cut_back_where_its_blocks_still_are() {
        // (a window of 64 and a chunk of 32 kept behind the newest token)
        let mut p = Pages::new(&[200, 200], &[None, Some(64 + 32)], 2, 1024);
        assert!(p.grow(0, 500));
        let free = (p.free(0).0, p.free(1).0);
        // cut by less than a chunk: tokens 404.. are held, a step at 480 reads from 416
        assert_eq!(p.shrink(0, 480, 32), 480);
        assert_eq!(p.held(0), 30, "the full group holds 480 tokens' blocks");
        assert!(p.table(0, 0)[30..].iter().all(|&b| b == 0) && p.table(0, 1)[30..].iter().all(|&b| b == 0));
        assert_eq!((p.free(0).0, p.free(1).0), (free.0 + 2, free.1 + 2));
        // it grows again as any sequence of 480 tokens does, and every block it holds is its own
        assert!(p.grow(0, 500));
        let mut all: Vec<u32> = p.table(0, 1).iter().copied().filter(|&b| b != 0).collect();
        all.sort_unstable();
        all.dedup();
        assert_eq!(all.len(), p.held(1));
        // cut by more than a chunk: a step at 300 reads from 236, whose blocks went back at 500 tokens; nothing changes
        let before: Vec<u32> = p.table(0, 1).to_vec();
        assert_eq!(p.shrink(0, 300, 32), 0);
        assert_eq!(p.table(0, 1), &before[..]);
        // a sequence that never gave a block back is cut anywhere
        assert!(p.grow(1, 90) && p.shrink(1, 20, 32) == 20);
        assert_eq!((p.held(0) - 32, p.table(1, 1).iter().filter(|&&b| b != 0).count()), (2, 2));
    }

    /// Growth takes blocks in every group or in none; a sequence past the context never grows.
    #[test]
    fn growth_is_all_or_nothing() {
        let mut p = Pages::new(&[11, 101], &[None, Some(48)], 3, 512);
        assert!(p.grow(0, 160));
        assert_eq!(p.free(0).0, 0);
        assert!(!p.grow(1, 16));
        assert!(p.table(1, 0).iter().chain(p.table(1, 1)).all(|&b| b == 0) && p.free(1).0 == 100 - 3, "a failed growth takes nothing");
        p.release(0);
        assert!(p.grow(1, 16) && !p.grow(2, 513));
    }

    /// The entries that changed are reported once, as one range covering them.
    #[test]
    fn changed_entries_are_reported_once() {
        let mut p = Pages::new(&[100, 100], &[None, Some(32)], 1, 512);
        assert!(p.grow(0, 50));
        assert_eq!(p.take_dirty(0, 0), Some((0, 4)));
        assert_eq!(p.take_dirty(0, 0), None);
        assert!(p.grow(0, 120));
        assert_eq!(p.take_dirty(0, 0), Some((4, 8)));
        // sliding: blocks 1..4 taken at 50 tokens; at 120, 1..4 given back and 5..8 taken
        assert_eq!(p.take_dirty(0, 1), Some((1, 8)));
    }

    /// One sequence alone holds what the capacity says and no more: the full group's blocks bound it, a sliding group with a window and a chunk's blocks does not.
    #[test]
    fn a_sequence_alone_holds_its_capacity() {
        let p = Pages::new(&[33, 9], &[None, Some(96)], 1, 4096);
        assert_eq!(p.capacity(), Some(32 * BLOCK));
        let mut p = Pages::new(&[33, 9], &[None, Some(96)], 1, 4096);
        assert!(p.grow(0, 32 * BLOCK) && !p.grow(0, 32 * BLOCK + 1));
        // a sliding pool too small for its window: it bounds the sequence instead
        let mut q = Pages::new(&[1000, 5], &[None, Some(96)], 1, 4096);
        let cap = q.capacity().unwrap();
        assert!(cap == 3 * BLOCK && q.grow(0, cap));
        // neither bounds it: none
        assert_eq!(Pages::new(&[1000, 9], &[Some(64), Some(96)], 1, 4096).capacity(), None);
    }

    /// Many sequences growing, sharing a start, being cut back and ending at random: a block's holders are the table entries that name it, a block with none is free and
    /// no other is, none is lost, and a growth or a share that fails leaves no trace. A sequence never holds a block past its length, so nothing it writes is another's.
    #[test]
    fn blocks_are_counted_and_never_lost() {
        let (slots, ctx, slack) = (6, 640, 32);
        let mut p = Pages::new(&[60, 30], &[None, Some(48 + slack)], slots, ctx);
        let mut len = vec![0usize; slots];
        let mut seed = 0x2545_f491_4f6c_dd1du64;
        let mut rnd = |m: usize| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed % m as u64) as usize
        };
        let (mut shares, mut cuts) = (0, 0);
        for _ in 0..20000 {
            let s = rnd(slots);
            let before: Vec<Vec<Vec<u32>>> = (0..slots).map(|s| (0..2).map(|g| p.table(s, g).to_vec()).collect()).collect();
            match rnd(10) {
                0 => {
                    p.release(s);
                    len[s] = 0;
                }
                1 | 2 if len[s] == 0 => {
                    // the start of another sequence
                    let from = rnd(slots);
                    let got = p.share(from, s, rnd(len[from] + 1), slack);
                    assert!(got % BLOCK == 0 && got <= len[from]);
                    if got == 0 {
                        assert!((0..slots).all(|x| (0..2).all(|g| p.table(x, g) == &before[x][g][..])));
                    }
                    shares += (got > 0) as usize;
                    len[s] = got;
                }
                3 if len[s] > 0 => {
                    let to = 1 + rnd(len[s]);
                    let got = p.shrink(s, to, slack);
                    assert!(got == 0 || got == to || got == to / BLOCK * BLOCK, "{got} for {to}");
                    if got == 0 {
                        assert!((0..2).all(|g| p.table(s, g) == &before[s][g][..]));
                    } else {
                        len[s] = got;
                        cuts += 1;
                    }
                }
                _ => {
                    let to = (len[s] + 1 + rnd(40)).min(ctx);
                    if p.grow(s, to) {
                        len[s] = to;
                    } else {
                        assert!((0..2).all(|g| p.table(s, g) == &before[s][g][..]));
                    }
                }
            }
            for g in 0..2 {
                let mut named = vec![0u32; p.pools[g].blocks as usize];
                for x in 0..slots {
                    for (b, &blk) in p.table(x, g).iter().enumerate() {
                        if blk != 0 {
                            named[blk as usize] += 1;
                            assert!(b * BLOCK < len[x], "sequence {x} of {} tokens holds entry {b}", len[x]);
                        }
                    }
                }
                assert_eq!(named, p.pools[g].refs, "a block's count is not its holders");
                let mut free = p.pools[g].free.clone();
                free.sort_unstable();
                let want: Vec<u32> = (1..p.pools[g].blocks).filter(|&b| named[b as usize] == 0).collect();
                assert_eq!(free, want, "the free blocks are not the ones no one holds");
            }
            // a block held by several is full and below the length of each: no one writes it
            for x in 0..slots {
                for g in 0..2 {
                    for (b, &blk) in p.table(x, g).iter().enumerate() {
                        assert!(blk == 0 || p.pools[g].refs[blk as usize] == 1 || (b + 1) * BLOCK <= len[x], "sequence {x} would write a block others hold");
                    }
                }
            }
        }
        // (the run did what it is for)
        assert!(shares > 100 && cuts > 200, "{shares} shares, {cuts} cuts");
    }

    /// A sequence that shared a start holds a sliding group's blocks from the window before that start on. As it grows it is given no block for the entries before (they
    /// were never its own, and an empty block there would pass for the tokens' keys and values): a cut back that reads them is refused.
    #[test]
    fn a_sharer_is_given_no_block_before_the_window_it_shared() {
        // (a window of 64 tokens and a chunk of 32)
        let mut p = Pages::new(&[200, 200], &[None, Some(64 + 32)], 2, 1024);
        assert!(p.grow(0, 208));
        // 192 tokens of it shared: the sliding group from the window before 192, entry 8, on
        assert_eq!(p.share(0, 1, 200, 32), 192);
        assert_eq!((p.table(1, 1)[7], p.table(1, 1)[8] != 0), (0, true));
        // it grows by a token: a sequence from its first token would hold the entries from 6 on (the window and a chunk behind 193); this one none before 8
        assert!(p.grow(1, 193));
        assert!(p.table(1, 1)[..8].iter().all(|&b| b == 0) && p.table(1, 1)[8..13].iter().all(|&b| b != 0));
        // a cut to 180 (to 176, the start of a block both hold) reads the window from entry 7: refused, and nothing changes
        assert_eq!((p.cut_len(1, 180, 32), p.shrink(1, 180, 32)), (0, 0));
        // grown by a chunk more it holds what any sequence of its length does; a cut to 200 is its own
        assert!(p.grow(1, 225) && p.table(1, 1)[..8].iter().all(|&b| b == 0) && p.table(1, 1)[8..15].iter().all(|&b| b != 0));
        assert_eq!(p.shrink(1, 200, 32), 200);
        // released, the slot begins as any other
        p.release(1);
        assert!(p.grow(1, 100) && p.table(1, 1)[0] != 0);
    }

    /// A sequence that begins as another does names that one's full blocks and holds them with it: they stay while either holds them, the other's end or cut gives back only
    /// what no one else holds, and a cut into a block both hold is to that block's start.
    #[test]
    fn a_start_is_shared_and_its_blocks_outlive_either_holder() {
        let mut p = Pages::new(&[200, 200], &[None, Some(64 + 32)], 3, 1024);
        assert!(p.grow(0, 100));
        // 70 tokens of it: 4 full blocks (64 tokens)
        assert_eq!(p.share(0, 1, 70, 32), 64);
        assert_eq!(
            (&p.table(1, 0)[..5], p.table(1, 1)[..4].to_vec()),
            (&[p.table(0, 0)[0], p.table(0, 0)[1], p.table(0, 0)[2], p.table(0, 0)[3], 0][..], p.table(0, 1)[..4].to_vec())
        );
        assert_eq!((p.held(0), p.held(1)), (7, 7), "no block more is held: the four are held twice");
        // the second grows on its own blocks
        assert!(p.grow(1, 90));
        assert!(p.table(1, 0)[4] != 0 && p.table(1, 0)[4] != p.table(0, 0)[4] && p.held(0) == 9);
        // less than a block, a sequence that holds something, a sequence and itself: nothing
        assert_eq!((p.share(0, 2, 15, 32), p.share(0, 1, 64, 32), p.share(0, 0, 64, 32)), (0, 0, 0));
        // the first ends: the shared blocks stay with the second
        p.release(0);
        assert_eq!((p.held(0), p.table(1, 0)[..6].iter().filter(|&&b| b != 0).count()), (6, 6));
        // a third shares the second's start; the second is cut into a block both hold: to that block's start
        assert_eq!(p.share(1, 2, 64, 32), 64);
        assert_eq!(p.shrink(1, 50, 32), 48);
        assert_eq!((p.table(1, 0)[..4].iter().filter(|&&b| b != 0).count(), p.table(2, 0)[3] != 0, p.held(0)), (3, true, 4));
        // a cut into a block of its own is to the token
        assert!(p.grow(2, 80));
        assert_eq!(p.shrink(2, 70, 32), 70);
        p.release(1);
        p.release(2);
        assert_eq!((p.held(0), p.held(1), p.free(0).0, p.free(1).0), (0, 0, 199, 199));
        // a sliding group: a start whose blocks the sequence gave up as it grew cannot be shared
        assert!(p.grow(0, 500));
        assert_eq!(p.share(0, 1, 200, 32), 0);
        assert_eq!(p.share(0, 1, 480, 32), 480, "tokens 416 on are held: what a step at 480 reads");
        assert!(p.table(1, 1)[..26].iter().all(|&b| b == 0) && p.table(1, 1)[26..30].iter().all(|&b| b != 0) && p.table(1, 0)[..30].iter().all(|&b| b != 0));
    }
}
