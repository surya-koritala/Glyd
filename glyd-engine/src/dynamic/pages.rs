//! Paged keys and values for the description executor (R4b, #141; docs/design/runtime.md "Paged blocks for every attention kind"): a sequence's cache is the blocks it holds, not a room of
//! the whole context. A block is BLOCK tokens. Layers of one kind (a layer group: the sliding layers of a window, or the full ones) share one pool of blocks and one block table a sequence:
//! the table's entry b is the pool's block that holds the sequence's tokens b BLOCK .. b BLOCK + BLOCK - 1, in every layer of the group (each ring has its own bytes of the block). A
//! sliding group keeps only the blocks a step can still read, the window and a chunk behind the newest token; older ones go back to the pool.
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
}

/// The blocks every sequence holds in every group.
pub struct Pages {
    pools: Vec<Pool>,
    /// [slot][group]: the block of each of the sequence's table entries (0: none).
    tabs: Vec<Vec<Vec<u32>>>,
    /// [slot][group]: the entries changed since `take_dirty`, as a range.
    dirty: Vec<Vec<Option<(usize, usize)>>>,
}

impl Pages {
    /// Pools of `blocks[g]` blocks (block 0 among them) for the groups, each keeping `keep[g]` tokens behind (None: all), for `slots` sequences of at most `ctx` tokens.
    pub fn new(blocks: &[u32], keep: &[Option<usize>], slots: usize, ctx: usize) -> Pages {
        let pools = blocks.iter().zip(keep).map(|(&b, &k)| Pool { keep: k, free: (1..b.max(1)).rev().collect(), blocks: b.max(1) }).collect::<Vec<_>>();
        let n = ctx.div_ceil(BLOCK);
        let tabs = (0..slots).map(|_| (0..pools.len()).map(|_| vec![0u32; n]).collect()).collect();
        let dirty = (0..slots).map(|_| vec![None; pools.len()]).collect();
        Pages { pools, tabs, dirty }
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

    /// Sequence `slot` given the blocks for its first `upto` tokens in every group, and its sliding groups' blocks a step can no longer read given back; false where some group has too
    /// few free (then nothing changes).
    pub fn grow(&mut self, slot: usize, upto: usize) -> bool {
        let max = self.tabs[slot][0].len();
        if upto.div_ceil(BLOCK) > max {
            return false;
        }
        for g in 0..self.pools.len() {
            let (lo, hi) = self.span(g, upto);
            let need = (lo..hi).filter(|&b| self.tabs[slot][g][b] == 0).count();
            // (blocks this call gives back come first)
            let back = (0..lo).filter(|&b| self.tabs[slot][g][b] != 0).count();
            if need > self.pools[g].free.len() + back {
                return false;
            }
        }
        for g in 0..self.pools.len() {
            let (lo, hi) = self.span(g, upto);
            for b in 0..lo {
                let blk = std::mem::take(&mut self.tabs[slot][g][b]);
                if blk != 0 {
                    self.pools[g].free.push(blk);
                    self.mark(slot, g, b);
                }
            }
            for b in lo..hi {
                if self.tabs[slot][g][b] == 0 {
                    self.tabs[slot][g][b] = self.pools[g].free.pop().expect("counted above");
                    self.mark(slot, g, b);
                }
            }
        }
        true
    }

    /// Every block of sequence `slot` back in its pool.
    pub fn release(&mut self, slot: usize) {
        for g in 0..self.pools.len() {
            for b in 0..self.tabs[slot][g].len() {
                let blk = std::mem::take(&mut self.tabs[slot][g][b]);
                if blk != 0 {
                    self.pools[g].free.push(blk);
                    self.mark(slot, g, b);
                }
            }
        }
    }

    /// Sequence `slot` cut back to its first `upto` tokens: the blocks past them back in their pools, so that it goes on from there as one that had grown no further. False, and
    /// nothing changes, where a group no longer holds what a step at `upto` reads: a sliding group gives back the blocks behind its window and `slack` tokens (a chunk: what `keep`
    /// holds beyond the window) as the sequence grows, so one that grew more than `slack` past `upto` cannot be cut there.
    pub fn shrink(&mut self, slot: usize, upto: usize, slack: usize) -> bool {
        let hi = upto.div_ceil(BLOCK);
        for g in 0..self.pools.len() {
            let lo = match self.pools[g].keep {
                Some(k) => upto.saturating_sub(k.saturating_sub(slack)) / BLOCK,
                None => 0,
            };
            if (lo.min(hi)..hi).any(|b| self.tabs[slot][g][b] == 0) {
                return false;
            }
        }
        for g in 0..self.pools.len() {
            for b in hi..self.tabs[slot][g].len() {
                let blk = std::mem::take(&mut self.tabs[slot][g][b]);
                if blk != 0 {
                    self.pools[g].free.push(blk);
                    self.mark(slot, g, b);
                }
            }
        }
        true
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

    /// Blocks held by every sequence in group `g`.
    #[cfg(test)]
    fn held(&self, g: usize) -> usize {
        self.tabs.iter().map(|t| t[g].iter().filter(|&&b| b != 0).count()).sum()
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
        assert!(p.shrink(0, 480, 32));
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
        assert!(!p.shrink(0, 300, 32));
        assert_eq!(p.table(0, 1), &before[..]);
        // a sequence that never gave a block back is cut anywhere
        assert!(p.grow(1, 90) && p.shrink(1, 20, 32));
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

    /// Many sequences growing and ending at random never share a block, never lose one, and a failed growth leaves no trace.
    #[test]
    fn blocks_are_never_shared_or_lost() {
        let (slots, ctx) = (6, 640);
        let mut p = Pages::new(&[60, 30], &[None, Some(80)], slots, ctx);
        let mut len = vec![0usize; slots];
        let mut seed = 0x2545_f491_4f6c_dd1du64;
        let mut rnd = |m: usize| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed % m as u64) as usize
        };
        for _ in 0..5000 {
            let s = rnd(slots);
            if rnd(10) == 0 {
                p.release(s);
                len[s] = 0;
            } else {
                let to = (len[s] + 1 + rnd(40)).min(ctx);
                let before: Vec<Vec<u32>> = (0..2).map(|g| p.table(s, g).to_vec()).collect();
                if p.grow(s, to) {
                    len[s] = to;
                } else {
                    assert!((0..2).all(|g| p.table(s, g) == &before[g][..]));
                }
            }
            for g in 0..2 {
                let mut held: Vec<u32> = (0..slots).flat_map(|s| p.table(s, g).iter().copied().filter(|&b| b != 0).collect::<Vec<_>>()).collect();
                let n = held.len();
                held.sort_unstable();
                held.dedup();
                assert_eq!(held.len(), n, "a block held twice");
                assert_eq!(n + p.free(g).0, p.free(g).1, "a block lost");
            }
        }
    }
}
