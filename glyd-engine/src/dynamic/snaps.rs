//! Recurrent states kept at block boundaries, by the name of the block that ends there (R4c, #142). A model with recurrent layers (Gated DeltaNet, Mamba-2) carries
//! the whole of a sequence's past in its states, which are what they are after its last token: a later request that begins as this sequence did can go on from a
//! place only where the states as they were there were kept. Every `every` tokens of a prompt (a multiple of a block), the states are copied into a pool of fixed
//! size, under the name of the block ending there (`kv::block_hash`: of its tokens and all before them, in the request's scope); the scheduler ends a prompt's pieces
//! on those boundaries (`ModelStep::snap_every`), so the copy is the states at that token exactly. A request then finds the newest boundary of its start where the
//! states were kept and every block of the attention layers up to it is held by name (`hit`): it takes those blocks and the states, and runs from there.
//!
//! The pool is a fixed number of copies made at load (`count`), so what the cache holds is known before the first request; the least recently used is written over.
//! A copy is kept with the block (group 0's) that held its name when it was made: if that block has gone and another holds the name now, the copy is another run's and
//! is not given with that run's blocks (`fresh`).

use crate::dev::Buf;
use crate::ffi::Result;

/// The memory the pool takes by default (GLYD_ENGINE_STATE_SNAP_MB), the most copies it holds, and the fewest that is worth a pool.
const BUDGET_MB: usize = 512;
const MOST: usize = 16;
const LEAST: usize = 2;

/// How many copies of a sequence's states of `bytes` the pool holds: as many as GLYD_ENGINE_STATE_SNAP_MB megabytes of memory take (512 by default), at most 16; 0
/// (no pool) where that is fewer than 2, or the model has no state. The loader and the planner both size the pool by this.
pub fn count(bytes: usize) -> usize {
    let mb = crate::tune!("GLYD_ENGINE_STATE_SNAP_MB").and_then(|v| v.parse().ok()).unwrap_or(BUDGET_MB);
    count_of(bytes, mb)
}

fn count_of(bytes: usize, mb: usize) -> usize {
    if bytes == 0 {
        return 0;
    }
    let n = ((mb << 20) / bytes).min(MOST);
    if n < LEAST {
        0
    } else {
        n
    }
}

/// The boundaries' spacing for a step of `chunk` prompt tokens: GLYD_ENGINE_STATE_EVERY tokens where set, else the chunk; a multiple of `block` (rounded down), and
/// at least one block.
pub fn every(chunk: usize, block: usize) -> usize {
    let want = crate::tune!("GLYD_ENGINE_STATE_EVERY").and_then(|v| v.parse().ok()).unwrap_or(chunk);
    (want / block).max(1) * block
}

/// Which copy to keep a name's states in, and which copy holds a name's: the least recently used is written over.
#[derive(Debug, Default)]
pub(crate) struct Lru {
    keys: Vec<u128>,
    /// Group 0's block that held the name when the copy was made.
    blocks: Vec<u32>,
    used: Vec<u64>,
    tick: u64,
}

impl Lru {
    pub(crate) fn new(n: usize) -> Lru {
        Lru { keys: vec![0; n], blocks: vec![0; n], used: vec![0; n], tick: 0 }
    }

    /// The copy that holds `key`'s states, made when `block` held its name (and so still that run's).
    pub(crate) fn find(&self, key: u128, block: u32) -> Option<usize> {
        (key != 0).then(|| self.keys.iter().zip(&self.blocks).position(|(&k, &b)| k == key && b == block)).flatten()
    }

    /// The copy to write `key`'s states into, made while `block` holds the name: None where a copy of this run's is kept already (the first stays); else a copy that
    /// held the name for another block (another run's, gone), an empty one, or the least recently used.
    pub(crate) fn place(&mut self, key: u128, block: u32) -> Option<usize> {
        if key == 0 || self.keys.is_empty() || self.find(key, block).is_some() {
            return None;
        }
        let i = self
            .keys
            .iter()
            .position(|&k| k == key)
            .or_else(|| self.keys.iter().position(|&k| k == 0))
            .unwrap_or_else(|| (0..self.keys.len()).min_by_key(|&i| self.used[i]).expect("a copy"));
        (self.keys[i], self.blocks[i]) = (key, block);
        self.touch(i);
        Some(i)
    }

    pub(crate) fn touch(&mut self, i: usize) {
        self.tick += 1;
        self.used[i] = self.tick;
    }
}

/// The pool: the index and its copies, one buffer a state each.
pub(crate) struct Snaps {
    /// The boundaries' spacing in tokens (0: no pool).
    pub every: usize,
    pub lru: Lru,
    pub bufs: Vec<Vec<Buf>>,
}

impl Snaps {
    pub(crate) fn none() -> Snaps {
        Snaps { every: 0, lru: Lru::default(), bufs: Vec::new() }
    }

    /// `n` copies of states of the sizes `bytes`, at boundaries every `every` tokens.
    pub(crate) fn new(n: usize, bytes: &[usize], every: usize) -> Result<Snaps> {
        let bufs = (0..n).map(|_| bytes.iter().map(|&b| Buf::zeroed(b)).collect::<Result<Vec<_>>>()).collect::<Result<_>>()?;
        Ok(Snaps { every, lru: Lru::new(n), bufs })
    }
}

/// The longest start of a prompt whose full blocks are named `keys` (in order) that a step can go on from with kept states: the most tokens j * `block`, a multiple of
/// `every`, such that the states after block j - 1 are kept (`kept(j)`) and the attention layers' blocks the step there reads are held by name (`fit(j)`, the blocks
/// `Pages::fit` finds among the first j names, is j). 0: none.
pub(crate) fn hit(keys: &[u128], every: usize, block: usize, kept: impl Fn(usize) -> bool, fit: impl Fn(usize) -> usize) -> usize {
    if every == 0 || every % block != 0 {
        return 0;
    }
    let per = every / block;
    (1..=keys.len() / per).rev().map(|k| k * per).find(|&j| kept(j) && fit(j) == j).map_or(0, |j| j * block)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The pool takes the budget's worth of copies, at most 16, and none where fewer than 2 fit or the model has no state.
    #[test]
    fn the_pool_is_sized_by_its_budget() {
        assert_eq!(count_of(48 << 20, 512), 10);
        assert_eq!(count_of(144 << 20, 512), 3);
        assert_eq!(count_of(300 << 20, 512), 0);
        assert_eq!(count_of(1 << 20, 512), 16);
        assert_eq!(count_of(0, 512), 0);
    }

    /// A name's copy is found while the block it was made with holds the name; the first copy of a run stays; a run's copy whose block has gone is written over by the
    /// next run's; with every copy taken, the least recently used is.
    #[test]
    fn copies_are_kept_by_name_and_block_the_oldest_written_over() {
        let mut l = Lru::new(2);
        assert_eq!(l.place(0, 5), None);
        let a = l.place(11, 5).unwrap();
        assert_eq!(l.find(11, 5), Some(a));
        assert_eq!(l.place(11, 5), None, "the first copy of the run stays");
        assert_eq!(l.find(11, 6), None, "another block holds the name: not this run's");
        assert_eq!(l.place(11, 6), Some(a), "the gone run's copy is written over");
        assert_eq!(l.find(11, 5), None);
        let b = l.place(22, 7).unwrap();
        assert_ne!(a, b);
        l.touch(a);
        let c = l.place(33, 8).unwrap();
        assert_eq!(c, b, "the least recently used is written over");
        assert_eq!((l.find(22, 7), l.find(11, 6), l.find(33, 8)), (None, Some(a), Some(c)));
    }

    /// The newest boundary of the start whose states are kept and whose blocks are held; not one past what is held, nor one whose states are not kept.
    #[test]
    fn a_start_is_found_at_the_newest_boundary_kept_and_held() {
        let keys: Vec<u128> = (1..=10).collect();
        // boundaries every 32 tokens of blocks of 16: after 2, 4, 6, 8, 10 blocks
        let kept = |j: usize| matches!(j, 2 | 4 | 8);
        assert_eq!(hit(&keys, 32, 16, kept, |j| j), 8 * 16);
        assert_eq!(hit(&keys, 32, 16, kept, |j| j.min(7)), 4 * 16, "block 8 is not held");
        assert_eq!(hit(&keys, 32, 16, |_| false, |j| j), 0);
        assert_eq!(hit(&keys[..3], 32, 16, kept, |j| j), 2 * 16);
        assert_eq!(hit(&keys, 0, 16, kept, |j| j), 0);
    }
}
