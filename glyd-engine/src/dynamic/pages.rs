//! Paged keys and values for the description executor (R4b, #141; docs/design/runtime.md "Paged blocks for every attention kind"): a sequence's cache is the blocks it holds, not a room of
//! the whole context. A block is BLOCK tokens. Layers of one kind (a layer group: the sliding layers of a window, or the full ones) share one pool of blocks and one block table a sequence:
//! the table's entry b is the pool's block that holds the sequence's tokens b BLOCK .. b BLOCK + BLOCK - 1, in every layer of the group (each ring has its own bytes of the block). A
//! sliding group keeps only the blocks a step can still read, the window and a chunk behind the newest token; older ones go back to the pool.
//!
//! A block may be held by more than one sequence (`share`: a prompt that begins as another sequence does names that sequence's full blocks and runs only the rest). A block
//! counts its holders and goes back to its pool when the last gives it up. A shared block is never written: a sequence writes from its own length on, a shared block is full
//! and below the lengths of all who hold it, and a sequence cut back (`shrink`) into a block that others hold is cut to that block's start instead.
//!
//! A full block has a name (R4c: `kv::block_hash`, of its tokens and all before them, which the scheduler tells with `name`), and a pool finds its blocks by name. A named
//! block that no sequence holds is not free yet: it stays, with its bytes, until its pool needs a block, the oldest such going first (a sequence gives its blocks up last
//! block first, so a chain loses its end before its start). So a prompt finds the blocks of its start whether or not a sequence still holds them (`adopt`): a conversation's
//! system prompt after that conversation has grown past it (a sliding group gave those blocks up long ago) or has ended. A named block is never written: a cut back into
//! one that the sequence alone holds takes its name away first.
//!
//! A pool's block may cover more tokens than BLOCK (`covers`): DeepSeek-V4's compressed entries, one for every `ratio` tokens, are BLOCK entries a block, so a block of theirs covers
//! BLOCK `ratio` tokens (R4b, #141). Every question below is asked in tokens and each group answers it in its own blocks: a sequence of n tokens holds a compressed group's
//! blocks 0 .. n / covers (rounded up), the block that holds its tokens past the last full one too. Such a block is named for the block of BLOCK tokens that ends it (its entries
//! are of every token before), so a start is found by name in whole blocks of every group (`fit`), and a start shared at a place within one of them leaves that block to the
//! caller (`share`: rows.rs copies it).
//!
//! Block 0 of every pool is never given out: every entry a sequence does not hold names it, so a write the engine makes there (a graph's capture runs a step at position 0 of a sequence
//! that holds nothing) lands where no one reads. The kernels read a position through the table (`g4_place` in kernels/gemma4.cu); this file is the host's side: which blocks a sequence
//! holds, and which entries changed since the tables were last copied to the GPU.

use crate::ffi::Result;
use std::collections::{BTreeMap, HashMap};

/// Tokens a block.
pub const BLOCK: usize = 16;

/// R4b's deterministic capacity: the pages a pass that writes tokens `n` to `upto` of a sequence writes in a ring (each may be made raw by values that do not
/// compress, and take a slab of its raw pool: `Dyn::kv12_room`).
pub fn pages_written(n: usize, upto: usize) -> usize {
    if upto > n {
        (upto - 1) / BLOCK - n / BLOCK + 1
    } else {
        0
    }
}

/// The pages a growth of a sequence adds to what the step has promised it: its pass writes tokens `n` to `upto`, and `had` is the pages (first, end) the step's growths
/// before promised it (a prompt is asked for as it is let in and again before its pass; a sequence's tokens do not change within a step, so both begin at one page). The
/// pages to promise more, and the sequence's promised pages after.
pub fn pages_more(had: Option<(usize, usize)>, n: usize, upto: usize) -> (usize, (usize, usize)) {
    let new = (n / BLOCK, n / BLOCK + pages_written(n, upto));
    match had {
        Some((a, b)) => {
            let all = (a.min(new.0), b.max(new.1));
            ((all.1 - all.0) - (b - a), all)
        }
        None => (new.1 - new.0, new),
    }
}

/// Whether a pass whose pages need `need` slabs of each raw pool may begin: every pool has them free (`spare`) beside what is promised already (`taken`).
pub fn room(need: &[i64], spare: &[i64], taken: &[i64]) -> bool {
    need.iter().zip(spare).zip(taken).all(|((n, s), t)| *n <= s - t)
}

/// The 12-bit pools a growth or a copy back reads (src/dynamic/kvpages.rs `Gpu`, the engine's; a test's made-up ones).
pub trait Pools {
    /// The swap's stream waited for: its copies out have given their pages' slabs and chunks back, its copies back have taken theirs.
    fn land(&mut self) -> Result<()>;
    /// The named blocks no sequence holds dropped, their pages cleared (their slabs and chunks back).
    fn drop_idle(&mut self) -> Result<()>;
    /// Each raw pool's slabs and the escape chunks free as the pools count them, once the engine's stream has run what it was given.
    fn free(&mut self) -> Result<(Vec<i64>, i64)>;
}

/// R4b's deterministic capacity as the host keeps it (`Dyn::kv12_room`): each raw pool's rings (a page of each may be made raw, a slab of the pool), and what this
/// step's growths promised each pool and the pages (first, end) each slot was promised (a slot asked twice in a step is promised its pages once, and one that gives its
/// blocks up before the step runs takes its promise with it). Every page is counted once, where it is: the pools are read with nothing in flight on the swap's stream
/// (`counted`), so a copy back's pages move into the pools' counts at one point, the reading after it is issued, wherever its copy has got to. (Counted as promised
/// until the scheduler saw its sequence copied back, a copy that had landed was counted twice, by the pools and by that promise: on an A10 Gemma 4's KV-sharing case
/// was refused a decode step its plan had room for, 1 slab of the 3 a page takes left of a pool of 64, and the engine ended with an error.)
#[derive(Default)]
pub struct Ledger {
    rings: Vec<i64>,
    promised: Vec<i64>,
    asked: Vec<(usize, usize, usize)>,
}

impl Ledger {
    pub fn new(rings: Vec<i64>) -> Ledger {
        Ledger { promised: vec![0; rings.len()], rings, ..Default::default() }
    }

    /// The slabs `pages` pages of a sequence may take of each raw pool: a slab for every page of every 12-bit ring that draws on it.
    pub fn need(&self, pages: usize) -> Vec<i64> {
        self.rings.iter().map(|&r| r * pages as i64).collect()
    }

    /// Each raw pool's slabs and the escape chunks free beside what is promised, read with every copy on the swap's stream landed: the pools' counts are then what
    /// the pages hold, each counted once (a copy back's pages loaded, a copy out's given back).
    pub fn counted(&self, p: &mut impl Pools) -> Result<(Vec<i64>, i64)> {
        p.land()?;
        let (slabs, chunks) = p.free()?;
        Ok(((0..slabs.len()).map(|c| slabs[c] - self.promised[c]).collect(), chunks))
    }

    /// Whether `slabs` of each raw pool and `chunks` escape chunks are free beside what is promised (`counted`); where not, the named blocks no one holds dropped
    /// (their pages cleared) and counted again. A refusal is so from counts with nothing in flight.
    fn fits(&self, p: &mut impl Pools, slabs: &[i64], chunks: i64) -> Result<bool> {
        for attempt in 0..2 {
            if attempt == 1 {
                p.drop_idle()?;
            }
            let (free, cfree) = self.counted(p)?;
            if room(slabs, &free, &vec![0; slabs.len()]) && chunks <= cfree {
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// A pass of sequence `slot`'s tokens `n` to `upto` (a chunk at most) may begin only where every raw pool has a slab for every page it writes beside what is promised
    /// (`fits`); its pages promised where it may.
    pub fn grow(&mut self, p: &mut impl Pools, slot: usize, n: usize, upto: usize) -> Result<bool> {
        let had = self.asked.iter().find(|a| a.0 == slot).map(|a| (a.1, a.2));
        let (more, pages) = pages_more(had, n, upto);
        let need = self.need(more);
        if need.iter().all(|&x| x == 0) {
            return Ok(true);
        }
        if !self.fits(p, &need, 0)? {
            return Ok(false);
        }
        self.promised.iter_mut().zip(&need).for_each(|(p, x)| *p += x);
        self.asked.retain(|a| a.0 != slot);
        self.asked.push((slot, pages.0, pages.1));
        Ok(true)
    }

    /// A copy back into the empty slot `slot` whose pages take `slabs` of each raw pool and `chunks` escape chunks as they load: whether they are free beside what is
    /// promised (`fits`). Once issued its pages are in the pools' counts from the next reading on.
    pub fn copy_back(&mut self, p: &mut impl Pools, slot: usize, slabs: &[i64], chunks: i64) -> Result<bool> {
        debug_assert!(self.asked.iter().all(|a| a.0 != slot), "a copy back into slot {slot}, which this step's growths promised pages");
        self.fits(p, slabs, chunks)
    }

    /// A step has run: what its growths promised is in the pools' counts now.
    pub fn stepped(&mut self) {
        self.promised.iter_mut().for_each(|p| *p = 0);
        self.asked.clear();
    }

    /// Slot `slot` gives its blocks up (released, or copied out to the host) before the step it was promised pages for has run: the promise goes with it.
    pub fn unpromise(&mut self, slot: usize) {
        if let Some(i) = self.asked.iter().position(|a| a.0 == slot) {
            let (_, first, end) = self.asked.swap_remove(i);
            let back = self.need(end - first);
            self.promised.iter_mut().zip(&back).for_each(|(p, x)| *p -= x);
        }
    }
}

/// One layer group's pool.
struct Pool {
    /// Tokens behind the newest a step may read (the window and a chunk); None: every token (full attention).
    keep: Option<usize>,
    /// Tokens a block covers: BLOCK, or BLOCK compressed entries' (DeepSeek-V4's: BLOCK `ratio`).
    covers: usize,
    /// Blocks not held, the lowest last (so the first given out are the lowest).
    free: Vec<u32>,
    blocks: u32,
    /// The sequences that hold each block (their table entries that name it).
    refs: Vec<u32>,
    /// Each block's name (0: none), and the named blocks by name.
    name: Vec<u128>,
    index: HashMap<u128, u32>,
    /// The named blocks no sequence holds, by the time each was given up (`age`, from `tick`): the oldest is the first taken for another use.
    idle: BTreeMap<u64, u32>,
    age: Vec<u64>,
    tick: u64,
    /// Names taken off blocks since `Pages::take_gone` (src/kv_events.rs).
    gone: Vec<u128>,
    /// R4d: each block's fence (0: none), the copy that may still read it (a swap's part: `Pages::release_fenced`), and the free blocks that have one, taken after the
    /// others (the first given up last, so taken first: a swap's first entries); the most recent fence of a block taken since `Pages::take_fence`.
    fence: Vec<u64>,
    cooling: Vec<u32>,
    hit: u64,
}

impl Pool {
    fn new(blocks: u32, keep: Option<usize>, covers: usize) -> Pool {
        let n = blocks.max(1);
        Pool {
            keep,
            covers,
            free: (1..n).rev().collect(),
            blocks: n,
            refs: vec![0; n as usize],
            name: vec![0; n as usize],
            index: HashMap::new(),
            idle: BTreeMap::new(),
            age: vec![0; n as usize],
            tick: 0,
            gone: Vec::new(),
            fence: vec![0; n as usize],
            cooling: Vec::new(),
            hit: 0,
        }
    }
    /// Blocks that can be had now: the free, and the named ones no sequence holds.
    fn available(&self) -> usize {
        self.free.len() + self.cooling.len() + self.idle.len()
    }
    /// A block for one holder, with no name: a free one (one no copy reads first), else the oldest named one that no sequence holds. Its fence, where it has one, is
    /// the caller's to wait for (`hit`).
    fn take(&mut self) -> Option<u32> {
        let b = match self.free.pop().or_else(|| self.cooling.pop()) {
            Some(b) => b,
            None => {
                let (&age, &b) = self.idle.iter().next()?;
                self.idle.remove(&age);
                self.unname(b);
                b
            }
        };
        self.refs[b as usize] = 1;
        self.hit = self.hit.max(std::mem::take(&mut self.fence[b as usize]));
        Some(b)
    }
    /// One holder gives block `b` up. When it was the last: free again, or, named, kept until the pool needs it.
    fn give(&mut self, b: u32) {
        let r = &mut self.refs[b as usize];
        debug_assert!(*r > 0, "block {b} given up twice");
        *r = r.saturating_sub(1);
        if *r > 0 {
            return;
        }
        if self.name[b as usize] == 0 {
            if self.fence[b as usize] > 0 {
                self.cooling.push(b);
            } else {
                self.free.push(b);
            }
        } else {
            self.tick += 1;
            self.age[b as usize] = self.tick;
            self.idle.insert(self.tick, b);
        }
    }
    /// One holder more of block `b`, which is held or waits by its name.
    fn hold(&mut self, b: u32) {
        if self.refs[b as usize] == 0 {
            self.idle.remove(&self.age[b as usize]);
        }
        self.refs[b as usize] += 1;
    }
    /// Block `b` has no name any more (it is about to be written, or used for something else).
    fn unname(&mut self, b: u32) {
        let n = std::mem::take(&mut self.name[b as usize]);
        if n != 0 {
            self.index.remove(&n);
            self.gone.push(n);
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
    /// The most tokens a sequence holds, in blocks of BLOCK.
    most: usize,
}

impl Pages {
    /// Pools of `blocks[g]` blocks (block 0 among them) for the groups, each keeping `keep[g]` tokens behind (None: all), for `slots` sequences of at most `ctx` tokens.
    pub fn new(blocks: &[u32], keep: &[Option<usize>], slots: usize, ctx: usize) -> Pages {
        Pages::covering(blocks, keep, &vec![BLOCK; blocks.len()], slots, ctx)
    }

    /// `new` with group g's blocks covering `covers[g]` tokens each (a multiple of BLOCK; more than BLOCK only for a full group: compressed entries').
    pub fn covering(blocks: &[u32], keep: &[Option<usize>], covers: &[usize], slots: usize, ctx: usize) -> Pages {
        debug_assert!(covers.iter().zip(keep).all(|(&c, k)| c % BLOCK == 0 && c > 0 && (c == BLOCK || k.is_none())));
        let pools = blocks.iter().zip(keep).zip(covers).map(|((&b, &k), &c)| Pool::new(b, k, c)).collect::<Vec<_>>();
        let tabs = (0..slots).map(|_| pools.iter().map(|p| vec![0u32; ctx.div_ceil(p.covers)]).collect()).collect();
        let dirty = (0..slots).map(|_| vec![None; pools.len()]).collect();
        let floor = vec![vec![0; pools.len()]; slots];
        Pages { pools, tabs, dirty, floor, most: ctx.div_ceil(BLOCK) }
    }

    /// Tokens a block of group `g` covers.
    pub fn covers(&self, g: usize) -> usize {
        self.pools[g].covers
    }

    /// Group `g`'s table entry that holds the tokens from `at` on where it is not a whole block (a block of `covers` tokens a sequence of `at` tokens holds part of).
    pub fn partial(&self, g: usize, at: usize) -> Option<usize> {
        let c = self.pools[g].covers;
        (!at.is_multiple_of(c)).then_some(at / c)
    }

    pub fn groups(&self) -> usize {
        self.pools.len()
    }

    /// Blocks that can be had in group `g` (free, or named and held by no sequence), and all of its blocks but block 0.
    pub fn free(&self, g: usize) -> (usize, usize) {
        (self.pools[g].available(), self.pools[g].blocks as usize - 1)
    }

    /// The most tokens one sequence can hold with nothing else holding blocks: a full group's blocks (but block 0) a BLOCK each; a sliding group no limit where it holds a window and a chunk
    /// (and the blocks they straddle), else what its blocks hold less one. None: no group limits it.
    pub fn capacity(&self) -> Option<usize> {
        self.pools
            .iter()
            .filter_map(|p| {
                let usable = p.blocks as usize - 1;
                match p.keep {
                    None => Some(usable * p.covers),
                    Some(k) if usable >= k.div_ceil(BLOCK) + 1 => None,
                    Some(_) => Some(usable.saturating_sub(1) * BLOCK),
                }
            })
            .min()
    }

    /// The table entries sequence `slot` needs to hold its first `upto` tokens in group `g`: from the first a step may still read to the last.
    fn span(&self, g: usize, upto: usize) -> (usize, usize) {
        let c = self.pools[g].covers;
        let hi = upto.div_ceil(c);
        let lo = match self.pools[g].keep {
            Some(k) => upto.saturating_sub(k) / c,
            None => 0,
        };
        (lo.min(hi), hi)
    }

    /// The entries sequence `slot`, which holds its first `n` tokens, holds in group `g` to grow to `upto` in passes of at most `slack` tokens: where one pass gets there, `span`; where it
    /// takes several, in a sliding group as many entries as `span` has, from the first its next pass reads on (the passes after take theirs as the window moves, giving back what it
    /// leaves). And within them the first entry the sequence is given a block for: none before the first it ever held (`floor`).
    fn span_from(&self, slot: usize, g: usize, n: usize, upto: usize, slack: usize) -> (usize, usize, usize) {
        let (lo, hi) = self.span(g, upto);
        let (near, _) = self.span(g, upto.min(n + slack));
        let hi = hi.min(near + (hi - lo));
        (near, near.max(self.floor[slot][g]).min(hi), hi)
    }

    /// Sequence `slot` given the blocks for its first `upto` tokens in every group, and its sliding groups' blocks a step can no longer read given back; false where some group has too
    /// few free (then nothing changes).
    pub fn grow(&mut self, slot: usize, upto: usize) -> bool {
        self.grow_from(slot, upto, upto, 0)
    }

    /// `grow` for a sequence that holds its first `n` tokens and goes on to `upto` in passes of at most `slack` tokens (a prompt's room, taken at once as the scheduler lets it in):
    /// a full group's blocks to `upto`; a sliding group's, where that is more than a pass, from what its next pass reads on, no more of them than `upto`'s window and a chunk.
    /// (The blocks of `upto`'s own window instead left a pool of one sequence's window and a chunk none for the first pass, whose growth then failed each time the sequence was
    /// let in again: one-user serve of Gemma 4 never answered a prompt longer than its window and a chunk. And they gave back the window before `n`, which the next pass reads.)
    pub fn grow_from(&mut self, slot: usize, n: usize, upto: usize, slack: usize) -> bool {
        if upto.div_ceil(BLOCK) > self.most {
            return false;
        }
        for g in 0..self.pools.len() {
            let (lo, from, hi) = self.span_from(slot, g, n, upto, slack);
            let need = (from..hi).filter(|&b| self.tabs[slot][g][b] == 0).count();
            // (blocks this call gives back come first: those it alone holds)
            let back = (0..lo).filter(|&b| self.tabs[slot][g][b] != 0 && self.pools[g].refs[self.tabs[slot][g][b] as usize] == 1).count();
            if need > self.pools[g].available() + back {
                return false;
            }
        }
        for g in 0..self.pools.len() {
            let (lo, from, hi) = self.span_from(slot, g, n, upto, slack);
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

    /// The table entries sequence `slot` holds in each group among the blocks of its first `upto` tokens, and its floor in each (R4d: what a swap copies, and what `restore`
    /// takes again).
    pub fn held_entries(&self, slot: usize, upto: usize) -> (Vec<Vec<usize>>, Vec<usize>) {
        let entries = (0..self.pools.len())
            .map(|g| (0..upto.div_ceil(self.pools[g].covers).min(self.tabs[slot][g].len())).filter(|&b| self.tabs[slot][g][b] != 0).collect())
            .collect();
        (entries, self.floor[slot].clone())
    }

    /// Empty sequence `slot` given a block of its own for each table entry `entries[g]` in every group, and the floors `floor` (what `held_entries` said of a sequence swapped to the host,
    /// whose blocks are copied back into these); false where some group has too few (nothing changes).
    pub fn restore(&mut self, slot: usize, entries: &[Vec<usize>], floor: &[usize]) -> bool {
        if self.tabs[slot].iter().any(|t| t.iter().any(|&b| b != 0)) || entries.len() != self.pools.len() {
            return false;
        }
        if entries.iter().enumerate().any(|(g, e)| e.len() > self.pools[g].available() || e.iter().any(|&b| b >= self.tabs[slot][g].len())) {
            return false;
        }
        for (g, e) in entries.iter().enumerate() {
            self.floor[slot][g] = floor.get(g).copied().unwrap_or(0);
            for &b in e {
                self.tabs[slot][g][b] = self.pools[g].take().expect("counted above");
                self.mark(slot, g, b);
            }
        }
        true
    }

    /// Every block of sequence `slot` back in its pool, the last first (of the named ones, which stay until the pool needs them, the sequence's end is then the first to go
    /// and its start the last).
    pub fn release(&mut self, slot: usize) {
        self.release_fenced(slot, |_, _| 0);
    }

    /// `release` of a sequence whose blocks a copy still reads (R4d: a swap out, on a stream of its own): each block keeps the fence `fence(g, entry)` (a number, 0: none,
    /// later copies' larger) until it is taken again, whoever else holds it now. A block with a fence is taken after the free ones that have none (`take_fence` then says
    /// the largest fence taken, which the step that writes them waits for); of a sequence's, its first entries first.
    pub fn release_fenced(&mut self, slot: usize, fence: impl Fn(usize, usize) -> u64) {
        for g in 0..self.pools.len() {
            self.floor[slot][g] = 0;
            for b in (0..self.tabs[slot][g].len()).rev() {
                let blk = std::mem::take(&mut self.tabs[slot][g][b]);
                if blk != 0 {
                    let f = &mut self.pools[g].fence[blk as usize];
                    *f = (*f).max(fence(g, b));
                    self.pools[g].give(blk);
                    self.mark(slot, g, b);
                }
            }
        }
    }

    /// The blocks sequence `slot` alone holds in each group that have no name (`release` gives them back free, their content read by no one again), with their entries.
    pub fn sole_unnamed(&self, slot: usize) -> Vec<Vec<(usize, u32)>> {
        (0..self.pools.len())
            .map(|g| {
                let p = &self.pools[g];
                self.tabs[slot][g]
                    .iter()
                    .enumerate()
                    .filter(|&(_, &b)| b != 0 && p.refs[b as usize] == 1 && p.name[b as usize] == 0)
                    .map(|(e, &b)| (e, b))
                    .collect()
            })
            .collect()
    }

    /// Every named block of group `g` that no sequence holds given its name up and freed (the start it held is not found again): the blocks, whose content no one reads.
    pub fn drop_idle(&mut self, g: usize) -> Vec<u32> {
        let p = &mut self.pools[g];
        let blocks: Vec<u32> = std::mem::take(&mut p.idle).into_values().collect();
        for &b in &blocks {
            p.unname(b);
            if p.fence[b as usize] > 0 {
                p.cooling.push(b);
            } else {
                p.free.push(b);
            }
        }
        blocks
    }

    /// The largest fence of the blocks taken since the last call (0: none): the copy a step that writes them waits for first.
    pub fn take_fence(&mut self) -> u64 {
        self.pools.iter_mut().map(|p| std::mem::take(&mut p.hit)).max().unwrap_or(0)
    }

    /// The first tokens of a sequence a step at `upto` reads in group `g`, as a table entry: a sliding group's window behind `upto` (what `keep` holds less `slack`, a chunk),
    /// a full group's first.
    fn reach(&self, g: usize, upto: usize, slack: usize) -> usize {
        match self.pools[g].keep {
            Some(k) => upto.saturating_sub(k.saturating_sub(slack)) / self.pools[g].covers,
            None => 0,
        }
    }

    /// Whether sequence `slot` holds, in every group, the blocks a step at `upto` reads (a sliding group's from its window on; `slack` as `shrink`'s).
    fn reads_held(&self, slot: usize, upto: usize, slack: usize) -> bool {
        (0..self.pools.len()).all(|g| {
            let hi = upto.div_ceil(self.pools[g].covers);
            (self.reach(g, upto, slack).min(hi)..hi).all(|b| self.tabs[slot][g][b] != 0)
        })
    }

    /// What sequence `slot` would hold cut back to go on within its first `upto` tokens (`shrink`, which makes the cut): `upto`, or the start of the block `upto` is in
    /// where others hold that block too (it would be written from `upto` on). 0 where a group no longer holds what a step there reads: a sliding group gives up the
    /// blocks behind its window and `slack` tokens (a chunk: what `keep` holds beyond the window) as the sequence grows, so one that grew more than `slack` past the place
    /// cannot be cut there.
    pub fn cut_len(&self, slot: usize, upto: usize, slack: usize) -> usize {
        // (each group's block the place is in, where others hold it too: to its start; a group of larger blocks may move the place to before another's)
        let mut upto = upto;
        while let Some(at) = (0..self.pools.len())
            .filter_map(|g| self.partial(g, upto).filter(|&b| self.pools[g].refs[self.tabs[slot][g][b] as usize] > 1).map(|b| b * self.pools[g].covers))
            .min()
        {
            upto = at;
        }
        if !self.reads_held(slot, upto, slack) {
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
        // (the block the cut is in is written from there on: it is no longer the tokens its name says, and a copy that read it, where it was another sequence's swapped out
        // and this one found it by name, is waited for as a block taken is)
        for g in 0..self.pools.len() {
            if let Some(b) = self.partial(g, upto) {
                let blk = self.tabs[slot][g][b];
                if blk != 0 {
                    let p = &mut self.pools[g];
                    p.unname(blk);
                    p.hit = p.hit.max(std::mem::take(&mut p.fence[blk as usize]));
                }
            }
        }
        // (the last first, as `release` gives them: of the named ones the end of what was cut off is the first to go)
        for g in 0..self.pools.len() {
            let hi = upto.div_ceil(self.pools[g].covers);
            for b in (hi..self.tabs[slot][g].len()).rev() {
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
        self.reads_held(slot, upto, slack)
    }

    /// The full blocks of the first `upto` tokens of sequence `from` named by sequence `to` as well, which must hold nothing: the tokens `to` then holds (a whole number of
    /// blocks), each of those blocks with one holder more. 0, and nothing changes, where `upto` is less than a block or `from` no longer holds what a step there reads (a
    /// sliding group's blocks behind its window; `slack` as `shrink`'s). Neither writes what the other reads: `to` goes on from the end of those blocks, `from` from its own
    /// length, which is past them. A group of larger blocks (`covers`) shares the whole ones; its block that holds the tokens past them (`partial`) is not shared, and `to`
    /// lacks it until it is given one with those tokens' entries (rows.rs copies them).
    pub fn share(&mut self, from: usize, to: usize, upto: usize, slack: usize) -> usize {
        let n = upto / BLOCK * BLOCK;
        if n == 0 || from == to || self.tabs[to].iter().any(|t| t.iter().any(|&b| b != 0)) {
            return 0;
        }
        let whole = |p: &Pages, g: usize| (p.reach(g, n, slack).min(n / p.pools[g].covers), n / p.pools[g].covers);
        for g in 0..self.pools.len() {
            let (lo, nb) = whole(self, g);
            if (lo..nb).any(|b| self.tabs[from][g][b] == 0) {
                return 0;
            }
        }
        for g in 0..self.pools.len() {
            let (lo, nb) = whole(self, g);
            self.floor[to][g] = lo;
            for b in lo..nb {
                let blk = self.tabs[from][g][b];
                self.pools[g].refs[blk as usize] += 1;
                self.tabs[to][g][b] = blk;
                self.mark(to, g, b);
            }
        }
        n
    }

    /// Sequence `slot`'s block `b` is full, and `key` is its name (`kv::block_hash`): in every group that holds it, the block is found by that name from now on, unless
    /// another block of the group has the name already (two sequences that ran the same tokens side by side: the first stays the one that is found).
    pub fn name(&mut self, slot: usize, b: usize, key: u128) {
        if key == 0 || b >= self.most {
            return;
        }
        for g in 0..self.pools.len() {
            // (a group of larger blocks: its block that this one ends)
            let Some(e) = self.entry_ending(g, b) else { continue };
            let blk = self.tabs[slot][g][e] as usize;
            let p = &mut self.pools[g];
            debug_assert!(blk == 0 || p.name[blk] == 0 || p.name[blk] == key, "block {blk} named twice");
            if blk != 0 && p.name[blk] == 0 && !p.index.contains_key(&key) {
                p.name[blk] = key;
                p.index.insert(key, blk as u32);
            }
        }
    }

    /// The names no block is found by any more since the last call (a block named so was taken for another use), each once: the cache's removed blocks
    /// (src/kv_events.rs).
    pub fn take_gone(&mut self) -> Vec<u128> {
        let mut out: Vec<u128> = self.pools.iter_mut().flat_map(|p| std::mem::take(&mut p.gone)).collect();
        out.sort_unstable();
        out.dedup();
        out
    }

    /// Group `g`'s table entry whose block ends where block `b` of BLOCK tokens does (None: none ends there).
    fn entry_ending(&self, g: usize, b: usize) -> Option<usize> {
        let r = self.pools[g].covers / BLOCK;
        (b + 1).is_multiple_of(r).then(|| (b + 1) / r - 1)
    }

    /// The block of group `g` that `key` names now, if any.
    pub fn named(&self, g: usize, key: u128) -> Option<u32> {
        self.pools.get(g).and_then(|p| p.index.get(&key).copied())
    }

    /// The longest start of a sequence whose full blocks are named `keys` (in order) that the pools hold by name and a step can go on from, taken by sequence `to`, which
    /// must hold nothing: the tokens `to` then holds (a whole number of blocks), each block with one holder more. A step at n blocks reads every block before in a full group
    /// and the window behind in a sliding one (`slack` as `shrink`'s), so that is what must be there: the most n such that it is. 0, and nothing changes: none.
    pub fn adopt(&mut self, to: usize, keys: &[u128], slack: usize) -> usize {
        if self.tabs[to].iter().any(|t| t.iter().any(|&b| b != 0)) {
            return 0;
        }
        let n = self.fit(keys, slack);
        for g in 0..self.pools.len() {
            let r = self.pools[g].covers / BLOCK;
            let lo = self.reach(g, n * BLOCK, slack).min(n / r);
            self.floor[to][g] = lo;
            for b in lo..n / r {
                let blk = self.pools[g].index[&keys[(b + 1) * r - 1]];
                self.pools[g].hold(blk);
                self.tabs[to][g][b] = blk;
                self.mark(to, g, b);
            }
        }
        n * BLOCK
    }

    /// The blocks `adopt` would give a sequence whose full blocks are named `keys`: the most n such that every block a step at n blocks reads is held by name. (A group of
    /// larger blocks holds by name only its whole ones: n is a whole number of them.)
    pub fn fit(&self, keys: &[u128], slack: usize) -> usize {
        let mut n = keys.len().min(self.most);
        loop {
            let was = n;
            for p in self.pools.iter().filter(|p| p.keep.is_none()) {
                // (its blocks named by the names of the blocks of BLOCK tokens that end them: its first block it lacks, or its last whole one, ends the start)
                let r = p.covers / BLOCK;
                let have = (0..n / r).position(|b| !p.index.contains_key(&keys[(b + 1) * r - 1])).unwrap_or(n / r);
                n = if have < n / r || r > 1 { have * r } else { n };
            }
            // (a sliding group that lacks block b of the window behind n lacks it for every length from b + 1 to n: those windows hold b too)
            'fit: while n > 0 {
                for g in (0..self.pools.len()).filter(|&g| self.pools[g].keep.is_some()) {
                    let lo = self.reach(g, n * BLOCK, slack).min(n);
                    if let Some(b) = (lo..n).rev().find(|&b| !self.pools[g].index.contains_key(&keys[b])) {
                        n = b;
                        continue 'fit;
                    }
                }
                break;
            }
            if n == was {
                return n;
            }
        }
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

    /// Whether any of sequence `slot`'s tables has entries changed since they were last taken (`take_dirty`).
    pub fn dirty(&self, slot: usize) -> bool {
        self.dirty[slot].iter().any(Option::is_some)
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

    /// The names taken off blocks that were taken for another use are told once (src/kv_events.rs: the cache's removed blocks); a block given up keeps its name.
    #[test]
    fn names_taken_off_reused_blocks_are_told_once() {
        let mut p = Pages::new(&[4], &[None], 2, 1024);
        assert!(p.grow(0, 48));
        for b in 0..3 {
            p.name(0, b, 100 + b as u128);
        }
        p.release(0);
        assert!(p.take_gone().is_empty(), "blocks given up wait by their names");
        assert!(p.grow(1, 32));
        let gone = p.take_gone();
        assert_eq!(gone.len(), 2);
        assert!(gone.iter().all(|k| (100..103).contains(k)));
        assert!(p.take_gone().is_empty());
    }

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

    /// A prompt swapped out between its passes and copied back (R4d) takes its room on to the prompt's end from what its next pass reads (`grow_from`), as it did when it was
    /// let in: the window before its tokens, which was copied out, keeps the blocks `restore` gave it. The blocks of the end's own window (`grow`) would give that window back.
    #[test]
    fn a_prompt_copied_back_keeps_the_window_its_next_pass_reads() {
        // (a window of 64 and a chunk of 32 kept behind the newest token; a prompt of 208 tokens, 126 of them run in passes of 32)
        let mut p = Pages::new(&[200, 200], &[None, Some(64 + 32)], 2, 1024);
        assert!(p.grow_from(0, 0, 209, 32));
        for n in [32, 64, 96, 126] {
            assert!(p.grow_from(0, n, 209, 32));
        }
        let (entries, floor) = p.held_entries(0, 126);
        // (the next pass, tokens 126 to 158, reads from token 62: entry 3 on)
        assert!(entries[1].contains(&3) && entries[1].contains(&7));
        p.release(0);
        assert!(p.restore(1, &entries, &floor));
        assert!(p.grow_from(1, 126, 209, 32));
        assert!(entries[1].iter().filter(|&&b| b >= 3).all(|&b| p.table(1, 1)[b] != 0), "an entry the next pass reads has no block");
        // (the end's own window, tokens 113 on: entry 7 on, so the entries before it would go)
        let mut q = Pages::new(&[200, 200], &[None, Some(64 + 32)], 2, 1024);
        assert!(q.restore(0, &entries, &floor) && q.grow(0, 209));
        assert!(q.table(0, 1)[3..7].iter().all(|&b| b == 0));
    }

    /// A sequence swapped out and back (R4d): `held_entries` lists the entries it holds of its tokens' blocks (a sliding group's window alone, a shared start's floor), and `restore`
    /// gives an empty slot a block of its own for each of them and the same floors, so a step there reads what it did; where a group has too few blocks nothing changes.
    #[test]
    fn a_swapped_sequence_is_restored_to_the_entries_it_held() {
        let mut p = Pages::new(&[40, 40], &[None, Some(64 + 32)], 3, 1024);
        assert!(p.grow(0, 170));
        // (sequence 1 shares sequence 0's first 160 tokens: its sliding group holds from the window before them on)
        assert_eq!(p.share(0, 1, 160, 32), 160);
        assert!(p.grow(1, 170));
        let (entries, floor) = p.held_entries(1, 170);
        assert_eq!(entries[0], (0..11).collect::<Vec<_>>());
        assert_eq!((entries[1].clone(), floor.clone()), ((6..11).collect::<Vec<_>>(), vec![0, 6]));
        // (only the blocks of the tokens it holds: entries grown ahead are not listed)
        assert_eq!(p.held_entries(0, 160).0[0].len(), 10);
        p.release(1);
        assert!(p.restore(2, &entries, &floor));
        assert_eq!(p.held_entries(2, 170), (entries.clone(), floor.clone()));
        // its blocks are its own, and it grows as the sequence it was would have (no block before its floor)
        assert!((0..11).all(|b| p.table(2, 0)[b] != p.table(0, 0)[b]));
        assert!(p.grow(2, 200) && p.table(2, 1)[..6].iter().all(|&b| b == 0));
        p.release(2);
        // a slot that holds blocks, or pools too small, take nothing
        assert!(!p.restore(0, &entries, &floor));
        let mut small = Pages::new(&[8, 8], &[None, Some(64 + 32)], 2, 1024);
        let before = small.free(0);
        assert!(!small.restore(1, &entries, &floor) && small.free(0) == before && small.held_entries(1, 1024).0.iter().all(Vec::is_empty));
    }

    /// A group whose blocks cover 64 tokens (compressed entries of 4 tokens each: BLOCK of them a block) beside a full and a sliding group of BLOCK tokens: a sequence holds its
    /// blocks to the one its last token is in; its blocks are named for the blocks of BLOCK tokens that end them, so a start is found by name in whole ones of them; a start
    /// shared within one leaves that block to the caller; a cut into one another sequence holds is to its start.
    #[test]
    fn a_group_of_larger_blocks_holds_and_shares_them_whole() {
        // (a window of 256 tokens and a chunk of 32)
        let mut p = Pages::covering(&[60, 60, 9], &[None, Some(256 + 32), None], &[BLOCK, BLOCK, 4 * BLOCK], 3, 1024);
        assert_eq!(p.capacity(), Some(8 * 4 * BLOCK), "the compressed group's 8 blocks of 64 tokens bound one sequence");
        assert!(p.grow(0, 65));
        assert_eq!((p.held(0), p.held(2)), (5, 2));
        assert_eq!(p.held_entries(0, 65).0[2], vec![0, 1]);
        let keys = names(1, 1, 0, 1024);
        // (its full blocks named after each pass, as the scheduler does)
        for to in (80..=200).step_by(20) {
            assert!(p.grow(0, to));
            for b in 0..to / BLOCK {
                p.name(0, b, keys[b]);
            }
        }
        // (12 blocks of BLOCK tokens named: the compressed group's first three, which they end, too)
        assert_eq!(p.pools[2].index.len(), 3);
        assert_eq!(p.fit(&keys[..12], 32), 12);
        assert_eq!(p.fit(&keys[..11], 32), 8, "a start in whole blocks of 64 tokens");
        assert_eq!(p.adopt(1, &keys[..12], 32), 192);
        assert_eq!(p.table(1, 2)[..4], [p.table(0, 2)[0], p.table(0, 2)[1], p.table(0, 2)[2], 0]);
        // a cut of the first into the compressed block both hold: to its start, though the blocks of BLOCK tokens there are its own
        assert_eq!(p.cut_len(0, 150, 32), 128);
        assert_eq!(p.shrink(0, 150, 32), 128);
        assert!(p.grow(0, 150) && p.table(0, 2)[2] != p.table(1, 2)[2]);
        p.release(1);
        // a start shared at 100 tokens: the compressed group's whole block; its block of tokens 64 to 99 is the caller's to give
        assert_eq!(p.share(0, 1, 100, 32), 96);
        assert_eq!((p.partial(2, 96), p.partial(0, 96)), (Some(1), None));
        assert!(p.table(1, 2)[0] == p.table(0, 2)[0] && p.table(1, 2)[1] == 0);
        assert!(p.grow(1, 96) && p.table(1, 2)[1] != 0 && p.table(1, 2)[1] != p.table(0, 2)[1]);
    }

    /// R4d: the blocks of a sequence swapped out are given back with fences (the parts of the copy that still reads them). A block with a fence is taken after every free one
    /// that has none, a swapped sequence's first entries first, and says its fence when taken, though it was held by another sequence when the swap gave it up.
    #[test]
    fn blocks_a_copy_reads_are_taken_last_and_say_so() {
        let mut p = Pages::new(&[12], &[None], 3, 1024);
        assert!(p.grow(0, 100));
        let own: Vec<u32> = p.table(0, 0)[..7].to_vec();
        assert_eq!(p.share(0, 1, 64, 32), 64);
        // (the copy's first part reads entries 0 and 1, the second the rest)
        p.release_fenced(0, |_, b| if b < 2 { 10 } else { 11 });
        assert_eq!((p.free(0).0, p.take_fence()), (7, 0));
        // the four free blocks no copy reads first, then the swapped sequence's first entry that no one holds (4), whose fence is the second part's
        assert!(p.grow(2, 64) && p.take_fence() == 0);
        assert!(p.grow(2, 80) && p.take_fence() == 11 && p.table(2, 0)[4] == own[4]);
        // the blocks it shared keep their fences: given up by the other sequence, they are taken first after it, its first entry first
        p.release(1);
        assert!(p.grow(2, 96) && p.take_fence() == 10 && p.table(2, 0)[5] == own[0]);
        assert_eq!(p.take_fence(), 0, "said once");
    }

    /// The names of the full blocks of `n` tokens of a stream: stream `a` to `fork`, stream `b` after.
    fn names(a: u32, b: u32, fork: usize, n: usize) -> Vec<u128> {
        let toks: Vec<u32> = (0..n).map(|i| (if i < fork { a } else { b }) * 100_003 + i as u32).collect();
        crate::kv::block_names(crate::kv::block_root(0), &toks)
    }

    /// A sequence's full blocks are found by name after it has grown far past them, and after it has ended: every block of a full group, a sliding group's window behind
    /// the place. A prompt that parts from it midway finds what the two share. What a pool needs for something else it takes from the named blocks no one holds, the
    /// oldest first, and those are then not found.
    #[test]
    fn a_start_is_found_by_name_whoever_holds_it() {
        // (a window of 64 tokens and a chunk of 32)
        let mut p = Pages::new(&[60, 60], &[None, Some(64 + 32)], 3, 1024);
        let keys = names(1, 1, 0, 500);
        // (it grows by less than a chunk a time and its full blocks are named after each, as the scheduler does: what a sliding group gives back has its name by then)
        for to in (20..=500).step_by(20) {
            assert!(p.grow(0, to));
            for b in 0..to / BLOCK {
                p.name(0, b, keys[b]);
            }
        }
        // the sliding group gave the blocks before entry 25 back as it grew: named, they wait (nothing is free that was not before)
        assert_eq!((p.held(1), p.pools[1].idle.len(), p.pools[1].free.len()), (7, 25, 59 - 32));
        // the same tokens: all 31 full blocks, the sliding group's from the window before 496 on
        assert_eq!(p.adopt(1, &keys, 32), 496);
        assert_eq!(p.table(1, 0)[..31], p.table(0, 0)[..31]);
        assert!(p.table(1, 1)[..27].iter().all(|&b| b == 0) && p.table(1, 1)[27..31] == p.table(0, 1)[27..31]);
        p.release(1);
        // a prompt that parts from it at token 210: 13 blocks, and of the sliding group the window before 208 (blocks that no sequence held)
        let other = names(1, 2, 210, 400);
        assert_eq!(other[..13], keys[..13]);
        assert_eq!(p.adopt(1, &other, 32), 208);
        assert!(p.table(1, 1)[..9].iter().all(|&b| b == 0) && p.table(1, 1)[9..13].iter().all(|&b| b != 0));
        assert_eq!(p.pools[1].idle.len(), 25 - 4);
        // it goes on with its own tokens: entry 13 is a block of its own, and no block is given it before the window it found
        assert!(p.grow(1, 240));
        assert!(p.table(1, 0)[13] != p.table(0, 0)[13] && p.table(1, 1)[..9].iter().all(|&b| b == 0));
        p.release(1);
        // the first sequence ends: everything it held waits by name, and is found
        p.release(0);
        assert_eq!((p.held(0), p.held(1)), (0, 0));
        assert_eq!(p.adopt(2, &keys, 32), 496);
        p.release(2);
        // another sequence needs the pool: the full group's 59 blocks, 28 of them free and 31 waiting, of which it takes the oldest, the chain's end
        assert!(p.grow(1, 40 * BLOCK));
        assert_eq!((p.pools[0].free.len(), p.pools[0].idle.len()), (0, 19));
        assert_eq!(p.adopt(2, &keys, 32), 19 * BLOCK, "the start of the chain is what is left");
    }

    /// A named block is never written. A sequence cut back into a full block it alone holds takes the block's name away (it writes there next); cut into one that another
    /// holds it is cut to the block's start, and the block keeps its name.
    #[test]
    fn a_cut_into_a_named_block_takes_its_name_away() {
        let mut p = Pages::new(&[60, 60], &[None, Some(64 + 32)], 3, 1024);
        let keys = names(1, 1, 0, 96);
        assert!(p.grow(0, 96));
        (0..6).for_each(|b| p.name(0, b, keys[b]));
        assert_eq!(p.shrink(0, 70, 32), 70);
        // block 4 (tokens 64 to 79) has no name now: the same tokens are found to 64 (block 5, given back, waits by its name for a sequence that has all six)
        assert!((0..2).all(|g| p.pools[g].name[p.table(0, g)[4] as usize] == 0 && p.pools[g].index.len() == 5));
        assert_eq!(p.adopt(1, &keys, 32), 64);
        // the two hold block 3: a cut of the first into it is to its start, and it is still found
        assert_eq!(p.shrink(0, 60, 32), 48);
        p.release(1);
        assert_eq!(p.adopt(1, &keys, 32), 64);
        // the first goes on with other tokens from 48: block 3 is a new one of its own
        assert!(p.grow(0, 80) && p.table(0, 0)[3] != p.table(1, 0)[3]);
    }

    /// Sequences of a few token streams growing, being found by name, cut back and ending at random, with a record of what each block holds: a named block holds the tokens
    /// its name says and is never written; a block is free, or waits by its name, or is held, and never two of these; what is found by name is what was asked for. Group 2's
    /// blocks cover 64 tokens (compressed entries of 4 tokens each).
    #[test]
    fn named_blocks_hold_what_their_names_say() {
        let (slots, ctx, slack) = (6, 640, 32);
        let mut p = Pages::covering(&[70, 40, 30], &[None, Some(48 + slack), None], &[BLOCK, BLOCK, 4 * BLOCK], slots, ctx);
        let (groups, covers) = (3, [BLOCK, BLOCK, 4 * BLOCK]);
        // (a block's name: that of the block of BLOCK tokens that ends it)
        let last = |g: usize, b: usize| (b + 1) * covers[g] / BLOCK - 1;
        // each sequence: (stream to its fork, stream after, the fork), its length, and its blocks' names
        let mut id = vec![(0u32, 0u32, 0usize); slots];
        let mut len = vec![0usize; slots];
        let mut keys: Vec<Vec<u128>> = vec![Vec::new(); slots];
        // what each block holds: the name of the full block of tokens written there (None: not a full block's)
        let mut holds: Vec<Vec<Option<u128>>> = (0..groups).map(|g| vec![None; p.pools[g].blocks as usize]).collect();
        let (mut seed, mut next) = (0x9e37_79b9_7f4a_7c15u64, 10u32);
        let mut rnd = |m: usize| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed % m as u64) as usize
        };
        let (mut found, mut cuts, mut taken) = (0, 0, 0);
        // (under Miri, which hashes each block's name in software a thousand times slower, the first 60 of the same steps)
        let steps = if cfg!(miri) { 60 } else { 30000 };
        for _ in 0..steps {
            let s = rnd(slots);
            match rnd(10) {
                0 => {
                    p.release(s);
                    len[s] = 0;
                }
                1..=3 if len[s] == 0 => {
                    // a new sequence: one of two streams to a fork, its own after
                    next += 1;
                    id[s] = (rnd(2) as u32, next, rnd(500));
                    keys[s] = names(id[s].0, id[s].1, id[s].2, ctx);
                    let got = p.adopt(s, &keys[s][..rnd(ctx / BLOCK)], slack);
                    assert!(got % BLOCK == 0);
                    for g in 0..groups {
                        for (b, &blk) in p.table(s, g).iter().enumerate() {
                            assert!(
                                blk == 0 || (b * covers[g] < got && holds[g][blk as usize] == Some(keys[s][last(g, b)])),
                                "entry {b} of group {g} is not the block asked for"
                            );
                        }
                    }
                    found += (got > 0) as usize;
                    len[s] = got;
                }
                4 if len[s] > 0 => {
                    let got = p.shrink(s, 1 + rnd(len[s]), slack);
                    if got > 0 {
                        // (from there on it is another sequence)
                        next += 1;
                        id[s] = (id[s].0, next, id[s].2.min(got));
                        if id[s].2 < got {
                            // its tokens before the cut are of two streams: the names stay those of what it holds
                            let old = keys[s].clone();
                            keys[s] = old[..got / BLOCK].iter().copied().chain(std::iter::repeat_n(0, ctx / BLOCK - got / BLOCK)).collect();
                        } else {
                            keys[s] = names(id[s].0, id[s].1, id[s].2, ctx);
                        }
                        len[s] = got;
                        cuts += 1;
                    }
                }
                _ if len[s] > 0 || rnd(3) == 0 => {
                    if len[s] == 0 {
                        next += 1;
                        id[s] = (rnd(2) as u32, next, rnd(500));
                        keys[s] = names(id[s].0, id[s].1, id[s].2, ctx);
                    }
                    let to = (len[s] + 1 + rnd(40)).min(ctx);
                    let idle: usize = p.pools.iter().map(|q| q.idle.len()).sum();
                    if p.grow(s, to) {
                        taken += (p.pools.iter().map(|q| q.idle.len()).sum::<usize>() < idle) as usize;
                        // the tokens from its length to `to` are written: no block with a name among them
                        for g in 0..groups {
                            let c = covers[g];
                            for b in len[s] / c..to.div_ceil(c) {
                                let blk = p.table(s, g)[b] as usize;
                                assert!(
                                    blk != 0 && p.pools[g].name[blk] == 0 && p.pools[g].refs[blk] == 1,
                                    "a block written that has a name or another holder"
                                );
                                holds[g][blk] = (to >= (b + 1) * c && keys[s][last(g, b)] != 0).then_some(keys[s][last(g, b)]);
                            }
                        }
                        for b in len[s] / BLOCK..to / BLOCK {
                            p.name(s, b, keys[s][b]);
                        }
                        len[s] = to;
                    }
                }
                _ => {}
            }
            for g in 0..groups {
                let q = &p.pools[g];
                let mut count = vec![0u32; q.blocks as usize];
                for x in 0..slots {
                    for (b, &blk) in p.table(x, g).iter().enumerate() {
                        count[blk as usize] += (blk != 0) as u32;
                        assert!(blk == 0 || b * covers[g] < len[x], "sequence {x} of {} tokens holds entry {b}", len[x]);
                    }
                }
                assert_eq!(count, q.refs, "a block's count is not its holders");
                for blk in 1..q.blocks as usize {
                    let (free, idle) = (q.free.contains(&(blk as u32)), q.idle.values().any(|&b| b as usize == blk));
                    assert_eq!((free, idle), (q.refs[blk] == 0 && q.name[blk] == 0, q.refs[blk] == 0 && q.name[blk] != 0), "block {blk} of group {g}");
                    if q.name[blk] != 0 {
                        assert_eq!(
                            (q.index.get(&q.name[blk]), holds[g][blk]),
                            (Some(&(blk as u32)), Some(q.name[blk])),
                            "block {blk} of group {g} and its name"
                        );
                    }
                }
                assert_eq!(q.index.len(), q.name.iter().filter(|&&n| n != 0).count());
                assert_eq!(q.free.len() + q.idle.len() + q.refs.iter().filter(|&&r| r > 0).count(), q.blocks as usize - 1, "a block lost");
            }
        }
        // (the run did what it is for: as many of each for its steps as 300 in 30,000)
        assert!(
            [found, cuts, taken].iter().all(|&x| x * 30000 > 300 * steps),
            "{found} found, {cuts} cuts, {taken} growths that took a named block in {steps} steps"
        );
    }

    /// The room of a prompt of several windows, taken at once (`grow_from`), in pools of one sequence's: the sliding group's blocks are those its first pass reads, and every pass after
    /// grows into them, holding no more than the pool. A sequence that goes on by more than a chunk keeps the window its next pass reads. `grow` to the prompt's end takes the blocks
    /// of its last window instead, and its first pass then has none.
    #[test]
    fn a_prompt_room_holds_what_its_first_pass_reads() {
        // (a window of 64 tokens and a chunk of 32: 7 blocks of the sliding group, as `Dyn::page` sizes one sequence's)
        let pools = || Pages::new(&[34, 8], &[None, Some(64 + 32)], 1, 512);
        let mut p = pools();
        assert!(p.grow_from(0, 0, 301, 32));
        assert!(p.table(0, 1)[..7].iter().all(|&b| b != 0) && p.held(1) == 7);
        for at in (0..300).step_by(32) {
            assert!(p.grow_from(0, at, (at + 32).min(300), 32) && p.held(1) <= 7, "the pass at {at}");
        }
        // at 300, on to 400 at once: the window before 300 (entries 14 to 18) is in the same blocks
        let before = p.table(0, 1)[14..19].to_vec();
        assert!(p.grow_from(0, 300, 400, 32));
        assert_eq!(p.table(0, 1)[14..19], before[..]);
        let mut q = pools();
        assert!(q.grow(0, 301) && !q.grow(0, 32));
    }

    /// A sequence asked for room twice in a step (as its prompt is let in, then before its pass) is promised each page once; a later ask that reaches further adds
    /// only the pages past the first's.
    #[test]
    fn a_step_promises_a_sequences_pages_once() {
        // (a prompt of 40 tokens let in: pages 0, 1, 2)
        let (more, had) = pages_more(None, 0, 41);
        assert_eq!((more, had), (3, (0, 3)));
        // (asked again before its pass: nothing more)
        assert_eq!(pages_more(Some(had), 0, 41), (0, (0, 3)));
        // (a pass of fewer tokens: nothing more; one that reaches a page further: that page)
        assert_eq!(pages_more(Some(had), 0, 20), (0, (0, 3)));
        assert_eq!(pages_more(Some(had), 0, 49), (1, (0, 4)));
        // (a decode step in the middle of a page: its page; no token: none)
        assert_eq!(pages_more(None, 37, 38), (1, (2, 3)));
        assert_eq!(pages_more(None, 37, 37), (0, (2, 2)));
    }

    /// A made-up GPU's 12-bit pools: each raw pool's slabs and those its status counts as taken, the escape chunks likewise, the copies on the swap's stream that have not
    /// landed, in order (the slabs of each pool and the chunks each takes; a copy out's release gives back, less than nothing), and the slabs the named blocks no one holds keep.
    struct MadeUp {
        slabs: Vec<i64>,
        taken: Vec<i64>,
        chunks: i64,
        ctaken: i64,
        side: std::collections::VecDeque<(Vec<i64>, i64)>,
        idle: Vec<i64>,
    }

    impl MadeUp {
        fn new(slabs: &[i64], taken: &[i64]) -> MadeUp {
            MadeUp { slabs: slabs.to_vec(), taken: taken.to_vec(), chunks: 1024, ctaken: 0, side: Default::default(), idle: vec![0; slabs.len()] }
        }
        /// The swap's stream runs its first copy: the GPU does, whatever the host reads meanwhile.
        fn run(&mut self) {
            if let Some((s, c)) = self.side.pop_front() {
                self.taken.iter_mut().zip(&s).for_each(|(t, x)| *t += x);
                self.ctaken += c;
            }
        }
    }

    impl Pools for MadeUp {
        fn land(&mut self) -> Result<()> {
            while !self.side.is_empty() {
                self.run();
            }
            Ok(())
        }
        fn drop_idle(&mut self) -> Result<()> {
            self.taken.iter_mut().zip(&mut self.idle).for_each(|(t, i)| *t -= std::mem::take(i));
            Ok(())
        }
        fn free(&mut self) -> Result<(Vec<i64>, i64)> {
            Ok((self.slabs.iter().zip(&self.taken).map(|(s, t)| s - t).collect(), self.chunks - self.ctaken))
        }
    }

    /// A sequence copied back is counted once, by the pools, whenever its copy lands: before the plan reads them, between the plan and the step's own reading, or after
    /// both. Gemma 4's KV-sharing case as an A10 ran it, every page raw: its sliding rings' raw pool of 64 slabs that its 3 rings draw on, the full ring's of 70; a
    /// sequence at 133 tokens holds 7 sliding pages and 9 full ones (21 and 9 slabs), one copied back 21 and 10. The plan's decode row (a page: 3 slabs and 1) is had
    /// at the step's own reading too (counted twice, the copy left 1 slab of the 3: the engine ended "no block left for token 134"); a growth after the copy landed is had
    /// (no sequence paused for room that is there); a second copy back of 21 sliding slabs is had (22 free, not 1); and where the pools are short a refusal waits for
    /// nothing in flight: a copy out's release on the swap's stream, the named blocks no one holds.
    #[test]
    fn a_copy_back_is_counted_once_whenever_it_lands() {
        let (held, back) = ([21i64, 9], [21i64, 10]);
        for lands in 0..3 {
            let mut g = MadeUp::new(&[64, 70], &held);
            let mut l = Ledger::new(vec![3, 1]);
            assert!(l.copy_back(&mut g, 1, &back, 0).unwrap(), "the copy back's room");
            g.side.push_back((back.to_vec(), 0));
            if lands == 0 {
                g.run();
            }
            let plan = l.counted(&mut g).unwrap();
            assert!(l.grow(&mut g, 0, 133, 134).unwrap(), "copy landed {lands}: the plan's decode row");
            if lands == 1 {
                g.run();
            }
            // (the step's own reading, as a pass outside the scheduler asks: its step's promises given up first)
            l.stepped();
            let hold = l.counted(&mut g).unwrap();
            assert_eq!(hold, plan, "copy landed {lands}: the plan's count and the step's");
            assert_eq!(hold.0, [64 - 21 - 21, 70 - 9 - 10], "copy landed {lands}: each page counted once, where it is");
            assert!(l.grow(&mut g, 0, 133, 134).unwrap(), "copy landed {lands}: the step's own reading");
            // (a second copy back of a sequence of the same pages, the first's copy landed or not)
            l.stepped();
            if lands == 2 {
                g.run();
            }
            assert!(l.copy_back(&mut g, 2, &[21, 4], 0).unwrap(), "copy landed {lands}: a second copy back, 22 slabs free");
        }
        // (short: a copy out's release on the swap's stream, then the named blocks no one holds, make the room; truly short, no)
        let mut g = MadeUp::new(&[64, 70], &[63, 0]);
        g.side.push_back((vec![-21, 0], 0));
        assert!(Ledger::new(vec![3, 1]).grow(&mut g, 0, 133, 134).unwrap() && g.side.is_empty(), "a copy out's release");
        let mut g = MadeUp::new(&[64, 70], &[63, 0]);
        g.idle = vec![6, 0];
        assert!(Ledger::new(vec![3, 1]).grow(&mut g, 0, 40, 41).unwrap() && g.idle == [0, 0], "the named blocks no one holds");
        let mut g = MadeUp::new(&[64, 70], &[62, 0]);
        assert!(!Ledger::new(vec![3, 1]).grow(&mut g, 0, 40, 41).unwrap(), "2 slabs free of the 3 a page takes");
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
        // (group 2: blocks of 64 tokens, compressed entries')
        let mut p = Pages::covering(&[60, 30, 20], &[None, Some(48 + slack), None], &[BLOCK, BLOCK, 4 * BLOCK], slots, ctx);
        let (groups, covers) = (3, [BLOCK, BLOCK, 4 * BLOCK]);
        let mut len = vec![0usize; slots];
        let mut seed = 0x2545_f491_4f6c_dd1du64;
        let mut rnd = |m: usize| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed % m as u64) as usize
        };
        let (mut shares, mut cuts) = (0, 0);
        // (under Miri, which interprets each step a thousand times slower, the first 400 of the same sequence of steps)
        let steps = if cfg!(miri) { 400 } else { 20000 };
        for _ in 0..steps {
            let s = rnd(slots);
            let before: Vec<Vec<Vec<u32>>> = (0..slots).map(|s| (0..groups).map(|g| p.table(s, g).to_vec()).collect()).collect();
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
                        assert!((0..slots).all(|x| (0..groups).all(|g| p.table(x, g) == &before[x][g][..])));
                    }
                    shares += (got > 0) as usize;
                    len[s] = got;
                }
                3 if len[s] > 0 => {
                    let to = 1 + rnd(len[s]);
                    let got = p.shrink(s, to, slack);
                    // (to the start of a block others hold too, of BLOCK tokens or of 64)
                    assert!(got == 0 || got == to || got == to / BLOCK * BLOCK || got == to / 64 * 64, "{got} for {to}");
                    if got == 0 {
                        assert!((0..groups).all(|g| p.table(s, g) == &before[s][g][..]));
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
                        assert!((0..groups).all(|g| p.table(s, g) == &before[s][g][..]));
                    }
                }
            }
            for g in 0..groups {
                let mut named = vec![0u32; p.pools[g].blocks as usize];
                for x in 0..slots {
                    for (b, &blk) in p.table(x, g).iter().enumerate() {
                        if blk != 0 {
                            named[blk as usize] += 1;
                            assert!(b * covers[g] < len[x], "sequence {x} of {} tokens holds entry {b}", len[x]);
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
                for g in 0..groups {
                    for (b, &blk) in p.table(x, g).iter().enumerate() {
                        assert!(
                            blk == 0 || p.pools[g].refs[blk as usize] == 1 || (b + 1) * covers[g] <= len[x],
                            "sequence {x} would write a block others hold"
                        );
                    }
                }
            }
        }
        // (the run did what it is for: as many shares and cuts for its steps as 100 and 200 in 20,000)
        assert!(shares * 20000 > 100 * steps && cuts * 20000 > 200 * steps, "{shares} shares, {cuts} cuts in {steps} steps");
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
