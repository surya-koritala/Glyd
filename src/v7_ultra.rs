//! The ultra level's parse: every position priced, the cheapest path
//! through the block taken.
//!
//! A binary-tree finder (positions with the same 4-byte hash kept in a
//! tree sorted by their suffix, over the 2 MB window, plus a 3-byte head
//! for the shortest matches) lists, at each position, the three repeat
//! offsets and every tree match longer than the ones before it -- the
//! tree is walked from the newest position down, so each match found is
//! farther back than the last and worth listing only if longer. A
//! forward dynamic program over the block then keeps, for each position,
//! the cheapest way to arrive there -- by one more literal, or by a match
//! of any length from any listed candidate -- with the repeat-offset
//! state that path carries, so a rep match is priced as the rep code it
//! would actually get. Prices are the coder's own: literal bytes, length
//! codes, offset codes and extra bits, in 1/256 bit, from the previous
//! block's statistics (the first block is parsed twice: once on flat
//! prices to gather statistics, then on those). The back-trace from the
//! block's end yields the sequences.
use crate::finder::{MatchLen, ScalarMatch};
use crate::format::MAX_BLOCK_SIZE;
use crate::v7_encode::Sequence;
use crate::v7_format::*;

const HASH4_BITS: u32 = 20;
const HASH3_BITS: u32 = 16;
/// Tree nodes compared per position, searching or inserting.
const DEPTH: usize = 64;
const NONE: u32 = u32::MAX;
/// A match this long is taken whole: no other candidate at its position
/// is priced and no position inside it is searched.
const SUFFICIENT_LEN: usize = 256;
/// Price unit: 1/256 bit.
const BIT: u32 = 256;
const PRIOR_WEIGHT: u32 = 2;
/// Added to every literal's price. The dynamic program prices each
/// decision at the current code frequencies, which is exact to first
/// order but blind to the regime it creates: a parse that turns literals
/// into short matches makes every short match and every ll = 0 cheaper,
/// which the fixed prices never credit, so the exact model settles on a
/// literal-heavy parse (dickens: 3.2x the literal bytes of zstd -19's).
/// Half a bit per literal nudges it over; on Silesia 0/0.5/1.0/1.5 bits
/// give 3.941/3.946/3.944/3.941, at 0.5-1.5% decode for the extra short
/// matches.
const LIT_SURCHARGE: u32 = BIT / 2;

#[derive(Clone, Copy)]
struct Node {
    price: u32,
    /// 0: reached by a literal, `litlen` of them in the run so far.
    mlen: u32,
    off: u32,
    litlen: u32,
    reps: [u32; 3],
}

const UNREACHED: Node = Node { price: u32::MAX, mlen: 0, off: 0, litlen: 0, reps: [0; 3] };

/// Code prices in 1/256 bit.
struct Prices {
    lit: [u32; 256],
    ll: [u32; LL_SYMBOLS],
    ml: [u32; ML_SYMBOLS],
    off: [u32; OFF_SYMBOLS],
}

/// Starting code counts, 4096 per table, from this parse's own output
/// over Silesia (`examples/prior_stats.rs` with ULTRA=1), counted
/// `PRIOR_WEIGHT` times in every block's prices, so a code the recent
/// blocks did not use keeps a price the parse can afford to try again;
/// without that, a block parsed on prices from a sparse block gets
/// sparser (an unused code costs log2(total) bits) and never recovers.
/// Weight 2 measured 0.13% over 1 on Silesia, 4 the same as 2, 1/4 -0.5%.
const PRIOR_LL: [u32; LL_SYMBOLS] = [1863, 1245, 423, 186, 91, 84, 58, 26, 25, 24, 16, 7, 3, 6, 14, 9, 4, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1];
const PRIOR_ML: [u32; ML_SYMBOLS] = [339, 204, 630, 279, 409, 379, 123, 231, 200, 169, 52, 137, 198, 24, 32, 54, 28, 17, 64, 20, 40, 10, 24, 28, 24, 12, 8, 61, 9, 7, 4, 18, 9, 28, 9, 9, 37, 12, 33, 55, 21, 11, 11, 13, 11, 2, 1, 1, 1, 1, 1, 1, 1, 1];
const PRIOR_OFF: [u32; OFF_SYMBOLS] = [554, 178, 82, 10, 2, 4, 19, 76, 89, 102, 107, 106, 108, 109, 132, 139, 158, 184, 213, 238, 267, 308, 305, 263, 204, 137];

impl Prices {
    /// Prices from the prior plus `s` (the recent blocks' counts), with
    /// `lit` standing in for the literal counts when `s` has none.
    fn of(s: &Stats, lit: &[u32; 256]) -> Prices {
        let mut ll = PRIOR_LL.map(|c| c * PRIOR_WEIGHT);
        let mut ml = PRIOR_ML.map(|c| c * PRIOR_WEIGHT);
        let mut off = PRIOR_OFF.map(|c| c * PRIOR_WEIGHT);
        for (a, b) in ll.iter_mut().zip(s.ll.iter()) {
            *a += b;
        }
        for (a, b) in ml.iter_mut().zip(s.ml.iter()) {
            *a += b;
        }
        for (a, b) in off.iter_mut().zip(s.off.iter()) {
            *a += b;
        }
        let mut litc = costs(if s.lit.iter().any(|&c| c > 0) { &s.lit } else { lit });
        for c in litc.iter_mut() {
            *c += LIT_SURCHARGE;
        }
        Prices { lit: litc, ll: costs(&ll), ml: costs(&ml), off: costs(&off) }
    }

    #[inline(always)]
    fn ll(&self, n: u32) -> u32 {
        let (c, nb, _) = ll_code(n);
        self.ll[c as usize] + nb as u32 * BIT
    }

    #[inline(always)]
    fn ml(&self, n: u32) -> u32 {
        let (c, nb, _) = ml_code(n);
        self.ml[c as usize] + nb as u32 * BIT
    }

    #[inline(always)]
    fn off(&self, off: u32, reps: &[u32; 3]) -> u32 {
        if off == reps[0] {
            self.off[0]
        } else if off == reps[1] {
            self.off[1]
        } else if off == reps[2] {
            self.off[2]
        } else {
            let k = 31 - off.leading_zeros();
            self.off[3 + k as usize] + k * BIT
        }
    }
}

/// -log2(p) of each symbol in 1/256 bit, add-one smoothed, at least one
/// bit: no prefix code spends less on a symbol.
fn costs<const N: usize>(hist: &[u32; N]) -> [u32; N] {
    let total: f64 = hist.iter().map(|&c| c as f64 + 1.0).sum();
    let lt = total.log2();
    let mut out = [0u32; N];
    for (o, &c) in out.iter_mut().zip(hist.iter()) {
        *o = (((lt - (c as f64 + 1.0).log2()) * BIT as f64).round() as u32).max(BIT);
    }
    out
}

/// Symbol counts: one block's, or the recent blocks' with each block
/// weighing half the next.
#[derive(Clone)]
pub struct Stats {
    lit: [u32; 256],
    ll: [u32; LL_SYMBOLS],
    ml: [u32; ML_SYMBOLS],
    off: [u32; OFF_SYMBOLS],
}

impl Stats {
    fn of(seqs: &[Sequence], literals: &[u8], reps0: [u32; 3]) -> Stats {
        let mut s = Stats { lit: [0; 256], ll: [0; LL_SYMBOLS], ml: [0; ML_SYMBOLS], off: [0; OFF_SYMBOLS] };
        for &b in literals {
            s.lit[b as usize] += 1;
        }
        let mut reps = reps0;
        for q in seqs {
            s.ll[ll_code(q.lit_len).0 as usize] += 1;
            if q.match_len == 0 {
                continue;
            }
            s.ml[ml_code(q.match_len).0 as usize] += 1;
            let (code, next) = rep_code(q.offset, &reps);
            s.off[code as usize] += 1;
            reps = next;
        }
        s
    }

    fn none() -> Stats {
        Stats { lit: [0; 256], ll: [0; LL_SYMBOLS], ml: [0; ML_SYMBOLS], off: [0; OFF_SYMBOLS] }
    }

    /// Halve these counts and add a block's.
    fn decay_into(&mut self, block: &Stats) {
        for (a, b) in self.lit.iter_mut().zip(block.lit.iter()) {
            *a = *a / 2 + b;
        }
        for (a, b) in self.ll.iter_mut().zip(block.ll.iter()) {
            *a = *a / 2 + b;
        }
        for (a, b) in self.ml.iter_mut().zip(block.ml.iter()) {
            *a = *a / 2 + b;
        }
        for (a, b) in self.off.iter_mut().zip(block.off.iter()) {
            *a = *a / 2 + b;
        }
    }
}

/// The repeat-offset code an offset gets and the state after it: the
/// decoder's `Reps::update`.
#[inline(always)]
fn rep_code(off: u32, r: &[u32; 3]) -> (u8, [u32; 3]) {
    if off == r[0] {
        (0, *r)
    } else if off == r[1] {
        (1, [r[1], r[0], r[2]])
    } else if off == r[2] {
        (2, [r[2], r[0], r[1]])
    } else {
        (3, [off, r[0], r[1]])
    }
}

#[inline(always)]
fn h4(p: *const u8, shift: u32) -> usize {
    let w = unsafe { std::ptr::read_unaligned(p as *const u32) };
    (w.wrapping_mul(0x9E37_79B1) >> shift) as usize
}

#[inline(always)]
fn h3(p: *const u8) -> usize {
    let w = unsafe { std::ptr::read_unaligned(p as *const u32) } & 0x00FF_FFFF;
    (w.wrapping_mul(0x9E37_79B1) >> (32 - HASH3_BITS)) as usize
}

/// The finder's tables and the parse's work arrays, allocated once per
/// thread and sized per call to the input (`clear`): up to 4 MB of heads
/// and 64 MB of tree for inputs that fill the 8 MB window, a few MB for
/// a 256 KB chunk.
pub struct UltraState {
    hash4: Vec<u32>,
    /// Bits `h4` keeps: 12 to `HASH4_BITS` by input size.
    hash4_shift: u32,
    hash3: Box<[u32]>,
    /// Per ring slot, the position's two children: the newest older
    /// position whose suffix sorts below it, and above it. The ring holds
    /// the window, or the whole input when that is smaller, so no two
    /// positions in reach of each other share a slot.
    tree: Vec<u32>,
    ring_mask: usize,
    /// Positions below this are in the tables.
    inserted: usize,
    opt: Vec<Node>,
    cands: Vec<(u32, u32)>,
    stats: Option<Stats>,
    /// Table writes of the first block's first pass: (hash4 index, old
    /// head, hash3 index, old head) per position, undone before its
    /// second pass.
    undo: Vec<[u32; 4]>,
}

impl UltraState {
    pub fn new() -> Box<Self> {
        Box::new(UltraState {
            hash4: Vec::new(),
            hash4_shift: 32 - 12,
            hash3: vec![NONE; 1 << HASH3_BITS].into_boxed_slice(),
            tree: Vec::new(),
            ring_mask: 0,
            inserted: 0,
            opt: Vec::new(),
            cands: Vec::new(),
            stats: None,
            undo: Vec::new(),
        })
    }

    /// Forget everything and size the tables for an input of `len` bytes:
    /// the output then depends only on the input.
    pub fn clear(&mut self, len: usize) {
        let ring = len.min(MAX_WINDOW as usize).max(1 << 16).next_power_of_two();
        let bits = (usize::BITS - len.max(1).leading_zeros()).clamp(12, HASH4_BITS);
        self.hash4.clear();
        self.hash4.resize(1 << bits, NONE);
        self.hash4_shift = 32 - bits;
        self.hash3.fill(NONE);
        self.tree.clear();
        self.tree.resize(2 * ring, NONE);
        self.ring_mask = ring - 1;
        self.inserted = 0;
        self.stats = None;
    }

    /// Insert positions `inserted..end` (the last three bytes of the
    /// input are never hashed: a 4-byte load must fit). With `log`, the
    /// first block's first pass, the writes are logged for `undo`.
    fn insert_upto(&mut self, input: &[u8], end: usize, log: bool) {
        let end = end.min(input.len().saturating_sub(3));
        while self.inserted < end {
            let pos = self.inserted;
            let (_, reach) = self.walk(input, pos, pos, log, false);
            // Positions covered by a match found here have their twins in
            // the tree already: skip to its last 8 bytes.
            self.inserted = pos + reach.saturating_sub(pos + 8).max(1);
        }
    }

    /// Take the logged insertions back out, newest first: the heads are
    /// restored and the block's tree slots cut, so the trees are what
    /// they were (nodes linked under a block position were only ever
    /// reached through it).
    fn undo(&mut self) {
        let n = self.undo.len();
        for (k, e) in self.undo.drain(..).rev().enumerate() {
            let pos = self.inserted - 1 - k;
            self.tree[2 * (pos & self.ring_mask)] = NONE;
            self.tree[2 * (pos & self.ring_mask) + 1] = NONE;
            self.hash4[e[0] as usize] = e[1];
            self.hash3[e[2] as usize] = e[3];
        }
        self.inserted -= n;
    }

    /// One tree walk from `pos`, which becomes its bucket's root: the
    /// walk compares `pos` with the positions on its path, links each
    /// under the right side of the one before, and, with `collect`,
    /// pushes every match longer than the last to `cands` (lengths
    /// clipped to `limit`). Suffixes are ordered over the whole input so
    /// the trees stay consistent across blocks. Positions older than the
    /// window are not followed: their slots belong to newer positions.
    /// Returns the longest usable length and the furthest end of any
    /// match seen (`insert_upto` skips the positions under it).
    fn walk(&mut self, input: &[u8], pos: usize, limit: usize, log: bool, collect: bool) -> (usize, usize) {
        let src = input.as_ptr();
        let cur = unsafe { src.add(pos) };
        let full = input.len() - pos;
        let (i4, i3) = (h4(cur, self.hash4_shift), h3(cur));
        if log {
            self.undo.push([i4 as u32, self.hash4[i4], i3 as u32, self.hash3[i3]]);
        }
        let mut m = self.hash4[i4];
        self.hash4[i4] = pos as u32;
        self.hash3[i3] = pos as u32;
        let low = pos.saturating_sub(MAX_WINDOW as usize - 1);
        let (mut sp, mut lp) = (2 * (pos & self.ring_mask), 2 * (pos & self.ring_mask) + 1);
        let (mut cls, mut cll) = (0usize, 0usize);
        let mut best = if collect { self.cands.last().map_or(MIN_MATCH as usize - 1, |c| c.0 as usize) } else { 0 };
        let mut reach = pos;
        let mut n = DEPTH;
        while n > 0 && m != NONE && (m as usize) < pos && (m as usize) >= low {
            n -= 1;
            let mi = m as usize;
            let node = 2 * (mi & self.ring_mask);
            let c = unsafe { src.add(mi) };
            let mut ml = cls.min(cll);
            ml += unsafe { ScalarMatch::prefix(cur.add(ml), c.add(ml), full - ml) };
            reach = reach.max(mi + ml);
            debug_assert_eq!(unsafe { ScalarMatch::prefix(cur, c, full) }, ml, "tree order broken at {pos}, node {mi}");
            if collect {
                let usable = ml.min(limit - pos);
                if usable > best {
                    self.cands.push((usable as u32, (pos - mi) as u32));
                    best = usable;
                }
            }
            if ml == full {
                // Equal to the end of the input: no order, and nothing
                // below this node can be placed.
                break;
            }
            if unsafe { *c.add(ml) < *cur.add(ml) } {
                // Below the current position: linked on the smaller side,
                // the walk continues into its larger subtree.
                self.tree[sp] = m;
                cls = ml;
                if mi <= low {
                    // At the window's edge: its children are not followed.
                    self.tree[lp] = NONE;
                    return (best, reach);
                }
                sp = node + 1;
                m = self.tree[node + 1];
            } else {
                self.tree[lp] = m;
                cll = ml;
                if mi <= low {
                    self.tree[sp] = NONE;
                    return (best, reach);
                }
                lp = node;
                m = self.tree[node];
            }
        }
        // Whatever was below the last node visited is dropped from the
        // tree: both open slots are closed.
        self.tree[sp] = NONE;
        self.tree[lp] = NONE;
        (best, reach)
    }

    /// Candidates at `pos`, into `cands` in the order found: the reps
    /// (any length), the 3-byte head, then the tree's, each longer than
    /// the last. Inserts `pos`. Returns the longest length.
    fn candidates(&mut self, input: &[u8], pos: usize, limit: usize, log: bool, reps: &[u32; 3]) -> usize {
        self.cands.clear();
        let max = limit - pos;
        if max < MIN_MATCH as usize {
            return 0;
        }
        let src = input.as_ptr();
        let cur = unsafe { src.add(pos) };
        let mut best = MIN_MATCH as usize - 1;
        for &r in reps {
            if r as usize <= pos && r >= 1 {
                let len = unsafe { ScalarMatch::prefix(cur, src.add(pos - r as usize), max) };
                if len >= MIN_MATCH as usize {
                    self.cands.push((len as u32, r));
                    best = best.max(len);
                }
            }
        }
        if input.len() - pos < 4 {
            return best;
        }
        if best < SUFFICIENT_LEN {
            let p3 = self.hash3[h3(cur)];
            if p3 != NONE && (p3 as usize) < pos && (pos - p3 as usize) < MAX_WINDOW as usize {
                let len = unsafe { ScalarMatch::prefix(cur, src.add(p3 as usize), max) };
                if len > best {
                    self.cands.push((len as u32, (pos - p3 as usize) as u32));
                    best = len;
                }
            }
        }
        // The tree is walked (and `pos` inserted) even when a rep is
        // already sufficient: later positions need it in the tree.
        if best >= SUFFICIENT_LEN {
            self.walk(input, pos, limit, log, false);
        } else {
            best = best.max(self.walk(input, pos, limit, log, true).0);
        }
        self.inserted = self.inserted.max(pos + 1);
        best
    }
}

/// Parse `input[block_start..block_start + block_len]` into `seqs` (the
/// last one literal-only) and `literals`, with `input[..block_start]` as
/// the window. `reps` is the block's initial repeat-offset state.
pub fn find_sequences_ultra(input: &[u8], block_start: usize, block_len: usize, st: &mut UltraState, reps: [u32; 3], seqs: &mut Vec<Sequence>, literals: &mut Vec<u8>) {
    assert!(block_len <= MAX_BLOCK_SIZE && block_start + block_len <= input.len());
    let block = &input[block_start..block_start + block_len];
    match st.stats.take() {
        Some(mut s) => {
            let prices = Prices::of(&s, &[0; 256]);
            parse(input, block_start, block_len, st, reps, &prices, false, seqs, literals);
            s.decay_into(&Stats::of(seqs, literals, reps));
            st.stats = Some(s);
        }
        None => {
            // First block: the prior and the block's own byte frequencies,
            // then again on what that parse produced.
            let mut hist = [0u32; 256];
            for &b in block {
                hist[b as usize] += 1;
            }
            parse(input, block_start, block_len, st, reps, &Prices::of(&Stats::none(), &hist), true, seqs, literals);
            let prices = Prices::of(&Stats::of(seqs, literals, reps), &hist);
            st.undo();
            parse(input, block_start, block_len, st, reps, &prices, false, seqs, literals);
            st.stats = Some(Stats::of(seqs, literals, reps));
        }
    }
}

fn parse(input: &[u8], block_start: usize, block_len: usize, st: &mut UltraState, reps: [u32; 3], prices: &Prices, log: bool, seqs: &mut Vec<Sequence>, literals: &mut Vec<u8>) {
    let block_end = block_start + block_len;
    st.opt.clear();
    st.opt.resize(block_len + 1, UNREACHED);
    st.opt[0] = Node { price: 0, mlen: 0, off: 0, litlen: 0, reps };
    st.insert_upto(input, block_start, false);
    let ll0 = prices.ll(0);
    let mut skip_until = 0usize;
    for cur in 0..block_len {
        let n = st.opt[cur];
        debug_assert!(n.price != u32::MAX);
        // One more literal.
        {
            let litlen = if n.mlen == 0 { n.litlen + 1 } else { 1 };
            let price = (n.price as i64 + prices.lit[input[block_start + cur] as usize] as i64 + prices.ll(litlen) as i64 - prices.ll(litlen - 1) as i64).max(0) as u32;
            if price < st.opt[cur + 1].price {
                st.opt[cur + 1] = Node { price, mlen: 0, off: 0, litlen, reps: n.reps };
            }
        }
        if cur < skip_until {
            continue;
        }
        let pos = block_start + cur;
        st.insert_upto(input, pos, log);
        let best = st.candidates(input, pos, block_end, log, &n.reps);
        if best < MIN_MATCH as usize {
            continue;
        }
        let base = n.price + ll0;
        if best >= SUFFICIENT_LEN {
            let &(len, off) = st.cands.iter().find(|c| c.0 as usize == best).unwrap();
            let price = base + prices.ml(len) + prices.off(off, &n.reps);
            let to = cur + len as usize;
            if price < st.opt[to].price {
                st.opt[to] = Node { price, mlen: len, off, litlen: 0, reps: rep_code(off, &n.reps).1 };
            }
            skip_until = to;
            continue;
        }
        // Reps price every length; a chain candidate only the lengths
        // past the previous one (it has the larger offset).
        let mut from = MIN_MATCH;
        for i in 0..st.cands.len() {
            let (len, off) = st.cands[i];
            let is_rep = off == n.reps[0] || off == n.reps[1] || off == n.reps[2];
            let lo = if is_rep { MIN_MATCH } else { from };
            let offp = prices.off(off, &n.reps);
            let next_reps = rep_code(off, &n.reps).1;
            for l in lo..=len {
                let price = base + prices.ml(l) + offp;
                let to = cur + l as usize;
                if price < st.opt[to].price {
                    st.opt[to] = Node { price, mlen: l, off, litlen: 0, reps: next_reps };
                }
            }
            if !is_rep {
                from = len + 1;
            }
        }
    }
    // Back-trace.
    seqs.clear();
    literals.clear();
    let mut cur = block_len;
    let mut trailing = 0usize;
    {
        let n = st.opt[cur];
        if n.mlen == 0 {
            trailing = n.litlen as usize;
            cur -= trailing;
        }
    }
    while cur > 0 {
        let n = st.opt[cur];
        debug_assert!(n.mlen > 0);
        let start = cur - n.mlen as usize;
        let ll = st.opt[start].litlen as usize;
        seqs.push(Sequence { lit_len: ll as u32, match_len: n.mlen, offset: n.off });
        cur = start - ll;
    }
    seqs.reverse();
    let mut at = block_start;
    for q in seqs.iter() {
        literals.extend_from_slice(&input[at..at + q.lit_len as usize]);
        at += q.lit_len as usize + q.match_len as usize;
    }
    debug_assert_eq!(at + trailing, block_end);
    literals.extend_from_slice(&input[at..block_end]);
    seqs.push(Sequence { lit_len: trailing as u32, match_len: 0, offset: 0 });
    st.insert_upto(input, block_end, log);
}

/// Order-0 cost in bits of coding `hist` with its own table.
fn entropy_bits<const N: usize>(hist: &[u32; N]) -> f64 {
    let total: u32 = hist.iter().sum();
    if total == 0 {
        return 0.0;
    }
    let t = total as f64;
    hist.iter().filter(|&&c| c > 0).map(|&c| c as f64 * (t / c as f64).log2()).sum()
}

/// Bytes a block spends beyond its symbols: framing (header, sub-header,
/// section size tables and paddings) and the entropy tables it writes.
const BLOCK_OVERHEAD_BITS: f64 = 8.0 * (32.0 + 26.0 + 5.0 * 29.0 + 90.0 + 90.0);
/// Candidate cuts are tried every this many sequences.
const SPLIT_STEP: usize = 256;
/// A part is never shorter than this much output: every block costs the
/// decoder its tables and tails (a 256 KB block decodes in ~120 us, a
/// block's fixed work is a few us), so a cut must be worth well over its
/// bytes.
const MIN_PART: usize = 48 * 1024;

/// Where a block's parse should be cut into blocks with their own
/// tables: the sequence indices at which new blocks start, in order, or
/// none. A cut is taken when the two parts coded on their own statistics
/// are cheaper than the whole coded on its by more than four times a
/// block's overhead, and both parts hold at least `MIN_PART` bytes; each
/// part is then considered again. The margin buys decode speed: on
/// Silesia, cutting at twice the overhead gains 0.33% ratio for 2.5%
/// decode, at four times 0.15% for 0.4%.
pub fn split_points(seqs: &[Sequence], literals: &[u8]) -> Vec<usize> {
    let mut cuts = Vec::new();
    let (mut lit_at, mut pos) = (0usize, 0usize);
    // Literal start and output position of each sequence, once.
    let (lit_starts, positions): (Vec<usize>, Vec<usize>) = seqs
        .iter()
        .map(|q| {
            let s = (lit_at, pos);
            lit_at += q.lit_len as usize;
            pos += (q.lit_len + q.match_len) as usize;
            s
        })
        .unzip();
    let end_pos = |b: usize| if b < seqs.len() { positions[b] } else { pos };
    let mut spans = vec![(0usize, seqs.len())];
    while let Some((from, to)) = spans.pop() {
        if to - from < 2 * SPLIT_STEP || end_pos(to) - positions[from] < 2 * MIN_PART {
            continue;
        }
        let lit_range = |a: usize, b: usize| lit_starts[a]..if b < seqs.len() { lit_starts[b] } else { literals.len() };
        let whole = cost(&seqs[from..to], &literals[lit_range(from, to)]);
        let mut best = (whole - 4.0 * BLOCK_OVERHEAD_BITS, 0usize);
        // Prefix statistics grow as the cut moves; the suffix's are the
        // whole's minus the prefix's.
        let mut pre = Stats::none();
        let all = stats_of(&seqs[from..to], &literals[lit_range(from, to)]);
        let mut k = from;
        while k + SPLIT_STEP <= to - SPLIT_STEP {
            add_stats(&mut pre, &seqs[k..k + SPLIT_STEP], &literals[lit_range(k, k + SPLIT_STEP)]);
            k += SPLIT_STEP;
            if positions[k] - positions[from] < MIN_PART || end_pos(to) - positions[k] < MIN_PART {
                continue;
            }
            let suf = sub_stats(&all, &pre);
            let c = stats_cost(&pre) + stats_cost(&suf);
            if c < best.0 {
                best = (c, k);
            }
        }
        if best.1 != 0 {
            cuts.push(best.1);
            spans.push((from, best.1));
            spans.push((best.1, to));
        }
    }
    cuts.sort_unstable();
    cuts
}

fn stats_of(seqs: &[Sequence], literals: &[u8]) -> Stats {
    let mut s = Stats::none();
    add_stats(&mut s, seqs, literals);
    s
}

/// Add a run of sequences' codes (offsets as if every one were a fresh
/// offset: rep codes depend on order, and a split resets them anyway) and
/// literal bytes.
fn add_stats(s: &mut Stats, seqs: &[Sequence], literals: &[u8]) {
    for &b in literals {
        s.lit[b as usize] += 1;
    }
    for q in seqs {
        s.ll[ll_code(q.lit_len).0 as usize] += 1;
        if q.match_len == 0 {
            continue;
        }
        s.ml[ml_code(q.match_len).0 as usize] += 1;
        let k = 31 - q.offset.max(1).leading_zeros();
        s.off[3 + k as usize] += 1;
    }
}

fn sub_stats(a: &Stats, b: &Stats) -> Stats {
    let mut s = a.clone();
    for (x, y) in s.lit.iter_mut().zip(b.lit.iter()) {
        *x -= y;
    }
    for (x, y) in s.ll.iter_mut().zip(b.ll.iter()) {
        *x -= y;
    }
    for (x, y) in s.ml.iter_mut().zip(b.ml.iter()) {
        *x -= y;
    }
    for (x, y) in s.off.iter_mut().zip(b.off.iter()) {
        *x -= y;
    }
    s
}

/// Order-0 coding cost of the codes in `s` (extra bits are the same
/// however the block is cut and are left out).
fn stats_cost(s: &Stats) -> f64 {
    entropy_bits(&s.lit) + entropy_bits(&s.ll) + entropy_bits(&s.ml) + entropy_bits(&s.off)
}

fn cost(seqs: &[Sequence], literals: &[u8]) -> f64 {
    stats_cost(&stats_of(seqs, literals))
}
