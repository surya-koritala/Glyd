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
const TREE_MASK: usize = MAX_WINDOW as usize - 1;
/// Tree nodes compared per position, searching or inserting.
fn depth() -> usize { std::env::var("ULTRA_DEPTH").ok().and_then(|v| v.parse().ok()).unwrap_or(128) }
const NONE: u32 = u32::MAX;
/// A match this long is taken whole: no other candidate at its position
/// is priced and no position inside it is searched.
const SUFFICIENT_LEN: usize = 128;
/// Price unit: 1/256 bit.
const BIT: u32 = 256;

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

/// Starting code counts, 4096 per table: the max level's parse over
/// Silesia, with weight moved onto the shortest matches (which that parse
/// never makes and this one does). They stay in every block's prices at
/// this weight, so a code the recent blocks did not use keeps a price the
/// parse can afford to try again; without that, a block parsed on prices
/// from a sparse block gets sparser (an unused code costs log2(total)
/// bits) and never recovers.
const PRIOR_LL: [u32; LL_SYMBOLS] = [2543, 744, 227, 147, 49, 77, 34, 44, 16, 29, 18, 30, 11, 20, 25, 23, 50, 6, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1];
const PRIOR_ML: [u32; ML_SYMBOLS] = [300, 600, 1035, 490, 298, 550, 268, 220, 157, 108, 72, 104, 105, 49, 36, 40, 340, 150, 23, 6, 5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1];
const PRIOR_OFF: [u32; OFF_SYMBOLS] = [209, 158, 59, 2, 3, 1, 12, 39, 65, 91, 103, 114, 129, 146, 186, 221, 269, 317, 362, 385, 388, 374, 288, 176];

impl Prices {
    /// Prices from the prior plus `s` (the recent blocks' counts), with
    /// `lit` standing in for the literal counts when `s` has none.
    fn of(s: &Stats, lit: &[u32; 256]) -> Prices {
        let mut ll = PRIOR_LL;
        let mut ml = PRIOR_ML;
        let mut off = PRIOR_OFF;
        for (a, b) in ll.iter_mut().zip(s.ll.iter()) {
            *a += b;
        }
        for (a, b) in ml.iter_mut().zip(s.ml.iter()) {
            *a += b;
        }
        for (a, b) in off.iter_mut().zip(s.off.iter()) {
            *a += b;
        }
        Prices { lit: costs(if s.lit.iter().any(|&c| c > 0) { &s.lit } else { lit }), ll: costs(&ll), ml: costs(&ml), off: costs(&off) }
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

/// -log2(p) of each symbol in 1/256 bit, add-one smoothed.
fn costs<const N: usize>(hist: &[u32; N]) -> [u32; N] {
    let total: f64 = hist.iter().map(|&c| c as f64 + 1.0).sum();
    let lt = total.log2();
    let mut out = [0u32; N];
    for (o, &c) in out.iter_mut().zip(hist.iter()) {
        *o = ((lt - (c as f64 + 1.0).log2()) * BIT as f64).round() as u32;
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
fn h4(p: *const u8) -> usize {
    let w = unsafe { std::ptr::read_unaligned(p as *const u32) };
    (w.wrapping_mul(0x9E37_79B1) >> (32 - HASH4_BITS)) as usize
}

#[inline(always)]
fn h3(p: *const u8) -> usize {
    let w = unsafe { std::ptr::read_unaligned(p as *const u32) } & 0x00FF_FFFF;
    (w.wrapping_mul(0x9E37_79B1) >> (32 - HASH3_BITS)) as usize
}

/// The finder's tables and the parse's work arrays: 28 MB, allocated
/// once per thread and cleared per call.
pub struct UltraState {
    hash4: Box<[u32]>,
    hash3: Box<[u32]>,
    /// Per window slot, the position's two children: the newest older
    /// position whose suffix sorts below it, and above it.
    tree: Box<[u32]>,
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
            hash4: vec![NONE; 1 << HASH4_BITS].into_boxed_slice(),
            hash3: vec![NONE; 1 << HASH3_BITS].into_boxed_slice(),
            tree: vec![NONE; 2 * MAX_WINDOW as usize].into_boxed_slice(),
            inserted: 0,
            opt: Vec::new(),
            cands: Vec::new(),
            stats: None,
            undo: Vec::new(),
        })
    }

    /// Forget everything: the output then depends only on the input.
    pub fn clear(&mut self) {
        self.hash4.fill(NONE);
        self.hash3.fill(NONE);
        self.tree.fill(NONE);
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
            self.walk(input, pos, pos, log, false);
            self.inserted = pos + 1;
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
            self.tree[2 * (pos & TREE_MASK)] = NONE;
            self.tree[2 * (pos & TREE_MASK) + 1] = NONE;
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
    fn walk(&mut self, input: &[u8], pos: usize, limit: usize, log: bool, collect: bool) -> usize {
        let src = input.as_ptr();
        let cur = unsafe { src.add(pos) };
        let full = input.len() - pos;
        let (i4, i3) = (h4(cur), h3(cur));
        if log {
            self.undo.push([i4 as u32, self.hash4[i4], i3 as u32, self.hash3[i3]]);
        }
        let mut m = self.hash4[i4];
        self.hash4[i4] = pos as u32;
        self.hash3[i3] = pos as u32;
        let low = pos.saturating_sub(MAX_WINDOW as usize - 1);
        let (mut sp, mut lp) = (2 * (pos & TREE_MASK), 2 * (pos & TREE_MASK) + 1);
        let (mut cls, mut cll) = (0usize, 0usize);
        let mut best = if collect { self.cands.last().map_or(MIN_MATCH as usize - 1, |c| c.0 as usize) } else { 0 };
        let mut n = depth();
        while n > 0 && m != NONE && (m as usize) < pos && (m as usize) >= low {
            n -= 1;
            let mi = m as usize;
            let node = 2 * (mi & TREE_MASK);
            let c = unsafe { src.add(mi) };
            let mut ml = cls.min(cll);
            ml += unsafe { ScalarMatch::prefix(cur.add(ml), c.add(ml), full - ml) };
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
                    return best;
                }
                sp = node + 1;
                m = self.tree[node + 1];
            } else {
                self.tree[lp] = m;
                cll = ml;
                if mi <= low {
                    self.tree[sp] = NONE;
                    return best;
                }
                lp = node;
                m = self.tree[node];
            }
        }
        // Whatever was below the last node visited is dropped from the
        // tree: both open slots are closed.
        self.tree[sp] = NONE;
        self.tree[lp] = NONE;
        best
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
            best = best.max(self.walk(input, pos, limit, log, true));
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
            if std::env::var("DUMP_PRICES").is_ok() {
                let lit_avg = block.iter().map(|&b| prices.lit[b as usize] as f64).sum::<f64>() / block.len() as f64 / 256.0;
                eprintln!("block at {block_start}: lit avg {lit_avg:.2} bits; ll(4) {:.2} ll(0) {:.2} ll(500) {:.2}; ml(4) {:.2} ml(5) {:.2} ml(8) {:.2}; off(1M) {:.2} off(1000) {:.2} off(rep0) {:.2}",
                    prices.ll(4) as f64 / 256.0, prices.ll(0) as f64 / 256.0, prices.ll(500) as f64 / 256.0,
                    prices.ml(4) as f64 / 256.0, prices.ml(5) as f64 / 256.0, prices.ml(8) as f64 / 256.0,
                    prices.off(1 << 20, &[1, 4, 8]) as f64 / 256.0, prices.off(1000, &[1, 4, 8]) as f64 / 256.0, prices.off(1, &[1, 4, 8]) as f64 / 256.0);
            }
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
