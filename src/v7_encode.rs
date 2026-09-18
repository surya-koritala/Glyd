//! Format v7 block encoder: a sequence list plus literals -> payload.
//!
//! Payload = SubHeader, then five sections. A coded section is:
//!   [table: 128 bytes of packed Huffman lengths, or 1 + 2 * n bytes of
//!    tANS counts (u8 count then u16 LE counts); omitted when reused]
//!   [8 x u32 LE sub-stream sizes]
//!   [8 sub-streams, each ending in bits::PAD zero bytes]
//! A raw section is the bytes themselves (literals, or one code byte per
//! sequence). The extra-bits section is always 8 raw padded sub-streams
//! behind a size table: sub-stream k holds, for sequences i == k (mod 8)
//! in order, the ll extra bits, then ml, then offset extra bits.
use crate::bits::{write_streams, PAD};
use crate::huff8;
use crate::tans;
use crate::v7_format::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Sequence {
    pub lit_len: u32,
    pub match_len: u32,
    pub offset: u32,
}

/// The previous block's entropy tables, carried forward so the next block
/// can reuse them instead of writing a fresh one.
pub struct Tables {
    pub lit_lengths: Option<[u8; 256]>,
    pub ll: Option<Vec<u16>>,
    pub ml: Option<Vec<u16>>,
    pub off: Option<Vec<u16>>,
}

impl Tables {
    pub fn none() -> Self {
        Tables { lit_lengths: None, ll: None, ml: None, off: None }
    }
}

pub struct Layout {
    pub sub: SubHeader,
    pub sections: [std::ops::Range<usize>; 5],
}

pub fn payload_layout(payload: &[u8]) -> Option<Layout> {
    let sub = SubHeader::parse(payload)?;
    let mut pos = SubHeader::BYTES;
    let mut sections: [std::ops::Range<usize>; 5] = std::array::from_fn(|_| 0..0);
    for s in 0..5 {
        let end = pos.checked_add(sub.sizes[s] as usize)?;
        if end > payload.len() {
            return None;
        }
        sections[s] = pos..end;
        pos = end;
    }
    if pos != payload.len() {
        return None;
    }
    Some(Layout { sub, sections })
}

/// Reuse when the two tables have identical support -- a symbol present
/// in one and absent (count 0) from the other never passes, regardless of
/// magnitude -- and every present count is within 1/8 of the previous
/// table's. Support must match exactly: reusing a table that assigns a
/// symbol zero probability while this block's data actually contains it
/// sends the tANS encoder's state machine out of bounds (a zero-count
/// symbol's `EncodeTable` entry is the placeholder `(0, 0)`, which is only
/// safe to hit for a symbol that truly never gets encoded). This is the
/// same support requirement the literal path already applies (see
/// `lit_reuse` below).
fn close(a: &[u16], b: &[u16]) -> bool {
    a.len() == b.len()
        && a.iter().zip(b).all(|(&x, &y)| {
            (x == 0) == (y == 0) && (x as i32 - y as i32).abs() <= (x as i32 / 8).max(2)
        })
}

/// Histogram over eight interleaved tables: consecutive equal symbols
/// (the common case in code streams, and runs in literals) otherwise
/// serialise on one counter's store-to-load forwarding.
fn hist8<const N: usize>(data: &[u8]) -> [u32; N] {
    let mut h = [[0u32; N]; 8];
    let mut it = data.chunks_exact(8);
    for c in &mut it {
        for k in 0..8 {
            h[k][c[k] as usize] += 1;
        }
    }
    for &b in it.remainder() {
        h[0][b as usize] += 1;
    }
    std::array::from_fn(|s| h.iter().map(|t| t[s]).sum())
}

/// Append the section for `codes`: tANS-coded against `counts` (either
/// freshly normalized for this block, or the previous block's when
/// `reusing`), or the codes raw if coding would not pay for itself.
/// Returns whether it was coded; a raw section never carries reuse.
fn encode_codes(codes: &[u8], hist: &[u32], counts: &[u16], reusing: bool, chunks: &mut Vec<u32>, out: &mut Vec<u8>) -> bool {
    let n_symbols = counts.len();
    let bits: f64 = (0..n_symbols)
        .map(|s| if hist[s] == 0 { 0.0 } else { hist[s] as f64 * -((counts[s] as f64) / tans::L as f64).log2() })
        .sum();
    let table_bytes = if reusing { 0 } else { 1 + 2 * n_symbols };
    let coded_estimate = (bits / 8.0) as usize + table_bytes + 8 * (4 + PAD);
    if coded_estimate + codes.len() / 50 >= codes.len() {
        out.extend_from_slice(codes);
        return false;
    }
    let et = tans::EncodeTable::build(counts).expect("normalized counts sum to L");
    if !reusing {
        out.push(n_symbols as u8);
        for &c in counts {
            out.extend_from_slice(&c.to_le_bytes());
        }
    }
    tans::encode8_into(codes, &et, chunks, out);
    true
}

/// Per-block scratch, owned by the caller and reused across blocks.
pub struct EncScratch {
    ll: Vec<u8>,
    ml: Vec<u8>,
    off: Vec<u8>,
    /// Per sequence: the ll, ml and offset extra bits concatenated in
    /// stream order (at most 18 + 18 + 20 = `EXTRA_BITS` for lengths
    /// within a block and offsets within the window), their count above.
    extra: Vec<u64>,
    chunks: Vec<u32>,
}

impl EncScratch {
    pub fn new() -> Self {
        EncScratch { ll: Vec::new(), ml: Vec::new(), off: Vec::new(), extra: Vec::new(), chunks: Vec::new() }
    }
}

/// Bits of `extra` per sequence, a `MAX_PUT` put.
const EXTRA_BITS: u32 = 56;
const _: () = assert!(EXTRA_BITS <= crate::bits::MAX_PUT);

/// Lengths must be at most `MAX_BLOCK_SIZE` (2^18, as the decoder
/// enforces per block) and offsets below `MAX_WINDOW`.
pub fn encode_block(seqs: &[Sequence], literals: &[u8], dict_id: u32, prev: &mut Tables, out: &mut Vec<u8>) {
    encode_block_with(seqs, literals, dict_id, prev, &mut EncScratch::new(), out)
}

/// `encode_block` with caller-owned scratch (`compress_into_max` keeps
/// one across a stream's blocks).
pub fn encode_block_with(seqs: &[Sequence], literals: &[u8], dict_id: u32, prev: &mut Tables, s: &mut EncScratch, out: &mut Vec<u8>) {
    let n = seqs.len();
    let base = out.len();
    out.resize(base + SubHeader::BYTES, 0);

    // Codes and extra bits, in sequence order (the rep state).
    s.ll.clear();
    s.ll.resize(n, 0);
    s.ml.clear();
    s.ml.resize(n, 0);
    s.off.clear();
    s.off.resize(n, 0);
    s.extra.clear();
    s.extra.resize(n, 0);
    let mut reps = Reps::new();
    let mut max_bits = 0u32;
    let codes = s.ll.iter_mut().zip(s.ml.iter_mut()).zip(s.off.iter_mut()).zip(s.extra.iter_mut());
    for (i, (q, (((ll, ml), off), extra))) in seqs.iter().zip(codes).enumerate() {
        let (llc, nb, e) = ll_code(q.lit_len);
        let mut v = e as u64;
        let mut bits = nb as u32;
        // The literal-only last sequence codes as a match of MIN_MATCH:
        // ml code 0 with no extra bits, which is what the decoder expects.
        let (mlc, nb, e) = ml_code(q.match_len.max(MIN_MATCH));
        v |= (e as u64) << bits;
        bits += nb as u32;
        let (offc, nb, e) = if q.match_len != 0 {
            reps.code_for(q.offset)
        } else {
            debug_assert_eq!(i, n - 1, "literal-only sequence must be last");
            (0, 0, 0)
        };
        v |= (e as u64) << bits;
        bits += nb as u32;
        max_bits = max_bits.max(bits);
        *ll = llc;
        *ml = mlc;
        *off = offc;
        *extra = v | (bits as u64) << EXTRA_BITS;
    }
    assert!(max_bits <= EXTRA_BITS, "sequence length beyond the block bound");

    // Literals.
    let hist: [u64; 256] = hist8::<256>(literals).map(|c| c as u64);
    let lengths = huff8::lengths_for(&hist);
    let lit_reuse = prev.lit_lengths.map_or(false, |p| {
        let est_prev: u64 = (0..256).map(|s| hist[s] * p[s] as u64).sum();
        let est_new: u64 = (0..256).map(|s| hist[s] * lengths[s] as u64).sum();
        p.iter().zip(lengths.iter()).all(|(&a, &b)| (a > 0) == (b > 0)) && est_prev <= est_new + (huff8::TABLE_BYTES as u64) * 8
    });
    let lit_lengths = if lit_reuse { prev.lit_lengths.unwrap() } else { lengths };
    let lit_coded_size = huff8::coded_size(&hist, &lit_lengths) - if lit_reuse { huff8::TABLE_BYTES } else { 0 } + 8 * (4 + PAD);
    let lit_coded = literals.len() >= 64 && lit_coded_size + literals.len() / 50 < literals.len();
    let lit_start = out.len();
    if lit_coded {
        if !lit_reuse {
            crate::huffman::pack_lengths(&lit_lengths, out);
        }
        huff8::encode_into(literals, &lit_lengths, out);
    } else {
        out.extend_from_slice(literals);
    }
    let lit_size = out.len() - lit_start;

    // Sequence code streams: one up-front decision (not a per-stream, then
    // shadowed re-encode) -- reuse the previous block's three tables only
    // when all three are present and each is close to this block's fresh
    // counts; otherwise recompute and write fresh tables for all three.
    let ll_hist = hist8::<{ tans::MAX_SYMBOLS }>(&s.ll);
    let ml_hist = hist8::<{ tans::MAX_SYMBOLS }>(&s.ml);
    let off_hist = hist8::<{ tans::MAX_SYMBOLS }>(&s.off);
    let ll_fresh = tans::normalize(&ll_hist, LL_SYMBOLS);
    let ml_fresh = tans::normalize(&ml_hist, ML_SYMBOLS);
    let off_fresh = tans::normalize(&off_hist, OFF_SYMBOLS);
    let mut seq_reuse = matches!(
        (&prev.ll, &prev.ml, &prev.off),
        (Some(a), Some(b), Some(c)) if close(a, &ll_fresh) && close(b, &ml_fresh) && close(c, &off_fresh)
    );
    let mut ll_counts = if seq_reuse { prev.ll.clone().unwrap() } else { ll_fresh.clone() };
    let mut ml_counts = if seq_reuse { prev.ml.clone().unwrap() } else { ml_fresh.clone() };
    let mut off_counts = if seq_reuse { prev.off.clone().unwrap() } else { off_fresh.clone() };
    let seq_start = out.len();
    let mut sizes = [0usize; 3];
    let mut coded = [false; 3];
    for _ in 0..2 {
        for (i, (codes, hist, counts)) in [(&s.ll, &ll_hist, &ll_counts), (&s.ml, &ml_hist, &ml_counts), (&s.off, &off_hist, &off_counts)].into_iter().enumerate() {
            let start = out.len();
            coded[i] = encode_codes(codes, hist, counts, seq_reuse, &mut s.chunks, out);
            sizes[i] = out.len() - start;
        }
        // The up-front reuse decision is optimistic (each `encode_codes`
        // call still independently applies the raw-beats-coding rule).
        // If reuse was assumed but one stream falls back to raw anyway,
        // the other two cannot be left with their table omitted under a
        // header that now has to report "not reused" -- that combination
        // is undecodable (a coded, non-reused section must carry its own
        // table). Redo all three fresh, no reuse, and let each decide
        // coded/raw again on its own.
        if !seq_reuse || coded.iter().all(|&c| c) {
            break;
        }
        seq_reuse = false;
        ll_counts = ll_fresh.clone();
        ml_counts = ml_fresh.clone();
        off_counts = off_fresh.clone();
        out.truncate(seq_start);
    }
    let seq_coded = coded.iter().all(|&c| c);

    // Extra bits: sub-stream k holds sequences k, k + 8, ... in order.
    let extra_start = out.len();
    write_streams(out, n.div_ceil(8) * EXTRA_BITS as usize, |k, w| {
        let e = &s.extra[k.min(n)..];
        let mut i = 0;
        // SAFETY: at most ceil(n / 8) sequences of at most EXTRA_BITS
        // bits each go into this stream, the `max_bits` reserved.
        unsafe {
            while i + 8 < e.len() {
                // Two sequences per put when they fit MAX_PUT together
                // (nearly always: a sequence averages ~15 extra bits).
                let (x, y) = (e[i], e[i + 8]);
                let (nx, ny) = ((x >> EXTRA_BITS) as u32, (y >> EXTRA_BITS) as u32);
                let (vx, vy) = (x & ((1u64 << EXTRA_BITS) - 1), y & ((1u64 << EXTRA_BITS) - 1));
                if nx + ny <= crate::bits::MAX_PUT {
                    w.put(vx | vy << nx, nx + ny);
                } else {
                    w.put(vx, nx);
                    w.put(vy, ny);
                }
                i += 16;
            }
            if i < e.len() {
                let x = e[i];
                w.put(x & ((1u64 << EXTRA_BITS) - 1), (x >> EXTRA_BITS) as u32);
            }
        }
    });
    let extra_size = out.len() - extra_start;

    let sub = SubHeader {
        coded: (lit_coded as u8) << S_LIT | (coded[0] as u8) << S_LL | (coded[1] as u8) << S_ML | (coded[2] as u8) << S_OFF,
        reuse: (lit_coded && lit_reuse) as u8 | (((seq_reuse && seq_coded) as u8) << 1),
        dict_id,
        sizes: [lit_size as u32, sizes[0] as u32, sizes[1] as u32, sizes[2] as u32, extra_size as u32],
    };
    let mut hdr = Vec::with_capacity(SubHeader::BYTES);
    sub.write(&mut hdr);
    out[base..base + SubHeader::BYTES].copy_from_slice(&hdr);

    if lit_coded {
        prev.lit_lengths = Some(lit_lengths);
    }
    if seq_coded {
        prev.ll = Some(ll_counts);
        prev.ml = Some(ml_counts);
        prev.off = Some(off_counts);
    } else {
        prev.ll = None;
        prev.ml = None;
        prev.off = None;
    }
}

/// v6 streams (tokens, offsets, extras) -> sequences. Milestone 2 bridge:
/// lets the v7 container and decoder be measured on the proven parse.
/// The streams must be well formed (they come straight from the finder).
pub fn sequences_from_streams(tokens: &[u8], offsets: &[u8], extras: &[u8], min_match: usize, seqs: &mut Vec<Sequence>) {
    use crate::format::{Token, ESCAPE_BASE_LIT, ESCAPE_CONT, LIT_CODE_ESCAPE, MATCH_CODE_ESCAPE};
    fn read_escape(extras: &[u8], e: &mut usize, base: usize) -> u32 {
        let v = extras[*e] as usize;
        *e += 1;
        if v != ESCAPE_CONT as usize {
            return (base + v) as u32;
        }
        let w = u16::from_le_bytes([extras[*e], extras[*e + 1]]) as usize;
        *e += 2;
        (base + v + w) as u32
    }
    let bias = min_match - 1;
    let mut e = 0usize;
    let mut o = 0usize;
    seqs.clear();
    seqs.reserve(tokens.len() + 1);
    // The format wants exactly one literal-only sequence, last. Merge any
    // interior literal-only tokens (the v6 MAX_LIT_LEN split) into the next.
    let mut carry = 0u32;
    for (i, &t) in tokens.iter().enumerate() {
        let tok = Token(t);
        let lc = tok.lit_code();
        let mc = tok.match_code();
        let lit_len = if lc == LIT_CODE_ESCAPE { read_escape(extras, &mut e, ESCAPE_BASE_LIT) } else { lc as u32 };
        if mc == 0 {
            if i + 1 < tokens.len() {
                carry += lit_len;
                continue;
            }
            seqs.push(Sequence { lit_len: lit_len + carry, match_len: 0, offset: 0 });
            return;
        }
        let match_len = if mc == MATCH_CODE_ESCAPE { read_escape(extras, &mut e, bias + 15) } else { (mc + bias) as u32 };
        let lo = u16::from_le_bytes([offsets[o], offsets[o + 1]]) as u32;
        o += 2;
        seqs.push(Sequence { lit_len: lit_len + carry, match_len, offset: lo | ((tok.off_hi() as u32) << 16) });
        carry = 0;
    }
    seqs.push(Sequence { lit_len: 0, match_len: 0, offset: 0 });
}

// ---------------------------------------------------------------------------
// Double-fast parse (zstd -3's shape): a long table keyed by an 8-byte hash
// and a short table keyed by a 5-byte hash, one candidate each; at every
// position the three repeat offsets are tried first, then the long
// candidate, then the short one. Lazy by one position in the long table
// only: a long candidate at pos + 1 that is 4+ bytes longer wins (the
// full re-probe -- reps and the short table too -- was worth 0.2% more
// ratio at three times the lazy step's cost). Window 2 MB.
//
// Table sizes are the ratio dial (Silesia, this coder): 17/16 bits (zstd
// -3's, 768 KB) 3.162; 18/17 (1.5 MB) 3.204; 18/18 (2 MB) 3.222, the
// parse ~7% slower than 17/16, the extra on the binaries (x-ray +30%).

pub const DFAST_LONG_BITS: u32 = 18;
pub const DFAST_SHORT_BITS: u32 = 18;
/// Misses before the probe step grows by one (as the fast finder).
const DFAST_SKIP_STRENGTH: u32 = 6;

pub struct DfastTables {
    long: Box<[u32; 1 << DFAST_LONG_BITS]>,
    short: Box<[u32; 1 << DFAST_SHORT_BITS]>,
}

impl DfastTables {
    /// Both tables empty. An entry is a tagged position (see `POS_BITS`);
    /// 0 doubles as empty, which is harmless: position 0 is verified by
    /// content like any other candidate. So is an entry left over from a
    /// previous input: a candidate at or past the current position is
    /// rejected and any other one is compared byte for byte, so the
    /// tables can be reused across inputs without clearing.
    pub fn new() -> Box<Self> {
        Box::new(DfastTables {
            long: vec![0; 1 << DFAST_LONG_BITS].into_boxed_slice().try_into().unwrap(),
            short: vec![0; 1 << DFAST_SHORT_BITS].into_boxed_slice().try_into().unwrap(),
        })
    }
}

/// Table entries: the position's low 24 bits under an 8-bit hash tag, so
/// a candidate whose tag differs is rejected from the entry alone, before
/// its bytes are loaded (the load chain hash -> entry -> bytes -> compare
/// is the probe's critical path). Distances are taken modulo 2^24: the
/// window (2^21) is smaller, and any candidate the check lets through is
/// compared byte for byte, so an entry from another 16 MB epoch or from a
/// previous input is only ever a wrong guess, never an unsafe one.
const POS_BITS: u32 = 24;
const POS_MASK: usize = (1 << POS_BITS) - 1;

#[inline(always)]
fn h8(w: u64) -> u64 {
    w.wrapping_mul(0x9E37_79B9_7F4A_7C15)
}
#[inline(always)]
fn h5(w: u64) -> u64 {
    (w << 24).wrapping_mul(889_523_592_379)
}
/// (index, tagged position) for the long table.
#[inline(always)]
fn long_slot(w: u64, pos: usize) -> (usize, u32) {
    let h = h8(w);
    ((h >> (64 - DFAST_LONG_BITS)) as usize, (pos & POS_MASK) as u32 | ((h >> (64 - DFAST_LONG_BITS - 8)) as u32) << POS_BITS)
}
#[inline(always)]
fn short_slot(w: u64, pos: usize) -> (usize, u32) {
    let h = h5(w);
    ((h >> (64 - DFAST_SHORT_BITS)) as usize, (pos & POS_MASK) as u32 | ((h >> (64 - DFAST_SHORT_BITS - 8)) as u32) << POS_BITS)
}
#[inline(always)]
unsafe fn eq4(a: *const u8, b: *const u8) -> bool {
    std::ptr::read_unaligned(a as *const u32) == std::ptr::read_unaligned(b as *const u32)
}
/// The candidate an entry `e` names for a probe at `pos` whose own
/// tagged entry is `mine`: its distance if the tags agree and it is at
/// least 1 and within the window (and not before the input), else None.
#[inline(always)]
fn candidate(e: u32, mine: u32, pos: usize) -> Option<usize> {
    let d = (pos.wrapping_sub(e as usize)) & POS_MASK;
    let lim = pos.min(MAX_WINDOW as usize - 1);
    if d.wrapping_sub(1) < lim && (e ^ mine) >> POS_BITS == 0 {
        Some(pos - d)
    } else {
        None
    }
}

/// The match at `pos`: the first of the repeat offsets (4-byte compare
/// each), the long candidate and the short one that verifies (4+ bytes;
/// a long candidate whose tag agrees shares 8 in practice), extended to
/// at most `block_end`; `(usize::MAX, 0)` if none. Records `pos` in both
/// tables. Reads 8 bytes at `pos`: `pos + 8 <= block_end` is the caller's
/// guarantee.
#[inline(always)]
unsafe fn probe(src: *const u8, pos: usize, block_end: usize, t: &mut DfastTables, r: &[u32; 3]) -> (usize, usize) {
    use crate::finder::{MatchLen, ScalarMatch};
    let p = src.add(pos);
    let w = std::ptr::read_unaligned(p as *const u64);
    let (il, ml) = long_slot(w, pos);
    let (is, ms) = short_slot(w, pos);
    let el = t.long[il];
    let es = t.short[is];
    t.long[il] = ml;
    t.short[is] = ms;
    for &o in r {
        let o = o as usize;
        if o <= pos && eq4(p, p.sub(o)) {
            return (pos - o, 4 + ScalarMatch::prefix(p.add(4), p.sub(o).add(4), block_end - pos - 4));
        }
    }
    for (e, mine) in [(el, ml), (es, ms)] {
        if let Some(c) = candidate(e, mine, pos) {
            let len = ScalarMatch::prefix(p, src.add(c), block_end - pos);
            if len >= 4 {
                return (c, len);
            }
        }
    }
    (usize::MAX, 0)
}

/// Parse `input[block_start..block_start + block_len]` into `seqs` and
/// `literals` (appended). Offsets are absolute distances, at least 1 and
/// under MAX_WINDOW; the last sequence is literal-only. `t` carries the
/// window across the blocks of one input. `reps` is only used to *find*
/// matches (the codes are assigned by `encode_block`, whose `Reps` starts
/// fresh per block, so the caller passes `[1, 4, 8]` at every block).
pub fn find_sequences_dfast(input: &[u8], block_start: usize, block_len: usize, t: &mut DfastTables, reps: &mut [u32; 3], seqs: &mut Vec<Sequence>, literals: &mut Vec<u8>) {
    use crate::finder::MatchLen;
    let src = input.as_ptr();
    let block_end = block_start + block_len;
    assert!(block_end <= input.len(), "block past the input");
    debug_assert!(reps.iter().all(|&o| o >= 1), "a zero repeat offset would verify against itself");
    // Every probe reads 8 bytes at `pos`; extensions stop at block_end.
    let limit = block_end.saturating_sub(8).max(block_start);
    let mut anchor = block_start;
    let mut pos = block_start;
    let mut step_nb: u32 = 1 << DFAST_SKIP_STRENGTH;
    let mut r = *reps;
    // The literal copies below run 16 bytes wild past their length.
    literals.reserve(block_len + 16);

    unsafe {
        while pos < limit {
            let (mut cand, mut rc) = probe(src, pos, block_end, t, &r);
            if cand == usize::MAX {
                pos += (step_nb >> DFAST_SKIP_STRENGTH) as usize;
                step_nb += 1;
                continue;
            }
            step_nb = 1 << DFAST_SKIP_STRENGTH;
            // Lazy: a long candidate one byte later that is 4+ bytes
            // longer wins.
            if pos + 1 < limit {
                let p1 = src.add(pos + 1);
                let w = std::ptr::read_unaligned(p1 as *const u64);
                let (il, ml) = long_slot(w, pos + 1);
                let (is, ms) = short_slot(w, pos + 1);
                let el = t.long[il];
                t.long[il] = ml;
                t.short[is] = ms;
                if let Some(c) = candidate(el, ml, pos + 1) {
                    let rc1 = crate::finder::ScalarMatch::prefix(p1, src.add(c), block_end - pos - 1);
                    if rc1 >= rc + 4 {
                        pos += 1;
                        cand = c;
                        rc = rc1;
                    }
                }
            }
            // Back-match into the pending literals.
            let mut mpos = pos;
            let mut c = cand;
            while mpos > anchor && c > 0 && *src.add(mpos - 1) == *src.add(c - 1) {
                mpos -= 1;
                c -= 1;
                rc += 1;
            }
            let offset = (mpos - c) as u32;
            let ll = mpos - anchor;
            let dst = literals.as_mut_ptr().add(literals.len());
            if ll <= 16 {
                // Two 8-byte copies, the second at the tail (they overlap
                // or coincide): reads stay under `mpos + 8 <= block_end`
                // and the reserve above covers the writes.
                let k = ll.saturating_sub(8);
                std::ptr::copy_nonoverlapping(src.add(anchor), dst, 8);
                std::ptr::copy_nonoverlapping(src.add(anchor + k), dst.add(k), 8);
            } else {
                std::ptr::copy_nonoverlapping(src.add(anchor), dst, ll);
            }
            literals.set_len(literals.len() + ll);
            seqs.push(Sequence { lit_len: ll as u32, match_len: rc as u32, offset });
            // The same update as Reps::code_for.
            if offset == r[1] {
                r.swap(0, 1);
            } else if offset == r[2] {
                r = [r[2], r[0], r[1]];
            } else if offset != r[0] {
                r = [offset, r[0], r[1]];
            }
            pos = mpos + rc;
            anchor = pos;
            // Index the match's second position and its tail (zstd's
            // insertions) so runs keep hashing.
            if pos < limit {
                let w = std::ptr::read_unaligned(src.add(mpos + 2) as *const u64);
                let (i, m) = long_slot(w, mpos + 2);
                t.long[i] = m;
                let (i, m) = short_slot(w, mpos + 2);
                t.short[i] = m;
                let w = std::ptr::read_unaligned(src.add(pos - 2) as *const u64);
                let (i, m) = long_slot(w, pos - 2);
                t.long[i] = m;
                let w = std::ptr::read_unaligned(src.add(pos - 1) as *const u64);
                let (i, m) = short_slot(w, pos - 1);
                t.short[i] = m;
            }
        }
    }
    literals.extend_from_slice(&input[anchor..block_end]);
    seqs.push(Sequence { lit_len: (block_end - anchor) as u32, match_len: 0, offset: 0 });
    *reps = r;
}
