//! Format v7 block encoder: a sequence list plus literals -> payload.
//!
//! Payload = SubHeader, then five sections. A coded section is:
//!   [table: 128 bytes of packed Huffman lengths, or 1 + 2 * n bytes of
//!    tANS counts (u8 count then u16 LE counts); omitted when reused]
//!   [8 x u32 LE sub-stream sizes]
//!   [8 sub-streams behind seven 24-bit sizes, then bits::PAD zero bytes]
//! A raw section is the bytes themselves (literals, or one code byte per
//! sequence). The extra-bits section is always 8 raw padded sub-streams
//! behind a size table: sub-stream k holds, for sequences i == k (mod 8)
//! in order, the ll extra bits, then ml, then offset extra bits.
use std::sync::Arc;
use crate::bits::{section_frame_bytes, write_section, Framing, PAD};
use crate::format::MAX_BLOCK_SIZE;
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
/// The counts and built tables are shared, not copied: every object
/// compressed with a dictionary starts from a clone of the dictionary's.
#[derive(Clone)]
pub struct Tables {
    pub lit_lengths: Option<[u8; 256]>,
    /// The codes of `lit_lengths`, built on first use.
    pub lit_codes: Option<Arc<huff8::Codes>>,
    pub ll: Option<Arc<[u16]>>,
    pub ml: Option<Arc<[u16]>>,
    pub off: Option<Arc<[u16]>>,
    /// The encoder's tables for `ll`, `ml`, `off`, built on first reuse.
    pub seq_enc: Option<Arc<[tans::EncodeTable; 3]>>,
}

impl Tables {
    pub fn none() -> Self {
        Tables { lit_lengths: None, lit_codes: None, ll: None, ml: None, off: None, seq_enc: None }
    }

    /// The encoder's tables of `ll`/`ml`/`off`, built if not yet.
    pub(crate) fn seq_enc(&mut self) -> &Arc<[tans::EncodeTable; 3]> {
        let (ll, ml, off) = (self.ll.as_ref().unwrap(), self.ml.as_ref().unwrap(), self.off.as_ref().unwrap());
        self.seq_enc.get_or_insert_with(|| {
            let build = |c: &[u16]| tans::EncodeTable::build(c).expect("normalized counts sum to L");
            Arc::new([build(ll), build(ml), build(off)])
        })
    }
}

pub struct Layout {
    pub sub: SubHeader,
    pub sections: [std::ops::Range<usize>; 5],
    /// Set by the decoder from the block header: the v8 section layout
    /// (`bits::Stream::split`, packed tANS counts, 26 offset codes)
    /// rather than v7's.
    pub v8: bool,
    /// The v9 framing: compact sub-header, `bits::Stream::split_compact`
    /// sections, one padding at the payload's end (the decoder's copy).
    pub compact: bool,
}

/// The layout of a v7/v8 payload.
pub fn payload_layout(payload: &[u8]) -> Option<Layout> {
    let sub = SubHeader::parse(payload)?;
    layout_from(payload, sub, SubHeader::BYTES, 0, false)
}

/// The layout of a block's payload as written (no padding), by its
/// header's version.
pub fn payload_layout_of(header: &crate::format::BlockHeader, payload: &[u8]) -> Option<Layout> {
    if header.version == crate::format::VERSION_V9 {
        payload_layout_compact(payload, false)
    } else {
        payload_layout(payload)
    }
}

/// The layout of a v9 payload: the sections, then, when `padded` (the
/// decoder's copy), `PAD` zero bytes.
pub fn payload_layout_compact(payload: &[u8], padded: bool) -> Option<Layout> {
    let tail = if padded { PAD } else { 0 };
    let (sub, used) = SubHeader::parse_compact(payload, tail)?;
    layout_from(payload, sub, used, tail, true)
}

fn layout_from(payload: &[u8], sub: SubHeader, start: usize, tail: usize, compact: bool) -> Option<Layout> {
    let mut pos = start;
    let mut sections: [std::ops::Range<usize>; 5] = std::array::from_fn(|_| 0..0);
    for s in 0..5 {
        let end = pos.checked_add(sub.sizes[s] as usize)?;
        if end > payload.len() {
            return None;
        }
        sections[s] = pos..end;
        pos = end;
    }
    if pos + tail != payload.len() {
        return None;
    }
    Some(Layout { sub, sections, v8: true, compact })
}

/// Bits `hist` costs under a tANS table of `counts`, or None when a symbol
/// present in the data has count 0 there: such a table must never be used
/// for this data (a zero-count symbol's `EncodeTable` entry is the
/// placeholder `(0, 0)`, which sends the encoder's state machine out of
/// bounds; the literal path applies the same rule, see `lit_reuse`). A
/// table can be reused when every present symbol has a count, whatever
/// the two tables' supports are otherwise: the decision is by cost.
fn table_cost(hist: &[u32], counts: &[u16]) -> Option<u64> {
    debug_assert_eq!(hist.len(), counts.len());
    // In 1/65536 bit, by integer arithmetic: the same decision everywhere.
    let mut bits = 0u64;
    for (&h, &c) in hist.iter().zip(counts) {
        if h != 0 {
            if c == 0 {
                return None;
            }
            bits += h as u64 * crate::fixlog::tans_cost_q16(c);
        }
    }
    Some(bits)
}

/// Literal sections below this many bytes, and sequence sections below
/// this many sequences, reuse covering previous tables without pricing
/// fresh ones (see `encode_block_coded`).
const SMALL_LITERALS: usize = 4096;
const SMALL_SEQUENCES: usize = 512;

/// A plain histogram, for short inputs (eight tables cost more to zero
/// than they save).
fn hist1(data: &[u8]) -> [u64; 256] {
    let mut h = [0u64; 256];
    for &b in data {
        h[b as usize] += 1;
    }
    h
}

/// Histogram over eight interleaved tables: consecutive equal symbols
/// (the common case in code streams, and runs in literals) otherwise
/// serialise on one counter's store-to-load forwarding.
fn hist8<const N: usize>(data: &[u8]) -> [u32; N] {
    if data.len() < 32 * N {
        // Eight tables cost more to zero and sum than they save.
        let mut h = [0u32; N];
        for &b in data {
            h[b as usize] += 1;
        }
        return h;
    }
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
fn encode_codes(codes: &[u8], hist: &[u32], counts: &[u16], reused: Option<&tans::EncodeTable>, framing: Framing, chunks: &mut Vec<u32>, out: &mut Vec<u8>) -> bool {
    let n_symbols = counts.len();
    let bits = table_cost(&hist[..n_symbols], counts).expect("table covers the data");
    let table_bytes = if reused.is_some() { 0 } else { tans_table_bytes(counts) };
    let coded_estimate = (bits >> 19) as usize + table_bytes + section_frame_bytes(framing);
    if coded_estimate + codes.len() / 50 >= codes.len() {
        out.extend_from_slice(codes);
        return false;
    }
    let own;
    let et = match reused {
        Some(et) => et,
        None => {
            write_tans_table(counts, out);
            own = tans::EncodeTable::build(counts).expect("normalized counts sum to L");
            &own
        }
    };
    tans::encode8_into(codes, et, chunks, framing, out);
    true
}

/// A tANS table as a v8 block carries it: the symbol count, then each
/// count as its bit width in a nibble followed by the bits below its top
/// one (a count of 0 or 1 is the nibble alone; 1024 is width 11), packed
/// little-endian. Small counts, the common case, take 4-7 bits.
fn tans_table_bits(counts: &[u16]) -> usize {
    counts.iter().map(|&c| 4 + (16 - c.leading_zeros() as usize).saturating_sub(1)).sum()
}

fn tans_table_bytes(counts: &[u16]) -> usize {
    1 + tans_table_bits(counts).div_ceil(8)
}

pub(crate) fn write_tans_table(counts: &[u16], out: &mut Vec<u8>) {
    out.push(counts.len() as u8);
    let start = out.len();
    out.resize(start + tans_table_bits(counts).div_ceil(8), 0);
    let mut bit = 0usize;
    let mut put = |v: u32, n: usize, bit: &mut usize| {
        // At most 15 bits at a time: three bytes cover any position.
        let byte = start + (*bit >> 3);
        let v = (v as u64) << (*bit & 7);
        for k in 0..3 {
            if byte + k < out.len() {
                out[byte + k] |= (v >> (8 * k)) as u8;
            }
        }
        *bit += n;
    };
    for &c in counts {
        debug_assert!(c as usize <= tans::L);
        let w = 16 - c.leading_zeros() as usize;
        put(w as u32, 4, &mut bit);
        if w >= 2 {
            put(c as u32 - (1 << (w - 1)), w - 1, &mut bit);
        }
    }
}

/// Per-block scratch, owned by the caller and reused across blocks.
#[derive(Default)]
pub struct EncScratch {
    ll: Vec<u8>,
    ml: Vec<u8>,
    off: Vec<u8>,
    /// Per sequence: the ll, ml and offset extra bits concatenated in
    /// stream order (at most 18 + 17 + 22 = `EXTRA_BITS` for lengths
    /// within a block and offsets within the window), their count above.
    extra: Vec<u64>,
    chunks: Vec<u32>,
}

impl EncScratch {
    pub fn new() -> Self {
        EncScratch { ll: Vec::new(), ml: Vec::new(), off: Vec::new(), extra: Vec::new(), chunks: Vec::new() }
    }

    /// Empty the per-sequence arrays for a new block.
    pub fn clear(&mut self) {
        self.ll.clear();
        self.ml.clear();
        self.off.clear();
        self.extra.clear();
    }

    /// Append one sequence's codes: lengths as `len_code` and the offset
    /// as `Reps::code_for` would code them, `rep` naming the rep slot the
    /// offset sits in (0..=2) or 3 for none. Branch-free: a rep hit is a
    /// coin toss on binaries. SAFETY: the caller has reserved room for
    /// this entry in all four arrays (`reserve_for`).
    #[inline(always)]
    unsafe fn push_codes(&mut self, lit_len: u32, match_len: u32, offset: u32, rep: u32) {
        use std::hint::select_unpredictable as sel;
        // The length codes' branches are on values below 64 and 131,
        // which almost every run and match is: well predicted.
        let (llc, lnb, le) = ll_code(lit_len);
        let (mlc, mnb, me) = ml_code(match_len);
        let (lnb, mnb) = (lnb as u32, mnb as u32);
        debug_assert!(offset >= 1);
        let k = 31 - offset.leading_zeros();
        let is_rep = rep < 3;
        let offc = sel(is_rep, rep, 3 + k) as u8;
        let onb = sel(is_rep, 0, k);
        let oe = sel(is_rep, 0, offset - (1 << k));
        let v = le as u64 | (me as u64) << lnb | (oe as u64) << (lnb + mnb);
        let bits = lnb + mnb + onb;
        let i = self.ll.len();
        debug_assert!(i < self.ll.capacity() && i < self.ml.capacity() && i < self.off.capacity() && i < self.extra.capacity());
        *self.ll.as_mut_ptr().add(i) = llc;
        *self.ml.as_mut_ptr().add(i) = mlc;
        *self.off.as_mut_ptr().add(i) = offc;
        *self.extra.as_mut_ptr().add(i) = v | (bits as u64) << EXTRA_BITS;
        self.ll.set_len(i + 1);
        self.ml.set_len(i + 1);
        self.off.set_len(i + 1);
        self.extra.set_len(i + 1);
    }

    /// Room for every sequence a block of `block_len` bytes can hold (a
    /// match is at least 4 bytes, plus the literal-only last one).
    fn reserve_for(&mut self, block_len: usize) {
        let n = block_len / 4 + 2;
        self.ll.reserve(n);
        self.ml.reserve(n);
        self.off.reserve(n);
        self.extra.reserve(n);
    }
}

/// Bits of `extra` per sequence, a `MAX_PUT` put.
/// Bits one sequence's three fields can hold: a literal run of 2^18 (18),
/// a match of up to 2^18 (17), an offset below 2^23 (22). One more than a
/// `put` carries, so `put_wide` splits such a sequence in two.
const EXTRA_BITS: u32 = 57;
const _: () = assert!(EXTRA_BITS <= crate::bits::MAX_PUT + 1);
const _: () = assert!(
    EXTRA_BITS as usize
        == extra_bits_of_code(Kind::Ll, ll_code(crate::format::MAX_BLOCK_SIZE as u32).0) as usize
            + extra_bits_of_code(Kind::Ml, ml_code(crate::format::MAX_BLOCK_SIZE as u32).0) as usize
            + (MAX_OFFSET_BITS - 1) as usize
);

/// One sequence's extra bits, in two puts when they exceed one.
#[inline(always)]
unsafe fn put_wide(w: &mut crate::bits::BitCursor, v: u64, n: u32) {
    if n <= crate::bits::MAX_PUT {
        w.put(v, n);
    } else {
        w.put(v & 0xFFFF_FFFF, 32);
        w.put(v >> 32, n - 32);
    }
}

/// Lengths must be at most `MAX_BLOCK_SIZE` (2^18, as the decoder
/// enforces per block), offsets at least 1 and below `MAX_WINDOW`, and
/// the last sequence literal-only (`match_len == 0`; it is coded as ml
/// code 0, which the decoder reads as "no match" only in last position),
/// with no other literal-only sequence.
pub fn encode_block(seqs: &[Sequence], literals: &[u8], dict_id: u32, prev: &mut Tables, out: &mut Vec<u8>) {
    encode_block_with(seqs, literals, dict_id, prev, &mut EncScratch::new(), false, out)
}

/// `encode_block` with caller-owned scratch (`compress_into_max` keeps
/// one across a stream's blocks).
pub fn encode_block_with(seqs: &[Sequence], literals: &[u8], dict_id: u32, prev: &mut Tables, s: &mut EncScratch, compact: bool, out: &mut Vec<u8>) {
    let n = seqs.len();
    debug_assert!(seqs.last().map_or(true, |q| q.match_len == 0), "the last sequence must be literal-only");
    // Codes and extra bits, in sequence order (the rep state).
    s.clear();
    let mut reps = Reps::new();
    let mut max_bits = 0u32;
    for (i, q) in seqs.iter().enumerate() {
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
        s.ll.push(llc);
        s.ml.push(mlc);
        s.off.push(offc);
        s.extra.push(v | (bits as u64) << EXTRA_BITS);
    }
    assert!(max_bits <= EXTRA_BITS, "sequence length beyond the block bound");
    encode_block_coded(literals, dict_id, prev, s, compact, out)
}

/// The block from codes already in the scratch (`find_sequences_dfast`
/// writes them as it parses; `encode_block_with` from a sequence list).
/// Every `extra` entry's count (its top byte) must be at most
/// `EXTRA_BITS`, which those two producers guarantee: the extras section
/// is written without further checks against that bound.
pub(crate) fn encode_block_coded(literals: &[u8], dict_id: u32, prev: &mut Tables, s: &mut EncScratch, compact: bool, out: &mut Vec<u8>) {
    let n = s.ll.len();
    debug_assert!(s.ml.len() == n && s.off.len() == n && s.extra.len() == n);
    debug_assert!(s.extra.iter().all(|&x| (x >> EXTRA_BITS) as u32 <= EXTRA_BITS));
    let base = out.len();
    // Room for the sub-header, written once the section sizes are known
    // (a compact one is shorter: the sections then slide back).
    let sub_bytes = if compact { SubHeader::COMPACT_MAX } else { SubHeader::BYTES };
    out.resize(base + sub_bytes, 0);

    // Literals. A small block (a small object) with a previous table
    // covering its symbols reuses it without pricing a fresh one: a
    // fresh table's ~90 bytes cannot pay for themselves on a literal
    // section that short, and building one is most of a small object's
    // encoding time.
    let hist: [u64; 256] = if literals.len() < 4096 { hist1(literals) } else { hist8::<256>(literals).map(|c| c as u64) };
    let small = literals.len() < SMALL_LITERALS;
    let covered = prev.lit_lengths.as_ref().map_or(false, |p| p.iter().zip(hist.iter()).all(|(&l, &h)| l > 0 || h == 0));
    let mut fresh: Option<[u8; 256]> = None;
    let lit_reuse = if small && covered {
        true
    } else {
        let lengths = huff8::lengths_for(&hist);
        let reuse = covered && {
            let p = prev.lit_lengths.as_ref().unwrap();
            let est_prev: u64 = (0..256).map(|s| hist[s] * p[s] as u64).sum();
            let est_new: u64 = (0..256).map(|s| hist[s] * lengths[s] as u64).sum();
            est_prev <= est_new + (crate::huffman::packed_lengths_v8_size(&lengths) as u64) * 8
        };
        fresh = Some(lengths);
        reuse
    };
    let lit_lengths = if lit_reuse { prev.lit_lengths.unwrap() } else { fresh.unwrap() };
    // Each section's framing: the block's, and for a compact block's
    // short sections a single stream.
    let framing = |symbols: usize| if compact { Framing::compact_for(symbols) } else { Framing::Wide };
    let lit_coded_size = huff8::coded_size(&hist, &lit_lengths) - if lit_reuse { crate::huffman::packed_lengths_v8_size(&lit_lengths) } else { 0 } + section_frame_bytes(framing(literals.len()));
    // A reused table costs nothing to carry, so even a short section is
    // coded when the bits say so; a fresh one is not worth building
    // under 64 literals.
    let lit_coded = literals.len() >= if lit_reuse { 8 } else { 64 } && lit_coded_size + literals.len() / 50 < literals.len();
    let lit_start = out.len();
    if lit_coded {
        if lit_reuse {
            // The previous table's codes, built once and kept with it.
            let codes = prev.lit_codes.get_or_insert_with(|| Arc::new(huff8::Codes::build(&lit_lengths)));
            huff8::encode_with(literals, codes, framing(literals.len()), out);
        } else {
            crate::huffman::pack_lengths_v8(&lit_lengths, out);
            let codes = huff8::Codes::build(&lit_lengths);
            huff8::encode_with(literals, &codes, framing(literals.len()), out);
            prev.lit_lengths = Some(lit_lengths);
            prev.lit_codes = Some(Arc::new(codes));
        }
    } else {
        out.extend_from_slice(literals);
    }
    let lit_size = out.len() - lit_start;

    // Sequence code streams: one up-front decision (not a per-stream, then
    // shadowed re-encode) -- reuse the previous block's three tables when
    // all three are present, cover this block's symbols, and together
    // cost no more than fresh tables plus their headers; otherwise
    // recompute and write fresh tables for all three.
    let ll_hist = hist8::<{ tans::MAX_SYMBOLS }>(&s.ll);
    let ml_hist = hist8::<{ tans::MAX_SYMBOLS }>(&s.ml);
    let off_hist = hist8::<{ tans::MAX_SYMBOLS }>(&s.off);
    let hists = [&ll_hist[..LL_SYMBOLS], &ml_hist[..ML_SYMBOLS], &off_hist[..OFF_SYMBOLS]];
    let prev_cost = match (&prev.ll, &prev.ml, &prev.off) {
        (Some(a), Some(b), Some(c)) => hists.iter().zip([a, b, c]).try_fold(0u64, |acc, (h, t)| table_cost(h, t).map(|b| acc + b)),
        _ => None,
    };
    // Fresh tables, built only when they might be used: a small block
    // (a small object) whose previous tables cover it reuses them
    // without pricing fresh ones (as with the literal table above).
    let mut fresh: Option<[Vec<u16>; 3]> = None;
    let fresh_tables = |fresh: &mut Option<[Vec<u16>; 3]>| {
        fresh.get_or_insert_with(|| [tans::normalize(&ll_hist, LL_SYMBOLS), tans::normalize(&ml_hist, ML_SYMBOLS), tans::normalize(&off_hist, OFF_SYMBOLS)]);
    };
    let mut seq_reuse = match prev_cost {
        Some(_) if n < SMALL_SEQUENCES => true,
        Some(prev_bits) => {
            fresh_tables(&mut fresh);
            let fresh_bits: u64 = hists.iter().zip(fresh.as_ref().unwrap()).map(|(h, t)| table_cost(h, t).expect("fresh table covers the data") + (((8 * tans_table_bytes(t)) as u64) << 16)).sum();
            prev_bits <= fresh_bits
        }
        None => false,
    };
    if !seq_reuse {
        fresh_tables(&mut fresh);
    }
    let (mut ll_counts, mut ml_counts, mut off_counts) = if seq_reuse {
        (prev.ll.clone().unwrap(), prev.ml.clone().unwrap(), prev.off.clone().unwrap())
    } else {
        let [a, b, c] = fresh.as_ref().unwrap();
        (Arc::from(&a[..]), Arc::from(&b[..]), Arc::from(&c[..]))
    };
    let seq_start = out.len();
    let mut sizes = [0usize; 3];
    let mut coded = [false; 3];
    for _ in 0..2 {
        let enc = if seq_reuse { Some(prev.seq_enc().clone()) } else { None };
        for (i, (codes, hist, counts)) in [(&s.ll, &ll_hist, &ll_counts), (&s.ml, &ml_hist, &ml_counts), (&s.off, &off_hist, &off_counts)].into_iter().enumerate() {
            let start = out.len();
            coded[i] = encode_codes(codes, hist, counts, enc.as_ref().map(|e| &e[i]), framing(n), &mut s.chunks, out);
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
        fresh_tables(&mut fresh);
        let [a, b, c] = fresh.as_ref().unwrap();
        ll_counts = Arc::from(&a[..]);
        ml_counts = Arc::from(&b[..]);
        off_counts = Arc::from(&c[..]);
        out.truncate(seq_start);
    }
    let seq_coded = coded.iter().all(|&c| c);

    // Extra bits: sub-stream k holds sequences k, k + 8, ... in order.
    let extra_start = out.len();
    let extra_framing = framing(n);
    let step = extra_framing.streams();
    write_section(out, extra_framing, n.div_ceil(step) * EXTRA_BITS as usize, |k, w| {
        let e = &s.extra[k.min(n)..];
        let mut i = 0;
        // SAFETY: at most ceil(n / streams) sequences of at most
        // EXTRA_BITS bits each go into this stream, the `max_bits` reserved.
        unsafe {
            if step == 1 {
                for &x in e {
                    put_wide(w, x & ((1u64 << EXTRA_BITS) - 1), (x >> EXTRA_BITS) as u32);
                }
                return;
            }
            while i + 8 < e.len() {
                // Two sequences per put when they fit MAX_PUT together
                // (nearly always: a sequence averages ~15 extra bits).
                let (x, y) = (e[i], e[i + 8]);
                let (nx, ny) = ((x >> EXTRA_BITS) as u32, (y >> EXTRA_BITS) as u32);
                let (vx, vy) = (x & ((1u64 << EXTRA_BITS) - 1), y & ((1u64 << EXTRA_BITS) - 1));
                if nx + ny <= crate::bits::MAX_PUT {
                    w.put(vx | vy << nx, nx + ny);
                } else {
                    put_wide(w, vx, nx);
                    put_wide(w, vy, ny);
                }
                i += 16;
            }
            if i < e.len() {
                let x = e[i];
                put_wide(w, x & ((1u64 << EXTRA_BITS) - 1), (x >> EXTRA_BITS) as u32);
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
    if compact {
        sub.write_compact(&mut hdr);
        let end = out.len();
        out.copy_within(base + sub_bytes..end, base + hdr.len());
        out.truncate(end - (sub_bytes - hdr.len()));
    } else {
        sub.write(&mut hdr);
    }
    out[base..base + hdr.len()].copy_from_slice(&hdr);

    if seq_coded {
        if !seq_reuse {
            prev.ll = Some(ll_counts);
            prev.ml = Some(ml_counts);
            prev.off = Some(off_counts);
            prev.seq_enc = None;
        }
    } else {
        prev.ll = None;
        prev.ml = None;
        prev.off = None;
        prev.seq_enc = None;
    }
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
    long: Vec<u32>,
    short: Vec<u32>,
    /// Index bits of each table: `DFAST_LONG_BITS` / `DFAST_SHORT_BITS`
    /// for a large input, fewer for a small one (`size_for`), so a call
    /// on a small object clears kilobytes, not megabytes.
    lbits: u32,
    sbits: u32,
}

impl DfastTables {
    /// Both tables empty. An entry is a tagged position (see `POS_BITS`);
    /// 0 doubles as empty, which is harmless: position 0 is verified by
    /// content like any other candidate. So is an entry left over from a
    /// previous input: under the modular distances `POS_BITS` describes,
    /// a candidate at or past the current position is not specially
    /// rejected, just let through to the same byte-for-byte check as any
    /// other, so reusing the tables across inputs without clearing is
    /// safe -- but a leftover can still win a match the empty table
    /// would not have found, so for output that depends only on the
    /// input, `clear` first (as `compress_into_max` does per call).
    pub fn new() -> Box<Self> {
        Box::new(DfastTables {
            long: vec![0; 1 << DFAST_LONG_BITS],
            short: vec![0; 1 << DFAST_SHORT_BITS],
            lbits: DFAST_LONG_BITS,
            sbits: DFAST_SHORT_BITS,
        })
    }

    /// Empty both tables (2 MB of stores at full size).
    pub fn clear(&mut self) {
        self.long[..1 << self.lbits].fill(0);
        self.short[..1 << self.sbits].fill(0);
    }

    /// Empty the tables and size them for an input of `len` bytes: the
    /// full 18 bits from 128 KB up, one bit per halving below, at least
    /// 10. Output depends only on the input either way.
    pub fn clear_for(&mut self, len: usize) {
        let bits = (usize::BITS - len.max(1).leading_zeros() + 1).clamp(10, DFAST_LONG_BITS);
        self.lbits = bits;
        self.sbits = bits.min(DFAST_SHORT_BITS);
        self.clear();
    }

    /// Index every position of `input[..end]` (a dictionary: the blocks
    /// parsed after `end` may match into it) in both tables, later
    /// positions winning a slot (they are the nearer, cheaper offsets).
    pub fn seed(&mut self, input: &[u8], end: usize) {
        for (pos, w) in input[..end].windows(8).enumerate() {
            let w = u64::from_ne_bytes(w.try_into().unwrap());
            let (i, m) = long_slot(w, pos, self.lbits);
            self.long[i] = m;
            let (i, m) = short_slot(w, pos, self.sbits);
            self.short[i] = m;
        }
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
fn long_slot(w: u64, pos: usize, bits: u32) -> (usize, u32) {
    let h = h8(w);
    ((h >> (64 - bits)) as usize, (pos & POS_MASK) as u32 | ((h >> (64 - bits - 8)) as u32) << POS_BITS)
}
#[inline(always)]
fn short_slot(w: u64, pos: usize, bits: u32) -> (usize, u32) {
    let h = h5(w);
    ((h >> (64 - bits)) as usize, (pos & POS_MASK) as u32 | ((h >> (64 - bits - 8)) as u32) << POS_BITS)
}
#[inline(always)]
unsafe fn eq4(a: *const u8, b: *const u8) -> bool {
    std::ptr::read_unaligned(a as *const u32) == std::ptr::read_unaligned(b as *const u32)
}
/// The candidate an entry `e` names for a probe at `pos` whose own
/// tagged entry is `mine`: its position if the tags agree and its
/// distance is at least 1 and within the window (and the input), else
/// None.
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

/// A position's two table slots: per table the index and the tagged
/// entry it writes there.
#[derive(Clone, Copy)]
struct Slot {
    il: usize,
    ml: u32,
    is: usize,
    ms: u32,
    /// The two hashes, for a dictionary's full-size tables.
    hl: u64,
    hs: u64,
}

impl Slot {
    /// Reads 8 bytes at `pos`: the caller guarantees `pos + 8 <= block_end`.
    #[inline(always)]
    unsafe fn at(src: *const u8, pos: usize, lbits: u32, sbits: u32) -> Slot {
        let w = std::ptr::read_unaligned(src.add(pos) as *const u64);
        let (il, ml) = long_slot(w, pos, lbits);
        let (is, ms) = short_slot(w, pos, sbits);
        Slot { il, ml, is, ms, hl: h8(w), hs: h5(w) }
    }

    /// The slot's indexes in a dictionary's full-size tables, and the
    /// tags the entries there must carry (the same hash bits below the
    /// index, at full size).
    #[inline(always)]
    fn dict_index(&self) -> (usize, u32, usize, u32) {
        let il = (self.hl >> (64 - DFAST_LONG_BITS)) as usize;
        let ml = ((self.hl >> (64 - DFAST_LONG_BITS - 8)) as u32) << POS_BITS;
        let is = (self.hs >> (64 - DFAST_SHORT_BITS)) as usize;
        let ms = ((self.hs >> (64 - DFAST_SHORT_BITS - 8)) as u32) << POS_BITS;
        (il, ml, is, ms)
    }
}

/// The match at `pos` (slot `c`, whose table entries `el` / `es` were
/// loaded before `pos` was written into the tables): the first of the
/// repeat offsets (4-byte compare each), the long candidate and the
/// short one that verifies (4+ bytes; a long candidate whose tag agrees
/// shares 8 in practice), extended to at most `block_end`; `(usize::MAX,
/// 0)` if none. `pos + 8 <= block_end` is the caller's guarantee.
/// A match found by `probe`: where its source bytes start, its offset,
/// and its length. `off == 0` is no match.
#[derive(Clone, Copy)]
struct Found {
    src: *const u8,
    off: usize,
    len: usize,
}

const NONE: Found = Found { src: std::ptr::null(), off: 0, len: 0 };

/// A prepared dictionary's tables for the parse: seeded on its content
/// (positions 0..len there), read only, consulted after the input's own.
pub struct DictTables {
    pub(crate) tables: Box<DfastTables>,
    pub(crate) content: *const u8,
    pub(crate) len: usize,
}

#[inline(always)]
unsafe fn probe<const D: bool>(src: *const u8, pos: usize, block_end: usize, c: &Slot, el: u32, es: u32, r: &[u32; 3], dict: Option<&DictTables>, eld: u32, esd: u32) -> Found {
    use crate::finder::{MatchLen, ScalarMatch};
    let p = src.add(pos);
    for &o in r {
        let o = o as usize;
        if o <= pos && eq4(p, p.sub(o)) {
            return Found { src: p.sub(o), off: o, len: 4 + ScalarMatch::prefix(p.add(4), p.sub(o).add(4), block_end - pos - 4) };
        }
    }
    let mut best = NONE;
    for (e, mine) in [(el, c.ml), (es, c.ms)] {
        if let Some(cand) = candidate(e, mine, pos) {
            let len = ScalarMatch::prefix(p, src.add(cand), block_end - pos);
            if len >= 4 {
                best = Found { src: src.add(cand), off: pos - cand, len };
                break;
            }
        }
    }
    if D {
        // The dictionary: an entry names a position in its content; the
        // offset counts through the content's tail and this input. The
        // longer of the input's and the dictionary's candidates wins.
        let d = dict.unwrap_unchecked();
        let (_, tl, _, ts) = c.dict_index();
        for (e, mine) in [(eld, tl), (esd, ts)] {
            if (e ^ mine) >> POS_BITS == 0 {
                let epos = (e as usize) & POS_MASK;
                let off = pos + d.len - epos;
                if epos < d.len && off < MAX_WINDOW as usize {
                    let len = ScalarMatch::prefix(p, d.content.add(epos), (block_end - pos).min(d.len - epos));
                    if len >= 4 && len > best.len {
                        best = Found { src: d.content.add(epos), off, len };
                        break;
                    }
                }
            }
        }
    }
    best
}

#[cold]
#[inline(never)]
fn lazy_win(pos: &mut usize, found: &mut Found, better: Found) {
    *pos += 1;
    *found = better;
}

/// Parse `input[block_start..block_start + block_len]` into `seqs` and
/// `literals` (appended) and, as `encode_block_with` would code them,
/// into `codes` (cleared first; `encode_block_coded` takes it from
/// there). Offsets are absolute distances, at least 1 and under
/// MAX_WINDOW; the last sequence is literal-only. `t` carries the window
/// across the blocks of one input. `reps` finds matches and codes them,
/// so it must be what `encode_block`'s `Reps` starts a block with: the
/// caller passes `[1, 4, 8]` at every block. The choice of match depends
/// on what `t` holds, so the output of an input's first block is only
/// determined by the input when `t` starts empty (`DfastTables::clear`).
///
/// Software pipelined as zstd's double-fast loop is: the slot and table
/// entries of `pos + 1` are computed and loaded while `pos` is checked,
/// so a mispredicted check does not restart that load chain; a hit's
/// lazy step and a step-1 miss's next probe both use them.
pub fn find_sequences_dfast(input: &[u8], block_start: usize, block_len: usize, t: &mut DfastTables, reps: &mut [u32; 3], seqs: &mut Vec<Sequence>, literals: &mut Vec<u8>, codes: &mut EncScratch) {
    find_sequences_dfast_impl::<false>(input, block_start, block_len, t, None, reps, seqs, literals, codes)
}

/// `find_sequences_dfast` with a prepared dictionary's tables: its
/// content is history before `input` (offsets reach through it), found
/// through its own tables after the input's.
pub fn find_sequences_dfast_dict(input: &[u8], block_start: usize, block_len: usize, t: &mut DfastTables, dict: &DictTables, reps: &mut [u32; 3], seqs: &mut Vec<Sequence>, literals: &mut Vec<u8>, codes: &mut EncScratch) {
    find_sequences_dfast_impl::<true>(input, block_start, block_len, t, Some(dict), reps, seqs, literals, codes)
}

#[inline(always)]
fn find_sequences_dfast_impl<const D: bool>(input: &[u8], block_start: usize, block_len: usize, t: &mut DfastTables, dict: Option<&DictTables>, reps: &mut [u32; 3], seqs: &mut Vec<Sequence>, literals: &mut Vec<u8>, codes: &mut EncScratch) {
    use crate::finder::MatchLen;
    let src = input.as_ptr();
    let (lb, sb) = (t.lbits, t.sbits);
    debug_assert!(!D || (dict.unwrap().tables.lbits == DFAST_LONG_BITS && dict.unwrap().tables.sbits == DFAST_SHORT_BITS));
    let block_end = block_start + block_len;
    assert!(block_len <= MAX_BLOCK_SIZE, "the extras packing relies on block_len <= MAX_BLOCK_SIZE");
    debug_assert_eq!(*reps, [1, 4, 8], "written codes assume the decoder's fresh Reps");
    assert!(block_end <= input.len(), "block past the input");
    debug_assert!(reps.iter().all(|&o| o >= 1), "a zero repeat offset would verify against itself");
    // Every probe reads 8 bytes at `pos` and 8 at `pos + 1`; extensions
    // stop at block_end.
    let limit = block_end.saturating_sub(9).max(block_start);
    let mut anchor = block_start;
    let mut pos = block_start;
    let mut step_nb: u32 = 1 << DFAST_SKIP_STRENGTH;
    let mut r = *reps;
    // The literal copies below run 16 bytes wild past their length.
    literals.reserve(block_len + 16);
    codes.clear();
    codes.reserve_for(block_len);

    if pos < limit {
        unsafe {
            let mut cur = Slot::at(src, pos, lb, sb);
            let mut el = t.long[cur.il];
            let mut es = t.short[cur.is];
            loop {
                t.long[cur.il] = cur.ml;
                t.short[cur.is] = cur.ms;
                let nxt = Slot::at(src, pos + 1, lb, sb);
                let el1 = t.long[nxt.il];
                let es1 = t.short[nxt.is];
                let (eld, esd) = if D {
                    let d = &dict.unwrap_unchecked().tables;
                    let (il, _, is, _) = cur.dict_index();
                    (d.long[il], d.short[is])
                } else {
                    (0, 0)
                };
                let mut found = probe::<D>(src, pos, block_end, &cur, el, es, &r, dict, eld, esd);
                if found.off == 0 {
                    let step = (step_nb >> DFAST_SKIP_STRENGTH) as usize;
                    step_nb += 1;
                    pos += step;
                    if pos >= limit {
                        break;
                    }
                    if step == 1 {
                        cur = nxt;
                        el = el1;
                        es = es1;
                    } else {
                        cur = Slot::at(src, pos, lb, sb);
                        el = t.long[cur.il];
                        es = t.short[cur.is];
                    }
                    continue;
                }
                step_nb = 1 << DFAST_SKIP_STRENGTH;
                // Lazy: a long candidate one byte later that is 4+ bytes
                // longer wins. pos + 1 is indexed either way.
                t.long[nxt.il] = nxt.ml;
                t.short[nxt.is] = nxt.ms;
                if let Some(c) = candidate(el1, nxt.ml, pos + 1) {
                    let rc1 = crate::finder::ScalarMatch::prefix(src.add(pos + 1), src.add(c), block_end - pos - 1);
                    if rc1 >= found.len + 4 {
                        // Out of line so this stays a (rarely taken)
                        // branch: as selects, the next position would
                        // wait for this probe's whole load chain.
                        let better = Found { src: src.add(c), off: pos + 1 - c, len: rc1 };
                        lazy_win(&mut pos, &mut found, better);
                    }
                }
                // Back-match into the pending literals: the source steps
                // back with the match, within its buffer (the input, or
                // the dictionary's content).
                let mut mpos = pos;
                let mut c = found.src;
                let mut rc = found.len;
                let lo = if D && found.off > pos { dict.unwrap_unchecked().content } else { src };
                while mpos > anchor && c > lo && *src.add(mpos - 1) == *c.sub(1) {
                    mpos -= 1;
                    c = c.sub(1);
                    rc += 1;
                }
                let offset = found.off as u32;
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
                // The codes, and the same update as Reps::code_for: the
                // offset moves to the front and the entries before its old
                // slot shift back one.
                {
                    use std::hint::select_unpredictable as sel;
                    let (e0, e1, e2) = (offset == r[0], offset == r[1], offset == r[2]);
                    codes.push_codes(ll as u32, rc as u32, offset, sel(e0, 0, sel(e1, 1, sel(e2, 2, 3))));
                    r = [offset, sel(e0, r[1], r[0]), sel(e0 | e1, r[2], r[1])];
                }
                pos = mpos + rc;
                anchor = pos;
                if pos >= limit {
                    break;
                }
                // Index the match's second position and its tail (zstd's
                // insertions) so runs keep hashing.
                let w = std::ptr::read_unaligned(src.add(mpos + 2) as *const u64);
                let (i, m) = long_slot(w, mpos + 2, lb);
                t.long[i] = m;
                let (i, m) = short_slot(w, mpos + 2, sb);
                t.short[i] = m;
                let w = std::ptr::read_unaligned(src.add(pos - 2) as *const u64);
                let (i, m) = long_slot(w, pos - 2, lb);
                t.long[i] = m;
                let w = std::ptr::read_unaligned(src.add(pos - 1) as *const u64);
                let (i, m) = short_slot(w, pos - 1, sb);
                t.short[i] = m;
                cur = Slot::at(src, pos, lb, sb);
                el = t.long[cur.il];
                es = t.short[cur.is];
            }
        }
    }
    literals.extend_from_slice(&input[anchor..block_end]);
    seqs.push(Sequence { lit_len: (block_end - anchor) as u32, match_len: 0, offset: 0 });
    // The literal-only last sequence codes as a match of MIN_MATCH with
    // rep code 0 and no extra bits, as `encode_block_with` codes it.
    // SAFETY: `reserve_for` above counted this entry.
    unsafe { codes.push_codes((block_end - anchor) as u32, MIN_MATCH, 1, 0) };
    *reps = r;
}
