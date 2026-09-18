//! Format v7 block decoder: three passes over thread-local scratch.
//!   1. sequence streams -> lit_len / match_len / offset arrays
//!   2. literal stream -> literal buffer
//!   3. copy loop over the arrays and the buffer
//!
//! Contract with `v7_encode`: the last sequence of a block is literal-only
//! (ml code 0, no offset); an interior ml code 0 is a match of length 3.
//! Every section size and sub-stream length is checked against its
//! container before any reader touches it, so corrupt input yields an
//! error, never a panic or an out-of-bounds access.
use crate::bits::{BitReader, PAD};
use crate::error::{CodecError, Result};
use crate::format::MAX_BLOCK_SIZE;
use crate::huff8;
use crate::tans;
use crate::v7_encode::{payload_layout, Layout};
use crate::v7_format::*;

/// Every sequence but the last carries a match of at least MIN_MATCH bytes.
const MAX_SEQ: usize = MAX_BLOCK_SIZE / MIN_MATCH as usize + 1;

/// Sequences are kept in groups of eight (one per extra-bits sub-stream,
/// one walk batch): a group holds the eight ll codes, then the eight ml
/// codes, then the eight offset codes (`codes`), and likewise the eight
/// literal lengths, match lengths and offsets (`seq`). Sequence i is at
/// `i / 8 * GROUP + i % 8` plus the field's lane offset. One pointer per
/// array in the walk instead of three, which is what keeps its eight
/// stream positions in registers.
pub const GROUP: usize = 24;
const GROUPS: usize = MAX_SEQ.div_ceil(8);

/// Index of sequence `i`'s lane-0 entry in a grouped array.
#[inline(always)]
fn at(i: usize) -> usize {
    i / 8 * GROUP + i % 8
}

pub struct Scratch {
    /// Grouped ll, ml and offset codes (see `GROUP`).
    pub codes: Vec<u8>,
    /// Grouped literal lengths, match lengths and offsets.
    pub seq: Vec<u32>,
    /// WILD_MARGIN bytes past MAX_BLOCK_SIZE for the wild copies' reads.
    pub lits: Vec<u8>,
}

/// A wild copy runs past its sequence's end: the fixed 3x32 tails of
/// `copy_seq_wild` write 128 bytes for a 33-byte run, 95 past its end.
/// A sequence is copied wild only with this much room left in `dst` and
/// in the literal buffer, exactly otherwise.
const WILD_MARGIN: usize = 96;

impl Scratch {
    pub fn new() -> Self {
        Scratch {
            codes: vec![0; GROUPS * GROUP],
            seq: vec![0; GROUPS * GROUP],
            lits: vec![0; MAX_BLOCK_SIZE + WILD_MARGIN],
        }
    }
}

thread_local! {
    static SCRATCH: std::cell::RefCell<Scratch> = std::cell::RefCell::new(Scratch::new());
}

pub fn with_scratch<T>(f: impl FnOnce(&mut Scratch) -> T) -> T {
    SCRATCH.with(|s| f(&mut s.borrow_mut()))
}

/// The previous block's decode tables, kept exactly as the encoder keeps
/// its own: the literal table survives raw-literal blocks, the three
/// sequence tables survive only blocks where all three streams are coded.
pub struct DecTables {
    pub lit: Option<huff8::Table>,
    pub ll: Option<tans::DecodeTable>,
    pub ml: Option<tans::DecodeTable>,
    pub off: Option<tans::DecodeTable>,
}

impl DecTables {
    pub fn none() -> Self {
        DecTables { lit: None, ll: None, ml: None, off: None }
    }
}

fn corrupt(msg: &'static str) -> CodecError {
    CodecError::CorruptedBitstream(msg)
}

/// Split a section's tail into 8 padded sub-streams behind a size table.
fn substreams(sec: &[u8]) -> Result<[&[u8]; 8]> {
    if sec.len() < 32 {
        return Err(corrupt("v7: sub-stream table truncated"));
    }
    let mut pos = 32usize;
    let mut out: [&[u8]; 8] = [&[]; 8];
    for k in 0..8 {
        let n = u32::from_le_bytes(sec[k * 4..k * 4 + 4].try_into().unwrap()) as usize;
        if n < PAD || n > sec.len() - pos {
            return Err(corrupt("v7: sub-stream out of section"));
        }
        out[k] = &sec[pos..pos + n];
        pos += n;
    }
    if pos != sec.len() {
        return Err(corrupt("v7: section has trailing bytes"));
    }
    Ok(out)
}

/// Decode one code stream (or copy it raw) into its lane of the grouped
/// `codes` (`codes` starts at the lane: code i goes to `at(i)`).
fn code_stream(sec: &[u8], coded: bool, reuse: bool, n_symbols: usize, n: usize, prev: &mut Option<tans::DecodeTable>, codes: &mut [u8]) -> Result<()> {
    if !coded {
        if sec.len() != n {
            return Err(corrupt("v7: raw code stream length"));
        }
        if sec.iter().any(|&c| c as usize >= n_symbols) {
            return Err(corrupt("v7: raw code out of range"));
        }
        for (i, &c) in sec.iter().enumerate() {
            codes[at(i)] = c;
        }
        return Ok(());
    }
    let mut pos = 0usize;
    if !reuse {
        let ns = *sec.first().ok_or(corrupt("v7: table truncated"))? as usize;
        if ns != n_symbols || sec.len() < 1 + 2 * ns {
            return Err(corrupt("v7: table symbol count"));
        }
        let mut counts = [0u16; tans::MAX_SYMBOLS];
        for (s, c) in counts[..ns].iter_mut().enumerate() {
            *c = u16::from_le_bytes([sec[1 + 2 * s], sec[2 + 2 * s]]);
        }
        // Rebuilt in place: the table lives in `DecTables` across blocks.
        if !prev.get_or_insert_with(tans::DecodeTable::empty).rebuild(&counts[..ns]) {
            *prev = None;
            return Err(corrupt("v7: tANS counts"));
        }
        pos = 1 + 2 * ns;
    }
    let table = prev.as_ref().ok_or(corrupt("v7: table reuse without a table"))?;
    let streams = substreams(&sec[pos..])?;
    tans::decode8_rows(table, &streams, n, GROUP, codes).map_err(|_| corrupt("v7: code stream overrun"))?;
    // A table built from `n_symbols` counts only ever yields those symbols.
    debug_assert!((0..n).all(|i| (codes[at(i)] as usize) < n_symbols));
    Ok(())
}

/// One field of the walk: `e` is a `*_WALK` entry, `base << 32 | mask <<
/// 8 | nb`; the value is `base` plus the low `nb` bits of `w`. Returns
/// the value, `w` shifted past them, and `nb`.
#[inline(always)]
fn field(w: u64, e: u64) -> (u32, u64, u32) {
    let nb = e as u32 & 0xff;
    let extra = w as u32 & (e as u32 >> 8);
    ((extra as u64 + (e >> 32)) as u32, w >> nb, nb)
}

/// Bytes a sequence can advance a stream: its three fields are at most
/// 19 + 19 + 20 = 58 bits (codes 31, 31 and 23), so the load address moves
/// by at most 8 bytes per sequence whatever the code bytes hold.
const SEQ_BYTES: usize = 8;

/// Sequences the fast walk can take on one stream before a load might
/// start past `last`: the next load is at `at`, each later one <= SEQ_BYTES
/// further. Mirrors `FastReader::safe_refills`.
#[inline(always)]
fn safe_seqs(at: usize, last: usize) -> usize {
    if at > last {
        0
    } else {
        (last - at) / SEQ_BYTES + 1
    }
}

/// Pass 1: the three code streams, then one walk over the extra bits in
/// the encoder's order (ll, ml, off per sequence, sub-stream i % 8).
/// Fills scratch.ll/ml/off; returns (literal total, match total).
///
/// The walk has the `huff8::decode` shape -- an unclamped batch loop with
/// a proven load bound, then the clamped `BitReader`s for the tail -- but
/// its hot state is one bit position per stream, not a `FastReader`: a
/// valid sequence's extra bits are at most 18 + 18 + 20 = 56 (values up to
/// MAX_BLOCK_SIZE = 2^18, offsets below 2^21), and one unaligned 8-byte
/// load shifted by the sub-byte position holds at least 57, so each
/// sequence is one load, three field extractions and one add to its
/// stream's position, with no accumulator, count or refill to maintain
/// (8 live registers for the 8 streams instead of 24). A corrupt stream
/// asking for 58 bits reads a zero for the last one and advances exactly
/// anyway; the values are garbage either way and the checks after the
/// walk catch them. Each iteration handles 8 sequences, one per stream;
/// the batch count is bounded by every stream's `safe_seqs` and stops
/// short of sequence n - 1 (which may be literal-only and always goes
/// through the tail). The tail's clamped readers start at the positions
/// the walk reached (`BitReader::new_at`), which keeps `overrun` exact.
fn sequences(payload: &[u8], layout: &Layout, n: usize, prev: &mut DecTables, s: &mut Scratch) -> Result<(usize, usize)> {
    let sub = &layout.sub;
    let coded = |i: usize| sub.coded & (1 << i) != 0;
    let reuse = sub.reuse & 0b10 != 0;
    code_stream(&payload[layout.sections[S_LL].clone()], coded(S_LL), reuse, LL_SYMBOLS, n, &mut prev.ll, &mut s.codes)?;
    code_stream(&payload[layout.sections[S_ML].clone()], coded(S_ML), reuse, ML_SYMBOLS, n, &mut prev.ml, &mut s.codes[8..])?;
    code_stream(&payload[layout.sections[S_OFF].clone()], coded(S_OFF), reuse, OFF_SYMBOLS, n, &mut prev.off, &mut s.codes[16..])?;
    if !(coded(S_LL) && coded(S_ML) && coded(S_OFF)) {
        // The encoder drops its tables whenever any stream went raw.
        prev.ll = None;
        prev.ml = None;
        prev.off = None;
    }

    let extra = substreams(&payload[layout.sections[S_EXTRA].clone()])?;
    // Stream k's next bit is `b<k>`, an absolute bit address (`ptr * 8 +
    // bit`, as in `tans::decode8`): eight scalars, not an array, so they
    // stay in registers, and no base register. `lasts[k]` is the last
    // address a load on stream k may start at.
    let start = |k: usize| (extra[k].as_ptr() as usize * 8) as u64;
    let (mut b0, mut b1, mut b2, mut b3) = (start(0), start(1), start(2), start(3));
    let (mut b4, mut b5, mut b6, mut b7) = (start(4), start(5), start(6), start(7));
    let lasts: [usize; 8] = std::array::from_fn(|k| extra[k][extra[k].len() - PAD..].as_ptr() as usize);
    let mut reps = Reps::new();
    let mut o = 0usize;
    loop {
        // Whole batches of 8 that stop short of sequence n - 1 and of any
        // stream's safe margin.
        let mut iters = (n - o).saturating_sub(1) / 8;
        for (k, b) in [b0, b1, b2, b3, b4, b5, b6, b7].into_iter().enumerate() {
            iters = iters.min(safe_seqs((b >> 3) as usize, lasts[k]));
        }
        if iters == 0 {
            break;
        }
        let (g0, g1) = (o / 8 * GROUP, (o / 8 + iters) * GROUP);
        for (c, out) in s.codes[g0..g1].chunks_exact(GROUP).zip(s.seq[g0..g1].chunks_exact_mut(GROUP)) {
            // One sequence per stream, written out (a loop this size is
            // not unrolled on its own, and the positions must not become
            // an indexed array).
            macro_rules! seq {
                ($k:literal, $b:ident) => {{
                    // SAFETY: `iters` <= every stream's safe_seqs at the
                    // start of this batch run and each iteration advances
                    // a stream by at most SEQ_BYTES, so this load starts at
                    // or before lasts[k], i.e. its 8 bytes are inside
                    // sub-stream k.
                    let w = unsafe { std::ptr::read_unaligned(($b >> 3) as *const u64) } >> ($b & 7);
                    let (ll, w, n1) = field(w, LL_WALK[c[$k] as usize]);
                    let (ml, w, n2) = field(w, ML_WALK[c[8 + $k] as usize]);
                    let offc = c[16 + $k];
                    let (ov, _, n3) = field(w, OFF_WALK[offc as usize]);
                    $b += n1 as u64;
                    $b += n2 as u64;
                    $b += n3 as u64;
                    out[$k] = ll;
                    out[8 + $k] = ml;
                    out[16 + $k] = reps.update(offc, ov);
                }};
            }
            seq!(0, b0);
            seq!(1, b1);
            seq!(2, b2);
            seq!(3, b3);
            seq!(4, b4);
            seq!(5, b5);
            seq!(6, b6);
            seq!(7, b7);
        }
        o += 8 * iters;
    }
    let rd = |k: usize, b: u64| BitReader::new_at(extra[k], (b - start(k)) as usize);
    let mut ers: [BitReader; 8] = [rd(0, b0), rd(1, b1), rd(2, b2), rd(3, b3), rd(4, b4), rd(5, b5), rd(6, b6), rd(7, b7)];

    // Tail: the last sequence, whatever a short stream's margin left, and
    // the literal-only rule -- on the clamped readers.
    for i in o..n {
        let r = &mut ers[i % 8];
        let j = at(i);
        let llc = s.codes[j];
        s.seq[j] = ll_value(llc, r.get(extra_bits_of_code(Kind::Ll, llc) as u32) as u32);
        let mlc = s.codes[8 + j];
        if mlc == 0 && i == n - 1 {
            s.seq[8 + j] = 0;
            s.seq[16 + j] = 0;
            continue;
        }
        let ml = ml_value(mlc, r.get(extra_bits_of_code(Kind::Ml, mlc) as u32) as u32);
        let offc = s.codes[16 + j];
        let off = reps.resolve(offc, r.get(extra_bits_of_code(Kind::Off, offc) as u32) as u32);
        s.seq[8 + j] = ml;
        s.seq[16 + j] = off;
    }
    // Totals over the arrays rather than in the walk (two fewer live
    // values there; this is a vectorised pass over L2-resident data): the
    // whole groups, then the last, partial one. The literal-only last
    // sequence stored ml = 0, so it adds nothing.
    let (mut lit_total, mut match_total) = (0usize, 0usize);
    for g in s.seq[..n / 8 * GROUP].chunks_exact(GROUP) {
        lit_total += g[..8].iter().map(|&v| v as usize).sum::<usize>();
        match_total += g[8..16].iter().map(|&v| v as usize).sum::<usize>();
    }
    for i in n / 8 * 8..n {
        lit_total += s.seq[at(i)] as usize;
        match_total += s.seq[8 + at(i)] as usize;
    }
    if ers.iter().any(|r| r.overrun()) {
        return Err(corrupt("v7: extra bits overrun"));
    }
    Ok((lit_total, match_total))
}

/// Pass 2: literals into scratch.lits[..n_lit].
fn literals(payload: &[u8], layout: &Layout, n_lit: usize, prev: &mut DecTables, s: &mut Scratch) -> Result<()> {
    let sec = &payload[layout.sections[S_LIT].clone()];
    if layout.sub.coded & (1 << S_LIT) == 0 {
        if sec.len() != n_lit {
            return Err(corrupt("v7: raw literal length"));
        }
        s.lits[..n_lit].copy_from_slice(sec);
        return Ok(());
    }
    let mut pos = 0usize;
    if layout.sub.reuse & 1 == 0 {
        if sec.len() < huff8::TABLE_BYTES {
            return Err(corrupt("v7: literal table truncated"));
        }
        let lengths = crate::huffman::unpack_lengths(&sec[..huff8::TABLE_BYTES]);
        prev.lit = Some(huff8::Table::build(&lengths).ok_or(corrupt("v7: literal code lengths"))?);
        pos = huff8::TABLE_BYTES;
    }
    let table = prev.lit.as_ref().ok_or(corrupt("v7: literal table reuse without a table"))?;
    let streams = substreams(&sec[pos..])?;
    huff8::decode(table, &streams, n_lit, &mut s.lits).map_err(|_| corrupt("v7: literal stream overrun"))
}

/// One sequence, wild: `ll` literals from `lit`, then `ml` (>= 1) bytes
/// from `d + ll - off`, in 32-byte stores that may run up to WILD_MARGIN
/// past the sequence's end. The shape measured in
/// `neon_decompress::copy_run`: one unconditional 32-byte copy, fixed
/// 3x32 tails to 128 bytes, a loop only past that, and a cold path for
/// offsets under 32.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn copy_seq_wild(lit: *const u8, d: *mut u8, ll: usize, ml: usize, off: usize) {
    use crate::neon_decompress::{copy32, short_match};
    copy32(lit, d);
    if ll > 32 {
        copy32(lit.add(32), d.add(32));
        copy32(lit.add(64), d.add(64));
        copy32(lit.add(96), d.add(96));
        if ll > 128 {
            let mut k = 128;
            while k < ll {
                copy32(lit.add(k), d.add(k));
                k += 32;
            }
        }
    }
    let d = d.add(ll);
    let src = d.sub(off);
    if off >= 32 {
        // Each 32-byte read ends at or before its write starts.
        copy32(src, d);
        if ml > 32 {
            copy32(src.add(32), d.add(32));
            copy32(src.add(64), d.add(64));
            copy32(src.add(96), d.add(96));
            if ml > 128 {
                let mut k = 128;
                while k < ml {
                    copy32(src.add(k), d.add(k));
                    k += 32;
                }
            }
        }
    } else {
        short_match(src, d, off, ml);
    }
}

/// Same contract, portable: 32-byte copy loops, a byte loop for
/// overlapping matches.
#[cfg(not(target_arch = "aarch64"))]
#[inline(always)]
unsafe fn copy_seq_wild(lit: *const u8, d: *mut u8, ll: usize, ml: usize, off: usize) {
    let mut k = 0;
    while k < ll {
        std::ptr::copy_nonoverlapping(lit.add(k), d.add(k), 32);
        k += 32;
    }
    let d = d.add(ll);
    let src = d.sub(off);
    if off >= 32 {
        let mut k = 0;
        while k < ml {
            std::ptr::copy_nonoverlapping(src.add(k), d.add(k), 32);
            k += 32;
        }
    } else {
        for k in 0..ml {
            *d.add(k) = *src.add(k);
        }
    }
}

/// Same contract, exact: nothing past the sequence's end is touched. The
/// last WILD_MARGIN bytes of a block whose `dst` has no slack (the
/// parallel path's) take this.
#[cold]
#[inline(never)]
unsafe fn copy_seq_exact(lit: *const u8, d: *mut u8, ll: usize, ml: usize, off: usize) {
    std::ptr::copy_nonoverlapping(lit, d, ll);
    let d = d.add(ll);
    let src = d.sub(off);
    for k in 0..ml {
        *d.add(k) = *src.add(k);
    }
}

/// Pass 3: copies. The caller has checked the totals: `sum(ll) == n_lit
/// <= MAX_BLOCK_SIZE` and `sum(ll + ml) == uncompressed_len <= dst.len()`,
/// so no sequence ends past the block or the decoded literals (prefix
/// sums of non-negative lengths), and a wild literal copy, which reads at
/// most WILD_MARGIN past its run, stays inside the literal buffer
/// (`MAX_BLOCK_SIZE + WILD_MARGIN` bytes). The fast loop therefore checks
/// only the offset against the window and the wild margin in `dst`; the
/// last sequence (the only one with `ml == 0`: pass 1 stores >= MIN_MATCH
/// everywhere else, for any input) and whatever lies within WILD_MARGIN
/// of `dst`'s end take the exact path with every check.
///
/// SAFETY: `buffer_start` must point into the same allocation as `dst`,
/// at or before `dst.as_ptr()`, with every byte between them initialised
/// (the window). `uncompressed_len <= dst.len()`, `n <= MAX_SEQ`,
/// `n_lit <= MAX_BLOCK_SIZE` and the totals above are the caller's checks.
unsafe fn copies(s: &Scratch, n: usize, n_lit: usize, dst: &mut [u8], buffer_start: *const u8, uncompressed_len: usize) -> Result<usize> {
    debug_assert!((0..n).map(|i| s.seq[at(i)] as usize).sum::<usize>() == n_lit && n_lit <= MAX_BLOCK_SIZE);
    debug_assert!((0..n).map(|i| s.seq[8 + at(i)] as usize).sum::<usize>() + n_lit == uncompressed_len && uncompressed_len <= dst.len());
    let base = dst.as_mut_ptr();
    let seq = s.seq.as_ptr();
    let mut d = base;
    let mut lp = s.lits.as_ptr();
    let mut i = 0usize;
    // A sequence ending at or before `wild_end` may be copied wild; a
    // `dst` shorter than the margin has no such point (not even `base`:
    // an empty sequence there would still take the 32-byte store).
    if dst.len() >= WILD_MARGIN {
        let wild_end = base.add(dst.len() - WILD_MARGIN);
        // Whole groups of eight before the last sequence, through one
        // group pointer; the rest (under eight sequences, plus whatever
        // lies within the margin) takes the exact loop below.
        'groups: while i + 8 < n {
            let g = seq.add(i / 8 * GROUP);
            for k in 0..8 {
                let ll = *g.add(k) as usize;
                let ml = *g.add(8 + k) as usize;
                let off = *g.add(16 + k) as usize;
                let m = d.add(ll);
                let available = m.offset_from(buffer_start) as usize;
                // One compare for both `off == 0` (wraps) and `off > available`.
                if off.wrapping_sub(1) >= available {
                    return Err(CodecError::OffsetOutOfBounds { offset: off, available });
                }
                let end = m.add(ml);
                if end > wild_end {
                    break 'groups;
                }
                copy_seq_wild(lp, d, ll, ml, off);
                d = end;
                lp = lp.add(ll);
                i += 1;
            }
        }
    }
    let mut written = d.offset_from(base) as usize;
    let mut lp = lp.offset_from(s.lits.as_ptr()) as usize;
    let lits = s.lits.as_ptr();
    let window = (base as *const u8).offset_from(buffer_start) as usize;
    while i < n {
        let j = at(i);
        let ll = *seq.add(j) as usize;
        let ml = *seq.add(8 + j) as usize;
        let off = *seq.add(16 + j) as usize;
        let end = written + ll + ml;
        let lend = lp + ll;
        if end > uncompressed_len || lend > n_lit {
            return Err(corrupt("v7: sequence exceeds block"));
        }
        let available = window + written + ll;
        if ml != 0 && (off == 0 || off > available) {
            return Err(CodecError::OffsetOutOfBounds { offset: off, available });
        }
        copy_seq_exact(lits.add(lp), base.add(written), ll, ml, off);
        written = end;
        lp = lend;
        i += 1;
    }
    if written != uncompressed_len || lp != n_lit {
        return Err(corrupt("v7: decoded length mismatch"));
    }
    Ok(written)
}

/// Decode one v7 payload into `dst[..uncompressed_len]`; `n_seq` and
/// `n_lit` come from the block header. Returns the bytes written.
///
/// # Safety
/// `buffer_start` must point into the same allocation as `dst`, at or
/// before `dst.as_ptr()`, with every byte between them initialised: that
/// is the match window (the previous blocks of the same chain).
pub unsafe fn decode_block(payload: &[u8], n_seq: usize, n_lit: usize, dst: &mut [u8], buffer_start: *const u8, uncompressed_len: usize, prev: &mut DecTables, scratch: &mut Scratch) -> Result<usize> {
    if uncompressed_len > dst.len() {
        return Err(CodecError::OutputBufferTooSmall { required: uncompressed_len, provided: dst.len() });
    }
    if uncompressed_len > MAX_BLOCK_SIZE || n_lit > MAX_BLOCK_SIZE || n_seq > MAX_SEQ {
        return Err(corrupt("v7: block header sizes"));
    }
    let layout = payload_layout(payload).ok_or(corrupt("v7: payload layout"))?;
    let sub = &layout.sub;
    let coded = |i: usize| sub.coded & (1 << i) != 0;
    if (sub.reuse & 1 != 0 && !coded(S_LIT)) || (sub.reuse & 2 != 0 && !(coded(S_LL) && coded(S_ML) && coded(S_OFF))) {
        return Err(corrupt("v7: table reuse on a raw stream"));
    }
    let (lit_total, match_total) = sequences(payload, &layout, n_seq, prev, scratch)?;
    if lit_total != n_lit || lit_total + match_total != uncompressed_len {
        return Err(corrupt("v7: sequence totals disagree with header"));
    }
    literals(payload, &layout, n_lit, prev, scratch)?;
    copies(scratch, n_seq, n_lit, dst, buffer_start, uncompressed_len)
}
