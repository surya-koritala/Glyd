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
use crate::bits::{BitReader, Stream, PAD};
use crate::error::{CodecError, Result};
use crate::format::MAX_BLOCK_SIZE;
use crate::huff8;
use crate::tans;
use crate::v7_encode::{payload_layout, payload_layout_compact, Layout};
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
#[derive(Clone)]
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

/// One section of a payload as the passes take it: its own bytes are
/// `bytes[..len]`; in a compact (v9) block `bytes` runs on to the end of
/// the payload, whose padding the streams' loads may reach.
#[derive(Clone, Copy)]
struct Section<'a> {
    bytes: &'a [u8],
    len: usize,
}

impl<'a> Section<'a> {
    fn of(payload: &'a [u8], layout: &Layout, i: usize) -> Section<'a> {
        let r = layout.sections[i].clone();
        if layout.compact {
            Section { bytes: &payload[r.start..], len: r.end - r.start }
        } else {
            Section { bytes: &payload[r.clone()], len: r.end - r.start }
        }
    }

    /// The section's own bytes.
    fn own(&self) -> &'a [u8] {
        &self.bytes[..self.len]
    }

    /// The section from `pos` on (past a table).
    fn from(&self, pos: usize) -> Section<'a> {
        Section { bytes: &self.bytes[pos..], len: self.len - pos }
    }
}

/// A section's 8 sub-streams. v9: `bits::Stream::split_compact`. v8:
/// `bits::Stream::split` (24-bit sizes, one padding at the end). v7: 8
/// u32 sizes, each stream carrying its own `PAD` zeros.
fn substreams(sec: Section, v8: bool, compact: bool) -> Result<([Stream; 8], bool)> {
    if compact {
        return Stream::split_compact(sec.bytes, sec.len).ok_or(corrupt("v9: sub-stream table"));
    }
    let sec = sec.own();
    if v8 {
        return Stream::split(sec).map(|s| (s, false)).ok_or(corrupt("v8: sub-stream table"));
    }
    if sec.len() < 32 {
        return Err(corrupt("v7: sub-stream table truncated"));
    }
    let mut pos = 32usize;
    let mut out = [Stream { bytes: &sec[sec.len() - PAD.min(sec.len())..], len: 0 }; 8];
    for k in 0..8 {
        let n = u32::from_le_bytes(sec[k * 4..k * 4 + 4].try_into().unwrap()) as usize;
        if n < PAD || n > sec.len() - pos {
            return Err(corrupt("v7: sub-stream out of section"));
        }
        out[k] = Stream { bytes: &sec[pos..pos + n], len: n - PAD };
        pos += n;
    }
    if pos != sec.len() {
        return Err(corrupt("v7: section has trailing bytes"));
    }
    Ok((out, false))
}

/// A v8 tANS table from the front of `bytes`: its counts and the bytes
/// it took (dictionaries carry tables the same way blocks do).
pub(crate) fn read_tans_table(bytes: &[u8], n_symbols: usize) -> Option<(Vec<u16>, usize)> {
    let mut counts = [0u16; tans::MAX_SYMBOLS];
    let used = tans_counts(bytes, true, n_symbols, &mut counts).ok()?;
    Some((counts[..n_symbols].to_vec(), used))
}

/// A tANS table's counts as the block carries them: `ns`, then in v8
/// each count as a 4-bit width and the bits below its top one
/// (`v7_encode::write_tans_table`); in v7, `ns` u16 LE counts. Returns
/// the bytes consumed.
fn tans_counts(sec: &[u8], v8: bool, n_symbols: usize, counts: &mut [u16; tans::MAX_SYMBOLS]) -> Result<usize> {
    let ns = *sec.first().ok_or(corrupt("v7: table truncated"))? as usize;
    if ns != n_symbols {
        return Err(corrupt("v7: table symbol count"));
    }
    if !v8 {
        if sec.len() < 1 + 2 * ns {
            return Err(corrupt("v7: table truncated"));
        }
        for (s, c) in counts[..ns].iter_mut().enumerate() {
            *c = u16::from_le_bytes([sec[1 + 2 * s], sec[2 + 2 * s]]);
        }
        return Ok(1 + 2 * ns);
    }
    let mut bit = 8usize;
    let get = |bit: usize, n: usize| -> Result<u32> {
        // n <= 11 bits from position `bit`: the three bytes that can hold them.
        let byte = bit >> 3;
        if (bit + n).div_ceil(8) > sec.len() {
            return Err(corrupt("v8: table truncated"));
        }
        let b = |k: usize| *sec.get(byte + k).unwrap_or(&0) as u32;
        Ok(((b(0) | b(1) << 8 | b(2) << 16) >> (bit & 7)) & ((1 << n) - 1))
    };
    for c in counts[..ns].iter_mut() {
        let w = get(bit, 4)? as usize;
        bit += 4;
        if w > 11 {
            return Err(corrupt("v8: table count width"));
        }
        *c = if w >= 2 {
            let m = get(bit, w - 1)?;
            bit += w - 1;
            ((1u32 << (w - 1)) + m) as u16
        } else {
            w as u16
        };
    }
    Ok(bit.div_ceil(8))
}

/// Decode one code stream (or copy it raw) into its lane of the grouped
/// `codes` (`codes` starts at the lane: code i goes to `at(i)`).
#[cfg_attr(target_arch = "x86_64", inline(always))]
fn code_stream(sec: Section, v8: bool, compact: bool, coded: bool, reuse: bool, n_symbols: usize, n: usize, prev: &mut Option<tans::DecodeTable>, codes: &mut [u8]) -> Result<()> {
    if !coded {
        let sec = sec.own();
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
        let mut counts = [0u16; tans::MAX_SYMBOLS];
        pos = tans_counts(sec.own(), v8, n_symbols, &mut counts)?;
        // Rebuilt in place: the table lives in `DecTables` across blocks.
        if !prev.get_or_insert_with(tans::DecodeTable::empty).rebuild(&counts[..n_symbols]) {
            *prev = None;
            return Err(corrupt("v7: tANS counts"));
        }
    }
    let table = prev.as_ref().ok_or(corrupt("v7: table reuse without a table"))?;
    let (streams, single) = substreams(sec.from(pos), v8, compact)?;
    tans::decode8_rows_with(table, &streams, n, GROUP, single, codes).map_err(|_| corrupt("v7: code stream overrun"))?;
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

/// The x86-64 field: width and base from the split tables (a byte and a
/// dword load that fold into the arithmetic), the extra bits by `bzhi`.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn field2<const K: usize>(w: u64, t: &WalkSplit, c: u8) -> (u32, u64, u32) {
    let n = t.nb[K][c as usize] as u32;
    ((w & ((1u64 << n) - 1)) as u32 + t.base[K][c as usize], w >> n, n)
}

/// Bytes a sequence can advance a stream: its three fields are at most
/// 19 + 19 + 22 = 60 bits (codes 31, 31 and 25), so the load address moves
/// by at most 8 bytes per sequence whatever the code bytes hold.
const SEQ_BYTES: usize = 8;
const _: () = assert!(
    SEQ_BYTES * 8
        >= (extra_bits_of_code(Kind::Ll, (LL_SYMBOLS - 1) as u8)
            + extra_bits_of_code(Kind::Ml, (ML_SYMBOLS - 1) as u8)
            + extra_bits_of_code(Kind::Off, (OFF_SYMBOLS - 1) as u8)) as usize
);

/// Sequences the fast walk can take on one stream before a load might
/// start past `last`: the next load is at `at`, each later one <= SEQ_BYTES
/// further. Mirrors `tans::safe_batches`.
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
/// Fills scratch.seq; returns (literal total, match total).
///
/// The walk has the `huff8::decode` shape -- an unclamped batch loop with
/// a proven load bound, then the clamped `BitReader`s for the tail -- but
/// its hot state is one bit position per stream, not a reader with an
/// accumulator and a count: a valid sequence's extra bits are at most
/// 18 + 17 + 22 = 57 (a literal run of MAX_BLOCK_SIZE = 2^18, a match
/// below it, an offset below 2^23; v7 blocks stop at 2^21), and one
/// unaligned 8-byte load shifted by the sub-byte position holds at least
/// 57, so each sequence is one load, three field extractions and one add
/// to its stream's position, with no accumulator, count or refill to
/// maintain (8 live registers for the 8 streams instead of 24). A
/// corrupt stream asking for more reads zeros for the last bits and
/// advances exactly anyway; the values are garbage either way and the
/// checks after the walk catch them. Each iteration handles 8 sequences, one per stream;
/// the batch count is bounded by every stream's `safe_seqs` and stops
/// short of sequence n - 1 (which may be literal-only and always goes
/// through the tail). The tail's clamped readers start at the positions
/// the walk reached (`BitReader::new_at`), which keeps `overrun` exact.
#[cfg_attr(target_arch = "x86_64", inline(always))]
fn sequences<const V8: bool>(payload: &[u8], layout: &Layout, n: usize, prev: &mut DecTables, s: &mut Scratch) -> Result<(usize, usize)> {
    let sub = &layout.sub;
    let coded = |i: usize| sub.coded & (1 << i) != 0;
    let reuse = sub.reuse & 0b10 != 0;
    // The version is a const parameter so the walk's tables are constant
    // addresses, not a register (the x86-64 walk has none to spare).
    let v8 = V8;
    let compact = layout.compact;
    let (ll_symbols, ml_symbols) = if v8 { (LL_SYMBOLS, ML_SYMBOLS) } else { (LL_SYMBOLS_V7, ML_SYMBOLS_V7) };
    code_stream(Section::of(payload, layout, S_LL), v8, compact, coded(S_LL), reuse, ll_symbols, n, &mut prev.ll, &mut s.codes)?;
    code_stream(Section::of(payload, layout, S_ML), v8, compact, coded(S_ML), reuse, ml_symbols, n, &mut prev.ml, &mut s.codes[8..])?;
    let off_symbols = if v8 { OFF_SYMBOLS } else { OFF_SYMBOLS_V7 };
    code_stream(Section::of(payload, layout, S_OFF), v8, compact, coded(S_OFF), reuse, off_symbols, n, &mut prev.off, &mut s.codes[16..])?;
    if !(coded(S_LL) && coded(S_ML) && coded(S_OFF)) {
        // The encoder drops its tables whenever any stream went raw.
        prev.ll = None;
        prev.ml = None;
        prev.off = None;
    }

    let section = Section::of(payload, layout, S_EXTRA);
    let sec = section.bytes;
    let (extra, single) = substreams(section, v8, compact)?;
    // Stream k's next bit is `b<k>`, an absolute bit address (`ptr * 8 +
    // bit`, as in `tans::decode8`): eight scalars, not an array, so they
    // stay in registers, and no base register. Built from the extras
    // *section*'s pointer plus each sub-stream's offset, not each
    // sub-stream's own slice pointer, so the address's provenance covers
    // every sub-stream. `lasts[k]` is the last address a load on stream k
    // may start at.
    let sec_addr = sec.as_ptr() as usize;
    // SAFETY: every `extra[k]` is a sub-slice of `sec`.
    let offs: [usize; 8] = std::array::from_fn(|k| unsafe { extra[k].bytes.as_ptr().offset_from(sec.as_ptr()) as usize });
    let start = |k: usize| ((sec_addr + offs[k]) * 8) as u64;
    let (mut b0, mut b1, mut b2, mut b3) = (start(0), start(1), start(2), start(3));
    let (mut b4, mut b5, mut b6, mut b7) = (start(4), start(5), start(6), start(7));
    let lasts: [usize; 8] = std::array::from_fn(|k| extra[k].bytes.as_ptr() as usize + extra[k].bytes.len() - PAD);
    let mut reps = Reps::new();
    let mut o = 0usize;
    #[cfg(target_arch = "x86_64")]
    let mut bp: [u64; 8] = [b0, b1, b2, b3, b4, b5, b6, b7];
    // The length-code tables of the block's version.
    #[cfg(not(target_arch = "x86_64"))]
    let (llw, mlw): (&[u64; 256], &[u64; 256]) = if v8 { (&LL_WALK, &ML_WALK) } else { (&LL_WALK_V7, &ML_WALK_V7) };
    #[cfg(target_arch = "x86_64")]
    let split: &WalkSplit = if v8 { &WALK_SPLIT } else { &WALK_SPLIT_V7 };
    loop {
        // A single-stream section (compact block) has every sequence in
        // stream 0: the clamped tail below takes them all.
        if single {
            break;
        }
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
                    // x86-64: the position lives in memory between its
                    // uses (a load and a store per sequence, both cheap)
                    // rather than competing for one of 16 registers.
                    #[cfg(target_arch = "x86_64")]
                    let mut $b: u64 = unsafe { std::ptr::read_volatile(&bp[$k]) };
                    let w = unsafe { std::ptr::read_unaligned(($b >> 3) as *const u64) } >> ($b & 7);
                    let offc = c[16 + $k];
                    #[cfg(not(target_arch = "x86_64"))]
                    let (ll, w, n1) = field(w, llw[c[$k] as usize]);
                    #[cfg(not(target_arch = "x86_64"))]
                    let (ml, w, n2) = field(w, mlw[c[8 + $k] as usize]);
                    #[cfg(not(target_arch = "x86_64"))]
                    let (ov, _, n3) = field(w, OFF_WALK[offc as usize]);
                    #[cfg(target_arch = "x86_64")]
                    let (ll, w, n1) = field2::<0>(w, split, c[$k]);
                    #[cfg(target_arch = "x86_64")]
                    let (ml, w, n2) = field2::<1>(w, split, c[8 + $k]);
                    #[cfg(target_arch = "x86_64")]
                    let (ov, _, n3) = field2::<2>(w, split, offc);
                    $b += n1 as u64;
                    $b += n2 as u64;
                    $b += n3 as u64;
                    #[cfg(target_arch = "x86_64")]
                    unsafe {
                        std::ptr::write_volatile(&mut bp[$k], $b);
                    }
                    out[$k] = ll;
                    out[8 + $k] = ml;
                    out[16 + $k] = reps.update(offc, ov);
                }};
            }
            #[cfg(target_arch = "x86_64")]
            let _ = (&mut b0, &mut b1, &mut b2, &mut b3, &mut b4, &mut b5, &mut b6, &mut b7);
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
        #[cfg(target_arch = "x86_64")]
        {
            [b0, b1, b2, b3, b4, b5, b6, b7] = bp;
        }
    }
    let rd = |k: usize, b: u64| BitReader::new_at(extra[k], (b - start(k)) as usize);
    let mut ers: [BitReader; 8] = [rd(0, b0), rd(1, b1), rd(2, b2), rd(3, b3), rd(4, b4), rd(5, b5), rd(6, b6), rd(7, b7)];

    // Tail: the last sequence, whatever a short stream's margin left, and
    // the literal-only rule -- on the clamped readers.
    for i in o..n {
        let r = &mut ers[if single { 0 } else { i % 8 }];
        let j = at(i);
        let llc = s.codes[j];
        let mlc = s.codes[8 + j];
        if v8 {
            s.seq[j] = ll_value(llc, r.get(extra_bits_of_code(Kind::Ll, llc) as u32) as u32);
        } else {
            s.seq[j] = len_value_v7(llc, r.get(extra_bits_of_code_v7(Kind::Ll, llc) as u32) as u32);
        }
        if mlc == 0 && i == n - 1 {
            s.seq[8 + j] = 0;
            s.seq[16 + j] = 0;
            continue;
        }
        let ml = if v8 {
            ml_value(mlc, r.get(extra_bits_of_code(Kind::Ml, mlc) as u32) as u32)
        } else {
            len_value_v7(mlc, r.get(extra_bits_of_code_v7(Kind::Ml, mlc) as u32) as u32) + MIN_MATCH
        };
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
#[cfg_attr(target_arch = "x86_64", inline(always))]
fn literals(payload: &[u8], layout: &Layout, n_lit: usize, prev: &mut DecTables, s: &mut Scratch) -> Result<()> {
    let section = Section::of(payload, layout, S_LIT);
    let sec = section.own();
    if layout.sub.coded & (1 << S_LIT) == 0 {
        if sec.len() != n_lit {
            return Err(corrupt("v7: raw literal length"));
        }
        s.lits[..n_lit].copy_from_slice(sec);
        return Ok(());
    }
    let mut pos = 0usize;
    if layout.sub.reuse & 1 == 0 {
        let lengths = if layout.v8 {
            let (lengths, n) = crate::huffman::unpack_lengths_v8(sec).ok_or(corrupt("v8: literal table truncated"))?;
            pos = n;
            lengths
        } else {
            if sec.len() < huff8::TABLE_BYTES {
                return Err(corrupt("v7: literal table truncated"));
            }
            pos = huff8::TABLE_BYTES;
            crate::huffman::unpack_lengths(&sec[..huff8::TABLE_BYTES])
        };
        prev.lit = Some(huff8::Table::build(&lengths).ok_or(corrupt("v7: literal code lengths"))?);
    }
    let table = prev.lit.as_ref().ok_or(corrupt("v7: literal table reuse without a table"))?;
    let (streams, single) = substreams(section.from(pos), layout.v8, layout.compact)?;
    huff8::decode_with(table, &streams, n_lit, single, &mut s.lits).map_err(|_| corrupt("v7: literal stream overrun"))
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

/// Same contract and shape on the other targets, with `copy_nonoverlapping`
/// standing in for the vector copy: LLVM emits one 32-byte move where AVX
/// is enabled (`decode_block_avx2`) and two 16-byte moves elsewhere.
#[cfg(not(target_arch = "aarch64"))]
#[inline(always)]
unsafe fn copy_seq_wild(lit: *const u8, d: *mut u8, ll: usize, ml: usize, off: usize) {
    #[inline(always)]
    unsafe fn copy32(s: *const u8, d: *mut u8) {
        std::ptr::copy_nonoverlapping(s, d, 32);
    }
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

/// Overlapping match (`off < 32`) in 16- or 8-byte steps, each reading
/// only bytes written at least a step earlier; below 8 the period is laid
/// down once by hand and then copied at a stride that is a multiple of it.
#[cfg(not(target_arch = "aarch64"))]
#[inline(always)]
unsafe fn short_match(s: *const u8, d: *mut u8, off: usize, ml: usize) {
    let end = d.add(ml);
    if off >= 16 {
        let (mut s, mut d) = (s, d);
        while d < end {
            std::ptr::copy_nonoverlapping(s, d, 16);
            s = s.add(16);
            d = d.add(16);
        }
    } else if off >= 8 {
        let (mut s, mut d) = (s, d);
        while d < end {
            std::ptr::copy_nonoverlapping(s, d, 8);
            s = s.add(8);
            d = d.add(8);
        }
    } else {
        for k in 0..8 {
            *d.add(k) = *s.add(k);
        }
        let stride = off * ((8 + off - 1) / off);
        let mut dd = d.add(8);
        let mut ss = dd.sub(stride);
        while dd < end {
            std::ptr::copy_nonoverlapping(ss, dd, 8);
            ss = ss.add(8);
            dd = dd.add(8);
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
#[cfg_attr(target_arch = "x86_64", inline(always))]
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
/// Wild copies (see `copies`) may write up to 95 bytes past
/// `uncompressed_len`, but never past `dst.len()` (the 96-byte
/// `WILD_MARGIN` rule).
///
/// # Safety
/// `buffer_start` must point into the same allocation as `dst`, at or
/// before `dst.as_ptr()`, with every byte between them initialised: that
/// is the match window (the previous blocks of the same chain).
pub unsafe fn decode_block(payload: &[u8], v8: bool, compact: bool, n_seq: usize, n_lit: usize, dst: &mut [u8], buffer_start: *const u8, uncompressed_len: usize, prev: &mut DecTables, scratch: &mut Scratch) -> Result<usize> {
    #[cfg(target_arch = "x86_64")]
    {
        if crate::has_avx2() {
            return decode_block_avx2(payload, v8, compact, n_seq, n_lit, dst, buffer_start, uncompressed_len, prev, scratch);
        }
    }
    decode_block_impl(payload, v8, compact, n_seq, n_lit, dst, buffer_start, uncompressed_len, prev, scratch)
}

/// The decoder compiled for AVX2 + BMI2: the same passes, with 32-byte
/// copies and single-uop variable shifts (`shrx`) in the bit loops.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,bmi2")]
unsafe fn decode_block_avx2(payload: &[u8], v8: bool, compact: bool, n_seq: usize, n_lit: usize, dst: &mut [u8], buffer_start: *const u8, uncompressed_len: usize, prev: &mut DecTables, scratch: &mut Scratch) -> Result<usize> {
    decode_block_impl(payload, v8, compact, n_seq, n_lit, dst, buffer_start, uncompressed_len, prev, scratch)
}

#[cfg_attr(target_arch = "x86_64", inline(always))]
unsafe fn decode_block_impl(payload: &[u8], v8: bool, compact: bool, n_seq: usize, n_lit: usize, dst: &mut [u8], buffer_start: *const u8, uncompressed_len: usize, prev: &mut DecTables, scratch: &mut Scratch) -> Result<usize> {
    if uncompressed_len > dst.len() {
        return Err(CodecError::OutputBufferTooSmall { required: uncompressed_len, provided: dst.len() });
    }
    if uncompressed_len > MAX_BLOCK_SIZE || n_lit > MAX_BLOCK_SIZE || n_seq > MAX_SEQ {
        return Err(corrupt("v7: block header sizes"));
    }
    let mut layout = if compact { payload_layout_compact(payload) } else { payload_layout(payload) }.ok_or(corrupt("v7: payload layout"))?;
    layout.v8 = v8;
    let sub = &layout.sub;
    let coded = |i: usize| sub.coded & (1 << i) != 0;
    if (sub.reuse & 1 != 0 && !coded(S_LIT)) || (sub.reuse & 2 != 0 && !(coded(S_LL) && coded(S_ML) && coded(S_OFF))) {
        return Err(corrupt("v7: table reuse on a raw stream"));
    }
    let (lit_total, match_total) = if v8 { sequences::<true>(payload, &layout, n_seq, prev, scratch)? } else { sequences::<false>(payload, &layout, n_seq, prev, scratch)? };
    if lit_total != n_lit || lit_total + match_total != uncompressed_len {
        return Err(corrupt("v7: sequence totals disagree with header"));
    }
    literals(payload, &layout, n_lit, prev, scratch)?;
    copies(scratch, n_seq, n_lit, dst, buffer_start, uncompressed_len)
}
