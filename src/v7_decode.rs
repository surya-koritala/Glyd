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

pub struct Scratch {
    pub ll: Vec<u32>,
    pub ml: Vec<u32>,
    pub off: Vec<u32>,
    /// ll, ml and off codes, in that order.
    pub codes: Vec<u8>,
    pub codes2: Vec<u8>,
    pub codes3: Vec<u8>,
    /// 64 bytes of margin past MAX_BLOCK_SIZE for the 32-byte wild copies.
    pub lits: Vec<u8>,
}

impl Scratch {
    pub fn new() -> Self {
        Scratch {
            ll: vec![0; MAX_SEQ],
            ml: vec![0; MAX_SEQ],
            off: vec![0; MAX_SEQ],
            codes: vec![0; MAX_SEQ],
            codes2: vec![0; MAX_SEQ],
            codes3: vec![0; MAX_SEQ],
            lits: vec![0; MAX_BLOCK_SIZE + 64],
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

/// Decode one code stream (or copy it raw) into `codes[..n]`.
fn code_stream(sec: &[u8], coded: bool, reuse: bool, n_symbols: usize, n: usize, prev: &mut Option<tans::DecodeTable>, codes: &mut [u8]) -> Result<()> {
    if !coded {
        if sec.len() != n {
            return Err(corrupt("v7: raw code stream length"));
        }
        if sec.iter().any(|&c| c as usize >= n_symbols) {
            return Err(corrupt("v7: raw code out of range"));
        }
        codes[..n].copy_from_slice(sec);
        return Ok(());
    }
    let mut pos = 0usize;
    if !reuse {
        let ns = *sec.first().ok_or(corrupt("v7: table truncated"))? as usize;
        if ns != n_symbols || sec.len() < 1 + 2 * ns {
            return Err(corrupt("v7: table symbol count"));
        }
        let counts: Vec<u16> = (0..ns).map(|s| u16::from_le_bytes([sec[1 + 2 * s], sec[2 + 2 * s]])).collect();
        *prev = Some(tans::DecodeTable::build(&counts).ok_or(corrupt("v7: tANS counts"))?);
        pos = 1 + 2 * ns;
    }
    let table = prev.as_ref().ok_or(corrupt("v7: table reuse without a table"))?;
    let streams = substreams(&sec[pos..])?;
    tans::decode8(table, &streams, n, codes).map_err(|_| corrupt("v7: code stream overrun"))?;
    // A table built from `n_symbols` counts only ever yields those symbols.
    debug_assert!(codes[..n].iter().all(|&c| (c as usize) < n_symbols));
    Ok(())
}

/// Pass 1: the three code streams, then one walk over the extra bits in
/// the encoder's order (ll, ml, off per sequence, sub-stream i % 8).
/// Fills scratch.ll/ml/off; returns (literal total, match total).
fn sequences(payload: &[u8], layout: &Layout, n: usize, prev: &mut DecTables, s: &mut Scratch) -> Result<(usize, usize)> {
    let sub = &layout.sub;
    let coded = |i: usize| sub.coded & (1 << i) != 0;
    let reuse = sub.reuse & 0b10 != 0;
    code_stream(&payload[layout.sections[S_LL].clone()], coded(S_LL), reuse, LL_SYMBOLS, n, &mut prev.ll, &mut s.codes)?;
    code_stream(&payload[layout.sections[S_ML].clone()], coded(S_ML), reuse, ML_SYMBOLS, n, &mut prev.ml, &mut s.codes2)?;
    code_stream(&payload[layout.sections[S_OFF].clone()], coded(S_OFF), reuse, OFF_SYMBOLS, n, &mut prev.off, &mut s.codes3)?;
    if !(coded(S_LL) && coded(S_ML) && coded(S_OFF)) {
        // The encoder drops its tables whenever any stream went raw.
        prev.ll = None;
        prev.ml = None;
        prev.off = None;
    }

    let extra = substreams(&payload[layout.sections[S_EXTRA].clone()])?;
    let mut ers: [BitReader; 8] = std::array::from_fn(|k| BitReader::new(extra[k]));
    let mut reps = Reps::new();
    let (mut lit_total, mut match_total) = (0usize, 0usize);
    for i in 0..n {
        let r = &mut ers[i % 8];
        let llc = s.codes[i];
        let ll = ll_value(llc, r.get(extra_bits_of_code(Kind::Ll, llc) as u32) as u32);
        s.ll[i] = ll;
        lit_total += ll as usize;
        let mlc = s.codes2[i];
        if mlc == 0 && i == n - 1 {
            s.ml[i] = 0;
            s.off[i] = 0;
            continue;
        }
        let ml = ml_value(mlc, r.get(extra_bits_of_code(Kind::Ml, mlc) as u32) as u32);
        let offc = s.codes3[i];
        let off = reps.resolve(offc, r.get(extra_bits_of_code(Kind::Off, offc) as u32) as u32);
        s.ml[i] = ml;
        s.off[i] = off;
        match_total += ml as usize;
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

/// Pass 3: copies. 32-byte wild copies wherever both the literal buffer's
/// margin and `dst.len()` allow, exact copies otherwise (the NEON loop
/// replaces this body in Task 8).
///
/// SAFETY: `buffer_start` must point into the same allocation as `dst`,
/// at or before `dst.as_ptr()`, with every byte between them initialised
/// (the window). `uncompressed_len <= dst.len()` is the caller's check.
unsafe fn copies(s: &Scratch, n: usize, n_lit: usize, dst: &mut [u8], buffer_start: *const u8, uncompressed_len: usize) -> Result<usize> {
    let base = dst.as_mut_ptr();
    let lits = s.lits.as_ptr();
    let mut written = 0usize;
    let mut lp = 0usize;
    for i in 0..n {
        let ll = s.ll[i] as usize;
        let ml = s.ml[i] as usize;
        if ll + ml > uncompressed_len - written || ll > n_lit - lp {
            return Err(corrupt("v7: sequence exceeds block"));
        }
        let d = base.add(written);
        let src = lits.add(lp);
        if lp + ll + 32 <= s.lits.len() && written + ll + 32 <= dst.len() {
            let mut k = 0;
            while k < ll {
                std::ptr::copy_nonoverlapping(src.add(k), d.add(k), 32);
                k += 32;
            }
        } else {
            std::ptr::copy_nonoverlapping(src, d, ll);
        }
        written += ll;
        lp += ll;
        if ml == 0 {
            continue;
        }
        let d = base.add(written);
        let off = s.off[i] as usize;
        let available = d.offset_from(buffer_start) as usize;
        if off == 0 || off > available {
            return Err(CodecError::OffsetOutOfBounds { offset: off, available });
        }
        let src = d.sub(off);
        if off >= 32 && written + ml + 32 <= dst.len() {
            // off >= 32: each 32-byte read ends at or before its write starts.
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
        written += ml;
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
