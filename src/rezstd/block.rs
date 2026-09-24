//! A compressed block's two sections as zstd 1.5.5 writes them: the
//! literals (`zstd_compress_literals.c`: raw, RLE, or Huffman with the
//! previous block's table when that costs less) and the sequences
//! (`zstd_compress_sequences.c` and `ZSTD_entropyCompressSeqStore` in
//! `zstd_compress.c`: the length and offset codes, each table chosen
//! among the predefined one, RLE and a fresh FSE table by the level-1
//! rules, then the interleaved bitstream). The entropy tables of one
//! block are the next block's `prev`.

use super::fast::{LongLength, SeqStore};
use super::fse::{self, highbit, BitWriter, CState, CTable, Repeat};
use super::huf::{self, HufTable};

pub const MAX_LL: usize = 35;
pub const MAX_ML: usize = 52;
pub const MAX_OFF: usize = 31;
const DEFAULT_MAX_OFF: usize = 28;
const LL_FSE_LOG: u32 = 9;
const ML_FSE_LOG: u32 = 9;
const OFF_FSE_LOG: u32 = 8;
const LONG_NB_SEQ: usize = 0x7F00;

pub static LL_BITS: [u8; MAX_LL + 1] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
];
pub static LL_DEFAULT_NORM: [i16; MAX_LL + 1] = [
    4, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 1, 1, 1, 1, 1, -1, -1, -1, -1,
];
pub const LL_DEFAULT_NORM_LOG: u32 = 6;
pub static ML_BITS: [u8; MAX_ML + 1] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
];
pub static ML_DEFAULT_NORM: [i16; MAX_ML + 1] = [
    1, 4, 3, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1,
];
pub const ML_DEFAULT_NORM_LOG: u32 = 6;
pub static OF_DEFAULT_NORM: [i16; DEFAULT_MAX_OFF + 1] = [
    1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1,
];
pub const OF_DEFAULT_NORM_LOG: u32 = 5;

/// `ZSTD_LLcode`.
pub fn ll_code(lit_len: u32) -> u8 {
    static LL_CODE: [u8; 64] = [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 16, 17, 17, 18, 18, 19, 19, 20, 20, 20, 20, 21, 21, 21, 21, 22, 22, 22, 22, 22, 22, 22, 22, 23, 23, 23, 23, 23, 23,
        23, 23, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24,
    ];
    if lit_len > 63 {
        (highbit(lit_len) + 19) as u8
    } else {
        LL_CODE[lit_len as usize]
    }
}

/// `ZSTD_MLcode` of a match length less three.
pub fn ml_code(ml_base: u32) -> u8 {
    static ML_CODE: [u8; 128] = [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 32, 33, 33, 34, 34, 35, 35, 36, 36, 36, 36, 37, 37,
        37, 37, 38, 38, 38, 38, 38, 38, 38, 38, 39, 39, 39, 39, 39, 39, 39, 39, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 41, 41, 41, 41, 41, 41, 41, 41, 41,
        41, 41, 41, 41, 41, 41, 41, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    ];
    if ml_base > 127 {
        (highbit(ml_base) + 36) as u8
    } else {
        ML_CODE[ml_base as usize]
    }
}

/// `symbolEncodingType_e`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Coding {
    Basic = 0,
    Rle = 1,
    Compressed = 2,
    Repeat = 3,
}

/// `ZSTD_hufCTables_t`.
#[derive(Clone, Copy)]
pub struct HufState {
    pub table: HufTable,
    pub repeat: Repeat,
}

/// `ZSTD_fseCTables_t`.
#[derive(Clone, Copy)]
pub struct FseState {
    pub ll: CTable,
    pub of: CTable,
    pub ml: CTable,
    pub ll_repeat: Repeat,
    pub of_repeat: Repeat,
    pub ml_repeat: Repeat,
}

/// `ZSTD_entropyCTables_t`: what a block leaves for the next one.
#[derive(Clone, Copy)]
pub struct Entropy {
    pub huf: HufState,
    pub fse: FseState,
}

impl Entropy {
    /// `ZSTD_reset_compressedBlockState`: nothing to repeat.
    pub fn fresh() -> Entropy {
        let blank = CTable::rle(0);
        Entropy {
            huf: HufState { table: HufTable::blank(), repeat: Repeat::None },
            fse: FseState { ll: blank, of: blank, ml: blank, ll_repeat: Repeat::None, of_repeat: Repeat::None, ml_repeat: Repeat::None },
        }
    }
}

/// `dstSize_tooSmall`: the only error the reference can raise on this
/// path with a `ZSTD_compressBound` buffer; the block is then kept raw.
pub struct TooSmall;

/// `ZSTD_minGain` for the fast strategy.
pub fn min_gain(src_size: usize) -> usize {
    (src_size >> 6) + 2
}

/// `ZSTD_noCompressLiterals`.
fn raw_literals(out: &mut Vec<u8>, cap: usize, lits: &[u8]) -> Result<(), TooSmall> {
    let n = lits.len();
    let fl_size = 1 + (n > 31) as usize + (n > 4095) as usize;
    if n + fl_size > cap {
        return Err(TooSmall);
    }
    match fl_size {
        1 => out.push((n << 3) as u8),
        2 => out.extend_from_slice(&(((1 << 2) + (n << 4)) as u16).to_le_bytes()),
        _ => out.extend_from_slice(&(((3 << 2) + (n << 4)) as u32).to_le_bytes()[..3]),
    }
    out.extend_from_slice(lits);
    Ok(())
}

/// `ZSTD_compressRleLiteralsBlock`.
fn rle_literals(out: &mut Vec<u8>, lits: &[u8]) {
    let n = lits.len();
    let fl_size = 1 + (n > 31) as usize + (n > 4095) as usize;
    match fl_size {
        1 => out.push((1 + (n << 3)) as u8),
        2 => out.extend_from_slice(&((1 + (1 << 2) + (n << 4)) as u16).to_le_bytes()),
        _ => out.extend_from_slice(&((1 + (3 << 2) + (n << 4)) as u32).to_le_bytes()[..3]),
    }
    out.push(lits[0]);
}

/// `ZSTD_compressLiterals` for the fast strategy: the literals section
/// appended to `out`, `next` set from `prev` and the block.
fn compress_literals(out: &mut Vec<u8>, cap: usize, lits: &[u8], prev: &HufState, next: &mut HufState, suspect_uncompressible: bool) -> Result<(), TooSmall> {
    let n = lits.len();
    let lh_size = 3 + (n >= 1024) as usize + (n >= 16384) as usize;
    let mut single_stream = n < 256;
    *next = *prev;
    // ZSTD_minLiteralsToCompress: 6 with a valid table, else 8 << 3.
    let min_literals = if prev.repeat == Repeat::Valid { 6 } else { 64 };
    if n < min_literals {
        return raw_literals(out, cap, lits);
    }
    if cap < lh_size + 1 {
        return Err(TooSmall);
    }
    let mut repeat = prev.repeat;
    let prefer_repeat = n <= 1024;
    if repeat == Repeat::Valid && lh_size == 3 {
        single_stream = true;
    }
    let coded = huf::compress(cap - lh_size, lits, !single_stream, &mut next.table, &mut repeat, prefer_repeat, suspect_uncompressible);
    let h_type = if repeat != Repeat::None { Coding::Repeat } else { Coding::Compressed };
    let coded = match coded {
        Ok(c) if !c.is_empty() && c.len() < n - min_gain(n) => c,
        _ => {
            *next = *prev;
            return raw_literals(out, cap, lits);
        }
    };
    if coded.len() == 1 && (n >= 8 || lits.iter().all(|&b| b == lits[0])) {
        *next = *prev;
        rle_literals(out, lits);
        return Ok(());
    }
    if h_type == Coding::Compressed {
        next.repeat = Repeat::Check;
    }
    let (n, c) = (n as u32, coded.len() as u32);
    let h = h_type as u32;
    match lh_size {
        3 => out.extend_from_slice(&(h + ((!single_stream as u32) << 2) + (n << 4) + (c << 14)).to_le_bytes()[..3]),
        4 => out.extend_from_slice(&(h + (2 << 2) + (n << 4) + (c << 18)).to_le_bytes()),
        _ => {
            out.extend_from_slice(&(h + (3 << 2) + (n << 4) + (c << 22)).to_le_bytes());
            out.push((c >> 10) as u8);
        }
    }
    out.extend_from_slice(&coded);
    Ok(())
}

/// `ZSTD_selectEncodingType` for a strategy below `ZSTD_lazy`.
fn select_coding(repeat: &mut Repeat, most_frequent: usize, nb_seq: usize, default_norm_log: u32, default_allowed: bool) -> Coding {
    if most_frequent == nb_seq {
        *repeat = Repeat::None;
        return if default_allowed && nb_seq <= 2 { Coding::Basic } else { Coding::Rle };
    }
    if default_allowed {
        let static_fse_nb_seq_max = 1000;
        let mult = 10 - 1; // 10 - strategy, the fast strategy being 1
        let dynamic_fse_nb_seq_min = ((1usize << default_norm_log) * mult) >> 3;
        if *repeat == Repeat::Valid && nb_seq < static_fse_nb_seq_max {
            return Coding::Repeat;
        }
        if nb_seq < dynamic_fse_nb_seq_min || most_frequent < (nb_seq >> (default_norm_log - 1)) {
            *repeat = Repeat::None;
            return Coding::Basic;
        }
    }
    *repeat = Repeat::Check;
    Coding::Compressed
}

/// `ZSTD_buildCTable`: `next` built (or copied from `prev`) for
/// `coding`; the table description, if any, appended to `out`.
#[allow(clippy::too_many_arguments)]
fn build_ctable(out: &mut Vec<u8>, cap: usize, next: &mut CTable, fse_log: u32, coding: Coding, count: &mut [u32], max: usize, codes: &[u8], default_norm: &[i16], default_norm_log: u32, default_max: usize, prev: &CTable) -> Result<(), TooSmall> {
    match coding {
        Coding::Rle => {
            *next = CTable::rle(max as u8);
            if cap == 0 {
                return Err(TooSmall);
            }
            out.push(codes[0]);
        }
        Coding::Repeat => *next = *prev,
        Coding::Basic => *next = CTable::build(default_norm, default_max, default_norm_log),
        Coding::Compressed => {
            let nb_seq = codes.len();
            let mut nb_seq_1 = nb_seq;
            let table_log = fse::optimal_table_log(fse_log, nb_seq, max as u32, 2);
            // The last symbol is coded by the state alone.
            if count[codes[nb_seq - 1] as usize] > 1 {
                count[codes[nb_seq - 1] as usize] -= 1;
                nb_seq_1 -= 1;
            }
            let mut norm = [0i16; fse::MAX_SYMBOLS];
            fse::normalize_count(&mut norm, table_log, count, nb_seq_1, max, nb_seq_1 >= 2048).expect("a table log above the minimum");
            let ncount = fse::write_ncount(cap, &norm, max, table_log).ok_or(TooSmall)?;
            out.extend_from_slice(&ncount);
            *next = CTable::build(&norm, max, table_log);
        }
    }
    Ok(())
}

/// `ZSTD_encodeSequences` on a 64-bit build (no long offsets).
#[allow(clippy::too_many_arguments)]
fn encode_sequences(cap: usize, store: &SeqStore, ll: &CTable, of: &CTable, ml: &CTable, ll_codes: &[u8], of_codes: &[u8], ml_codes: &[u8]) -> Result<Vec<u8>, TooSmall> {
    let mut w = BitWriter::new(cap).ok_or(TooSmall)?;
    let seqs = &store.seqs;
    let n = seqs.len();
    let last = n - 1;
    let mut state_ml = CState::new(ml, ml_codes[last]);
    let mut state_of = CState::new(of, of_codes[last]);
    let mut state_ll = CState::new(ll, ll_codes[last]);
    w.add(seqs[last].lit_len as u64, LL_BITS[ll_codes[last] as usize] as u32);
    w.add(seqs[last].ml_base as u64, ML_BITS[ml_codes[last] as usize] as u32);
    w.add(seqs[last].off_base as u64, of_codes[last] as u32);
    for i in (0..last).rev() {
        state_of.encode(&mut w, of_codes[i]);
        state_ml.encode(&mut w, ml_codes[i]);
        state_ll.encode(&mut w, ll_codes[i]);
        w.add(seqs[i].lit_len as u64, LL_BITS[ll_codes[i] as usize] as u32);
        w.add(seqs[i].ml_base as u64, ML_BITS[ml_codes[i] as usize] as u32);
        w.add(seqs[i].off_base as u64, of_codes[i] as u32);
    }
    state_ml.flush(&mut w);
    state_of.flush(&mut w);
    state_ll.flush(&mut w);
    w.close().ok_or(TooSmall)
}

/// `ZSTD_entropyCompressSeqStore_internal`: the block's sections in
/// `cap` bytes; Ok(empty) where the reference returns 0.
fn compress_internal(store: &SeqStore, prev: &Entropy, next: &mut Entropy, cap: usize) -> Result<Vec<u8>, TooSmall> {
    let mut out = Vec::new();
    let nb_seq = store.seqs.len();
    let nb_lits = store.lits.len();
    let suspect_uncompressible = nb_seq == 0 || nb_lits / nb_seq >= 20;
    compress_literals(&mut out, cap, &store.lits, &prev.huf, &mut next.huf, suspect_uncompressible)?;
    if cap - out.len() < 3 + 1 {
        return Err(TooSmall);
    }
    if nb_seq < 128 {
        out.push(nb_seq as u8);
    } else if nb_seq < LONG_NB_SEQ {
        out.push((nb_seq >> 8) as u8 + 0x80);
        out.push(nb_seq as u8);
    } else {
        out.push(0xFF);
        out.extend_from_slice(&((nb_seq - LONG_NB_SEQ) as u16).to_le_bytes());
    }
    if nb_seq == 0 {
        next.fse = prev.fse;
        return Ok(out);
    }
    let seq_head = out.len();
    out.push(0);
    // ZSTD_seqToCodes.
    let mut ll_codes: Vec<u8> = store.seqs.iter().map(|s| ll_code(s.lit_len as u32)).collect();
    let of_codes: Vec<u8> = store.seqs.iter().map(|s| highbit(s.off_base) as u8).collect();
    let mut ml_codes: Vec<u8> = store.seqs.iter().map(|s| ml_code(s.ml_base as u32)).collect();
    match store.long_type {
        LongLength::Literals => ll_codes[store.long_pos as usize] = MAX_LL as u8,
        LongLength::Match => ml_codes[store.long_pos as usize] = MAX_ML as u8,
        LongLength::None => {}
    }
    // ZSTD_buildSequencesStatistics: a table per code stream.
    let mut count = [0u32; fse::MAX_SYMBOLS];
    let mut last_count_size = 0usize;
    let (max, most_frequent) = fse::histogram(&mut count, MAX_LL, &ll_codes);
    next.fse.ll_repeat = prev.fse.ll_repeat;
    let ll_type = select_coding(&mut next.fse.ll_repeat, most_frequent as usize, nb_seq, LL_DEFAULT_NORM_LOG, true);
    let at = out.len();
    build_ctable(&mut out, cap - at, &mut next.fse.ll, LL_FSE_LOG, ll_type, &mut count, max, &ll_codes, &LL_DEFAULT_NORM, LL_DEFAULT_NORM_LOG, MAX_LL, &prev.fse.ll)?;
    if ll_type == Coding::Compressed {
        last_count_size = out.len() - at;
    }
    let (max, most_frequent) = fse::histogram(&mut count, MAX_OFF, &of_codes);
    let default_allowed = max <= DEFAULT_MAX_OFF;
    next.fse.of_repeat = prev.fse.of_repeat;
    let of_type = select_coding(&mut next.fse.of_repeat, most_frequent as usize, nb_seq, OF_DEFAULT_NORM_LOG, default_allowed);
    let at = out.len();
    build_ctable(&mut out, cap - at, &mut next.fse.of, OFF_FSE_LOG, of_type, &mut count, max, &of_codes, &OF_DEFAULT_NORM, OF_DEFAULT_NORM_LOG, DEFAULT_MAX_OFF, &prev.fse.of)?;
    if of_type == Coding::Compressed {
        last_count_size = out.len() - at;
    }
    let (max, most_frequent) = fse::histogram(&mut count, MAX_ML, &ml_codes);
    next.fse.ml_repeat = prev.fse.ml_repeat;
    let ml_type = select_coding(&mut next.fse.ml_repeat, most_frequent as usize, nb_seq, ML_DEFAULT_NORM_LOG, true);
    let at = out.len();
    build_ctable(&mut out, cap - at, &mut next.fse.ml, ML_FSE_LOG, ml_type, &mut count, max, &ml_codes, &ML_DEFAULT_NORM, ML_DEFAULT_NORM_LOG, MAX_ML, &prev.fse.ml)?;
    if ml_type == Coding::Compressed {
        last_count_size = out.len() - at;
    }
    out[seq_head] = ((ll_type as u8) << 6) + ((of_type as u8) << 4) + ((ml_type as u8) << 2);
    let bitstream = encode_sequences(cap - out.len(), store, &next.fse.ll, &next.fse.of, &next.fse.ml, &ll_codes, &of_codes, &ml_codes)?;
    // zstd up to 1.3.4 misread a block whose last table description
    // and bitstream make less than four bytes: such a block is kept raw.
    if last_count_size != 0 && last_count_size + bitstream.len() < 4 {
        return Ok(Vec::new());
    }
    out.extend_from_slice(&bitstream);
    Ok(out)
}

/// `ZSTD_entropyCompressSeqStore`: the compressed block body, or None
/// when the block is not compressible enough (kept raw).
pub fn compress(store: &SeqStore, prev: &Entropy, next: &mut Entropy, cap: usize, src_size: usize) -> Option<Vec<u8>> {
    let out = match compress_internal(store, prev, next, cap) {
        Ok(out) => out,
        Err(TooSmall) => {
            assert!(src_size <= cap, "a ZSTD_compressBound buffer always holds a raw block");
            return None;
        }
    };
    if out.is_empty() || out.len() >= src_size - min_gain(src_size) {
        return None;
    }
    Some(out)
}
