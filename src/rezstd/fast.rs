//! The level-1 match finder: zstd 1.5.5's
//! `ZSTD_compressBlock_fast_noDict_generic` (`zstd_fast.c`) step for
//! step, with its pipelined pairs of positions, the step that grows
//! every 128 bytes without a match, the repcode checked one step
//! ahead, the backward extension, the two table entries filled after
//! a match and the repcode chained right after it. Positions are the
//! reference's indexes: the input's first byte is index 2
//! (`ZSTD_WINDOW_START_INDEX`), so an empty table slot (0) never
//! passes as a candidate.

use super::CParams;

/// `ZSTD_WINDOW_START_INDEX`.
pub const START_INDEX: u32 = 2;
/// `HASH_READ_SIZE`: the hash reads eight bytes.
const HASH_READ_SIZE: usize = 8;
/// `kStepIncr`: the step grows after this many bytes without a match.
const STEP_INCR: usize = 1 << (8 - 1);
/// `MINMATCH`: what the stored match length is counted from.
pub const MINMATCH: usize = 3;

/// One sequence as `seqDef` stores it: the offset code (repcode 1-3,
/// or offset + 3), the literal count and the match length less three,
/// both truncated to 16 bits with the one long length noted aside.
#[derive(Clone, Copy, Debug)]
pub struct Seq {
    pub off_base: u32,
    pub lit_len: u16,
    pub ml_base: u16,
}

/// Which of a block's lengths, if any, did not fit 16 bits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LongLength {
    None,
    Literals,
    Match,
}

/// `seqStore_t`: a block's literals and sequences.
pub struct SeqStore {
    pub lits: Vec<u8>,
    pub seqs: Vec<Seq>,
    pub long_type: LongLength,
    pub long_pos: u32,
}

impl SeqStore {
    fn new() -> SeqStore {
        SeqStore { lits: Vec::new(), seqs: Vec::new(), long_type: LongLength::None, long_pos: 0 }
    }

    /// `ZSTD_storeSeq`.
    fn store(&mut self, literals: &[u8], off_base: u32, match_len: usize) {
        self.lits.extend_from_slice(literals);
        if literals.len() > 0xFFFF {
            self.long_type = LongLength::Literals;
            self.long_pos = self.seqs.len() as u32;
        }
        let ml_base = match_len - MINMATCH;
        if ml_base > 0xFFFF {
            self.long_type = LongLength::Match;
            self.long_pos = self.seqs.len() as u32;
        }
        self.seqs.push(Seq { off_base, lit_len: literals.len() as u16, ml_base: ml_base as u16 });
    }
}

#[inline]
fn read32(s: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(s[at..at + 4].try_into().unwrap())
}

#[inline]
fn read64(s: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(s[at..at + 8].try_into().unwrap())
}

/// `ZSTD_hashPtr`: the hash of `mls` bytes at `at` into `hlog` bits.
#[inline]
fn hash(s: &[u8], at: usize, hlog: u32, mls: u32) -> usize {
    match mls {
        5 => ((read64(s, at) << 24).wrapping_mul(889523592379u64) >> (64 - hlog)) as usize,
        6 => ((read64(s, at) << 16).wrapping_mul(227718039650203u64) >> (64 - hlog)) as usize,
        7 => ((read64(s, at) << 8).wrapping_mul(58295818150454627u64) >> (64 - hlog)) as usize,
        8 => (read64(s, at).wrapping_mul(0xCF1BBCDCB7A56463u64) >> (64 - hlog)) as usize,
        _ => (read32(s, at).wrapping_mul(2654435761u32) >> (32 - hlog)) as usize,
    }
}

/// `ZSTD_count`: bytes equal from `a` and `b` on, `a` staying below `end`.
#[inline]
fn count(s: &[u8], a: usize, b: usize, end: usize) -> usize {
    let mut n = 0;
    while a + n < end && s[a + n] == s[b + n] {
        n += 1;
    }
    n
}

/// The window's state carried across blocks (`ZSTD_window_t`): the
/// index below which no match may reach.
pub struct Window {
    pub dict_limit: u32,
    pub low_limit: u32,
}

impl Window {
    pub fn new() -> Window {
        Window { dict_limit: START_INDEX, low_limit: START_INDEX }
    }

    /// `ZSTD_window_enforceMaxDist` before a block ending at `block_end`.
    pub fn enforce_max_dist(&mut self, block_end: usize, window_log: u32) {
        let block_end_idx = block_end as u32 + START_INDEX;
        let max_dist = 1u32 << window_log;
        if block_end_idx > max_dist {
            let new_low = block_end_idx - max_dist;
            if self.low_limit < new_low {
                self.low_limit = new_low;
            }
            if self.dict_limit < self.low_limit {
                self.dict_limit = self.low_limit;
            }
        }
    }

    /// `ZSTD_getLowestPrefixIndex`.
    fn lowest_prefix(&self, curr: u32, window_log: u32) -> u32 {
        let max_distance = 1u32 << window_log;
        if curr - self.dict_limit > max_distance {
            curr - max_distance
        } else {
            self.dict_limit
        }
    }
}

/// The block `input[start..end]` parsed into `SeqStore`, the table and
/// the repcodes updated for the next block.
pub fn compress_block(input: &[u8], start: usize, end: usize, table: &mut [u32], p: &CParams, window: &Window, rep: &mut [u32; 3]) -> SeqStore {
    let mut store = SeqStore::new();
    let hlog = p.hash_log;
    let mls = p.min_match;
    let step_size: usize = if p.target_length > 1 { p.target_length as usize + 1 } else { 2 };
    let idx = |pos: usize| pos as u32 + START_INDEX;
    let pos = |i: u32| (i - START_INDEX) as usize;
    let end_index = idx(end);
    let prefix_start_index = window.lowest_prefix(end_index, p.window_log);
    let prefix_start = pos(prefix_start_index);
    let ilimit = end as isize - HASH_READ_SIZE as isize;
    let mut anchor = start;
    let mut ip0 = start;
    let (mut rep1, mut rep2) = (rep[0], rep[1]);
    let (mut saved1, mut saved2) = (0u32, 0u32);
    if ip0 == prefix_start {
        ip0 += 1;
    }
    {
        let curr = idx(ip0);
        let window_low = window.lowest_prefix(curr, p.window_log);
        let max_rep = curr - window_low;
        if rep2 > max_rep {
            saved2 = rep2;
            rep2 = 0;
        }
        if rep1 > max_rep {
            saved1 = rep1;
            rep1 = 0;
        }
    }
    // A found match: its start, its start in the past, its offset code
    // and the length already known.
    struct Found {
        ip0: usize,
        match0: usize,
        off_base: u32,
        len: usize,
    }
    'start: loop {
        let mut step = step_size;
        let mut next_step = ip0 + STEP_INCR;
        let mut ip1 = ip0 + 1;
        let mut ip2 = ip0 + step;
        let mut ip3 = ip2 + 1;
        let mut current0: u32 = 0;
        let found: Option<Found> = 'search: {
            if ip3 as isize >= ilimit {
                break 'search None;
            }
            let mut hash0 = hash(input, ip0, hlog, mls);
            let mut hash1 = hash(input, ip1, hlog, mls);
            let mut cand = table[hash0];
            loop {
                // The repcode is checked at ip2 while ip0 is looked up.
                let rval = read32(input, ip2 - rep1 as usize);
                current0 = idx(ip0);
                table[hash0] = current0;
                if read32(input, ip2) == rval && rep1 > 0 {
                    let mut ip0 = ip2;
                    let mut match0 = ip0 - rep1 as usize;
                    let back = (input[ip0 - 1] == input[match0 - 1]) as usize;
                    ip0 -= back;
                    match0 -= back;
                    table[hash1] = idx(ip1);
                    break 'search Some(Found { ip0, match0, off_base: 1, len: back + 4 });
                }
                let mval = if cand >= prefix_start_index { read32(input, pos(cand)) } else { read32(input, ip0) ^ 1 };
                if read32(input, ip0) == mval {
                    table[hash1] = idx(ip1);
                    break 'search Some(Found { ip0, match0: pos(cand), off_base: 0, len: 0 });
                }
                cand = table[hash1];
                hash0 = hash1;
                hash1 = hash(input, ip2, hlog, mls);
                ip0 = ip1;
                ip1 = ip2;
                ip2 = ip3;
                current0 = idx(ip0);
                table[hash0] = current0;
                let mval = if cand >= prefix_start_index { read32(input, pos(cand)) } else { read32(input, ip0) ^ 1 };
                if read32(input, ip0) == mval {
                    // The entry for ip1 is written only when it stays
                    // below where the search resumes after the match.
                    if step <= 4 {
                        table[hash1] = idx(ip1);
                    }
                    break 'search Some(Found { ip0, match0: pos(cand), off_base: 0, len: 0 });
                }
                cand = table[hash1];
                hash0 = hash1;
                hash1 = hash(input, ip2, hlog, mls);
                ip0 = ip1;
                ip1 = ip2;
                ip2 = ip0 + step;
                ip3 = ip1 + step;
                if ip2 >= next_step {
                    step += 1;
                    next_step += STEP_INCR;
                }
                if ip3 as isize >= ilimit {
                    break 'search None;
                }
            }
        };
        let Some(mut f) = found else {
            // The repcodes invalidated at the start come back if they
            // were not replaced.
            saved2 = if saved1 != 0 && rep1 != 0 { saved1 } else { saved2 };
            rep[0] = if rep1 != 0 { rep1 } else { saved1 };
            rep[1] = if rep2 != 0 { rep2 } else { saved2 };
            break 'start;
        };
        if f.off_base == 0 {
            // A table match: its offset becomes repcode 1, then the
            // match is extended backwards.
            rep2 = rep1;
            rep1 = (f.ip0 - f.match0) as u32;
            f.off_base = rep1 + 3;
            f.len = 4;
            while f.ip0 > anchor && f.match0 > prefix_start && input[f.ip0 - 1] == input[f.match0 - 1] {
                f.ip0 -= 1;
                f.match0 -= 1;
                f.len += 1;
            }
        }
        f.len += count(input, f.ip0 + f.len, f.match0 + f.len, end);
        store.store(&input[anchor..f.ip0], f.off_base, f.len);
        ip0 = f.ip0 + f.len;
        anchor = ip0;
        if ip0 as isize <= ilimit {
            // Fill the table, then take repcode 2 matches right here.
            let c2 = pos(current0) + 2;
            table[hash(input, c2, hlog, mls)] = current0 + 2;
            table[hash(input, ip0 - 2, hlog, mls)] = idx(ip0 - 2);
            if rep2 > 0 {
                while ip0 as isize <= ilimit && read32(input, ip0) == read32(input, ip0 - rep2 as usize) {
                    let rlen = count(input, ip0 + 4, ip0 + 4 - rep2 as usize, end) + 4;
                    std::mem::swap(&mut rep1, &mut rep2);
                    table[hash(input, ip0, hlog, mls)] = idx(ip0);
                    ip0 += rlen;
                    store.store(&[], 1, rlen);
                    anchor = ip0;
                }
            }
        }
    }
    // ZSTD_storeLastLiterals.
    store.lits.extend_from_slice(&input[anchor..end]);
    store
}
