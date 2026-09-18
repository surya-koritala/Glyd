//! Match finder, adapted from LZAV 4.3 by Aleksey Vaneev (MIT).
//!
//! Measured on Silesia against our previous 4-byte-hash, 1-way, lazy finder:
//! LZAV's parse emits 23% fewer tokens (average match 12.47 bytes against
//! 9.60) for 10.7% more literal bytes, a net 12.3 MB less output, and 41.6%
//! of its matches lie beyond a 64 KB window. The pieces that make that work
//! together, none of which is enough alone:
//!
//! - A 6-byte hash (4-byte word plus 2 more bytes) so every hit is at least
//!   the 6-byte minimum reference.
//! - Two (word, position) tuples per bucket, 16 bytes, in a 1 MB table. A
//!   failed probe is a register compare; the source is read only to confirm
//!   the 2 extra bytes of a 4-byte hit.
//! - An 8 MB window. This was refuted for a 1-way table, which thrashes; it
//!   pays once buckets keep two candidates.
//! - Back-matching: a hit is extended backwards into the pending literals.
//! - A running match-rate average that widens the step on incompressible
//!   data instead of hashing every byte.
//! - No lazy matching, and a match is never updated into the table when its
//!   offset is 273 or less, which keeps same-byte runs compressing.
//!
//! The finder is generic over the prefix-compare primitive so the AVX2 build
//! and the portable build run the same parse.

use crate::format::{
    encode_lit, encode_match, offset_width, push_offset, Token, MAX_LIT_LEN, MIN_MATCH_LEN,
    MIN_MATCH_LEN_DENSE, WINDOW_SIZE,
};

pub const HASH_BITS: u32 = 16;
pub const HASH_SIZE: usize = 1 << HASH_BITS;

/// LZAV's longest reference: 6 + 15 + 255 + 254. Kept so the parse matches
/// the one measured; the format itself allows longer matches.
pub const MAX_REF_LEN: usize = 530;
/// Matches at offsets below this are skipped; they are cheap to encode but
/// slow to copy, and the run they usually belong to is found at the next
/// position with a larger offset.
pub const MIN_OFFSET: usize = 8;
/// A match found at an offset up to this is not written into the table, so a
/// same-byte run keeps referencing its start and match lengths grow.
pub const NO_UPDATE_OFFSET: usize = 273;
/// Longest backward extension into pending literals.
pub const BACK_MATCH_MAX: usize = 16;

/// Hash bucket: two (match word, position) tuples. `w1`/`p1` is the newer.
#[derive(Copy, Clone, Default)]
#[repr(C, align(16))]
pub struct Bucket {
    pub w1: u32,
    pub p1: u32,
    pub w2: u32,
    pub p2: u32,
}

pub type HashTable = [Bucket; HASH_SIZE];

/// A hash table on the heap, filled by `init_table` before any probe. Built
/// through a Vec so the 1 MB never lands on the stack.
pub fn new_table() -> Box<HashTable> {
    let v: Vec<Bucket> = vec![Bucket::default(); HASH_SIZE];
    let b: Box<[Bucket]> = v.into_boxed_slice();
    // SAFETY: the slice has exactly HASH_SIZE elements, so the layouts match.
    unsafe { Box::from_raw(Box::into_raw(b) as *mut HashTable) }
}

/// Fill the table so every entry is truthful: the word stored really is the
/// word at the stored position. LZAV relies on the stored word to prove a
/// 4-byte match without reading the source, so a zeroed table would claim
/// the source starts with four zero bytes.
pub fn init_table(table: &mut HashTable, src: &[u8]) {
    let w = if src.len() >= 4 {
        u32::from_le_bytes([src[0], src[1], src[2], src[3]])
    } else {
        0
    };
    table.fill(Bucket { w1: w, p1: 0, w2: w, p2: 0 });
}

/// komihash-style 6-byte hash, as in LZAV.
#[inline(always)]
fn hash6(iw1: u32, iw2: u16) -> usize {
    let seed1 = 0x243F_6A88u32 ^ iw1;
    let hm = (seed1 as u64).wrapping_mul((0x85A3_08D3u32 ^ iw2 as u32) as u64);
    let hval = (hm as u32) ^ ((hm >> 32) as u32);
    ((hval >> 4) as usize) & (HASH_SIZE - 1)
}

/// Common-prefix length of the byte sequences at `a` and `b`, at most `max`.
pub trait MatchLen {
    unsafe fn prefix(a: *const u8, b: *const u8, max: usize) -> usize;
}

pub struct ScalarMatch;

impl MatchLen for ScalarMatch {
    #[inline(always)]
    unsafe fn prefix(mut a: *const u8, mut b: *const u8, max: usize) -> usize {
        let mut n = 0;
        while n + 8 <= max {
            let x = std::ptr::read_unaligned(a as *const u64);
            let y = std::ptr::read_unaligned(b as *const u64);
            if x != y {
                return n + ((x ^ y).trailing_zeros() / 8) as usize;
            }
            n += 8;
            a = a.add(8);
            b = b.add(8);
        }
        while n < max && *a == *b {
            n += 1;
            a = a.add(1);
            b = b.add(1);
        }
        n
    }
}

/// Parse policy. `Lzav` is the measured LZAV parse. `Dense` is the retry for
/// blocks it cannot compress: a 4-byte hash, 4-byte minimum match, every
/// position hashed, any offset, overlapping copies allowed. Slower per byte,
/// but it only runs on blocks the first pass would otherwise store raw.
pub trait Mode {
    const MIN_MATCH: usize;
    /// Bytes 4 and 5 that must be confirmed against the candidate before a
    /// match is trusted, as a mask over the u16 at position + 4. The stored
    /// word proves 4 bytes; the minimum match must not exceed 4 plus these.
    const VERIFY_MASK: u16 = if Self::MIN_MATCH >= 6 { 0xFFFF } else if Self::MIN_MATCH == 5 { 0x00FF } else { 0 };
    /// Widen the step on poor data.
    const SKIP: bool;
    /// Cap match length at the offset so no copy overlaps itself.
    const CAP_BY_OFFSET: bool;
    const MIN_OFFSET: usize;
}

pub struct Lzav;
impl Mode for Lzav {
    const MIN_MATCH: usize = MIN_MATCH_LEN;
    const SKIP: bool = true;
    const CAP_BY_OFFSET: bool = true;
    const MIN_OFFSET: usize = MIN_OFFSET;
}

pub struct Dense;
impl Mode for Dense {
    const MIN_MATCH: usize = MIN_MATCH_LEN_DENSE;
    const SKIP: bool = true;
    const CAP_BY_OFFSET: bool = true;
    const MIN_OFFSET: usize = MIN_OFFSET;
}

/// Output streams of one block.
pub struct Streams<'a> {
    /// Minimum match of the parse that fills these streams; sets the length bias.
    pub min_match: usize,
    pub tokens: &'a mut Vec<u8>,
    pub offsets: &'a mut Vec<u8>,
    pub extras: &'a mut Vec<u8>,
    pub literals: &'a mut Vec<u8>,
}

impl<'a> Streams<'a> {
    /// One token: `lc` literals from `lit_src`, then a match of `rc` bytes at
    /// `offset` (rc == 0 means none). Literal runs longer than one escape
    /// are split into leading literal-only tokens.
    #[inline(always)]
    pub unsafe fn emit(&mut self, mut lit_src: *const u8, mut lc: usize, rc: usize, offset: usize) {
        while lc > MAX_LIT_LEN {
            let code = encode_lit(MAX_LIT_LEN, self.extras);
            self.tokens.push(Token::from_codes(code, 0, 1).0);
            self.literals
                .extend_from_slice(std::slice::from_raw_parts(lit_src, MAX_LIT_LEN));
            lit_src = lit_src.add(MAX_LIT_LEN);
            lc -= MAX_LIT_LEN;
        }
        let lcode = encode_lit(lc, self.extras);
        let mcode = encode_match(rc, self.extras, self.min_match);
        let width = if rc > 0 { offset_width(offset) } else { 1 };
        self.tokens.push(Token::from_codes(lcode, mcode, width).0);
        if rc > 0 {
            push_offset(self.offsets, offset, width);
        }
        if lc > 0 {
            self.literals
                .extend_from_slice(std::slice::from_raw_parts(lit_src, lc));
        }
    }
}

/// Parse `full_input[block_start..block_start + block_len]` with history
/// back to `full_input[0]`. Positions in `table` are absolute in
/// `full_input`, so the table can be carried across consecutive blocks.
///
/// # Safety
/// `table` must have been filled by `init_table` on the same `full_input`
/// (or a prefix of it) and every position it holds must be below
/// `block_start + block_len`.
#[inline(always)]
pub unsafe fn find_matches<M: MatchLen, P: Mode>(
    full_input: &[u8],
    block_start: usize,
    block_len: usize,
    table: &mut HashTable,
    out: &mut Streams,
) {
    let src = full_input.as_ptr();
    let block_end = block_start + block_len;
    // Hashing reads 6 bytes; a match must have room for the minimum.
    let hash_limit = block_end.saturating_sub(P::MIN_MATCH + 3).max(block_start);

    let mut anchor = block_start; // start of pending literals
    let mut pos = block_start;
    // Running average of hash match rate times reference length, *2^15.
    let mut mavg: isize = 100 << 21;
    let mut rndb: usize = 0; // PRNG bit derived from the non-matching offset

    while pos < hash_limit {
        let iw1 = std::ptr::read_unaligned(src.add(pos) as *const u32);
        let iw2 = std::ptr::read_unaligned(src.add(pos + 4) as *const u16) & P::VERIFY_MASK;
        let h = hash6(iw1, iw2);
        let bucket = table.get_unchecked_mut(h);
        let hw1 = bucket.w1;

        // Which tuple, if any, holds a 6-byte hit. `word1_hit` says whether
        // tuple 1's word matched, which is what the update policy keys on.
        let word1_hit = iw1 == hw1;
        let mut cand = usize::MAX;
        if word1_hit {
            let c = bucket.p1 as usize;
            if P::VERIFY_MASK == 0 || std::ptr::read_unaligned(src.add(c + 4) as *const u16) & P::VERIFY_MASK == iw2 {
                cand = c;
            } else if iw1 == bucket.w2 {
                let c2 = bucket.p2 as usize;
                if std::ptr::read_unaligned(src.add(c2 + 4) as *const u16) & P::VERIFY_MASK == iw2 {
                    cand = c2;
                }
            }
        } else if iw1 == bucket.w2 {
            let c2 = bucket.p2 as usize;
            if P::VERIFY_MASK == 0 || std::ptr::read_unaligned(src.add(c2 + 4) as *const u16) & P::VERIFY_MASK == iw2 {
                cand = c2;
            }
        }

        if cand == usize::MAX {
            // No match: the new word takes tuple 2.
            bucket.w2 = iw1;
            bucket.p2 = pos as u32;
            mavg -= mavg >> 11;
            let ipo = pos;
            if P::SKIP && mavg < (200 << 14) && pos != anchor {
                // Speed-up on poor data: keep hash evaluations near 45% of
                // positions, dithered so runs do not alias.
                pos += 1 + rndb;
                rndb = ipo & 1;
                if mavg < (130 << 14) {
                    pos += 1;
                    if mavg < (100 << 14) {
                        pos += (100 - (mavg >> 14)) as usize;
                    }
                }
            }
            pos += 1;
            continue;
        }

        let d = pos.wrapping_sub(cand);
        if d < P::MIN_OFFSET || d >= WINDOW_SIZE {
            // Too near to be worth a token, or fell out of the window. Out of
            // the window, the position replaces the tuple whose word it hit.
            if d >= WINDOW_SIZE {
                if word1_hit {
                    bucket.p1 = pos as u32;
                } else {
                    bucket.p2 = pos as u32;
                }
            }
            pos += 1;
            continue;
        }

        // Match length is capped at the offset so no copy overlaps itself,
        // and at the block end.
        let mut ml = if P::CAP_BY_OFFSET { d.min(MAX_REF_LEN) } else { MAX_REF_LEN };
        if pos + ml > block_end {
            ml = block_end - pos;
        }

        if d > NO_UPDATE_OFFSET {
            // Update the matching entry. Inside a run the older entry is
            // kept so offsets keep growing.
            if word1_hit {
                bucket.p1 = pos as u32;
            } else {
                bucket.w2 = hw1;
                bucket.p2 = bucket.p1;
                bucket.w1 = iw1;
                bucket.p1 = pos as u32;
            }
        }

        let mut rc = P::MIN_MATCH
            + M::prefix(
                src.add(pos + P::MIN_MATCH),
                src.add(cand + P::MIN_MATCH),
                ml - P::MIN_MATCH,
            );
        let mut lc = pos - anchor;
        let mut mpos = pos;
        if lc != 0 {
            // Back-match: consume pending literals that also match.
            let mut room = (ml - rc).min(lc).min(cand).min(BACK_MATCH_MAX);
            let mut bmc = 0usize;
            while room > 0 && *src.add(mpos - 1 - bmc) == *src.add(cand - 1 - bmc) {
                bmc += 1;
                room -= 1;
            }
            if bmc != 0 {
                rc += bmc;
                mpos -= bmc;
                lc -= bmc;
            }
        }

        out.emit(src.add(anchor), lc, rc, d);
        pos = mpos + rc;
        anchor = pos;
        mavg += ((rc << 21) as isize - mavg) >> 10;
    }

    let trailing = block_end - anchor;
    if trailing > 0 {
        out.emit(src.add(anchor), trailing, 0, 0);
    }
}
