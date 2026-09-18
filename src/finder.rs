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
    Token, ESCAPE_BASE_LIT, ESCAPE_CONT, LIT_CODE_ESCAPE, LIT_DIRECT_MAX, MATCH_CODE_ESCAPE,
    MAX_LIT_LEN, MIN_MATCH_LEN, MIN_MATCH_LEN_DENSE, WINDOW_SIZE,
};

/// Default finder window; see `window_size`. The v6 format holds 17-bit
/// offsets, so this is also the maximum.
pub const FINDER_WINDOW: usize = WINDOW_SIZE;

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

/// How far back the finder looks, in bytes. The format can address 16 MB
/// (`WINDOW_SIZE` caps the override), but the default is FINDER_WINDOW:
/// a small window keeps every match source in L2, which measured the best
/// and by far the most stable decode (57% of liblz4 in the same run at 256 KB,
/// against a noisy 51-55% at 8 MB) at a ratio well above the 2.10 floor.
/// Overridable once per process with `ALATIROK_WINDOW` for sweeps; read once
/// and cached.
#[inline]
fn window_size() -> usize {
    use std::sync::OnceLock;
    static W: OnceLock<usize> = OnceLock::new();
    *W.get_or_init(|| {
        std::env::var("ALATIROK_WINDOW")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .map(|w| w.clamp(1, WINDOW_SIZE))
            .unwrap_or(FINDER_WINDOW)
    })
}

/// komihash-style 6-byte hash, as in LZAV.
#[inline(always)]
fn hash6(iw1: u32, iw2: u32) -> usize {
    let seed1 = 0x243F_6A88u32 ^ iw1;
    let hm = (seed1 as u64).wrapping_mul((0x85A3_08D3u32 ^ (iw2 & 0xFFFF)) as u64);
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
    /// Bytes 4..7 that must be confirmed against the candidate before a match
    /// is trusted, as a mask over the u32 at position + 4. The stored word
    /// proves 4 bytes; the minimum match must not exceed 4 plus these, so it
    /// is sound for minimums 4 through 8. (Trusting 5 with 4 proven, and 8
    /// with 6, each corrupted output during development.)
    const VERIFY_MASK: u32 = match Self::MIN_MATCH {
        0..=4 => 0,
        5 => 0x0000_00FF,
        6 => 0x0000_FFFF,
        7 => 0x00FF_FFFF,
        _ => 0xFFFF_FFFF,
    };
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

/// Write cursors for one block, held by value in the finder's frame so they
/// stay in registers: as Vec lengths (or as fields behind `&mut Streams`,
/// which any raw-pointer store may alias) every token paid serial
/// load/store chains through memory, measured 6.5 ns per token.
#[derive(Clone, Copy)]
pub struct Cursors {
    min_match: usize,
    tok: *mut u8,
    off: *mut u8,
    ext: *mut u8,
    lit: *mut u8,
}

impl<'a> Streams<'a> {
    pub fn new(min_match: usize, tokens: &'a mut Vec<u8>, offsets: &'a mut Vec<u8>, extras: &'a mut Vec<u8>, literals: &'a mut Vec<u8>) -> Self {
        Streams { min_match, tokens, offsets, extras, literals }
    }

    /// Reserve so `Cursors::emit` can write without capacity checks for a
    /// block of `block_len` bytes: at most one token per input byte plus
    /// one, two offset bytes per token, under one extras byte per input
    /// byte (a token with extras spans at least 7 bytes and carries at most
    /// 6), and the literals plus a 32-byte wild-copy margin.
    #[inline(always)]
    pub fn begin_block(&mut self, block_len: usize) -> Cursors {
        self.tokens.reserve(block_len + 64);
        self.offsets.reserve(2 * block_len + 64);
        self.extras.reserve(block_len + 64);
        self.literals.reserve(block_len + 64);
        unsafe {
            Cursors {
                min_match: self.min_match,
                tok: self.tokens.as_mut_ptr().add(self.tokens.len()),
                off: self.offsets.as_mut_ptr().add(self.offsets.len()),
                ext: self.extras.as_mut_ptr().add(self.extras.len()),
                lit: self.literals.as_mut_ptr().add(self.literals.len()),
            }
        }
    }

    /// Publish the cursors as the Vec lengths.
    #[inline(always)]
    pub fn finish(&mut self, c: Cursors) {
        unsafe {
            self.tokens.set_len(c.tok.offset_from(self.tokens.as_ptr()) as usize);
            self.offsets.set_len(c.off.offset_from(self.offsets.as_ptr()) as usize);
            self.extras.set_len(c.ext.offset_from(self.extras.as_ptr()) as usize);
            self.literals.set_len(c.lit.offset_from(self.literals.as_ptr()) as usize);
        }
    }
}

impl Cursors {
    /// Escaped length: one byte, or 255 plus a u16 for the rest.
    #[inline(always)]
    unsafe fn push_escape(&mut self, value: usize, base: usize) {
        let v = value - base;
        if v < 255 {
            *self.ext = v as u8;
            self.ext = self.ext.add(1);
        } else {
            let rest = (v - 255) as u16;
            debug_assert!(v - 255 <= 65535);
            *self.ext = ESCAPE_CONT;
            std::ptr::write_unaligned(self.ext.add(1) as *mut u16, rest.to_le());
            self.ext = self.ext.add(3);
        }
    }

    /// One token: `lc` literals from `lit_src`, then a match of `rc` bytes at
    /// `offset` (rc == 0 means none). Literal runs longer than one escape
    /// are split into leading literal-only tokens. `src_end` bounds the
    /// literal wild copy's over-read.
    ///
    /// # Safety
    /// Must come from `Streams::begin_block` for the block being emitted.
    #[inline(always)]
    pub unsafe fn emit(&mut self, mut lit_src: *const u8, mut lc: usize, rc: usize, offset: usize, src_end: *const u8) {
        while lc > MAX_LIT_LEN {
            self.push_escape(MAX_LIT_LEN, ESCAPE_BASE_LIT);
            *self.tok = Token::from_codes(LIT_CODE_ESCAPE, 0, 0).0;
            self.tok = self.tok.add(1);
            std::ptr::copy_nonoverlapping(lit_src, self.lit, MAX_LIT_LEN);
            self.lit = self.lit.add(MAX_LIT_LEN);
            lit_src = lit_src.add(MAX_LIT_LEN);
            lc -= MAX_LIT_LEN;
        }
        // Escapes without branches: the escape byte is written at the
        // cursor unconditionally and the cursor advances by the condition
        // (19% of tokens escape the literal, 14% the match; as branches
        // they mispredicted). Only the 255 continuation is a branch.
        let bias = self.min_match - 1;
        let nl = (lc > LIT_DIRECT_MAX) as usize;
        let nm = (rc > bias + 14) as usize;
        if (lc >= ESCAPE_BASE_LIT + 255) | (rc >= bias + 15 + 255) {
            if nl != 0 {
                self.push_escape(lc, ESCAPE_BASE_LIT);
            }
            if nm != 0 {
                self.push_escape(rc, bias + 15);
            }
        } else {
            // Both escape bytes in one u16 store: the match byte sits at
            // position nl, the literal byte (if present) at 0.
            let lb = (lc.wrapping_sub(ESCAPE_BASE_LIT) as u8) as u16 & 0u16.wrapping_sub(nl as u16);
            let mb = (rc.wrapping_sub(bias + 15) as u8) as u16;
            std::ptr::write_unaligned(self.ext as *mut u16, (lb | (mb << (8 * nl))).to_le());
            self.ext = self.ext.add(nl + nm);
        }
        let lcode = lc.min(LIT_CODE_ESCAPE);
        let mcode = if rc == 0 { 0 } else { (rc - bias).min(MATCH_CODE_ESCAPE) };
        let hi = if rc > 0 { offset >> 16 } else { 0 };
        *self.tok = Token::from_codes(lcode, mcode, hi).0;
        self.tok = self.tok.add(1);
        // Offset written unconditionally (99.99% of tokens carry one);
        // the cursor advances only when it does.
        std::ptr::write_unaligned(self.off as *mut u16, (offset as u16).to_le());
        self.off = self.off.add(2 * (rc > 0) as usize);
        // Literal wild copy: 32 bytes unconditionally (lc is often 0 and
        // that branch mispredicts), more only for long runs.
        if lit_src.add(lc + 32) <= src_end {
            std::ptr::copy_nonoverlapping(lit_src, self.lit, 16);
            if lc > 16 {
                let mut k = 16;
                while k < lc {
                    std::ptr::copy_nonoverlapping(lit_src.add(k), self.lit.add(k), 32);
                    k += 32;
                }
            }
        } else {
            std::ptr::copy_nonoverlapping(lit_src, self.lit, lc);
        }
        self.lit = self.lit.add(lc);
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
    let src_end = src.add(full_input.len());
    let block_end = block_start + block_len;
    let mut cur = out.begin_block(block_len);
    // Hashing reads 6 bytes; a match must have room for the minimum.
    // Hashing reads 4 bytes at pos and verification 4 more at pos + 4.
    let hash_limit = block_end.saturating_sub(P::MIN_MATCH.max(5) + 3).max(block_start);
    let window = window_size();

    let mut anchor = block_start; // start of pending literals
    let mut pos = block_start;
    // Running average of hash match rate times reference length, *2^15.
    let mut mavg: isize = 100 << 21;
    let mut rndb: usize = 0; // PRNG bit derived from the non-matching offset

    while pos < hash_limit {
        let iw1 = std::ptr::read_unaligned(src.add(pos) as *const u32);
        let iw2 = if P::VERIFY_MASK == 0 { 0 } else { std::ptr::read_unaligned(src.add(pos + 4) as *const u32) & P::VERIFY_MASK };
        let h = hash6(iw1, iw2);
        let bucket = table.get_unchecked_mut(h);
        let hw1 = bucket.w1;

        // Which tuple, if any, holds a 6-byte hit. `word1_hit` says whether
        // tuple 1's word matched, which is what the update policy keys on.
        let word1_hit = iw1 == hw1;
        let mut cand = usize::MAX;
        if word1_hit {
            let c = bucket.p1 as usize;
            if P::VERIFY_MASK == 0 || std::ptr::read_unaligned(src.add(c + 4) as *const u32) & P::VERIFY_MASK == iw2 {
                cand = c;
            } else if iw1 == bucket.w2 {
                let c2 = bucket.p2 as usize;
                if std::ptr::read_unaligned(src.add(c2 + 4) as *const u32) & P::VERIFY_MASK == iw2 {
                    cand = c2;
                }
            }
        } else if iw1 == bucket.w2 {
            let c2 = bucket.p2 as usize;
            if P::VERIFY_MASK == 0 || std::ptr::read_unaligned(src.add(c2 + 4) as *const u32) & P::VERIFY_MASK == iw2 {
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
        if d < P::MIN_OFFSET || d >= window {
            // Too near to be worth a token, or fell out of the window. Out of
            // the window, the position replaces the tuple whose word it hit.
            if d >= window {
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

        cur.emit(src.add(anchor), lc, rc, d, src_end);
        pos = mpos + rc;
        anchor = pos;
        mavg += ((rc << 21) as isize - mavg) >> 10;
    }

    let trailing = block_end - anchor;
    if trailing > 0 {
        cur.emit(src.add(anchor), trailing, 0, 0, src_end);
    }
    out.finish(cur);
}

// ---------------------------------------------------------------------------
// Fast level (GOAL3 S3): an LZ4-class finder. One position per bucket in a
// table small enough to stay in L1, 4-byte multiplicative hash, a single
// u64 compare that both tests the candidate and yields the match length,
// LZ4's skip acceleration on misses, minimum match 7 (the v6 token floor),
// and the same back-match as the default parse since it costs a few
// well-predicted byte compares per token.

/// 32 KB table. Swept on Silesia (5-byte hash, minimum match 5, parse
/// only): 12 bits 0.72 GB/s at 2.00, 13 bits 0.68 at 2.10, 14 bits 0.57
/// at 2.18, 16 bits 0.38 at 2.25. 13 is the liblz4 point on both axes.
pub const FAST_HASH_BITS: u32 = 13;
pub const FAST_HASH_SIZE: usize = 1 << FAST_HASH_BITS;
pub type FastTable = [u32; FAST_HASH_SIZE];

/// Offsets down to 1 are accepted (the default parse refuses < 8 to keep
/// its copies wide): on Silesia that is +0.08 ratio at the same speed.
///
/// 5-byte hash, as liblz4 on 64-bit: every hit is a 5-byte candidate, so
/// the minimum-5 parse wastes fewer compares and mispredicts less.
#[inline(always)]
unsafe fn hash5(p: *const u8) -> usize {
    ((std::ptr::read_unaligned(p as *const u64) << 24).wrapping_mul(889523592379u64) >> (64 - FAST_HASH_BITS)) as usize
}

/// Parse `full_input[block_start..block_start + block_len]` greedily.
/// `table` holds absolute positions into `full_input` (0 = empty, which is
/// harmless: position 0 is either verified or too near).
///
/// # Safety
/// Every position in `table` must be below `block_start + block_len`.
pub unsafe fn find_matches_fast<P: Mode>(
    full_input: &[u8],
    block_start: usize,
    block_len: usize,
    table: &mut FastTable,
    out: &mut Streams,
) {
    const { assert!(P::MIN_MATCH <= 8) };
    let min = P::MIN_MATCH;
    const SKIP_STRENGTH: u32 = 6;
    let src = full_input.as_ptr();
    let src_end = src.add(full_input.len());
    let block_end = block_start + block_len;
    let mut cur = out.begin_block(block_len);
    // The u64 compare reads 8 bytes at pos and at the candidate.
    let limit = block_end.saturating_sub(8).max(block_start);
    let window = window_size();

    let mut anchor = block_start;
    let mut pos = block_start;
    let mut search_nb: u32 = 1 << SKIP_STRENGTH;

    'outer: loop {
        // Find the next match.
        let (cand, mut rc);
        loop {
            if pos >= limit {
                break 'outer;
            }
            let h = hash5(src.add(pos));
            let c = *table.get_unchecked(h) as usize;
            *table.get_unchecked_mut(h) = pos as u32;
            let step = (search_nb >> SKIP_STRENGTH) as usize;
            search_nb += 1;
            let d = pos.wrapping_sub(c);
            if d >= 1 && d < window {
                let x = std::ptr::read_unaligned(src.add(pos) as *const u64)
                    ^ std::ptr::read_unaligned(src.add(c) as *const u64);
                let len = if x == 0 { 8 } else { (x.trailing_zeros() / 8) as usize };
                if len >= min {
                    cand = c;
                    rc = len;
                    break;
                }
            }
            pos += step;
        }
        search_nb = 1 << SKIP_STRENGTH;

        let d = pos - cand;
        let mut ml = block_end - pos;
        if ml > MAX_REF_LEN {
            ml = MAX_REF_LEN;
        }
        if rc == 8 && ml > 8 {
            rc = 8 + ScalarMatch::prefix(src.add(pos + 8), src.add(cand + 8), ml - 8);
        }
        rc = rc.min(ml);

        let mut lc = pos - anchor;
        let mut mpos = pos;
        if lc != 0 {
            let mut room = lc.min(cand).min(BACK_MATCH_MAX);
            let mut bmc = 0usize;
            while room > 0 && *src.add(mpos - 1 - bmc) == *src.add(cand - 1 - bmc) {
                bmc += 1;
                room -= 1;
            }
            rc += bmc;
            mpos -= bmc;
            lc -= bmc;
        }

        cur.emit(src.add(anchor), lc, rc, d, src_end);
        pos = mpos + rc;
        anchor = pos;
        // Index the last position of the match so a run continues to hash.
        if pos >= 2 && pos < limit {
            *table.get_unchecked_mut(hash5(src.add(pos - 2))) = (pos - 2) as u32;
        }
    }

    let trailing = block_end - anchor;
    if trailing > 0 {
        cur.emit(src.add(anchor), trailing, 0, 0, src_end);
    }
    out.finish(cur);
}
