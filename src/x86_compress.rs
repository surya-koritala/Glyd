#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use crate::finder::{find_matches, init_table, HashTable, Lzav, MatchLen, Mode, Streams};

/// Measure common prefix length using AVX2 vector comparisons.
#[target_feature(enable = "avx2")]
#[target_feature(enable = "bmi2")]
pub unsafe fn common_prefix_len_avx2(mut a: *const u8, mut b: *const u8, max_len: usize) -> usize {
    let mut len = 0;
    while len + 32 <= max_len {
        let v_a = _mm256_loadu_si256(a as *const __m256i);
        let v_b = _mm256_loadu_si256(b as *const __m256i);
        let eq = _mm256_cmpeq_epi8(v_a, v_b);
        let mask = _mm256_movemask_epi8(eq) as u32;
        if mask != 0xFFFF_FFFF {
            let matching = (!mask).trailing_zeros() as usize;
            return len + matching;
        }
        len += 32;
        a = a.add(32);
        b = b.add(32);
    }
    while len + 8 <= max_len {
        let x = std::ptr::read_unaligned(a as *const u64);
        let y = std::ptr::read_unaligned(b as *const u64);
        if x != y {
            return len + ((x ^ y).trailing_zeros() / 8) as usize;
        }
        len += 8;
        a = a.add(8);
        b = b.add(8);
    }
    while len < max_len && *a == *b {
        len += 1;
        a = a.add(1);
        b = b.add(1);
    }
    len
}

pub struct Avx2Match;

impl MatchLen for Avx2Match {
    #[inline(always)]
    unsafe fn prefix(a: *const u8, b: *const u8, max: usize) -> usize {
        common_prefix_len_avx2(a, b, max)
    }
}

/// Chained block compressor: history reaches back through `full_input` to
/// the start of the window, across block boundaries.
#[target_feature(enable = "avx2")]
#[target_feature(enable = "bmi2")]
pub unsafe fn compress_chained_avx2<P: Mode>(
    full_input: &[u8],
    block_start: usize,
    block_len: usize,
    table: &mut HashTable,
    tokens: &mut Vec<u8>,
    offsets: &mut Vec<u8>,
    extras: &mut Vec<u8>,
    literals: &mut Vec<u8>,
) {
    let mut out = Streams { min_match: P::MIN_MATCH, tokens, offsets, extras, literals };
    find_matches::<Avx2Match, P>(full_input, block_start, block_len, table, &mut out);
}

/// Single-block standalone compressor (used when compressing independent blocks in parallel).
#[target_feature(enable = "avx2")]
#[target_feature(enable = "bmi2")]
pub unsafe fn compress_avx2(
    src: &[u8],
    table: &mut HashTable,
    tokens: &mut Vec<u8>,
    offsets: &mut Vec<u8>,
    extras: &mut Vec<u8>,
    literals: &mut Vec<u8>,
) {
    init_table(table, src);
    compress_chained_avx2::<Lzav>(src, 0, src.len(), table, tokens, offsets, extras, literals);
}
