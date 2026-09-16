#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use crate::format::{Token, MAX_LIT_LEN, MAX_MATCH_LEN, MIN_MATCH_LEN, MAX_BLOCK_SIZE};

// 16,384 entries * 4 bytes = 64 KB (fits comfortably in Zen 4 L2 cache)
pub const HASH_BITS: u32 = 16;
pub const HASH_SIZE: usize = 1 << HASH_BITS;

#[inline(always)]
fn hash4(v: u32) -> usize {
    ((v.wrapping_mul(0x9E3779B1)) >> (32 - HASH_BITS)) as usize
}

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

    while len < max_len && *a == *b {
        len += 1;
        a = a.add(1);
        b = b.add(1);
    }
    len
}

/// Chained block compressor: allows matches up to 64 KB into the past across block boundaries.
/// This closes the compression ratio gap with continuous-stream compressors like LZ4.
#[target_feature(enable = "avx2")]
#[target_feature(enable = "bmi2")]
pub unsafe fn compress_chained_avx2(
    full_input: &[u8],
    block_start: usize,
    block_len: usize,
    table: &mut [u32; HASH_SIZE],
    tokens: &mut Vec<Token>,
    offsets: &mut Vec<u16>,
    literals: &mut Vec<u8>,
) {
    if block_len < MIN_MATCH_LEN {
        let chunk = &full_input[block_start..block_start + block_len];
        let mut rem = chunk;
        while !rem.is_empty() {
            let c = rem.len().min(MAX_LIT_LEN);
            tokens.push(Token::new(c, 0));
            literals.extend_from_slice(&rem[..c]);
            rem = &rem[c..];
        }
        return;
    }

    let src_ptr = full_input.as_ptr();
    let block_end = block_start + block_len;
    let limit = block_end - MIN_MATCH_LEN;

    let mut anchor = block_start;
    let mut pos = block_start;

    let mut forward_step = 1usize;
    let mut step_skip = 1usize;

    while pos < limit {
        let val = std::ptr::read_unaligned(src_ptr.add(pos) as *const u32);
        let h = hash4(val);
        let candidate = table[h] as usize;
        table[h] = pos as u32;

        let offset = pos.wrapping_sub(candidate);
        // Match can reach back up to 65,535 bytes (strictly < 65536 to fit in u16)
        if offset > 0 && offset < MAX_BLOCK_SIZE && candidate < pos {
            let candidate_val = std::ptr::read_unaligned(src_ptr.add(candidate) as *const u32);
            if val == candidate_val {
                let max_possible_match = block_end - pos;
                let match_len = common_prefix_len_avx2(
                    src_ptr.add(pos),
                    src_ptr.add(candidate),
                    max_possible_match,
                );

                if match_len >= MIN_MATCH_LEN {
                    // Flush pending literals
                    let mut lit_count = pos - anchor;
                    let mut lit_src = anchor;

                    while lit_count > MAX_LIT_LEN {
                        tokens.push(Token::new(MAX_LIT_LEN, 0));
                        literals.extend_from_slice(std::slice::from_raw_parts(
                            src_ptr.add(lit_src),
                            MAX_LIT_LEN,
                        ));
                        lit_src += MAX_LIT_LEN;
                        lit_count -= MAX_LIT_LEN;
                    }

                    let first_match_chunk = match_len.min(MAX_MATCH_LEN);
                    tokens.push(Token::new(lit_count, first_match_chunk));
                    offsets.push(offset as u16);
                    if lit_count > 0 {
                        literals.extend_from_slice(std::slice::from_raw_parts(
                            src_ptr.add(lit_src),
                            lit_count,
                        ));
                    }

                    let mut rem_match = match_len - first_match_chunk;
                    while rem_match > 0 {
                        let chunk = rem_match.min(MAX_MATCH_LEN);
                        tokens.push(Token::new(0, chunk));
                        offsets.push(offset as u16);
                        rem_match -= chunk;
                    }

                    pos += match_len;
                    anchor = pos;
                    forward_step = 1;
                    step_skip = 1;
                    continue;
                }
            }
        }

        pos += forward_step;
        step_skip += 1;
        forward_step = (step_skip >> 5).max(1);
    }

    // Flush trailing literals
    let mut trailing = block_end - anchor;
    let mut lit_src = anchor;
    while trailing > 0 {
        let chunk = trailing.min(MAX_LIT_LEN);
        tokens.push(Token::new(chunk, 0));
        literals.extend_from_slice(std::slice::from_raw_parts(
            src_ptr.add(lit_src),
            chunk,
        ));
        lit_src += chunk;
        trailing -= chunk;
    }
}

/// Single-block standalone compressor (used when compressing independent blocks in parallel).
#[target_feature(enable = "avx2")]
#[target_feature(enable = "bmi2")]
pub unsafe fn compress_avx2(
    src: &[u8],
    tokens: &mut Vec<Token>,
    offsets: &mut Vec<u16>,
    literals: &mut Vec<u8>,
) {
    let mut table = [0u32; HASH_SIZE];
    compress_chained_avx2(src, 0, src.len(), &mut table, tokens, offsets, literals);
}
