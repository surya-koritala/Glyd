#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use crate::format::{encode_lit, encode_match, Token, MAX_LIT_LEN, MAX_MATCH_LEN, MIN_MATCH_LEN, WINDOW_SIZE};

// 16,384 entries * 4 bytes = 64 KB (fits comfortably in Zen 4 L2 cache)
pub const HASH_BITS: u32 = 16;
/// Only probe pos+1 when the match found at pos is shorter than this.
pub const LAZY_MATCH_THRESHOLD: usize = 32;
/// How far ahead to prefetch hash buckets. The table is larger than L1, so
/// every probe would otherwise stall on L2 latency.
pub const PREFETCH_DIST: usize = 8;
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
    extras: &mut Vec<u16>,
    literals: &mut Vec<u8>,
) {
    if block_len < MIN_MATCH_LEN {
        let chunk = &full_input[block_start..block_start + block_len];
        if !chunk.is_empty() {
            emit_literal_run(chunk, tokens, extras, literals);
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

        // Warm the bucket this loop will need a few positions from now.
        if pos + PREFETCH_DIST < limit {
            let fval = std::ptr::read_unaligned(src_ptr.add(pos + PREFETCH_DIST) as *const u32);
            let fh = hash4(fval);
            _mm_prefetch(table.as_ptr().add(fh) as *const i8, _MM_HINT_T0);
        }

        let candidate = table[h] as usize;
        table[h] = pos as u32;

        let offset = pos.wrapping_sub(candidate);
        // Match can reach back up to 65,535 bytes (strictly < 65536 to fit in u16)
        if offset > 0 && offset < WINDOW_SIZE && candidate < pos {
            let candidate_val = std::ptr::read_unaligned(src_ptr.add(candidate) as *const u32);
            if val == candidate_val {
                let max_possible_match = block_end - pos;
                let mut match_len = common_prefix_len_avx2(
                    src_ptr.add(pos),
                    src_ptr.add(candidate),
                    max_possible_match,
                );

                if match_len >= MIN_MATCH_LEN {
                    let mut match_offset = offset;
                    // Lazy matching: probe pos + 1 only when the first match is
                    // short. A long match is already cheap to encode, so probing
                    // past it costs compression speed for no ratio gain.
                    if match_len < LAZY_MATCH_THRESHOLD && pos + 1 < limit {
                        let val2 = std::ptr::read_unaligned(src_ptr.add(pos + 1) as *const u32);
                        let h2 = hash4(val2);
                        let candidate2 = table[h2] as usize;
                        let offset2 = (pos + 1).wrapping_sub(candidate2);
                        if offset2 > 0 && offset2 < WINDOW_SIZE && candidate2 < pos + 1 {
                            let candidate2_val = std::ptr::read_unaligned(src_ptr.add(candidate2) as *const u32);
                            if val2 == candidate2_val {
                                let match_len2 = common_prefix_len_avx2(
                                    src_ptr.add(pos + 1),
                                    src_ptr.add(candidate2),
                                    block_end - (pos + 1),
                                );
                                if match_len2 > match_len {
                                    table[h2] = (pos + 1) as u32;
                                    pos += 1;
                                    match_len = match_len2;
                                    match_offset = offset2;
                                }
                            }
                        }
                    }

                    // Flush pending literals
                    let lit_count = pos - anchor;
                    let lit_src = anchor;

                    // Literal runs longer than one escape field are split off
                    // into their own literal-only tokens first.
                    let (first_lit_len, rem_lit_src) = if lit_count > MAX_LIT_LEN {
                        let head = lit_count - MAX_LIT_LEN;
                        emit_literal_run(
                            std::slice::from_raw_parts(src_ptr.add(lit_src), head),
                            tokens,
                            extras,
                            literals,
                        );
                        (MAX_LIT_LEN, lit_src + head)
                    } else {
                        (lit_count, lit_src)
                    };

                    let first_match_chunk = match_len.min(MAX_MATCH_LEN);
                    let (lc, le) = encode_lit(first_lit_len);
                    let (mc, me) = encode_match(first_match_chunk);
                    tokens.push(Token::from_codes(lc, mc));
                    if let Some(v) = le { extras.push(v); }
                    if let Some(v) = me { extras.push(v); }
                    offsets.push(match_offset as u16);
                    if first_lit_len > 0 {
                        literals.extend_from_slice(std::slice::from_raw_parts(
                            src_ptr.add(rem_lit_src),
                            first_lit_len,
                        ));
                    }

                    let mut rem_match = match_len - first_match_chunk;
                    while rem_match > 0 {
                        let chunk = rem_match.min(MAX_MATCH_LEN);
                        let (mc2, me2) = encode_match(chunk);
                        tokens.push(Token::from_codes(0, mc2));
                        if let Some(v) = me2 { extras.push(v); }
                        offsets.push(match_offset as u16);
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
    let trailing = block_end - anchor;
    if trailing > 0 {
        emit_literal_run(
            std::slice::from_raw_parts(src_ptr.add(anchor), trailing),
            tokens,
            extras,
            literals,
        );
    }
}

/// Emit a literal-only run of any length as one or more tokens carrying no
/// match, splitting at MAX_LIT_LEN so each length fits one u16 escape.
#[inline]
pub fn emit_literal_run(
    mut run: &[u8],
    tokens: &mut Vec<Token>,
    extras: &mut Vec<u16>,
    literals: &mut Vec<u8>,
) {
    while !run.is_empty() {
        let n = run.len().min(MAX_LIT_LEN);
        let (lc, le) = encode_lit(n);
        tokens.push(Token::from_codes(lc, 0));
        if let Some(v) = le {
            extras.push(v);
        }
        literals.extend_from_slice(&run[..n]);
        run = &run[n..];
    }
}

/// Single-block standalone compressor (used when compressing independent blocks in parallel).
#[target_feature(enable = "avx2")]
#[target_feature(enable = "bmi2")]
pub unsafe fn compress_avx2(
    src: &[u8],
    tokens: &mut Vec<Token>,
    offsets: &mut Vec<u16>,
    extras: &mut Vec<u16>,
    literals: &mut Vec<u8>,
) {
    let mut table = [0u32; HASH_SIZE];
    compress_chained_avx2(src, 0, src.len(), &mut table, tokens, offsets, extras, literals);
}
