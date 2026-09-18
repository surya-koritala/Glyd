#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use crate::error::{CodecError, Result};
use crate::fallback::read_escape;
use crate::format::{ESCAPE_BASE_LIT, ESCAPE_CONT, OFFSET_BYTES,
    TOKEN_LIT_ESCAPE, TOKEN_MATCH_ESCAPE, TOKEN_OFF_SHIFT};

/// 32-Byte AVX2 Decompressor.
///
/// `dst` is the block's own output region plus padding; `buffer_start` is
/// where the match window begins, at or before `dst`. `table` and
/// `esc_base_match` select the length bias (ordinary or FLAG_DENSE).
#[target_feature(enable = "avx2")]
#[target_feature(enable = "bmi2")]
pub unsafe fn decompress_avx2(
    tokens: *const u8,
    num_tokens: usize,
    offsets: *const u8,
    offsets_len: usize,
    extras: *const u8,
    extras_len: usize,
    literals: &[u8],
    dst: &mut [u8],
    buffer_start: *const u8,
    uncompressed_len: usize,
    table: &[u32; 256],
    esc_base_match: usize,
) -> Result<usize> {
    if uncompressed_len > dst.len() {
        return Err(CodecError::OutputBufferTooSmall {
            required: uncompressed_len,
            provided: dst.len(),
        });
    }
    let mut dst_ptr = dst.as_mut_ptr();
    let block_start = dst_ptr;
    let block_end = dst_ptr.add(uncompressed_len);
    let safe_limit = if uncompressed_len >= 64 {
        block_end.sub(64)
    } else {
        block_start
    };

    let mut lit_ptr = literals.as_ptr();
    let lit_limit = literals.as_ptr().add(literals.len());
    let mut off_pos = 0usize;
    let mut extra_idx = 0usize;
    let mut token_idx = 0;

    // Fast phase: 32 tokens at a time.
    //
    // Ablation (examples/dec_ablate.rs) put 60% of decode time in the token
    // walk, and half of that in escapes, which a fifth of tokens carry and
    // which mispredict. So the walk is vectorized: one AVX2 pass turns 32
    // token bytes into literal and match lengths, with escape lanes marked;
    // the escaped lanes are then patched from the extras stream in a short
    // loop over the escape mask (no data-dependent branch per token); and a
    // copy-only loop consumes the arrays. Bounds are checked once per chunk
    // from the chunk's totals. v6's constant-stride offsets mean the copy loop
    // reads offsets straight from the stream, no positions to compute.
    //
    // A chunk that cannot be taken (block tail, a 255-continuation escape,
    // any stream too short) is left to the careful one-token step below, and
    // the fast phase resumes on the next token.
    const CHUNK: usize = 32;
    let bias = (esc_base_match - 15) as i8;
    let seven = _mm256_set1_epi8(7);
    let fifteen = _mm256_set1_epi8(15);
    let zero = _mm256_setzero_si256();
    let v_bias = _mm256_set1_epi8(bias);

    let mut litl = [0u16; CHUNK];
    let mut mll = [0u16; CHUNK];
    // Tokens to take carefully before the next chunk attempt. A chunk
    // rejected after its pre-pass (a 255 continuation, a stream near its
    // end) would otherwise be retried one token later, still containing the
    // same offender, up to 31 times; measured at ~18 ms of a ~47 ms decode.
    let mut careful = 0usize;

    'tokens: loop {
        // Try a chunk.
        'chunk: {
            if careful > 0 {
                careful -= 1;
                break 'chunk;
            }
            if token_idx + CHUNK > num_tokens || dst_ptr > safe_limit {
                break 'chunk;
            }
            let v = _mm256_loadu_si256(tokens.add(token_idx) as *const __m256i);
            let lit = _mm256_and_si256(v, seven);
            let mc = _mm256_and_si256(_mm256_srli_epi16(v, 3), fifteen);
            let lit_esc = _mm256_cmpeq_epi8(lit, seven);
            let m_esc = _mm256_cmpeq_epi8(mc, fifteen);
            let m_zero = _mm256_cmpeq_epi8(mc, zero);
            // Match length: code + bias, zero where there is no match. Escape
            // lanes hold the placeholder 15 + bias until patched.
            let ml = _mm256_andnot_si256(m_zero, _mm256_add_epi8(mc, v_bias));
            let esc_mask = _mm256_movemask_epi8(_mm256_or_si256(lit_esc, m_esc)) as u32;
            let lit_esc_mask = _mm256_movemask_epi8(lit_esc) as u32;
            let m_esc_mask = _mm256_movemask_epi8(m_esc) as u32;
            let n_match = CHUNK - (_mm256_movemask_epi8(m_zero) as u32).count_ones() as usize;

            // Widen to u16 arrays.
            let lit_lo = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(lit));
            let lit_hi = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(lit, 1));
            let ml_lo = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(ml));
            let ml_hi = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(ml, 1));
            _mm256_storeu_si256(litl.as_mut_ptr() as *mut __m256i, lit_lo);
            _mm256_storeu_si256(litl.as_mut_ptr().add(16) as *mut __m256i, lit_hi);
            _mm256_storeu_si256(mll.as_mut_ptr() as *mut __m256i, ml_lo);
            _mm256_storeu_si256(mll.as_mut_ptr().add(16) as *mut __m256i, ml_hi);

            // Chunk totals from the byte vectors (sum of 8-byte groups).
            let sum8 = |x: __m256i| -> usize {
                let s = _mm256_sad_epu8(x, zero);
                (_mm256_extract_epi64(s, 0) + _mm256_extract_epi64(s, 1)
                    + _mm256_extract_epi64(s, 2) + _mm256_extract_epi64(s, 3)) as usize
            };
            let mut lit_sum = sum8(lit);
            let mut ml_sum = sum8(ml);

            // Patch escaped lanes from the extras stream: one byte per
            // escaped field, literal first, in token order. A 255 byte means
            // a u16 continuation, which this path does not take.
            let mut e = extra_idx;
            if esc_mask != 0 {
                let need = esc_mask.count_ones() as usize
                    + (lit_esc_mask & m_esc_mask).count_ones() as usize;
                if e + need > extras_len {
                    careful = CHUNK - 1;
                    break 'chunk;
                }
                // Branch-free per escaped token: which field(s) escaped is
                // ~50/50 and unpredictable, so both lanes are selected
                // arithmetically and stored unconditionally. (With `if`s here
                // the mispredicts cost as much as the old per-token branch.)
                let mut bits = esc_mask;
                let mut cont = 0usize;
                while bits != 0 {
                    let i = bits.trailing_zeros() as usize;
                    bits &= bits - 1;
                    let l = ((lit_esc_mask >> i) & 1) as usize;
                    let m = ((m_esc_mask >> i) & 1) as usize;
                    let b1 = *extras.add(e) as usize;
                    let b2 = *extras.add(e + l) as usize;
                    cont |= (l & (b1 == ESCAPE_CONT as usize) as usize)
                        | (m & (b2 == ESCAPE_CONT as usize) as usize);
                    // Literal lane: current value, or base + byte when escaped
                    // (a token whose match escaped keeps its own literal count).
                    let cur_l = *litl.get_unchecked(i) as usize;
                    let lm = l.wrapping_neg();
                    let nl = (cur_l & !lm) | ((ESCAPE_BASE_LIT + b1) & lm);
                    lit_sum += b1 * l;
                    *litl.get_unchecked_mut(i) = nl as u16;
                    // Match lane: current value, or base + byte when escaped.
                    let cur = *mll.get_unchecked(i) as usize;
                    let mm = m.wrapping_neg();
                    let nm = (cur & !mm) | ((esc_base_match + b2) & mm);
                    ml_sum += b2 * m;
                    *mll.get_unchecked_mut(i) = nm as u16;
                    e += l + m;
                }
                if cont != 0 {
                    careful = CHUNK - 1;
                    break 'chunk;
                }
            }

            // Chunk bounds: wild stores need 64 bytes past the chunk's last
            // byte; literal loads need 64 past the chunk's literals; offsets
            // need OFFSET_BYTES per match.
            let remaining = block_end.offset_from(dst_ptr) as usize;
            if lit_sum + ml_sum + 64 > remaining
                || lit_ptr.add(lit_sum + 64) > lit_limit
                || off_pos + n_match * OFFSET_BYTES > offsets_len
            {
                careful = CHUNK - 1;
                break 'chunk;
            }

            // Copy-only loop.
            let tok_base = tokens.add(token_idx);
            for i in 0..CHUNK {
                let lit = *litl.get_unchecked(i) as usize;
                let ml = *mll.get_unchecked(i) as usize;

                let v = _mm256_loadu_si256(lit_ptr as *const __m256i);
                _mm256_storeu_si256(dst_ptr as *mut __m256i, v);
                if lit > 32 {
                    let mut n = 32usize;
                    while n < lit {
                        let v = _mm256_loadu_si256(lit_ptr.add(n) as *const __m256i);
                        _mm256_storeu_si256(dst_ptr.add(n) as *mut __m256i, v);
                        n += 32;
                    }
                }
                lit_ptr = lit_ptr.add(lit);
                dst_ptr = dst_ptr.add(lit);

                if ml != 0 {
                    let lo = std::ptr::read_unaligned(offsets.add(off_pos) as *const u16) as usize;
                    let offset = lo | (((*tok_base.add(i) >> 7) as usize) << 16);
                    off_pos += OFFSET_BYTES;
                    let available = dst_ptr.offset_from(buffer_start) as usize;
                    if offset == 0 || offset > available {
                        return Err(CodecError::OffsetOutOfBounds { offset, available });
                    }
                    let match_src = dst_ptr.sub(offset);
                    if offset >= 32 {
                        let m = _mm256_loadu_si256(match_src as *const __m256i);
                        _mm256_storeu_si256(dst_ptr as *mut __m256i, m);
                        if ml > 32 {
                            let mut n = 32usize;
                            while n < ml {
                                let m = _mm256_loadu_si256(match_src.add(n) as *const __m256i);
                                _mm256_storeu_si256(dst_ptr.add(n) as *mut __m256i, m);
                                n += 32;
                            }
                        }
                    } else if offset >= 16 {
                        let match_end = dst_ptr.add(ml);
                        let mut s = match_src;
                        let mut d = dst_ptr;
                        while d < match_end {
                            let m = _mm_loadu_si128(s as *const __m128i);
                            _mm_storeu_si128(d as *mut __m128i, m);
                            s = s.add(16);
                            d = d.add(16);
                        }
                    } else if offset >= 8 {
                        let match_end = dst_ptr.add(ml);
                        let mut s = match_src;
                        let mut d = dst_ptr;
                        while d < match_end {
                            std::ptr::copy_nonoverlapping(s, d, 8);
                            s = s.add(8);
                            d = d.add(8);
                        }
                    } else {
                        let match_end = dst_ptr.add(ml);
                        let mut s = match_src;
                        let mut d = dst_ptr;
                        while d < match_end {
                            *d = *s;
                            d = d.add(1);
                            s = s.add(1);
                        }
                    }
                    dst_ptr = dst_ptr.add(ml);
                }
            }
            token_idx += CHUNK;
            extra_idx = e;
            continue 'tokens;
        }

        // Careful step: one token, every access checked.
        if token_idx >= num_tokens {
            break 'tokens;
        }
        let tv = *table.get_unchecked(*tokens.add(token_idx) as usize);
        token_idx += 1;

        let mut lit_len = (tv & 0xFF) as usize;
        let mut match_len = ((tv >> 8) & 0xFF) as usize;
        if tv & TOKEN_LIT_ESCAPE != 0 {
            lit_len = read_escape(extras, extras_len, &mut extra_idx, ESCAPE_BASE_LIT)?;
        }
        if tv & TOKEN_MATCH_ESCAPE != 0 {
            match_len = read_escape(extras, extras_len, &mut extra_idx, esc_base_match)?;
        }

        // A corrupted token must never write past the declared block length.
        // This check has to happen before any copy, not after.
        let remaining = if dst_ptr <= block_end {
            block_end.offset_from(dst_ptr) as usize
        } else {
            0
        };
        if lit_len + match_len > remaining {
            return Err(CodecError::CorruptedBitstream(
                "Token output exceeds declared block length",
            ));
        }

        if lit_len > 0 {
            if lit_ptr.add(lit_len) > lit_limit {
                return Err(CodecError::CorruptedBitstream("Literal stream overrun"));
            }
            if lit_ptr.add(lit_len + 32) <= lit_limit && remaining >= lit_len + 32 {
                let mut n = 0usize;
                loop {
                    let v = _mm256_loadu_si256(lit_ptr.add(n) as *const __m256i);
                    _mm256_storeu_si256(dst_ptr.add(n) as *mut __m256i, v);
                    n += 32;
                    if n >= lit_len {
                        break;
                    }
                }
            } else {
                std::ptr::copy_nonoverlapping(lit_ptr, dst_ptr, lit_len);
            }
            lit_ptr = lit_ptr.add(lit_len);
            dst_ptr = dst_ptr.add(lit_len);
        }

        if match_len > 0 {
            if off_pos + OFFSET_BYTES > offsets_len {
                return Err(CodecError::CorruptedBitstream("Insufficient match offsets in bitstream"));
            }
            let lo = u16::from_le_bytes([*offsets.add(off_pos), *offsets.add(off_pos + 1)]) as usize;
            let offset = lo | ((((tv >> TOKEN_OFF_SHIFT) & 1) as usize) << 16);
            off_pos += OFFSET_BYTES;

            let available = dst_ptr.offset_from(buffer_start) as usize;
            if offset == 0 || offset > available {
                return Err(CodecError::OffsetOutOfBounds { offset, available });
            }

            let match_src = dst_ptr.sub(offset);
            let match_end = dst_ptr.add(match_len);
            if remaining >= lit_len + match_len + 32 && offset >= 32 {
                let mut s = match_src;
                let mut d = dst_ptr;
                while d < match_end {
                    let v = _mm256_loadu_si256(s as *const __m256i);
                    _mm256_storeu_si256(d as *mut __m256i, v);
                    s = s.add(32);
                    d = d.add(32);
                }
            } else if remaining >= lit_len + match_len + 16 && offset >= 16 {
                let mut s = match_src;
                let mut d = dst_ptr;
                while d < match_end {
                    let v = _mm_loadu_si128(s as *const __m128i);
                    _mm_storeu_si128(d as *mut __m128i, v);
                    s = s.add(16);
                    d = d.add(16);
                }
            } else if remaining >= lit_len + match_len + 8 && offset >= 8 {
                let mut s = match_src;
                let mut d = dst_ptr;
                while d < match_end {
                    std::ptr::copy_nonoverlapping(s, d, 8);
                    s = s.add(8);
                    d = d.add(8);
                }
            } else {
                let mut s = match_src;
                let mut d = dst_ptr;
                while d < match_end {
                    *d = *s;
                    d = d.add(1);
                    s = s.add(1);
                }
            }
            dst_ptr = dst_ptr.add(match_len);
        }
    }

    let written = dst_ptr.offset_from(block_start) as usize;
    if written != uncompressed_len {
        return Err(CodecError::CorruptedBitstream("Decompressed length mismatch"));
    }

    Ok(written)
}
