#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use crate::error::{CodecError, Result};
use crate::fallback::read_escape;
use crate::format::{ESCAPE_BASE_LIT, OFFSET_MASK, TOKEN_ESCAPE_MASK,
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

    // Fast phase. Ablation (examples/dec_ablate.rs) priced the per-token
    // bound checks at ~8 ms of a ~66 ms decode, for four compares that almost
    // never trip. So bounds are paid in bulk: a token without an escape reads
    // at most 2 literal bytes and 3 offset bytes and advances the output by
    // at most 21, while each of its two 32-byte wild stores needs 64 bytes of
    // headroom. From the cursors, `credit` is the number of such tokens that
    // are provably safe; the loop then spends one decrement per token and
    // recomputes only when the credit runs out or after an escape, whose
    // lengths are unbounded and therefore checked on the spot.
    macro_rules! credit {
        () => {{
            let out = if dst_ptr <= safe_limit { safe_limit.offset_from(dst_ptr) as usize >> 5 } else { 0 };
            let lit = if lit_ptr.add(64) <= lit_limit { lit_limit.offset_from(lit_ptr.add(64)) as usize >> 1 } else { 0 };
            let off = if off_pos + 4 <= offsets_len { (offsets_len - 4 - off_pos) >> 2 } else { 0 };
            let tok = num_tokens - token_idx;
            out.min(lit).min(off).min(tok)
        }};
    }
    let mut credit = credit!();
    'fast: while credit > 0 {
        let tv = *table.get_unchecked(*tokens.add(token_idx) as usize);
        let mut lit_len = (tv & 0xFF) as usize;
        let mut match_len = ((tv >> 8) & 0xFF) as usize;
        let width = ((tv >> TOKEN_OFF_SHIFT) & 7) as usize;
        let escaped = tv & TOKEN_ESCAPE_MASK != 0;
        if escaped {
            let mut e = extra_idx;
            if tv & TOKEN_LIT_ESCAPE != 0 {
                lit_len = read_escape(extras, extras_len, &mut e, ESCAPE_BASE_LIT)?;
            }
            if tv & TOKEN_MATCH_ESCAPE != 0 {
                match_len = read_escape(extras, extras_len, &mut e, esc_base_match)?;
            }
            let remaining = block_end.offset_from(dst_ptr) as usize;
            if lit_len + match_len + 64 > remaining || lit_ptr.add(lit_len + 64) > lit_limit {
                break 'fast;
            }
            extra_idx = e;
        }
        token_idx += 1;
        credit -= 1;

        let v = _mm256_loadu_si256(lit_ptr as *const __m256i);
        _mm256_storeu_si256(dst_ptr as *mut __m256i, v);
        if lit_len > 32 {
            let mut n = 32usize;
            while n < lit_len {
                let v = _mm256_loadu_si256(lit_ptr.add(n) as *const __m256i);
                _mm256_storeu_si256(dst_ptr.add(n) as *mut __m256i, v);
                n += 32;
            }
        }
        lit_ptr = lit_ptr.add(lit_len);
        dst_ptr = dst_ptr.add(lit_len);

        if match_len != 0 {
            let raw = std::ptr::read_unaligned(offsets.add(off_pos) as *const u32);
            let offset = (raw & *OFFSET_MASK.get_unchecked(width)) as usize;
            off_pos += width;
            let available = dst_ptr.offset_from(buffer_start) as usize;
            if offset == 0 || offset > available {
                return Err(CodecError::OffsetOutOfBounds { offset, available });
            }
            let match_src = dst_ptr.sub(offset);
            if offset >= 32 {
                let m = _mm256_loadu_si256(match_src as *const __m256i);
                _mm256_storeu_si256(dst_ptr as *mut __m256i, m);
                if match_len > 32 {
                    let mut n = 32usize;
                    while n < match_len {
                        let m = _mm256_loadu_si256(match_src.add(n) as *const __m256i);
                        _mm256_storeu_si256(dst_ptr.add(n) as *mut __m256i, m);
                        n += 32;
                    }
                }
            } else if offset >= 16 {
                let match_end = dst_ptr.add(match_len);
                let mut s = match_src;
                let mut d = dst_ptr;
                while d < match_end {
                    let m = _mm_loadu_si128(s as *const __m128i);
                    _mm_storeu_si128(d as *mut __m128i, m);
                    s = s.add(16);
                    d = d.add(16);
                }
            } else if offset >= 8 {
                let match_end = dst_ptr.add(match_len);
                let mut s = match_src;
                let mut d = dst_ptr;
                while d < match_end {
                    std::ptr::copy_nonoverlapping(s, d, 8);
                    s = s.add(8);
                    d = d.add(8);
                }
            } else {
                let match_end = dst_ptr.add(match_len);
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

        // An escape consumed more than a token's budget (2 literal bytes, 32
        // output bytes); charge the excess in token units rather than
        // recomputing, since a quarter of tokens escape. Recompute only when
        // the credit is exhausted.
        if escaped {
            let extra = (lit_len >> 1) + ((lit_len + match_len) >> 5);
            credit = credit.saturating_sub(extra);
        }
        if credit == 0 {
            credit = credit!();
        }
    }

    // Careful loop: the block's tail and anything the fast phase declined.
    // Every access checked.
    while token_idx < num_tokens {
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
                // Long literal run with room: 32-byte wild copies.
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
            let width = ((tv >> TOKEN_OFF_SHIFT) & 7) as usize;
            if off_pos + width > offsets_len {
                return Err(CodecError::CorruptedBitstream("Insufficient match offsets in bitstream"));
            }
            let mut offset = 0usize;
            for i in 0..width {
                offset |= (*offsets.add(off_pos + i) as usize) << (8 * i);
            }
            off_pos += width;

            let available = dst_ptr.offset_from(buffer_start) as usize;
            if offset == 0 || offset > available {
                return Err(CodecError::OffsetOutOfBounds { offset, available });
            }

            let match_src = dst_ptr.sub(offset);
            let match_end = dst_ptr.add(match_len);
            // With 32 bytes of headroom past the match, wide copies are safe;
            // the block's last bytes and near offsets take the exact paths.
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
