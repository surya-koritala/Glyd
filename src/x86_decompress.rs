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
    // Fast Phase. Offsets are read as one 4-byte load, so it needs 4 bytes
    // of offset stream left; the last few tokens take the tail phase.
    while token_idx < num_tokens && dst_ptr <= safe_limit && off_pos + 4 <= offsets_len {
        let tv = *table.get_unchecked(*tokens.add(token_idx) as usize);
        let mut lit_len = (tv & 0xFF) as usize;
        let mut match_len = ((tv >> 8) & 0xFF) as usize;
        let mut e = extra_idx;
        if tv & TOKEN_ESCAPE_MASK != 0 {
            if tv & TOKEN_LIT_ESCAPE != 0 {
                lit_len = read_escape(extras, extras_len, &mut e, ESCAPE_BASE_LIT)?;
            }
            if tv & TOKEN_MATCH_ESCAPE != 0 {
                match_len = read_escape(extras, extras_len, &mut e, esc_base_match)?;
            }
        }

        let remaining = block_end.offset_from(dst_ptr) as usize;
        if lit_len + match_len + 32 > remaining {
            break;
        }
        extra_idx = e;
        token_idx += 1;

        if lit_ptr.add(lit_len) > lit_limit {
            return Err(CodecError::CorruptedBitstream("Literal stream overrun"));
        }
        // Literal wildcopy: unconditional 32-byte stores. Overshoot is
        // absorbed by the destination headroom guaranteed above and by the
        // literal-stream padding, so no per-token length branch is needed.
        if lit_ptr.add(lit_len + 32) <= lit_limit {
            let mut n = 0usize;
            loop {
                let v = _mm256_loadu_si256(lit_ptr.add(n) as *const __m256i);
                _mm256_storeu_si256(dst_ptr.add(n) as *mut __m256i, v);
                n += 32;
                if n >= lit_len {
                    break;
                }
            }
        } else if lit_len > 0 {
            std::ptr::copy_nonoverlapping(lit_ptr, dst_ptr, lit_len);
        }
        lit_ptr = lit_ptr.add(lit_len);
        dst_ptr = dst_ptr.add(lit_len);

        if match_len > 0 {
            // Offset: one unaligned load masked to the token's width.
            let width = ((tv >> TOKEN_OFF_SHIFT) & 7) as usize;
            let raw = std::ptr::read_unaligned(offsets.add(off_pos) as *const u32);
            let offset = (raw & *OFFSET_MASK.get_unchecked(width)) as usize;
            off_pos += width;

            let available = dst_ptr.offset_from(buffer_start) as usize;
            if offset == 0 || offset > available {
                return Err(CodecError::OffsetOutOfBounds { offset, available });
            }

            let match_src = dst_ptr.sub(offset);
            let match_end = dst_ptr.add(match_len);

            // The compressor never emits a match longer than its offset, so
            // for offsets of 8 and up the 32-byte copies never read a byte
            // they have not yet written. Shorter offsets only arise from
            // foreign encoders and take the byte loop.
            if offset >= 32 {
                let mut s = match_src;
                let mut d = dst_ptr;
                while d < match_end {
                    let v = _mm256_loadu_si256(s as *const __m256i);
                    _mm256_storeu_si256(d as *mut __m256i, v);
                    s = s.add(32);
                    d = d.add(32);
                }
            } else if offset >= 16 {
                let mut s = match_src;
                let mut d = dst_ptr;
                while d < match_end {
                    let v = _mm_loadu_si128(s as *const __m128i);
                    _mm_storeu_si128(d as *mut __m128i, v);
                    s = s.add(16);
                    d = d.add(16);
                }
            } else if offset >= 8 {
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

    // Boundary Tail Phase: byte-exact, every access checked.
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
            std::ptr::copy_nonoverlapping(lit_ptr, dst_ptr, lit_len);
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

            let mut s = dst_ptr.sub(offset);
            let mut d = dst_ptr;
            for _ in 0..match_len {
                *d = *s;
                s = s.add(1);
                d = d.add(1);
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
