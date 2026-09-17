#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use crate::error::{CodecError, Result};
use crate::format::{Token, LIT_CODE_ESCAPE, MATCH_CODE_BIAS, MATCH_CODE_ESCAPE,
    TOKEN_ESCAPE_MASK, TOKEN_LIT_ESCAPE, TOKEN_MATCH_ESCAPE, TOKEN_TABLE};

/// Precomputed 16-byte shuffle masks for repeating patterns of period 1..15.
/// SHUFFLE_MASKS[offset][chunk][i] = ((chunk * 16 + i) % offset) as u8
pub static SHUFFLE_MASKS: [[[u8; 16]; 4]; 16] = {
    let mut table = [[[0u8; 16]; 4]; 16];
    let mut offset = 1;
    while offset <= 15 {
        let mut chunk = 0;
        while chunk < 4 {
            let mut i = 0;
            while i < 16 {
                table[offset][chunk][i] = ((chunk * 16 + i) % offset) as u8;
                i += 1;
            }
            chunk += 1;
        }
        offset += 1;
    }
    table
};

/// Smallest multiple of offset that is >= 32.
pub static PERIODIC_SAFE_OFFSETS: [usize; 16] = {
    let mut table = [0usize; 16];
    let mut k = 1;
    while k <= 15 {
        table[k] = ((32 + k - 1) / k) * k;
        k += 1;
    }
    table
};

/// 32-Byte AVX2 Decompressor
#[target_feature(enable = "avx2")]
#[target_feature(enable = "bmi2")]
pub unsafe fn decompress_avx2(
    tokens: *const Token,
    num_tokens: usize,
    offsets: *const u16,
    num_offsets: usize,
    extras: *const u16,
    num_extras: usize,
    literals: &[u8],
    dst: &mut [u8],
    buffer_start: *const u8,
    uncompressed_len: usize,
) -> Result<usize> {
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
    let mut offset_idx = 0;
    let mut extra_idx = 0;
    let mut token_idx = 0;

    // Fast Phase
    while token_idx < num_tokens && dst_ptr <= safe_limit {
        let token = std::ptr::read_unaligned(tokens.add(token_idx));
        let tv = *TOKEN_TABLE.get_unchecked(token.0 as usize);
        let mut lit_len = (tv & 0xFF) as usize;
        let mut match_len = ((tv >> 8) & 0xFF) as usize;
        let mut e = extra_idx;
        if tv & TOKEN_ESCAPE_MASK != 0 {
            let need = ((tv & TOKEN_LIT_ESCAPE) != 0) as usize
                + ((tv & TOKEN_MATCH_ESCAPE) != 0) as usize;
            if e + need > num_extras {
                return Err(CodecError::CorruptedBitstream("Insufficient extras in bitstream"));
            }
            if tv & TOKEN_LIT_ESCAPE != 0 {
                lit_len = std::ptr::read_unaligned(extras.add(e)) as usize;
                e += 1;
            }
            if tv & TOKEN_MATCH_ESCAPE != 0 {
                match_len = std::ptr::read_unaligned(extras.add(e)) as usize;
                e += 1;
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
            if offset_idx >= num_offsets {
                return Err(CodecError::CorruptedBitstream("Insufficient match offsets in bitstream"));
            }
            let offset = std::ptr::read_unaligned(offsets.add(offset_idx)) as usize;
            offset_idx += 1;

            let available = dst_ptr.offset_from(buffer_start) as usize;
            if offset == 0 || offset > available {
                return Err(CodecError::OffsetOutOfBounds { offset, available });
            }

            let match_src = dst_ptr.sub(offset);
            let match_end = dst_ptr.add(match_len);

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
            } else if offset == 1 {
                let v = _mm256_set1_epi8(*match_src as i8);
                let mut d = dst_ptr;
                while d < match_end {
                    _mm256_storeu_si256(d as *mut __m256i, v);
                    d = d.add(32);
                }
            } else if offset == 2 {
                let v = _mm256_set1_epi16(std::ptr::read_unaligned(match_src as *const i16));
                let mut d = dst_ptr;
                while d < match_end {
                    _mm256_storeu_si256(d as *mut __m256i, v);
                    d = d.add(32);
                }
            } else if offset == 4 {
                let v = _mm256_set1_epi32(std::ptr::read_unaligned(match_src as *const i32));
                let mut d = dst_ptr;
                while d < match_end {
                    _mm256_storeu_si256(d as *mut __m256i, v);
                    d = d.add(32);
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

    // Boundary Tail Phase
    while token_idx < num_tokens {
        let token = std::ptr::read_unaligned(tokens.add(token_idx));
        token_idx += 1;

        let lc = token.lit_code();
        let mc = token.match_code();
        let lit_len = if lc == LIT_CODE_ESCAPE {
            if extra_idx >= num_extras {
                return Err(CodecError::CorruptedBitstream("Insufficient extras in bitstream"));
            }
            let v = std::ptr::read_unaligned(extras.add(extra_idx)) as usize;
            extra_idx += 1;
            v
        } else {
            lc
        };
        let match_len = if mc == 0 {
            0
        } else if mc == MATCH_CODE_ESCAPE {
            if extra_idx >= num_extras {
                return Err(CodecError::CorruptedBitstream("Insufficient extras in bitstream"));
            }
            let v = std::ptr::read_unaligned(extras.add(extra_idx)) as usize;
            extra_idx += 1;
            v
        } else {
            mc + MATCH_CODE_BIAS
        };

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
            if offset_idx >= num_offsets {
                return Err(CodecError::CorruptedBitstream("Insufficient match offsets in bitstream"));
            }
            let offset = std::ptr::read_unaligned(offsets.add(offset_idx)) as usize;
            offset_idx += 1;

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
