#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use crate::error::{CodecError, Result};
use crate::format::Token;

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

/// 64-Byte AVX-512 Decompressor for Zen 4 / Skylake-X+
#[target_feature(enable = "avx512f")]
#[target_feature(enable = "avx512bw")]
#[target_feature(enable = "bmi2")]
pub unsafe fn decompress_avx512(
    tokens: *const Token,
    num_tokens: usize,
    offsets: *const u16,
    num_offsets: usize,
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
    let mut token_idx = 0;

    // Fast Phase: zero boundary checks in the hot loop
    while token_idx < num_tokens && dst_ptr <= safe_limit {
        let token = std::ptr::read_unaligned(tokens.add(token_idx));
        let (lit_len, match_len) = if token.is_extended_literal() {
            if offset_idx >= num_offsets {
                return Err(CodecError::CorruptedBitstream("Insufficient match offsets in bitstream"));
            }
            let ext_len = std::ptr::read_unaligned(offsets.add(offset_idx)) as usize;
            offset_idx += 1;
            (ext_len, 0)
        } else {
            (token.lit_len(), token.match_len())
        };

        let remaining = block_end.offset_from(dst_ptr) as usize;
        if lit_len + match_len + 64 > remaining {
            if token.is_extended_literal() {
                offset_idx -= 1;
            }
            break;
        }
        token_idx += 1;

        if lit_len > 0 {
            if lit_ptr.add(lit_len) > lit_limit {
                return Err(CodecError::CorruptedBitstream("Literal stream overrun"));
            }
            if lit_ptr.add(32) <= lit_limit && lit_len <= 32 {
                let v = _mm256_loadu_si256(lit_ptr as *const __m256i);
                _mm256_storeu_si256(dst_ptr as *mut __m256i, v);
            } else {
                let mut copied = 0;
                while copied + 32 <= lit_len && lit_ptr.add(copied + 32) <= lit_limit {
                    let v = _mm256_loadu_si256(lit_ptr.add(copied) as *const __m256i);
                    _mm256_storeu_si256(dst_ptr.add(copied) as *mut __m256i, v);
                    copied += 32;
                }
                if copied < lit_len {
                    std::ptr::copy_nonoverlapping(lit_ptr.add(copied), dst_ptr.add(copied), lit_len - copied);
                }
            }

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

    // Boundary Tail Phase: handles remaining tokens with exact non-overshooting copies
    while token_idx < num_tokens {
        let token = std::ptr::read_unaligned(tokens.add(token_idx));
        token_idx += 1;

        let (lit_len, match_len) = if token.is_extended_literal() {
            if offset_idx >= num_offsets {
                return Err(CodecError::CorruptedBitstream("Insufficient match offsets in bitstream"));
            }
            let ext_len = std::ptr::read_unaligned(offsets.add(offset_idx)) as usize;
            offset_idx += 1;
            (ext_len, 0)
        } else {
            (token.lit_len(), token.match_len())
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

/// 32-Byte AVX2 Decompressor
#[target_feature(enable = "avx2")]
#[target_feature(enable = "bmi2")]
pub unsafe fn decompress_avx2(
    tokens: *const Token,
    num_tokens: usize,
    offsets: *const u16,
    num_offsets: usize,
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
    let mut token_idx = 0;

    // Fast Phase
    while token_idx < num_tokens && dst_ptr <= safe_limit {
        let token = std::ptr::read_unaligned(tokens.add(token_idx));
        let (lit_len, match_len) = if token.is_extended_literal() {
            if offset_idx >= num_offsets {
                return Err(CodecError::CorruptedBitstream("Insufficient match offsets in bitstream"));
            }
            let ext_len = std::ptr::read_unaligned(offsets.add(offset_idx)) as usize;
            offset_idx += 1;
            (ext_len, 0)
        } else {
            (token.lit_len(), token.match_len())
        };

        let remaining = block_end.offset_from(dst_ptr) as usize;
        if lit_len + match_len + 32 > remaining {
            if token.is_extended_literal() {
                offset_idx -= 1;
            }
            break;
        }
        token_idx += 1;

        if lit_len > 0 {
            if lit_ptr.add(lit_len) > lit_limit {
                return Err(CodecError::CorruptedBitstream("Literal stream overrun"));
            }
            if lit_ptr.add(32) <= lit_limit && lit_len <= 32 {
                let v0 = _mm256_loadu_si256(lit_ptr as *const __m256i);
                _mm256_storeu_si256(dst_ptr as *mut __m256i, v0);
            } else {
                let mut copied = 0;
                while copied + 32 <= lit_len && lit_ptr.add(copied + 32) <= lit_limit {
                    let v = _mm256_loadu_si256(lit_ptr.add(copied) as *const __m256i);
                    _mm256_storeu_si256(dst_ptr.add(copied) as *mut __m256i, v);
                    copied += 32;
                }
                if copied < lit_len {
                    std::ptr::copy_nonoverlapping(lit_ptr.add(copied), dst_ptr.add(copied), lit_len - copied);
                }
            }

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

        let (lit_len, match_len) = if token.is_extended_literal() {
            if offset_idx >= num_offsets {
                return Err(CodecError::CorruptedBitstream("Insufficient match offsets in bitstream"));
            }
            let ext_len = std::ptr::read_unaligned(offsets.add(offset_idx)) as usize;
            offset_idx += 1;
            (ext_len, 0)
        } else {
            (token.lit_len(), token.match_len())
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
