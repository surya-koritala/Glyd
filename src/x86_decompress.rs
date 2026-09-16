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
    tokens: &[Token],
    offsets: &[u16],
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
    let num_tokens = tokens.len();

    // Fast Phase: zero boundary checks in the hot loop
    while token_idx < num_tokens && dst_ptr <= safe_limit {
        let token = *tokens.get_unchecked(token_idx);
        let lit_len = token.lit_len();
        let match_len = token.match_len();

        if dst_ptr.add(lit_len + match_len + 64) > block_end {
            break;
        }
        token_idx += 1;

        if lit_len > 0 {
            if lit_ptr.add(32) <= lit_limit {
                if lit_len <= 8 {
                    std::ptr::copy_nonoverlapping(lit_ptr, dst_ptr, 8);
                } else if lit_len <= 16 {
                    std::ptr::copy_nonoverlapping(lit_ptr, dst_ptr, 16);
                } else {
                    let v = _mm256_loadu_si256(lit_ptr as *const __m256i);
                    _mm256_storeu_si256(dst_ptr as *mut __m256i, v);
                }
            } else {
                std::ptr::copy_nonoverlapping(lit_ptr, dst_ptr, lit_len);
            }

            lit_ptr = lit_ptr.add(lit_len);
            dst_ptr = dst_ptr.add(lit_len);
        }

        if match_len > 0 {
            if offset_idx >= offsets.len() {
                return Err(CodecError::CorruptedBitstream("Insufficient match offsets in bitstream"));
            }
            let offset = *offsets.get_unchecked(offset_idx) as usize;
            offset_idx += 1;

            let available = dst_ptr.offset_from(buffer_start) as usize;
            if offset == 0 || offset > available {
                return Err(CodecError::OffsetOutOfBounds { offset, available });
            }

            let match_src = dst_ptr.sub(offset);

            if offset >= 64 {
                let match_end = dst_ptr.add(match_len);
                let mut s = match_src;
                let mut d = dst_ptr;
                while d < match_end {
                    let v0 = _mm512_loadu_si512(s as *const _);
                    _mm512_storeu_si512(d as *mut _, v0);
                    s = s.add(64);
                    d = d.add(64);
                }
            } else if offset >= 32 {
                let match_end = dst_ptr.add(match_len);
                let mut s = match_src;
                let mut d = dst_ptr;
                while d < match_end {
                    let v = _mm256_loadu_si256(s as *const __m256i);
                    _mm256_storeu_si256(d as *mut __m256i, v);
                    s = s.add(32);
                    d = d.add(32);
                }
            } else if offset >= 16 {
                let match_end = dst_ptr.add(match_len);
                let mut s = match_src;
                let mut d = dst_ptr;
                while d < match_end {
                    let v = _mm_loadu_si128(s as *const __m128i);
                    _mm_storeu_si128(d as *mut __m128i, v);
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
                let raw_pattern = _mm_loadu_si128(match_src as *const __m128i);
                let masks = &SHUFFLE_MASKS[offset];

                let c0 = _mm_shuffle_epi8(
                    raw_pattern,
                    _mm_loadu_si128(masks[0].as_ptr() as *const __m128i),
                );
                _mm_storeu_si128(dst_ptr as *mut __m128i, c0);

                if match_len > 16 {
                    let c1 = _mm_shuffle_epi8(
                        raw_pattern,
                        _mm_loadu_si128(masks[1].as_ptr() as *const __m128i),
                    );
                    _mm_storeu_si128(dst_ptr.add(16) as *mut __m128i, c1);

                    if match_len > 32 {
                        let c2 = _mm_shuffle_epi8(
                            raw_pattern,
                            _mm_loadu_si128(masks[2].as_ptr() as *const __m128i),
                        );
                        _mm_storeu_si128(dst_ptr.add(32) as *mut __m128i, c2);

                        if match_len > 48 {
                            let c3 = _mm_shuffle_epi8(
                                raw_pattern,
                                _mm_loadu_si128(masks[3].as_ptr() as *const __m128i),
                            );
                            _mm_storeu_si128(dst_ptr.add(48) as *mut __m128i, c3);

                            if match_len > 64 {
                                let safe_dist = PERIODIC_SAFE_OFFSETS[offset];
                                let mut d = dst_ptr.add(64);
                                let match_end = dst_ptr.add(match_len);
                                while d < match_end {
                                    let v = _mm256_loadu_si256(d.sub(safe_dist) as *const __m256i);
                                    _mm256_storeu_si256(d as *mut __m256i, v);
                                    d = d.add(32);
                                }
                            }
                        }
                    }
                }
            }

            dst_ptr = dst_ptr.add(match_len);
        }
    }

    // Boundary Tail Phase: handles remaining tokens with exact non-overshooting copies
    while token_idx < num_tokens {
        let token = *tokens.get_unchecked(token_idx);
        token_idx += 1;

        let lit_len = token.lit_len();
        let match_len = token.match_len();

        if lit_len > 0 {
            std::ptr::copy_nonoverlapping(lit_ptr, dst_ptr, lit_len);
            lit_ptr = lit_ptr.add(lit_len);
            dst_ptr = dst_ptr.add(lit_len);
        }

        if match_len > 0 {
            if offset_idx >= offsets.len() {
                return Err(CodecError::CorruptedBitstream("Insufficient match offsets in bitstream"));
            }
            let offset = *offsets.get_unchecked(offset_idx) as usize;
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
    tokens: &[Token],
    offsets: &[u16],
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
    let num_tokens = tokens.len();

    // Fast Phase
    while token_idx < num_tokens && dst_ptr <= safe_limit {
        let token = *tokens.get_unchecked(token_idx);
        let lit_len = token.lit_len();
        let match_len = token.match_len();

        if dst_ptr.add(lit_len + match_len + 32) > block_end {
            break;
        }
        token_idx += 1;

        if lit_len > 0 {
            if lit_ptr.add(32) <= lit_limit {
                if lit_len <= 8 {
                    std::ptr::copy_nonoverlapping(lit_ptr, dst_ptr, 8);
                } else if lit_len <= 16 {
                    std::ptr::copy_nonoverlapping(lit_ptr, dst_ptr, 16);
                } else {
                    let v0 = _mm256_loadu_si256(lit_ptr as *const __m256i);
                    _mm256_storeu_si256(dst_ptr as *mut __m256i, v0);
                }
            } else {
                std::ptr::copy_nonoverlapping(lit_ptr, dst_ptr, lit_len);
            }

            lit_ptr = lit_ptr.add(lit_len);
            dst_ptr = dst_ptr.add(lit_len);
        }

        if match_len > 0 {
            if offset_idx >= offsets.len() {
                return Err(CodecError::CorruptedBitstream("Insufficient match offsets in bitstream"));
            }
            let offset = *offsets.get_unchecked(offset_idx) as usize;
            offset_idx += 1;

            let available = dst_ptr.offset_from(buffer_start) as usize;
            if offset == 0 || offset > available {
                return Err(CodecError::OffsetOutOfBounds { offset, available });
            }

            let match_src = dst_ptr.sub(offset);

            if offset >= 32 {
                let match_end = dst_ptr.add(match_len);
                let mut s = match_src;
                let mut d = dst_ptr;
                while d < match_end {
                    let v = _mm256_loadu_si256(s as *const __m256i);
                    _mm256_storeu_si256(d as *mut __m256i, v);
                    s = s.add(32);
                    d = d.add(32);
                }
            } else if offset >= 16 {
                let match_end = dst_ptr.add(match_len);
                let mut s = match_src;
                let mut d = dst_ptr;
                while d < match_end {
                    let v = _mm_loadu_si128(s as *const __m128i);
                    _mm_storeu_si128(d as *mut __m128i, v);
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
                let raw_pattern = _mm_loadu_si128(match_src as *const __m128i);
                let masks = &SHUFFLE_MASKS[offset];

                let c0 = _mm_shuffle_epi8(
                    raw_pattern,
                    _mm_loadu_si128(masks[0].as_ptr() as *const __m128i),
                );
                _mm_storeu_si128(dst_ptr as *mut __m128i, c0);

                if match_len > 16 {
                    let c1 = _mm_shuffle_epi8(
                        raw_pattern,
                        _mm_loadu_si128(masks[1].as_ptr() as *const __m128i),
                    );
                    _mm_storeu_si128(dst_ptr.add(16) as *mut __m128i, c1);

                    if match_len > 32 {
                        let c2 = _mm_shuffle_epi8(
                            raw_pattern,
                            _mm_loadu_si128(masks[2].as_ptr() as *const __m128i),
                        );
                        _mm_storeu_si128(dst_ptr.add(32) as *mut __m128i, c2);

                        if match_len > 48 {
                            let c3 = _mm_shuffle_epi8(
                                raw_pattern,
                                _mm_loadu_si128(masks[3].as_ptr() as *const __m128i),
                            );
                            _mm_storeu_si128(dst_ptr.add(48) as *mut __m128i, c3);

                            if match_len > 64 {
                                let safe_dist = PERIODIC_SAFE_OFFSETS[offset];
                                let mut d = dst_ptr.add(64);
                                let match_end = dst_ptr.add(match_len);
                                while d < match_end {
                                    let v = _mm256_loadu_si256(d.sub(safe_dist) as *const __m256i);
                                    _mm256_storeu_si256(d as *mut __m256i, v);
                                    d = d.add(32);
                                }
                            }
                        }
                    }
                }
            }

            dst_ptr = dst_ptr.add(match_len);
        }
    }

    // Boundary Tail Phase
    while token_idx < num_tokens {
        let token = *tokens.get_unchecked(token_idx);
        token_idx += 1;

        let lit_len = token.lit_len();
        let match_len = token.match_len();

        if lit_len > 0 {
            std::ptr::copy_nonoverlapping(lit_ptr, dst_ptr, lit_len);
            lit_ptr = lit_ptr.add(lit_len);
            dst_ptr = dst_ptr.add(lit_len);
        }

        if match_len > 0 {
            if offset_idx >= offsets.len() {
                return Err(CodecError::CorruptedBitstream("Insufficient match offsets in bitstream"));
            }
            let offset = *offsets.get_unchecked(offset_idx) as usize;
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
