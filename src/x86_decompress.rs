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

    let mut lit_ptr = literals.as_ptr();
    let mut offset_idx = 0;

    for &token in tokens {
        let lit_len = token.lit_len();
        let match_len = token.match_len();

        // 1. Literal copy (single 64-byte vector store covers any lit_len <= 31)
        if lit_len > 0 {
            let v0 = _mm512_loadu_si512(lit_ptr as *const _);
            _mm512_storeu_si512(dst_ptr as *mut _, v0);

            lit_ptr = lit_ptr.add(lit_len);
            dst_ptr = dst_ptr.add(lit_len);
        }

        // 2. Match copy (up to 2047 bytes)
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
                // Wide 64-byte AVX-512 non-overlapping path
                if match_len <= 64 {
                    let v0 = _mm512_loadu_si512(match_src as *const _);
                    _mm512_storeu_si512(dst_ptr as *mut _, v0);
                } else {
                    let mut m = match_len;
                    let mut s = match_src;
                    let mut d = dst_ptr;
                    while m >= 64 {
                        let v = _mm512_loadu_si512(s as *const _);
                        _mm512_storeu_si512(d as *mut _, v);
                        s = s.add(64);
                        d = d.add(64);
                        m -= 64;
                    }
                    if m > 0 {
                        let v = _mm512_loadu_si512(s as *const _);
                        _mm512_storeu_si512(d as *mut _, v);
                    }
                }
            } else if offset >= 32 {
                // 32-byte AVX2 path
                let mut m = match_len;
                let mut s = match_src;
                let mut d = dst_ptr;
                while m >= 32 {
                    let v = _mm256_loadu_si256(s as *const __m256i);
                    _mm256_storeu_si256(d as *mut __m256i, v);
                    s = s.add(32);
                    d = d.add(32);
                    m -= 32;
                }
                if m > 0 {
                    let v = _mm256_loadu_si256(s as *const __m256i);
                    _mm256_storeu_si256(d as *mut __m256i, v);
                }
            } else if offset >= 16 {
                // 16-byte SSE path
                let mut m = match_len;
                let mut s = match_src;
                let mut d = dst_ptr;
                while m >= 16 {
                    let v = _mm_loadu_si128(s as *const __m128i);
                    _mm_storeu_si128(d as *mut __m128i, v);
                    s = s.add(16);
                    d = d.add(16);
                    m -= 16;
                }
                if m > 0 {
                    let v = _mm_loadu_si128(s as *const __m128i);
                    _mm_storeu_si128(d as *mut __m128i, v);
                }
            } else {
                // Small offset (1..15): Replicate using multi-chunk shuffle masks
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
                                let mut remaining = match_len - 64;
                                while remaining >= 32 {
                                    let v = _mm256_loadu_si256(d.sub(safe_dist) as *const __m256i);
                                    _mm256_storeu_si256(d as *mut __m256i, v);
                                    d = d.add(32);
                                    remaining -= 32;
                                }
                                if remaining > 0 {
                                    let v = _mm256_loadu_si256(d.sub(safe_dist) as *const __m256i);
                                    _mm256_storeu_si256(d as *mut __m256i, v);
                                }
                            }
                        }
                    }
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

    let mut lit_ptr = literals.as_ptr();
    let mut offset_idx = 0;

    for &token in tokens {
        let lit_len = token.lit_len();
        let match_len = token.match_len();

        // 1. Literal copy (up to 31 bytes)
        if lit_len > 0 {
            if lit_len <= 8 {
                std::ptr::copy_nonoverlapping(lit_ptr, dst_ptr, 8);
            } else {
                let v0 = _mm256_loadu_si256(lit_ptr as *const __m256i);
                _mm256_storeu_si256(dst_ptr as *mut __m256i, v0);
            }

            lit_ptr = lit_ptr.add(lit_len);
            dst_ptr = dst_ptr.add(lit_len);
        }

        // 2. Match copy (up to 2047 bytes)
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

            if offset >= 8 {
                std::ptr::copy_nonoverlapping(match_src, dst_ptr, 8);
                if match_len > 8 {
                    if offset >= 32 {
                        let mut m = match_len - 8;
                        let mut s = match_src.add(8);
                        let mut d = dst_ptr.add(8);
                        while m >= 32 {
                            let v = _mm256_loadu_si256(s as *const __m256i);
                            _mm256_storeu_si256(d as *mut __m256i, v);
                            s = s.add(32);
                            d = d.add(32);
                            m -= 32;
                        }
                        if m > 0 {
                            let v = _mm256_loadu_si256(s as *const __m256i);
                            _mm256_storeu_si256(d as *mut __m256i, v);
                        }
                    } else if offset >= 16 {
                        let mut m = match_len - 8;
                        let mut s = match_src.add(8);
                        let mut d = dst_ptr.add(8);
                        while m >= 16 {
                            let v = _mm_loadu_si128(s as *const __m128i);
                            _mm_storeu_si128(d as *mut __m128i, v);
                            s = s.add(16);
                            d = d.add(16);
                            m -= 16;
                        }
                        if m > 0 {
                            let v = _mm_loadu_si128(s as *const __m128i);
                            _mm_storeu_si128(d as *mut __m128i, v);
                        }
                    } else {
                        let mut m = match_len - 8;
                        let mut s = match_src.add(8);
                        let mut d = dst_ptr.add(8);
                        while m >= 8 {
                            std::ptr::copy_nonoverlapping(s, d, 8);
                            s = s.add(8);
                            d = d.add(8);
                            m -= 8;
                        }
                        if m > 0 {
                            std::ptr::copy_nonoverlapping(s, d, 8);
                        }
                    }
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
                                let mut remaining = match_len - 64;
                                while remaining >= 32 {
                                    let v = _mm256_loadu_si256(d.sub(safe_dist) as *const __m256i);
                                    _mm256_storeu_si256(d as *mut __m256i, v);
                                    d = d.add(32);
                                    remaining -= 32;
                                }
                                if remaining > 0 {
                                    let v = _mm256_loadu_si256(d.sub(safe_dist) as *const __m256i);
                                    _mm256_storeu_si256(d as *mut __m256i, v);
                                }
                            }
                        }
                    }
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
