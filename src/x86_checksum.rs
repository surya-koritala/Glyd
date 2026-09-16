//! AVX2 SIMD-accelerated 32-bit Adler-like checksum engine.
//! Processes 32 bytes per cycle using _mm256_maddubs_epi16 and _mm256_sad_epu8,
//! matching the scalar definition bit-for-bit.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn checksum_avx2(data: &[u8]) -> u32 {
    let mut a: u32 = 1;
    let mut b: u32 = 0;

    let len = data.len();
    let mut ptr = data.as_ptr();
    let mut remaining = len;

    #[rustfmt::skip]
    let weights = _mm256_setr_epi8(
        32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17,
        16, 15, 14, 13, 12, 11, 10,  9,  8,  7,  6,  5,  4,  3,  2,  1,
    );
    let ones_16 = _mm256_set1_epi16(1);
    let zero = _mm256_setzero_si256();

    while remaining >= 32 {
        let chunk = _mm256_loadu_si256(ptr as *const __m256i);

        let sad = _mm256_sad_epu8(chunk, zero);
        let sum0 = _mm256_extract_epi64(sad, 0) as u32;
        let sum1 = _mm256_extract_epi64(sad, 1) as u32;
        let sum2 = _mm256_extract_epi64(sad, 2) as u32;
        let sum3 = _mm256_extract_epi64(sad, 3) as u32;
        let chunk_sum = sum0.wrapping_add(sum1).wrapping_add(sum2).wrapping_add(sum3);

        let madd = _mm256_maddubs_epi16(chunk, weights);
        let dot = _mm256_madd_epi16(madd, ones_16);
        let d0 = _mm256_extract_epi32(dot, 0) as u32;
        let d1 = _mm256_extract_epi32(dot, 1) as u32;
        let d2 = _mm256_extract_epi32(dot, 2) as u32;
        let d3 = _mm256_extract_epi32(dot, 3) as u32;
        let d4 = _mm256_extract_epi32(dot, 4) as u32;
        let d5 = _mm256_extract_epi32(dot, 5) as u32;
        let d6 = _mm256_extract_epi32(dot, 6) as u32;
        let d7 = _mm256_extract_epi32(dot, 7) as u32;
        let weighted_sum = d0.wrapping_add(d1).wrapping_add(d2).wrapping_add(d3)
            .wrapping_add(d4).wrapping_add(d5).wrapping_add(d6).wrapping_add(d7);

        b = b.wrapping_add(a.wrapping_mul(32)).wrapping_add(weighted_sum);
        a = a.wrapping_add(chunk_sum);

        ptr = ptr.add(32);
        remaining -= 32;
    }

    let tail = std::slice::from_raw_parts(ptr, remaining);
    for &byte in tail {
        a = a.wrapping_add(byte as u32);
        b = b.wrapping_add(a);
    }

    (b << 16) | (a & 0xFFFF)
}
