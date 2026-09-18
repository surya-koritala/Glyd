//! NEON port of the AVX2 Adler-like checksum. All arithmetic is wrapping
//! u32, so lanes may wrap freely; the result matches the scalar definition
//! bit for bit.
use std::arch::aarch64::*;

pub unsafe fn checksum_neon(data: &[u8]) -> u32 {
    // With S_k the byte sum of 32-byte block k (k = 0..n) and W_k its
    // position-weighted sum, the scalar recurrence gives
    //   a = a0 + sum S_k
    //   b = b0 + 32 n a0 + 32 sum_k (n-1-k) S_k + sum W_k
    // and sum_k (n-1-k) S_k = (n-1) sum S_k - sum k S_k. Everything is a
    // lane-wise wrapping sum, so there is no scalar chain in the loop.
    let mut a: u32 = 1;
    let mut b: u32 = 0;
    let mut ptr = data.as_ptr();
    let mut remaining = data.len();

    let w0 = vld1_u8([32u8, 31, 30, 29, 28, 27, 26, 25].as_ptr());
    let w1 = vld1_u8([24u8, 23, 22, 21, 20, 19, 18, 17].as_ptr());
    let w2 = vld1_u8([16u8, 15, 14, 13, 12, 11, 10, 9].as_ptr());
    let w3 = vld1_u8([8u8, 7, 6, 5, 4, 3, 2, 1].as_ptr());
    // Four weighted accumulators: chained into one, the vpadal latency made
    // the loop 12 cycles per 32 bytes.
    let mut acc_w0 = vdupq_n_u32(0);
    let mut acc_w1 = vdupq_n_u32(0);
    let mut acc_w2 = vdupq_n_u32(0);
    let mut acc_w3 = vdupq_n_u32(0);
    let mut acc_s = vdupq_n_u32(0);
    let mut acc_ks = vdupq_n_u32(0);
    let mut k: u32 = 0;

    while remaining >= 32 {
        let c0 = vld1q_u8(ptr);
        let c1 = vld1q_u8(ptr.add(16));
        acc_w0 = vpadalq_u16(acc_w0, vmull_u8(vget_low_u8(c0), w0));
        acc_w1 = vpadalq_u16(acc_w1, vmull_u8(vget_high_u8(c0), w1));
        acc_w2 = vpadalq_u16(acc_w2, vmull_u8(vget_low_u8(c1), w2));
        acc_w3 = vpadalq_u16(acc_w3, vmull_u8(vget_high_u8(c1), w3));
        let s32 = vpaddlq_u16(vaddq_u16(vpaddlq_u8(c0), vpaddlq_u8(c1)));
        acc_s = vaddq_u32(acc_s, s32);
        acc_ks = vmlaq_n_u32(acc_ks, s32, k);
        k += 1;
        ptr = ptr.add(32);
        remaining -= 32;
    }
    if k > 0 {
        let sum_s = vaddvq_u32(acc_s);
        let sum_ks = vaddvq_u32(acc_ks);
        let cross = (k - 1).wrapping_mul(sum_s).wrapping_sub(sum_ks);
        b = b.wrapping_add(k.wrapping_mul(32).wrapping_mul(a))
            .wrapping_add(cross.wrapping_mul(32))
            .wrapping_add(vaddvq_u32(vaddq_u32(vaddq_u32(acc_w0, acc_w1), vaddq_u32(acc_w2, acc_w3))));
        a = a.wrapping_add(sum_s);
    }

    for &byte in std::slice::from_raw_parts(ptr, remaining) {
        a = a.wrapping_add(byte as u32);
        b = b.wrapping_add(a);
    }
    (b << 16) | (a & 0xFFFF)
}

#[cfg(test)]
mod tests {
    #[test]
    fn matches_scalar() {
        let mut x = 12345u64;
        for len in [0usize, 1, 31, 32, 33, 100, 4096, 65537, 1 << 20] {
            let v: Vec<u8> = (0..len).map(|_| { x ^= x << 13; x ^= x >> 7; x ^= x << 17; x as u8 }).collect();
            assert_eq!(unsafe { super::checksum_neon(&v) }, crate::format::compute_checksum_scalar(&v), "len {}", len);
        }
    }
}
