//! XXH64 as zstd's `xxhash.h` computes it: the frame checksum is the
//! low 32 bits of the content's hash with seed 0.

const P1: u64 = 0x9E3779B185EBCA87;
const P2: u64 = 0xC2B2AE3D27D4EB4F;
const P3: u64 = 0x165667B19E3779F9;
const P4: u64 = 0x85EBCA77C2B2AE63;
const P5: u64 = 0x27D4EB2F165667C5;

#[inline]
fn round(acc: u64, input: u64) -> u64 {
    acc.wrapping_add(input.wrapping_mul(P2)).rotate_left(31).wrapping_mul(P1)
}

#[inline]
fn merge_round(acc: u64, val: u64) -> u64 {
    (acc ^ round(0, val)).wrapping_mul(P1).wrapping_add(P4)
}

#[inline]
fn read64(s: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(s[at..at + 8].try_into().unwrap())
}

/// `XXH64(input, len, 0)`.
pub fn xxh64(input: &[u8]) -> u64 {
    let n = input.len();
    let mut p = 0usize;
    let mut h = if n >= 32 {
        let (mut v1, mut v2, mut v3, mut v4) = (P1.wrapping_add(P2), P2, 0u64, 0u64.wrapping_sub(P1));
        while p + 32 <= n {
            v1 = round(v1, read64(input, p));
            v2 = round(v2, read64(input, p + 8));
            v3 = round(v3, read64(input, p + 16));
            v4 = round(v4, read64(input, p + 24));
            p += 32;
        }
        let h = v1.rotate_left(1).wrapping_add(v2.rotate_left(7)).wrapping_add(v3.rotate_left(12)).wrapping_add(v4.rotate_left(18));
        let h = merge_round(h, v1);
        let h = merge_round(h, v2);
        let h = merge_round(h, v3);
        merge_round(h, v4)
    } else {
        P5
    };
    h = h.wrapping_add(n as u64);
    // XXH64_finalize.
    while p + 8 <= n {
        h ^= round(0, read64(input, p));
        h = h.rotate_left(27).wrapping_mul(P1).wrapping_add(P4);
        p += 8;
    }
    if p + 4 <= n {
        h ^= (u32::from_le_bytes(input[p..p + 4].try_into().unwrap()) as u64).wrapping_mul(P1);
        h = h.rotate_left(23).wrapping_mul(P2).wrapping_add(P3);
        p += 4;
    }
    for &b in &input[p..] {
        h ^= (b as u64).wrapping_mul(P5);
        h = h.rotate_left(11).wrapping_mul(P1);
    }
    // XXH64_avalanche.
    h ^= h >> 33;
    h = h.wrapping_mul(P2);
    h ^= h >> 29;
    h = h.wrapping_mul(P3);
    h ^ (h >> 32)
}

#[cfg(test)]
mod tests {
    use super::xxh64;

    #[test]
    fn known_values() {
        // Reference values of XXH64 with seed 0.
        assert_eq!(xxh64(b""), 0xEF46DB3751D8E999);
        assert_eq!(xxh64(b"a"), 0xD24EC4F1A98C6E5B);
        assert_eq!(xxh64(b"abc"), 0x44BC2CF5AD770999);
        let s: Vec<u8> = (0..100u8).collect();
        assert_eq!(xxh64(&s), 0x6AC1E58032166597);
    }
}
