//! Base-2 logarithms in fixed point, by integer arithmetic only, so the
//! encoders' cost estimates (and with them their decisions and output)
//! are the same on every platform; `f64::log2` is the platform's libm.

/// log2(v) with 16 fractional bits, `v >= 1`: the integer part from the
/// bit length, then 16 fraction bits by squaring the mantissa.
pub const fn log2_q16(v: u64) -> u64 {
    debug_assert!(v >= 1);
    let k = 63 - v.leading_zeros();
    // The mantissa in [1, 2) as a Q62 fixed-point number (a 64-bit value
    // drops its lowest bit).
    let mut m = if k <= 62 { (v as u128) << (62 - k) } else { (v as u128) >> (k - 62) };
    let mut frac = 0u64;
    let mut i = 0;
    while i < 16 {
        m = (m * m) >> 62;
        frac <<= 1;
        if m >= 1u128 << 63 {
            frac |= 1;
            m >>= 1;
        }
        i += 1;
    }
    (k as u64) << 16 | frac
}

/// The cost, in 1/65536 bit, of one symbol of count `c` in a total of
/// `t`: log2(t) - log2(c).
#[inline]
pub const fn cost_q16(c: u64, t: u64) -> u64 {
    log2_q16(t).saturating_sub(log2_q16(c))
}

/// log2(v) with 16 fractional bits from an 8-bit mantissa table, for
/// decisions made many thousand times a block (the block splitter):
/// within 0.006 bit of `log2_q16`, at a lookup instead of sixteen
/// 128-bit squarings. `v >= 1`.
#[inline]
pub fn log2_fast_q16(v: u32) -> u64 {
    static FRAC: [u16; 256] = {
        let mut t = [0u16; 256];
        let mut i = 0;
        while i < 256 {
            // log2(1 + i/256) in Q16, from the exact routine.
            t[i] = (log2_q16(256 + i as u64) - (8 << 16)) as u16;
            i += 1;
        }
        t
    };
    debug_assert!(v >= 1);
    let k = 31 - v.leading_zeros();
    let mant = ((v << (31 - k)) >> 23) as usize & 0xff;
    ((k as u64) << 16) + FRAC[mant] as u64
}

/// `cost_q16(c, L)` for every tANS count `c` in `1..=L`, from a table:
/// the count tables are priced symbol by symbol on every block, and
/// a small object's block is mostly that pricing (`log2_q16` squares a
/// 128-bit mantissa sixteen times).
pub fn tans_cost_q16(c: u16) -> u64 {
    const L: usize = crate::tans::L;
    static COST: [u32; L + 1] = {
        let mut t = [0u32; L + 1];
        let mut c = 1;
        while c <= L {
            t[c] = cost_q16(c as u64, L as u64) as u32;
            c += 1;
        }
        t
    };
    COST[c as usize] as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fast_log_within_a_hundredth_of_a_bit() {
        for v in [1u32, 2, 3, 5, 7, 100, 255, 256, 257, 1000, 4095, 4096, 30000, u32::MAX] {
            let got = log2_fast_q16(v) as f64 / 65536.0;
            let want = (v as f64).log2();
            assert!((got - want).abs() < 0.01, "log2({v}) = {want}, table {got}");
        }
    }

    #[test]
    fn matches_f64_to_a_few_ten_thousandths() {
        for v in [1u64, 2, 3, 7, 10, 100, 1023, 1024, 1025, 65535, 1 << 20, (1 << 40) + 12345] {
            let got = log2_q16(v) as f64 / 65536.0;
            let want = (v as f64).log2();
            assert!((got - want).abs() < 2e-4, "log2({v}): {got} vs {want}");
        }
    }
}

#[cfg(test)]
mod more_tests {
    use super::*;
    #[test]
    fn exhaustive_small_and_random_large() {
        let mut worst = 0f64;
        for v in 1u64..=1 << 16 {
            let got = log2_q16(v) as f64 / 65536.0;
            let want = (v as f64).log2();
            if (got - want).abs() > 0.5 {
                panic!("log2({v}) = {got}, want {want}");
            }
            worst = worst.max((got - want).abs());
        }
        let mut x = 0x9E37_79B9_7F4A_7C15u64;
        for _ in 0..100_000 {
            x ^= x << 13; x ^= x >> 7; x ^= x << 17;
            let v = x >> (x % 40);
            let got = log2_q16(v.max(1)) as f64 / 65536.0;
            let want = (v.max(1) as f64).log2();
            worst = worst.max((got - want).abs());
        }
        assert!(worst < 2e-4, "worst error {worst}");
    }
}
