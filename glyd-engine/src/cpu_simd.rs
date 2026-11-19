//! The kernels of src/cpu_experts.rs: a row layout's runs decoded and multiplied into 16 lane sums, once for each instruction set (`Isa`) and written once: the scalar one, NEON, AVX2 with FMA and AVX-512. A run's 32 codes
//! come out as two vectors of 16 lanes, the low-nibble plane and the high-nibble plane of the run's 16 bytes (`LANE_LO` and `LANE_HI` name the weights), already minus the code bias; the first product takes the run's two planes
//! against z in the same order (cpu_experts.rs puts z there once for a token), the second adds a row to the 1,024 outputs in the same order. Every operation is the same in every instruction set (a multiply, a fused
//! multiply-add, a fixed tree of adds), so the kernels agree to the bit.
//!
//! The codebook rows of src/cpu_hb.rs (`up_cb`, `down_cb`) use the same instruction sets: a row's table of 4 or 8 entries is held in the instruction set's form (`Isa::table`), and a run's codes are looked up in it 16 at
//! a time, in natural order (`Isa::decode_cb`: AVX2 `vpermps` on a table of 8, AVX-512 `vpermps` on a table of 16, NEON `tbl` on the table's bf16 bytes). The arithmetic after the lookup is the same in all four,
//! so these kernels agree to the bit too.

use crate::cpu_experts::{bf16_to_f32, Kernel, RowsView, LANE_HI, LANE_LO};
use crate::cpu_hb::{strides, CbView};
#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

/// 16 lanes of fp32 and what the kernels do with them.
pub(crate) trait Isa {
    type F: Copy;
    unsafe fn zero() -> Self::F;
    unsafe fn splat(x: f32) -> Self::F;
    unsafe fn load(p: *const f32) -> Self::F;
    unsafe fn store(p: *mut f32, v: Self::F);
    unsafe fn mul(a: Self::F, b: Self::F) -> Self::F;
    /// a + b.
    unsafe fn add(a: Self::F, b: Self::F) -> Self::F;
    /// a b + c, rounded once.
    unsafe fn fma(a: Self::F, b: Self::F, c: Self::F) -> Self::F;
    /// The lanes added as a tree: l with l + 8, then l + 4, l + 2, l + 1.
    unsafe fn hsum(a: Self::F) -> f32;
    /// A run's codes minus the bias (2^(B - 1)), as fp32: the low-nibble plane, then the high-nibble plane. `lo` is the run's 16 bytes, `hi` its bytes of high bits (not read at four bits).
    unsafe fn decode<const B: u32>(lo: *const u8, hi: *const u8) -> (Self::F, Self::F);
    /// The cache line of `p` asked for.
    unsafe fn prefetch(p: *const u8);
    /// A codebook row's table in this instruction set's form (`cb`: the row's 2^B bf16 entries, little-endian; nothing past them is read).
    type T: Copy;
    unsafe fn table<const B: u32>(cb: *const u8) -> Self::T;
    /// A codebook run's 32 codes (`run`: its 4 B bytes, nothing past them is read) looked up in `t`, as fp32: the codes 0 to 15, then 16 to 31. B is 2 to 4 (a table of up to 16 entries).
    unsafe fn decode_cb<const B: u32>(t: Self::T, run: *const u8) -> (Self::F, Self::F);
    /// A ternary row's table, three bf16 entries (6 bytes: nothing past them is read), in this instruction set's form.
    unsafe fn table_tern(cb: *const u8) -> Self::T;
    /// 32 bytes of ternary codes (`bytes`: five codes a byte, base 3, the first the least significant digit) looked up in `t`: for each of the five digits, the weights of bytes 0 to 15 and of bytes 16 to 31.
    unsafe fn decode_tern(t: Self::T, bytes: *const u8) -> [(Self::F, Self::F); 5];
    /// A trellis row's 16 weights from its words j (`hi`) and j + 1 (`lo`): lane i `tq_lane(hi, lo, i)`, integers as fp32.
    unsafe fn decode_tq(hi: u32, lo: u32) -> Self::F;
    /// 16 weights of a format q row of K bits a weight (q3, q4), the first's state at bit R of the word at x (`tqk_words`: x and the next three words readable): lane i `tqk_lane::<K>(x, R, i)`,
    /// integers as fp32.
    #[inline(always)]
    unsafe fn decode_tqk<const K: usize, const R: u32>(x: *const u32) -> Self::F {
        let w: [u32; 4] = core::array::from_fn(|q| u32::from_le(x.add(q).read_unaligned()));
        let v: [f32; 16] = core::array::from_fn(|i| tqk_lane::<K>(w, R, i));
        Self::load(v.as_ptr())
    }
    /// Two runs of 32 weights from the word at x (its bit 0 the first state's; K words and the next four readable): `decode_tqk` of each.
    #[inline(always)]
    unsafe fn decode_tqk_pair<const K: usize>(x: *const u32) -> (Self::F, Self::F) {
        (Self::decode_tqk::<K, 0>(x), if K == 3 { Self::decode_tqk::<K, 16>(x.add(1)) } else { Self::decode_tqk::<K, 0>(x.add(2)) })
    }
}

/// The trellis rows' value of a state s (14 bits): K (bytesum(s TQ_A + TQ_B) - 510), the four bytes' sum centred and of unit variance (src/qx_trellis.rs and kernels/qwen4x.cu, qx_tq_*, encode with them).
pub(crate) const TQ_A: u32 = 1_536_988_885;
pub(crate) const TQ_B: u32 = 2_891_336_453;
/// 1 / the standard deviation of the sum of four uniform bytes, sqrt(4 (256^2 - 1) / 12).
pub(crate) const TQ_K: f32 = 0.006_765_867_2;

/// Weight 16 j + i of a trellis row before its scale and K, from its words j (`hi`) and j + 1 (`lo`): the 14 bits of hi:lo from bit 2 i (most significant first) as the state, bytesum(s TQ_A + TQ_B) - 510.
#[inline(always)]
pub(crate) fn tq_lane(hi: u32, lo: u32, i: usize) -> f32 {
    let s = ((((hi as u64) << 32 | lo as u64) << (2 * i)) >> 50) as u32;
    tq_value(s)
}

/// A state's value before its row's scale and K: bytesum(s TQ_A + TQ_B) - 510.
#[inline(always)]
pub(crate) fn tq_value(s: u32) -> f32 {
    let x = s.wrapping_mul(TQ_A).wrapping_add(TQ_B);
    ((x & 255) + ((x >> 8) & 255) + ((x >> 16) & 255) + (x >> 24)) as f32 - 510.0
}

/// The state of weight t of a format q row of `k` bits a weight (`words`: the row's `k cols / 32` little-endian words, a circular stream, most significant bit first): its 14 bits from bit k t
/// (q, q3, q4: k 2, 3, 4), the value as `tq_value`.
#[inline(always)]
pub(crate) fn tq_state_k(words: &[u32], k: usize, t: usize) -> u32 {
    let (p, w) = (k * t, words.len());
    let (hi, lo) = (words[p / 32], words[(p / 32 + 1) % w]);
    ((((hi as u64) << 32 | lo as u64) << (p % 32)) >> 50) as u32
}

/// Lane i of 16 weights of a format q row of K bits a weight whose first state starts at bit r of the words x (`tqk_group`): its value before the row's scale and TQ_K.
#[inline(always)]
pub(crate) fn tqk_lane<const K: usize>(x: [u32; 4], r: u32, i: usize) -> f32 {
    let p = r as usize + K * i;
    tq_value(((((x[p / 32] as u64) << 32 | x[p / 32 + 1] as u64) << (p % 32)) >> 50) as u32)
}

/// The words of run g (weights 16 g to 16 g + 15) of row `row` of a format q matrix of K bits a weight (`w` words a row), from the one its first bit is in (at bit 16 K g % 32 of it): the row's own
/// where four are there, else four (around the row's end) copied to `buf`.
#[inline(always)]
unsafe fn tqk_words<const K: usize>(codes: *const u8, w: usize, row: usize, g: usize, buf: &mut [u32; 4]) -> *const u32 {
    let (j, base) = (16 * K * g / 32, codes.add(4 * row * w).cast::<u32>());
    if j + 4 <= w {
        return base.add(j);
    }
    for (q, b) in buf.iter_mut().enumerate() {
        *b = base.add((j + q) % w).read_unaligned();
    }
    buf.as_ptr()
}

/// Runs g and g + 1 (g even) of row `row`: their weights (`decode_tqk`), the first's state at bit 0 of its word, the second's at bit 16 at 3 bits a weight, 0 at 4 (`buf`: the words of a run
/// around the row's end, the caller's, so that the loop does not set it each time).
#[inline(always)]
unsafe fn tqk_runs<I: Isa, const K: usize>(codes: *const u8, w: usize, row: usize, g: usize, buf: &mut [[u32; 4]; 2]) -> (I::F, I::F) {
    let [b0, b1] = buf;
    let (p0, p1) = (tqk_words::<K>(codes, w, row, g, b0), tqk_words::<K>(codes, w, row, g + 1, b1));
    (I::decode_tqk::<K, 0>(p0), if K == 3 { I::decode_tqk::<K, 16>(p1) } else { I::decode_tqk::<K, 0>(p1) })
}

/// Runs g and g + 1 (g even) of row `row`, as `tqk_runs`: straight from the row's words but for the row's last pair (`runs`: the row's runs), whose second run's words go around its end.
#[inline(always)]
unsafe fn tqk_pair<I: Isa, const K: usize>(codes: *const u8, w: usize, row: usize, g: usize, runs: usize, buf: &mut [[u32; 4]; 2]) -> (I::F, I::F) {
    if g + 2 < runs {
        I::decode_tqk_pair::<K>(codes.cast::<u32>().add(row * w + K * g / 2))
    } else {
        tqk_runs::<I, K>(codes, w, row, g, buf)
    }
}

/// For each lane of a run of a format q row of K bits a weight whose first state starts at bit r: the word its state starts in, and the bit there (`decode_tqk`'s gathers and shifts).
#[cfg(target_arch = "x86_64")]
const fn tqk_tab(k: usize, r: usize) -> [[i32; 16]; 2] {
    let mut t = [[0i32; 16]; 2];
    let mut i = 0;
    while i < 16 {
        t[0][i] = ((r + k * i) / 32) as i32;
        t[1][i] = ((r + k * i) % 32) as i32;
        i += 1;
    }
    t
}
#[cfg(target_arch = "x86_64")]
const TQK_TABS: [[[i32; 16]; 2]; 3] = [tqk_tab(3, 0), tqk_tab(3, 16), tqk_tab(4, 0)];

/// The table of `tqk_tab` for K and R.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn tqk_tabs<const K: usize, const R: u32>() -> &'static [[i32; 16]; 2] {
    &TQK_TABS[if K == 4 { 2 } else { (R != 0) as usize }]
}

/// AVX-512 with VBMI2: a run's 16 windows (`decode_tqk`'s states before the shift right) from its three words broadcast (b0, b1, b2) as two funnel shifts, the first and second word's and the
/// second and third's, blended where a lane's state starts in the second word: no lane permutes (two of them a run cost as much as the rest of the decode on Zen 4).
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn tqk_win512v<const K: usize, const R: u32>(b0: __m512i, b1: __m512i, b2: __m512i) -> __m512i {
    let t = tqk_tabs::<K, R>();
    // (the select by a mask vector, all ones in the lanes whose state starts in the second word: no mask register to load each time)
    let m = _mm512_sub_epi32(_mm512_setzero_si512(), _mm512_loadu_si512(t[0].as_ptr().cast()));
    let sh = _mm512_loadu_si512(t[1].as_ptr().cast());
    _mm512_ternarylogic_epi32::<0xCA>(m, _mm512_shldv_epi32(b1, b2, sh), _mm512_shldv_epi32(b0, b1, sh))
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn tqk_windows512v<const K: usize, const R: u32>(x: *const u32) -> __m512i {
    let b = |q: usize| _mm512_set1_epi32(x.add(q).read_unaligned() as i32);
    tqk_win512v::<K, R>(b(0), b(1), b(2))
}

/// AVX-512 with VNNI: 16 states' values (before the row's scale and TQ_K) from their windows.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn tq_values512v(w: __m512i) -> __m512 {
    let x = _mm512_add_epi32(_mm512_mullo_epi32(_mm512_srli_epi32::<18>(w), _mm512_set1_epi32(TQ_A as i32)), _mm512_set1_epi32(TQ_B as i32));
    _mm512_cvtepi32_ps(_mm512_dpbusd_epi32(_mm512_set1_epi32(-510), x, _mm512_set1_epi8(1)))
}

/// The same for AVX-512 without VBMI2: `hi` and `lo` of each lane by blends, and the shifts.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn tqk_windows512<const K: usize, const R: u32>(x: *const u32) -> (__m512i, __m512i, __m512i) {
    let t = tqk_tabs::<K, R>();
    let m: __mmask16 = (0..16).fold(0u16, |m, i| m | (((R as usize + K * i >= 32) as u16) << i));
    let (b0, b1, b2) = (_mm512_set1_epi32(x.read_unaligned() as i32), _mm512_set1_epi32(x.add(1).read_unaligned() as i32), _mm512_set1_epi32(x.add(2).read_unaligned() as i32));
    (_mm512_mask_blend_epi32(m, b0, b1), _mm512_mask_blend_epi32(m, b1, b2), _mm512_loadu_si512(t[1].as_ptr().cast()))
}

/// `R` rows at a time of a format q matrix of K bits a weight (q3, q4), as `dot_rows_tq`: per row the even and the odd runs of 16 weights against z (natural order) into two vectors of lane sums, a fused
/// multiply-add per run, added as the two trees of lanes, times the row's scale, times `TQ_K`, into `sums[0..r1 - r0]`.
#[inline(always)]
unsafe fn dot_rows_tqk<I: Isa, const R: usize, const K: usize>(m: &GlmMat, z: *const f32, r0: usize, r1: usize, sums: *mut f32) {
    let (w, runs) = (K * m.cols / 32, m.cols / 16);
    let (codes, book) = (m.codes.as_ptr(), m.book.as_ptr());
    let mut buf = [[0u32; 4]; 2];
    let mut r = r0;
    while r < r1 {
        let n = R.min(r1 - r);
        let (mut a0, mut a1) = ([I::zero(); R], [I::zero(); R]);
        for g in (0..runs).step_by(2) {
            let (z0, z1) = (I::load(z.add(16 * g)), I::load(z.add(16 * g + 16)));
            for j in 0..n {
                let (v0, v1) = tqk_pair::<I, K>(codes, w, r + j, g, runs, &mut buf);
                a0[j] = I::fma(v0, z0, a0[j]);
                a1[j] = I::fma(v1, z1, a1[j]);
            }
        }
        for j in 0..n {
            *sums.add(r + j - r0) = (I::hsum(a0[j]) + I::hsum(a1[j])) * tq_scale(book, r + j) * TQ_K;
        }
        r += n;
    }
}

/// `dot_rows_tqk` for `T` tokens at once, a row at a time: each run decoded once, each token's sums the same multiply-adds in the same order as alone.
#[inline(always)]
unsafe fn dot_rows_tqk_t<I: Isa, const K: usize, const T: usize>(m: &GlmMat, zs: [*const f32; T], r0: usize, r1: usize, sums: [*mut f32; T]) {
    let (w, runs) = (K * m.cols / 32, m.cols / 16);
    let (codes, book) = (m.codes.as_ptr(), m.book.as_ptr());
    let mut buf = [[0u32; 4]; 2];
    for r in r0..r1 {
        let (mut a0, mut a1) = ([I::zero(); T], [I::zero(); T]);
        for g in (0..runs).step_by(2) {
            let (v0, v1) = tqk_pair::<I, K>(codes, w, r, g, runs, &mut buf);
            for t in 0..T {
                a0[t] = I::fma(v0, I::load(zs[t].add(16 * g)), a0[t]);
                a1[t] = I::fma(v1, I::load(zs[t].add(16 * g + 16)), a1[t]);
            }
        }
        for t in 0..T {
            *sums[t].add(r - r0) = (I::hsum(a0[t]) + I::hsum(a1[t])) * tq_scale(book, r) * TQ_K;
        }
    }
}

/// Outputs `o0` to `o0 + 16 N` (o0 a multiple of 16) of the sum over the rows n of a format q matrix of K bits a weight of `act[n]` times row n, as `down_cols_tq`.
#[inline(always)]
unsafe fn down_cols_tqk<I: Isa, const N: usize, const K: usize>(m: &GlmMat, act: *const f32, o0: usize, out: *mut f32) {
    const AHEAD: usize = 8;
    let (w, g0, runs) = (K * m.cols / 32, o0 / 16, m.cols / 16);
    let (codes, book) = (m.codes.as_ptr(), m.book.as_ptr());
    let mut buf = [[0u32; 4]; 2];
    let mut acc = [I::zero(); N];
    for n in 0..m.rows {
        if n + AHEAD < m.rows {
            I::prefetch(codes.add(4 * ((n + AHEAD) * w + K * g0 / 2)));
        }
        let a = I::splat(*act.add(n) * tq_scale(book, n) * TQ_K);
        for k in (0..N).step_by(2) {
            let (v0, v1) = tqk_pair::<I, K>(codes, w, n, g0 + k, runs, &mut buf);
            acc[k] = I::fma(v0, a, acc[k]);
            acc[k + 1] = I::fma(v1, a, acc[k + 1]);
        }
    }
    for (k, v) in acc.into_iter().enumerate() {
        I::store(out.add(16 * k), v);
    }
}

/// `down_cols_tqk` for `T` tokens at once (`acts`): each run decoded once, each token's accumulators the same multiply-adds in the same order as alone.
#[inline(always)]
unsafe fn down_cols_tqk_t<I: Isa, const N: usize, const K: usize, const T: usize>(m: &GlmMat, acts: [*const f32; T], o0: usize, outs: [*mut f32; T]) {
    let (w, g0, runs) = (K * m.cols / 32, o0 / 16, m.cols / 16);
    let (codes, book) = (m.codes.as_ptr(), m.book.as_ptr());
    let mut buf = [[0u32; 4]; 2];
    let mut acc = [[I::zero(); N]; T];
    for n in 0..m.rows {
        let a: [I::F; T] = core::array::from_fn(|t| I::splat(*acts[t].add(n) * tq_scale(book, n) * TQ_K));
        for k in (0..N).step_by(2) {
            let (v0, v1) = tqk_pair::<I, K>(codes, w, n, g0 + k, runs, &mut buf);
            for t in 0..T {
                acc[t][k] = I::fma(v0, a[t], acc[t][k]);
                acc[t][k + 1] = I::fma(v1, a[t], acc[t][k + 1]);
            }
        }
    }
    for t in 0..T {
        for (k, v) in acc[t].into_iter().enumerate() {
            I::store(outs[t].add(16 * k), v);
        }
    }
}

/// Word j of row `row` of a trellis matrix of `w` words a row (j = w: word 0, around the end).
#[inline(always)]
unsafe fn tq_word(codes: *const u8, w: usize, row: usize, j: usize) -> u32 {
    u32::from_le(codes.add(4 * (row * w + if j == w { 0 } else { j })).cast::<u32>().read_unaligned())
}

/// Row `row`'s scale (its bf16 table entry).
#[inline(always)]
unsafe fn tq_scale(book: *const u8, row: usize) -> f32 {
    f32::from_bits((u16::from_le(book.add(2 * row).cast::<u16>().read_unaligned()) as u32) << 16)
}

/// `R` rows at a time of a trellis matrix (`m.cols` a multiple of 256): per row the even and the odd runs of 16 weights against z (natural order) into two vectors of lane sums, a fused multiply-add per run, added as
/// the two trees of lanes, times the row's scale, times `TQ_K`, into `sums[0..r1 - r0]`.
#[inline(always)]
unsafe fn dot_rows_tq<I: Isa, const R: usize>(m: &GlmMat, z: *const f32, r0: usize, r1: usize, sums: *mut f32) {
    let w = m.cols / 16;
    let (codes, book) = (m.codes.as_ptr(), m.book.as_ptr());
    let mut r = r0;
    while r < r1 {
        let n = R.min(r1 - r);
        let (mut a0, mut a1) = ([I::zero(); R], [I::zero(); R]);
        for k in (0..w).step_by(2) {
            let (z0, z1) = (I::load(z.add(16 * k)), I::load(z.add(16 * k + 16)));
            for j in 0..n {
                let (w0, w1, w2) = (tq_word(codes, w, r + j, k), tq_word(codes, w, r + j, k + 1), tq_word(codes, w, r + j, k + 2));
                a0[j] = I::fma(I::decode_tq(w0, w1), z0, a0[j]);
                a1[j] = I::fma(I::decode_tq(w1, w2), z1, a1[j]);
            }
        }
        for j in 0..n {
            *sums.add(r + j - r0) = (I::hsum(a0[j]) + I::hsum(a1[j])) * tq_scale(book, r + j) * TQ_K;
        }
        r += n;
    }
}

/// `dot_rows_tq` for `T` tokens at once, a row at a time: each run of 16 weights decoded once and multiplied by every token's values, each token's sums the same multiply-adds in the same order as `dot_rows_tq`
/// makes them alone.
#[inline(always)]
unsafe fn dot_rows_tq_t<I: Isa, const T: usize>(m: &GlmMat, zs: [*const f32; T], r0: usize, r1: usize, sums: [*mut f32; T]) {
    let w = m.cols / 16;
    let (codes, book) = (m.codes.as_ptr(), m.book.as_ptr());
    for r in r0..r1 {
        let (mut a0, mut a1) = ([I::zero(); T], [I::zero(); T]);
        for k in (0..w).step_by(2) {
            let (w0, w1, w2) = (tq_word(codes, w, r, k), tq_word(codes, w, r, k + 1), tq_word(codes, w, r, k + 2));
            let (v0, v1) = (I::decode_tq(w0, w1), I::decode_tq(w1, w2));
            for t in 0..T {
                a0[t] = I::fma(v0, I::load(zs[t].add(16 * k)), a0[t]);
                a1[t] = I::fma(v1, I::load(zs[t].add(16 * k + 16)), a1[t]);
            }
        }
        for t in 0..T {
            *sums[t].add(r - r0) = (I::hsum(a0[t]) + I::hsum(a1[t])) * tq_scale(book, r) * TQ_K;
        }
    }
}

/// `down_cols_tq` for `T` tokens at once (`acts`), outputs `o0` to `o0 + 16 N`: each run decoded once, each token's accumulators the same multiply-adds in the same order as alone.
#[inline(always)]
unsafe fn down_cols_tq_t<I: Isa, const N: usize, const T: usize>(m: &GlmMat, acts: [*const f32; T], o0: usize, outs: [*mut f32; T]) {
    let (w, j0) = (m.cols / 16, o0 / 16);
    let (codes, book) = (m.codes.as_ptr(), m.book.as_ptr());
    let mut acc = [[I::zero(); N]; T];
    for n in 0..m.rows {
        let a: [I::F; T] = core::array::from_fn(|t| I::splat(*acts[t].add(n) * tq_scale(book, n) * TQ_K));
        for k in 0..N {
            let v = I::decode_tq(tq_word(codes, w, n, j0 + k), tq_word(codes, w, n, j0 + k + 1));
            for t in 0..T {
                acc[t][k] = I::fma(v, a[t], acc[t][k]);
            }
        }
    }
    for t in 0..T {
        for (k, v) in acc[t].into_iter().enumerate() {
            I::store(outs[t].add(16 * k), v);
        }
    }
}

/// Outputs `o0` to `o0 + 16 N` (o0 a multiple of 16) of the sum over the rows n of a trellis matrix of `act[n]` times row n: N vectors of accumulators over the rows in order, each run of 16 weights times act[n] times the
/// row's scale times `TQ_K` in a fused multiply-add, into `out[0..16 N]`.
#[inline(always)]
unsafe fn down_cols_tq<I: Isa, const N: usize>(m: &GlmMat, act: *const f32, o0: usize, out: *mut f32) {
    const AHEAD: usize = 8;
    let (w, j0) = (m.cols / 16, o0 / 16);
    let (codes, book) = (m.codes.as_ptr(), m.book.as_ptr());
    let mut acc = [I::zero(); N];
    for n in 0..m.rows {
        if n + AHEAD < m.rows {
            I::prefetch(codes.add(4 * ((n + AHEAD) * w + j0)));
        }
        let a = I::splat(*act.add(n) * tq_scale(book, n) * TQ_K);
        for (k, acc) in acc.iter_mut().enumerate() {
            *acc = I::fma(I::decode_tq(tq_word(codes, w, n, j0 + k), tq_word(codes, w, n, j0 + k + 1)), a, *acc);
        }
    }
    for (k, v) in acc.into_iter().enumerate() {
        I::store(out.add(16 * k), v);
    }
}

/// Digit `d` (0 to 4) of a ternary byte, as the pack writes them: byte = c0 + 3 c1 + 9 c2 + 27 c3 + 81 c4. The division by three is a multiplication and a shift, exact below 256: (x * 171) >> 9.
#[inline(always)]
pub(crate) fn tern_digit(b: u8, d: usize) -> usize {
    let mut x = b as usize;
    for _ in 0..d {
        x = (x * 171) >> 9;
    }
    x - 3 * ((x * 171) >> 9)
}

// ---- where a weight's high bits are ---------------------------------------------------------------------------------------------------------------------

/// Bit of a run's fifth-bit word that holds weight i's, and the bit offset of weight i's two high bits in a run's 8 bytes.
const fn pos5(i: usize) -> usize {
    i / 2 + 16 * (i % 2)
}

const fn pos6(i: usize) -> usize {
    32 * (i / 16) + 2 * ((i / 2) % 8) + 16 * (i % 2)
}

/// For the 16 lanes of a plane (`plane`: `LANE_LO` or `LANE_HI`): the byte of the run's high bits that lane p's are in, and the bit (five bits: a mask) or the shift (six) in that byte.
const fn high_byte(plane: &[usize; 16], six: bool) -> [u8; 16] {
    let mut o = [0u8; 16];
    let mut p = 0;
    while p < 16 {
        o[p] = ((if six { pos6(plane[p]) } else { pos5(plane[p]) }) / 8) as u8;
        p += 1;
    }
    o
}

const fn high_mask5(plane: &[usize; 16]) -> [u8; 16] {
    let mut o = [0u8; 16];
    let mut p = 0;
    while p < 16 {
        o[p] = 1 << (pos5(plane[p]) % 8);
        p += 1;
    }
    o
}

const fn high_shift6(plane: &[usize; 16]) -> [u8; 16] {
    let mut o = [0u8; 16];
    let mut p = 0;
    while p < 16 {
        o[p] = (pos6(plane[p]) % 8) as u8;
        p += 1;
    }
    o
}

const IDX5: [u8; 16] = high_byte(&LANE_LO, false);
const MASK5_LO: [u8; 16] = high_mask5(&LANE_LO);
const MASK5_HI: [u8; 16] = high_mask5(&LANE_HI);
const IDX6: [u8; 16] = high_byte(&LANE_LO, true);
const SHIFT6_LO: [u8; 16] = high_shift6(&LANE_LO);
const SHIFT6_HI: [u8; 16] = high_shift6(&LANE_HI);
/// Six bits, for the instruction sets that have no shift by lane: 0xFF where the lane's shift is the plane's first plus four (the low plane's are 0 or 4, the high plane's 2 or 6).
#[cfg(target_arch = "x86_64")]
const SEL6: [u8; 16] = {
    let (mut o, mut p) = ([0u8; 16], 0);
    while p < 16 {
        o[p] = if SHIFT6_LO[p] >= 4 { 0xFF } else { 0 };
        p += 1;
    }
    o
};

const fn same(a: &[u8; 16], b: &[u8; 16]) -> bool {
    let mut p = 0;
    while p < 16 {
        if a[p] != b[p] {
            return false;
        }
        p += 1;
    }
    true
}

// (the tables are read the way the instruction sets use them: both planes take the same byte, the shifts of a plane are two values four apart, the fifth bit's masks of the planes are one place apart)
const _: () = {
    assert!(same(&high_byte(&LANE_HI, false), &IDX5) && same(&high_byte(&LANE_HI, true), &IDX6));
    let mut p = 0;
    while p < 16 {
        assert!(SHIFT6_LO[p] == 0 || SHIFT6_LO[p] == 4);
        assert!(SHIFT6_HI[p] == SHIFT6_LO[p] + 2);
        assert!(MASK5_HI[p] as u32 == 2 * MASK5_LO[p] as u32);
        p += 1;
    }
};

#[inline(always)]
unsafe fn scale_at(sc: *const u8, i: usize) -> f32 {
    bf16_to_f32(u16::from_le(sc.add(2 * i).cast::<u16>().read_unaligned()))
}

// ---- the products, once ---------------------------------------------------------------------------------------------------------------------------

/// `R` rows from row `r`: their first-product sums into `sums[0..R]`.
#[inline(always)]
unsafe fn up_block<I: Isa, const B: u32, const R: usize>(rows: &RowsView, zp: *const f32, r: usize, sums: *mut f32) {
    let (runs, stride) = (rows.runs(), rows.stride());
    let hb = 4 * (B as usize - 4);
    let (data, sc) = (rows.data.as_ptr(), rows.scales.as_ptr());
    let mut acc = [I::zero(); R];
    for k in 0..runs {
        let zl = I::load(zp.add(32 * k));
        let zh = I::load(zp.add(32 * k + 16));
        for (j, a) in acc.iter_mut().enumerate() {
            let row = data.add((r + j) * stride);
            let (lo, hi) = I::decode::<B>(row.add(16 * k), row.add(16 * runs + hb * k));
            let t = I::fma(hi, zh, I::mul(lo, zl));
            *a = I::fma(t, I::splat(scale_at(sc, (r + j) * runs + k)), *a);
        }
    }
    for (j, a) in acc.into_iter().enumerate() {
        *sums.add(j) = I::hsum(a);
    }
}

/// The first-product sums of rows r0 to r1 (z permuted: `zp`) into `sums`, `R` rows at a time.
#[inline(always)]
unsafe fn up_rows<I: Isa, const B: u32, const R: usize>(rows: &RowsView, zp: *const f32, r0: usize, r1: usize, sums: *mut f32) {
    let mut r = r0;
    while r + R <= r1 {
        up_block::<I, B, R>(rows, zp, r, sums.add(r - r0));
        r += R;
    }
    while r < r1 {
        up_block::<I, B, 1>(rows, zp, r, sums.add(r - r0));
        r += 1;
    }
}

/// The 1,024 outputs `acc` (in the kernels' order) = the sum over the active neurons (row, activation) of the row's weights times the activation, one fused multiply-add per output and neuron, in order.
#[inline(always)]
unsafe fn down_acc<I: Isa, const B: u32>(rows: &RowsView, active: &[(u32, f32)], acc: *mut f32) {
    let (runs, stride) = (rows.runs(), rows.stride());
    let hb = 4 * (B as usize - 4);
    let (data, sc) = (rows.data.as_ptr(), rows.scales.as_ptr());
    for k in 0..runs {
        I::store(acc.add(32 * k), I::zero());
        I::store(acc.add(32 * k + 16), I::zero());
    }
    // a neuron's row is a few cache lines the hardware does not expect: the next ones are asked for as this one is read
    const AHEAD: usize = 3;
    let touch = |j: u32| {
        let (row, s) = (data.add(j as usize * stride), sc.add(j as usize * runs * 2));
        (0..stride).step_by(64).for_each(|o| I::prefetch(row.add(o)));
        (0..runs * 2).step_by(64).for_each(|o| I::prefetch(s.add(o)));
    };
    active.iter().take(AHEAD).for_each(|&(j, _)| touch(j));
    for (n, &(j, a)) in active.iter().enumerate() {
        if let Some(&(next, _)) = active.get(n + AHEAD) {
            touch(next);
        }
        let (row, s) = (data.add(j as usize * stride), sc.add(j as usize * runs * 2));
        for k in 0..runs {
            let (lo, hi) = I::decode::<B>(row.add(16 * k), row.add(16 * runs + hb * k));
            let c = I::splat(a * scale_at(s, k));
            let (ol, oh) = (I::load(acc.add(32 * k)), I::load(acc.add(32 * k + 16)));
            I::store(acc.add(32 * k), I::fma(lo, c, ol));
            I::store(acc.add(32 * k + 16), I::fma(hi, c, oh));
        }
    }
}

// each instruction set's entries: the bits chosen once a call, the code in the instruction set's features
macro_rules! entries {
    ($up:ident, $down:ident, $isa:ty, $r:literal $(, $feat:literal)?) => {
        $(#[target_feature(enable = $feat)])?
        unsafe fn $up(rows: &RowsView, zp: *const f32, r0: usize, r1: usize, sums: *mut f32) {
            match rows.bits {
                4 => up_rows::<$isa, 4, $r>(rows, zp, r0, r1, sums),
                5 => up_rows::<$isa, 5, $r>(rows, zp, r0, r1, sums),
                _ => up_rows::<$isa, 6, $r>(rows, zp, r0, r1, sums),
            }
        }

        $(#[target_feature(enable = $feat)])?
        unsafe fn $down(rows: &RowsView, active: &[(u32, f32)], acc: *mut f32) {
            match rows.bits {
                4 => down_acc::<$isa, 4>(rows, active, acc),
                5 => down_acc::<$isa, 5>(rows, active, acc),
                _ => down_acc::<$isa, 6>(rows, active, acc),
            }
        }
    };
}

entries!(up_scalar, down_scalar, Scalar, 1);
#[cfg(target_arch = "aarch64")]
entries!(up_neon, down_neon, Neon, 2);
#[cfg(target_arch = "x86_64")]
entries!(up_avx2, down_avx2, Avx2, 2, "avx2,fma");
#[cfg(target_arch = "x86_64")]
entries!(up_avx512, down_avx512, Avx512, 4, "avx512f,avx2,fma");

/// The first-product sums of rows r0 to r1 of `rows` against `zp` (z in the kernels' order, `rows.cols` values) into `sums[0..r1 - r0]`, by `kernel` (which this computer runs: `Kernel::runs_here`).
pub(crate) fn up(kernel: Kernel, rows: &RowsView, zp: &[f32], r0: usize, r1: usize, sums: &mut [f32]) {
    assert!(rows.is_valid() && r0 <= r1 && r1 <= rows.rows && zp.len() >= rows.cols && sums.len() >= r1 - r0, "rows {r0} to {r1} of {}, z of {} values, {} sums", rows.rows, zp.len(), sums.len());
    let (z, s) = (zp.as_ptr(), sums.as_mut_ptr());
    // SAFETY: the reads and writes are inside the slices (checked above), and the kernel's instructions are ones this computer has.
    unsafe {
        match kernel {
            Kernel::Scalar => up_scalar(rows, z, r0, r1, s),
            #[cfg(target_arch = "aarch64")]
            Kernel::Neon => up_neon(rows, z, r0, r1, s),
            #[cfg(target_arch = "x86_64")]
            Kernel::Avx2 => up_avx2(rows, z, r0, r1, s),
            #[cfg(target_arch = "x86_64")]
            Kernel::Avx512 => up_avx512(rows, z, r0, r1, s),
            _ => unreachable!("a kernel this computer does not run"),
        }
    }
}

/// The outputs `acc[0..cols]` (in the kernels' order) of the active neurons (row, activation) of `rows`, by `kernel`.
pub(crate) fn down(kernel: Kernel, rows: &RowsView, active: &[(u32, f32)], acc: &mut [f32]) {
    assert!(rows.is_valid() && acc.len() >= rows.cols && active.iter().all(|&(j, _)| (j as usize) < rows.rows), "{} outputs of {} and a neuron past the {} rows", acc.len(), rows.cols, rows.rows);
    let a = acc.as_mut_ptr();
    // SAFETY: as in `up`.
    unsafe {
        match kernel {
            Kernel::Scalar => down_scalar(rows, active, a),
            #[cfg(target_arch = "aarch64")]
            Kernel::Neon => down_neon(rows, active, a),
            #[cfg(target_arch = "x86_64")]
            Kernel::Avx2 => down_avx2(rows, active, a),
            #[cfg(target_arch = "x86_64")]
            Kernel::Avx512 => down_avx512(rows, active, a),
            _ => unreachable!("a kernel this computer does not run"),
        }
    }
}

// ---- the codebook rows' products, once ------------------------------------------------------------------------------------------------------------

/// `R` rows from row `r` of a codebook matrix: their first-product sums against `z` (natural order) into `sums[0..R]`: 16 lane sums a row, lane p taking the run's codes 32 k + p (times z) and 32 k + 16 + p (fused in),
/// each run's `t` times the run's scale added by a fused multiply-add.
#[inline(always)]
unsafe fn up_block_cb<I: Isa, const B: u32, const R: usize>(m: &CbView, z: *const f32, r: usize, sums: *mut f32) {
    let (cs, ss, bs) = strides(B, m.cols);
    let runs = m.cols / 32;
    let (codes, sc, cb) = (m.codes.as_ptr(), m.scales.as_ptr(), m.codebook.as_ptr());
    let tabs: [I::T; R] = core::array::from_fn(|j| I::table::<B>(cb.add((r + j) * bs)));
    let mut acc = [I::zero(); R];
    for k in 0..runs {
        let zl = I::load(z.add(32 * k));
        let zh = I::load(z.add(32 * k + 16));
        for (j, a) in acc.iter_mut().enumerate() {
            let (lo, hi) = I::decode_cb::<B>(tabs[j], codes.add((r + j) * cs + 4 * B as usize * k));
            let t = I::fma(hi, zh, I::mul(lo, zl));
            *a = I::fma(t, I::splat(scale_at(sc.add((r + j) * ss), k)), *a);
        }
    }
    for (j, a) in acc.into_iter().enumerate() {
        *sums.add(j) = I::hsum(a);
    }
}

/// The first-product sums of rows r0 to r1 of a codebook matrix, `R` rows at a time.
#[inline(always)]
unsafe fn up_rows_cb<I: Isa, const B: u32, const R: usize>(m: &CbView, z: *const f32, r0: usize, r1: usize, sums: *mut f32) {
    let mut r = r0;
    while r + R <= r1 {
        up_block_cb::<I, B, R>(m, z, r, sums.add(r - r0));
        r += R;
    }
    while r < r1 {
        up_block_cb::<I, B, 1>(m, z, r, sums.add(r - r0));
        r += 1;
    }
}

/// The outputs `acc` (natural order) = the sum over the active neurons (row, activation) of the row's weights times the activation: per run `c = a s`, then one fused multiply-add per output and neuron, in order.
#[inline(always)]
unsafe fn down_acc_cb<I: Isa, const B: u32>(m: &CbView, active: &[(u32, f32)], acc: *mut f32) {
    let (cs, ss, bs) = strides(B, m.cols);
    let runs = m.cols / 32;
    let (codes, sc, cb) = (m.codes.as_ptr(), m.scales.as_ptr(), m.codebook.as_ptr());
    for k in 0..runs {
        I::store(acc.add(32 * k), I::zero());
        I::store(acc.add(32 * k + 16), I::zero());
    }
    // the three arrays of a neuron's row are a few cache lines the hardware does not expect: the next rows' are asked for as this one is read
    const AHEAD: usize = 3;
    let touch = |j: u32| {
        let j = j as usize;
        (0..cs).step_by(64).for_each(|o| I::prefetch(codes.add(j * cs + o)));
        I::prefetch(sc.add(j * ss));
        I::prefetch(cb.add(j * bs));
    };
    active.iter().take(AHEAD).for_each(|&(j, _)| touch(j));
    for (n, &(j, a)) in active.iter().enumerate() {
        if let Some(&(next, _)) = active.get(n + AHEAD) {
            touch(next);
        }
        let j = j as usize;
        let (row, s, t) = (codes.add(j * cs), sc.add(j * ss), I::table::<B>(cb.add(j * bs)));
        for k in 0..runs {
            let (lo, hi) = I::decode_cb::<B>(t, row.add(4 * B as usize * k));
            let c = I::splat(a * scale_at(s, k));
            let (ol, oh) = (I::load(acc.add(32 * k)), I::load(acc.add(32 * k + 16)));
            I::store(acc.add(32 * k), I::fma(lo, c, ol));
            I::store(acc.add(32 * k + 16), I::fma(hi, c, oh));
        }
    }
}

macro_rules! entries_cb {
    ($up:ident, $down:ident, $isa:ty, $r:literal $(, $feat:literal)?) => {
        $(#[target_feature(enable = $feat)])?
        unsafe fn $up(m: &CbView, z: *const f32, r0: usize, r1: usize, sums: *mut f32) {
            match m.bits {
                2 => up_rows_cb::<$isa, 2, $r>(m, z, r0, r1, sums),
                _ => up_rows_cb::<$isa, 3, $r>(m, z, r0, r1, sums),
            }
        }

        $(#[target_feature(enable = $feat)])?
        unsafe fn $down(m: &CbView, active: &[(u32, f32)], acc: *mut f32) {
            match m.bits {
                2 => down_acc_cb::<$isa, 2>(m, active, acc),
                _ => down_acc_cb::<$isa, 3>(m, active, acc),
            }
        }
    };
}

entries_cb!(up_cb_scalar, down_cb_scalar, Scalar, 1);
#[cfg(target_arch = "aarch64")]
entries_cb!(up_cb_neon, down_cb_neon, Neon, 2);
#[cfg(target_arch = "x86_64")]
entries_cb!(up_cb_avx2, down_cb_avx2, Avx2, 2, "avx2,fma");
#[cfg(target_arch = "x86_64")]
entries_cb!(up_cb_avx512, down_cb_avx512, Avx512, 4, "avx512f,avx2,fma");

// ---- the codebook rows of GLM-5.3-Flash's pack (glyd-hbr-codes/2, src/cpu_cb.rs): ternary, two, three and four bits, no run scales -------------------------------

/// A format of the pack: the row's codes and the entries of its table. Two, three and four bits are bitstreams (cb2: byte b holds the codes 4 b to 4 b + 3 at bits 0-1, 2-3, 4-5, 6-7; cb3: the codes 8 g to 8 g + 7 are the
/// 24-bit little-endian integer of bytes 3 g to 3 g + 2, code m at bits 3 m to 3 m + 2; cb4: the low nibble first); ternary holds five codes a byte as base-3 digits (byte = c0 + 3 c1 + 9 c2 + 27 c3 + 81 c4, the last byte
/// of a row padded with zeros). Tq: a trellis row (src/qx_trellis.rs), 2 bits a weight as a circular stream of `cols / 16` little-endian 32-bit words, its table one bf16 entry, the row's scale. Tq3, Tq4: the
/// same at 3 and 4 bits a weight (`tq_state_k`), `3 cols / 32` and `cols / 8` words.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fmt {
    Tern,
    Cb2,
    Cb3,
    Cb4,
    Tq,
    Tq3,
    Tq4,
}

impl Fmt {
    /// Entries of a row's table.
    pub const fn entries(self) -> usize {
        match self {
            Fmt::Tern => 3,
            Fmt::Cb2 => 4,
            Fmt::Cb3 => 8,
            Fmt::Cb4 => 16,
            Fmt::Tq | Fmt::Tq3 | Fmt::Tq4 => 1,
        }
    }

    /// Bytes of a row's codes for `cols` values.
    pub const fn row_bytes(self, cols: usize) -> usize {
        match self {
            Fmt::Tern => cols.div_ceil(5),
            Fmt::Cb2 | Fmt::Tq => cols / 4,
            Fmt::Cb3 | Fmt::Tq3 => cols * 3 / 8,
            Fmt::Cb4 | Fmt::Tq4 => cols / 2,
        }
    }

    /// A format q row's bits a weight (None: a codebook row).
    pub const fn tq_bits(self) -> Option<usize> {
        match self {
            Fmt::Tq => Some(2),
            Fmt::Tq3 => Some(3),
            Fmt::Tq4 => Some(4),
            _ => None,
        }
    }

    /// Whether the rows are format q's (a scale a row, the weights from their states).
    pub const fn is_tq(self) -> bool {
        self.tq_bits().is_some()
    }

    /// Outputs an item of the second product takes: 160 (32 bytes) of a ternary row, 256 of the others.
    pub const fn slice(self) -> usize {
        match self {
            Fmt::Tern => 160,
            _ => 256,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Fmt::Tern => "t",
            Fmt::Cb2 => "a",
            Fmt::Cb3 => "b",
            Fmt::Cb4 => "c",
            Fmt::Tq => "q",
            Fmt::Tq3 => "q3",
            Fmt::Tq4 => "q4",
        }
    }

    pub fn parse(s: &str) -> Option<Fmt> {
        Some(match s {
            "t" => Fmt::Tern,
            "a" => Fmt::Cb2,
            "b" => Fmt::Cb3,
            "c" => Fmt::Cb4,
            "q" => Fmt::Tq,
            "q3" => Fmt::Tq3,
            "q4" => Fmt::Tq4,
            _ => return None,
        })
    }
}

/// A matrix of the pack: `rows` rows of `cols` values, their codes (a row's bytes one after another) and their tables (`entries` bf16 numbers a row, ascending, the weights themselves).
#[derive(Clone, Copy, Debug)]
pub struct GlmMat<'a> {
    pub fmt: Fmt,
    pub rows: usize,
    pub cols: usize,
    pub codes: &'a [u8],
    pub book: &'a [u8],
}

/// A ternary row is read in groups of 160 values (32 bytes): the groups of a row, the last one partly padding.
pub(crate) const fn tern_groups(cols: usize) -> usize {
    cols.div_ceil(160)
}

impl GlmMat<'_> {
    /// Whether the fields are a layout's (the kernels read raw memory): the arrays the formats' sizes, the columns of a bitstream matrix in 256s (a ternary one's in any number).
    pub fn is_valid(&self) -> bool {
        self.rows > 0
            && self.cols > 0
            && (self.fmt == Fmt::Tern || self.cols % 256 == 0)
            && self.codes.len() == self.rows * self.fmt.row_bytes(self.cols)
            && self.book.len() == self.rows * 2 * self.fmt.entries()
    }

    /// Values of the vector the first product takes: the columns, or for ternary rows the groups' 160 each (`tern_permute`).
    pub fn z_len(&self) -> usize {
        if self.fmt == Fmt::Tern {
            160 * tern_groups(self.cols)
        } else {
            self.cols
        }
    }
}

/// The vector a ternary row's first product takes: in each group of 160 values, the value of code 5 b + d (byte b, digit d) at 32 d + b, so that a digit's 32 bytes meet 32 consecutive numbers; zero where the group
/// has no value. `out` is `160 * tern_groups(z.len())` long.
pub(crate) fn tern_permute(z: &[f32], out: &mut [f32]) {
    for (g, grp) in out.chunks_exact_mut(160).enumerate() {
        for d in 0..5 {
            for b in 0..32 {
                grp[32 * d + b] = z.get(160 * g + 5 * b + d).copied().unwrap_or(0.0);
            }
        }
    }
}

/// `R` rows at a time of a bitstream matrix of `B` bits (2 to 4; `m.cols` a multiple of 256): the sums of the rows' weights times `z` (natural order) into `sums[0..r1 - r0]`: per row two vectors of 16 lane sums (the
/// run of 32 codes's codes 0 to 15 and 16 to 31), each a fused multiply-add per run, added as the two trees of lanes.
#[inline(always)]
unsafe fn dot_rows_cbn<I: Isa, const R: usize, const B: u32>(m: &GlmMat, z: *const f32, r0: usize, r1: usize, sums: *mut f32) {
    let (cs, runs, rbytes, ent) = (m.fmt.row_bytes(m.cols), m.cols / 32, 4 * B as usize, 2usize << B);
    let (codes, cb) = (m.codes.as_ptr(), m.book.as_ptr());
    // (a run's bytes are a fraction of a cache line: the next block's rows are asked for a line at a time, one in `per` runs)
    let per = (64 / rbytes).max(1);
    let mut r = r0;
    while r < r1 {
        let n = R.min(r1 - r);
        if n == R {
            let tabs: [I::T; R] = core::array::from_fn(|j| I::table::<B>(cb.add((r + j) * ent)));
            let (mut a0, mut a1) = ([I::zero(); R], [I::zero(); R]);
            // (the processor's own prefetching starts over at every page, and a block of rows is a few KB)
            let ahead = r + 2 * R <= r1;
            for k in 0..runs {
                if ahead && k % per == 0 {
                    for j in 0..R {
                        I::prefetch(codes.add((r + R + j) * cs + rbytes * k));
                    }
                }
                let (zl, zh) = (I::load(z.add(32 * k)), I::load(z.add(32 * k + 16)));
                for j in 0..R {
                    let (lo, hi) = I::decode_cb::<B>(tabs[j], codes.add((r + j) * cs + rbytes * k));
                    a0[j] = I::fma(lo, zl, a0[j]);
                    a1[j] = I::fma(hi, zh, a1[j]);
                }
            }
            for j in 0..R {
                *sums.add(r + j - r0) = I::hsum(a0[j]) + I::hsum(a1[j]);
            }
        } else {
            // (the rows left, one at a time)
            for row in r..r + n {
                let t = I::table::<B>(cb.add(row * ent));
                let (mut a0, mut a1) = (I::zero(), I::zero());
                for k in 0..runs {
                    let (lo, hi) = I::decode_cb::<B>(t, codes.add(row * cs + rbytes * k));
                    a0 = I::fma(lo, I::load(z.add(32 * k)), a0);
                    a1 = I::fma(hi, I::load(z.add(32 * k + 16)), a1);
                }
                *sums.add(row - r0) = I::hsum(a0) + I::hsum(a1);
            }
        }
        r += n;
    }
}

/// Outputs `o0` to `o0 + 256` of the sum over the rows (neurons) n of a bitstream matrix of `B` bits of `act[n]` times row n's weights: 16 vectors of accumulators in registers over all the neurons in order, one fused
/// multiply-add each, into `out[0..256]` (o0 a multiple of 256).
#[inline(always)]
unsafe fn down_cols_cbn<I: Isa, const B: u32>(m: &GlmMat, act: *const f32, o0: usize, out: *mut f32) {
    const AHEAD: usize = 8;
    let (cs, rbytes, ent) = (m.fmt.row_bytes(m.cols), 4 * B as usize, 2usize << B);
    let (codes, cb) = (m.codes.as_ptr(), m.book.as_ptr());
    // the slice's bytes in a row: 256 values of B bits, 32 B bytes from the byte its first value is in
    let (at, len) = (o0 * B as usize / 8, 32 * B as usize);
    let mut acc = [I::zero(); 16];
    for n in 0..m.rows {
        if n + AHEAD < m.rows {
            let p = codes.add((n + AHEAD) * cs + at);
            for off in (0..len).step_by(64) {
                I::prefetch(p.add(off));
            }
            I::prefetch(p.add(len - 1));
        }
        let a = I::splat(*act.add(n));
        let t = I::table::<B>(cb.add(n * ent));
        let row = codes.add(n * cs + at);
        for k in 0..8 {
            let (lo, hi) = I::decode_cb::<B>(t, row.add(rbytes * k));
            acc[2 * k] = I::fma(lo, a, acc[2 * k]);
            acc[2 * k + 1] = I::fma(hi, a, acc[2 * k + 1]);
        }
    }
    for (i, v) in acc.into_iter().enumerate() {
        I::store(out.add(16 * i), v);
    }
}

/// `dot_rows_cbn` for `T` tokens at once (`zs`, each `m.cols` values, natural order): a run's codes decoded once and multiplied by each token's values, every token's sums the same multiply-adds in the same order as
/// `dot_rows_cbn` makes them alone.
#[inline(always)]
unsafe fn dot_rows_cbn_t<I: Isa, const R: usize, const B: u32, const T: usize>(m: &GlmMat, zs: [*const f32; T], r0: usize, r1: usize, sums: [*mut f32; T]) {
    let (cs, runs, rbytes, ent) = (m.fmt.row_bytes(m.cols), m.cols / 32, 4 * B as usize, 2usize << B);
    let (codes, cb) = (m.codes.as_ptr(), m.book.as_ptr());
    let per = (64 / rbytes).max(1);
    let mut r = r0;
    while r < r1 {
        let n = R.min(r1 - r);
        if n == R {
            let tabs: [I::T; R] = core::array::from_fn(|j| I::table::<B>(cb.add((r + j) * ent)));
            let (mut a0, mut a1) = ([[I::zero(); T]; R], [[I::zero(); T]; R]);
            let ahead = r + 2 * R <= r1;
            for k in 0..runs {
                if ahead && k % per == 0 {
                    for j in 0..R {
                        I::prefetch(codes.add((r + R + j) * cs + rbytes * k));
                    }
                }
                let zl: [I::F; T] = core::array::from_fn(|t| I::load(zs[t].add(32 * k)));
                let zh: [I::F; T] = core::array::from_fn(|t| I::load(zs[t].add(32 * k + 16)));
                for j in 0..R {
                    let (lo, hi) = I::decode_cb::<B>(tabs[j], codes.add((r + j) * cs + rbytes * k));
                    for t in 0..T {
                        a0[j][t] = I::fma(lo, zl[t], a0[j][t]);
                        a1[j][t] = I::fma(hi, zh[t], a1[j][t]);
                    }
                }
            }
            for j in 0..R {
                for t in 0..T {
                    *sums[t].add(r + j - r0) = I::hsum(a0[j][t]) + I::hsum(a1[j][t]);
                }
            }
        } else {
            for row in r..r + n {
                let tb = I::table::<B>(cb.add(row * ent));
                let (mut a0, mut a1) = ([I::zero(); T], [I::zero(); T]);
                for k in 0..runs {
                    let (lo, hi) = I::decode_cb::<B>(tb, codes.add(row * cs + rbytes * k));
                    for t in 0..T {
                        a0[t] = I::fma(lo, I::load(zs[t].add(32 * k)), a0[t]);
                        a1[t] = I::fma(hi, I::load(zs[t].add(32 * k + 16)), a1[t]);
                    }
                }
                for t in 0..T {
                    *sums[t].add(row - r0) = I::hsum(a0[t]) + I::hsum(a1[t]);
                }
            }
        }
        r += n;
    }
}

/// `down_cols_cbn` for `T` tokens at once, outputs `o0` to `o0 + 128` (o0 a multiple of 128): a row's codes decoded once and multiplied by each token's activation, 8 vectors of accumulators a token, every output the
/// same chain of multiply-adds over the neurons in order as `down_cols_cbn` makes it.
#[inline(always)]
unsafe fn down_cols_cbn_t<I: Isa, const B: u32, const T: usize>(m: &GlmMat, acts: [*const f32; T], o0: usize, outs: [*mut f32; T]) {
    const AHEAD: usize = 8;
    let (cs, rbytes, ent) = (m.fmt.row_bytes(m.cols), 4 * B as usize, 2usize << B);
    let (codes, cb) = (m.codes.as_ptr(), m.book.as_ptr());
    let (at, len) = (o0 * B as usize / 8, 16 * B as usize);
    let mut acc = [[I::zero(); 8]; T];
    for n in 0..m.rows {
        if n + AHEAD < m.rows {
            let p = codes.add((n + AHEAD) * cs + at);
            I::prefetch(p);
            I::prefetch(p.add(len - 1));
        }
        let tb = I::table::<B>(cb.add(n * ent));
        let row = codes.add(n * cs + at);
        let a: [I::F; T] = core::array::from_fn(|t| I::splat(*acts[t].add(n)));
        for k in 0..4 {
            let (lo, hi) = I::decode_cb::<B>(tb, row.add(rbytes * k));
            for t in 0..T {
                acc[t][2 * k] = I::fma(lo, a[t], acc[t][2 * k]);
                acc[t][2 * k + 1] = I::fma(hi, a[t], acc[t][2 * k + 1]);
            }
        }
    }
    for t in 0..T {
        for (i, v) in acc[t].into_iter().enumerate() {
            I::store(outs[t].add(16 * i), v);
        }
    }
}

macro_rules! entries_glm_t {
    ($m:ident, $isa:ty $(, $feat:literal)?) => {
        mod $m {
            use super::*;

            $(#[target_feature(enable = $feat)])?
            pub(super) unsafe fn dot<const B: u32, const T: usize>(m: &GlmMat, zs: [*const f32; T], r0: usize, r1: usize, s: [*mut f32; T]) {
                dot_rows_cbn_t::<$isa, 2, B, T>(m, zs, r0, r1, s)
            }
            $(#[target_feature(enable = $feat)])?
            pub(super) unsafe fn down<const B: u32, const T: usize>(m: &GlmMat, acts: [*const f32; T], o0: usize, outs: [*mut f32; T]) {
                down_cols_cbn_t::<$isa, B, T>(m, acts, o0, outs)
            }
        }
    };
}

/// The trellis rows' kernels for several tokens, per instruction set.
macro_rules! entries_tq_t {
    ($m:ident, $isa:ty $(, $feat:literal)?) => {
        mod $m {
            use super::*;

            $(#[target_feature(enable = $feat)])?
            pub(super) unsafe fn dot<const T: usize>(m: &GlmMat, zs: [*const f32; T], r0: usize, r1: usize, s: [*mut f32; T]) {
                dot_rows_tq_t::<$isa, T>(m, zs, r0, r1, s)
            }
            $(#[target_feature(enable = $feat)])?
            pub(super) unsafe fn down<const T: usize>(m: &GlmMat, acts: [*const f32; T], o0: usize, outs: [*mut f32; T]) {
                down_cols_tq_t::<$isa, 8, T>(m, acts, o0, outs)
            }
        }
    };
}

entries_tq_t!(tqt_scalar, Scalar);
#[cfg(target_arch = "aarch64")]
entries_tq_t!(tqt_neon, Neon);
#[cfg(target_arch = "x86_64")]
entries_tq_t!(tqt_avx2, Avx2, "avx2,fma");
#[cfg(target_arch = "x86_64")]
entries_tq_t!(tqt_avx512, Avx512, "avx512f,avx2,fma");
#[cfg(target_arch = "x86_64")]
entries_tq_t!(tqt_avx512v, Avx512Vnni, "avx512f,avx512bw,avx512vnni,avx512vbmi2,avx2,fma");

/// The q3 and q4 rows' kernels for several tokens, per instruction set.
macro_rules! entries_tqk_t {
    ($m:ident, $isa:ty $(, $feat:literal)?) => {
        mod $m {
            use super::*;

            $(#[target_feature(enable = $feat)])?
            pub(super) unsafe fn dot<const K: usize, const T: usize>(m: &GlmMat, zs: [*const f32; T], r0: usize, r1: usize, s: [*mut f32; T]) {
                dot_rows_tqk_t::<$isa, K, T>(m, zs, r0, r1, s)
            }
            $(#[target_feature(enable = $feat)])?
            pub(super) unsafe fn down<const K: usize, const T: usize>(m: &GlmMat, acts: [*const f32; T], o0: usize, outs: [*mut f32; T]) {
                down_cols_tqk_t::<$isa, 8, K, T>(m, acts, o0, outs)
            }
        }
    };
}

entries_tqk_t!(tqkt_scalar, Scalar);
#[cfg(target_arch = "aarch64")]
entries_tqk_t!(tqkt_neon, Neon);
#[cfg(target_arch = "x86_64")]
entries_tqk_t!(tqkt_avx2, Avx2, "avx2,fma");
#[cfg(target_arch = "x86_64")]
entries_tqk_t!(tqkt_avx512, Avx512, "avx512f,avx2,fma");
#[cfg(target_arch = "x86_64")]
entries_tqk_t!(tqkt_avx512v, Avx512Vnni, "avx512f,avx512bw,avx512vnni,avx512vbmi2,avx2,fma");

entries_glm_t!(glmt_scalar, Scalar);
#[cfg(target_arch = "aarch64")]
entries_glm_t!(glmt_neon, Neon);
#[cfg(target_arch = "x86_64")]
entries_glm_t!(glmt_avx2, Avx2, "avx2,fma");
#[cfg(target_arch = "x86_64")]
entries_glm_t!(glmt_avx512, Avx512, "avx512f,avx2,fma");

/// The first-product sums of rows r0 to r1 of the codebook matrix `m` (two to four bits) for up to four tokens at once (`zs`, natural order), into `sums[t][0..r1 - r0]`: each token's sums as `dot_glm` makes them.
pub(crate) fn dot_glm_multi(kernel: Kernel, m: &GlmMat, zs: &[&[f32]], r0: usize, r1: usize, sums: &mut [&mut [f32]]) {
    assert!(m.is_valid() && m.fmt != Fmt::Tern && (1..=4).contains(&zs.len()) && zs.len() == sums.len() && r0 <= r1 && r1 <= m.rows, "rows {r0} to {r1} of {}, {} tokens", m.rows, zs.len());
    assert!(zs.iter().all(|z| z.len() >= m.cols) && sums.iter().all(|s| s.len() >= r1 - r0), "the tokens' values or sums are short");
    if let Some(k @ 3..=4) = m.fmt.tq_bits() {
        macro_rules! tqk {
            ($md:ident, $k:literal) => {{
                // SAFETY: the reads and writes are inside the slices (checked above), and the kernel's instructions are ones this computer has.
                unsafe {
                    match zs.len() {
                        1 => $md::dot::<$k, 1>(m, [zs[0].as_ptr()], r0, r1, [sums[0].as_mut_ptr()]),
                        2 => $md::dot::<$k, 2>(m, [zs[0].as_ptr(), zs[1].as_ptr()], r0, r1, [sums[0].as_mut_ptr(), sums[1].as_mut_ptr()]),
                        3 => $md::dot::<$k, 3>(m, [zs[0].as_ptr(), zs[1].as_ptr(), zs[2].as_ptr()], r0, r1, [sums[0].as_mut_ptr(), sums[1].as_mut_ptr(), sums[2].as_mut_ptr()]),
                        _ => $md::dot::<$k, 4>(m, [zs[0].as_ptr(), zs[1].as_ptr(), zs[2].as_ptr(), zs[3].as_ptr()], r0, r1, [sums[0].as_mut_ptr(), sums[1].as_mut_ptr(), sums[2].as_mut_ptr(), sums[3].as_mut_ptr()]),
                    }
                }
            }};
        }
        macro_rules! by_k {
            ($md:ident) => {
                if k == 3 {
                    tqk!($md, 3)
                } else {
                    tqk!($md, 4)
                }
            };
        }
        match kernel {
            Kernel::Scalar => by_k!(tqkt_scalar),
            #[cfg(target_arch = "aarch64")]
            Kernel::Neon => by_k!(tqkt_neon),
            #[cfg(target_arch = "x86_64")]
            Kernel::Avx2 => by_k!(tqkt_avx2),
            #[cfg(target_arch = "x86_64")]
            Kernel::Avx512 if tq_vnni() => by_k!(tqkt_avx512v),
            #[cfg(target_arch = "x86_64")]
            Kernel::Avx512 => by_k!(tqkt_avx512),
            _ => unreachable!("a kernel this computer does not run"),
        }
        return;
    }
    if m.fmt == Fmt::Tq {
        macro_rules! tq {
            ($md:ident) => {{
                // SAFETY: the reads and writes are inside the slices (checked above), and the kernel's instructions are ones this computer has.
                unsafe {
                    match zs.len() {
                        1 => $md::dot::<1>(m, [zs[0].as_ptr()], r0, r1, [sums[0].as_mut_ptr()]),
                        2 => $md::dot::<2>(m, [zs[0].as_ptr(), zs[1].as_ptr()], r0, r1, [sums[0].as_mut_ptr(), sums[1].as_mut_ptr()]),
                        3 => $md::dot::<3>(m, [zs[0].as_ptr(), zs[1].as_ptr(), zs[2].as_ptr()], r0, r1, [sums[0].as_mut_ptr(), sums[1].as_mut_ptr(), sums[2].as_mut_ptr()]),
                        _ => $md::dot::<4>(m, [zs[0].as_ptr(), zs[1].as_ptr(), zs[2].as_ptr(), zs[3].as_ptr()], r0, r1, [sums[0].as_mut_ptr(), sums[1].as_mut_ptr(), sums[2].as_mut_ptr(), sums[3].as_mut_ptr()]),
                    }
                }
            }};
        }
        match kernel {
            Kernel::Scalar => tq!(tqt_scalar),
            #[cfg(target_arch = "aarch64")]
            Kernel::Neon => tq!(tqt_neon),
            #[cfg(target_arch = "x86_64")]
            Kernel::Avx2 => tq!(tqt_avx2),
            #[cfg(target_arch = "x86_64")]
            Kernel::Avx512 if tq_vnni() => tq!(tqt_avx512v),
            #[cfg(target_arch = "x86_64")]
            Kernel::Avx512 => tq!(tqt_avx512),
            _ => unreachable!("a kernel this computer does not run"),
        }
        return;
    }
    macro_rules! go {
        ($md:ident, $b:literal) => {{
            // SAFETY: the reads and writes are inside the slices (checked above), and the kernel's instructions are ones this computer has.
            unsafe {
                match zs.len() {
                    1 => $md::dot::<$b, 1>(m, [zs[0].as_ptr()], r0, r1, [sums[0].as_mut_ptr()]),
                    2 => $md::dot::<$b, 2>(m, [zs[0].as_ptr(), zs[1].as_ptr()], r0, r1, [sums[0].as_mut_ptr(), sums[1].as_mut_ptr()]),
                    3 => {
                        let (a, b) = sums.split_at_mut(1);
                        let (b, c) = b.split_at_mut(1);
                        $md::dot::<$b, 2>(m, [zs[0].as_ptr(), zs[1].as_ptr()], r0, r1, [a[0].as_mut_ptr(), b[0].as_mut_ptr()]);
                        $md::dot::<$b, 1>(m, [zs[2].as_ptr()], r0, r1, [c[0].as_mut_ptr()])
                    }
                    _ => $md::dot::<$b, 4>(m, [zs[0].as_ptr(), zs[1].as_ptr(), zs[2].as_ptr(), zs[3].as_ptr()], r0, r1, [sums[0].as_mut_ptr(), sums[1].as_mut_ptr(), sums[2].as_mut_ptr(), sums[3].as_mut_ptr()]),
                }
            }
        }};
    }
    macro_rules! by_fmt {
        ($md:ident) => {
            match m.fmt {
                Fmt::Cb2 => go!($md, 2),
                Fmt::Cb3 => go!($md, 3),
                _ => go!($md, 4),
            }
        };
    }
    match kernel {
        Kernel::Scalar => by_fmt!(glmt_scalar),
        #[cfg(target_arch = "aarch64")]
        Kernel::Neon => by_fmt!(glmt_neon),
        #[cfg(target_arch = "x86_64")]
        Kernel::Avx2 => by_fmt!(glmt_avx2),
        #[cfg(target_arch = "x86_64")]
        Kernel::Avx512 => by_fmt!(glmt_avx512),
        _ => unreachable!("a kernel this computer does not run"),
    }
}

/// Outputs `o0` to `o0 + 128` (o0 a multiple of 128) of the second product of the codebook matrix `m` (two to four bits) for one or two tokens at once (`acts`), into `outs[t][0..128]`: each token's as `down_glm` makes them.
pub(crate) fn down_glm_multi(kernel: Kernel, m: &GlmMat, acts: &[&[f32]], o0: usize, outs: &mut [&mut [f32]]) {
    assert!(m.is_valid() && m.fmt != Fmt::Tern && (1..=2).contains(&acts.len()) && acts.len() == outs.len() && o0 % 128 == 0 && o0 + 128 <= m.cols, "outputs {o0} of {}, {} tokens", m.cols, acts.len());
    assert!(acts.iter().all(|a| a.len() >= m.rows) && outs.iter().all(|o| o.len() >= 128), "the activations or outputs are short");
    if let Some(k @ 3..=4) = m.fmt.tq_bits() {
        macro_rules! tqk {
            ($md:ident, $k:literal) => {{
                // SAFETY: as in `dot_glm_multi`: 128 outputs from o0 of each token's (checked above).
                unsafe {
                    if acts.len() == 1 {
                        $md::down::<$k, 1>(m, [acts[0].as_ptr()], o0, [outs[0].as_mut_ptr()])
                    } else {
                        let (a, b) = outs.split_at_mut(1);
                        $md::down::<$k, 2>(m, [acts[0].as_ptr(), acts[1].as_ptr()], o0, [a[0].as_mut_ptr(), b[0].as_mut_ptr()])
                    }
                }
            }};
        }
        macro_rules! by_k {
            ($md:ident) => {
                if k == 3 {
                    tqk!($md, 3)
                } else {
                    tqk!($md, 4)
                }
            };
        }
        match kernel {
            Kernel::Scalar => by_k!(tqkt_scalar),
            #[cfg(target_arch = "aarch64")]
            Kernel::Neon => by_k!(tqkt_neon),
            #[cfg(target_arch = "x86_64")]
            Kernel::Avx2 => by_k!(tqkt_avx2),
            #[cfg(target_arch = "x86_64")]
            Kernel::Avx512 if tq_vnni() => by_k!(tqkt_avx512v),
            #[cfg(target_arch = "x86_64")]
            Kernel::Avx512 => by_k!(tqkt_avx512),
            _ => unreachable!("a kernel this computer does not run"),
        }
        return;
    }
    if m.fmt == Fmt::Tq {
        macro_rules! tq {
            ($md:ident) => {{
                // SAFETY: as in `dot_glm_multi`: 128 outputs from o0 of each token's (checked above).
                unsafe {
                    if acts.len() == 1 {
                        $md::down::<1>(m, [acts[0].as_ptr()], o0, [outs[0].as_mut_ptr()])
                    } else {
                        let (a, b) = outs.split_at_mut(1);
                        $md::down::<2>(m, [acts[0].as_ptr(), acts[1].as_ptr()], o0, [a[0].as_mut_ptr(), b[0].as_mut_ptr()])
                    }
                }
            }};
        }
        match kernel {
            Kernel::Scalar => tq!(tqt_scalar),
            #[cfg(target_arch = "aarch64")]
            Kernel::Neon => tq!(tqt_neon),
            #[cfg(target_arch = "x86_64")]
            Kernel::Avx2 => tq!(tqt_avx2),
            #[cfg(target_arch = "x86_64")]
            Kernel::Avx512 if tq_vnni() => tq!(tqt_avx512v),
            #[cfg(target_arch = "x86_64")]
            Kernel::Avx512 => tq!(tqt_avx512),
            _ => unreachable!("a kernel this computer does not run"),
        }
        return;
    }
    macro_rules! go {
        ($md:ident, $b:literal) => {{
            // SAFETY: as in `dot_glm_multi`.
            unsafe {
                if acts.len() == 1 {
                    $md::down::<$b, 1>(m, [acts[0].as_ptr()], o0, [outs[0].as_mut_ptr()])
                } else {
                    let (a, b) = outs.split_at_mut(1);
                    $md::down::<$b, 2>(m, [acts[0].as_ptr(), acts[1].as_ptr()], o0, [a[0].as_mut_ptr(), b[0].as_mut_ptr()])
                }
            }
        }};
    }
    macro_rules! by_fmt {
        ($md:ident) => {
            match m.fmt {
                Fmt::Cb2 => go!($md, 2),
                Fmt::Cb3 => go!($md, 3),
                _ => go!($md, 4),
            }
        };
    }
    match kernel {
        Kernel::Scalar => by_fmt!(glmt_scalar),
        #[cfg(target_arch = "aarch64")]
        Kernel::Neon => by_fmt!(glmt_neon),
        #[cfg(target_arch = "x86_64")]
        Kernel::Avx2 => by_fmt!(glmt_avx2),
        #[cfg(target_arch = "x86_64")]
        Kernel::Avx512 => by_fmt!(glmt_avx512),
        _ => unreachable!("a kernel this computer does not run"),
    }
}

/// The group of 160 values (32 bytes) of a ternary row at `g`: its bytes, or for the last, partial group a copy of the row's bytes left padded with zeros.
#[inline(always)]
unsafe fn tern_bytes(row: *const u8, row_bytes: usize, g: usize, pad: &mut [u8; 32]) -> *const u8 {
    let (at, left) = (32 * g, row_bytes - 32 * g);
    if left >= 32 {
        row.add(at)
    } else {
        *pad = [0; 32];
        core::ptr::copy_nonoverlapping(row.add(at), pad.as_mut_ptr(), left);
        pad.as_ptr()
    }
}

/// `R` rows at a time of a ternary matrix: the sums of the rows' weights times `zq` (`tern_permute`d) into `sums[0..r1 - r0]`. A row is read in groups of 32 bytes: lane l of 32 (two vectors of 16) takes the five
/// digits of byte l, the weights times the numbers of the five planes added by a multiplication and four fused multiply-adds, and the group's lane sums are added to the lane's total; the totals' two vectors are
/// added as trees of lanes as the bitstream kernels' are.
#[inline(always)]
unsafe fn dot_rows_tern<I: Isa, const R: usize>(m: &GlmMat, zq: *const f32, r0: usize, r1: usize, sums: *mut f32) {
    let (cs, groups) = (m.fmt.row_bytes(m.cols), tern_groups(m.cols));
    let (codes, cb) = (m.codes.as_ptr(), m.book.as_ptr());
    let mut r = r0;
    while r < r1 {
        let n = R.min(r1 - r);
        let mut tabs = [I::table_tern(cb.add(r * 6)); R];
        for j in 1..n {
            tabs[j] = I::table_tern(cb.add((r + j) * 6));
        }
        let (mut a0, mut a1) = ([I::zero(); R], [I::zero(); R]);
        let mut pad = [[0u8; 32]; R];
        for g in 0..groups {
            let z = zq.add(160 * g);
            let mut zs = [(I::zero(), I::zero()); 5];
            for d in 0..5 {
                zs[d] = (I::load(z.add(32 * d)), I::load(z.add(32 * d + 16)));
            }
            for j in 0..n {
                let bytes = tern_bytes(codes.add((r + j) * cs), cs, g, &mut pad[j]);
                let w = I::decode_tern(tabs[j], bytes);
                let (mut t0, mut t1) = (I::mul(w[0].0, zs[0].0), I::mul(w[0].1, zs[0].1));
                for d in 1..5 {
                    t0 = I::fma(w[d].0, zs[d].0, t0);
                    t1 = I::fma(w[d].1, zs[d].1, t1);
                }
                // (the lane totals: a sum of the group's, not a chain through the planes)
                a0[j] = I::add(a0[j], t0);
                a1[j] = I::add(a1[j], t1);
            }
        }
        for j in 0..n {
            *sums.add(r + j - r0) = I::hsum(a0[j]) + I::hsum(a1[j]);
        }
        r += n;
    }
}

/// Outputs 160 g to 160 g + 160 (or the matrix's last, up to its width) of the sum over the rows (neurons) n of a ternary matrix of `act[n]` times row n's weights, in order, into `out[0..160]` (natural order): ten
/// vectors of accumulators (five digit planes of 32 lanes) in registers, one fused multiply-add each, put in place at the end.
#[inline(always)]
unsafe fn down_cols_tern<I: Isa>(m: &GlmMat, act: *const f32, g: usize, out: *mut f32) {
    const AHEAD: usize = 8;
    let cs = m.fmt.row_bytes(m.cols);
    let (codes, cb) = (m.codes.as_ptr(), m.book.as_ptr());
    let mut acc = [(I::zero(), I::zero()); 5];
    let mut pad = [0u8; 32];
    for n in 0..m.rows {
        let row = codes.add(n * cs);
        if n + AHEAD < m.rows {
            I::prefetch(row.add(AHEAD * cs + 32 * g));
        }
        let a = I::splat(*act.add(n));
        let t = I::table_tern(cb.add(n * 6));
        let w = I::decode_tern(t, tern_bytes(row, cs, g, &mut pad));
        for d in 0..5 {
            acc[d] = (I::fma(w[d].0, a, acc[d].0), I::fma(w[d].1, a, acc[d].1));
        }
    }
    // plane d lane l is the output 5 l + d of the group
    let mut buf = [0f32; 160];
    for (d, (lo, hi)) in acc.into_iter().enumerate() {
        let mut lanes = [0f32; 32];
        I::store(lanes.as_mut_ptr(), lo);
        I::store(lanes.as_mut_ptr().add(16), hi);
        for (l, v) in lanes.into_iter().enumerate() {
            buf[5 * l + d] = v;
        }
    }
    core::ptr::copy_nonoverlapping(buf.as_ptr(), out, 160.min(m.cols - 160 * g));
}

macro_rules! entries_glm {
    ($m:ident, $isa:ty, $r:literal, $rt:literal $(, $feat:literal)?) => {
        mod $m {
            use super::*;

            $(#[target_feature(enable = $feat)])?
            pub(super) unsafe fn dot2(m: &GlmMat, z: *const f32, r0: usize, r1: usize, s: *mut f32) {
                dot_rows_cbn::<$isa, $r, 2>(m, z, r0, r1, s)
            }
            $(#[target_feature(enable = $feat)])?
            pub(super) unsafe fn dot3(m: &GlmMat, z: *const f32, r0: usize, r1: usize, s: *mut f32) {
                dot_rows_cbn::<$isa, $r, 3>(m, z, r0, r1, s)
            }
            $(#[target_feature(enable = $feat)])?
            pub(super) unsafe fn dot4(m: &GlmMat, z: *const f32, r0: usize, r1: usize, s: *mut f32) {
                dot_rows_cbn::<$isa, $r, 4>(m, z, r0, r1, s)
            }
            $(#[target_feature(enable = $feat)])?
            pub(super) unsafe fn dot_t(m: &GlmMat, z: *const f32, r0: usize, r1: usize, s: *mut f32) {
                dot_rows_tern::<$isa, $rt>(m, z, r0, r1, s)
            }
            $(#[target_feature(enable = $feat)])?
            pub(super) unsafe fn down2(m: &GlmMat, a: *const f32, o0: usize, out: *mut f32) {
                down_cols_cbn::<$isa, 2>(m, a, o0, out)
            }
            $(#[target_feature(enable = $feat)])?
            pub(super) unsafe fn down3(m: &GlmMat, a: *const f32, o0: usize, out: *mut f32) {
                down_cols_cbn::<$isa, 3>(m, a, o0, out)
            }
            $(#[target_feature(enable = $feat)])?
            pub(super) unsafe fn down4(m: &GlmMat, a: *const f32, o0: usize, out: *mut f32) {
                down_cols_cbn::<$isa, 4>(m, a, o0, out)
            }
            $(#[target_feature(enable = $feat)])?
            pub(super) unsafe fn down_t(m: &GlmMat, a: *const f32, g: usize, out: *mut f32) {
                down_cols_tern::<$isa>(m, a, g, out)
            }
            $(#[target_feature(enable = $feat)])?
            pub(super) unsafe fn dot_q(m: &GlmMat, z: *const f32, r0: usize, r1: usize, s: *mut f32) {
                dot_rows_tq::<$isa, $r>(m, z, r0, r1, s)
            }
            $(#[target_feature(enable = $feat)])?
            pub(super) unsafe fn down_q(m: &GlmMat, a: *const f32, o0: usize, out: *mut f32) {
                down_cols_tq::<$isa, 16>(m, a, o0, out)
            }
            $(#[target_feature(enable = $feat)])?
            pub(super) unsafe fn dot_q3(m: &GlmMat, z: *const f32, r0: usize, r1: usize, s: *mut f32) {
                dot_rows_tqk::<$isa, $r, 3>(m, z, r0, r1, s)
            }
            $(#[target_feature(enable = $feat)])?
            pub(super) unsafe fn down_q3(m: &GlmMat, a: *const f32, o0: usize, out: *mut f32) {
                down_cols_tqk::<$isa, 16, 3>(m, a, o0, out)
            }
            $(#[target_feature(enable = $feat)])?
            pub(super) unsafe fn dot_q4(m: &GlmMat, z: *const f32, r0: usize, r1: usize, s: *mut f32) {
                dot_rows_tqk::<$isa, $r, 4>(m, z, r0, r1, s)
            }
            $(#[target_feature(enable = $feat)])?
            pub(super) unsafe fn down_q4(m: &GlmMat, a: *const f32, o0: usize, out: *mut f32) {
                down_cols_tqk::<$isa, 16, 4>(m, a, o0, out)
            }
        }
    };
}

entries_glm!(glm_scalar, Scalar, 1, 1);
#[cfg(target_arch = "aarch64")]
entries_glm!(glm_neon, Neon, 2, 1);
#[cfg(target_arch = "x86_64")]
entries_glm!(glm_avx2, Avx2, 2, 1, "avx2,fma");
#[cfg(target_arch = "x86_64")]
entries_glm!(glm_avx512, Avx512, 4, 2, "avx512f,avx2,fma");

type DotFn = unsafe fn(&GlmMat, *const f32, usize, usize, *mut f32);
type DownFn = unsafe fn(&GlmMat, *const f32, usize, *mut f32);

/// AVX-512 with VNNI and VBMI2: `Avx512`'s arithmetic, the trellis rows' decode in fewer instructions (the window by `vpshldvd`, the byte sum by `vpdpbusd` from -510): the same integers, so the same bits.
#[cfg(target_arch = "x86_64")]
pub(crate) struct Avx512Vnni;

#[cfg(target_arch = "x86_64")]
impl Isa for Avx512Vnni {
    type F = __m512;
    type T = __m512;
    #[inline(always)]
    unsafe fn zero() -> __m512 {
        Avx512::zero()
    }
    #[inline(always)]
    unsafe fn splat(x: f32) -> __m512 {
        Avx512::splat(x)
    }
    #[inline(always)]
    unsafe fn load(p: *const f32) -> __m512 {
        Avx512::load(p)
    }
    #[inline(always)]
    unsafe fn store(p: *mut f32, v: __m512) {
        Avx512::store(p, v)
    }
    #[inline(always)]
    unsafe fn mul(a: __m512, b: __m512) -> __m512 {
        Avx512::mul(a, b)
    }
    #[inline(always)]
    unsafe fn add(a: __m512, b: __m512) -> __m512 {
        Avx512::add(a, b)
    }
    #[inline(always)]
    unsafe fn fma(a: __m512, b: __m512, c: __m512) -> __m512 {
        Avx512::fma(a, b, c)
    }
    #[inline(always)]
    unsafe fn hsum(a: __m512) -> f32 {
        Avx512::hsum(a)
    }
    #[inline(always)]
    unsafe fn decode<const B: u32>(lo: *const u8, hi: *const u8) -> (__m512, __m512) {
        Avx512::decode::<B>(lo, hi)
    }
    #[inline(always)]
    unsafe fn prefetch(p: *const u8) {
        Avx512::prefetch(p)
    }
    #[inline(always)]
    unsafe fn table<const B: u32>(cb: *const u8) -> __m512 {
        Avx512::table::<B>(cb)
    }
    #[inline(always)]
    unsafe fn decode_cb<const B: u32>(t: __m512, run: *const u8) -> (__m512, __m512) {
        Avx512::decode_cb::<B>(t, run)
    }
    #[inline(always)]
    unsafe fn table_tern(cb: *const u8) -> __m512 {
        Avx512::table_tern(cb)
    }
    #[inline(always)]
    unsafe fn decode_tern(t: __m512, bytes: *const u8) -> [(__m512, __m512); 5] {
        Avx512::decode_tern(t, bytes)
    }
    #[inline(always)]
    unsafe fn decode_tq(hi: u32, lo: u32) -> __m512 {
        let sh = _mm512_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30);
        let w = _mm512_shldv_epi32(_mm512_set1_epi32(hi as i32), _mm512_set1_epi32(lo as i32), sh);
        let x = _mm512_add_epi32(_mm512_mullo_epi32(_mm512_srli_epi32::<18>(w), _mm512_set1_epi32(TQ_A as i32)), _mm512_set1_epi32(TQ_B as i32));
        _mm512_cvtepi32_ps(_mm512_dpbusd_epi32(_mm512_set1_epi32(-510), x, _mm512_set1_epi8(1)))
    }
    #[inline(always)]
    unsafe fn decode_tqk<const K: usize, const R: u32>(x: *const u32) -> __m512 {
        tq_values512v(tqk_windows512v::<K, R>(x))
    }
    #[inline(always)]
    unsafe fn decode_tqk_pair<const K: usize>(x: *const u32) -> (__m512, __m512) {
        // (the two runs' words broadcast once: three runs' worth of four at 3 bits, five at 4)
        let b = |q: usize| _mm512_set1_epi32(x.add(q).read_unaligned() as i32);
        let (b0, b1, b2, b3) = (b(0), b(1), b(2), b(3));
        let wa = tqk_win512v::<K, 0>(b0, b1, b2);
        let wb = if K == 3 { tqk_win512v::<K, 16>(b1, b2, b3) } else { tqk_win512v::<K, 0>(b2, b3, b(4)) };
        (tq_values512v(wa), tq_values512v(wb))
    }
}

/// The trellis rows' kernels with `Avx512Vnni`.
#[cfg(target_arch = "x86_64")]
mod glm_avx512v {
    use super::*;

    #[target_feature(enable = "avx512f,avx512bw,avx512vnni,avx512vbmi2,avx2,fma")]
    pub(super) unsafe fn dot_q(m: &GlmMat, z: *const f32, r0: usize, r1: usize, s: *mut f32) {
        dot_rows_tq::<Avx512Vnni, 4>(m, z, r0, r1, s)
    }
    #[target_feature(enable = "avx512f,avx512bw,avx512vnni,avx512vbmi2,avx2,fma")]
    pub(super) unsafe fn down_q(m: &GlmMat, a: *const f32, o0: usize, out: *mut f32) {
        down_cols_tq::<Avx512Vnni, 16>(m, a, o0, out)
    }
    #[target_feature(enable = "avx512f,avx512bw,avx512vnni,avx512vbmi2,avx2,fma")]
    pub(super) unsafe fn dot_q3(m: &GlmMat, z: *const f32, r0: usize, r1: usize, s: *mut f32) {
        dot_rows_tqk::<Avx512Vnni, 4, 3>(m, z, r0, r1, s)
    }
    #[target_feature(enable = "avx512f,avx512bw,avx512vnni,avx512vbmi2,avx2,fma")]
    pub(super) unsafe fn down_q3(m: &GlmMat, a: *const f32, o0: usize, out: *mut f32) {
        down_cols_tqk::<Avx512Vnni, 16, 3>(m, a, o0, out)
    }
    #[target_feature(enable = "avx512f,avx512bw,avx512vnni,avx512vbmi2,avx2,fma")]
    pub(super) unsafe fn dot_q4(m: &GlmMat, z: *const f32, r0: usize, r1: usize, s: *mut f32) {
        dot_rows_tqk::<Avx512Vnni, 4, 4>(m, z, r0, r1, s)
    }
    #[target_feature(enable = "avx512f,avx512bw,avx512vnni,avx512vbmi2,avx2,fma")]
    pub(super) unsafe fn down_q4(m: &GlmMat, a: *const f32, o0: usize, out: *mut f32) {
        down_cols_tqk::<Avx512Vnni, 16, 4>(m, a, o0, out)
    }
}

/// Whether this computer has AVX-512's VNNI and VBMI2 (and BW) besides what `Kernel::Avx512` needs: the trellis rows then take `glm_avx512v`.
#[cfg(target_arch = "x86_64")]
fn tq_vnni() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| is_x86_feature_detected!("avx512vnni") && is_x86_feature_detected!("avx512vbmi2") && is_x86_feature_detected!("avx512bw") && Kernel::Avx512.runs_here())
}

fn glm_fns(kernel: Kernel, fmt: Fmt) -> (DotFn, DownFn) {
    macro_rules! pick {
        ($m:ident) => {
            match fmt {
                Fmt::Cb2 => ($m::dot2 as DotFn, $m::down2 as DownFn),
                Fmt::Cb3 => ($m::dot3 as DotFn, $m::down3 as DownFn),
                Fmt::Cb4 => ($m::dot4 as DotFn, $m::down4 as DownFn),
                Fmt::Tern => ($m::dot_t as DotFn, $m::down_t as DownFn),
                Fmt::Tq => ($m::dot_q as DotFn, $m::down_q as DownFn),
                Fmt::Tq3 => ($m::dot_q3 as DotFn, $m::down_q3 as DownFn),
                Fmt::Tq4 => ($m::dot_q4 as DotFn, $m::down_q4 as DownFn),
            }
        };
    }
    #[cfg(target_arch = "x86_64")]
    if fmt.is_tq() && kernel == Kernel::Avx512 && tq_vnni() {
        use glm_avx512v as v;
        return match fmt {
            Fmt::Tq3 => (v::dot_q3 as DotFn, v::down_q3 as DownFn),
            Fmt::Tq4 => (v::dot_q4 as DotFn, v::down_q4 as DownFn),
            _ => (v::dot_q as DotFn, v::down_q as DownFn),
        };
    }
    match kernel {
        Kernel::Scalar => pick!(glm_scalar),
        #[cfg(target_arch = "aarch64")]
        Kernel::Neon => pick!(glm_neon),
        #[cfg(target_arch = "x86_64")]
        Kernel::Avx2 => pick!(glm_avx2),
        #[cfg(target_arch = "x86_64")]
        Kernel::Avx512 => pick!(glm_avx512),
        _ => unreachable!("a kernel this computer does not run"),
    }
}

/// The sums of rows r0 to r1 of the matrix `m` (no run scales) against `z` (`m.z_len()` values: natural order, or for ternary rows `tern_permute`d) into `sums[0..r1 - r0]`, by `kernel` (which this computer runs).
pub(crate) fn dot_glm(kernel: Kernel, m: &GlmMat, z: &[f32], r0: usize, r1: usize, sums: &mut [f32]) {
    assert!(m.is_valid() && r0 <= r1 && r1 <= m.rows && z.len() >= m.z_len() && sums.len() >= r1 - r0, "rows {r0} to {r1} of {}, z of {} values, {} sums", m.rows, z.len(), sums.len());
    let (f, _) = glm_fns(kernel, m.fmt);
    // SAFETY: the reads and writes are inside the slices (checked above: the arrays have the layout's sizes, a ternary group's loads are of copies where it is partial), and the kernel's instructions are ones this computer has.
    unsafe { f(m, z.as_ptr(), r0, r1, sums.as_mut_ptr()) }
}

/// Outputs `o0` (a multiple of `m.fmt.slice()`) to `o0 + slice` (cut at the width) of the sum over the rows n of the matrix `m` of `act[n]` times row n, into `out[0..]`, by `kernel`.
pub(crate) fn down_glm(kernel: Kernel, m: &GlmMat, act: &[f32], o0: usize, out: &mut [f32]) {
    let slice = m.fmt.slice();
    assert!(m.is_valid() && act.len() >= m.rows && o0 % slice == 0 && o0 < m.cols && out.len() >= slice.min(m.cols - o0), "{} neurons of {} outputs, {} activations, outputs {o0} on, room for {}", m.rows, m.cols, act.len(), out.len());
    assert!(m.fmt == Fmt::Tern || o0 + slice <= m.cols, "a slice of {slice} outputs at {o0} of {}", m.cols);
    let (_, f) = glm_fns(kernel, m.fmt);
    // SAFETY: as in `dot_glm`.
    unsafe { f(m, act.as_ptr(), if m.fmt == Fmt::Tern { o0 / 160 } else { o0 }, out.as_mut_ptr()) }
}

/// The first-product sums of rows r0 to r1 of the codebook matrix `m` against `z` (natural order, `m.cols` values) into `sums[0..r1 - r0]`, by `kernel` (which this computer runs: `Kernel::runs_here`).
pub(crate) fn up_cb(kernel: Kernel, m: &CbView, z: &[f32], r0: usize, r1: usize, sums: &mut [f32]) {
    assert!(m.is_valid() && r0 <= r1 && r1 <= m.rows && z.len() >= m.cols && sums.len() >= r1 - r0, "rows {r0} to {r1} of {}, z of {} values, {} sums", m.rows, z.len(), sums.len());
    let (z, s) = (z.as_ptr(), sums.as_mut_ptr());
    // SAFETY: the reads and writes are inside the slices (checked above: `is_valid` gives each row its bytes, a run's loads stop at its 4 B bytes), and the kernel's instructions are ones this computer has.
    unsafe {
        match kernel {
            Kernel::Scalar => up_cb_scalar(m, z, r0, r1, s),
            #[cfg(target_arch = "aarch64")]
            Kernel::Neon => up_cb_neon(m, z, r0, r1, s),
            #[cfg(target_arch = "x86_64")]
            Kernel::Avx2 => up_cb_avx2(m, z, r0, r1, s),
            #[cfg(target_arch = "x86_64")]
            Kernel::Avx512 => up_cb_avx512(m, z, r0, r1, s),
            _ => unreachable!("a kernel this computer does not run"),
        }
    }
}

/// The outputs `acc[0..cols]` (natural order) of the active neurons (row, activation) of the codebook matrix `m`, by `kernel`.
pub(crate) fn down_cb(kernel: Kernel, m: &CbView, active: &[(u32, f32)], acc: &mut [f32]) {
    assert!(m.is_valid() && acc.len() >= m.cols && active.iter().all(|&(j, _)| (j as usize) < m.rows), "{} outputs of {} and a neuron past the {} rows", acc.len(), m.cols, m.rows);
    let a = acc.as_mut_ptr();
    // SAFETY: as in `up_cb`.
    unsafe {
        match kernel {
            Kernel::Scalar => down_cb_scalar(m, active, a),
            #[cfg(target_arch = "aarch64")]
            Kernel::Neon => down_cb_neon(m, active, a),
            #[cfg(target_arch = "x86_64")]
            Kernel::Avx2 => down_cb_avx2(m, active, a),
            #[cfg(target_arch = "x86_64")]
            Kernel::Avx512 => down_cb_avx512(m, active, a),
            _ => unreachable!("a kernel this computer does not run"),
        }
    }
}

// ---- scalar ---------------------------------------------------------------------------------------------------------------------------------------

/// The reference: the codes read by the layout's formulas (weight by weight, in the order of the lanes), the arithmetic one value at a time.
pub(crate) struct Scalar;

impl Isa for Scalar {
    type F = [f32; 16];
    type T = [f32; 16];

    #[inline(always)]
    unsafe fn zero() -> [f32; 16] {
        [0.0; 16]
    }

    #[inline(always)]
    unsafe fn splat(x: f32) -> [f32; 16] {
        [x; 16]
    }

    #[inline(always)]
    unsafe fn load(p: *const f32) -> [f32; 16] {
        p.cast::<[f32; 16]>().read_unaligned()
    }

    #[inline(always)]
    unsafe fn store(p: *mut f32, v: [f32; 16]) {
        p.cast::<[f32; 16]>().write_unaligned(v)
    }

    #[inline(always)]
    unsafe fn mul(a: [f32; 16], b: [f32; 16]) -> [f32; 16] {
        core::array::from_fn(|i| a[i] * b[i])
    }

    #[inline(always)]
    unsafe fn add(a: [f32; 16], b: [f32; 16]) -> [f32; 16] {
        core::array::from_fn(|i| a[i] + b[i])
    }

    #[inline(always)]
    unsafe fn fma(a: [f32; 16], b: [f32; 16], c: [f32; 16]) -> [f32; 16] {
        core::array::from_fn(|i| a[i].mul_add(b[i], c[i]))
    }

    #[inline(always)]
    unsafe fn hsum(a: [f32; 16]) -> f32 {
        let b: [f32; 8] = core::array::from_fn(|l| a[l] + a[l + 8]);
        let c: [f32; 4] = core::array::from_fn(|l| b[l] + b[l + 4]);
        let d: [f32; 2] = core::array::from_fn(|l| c[l] + c[l + 2]);
        d[0] + d[1]
    }

    #[inline(always)]
    unsafe fn decode<const B: u32>(lo: *const u8, hi: *const u8) -> ([f32; 16], [f32; 16]) {
        let word = |p: *const u8, q: usize| u32::from_le_bytes(p.add(4 * q).cast::<[u8; 4]>().read_unaligned());
        let lw: [u32; 4] = core::array::from_fn(|q| word(lo, q));
        let hw: [u32; 2] = core::array::from_fn(|q| if (B == 5 && q == 0) || B == 6 { word(hi, q) } else { 0 });
        let value = |i: usize| {
            let (pr, h) = (i / 2, i % 2);
            let mut u = (lw[pr / 4] >> (4 * (pr % 4) + 16 * h)) & 15;
            u |= match B {
                6 => ((hw[pr / 8] >> (2 * (pr % 8) + 16 * h)) & 3) << 4,
                5 => ((hw[0] >> (pr + 16 * h)) & 1) << 4,
                _ => 0,
            };
            (u as i32 - (1 << (B - 1))) as f32
        };
        (core::array::from_fn(|p| value(LANE_LO[p])), core::array::from_fn(|p| value(LANE_HI[p])))
    }

    #[inline(always)]
    unsafe fn prefetch(_: *const u8) {}

    #[inline(always)]
    unsafe fn table<const B: u32>(cb: *const u8) -> [f32; 16] {
        core::array::from_fn(|c| if c < 1 << B { bf16_to_f32(u16::from_le(cb.add(2 * c).cast::<u16>().read_unaligned())) } else { 0.0 })
    }

    #[inline(always)]
    unsafe fn decode_cb<const B: u32>(t: [f32; 16], run: *const u8) -> ([f32; 16], [f32; 16]) {
        // (the run's bytes, a zero byte after them: code j is the B bits from stream bit B j, two bytes read)
        let mut bytes = [0u8; 4 * 4 + 1];
        for (i, b) in bytes.iter_mut().take(4 * B as usize).enumerate() {
            *b = run.add(i).read();
        }
        let code = |j: usize| {
            let at = B as usize * j;
            let w = bytes[at / 8] as u32 | (bytes[at / 8 + 1] as u32) << 8;
            t[(w >> (at % 8)) as usize & ((1 << B) - 1)]
        };
        (core::array::from_fn(code), core::array::from_fn(|p| code(16 + p)))
    }

    #[inline(always)]
    unsafe fn table_tern(cb: *const u8) -> [f32; 16] {
        core::array::from_fn(|c| if c < 3 { bf16_to_f32(u16::from_le(cb.add(2 * c).cast::<u16>().read_unaligned())) } else { 0.0 })
    }

    #[inline(always)]
    unsafe fn decode_tern(t: [f32; 16], bytes: *const u8) -> [([f32; 16], [f32; 16]); 5] {
        core::array::from_fn(|d| (core::array::from_fn(|l| t[tern_digit(bytes.add(l).read(), d)]), core::array::from_fn(|l| t[tern_digit(bytes.add(16 + l).read(), d)])))
    }

    #[inline(always)]
    unsafe fn decode_tq(hi: u32, lo: u32) -> [f32; 16] {
        core::array::from_fn(|i| tq_lane(hi, lo, i))
    }
}

// ---- NEON -----------------------------------------------------------------------------------------------------------------------------------------

/// aarch64: a vector is 4 registers of 4 lanes, the lanes p = 4 q + r of the plane's byte p in register q.
#[cfg(target_arch = "aarch64")]
pub(crate) struct Neon;

/// The bytes 3 and 2 of the fp32 of (code - 8) for a four-bit code (the other two bytes of these small integers are 0).
#[cfg(target_arch = "aarch64")]
const fn four_bit_bytes(shift: u32) -> [u8; 16] {
    let mut o = [0u8; 16];
    let mut u = 0;
    while u < 16 {
        let bits = ((u as i32 - 8) as f32).to_bits();
        assert!(bits & 0xFFFF == 0);
        o[u] = (bits >> shift) as u8;
        u += 1;
    }
    o
}

#[cfg(target_arch = "aarch64")]
const F4_BYTE3: [u8; 16] = four_bit_bytes(24);
#[cfg(target_arch = "aarch64")]
const F4_BYTE2: [u8; 16] = four_bit_bytes(16);

// The codebook runs' codes as bytes, one a lane, for `tbl`: a plane (0: the run's codes 0 to 15, 1: 16 to 31). Two bits: code j of the plane is in byte 4 plane + j / 4 of the run's 8 (the run's bytes twice make a
// 16-byte register) at bit 2 (j mod 4), brought down by shifting right (negative counts). Three bits: code j is bits 3 (j mod 8) .. +2 of the 24-bit group 2 plane + j / 8 (bytes 3 g .. 3 g + 2 of the run's 12): the
// byte its first bit is in, and the next byte (the last, 11, is its own next) for when it straddles, `shr` and `shl` putting the two parts of the code in place (a byte shifted left by 8 is 0).
#[cfg(target_arch = "aarch64")]
const fn cb2_idx(plane: usize) -> [u8; 16] {
    let (mut o, mut j) = ([0u8; 16], 0);
    while j < 16 {
        o[j] = (4 * plane + j / 4) as u8;
        j += 1;
    }
    o
}

#[cfg(target_arch = "aarch64")]
const fn cb2_shr() -> [i8; 16] {
    let (mut o, mut j) = ([0i8; 16], 0);
    while j < 16 {
        o[j] = -2 * (j % 4) as i8;
        j += 1;
    }
    o
}

#[cfg(target_arch = "aarch64")]
const fn cb3_idx(plane: usize, next: bool) -> [u8; 16] {
    let (mut o, mut j) = ([0u8; 16], 0);
    while j < 16 {
        let first = 3 * (2 * plane + j / 8) + 3 * (j % 8) / 8;
        o[j] = if next && first < 11 { first as u8 + 1 } else { first as u8 };
        j += 1;
    }
    o
}

#[cfg(target_arch = "aarch64")]
const fn cb3_shift(left: bool) -> [i8; 16] {
    let (mut o, mut j) = ([0i8; 16], 0);
    while j < 16 {
        let s = (3 * (j % 8) % 8) as i8;
        o[j] = if left { 8 - s } else { -s };
        j += 1;
    }
    o
}

/// The byte indices of a table of up to 8 bf16 entries that pick the entries' high bytes (`first` 1) or their low bytes (0).
#[cfg(target_arch = "aarch64")]
const fn bf16_bytes(first: u8) -> [u8; 16] {
    let (mut o, mut j) = ([0u8; 16], 0);
    while j < 8 {
        o[j] = 2 * j as u8 + first;
        j += 1;
    }
    o
}

#[cfg(target_arch = "aarch64")]
const CB2_IDX: [[u8; 16]; 2] = [cb2_idx(0), cb2_idx(1)];
#[cfg(target_arch = "aarch64")]
const CB2_SHR: [i8; 16] = cb2_shr();
#[cfg(target_arch = "aarch64")]
const CB3_IDX: [[[u8; 16]; 2]; 2] = [[cb3_idx(0, false), cb3_idx(0, true)], [cb3_idx(1, false), cb3_idx(1, true)]];
#[cfg(target_arch = "aarch64")]
const CB3_SHR: [i8; 16] = cb3_shift(false);
#[cfg(target_arch = "aarch64")]
const CB3_SHL: [i8; 16] = cb3_shift(true);
#[cfg(target_arch = "aarch64")]
const BF16_HI: [u8; 16] = bf16_bytes(1);
#[cfg(target_arch = "aarch64")]
const BF16_LO: [u8; 16] = bf16_bytes(0);

#[cfg(target_arch = "aarch64")]
impl Neon {
    /// 16 four-bit codes (bytes) minus 8, as fp32, by two table lookups of the top bytes of the floats and a widening shift.
    #[inline(always)]
    unsafe fn widen4(u: uint8x16_t) -> [float32x4_t; 4] {
        Neon::from_top(vqtbl1q_u8(vld1q_u8(F4_BYTE3.as_ptr()), u), vqtbl1q_u8(vld1q_u8(F4_BYTE2.as_ptr()), u))
    }

    /// 16 fp32 from their top two bytes (bytes 3 and 2: `b3` and `b2`; the other two bytes are 0): the high halves widened by a shift.
    #[inline(always)]
    unsafe fn from_top(b3: uint8x16_t, b2: uint8x16_t) -> [float32x4_t; 4] {
        let (p0, p1) = (vreinterpretq_u16_u8(vzip1q_u8(b2, b3)), vreinterpretq_u16_u8(vzip2q_u8(b2, b3)));
        [
            vreinterpretq_f32_u32(vshll_n_u16::<16>(vget_low_u16(p0))),
            vreinterpretq_f32_u32(vshll_high_n_u16::<16>(p0)),
            vreinterpretq_f32_u32(vshll_n_u16::<16>(vget_low_u16(p1))),
            vreinterpretq_f32_u32(vshll_high_n_u16::<16>(p1)),
        ]
    }

    /// 16 codes (bytes) minus the bias, as fp32.
    #[inline(always)]
    unsafe fn widen<const B: u32>(u: uint8x16_t) -> [float32x4_t; 4] {
        let s = vsubq_s8(vreinterpretq_s8_u8(u), vdupq_n_s8(1 << (B - 1)));
        let (w0, w1) = (vmovl_s8(vget_low_s8(s)), vmovl_high_s8(s));
        [vcvtq_f32_s32(vmovl_s16(vget_low_s16(w0))), vcvtq_f32_s32(vmovl_high_s16(w0)), vcvtq_f32_s32(vmovl_s16(vget_low_s16(w1))), vcvtq_f32_s32(vmovl_high_s16(w1))]
    }
}

#[cfg(target_arch = "aarch64")]
impl Isa for Neon {
    type F = [float32x4_t; 4];
    type T = (uint8x16_t, uint8x16_t);

    #[inline(always)]
    unsafe fn zero() -> Self::F {
        [vdupq_n_f32(0.0); 4]
    }

    #[inline(always)]
    unsafe fn splat(x: f32) -> Self::F {
        [vdupq_n_f32(x); 4]
    }

    #[inline(always)]
    unsafe fn load(p: *const f32) -> Self::F {
        [vld1q_f32(p), vld1q_f32(p.add(4)), vld1q_f32(p.add(8)), vld1q_f32(p.add(12))]
    }

    #[inline(always)]
    unsafe fn store(p: *mut f32, v: Self::F) {
        vst1q_f32(p, v[0]);
        vst1q_f32(p.add(4), v[1]);
        vst1q_f32(p.add(8), v[2]);
        vst1q_f32(p.add(12), v[3]);
    }

    #[inline(always)]
    unsafe fn mul(a: Self::F, b: Self::F) -> Self::F {
        [vmulq_f32(a[0], b[0]), vmulq_f32(a[1], b[1]), vmulq_f32(a[2], b[2]), vmulq_f32(a[3], b[3])]
    }

    #[inline(always)]
    unsafe fn add(a: Self::F, b: Self::F) -> Self::F {
        [vaddq_f32(a[0], b[0]), vaddq_f32(a[1], b[1]), vaddq_f32(a[2], b[2]), vaddq_f32(a[3], b[3])]
    }

    #[inline(always)]
    unsafe fn fma(a: Self::F, b: Self::F, c: Self::F) -> Self::F {
        [vfmaq_f32(c[0], a[0], b[0]), vfmaq_f32(c[1], a[1], b[1]), vfmaq_f32(c[2], a[2], b[2]), vfmaq_f32(c[3], a[3], b[3])]
    }

    #[inline(always)]
    unsafe fn hsum(a: Self::F) -> f32 {
        let c = vaddq_f32(vaddq_f32(a[0], a[2]), vaddq_f32(a[1], a[3]));
        let d = vaddq_f32(c, vextq_f32::<2>(c, c));
        vgetq_lane_f32::<0>(d) + vgetq_lane_f32::<1>(d)
    }

    #[inline(always)]
    unsafe fn decode<const B: u32>(lo: *const u8, hi: *const u8) -> (Self::F, Self::F) {
        let v = vld1q_u8(lo);
        let (mut a, mut b) = (vandq_u8(v, vdupq_n_u8(0x0F)), vshrq_n_u8::<4>(v));
        if B == 4 {
            return (Neon::widen4(a), Neon::widen4(b));
        }
        if B == 5 {
            let x = vqtbl1q_u8(vreinterpretq_u8_u32(vdupq_n_u32(hi.cast::<u32>().read_unaligned())), vld1q_u8(IDX5.as_ptr()));
            a = vorrq_u8(a, vandq_u8(vtstq_u8(x, vld1q_u8(MASK5_LO.as_ptr())), vdupq_n_u8(0x10)));
            b = vorrq_u8(b, vandq_u8(vtstq_u8(x, vld1q_u8(MASK5_HI.as_ptr())), vdupq_n_u8(0x10)));
        } else if B == 6 {
            let h = vld1_u8(hi);
            let x = vqtbl1q_u8(vcombine_u8(h, h), vld1q_u8(IDX6.as_ptr()));
            let (sl, sh) = (vnegq_s8(vreinterpretq_s8_u8(vld1q_u8(SHIFT6_LO.as_ptr()))), vnegq_s8(vreinterpretq_s8_u8(vld1q_u8(SHIFT6_HI.as_ptr()))));
            a = vorrq_u8(a, vshlq_n_u8::<4>(vandq_u8(vshlq_u8(x, sl), vdupq_n_u8(3))));
            b = vorrq_u8(b, vshlq_n_u8::<4>(vandq_u8(vshlq_u8(x, sh), vdupq_n_u8(3))));
        }
        (Neon::widen::<B>(a), Neon::widen::<B>(b))
    }

    #[inline(always)]
    unsafe fn prefetch(p: *const u8) {
        core::arch::asm!("prfm pldl1keep, [{p}]", p = in(reg) p, options(nostack, readonly, preserves_flags));
    }

    /// The entries' high bytes in one register and their low bytes in another (a code indexes both: `tbl`).
    #[inline(always)]
    unsafe fn table<const B: u32>(cb: *const u8) -> Self::T {
        if B == 4 {
            // (16 entries: the low bytes and the high bytes of the bf16 numbers apart)
            let p = vld2q_u8(cb);
            return (p.1, p.0);
        }
        let v = if B == 2 { vcombine_u8(vld1_u8(cb), vdup_n_u8(0)) } else { vld1q_u8(cb) };
        (vqtbl1q_u8(v, vld1q_u8(BF16_HI.as_ptr())), vqtbl1q_u8(v, vld1q_u8(BF16_LO.as_ptr())))
    }

    #[inline(always)]
    unsafe fn table_tern(cb: *const u8) -> Self::T {
        let mut b = [0u8; 16];
        core::ptr::copy_nonoverlapping(cb, b.as_mut_ptr(), 6);
        let v = vld1q_u8(b.as_ptr());
        (vqtbl1q_u8(v, vld1q_u8(BF16_HI.as_ptr())), vqtbl1q_u8(v, vld1q_u8(BF16_LO.as_ptr())))
    }

    #[inline(always)]
    unsafe fn decode_tern(t: Self::T, bytes: *const u8) -> [(Self::F, Self::F); 5] {
        let mut idx = [[0u8; 32]; 5];
        for l in 0..32 {
            let b = bytes.add(l).read();
            for (d, plane) in idx.iter_mut().enumerate() {
                plane[l] = tern_digit(b, d) as u8;
            }
        }
        let look = |p: *const u8| Neon::from_top(vqtbl1q_u8(t.0, vld1q_u8(p)), vqtbl1q_u8(t.1, vld1q_u8(p)));
        core::array::from_fn(|d| (look(idx[d].as_ptr()), look(idx[d].as_ptr().add(16))))
    }

    #[inline(always)]
    unsafe fn decode_cb<const B: u32>(t: Self::T, run: *const u8) -> (Self::F, Self::F) {
        if B == 4 {
            // (the low nibbles are the even codes of the run's 16 bytes, the high nibbles the odd ones: zipped, the codes in order)
            let v = vld1q_u8(run);
            let (a, b) = (vandq_u8(v, vdupq_n_u8(0x0F)), vshrq_n_u8::<4>(v));
            let look = |u: uint8x16_t| Neon::from_top(vqtbl1q_u8(t.0, u), vqtbl1q_u8(t.1, u));
            return (look(vzip1q_u8(a, b)), look(vzip2q_u8(a, b)));
        }
        let (lo, hi) = if B == 2 {
            let v = vld1_u8(run);
            let (v, shr, three) = (vcombine_u8(v, v), vld1q_s8(CB2_SHR.as_ptr()), vdupq_n_u8(3));
            let plane = |i: &[u8; 16]| vandq_u8(vshlq_u8(vqtbl1q_u8(v, vld1q_u8(i.as_ptr())), shr), three);
            (plane(&CB2_IDX[0]), plane(&CB2_IDX[1]))
        } else {
            let v = vcombine_u8(vld1_u8(run), vreinterpret_u8_u32(vdup_n_u32(run.add(8).cast::<u32>().read_unaligned())));
            let (shr, shl, seven) = (vld1q_s8(CB3_SHR.as_ptr()), vld1q_s8(CB3_SHL.as_ptr()), vdupq_n_u8(7));
            let plane = |i: &[[u8; 16]; 2]| {
                let (a, b) = (vqtbl1q_u8(v, vld1q_u8(i[0].as_ptr())), vqtbl1q_u8(v, vld1q_u8(i[1].as_ptr())));
                vandq_u8(vorrq_u8(vshlq_u8(a, shr), vshlq_u8(b, shl)), seven)
            };
            (plane(&CB3_IDX[0]), plane(&CB3_IDX[1]))
        };
        let look = |u: uint8x16_t| Neon::from_top(vqtbl1q_u8(t.0, u), vqtbl1q_u8(t.1, u));
        (look(lo), look(hi))
    }

    #[inline(always)]
    unsafe fn decode_tq(hi: u32, lo: u32) -> Self::F {
        let v: [f32; 16] = core::array::from_fn(|i| tq_lane(hi, lo, i));
        Self::load(v.as_ptr())
    }
}

// ---- x86-64 ---------------------------------------------------------------------------------------------------------------------------------------

/// A run's codes minus the bias, as 16 signed bytes a plane (the low-nibble plane, then the high-nibble one): AVX2 and AVX-512 widen them to fp32 their own ways.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn x86_planes<const B: u32>(lo: *const u8, hi: *const u8) -> (__m128i, __m128i) {
    let v = _mm_loadu_si128(lo.cast());
    let m = _mm_set1_epi8(0x0F);
    let (mut a, mut b) = (_mm_and_si128(v, m), _mm_and_si128(_mm_srli_epi16::<4>(v), m));
    if B == 5 {
        let x = _mm_shuffle_epi8(_mm_cvtsi32_si128(hi.cast::<i32>().read_unaligned()), _mm_loadu_si128(IDX5.as_ptr().cast()));
        let (ml, mh) = (_mm_loadu_si128(MASK5_LO.as_ptr().cast()), _mm_loadu_si128(MASK5_HI.as_ptr().cast()));
        let bit = _mm_set1_epi8(0x10);
        a = _mm_or_si128(a, _mm_and_si128(_mm_cmpeq_epi8(_mm_and_si128(x, ml), ml), bit));
        b = _mm_or_si128(b, _mm_and_si128(_mm_cmpeq_epi8(_mm_and_si128(x, mh), mh), bit));
    } else if B == 6 {
        let x = _mm_shuffle_epi8(_mm_loadl_epi64(hi.cast()), _mm_loadu_si128(IDX6.as_ptr().cast()));
        let sel = _mm_loadu_si128(SEL6.as_ptr().cast());
        let three = _mm_set1_epi8(3);
        // (the shifts are 0 or 4 for the low plane, 2 or 6 for the high: a 16-bit shift, then the bits of the byte that are not the field's cleared)
        let fl = _mm_blendv_epi8(x, _mm_srli_epi16::<4>(x), sel);
        let x2 = _mm_srli_epi16::<2>(x);
        let fh = _mm_blendv_epi8(x2, _mm_srli_epi16::<4>(x2), sel);
        a = _mm_or_si128(a, _mm_slli_epi16::<4>(_mm_and_si128(fl, three)));
        b = _mm_or_si128(b, _mm_slli_epi16::<4>(_mm_and_si128(fh, three)));
    }
    let bias = _mm_set1_epi8(1 << (B - 1));
    (_mm_sub_epi8(a, bias), _mm_sub_epi8(b, bias))
}

/// Eight lanes of `decode_tq` (AVX2): the words' 14 bits from bit `sh` of hi:lo, hashed, the bytes summed (byte pairs as 16 bits, then the pairs), less 510.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn tq8(h: __m256i, l: __m256i, sh: __m256i) -> __m256 {
    let w = _mm256_or_si256(_mm256_sllv_epi32(h, sh), _mm256_srlv_epi32(l, _mm256_sub_epi32(_mm256_set1_epi32(32), sh)));
    let x = _mm256_add_epi32(_mm256_mullo_epi32(_mm256_srli_epi32::<18>(w), _mm256_set1_epi32(TQ_A as i32)), _mm256_set1_epi32(TQ_B as i32));
    let s = _mm256_madd_epi16(_mm256_maddubs_epi16(x, _mm256_set1_epi8(1)), _mm256_set1_epi16(1));
    _mm256_sub_ps(_mm256_cvtepi32_ps(s), _mm256_set1_ps(510.0))
}

/// x86-64 with AVX2 and FMA: a vector is 2 registers of 8 lanes.
#[cfg(target_arch = "x86_64")]
pub(crate) struct Avx2;

#[cfg(target_arch = "x86_64")]
impl Isa for Avx2 {
    type F = (__m256, __m256);
    type T = (__m256, __m256);

    #[inline(always)]
    unsafe fn zero() -> Self::F {
        (_mm256_setzero_ps(), _mm256_setzero_ps())
    }

    #[inline(always)]
    unsafe fn splat(x: f32) -> Self::F {
        (_mm256_set1_ps(x), _mm256_set1_ps(x))
    }

    #[inline(always)]
    unsafe fn load(p: *const f32) -> Self::F {
        (_mm256_loadu_ps(p), _mm256_loadu_ps(p.add(8)))
    }

    #[inline(always)]
    unsafe fn store(p: *mut f32, v: Self::F) {
        _mm256_storeu_ps(p, v.0);
        _mm256_storeu_ps(p.add(8), v.1);
    }

    #[inline(always)]
    unsafe fn mul(a: Self::F, b: Self::F) -> Self::F {
        (_mm256_mul_ps(a.0, b.0), _mm256_mul_ps(a.1, b.1))
    }

    #[inline(always)]
    unsafe fn add(a: Self::F, b: Self::F) -> Self::F {
        (_mm256_add_ps(a.0, b.0), _mm256_add_ps(a.1, b.1))
    }

    #[inline(always)]
    unsafe fn fma(a: Self::F, b: Self::F, c: Self::F) -> Self::F {
        (_mm256_fmadd_ps(a.0, b.0, c.0), _mm256_fmadd_ps(a.1, b.1, c.1))
    }

    #[inline(always)]
    unsafe fn hsum(a: Self::F) -> f32 {
        let b = _mm256_add_ps(a.0, a.1);
        hsum8(b)
    }

    #[inline(always)]
    unsafe fn decode<const B: u32>(lo: *const u8, hi: *const u8) -> (Self::F, Self::F) {
        let (a, b) = x86_planes::<B>(lo, hi);
        let widen = |x: __m128i| (_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(x)), _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128::<8>(x))));
        (widen(a), widen(b))
    }

    #[inline(always)]
    unsafe fn prefetch(p: *const u8) {
        _mm_prefetch::<_MM_HINT_T0>(p.cast());
    }

    /// The row's table in registers of 8 lanes (a table of 4 entries repeated; of 16, entries 0 to 7 and 8 to 15): `vpermps` then takes an entry by the low three bits of a lane, so a code's neighbours' bits above its own
    /// do no harm; at four bits the fourth bit of the code picks between the two registers.
    #[inline(always)]
    unsafe fn table<const B: u32>(cb: *const u8) -> Self::T {
        let v = if B == 2 {
            let q = _mm_loadl_epi64(cb.cast());
            _mm_unpacklo_epi64(q, q)
        } else {
            _mm_loadu_si128(cb.cast())
        };
        let f = |v: __m128i| _mm256_castsi256_ps(_mm256_slli_epi32::<16>(_mm256_cvtepu16_epi32(v)));
        (f(v), if B == 4 { f(_mm_loadu_si128(cb.add(16).cast())) } else { _mm256_setzero_ps() })
    }

    #[inline(always)]
    unsafe fn decode_cb<const B: u32>(t: Self::T, run: *const u8) -> (Self::F, Self::F) {
        let w = x86_windows::<B>(run);
        let look = |window: i32, shifts: __m256i| {
            let idx = _mm256_srlv_epi32(_mm256_set1_epi32(window), shifts);
            if B == 4 {
                _mm256_blendv_ps(_mm256_permutevar8x32_ps(t.0, idx), _mm256_permutevar8x32_ps(t.1, idx), _mm256_castsi256_ps(_mm256_slli_epi32::<28>(idx)))
            } else {
                _mm256_permutevar8x32_ps(t.0, idx)
            }
        };
        if B == 2 {
            let (s0, s1) = (_mm256_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14), _mm256_setr_epi32(16, 18, 20, 22, 24, 26, 28, 30));
            ((look(w[0], s0), look(w[0], s1)), (look(w[1], s0), look(w[1], s1)))
        } else {
            let s = if B == 3 { _mm256_setr_epi32(0, 3, 6, 9, 12, 15, 18, 21) } else { _mm256_setr_epi32(0, 4, 8, 12, 16, 20, 24, 28) };
            ((look(w[0], s), look(w[1], s)), (look(w[2], s), look(w[3], s)))
        }
    }

    #[inline(always)]
    unsafe fn table_tern(cb: *const u8) -> Self::T {
        let v = (cb.cast::<u32>().read_unaligned() as u64) | ((cb.add(4).cast::<u16>().read_unaligned() as u64) << 32);
        (_mm256_castsi256_ps(_mm256_slli_epi32::<16>(_mm256_cvtepu16_epi32(_mm_set1_epi64x(v as i64)))), _mm256_setzero_ps())
    }

    #[inline(always)]
    unsafe fn decode_tern(t: Self::T, bytes: *const u8) -> [(Self::F, Self::F); 5] {
        let (c171, three) = (_mm256_set1_epi32(171), _mm256_set1_epi32(3));
        // eight bytes' five digits, a digit a register of 8 lanes (x / 3 as (x * 171) >> 9); a macro and no closure, so that every intrinsic is inlined in the caller's instruction set
        macro_rules! digits {
            ($p:expr) => {{
                let w = _mm256_cvtepu8_epi32(_mm_loadl_epi64($p.cast()));
                let q0 = _mm256_srli_epi32::<9>(_mm256_mullo_epi32(w, c171));
                let q1 = _mm256_srli_epi32::<9>(_mm256_mullo_epi32(q0, c171));
                let q2 = _mm256_srli_epi32::<9>(_mm256_mullo_epi32(q1, c171));
                let q3 = _mm256_srli_epi32::<9>(_mm256_mullo_epi32(q2, c171));
                [
                    _mm256_sub_epi32(w, _mm256_mullo_epi32(q0, three)),
                    _mm256_sub_epi32(q0, _mm256_mullo_epi32(q1, three)),
                    _mm256_sub_epi32(q1, _mm256_mullo_epi32(q2, three)),
                    _mm256_sub_epi32(q2, _mm256_mullo_epi32(q3, three)),
                    q3,
                ]
            }};
        }
        let g = [digits!(bytes), digits!(bytes.add(8)), digits!(bytes.add(16)), digits!(bytes.add(24))];
        let z = _mm256_setzero_ps();
        let mut out = [(( z, z), (z, z)); 5];
        for d in 0..5 {
            out[d] = ((_mm256_permutevar8x32_ps(t.0, g[0][d]), _mm256_permutevar8x32_ps(t.0, g[1][d])), (_mm256_permutevar8x32_ps(t.0, g[2][d]), _mm256_permutevar8x32_ps(t.0, g[3][d])));
        }
        out
    }

    #[inline(always)]
    unsafe fn decode_tq(hi: u32, lo: u32) -> Self::F {
        let (h, l) = (_mm256_set1_epi32(hi as i32), _mm256_set1_epi32(lo as i32));
        (tq8(h, l, _mm256_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14)), tq8(h, l, _mm256_setr_epi32(16, 18, 20, 22, 24, 26, 28, 30)))
    }

    #[inline(always)]
    unsafe fn decode_tqk<const K: usize, const R: u32>(x: *const u32) -> Self::F {
        // (a half of 8 lanes whose states start in one word takes that word and the next broadcast; at 3 bits one half's start in two, and takes them by a blend: no lane permutes)
        let t = tqk_tabs::<K, R>();
        let b = |q: usize| _mm256_set1_epi32(x.add(q).read_unaligned() as i32);
        let sh = |o: usize| _mm256_loadu_si256(t[1][o..].as_ptr().cast());
        let mixed = |o: usize| {
            let m = _mm256_cmpgt_epi32(_mm256_loadu_si256(t[0][o..].as_ptr().cast()), _mm256_setzero_si256());
            tq8(_mm256_blendv_epi8(b(0), b(1), m), _mm256_blendv_epi8(b(1), b(2), m), sh(o))
        };
        match (K, R) {
            (4, _) => (tq8(b(0), b(1), sh(0)), tq8(b(1), b(2), sh(8))),
            (_, 0) => (tq8(b(0), b(1), sh(0)), mixed(8)),
            _ => (mixed(0), tq8(b(1), b(2), sh(8))),
        }
    }
}

/// A codebook run's bits as 32-bit windows that start at a group's first code: the groups of 16 codes at two bits (windows 0 and 1), of 8 at three bits (windows 0 to 3: the groups' 24 bits and, above them, the
/// next bits of the run or zeros). Only the run's own 4 B bytes are read. (Built for the tests on any processor: they check the windows and the table lookups' bits against the codes.)
#[cfg(any(target_arch = "x86_64", test))]
#[inline(always)]
pub(crate) unsafe fn x86_windows<const B: u32>(run: *const u8) -> [i32; 4] {
    if B == 4 {
        // (four bits: windows of 8 codes, the run's 16 bytes as four words)
        return core::array::from_fn(|q| run.add(4 * q).cast::<i32>().read_unaligned());
    }
    let a = run.cast::<u64>().read_unaligned();
    if B == 2 {
        [a as u32 as i32, (a >> 32) as u32 as i32, 0, 0]
    } else {
        let b = run.add(8).cast::<u32>().read_unaligned() as u64;
        [a as u32 as i32, (a >> 24) as u32 as i32, ((a >> 48) | (b << 16)) as u32 as i32, (b >> 8) as u32 as i32]
    }
}

/// The 8 lanes b of a tree's first step, added as its others: l with l + 4, l + 2, l + 1.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn hsum8(b: __m256) -> f32 {
    let c = _mm_add_ps(_mm256_castps256_ps128(b), _mm256_extractf128_ps::<1>(b));
    let d = _mm_add_ps(c, _mm_movehl_ps(c, c));
    _mm_cvtss_f32(_mm_add_ss(d, _mm_shuffle_ps::<1>(d, d)))
}

/// x86-64 with AVX-512 (the foundation instructions): a vector is one register of 16 lanes.
#[cfg(target_arch = "x86_64")]
pub(crate) struct Avx512;

#[cfg(target_arch = "x86_64")]
impl Isa for Avx512 {
    type F = __m512;
    type T = __m512;

    #[inline(always)]
    unsafe fn zero() -> Self::F {
        _mm512_setzero_ps()
    }

    #[inline(always)]
    unsafe fn splat(x: f32) -> Self::F {
        _mm512_set1_ps(x)
    }

    #[inline(always)]
    unsafe fn load(p: *const f32) -> Self::F {
        _mm512_loadu_ps(p)
    }

    #[inline(always)]
    unsafe fn store(p: *mut f32, v: Self::F) {
        _mm512_storeu_ps(p, v)
    }

    #[inline(always)]
    unsafe fn mul(a: Self::F, b: Self::F) -> Self::F {
        _mm512_mul_ps(a, b)
    }

    #[inline(always)]
    unsafe fn add(a: Self::F, b: Self::F) -> Self::F {
        _mm512_add_ps(a, b)
    }

    #[inline(always)]
    unsafe fn fma(a: Self::F, b: Self::F, c: Self::F) -> Self::F {
        _mm512_fmadd_ps(a, b, c)
    }

    #[inline(always)]
    unsafe fn hsum(a: Self::F) -> f32 {
        let upper = _mm256_castpd_ps(_mm512_extractf64x4_pd::<1>(_mm512_castps_pd(a)));
        hsum8(_mm256_add_ps(_mm512_castps512_ps256(a), upper))
    }

    #[inline(always)]
    unsafe fn decode<const B: u32>(lo: *const u8, hi: *const u8) -> (Self::F, Self::F) {
        let (a, b) = x86_planes::<B>(lo, hi);
        (_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(a)), _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(b)))
    }

    #[inline(always)]
    unsafe fn prefetch(p: *const u8) {
        _mm_prefetch::<_MM_HINT_T0>(p.cast());
    }

    /// The row's table in a register of 16 lanes (4 entries four times, or 8 entries twice): `vpermps` takes an entry by the low four bits of a lane, so the bits above a code's own do no harm.
    #[inline(always)]
    unsafe fn table<const B: u32>(cb: *const u8) -> Self::T {
        let v = if B == 2 {
            _mm256_set1_epi64x(cb.cast::<i64>().read_unaligned())
        } else if B == 3 {
            _mm256_broadcastsi128_si256(_mm_loadu_si128(cb.cast()))
        } else {
            _mm256_loadu_si256(cb.cast())
        };
        _mm512_castsi512_ps(_mm512_slli_epi32::<16>(_mm512_cvtepu16_epi32(v)))
    }

    #[inline(always)]
    unsafe fn decode_cb<const B: u32>(t: Self::T, run: *const u8) -> (Self::F, Self::F) {
        let w = x86_windows::<B>(run);
        let look = |windows: __m512i, shifts: __m512i| _mm512_permutexvar_ps(_mm512_srlv_epi32(windows, shifts), t);
        if B == 2 {
            let s = _mm512_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30);
            (look(_mm512_set1_epi32(w[0]), s), look(_mm512_set1_epi32(w[1]), s))
        } else {
            let s = if B == 3 {
                _mm512_setr_epi32(0, 3, 6, 9, 12, 15, 18, 21, 0, 3, 6, 9, 12, 15, 18, 21)
            } else {
                _mm512_setr_epi32(0, 4, 8, 12, 16, 20, 24, 28, 0, 4, 8, 12, 16, 20, 24, 28)
            };
            let pair = |a: i32, b: i32| _mm512_mask_blend_epi32(0xFF00, _mm512_set1_epi32(a), _mm512_set1_epi32(b));
            (look(pair(w[0], w[1]), s), look(pair(w[2], w[3]), s))
        }
    }

    #[inline(always)]
    unsafe fn table_tern(cb: *const u8) -> Self::T {
        let v = (cb.cast::<u32>().read_unaligned() as u64) | ((cb.add(4).cast::<u16>().read_unaligned() as u64) << 32);
        _mm512_castsi512_ps(_mm512_slli_epi32::<16>(_mm512_cvtepu16_epi32(_mm256_set1_epi64x(v as i64))))
    }

    #[inline(always)]
    unsafe fn decode_tern(t: Self::T, bytes: *const u8) -> [(Self::F, Self::F); 5] {
        let (c171, three) = (_mm512_set1_epi32(171), _mm512_set1_epi32(3));
        // sixteen bytes' five digits, a digit a register of 16 lanes (x / 3 as (x * 171) >> 9); a macro and no closure, so that every intrinsic is inlined in the caller's instruction set
        macro_rules! digits {
            ($p:expr) => {{
                let w = _mm512_cvtepu8_epi32(_mm_loadu_si128($p.cast()));
                let q0 = _mm512_srli_epi32::<9>(_mm512_mullo_epi32(w, c171));
                let q1 = _mm512_srli_epi32::<9>(_mm512_mullo_epi32(q0, c171));
                let q2 = _mm512_srli_epi32::<9>(_mm512_mullo_epi32(q1, c171));
                let q3 = _mm512_srli_epi32::<9>(_mm512_mullo_epi32(q2, c171));
                [
                    _mm512_sub_epi32(w, _mm512_mullo_epi32(q0, three)),
                    _mm512_sub_epi32(q0, _mm512_mullo_epi32(q1, three)),
                    _mm512_sub_epi32(q1, _mm512_mullo_epi32(q2, three)),
                    _mm512_sub_epi32(q2, _mm512_mullo_epi32(q3, three)),
                    q3,
                ]
            }};
        }
        let (lo, hi) = (digits!(bytes), digits!(bytes.add(16)));
        let z = _mm512_setzero_ps();
        let mut out = [(z, z); 5];
        for d in 0..5 {
            out[d] = (_mm512_permutexvar_ps(lo[d], t), _mm512_permutexvar_ps(hi[d], t));
        }
        out
    }

    #[inline(always)]
    unsafe fn decode_tq(hi: u32, lo: u32) -> Self::F {
        let sh = _mm512_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30);
        let (h, l) = (_mm512_set1_epi32(hi as i32), _mm512_set1_epi32(lo as i32));
        let w = _mm512_or_si512(_mm512_sllv_epi32(h, sh), _mm512_srlv_epi32(l, _mm512_sub_epi32(_mm512_set1_epi32(32), sh)));
        let x = _mm512_add_epi32(_mm512_mullo_epi32(_mm512_srli_epi32::<18>(w), _mm512_set1_epi32(TQ_A as i32)), _mm512_set1_epi32(TQ_B as i32));
        // (the four bytes' sum: byte pairs, then the two halves)
        let m8 = _mm512_set1_epi32(0x00FF_00FF);
        let t = _mm512_add_epi32(_mm512_and_si512(x, m8), _mm512_and_si512(_mm512_srli_epi32::<8>(x), m8));
        let s = _mm512_add_epi32(_mm512_and_si512(t, _mm512_set1_epi32(0xFFFF)), _mm512_srli_epi32::<16>(t));
        _mm512_sub_ps(_mm512_cvtepi32_ps(s), _mm512_set1_ps(510.0))
    }

    #[inline(always)]
    unsafe fn decode_tqk<const K: usize, const R: u32>(x: *const u32) -> Self::F {
        let (hi, lo, sh) = tqk_windows512::<K, R>(x);
        let w = _mm512_or_si512(_mm512_sllv_epi32(hi, sh), _mm512_srlv_epi32(lo, _mm512_sub_epi32(_mm512_set1_epi32(32), sh)));
        let x = _mm512_add_epi32(_mm512_mullo_epi32(_mm512_srli_epi32::<18>(w), _mm512_set1_epi32(TQ_A as i32)), _mm512_set1_epi32(TQ_B as i32));
        let m8 = _mm512_set1_epi32(0x00FF_00FF);
        let t = _mm512_add_epi32(_mm512_and_si512(x, m8), _mm512_and_si512(_mm512_srli_epi32::<8>(x), m8));
        let s = _mm512_add_epi32(_mm512_and_si512(t, _mm512_set1_epi32(0xFFFF)), _mm512_srli_epi32::<16>(t));
        _mm512_sub_ps(_mm512_cvtepi32_ps(s), _mm512_set1_ps(510.0))
    }
}

#[cfg(test)]
mod tq_tests {
    use super::*;

    #[test]
    fn tq_kernels_agree() {
        for fmt in [Fmt::Tq, Fmt::Tq3, Fmt::Tq4] {
            agree(fmt);
        }
    }

    /// Every kernel of a format q of `fmt`'s bits against the weights as `tq_state_k` and `tq_value` define them, and against each other to the bit.
    fn agree(fmt: Fmt) {
        let (rows, cols) = (24, 512);
        let k = fmt.tq_bits().unwrap();
        let w = k * cols / 32;
        let mut x = 0x2545_F491_4F6C_DD1Du64;
        let mut next = move || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        let codes: Vec<u8> = (0..rows * w * 4).map(|_| next() as u8).collect();
        let book: Vec<u8> = (0..rows).flat_map(|r| (((0.5f32 + r as f32 / 16.0).to_bits() >> 16) as u16).to_le_bytes()).collect();
        let m = GlmMat { fmt, rows, cols, codes: &codes, book: &book };
        assert!(m.is_valid());
        let z: Vec<f32> = (0..cols).map(|_| (next() % 2001) as f32 / 1000.0 - 1.0).collect();
        let act: Vec<f32> = (0..rows).map(|_| (next() % 2001) as f32 / 1000.0 - 1.0).collect();
        let words: Vec<Vec<u32>> = (0..rows).map(|r| (0..w).map(|j| u32::from_le_bytes(codes[4 * (r * w + j)..][..4].try_into().unwrap())).collect()).collect();
        let weight = |r: usize, t: usize| tq_value(tq_state_k(&words[r], k, t)) as f64 * TQ_K as f64 * f32::from_bits((u16::from_le_bytes([book[2 * r], book[2 * r + 1]]) as u32) << 16) as f64;
        if fmt == Fmt::Tq {
            // (the two-bit lanes as `tq_lane` reads them)
            assert!((0..cols).all(|t| tq_lane(words[3][t / 16], words[3][(t / 16 + 1) % w], t % 16) == tq_value(tq_state_k(&words[3], 2, t))));
        }
        let want_dot: Vec<f64> = (0..rows).map(|r| (0..cols).map(|t| weight(r, t) * z[t] as f64).sum()).collect();
        let want_down: Vec<f64> = (0..cols).map(|t| (0..rows).map(|r| weight(r, t) * act[r] as f64).sum()).collect();
        let mut first: Option<(Kernel, Vec<f32>, Vec<f32>, Vec<f32>)> = None;
        for kernel in Kernel::ALL.into_iter().filter(|k| k.runs_here()) {
            let (mut dot, mut down, mut down128) = (vec![0f32; rows], vec![0f32; cols], vec![0f32; cols]);
            dot_glm(kernel, &m, &z, 0, rows, &mut dot);
            for o0 in (0..cols).step_by(256) {
                down_glm(kernel, &m, &act, o0, &mut down[o0..]);
            }
            for o0 in (0..cols).step_by(128) {
                down_glm_multi(kernel, &m, &[&act], o0, &mut [&mut down128[o0..o0 + 128]]);
            }
            // several tokens at once (the decode shared): each token's bits as alone
            let z2: Vec<f32> = z.iter().rev().copied().collect();
            let act2: Vec<f32> = act.iter().map(|a| -a * 0.5).collect();
            let (mut s1, mut s2) = (vec![0f32; rows], vec![0f32; rows]);
            dot_glm_multi(kernel, &m, &[&z, &z2, &z, &z2], 0, rows, &mut [&mut s1, &mut s2, &mut vec![0f32; rows], &mut vec![0f32; rows]]);
            let mut alone2 = vec![0f32; rows];
            dot_glm(kernel, &m, &z2, 0, rows, &mut alone2);
            let (mut o1, mut o2) = (vec![0f32; 128], vec![0f32; 128]);
            down_glm_multi(kernel, &m, &[&act, &act2], 128, &mut [&mut o1, &mut o2]);
            let mut alone_o2 = vec![0f32; 128];
            down_glm_multi(kernel, &m, &[&act2], 128, &mut [&mut alone_o2]);
            let same = |a: &[f32], b: &[f32]| a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits());
            assert!(same(&s1, &dot) && same(&s2, &alone2) && same(&o1, &down128[128..256]) && same(&o2, &alone_o2), "{}: tokens together differ from alone", kernel.name());
            for (g, want) in dot.iter().zip(&want_dot).chain(down.iter().zip(&want_down)).chain(down128.iter().zip(&want_down)) {
                assert!((*g as f64 - want).abs() <= 1e-4 * (1.0 + want.abs()), "{}: {g} vs {want}", kernel.name());
            }
            #[cfg(target_arch = "x86_64")]
            if kernel == Kernel::Avx512 && tq_vnni() {
                let (mut d2, mut w2) = (vec![0f32; rows], vec![0f32; cols]);
                use glm_avx512 as p;
                let (dot_p, down_p): (DotFn, DownFn) = match fmt {
                    Fmt::Tq => (p::dot_q, p::down_q),
                    Fmt::Tq3 => (p::dot_q3, p::down_q3),
                    _ => (p::dot_q4, p::down_q4),
                };
                // SAFETY: as in dot_glm and down_glm (the sizes checked there for the same arrays).
                unsafe {
                    dot_p(&m, z.as_ptr(), 0, rows, d2.as_mut_ptr());
                    for o0 in (0..cols).step_by(256) {
                        down_p(&m, act.as_ptr(), o0, w2[o0..].as_mut_ptr());
                    }
                }
                let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
                assert!(bits(&d2) == bits(&dot) && bits(&w2) == bits(&down), "the VNNI decode and the plain AVX-512 one differ");
            }
            match &first {
                None => first = Some((kernel, dot, down, down128)),
                Some((k0, d0, w0, x0)) => {
                    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
                    assert!(bits(d0) == bits(&dot) && bits(w0) == bits(&down) && bits(x0) == bits(&down128), "{} and {} differ ({})", k0.name(), kernel.name(), fmt.name());
                }
            }
        }
    }
}
