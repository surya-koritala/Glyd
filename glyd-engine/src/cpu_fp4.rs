//! Kimi-K3's experts on the host, read where the expert cache's host tier holds them (GLY-330): MXFP4 matrices in the GPU's exact FP4 block layouts
//! (glyd_gpu::pack's Fp4 and Fp4R: Int4's steps of 64 rows x 16 columns of codes, a row's scale for each run of 32 columns, as released or packed). A weight is
//! its E2M1 code times its run's power of two (exact in fp32); the sums are fp32, rounded where the GPU's are: the first product to bf16, the activation (situ)
//! to bf16, and the second product's fp32 sum times the pair's weight, as the GPU's weighted second product writes a pair's row (src/moe.rs `rows_sum`), not rounded
//! first. The activations' exp and tanh are the GPU kernels' own (`exp_one`, `tanh_one`: kernels/moe.cu's moe_exp and moe_tanh), so a pair is the GPU's to the bit
//! where the products' sums are.

use crate::cpu_experts::Kernel;
use crate::ffi::{err, Result};
use glyd_gpu::pack::{fp4_block_scale, INT4_STEP_BYTES};

/// An exact FP4 matrix [rows, cols] where the host tier holds it: its steps (`data`) and scales (`blocks`); `packed` the Fp4R layout's scales.
#[derive(Clone, Copy)]
pub struct Fp4View<'a> {
    pub data: &'a [u8],
    pub blocks: &'a [u8],
    pub rows: usize,
    pub cols: usize,
    pub packed: bool,
}

impl Fp4View<'_> {
    pub fn is_valid(&self) -> bool {
        let (rbs, ks) = (self.rows / 64, self.cols / 16);
        let scales = if self.packed { rbs * (64 + ks * 8) } else { rbs * ks / 2 * 64 };
        self.rows > 0
            && self.rows % 64 == 0
            && self.cols > 0
            && self.cols % 32 == 0
            && self.data.len() >= rbs * ks * INT4_STEP_BYTES
            && self.blocks.len() >= scales
    }

    /// Row r's scale for step k: 2^(e - 127) of its E8M0 byte.
    fn scale(&self, r: usize, k: usize) -> f32 {
        f32::from_bits((fp4_block_scale(self.blocks, self.cols, self.packed, r, k) as u32) << 23)
    }
}

const E2M1: [f32; 16] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0];

/// y[r - r0] = row r of W times x (fp32), for the rows r0..r1 (whole blocks of 64).
pub fn rows(kernel: Kernel, w: &Fp4View, x: &[f32], r0: usize, r1: usize, y: &mut [f32]) {
    assert!(r0 % 64 == 0 && r1 % 64 == 0 && r1 <= w.rows && x.len() >= w.cols && y.len() >= r1 - r0);
    #[cfg(target_arch = "x86_64")]
    if matches!(kernel, Kernel::Avx2 | Kernel::Avx512) {
        // SAFETY: the kernel runs here (the caller's `Kernel` is one `runs_here` allowed); the slices are checked above and by `is_valid`.
        unsafe {
            let xv = x_vectors(x, w.cols / 16);
            for rb in r0 / 64..r1 / 64 {
                block_avx2(w, &xv, rb, &mut y[rb * 64 - r0..][..64]);
            }
        }
        return;
    }
    #[cfg(target_arch = "aarch64")]
    if kernel == Kernel::Neon {
        // SAFETY: NEON is in every aarch64 target Rust builds for; the slices are checked above and by `is_valid`.
        unsafe {
            let xv = x_vectors_neon(x, w.cols / 16);
            for rb in r0 / 64..r1 / 64 {
                block_neon(w, &xv, rb, &mut y[rb * 64 - r0..][..64]);
            }
        }
        return;
    }
    let _ = kernel;
    for rb in r0 / 64..r1 / 64 {
        block_scalar(w, x, rb, &mut y[rb * 64 - r0..][..64]);
    }
}

/// Row block rb of W times x. A step's lane l (4 g + t) holds, in its four words q, the codes of rows 8 n + g and columns 8 jh + 2 t + h, where the code's
/// place in the word is 4 (pr % 4) + 16 h and pr = 4 q + (its place's quarter) = 2 n + jh (glyd_gpu::pack's lane_place and int4_steps).
fn block_scalar(w: &Fp4View, x: &[f32], rb: usize, out: &mut [f32]) {
    let ks = w.cols / 16;
    let mut acc = [0f32; 64];
    for k in 0..ks {
        let d = &w.data[(rb * ks + k) * INT4_STEP_BYTES..][..INT4_STEP_BYTES];
        let xs = &x[16 * k..16 * k + 16];
        let mut st = [0f32; 64];
        for l in 0..32 {
            let (g, t) = (l / 4, l % 4);
            for q in 0..4 {
                let at = 8 * (l + 32 * (q / 2)) + 4 * (q % 2);
                let word = u32::from_le_bytes([d[at], d[at + 1], d[at + 2], d[at + 3]]);
                for p4 in 0..4 {
                    let pr = 4 * q + p4;
                    let (n, jh) = (pr / 2, pr % 2);
                    for h in 0..2 {
                        let c = (word >> (4 * p4 + 16 * h)) & 15;
                        st[8 * n + g] += E2M1[c as usize] * xs[8 * jh + 2 * t + h];
                    }
                }
            }
        }
        for (r, a) in acc.iter_mut().enumerate() {
            *a += st[r] * w.scale(rb * 64 + r, k);
        }
    }
    out.copy_from_slice(&acc);
}

/// `block_scalar` eight lanes at a time: a step's words for lanes 8 lg .. 8 lg + 8 (two of them a lane, interleaved) loaded and split, each code's value
/// (a lookup of its size and its sign) times the columns' x for those lanes (`xv`: the step's four vectors of them, made once a call), summed a run of two
/// steps at a time and then times the run's scale for each lane's row; the lanes of a row (four: t) added at the block's end. A pass a (qq, lg), its four
/// accumulators held in registers over the run's steps: every accumulator's sums in the order a pass over the steps had made them (the same bits), where
/// 32 accumulators at once had spilled, each product a load and a store (about 4 cycles a weight on a Zen 3).
///
/// # Safety
/// AVX2 and FMA must be present; the slices are as `rows` checks, `xv` holds `cols / 16` steps' four vectors.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn block_avx2(w: &Fp4View, xv: &[[std::arch::x86_64::__m256; 4]], rb: usize, out: &mut [f32]) {
    use std::arch::x86_64::*;
    let ks = w.cols / 16;
    let mag = _mm256_setr_ps(0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0);
    let (seven, eight) = (_mm256_set1_epi32(7), _mm256_set1_epi32(8));
    let split = _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7);
    // (total[n * 4 + lg]: rows 8 n + 2 lg and 8 n + 2 lg + 1, each over its four lanes t)
    let mut total = [_mm256_setzero_ps(); 32];
    for run in 0..ks / 2 {
        for qq in 0..2 {
            for lg in 0..4 {
                // (acc[j]: n = 4 qq + j)
                let mut acc = [_mm256_setzero_ps(); 4];
                for k in 2 * run..2 * run + 2 {
                    let d = w.data.as_ptr().add((rb * ks + k) * INT4_STEP_BYTES) as *const __m256i;
                    let x4 = &xv[k];
                    // (u32 64 qq + 16 lg on: lanes 8 lg .. 8 lg + 8, their words q = 2 qq and 2 qq + 1 interleaved)
                    let a = _mm256_permutevar8x32_epi32(_mm256_loadu_si256(d.add(8 * qq + 2 * lg)), split);
                    let b = _mm256_permutevar8x32_epi32(_mm256_loadu_si256(d.add(8 * qq + 2 * lg + 1)), split);
                    let words = [_mm256_permute2x128_si256(a, b, 0x20), _mm256_permute2x128_si256(a, b, 0x31)];
                    for (qm, &word) in words.iter().enumerate() {
                        for p4 in 0..4 {
                            let (j, jh) = ((4 * qm + p4) / 2, p4 % 2);
                            for h in 0..2 {
                                let c = _mm256_srlv_epi32(word, _mm256_set1_epi32((4 * p4 + 16 * h) as i32));
                                let size = _mm256_permutevar8x32_ps(mag, _mm256_and_si256(c, seven));
                                let sign = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_and_si256(c, eight), 28));
                                let v = _mm256_xor_ps(size, sign);
                                acc[j] = _mm256_fmadd_ps(v, x4[2 * jh + h], acc[j]);
                            }
                        }
                    }
                }
                // the run's scale of each lane's row (8 n + 2 lg + i / 4)
                for (j, a) in acc.iter().enumerate() {
                    let n = 4 * qq + j;
                    let (r0, r1) = (rb * 64 + 8 * n + 2 * lg, rb * 64 + 8 * n + 2 * lg + 1);
                    let (s0, s1) = (w.scale(r0, 2 * run), w.scale(r1, 2 * run));
                    let sc = _mm256_setr_ps(s0, s0, s0, s0, s1, s1, s1, s1);
                    let i = n * 4 + lg;
                    total[i] = _mm256_fmadd_ps(*a, sc, total[i]);
                }
            }
        }
    }
    for n in 0..8 {
        for lg in 0..4 {
            let mut v = [0f32; 8];
            _mm256_storeu_ps(v.as_mut_ptr(), total[n * 4 + lg]);
            out[8 * n + 2 * lg] = (v[0] + v[1]) + (v[2] + v[3]);
            out[8 * n + 2 * lg + 1] = (v[4] + v[5]) + (v[6] + v[7]);
        }
    }
}

/// The columns' x of every step for `block_avx2`, for each lane of eight (t = i % 4) and each (jh, h) at 2 jh + h.
///
/// # Safety
/// AVX must be present; x holds 16 a step.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn x_vectors(x: &[f32], ks: usize) -> Vec<[std::arch::x86_64::__m256; 4]> {
    use std::arch::x86_64::*;
    (0..ks)
        .map(|k| {
            let xs = &x[16 * k..16 * k + 16];
            let xp = |jh: usize, h: usize| {
                _mm256_setr_ps(
                    xs[8 * jh + h],
                    xs[8 * jh + 2 + h],
                    xs[8 * jh + 4 + h],
                    xs[8 * jh + 6 + h],
                    xs[8 * jh + h],
                    xs[8 * jh + 2 + h],
                    xs[8 * jh + 4 + h],
                    xs[8 * jh + 6 + h],
                )
            };
            [xp(0, 0), xp(0, 1), xp(1, 0), xp(1, 1)]
        })
        .collect()
}

/// The top two bytes of each E2M1 code's fp32 value (the other two are 0), for NEON's table lookups.
#[cfg(target_arch = "aarch64")]
const E2M1_TOP: [[u8; 16]; 2] = {
    let mut t = [[0u8; 16]; 2];
    let mut c = 0;
    while c < 16 {
        let bits = E2M1_BITS[c];
        t[0][c] = (bits >> 24) as u8;
        t[1][c] = (bits >> 16) as u8;
        c += 1;
    }
    t
};

/// The E2M1 values' fp32 bits (E2M1's, as a constant expression).
#[cfg(target_arch = "aarch64")]
const E2M1_BITS: [u32; 16] = [
    0x0000_0000,
    0x3F00_0000,
    0x3F80_0000,
    0x3FC0_0000,
    0x4000_0000,
    0x4040_0000,
    0x4080_0000,
    0x40C0_0000,
    0x8000_0000,
    0xBF00_0000,
    0xBF80_0000,
    0xBFC0_0000,
    0xC000_0000,
    0xC040_0000,
    0xC080_0000,
    0xC0C0_0000,
];

/// `block_scalar` four lanes at a time on NEON: lanes 4 g .. 4 g + 3 (t = 0 .. 3, rows 8 n + g) of a step, their two words of a half (q = 2 qh and 2 qh + 1) split apart by one
/// de-interleaving load, each code's value by two table lookups (its float's top bytes) times the columns' x for those lanes (`xv`), summed over a run of two steps in four
/// accumulators (n = 4 qh .. 4 qh + 3) held in registers, then times the run's scale of each row; a row's four lanes added at the block's end. Arm's server and desktop cores
/// (Neoverse V2 in Grace, N2, the GB10's) run SVE2 at 128 bits, NEON's width, so NEON is the kernel there too.
///
/// # Safety
/// The slices are as `rows` checks, `xv` holds `cols / 16` steps' four vectors.
#[cfg(target_arch = "aarch64")]
unsafe fn block_neon(w: &Fp4View, xv: &[[std::arch::aarch64::float32x4_t; 4]], rb: usize, out: &mut [f32]) {
    use std::arch::aarch64::*;
    let ks = w.cols / 16;
    let (t3, t2) = (vld1q_u8(E2M1_TOP[0].as_ptr()), vld1q_u8(E2M1_TOP[1].as_ptr()));
    let fifteen = vdupq_n_u32(15);
    // (a code in the low byte of each 32-bit lane: the value's top bytes looked up there, the lane's other bytes look up code 0's, which are 0)
    let value = |c: uint32x4_t| -> float32x4_t {
        let i = vreinterpretq_u8_u32(c);
        let hi = vshlq_n_u32::<24>(vreinterpretq_u32_u8(vqtbl1q_u8(t3, i)));
        let lo = vshlq_n_u32::<16>(vreinterpretq_u32_u8(vqtbl1q_u8(t2, i)));
        vreinterpretq_f32_u32(vorrq_u32(hi, lo))
    };
    let mut total = [vdupq_n_f32(0.0); 64];
    for run in 0..ks / 2 {
        for qh in 0..2 {
            for g in 0..8 {
                let mut acc = [vdupq_n_f32(0.0); 4];
                for k in 2 * run..2 * run + 2 {
                    let d = w.data.as_ptr().add((rb * ks + k) * INT4_STEP_BYTES + 8 * (4 * g + 32 * qh)) as *const u32;
                    let words = vld2q_u32(d);
                    let x4 = &xv[k];
                    for (qm, word) in [words.0, words.1].into_iter().enumerate() {
                        for p4 in 0..4 {
                            let (j, jh) = ((4 * qm + p4) / 2, p4 % 2);
                            for h in 0..2 {
                                let c = vandq_u32(vshlq_u32(word, vdupq_n_s32(-((4 * p4 + 16 * h) as i32))), fifteen);
                                acc[j] = vfmaq_f32(acc[j], value(c), x4[2 * jh + h]);
                            }
                        }
                    }
                }
                for (j, a) in acc.iter().enumerate() {
                    let r = 8 * (4 * qh + j) + g;
                    total[r] = vfmaq_n_f32(total[r], *a, w.scale(rb * 64 + r, 2 * run));
                }
            }
        }
    }
    for (o, t) in out.iter_mut().zip(&total) {
        let v = [vgetq_lane_f32::<0>(*t), vgetq_lane_f32::<1>(*t), vgetq_lane_f32::<2>(*t), vgetq_lane_f32::<3>(*t)];
        *o = (v[0] + v[1]) + (v[2] + v[3]);
    }
}

/// The columns' x of every step for `block_neon`: for each (jh, h) at 2 jh + h, lane t holds column 8 jh + 2 t + h.
///
/// # Safety
/// x holds 16 a step.
#[cfg(target_arch = "aarch64")]
unsafe fn x_vectors_neon(x: &[f32], ks: usize) -> Vec<[std::arch::aarch64::float32x4_t; 4]> {
    use std::arch::aarch64::*;
    (0..ks)
        .map(|k| {
            let xs = &x[16 * k..16 * k + 16];
            let xp = |jh: usize, h: usize| {
                let v = [xs[8 * jh + h], xs[8 * jh + 2 + h], xs[8 * jh + 4 + h], xs[8 * jh + 6 + h]];
                vld1q_f32(v.as_ptr())
            };
            [xp(0, 0), xp(0, 1), xp(1, 0), xp(1, 1)]
        })
        .collect()
}

/// An FP4 expert where the host tier holds it (Kimi-K3's, DeepSeek-V4's): its first matrix [2 inter, latent] (gate then up) and its second [latent, inter].
#[derive(Clone, Copy)]
pub struct Fp4Expert<'a> {
    pub up: Fp4View<'a>,
    pub down: Fp4View<'a>,
}

impl Fp4Expert<'_> {
    /// The expert's bytes in the host tier (both matrices' steps and scales): what a call reads of it.
    pub fn bytes(&self) -> usize {
        self.up.data.len() + self.up.blocks.len() + self.down.data.len() + self.down.blocks.len()
    }
}

/// A pair of the router's choices the host computes: its expert, its token's row of z, its weight.
pub struct Fp4Pair<'a> {
    pub expert: Fp4Expert<'a>,
    pub token: usize,
    pub weight: f32,
}

/// The activation between a pair's two products and where its weight goes, as the GPU's kernel of the layer computes them.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Act {
    /// Kimi-K3's situ (moe.cu's MOE_SITU): the first product rounded to bf16, situ(gate, up) rounded to bf16, the second product's fp32 sum times the weight, as the
    /// GPU's weighted second product writes a pair's row (unrounded: the layer's sum rounds a token's pairs once).
    Situ { beta: f32, linear: f32 },
    /// DeepSeek-V4's release (kernels/dsv4.cu's v4_moe_act): bf16(weight silu(min(gate, lim)) clamp(up, -lim, lim)) of the first product as bf16, each block of 128 rounded to E4M3 at its power-of-two scale where
    /// `fp8`; the second product (fp32) is the pair's output, its weight already in it.
    Release { lim: f32, fp8: bool },
}

pub(crate) fn bf16(v: f32) -> f32 {
    f32::from_bits((((v.to_bits() as u64 + 0x7FFF + ((v.to_bits() >> 16) & 1) as u64) as u32) >> 16) << 16)
}

/// Cephes' expf and tanhf constants as the f32 they round to, by their bits (kernels/moe.cu's moe_exp and moe_tanh write the same bits): the exponent's bounds (where e^x
/// leaves the normal numbers), log2 e, ln 2 in two parts (high exact; ln 2 = high - low), the exp polynomial's and the tanh polynomial's coefficients from the highest power down,
/// and where tanh is 1.
const EXP_MAX: f32 = f32::from_bits(0x42b1_7218);
const EXP_MIN: f32 = f32::from_bits(0xc2ae_ac50);
const LOG2E: f32 = f32::from_bits(0x3fb8_aa3b);
const LN2_HI: f32 = f32::from_bits(0x3f31_8000);
const LN2_LO: f32 = f32::from_bits(0x395e_8083);
const EXP_P: [u32; 6] = [0x3950_6967, 0x3ab7_43ce, 0x3c08_8908, 0x3d2a_a9c1, 0x3e2a_aaaa, 0x3f00_0000];
const TANH_MAX: f32 = f32::from_bits(0x4231_7218);
const TANH_P: [u32; 5] = [0xbbba_f0ea, 0x3ca9_134e, 0xbd5c_1e2d, 0x3e08_8393, 0xbeaa_aa99];

/// e^x as kernels/moe.cu's moe_exp computes it, to the bit: Cephes' expf (n = x log2 e rounded to an integer, x - n ln 2 in two parts by fused multiply-adds, a degree-6
/// polynomial by Horner's rule in fused multiply-adds, times 2^n in two exact halves), every step an IEEE operation rounded to nearest. Its relative error is about 1e-7, as the
/// CUDA library's expf (2 last places at most), which the release's activations are computed with and which the host cannot reproduce. Past `EXP_MAX`: infinite; below
/// `EXP_MIN` (under the least normal number): 0.
pub(crate) fn exp_one(x: f32) -> f32 {
    if x > EXP_MAX {
        return f32::INFINITY;
    }
    if x < EXP_MIN {
        return 0.0;
    }
    let n = (x * LOG2E).round_ties_even();
    let r = n.mul_add(LN2_LO, n.mul_add(-LN2_HI, x));
    let p = EXP_P[1..].iter().fold(f32::from_bits(EXP_P[0]), |p, &c| p.mul_add(r, f32::from_bits(c)));
    let y = p.mul_add(r * r, r) + 1.0;
    let n = n as i32;
    y * f32::from_bits(((n / 2 + 127) as u32) << 23) * f32::from_bits(((n - n / 2 + 127) as u32) << 23)
}

/// tanh(x) as kernels/moe.cu's moe_tanh computes it, to the bit: Cephes' tanhf (an odd polynomial below 0.625 in size, 1 - 2 / (e^2|x| + 1) by `exp_one` from there, 1 past
/// `TANH_MAX`), every step an IEEE operation rounded to nearest; its relative error is about 1e-7, as the CUDA library's tanhf.
pub(crate) fn tanh_one(x: f32) -> f32 {
    let z = x.abs();
    if z > TANH_MAX {
        return 1f32.copysign(x);
    }
    if z >= 0.625 {
        return (1.0 - 2.0 / (exp_one(z + z) + 1.0)).copysign(x);
    }
    let s = x * x;
    let p = TANH_P[1..].iter().fold(f32::from_bits(TANH_P[0]), |p, &c| p.mul_add(s, f32::from_bits(c)));
    (p * s).mul_add(x, x)
}

/// situ(a, b) = beta tanh(a / beta) sigmoid(a) x linear tanh(b / linear) in fp32, unrounded, as kernels/moe.cu's moe_situ computes it to the bit: its operations in that
/// order (the sigmoid a division), its tanh and exp the GPU's own (`tanh_one`, `exp_one`).
pub(crate) fn situ_f32(a: f32, b: f32, beta: f32, linear: f32) -> f32 {
    beta * tanh_one(a / beta) / (1.0 + exp_one(-a)) * linear * tanh_one(b / linear)
}

/// `situ_f32` rounded to bf16 (moe.cu's MOE_SITU).
fn situ(a: f32, b: f32, (beta, linear): (f32, f32)) -> f32 {
    bf16(situ_f32(a, b, beta, linear))
}

/// The release's weighted activation of a pair's gate and up (bf16 values; kernels/dsv4.cu's v4_moe_act, to the bit): bf16(w silu(min(g, lim)) clamp(u, -lim, lim)), its exp the
/// GPU's own (`exp_one`).
fn release(g: f32, u: f32, w: f32, lim: f32) -> f32 {
    let (g, u) = (g.min(lim), u.clamp(-lim, lim));
    bf16(w * (g / (1.0 + exp_one(-g)) * u))
}

/// kernel.py's power-of-two scale: 2^ceil(log2(amax / max)), from the f32 bits.
pub(crate) fn pow2_scale(amax: f32, inv_max: f32) -> f32 {
    let b = (amax * inv_max).to_bits();
    let e = ((b >> 23) & 0xFF) as i32 - 127 + ((b & 0x7F_FFFF) != 0) as i32;
    f32::from_bits(((e + 127) as u32) << 23)
}

/// v (at most 448 in size) rounded to E4M3, to nearest and ties to even.
pub fn e4m3_round(v: f32) -> f32 {
    let a = v.abs();
    let e = (((a.to_bits() >> 23) & 0xFF) as i32 - 127).max(-6);
    let q = f32::from_bits(((e - 3 + 127) as u32) << 23);
    ((a / q).round_ties_even() * q).copysign(v)
}

/// kernel.py's act_quant in place (FP8, blocks of `block`, power-of-two scales): each value rounded to E4M3 at its block's scale, back as bf16.
pub fn fp8_sim(x: &mut [f32], block: usize) {
    for b in x.chunks_mut(block) {
        let amax = b.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-4);
        let s = pow2_scale(amax, 1.0 / 448.0);
        for v in b {
            *v = bf16(e4m3_round((*v / s).clamp(-448.0, 448.0)) * s);
        }
    }
}

/// The pairs' numbers by expert (an expert is the address of its first matrix's bytes), each expert's in the pairs' order, the experts in the order they first appear: the host reads each expert's bytes once a
/// call for all of its pairs.
pub fn by_expert(pairs: &[Fp4Pair]) -> Vec<Vec<usize>> {
    let mut at: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
    let mut groups: Vec<Vec<usize>> = Vec::new();
    for (p, pair) in pairs.iter().enumerate() {
        let g = *at.entry(pair.expert.up.data.as_ptr() as usize).or_insert_with(|| {
            groups.push(Vec::new());
            groups.len() - 1
        });
        groups[g].push(p);
    }
    groups
}

/// A raw pointer the crew's items write disjoint parts of.
#[derive(Clone, Copy)]
struct Out(*mut f32);
unsafe impl Send for Out {}
unsafe impl Sync for Out {}

impl Out {
    /// `len` values from `at` on (a method: a closure that calls it holds the whole `Out`, which is Sync, and not its pointer alone).
    ///
    /// # Safety
    /// The caller writes a part of the buffer no other item writes, while it is live.
    unsafe fn part(self, at: usize, len: usize) -> &'static mut [f32] {
        std::slice::from_raw_parts_mut(self.0.add(at), len)
    }
}

/// out[p latent ..] = pair p's output with its weight (`act`), fp32 as the GPU's weighted second product keeps a pair's row: the second matrix times the activation of [gate | up], the first matrix times its token's row of z (bf16),
/// on `threads` threads (the crew of src/cpu_experts.rs; `pin`: its threads each kept to a core). An item of work is a block of 64 rows of an expert's matrix for every pair of
/// that expert (`by_expert`), so a call reads each expert's bytes from memory once whatever its pairs (the block stays in the core's cache between them); each pair's sums are
/// the ones it has alone, to the bit.
pub fn experts(kernel: Kernel, z: &[u16], pairs: &[Fp4Pair], out: &mut [f32], act: Act, threads: usize, pin: bool) -> Result<()> {
    #[cfg(feature = "dev")]
    if let Some(gpu) = *SM86.lock().map_err(|_| crate::ffi::Error("the replay's lock".into()))? {
        return experts_sm86(z, pairs, out, act, gpu);
    }
    let Some(first) = pairs.first() else { return Ok(()) };
    let (two, latent) = (first.expert.up.rows, first.expert.up.cols);
    let inter = two / 2;
    for (p, pair) in pairs.iter().enumerate() {
        let e = pair.expert;
        if !e.up.is_valid()
            || !e.down.is_valid()
            || (e.up.rows, e.up.cols, e.down.rows, e.down.cols) != (two, latent, latent, inter)
            || inter % 64 != 0
            || (pair.token + 1) * latent > z.len()
        {
            return err(format!(
                "pair {p}: an expert of {} x {} and {} x {} for token {}, where {two} x {latent}, {latent} x {inter} and rows of z are wanted",
                e.up.rows, e.up.cols, e.down.rows, e.down.cols, pair.token
            ));
        }
    }
    if out.len() != pairs.len() * latent {
        return err(format!("{} values for the host's sums, where {} pairs of {latent} are wanted", out.len(), pairs.len()));
    }
    if matches!(act, Act::Release { fp8: true, .. }) && inter % 128 != 0 {
        return err(format!("experts of {inter} neurons, where the activation's blocks of 128 are rounded to E4M3"));
    }
    let zf: Vec<Vec<f32>> =
        pairs.iter().map(|p| z[p.token * latent..(p.token + 1) * latent].iter().map(|&h| f32::from_bits((h as u32) << 16)).collect()).collect();
    let groups = by_expert(pairs);
    // the first products: an expert's rows in blocks of 64, each block for every pair of the expert
    let mut gu = vec![0f32; pairs.len() * two];
    let (gp, ub) = (Out(gu.as_mut_ptr()), two / 64);
    crate::cpu_experts::run(threads, pin, groups.len() * ub, &|item| {
        let (g, b) = (item / ub, item % ub);
        for &p in &groups[g] {
            // SAFETY: item (g, b) writes rows 64 b .. 64 b + 64 of the parts of gu of its expert's pairs alone, which are live for the call.
            let y = unsafe { gp.part(p * two + 64 * b, 64) };
            rows(kernel, &pairs[p].expert.up, &zf[p], 64 * b, 64 * b + 64, y);
        }
    });
    let mut a = vec![0f32; pairs.len() * inter];
    for (p, h) in a.chunks_exact_mut(inter).enumerate() {
        let (g, u) = (&gu[p * two..p * two + inter], &gu[p * two + inter..(p + 1) * two]);
        match act {
            Act::Situ { beta, linear } => h.iter_mut().enumerate().for_each(|(i, v)| *v = situ(bf16(g[i]), bf16(u[i]), (beta, linear))),
            Act::Release { lim, fp8 } => {
                let w = pairs[p].weight;
                for (i, v) in h.iter_mut().enumerate() {
                    *v = release(bf16(g[i]), bf16(u[i]), w, lim);
                }
                if fp8 {
                    fp8_sim(h, 128);
                }
            }
        }
    }
    // the second products, an expert's blocks for every pair of it again
    let (op, db) = (Out(out.as_mut_ptr()), latent / 64);
    crate::cpu_experts::run(threads, pin, groups.len() * db, &|item| {
        let (g, b) = (item / db, item % db);
        for &p in &groups[g] {
            // SAFETY: item (g, b) writes rows 64 b .. 64 b + 64 of the parts of out of its expert's pairs alone, which are live for the call.
            let y = unsafe { op.part(p * latent + 64 * b, 64) };
            rows(kernel, &pairs[p].expert.down, &a[p * inter..(p + 1) * inter], 64 * b, 64 * b + 64, y);
            if let Act::Situ { .. } = act {
                let w = pairs[p].weight;
                y.iter_mut().for_each(|v| *v *= w);
            }
        }
    });
    Ok(())
}

/// (development) Where `Some`, `experts` computes each pair in the order of the GPU's step products on an A10's tensor cores (`experts_sm86`), the layer's
/// experts and the GPU's slots given (glyd-gpu's moe_splits: its multiprocessors, twice on an HBM part): a check's (`moe::PAIRS`), never the product path's.
#[cfg(feature = "dev")]
pub(crate) static SM86: std::sync::Mutex<Option<Sm86>> = std::sync::Mutex::new(None);

/// (development) What `experts_sm86` needs of the GPU beside it: the layer's experts and the library's slots (its K split: `splits`).
#[cfg(feature = "dev")]
#[derive(Clone, Copy, Debug)]
pub(crate) struct Sm86 {
    pub experts: usize,
    pub slots: usize,
}

#[cfg(feature = "dev")]
impl Sm86 {
    /// The blocks glyd-gpu's step products split a product's K over (moe_splits: by the layer's shape, `rows` an expert, and the slots alone).
    fn splits(&self, rows: usize, k: usize) -> usize {
        let units = (rows / 64 * self.experts).max(1);
        (self.slots / units).min(k / 256).max(1)
    }
}

/// (development) The sum of a block of eight products with the accumulator `c` as mma.sync.m16n8k16 (bf16 inputs, fp32 accumulator) makes it on an A10's tensor
/// cores (sm86; a probe of 2,560,000 random outputs, 2026-11-26, every one to the bit): each term aligned to the largest of their exponents (a product's the sum of
/// its two factors' exponents, the accumulator's its own), 24 bits kept below that exponent and those under them dropped toward zero, the terms added exactly, and
/// the sum rounded to fp32 toward zero. `p`: (the product, exact in fp32; its factors' exponents' sum), a zero one left out.
#[cfg(feature = "dev")]
fn mma8(c: f32, p: &[(f32, i32)]) -> f32 {
    let ex = |v: f32| ((v.to_bits() >> 23) & 0xFF).max(1) as i32 - 127;
    let terms = p.iter().filter(|t| t.0 != 0.0).copied().chain((c != 0.0).then(|| (c, ex(c))));
    let Some(top) = terms.clone().map(|t| t.1).max() else { return 0.0 };
    let sum: i64 = terms.map(|(v, _)| (v as f64 * 2f64.powi(24 - top)).trunc() as i64).sum();
    let (sign, mut m) = (if sum < 0 { -1.0 } else { 1.0 }, sum.unsigned_abs());
    let drop = (64 - m.leading_zeros() as i32 - 24).max(0);
    m = m >> drop << drop;
    (sign * m as f64 * 2f64.powi(top - 24)) as f32
}

/// (development) Rows of W x as glyd-gpu's step products make them (mma_moe_kernel, no activation, on an A10): K in `splits` blocks, each block's steps in eight runs
/// (its warps), each run's steps one after another through the tensor cores (`mma8` twice a step: columns 0-7, then 8-15), the runs added as warp_sums adds them
/// ((w0 + w4) + (w1 + w5)) + (w2 + w6)) + (w3 + w7), the blocks added in order from 0, and the bias's 0 added. `x`: the input row (bf16 values).
#[cfg(feature = "dev")]
fn rows_sm86(w: &Fp4View, x: &[f32], gpu: Sm86) -> Vec<f32> {
    let (ks, rbs) = (w.cols / 16, w.rows / 64);
    let ex = |v: f32| ((v.to_bits() >> 23) & 0xFF).max(1) as i32 - 127;
    let splits = gpu.splits(w.rows, w.cols);
    let mut y = vec![0f32; w.rows];
    for rb in 0..rbs {
        // (each step's weights [64 rows, 16 columns], as `block_scalar` reads them, times their run's scale: exact)
        let steps: Vec<[[f32; 16]; 64]> = (0..ks)
            .map(|k| {
                let d = &w.data[(rb * ks + k) * INT4_STEP_BYTES..][..INT4_STEP_BYTES];
                let mut t = [[0f32; 16]; 64];
                for l in 0..32 {
                    let (g, tt) = (l / 4, l % 4);
                    for q in 0..4 {
                        let at = 8 * (l + 32 * (q / 2)) + 4 * (q % 2);
                        let word = u32::from_le_bytes([d[at], d[at + 1], d[at + 2], d[at + 3]]);
                        for p4 in 0..4 {
                            let (n, jh) = ((4 * q + p4) / 2, (4 * q + p4) % 2);
                            for h in 0..2 {
                                t[8 * n + g][8 * jh + 2 * tt + h] = E2M1[((word >> (4 * p4 + 16 * h)) & 15) as usize];
                            }
                        }
                    }
                }
                for (r, row) in t.iter_mut().enumerate() {
                    let sc = w.scale(rb * 64 + r, k);
                    row.iter_mut().for_each(|v| *v *= sc);
                }
                t
            })
            .collect();
        for r in 0..64 {
            let mut a = 0f32;
            for z in 0..splits {
                let (k0, k1) = (z * ks / splits, (z + 1) * ks / splits);
                let per = (k1 - k0).div_ceil(8);
                let mut wsum = [0f32; 8];
                for (c, ws) in wsum.iter_mut().enumerate() {
                    let s0 = (k0 + c * per).min(k1);
                    for s in s0..(s0 + per).min(k1) {
                        for half in 0..2 {
                            let p: Vec<(f32, i32)> = (8 * half..8 * half + 8)
                                .map(|j| {
                                    let (wv, xv) = (steps[s][r][j], x[16 * s + j]);
                                    (wv * xv, if wv == 0.0 || xv == 0.0 { 0 } else { ex(wv) + ex(xv) })
                                })
                                .collect();
                            *ws = mma8(*ws, &p);
                        }
                    }
                }
                let part = (((wsum[0] + wsum[4]) + (wsum[1] + wsum[5])) + (wsum[2] + wsum[6])) + (wsum[3] + wsum[7]);
                a = if splits == 1 { part } else { a + part };
            }
            y[rb * 64 + r] = a + 0.0;
        }
    }
    y
}

/// (development) `experts` with every product in the order of the GPU's step products on an A10 (`rows_sm86`), one thread: the first product's rows rounded to bf16,
/// the activation as `experts` makes it, the second product's rows, times the weight where `act` puts it there.
#[cfg(feature = "dev")]
fn experts_sm86(z: &[u16], pairs: &[Fp4Pair], out: &mut [f32], act: Act, gpu: Sm86) -> Result<()> {
    let Some(first) = pairs.first() else { return Ok(()) };
    let (two, latent) = (first.expert.up.rows, first.expert.up.cols);
    let inter = two / 2;
    if out.len() != pairs.len() * latent {
        return err(format!("{} values for the host's sums, where {} pairs of {latent} are wanted", out.len(), pairs.len()));
    }
    for (pair, o) in pairs.iter().zip(out.chunks_exact_mut(latent)) {
        let zf: Vec<f32> = z[pair.token * latent..(pair.token + 1) * latent].iter().map(|&h| f32::from_bits((h as u32) << 16)).collect();
        let gu = rows_sm86(&pair.expert.up, &zf, gpu);
        let mut h: Vec<f32> = match act {
            Act::Situ { beta, linear } => (0..inter).map(|i| situ(bf16(gu[i]), bf16(gu[inter + i]), (beta, linear))).collect(),
            Act::Release { lim, .. } => (0..inter).map(|i| release(bf16(gu[i]), bf16(gu[inter + i]), pair.weight, lim)).collect(),
        };
        if let Act::Release { fp8: true, .. } = act {
            fp8_sim(&mut h, 128);
        }
        let y = rows_sm86(&pair.expert.down, &h, gpu);
        match act {
            Act::Situ { .. } => o.iter_mut().zip(&y).for_each(|(v, &d)| *v = d * pair.weight),
            Act::Release { .. } => o.copy_from_slice(&y),
        }
    }
    Ok(())
}

/// `actcheck` (development): the activations the expert cache's host computes beside the GPU, each as the GPU's kernel computes it and as the host does (`experts`), on the same bf16
/// inputs: Kimi-K3's situ (kernels/moe.cu's moe_act) and the release's weighted SiLU of DeepSeek-V4 (kernels/dsv4.cu's v4_moe_act, a weight of one, no E4M3 step). The inputs: every
/// gate bf16 holds from -64 to 64 against ups of each size, and every such up against gates of each size; the outputs (bf16) that differ, the most last places apart, and the first.
#[cfg(feature = "dev")]
pub fn act_check(_a: &crate::args::Args) -> Result<()> {
    use crate::dev::Buf;
    let dev = crate::dev::Dev::new(0)?;
    let k = crate::moe::MoeKern::load(&dev)?;
    let release_k: crate::kt::v4_moe_act = dev.typed()?;
    let s = dev.stream;
    let f = |b: u16| f32::from_bits((b as u32) << 16);
    // (every bf16 of size at most 64, both signs; a spread of the other input)
    let all: Vec<u16> = (0..=u16::MAX).filter(|&b| f(b).is_finite() && f(b).abs() <= 64.0).collect();
    let spread: Vec<u16> =
        [-40.0f32, -17.0, -3.0, -0.75, -0.0625, 0.0078125, 0.5, 1.0, 2.5, 6.0, 23.0, 50.0].iter().map(|&v| (bf16(v).to_bits() >> 16) as u16).collect();
    let mut pairs: Vec<(u16, u16)> = all.iter().flat_map(|&g| spread.iter().map(move |&u| (g, u))).collect();
    pairs.extend(all.iter().flat_map(|&u| spread.iter().map(move |&g| (g, u))));
    const I: usize = 1024;
    let rows = pairs.len().div_ceil(I);
    pairs.resize(rows * I, (0, 0));
    let mut gu = vec![0u16; rows * 2 * I];
    for (n, &(g, u)) in pairs.iter().enumerate() {
        (gu[n / I * 2 * I + n % I], gu[n / I * 2 * I + I + n % I]) = (g, u);
    }
    let (din, dout) = (Buf::new(gu.len() * 2)?, Buf::new(rows * I * 2)?);
    din.write(0, &gu)?;
    let ord = |b: u16| if b & 0x8000 != 0 { -((b & 0x7fff) as i32) } else { b as i32 };
    let mut out = vec![0u16; rows * I];
    for name in ["situ", "the release's SiLU"] {
        // SAFETY: the buffers hold `rows` rows of [2 I] and [I] bf16; the stream is the Dev's.
        unsafe {
            if name == "situ" {
                k.act(s, din.ptr, dout.ptr, rows, I, &crate::moe::Act::Situ { beta: 4.0, linear: 25.0 })?;
            } else {
                crate::launch!(release_k, [rows as u32, (I / 512) as u32, 1], [128, 1, 1], s; din.ptr, dout.ptr, 0u64, 0u64, 0i32, I as i32, f32::INFINITY, 0i32)?;
            }
        }
        dev.sync()?;
        dout.read(0, &mut out)?;
        let (mut differ, mut most, mut first) = (0usize, 0u32, None);
        for (n, &(g, u)) in pairs.iter().enumerate() {
            let host = if name == "situ" { situ(f(g), f(u), (4.0, 25.0)) } else { release(f(g), f(u), 1.0, f32::INFINITY) };
            let h = (host.to_bits() >> 16) as u16;
            let d = (ord(h) - ord(out[n])).unsigned_abs();
            if d > 0 {
                differ += 1;
                most = most.max(d);
                first.get_or_insert((f(g), f(u), f(out[n]), host));
            }
        }
        println!(
            "{name}: {} inputs, the GPU's kernel against the host's: {differ} outputs differ, the most {most} last places apart{}",
            pairs.len(),
            first.map_or(String::new(), |(g, u, a, b)| format!("; the first: gate {g:e}, up {u:e}: the GPU {a:e}, the host {b:e}"))
        );
    }
    Ok(())
}

/// `fp4bench [--pairs 1,3,16] [--threads 8,32,64,0] [--pin] [--reps 20]` (development): Kimi-K3-sized experts (6144 x 3584 and 3584 x 3072, the packed-scale
/// layout, random) through `experts` on this computer: each call's time for each number of pairs and of threads, the median of `reps` calls one after another
/// (0 threads: all of them).
#[cfg(feature = "dev")]
pub fn bench(a: &crate::args::Args) -> Result<()> {
    let list = |k: &str, d: &str| -> Result<Vec<usize>> {
        a.flag(k).unwrap_or(d).split(',').map(|x| x.trim().parse::<usize>().map_err(|_| crate::ffi::Error(format!("--{k} {x}: not a number")))).collect()
    };
    let (pairs_l, threads_l, reps, pin) = (list("pairs", "1,3,16")?, list("threads", "8,32,64,0")?, a.num("reps", 20usize)?, a.has("pin"));
    let (latent, inter) = (3584, 3072);
    let most = pairs_l.iter().copied().max().unwrap_or(1);
    let mut s = 0x2545_f491_4f6c_dd1du64;
    let mut rnd = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    let mut pack = |rows: usize, cols: usize| {
        let q: Vec<u8> = (0..rows * cols / 2).map(|_| rnd() as u8).collect();
        let e: Vec<u8> = (0..rows * cols / 32).map(|_| 120 + (rnd() % 4) as u8).collect();
        glyd_gpu::pack::pack_fp4r_checked(&q, &e, rows, cols).ok_or_else(|| crate::ffi::Error("a pack".into()))
    };
    let ups: Vec<_> = (0..most).map(|_| pack(2 * inter, latent)).collect::<Result<_>>()?;
    let downs: Vec<_> = (0..most).map(|_| pack(latent, inter)).collect::<Result<_>>()?;
    fn view(p: &glyd_gpu::pack::Tiered, r: usize, c: usize) -> Fp4View<'_> {
        Fp4View { data: &p.data, blocks: &p.blocks, rows: r, cols: c, packed: true }
    }
    let z: Vec<u16> = (0..latent).map(|i| (bf16(((i * 13 % 29) as f32 - 14.0) / 64.0).to_bits() >> 16) as u16).collect();
    let kernel = crate::cpu_experts::checked_kernel();
    let bytes = ups[0].data.len() + ups[0].blocks.len() + downs[0].data.len() + downs[0].blocks.len();
    println!("{:?} kernel, {} threads here, an expert {:.1} MB", kernel, std::thread::available_parallelism().map_or(1, |n| n.get()), bytes as f64 / 1e6);
    for &t in &threads_l {
        for &np in &pairs_l {
            let pairs: Vec<Fp4Pair> = (0..np)
                .map(|i| Fp4Pair { expert: Fp4Expert { up: view(&ups[i], 2 * inter, latent), down: view(&downs[i], latent, inter) }, token: 0, weight: 0.1 })
                .collect();
            let mut out = vec![0f32; np * latent];
            let mut us: Vec<f64> = (0..reps)
                .map(|_| {
                    let t0 = std::time::Instant::now();
                    experts(kernel, &z, &pairs, &mut out, Act::Situ { beta: 4.0, linear: 25.0 }, t, pin).map(|_| t0.elapsed().as_secs_f64() * 1e6)
                })
                .collect::<Result<_>>()?;
            us.sort_by(f64::total_cmp);
            let m = us[us.len() / 2];
            println!("threads {t:>3} pin {pin}: {np:>2} pairs {m:>8.0} us a call ({:.0} us a pair, {:.1} GB/s)", m / np as f64, (bytes * np) as f64 / m / 1e3);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn random_pack(rows: usize, cols: usize, packed: bool, seed: u64) -> glyd_gpu::pack::Tiered {
        let mut s = seed;
        let mut rnd = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        let q: Vec<u8> = (0..rows * cols / 2).map(|_| rnd() as u8).collect();
        // (scales within 4 of each row's smallest where packed, else anywhere from 100 to 127)
        let e: Vec<u8> = (0..rows * cols / 32).map(|_| if packed { 120 + (rnd() % 4) as u8 } else { 100 + (rnd() % 28) as u8 }).collect();
        if packed { glyd_gpu::pack::pack_fp4r_checked(&q, &e, rows, cols) } else { glyd_gpu::pack::pack_fp4_checked(&q, &e, rows, cols) }.expect("a pack")
    }

    /// Each kernel's rows of W x are the dense product of the pack as the GPU decodes it (`unpack_fp4`), within fp32 rounding, in both layouts.
    #[test]
    fn rows_are_the_decoded_matrix_times_x() {
        let (nr, nc) = (192, 128);
        for packed in [true, false] {
            let p = random_pack(nr, nc, packed, 0x1234_5678_9abc_def1 + packed as u64);
            let mut dense = vec![0u16; nr * nc];
            assert!(glyd_gpu::pack::unpack_fp4(&p, &mut dense));
            let x: Vec<f32> = (0..nc).map(|i| ((i * 37 % 101) as f32 - 50.0) / 25.0).collect();
            let want: Vec<f64> = (0..nr).map(|r| (0..nc).map(|c| f32::from_bits((dense[r * nc + c] as u32) << 16) as f64 * x[c] as f64).sum()).collect();
            let v = Fp4View { data: &p.data, blocks: &p.blocks, rows: nr, cols: nc, packed };
            assert!(v.is_valid());
            for k in Kernel::ALL.into_iter().filter(|k| k.runs_here()) {
                let mut y = vec![0f32; 128];
                rows(k, &v, &x, 64, 192, &mut y);
                for (i, &got) in y.iter().enumerate() {
                    let w = want[64 + i];
                    assert!((got as f64 - w).abs() <= 1e-5 * w.abs().max(1.0), "{k:?} packed {packed} row {}: {got} against {w}", 64 + i);
                }
            }
        }
    }

    /// `mma8`, twice a step as the GPU's step products call it, makes an A10's tensor cores' outputs to the bit: 24 outputs of mma.sync.m16n8k16 (bf16, fp32) read
    /// back from an A10 (handoff's probe, 2026-11-26), each one that a chain of fused multiply-adds in the columns' order does not make.
    #[cfg(feature = "dev")]
    #[test]
    fn the_a10s_tensor_core_sums_are_mma8s() {
        const A10: [([u16; 16], [u16; 16], u32, u32); 24] = [
            (
                [0xbfb5, 0x4060, 0x3f8f, 0x40ec, 0xbfe9, 0x3e0e, 0xc05c, 0x0000, 0x4093, 0xbf69, 0xc0b6, 0x3e69, 0x3ec6, 0xc0bb, 0x3e54, 0x0000],
                [0x3da1, 0xbc10, 0xbe80, 0xbe3a, 0xbeba, 0x3c0e, 0x3dd1, 0xbc47, 0xbc3b, 0x3d44, 0xbc70, 0x3de1, 0xbe2e, 0xbefc, 0xbc77, 0x8000],
                0x3c8e5516,
                0x3fb1a885,
            ),
            (
                [0xbfb5, 0x4060, 0x3f8f, 0x40ec, 0xbfe9, 0x3e0e, 0xc05c, 0x0000, 0x4093, 0xbf69, 0xc0b6, 0x3e69, 0x3ec6, 0xc0bb, 0x3e54, 0x0000],
                [0x3c18, 0xbca3, 0x8000, 0x3bea, 0xbe30, 0x3c3d, 0xbe43, 0x3e9b, 0x3eb5, 0x0000, 0xbcdd, 0x3e7f, 0xbd48, 0x3e16, 0xbc1d, 0x3e68],
                0x00000000,
                0x3ff2b501,
            ),
            (
                [0xbfb5, 0x4060, 0x3f8f, 0x40ec, 0xbfe9, 0x3e0e, 0xc05c, 0x0000, 0x4093, 0xbf69, 0xc0b6, 0x3e69, 0x3ec6, 0xc0bb, 0x3e54, 0x0000],
                [0xbd77, 0x3d76, 0xbbb0, 0xbc7d, 0x8000, 0xbec4, 0xbc77, 0x3ef6, 0x0000, 0x0000, 0xbe77, 0xbb8a, 0xbbef, 0x0000, 0x3ec0, 0x3e0e],
                0x00000000,
                0x3fcf5e9c,
            ),
            (
                [0xbfb5, 0x4060, 0x3f8f, 0x40ec, 0xbfe9, 0x3e0e, 0xc05c, 0x0000, 0x4093, 0xbf69, 0xc0b6, 0x3e69, 0x3ec6, 0xc0bb, 0x3e54, 0x0000],
                [0x3ebb, 0x3e96, 0xbe6e, 0x3e69, 0xbd67, 0xbe6b, 0x3e72, 0x3e36, 0xbd4c, 0xbcdb, 0x3b8d, 0xbece, 0x3da0, 0xbd1b, 0xbdc5, 0x3bc0],
                0xbdae74a4,
                0x3f817c01,
            ),
            (
                [0x8000, 0xbe57, 0x3f43, 0x404f, 0xbebd, 0xbe7a, 0x8000, 0xc087, 0xc00a, 0xc074, 0xbe4a, 0xbea9, 0x3f96, 0x3eed, 0x3f43, 0xbec6],
                [0x3da1, 0xbc10, 0xbe80, 0xbe3a, 0xbeba, 0x3c0e, 0x3dd1, 0xbc47, 0xbc3b, 0x3d44, 0xbc70, 0x3de1, 0xbe2e, 0xbefc, 0xbc77, 0x8000],
                0x39b25440,
                0xbf9c6f5a,
            ),
            (
                [0x8000, 0xbe57, 0x3f43, 0x404f, 0xbebd, 0xbe7a, 0x8000, 0xc087, 0xc00a, 0xc074, 0xbe4a, 0xbea9, 0x3f96, 0x3eed, 0x3f43, 0xbec6],
                [0x3ebb, 0x3e96, 0xbe6e, 0x3e69, 0xbd67, 0xbe6b, 0x3e72, 0x3e36, 0xbd4c, 0xbcdb, 0x3b8d, 0xbece, 0x3da0, 0xbd1b, 0xbdc5, 0x3bc0],
                0x3d03ed17,
                0x3e491d40,
            ),
            (
                [0x3e2e, 0x3e3b, 0x4081, 0x0000, 0xbfa6, 0xbffc, 0xbfcd, 0xbf93, 0xbe07, 0x400e, 0x3fee, 0xbfa9, 0x3e34, 0x40cd, 0x3ff8, 0x3f8f],
                [0x3e3a, 0x3c83, 0x3e3e, 0x3e3d, 0xbd03, 0x0000, 0xbcbf, 0x3e88, 0x3e03, 0xbe0e, 0xbc5d, 0xbc06, 0x3d20, 0x8000, 0xbe62, 0x3e5c],
                0x3ec3a51b,
                0x3ed6119e,
            ),
            (
                [0x40e9, 0xbe16, 0x8000, 0xbf12, 0xc09c, 0xbe3a, 0x3f76, 0x3ff4, 0x3e1c, 0x3f40, 0x8000, 0xc03d, 0x3f87, 0xc0cf, 0x0000, 0x3fdb],
                [0x3cf8, 0xbd57, 0xbe70, 0x3d1c, 0xbdbc, 0x3c81, 0xbef6, 0xbda0, 0x3c9e, 0xbdac, 0x3df6, 0x3deb, 0x3ce7, 0x3d56, 0xbcbe, 0x3c98],
                0x3c5ed128,
                0xbf1f1227,
            ),
            (
                [0x40e9, 0xbe16, 0x8000, 0xbf12, 0xc09c, 0xbe3a, 0x3f76, 0x3ff4, 0x3e1c, 0x3f40, 0x8000, 0xc03d, 0x3f87, 0xc0cf, 0x0000, 0x3fdb],
                [0x3e3a, 0x3c83, 0x3e3e, 0x3e3d, 0xbd03, 0x0000, 0xbcbf, 0x3e88, 0x3e03, 0xbe0e, 0xbc5d, 0xbc06, 0x3d20, 0x8000, 0xbe62, 0x3e5c],
                0xbd24d94c,
                0x400a6e8d,
            ),
            (
                [0x40e9, 0xbe16, 0x8000, 0xbf12, 0xc09c, 0xbe3a, 0x3f76, 0x3ff4, 0x3e1c, 0x3f40, 0x8000, 0xc03d, 0x3f87, 0xc0cf, 0x0000, 0x3fdb],
                [0x3ebb, 0x3e96, 0xbe6e, 0x3e69, 0xbd67, 0xbe6b, 0x3e72, 0x3e36, 0xbd4c, 0xbcdb, 0x3b8d, 0xbece, 0x3da0, 0xbd1b, 0xbdc5, 0x3bc0],
                0xbd6bb722,
                0x4099e695,
            ),
            (
                [0x40e9, 0xbe16, 0x8000, 0xbf12, 0xc09c, 0xbe3a, 0x3f76, 0x3ff4, 0x3e1c, 0x3f40, 0x8000, 0xc03d, 0x3f87, 0xc0cf, 0x0000, 0x3fdb],
                [0x3c4d, 0xbb83, 0x8000, 0x3edd, 0x3e9f, 0x3dc6, 0x8000, 0x3c12, 0x3d5a, 0x3eae, 0x3ce1, 0xbb81, 0xbdb9, 0xbdb7, 0x3e04, 0xbe59],
                0x00000000,
                0xbfa3184b,
            ),
            (
                [0xbf3d, 0xc064, 0x3fee, 0x40f6, 0x3e59, 0x3faa, 0xbfc4, 0xbe92, 0xbe54, 0xc03e, 0xbe8a, 0xc00f, 0x3e67, 0xbe07, 0xbfa5, 0xbee8],
                [0x3cf8, 0xbd57, 0xbe70, 0x3d1c, 0xbdbc, 0x3c81, 0xbef6, 0xbda0, 0x3c9e, 0xbdac, 0x3df6, 0x3deb, 0x3ce7, 0x3d56, 0xbcbe, 0x3c98],
                0x3c2a2366,
                0x3f44de77,
            ),
            (
                [0xbf3d, 0xc064, 0x3fee, 0x40f6, 0x3e59, 0x3faa, 0xbfc4, 0xbe92, 0xbe54, 0xc03e, 0xbe8a, 0xc00f, 0x3e67, 0xbe07, 0xbfa5, 0xbee8],
                [0x3ebb, 0x3e96, 0xbe6e, 0x3e69, 0xbd67, 0xbe6b, 0x3e72, 0x3e36, 0xbd4c, 0xbcdb, 0x3b8d, 0xbece, 0x3da0, 0xbd1b, 0xbdc5, 0x3bc0],
                0xbd7226b3,
                0x3eb18716,
            ),
            (
                [0x0000, 0xc035, 0x40b2, 0xbe2d, 0xc079, 0xbeba, 0xbed2, 0x3f17, 0xbeff, 0x3e10, 0xbea0, 0xc0d6, 0x0000, 0xbf20, 0x408a, 0x3e8b],
                [0x3da1, 0xbc10, 0xbe80, 0xbe3a, 0xbeba, 0x3c0e, 0x3dd1, 0xbc47, 0xbc3b, 0x3d44, 0xbc70, 0x3de1, 0xbe2e, 0xbefc, 0xbc77, 0x8000],
                0xbc1587d8,
                0xbeea8c62,
            ),
            (
                [0x0000, 0xc035, 0x40b2, 0xbe2d, 0xc079, 0xbeba, 0xbed2, 0x3f17, 0xbeff, 0x3e10, 0xbea0, 0xc0d6, 0x0000, 0xbf20, 0x408a, 0x3e8b],
                [0x3e3a, 0x3c83, 0x3e3e, 0x3e3d, 0xbd03, 0x0000, 0xbcbf, 0x3e88, 0x3e03, 0xbe0e, 0xbc5d, 0xbc06, 0x3d20, 0x8000, 0xbe62, 0x3e5c],
                0x3c4a87b6,
                0x3eae938c,
            ),
            (
                [0x0000, 0xc035, 0x40b2, 0xbe2d, 0xc079, 0xbeba, 0xbed2, 0x3f17, 0xbeff, 0x3e10, 0xbea0, 0xc0d6, 0x0000, 0xbf20, 0x408a, 0x3e8b],
                [0x3ebb, 0x3e96, 0xbe6e, 0x3e69, 0xbd67, 0xbe6b, 0x3e72, 0x3e36, 0xbd4c, 0xbcdb, 0x3b8d, 0xbece, 0x3da0, 0xbd1b, 0xbdc5, 0x3bc0],
                0x3c451438,
                0x3ef7a3c4,
            ),
            (
                [0x3fde, 0xbf77, 0xc0f3, 0x0000, 0x0000, 0x3ff9, 0xbfd9, 0x3fba, 0xc0a1, 0x3f72, 0xbefd, 0xc05b, 0x3fcc, 0x3f37, 0x3fac, 0x3e16],
                [0x3da1, 0xbc10, 0xbe80, 0xbe3a, 0xbeba, 0x3c0e, 0x3dd1, 0xbc47, 0xbc3b, 0x3d44, 0xbc70, 0x3de1, 0xbe2e, 0xbefc, 0xbc77, 0x8000],
                0x3dcaa198,
                0x3f879c49,
            ),
            (
                [0x3fde, 0xbf77, 0xc0f3, 0x0000, 0x0000, 0x3ff9, 0xbfd9, 0x3fba, 0xc0a1, 0x3f72, 0xbefd, 0xc05b, 0x3fcc, 0x3f37, 0x3fac, 0x3e16],
                [0x3ebb, 0x3e96, 0xbe6e, 0x3e69, 0xbd67, 0xbe6b, 0x3e72, 0x3e36, 0xbd4c, 0xbcdb, 0x3b8d, 0xbece, 0x3da0, 0xbd1b, 0xbdc5, 0x3bc0],
                0x3e413e35,
                0x405237bc,
            ),
            (
                [0x0000, 0x3fdc, 0xbf98, 0x0000, 0x3f2c, 0xbf26, 0xbed2, 0x3fc6, 0x3f99, 0x40b8, 0xbe20, 0x3e4c, 0x8000, 0x4089, 0x3fd0, 0x40cc],
                [0x3e3a, 0x3c83, 0x3e3e, 0x3e3d, 0xbd03, 0x0000, 0xbcbf, 0x3e88, 0x3e03, 0xbe0e, 0xbc5d, 0xbc06, 0x3d20, 0x8000, 0xbe62, 0x3e5c],
                0xbe5ee576,
                0x3eb5fe65,
            ),
            (
                [0x0000, 0x3fdc, 0xbf98, 0x0000, 0x3f2c, 0xbf26, 0xbed2, 0x3fc6, 0x3f99, 0x40b8, 0xbe20, 0x3e4c, 0x8000, 0x4089, 0x3fd0, 0x40cc],
                [0x3c4d, 0xbb83, 0x8000, 0x3edd, 0x3e9f, 0x3dc6, 0x8000, 0x3c12, 0x3d5a, 0x3eae, 0x3ce1, 0xbb81, 0xbdb9, 0xbdb7, 0x3e04, 0xbe59],
                0x00000000,
                0x3f24382a,
            ),
            (
                [0xc0d4, 0xc016, 0xbeae, 0x3fa7, 0x406e, 0x0000, 0x8000, 0xbe3e, 0xbff0, 0xbf07, 0xbfac, 0xc0a6, 0x3f47, 0xc090, 0x4091, 0x3fbc],
                [0x3da1, 0xbc10, 0xbe80, 0xbe3a, 0xbeba, 0x3c0e, 0x3dd1, 0xbc47, 0xbc3b, 0x3d44, 0xbc70, 0x3de1, 0xbe2e, 0xbefc, 0xbc77, 0x8000],
                0xbcc0f5d8,
                0xbf10699c,
            ),
            (
                [0xc0d4, 0xc016, 0xbeae, 0x3fa7, 0x406e, 0x0000, 0x8000, 0xbe3e, 0xbff0, 0xbf07, 0xbfac, 0xc0a6, 0x3f47, 0xc090, 0x4091, 0x3fbc],
                [0x3cf8, 0xbd57, 0xbe70, 0x3d1c, 0xbdbc, 0x3c81, 0xbef6, 0xbda0, 0x3c9e, 0xbdac, 0x3df6, 0x3deb, 0x3ce7, 0x3d56, 0xbcbe, 0x3c98],
                0xbf25844b,
                0xbffb07e1,
            ),
            (
                [0xc0d4, 0xc016, 0xbeae, 0x3fa7, 0x406e, 0x0000, 0x8000, 0xbe3e, 0xbff0, 0xbf07, 0xbfac, 0xc0a6, 0x3f47, 0xc090, 0x4091, 0x3fbc],
                [0xbd77, 0x3d76, 0xbbb0, 0xbc7d, 0x8000, 0xbec4, 0xbc77, 0x3ef6, 0x0000, 0x0000, 0xbe77, 0xbb8a, 0xbbef, 0x0000, 0x3ec0, 0x3e0e],
                0x00000000,
                0x40193efd,
            ),
            (
                [0xc0d4, 0xc016, 0xbeae, 0x3fa7, 0x406e, 0x0000, 0x8000, 0xbe3e, 0xbff0, 0xbf07, 0xbfac, 0xc0a6, 0x3f47, 0xc090, 0x4091, 0x3fbc],
                [0x3e3a, 0x3c83, 0x3e3e, 0x3e3d, 0xbd03, 0x0000, 0xbcbf, 0x3e88, 0x3e03, 0xbe0e, 0xbc5d, 0xbc06, 0x3d20, 0x8000, 0xbe62, 0x3e5c],
                0xbbd115c0,
                0xbfffbf85,
            ),
        ];
        let f = |b: u16| f32::from_bits((b as u32) << 16);
        let ex = |v: f32| ((v.to_bits() >> 23) & 0xFF).max(1) as i32 - 127;
        for (i, (a, b, c, d)) in A10.iter().enumerate() {
            let p = |k: usize| {
                let (x, y) = (f(a[k]), f(b[k]));
                (x * y, if x == 0.0 || y == 0.0 { 0 } else { ex(x) + ex(y) })
            };
            let lo: Vec<(f32, i32)> = (0..8).map(p).collect();
            let hi: Vec<(f32, i32)> = (8..16).map(p).collect();
            let got = mma8(mma8(f32::from_bits(*c), &lo), &hi);
            let fma = (0..16).fold(f32::from_bits(*c), |v, k| f(a[k]).mul_add(f(b[k]), v));
            assert_eq!(got.to_bits(), *d, "output {i}: {got:e} against the A10's {:e}", f32::from_bits(*d));
            assert_ne!(fma.to_bits(), *d, "output {i}");
        }
    }

    /// The time of Kimi-K3's expert (6144 x 3584 and 3584 x 3072) a kernel and a crew take, 16 pairs (`cargo test --release -- --ignored fp4_expert_time --nocapture`).
    #[test]
    #[ignore]
    fn fp4_expert_time() {
        let (latent, inter) = (3584, 3072);
        let ups: Vec<_> = (0..16).map(|i| random_pack(2 * inter, latent, true, 100 + i)).collect();
        let downs: Vec<_> = (0..16).map(|i| random_pack(latent, inter, true, 200 + i)).collect();
        fn view(p: &glyd_gpu::pack::Tiered, r: usize, c: usize) -> Fp4View<'_> {
            Fp4View { data: &p.data, blocks: &p.blocks, rows: r, cols: c, packed: true }
        }
        let z: Vec<u16> = (0..latent).map(|i| (bf16(((i * 13 % 29) as f32 - 14.0) / 64.0).to_bits() >> 16) as u16).collect();
        let pairs: Vec<Fp4Pair> = (0..16)
            .map(|i| Fp4Pair { expert: Fp4Expert { up: view(&ups[i], 2 * inter, latent), down: view(&downs[i], latent, inter) }, token: 0, weight: 0.1 })
            .collect();
        let bytes = 16 * (ups[0].data.len() + ups[0].blocks.len() + downs[0].data.len() + downs[0].blocks.len());
        let threads = std::thread::available_parallelism().map_or(8, |n| n.get());
        for k in Kernel::ALL.into_iter().filter(|k| k.runs_here()) {
            for t in [1, threads] {
                let mut out = vec![0f32; 16 * latent];
                let t0 = std::time::Instant::now();
                experts(
                    k,
                    &z,
                    &pairs[..if t == 1 { 2 } else { 16 }],
                    &mut out[..if t == 1 { 2 } else { 16 } * latent],
                    Act::Situ { beta: 4.0, linear: 25.0 },
                    t,
                    false,
                )
                .unwrap();
                let s = t0.elapsed().as_secs_f64();
                let n = if t == 1 { 2 } else { 16 };
                println!(
                    "{k:?} on {t} threads: {n} experts in {:.2} ms, {:.2} ms an expert, {:.1} GB/s",
                    s * 1e3,
                    s * 1e3 / n as f64,
                    (bytes / 16 * n) as f64 / s / 1e9
                );
            }
        }
    }

    /// The activations' exp and tanh (`exp_one`, `tanh_one`: the GPU kernels' own) against the true values (f64) over every 1,021st f32 up to 88 in size (exp where its result
    /// is a normal number: from `EXP_MIN`; tanh to 50): the most last places apart of each, within the CUDA library's expf's and tanhf's documented 2.
    #[test]
    fn the_activations_exp_and_tanh_are_as_close_as_the_cuda_librarys() {
        let ulps = |got: f32, want: f64| {
            let w = (want as f32).abs();
            (got as f64 - want).abs() / (f32::from_bits(w.to_bits() + 1) - w) as f64
        };
        let (mut e, mut t) = ((0f64, 0f32), (0f64, 0f32));
        for b in (0..0x42b0_0000u32).step_by(1021) {
            for x in [f32::from_bits(b), -f32::from_bits(b)] {
                let u = if x >= EXP_MIN { ulps(exp_one(x), (x as f64).exp()) } else { 0.0 };
                if u > e.0 {
                    e = (u, x);
                }
                if x.abs() <= 50.0 && x != 0.0 {
                    let u = ulps(tanh_one(x), (x as f64).tanh());
                    if u > t.0 {
                        t = (u, x);
                    }
                }
            }
        }
        eprintln!("exp_one: the most {:.3} last places apart (at {:e}); tanh_one: {:.3} (at {:e})", e.0, e.1, t.0, t.1);
        assert!(e.0 <= 2.0 && t.0 <= 2.0, "exp {e:?}, tanh {t:?}");
        assert_eq!((exp_one(0.0), exp_one(-200.0), exp_one(100.0), tanh_one(0.0), tanh_one(-60.0)), (1.0, 0.0, f32::INFINITY, 0.0, -1.0));
    }

    /// A pair's output is the second matrix times situ of the first's halves, its fp32 sum times its weight with no rounding between (the GPU's weighted second
    /// product's row); the threads change nothing.
    #[test]
    fn a_pair_is_its_two_products_and_situ() {
        let (latent, inter) = (128, 64);
        let (up, down) = (random_pack(2 * inter, latent, true, 7), random_pack(latent, inter, true, 9));
        let ex = Fp4Expert {
            up: Fp4View { data: &up.data, blocks: &up.blocks, rows: 2 * inter, cols: latent, packed: true },
            down: Fp4View { data: &down.data, blocks: &down.blocks, rows: latent, cols: inter, packed: true },
        };
        let z: Vec<u16> = (0..2 * latent).map(|i| (bf16(((i * 13 % 29) as f32 - 14.0) / 64.0).to_bits() >> 16) as u16).collect();
        let pairs = [Fp4Pair { expert: ex, token: 1, weight: 0.25 }, Fp4Pair { expert: ex, token: 0, weight: 0.5 }];
        let act = Act::Situ { beta: 4.0, linear: 25.0 };
        let mut one = vec![0f32; 2 * latent];
        experts(Kernel::Scalar, &z, &pairs, &mut one, act, 1, false).unwrap();
        // (by hand, for the first pair: scalar rows of each matrix)
        let zt: Vec<f32> = z[latent..2 * latent].iter().map(|&h| f32::from_bits((h as u32) << 16)).collect();
        let mut gu = vec![0f32; 2 * inter];
        rows(Kernel::Scalar, &ex.up, &zt, 0, 2 * inter, &mut gu);
        let a: Vec<f32> = (0..inter).map(|i| situ(bf16(gu[i]), bf16(gu[inter + i]), (4.0, 25.0))).collect();
        let mut d = vec![0f32; latent];
        rows(Kernel::Scalar, &ex.down, &a, 0, latent, &mut d);
        let want: Vec<f32> = d.iter().map(|&v| v * 0.25).collect();
        assert_eq!(&one[..latent], &want[..]);
        // (a sum rounded before its weight is another row: the host's pair is the GPU's only where it rounds where the GPU does)
        assert!(d.iter().any(|&v| bf16(v) * 0.25 != v * 0.25));
        let mut four = vec![0f32; 2 * latent];
        experts(Kernel::Scalar, &z, &pairs, &mut four, act, 4, false).unwrap();
        assert_eq!(one, four);
    }

    /// A call of eight rows whose pairs share three experts: the pairs are grouped by expert (each expert's bytes read once for its rows, `by_expert`), and every pair's output is the one it has in a call of
    /// its own, to the bit, with each kernel and with the situ and the release's activations.
    #[test]
    fn pairs_of_one_expert_are_computed_together_and_as_alone() {
        let (latent, inter, rows_n) = (128, 128, 8);
        let ups: Vec<_> = (0..3).map(|i| random_pack(2 * inter, latent, true, 40 + i)).collect();
        let downs: Vec<_> = (0..3).map(|i| random_pack(latent, inter, true, 50 + i)).collect();
        let ex = |e: usize| Fp4Expert {
            up: Fp4View { data: &ups[e].data, blocks: &ups[e].blocks, rows: 2 * inter, cols: latent, packed: true },
            down: Fp4View { data: &downs[e].data, blocks: &downs[e].blocks, rows: latent, cols: inter, packed: true },
        };
        let z: Vec<u16> = (0..rows_n * latent).map(|i| (bf16(((i * 13 % 31) as f32 - 15.0) / 32.0).to_bits() >> 16) as u16).collect();
        // (two choices a row among three experts: expert 0 by every row, 1 and 2 by turns)
        let pairs: Vec<Fp4Pair> =
            (0..rows_n).flat_map(|t| [(0, t), (1 + t % 2, t)]).map(|(e, t)| Fp4Pair { expert: ex(e), token: t, weight: 0.125 * (1 + t) as f32 }).collect();
        let groups = by_expert(&pairs);
        assert_eq!(groups.len(), 3);
        assert_eq!(groups[0], (0..rows_n).map(|t| 2 * t).collect::<Vec<_>>());
        assert_eq!(groups.iter().map(Vec::len).sum::<usize>(), pairs.len());
        assert_eq!(ex(0).bytes(), ups[0].data.len() + ups[0].blocks.len() + downs[0].data.len() + downs[0].blocks.len());
        for act in [Act::Situ { beta: 4.0, linear: 25.0 }, Act::Release { lim: 10.0, fp8: true }, Act::Release { lim: 10.0, fp8: false }] {
            for k in Kernel::ALL.into_iter().filter(|k| k.runs_here()) {
                let mut together = vec![0f32; pairs.len() * latent];
                experts(k, &z, &pairs, &mut together, act, 3, false).unwrap();
                for (p, pair) in pairs.iter().enumerate() {
                    let mut alone = vec![0f32; latent];
                    experts(k, &z, std::slice::from_ref(pair), &mut alone, act, 1, false).unwrap();
                    assert_eq!(&together[p * latent..(p + 1) * latent], &alone[..], "{act:?} {k:?} pair {p}");
                }
            }
        }
    }

    /// The release's activation: the weight inside bf16(w silu(min(g, lim)) clamp(u, +-lim)), each block of 128 rounded to E4M3 at its power-of-two scale, and the second product's fp32 the output.
    #[test]
    fn the_release_activation_has_the_weight_inside() {
        let (latent, inter) = (64, 128);
        let (up, down) = (random_pack(2 * inter, latent, true, 3), random_pack(latent, inter, true, 5));
        let ex = Fp4Expert {
            up: Fp4View { data: &up.data, blocks: &up.blocks, rows: 2 * inter, cols: latent, packed: true },
            down: Fp4View { data: &down.data, blocks: &down.blocks, rows: latent, cols: inter, packed: true },
        };
        let z: Vec<u16> = (0..latent).map(|i| (bf16(((i * 7 % 23) as f32 - 11.0) / 8.0).to_bits() >> 16) as u16).collect();
        let (lim, w) = (10.0f32, 0.375f32);
        let mut got = vec![0f32; latent];
        experts(Kernel::Scalar, &z, &[Fp4Pair { expert: ex, token: 0, weight: w }], &mut got, Act::Release { lim, fp8: true }, 1, false).unwrap();
        let zf: Vec<f32> = z.iter().map(|&h| f32::from_bits((h as u32) << 16)).collect();
        let mut gu = vec![0f32; 2 * inter];
        rows(Kernel::Scalar, &ex.up, &zf, 0, 2 * inter, &mut gu);
        let mut h: Vec<f32> = (0..inter)
            .map(|i| {
                let (g, u) = (bf16(gu[i]).min(lim), bf16(gu[inter + i]).clamp(-lim, lim));
                bf16(w * (g / (1.0 + exp_one(-g)) * u))
            })
            .collect();
        fp8_sim(&mut h, 128);
        assert!(h.iter().all(|&v| v == bf16(v)));
        let mut want = vec![0f32; latent];
        rows(Kernel::Scalar, &ex.down, &h, 0, latent, &mut want);
        assert_eq!(got, want);
        assert_eq!((e4m3_round(1.0625), e4m3_round(1.1875), pow2_scale(449.0, 1.0 / 448.0)), (1.0, 1.25, 2.0));
    }
}
