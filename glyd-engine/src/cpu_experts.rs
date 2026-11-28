//! The CPU half of the sparse experts offload (Linear GLY-130): a routed expert of Nemotron 3 Super (40 layers of 512 experts, 22 a token; a latent of 1,024, up [2,688 x 1,024], relu squared with no gate, down [1,024 x 2,688])
//! computed in place from system RAM for a pair (token, expert) whose expert is not on the GPU, where the other way is a copy of its 2.75 MB over PCIe. After relu squared 92% of the 2,688 activations are exactly 0 for a token
//! (benchmarks/weights/offload-probe-2026-10-10), so a pair reads up in full (1,024 values for each of 2,688 neurons), lists the neurons that are not 0 and reads down for those neurons only: down is held transposed, one neuron's
//! 1,024 weights a row. What goes back to the GPU is the pair's 1,024-wide result, 4 KB in fp32: the routing weight times the sum of the active neurons' rows, each times its activation.
//!
//! The row layout (`Rows`): a matrix [rows, cols] (cols a multiple of 32) with each row contiguous, in runs of 32 values with one bf16 scale a run: the codes and the scales that glyd-gpu's `pack_int4`, `pack_int5` and `pack_int6`
//! make for that run of the row (the same fit and rounding), placed a row at a time where the block layouts place them in the tensor cores' steps of 64 rows by 16 columns (`Rows::from_block` places a block pack's own codes and
//! scales, no fit again). `data` is `stride` bytes a row: the runs' codes' low four bits, 16 bytes a run (run k at 16 k: the code of the weight `LANE_LO[p]` of the run in the low nibble of byte p and the code of the weight
//! `LANE_HI[p]` in its high nibble), then at five bits 4 bytes a run (the codes' fifth bit, weight i of the run at bit i / 2 + 16 (i mod 2) of a little-endian word) and at six bits 8 bytes a run (the high two bits, weight i
//! at bits 2 ((i / 2) mod 8) + 16 (i mod 2) of word i / 16), at 16 `runs` + 4 (bits - 4) k; the sum is rounded up to 16 bytes. `scales` is `cols / 32` little-endian bf16 a row, apart from `data`. A weight is
//! `(code - 2^(bits - 1)) scale`; a 1,024-wide row is 512 bytes of codes and 64 of scales at four bits, 640 + 64 at five, 768 + 64 at six.
//!
//! A pair: z's 1,024 values (bf16 bits, or fp32) against each neuron's row of up in fp32, the sum rounded to bf16 (the GPU stores the first product in bf16 where the activation is not fused into it), relu squared of that rounded to
//! bf16 once, the neurons whose activation is not 0 listed in order; then, for those neurons, the activation times the row's scale times its codes added into 1,024 fp32 sums, and the routing weight multiplied in last (the GPU writes
//! its weighted product as `y * w` in fp32). The caller adds a token's pairs in fp32 and rounds to bf16 once, as the GPU's sum does. The weight a code stands for is used as `(code - bias) scale` in fp32, where the GPU rounds that
//! product to bf16 once (exactly for 4 of a four-bit code's 8 magnitudes, otherwise by at most half a bf16 step, 2^-9 to 2^-8 of the weight): that is what lets a run's scale multiply the run's 32 products once, and rounding every
//! weight to bf16 first was not done. Against the pair computed in f64 from the GPU's weights (`unpack_int4` and its kin) a result is off by 1.5e-3 to 3.7e-3 of its norm in the tests' experts (192 to 2,688 neurons), about a bf16 step (the tests print the
//! errors; more where only a few neurons are on, as one activation rounded to the next bf16 is then much of the result), and by 1.5e-7 or less against the same weights unrounded.
//!
//! The order of the float sums is fixed by the layout and the same in every kernel: a row's first product keeps 16 lane sums, lane p taking the run's low-nibble plane's weight p and its high-nibble plane's, each run adding
//! `fma(hi, z_hi, lo * z_lo)` times the run's scale into them, and the 16 lanes are added in a tree (l with l + 8, then l + 4, l + 2, l + 1); an output of the second product is a chain of fused multiply-adds over the active
//! neurons in order. Every kernel (`Kernel`: a scalar one, NEON on aarch64, AVX2 with FMA and AVX-512 on x86-64, chosen at run time) makes the same bits. The work is cut in items of 128 neurons of a pair's first product, which the
//! threads take in turn (this thread is one of them; the workers of each count stay, asleep between calls), and the thread that ends a pair's last item does its second product: the cut changes no sum, so the bits do not depend on
//! the number of threads either.

use crate::cpu_simd as simd;
use crate::ffi::{err, Result};
use glyd_gpu::pack::{int4_value, int5_value, int6_value, Tiered, INT4_STEP_BYTES, INT4_TAG, INT5_STEP_BYTES, INT5_TAG, INT6_STEP_BYTES, INT6_TAG};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, OnceLock, PoisonError};

#[cfg(not(target_endian = "little"))]
compile_error!("the row layout is read as little-endian words");

/// Values in a run (one scale).
pub const RUN: usize = 32;

/// Neurons a work item takes the first product of.
const ROW_BLOCK: usize = 128;

/// The weight of a run that lane p of the low-nibble plane holds (the weight of a run in byte p of its 16 bytes, low nibble), and of the high-nibble plane: the block layouts' lane words in the order of their nibbles.
pub const LANE_LO: [usize; 16] = [0, 4, 1, 5, 8, 12, 9, 13, 16, 20, 17, 21, 24, 28, 25, 29];
pub const LANE_HI: [usize; 16] = [2, 6, 3, 7, 10, 14, 11, 15, 18, 22, 19, 23, 26, 30, 27, 31];

/// The bf16 `w` as an f32 (exact).
#[inline(always)]
pub fn bf16_to_f32(w: u16) -> f32 {
    f32::from_bits((w as u32) << 16)
}

/// `x` rounded to bf16, to nearest and ties to even (a NaN stays one), as an f32.
#[inline(always)]
pub fn bf16_round(x: f32) -> f32 {
    let b = x.to_bits();
    if x.is_nan() {
        return f32::from_bits((b | 0x0040_0000) & 0xFFFF_0000);
    }
    f32::from_bits((b + 0x7FFF + ((b >> 16) & 1)) & 0xFFFF_0000)
}

/// The bytes of codes a row of `cols` values takes at `bits` (4, 5 or 6): 16 + 4 (bits - 4) bytes a run, rounded up to 16.
pub fn row_stride(bits: u32, cols: usize) -> usize {
    (cols / RUN * (16 + 4 * (bits as usize - 4))).next_multiple_of(16)
}

/// The first matrices of `experts` experts held end to end in the block layout of `tag` (`data` and `scales` as `moe::Joined` holds them: each expert's codes, then its scales, expert after expert) placed in the row layout, on
/// `threads` threads (0: as many as the computer has): the codes and the scales of the experts end to end, `rows * row_stride(bits, cols)` and `rows * cols / 32 * 2` bytes an expert. The copy of the GPU's matrices that the
/// kernels below read from the computer's memory.
pub fn rows_of_experts(tag: u32, experts: usize, rows: usize, cols: usize, data: &[u8], scales: &[u8], threads: usize) -> Result<(Vec<u8>, Vec<u8>)> {
    let (ds, ss) = crate::moe::expert_strides(tag, rows, cols)?;
    let (ds, ss) = (ds as usize, ss as usize);
    if experts == 0 || cols % RUN != 0 || data.len() != experts * ds || scales.len() != experts * ss {
        return err(format!(
            "{experts} experts of {rows} x {cols} in the layout of tag {tag} take {} + {} bytes: {} + {} are given",
            experts * ds,
            experts * ss,
            data.len(),
            scales.len()
        ));
    }
    let bits = match tag {
        INT6_TAG => 6,
        INT5_TAG => 5,
        _ => 4,
    };
    let (rd, rs) = (rows * row_stride(bits, cols), rows * (cols / RUN) * 2);
    let (mut od, mut os) = (vec![0u8; experts * rd], vec![0u8; experts * rs]);
    let threads = if threads == 0 { std::thread::available_parallelism().map_or(1, |n| n.get()) } else { threads }.clamp(1, experts);
    let per = experts.div_ceil(threads);
    let failed: Mutex<Option<crate::ffi::Error>> = Mutex::new(None);
    std::thread::scope(|sc| {
        for (c, (d, s)) in od.chunks_mut(per * rd).zip(os.chunks_mut(per * rs)).enumerate() {
            let failed = &failed;
            sc.spawn(move || {
                for (i, (d, s)) in d.chunks_mut(rd).zip(s.chunks_mut(rs)).enumerate() {
                    let e = c * per + i;
                    match Rows::from_block_parts(tag, rows, cols, &data[e * ds..(e + 1) * ds], &scales[e * ss..(e + 1) * ss]) {
                        Ok(r) => {
                            d.copy_from_slice(&r.data);
                            s.copy_from_slice(&r.scales);
                        }
                        Err(x) => {
                            lock(failed).get_or_insert(x);
                            return;
                        }
                    }
                }
            });
        }
    });
    match failed.into_inner().unwrap_or_else(PoisonError::into_inner) {
        Some(x) => Err(x),
        None => Ok((od, os)),
    }
}

/// A matrix [rows, cols] in the row layout of `bits` (4, 5 or 6): see the top of this file.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rows {
    pub bits: u32,
    pub rows: usize,
    pub cols: usize,
    /// rows x `stride()` bytes
    pub data: Vec<u8>,
    /// rows x cols / 32 little-endian bf16
    pub scales: Vec<u8>,
}

impl Rows {
    /// The matrix whose codes are `data` and scales `scales`: the sizes must be a layout's.
    pub fn new(bits: u32, rows: usize, cols: usize, data: Vec<u8>, scales: Vec<u8>) -> Result<Rows> {
        if !(4..=6).contains(&bits) {
            return err(format!("a row layout of {bits} bits: four, five and six are held"));
        }
        if rows == 0 || cols == 0 || !cols.is_multiple_of(RUN) {
            return err(format!("a matrix of {rows} x {cols}: rows, and columns in runs of 32, are wanted"));
        }
        let (stride, runs) = (row_stride(bits, cols), cols / RUN);
        if data.len() != rows * stride || scales.len() != rows * runs * 2 {
            return err(format!(
                "{rows} rows of {cols} at {bits} bits take {} + {} bytes: {} + {} are given",
                rows * stride,
                rows * runs * 2,
                data.len(),
                scales.len()
            ));
        }
        Ok(Rows { bits, rows, cols, data, scales })
    }

    /// The matrix whose weight (r, c) has the code `codes[r * cols + c]` (the packers' u: 0 to 2^bits - 1, bias 2^(bits - 1)) and the scale `scales[r * cols / 32 + c / 32]` (bf16 bits).
    pub fn from_codes(bits: u32, rows: usize, cols: usize, codes: &[u8], scales: &[u16]) -> Result<Rows> {
        if !(4..=6).contains(&bits) || rows == 0 || cols == 0 || !cols.is_multiple_of(RUN) {
            return err(format!("a matrix of {rows} x {cols} at {bits} bits: four, five or six bits, and rows and columns in runs of 32, are wanted"));
        }
        let (runs, stride, hb) = (cols / RUN, row_stride(bits, cols), 4 * (bits as usize - 4));
        if codes.len() != rows * cols || scales.len() != rows * runs || codes.iter().any(|&u| (u as u32) >> bits != 0) {
            return err(format!("{} codes and {} scales for {rows} x {cols} at {bits} bits", codes.len(), scales.len()));
        }
        let mut data = vec![0u8; rows * stride];
        for r in 0..rows {
            let row = &mut data[r * stride..(r + 1) * stride];
            for k in 0..runs {
                let (mut lo, mut hi) = ([0u32; 4], [0u32; 2]);
                for i in 0..RUN {
                    let (pr, h, u) = (i / 2, i % 2, codes[r * cols + RUN * k + i] as u32);
                    lo[pr / 4] |= (u & 15) << (4 * (pr % 4) + 16 * h);
                    match bits {
                        6 => hi[pr / 8] |= (u >> 4) << (2 * (pr % 8) + 16 * h),
                        5 => hi[0] |= (u >> 4) << (pr + 16 * h),
                        _ => {}
                    }
                }
                for (q, v) in lo.iter().enumerate() {
                    row[16 * k + 4 * q..16 * k + 4 * q + 4].copy_from_slice(&v.to_le_bytes());
                }
                for (q, v) in hi.iter().take(hb / 4).enumerate() {
                    let at = 16 * runs + hb * k + 4 * q;
                    row[at..at + 4].copy_from_slice(&v.to_le_bytes());
                }
            }
        }
        Rows::new(bits, rows, cols, data, scales.iter().flat_map(|s| s.to_le_bytes()).collect())
    }

    /// The matrix of a pack in the six-, five- or four-bit block layout (`pack_int6`, `pack_int5`, `pack_int4`), the very codes and scales of its runs placed a row at a time.
    pub fn from_block(p: &Tiered) -> Result<Rows> {
        Rows::from_block_parts(p.tiers[0] >> 24, p.rows, p.cols, &p.data, &p.blocks)
    }

    /// `from_block` for the bytes of a pack in the layout of `tag` that something else holds (an expert's among the cache's experts end to end).
    pub fn from_block_parts(tag: u32, rows: usize, cols: usize, data: &[u8], blocks: &[u8]) -> Result<Rows> {
        let (bits, step_bytes) = match tag {
            INT6_TAG => (6, INT6_STEP_BYTES),
            INT5_TAG => (5, INT5_STEP_BYTES),
            INT4_TAG => (4, INT4_STEP_BYTES),
            t => return err(format!("a pack of layout {t}: the kestrel, linnet and wren layouts are read")),
        };
        let (rbs, ks, runs) = (rows / 64, cols / 16, cols / RUN);
        if rows == 0
            || !rows.is_multiple_of(64)
            || cols == 0
            || !cols.is_multiple_of(RUN)
            || data.len() != rbs * ks * step_bytes
            || blocks.len() != rbs * ks / 2 * 128
        {
            return err(format!(
                "a block pack of {rows} x {cols} with {} + {} bytes is not one of the kestrel, linnet and wren layouts",
                data.len(),
                blocks.len()
            ));
        }
        let (mut codes, mut scales) = (vec![0u8; rows * cols], vec![0u16; rows * runs]);
        let word = |b: &[u8], at: usize| u32::from_le_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]]);
        for rb in 0..rbs {
            for k in 0..ks {
                let step = rb * ks + k;
                let d = &data[step * step_bytes..(step + 1) * step_bytes];
                let b = &blocks[step / 2 * 128..step / 2 * 128 + 128];
                for l in 0..32 {
                    // (the lane's words as unpack_int6, unpack_int5 and unpack_int4 read them: the high bits where there are any, then the low four bits)
                    let (hi, at) = match bits {
                        6 => ([word(d, 4 * l), word(d, 4 * (32 + l))], 256),
                        5 => ([word(d, 4 * l), 0], 128),
                        _ => ([0, 0], 0),
                    };
                    let lo = [word(d, at + 8 * l), word(d, at + 8 * l + 4), word(d, at + 8 * (l + 32)), word(d, at + 8 * (l + 32) + 4)];
                    for i in 0..32 {
                        let (pr, h) = (i / 2, i % 2);
                        let mut u = (lo[pr / 4] >> (4 * (pr % 4) + 16 * h)) & 15;
                        u |= match bits {
                            6 => ((hi[pr / 8] >> (2 * (pr % 8) + 16 * h)) & 3) << 4,
                            5 => ((hi[0] >> (pr + 16 * h)) & 1) << 4,
                            _ => 0,
                        };
                        // (the lane's weight i of the step: row 8 (i / 4) + l / 4, column 8 ((i mod 4) / 2) + 2 (l mod 4) + i mod 2 of the step's 64 x 16)
                        let (rr, cc) = (8 * (i / 4) + l / 4, 8 * ((i % 4) / 2) + 2 * (l % 4) + (i & 1));
                        codes[(rb * 64 + rr) * cols + 16 * k + cc] = u as u8;
                    }
                }
                if k % 2 == 0 {
                    for g in 0..8 {
                        for n in 0..8 {
                            scales[(rb * 64 + 8 * n + g) * runs + k / 2] = u16::from_le_bytes([b[2 * (8 * g + n)], b[2 * (8 * g + n) + 1]]);
                        }
                    }
                }
            }
        }
        Rows::from_codes(bits, rows, cols, &codes, &scales)
    }

    /// Whether the fields are a layout's (`new` makes sure of it; the fields are public, and the kernels read raw memory).
    pub fn is_valid(&self) -> bool {
        (4..=6).contains(&self.bits)
            && self.rows > 0
            && self.cols > 0
            && self.cols.is_multiple_of(RUN)
            && self.data.len() == self.rows * self.stride()
            && self.scales.len() == self.rows * self.runs() * 2
    }

    /// Runs in a row.
    #[inline(always)]
    pub fn runs(&self) -> usize {
        self.cols / RUN
    }

    /// Bytes of codes in a row.
    #[inline(always)]
    pub fn stride(&self) -> usize {
        row_stride(self.bits, self.cols)
    }

    /// Bytes of codes and scales a row takes.
    pub fn row_bytes(&self) -> usize {
        self.stride() + 2 * self.runs()
    }

    /// The code of weight (r, c), read by the layout's formulas.
    pub fn code(&self, r: usize, c: usize) -> u8 {
        let (k, i) = (c / RUN, c % RUN);
        let (runs, stride) = (self.runs(), self.stride());
        let row = &self.data[r * stride..(r + 1) * stride];
        let word = |at: usize| u32::from_le_bytes([row[at], row[at + 1], row[at + 2], row[at + 3]]);
        let (pr, h) = (i / 2, i % 2);
        let mut u = (word(16 * k + 4 * (pr / 4)) >> (4 * (pr % 4) + 16 * h)) & 15;
        u |= match self.bits {
            6 => ((word(16 * runs + 8 * k + 4 * (pr / 8)) >> (2 * (pr % 8) + 16 * h)) & 3) << 4,
            5 => ((word(16 * runs + 4 * k) >> (pr + 16 * h)) & 1) << 4,
            _ => 0,
        };
        u as u8
    }

    /// The scale of run k of row r (bf16 bits).
    pub fn scale(&self, r: usize, k: usize) -> u16 {
        let at = 2 * (r * self.runs() + k);
        u16::from_le_bytes([self.scales[at], self.scales[at + 1]])
    }

    /// The weight (r, c) as the GPU decodes it: bf16 bits of `(code - bias) scale` rounded once.
    pub fn weight(&self, r: usize, c: usize) -> u16 {
        let (u, s) = (self.code(r, c), self.scale(r, c / RUN));
        match self.bits {
            6 => int6_value(u, s),
            5 => int5_value(u, s),
            _ => int4_value(u, s),
        }
    }
}

impl Rows {
    /// The matrix as the kernels read it, borrowed.
    pub fn view(&self) -> RowsView<'_> {
        RowsView { bits: self.bits, rows: self.rows, cols: self.cols, data: &self.data, scales: &self.scales }
    }
}

/// A matrix in the row layout that memory somewhere else holds (the expert cache's host memory, where an expert's codes and scales are a stride's bytes among the layer's): what the kernels read. The fields are those of `Rows`.
#[derive(Clone, Copy, Debug)]
pub struct RowsView<'a> {
    pub bits: u32,
    pub rows: usize,
    pub cols: usize,
    pub data: &'a [u8],
    pub scales: &'a [u8],
}

impl RowsView<'_> {
    /// Whether the fields are a layout's (the kernels read raw memory).
    pub fn is_valid(&self) -> bool {
        (4..=6).contains(&self.bits)
            && self.rows > 0
            && self.cols > 0
            && self.cols.is_multiple_of(RUN)
            && self.data.len() == self.rows * self.stride()
            && self.scales.len() == self.rows * self.runs() * 2
    }

    /// Runs in a row.
    #[inline(always)]
    pub fn runs(&self) -> usize {
        self.cols / RUN
    }

    /// Bytes of codes in a row.
    #[inline(always)]
    pub fn stride(&self) -> usize {
        row_stride(self.bits, self.cols)
    }
}

/// A routed expert: its first matrix `up` and its second one transposed, `down` (row j holds the weights that neuron j's activation multiplies), both [neurons, hidden].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Expert {
    pub up: Rows,
    pub down: Rows,
}

/// An expert's two matrices, borrowed (`Expert::view`, or the cache's host memory).
#[derive(Clone, Copy, Debug)]
pub struct ExpertView<'a> {
    pub up: RowsView<'a>,
    pub down: RowsView<'a>,
}

impl ExpertView<'_> {
    pub fn inter(&self) -> usize {
        self.up.rows
    }

    pub fn hidden(&self) -> usize {
        self.up.cols
    }
}

impl Expert {
    pub fn new(up: Rows, down: Rows) -> Result<Expert> {
        if (up.rows, up.cols) != (down.rows, down.cols) {
            return err(format!(
                "an expert's up is {} x {} and its transposed down {} x {}: both are [neurons, hidden]",
                up.rows, up.cols, down.rows, down.cols
            ));
        }
        Ok(Expert { up, down })
    }

    /// Neurons (the intermediate size) and the width of the input and the output.
    pub fn inter(&self) -> usize {
        self.up.rows
    }

    pub fn hidden(&self) -> usize {
        self.up.cols
    }

    pub fn view(&self) -> ExpertView<'_> {
        ExpertView { up: self.up.view(), down: self.down.view() }
    }
}

/// The experts' inputs, one row of `hidden` values a token: bf16 bits, or fp32 (used as they are).
#[derive(Clone, Copy, Debug)]
pub enum Z<'a> {
    F32(&'a [f32]),
    Bf16(&'a [u16]),
}

impl Z<'_> {
    fn len(&self) -> usize {
        match self {
            Z::F32(v) => v.len(),
            Z::Bf16(v) => v.len(),
        }
    }

    /// Token t's row, as fp32, in the order the kernels read it (see `LANE_LO`): lane p of a run's low-nibble plane, then of its high-nibble plane.
    fn row_permuted(&self, t: usize, hidden: usize, out: &mut [f32]) {
        let at = |i: usize| match self {
            Z::F32(v) => v[t * hidden + i],
            Z::Bf16(v) => bf16_to_f32(v[t * hidden + i]),
        };
        for k in 0..hidden / RUN {
            for p in 0..16 {
                out[RUN * k + p] = at(RUN * k + LANE_LO[p]);
                out[RUN * k + 16 + p] = at(RUN * k + LANE_HI[p]);
            }
        }
    }
}

/// A token's use of an expert: token `token` (a row of z) through `expert`, its result times `weight` (the routing weight).
#[derive(Clone, Copy, Debug)]
pub struct Pair<'a> {
    pub expert: &'a Expert,
    pub token: usize,
    pub weight: f32,
}

/// A `Pair` of an expert that is borrowed.
#[derive(Clone, Copy, Debug)]
pub struct PairView<'a> {
    pub expert: ExpertView<'a>,
    pub token: usize,
    pub weight: f32,
}

/// The code that makes the products: the scalar one runs anywhere, the others where the processor has what they use.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kernel {
    Scalar,
    /// aarch64's NEON
    Neon,
    /// x86-64's AVX2 with FMA
    Avx2,
    /// x86-64's AVX-512 (the foundation instructions, with AVX2 and FMA)
    Avx512,
}

impl Kernel {
    pub const ALL: [Kernel; 4] = [Kernel::Scalar, Kernel::Neon, Kernel::Avx2, Kernel::Avx512];

    pub fn name(self) -> &'static str {
        match self {
            Kernel::Scalar => "scalar",
            Kernel::Neon => "neon",
            Kernel::Avx2 => "avx2",
            Kernel::Avx512 => "avx512",
        }
    }

    pub fn parse(s: &str) -> Option<Kernel> {
        Kernel::ALL.into_iter().find(|k| k.name() == s)
    }

    /// Whether this computer can run it.
    pub fn runs_here(self) -> bool {
        match self {
            Kernel::Scalar => true,
            #[cfg(target_arch = "aarch64")]
            // (NEON is part of every aarch64 target: known when compiled, which keeps the standard library's table of feature names out of the program)
            Kernel::Neon => cfg!(target_feature = "neon"),
            #[cfg(target_arch = "x86_64")]
            Kernel::Avx2 => is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma"),
            #[cfg(target_arch = "x86_64")]
            Kernel::Avx512 => is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma"),
            _ => false,
        }
    }

    /// The kernels this computer runs, the scalar one first.
    pub fn available() -> Vec<Kernel> {
        Kernel::ALL.into_iter().filter(|k| k.runs_here()).collect()
    }

    /// The fastest one this computer runs.
    pub fn best() -> Kernel {
        Kernel::ALL.into_iter().rev().find(|k| k.runs_here()).unwrap_or(Kernel::Scalar)
    }
}

/// Whether `kernel` makes the scalar kernel's bits on small experts of each width (random codes and scales, z of bf16 values, 2 tokens and 3 pairs): a kernel that does not has a fault of its own, and its sums are not to be used.
pub fn agrees_with_scalar(kernel: Kernel) -> bool {
    let mut x = 0x9E37_79B9_7F4A_7C15u64;
    let mut next = move || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        x
    };
    let (inter, hidden) = (192usize, 96usize);
    for bits in 4..=6u32 {
        // (random codes, scales between 1 and 1.75: about half the neurons are on)
        let matrix = |next: &mut dyn FnMut() -> u64| {
            let codes: Vec<u8> = (0..inter * hidden).map(|_| (next() % (1 << bits)) as u8).collect();
            let scales: Vec<u16> = (0..inter * hidden / RUN).map(|_| 0x3C00 + (next() % 0x300) as u16).collect();
            Rows::from_codes(bits, inter, hidden, &codes, &scales)
        };
        let made: Result<Vec<Expert>> = (0..2).map(|_| Expert::new(matrix(&mut next)?, matrix(&mut next)?)).collect();
        let Ok(made) = made else { return false };
        let z: Vec<u16> = (0..2 * hidden).map(|_| (0x3F00 + (next() & 0xFF)) as u16).collect();
        let pairs: Vec<Pair> = (0..3).map(|i| Pair { expert: &made[i % 2], token: i % 2, weight: 0.5 + i as f32 }).collect();
        let (mut want, mut got) = (vec![0f32; 3 * hidden], vec![1f32; 3 * hidden]);
        let (a, b) = (experts_with(Kernel::Scalar, Z::Bf16(&z), &pairs, &mut want, 1), experts_with(kernel, Z::Bf16(&z), &pairs, &mut got, 1));
        if !matches!((&a, &b), (Ok(x), Ok(y)) if x == y) || want.iter().zip(&got).any(|(p, q)| p.to_bits() != q.to_bits()) {
            return false;
        }
    }
    true
}

/// The fastest kernel this computer runs that `agrees_with_scalar` (the scalar one at the least), found once: the instruction sets the kernels use were written for and tested on the computers the records name, and a
/// computer with another is told so by a line on stderr and given the next.
pub fn checked_kernel() -> Kernel {
    static CHOSEN: OnceLock<Kernel> = OnceLock::new();
    *CHOSEN.get_or_init(|| {
        for k in Kernel::ALL.into_iter().rev().filter(|k| k.runs_here()) {
            if k == Kernel::Scalar || agrees_with_scalar(k) {
                return k;
            }
            eprintln!("glyd-engine: the {} kernel of the host's experts does not make the scalar kernel's bits on this computer: using the next", k.name());
        }
        Kernel::Scalar
    })
}

/// The pairs' results with the best kernel: `out` [pairs, hidden] gets pair p's 1,024-wide row (the routing weight times the sum of the active neurons' down rows, each times its activation), `z` [tokens, hidden] the inputs, and
/// the threads (0: as many as the computer has) share the work. Returns the number of active neurons of each pair. See [`experts_with`].
pub fn experts(z: Z, pairs: &[Pair], out: &mut [f32], threads: usize) -> Result<Vec<u32>> {
    experts_with(Kernel::best(), z, pairs, out, threads)
}

/// [`experts`] with the kernel asked for (an error where this computer cannot run it).
pub fn experts_with(kernel: Kernel, z: Z, pairs: &[Pair], out: &mut [f32], threads: usize) -> Result<Vec<u32>> {
    let views: Vec<PairView> = pairs.iter().map(|p| PairView { expert: p.expert.view(), token: p.token, weight: p.weight }).collect();
    experts_views(kernel, z, &views, out, threads, false)
}

/// [`experts_with`] for experts that are borrowed; `pin`: the threads of the crew each stay on a core of their own (see `Workers`).
pub fn experts_views(kernel: Kernel, z: Z, pairs: &[PairView], out: &mut [f32], threads: usize, pin: bool) -> Result<Vec<u32>> {
    if !kernel.runs_here() {
        return err(format!("the {} kernel cannot run on this computer", kernel.name()));
    }
    let Some(first) = pairs.first() else { return Ok(Vec::new()) };
    let hidden = first.expert.hidden();
    if !z.len().is_multiple_of(hidden) || out.len() != pairs.len() * hidden {
        return err(format!("z holds {} values and out {}, where rows of {hidden} and {} pairs are wanted", z.len(), out.len(), pairs.len()));
    }
    let tokens = z.len() / hidden;
    // z's rows that the pairs use, in the kernels' order
    let (mut at, mut used) = (vec![usize::MAX; tokens], Vec::new());
    for (p, pair) in pairs.iter().enumerate() {
        let e = pair.expert;
        if !e.up.is_valid() || !e.down.is_valid() || (e.up.rows, e.up.cols) != (e.down.rows, e.down.cols) || e.hidden() != hidden || pair.token >= tokens {
            return err(format!(
                "pair {p}: an expert of width {} and token {}, where {hidden} and 0 to {} are wanted",
                e.hidden(),
                pair.token,
                tokens.saturating_sub(1)
            ));
        }
        if at[pair.token] == usize::MAX {
            at[pair.token] = used.len();
            used.push(pair.token);
        }
    }
    let mut zp = vec![0f32; used.len() * hidden];
    for (a, &t) in used.iter().enumerate() {
        z.row_permuted(t, hidden, &mut zp[a * hidden..(a + 1) * hidden]);
    }
    // the items: a pair's neurons in blocks, the pairs in order
    let mut first_item = vec![0usize];
    for p in pairs {
        first_item.push(first_item.last().unwrap() + p.expert.inter().div_ceil(ROW_BLOCK));
    }
    let total = *first_item.last().unwrap();
    let remaining: Vec<AtomicUsize> = (0..pairs.len()).map(|p| AtomicUsize::new(first_item[p + 1] - first_item[p])).collect();
    let lists: Vec<OnceLock<Vec<(u32, f32)>>> = (0..total).map(|_| OnceLock::new()).collect();
    let active: Vec<AtomicU32> = (0..pairs.len()).map(|_| AtomicU32::new(0)).collect();
    let rows: Vec<Mutex<&mut [f32]>> = out.chunks_mut(hidden).map(Mutex::new).collect();
    // the second product of pair p, once the last block of its neurons is done
    let finish = |p: usize| {
        let (pair, list) = (&pairs[p], &lists[first_item[p]..first_item[p + 1]]);
        let neurons: Vec<(u32, f32)> = list.iter().flat_map(|l| l.get().expect("a block of the pair is done").iter().copied()).collect();
        active[p].store(neurons.len() as u32, Ordering::Relaxed);
        let mut acc = vec![0f32; hidden];
        simd::down(kernel, &pair.expert.down, &neurons, &mut acc);
        let mut row = rows[p].lock().expect("a row of the output");
        for k in 0..hidden / RUN {
            for l in 0..16 {
                row[RUN * k + LANE_LO[l]] = acc[RUN * k + l] * pair.weight;
                row[RUN * k + LANE_HI[l]] = acc[RUN * k + 16 + l] * pair.weight;
            }
        }
    };
    let work = |item: usize| {
        let p = first_item.partition_point(|&s| s <= item) - 1;
        let (pair, b) = (&pairs[p], item - first_item[p]);
        let (r0, r1) = (b * ROW_BLOCK, ((b + 1) * ROW_BLOCK).min(pair.expert.inter()));
        let mut sums = [0f32; ROW_BLOCK];
        let t = at[pair.token];
        simd::up(kernel, &pair.expert.up, &zp[t * hidden..(t + 1) * hidden], r0, r1, &mut sums[..r1 - r0]);
        let mut list = Vec::new();
        for (i, &s) in sums[..r1 - r0].iter().enumerate() {
            let h = bf16_round(s);
            if h > 0.0 {
                let a = bf16_round(h * h);
                if a != 0.0 {
                    list.push(((r0 + i) as u32, a));
                }
            }
        }
        lists[item].set(list).expect("a block is made once");
        if remaining[p].fetch_sub(1, Ordering::AcqRel) == 1 {
            finish(p);
        }
    };
    run(threads, pin, total, &work);
    Ok(active.iter().map(|a| a.load(Ordering::Relaxed)).collect())
}

/// One job of the workers: `items` calls of `f`, taken in turn.
struct Job {
    f: *const (dyn Fn(usize) + Sync),
    items: usize,
    next: AtomicUsize,
    done: AtomicUsize,
    panicked: AtomicUsize,
}

// (`f` points at a closure that `Workers::run` keeps alive until every item is done)
unsafe impl Send for Job {}
unsafe impl Sync for Job {}

struct Shared {
    /// (the job's number, the job)
    job: Mutex<(u64, Option<Arc<Job>>)>,
    wake: Condvar,
    finished: Condvar,
    /// The number of the job last handed out, which workers that are spinning for the next job read (a thread woken from a condition variable on a core that sleeps takes tens of microseconds to run: a call to the crew
    /// every millisecond or two is better served by workers that look for the next job for a while before they sleep).
    gen: AtomicU64,
}

/// How long a worker looks for the next job before it sleeps, and the thread that waits for a job's end looks for it before it sleeps, in microseconds (`GLYD_CREW_SPIN_US`, default 2,000 for a crew kept to cores and 0 for
/// one that is not; `set_spin_us` changes it).
static SPIN_US: AtomicU64 = AtomicU64::new(u64::MAX);

fn spin_us(pin: bool) -> u64 {
    let v = SPIN_US.load(Ordering::Relaxed);
    if v != u64::MAX {
        return v;
    }
    let d = crate::tune!("GLYD_CREW_SPIN_US").and_then(|x| x.parse().ok()).unwrap_or(if pin { 2_000 } else { 0 });
    SPIN_US.store(d, Ordering::Relaxed);
    d
}

/// The spin time of every crew set (microseconds; 0: none: the threads sleep at once).
pub fn set_spin_us(us: u64) {
    SPIN_US.store(us, Ordering::Relaxed);
}

/// Looks at `ready` until it is true or `us` microseconds have passed; whether it is.
fn spin_until(us: u64, ready: &dyn Fn() -> bool) -> bool {
    if ready() {
        return true;
    }
    if us == 0 {
        return false;
    }
    let t0 = std::time::Instant::now();
    loop {
        for _ in 0..64 {
            if ready() {
                return true;
            }
            std::hint::spin_loop();
        }
        if t0.elapsed().as_micros() as u64 >= us {
            return ready();
        }
    }
}

/// A lock that a thread that panicked did not leave unusable: what these locks guard is a counter and a pointer, set whole.
fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Threads that wait for jobs: `run` hands them items and takes items itself. One job at a time.
struct Workers {
    shared: Arc<Shared>,
    busy: Mutex<()>,
    pin: bool,
}

impl Shared {
    /// Calls `f` for the items of `job` that nobody has taken.
    fn take(&self, job: &Job) {
        loop {
            let i = job.next.fetch_add(1, Ordering::Relaxed);
            if i >= job.items {
                return;
            }
            // SAFETY: `f` outlives the job (see `Workers::run`), and an item is called once.
            if catch_unwind(AssertUnwindSafe(|| unsafe { (*job.f)(i) })).is_err() {
                job.panicked.store(1, Ordering::Relaxed);
            }
            if job.done.fetch_add(1, Ordering::AcqRel) + 1 == job.items {
                drop(lock(&self.job));
                self.finished.notify_all();
            }
        }
    }
}

impl Workers {
    /// `threads` workers; `pin`: worker i stays on the core `pin_core(i + 1)` names (the thread that calls `run` takes the first), so that the crew and the threads around it do not trade cores.
    fn new(threads: usize, pin: bool) -> Workers {
        let shared = Arc::new(Shared { job: Mutex::new((0, None)), wake: Condvar::new(), finished: Condvar::new(), gen: AtomicU64::new(0) });
        for i in 0..threads {
            let shared = Arc::clone(&shared);
            std::thread::spawn(move || {
                if pin {
                    pin_to(i + 1);
                }
                let mut seen = 0;
                loop {
                    // (the next job looked for a while before the thread sleeps)
                    spin_until(spin_us(pin), &|| shared.gen.load(Ordering::Acquire) != seen);
                    let job = {
                        let mut g = lock(&shared.job);
                        while g.0 == seen {
                            g = shared.wake.wait(g).unwrap_or_else(PoisonError::into_inner);
                        }
                        seen = g.0;
                        g.1.clone()
                    };
                    if let Some(job) = job {
                        shared.take(&job);
                    }
                }
            });
        }
        Workers { shared, busy: Mutex::new(()), pin }
    }

    /// `f(0)` to `f(items - 1)` on the workers and this thread, returning when all are done (and panicking, then, if one of them did).
    fn run(&self, items: usize, f: &(dyn Fn(usize) + Sync)) {
        let panicked = {
            let _one_at_a_time = lock(&self.busy);
            // SAFETY: the job holds `f` until `done` counts every item, and this block does not end before it does; a worker that comes late finds `next` past the items and calls nothing.
            let f: *const (dyn Fn(usize) + Sync) = unsafe { std::mem::transmute::<&(dyn Fn(usize) + Sync), &'static (dyn Fn(usize) + Sync)>(f) };
            let job = Arc::new(Job { f, items, next: AtomicUsize::new(0), done: AtomicUsize::new(0), panicked: AtomicUsize::new(0) });
            {
                let mut g = lock(&self.shared.job);
                *g = (g.0 + 1, Some(Arc::clone(&job)));
                self.shared.gen.store(g.0, Ordering::Release);
            }
            self.shared.wake.notify_all();
            self.shared.take(&job);
            // (the end of the job looked for a while, then slept for)
            spin_until(spin_us(self.pin), &|| job.done.load(Ordering::Acquire) == items);
            let mut g = lock(&self.shared.job);
            while job.done.load(Ordering::Acquire) != items {
                g = self.shared.finished.wait(g).unwrap_or_else(PoisonError::into_inner);
            }
            g.1 = None;
            job.panicked.load(Ordering::Relaxed) != 0
        };
        assert!(!panicked, "a work item panicked");
    }
}

/// The cores this process may run on, read once before any thread is kept to one (a thread made later inherits its maker's cores, so a mask read later may be one core).
#[cfg(target_os = "linux")]
fn process_cores() -> &'static [usize] {
    static CORES: OnceLock<Vec<usize>> = OnceLock::new();
    CORES.get_or_init(|| {
        let mut mask = [0u64; 16];
        // SAFETY: the mask holds the 1,024 bits of cores the call is told it has; pid 0 is this thread.
        if unsafe { sched_getaffinity(0, std::mem::size_of_val(&mask), mask.as_mut_ptr()) } != 0 {
            return Vec::new();
        }
        (0..1024).filter(|&c| mask[c / 64] >> (c % 64) & 1 == 1).collect()
    })
}

#[cfg(target_os = "linux")]
extern "C" {
    fn sched_getaffinity(pid: i32, size: usize, mask: *mut u64) -> i32;
    fn sched_setaffinity(pid: i32, size: usize, mask: *const u64) -> i32;
}

/// The core the i-th thread of a pinned crew stays on: the i-th of the cores this process may run on (past the last, round again from the second: the first is the thread that calls). None where the system does not say or
/// the process has one core (a pinned crew is Linux's: elsewhere threads are not kept to cores).
#[cfg(target_os = "linux")]
fn pin_core(i: usize) -> Option<usize> {
    let cores = process_cores();
    (cores.len() > 1).then(|| if i == 0 { cores[0] } else { cores[1 + (i - 1) % (cores.len() - 1)] })
}

/// This thread kept to the core `pin_core(i)` names.
#[cfg(target_os = "linux")]
fn pin_to(i: usize) {
    let Some(c) = pin_core(i) else { return };
    let mut mask = [0u64; 16];
    mask[c / 64] = 1u64 << (c % 64);
    // SAFETY: as in `process_cores`; a refusal leaves the thread where it is.
    unsafe { sched_setaffinity(0, std::mem::size_of_val(&mask), mask.as_ptr()) };
}

#[cfg(not(target_os = "linux"))]
fn pin_to(_: usize) {}

/// The thread that calls kept to the first core while it takes part in a job of a pinned crew, and let go to the cores it had after.
struct Held(#[cfg(target_os = "linux")] Option<[u64; 16]>);

impl Held {
    fn new(pin: bool) -> Held {
        #[cfg(target_os = "linux")]
        {
            if !pin {
                return Held(None);
            }
            let _ = process_cores();
            let mut old = [0u64; 16];
            // SAFETY: as in `process_cores`.
            let had = unsafe { sched_getaffinity(0, std::mem::size_of_val(&old), old.as_mut_ptr()) } == 0;
            pin_to(0);
            Held(had.then_some(old))
        }
        #[cfg(not(target_os = "linux"))]
        {
            let _ = pin;
            Held()
        }
    }
}

impl Drop for Held {
    fn drop(&mut self) {
        #[cfg(target_os = "linux")]
        if let Some(old) = &self.0 {
            // SAFETY: as in `process_cores`.
            unsafe { sched_setaffinity(0, std::mem::size_of_val(old), old.as_ptr()) };
        }
    }
}

/// The threads of a crew that the cores a computer has leave: all but two, at least one (the cores the decode loop and the driver's threads do not use).
pub fn default_threads() -> usize {
    std::thread::available_parallelism().map_or(1, |n| n.get()).saturating_sub(2).max(1)
}

/// `f(0)` to `f(items - 1)` on `threads` threads (0: as many as the computer has; this one is one of them). The workers of each count stay for the next call; `pin`: each on a core of its own, this thread too.
pub(crate) fn run(threads: usize, pin: bool, items: usize, f: &(dyn Fn(usize) + Sync)) {
    let threads = if threads == 0 { std::thread::available_parallelism().map_or(1, |n| n.get()) } else { threads };
    if threads.min(items) <= 1 {
        return (0..items).for_each(f);
    }
    static CREWS: Mutex<Vec<(usize, bool, Arc<Workers>)>> = Mutex::new(Vec::new());
    let crew = {
        let mut crews = lock(&CREWS);
        match crews.iter().find(|(n, p, _)| (*n, *p) == (threads, pin)) {
            Some((_, _, w)) => Arc::clone(w),
            None => {
                let w = Arc::new(Workers::new(threads - 1, pin));
                crews.push((threads, pin, Arc::clone(&w)));
                w
            }
        }
    };
    let _held = Held::new(pin);
    crew.run(items, f);
}

#[cfg(test)]
mod tests {
    use super::*;
    use glyd_gpu::pack::{pack_int4, pack_int5, pack_int6, unpack_int4, unpack_int5, unpack_int6};

    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
        }

        /// Uniform in [0, 1).
        fn f(&mut self) -> f32 {
            (self.next() >> 40) as f32 / (1u64 << 24) as f32
        }

        /// A bell: a sum of four uniforms, spread 0.1 about 0.
        fn bell(&mut self) -> f32 {
            (self.f() + self.f() + self.f() + self.f() - 2.0) * 0.1
        }
    }

    fn bits_of(x: f32) -> u16 {
        (bf16_round(x).to_bits() >> 16) as u16
    }

    /// A matrix of bf16 weights, a bell about 0 (`rows_on`: a sign a row, where it is given: a row of `true` is non-negative, one of `false` non-positive).
    fn weights(rng: &mut Rng, rows: usize, cols: usize, rows_on: Option<&[bool]>) -> Vec<u16> {
        let mut w = Vec::with_capacity(rows * cols);
        for r in 0..rows {
            for _ in 0..cols {
                let g = rng.bell();
                w.push(bits_of(match rows_on {
                    None => g,
                    Some(on) => g.abs() * if on[r] { 1.0 } else { -1.0 },
                }));
            }
        }
        w
    }

    fn pack(bits: u32, w: &[u16], rows: usize, cols: usize) -> Tiered {
        match bits {
            6 => pack_int6(w, rows, cols),
            5 => pack_int5(w, rows, cols),
            _ => pack_int4(w, rows, cols),
        }
        .expect("a packable matrix")
    }

    fn unpack(p: &Tiered) -> Vec<u16> {
        let mut out = vec![0u16; p.rows * p.cols];
        let ok = match p.tiers[0] >> 24 {
            INT6_TAG => unpack_int6(p, &mut out),
            INT5_TAG => unpack_int5(p, &mut out),
            _ => unpack_int4(p, &mut out),
        };
        assert!(ok);
        out
    }

    // ---- the layout

    /// Lane p of the low-nibble plane holds the weight the formulas of a run's words give: byte p of the 16 low-bit bytes, in its low nibble; the high-nibble plane in the high nibble.
    #[test]
    fn the_lanes_are_the_layouts() {
        let mut seen = [false; 32];
        for p in 0..16 {
            for (plane, lane) in [(0, LANE_LO[p]), (1, LANE_HI[p])] {
                assert!(!seen[lane]);
                seen[lane] = true;
                // weight i = lane: pair pr = i / 2, half h = i mod 2: nibble 4 (pr mod 4) + 16 h of word pr / 4
                let (pr, h) = (lane / 2, lane % 2);
                let bit = 4 * (pr / 4) * 8 + 4 * (pr % 4) + 16 * h;
                assert_eq!((bit / 8, (bit / 4) % 2), (p, plane), "weight {lane}");
            }
        }
        assert!(seen.iter().all(|&s| s));
    }

    /// The bytes of a run, as the top of the file says them.
    #[test]
    fn the_bytes_of_a_run() {
        let mut rng = Rng(7);
        for bits in 4..=6u32 {
            let (rows, cols) = (3, 96);
            let codes: Vec<u8> = (0..rows * cols).map(|_| (rng.next() % (1 << bits)) as u8).collect();
            let scales: Vec<u16> = (0..rows * cols / 32).map(|_| rng.next() as u16 & 0x7FFF).collect();
            let m = Rows::from_codes(bits, rows, cols, &codes, &scales).unwrap();
            let (runs, stride) = (cols / 32, 16 * (cols / 32) + 4 * (bits as usize - 4) * (cols / 32));
            assert_eq!((m.stride(), m.data.len()), (stride.next_multiple_of(16), rows * stride.next_multiple_of(16)));
            for r in 0..rows {
                let row = &m.data[r * m.stride()..(r + 1) * m.stride()];
                for k in 0..runs {
                    let c = |i: usize| codes[r * cols + 32 * k + i] as u32;
                    for p in 0..16 {
                        assert_eq!(row[16 * k + p] as u32, (c(LANE_LO[p]) & 15) | (c(LANE_HI[p]) & 15) << 4);
                    }
                    // the high bits: five bits a word of the run's i-th weight at bit i / 2 + 16 (i mod 2); six bits two words, 2 ((i / 2) mod 8) + 16 (i mod 2) of word i / 16
                    for i in 0..32 {
                        let word = |q: usize, hb: usize| u32::from_le_bytes(row[16 * runs + hb * k + 4 * q..][..4].try_into().unwrap());
                        match bits {
                            5 => assert_eq!((word(0, 4) >> (i / 2 + 16 * (i % 2))) & 1, c(i) >> 4),
                            6 => assert_eq!((word(i / 16, 8) >> (2 * ((i / 2) % 8) + 16 * (i % 2))) & 3, c(i) >> 4),
                            _ => {}
                        }
                    }
                }
                for k in 0..runs {
                    assert_eq!(m.scale(r, k), scales[r * runs + k]);
                    assert_eq!(&m.scales[2 * (r * runs + k)..][..2], &scales[r * runs + k].to_le_bytes());
                }
                for c in 0..cols {
                    assert_eq!(m.code(r, c), codes[r * cols + c]);
                }
            }
        }
        assert!(Rows::from_codes(4, 1, 32, &[16; 32], &[0]).is_err(), "a code past the bits");
        assert!(Rows::from_codes(7, 1, 32, &[0; 32], &[0]).is_err() && Rows::from_codes(4, 1, 33, &[0; 33], &[0]).is_err());
        assert!(Rows::new(4, 2, 64, vec![0; 2 * 32], vec![0; 2 * 2 * 2]).is_ok() && Rows::new(4, 2, 64, vec![0; 2 * 32], vec![0; 7]).is_err());
    }

    /// The row layout holds the codes and scales that pack_int6, pack_int5 and pack_int4 make: the weights (r, c) decode to what unpack gives, and a code is the packer's rule's, from the matrix and the scale of its run.
    #[test]
    fn the_row_layout_is_the_packers_codes_and_scales() {
        let mut rng = Rng(11);
        for bits in 4..=6u32 {
            let (rows, cols) = (128, 192);
            let w = weights(&mut rng, rows, cols, None);
            let p = pack(bits, &w, rows, cols);
            let m = Rows::from_block(&p).unwrap();
            let back = unpack(&p);
            let q = ((1u32 << (bits - 1)) - 1) as f32;
            for r in 0..rows {
                for c in 0..cols {
                    assert_eq!(m.weight(r, c), back[r * cols + c], "bits {bits} weight ({r}, {c})");
                    // (the packer's rule: clamp(round_half_even(w / s), -q, q) + q + 1; a run with no scale is all q + 1)
                    let s = bf16_to_f32(m.scale(r, c / 32));
                    let want = if s == 0.0 { q as i32 + 1 } else { (bf16_to_f32(w[r * cols + c]) / s).round_ties_even().clamp(-q, q) as i32 + q as i32 + 1 };
                    assert_eq!(m.code(r, c) as i32, want, "bits {bits} code ({r}, {c})");
                }
            }
        }
        // a matrix that is not a block pack
        let mut p = pack(4, &weights(&mut rng, 64, 32, None), 64, 32);
        p.tiers[0] = 2 << 24;
        assert!(Rows::from_block(&p).is_err());
    }

    #[test]
    fn rounding_to_bf16() {
        for (x, want) in [
            (0x3F80_8000u32, 0x3F80_0000u32),
            (0x3F81_8000, 0x3F82_0000),
            (0x3F80_8001, 0x3F81_0000),
            (0xBF80_8000, 0xBF80_0000),
            (0x7F80_0000, 0x7F80_0000),
            (0x7F7F_FFFF, 0x7F80_0000),
            (0x0000_8000, 0),
            (0x0000_8001, 0x0001_0000),
        ] {
            assert_eq!(bf16_round(f32::from_bits(x)).to_bits(), want, "{x:#x}");
        }
        assert!(bf16_round(f32::NAN).is_nan() && bf16_round(f32::from_bits(0x7F80_0001)).is_nan() && bf16_round(f32::from_bits(0xFFFF_FFFF)).is_nan());
        for _ in 0..1000 {
            let v = f32::from_bits(Rng(5).next() as u32);
            assert!(v.is_nan() || bf16_round(bf16_round(v)).to_bits() == bf16_round(v).to_bits());
        }
    }

    // ---- the products

    /// An expert of random weights, and the weights the GPU decodes for them (bf16 as f64, from glyd-gpu's unpack): up [inter, hidden] and the transposed down [inter, hidden].
    struct Made {
        expert: Expert,
        up: Vec<f64>,
        down: Vec<f64>,
    }

    fn make(bits: u32, rng: &mut Rng, inter: usize, hidden: usize, up_rows_on: Option<&[bool]>, down_rows_on: Option<&[bool]>) -> Made {
        let (pu, pd) =
            (pack(bits, &weights(rng, inter, hidden, up_rows_on), inter, hidden), pack(bits, &weights(rng, inter, hidden, down_rows_on), inter, hidden));
        let f = |p: &Tiered| unpack(p).into_iter().map(|b| bf16_to_f32(b) as f64).collect();
        Made { up: f(&pu), down: f(&pd), expert: Expert::new(Rows::from_block(&pu).unwrap(), Rows::from_block(&pd).unwrap()).unwrap() }
    }

    /// An expert whose up rows are on (against a positive z) with the chance `active`.
    fn make_sparse(bits: u32, rng: &mut Rng, inter: usize, hidden: usize, active: f32) -> Made {
        let rows = mask(rng, inter, active);
        make(bits, rng, inter, hidden, Some(&rows), None)
    }

    /// z: bf16 values, positive (experts whose rows have a sign make their rows' sums that sign) or of both.
    fn input(rng: &mut Rng, tokens: usize, hidden: usize, positive: bool) -> Vec<f32> {
        (0..tokens * hidden).map(|_| bf16_round(if positive { 0.25 + 1.5 * rng.f() } else { 4.0 * (rng.f() - 0.5) })).collect()
    }

    /// The pair by the book, in f64 from the GPU's weights: the first product's sum rounded to bf16, relu squared rounded to bf16 once, the second product's sums times the routing weight. Also the active neurons' number.
    fn reference(m: &Made, z: &[f32], w: f32) -> (Vec<f64>, u32) {
        let (inter, hidden) = (m.expert.inter(), m.expert.hidden());
        let act: Vec<f32> = (0..inter)
            .map(|j| {
                let h = bf16_round((0..hidden).map(|c| m.up[j * hidden + c] * z[c] as f64).sum::<f64>() as f32);
                if h > 0.0 {
                    bf16_round(h * h)
                } else {
                    0.0
                }
            })
            .collect();
        let out = (0..hidden).map(|o| w as f64 * (0..inter).map(|j| act[j] as f64 * m.down[j * hidden + o]).sum::<f64>()).collect();
        (out, act.iter().filter(|&&a| a != 0.0).count() as u32)
    }

    /// The relative error in L2 (the vectors' norm if the reference is 0).
    fn rel(got: &[f32], want: &[f64]) -> f64 {
        let (e, n) = got.iter().zip(want).fold((0.0, 0.0), |(e, n), (&g, &w)| (e + (g as f64 - w).powi(2), n + w * w));
        if n == 0.0 {
            e.sqrt()
        } else {
            (e / n).sqrt()
        }
    }

    fn mask(rng: &mut Rng, n: usize, active: f32) -> Vec<bool> {
        (0..n).map(|_| rng.f() < active).collect()
    }

    fn exact_weights(m: &Rows) -> Vec<f64> {
        let bias = (1i32 << (m.bits - 1)) as f64;
        (0..m.rows * m.cols).map(|i| (m.code(i / m.cols, i % m.cols) as f64 - bias) * bf16_to_f32(m.scale(i / m.cols, i % m.cols / RUN)) as f64).collect()
    }

    /// `reference` with the weights as `(code - bias) scale` and no rounding of that product to bf16 (what the kernels use).
    fn reference_exact(m: &Made, z: &[f32], w: f32) -> Vec<f64> {
        let exact = Made { expert: m.expert.clone(), up: exact_weights(&m.expert.up), down: exact_weights(&m.expert.down) };
        reference(&exact, z, w).0
    }

    /// Every kernel against the pair by the book in f64 from the GPU's weights, at 4, 5 and 6 bits: about 90% of the neurons off (rows of one sign against a positive z), half off, all off, all on, and weights and z of both signs.
    /// What the kernels differ by is the weights' bf16 rounding (they use `(code - bias) scale` as it is) and the float sums' order; against the same weights unrounded the error is the sums' alone.
    #[test]
    fn pairs_against_float64() {
        let mut rng = Rng(21);
        let (inter, hidden) = (192, 256);
        let mut worst = (0f64, 0f64);
        for bits in 4..=6u32 {
            for (name, active) in [("sparse", Some(0.1)), ("half", Some(0.5)), ("none", Some(0.0)), ("all", Some(1.0)), ("mixed", None)] {
                let rows = active.map(|a| mask(&mut rng, inter, a));
                let m = make(bits, &mut rng, inter, hidden, rows.as_deref(), None);
                let z = input(&mut rng, 1, hidden, active.is_some());
                let (want, n) = reference(&m, &z, 2.5);
                let exact = reference_exact(&m, &z, 2.5);
                for k in Kernel::available() {
                    let mut out = vec![f32::NAN; hidden];
                    let got = experts_with(k, Z::F32(&z), &[Pair { expert: &m.expert, token: 0, weight: 2.5 }], &mut out, 1).unwrap();
                    let (e_gpu, e_exact) = (rel(&out, &want), rel(&out, &exact));
                    worst = (worst.0.max(e_gpu), worst.1.max(e_exact));
                    eprintln!("bits {bits} {name:6} {:7}: {} active (by the book {n}), error {e_gpu:.2e} against the GPU's weights, {e_exact:.2e} against the same weights unrounded", k.name(), got[0]);
                    if active.is_some() {
                        assert_eq!(got[0], n, "bits {bits} {name} {}", k.name());
                    } else {
                        assert!(got[0].abs_diff(n) <= 2, "bits {bits} {name} {}: {} and {n}", k.name(), got[0]);
                    }
                    assert!(e_gpu < 6e-3 && e_exact < 1e-4, "bits {bits} {name} {}: {e_gpu} and {e_exact}", k.name());
                    if name == "none" {
                        assert!(out.iter().all(|&x| x == 0.0), "no active neuron, no output");
                    }
                }
            }
        }
        eprintln!("worst error: {:.2e} against the GPU's weights, {:.2e} unrounded", worst.0, worst.1);
    }

    /// Every kernel makes the bits the scalar one does: experts and z (fp32 that are not bf16 values) of 4, 5 and 6 bits, a neuron count that is not a block's, pairs of several tokens and experts.
    #[test]
    fn kernels_make_the_same_bits() {
        let mut rng = Rng(33);
        let (inter, hidden) = (320, 160);
        for bits in 4..=6u32 {
            let made: Vec<Made> = (0..4).map(|e| make_sparse(bits, &mut rng, inter, hidden, 0.05 + 0.25 * e as f32)).collect();
            let z: Vec<f32> = (0..3 * hidden).map(|_| 2.0 * (rng.f() - 0.1)).collect();
            let pairs: Vec<Pair> = (0..9).map(|i| Pair { expert: &made[(i * 5) % 4].expert, token: (i * 2) % 3, weight: 0.5 + rng.f() }).collect();
            let mut want = vec![0f32; pairs.len() * hidden];
            let n_want = experts_with(Kernel::Scalar, Z::F32(&z), &pairs, &mut want, 1).unwrap();
            assert!(n_want.iter().any(|&n| n > 0) && want.iter().any(|&x| x != 0.0));
            for k in Kernel::available() {
                let mut got = vec![0f32; want.len()];
                let n = experts_with(k, Z::F32(&z), &pairs, &mut got, 1).unwrap();
                assert_eq!(n, n_want, "bits {bits} {}", k.name());
                assert!(got.iter().zip(&want).all(|(a, b)| a.to_bits() == b.to_bits()), "bits {bits}: the {} kernel differs from the scalar one", k.name());
            }
        }
    }

    /// The threads change no bit: the work cut in other items, taken in other turns, is the same sums.
    #[test]
    fn threads_change_no_bit() {
        let mut rng = Rng(44);
        let (inter, hidden) = (576, 128);
        let made: Vec<Made> = (0..5).map(|e| make_sparse(4, &mut rng, inter, hidden, 0.04 * (1 + e) as f32)).collect();
        let z: Vec<f32> = (0..4 * hidden).map(|_| bf16_round(0.25 + rng.f())).collect();
        let pairs: Vec<Pair> = (0..11).map(|i| Pair { expert: &made[(i * 3) % 5].expert, token: i % 4, weight: 1.0 + rng.f() }).collect();
        let mut want = vec![0f32; pairs.len() * hidden];
        let n_want = experts(Z::F32(&z), &pairs, &mut want, 1).unwrap();
        for threads in [2, 3, 7, 10, 0, 64] {
            let mut got = vec![1f32; want.len()];
            assert_eq!(experts(Z::F32(&z), &pairs, &mut got, threads).unwrap(), n_want);
            assert!(got.iter().zip(&want).all(|(a, b)| a.to_bits() == b.to_bits()), "{threads} threads");
        }
        // one pair, fewer items than threads; and none
        let mut one = vec![0f32; hidden];
        experts(Z::F32(&z), &pairs[..1], &mut one, 8).unwrap();
        assert!(one.iter().zip(&want[..hidden]).all(|(a, b)| a.to_bits() == b.to_bits()));
        assert!(experts(Z::F32(&z), &[], &mut [], 4).unwrap().is_empty());
    }

    /// z given as bf16 bits is z given as the fp32 values of them.
    #[test]
    fn z_in_bf16() {
        let mut rng = Rng(55);
        let (inter, hidden) = (128, 64);
        let m = make(4, &mut rng, inter, hidden, None, None);
        let zb: Vec<u16> = (0..2 * hidden).map(|_| bits_of(4.0 * (rng.f() - 0.5))).collect();
        let zf: Vec<f32> = zb.iter().map(|&b| bf16_to_f32(b)).collect();
        let pairs = [Pair { expert: &m.expert, token: 1, weight: 1.5 }, Pair { expert: &m.expert, token: 0, weight: 0.5 }];
        let (mut a, mut b) = (vec![0f32; 2 * hidden], vec![0f32; 2 * hidden]);
        assert_eq!(experts(Z::Bf16(&zb), &pairs, &mut a, 2).unwrap(), experts(Z::F32(&zf), &pairs, &mut b, 2).unwrap());
        assert!(a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits()) && a.iter().any(|&x| x != 0.0));
    }

    #[test]
    fn what_is_refused() {
        let mut rng = Rng(66);
        let (a, b) = (make(4, &mut rng, 64, 64, None, None), make(4, &mut rng, 64, 32, None, None));
        let z = vec![0.5f32; 2 * 64];
        let ok = [Pair { expert: &a.expert, token: 1, weight: 1.0 }];
        let mut out = vec![0f32; 64];
        assert!(experts(Z::F32(&z), &ok, &mut out, 1).is_ok());
        assert!(experts(Z::F32(&z), &ok, &mut out[..63], 1).is_err(), "out too short");
        assert!(experts(Z::F32(&z[..100]), &ok, &mut out, 1).is_err(), "z not rows of 64");
        assert!(experts(Z::F32(&z), &[Pair { token: 2, ..ok[0] }], &mut out, 1).is_err(), "a token past z");
        assert!(experts(Z::F32(&z), &[ok[0], Pair { expert: &b.expert, token: 0, weight: 1.0 }], &mut vec![0f32; 128], 1).is_err(), "experts of other widths");
        if let Some(k) = Kernel::ALL.into_iter().find(|k| !k.runs_here()) {
            assert!(experts_with(k, Z::F32(&z), &ok, &mut out, 1).is_err(), "{} is not here", k.name());
        }
        assert!(Expert::new(a.expert.up.clone(), b.expert.down.clone()).is_err());
        assert_eq!(Kernel::parse("scalar"), Some(Kernel::Scalar));
        assert!(Kernel::available()[0] == Kernel::Scalar && Kernel::available().contains(&Kernel::best()));
    }

    /// Callers on several threads at once (the workers take one job at a time) get what each alone does.
    #[test]
    fn callers_at_once() {
        let mut rng = Rng(77);
        let (inter, hidden) = (256, 128);
        let m = make_sparse(5, &mut rng, inter, hidden, 0.1);
        let z: Vec<f32> = (0..hidden).map(|_| bf16_round(0.25 + rng.f())).collect();
        let pairs: Vec<Pair> = (0..6).map(|i| Pair { expert: &m.expert, token: 0, weight: 0.5 + i as f32 }).collect();
        let mut want = vec![0f32; pairs.len() * hidden];
        experts(Z::F32(&z), &pairs, &mut want, 1).unwrap();
        std::thread::scope(|s| {
            for t in 0..6 {
                let (z, pairs, want) = (&z, &pairs, &want);
                s.spawn(move || {
                    for _ in 0..20 {
                        let mut got = vec![0f32; want.len()];
                        experts(Z::F32(z), pairs, &mut got, 2 + t % 3).unwrap();
                        assert!(got.iter().zip(want).all(|(a, b)| a.to_bits() == b.to_bits()));
                    }
                });
            }
        });
    }

    /// An expert of Nemotron 3 Super's shape (2,688 neurons of 1,024) at four bits, about 92% of its neurons off, all kernels and many threads, against f64.
    #[test]
    fn an_expert_of_the_models_shape() {
        let mut rng = Rng(88);
        let (inter, hidden) = (2688, 1024);
        let m = make_sparse(4, &mut rng, inter, hidden, 0.082);
        let z = input(&mut rng, 1, hidden, true);
        let (want, n) = reference(&m, &z, 5.0);
        assert!((100..400).contains(&n), "{n} neurons on");
        for k in Kernel::available() {
            let mut out = vec![0f32; hidden];
            assert_eq!(experts_with(k, Z::F32(&z), &[Pair { expert: &m.expert, token: 0, weight: 5.0 }], &mut out, 5).unwrap(), [n]);
            let e = rel(&out, &want);
            eprintln!("{}: {n} neurons on, error {e:.2e}", k.name());
            assert!(e < 6e-3, "{} {e}", k.name());
        }
    }

    /// Experts held end to end in a block layout are placed in the row layout one expert at a time, on any number of threads: each expert's bytes are `Rows::from_block`'s of its own pack.
    #[test]
    fn experts_end_to_end_are_placed_in_rows() {
        let mut rng = Rng(99);
        let (n, rows, cols) = (5, 128, 96);
        for bits in 4..=6u32 {
            let packs: Vec<Tiered> = (0..n).map(|_| pack(bits, &weights(&mut rng, rows, cols, None), rows, cols)).collect();
            let (data, scales): (Vec<u8>, Vec<u8>) =
                (packs.iter().flat_map(|p| p.data.clone()).collect(), packs.iter().flat_map(|p| p.blocks.clone()).collect());
            let tag = packs[0].tiers[0] >> 24;
            let own: Vec<Rows> = packs.iter().map(|p| Rows::from_block(p).unwrap()).collect();
            for threads in [1, 2, 7, 0] {
                let (d, s) = rows_of_experts(tag, n, rows, cols, &data, &scales, threads).unwrap();
                let (rd, rs) = (own[0].data.len(), own[0].scales.len());
                assert!(d.len() == n * rd && s.len() == n * rs);
                for (e, r) in own.iter().enumerate() {
                    assert!(d[e * rd..(e + 1) * rd] == r.data[..] && s[e * rs..(e + 1) * rs] == r.scales[..], "bits {bits}, {threads} threads: expert {e}");
                }
            }
            assert!(rows_of_experts(tag, n, rows, cols, &data[1..], &scales, 2).is_err() && rows_of_experts(tag, 0, rows, cols, &[], &[], 2).is_err());
            assert!(rows_of_experts(INT4_TAG + 4, n, rows, cols, &data, &scales, 2).is_err(), "a layout that is not a block one");
        }
    }

    /// Every kernel this computer runs makes the scalar kernel's bits on the small experts the check uses, and the kernel chosen is one of them.
    #[test]
    fn the_checked_kernel_is_one_that_agrees() {
        for k in Kernel::available() {
            assert!(agrees_with_scalar(k), "{}", k.name());
        }
        assert!(Kernel::available().contains(&checked_kernel()));
        if let Some(k) = Kernel::ALL.into_iter().find(|k| !k.runs_here()) {
            assert!(!agrees_with_scalar(k), "{} cannot run here", k.name());
        }
    }

    /// Experts that are borrowed from bytes end to end are the owned experts: the same bits as `experts`, on a crew kept to cores and not.
    #[test]
    fn borrowed_experts_make_the_same_bits() {
        let mut rng = Rng(101);
        let (inter, hidden) = (320, 160);
        let made: Vec<Made> = (0..4).map(|e| make_sparse(4, &mut rng, inter, hidden, 0.05 + 0.25 * e as f32)).collect();
        let z: Vec<u16> = (0..3 * hidden).map(|_| bits_of(2.0 * (rng.f() - 0.1))).collect();
        // the experts' matrices end to end in one buffer each, as the cache's host memory holds them
        let cat = |f: &dyn Fn(&Expert) -> &Rows| -> (Vec<u8>, Vec<u8>) {
            (made.iter().flat_map(|m| f(&m.expert).data.clone()).collect(), made.iter().flat_map(|m| f(&m.expert).scales.clone()).collect())
        };
        let (ud, us) = cat(&|e| &e.up);
        let (dd, ds) = cat(&|e| &e.down);
        let rd = made[0].expert.up.data.len();
        fn view<'a>(m: &Rows, d: &'a [u8], s: &'a [u8], e: usize) -> RowsView<'a> {
            let (rd, rs) = (m.data.len(), m.scales.len());
            RowsView { bits: m.bits, rows: m.rows, cols: m.cols, data: &d[e * rd..(e + 1) * rd], scales: &s[e * rs..(e + 1) * rs] }
        }
        let views: Vec<PairView> = (0..9)
            .map(|i| {
                let e = (i * 5) % 4;
                PairView {
                    expert: ExpertView { up: view(&made[e].expert.up, &ud, &us, e), down: view(&made[e].expert.down, &dd, &ds, e) },
                    token: (i * 2) % 3,
                    weight: 0.5 + 0.25 * i as f32,
                }
            })
            .collect();
        let owned: Vec<Pair> = views.iter().enumerate().map(|(i, v)| Pair { expert: &made[(i * 5) % 4].expert, token: v.token, weight: v.weight }).collect();
        let mut want = vec![0f32; owned.len() * hidden];
        let n_want = experts(Z::Bf16(&z), &owned, &mut want, 1).unwrap();
        assert!(n_want.iter().any(|&n| n > 0) && want.iter().any(|&x| x != 0.0));
        for (threads, pin) in [(1, false), (3, false), (3, true), (0, true)] {
            let mut got = vec![1f32; want.len()];
            assert_eq!(experts_views(Kernel::best(), Z::Bf16(&z), &views, &mut got, threads, pin).unwrap(), n_want);
            assert!(got.iter().zip(&want).all(|(a, b)| a.to_bits() == b.to_bits()), "{threads} threads, pinned {pin}");
        }
        // a view of a matrix that is not a layout's is refused
        let mut bad = views.clone();
        bad[0].expert.down.data = &dd[..rd - 1];
        assert!(experts_views(Kernel::best(), Z::Bf16(&z), &bad, &mut vec![0f32; bad.len() * hidden], 1, false).is_err());
        assert!(default_threads() >= 1);
    }

    // ---- the workers

    /// Every item is called once, for any count of threads and items, and the callers do not need `f` to outlive anything.
    #[test]
    fn every_item_once() {
        for (threads, items) in [(1, 0), (1, 5), (2, 1), (3, 100), (7, 1000), (0, 257), (16, 3)] {
            let seen: Vec<AtomicUsize> = (0..items).map(|_| AtomicUsize::new(0)).collect();
            run(threads, false, items, &|i| {
                seen[i].fetch_add(1, Ordering::Relaxed);
            });
            assert!(seen.iter().all(|s| s.load(Ordering::Relaxed) == 1), "{threads} threads, {items} items");
        }
    }

    /// An item that panics is reported to the caller, once the others are done, and the workers are there for the next job.
    #[test]
    fn a_panicking_item_is_reported() {
        let done = AtomicUsize::new(0);
        let r = catch_unwind(AssertUnwindSafe(|| {
            run(4, false, 50, &|i| {
                if i == 17 {
                    panic!("item 17");
                }
                done.fetch_add(1, Ordering::Relaxed);
            })
        }));
        assert!(r.is_err() && done.load(Ordering::Relaxed) == 49);
        let again = AtomicUsize::new(0);
        run(4, false, 50, &|_| {
            again.fetch_add(1, Ordering::Relaxed);
        });
        assert_eq!(again.load(Ordering::Relaxed), 50);
    }
}
