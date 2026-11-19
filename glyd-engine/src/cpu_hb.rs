//! The CPU half of the Hummingbird expert offload (Linear GLY-98, GLY-134): a routed expert of Nemotron 3 Super held as codebook rows, computed in place from system RAM for a pair (token, expert) whose expert is not on
//! the GPU. It is src/cpu_experts.rs redone for the codebook rows: a matrix [rows, cols] of neuron rows (cols a multiple of 32), each row a table of 4 (two bits) or 8 (three bits) bf16 entries, a little-endian bitstream of
//! codes that index the table, and one bf16 scale for each run of 32 values; a weight is `scale * table[code]`, both bf16, their product exact in fp32 and not rounded to bf16. The three arrays (`codes`, `scales`,
//! `codebook`; the byte layout is the Hummingbird layout's, 328 bytes a row at two bits and 464 at three, at 1,024 columns) are separate and are read where they are (`CbView` borrows them, so a host tier's pinned memory
//! needs no copy); `CbRows` owns them and packs them from one code a byte (`from_codes`).
//!
//! A pair is as in cpu_experts.rs: z' (the rotated input, `hidden` values a token, fp32 or bf16 bits) against each neuron's row of `up` in fp32, the sum rounded to bf16, relu squared of that rounded to bf16 once, the
//! neurons whose activation is not 0 listed in order; then, for those neurons, the activation times the run's scale times the row's table entry added into `hidden` fp32 sums, and the routing weight multiplied in last. The
//! result is in the rotated basis (the caller adds a token's pairs and rotates back), and `down` is held transposed, row n the `hidden` output weights of neuron n.
//!
//! The order of the float sums is fixed and the same in every kernel (`cpu_simd`): a row's first product keeps 16 lane sums, lane p taking the run's codes 32 k + p and 32 k + 16 + p (natural order, so z is used as it
//! is), each run adding `fma(wh z_hi, wl z_lo)` times the run's scale into them, and the 16 lanes are added in a tree (l with l + 8, then l + 4, l + 2, l + 1); an output of the second product is a chain of fused
//! multiply-adds of the table entry and `activation * scale` over the active neurons in order. Every kernel (`Kernel`: scalar, NEON, AVX2 with FMA, AVX-512) makes the same bits, and the work is cut as cpu_experts.rs
//! cuts it (items of 128 neurons of a pair's first product, the thread that ends a pair's last item doing its second product), so the bits do not depend on the number of threads either.

use crate::cpu_experts::{bf16_round, bf16_to_f32, run, Kernel, Z};
use crate::cpu_simd as simd;
use crate::ffi::{err, Result};
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};

/// Values in a run (one scale).
const RUN: usize = 32;

/// Neurons a work item takes the first product of.
const ROW_BLOCK: usize = 128;

/// The bytes a row of codes, of scales and of codebook take at `bits` (2 or 3) and `cols` values (a multiple of 32).
pub fn strides(bits: u32, cols: usize) -> (usize, usize, usize) {
    (cols / 8 * bits as usize, cols / RUN * 2, 2 << bits)
}

/// The bytes of codes, scales and codebook of `rows` rows, if the shape is a layout's.
fn sizes(bits: u32, rows: usize, cols: usize) -> Option<(usize, usize, usize)> {
    if !(2..=3).contains(&bits) || rows == 0 || cols == 0 || !cols.is_multiple_of(RUN) {
        return None;
    }
    let (c, s, b) = strides(bits, cols);
    Some((rows.checked_mul(c)?, rows.checked_mul(s)?, rows.checked_mul(b)?))
}

/// A matrix [rows, cols] of codebook rows, borrowed: `codes` (rows x `strides().0` bytes), `scales` (rows x cols / 32 little-endian bf16) and `codebook` (rows x 2^bits little-endian bf16).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CbView<'a> {
    pub bits: u32,
    pub rows: usize,
    pub cols: usize,
    pub codes: &'a [u8],
    pub scales: &'a [u8],
    pub codebook: &'a [u8],
}

impl CbView<'_> {
    /// Whether the fields are a layout's (the kernels read raw memory; `CbRows::new` makes sure of it, the fields here are public).
    pub fn is_valid(&self) -> bool {
        sizes(self.bits, self.rows, self.cols) == Some((self.codes.len(), self.scales.len(), self.codebook.len()))
    }

    /// Runs in a row.
    pub fn runs(&self) -> usize {
        self.cols / RUN
    }

    /// The code of weight (n, j), read bit by bit from the row's stream (stream bit i is bit i mod 8 of byte i div 8; a code's lowest bit is its lowest stream bit).
    pub fn code(&self, n: usize, j: usize) -> u8 {
        let (bits, cs) = (self.bits as usize, strides(self.bits, self.cols).0);
        let row = &self.codes[n * cs..(n + 1) * cs];
        (0..bits).map(|b| ((row[(bits * j + b) / 8] >> ((bits * j + b) % 8)) & 1) << b).sum()
    }

    /// The scale of run k of row n (bf16 bits).
    pub fn scale(&self, n: usize, k: usize) -> u16 {
        let at = 2 * (n * self.runs() + k);
        u16::from_le_bytes([self.scales[at], self.scales[at + 1]])
    }

    /// Entry c of row n's table (bf16 bits).
    pub fn entry(&self, n: usize, c: usize) -> u16 {
        let at = 2 * (n << self.bits | c);
        u16::from_le_bytes([self.codebook[at], self.codebook[at + 1]])
    }
}

/// A matrix [rows, cols] of codebook rows, owned: see [`CbView`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CbRows {
    pub bits: u32,
    pub rows: usize,
    pub cols: usize,
    pub codes: Vec<u8>,
    pub scales: Vec<u8>,
    pub codebook: Vec<u8>,
}

impl CbRows {
    /// The matrix whose arrays are these: the sizes must be a layout's.
    pub fn new(bits: u32, rows: usize, cols: usize, codes: Vec<u8>, scales: Vec<u8>, codebook: Vec<u8>) -> Result<CbRows> {
        let Some(want) = sizes(bits, rows, cols) else {
            return err(format!("a matrix of {rows} x {cols} at level {bits}: not a shape or level this engine reads"));
        };
        if (codes.len(), scales.len(), codebook.len()) != want {
            return err(format!("{rows} rows of {cols} at {bits} bits take {} + {} + {} bytes: {} + {} + {} are given", want.0, want.1, want.2, codes.len(), scales.len(), codebook.len()));
        }
        Ok(CbRows { bits, rows, cols, codes, scales, codebook })
    }

    /// The matrix whose weight (r, c) has the code `codes[r * cols + c]` (0 to 2^bits - 1), the scale `scales[r * cols / 32 + c / 32]` and the table `codebook[r * 2^bits..][..2^bits]` (bf16 bits), packed as the layout says.
    pub fn from_codes(bits: u32, rows: usize, cols: usize, codes: &[u8], scales: &[u16], codebook: &[u16]) -> Result<CbRows> {
        let Some((cn, sn, bn)) = sizes(bits, rows, cols) else {
            return err(format!("a matrix of {rows} x {cols} at level {bits}: not a shape or level this engine reads"));
        };
        if rows.checked_mul(cols) != Some(codes.len()) || scales.len() != sn / 2 || codebook.len() != bn / 2 || codes.iter().any(|&c| (c as u32) >> bits != 0) {
            return err(format!("{} codes, {} scales and {} table entries for {rows} x {cols} at {bits} bits", codes.len(), scales.len(), codebook.len()));
        }
        let (cs, bits) = (strides(bits, cols).0, bits as usize);
        let mut packed = vec![0u8; cn];
        for (r, row) in packed.chunks_mut(cs).enumerate() {
            for (j, &c) in codes[r * cols..(r + 1) * cols].iter().enumerate() {
                for b in 0..bits {
                    row[(bits * j + b) / 8] |= ((c >> b) & 1) << ((bits * j + b) % 8);
                }
            }
        }
        CbRows::new(bits as u32, rows, cols, packed, scales.iter().flat_map(|s| s.to_le_bytes()).collect(), codebook.iter().flat_map(|e| e.to_le_bytes()).collect())
    }

    pub fn view(&self) -> CbView<'_> {
        CbView { bits: self.bits, rows: self.rows, cols: self.cols, codes: &self.codes, scales: &self.scales, codebook: &self.codebook }
    }
}

/// A routed expert: its first matrix `up` and its second one transposed, `down` (row n holds the weights that neuron n's activation multiplies), both [neurons, hidden], at the same bits.
#[derive(Clone, Copy, Debug)]
pub struct HbExpert<'a> {
    pub up: CbView<'a>,
    pub down: CbView<'a>,
}

impl<'a> HbExpert<'a> {
    pub fn new(up: CbView<'a>, down: CbView<'a>) -> Result<HbExpert<'a>> {
        if !up.is_valid() || !down.is_valid() || (up.bits, up.rows, up.cols) != (down.bits, down.rows, down.cols) {
            return err(format!("an expert's up is {} x {} at {} bits and its transposed down {} x {} at {} bits: valid matrices of the same shape are wanted", up.rows, up.cols, up.bits, down.rows, down.cols, down.bits));
        }
        Ok(HbExpert { up, down })
    }

    /// Neurons (the intermediate size) and the width of the input and the output.
    pub fn inter(&self) -> usize {
        self.up.rows
    }

    pub fn hidden(&self) -> usize {
        self.up.cols
    }
}

/// A token's use of an expert: token `token` (a row of z) through `expert`, its result times `weight` (the routing weight).
#[derive(Clone, Copy, Debug)]
pub struct HbPair<'a> {
    pub expert: &'a HbExpert<'a>,
    pub token: usize,
    pub weight: f32,
}

/// The pairs' results with the best kernel: `out` [pairs, hidden] gets pair p's row (the routing weight times the sum of the active neurons' down rows, each times its activation; in the rotated basis), `z` [tokens, hidden]
/// the rotated inputs, and the threads (0: as many as the computer has) share the work. Returns the number of active neurons of each pair. See [`experts_with`].
pub fn experts(z: Z, pairs: &[HbPair], out: &mut [f32], threads: usize) -> Result<Vec<u32>> {
    experts_with(Kernel::best(), z, pairs, out, threads)
}

/// [`experts`] with the kernel asked for (an error where this computer cannot run it).
pub fn experts_with(kernel: Kernel, z: Z, pairs: &[HbPair], out: &mut [f32], threads: usize) -> Result<Vec<u32>> {
    experts_pinned(kernel, z, pairs, out, threads, false)
}

/// [`experts_with`]; `pin`: the threads of the crew each stay on a core of their own (see `Workers`).
pub fn experts_pinned(kernel: Kernel, z: Z, pairs: &[HbPair], out: &mut [f32], threads: usize, pin: bool) -> Result<Vec<u32>> {
    if !kernel.runs_here() {
        return err(format!("the {} kernel cannot run on this computer", kernel.name()));
    }
    let Some(first) = pairs.first() else { return Ok(Vec::new()) };
    let hidden = first.expert.hidden();
    let zlen = match z {
        Z::F32(v) => v.len(),
        Z::Bf16(v) => v.len(),
    };
    if hidden == 0 || !zlen.is_multiple_of(hidden) || out.len() != pairs.len() * hidden {
        return err(format!("z holds {zlen} values and out {}, where rows of {hidden} and {} pairs are wanted", out.len(), pairs.len()));
    }
    let tokens = zlen / hidden;
    // z's rows that the pairs use, as fp32
    let (mut at, mut used) = (vec![usize::MAX; tokens], Vec::new());
    for (p, pair) in pairs.iter().enumerate() {
        let e = pair.expert;
        if !e.up.is_valid() || !e.down.is_valid() || (e.up.bits, e.up.rows, e.up.cols) != (e.down.bits, e.down.rows, e.down.cols) || e.hidden() != hidden || pair.token >= tokens {
            return err(format!("pair {p}: an expert of width {} and token {}, where {hidden} and 0 to {} are wanted", e.hidden(), pair.token, tokens.saturating_sub(1)));
        }
        if at[pair.token] == usize::MAX {
            at[pair.token] = used.len();
            used.push(pair.token);
        }
    }
    let mut zp = vec![0f32; used.len() * hidden];
    for (row, &t) in zp.chunks_mut(hidden).zip(&used) {
        match z {
            Z::F32(v) => row.copy_from_slice(&v[t * hidden..(t + 1) * hidden]),
            Z::Bf16(v) => row.iter_mut().zip(&v[t * hidden..(t + 1) * hidden]).for_each(|(d, &b)| *d = bf16_to_f32(b)),
        }
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
        simd::down_cb(kernel, &pair.expert.down, &neurons, &mut acc);
        let mut row = rows[p].lock().expect("a row of the output");
        for (o, &a) in row.iter_mut().zip(&acc) {
            *o = a * pair.weight;
        }
    };
    let work = |item: usize| {
        let p = first_item.partition_point(|&s| s <= item) - 1;
        let (pair, b) = (&pairs[p], item - first_item[p]);
        let (r0, r1) = (b * ROW_BLOCK, ((b + 1) * ROW_BLOCK).min(pair.expert.inter()));
        let mut sums = [0f32; ROW_BLOCK];
        let t = at[pair.token];
        simd::up_cb(kernel, &pair.expert.up, &zp[t * hidden..(t + 1) * hidden], r0, r1, &mut sums[..r1 - r0]);
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

/// Whether `kernel` makes the scalar kernel's bits on small experts of each width (random codes, scales and tables, z of both signs, 2 tokens and 3 pairs): a kernel that does not has a fault of its own, and its sums are not
/// to be used (cpu_experts.rs's `agrees_with_scalar` for the codebook rows).
pub fn agrees_with_scalar(kernel: Kernel) -> bool {
    let mut x = 0x9E37_79B9_7F4A_7C15u64;
    let mut next = move || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        x
    };
    let (inter, hidden) = (192usize, 96usize);
    let bf = |v: f32| (bf16_round(v).to_bits() >> 16) as u16;
    for bits in 2..=3u32 {
        let matrix = |next: &mut dyn FnMut() -> u64| {
            let codes: Vec<u8> = (0..inter * hidden).map(|_| (next() % (1 << bits)) as u8).collect();
            let scales: Vec<u16> = (0..inter * hidden / RUN).map(|_| bf((next() % 2000) as f32 / 1000.0 - 0.8)).collect();
            let table: Vec<u16> = (0..inter << bits).map(|_| bf((next() % 2000) as f32 / 1000.0 - 1.0)).collect();
            CbRows::from_codes(bits, inter, hidden, &codes, &scales, &table)
        };
        let made: Result<Vec<(CbRows, CbRows)>> = (0..2).map(|_| Ok((matrix(&mut next)?, matrix(&mut next)?))).collect();
        let Ok(made) = made else { return false };
        let experts: Result<Vec<HbExpert>> = made.iter().map(|(u, d)| HbExpert::new(u.view(), d.view())).collect();
        let Ok(experts) = experts else { return false };
        let z: Vec<f32> = (0..2 * hidden).map(|_| bf16_round((next() % 1000) as f32 / 250.0 - 2.0)).collect();
        let pairs: Vec<HbPair> = (0..3).map(|i| HbPair { expert: &experts[i % 2], token: i % 2, weight: 0.5 + i as f32 }).collect();
        let (mut want, mut got) = (vec![0f32; 3 * hidden], vec![1f32; 3 * hidden]);
        let (a, b) = (experts_with(Kernel::Scalar, Z::F32(&z), &pairs, &mut want, 1), experts_with(kernel, Z::F32(&z), &pairs, &mut got, 1));
        if !matches!((&a, &b), (Ok(x), Ok(y)) if x == y) || want.iter().zip(&got).any(|(p, q)| p.to_bits() != q.to_bits()) {
            return false;
        }
    }
    true
}

/// The fastest kernel this computer runs that `agrees_with_scalar` here (the scalar one at the least), found once; a computer with another is told so by a line on stderr and given the next.
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

#[cfg(test)]
mod tests {
    use super::*;

    // ---- the layout

    /// The bytes of a run written out by hand from the layout: two bits, byte b holds codes 4 b to 4 b + 3, code 4 b + m at bits 2 m and 2 m + 1.
    #[test]
    fn the_bytes_of_a_run_at_two_bits() {
        // codes 0 to 7: 1 2 3 0 | 3 3 0 1; 8 to 11: 2 0 1 3; 12 to 27: 0; 28 to 31: 3
        //   byte 0 = 1 + 2 4 + 3 16 + 0 64 = 0x39, byte 1 = 3 + 3 4 + 0 16 + 1 64 = 0x4F, byte 2 = 2 + 0 4 + 1 16 + 3 64 = 0xD2, bytes 3 to 6 = 0, byte 7 = 0xFF
        let mut codes = [0u8; 32];
        codes[..12].copy_from_slice(&[1, 2, 3, 0, 3, 3, 0, 1, 2, 0, 1, 3]);
        codes[28..].fill(3);
        let m = CbRows::from_codes(2, 1, 32, &codes, &[0x3F80], &[0x3F80, 0x4000, 0x4040, 0x4080]).unwrap();
        assert_eq!(m.codes, [0x39, 0x4F, 0xD2, 0, 0, 0, 0, 0xFF]);
        assert_eq!((m.scales.as_slice(), m.codebook.as_slice()), ([0x80, 0x3F].as_slice(), [0x80, 0x3F, 0x00, 0x40, 0x40, 0x40, 0x80, 0x40].as_slice()));
        // two rows: the second starts at byte 8 of the codes, byte 2 of the scales, byte 8 of the codebook
        let two = CbRows::from_codes(2, 2, 32, &[codes.as_slice(), &[3; 32]].concat(), &[1, 2], &[1, 2, 3, 4, 5, 6, 7, 8]).unwrap();
        assert_eq!(&two.codes[..8], m.codes.as_slice());
        assert_eq!(&two.codes[8..], [0xFF; 8]);
        assert_eq!((two.view().scale(1, 0), two.view().entry(1, 2), two.view().entry(0, 3)), (2, 7, 4));
    }

    /// Three bits: codes 8 g to 8 g + 7 are the 24-bit little-endian integer of bytes 3 g to 3 g + 2, code 8 g + m at its bits 3 m to 3 m + 2, so the codes 2 and 5 of a group straddle a byte.
    #[test]
    fn the_bytes_of_a_run_at_three_bits() {
        // group 0: 5 1 6 2 7 0 3 4 = 5 + 1 8 + 6 64 + 2 512 + 7 4096 + 0 + 3 2^18 + 4 2^21 = 0x8C758D (code 2 = bits 6, 7 of byte 0 and bit 0 of byte 1: 0b10 + 0b100 = 6)
        // group 1: all 7 = 0xFFFFFF; group 2: code 0 = 1 = 0x000001; group 3: code 7 = 7 = 0xE00000
        let mut codes = [0u8; 32];
        codes[..8].copy_from_slice(&[5, 1, 6, 2, 7, 0, 3, 4]);
        codes[8..16].fill(7);
        codes[16] = 1;
        codes[31] = 7;
        let table: Vec<u16> = (0..8).map(|c| 0x3F80 + c).collect();
        let m = CbRows::from_codes(3, 1, 32, &codes, &[0x4000], &table).unwrap();
        assert_eq!(m.codes, [0x8D, 0x75, 0x8C, 0xFF, 0xFF, 0xFF, 0x01, 0x00, 0x00, 0x00, 0x00, 0xE0]);
        assert_eq!(m.codebook.len(), 16);
        assert_eq!(&m.codebook[..4], [0x80, 0x3F, 0x81, 0x3F]);
        let v = m.view();
        assert!((0..32).all(|j| v.code(0, j) == codes[j]));
        // the straddling codes read from the bytes by hand: code 2 (byte 0 bits 6, 7 and byte 1 bit 0), code 5 (byte 1 bit 7 and byte 2 bits 0, 1)
        assert_eq!((m.codes[0] >> 6) | (m.codes[1] & 1) << 2, 6);
        assert_eq!((m.codes[1] >> 7) | (m.codes[2] & 3) << 1, 0);
    }

    /// The packer and the reader against each other and against a stream written bit by bit, for random matrices of every shape: the reader finds the codes, the scales and the entries again.
    #[test]
    fn code_reader_and_packer() {
        let mut rng = Rng(3);
        for bits in 2..=3u32 {
            for (rows, cols) in [(1, 32), (5, 64), (7, 128), (3, 1024)] {
                let codes: Vec<u8> = (0..rows * cols).map(|_| (rng.next() % (1 << bits)) as u8).collect();
                let scales: Vec<u16> = (0..rows * cols / 32).map(|_| rng.next() as u16).collect();
                let table: Vec<u16> = (0..rows << bits).map(|_| rng.next() as u16).collect();
                let m = CbRows::from_codes(bits, rows, cols, &codes, &scales, &table).unwrap();
                let v = m.view();
                assert!(v.is_valid());
                let cs = cols * bits as usize / 8;
                for r in 0..rows {
                    // (a stream of the row's codes, a bit at a time: the code j's bit b is stream bit bits j + b)
                    let stream: Vec<bool> = (0..cols).flat_map(|j| (0..bits).map(move |b| (j, b))).map(|(j, b)| codes[r * cols + j] >> b & 1 == 1).collect();
                    for (i, &bit) in stream.iter().enumerate() {
                        assert_eq!(m.codes[r * cs + i / 8] >> (i % 8) & 1 == 1, bit, "bits {bits} row {r} stream bit {i}");
                    }
                    for j in 0..cols {
                        assert_eq!(v.code(r, j), codes[r * cols + j]);
                    }
                    (0..cols / 32).for_each(|k| assert_eq!(v.scale(r, k), scales[r * cols / 32 + k]));
                    (0..1 << bits).for_each(|c| assert_eq!(v.entry(r, c), table[r << bits | c]));
                }
            }
        }
    }

    #[test]
    fn strides_and_sizes() {
        assert_eq!((strides(2, 1024), strides(3, 1024)), ((256, 64, 8), (384, 64, 16)));
        assert_eq!((strides(2, 128), strides(3, 128)), ((32, 8, 8), (48, 8, 16)));
        let row = |bits| [strides(bits, 1024).0, strides(bits, 1024).1, strides(bits, 1024).2].iter().sum::<usize>();
        assert_eq!((row(2), row(3)), (328, 464));
        // an expert of 2 x 2,688 rows
        assert_eq!((2 * 2688 * row(3), 2 * 2688 * row(2)), (2_494_464, 1_763_328));
        assert_eq!(sizes(3, 2688, 1024), Some((2688 * 384, 2688 * 64, 2688 * 16)));
        assert_eq!(sizes(2, usize::MAX, 1024), None, "no overflow");
        assert!([sizes(1, 1, 32), sizes(4, 1, 32), sizes(2, 0, 32), sizes(2, 1, 0), sizes(2, 1, 48)].iter().all(Option::is_none));
    }

    // ---- the products

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

    /// A matrix of random codes, tables and scales. `on`: a sign for each row, where it is given: a row of `true` has tables of non-negative entries, one of `false` non-positive, and every scale is positive, so against a
    /// positive z the row's sum has that sign. Without it the entries and the scales have both signs. `zeros`: a tenth of the scales are 0 or -0.
    fn matrix(rng: &mut Rng, bits: u32, rows: usize, cols: usize, on: Option<&[bool]>, zeros: bool) -> CbRows {
        let codes: Vec<u8> = (0..rows * cols).map(|_| (rng.next() % (1 << bits)) as u8).collect();
        let mut scales: Vec<u16> = (0..rows * cols / 32).map(|_| bits_of(if on.is_some() { 0.05 + 0.3 * rng.f() } else { 2.0 * rng.bell() })).collect();
        if zeros {
            for s in scales.iter_mut() {
                if rng.f() < 0.1 {
                    *s = if rng.f() < 0.5 { 0 } else { 0x8000 };
                }
            }
        }
        let table: Vec<u16> = (0..rows << bits)
            .map(|i| {
                let g = rng.bell();
                bits_of(match on {
                    None => g,
                    Some(on) => g.abs() * if on[i >> bits] { 1.0 } else { -1.0 },
                })
            })
            .collect();
        CbRows::from_codes(bits, rows, cols, &codes, &scales, &table).unwrap()
    }

    /// An expert (up and the transposed down) of random weights.
    struct Made {
        up: CbRows,
        down: CbRows,
    }

    impl Made {
        fn expert(&self) -> HbExpert<'_> {
            HbExpert::new(self.up.view(), self.down.view()).unwrap()
        }
    }

    fn make(rng: &mut Rng, bits: u32, inter: usize, hidden: usize, on: Option<&[bool]>, zeros: bool) -> Made {
        Made { up: matrix(rng, bits, inter, hidden, on, zeros), down: matrix(rng, bits, inter, hidden, None, zeros) }
    }

    fn mask(rng: &mut Rng, n: usize, active: f32) -> Vec<bool> {
        (0..n).map(|_| rng.f() < active).collect()
    }

    /// An expert whose up rows are on (against a positive z) with the chance `active`.
    fn make_sparse(rng: &mut Rng, bits: u32, inter: usize, hidden: usize, active: f32) -> Made {
        let rows = mask(rng, inter, active);
        make(rng, bits, inter, hidden, Some(&rows), false)
    }

    /// z for `tokens` tokens: bf16 values, positive or of both signs.
    fn input(rng: &mut Rng, tokens: usize, hidden: usize, positive: bool) -> Vec<f32> {
        (0..tokens * hidden).map(|_| bf16_round(if positive { 0.25 + 1.5 * rng.f() } else { 4.0 * (rng.f() - 0.5) })).collect()
    }

    /// The weights of a matrix, exactly: the table's entry times the run's scale, as f64.
    fn exact(m: &CbView) -> Vec<f64> {
        (0..m.rows * m.cols).map(|i| (bf16_to_f32(m.entry(i / m.cols, m.code(i / m.cols, i % m.cols) as usize)) * bf16_to_f32(m.scale(i / m.cols, i % m.cols / RUN))) as f64).collect()
    }

    /// The pair by the arithmetic of the spec, a value at a time in f32 (written out apart from the kernels): the active neurons with their activations, and the output row.
    fn spec_pair(e: &HbExpert, z: &[f32], weight: f32) -> (Vec<(u32, f32)>, Vec<f32>) {
        let (inter, hidden) = (e.inter(), e.hidden());
        let tbl = |m: &CbView, n: usize, j: usize| bf16_to_f32(m.entry(n, m.code(n, j) as usize));
        let mut list = Vec::new();
        for n in 0..inter {
            let mut l = [0f32; 16];
            for k in 0..hidden / RUN {
                let s = bf16_to_f32(e.up.scale(n, k));
                for (p, acc) in l.iter_mut().enumerate() {
                    let (wl, wh) = (tbl(&e.up, n, RUN * k + p), tbl(&e.up, n, RUN * k + 16 + p));
                    let t = wh.mul_add(z[RUN * k + 16 + p], wl * z[RUN * k + p]);
                    *acc = t.mul_add(s, *acc);
                }
            }
            for half in [8, 4, 2, 1] {
                for i in 0..half {
                    l[i] += l[i + half];
                }
            }
            let h = bf16_round(l[0]);
            if h > 0.0 {
                let a = bf16_round(h * h);
                if a != 0.0 {
                    list.push((n as u32, a));
                }
            }
        }
        let mut acc = vec![0f32; hidden];
        for &(n, a) in &list {
            for k in 0..hidden / RUN {
                let c = a * bf16_to_f32(e.down.scale(n as usize, k));
                for i in 0..RUN {
                    acc[RUN * k + i] = tbl(&e.down, n as usize, RUN * k + i).mul_add(c, acc[RUN * k + i]);
                }
            }
        }
        (list, acc.iter().map(|&x| x * weight).collect())
    }

    /// The activations by the book: the first sums in f64 from the exact weights, rounded to bf16, relu squared rounded to bf16 once.
    fn activations_book(up: &[f64], hidden: usize, z: &[f32]) -> Vec<f32> {
        up.chunks(hidden)
            .map(|row| {
                let h = bf16_round(row.iter().zip(z).map(|(&w, &x)| w * x as f64).sum::<f64>() as f32);
                if h > 0.0 {
                    bf16_round(h * h)
                } else {
                    0.0
                }
            })
            .collect()
    }

    /// The second product in f64 from the exact weights, for activations given: the routing weight times the sum over the neurons.
    fn second_f64(down: &[f64], hidden: usize, acts: &[f32], w: f32) -> Vec<f64> {
        (0..hidden).map(|o| w as f64 * acts.iter().enumerate().map(|(n, &a)| a as f64 * down[n * hidden + o]).sum::<f64>()).collect()
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

    fn same_bits(a: &[f32], b: &[f32]) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
    }

    /// The scalar kernel is the spec's arithmetic, and every kernel makes the bits the scalar one does: experts of 2 and 3 bits and of other shapes (cols 128, 256 and 1024, neuron counts that are not a block's), z as fp32
    /// values that are not bf16 values, pairs of several tokens and experts, about 8% and 50% of the neurons active, none, all, and weights of both signs with zero scales.
    #[test]
    fn kernels_make_the_same_bits() {
        let mut rng = Rng(33);
        for (inter, hidden) in [(300, 128), (333, 256), (150, 1024)] {
            let mut made = Vec::new();
            for bits in 2..=3u32 {
                for active in [0.08, 0.5, 0.0, 1.0] {
                    made.push(make_sparse(&mut rng, bits, inter, hidden, active));
                }
                made.push(make(&mut rng, bits, inter, hidden, None, true));
            }
            let ex: Vec<HbExpert> = made.iter().map(Made::expert).collect();
            // (tokens 0 and 1 are positive, token 2 of both signs)
            let mut z: Vec<f32> = (0..2 * hidden).map(|_| 0.25 + 1.5 * rng.f()).collect();
            z.extend((0..hidden).map(|_| 2.0 * (rng.f() - 0.5)));
            assert!(z.iter().any(|&x| bf16_round(x) != x));
            let pairs: Vec<HbPair> = (0..13).map(|i| HbPair { expert: &ex[(i * 3) % 10], token: i % 3, weight: 0.5 + rng.f() }).collect();
            let mut want = vec![0f32; pairs.len() * hidden];
            let n_want = experts_with(Kernel::Scalar, Z::F32(&z), &pairs, &mut want, 1).unwrap();
            assert!(n_want.iter().any(|&n| n > 0) && want.iter().any(|&x| x != 0.0));
            for (i, p) in pairs.iter().enumerate() {
                let (list, row) = spec_pair(p.expert, &z[p.token * hidden..(p.token + 1) * hidden], p.weight);
                assert_eq!((n_want[i], same_bits(&want[i * hidden..(i + 1) * hidden], &row)), (list.len() as u32, true), "{inter} x {hidden}: pair {i} against the spec");
                let e = (i * 3) % 10 % 5;
                if p.token < 2 && e == 2 {
                    assert!(n_want[i] == 0 && row.iter().all(|&x| x == 0.0), "no neuron on");
                }
                if p.token < 2 && e == 3 {
                    assert_eq!(n_want[i] as usize, inter, "every neuron on");
                }
            }
            for k in Kernel::available() {
                let mut got = vec![0f32; want.len()];
                let n = experts_with(k, Z::F32(&z), &pairs, &mut got, 1).unwrap();
                assert_eq!(n, n_want, "{inter} x {hidden}: {}", k.name());
                assert!(same_bits(&got, &want), "{inter} x {hidden}: the {} kernel differs from the scalar one", k.name());
            }
        }
    }

    /// The x86 kernels' code lookups as plain integer arithmetic (this computer cannot run those kernels): a code is found in a run's windows by its lane's shift, then the low 3 bits (AVX2, a table of 8 lanes) or 4 bits
    /// (AVX-512, 16 lanes) of the shifted window index the row's table repeated to fill the register, and the entry is the one the layout's code names.
    #[test]
    fn the_x86_lookups_by_the_book() {
        let mut rng = Rng(9);
        for bits in 2..=3u32 {
            for _ in 0..300 {
                let codes: Vec<u8> = (0..32).map(|_| (rng.next() % (1 << bits)) as u8).collect();
                let table: Vec<u16> = (0..1 << bits).map(|_| rng.next() as u16).collect();
                let m = CbRows::from_codes(bits, 1, 32, &codes, &[0], &table).unwrap();
                // SAFETY: the run is the row's 4 * bits bytes, which are all that is read.
                let w = unsafe {
                    if bits == 2 {
                        simd::x86_windows::<2>(m.codes.as_ptr())
                    } else {
                        simd::x86_windows::<3>(m.codes.as_ptr())
                    }
                };
                for (j, &c) in codes.iter().enumerate() {
                    let (window, shift) = if bits == 2 { (w[j / 16], 2 * (j % 16)) } else { (w[j / 8], 3 * (j % 8)) };
                    let idx = window as u32 >> shift;
                    let (avx2, avx512) = (table[(idx & 7) as usize % table.len()], table[(idx & 15) as usize % table.len()]);
                    assert_eq!((avx2, avx512), (table[c as usize], table[c as usize]), "bits {bits} code {j}");
                }
            }
        }
    }

    /// The arrays of any alignment: the same expert held in slices that start at an odd address gives the same bits.
    #[test]
    fn arrays_of_any_alignment() {
        let mut rng = Rng(34);
        let (inter, hidden) = (136, 128);
        for bits in 2..=3u32 {
            let m = make_sparse(&mut rng, bits, inter, hidden, 0.3);
            let off = |v: &[u8]| [&[0xAAu8][..], v].concat();
            let (uc, us, ub, dc, ds, db) = (off(&m.up.codes), off(&m.up.scales), off(&m.up.codebook), off(&m.down.codes), off(&m.down.scales), off(&m.down.codebook));
            fn view<'a>(m: &CbRows, c: &'a [u8], s: &'a [u8], b: &'a [u8]) -> CbView<'a> {
                CbView { bits: m.bits, rows: m.rows, cols: m.cols, codes: &c[1..], scales: &s[1..], codebook: &b[1..] }
            }
            let moved = HbExpert::new(view(&m.up, &uc, &us, &ub), view(&m.down, &dc, &ds, &db)).unwrap();
            let z = input(&mut rng, 1, hidden, true);
            let (a, b) = (m.expert(), moved);
            let mut want = vec![0f32; hidden];
            experts_with(Kernel::Scalar, Z::F32(&z), &[HbPair { expert: &a, token: 0, weight: 1.5 }], &mut want, 1).unwrap();
            assert!(want.iter().any(|&x| x != 0.0));
            for k in Kernel::available() {
                let mut got = vec![0f32; hidden];
                experts_with(k, Z::F32(&z), &[HbPair { expert: &b, token: 0, weight: 1.5 }], &mut got, 1).unwrap();
                assert!(same_bits(&got, &want), "bits {bits}: {}", k.name());
            }
        }
    }

    /// Every kernel against the pair computed in f64 from the exact weights (the first sums rounded to bf16 as the kernels round them), at 2 and 3 bits: about 8% of the neurons on, half, none, all, and weights and z of
    /// both signs with zero scales. What the kernels differ from the book by is the float sums' order alone (an activation that the book rounds the other way is one bf16 step off); against the book's second product given the
    /// kernel's own activations the error is the second sum's.
    #[test]
    fn pairs_against_float64() {
        let mut rng = Rng(21);
        let (inter, hidden) = (192, 256);
        let mut worst = (0f64, 0f64);
        for bits in 2..=3u32 {
            for (name, active) in [("sparse", Some(0.08)), ("half", Some(0.5)), ("none", Some(0.0)), ("all", Some(1.0)), ("mixed", None)] {
                let rows = active.map(|a| mask(&mut rng, inter, a));
                let m = make(&mut rng, bits, inter, hidden, rows.as_deref(), name == "mixed");
                let e = m.expert();
                let z = input(&mut rng, 1, hidden, active.is_some());
                let (up, down) = (exact(&e.up), exact(&e.down));
                let book = activations_book(&up, hidden, &z);
                let n = book.iter().filter(|&&a| a != 0.0).count() as u32;
                let want = second_f64(&down, hidden, &book, 2.5);
                let (list, _) = spec_pair(&e, &z, 2.5);
                let mut dense = vec![0f32; inter];
                list.iter().for_each(|&(i, a)| dense[i as usize] = a);
                let own = second_f64(&down, hidden, &dense, 2.5);
                for k in Kernel::available() {
                    let mut out = vec![f32::NAN; hidden];
                    let got = experts_with(k, Z::F32(&z), &[HbPair { expert: &e, token: 0, weight: 2.5 }], &mut out, 1).unwrap();
                    let (e_book, e_own) = (rel(&out, &want), rel(&out, &own));
                    worst = (worst.0.max(e_book), worst.1.max(e_own));
                    eprintln!("bits {bits} {name:6} {:7}: {} active (by the book {n}), error {e_book:.2e} against the book, {e_own:.2e} against the same activations", k.name(), got[0]);
                    assert!(got[0].abs_diff(n) <= 1, "bits {bits} {name} {}: {} and {n}", k.name(), got[0]);
                    // (the second sum is a chain of up to 192 fused multiply-adds of terms of both signs: its f32 rounding is about 3e-8 sqrt(active neurons) of the result's norm)
                    assert!(e_book < 1e-4 && e_own < 5e-7, "bits {bits} {name} {}: {e_book} against the book, {e_own} against the same activations", k.name());
                    if name == "none" {
                        assert!(got[0] == 0 && out.iter().all(|&x| x == 0.0), "no active neuron, no output");
                    }
                }
            }
        }
        eprintln!("worst error: {:.2e} against the book, {:.2e} against the same activations", worst.0, worst.1);
    }

    /// The threads change no bit: the work cut in other items, taken in other turns, is the same sums.
    #[test]
    fn threads_change_no_bit() {
        let mut rng = Rng(44);
        let (inter, hidden) = (576, 128);
        let made: Vec<Made> = (0..5).map(|e| make_sparse(&mut rng, 2 + e as u32 % 2, inter, hidden, 0.04 * (1 + e) as f32)).collect();
        let ex: Vec<HbExpert> = made.iter().map(Made::expert).collect();
        let z: Vec<f32> = (0..4 * hidden).map(|_| bf16_round(0.25 + rng.f())).collect();
        let pairs: Vec<HbPair> = (0..11).map(|i| HbPair { expert: &ex[(i * 3) % 5], token: i % 4, weight: 1.0 + rng.f() }).collect();
        let mut want = vec![0f32; pairs.len() * hidden];
        let n_want = experts(Z::F32(&z), &pairs, &mut want, 1).unwrap();
        assert!(n_want.iter().any(|&n| n > 0));
        for threads in [2, 3, 7, 10, 0, 64] {
            let mut got = vec![1f32; want.len()];
            assert_eq!(experts(Z::F32(&z), &pairs, &mut got, threads).unwrap(), n_want);
            assert!(same_bits(&got, &want), "{threads} threads");
        }
        // one pair, fewer items than threads; and none
        let mut one = vec![0f32; hidden];
        experts(Z::F32(&z), &pairs[..1], &mut one, 8).unwrap();
        assert!(same_bits(&one, &want[..hidden]));
        assert!(experts(Z::F32(&z), &[], &mut [], 4).unwrap().is_empty());
    }

    /// z given as bf16 bits is z given as the fp32 values of them.
    #[test]
    fn z_in_bf16() {
        let mut rng = Rng(55);
        let (inter, hidden) = (128, 64);
        let made = [make(&mut rng, 3, inter, hidden, None, false), make(&mut rng, 2, inter, hidden, None, false)];
        let ex: Vec<HbExpert> = made.iter().map(Made::expert).collect();
        let zb: Vec<u16> = (0..2 * hidden).map(|_| bits_of(4.0 * (rng.f() - 0.5))).collect();
        let zf: Vec<f32> = zb.iter().map(|&b| bf16_to_f32(b)).collect();
        let pairs = [HbPair { expert: &ex[0], token: 1, weight: 1.5 }, HbPair { expert: &ex[1], token: 0, weight: 0.5 }, HbPair { expert: &ex[0], token: 0, weight: 2.0 }];
        let (mut a, mut b) = (vec![0f32; 3 * hidden], vec![0f32; 3 * hidden]);
        assert_eq!(experts(Z::Bf16(&zb), &pairs, &mut a, 2).unwrap(), experts(Z::F32(&zf), &pairs, &mut b, 2).unwrap());
        assert!(same_bits(&a, &b) && a.iter().any(|&x| x != 0.0));
    }

    /// An expert of Nemotron 3 Super's shape (2,688 neurons of 1,024) at 3 and 2 bits, about 92% of its neurons off, all kernels and many threads: the bits of the spec's arithmetic, and the error against f64.
    #[test]
    fn an_expert_of_the_models_shape() {
        let mut rng = Rng(88);
        let (inter, hidden) = (2688, 1024);
        for bits in [3, 2] {
            let m = make_sparse(&mut rng, bits, inter, hidden, 0.082);
            let e = m.expert();
            let z = input(&mut rng, 1, hidden, true);
            let (list, spec) = spec_pair(&e, &z, 5.0);
            let (up, down) = (exact(&e.up), exact(&e.down));
            let book = activations_book(&up, hidden, &z);
            let n = book.iter().filter(|&&a| a != 0.0).count();
            assert!((100..400).contains(&n) && list.len().abs_diff(n) <= 1, "{n} neurons on, {} by the spec", list.len());
            let want = second_f64(&down, hidden, &book, 5.0);
            for k in Kernel::available() {
                let mut out = vec![0f32; hidden];
                assert_eq!(experts_with(k, Z::F32(&z), &[HbPair { expert: &e, token: 0, weight: 5.0 }], &mut out, 5).unwrap(), [list.len() as u32]);
                let err = rel(&out, &want);
                eprintln!("bits {bits} {:7}: {n} neurons on, error {err:.2e} against f64", k.name());
                assert!(same_bits(&out, &spec) && err < 1e-4, "bits {bits} {}: {err}", k.name());
                let start = std::time::Instant::now();
                for _ in 0..20 {
                    experts_with(k, Z::F32(&z), &[HbPair { expert: &e, token: 0, weight: 5.0 }], &mut out, 1).unwrap();
                }
                eprintln!("bits {bits} {:7}: {} us a pair on one thread", k.name(), start.elapsed().as_micros() / 20);
            }
        }
    }

    /// Callers on several threads at once (the workers take one job at a time) get what each alone does.
    #[test]
    fn callers_at_once() {
        let mut rng = Rng(77);
        let (inter, hidden) = (256, 128);
        let m = make_sparse(&mut rng, 3, inter, hidden, 0.1);
        let e = m.expert();
        let z: Vec<f32> = (0..hidden).map(|_| bf16_round(0.25 + rng.f())).collect();
        let pairs: Vec<HbPair> = (0..6).map(|i| HbPair { expert: &e, token: 0, weight: 0.5 + i as f32 }).collect();
        let mut want = vec![0f32; pairs.len() * hidden];
        experts(Z::F32(&z), &pairs, &mut want, 1).unwrap();
        std::thread::scope(|s| {
            for t in 0..6 {
                let (z, pairs, want) = (&z, &pairs, &want);
                s.spawn(move || {
                    for _ in 0..20 {
                        let mut got = vec![0f32; want.len()];
                        experts(Z::F32(z), pairs, &mut got, 2 + t % 3).unwrap();
                        assert!(same_bits(&got, want));
                    }
                });
            }
        });
    }

    #[test]
    fn what_is_refused() {
        let mut rng = Rng(66);
        let (a, b, c) = (make(&mut rng, 3, 64, 64, None, false), make(&mut rng, 3, 64, 32, None, false), make(&mut rng, 2, 64, 64, None, false));
        let (ea, eb) = (a.expert(), b.expert());
        let z = vec![0.5f32; 2 * 64];
        let ok = [HbPair { expert: &ea, token: 1, weight: 1.0 }];
        let mut out = vec![0f32; 64];
        assert!(experts(Z::F32(&z), &ok, &mut out, 1).is_ok());
        assert!(experts(Z::F32(&z), &ok, &mut out[..63], 1).is_err(), "out too short");
        assert!(experts(Z::F32(&z[..100]), &ok, &mut out, 1).is_err(), "z not rows of 64");
        assert!(experts(Z::Bf16(&[0; 100]), &ok, &mut out, 1).is_err(), "z not rows of 64, in bf16");
        assert!(experts(Z::F32(&z), &[HbPair { token: 2, ..ok[0] }], &mut out, 1).is_err(), "a token past z");
        assert!(experts(Z::F32(&z), &[ok[0], HbPair { expert: &eb, token: 0, weight: 1.0 }], &mut vec![0f32; 128], 1).is_err(), "experts of other widths");
        if let Some(k) = Kernel::ALL.into_iter().find(|k| !k.runs_here()) {
            assert!(experts_with(k, Z::F32(&z), &ok, &mut out, 1).is_err(), "{} is not here", k.name());
        }
        // experts that are not a layout's: an array too short, a shape that is not the same, other bits
        let bad = HbExpert { up: CbView { codes: &a.up.codes[1..], ..ea.up }, down: ea.down };
        assert!(experts(Z::F32(&z), &[HbPair { expert: &bad, ..ok[0] }], &mut out, 1).is_err(), "codes too short");
        let bad = HbExpert { up: ea.up, down: CbView { scales: &a.down.scales[..a.down.scales.len() - 2], ..ea.down } };
        assert!(experts(Z::F32(&z), &[HbPair { expert: &bad, ..ok[0] }], &mut out, 1).is_err(), "scales too short");
        let bad = HbExpert { up: ea.up, down: CbView { cols: 32, ..ea.down } };
        assert!(experts(Z::F32(&z), &[HbPair { expert: &bad, ..ok[0] }], &mut out, 1).is_err(), "shapes of up and down");
        let bad = HbExpert { up: ea.up, down: c.down.view() };
        assert!(experts(Z::F32(&z), &[HbPair { expert: &bad, ..ok[0] }], &mut out, 1).is_err(), "bits of up and down");
        assert!(HbExpert::new(ea.up, eb.down).is_err() && HbExpert::new(ea.up, c.down.view()).is_err());
        assert!(HbExpert::new(ea.up, ea.down).is_ok());
        // the matrices
        assert!(CbRows::new(3, 2, 64, vec![0; 2 * 24], vec![0; 2 * 2 * 2], vec![0; 2 * 16]).is_ok());
        assert!(CbRows::new(3, 2, 64, vec![0; 2 * 24], vec![0; 7], vec![0; 2 * 16]).is_err() && CbRows::new(4, 2, 64, vec![0; 64], vec![0; 8], vec![0; 64]).is_err());
        assert!(CbRows::from_codes(2, 1, 32, &[4; 32], &[0], &[0; 4]).is_err(), "a code past the bits");
        assert!(CbRows::from_codes(2, 1, 32, &[0; 32], &[0], &[0; 8]).is_err() && CbRows::from_codes(2, 1, 33, &[0; 33], &[0], &[0; 4]).is_err());
        assert!(CbRows::from_codes(1, 1, 32, &[0; 32], &[0], &[0; 2]).is_err() && CbRows::from_codes(2, 0, 32, &[], &[], &[]).is_err());
    }

    #[test]
    fn the_checked_kernel_is_one_that_agrees() {
        for k in Kernel::available() {
            assert!(agrees_with_scalar(k), "{}", k.name());
        }
        assert!(Kernel::available().contains(&checked_kernel()));
    }
}
