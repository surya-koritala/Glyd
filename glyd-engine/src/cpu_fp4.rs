//! Kimi-K3's experts on the host, read where the expert cache's host tier holds them (GLY-330): MXFP4 matrices in the GPU's exact FP4 block layouts
//! (glyd_gpu::pack's Fp4 and Fp4R: Int4's steps of 64 rows x 16 columns of codes, a row's scale for each run of 32 columns, as released or packed). A weight is
//! its E2M1 code times its run's power of two (exact in fp32); the sums are fp32, rounded where the GPU's are: the first product to bf16, the activation (situ)
//! to bf16, the second product to bf16, then times the pair's weight.

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

/// A Kimi-K3 expert where the host tier holds it: its first matrix [2 inter, latent] (gate then up) and its second [latent, inter].
#[derive(Clone, Copy)]
pub struct Fp4Expert<'a> {
    pub up: Fp4View<'a>,
    pub down: Fp4View<'a>,
}

/// A pair of the router's choices the host computes: its expert, its token's row of z, its weight.
pub struct Fp4Pair<'a> {
    pub expert: Fp4Expert<'a>,
    pub token: usize,
    pub weight: f32,
}

fn bf16(v: f32) -> f32 {
    f32::from_bits((((v.to_bits() as u64 + 0x7FFF + ((v.to_bits() >> 16) & 1) as u64) as u32) >> 16) << 16)
}

/// situ(a, b) = beta tanh(a / beta) sigmoid(a) x linear tanh(b / linear), in fp32, rounded to bf16 (moe.cu's MOE_SITU, kimi_ref's situ).
fn situ(a: f32, b: f32, (beta, linear): (f32, f32)) -> f32 {
    bf16(beta * (a / beta).tanh() * (1.0 / (1.0 + (-a).exp())) * linear * (b / linear).tanh())
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

/// out[p latent ..] = pair p's output times its weight: the second matrix times situ(gate, up), [gate | up] the first matrix times its token's row of z (bf16),
/// on `threads` threads (the crew of src/cpu_experts.rs; `pin`: its threads each kept to a core).
pub fn experts(kernel: Kernel, z: &[u16], pairs: &[Fp4Pair], out: &mut [f32], act: (f32, f32), threads: usize, pin: bool) -> Result<()> {
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
    let zf: Vec<Vec<f32>> =
        pairs.iter().map(|p| z[p.token * latent..(p.token + 1) * latent].iter().map(|&h| f32::from_bits((h as u32) << 16)).collect()).collect();
    // the first products: each pair's rows in blocks of 64
    let mut gu = vec![0f32; pairs.len() * two];
    let (gp, ub) = (Out(gu.as_mut_ptr()), two / 64);
    crate::cpu_experts::run(threads, pin, pairs.len() * ub, &|item| {
        let (p, b) = (item / ub, item % ub);
        // SAFETY: item (p, b) writes rows 64 b .. 64 b + 64 of pair p's part of gu alone, which is live for the call.
        let y = unsafe { gp.part(p * two + 64 * b, 64) };
        rows(kernel, &pairs[p].expert.up, &zf[p], 64 * b, 64 * b + 64, y);
    });
    let a: Vec<f32> =
        (0..pairs.len()).flat_map(|p| (0..inter).map(move |i| (p, i))).map(|(p, i)| situ(bf16(gu[p * two + i]), bf16(gu[p * two + inter + i]), act)).collect();
    // the second products
    let (op, db) = (Out(out.as_mut_ptr()), latent / 64);
    crate::cpu_experts::run(threads, pin, pairs.len() * db, &|item| {
        let (p, b) = (item / db, item % db);
        // SAFETY: item (p, b) writes rows 64 b .. 64 b + 64 of pair p's part of out alone, which is live for the call.
        let y = unsafe { op.part(p * latent + 64 * b, 64) };
        rows(kernel, &pairs[p].expert.down, &a[p * inter..(p + 1) * inter], 64 * b, 64 * b + 64, y);
        let w = pairs[p].weight;
        for v in y.iter_mut() {
            *v = bf16(*v) * w;
        }
    });
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
                    experts(kernel, &z, &pairs, &mut out, (4.0, 25.0), t, pin).map(|_| t0.elapsed().as_secs_f64() * 1e6)
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
                experts(k, &z, &pairs[..if t == 1 { 2 } else { 16 }], &mut out[..if t == 1 { 2 } else { 16 } * latent], (4.0, 25.0), t, false).unwrap();
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

    /// A pair's output is the second matrix times situ of the first's halves, times its weight; the threads change nothing.
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
        let act = (4.0, 25.0);
        let mut one = vec![0f32; 2 * latent];
        experts(Kernel::Scalar, &z, &pairs, &mut one, act, 1, false).unwrap();
        // (by hand, for the first pair: scalar rows of each matrix)
        let zt: Vec<f32> = z[latent..2 * latent].iter().map(|&h| f32::from_bits((h as u32) << 16)).collect();
        let mut gu = vec![0f32; 2 * inter];
        rows(Kernel::Scalar, &ex.up, &zt, 0, 2 * inter, &mut gu);
        let a: Vec<f32> = (0..inter).map(|i| situ(bf16(gu[i]), bf16(gu[inter + i]), act)).collect();
        let mut d = vec![0f32; latent];
        rows(Kernel::Scalar, &ex.down, &a, 0, latent, &mut d);
        let want: Vec<f32> = d.iter().map(|&v| bf16(v) * 0.25).collect();
        assert_eq!(&one[..latent], &want[..]);
        let mut four = vec![0f32; 2 * latent];
        experts(Kernel::Scalar, &z, &pairs, &mut four, act, 4, false).unwrap();
        assert_eq!(one, four);
    }
}
