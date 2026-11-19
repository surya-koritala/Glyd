//! `glyd-engine cpuexperts`: the CPU half of the sparse experts offload (src/cpu_experts.rs) timed on this computer, with experts of Nemotron 3 Super's shape (2,688 neurons of 1,024; 22 of 512 experts a token in each of 40 layers).
//! The experts' bytes are made here (random codes and scales in the row layout, no checkpoint and no GPU), with the neurons that relu squared leaves on chosen row by row so that a pair has the number of active neurons the
//! offload probe measured (benchmarks/weights/offload-probe-2026-10-10: the number of active neurons of a pair over the 43,253,760 held-out pairs, its quantiles below): an active row is one whose codes are all at or above
//! the bias, an off one has them all below, and z is positive, so that the first product's sums are positive or negative as the row says. Reported: the computer's streaming read bandwidth (all the threads reading a
//! buffer), pairs a second and GB/s read for one call of many pairs, and the time a decode token takes for M missed experts (the misses spread over the 40 layers, a call a layer, each layer's z its own).
//!
//!   glyd-engine cpuexperts [--bits 4|5|6] [--threads N] [--kernel auto|scalar|neon|avx2|avx512] [--experts N (1024)] [--pairs N (512)] [--reps N (5)] [--tokens N (20)] [--misses M,M,... (100,230,400)] [--gib N] [--seed S]

use crate::args::Args;
use crate::cpu_experts::{experts_with, row_stride, run as run_items, Expert, Kernel, Pair, Rows, Z};
use crate::ffi::{err, Error, Result};
use std::time::Instant;

/// Neurons and width of Nemotron 3 Super's experts, and its layers with experts.
const INTER: usize = 2688;
const HIDDEN: usize = 1024;
const LAYERS: usize = 40;

/// The quantiles (0 to 100%, a value for each percent) of the number of neurons that are not 0 after relu squared in a pair of the probe's held-out text, all 40 layers pooled (the probe's `hz` histogram).
const ACTIVE_QUANTILES: [u16; 101] = [
    0, 3, 7, 11, 16, 20, 24, 28, 31, 35, 39, 42, 45, 48, 51, 54, 57, 60, 63, 66, 69, 71, 74, 77, 80, 82, 85, 88, 90, 93, 96, 99, 102, 104, 107, 110, 113, 116, 119, 122, 125, 128, 131, 135, 138, 141, 144, 148, 151, 155, 158, 162, 166, 170, 174,
    178, 182, 186, 190, 194, 199, 203, 208, 213, 217, 222, 228, 233, 238, 244, 249, 255, 261, 267, 274, 280, 287, 294, 302, 310, 318, 327, 336, 345, 355, 366, 378, 391, 405, 420, 437, 457, 479, 506, 540, 583, 645, 762, 1186, 1367, 2331,
];

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }

    /// `k` distinct numbers of 0 .. n.
    fn pick(&mut self, n: usize, k: usize) -> Vec<usize> {
        let mut all: Vec<usize> = (0..n).collect();
        for i in 0..k.min(n) {
            let j = i + self.below(n - i);
            all.swap(i, j);
        }
        all.truncate(k.min(n));
        all
    }

    /// A number of active neurons from the probe's distribution (linear between the percent quantiles).
    fn active(&mut self) -> usize {
        let x = (self.next() >> 11) as f64 / (1u64 << 53) as f64 * 100.0;
        let (i, f) = (x as usize, x.fract());
        let (a, b) = (ACTIVE_QUANTILES[i] as f64, ACTIVE_QUANTILES[(i + 1).min(100)] as f64);
        ((a + (b - a) * f).round() as usize).min(INTER)
    }

    fn fill(&mut self, bytes: &mut [u8]) {
        for c in bytes.chunks_mut(8) {
            let v = self.next().to_le_bytes();
            c.copy_from_slice(&v[..c.len()]);
        }
    }
}

/// A matrix [INTER, HIDDEN] of random codes in the row layout: row r's codes are all at or above the bias if `on[r]`, all below it if not (`on` empty: any codes), scales positive bf16 between 2^-7 and 2.
fn matrix(bits: u32, on: &[bool], rng: &mut Rng) -> Result<Rows> {
    let (runs, stride) = (HIDDEN / 32, row_stride(bits, HIDDEN));
    let (mut data, mut scales) = (vec![0u8; INTER * stride], vec![0u8; INTER * runs * 2]);
    rng.fill(&mut data);
    rng.fill(&mut scales);
    for s in scales.chunks_mut(2) {
        let v = 0x3C00 + (u16::from_le_bytes([s[0], s[1]]) & 0x03FF);
        s.copy_from_slice(&v.to_le_bytes());
    }
    for (r, row) in data.chunks_mut(stride).enumerate().filter(|_| !on.is_empty()) {
        let up = on[r];
        let (lo, hi) = row.split_at_mut(16 * runs);
        for b in lo.iter_mut() {
            // (four bits: both nibbles' top bit, which is the bias's)
            *b = if bits == 4 { if up { *b | 0x88 } else { *b & 0x77 } } else { *b };
        }
        // five and six bits: the top bit of every code, a word of the run's high bits at a time
        for w in hi.chunks_mut(4) {
            let v = u32::from_le_bytes([w[0], w[1], w[2], w[3]]);
            let v = match (bits, up) {
                (5, true) => u32::MAX,
                (5, false) => 0,
                (_, true) => 0xAAAA_AAAA | (v & 0x5555_5555),
                (_, false) => v & 0x5555_5555,
            };
            w.copy_from_slice(&v.to_le_bytes());
        }
    }
    Rows::new(bits, INTER, HIDDEN, data, scales)
}

/// An expert with `active` of its neurons on (against a positive z) and random down rows.
fn expert(bits: u32, active: usize, rng: &mut Rng) -> Result<(Expert, usize)> {
    let mut on = vec![false; INTER];
    rng.pick(INTER, active).into_iter().for_each(|r| on[r] = true);
    Ok((Expert::new(matrix(bits, &on, rng)?, matrix(bits, &[], rng)?)?, active))
}

/// A token's z: positive bf16 values between 0.5 and 2.
fn z_row(rng: &mut Rng) -> Vec<u16> {
    (0..HIDDEN).map(|_| 0x3F00 + (rng.next() & 0xFF) as u16).collect()
}

fn cpu_name() -> String {
    let run = |cmd: &str, args: &[&str]| std::process::Command::new(cmd).args(args).output().ok().filter(|o| o.status.success()).map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string());
    run("sysctl", &["-n", "machdep.cpu.brand_string"])
        .or_else(|| std::fs::read_to_string("/proc/cpuinfo").ok().and_then(|t| t.lines().find(|l| l.starts_with("model name")).and_then(|l| l.split_once(':')).map(|(_, v)| v.trim().to_string())))
        .unwrap_or_else(|| "an unknown processor".to_string())
}

/// GB/s of memory that `threads` threads read, best of `reps`, from a buffer of `gib` GiB.
fn stream_read(threads: usize, gib: f64, reps: usize) -> f64 {
    let each = ((gib * (1u64 << 30) as f64 / 8.0) as usize / threads / 8 * 8).max(8);
    let buf: Vec<u64> = (0..(each * threads) as u64).collect();
    let sum = |c: &[u64]| {
        let mut a = [0u64; 8];
        for w in c.as_chunks::<8>().0 {
            for i in 0..8 {
                a[i] = a[i].wrapping_add(w[i]);
            }
        }
        a.iter().fold(0u64, |s, x| s.wrapping_add(*x))
    };
    let mut best = 0f64;
    for _ in 0..reps {
        let t = Instant::now();
        let total = std::thread::scope(|s| {
            let parts: Vec<_> = buf.chunks(each).map(|c| s.spawn(move || sum(c))).collect();
            parts.into_iter().fold(0u64, |t, p| t.wrapping_add(p.join().unwrap()))
        });
        std::hint::black_box(total);
        best = best.max(buf.len() as f64 * 8.0 / t.elapsed().as_secs_f64() / 1e9);
    }
    best
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.total_cmp(b));
    v[v.len() / 2]
}

/// Pairs of token 0 through the experts `ids` of the pool, with random routing weights.
fn pairs_of<'a>(pool: &'a [(Expert, usize)], ids: &[usize], rng: &mut Rng) -> Vec<Pair<'a>> {
    ids.iter().map(|&e| Pair { expert: &pool[e].0, token: 0, weight: 0.5 + (rng.next() >> 40) as f32 / (1u64 << 24) as f32 }).collect()
}

/// A codebook matrix [INTER, HIDDEN] of random codes at `bits` (2 or 3): row r's table is all positive if `on[r]`, all negative if not, the scales positive, so that against a positive z the row's sum has the sign made.
fn cb_matrix(bits: u32, on: &[bool], rng: &mut Rng) -> Result<crate::cpu_hb::CbRows> {
    let (cs, ss, bs) = crate::cpu_hb::strides(bits, HIDDEN);
    let (mut codes, mut scales, mut book) = (vec![0u8; INTER * cs], vec![0u8; INTER * ss], vec![0u8; INTER * bs]);
    rng.fill(&mut codes);
    for s in scales.chunks_mut(2) {
        s.copy_from_slice(&(0x3C00 + (rng.next() & 0x03FF) as u16).to_le_bytes());
    }
    for (r, row) in book.chunks_mut(bs).enumerate() {
        let up = on.get(r).copied().unwrap_or(true);
        for e in row.chunks_mut(2) {
            // (bf16 between 0.5 and 1.5, with the sign bit set for an off row)
            e.copy_from_slice(&(0x3F00 + (rng.next() & 0x00FF) as u16 | if up { 0 } else { 0x8000 }).to_le_bytes());
        }
    }
    crate::cpu_hb::CbRows::new(bits, INTER, HIDDEN, codes, scales, book)
}

/// `bytes` of memory aligned to 2 MiB that the kernel is asked to back with huge pages (`madvise(MADV_HUGEPAGE)`: the transparent huge pages of a system in `madvise` mode), not zeroed, never freed (a benchmark's).
struct Arena(*mut u8);

fn huge_arena(bytes: usize) -> Arena {
    #[cfg(unix)]
    extern "C" {
        fn madvise(addr: *mut std::ffi::c_void, len: usize, advice: i32) -> i32;
    }
    let len = bytes.div_ceil(2 << 20) * (2 << 20);
    let layout = std::alloc::Layout::from_size_align(len, 2 << 20).expect("a layout");
    // SAFETY: a fresh allocation of `len` bytes, advised before it is touched; the advice is a hint a refusal of which leaves ordinary pages.
    unsafe {
        let p = std::alloc::alloc(layout);
        assert!(!p.is_null(), "{len} bytes");
        #[cfg(unix)]
        madvise(p.cast(), len, 14);
        Arena(p)
    }
}

/// `cpuexperts --hb`: the same for the Hummingbird experts (codebook rows of 2 and 3 bits, src/cpu_hb.rs), 40% of them at three bits: calls of N pairs in the way the decode steps make them (a token's z, N distinct experts of the pool, 1 to 64 pairs), the time of a call and
/// what a pair adds to it, and the bytes a call reads (`--huge`: from memory the kernel backs with huge pages).
fn run_hb(a: &Args) -> Result<()> {
    use crate::cpu_hb::{experts_pinned, HbExpert, HbPair};
    let threads = match a.num("threads", 0usize)? {
        0 => std::thread::available_parallelism().map_or(1, |n| n.get()),
        n => n,
    };
    let (n_experts, reps) = (a.num("experts", 256usize)?, a.num("reps", 200usize)?);
    let mut rng = Rng(a.num("seed", 1u64)?.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
    let kernel = crate::cpu_hb::checked_kernel();
    println!("{} ({}), {threads} threads, kernel {}: Hummingbird experts of {INTER} neurons x {HIDDEN}", cpu_name(), std::env::consts::ARCH, kernel.name());
    println!("streaming read, {threads} threads: {:.1} GB/s", stream_read(threads, 1.0, 3));
    let t = Instant::now();
    let mut pool: Vec<(crate::cpu_hb::CbRows, crate::cpu_hb::CbRows, u32)> = Vec::new();
    for i in 0..n_experts {
        let bits = if i % 5 < 2 { 3 } else { 2 };
        let mut on = vec![false; INTER];
        let n_on = rng.active();
        rng.pick(INTER, n_on).into_iter().for_each(|r| on[r] = true);
        pool.push((cb_matrix(bits, &on, &mut rng)?, cb_matrix(bits, &[], &mut rng)?, bits));
    }
    // (--huge: the experts' bytes copied into memory the kernel is asked to back with 2 MiB pages, as the host tier could be; else the pool's own 4 KiB pages)
    let arena = a.has("huge").then(|| huge_arena(pool.iter().map(|(u, d, _)| u.codes.len() + u.scales.len() + u.codebook.len() + d.codes.len() + d.scales.len() + d.codebook.len() + 6 * 64).sum()));
    let mut at = 0usize;
    let mut homed = |m: &crate::cpu_hb::CbRows| -> crate::cpu_hb::CbView<'static> {
        let arena = arena.as_ref().expect("the arena");
        let mut put = |bytes: &[u8]| -> &'static [u8] {
            at = at.div_ceil(64) * 64;
            // SAFETY: the arena is leaked (a benchmark's), `at` stays inside it, and the ranges are disjoint.
            let dst = unsafe { std::slice::from_raw_parts_mut(arena.0.add(at), bytes.len()) };
            dst.copy_from_slice(bytes);
            at += bytes.len();
            dst
        };
        crate::cpu_hb::CbView { bits: m.bits, rows: m.rows, cols: m.cols, codes: put(&m.codes), scales: put(&m.scales), codebook: put(&m.codebook) }
    };
    let views: Vec<(crate::cpu_hb::CbView, crate::cpu_hb::CbView)> = pool.iter().map(|(u, d, _)| if arena.is_some() { (homed(u), homed(d)) } else { (u.view(), d.view()) }).collect();
    let experts: Vec<HbExpert> = views.iter().map(|(u, d)| HbExpert::new(*u, *d)).collect::<Result<_>>()?;
    let bytes_of = |e: usize| (INTER * (crate::cpu_hb::strides(pool[e].2, HIDDEN).0 + crate::cpu_hb::strides(pool[e].2, HIDDEN).1 + crate::cpu_hb::strides(pool[e].2, HIDDEN).2)) as f64;
    println!("{n_experts} experts made in {:.1} s ({:.2} GB)", t.elapsed().as_secs_f64(), (0..n_experts).map(|e| 2.0 * bytes_of(e)).sum::<f64>() / 1e9);
    let z: Vec<f32> = (0..HIDDEN).map(|_| 0.5 + (rng.next() >> 40) as f32 / (1u64 << 24) as f32).collect();
    for n in [1usize, 2, 4, 6, 10, 16, 32, 64] {
        let (mut times, mut out) = (Vec::new(), vec![0f32; n * HIDDEN]);
        let mut read = 0.0;
        for rep in 0..reps + 5 {
            let ids = rng.pick(n_experts, n);
            let pairs: Vec<HbPair> = ids.iter().map(|&e| HbPair { expert: &experts[e], token: 0, weight: 1.0 }).collect();
            let t = Instant::now();
            experts_pinned(kernel, Z::F32(&z), &pairs, &mut out, threads, a.has("pin"))?;
            let s = t.elapsed().as_secs_f64();
            if rep >= 5 {
                times.push(s);
                read = ids.iter().map(|&e| bytes_of(e) / 2.0).sum::<f64>();
            }
        }
        let med = median(&mut times);
        println!("a call of {n:>2} pairs: {:>7.1} us median ({:.1} us a pair, {:.1} GB/s of the up matrices' bytes)", med * 1e6, med * 1e6 / n as f64, read / med / 1e9);
    }
    Ok(())
}

/// A codebook matrix [rows, cols] of random codes, positive scales around 1 and tables of bf16 entries between -1.5 and 1.5.
fn cb_random(bits: u32, rows: usize, cols: usize, rng: &mut Rng) -> Result<crate::cpu_hb::CbRows> {
    let (cs, ss, bs) = crate::cpu_hb::strides(bits, cols);
    let (mut codes, mut scales, mut book) = (vec![0u8; rows * cs], vec![0u8; rows * ss], vec![0u8; rows * bs]);
    rng.fill(&mut codes);
    for s in scales.chunks_mut(2) {
        s.copy_from_slice(&(0x3C00 + (rng.next() & 0x03FF) as u16).to_le_bytes());
    }
    for e in book.chunks_mut(2) {
        let v = (rng.next() & 0x00FF) as u16;
        e.copy_from_slice(&(0x3F00 + v | if rng.next() & 1 == 1 { 0x8000 } else { 0 }).to_le_bytes());
    }
    crate::cpu_hb::CbRows::new(bits, rows, cols, codes, scales, book)
}

/// `cpuexperts --glm`: GLM-5.3-Flash's routed experts (SwiGLU, 2,048 neurons of 4,096, every neuron active: gate and up [2048, 4096], and the second matrix held transposed, [2048 neurons, 4096 outputs]) as codebook rows at
/// `--bits` 2 or 3, run with the Hummingbird CPU kernels: the first products split by blocks of 128 rows, the activation silu(min(gate, 10)) max(min(up, 10), -10), the second product split by blocks of 128 neurons, whose
/// partial sums are added. Calls of N experts (1 to 16) of a pool of `--experts` (48: about 380 MB at two bits, from memory; 4 fit the L3 caches): the time of a call, an expert, the weights it passes a second and the bytes it reads.
fn run_glm(a: &Args) -> Result<()> {
    use crate::cpu_hb::CbRows;
    use std::sync::Mutex;
    const NI: usize = 2048;
    const NH: usize = 4096;
    const BLOCK: usize = 128;
    let threads = match a.num("threads", 0usize)? {
        0 => std::thread::available_parallelism().map_or(1, |n| n.get()),
        n => n,
    };
    let (bits, n_experts, reps) = (a.num("bits", 2u32)?, a.num("experts", 48usize)?, a.num("reps", 60usize)?);
    let (pin, mut rng) = (a.has("pin"), Rng(a.num("seed", 1u64)?.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1));
    let kernel = crate::cpu_hb::checked_kernel();
    println!("{} ({}), {threads} threads, kernel {}: GLM-5.3-Flash experts, {bits}-bit codebook rows", cpu_name(), std::env::consts::ARCH, kernel.name());
    println!("streaming read, {threads} threads: {:.1} GB/s", stream_read(threads, 1.0, 3));
    let pool: Vec<[CbRows; 3]> = (0..n_experts).map(|_| Ok([cb_random(bits, NI, NH, &mut rng)?, cb_random(bits, NI, NH, &mut rng)?, cb_random(bits, NI, NH, &mut rng)?])).collect::<Result<_>>()?;
    let (cs, ss, bs) = crate::cpu_hb::strides(bits, NH);
    let bytes = (3 * NI * (cs + ss + bs)) as f64;
    println!("{n_experts} experts made ({:.0} MB), {:.2} MB an expert", n_experts as f64 * bytes / 1e6, bytes / 1e6);
    let z: Vec<f32> = (0..NH).map(|_| (rng.next() >> 40) as f32 / (1u64 << 23) as f32 - 1.0).collect();
    for n in [1usize, 2, 4, 8, 16].into_iter().filter(|&n| n <= n_experts) {
        let mut times = Vec::new();
        for rep in 0..reps + 5 {
            let ids = rng.pick(n_experts, n);
            let t = Instant::now();
            let gate_up: Vec<Mutex<Vec<f32>>> = (0..n * 2).map(|_| Mutex::new(vec![0f32; NI])).collect();
            let per = NI / BLOCK;
            run_items(threads, pin, n * 2 * per, &|item| {
                let (em, b) = (item / per, item % per);
                let mut sums = [0f32; BLOCK];
                crate::cpu_simd::up_cb(kernel, &pool[ids[em / 2]][em % 2].view(), &z, b * BLOCK, (b + 1) * BLOCK, &mut sums);
                gate_up[em].lock().unwrap()[b * BLOCK..(b + 1) * BLOCK].copy_from_slice(&sums);
            });
            let act: Vec<Vec<(u32, f32)>> = (0..n)
                .map(|e| {
                    let (g, u) = (gate_up[2 * e].lock().unwrap(), gate_up[2 * e + 1].lock().unwrap());
                    (0..NI).map(|j| (j as u32, g[j].min(10.0) / (1.0 + (-g[j].min(10.0)).exp()) * u[j].clamp(-10.0, 10.0))).collect()
                })
                .collect();
            let out: Vec<Mutex<Vec<f32>>> = (0..n).map(|_| Mutex::new(vec![0f32; NH])).collect();
            run_items(threads, pin, n * per, &|item| {
                let (e, b) = (item / per, item % per);
                let mut acc = vec![0f32; NH];
                crate::cpu_simd::down_cb(kernel, &pool[ids[e]][2].view(), &act[e][b * BLOCK..(b + 1) * BLOCK], &mut acc);
                out[e].lock().unwrap().iter_mut().zip(&acc).for_each(|(o, x)| *o += *x);
            });
            std::hint::black_box(&out);
            if rep >= 5 {
                times.push(t.elapsed().as_secs_f64());
            }
        }
        let med = median(&mut times);
        println!("a call of {n:>2} experts: {:>8.1} us ({:.1} us an expert), {:.0} G weight-ops/s, {:.1} GB/s read", med * 1e6, med * 1e6 / n as f64, n as f64 * 3.0 * (NI * NH) as f64 / med / 1e9, n as f64 * bytes / med / 1e9);
    }
    Ok(())
}

/// A pack file's bytes mapped read-only (the way the engine's host tier holds them), unmapped when dropped.
struct PackMap {
    ptr: *mut u8,
    len: usize,
}

#[cfg(unix)]
extern "C" {
    fn mmap(addr: *mut u8, len: usize, prot: i32, flags: i32, fd: i32, off: i64) -> *mut u8;
    fn munmap(addr: *mut u8, len: usize) -> i32;
    #[cfg(target_os = "linux")]
    fn posix_fadvise(fd: i32, off: i64, len: i64, advice: i32) -> i32;
}

impl PackMap {
    fn open(path: &std::path::Path, drop_cached: bool) -> Result<PackMap> {
        #[cfg(unix)]
        use std::os::unix::io::AsRawFd;
        let f = std::fs::File::open(path).map_err(|e| Error(format!("{}: {e}", path.display())))?;
        let len = f.metadata().map_err(|e| Error(format!("{}: {e}", path.display())))?.len() as usize;
        #[cfg(target_os = "linux")]
        if drop_cached {
            // SAFETY: an advice about a file we hold open (POSIX_FADV_DONTNEED = 4): its clean cached pages are let go.
            unsafe { posix_fadvise(f.as_raw_fd(), 0, 0, 4) };
        }
        #[cfg(not(target_os = "linux"))]
        let _ = drop_cached;
        // SAFETY: a read-only private mapping of a file held open; it outlives the descriptor.
        #[cfg(unix)]
        let ptr = unsafe { mmap(std::ptr::null_mut(), len, 1, 2, f.as_raw_fd(), 0) };
        #[cfg(unix)]
        if ptr as usize == usize::MAX {
            return err(format!("{}: cannot be mapped", path.display()));
        }
        #[cfg(windows)]
        let ptr = crate::os::win::map_read(&f).map_err(|e| Error(format!("{}: cannot be mapped: {e}", path.display())))? as *mut u8;
        Ok(PackMap { ptr, len })
    }

    fn bytes(&self, offset: u64, n: u64) -> &[u8] {
        assert!(offset + n <= self.len as u64);
        // SAFETY: inside the mapping, which lives as long as `self` and is only read.
        unsafe { std::slice::from_raw_parts(self.ptr.add(offset as usize), n as usize) }
    }
}

impl Drop for PackMap {
    fn drop(&mut self) {
        // SAFETY: mapped by `open`, unmapped once.
        #[cfg(unix)]
        unsafe { munmap(self.ptr, self.len) };
        #[cfg(windows)]
        unsafe { crate::os::win::unmap(self.ptr.cast()) };
    }
}

/// `cpuexperts --cb2 --pack DIR [--layer L (3)] [--copy] [--cold N]`: the real experts of a layer file of a `glyd-hbr-codes/2` pack, read where the file is mapped (or, with `--copy`, from copies in the heap), with the
/// same calls as the random experts' (all the file's experts, 1.37 GB, so that the reads come from memory), and with `--cold N` N calls of one expert each whose pages were let go of first (the file's cached pages
/// dropped before the mapping): the time of a call that has to read its expert from the disk.
fn run_cb2_pack(a: &Args, dir: &str) -> Result<()> {
    use crate::cpu_cb::{experts_pinned, GlmExpert, GlmPair};
    use crate::cpu_hb::CbView;
    use glyd_gpu::safetensors::File;
    const HID: usize = 4096;
    let threads = match a.num("threads", 0usize)? {
        0 => std::thread::available_parallelism().map_or(1, |n| n.get()),
        n => n,
    };
    let (layer, reps, pin, copy, cold) = (a.num("layer", 3usize)?, a.num("reps", 60usize)?, a.has("pin"), a.has("copy"), a.num("cold", 0usize)?);
    let mut rng = Rng(a.num("seed", 1u64)?.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
    let kernel = crate::cpu_hb::checked_kernel();
    let path = std::path::Path::new(dir).join(format!("layer-{layer:03}.safetensors"));
    let file = File::open(&path).map_err(|e| Error(format!("{}: {e}", path.display())))?;
    let map = PackMap::open(&path, cold > 0)?;
    let mut kept: Vec<usize> = file.tensors.iter().filter_map(|(n, _)| n.strip_suffix(".u.0").and_then(|e| e.strip_prefix('e')).and_then(|e| e.parse().ok())).collect();
    kept.sort();
    println!("{} ({}), {threads} threads, kernel {}: layer {layer} of {dir}: {} experts, {}", cpu_name(), std::env::consts::ARCH, kernel.name(), kept.len(), if copy { "copied to the heap" } else { "read where the file is mapped" });
    let arrays = |e: usize| -> Result<[&[u8]; 4]> {
        let get = |what: &str| -> Result<&[u8]> {
            let name = format!("e{e:03}.{what}");
            let t = file.get(&name).ok_or_else(|| Error(format!("{name}: not in the file")))?;
            Ok(map.bytes(t.offset, t.bytes))
        };
        Ok([get("u.0")?, get("u.1")?, get("d.0")?, get("d.1")?])
    };
    let copies: Vec<[Vec<u8>; 4]> = if copy { kept.iter().map(|&e| arrays(e).map(|m| m.map(|s| s.to_vec()))).collect::<Result<_>>()? } else { Vec::new() };
    let experts: Vec<GlmExpert> = kept
        .iter()
        .enumerate()
        .map(|(i, &e)| {
            let m: [&[u8]; 4] = if copy { [&copies[i][0], &copies[i][1], &copies[i][2], &copies[i][3]] } else { arrays(e)? };
            let view = |rows: usize, codes: &'static [u8], codebook: &'static [u8]| CbView { bits: 2, rows, cols: HID, codes, scales: &[], codebook };
            // SAFETY: the mapping and the copies outlive the experts (both are dropped after this function's last use of them).
            let (uc, ub, dc, db) = unsafe { (std::mem::transmute::<&[u8], &'static [u8]>(m[0]), std::mem::transmute::<&[u8], &'static [u8]>(m[1]), std::mem::transmute::<&[u8], &'static [u8]>(m[2]), std::mem::transmute::<&[u8], &'static [u8]>(m[3])) };
            GlmExpert::new(view(4096, uc, ub), view(2048, dc, db), 10.0)
        })
        .collect::<Result<_>>()?;
    let bytes = experts[0].bytes() as f64;
    let z: Vec<f32> = (0..HID).map(|_| (rng.next() >> 40) as f32 / (1u64 << 23) as f32 - 1.0).collect();
    if cold > 0 {
        // the first reads: a call of one expert nothing of which is in memory (its pages let go of before the mapping)
        let order = rng.pick(kept.len(), cold.min(kept.len()));
        let mut out = vec![0f32; HID];
        let mut ms = Vec::new();
        for &i in &order {
            let t = Instant::now();
            experts_pinned(kernel, Z::F32(&z), &[GlmPair { expert: &experts[i], token: 0, weight: 0.5 }], &mut out, threads, pin)?;
            ms.push(t.elapsed().as_secs_f64() * 1e3);
        }
        let med = median(&mut ms.clone());
        println!("first reads (pages let go of): {} calls of 1 expert: median {:.2} ms, {:.0} to {:.0} ms, {:.2} GB/s of the expert's bytes at the median", ms.len(), med, ms.iter().cloned().fold(f64::MAX, f64::min), ms.iter().cloned().fold(0.0, f64::max), bytes / (med / 1e3) / 1e9);
        let t = Instant::now();
        let again = order.iter().map(|&i| GlmPair { expert: &experts[i], token: 0, weight: 0.5 }).collect::<Vec<_>>();
        let mut outs = vec![0f32; again.len() * HID];
        experts_pinned(kernel, Z::F32(&z), &again, &mut outs, threads, pin)?;
        println!("the same {} experts again in one call (now in memory): {:.2} ms, {:.1} GB/s", again.len(), t.elapsed().as_secs_f64() * 1e3, again.len() as f64 * bytes / t.elapsed().as_secs_f64() / 1e9);
        return Ok(());
    }
    // every expert read once first, so that the sweep below is of pages in memory
    let mut warm = vec![0f32; HID];
    for e in &experts {
        experts_pinned(kernel, Z::F32(&z), &[GlmPair { expert: e, token: 0, weight: 0.5 }], &mut warm, threads, pin)?;
    }
    for n in [1usize, 2, 4, 8, 16].into_iter().filter(|&n| n <= experts.len()) {
        let (mut times, mut out) = (Vec::new(), vec![0f32; n * HID]);
        for rep in 0..reps + 5 {
            let ids = rng.pick(experts.len(), n);
            let pairs: Vec<GlmPair> = ids.iter().map(|&e| GlmPair { expert: &experts[e], token: 0, weight: 0.5 }).collect();
            let t = Instant::now();
            experts_pinned(kernel, Z::F32(&z), &pairs, &mut out, threads, pin)?;
            if rep >= 5 {
                times.push(t.elapsed().as_secs_f64());
            }
        }
        let med = median(&mut times);
        println!("a call of {n:>2} pairs: {:>8.1} us ({:.1} us a pair), {:.1} GB/s of the experts' bytes", med * 1e6, med * 1e6 / n as f64, n as f64 * bytes / med / 1e9);
    }
    drop(experts);
    Ok(())
}

/// `cpuexperts --cb2`: GLM-5.3-Flash's experts as the `glyd-hbr-codes/2` pack holds them (src/cpu_cb.rs: two-bit codebook rows with no run scales, up [4096, 4096] and the transposed down [2048, 4096], clamped SwiGLU) on random codes
/// and tables: calls of N pairs (distinct experts of a pool of `--experts`, 48 by default: about 300 MB, from memory) over `--threads`, the time of a call and of a pair, and the bytes of the experts' arrays a second.
fn run_cb2(a: &Args) -> Result<()> {
    if let Some(dir) = a.flag("pack") {
        return run_cb2_pack(a, dir);
    }
    use crate::cpu_cb::{experts_pinned, GlmExpert, GlmPair};
    const INTER: usize = 2048;
    const HID: usize = 4096;
    let threads = match a.num("threads", 0usize)? {
        0 => std::thread::available_parallelism().map_or(1, |n| n.get()),
        n => n,
    };
    let (n_experts, reps, pin) = (a.num("experts", 48usize)?, a.num("reps", 60usize)?, a.has("pin"));
    let mut rng = Rng(a.num("seed", 1u64)?.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
    let kernel = crate::cpu_hb::checked_kernel();
    println!("{} ({}), {threads} threads, kernel {}: GLM-5.3-Flash experts, two-bit codebook rows with no run scales (glyd-hbr-codes/2)", cpu_name(), std::env::consts::ARCH, kernel.name());
    println!("streaming read, {threads} threads: {:.1} GB/s", stream_read(threads, 1.0, 3));
    // the formats: --fmt cb2 (default), cb3, cb4, tern, or X+Y for the up and the down matrix
    let fmts = a.flag("fmt").unwrap_or("cb2");
    let (fu, fd) = match fmts.split_once('+') {
        Some((u, d)) => (u, d),
        None => (fmts, fmts),
    };
    let (fu, fd) = (crate::cpu_cb::Fmt::parse(fu).ok_or_else(|| Error(format!("--fmt {fmts}: tern, cb2, cb3 or cb4, or two of them with a + between")))?, crate::cpu_cb::Fmt::parse(fd).ok_or_else(|| Error(format!("--fmt {fmts}: tern, cb2, cb3 or cb4")))?);
    println!("formats: up {} down {}", fu.name(), fd.name());
    let mut matrix = |fmt: crate::cpu_cb::Fmt, rows: usize| -> (Vec<u8>, Vec<u8>) {
        let mut codes = vec![0u8; rows * fmt.row_bytes(HID)];
        rng.fill(&mut codes);
        if fmt == crate::cpu_cb::Fmt::Tern {
            // (a byte is five base-3 digits: 0 to 242)
            codes.iter_mut().for_each(|b| *b %= 243);
        }
        let book: Vec<u8> = (0..rows * fmt.entries()).flat_map(|_| (((rng.next() >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0).to_bits() >> 16).to_le_bytes()[..2].to_vec()).collect();
        (codes, book)
    };
    let pool: Vec<((Vec<u8>, Vec<u8>), (Vec<u8>, Vec<u8>))> = (0..n_experts).map(|_| (matrix(fu, 2 * INTER), matrix(fd, INTER))).collect();
    fn view(fmt: crate::cpu_cb::Fmt, rows: usize, m: &(Vec<u8>, Vec<u8>)) -> crate::cpu_cb::GlmMat<'_> {
        crate::cpu_cb::GlmMat { fmt, rows, cols: HID, codes: &m.0, book: &m.1 }
    }
    let experts: Vec<GlmExpert> = pool.iter().map(|(u, d)| GlmExpert::mixed(view(fu, 2 * INTER, u), view(fd, INTER, d), 10.0)).collect::<Result<_>>()?;
    let bytes = experts[0].bytes() as f64;
    println!("{n_experts} experts made ({:.0} MB), {:.2} MB an expert", n_experts as f64 * bytes / 1e6, bytes / 1e6);
    let z: Vec<f32> = (0..HID).map(|_| (rng.next() >> 40) as f32 / (1u64 << 23) as f32 - 1.0).collect();
    for n in [1usize, 2, 4, 8, 16, 32, 64].into_iter().filter(|&n| n <= n_experts) {
        let (mut times, mut out) = (Vec::new(), vec![0f32; n * HID]);
        for rep in 0..reps + 5 {
            let ids = rng.pick(n_experts, n);
            let pairs: Vec<GlmPair> = ids.iter().map(|&e| GlmPair { expert: &experts[e], token: 0, weight: 0.5 }).collect();
            let t = Instant::now();
            experts_pinned(kernel, Z::F32(&z), &pairs, &mut out, threads, pin)?;
            if rep >= 5 {
                times.push(t.elapsed().as_secs_f64());
            }
        }
        let med = median(&mut times);
        println!("a call of {n:>2} pairs: {:>8.1} us ({:.1} us a pair), {:.1} GB/s of the experts' bytes, {:.0} G weight-ops/s", med * 1e6, med * 1e6 / n as f64, n as f64 * bytes / med / 1e9, n as f64 * 3.0 * (INTER * HID) as f64 / med / 1e9);
    }
    Ok(())
}

pub fn run(a: &Args) -> Result<()> {
    if a.has("cb2") {
        return run_cb2(a);
    }
    if a.has("glm") {
        return run_glm(a);
    }
    if a.has("hb") {
        return run_hb(a);
    }
    let bits: u32 = a.num("bits", 4)?;
    if !(4..=6).contains(&bits) {
        return err(format!("--bits {bits}: 4, 5 or 6"));
    }
    let kernel = match a.flag("kernel") {
        None | Some("auto") => Kernel::best(),
        Some(k) => Kernel::parse(k).filter(|k| k.runs_here()).ok_or_else(|| Error(format!("--kernel {k}: one of {}", Kernel::available().iter().map(|k| k.name()).collect::<Vec<_>>().join(", "))))?,
    };
    let threads = match a.num("threads", 0usize)? {
        0 => std::thread::available_parallelism().map_or(1, |n| n.get()),
        n => n,
    };
    let (n_experts, n_pairs, reps, tokens): (usize, usize, usize, usize) = (a.num("experts", 1024)?, a.num("pairs", 512)?, a.num("reps", 5)?, a.num("tokens", 20)?);
    let misses: Vec<usize> = a.flag("misses").unwrap_or("100,230,400").split(',').map(|m| m.trim().parse().map_err(|_| Error(format!("--misses {m}: not a number")))).collect::<Result<_>>()?;
    let mut rng = Rng(a.num("seed", 1u64)?.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
    if n_pairs > n_experts || n_pairs == 0 || reps == 0 || tokens == 0 || misses.iter().any(|&m| m == 0 || m > n_experts * LAYERS) {
        return err(format!("--pairs {n_pairs} is at most --experts {n_experts}; --reps and --tokens are at least 1; --misses {misses:?} are at most {} (a layer's experts times {LAYERS})", n_experts * LAYERS));
    }
    let (stride, row_bytes) = (row_stride(bits, HIDDEN) as f64, (row_stride(bits, HIDDEN) + HIDDEN / 32 * 2) as f64);
    println!("{} ({}), {threads} threads, kernel {} (this computer runs: {})", cpu_name(), std::env::consts::ARCH, kernel.name(), Kernel::available().iter().map(|k| k.name()).collect::<Vec<_>>().join(" "));
    println!("experts of {INTER} neurons x {HIDDEN} at {bits} bits: {:.2} MB of codes and scales a matrix, {:.2} MB an expert", INTER as f64 * row_bytes / 1e6, 2.0 * INTER as f64 * row_bytes / 1e6);
    println!("streaming read, {threads} threads: {:.1} GB/s", stream_read(threads, a.num("gib", 1.0f64)?, 3));

    let t = Instant::now();
    let pool: Vec<(Expert, usize)> = (0..n_experts)
        .map(|_| {
            let n = rng.active();
            expert(bits, n, &mut rng)
        })
        .collect::<Result<_>>()?;
    let mut on: Vec<usize> = pool.iter().map(|e| e.1).collect();
    on.sort_unstable();
    let off = |n: usize| 100.0 * (1.0 - n as f64 / INTER as f64);
    let mean = on.iter().sum::<usize>() as f64 / n_experts as f64;
    println!(
        "{n_experts} experts made in {:.1} s ({:.2} GB); neurons off after relu squared: {:.1}% on average (p10 {:.1}%, p90 {:.1}%; the probe's: 91.8%, 83.7%, 98.5%), {mean:.0} on a pair",
        t.elapsed().as_secs_f64(),
        n_experts as f64 * 2.0 * INTER as f64 * row_bytes / 1e9,
        off(mean as usize),
        off(on[n_experts * 9 / 10]),
        off(on[n_experts / 10])
    );
    alone(kernel, &pool[0].0, &mut rng);
    // a layer's z for each of the 40 layers; the bytes a pair reads: all of up and down's rows of the neurons on, as (codes, codes and scales)
    let zs: Vec<Vec<u16>> = (0..LAYERS).map(|_| z_row(&mut rng)).collect();
    let read = |ids: &[usize]| ids.iter().fold((0.0, 0.0), |(c, t), &e| (c + (INTER + pool[e].1) as f64 * stride, t + (INTER + pool[e].1) as f64 * row_bytes));

    // the kernel against the scalar one, bit for bit, on a few pairs, and the neurons it found against the ones made on
    let ids = rng.pick(n_experts, 4);
    let pairs = pairs_of(&pool, &ids, &mut rng);
    let (mut got, mut want) = (vec![0f32; ids.len() * HIDDEN], vec![0f32; ids.len() * HIDDEN]);
    let n_got = experts_with(kernel, Z::Bf16(&zs[0]), &pairs, &mut got, threads)?;
    let n_want = experts_with(Kernel::Scalar, Z::Bf16(&zs[0]), &pairs, &mut want, threads)?;
    let same = got.iter().zip(&want).all(|(x, y)| x.to_bits() == y.to_bits()) && n_got == n_want && n_got.iter().zip(&ids).all(|(&n, &e)| n as usize == pool[e].1);
    println!("check: the {} kernel against the scalar one on {} pairs: {}", kernel.name(), ids.len(), if same { "the same bits, the neurons on are those made on" } else { "DIFFERENT" });
    if !same {
        return err("the kernel's result is not the scalar kernel's");
    }

    // one call of many pairs
    let (mut secs, mut best, mut out) = (Vec::new(), (f64::MAX, 0.0, 0.0), vec![0f32; n_pairs * HIDDEN]);
    for rep in 0..=reps {
        let ids = rng.pick(n_experts, n_pairs);
        let pairs = pairs_of(&pool, &ids, &mut rng);
        let t = Instant::now();
        experts_with(kernel, Z::Bf16(&zs[0]), &pairs, &mut out, threads)?;
        let s = t.elapsed().as_secs_f64();
        if rep == 0 {
            continue; // (the first call wakes the workers and the cores)
        }
        secs.push(s);
        if s < best.0 {
            best = (s, read(&ids).0 / 1e9, read(&ids).1 / 1e9);
        }
    }
    let med = median(&mut secs);
    println!(
        "one call of {n_pairs} pairs: {:.1} ms best ({:.0} pairs/s, {:.1} GB/s of codes and scales read, {:.1} GB/s of codes alone), {:.1} ms median ({:.0} pairs/s)",
        best.0 * 1e3,
        n_pairs as f64 / best.0,
        best.2 / best.0,
        best.1 / best.0,
        med * 1e3,
        n_pairs as f64 / med
    );

    // a decode token: M misses spread over the layers, a call a layer
    for &m in &misses {
        let (mut ms, mut calls, mut gb) = (Vec::new(), Vec::new(), 0.0);
        for tok in 0..tokens + 3 {
            let (mut total, mut bytes) = (0.0, 0.0);
            for (l, z) in zs.iter().enumerate() {
                let n = m / LAYERS + usize::from(l < m % LAYERS);
                if n == 0 {
                    continue;
                }
                let ids = rng.pick(n_experts, n);
                let pairs = pairs_of(&pool, &ids, &mut rng);
                let mut out = vec![0f32; n * HIDDEN];
                let t = Instant::now();
                experts_with(kernel, Z::Bf16(z), &pairs, &mut out, threads)?;
                let s = t.elapsed().as_secs_f64();
                total += s;
                bytes += read(&ids).1 / 1e9;
                calls.push(s);
            }
            if tok >= 3 {
                // (the first three tokens are not timed: the cores are woken)
                ms.push(total * 1e3);
                gb = bytes;
            } else {
                calls.clear();
            }
        }
        let (best, med) = (ms.iter().cloned().fold(f64::MAX, f64::min), median(&mut ms));
        println!("a decode token with {m} missed experts ({}-{} a layer, {tokens} tokens): {med:.2} ms median, {best:.2} ms best; a layer's call {:.0} us median; {:.1} GB/s read at the median", m / LAYERS, m / LAYERS + usize::from(m % LAYERS > 0), median(&mut calls) * 1e6, gb / (med / 1e3));
    }
    Ok(())
}

/// One thread, one expert in the cache: the kernels' own speed, which the memory does not limit (the cores' share of a computer's reading is this rate times their number).
fn alone(kernel: Kernel, e: &Expert, rng: &mut Rng) {
    use crate::cpu_simd as simd;
    let z: Vec<f32> = (0..HIDDEN).map(|_| 0.5 + (rng.next() >> 40) as f32 / (1u64 << 24) as f32).collect();
    let (mut sums, mut acc) = (vec![0f32; INTER], vec![0f32; HIDDEN]);
    let active: Vec<(u32, f32)> = rng.pick(INTER, 220).into_iter().map(|j| (j as u32, 1000.0)).collect();
    let time = |f: &mut dyn FnMut()| {
        (0..3).map(|_| { let t = Instant::now(); (0..100).for_each(|_| f()); t.elapsed().as_secs_f64() / 100.0 }).fold(f64::MAX, f64::min)
    };
    let up = time(&mut || simd::up(kernel, &e.up.view(), &z, 0, INTER, &mut sums));
    let down = time(&mut || simd::down(kernel, &e.down.view(), &active, &mut acc));
    println!(
        "the {} kernel alone, one thread, an expert in the cache: up {:.0} us ({:.1} G weights/s, {:.1} GB/s of codes and scales), down for 220 neurons {:.0} us ({:.1} G weights/s)",
        kernel.name(),
        up * 1e6,
        (INTER * HIDDEN) as f64 / up / 1e9,
        (INTER * e.up.row_bytes()) as f64 / up / 1e9,
        down * 1e6,
        (220 * HIDDEN) as f64 / down / 1e9
    );
}
