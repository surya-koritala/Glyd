//! The skinny products of the four-, five- and six-bit block layouts (kernels/skinny.cu): y [M, N] = x [M, K] W^T for M of 1 to 16 rows (9 to 16: a second tile of columns on the same decoded weights), the always-on matrices of a decode step and of a verification of a few drafts. The
//! library's tensor-core products pad M and leave the memory half idle at these sizes; these read the layout's steps as they are, make the weights bf16 pairs in registers and multiply them on the tensor cores, and are
//! bound by the bytes. `SkinnyKern::gemv` takes a `Pack` of the Int4, Int5 or Int6 layout as the engine holds it (`Words::Tiered`, tiers[0] >> 24 the layout's number) and bf16 rows, and writes bf16 rows, as `Lib::linear`
//! does; the result is deterministic, and the same as the library's to the sums' rounding (both round each weight, (code - bias) times its scale, to bf16 once and sum in fp32: their errors against the exact product
//! are equal to the digit).
//!
//! `glyd-engine skinnybench [--bits 4,5,6] [--prefill 2048,8192] [--m 1,2,4,8] [--shapes all|NAME,...] [--mb N (256)]` (a development build's): the kernels against the CPU decode of a packed random matrix and against the library's product, then the
//! time of both at GLM-5.3-Flash's always-on shapes with the weights cycled through MB megabytes, so that they come from memory and not from the cache, and the bytes a second as a share of the card's 736 GB/s.

use crate::dev::{Dev, Kernel};
use crate::ffi::{err, Pack, Result, Stream, Words};
use crate::launch;

macro_rules! kname {
    ($n:literal) => {
        $n
    };
}

/// The layouts' numbers (tiers[0] >> 24; glyd-gpu's INT6_TAG, INT5_TAG and INT4_TAG).
const INT6: u32 = 4;
const INT5: u32 = 5;
const INT4: u32 = 6;
/// The exact four-bit layout (glyd-gpu's INT4H_TAG): Int4's steps and codes, a scale of its own form over the pack's base (tiers[1]).
const INT4H: u32 = 13;

pub struct SkinnyKern {
    // [Int4, Int5, Int6]
    x: [Kernel; 3],
    x4h: crate::kt::skinnyx4h,
}

impl SkinnyKern {
    pub fn load(dev: &Dev) -> Result<SkinnyKern> {
        Ok(SkinnyKern { x: [dev.kernel(kname!("skinnyx4"))?, dev.kernel(kname!("skinnyx5"))?, dev.kernel(kname!("skinnyx6"))?], x4h: dev.typed()? })
    }

    /// Whether `w` is a matrix these products take: the Int4, Int5 or Int6 layout (tiered words), rows a multiple of 64, columns of 32.
    pub fn takes(w: &Pack) -> bool {
        matches!(w.words, Words::Tiered { tiers, .. } if matches!(tiers[0] >> 24, INT4 | INT5 | INT6 | INT4H))
            && w.rows > 0
            && w.rows % 64 == 0
            && w.cols > 0
            && w.cols % 32 == 0
    }

    /// Where a block of 64 rows of `w` starts (as `gemv` reads it): its steps at data + r * .2 and its scale words at .1 + r * .3, for the L2 prefetch of a product to come.
    pub fn regions(w: &Pack) -> Option<(u64, u64, u64, u64)> {
        let Words::Tiered { blocks, tiers, .. } = w.words else { return None };
        if !SkinnyKern::takes(w) {
            return None;
        }
        let sb = match tiers[0] >> 24 {
            INT5 => 640,
            INT6 => 768,
            _ => 512,
        };
        let ks = (w.cols / 16) as u64;
        Some((w.data, blocks, ks * sb, ks / 2 * 128))
    }

    /// y [m, rows] (bf16) = x [m, cols] (bf16) W^T for `w` in the Int4, Int5 or Int6 layout, `m` from 1 to 8, on `s`: the same bf16 weights and fp32 sums as `Lib::linear` makes (the same error against the exact product), from
    /// 95 to 101% of the card's bandwidth on the matrices of 18 MB and more at any `m`, against the library's 80 to 89%; also faster than the library on the small ones (a 5 MB matrix 0 to 40%).
    ///
    /// # Safety
    /// Live device buffers: `w`'s, x of m x cols bf16 words (4-byte aligned), y of m x rows.
    pub unsafe fn gemv(&self, s: Stream, w: &Pack, x: u64, y: u64, m: usize) -> Result<()> {
        let Words::Tiered { blocks, tiers, .. } = w.words else { return err("the skinny products take the block words of the Int4, Int5 and Int6 layouts") };
        if !SkinnyKern::takes(w) || !(1..=16).contains(&m) || x % 4 != 0 {
            return err(format!("a skinny product of {m} rows over {} x {} of layout {}: the Int4, Int5 or Int6 layout (rows in 64s, columns in 32s), 1 to 16 rows and 4-byte aligned x are wanted", w.rows, w.cols, tiers[0] >> 24));
        }
        if tiers[0] >> 24 == INT4H {
            return launch!(self.x4h, [(w.rows / 64) as u32, 1, 1], [256, 1, 1], s; w.data, blocks, x, y, w.rows as i32, w.cols as i32, m as i32, tiers[1]);
        }
        let b = match tiers[0] >> 24 {
            INT4 => 0,
            INT5 => 1,
            _ => 2,
        };
        launch!(self.x[b], [(w.rows / 64) as u32, 1, 1], [256, 1, 1], s; w.data, blocks, x, y, w.rows as i32, w.cols as i32, m as i32)
    }

    /// `gemv`'s rows r0 to r1 alone (multiples of 64; a 64-row block's bytes are its own, in order), into their places of y [m, rows].
    ///
    /// # Safety
    /// As `gemv`.
    pub unsafe fn gemv_rows(&self, s: Stream, w: &Pack, x: u64, y: u64, m: usize, r0: usize, r1: usize) -> Result<()> {
        let Words::Tiered { blocks, tiers, .. } = w.words else { return err("the skinny products take the block words of the Int4, Int5 and Int6 layouts") };
        if !SkinnyKern::takes(w) || !(1..=16).contains(&m) || x % 4 != 0 || r0 % 64 != 0 || r1 % 64 != 0 || r0 >= r1 || r1 > w.rows as usize {
            return err(format!("a skinny product of rows {r0} to {r1} of {} x {}, {m} rows of x", w.rows, w.cols));
        }
        let (b, sb) = match tiers[0] >> 24 {
            INT4 | INT4H => (0, 512u64),
            INT5 => (1, 640),
            _ => (2, 768),
        };
        // (kernels/skinny.cu: a block of 64 rows is ks steps of SB bytes and ks / 2 words of 128 bytes)
        let (ks, b0) = ((w.cols / 16) as u64, (r0 / 64) as u64);
        if tiers[0] >> 24 == INT4H {
            return launch!(self.x4h, [((r1 - r0) / 64) as u32, 1, 1], [256, 1, 1], s; w.data + b0 * ks * sb, blocks + b0 * (ks / 2) * 128, x, y + 2 * r0 as u64, w.rows as i32, w.cols as i32, m as i32, tiers[1]);
        }
        launch!(self.x[b], [((r1 - r0) / 64) as u32, 1, 1], [256, 1, 1], s; w.data + b0 * ks * sb, blocks + b0 * (ks / 2) * 128, x, y + 2 * r0 as u64, w.rows as i32, w.cols as i32, m as i32)
    }
}

#[cfg(feature = "dev")]
pub use bench::run;

#[cfg(feature = "dev")]
mod bench {
    use super::*;
    use crate::args::Args;
    use crate::dev::{Buf, Timer};
    use crate::ffi::{Error, Lib};
    use glyd_gpu::pack::{pack_int4, pack_int5, pack_int6, unpack_int4, unpack_int5, unpack_int6, Tiered};

    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }

        fn f(&mut self, lo: f32, hi: f32) -> f32 {
            lo + (self.next() >> 40) as f32 / (1u64 << 24) as f32 * (hi - lo)
        }
    }

    fn bf16(v: f32) -> u16 {
        let b = v.to_bits();
        ((b.wrapping_add(0x7FFF + ((b >> 16) & 1))) >> 16) as u16
    }

    fn f32_of(b: u16) -> f32 {
        f32::from_bits((b as u32) << 16)
    }

    /// GLM-5.3-Flash's always-on matrices (config.json: hidden 4096, 64 heads; KDA heads of 128; MLA q_lora 1536, kv_lora 512, q/v heads of 256; dense intermediate 12288, shared expert 2048, vocabulary 154,880), as (name, rows, columns).
    const SHAPES: &[(&str, usize, usize)] = &[
        ("kda q/k/v", 8192, 4096),
        ("kda o", 4096, 8192),
        ("mla q_a", 1536, 4096),
        ("mla q_b", 16384, 1536),
        ("mla kv_a", 512, 4096),
        ("mla kv_b", 32768, 512),
        ("mla o", 4096, 16384),
        ("dense gate/up", 12288, 4096),
        ("dense down", 4096, 12288),
        ("shared gate/up", 2048, 4096),
        ("shared down", 4096, 2048),
        ("head", 154880, 4096),
        ("qx gdn qkvz", 16384, 2560),
        ("qx gdn out", 2560, 6144),
        ("qx attn qkv", 13312, 2560),
        ("qx attn o", 2560, 6144),
        ("qx head", 248320, 2560),
    ];

    /// A pack on the device: the bytes of `t`, or of random codes under valid scales of the shape (rows, cols) at `bits`.
    #[allow(dead_code)] // (the buffers are held for the pack's addresses)
    struct Dev1 {
        data: Buf,
        blocks: Buf,
        base: Buf,
        pack: Pack,
    }

    fn upload(t: &Tiered) -> Result<Dev1> {
        let (data, blocks, base) = (Buf::new(t.data.len())?, Buf::new(t.blocks.len())?, Buf::new(t.block_base.len() * 4)?);
        data.write(0, &t.data)?;
        blocks.write(0, &t.blocks)?;
        base.write(0, &t.block_base)?;
        let pack = Pack {
            data: data.ptr,
            words: Words::Tiered { blocks: blocks.ptr, block_base: base.ptr, tiers: t.tiers },
            rows: t.rows as i64,
            cols: t.cols as i64,
        };
        Ok(Dev1 { data, blocks, base, pack })
    }

    /// Random bytes in the layout's shape: codes anywhere, scales bf16 around 0.01 (the codes of an Int4 or Int5 pack are 1 to 15 or 31, but the products do not care which).
    fn random_pack(bits: u32, rows: usize, cols: usize, r: &mut Rng) -> Tiered {
        let steps = rows / 64 * (cols / 16);
        let step = match bits {
            4 => 512,
            5 => 640,
            _ => 768,
        };
        let mut data = vec![0u8; steps * step];
        data.chunks_mut(8).for_each(|c| c.copy_from_slice(&r.next().to_le_bytes()[..c.len()]));
        let blocks: Vec<u8> = (0..steps / 2 * 64).flat_map(|_| bf16(r.f(0.005, 0.02)).to_le_bytes()).collect();
        Tiered {
            rows,
            cols,
            data,
            blocks,
            block_base: vec![0; steps + 1],
            tiers: [
                (match bits {
                    4 => INT4,
                    5 => INT5,
                    _ => INT6,
                }) << 24,
                0,
                0,
            ],
        }
    }

    fn check(dev: &Dev, lib: &Lib, kern: &SkinnyKern) -> Result<()> {
        let mut r = Rng(0x1234_5678_9ABC_DEF1);
        for bits in [4u32, 5, 6] {
            let (rows, cols) = (192usize, 1056usize);
            let w: Vec<u16> = (0..rows * cols).map(|_| bf16(r.f(-0.08, 0.08))).collect();
            let t = (match bits {
                4 => pack_int4(&w, rows, cols),
                5 => pack_int5(&w, rows, cols),
                _ => pack_int6(&w, rows, cols),
            })
            .ok_or_else(|| Error("a random matrix that does not pack".into()))?;
            let mut hat = vec![0u16; rows * cols];
            assert!(match bits {
                4 => unpack_int4(&t, &mut hat),
                5 => unpack_int5(&t, &mut hat),
                _ => unpack_int6(&t, &mut hat),
            });
            let d = upload(&t)?;
            for m in [1usize, 2, 3, 4, 5, 8, 9, 16] {
                let x: Vec<u16> = (0..m * cols).map(|_| bf16(r.f(-1.0, 1.0))).collect();
                let xb = Buf::new(x.len() * 2)?;
                xb.write(0, &x)?;
                let y_lib = Buf::zeroed(m * rows * 2)?;
                let ws = Buf::zeroed(lib.linear_workspace(false, rows as i64, cols as i64, m as i64)?.max(16))?;
                let done = Buf::zeroed(1 << 20)?;
                // SAFETY: live device buffers of the sizes the products read and write.
                unsafe { lib.linear(&d.pack, xb.ptr, m as i64, y_lib.ptr, ws.ptr, ws.bytes, done.ptr, dev.stream)? };
                dev.sync()?;
                let mut b = vec![0u16; m * rows];
                y_lib.read(0, &mut b)?;
                {
                    let y_mine = Buf::zeroed(m * rows * 2)?;
                    // SAFETY: live device buffers of the sizes the products read and write.
                    unsafe { kern.gemv(dev.stream, &d.pack, xb.ptr, y_mine.ptr, m)? };
                    dev.sync()?;
                    let mut a = vec![0u16; m * rows];
                    y_mine.read(0, &mut a)?;
                    let (mut e_ref, mut e_lib, mut n2) = (0f64, 0f64, 0f64);
                    for i in 0..m {
                        for o in 0..rows {
                            let want: f64 = (0..cols).map(|c| f32_of(hat[o * cols + c]) as f64 * f32_of(x[i * cols + c]) as f64).sum();
                            let (got, lb) = (f32_of(a[i * rows + o]) as f64, f32_of(b[i * rows + o]) as f64);
                            e_ref += (got - want) * (got - want);
                            e_lib += (lb - want) * (lb - want);
                            n2 += want * want;
                        }
                    }
                    let (rel_mine, rel_lib) = ((e_ref / n2).sqrt(), (e_lib / n2).sqrt());
                    let ok = rel_mine < 4e-3;
                    println!(
                        "{} int{bits} {m} row{}: {rows} x {cols}: the skinny product {rel_mine:.2e} (the library's {rel_lib:.2e}) off the f64 product of the decoded matrix, in the norm of it",
                        if ok { "ok  " } else { "FAIL" },
                        if m == 1 { "" } else { "s" }
                    );
                    if !ok {
                        return err("a skinny product is not within its bound");
                    }
                }
            }
        }
        Ok(())
    }

    pub fn run(a: &Args) -> Result<()> {
        let dev = Dev::new(0)?;
        let lib = Lib::find()?;
        let kern = SkinnyKern::load(&dev)?;
        println!("{} (sm_{}{}), {}: the skinny products of the Int4 and Int5 layouts", dev.name, dev.cc.0, dev.cc.1, lib.path);
        check(&dev, &lib, &kern)?;
        if a.has("no-speed") {
            return Ok(());
        }
        if let Some(spec) = a.flag("prefill") {
            // --prefill M,M,...: the library's product for a prompt chunk of M rows on each shape (TFLOPS against the card's 104 of bf16 with fp32 sums)
            let mlist: Vec<usize> =
                spec.split(',').map(|x| x.trim().parse().map_err(|_| Error(format!("--prefill {x}: not a number")))).collect::<Result<_>>()?;
            let bits: Vec<u32> = a.flag("bits").unwrap_or("5").split(',').map(|x| x.trim().parse().unwrap_or(5)).collect();
            let want: Vec<&str> = a.flag("shapes").filter(|s| *s != "all").map(|s| s.split(',').collect()).unwrap_or_default();
            let mut r = Rng(0xFEED_0000_0000_0002);
            let t = Timer::new()?;
            println!("{:<15} {:>7} {:>7} {:>3} {:>6} | {:>9} {:>8} {:>6}", "matrix", "rows", "cols", "bits", "M", "lib us", "TFLOPS", "of 104");
            for &(name, rows, cols) in SHAPES.iter().filter(|s| want.is_empty() || want.iter().any(|w| s.0.starts_with(w))) {
                for &b in &bits {
                    let d = upload(&random_pack(b, rows, cols, &mut r))?;
                    for &m in &mlist {
                        let xs: Vec<u16> = (0..m * cols).map(|_| bf16(r.f(-1.0, 1.0))).collect();
                        let x = Buf::new(xs.len() * 2)?;
                        x.write(0, &xs)?;
                        let y = Buf::zeroed(m * rows * 2)?;
                        let ws = Buf::zeroed(lib.linear_workspace(false, rows as i64, cols as i64, m as i64)?.max(16))?;
                        let done = Buf::zeroed(1 << 20)?;
                        // SAFETY: live device buffers of the sizes the product reads and writes.
                        let once = || unsafe { lib.linear(&d.pack, x.ptr, m as i64, y.ptr, ws.ptr, ws.bytes, done.ptr, dev.stream) };
                        for _ in 0..2 {
                            once()?;
                        }
                        dev.sync()?;
                        let reps = if m >= 2048 { 6 } else { 20 };
                        t.start(dev.stream)?;
                        for _ in 0..reps {
                            once()?;
                        }
                        let us = t.stop(dev.stream)? as f64 * 1e3 / reps as f64;
                        let tf = 2.0 * m as f64 * rows as f64 * cols as f64 / us / 1e6;
                        println!("{name:<15} {rows:>7} {cols:>7} {b:>3} {m:>6} | {us:>9.1} {tf:>8.1} {:>5.0}%", 100.0 * tf / 104.0);
                    }
                }
            }
            return Ok(());
        }
        let list = |name: &str, default: &str| -> Result<Vec<usize>> {
            a.flag(name).unwrap_or(default).split(',').map(|x| x.trim().parse().map_err(|_| Error(format!("--{name} {x}: not a number")))).collect()
        };
        let (bits, ms) = (list("bits", "4,5,6")?, list("m", "1,2,4,8")?);
        let want: Vec<&str> = a.flag("shapes").filter(|s| *s != "all").map(|s| s.split(',').collect()).unwrap_or_default();
        let cycle = a.num("mb", 256usize)? << 20;
        let mut r = Rng(0xFEED_0000_0000_0001);
        let t = Timer::new()?;
        println!(
            "{:<15} {:>7} {:>7} {:>5} {:>3} | {:>9} {:>9} {:>6} | {:>9} {:>9} {:>6} | {:>6}",
            "matrix", "rows", "cols", "MB", "M", "mine us", "GB/s", "of 736", "lib us", "GB/s", "of 736", "x"
        );
        for &(name, rows, cols) in SHAPES.iter().filter(|s| want.is_empty() || want.iter().any(|w| s.0.starts_with(w))) {
            for &b in &bits {
                let tier = random_pack(b as u32, rows, cols, &mut r);
                let bytes = tier.data.len() + tier.blocks.len();
                // the copies the runs cycle through: enough bytes that a run reads memory, not the L2
                let copies = (cycle / bytes).clamp(1, 64);
                let packs: Vec<Dev1> = (0..copies).map(|_| upload(&tier)).collect::<Result<_>>()?;
                for &m in &ms {
                    let xs: Vec<u16> = (0..m * cols).map(|_| bf16(r.f(-1.0, 1.0))).collect();
                    let x = Buf::new(xs.len() * 2)?;
                    x.write(0, &xs)?;
                    let (y, y2) = (Buf::zeroed(m * rows * 2)?, Buf::zeroed(m * rows * 2)?);
                    let ws = Buf::zeroed(lib.linear_workspace(false, rows as i64, cols as i64, m as i64)?.max(16))?;
                    let done = Buf::zeroed(1 << 20)?;
                    let reps = (copies * 4).max(40);
                    let time = |mine: bool| -> Result<f64> {
                        let once = |i: usize| -> Result<()> {
                            let p = &packs[i % copies].pack;
                            // SAFETY: live device buffers of the sizes the products read and write.
                            unsafe {
                                if mine {
                                    kern.gemv(dev.stream, p, x.ptr, y.ptr, m)
                                } else {
                                    lib.linear(p, x.ptr, m as i64, y2.ptr, ws.ptr, ws.bytes, done.ptr, dev.stream)
                                }
                            }
                        };
                        for i in 0..copies.min(4) {
                            once(i)?;
                        }
                        dev.sync()?;
                        t.start(dev.stream)?;
                        for i in 0..reps {
                            once(i)?;
                        }
                        Ok(t.stop(dev.stream)? as f64 * 1e3 / reps as f64)
                    };
                    let (us_mine, us_lib) = (time(true)?, time(false)?);
                    let gbs = |us: f64| bytes as f64 / us / 1e3;
                    println!(
                        "{name:<15} {rows:>7} {cols:>7} {:>5.1} {m:>3} | {us_mine:>9.1} {:>9.1} {:>5.0}% | {us_lib:>9.1} {:>9.1} {:>5.0}% | {:>5.2}x  (int{b})",
                        bytes as f64 / 1e6,
                        gbs(us_mine),
                        100.0 * gbs(us_mine) / 736.0,
                        gbs(us_lib),
                        100.0 * gbs(us_lib) / 736.0,
                        us_lib / us_mine
                    );
                }
            }
        }
        Ok(())
    }
}
