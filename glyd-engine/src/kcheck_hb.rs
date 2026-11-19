//! `glyd-engine hbcheck [CASE...] [--speed] [--no-check]`: the Hummingbird layer's kernels on the GPU (kernels/moe_hb.cu through src/moe_hb.rs: the rotation, the first product over codebook rows with relu squared, the sparse second
//! product, the weighted sum and the rotation back) against the pair computed in float64 from the weights as the packs decode them, on random experts of 3 and 2 bits at small shapes and at the real one (2,688 neuron rows over a
//! latent of 1,024), a decode step's tokens and a prompt's. Each line is a case: the output's error against the reference (relative in L2 for each token, bound 6e-3: the sums are fp32 in another order, the output is bf16, a rounding
//! of the first product's sum that falls the other way moves a neuron's activation by a bf16 place), the activations of the first product (the share more than a bf16 place off), that a second launch gives the same bits, and that
//! the same experts in other places of the stack (the expert cache's) give the same bits; the router's choices sent to a group's numbers by `hb_group_ids` are held to the host's. `--speed`: the time of the layer's launches at the
//! real shape for 1 token (a decode step: 22 experts of 512) and 2,048 (a prompt chunk), on experts of arbitrary bytes whose first rows are mostly dead (92% of the neurons zero, as relu squared leaves them). Exit status 1 where a bound is not kept.

use crate::args::Args;
use crate::dev::{Buf, Dev, Timer};
use crate::ffi::{err, Error, Lib, Result};
use crate::moe_hb::{HbHost, HbKern, HbPlace, HbStack};
use glyd_gpu::codebook::{decode_codebook_rows, pack_codebook_rows, CbRows};

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    /// Uniform in [lo, hi).
    fn f(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * ((self.next() >> 40) as f32 / (1u64 << 24) as f32)
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

fn bf16_bits(x: f32) -> u16 {
    ((x.to_bits() + 0x7FFF + ((x.to_bits() >> 16) & 1)) >> 16) as u16
}

fn bf16_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

/// Places of bf16 between two values.
fn places(a: u16, b: u16) -> i32 {
    let key = |u: u16| if u & 0x8000 != 0 { -((u & 0x7FFF) as i32) } else { u as i32 };
    (key(a) - key(b)).abs()
}

/// A case: `experts` experts of `bits` bits, `inter` neuron rows over a latent of `cols`, `tokens` tokens each choosing `k` experts.
struct Case {
    name: &'static str,
    bits: u32,
    experts: usize,
    inter: usize,
    cols: usize,
    tokens: usize,
    k: usize,
}

fn cases() -> Vec<Case> {
    let c = |name, bits, experts, inter, cols, tokens, k| Case { name, bits, experts, inter, cols, tokens, k };
    vec![
        c("tiny3", 3, 16, 128, 128, 3, 4),
        c("tiny2", 2, 16, 128, 128, 5, 4),
        c("mid3", 3, 24, 320, 512, 4, 6),
        c("mid2", 2, 24, 96, 256, 7, 5),
        c("super3", 3, 30, 2688, 1024, 1, 22),
        c("super2", 2, 30, 2688, 1024, 2, 22),
        c("prompt3", 3, 30, 2688, 1024, 24, 22),
    ]
}

/// An expert's two matrices of random codes, scales (of both signs, some zero: a zero scale is a dead neuron's run) and ascending codebooks.
fn expert(r: &mut Rng, bits: u32, inter: usize, cols: usize) -> Result<(CbRows, CbRows)> {
    let mut one = || -> Result<CbRows> {
        let codes: Vec<u8> = (0..inter * cols).map(|_| r.below(1 << bits) as u8).collect();
        let scales: Vec<u16> = (0..inter * cols / 32).map(|_| if r.below(9) == 0 { 0 } else { bf16_bits(r.f(-0.01, 0.05)) }).collect();
        let mut book: Vec<u16> = (0..inter << bits).map(|_| bf16_bits(r.f(-1.0, 1.0))).collect();
        book.chunks_mut(1 << bits).for_each(|e| e.sort_by(|a, b| bf16_f32(*a).total_cmp(&bf16_f32(*b))));
        pack_codebook_rows(&codes, &scales, &book, bits, inter, cols).ok_or_else(|| Error("a random matrix that does not pack".into()))
    };
    Ok((one()?, one()?))
}

/// The natural-order Hadamard transform of `v` (a power of two long), unnormalized.
fn hadamard(v: &mut [f64]) {
    let mut h = 1;
    while h < v.len() {
        for i in (0..v.len()).step_by(2 * h) {
            for j in i..i + h {
                (v[j], v[j + h]) = (v[j] + v[j + h], v[j] - v[j + h]);
            }
        }
        h *= 2;
    }
}

/// What the reference computes of a case: the output y [tokens, cols] in f64, and the activations of each pair [pairs][inter] (the first product's sum rounded to bf16, relu squared rounded to bf16 once).
struct Reference {
    y: Vec<f64>,
    acts: Vec<Vec<u16>>,
}

fn reference(c: &Case, pairs: &[(CbRows, CbRows)], z: &[u16], signs: &[i8], ids: &[i64], wts: &[f32]) -> Reference {
    let (n, norm) = (c.cols, 1.0 / (c.cols as f64).sqrt());
    let mut zr = vec![vec![0f64; n]; c.tokens];
    for (t, row) in zr.iter_mut().enumerate() {
        for j in 0..n {
            row[j] = signs[j] as f64 * bf16_f32(z[t * n + j]) as f64;
        }
        hadamard(row);
        row.iter_mut().for_each(|x| *x *= norm);
    }
    let mut yp = vec![vec![0f64; n]; c.tokens];
    let mut acts = vec![Vec::new(); ids.len()];
    let (mut up, mut down) = (vec![0f32; c.inter * n], vec![0f32; c.inter * n]);
    for e in 0..c.experts {
        let mine: Vec<usize> = (0..ids.len()).filter(|&j| ids[j] == e as i64).collect();
        if mine.is_empty() {
            continue;
        }
        assert!(decode_codebook_rows(&pairs[e].0, &mut up) && decode_codebook_rows(&pairs[e].1, &mut down));
        for j in mine {
            let t = j / c.k;
            let mut act = vec![0u16; c.inter];
            for (row, a) in act.iter_mut().enumerate() {
                let u: f64 = (0..n).map(|i| up[row * n + i] as f64 * zr[t][i]).sum();
                let h = bf16_f32(bf16_bits(u as f32));
                *a = if h > 0.0 { bf16_bits(h * h) } else { 0 };
                if bf16_f32(*a) != 0.0 {
                    for i in 0..n {
                        yp[t][i] += wts[j] as f64 * bf16_f32(*a) as f64 * down[row * n + i] as f64;
                    }
                }
            }
            acts[j] = act;
        }
    }
    let mut y = Vec::with_capacity(c.tokens * n);
    for row in yp.iter_mut() {
        hadamard(row);
        y.extend(row.iter().enumerate().map(|(i, x)| signs[i] as f64 * x * norm));
    }
    Reference { y, acts }
}

/// A run of the layer's launches on the GPU for a stack, the choices `via` (each pair's expert's place in the stack) and `ids` (the router's): the output, the plan and the activations in its order.
struct Out {
    y: Vec<u16>,
    plan: Vec<i32>,
    a: Vec<u16>,
}

#[allow(clippy::too_many_arguments)]
unsafe fn launches(dev: &Dev, lib: &Lib, kern: &HbKern, c: &Case, st: &HbStack, z: &Buf, signs: &Buf, ids: &Buf, via: &Buf, wts: &Buf) -> Result<Out> {
    let (s, pairs) = (dev.stream, c.tokens * c.k);
    let split = kern.split(pairs);
    let (zr, plan) = (Buf::zeroed(c.tokens * c.cols * 4)?, Buf::zeroed((2 + 2 * c.experts + pairs) * 4)?);
    let (a, part, y) = (Buf::zeroed(pairs * c.inter * 2)?, Buf::zeroed(pairs * split * c.cols * 4)?, Buf::zeroed(c.tokens * c.cols * 2)?);
    kern.rot_in(s, z.ptr, c.cols, signs.ptr, zr.ptr, c.tokens, c.cols)?;
    lib.moe_way(via.ptr, pairs as i64, c.experts as i64, plan.ptr, s)?;
    kern.up(s, c.bits, st, zr.ptr, plan.ptr, via.ptr, a.ptr, c.experts, c.inter, c.cols, c.k, pairs)?;
    kern.down(s, c.bits, st, a.ptr, plan.ptr, via.ptr, part.ptr, c.experts, c.inter, c.cols, split, pairs)?;
    kern.sum_rot(s, part.ptr, wts.ptr, ids.ptr, signs.ptr, y.ptr, 0, c.tokens, c.experts, c.k, c.cols, split)?;
    dev.sync()?;
    let (mut yh, mut ph, mut ah) = (vec![0u16; c.tokens * c.cols], vec![0i32; 2 + 2 * c.experts + pairs], vec![0u16; pairs * c.inter]);
    y.read(0, &mut yh)?;
    plan.read(0, &mut ph)?;
    a.read(0, &mut ah)?;
    Ok(Out { y: yh, plan: ph, a: ah })
}

fn upload<T: Copy>(v: &[T]) -> Result<Buf> {
    let b = Buf::new(std::mem::size_of_val(v))?;
    b.write(0, v)?;
    Ok(b)
}

/// One case: the stack of its experts against the reference, twice, and with the experts in other places.
fn run_case(dev: &Dev, lib: &Lib, kern: &HbKern, c: &Case, failed: &mut Vec<String>) -> Result<()> {
    let mut r = Rng(0x5EED_0000 ^ (c.experts * 131 + c.inter * 7 + c.bits as usize) as u64);
    let pairs: Vec<(CbRows, CbRows)> = (0..c.experts).map(|_| expert(&mut r, c.bits, c.inter, c.cols)).collect::<Result<_>>()?;
    let place = HbPlace::new(c.bits, c.inter, c.cols)?;
    let st = HbStack::upload(&HbHost::of(place, &pairs)?)?;
    let signs: Vec<i8> = (0..c.cols).map(|_| if r.below(2) == 0 { 1 } else { -1 }).collect();
    let z: Vec<u16> = (0..c.tokens * c.cols).map(|_| bf16_bits(r.f(-3.0, 3.0))).collect();
    let mut ids = Vec::new();
    for _ in 0..c.tokens {
        let mut pool: Vec<usize> = (0..c.experts).collect();
        for j in 0..c.k {
            let at = j + r.below(pool.len() - j);
            pool.swap(j, at);
            ids.push(pool[j] as i64);
        }
    }
    let wts: Vec<f32> = (0..ids.len()).map(|_| r.f(0.2, 2.5)).collect();
    let want = reference(c, &pairs, &z, &signs, &ids, &wts);
    let (zb, sb, ib, wb) = (upload(&z)?, upload(&signs)?, upload(&ids)?, upload(&wts)?);
    // SAFETY: live device buffers of the sizes the kernels read and write.
    let got = unsafe { launches(dev, lib, kern, c, &st, &zb, &sb, &ib, &ib, &wb)? };
    // the output against the reference, token by token
    let mut worst = 0f64;
    for t in 0..c.tokens {
        let (mut e2, mut n2) = (0f64, 0f64);
        for i in 0..c.cols {
            let (g, w) = (bf16_f32(got.y[t * c.cols + i]) as f64, want.y[t * c.cols + i]);
            e2 += (g - w) * (g - w);
            n2 += w * w;
        }
        worst = worst.max((e2 / n2.max(1e-30)).sqrt());
    }
    // the activations (rows in the plan's order) against the reference's
    let hits = got.plan[0] as usize;
    let total = got.plan[1 + c.experts + hits] as usize;
    let (mut off, mut count, mut zeros) = (0usize, 0usize, 0usize);
    for i in 0..total {
        let j = got.plan[2 + 2 * c.experts + i] as usize;
        for n in 0..c.inter {
            let (g, w) = (got.a[i * c.inter + n], want.acts[j][n]);
            count += 1;
            off += (places(g, w) > 1) as usize;
            zeros += ((g & 0x7FFF == 0) != (w & 0x7FFF == 0)) as usize;
        }
    }
    let active: usize = want.acts.iter().map(|a| a.iter().filter(|&&x| x != 0).count()).sum();
    // the same call again, and the experts in other places of the stack (a reversed order: the pair's place is its expert's from the end)
    // SAFETY: as above.
    let again = unsafe { launches(dev, lib, kern, c, &st, &zb, &sb, &ib, &ib, &wb)? };
    let order: Vec<usize> = (0..c.experts).rev().collect();
    let moved: Vec<(CbRows, CbRows)> = order.iter().map(|&e| pairs[e].clone()).collect();
    let st2 = HbStack::upload(&HbHost::of(place, &moved)?)?;
    let via: Vec<i64> = ids.iter().map(|&e| (c.experts as i64 - 1) - e).collect();
    let vb = upload(&via)?;
    // SAFETY: as above.
    let other = unsafe { launches(dev, lib, kern, c, &st2, &zb, &sb, &ib, &vb, &wb)? };
    let (same, place_free) = (again.y == got.y && again.a == got.a, other.y == got.y);
    let ok = worst < 6e-3 && off * 100 <= count && zeros * 1000 <= count && same && place_free;
    println!(
        "{} {:8} {} bits, {} experts of {} x {}, {} tokens of {}: output {worst:.2e} of the reference's norm (bound 6e-3), activations {:.3}% more than a place off and {:.3}% on/off differing of {count} ({active} on), again {}, other places {}",
        if ok { "ok  " } else { "FAIL" },
        c.name,
        c.bits,
        c.experts,
        c.inter,
        c.cols,
        c.tokens,
        c.k,
        100.0 * off as f64 / count.max(1) as f64,
        100.0 * zeros as f64 / count.max(1) as f64,
        if same { "the same bits" } else { "DIFFERENT" },
        if place_free { "the same bits" } else { "DIFFERENT" }
    );
    if !ok {
        failed.push(c.name.to_string());
    }
    Ok(())
}

/// `hb_group_ids` against the host's: experts of two groups, choices past the layer's experts and below zero.
fn group_ids_case(dev: &Dev, kern: &HbKern, failed: &mut Vec<String>) -> Result<()> {
    let mut r = Rng(77);
    let experts = 100usize;
    let group_of: Vec<u8> = (0..experts).map(|_| r.below(2) as u8).collect();
    let mut seen = [0u16; 2];
    let local_of: Vec<u16> = group_of
        .iter()
        .map(|&g| {
            seen[g as usize] += 1;
            seen[g as usize] - 1
        })
        .collect();
    let ids: Vec<i64> = (0..1000).map(|i| if i % 97 == 0 { -1 } else if i % 89 == 0 { experts as i64 + 3 } else { r.below(experts) as i64 }).collect();
    let (ib, gb, lb) = (upload(&ids)?, upload(&group_of)?, upload(&local_of.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>())?);
    let mut ok = true;
    for g in 0..2usize {
        let out = Buf::zeroed(ids.len() * 8)?;
        // SAFETY: live device buffers of the sizes the kernel reads and writes.
        unsafe { kern.group_ids(dev.stream, ib.ptr, gb.ptr, lb.ptr, g, out.ptr, ids.len(), experts)? };
        dev.sync()?;
        let mut got = vec![0i64; ids.len()];
        out.read(0, &mut got)?;
        let want: Vec<i64> = ids.iter().map(|&e| if e >= 0 && (e as usize) < experts && group_of[e as usize] as usize == g { local_of[e as usize] as i64 } else { -1 }).collect();
        ok &= got == want;
    }
    println!("{} hb_group_ids: 1,000 choices (some below zero and past the experts) sent to the numbers of two groups", if ok { "ok  " } else { "FAIL" });
    if !ok {
        failed.push("hb_group_ids".into());
    }
    Ok(())
}

/// The time of the layer's launches at the real shape: `tokens` tokens of 22 experts of `experts`, on a stack of arbitrary bytes whose up matrix has 92% of its neurons' scales zero (dead).
fn speed(dev: &Dev, lib: &Lib, kern: &HbKern, bits: u32, tokens: usize, experts: usize) -> Result<()> {
    let c = Case { name: "real", bits, experts, inter: 2688, cols: 1024, tokens, k: 22 };
    let place = HbPlace::new(bits, c.inter, c.cols)?;
    let mut r = Rng(99 + bits as u64);
    let (mut data, mut meta) = (vec![0u8; experts * place.data], vec![0u8; experts * place.meta]);
    data.chunks_mut(8).for_each(|b| b.copy_from_slice(&r.next().to_le_bytes()[..b.len()]));
    for e in 0..experts {
        let m = &mut meta[e * place.meta..(e + 1) * place.meta];
        for (k, w) in m.chunks_exact_mut(2).enumerate() {
            // (the up matrix's scales are the first 64 bytes a neuron of the expert's meta: 92% of the neurons have theirs zero, so that they stay off)
            let dead = k * 2 < place.up_book && (k * 2 / 64 * 37) % 100 < 92;
            let v = if dead { 0 } else { bf16_bits(r.f(0.001, 0.05)) };
            w.copy_from_slice(&v.to_le_bytes());
        }
    }
    let st = HbStack::upload(&HbHost { place, experts, data, meta })?;
    let pairs = tokens * c.k;
    let ids: Vec<i64> = (0..pairs).map(|j| ((j % c.k) * 7 + (j / c.k) * 3 + j % 11) as i64 % experts as i64).collect();
    let z: Vec<u16> = (0..tokens * c.cols).map(|_| bf16_bits(r.f(-2.0, 2.0))).collect();
    let signs: Vec<i8> = (0..c.cols).map(|i| if i % 3 == 0 { -1 } else { 1 }).collect();
    let wts: Vec<f32> = vec![0.5; pairs];
    let (zb, sb, ib, wb) = (upload(&z)?, upload(&signs)?, upload(&ids)?, upload(&wts)?);
    let t = Timer::new()?;
    let reps = if tokens > 64 { 3 } else { 20 };
    // SAFETY: live device buffers of the sizes the kernels read and write.
    unsafe { launches(dev, lib, kern, &c, &st, &zb, &sb, &ib, &ib, &wb)? };
    t.start(dev.stream)?;
    for _ in 0..reps {
        // SAFETY: as above.
        unsafe { launches(dev, lib, kern, &c, &st, &zb, &sb, &ib, &ib, &wb)? };
    }
    let ms = t.stop(dev.stream)? as f64 / reps as f64;
    println!("speed {bits} bits, {tokens} tokens of 22 of {experts} experts: {ms:.3} ms a layer (with the buffers made and read each time), {:.1} us a pair", ms * 1e3 / pairs as f64);
    Ok(())
}

pub fn run(a: &Args) -> Result<bool> {
    let want: Vec<&str> = a.pos.iter().skip(1).map(String::as_str).collect();
    let all: Vec<&'static str> = cases().iter().map(|c| c.name).collect();
    if let Some(w) = want.iter().find(|w| !all.contains(w)) {
        return err(format!("{w}: no such case ({})", all.join(", ")));
    }
    let dev = Dev::new(0)?;
    let lib = Lib::find()?;
    let kern = HbKern::load(&dev)?;
    println!("{} (sm_{}{}), {}: the Hummingbird layer's kernels on the GPU against float64", dev.name, dev.cc.0, dev.cc.1, lib.path);
    let mut failed = Vec::new();
    if !a.has("no-check") {
        group_ids_case(&dev, &kern, &mut failed)?;
        for c in cases().iter().filter(|c| want.is_empty() || want.contains(&c.name)) {
            // (an error of the GPU or the library ends that case: it is said, and the others go on)
            if let Err(e) = run_case(&dev, &lib, &kern, c, &mut failed) {
                println!("FAIL {}: {e}", c.name);
                failed.push(format!("{}: {e}", c.name));
            }
        }
        println!("{}", if failed.is_empty() { "all of the Hummingbird kernels within their bounds" } else { "FAILED" });
    }
    if a.has("speed") {
        for (bits, tokens) in [(3, 1), (2, 1), (3, 2048)] {
            if let Err(e) = speed(&dev, &lib, &kern, bits, tokens, 512) {
                println!("FAIL speed {bits} bits {tokens} tokens: {e}");
                failed.push(format!("speed {bits}: {e}"));
            }
        }
    }
    if !failed.is_empty() {
        println!("FAILED: {}", failed.join("; "));
    }
    Ok(failed.is_empty())
}
