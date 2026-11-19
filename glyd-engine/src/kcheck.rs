//! `glyd-engine kcheck IN_DIR OUT_DIR`: the engine's own kernels (norm, norm with the residual add, query/key norm with rotary,
//! SiLU product) run on inputs another program wrote, their outputs written back as bf16 bits: tools/kcheck.py compares them
//! with vLLM's own operations. Files are raw little-endian bf16 (u16) unless noted; *.txt hold the sizes.

use crate::dev::{Buf, Dev};
use crate::ffi::{err, Error, Result};
use crate::kern::Kern;
use std::path::Path;

fn io<T>(r: std::io::Result<T>) -> Result<T> {
    r.map_err(|e| Error(e.to_string()))
}

fn nums(dir: &Path, name: &str) -> Result<Vec<f64>> {
    let t = io(std::fs::read_to_string(dir.join(name)))?;
    t.split_whitespace().map(|x| x.parse::<f64>().map_err(|_| Error(format!("{name}: {x}: not a number")))).collect()
}

fn load(dir: &Path, name: &str) -> Result<Buf> {
    let b = io(std::fs::read(dir.join(name)))?;
    let buf = Buf::new(b.len())?;
    buf.write(0, &b)?;
    Ok(buf)
}

fn save(dir: &Path, name: &str, b: &Buf) -> Result<()> {
    let mut v = vec![0u8; b.bytes];
    b.read(0, &mut v)?;
    io(std::fs::write(dir.join(name), v))
}

pub fn run(ind: &Path, outd: &Path) -> Result<()> {
    io(std::fs::create_dir_all(outd))?;
    let dev = Dev::new(0)?;
    let k = Kern::load(&dev)?;
    let s = dev.stream;
    // SAFETY (this whole function): buffers made here of the sizes the files give, used on the device's stream and read after a synchronize.
    unsafe {
        let m = nums(ind, "norm.txt")?;
        let (rows, kk, eps) = (m[0] as u32, m[1] as i32, m[2] as f32);
        let (x, y, w) = (load(ind, "norm_x.bin")?, load(ind, "norm_y.bin")?, load(ind, "norm_w.bin")?);
        let out = Buf::new(x.bytes)?;
        k.rmsnorm(s, x.ptr, w.ptr, out.ptr, rows, kk, eps)?;
        dev.sync()?;
        save(outd, "norm_out.bin", &out)?;
        let x2 = load(ind, "norm_x.bin")?;
        k.rmsnorm_add(s, x2.ptr, y.ptr, w.ptr, out.ptr, rows, kk, eps)?;
        dev.sync()?;
        save(outd, "add_x.bin", &x2)?;
        save(outd, "add_out.bin", &out)?;

        let m = nums(ind, "rope.txt")?;
        let (t, nq, nk, eps) = (m[0] as u32, m[1] as i32, m[2] as i32, m[4] as f32);
        // the engine's own rotary tables, for the caller to compare with torch's
        let (c, sn_) = crate::weights::rope_tables(m[5], 128, m[3] as usize);
        io(std::fs::write(outd.join("my_cos.bin"), crate::weights::as_bytes(&c)))?;
        io(std::fs::write(outd.join("my_sin.bin"), crate::weights::as_bytes(&sn_)))?;
        let (qkv, qw, kw, cs, sn, pos) = (load(ind, "rope_qkv.bin")?, load(ind, "rope_qw.bin")?, load(ind, "rope_kw.bin")?, load(ind, "rope_cos.bin")?, load(ind, "rope_sin.bin")?, load(ind, "rope_pos.bin")?);
        let (qo, ko) = (Buf::new(t as usize * nq as usize * 128 * 2)?, Buf::new(t as usize * nk as usize * 128 * 2)?);
        k.qk_norm_rope(s, qkv.ptr, (nq + 2 * nk) * 128, qw.ptr, kw.ptr, cs.ptr, sn.ptr, pos.ptr, qo.ptr, ko.ptr, t, nq, nk, eps)?;
        dev.sync()?;
        save(outd, "rope_q.bin", &qo)?;
        save(outd, "rope_k.bin", &ko)?;

        let m = nums(ind, "silu.txt")?;
        let (t, inter) = (m[0] as u32, m[1] as i32);
        let gu = load(ind, "silu_gu.bin")?;
        let o = Buf::new(t as usize * inter as usize * 2)?;
        k.silu_mul(s, gu.ptr, o.ptr, t, inter)?;
        dev.sync()?;
        save(outd, "silu_out.bin", &o)?;
    }
    if !outd.join("silu_out.bin").exists() {
        return err("no output written");
    }
    Ok(())
}

/// bf16 bits as the number that grows with the value (-0 and +0 the same), as the kernel orders them.
fn key(b: u16) -> u32 {
    let b = if b == 0x8000 { 0 } else { b as u32 };
    if b & 0x8000 != 0 { !b & 0xFFFF } else { b | 0x8000 }
}

/// The largest 64 of a row's bf16 values by the order the sampler takes its candidates in: largest first, equal values by the smaller index.
fn host_top(row: &[u16]) -> Vec<(i32, f32)> {
    let mut idx: Vec<usize> = (0..row.len()).collect();
    idx.sort_by(|&a, &b| key(row[b]).cmp(&key(row[a])).then(a.cmp(&b)));
    idx[..crate::kern::TOPK].iter().map(|&i| (i as i32, f32::from_bits((row[i] as u32) << 16))).collect()
}

/// The sum over a row of exp((x - max) / t), the host's way: the quotient in fp32, the exponential and the sum in fp64.
fn host_mass(row: &[u16], t: f32) -> f64 {
    let at = |b: u16| f32::from_bits((b as u32) << 16);
    let max = row.iter().map(|&b| at(b)).fold(f32::NEG_INFINITY, f32::max);
    row.iter().map(|&b| (((at(b) - max) / t) as f64).exp()).sum()
}

/// A step's candidates (what the sampler is given, `crate::model::TopGpu`) against the row of bf16 bits they were made from: the 64 largest as the host orders them, token and logit bit for bit, and where the answer has a sum of the row's
/// exponentials at temperature `t`, that within 2e-6 of the host's in fp64 (the sampler allows 1e-5). The first difference, in words.
pub fn top_vs_row(top: &crate::sampler::Top, row: &[u16], t: f32) -> std::result::Result<(), String> {
    let want = host_top(row);
    if top.cands.len() != want.len() {
        return Err(format!("{} candidates, not {}", top.cands.len(), want.len()));
    }
    for (j, (w, g)) in want.iter().zip(&top.cands).enumerate() {
        if w.0 as u32 != g.0 || !(w.1.to_bits() == g.1.to_bits() || (w.1 == 0.0 && g.1 == 0.0)) {
            return Err(format!("candidate {j}: the row's {w:?}, the GPU's {g:?}"));
        }
    }
    if let Some(m) = top.mass {
        let h = host_mass(row, t);
        let rel = (m / h - 1.0).abs();
        if rel.is_nan() || rel >= 2e-6 {
            return Err(format!("the sum of the exponentials at temperature {t}: the row's {h}, the GPU's {m} ({rel:.2e} apart)"));
        }
    }
    Ok(())
}

/// The sampling kernel (the 64 largest logits of a row) on rows of every shape it has trouble with (many equal values at the boundary, all equal, both zeros, ramps, infinities, a vocabulary's size and the smallest ones), each against the
/// host's order, and its time for a step's 8 rows; and the kernel that sums a row's exponentials (the top-p cut's normalizer) on the finite ones at three temperatures, each against the host's sum in fp64, and its time for a row of
/// Nemotron 3 Super's vocabulary and of Qwen3.5's. False where any row differs.
pub fn topk_check(dev: &Dev) -> Result<bool> {
    let k = Kern::load(dev)?;
    let mut x = 0x9E37_79B9_7F4A_7C15u64;
    let mut rnd = move || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        x
    };
    let bf = |f: f32| (f.to_bits() >> 16) as u16;
    let mut bad = 0usize;
    let mut checked = 0usize;
    let (mut sums, mut off, mut worst) = (0usize, 0usize, 0f64);
    for (v, name) in [(151_936usize, "a vocabulary"), (64, "64 values"), (72, "72 values"), (4096, "4,096 values")] {
        let mut rows: Vec<(String, Vec<u16>)> = Vec::new();
        // a spread of values (the sum of three uniforms: roughly a bell)
        rows.push(("a spread".into(), (0..v).map(|_| bf(((rnd() % 1000 + rnd() % 1000 + rnd() % 1000) as f32 / 150.0) - 10.0)).collect()));
        // 16 levels: the 64th largest has many equals
        rows.push(("16 levels".into(), (0..v).map(|_| bf((rnd() % 16) as f32 * 0.5 - 4.0)).collect()));
        rows.push(("all equal".into(), vec![bf(1.0); v]));
        rows.push(("both zeros".into(), (0..v).map(|_| if rnd() & 1 == 0 { 0x0000 } else { 0x8000 }).collect()));
        // exactly 70 values at the top, in places of the generator's: 64 of them are taken, the first by index
        let mut r = vec![bf(-3.0); v];
        for _ in 0..70.min(v) {
            r[(rnd() % v as u64) as usize] = bf(5.0);
        }
        rows.push(("a tie across the boundary".into(), r));
        // exactly 63 above and the rest equal
        let mut r = vec![bf(0.25); v];
        for i in 0..63 {
            r[(i * 7 + 3) % v] = bf(2.0 + i as f32);
        }
        rows.push(("63 above, the rest equal".into(), r));
        rows.push(("falling".into(), (0..v).map(|i| bf(-(i as f32) / 64.0)).collect()));
        rows.push(("rising".into(), (0..v).map(|i| bf(i as f32 / 64.0)).collect()));
        let mut r: Vec<u16> = (0..v).map(|_| bf((rnd() % 100) as f32 - 50.0)).collect();
        r[v / 3] = 0x7f80;
        r[v / 2] = 0xff80;
        r[v - 1] = 0x7f80;
        rows.push(("infinities".into(), r));
        rows.push(("small and negative".into(), (0..v).map(|_| bf(-100.0 - (rnd() % 64) as f32)).collect()));
        let n = rows.len();
        let data: Vec<u16> = rows.iter().flat_map(|r| r.1.iter().copied()).collect();
        let logits = Buf::new(2 * data.len())?;
        logits.write(0, &data)?;
        let (tok, val) = (Buf::zeroed(4 * n * crate::kern::TOPK)?, Buf::zeroed(4 * n * crate::kern::TOPK)?);
        // SAFETY: live buffers of the sizes the kernel reads and writes; the stream is synchronized before they are read.
        unsafe { k.topk_rows(dev.stream, logits.ptr, v as i32, n as u32, tok.ptr, val.ptr)? };
        dev.sync()?;
        let (mut gt, mut gv) = (vec![0i32; n * crate::kern::TOPK], vec![0f32; n * crate::kern::TOPK]);
        tok.read(0, &mut gt)?;
        val.read(0, &mut gv)?;
        for (i, (label, row)) in rows.iter().enumerate() {
            checked += 1;
            let want = host_top(row);
            let got: Vec<(i32, f32)> = (0..crate::kern::TOPK).map(|j| (gt[i * crate::kern::TOPK + j], gv[i * crate::kern::TOPK + j])).collect();
            let same = want.iter().zip(&got).all(|(a, b)| a.0 == b.0 && (a.1.to_bits() == b.1.to_bits() || (a.1 == 0.0 && b.1 == 0.0)));
            if !same {
                bad += 1;
                let at = want.iter().zip(&got).position(|(a, b)| a.0 != b.0 || a.1.to_bits() != b.1.to_bits()).unwrap_or(0);
                println!("DIFFERENT: {name}, {label}: first at {at}: host {:?}, kernel {:?}", want[at], got[at]);
            }
        }
        // the sum of the exponentials, on the rows with no infinity, at three temperatures (each row's largest, as the argmax leaves it, given)
        let at = |b: u16| f32::from_bits((b as u32) << 16);
        let maxes: Vec<f32> = rows.iter().map(|r| r.1.iter().map(|&b| at(b)).fold(f32::NEG_INFINITY, f32::max)).collect();
        let (maxv, out) = (Buf::new(4 * n)?, Buf::zeroed(8 * n)?);
        maxv.write(0, &maxes)?;
        for t in [0.6f32, 1.0, 1.7] {
            // SAFETY: live buffers of the sizes the kernel reads and writes; the stream is synchronized before they are read.
            unsafe { k.exp_sum_rows(dev.stream, logits.ptr, v as i32, n as u32, maxv.ptr, t, out.ptr)? };
            dev.sync()?;
            let mut got = vec![0f64; n];
            out.read(0, &mut got)?;
            for (i, (label, row)) in rows.iter().enumerate().filter(|(_, r)| r.1.iter().all(|&b| b & 0x7F80 != 0x7F80)) {
                let want = host_mass(row, t);
                let rel = (got[i] / want - 1.0).abs();
                sums += 1;
                worst = worst.max(rel);
                if rel.is_nan() || rel >= 2e-6 {
                    off += 1;
                    println!("DIFFERENT: {name}, {label}, temperature {t}: the sum of the exponentials is {} on the host, {} on the GPU ({rel:.2e} apart)", want, got[i]);
                }
            }
        }
    }
    // the time of a step's 8 rows of a vocabulary's size
    let v = 151_936usize;
    let rows = 8usize;
    let logits = Buf::new(2 * v * rows)?;
    logits.write(0, &(0..v * rows).map(|_| bf((rnd() % 1000) as f32 / 100.0)).collect::<Vec<u16>>())?;
    let (tok, val) = (Buf::zeroed(4 * rows * crate::kern::TOPK)?, Buf::zeroed(4 * rows * crate::kern::TOPK)?);
    let t = crate::dev::Timer::new()?;
    // SAFETY: as above.
    let ms = unsafe {
        k.topk_rows(dev.stream, logits.ptr, v as i32, rows as u32, tok.ptr, val.ptr)?;
        t.start(dev.stream)?;
        for _ in 0..50 {
            k.topk_rows(dev.stream, logits.ptr, v as i32, rows as u32, tok.ptr, val.ptr)?;
        }
        t.stop(dev.stream)? / 50.0
    };
    println!("top-64 kernel: {checked} rows of {} shapes checked against the host's order, {bad} differ; {ms:.3} ms for a step's {rows} rows of {v} logits", 10);
    // the sum of the exponentials: a row of Nemotron 3 Super's vocabulary and of Qwen3.5's
    let mut times = Vec::new();
    for v in [131_072usize, 248_320] {
        let logits = Buf::new(2 * v)?;
        logits.write(0, &(0..v).map(|_| bf((rnd() % 1000) as f32 / 100.0)).collect::<Vec<u16>>())?;
        let (maxv, out) = (Buf::new(4)?, Buf::zeroed(8)?);
        maxv.write(0, &[9.99f32])?;
        let t = crate::dev::Timer::new()?;
        // SAFETY: as above.
        let ms = unsafe {
            k.exp_sum_rows(dev.stream, logits.ptr, v as i32, 1, maxv.ptr, 1.0, out.ptr)?;
            t.start(dev.stream)?;
            for _ in 0..50 {
                k.exp_sum_rows(dev.stream, logits.ptr, v as i32, 1, maxv.ptr, 1.0, out.ptr)?;
            }
            t.stop(dev.stream)? / 50.0
        };
        times.push(format!("{ms:.3} ms for a row of {v}"));
    }
    println!("sum-of-exponentials kernel: {sums} sums (the rows with no infinity, 3 temperatures) checked against the host's in fp64, the worst {worst:.2e} apart (the bound is 2e-6), {off} beyond; {}", times.join(", "));
    Ok(bad == 0 && off == 0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sampler::Top;

    /// `top_vs_row` holds the candidates a GPU gives to the row they come from, where the check of `nhtoy` and `topkcheck` can only be run on one: the row's own answer is accepted (a sum 1e-6 off too), and a candidate out of
    /// order, a logit a place off, a missing candidate and a sum 1e-5 off are each found, with the first difference said.
    #[test]
    fn the_candidates_are_held_to_the_row() {
        let mut x = 0x2545_F491_4F6C_DD1Du64;
        let mut next = move || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        // 5,000 bf16 values with many equal ones
        let row: Vec<u16> = (0..5000).map(|_| (((next() % 4000) as f32 / 400.0 - 5.0).to_bits() >> 16) as u16).collect();
        let t = 0.8f32;
        let top = Top { cands: host_top(&row).iter().map(|&(i, v)| (i as u32, v)).collect(), mass: Some(host_mass(&row, t)) };
        assert_eq!(top_vs_row(&top, &row, t), Ok(()));
        let sum = top.mass.unwrap();
        assert_eq!(top_vs_row(&Top { mass: Some(sum * (1.0 + 1e-6)), ..top.clone() }, &row, t), Ok(()));
        assert!(top_vs_row(&Top { mass: Some(sum * (1.0 + 1e-5)), ..top.clone() }, &row, t).unwrap_err().contains("sum of the exponentials"));
        assert_eq!(top_vs_row(&Top { mass: None, ..top.clone() }, &row, t), Ok(()), "no sum asked for");
        let mut swapped = top.clone();
        swapped.cands.swap(3, 4);
        assert!(top_vs_row(&swapped, &row, t).unwrap_err().starts_with("candidate 3"));
        let mut off = top.clone();
        off.cands[7].1 = f32::from_bits(off.cands[7].1.to_bits() + 0x1_0000);
        assert!(top_vs_row(&off, &row, t).unwrap_err().starts_with("candidate 7"));
        let mut short = top.clone();
        short.cands.pop();
        assert!(top_vs_row(&short, &row, t).unwrap_err().contains("63 candidates"));
    }
}
