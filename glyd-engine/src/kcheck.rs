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
        let (qkv, qw, kw, cs, sn, pos) = (
            load(ind, "rope_qkv.bin")?,
            load(ind, "rope_qw.bin")?,
            load(ind, "rope_kw.bin")?,
            load(ind, "rope_cos.bin")?,
            load(ind, "rope_sin.bin")?,
            load(ind, "rope_pos.bin")?,
        );
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
    if b & 0x8000 != 0 {
        !b & 0xFFFF
    } else {
        b | 0x8000
    }
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

/// A float's bits as the number that grows with the value (-0 and +0 the same), as `topk64_f32` orders them.
fn key32(x: f32) -> u32 {
    let b = x.to_bits();
    let b = if b == 0x8000_0000 { 0 } else { b };
    if b & 0x8000_0000 != 0 {
        !b
    } else {
        b | 0x8000_0000
    }
}

/// The largest 64 of a row of floats as `topk64_f32` takes them: largest first, equal values by the smaller index.
fn host_top32(w: &[f32]) -> Vec<(i32, f32)> {
    let mut idx: Vec<usize> = (0..w.len()).collect();
    idx.sort_by(|&a, &b| key32(w[b]).cmp(&key32(w[a])).then(a.cmp(&b)));
    idx[..crate::kern::TOPK].iter().map(|&i| (i as i32, w[i])).collect()
}

/// The kinds of `changes`.
const CHANGES: usize = 11;

/// A row's changes for the check (src/sampler.rs `Resolved`), of kind `kind` (of CHANGES), at random for a row of `v`: a mask of half the tokens; one of 5% with biases, the repetition
/// penalty and the times said; no mask and the three penalties on 500 tokens with biases of the server's bounds; ten tokens allowed and two others put back at 0 (a grammar's end under a
/// least length); none allowed and three put back; a repetition penalty below 1 and a negative frequency penalty on 300 tokens said up to 1,000 times; one token allowed and 64 changed;
/// then each change alone: a bias on 100 tokens, the repetition penalty on 400, the presence penalty on 300, the frequency penalty on 300 said up to 1,000 times.
fn changes(kind: usize, v: usize, rnd: &mut dyn FnMut() -> u64) -> crate::sampler::Resolved {
    use crate::sampler::{Entry, Resolved};
    let n = v.div_ceil(32);
    let pick = |rnd: &mut dyn FnMut() -> u64| (rnd() % v as u64) as u32;
    let dense = |density: f64, rnd: &mut dyn FnMut() -> u64| -> Vec<u32> {
        let mut w = vec![0u32; n];
        for t in 0..v {
            if (rnd() % 1_000_000) as f64 / 1e6 < density {
                w[t / 32] |= 1 << (t % 32);
            }
        }
        w
    };
    let mut ents: std::collections::BTreeMap<u32, Entry> = Default::default();
    let mut add = |t: u32, f: &dyn Fn(&mut Entry)| f(ents.entry(t).or_insert(Entry { token: t, bias: None, seen: false, said: 0, rescue: false }));
    let (mut words, mut rep, mut pres, mut freq) = (None, 1.0f32, 0.0f32, 0.0f32);
    match kind % CHANGES {
        0 => words = Some(dense(0.5, rnd)),
        1 => {
            words = Some(dense(0.05, rnd));
            (rep, pres, freq) = (1.2, 0.4, 0.1);
            for j in 0..40 {
                let (t, b, said) = (pick(rnd), (rnd() % 2001) as f32 / 10.0 - 100.0, (rnd() % 5) as u32);
                add(t, &|e| {
                    e.bias = (j % 3 == 0).then_some(b);
                    e.seen = j % 2 == 0;
                    e.said = said;
                });
            }
        }
        2 => {
            (rep, pres, freq) = (1.3, 0.5, 0.25);
            for j in 0..500 {
                let (t, said) = (pick(rnd), (rnd() % 50) as u32);
                add(t, &|e| {
                    e.seen = true;
                    e.said = if j % 3 == 0 { said } else { 0 };
                });
            }
            for j in 0..20 {
                let t = pick(rnd);
                add(t, &|e| e.bias = Some(if j % 2 == 0 { 100.0 } else { -100.0 }));
            }
        }
        3 | 4 => {
            let mut w = vec![0u32; n];
            for _ in 0..if kind % CHANGES == 3 { 10 } else { 0 } {
                let t = pick(rnd) as usize;
                w[t / 32] |= 1 << (t % 32);
            }
            words = Some(w);
            (rep, pres, freq) = (1.5, 0.3, 0.2);
            for j in 0..if kind % CHANGES == 3 { 2 } else { 3 } {
                let t = pick(rnd);
                add(t, &|e| {
                    e.rescue = true;
                    e.seen = j == 0;
                    e.said = j as u32;
                });
            }
        }
        5 => {
            (rep, freq) = (0.7, -0.3);
            for _ in 0..300 {
                let (t, said, b) = (pick(rnd), (rnd() % 1001) as u32, (rnd() % 7) as f32 - 3.0);
                add(t, &|e| {
                    e.seen = said % 2 == 0;
                    e.said = said;
                    e.bias = (said % 5 == 0).then_some(b);
                });
            }
        }
        7 => {
            for j in 0..100 {
                let (t, b) = (pick(rnd), if j % 4 == 0 { 100.0 } else { (rnd() % 2001) as f32 / 10.0 - 100.0 });
                add(t, &|e| e.bias = Some(b));
            }
        }
        8 => {
            rep = 1.3;
            for _ in 0..400 {
                let t = pick(rnd);
                add(t, &|e| e.seen = true);
            }
        }
        9 | 10 => {
            (pres, freq) = if kind % CHANGES == 9 { (0.7, 0.0) } else { (0.0, 0.35) };
            for _ in 0..300 {
                let (t, said) = (pick(rnd), 1 + (rnd() % 1000) as u32);
                add(t, &|e| e.said = said);
            }
        }
        _ => {
            let mut w = vec![0u32; n];
            let t = pick(rnd) as usize;
            w[t / 32] |= 1 << (t % 32);
            words = Some(w);
            (rep, pres) = (1.1, 1.0);
            add(t as u32, &|e| e.said = 3);
            for _ in 0..64 {
                let (t, b) = (pick(rnd), (rnd() % 100) as f32 / 7.0);
                add(t, &|e| {
                    e.bias = Some(b);
                    e.seen = true;
                });
            }
        }
    }
    Resolved { words, entries: ents.into_values().collect(), repetition: rep, presence: pres, frequency: freq }
}

/// The rows the draws see (`ProcGpu`: a row's mask, penalties and bias put on it, src/sampler.rs `Resolved`) of the rows `rows` (bf16 bits, at `logits` on the device), a round of
/// changes for each kind (`changes`) so that every row has each: the GPU's row against the host's (`Resolved::apply`, which src/sampler.rs holds to `Sampler::processed`) bit for bit,
/// its 64 largest against the host's order of it, and a draw's questions of it at two temperatures (`draw_queries`: its total, bins and tokens, each the host's bit for bit).
/// Returns the rows checked, those that differ, the questions checked, those that differ, and 0.
fn proc_check(
    k: &Kern,
    dev: &Dev,
    logits: &Buf,
    v: usize,
    rows: &[(String, Vec<u16>)],
    name: &str,
    rnd: &mut dyn FnMut() -> u64,
) -> Result<(usize, usize, usize, usize, f64)> {
    let n = rows.len();
    let (mut checked, mut bad, mut sums, mut off, worst) = (0, 0, 0, 0, 0f64);
    let raw: Vec<Vec<f32>> = rows.iter().map(|r| r.1.iter().map(|&b| f32::from_bits((b as u32) << 16)).collect()).collect();
    let mut g = crate::kern::ProcGpu::new(v);
    let (tok, val) = (Buf::zeroed(4 * n * crate::kern::TOPK)?, Buf::zeroed(4 * n * crate::kern::TOPK)?);
    for round in 0..CHANGES {
        let res: Vec<crate::sampler::Resolved> = (0..n).map(|i| changes(i + round, v, rnd)).collect();
        let list: Vec<(usize, &crate::sampler::Resolved)> = res.iter().enumerate().collect();
        // SAFETY: n rows of v bf16 logits, and TOPK candidates' room for each; synchronized before anything is read
        unsafe { g.queue(k, dev, &list, logits.ptr, tok.ptr, val.ptr)? };
        dev.sync()?;
        let (mut gt, mut gv) = (vec![0i32; n * crate::kern::TOPK], vec![0f32; n * crate::kern::TOPK]);
        tok.read(0, &mut gt)?;
        val.read(0, &mut gv)?;
        for (i, (label, _)) in rows.iter().enumerate() {
            checked += 1;
            let what = format!("{name}, {label}, changes of kind {}", (i + round) % CHANGES);
            let host = res[i].apply(&raw[i]);
            let mut w = vec![0f32; v];
            let at = g.row_of(i).expect("a row made");
            // SAFETY: W's row i holds v floats; the stream was synchronized above and is again before `w` is read
            unsafe { crate::dev::copy_d2h_async(w.as_mut_ptr() as *mut _, at, 4 * v, dev.stream)? };
            dev.sync()?;
            if let Some(t) = (0..v).find(|&t| w[t].to_bits() != host[t].to_bits()) {
                bad += 1;
                println!("DIFFERENT: {what}: the row's token {t} is {:?} on the GPU, {:?} on the host", w[t], host[t]);
                continue;
            }
            let wanted = host_top32(&host);
            let got: Vec<(i32, f32)> = (0..crate::kern::TOPK).map(|j| (gt[i * crate::kern::TOPK + j], gv[i * crate::kern::TOPK + j])).collect();
            if let Some(j) = wanted.iter().zip(&got).position(|(a, b)| a.0 != b.0 || a.1.to_bits() != b.1.to_bits()) {
                bad += 1;
                println!("DIFFERENT: {what}: candidate {j}: host {:?}, kernel {:?}", wanted[j], got[j]);
                continue;
            }
            // (a draw's questions of a row whose largest is a number and that holds no other infinity or NaN)
            let max = wanted[0].1;
            if !max.is_finite() || host.iter().any(|x| x.is_nan() || *x == f32::INFINITY) {
                continue;
            }
            let maxv = val.ptr + (4 * crate::kern::TOPK * i) as u64;
            for t in [0.7f32, 1.0] {
                let (c, o) = draw_queries(k, dev, at, true, maxv, &host, t, &what)?;
                (sums, off) = (sums + c, off + o);
            }
        }
    }
    Ok((checked, bad, sums, off, worst))
}

/// A draw's questions of the row of `v` at `row` on the GPU (bf16 logits, or floats where `f32`; its largest at `maxv`, f32) at temperature `t` against the host's of `host` (the
/// same row widened: `sampler::emu`): the total of its weights; its bins for every token, for a min-p's floor past the 100th token's key, and for the largest bin's range by its
/// keys' differing bits; and that range's tokens where they are SELECT_CAP or fewer. Each the host's bit for bit (whole numbers: no order of sums is the host's or the GPU's). The
/// questions asked and those that differ (each said).
#[allow(clippy::too_many_arguments)]
pub fn draw_queries(k: &Kern, dev: &Dev, row: u64, f32: bool, maxv: u64, host: &[f32], t: f32, what: &str) -> Result<(usize, usize)> {
    use crate::sampler::{emu, Query, SELECT_CAP};
    let v = host.len();
    let (mut asked, mut bad) = (0, 0);
    let max = emu::top(host)[0].1;
    let out = Buf::zeroed(crate::kern::DRAW_BINS_BYTES.max(8 + 8 * (SELECT_CAP + 1)))?;
    // SAFETY: the row of v (bf16 or f32) and its largest; one u64 at `out`, read after the synchronize
    unsafe { k.draw_total(dev.stream, row, f32, v, 1, maxv, t, out.ptr)? };
    dev.sync()?;
    let mut got = [0u64; 1];
    out.read(0, &mut got)?;
    asked += 1;
    if got[0] != emu::total(host, max, t) {
        bad += 1;
        println!("DIFFERENT: {what}, temperature {t}: the total of the weights is {} on the GPU, {} on the host", got[0], emu::total(host, max, t));
    }
    let all = Query::all(max, t);
    let order = emu::tokens(host, &all);
    let mut keyed: Vec<u64> = order.iter().map(|&(tk, x)| crate::sampler::key(x, tk)).collect();
    keyed.sort_unstable_by(|a, b| b.cmp(a));
    let lo = keyed.get(100).copied().unwrap_or(0);
    let hb = emu::bins(host, &all);
    let big = (0..256).max_by_key(|&b| hb.count[b]).unwrap_or(0);
    let (least, most) = (hb.least[big], hb.most[big]);
    let narrow = Query { lo: least.min(most), hi: most, shift: (63 - (least ^ most).leading_zeros().min(63)).saturating_sub(7), ..all };
    for q in [all, Query { lo, floor: crate::sampler::min_floor(0.05), ..all }, narrow] {
        // SAFETY: as above; DRAW_BINS_BYTES at `out`
        unsafe { k.draw_bins(dev.stream, row, f32, v, &q, out.ptr)? };
        dev.sync()?;
        let mut w = vec![0u32; crate::kern::DRAW_BINS_BYTES / 4];
        out.read(0, &mut w)?;
        asked += 1;
        let (g, h) = (crate::kern::bins_of_words(&w), emu::bins(host, &q));
        if g != h {
            bad += 1;
            let b = (0..256).find(|&b| (g.count[b], g.weight[b], g.least[b], g.most[b]) != (h.count[b], h.weight[b], h.least[b], h.most[b])).unwrap_or(0);
            println!(
                "DIFFERENT: {what}, temperature {t}, {q:?}: bin {b} is {:?} on the GPU, {:?} on the host",
                (g.count[b], g.weight[b], g.least[b], g.most[b]),
                (h.count[b], h.weight[b], h.least[b], h.most[b])
            );
        }
    }
    let mut want = emu::tokens(host, &narrow);
    if want.len() <= SELECT_CAP {
        let cap = SELECT_CAP + 1;
        // SAFETY: as above; a count, then `cap` tokens and logits at `out`
        unsafe { k.draw_tokens(dev.stream, row, f32, v, &narrow, cap, out.ptr, out.ptr + 8, out.ptr + 8 + 4 * cap as u64)? };
        dev.sync()?;
        let mut n = [0u32; 1];
        out.read(0, &mut n)?;
        let n = n[0] as usize;
        let (mut tk, mut x) = (vec![0i32; n.min(cap)], vec![0f32; n.min(cap)]);
        out.read(8, &mut tk)?;
        out.read(8 + 4 * cap, &mut x)?;
        let mut got: Vec<(u32, f32)> = tk.into_iter().map(|a| a as u32).zip(x).collect();
        got.sort_by_key(|a| a.0);
        want.sort_by_key(|a| a.0);
        asked += 1;
        if n != want.len() || got.iter().zip(&want).any(|(a, b)| a.0 != b.0 || a.1.to_bits() != b.1.to_bits()) {
            bad += 1;
            println!("DIFFERENT: {what}, temperature {t}: the largest bin's range holds {n} tokens on the GPU, {} on the host (or others)", want.len());
        }
    }
    Ok((asked, bad))
}

/// Every float from -0 down to -32 weighed on the GPU (`Kern::draw_weight_sums`) against the host's `sampler::weight`: the sums of the weights, and of each times its float's bits,
/// the same (mod 2^64; 1.1 billion floats). False where they differ.
pub fn weights_check(dev: &Dev) -> Result<bool> {
    let k = Kern::load(dev)?;
    let out = Buf::zeroed(16)?;
    let t = std::time::Instant::now();
    // SAFETY: two u64 at `out`, read after the synchronize
    unsafe { k.draw_weight_sums(dev.stream, out.ptr)? };
    dev.sync()?;
    let gpu_ms = t.elapsed().as_secs_f64() * 1e3;
    let mut g = [0u64; 2];
    out.read(0, &mut g)?;
    let t = std::time::Instant::now();
    let (mut sw, mut sh) = (0u64, 0u64);
    for b in 0x8000_0001u32..=0xC200_0000 {
        let w = crate::sampler::weight(f32::from_bits(b));
        sw = sw.wrapping_add(w);
        sh = sh.wrapping_add(w.wrapping_mul(b as u64));
    }
    let same = g == [sw, sh];
    println!(
        "{} the draw's weights of every float from -0 to -32: GPU sums {:#x} {:#x}, host {:#x} {:#x} ({gpu_ms:.0} ms and {:.0} ms)",
        if same { "PASS" } else { "FAIL" },
        g[0],
        g[1],
        sw,
        sh,
        t.elapsed().as_secs_f64() * 1e3
    );
    Ok(same)
}

/// A step's candidates (what the sampler is given, `ModelStep::top`) against the row of bf16 bits they were made from: the 64 largest as the host orders them, token and logit bit
/// for bit, and where `total` is given (a draw's total of the row's weights at its temperature, `ModelStep::total`), that the host's to the unit. The first difference, in words.
pub fn top_vs_row(cands: &[(u32, f32)], total: Option<(u64, f32)>, row: &[u16]) -> std::result::Result<(), String> {
    let want = host_top(row);
    if cands.len() != want.len() {
        return Err(format!("{} candidates, not {}", cands.len(), want.len()));
    }
    for (j, (w, g)) in want.iter().zip(cands).enumerate() {
        if w.0 as u32 != g.0 || !(w.1.to_bits() == g.1.to_bits() || (w.1 == 0.0 && g.1 == 0.0)) {
            return Err(format!("candidate {j}: the row's {w:?}, the GPU's {g:?}"));
        }
    }
    if let Some((got, t)) = total {
        let host: Vec<f32> = row.iter().map(|&b| f32::from_bits((b as u32) << 16)).collect();
        let h = crate::sampler::emu::total(&host, want[0].1, t);
        if got != h {
            return Err(format!("the total of the weights at temperature {t}: the row's {h}, the GPU's {got}"));
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
    let (mut pchecked, mut pbad, mut psums, mut poff, mut pworst) = (0usize, 0usize, 0usize, 0usize, 0f64);
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
        // the gather (`Kern::gather_row`: a log-probability's token's own logit): on every row above, the first, the last and 100 tokens of the generator's
        // places, each the row's own value bit for bit (a bf16 widened: the zeros' signs and the infinities too)
        {
            let mut idx: Vec<i32> = vec![0, v as i32 - 1];
            idx.extend((0..100).map(|_| (rnd() % v as u64) as i32));
            let (ib, ob) = (Buf::new(4 * idx.len())?, Buf::zeroed(4 * idx.len())?);
            ib.write(0, &idx)?;
            for (i, (label, row)) in rows.iter().enumerate() {
                checked += 1;
                // SAFETY: row i of the n rows of v bf16 values; idx holds its tokens below v and out room for as many f32; synchronized before out is read.
                unsafe { k.gather_row(dev.stream, logits.ptr, v, i, ib.ptr, idx.len(), ob.ptr)? };
                dev.sync()?;
                let mut got = vec![0f32; idx.len()];
                ob.read(0, &mut got)?;
                if let Some(j) = idx.iter().zip(&got).position(|(&t, g)| g.to_bits() != ((row[t as usize] as u32) << 16)) {
                    bad += 1;
                    println!(
                        "DIFFERENT: {name}, {label}: gathered token {} is {:?}, the row has {:?}",
                        idx[j],
                        got[j],
                        f32::from_bits((row[idx[j] as usize] as u32) << 16)
                    );
                }
            }
        }
        // the rows the draws see (the penalties, a bias and the tokens taken out put on each row on the GPU), on every row above with changes of every kind
        {
            let (c, b, s, o, w) = proc_check(&k, dev, &logits, v, &rows, name, &mut rnd)?;
            (pchecked, pbad, psums, poff, pworst) = (pchecked + c, pbad + b, psums + s, poff + o, pworst.max(w));
        }
        // a draw's questions of the model's rows (bf16), each of whose largest is a number, about the largest the argmax leaves
        for (i, (label, row)) in rows.iter().enumerate() {
            let host: Vec<f32> = row.iter().map(|&b| f32::from_bits((b as u32) << 16)).collect();
            if host.iter().any(|x| x.is_nan() || x.is_infinite()) {
                continue;
            }
            let maxv = Buf::new(4)?;
            maxv.write(0, &[crate::sampler::emu::top(&host)[0].1])?;
            for t in [0.6f32, 1.0] {
                let (c, o) = draw_queries(&k, dev, logits.ptr + (2 * v * i) as u64, false, maxv.ptr, &host, t, &format!("{name}, {label}"))?;
                (psums, poff) = (psums + c, poff + o);
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
                    println!(
                        "DIFFERENT: {name}, {label}, temperature {t}: the sum of the exponentials is {} on the host, {} on the GPU ({rel:.2e} apart)",
                        want, got[i]
                    );
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
    println!(
        "top-64 kernel: {checked} rows of {} shapes checked against the host's order, {bad} differ; {ms:.3} ms for a step's {rows} rows of {v} logits",
        10
    );
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
    // the rows the draws see, made for a step's 8 rows of a vocabulary's size: four with 500 tokens' logits changed and none taken out, four with half the tokens taken out
    let (v, rows) = (151_936usize, 8usize);
    let logits = Buf::new(2 * v * rows)?;
    logits.write(0, &(0..v * rows).map(|_| bf((rnd() % 1000) as f32 / 100.0)).collect::<Vec<u16>>())?;
    let res: Vec<crate::sampler::Resolved> = (0..rows).map(|i| changes(if i % 2 == 0 { 2 } else { 0 }, v, &mut rnd)).collect();
    let list: Vec<(usize, &crate::sampler::Resolved)> = res.iter().enumerate().collect();
    let (tok, val) = (Buf::zeroed(4 * rows * crate::kern::TOPK)?, Buf::zeroed(4 * rows * crate::kern::TOPK)?);
    let mut g = crate::kern::ProcGpu::new(v);
    let t = crate::dev::Timer::new()?;
    // SAFETY: as above; each queue's copies have run before the next is queued (the stream runs them in order, and its pinned bytes are rewritten only after a synchronize)
    let ms = unsafe {
        g.queue(&k, dev, &list, logits.ptr, tok.ptr, val.ptr)?;
        dev.sync()?;
        let mut total = 0.0;
        for _ in 0..20 {
            t.start(dev.stream)?;
            g.queue(&k, dev, &list, logits.ptr, tok.ptr, val.ptr)?;
            total += t.stop(dev.stream)?;
            dev.sync()?;
        }
        total / 20.0
    };
    println!(
        "the rows the draws see: {pchecked} rows with changes of {CHANGES} kinds (masks, penalties, biases, ends put back, each change alone) checked against the host's, {pbad} differ (the row bit for bit, its 64 largest); \
         {psums} draw's questions of those rows and the model's (totals, bins, tokens) checked, {poff} not the host's bit for bit; {ms:.3} ms to make a step's {rows} rows of {v} and their candidates"
    );
    Ok(bad == 0 && off == 0 && pbad == 0 && poff == 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `top_vs_row` holds the candidates a GPU gives to the row they come from, where the check of `nhtoy` and `topkcheck` can only be run on one: the row's own answer is accepted, and a
    /// candidate out of order, a logit a place off, a missing candidate and a total a unit off are each found, with the first difference said.
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
        let cands: Vec<(u32, f32)> = host_top(&row).iter().map(|&(i, v)| (i as u32, v)).collect();
        let host: Vec<f32> = row.iter().map(|&b| f32::from_bits((b as u32) << 16)).collect();
        let total = crate::sampler::emu::total(&host, cands[0].1, t);
        assert_eq!(top_vs_row(&cands, Some((total, t)), &row), Ok(()));
        assert!(top_vs_row(&cands, Some((total + 1, t)), &row).unwrap_err().contains("total of the weights"));
        assert_eq!(top_vs_row(&cands, None, &row), Ok(()), "no total asked for");
        let mut swapped = cands.clone();
        swapped.swap(3, 4);
        assert!(top_vs_row(&swapped, None, &row).unwrap_err().starts_with("candidate 3"));
        let mut off = cands.clone();
        off[7].1 = f32::from_bits(off[7].1.to_bits() + 0x1_0000);
        assert!(top_vs_row(&off, None, &row).unwrap_err().starts_with("candidate 7"));
        let mut short = cands.clone();
        short.pop();
        assert!(top_vs_row(&short, None, &row).unwrap_err().contains("63 candidates"));
    }
}
