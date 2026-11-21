//! Qwen3.8-Flash-Next's layers and model against transformers' (a development build's checks; the references are tools/qwen4x_ref.py's for the random tiny model and ~/glyd-scratch/qwen/qwen_ref.py's for the real one, in the
//! stage names both write):
//!   `glyd-engine qxlayers MODEL REF [--layers A:B] [--layout mma|i6|i5] [--level i6|i5|i4] [--tol X]`   each layer on the reference's input streams (all its positions in one call), every stage computed from the
//!       reference's input of that stage (the Gated Residuals, the token mixer, the streams' update, the router and the experts, PLE and its n-gram rows), then the whole layer from its input; the largest difference and the
//!       relative norm of the difference of each against the reference's. The layers are loaded one at a time (the real model's experts at `--level`), the rest of the matrices at `--layout` (default mma: lossless).
//!   `glyd-engine qxtoy MODEL REF [--chunk C] [--prefill P] [--layout L] [--level L]`   the whole model: a prompt of P tokens in chunks of C, then the rest of the reference's tokens one decode step each; the logits of every
//!       position against the reference's (KL divergence, top-1 agreement).

use crate::args::Args;
use crate::ffi::{err, Error, Result};
use crate::model_qwen4x::{gpu_experts, EngineQx, OptsQx, QxMoe};
use crate::qwen4x::{ConfigQx, QxKind};
use crate::weights::Layout;
use std::path::Path;

fn io<T>(r: std::io::Result<T>) -> Result<T> {
    r.map_err(|e| Error(e.to_string()))
}

fn words(dir: &Path, name: &str) -> Result<Vec<u16>> {
    let b = io(std::fs::read(dir.join(name))).map_err(|e| Error(format!("{}: {}", dir.join(name).display(), e.0)))?;
    Ok(b.as_chunks::<2>().0.iter().map(|x| u16::from_le_bytes(*x)).collect())
}

fn f32s(dir: &Path, name: &str) -> Result<Vec<f32>> {
    let b = io(std::fs::read(dir.join(name))).map_err(|e| Error(format!("{}: {}", dir.join(name).display(), e.0)))?;
    Ok(b.as_chunks::<4>().0.iter().map(|x| f32::from_le_bytes(*x)).collect())
}

fn i32s(dir: &Path, name: &str) -> Result<Vec<i32>> {
    Ok(f32s(dir, name)?.into_iter().map(|x| x.to_bits() as i32).collect())
}

fn widen(w: &[u16]) -> Vec<f32> {
    w.iter().map(|&x| f32::from_bits((x as u32) << 16)).collect()
}

fn bf16(x: f32) -> u16 {
    let b = x.to_bits();
    (b.wrapping_add(0x7fff + ((b >> 16) & 1)) >> 16) as u16
}

/// One stage against the reference: the largest difference, the largest reference value, the norm of the difference over the reference's; within `tol` (relative norm) or not, printed.
fn stage(name: &str, got: &[f32], want: &[f32], tol: f64) -> bool {
    if got.len() != want.len() {
        println!("  {name:<28} {} values, where the reference has {}", got.len(), want.len());
        return false;
    }
    let (mut dmax, mut wmax, mut d2, mut w2, mut bad) = (0f64, 0f64, 0f64, 0f64, 0usize);
    for (&g, &w) in got.iter().zip(want) {
        if !g.is_finite() {
            bad += 1;
            continue;
        }
        let d = (g - w).abs() as f64;
        dmax = dmax.max(d);
        wmax = wmax.max(w.abs() as f64);
        d2 += d * d;
        w2 += (w as f64) * (w as f64);
    }
    let rel = (d2 / w2.max(1e-30)).sqrt();
    let ok = bad == 0 && rel <= tol;
    println!(
        "  {name:<28} max diff {dmax:.3e} (of {wmax:.3e})  rel {rel:.2e}{}  {}",
        if bad > 0 { format!("  {bad} not finite") } else { String::new() },
        if ok { "ok" } else { "OFF" }
    );
    ok
}

// (the engine's stream does not block: every read and write of its buffers waits for it first)
fn read_f32(e: &EngineQx, b: &crate::dev::Buf, n: usize) -> Result<Vec<f32>> {
    e.dev.sync()?;
    let mut v = vec![0f32; n];
    b.read(0, &mut v)?;
    Ok(v)
}

fn read_bf16(e: &EngineQx, b: &crate::dev::Buf, n: usize) -> Result<Vec<f32>> {
    e.dev.sync()?;
    let mut v = vec![0u16; n];
    b.read(0, &mut v)?;
    Ok(widen(&v))
}

fn put<T: Copy>(e: &EngineQx, b: &crate::dev::Buf, v: &[T]) -> Result<()> {
    e.dev.sync()?;
    b.write(0, v)
}

/// The GPU's lock file of the shared machine held (flock, exclusive) while the value lives: `--lock PATH` (none: no lock).
pub struct GpuLock(#[allow(dead_code)] std::fs::File);

pub fn gpu_lock(a: &Args) -> Result<Option<GpuLock>> {
    #[cfg(unix)]
    extern "C" {
        fn flock(fd: i32, op: i32) -> i32;
    }
    let Some(path) = a.flag("lock") else { return Ok(None) };
    let f = std::fs::OpenOptions::new().create(true).append(true).open(path).map_err(|e| Error(format!("{path}: {e}")))?;
    // SAFETY: a descriptor we hold open; LOCK_EX (2) waits for the lock, released when the file is closed.
    #[cfg(unix)]
    if unsafe { flock(std::os::unix::io::AsRawFd::as_raw_fd(&f), 2) } != 0 {
        return err(format!("{path}: flock: {}", std::io::Error::last_os_error()));
    }
    #[cfg(windows)]
    {
        err(format!("{path}: --lock takes flock, a Linux call ({} unlocked)", f.metadata().map_or(0, |m| m.len())))
    }
    #[cfg(not(windows))]
    Ok(Some(GpuLock(f)))
}

fn layout_flag(a: &Args, name: &str, d: Layout) -> Result<Layout> {
    match a.flag(name) {
        None => Ok(d),
        Some(s) => Layout::by_name(s).ok_or_else(|| Error(format!("--{name} {s}: a layout's name (mma, i6, i5, i4)"))),
    }
}

/// `qxlayers`: see the module's comment. Returns whether every stage is within its tolerance.
pub fn layers(model: &Path, refd: &Path, a: &Args) -> Result<bool> {
    let c = ConfigQx::load(model)?;
    let ids: Vec<u32> = i32s(refd, "ids.bin")?.into_iter().map(|x| x as u32).collect();
    let t = ids.len();
    let (hid, hw, k) = (c.hidden, c.hc * c.hidden, c.top_k);
    let (first, end) = match a.flag("layers") {
        Some(r) => {
            let (x, y) = r.split_once(':').ok_or_else(|| Error("--layers A:B".into()))?;
            (x.parse::<usize>().map_err(|_| Error("--layers A:B".into()))?, y.parse::<usize>().map_err(|_| Error("--layers A:B".into()))?.min(c.layers))
        }
        None => (0, c.layers),
    };
    let (layout, level) = (layout_flag(a, "layout", Layout::Tiered)?, layout_flag(a, "level", Layout::Int6)?);
    let tol: f64 = a.num("tol", 2e-2f64)?;
    // (the experts are held at `--level` where the reference has them in bf16: their output differs by the level's rounding, about 2% at six bits)
    let tol_x: f64 = a.num("tol-experts", 4e-2f64)?;
    let make = gpu_experts(level);
    let mut all_ok = true;
    for l in first..end {
        let t0 = std::time::Instant::now();
        let o = OptsQx { ctx: t + 64, chunk: t, layout, only: Some(vec![l]), chunked: true, ..Default::default() };
        let mut e = EngineQx::new(model, &o, Some(&make))?;
        let _lock = gpu_lock(a)?; // (the layer loaded: the GPU's work alone under the machine's lock)
        println!("layer {l} ({:?}): loaded in {:.1} s; {}", c.kinds[l], t0.elapsed().as_secs_f64(), e.describe());
        let x_in: Vec<u16> =
            if l == 0 { words(refd, "embed.bin")?.chunks(hid).flat_map(|r| r.repeat(c.hc)).collect() } else { words(refd, &format!("L{}.out.bin", l - 1))? };
        let r = |stage: &str| words(refd, &format!("L{l}.{stage}.bin"));
        e.reset()?;
        put(&e, &e.pos, &(0..t as i32).collect::<Vec<_>>())?;
        let ly = e.w.layers[l].as_ref().ok_or_else(|| Error(format!("layer {l} not loaded")))?;
        let mut ok = true;
        // PLE: the n-gram rows (the hashes and the table's rows, on the host), then the layer's addition to the streams
        let mut x0 = x_in.clone();
        if let Some(p) = &ly.ple {
            let g = e.w.ngram.as_ref().ok_or_else(|| Error("a PLE layer without its table".into()))?;
            let want_ids: Vec<i64> = std::fs::read(refd.join(format!("L{l}.ng_ids.bin")))
                .map_err(|x| Error(x.to_string()))?
                .as_chunks::<8>()
                .0
                .iter()
                .map(|x| i64::from_le_bytes(*x))
                .collect();
            let got_ids: Vec<i64> = g.contexts([c.eos_ngram; 2], &ids).into_iter().flat_map(|(t0, c1, c2)| g.ids(t0, c1, c2)).map(|x| x as i64).collect();
            let same = got_ids.iter().zip(&want_ids).filter(|(x, y)| x == y).count();
            println!(
                "  {:<28} {same} of {} the reference's{}",
                "n-gram rows (hashes)",
                want_ids.len(),
                if same == want_ids.len() && got_ids.len() == want_ids.len() { "  ok" } else { "  OFF" }
            );
            ok &= same == want_ids.len() && got_ids.len() == want_ids.len();
            let mut rows = vec![0u8; t * g.heads * g.row * 2];
            g.gather([c.eos_ngram; 2], &ids, &mut rows);
            let rows16: Vec<u16> = rows.as_chunks::<2>().0.iter().map(|x| u16::from_le_bytes(*x)).collect();
            ok &= stage("n-gram embedding", &widen(&rows16), &widen(&r("ng_emb")?), 0.0);
            put(&e, &e.ple_e, &rows16)?;
            put(&e, &e.x, &x_in)?;
            // SAFETY: live buffers of the engine, sized for a chunk of t rows; synchronized before the reads.
            unsafe { e.ple_half(p, t)? };
            let want: Vec<f32> = x_in.iter().zip(r("ple")?).map(|(&x, p)| widen(&[bf16(widen(&[x])[0] + widen(&[p])[0])])[0]).collect();
            ok &= stage("PLE (streams after it)", &read_bf16(&e, &e.x, t * hw)?, &want, tol);
            x0 = want.iter().map(|&v| bf16(v)).collect();
        }
        let s = e.dev.stream;
        // SAFETY: as above: every call on the engine's live buffers sized for t rows, each stage's inputs written before it and its outputs read after a synchronize.
        unsafe {
            // the token mixer's Gated Residual on the layer's input
            put(&e, &e.x, &x0)?;
            e.gr_mix(&ly.gr[0], e.x.ptr, t, e.h.ptr, e.inj[0].ptr)?;
            ok &= stage("attn Gated Residual: input", &read_bf16(&e, &e.h, t * hid)?, &widen(&r("a_in")?), tol);
            ok &= stage("attn Gated Residual: weights", &read_f32(&e, &e.inj[0], t * c.hc)?, &f32s(refd, &format!("L{l}.a_inj.bin"))?, tol);
            // (a decode step's few rows: the fused launches where the bottleneck is held as int8 rows)
            if matches!(ly.gr[0].w, crate::model_qwen4x::GrW::Q { .. }) {
                let m = t.min(8);
                e.gr_mix(&ly.gr[0], e.x.ptr, m, e.h.ptr, e.inj[0].ptr)?;
                ok &= stage("attn Gated Residual: fused, input", &read_bf16(&e, &e.h, m * hid)?, &widen(&r("a_in")?[..m * hid]), tol);
                ok &= stage(
                    "attn Gated Residual: fused, weights",
                    &read_f32(&e, &e.inj[0], m * c.hc)?,
                    &f32s(refd, &format!("L{l}.a_inj.bin"))?[..m * c.hc],
                    tol,
                );
            }
            // the token mixer on the reference's input
            put(&e, &e.h, &r("a_in")?)?;
            let name = if ly.kind == QxKind::Gdn { "GDN" } else { "attention" };
            match ly.kind {
                QxKind::Gdn | QxKind::Attn => e.mixer_only(ly, t, e.pos.ptr, 0)?,
            }
            if ly.kind == QxKind::Attn && refd.join(format!("L{l}.sel.bin")).exists() && t > c.ix_tokens() {
                let tm = c.ix_tokens();
                let want = i32s(refd, &format!("L{l}.sel.bin"))?;
                e.dev.sync()?;
                let (mut got, mut cnt) = (vec![0i32; t * tm], vec![0i32; t]);
                e.sel.read(0, &mut got)?;
                e.cnt.read(0, &mut cnt)?;
                let same = (0..t).filter(|&i| got[i * tm..i * tm + cnt[i] as usize].iter().eq(want[i * tm..(i + 1) * tm].iter().filter(|&&x| x >= 0))).count();
                // (a block whose score is within the roundings of the next one's may be picked instead of it: the rows that pick other tokens are left out of the output's comparison)
                let differ: Vec<usize> =
                    (0..t).filter(|&i| !got[i * tm..i * tm + cnt[i] as usize].iter().eq(want[i * tm..(i + 1) * tm].iter().filter(|&&x| x >= 0))).collect();
                println!("  {:<28} {same} of {t} rows pick the reference's tokens{}", "indexer", if same * 100 >= t * 95 { "  ok" } else { "  OFF" });
                for &i in differ.iter().take(2) {
                    println!(
                        "    row {i}: engine {:?}, reference {:?}",
                        &got[i * tm..i * tm + cnt[i] as usize],
                        want[i * tm..(i + 1) * tm].iter().filter(|&&x| x >= 0).collect::<Vec<_>>()
                    );
                }
                ok &= same * 100 >= t * 95;
                let rows: Vec<usize> = (0..t).filter(|i| !differ.contains(i)).collect();
                let pick = |v: &[f32]| rows.iter().flat_map(|&row| v[row * hid..(row + 1) * hid].to_vec()).collect::<Vec<f32>>();
                ok &= stage(&format!("{name}: output (same picks)"), &pick(&read_bf16(&e, &e.o, t * hid)?), &pick(&widen(&r("a_out")?)), tol);
            } else {
                ok &= stage(&format!("{name}: output"), &read_bf16(&e, &e.o, t * hid)?, &widen(&r("a_out")?), tol);
            }
            // the streams' update from the reference's output and weights
            put(&e, &e.x, &x0)?;
            put(&e, &e.o, &r("a_out")?)?;
            put(&e, &e.inj[0], &f32s(refd, &format!("L{l}.a_inj.bin"))?)?;
            e.kq.hc_post(s, e.x.ptr, e.o.ptr, e.inj[0].ptr, t, hid)?;
            ok &= stage("streams after attn", &read_bf16(&e, &e.x, t * hw)?, &widen(&r("mid")?), tol);
            // the experts' Gated Residual, the router, the experts, the update
            put(&e, &e.x, &r("mid")?)?;
            e.gr_mix(&ly.gr[1], e.x.ptr, t, e.h.ptr, e.inj[1].ptr)?;
            ok &= stage("mlp Gated Residual: input", &read_bf16(&e, &e.h, t * hid)?, &widen(&r("m_in")?), tol);
            ok &= stage("mlp Gated Residual: weights", &read_f32(&e, &e.inj[1], t * c.hc)?, &f32s(refd, &format!("L{l}.m_inj.bin"))?, tol);
            put(&e, &e.h, &r("m_in")?)?;
            ly.moe.as_ref().ok_or_else(|| Error("no experts".into()))?.forward(&e, e.h.ptr, t, e.o.ptr)?;
            let (gi, gw) = QxMoe::routed(&e, t, k)?;
            let (wi, ww) = (i32s(refd, &format!("L{l}.r_ids.bin"))?, f32s(refd, &format!("L{l}.r_w.bin"))?);
            let mut same = 0;
            let mut wd = 0f32;
            for row in 0..t {
                let mut a1: Vec<(i64, f32)> = (0..k).map(|j| (gi[row * k + j], gw[row * k + j])).collect();
                let mut b1: Vec<(i64, f32)> = (0..k).map(|j| (wi[row * k + j] as i64, ww[row * k + j])).collect();
                a1.sort_by_key(|p| p.0);
                b1.sort_by_key(|p| p.0);
                if a1.iter().map(|p| p.0).eq(b1.iter().map(|p| p.0)) {
                    same += 1;
                    wd = a1.iter().zip(&b1).fold(wd, |m, (x, y)| m.max((x.1 - y.1).abs()));
                }
            }
            println!(
                "  {:<28} {same} of {t} rows choose the reference's experts; their weights within {wd:.2e}{}",
                "router",
                if same * 100 >= t * 97 { "  ok" } else { "  OFF" }
            );
            ok &= same * 100 >= t * 97;
            let (go, wo) = (read_bf16(&e, &e.o, t * hid)?, widen(&r("m_out")?));
            let keep: Vec<usize> = (0..t)
                .filter(|&row| {
                    let mut a1: Vec<i64> = gi[row * k..(row + 1) * k].to_vec();
                    let mut b1: Vec<i64> = wi[row * k..(row + 1) * k].iter().map(|&x| x as i64).collect();
                    a1.sort_unstable();
                    b1.sort_unstable();
                    a1 == b1
                })
                .collect();
            let pick = |v: &[f32]| keep.iter().flat_map(|&row| v[row * hid..(row + 1) * hid].to_vec()).collect::<Vec<f32>>();
            ok &= stage("experts: output (same picks)", &pick(&go), &pick(&wo), tol_x);
            put(&e, &e.x, &r("mid")?)?;
            put(&e, &e.o, &r("m_out")?)?;
            put(&e, &e.inj[1], &f32s(refd, &format!("L{l}.m_inj.bin"))?)?;
            e.kq.hc_post(s, e.x.ptr, e.o.ptr, e.inj[1].ptr, t, hid)?;
            ok &= stage("streams after the layer", &read_bf16(&e, &e.x, t * hw)?, &widen(&r("out")?), tol);
        }
        // the whole layer from its input, the engine's own values throughout
        e.zero_states()?;
        put(&e, &e.x, &x_in)?;
        if ly.ple.is_some() {
            let g = e.w.ngram.as_ref().ok_or_else(|| Error("no table".into()))?;
            let mut rows = vec![0u8; t * g.heads * g.row * 2];
            g.gather([c.eos_ngram; 2], &ids, &mut rows);
            put(&e, &e.ple_e, &rows)?;
        }
        // SAFETY: as above.
        unsafe {
            if let Some(p) = &ly.ple {
                e.ple_half(p, t)?;
            }
            e.mixer_half(ly, t, e.pos.ptr, 0)?;
            e.mlp_half(ly, t)?;
        }
        ok &= stage("whole layer (its own values)", &read_bf16(&e, &e.x, t * hw)?, &widen(&r("out")?), tol_x);
        println!("layer {l}: {} in {:.1} s", if ok { "within tolerance" } else { "OFF" }, t0.elapsed().as_secs_f64());
        all_ok &= ok;
        drop(e);
    }
    Ok(all_ok)
}

/// KL(p || q) of the softmaxes of two logit rows, and whether their argmaxes agree (counted only where the reference's best logit leads its second by more than 0.1: a lead within the roundings is a coin's toss).
fn kl_row(want: &[f32], got: &[f32]) -> (f64, Option<bool>) {
    let lse = |v: &[f32]| {
        let m = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
        m + v.iter().map(|&x| (x as f64 - m).exp()).sum::<f64>().ln()
    };
    let (lp, lq) = (lse(want), lse(got));
    let kl = want
        .iter()
        .zip(got)
        .map(|(&p, &q)| {
            let lpp = p as f64 - lp;
            lpp.exp() * (lpp - (q as f64 - lq))
        })
        .sum::<f64>();
    let am = |v: &[f32]| v.iter().enumerate().fold((0, f32::NEG_INFINITY), |b, (i, &x)| if x > b.1 { (i, x) } else { b }).0;
    let best = am(want);
    let second = want.iter().enumerate().filter(|&(i, _)| i != best).fold(f32::NEG_INFINITY, |m, (_, &x)| m.max(x));
    (kl.max(0.0), (want[best] - second > 0.1).then(|| best == am(got)))
}

/// `qxtoy`: see the module's comment.
pub fn toy(model: &Path, refd: &Path, a: &Args) -> Result<bool> {
    let c = ConfigQx::load(model)?;
    let ids: Vec<u32> = i32s(refd, "ids.bin")?.into_iter().map(|x| x as u32).collect();
    let t = ids.len();
    let v = c.vocab;
    let want = f32s(refd, "logits.bin")?;
    if want.len() != t * v {
        return err(format!("logits.bin: {} values, where {t} positions of {v} are wanted", want.len()));
    }
    let (chunk, prefill): (usize, usize) = (a.num("chunk", 64usize)?, a.num("prefill", t * 3 / 4)?);
    let (layout, level) = (layout_flag(a, "layout", Layout::Tiered)?, layout_flag(a, "level", Layout::Int6)?);
    let make = gpu_experts(level);
    let o = OptsQx { ctx: t + 64, chunk, layout, chunked: true, ..Default::default() };
    let mut e = EngineQx::new(model, &o, Some(&make))?;
    println!("{}", e.describe());
    let _lock = gpu_lock(a)?;
    e.reset()?;
    let mut got: Vec<Vec<f32>> = Vec::new();
    let mut at = 0;
    while at < prefill {
        let n = chunk.min(prefill - at);
        e.prefill(&ids[at..at + n])?;
        let h = e.final_rows(n)?;
        let lg = widen(&e.head_rows(&h)?);
        got.extend(lg.chunks(v).map(|r| r.to_vec()));
        at += n;
    }
    for &tok in &ids[prefill..t] {
        e.step(tok)?;
        got.push(e.logits()?);
    }
    got.truncate(t);
    // (KL sum and largest, top-1 agreements and positions with a clear best token) for the prompt's rows and the decode steps'
    let mut acc = [(0f64, 0f64, 0usize, 0usize); 2];
    for (i, g) in got.iter().enumerate() {
        let (kl, ag) = kl_row(&want[i * v..(i + 1) * v], g);
        let a = &mut acc[(i >= prefill) as usize];
        a.0 += kl;
        a.1 = a.1.max(kl);
        if let Some(x) = ag {
            a.2 += x as usize;
            a.3 += 1;
        }
    }
    let rows = [prefill, t - prefill];
    for (j, what) in ["prompt", "decode"].iter().enumerate() {
        let a = acc[j];
        println!(
            "{what} ({} positions{}): KL mean {:.2e} max {:.2e}; top-1 {} of the {} with a clear best token",
            rows[j],
            if j == 0 { format!(" in chunks of {chunk}") } else { String::new() },
            a.0 / rows[j].max(1) as f64,
            a.1,
            a.2,
            a.3
        );
    }
    let ok = acc.iter().all(|a| a.1 < 0.02 && a.2 * 100 >= a.3 * 97);
    println!("{}", if ok { "the model is the reference's" } else { "OFF" });
    Ok(ok)
}

/// `glyd-engine qxcb MODEL PACK REF --layers L,L [--rows N] [--slots N]`: a layer's experts from the codebook pack through the expert cache, on the first N rows of the reference's experts' input: every pair on the GPU (the
/// cache's copies, kernels/qwen4x.cu) against every pair on the host (src/cpu_cb.rs, the same codes), both against the reference's bf16 experts (the pack's rounding), and the router's choices.
pub fn cb(model: &Path, pack_dir: &Path, refd: &Path, a: &Args) -> Result<bool> {
    use crate::moe_glm_cb::CbPack;
    use crate::moe_qwen4x::{factory, QxCbShared};
    use crate::offload::{ExpertStore, Mode, Offload};
    let c = ConfigQx::load(model)?;
    let pack = CbPack::open_dims(pack_dir, c.hidden, c.inter)?;
    let layers: Vec<usize> =
        a.flag("layers").unwrap_or("0").split(',').map(|x| x.trim().parse().map_err(|_| Error(format!("--layers {x}")))).collect::<Result<_>>()?;
    let t_all = i32s(refd, "ids.bin")?.len();
    let t: usize = a.num("rows", 32usize)?.min(t_all);
    let hid = c.hidden;
    let mut all_ok = true;
    for &l in &layers {
        let m_in = words(refd, &format!("L{l}.m_in.bin"))?;
        let want = widen(&words(refd, &format!("L{l}.m_out.bin"))?[..t * hid]);
        let mut outs = Vec::new();
        for mode in [Mode::Copy, Mode::Cpu] {
            let store = ExpertStore::new();
            store.set_offload(Offload { mode, threads: a.num("cpu-threads", 16usize)?, ..Offload::default() });
            let common = QxCbShared::new();
            let make = factory(&pack, a.num("slots", 64usize)?, &store, &common);
            let o = OptsQx { ctx: t + 64, chunk: t, layout: Layout::Tiered, only: Some(vec![l]), chunked: true, ..Default::default() };
            let e = EngineQx::new(model, &o, Some(&make))?;
            let _lock = gpu_lock(a)?;
            let ly = e.w.layers[l].as_ref().ok_or_else(|| Error("not loaded".into()))?;
            put(&e, &e.h, &m_in[..t * hid])?;
            let t0 = std::time::Instant::now();
            // SAFETY: live buffers of the engine sized for t rows; read after a synchronize.
            unsafe { ly.moe.as_ref().ok_or_else(|| Error("no experts".into()))?.forward(&e, e.h.ptr, t, e.o.ptr)? };
            e.dev.sync()?;
            println!(
                "layer {l}, {} rows, mode {}: {:.1} ms; {}",
                t,
                mode.name(),
                t0.elapsed().as_secs_f64() * 1e3,
                store.report().unwrap_or_default().replace('\n', " | ")
            );
            outs.push(read_bf16(&e, &e.o, t * hid)?);
        }
        let mut ok = stage("experts: GPU against host", &outs[0], &outs[1], 1e-2);
        stage("experts: GPU against bf16", &outs[0], &want, 1.0);
        stage("experts: host against bf16", &outs[1], &want, 1.0);
        ok &= outs.iter().all(|o| o.iter().all(|v| v.is_finite()));
        all_ok &= ok;
    }
    Ok(all_ok)
}

/// `qxbench [--copies N] [--reps N]`: the few-rows kernels of the Gated Residual and the shared expert timed on random weights of the real sizes (N copies of each matrix in turn, 32 by default: more than the L2
/// cache holds, as the layers' are), for 1, 3 and 8 rows: microseconds a launch and the weights' GB/s. Under 1 GB of the GPU's memory.
/// `qxprod [--fmt 0|1|2|4] [--overlap K] [--rows 1,2,4,8] [--reps R] [--dump FILE]`: the experts' first and second products (qx_cb_up, qx_cb_down) on places of one format (0: two-bit, 1: three, 2: four, 4: format
/// q), each call 10 places a row, a row sharing K of them with the row before (a verification's rows), every call on places after the last call's (1.2 GB and more in all, not the L2's), timed over R calls back to
/// back: microseconds a call and the GB/s of the call's distinct places' bytes, for each count of rows. `--dump`: the first call's sums of each count (first, then second products) appended to FILE, to compare builds.
/// For 2 to 8 rows, the second products on the tensor cores too (`KernQc::down_t`), their pairs' sums against qx_cb_down's (the norm of the difference over the norm).
pub fn bench_products(a: &Args) -> Result<()> {
    use crate::dev::{Buf, Dev, Timer};
    let dev = Dev::new(0)?;
    let kq = crate::moe_qwen4x::KernQc::load(&dev)?;
    let (fmt, overlap, reps): (usize, usize, usize) = (a.num("fmt", 0usize)?, a.num("overlap", 0usize)?, a.num("reps", 200usize)?);
    let rows: Vec<usize> =
        a.flag("rows").unwrap_or("1,2,4,8").split(',').map(|x| x.parse::<usize>().map_err(|_| Error(format!("--rows {x}")))).collect::<Result<_>>()?;
    let (h, i, k) = (2560usize, 640usize, 10usize);
    if ![0, 1, 2, 4].contains(&fmt) || overlap >= k || rows.iter().any(|&n| n == 0) {
        return err(format!("--fmt {fmt} --overlap {overlap} --rows {rows:?}: formats 0, 1, 2, 4, fewer than {k} shared, a row or more"));
    }
    // a row of 2,560 codes and its codebook: bytes (qc_row_bytes, qc_book_bytes)
    let (rb, kb) = ([640usize, 960, 1280, 512, 640][fmt], [8usize, 16, 32, 6, 2][fmt]);
    let (cstride, bstride) = ((3 * i * rb).next_multiple_of(256), (3 * i * kb + 16).next_multiple_of(16));
    let places = (1_500_000_000 / cstride).min(1200);
    let codes = Buf::new(places * cstride)?;
    let books = Buf::new(places * bstride)?;
    let mut seed = 0x9e37_79b9u64;
    let mut rnd = |m: usize| -> Vec<u8> {
        (0..m)
            .map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                seed as u8
            })
            .collect()
    };
    for pl in 0..places {
        codes.write(pl * cstride, &rnd(cstride))?;
    }
    // codebooks of small bf16 values (format q: a row's scale), the place's formats in its last byte
    let mut bk: Vec<u8> =
        (0..places * bstride / 2).flat_map(|j| half_bits(((j % 7) as f32 - 3.0) / 64.0 + if fmt == 4 { 0.1 } else { 0.0 }).to_le_bytes()).collect();
    for pl in 0..places {
        bk[pl * bstride + bstride - 1] = (fmt | fmt << 4) as u8;
    }
    books.write(0, &bk)?;
    let mixed = fmt != 0;
    let nmax = rows.iter().copied().max().unwrap_or(1);
    let xr = Buf::new(nmax * h * 4)?;
    let xv: Vec<f32> = (0..nmax * h).map(|j| ((j % 13) as f32 - 6.0) / 8.0 + ((j * 7919 % 101) as f32 - 50.0) / 400.0).collect();
    xr.write(0, &xv.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>())?;
    let gu = Buf::zeroed(nmax * k * 2 * i * 4)?;
    let part = Buf::zeroed(nmax * k * 16 * h * 4)?;
    let s = dev.stream;
    let t = Timer::new()?;
    let pp = [codes.ptr, books.ptr];
    let mut dump = Vec::new();
    println!("  format {fmt}, rows sharing {overlap} of {k} places with the row before; {places} places of {:.2} MB", (cstride + bstride) as f64 / 1e6);
    for &n in &rows {
        let (pairs, split) = (n * k, crate::moe_qwen4x::runs_for(n * k));
        // row r's places: the call's first + r (k - overlap) + 0..k
        let span = k + (n - 1) * (k - overlap);
        let calls = places / span;
        let ways: Vec<Buf> = (0..calls)
            .map(|c| {
                let w: Vec<i64> = (0..pairs).map(|p| (c * span + (p / k) * (k - overlap) + p % k) as i64).collect();
                let b = Buf::new(pairs * 8)?;
                b.write(0, &w)?;
                Ok(b)
            })
            .collect::<Result<_>>()?;
        let time = |f: &mut dyn FnMut(usize) -> Result<()>| -> Result<f64> {
            f(0)?;
            dev.sync()?;
            t.start(s)?;
            for r in 0..reps {
                f(r % calls)?;
            }
            Ok(t.stop(s)? as f64 * 1e3 / reps as f64)
        };
        // SAFETY: buffers of the sizes the kernels read and write.
        let (up, down) = unsafe {
            (
                time(&mut |c| kq.up(s, pp, [cstride, bstride], ways[c].ptr, xr.ptr, gu.ptr, k, pairs, mixed))?,
                time(&mut |c| kq.down(s, pp, [cstride, bstride], ways[c].ptr, gu.ptr, part.ptr, split, pairs, mixed))?,
            )
        };
        let (mu, md) = ((span * 2 * i * rb) as f64 / 1e6, (span * i * rb) as f64 / 1e6);
        println!(
            "  {n} rows {span:>3} places  first products {up:>7.1} us {:>5.0} GB/s  second {down:>7.1} us {:>5.0} GB/s  both {:>5.0} GB/s",
            mu / up * 1e3,
            md / down * 1e3,
            (mu + md) / (up + down) * 1e3
        );
        if a.has("dump") {
            // SAFETY: as above.
            unsafe {
                kq.up(s, pp, [cstride, bstride], ways[0].ptr, xr.ptr, gu.ptr, k, pairs, mixed)?;
                kq.down(s, pp, [cstride, bstride], ways[0].ptr, gu.ptr, part.ptr, split, pairs, mixed)?;
            }
            dev.sync()?;
            let (mut g, mut q) = (vec![0u8; pairs * 2 * i * 4], vec![0u8; pairs * split * h * 4]);
            gu.read(0, &mut g)?;
            part.read(0, &mut q)?;
            dump.extend(g);
            dump.extend(q);
        }
        if (2..=8).contains(&n) {
            // the second products on the tensor cores (from the first products' sums): their time, and each pair's sums against qx_cb_down's (its runs added)
            let part1 = Buf::zeroed(pairs * h * 4)?;
            // SAFETY: as above.
            let dt = unsafe { time(&mut |c| kq.down_t(s, pp, [cstride, bstride], ways[c].ptr, gu.ptr, part1.ptr, pairs, mixed))? };
            unsafe {
                kq.up(s, pp, [cstride, bstride], ways[0].ptr, xr.ptr, gu.ptr, k, pairs, mixed)?;
                kq.down(s, pp, [cstride, bstride], ways[0].ptr, gu.ptr, part.ptr, split, pairs, mixed)?;
                kq.down_t(s, pp, [cstride, bstride], ways[0].ptr, gu.ptr, part1.ptr, pairs, mixed)?;
            }
            dev.sync()?;
            let (mut q, mut q1) = (vec![0f32; pairs * split * h], vec![0f32; pairs * h]);
            part.read(0, &mut q)?;
            part1.read(0, &mut q1)?;
            let (mut d2, mut n2) = (0f64, 0f64);
            for p in 0..pairs {
                for o in 0..h {
                    let want: f64 = (0..split).map(|r| q[(p * split + r) * h + o] as f64).sum();
                    d2 += (q1[p * h + o] as f64 - want).powi(2);
                    n2 += want * want;
                }
            }
            println!(
                "  {n} rows {span:>3} places  second on tensor cores {dt:>7.1} us {:>5.0} GB/s  (sums against qx_cb_down's: {:.2e} of their norm)",
                md / dt * 1e3,
                (d2 / n2.max(1e-30)).sqrt()
            );
        }
    }
    if let Some(path) = a.flag("dump") {
        std::fs::write(path, &dump).map_err(|e| Error(format!("{path}: {e}")))?;
    }
    Ok(())
}

pub fn bench(a: &Args) -> Result<()> {
    use crate::dev::{Buf, Dev, Ev};
    use crate::launch;
    let dev = Dev::new(0)?;
    let kq = crate::model_qwen4x::KernQx::load(&dev)?;
    let (copies, reps): (usize, usize) = (a.num("copies", 32usize)?, a.num("reps", 200usize)?);
    let (d, r, hs, si) = (2560usize, 320usize, 2560usize, 640usize);
    let w4 = 4 * d;
    let mut seed = 0x1234_5678u64;
    let mut rnd = |n: usize| -> Vec<u8> {
        (0..n)
            .map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                seed as u8
            })
            .collect()
    };
    // bf16 values in [-1, 1) from random bytes; int8 weights as they come
    let bf = |raw: Vec<u8>| -> Vec<u8> { raw.chunks(2).flat_map(|c| (half_bits((c[0] as f32 - 128.0) / 128.0)).to_le_bytes()).collect() };
    let up = |v: &[u8]| -> Result<Buf> {
        let b = Buf::new(v.len())?;
        b.write(0, v)?;
        Ok(b)
    };
    if a.has("dec") {
        // a decode step's indexer scores, picks and attention over picked tokens of one attention layer at a position (--pos, 32,000 by default) of a 33,000-token context, n rows (--n, 1): microseconds a launch
        let kg = crate::model_glm::KernGlm::load(&dev)?;
        let (ctx, n, at): (usize, usize, usize) = (a.num("ctx", 33000usize)?, a.num("n", 1usize)?, a.num("pos", 32000usize)?);
        let (ixh, ixd, kp, budget, nq, nk, hd) = (4usize, 128usize, 4usize, 2048usize, 24usize, 2usize, 256usize);
        let (pmax, ksel, tmax) = (ctx / kp, budget / kp, budget + kp - 1);
        let qd = 2 * nq * hd + 2 * nk * hd;
        let pk = up(&bf(rnd((pmax + 1) * ixd * 2)))?;
        let iq = up(&bf(rnd(n * ixh * ixd * 2)))?;
        let ones = up(&(0..8).flat_map(|_| half_bits(1.0).to_le_bytes()).collect::<Vec<u8>>())?;
        let pos = up(&(0..n).flat_map(|i| ((at + i) as i32).to_le_bytes()).collect::<Vec<u8>>())?;
        let (scores, sel, cnt) = (Buf::zeroed(n * pmax * 4)?, Buf::zeroed(n * tmax * 4)?, Buf::zeroed(n * 4)?);
        let (kc, vc) = (up(&bf(rnd(ctx * nk * hd * 2)))?, up(&bf(rnd(ctx * nk * hd * 2)))?);
        let (qo, mix, core) = (up(&bf(rnd(n * nq * hd * 2)))?, up(&bf(rnd(n * qd * 2)))?, Buf::zeroed(n * nq * hd * 2)?);
        let part = Buf::zeroed(8 * nq * 64 * 258 * 4)?;
        let s = dev.stream;
        let (e0, e1) = (Ev::new()?, Ev::new()?);
        // (20 calls captured in a graph and the graph timed, as the decode step's graphs run them: a launch from the host takes longer than these kernels)
        let time = |what: &str, f: &mut dyn FnMut() -> Result<()>| -> Result<()> {
            f()?;
            let gr = dev.capture(|| {
                for _ in 0..20 {
                    f()?;
                }
                Ok(())
            })?;
            gr.launch(s)?;
            e0.record(s)?;
            let k = reps.div_ceil(20);
            for _ in 0..k {
                gr.launch(s)?;
            }
            e1.record(s)?;
            dev.sync()?;
            println!("  dec pos {at} n {n}  {what:<20} {:>7.1} us", e1.since(&e0)? as f64 * 1e3 / (k * 20) as f64);
            Ok(())
        };
        // SAFETY: live buffers of the sizes the kernels read and write.
        unsafe {
            time("indexer scores", &mut || {
                kg.idx_score(
                    s,
                    iq.ptr,
                    (ixh * ixd) as i32,
                    ones.ptr,
                    0,
                    pk.ptr,
                    pos.ptr,
                    scores.ptr,
                    pmax as i32,
                    ixh as i32,
                    ixd as i32,
                    kp as i32,
                    ksel as i32,
                    1.0 / (ixd as f32).sqrt(),
                    1.0,
                    n as u32,
                )
            })?;
            time("select", &mut || kg.select(s, scores.ptr, pmax as i32, pos.ptr, sel.ptr, cnt.ptr, tmax as i32, kp as i32, ksel as i32, true, n as u32))?;
            let sc2 = Buf::zeroed(n * pmax * 4)?;
            time("indexer scores (tc)", &mut || kq.ix_score(s, iq.ptr, pk.ptr, pos.ptr, sc2.ptr, n, pmax, ksel, pmax, 1.0 / (ixd as f32).sqrt()))?;
            {
                dev.sync()?;
                let (mut x, mut y) = (vec![0f32; n * pmax], vec![0f32; n * pmax]);
                scores.read(0, &mut x)?;
                sc2.read(0, &mut y)?;
                let p_last = (at + 1) / kp;
                let (worst, big) = (0..n)
                    .flat_map(|i| (0..p_last.min(pmax)).map(move |p| i * pmax + p))
                    .fold((0f32, 0f32), |(w, b), k| (w.max((x[k] - y[k]).abs()), b.max(x[k].abs())));
                println!("  dec: the two indexers' scores differ by up to {worst:.3e} (scores up to {big:.3e})");
            }
            let (sel2, cnt2) = (Buf::zeroed(n * tmax * 4)?, Buf::zeroed(n * 4)?);
            time("select (qx)", &mut || kq.select(s, scores.ptr, pmax, pos.ptr, sel2.ptr, cnt2.ptr, tmax, kp, ksel, n))?;
            // the two selects alike: on the scores, and on scores of many ties (eighths up to 5)
            let same = |what: &str| -> Result<()> {
                dev.sync()?;
                let (mut x, mut y, mut cx, mut cy) = (vec![0i32; n * tmax], vec![0i32; n * tmax], vec![0i32; n], vec![0i32; n]);
                sel.read(0, &mut x)?;
                sel2.read(0, &mut y)?;
                cnt.read(0, &mut cx)?;
                cnt2.read(0, &mut cy)?;
                let ok = cx == cy && (0..n).all(|i| x[i * tmax..i * tmax + cx[i] as usize] == y[i * tmax..i * tmax + cx[i] as usize]);
                println!("  dec: the two selects alike ({what}): {ok} (counts {cx:?})");
                Ok(())
            };
            same("the scores")?;
            let ties: Vec<u8> = rnd(n * pmax).iter().flat_map(|&b| ((b % 41) as f32 / 8.0).to_le_bytes()).collect();
            scores.write(0, &ties)?;
            kg.select(s, scores.ptr, pmax as i32, pos.ptr, sel.ptr, cnt.ptr, tmax as i32, kp as i32, ksel as i32, true, n as u32)?;
            kq.select(s, scores.ptr, pmax, pos.ptr, sel2.ptr, cnt2.ptr, tmax, kp, ksel, n)?;
            same("ties")?;
            let zeros = vec![0u8; n * pmax * 4];
            scores.write(0, &zeros)?;
            kg.select(s, scores.ptr, pmax as i32, pos.ptr, sel.ptr, cnt.ptr, tmax as i32, kp as i32, ksel as i32, true, n as u32)?;
            kq.select(s, scores.ptr, pmax, pos.ptr, sel2.ptr, cnt2.ptr, tmax, kp, ksel, n)?;
            same("all zeros")?;
            // (the picks back to the real scores' for the attention)
            kq.ix_score(s, iq.ptr, pk.ptr, pos.ptr, scores.ptr, n, pmax, ksel, pmax, 1.0 / (ixd as f32).sqrt())?;
            kq.select(s, scores.ptr, pmax, pos.ptr, sel.ptr, cnt.ptr, tmax, kp, ksel, n)?;
            for sp in [16usize, 32, 64] {
                time(&format!("attention ({sp} splits)"), &mut || {
                    kq.attn_sel(s, qo.ptr, nq * hd, mix.ptr, qd, kc.ptr, vc.ptr, sel.ptr, cnt.ptr, tmax, core.ptr, nq * hd, nq, nk, part.ptr, sp, n)
                })?;
            }
            dev.sync()?;
            let mut want = vec![0u16; n * nq * hd];
            core.read(0, &mut want)?;
            let f = |x: u16| f32::from_bits((x as u32) << 16);
            for wpb in [4usize, 8] {
                time(&format!("attn tc ({wpb} warps)"), &mut || {
                    kq.attn_dec(s, qo.ptr, nq * hd, mix.ptr, qd, kc.ptr, vc.ptr, sel.ptr, cnt.ptr, tmax, core.ptr, nq * hd, part.ptr, wpb, n)
                })?;
                dev.sync()?;
                let mut got = vec![0u16; n * nq * hd];
                core.read(0, &mut got)?;
                let (sd, sq) = got.iter().zip(&want).fold((0f64, 0f64), |(d, q), (x, y)| (d + ((f(*x) - f(*y)) as f64).powi(2), q + (f(*y) as f64).powi(2)));
                println!("  dec: tc attention ({wpb} warps) against attn_sel: relative {:.3e}", (sd / sq.max(1e-30)).sqrt());
            }
        }
        return Ok(());
    }
    if a.has("router") {
        // a decode step's front (qx_router_rot: the router's logits, the picks, x' = R x) on n rows (--n, 4) of 512 experts: microseconds a launch, 40 launches captured in a graph, each on its own copy of the
        // router's rows (out of the L2, as a decode step's 48 layers read them)
        use crate::moe::{Act, MoeCfg, Router, Scoring};
        let kc = crate::moe_qwen4x::KernQc::load(&dev)?;
        let (n, e, h) = (a.num("n", 4usize)?, a.num("e", 512usize)?, 2560usize);
        let c = MoeCfg {
            experts: e,
            router_hidden: h,
            hidden: h,
            inter: 640,
            router: Router { scoring: Scoring::Softmax, top_k: a.num("k", 10usize)?, renorm: true, scale: 1.0, bf16_logits: true },
            act: Act::SwiGlu { limit: None },
        };
        let ws: Vec<Buf> = (0..40).map(|_| up(&bf(rnd(e * h * 2)))).collect::<Result<_>>()?;
        let (x, sg) = (up(&bf(rnd(n * h * 2)))?, up(&rnd(h))?);
        let local = up(&(0..e as i32).flat_map(|i| i.to_le_bytes()).collect::<Vec<u8>>())?;
        let (logits, ids, wts, xr, xrb, gids) = (
            Buf::zeroed(n * e * 4)?,
            Buf::zeroed(n * 10 * 8)?,
            Buf::zeroed(n * 10 * 4)?,
            Buf::zeroed(n * h * 4)?,
            Buf::zeroed(n * h * 2)?,
            Buf::zeroed(n * 10 * 8)?,
        );
        let s = dev.stream;
        let (e0, e1) = (Ev::new()?, Ev::new()?);
        // SAFETY: live buffers of the sizes the kernel reads and writes.
        let call = |i: usize| unsafe {
            kc.router_rot(s, x.ptr, h, ws[i].ptr, logits.ptr, n, &c, 0, ids.ptr, wts.ptr, sg.ptr, xr.ptr, xrb.ptr, local.ptr, gids.ptr, e, [0, 0, 0], 0, 0)
        };
        call(0)?;
        let gr = dev.capture(|| (0..40).try_for_each(call))?;
        gr.launch(s)?;
        e0.record(s)?;
        let k = reps.div_ceil(40);
        for _ in 0..k {
            gr.launch(s)?;
        }
        e1.record(s)?;
        dev.sync()?;
        let us = e1.since(&e0)? as f64 * 1e3 / (k * 40) as f64;
        println!("  router n {n}  {us:>7.1} us  {:>6.0} GB/s", (e * h * 2) as f64 / us / 1e3);
        return Ok(());
    }
    if a.has("gdn") {
        // the two-pass delta rule on a prompt slice of the engine's 1,024 rows, random inputs of the real sizes (16 key heads, 48 value heads): the chunks' part, then the scan, microseconds a call
        let (n, nk, nv) = (a.num("rows", 1024usize)?, 16usize, 48usize);
        let cd = 2 * nk * 128 + nv * 128;
        let xc = up(&bf(rnd(n * cd * 2)))?;
        let ba = up(&bf(rnd(n * 2 * nv * 2)))?;
        let alog = up(&(0..nv).flat_map(|i| ((i % 8) as f32 * 0.25 - 1.0).to_le_bytes()).collect::<Vec<u8>>())?;
        let dtb = up(&bf(rnd(nv * 2)))?;
        let (state, core) = (Buf::zeroed(nv * 128 * 128 * 4)?, Buf::zeroed(n * nv * 128 * 2)?);
        let s = dev.stream;
        let (e0, e1) = (Ev::new()?, Ev::new()?);
        if a.has("check") {
            // the two passes in slices of n rows against the one-pass kernel (Kern35::delta, fp32) over 4 n rows from a zero state: the outputs' and the final state's relative errors, and two runs alike
            let k35 = crate::kern35::Kern35::load(&dev)?;
            let m = 4 * n;
            let (xc4, ba4) = (up(&bf(rnd(m * cd * 2)))?, up(&bf(rnd(m * 2 * nv * 2)))?);
            let (st_r, out_r) = (Buf::zeroed(nv * 128 * 128 * 4)?, Buf::zeroed(m * nv * 128 * 2)?);
            // SAFETY: live buffers of the sizes the kernels read and write.
            unsafe { k35.delta(s, true, xc4.ptr, cd as i32, ba4.ptr, 2 * nv as i32, alog.ptr, dtb.ptr, st_r.ptr, out_r.ptr, m as i32, nk as i32, nv as i32)? };
            let read = |b: &Buf, bf16: bool| -> Result<Vec<f32>> {
                dev.sync()?;
                if bf16 {
                    let mut v = vec![0u16; b.bytes / 2];
                    b.read(0, &mut v)?;
                    Ok(v.iter().map(|&x| f32::from_bits((x as u32) << 16)).collect())
                } else {
                    let mut v = vec![0f32; b.bytes / 4];
                    b.read(0, &mut v)?;
                    Ok(v)
                }
            };
            let rel = |x: &[f32], y: &[f32]| {
                (x.iter().zip(y).map(|(p, q)| ((p - q) as f64).powi(2)).sum::<f64>() / y.iter().map(|q| (*q as f64).powi(2)).sum::<f64>().max(1e-30)).sqrt()
            };
            let (o_ref, s_ref) = (read(&out_r, true)?, read(&st_r, false)?);
            let mut runs = Vec::new();
            for _ in 0..2 {
                let (st, out) = (Buf::zeroed(nv * 128 * 128 * 4)?, Buf::zeroed(m * nv * 128 * 2)?);
                for i in (0..m).step_by(n) {
                    // SAFETY: as above.
                    unsafe {
                        kq.delta2(
                            s,
                            xc4.ptr + (i * cd * 2) as u64,
                            cd,
                            ba4.ptr + (i * 2 * nv * 2) as u64,
                            2 * nv,
                            alog.ptr,
                            dtb.ptr,
                            st.ptr,
                            out.ptr + (i * nv * 128 * 2) as u64,
                            n,
                            nk,
                            nv,
                        )?
                    };
                }
                runs.push((read(&out, true)?, read(&st, false)?));
            }
            println!(
                "  gdn check, {m} rows in slices of {n}: outputs relative {:.3e}, state relative {:.3e}; two runs alike: {}",
                rel(&runs[0].0, &o_ref),
                rel(&runs[0].1, &s_ref),
                runs[0] == runs[1]
            );
        }
        for (what, parts) in [("chunks' part", 1u8), ("scan", 2), ("both", 3)] {
            // SAFETY: live buffers of the sizes the kernels read and write.
            let call = || unsafe { kq.delta2_parts(s, xc.ptr, cd, ba.ptr, 2 * nv, alog.ptr, dtb.ptr, state.ptr, core.ptr, n, nk, nv, parts) };
            call()?;
            e0.record(s)?;
            for _ in 0..reps {
                call()?;
            }
            e1.record(s)?;
            dev.sync()?;
            println!("  gdn {n} rows  {what:<14} {:>7.1} us", e1.since(&e0)? as f64 * 1e3 / reps as f64);
        }
        return Ok(());
    }
    let x = up(&bf(rnd(8 * w4 * 2)))?;
    let wn = up(&bf(rnd(w4 * 2)))?;
    let injw = up(&bf(rnd(4 * w4 * 2)))?;
    let scale = |n: usize| -> Vec<u8> { (0..n).flat_map(|_| 0.01f32.to_le_bytes()).collect() };
    let dq: Vec<Buf> = (0..copies).map(|_| up(&rnd(r * w4))).collect::<Result<_>>()?;
    let uq: Vec<Buf> = (0..copies).map(|_| up(&rnd(w4 * r))).collect::<Result<_>>()?;
    let (ds, us) = (up(&scale(r))?, up(&scale(w4))?);
    let shq: Vec<Buf> = (0..copies).map(|_| up(&rnd(2 * si * hs))).collect::<Result<_>>()?;
    let sdq: Vec<Buf> = (0..copies).map(|_| up(&rnd(hs * si))).collect::<Result<_>>()?;
    let (shs, sds, gw) = (up(&scale(2 * si))?, up(&scale(hs))?, up(&bf(rnd(hs * 2)))?);
    let (rs, lo, inj, h) = (Buf::zeroed(8 * 4 * 4)?, up(&bf(rnd(8 * r * 2)))?, Buf::zeroed(8 * 4 * 4)?, Buf::zeroed(8 * d * 2)?);
    let (act, sg, sh) = (up(&bf(rnd(8 * si * 2)))?, up(&(0..8).flat_map(|_| 0.5f32.to_le_bytes()).collect::<Vec<u8>>())?, Buf::zeroed(8 * hs * 2)?);
    // (the mixer's pending update, as the decode step's Gated Residuals take it)
    let (po, pinj) = (up(&bf(rnd(8 * d * 2)))?, up(&(0..32).flat_map(|j| (0.25 + (j % 4) as f32 / 8.0).to_le_bytes()).collect::<Vec<u8>>())?);
    let post = (po.ptr, pinj.ptr);
    let names = |b: &str| [format!("{b}_1"), format!("{b}_2"), format!("{b}_4"), format!("{b}_8")];
    let load = |b: &str| -> Result<Vec<crate::dev::Kernel>> { names(b).iter().map(|n| dev.kernel(n)).collect() };
    let (sh1, sh2) = (load("qx_sh1")?, load("qx_sh2")?);
    let pick = |n: usize| match n {
        1 => 0,
        2 => 1,
        3 | 4 => 2,
        _ => 3,
    };
    let s = dev.stream;
    let (e0, e1) = (Ev::new()?, Ev::new()?);
    for n in [1usize, 2, 4, 8] {
        let time = |what: &str, bytes: usize, f: &mut dyn FnMut(usize) -> Result<()>| -> Result<()> {
            for i in 0..copies {
                f(i)?;
            }
            e0.record(s)?;
            for i in 0..reps {
                f(i % copies)?;
            }
            e1.record(s)?;
            dev.sync()?;
            let us_ = e1.since(&e0)? as f64 * 1e3 / reps as f64;
            println!("  {n} rows  {what:<28} {us_:>7.1} us  {:>6.0} GB/s", bytes as f64 / us_ / 1e3);
            Ok(())
        };
        // SAFETY: live buffers of the sizes the kernels read and write.
        unsafe {
            time("gated residual: norm, down", r * w4, &mut |i| {
                kq.gr1(s, x.ptr, wn.ptr, dq[i].ptr, ds.ptr, injw.ptr, rs.ptr, lo.ptr, inj.ptr, n, d, r, 1e-6, post)
            })?;
            time("  the same by rows", r * w4, &mut |i| {
                kq.gr1_rows(s, x.ptr, wn.ptr, dq[i].ptr, ds.ptr, injw.ptr, rs.ptr, lo.ptr, inj.ptr, n, d, r, 1e-6, post)
            })?;
            time("  the same by streams", r * w4, &mut |i| {
                kq.gr1q(s, x.ptr, wn.ptr, dq[i].ptr, ds.ptr, injw.ptr, rs.ptr, lo.ptr, inj.ptr, n, d, r, 1e-6, post)
            })?;
            // (the others against the slices of rows, on the same inputs)
            let get = |f: &mut dyn FnMut() -> Result<()>| -> Result<(Vec<u16>, Vec<f32>, Vec<f32>)> {
                f()?;
                dev.sync()?;
                let (mut l, mut j, mut q) = (vec![0u16; n * r], vec![0f32; n * 4], vec![0f32; n * 4]);
                lo.read(0, &mut l)?;
                inj.read(0, &mut j)?;
                rs.read(0, &mut q)?;
                Ok((l, j, q))
            };
            let a2 = get(&mut || kq.gr1_rows(s, x.ptr, wn.ptr, dq[0].ptr, ds.ptr, injw.ptr, rs.ptr, lo.ptr, inj.ptr, n, d, r, 1e-6, post))?;
            let a1 = get(&mut || kq.gr1(s, x.ptr, wn.ptr, dq[0].ptr, ds.ptr, injw.ptr, rs.ptr, lo.ptr, inj.ptr, n, d, r, 1e-6, post))?;
            let a3 = get(&mut || kq.gr1q(s, x.ptr, wn.ptr, dq[0].ptr, ds.ptr, injw.ptr, rs.ptr, lo.ptr, inj.ptr, n, d, r, 1e-6, post))?;
            let a4 = get(&mut || kq.gr1q(s, x.ptr, wn.ptr, dq[0].ptr, ds.ptr, injw.ptr, rs.ptr, lo.ptr, inj.ptr, n, d, r, 1e-6, post))?;
            let f = |v: u16| f32::from_bits((v as u32) << 16);
            let mx = a2.0.iter().map(|&q| f(q).abs()).fold(0f32, f32::max);
            for (what, a) in [("gr1", &a1), ("by streams", &a3)] {
                let dl = a.0.iter().zip(&a2.0).map(|(&p, &q)| (f(p) - f(q)).abs()).fold(0f32, f32::max);
                let di = a.1.iter().zip(&a2.1).map(|(p, q)| (p - q).abs()).fold(0f32, f32::max);
                let dr = a.2.iter().zip(&a2.2).map(|(p, q)| (p - q).abs() / q.abs().max(1e-30)).fold(0f32, f32::max);
                println!("  {n} rows  {what} against by rows: lo max diff {dl:.3e} (of {mx:.3e}), inj {di:.3e}, rs relative {dr:.3e}");
            }
            println!("  {n} rows  by streams twice alike: {}", a3 == a4);
            time("gated residual: up, mix", w4 * r, &mut |i| kq.gr2(s, x.ptr, wn.ptr, rs.ptr, lo.ptr, uq[i].ptr, us.ptr, h.ptr, n, d, r, post))?;
            let k1 = &sh1[pick(n)];
            time(
                "shared expert: gate, up",
                2 * si * hs,
                &mut |i| launch!(k1, [(si / 8) as u32 + 1, 1, 1], [256, 1, 1], s; x.ptr, shq[i].ptr, shs.ptr, gw.ptr, act.ptr, sg.ptr, n as i32, hs as i32, si as i32),
            )?;
            let k2 = &sh2[pick(n)];
            time(
                "shared expert: down",
                hs * si,
                &mut |i| launch!(k2, [(hs as u32).div_ceil(32), 1, 1], [256, 1, 1], s; act.ptr, sdq[i].ptr, sds.ptr, sg.ptr, sh.ptr, n as i32, hs as i32, si as i32),
            )?;
        }
    }
    Ok(())
}

/// The bf16 bits of `x` (rounded to nearest even).
fn half_bits(x: f32) -> u16 {
    let b = x.to_bits();
    ((b + 0x7fff + ((b >> 16) & 1)) >> 16) as u16
}

/// `qxhost PACK [--layer L] [--pairs 1,2,4,10,30] [--reps N] [--threads N] [--side DIR]`: the host's experts (`cpu_cb::experts_pinned`, as the expert cache calls it) on layer L's experts of the pack, warm: microseconds a call
/// and a pair for calls of each count of pairs (distinct random experts, one token each), and the GB/s of the experts' bytes. No GPU.
pub fn host_bench(pack_dir: &Path, a: &Args) -> Result<()> {
    use crate::cpu_cb::{experts_pinned, GlmExpert, GlmMat, GlmPair};
    use crate::cpu_experts::Z;
    let mut pack = crate::moe_glm_cb::CbPack::open_dims(pack_dir, 2560, 640)?;
    if let Some(d) = a.flag("side") {
        pack = pack.with_side(Path::new(d))?;
    }
    let l = a.num("layer", 3usize)?;
    let lm = pack.map_layer(l)?;
    let (threads, reps) = (a.num("threads", 16usize)?, a.num("reps", 200usize)?);
    let counts: Vec<usize> = a.flag("pairs").unwrap_or("1,2,4,10,30").split(',').filter_map(|x| x.parse().ok()).collect();
    let kernel = crate::cpu_hb::checked_kernel();
    let (hid, inter) = (2560usize, 640usize);
    let experts: Vec<GlmExpert> = lm
        .descs
        .iter()
        .map(|d| {
            let (cl, bl) = d.bytes();
            // SAFETY: the descriptors point into the layer's mapping (checked by `map_layer`), which `lm` keeps.
            let (codes, books) = unsafe { (std::slice::from_raw_parts(d.codes, cl), std::slice::from_raw_parts(d.books, bl)) };
            let (dc, uc) = codes.split_at(inter * d.down.row_bytes(hid));
            let (db, ub) = books.split_at(inter * d.down.entries() * 2);
            GlmExpert::mixed(
                GlmMat { fmt: d.up, rows: 2 * inter, cols: hid, codes: uc, book: ub },
                GlmMat { fmt: d.down, rows: inter, cols: hid, codes: dc, book: db },
                f32::MAX,
            )
        })
        .collect::<Result<_>>()?;
    let bytes_of = |i: usize| -> usize {
        let (c, b) = lm.descs[i].bytes();
        c + b
    };
    // the layer's file read in (warm, as the cache's host tier is in a decode); --populate: its page tables filled too
    lm.map.will_need(0, lm.map.len);
    if a.has("populate") {
        let t0 = std::time::Instant::now();
        let ok = lm.map.populate(0, lm.map.len) && lm.place.as_ref().is_none_or(|p| p.maps.iter().all(|m| m.populate(0, m.len)));
        println!("page tables filled: {ok} in {:.1} ms", t0.elapsed().as_secs_f64() * 1e3);
    }
    let mut seed = 0x9e37_79b9u64;
    let mut next = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    let z: Vec<u16> = (0..hid).map(|_| 0x3c00 ^ (next() as u16 & 0x80ff)).collect();
    println!("layer {l}: {} experts, {} threads, the {} kernel", experts.len(), threads, kernel.name());
    for &n in &counts {
        let mut t = 0f64;
        let mut bytes = 0usize;
        let mut out = vec![0f32; n * hid];
        for rep in 0..reps + 10 {
            let mut ids: Vec<usize> = Vec::new();
            while ids.len() < n {
                let e = next() as usize % experts.len();
                if !ids.contains(&e) {
                    ids.push(e);
                }
            }
            let pairs: Vec<GlmPair> = ids.iter().map(|&e| GlmPair { expert: &experts[e], token: 0, weight: 0.1 }).collect();
            let t0 = std::time::Instant::now();
            experts_pinned(kernel, Z::Bf16(&z), &pairs, &mut out, threads, true)?;
            if rep >= 10 {
                t += t0.elapsed().as_secs_f64();
                bytes += ids.iter().map(|&e| bytes_of(e)).sum::<usize>();
            }
        }
        let us = t * 1e6 / reps as f64;
        println!("  {n:>3} pairs a call: {us:>8.1} us a call, {:>6.1} us a pair, {:>5.1} GB/s", us / n as f64, bytes as f64 / t / 1e9);
    }
    Ok(())
}
