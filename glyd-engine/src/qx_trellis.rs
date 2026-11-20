//! Trellis-coded experts for Qwen3.8-Flash-Next (development): each 2,560-weight row of the rotated experts as a tail-biting bitshift trellis of K bits a weight (format q: K = 2; q3, q4: K = 3, 4).
//! A weight's value is a multiply-add hash of its state (14 bits of the stream from bit K t) and the sum of its bytes, times the row's scale (src/cpu_simd.rs `tq_state_k`, `tq_value`). The kernels:
//! kernels/qwen4x.cu qx_tq_viterbi, qx_tq_back, qx_tq_dequant at 2 bits; qx_tq_vitK, qx_tq_backK, qx_tq_deqK at 3 and 4. The Viterbi search finds each row's stream; a second search closes the stream
//! on itself (the start's 14 - K high bits from the first search's best path), so a row is exactly K bits a weight and any weight decodes from its own 14 bits. (Gaussian rows: 0.0704, 0.0187 and
//! 0.0051 squared error at 2, 3 and 4 bits, 0.60, 0.55 and 0.56 times the best scalar code's.)
//!
//!   glyd-engine qxtq check [--bits K] [--rows N] [SNAPSHOT PACK --layer L --experts E]   the mean squared error on N Gaussian rows against the best 2^K-level scalar code, the GPU's decoder and the
//!       CPU's against the encoder; with SNAPSHOT, the relative error of layer L's first E experts' rows (rotated with PACK's signs) against a 2^K-entry codebook a row
//!   glyd-engine qxtq encode SNAPSHOT PACK OUT --layers A:B [--bits K | --plan PLAN]   layers A..B-1's routed experts coded, rotated with PACK's signs: OUT/layer-LLL.safetensors (each expert's
//!       e{e:04}.d.0 and .u.0, the downT and gate-and-up rows' streams, U8 [rows, K 320]; then .d.1 and .u.1, their scales, BF16 [rows, 1]: the layout the engine's pack reader takes) and OUT/meta.json.
//!       PLAN: {"layers": {"L": [a format an expert: "q", "q3", "q4" (or "cb2", "cb3", "cb4" for the same bits)]}}; --reuse DIR: the experts a pack made with the same rotations holds in the
//!       format the plan wants, copied from it, not coded again
//!   glyd-engine qxtq gpucheck PACK L [--expert E]              expert E of layer L through the GPU's expert kernels (qx_cb_up, qx_cb_down, as the decode runs them from a place) against the CPU's
//!   glyd-engine qxtq decode PACK L OUT                         layer L's experts back to bf16 on the CPU (gate_up_proj [512, 1280, 2560], down_proj [512, 2560, 640]) in OUT, for the reference's KL:
//!       format q's rows and the codebook rows (a, b, c) alike
//! --gain G: the search's codebook scale; --a A, --b B (check only): other hash constants to try. A pack is written with the engine's constants (src/cpu_simd.rs) and says which version of them in its
//! meta.json ("q": 1 for 2-bit rows alone, 2 with q3 or q4 rows), nothing more.

use crate::args::Args;
use crate::cpu_simd::{tq_state_k, tq_value, Fmt, TQ_A as HASH_A, TQ_B as HASH_B, TQ_K};
use crate::dev::{Buf, Dev, Kernel};
use crate::ffi::{err, Error, Result, Stream};
use crate::launch;
use glyd_gpu::json::{self, Value};
use glyd_gpu::safetensors::{Checkpoint, File};
use std::path::Path;
use std::time::Instant;

/// A row's weights.
const T: usize = 2560;
const NE: usize = 512;
const UP: usize = 1280;
const DN: usize = 640;
/// The rows a search takes at once (their choices: T 1024 bytes a row at 2 bits, twice that at 3 and 4).
const BATCH: usize = 640;
const GAIN: f32 = 1.0;

/// A row's bits a weight.
#[derive(Clone, Copy, PartialEq)]
struct Shape {
    k: usize,
}

impl Shape {
    fn of(f: Fmt) -> Shape {
        Shape { k: f.tq_bits().expect("a format q row") }
    }

    /// The row's 32-bit words.
    fn words(self) -> usize {
        self.k * T / 32
    }

    fn fmt(self) -> Fmt {
        match self.k {
            2 => Fmt::Tq,
            3 => Fmt::Tq3,
            _ => Fmt::Tq4,
        }
    }

    /// The search's dynamic shared memory: the states' costs (at 3 and 4 bits a float of padding in 32) and the row.
    fn smem(self) -> u32 {
        let s = 1usize << 14;
        ((if self.k == 2 { s } else { s + s / 32 } + T) * 4) as u32
    }
}

struct Tq {
    sh: Shape,
    viterbi: Kernel,
    back: Kernel,
    dequant: Kernel,
    rot: Kernel,
    a: u32,
    b: u32,
    gain: f32,
}

/// Rows coded: each row's stream (`words` words), scale, and squared error at that scale.
struct Coded {
    codes: Vec<u32>,
    scale: Vec<f32>,
    err: Vec<f32>,
}

impl Tq {
    fn load(dev: &Dev, sh: Shape, a: u32, b: u32, gain: f32) -> Result<Tq> {
        let names = if sh.k == 2 {
            ["qx_tq_viterbi".to_string(), "qx_tq_back".to_string(), "qx_tq_dequant".to_string()]
        } else {
            ["vit", "back", "deq"].map(|x| format!("qx_tq_{x}{}", sh.k))
        };
        let viterbi = dev.kernel(&names[0])?;
        viterbi.allow_smem(sh.smem())?;
        Ok(Tq { sh, viterbi, back: dev.kernel(&names[1])?, dequant: dev.kernel(&names[2])?, rot: dev.kernel("qx_cb_rot_in")?, a, b, gain })
    }

    /// The n rows at `rows` (fp32 [n, T], rotated) coded, `BATCH` at a time (`dec`: `dec_buf`).
    fn code(&self, rows: &Buf, n: usize, dec: &Buf) -> Result<Coded> {
        let s = Stream::DEFAULT;
        let w = self.sh.words();
        let (end, h, codes, scale, errs) = (Buf::new(BATCH * 4)?, Buf::new(BATCH * 4)?, Buf::new(BATCH * w * 4)?, Buf::new(BATCH * 4)?, Buf::new(BATCH * 4)?);
        let mut c = Coded { codes: vec![0; n * w], scale: vec![0.0; n], err: vec![0.0; n] };
        for r0 in (0..n).step_by(BATCH) {
            let m = BATCH.min(n - r0);
            let rp = rows.ptr + (r0 * T * 4) as u64;
            let backs = [(m as u32).div_ceil(64), 1, 1];
            // SAFETY: the buffers are live and as large as the kernels read and write for m rows of T (dec: BATCH rows' choices).
            unsafe {
                launch!(self.viterbi, [m as u32, 1, 1], [1024, 1, 1], smem self.sh.smem(), s; rp, T as i32, self.gain, self.a, self.b, 0u64, dec.ptr, end.ptr)?;
                launch!(self.back, backs, [64, 1, 1], s; dec.ptr, end.ptr, T as i32, m as i32, h.ptr, rp, self.a, self.b, 0u64, 0u64, 0u64)?;
                launch!(self.viterbi, [m as u32, 1, 1], [1024, 1, 1], smem self.sh.smem(), s; rp, T as i32, self.gain, self.a, self.b, h.ptr, dec.ptr, end.ptr)?;
                launch!(self.back, backs, [64, 1, 1], s; dec.ptr, end.ptr, T as i32, m as i32, 0u64, rp, self.a, self.b, codes.ptr, scale.ptr, errs.ptr)?;
            }
            codes.read(0, &mut c.codes[r0 * w..(r0 + m) * w])?;
            scale.read(0, &mut c.scale[r0..r0 + m])?;
            errs.read(0, &mut c.err[r0..r0 + m])?;
        }
        Ok(c)
    }

    /// x' = R x of n bf16 rows (`x`, [n, T]) into `xr` (fp32 [n, T]).
    fn rotate(&self, x: &Buf, n: usize, sg: &Buf, xr: &Buf) -> Result<()> {
        // SAFETY: x holds n rows of T bf16, sg T signs, xr n rows of T fp32.
        unsafe { launch!(self.rot, [n as u32, (T / 512) as u32, 1], [512, 1, 1], Stream::DEFAULT; x.ptr, T as i32, sg.ptr, xr.ptr, 0u64) }
    }

    /// n rows back from their streams and scales: the rotated values to `xr` (0: not wanted).
    fn decode(&self, codes: &Buf, scale: &Buf, n: usize, xr: u64) -> Result<()> {
        // SAFETY: codes holds n rows of the format's words, scale n floats; xr n rows of T.
        unsafe {
            launch!(self.dequant, [n as u32, (T / 512) as u32, 1], [512, 1, 1], Stream::DEFAULT; codes.ptr, scale.ptr, T as i32, self.a, self.b, 0u64, xr, 0u64)
        }
    }
}

/// The searches' choices for `BATCH` rows (2 bytes a group of states a step: the widest, 3 and 4 bits).
fn dec_buf() -> Result<Buf> {
    Buf::new(BATCH * T * 1024 * 2)
}

/// Unit Gaussian samples (xorshift64* and Box-Muller).
fn gauss(seed: u64, n: usize) -> Vec<f32> {
    let mut x = seed.wrapping_mul(0x9E6C_63D0_676A_9A99) | 1;
    let mut uniform = move || {
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        ((x.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    };
    (0..n)
        .map(|_| {
            let (u, v) = (uniform(), uniform());
            ((-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()) as f32
        })
        .collect()
}

/// The squared error of a row's best 2^k-level scalar code (Lloyd's rounds from the unit Gaussian's optimal levels times the row's RMS).
fn lloyd(row: &[f32], k: usize) -> f64 {
    let half: &[f64] = match k {
        2 => &[0.4528, 1.5104],
        3 => &[0.2451, 0.7560, 1.3439, 2.1520],
        _ => &[0.1284, 0.3881, 0.6568, 0.9424, 1.2562, 1.6181, 2.0690, 2.7326],
    };
    let rms = (row.iter().map(|&v| (v as f64).powi(2)).sum::<f64>() / row.len() as f64).sqrt().max(1e-30);
    let mut c: Vec<f64> = half.iter().rev().map(|l| -l * rms).chain(half.iter().map(|l| l * rms)).collect();
    let n = c.len();
    let pick = |c: &[f64], v: f64| (1..n).filter(|&i| v > (c[i - 1] + c[i]) / 2.0).count();
    for _ in 0..10 {
        let (mut sum, mut cnt) = (vec![0f64; n], vec![0f64; n]);
        row.iter().for_each(|&v| {
            let i = pick(&c, v as f64);
            sum[i] += v as f64;
            cnt[i] += 1.0;
        });
        (0..n).filter(|&i| cnt[i] > 0.0).for_each(|i| c[i] = sum[i] / cnt[i]);
        c.sort_by(f64::total_cmp);
    }
    row.iter().map(|&v| (v as f64 - c[pick(&c, v as f64)]).powi(2)).sum()
}

/// A pack's meta.json.
fn meta(pack: &Path) -> Result<Value> {
    let path = pack.join("meta.json");
    let text = std::fs::read_to_string(&path).map_err(|e| Error(format!("{}: {e}", path.display())))?;
    json::parse(&text).map_err(|e| Error(format!("{}: {e}", path.display())))
}

/// Layer l's rotation signs (+1 / -1) from a pack's meta.json, as i8.
fn signs(pack: &Path, l: usize) -> Result<Vec<i8>> {
    let v = meta(pack)?;
    let s = v
        .get("layers")
        .and_then(|x| x.get(&l.to_string()))
        .and_then(|x| x.get("s"))
        .and_then(Value::as_u64s)
        .ok_or_else(|| Error(format!("{}: layer {l}: no s", pack.display())))?;
    if s.len() != T {
        return err(format!("{}: layer {l}: {} signs", pack.display(), s.len()));
    }
    Ok(s.into_iter().map(|b| if b == 1 { 1 } else { -1 }).collect())
}

/// Layer l's routed experts from the checkpoint (layer 48: the MTP head's): gate_up_proj [512, 1280, 2560] and down_proj [512, 2560, 640], bf16 words.
fn experts(ck: &Checkpoint, l: usize) -> Result<(Vec<u16>, Vec<u16>)> {
    let (mut gu, mut dn) = (vec![0u16; NE * UP * T], vec![0u16; NE * T * DN]);
    let p = if l == 48 { "mtp.layers.0.mlp.experts.".to_string() } else { format!("model.language_model.layers.{l}.mlp.experts.") };
    ck.read_bf16(&format!("{p}gate_up_proj"), &mut gu).map_err(|e| Error(format!("layer {l}: {e}")))?;
    ck.read_bf16(&format!("{p}down_proj"), &mut dn).map_err(|e| Error(format!("layer {l}: {e}")))?;
    Ok((gu, dn))
}

/// Expert e's downT rows ([640, 2560]: down_proj's columns) from down_proj.
fn down_t(dn: &[u16], e: usize) -> Vec<u16> {
    let m = &dn[e * T * DN..(e + 1) * T * DN];
    let mut out = vec![0u16; DN * T];
    for i in 0..T {
        for j in 0..DN {
            out[j * T + i] = m[i * DN + j];
        }
    }
    out
}

/// fp32 to bf16, rounded to nearest even.
fn bf16_of(v: f32) -> u16 {
    let b = v.to_bits();
    ((b + 0x7FFF + ((b >> 16) & 1)) >> 16) as u16
}

fn f32_of(h: u16) -> f32 {
    f32::from_bits((h as u32) << 16)
}

fn as_bytes<X: Copy>(v: &[X]) -> &[u8] {
    // SAFETY: plain data viewed as its bytes.
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

/// A row's rotated values from its codes and table, on the CPU: format q's from the states, the codebook rows' (cb2, cb3, cb4: code j at bit B j, least significant first) from their tables.
fn row_values(f: Fmt, codes: &[u8], book: &[u8], out: &mut [f32]) -> Result<()> {
    if let Some(k) = f.tq_bits() {
        let words: Vec<u32> = codes.chunks_exact(4).map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
        let sc = f32_of(u16::from_le_bytes([book[0], book[1]])) * TQ_K;
        for (t, o) in out.iter_mut().enumerate() {
            *o = tq_value(tq_state_k(&words, k, t)) * sc;
        }
        return Ok(());
    }
    let b = match f {
        Fmt::Cb2 => 2,
        Fmt::Cb3 => 3,
        Fmt::Cb4 => 4,
        _ => return err(format!("qxtq decode: format {} rows are not read here", f.name())),
    };
    for (j, o) in out.iter_mut().enumerate() {
        let at = b * j;
        let w = codes[at / 8] as u32 | (codes.get(at / 8 + 1).copied().unwrap_or(0) as u32) << 8;
        let c = ((w >> (at % 8)) & ((1 << b) - 1)) as usize;
        *o = f32_of(u16::from_le_bytes([book[2 * c], book[2 * c + 1]]));
    }
    Ok(())
}

/// R^T y of a row in place (each 512 values' Walsh-Hadamard transform over sqrt(512), then the signs), as qx_tq_dequant makes it.
fn unrotate(y: &mut [f32], sg: &[i8]) {
    for blk in y.chunks_exact_mut(512) {
        let mut h = 1;
        while h < 512 {
            for r in 0..256 {
                let i = (r / h) * 2 * h + r % h;
                let (a, c) = (blk[i], blk[i + h]);
                blk[i] = a + c;
                blk[i + h] = a - c;
            }
            h <<= 1;
        }
    }
    for (v, &s) in y.iter_mut().zip(sg) {
        *v = *v * 0.044194173824159216 * s as f32;
    }
}

/// --bits's shape (2 by default).
fn shape(a: &Args) -> Result<Shape> {
    match a.num("bits", 2usize)? {
        k @ 2..=4 => Ok(Shape { k }),
        k => err(format!("--bits {k}: 2, 3 or 4")),
    }
}

pub fn cmd(a: &Args) -> Result<()> {
    let (ha, hb, gain) = (a.num("a", HASH_A)?, a.num("b", HASH_B)?, a.num("gain", GAIN)?);
    match a.need(1, "check, encode or decode")? {
        "check" => check(a, ha, hb, gain),
        "encode" => encode(a, gain),
        "decode" => decode(a),
        "gpucheck" => gpucheck(a),
        c => err(format!("qxtq {c}: check, encode, decode or gpucheck")),
    }
}

fn check(a: &Args, ha: u32, hb: u32, gain: f32) -> Result<()> {
    let sh = shape(a)?;
    let dev = Dev::new(0)?;
    let tq = Tq::load(&dev, sh, ha, hb, gain)?;
    let n: usize = a.num("rows", 128)?;
    let g = gauss(a.num("seed", 1u64)?, n * T);
    let (rows, dec) = (Buf::new(n * T * 4)?, dec_buf()?);
    rows.write(0, &g)?;
    let t0 = Instant::now();
    let c = tq.code(&rows, n, &dec)?;
    let ms = t0.elapsed().as_secs_f64() * 1e3;
    let mse = c.err.iter().map(|&e| e as f64).sum::<f64>() / (n * T) as f64;
    let sc = g.chunks(T).map(|r| lloyd(r, sh.k)).sum::<f64>() / (n * T) as f64;
    // the decoders against the encoder: the GPU's rows from their streams (the encoder's hash constants), and the CPU's (the engine's) where the constants are the engine's
    let w = sh.words();
    let (cb, sb, xr) = (Buf::new(n * w * 4)?, Buf::new(n * 4)?, Buf::new(n * T * 4)?);
    cb.write(0, &c.codes)?;
    sb.write(0, &c.scale)?;
    tq.decode(&cb, &sb, n, xr.ptr)?;
    let mut back = vec![0f32; n * T];
    xr.read(0, &mut back)?;
    let mse2 = g.iter().zip(&back).map(|(&x, &y)| (x as f64 - y as f64).powi(2)).sum::<f64>() / (n * T) as f64;
    let db = |m: f64| -10.0 * m.log10();
    println!(
        "gaussian, {n} rows of {T}, {} bits: trellis {mse:.5} ({:.2} dB; decoded {mse2:.5}), best {}-level scalar {sc:.5} ({:.2} dB): {:.3}x its error; {:.2} ms a row (gain {gain}, a {ha}, b {hb})",
        sh.k,
        db(mse),
        1 << sh.k,
        db(sc),
        mse / sc,
        ms / n as f64
    );
    if (mse2 - mse).abs() > 1e-3 * mse.max(1e-9) + 1e-6 {
        return err(format!("the decoder's error {mse2} is not the encoder's {mse}"));
    }
    if (ha, hb) == (HASH_A, HASH_B) {
        let mut cpu = vec![0f32; T];
        let mut worst = 0f64;
        for r in 0..n.min(16) {
            let scale = bf16_of(c.scale[r]);
            row_values(sh.fmt(), as_bytes(&c.codes[r * w..(r + 1) * w]), &scale.to_le_bytes(), &mut cpu)?;
            // (the GPU's rows at the fp32 scale, the CPU's at the bf16 one the pack holds)
            let k = f32_of(scale) as f64 / c.scale[r] as f64;
            worst = cpu.iter().zip(&back[r * T..(r + 1) * T]).fold(worst, |m, (&x, &y)| m.max((x as f64 - k * y as f64).abs()));
        }
        println!("the CPU's values against the GPU's: largest difference {worst:.2e}");
        if worst > 1e-4 {
            return err(format!("the CPU's decode differs from the GPU's by {worst:.2e}"));
        }
    }
    let Some(snap) = a.pos.get(2) else { return Ok(()) };
    let pack = Path::new(a.need(3, "PACK (its meta.json's signs)")?);
    let (l, experts_n) = (a.num("layer", 40usize)?, a.num("experts", 8usize)?);
    let ck = Checkpoint::open(Path::new(snap)).map_err(|e| Error(format!("{snap}: {e}")))?;
    let (gu, dn) = experts(&ck, l)?;
    let sg = Buf::new(T)?;
    sg.write(0, &signs(pack, l)?)?;
    let (x, xr) = (Buf::new(UP * T * 2)?, Buf::new(UP * T * 4)?);
    for (name, rows_n) in [("up", UP), ("downT", DN)] {
        let (mut e2, mut s2, mut sc2) = (0f64, 0f64, 0f64);
        let t0 = Instant::now();
        for e in 0..experts_n {
            let m = if rows_n == UP { gu[e * UP * T..(e + 1) * UP * T].to_vec() } else { down_t(&dn, e) };
            x.write(0, &m)?;
            tq.rotate(&x, rows_n, &sg, &xr)?;
            let c = tq.code(&xr, rows_n, &dec)?;
            let mut r = vec![0f32; rows_n * T];
            xr.read(0, &mut r)?;
            e2 += c.err.iter().map(|&v| v as f64).sum::<f64>();
            s2 += r.iter().map(|&v| (v as f64).powi(2)).sum::<f64>();
            sc2 += r.chunks(T).map(|row| lloyd(row, sh.k)).sum::<f64>();
        }
        println!(
            "layer {l} {name}, experts 0..{experts_n}: relative error trellis {:.4}, {}-entry codebook a row {:.4} ({:.1} s)",
            (e2 / s2).sqrt(),
            1 << sh.k,
            (sc2 / s2).sqrt(),
            t0.elapsed().as_secs_f64()
        );
    }
    Ok(())
}

/// The formats of layer l's experts: --plan's, or --bits's for all.
fn plan_of(a: &Args, l: usize) -> Result<Vec<Fmt>> {
    let Some(p) = a.flag("plan") else {
        return Ok(vec![shape(a)?.fmt(); NE]);
    };
    let text = std::fs::read_to_string(p).map_err(|e| Error(format!("{p}: {e}")))?;
    let v = json::parse(&text).map_err(|e| Error(format!("{p}: {e}")))?;
    let names = v.get("layers").and_then(|x| x.get(&l.to_string())).and_then(Value::as_array).ok_or_else(|| Error(format!("{p}: no layer {l}")))?;
    let f: Vec<Fmt> = names
        .iter()
        .map(|x| match x.as_str() {
            Some("q" | "cb2") => Ok(Fmt::Tq),
            Some("q3" | "cb3") => Ok(Fmt::Tq3),
            Some("q4" | "cb4") => Ok(Fmt::Tq4),
            o => err(format!("{p}: layer {l}: {o:?} is not q, q3 or q4")),
        })
        .collect::<Result<_>>()?;
    if f.len() != NE {
        return err(format!("{p}: layer {l}: {} experts", f.len()));
    }
    Ok(f)
}

fn encode(a: &Args, gain: f32) -> Result<()> {
    let (snap, pack, out) = (Path::new(a.need(2, "SNAPSHOT")?), Path::new(a.need(3, "PACK (its meta.json's signs)")?), Path::new(a.need(4, "OUT")?));
    let range = a.flag("layers").ok_or_else(|| Error("qxtq encode needs --layers A:B".into()))?;
    let (first, end) = range
        .split_once(':')
        .and_then(|(x, y)| Some((x.parse::<usize>().ok()?, y.parse::<usize>().ok()?)))
        .ok_or_else(|| Error(format!("--layers {range}: not A:B")))?;
    std::fs::create_dir_all(out).map_err(|e| Error(format!("{}: {e}", out.display())))?;
    let dev = Dev::new(0)?;
    let ck = Checkpoint::open(snap).map_err(|e| Error(format!("{}: {e}", snap.display())))?;
    let (x, xr, sg, dec) = (Buf::new(UP * T * 2)?, Buf::new(UP * T * 4)?, Buf::new(T)?, dec_buf()?);
    // (the searches of each width the layers want, made once)
    let mut tqs: Vec<Tq> = Vec::new();
    for l in first..end {
        let t0 = Instant::now();
        let fmts = plan_of(a, l)?;
        for f in &fmts {
            if !tqs.iter().any(|t| t.sh.fmt() == *f) {
                tqs.push(Tq::load(&dev, Shape::of(*f), HASH_A, HASH_B, gain)?);
            }
        }
        let s = signs(pack, l)?;
        sg.write(0, &s)?;
        let (gu, dn) = experts(&ck, l)?;
        // --reuse DIR: the experts DIR holds in the format the plan wants, copied from it (DIR made with the same rotations)
        let reuse = match a.flag("reuse").map(Path::new) {
            Some(d) if meta(d)?.get("layers").and_then(|x| x.get(&l.to_string())).is_some() => {
                if signs(d, l)? != s {
                    return err(format!("--reuse {}: layer {l}'s rotation is not PACK's", d.display()));
                }
                let lm = meta(d)?;
                let theirs: Vec<Option<Fmt>> = lm
                    .get("layers")
                    .and_then(|x| x.get(&l.to_string()))
                    .and_then(|x| x.get("f"))
                    .and_then(Value::as_array)
                    .map(|v| v.iter().map(|x| x.as_str().and_then(Fmt::parse)).collect())
                    .unwrap_or_default();
                Some((File::open(&d.join(format!("layer-{l:03}.safetensors"))).map_err(|x| Error(format!("--reuse layer {l}: {x}")))?, theirs))
            }
            _ => None,
        };
        let read = t0.elapsed().as_secs_f64();
        // (an expert's codes, downT's then up's, then its scales: the engine reads each expert's codes and scales as one range each)
        let mut parts: Vec<[(String, &'static str, Vec<u64>, Vec<u8>); 4]> = Vec::with_capacity(NE);
        let (mut e2, mut w2, mut reused) = ([0f64; 2], [0f64; 2], 0usize);
        for (e, f) in fmts.iter().enumerate() {
            let tq = tqs.iter().find(|t| t.sh.fmt() == *f).ok_or_else(|| Error("no search".into()))?;
            let rb = (tq.sh.words() * 4) as u64;
            let rows = |n: usize, w: u64| vec![n as u64, w];
            if let Some((rf, _)) = reuse.as_ref().filter(|(_, t)| t.get(e) == Some(&Some(*f))) {
                let get = |name: String| -> Result<Vec<u8>> {
                    rf.read(rf.get(&name).ok_or_else(|| Error(format!("--reuse layer {l}: no {name}")))?).map_err(|x| Error(format!("--reuse layer {l}: {x}")))
                };
                parts.push([
                    (format!("e{e:04}.d.0"), "U8", rows(DN, rb), get(format!("e{e:04}.d.0"))?),
                    (format!("e{e:04}.u.0"), "U8", rows(UP, rb), get(format!("e{e:04}.u.0"))?),
                    (format!("e{e:04}.d.1"), "BF16", rows(DN, 1), get(format!("e{e:04}.d.1"))?),
                    (format!("e{e:04}.u.1"), "BF16", rows(UP, 1), get(format!("e{e:04}.u.1"))?),
                ]);
                reused += 1;
                continue;
            }
            let mut got: Vec<(Vec<u8>, Vec<u8>)> = Vec::with_capacity(2);
            for (k, rows_n) in [DN, UP].into_iter().enumerate() {
                let m = if k == 0 { down_t(&dn, e) } else { gu[e * UP * T..(e + 1) * UP * T].to_vec() };
                x.write(0, &m)?;
                tq.rotate(&x, rows_n, &sg, &xr)?;
                let c = tq.code(&xr, rows_n, &dec)?;
                e2[1 - k] += c.err.iter().map(|&v| v as f64).sum::<f64>();
                w2[1 - k] += m.iter().map(|&b| (f32_of(b) as f64).powi(2)).sum::<f64>();
                let scales: Vec<u16> = c.scale.iter().map(|&v| bf16_of(v)).collect();
                got.push((as_bytes(&c.codes).to_vec(), as_bytes(&scales).to_vec()));
            }
            let [(dc, ds), (uc, us)]: [(Vec<u8>, Vec<u8>); 2] = got.try_into().map_err(|_| Error("two matrices an expert".into()))?;
            parts.push([
                (format!("e{e:04}.d.0"), "U8", rows(DN, rb), dc),
                (format!("e{e:04}.u.0"), "U8", rows(UP, rb), uc),
                (format!("e{e:04}.d.1"), "BF16", rows(DN, 1), ds),
                (format!("e{e:04}.u.1"), "BF16", rows(UP, 1), us),
            ]);
        }
        let fname = format!("layer-{l:03}.safetensors");
        let tensors: Vec<(&str, &str, Vec<u64>, &[u8])> =
            parts.iter().flat_map(|p| p.iter().map(|(n, d, s, b)| (n.as_str(), *d, s.clone(), b.as_slice()))).collect();
        let tmp = out.join(format!("{fname}.part"));
        crate::pack::write_safetensors(&tmp, &[("format", "pt".to_string())], &tensors)?;
        std::fs::rename(&tmp, out.join(&fname)).map_err(|e| Error(format!("{fname}: {e}")))?;
        let bytes = std::fs::metadata(out.join(&fname)).map(|m| m.len()).unwrap_or(0);
        let sha = crate::hub::sha256_of(&out.join(&fname)).map_err(|e| Error(format!("{fname}: {e}")))?;
        let sl: Vec<String> = s.iter().map(|&v| if v > 0 { "1" } else { "0" }.to_string()).collect();
        let fl: Vec<String> = fmts.iter().map(|f| format!("\"{}\"", f.name())).collect();
        let entry = format!("{{\"file\": \"{fname}\", \"bytes\": {bytes}, \"sha256\": \"{sha}\", \"s\": [{}], \"f\": [{}]}}", sl.join(", "), fl.join(", "));
        std::fs::write(out.join(format!("layer-{l:03}.json")), entry).map_err(|e| Error(format!("layer {l}: {e}")))?;
        write_meta(out)?;
        let count = |f: Fmt| fmts.iter().filter(|&&x| x == f).count();
        println!(
            "layer {l}: {:.0} s (reading {read:.0}); q {} q3 {} q4 {} ({reused} reused); relative error of the coded gate-up {:.4}, down {:.4}",
            t0.elapsed().as_secs_f64(),
            count(Fmt::Tq),
            count(Fmt::Tq3),
            count(Fmt::Tq4),
            (e2[0] / w2[0].max(1e-30)).sqrt(),
            (e2[1] / w2[1].max(1e-30)).sqrt()
        );
    }
    Ok(())
}

/// OUT/meta.json from the layers' entries (OUT/layer-LLL.json) written so far: "q" 1 where every row is 2-bit, 2 where any is q3 or q4.
fn write_meta(out: &Path) -> Result<()> {
    let mut layers: Vec<(usize, String)> = Vec::new();
    for d in std::fs::read_dir(out).map_err(|e| Error(format!("{}: {e}", out.display())))?.flatten() {
        let name = d.file_name().to_string_lossy().into_owned();
        if let Some(l) = name.strip_prefix("layer-").and_then(|x| x.strip_suffix(".json")).and_then(|x| x.parse::<usize>().ok()) {
            layers.push((l, std::fs::read_to_string(d.path()).map_err(|e| Error(format!("{name}: {e}")))?));
        }
    }
    layers.sort();
    let q = if layers.iter().any(|(_, v)| v.contains("\"q3\"") || v.contains("\"q4\"")) { 2 } else { 1 };
    let body: Vec<String> = layers.iter().map(|(l, v)| format!("\"{l}\": {v}")).collect();
    let text = format!("{{\"format\": \"glyd-x/1\", \"hidden\": {T}, \"intermediate\": {DN}, \"q\": {q}, \"layers\": {{{}}}}}", body.join(", "));
    let tmp = out.join("meta.json.part");
    std::fs::write(&tmp, text).map_err(|e| Error(format!("meta.json: {e}")))?;
    std::fs::rename(&tmp, out.join("meta.json")).map_err(|e| Error(format!("meta.json: {e}")))
}

fn decode(a: &Args) -> Result<()> {
    let (pack, l, out) =
        (Path::new(a.need(2, "PACK")?), a.need(3, "LAYER")?.parse::<usize>().map_err(|_| Error("LAYER: not a number".into()))?, Path::new(a.need(4, "OUT")?));
    let v = meta(pack)?;
    let entry = v.get("layers").and_then(|x| x.get(&l.to_string())).ok_or_else(|| Error(format!("{}: no layer {l}", pack.display())))?;
    let fmts: Vec<Fmt> = entry
        .get("f")
        .and_then(Value::as_array)
        .ok_or_else(|| Error(format!("{}: layer {l}: no f", pack.display())))?
        .iter()
        .map(|x| x.as_str().and_then(Fmt::parse).ok_or_else(|| Error(format!("{}: layer {l}: a format this decode does not read: {x:?}", pack.display()))))
        .collect::<Result<_>>()?;
    if fmts.len() != NE || fmts.iter().any(|f| !f.is_tq() && !matches!(f, Fmt::Cb2 | Fmt::Cb3 | Fmt::Cb4)) {
        return err(format!("{}: layer {l}: {} experts, or one of a format other than q, q3, q4, a, b, c", pack.display(), fmts.len()));
    }
    let q = v.get("q").and_then(Value::as_u64).unwrap_or(0);
    if fmts.iter().any(|f| f.is_tq()) && !(q == 1 && fmts.iter().all(|&f| f != Fmt::Tq3 && f != Fmt::Tq4) || q == 2) {
        return err(format!("{}: not a format q pack of this engine's version", pack.display()));
    }
    let file = entry.get("file").and_then(Value::as_str).map(String::from).unwrap_or_else(|| format!("layer-{l:03}.safetensors"));
    let sg = signs(pack, l)?;
    let f = File::open(&pack.join(&file)).map_err(|e| Error(format!("layer {l}: {e}")))?;
    let read = |name: &str| -> Result<Vec<u8>> {
        f.read(f.get(name).ok_or_else(|| Error(format!("layer {l}: no {name}")))?).map_err(|e| Error(format!("layer {l}: {name}: {e}")))
    };
    let (mut gu, mut dn) = (vec![0u16; NE * UP * T], vec![0u16; NE * T * DN]);
    let threads = std::thread::available_parallelism().map_or(8, |n| n.get());
    let per = NE.div_ceil(threads);
    let t0 = Instant::now();
    // (each thread its run of experts: their bytes read here, the rows decoded there)
    let mut jobs = Vec::new();
    for e in 0..NE {
        let name = |m: &str, k: usize| format!("e{e:04}.{m}.{k}");
        jobs.push((read(&name("u", 0))?, read(&name("u", 1))?, read(&name("d", 0))?, read(&name("d", 1))?));
    }
    std::thread::scope(|scope| -> Result<()> {
        let mut hs = Vec::new();
        for ((jb, (gu, dn)), e0) in jobs.chunks(per).zip(gu.chunks_mut(per * UP * T).zip(dn.chunks_mut(per * T * DN))).zip((0..NE).step_by(per)) {
            let (fmts, sg) = (&fmts, &sg);
            hs.push(scope.spawn(move || -> Result<()> {
                let mut row = vec![0f32; T];
                for (i, (uc, ub, dc, db)) in jb.iter().enumerate() {
                    // (an expert's two matrices are of one format in these packs)
                    let f = fmts[e0 + i];
                    let (rb, kb) = (f.row_bytes(T), 2 * f.entries());
                    for r in 0..UP {
                        row_values(f, &uc[r * rb..(r + 1) * rb], &ub[r * kb..(r + 1) * kb], &mut row)?;
                        unrotate(&mut row, sg);
                        for (o, &v) in gu[(i * UP + r) * T..(i * UP + r + 1) * T].iter_mut().zip(&row) {
                            *o = bf16_of(v);
                        }
                    }
                    let m = &mut dn[i * T * DN..(i + 1) * T * DN];
                    for j in 0..DN {
                        row_values(f, &dc[j * rb..(j + 1) * rb], &db[j * kb..(j + 1) * kb], &mut row)?;
                        unrotate(&mut row, sg);
                        for (t, &v) in row.iter().enumerate() {
                            m[t * DN + j] = bf16_of(v);
                        }
                    }
                }
                Ok(())
            }));
        }
        hs.into_iter().try_for_each(|h| h.join().map_err(|_| Error("a decode thread panicked".into()))?)
    })?;
    let tensors: [(&str, &str, Vec<u64>, &[u8]); 2] = [
        ("gate_up_proj", "BF16", vec![NE as u64, UP as u64, T as u64], as_bytes(&gu)),
        ("down_proj", "BF16", vec![NE as u64, T as u64, DN as u64], as_bytes(&dn)),
    ];
    crate::pack::write_safetensors(out, &[("format", "pt".to_string())], &tensors)?;
    eprintln!("layer {l}: decoded in {:.1} s", t0.elapsed().as_secs_f64());
    // --snap SNAPSHOT: the decoded weights' relative error against the checkpoint's
    if let Some(snap) = a.flag("snap") {
        let ck = Checkpoint::open(Path::new(snap)).map_err(|e| Error(format!("{snap}: {e}")))?;
        let (wu, wd) = experts(&ck, l)?;
        let rel = |x: &[u16], y: &[u16]| {
            (x.iter().zip(y).map(|(&p, &q)| (f32_of(p) as f64 - f32_of(q) as f64).powi(2)).sum::<f64>()
                / y.iter().map(|&q| (f32_of(q) as f64).powi(2)).sum::<f64>())
            .sqrt()
        };
        println!("layer {l}: relative error gate-up {:.4}, down {:.4}", rel(&gu, &wu), rel(&dn, &wd));
    }
    Ok(())
}

fn gpucheck(a: &Args) -> Result<()> {
    use crate::cpu_simd::{dot_glm, down_glm, GlmMat};
    let (pack, l, e) =
        (Path::new(a.need(2, "PACK")?), a.need(3, "LAYER")?.parse::<usize>().map_err(|_| Error("LAYER: not a number".into()))?, a.num("expert", 0usize)?);
    let f = File::open(&pack.join(format!("layer-{l:03}.safetensors"))).map_err(|x| Error(format!("layer {l}: {x}")))?;
    let read = |name: String| -> Result<Vec<u8>> {
        f.read(f.get(&name).ok_or_else(|| Error(format!("layer {l}: no {name}")))?).map_err(|x| Error(format!("layer {l}: {name}: {x}")))
    };
    let (dc, uc, ds, us) = (read(format!("e{e:04}.d.0"))?, read(format!("e{e:04}.u.0"))?, read(format!("e{e:04}.d.1"))?, read(format!("e{e:04}.u.1"))?);
    // the expert's format from its rows' bytes
    let fmt = match uc.len() / UP {
        640 => Fmt::Tq,
        960 => Fmt::Tq3,
        _ => Fmt::Tq4,
    };
    // a place as the cache makes one: codes downT then up; books downT's then up's, the formats' byte (up's nibble, down's) last
    let codes = [dc.as_slice(), uc.as_slice()].concat();
    let bstride = (ds.len() + us.len() + 16).next_multiple_of(16);
    let mut books = vec![0u8; bstride];
    books[..ds.len()].copy_from_slice(&ds);
    books[ds.len()..ds.len() + us.len()].copy_from_slice(&us);
    let nib = match fmt {
        Fmt::Tq => 4u8,
        Fmt::Tq3 => 5,
        _ => 6,
    };
    books[bstride - 1] = nib | nib << 4;
    let x = gauss(a.num("seed", 7u64)?, T);
    let dev = Dev::new(0)?;
    let (up, down) = (dev.kernel("qx_cb_up")?, dev.kernel("qx_cb_down")?);
    let (cb, bb, way, xr, gu, part) = (Buf::new(codes.len())?, Buf::new(bstride)?, Buf::new(8)?, Buf::new(T * 4)?, Buf::new(UP * 4)?, Buf::new(T * 4)?);
    cb.write(0, &codes)?;
    bb.write(0, &books)?;
    way.write(0, &[0i64])?;
    xr.write(0, &x)?;
    let s = Stream::DEFAULT;
    // SAFETY: one pair (way [0]), its place the buffers above, k 1, one run of the second product.
    unsafe {
        launch!(up, [(UP / 32) as u32, 1, 1], [256, 1, 1], s; cb.ptr, codes.len() as u64, bb.ptr, bstride as u64, way.ptr, xr.ptr, gu.ptr, 1i32, 1i32)?;
        launch!(down, [1, 1, 1], [256, 1, 1], s; cb.ptr, codes.len() as u64, bb.ptr, bstride as u64, way.ptr, gu.ptr, part.ptr, 1i32, f32::MAX, 1i32)?;
    }
    dev.sync()?;
    let (mut g_gu, mut g_out) = (vec![0f32; UP], vec![0f32; T]);
    gu.read(0, &mut g_gu)?;
    part.read(0, &mut g_out)?;
    let kernel = crate::cpu_experts::Kernel::best();
    let (um, dm) = (GlmMat { fmt, rows: UP, cols: T, codes: &uc, book: &us }, GlmMat { fmt, rows: DN, cols: T, codes: &dc, book: &ds });
    let mut c_gu = vec![0f32; UP];
    dot_glm(kernel, &um, &x, 0, UP, &mut c_gu);
    // the activations as qx_cb_down makes them, from the GPU's sums, so that each product is checked alone
    let bf = |v: f32| f32_of(bf16_of(v));
    let act: Vec<f32> = (0..DN).map(|n| bf(g_gu[n] / (1.0 + (-g_gu[n]).exp()) * g_gu[DN + n])).collect();
    let mut c_out = vec![0f32; T];
    for o0 in (0..T).step_by(256) {
        down_glm(kernel, &dm, &act, o0, &mut c_out[o0..]);
    }
    let rel = |g: &[f32], c: &[f32]| {
        (g.iter().zip(c).map(|(a, b)| (*a as f64 - *b as f64).powi(2)).sum::<f64>() / c.iter().map(|b| (*b as f64).powi(2)).sum::<f64>().max(1e-30)).sqrt()
    };
    let (ru, rd) = (rel(&g_gu, &c_gu), rel(&g_out, &c_out));
    println!("layer {l} expert {e} ({}): GPU against CPU ({}), relative difference gate-up {ru:.2e}, down {rd:.2e}", fmt.name(), kernel.name());
    if ru > 1e-4 || rd > 1e-4 {
        return err(format!("the GPU's trellis products differ from the CPU's ({ru:.2e}, {rd:.2e})"));
    }
    Ok(())
}
