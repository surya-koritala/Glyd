//! `glyd-engine kcheck35`: Qwen3.5's kernels (kernels/qwen35.cu) against the CPU reference (src/qwen35_ref.rs, and the plain formulas here) on random inputs of the 2026
//! models' head sizes. Each line says how far the GPU is from the CPU and the bound it must keep. What is exact on both sides is required equal (the residual sum, the convolution's
//! state, the values copied into the cache). A bf16 output both sides compute with the same roundings, in another order of a sum or with another `exp`, is bounded in places of
//! bf16 on every value. A bf16 output of a long sum (the delta rule, the attention) is bounded relative to the largest value, its places shown but not bounded: where such a sum
//! cancels, a small output is many places off while the difference stays small. The attention rounds its probabilities to bf16 for the values (as FlashAttention does), the fp64
//! CPU does not; a decode step's attention sums its keys in parts, so it is not the prompt kernel's bits. The fp32 states are bounded relative to their largest value. Exit status
//! 1 where a bound is not kept. The gated attention is the attention kind's kernels (kernels/attn.cu) with the Qwen3.5 layout's parameters (`dynamic::qwen35::attention`).

use crate::dev::{Buf, Dev};
use crate::dynamic::attn::{AttnArgs, KernAttn, SPLITS};
use crate::ffi::Result;
use crate::kern::Kern35;
use crate::qwen35_ref::{bf16, conv_step, delta_step, gated_norm, l2norm, sigmoid, softplus};

struct Rng(u64);

impl Rng {
    /// Uniform in [lo, hi).
    fn f(&mut self, lo: f32, hi: f32) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        lo + (hi - lo) * ((self.0 >> 40) as f32 / (1u64 << 24) as f32)
    }

    /// n bf16 values in [lo, hi), as f32.
    fn v(&mut self, n: usize, lo: f32, hi: f32) -> Vec<f32> {
        (0..n).map(|_| bf16(self.f(lo, hi))).collect()
    }
}

fn bits(v: &[f32]) -> Vec<u16> {
    v.iter().map(|x| (bf16(*x).to_bits() >> 16) as u16).collect()
}

fn floats(v: &[u16]) -> Vec<f32> {
    v.iter().map(|&b| f32::from_bits((b as u32) << 16)).collect()
}

fn up(v: &[f32]) -> Result<Buf> {
    let b = Buf::new(v.len() * 2)?;
    b.write(0, &bits(v))?;
    Ok(b)
}

fn up32(v: &[f32]) -> Result<Buf> {
    let b = Buf::new(v.len() * 4)?;
    b.write(0, v)?;
    Ok(b)
}

fn down(b: &Buf, n: usize) -> Result<Vec<f32>> {
    let mut v = vec![0u16; n];
    b.read(0, &mut v)?;
    Ok(floats(&v))
}

fn down32(b: &Buf, n: usize) -> Result<Vec<f32>> {
    let mut v = vec![0f32; n];
    b.read(0, &mut v)?;
    Ok(v)
}

/// Places of bf16 between two bf16 values (as f32), and how many are equal.
fn ulps(a: &[f32], b: &[f32]) -> (u32, usize) {
    let key = |x: f32| {
        let u = (x.to_bits() >> 16) as i32;
        if u & 0x8000 != 0 {
            -(u & 0x7fff)
        } else {
            u
        }
    };
    let mut worst = 0u32;
    let mut same = 0usize;
    for (x, y) in a.iter().zip(b) {
        let d = (key(*x) - key(*y)).unsigned_abs();
        worst = worst.max(d);
        same += (d == 0) as usize;
    }
    (worst, same)
}

/// The largest difference relative to the largest value of `b`.
fn rel(a: &[f32], b: &[f32]) -> f64 {
    let scale = b.iter().fold(0f32, |m, x| m.max(x.abs())).max(1e-30) as f64;
    a.iter().zip(b).map(|(x, y)| (x - y).abs() as f64).fold(0.0, f64::max) / scale
}

/// A bf16 place of the largest value.
const PLACE: f64 = 1.0 / 256.0;

struct Tally {
    failed: Vec<String>,
}

impl Tally {
    fn keep(&mut self, ok: bool, name: &str) -> &'static str {
        if !ok {
            self.failed.push(name.to_string());
        }
        if ok {
            "ok  "
        } else {
            "FAIL"
        }
    }

    /// Every value within `bound` places of bf16.
    fn ulp(&mut self, name: &str, gpu: &[f32], cpu: &[f32], bound: u32) {
        let (w, same) = ulps(gpu, cpu);
        let v = self.keep(w <= bound && gpu.iter().all(|x| x.is_finite()), name);
        println!(
            "{v} {name}: {} values, {same} equal ({:.2}%), at most {w} bf16 places apart (bound {bound})",
            gpu.len(),
            100.0 * same as f64 / gpu.len().max(1) as f64
        );
    }

    /// The largest difference within `bound` of the largest value; for bf16 outputs (`bf16_out`) the places are shown too, not bounded.
    fn rel(&mut self, name: &str, gpu: &[f32], cpu: &[f32], bound: f64, bf16_out: bool) {
        let r = rel(gpu, cpu);
        let v = self.keep(r <= bound && gpu.iter().all(|x| x.is_finite()), name);
        let places = if bf16_out {
            let (w, same) = ulps(gpu, cpu);
            format!(", {same} equal ({:.2}%), at most {w} bf16 places apart (not bounded)", 100.0 * same as f64 / gpu.len().max(1) as f64)
        } else {
            String::new()
        };
        println!("{v} {name}: {} values, largest difference {r:.2e} of the largest value (bound {bound:.1e}){places}", gpu.len());
    }

    fn exact(&mut self, name: &str, gpu: &[f32], cpu: &[f32]) {
        let ok = gpu.len() == cpu.len() && gpu.iter().zip(cpu).all(|(a, b)| a.to_bits() == b.to_bits());
        let v = self.keep(ok, name);
        println!("{v} {name}: {} values, {}", gpu.len(), if ok { "bit for bit" } else { "NOT bit for bit" });
    }
}

/// The gated delta rule's inputs for `n` tokens of `nk` key heads and `nv` value heads (128 dims each): the convolution's outputs [n, C], b and a [n, 2 nv], A_log, dt_bias.
struct Delta {
    n: usize,
    nk: usize,
    nv: usize,
    xc: Vec<f32>,
    ba: Vec<f32>,
    a_log: Vec<f32>,
    dt_bias: Vec<f32>,
    s0: Vec<f32>,
}

impl Delta {
    fn new(r: &mut Rng, n: usize, nk: usize, nv: usize) -> Delta {
        let c = 2 * nk * 128 + nv * 128;
        Delta {
            n,
            nk,
            nv,
            xc: r.v(n * c, -1.0, 1.0),
            ba: r.v(n * 2 * nv, -3.0, 3.0),
            a_log: (0..nv).map(|_| r.f(-1.0, 1.0)).collect(),
            dt_bias: r.v(nv, -1.0, 1.0),
            s0: (0..nv * 128 * 128).map(|_| r.f(-0.1, 0.1)).collect(),
        }
    }

    fn c(&self) -> usize {
        2 * self.nk * 128 + self.nv * 128
    }

    /// The CPU's outputs [n, nv 128] (bf16) and final state, token by token (`round_qk`: the normed queries and keys rounded to bf16 first, as the chunked kernel keeps them).
    fn cpu(&self, round_qk: bool) -> (Vec<f32>, Vec<f32>) {
        let (c, kd, r) = (self.c(), self.nk * 128, self.nv / self.nk);
        let mut s = self.s0.clone();
        let mut out = vec![0f32; self.n * self.nv * 128];
        let scale = 1.0 / (128f32).sqrt();
        for t in 0..self.n {
            let x = &self.xc[t * c..(t + 1) * c];
            for h in 0..self.nv {
                let kh = h / r;
                let mut q = x[kh * 128..(kh + 1) * 128].to_vec();
                let mut k = x[kd + kh * 128..kd + (kh + 1) * 128].to_vec();
                l2norm(&mut q);
                l2norm(&mut k);
                if round_qk {
                    q.iter_mut().for_each(|e| *e = bf16(*e));
                    k.iter_mut().for_each(|e| *e = bf16(*e));
                }
                q.iter_mut().for_each(|e| *e *= scale);
                let v = &x[2 * kd + h * 128..2 * kd + (h + 1) * 128];
                let beta = sigmoid(self.ba[t * 2 * self.nv + h]);
                let g = -self.a_log[h].exp() * softplus(self.ba[t * 2 * self.nv + self.nv + h] + self.dt_bias[h]);
                let o = delta_step(&mut s[h * 128 * 128..(h + 1) * 128 * 128], &q, &k, v, g, beta);
                for (j, x) in o.into_iter().enumerate() {
                    out[(t * self.nv + h) * 128 + j] = bf16(x);
                }
            }
        }
        (out, s)
    }

    /// The GPU's outputs and final state, by the recurrent kernel or the chunked one.
    fn gpu(&self, k: &Kern35, dev: &Dev, chunked: bool) -> Result<(Vec<f32>, Vec<f32>)> {
        let (xc, ba, al, dt, st) = (up(&self.xc)?, up(&self.ba)?, up32(&self.a_log)?, up(&self.dt_bias)?, up32(&self.s0)?);
        let core = Buf::zeroed(self.n * self.nv * 128 * 2)?;
        // SAFETY: buffers made here of the sizes the kernel reads and writes; read after a synchronize.
        unsafe {
            k.delta(
                dev.stream,
                chunked,
                xc.ptr,
                self.c() as i32,
                ba.ptr,
                2 * self.nv as i32,
                al.ptr,
                dt.ptr,
                st.ptr,
                core.ptr,
                self.n as i32,
                self.nk as i32,
                self.nv as i32,
            )?
        };
        dev.sync()?;
        Ok((down(&core, self.n * self.nv * 128)?, down32(&st, self.s0.len())?))
    }
}

/// The Qwen3.5 layout's attention arguments for n rows of nq query heads over nk key-value heads (heads of 256, the rotary on 2 `half` dims), the pointers null.
fn args35(n: usize, nq: usize, nk: usize, half: usize, eps: f32) -> AttnArgs {
    crate::dynamic::qwen35::attention(256, nq, nk, 2 * half, (0, 0), 0).args(n, eps)
}

/// Head-256 attention of `n` rows at positions p0 .. over random keys (nq query heads over nk key-value heads), times sigmoid(gate), against a plain fp64 softmax: the prompt's
/// kernels (on the tensor cores, and a warp a head), and each row as a decode step of its own (the keys in SPLITS parts, then combined).
#[allow(clippy::too_many_arguments)]
fn attn_case(t: &mut Tally, k: &KernAttn, lib: &crate::ffi::Lib, dev: &Dev, r: &mut Rng, p0: usize, n: usize, nq: usize, nk: usize, what: &str) -> Result<()> {
    let stride = nq * 512 + 2 * nk * 256;
    let (keys, vals) = (r.v((p0 + n) * nk * 256, -2.0, 2.0), r.v((p0 + n) * nk * 256, -2.0, 2.0));
    let q = r.v(n * nq * 256, -2.0, 2.0);
    let gate = r.v(n * stride, -4.0, 4.0);
    let pos: Vec<i32> = (0..n).map(|i| (p0 + i) as i32).collect();
    let g = nq / nk;
    let mut want = vec![0f32; n * nq * 256];
    for i in 0..n {
        for h in 0..nq {
            let kvh = h / g;
            let qr = &q[(i * nq + h) * 256..][..256];
            let sc: Vec<f64> =
                (0..=p0 + i).map(|j| qr.iter().zip(&keys[(j * nk + kvh) * 256..][..256]).map(|(a, b)| (*a as f64) * (*b as f64)).sum::<f64>() / 16.0).collect();
            let m = sc.iter().cloned().fold(f64::MIN, f64::max);
            let e: Vec<f64> = sc.iter().map(|x| (x - m).exp()).collect();
            let l: f64 = e.iter().sum();
            for d in 0..256 {
                let o: f64 = e.iter().enumerate().map(|(j, p)| p * vals[(j * nk + kvh) * 256 + d] as f64).sum::<f64>() / l;
                let gt = gate[i * stride + h * 512 + 256 + d];
                want[(i * nq + h) * 256 + d] = bf16(bf16(o as f32) * sigmoid(gt));
            }
        }
    }
    let (qb, gb, kb, vb, pb) = (up(&q)?, up(&gate)?, up(&keys)?, up(&vals)?, Buf::new(4 * n)?);
    pb.write(0, &pos)?;
    let outs = [Buf::zeroed(want.len() * 2)?, Buf::zeroed(want.len() * 2)?, Buf::zeroed(want.len() * 2)?];
    let part = Buf::zeroed(n * nq * SPLITS * 258 * 4)?;
    let mut a = args35(n, nq, nk, 32, 1e-6);
    (a.qo, a.kc, a.vc, a.cap, a.kpos, a.gate, a.part) = (qb.ptr, kb.ptr, vb.ptr, (p0 + n) as i32, pb.ptr, gb.ptr + 512, part.ptr);
    // SAFETY: buffers made here of the sizes the kernels read and write; read after a synchronize.
    unsafe {
        k.prompt(lib, dev.stream, &AttnArgs { out: outs[0].ptr, ..a }, p0)?;
        k.rows(dev.stream, &AttnArgs { out: outs[1].ptr, ..a })?;
        k.decode(dev.stream, &AttnArgs { out: outs[2].ptr, ..a })?;
    }
    dev.sync()?;
    for (o, how) in outs.iter().zip([
        "a prompt's rows on the tensor cores (the GPU library's kernel)".to_string(),
        "a prompt's rows, a warp a head".to_string(),
        format!("each row a decode step (the keys in {SPLITS} parts, combined: not the prompt kernels' bits)"),
    ]) {
        t.rel(&format!("attention {what}: {how}, times sigmoid(gate), against the CPU"), &down(o, want.len())?, &want, 1e-2, true);
    }
    Ok(())
}

/// The gated delta rule of nk key heads and nv value heads: 150 tokens token by token and in chunks of 64 against the CPU, the chunks against token by token, one decode step.
fn delta_case(t: &mut Tally, k: &Kern35, dev: &Dev, r: &mut Rng, nk: usize, nv: usize, what: &str) -> Result<()> {
    let d = Delta::new(r, 150, nk, nv);
    let (want, want_s) = d.cpu(false);
    let (got, got_s) = d.gpu(k, dev, false)?;
    t.rel(&format!("gated delta rule {what}, token by token: outputs (150 tokens)"), &got, &want, PLACE, true);
    t.rel(&format!("gated delta rule {what}, token by token: the state after"), &got_s, &want_s, 1e-4, false);
    let (want_r, want_rs) = d.cpu(true);
    let (got_c, got_cs) = d.gpu(k, dev, true)?;
    t.rel(&format!("gated delta rule {what} in chunks of 64 (normed q, k in bf16): outputs (150 tokens: two chunks and 22)"), &got_c, &want_r, PLACE, true);
    t.rel(&format!("gated delta rule {what} in chunks: the state after"), &got_cs, &want_rs, 1e-4, false);
    t.rel(&format!("gated delta rule {what}: chunks against token by token (q, k in fp32), outputs"), &got_c, &got, 2e-2, true);
    let one = Delta::new(r, 1, nk, nv);
    let (w1, ws1) = one.cpu(false);
    let (g1, gs1) = one.gpu(k, dev, false)?;
    t.rel(&format!("gated delta rule {what}, a decode step: outputs"), &g1, &w1, PLACE, true);
    t.rel(&format!("gated delta rule {what}, a decode step: the state after"), &gs1, &ws1, 1e-5, false);
    Ok(())
}

pub fn run() -> Result<bool> {
    let dev = Dev::new(0)?;
    let k = Kern35::load(&dev)?;
    let (ka, lib) = (KernAttn::load(&dev)?, crate::ffi::Lib::find()?);
    let s = dev.stream;
    let mut r = Rng(0x2026_1006_0035_0001);
    let mut t = Tally { failed: Vec::new() };
    println!("{} (sm_{}{}): Qwen3.5's kernels against the CPU", dev.name, dev.cc.0, dev.cc.1);

    // the zero-centered norms, over the 9B's hidden size
    {
        let (rows, kk, eps) = (5usize, 4096usize, 1e-6f32);
        let (x, y, w) = (r.v(rows * kk, -4.0, 4.0), r.v(rows * kk, -4.0, 4.0), r.v(kk, -0.5, 0.5));
        let mut want = Vec::new();
        let mut want_add = Vec::new();
        let mut want_x = Vec::new();
        for i in 0..rows {
            let xr = &x[i * kk..(i + 1) * kk];
            let sc = 1.0 / (xr.iter().map(|v| v * v).sum::<f32>() / kk as f32 + eps).sqrt();
            want.extend(xr.iter().zip(&w).map(|(v, ww)| bf16(v * sc * (1.0 + ww))));
            let z: Vec<f32> = xr.iter().zip(&y[i * kk..(i + 1) * kk]).map(|(a, b)| a + b).collect();
            let sc = 1.0 / (z.iter().map(|v| v * v).sum::<f32>() / kk as f32 + eps).sqrt();
            want_add.extend(z.iter().zip(&w).map(|(v, ww)| bf16(v * sc * (1.0 + ww))));
            want_x.extend(z.iter().map(|v| bf16(*v)));
        }
        let (xb, yb, wb, out) = (up(&x)?, up(&y)?, up(&w)?, Buf::new(rows * kk * 2)?);
        // SAFETY: buffers made here of the sizes the kernels read and write; read after a synchronize.
        unsafe { k.rmsnorm(s, xb.ptr, wb.ptr, out.ptr, rows as u32, kk as i32, eps)? };
        dev.sync()?;
        t.ulp("norm (1 + w)", &down(&out, rows * kk)?, &want, 1);
        // SAFETY: as above.
        unsafe { k.rmsnorm_add(s, xb.ptr, yb.ptr, wb.ptr, out.ptr, rows as u32, kk as i32, eps)? };
        dev.sync()?;
        t.ulp("norm (1 + w) with the residual: the norm", &down(&out, rows * kk)?, &want_add, 1);
        t.exact("norm (1 + w) with the residual: the sum", &down(&xb, rows * kk)?, &want_x);
    }

    // the queries' and keys' norm and rotary, the cache's writes (the 9B's heads: 16 query, 4 key-value)
    {
        let (n, nq, nk, half, eps, p0) = (7usize, 16usize, 4usize, 32usize, 1e-6f32, 40usize);
        let stride = nq * 512 + 2 * nk * 256;
        let qkv = r.v(n * stride, -3.0, 3.0);
        let (qw, kw) = (r.v(256, -0.5, 0.5), r.v(256, -0.5, 0.5));
        let (cs, sn) = crate::weights::rope_tables(1e7, 2 * half, p0 + n);
        let (cf, sf) = (floats(&cs), floats(&sn));
        let pos: Vec<i32> = (0..n).map(|i| (p0 + i) as i32).collect();
        let head = |x: &[f32], w: &[f32], p: usize| -> Vec<f32> {
            let sc = 1.0 / (x.iter().map(|v| v * v).sum::<f32>() / 256.0 + eps).sqrt();
            let nrm: Vec<f32> = x.iter().zip(w).map(|(v, ww)| v * sc * (1.0 + ww)).collect();
            (0..256)
                .map(|i| {
                    let v = if i < half {
                        nrm[i] * cf[p * half + i] - nrm[i + half] * sf[p * half + i]
                    } else if i < 2 * half {
                        nrm[i] * cf[p * half + i - half] + nrm[i - half] * sf[p * half + i - half]
                    } else {
                        nrm[i]
                    };
                    bf16(v)
                })
                .collect()
        };
        let (mut wq, mut wk, mut wv) = (Vec::new(), vec![0f32; (p0 + n) * nk * 256], vec![0f32; (p0 + n) * nk * 256]);
        for i in 0..n {
            let row = &qkv[i * stride..(i + 1) * stride];
            for h in 0..nq {
                wq.extend(head(&row[h * 512..h * 512 + 256], &qw, p0 + i));
            }
            for h in 0..nk {
                let kk = head(&row[nq * 512 + h * 256..nq * 512 + (h + 1) * 256], &kw, p0 + i);
                wk[((p0 + i) * nk + h) * 256..][..256].copy_from_slice(&kk);
                wv[((p0 + i) * nk + h) * 256..][..256].copy_from_slice(&row[nq * 512 + nk * 256 + h * 256..][..256]);
            }
        }
        let (qb, qwb, kwb, cb, sb) = (up(&qkv)?, up(&qw)?, up(&kw)?, Buf::new(cs.len() * 2)?, Buf::new(sn.len() * 2)?);
        cb.write(0, &cs)?;
        sb.write(0, &sn)?;
        let pb = Buf::new(4 * n)?;
        pb.write(0, &pos)?;
        let (qo, kc, vc) = (Buf::zeroed(n * nq * 256 * 2)?, Buf::zeroed(wk.len() * 2)?, Buf::zeroed(wv.len() * 2)?);
        let mut a = args35(n, nq, nk, half, eps);
        (a.src, a.qw, a.kw, a.cs, a.sn, a.qpos) = (qb.ptr, qwb.ptr, kwb.ptr, cb.ptr, sb.ptr, pb.ptr);
        (a.qo, a.kc, a.vc, a.cap) = (qo.ptr, kc.ptr, vc.ptr, (p0 + n) as i32);
        // SAFETY: as above.
        unsafe { ka.prep(s, &a, nq + nk, (crate::kern::NextPf::default(), 0))? };
        dev.sync()?;
        t.ulp("query norm and rotary (64 of 256 dims)", &down(&qo, wq.len())?, &wq, 1);
        t.ulp("key norm and rotary into the cache", &down(&kc, wk.len())?, &wk, 1);
        t.exact("values into the cache", &down(&vc, wv.len())?, &wv);
    }

    // the attention: a short context (the keys in 2 of the 16 parts), then 1,500 positions (every part holds keys), with the 9B's 4 query heads to a key-value head and the 27B's 6
    attn_case(&mut t, &ka, &lib, &dev, &mut r, 40, 7, 16, 4, "(16 over 4 heads, positions 40 to 46)")?;
    attn_case(&mut t, &ka, &lib, &dev, &mut r, 1500, 3, 16, 4, "(16 over 4 heads, positions 1,500 to 1,502)")?;
    attn_case(&mut t, &ka, &lib, &dev, &mut r, 1500, 3, 24, 4, "(24 over 4 heads, positions 1,500 to 1,502)")?;

    // the convolution and SiLU (the 0.8B's channels: 6,144), 37 tokens from a random state; the reference sums in the kernel's order and rounds once, after SiLU (vLLM's)
    {
        let (n, c, xs) = (37usize, 6144usize, 6144 + 2048);
        let x = r.v(n * xs, -3.0, 3.0);
        let w = r.v(c * 4, -0.6, 0.6);
        let st0 = r.v(3 * c, -3.0, 3.0);
        // the reference keeps a channel's state [kernel - 1] together; the kernel keeps [3, C]
        let mut st: Vec<f32> = (0..c).flat_map(|ch| (0..3).map(move |j| (ch, j))).map(|(ch, j)| st0[j * c + ch]).collect();
        let mut want = Vec::new();
        for i in 0..n {
            want.extend(conv_step(&w, &mut st, &x[i * xs..i * xs + c], 4));
        }
        let want_st: Vec<f32> = (0..3).flat_map(|j| (0..c).map(move |ch| (ch, j))).map(|(ch, j)| st[ch * 3 + j]).collect();
        let (xb, wb, sb, yb) = (up(&x)?, up(&w)?, up(&st0)?, Buf::zeroed(n * c * 2)?);
        // SAFETY: as above.
        unsafe { k.conv(s, xb.ptr, xs as i32, wb.ptr, sb.ptr, yb.ptr, n as i32, c as i32)? };
        dev.sync()?;
        t.ulp("causal convolution and SiLU (one rounding, vLLM's)", &down(&yb, n * c)?, &want, 1);
        t.exact("causal convolution: the state after", &down(&sb, 3 * c)?, &want_st);
    }

    // the gated delta rule, a few heads at each ratio of value heads to key heads of the 2026 models: 1 (0.8B, 2B), 2 (4B, 9B), 3 (27B)
    delta_case(&mut t, &k, &dev, &mut r, 2, 2, "(2 value heads over 2 key heads)")?;
    delta_case(&mut t, &k, &dev, &mut r, 2, 4, "(4 over 2)")?;
    delta_case(&mut t, &k, &dev, &mut r, 2, 6, "(6 over 2)")?;

    // the gated norm (32 value heads; its weight in bf16, as the loader holds it) and the small product
    {
        let (n, nv, eps) = (9usize, 32usize, 1e-6f32);
        let (o, z, w) = (r.v(n * nv * 128, -2.0, 2.0), r.v(n * nv * 128, -4.0, 4.0), r.v(128, 0.5, 1.5));
        let mut want = Vec::new();
        for i in 0..n * nv {
            want.extend(gated_norm(&o[i * 128..(i + 1) * 128], &z[i * 128..(i + 1) * 128], &w, eps));
        }
        let (ob, zb, wb) = (up(&o)?, up(&z)?, up32(&w)?);
        // SAFETY: as above.
        unsafe { k.gated_norm(s, ob.ptr, zb.ptr, (nv * 128) as i32, wb.ptr, n as u32, nv as i32, eps)? };
        dev.sync()?;
        t.ulp("gated norm (the weight in bf16, one rounding: vLLM's RMSNormGated)", &down(&ob, n * nv * 128)?, &want, 1);

        let (m, rows, cols) = (11usize, 32usize, 1024usize);
        let (x, w) = (r.v(m * cols, -2.0, 2.0), r.v(rows * cols, -0.2, 0.2));
        let want: Vec<f32> = (0..m * rows).map(|e| bf16((0..cols).map(|c| x[(e / rows) * cols + c] * w[(e % rows) * cols + c]).sum())).collect();
        let (xb, wb, yb) = (up(&x)?, up(&w)?, Buf::zeroed(m * rows * 2)?);
        // SAFETY: as above.
        unsafe { k.small_linear(s, xb.ptr, cols as i32, wb.ptr, yb.ptr, rows as i32, m as u32, rows as i32, cols as i32)? };
        dev.sync()?;
        t.ulp("the small product (b and a of the 0.8B)", &down(&yb, m * rows)?, &want, 1);
    }

    if t.failed.is_empty() {
        println!("all of Qwen3.5's kernels within their bounds");
        Ok(true)
    } else {
        println!("FAILED: {}", t.failed.join("; "));
        Ok(false)
    }
}
