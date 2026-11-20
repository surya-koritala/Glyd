//! `glyd-engine moecheck [CASE...] [--speed] [--no-check] [--dump DIR | --cpu-dump DIR]`: the mixture-of-experts layer on the GPU (src/moe.rs: kernels/moe.cu, and the library's grouped products over Int5, Int6 and Int4 packs) against the CPU
//! reference (src/moe_ref.rs, on the weights as the packs decode them), on random routers and experts at the shapes of the four models' experts (8 of 128, 8 of 288, 22 of 512 choices; widths cut down so that the
//! host packs and multiplies them in a minute) and token counts from 1 (a decode step) to a prompt chunk. Each line is a case, a layout and a count of tokens, with how far the GPU is from the CPU at each stage
//! and the bound it keeps: the router's logits (fp32), its experts and weights (equal; a tie the CPU breaks otherwise is allowed where the two scores are equal to 1e-6), the plan of the pairs sorted by expert (exact), the
//! experts' intermediate rows, the routed sum, and the sum with the shared expert; a bf16 output is bounded by its largest value (the sums are in another order, the roundings are the same). Then single experts by their own packs at offsets in the stack (the first product of the first, a middle and the last expert against the CPU's: the strides are right), and the decode step as a
//! CUDA graph, replayed twice, against the eager one (the same bits). `--speed`: the layer's time at the four models' real shapes (packs of arbitrary bytes: the kernels' time does not depend on them) at 1 token and up.
//! The Nemotron case runs twice, with the second matrices in a block layout (the library's dense product) and in the row layout (the sparse product of src/moe_rows.rs, at every count of tokens, its CPU reference the same dense
//! product on the weights the packs decode to: the sums over every neuron against the sums over the neurons that are not zero). `--speed` times both at the real shape, on activations that are 92% zeros.
//! `--dump DIR` writes a case's inputs, the weights as decoded and the GPU's outputs for tools/moe_ref.py (a PyTorch reference); `--cpu-dump DIR` the same with the CPU reference in the GPU's place (no GPU used). Exit status 1
//! where a bound is not kept.

use crate::args::Args;
use crate::dev::{memset_async, Buf, Dev, Timer};
use crate::ffi::{err, Error, Lib, Result};
use crate::moe::{expert_strides, Act, ExpertStack, Joined, MoeCfg, MoeLayer, MoeScratch, MoeWeights, Router, Scoring, Shared};
use crate::moe_ref::{plan_of, read_plan, route, scores, RefLayer, RefShared};
use crate::moe_rows::{row_strides, Down, RowStack, RowsJoined};
use crate::qwen35_ref::bf16;
use crate::weights::upload_tiered;
use glyd_gpu::pack::{
    pack_int4, pack_int5, pack_int6, pack_rows, transpose, unpack_int4, unpack_int5, unpack_int6, unpack_rows, Tiered, INT4_TAG, INT5_TAG, INT6_TAG,
};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};

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
    fn uni(&mut self, n: usize, lo: f32, hi: f32) -> Vec<f32> {
        (0..n).map(|_| bf16(self.f(lo, hi))).collect()
    }

    /// n bf16 words of a bell (a sum of four uniforms of width `scale`: a trained matrix's spread, about 0.58 `scale`).
    fn bell(&mut self, n: usize, scale: f32) -> Vec<u16> {
        (0..n).map(|_| (bf16((self.f(-0.5, 0.5) + self.f(-0.5, 0.5) + self.f(-0.5, 0.5) + self.f(-0.5, 0.5)) * scale).to_bits() >> 16) as u16).collect()
    }
}

fn bits(v: &[f32]) -> Vec<u16> {
    v.iter().map(|x| (bf16(*x).to_bits() >> 16) as u16).collect()
}

fn floats(v: &[u16]) -> Vec<f32> {
    v.iter().map(|&b| f32::from_bits((b as u32) << 16)).collect()
}

fn up_bf(v: &[f32]) -> Result<Buf> {
    let b = Buf::new(v.len() * 2)?;
    b.write(0, &bits(v))?;
    Ok(b)
}

fn up_f32(v: &[f32]) -> Result<Buf> {
    let b = Buf::new(v.len() * 4)?;
    b.write(0, v)?;
    Ok(b)
}

fn rd<T: Copy + Default>(b: &Buf, n: usize) -> Result<Vec<T>> {
    let mut v = vec![T::default(); n];
    b.read(0, &mut v)?;
    Ok(v)
}

/// The largest difference relative to the largest value of `b`; infinite where a difference is not a number (a NaN of the GPU's) or the lengths differ.
fn rel(a: &[f32], b: &[f32]) -> f64 {
    if a.len() != b.len() {
        return f64::INFINITY;
    }
    let scale = b.iter().fold(0f32, |m, x| m.max(x.abs())).max(1e-30) as f64;
    a.iter().zip(b).fold(0.0, |m: f64, (x, y)| match (x - y).abs() as f64 {
        d if d.is_nan() => f64::INFINITY,
        d => m.max(d),
    }) / scale
}

/// `n` jobs on as many threads as the machine has, the results in order.
fn par<T: Send>(n: usize, f: &(dyn Fn(usize) -> T + Sync)) -> Vec<T> {
    let threads = std::thread::available_parallelism().map_or(1, |x| x.get()).min(n.max(1));
    let next = AtomicUsize::new(0);
    let parts: Vec<Vec<(usize, T)>> = std::thread::scope(|sc| {
        let hs: Vec<_> = (0..threads)
            .map(|_| {
                sc.spawn(|| {
                    let mut v = Vec::new();
                    loop {
                        let i = next.fetch_add(1, Ordering::SeqCst);
                        if i >= n {
                            return v;
                        }
                        v.push((i, f(i)));
                    }
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().expect("a worker")).collect()
    });
    let mut all: Vec<(usize, T)> = parts.into_iter().flatten().collect();
    all.sort_by_key(|x| x.0);
    all.into_iter().map(|x| x.1).collect()
}

/// A block layout under test: the packer and its decode (the bits the GPU decodes to), and the bits of its codes (its row layout's too).
struct Fmt {
    name: &'static str,
    tag: u32,
    bits: u32,
    pack: fn(&[u16], usize, usize) -> Option<Tiered>,
    unpack: fn(&Tiered, &mut [u16]) -> bool,
    /// The experts held lossless: the int5 case's weights (what its packs decode to) packed in the tiered layout, a stack each (`ExpertStack::lossless`).
    lossless: bool,
}

const FMTS: [Fmt; 3] = [
    Fmt { name: "int5", tag: INT5_TAG, bits: 5, pack: pack_int5, unpack: unpack_int5, lossless: false },
    Fmt { name: "int6", tag: INT6_TAG, bits: 6, pack: pack_int6, unpack: unpack_int6, lossless: false },
    Fmt { name: "int4", tag: INT4_TAG, bits: 4, pack: pack_int4, unpack: unpack_int4, lossless: false },
];

const LOSSLESS: Fmt = Fmt { name: "tiered (lossless)", tag: INT5_TAG, bits: 5, pack: pack_int5, unpack: unpack_int5, lossless: true };

/// The exact FP4 layout, timed alone (`--speed`: its packs are an MX FP4 checkpoint's, made of no bf16 weights).
const FP4: Fmt = Fmt {
    name: "fp4 (exact)",
    tag: glyd_gpu::pack::FP4_TAG,
    bits: 4,
    pack: |w, r, c| mx_fp4(w, r, c, false),
    unpack: glyd_gpu::pack::unpack_fp4,
    lossless: false,
};
const FP4R: Fmt = Fmt {
    name: "fp4 (exact, packed scales)",
    tag: glyd_gpu::pack::FP4R_TAG,
    bits: 4,
    pack: |w, r, c| mx_fp4(w, r, c, true),
    unpack: glyd_gpu::pack::unpack_fp4,
    lossless: false,
};

/// bf16 weights [rows, cols] as an MX FP4 checkpoint holds them (each run of 32 of a row: E2M1 codes, two a byte, the first in the low four bits, and its power-of-two scale, the
/// smallest that holds the run's largest value), packed in the FP4 layout (`packed`: with packed scales). The case's reference reads the pack's own values back.
fn mx_fp4(w: &[u16], rows: usize, cols: usize, packed: bool) -> Option<Tiered> {
    const E2M1: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    if cols % 32 != 0 || w.len() != rows * cols {
        return None;
    }
    let (mut q, mut e) = (vec![0u8; rows * cols / 2], vec![0u8; rows * cols / 32]);
    for (g, run) in w.chunks(32).enumerate() {
        let v: Vec<f32> = run.iter().map(|&b| f32::from_bits((b as u32) << 16)).collect();
        let amax = v.iter().fold(0f32, |m, x| m.max(x.abs())).max(1e-30);
        let ex = (amax / 6.0).log2().ceil().clamp(-125.0, 1.0) as i32;
        e[g] = (ex + 127) as u8;
        for (j, &x) in v.iter().enumerate() {
            let y = (x.abs() / 2f32.powi(ex)).min(6.0);
            let m = (0..8).min_by(|&a, &b| (E2M1[a] - y).abs().total_cmp(&(E2M1[b] - y).abs())).unwrap_or(0) as u8;
            let code = m | if x < 0.0 && m != 0 { 8 } else { 0 };
            let i = g * 32 + j;
            q[i / 2] |= if i % 2 == 0 { code } else { code << 4 };
        }
    }
    if packed {
        glyd_gpu::pack::pack_fp4r_checked(&q, &e, rows, cols)
    } else {
        glyd_gpu::pack::pack_fp4_checked(&q, &e, rows, cols)
    }
}

/// A case: a layer's config, whether its router has a selection bias and a per-expert scale, its shared expert (intermediate size, gate), the counts of tokens to run, and the one whose inputs `--dump` writes.
/// `rows`: the second matrices in the row layout (the sparse product).
struct Case {
    name: &'static str,
    cfg: MoeCfg,
    bias: bool,
    escale: bool,
    shared: Option<(usize, bool)>,
    tokens: &'static [usize],
    dump_t: usize,
    rows: bool,
}

fn cases() -> Vec<Case> {
    let swiglu = Act::SwiGlu { limit: None };
    vec![
        // Qwen3.5/3.6's router and experts: softmax over bf16 logits, SwiGLU fused into the product, a shared expert with a sigmoid gate; the counts reach the prompt kernel (8 pairs a token: 8,800 of 128 experts)
        Case {
            name: "swiglu-8of128",
            cfg: MoeCfg { experts: 128, router_hidden: 512, hidden: 512, inter: 256, router: Router::softmax(8), act: swiglu },
            bias: false,
            escale: false,
            shared: Some((256, true)),
            tokens: &[1, 2, 5, 17, 33, 64, 200, 1100],
            dump_t: 33,
            rows: false,
        },
        // Gemma 4's: GELU fused, the per-expert scale, the dense MLP beside as the shared expert without a gate
        Case {
            name: "geglu-8of128",
            cfg: MoeCfg { experts: 128, router_hidden: 512, hidden: 512, inter: 192, router: Router::softmax(8), act: Act::GeGlu },
            bias: false,
            escale: true,
            shared: Some((192, false)),
            tokens: &[1, 9, 40, 300],
            dump_t: 40,
            rows: false,
        },
        // GLM-5.x's: sigmoid scores picked with a bias, x2.5, SwiGLU clamped (at 0.5 here, so that it bites; GLM's is 10) so that the activation is the kernel's, a shared expert without a gate
        Case {
            name: "clamp-8of288",
            cfg: MoeCfg { experts: 288, router_hidden: 512, hidden: 512, inter: 256, router: Router::sigmoid(8, 2.5), act: Act::SwiGlu { limit: Some(0.5) } },
            bias: true,
            escale: false,
            shared: Some((256, false)),
            tokens: &[1, 4, 33, 150],
            dump_t: 33,
            rows: false,
        },
        // Nemotron 3's: sigmoid with the bias, x5, 22 of 512, relu squared without a gate, the router over a width twice the experts' (the latent), the shared expert reading the router's width
        Case {
            name: "relu2-22of512",
            cfg: MoeCfg { experts: 512, router_hidden: 512, hidden: 256, inter: 384, router: Router::sigmoid(22, 5.0), act: Act::Relu2 },
            bias: true,
            escale: false,
            shared: Some((384, false)),
            tokens: &[1, 3, 16, 40, 128, 1600],
            dump_t: 40,
            rows: false,
        },
        // the same with the second matrices in the row layout: the sparse product (src/moe_rows.rs), every count of tokens
        // Kimi-K3's kind: situ, 16 a token, sigmoid scores with a bias, experts on a latent narrower than the router's input
        Case {
            name: "situ-16of128",
            cfg: MoeCfg {
                experts: 128,
                router_hidden: 512,
                hidden: 256,
                inter: 256,
                router: Router::sigmoid(16, 1.0),
                act: Act::Situ { beta: 4.0, linear: 25.0 },
            },
            bias: true,
            escale: false,
            shared: Some((512, false)),
            tokens: &[1, 3, 17, 40],
            dump_t: 17,
            rows: false,
        },
        Case {
            name: "relu2-rows-22of512",
            cfg: MoeCfg { experts: 512, router_hidden: 512, hidden: 256, inter: 384, router: Router::sigmoid(22, 5.0), act: Act::Relu2 },
            bias: true,
            escale: false,
            shared: Some((384, false)),
            tokens: &[1, 3, 16, 40, 128, 1600],
            dump_t: 40,
            rows: true,
        },
    ]
}

/// The host's copies of a layer's weights (the experts' as the packs decode them).
struct Host {
    router: Vec<f32>,
    bias: Option<Vec<f32>>,
    escale: Option<Vec<f32>>,
    up: Vec<f32>,
    down: Vec<f32>,
    s_up: Vec<f32>,
    s_down: Vec<f32>,
    s_gate: Option<Vec<f32>>,
}

/// E experts' matrices [rows, cols] (bf16 words) packed one at a time on all the threads and joined, as a loader of per-expert tensors does; and what they decode to.
fn pack_host(f: &Fmt, w: &[u16], experts: usize, rows: usize, cols: usize) -> Result<(Joined, Vec<f32>)> {
    let per = rows * cols;
    let packs: Vec<Option<Tiered>> = par(experts, &|e| (f.pack)(&w[e * per..(e + 1) * per], rows, cols));
    let packs: Vec<Tiered> = packs.into_iter().collect::<Option<_>>().ok_or_else(|| Error(format!("{}: a matrix of {rows} x {cols} did not pack", f.name)))?;
    let deq: Vec<Vec<f32>> = par(experts, &|e| {
        let mut out = vec![0u16; per];
        assert!((f.unpack)(&packs[e], &mut out));
        floats(&out)
    });
    Ok((Joined::of(&packs.iter().collect::<Vec<_>>())?, deq.concat()))
}

/// A dense matrix packed in the layout, and what it decodes to.
fn pack_one(f: &Fmt, w: &[u16], rows: usize, cols: usize) -> Result<(Tiered, Vec<f32>)> {
    let p = (f.pack)(w, rows, cols).ok_or_else(|| Error(format!("{}: a matrix of {rows} x {cols} did not pack", f.name)))?;
    let mut out = vec![0u16; rows * cols];
    assert!((f.unpack)(&p, &mut out));
    Ok((p, floats(&out)))
}

/// E experts' second matrices [rows, cols] (the transposes [cols, rows] in the row layout) packed one at a time on all the threads and joined; and what they decode to, [rows, cols] as the dense product multiplies by them.
fn pack_host_rows(f: &Fmt, w: &[u16], experts: usize, rows: usize, cols: usize) -> Result<(RowsJoined, Vec<f32>)> {
    let per = rows * cols;
    let packs: Vec<Option<Tiered>> = par(experts, &|e| pack_rows(f.bits, &transpose(&w[e * per..(e + 1) * per], rows, cols), cols, rows));
    let packs: Vec<Tiered> =
        packs.into_iter().collect::<Option<_>>().ok_or_else(|| Error(format!("{}: a matrix of {rows} x {cols} did not pack in the row layout", f.name)))?;
    let deq: Vec<Vec<f32>> = par(experts, &|e| {
        let mut out = vec![0u16; per];
        assert!(unpack_rows(&packs[e], &mut out));
        floats(&transpose(&out, cols, rows))
    });
    Ok((RowsJoined::of(&packs.iter().collect::<Vec<_>>())?, deq.concat()))
}

/// The second matrices of a layer on the host, in the layout of the case.
enum Dn {
    Block(Joined),
    Rows(RowsJoined),
}

/// A layer on the host: its weights as the packs decode them, and the packs to upload (the experts' joined; the shared expert's).
struct Built {
    host: Host,
    up: Joined,
    down: Dn,
    shared: Option<(Tiered, Tiered)>,
}

fn build_host(c: &Case, f: &Fmt, r: &mut Rng) -> Result<Built> {
    let (e, hr, h, i, u) = (c.cfg.experts, c.cfg.router_hidden, c.cfg.hidden, c.cfg.inter, c.cfg.up_rows());
    let router = floats(&r.bell(e * hr, 0.09));
    let bias: Option<Vec<f32>> = c.bias.then(|| (0..e).map(|_| r.f(-0.5, 0.5)).collect());
    let escale: Option<Vec<f32>> = c.escale.then(|| (0..e).map(|_| r.f(0.5, 1.5)).collect());
    let (up_w, down_w) = (r.bell(e * u * h, 0.05), r.bell(e * h * i, 0.05));
    let (up, up_deq) = pack_host(f, &up_w, e, u, h)?;
    let (down, down_deq) = if c.rows {
        let (j, d) = pack_host_rows(f, &down_w, e, h, i)?;
        (Dn::Rows(j), d)
    } else {
        let (j, d) = pack_host(f, &down_w, e, h, i)?;
        (Dn::Block(j), d)
    };
    let (mut shared, mut s_up, mut s_down, mut s_gate) = (None, Vec::new(), Vec::new(), None);
    if let Some((si, gate)) = c.shared {
        let rows = if c.cfg.act.gated() { 2 * si } else { si };
        let (up_p, ud) = pack_one(f, &r.bell(rows * hr, 0.05), rows, hr)?;
        let (down_p, dd) = pack_one(f, &r.bell(h * si, 0.05), h, si)?;
        shared = Some((up_p, down_p));
        (s_up, s_down, s_gate) = (ud, dd, gate.then(|| floats(&r.bell(hr, 0.09))));
    }
    Ok(Built { host: Host { router, bias, escale, up: up_deq, down: down_deq, s_up, s_down, s_gate }, up, down, shared })
}

/// The experts' stacked matrix [experts rows, cols] of the weights `w` (bf16 values, as floats) packed losslessly as one (`ExpertStack::lossless`).
fn lossless_stack(w: &[f32], experts: usize, cols: usize) -> Result<ExpertStack> {
    let words: Vec<u16> = w.iter().map(|x| (x.to_bits() >> 16) as u16).collect();
    let p = glyd_gpu::pack::pack_tiered_checked(&words, w.len() / cols, cols).ok_or_else(|| Error("the experts did not pack in the tiered layout".into()))?;
    ExpertStack::lossless(crate::weights::upload_tiered(&p)?, experts)
}

/// The layer on the GPU from `b`'s packs; `lossless`: its experts' weights (what the block layout decodes to) packed in the tiered layout instead, a stack each.
fn upload(c: &Case, b: Built, lossless: bool) -> Result<(MoeLayer, Host)> {
    let h = &b.host;
    let (e, hid) = (c.cfg.experts, c.cfg.hidden);
    let w = MoeWeights {
        router: up_bf(&h.router)?,
        bias: h.bias.as_ref().map(|v| up_f32(v)).transpose()?,
        escale: h.escale.as_ref().map(|v| up_f32(v)).transpose()?,
        up: if lossless { lossless_stack(&h.up, e, hid)? } else { ExpertStack::upload(&b.up)? },
        down: match &b.down {
            Dn::Block(_) if lossless => Down::Block(lossless_stack(&h.down, e, c.cfg.inter)?),
            Dn::Block(j) => Down::Block(ExpertStack::upload(j)?),
            Dn::Rows(j) => Down::Rows(RowStack::upload(j)?),
        },
    };
    let shared = match (c.shared, &b.shared) {
        (Some((si, _)), Some((u, d))) => {
            Some(Shared { inter: si, up: upload_tiered(u)?, down: upload_tiered(d)?, gate: h.s_gate.as_ref().map(|g| up_bf(g)).transpose()? })
        }
        _ => None,
    };
    Ok((MoeLayer::new(c.cfg, w, shared)?, b.host))
}

struct Tally {
    failed: Vec<String>,
}

/// GPU ids against the CPU's from the same logits: (tokens whose experts differ by more than a tie, tokens whose experts differ by a tie only, the largest difference of the weights relative to the largest, over the tokens
/// whose experts are the same). A tie: the chosen scores, largest first, equal to 1e-6 (the CPU's exp and the GPU's differ in the last place, and two scores that close may fall either way).
fn compare_router(c: &MoeCfg, h: &Host, logits: &[f32], ids: &[i64], wts: &[f32], t: usize) -> (usize, usize, f64) {
    let (e, k) = (c.experts, c.router.top_k);
    let (mut bad, mut ties, mut worst) = (0, 0, 0f64);
    for tok in 0..t {
        let row = &logits[tok * e..(tok + 1) * e];
        let (cid, cw) = route(&c.router, row, h.bias.as_deref(), h.escale.as_deref());
        let gid = &ids[tok * k..(tok + 1) * k];
        if gid.iter().zip(&cid).all(|(g, x)| *g == *x as i64) {
            worst = worst.max(rel(&wts[tok * k..(tok + 1) * k], &cw));
            continue;
        }
        let s = scores(&c.router, row);
        let rank = |e: usize| if c.router.scoring != Scoring::Softmax { s[e] + h.bias.as_ref().map_or(0.0, |b| b[e]) } else { s[e] };
        let ranks = |ids: &mut dyn Iterator<Item = i64>| -> Option<Vec<f32>> {
            let mut v: Option<Vec<f32>> = ids.map(|x| if x >= 0 && (x as usize) < s.len() { Some(rank(x as usize)) } else { None }).collect();
            if let Some(v) = v.as_mut() {
                v.sort_by(|a, b| (b + 0.0).total_cmp(&(a + 0.0)));
            }
            v
        };
        let (a, b) = (ranks(&mut gid.iter().copied()), ranks(&mut cid.iter().map(|&x| x as i64)));
        match (a, b) {
            (Some(a), Some(b)) if a.iter().zip(&b).all(|(x, y)| (x - y).abs() <= 1e-6 * y.abs().max(1e-12)) => ties += 1,
            _ => bad += 1,
        }
    }
    (bad, ties, worst)
}

/// What the GPU left of one count of tokens: the router's logits, experts and weights, the plan, the experts' intermediate rows (in the plan's order), the routed sum, the sum with the shared expert.
struct Stages {
    logits: Vec<f32>,
    ids: Vec<i64>,
    wts: Vec<f32>,
    plan: Vec<i32>,
    rows: Vec<f32>,
    routed: Vec<f32>,
    fin: Vec<f32>,
}

/// The stages as a correct GPU would leave them, made on the CPU from the reference: the device's layouts of the plan (the hits, where each one's pairs start, the pairs by expert; what lies past the hits is not defined)
/// and of the rows (the pairs in the plan's order).
fn cpu_stages(c: &Case, h: &Host, t: usize, xr: &[f32], xe: &[f32]) -> Stages {
    let cfg = &c.cfg;
    let (e, k, hr, hid, inter) = (cfg.experts, cfg.router.top_k, cfg.router_hidden, cfg.hidden, cfg.inter);
    let refl = RefLayer { cfg, router: &h.router, bias: h.bias.as_deref(), escale: h.escale.as_deref(), up: &h.up, down: &h.down };
    let (mut logits, mut ids, mut wts) = (Vec::new(), Vec::new(), Vec::new());
    for i in 0..t {
        let l = refl.logits(&xr[i * hr..(i + 1) * hr]);
        let (id, w) = route(&cfg.router, &l, h.bias.as_deref(), h.escale.as_deref());
        logits.extend(l);
        ids.extend(id.iter().map(|&x| x as i64));
        wts.extend(w);
    }
    let p = plan_of(&ids, e);
    let mut plan = vec![-7i32; 2 + 2 * e + t * k];
    plan[0] = p.hits.len() as i32;
    plan[1..1 + p.hits.len()].copy_from_slice(&p.hits);
    plan[1 + e..2 + e + p.hits.len()].copy_from_slice(&p.start);
    plan[2 + 2 * e..2 + 2 * e + p.order.len()].copy_from_slice(&p.order);
    let mut rows = vec![0f32; t * k * inter];
    for (i, &j) in p.order.iter().enumerate() {
        let (tok, ex) = (j as usize / k, ids[j as usize] as usize);
        rows[i * inter..(i + 1) * inter].copy_from_slice(&refl.hidden_row(ex, &xe[tok * hid..(tok + 1) * hid]));
    }
    let routed = refl.layer(xe, t, &ids, &wts);
    let mut fin = routed.clone();
    if let Some((si, _)) = c.shared {
        let rs = RefShared { inter: si, up: &h.s_up, down: &h.s_down, gate: h.s_gate.as_deref() };
        for i in 0..t {
            rs.add(&cfg.act, &xr[i * hr..(i + 1) * hr], &mut fin[i * hid..(i + 1) * hid]);
        }
    }
    Stages { logits, ids, wts, plan, rows, routed, fin }
}

/// `--cpu-dump DIR`: the dump of each case at its `dump_t`, with the CPU reference standing in for the GPU (no GPU is used): what tools/moe_ref.py is tried on where there is none, and its first step on the pod.
fn cpu_dump(dir: &Path, want: &[&str]) -> Result<()> {
    for c in cases().iter().filter(|c| want.is_empty() || want.contains(&c.name)) {
        for f in &FMTS {
            let mut r = seed(c, f);
            let b = build_host(c, f, &mut r)?;
            let (hr, hid, t) = (c.cfg.router_hidden, c.cfg.hidden, c.dump_t);
            let xr = r.uni(t * hr, -2.0, 2.0);
            let xe = if hr == hid { xr.clone() } else { r.uni(t * hid, -2.0, 2.0) };
            let st = cpu_stages(c, &b.host, t, &xr, &xe);
            dump_case(dir, c, f, &b.host, t, &xr, &xe, &st.ids, &st.wts, &st.fin)?;
            println!("{} {}: {t} tokens dumped from the CPU reference", c.name, f.name);
        }
    }
    Ok(())
}

/// The layer on the GPU for `t` tokens (xr [t, router_hidden], xe [t, hidden]), each stage read back.
fn gpu_stages(dev: &Dev, lib: &Lib, layer: &MoeLayer, sc: &MoeScratch, t: usize, xr: &[f32], xe: &[f32]) -> Result<Stages> {
    let cfg = &layer.cfg;
    let (e, pairs, hr, hid) = (cfg.experts, t * cfg.router.top_k, cfg.router_hidden, cfg.hidden);
    let (xr_b, xe_b, y) = (up_bf(xr)?, up_bf(xe)?, Buf::zeroed(t * hid * 2)?);
    let s = dev.stream;
    // SAFETY (the calls below): buffers made here and by the layer, of the sizes its kernels and the library's products read and write; read after a synchronize.
    unsafe {
        layer.route(sc, s, xr_b.ptr, hr, t)?;
        layer.experts(lib, sc, s, xe_b.ptr, t, y.ptr)?;
    }
    dev.sync()?;
    let rows_buf = layer.rows(sc);
    let mut st = Stages {
        logits: rd(&sc.logits, t * e)?,
        ids: rd(&sc.ids, pairs)?,
        wts: rd(&sc.wts, pairs)?,
        plan: rd(&sc.plan, 2 + 2 * e + pairs)?,
        rows: floats(&rd::<u16>(rows_buf, pairs * cfg.inter)?),
        routed: floats(&rd::<u16>(&y, t * hid)?),
        fin: Vec::new(),
    };
    unsafe { layer.add_shared(lib, sc, s, xr_b.ptr, t, y.ptr)? };
    dev.sync()?;
    st.fin = floats(&rd::<u16>(&y, t * hid)?);
    Ok(st)
}

/// The stages against the CPU's, from the GPU's own experts and weights where a stage takes them: whether every one is within its bound, and the line that says how far each is.
#[allow(clippy::too_many_arguments)]
fn judge(c: &Case, cfg: &MoeCfg, h: &Host, name: &str, t: usize, xr: &[f32], xe: &[f32], st: &Stages) -> (bool, String) {
    let (e, k, hr, hid, inter, pairs) = (cfg.experts, cfg.router.top_k, cfg.router_hidden, cfg.hidden, cfg.inter, t * cfg.router.top_k);
    // (stages of the sizes they have, and experts that are experts or none: else nothing below can be read against the CPU's)
    if st.ids.len() != pairs || st.plan.len() != 2 + 2 * e + pairs || st.ids.iter().any(|&x| x < -1 || x >= e as i64) {
        return (
            false,
            format!(
                "FAIL {name}: the experts the router chose ({} of {pairs} values, ids from {} to {}) are not experts of {e}",
                st.ids.len(),
                st.ids.iter().min().unwrap_or(&0),
                st.ids.iter().max().unwrap_or(&0)
            ),
        );
    }
    let refl = RefLayer { cfg, router: &h.router, bias: h.bias.as_deref(), escale: h.escale.as_deref(), up: &h.up, down: &h.down };
    // the router
    let want_logits: Vec<f32> = (0..t).flat_map(|i| refl.logits(&xr[i * hr..(i + 1) * hr])).collect();
    let r_logits = rel(&st.logits, &want_logits);
    let (bad_ids, ties, r_wts) = compare_router(cfg, h, &st.logits, &st.ids, &st.wts, t);
    // the pairs sorted by expert
    let plan = read_plan(&st.plan, e, pairs);
    let plan_ok = plan == plan_of(&st.ids, e);
    // the experts' intermediate rows in the plan's order (all of them to 64 tokens, else every 17th)
    let (mut gr, mut wr) = (Vec::new(), Vec::new());
    for (i, &j) in plan.order.iter().enumerate().filter(|(i, j)| (t <= 64 || i % 17 == 0) && **j >= 0 && (**j as usize) < pairs && st.ids[**j as usize] >= 0) {
        let (tok, ex) = (j as usize / k, st.ids[j as usize] as usize);
        wr.extend(refl.hidden_row(ex, &xe[tok * hid..(tok + 1) * hid]));
        gr.extend_from_slice(&st.rows[i * inter..(i + 1) * inter]);
    }
    let r_rows = rel(&gr, &wr);
    // the routed sum, and with the shared expert
    let want_routed = refl.layer(xe, t, &st.ids, &st.wts);
    let r_routed = rel(&st.routed, &want_routed);
    let (mut want, mut shown) = (want_routed.clone(), String::new());
    if let Some((si, _)) = c.shared {
        let rs = RefShared { inter: si, up: &h.s_up, down: &h.s_down, gate: h.s_gate.as_deref() };
        for i in 0..t {
            rs.add(&cfg.act, &xr[i * hr..(i + 1) * hr], &mut want[i * hid..(i + 1) * hid]);
        }
    }
    let r_final = rel(&st.fin, &want);
    if let Some((si, gate)) = c.shared {
        shown = format!(", with the shared expert ({si}{}) {r_final:.1e}", if gate { ", gated" } else { "" });
    }
    let (b_logits, b_wts, b_rows, b_sum) = (2e-5, 1e-5, 1.5e-2, 1e-2);
    let ok = r_logits <= b_logits
        && bad_ids == 0
        && r_wts <= b_wts
        && plan_ok
        && r_rows <= b_rows
        && r_routed <= b_sum
        && r_final <= b_sum
        && st.fin.iter().all(|x| x.is_finite());
    let line = format!(
        "{} {name}: logits {r_logits:.1e}, experts {}/{} tokens equal{}, weights {r_wts:.1e}, plan {}, intermediate rows {r_rows:.1e}, routed sum {r_routed:.1e}{shown} (bounds {b_logits:.0e}, -, {b_wts:.0e}, exact, {b_rows:.1e}, {b_sum:.0e}, {b_sum:.0e})",
        if ok { "ok  " } else { "FAIL" },
        t - bad_ids - ties,
        t,
        if ties > 0 { format!(" (+{ties} by a tie the other way)") } else { String::new() },
        if plan_ok { "exact" } else { "DIFFERENT" }
    );
    (ok, line)
}

/// One count of tokens through the layer: every stage against the CPU, one line.
#[allow(clippy::too_many_arguments)]
fn check_t(
    dev: &Dev,
    lib: &Lib,
    layer: &MoeLayer,
    sc: &MoeScratch,
    h: &Host,
    c: &Case,
    f: &Fmt,
    t: usize,
    r: &mut Rng,
    tally: &mut Tally,
    dump: Option<&Path>,
) -> Result<()> {
    let (hr, hid) = (layer.cfg.router_hidden, layer.cfg.hidden);
    let xr = r.uni(t * hr, -2.0, 2.0);
    let xe = if hr == hid { xr.clone() } else { r.uni(t * hid, -2.0, 2.0) };
    let st = gpu_stages(dev, lib, layer, sc, t, &xr, &xe)?;
    let name = format!("{} {} T={t}", c.name, f.name);
    let (ok, line) = judge(c, &layer.cfg, h, &name, t, &xr, &xe, &st);
    if !ok {
        tally.failed.push(name);
    }
    println!("{line}");
    if let Some(d) = dump.filter(|_| t == c.dump_t) {
        dump_case(d, c, f, h, t, &xr, &xe, &st.ids, &st.wts, &st.fin)?;
    }
    Ok(())
}

/// A single expert's first product by its own pack at an offset in the stack (the first, one in the middle and the last expert) against the CPU's sums rounded to bf16: the strides of `ExpertStack::expert` are right,
/// and the library's dense products take an expert's bytes as they are.
#[allow(clippy::too_many_arguments)]
fn view_check(dev: &Dev, lib: &Lib, layer: &MoeLayer, sc: &MoeScratch, h: &Host, c: &Case, f: &Fmt, r: &mut Rng, tally: &mut Tally) -> Result<()> {
    let (e, hid, u) = (layer.cfg.experts, layer.cfg.hidden, layer.cfg.up_rows());
    let refl = RefLayer { cfg: &layer.cfg, router: &h.router, bias: h.bias.as_deref(), escale: h.escale.as_deref(), up: &h.up, down: &h.down };
    let (mut worst, mut shown) = (0f64, Vec::new());
    for ex in [0, e / 2 + 1, e - 1] {
        let x = r.uni(hid, -2.0, 2.0);
        let (xb, y) = (up_bf(&x)?, Buf::zeroed(u * 2)?);
        // SAFETY: buffers made here and by the layer, of the sizes the call reads and writes; read after a synchronize.
        unsafe { layer.expert_up(lib, sc, dev.stream, ex, xb.ptr, 1, y.ptr)? };
        dev.sync()?;
        let want: Vec<f32> = refl.up_sums(ex, &x).iter().map(|&v| bf16(v as f32)).collect();
        worst = worst.max(rel(&floats(&rd::<u16>(&y, u)?), &want));
        shown.push(ex);
    }
    let name = format!("{} {}: single experts {shown:?} by their own packs at offsets", c.name, f.name);
    let ok = worst <= 1e-2;
    if !ok {
        tally.failed.push(name.clone());
    }
    println!("{} {name}: the first product {worst:.1e} from the CPU's (bound 1e-2)", if ok { "ok  " } else { "FAIL" });
    Ok(())
}

/// The layer's decode step (one token) as a CUDA graph, replayed twice, against the eager one: the same bits (nothing in it reads a result on the host).
#[allow(clippy::too_many_arguments)]
fn graph_check(dev: &Dev, lib: &Lib, layer: &MoeLayer, sc: &MoeScratch, c: &Case, f: &Fmt, r: &mut Rng, tally: &mut Tally) -> Result<()> {
    let (hr, hid) = (layer.cfg.router_hidden, layer.cfg.hidden);
    let xr = r.uni(hr, -2.0, 2.0);
    let xe = if hr == hid { xr.clone() } else { r.uni(hid, -2.0, 2.0) };
    let (xr_b, xe_b, y) = (up_bf(&xr)?, up_bf(&xe)?, Buf::zeroed(hid * 2)?);
    let s = dev.stream;
    let step = || -> Result<()> {
        // SAFETY: the layer's buffers and these, of the sizes its calls read and write.
        unsafe {
            layer.route(sc, s, xr_b.ptr, hr, 1)?;
            layer.experts(lib, sc, s, xe_b.ptr, 1, y.ptr)?;
            layer.add_shared(lib, sc, s, xr_b.ptr, 1, y.ptr)
        }
    };
    step()?;
    dev.sync()?;
    let eager: Vec<u16> = rd(&y, hid)?;
    let g = dev.capture(step)?;
    let mut same = true;
    for _ in 0..2 {
        y.zero()?;
        g.launch(s)?;
        dev.sync()?;
        same &= rd::<u16>(&y, hid)? == eager;
    }
    let name = format!("{} {}: the decode step as a CUDA graph", c.name, f.name);
    if !same {
        tally.failed.push(name.clone());
    }
    println!("{} {name}, replayed twice: {}", if same { "ok  " } else { "FAIL" }, if same { "the eager step's bits" } else { "NOT the eager step's bits" });
    Ok(())
}

fn write(dir: &Path, name: &str, bytes: &[u8]) -> Result<()> {
    std::fs::write(dir.join(name), bytes).map_err(|e| Error(format!("{}: {e}", dir.join(name).display())))
}

fn raw<T: Copy>(v: &[T]) -> &[u8] {
    // SAFETY: plain data (u16, i64, f32), no padding.
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

/// A case's inputs, the weights as decoded and the GPU's outputs, for tools/moe_ref.py: little-endian binary files and cfg.json.
#[allow(clippy::too_many_arguments)]
fn dump_case(root: &Path, c: &Case, f: &Fmt, h: &Host, t: usize, xr: &[f32], xe: &[f32], ids: &[i64], wts: &[f32], y: &[f32]) -> Result<()> {
    let dir = root.join(format!("{}-{}", c.name, f.name));
    std::fs::create_dir_all(&dir).map_err(|e| Error(format!("{}: {e}", dir.display())))?;
    let g = &c.cfg;
    let (act, limit) = match g.act {
        Act::SwiGlu { limit } => ("swiglu", limit),
        Act::GeGlu => ("geglu", None),
        Act::Relu2 => ("relu2", None),
        Act::Situ { .. } => ("situ", None),
    };
    let shared = c.shared.map_or("null".to_string(), |(i, gate)| format!("{{\"inter\":{i},\"gate\":{gate}}}"));
    let cfg = format!(
        "{{\"name\":\"{}\",\"layout\":\"{}\",\"tokens\":{t},\"experts\":{},\"top_k\":{},\"router_hidden\":{},\"hidden\":{},\"inter\":{},\"scoring\":\"{}\",\"renorm\":{},\"scale\":{},\"bf16_logits\":{},\"act\":\"{act}\",\"limit\":{},\"bias\":{},\"escale\":{},\"shared\":{shared}}}\n",
        c.name,
        f.name,
        g.experts,
        g.router.top_k,
        g.router_hidden,
        g.hidden,
        g.inter,
        if g.router.scoring == Scoring::Sigmoid { "sigmoid" } else { "softmax" },
        g.router.renorm,
        g.router.scale,
        g.router.bf16_logits,
        limit.map_or("null".to_string(), |l| l.to_string()),
        c.bias,
        c.escale
    );
    write(&dir, "cfg.json", cfg.as_bytes())?;
    write(&dir, "x_route.bf16", raw(&bits(xr)))?;
    write(&dir, "x_exp.bf16", raw(&bits(xe)))?;
    write(&dir, "router.bf16", raw(&bits(&h.router)))?;
    write(&dir, "up.bf16", raw(&bits(&h.up)))?;
    write(&dir, "down.bf16", raw(&bits(&h.down)))?;
    write(&dir, "ids.i64", raw(ids))?;
    write(&dir, "wts.f32", raw(wts))?;
    write(&dir, "y.bf16", raw(&bits(y)))?;
    for (name, v) in [("bias.f32", &h.bias), ("escale.f32", &h.escale)] {
        if let Some(v) = v {
            write(&dir, name, raw(v))?;
        }
    }
    if c.shared.is_some() {
        write(&dir, "s_up.bf16", raw(&bits(&h.s_up)))?;
        write(&dir, "s_down.bf16", raw(&bits(&h.s_down)))?;
    }
    if let Some(gate) = &h.s_gate {
        write(&dir, "s_gate.bf16", raw(&bits(gate)))?;
    }
    Ok(())
}

fn seed(c: &Case, f: &Fmt) -> Rng {
    Rng(0x2026_1010_0101_0001 ^ (c.cfg.experts as u64) << 20 ^ f.tag as u64)
}

fn run_case(dev: &Dev, lib: &Lib, c: &Case, f: &Fmt, tally: &mut Tally, dump: Option<&Path>) -> Result<()> {
    let mut r = seed(c, f);
    let t0 = std::time::Instant::now();
    let (layer, host) = upload(c, build_host(c, f, &mut r)?, f.lossless)?;
    let sc = MoeScratch::new(dev, lib, &[&layer], *c.tokens.iter().max().expect("a case has counts of tokens"))?;
    println!(
        "---- {} {}: {} experts, {} of them a token, router over {}, experts over {} (intermediate {}, {:?}), packed in {:.1} s: {:.1} MB of weights, {:.1} MB of scratch",
        c.name,
        f.name,
        c.cfg.experts,
        c.cfg.router.top_k,
        c.cfg.router_hidden,
        c.cfg.hidden,
        c.cfg.inter,
        c.cfg.act,
        t0.elapsed().as_secs_f64(),
        layer.weights_bytes() as f64 / 1e6,
        sc.bytes() as f64 / 1e6
    );
    for &t in c.tokens {
        check_t(dev, lib, &layer, &sc, &host, c, f, t, &mut r, tally, dump)?;
    }
    // (a lossless stack's experts are not at fixed strides: no expert's own pack is taken of it)
    if !f.lossless {
        view_check(dev, lib, &layer, &sc, &host, c, f, &mut r, tally)?;
    }
    graph_check(dev, lib, &layer, &sc, c, f, &mut r, tally)
}

// ---- speed ------------------------------------------------------------------------------------------------------------------------------

/// A model's layer at its real shapes: its name, its MoE layers, the config, whether the router has a bias and a per-expert scale, the shared expert (intermediate size, gate, width of its output).
struct Preset {
    name: &'static str,
    layers: usize,
    cfg: MoeCfg,
    bias: bool,
    escale: bool,
    shared: Option<(usize, bool)>,
}

fn presets() -> Vec<Preset> {
    vec![
        Preset {
            name: "qwen36",
            layers: 40,
            cfg: MoeCfg { experts: 256, router_hidden: 2048, hidden: 2048, inter: 512, router: Router::softmax(8), act: Act::SwiGlu { limit: None } },
            bias: false,
            escale: false,
            shared: Some((512, true)),
        },
        Preset {
            name: "gemma4",
            layers: 30,
            cfg: MoeCfg { experts: 128, router_hidden: 2816, hidden: 2816, inter: 704, router: Router::softmax(8), act: Act::GeGlu },
            bias: false,
            escale: true,
            shared: Some((2112, false)),
        },
        Preset {
            name: "dsv4",
            layers: 43,
            cfg: MoeCfg {
                experts: 256,
                router_hidden: 4096,
                hidden: 4096,
                inter: 2048,
                router: Router::sigmoid(6, 1.5),
                act: Act::SwiGlu { limit: Some(10.0) },
            },
            bias: true,
            escale: false,
            shared: Some((2048, false)),
        },
        Preset {
            name: "glm53",
            layers: 42,
            cfg: MoeCfg {
                experts: 288,
                router_hidden: 4096,
                hidden: 4096,
                inter: 2048,
                router: Router::sigmoid(8, 2.5),
                act: Act::SwiGlu { limit: Some(10.0) },
            },
            bias: true,
            escale: false,
            shared: Some((2048, false)),
        },
        Preset {
            name: "nemotron3",
            layers: 40,
            cfg: MoeCfg { experts: 512, router_hidden: 4096, hidden: 1024, inter: 2688, router: Router::sigmoid(22, 5.0), act: Act::Relu2 },
            bias: true,
            escale: false,
            shared: Some((5376, false)),
        },
    ]
}

/// E experts' matrices of arbitrary bytes in a block layout (codes 0xA5, scales a bf16 of 0.0114): the time of the products does not depend on the values.
fn arbitrary_stack(dev: &Dev, tag: u32, experts: usize, rows: usize, cols: usize) -> Result<ExpertStack> {
    let (d, sc) = expert_strides(tag, rows, cols)?;
    let (data, scales) = (Buf::new((d * experts as u64) as usize)?, Buf::new((sc * experts as u64) as usize)?);
    memset_async(data.ptr, 0xA5, data.bytes, dev.stream)?;
    memset_async(scales.ptr, 0x3C, scales.bytes, dev.stream)?;
    dev.sync()?;
    ExpertStack::from_buffers(tag, [tag << 24, 0, 0], experts, rows, cols, data, scales)
}

fn arbitrary_pack(tag: u32, rows: usize, cols: usize) -> Result<crate::weights::PackBuf> {
    let (d, sc) = expert_strides(tag, rows, cols)?;
    upload_tiered(&Tiered {
        rows,
        cols,
        data: vec![0xA5; d as usize],
        blocks: vec![0x3C; sc as usize],
        block_base: vec![0; rows / 64 * (cols / 16) + 1],
        tiers: [tag << 24, 0, 0],
    })
}

fn speed(dev: &Dev, lib: &Lib, want: &[&str]) -> Vec<String> {
    println!("---- speed: the layer at the four models' real shapes, packs of arbitrary bytes, the step as a CUDA graph of several layers on different tokens (so that the weights are read from memory, not found in the cache)");
    let mut r = Rng(0x2026_1010_0202_0002);
    let mut failed = Vec::new();
    for p in presets().into_iter().filter(|p| want.is_empty() || want.contains(&p.name)) {
        for f in FMTS.iter().chain([&FP4, &FP4R]) {
            // (an error of the GPU or the library ends that model and layout: it is said, and the others go on)
            if let Err(e) = speed_one(dev, lib, &p, f, &mut r) {
                println!("FAIL speed {} {}: {e}", p.name, f.name);
                failed.push(format!("speed {} {}: {e}", p.name, f.name));
            }
            if p.cfg.act == Act::Relu2 {
                if let Err(e) = speed_sparse(dev, lib, &p, f, &mut r) {
                    println!("FAIL speed {} {} sparse: {e}", p.name, f.name);
                    failed.push(format!("speed {} {} sparse: {e}", p.name, f.name));
                }
            }
        }
    }
    failed
}

/// One model's layer in one layout at the counts of tokens: random routers and packs of arbitrary bytes at the real shapes, the layers of a step as a CUDA graph.
fn speed_one(dev: &Dev, lib: &Lib, p: &Preset, f: &Fmt, r: &mut Rng) -> Result<()> {
    let counts = [1usize, 2, 4, 8, 16, 64, 512, 2048];
    let c = &p.cfg;
    let (e, hr, h, i, u) = (c.experts, c.router_hidden, c.hidden, c.inter, c.up_rows());
    let router = r.bell(e * hr, 0.09);
    let down = arbitrary_stack(dev, f.tag, e, h, i)?;
    let w = MoeWeights {
        router: {
            let b = Buf::new(router.len() * 2)?;
            b.write(0, &router)?;
            b
        },
        bias: p.bias.then(|| up_f32(&vec![0.1; e])).transpose()?,
        escale: p.escale.then(|| up_f32(&vec![1.0; e])).transpose()?,
        up: arbitrary_stack(dev, f.tag, e, u, h)?,
        down: Down::Block(down),
    };
    // (the shared expert's output is the model's width: the router's)
    let Down::Block(down) = &w.down else { unreachable!("a dense layer") };
    let (per_expert, out_w) = (w.up.data_stride + w.up.scale_stride + down.data_stride + down.scale_stride, hr);
    let shared = match p.shared {
        Some((si, gate)) => {
            let rows = if c.act.gated() { 2 * si } else { si };
            Some(Shared {
                inter: si,
                up: arbitrary_pack(f.tag, rows, hr)?,
                down: arbitrary_pack(f.tag, out_w, si)?,
                gate: gate.then(|| Buf::zeroed(hr * 2)).transpose()?,
            })
        }
        None => None,
    };
    let layer = MoeLayer::new(*c, w, shared)?;
    let sc = MoeScratch::new(dev, lib, &[&layer], *counts.last().expect("counts"))?;
    println!(
        "{} {} ({} layers; {} experts, {} a token; router over {}, experts over {}, intermediate {}, {:?}): weights {:.2} GB a layer, scratch {:.2} GB for all",
        p.name,
        f.name,
        p.layers,
        e,
        c.router.top_k,
        hr,
        h,
        i,
        c.act,
        layer.weights_bytes() as f64 / 1e9,
        sc.bytes() as f64 / 1e9
    );
    for &t in &counts {
        let reps = match t {
            1..=16 => 16,
            64 => 8,
            512 => 3,
            _ => 2,
        };
        let xr = up_bf(&r.uni(reps * t * hr, -2.0, 2.0))?;
        let xe = if hr == h { None } else { Some(up_bf(&r.uni(reps * t * h, -2.0, 2.0))?) };
        let (y, ys) = (Buf::zeroed(t * h * 2)?, Buf::zeroed(t * out_w * 2)?);
        let s = dev.stream;
        let step = || -> Result<()> {
            for q in 0..reps {
                let (xrp, xep) = (xr.ptr + (q * t * hr * 2) as u64, xe.as_ref().map_or(xr.ptr + (q * t * hr * 2) as u64, |b| b.ptr + (q * t * h * 2) as u64));
                // SAFETY: the layer's buffers and these, of the sizes its calls read and write.
                unsafe {
                    layer.route(&sc, s, xrp, hr, t)?;
                    layer.experts(lib, &sc, s, xep, t, y.ptr)?;
                    layer.add_shared(lib, &sc, s, xrp, t, if out_w == h { y.ptr } else { ys.ptr })?;
                }
            }
            Ok(())
        };
        step()?;
        dev.sync()?;
        let g = dev.capture(step)?;
        let tm = Timer::new()?;
        let mut ms: Vec<f32> = Vec::new();
        for _ in 0..5 {
            tm.start(s)?;
            g.launch(s)?;
            ms.push(tm.stop(s)?);
        }
        ms.sort_by(|a, b| a.partial_cmp(b).expect("a time"));
        let us = ms[2] as f64 * 1e3 / reps as f64;
        // the experts' weights a call reads: the distinct experts of t k random choices
        let distinct = e as f64 * (1.0 - (1.0 - 1.0 / e as f64).powi((t * c.router.top_k) as i32));
        let gbs = distinct * per_expert as f64 / (us * 1e3);
        println!(
            "  T={t:<5} {us:9.1} us a layer = {:8.0} tokens/s through it; {} layers: {:6.1} tokens/s; the experts' packs {distinct:.0} x {:.2} MB read at {gbs:.0} GB/s",
            t as f64 * 1e6 / us,
            p.layers,
            t as f64 * 1e6 / (us * p.layers as f64),
            per_expert as f64 / 1e6
        );
    }
    Ok(())
}

/// E experts' matrices of arbitrary bytes in a row layout of `bits`: the time of the sparse product depends on the activations alone.
fn arbitrary_rows(dev: &Dev, bits: u32, experts: usize, rows: usize, cols: usize) -> Result<RowStack> {
    let tag = [glyd_gpu::pack::INT4_ROW_TAG, glyd_gpu::pack::INT5_ROW_TAG, glyd_gpu::pack::INT6_ROW_TAG][bits as usize - 4];
    let (d, sc) = row_strides(tag, rows, cols)?;
    let (data, scales) = (Buf::new((d * experts as u64) as usize)?, Buf::new((sc * experts as u64) as usize)?);
    memset_async(data.ptr, 0xA5, data.bytes, dev.stream)?;
    memset_async(scales.ptr, 0x3C, scales.bytes, dev.stream)?;
    dev.sync()?;
    RowStack::from_buffers(tag, experts, rows, cols, data, scales)
}

/// Nemotron 3 Super's layer at the real shape with its second matrices dense (the library's block layout) and sparse (the row layout), on activations that are about 92% zeros: the first matrices are four random experts'
/// packs tiled to the layer's 512, of weights with a negative mean (0.0013 a weight, against inputs between 0.5 and 1.5 over 1,024 products: about 92% of the sums are under zero), the second matrices are bytes
/// of no meaning (the time of either product does not depend on them). The layer's time a call at the counts of tokens a step has, the layers of a step as a CUDA graph, and the share of the neurons that are not zero.
fn speed_sparse(dev: &Dev, lib: &Lib, p: &Preset, f: &Fmt, r: &mut Rng) -> Result<()> {
    let counts = [1usize, 2, 4, 8, 16, 64, 512, 2048];
    let c = &p.cfg;
    let (e, hr, h, i, u) = (c.experts, c.router_hidden, c.hidden, c.inter, c.up_rows());
    let distinct = 4usize;
    let up_w = bits(&floats(&r.bell(distinct * u * h, 0.05)).iter().map(|x| x - 0.0013).collect::<Vec<_>>());
    let (j, _) = pack_host(f, &up_w, distinct, u, h)?;
    let tiled =
        Joined { tag: j.tag, tiers: j.tiers, experts: e, rows: j.rows, cols: j.cols, data: j.data.repeat(e / distinct), scales: j.scales.repeat(e / distinct) };
    let router = r.bell(e * hr, 0.09);
    let make = |down: Down| -> Result<MoeLayer> {
        let w = MoeWeights {
            router: {
                let b = Buf::new(router.len() * 2)?;
                b.write(0, &router)?;
                b
            },
            bias: p.bias.then(|| up_f32(&vec![0.1; e])).transpose()?,
            escale: None,
            up: ExpertStack::upload(&tiled)?,
            down,
        };
        let (si, _) = p.shared.expect("the preset has a shared expert");
        MoeLayer::new(*c, w, Some(Shared { inter: si, up: arbitrary_pack(f.tag, si, hr)?, down: arbitrary_pack(f.tag, hr, si)?, gate: None }))
    };
    let (dense, sparse) = (make(Down::Block(arbitrary_stack(dev, f.tag, e, h, i)?))?, make(Down::Rows(arbitrary_rows(dev, f.bits, e, i, h)?))?);
    let sc = MoeScratch::new(dev, lib, &[&dense, &sparse], *counts.last().expect("counts"))?;
    println!(
        "{} {}: the second product dense (the library's block layout, {:.2} GB a layer) and sparse (the row layout, the same bytes), activations about 92% zeros; the layer's time a call, {} layers a step",
        p.name,
        f.name,
        dense.weights_bytes() as f64 / 1e9,
        p.layers
    );
    for &t in &counts {
        let reps = match t {
            1..=16 => 16,
            64 => 8,
            512 => 3,
            _ => 2,
        };
        let xr = up_bf(&r.uni(reps * t * hr, 0.5, 1.5))?;
        let xe = up_bf(&r.uni(reps * t * h, 0.5, 1.5))?;
        let (y, ys) = (Buf::zeroed(t * h * 2)?, Buf::zeroed(t * hr * 2)?);
        let s = dev.stream;
        let mut us = [0f64; 2];
        let mut active = f64::NAN;
        for (k, layer) in [&dense, &sparse].into_iter().enumerate() {
            let step = || -> Result<()> {
                for q in 0..reps {
                    let (xrp, xep) = (xr.ptr + (q * t * hr * 2) as u64, xe.ptr + (q * t * h * 2) as u64);
                    // SAFETY: the layer's buffers and these, of the sizes its calls read and write.
                    unsafe {
                        layer.route(&sc, s, xrp, hr, t)?;
                        layer.experts(lib, &sc, s, xep, t, y.ptr)?;
                        layer.add_shared(lib, &sc, s, xrp, t, ys.ptr)?;
                    }
                }
                Ok(())
            };
            step()?;
            dev.sync()?;
            if k == 1 && t <= 64 {
                // (the last call's rows: the activations after relu squared, in the plan's order)
                let pairs = t * c.router.top_k;
                let rows: Vec<u16> = rd(layer.rows(&sc), pairs * i)?;
                active = rows.iter().filter(|&&x| x & 0x7fff != 0).count() as f64 / rows.len() as f64;
            }
            let g = dev.capture(step)?;
            let tm = Timer::new()?;
            let mut ms: Vec<f32> = Vec::new();
            for _ in 0..5 {
                tm.start(s)?;
                g.launch(s)?;
                ms.push(tm.stop(s)?);
            }
            ms.sort_by(|a, b| a.partial_cmp(b).expect("a time"));
            us[k] = ms[2] as f64 * 1e3 / reps as f64;
        }
        println!(
            "  T={t:<5} dense {:8.1} us, sparse {:8.1} us a layer (x{:.2}); {} layers: {:6.1} against {:6.1} tokens/s through them{}",
            us[0],
            us[1],
            us[0] / us[1],
            p.layers,
            t as f64 * 1e6 / (us[0] * p.layers as f64),
            t as f64 * 1e6 / (us[1] * p.layers as f64),
            if active.is_nan() { String::new() } else { format!("; {:.1}% of the neurons not zero", active * 100.0) }
        );
    }
    Ok(())
}

pub fn run(a: &Args) -> Result<bool> {
    let want: Vec<&str> = a.pos.iter().skip(1).map(String::as_str).collect();
    let all: Vec<&'static str> = cases().iter().map(|c| c.name).chain(presets().iter().map(|p| p.name)).collect();
    if let Some(w) = want.iter().find(|w| !all.contains(w)) {
        return err(format!("{w}: no such case ({})", all.join(", ")));
    }
    if let Some(d) = a.flag("cpu-dump") {
        cpu_dump(Path::new(d), &want)?;
        return Ok(true);
    }
    let dev = Dev::new(0)?;
    let lib = Lib::find()?;
    println!("{} (sm_{}{}), {}: the mixture-of-experts layer on the GPU against the CPU", dev.name, dev.cc.0, dev.cc.1, lib.path);
    let dump = a.flag("dump").map(Path::new);
    let mut tally = Tally { failed: Vec::new() };
    if !a.has("no-check") {
        for c in cases().iter().filter(|c| want.is_empty() || want.contains(&c.name)) {
            // (each block layout, then the experts held lossless; not in the row layout; the FP4 layouts of an MX FP4 checkpoint for the situ case, Kimi-K3's)
            for f in FMTS.iter().chain((!c.rows).then_some(&LOSSLESS)).chain(if c.name.starts_with("situ") { vec![&FP4, &FP4R] } else { Vec::new() }) {
                // (an error of the GPU or the library ends that case and layout: it is said, and the others go on)
                if let Err(e) = run_case(&dev, &lib, c, f, &mut tally, dump) {
                    println!("FAIL {} {}: {e}", c.name, f.name);
                    tally.failed.push(format!("{} {}: {e}", c.name, f.name));
                }
            }
        }
        if tally.failed.is_empty() {
            println!("all of the mixture-of-experts layer's stages within their bounds");
        } else {
            println!("FAILED: {}", tally.failed.join("; "));
        }
    }
    if a.has("speed") {
        tally.failed.extend(speed(&dev, &lib, &want));
    }
    Ok(tally.failed.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A small layer's weights on the host (bf16 values), as `build` makes them but not packed.
    fn host(c: &Case, r: &mut Rng) -> Host {
        let (e, hr, h, i, u) = (c.cfg.experts, c.cfg.router_hidden, c.cfg.hidden, c.cfg.inter, c.cfg.up_rows());
        let (shared_rows, si) = c.shared.map_or((0, 0), |(si, _)| (if c.cfg.act.gated() { 2 * si } else { si }, si));
        Host {
            router: floats(&r.bell(e * hr, 0.3)),
            bias: c.bias.then(|| (0..e).map(|_| r.f(-0.5, 0.5)).collect()),
            escale: c.escale.then(|| (0..e).map(|_| r.f(0.5, 1.5)).collect()),
            up: floats(&r.bell(e * u * h, 0.3)),
            down: floats(&r.bell(e * h * i, 0.3)),
            s_up: floats(&r.bell(shared_rows * hr, 0.3)),
            s_down: floats(&r.bell(h * si, 0.3)),
            s_gate: c.shared.filter(|s| s.1).map(|_| floats(&r.bell(hr, 0.3))),
        }
    }

    fn small(act: Act, router: Router, bias: bool, escale: bool, shared: Option<(usize, bool)>) -> Case {
        Case {
            name: "small",
            cfg: MoeCfg { experts: 16, router_hidden: 32, hidden: 32, inter: 16, router, act },
            bias,
            escale,
            shared,
            tokens: &[1],
            dump_t: 1,
            rows: false,
        }
    }

    /// What the check says of a GPU that is right, and of one that is wrong in each stage (a stage corrupted by a tenth of the largest value, a pair of the plan swapped, an expert changed, a weight a thousandth off).
    #[test]
    fn the_judgement_passes_the_right_stages_and_fails_each_wrong_one() {
        let swiglu = Act::SwiGlu { limit: None };
        for c in [
            small(swiglu, Router::softmax(4), false, false, Some((16, true))),
            small(Act::GeGlu, Router::softmax(4), false, true, Some((16, false))),
            small(Act::Relu2, Router::sigmoid(4, 2.5), true, false, Some((16, false))),
            small(Act::SwiGlu { limit: Some(0.3) }, Router::sigmoid(3, 5.0), true, false, None),
        ] {
            let mut r = Rng(11);
            let h = host(&c, &mut r);
            let t = 6;
            let xr = r.uni(t * 32, -2.0, 2.0);
            let right = cpu_stages(&c, &h, t, &xr, &xr);
            let (ok, line) = judge(&c, &c.cfg, &h, "right", t, &xr, &xr, &right);
            assert!(ok, "{line}");
            let wrong = |what: &str, f: &dyn Fn(&mut Stages)| {
                let mut st = cpu_stages(&c, &h, t, &xr, &xr);
                f(&mut st);
                let (ok, line) = judge(&c, &c.cfg, &h, what, t, &xr, &xr, &st);
                assert!(!ok, "{what} was not found: {line}");
            };
            let top = |v: &[f32]| v.iter().fold(0f32, |m, x| m.max(x.abs()));
            wrong("a logit", &|s| s.logits[3] += 0.01 * top(&s.logits));
            wrong("a weight", &|s| s.wts[2] *= 1.001);
            wrong("an expert", &|s| s.ids[1] = (s.ids[1] + 1) % 16);
            wrong("the plan's order", &|s| s.plan.swap(2 + 32, 2 + 32 + 1));
            wrong("a row", &|s| {
                let m = top(&s.rows);
                s.rows[5] += 0.1 * m
            });
            wrong("the routed sum", &|s| {
                let m = top(&s.routed);
                s.routed[7] += 0.1 * m
            });
            wrong("the final sum", &|s| {
                let m = top(&s.fin);
                s.fin[9] -= 0.1 * m
            });
            wrong("a value that is not a number", &|s| s.fin[0] = f32::NAN);
        }
    }

    /// A stage the GPU leaves as nothing a number or nothing sane is a failure, not a pass or a panic: NaN in a stage, experts that are not experts, a plan of garbage.
    #[test]
    fn what_is_not_a_number_or_not_sane_fails() {
        assert_eq!(rel(&[1.0, f32::NAN], &[1.0, 1.0]), f64::INFINITY);
        assert_eq!(rel(&[1.0], &[1.0, 1.0]), f64::INFINITY);
        assert_eq!(rel(&[1.0, 2.0], &[1.0, 2.0]), 0.0);
        let c = small(Act::SwiGlu { limit: None }, Router::softmax(4), false, false, Some((16, true)));
        let mut r = Rng(21);
        let h = host(&c, &mut r);
        let (t, xr) = (3, r.uni(3 * 32, -2.0, 2.0));
        let judged = |f: &dyn Fn(&mut Stages)| {
            let mut st = cpu_stages(&c, &h, t, &xr, &xr);
            f(&mut st);
            judge(&c, &c.cfg, &h, "stage", t, &xr, &xr, &st)
        };
        assert!(judged(&|_| {}).0);
        for (what, f) in [
            ("a NaN in the logits", &(|s: &mut Stages| s.logits[1] = f32::NAN) as &dyn Fn(&mut Stages)),
            ("a NaN in the rows", &|s: &mut Stages| s.rows[7] = f32::NAN),
            ("a NaN in the routed sum", &|s: &mut Stages| s.routed[2] = f32::NAN),
            ("an expert past the last", &|s: &mut Stages| s.ids[4] = 9999),
            ("a plan of garbage", &|s: &mut Stages| s.plan.iter_mut().for_each(|x| *x = i32::MAX)),
            ("a plan of negative counts", &|s: &mut Stages| s.plan.iter_mut().for_each(|x| *x = -5)),
        ] {
            let (ok, line) = judged(f);
            assert!(!ok, "{what} passed: {line}");
        }
    }

    /// An expert the CPU breaks a tie the other way is allowed where the two scores are equal; one that differs by more is not.
    #[test]
    fn ties_are_allowed_and_differences_are_not() {
        let c = small(Act::SwiGlu { limit: None }, Router { top_k: 1, ..Router::softmax(1) }, false, false, None);
        let h = host(&c, &mut Rng(3));
        let mut logits = vec![0f32; 16];
        logits[2] = 1.0;
        logits[5] = 1.0; // (equal: the CPU picks 2)
        let w = [1.0f32];
        assert_eq!(compare_router(&c.cfg, &h, &logits, &[2], &w, 1), (0, 0, 0.0));
        assert_eq!(compare_router(&c.cfg, &h, &logits, &[5], &w, 1), (0, 1, 0.0));
        assert_eq!(compare_router(&c.cfg, &h, &logits, &[3], &w, 1).0, 1);
        assert_eq!(compare_router(&c.cfg, &h, &logits, &[-1], &w, 1).0, 1);
        logits[5] = 0.9; // (not equal any more, even in bf16: the CPU's pick is 2, and 5 is a real difference)
        assert_eq!(compare_router(&c.cfg, &h, &logits, &[5], &w, 1).0, 1);
    }

    /// The experts packed one at a time and joined are the pack of the experts stacked, and decode to what that pack decodes to.
    #[test]
    fn host_packing_is_the_stacked_pack() {
        let mut r = Rng(5);
        let (e, rows, cols) = (3usize, 64usize, 64usize);
        let w = r.bell(e * rows * cols, 0.05);
        for f in &FMTS {
            let (j, deq) = pack_host(f, &w, e, rows, cols).unwrap();
            let stacked = (f.pack)(&w, e * rows, cols).unwrap();
            assert_eq!((&j.data, &j.scales), (&stacked.data, &stacked.blocks), "{}", f.name);
            let mut out = vec![0u16; e * rows * cols];
            assert!((f.unpack)(&stacked, &mut out));
            assert_eq!(deq, floats(&out), "{}", f.name);
            assert_eq!(j.tag, f.tag);
        }
        assert!(pack_host(&FMTS[0], &w[..63 * 64], 1, 63, 64).is_err(), "63 rows");
    }

    /// The experts' second matrices packed in the row layout one at a time and joined are the row packs of their transposes end to end, and decode (turned back to the matrix's orientation) to what the row decode of
    /// the pack gives, at the three widths; a weight that is not finite does not pack.
    #[test]
    fn host_packing_in_rows_is_the_experts_row_packs() {
        let mut r = Rng(9);
        let (e, rows, cols) = (3usize, 64usize, 96usize);
        let w = r.bell(e * rows * cols, 0.05);
        for f in &FMTS {
            let (j, deq) = pack_host_rows(f, &w, e, rows, cols).unwrap();
            let packs: Vec<Tiered> =
                (0..e).map(|x| pack_rows(f.bits, &transpose(&w[x * rows * cols..(x + 1) * rows * cols], rows, cols), cols, rows).unwrap()).collect();
            assert_eq!(
                (&j.data, &j.scales),
                (&packs.iter().flat_map(|p| p.data.clone()).collect::<Vec<_>>(), &packs.iter().flat_map(|p| p.blocks.clone()).collect::<Vec<_>>()),
                "{}",
                f.name
            );
            assert_eq!((j.experts, j.rows, j.cols), (e, cols, rows), "{}: the pack's rows are the weight's columns", f.name);
            let mut want = Vec::new();
            for p in &packs {
                let mut t = vec![0u16; rows * cols];
                assert!(unpack_rows(p, &mut t));
                want.extend(floats(&transpose(&t, cols, rows)));
            }
            assert_eq!(deq, want, "{}", f.name);
        }
        let mut inf = w.clone();
        inf[5] = 0x7F80;
        assert!(pack_host_rows(&FMTS[0], &inf, e, rows, cols).is_err());
    }

    /// Every case and every model's preset is a layer the products take, with a shared expert whose matrices they take too, and the cases reach the counts they are meant to.
    #[test]
    fn the_cases_and_presets_are_shapes_the_products_take() {
        let fits = |rows: usize, cols: usize| FMTS.iter().all(|f| expert_strides(f.tag, rows, cols).is_ok());
        for c in cases() {
            c.cfg.check().unwrap();
            if c.rows {
                crate::moe_rows::rows_fit(c.cfg.experts, c.cfg.inter, c.cfg.hidden, &c.cfg).unwrap();
            }
            assert!(c.tokens.contains(&c.dump_t), "{}", c.name);
            assert!(c.tokens.contains(&1), "{} has no decode step", c.name);
            let (hr, h, i, u) = (c.cfg.router_hidden, c.cfg.hidden, c.cfg.inter, c.cfg.up_rows());
            assert!(fits(u, h) && fits(h, i), "{}", c.name);
            if let Some((si, _)) = c.shared {
                assert!(fits(if c.cfg.act.gated() { 2 * si } else { si }, hr) && fits(h, si), "{}'s shared expert", c.name);
            }
        }
        // the prompt kernel starts at 48 pairs an expert (64 on an A100): some case has more than that
        assert!(cases().iter().any(|c| c.tokens.iter().any(|&t| t * c.cfg.router.top_k >= 64 * c.cfg.experts)));
        for p in presets() {
            p.cfg.check().unwrap();
            let (hr, h, i, u) = (p.cfg.router_hidden, p.cfg.hidden, p.cfg.inter, p.cfg.up_rows());
            assert!(fits(u, h) && fits(h, i), "{}", p.name);
            if let Some((si, _)) = p.shared {
                assert!(fits(if p.cfg.act.gated() { 2 * si } else { si }, hr) && fits(hr, si), "{}'s shared expert", p.name);
            }
        }
        assert_eq!(par(7, &|i| i * i), vec![0, 1, 4, 9, 16, 25, 36]);
    }
}
