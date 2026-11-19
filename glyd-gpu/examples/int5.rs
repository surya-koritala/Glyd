//! The five-bit block layout on the GPU (gpu/glyd_gpu.cu's Int5) against the packer's own decode: matrices of a fixed seed packed by
//! `pack_int5`, decoded on the GPU by `unpack` and compared bit for bit with `unpack_int5`; then multiplied by `Library::linear` at 1 to
//! 512 tokens, each product compared with the same product of the decoded matrix packed in the tiered layout (lossless: the same weights):
//! bit for bit where both take the same kernel (17 tokens on); to 16 tokens, where the tiered layout's own small products add in another
//! order, both layouts' every output against the product in f64, within one rounding to bf16 (2^-8 of the value) and an f32 sum's error
//! (k 2^-24 of its terms' magnitudes).
//!
//!     cargo run --release -p glyd-gpu --example int5

use glyd_gpu::pack::{pack_int5, pack_tiered_checked, unpack_int5};
use glyd_gpu::{cuda, Error, Library, Matrix, Pack, Product, Stream, Tiered, NOT_SUPPORTED};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn f32_of(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

fn bf16_rne(x: f32) -> u16 {
    let b = x.to_bits();
    ((b + 0x7FFF + ((b >> 16) & 1)) >> 16) as u16
}

/// n bf16 values of a trained matrix's spread (a sum of uniforms, scaled about 0.02), from a seed.
fn weights(n: usize, seed: u64, scale: f32) -> Vec<u16> {
    let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let mut next = move || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        (x >> 40) as f32 / (1u64 << 24) as f32
    };
    (0..n).map(|_| bf16_rne(((next() + next() + next() + next()) - 2.0) * scale)).collect()
}

fn run() -> Result<()> {
    let lib = Library::find()?;
    let ctx = cuda::Context::new(0)?;
    println!("{} ({}), the library's C API {}", ctx.name()?, lib.gpu()?, lib.api_version());
    let mut failures = 0;
    for (o, k, seed) in [(256usize, 512usize, 1u64), (1024, 5120, 2), (4096, 4096, 3), (6144, 5120, 4)] {
        let w = weights(o * k, seed, 0.03);
        let p = pack_int5(&w, o, k).ok_or("pack_int5 refused finite weights")?;
        let mut want = vec![0u16; o * k];
        if !unpack_int5(&p, &mut want) {
            return Err("unpack_int5 refused its own pack".into());
        }
        let (d, b, bb) = (ctx.upload(&p.data)?, ctx.upload(&p.blocks)?, ctx.upload(&p.block_base)?);
        let m5 = Matrix { pack: Pack::Tiered(Tiered { data: d.ptr(), blocks: b.ptr(), block_base: bb.ptr(), tiers: p.tiers }), rows: o as i64, cols: k as i64 };
        let whole = ctx.alloc(o * k * 2)?;
        // SAFETY: the pack's buffers uploaded whole; out holds o x k bf16.
        unsafe { lib.unpack(&m5, 0, o as i64, whole.ptr(), 0, Stream::DEFAULT)? };
        ctx.synchronize()?;
        let mut got = vec![0u16; o * k];
        whole.read(&mut got)?;
        let differ = got.iter().zip(&want).filter(|(a, b)| a != b).count();
        println!("[{o}, {k}] unpack: {} of {} weights as unpack_int5 decodes them{}", o * k - differ, o * k, if differ == 0 { " (bit for bit)" } else { "  FAIL" });
        failures += (differ != 0) as usize;
        // the decoded matrix in the tiered layout (lossless), its products the reference
        let t = pack_tiered_checked(&want, o, k).ok_or("the decoded matrix did not pack tiered")?;
        let (td, tb, tbb) = (ctx.upload(&t.data)?, ctx.upload(&t.blocks)?, ctx.upload(&t.block_base)?);
        let mt = Matrix { pack: Pack::Tiered(Tiered { data: td.ptr(), blocks: tb.ptr(), block_base: tbb.ptr(), tiers: t.tiers }), rows: o as i64, cols: k as i64 };
        for m in [1usize, 2, 8, 16, 33, 64, 65, 200, 512] {
            let x = weights(m * k, 100 + m as u64, 1.0);
            let xd = ctx.upload(&x)?;
            let (y5, yt) = (ctx.alloc(m * o * 2)?, ctx.alloc(m * o * 2)?);
            let need = |w: &Matrix| match lib.linear_workspace(w, m as i64, None) {
                Err(Error::Cuda { status: NOT_SUPPORTED, .. }) => Ok(None),
                r => r.map(Some),
            };
            let (n5, nt) = (need(&m5)?, need(&mt)?);
            let (Some(n5), Some(nt)) = (n5, nt) else {
                println!("[{o}, {k}] {m:>3} tokens: no kernel takes it in {} (W decoded for a GEMM of the caller's)", if n5.is_none() && nt.is_none() { "either layout" } else { "one layout  FAIL" });
                failures += (n5.is_some() != nt.is_some()) as usize;
                continue;
            };
            let ws = ctx.alloc(n5.max(nt).max(16))?;
            let mut done = ctx.alloc((m.div_ceil(128) * (o / 64)).max(1024) * 4)?;
            let counters = done.ptr();
            let prod = |y: &cuda::Buffer, bytes: usize| Product { x: xd.ptr(), m: m as i64, bias: std::ptr::null(), y: y.ptr(), workspace: ws.ptr(), workspace_bytes: bytes, done: counters };
            done.zero()?;
            // SAFETY: x [m, k] and y [m, o] in device memory, the workspace of the query's bytes, the counters zeroed, the packs uploaded whole.
            unsafe { lib.linear(&m5, &prod(&y5, n5), None, Stream::DEFAULT)? };
            ctx.synchronize()?;
            done.zero()?;
            // SAFETY: as above.
            unsafe { lib.linear(&mt, &prod(&yt, nt), None, Stream::DEFAULT)? };
            ctx.synchronize()?;
            let (mut a, mut bt) = (vec![0u16; m * o], vec![0u16; m * o]);
            y5.read(&mut a)?;
            yt.read(&mut bt)?;
            // (bf16 places apart: the words' distance as signed magnitudes)
            let place = |v: u16| if v & 0x8000 != 0 { -((v & 0x7FFF) as i32) } else { v as i32 };
            let same = a.iter().zip(&bt).filter(|(p, q)| p == q).count();
            let far = a.iter().zip(&bt).map(|(&p, &q)| (place(p) - place(q)).abs()).max().unwrap_or(0);
            let mut line = format!("[{o}, {k}] {m:>3} tokens: {same} of {} outputs as the tiered layout's product of the same weights, the rest at most {far} bf16 places apart", m * o);
            let mut ok = m <= 16 || same == m * o;
            if m <= 16 {
                let (mut worst, mut bad5, mut badt) = (0f64, 0usize, 0usize);
                for t in 0..m {
                    for r in 0..o {
                        let (mut exact, mut mag) = (0f64, 0f64);
                        for c in 0..k {
                            let v = f32_of(x[t * k + c]) as f64 * f32_of(want[r * k + c]) as f64;
                            exact += v;
                            mag += v.abs();
                        }
                        let bound = exact.abs() * 2f64.powi(-8) + mag * k as f64 * 2f64.powi(-24);
                        let (d5, dt) = ((f32_of(a[t * o + r]) as f64 - exact).abs(), (f32_of(bt[t * o + r]) as f64 - exact).abs());
                        worst = worst.max(d5 / exact.abs().max(1e-6));
                        bad5 += (d5 > bound) as usize;
                        badt += (dt > bound) as usize;
                    }
                }
                line += &format!("; against f64, the largest relative difference {worst:.2e}, outside one rounding and the sum's error: {bad5} five-bit, {badt} tiered");
                ok &= bad5 == 0;
            }
            println!("{line}{}", if ok { "" } else { "  FAIL" });
            failures += (!ok) as usize;
        }
    }
    if failures > 0 {
        return Err(format!("{failures} checks failed").into());
    }
    println!("all checks passed");
    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("int5: {e}");
        std::process::exit(1);
    }
}
