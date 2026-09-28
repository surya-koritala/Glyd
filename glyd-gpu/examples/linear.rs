//! The library's products by route from Rust: a pack of a model saved by
//! glyd.save_pretrained (its tiered layout) multiplied on the GPU by
//! `Library::linear` for 1 to 2000 tokens, the route this GPU takes for each
//! count (as glyd.gpu's Linears take it), checked bit for bit against that
//! route's kernel called on its own, and one token against the product of the
//! matrix decoded, in f32.
//!
//!     cargo run --release -p glyd-gpu --example linear -- GLYD_DIR [PACK]

use glyd_gpu::safetensors::Checkpoint;
use glyd_gpu::{cuda, json, Library, Matrix, Pack, Product, Route, Stream, Tiered};
use std::path::Path;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn read(c: &Checkpoint, name: &str) -> Result<Vec<u8>> {
    let (f, t) = c.get(name).ok_or_else(|| format!("{name}: not in {}", c.dir.display()))?;
    Ok(f.read(t)?)
}

/// bf16 from f32, truncated.
fn bf16(x: f32) -> u16 {
    (x.to_bits() >> 16) as u16
}

fn f32_of(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

fn run(args: &[String]) -> Result<()> {
    let dir = Path::new(&args[1]);
    let lib = Library::find()?;
    let manifest = json::parse(&std::fs::read_to_string(dir.join("glyd.json"))?)?;
    let packs = manifest.get("packs").and_then(json::Value::as_object).ok_or("glyd.json: no packs")?;
    let name = args.get(2).cloned().unwrap_or_else(|| packs[0].0.clone());
    let e = manifest.get("packs").and_then(|p| p.get(&name)).ok_or_else(|| format!("{name}: no such pack"))?;
    let shape = e.get("shape").and_then(json::Value::as_u64s).ok_or("no shape")?;
    let t = e.get("tiers").and_then(json::Value::as_u64s).ok_or("no tiers")?;
    let (o, k) = (shape[0] as usize, shape[1] as usize);
    let saved = Checkpoint::open(dir)?;
    let (data, blocks, base) = (read(&saved, &format!("{name}.glyd_data"))?, read(&saved, &format!("{name}.glyd_blocks"))?, read(&saved, &format!("{name}.glyd_block_base"))?);
    if data.len() != o * k / 1024 * 1280 || base.len() != (o * k / 1024 + 1) * 4 {
        return Err(format!("{name}: its buffers are not its shape's").into());
    }

    let ctx = cuda::Context::new(0)?;
    let (d, b, bb) = (ctx.upload(&data)?, ctx.upload(&blocks)?, ctx.upload(&base)?);
    let w = Matrix { pack: Pack::Tiered(Tiered { data: d.ptr(), blocks: b.ptr(), block_base: bb.ptr(), tiers: [t[0] as u32, t[1] as u32, t[2] as u32] }), rows: o as i64, cols: k as i64 };
    let gpu = lib.gpu()?;
    println!("{name} [{o}, {k}] on {} ({gpu}), the library's C API {}", ctx.name()?, lib.api_version());

    // W decoded, for one token's product in f32
    let whole = ctx.alloc(o * k * 2)?;
    // SAFETY: the pack's buffers uploaded whole (their sizes checked), out holds o x k bf16.
    unsafe { lib.unpack(&w, 0, o as i64, whole.ptr(), 0, Stream::DEFAULT)? };
    ctx.synchronize()?;
    let mut wh = vec![0u16; o * k];
    whole.read(&mut wh)?;

    let mut seed = 0x9E37_79B9_7F4A_7C15u64;
    for &m in &[1usize, 7, 16, 17, 33, 64, 65, 128, 129, 600, 1100, 2000] {
        let x: Vec<u16> = (0..m * k)
            .map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                bf16(((seed >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 2.0)
            })
            .collect();
        let xg = ctx.upload(&x)?;
        let (route, _) = lib.route(gpu, &w, m as i64)?;
        let units = m.div_ceil(128) * (o / 64);
        let mut done = ctx.alloc(units * 4)?;
        done.zero()?;
        let (y1, y2) = (ctx.alloc(m * o * 2)?, ctx.alloc(m * o * 2)?);
        let need = lib.linear_workspace(&w, m as i64, None)?;
        let ws = ctx.alloc(need.max(1))?;
        let p = |y: &cuda::Buffer, bytes| Product { x: xg.ptr(), m: m as i64, bias: std::ptr::null(), y: y.ptr(), workspace: ws.ptr(), workspace_bytes: bytes, done: done.ptr() };
        // SAFETY: X [m, k] and Y [m, o] in device memory, the workspace of the query's bytes, the counters zeroed.
        unsafe { lib.linear(&w, &p(&y1, need), None, Stream::DEFAULT)? };
        // the route's kernel on its own (a prompt's DECODE and AHEAD by the prompt kernel, as linear takes them)
        let kernel = match route {
            Route::Gemm => lib.gemm_workspace(&w, m as i64)?,
            _ => lib.gemm_big_workspace(&w, m as i64, 0)?,
        };
        let ws2 = ctx.alloc(kernel.max(1))?;
        let p2 = Product { workspace: ws2.ptr(), workspace_bytes: kernel, ..p(&y2, kernel) };
        // SAFETY: as above.
        unsafe {
            match route {
                Route::Gemm => lib.gemm(&w, &p2, Stream::DEFAULT)?,
                _ => lib.gemm_big(&w, &p2, 0, Stream::DEFAULT)?,
            }
        }
        ctx.synchronize()?;
        let (mut a, mut c) = (vec![0u16; m * o], vec![0u16; m * o]);
        y1.read(&mut a)?;
        y2.read(&mut c)?;
        if a != c {
            return Err(format!("{m} tokens: linear ({route:?}) is not its kernel's, bit for bit").into());
        }
        let mut worst = 0f32;
        if m == 1 {
            let (mut err, mut top) = (0f32, 0f32);
            for r in 0..o {
                let s: f32 = (0..k).map(|j| f32_of(x[j]) * f32_of(wh[r * k + j])).sum();
                err = err.max((f32_of(a[r]) - s).abs());
                top = top.max(s.abs());
            }
            worst = err / top;
            if worst > 1e-2 {
                return Err(format!("one token: {worst} off the f32 product").into());
            }
        }
        println!("  {m:>4} tokens: {route:?}, linear bit for bit its kernel's{}", if m == 1 { format!(", {worst:.1e} off f32") } else { String::new() });
    }
    Ok(())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: {} GLYD_DIR [PACK]", args[0]);
        std::process::exit(2);
    }
    if let Err(e) = run(&args) {
        eprintln!("linear: {e}");
        std::process::exit(1);
    }
}
