//! A matrix of a model saved by glyd.save_pretrained (glyd-v1: safetensors
//! and glyd.json) decoded on the GPU by the library's C API through this
//! crate, and checked against the bf16 checkpoint it was packed from, bit for
//! bit: gpu/examples/unpack.c in Rust. No Python.
//!
//!     cargo run --release -p glyd-gpu --example unpack -- GLYD_DIR BF16_DIR [PACK]
//!
//! GLYD_DIR: the saved model; BF16_DIR: its source checkpoint's directory;
//! PACK: a pack's name in glyd.json (default: the first). A pack of merged
//! Linears (q, k, v; gate, up) is checked tensor by tensor. The library:
//! $GLYD_GPU_LIB, else libglyd_gpu_cuda13.so (or cuda12) on the loader's
//! path; the GPU: the first.

use glyd_gpu::safetensors::Checkpoint;
use glyd_gpu::{cuda, json, Library, Matrix, Pack, Stream, Tiered};
use std::path::Path;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn fail(what: &str, why: impl std::fmt::Display) -> Box<dyn std::error::Error> {
    format!("{what}: {why}").into()
}

/// Tensor `name` of the checkpoint, as `dtype`.
fn read(c: &Checkpoint, name: &str, dtype: &str) -> Result<(Vec<u8>, Vec<u64>)> {
    let (f, t) = c.get(name).ok_or_else(|| fail(name, format!("not in {}", c.dir.display())))?;
    if t.dtype != dtype {
        return Err(fail(name, format!("{}, where this example reads {dtype}", t.dtype)));
    }
    Ok((f.read(t)?, t.shape.clone()))
}

fn run(args: &[String]) -> Result<bool> {
    let (glyd, orig) = (Path::new(&args[1]), Path::new(&args[2]));
    let lib = Library::find()?;
    println!("libglyd_gpu: C API {}, CUDA runtime {}", lib.api_version(), lib.cuda_version());

    let path = glyd.join("glyd.json");
    let manifest = json::parse(&std::fs::read_to_string(&path)?).map_err(|e| fail(&path.display().to_string(), e))?;
    let packs = manifest.get("packs").and_then(json::Value::as_object).filter(|p| !p.is_empty()).ok_or_else(|| fail("glyd.json", "no packs"))?;
    let name = args.get(3).cloned().unwrap_or_else(|| packs[0].0.clone());
    let pack = manifest.get("packs").and_then(|p| p.get(&name)).ok_or_else(|| fail(&name, "no such pack in glyd.json"))?;
    if pack.get("experts").is_some() {
        return Err(fail(&name, "a mixture of experts' layer: this example takes a Linear's"));
    }
    let shape = pack.get("shape").and_then(json::Value::as_u64s).filter(|s| s.len() == 2).ok_or_else(|| fail(&name, "no shape in glyd.json"))?;
    let tiers = pack.get("tiers").and_then(json::Value::as_u64s).filter(|t| t.len() == 3 && t.iter().all(|&x| x <= u32::MAX as u64)).ok_or_else(|| fail(&name, "no tiers (three 32-bit words) in glyd.json"))?;
    // W [O, K] as the tiered layout holds it (O a multiple of 64, K of 16), up to 2^40 weights
    let (o, k) = (shape[0], shape[1]);
    if o < 64 || o % 64 != 0 || k < 16 || k % 16 != 0 || o > (1 << 40) / k {
        return Err(fail(&name, format!("[{o}, {k}] in glyd.json: not a tiered pack's shape")));
    }
    let (n, steps) = ((o * k) as usize, (o * k / 1024) as usize);

    // The pack's buffers, checked against its shape before any reaches the GPU: 1280 bytes a step; each step's
    // block within blocks from block_base on (the kernel reads 128 bytes before a block, 256 past the last).
    let saved = Checkpoint::open(glyd)?;
    let (data, _) = read(&saved, &format!("{name}.glyd_data"), "U8")?;
    let (blocks, _) = read(&saved, &format!("{name}.glyd_blocks"), "U8")?;
    let (bb, _) = read(&saved, &format!("{name}.glyd_block_base"), "I32")?;
    if data.len() != steps * 1280 || bb.len() != (steps + 1) * 4 {
        return Err(fail(&name, format!("data of {} bytes, block_base of {}: not its shape's", data.len(), bb.len())));
    }
    let base: Vec<i32> = bb.as_chunks::<4>().0.iter().map(|b| i32::from_le_bytes(*b)).collect();
    for (s, &b) in base.iter().enumerate() {
        if b < 128 || (s > 0 && b < base[s - 1]) || b as usize + 256 > blocks.len() {
            return Err(fail(&name, format!("block_base[{s}] = {b}: its blocks ({} bytes) do not hold it", blocks.len())));
        }
    }

    let ctx = cuda::Context::new(0)?;
    let (d, b, bbase) = (ctx.upload(&data)?, ctx.upload(&blocks)?, ctx.upload(&base)?);
    let out = ctx.alloc(n * 2)?;
    let tiers = [tiers[0] as u32, tiers[1] as u32, tiers[2] as u32];
    let w = Matrix { pack: Pack::Tiered(Tiered { data: d.ptr(), blocks: b.ptr(), block_base: bbase.ptr(), tiers }), rows: o as i64, cols: k as i64 };
    // SAFETY: the pack's arrays uploaded whole and checked above; out holds rows x cols bf16.
    unsafe { lib.unpack(&w, 0, o as i64, out.ptr(), 0, Stream::DEFAULT)? };
    ctx.synchronize()?;
    let mut got = vec![0u16; n];
    out.read(&mut got)?;
    println!("{name}: [{o}, {k}], {:.2} bits a weight packed, decoded on the GPU", (data.len() + blocks.len() + bb.len() + 12) as f64 * 8.0 / n as f64);

    // Its tensors (a merged pack's Linears, their rows in turn) against the checkpoint's.
    let source = Checkpoint::open(orig)?;
    let (mut row, mut differ) = (0u64, 0u64);
    for t in pack.get("tensors").and_then(json::Value::as_array).unwrap_or(&[]) {
        let tn = t.get("name").and_then(json::Value::as_str).ok_or_else(|| fail(&name, "a tensor without its name in glyd.json"))?;
        let ts = t.get("shape").and_then(json::Value::as_u64s).filter(|s| s.len() == 2).ok_or_else(|| fail(tn, "no shape in glyd.json"))?;
        if ts[0] < 1 || ts[0] > o - row || ts[1] != k {
            return Err(fail(tn, format!("{ts:?} in glyd.json: not the pack's rows from row {row}")));
        }
        let (bytes, shape) = read(&source, tn, "BF16")?;
        if shape != ts {
            return Err(fail(tn, format!("{shape:?} in the checkpoint, {ts:?} in glyd.json")));
        }
        let at = (row * k) as usize;
        let d = bytes.as_chunks::<2>().0.iter().zip(&got[at..]).filter(|(a, &b)| u16::from_le_bytes(**a) != b).count() as u64;
        println!("  {tn} [{}, {k}]: {}", ts[0], if d > 0 { "differs" } else { "the checkpoint's, bit for bit" });
        row += ts[0];
        differ += d;
    }
    if row != o {
        return Err(fail(&name, "its tensors' rows are not its own"));
    }
    if differ > 0 {
        println!("{differ} of {n} weights differ");
    }
    Ok(differ == 0)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: {} GLYD_DIR BF16_DIR [PACK]", args[0]);
        std::process::exit(2);
    }
    match run(&args) {
        Ok(same) => std::process::exit(if same { 0 } else { 1 }),
        Err(e) => {
            eprintln!("unpack: {e}");
            std::process::exit(1);
        }
    }
}
