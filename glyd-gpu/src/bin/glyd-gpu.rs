//! glyd-gpu: a model's weights packed for Glyd's GPU layouts on the CPU, and a saved one checked:
//! `python -m glyd.gpu pack` and `verify` without Python. The glyd CLI runs it for `glyd pack` and `glyd verify`.

use glyd_gpu::{cuda, save, Library};
use std::path::Path;
use std::time::Instant;

const USAGE: &str = "usage:
  glyd pack MODEL OUT [--no-merge] [--layout mma|mma12] [--threads N]
      MODEL (a directory, or a Hugging Face repo id in the local cache) packed on the CPU, each pack
      checked against its weights, and saved in OUT as glyd-v1 (glyd-v2: a mixture of experts), as
      python -m glyd.gpu pack saves it, byte for byte (Qwen3, Qwen2, Llama, Mistral, Granite,
      GraniteMoe; other families: python -m glyd.gpu pack)
      --no-merge      q, k, v and gate, up saved as packs of their own
      --layout mma12  the 12-bit layout (glyd-v3), which an A10, A100 or H100 loads without packing again
  glyd verify PATH [--device cpu|cuda:N] [--threads N]
      a saved model's every pack decoded (on the CPU, or on a GPU by the Glyd GPU library:
      $GLYD_GPU_LIB, else libglyd_gpu_cuda13.so or cuda12 on the loader's path) and each tensor's
      sha256 checked against glyd.json
  --threads N: at most N threads (default: $GLYD_THREADS, else every core)
  (glyd-gpu pack / glyd-gpu verify: the same)";

/// The options: `--threads N`, `--device D`, flags; the rest in order.
struct Args {
    free: Vec<String>,
    threads: usize,
    device: String,
    layout: save::Layout,
    no_merge: bool,
}

fn parse(args: &[String]) -> Result<Args, String> {
    let mut a = Args { free: Vec::new(), threads: 0, device: "cpu".into(), layout: save::Layout::Tiered, no_merge: false };
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--threads" => {
                a.threads = args.get(i + 1).and_then(|n| n.parse().ok()).filter(|&n| n > 0).ok_or("--threads takes a number")?;
                i += 1;
            }
            "--device" => {
                a.device = args.get(i + 1).cloned().ok_or("--device takes cpu or cuda:N")?;
                i += 1;
            }
            "--layout" => {
                a.layout = match args.get(i + 1).map(String::as_str) {
                    Some("mma") => save::Layout::Tiered,
                    Some("mma12") => save::Layout::Twelve,
                    _ => return Err("--layout takes mma or mma12".into()),
                };
                i += 1;
            }
            "--no-merge" => a.no_merge = true,
            s if s.starts_with("--") => return Err(format!("{s}: no such option\n{USAGE}")),
            s => a.free.push(s.to_string()),
        }
        i += 1;
    }
    if a.threads == 0 {
        a.threads = std::env::var("GLYD_THREADS").ok().and_then(|n| n.parse().ok()).filter(|&n| n > 0).unwrap_or_else(|| std::thread::available_parallelism().map_or(1, |n| n.get()));
    }
    Ok(a)
}

fn run(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let a = parse(&args[1..])?;
    match (args.first().map(String::as_str), a.free.as_slice()) {
        (Some("pack"), [model, out]) => {
            let t = Instant::now();
            let source = save::Source::find(model)?;
            let s = save::save(&source, Path::new(out), a.threads, save::SHARD_BYTES, !a.no_merge, a.layout)?;
            let secs = t.elapsed().as_secs_f64();
            println!("{model}: {} tensors packed and checked, saved in {out}", s.checked);
            eprintln!("{:.2} GB of bf16 in {secs:.1} s ({:.2} GB/s, {} threads), {} shard{}", s.bytes as f64 / 1e9, s.bytes as f64 / 1e9 / secs, a.threads, s.shards, if s.shards == 1 { "" } else { "s" });
            Ok(())
        }
        (Some("verify"), [path]) => {
            let n = if a.device == "cpu" {
                save::verify(Path::new(path), a.threads, &save::Decoder::Cpu)?
            } else {
                let ordinal = a.device.strip_prefix("cuda:").or(if a.device == "cuda" { Some("0") } else { None }).and_then(|n| n.parse().ok()).ok_or("--device takes cpu or cuda:N")?;
                let (lib, ctx) = (Library::find()?, cuda::Context::new(ordinal)?);
                save::verify(Path::new(path), a.threads, &save::Decoder::Gpu(&lib, &ctx))?
            };
            println!("{path}: {n} tensors decode to glyd.json's sha256");
            Ok(())
        }
        _ => Err(USAGE.into()),
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|a| a == "--help" || a == "-h") {
        println!("{USAGE}");
        return;
    }
    if let Err(e) = run(&args) {
        eprintln!("glyd {}: {e}", args.first().map_or("", String::as_str));
        std::process::exit(if e.to_string() == USAGE { 2 } else { 1 });
    }
}
