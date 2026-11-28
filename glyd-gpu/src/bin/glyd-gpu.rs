//! glyd-gpu: a model's weights packed for Glyd's GPU layouts on the CPU, and a saved one checked:
//! `python -m glyd.gpu pack` and `verify` without Python. The glyd CLI runs it for `glyd pack` and `glyd verify`.

use glyd_gpu::{cuda, save, Library};
use std::path::Path;
use std::time::Instant;

const USAGE: &str = "usage:
  glyd pack MODEL OUT [--no-merge] [--layout mma|mma12|a|i6|i5|i4|i3] [--plan PLAN.json [--codes DIR]] [--shard-bytes N] [--threads N]
      MODEL (a directory, or a Hugging Face repo id in the local cache) packed on the CPU, each pack
      checked against its weights, and saved in OUT as glyd-v4 (each matrix in the smaller of its
      two layouts), as python -m glyd.gpu pack saves it, byte for byte (Qwen3, Qwen2, Llama,
      Mistral, Granite, GraniteMoe; other families: python -m glyd.gpu pack)
      --no-merge      q, k, v and gate, up saved as packs of their own
      --layout mma    the tiered layout, lossless (the default; the level penguin, also --layout penguin)
      --layout mma12  the 12-bit layout (glyd-v3), which an A10, A100 or H100 loads without packing again
      --layout a      Candidate A, lossy (glyd-v6): every bf16 weight rounded to three fraction bits, each matrix in
                      the compact tiered layout (6.8 bits a weight or so)
      --layout i6     six-bit block codes, lossy (glyd-v7; the level kestrel, 6.53 bits a weight); kestrel is the same
      --layout i5     five-bit block codes, lossy (glyd-v7; the level linnet, 5.53 bits a weight); linnet is the same
      --layout i4     four-bit block codes, lossy (glyd-v7; the level wren, 4.53 bits a weight); wren is the same
      --layout i3     three-bit block codes, lossy (glyd-v7; no level, 3.53 bits a weight: too coarse for a model alone)
      --plan PLAN     what to save, in order, as huggingface/plan.py writes it for the model transformers builds
                      (any family it plans; --no-merge is the plan's; for --layout i6|i5|i4|i3 make it with plan.py --lossy,
                      which packs the embedding too; with those, a linear's \"layout\": i6, i5, i4 or i3 puts that
                      pack in another block layout than the save's, its entry in glyd.json naming it; i6r, i5r or i4r
                      put one weight's transpose in the row layout of those bits, each row one input's weights, for a
                      kernel that reads only the rows it needs (an experts' down matrix after relu squared); verify
                      decodes those on the CPU, as it does a Hummingbird directory's codebook rows, c3r and c2r
                      (layout hb, made by glyd-engine pack MODEL OUT --layout hummingbird --codes DIR))
      --codes DIR     with --plan and --layout i6|i5: the calibrated codes and scales in DIR (research/lossy-2026/calib.py
                      quantize: NAME.codes, NAME.scales for each matrix NAME of the source, and calibration.json) packed
                      as they are, bit for bit, for the linears that have them (the embedding and the output layer have
                      none: packed from their weights); glyd.json records the calibration
      --shard-bytes N a shard closed at N bytes (default 5000000000, save_pretrained's): the host holds about
                      three shards' worth at most
  glyd verify PATH [--device cpu|cuda:N] [--threads N]
      a saved model (glyd-v1 to glyd-v7) and its every pack decoded (on the CPU, or on a GPU by the Glyd GPU library:
      $GLYD_GPU_LIB, else libglyd_gpu_cuda13.so or cuda12 on the loader's path) and each packed
      tensor's sha256 checked against glyd.json; every tensor saved as it is by its own there (a save
      of glyd 0.25 on); each file's tensors back to back, and each a pack's or of a sha256 there
  --threads N: at most N threads (default: $GLYD_THREADS, else every core). Memory: pack holds two
      shards (about 5 GB each) and at most a shard's worth of weights in flight, whatever N; verify on
      the CPU a pack and its matrix a thread, and one more
  (glyd-gpu pack / glyd-gpu verify: the same)";

/// The options: `--threads N`, `--device D`, flags; the rest in order.
struct Args {
    free: Vec<String>,
    threads: usize,
    device: String,
    layout: save::Layout,
    no_merge: bool,
    plan: Option<String>,
    codes: Option<String>,
    shard_bytes: u64,
}

fn parse(args: &[String]) -> Result<Args, String> {
    let mut a = Args {
        free: Vec::new(),
        threads: 0,
        device: "cpu".into(),
        layout: save::Layout::Tiered,
        no_merge: false,
        plan: None,
        codes: None,
        shard_bytes: save::SHARD_BYTES,
    };
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
                    Some("a") => save::Layout::Compact,
                    Some("i6" | "kestrel") => save::Layout::Int6,
                    Some("i5" | "linnet") => save::Layout::Int5,
                    Some("i4" | "wren") => save::Layout::Int4,
                    Some("i3") => save::Layout::Int3,
                    Some("penguin") => save::Layout::Tiered,
                    _ => return Err("--layout takes mma, mma12, a, i6 (kestrel), i5 (linnet), i4 (wren), i3 or penguin".into()),
                };
                i += 1;
            }
            "--no-merge" => a.no_merge = true,
            "--shard-bytes" => {
                a.shard_bytes = args.get(i + 1).and_then(|n| n.parse().ok()).filter(|&n| n > 0).ok_or("--shard-bytes takes a number")?;
                i += 1;
            }
            "--plan" => {
                a.plan = Some(args.get(i + 1).cloned().ok_or("--plan takes a file")?);
                i += 1;
            }
            "--codes" => {
                a.codes = Some(args.get(i + 1).cloned().ok_or("--codes takes a directory")?);
                i += 1;
            }
            s if s.starts_with("--") => return Err(format!("{s}: no such option\n{USAGE}")),
            s => a.free.push(s.to_string()),
        }
        i += 1;
    }
    if a.threads == 0 {
        a.threads = std::env::var("GLYD_THREADS")
            .ok()
            .and_then(|n| n.parse().ok())
            .filter(|&n| n > 0)
            .unwrap_or_else(|| std::thread::available_parallelism().map_or(1, |n| n.get()));
    }
    Ok(a)
}

fn run(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let a = parse(&args[1..])?;
    match (args.first().map(String::as_str), a.free.as_slice()) {
        (Some("pack"), [model, out]) => {
            let t = Instant::now();
            let source = save::Source::find(model)?;
            let s = match (&a.plan, &a.codes) {
                (Some(p), None) => save::save_planned(
                    &source,
                    Path::new(out),
                    a.threads,
                    a.shard_bytes,
                    a.layout,
                    &std::fs::read_to_string(p).map_err(|e| format!("{p}: {e}"))?,
                )?,
                (Some(p), Some(c)) => save::save_calibrated(
                    &source,
                    Path::new(out),
                    a.threads,
                    a.shard_bytes,
                    a.layout,
                    &std::fs::read_to_string(p).map_err(|e| format!("{p}: {e}"))?,
                    Path::new(c),
                )?,
                (None, None) => save::save(&source, Path::new(out), a.threads, a.shard_bytes, !a.no_merge, a.layout)?,
                (None, Some(_)) => return Err("--codes goes with --plan".into()),
            };
            let secs = t.elapsed().as_secs_f64();
            let coded = if a.codes.is_some() { format!(", {} from calibrated codes, the rest from their weights", s.calibrated) } else { String::new() };
            println!("{model}: {} tensors packed and checked{coded}, saved in {out} ({} saved as they are, with their sha256)", s.checked, s.hashed);
            eprintln!(
                "{:.2} GB of bf16 in {secs:.1} s ({:.2} GB/s, {} threads), {} shard{}",
                s.bytes as f64 / 1e9,
                s.bytes as f64 / 1e9 / secs,
                a.threads,
                s.shards,
                if s.shards == 1 { "" } else { "s" }
            );
            Ok(())
        }
        (Some("verify"), [path]) => {
            let v = if a.device == "cpu" {
                save::verify(Path::new(path), a.threads, &save::Decoder::Cpu)?
            } else {
                let ordinal = a
                    .device
                    .strip_prefix("cuda:")
                    .or(if a.device == "cuda" { Some("0") } else { None })
                    .and_then(|n| n.parse().ok())
                    .ok_or("--device takes cpu or cuda:N")?;
                let (lib, ctx) = (Library::find()?, cuda::Context::new(ordinal)?);
                save::verify(Path::new(path), a.threads, &save::Decoder::Gpu(&lib, &ctx))?
            };
            for k in &v.skipped {
                eprintln!(
                    "glyd verify: warning: {path}/glyd.json: {k:?}, a key of a newer glyd's save than this one ({}): skipped, the rest checked",
                    env!("CARGO_PKG_VERSION")
                );
            }
            let rest = if v.unchecked > 0 {
                format!("{} saved as they are not checked (saved before glyd 0.25: no sha256 for them)", v.unchecked)
            } else {
                format!("{} saved as they are match theirs", v.hashed)
            };
            println!("{path}: {} tensors decode to glyd.json's sha256, {rest}", v.packed);
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
