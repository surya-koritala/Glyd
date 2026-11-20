//! glyd-engine: Glyd's own engine for Qwen3 on CUDA.
//!
//!   glyd-engine run MODEL[:LEVEL] [--prompt TEXT] [--no-think] [--context N] [--users N] [--port N] [--host H] [--temp T] [--layout auto|mma|mma12] [--expert-cache GIB] [--host-memory GIB]
//!                                                             what `glyd run` starts: the model made ready on the first run (downloaded, packed for this GPU,
//!                                                             its KV tables made), then a chat in the terminal and at one URL, or one answer with --prompt
//!                                                             LEVEL: penguin (lossless, the default), kestrel (about 6.5 bits per weight; --lossy is the same), swift (about 5.5), wren (about 4.5,
//!                                                             a development build's) or hummingbird (Nemotron 3 Super's experts at about 3)
//!                                                             --expert-cache GIB: Nemotron 3 Super with its experts in the computer's memory and a cache of them in GIB GiB of the GPU's (without it, the cache is what the GPU has left when it cannot hold the model)
//!                                                             --cpu-threads N: the threads that compute the experts that are not on the GPU (the computer's cores less two)
//!                                                             --host-memory GIB|auto: the computer's memory the experts may take (auto: what it has free, less 6 GiB for the system); the Hummingbird level keeps the rest of its
//!                                                             experts in the packed directory, on the SSD, and reads one when the router asks for it
//!   glyd-engine serve MODEL [--port N] [--host H] [--context N] [--users N] [--host-memory GIB]   the OpenAI-compatible API and the chat page
//!   glyd-engine fetch MODEL                                      the model's files downloaded into the Hugging Face cache (no GPU)
//!   glyd-engine prepare MODEL [--layout ...]                     downloaded and packed for this GPU, the KV tables left for the first run
//!   glyd-engine doctor | login | supports MODEL                  what this computer has for it; a Hugging Face token saved; whether the engine runs MODEL (exit 3: no)
//!   glyd-engine preflight [MODEL[:LEVEL]] [--costs]               whether the engine will certainly run MODEL on this computer, said before anything is downloaded (exit 3: no, and why on stderr); --costs: its cost profile
//!   glyd-engine costs MODEL[:LEVEL] [--predict [--rates FILE]]    the model's cost profile at the level (what each layer holds, what a token reads; docs/cost-profile.md), and predicted decode speed
//!   glyd-engine rates [--dir DIR]                                 this machine's memory tiers measured: the GPU's memory, the link to it, the computer's memory, the disk under DIR
//!   glyd-engine fit MODEL[:LEVEL] [--machine NAME|FILE] [--context N] [--users N]
//!                                                             the plan before anything is downloaded: level, placement, context, predicted tokens/s (src/plan.rs)
//!   glyd-engine rangeworker --model MODEL[:LEVEL] --layers A..B [--ctx N] [--chunk N] [--head-rows N] [--sessions N] [--device D]
//!                                                             one range of a Qwen3.5 model's layers on this GPU, its frames on the standard input and output: one model across
//!                                                             processes or machines (docs/range-worker.md, glyd-range/1)
//!   glyd-engine --version
//!
//! A development build (the default features) has the checks and the measurements too: src/devcmd.rs.

use glyd_engine::args::Args;
use glyd_engine::ffi::{err, Result};

#[cfg(feature = "dev")]
mod devcmd;

#[cfg(feature = "dev")]
const USAGE: &str = "usage: glyd-engine run MODEL[:LEVEL] [--prompt TEXT] [--host-memory GIB] | serve MODEL [--port N] [--host-memory GIB] | connect [TOOL] | doctor | login | supports MODEL | preflight [MODEL[:LEVEL]] | fetch MODEL | prepare MODEL | verify MODEL | costs MODEL[:LEVEL] [--predict] | rates [--dir DIR] | fit MODEL[:LEVEL] [--machine NAME|FILE] | rangeworker --model MODEL[:LEVEL] --layers A..B | --version\n  LEVEL: penguin (lossless, the default), kestrel (about 6.5 bits per weight; --lossy is the same), swift (about 5.5), wren (about 4.5) or hummingbird (Nemotron 3 Super's experts at about 3)";
#[cfg(not(feature = "dev"))] // (wren runs in a development build only)
const USAGE: &str = "usage: glyd-engine run MODEL[:LEVEL] [--prompt TEXT] [--host-memory GIB] | serve MODEL [--port N] [--host-memory GIB] | connect [TOOL] | doctor | login | supports MODEL | preflight [MODEL[:LEVEL]] | fetch MODEL | prepare MODEL | verify MODEL | costs MODEL[:LEVEL] [--predict] | rates [--dir DIR] | fit MODEL[:LEVEL] [--machine NAME|FILE] | rangeworker --model MODEL[:LEVEL] --layers A..B | --version\n  LEVEL: penguin (lossless, the default), kestrel (about 6.5 bits per weight; --lossy is the same), swift (about 5.5) or hummingbird (Nemotron 3 Super's experts at about 3)";

/// glibc's allocator told to keep freed buffers up to 32 MB (its largest such setting) in the heap and not hand freed memory back: a load's threads (one a GPU,
/// each with its packers) free buffers of a few MB to tens of MB by the thousand, and every unmapping stops every CPU the process runs on to flush its TLB (the
/// 8-GPU Kimi-K3 load spent about half its time there).
#[cfg(all(target_os = "linux", target_env = "gnu"))]
fn keep_buffers() {
    extern "C" {
        fn mallopt(param: i32, value: i32) -> i32;
    }
    const M_TRIM_THRESHOLD: i32 = -1;
    const M_MMAP_THRESHOLD: i32 = -3;
    // SAFETY: mallopt only sets the allocator's parameters; it is called before any thread starts.
    unsafe {
        mallopt(M_MMAP_THRESHOLD, 32 << 20);
        mallopt(M_TRIM_THRESHOLD, i32::MAX);
    }
}

#[cfg(not(all(target_os = "linux", target_env = "gnu")))]
fn keep_buffers() {}

fn main() {
    keep_buffers();
    if let Err(e) = run() {
        eprintln!("glyd-engine: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let mut a = Args::parse();
    // a model given as a GGUF file (or a Hub repo of GGUF files) is read as a Hugging Face directory (src/gguf.rs)
    if let Some(code) = glyd_engine::gguf::resolve_args(&mut a) {
        std::process::exit(code);
    }
    // a Hugging Face token given as a flag is the Hub client's before the environment's and the saved login's (glyd run and glyd serve move it to HF_TOKEN before they start this program)
    if let Some(t) = a.flag("hf-token").filter(|t| !t.is_empty()) {
        glyd_engine::hub::set_token(t);
        eprintln!("glyd-engine: the Hugging Face token is on this program's command line, which ps shows: next time set HF_TOKEN=TOKEN in the environment, or run glyd login.");
    }
    let cmd = a.pos.first().map(String::as_str);
    if a.has("version") || matches!(cmd, Some("version" | "-V" | "-v")) {
        println!("glyd-engine {}", env!("CARGO_PKG_VERSION"));
        return Ok(());
    }
    if a.has("help") || matches!(cmd, Some("help" | "-h")) {
        println!("{USAGE}");
        return Ok(());
    }
    match cmd {
        Some("run") => std::process::exit(glyd_engine::cli::run(&a)),
        Some("serve") => std::process::exit(glyd_engine::cli::serve(&a)),
        Some("doctor") => std::process::exit(glyd_engine::cli::doctor(&a)),
        // what to give a program (Open WebUI, Continue, Claude Code, an SDK) so that it talks to the server that runs here
        Some("connect") => std::process::exit(glyd_engine::connect::connect(&a)),
        Some("login") => std::process::exit(glyd_engine::cli::login(&a)),
        Some("supports") => std::process::exit(glyd_engine::cli::supports(&a)),
        Some("preflight") => std::process::exit(glyd_engine::cli::preflight(&a)),
        Some("fetch") => std::process::exit(glyd_engine::cli::fetch(&a)),
        Some("verify") => std::process::exit(glyd_engine::cli::verify(&a)),
        Some("prepare") => std::process::exit(glyd_engine::cli::prepare_cmd(&a)),
        // the cost profile of a model at a level, and this machine's memory tiers (docs/cost-profile.md)
        Some("costs") => glyd_engine::costs::costs_cmd(&a),
        Some("rates") => glyd_engine::costs::tiers_cmd(&a),
        // what a run of MODEL would do on this machine (or a described one), before anything is downloaded
        Some("fit") => std::process::exit(glyd_engine::plan::fit_cmd(&a)),
        // a range of a model's layers as a process: glyd-range/1 on the standard input and output (docs/range-worker.md)
        Some("rangeworker") => glyd_engine::dynamic::range::worker(&a),
        // Qwen3.8-Flash-Next behind glyd-hummingbird's tools/qwen_shim.py: one JSON request a line on stdin, one answer a line on stdout
        // (with --port or --host: the HTTP server, the chat page and the OpenAI and Anthropic APIs; else the JSON-lines protocol on the standard input)
        Some("qwenbench") => glyd_engine::qwen_tools::bench(&a),
        Some("qwenmcp") => glyd_engine::qwen_tools::mcp(&a),
        Some("qwenget") => glyd_engine::qwen4x_serve::get(a.need(1, "REPO")?, std::path::Path::new(a.need(2, "DIR")?), &a),
        Some("qwenserve") if a.has("port") || a.has("host") => {
            glyd_engine::qwen4x_serve::http(std::path::Path::new(a.need(1, "CKPT")?), std::path::Path::new(a.need(2, "PACK")?), &a)
        }
        Some("qwenserve") => glyd_engine::qwen4x_serve::serve(std::path::Path::new(a.need(1, "CKPT")?), std::path::Path::new(a.need(2, "PACK")?), &a),
        #[cfg(feature = "dev")]
        Some(c) => devcmd::run(c, &a),
        _ => err(USAGE),
    }
}
