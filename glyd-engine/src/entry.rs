//! The engine's command line as a library call: the `glyd-engine` program (src/main.rs) and `glyd` (the one program, with the engine linked: the `engine` feature
//! of the root crate) both run it, so `glyd run MODEL` and `glyd-engine run MODEL` are the same code.

use crate::args::Args;
use crate::ffi::{err, Result};

#[cfg(feature = "dev")]
pub const USAGE: &str = "usage: glyd-engine run MODEL[:LEVEL] [--prompt TEXT] [--host-memory GIB] | serve MODEL [--port N] [--host-memory GIB] | doctor | login | supports MODEL | preflight [MODEL[:LEVEL]] | fetch MODEL | prepare MODEL | verify MODEL | costs MODEL[:LEVEL] [--predict] | rates [--dir DIR] | fit MODEL[:LEVEL] [--machine NAME|FILE] | rangeworker --model MODEL[:LEVEL] --layers A..B | --version\n  LEVEL: penguin (lossless, the default), kestrel (about 6.5 bits per weight; --lossy is the same), swift (about 5.5), wren (about 4.5) or hummingbird (Nemotron 3 Super's experts at about 3)";
#[cfg(not(feature = "dev"))] // (wren runs in a development build only)
pub const USAGE: &str = "usage: glyd-engine run MODEL[:LEVEL] [--prompt TEXT] [--host-memory GIB] | serve MODEL [--port N] [--host-memory GIB] | doctor | login | supports MODEL | preflight [MODEL[:LEVEL]] | fetch MODEL | prepare MODEL | verify MODEL | costs MODEL[:LEVEL] [--predict] | rates [--dir DIR] | fit MODEL[:LEVEL] [--machine NAME|FILE] | rangeworker --model MODEL[:LEVEL] --layers A..B | --version\n  LEVEL: penguin (lossless, the default), kestrel (about 6.5 bits per weight; --lossy is the same), swift (about 5.5) or hummingbird (Nemotron 3 Super's experts at about 3)";

/// The commands of the engine that `glyd` takes as its own where the engine is linked (the same words as `glyd-engine`'s): what a model is run, served, fetched, packed-for-this-GPU, checked or costed with.
pub const COMMANDS: &[&str] = &[
    "run",
    "serve",
    "doctor",
    "login",
    "supports",
    "preflight",
    "fetch",
    "pull",
    "ls",
    "rm",
    "service",
    "ps",
    "stop",
    "verify",
    "prepare",
    "costs",
    "rates",
    "fit",
    "rangeworker",
];

/// glibc's allocator told to keep freed buffers up to 32 MB (its largest such setting) in the heap and not hand freed memory back: a load's threads (one a GPU,
/// each with its packers) free buffers of a few MB to tens of MB by the thousand, and every unmapping stops every CPU the process runs on to flush its TLB (the
/// 8-GPU Kimi-K3 load spent about half its time there). Called before any thread starts.
#[cfg(all(target_os = "linux", target_env = "gnu"))]
pub fn keep_buffers() {
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
pub fn keep_buffers() {}

/// A development build's commands (src/devcmd.rs, which only the `glyd-engine` program has): the ones the library does not know by name.
pub type Dev<'a> = Option<&'a dyn Fn(&str, &Args) -> Result<()>>;

/// The program as `prog` ("glyd-engine" or "glyd"): `args` are the words after its name; an error is said as `prog: ...` and exits 1.
pub fn main(prog: &str, args: Vec<String>, dev: Dev) -> ! {
    keep_buffers();
    if let Err(e) = run(prog, Args::from(args), dev) {
        eprintln!("{prog}: {e}");
        std::process::exit(1);
    }
    std::process::exit(0);
}

pub fn run(prog: &str, mut a: Args, dev: Dev) -> Result<()> {
    // a model given by its short name (qwen3.5:27b) is its Hub repo from here on (src/models.rs)
    if let Err(e) = crate::models::resolve_args(&mut a) {
        eprintln!("{prog}: {e}");
        std::process::exit(1);
    }
    // a model given as a GGUF file (or a Hub repo of GGUF files) is read as a Hugging Face directory (src/gguf.rs)
    if let Some(code) = crate::gguf::resolve_args(&mut a) {
        std::process::exit(code);
    }
    // a Hugging Face token given as a flag is the Hub client's before the environment's and the saved login's (glyd run and glyd serve move it to HF_TOKEN before they start this program)
    if let Some(t) = a.flag("hf-token").filter(|t| !t.is_empty()) {
        crate::hub::set_token(t);
        eprintln!("{prog}: the Hugging Face token is on this program's command line, which ps shows: next time set HF_TOKEN=TOKEN in the environment, or run glyd login.");
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
        // where the service is running, the chat goes through it (src/service.rs); else a process of its own as before
        Some("run") => std::process::exit(crate::service::attach(&a).unwrap_or_else(|| crate::cli::run(&a))),
        Some("serve") => std::process::exit(crate::cli::serve(&a)),
        Some("doctor") => std::process::exit(crate::cli::doctor(&a)),
        Some("login") => std::process::exit(crate::cli::login(&a)),
        Some("supports") => std::process::exit(crate::cli::supports(&a)),
        Some("preflight") => std::process::exit(crate::cli::preflight(&a)),
        Some("fetch") => std::process::exit(crate::cli::fetch(&a)),
        // the models in the cache: downloaded by name (a level's pre-packed repo where there is one), listed with their level and size, removed (src/models.rs)
        Some("pull") => std::process::exit(crate::models::pull(&a)),
        Some("ls" | "list") => std::process::exit(crate::models::ls(&a)),
        Some("rm") => std::process::exit(crate::models::rm(&a)),
        // the background service: models loaded on request and unloaded when idle; what it holds; a model unloaded (src/service.rs)
        Some("service") => std::process::exit(crate::service::run(&a)),
        Some("ps") => std::process::exit(crate::service::ps(&a)),
        Some("stop") => std::process::exit(crate::service::stop(&a)),
        Some("verify") => std::process::exit(crate::cli::verify(&a)),
        Some("prepare") => std::process::exit(crate::cli::prepare_cmd(&a)),
        // the cost profile of a model at a level, and this machine's memory tiers (docs/cost-profile.md)
        Some("costs") => crate::costs::costs_cmd(&a),
        Some("rates") => crate::costs::tiers_cmd(&a),
        // what a run of MODEL would do on this machine (or a described one), before anything is downloaded
        Some("fit") => std::process::exit(crate::plan::fit_cmd(&a)),
        // a range of a model's layers as a process: glyd-range/1 on the standard input and output (docs/range-worker.md)
        Some("rangeworker") => crate::dynamic::range::worker(&a),
        // Qwen3.8-Flash-Next behind glyd-hummingbird's tools/qwen_shim.py: one JSON request a line on stdin, one answer a line on stdout
        // (with --port or --host: the HTTP server, the chat page and the OpenAI and Anthropic APIs; else the JSON-lines protocol on the standard input)
        Some("qwenbench") => crate::qwen_tools::bench(&a),
        Some("qwenmcp") => crate::qwen_tools::mcp(&a),
        Some("qwenget") => crate::qwen4x_serve::get(a.need(1, "REPO")?, std::path::Path::new(a.need(2, "DIR")?), &a),
        Some("qwenserve") if a.has("port") || a.has("host") => {
            crate::qwen4x_serve::http(std::path::Path::new(a.need(1, "CKPT")?), std::path::Path::new(a.need(2, "PACK")?), &a)
        }
        Some("qwenserve") => crate::qwen4x_serve::serve(std::path::Path::new(a.need(1, "CKPT")?), std::path::Path::new(a.need(2, "PACK")?), &a),
        Some(c) if dev.is_some() => (dev.unwrap())(c, &a),
        _ => err(USAGE),
    }
}
