use std::env;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

fn print_usage() {
    eprintln!(
        r#"Glyd: LLM weights and KV cache compressed without loss, the engine that runs them, and compression for files
Usage: glyd run MODEL[:LEVEL]       chat with a model in the terminal and at one URL (MODEL: a Hub repo, a directory or a GGUF file)
       glyd serve MODEL[:LEVEL]     the OpenAI-compatible API and the chat page
       glyd pull MODEL              download a model (glyd fetch is the same)
       glyd fit MODEL               what a run of MODEL would do on this computer, before anything is downloaded
       glyd doctor | login          what this computer has for it; save a Hugging Face token
       glyd supports | preflight | prepare | costs | rates | rangeworker   the engine's other commands (glyd run --help)
       glyd pack MODEL OUT | glyd verify PATH   a model's weights for Glyd's GPU layouts, packed and checked on the CPU
       glyd [OPTIONS] [INPUT] [-o OUTPUT]       compress or decompress a file (the options below)
       (the store, which compresses across objects, is the glyd-store command)
       The engine is in this program where it was built with it (glyd --version says so), else the glyd-engine command beside it or on PATH.
       On a computer with the Python tool and vLLM, run and serve are its where the engine does not take the model; GLYD_ENGINE=1
       uses the engine on any GPU it runs on, GLYD_ENGINE=0 keeps vLLM; a file named run is ./run
Exit status: 0 done, 1 an error, 3 not supported (glyd supports MODEL: the engine does not run it)

Options:
    -c, --compress         Compress input (default if output is .glyd)
    -1, --fast             Fast level: LZ4-class compression speed, ratio ~2.10
    -t, --turbo            Turbo level: fastest decode, ~6% less ratio
    -9, --max              Max level: entropy coded, ratio above zstd -3
    -L, --long             With --max: repeats up to 128 MB back found in a pass before the parse
                           (as zstd --long); fewer bytes on events and logs, slower to write
    -D, --dense            With --max: 128 MB units parsed on all cores, one core's bytes at any
                           core count (5-9% fewer on files of a few hundred MB); reads scale
                           only with the units. For objects written once and read rarely
    -19, --ultra           Ultra level: optimal parse, ratio above zstd -16; slow to compress
    -C, --cold             Cold level: context mixing, the smallest output, 1-2 MB/s per core
                           each way; for what is stored for years and read rarely
    -r, --records          Record mode: logs and table dumps as typed columns before the level
    -B, --base <FILE>      Base mode: compress a new version against this old one (--max, or
                           --ultra); decoding needs the same file
    -S, --shape <FILE>     Shape dictionary: compress or decompress a small object (an event, a
                           small log or CSV) with a dictionary trained on a sample of such data
        --shape-train      Train a shape dictionary on the input (a few MB) and write it to -o
    -P, --pack             Pack the input files (many small objects) into one stream with an
                           index, record mode where it pays; any one object is read back alone
    -U, --unpack <DIR>     Write a pack's objects into DIR as 000000, 000001, ...
    -d, --decompress       Decompress input (default if input is .glyd)
        --content          With -d: the content of a gzip or zlib object stored opened (what
                           gunzip gives), without re-creating the stream
    -m, --multi-core       Use multi-core parallel engine (default)
    -s, --single-core      Force single-core sequential engine
    -o, --output <FILE>    Specify destination output file (defaults to stdout if piped)
    -b, --bench            Benchmark compression and decompression throughput
    -v, --version          Print version and CPU SIMD hardware capabilities
    -h, --help             Print this help message

Examples:
    glyd input.tar -o input.tar.glyd
    glyd -d input.tar.glyd -o input.tar
    glyd --base dump-monday.sql dump-tuesday.sql -o tuesday.glyd
    glyd -d --base dump-monday.sql tuesday.glyd -o dump-tuesday.sql
    glyd --pack events/*.json -o events.glyd
    glyd --unpack out/ events.glyd
    glyd --shape-train sample.log -o events.shape
    glyd --shape events.shape event.log -o event.glyd
    glyd -d --shape events.shape event.glyd -o event.log
    cat large.json | glyd -c > large.json.glyd
    cat large.json.glyd | glyd -d > large.json
    glyd -b dataset.bin
"#
    );
}

fn print_version() {
    println!("glyd {}", env!("CARGO_PKG_VERSION"));
    if LINKED {
        println!("engine: linked (glyd run, serve, pull, fit and doctor are this program's own)");
    }
    // The kernels are AVX2 (x86-64) and NEON (aarch64); everything else runs scalar.
    #[cfg(target_arch = "x86_64")]
    println!("SIMD: {}", if is_x86_feature_detected!("avx2") { "AVX2" } else { "scalar" });
    #[cfg(target_arch = "aarch64")]
    println!("SIMD: NEON");
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    println!("SIMD: scalar");
}

/// `glyd pack` and `glyd verify`: the glyd-gpu command's (the GPU weights' packer and checker, a program of its own
/// under their license), found beside this one, else on PATH; its exit status this one's.
fn gpu_command(args: &[String]) -> ! {
    let beside = env::current_exe().ok().and_then(|p| p.parent().map(|d| d.join("glyd-gpu"))).filter(|p| p.exists());
    let exe = beside.unwrap_or_else(|| "glyd-gpu".into());
    match std::process::Command::new(&exe).args(args).status() {
        Ok(s) => std::process::exit(s.code().unwrap_or(1)),
        Err(e) => {
            eprintln!(
                "glyd {}: the glyd-gpu command ({}): {e}; it ships beside glyd (cargo install --git https://github.com/surya-koritala/Glyd glyd-gpu)",
                args[0],
                exe.display()
            );
            std::process::exit(1);
        }
    }
}

/// Whether a file is a console script of Glyd's Python tool: a `#!` line, and the module it imports (glyd.cli) in its first 4 KB.
fn is_python_tool(path: &Path) -> bool {
    let mut head = Vec::new();
    let read = std::fs::File::open(path).and_then(|f| f.take(4096).read_to_end(&mut head));
    read.is_ok() && head.starts_with(b"#!") && head.windows(8).any(|w| w == b"glyd.cli")
}

/// The Python tool's `glyd`: the first one on PATH that is not this program, else where `uv tool install` puts it (UV_TOOL_BIN_DIR,
/// XDG_BIN_HOME, ~/.local/bin), which a terminal opened before the install does not have on its PATH yet. An empty entry of PATH
/// (the current directory) is skipped.
fn find_python_tool() -> Option<PathBuf> {
    let me = env::current_exe().ok().and_then(|p| p.canonicalize().ok());
    let mut dirs: Vec<PathBuf> = env::var_os("PATH").map(|p| env::split_paths(&p).collect()).unwrap_or_default();
    let home = env::var_os("HOME").map(PathBuf::from);
    for v in ["UV_TOOL_BIN_DIR", "XDG_BIN_HOME"] {
        dirs.extend(env::var_os(v).map(PathBuf::from));
    }
    dirs.extend(home.map(|h| h.join(".local").join("bin")));
    dirs.into_iter()
        .filter(|d| !d.as_os_str().is_empty())
        .map(|d| d.join("glyd"))
        .find(|exe| exe.is_file() && exe.canonicalize().ok() != me && is_python_tool(exe))
}

/// `glyd run`, `glyd serve`, `glyd doctor` and `glyd login` are the local chat's commands, in Glyd's Python tool (install.sh,
/// `pip install "glyd[vllm]"`): not a file's name here, which is what `glyd run MODEL` was before, with the usage text for an answer.
/// The tool's `glyd` takes them over from this program (its Ctrl-C and its exit status are its own), started with GLYD_FORWARDED=1:
/// the tool passes the compression commands the other way with the same variable and takes none of them back, so an older tool that
/// sends `run` here cannot start a loop. Where there is no tool the answer says what to do, or that this computer is not for it.
/// `extra`: variables for the tool alone (GLYD_ENGINE=0, where this program has already said that the command goes the vLLM way).
fn python_command(args: &[String], extra: &[(&str, &str)]) -> ! {
    let forwarded = env::var_os("GLYD_FORWARDED").map_or(false, |v| !v.is_empty()); // (empty is not set, as the Python tool takes it)
    if !forwarded {
        if let Some(exe) = find_python_tool() {
            let mut cmd = std::process::Command::new(&exe);
            cmd.args(args).env("GLYD_FORWARDED", "1").envs(extra.iter().copied());
            #[cfg(unix)]
            let e = std::os::unix::process::CommandExt::exec(&mut cmd);
            #[cfg(not(unix))]
            let e = match cmd.status() {
                Ok(s) => std::process::exit(s.code().unwrap_or(1)),
                Err(e) => e,
            };
            eprintln!("glyd {}: could not start {}: {e}", args[0], exe.display());
            std::process::exit(126);
        }
    }
    let why = if forwarded {
        "the Python tool's glyd sent it back to this program, so it has no such command: update it with curl -LsSf https://getglyd.com/install.sh | sh"
    } else if cfg!(target_os = "linux") {
        "that is the local chat's command, in Glyd's Python tool, which is not installed here. On Linux with an NVIDIA GPU: curl -LsSf https://getglyd.com/install.sh | sh"
    } else {
        "glyd run needs Linux with an NVIDIA GPU, and this computer is not one (this program compresses files: glyd --help)"
    };
    eprintln!("glyd {}: {why} (a file named {} is compressed as ./{})", args[0], args[0], args[0]);
    std::process::exit(127);
}

/// What GLYD_ENGINE says: Off (0, off, no, false or vllm: the Python tool's vLLM path), Auto (unset, 1, on, yes, true or auto), or the path of an engine program.
#[derive(Debug, PartialEq)]
enum Engine {
    Off,
    Auto,
    Path(PathBuf),
}

fn engine_choice() -> Engine {
    let v = env::var("GLYD_ENGINE").unwrap_or_default();
    match v.trim().to_ascii_lowercase().as_str() {
        "0" | "off" | "no" | "false" | "vllm" => Engine::Off,
        "" | "1" | "on" | "yes" | "true" | "auto" => Engine::Auto,
        _ => Engine::Path(PathBuf::from(v.trim())),
    }
}

/// Whether GLYD_ENGINE asks for the engine by name (1, on, yes or true, any case); `auto` and nothing leave it to the default: `glyd serve` is vLLM's, as the engine serves up to 8 users at once.
fn engine_asked_for() -> bool {
    matches!(env::var("GLYD_ENGINE").unwrap_or_default().trim().to_ascii_lowercase().as_str(), "1" | "on" | "yes" | "true")
}

/// Glyd's engine, the glyd-engine command (a program of its own under the GPU code's license): the program GLYD_ENGINE names, else beside this program, else
/// on PATH. (The one in the Python tool's package is the Python tool's to start: it is what `run` and `serve` are passed to.)
fn find_engine(choice: &Engine) -> Option<PathBuf> {
    if let Engine::Path(p) = choice {
        return Some(p.clone()).filter(|p| p.is_file());
    }
    let beside = env::current_exe().ok().and_then(|p| p.parent().map(|d| d.join("glyd-engine"))).filter(|p| p.is_file());
    beside.or_else(|| {
        env::var_os("PATH").and_then(|p| env::split_paths(&p).filter(|d| !d.as_os_str().is_empty()).map(|d| d.join("glyd-engine")).find(|e| e.is_file()))
    })
}

/// The engine's commands about a model that are not the local chat's: in a Rust-only install (install.sh without the Python tool) this program
/// passes them to the engine; a file of one of these names (./fetch) is still compressed where the engine is not installed.
fn is_model_command(cmd: &str) -> bool {
    matches!(cmd, "fetch" | "preflight" | "prepare" | "supports" | "costs" | "rates")
}

/// The engine's commands that the Python tool has no such command for: they are the engine's wherever one is here (linked, beside this program or on PATH), the tool or no tool.
/// `pull` is `fetch` (the engine's own word for the download); a file of one of these names (./fit) is still compressed where there is no engine.
fn is_engine_command(cmd: &str) -> bool {
    matches!(cmd, "fit" | "pull" | "rangeworker")
}

/// Whether the engine is linked into this program (the `engine` feature: the release tarballs' glyd).
const LINKED: bool = cfg!(feature = "engine");

/// What the routing of a command goes by, read from this computer once (and given by hand in the tests).
struct Facts {
    /// GLYD_FORWARDED is set: the Python tool's glyd sent the command here.
    forwarded: bool,
    /// The Python tool's glyd is installed.
    python_tool: bool,
    /// The glyd-gpu command is beside this program.
    gpu_tool_beside: bool,
    /// What GLYD_ENGINE says.
    choice: Engine,
    /// GLYD_ENGINE asks for the engine by name (1, on, yes, true).
    asked: bool,
    /// An engine program found (GLYD_ENGINE's path, beside this program, on PATH): not the linked one.
    external: Option<PathBuf>,
    /// The command's word is a file here (`./fit`), which the compression program takes where there is no engine.
    is_file: bool,
}

impl Facts {
    fn read(cmd: &str) -> Facts {
        let choice = engine_choice();
        Facts {
            forwarded: env::var_os("GLYD_FORWARDED").is_some_and(|v| !v.is_empty()), // (empty is not set, as the Python tool takes it)
            python_tool: find_python_tool().is_some(),
            gpu_tool_beside: gpu_tool_beside(),
            asked: engine_asked_for(),
            external: find_engine(&choice),
            choice,
            is_file: Path::new(cmd).exists(),
        }
    }

    /// Whether there is an engine to run a command: the linked one, or a program found.
    fn engine_here(&self) -> bool {
        LINKED || self.external.is_some()
    }
}

/// Where a command goes.
#[derive(Debug, PartialEq)]
enum Route {
    /// A file to compress or decompress (or this program's own flags).
    Codec,
    /// The glyd-gpu command.
    GpuTool,
    /// The Python tool's `glyd` (which says what to do where there is none), with these variables for it alone.
    Python(Vec<(&'static str, String)>),
    /// The engine linked into this program.
    Engine,
    /// An engine program of its own: GLYD_ENGINE's, beside this program, or on PATH.
    EngineProgram(PathBuf),
    /// An engine command and no engine here.
    NoEngine,
}

/// `run`, `serve`, `doctor`, `login` and the engine's other commands about a model. Where there is a Python tool, `run` and `serve` are its: it asks the engine (the program found, never the linked
/// one, which the tool cannot start: GLYD_ENGINE names a file) whether it is used on this GPU and will certainly run the model, before anything is downloaded, runs it as its child where it will, and
/// runs vLLM, with one line saying why, where it will not; `serve` is vLLM's unless GLYD_ENGINE asks for the engine (the engine serves up to 8 users at once). This program says nothing of its own there:
/// it tells the tool which engine it found, and the tool, which has vLLM to fall back to, does the rest. Where there is no tool there is no other runner: the engine (linked, else a program found)
/// runs the command, and its refusals are the last word. `doctor` and `login` are the engine's where one is here, else the tool's.
fn chat_route(cmd: &str, f: &Facts) -> Route {
    if f.forwarded {
        return Route::Python(vec![]); // (it says what this program would say: the tool has the engine's choice to make, and no loop to start)
    }
    if is_model_command(cmd) && f.python_tool {
        return Route::Python(vec![]); // (the tool knows the models vLLM runs too)
    }
    if matches!(cmd, "run" | "serve") && f.python_tool {
        let found = if f.choice == Engine::Auto && (cmd == "run" || f.asked) { f.external.as_ref().and_then(|p| p.to_str()) } else { None }; // (a path in GLYD_ENGINE asks the tool for the engine on `serve`: it is only given where it was asked for)
        return match found {
            Some(path) if f.asked => Route::Python(vec![("GLYD_ENGINE", path.to_string()), ("GLYD_ENGINE_ASKED", "1".to_string())]), // (the user's 1, which the path takes the place of)
            Some(path) => Route::Python(vec![("GLYD_ENGINE", path.to_string())]),
            None => Route::Python(vec![]), // (GLYD_ENGINE=0, or a program the user named, which the tool is told by the environment; or no engine program here: the tool's own package has one, or vLLM runs it)
        };
    }
    match &f.choice {
        Engine::Off => Route::Python(vec![]),
        Engine::Path(_) => f.external.clone().map_or(Route::Python(vec![]), Route::EngineProgram), // (the program the user named runs it, linked or not)
        Engine::Auto if LINKED => Route::Engine,
        Engine::Auto => f.external.clone().map_or(Route::Python(vec![]), Route::EngineProgram),
    }
}

/// Where `glyd CMD ...` goes (`cmd`: the first word after the program's name).
fn route(cmd: &str, f: &Facts) -> Route {
    if cmd == "verify" && !f.gpu_tool_beside && !f.python_tool && f.engine_here() {
        return chat_route(cmd, f); // (the Rust-only install: no glyd-gpu beside this program, so `glyd verify MODEL` is the engine's)
    }
    if matches!(cmd, "pack" | "verify") {
        return Route::GpuTool; // (a file named pack or verify: ./pack)
    }
    if matches!(cmd, "run" | "serve" | "doctor" | "login") {
        return chat_route(cmd, f); // (a file named run: ./run)
    }
    if is_model_command(cmd) && (f.python_tool || f.engine_here()) {
        return chat_route(cmd, f); // (a model's command: the Python tool's where there is one, else the engine's, as in the Rust-only install)
    }
    if is_engine_command(cmd) {
        // (the Python tool has no such command: the engine runs it, or nothing does)
        let engine = match &f.choice {
            Engine::Off => None,
            Engine::Path(_) => f.external.clone().map(Route::EngineProgram),
            Engine::Auto if LINKED => Some(Route::Engine),
            Engine::Auto => f.external.clone().map(Route::EngineProgram),
        };
        return engine.unwrap_or(if f.is_file { Route::Codec } else { Route::NoEngine });
    }
    Route::Codec
}

/// Whether the glyd-gpu command (the packer and file checker) is beside this program, as the full install has it.
fn gpu_tool_beside() -> bool {
    env::current_exe().ok().and_then(|p| p.parent().map(|d| d.join("glyd-gpu"))).is_some_and(|p| p.exists())
}

/// The model of `glyd run MODEL ...`: the first word that is not a flag or a flag's value.
fn model_arg(args: &[String]) -> Option<&str> {
    const WITH_VALUE: [&str; 12] =
        ["--prompt", "-p", "--context", "--ctx", "--port", "--host", "--temp", "--layout", "--users", "--tables", "--api-key", "--hf-token"];
    let mut it = args.iter().skip(1);
    while let Some(a) = it.next() {
        if a == "--" {
            return None;
        }
        if WITH_VALUE.contains(&a.as_str()) {
            it.next();
        } else if !a.starts_with('-') {
            return Some(a.as_str());
        }
    }
    None
}

/// A flag that carries a secret (`--api-key KEY`, `--api-key=KEY`) taken out of `args` with its value, for the engine's environment: a secret on a command line is in `ps` and in a log. None where the flag is not
/// there or has no value after it (`--hf-token` alone means the saved login: it stays). Words after a lone `--` are not looked at, nor are the values of the engine's other flags.
fn take_secret(args: &mut Vec<String>, names: &[&str]) -> Option<String> {
    const WITH_VALUE: [&str; 10] = ["--prompt", "-p", "--context", "--ctx", "--port", "--host", "--temp", "--layout", "--users", "--tables"];
    let (mut found, mut i) = (None, 0);
    while i < args.len() {
        if args[i] == "--" {
            break;
        }
        let (name, eq) = match args[i].split_once('=') {
            Some((n, v)) => (n.to_string(), Some(v.to_string())),
            None => (args[i].clone(), None),
        };
        if names.contains(&name.as_str()) {
            if let Some(v) = eq {
                found = Some(v);
                args.remove(i);
                continue;
            }
            if i + 1 < args.len() && !args[i + 1].starts_with('-') {
                found = Some(args[i + 1].clone());
                args.drain(i..=i + 1);
                continue;
            }
        } else if WITH_VALUE.contains(&name.as_str()) && eq.is_none() {
            i += 1; // (the flag's value is not a flag of ours, whatever it looks like)
        }
        i += 1;
    }
    found
}

/// Runs a route's command: `args` are the words after this program's name. Never returns for a route but Codec.
fn go(route: Route, args: &[String]) -> ! {
    let cmd = args[0].as_str();
    match route {
        Route::Codec => unreachable!("the compression program's own"),
        Route::GpuTool => gpu_command(args),
        Route::Python(extra) => {
            let extra: Vec<(&str, &str)> = extra.iter().map(|(k, v)| (*k, v.as_str())).collect();
            python_command(args, &extra)
        }
        Route::NoEngine => {
            eprintln!(
                "glyd {cmd}: the engine is not here: this program was built without it, and there is no glyd-engine command beside it or on PATH. Install the engine's release (curl -LsSf https://getglyd.com/install.sh | sh) \
                 (a file named {cmd} is compressed as ./{cmd})"
            );
            std::process::exit(127);
        }
        Route::Engine | Route::EngineProgram(_) => {
            let model = if matches!(cmd, "run" | "serve") && !args.iter().any(|a| a == "-h" || a == "--help") { model_arg(args) } else { None };
            if let Some(model) = model {
                let on = if matches!(route, Route::Engine) { "the engine" } else { "glyd-engine" };
                eprintln!("glyd {cmd}: {model} on {on} (GLYD_ENGINE=0: vLLM instead).");
            }
            // an API key or a token given as a flag goes to the engine in its environment, not on its command line (where `ps` and a log show it)
            let mut run_args = args.to_vec();
            let mut secrets = Vec::new();
            for (names, var, what) in
                [(&["--api-key", "--api_key"][..], "GLYD_API_KEY", "API key"), (&["--hf-token", "--hf_token"][..], "HF_TOKEN", "Hugging Face token")]
            {
                if let Some(v) = take_secret(&mut run_args, names) {
                    secrets.push((var, v));
                    eprintln!("glyd {cmd}: the {what} goes to the engine in {var}, not on its command line (next time set {var} yourself: a shell's history keeps what you type).");
                }
            }
            match route {
                Route::EngineProgram(engine) => {
                    let mut c = std::process::Command::new(&engine);
                    c.envs(secrets).args(&run_args);
                    #[cfg(unix)]
                    let e = std::os::unix::process::CommandExt::exec(&mut c);
                    #[cfg(not(unix))]
                    let e = match c.status() {
                        Ok(s) => std::process::exit(s.code().unwrap_or(1)),
                        Err(e) => e,
                    };
                    eprintln!("glyd {}: could not start {}: {e}", args[0], engine.display());
                    std::process::exit(126);
                }
                _ => {
                    for (var, v) in secrets {
                        env::set_var(var, v); // (this program has started no thread yet)
                    }
                    run_linked_engine(run_args)
                }
            }
        }
    }
}

/// The linked engine's command line, as `glyd-engine` has it: `args` are the words after the program's name.
#[cfg(feature = "engine")]
fn run_linked_engine(args: Vec<String>) -> ! {
    glyd_engine::entry::main("glyd", args, None)
}

#[cfg(not(feature = "engine"))]
fn run_linked_engine(_: Vec<String>) -> ! {
    unreachable!("a route to the linked engine is made only where it is linked")
}

fn main() -> io::Result<()> {
    let args: Vec<String> = env::args().collect();
    // `glyd-engine`, a link to this program (or a copy of it) where the engine is linked: the engine's own command line, as the program of that name had it
    #[cfg(feature = "engine")]
    if env::args_os().next().is_some_and(|a| Path::new(&a).file_stem().is_some_and(|s| s == "glyd-engine")) {
        glyd_engine::entry::main("glyd-engine", args[1..].to_vec(), None);
    }
    if args.len() == 1 {
        print_usage();
        return Ok(());
    }
    match route(&args[1], &Facts::read(&args[1])) {
        Route::Codec => {}
        other => go(other, &args[1..]),
    }

    let mut input_path: Option<String> = None;
    let mut output_path: Option<String> = None;
    let mut mode_compress = None;
    let mut content = false;
    let mut multi_core = true;
    let mut benchmark_mode = false;
    let mut fast = false;
    let mut turbo = false;
    let mut max = false;
    let mut long = false;
    let mut dense = false;
    let mut ultra = false;
    let mut cold = false;
    let mut records = false;
    let mut base_path: Option<String> = None;
    let mut shape_path: Option<String> = None;
    let mut shape_train = false;
    let mut pack = false;
    let mut unpack_dir: Option<String> = None;
    let mut inputs: Vec<String> = Vec::new();

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "-h" | "--help" => {
                print_usage();
                return Ok(());
            }
            "-v" | "--version" => {
                print_version();
                return Ok(());
            }
            "-c" | "--compress" => mode_compress = Some(true),
            "-1" | "--fast" => fast = true,
            "-t" | "--turbo" => turbo = true,
            "-9" | "--max" => max = true,
            "-L" | "--long" => long = true,
            "-D" | "--dense" => dense = true,
            "-19" | "--ultra" => ultra = true,
            "-C" | "--cold" => cold = true,
            "-r" | "--records" => records = true,
            "-B" | "--base" => {
                if i + 1 < args.len() {
                    base_path = Some(args[i + 1].clone());
                    i += 1;
                } else {
                    eprintln!("Error: --base requires a file");
                    std::process::exit(1);
                }
            }
            "-S" | "--shape" => {
                if i + 1 < args.len() {
                    shape_path = Some(args[i + 1].clone());
                    i += 1;
                } else {
                    eprintln!("Error: --shape requires a file");
                    std::process::exit(1);
                }
            }
            "--shape-train" => shape_train = true,
            "-P" | "--pack" => pack = true,
            "-U" | "--unpack" => {
                if i + 1 < args.len() {
                    unpack_dir = Some(args[i + 1].clone());
                    i += 1;
                } else {
                    eprintln!("Error: --unpack requires a directory");
                    std::process::exit(1);
                }
            }
            "-d" | "--decompress" => mode_compress = Some(false),
            "--content" => {
                mode_compress = Some(false);
                content = true;
            }
            "-m" | "--multi-core" => multi_core = true,
            "-s" | "--single-core" => {
                // One thread everywhere: record units, container entries
                // and the decoders' units run on the calling thread too.
                multi_core = false;
                glyd::set_threads(1);
            }
            "-b" | "--bench" => benchmark_mode = true,
            "-o" | "--output" => {
                if i + 1 < args.len() {
                    output_path = Some(args[i + 1].clone());
                    i += 1;
                } else {
                    eprintln!("Error: -o requires an output filename");
                    std::process::exit(1);
                }
            }
            other => {
                if !other.starts_with('-') && input_path.is_none() {
                    input_path = Some(other.to_string());
                    inputs.push(other.to_string());
                } else if !other.starts_with('-') && pack {
                    inputs.push(other.to_string());
                } else {
                    eprintln!("Unknown option: {}", other);
                    print_usage();
                    std::process::exit(1);
                }
            }
        }
        i += 1;
    }

    // Benchmark mode
    if benchmark_mode {
        let path = match input_path {
            Some(ref p) => p,
            None => {
                eprintln!("Error: Benchmark requires an input file");
                std::process::exit(1);
            }
        };
        run_benchmark(path);
        return Ok(());
    }

    if pack {
        let files: Vec<Vec<u8>> = inputs.iter().map(std::fs::read).collect::<io::Result<_>>()?;
        let refs: Vec<&[u8]> = files.iter().map(|v| v.as_slice()).collect();
        let level: fn(&[u8], &mut Vec<u8>) = if cold {
            glyd::compress_into_cold
        } else if ultra {
            glyd::compress_into_ultra
        } else {
            glyd::compress_into_max
        };
        let mut out = Vec::new();
        glyd::compress_pack(&refs, &mut out, level);
        return write_out(&output_path, &out);
    }
    if let Some(dir) = unpack_dir {
        let data = match input_path {
            Some(ref p) => std::fs::read(p)?,
            None => {
                eprintln!("Error: --unpack needs the pack file");
                std::process::exit(1);
            }
        };
        let objects = match glyd::decompress_pack(&data) {
            Ok(o) => o,
            Err(e) => {
                eprintln!("Decompression failed: {}", e);
                std::process::exit(1);
            }
        };
        std::fs::create_dir_all(&dir)?;
        for (i, o) in objects.iter().enumerate() {
            std::fs::write(Path::new(&dir).join(format!("{:06}", i)), o)?;
        }
        return Ok(());
    }

    // Read input data
    // A file is mapped, not copied: the read then costs page faults
    // spread over the compressing threads instead of a pass before.
    let mut mapped;
    let owned;
    let input_data: &[u8] = match input_path {
        Some(ref p) if p != "-" => {
            mapped = glyd::mmap::Mapping::read_only(Path::new(p))?;
            mapped.will_need();
            mapped.populate();
            mapped.bytes()
        }
        _ => {
            let mut data = Vec::new();
            io::stdin().read_to_end(&mut data)?;
            owned = data;
            &owned
        }
    };

    // Determine compress vs decompress
    let should_compress = mode_compress.unwrap_or_else(|| if let Some(ref p) = input_path { !p.ends_with(".glyd") } else { true });

    let base = match base_path {
        Some(ref p) => Some(std::fs::read(p)?),
        None => None,
    };
    if shape_train {
        let dict = match glyd::ShapeDict::train(&input_data) {
            Some(d) => d,
            None => {
                eprintln!("Error: the input is not record-shaped (delimited lines, JSON lines or a log)");
                std::process::exit(1);
            }
        };
        return write_out(&output_path, &dict.to_bytes());
    }
    if let Some(ref p) = shape_path {
        let dict = match glyd::ShapeDict::from_bytes(&std::fs::read(p)?) {
            Some(d) => d,
            None => {
                eprintln!("Error: {} is not a shape dictionary", p);
                std::process::exit(1);
            }
        };
        let out = if should_compress {
            let mut out = Vec::new();
            dict.compress(&input_data, &mut out);
            out
        } else {
            match dict.decompress(&input_data) {
                Ok(d) => d,
                Err(e) => {
                    eprintln!("Decompression failed: {}", e);
                    std::process::exit(1);
                }
            }
        };
        return write_out(&output_path, &out);
    }
    let output_data = if should_compress {
        let mut out = Vec::with_capacity(input_data.len() / 2 + 1024);
        if let Some(ref base) = base {
            glyd::compress_with_base(base, &input_data, &mut out, ultra);
            match output_path {
                Some(ref p) if p != "-" => std::fs::write(p, &out)?,
                _ => {
                    io::stdout().write_all(&out)?;
                    io::stdout().flush()?;
                }
            }
            return Ok(());
        }
        let mc = multi_core && input_data.len() > glyd::format::MAX_BLOCK_SIZE;
        let level: fn(&[u8], &mut Vec<u8>) = match (mc, fast, turbo, max) {
            (true, _, _, _) if cold => glyd::compress_parallel_into_cold,
            (false, _, _, _) if cold => glyd::compress_into_cold,
            (true, _, _, _) if ultra => glyd::compress_parallel_into_ultra,
            (false, _, _, _) if ultra => glyd::compress_into_ultra,
            (true, _, _, true) if long => glyd::compress_parallel_into_max_long,
            (false, _, _, true) if long => glyd::compress_into_max_long,
            (true, _, _, true) => glyd::compress_parallel_into_max,
            (false, _, _, true) => glyd::compress_into_max,
            (true, true, _, _) => glyd::compress_parallel_into_fast,
            (true, _, true, _) => glyd::compress_parallel_into_turbo,
            (true, _, _, _) => glyd::compress_parallel_into,
            (false, true, _, _) => glyd::compress_into_fast,
            (false, _, true, _) => glyd::compress_into_turbo,
            (false, _, _, _) => glyd::compress_into,
        };
        let stream_level: Option<(fn(&[u8], &mut Vec<u8>), usize)> = match (mc, records, cold, ultra, max, fast, turbo) {
            (true, false, false, false, true, _, _) => {
                Some((if long { glyd::compress_into_max_long } else { glyd::compress_into_max }, glyd::format::PARALLEL_UNIT_MAX))
            }
            (true, false, false, true, _, _, _) => Some((glyd::compress_into_ultra, glyd::format::PARALLEL_UNIT_ULTRA)),
            _ => None,
        };
        // A deflate container is opened by the level itself (which keeps
        // it closed when that is smaller), not streamed.
        let stream_level = if glyd::deflate::is_container(input_data) { None } else { stream_level };
        if let Some((unit_level, smallest)) = stream_level {
            // Units written as they finish: the output never sits whole
            // in memory, and the write overlaps the compressing.
            let mut sink: Box<dyn Write + Send> = match output_path {
                Some(ref p) if p != "-" => Box::new(std::io::BufWriter::with_capacity(1 << 20, std::fs::File::create(p)?)),
                _ => Box::new(io::stdout()),
            };
            if max && !ultra && dense {
                glyd::compress_max_stream(input_data, |part| sink.write_all(part))?;
            } else {
                glyd::compress_stream(input_data, unit_level, smallest, |unit| sink.write_all(unit))?;
            }
            sink.flush()?;
            return Ok(());
        }
        if !mc && max && !ultra && !cold && !records && !dense && base.is_none() {
            // One core: the blocks are written by a second thread as
            // they are made (zstd's CLI has an I/O thread beside its one
            // worker too), so the write is not a pass after the parse.
            let (tx, rx) = std::sync::mpsc::sync_channel::<Vec<u8>>(8);
            let mut sink: Box<dyn Write + Send> = match output_path {
                Some(ref p) if p != "-" => Box::new(std::io::BufWriter::with_capacity(1 << 20, std::fs::File::create(p)?)),
                _ => Box::new(io::stdout()),
            };
            let writer = std::thread::spawn(move || -> io::Result<()> {
                for chunk in rx {
                    sink.write_all(&chunk)?;
                }
                sink.flush()
            });
            glyd::compress_max_to(input_data, long, |chunk| {
                let _ = tx.send(chunk);
            });
            drop(tx);
            return writer.join().expect("writer thread");
        }
        if records && cold {
            glyd::compress_records_into_cold(&input_data, &mut out);
        } else if max && !ultra && (records || glyd::deflate::is_container(input_data)) {
            // An opened container's content takes record mode where
            // that pays (a gzipped log is a log).
            if dense {
                glyd::compress_records_into_max_dense(&input_data, &mut out)
            } else if long {
                glyd::compress_records_into_max_long(&input_data, &mut out)
            } else {
                glyd::compress_records_into_max(&input_data, &mut out)
            }
        } else if ultra && !cold && glyd::deflate::is_container(input_data) {
            glyd::compress_records_into_ultra(&input_data, &mut out);
        } else if records {
            // Record mode parallelises over its own units; the level
            // inside a unit is the sequential one.
            let unit_level: fn(&[u8], &mut Vec<u8>) = match (fast, turbo, max, ultra) {
                (_, _, _, true) => glyd::compress_into_ultra,
                (_, _, true, _) if long => glyd::compress_into_max_long,
                (_, _, true, _) => glyd::compress_into_max,
                (true, _, _, _) => glyd::compress_into_fast,
                (_, true, _, _) => glyd::compress_into_turbo,
                _ => glyd::compress_into,
            };
            glyd::compress_records_with(&input_data, &mut out, unit_level);
        } else {
            level(&input_data, &mut out);
        }
        out
    } else {
        // Decoded a batch of units at a time into one reused buffer and
        // written as it goes: the memory is a batch, not the file.
        let mut out: Box<dyn Write + Send> = match output_path {
            Some(ref p) if p != "-" => Box::new(std::fs::File::create(p)?),
            _ => Box::new(io::stdout()),
        };
        let result = if content {
            let r = match base {
                Some(ref base) => glyd::decompress_content_with_base(base, &input_data),
                None => glyd::decompress_content(&input_data),
            };
            match r {
                Ok(data) => out.write_all(&data),
                Err(e) => Err(io::Error::new(io::ErrorKind::InvalidData, e)),
            }
        } else if glyd::needs_base(&input_data) {
            match base {
                Some(ref base) => glyd::decompress_stream_with_base(base, &input_data, |batch| out.write_all(batch)),
                None => Err(io::Error::new(io::ErrorKind::InvalidData, "this file was compressed against a base: pass it with --base")),
            }
        } else if input_data.len() > glyd::format::MAX_BLOCK_SIZE {
            // One core or all: the streaming path either way (units into
            // a reused buffer, written as they are decoded); a whole
            // output buffer faulted in page by page was most of a read.
            if !multi_core {
                glyd::set_threads(1);
            }
            glyd::decompress_stream(&input_data, |batch| out.write_all(batch))
        } else {
            match glyd::decompress(&input_data) {
                Ok(data) => out.write_all(&data),
                Err(e) => Err(io::Error::new(io::ErrorKind::InvalidData, e)),
            }
        };
        if let Err(e) = result {
            if e.kind() == io::ErrorKind::InvalidData {
                eprintln!("Decompression failed: {}", e);
                std::process::exit(1);
            }
            return Err(e);
        }
        out.flush()?;
        return Ok(());
    };

    // Determine output destination
    match output_path {
        Some(ref p) if p != "-" => {
            std::fs::write(p, &output_data)?;
        }
        _ => {
            io::stdout().write_all(&output_data)?;
            io::stdout().flush()?;
        }
    }

    Ok(())
}

fn write_out(output_path: &Option<String>, data: &[u8]) -> io::Result<()> {
    match output_path {
        Some(p) if p != "-" => std::fs::write(p, data),
        _ => {
            io::stdout().write_all(data)?;
            io::stdout().flush()
        }
    }
}

fn run_benchmark(path_str: &str) {
    let path = Path::new(path_str);
    let data = match std::fs::read(path) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("Failed to read {}: {}", path_str, e);
            return;
        }
    };

    let len = data.len();
    println!("Benchmarking: {} ({:.2} MB)", path_str, len as f64 / (1024.0 * 1024.0));

    // Sequential compression
    let start = Instant::now();
    let comp_seq = glyd::compress(&data);
    let seq_comp_sec = start.elapsed().as_secs_f64();
    let seq_comp_gb = (len as f64 / (1024.0 * 1024.0 * 1024.0)) / seq_comp_sec;

    // Parallel compression
    let start = Instant::now();
    let comp_par = glyd::compress_parallel(&data);
    let par_comp_sec = start.elapsed().as_secs_f64();
    let par_comp_gb = (len as f64 / (1024.0 * 1024.0 * 1024.0)) / par_comp_sec;

    let ratio = len as f64 / comp_seq.len() as f64;
    println!("  Compression ratio: {:.2}x ({:.2} MB -> {:.2} MB)", ratio, len as f64 / (1024.0 * 1024.0), comp_seq.len() as f64 / (1024.0 * 1024.0));
    println!("  1-Core Compression:   {:>7.2} GB/s", seq_comp_gb);
    println!("  Multi-Core Compress:  {:>7.2} GB/s", par_comp_gb);

    // Decompression benchmark
    let iters = (300 * 1024 * 1024 / len).max(5);
    let mut dst = vec![0u8; len + 128];

    let start = Instant::now();
    for _ in 0..iters {
        glyd::decompress_into_raw(&comp_seq, &mut dst).unwrap();
    }
    let raw_1c_gb = (len as f64 * iters as f64 / (1024.0 * 1024.0 * 1024.0)) / start.elapsed().as_secs_f64();

    let start = Instant::now();
    for _ in 0..iters {
        glyd::decompress_into(&comp_seq, &mut dst).unwrap();
    }
    let ver_1c_gb = (len as f64 * iters as f64 / (1024.0 * 1024.0 * 1024.0)) / start.elapsed().as_secs_f64();

    let start = Instant::now();
    for _ in 0..iters {
        glyd::decompress_parallel_into_raw(&comp_par, &mut dst).unwrap();
    }
    let raw_par_gb = (len as f64 * iters as f64 / (1024.0 * 1024.0 * 1024.0)) / start.elapsed().as_secs_f64();

    let start = Instant::now();
    for _ in 0..iters {
        glyd::decompress_parallel_into(&comp_par, &mut dst).unwrap();
    }
    let ver_par_gb = (len as f64 * iters as f64 / (1024.0 * 1024.0 * 1024.0)) / start.elapsed().as_secs_f64();

    println!("  1-Core Decompress (Raw):  {:>7.2} GB/s", raw_1c_gb);
    println!("  1-Core Decompress (Ver):  {:>7.2} GB/s", ver_1c_gb);
    println!("  Multi-Core Decomp (Raw):  {:>7.2} GB/s", raw_par_gb);
    println!("  Multi-Core Decomp (Ver):  {:>7.2} GB/s", ver_par_gb);
}

#[cfg(test)]
mod tests {
    use super::{is_engine_command, model_arg, route, Engine, Facts, Route};
    use std::path::PathBuf;

    /// A computer with nothing installed, which a test adds to.
    fn bare() -> Facts {
        Facts { forwarded: false, python_tool: false, gpu_tool_beside: false, choice: Engine::Auto, asked: false, external: None, is_file: false }
    }

    fn engine_program() -> Option<PathBuf> {
        Some(PathBuf::from("/opt/glyd/glyd-engine"))
    }

    #[test]
    fn files_and_compression_options_are_the_codecs() {
        let f = bare();
        for cmd in ["input.tar", "-d", "--help", "-b", "events.json"] {
            assert_eq!(route(cmd, &f), Route::Codec, "{cmd}");
        }
    }

    #[test]
    fn pack_and_verify_are_the_gpu_tools_unless_this_is_a_rust_only_install() {
        let mut f = bare();
        assert_eq!(route("pack", &f), Route::GpuTool);
        assert_eq!(route("verify", &f), if super::LINKED { Route::Engine } else { Route::GpuTool }); // (no engine, no glyd-gpu: the packer's answer says what is missing)
        f.gpu_tool_beside = true;
        f.external = engine_program();
        assert_eq!(route("verify", &f), Route::GpuTool); // (the full install has glyd-gpu beside)
        f.gpu_tool_beside = false;
        assert_eq!(route("verify", &f), Route::EngineProgram(PathBuf::from("/opt/glyd/glyd-engine")).or_linked()); // (the Rust-only install: `verify MODEL` is the engine's)
        f.python_tool = true;
        assert_eq!(route("verify", &f), Route::GpuTool); // (the tool's, through glyd-gpu)
    }

    impl Route {
        /// The route to the linked engine where there is one, else this one.
        fn or_linked(self) -> Route {
            if super::LINKED {
                Route::Engine
            } else {
                self
            }
        }
    }

    #[test]
    fn the_python_tool_takes_run_and_serve_and_is_told_which_engine_was_found() {
        let mut f = bare();
        f.python_tool = true;
        assert_eq!(route("run", &f), Route::Python(vec![]));
        f.external = engine_program();
        assert_eq!(route("run", &f), Route::Python(vec![("GLYD_ENGINE", "/opt/glyd/glyd-engine".into())]));
        assert_eq!(route("serve", &f), Route::Python(vec![])); // (serve is vLLM's unless the engine is asked for)
        f.asked = true;
        assert_eq!(route("serve", &f), Route::Python(vec![("GLYD_ENGINE", "/opt/glyd/glyd-engine".into()), ("GLYD_ENGINE_ASKED", "1".into())]));
        f.choice = Engine::Off;
        assert_eq!(route("run", &f), Route::Python(vec![])); // (GLYD_ENGINE=0: vLLM)
    }

    #[test]
    fn a_command_the_tool_sent_back_is_not_sent_again() {
        let mut f = bare();
        f.python_tool = true;
        f.forwarded = true;
        for cmd in ["run", "serve", "doctor", "login", "fetch"] {
            assert_eq!(route(cmd, &f), Route::Python(vec![]), "{cmd}"); // (python_command refuses it: it would be a loop)
        }
    }

    #[test]
    fn with_no_tool_the_engine_runs_the_model_commands() {
        let mut f = bare();
        // no engine anywhere: the answer is python_command's (what to install)
        assert_eq!(route("run", &f), if super::LINKED { Route::Engine } else { Route::Python(vec![]) });
        f.external = engine_program();
        let want = if super::LINKED { Route::Engine } else { Route::EngineProgram(PathBuf::from("/opt/glyd/glyd-engine")) };
        for cmd in ["run", "serve", "doctor", "login", "fetch", "preflight", "supports", "costs"] {
            assert_eq!(route(cmd, &f), want, "{cmd}");
        }
        f.choice = Engine::Off;
        assert_eq!(route("run", &f), Route::Python(vec![])); // (GLYD_ENGINE=0 and no tool: python_command says so)
    }

    #[test]
    fn a_program_the_user_names_runs_the_command_linked_or_not() {
        let mut f = bare();
        f.choice = Engine::Path(PathBuf::from("/x/my-engine"));
        f.external = Some(PathBuf::from("/x/my-engine"));
        assert_eq!(route("run", &f), Route::EngineProgram(PathBuf::from("/x/my-engine")));
        f.external = None; // (not a file)
        assert_eq!(route("run", &f), Route::Python(vec![]));
    }

    #[test]
    fn fit_pull_and_rangeworker_are_the_engines_with_or_without_the_tool() {
        for cmd in ["fit", "pull", "rangeworker"] {
            assert!(is_engine_command(cmd));
            let mut f = bare();
            f.python_tool = true;
            f.external = engine_program();
            let want = if super::LINKED { Route::Engine } else { Route::EngineProgram(PathBuf::from("/opt/glyd/glyd-engine")) };
            assert_eq!(route(cmd, &f), want, "{cmd}");
            // with no engine: a file of that name is compressed, else the answer says the engine is not here
            let mut f = bare();
            let nowhere = if super::LINKED { Route::Engine } else { Route::NoEngine };
            assert_eq!(route(cmd, &f), nowhere, "{cmd}");
            f.is_file = true;
            assert_eq!(route(cmd, &f), if super::LINKED { Route::Engine } else { Route::Codec }, "{cmd}");
            f.choice = Engine::Off;
            assert_eq!(route(cmd, &f), if f.is_file { Route::Codec } else { Route::NoEngine }, "{cmd}");
        }
    }

    fn a(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn the_model_of_a_run() {
        assert_eq!(model_arg(&a(&["run", "Qwen/Qwen3-8B"])), Some("Qwen/Qwen3-8B"));
        assert_eq!(model_arg(&a(&["run", "--context", "4096", "Qwen/Qwen3-8B", "--prompt", "hi"])), Some("Qwen/Qwen3-8B"));
        assert_eq!(model_arg(&a(&["run", "-p", "hi", "--no-think", "m"])), Some("m"));
        assert_eq!(model_arg(&a(&["run", "--no-think", "m", "--", "--max-model-len", "2048"])), Some("m"));
        assert_eq!(model_arg(&a(&["run", "--", "x"])), None);
        assert_eq!(model_arg(&a(&["doctor"])), None);
    }
}
