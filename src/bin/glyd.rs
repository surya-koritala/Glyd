use std::env;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

fn print_usage() {
    eprintln!(
        r#"Glyd: compression for cloud storage
Usage: glyd [OPTIONS] [INPUT] [-o OUTPUT]
       glyd pack MODEL OUT | glyd verify PATH   (a model's weights for Glyd's GPU layouts:
       packed and checked on the CPU by the glyd-gpu command; glyd pack --help)
       (the store, which compresses across objects, is the glyd-store command)
       glyd run MODEL | serve MODEL | fetch MODEL | verify MODEL | doctor | login   (the local chat, on Linux with an NVIDIA GPU: the
       Python tool's commands, which this program passes to it; glyd run uses Glyd's own engine, the
       glyd-engine command, on the GPUs where it is measured at least as fast as vLLM; GLYD_ENGINE=1
       uses it on any GPU it runs on, GLYD_ENGINE=0 keeps vLLM; a file named run is ./run)

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

/// `glyd run`, `glyd serve`, `glyd doctor` and `glyd login`. Where there is a Python tool, `run` and `serve` are its: for `run` it asks the engine (the program GLYD_ENGINE names, else the one this program
/// found beside it or on PATH, else the one of the tool's own package) whether it is used on this GPU (where it is measured at least as fast as vLLM) and will certainly run the model, before anything is
/// downloaded, runs the engine as its child where it will, and runs vLLM, with one line saying why, where it will not and where the engine refuses before it is up; `serve` is vLLM's unless GLYD_ENGINE asks for
/// the engine (the engine serves up to 8 users at once). This program says nothing of its own: it tells the tool which engine it found (GLYD_ENGINE, where the user named none, and for `serve` where
/// the user asked for the engine), and the tool, which has vLLM to fall back to, does the rest. Where there is no tool there is no other runner: an engine program found here runs the
/// command (an API key or a token given as a flag goes to it in its environment, not on its command line), and its refusals are the last word. `doctor` and `login` are the engine's where one is found here, else the tool's.
fn local_chat(args: &[String]) -> ! {
    let forwarded = env::var_os("GLYD_FORWARDED").map_or(false, |v| !v.is_empty());
    let (cmd, choice) = (args[0].as_str(), engine_choice());
    if forwarded {
        python_command(args, &[]); // (it says what this program would say: the tool has the engine's choice to make, and no loop to start)
    }
    if is_model_command(cmd) && find_python_tool().is_some() {
        python_command(args, &[]); // (the tool knows the models vLLM runs too)
    }
    if matches!(cmd, "run" | "serve") && find_python_tool().is_some() {
        let found = if choice == Engine::Auto && (cmd == "run" || engine_asked_for()) { find_engine(&choice) } else { None }; // (a path in GLYD_ENGINE asks the tool for the engine on `serve`: it is only given where it was asked for)
        match found.as_ref().and_then(|p| p.to_str()) {
            Some(path) if engine_asked_for() => python_command(args, &[("GLYD_ENGINE", path), ("GLYD_ENGINE_ASKED", "1")]), // (the user's 1, which the path takes the place of)
            Some(path) => python_command(args, &[("GLYD_ENGINE", path)]),
            None => python_command(args, &[]), // (GLYD_ENGINE=0, or a program the user named, which the tool is told by the environment; or no engine here: the tool's own package has one, or vLLM runs it)
        }
    }
    let engine = if choice == Engine::Off { None } else { find_engine(&choice) };
    let Some(engine) = engine else { python_command(args, &[]) };
    let model = if matches!(cmd, "run" | "serve") && !args.iter().any(|a| a == "-h" || a == "--help") { model_arg(args) } else { None };
    if let Some(model) = model {
        eprintln!("glyd {cmd}: {model} on glyd-engine (GLYD_ENGINE=0: vLLM instead).");
    }
    let mut c = std::process::Command::new(&engine);
    // an API key or a token given as a flag goes to the engine in its environment, not on its command line (where `ps` and a log show it)
    let mut run_args = args.to_vec();
    for (names, var, what) in
        [(&["--api-key", "--api_key"][..], "GLYD_API_KEY", "API key"), (&["--hf-token", "--hf_token"][..], "HF_TOKEN", "Hugging Face token")]
    {
        if let Some(v) = take_secret(&mut run_args, names) {
            c.env(var, v);
            eprintln!("glyd {cmd}: the {what} goes to the engine in {var}, not on its command line (next time set {var} yourself: a shell's history keeps what you type).");
        }
    }
    c.args(&run_args);
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

fn main() -> io::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() == 1 {
        print_usage();
        return Ok(());
    }
    if args[1] == "verify" && !gpu_tool_beside() && find_python_tool().is_none() && find_engine(&engine_choice()).is_some() {
        local_chat(&args[1..]); // (the Rust-only install: no glyd-gpu beside this program, so `glyd verify MODEL` is the engine's)
    }
    if matches!(args[1].as_str(), "pack" | "verify") {
        gpu_command(&args[1..]); // (a file named pack or verify: ./pack)
    }
    if matches!(args[1].as_str(), "run" | "serve" | "doctor" | "login") {
        local_chat(&args[1..]); // (a file named run: ./run)
    }
    if is_model_command(&args[1]) && (find_python_tool().is_some() || find_engine(&engine_choice()).is_some()) {
        local_chat(&args[1..]); // (a model's command: the Python tool's where there is one, else the engine's, as in the Rust-only install)
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
    use super::model_arg;

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
