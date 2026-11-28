//! GLY-362: one command per machine.
//!
//! `starlings up` runs a coordinator, a client and the model's first stage here, and the second
//! stage here too (`--machines 1`, the default) or on another machine (`--machines 2`), which
//! runs `starlings join CODE`. Every step is this binary's own command, as an operator would run
//! it. Two codes cross between the machines by hand: the join code (this cluster's one-time
//! invitation, the model's context and this GPU's free memory) and the approval code (the other
//! machine's node, fingerprint, address, engine, model and free memory). From the same numbers
//! both sides plan the same split. Neither machine runs a model before its operator confirms.
use base64::Engine as _;
use starlings_adapters::partial::glyd;
use starlings_cli::{
    routes::{PipelineRoute, PipelineStage, StageRoute},
    split::{self, Shape},
};
use starlings_protocol::{Digest, Version};
use starlings_security::{Invitation, PrivateDir};
use std::{
    collections::BTreeMap,
    ffi::OsString,
    fs,
    io::{BufRead, Write},
    net::{IpAddr, Ipv4Addr, SocketAddr},
    path::{Path, PathBuf},
    process::{Child, Command},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::{Duration, Instant},
};

type Result<T> = std::result::Result<T, String>;

const USAGE: &str = "usage: starlings up --model DIR|HUB_NAME [--engine FILE] [--gpu-lib FILE] \
                     [--machines 1|2] [--host IP] [--ctx N] [--sessions 1|2] [--drafts N] [--yes]\n\
                     \x20      starlings join --model DIR|HUB_NAME [--engine FILE] [--gpu-lib FILE] \
                     [--host IP] [--yes]";
// ponytail: fixed ports; flags when two of these run on one machine.
const REGISTRY: u16 = 9741;
const CHAT: &str = "127.0.0.1:9750";
const UI: &str = "127.0.0.1:9751";
const STAGE_PORTS: [u16; 2] = [9760, 9770];
const CHUNK: u32 = 32;

struct Options {
    model: PathBuf,
    engine: PathBuf,
    gpu_lib: Option<PathBuf>,
    machines: u32,
    host: Option<IpAddr>,
    context: u32,
    sessions: u32,
    drafts: u32,
    yes: bool,
    model_files: Digest,
}
impl Options {
    fn parse(args: &[OsString]) -> Result<Self> {
        let mut options = Self {
            model: PathBuf::new(),
            engine: PathBuf::new(),
            gpu_lib: None,
            machines: 1,
            host: None,
            context: 4096,
            sessions: 1,
            drafts: 7,
            yes: false,
            model_files: starlings_protocol::byte_digest(b"not resolved"),
        };
        let mut args = args.iter();
        while let Some(flag) = args.next() {
            if flag == "--yes" {
                options.yes = true;
                continue;
            }
            let value = args.next().ok_or(USAGE)?;
            let number = || value.to_str().and_then(|v| v.parse().ok()).ok_or(USAGE);
            let path = || fs::canonicalize(value).map_err(|e| format!("{}: {e}", value.display()));
            match flag.to_str().ok_or(USAGE)? {
                // A directory, or a Hub name the engine fetches (below).
                "--model" => options.model = PathBuf::from(value),
                "--engine" => options.engine = path()?,
                "--gpu-lib" => options.gpu_lib = Some(path()?),
                "--machines" => options.machines = number()?,
                "--host" => {
                    options.host = Some(value.to_str().and_then(|v| v.parse().ok()).ok_or(USAGE)?)
                }
                "--ctx" => options.context = number()?,
                "--sessions" => options.sessions = number()?,
                "--drafts" => options.drafts = number()?,
                _ => return Err(USAGE.into()),
            }
        }
        if options.model.as_os_str().is_empty() || !(1..=2).contains(&options.machines) {
            return Err(USAGE.into());
        }
        options.validate_workload()?;
        // The engine and its GPU library default to the ones installed beside this binary.
        let beside = |name: &str| {
            std::env::current_exe()
                .ok()
                .and_then(|e| e.parent().map(|d| d.join(name)))
                .filter(|p| p.exists())
        };
        if options.engine.as_os_str().is_empty() {
            options.engine = beside("glyd-engine")
                .ok_or("no --engine, and no glyd-engine beside starlings")?
                .canonicalize()
                .map_err(|e| e.to_string())?;
        }
        if options.gpu_lib.is_none() {
            options.gpu_lib = beside("libglyd_gpu_cuda13.so");
        }
        options.model = model_dir(&options.model, &options.engine, options.yes)?;
        options.model_files = starlings_cli::model_files::digest(&options.model)?;
        Ok(options)
    }
    fn validate_workload(&self) -> Result<()> {
        if !(1..=2).contains(&self.sessions)
            || self.context < CHUNK + self.max_output()
            || self.drafts >= starlings_protocol::MAX_DECODE_ROWS.min(CHUNK)
        {
            return Err("invalid context, sessions or drafts for this pipeline".into());
        }
        Ok(())
    }
    fn max_output(&self) -> u32 {
        1024.min(self.context / 4)
    }
}

/// A model directory (absolute, or starting with `./` or `../`), or a Hub name
/// (`glyd/Qwen3.8-27B-linnet`) that the engine fetches into the Hugging Face cache once the operator
/// agrees. A directory in the current directory never stands in for a Hub name (GLY-366).
fn model_dir(model: &Path, engine: &Path, yes: bool) -> Result<PathBuf> {
    let text = model.to_string_lossy();
    let name = Some(text.as_ref()).filter(|n| {
        let parts: Vec<_> = n.split('/').collect();
        !model.is_absolute()
            && parts.len() == 2
            && parts.iter().all(|p| !p.is_empty() && !p.starts_with('.'))
    });
    let Some(name) = name else {
        return model
            .canonicalize()
            .map_err(|e| format!("{}: {e}", model.display()));
    };
    let hub = std::env::var_os("HF_HUB_CACHE")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HF_HOME").map(|h| PathBuf::from(h).join("hub")))
        .or_else(|| dirs::home_dir().map(|h| h.join(".cache/huggingface/hub")))
        .ok_or("no Hugging Face cache directory")?;
    let repo = hub.join(format!("models--{}", name.replace('/', "--")));
    if !repo.join("refs/main").exists() {
        let digest =
            starlings_adapters::service::executable_digest(engine).map_err(|e| format!("{e:?}"))?;
        confirm(
            &format!(
                "Download {name} with {} (sha256 {}...)?",
                engine.display(),
                &digest.as_str()[..12]
            ),
            yes,
        )?;
        let fetched = Command::new(engine)
            .args(["fetch", name])
            .status()
            .map_err(|e| format!("{}: {e}", engine.display()))?;
        if !fetched.success() {
            return Err(format!("could not fetch {name}"));
        }
    }
    let revision = fs::read_to_string(repo.join("refs/main"))
        .map_err(|e| format!("{}: {e}", repo.display()))?;
    let revision = revision.trim();
    if revision.len() != 40 || !revision.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err("invalid cached model revision".into());
    }
    repo.join("snapshots")
        .join(revision)
        .canonicalize()
        .map_err(|e| e.to_string())
}

/// From the first machine to the second: everything the second needs to plan its stage.
#[derive(serde::Serialize, serde::Deserialize)]
struct JoinCode {
    invitation: serde_json::Value,
    free: u64,
    context: u32,
    sessions: u32,
    drafts: u32,
    manifest: Digest,
}
/// From the second machine back: what the first approves and plans with.
#[derive(serde::Serialize, serde::Deserialize, Clone)]
struct ApprovalCode {
    node: String,
    fingerprint: String,
    host: IpAddr,
    engine: Digest,
    manifest: Digest,
    free: u64,
}
/// Byte arrays (as the invitation record writes them) become base64 strings, and back, so a join
/// code fits one pasted terminal line (at most 4095 bytes there).
fn pack(value: serde_json::Value) -> serde_json::Value {
    use serde_json::Value;
    match value {
        Value::Array(items)
            if items.len() > 8 && items.iter().all(|i| i.as_u64().is_some_and(|n| n < 256)) =>
        {
            let bytes: Vec<u8> = items
                .iter()
                .filter_map(|i| i.as_u64())
                .map(|n| n as u8)
                .collect();
            serde_json::json!({"$b64": base64::engine::general_purpose::STANDARD_NO_PAD.encode(bytes)})
        }
        Value::Array(items) => Value::Array(items.into_iter().map(pack).collect()),
        Value::Object(map) => Value::Object(map.into_iter().map(|(k, v)| (k, pack(v))).collect()),
        other => other,
    }
}
fn unpack(value: serde_json::Value) -> Result<serde_json::Value> {
    use serde_json::Value;
    match value {
        Value::Object(map) if map.len() == 1 && map.contains_key("$b64") => {
            let bytes = base64::engine::general_purpose::STANDARD_NO_PAD
                .decode(
                    map["$b64"]
                        .as_str()
                        .ok_or("not a code from the other machine")?,
                )
                .map_err(|_| "not a code from the other machine")?;
            Ok(Value::Array(bytes.into_iter().map(Value::from).collect()))
        }
        Value::Array(items) => items
            .into_iter()
            .map(unpack)
            .collect::<Result<Vec<_>>>()
            .map(Value::Array),
        Value::Object(map) => map
            .into_iter()
            .map(|(k, v)| unpack(v).map(|v| (k, v)))
            .collect::<Result<serde_json::Map<_, _>>>()
            .map(Value::Object),
        other => Ok(other),
    }
}
/// A value both operators compare by eye: 16 hex digits of the text's SHA-256.
fn check(text: &str) -> String {
    let digest = starlings_protocol::byte_digest(text.as_bytes());
    let h = &digest.as_str()[..16];
    format!("{}-{}-{}-{}", &h[..4], &h[4..8], &h[8..12], &h[12..])
}
/// The cluster's check, from the CA certificate its invitation pins.
fn cluster_check(invitation: &serde_json::Value) -> String {
    check(&invitation["ca"].to_string())
}
/// What runs here: the engine and the GPU library (path and digest), and the model.
fn summary(o: &Options, layers: u32) -> Result<()> {
    let digest = |p: &Path| {
        starlings_adapters::service::executable_digest(p)
            .map(|d| d.as_str()[..12].to_owned())
            .map_err(|e| format!("{e:?}"))
    };
    println!(
        "  engine   {} (sha256 {}...)",
        o.engine.display(),
        digest(&o.engine)?
    );
    if let Some(lib) = &o.gpu_lib {
        println!("  GPU lib  {} (sha256 {}...)", lib.display(), digest(lib)?);
    }
    println!("  model    {} ({layers} layers)", o.model.display());
    Ok(())
}
fn read_line(prompt: &str) -> Result<String> {
    print!("{prompt}");
    std::io::stdout().flush().map_err(|e| e.to_string())?;
    let mut line = String::new();
    std::io::stdin()
        .lock()
        .read_line(&mut line)
        .map_err(|e| e.to_string())?;
    Ok(line.trim().to_owned())
}
fn encode(value: &impl serde::Serialize) -> Result<String> {
    let json = serde_json::to_vec(value).map_err(|e| e.to_string())?;
    Ok(base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(json))
}
fn decode<T: serde::de::DeserializeOwned>(code: &str) -> Result<T> {
    let json = base64::engine::general_purpose::URL_SAFE_NO_PAD
        .decode(code.trim())
        .map_err(|_| "not a code from the other machine")?;
    serde_json::from_slice(&json).map_err(|_| "not a code from the other machine".into())
}

/// This binary's own command, as the operator would type it; its output, or why it failed.
fn cli(state: &Path, args: &[&str]) -> Result<String> {
    // Below the daemon's eight admin requests a second.
    thread::sleep(Duration::from_millis(220));
    let out = Command::new(std::env::current_exe().map_err(|e| e.to_string())?)
        .args(args)
        .arg("--state-dir")
        .arg(state)
        .output()
        .map_err(|e| e.to_string())?;
    let text =
        String::from_utf8_lossy(&out.stdout).into_owned() + &String::from_utf8_lossy(&out.stderr);
    if out.status.success() {
        Ok(text)
    } else {
        Err(format!("starlings {}: {}", args.join(" "), text.trim()))
    }
}
fn field(text: &str, name: &str) -> Result<String> {
    text.split_whitespace()
        .find_map(|w| w.strip_prefix(name))
        .map(|v| v.trim_end_matches(';').to_owned())
        .ok_or(format!("no {name} in: {text}"))
}
fn private(path: &Path) -> Result<PrivateDir> {
    PrivateDir::open(path).map_err(|e| format!("{}: {e:?}", path.display()))
}
fn write(dir: &Path, name: &str, value: &impl serde::Serialize) -> Result<PathBuf> {
    let bytes = serde_json::to_vec_pretty(value).map_err(|e| e.to_string())?;
    private(dir)?
        .write_atomic_durable(name, &bytes)
        .map_err(|e| format!("{name}: {e:?}"))?;
    Ok(dir.join(name))
}
fn read_json(path: &Path) -> serde_json::Value {
    fs::read(path)
        .ok()
        .and_then(|b| serde_json::from_slice(&b).ok())
        .unwrap_or_default()
}
fn confirm(question: &str, yes: bool) -> Result<()> {
    if yes {
        return Ok(());
    }
    print!("{question} [y/N] ");
    std::io::stdout().flush().map_err(|e| e.to_string())?;
    let mut answer = String::new();
    std::io::stdin()
        .lock()
        .read_line(&mut answer)
        .map_err(|e| e.to_string())?;
    if matches!(answer.trim(), "y" | "Y" | "yes") {
        Ok(())
    } else {
        Err("not started".into())
    }
}
fn gib(bytes: u64) -> String {
    format!("{:.1} GiB", bytes as f64 / f64::from(1u32 << 30))
}
/// A Ctrl-C (or SIGTERM) from here on stops every daemon this started.
fn stop_flag() -> Result<Arc<AtomicBool>> {
    let stop = Arc::new(AtomicBool::new(false));
    let flag = stop.clone();
    ctrlc::set_handler(move || flag.store(true, Ordering::Relaxed)).map_err(|e| e.to_string())?;
    Ok(stop)
}
/// The address this machine reaches `peer` from (no packet is sent).
fn address_toward(peer: SocketAddr) -> Result<IpAddr> {
    let any: SocketAddr = if peer.is_ipv4() {
        (Ipv4Addr::UNSPECIFIED, 0).into()
    } else {
        "[::]:0".parse().map_err(|_| "address")?
    };
    let socket = std::net::UdpSocket::bind(any).map_err(|e| e.to_string())?;
    socket.connect(peer).map_err(|e| e.to_string())?;
    let ip = socket.local_addr().map_err(|e| e.to_string())?.ip();
    if !starlings_security::network::local_address((ip, 1).into(), false) {
        return Err(format!(
            "{ip} is not a LAN address; pass --host with this machine's LAN address"
        ));
    }
    Ok(ip)
}

/// Daemons started here, each with its launch arguments; any still running are stopped when
/// this ends.
struct Daemons(Vec<(PathBuf, Vec<String>, Child)>);
impl Daemons {
    fn start(&mut self, state: &Path, args: &[&str]) -> Result<()> {
        let mut log = fs::File::options();
        // The coordinator's log carries the UI's secret link.
        #[cfg(unix)]
        std::os::unix::fs::OpenOptionsExt::mode(&mut log, 0o600);
        let log = log
            .create(true)
            .append(true)
            .open(state.join("daemon.log"))
            .map_err(|e| e.to_string())?;
        let mut child = Command::new(std::env::current_exe().map_err(|e| e.to_string())?)
            .arg("daemon")
            .args(args)
            .arg("--state-dir")
            .arg(state)
            .stdout(log.try_clone().map_err(|e| e.to_string())?)
            .stderr(log)
            .spawn()
            .map_err(|e| e.to_string())?;
        let deadline = Instant::now() + Duration::from_secs(60);
        // Ready once its admin channel answers.
        while !(state.join("admin.cap").exists() && cli(state, &["cluster", "status"]).is_ok()) {
            if child.try_wait().map_err(|e| e.to_string())?.is_some() || Instant::now() > deadline {
                let _ = child.kill();
                return Err(format!(
                    "daemon in {} did not start; see its daemon.log",
                    state.display()
                ));
            }
            thread::sleep(Duration::from_millis(100));
        }
        let args = args.iter().map(|a| a.to_string()).collect();
        self.0.push((state.into(), args, child));
        Ok(())
    }
    fn stop_one(state: &Path, mut child: Child) {
        if child.try_wait().ok().flatten().is_none() {
            let _ = cli(state, &["cluster", "stop"]);
            let deadline = Instant::now() + Duration::from_secs(20);
            while child.try_wait().ok().flatten().is_none() && Instant::now() < deadline {
                thread::sleep(Duration::from_millis(100));
            }
            let _ = child.kill();
            let _ = child.wait();
        }
    }
    fn stop(&mut self) {
        for (state, _, child) in self.0.drain(..).rev() {
            Self::stop_one(&state, child);
        }
    }
    /// Until a stop, or until a daemon exits on its own (an error). A stage whose engine died
    /// reports failed and takes nothing more; it is restarted from the same approved launch,
    /// at most three times in ten minutes.
    fn supervise(&mut self, stop: &AtomicBool) -> Result<()> {
        let (mut restarts, mut checked) = (Vec::<Instant>::new(), Instant::now());
        while !stop.load(Ordering::Relaxed) {
            for (dir, _, child) in &mut self.0 {
                if child.try_wait().map_err(|e| e.to_string())?.is_some() {
                    return Err(format!(
                        "the daemon in {} stopped; see its daemon.log",
                        dir.display()
                    ));
                }
            }
            if checked.elapsed() >= Duration::from_secs(2) {
                checked = Instant::now();
                let failed = self
                    .0
                    .iter()
                    .position(|(dir, args, _)| args[0] == "--stage" && stage_ready(dir).is_err());
                if let Some(i) = failed {
                    restarts.retain(|t| t.elapsed() < Duration::from_secs(600));
                    if restarts.len() >= 3 {
                        return Err("a stage keeps failing; see its daemon.log".into());
                    }
                    restarts.push(Instant::now());
                    let (dir, args, child) = self.0.remove(i);
                    println!("The stage in {} failed; restarting it.", dir.display());
                    Self::stop_one(&dir, child);
                    let args: Vec<&str> = args.iter().map(String::as_str).collect();
                    self.start(&dir, &args)?;
                }
            }
            thread::sleep(Duration::from_millis(200));
        }
        println!("Stopping.");
        Ok(())
    }
}
impl Drop for Daemons {
    fn drop(&mut self) {
        self.stop();
    }
}

/// Free bytes of each NVIDIA GPU, in index order. The engine needs a driver for CUDA 13 (580 or
/// newer); an older one is named here rather than failing later at load.
fn gpus() -> Result<Vec<u64>> {
    let out = Command::new("nvidia-smi")
        .args([
            "--query-gpu=memory.free,driver_version",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .map_err(|e| format!("nvidia-smi: {e} (an NVIDIA driver is needed)"))?;
    let mut free = Vec::new();
    for line in String::from_utf8_lossy(&out.stdout).lines() {
        let Some((mib, driver)) = line.split_once(',') else {
            continue;
        };
        let major: u32 = driver
            .trim()
            .split('.')
            .next()
            .and_then(|m| m.parse().ok())
            .unwrap_or(0);
        if major < 580 {
            return Err(format!(
                "NVIDIA driver {} is too old: the engine needs 580 or newer (CUDA 13)",
                driver.trim()
            ));
        }
        free.push(
            mib.trim()
                .parse::<u64>()
                .map_err(|_| "nvidia-smi: unreadable output")?
                << 20,
        );
    }
    if free.is_empty() {
        return Err("no NVIDIA GPU found (nvidia-smi)".into());
    }
    Ok(free)
}

/// The model's end-of-text and end-of-turn IDs end an answer.
fn stop_tokens(model: &Path) -> Vec<u32> {
    let ends = |content: &serde_json::Value| {
        ["<|endoftext|>", "<|im_end|>"].contains(&content.as_str().unwrap_or(""))
    };
    let mut ids: Vec<u32> = read_json(&model.join("tokenizer.json"))["added_tokens"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|t| ends(&t["content"]))
        .filter_map(|t| t["id"].as_u64())
        .map(|id| id as u32)
        .collect();
    let config = read_json(&model.join("config.json"));
    if let Some(eos) = config
        .get("text_config")
        .unwrap_or(&config)
        .get("eos_token_id")
        .and_then(|e| e.as_u64())
    {
        ids.push(eos as u32);
    }
    ids.sort_unstable();
    ids.dedup();
    ids
}

/// What the plan reserves for a stage of these layers.
fn stage_bytes(shape: &Shape, range: [u32; 2]) -> u64 {
    let [start, end] = range.map(|n| n as usize);
    let mut bytes: u64 = shape.layers[start..end].iter().sum();
    if start == 0 {
        bytes += shape.first;
    }
    if end == shape.layers.len() {
        bytes += shape.last;
    }
    bytes
}

/// One stage on this machine, described by the engine itself (which loads nothing): its route
/// and the model's manifest digest.
fn local_stage(
    o: &Options,
    shape: &Shape,
    range: [u32; 2],
    (control, host, port, gpu): (IpAddr, IpAddr, u16, usize),
    describe_dir: &Path,
) -> Result<(StageRoute, Digest)> {
    o.validate_workload()?;
    if range[0] >= range[1] || range[1] as usize > shape.layers.len() {
        return Err("stage layers are outside this model".into());
    }
    let mut environment = BTreeMap::from([("CUDA_VISIBLE_DEVICES".to_owned(), gpu.to_string())]);
    if let Some(lib) = &o.gpu_lib {
        environment.insert("GLYD_GPU_LIB".into(), lib.to_string_lossy().into_owned());
    }
    let digest = |path: &Path| {
        starlings_adapters::service::executable_digest(path).map_err(|e| format!("{e:?}"))
    };
    let at = |p: u16| SocketAddr::from((host, p));
    let last = range[1] as usize == shape.layers.len();
    let mut stage = StageRoute {
        version: Version,
        listen: (control, port).into(),
        data_listen: at(port + 1),
        executable: o.engine.clone(),
        executable_sha256: digest(&o.engine)?,
        model: o.model.clone(),
        model_files_sha256: o.model_files.clone(),
        layers: range,
        context_tokens: o.context,
        chunk_tokens: CHUNK,
        max_prompt_tokens: o.context - o.max_output(),
        max_output_tokens: o.max_output(),
        descriptor: starlings_protocol::byte_digest(b"set by describe"),
        execution_profile: "glyd-range".into(),
        boundary_profile: "qwen35-residual-mlp-bf16".into(),
        backend_id: "glyd-rangeworker".into(),
        device_id: "cuda:0".into(),
        // The core's bookkeeping limit: each request's reservation (the plan's), per session.
        engine_bytes_limit: starlings_protocol::Decimal(
            u64::from(o.sessions) * (stage_bytes(shape, range) + split::RESERVE),
        ),
        environment,
        gpu_library_sha256: o.gpu_lib.as_deref().map(digest).transpose()?,
        head_rows: if last { (o.drafts + 1).min(CHUNK) } else { 0 },
        sessions: o.sessions,
        more_data_listen: (1..o.sessions as u16).map(|s| at(port + 1 + s)).collect(),
    };
    let (process, mut stdout, mut stdin) = glyd::RangeProcess::spawn(
        &stage.executable,
        &stage.executable_sha256,
        &stage.arguments(),
        &stage.environment,
        stage.gpu_library_sha256.as_ref(),
        describe_dir,
    )
    .map_err(|e| format!("engine launch: {e:?}"))?;
    let described = glyd::describe(&mut stdout, &mut stdin);
    process.stop();
    let described = described.map_err(|e| format!("engine describe: {e:?}"))?;
    if described["model"]["layers"].as_u64() != Some(shape.layers.len() as u64) {
        return Err("the engine reads a different layer count than config.json".into());
    }
    stage.descriptor = glyd::descriptor_digest(&described).map_err(|e| format!("{e:?}"))?;
    let manifest = o.model_files.clone();
    let check = starlings_protocol::content_hash(&stage).map_err(|e| format!("{e:?}"))?;
    stage
        .validate(&check)
        .map_err(|e| format!("stage refused: {e:?}"))?;
    Ok((stage, manifest))
}

fn launch<T: serde::Serialize>(
    daemons: &mut Daemons,
    dir: &Path,
    flag: &str,
    record: &T,
    validate: impl FnOnce(&Digest) -> std::result::Result<(), starlings_security::Error>,
    extra: &[&str],
) -> Result<()> {
    let digest = starlings_protocol::content_hash(record).map_err(|e| format!("{e:?}"))?;
    validate(&digest).map_err(|e| format!("{flag} record refused: {e:?}"))?;
    let file = write(dir, "launch.json", record)?;
    let file = file.to_string_lossy();
    let args: Vec<&str> = [flag, &file, "--approve", digest.as_str()]
        .into_iter()
        .chain(extra.iter().copied())
        .collect();
    daemons.start(dir, &args)
}

/// Waits, a step at a time, until `ready` says so; `what` names it in the error.
fn wait(
    stop: &AtomicBool,
    limit: Duration,
    what: &str,
    mut ready: impl FnMut() -> bool,
) -> Result<()> {
    let deadline = Instant::now() + limit;
    while !ready() {
        if stop.load(Ordering::Relaxed) {
            return Err("stopped".into());
        }
        if Instant::now() > deadline {
            return Err(format!("{what} timed out"));
        }
        thread::sleep(Duration::from_millis(500));
    }
    Ok(())
}
fn stage_ready(dir: &Path) -> Result<bool> {
    let service = cli(dir, &["cluster", "service"]).unwrap_or_default();
    let state = service
        .strip_prefix("SERVICE: ")
        .and_then(|s| serde_json::from_str::<serde_json::Value>(s.trim()).ok())
        .map(|s| s["state"].as_str().unwrap_or_default().to_owned())
        .unwrap_or_default();
    if state == "failed" {
        return Err(format!(
            "stage in {} did not load; see its daemon.log",
            dir.display()
        ));
    }
    Ok(state == "ready")
}

/// Enrolls one of this machine's own nodes (approved by running `up` here).
fn enroll_local(coordinator: &Path, role: &str, dir: &Path) -> Result<String> {
    let invite = invite(coordinator, role, dir)?;
    let prepared = cli(dir, &["cluster", "prepare", &invite.to_string_lossy()])?;
    let node = field(&prepared, "node=")?;
    cli(
        coordinator,
        &[
            "cluster",
            "approve",
            &node,
            &field(&prepared, "fingerprint=")?,
        ],
    )?;
    cli(dir, &["cluster", "join", &invite.to_string_lossy()])?;
    Ok(node)
}
/// A one-time invitation (five minutes) for `dir`'s node, in the coordinator's directory.
fn invite(coordinator: &Path, role: &str, dir: &Path) -> Result<PathBuf> {
    let name = format!(
        "{}.json",
        dir.file_name().unwrap_or_default().to_string_lossy()
    );
    let invite = coordinator.join(&name);
    let _ = fs::remove_file(&invite);
    let invited = cli(
        coordinator,
        &["cluster", "invite", role, &invite.to_string_lossy()],
    );
    let deadline = Instant::now() + Duration::from_secs(10);
    while !invite.exists() {
        if Instant::now() > deadline {
            invited?;
            return Err("no invitation written".into());
        }
        thread::sleep(Duration::from_millis(50));
    }
    Ok(invite)
}
fn permit(coordinator: &Path, node: &str, manifest: &Digest, action: &str) -> Result<String> {
    let scopes = write(
        coordinator,
        "scopes.json",
        &serde_json::json!([{"model": manifest, "action": action}]),
    )?;
    cli(
        coordinator,
        &["cluster", "permissions", node, &scopes.to_string_lossy()],
    )
}

/// The state directory, private. On a fresh machine its parents (`~/.local/state`) may not exist
/// yet: they are made private too, since a group-writable parent is refused.
fn state_root(state: &Path) -> Result<()> {
    if let Some(parent) = state.parent().filter(|p| !p.exists()) {
        let mut dirs = fs::DirBuilder::new();
        #[cfg(unix)]
        std::os::unix::fs::DirBuilderExt::mode(&mut dirs, 0o700);
        dirs.recursive(true)
            .create(parent)
            .map_err(|e| format!("{}: {e}", parent.display()))?;
    }
    PrivateDir::ensure(state)
        .map(|_| ())
        .map_err(|e| format!("{}: {e:?}", state.display()))
}

/// Where each machine's second stage is, for the plan.
#[derive(serde::Serialize, serde::Deserialize)]
struct Remote {
    host: IpAddr,
    engine: Digest,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct SavedUp {
    machines: u32,
    cut: u32,
    nodes: Vec<String>,
    manifest: Digest,
    peer: Option<Remote>,
    context: u32,
    sessions: u32,
    drafts: u32,
    gpus: usize,
    host: IpAddr,
}
impl SavedUp {
    fn validate(&self, o: &Options, layers: u32, gpus: usize, host: IpAddr) -> Result<()> {
        if self.machines != o.machines
            || self.manifest != o.model_files
            || (self.context, self.sessions, self.drafts) != (o.context, o.sessions, o.drafts)
            || self.gpus != gpus
            || self.host != host
            || self.peer.is_some() != (o.machines == 2)
            || self.nodes.len() != 3
            || self.nodes.iter().any(String::is_empty)
            || self
                .nodes
                .iter()
                .collect::<std::collections::BTreeSet<_>>()
                .len()
                != 3
        {
            return Err(
                "saved setup does not match this model, workload or devices; use a new --state-dir"
                    .into(),
            );
        }
        validate_cut(self.cut, layers)
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
struct SavedJoin {
    cut: u32,
    host: IpAddr,
    context: u32,
    sessions: u32,
    drafts: u32,
    manifest: Digest,
}
fn validate_cut(cut: u32, layers: u32) -> Result<()> {
    if cut == 0 || cut >= layers {
        return Err("saved split is outside this model; use a new --state-dir".into());
    }
    Ok(())
}

pub fn run(state: &Path, args: &[OsString]) -> Result<()> {
    let o = Options::parse(args)?;
    state_root(state)?;
    let dirs = ["coordinator", "stage0", "stage1", "client"].map(|n| state.join(n));
    let [coordinator, stage0, stage1, client] = &dirs;
    let two = o.machines == 2;
    let host = match (two, o.host) {
        (false, _) => IpAddr::V4(Ipv4Addr::LOCALHOST),
        (true, Some(host)) => host,
        (true, None) => address_toward((Ipv4Addr::new(192, 0, 2, 1), 9).into())?,
    };
    let shape = split::read(&o.model, o.context, o.sessions)?;
    let layers = shape.layers.len() as u32;
    let free = gpus()?;
    let saved = read_json(&state.join("up.json"));
    if coordinator.exists() && saved["machines"] != o.machines {
        return Err(format!(
            "{} holds another or an unfinished setup: remove it to start over, or use a new \
             --state-dir",
            state.display()
        ));
    }
    let describe_dir = state.join("describe");

    let mut daemons = Daemons(Vec::new());
    let stop;
    // The split, the local stages, the nodes and (two machines) the other machine.
    let (cut, nodes, manifest, remote) = if !saved.is_null() {
        let saved: SavedUp = serde_json::from_value(saved)
            .map_err(|e| format!("invalid or older up.json; use a new --state-dir: {e}"))?;
        saved.validate(&o, layers, free.len().min(2), host)?;
        let SavedUp {
            cut,
            nodes,
            manifest,
            peer: remote,
            ..
        } = saved;
        println!("Starlings, the saved setup:");
        summary(&o, layers)?;
        println!(
            "  layers   0..{cut} here, {cut}..{layers} on {}",
            remote
                .as_ref()
                .map_or("this machine too".to_owned(), |r| r.host.to_string())
        );
        confirm("Start the saved setup again?", o.yes)?;
        stop = stop_flag()?;
        (cut, nodes, manifest, remote)
    } else if !two {
        let (devices, room) = match free[..] {
            [a, b, ..] => ([0, 1], [a, b]),
            [a] => ([0, 0], [a / 2, a / 2]),
            [] => unreachable!(),
        };
        let cut = split::cut(&shape, room, split::RESERVE).ok_or(format!(
            "{layers} layers at --ctx {} do not fit {} and {} free",
            o.context,
            gib(room[0]),
            gib(room[1])
        ))?;
        let (_, manifest) = local_stage(
            &o,
            &shape,
            [0, cut],
            (
                IpAddr::V4(Ipv4Addr::LOCALHOST),
                host,
                STAGE_PORTS[0],
                devices[0],
            ),
            &describe_dir,
        )?;
        println!("Starlings on this machine:");
        summary(&o, layers)?;
        for (i, range) in [[0, cut], [cut, layers]].into_iter().enumerate() {
            println!(
                "  stage {i} layers {}..{} on GPU {}: about {} of {} free",
                range[0],
                range[1],
                devices[i],
                gib(stage_bytes(&shape, range)),
                gib(free[devices[i]]),
            );
        }
        confirm("Start?", o.yes)?;
        stop = stop_flag()?;
        cli(
            coordinator,
            &["cluster", "init", "--listen", &format!("{host}:{REGISTRY}")],
        )?;
        daemons.start(coordinator, &[])?;
        let mut nodes = Vec::new();
        for (role, dir) in [("worker", stage0), ("worker", stage1), ("client", client)] {
            let node = enroll_local(coordinator, role, dir)?;
            permit(coordinator, &node, &manifest, "infer")?;
            nodes.push(node);
        }
        (cut, nodes, manifest, None)
    } else {
        // Two machines: this one first, then the other's code.
        let (_, manifest) = local_stage(
            &o,
            &shape,
            [0, 1],
            (IpAddr::V4(Ipv4Addr::LOCALHOST), host, STAGE_PORTS[0], 0),
            &describe_dir,
        )?;
        println!("Starlings on this machine, the first of two:");
        summary(&o, layers)?;
        println!(
            "  cluster  on {host}:{REGISTRY} (this LAN), layers from 0; the split is set once"
        );
        println!("           the other machine answers");
        confirm("Start the cluster?", o.yes)?;
        stop = stop_flag()?;
        cli(
            coordinator,
            &["cluster", "init", "--listen", &format!("{host}:{REGISTRY}")],
        )?;
        daemons.start(coordinator, &[])?;
        let mut nodes = Vec::new();
        for (role, dir) in [("worker", stage0), ("client", client)] {
            let node = enroll_local(coordinator, role, dir)?;
            permit(coordinator, &node, &manifest, "infer")?;
            nodes.push(node);
        }
        let invitation = invite(coordinator, "worker", stage1)?;
        let cluster = cluster_check(&read_json(&invitation));
        let code = encode(&JoinCode {
            invitation: pack(read_json(&invitation)),
            free: free[0],
            context: o.context,
            sessions: o.sessions,
            drafts: o.drafts,
            manifest: manifest.clone(),
        })?;
        let code_file = write(state, "join-code.json", &code)?;
        println!("\nOn the other machine, within five minutes, run");
        println!("  starlings join --model {}", o.model.display());
        println!(
            "and paste this join code when it asks (it is also in {}):\n",
            code_file.display()
        );
        println!("{code}\n");
        println!("It shows this cluster check; it must read the same there: {cluster}\n");
        let peer: ApprovalCode = decode(&read_line("Then paste the code it shows here: ")?)?;
        if peer.manifest != manifest {
            return Err("the other machine has a different model".into());
        }
        let cut = split::cut(&shape, [free[0], peer.free], split::RESERVE).ok_or(format!(
            "{layers} layers at --ctx {} do not fit {} here and {} there",
            o.context,
            gib(free[0]),
            gib(peer.free)
        ))?;
        println!("The other machine: node {} at {}", peer.node, peer.host);
        println!(
            "  this machine:  layers 0..{cut}, about {} of {} free",
            gib(stage_bytes(&shape, [0, cut])),
            gib(free[0])
        );
        println!(
            "  other machine: layers {cut}..{layers}, about {} of {} free",
            gib(stage_bytes(&shape, [cut, layers])),
            gib(peer.free)
        );
        // Another machine is approved by hand, even with --yes: its operator's screen shows the
        // check of its node and key, and a code swapped on the way gives another (GLY-366).
        let typed = read_line("To approve it and start, type the check it shows: ")?;
        let norm = |c: &str| c.to_ascii_lowercase().replace(['-', ' '], "");
        if norm(&typed) != norm(&check(&format!("{} {}", peer.node, peer.fingerprint))) {
            return Err("the check does not match: not approved".into());
        }
        cli(
            coordinator,
            &["cluster", "approve", &peer.node, &peer.fingerprint],
        )?;
        // Its permission takes once it has joined.
        println!("Waiting for it to join...");
        wait(
            &stop,
            Duration::from_secs(300),
            "the other machine's join",
            || permit(coordinator, &peer.node, &manifest, "infer").is_ok(),
        )?;
        nodes.insert(1, peer.node.clone());
        // The invitation is spent.
        let _ = fs::remove_file(&code_file);
        let _ = fs::remove_file(&invitation);
        let remote = Remote {
            host: peer.host,
            engine: peer.engine.clone(),
        };
        (cut, nodes, manifest, Some(remote))
    };
    if !daemons.0.is_empty() {
        let config: serde_json::Value =
            crate::cluster::read(&private(coordinator)?, "config.json")?;
        let me = config["node"]
            .as_str()
            .ok_or("coordinator config has no node")?;
        permit(coordinator, me, &manifest, "control")?;
        daemons.stop();
        write(
            state,
            "up.json",
            &serde_json::json!({"machines": o.machines, "cut": cut, "nodes": nodes,
                "manifest": manifest, "peer": remote, "context": o.context,
                "sessions": o.sessions, "drafts": o.drafts, "gpus": free.len().min(2),
                "host": host}),
        )?;
    }

    // This machine's stages, and the pipeline's plan template (filled for each request).
    let gpu1 = usize::from(free.len() > 1);
    let mut locals = vec![(
        stage0,
        local_stage(
            &o,
            &shape,
            [0, cut],
            (IpAddr::V4(Ipv4Addr::LOCALHOST), host, STAGE_PORTS[0], 0),
            &describe_dir,
        )?
        .0,
    )];
    if remote.is_none() {
        let range = [cut, layers];
        locals.push((
            stage1,
            local_stage(
                &o,
                &shape,
                range,
                (host, host, STAGE_PORTS[1], gpu1),
                &describe_dir,
            )?
            .0,
        ));
    }
    let second = match &remote {
        Some(r) => PipelineStage {
            endpoint: (r.host, STAGE_PORTS[1]).into(),
            data: (r.host, STAGE_PORTS[1] + 1).into(),
            more_data: (1..o.sessions as u16)
                .map(|s| (r.host, STAGE_PORTS[1] + 1 + s).into())
                .collect(),
        },
        None => PipelineStage {
            endpoint: locals[1].1.listen,
            data: locals[1].1.data_listen,
            more_data: locals[1].1.more_data_listen.clone(),
        },
    };
    let engines = [
        locals[0].1.executable_sha256.clone(),
        remote.as_ref().map_or_else(
            || locals[1].1.executable_sha256.clone(),
            |r| r.engine.clone(),
        ),
    ];
    let config: serde_json::Value = crate::cluster::read(&private(coordinator)?, "config.json")?;
    let at = |seconds: i64| {
        (time::OffsetDateTime::now_utc() + time::Duration::seconds(seconds))
            .format(&time::format_description::well_known::Rfc3339)
            .map_err(|e| e.to_string())
    };
    let plan_stages: Vec<_> = [[0, cut], [cut, layers]]
        .into_iter()
        .enumerate()
        .map(|(i, range)| {
            serde_json::json!({
                "index": i, "node_id": nodes[i], "device_id": "cuda:0",
                "backend_id": "glyd-rangeworker", "backend_build_digest": engines[i],
                "layer_start": range[0], "layer_end_exclusive": range[1],
                "owns_embedding": i == 0, "owns_final_norm_and_head": i == 1,
                "memory_pool_id": "cuda:0",
                "resident_bytes": stage_bytes(&shape, range).to_string(),
                "state_budget_bytes": (256u64 << 20).to_string(),
                "scratch_budget_bytes": (64u64 << 20).to_string(),
                "safety_reserve_bytes": (512u64 << 20).to_string(),
            })
        })
        .collect();
    let mut plan: starlings_protocol::PlacementPlan = serde_json::from_value(serde_json::json!({
        "kind": "PlacementPlan", "protocol_version": 1, "fixture": false,
        "plan_id": uuid::Uuid::new_v4().to_string(), "plan_digest": "0".repeat(64),
        "cluster_id": config["cluster"], "authority_epoch": "1", "manifest_digest": manifest,
        "mode": "layer_pipeline", "model_layer_count": layers,
        "execution_profile": "glyd-range", "boundary_profile": "qwen35-residual-mlp-bf16",
        "context_limit": o.context, "max_sessions": 1, "stages": plan_stages,
        "created_at": at(0)?, "expires_at": at(60)?,
    }))
    .map_err(|e| format!("plan: {e}"))?;
    plan.plan_digest = starlings_protocol::canonical_digest(
        &serde_json::to_value(&plan).map_err(|e| e.to_string())?,
        "plan_digest",
    )
    .map_err(|e| format!("{e:?}"))?;
    let pipeline = PipelineRoute {
        version: Version,
        listen: CHAT.parse().map_err(|_| "chat address")?,
        plan,
        stages: vec![
            PipelineStage {
                endpoint: locals[0].1.listen,
                data: locals[0].1.data_listen,
                more_data: locals[0].1.more_data_listen.clone(),
            },
            second,
        ],
        chunk_tokens: CHUNK,
        max_output_tokens: o.max_output(),
        lifetime_ms: 60_000,
        stop_tokens: stop_tokens(&o.model),
        draft_tokens: o.drafts,
    };

    // The approved launches: each daemon gets exactly the record built above.
    launch(
        &mut daemons,
        coordinator,
        "--pipeline",
        &pipeline,
        |d| pipeline.validate(d),
        &["--ui", UI, "--ui-client", nodes.last().ok_or("no client")?],
    )?;
    for (dir, stage) in &locals {
        launch(
            &mut daemons,
            dir,
            "--stage",
            stage,
            |d| stage.validate(d),
            &[],
        )?;
    }
    println!("Loading the model...");
    for (dir, _) in &locals {
        let mut failed = None;
        wait(&stop, Duration::from_secs(900), "loading", || {
            stage_ready(dir).unwrap_or_else(|e| {
                failed = Some(e);
                true
            })
        })?;
        if let Some(error) = failed {
            return Err(error);
        }
    }
    if remote.is_some() {
        println!("Waiting for the other machine's stage...");
    }
    // The coordinator issues a request's stages only once their samples say ready.
    for node in &nodes[..2] {
        wait(
            &stop,
            Duration::from_secs(900),
            "the stages' ready samples",
            || {
                cli(coordinator, &["cluster", "resources", node])
                    .ok()
                    .and_then(|v| {
                        serde_json::from_str::<serde_json::Value>(
                            v.strip_prefix("RESOURCES: ")?.trim(),
                        )
                        .ok()
                    })
                    .is_some_and(|v| v["sample"]["capabilities"]["state"] == "ready")
            },
        )?;
    }
    // A one-hour token for API clients on this machine (the UI holds its own).
    let token = coordinator.join("token.json");
    let _ = fs::remove_file(&token);
    let scopes = write(
        coordinator,
        "scopes.json",
        &serde_json::json!([{"model": manifest, "action": "infer"}]),
    )?;
    cli(
        coordinator,
        &[
            "cluster",
            "token",
            nodes.last().ok_or("no client")?,
            &scopes.to_string_lossy(),
            "3600",
            &token.to_string_lossy(),
        ],
    )?;
    let log = fs::read_to_string(coordinator.join("daemon.log")).unwrap_or_default();
    let ui = log
        .lines()
        .rev()
        .find_map(|l| l.strip_prefix("UI_READY: "))
        .unwrap_or("(see the coordinator's daemon.log)");
    println!("READY: open {ui}");
    println!("  API: http://{CHAT}/v1/chat/completions, model {manifest}");
    println!("  bearer token for the API (one hour): {}", token.display());
    println!("Ctrl-C stops everything.");
    daemons.supervise(&stop)
}

/// `starlings join ...`: the second machine's stage of a two-machine `up`. The join code is pasted
/// at a prompt, never given on the command line (it carries a one-time invitation).
pub fn join(state: &Path, args: &[OsString]) -> Result<()> {
    let mut o = Options::parse(args)?;
    state_root(state)?;
    let worker = state.join("stage");
    let saved = read_json(&state.join("join.json"));
    let (cut, host) = if worker.join("config.json").exists() {
        let saved: SavedJoin = serde_json::from_value(saved)
            .map_err(|e| format!("invalid or older join.json; use a new --state-dir: {e}"))?;
        if saved.manifest != o.model_files || o.host.is_some_and(|h| h != saved.host) {
            return Err(
                "saved stage does not match this model or host; use a new --state-dir".into(),
            );
        }
        (o.context, o.sessions, o.drafts) = (saved.context, saved.sessions, saved.drafts);
        o.validate_workload()?;
        validate_cut(
            saved.cut,
            split::read(&o.model, o.context, o.sessions)?.layers.len() as u32,
        )?;
        (saved.cut, saved.host)
    } else {
        let code: JoinCode = decode(&read_line("Paste the join code from the first machine: ")?)?;
        (o.context, o.sessions, o.drafts) = (code.context, code.sessions, code.drafts);
        o.validate_workload()?;
        if code.manifest != o.model_files {
            return Err("the join code approves a different model's files".into());
        }
        let invitation_value = unpack(code.invitation)?;
        let cluster = cluster_check(&invitation_value);
        let invitation = write(state, "invitation.json", &invitation_value)?;
        let endpoint = Invitation::load(&private(state)?, "invitation.json")
            .map_err(|e| format!("invitation: {e:?}"))?
            .endpoint();
        let host = o.host.map_or_else(|| address_toward(endpoint), Ok)?;
        let shape = split::read(&o.model, o.context, o.sessions)?;
        let layers = shape.layers.len() as u32;
        let free = gpus()?[0];
        let cut = split::cut(&shape, [code.free, free], split::RESERVE).ok_or(format!(
            "{layers} layers at context {} do not fit {} there and {} here",
            o.context,
            gib(code.free),
            gib(free)
        ))?;
        let (stage, manifest) = local_stage(
            &o,
            &shape,
            [cut, layers],
            (host, host, STAGE_PORTS[1], 0),
            &state.join("describe"),
        )?;
        println!("Starlings: this machine joins the cluster at {endpoint}");
        println!("  cluster check {cluster} (the first machine shows it; it must be the same)");
        summary(&o, layers)?;
        println!(
            "  layers   {cut}..{layers} on GPU 0, about {} of {} free; context {}, {} session(s), \
             {} drafts",
            gib(stage_bytes(&shape, [cut, layers])),
            gib(free),
            o.context,
            o.sessions,
            o.drafts
        );
        confirm("Join and serve these layers?", o.yes)?;
        // An earlier attempt that never joined left only its unapproved key.
        if worker.exists() {
            fs::remove_dir_all(&worker).map_err(|e| e.to_string())?;
        }
        let prepared = cli(
            &worker,
            &["cluster", "prepare", &invitation.to_string_lossy()],
        )?;
        let (node, fingerprint) = (
            field(&prepared, "node=")?,
            field(&prepared, "fingerprint=")?,
        );
        let approval = encode(&ApprovalCode {
            node: node.clone(),
            fingerprint: fingerprint.clone(),
            host,
            engine: stage.executable_sha256.clone(),
            manifest,
            free,
        })?;
        println!("\nPaste this on the first machine:\n\n{approval}\n");
        println!(
            "It then asks for this check; type it there: {}\n",
            check(&format!("{node} {fingerprint}"))
        );
        println!("Waiting for its approval...");
        let stop = stop_flag()?;
        wait(&stop, Duration::from_secs(300), "approval", || {
            cli(&worker, &["cluster", "join", &invitation.to_string_lossy()]).is_ok()
        })?;
        write(
            state,
            "join.json",
            &serde_json::json!({"cut": cut, "host": host, "context": o.context,
                "sessions": o.sessions, "drafts": o.drafts, "manifest": o.model_files}),
        )?;
        // The invitation is spent (the worker keeps what it needs in its own state).
        let _ = fs::remove_file(&invitation);
        return serve(&o, state, &worker, cut, host, stop);
    };
    println!("Starlings, the saved stage:");
    summary(
        &o,
        split::read(&o.model, o.context, o.sessions)?.layers.len() as u32,
    )?;
    println!("  layers   {cut}.. on GPU 0, at {host}");
    confirm("Serve the saved stage again?", o.yes)?;
    serve(&o, state, &worker, cut, host, stop_flag()?)
}
fn serve(
    o: &Options,
    state: &Path,
    worker: &Path,
    cut: u32,
    host: IpAddr,
    stop: Arc<AtomicBool>,
) -> Result<()> {
    let shape = split::read(&o.model, o.context, o.sessions)?;
    let layers = shape.layers.len() as u32;
    let (stage, _) = local_stage(
        o,
        &shape,
        [cut, layers],
        (host, host, STAGE_PORTS[1], 0),
        &state.join("describe"),
    )?;
    let mut daemons = Daemons(Vec::new());
    launch(
        &mut daemons,
        worker,
        "--stage",
        &stage,
        |d| stage.validate(d),
        &[],
    )?;
    println!("Loading layers {cut}..{layers}...");
    let mut failed = None;
    wait(&stop, Duration::from_secs(900), "loading", || {
        stage_ready(worker).unwrap_or_else(|e| {
            failed = Some(e);
            true
        })
    })?;
    if let Some(error) = failed {
        return Err(error);
    }
    println!("READY: serving layers {cut}..{layers}. Ctrl-C stops.");
    daemons.supervise(&stop)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn options() -> Options {
        Options {
            model: PathBuf::from("/model"),
            engine: PathBuf::from("/engine"),
            gpu_lib: None,
            machines: 1,
            host: None,
            context: 4096,
            sessions: 1,
            drafts: 7,
            yes: true,
            model_files: starlings_protocol::byte_digest(b"checkpoint"),
        }
    }

    #[test]
    fn saved_setup_rejects_changed_model_workload_topology_and_invalid_splits() {
        let o = options();
        let host = Ipv4Addr::LOCALHOST.into();
        let saved = SavedUp {
            machines: 1,
            cut: 1,
            nodes: vec!["a".into(), "b".into(), "c".into()],
            manifest: o.model_files.clone(),
            peer: None,
            context: 4096,
            sessions: 1,
            drafts: 7,
            gpus: 2,
            host,
        };
        saved.validate(&o, 2, 2, host).unwrap();
        assert!(saved.validate(&o, 1, 2, host).is_err());
        assert!(saved.validate(&o, 2, 1, host).is_err());
        for (key, value) in [
            (
                "manifest",
                serde_json::json!(starlings_protocol::byte_digest(b"changed")),
            ),
            ("machines", serde_json::json!(2)),
            ("context", serde_json::json!(2048)),
            ("sessions", serde_json::json!(2)),
            ("drafts", serde_json::json!(0)),
            ("cut", serde_json::json!(0)),
            ("cut", serde_json::json!(2)),
            ("cut", serde_json::json!(u64::MAX)),
            ("nodes", serde_json::json!(["a"])),
            ("nodes", serde_json::json!(["a", "a", "c"])),
            ("host", serde_json::json!("192.0.2.1")),
        ] {
            let mut record = serde_json::to_value(&saved).unwrap();
            record[key] = value;
            assert!(
                !serde_json::from_value::<SavedUp>(record)
                    .is_ok_and(|s| s.validate(&o, 2, 2, host).is_ok()),
                "{key}"
            );
        }
        let mut legacy = serde_json::to_value(&saved).unwrap();
        legacy.as_object_mut().unwrap().remove("context");
        assert!(serde_json::from_value::<SavedUp>(legacy).is_err());
    }

    #[test]
    fn invalid_workloads_are_refused_before_fetch_or_engine_launch() {
        for args in [
            ["--sessions", "0"],
            ["--sessions", "3"],
            ["--ctx", "0"],
            ["--ctx", "31"],
            ["--drafts", "8"],
            ["--drafts", "4294967295"],
        ] {
            let args = ["--model", "org/model", args[0], args[1]].map(OsString::from);
            assert!(
                Options::parse(&args)
                    .err()
                    .unwrap()
                    .contains("invalid context")
            );
        }
        options().validate_workload().unwrap();
    }

    #[test]
    fn stop_tokens_are_the_end_of_text_and_turn_ids() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("tokenizer.json"),
            r#"{"added_tokens":[{"id":7,"content":"<|im_end|>"},{"id":5,"content":"<|endoftext|>"},{"id":6,"content":"<|im_start|>"}]}"#,
        )
        .unwrap();
        std::fs::write(
            dir.path().join("config.json"),
            r#"{"text_config":{"eos_token_id":7}}"#,
        )
        .unwrap();
        assert_eq!(super::stop_tokens(dir.path()), vec![5, 7]);
    }
    #[test]
    fn packed_invitations_round_trip() {
        let value = serde_json::json!({"ca": (0..300).map(|i| i % 256).collect::<Vec<_>>(),
            "endpoint": "10.0.0.1:9741", "short": [1, 2, 3], "nested": {"token": vec![7; 32]}});
        let packed = super::pack(value.clone());
        assert!(packed["ca"]["$b64"].is_string());
        assert_eq!(packed["short"], serde_json::json!([1, 2, 3]));
        assert_eq!(super::unpack(packed).unwrap(), value);
        assert_eq!(super::check("a"), super::check("a"));
        assert_ne!(super::check("a b"), super::check("a c"));
        assert_eq!(super::check("x").len(), 19);
    }
    #[test]
    fn codes_round_trip_and_refuse_junk() {
        let code = super::encode(&serde_json::json!({"a": 1})).unwrap();
        assert_eq!(
            super::decode::<serde_json::Value>(&format!(" {code}\n")).unwrap(),
            serde_json::json!({"a": 1})
        );
        assert!(super::decode::<serde_json::Value>("not base64!").is_err());
    }
}
