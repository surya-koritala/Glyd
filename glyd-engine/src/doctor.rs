//! `glyd doctor` (ROADMAP P10): what this machine has and what it can run, said as checks, each with the one line that fixes it, and the same as one JSON
//! bundle a user attaches to an issue (`--json`; `--bundle [FILE]` writes it to a file). The probing (`Facts::probe`) is apart from the checking (`checks`),
//! which is a pure function of the facts, so every failure of the install matrix is a unit test; the bundle's text goes through `redact` (tokens, keys and the
//! home folder) before it leaves.

use crate::args::Args;
use crate::dev::{Dev, Gpu};
use crate::plan::{self, LoadPlan, Machine, HOST_MARGIN};
use crate::ui::{gb, gib};
use crate::{hub, machine, store};
use std::path::{Path, PathBuf};

const VERSION: &str = env!("CARGO_PKG_VERSION");
/// The bundle's schema, its `schema` member.
pub const SCHEMA: &str = "glyd-doctor/1";
/// The small model the sample plan is made for: Gemma 4 12B, whose config.json is in the binary, so the plan needs no network.
const SAMPLE_MODEL: &str = "google/gemma-4-12B-it";
const SAMPLE_CONFIG: &str = include_str!("../tests/fixtures/gemma4_12b_config.json");

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    /// Fine.
    Ok,
    /// Worth knowing, not a failure (cuBLAS absent, a port taken that a run does not need, a model too big for this GPU).
    Note,
    /// Something a run needs is missing or wrong: `doctor` exits 1.
    Fail,
}

impl Status {
    fn word(self) -> &'static str {
        match self {
            Status::Ok => "ok",
            Status::Note => "note",
            Status::Fail => "fail",
        }
    }
}

/// One thing checked: its place in the failure matrix (`id`), the label of its line, what was found, and where it failed or is worth a note, the one line that fixes it.
#[derive(Clone, Debug, PartialEq)]
pub struct Check {
    pub id: &'static str,
    pub label: &'static str,
    pub status: Status,
    pub found: String,
    pub fix: Option<String>,
}

fn check(id: &'static str, label: &'static str, status: Status, found: impl Into<String>, fix: Option<&str>) -> Check {
    Check { id, label, status, found: found.into(), fix: fix.map(str::to_string) }
}

/// A folder and the disk space free where it is (or would be).
#[derive(Clone, Debug)]
pub struct Dir {
    pub path: String,
    pub free: Option<u64>,
}

#[derive(Clone, Debug)]
pub enum HubState {
    Offline,
    Reachable(String),
    Unreachable(String, String),
}

/// A port this program uses, and who holds it.
#[derive(Clone, Debug)]
pub struct Listener {
    pub port: u16,
    pub role: &'static str,
    /// None: free; Some(who): held (who: the program, where the system says).
    pub held_by: Option<String>,
}

/// A model in a cache and the bytes it takes there.
#[derive(Clone, Debug)]
pub struct Cached {
    pub name: String,
    pub bytes: u64,
}

/// Everything `doctor` reads of the machine, so that what it concludes is a function of these.
#[derive(Clone, Debug)]
pub struct Facts {
    pub os: String,
    pub arch: String,
    pub gpus: Result<Vec<Gpu>, String>,
    pub library: Result<String, String>,
    pub cublas: Option<String>,
    /// The computer's memory: (total, available), bytes.
    pub ram: Option<(u64, u64)>,
    pub cache: Dir,
    pub models: Dir,
    pub hub_token: bool,
    pub hub: HubState,
    pub listeners: Vec<Listener>,
    pub hub_models: Vec<Cached>,
    pub packed_models: Vec<Cached>,
    /// This binary is the development build that runs AMD GPUs (GLYD_HIP=1).
    pub hip: bool,
    /// A GPU of the machine runs under Windows' WDDM (dev.rs `Dev::wddm`).
    pub wddm: bool,
}

fn dir_of(p: &Path) -> Dir {
    let mut q = p.to_path_buf();
    while !q.exists() && q.pop() {}
    Dir { path: p.display().to_string(), free: hub::disk_free(&q) }
}

/// `MemTotal` and `MemAvailable` of a /proc/meminfo's text, in bytes.
pub fn parse_ram(text: &str) -> Option<(u64, u64)> {
    let kb = |key: &str| text.lines().find_map(|l| l.strip_prefix(key)?.split_whitespace().next()?.parse::<u64>().ok()?.checked_mul(1024));
    Some((kb("MemTotal:")?, kb("MemAvailable:")?))
}

fn bytes_in(dir: &Path) -> u64 {
    std::fs::read_dir(dir).map_or(0, |d| {
        d.flatten()
            .map(|e| match e.metadata() {
                Ok(m) if m.is_dir() => bytes_in(&e.path()),
                Ok(m) => m.len(),
                Err(_) => 0,
            })
            .sum()
    })
}

/// The models in the Hub's cache (`models--ORG--NAME`: the blobs are the bytes) and the packed copies of this program's cache, the largest first, at most 40.
fn cached_models(hub_dir: &Path, packed_dir: &Path) -> (Vec<Cached>, Vec<Cached>) {
    let list = |dir: &Path, blobs: bool| -> Vec<Cached> {
        let mut v: Vec<Cached> = std::fs::read_dir(dir)
            .map(|d| {
                d.flatten()
                    .filter(|e| e.path().is_dir() && !e.file_name().to_string_lossy().starts_with('.'))
                    .map(|e| {
                        let n = e.file_name().to_string_lossy().into_owned();
                        let name = n.strip_prefix("models--").map_or(n.clone(), |r| r.replace("--", "/"));
                        Cached { name, bytes: bytes_in(&if blobs { e.path().join("blobs") } else { e.path() }) }
                    })
                    .collect()
            })
            .unwrap_or_default();
        v.sort_by_key(|c| std::cmp::Reverse(c.bytes));
        v.truncate(40);
        v
    };
    (list(hub_dir, true), list(packed_dir, false))
}

/// The ports a run, a server and the service use here: the service's (GLYD_SERVICE_PORT, else its default) and the first a server tries (8000).
fn listeners() -> Vec<Listener> {
    let service = std::env::var("GLYD_SERVICE_PORT").ok().and_then(|p| p.parse().ok()).unwrap_or(crate::service::DEFAULT_PORT);
    [(service, "the service (glyd service)"), (8000, "a server (glyd serve; the next free port up to 8020 where this is taken)")]
        .into_iter()
        .map(|(port, role)| {
            let free = std::net::TcpListener::bind(("127.0.0.1", port)).is_ok();
            Listener {
                port,
                role,
                held_by: (!free).then(|| crate::service::port_owner(port).unwrap_or_else(|| "a program the system does not name".to_string())),
            }
        })
        .collect()
}

impl Facts {
    pub fn probe() -> Facts {
        let gpus = match Dev::probe_all() {
            Ok(all) if machine::pick(&all).is_some() => Ok(all),
            Ok(_) => Err("the driver shows none".to_string()),
            Err(e) => Err(e.0),
        };
        let wddm = gpus.as_ref().is_ok_and(|all| all.iter().any(|g| Dev::wddm(g.ordinal)));
        let (hub_dir, cache) = (hub::hub_cache(), store::cache_root());
        let (hub_models, packed_models) = cached_models(&hub_dir, &store::packed_root());
        let h = hub::Hub::from_env();
        let hub_state = if hub::offline() {
            HubState::Offline
        } else {
            match h.list("Qwen/Qwen3-0.6B", "main") {
                Ok(_) => HubState::Reachable(h.endpoint().to_string()),
                Err(e) => HubState::Unreachable(h.endpoint().to_string(), e.to_string()),
            }
        };
        Facts {
            os: std::env::consts::OS.to_string(),
            arch: std::env::consts::ARCH.to_string(),
            gpus,
            library: crate::ffi::Lib::find().map(|l| l.path.clone()).map_err(|e| e.0),
            cublas: crate::ffi::Blas::find().map(|b| b.path.clone()),
            ram: crate::os::meminfo().and_then(|t| parse_ram(&t)),
            cache: dir_of(&cache),
            models: dir_of(&hub_dir),
            hub_token: h.has_token(),
            hub: hub_state,
            listeners: listeners(),
            hub_models,
            packed_models,
            hip: cfg!(glyd_hip),
            wddm,
        }
    }

    /// This computer as the planner sees it (what `Machine::probe` makes of the same facts).
    pub fn machine(&self) -> Machine {
        let vendor = if self.hip { plan::Vendor::Amd } else { plan::Vendor::Nvidia };
        let devices = self
            .gpus
            .as_ref()
            .map(|all| {
                all.iter()
                    .map(|g| plan::Device {
                        name: g.name.clone(),
                        vendor,
                        cc: g.cc,
                        total: g.total,
                        free: g.free,
                        bandwidth: plan::bandwidth_of(&g.name).unwrap_or(300.0),
                        driver: g.driver,
                        unified: plan::is_unified(&g.name, vendor),
                    })
                    .collect()
            })
            .unwrap_or_default();
        Machine {
            name: "this computer".into(),
            devices,
            ram_free: self.ram.map_or(8 << 30, |r| r.1),
            ram_bandwidth: 50.0,
            cores: std::thread::available_parallelism().map_or(4, |n| n.get()),
            arch: self.arch.clone(),
        }
    }
}

/// The plan `run` would load for the sample model on this machine (the level the ladder takes), or why none fits.
pub fn sample_plan(m: &Machine) -> Result<LoadPlan, String> {
    let a = Args::from(["fit".to_string(), SAMPLE_MODEL.to_string()]);
    let row = crate::support::row(crate::support::family(SAMPLE_CONFIG, SAMPLE_MODEL).map_err(|r| r.what)?);
    let sh = plan::Shape::from_config(SAMPLE_CONFIG)?;
    match plan::exact_options(&a, SAMPLE_MODEL, SAMPLE_CONFIG, None, row, &sh, m, 1, true) {
        Some((options, refusals)) => match plan::choose(&options, plan::TARGET_CTX) {
            Some(c) => c.load.ok_or_else(|| "no plan".to_string()),
            None => Err(refusals.into_iter().next().map_or_else(|| "nothing fits".to_string(), |r| format!("{} ({})", r.what, r.fix))),
        },
        None => Err("this machine has no GPU to plan for".to_string()),
    }
}

/// A failure class of the install matrix: what goes wrong, the check that finds it, and its one-line fix. The matrix is the contract that no refusal or error
/// class of `run`, `serve` and `service` is undiagnosable (tests: every check named here is made, and fails, for the facts that cause it).
pub struct Failure {
    pub class: &'static str,
    pub check: &'static str,
    pub fix: &'static str,
}

pub const MATRIX: &[Failure] = &[
    Failure {
        class: "no NVIDIA driver, or no GPU visible to the process",
        check: "gpu",
        fix: "Check nvidia-smi; install a current NVIDIA driver (glyd doctor names the oldest this build runs on); in a container give it the GPU (--gpus all); an empty CUDA_VISIBLE_DEVICES hides every GPU",
    },
    Failure { class: "GPU older than Ampere", check: "gpu-age", fix: "Nothing on this card: RTX 30 series and later, A10, A100, L4, L40S and H100 run" },
    Failure { class: "driver older than the engine's kernels", check: "driver", fix: "Update the NVIDIA driver to the version `glyd doctor` names (sudo ubuntu-drivers install, then reboot)" },
    Failure {
        class: "GPU library not found",
        check: "gpu-library",
        fix: "bash gpu/build_lib.sh target/release, or GLYD_GPU_LIB=/path/to/libglyd_gpu_cuda13.so",
    },
    Failure {
        class: "AMD or Apple GPU, or an Arm computer",
        check: "machine",
        fix: "The engine runs on NVIDIA GPUs on x86-64 (AMD: GLYD_HIP=1 development build; Apple and Arm: ROADMAP V3 and P18)",
    },
    Failure {
        class: "the computer's memory is too small for a mixture of experts' experts",
        check: "ram",
        fix: "A smaller model or a lower level, more memory, or --host-memory GIB for what the experts may take",
    },
    Failure {
        class: "the disk is nearly full",
        check: "cache",
        fix: "Free space, or GLYD_CACHE=/big/drive for packed models and HF_HOME=/big/drive for downloads",
    },
    Failure { class: "the models' disk is nearly full", check: "models", fix: "Free space, or HF_HOME=/big/drive" },
    Failure { class: "a gated model asks for a token", check: "hub-token", fix: "Accept the model's licence on its page, then glyd login (or set HF_TOKEN)" },
    Failure {
        class: "the Hub is not reachable",
        check: "hub",
        fix: "Check the network, GLYD_IPV4=1, HF_ENDPOINT for a mirror; a downloaded model runs with HF_HUB_OFFLINE=1",
    },
    Failure {
        class: "the model does not fit this GPU",
        check: "plan",
        fix: "glyd fit MODEL shows each level's size; a lower level, --context 4096, or another GPU",
    },
    Failure {
        class: "the service's port is in use",
        check: "service-port",
        fix: "Stop the program named, or glyd service --port N (GLYD_SERVICE_PORT=N for the other commands)",
    },
    Failure { class: "a server's port is in use", check: "server-port", fix: "Nothing: a server takes the next free port up to 8020; --port N chooses one" },
];

/// Every check, in the order `doctor` says them.
pub fn checks(f: &Facts) -> Vec<Check> {
    let mut v = Vec::new();
    let m = f.machine();
    let main = f.gpus.as_ref().ok().and_then(|all| machine::pick(all).cloned());
    match (&f.gpus, &main) {
        (Ok(all), Some(g)) => {
            v.push(check(
                "gpu",
                "GPU",
                Status::Ok,
                format!(
                    "{} (compute capability {}.{}), {} free of {}; the driver runs CUDA {}.{}",
                    g.name,
                    g.cc.0,
                    g.cc.1,
                    gib(g.free),
                    gib(g.total),
                    g.driver / 1000,
                    g.driver % 1000 / 10
                ),
                None,
            ));
            if g.cc.0 < 8 {
                v.push(check("gpu-age", "", Status::Fail, "older than the engine runs on (Ampere or newer)", Some(MATRIX[1].fix)));
            }
            match crate::cli::driver_problem(g) {
                Some(fix) => v.push(check("driver", "driver", Status::Fail, fix, Some(MATRIX[2].fix))),
                None => v.push(check(
                    "driver",
                    "driver",
                    Status::Ok,
                    format!("CUDA {}.{}, the engine needs {}", g.driver / 1000, g.driver % 1000 / 10, machine::driver_need(g.cc)),
                    None,
                )),
            }
            if f.wddm {
                let (found, fix) = machine::wddm_note(&g.name, g.free, g.total);
                v.push(check("wddm", "WDDM", Status::Note, found, Some(&fix)));
            }
            if all.len() > 1 {
                v.push(check(
                    "gpus",
                    "",
                    Status::Ok,
                    format!("{} GPUs: the one with the most free memory is used (CUDA_VISIBLE_DEVICES names the ones to choose from)", all.len()),
                    None,
                ));
            }
            let (l, why) = crate::weights::best_layout(g.cc, g.total, 0, 0);
            let other = if l == crate::weights::Layout::Tiered { "mma12" } else { "mma" };
            v.push(check("layout", "layout", Status::Ok, format!("{} (the default here: {why}); --layout {other} for the other", l.name()), None));
        }
        _ => {
            let why = f.gpus.as_ref().err().cloned().unwrap_or_else(|| "the driver shows none".to_string());
            v.push(check("gpu", "GPU", Status::Fail, format!("none usable: {why}"), Some(MATRIX[0].fix)));
        }
    }
    match &f.library {
        Ok(p) => v.push(check("gpu-library", "GPU library", Status::Ok, p.clone(), None)),
        Err(e) => v.push(check("gpu-library", "GPU library", Status::Fail, format!("not found: {e}"), Some(MATRIX[3].fix))),
    }
    match &f.cublas {
        Some(p) => v.push(check("cublas", "cuBLAS", Status::Ok, format!("found ({p}): long prompts run through it"), None)),
        None => v.push(check(
            "cublas",
            "cuBLAS",
            Status::Note,
            "not found: long prompts use the engine's own products (a little slower)",
            Some("Optional: install the CUDA 13 cuBLAS, or set GLYD_CUBLAS to it"),
        )),
    }
    // the backends: each one detected or not, its version, and accepted or rejected with the reason (the refusal `run` gives)
    let refusal = crate::support::machine(&m);
    let cuda_seen = main.is_some() && !f.hip;
    v.push(match (&main, &refusal) {
        (Some(g), None) if cuda_seen => {
            check("backend-cuda", "backend CUDA", Status::Ok, format!("detected (driver CUDA {}.{}), accepted", g.driver / 1000, g.driver % 1000 / 10), None)
        }
        (Some(g), Some(r)) if cuda_seen => check(
            "backend-cuda",
            "backend CUDA",
            Status::Note,
            format!("detected (driver CUDA {}.{}), rejected: {}", g.driver / 1000, g.driver % 1000 / 10, r.what),
            Some(&r.fix),
        ),
        _ => check("backend-cuda", "backend CUDA", Status::Note, "not detected", None),
    });
    v.push(if f.hip && main.is_some() {
        match &refusal {
            None => check("backend-rocm", "backend ROCm", Status::Ok, "detected (development build), accepted", None),
            Some(r) => check("backend-rocm", "backend ROCm", Status::Note, format!("detected (development build), rejected: {}", r.what), Some(&r.fix)),
        }
    } else {
        check("backend-rocm", "backend ROCm", Status::Note, "not in this build (a development build with GLYD_HIP=1 runs AMD GPUs: ROADMAP V1)", None)
    });
    v.push(check(
        "backend-metal",
        "backend Metal",
        Status::Note,
        if f.os == "macos" {
            "detected (this is a Mac), rejected: Apple GPUs are not supported yet (ROADMAP V3)"
        } else {
            "not detected (planned: ROADMAP V3)"
        },
        None,
    ));
    v.push(check("backend-cpu", "backend CPU", Status::Note, "not built yet (ROADMAP V4): without a GPU glyd compresses files", None));
    // a refusal that the GPU and driver checks above do not already say (AMD, Apple, an Arm computer)
    if let (Some(g), Some(r)) = (&main, &refusal) {
        if g.cc.0 >= 8 && g.driver >= machine::driver_min(g.cc) {
            v.push(check("machine", "machine", Status::Fail, r.what.clone(), Some(&r.fix)));
        }
    } else if let (None, Some(r)) = (&main, &refusal) {
        let _ = r; // (no GPU: the GPU line says it)
    }
    // the memory budget: the terms a plan is made from
    let a_gib = |b: u64| format!("{:.1} GiB", b as f64 / (1u64 << 30) as f64);
    if let Some(g) = &main {
        let (reserve, work, dyn_work) = (plan::reserve_of(g.free), plan::LOSSY_WORK, plan::dynamic_work_of(g.free));
        v.push(check(
            "budget",
            "GPU budget",
            Status::Ok,
            format!(
                "{} free of {} ({} used by other programs and the driver); the engine keeps {} and its work takes {} ({} on the dynamic route): {} for the weights and the cache",
                a_gib(g.free),
                a_gib(g.total),
                a_gib(g.total.saturating_sub(g.free)),
                a_gib(reserve),
                a_gib(work),
                a_gib(dyn_work),
                a_gib(plan::room_of(g.free).saturating_sub(work))
            ),
            None,
        ));
    }
    match f.ram {
        Some((total, avail)) => {
            let room = avail.saturating_sub(HOST_MARGIN);
            // a computer that cannot hold a small mixture's experts beside the system says so; below the margin nothing is left for experts at all
            let status = if avail < HOST_MARGIN + (2 << 30) { Status::Note } else { Status::Ok };
            v.push(check(
                "ram",
                "RAM",
                status,
                format!(
                    "{} available of {}; {} kept for the system: {} for experts held in memory",
                    a_gib(avail),
                    a_gib(total),
                    a_gib(HOST_MARGIN),
                    a_gib(room)
                ),
                (status == Status::Note).then_some(MATRIX[5].fix),
            ));
        }
        None => v.push(check("ram", "RAM", Status::Note, "the system does not say how much is available", None)),
    }
    // what this machine can run: the sample plan
    v.push(match sample_plan(&m) {
        Ok(p) => check("plan", "sample plan", Status::Ok, format!("{SAMPLE_MODEL}: {}", p.text()), None),
        Err(why) => check("plan", "sample plan", Status::Note, format!("{SAMPLE_MODEL} does not load here: {why}"), Some(MATRIX[10].fix)),
    });
    let disk = |id: &'static str, label: &'static str, d: &Dir, fix: &'static str| {
        let free = d.free.map_or("unknown".to_string(), gb);
        let low = d.free.is_some_and(|b| b < 10 << 30);
        check(id, label, if low { Status::Note } else { Status::Ok }, format!("{} ({free} free)", d.path), low.then_some(fix))
    };
    v.push(disk("cache", "cache", &f.cache, MATRIX[6].fix));
    v.push(disk("models", "models", &f.models, MATRIX[7].fix));
    let held = |v: &[Cached]| format!("{} models, {}", v.len(), gb(v.iter().map(|c| c.bytes).sum()));
    v.push(check("cached", "cached", Status::Ok, format!("{} in the Hub's cache, {} packed here", held(&f.hub_models), held(&f.packed_models)), None));
    v.push(if f.hub_token {
        check("hub-token", "Hub token", Status::Ok, "found", None)
    } else {
        check("hub-token", "Hub token", Status::Note, "none (public models only; glyd login for a gated one)", Some(MATRIX[8].fix))
    });
    v.push(match &f.hub {
        HubState::Reachable(e) => check("hub", "Hub", Status::Ok, format!("{e} reachable"), None),
        HubState::Offline => check("hub", "Hub", Status::Ok, "offline (HF_HUB_OFFLINE)", None),
        HubState::Unreachable(e, why) => check("hub", "Hub", Status::Fail, format!("{e}: {why}"), Some(MATRIX[9].fix)),
    });
    for l in &f.listeners {
        let (id, fix) = if l.role.starts_with("the service") { ("service-port", MATRIX[11].fix) } else { ("server-port", MATRIX[12].fix) };
        v.push(match &l.held_by {
            None => check(id, "port", Status::Ok, format!("{} is free for {}", l.port, l.role), None),
            Some(who) => check(id, "port", Status::Note, format!("{} is in use by {who} ({})", l.port, l.role), Some(fix)),
        });
    }
    v
}

/// The checks as the terminal shows them: the lines `doctor` has always printed (a label and what was found), the fix under a line that is not fine, and the verdict.
pub fn text(cs: &[Check]) -> (String, i32) {
    let mut s = format!("glyd engine {VERSION}\n");
    for c in cs {
        s += &format!("  {:<14}{}\n", c.label, c.found);
        if let (Some(fix), true) = (&c.fix, c.status != Status::Ok) {
            s += &format!("  {:<14}fix: {fix}\n", "");
        }
    }
    let bad = cs.iter().filter(|c| c.status == Status::Fail).count();
    if bad == 0 {
        s += "Everything glyd run needs is here.\n";
        (s, 0)
    } else {
        s += &format!("{bad} thing(s) to fix.\n");
        (s, 1)
    }
}

// ---- the bundle --------------------------------------------------------------------------------------------------------------------------------------------------------------------------

const SECRET_WORDS: [&str; 10] = ["api-key", "api_key", "apikey", "hf-token", "hf_token", "token", "secret", "password", "passwd", "authorization"];
const TOKEN_PREFIXES: [&str; 8] = ["hf_", "sk-", "ghp_", "gho_", "github_pat_", "xoxb-", "xoxp-", "AIza"];

fn word_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.' | '/' | '+' | '=')
}

/// `text` with what must not leave the machine taken out: the home folder (as `~`), tokens by their well-known prefixes (`hf_`, `sk-`, ...: kept as the prefix
/// and `<redacted>`), a `Bearer` credential, and the value after a secret's name (`--api-key X`, `HF_TOKEN=X`, `"token": "X"`, `password: X`).
pub fn redact(text: &str, home: &str) -> String {
    redact_with(text, home, &[])
}

/// `redact`, and each of `secrets` (a token or key this machine holds, 6 characters or more) taken out wherever it is, whatever is around it.
pub fn redact_with(text: &str, home: &str, secrets: &[String]) -> String {
    let mut s = text.to_string();
    for secret in secrets.iter().filter(|x| x.len() >= 6) {
        s = s.replace(secret.as_str(), "<redacted>");
    }
    if home.len() > 1 {
        s = s.replace(&home.replace('\\', "\\\\"), "~").replace(home, "~");
    }
    s = redact_urls(&s);
    s = redact_prefixed(&s);
    s = redact_bearer(&s);
    redact_keyed(&s)
}

/// `scheme://user:password@host` as `scheme://<redacted>@host` (a proxy or a mirror with its credentials in the address).
fn redact_urls(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(i) = rest.find("://") {
        let (head, tail) = rest.split_at(i + 3);
        out += head;
        let end = tail.find(|c: char| c.is_whitespace() || matches!(c, '/' | '"' | '\'' | '\\' | '<' | '>')).unwrap_or(tail.len());
        match tail[..end].rfind('@') {
            Some(at) => {
                out += "<redacted>";
                rest = &tail[at..];
            }
            None => rest = tail,
        }
    }
    out + rest
}

fn redact_prefixed(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let chars: Vec<(usize, char)> = s.char_indices().collect();
    let mut i = 0;
    while i < chars.len() {
        let at = chars[i].0;
        let boundary = i == 0 || !chars[i - 1].1.is_ascii_alphanumeric();
        if let Some(p) = TOKEN_PREFIXES.iter().find(|p| boundary && s[at..].starts_with(**p)) {
            let run = s[at + p.len()..].chars().take_while(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-')).count();
            if run >= 8 {
                out += p;
                out += "<redacted>";
                i += p.chars().count() + run;
                continue;
            }
        }
        out.push(chars[i].1);
        i += 1;
    }
    out
}

fn redact_bearer(s: &str) -> String {
    let lower = s.to_ascii_lowercase();
    let mut out = String::new();
    let mut at = 0;
    while let Some(j) = lower[at..].find("bearer ").or_else(|| lower[at..].find("basic ")) {
        let start = at + j + if lower[at + j..].starts_with("bearer") { 7 } else { 6 };
        let run = s[start..].chars().take_while(|c| word_char(*c)).map(char::len_utf8).sum::<usize>();
        out += &s[at..start];
        if run >= 8 {
            out += "<redacted>";
        } else {
            out += &s[start..start + run];
        }
        at = start + run;
    }
    out + &s[at..]
}

fn redact_keyed(s: &str) -> String {
    let lower = s.to_ascii_lowercase();
    let bytes = s.as_bytes();
    let mut out = String::with_capacity(s.len());
    let mut at = 0;
    'scan: while at < s.len() {
        for key in SECRET_WORDS {
            if lower[at..].starts_with(key) && (at == 0 || !bytes[at - 1].is_ascii_alphanumeric()) {
                let mut j = at + key.len();
                // (the key is the whole word: `tokens` and `hf_token_path` are not it)
                if j < s.len() && (bytes[j].is_ascii_alphanumeric() || bytes[j] == b'_') {
                    continue;
                }
                let sep_start = j;
                let mut plist = false;
                // (a property list's `<key>NAME</key>` and its `<string>VALUE</string>`)
                if s[j..].starts_with("</key>") {
                    j += 6;
                    while j < s.len() && bytes[j].is_ascii_whitespace() {
                        j += 1;
                    }
                    if s[j..].starts_with("<string>") {
                        j += 8;
                        plist = true;
                    }
                }
                while j < s.len() && matches!(bytes[j], b'=' | b':' | b' ' | b'"' | b'\'' | b'\\') {
                    j += 1;
                }
                let value = s[j..]
                    .chars()
                    .take_while(|c| !c.is_whitespace() && !matches!(c, '"' | '\'' | ',' | '\\' | '}' | ']' | ';' | '&') && !(plist && *c == '<'))
                    .map(char::len_utf8)
                    .sum::<usize>();
                let v = &s[j..j + value];
                let plain = v.is_empty()
                    // (only a value that is already wholly redacted, or is the marker `<set>`: a secret that merely contains or sits beside one is redacted)
                    || v == "<redacted>"
                    || v == "<set>"
                    || TOKEN_PREFIXES.iter().any(|p| v.strip_prefix(p) == Some("<redacted>"))
                    || matches!(v, "null" | "true" | "false" | "none" | "found")
                    || v.eq_ignore_ascii_case("bearer")
                    || v.eq_ignore_ascii_case("basic")
                    || v.chars().all(|c| c.is_ascii_digit());
                if j > sep_start
                    && !plain
                    && (key != "token" || value >= 8)
                    && (s[sep_start..j].contains(['=', ':', ' ']) || s[sep_start..j].starts_with("</key>"))
                {
                    out += &s[at..j];
                    out += "<redacted>";
                    at = j + value;
                    continue 'scan;
                }
            }
        }
        let c = s[at..].chars().next().unwrap_or(' ');
        out.push(c);
        at += c.len_utf8();
    }
    out
}

/// The tokens and keys this machine holds, by value: the bundle takes each out wherever it turns up.
fn secrets() -> Vec<String> {
    let mut v: Vec<String> = ["GLYD_API_KEY", "VLLM_API_KEY", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"].iter().filter_map(|k| std::env::var(k).ok()).collect();
    v.extend(hub::token());
    v
}

/// The environment's names the bundle lists, with the value of each (a name that holds a secret says only that it is set).
fn environment() -> Vec<(String, String)> {
    let mut v: Vec<(String, String)> = std::env::vars()
        .filter(|(k, _)| {
            ["GLYD_", "HF_", "HUGGING", "CUDA", "NVIDIA", "XDG_", "HIP", "ROCM"].iter().any(|p| k.starts_with(p)) || k == "LD_LIBRARY_PATH" || k == "PATH"
        })
        .map(|(k, val)| {
            let up = k.to_ascii_uppercase();
            let secret = ["KEY", "TOKEN", "SECRET", "PASSWORD", "PASSWD"].iter().any(|w| up.contains(w)) && !up.ends_with("_PATH");
            (k, if secret { "<set>".to_string() } else { val })
        })
        .collect();
    v.sort();
    v
}

/// The last `n` lines of the newest `count` files of the service's log folder: (name, lines).
fn recent_logs(dir: &Path, count: usize, n: usize) -> Vec<(String, Vec<String>)> {
    let mut files: Vec<(std::time::SystemTime, PathBuf)> = std::fs::read_dir(dir)
        .map(|d| d.flatten().filter(|e| e.path().is_file()).filter_map(|e| Some((e.metadata().ok()?.modified().ok()?, e.path()))).collect())
        .unwrap_or_default();
    files.sort_by_key(|f| std::cmp::Reverse(f.0));
    files
        .into_iter()
        .take(count)
        .map(|(_, p)| {
            let text = std::fs::read_to_string(&p).unwrap_or_default();
            let lines: Vec<String> = text.lines().rev().take(n).map(str::to_string).collect::<Vec<_>>().into_iter().rev().collect();
            (p.file_name().map_or(String::new(), |n| n.to_string_lossy().into_owned()), lines)
        })
        .collect()
}

/// The service's unit or plist where one is installed (config a user may have changed), its text.
fn service_config() -> Option<(String, String)> {
    let home = PathBuf::from(crate::os::home());
    let candidates = [
        home.join(".config/systemd/user/glyd.service"),
        home.join("Library/LaunchAgents/com.getglyd.glyd.plist"),
        PathBuf::from(std::env::var_os("XDG_CONFIG_HOME").unwrap_or_default()).join("systemd/user/glyd.service"),
    ];
    candidates.iter().find_map(|p| Some((p.display().to_string(), std::fs::read_to_string(p).ok()?)))
}

/// The bundle as JSON text (`glyd-doctor/1`), redacted: the versions, the checks with their fixes, the memory terms, the sample plan, the failure matrix, the
/// environment, the service's configuration and the end of its recent logs.
#[allow(clippy::too_many_arguments)]
pub fn bundle(
    f: &Facts,
    cs: &[Check],
    logs: Option<Vec<(String, Vec<String>)>>,
    config: Option<(String, String)>,
    env: Vec<(String, String)>,
    home: &str,
    secrets: &[String],
) -> String {
    let m = f.machine();
    let plan = match sample_plan(&m) {
        Ok(p) => serde_json::from_str(&p.json(SAMPLE_MODEL)).unwrap_or_default(),
        Err(why) => serde_json::json!({"schema": plan::SCHEMA, "model": SAMPLE_MODEL, "plan": null, "why": why}),
    };
    let gpus: Vec<serde_json::Value> = f
        .gpus
        .as_ref()
        .map(|all| {
            all.iter()
                .map(|g| {
                    serde_json::json!({
                        "ordinal": g.ordinal, "name": g.name, "compute_capability": format!("{}.{}", g.cc.0, g.cc.1), "driver_cuda": format!("{}.{}", g.driver / 1000, g.driver % 1000 / 10),
                        "total_bytes": g.total, "free_bytes": g.free, "reserve_bytes": plan::reserve_of(g.free), "lossy_work_bytes": plan::LOSSY_WORK,
                        "dynamic_work_bytes": plan::dynamic_work_of(g.free),
                    })
                })
                .collect()
        })
        .unwrap_or_default();
    let list = |v: &[Cached]| v.iter().map(|c| serde_json::json!({"name": c.name, "bytes": c.bytes})).collect::<Vec<_>>();
    let v = serde_json::json!({
        "schema": SCHEMA,
        "glyd_engine": VERSION,
        "os": f.os, "arch": f.arch,
        "ok": !cs.iter().any(|c| c.status == Status::Fail),
        "checks": cs.iter().map(|c| serde_json::json!({"id": c.id, "label": c.label, "status": c.status.word(), "found": c.found, "fix": c.fix})).collect::<Vec<_>>(),
        "gpus": gpus,
        "gpu_error": f.gpus.as_ref().err(),
        "ram": f.ram.map(|(t, a)| serde_json::json!({"total_bytes": t, "available_bytes": a, "host_margin_bytes": HOST_MARGIN})),
        "sample_plan": plan,
        "listeners": f.listeners.iter().map(|l| serde_json::json!({"port": l.port, "role": l.role, "held_by": l.held_by})).collect::<Vec<_>>(),
        "models": {"hub": list(&f.hub_models), "packed": list(&f.packed_models)},
        "matrix": MATRIX.iter().map(|x| serde_json::json!({"class": x.class, "check": x.check, "fix": x.fix})).collect::<Vec<_>>(),
        "environment": env.iter().map(|(k, val)| serde_json::json!({"name": k, "value": val})).collect::<Vec<_>>(),
        "service_config": config.map(|(p, t)| serde_json::json!({"path": p, "text": t})),
        "logs_included": logs.is_some(),
        "logs": logs.unwrap_or_default().into_iter().map(|(n, lines)| serde_json::json!({"file": n, "last_lines": lines})).collect::<Vec<_>>(),
    });
    redact_with(&serde_json::to_string_pretty(&v).unwrap_or_default(), home, secrets)
}

/// `glyd doctor [--json] [--bundle [FILE]]`: the checks on the terminal; with `--json` the bundle on standard output instead; with `--bundle` the bundle in a
/// file (`glyd-bundle-UNIXTIME.json` here where none is named) and where it is. Exit 0 where everything a run needs is here, else 1.
pub fn run(a: &Args) -> i32 {
    let f = Facts::probe();
    let cs = checks(&f);
    let (words, code) = text(&cs);
    let want_file = a.flag("bundle").is_some() || a.has("bundle");
    if a.has("json") || want_file {
        let home = crate::os::home().to_string_lossy().into_owned();
        let logs = (!a.has("no-logs")).then(|| recent_logs(&crate::service::log_dir(), 3, 40));
        let json = bundle(&f, &cs, logs, service_config(), environment(), &home, &secrets());
        if want_file {
            let name = match a.flag("bundle") {
                Some(p) if !p.is_empty() => p.to_string(),
                _ => format!("glyd-bundle-{}.json", std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs())),
            };
            if let Err(e) = std::fs::write(&name, &json) {
                eprintln!("glyd doctor: {name}: {e}");
                return 1;
            }
            print!("{words}");
            println!(
                "Support bundle written to {name}: attach it to the issue. It holds the checks above, the environment (GLYD_*, HF_*, CUDA_*, PATH), the service's configuration{}. Tokens, keys, credentials in addresses and your home folder are taken out of it; read it before you post it.",
                if a.has("no-logs") { " (no logs: --no-logs)" } else { " and the last 40 lines of its 3 newest logs (--no-logs leaves them out)" }
            );
        } else {
            println!("{json}");
        }
        return code;
    }
    print!("{words}");
    code
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gpu(name: &str, cc: (i32, i32), total_gib: u64, free_gib: u64, driver: i32) -> Gpu {
        Gpu { name: name.into(), cc, sms: 0, total: total_gib << 30, free: free_gib << 30, driver, count: 1, ordinal: 0 }
    }

    /// A machine where everything is fine: an RTX 4080 SUPER, CUDA 13 driver, 56 GiB of memory, room on the disks, the Hub there, the ports free.
    fn healthy() -> Facts {
        Facts {
            os: "linux".into(),
            arch: "x86_64".into(),
            gpus: Ok(vec![gpu("NVIDIA GeForce RTX 4080 SUPER", (8, 9), 16, 15, 13000)]),
            library: Ok("/usr/lib/libglyd_gpu_cuda13.so".into()),
            cublas: Some("/usr/lib/libcublas.so.13".into()),
            ram: Some((64 << 30, 56 << 30)),
            cache: Dir { path: "/home/u/.cache/glyd".into(), free: Some(500 << 30) },
            models: Dir { path: "/home/u/.cache/huggingface/hub".into(), free: Some(500 << 30) },
            hub_token: true,
            hub: HubState::Reachable("https://huggingface.co".into()),
            listeners: vec![
                Listener { port: 11434, role: "the service (glyd service)", held_by: None },
                Listener { port: 8000, role: "a server (glyd serve; the next free port up to 8020 where this is taken)", held_by: None },
            ],
            hub_models: vec![Cached { name: "google/gemma-4-12B-it".into(), bytes: 24 << 30 }],
            packed_models: vec![],
            hip: false,
            wddm: false,
        }
    }

    fn get<'a>(cs: &'a [Check], id: &str) -> Vec<&'a Check> {
        cs.iter().filter(|c| c.id == id).collect()
    }

    #[test]
    fn a_healthy_machine_has_nothing_to_fix() {
        let cs = checks(&healthy());
        assert!(cs.iter().all(|c| c.status != Status::Fail), "{cs:#?}");
        let (words, code) = text(&cs);
        assert_eq!(code, 0, "{words}");
        assert!(words.starts_with("glyd engine ") && words.ends_with("Everything glyd run needs is here.\n"), "{words}");
        // (the sample plan: Gemma 4 12B loads on a 16 GB card at a lossy level, with its numbers)
        let plan = get(&cs, "plan")[0];
        assert!(plan.status == Status::Ok && plan.found.contains("tokens of context") && plan.found.contains("gemma-4-12B-it"), "{plan:?}");
        assert!(get(&cs, "backend-cuda")[0].found.contains("accepted"));
        assert!(get(&cs, "budget")[0].found.contains("for the weights and the cache"));
    }

    /// The install matrix: every failure class has a check that fails for the facts that cause it, with the matrix's one-line fix, and every check the matrix
    /// names is made.
    #[test]
    fn every_class_of_the_failure_matrix_is_diagnosed_with_its_fix() {
        let fixes = |cs: &[Check], id: &str| get(cs, id).into_iter().find(|c| c.status != Status::Ok).and_then(|c| c.fix.clone());
        let fix_of = |id: &str| MATRIX.iter().find(|x| x.check == id).unwrap().fix.to_string();
        let mut seen = std::collections::BTreeSet::new();
        let mut case = |name: &str, f: Facts, id: &'static str, status: Status| {
            let cs = checks(&f);
            let c = get(&cs, id).into_iter().find(|c| c.status == status).unwrap_or_else(|| panic!("{name}: no {id} check of status {status:?} in {cs:#?}"));
            if id == "machine" {
                // (the refusal's own fix: `run` says the same)
                assert!(c.fix.as_ref().is_some_and(|x| !x.is_empty()), "{name}: the fix");
            } else {
                assert_eq!(c.fix.as_deref(), Some(fix_of(id).as_str()), "{name}: the fix");
            }
            assert!(!c.found.is_empty(), "{name}");
            assert_eq!(fixes(&cs, id), c.fix);
            if status == Status::Fail {
                assert_eq!(text(&cs).1, 1, "{name}");
            }
            seen.insert(id);
        };
        let mut f = healthy();
        f.gpus = Err("no CUDA driver found (libcuda.so.1: cannot open shared object file)".into());
        case("no driver", f, "gpu", Status::Fail);
        let mut f = healthy();
        f.gpus = Ok(vec![gpu("NVIDIA GeForce GTX 1080", (6, 1), 8, 7, 13000)]);
        case("pascal", f, "gpu-age", Status::Fail);
        let mut f = healthy();
        f.gpus = Ok(vec![gpu("NVIDIA GeForce RTX 4080 SUPER", (8, 9), 16, 15, machine::driver_min((8, 9)) - 1)]);
        let cs = checks(&f);
        assert!(get(&cs, "driver")[0].status == Status::Fail && get(&cs, "driver")[0].found.contains("too old"), "{cs:#?}");
        assert!(get(&cs, "driver")[0].fix.is_some());
        case("old driver", f, "driver", Status::Fail);
        // a GPU under Windows' WDDM is said so, with what it does and the setting that stops it; a machine that is not says nothing
        let mut f = healthy();
        f.wddm = true;
        let cs = checks(&f);
        let w = get(&cs, "wddm");
        assert!(
            w.len() == 1
                && w[0].status == Status::Note
                && w[0].found.contains("WDDM")
                && w[0].found.contains("without an error")
                && w[0].fix.as_deref().is_some_and(|x| x.contains("Sysmem Fallback Policy")),
            "{cs:#?}"
        );
        assert!(get(&checks(&healthy()), "wddm").is_empty());
        let mut f = healthy();
        f.library = Err("libglyd_gpu_cuda13.so is not beside the program".into());
        case("no library", f, "gpu-library", Status::Fail);
        let mut f = healthy();
        f.arch = "aarch64".into();
        case("arm", f, "machine", Status::Fail);
        let mut f = healthy();
        f.ram = Some((8 << 30, 5 << 30));
        case("little memory", f, "ram", Status::Note);
        let mut f = healthy();
        f.cache.free = Some(2 << 30);
        case("full cache disk", f, "cache", Status::Note);
        let mut f = healthy();
        f.models.free = Some(2 << 30);
        case("full models disk", f, "models", Status::Note);
        let mut f = healthy();
        f.hub_token = false;
        case("no token", f, "hub-token", Status::Note);
        let mut f = healthy();
        f.hub = HubState::Unreachable("https://huggingface.co".into(), "dns error: failed to lookup address".into());
        case("hub down", f, "hub", Status::Fail);
        let mut f = healthy();
        f.gpus = Ok(vec![gpu("NVIDIA GeForce RTX 4060 Laptop GPU", (8, 9), 8, 3, 13000)]);
        case("a card too small for the sample", f, "plan", Status::Note);
        let mut f = healthy();
        f.listeners[0].held_by = Some("ollama (pid 812)".into());
        case("service port taken", f.clone(), "service-port", Status::Note);
        assert!(get(&checks(&f), "service-port")[0].found.contains("ollama (pid 812)"));
        let mut f = healthy();
        f.listeners[1].held_by = Some("python (pid 9)".into());
        case("server port taken", f, "server-port", Status::Note);
        for x in MATRIX {
            assert!(seen.contains(x.check), "the matrix names {}: no case makes it", x.check);
        }
        // (and each failing check is a failing exit)
        let mut amd = healthy();
        amd.hip = true;
        amd.gpus = Ok(vec![gpu("AMD Radeon RX 7900 XTX", (11, 0), 24, 23, 0)]);
        let cs = checks(&amd);
        assert!(cs.iter().any(|c| c.status == Status::Fail), "{cs:#?}");
        assert!(get(&cs, "backend-rocm")[0].found.contains("rejected") || get(&cs, "backend-rocm")[0].found.contains("accepted"));
    }

    #[test]
    #[cfg(not(kernels12))] // (CUDA 12.7 is a rejected driver only for a build with CUDA 13's kernels alone)
    fn the_backends_say_detected_version_and_why_rejected() {
        let mut f = healthy();
        f.gpus = Ok(vec![gpu("NVIDIA GeForce RTX 4080 SUPER", (8, 9), 16, 15, 12070)]);
        let cs = checks(&f);
        let cuda = get(&cs, "backend-cuda")[0];
        assert!(cuda.found.contains("detected (driver CUDA 12.7)") && cuda.found.contains("rejected: the driver supports CUDA 12.7"), "{cuda:?}");
        assert!(get(&cs, "backend-metal")[0].found.contains("not detected") && get(&cs, "backend-cpu")[0].found.contains("not built"));
        f.os = "macos".into();
        f.gpus = Err("no CUDA driver".into());
        assert!(get(&checks(&f), "backend-metal")[0].found.contains("rejected: Apple GPUs are not supported yet"));
    }

    #[test]
    fn the_memory_terms_are_the_plans_constants() {
        let f = healthy();
        let g = &f.gpus.as_ref().unwrap()[0];
        let b = get(&checks(&f), "budget")[0].found.clone();
        let a_gib = |n: u64| format!("{:.1} GiB", n as f64 / (1u64 << 30) as f64);
        assert!(b.contains(&format!("the engine keeps {}", a_gib(plan::reserve_of(g.free)))), "{b}");
        assert!(b.contains(&format!("its work takes {}", a_gib(plan::LOSSY_WORK))), "{b}");
        let ram = get(&checks(&f), "ram")[0].found.clone();
        assert!(ram.contains("56.0 GiB available of 64.0 GiB") && ram.contains("6.0 GiB kept for the system: 50.0 GiB"), "{ram}");
        assert_eq!(parse_ram("MemTotal:  65536000 kB\nMemFree: 1 kB\nMemAvailable:  32768000 kB\n"), Some((65536000 * 1024, 32768000 * 1024)));
        assert_eq!(parse_ram("MemTotal: 1 kB\n"), None);
    }

    #[test]
    fn redaction_takes_out_tokens_keys_and_the_home_folder() {
        let home = "/Users/alice";
        let cases = [
            ("token hf_abcdefghijklmnopqrstuvwxyz1234 here", "token hf_<redacted> here"),
            ("export HF_TOKEN=hf_AAAAAAAAAAAAAAAAAAAA", "export HF_TOKEN=hf_<redacted>"),
            ("glyd serve --api-key s3cr3t-value --port 8000", "glyd serve --api-key <redacted> --port 8000"),
            ("Authorization: Bearer abcdef123456789", "Authorization: Bearer <redacted>"),
            (r#"{"api_key": "sk-live-1234567890abcdef", "n": 1}"#, r#"{"api_key": "sk-<redacted>", "n": 1}"#),
            ("sk-1234567890abcdefghij is a key", "sk-<redacted> is a key"),
            ("/Users/alice/.cache/glyd and /Users/alice/models", "~/.cache/glyd and ~/models"),
            ("password=hunter2", "password=<redacted>"),
            ("password=pa<ss>word123 next", "password=<redacted> next"),
            ("password=<redacted>tail-secret1", "password=<redacted>"),
            ("api_key=<x>secretvalue1", "api_key=<redacted>"),
            ("GLYD_API_KEY=<set>", "GLYD_API_KEY=<set>"),
            ("GLYD_API_KEY=abcdefghijkl in the log", "GLYD_API_KEY=<redacted> in the log"),
            ("Environment=VLLM_API_KEY=s3cr3tvalue", "Environment=VLLM_API_KEY=<redacted>"),
            ("<key>HF_TOKEN</key>\n  <string>plainsecret</string>", "<key>HF_TOKEN</key>\n  <string><redacted></string>"),
            ("HF_ENDPOINT=https://user:p4ss@mirror.example.com/x", "HF_ENDPOINT=https://<redacted>@mirror.example.com/x"),
            ("http://proxy:3128 is fine and https://a@b.c too", "http://proxy:3128 is fine and https://<redacted>@b.c too"),
            // (not secrets: a count of tokens, a path to a token file, a number)
            ("12 tokens: 4096", "12 tokens: 4096"),
            ("HF_TOKEN_PATH=/tmp/t", "HF_TOKEN_PATH=/tmp/t"),
            ("token: 12345", "token: 12345"),
        ];
        for (input, want) in cases {
            assert_eq!(redact(input, home), want, "{input}");
        }
        // (JSON escaping and a Windows home)
        assert_eq!(redact(r#""path": "C:\\Users\\bob\\glyd""#, r"C:\Users\bob"), r#""path": "~\\glyd""#);
    }

    #[test]
    fn the_bundle_is_one_redacted_json_with_everything_a_report_needs() {
        let f = healthy();
        let cs = checks(&f);
        let logs = vec![("service-gemma.log".to_string(), vec!["loading".to_string(), "HF_TOKEN=hf_SECRETSECRETSECRET1234 at /home/u/x".to_string()])];
        let config = Some(("/home/u/.config/systemd/user/glyd.service".to_string(), "ExecStart=/home/u/glyd service --api-key hunter2hunter2\n".to_string()));
        let env = vec![("GLYD_API_KEY".to_string(), "<set>".to_string()), ("HF_HOME".to_string(), "/home/u/hf".to_string())];
        let text = bundle(&f, &cs, Some(logs), config, env, "/home/u", &["live-key-0123456789".to_string()]);
        assert!(!text.contains("hf_SECRET") && !text.contains("hunter2") && !text.contains("/home/u/"), "{text}");
        // (a secret this machine holds is out wherever it is, as is a credential in an address)
        let mut g = healthy();
        g.hub = HubState::Unreachable("https://user:pw123@mirror.example.com".into(), "refused with live-key-0123456789".into());
        let other = bundle(&g, &checks(&g), None, None, vec![], "/home/u", &["live-key-0123456789".to_string()]);
        assert!(!other.contains("pw123") && !other.contains("live-key-0123456789") && other.contains("mirror.example.com"), "{other}");
        let o: serde_json::Value = serde_json::from_str(&other).unwrap();
        assert_eq!((o["logs_included"].as_bool(), o["logs"].as_array().unwrap().len(), o["ok"].as_bool()), (Some(false), 0, Some(false)));
        let v: serde_json::Value = serde_json::from_str(&text).unwrap();
        for k in [
            "schema",
            "glyd_engine",
            "os",
            "arch",
            "ok",
            "checks",
            "gpus",
            "ram",
            "sample_plan",
            "listeners",
            "models",
            "matrix",
            "environment",
            "service_config",
            "logs",
        ] {
            assert!(v.get(k).is_some(), "{k} is in glyd-doctor/1");
        }
        assert_eq!((v["schema"].as_str(), v["ok"].as_bool()), (Some("glyd-doctor/1"), Some(true)));
        assert_eq!(v["checks"].as_array().unwrap().len(), cs.len());
        assert_eq!(v["sample_plan"]["schema"].as_str(), Some("glyd-plan/1"));
        assert_eq!(v["matrix"].as_array().unwrap().len(), MATRIX.len());
        assert!(v["gpus"][0]["reserve_bytes"].as_u64().is_some());
        assert_eq!(v["logs"][0]["last_lines"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn recent_logs_are_the_last_lines_of_the_newest_files() {
        let dir = std::env::temp_dir().join(format!("glyd-doctor-logs-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("a.log"), (0..100).map(|i| format!("line {i}\n")).collect::<String>()).unwrap();
        let logs = recent_logs(&dir, 3, 5);
        assert_eq!(logs.len(), 1);
        assert_eq!((logs[0].0.as_str(), logs[0].1.len(), logs[0].1[4].as_str()), ("a.log", 5, "line 99"));
        assert!(recent_logs(&dir.join("none"), 3, 5).is_empty());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn the_models_in_the_caches_are_listed_by_name_and_size() {
        let root = std::env::temp_dir().join(format!("glyd-doctor-models-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let (hubd, packed) = (root.join("hub"), root.join("engine"));
        std::fs::create_dir_all(hubd.join("models--google--gemma-4-12B-it/blobs")).unwrap();
        std::fs::write(hubd.join("models--google--gemma-4-12B-it/blobs/abc"), vec![0u8; 1000]).unwrap();
        std::fs::create_dir_all(packed.join("gemma-wren")).unwrap();
        std::fs::write(packed.join("gemma-wren/x.safetensors"), vec![0u8; 500]).unwrap();
        let (h, p) = cached_models(&hubd, &packed);
        assert_eq!((h[0].name.as_str(), h[0].bytes, p[0].name.as_str(), p[0].bytes), ("google/gemma-4-12B-it", 1000, "gemma-wren", 500));
        std::fs::remove_dir_all(&root).unwrap();
    }
}
