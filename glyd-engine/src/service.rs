//! The background service: `glyd service` owns the API port, loads a model on the first request that names it and unloads it when it has been idle for the keep-alive,
//! with several models held at once (the least recently used idle one makes room).
//!
//! It is a supervisor, not a second server: each model is `glyd serve MODEL` as a child process on a loopback port of its own (the server's code, constrained decoding and
//! all, runs unchanged), and the service passes the request through to it and streams the answer back. A model leaves the GPU by its process ending, which frees all of its
//! memory. The service answers `/health`, `/v1/models`, `/v1/glyd/service` (what `glyd ps` shows) and `/v1/glyd/unload` (`glyd stop`) itself.
//!
//! Secure by default: it listens on 127.0.0.1 and refuses a Host that is not a loopback name and an Origin that is not its own (as `glyd serve` does); `--host` is the user's
//! word to open it (with the same warning); the API key (GLYD_API_KEY) guards /v1; and it loads only models that are downloaded already (`glyd pull`): a request cannot make it
//! fetch a model or read a path.

use crate::args::Args;
use crate::models::{self, Manifest};
use crate::store;
use std::collections::HashSet;
use std::io::{BufRead, Read, Write};
use std::net::{Shutdown, SocketAddr, TcpListener, TcpStream, ToSocketAddrs};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

/// The port the service has unless it is told another: fixed, so that a client and `glyd run` find it (11434 is also where Ollama's clients look). A port in use is an error that
/// names the program, never a move to another port.
pub const DEFAULT_PORT: u16 = 11434;
/// How long a model stays loaded after its last request.
pub const DEFAULT_KEEP_ALIVE: Duration = Duration::from_secs(300);
/// How many models the service holds at once (the least recently used idle one is unloaded to make room).
pub const DEFAULT_MAX_LOADED: usize = 2;
/// How long a request waits for a place where every loaded model is in use.
const DEFAULT_WAIT_FOR_ROOM: Duration = Duration::from_secs(30);
/// How long a model may take to load before the request that waits for it is answered with a refusal (a 100 GB model on a slow disk takes long).
const LOAD_TIMEOUT: Duration = Duration::from_secs(1800);
/// The largest request body the service takes (images go in as base64).
const MAX_BODY: usize = 128 << 20;
const MAX_HEAD: usize = 64 << 10;

// ---- small pure pieces ---------------------------------------------------------------------------------------------------------------------------------------

/// `5m`, `90s`, `2h`, `1h30m`, `0` (unload as soon as the request is done), a bare number (seconds); `never` or `-1`: None (kept until the service stops).
pub fn parse_duration(s: &str) -> Result<Option<Duration>, String> {
    let t = s.trim().to_ascii_lowercase();
    if t == "never" || t == "-1" || t == "inf" {
        return Ok(None);
    }
    let bad = || format!("{s}: a time as 90s, 5m, 2h or 1h30m, 0 (unload at once) or never");
    if t.is_empty() {
        return Err(bad());
    }
    if let Ok(n) = t.parse::<u64>() {
        return Ok(Some(Duration::from_secs(n)));
    }
    let (mut total, mut num) = (0u64, String::new());
    for c in t.chars() {
        if c.is_ascii_digit() {
            num.push(c);
            continue;
        }
        let n: u64 = num.parse().map_err(|_| bad())?;
        num.clear();
        total += n * match c {
            's' => 1,
            'm' => 60,
            'h' => 3600,
            _ => return Err(bad()),
        };
    }
    if !num.is_empty() {
        return Err(bad());
    }
    Ok(Some(Duration::from_secs(total)))
}

/// A time for a person: `4m12s`, `3h`.
fn human(d: Duration) -> String {
    let s = d.as_secs();
    match s {
        0..=59 => format!("{s}s"),
        60..=3599 => format!("{}m{:02}s", s / 60, s % 60),
        _ => format!("{}h{:02}m", s / 3600, s % 3600 / 60),
    }
}

/// The program named in `lsof -Fpc` output (`p1234\ncollama\n`).
pub fn parse_lsof(out: &str) -> Option<String> {
    let (mut pid, mut name) = (None, None);
    for l in out.lines() {
        match l.split_at_checked(1) {
            Some(("p", r)) if pid.is_none() => pid = Some(r.to_string()),
            Some(("c", r)) if name.is_none() => name = Some(r.to_string()),
            _ => {}
        }
    }
    Some(format!("{} (pid {})", name?, pid?))
}

/// The program named in `ss -ltnpH` output (`LISTEN 0 4096 127.0.0.1:11434 0.0.0.0:* users:(("ollama",pid=812,fd=3))`).
pub fn parse_ss(out: &str) -> Option<String> {
    let u = out.lines().find_map(|l| l.split_once("users:((\"")?.1.into())?;
    let (name, rest) = u.split_once('"')?;
    let pid: String = rest.split_once("pid=")?.1.chars().take_while(char::is_ascii_digit).collect();
    Some(format!("{name} (pid {pid})"))
}

/// The program that listens on `port`, where the system says (lsof, else ss; Windows: netstat and tasklist).
pub fn port_owner(port: u16) -> Option<String> {
    let run = |prog: &str, args: &[&str]| {
        std::process::Command::new(prog)
            .args(args)
            .stderr(std::process::Stdio::null())
            .output()
            .ok()
            .filter(|o| o.status.success() || !o.stdout.is_empty())
            .map(|o| String::from_utf8_lossy(&o.stdout).into_owned())
    };
    if cfg!(windows) {
        let net = run("netstat", &["-ano", "-p", "TCP"])?;
        let pid = net
            .lines()
            .find(|l| l.contains("LISTENING") && l.split_whitespace().nth(1).is_some_and(|a| a.ends_with(&format!(":{port}"))))?
            .split_whitespace()
            .last()?
            .to_string();
        let t = run("tasklist", &["/FI", &format!("PID eq {pid}"), "/FO", "CSV", "/NH"])?;
        let name = t.split(',').next()?.trim_matches(['"', ' ', '\r', '\n']).to_string();
        return Some(format!("{name} (pid {pid})"));
    }
    run("lsof", &["-nP", &format!("-iTCP:{port}"), "-sTCP:LISTEN", "-Fpc"])
        .and_then(|o| parse_lsof(&o))
        .or_else(|| run("ss", &["-ltnpH", "sport", "=", &format!(":{port}")]).and_then(|o| parse_ss(&o)))
}

/// Where the service's logs are (a model's worker writes one file each).
pub fn log_dir() -> PathBuf {
    let var = |k: &str| std::env::var_os(k).filter(|v| !v.is_empty()).map(PathBuf::from);
    var("XDG_STATE_HOME").unwrap_or_else(|| Path::new(&crate::os::home()).join(".local").join("state")).join("glyd").join("logs")
}

fn api_key() -> Option<String> {
    ["GLYD_API_KEY", "VLLM_API_KEY"].iter().find_map(|k| std::env::var(k).ok()).filter(|k| !k.is_empty())
}

fn key_ok(key: &Option<String>, auth: Option<&str>) -> bool {
    use sha2::{Digest, Sha256};
    let Some(key) = key else { return true };
    let Some((scheme, token)) = auth.and_then(|a| a.split_once(' ')) else { return false };
    let (got, want): ([u8; 32], [u8; 32]) = (Sha256::digest(token.as_bytes()).into(), Sha256::digest(key.as_bytes()).into());
    scheme.eq_ignore_ascii_case("bearer") && got.iter().zip(&want).fold(0u8, |d, (a, b)| d | (a ^ b)) == 0
}

// ---- who is the service: a secret of this account's --------------------------------------------------------------------------------------------------------

/// Where the service keeps this account's secret (a file only this account can read): the service proves it has it, and a command that would send a key, or stop the service,
/// first asks for the proof. A program that took the port before the service did cannot give it, and a user of the computer who is not this account cannot ask for a stop.
fn secret_path() -> PathBuf {
    log_dir().parent().map_or_else(|| PathBuf::from("service.secret"), |d| d.join("service.secret"))
}

fn random_hex(bytes: usize) -> String {
    let mut b = vec![0u8; bytes];
    #[cfg(unix)]
    let got = std::fs::File::open("/dev/urandom").and_then(|mut f| f.read_exact(&mut b)).is_ok();
    #[cfg(not(unix))]
    let got = false;
    if !got {
        // (no system source: the keys of std's hasher, which the system seeds, one for each eight bytes)
        use std::hash::{BuildHasher, Hasher};
        for (i, c) in b.chunks_mut(8).enumerate() {
            let mut h = std::collections::hash_map::RandomState::new().build_hasher();
            h.write_usize(i);
            c.copy_from_slice(&h.finish().to_le_bytes()[..c.len()]);
        }
    }
    b.iter().map(|x| format!("{x:02x}")).collect()
}

/// The account's secret, made (mode 0600) where there is none yet.
fn ensure_secret() -> Result<String, String> {
    let path = secret_path();
    if let Some(s) = read_secret() {
        return Ok(s);
    }
    if let Some(d) = path.parent() {
        let _ = std::fs::create_dir_all(d);
    }
    let secret = random_hex(32);
    let mut o = std::fs::OpenOptions::new();
    o.write(true).create(true).truncate(true);
    #[cfg(unix)]
    std::os::unix::fs::OpenOptionsExt::mode(&mut o, 0o600);
    o.open(&path).and_then(|mut f| f.write_all(secret.as_bytes())).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(secret)
}

fn read_secret() -> Option<String> {
    std::fs::read_to_string(secret_path()).ok().map(|s| s.trim().to_string()).filter(|s| s.len() >= 32)
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

fn sha_hex(s: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    hex(&Sha256::digest(s))
}

/// HMAC-SHA256 (RFC 2104).
fn hmac_sha256(key: &[u8], msg: &[u8]) -> [u8; 32] {
    use sha2::{Digest, Sha256};
    let mut k = [0u8; 64];
    if key.len() > 64 {
        k[..32].copy_from_slice(&Sha256::digest(key));
    } else {
        k[..key.len()].copy_from_slice(key);
    }
    let (mut ipad, mut opad) = ([0x36u8; 64], [0x5cu8; 64]);
    for i in 0..64 {
        ipad[i] ^= k[i];
        opad[i] ^= k[i];
    }
    let inner = Sha256::new().chain_update(ipad).chain_update(msg).finalize();
    Sha256::new().chain_update(opad).chain_update(inner).finalize().into()
}

/// What the service on `port` answers to a challenge: it shows it has the secret without saying it, and the answer is for that port (one answer is no use at another).
fn proof(secret: &str, port: u16, challenge: &str) -> String {
    hex(&hmac_sha256(secret.as_bytes(), format!("glyd-health\0{port}\0{challenge}").as_bytes()))
}

fn now_secs() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs())
}

/// The header value that authorizes one control request (`stop`, `unload`) to the service on `port`: a MAC of the request and the time, which the secret itself is never sent for and
/// which is good for a minute and for this request alone.
fn control_auth(secret: &str, port: u16, method: &str, path: &str, body: &str, ts: u64) -> String {
    let msg = format!("glyd-control\0{port}\0{method}\0{path}\0{ts}\0{}", sha_hex(body.as_bytes()));
    format!("{ts}.{}", hex(&hmac_sha256(secret.as_bytes(), msg.as_bytes())))
}

/// Whether what answers on `addr` is the service of the account that has `secret`: it is asked for the proof of a challenge made now.
fn verified_with(addr: (&str, u16), secret: &str) -> bool {
    let challenge = random_hex(16);
    match get(addr, &format!("/health?challenge={challenge}"), None, Duration::from_millis(800)) {
        Ok((200, body)) => serde_json::from_str::<serde_json::Value>(&body)
            .ok()
            .and_then(|v| v["proof"].as_str().map(str::to_string))
            .is_some_and(|p| p == proof(secret, addr.1, &challenge)),
        _ => false,
    }
}

/// Whether the service of this account answers on `port`.
pub fn verified(port: u16) -> bool {
    read_secret().is_some_and(|s| verified_with(("127.0.0.1", port), &s))
}

// ---- the models the service can serve -----------------------------------------------------------------------------------------------------------------------

/// What a request's `model` came to: the repo and level as `glyd serve` takes them, and the name the service shows for it.
#[derive(Debug, Clone, PartialEq)]
pub struct Spec {
    /// `Qwen/Qwen3.5-9B` or `Qwen/Qwen3.5-9B:swift`: the model word of `glyd serve`, and the key of the loaded model.
    pub arg: String,
    /// The name `glyd ps` shows.
    pub name: String,
}

/// The model a request names, as the service can serve it: a short name or a Hub id that is downloaded. Err: (status, message) for the client.
pub fn spec_of(name: &str, manifest: &Manifest, local: &[models::Local]) -> Result<Spec, (u16, String)> {
    let not_found = |m: String| (404u16, m);
    if name.is_empty() {
        return Err(not_found("the request names no model (\"model\": a name from /v1/models)".into()));
    }
    let (repo, level, shown) = match models::resolve(name, manifest) {
        Ok(Some(r)) => (r.repo.clone(), r.level.clone(), r.name),
        Ok(None) => {
            let (repo, tag) = crate::levels::split_tag(name);
            if !store::looks_like_repo(repo) {
                return Err(not_found(format!(
                    "{name}: the service serves models by name (qwen3.5:9b) or Hub id (ORG/NAME), not by path: glyd serve DIR does that"
                )));
            }
            (repo.to_string(), tag, name.to_string())
        }
        Err(e) => return Err(not_found(e)),
    };
    if let Some(l) = &level {
        if crate::levels::by_name(l).is_none() {
            return Err(not_found(format!("{name}: :{l} is not a level. The levels: {}", crate::levels::names_now().join(", "))));
        }
    }
    // downloaded: the repo itself (a level of it is packed here), or, where that is not here, its pre-packed repo at the level, which is then the model served (by its own name: it is what is here)
    let packed = level.as_deref().and_then(crate::levels::by_name).filter(|l| l.layout.is_some()).map(|l| crate::levels::packed_repo(&repo, l.name));
    let here = |r: &str| local.iter().any(|m| m.repo == r);
    let arg = if here(&repo) {
        match &level {
            Some(l) => format!("{repo}:{l}"),
            None => repo,
        }
    } else if let Some(p) = packed.filter(|p| here(p)) {
        p
    } else {
        return Err(not_found(format!("{name} is not downloaded: glyd pull {name}")));
    };
    Ok(Spec { arg, name: shown })
}

// ---- the workers -----------------------------------------------------------------------------------------------------------------------------------------------

/// A loaded model: a program that serves it on a loopback port.
pub trait Handle: Send {
    fn port(&self) -> u16;
    /// Whether the program is still there.
    fn alive(&mut self) -> bool;
    /// Ends the program (its memory is the GPU's again when it has).
    fn stop(&mut self);
}

/// How a model is loaded: the real one starts `glyd serve`; a test's starts a stand-in.
pub trait Backend: Send + Sync {
    fn start(&self, spec: &Spec) -> Result<Box<dyn Handle>, String>;
}

/// `glyd serve MODEL` as a child process.
pub struct Processes {
    pub context: Option<String>,
    pub users: Option<String>,
}

struct Child {
    child: std::process::Child,
    port: u16,
}

impl Handle for Child {
    fn port(&self) -> u16 {
        self.port
    }
    fn alive(&mut self) -> bool {
        matches!(self.child.try_wait(), Ok(None))
    }
    fn stop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// The port a worker says it serves on (`NAME is served at http://127.0.0.1:41233: the chat page ...`): the worker binds port 0 itself, so no other program can have taken the port between a
/// choice and its bind.
pub fn served_port(log: &str) -> Option<u16> {
    let rest = log.lines().find_map(|l| l.split_once(" is served at http://127.0.0.1:").map(|(_, r)| r))?;
    rest.chars().take_while(char::is_ascii_digit).collect::<String>().parse().ok().filter(|p| *p != 0)
}

/// The last lines of a worker's log, for the refusal that says why it did not come up.
fn tail(path: &Path, lines: usize) -> String {
    let text = std::fs::read_to_string(path).unwrap_or_default();
    let v: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();
    v[v.len().saturating_sub(lines)..].join("\n")
}

impl Backend for Processes {
    fn start(&self, spec: &Spec) -> Result<Box<dyn Handle>, String> {
        let exe = std::env::current_exe().map_err(|e| e.to_string())?;
        let dir = log_dir();
        let _ = std::fs::create_dir_all(&dir);
        let stamp = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs());
        let log = dir.join(format!("service-{}-{stamp}.log", spec.arg.replace(['/', ':'], "-")));
        // (the log is the worker's alone: the port is read from it)
        let mut o = std::fs::OpenOptions::new();
        o.write(true).create(true).truncate(true);
        #[cfg(unix)]
        std::os::unix::fs::OpenOptionsExt::mode(&mut o, 0o600);
        let out = o.open(&log).map_err(|e| format!("{}: {e}", log.display()))?;
        let mut cmd = std::process::Command::new(&exe);
        cmd.arg("serve").arg(&spec.arg).arg("--port").arg("0");
        for (flag, v) in [("--context", &self.context), ("--users", &self.users)] {
            if let Some(v) = v {
                cmd.arg(flag).arg(v);
            }
        }
        // (the worker never attaches to a service, or starts one)
        // (the worker downloads nothing: what a request may load is what `glyd pull` fetched)
        cmd.env("GLYD_SERVICE", "0")
            .env("HF_HUB_OFFLINE", "1")
            .stdin(std::process::Stdio::null())
            .stdout(out.try_clone().map_err(|e| e.to_string())?)
            .stderr(out);
        let mut child = Child { child: cmd.spawn().map_err(|e| format!("{}: {e}", exe.display()))?, port: 0 };
        let t0 = Instant::now();
        loop {
            if let Ok(Some(status)) = child.child.try_wait() {
                return Err(format!("the worker for {} ended ({status}) before it was ready; its log, {}:\n{}", spec.arg, log.display(), tail(&log, 8)));
            }
            // (the worker opens its port when its model is loaded, and says which in its log)
            if let Some(port) = served_port(&std::fs::read_to_string(&log).unwrap_or_default()) {
                if get(("127.0.0.1", port), "/health", None, Duration::from_secs(2)).is_ok_and(|(code, _)| code == 200) {
                    child.port = port;
                    return Ok(Box::new(child));
                }
            }
            if t0.elapsed() > LOAD_TIMEOUT {
                child.stop();
                return Err(format!("{} did not load in {}; its log: {}", spec.arg, human(LOAD_TIMEOUT), log.display()));
            }
            std::thread::sleep(Duration::from_millis(250));
        }
    }
}

// ---- the pool --------------------------------------------------------------------------------------------------------------------------------------------------

struct Entry {
    spec: Spec,
    handle: Box<dyn Handle>,
    started: Instant,
    last_used: Instant,
    in_flight: usize,
    requests: u64,
}

#[derive(Default)]
struct State {
    entries: Vec<Entry>,
    /// Models being loaded now: a second request for one waits for the first's load.
    loading: HashSet<String>,
}

pub struct Config {
    pub keep_alive: Option<Duration>,
    pub max_loaded: usize,
    /// How long a request waits for a place where every loaded model is in use, before it is answered 503.
    pub wait_for_room: Duration,
}

/// A model held for a request: its port, until it is released.
pub struct Lease {
    pub port: u16,
    key: String,
}

pub struct Pool {
    backend: Box<dyn Backend>,
    state: Mutex<State>,
    changed: Condvar,
    cfg: Config,
    started: Instant,
}

/// The index of the entry to unload to make room: the least recently used one nothing is using.
fn victim(entries: &[(Instant, usize)]) -> Option<usize> {
    entries.iter().enumerate().filter(|(_, (_, in_flight))| *in_flight == 0).min_by_key(|(_, (t, _))| *t).map(|(i, _)| i)
}

impl Pool {
    pub fn new(backend: Box<dyn Backend>, cfg: Config) -> Pool {
        Pool { backend, state: Mutex::new(State::default()), changed: Condvar::new(), cfg, started: Instant::now() }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// The model for a request, loaded if it is not (one load at a time per model; the least recently used idle model is unloaded where `max_loaded` is held). Err: (status, why).
    pub fn acquire(&self, spec: &Spec) -> Result<Lease, (u16, String)> {
        let key = spec.arg.clone();
        let mut st = self.lock();
        loop {
            if let Some(i) = st.entries.iter().position(|e| e.spec.arg == key) {
                if st.entries[i].handle.alive() {
                    let e = &mut st.entries[i];
                    e.in_flight += 1;
                    e.last_used = Instant::now();
                    return Ok(Lease { port: e.handle.port(), key });
                }
                let mut dead = st.entries.remove(i);
                dead.handle.stop(); // (it ended by itself: the next request loads it again)
                continue;
            }
            if st.loading.contains(&key) {
                st = self.changed.wait(st).unwrap_or_else(|e| e.into_inner());
                continue;
            }
            st.loading.insert(key.clone());
            break;
        }
        // room: the least recently used idle models out while the pool is full
        let mut stopped = Vec::new();
        let wait_until = Instant::now() + self.cfg.wait_for_room;
        // (models being loaded count too: two different models that arrive together must not both pass for the last place)
        while st.entries.len() + st.loading.len() > self.cfg.max_loaded.max(1) {
            let view: Vec<(Instant, usize)> = st.entries.iter().map(|e| (e.last_used, e.in_flight)).collect();
            match victim(&view) {
                Some(i) => stopped.push(st.entries.remove(i)),
                // (every place is a model in use or one that is loading: wait for one to be done, then look again, for a while)
                None if Instant::now() < wait_until => {
                    st = self.changed.wait_timeout(st, wait_until - Instant::now()).unwrap_or_else(|e| e.into_inner()).0;
                }
                None => {
                    st.loading.remove(&key);
                    self.changed.notify_all();
                    return Err((503, format!("{} models are loaded and all are busy: try again when one has finished", st.entries.len())));
                }
            }
        }
        drop(st);
        for mut e in stopped {
            e.handle.stop();
        }
        let mut result = self.backend.start(spec);
        if result.is_err() {
            // (a load that fails may be for want of memory: the idle models go, and it is tried once more)
            let idle: Vec<Entry> = {
                let mut st = self.lock();
                let mut out = Vec::new();
                while let Some(i) = victim(&st.entries.iter().map(|e| (e.last_used, e.in_flight)).collect::<Vec<_>>()) {
                    out.push(st.entries.remove(i));
                }
                out
            };
            if !idle.is_empty() {
                for mut e in idle {
                    e.handle.stop();
                }
                result = self.backend.start(spec);
            }
        }
        let mut st = self.lock();
        st.loading.remove(&key);
        let out = match result {
            Ok(handle) => {
                let port = handle.port();
                let now = Instant::now();
                st.entries.push(Entry { spec: spec.clone(), handle, started: now, last_used: now, in_flight: 1, requests: 0 });
                Ok(Lease { port, key })
            }
            Err(why) => Err((503, why)),
        };
        self.changed.notify_all();
        out
    }

    /// The request is done with its model.
    pub fn release(&self, lease: Lease) {
        let mut st = self.lock();
        if let Some(e) = st.entries.iter_mut().find(|e| e.spec.arg == lease.key) {
            e.in_flight = e.in_flight.saturating_sub(1);
            e.last_used = Instant::now();
            e.requests += 1;
        }
        self.changed.notify_all();
    }

    /// The models idle for the keep-alive are unloaded; how many.
    pub fn reap(&self) -> usize {
        let Some(keep) = self.cfg.keep_alive else { return 0 };
        let gone: Vec<Entry> = {
            let mut st = self.lock();
            let now = Instant::now();
            let (gone, kept): (Vec<Entry>, Vec<Entry>) =
                std::mem::take(&mut st.entries).into_iter().partition(|e| e.in_flight == 0 && now.duration_since(e.last_used) >= keep);
            st.entries = kept;
            gone
        };
        let n = gone.len();
        for mut e in gone {
            e.handle.stop();
        }
        n
    }

    /// Unloads the model with this key (or name), or every idle one (`None`); how many went. A model in use is left.
    pub fn unload(&self, which: Option<&str>) -> usize {
        let gone: Vec<Entry> = {
            let mut st = self.lock();
            let (gone, kept): (Vec<Entry>, Vec<Entry>) =
                std::mem::take(&mut st.entries).into_iter().partition(|e| e.in_flight == 0 && which.map_or(true, |w| e.spec.arg == w || e.spec.name == w));
            st.entries = kept;
            gone
        };
        let n = gone.len();
        for mut e in gone {
            e.handle.stop();
        }
        n
    }

    /// Ends every model's program (the service is stopping).
    pub fn shutdown(&self) {
        let all: Vec<Entry> = std::mem::take(&mut self.lock().entries);
        for mut e in all {
            e.handle.stop();
        }
    }

    pub fn status_json(&self, port: u16) -> String {
        let st = self.lock();
        let now = Instant::now();
        let models: Vec<serde_json::Value> = st
            .entries
            .iter()
            .map(|e| {
                let idle = now.duration_since(e.last_used);
                serde_json::json!({
                    "name": e.spec.name, "model": e.spec.arg, "loaded_secs": now.duration_since(e.started).as_secs(), "idle_secs": idle.as_secs(),
                    "in_flight": e.in_flight, "requests": e.requests,
                    "unload_in_secs": self.cfg.keep_alive.filter(|_| e.in_flight == 0).map(|k| k.saturating_sub(idle).as_secs()),
                })
            })
            .collect();
        serde_json::json!({
            "service": "glyd", "version": env!("CARGO_PKG_VERSION"), "port": port, "up_secs": self.started.elapsed().as_secs(),
            "keep_alive_secs": self.cfg.keep_alive.map(|k| k.as_secs()), "max_loaded": self.cfg.max_loaded, "models": models,
        })
        .to_string()
    }

    pub fn loaded(&self) -> Vec<String> {
        self.lock().entries.iter().map(|e| e.spec.name.clone()).collect()
    }
}

// ---- HTTP --------------------------------------------------------------------------------------------------------------------------------------------------------

struct Request {
    method: String,
    path: String,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
}

impl Request {
    fn header(&self, name: &str) -> Option<&str> {
        self.headers.iter().find(|(k, _)| k.eq_ignore_ascii_case(name)).map(|(_, v)| v.as_str())
    }
}

/// A request read from `s`: its head (capped), and its body by Content-Length (capped; a chunked body is refused).
fn read_request(s: &mut TcpStream) -> Result<Request, (u16, String)> {
    // (the head within HEAD_TIMEOUT and the body within BODY_TIMEOUT of it, whole: a client that sends a byte a minute does not hold the connection; a write waits WRITE_TIMEOUT)
    s.set_write_timeout(Some(crate::server::WRITE_TIMEOUT)).ok();
    let head_until = std::time::Instant::now() + crate::server::HEAD_TIMEOUT;
    let mut buf = Vec::new();
    let mut byte = [0u8; 4096];
    let end = loop {
        if let Some(p) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
            break p;
        }
        if buf.len() > MAX_HEAD {
            return Err((431, "the request's head is too long".into()));
        }
        match crate::server::read_until_deadline(s, head_until, &mut byte) {
            Ok(0) => return Err((400, "the connection closed before the request was whole".into())),
            Ok(n) => buf.extend_from_slice(&byte[..n]),
            Err(e) => return Err(crate::server::read_failure(&e)),
        }
    };
    let head = String::from_utf8_lossy(&buf[..end]).into_owned();
    let mut lines = head.split("\r\n");
    let mut first = lines.next().unwrap_or("").split(' ');
    let (method, path) = (first.next().unwrap_or("").to_string(), first.next().unwrap_or("").to_string());
    if method.is_empty() || !path.starts_with('/') {
        return Err((400, "not an HTTP request".into()));
    }
    let headers: Vec<(String, String)> = lines.filter_map(|l| l.split_once(':').map(|(k, v)| (k.trim().to_string(), v.trim().to_string()))).collect();
    let mut req = Request { method, path, headers, body: Vec::new() };
    if req.header("transfer-encoding").is_some_and(|v| v.to_ascii_lowercase().contains("chunked")) {
        return Err((411, "a request body needs a Content-Length (chunked requests are not taken)".into()));
    }
    let len: usize = req.header("content-length").and_then(|v| v.parse().ok()).unwrap_or(0);
    // (curl sends this for a body over a kilobyte and waits a second for the answer)
    if req.header("expect").is_some_and(|v| v.eq_ignore_ascii_case("100-continue")) && len > 0 {
        let _ = s.write_all(b"HTTP/1.1 100 Continue\r\n\r\n");
    }
    if len > MAX_BODY {
        return Err((413, format!("the request body is over {} MB", MAX_BODY >> 20)));
    }
    let mut body = buf[end + 4..].to_vec();
    let body_until = std::time::Instant::now() + crate::server::BODY_TIMEOUT;
    while body.len() < len {
        let n = crate::server::read_until_deadline(s, body_until, &mut byte).map_err(|e| crate::server::read_failure(&e))?;
        if n == 0 {
            return Err((400, "the connection closed before the body was whole".into()));
        }
        body.extend_from_slice(&byte[..n]);
    }
    body.truncate(len);
    req.body = body;
    Ok(req)
}

fn reason(code: u16) -> &'static str {
    match code {
        200 => "OK",
        400 => "Bad Request",
        401 => "Unauthorized",
        403 => "Forbidden",
        404 => "Not Found",
        411 => "Length Required",
        408 => "Request Timeout",
        413 => "Payload Too Large",
        421 => "Misdirected Request",
        431 => "Request Header Fields Too Large",
        503 => "Service Unavailable",
        _ => "Error",
    }
}

fn respond(s: &mut TcpStream, code: u16, ctype: &str, body: &str) {
    let head = format!("HTTP/1.1 {code} {}\r\nContent-Type: {ctype}\r\nContent-Length: {}\r\nCache-Control: no-store\r\nX-Content-Type-Options: nosniff\r\nConnection: close\r\n\r\n", reason(code), body.len());
    let _ = s.write_all(head.as_bytes()).and_then(|_| s.write_all(body.as_bytes()));
    let _ = s.flush();
}

/// An OpenAI-style error answer.
fn error(s: &mut TcpStream, code: u16, why: &str) {
    let kind = match code {
        401 => "authentication_error",
        404 => "invalid_request_error",
        503 => "service_unavailable",
        _ => "invalid_request_error",
    };
    respond(s, code, "application/json", &serde_json::json!({"error": {"message": why, "type": kind, "code": code}}).to_string());
}

/// A GET to `addr` (the answer read to its end): (status, body).
pub fn get(addr: impl ToSocketAddrs, path: &str, key: Option<&str>, timeout: Duration) -> Result<(u16, String), String> {
    send(addr, "GET", path, None, key, timeout)
}

/// One request and its whole answer (the service and its workers close the connection after one).
pub fn send(addr: impl ToSocketAddrs, method: &str, path: &str, body: Option<&str>, key: Option<&str>, timeout: Duration) -> Result<(u16, String), String> {
    send_as(addr, method, path, body, key, None, timeout)
}

/// `send`, with a control request's authorization (`control_auth`) in a header.
pub fn send_as(
    addr: impl ToSocketAddrs,
    method: &str,
    path: &str,
    body: Option<&str>,
    key: Option<&str>,
    auth: Option<&str>,
    timeout: Duration,
) -> Result<(u16, String), String> {
    let a: SocketAddr = addr.to_socket_addrs().map_err(|e| e.to_string())?.next().ok_or("no address")?;
    let mut s = TcpStream::connect_timeout(&a, timeout).map_err(|e| e.to_string())?;
    s.set_read_timeout(Some(timeout)).ok();
    let mut req = format!("{method} {path} HTTP/1.1\r\nHost: {a}\r\nConnection: close\r\n");
    if let Some(k) = key {
        req.push_str(&format!("Authorization: Bearer {k}\r\n"));
    }
    if let Some(x) = auth {
        req.push_str(&format!("X-Glyd-Auth: {x}\r\n"));
    }
    match body {
        Some(b) => req.push_str(&format!("Content-Type: application/json\r\nContent-Length: {}\r\n\r\n{b}", b.len())),
        None => req.push_str("\r\n"),
    }
    s.write_all(req.as_bytes()).map_err(|e| e.to_string())?;
    let mut out = Vec::new();
    s.read_to_end(&mut out).map_err(|e| e.to_string())?;
    let text = String::from_utf8_lossy(&out).into_owned();
    let (head, body) = text.split_once("\r\n\r\n").ok_or("no answer")?;
    let code = head.split(' ').nth(1).and_then(|c| c.parse().ok()).ok_or("no status")?;
    Ok((code, body.to_string()))
}

/// What the proxy knows about itself.
struct Server {
    pool: Pool,
    manifest: Manifest,
    /// The models downloaded (the Hub's cache; a test's are given).
    local: Box<dyn Fn() -> Vec<models::Local> + Send + Sync>,
    key: Option<String>,
    /// This account's secret: what proves the service, and what a command to stop or unload carries.
    secret: String,
    local_only: bool,
    port: u16,
    stop: Arc<std::sync::atomic::AtomicBool>,
}

const PAGE: &str = "<!doctype html><meta charset=utf-8><title>glyd</title><body style=\"font:16px system-ui;max-width:40em;margin:3em auto\"><h1>glyd</h1><p>The service is up. It loads a model when a request names one and unloads it when it is idle.</p><p>Models: <a href=\"/v1/models\">/v1/models</a> &middot; what is loaded: <a href=\"/v1/glyd/service\">/v1/glyd/service</a>. Download one with <code>glyd pull qwen3</code>.</p>";

/// The `model` of a request body (JSON).
fn model_of(body: &[u8]) -> Option<String> {
    serde_json::from_slice::<serde_json::Value>(body).ok()?.get("model")?.as_str().map(str::to_string)
}

impl Server {
    /// Whether a control request carries the MAC of this account's secret, for this port, this request and a time within a minute.
    fn control_ok(&self, req: &Request, path: &str) -> bool {
        let Some((ts, mac)) = req.header("x-glyd-auth").and_then(|h| h.split_once('.')) else { return false };
        let Ok(t) = ts.parse::<u64>() else { return false };
        if now_secs().abs_diff(t) > 60 {
            return false;
        }
        let want = control_auth(&self.secret, self.port, &req.method, path, &String::from_utf8_lossy(&req.body), t);
        let got = format!("{ts}.{mac}");
        got.len() == want.len() && got.bytes().zip(want.bytes()).fold(0u8, |d, (a, b)| d | (a ^ b)) == 0
    }

    fn models_json(&self) -> String {
        let local = (self.local)();
        let loaded = self.pool.loaded();
        let mut ids: Vec<String> = Vec::new();
        for l in &local {
            let id = l.name.clone().map(|n| if l.level == "penguin" { n } else { format!("{n}:{}", l.level) }).unwrap_or_else(|| l.repo.clone());
            if !ids.contains(&id) {
                ids.push(id);
            }
        }
        for n in loaded {
            if !ids.contains(&n) {
                ids.push(n);
            }
        }
        let data: Vec<serde_json::Value> = ids.iter().map(|id| serde_json::json!({"id": id, "object": "model", "created": 0, "owned_by": "glyd"})).collect();
        serde_json::json!({"object": "list", "data": data}).to_string()
    }

    fn handle(&self, mut s: TcpStream) {
        let req = match read_request(&mut s) {
            Ok(r) => r,
            Err((code, why)) => return error(&mut s, code, &why),
        };
        let (path, query) = req.path.split_once('?').map_or((req.path.as_str(), ""), |(p, q)| (p, q));
        let path = path.to_string();
        // a name that DNS points at this computer, and another web page's script, are refused as `glyd serve` refuses them (an address the user chose with --host turns this off)
        if let Some((code, why)) = crate::server::refusal(req.header("host").unwrap_or(""), req.header("origin").unwrap_or(""), self.local_only) {
            return error(&mut s, code, &why);
        }
        let is_api = path.starts_with("/v1/");
        if is_api && !key_ok(&self.key, req.header("authorization")) {
            return error(&mut s, 401, "an API key is needed: Authorization: Bearer KEY");
        }
        match (req.method.as_str(), path.as_str()) {
            ("GET", "/health") => {
                // (a challenge is answered with the proof of the secret: `glyd ps`, `stop` and `run` ask before they send a key or a prompt)
                let proof = query
                    .split('&')
                    .find_map(|kv| kv.strip_prefix("challenge="))
                    .filter(|c| !c.is_empty() && c.len() <= 128)
                    .map(|c| proof(&self.secret, self.port, c));
                respond(&mut s, 200, "application/json", &serde_json::json!({"status": "ok", "service": "glyd", "proof": proof}).to_string());
            }
            ("GET", "/") => respond(&mut s, 200, "text/html; charset=utf-8", PAGE),
            ("GET", "/v1/models") => respond(&mut s, 200, "application/json", &self.models_json()),
            ("GET", "/v1/glyd/service") => respond(&mut s, 200, "application/json", &self.pool.status_json(self.port)),
            // (stopping the service and unloading a model are this account's to do: they carry the secret; another user of the computer who reaches the port is refused)
            ("POST", "/v1/glyd/unload" | "/v1/glyd/shutdown") if !self.control_ok(&req, &path) => {
                error(&mut s, 403, "this is for the account that runs the service: glyd stop and glyd service stop carry its secret")
            }
            ("POST", "/v1/glyd/unload") => {
                let which = model_of(&req.body);
                let alt = serde_json::from_slice::<serde_json::Value>(&req.body).ok().and_then(|v| v.get("name")?.as_str().map(str::to_string));
                let all = serde_json::from_slice::<serde_json::Value>(&req.body).ok().and_then(|v| v.get("all")?.as_bool()).unwrap_or(false);
                if which.is_none() && !all {
                    return error(&mut s, 400, "\"model\": a loaded model's name, or \"all\": true");
                }
                let n = if all {
                    self.pool.unload(None)
                } else {
                    self.pool.unload(which.as_deref()) + alt.filter(|a| Some(a) != which.as_ref()).map_or(0, |a| self.pool.unload(Some(&a)))
                };
                respond(&mut s, 200, "application/json", &serde_json::json!({"unloaded": n}).to_string());
            }
            ("POST", "/v1/glyd/shutdown") => {
                respond(&mut s, 200, "application/json", "{\"stopping\":true}");
                self.stop.store(true, std::sync::atomic::Ordering::SeqCst);
            }
            ("POST", p) if p.starts_with("/v1/") => {
                let name = model_of(&req.body).unwrap_or_default();
                let local = (self.local)();
                let spec = match spec_of(&name, &self.manifest, &local) {
                    Ok(sp) => sp,
                    Err((code, why)) => return error(&mut s, code, &why),
                };
                match self.pool.acquire(&spec) {
                    Ok(lease) => {
                        forward(&mut s, &req, lease.port);
                        self.pool.release(lease);
                    }
                    Err((code, why)) => error(&mut s, code, &why),
                }
            }
            ("GET", p) if p.starts_with("/v1/") => {
                // (the model-bound reads of the server: its info, a request's record) go to the model in use last, if one is loaded
                match self.pool.loaded().last().cloned() {
                    Some(name) => {
                        let local = (self.local)();
                        match spec_of(&name, &self.manifest, &local).ok().and_then(|sp| self.pool.acquire(&sp).ok()) {
                            Some(lease) => {
                                forward(&mut s, &req, lease.port);
                                self.pool.release(lease);
                            }
                            None => error(&mut s, 404, "no model is loaded"),
                        }
                    }
                    None => error(&mut s, 404, "no model is loaded: a request that names one loads it"),
                }
            }
            _ => error(&mut s, 404, "not found"),
        }
    }
}

/// The request passed to the worker on `port` and its answer streamed back to the end (the connection is closed after it on both sides, so a server-sent stream is passed as it comes).
fn forward(client: &mut TcpStream, req: &Request, port: u16) {
    let Ok(mut up) = TcpStream::connect(("127.0.0.1", port)) else { return error(client, 503, "the model's program is not answering") };
    let mut head = format!("{} {} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\n", req.method, req.path);
    for (k, v) in &req.headers {
        // (the worker is on this computer's own address: it takes the service's checks as done; hop-by-hop headers and the length, which is set below, are not passed)
        if ["host", "origin", "connection", "content-length", "transfer-encoding", "keep-alive", "te", "upgrade", "proxy-connection", "expect"]
            .contains(&k.to_ascii_lowercase().as_str())
        {
            continue;
        }
        head.push_str(&format!("{k}: {v}\r\n"));
    }
    head.push_str(&format!("Content-Length: {}\r\n\r\n", req.body.len()));
    if up.write_all(head.as_bytes()).and_then(|_| up.write_all(&req.body)).is_err() {
        return error(client, 503, "the model's program is not answering");
    }
    client.set_nodelay(true).ok();
    let mut buf = [0u8; 16 << 10];
    loop {
        match up.read(&mut buf) {
            Ok(0) | Err(_) => break,
            Ok(n) => {
                if client.write_all(&buf[..n]).is_err() {
                    break; // (the client left: the worker's request ends when its write fails)
                }
                let _ = client.flush();
            }
        }
    }
    let _ = up.shutdown(Shutdown::Both);
}

/// Starts the service on `addr`: returns the address it listens on and the flag that stops it. The threads run until the flag is set.
fn serve(
    addr: &str,
    pool: Pool,
    manifest: Manifest,
    local: Box<dyn Fn() -> Vec<models::Local> + Send + Sync>,
    key: Option<String>,
    secret: String,
    local_only: bool,
) -> Result<(SocketAddr, Arc<std::sync::atomic::AtomicBool>, Arc<Server>), std::io::Error> {
    let listener = TcpListener::bind(addr)?;
    let bound = listener.local_addr()?;
    let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let sv = Arc::new(Server { pool, manifest, local, key, secret, local_only, port: bound.port(), stop: stop.clone() });
    let accept = sv.clone();
    std::thread::spawn(move || {
        for conn in listener.incoming() {
            let Ok(s) = conn else { continue };
            let sv = accept.clone();
            std::thread::spawn(move || sv.handle(s));
        }
    });
    let reaper = sv.clone();
    std::thread::spawn(move || {
        while !reaper.stop.load(std::sync::atomic::Ordering::SeqCst) {
            std::thread::sleep(Duration::from_millis(500));
            reaper.pool.reap();
        }
    });
    Ok((bound, stop, sv))
}

// ---- the commands ----------------------------------------------------------------------------------------------------------------------------------------------

/// The port the service commands look at: --port, else GLYD_SERVICE_PORT, else the default.
fn port_of(a: &Args) -> Result<u16, String> {
    match a.flag("port") {
        Some(p) => p.parse().map_err(|_| format!("--port {p}: not a port")),
        None => match std::env::var("GLYD_SERVICE_PORT").ok().filter(|v| !v.is_empty()) {
            Some(p) => p.parse().map_err(|_| format!("GLYD_SERVICE_PORT={p}: not a port")),
            None => Ok(DEFAULT_PORT),
        },
    }
}

/// `glyd service [--port N] [--host H] [--keep-alive 5m|never] [--max-loaded N] [--context N] [--users N]`: the service, in the foreground (what the user unit runs).
pub fn run(a: &Args) -> i32 {
    // (an `install`, `uninstall`, `status` or `stop` is a word after the command)
    match a.pos.get(1).map(String::as_str) {
        Some("install") => return install(a),
        Some("uninstall") => return uninstall(),
        Some("status") => return ps(a),
        Some("stop") => return stop_service(a),
        Some(other) => {
            eprintln!("glyd service: {other}: install, uninstall, status or stop (no word: run the service here)");
            return 1;
        }
        None => {}
    }
    let (keep_alive, max_loaded) = match (a.flag("keep-alive").map_or(Ok(Some(DEFAULT_KEEP_ALIVE)), parse_duration), a.num("max-loaded", DEFAULT_MAX_LOADED)) {
        (Ok(k), Ok(m)) => (k, m),
        (Err(e), _) => {
            eprintln!("glyd service: --keep-alive {e}");
            return 1;
        }
        (_, Err(e)) => {
            eprintln!("glyd service: {e}");
            return 1;
        }
    };
    let port = match port_of(a) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("glyd service: {e}");
            return 1;
        }
    };
    let host = a.flag("host").unwrap_or("127.0.0.1").to_string();
    // (the checks of Host and Origin stay on for any address of this computer's own, a loopback one the user names included: an address that is not is the user's word to open it)
    let local_only = crate::server::is_loopback_addr(&host);
    let key = api_key();
    if let Some(why) = crate::server::bind_refusal(&host, key.is_some(), a.has("insecure-no-auth")) {
        eprintln!("glyd service: {why}");
        return 1;
    }
    match crate::server::parse_access(a.flag("allowed-origins"), a.flag("allowed-hosts")) {
        Ok(x) => crate::server::set_access(x),
        Err(e) => {
            eprintln!("glyd service: {e}");
            return 1;
        }
    }
    let pool = Pool::new(
        Box::new(Processes { context: a.flag("context").map(str::to_string), users: a.flag("users").map(str::to_string) }),
        Config { keep_alive, max_loaded, wait_for_room: DEFAULT_WAIT_FOR_ROOM },
    );
    let secret = match ensure_secret() {
        Ok(s) => s,
        Err(e) => {
            eprintln!("glyd service: no secret for this account's commands: {e}");
            return 1;
        }
    };
    let manifest = Manifest::load();
    let scan_with = manifest.clone();
    let local = Box::new(move || models::scan(&crate::hub::hub_cache(), &scan_with));
    let (bound, stop, sv) = match serve(&format!("{host}:{port}"), pool, manifest, local, key.clone(), secret, local_only) {
        Ok(x) => x,
        Err(e) if e.kind() == std::io::ErrorKind::AddrInUse => {
            let who = port_owner(port).map_or("another program".to_string(), |w| w);
            eprintln!("glyd service: port {port} is in use by {who}. Stop it, or give the service another port: glyd service --port N (glyd ps, stop and run take --port N or GLYD_SERVICE_PORT=N too).");
            return 1;
        }
        Err(e) => {
            eprintln!("glyd service: {host}:{port}: {e}");
            return 1;
        }
    };
    if let Some(h) = a.flag("host").filter(|h| !local_only && !h.is_empty()) {
        eprintln!(
            "glyd service: listening on {h}: anyone who can reach this computer's network address can send the models prompts, and the traffic is plain HTTP, not encrypted (use a VPN or an SSH tunnel). {}",
            if key.is_some() { "The API key guards /v1 only." } else { "There is NO API key (--insecure-no-auth): set one with GLYD_API_KEY=KEY (in the environment, not on the command line)." }
        );
    }
    eprintln!(
        "glyd service: listening on http://{bound} (models load on request, unload after {} idle, up to {max_loaded} held; log of each: {})",
        keep_alive.map_or("never".to_string(), human),
        log_dir().display()
    );
    while !stop.load(std::sync::atomic::Ordering::SeqCst) {
        std::thread::sleep(Duration::from_millis(200));
    }
    std::thread::sleep(Duration::from_millis(200)); // (the answer to the stop request goes out first)
    sv.pool.shutdown();
    0
}

fn service_key(a: &Args) -> Option<String> {
    let _ = a;
    api_key()
}

/// `glyd ps [--port N] [--json]`: what the service holds.
pub fn ps(a: &Args) -> i32 {
    let port = match port_of(a) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("glyd ps: {e}");
            return 1;
        }
    };
    if !verified(port) {
        match get(("127.0.0.1", port), "/health", None, Duration::from_millis(800)) {
            Ok(_) => eprintln!(
                "glyd ps: something answers on port {port}, but it is not the glyd service of this account (it does not prove the secret in {}): {}",
                secret_path().display(),
                port_owner(port).unwrap_or_else(|| "another program".into())
            ),
            Err(_) => eprintln!("glyd ps: the service is not running on port {port}. glyd service install starts it at login (glyd service runs it here)."),
        }
        return 1;
    }
    let (code, body) = match get(("127.0.0.1", port), "/v1/glyd/service", service_key(a).as_deref(), Duration::from_secs(3)) {
        Ok(x) => x,
        Err(e) => {
            eprintln!("glyd ps: {e}");
            return 1;
        }
    };
    if code != 200 {
        eprintln!("glyd ps: the service answered {code}: {}", body.trim());
        return 1;
    }
    if a.has("json") {
        println!("{body}");
        return 0;
    }
    let Ok(v) = serde_json::from_str::<serde_json::Value>(&body) else {
        eprintln!("glyd ps: port {port} is not a glyd service");
        return 1;
    };
    let secs = |x: Option<&serde_json::Value>| x.and_then(|x| x.as_u64()).map(|s| human(Duration::from_secs(s)));
    let models = v["models"].as_array().cloned().unwrap_or_default();
    println!(
        "glyd service on port {port}, up {}; models unload after {} idle, {} held at most",
        secs(v.get("up_secs")).unwrap_or_default(),
        secs(v.get("keep_alive_secs")).unwrap_or("never".into()),
        v["max_loaded"]
    );
    if models.is_empty() {
        println!("No model is loaded: the first request that names one loads it.");
        return 0;
    }
    let w = models.iter().map(|m| m["name"].as_str().unwrap_or("").len()).max().unwrap_or(4).max(4);
    println!("{:<w$}  {:>8}  {:>8}  {:>8}  {:>6}  UNLOADS IN", "NAME", "LOADED", "IDLE", "REQUESTS", "ACTIVE");
    for m in &models {
        println!(
            "{:<w$}  {:>8}  {:>8}  {:>8}  {:>6}  {}",
            m["name"].as_str().unwrap_or(""),
            secs(m.get("loaded_secs")).unwrap_or_default(),
            secs(m.get("idle_secs")).unwrap_or_default(),
            m["requests"].to_string(),
            m["in_flight"].to_string(),
            secs(m.get("unload_in_secs")).unwrap_or("-".into())
        );
    }
    0
}

/// `glyd stop MODEL | --all [--port N]`: the model unloaded now (the service stays).
pub fn stop(a: &Args) -> i32 {
    let port = match port_of(a) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("glyd stop: {e}");
            return 1;
        }
    };
    let body = match (a.pos.get(1), a.has("all")) {
        (Some(m), false) => {
            // (the name `glyd ps` shows, or a short name or a Hub id, which is the loaded model's `model`)
            let given = models::resolve(m, &Manifest::load()).ok().flatten().map_or(m.clone(), |r| r.model());
            serde_json::json!({"model": given, "name": m}).to_string()
        }
        (None, true) => "{\"all\":true}".to_string(),
        _ => {
            eprintln!("glyd stop: a model's name (glyd ps shows them), or --all. glyd service stop ends the service itself.");
            return 1;
        }
    };
    let Some(secret) = read_secret().filter(|_| verified(port)) else {
        eprintln!("glyd stop: the service of this account is not running on port {port}");
        return 1;
    };
    let auth = control_auth(&secret, port, "POST", "/v1/glyd/unload", &body, now_secs());
    match send_as(("127.0.0.1", port), "POST", "/v1/glyd/unload", Some(&body), service_key(a).as_deref(), Some(&auth), Duration::from_secs(60)) {
        Ok((200, b)) => {
            let n = serde_json::from_str::<serde_json::Value>(&b).ok().and_then(|v| v["unloaded"].as_u64()).unwrap_or(0);
            if n == 0 {
                eprintln!("glyd stop: no idle model of that name is loaded (glyd ps shows what is; one in use is left until it is done)");
                return 1;
            }
            println!("Unloaded {n} model{}", if n == 1 { "" } else { "s" });
            0
        }
        Ok((code, b)) => {
            eprintln!("glyd stop: the service answered {code}: {}", b.trim());
            1
        }
        Err(_) => {
            eprintln!("glyd stop: the service is not running on port {port}");
            1
        }
    }
}

/// `glyd service stop`: the service ends (the models with it): through the user unit where there is one, else by asking it.
fn stop_service(a: &Args) -> i32 {
    if installed() && manager_stop() {
        println!("The service is stopped (glyd service install starts it again at login; glyd service runs it here)");
        return 0;
    }
    let port = port_of(a).unwrap_or(DEFAULT_PORT);
    let Some(secret) = read_secret().filter(|_| verified(port)) else {
        eprintln!("glyd service stop: no service of this account is running on port {port}");
        return 1;
    };
    let auth = control_auth(&secret, port, "POST", "/v1/glyd/shutdown", "{}", now_secs());
    match send_as(("127.0.0.1", port), "POST", "/v1/glyd/shutdown", Some("{}"), service_key(a).as_deref(), Some(&auth), Duration::from_secs(10)) {
        Ok((200, _)) => {
            println!("The service is stopping");
            0
        }
        _ => {
            eprintln!("glyd service stop: no service is running on port {port}");
            1
        }
    }
}

// ---- the user-level service of each OS --------------------------------------------------------------------------------------------------------------------------

/// A word of a systemd unit's command line, quoted where it needs it.
pub fn systemd_quote(s: &str) -> String {
    if s.chars().any(|c| c.is_whitespace() || matches!(c, '"' | '\\' | '$' | '%')) {
        format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\"").replace('%', "%%").replace('$', "$$"))
    } else {
        s.to_string()
    }
}

/// The user unit for systemd (`~/.config/systemd/user/glyd.service`).
pub fn systemd_unit(exe: &str, args: &[String]) -> String {
    let cmd: Vec<String> = std::iter::once(exe.to_string()).chain(["service".to_string()]).chain(args.iter().cloned()).map(|w| systemd_quote(&w)).collect();
    format!(
        "[Unit]\nDescription=Glyd: models loaded on request, unloaded when idle\nAfter=network.target\n\n[Service]\nExecStart={}\nRestart=on-failure\nRestartSec=3\n\n[Install]\nWantedBy=default.target\n",
        cmd.join(" ")
    )
}

fn xml(s: &str) -> String {
    s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;")
}

/// The launchd agent (`~/Library/LaunchAgents/com.getglyd.glyd.plist`).
pub fn launchd_plist(exe: &str, args: &[String], log: &str) -> String {
    let words: String = std::iter::once(exe.to_string())
        .chain(["service".to_string()])
        .chain(args.iter().cloned())
        .map(|w| format!("    <string>{}</string>\n", xml(&w)))
        .collect();
    format!(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<!DOCTYPE plist PUBLIC \"-//Apple//DTD PLIST 1.0//EN\" \"http://www.apple.com/DTDs/PropertyList-1.0.dtd\">\n<plist version=\"1.0\">\n<dict>\n  <key>Label</key>\n  <string>com.getglyd.glyd</string>\n  <key>ProgramArguments</key>\n  <array>\n{words}  </array>\n  <key>RunAtLoad</key>\n  <true/>\n  <key>KeepAlive</key>\n  <true/>\n  <key>StandardOutPath</key>\n  <string>{log}</string>\n  <key>StandardErrorPath</key>\n  <string>{log}</string>\n</dict>\n</plist>\n",
        log = xml(log)
    )
}

fn unit_path() -> PathBuf {
    let home = PathBuf::from(crate::os::home());
    if cfg!(target_os = "macos") {
        home.join("Library/LaunchAgents/com.getglyd.glyd.plist")
    } else {
        std::env::var_os("XDG_CONFIG_HOME")
            .filter(|v| !v.is_empty())
            .map(PathBuf::from)
            .unwrap_or_else(|| home.join(".config"))
            .join("systemd/user/glyd.service")
    }
}

fn installed() -> bool {
    unit_path().is_file()
}

fn sh(prog: &str, args: &[&str]) -> bool {
    std::process::Command::new(prog).args(args).stdout(std::process::Stdio::null()).stderr(std::process::Stdio::null()).status().is_ok_and(|s| s.success())
}

#[cfg(unix)]
fn uid() -> String {
    std::process::Command::new("id")
        .arg("-u")
        .output()
        .ok()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .filter(|u| !u.is_empty())
        .unwrap_or_else(|| "501".into())
}

#[cfg(not(unix))]
fn uid() -> String {
    String::new()
}

fn manager_stop() -> bool {
    if cfg!(target_os = "macos") {
        sh("launchctl", &["bootout", &format!("gui/{}", uid()), &unit_path().to_string_lossy()])
    } else {
        sh("systemctl", &["--user", "stop", "glyd.service"])
    }
}

/// The flags of this command that the service is started with.
fn service_flags(a: &Args) -> Vec<String> {
    let mut v = Vec::new();
    // (the port too where only GLYD_SERVICE_PORT names it: the unit does not have this shell's environment)
    if !a.has("port") {
        if let Ok(p) = port_of(a).map(|p| p.to_string()) {
            if p != DEFAULT_PORT.to_string() {
                v.extend(["--port".to_string(), p]);
            }
        }
    }
    for f in ["port", "host", "keep-alive", "max-loaded", "context", "users"] {
        if let Some(x) = a.flag(f) {
            v.push(format!("--{f}"));
            v.push(x.to_string());
        }
    }
    v
}

/// `glyd service install [--port N] [--keep-alive T] ...`: the user-level service of this OS, started now and at every login. No administrator rights: it is the user's own unit.
fn install(a: &Args) -> i32 {
    let exe = match std::env::current_exe() {
        Ok(e) => e.to_string_lossy().into_owned(),
        Err(e) => {
            eprintln!("glyd service install: {e}");
            return 1;
        }
    };
    // (a unit has no API key: it is for this computer's own address; an address the user chose is run by hand, with GLYD_API_KEY in that shell's environment)
    if a.flag("host").is_some_and(|h| !crate::server::is_loopback_addr(h)) {
        eprintln!("glyd service install: not on {}: a unit started at login has no API key, and anyone who can reach that address could send the models prompts. Run it by hand there: GLYD_API_KEY=KEY glyd service --host ...", a.flag("host").unwrap_or(""));
        return 1;
    }
    let flags = service_flags(a);
    let path = unit_path();
    if cfg!(windows) {
        eprintln!("glyd service install: the Windows service is not here yet (ROADMAP.md, M1 P3). `glyd service` runs the service in a terminal, and Task Scheduler can start it at logon.");
        return 1;
    }
    let text =
        if cfg!(target_os = "macos") { launchd_plist(&exe, &flags, &log_dir().join("service.log").to_string_lossy()) } else { systemd_unit(&exe, &flags) };
    if let Some(d) = path.parent() {
        let _ = std::fs::create_dir_all(d);
    }
    let _ = std::fs::create_dir_all(log_dir());
    if let Err(e) = std::fs::write(&path, text) {
        eprintln!("glyd service install: {}: {e}", path.display());
        return 1;
    }
    let port = port_of(a).unwrap_or(DEFAULT_PORT);
    let ok = if cfg!(target_os = "macos") {
        let domain = format!("gui/{}", uid());
        let _ = sh("launchctl", &["bootout", &domain, &path.to_string_lossy()]);
        sh("launchctl", &["bootstrap", &domain, &path.to_string_lossy()])
    } else {
        sh("systemctl", &["--user", "daemon-reload"])
            && sh("systemctl", &["--user", "enable", "glyd.service"])
            && sh("systemctl", &["--user", "restart", "glyd.service"])
    };
    if !ok {
        eprintln!(
            "glyd service install: {} is written, but the system's service manager would not start it (no systemd user session here?). Run it with: glyd service{}",
            path.display(),
            flags.iter().map(|f| format!(" {f}")).collect::<String>()
        );
        return 1;
    }
    // (up, or the port is taken: the answer names who has it)
    for _ in 0..20 {
        if verified(port) {
            println!(
                "The service is running: http://127.0.0.1:{port} (models load on request and unload when idle; glyd ps shows them). It starts at login{}.",
                if cfg!(target_os = "linux") { "; loginctl enable-linger $USER makes it start at boot" } else { "" }
            );
            return 0;
        }
        std::thread::sleep(Duration::from_millis(250));
    }
    match port_owner(port) {
        Some(who) if !who.contains("glyd") => eprintln!("glyd service install: the service did not come up: port {port} is in use by {who}. Stop it, or install on another port: glyd service install --port N"),
        _ => eprintln!("glyd service install: the service did not answer on port {port} yet: {}", if cfg!(target_os = "macos") { "launchctl print gui/$(id -u)/com.getglyd.glyd" } else { "systemctl --user status glyd.service" }),
    }
    1
}

fn uninstall() -> i32 {
    let path = unit_path();
    if !path.is_file() {
        println!("The service is not installed");
        return 0;
    }
    if cfg!(target_os = "macos") {
        manager_stop();
    } else {
        sh("systemctl", &["--user", "disable", "--now", "glyd.service"]);
    }
    let _ = std::fs::remove_file(&path);
    if !cfg!(target_os = "macos") {
        sh("systemctl", &["--user", "daemon-reload"]);
    }
    println!("The service is stopped and removed ({})", path.display());
    0
}

// ---- `glyd run` attached to the service ------------------------------------------------------------------------------------------------------------------------

/// The flags of `glyd run` that are one process's own (its context, its GPU, its address): a run with one of them is not the service's.
const OWN_FLAGS: [&str; 9] = ["context", "ctx", "users", "host", "port", "layout", "expert-cache", "host-memory", "cpu-threads"];

/// Whether this account's glyd service answers on `port`.
pub fn running(port: u16) -> bool {
    verified(port) // (a key or a prompt is sent only to the account's own service: a program that took the port is not it)
}

/// `glyd run MODEL` where the service is running: the chat goes through it (so the model it holds is the one used, and no second copy takes the GPU). None where there is no service, or
/// the run asks for what only a process of its own has (--context, --port, ...), or GLYD_SERVICE=0.
pub fn attach(a: &Args) -> Option<i32> {
    if std::env::var("GLYD_SERVICE").is_ok_and(|v| v == "0") || OWN_FLAGS.iter().any(|f| a.has(f)) || a.has("lossy") {
        return None;
    }
    let port = port_of(a).ok()?;
    if !running(port) {
        return None;
    }
    let model = a.pos.get(1)?.clone();
    // (a model that is not downloaded is the run's own to download: it is not asked of the service, which loads only what is there)
    let manifest = Manifest::load();
    spec_of(&model, &manifest, &models::scan(&crate::hub::hub_cache(), &manifest)).ok()?;
    let name = model.clone();
    eprintln!("glyd run: {name} through the service on port {port} (GLYD_SERVICE=0: a copy of its own)");
    let key = api_key();
    let mut messages: Vec<serde_json::Value> = Vec::new();
    let one_shot = a.flag("prompt").map(str::to_string);
    let stdin = std::io::stdin();
    loop {
        let line = match &one_shot {
            Some(p) if messages.is_empty() => p.clone(),
            Some(_) => return Some(0),
            None => {
                eprint!("> ");
                let mut l = String::new();
                match stdin.lock().read_line(&mut l) {
                    Ok(0) | Err(_) => return Some(0),
                    Ok(_) => l.trim().to_string(),
                }
            }
        };
        if line.is_empty() {
            continue;
        }
        messages.push(serde_json::json!({"role": "user", "content": line}));
        let mut body = serde_json::json!({"model": model, "messages": messages, "stream": true});
        if a.has("no-think") {
            body["chat_template_kwargs"] = serde_json::json!({"enable_thinking": false});
        }
        // (proved again for each request, just before it is sent: the program on the port is the account's own service when its prompt and its key go)
        if !verified(port) {
            eprintln!("glyd run: the service on port {port} is gone or is not this account's: nothing was sent to it");
            return Some(1);
        }
        match chat_stream(port, &body.to_string(), key.as_deref()) {
            Ok(reply) => {
                println!();
                messages.push(serde_json::json!({"role": "assistant", "content": reply}));
            }
            Err(e) => {
                eprintln!("glyd run: {e}");
                return Some(1);
            }
        }
    }
}

/// The text of a streamed chat answer from the service, printed as it comes; the whole text.
fn chat_stream(port: u16, body: &str, key: Option<&str>) -> Result<String, String> {
    let mut s = TcpStream::connect(("127.0.0.1", port)).map_err(|e| e.to_string())?;
    let mut req = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\nContent-Type: application/json\r\nContent-Length: {}\r\n",
        body.len()
    );
    if let Some(k) = key {
        req.push_str(&format!("Authorization: Bearer {k}\r\n"));
    }
    req.push_str("\r\n");
    s.write_all(req.as_bytes()).and_then(|_| s.write_all(body.as_bytes())).map_err(|e| e.to_string())?;
    let mut r = std::io::BufReader::new(s);
    let mut status = String::new();
    r.read_line(&mut status).map_err(|e| e.to_string())?;
    let code: u16 = status.split(' ').nth(1).and_then(|c| c.parse().ok()).unwrap_or(0);
    // (the head, then the body)
    let mut chunked = false;
    loop {
        let mut l = String::new();
        if r.read_line(&mut l).map_err(|e| e.to_string())? == 0 || l.trim().is_empty() {
            break;
        }
        chunked |= l.to_ascii_lowercase().starts_with("transfer-encoding") && l.to_ascii_lowercase().contains("chunked");
    }
    if code != 200 {
        let mut rest = String::new();
        r.read_to_string(&mut rest).ok();
        let why = serde_json::from_str::<serde_json::Value>(rest.trim()).ok().and_then(|v| v["error"]["message"].as_str().map(str::to_string)).unwrap_or(rest);
        return Err(format!("the service answered {code}: {}", why.trim()));
    }
    let mut out = String::new();
    let mut line = String::new();
    loop {
        line.clear();
        if r.read_line(&mut line).map_err(|e| e.to_string())? == 0 {
            break;
        }
        // (a chunked body's size lines are skipped: the data lines are what is read)
        let Some(data) = line.trim().strip_prefix("data:") else { continue };
        let data = data.trim();
        if data == "[DONE]" {
            break;
        }
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(data) {
            if let Some(t) = v["choices"][0]["delta"]["content"].as_str() {
                print!("{t}");
                let _ = std::io::stdout().flush();
                out.push_str(t);
            }
        }
    }
    let _ = chunked;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn times_are_read_as_a_person_writes_them() {
        let d = |s: &str| parse_duration(s).unwrap();
        assert_eq!(d("5m"), Some(Duration::from_secs(300)));
        assert_eq!(d("90s"), Some(Duration::from_secs(90)));
        assert_eq!(d("1h30m"), Some(Duration::from_secs(5400)));
        assert_eq!(d("2H"), Some(Duration::from_secs(7200)));
        assert_eq!(d("0"), Some(Duration::ZERO));
        assert_eq!(d("45"), Some(Duration::from_secs(45)));
        assert_eq!((d("never"), d("-1")), (None, None));
        for bad in ["", "m", "5x", "5m3", "abc"] {
            assert!(parse_duration(bad).is_err(), "{bad}");
        }
        assert_eq!(
            (human(Duration::from_secs(9)), human(Duration::from_secs(252)), human(Duration::from_secs(10800))),
            ("9s".into(), "4m12s".into(), "3h00m".into())
        );
    }

    #[test]
    fn the_program_on_a_port_is_named_from_what_the_system_prints() {
        assert_eq!(parse_lsof("p812\ncollama\n").as_deref(), Some("ollama (pid 812)"));
        assert_eq!(parse_lsof("p812\n"), None);
        assert_eq!(parse_ss("LISTEN 0 4096 127.0.0.1:11434 0.0.0.0:* users:((\"ollama\",pid=812,fd=3))\n").as_deref(), Some("ollama (pid 812)"));
        assert_eq!(parse_ss("LISTEN 0 4096 127.0.0.1:11434 0.0.0.0:*\n"), None);
    }

    #[test]
    fn the_units_are_the_users_own_and_quote_what_needs_it() {
        let u = systemd_unit("/home/a b/.local/share/glyd/cli/glyd", &["--port".into(), "11435".into(), "--keep-alive".into(), "10m".into()]);
        assert!(u.contains("ExecStart=\"/home/a b/.local/share/glyd/cli/glyd\" service --port 11435 --keep-alive 10m\n"), "{u}");
        assert!(u.contains("WantedBy=default.target") && u.contains("Restart=on-failure") && !u.contains("User="), "{u}");
        let p = launchd_plist("/Users/a&b/glyd", &["--port".into(), "1".into()], "/tmp/x.log");
        assert!(
            p.contains("<string>/Users/a&amp;b/glyd</string>")
                && p.contains("<string>service</string>")
                && p.contains("<key>RunAtLoad</key>")
                && p.contains("com.getglyd.glyd"),
            "{p}"
        );
    }

    #[test]
    fn a_request_names_a_downloaded_model_or_is_refused_with_what_to_do() {
        let m = Manifest::load();
        let local = |repos: &[&str]| -> Vec<models::Local> {
            repos.iter().map(|r| models::Local { repo: r.to_string(), bytes: 1, level: "penguin".into(), name: None }).collect()
        };
        let have = local(&["Qwen/Qwen3-8B", "glyd/Qwen3-8B-swift", "someone/else"]);
        let ok = |n: &str| spec_of(n, &m, &have).unwrap();
        assert_eq!(ok("qwen3"), Spec { arg: "Qwen/Qwen3-8B".into(), name: "qwen3:8b".into() });
        assert_eq!(ok("qwen3:8b:swift").arg, "Qwen/Qwen3-8B:swift");
        assert_eq!(ok("Qwen/Qwen3-8B").arg, "Qwen/Qwen3-8B");
        assert_eq!(ok("someone/else").arg, "someone/else");
        let (c, why) = spec_of("qwen3:32b", &m, &have).unwrap_err();
        assert!(c == 404 && why.contains("is not downloaded: glyd pull qwen3:32b"), "{why}");
        // no path is loaded, no name that is not one, no level that is none
        for bad in ["/etc", "./m", "~/m", "model.gguf", "nonsense", "", "qwen3:8b:fast"] {
            assert_eq!(spec_of(bad, &m, &have).unwrap_err().0, 404, "{bad}");
        }
        assert!(spec_of("Qwen/Qwen3-8B:swift", &m, &local(&[])).is_err());
        // the pre-packed repo of a level alone is enough
        assert_eq!(spec_of("qwen3:swift", &m, &local(&["glyd/Qwen3-8B-swift"])).unwrap().arg, "glyd/Qwen3-8B-swift");
        assert_eq!(spec_of("Qwen/Qwen3-8B:swift", &m, &local(&["glyd/Qwen3-8B-swift"])).unwrap().arg, "glyd/Qwen3-8B-swift");
        // a pre-packed repo given by its own name is what it is
        assert_eq!(spec_of("glyd/Qwen3-8B-swift", &m, &local(&["glyd/Qwen3-8B-swift"])).unwrap().arg, "glyd/Qwen3-8B-swift");
    }

    #[test]
    fn the_victim_is_the_least_recently_used_idle_model() {
        let t = Instant::now();
        let at = |s: u64| t + Duration::from_secs(s);
        assert_eq!(victim(&[(at(5), 0), (at(1), 0), (at(3), 0)]), Some(1));
        assert_eq!(victim(&[(at(5), 0), (at(1), 2), (at(3), 0)]), Some(2));
        assert_eq!(victim(&[(at(1), 1)]), None);
        assert_eq!(victim(&[]), None);
    }

    #[test]
    fn a_workers_port_is_read_from_what_it_says() {
        assert_eq!(
            served_port("Qwen3-0.6B is served at http://127.0.0.1:41233: the chat page at /, the API at /v1 (OpenAI-compatible). Ctrl-C stops it."),
            Some(41233)
        );
        assert_eq!(served_port("Loading...\nx is served at http://127.0.0.1:8000: y"), Some(8000));
        assert_eq!(served_port("x is served at http://127.0.0.1:0: y"), None);
        assert_eq!(served_port("Packing..."), None);
    }

    #[test]
    fn the_key_is_checked_as_the_server_does() {
        let k = Some("secret".to_string());
        assert!(key_ok(&None, None));
        assert!(key_ok(&k, Some("Bearer secret")) && key_ok(&k, Some("bearer secret")));
        assert!(!key_ok(&k, None) && !key_ok(&k, Some("Bearer wrong")) && !key_ok(&k, Some("Basic secret")) && !key_ok(&k, Some("secret")));
    }

    // ---- the pool and the proxy, with workers that are threads -----------------------------------------------------------------------------------------------

    /// A worker that is a thread: it answers /health, and a POST with a stream of three events (or what it was told to say).
    struct Fake {
        port: u16,
        up: Arc<std::sync::atomic::AtomicBool>,
        name: String,
        log: Arc<Mutex<Vec<String>>>,
        live: Arc<AtomicUsize>,
    }

    impl Handle for Fake {
        fn port(&self) -> u16 {
            self.port
        }
        fn alive(&mut self) -> bool {
            self.up.load(Ordering::SeqCst)
        }
        fn stop(&mut self) {
            if self.up.swap(false, Ordering::SeqCst) {
                self.live.fetch_sub(1, Ordering::SeqCst);
            }
            self.log.lock().unwrap().push(format!("stop {}", self.name));
            let _ = TcpStream::connect(("127.0.0.1", self.port)); // (wakes its accept)
        }
    }

    struct Fakes {
        log: Arc<Mutex<Vec<String>>>,
        starts: AtomicUsize,
        fail_when_more_than: Option<usize>,
        live: Arc<AtomicUsize>,
        delay: Duration,
    }

    impl Fakes {
        fn new() -> Arc<Fakes> {
            Arc::new(Fakes { log: Arc::default(), starts: AtomicUsize::new(0), fail_when_more_than: None, live: Arc::default(), delay: Duration::ZERO })
        }
    }

    impl Backend for Arc<Fakes> {
        fn start(&self, spec: &Spec) -> Result<Box<dyn Handle>, String> {
            std::thread::sleep(self.delay);
            // (a load that would not fit: it fails while more than this many are loaded)
            if self.fail_when_more_than.is_some_and(|n| self.live.load(Ordering::SeqCst) >= n) {
                return Err("out of memory".into());
            }
            self.starts.fetch_add(1, Ordering::SeqCst);
            self.log.lock().unwrap().push(format!("start {}", spec.arg));
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            let port = listener.local_addr().unwrap().port();
            let up = Arc::new(std::sync::atomic::AtomicBool::new(true));
            let (u2, name) = (up.clone(), spec.arg.clone());
            self.live.fetch_add(1, Ordering::SeqCst);
            std::thread::spawn(move || {
                for conn in listener.incoming() {
                    if !u2.load(Ordering::SeqCst) {
                        break;
                    }
                    let Ok(mut s) = conn else { continue };
                    let req = read_request(&mut s);
                    match req {
                        Ok(r) if r.path == "/health" => respond(&mut s, 200, "application/json", "{\"status\":\"ok\"}"),
                        Ok(r) => {
                            // an event stream, a piece at a time
                            let host = r.header("host").unwrap_or("").to_string();
                            let origin = r.header("origin").is_some();
                            let _ = s.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n");
                            for piece in [
                                format!(
                                    "data: {{\"id\":\"{name}\",\"host\":\"{host}\",\"origin\":{origin},\"body\":{}}}\n\n",
                                    serde_json::Value::String(String::from_utf8_lossy(&r.body).into_owned())
                                ),
                                "data: [DONE]\n\n".to_string(),
                            ] {
                                let _ = s.write_all(piece.as_bytes());
                                let _ = s.flush();
                                std::thread::sleep(Duration::from_millis(20));
                            }
                        }
                        Err(_) => {}
                    }
                }
            });
            Ok(Box::new(Fake { port, up, name: spec.arg.clone(), log: self.log.clone(), live: self.live.clone() }))
        }
    }

    fn spec(arg: &str) -> Spec {
        Spec { arg: arg.into(), name: arg.into() }
    }

    fn pool(f: &Arc<Fakes>, keep: Option<Duration>, max: usize) -> Pool {
        Pool::new(Box::new(f.clone()), Config { keep_alive: keep, max_loaded: max, wait_for_room: Duration::ZERO })
    }

    #[test]
    fn a_model_loads_on_its_first_request_and_is_reused() {
        let f = Fakes::new();
        let p = pool(&f, None, 2);
        let l = p.acquire(&spec("a/x")).unwrap();
        p.release(l);
        let l = p.acquire(&spec("a/x")).unwrap();
        p.release(l);
        assert_eq!(f.starts.load(Ordering::SeqCst), 1);
        assert_eq!(p.loaded(), ["a/x"]);
        let st: serde_json::Value = serde_json::from_str(&p.status_json(1)).unwrap();
        assert_eq!((st["models"][0]["requests"].as_u64(), st["models"][0]["in_flight"].as_u64(), st["service"].as_str()), (Some(2), Some(0), Some("glyd")));
        p.shutdown();
    }

    #[test]
    fn requests_for_one_model_that_come_together_load_it_once() {
        let f = Arc::new(Fakes { delay: Duration::from_millis(200), ..Arc::try_unwrap(Fakes::new()).ok().unwrap() });
        let p = Arc::new(pool(&f, None, 2));
        let ts: Vec<_> = (0..4)
            .map(|_| {
                let p = p.clone();
                std::thread::spawn(move || {
                    let l = p.acquire(&spec("a/x")).unwrap();
                    std::thread::sleep(Duration::from_millis(30));
                    p.release(l);
                })
            })
            .collect();
        for t in ts {
            t.join().unwrap();
        }
        assert_eq!(f.starts.load(Ordering::SeqCst), 1);
        p.shutdown();
    }

    #[test]
    fn an_idle_model_is_unloaded_after_the_keep_alive_and_a_busy_one_is_not() {
        let f = Fakes::new();
        let p = pool(&f, Some(Duration::from_millis(100)), 2);
        let busy = p.acquire(&spec("a/busy")).unwrap();
        let l = p.acquire(&spec("a/idle")).unwrap();
        p.release(l);
        assert_eq!(p.reap(), 0, "not yet idle for the keep-alive");
        std::thread::sleep(Duration::from_millis(150));
        assert_eq!(p.reap(), 1);
        assert_eq!(p.loaded(), ["a/busy"]);
        assert!(f.log.lock().unwrap().contains(&"stop a/idle".to_string()));
        p.release(busy);
        std::thread::sleep(Duration::from_millis(150));
        assert_eq!(p.reap(), 1);
        assert!(p.loaded().is_empty());
        // and loads again on the next request
        let l = p.acquire(&spec("a/idle")).unwrap();
        p.release(l);
        assert_eq!(f.starts.load(Ordering::SeqCst), 3);
        p.shutdown();
        // never: nothing is unloaded by time
        let p = pool(&f, None, 2);
        let l = p.acquire(&spec("a/y")).unwrap();
        p.release(l);
        std::thread::sleep(Duration::from_millis(50));
        assert_eq!(p.reap(), 0);
        p.shutdown();
    }

    #[test]
    fn the_least_recently_used_idle_model_makes_room_and_a_pool_of_busy_ones_refuses() {
        let f = Fakes::new();
        let p = pool(&f, None, 2);
        let a = p.acquire(&spec("a/a")).unwrap();
        p.release(a);
        std::thread::sleep(Duration::from_millis(10));
        let b = p.acquire(&spec("a/b")).unwrap();
        p.release(b);
        let c = p.acquire(&spec("a/c")).unwrap(); // (a is the least recently used)
        assert_eq!(p.loaded(), ["a/b", "a/c"]);
        let b = p.acquire(&spec("a/b")).unwrap();
        let err = p.acquire(&spec("a/d")).err().expect("both are in use");
        assert_eq!(err.0, 503, "{}", err.1);
        assert!(err.1.contains("busy"), "{}", err.1);
        p.release(b);
        p.release(c);
        p.shutdown();
    }

    #[test]
    fn two_different_models_that_arrive_together_do_not_both_take_the_last_place() {
        let f = Arc::new(Fakes { delay: Duration::from_millis(150), ..Arc::try_unwrap(Fakes::new()).ok().unwrap() });
        let p = Arc::new(Pool::new(Box::new(f.clone()), Config { keep_alive: None, max_loaded: 1, wait_for_room: Duration::from_secs(5) }));
        let ts: Vec<_> = ["a/a", "a/b"]
            .iter()
            .map(|n| {
                let (p, n) = (p.clone(), n.to_string());
                std::thread::spawn(move || {
                    let l = p.acquire(&spec(&n)).expect("one waits for the other's load, then takes its place");
                    std::thread::sleep(Duration::from_millis(20));
                    p.release(l);
                })
            })
            .collect();
        for t in ts {
            t.join().unwrap();
        }
        assert_eq!(p.loaded().len(), 1, "never more than the limit: {:?}", p.loaded());
        assert_eq!(f.starts.load(Ordering::SeqCst), 2);
        p.shutdown();
    }

    #[test]
    fn a_load_that_fails_for_want_of_memory_is_tried_again_with_the_idle_models_unloaded() {
        let f = Arc::new(Fakes { fail_when_more_than: Some(1), ..Arc::try_unwrap(Fakes::new()).ok().unwrap() });
        let p = pool(&f, None, 3);
        let l = p.acquire(&spec("a/small")).unwrap();
        p.release(l);
        let l = p.acquire(&spec("a/big")).unwrap(); // (does not fit beside small: small goes, big loads)
        assert_eq!(p.loaded(), ["a/big"]);
        // a model in use is not taken out for it
        let err = p.acquire(&spec("a/other")).err().expect("no room");
        assert_eq!(err.0, 503);
        assert!(err.1.contains("out of memory"), "{}", err.1);
        assert_eq!(p.loaded(), ["a/big"]);
        p.release(l);
        p.shutdown();
    }

    #[test]
    fn a_worker_that_ended_is_started_again() {
        let f = Fakes::new();
        let p = pool(&f, None, 2);
        let l = p.acquire(&spec("a/x")).unwrap();
        p.release(l);
        p.lock().entries[0].handle.stop(); // (it crashed)
        let l = p.acquire(&spec("a/x")).unwrap();
        p.release(l);
        assert_eq!(f.starts.load(Ordering::SeqCst), 2);
        p.shutdown();
    }

    #[test]
    fn unload_takes_a_named_model_or_all_the_idle_ones() {
        let f = Fakes::new();
        let p = pool(&f, None, 3);
        for n in ["a/a", "a/b", "a/c"] {
            let l = p.acquire(&spec(n)).unwrap();
            p.release(l);
        }
        let busy = p.acquire(&spec("a/c")).unwrap();
        assert_eq!(p.unload(Some("a/a")), 1);
        assert_eq!(p.unload(Some("a/zzz")), 0);
        assert_eq!(p.unload(None), 1, "b: c is in use");
        assert_eq!(p.loaded(), ["a/c"]);
        p.release(busy);
        p.shutdown();
    }

    /// A service on a free loopback port, with fake workers and `a/x` and `a/y` downloaded.
    fn live_service(f: &Arc<Fakes>, keep: Option<Duration>, key: Option<&str>, local_only: bool) -> (SocketAddr, Arc<Server>) {
        let have = || ["a/x", "a/y", "a/z"].iter().map(|r| models::Local { repo: r.to_string(), bytes: 1, level: "penguin".into(), name: None }).collect();
        let (addr, _stop, sv) =
            serve("127.0.0.1:0", pool(f, keep, 2), Manifest::load(), Box::new(have), key.map(str::to_string), SECRET.to_string(), local_only).unwrap();
        (addr, sv)
    }

    fn http(addr: SocketAddr, method: &str, path: &str, headers: &[(&str, &str)], body: &str) -> (u16, String) {
        let mut s = TcpStream::connect(addr).unwrap();
        let mut req = format!(
            "{method} {path} HTTP/1.1\r\nHost: {}\r\nConnection: close\r\n",
            headers.iter().find(|(k, _)| *k == "Host").map_or(addr.to_string(), |(_, v)| v.to_string())
        );
        for (k, v) in headers.iter().filter(|(k, _)| *k != "Host") {
            req.push_str(&format!("{k}: {v}\r\n"));
        }
        req.push_str(&format!("Content-Length: {}\r\n\r\n{body}", body.len()));
        s.write_all(req.as_bytes()).unwrap();
        let mut out = String::new();
        s.read_to_string(&mut out).unwrap();
        let (head, body) = out.split_once("\r\n\r\n").unwrap_or((&out, ""));
        (head.split(' ').nth(1).and_then(|c| c.parse().ok()).unwrap_or(0), body.to_string())
    }

    #[test]
    fn the_service_answers_for_itself_refuses_what_serve_refuses_and_guards_v1_with_the_key() {
        let f = Fakes::new();
        let (addr, _sv) = live_service(&f, None, Some("k"), true);
        assert_eq!(http(addr, "GET", "/health", &[], "").0, 200);
        assert_eq!(http(addr, "GET", "/", &[], "").0, 200);
        // v1 needs the key; the page and the health check do not
        assert_eq!(http(addr, "GET", "/v1/models", &[], "").0, 401);
        let auth = [("Authorization", "Bearer k")];
        let (c, b) = http(addr, "GET", "/v1/models", &auth, "");
        assert!(c == 200 && b.contains("\"id\":\"a/x\""), "{b}");
        let (c, b) = http(addr, "GET", "/v1/glyd/service", &auth, "");
        assert!(c == 200 && b.contains("\"service\":\"glyd\"") && b.contains("\"models\":[]"), "{b}");
        // a name that DNS points here, and another page's script
        assert_eq!(http(addr, "GET", "/health", &[("Host", "evil.example:11434")], "").0, 421);
        assert_eq!(http(addr, "GET", "/health", &[("Origin", "http://evil.example")], "").0, 403);
        // no model named, a path, a name that is not downloaded
        let (c, b) = http(addr, "POST", "/v1/chat/completions", &auth, "{\"messages\":[]}");
        assert!(c == 404 && b.contains("names no model"), "{b}");
        let (c, b) = http(addr, "POST", "/v1/chat/completions", &auth, "{\"model\":\"/etc/passwd\"}");
        assert!(c == 404 && b.contains("not by path"), "{b}");
        let (c, b) = http(addr, "POST", "/v1/chat/completions", &auth, "{\"model\":\"b/nothere\"}");
        assert!(c == 404 && b.contains("is not downloaded: glyd pull b/nothere"), "{b}");
        assert_eq!(f.starts.load(Ordering::SeqCst), 0, "nothing was loaded for a refusal");
        assert_eq!(http(addr, "GET", "/nothing", &[], "").0, 404);
        assert_eq!(
            http(addr, "POST", "/v1/glyd/unload", &[("Authorization", "Bearer k"), ("X-Glyd-Auth", &auth_for(addr, "/v1/glyd/unload", "{}"))], "{}").0,
            400
        );
    }

    fn free_port() -> std::io::Result<u16> {
        Ok(TcpListener::bind("127.0.0.1:0")?.local_addr()?.port())
    }

    const SECRET: &str = "s3cret-s3cret-s3cret-s3cret-s3cret-s3cret";

    #[test]
    fn the_service_proves_it_has_the_secret_and_another_listener_cannot() {
        let f = Fakes::new();
        let (addr, _sv) = live_service(&f, None, None, true);
        assert!(verified_with(("127.0.0.1", addr.port()), SECRET));
        assert!(!verified_with(("127.0.0.1", addr.port()), "another-accounts-secret-another-accounts"));
        // a program that took the port: it answers /health the same, and has no proof
        let squat = TcpListener::bind("127.0.0.1:0").unwrap();
        let sp = squat.local_addr().unwrap().port();
        std::thread::spawn(move || {
            for c in squat.incoming().flatten() {
                let mut c = c;
                let _ = read_request(&mut c);
                respond(&mut c, 200, "application/json", "{\"status\":\"ok\",\"service\":\"glyd\",\"proof\":\"0000\"}");
            }
        });
        assert!(!verified_with(("127.0.0.1", sp), SECRET));
        assert!(!verified_with(("127.0.0.1", free_port().unwrap()), SECRET), "nothing listens");
        // the secret is only ever a hash on the wire
        let (_, body) = http(addr, "GET", "/health?challenge=abc", &[], "");
        assert!(!body.contains(SECRET) && body.contains(&proof(SECRET, addr.port(), "abc")), "{body}");
        // an answer is for the port that gave it
        assert_ne!(proof(SECRET, 1, "abc"), proof(SECRET, 2, "abc"));
        assert_ne!(random_hex(16), random_hex(16));
    }

    /// The control header of a request to the test service, as `glyd stop` makes it.
    fn auth_for(addr: SocketAddr, path: &str, body: &str) -> String {
        control_auth(SECRET, addr.port(), "POST", path, body, now_secs())
    }

    #[test]
    fn hmac_is_hmac() {
        // RFC 4231, test case 2
        assert_eq!(hex(&hmac_sha256(b"Jefe", b"what do ya want for nothing?")), "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843");
        // a key longer than a block
        assert_eq!(
            hex(&hmac_sha256(&[0xaa; 131], b"Test Using Larger Than Block-Size Key - Hash Key First")),
            "60e431591ee0b67f0d8a26aacbf5b77f8e0bc6213728c5140546040f0ee37f54"
        );
    }

    #[test]
    fn stopping_and_unloading_are_the_accounts_own() {
        let f = Fakes::new();
        let (addr, sv) = live_service(&f, None, None, true);
        let l = sv.pool.acquire(&spec("a/x")).unwrap();
        sv.pool.release(l);
        let body = "{\"model\":\"a/x\"}";
        let u = "/v1/glyd/unload";
        assert_eq!(http(addr, "POST", u, &[], body).0, 403);
        assert_eq!(http(addr, "POST", u, &[("X-Glyd-Auth", "1.00")], body).0, 403);
        assert_eq!(http(addr, "POST", "/v1/glyd/shutdown", &[], "{}").0, 403);
        // another secret, another port's, another request's, a stale one: none is good
        let other = control_auth("another-accounts-secret-another-accounts", addr.port(), "POST", u, body, now_secs());
        assert_eq!(http(addr, "POST", u, &[("X-Glyd-Auth", &other)], body).0, 403);
        let elsewhere = control_auth(SECRET, addr.port() ^ 1, "POST", u, body, now_secs());
        assert_eq!(http(addr, "POST", u, &[("X-Glyd-Auth", &elsewhere)], body).0, 403);
        assert_eq!(http(addr, "POST", u, &[("X-Glyd-Auth", &auth_for(addr, u, "{\"all\":true}"))], body).0, 403, "a MAC of another body");
        assert_eq!(http(addr, "POST", "/v1/glyd/shutdown", &[("X-Glyd-Auth", &auth_for(addr, u, body))], "{}").0, 403, "a MAC of another path");
        let stale = control_auth(SECRET, addr.port(), "POST", u, body, now_secs() - 120);
        assert_eq!(http(addr, "POST", u, &[("X-Glyd-Auth", &stale)], body).0, 403);
        assert_eq!(sv.pool.loaded(), ["a/x"], "nothing was unloaded for them");
        let (c, b) = http(addr, "POST", u, &[("X-Glyd-Auth", &auth_for(addr, u, body))], body);
        assert!(c == 200 && b.contains("\"unloaded\":1"), "{b}");
        sv.pool.shutdown();
    }

    #[test]
    fn an_address_the_user_chose_takes_any_host() {
        let f = Fakes::new();
        let (addr, _sv) = live_service(&f, None, None, false);
        assert_eq!(http(addr, "GET", "/health", &[("Host", "192.168.1.5:11434")], "").0, 200);
    }

    #[test]
    fn a_request_loads_its_model_streams_the_answer_and_the_model_is_unloaded_when_idle() {
        let f = Fakes::new();
        let (addr, sv) = live_service(&f, Some(Duration::from_millis(300)), None, true);
        let ask = |model: &str| {
            http(
                addr,
                "POST",
                "/v1/chat/completions",
                &[("Origin", &format!("http://{addr}")), ("Content-Type", "application/json")],
                &format!("{{\"model\":\"{model}\",\"stream\":true}}"),
            )
        };
        let (c, b) = ask("a/x");
        assert_eq!(c, 200, "{b}");
        // the worker saw its own address as Host, no Origin (the service's checks were the ones that counted), and the body as it was sent
        assert!(
            b.contains("\"host\":\"127.0.0.1:")
                && b.contains("\"origin\":false")
                && b.contains("\\\"model\\\":\\\"a/x\\\"")
                && b.trim_end().ends_with("data: [DONE]"),
            "{b}"
        );
        assert_eq!(ask("a/x").0, 200);
        assert_eq!(f.starts.load(Ordering::SeqCst), 1, "loaded once, used twice");
        let (_, st) = http(addr, "GET", "/v1/glyd/service", &[], "");
        assert!(st.contains("\"name\":\"a/x\"") && st.contains("\"requests\":2"), "{st}");
        // idle for the keep-alive: the service takes it out (its reaper runs twice a second)
        let t0 = Instant::now();
        while !sv.pool.loaded().is_empty() && t0.elapsed() < Duration::from_secs(5) {
            std::thread::sleep(Duration::from_millis(50));
        }
        assert!(sv.pool.loaded().is_empty(), "still loaded after the keep-alive");
        assert_eq!(ask("a/x").0, 200);
        assert_eq!(f.starts.load(Ordering::SeqCst), 2, "loaded again on the next request");
        // glyd stop: unload by name
        let (c, b) =
            http(addr, "POST", "/v1/glyd/unload", &[("X-Glyd-Auth", &auth_for(addr, "/v1/glyd/unload", "{\"model\":\"a/x\"}"))], "{\"model\":\"a/x\"}");
        assert!(c == 200 && b.contains("\"unloaded\":1"), "{b}");
        sv.pool.shutdown();
    }

    #[test]
    fn two_models_are_held_and_a_third_makes_the_oldest_idle_one_leave() {
        let f = Fakes::new();
        let (addr, sv) = live_service(&f, None, None, true);
        let ask = |model: &str| http(addr, "POST", "/v1/chat/completions", &[], &format!("{{\"model\":\"{model}\"}}")).0;
        assert_eq!((ask("a/x"), ask("a/y")), (200, 200));
        assert_eq!(sv.pool.loaded(), ["a/x", "a/y"]);
        assert_eq!(ask("a/z"), 200);
        assert_eq!(sv.pool.loaded(), ["a/y", "a/z"]);
        sv.pool.shutdown();
    }
}
