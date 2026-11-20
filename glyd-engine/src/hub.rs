//! The Hugging Face Hub, for a model that is not on this machine yet: the repo's files listed, then downloaded into the cache `huggingface_hub` keeps (`$HF_HUB_CACHE`, `models--ORG--NAME/` with
//! `blobs/`, `snapshots/<commit>/` and `refs/`), so that every tool on the machine shares one copy. A large file comes as ranges on several connections, is kept where it stopped (a run again
//! goes on from there) and is checked by size and, where the Hub gives it, by SHA-256. The token (HF_TOKEN, or the file `hf login` wrote) goes to the Hub's own host only, never to the
//! storage a file redirects to.

use crate::os::FileExt;
use glyd_gpu::json::Value;
use sha2::{Digest, Sha256};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Duration;

const RANGE: u64 = 32 << 20;
const TRIES: u32 = 5;

/// Why the Hub did not give what was asked.
#[derive(Debug)]
pub enum HubError {
    /// 401, 403 or 404: no such repo, or a gated or private one (the Hub answers all three alike to a visitor).
    Denied(u16),
    /// No answer: the network, the Hub's 5xx, a connection that broke.
    Network(String),
    /// Anything else: a disk that is full, a file that does not match its checksum, an answer that makes no sense.
    Other(String),
}

impl std::fmt::Display for HubError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HubError::Denied(c) => write!(f, "the Hub answered {c}"),
            HubError::Network(m) | HubError::Other(m) => f.write_str(m),
        }
    }
}

impl std::error::Error for HubError {}

impl From<HubError> for crate::ffi::Error {
    fn from(e: HubError) -> crate::ffi::Error {
        crate::ffi::Error(e.to_string())
    }
}

type Res<T> = std::result::Result<T, HubError>;

fn net(e: impl std::fmt::Display) -> HubError {
    HubError::Network(e.to_string())
}

fn other(e: impl std::fmt::Display) -> HubError {
    HubError::Other(e.to_string())
}

/// A file of a repo as the Hub lists it.
#[derive(Clone, Debug, PartialEq)]
pub struct Entry {
    pub path: String,
    pub size: u64,
    /// The file's name in `blobs/`: its SHA-256 where it is a large file (LFS), else its git blob id.
    pub blob: String,
    pub sha256: Option<String>,
}

/// A repo at one commit.
#[derive(Clone, Debug)]
pub struct Remote {
    pub repo: String,
    /// The name it was asked by (a branch, a tag or a commit) and the commit that stands for.
    pub revision: String,
    pub commit: String,
    pub files: Vec<Entry>,
    /// The checkpoint's parameters by dtype ("BF16": 8190735360), where the Hub counts them.
    pub dtypes: Vec<(String, u64)>,
}

pub struct Hub {
    endpoint: String,
    token: Option<String>,
    /// Connections as the system routes them (IPv6 first, where the name has an address of it).
    agent: ureq::Agent,
    /// The same, to IPv4 addresses only.
    agent4: ureq::Agent,
    /// Whether every request goes by IPv4: GLYD_IPV4=1 says so, and so does an IPv6 request that failed where IPv4 then worked (see `send`).
    v4: AtomicBool,
    cache: PathBuf,
    range: u64,
    threads: usize,
}

/// The cache directory as huggingface_hub names it: $HF_HUB_CACHE, else $HF_HOME/hub, else ~/.cache/huggingface/hub.
pub fn hub_cache() -> PathBuf {
    let var = |k: &str| std::env::var_os(k).filter(|v| !v.is_empty()).map(PathBuf::from);
    if let Some(p) = var("HF_HUB_CACHE").or_else(|| var("HUGGINGFACE_HUB_CACHE")) {
        return p;
    }
    let home = var("HF_HOME").unwrap_or_else(|| var("XDG_CACHE_HOME").unwrap_or_else(|| Path::new(&crate::os::home()).join(".cache")).join("huggingface"));
    home.join("hub")
}

/// A token given as a flag (`--hf-token TOKEN`): it comes before the environment's and the saved login's.
static TOKEN_FLAG: std::sync::Mutex<Option<String>> = std::sync::Mutex::new(None);

/// Sets the token a flag gave (an empty one is none).
pub fn set_token(t: &str) {
    if let Ok(mut g) = TOKEN_FLAG.lock() {
        *g = Some(t.trim().to_string()).filter(|t| !t.is_empty());
    }
}

/// The token, from where it is first found: the flag, HF_TOKEN, HUGGING_FACE_HUB_TOKEN, the saved login (`file`). A blank one is none.
fn pick_token(flag: Option<String>, env: &dyn Fn(&str) -> Option<String>, file: Option<String>) -> Option<String> {
    let clean = |t: Option<String>| t.map(|t| t.trim().to_string()).filter(|t| !t.is_empty());
    clean(flag).or_else(|| ["HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"].iter().find_map(|k| clean(env(k)))).or_else(|| clean(file))
}

/// The user's token: the flag `--hf-token`, HF_TOKEN (or HUGGING_FACE_HUB_TOKEN), else the file `hf login` and `glyd login` write (HF_TOKEN_PATH, else $HF_HOME/token).
pub fn token() -> Option<String> {
    let flag = TOKEN_FLAG.lock().ok().and_then(|g| g.clone());
    let path = match std::env::var_os("HF_TOKEN_PATH").filter(|v| !v.is_empty()) {
        Some(p) => PathBuf::from(p),
        None => {
            let home = std::env::var_os("HF_HOME").filter(|v| !v.is_empty()).map(PathBuf::from).unwrap_or_else(|| {
                std::env::var_os("XDG_CACHE_HOME")
                    .filter(|v| !v.is_empty())
                    .map(PathBuf::from)
                    .unwrap_or_else(|| Path::new(&crate::os::home()).join(".cache"))
                    .join("huggingface")
            });
            home.join("token")
        }
    };
    pick_token(flag, &|k| std::env::var(k).ok(), std::fs::read_to_string(path).ok())
}

/// Whether a variable is switched on the way the Hub's own tools read one.
fn env_on(name: &str) -> bool {
    std::env::var(name).is_ok_and(|v| matches!(v.as_str(), "1" | "true" | "TRUE" | "True" | "yes"))
}

/// Whether HF_HUB_OFFLINE asks for no network at all.
pub fn offline() -> bool {
    env_on("HF_HUB_OFFLINE")
}

fn host_of(url: &str) -> &str {
    let rest = url.split_once("://").map_or(url, |(_, r)| r);
    rest.split('/').next().unwrap_or("")
}

/// Where a redirect's `Location` points, from the URL that gave it.
fn join(base: &str, loc: &str) -> String {
    if loc.contains("://") {
        return loc.to_string();
    }
    let (scheme, rest) = base.split_once("://").unwrap_or(("https", base));
    let host = rest.split('/').next().unwrap_or("");
    if loc.starts_with('/') {
        return format!("{scheme}://{host}{loc}");
    }
    let dir = rest.split(['?', '#']).next().unwrap_or("").rsplit_once('/').map_or("", |(d, _)| d);
    format!("{scheme}://{dir}/{loc}")
}

/// The free bytes of the filesystem holding `path` (an existing directory), where it can be told (on Linux).
#[cfg(not(target_os = "linux"))]
pub fn disk_free(_path: &Path) -> Option<u64> {
    None
}

#[cfg(target_os = "linux")]
pub fn disk_free(path: &Path) -> Option<u64> {
    #[repr(C)]
    struct StatVfs {
        f_bsize: u64,
        f_frsize: u64,
        f_blocks: u64,
        f_bfree: u64,
        f_bavail: u64,
        rest: [u64; 11],
    }
    extern "C" {
        fn statvfs(path: *const std::os::raw::c_char, buf: *mut StatVfs) -> i32;
    }
    let c = std::ffi::CString::new(path.as_os_str().as_encoded_bytes()).ok()?;
    let mut s = StatVfs { f_bsize: 0, f_frsize: 0, f_blocks: 0, f_bfree: 0, f_bavail: 0, rest: [0; 11] };
    // SAFETY: a NUL-ended path and a buffer larger than glibc's and musl's struct statvfs on 64-bit Linux (112 bytes: 5 words above and 11 here).
    if unsafe { statvfs(c.as_ptr(), &mut s) } != 0 {
        return None;
    }
    Some(s.f_bavail.saturating_mul(if s.f_frsize > 0 { s.f_frsize } else { s.f_bsize }))
}

impl Hub {
    /// The Hub as the environment names it (HF_ENDPOINT, the token, the cache); `GLYD_DOWNLOAD_THREADS` connections for a large file (8); GLYD_IPV4=1 for IPv4 alone.
    pub fn from_env() -> Hub {
        let threads = crate::tune!("GLYD_DOWNLOAD_THREADS").and_then(|v| v.parse().ok()).filter(|&n| n > 0).unwrap_or(8usize);
        let hub = Hub::new(
            &std::env::var("HF_ENDPOINT").ok().filter(|v| !v.is_empty()).unwrap_or_else(|| "https://huggingface.co".to_string()),
            token(),
            hub_cache(),
            RANGE,
            threads,
        );
        hub.v4.store(env_on("GLYD_IPV4"), Ordering::Relaxed);
        hub
    }

    pub fn new(endpoint: &str, token: Option<String>, cache: PathBuf, range: u64, threads: usize) -> Hub {
        let agent = |family| {
            ureq::Agent::new_with_config(
                ureq::Agent::config_builder()
                    .ip_family(family)
                    .http_status_as_error(false)
                    .max_redirects(0)
                    .timeout_connect(Some(Duration::from_secs(15)))
                    .timeout_recv_response(Some(Duration::from_secs(60)))
                    .timeout_recv_body(Some(Duration::from_secs(60)))
                    .user_agent(concat!("glyd-engine/", env!("CARGO_PKG_VERSION")))
                    .build(),
            )
        };
        Hub {
            endpoint: endpoint.trim_end_matches('/').to_string(),
            token,
            agent: agent(ureq::config::IpFamily::Any),
            agent4: agent(ureq::config::IpFamily::Ipv4Only),
            v4: AtomicBool::new(false),
            cache,
            range: range.max(1),
            threads: threads.max(1),
        }
    }

    pub fn cache(&self) -> &Path {
        &self.cache
    }

    pub fn has_token(&self) -> bool {
        self.token.is_some()
    }

    /// One GET with its headers. Over IPv4 alone once `v4` says so. Otherwise as the system routes it, and where that fails (no connection, no answer before the timeout, a connection reset: a
    /// network that reaches the Hub over IPv6 but not the storage it redirects to does all three) once more over IPv4; if that works, IPv4 is the way for the rest of the run. A failure on both
    /// is the first one's (a machine with no IPv4 address to try says nothing of what went wrong).
    fn send(&self, url: &str, headers: &[(&str, String)]) -> Res<ureq::http::Response<ureq::Body>> {
        let once = |agent: &ureq::Agent| {
            let mut req = agent.get(url);
            for (k, v) in headers {
                req = req.header(*k, v);
            }
            req.call()
        };
        if self.v4.load(Ordering::Relaxed) {
            return once(&self.agent4).map_err(net);
        }
        match once(&self.agent) {
            Ok(resp) => Ok(resp),
            Err(first) => match once(&self.agent4) {
                Ok(resp) => {
                    if !self.v4.swap(true, Ordering::Relaxed) {
                        eprintln!("\nthe connection over IPv6 failed ({first}); going on over IPv4 (GLYD_IPV4=1 skips the wait)");
                    }
                    Ok(resp)
                }
                Err(_) => Err(net(first)),
            },
        }
    }

    /// A GET, the redirects followed by hand: the token only to the host it was meant for.
    fn get(&self, url: &str, range: Option<(u64, u64)>) -> Res<ureq::http::Response<ureq::Body>> {
        let home = host_of(&self.endpoint).to_string();
        let mut url = url.to_string();
        for _ in 0..6 {
            let mut headers = Vec::new();
            if let (Some(t), true) = (&self.token, host_of(&url) == home) {
                headers.push(("Authorization", format!("Bearer {t}")));
            }
            if let Some((a, b)) = range {
                headers.push(("Range", format!("bytes={a}-{b}")));
            }
            let resp = self.send(&url, &headers)?;
            if matches!(resp.status().as_u16(), 301 | 302 | 303 | 307 | 308) {
                let loc = resp.headers().get("location").and_then(|v| v.to_str().ok()).ok_or_else(|| other("a redirect without a place to go"))?;
                url = join(&url, loc);
                continue;
            }
            return Ok(resp);
        }
        Err(other("too many redirects"))
    }

    fn json(&self, url: &str) -> Res<Value> {
        let mut r = self.get(url, None)?;
        let s = r.status().as_u16();
        if matches!(s, 401 | 403 | 404) {
            return Err(HubError::Denied(s));
        }
        if s != 200 {
            return Err(net(format!("{url}: {s}")));
        }
        let text = r.body_mut().with_config().limit(64 << 20).read_to_string().map_err(net)?;
        glyd_gpu::json::parse(&text).map_err(|e| other(format!("{url}: {e}")))
    }

    /// The repo's files at `revision` (a branch, a tag or a commit): the Hub's listing with sizes, checksums and the commit the name stands for.
    pub fn list(&self, repo: &str, revision: &str) -> Res<Remote> {
        let v = self.json(&format!("{}/api/models/{repo}/revision/{revision}?blobs=true", self.endpoint))?;
        let commit = v.get("sha").and_then(Value::as_str).ok_or_else(|| other(format!("{repo}: the Hub named no commit")))?.to_string();
        let mut files = Vec::new();
        for f in v.get("siblings").and_then(Value::as_array).unwrap_or(&[]) {
            let Some(path) = f.get("rfilename").and_then(Value::as_str) else { continue };
            let lfs = f.get("lfs");
            let sha256 = lfs.and_then(|l| l.get("sha256")).and_then(Value::as_str).map(str::to_string);
            let size = lfs.and_then(|l| l.get("size")).and_then(Value::as_u64).or_else(|| f.get("size").and_then(Value::as_u64)).unwrap_or(0);
            let blob = sha256
                .clone()
                .or_else(|| f.get("blobId").and_then(Value::as_str).map(str::to_string))
                .ok_or_else(|| other(format!("{repo}: {path} has no id")))?;
            files.push(Entry { path: path.to_string(), size, blob, sha256 });
        }
        let dtypes = v
            .get("safetensors")
            .and_then(|s| s.get("parameters"))
            .and_then(Value::as_object)
            .map(|o| o.iter().filter_map(|(k, n)| n.as_u64().map(|n| (k.clone(), n))).collect())
            .unwrap_or_default();
        Ok(Remote { repo: repo.to_string(), revision: revision.to_string(), commit, files, dtypes })
    }

    /// A small file of a repo as text (config.json of the model's main branch, to see what it is before anything is downloaded).
    pub fn text(&self, repo: &str, revision: &str, path: &str) -> Res<String> {
        let mut r = self.get(&format!("{}/{repo}/resolve/{revision}/{path}", self.endpoint), None)?;
        let s = r.status().as_u16();
        if matches!(s, 401 | 403 | 404) {
            return Err(HubError::Denied(s));
        }
        if s != 200 {
            return Err(net(format!("{repo}/{path}: {s}")));
        }
        r.body_mut().with_config().limit(16 << 20).read_to_string().map_err(net)
    }

    /// Whose token this is (the Hub's `whoami`), or Denied where it is not accepted.
    pub fn whoami(&self) -> Res<String> {
        let v = self.json(&format!("{}/api/whoami-v2", self.endpoint))?;
        Ok(v.get("name").and_then(Value::as_str).unwrap_or("").to_string())
    }

    pub fn endpoint(&self) -> &str {
        &self.endpoint
    }

    fn repo_dir(&self, repo: &str) -> PathBuf {
        self.cache.join(format!("models--{}", repo.replace('/', "--")))
    }

    /// The snapshot directory of `remote` (its files linked in as they complete).
    pub fn snapshot(&self, remote: &Remote) -> PathBuf {
        self.repo_dir(&remote.repo).join("snapshots").join(&remote.commit)
    }

    /// Of `wanted` (a subset of the remote's files), the bytes not in the snapshot yet.
    pub fn missing(&self, remote: &Remote, wanted: &[Entry]) -> u64 {
        let snap = self.snapshot(remote);
        wanted.iter().filter(|e| std::fs::metadata(snap.join(&e.path)).map_or(true, |m| m.len() != e.size)).map(|e| e.size).sum()
    }

    /// The `wanted` files of `remote` in the cache, each downloaded unless it is there (a file of the size the Hub gave), `progress(done, total)` called as bytes arrive (done counts the
    /// files already there). Returns the snapshot directory.
    pub fn fetch(&self, remote: &Remote, wanted: &[Entry], progress: &dyn Fn(u64, u64)) -> Res<PathBuf> {
        let root = self.repo_dir(&remote.repo);
        let (blobs, snap) = (root.join("blobs"), root.join("snapshots").join(&remote.commit));
        std::fs::create_dir_all(&blobs).map_err(other)?;
        std::fs::create_dir_all(&snap).map_err(other)?;
        let total: u64 = wanted.iter().map(|e| e.size).sum();
        let done = AtomicU64::new(0);
        for e in wanted {
            let link = snap.join(&e.path);
            if std::fs::metadata(&link).is_ok_and(|m| m.len() == e.size) {
                done.fetch_add(e.size, Ordering::Relaxed);
                continue;
            }
            progress(done.load(Ordering::Relaxed), total);
            let blob = self.fetch_blob(remote, e, &blobs, &done, total, progress)?;
            if let Some(dir) = link.parent() {
                std::fs::create_dir_all(dir).map_err(other)?;
            }
            let _ = std::fs::remove_file(&link);
            let depth = e.path.matches('/').count() + 2;
            let target = format!("{}blobs/{}", "../".repeat(depth), e.blob);
            #[cfg(unix)]
            std::os::unix::fs::symlink(&target, &link).or_else(|_| std::fs::copy(&blob, &link).map(|_| ())).map_err(other)?;
            // (Windows: a hard link, which needs no rights a symbolic one does; a copy where the cache is on a filesystem without them)
            #[cfg(windows)]
            std::fs::hard_link(&blob, &link).or_else(|_| std::fs::copy(&blob, &link).map(|_| ())).map_err(other)?;
            #[cfg(windows)]
            let _ = target;
            progress(done.load(Ordering::Relaxed), total);
        }
        Ok(snap)
    }

    /// Records the commit under the name it was asked by (`refs/main`), which is what makes the snapshot the one a lookup by name finds: after the whole set of files is in, not before.
    pub fn pin(&self, remote: &Remote) -> Res<()> {
        if remote.revision != remote.commit {
            let refs = self.repo_dir(&remote.repo).join("refs");
            std::fs::create_dir_all(&refs).map_err(other)?;
            std::fs::write(refs.join(&remote.revision), &remote.commit).map_err(other)?;
        }
        Ok(())
    }

    /// The Hub's reply to a request for a range of a file: where the bytes are (the Hub's `resolve`, which redirects to the storage).
    fn resolve_url(&self, remote: &Remote, e: &Entry) -> String {
        let path: String = e.path.split('/').map(|p| p.replace(' ', "%20")).collect::<Vec<_>>().join("/");
        format!("{}/{}/resolve/{}/{path}", self.endpoint, remote.repo, remote.commit)
    }

    #[allow(clippy::chunks_exact_to_as_chunks)] // (as_chunks needs a newer compiler than some machines have)
    fn fetch_blob(&self, remote: &Remote, e: &Entry, blobs: &Path, done: &AtomicU64, total: u64, progress: &dyn Fn(u64, u64)) -> Res<PathBuf> {
        let blob = blobs.join(&e.blob);
        if std::fs::metadata(&blob).is_ok_and(|m| m.len() == e.size) {
            done.fetch_add(e.size, Ordering::Relaxed);
            return Ok(blob);
        }
        let part = blobs.join(format!("{}.incomplete", e.blob));
        let map_path = blobs.join(format!("{}.ranges", e.blob));
        let file = std::fs::OpenOptions::new().read(true).write(true).create(true).truncate(false).open(&part).map_err(other)?;
        let n = e.size.div_ceil(self.range).max(1) as usize;
        let mut have = vec![false; n];
        if file.metadata().map_err(other)?.len() == e.size {
            if let Ok(b) = std::fs::read(&map_path) {
                for c in b.chunks_exact(4) {
                    let i = u32::from_le_bytes(c.try_into().unwrap()) as usize;
                    if i < n {
                        have[i] = true;
                    }
                }
            }
        } else {
            crate::os::sparse(&file);
            file.set_len(e.size).map_err(other)?;
            let _ = std::fs::remove_file(&map_path);
        }
        let range_len = |i: usize| (e.size - i as u64 * self.range).min(self.range);
        let todo: Vec<usize> = (0..n).filter(|&i| !have[i]).collect();
        done.fetch_add(have.iter().enumerate().filter(|(_, &h)| h).map(|(i, _)| range_len(i)).sum::<u64>(), Ordering::Relaxed);
        if !todo.is_empty() {
            let map = Mutex::new(std::fs::OpenOptions::new().create(true).append(true).open(&map_path).map_err(other)?);
            let url = Mutex::new((self.resolve(remote, e)?, std::time::Instant::now()));
            let next = AtomicUsize::new(0);
            let failed: Mutex<Option<HubError>> = Mutex::new(None);
            let work = || loop {
                let k = next.fetch_add(1, Ordering::SeqCst);
                if k >= todo.len() || failed.lock().unwrap().is_some() {
                    break;
                }
                let i = todo[k];
                let a = i as u64 * self.range;
                match self.range_to(remote, e, &url, &file, a, range_len(i), a, done) {
                    Ok(()) => {
                        use std::io::Write;
                        let _ = map.lock().unwrap().write_all(&(i as u32).to_le_bytes());
                    }
                    Err(err) => {
                        failed.lock().unwrap().get_or_insert(err);
                        break;
                    }
                }
            };
            ramped(self.threads, &|| todo.len().saturating_sub(next.load(Ordering::Relaxed)), done, &work, &|| progress(done.load(Ordering::Relaxed), total));
            if let Some(err) = failed.into_inner().unwrap() {
                return Err(err);
            }
        }
        file.sync_all().map_err(other)?;
        drop(file);
        if let Some(want) = &e.sha256 {
            let got = sha256_of(&part).map_err(other)?;
            if &got != want {
                let _ = std::fs::remove_file(&part);
                let _ = std::fs::remove_file(&map_path);
                return Err(other(format!(
                    "{}: the download does not match the Hub's checksum (SHA-256 {got}, wanted {want}); it will be fetched again on the next run",
                    e.path
                )));
            }
        }
        std::fs::rename(&part, &blob).map_err(other)?;
        let _ = std::fs::remove_file(&map_path);
        Ok(blob)
    }

    /// Where a file's bytes are served from now: the storage's address after the Hub's redirects (it expires, so it is asked for again after a while).
    fn resolve(&self, remote: &Remote, e: &Entry) -> Res<String> {
        let home = host_of(&self.endpoint).to_string();
        let mut url = self.resolve_url(remote, e);
        for _ in 0..6 {
            let mut headers = vec![("Range", "bytes=0-0".to_string())];
            if let (Some(t), true) = (&self.token, host_of(&url) == home) {
                headers.push(("Authorization", format!("Bearer {t}")));
            }
            let resp = self.send(&url, &headers)?;
            let s = resp.status().as_u16();
            if matches!(s, 301 | 302 | 303 | 307 | 308) {
                let loc = resp.headers().get("location").and_then(|v| v.to_str().ok()).ok_or_else(|| other("a redirect without a place to go"))?;
                let next = join(&url, loc);
                // the storage is where the token stops: its address is the one the ranges are asked of
                if host_of(&next) != home {
                    return Ok(next);
                }
                url = next;
                continue;
            }
            return match s {
                200 | 206 => Ok(url),
                401 | 403 | 404 => Err(HubError::Denied(s)),
                _ => Err(net(format!("{}: {s}", e.path))),
            };
        }
        Err(other("too many redirects"))
    }

    /// Bytes `a .. a + len` of a file into `file` from offset `at` on; a failed try is made again (a new address for the storage where its was refused).
    #[allow(clippy::too_many_arguments)]
    fn range_to(
        &self,
        remote: &Remote,
        e: &Entry,
        url: &Mutex<(String, std::time::Instant)>,
        file: &std::fs::File,
        a: u64,
        len: u64,
        at: u64,
        done: &AtomicU64,
    ) -> Res<()> {
        let mut last = net("no try");
        for attempt in 0..TRIES {
            if attempt > 0 {
                std::thread::sleep(Duration::from_secs(1 << (attempt - 1).min(4)));
            }
            let target = {
                let mut u = url.lock().unwrap();
                if u.1.elapsed() > Duration::from_secs(600) || matches!(last, HubError::Denied(_)) {
                    *u = (self.resolve(remote, e)?, std::time::Instant::now());
                }
                u.0.clone()
            };
            let mut got = 0u64;
            let r = (|| -> Res<()> {
                let mut resp = self.get(&target, Some((a, a + len - 1)))?;
                let s = resp.status().as_u16();
                if !(s == 206 || (s == 200 && a == 0 && len == e.size)) {
                    return Err(if matches!(s, 401 | 403) {
                        HubError::Denied(s)
                    } else if s == 200 {
                        other("the server does not do ranges")
                    } else {
                        net(format!("{}: {s}", e.path))
                    });
                }
                let mut r = resp.body_mut().as_reader();
                let mut buf = vec![0u8; 1 << 20];
                while got < len {
                    let want = buf.len().min((len - got) as usize);
                    let n = r.read(&mut buf[..want]).map_err(net)?;
                    if n == 0 {
                        return Err(net(format!("{}: the connection ended {} bytes short", e.path, len - got)));
                    }
                    file.write_all_at(&buf[..n], at + got)
                        .map_err(|err| if err.raw_os_error() == Some(28) { other("the disk is full") } else { other(err) })?;
                    got += n as u64;
                    done.fetch_add(n as u64, Ordering::Relaxed);
                }
                Ok(())
            })();
            match r {
                Ok(()) => return Ok(()),
                Err(HubError::Other(m)) => {
                    done.fetch_sub(got, Ordering::Relaxed);
                    return Err(HubError::Other(m));
                }
                Err(err) => {
                    done.fetch_sub(got, Ordering::Relaxed);
                    last = err;
                }
            }
        }
        Err(last)
    }

    /// A safetensors file's header by two ranges (its length, then its JSON): the tensors' (dtype, shape, bytes `a .. b` from the data's start) by name, and where the data starts.
    pub(crate) fn header(&self, remote: &Remote, e: &Entry) -> Res<(Vec<(String, String, Vec<u64>, u64, u64)>, u64)> {
        let url = self.resolve(remote, e)?;
        let bytes = |a: u64, b: u64| -> Res<Vec<u8>> {
            let mut resp = self.get(&url, Some((a, b)))?;
            if resp.status().as_u16() != 206 {
                return Err(net(format!("{}: {} for a range", e.path, resp.status().as_u16())));
            }
            resp.body_mut().with_config().limit(256 << 20).read_to_vec().map_err(net)
        };
        let n = u64::from_le_bytes(bytes(0, 7)?.try_into().map_err(|_| other(format!("{}: no header length", e.path)))?);
        let text = String::from_utf8(bytes(8, 8 + n - 1)?).map_err(other)?;
        let v = glyd_gpu::json::parse(&text).map_err(|err| other(format!("{}: {err}", e.path)))?;
        let mut out = Vec::new();
        for (name, t) in v.as_object().unwrap_or(&[]) {
            if name == "__metadata__" {
                continue;
            }
            let (dtype, shape, off) =
                (t.get("dtype").and_then(Value::as_str), t.get("shape").and_then(Value::as_u64s), t.get("data_offsets").and_then(Value::as_u64s));
            match (dtype, shape, off) {
                (Some(d), Some(sh), Some(o)) if o.len() == 2 => out.push((name.clone(), d.to_string(), sh, o[0], o[1])),
                _ => return Err(other(format!("{}: {name}: not a tensor's entry", e.path))),
            }
        }
        Ok((out, 8 + n))
    }

    /// A checkpoint cut to its first `layers` layers, fetched by ranges: of each safetensors file its header, then the bytes of the tensors `keep` takes (by name and
    /// shape: the shape it keeps, the same or fewer rows, which are the first ones), into `out` as one model.safetensors (each tensor in pieces of the ranges' size, by
    /// the threads). The tensors and bytes it holds.
    pub fn slice(&self, remote: &Remote, keep: &dyn Fn(&str, &[u64]) -> Option<Vec<u64>>, out: &Path, progress: &dyn Fn(u64, u64)) -> Res<(usize, u64)> {
        let files: Vec<&Entry> = remote.files.iter().filter(|e| e.path.ends_with(".safetensors") && !e.path.contains('/')).collect();
        // (file, name, dtype, shape, its bytes' place in the file, their length)
        let mut ts: Vec<(usize, String, String, Vec<u64>, u64, u64)> = Vec::new();
        for (fi, e) in files.iter().enumerate() {
            let (h, data) = self.header(remote, e)?;
            for (name, d, sh, a, b) in h {
                if let Some(kept) = keep(&name, &sh) {
                    // (the first rows: a row's bytes are the tensor's over its rows)
                    let rows = sh.first().copied().unwrap_or(1).max(1);
                    let len = if kept == sh { b - a } else { (b - a) / rows * kept.first().copied().unwrap_or(rows).min(rows) };
                    ts.push((fi, name, d, kept, data + a, len));
                }
            }
        }
        ts.sort_by(|x, y| x.1.cmp(&y.1));
        let esc = |s: &str| s.replace('\\', "\\\\").replace('"', "\\\"");
        let (mut at, mut parts) = (0u64, Vec::new());
        for (_, name, d, sh, _, len) in &ts {
            parts.push(format!(
                "\"{}\":{{\"dtype\":\"{d}\",\"shape\":[{}],\"data_offsets\":[{at},{}]}}",
                esc(name),
                sh.iter().map(u64::to_string).collect::<Vec<_>>().join(","),
                at + len
            ));
            at += len;
        }
        let mut h = format!("{{\"__metadata__\":{{\"format\":\"pt\"}},{}}}", parts.join(","));
        while (8 + h.len()) % 8 != 0 {
            h.push(' ');
        }
        let base = 8 + h.len() as u64;
        std::fs::create_dir_all(out).map_err(other)?;
        let path = out.join("model.safetensors");
        let file = std::fs::OpenOptions::new().read(true).write(true).create(true).truncate(true).open(&path).map_err(other)?;
        file.set_len(base + at).map_err(other)?;
        file.write_all_at(&(h.len() as u64).to_le_bytes(), 0).map_err(other)?;
        file.write_all_at(h.as_bytes(), 8).map_err(other)?;
        // the pieces: (tensor, its bytes' place in the source, their length, where in the file)
        let mut pieces = Vec::new();
        let mut dst = base;
        for (ti, t) in ts.iter().enumerate() {
            let mut o = 0u64;
            while o < t.5 {
                let n = (t.5 - o).min(self.range);
                pieces.push((ti, t.4 + o, n, dst + o));
                o += n;
            }
            dst += t.5;
        }
        let urls: Vec<Mutex<(String, std::time::Instant)>> =
            files.iter().map(|e| Ok(Mutex::new((self.resolve(remote, e)?, std::time::Instant::now())))).collect::<Res<_>>()?;
        let (next, done, failed) = (AtomicUsize::new(0), AtomicU64::new(0), Mutex::new(None::<HubError>));
        let work = || loop {
            let k = next.fetch_add(1, Ordering::SeqCst);
            if k >= pieces.len() || failed.lock().unwrap().is_some() {
                break;
            }
            let (ti, a, n, d) = pieces[k];
            let fi = ts[ti].0;
            if let Err(err) = self.range_to(remote, files[fi], &urls[fi], &file, a, n, d, &done) {
                failed.lock().unwrap().get_or_insert(err);
                break;
            }
        };
        ramped(self.threads, &|| pieces.len().saturating_sub(next.load(Ordering::Relaxed)), &done, &work, &|| progress(done.load(Ordering::Relaxed), at));
        if let Some(err) = failed.into_inner().unwrap() {
            return Err(err);
        }
        file.sync_all().map_err(other)?;
        Ok((ts.len(), at))
    }
}

/// `dlrate URL [--mb N] [--conns K] [--buf B]` (development, GLY-356): N MB (256) of a URL read by K connections (1, 8 apart) through the downloader's client,
/// as `range_to` reads (B KiB a read, 1,024), into nothing: each connection's MB/s and the total, against curl's on the same link.
#[cfg(feature = "dev")]
pub fn rate_cmd(a: &crate::args::Args) -> crate::ffi::Result<()> {
    let url = a.need(1, "URL")?.to_string();
    let (mb, buf) = (a.num("mb", 256u64)?, a.num("buf", 1024usize)? << 10);
    let conns: Vec<usize> = a
        .flag("conns")
        .unwrap_or("1,8")
        .split(',')
        .map(|x| x.trim().parse().map_err(|_| crate::ffi::Error(format!("--conns {x}: not a number"))))
        .collect::<crate::ffi::Result<_>>()?;
    let hub = Hub::from_env();
    for k in conns {
        let t0 = std::time::Instant::now();
        let each: Vec<(u64, f64)> = std::thread::scope(|s| {
            let hs: Vec<_> = (0..k as u64)
                .map(|i| {
                    let (hub, url) = (&hub, &url);
                    s.spawn(move || -> (u64, f64) {
                        let (a0, len) = (i * (mb << 20), mb << 20);
                        let t = std::time::Instant::now();
                        let Ok(mut resp) = hub.get(url, Some((a0, a0 + len - 1))) else { return (0, 0.0) };
                        let mut r = resp.body_mut().as_reader();
                        let (mut b, mut got) = (vec![0u8; buf], 0u64);
                        while got < len {
                            match r.read(&mut b) {
                                Ok(0) | Err(_) => break,
                                Ok(n) => got += n as u64,
                            }
                        }
                        (got, t.elapsed().as_secs_f64())
                    })
                })
                .collect();
            hs.into_iter().map(|h| h.join().unwrap_or((0, 0.0))).collect()
        });
        let total: u64 = each.iter().map(|x| x.0).sum();
        let per: Vec<String> = each.iter().map(|(g, s)| format!("{:.1}", *g as f64 / s.max(1e-9) / 1e6)).collect();
        println!(
            "{k} connections: {:.0} MB in {:.1} s, {:.1} MB/s in all; each {} MB/s",
            total as f64 / 1e6,
            t0.elapsed().as_secs_f64(),
            total as f64 / t0.elapsed().as_secs_f64() / 1e6,
            per.join(", ")
        );
    }
    Ok(())
}

/// `hubslice REPO OUT --layers N [--experts E] [--revision R] [--prefix P]`: a checkpoint's first N layers and the rest of its language model (embeddings, final norm; P: the language
/// model's names' prefix, "model.language_model." where the checkpoint has it, else "model."), and lm_head, fetched by ranges into OUT/model.safetensors, with config.json's layer
/// count (and its lists of one entry a layer) cut to N: a model too big for a machine's disk checked layer by layer on real weights. E: each MoE layer's first E experts alone, its
/// router's first E rows and config.json's expert count E (a layer's real widths and kernels on a disk that does not hold its experts; the choices are not the model's).
pub fn slice_cmd(a: &crate::args::Args) -> crate::ffi::Result<()> {
    let (repo, out) = (a.need(1, "REPO")?, Path::new(a.need(2, "OUT")?));
    let (n, rev) = (a.num("layers", 1usize)?, a.flag("revision").unwrap_or("main"));
    let hub = Hub::from_env();
    let remote = hub.list(repo, rev)?;
    let cfg = glyd_gpu::json::parse(&hub.text(repo, &remote.commit, "config.json")?).map_err(|e| crate::ffi::Error(format!("config.json: {e}")))?;
    let names = hub.text(repo, &remote.commit, "model.safetensors.index.json").unwrap_or_default();
    let prefix = a.flag("prefix").map(str::to_string).unwrap_or_else(|| {
        if names.contains("\"model.language_model.") {
            "model.language_model.".into()
        } else {
            "model.".into()
        }
    });
    let experts = a.flag("experts").map(|v| v.parse::<u64>().map_err(|_| crate::ffi::Error(format!("--experts {v}: a number")))).transpose()?;
    let take = |name: &str| -> bool {
        // (Kimi-K3's head is language_model.lm_head)
        if name == "lm_head.weight" || name == "language_model.lm_head.weight" {
            return true;
        }
        let Some(rest) = name.strip_prefix(prefix.as_str()) else { return false };
        match rest.strip_prefix("layers.").and_then(|r| r.split('.').next()).and_then(|l| l.parse::<usize>().ok()) {
            Some(l) => l < n,
            None => !rest.starts_with("layers."),
        }
    };
    let keep = |name: &str, shape: &[u64]| -> Option<Vec<u64>> {
        if !take(name) {
            return None;
        }
        let Some(e) = experts else { return Some(shape.to_vec()) };
        // (an expert past E left out; the router's rows and its bias past E too)
        if let Some(x) = name.split(".experts.").nth(1).and_then(|r| r.split('.').next()).and_then(|x| x.parse::<u64>().ok()) {
            return (x < e).then(|| shape.to_vec());
        }
        let router = name.ends_with(".gate.weight") || name.ends_with(".gate.e_score_correction_bias");
        Some(if router && shape.first().is_some_and(|&r| r > e) { std::iter::once(e).chain(shape[1..].iter().copied()).collect() } else { shape.to_vec() })
    };
    // config.json: num_hidden_layers N where it is said (its own and text_config's), each list as long as the layers were cut to N, and Kimi-K3's lists of 1-based layer numbers to those up to N
    fn cut(v: &mut Value, n: usize, layers: &mut Option<usize>) {
        if let Value::Object(kv) = v {
            if let Some(l) = kv.iter().find(|(k, _)| k == "num_hidden_layers").and_then(|(_, v)| v.as_u64()) {
                *layers = Some(l as usize);
            }
            for (k, x) in kv.iter_mut() {
                match x {
                    Value::Number(_) if k == "num_hidden_layers" => *x = Value::Number(n.to_string()),
                    Value::Array(xs) if layers.is_some_and(|l| xs.len() == l) => xs.truncate(n),
                    Value::Array(xs) if k == "kda_layers" || k == "full_attn_layers" => xs.retain(|v| v.as_u64().is_some_and(|l| l as usize <= n)),
                    Value::Object(_) => cut(x, n, &mut None),
                    _ => {}
                }
            }
        }
    }
    let mut cfg = cfg;
    cut(&mut cfg, n, &mut None);
    if let Some(e) = experts {
        // (the expert count, wherever the config says it: its own or text_config's)
        fn set(v: &mut Value, e: u64) {
            if let Value::Object(kv) = v {
                for (k, x) in kv.iter_mut() {
                    match x {
                        Value::Number(_) if ["num_experts", "n_routed_experts", "num_local_experts"].contains(&k.as_str()) => *x = Value::Number(e.to_string()),
                        Value::Object(_) => set(x, e),
                        _ => {}
                    }
                }
            }
        }
        set(&mut cfg, e);
    }
    std::fs::create_dir_all(out).map_err(|e| crate::ffi::Error(e.to_string()))?;
    std::fs::write(out.join("config.json"), glyd_gpu::json::to_python(&cfg, 2)).map_err(|e| crate::ffi::Error(e.to_string()))?;
    let t0 = std::time::Instant::now();
    let (count, bytes) = hub.slice(&remote, &keep, out, &|d, t| eprint!("\r{:.1} of {:.1} GB", d as f64 / 1e9, t as f64 / 1e9))?;
    eprintln!();
    println!(
        "{repo}@{}: {count} tensors of its first {n} layers and the rest ({prefix}*, lm_head), {:.2} GB in {:.0} s, into {}",
        remote.commit,
        bytes as f64 / 1e9,
        t0.elapsed().as_secs_f64(),
        out.display()
    );
    Ok(())
}

/// The files of a repo this engine reads: the checkpoint's safetensors at the top (the other copies in folders, and Mistral's consolidated one where both are there), and the small files
/// at the top (config, index, tokenizer, chat template, generation config).
pub fn wanted(remote: &Remote) -> Vec<Entry> {
    let st: Vec<&Entry> = remote.files.iter().filter(|e| e.path.ends_with(".safetensors") && !e.path.contains('/')).collect();
    let all_consolidated = st.iter().all(|e| e.path.starts_with("consolidated"));
    remote
        .files
        .iter()
        .filter(|e| {
            let top = !e.path.contains('/');
            let weight = e.path.ends_with(".safetensors") && top && (all_consolidated || !e.path.starts_with("consolidated"));
            let small = top
                && !e.path.ends_with(".safetensors")
                && (e.path.ends_with(".json")
                    || e.path.ends_with(".txt")
                    || e.path.ends_with(".model")
                    || e.path.ends_with(".jinja")
                    || e.path.starts_with("tokenizer"));
            // (a draft model a pre-packed repo carries in draft/: Gemma 4's "-assistant", which the engine drafts with)
            let draft = e.path.starts_with("draft/") && !e.path[6..].contains('/');
            weight || small || draft
        })
        .cloned()
        .collect()
}

/// `work` (a worker: it takes pieces until none are left) on `start` threads, and more while they make the download faster: every 3 s, 8 more where the last
/// ones raised the rate by a fifth or more, up to 64 (GLY-356: a home link is full at 8 connections, a datacenter's gives each a few MB/s); `left` the pieces
/// not yet taken, `done` the bytes, `tick` the progress (every 200 ms) until every worker is done.
fn ramped(start: usize, left: &(dyn Fn() -> usize + Sync), done: &AtomicU64, work: &(dyn Fn() + Sync), tick: &dyn Fn()) {
    const MOST: usize = 64;
    let running = AtomicUsize::new(0);
    std::thread::scope(|s| {
        let spawn = |n: usize| {
            for _ in 0..n {
                running.fetch_add(1, Ordering::SeqCst);
                s.spawn(|| {
                    work();
                    running.fetch_sub(1, Ordering::SeqCst);
                });
            }
        };
        let mut workers = start.min(left()).max(1);
        spawn(workers);
        let (mut t0, mut d0, mut rate0, mut growing) = (std::time::Instant::now(), done.load(Ordering::Relaxed), 0f64, true);
        while running.load(Ordering::SeqCst) > 0 {
            tick();
            std::thread::sleep(Duration::from_millis(200));
            if growing && t0.elapsed() >= Duration::from_secs(3) {
                let d = done.load(Ordering::Relaxed);
                let rate = d.saturating_sub(d0) as f64 / t0.elapsed().as_secs_f64();
                if rate0 > 0.0 && rate < rate0 * 1.2 {
                    growing = false;
                } else if workers < MOST && left() > workers {
                    spawn(8);
                    workers += 8;
                }
                (t0, d0, rate0) = (std::time::Instant::now(), d, rate);
            }
        }
    });
}

pub fn sha256_of(path: &Path) -> std::io::Result<String> {
    let mut f = std::fs::File::open(path)?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 8 << 20];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(h.finalize().iter().map(|b| format!("{b:02x}")).collect())
}

/// Writes the token where `hf login` does (mode 0600), for the next run to find.
pub fn save_token(token: &str) -> std::io::Result<PathBuf> {
    use std::io::Write;
    #[cfg(unix)]
    use std::os::unix::fs::OpenOptionsExt;
    let path = match std::env::var_os("HF_TOKEN_PATH").filter(|v| !v.is_empty()) {
        Some(p) => PathBuf::from(p),
        None => hub_cache().parent().map_or_else(|| PathBuf::from("token"), |p| p.join("token")),
    };
    if let Some(d) = path.parent() {
        std::fs::create_dir_all(d)?;
    }
    // (Windows: a file under the user's profile is the user's alone by its folder's rights)
    #[cfg(unix)]
    let mut f = std::fs::OpenOptions::new().write(true).create(true).truncate(true).mode(0o600).open(&path)?;
    #[cfg(windows)]
    let mut f = std::fs::OpenOptions::new().write(true).create(true).truncate(true).open(&path)?;
    f.write_all(token.trim().as_bytes())?;
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The token as the Hub's own tools take it, and the flag `--hf-token` first of all: the flag, HF_TOKEN, HUGGING_FACE_HUB_TOKEN, the saved login; a blank one is none.
    #[test]
    fn the_token_comes_from_the_flag_the_environment_or_the_saved_login() {
        let env = |pairs: &'static [(&'static str, &'static str)]| move |k: &str| pairs.iter().find(|(n, _)| *n == k).map(|(_, v)| v.to_string());
        let all = env(&[("HF_TOKEN", "e1"), ("HUGGING_FACE_HUB_TOKEN", "e2")]);
        assert_eq!(pick_token(Some("flag".into()), &all, Some("file".into())), Some("flag".into()));
        assert_eq!(pick_token(None, &all, Some("file".into())), Some("e1".into()));
        assert_eq!(pick_token(None, &env(&[("HUGGING_FACE_HUB_TOKEN", " e2 ")]), Some("file".into())), Some("e2".into()));
        assert_eq!(pick_token(None, &env(&[]), Some("file\n".into())), Some("file".into()));
        assert_eq!(pick_token(None, &env(&[]), None), None);
        assert_eq!(pick_token(Some("  ".into()), &env(&[("HF_TOKEN", "")]), Some("".into())), None, "blank ones are none");
        assert_eq!(pick_token(Some("".into()), &all, None), Some("e1".into()), "an empty flag is not a token: the environment's is");
    }
    use std::io::{BufRead, BufReader, Write};
    use std::net::{TcpListener, TcpStream};
    use std::sync::Arc;

    /// The Hub in miniature, on 127.0.0.1: the listing, `resolve` that redirects to a second host name of the same server ("storage", localhost) which serves ranges. Every request is noted.
    type Log = Arc<Mutex<Vec<(String, Option<String>, bool)>>>;
    struct Mock {
        port: u16,
        log: Log,
        files: Arc<Vec<(String, Vec<u8>)>>,
        corrupt: Arc<Mutex<bool>>,
    }

    fn bytes_of(seed: u8, n: usize) -> Vec<u8> {
        (0..n).map(|i| (i as u32).wrapping_mul(2654435761).wrapping_add(seed as u32 * 97).to_le_bytes()[2]).collect()
    }

    fn sha(b: &[u8]) -> String {
        Sha256::digest(b).iter().map(|x| format!("{x:02x}")).collect()
    }

    fn serve(mock: &Mock, mut s: TcpStream) {
        let mut r = BufReader::new(s.try_clone().unwrap());
        let mut line = String::new();
        if r.read_line(&mut line).unwrap_or(0) == 0 {
            return;
        }
        let path = line.split(' ').nth(1).unwrap_or("").to_string();
        let (mut range, mut auth) = (None, false);
        loop {
            let mut h = String::new();
            r.read_line(&mut h).unwrap();
            if h.trim().is_empty() {
                break;
            }
            let low = h.to_ascii_lowercase();
            if let Some(v) = low.strip_prefix("range:") {
                range = Some(v.trim().to_string());
            }
            if low.starts_with("authorization:") {
                auth = true;
            }
        }
        mock.log.lock().unwrap().push((path.clone(), range.clone(), auth));
        let reply = |s: &mut TcpStream, code: &str, extra: &str, body: &[u8]| {
            let _ = write!(s, "HTTP/1.1 {code}\r\n{extra}Content-Length: {}\r\nConnection: close\r\n\r\n", body.len());
            let _ = s.write_all(body);
        };
        let find = |name: &str| mock.files.iter().find(|(n, _)| n == name).map(|(_, b)| b.clone());
        if path.starts_with("/api/models/org/model/revision/main") {
            let sibs: Vec<String> = mock
                .files
                .iter()
                .map(|(n, b)| {
                    if b.len() > 4000 {
                        format!("{{\"rfilename\":\"{n}\",\"size\":{},\"blobId\":\"ptr{n}\",\"lfs\":{{\"sha256\":\"{}\",\"size\":{}}}}}", 133, sha(b), b.len())
                    } else {
                        format!("{{\"rfilename\":\"{n}\",\"size\":{},\"blobId\":\"git{}\"}}", b.len(), n.replace('/', "_"))
                    }
                })
                .collect();
            let body = format!("{{\"sha\":\"{}\",\"siblings\":[{}],\"safetensors\":{{\"parameters\":{{\"BF16\":1000}}}}}}", "c".repeat(40), sibs.join(","));
            reply(&mut s, "200 OK", "Content-Type: application/json\r\n", body.as_bytes());
        } else if let Some(name) = path.strip_prefix(&format!("/org/model/resolve/{}/", "c".repeat(40))) {
            // the storage is another host name: localhost, not 127.0.0.1
            reply(&mut s, "302 Found", &format!("Location: http://localhost:{}/store/{name}\r\n", mock.port), b"");
        } else if let Some(name) = path.strip_prefix("/store/") {
            let Some(mut b) = find(name) else { return reply(&mut s, "404 Not Found", "", b"") };
            if auth {
                return reply(&mut s, "400 Bad Request", "", b"the storage got the token");
            }
            if *mock.corrupt.lock().unwrap() && name == "model.safetensors" {
                b[10] ^= 1;
            }
            match range.as_deref().and_then(|r| r.strip_prefix("bytes=")).and_then(|r| r.split_once('-')) {
                Some((a, e)) => {
                    let (a, e): (usize, usize) = (a.parse().unwrap(), e.parse::<usize>().unwrap().min(b.len() - 1));
                    reply(&mut s, "206 Partial Content", &format!("Content-Range: bytes {a}-{e}/{}\r\n", b.len()), &b[a..=e]);
                }
                None => reply(&mut s, "200 OK", "", &b),
            }
        } else {
            reply(&mut s, "404 Not Found", "", b"");
        }
    }

    fn mock() -> (Mock, Hub, PathBuf) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let files = vec![
            ("config.json".to_string(), b"{\"a\": 1}".to_vec()),
            ("model.safetensors".to_string(), bytes_of(1, 10_000)),
            ("sub/tokenizer.json".to_string(), bytes_of(2, 900)),
        ];
        let m = Mock { port, log: Arc::new(Mutex::new(Vec::new())), files: Arc::new(files), corrupt: Arc::new(Mutex::new(false)) };
        let (log, files, corrupt) = (m.log.clone(), m.files.clone(), m.corrupt.clone());
        std::thread::spawn(move || {
            for s in listener.incoming().flatten() {
                let mm = Mock { port, log: log.clone(), files: files.clone(), corrupt: corrupt.clone() };
                std::thread::spawn(move || serve(&mm, s));
            }
        });
        let dir = std::env::temp_dir().join(format!("glyd-engine-hub-{}-{port}", std::process::id()));
        let hub = Hub::new(&format!("http://127.0.0.1:{port}"), Some("secret".to_string()), dir.clone(), 1024, 4);
        (m, hub, dir)
    }

    #[test]
    fn a_repo_listed_downloaded_resumed_and_checked() {
        let (m, hub, dir) = mock();
        let remote = hub.list("org/model", "main").unwrap();
        assert_eq!(remote.commit, "c".repeat(40));
        assert_eq!(remote.dtypes, vec![("BF16".to_string(), 1000)]);
        let big = remote.files.iter().find(|e| e.path == "model.safetensors").unwrap();
        assert_eq!((big.size, big.blob.len(), big.sha256.is_some()), (10_000, 64, true));
        let want = wanted(&remote);
        assert_eq!(want.iter().map(|e| e.path.as_str()).collect::<Vec<_>>(), ["config.json", "model.safetensors"], "a file in a folder is not read");
        assert_eq!(hub.missing(&remote, &want), 8 + 10_000);
        let seen = Mutex::new(Vec::new());
        let snap = hub.fetch(&remote, &want, &|d, t| seen.lock().unwrap().push((d, t))).unwrap();
        assert_eq!(std::fs::read(snap.join("model.safetensors")).unwrap(), m.files[1].1);
        assert_eq!(std::fs::read(snap.join("config.json")).unwrap(), b"{\"a\": 1}");
        assert!(std::fs::symlink_metadata(snap.join("config.json")).unwrap().file_type().is_symlink());
        assert_eq!(seen.lock().unwrap().last().copied(), Some((10_008, 10_008)));
        assert_eq!(hub.missing(&remote, &want), 0);
        assert!(!dir.join("models--org--model").join("refs").join("main").exists(), "the name points at the commit only once pinned");
        hub.pin(&remote).unwrap();
        assert_eq!(std::fs::read_to_string(dir.join("models--org--model").join("refs").join("main")).unwrap(), "c".repeat(40));
        {
            let log = m.log.lock().unwrap();
            // the token went to the Hub's host and not to the storage; the file came as ranges
            assert!(log.iter().filter(|(p, _, _)| p.starts_with("/store/")).all(|(_, _, auth)| !auth));
            assert!(log.iter().filter(|(p, _, _)| p.starts_with("/api/") || p.contains("/resolve/")).all(|(_, _, auth)| *auth));
            assert_eq!(
                log.iter()
                    .filter(|(p, r, _)| p == "/store/model.safetensors"
                        && r.as_ref().is_some_and(|r| r.starts_with("bytes="))
                        && r.as_deref() != Some("bytes=0-0"))
                    .count(),
                10
            );
        }
        // a second fetch asks for nothing
        let n = m.log.lock().unwrap().len();
        hub.fetch(&remote, &want, &|_, _| {}).unwrap();
        assert_eq!(m.log.lock().unwrap().len(), n);
        std::fs::remove_dir_all(&dir).unwrap();
        // a download that stopped: 4 of the 10 ranges there, the others asked for
        let blobs = dir.join("models--org--model").join("blobs");
        std::fs::create_dir_all(&blobs).unwrap();
        let part = blobs.join(format!("{}.incomplete", big.blob));
        let mut data = vec![0u8; 10_000];
        let mut map = Vec::new();
        for i in [0usize, 1, 2, 3] {
            data[i * 1024..(i + 1) * 1024].copy_from_slice(&m.files[1].1[i * 1024..(i + 1) * 1024]);
            map.extend((i as u32).to_le_bytes());
        }
        std::fs::write(&part, &data).unwrap();
        std::fs::write(blobs.join(format!("{}.ranges", big.blob)), &map).unwrap();
        m.log.lock().unwrap().clear();
        let snap = hub.fetch(&remote, std::slice::from_ref(big), &|_, _| {}).unwrap();
        assert_eq!(std::fs::read(snap.join("model.safetensors")).unwrap(), m.files[1].1);
        let asked: Vec<String> = m
            .log
            .lock()
            .unwrap()
            .iter()
            .filter(|(p, _, _)| p == "/store/model.safetensors")
            .filter_map(|(_, r, _)| r.clone())
            .filter(|r| r != "bytes=0-0")
            .collect();
        assert_eq!(asked.len(), 6, "{asked:?}");
        assert!(
            asked.iter().all(|r| !r.starts_with("bytes=0-") && !r.starts_with("bytes=1024-") && !r.starts_with("bytes=2048-") && !r.starts_with("bytes=3072-")),
            "{asked:?}"
        );
        std::fs::remove_dir_all(&dir).unwrap();
        // a file that does not match its checksum is not kept
        *m.corrupt.lock().unwrap() = true;
        let e = hub.fetch(&remote, std::slice::from_ref(big), &|_, _| {}).unwrap_err();
        assert!(e.to_string().contains("checksum"), "{e}");
        assert!(!dir.join("models--org--model").join("blobs").join(&big.blob).exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A connection over the system's route that fails (here an agent that takes IPv6 addresses alone, which the mock's IPv4 ones are not: it reaches nothing, as an IPv6 path that resets or hangs
    /// does): the request is made again over IPv4, the download comes through whole, and IPv4 is the way from then on.
    #[test]
    fn a_failed_ipv6_request_is_made_again_over_ipv4_and_ipv4_stays() {
        let (m, mut hub, dir) = mock();
        hub.agent = ureq::Agent::new_with_config(
            ureq::Agent::config_builder().ip_family(ureq::config::IpFamily::Ipv6Only).http_status_as_error(false).max_redirects(0).build(),
        );
        assert!(!hub.v4.load(Ordering::Relaxed));
        let remote = hub.list("org/model", "main").unwrap();
        assert!(hub.v4.load(Ordering::Relaxed), "a request that failed over the system's route and worked over IPv4 makes IPv4 the way");
        let want = wanted(&remote);
        let snap = hub.fetch(&remote, &want, &|_, _| {}).unwrap();
        assert_eq!(std::fs::read(snap.join("model.safetensors")).unwrap(), m.files[1].1);
        assert_eq!(std::fs::read(snap.join("config.json")).unwrap(), b"{\"a\": 1}");
        // a Hub that is not told IPv4 and finds both ways closed has the first way's error, and no way chosen
        let dead = Hub::new("http://127.0.0.1:1", None, dir.clone(), 1024, 2);
        assert!(matches!(dead.list("org/model", "main"), Err(HubError::Network(_))));
        assert!(!dead.v4.load(Ordering::Relaxed));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A Hub told to use IPv4 (GLYD_IPV4=1) never goes by IPv6: a server that is reachable at an IPv6 address alone (one that answers 404 to everything) is answered by the Hub that is not told, and is not reached by the
    /// Hub that is. (A computer without an IPv6 loopback has nothing to test.)
    #[test]
    fn a_hub_told_to_use_ipv4_does_not_try_ipv6() {
        let Ok(listener) = TcpListener::bind("[::1]:0") else { return };
        let port = listener.local_addr().unwrap().port();
        std::thread::spawn(move || {
            for mut s in listener.incoming().flatten() {
                let mut r = BufReader::new(s.try_clone().unwrap());
                let mut h = String::new();
                while r.read_line(&mut h).is_ok_and(|n| n > 2) {
                    h.clear();
                }
                let _ = s.write_all(b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n");
            }
        });
        let dir = std::env::temp_dir().join(format!("glyd-engine-hub-v6-{}-{port}", std::process::id()));
        let endpoint = format!("http://[::1]:{port}");
        let open = Hub::new(&endpoint, None, dir.clone(), 1024, 2);
        assert!(matches!(open.list("org/model", "main"), Err(HubError::Denied(404))));
        assert!(!open.v4.load(Ordering::Relaxed), "an answer, even a 404, is no failure");
        let forced = Hub::new(&endpoint, None, dir, 1024, 2);
        forced.v4.store(true, Ordering::Relaxed);
        assert!(matches!(forced.list("org/model", "main"), Err(HubError::Network(_))));
    }

    #[test]
    fn a_repo_that_is_not_there() {
        let (_m, hub, dir) = mock();
        assert!(matches!(hub.list("org/other", "main"), Err(HubError::Denied(404))));
        let _ = std::fs::remove_dir_all(&dir);
        assert!(matches!(Hub::new("http://127.0.0.1:1", None, dir.clone(), 1024, 2).list("org/model", "main"), Err(HubError::Network(_))));
    }

    #[test]
    fn urls_and_the_cache() {
        assert_eq!(host_of("https://cas.example.com/a/b?c=d"), "cas.example.com");
        assert_eq!(join("https://huggingface.co/org/m/resolve/x/f", "/api/resolve-cache/x"), "https://huggingface.co/api/resolve-cache/x");
        assert_eq!(join("https://huggingface.co/org/m/resolve/x/f", "https://cdn.example.com/y?z=1"), "https://cdn.example.com/y?z=1");
        assert_eq!(join("https://h.co/a/b/c", "d"), "https://h.co/a/b/d");
        if cfg!(target_os = "linux") {
            assert!(disk_free(&std::env::temp_dir()).unwrap() > 1 << 20);
        }
    }
}
