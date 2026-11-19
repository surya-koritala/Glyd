//! Explicitly approved local service processes. Reachability is not inference qualification.
use process_wrap::std::{ChildWrapper, CommandWrap};
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};
use starlings_protocol::{ChatRequest, Digest, FixtureMode, content_hash, encode, parse_json};
use std::{
    collections::BTreeMap,
    fs::File,
    io::{BufRead, BufReader, Read},
    net::SocketAddr,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};

type Result<T> = std::result::Result<T, Error>;
mod glyd;
mod glyd_memory;
pub use glyd::{
    GlydBuild, GlydIdentity, GlydLevel, GlydModel, GlydPrecision, GlydSource, RequestCancellation,
};
pub use glyd_memory::GlydMemory;
use glyd_memory::GlydMemorySample;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    Invalid,
    Unapproved,
    Unavailable,
    Transport,
    Capacity,
    Cancelled,
    StopUnconfirmed,
}
impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "backend: {self:?}")
    }
}
impl std::error::Error for Error {}

/// Approved wire dialect; selecting one confers no backend or native qualification.
#[derive(Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum ServiceApi {
    #[default]
    #[serde(rename = "input-tokens/1")]
    InputTokensV1,
    #[serde(rename = "glyd-serve/1")]
    GlydServeV1,
}
impl ServiceApi {
    fn is_default(&self) -> bool {
        *self == Self::InputTokensV1
    }
    fn count_path(self) -> &'static str {
        match self {
            Self::InputTokensV1 => "/v1/chat/completions/input_tokens",
            Self::GlydServeV1 => "/v1/glyd/tokenize",
        }
    }
}
/// Operator-owned configuration, approved by its full digest. Never accept this from a chat request.
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ServiceConfig {
    pub executable: PathBuf,
    pub executable_sha256: Digest,
    pub arguments: Vec<String>,
    pub directory: PathBuf,
    pub environment: BTreeMap<String, String>,
    pub key_environment: String,
    pub endpoint: SocketAddr,
    pub backend_version: String,
    #[serde(default, skip_serializing_if = "ServiceApi::is_default")]
    pub api: ServiceApi,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub glyd_identity: Option<GlydIdentity>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub properties: Option<ExpectedProperties>,
    pub manifest_digest: Digest,
    pub execution_profile: String,
    pub backend_model: String,
    pub reasoning_content: bool,
    pub request_timeout_ms: u32,
}
/// Optional installed-service check. Approval binds these expectations, not an attestation.
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExpectedProperties {
    pub build_info: String,
    pub context_tokens: u32,
}
/// Narrow reported metadata; paths, templates and unknown properties are never retained.
#[derive(Clone, Debug, Serialize)]
pub struct Properties {
    pub build_info: String,
    pub context_tokens: u32,
    pub system_role: bool,
    pub string_content: bool,
    pub typed_content: bool,
}
impl ServiceConfig {
    pub fn digest(&self) -> Result<Digest> {
        content_hash(self).map_err(|_| Error::Invalid)
    }
    /// Check configuration and executable identity without starting a process.
    pub fn validate(&self, approved: &Digest) -> Result<()> {
        self.validate_observed(approved, |_| {})
    }
    /// Static progress only; the observer must not block or reenter validation.
    pub fn validate_observed(
        &self,
        approved: &Digest,
        mut phase: impl FnMut(&'static str),
    ) -> Result<()> {
        self.validate_settings_observed(approved, &mut phase)?;
        phase("executable_digest");
        if executable_digest(&self.executable)? != self.executable_sha256 {
            return Err(Error::Unapproved);
        }
        Ok(())
    }
    /// Check approved settings and paths, without reading executable bytes.
    /// Service launchers must still perform full validation immediately before spawn.
    pub fn validate_settings_observed(
        &self,
        approved: &Digest,
        mut phase: impl FnMut(&'static str),
    ) -> Result<()> {
        phase("configuration_digest");
        if &self.digest()? != approved {
            return Err(Error::Unapproved);
        }
        phase("configuration_paths");
        if !self.endpoint.ip().is_loopback()
            || self.endpoint.port() == 0
            || !self.executable.is_absolute()
            || !self.directory.is_absolute()
            || self.executable.canonicalize().ok().as_ref() != Some(&self.executable)
            || self.directory.canonicalize().ok().as_ref() != Some(&self.directory)
            || !self.directory.is_dir()
            || !self.executable.is_file()
            || self.arguments.len() > 128
            || !(1000..=600_000).contains(&self.request_timeout_ms)
            || self.environment.len() > 32
            || !valid_name(&self.key_environment)
            || self.environment.contains_key(&self.key_environment)
            || self.environment.keys().any(|key| !valid_name(key))
            || match self.api {
                ServiceApi::InputTokensV1 => self.glyd_identity.is_some(),
                ServiceApi::GlydServeV1 => {
                    self.properties.is_some()
                        || self
                            .glyd_identity
                            .as_ref()
                            .is_none_or(|i| !i.valid_for(self))
                }
            }
            || self.properties.as_ref().is_some_and(|p| {
                p.build_info.is_empty()
                    || p.build_info.len() > 256
                    || !p.build_info.bytes().all(|c| c.is_ascii_graphic())
                    || !(1..=2_000_000).contains(&p.context_tokens)
            })
            || [
                &self.backend_version,
                &self.backend_model,
                &self.execution_profile,
            ]
            .iter()
            .any(|v| v.is_empty() || v.len() > 256 || v.contains('\0'))
            || serde_json::to_vec(self).map_err(|_| Error::Invalid)?.len() > 64 * 1024
        {
            return Err(Error::Invalid);
        }
        Ok(())
    }
}
fn valid_name(key: &str) -> bool {
    !key.is_empty()
        && key.len() <= 128
        && key
            .bytes()
            .all(|c| c.is_ascii_uppercase() || c == b'_' || c.is_ascii_digit())
}
pub fn executable_digest(path: &Path) -> Result<Digest> {
    file_digest(&mut File::open(path).map_err(|_| Error::Unavailable)?)
}
/// The bytes of an already open file, from its current offset.
pub(crate) fn file_digest(file: &mut File) -> Result<Digest> {
    if file.metadata().map_err(|_| Error::Unavailable)?.len() > 512 * 1024 * 1024 {
        return Err(Error::Capacity);
    }
    let mut hash = Sha256::new();
    let mut buf = [0; 64 * 1024];
    let mut total = 0;
    loop {
        let n = file.read(&mut buf).map_err(|_| Error::Unavailable)?;
        if n == 0 {
            break;
        }
        total += n;
        if total > 512 * 1024 * 1024 {
            return Err(Error::Capacity);
        }
        hash.update(&buf[..n]);
    }
    hash.finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect::<String>()
        .try_into()
        .map_err(|_| Error::Invalid)
}

struct Process {
    child: Box<dyn ChildWrapper>,
    stopped: bool,
    #[cfg(unix)]
    signal_allowed: bool,
}
impl Process {
    fn signal(&mut self) -> Result<()> {
        if self.stopped {
            return Ok(());
        }
        #[cfg(unix)]
        if !self.signal_allowed {
            return Err(Error::StopUnconfirmed);
        }
        self.child.start_kill().map_err(|_| Error::StopUnconfirmed)
    }
    fn reap(&mut self) -> Result<bool> {
        // Never signal a POSIX group ID after reaping may have made it reusable.
        #[cfg(unix)]
        {
            self.signal_allowed = false;
        }
        self.stopped |= match self.child.try_wait() {
            Ok(status) => status.is_some(),
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => false,
            Err(_error) => {
                #[cfg(test)]
                eprintln!("owned child wait failed: {_error:?}");
                return Err(Error::StopUnconfirmed);
            }
        };
        Ok(self.stopped)
    }
    #[cfg(unix)]
    fn pid(&self) -> Result<rustix::process::Pid> {
        rustix::process::Pid::from_raw(
            i32::try_from(self.child.id()).map_err(|_| Error::StopUnconfirmed)?,
        )
        .ok_or(Error::StopUnconfirmed)
    }
}
#[derive(Clone)]
pub struct Cancellation {
    process: Arc<Mutex<Process>>,
    cancelled: Arc<AtomicBool>,
}
impl Cancellation {
    /// Withdraw output before signalling the owned process group/job. No PID supplied by callers.
    pub fn cancel(&self) -> Result<()> {
        self.cancelled.store(true, Ordering::Release);
        let mut process = self.process.lock().map_err(|_| Error::StopUnconfirmed)?;
        process.signal()
    }
}

pub struct Service {
    config: ServiceConfig,
    key: String,
    agent: ureq::Agent,
    control: Cancellation,
    ready: bool,
    counted: Option<(Digest, u32)>,
    properties: Option<Properties>,
    glyd_identity: Option<GlydIdentity>,
    glyd_memory: Option<GlydMemorySample>,
    request: Option<RequestCancellation>,
}
impl Service {
    pub fn configuration(&self) -> &ServiceConfig {
        &self.config
    }
    pub fn properties(&self) -> Option<&Properties> {
        self.properties.as_ref().filter(|_| self.ready)
    }
    pub fn glyd_identity(&self) -> Option<&GlydIdentity> {
        self.glyd_identity.as_ref().filter(|_| self.ready)
    }
    /// Fresh self-reported allocator sample. Ordinals are not physical pool IDs;
    /// excluded overhead and retained KV/state prevent inferring allocatable capacity or drain.
    pub fn glyd_memory(&self) -> Option<&GlydMemory> {
        self.glyd_memory
            .as_ref()
            .filter(|_| self.ready)?
            .fresh(Instant::now())
    }
    /// Available after counting; bound to that request, not whichever request runs later.
    pub fn request_cancellation(&self) -> Option<RequestCancellation> {
        self.request.clone()
    }
    /// Starts only the exact approved configuration. No shell, PATH lookup, download or fallback.
    pub fn spawn(config: ServiceConfig, approved: &Digest, key: String) -> Result<Self> {
        Self::spawn_observed(config, approved, key, |_| {})
    }
    /// Static progress only; the observer must not block or reenter the service.
    pub fn spawn_observed(
        config: ServiceConfig,
        approved: &Digest,
        key: String,
        mut phase: impl FnMut(&'static str),
    ) -> Result<Self> {
        config.validate_observed(approved, &mut phase)?;
        phase("key_validation");
        if key.len() < 32 || key.len() > 128 || !key.bytes().all(|b| b.is_ascii_alphanumeric()) {
            return Err(Error::Invalid);
        }
        let mut command = Command::new(&config.executable);
        command
            .args(&config.arguments)
            .current_dir(&config.directory)
            .env_clear()
            .envs(&config.environment)
            .env(&config.key_environment, &key)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null());
        let mut command = CommandWrap::from(command);
        #[cfg(unix)]
        command.wrap(process_wrap::std::ProcessGroup::leader());
        #[cfg(windows)]
        command.wrap(process_wrap::std::JobObject);
        phase("process_spawn");
        let child = command.spawn().map_err(|_| Error::Unavailable)?;
        let request_timeout = Duration::from_millis(u64::from(config.request_timeout_ms));
        Ok(Self {
            config,
            key,
            agent: ureq::Agent::config_builder()
                .proxy(None)
                .max_redirects(0)
                .http_status_as_error(false)
                .max_response_header_size(16 * 1024)
                .timeout_global(Some(request_timeout))
                .build()
                .into(),
            control: Cancellation {
                process: Arc::new(Mutex::new(Process {
                    child,
                    stopped: false,
                    #[cfg(unix)]
                    signal_allowed: true,
                })),
                cancelled: Arc::new(AtomicBool::new(false)),
            },
            ready: false,
            counted: None,
            properties: None,
            glyd_identity: None,
            glyd_memory: None,
            request: None,
        })
    }
    pub fn cancellation(&self) -> Cancellation {
        self.control.clone()
    }
    /// Check the owned parent without probing the HTTP endpoint.
    pub fn check_alive(&self) -> Result<()> {
        if self.control.cancelled.load(Ordering::Acquire) {
            return Err(Error::Cancelled);
        }
        let mut process = self
            .control
            .process
            .lock()
            .map_err(|_| Error::Unavailable)?;
        if process.stopped {
            return Err(Error::Unavailable);
        }
        #[cfg(unix)]
        {
            use rustix::process::{WaitId, WaitIdOptions, waitid};
            // Observe exit without reaping: the owned leader's PID cannot be reused
            // before its remaining group members have been signalled.
            match waitid(
                WaitId::Pid(process.pid()?),
                WaitIdOptions::EXITED | WaitIdOptions::NOHANG | WaitIdOptions::NOWAIT,
            ) {
                Ok(None) => Ok(()),
                Ok(Some(_)) => {
                    let _ = process.signal();
                    let _ = process.reap();
                    Err(Error::Unavailable)
                }
                Err(rustix::io::Errno::INTR) => Err(Error::Unavailable),
                Err(_) => {
                    process.signal_allowed = false;
                    Err(Error::StopUnconfirmed)
                }
            }
        }
        #[cfg(not(unix))]
        if process
            .child
            .try_wait()
            .map_err(|_| Error::Unavailable)?
            .is_some()
        {
            let _ = process.child.start_kill();
            process.stopped = true;
            return Err(Error::Unavailable);
        }
        #[cfg(not(unix))]
        Ok(())
    }
    /// One bounded readiness attempt; caller may retry explicitly while loading.
    /// A matching API model is reachability evidence only, not model/profile attestation.
    pub fn probe(&mut self) -> Result<()> {
        self.probe_observed(|_| {})
    }
    /// Static progress only. The observer must be nonblocking and must not reenter this service.
    pub fn probe_observed(&mut self, mut phase: impl FnMut(&'static str)) -> Result<()> {
        let deadline = Instant::now() + Duration::from_secs(2);
        let remaining = || {
            deadline
                .checked_duration_since(Instant::now())
                .filter(|duration| !duration.is_zero())
                .ok_or(Error::Unavailable)
        };
        self.ready = false;
        self.counted = None;
        self.properties = None;
        self.glyd_identity = None;
        phase("process_alive");
        self.check_alive()?;
        phase("model_auth_refusal");
        let refused = self
            .agent
            .get(format!("http://{}/v1/models", self.config.endpoint))
            .config()
            .timeout_global(Some(remaining()?))
            .build()
            .call()
            .map_err(|_| Error::Transport)?;
        match refused.status().as_u16() {
            503 => return Err(Error::Unavailable),
            401 | 403 => (),
            _ => return Err(Error::Unapproved),
        }
        phase("health");
        let health = self
            .agent
            .get(format!("http://{}/health", self.config.endpoint))
            .header("Authorization", format!("Bearer {}", self.key))
            .config()
            .timeout_global(Some(remaining()?))
            .build()
            .call()
            .map_err(|_| Error::Transport)?;
        if health.status().as_u16() != 200 {
            return Err(Error::Unavailable);
        }
        phase("model_alias");
        let mut response = self
            .agent
            .get(format!("http://{}/v1/models", self.config.endpoint))
            .header("Authorization", format!("Bearer {}", self.key))
            .config()
            .timeout_global(Some(remaining()?))
            .build()
            .call()
            .map_err(|_| Error::Transport)?;
        check_response(&response, "application/json")?;
        let value = read_json(response.body_mut().as_reader(), 64 * 1024)?;
        let models = value["data"].as_array().ok_or(Error::Invalid)?;
        if models.len() != 1 || models[0]["id"].as_str() != Some(self.config.backend_model.as_str())
        {
            return Err(Error::Unapproved);
        }
        if self.config.properties.is_some() || self.config.glyd_identity.is_some() {
            let path = if self.config.api == ServiceApi::GlydServeV1 {
                "/v1/glyd/info"
            } else {
                "/props"
            };
            phase("properties_auth_refusal");
            let refused = self
                .agent
                .get(format!("http://{}{path}", self.config.endpoint))
                .config()
                .timeout_global(Some(remaining()?))
                .build()
                .call()
                .map_err(|_| Error::Transport)?;
            match refused.status().as_u16() {
                503 => return Err(Error::Unavailable),
                401 | 403 => (),
                _ => return Err(Error::Unapproved),
            }
            phase("properties");
            let mut response = self
                .agent
                .get(format!("http://{}{path}", self.config.endpoint))
                .header("Authorization", format!("Bearer {}", self.key))
                .config()
                .timeout_global(Some(remaining()?))
                .build()
                .call()
                .map_err(|_| Error::Transport)?;
            check_response(&response, "application/json")?;
            let value = read_json(response.body_mut().as_reader(), 64 * 1024)?;
            if self.config.glyd_identity.is_some() {
                self.accept_glyd_info(&value)?;
            } else if let Some(expected) = &self.config.properties {
                self.properties = Some(parse_properties(
                    &value,
                    expected,
                    &self.config.backend_model,
                )?);
            }
        }
        phase("process_alive_final");
        self.check_alive()?;
        remaining()?;
        self.ready = true;
        phase("complete");
        Ok(())
    }
    fn accept_glyd_info(&mut self, value: &serde_json::Value) -> Result<()> {
        let identity = GlydIdentity::from_info(value)?;
        if self.config.glyd_identity.as_ref() != Some(&identity)
            || !identity.valid_for(&self.config)
        {
            return Err(Error::Unapproved);
        }
        let sample = GlydMemorySample::observe(self.glyd_memory.as_ref(), value, Instant::now())?;
        self.glyd_identity = Some(identity);
        self.glyd_memory = Some(sample);
        Ok(())
    }
    fn refresh_glyd_info(&mut self) -> Result<()> {
        self.check_alive()?;
        let mut response = self
            .agent
            .get(format!("http://{}/v1/glyd/info", self.config.endpoint))
            .header("Authorization", format!("Bearer {}", self.key))
            .config()
            .timeout_global(Some(Duration::from_secs(2)))
            .build()
            .call()
            .map_err(|_| Error::Transport)?;
        check_response(&response, "application/json")?;
        self.accept_glyd_info(&read_json(response.body_mut().as_reader(), 64 * 1024)?)?;
        self.check_alive()
    }
    fn finish_glyd_request(&mut self) -> Result<()> {
        if let Some(control) = &self.request {
            control.finish()?;
            // An ended ID is not enough: the same service must be ready and idle,
            // with coherent retained allocations, before terminal output or reuse.
            self.refresh_glyd_info()?;
        }
        Ok(())
    }
    /// CPU slot-idle acknowledgement after response completion, not GPU/device drain.
    pub fn confirm_request_idle(&mut self) -> Result<()> {
        if self.config.api != ServiceApi::InputTokensV1 {
            return Err(Error::Unapproved);
        }
        let deadline = Instant::now() + Duration::from_secs(2);
        loop {
            self.check_alive()?;
            let remaining = deadline
                .checked_duration_since(Instant::now())
                .ok_or(Error::Unavailable)?;
            let mut response = self
                .agent
                .get(format!("http://{}/slots", self.config.endpoint))
                .header("Authorization", format!("Bearer {}", self.key))
                .config()
                .timeout_global(Some(remaining))
                .build()
                .call()
                .map_err(|_| Error::Transport)?;
            check_response(&response, "application/json")?;
            let value = read_json(response.body_mut().as_reader(), 64 * 1024)?;
            let slots = value.as_array().ok_or(Error::Invalid)?;
            if slots.len() != 1 {
                return Err(Error::Invalid);
            }
            match slots[0]["is_processing"].as_bool() {
                Some(false) => {
                    self.check_alive()?;
                    return Ok(());
                }
                Some(true) => std::thread::sleep(Duration::from_millis(20)),
                None => return Err(Error::Invalid),
            }
        }
    }
    /// One synchronous request per owned service. Unconfirmed failure stops the process; never retry.
    /// `emit` applies the caller's live authorization and backpressure before accepting each event.
    pub fn generate(
        &mut self,
        request: &ChatRequest,
        mut emit: impl FnMut(&[u8]) -> Result<()>,
    ) -> Result<()> {
        if self.config.api == ServiceApi::GlydServeV1 && self.counted.is_none() {
            self.request = Some(RequestCancellation::new(self));
        }
        let mut result = self.generate_inner(request, &mut emit);
        if let Some(control) = &self.request {
            if result.is_ok() {
                result = control.finish();
            }
            if matches!(result, Err(Error::Cancelled | Error::Transport)) {
                if result == Err(Error::Cancelled) || control.is_cancelled() {
                    let _ = control.cancel();
                }
                if control.cancelled_and_ended() && self.refresh_glyd_info().is_ok() {
                    return Err(Error::Cancelled);
                }
            }
        }
        if result.is_err() {
            self.ready = false;
            let _ = self.control.cancel();
        }
        result
    }
    /// Exact backend template/tokenizer counting extension. No byte estimate or fallback.
    /// The next generation must use this exact body and report matching prompt usage.
    pub fn count_prompt(&mut self, request: &ChatRequest) -> Result<u32> {
        self.counted = None;
        if self.config.api == ServiceApi::GlydServeV1 {
            self.request = Some(RequestCancellation::new(self));
        }
        let result = (|| {
            let body = self.request_body(request)?;
            let mut response = self
                .agent
                .post(format!(
                    "http://{}{}",
                    self.config.endpoint,
                    self.config.api.count_path()
                ))
                .header("Authorization", format!("Bearer {}", self.key))
                .header("Content-Type", "application/json")
                .header("Accept-Encoding", "identity")
                .config()
                .timeout_global(Some(Duration::from_secs(2)))
                .build()
                .send(&body)
                .map_err(|_| Error::Transport)?;
            check_response(&response, "application/json")?;
            let value = read_json(response.body_mut().as_reader(), 4096)?;
            if self.config.glyd_identity.as_ref().is_some_and(|i| {
                value["max_model_len"].as_u64() != Some(u64::from(i.context_tokens))
            }) {
                return Err(Error::Unapproved);
            }
            let count = parse_prompt_count(value, self.config.api, request.max_tokens.get())?;
            self.check_alive()?;
            self.counted = Some((starlings_protocol::byte_digest(&body), count));
            Ok(count)
        })();
        if result.is_err() {
            self.ready = false;
            let _ = self.control.cancel();
        }
        result
    }
    fn request_body(&self, request: &ChatRequest) -> Result<Vec<u8>> {
        self.check_alive()?;
        if !self.ready || request.model != self.config.manifest_digest.as_str() {
            return Err(Error::Unapproved);
        }
        if request.temperature.is_none() || request.top_p.is_none() || request.seed.is_none() {
            return Err(Error::Invalid);
        }
        encode(request, FixtureMode::Reject).map_err(|_| Error::Invalid)?;
        let mut backend = request.clone();
        backend.model.clone_from(&self.config.backend_model);
        let mut body = serde_json::to_value(&backend).map_err(|_| Error::Invalid)?;
        if self.config.api == ServiceApi::GlydServeV1 {
            body["request_id"] = self
                .request
                .as_ref()
                .ok_or(Error::Unapproved)?
                .body_id()
                .into();
        }
        if request.stream == Some(true) {
            body["stream_options"] = serde_json::json!({"include_usage":true});
        }
        serde_json::to_vec(&body).map_err(|_| Error::Invalid)
    }
    fn generate_inner(
        &mut self,
        request: &ChatRequest,
        emit: &mut impl FnMut(&[u8]) -> Result<()>,
    ) -> Result<()> {
        if self.config.api == ServiceApi::GlydServeV1 {
            self.refresh_glyd_info()?;
        }
        let body = self.request_body(request)?;
        let mut output = Output::default();
        if let Some(control) = &self.request {
            control.dispatch()?;
            output.id = Some(control.id().to_owned());
        }
        if let Some((expected, tokens)) = self.counted.take() {
            if starlings_protocol::byte_digest(&body) != expected {
                return Err(Error::Unapproved);
            }
            output.expected_prompt = Some(tokens);
        }
        let mut response = self
            .agent
            .post(format!(
                "http://{}/v1/chat/completions",
                self.config.endpoint
            ))
            .header("Authorization", format!("Bearer {}", self.key))
            .header("Content-Type", "application/json")
            .header("Accept-Encoding", "identity")
            .send(&body)
            .map_err(|_| Error::Transport)?;
        if request.stream == Some(true) {
            check_response(&response, "text/event-stream")?;
            let mut reader = BufReader::new(response.body_mut().as_reader());
            let mut total = 0;
            let mut data = Vec::new();
            let mut terminal: Vec<Vec<u8>> = Vec::new();
            loop {
                self.check_generation()?;
                let mut line = Vec::new();
                let n = reader
                    .by_ref()
                    .take(64 * 1024 + 1)
                    .read_until(b'\n', &mut line)
                    .map_err(|_| Error::Transport)?;
                total += n;
                if n == 0 {
                    return Err(Error::Transport);
                }
                if n > 64 * 1024 || total > 16 * 1024 * 1024 {
                    return Err(Error::Capacity);
                }
                if line.pop() != Some(b'\n') {
                    return Err(Error::Transport);
                }
                if line.last() == Some(&b'\r') {
                    line.pop();
                }
                if line.is_empty() {
                    if data.is_empty() {
                        continue;
                    }
                    data.pop();
                    if data == b"[DONE]" {
                        if !output.finished || !output.usage {
                            return Err(Error::Invalid);
                        }
                        self.finish_glyd_request()?;
                        for bytes in terminal {
                            self.check_generation()?;
                            emit(&bytes)?;
                        }
                        return Ok(());
                    }
                    let mut value = parse_json(&data).map_err(|_| Error::Invalid)?;
                    normalize_output(&mut value, self.config.api, true)?;
                    validate_output(
                        &value,
                        &self.config,
                        true,
                        request.max_tokens.get(),
                        &mut output,
                    )?;
                    self.check_generation()?;
                    if self.config.api == ServiceApi::GlydServeV1 {
                        let bytes = serde_json::to_vec(&value).map_err(|_| Error::Invalid)?;
                        if output.finished {
                            if terminal.len() >= 2 {
                                return Err(Error::Capacity);
                            }
                            terminal.push(bytes);
                        } else {
                            emit(&bytes)?;
                        }
                    } else {
                        emit(&data)?;
                    }
                    data.clear();
                } else if let Some(part) = line.strip_prefix(b"data:") {
                    data.extend_from_slice(part.strip_prefix(b" ").unwrap_or(part));
                    data.push(b'\n');
                    if data.len() > 64 * 1024 {
                        return Err(Error::Capacity);
                    }
                } else if !line.starts_with(b":") {
                    return Err(Error::Invalid);
                }
            }
        } else {
            check_response(&response, "application/json")?;
            let mut value = read_json(
                response.body_mut().as_reader(),
                starlings_protocol::MAX_JSON_BYTES,
            )?;
            normalize_output(&mut value, self.config.api, false)?;
            validate_output(
                &value,
                &self.config,
                false,
                request.max_tokens.get(),
                &mut output,
            )?;
            self.check_generation()?;
            self.finish_glyd_request()?;
            self.check_generation()?;
            emit(&serde_json::to_vec(&value).map_err(|_| Error::Invalid)?)
        }
    }
    fn check_generation(&self) -> Result<()> {
        self.check_alive()?;
        if let Some(control) = &self.request {
            control.check()?;
        }
        Ok(())
    }
    /// Kill the group/job and observe parent exit within two seconds. This does not prove GPU drain.
    /// Native qualification must establish process-tree/device cleanup before releasing reservations.
    pub fn stop(&mut self) -> Result<()> {
        self.ready = false;
        let _ = self.control.cancel();
        let deadline = Instant::now() + Duration::from_secs(2);
        loop {
            let mut process = self
                .control
                .process
                .lock()
                .map_err(|_| Error::StopUnconfirmed)?;
            if process.stopped || process.reap()? {
                return Ok(());
            }
            drop(process);
            if Instant::now() >= deadline {
                return Err(Error::StopUnconfirmed);
            }
            std::thread::sleep(Duration::from_millis(5));
        }
    }
    /// Stop, then confirm that the original POSIX process group no longer exists.
    /// No signal is sent during confirmation. Escaped descendants and GPU work are
    /// outside this proof; native qualification must establish those constraints.
    /// Windows returns Unavailable until job-wide completion is independently exposed.
    pub fn confirm_process_group_exit(&mut self) -> Result<()> {
        self.stop()?;
        #[cfg(unix)]
        {
            let pid = self
                .control
                .process
                .lock()
                .map_err(|_| Error::StopUnconfirmed)?
                .pid()?;
            let deadline = Instant::now() + Duration::from_secs(2);
            loop {
                match rustix::process::test_kill_process_group(pid) {
                    Err(rustix::io::Errno::SRCH) => return Ok(()),
                    _ if Instant::now() < deadline => {
                        // Permission/exit races are not proof of absence. Continue
                        // observing within the same bound; only ESRCH confirms exit.
                        std::thread::sleep(Duration::from_millis(5))
                    }
                    _status => {
                        #[cfg(test)]
                        eprintln!("owned group absence unconfirmed: {_status:?}");
                        return Err(Error::StopUnconfirmed);
                    }
                }
            }
        }
        #[cfg(not(unix))]
        Err(Error::Unavailable)
    }
}

fn parse_properties(
    value: &serde_json::Value,
    expected: &ExpectedProperties,
    model: &str,
) -> Result<Properties> {
    let build_info = value["build_info"].as_str().ok_or(Error::Invalid)?;
    let context_tokens = value["default_generation_settings"]["n_ctx"]
        .as_u64()
        .and_then(|n| u32::try_from(n).ok())
        .ok_or(Error::Invalid)?;
    let caps = &value["chat_template_caps"];
    let report = Properties {
        build_info: expected.build_info.clone(),
        context_tokens,
        system_role: caps["supports_system_role"]
            .as_bool()
            .ok_or(Error::Invalid)?,
        string_content: caps["supports_string_content"]
            .as_bool()
            .ok_or(Error::Invalid)?,
        typed_content: caps["supports_typed_content"]
            .as_bool()
            .ok_or(Error::Invalid)?,
    };
    if build_info != expected.build_info
        || context_tokens != expected.context_tokens
        || value["model_alias"].as_str() != Some(model)
        || value["total_slots"].as_u64() != Some(1)
        || !report.system_role
        || !(report.string_content || report.typed_content)
    {
        return Err(Error::Unapproved);
    }
    Ok(report)
}
impl Drop for Service {
    fn drop(&mut self) {
        let _ = self.stop();
    }
}
fn check_response(response: &ureq::http::Response<ureq::Body>, expected: &str) -> Result<()> {
    if response.status().as_u16() != 200
        || response
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .map(|s| s.split(';').next().unwrap_or("").trim())
            != Some(expected)
        || response
            .headers()
            .get("content-encoding")
            .is_some_and(|v| v != "identity")
    {
        return Err(Error::Transport);
    }
    Ok(())
}
fn parse_prompt_count(value: serde_json::Value, api: ServiceApi, max_tokens: u32) -> Result<u32> {
    let tokens = match api {
        ServiceApi::InputTokensV1 => {
            #[derive(Deserialize)]
            #[serde(deny_unknown_fields)]
            struct Count {
                object: String,
                input_tokens: u32,
            }
            let count: Count = serde_json::from_value(value).map_err(|_| Error::Invalid)?;
            if count.object != "response.input_tokens" {
                return Err(Error::Invalid);
            }
            count.input_tokens
        }
        ServiceApi::GlydServeV1 => {
            #[derive(Deserialize)]
            #[serde(deny_unknown_fields)]
            struct Count {
                prompt_tokens: u32,
                max_model_len: u32,
            }
            let count: Count = serde_json::from_value(value).map_err(|_| Error::Invalid)?;
            if !(1..=2_000_000).contains(&count.max_model_len)
                || count
                    .prompt_tokens
                    .checked_add(max_tokens)
                    .is_none_or(|n| n > count.max_model_len)
            {
                return Err(Error::Invalid);
            }
            count.prompt_tokens
        }
    };
    if !(1..=1_048_576).contains(&tokens) {
        return Err(Error::Invalid);
    }
    Ok(tokens)
}

fn normalize_output(value: &mut serde_json::Value, api: ServiceApi, stream: bool) -> Result<()> {
    if api != ServiceApi::GlydServeV1 {
        return Ok(());
    }
    for choice in value
        .get_mut("choices")
        .and_then(serde_json::Value::as_array_mut)
        .ok_or(Error::Invalid)?
    {
        let fields = choice
            .get_mut(if stream { "delta" } else { "message" })
            .and_then(serde_json::Value::as_object_mut)
            .ok_or(Error::Invalid)?;
        if let Some(alias) = fields.remove("reasoning")
            && ((!alias.is_null() && !alias.is_string())
                || fields.get("reasoning_content") != Some(&alias))
        {
            return Err(Error::Invalid);
        }
    }
    Ok(())
}

fn read_json(reader: impl Read, limit: usize) -> Result<serde_json::Value> {
    let mut bytes = Vec::new();
    reader
        .take(limit as u64 + 1)
        .read_to_end(&mut bytes)
        .map_err(|_| Error::Transport)?;
    if bytes.len() > limit {
        return Err(Error::Capacity);
    }
    parse_json(&bytes).map_err(|_| Error::Invalid)
}
#[derive(Default)]
struct Output {
    id: Option<String>,
    finished: bool,
    usage: bool,
    expected_prompt: Option<u32>,
}
fn validate_output(
    value: &serde_json::Value,
    config: &ServiceConfig,
    stream: bool,
    max_tokens: u32,
    output: &mut Output,
) -> Result<()> {
    let id = value["id"]
        .as_str()
        .filter(|id| !id.is_empty() && id.len() <= 256)
        .ok_or(Error::Invalid)?;
    if output.id.as_ref().is_some_and(|old| old != id)
        || output.usage
        || value["object"]
            != if stream {
                "chat.completion.chunk"
            } else {
                "chat.completion"
            }
    {
        return Err(Error::Invalid);
    }
    output.id = Some(id.to_owned());
    if value["model"].as_str() != Some(config.backend_model.as_str()) {
        return Err(Error::Unapproved);
    }
    let choices = value["choices"].as_array().ok_or(Error::Invalid)?;
    if choices.len() > 1 || (!stream && choices.len() != 1) {
        return Err(Error::Invalid);
    }
    if let Some(choice) = choices.first() {
        if output.finished || choice["index"] != 0 {
            return Err(Error::Invalid);
        }
        let message = &choice[if stream { "delta" } else { "message" }];
        let fields = message.as_object().ok_or(Error::Invalid)?;
        if fields
            .keys()
            .any(|k| !matches!(k.as_str(), "role" | "content" | "reasoning_content"))
            || fields.get("role").is_some_and(|v| v != "assistant")
            || ["content", "reasoning_content"].iter().any(|k| {
                fields
                    .get(*k)
                    .is_some_and(|v| !v.is_null() && !v.is_string())
            })
            || (!config.reasoning_content
                && fields
                    .get("reasoning_content")
                    .is_some_and(|v| !v.is_null()))
        {
            return Err(Error::Invalid);
        }
        match choice["finish_reason"].as_str() {
            Some("stop" | "length") => output.finished = true,
            None if stream && choice["finish_reason"].is_null() => (),
            _ => return Err(Error::Invalid),
        }
    }
    if let Some(usage) = value.get("usage").filter(|v| !v.is_null()) {
        if !output.finished {
            return Err(Error::Invalid);
        }
        let prompt = usage["prompt_tokens"].as_u64().ok_or(Error::Invalid)?;
        let completion = usage["completion_tokens"].as_u64().ok_or(Error::Invalid)?;
        let total = usage["total_tokens"].as_u64().ok_or(Error::Invalid)?;
        if prompt > 1_048_576
            || output
                .expected_prompt
                .is_some_and(|expected| prompt != u64::from(expected))
            || completion > u64::from(max_tokens)
            || prompt.checked_add(completion) != Some(total)
        {
            return Err(Error::Invalid);
        }
        output.usage = true;
    } else if !stream {
        return Err(Error::Invalid);
    }
    Ok(())
}

#[cfg(test)]
mod tests;

#[cfg(any(test, feature = "test-backend"))]
pub mod fixture;
