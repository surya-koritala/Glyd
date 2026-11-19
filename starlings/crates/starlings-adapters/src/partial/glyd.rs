//! Owned glyd-range/1 pipes. Receipts do not grant authority, native eligibility or device drain.
use super::*;
use serde_json::{Value, json};
use starlings_protocol::{Digest, InvocationId, content_hash};
use std::{
    collections::{BTreeMap, BTreeSet},
    io::{Read, Write},
};

const HEAD_LIMIT: usize = 1024 * 1024;
const BODY_LIMIT: usize = 64 * 1024 * 1024;

/// The owner approves the actual descriptor separately from immutable package/asset identities.
/// Hash the descriptor without its request `id`, `ok`, process `pid` and `incarnation`.
pub fn descriptor_digest(value: &Value) -> Result<Digest> {
    let mut value = value.as_object().ok_or(Error::Invalid)?.clone();
    for field in ["id", "ok", "pid", "incarnation"] {
        value.remove(field);
    }
    content_hash(&value).map_err(|_| Error::Invalid)
}

/// One approved `glyd-engine rangeworker` child: exactly the approved bytes, no shell or PATH
/// search, an explicit environment, its own process group (job on Windows) and private stdio
/// pipes. Clones share the child, so a watchdog can stop it while a call is blocked on its
/// pipes. Stopping kills the whole group; that is not proof of device drain.
#[derive(Clone)]
pub struct RangeProcess(std::sync::Arc<std::sync::Mutex<Owned>>);
struct Owned {
    child: Box<dyn process_wrap::std::ChildWrapper>,
    _pin: Pin,
}
impl Drop for Owned {
    fn drop(&mut self) {
        let _ = self.child.kill();
    }
}
/// Keeps what the child runs and opens fixed from hash to use: on Linux the engine is a sealed
/// memfd, elsewhere a private copy; `GLYD_GPU_LIB` is always a private copy. Copies live in a
/// fresh 0700 directory removed after the child.
// ponytail: a crashed daemon leaves its copies in the private directory.
#[derive(Default)]
struct Pin {
    directory: Option<std::path::PathBuf>,
    #[cfg(target_os = "linux")]
    _memfd: Option<std::fs::File>,
}
impl Drop for Pin {
    fn drop(&mut self) {
        if let Some(directory) = &self.directory {
            let _ = std::fs::remove_dir_all(directory);
        }
    }
}
impl Pin {
    /// Copy `source` into this child's private directory; the copy's bytes are checked.
    fn copy(
        &mut self,
        private: &std::path::Path,
        source: &std::path::Path,
        name: &str,
        approved: &Digest,
    ) -> Result<std::path::PathBuf> {
        let directory = match &self.directory {
            Some(directory) => directory.clone(),
            None => {
                let directory = private.join(uuid::Uuid::new_v4().simple().to_string());
                let mut builder = std::fs::DirBuilder::new();
                builder.recursive(true);
                #[cfg(unix)]
                std::os::unix::fs::DirBuilderExt::mode(&mut builder, 0o700);
                builder.create(&directory).map_err(|_| Error::State)?;
                self.directory = Some(directory.clone());
                directory
            }
        };
        let target = directory.join(name);
        let mut options = std::fs::OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        std::os::unix::fs::OpenOptionsExt::mode(&mut options, 0o700);
        let mut copy = options.open(&target).map_err(|_| Error::State)?;
        let mut file = std::fs::File::open(source).map_err(|_| Error::Invalid)?;
        std::io::copy(&mut file, &mut copy).map_err(|_| Error::State)?;
        drop(copy);
        let mut copy = std::fs::File::open(&target).map_err(|_| Error::State)?;
        if crate::service::file_digest(&mut copy).map_err(|_| Error::Invalid)? != *approved {
            return Err(Error::Invalid);
        }
        Ok(target)
    }
    /// The engine sealed in memory (no write, grow, shrink or further seals), its sealed
    /// bytes checked, then run as /proc/self/fd/N.
    #[cfg(target_os = "linux")]
    fn sealed(
        &mut self,
        source: &std::path::Path,
        approved: &Digest,
    ) -> Result<std::path::PathBuf> {
        use rustix::fs::{MemfdFlags, SealFlags};
        use std::{io::Seek, os::fd::AsRawFd};
        let flags = MemfdFlags::CLOEXEC | MemfdFlags::ALLOW_SEALING;
        // Kernels before 6.3 refuse MFD_EXEC; on them a memfd is executable by default.
        let memfd = rustix::fs::memfd_create("starlings-engine", flags | MemfdFlags::EXEC)
            .or_else(|_| rustix::fs::memfd_create("starlings-engine", flags))
            .map_err(|_| Error::State)?;
        let mut memfd = std::fs::File::from(memfd);
        let mut file = std::fs::File::open(source).map_err(|_| Error::Invalid)?;
        std::io::copy(&mut file, &mut memfd).map_err(|_| Error::State)?;
        rustix::fs::fcntl_add_seals(
            &memfd,
            SealFlags::WRITE | SealFlags::GROW | SealFlags::SHRINK | SealFlags::SEAL,
        )
        .map_err(|_| Error::State)?;
        memfd.rewind().map_err(|_| Error::State)?;
        if crate::service::file_digest(&mut memfd).map_err(|_| Error::Invalid)? != *approved {
            return Err(Error::Invalid);
        }
        let path = format!("/proc/self/fd/{}", memfd.as_raw_fd()).into();
        self._memfd = Some(memfd);
        Ok(path)
    }
}
impl RangeProcess {
    /// `private` is a directory only this user can write; a `GLYD_GPU_LIB` in `environment`
    /// needs `gpu_library`, its approved digest, and the child gets a checked copy.
    pub fn spawn(
        executable: &std::path::Path,
        approved: &Digest,
        arguments: &[String],
        environment: &BTreeMap<String, String>,
        gpu_library: Option<&Digest>,
        private: &std::path::Path,
    ) -> Result<(Self, std::process::ChildStdout, std::process::ChildStdin)> {
        use std::process::Stdio;
        if !executable.is_absolute() || !private.is_absolute() {
            return Err(Error::Invalid);
        }
        let mut pin = Pin::default();
        #[cfg(target_os = "linux")]
        let program = pin.sealed(executable, approved)?;
        #[cfg(not(target_os = "linux"))]
        let program = pin.copy(private, executable, "engine", approved)?;
        let mut environment = environment.clone();
        match (environment.get_mut("GLYD_GPU_LIB"), gpu_library) {
            (Some(library), Some(approved)) => {
                let copy = pin.copy(
                    private,
                    std::path::Path::new(library),
                    "gpu-library",
                    approved,
                )?;
                *library = copy.to_str().ok_or(Error::Invalid)?.to_owned();
            }
            (None, None) => {}
            _ => return Err(Error::Invalid),
        }
        let mut command = std::process::Command::new(program);
        command
            .args(arguments)
            .env_clear()
            .envs(&environment)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null());
        let mut command = process_wrap::std::CommandWrap::from(command);
        #[cfg(unix)]
        command.wrap(process_wrap::std::ProcessGroup::leader());
        #[cfg(windows)]
        command.wrap(process_wrap::std::JobObject);
        let mut child = command.spawn().map_err(|_| Error::State)?;
        let (Some(stdout), Some(stdin)) = (child.stdout().take(), child.stdin().take()) else {
            let _ = child.kill();
            return Err(Error::State);
        };
        let owned = Owned { child, _pin: pin };
        Ok((
            Self(std::sync::Arc::new(std::sync::Mutex::new(owned))),
            stdout,
            stdin,
        ))
    }
    pub fn alive(&self) -> bool {
        self.0
            .lock()
            .is_ok_and(|mut owned| matches!(owned.child.try_wait(), Ok(None)))
    }
    /// Signal the whole group without waiting; a call blocked on its pipes then fails.
    /// The last clone's drop kills and reaps it.
    pub fn stop(&self) {
        if let Ok(mut owned) = self.0.lock() {
            let _ = owned.child.start_kill();
        }
    }
}

/// One `describe` on fresh pipes, before an approval exists. The caller compares its
/// `descriptor_digest` with the operator's approved digest before using anything in it.
pub fn describe(reader: &mut impl Read, writer: &mut impl Write) -> Result<Value> {
    let head = br#"{"op":"describe","id":"approve"}"#;
    writer
        .write_all(&(head.len() as u32).to_le_bytes())
        .and_then(|_| writer.write_all(&[0; 4]))
        .and_then(|_| writer.write_all(head))
        .and_then(|_| writer.flush())
        .map_err(|_| Error::State)?;
    let mut prefix = [0; 8];
    reader.read_exact(&mut prefix).map_err(|_| Error::State)?;
    let length = u32::from_le_bytes(prefix[..4].try_into().unwrap()) as usize;
    if prefix[4..] != [0; 4] || length == 0 || length > HEAD_LIMIT {
        return Err(Error::Invalid);
    }
    let mut bytes = vec![0; length];
    reader.read_exact(&mut bytes).map_err(|_| Error::State)?;
    let value = starlings_protocol::parse_json(&bytes).map_err(|_| Error::Invalid)?;
    if value["id"] != "approve" || value["ok"] != true {
        return Err(Error::Invalid);
    }
    Ok(value)
}

/// The Qwen3.5 dense boundary of an approved descriptor: token ids into the first stage, the
/// bf16 residual/MLP-output pair between stages, one sampled token out of the last.
/// `workload` carries this stage's maxima; each session states its own counts.
pub fn qwen_boundary(
    descriptor: &Value,
    build: &Digest,
    execution_profile: &str,
    boundary_profile: &str,
    workload: starlings_protocol::ProfileWorkload,
) -> Result<(BoundaryContract, bool, bool, u32)> {
    let (first, last) = (
        descriptor["stage"]["first"]
            .as_bool()
            .ok_or(Error::Invalid)?,
        descriptor["stage"]["last"]
            .as_bool()
            .ok_or(Error::Invalid)?,
    );
    let hidden = BoundedU32::new(
        u32::try_from(number(&descriptor["model"], "hidden")?).map_err(|_| Error::Invalid)?,
    )
    .map_err(|_| Error::Invalid)?;
    let vocabulary =
        u32::try_from(number(&descriptor["model"], "vocab")?).map_err(|_| Error::Invalid)?;
    let pair = || {
        [
            TensorHeaderTensorRole::Residual,
            TensorHeaderTensorRole::Hidden,
        ]
        .map(|role| TensorPattern {
            role,
            dtype: TensorHeaderDtype::Bf16,
            shape: vec![Dimension::Tokens, Dimension::Fixed(hidden)],
        })
        .to_vec()
    };
    let tokens = || {
        vec![TensorPattern {
            role: TensorHeaderTensorRole::TokenIds,
            dtype: TensorHeaderDtype::U32,
            shape: vec![Dimension::Tokens],
        }]
    };
    let part = |key: &str| content_hash(&descriptor[key]).map_err(|_| Error::Invalid);
    let contract = BoundaryContract {
        version: Version,
        fixture: false,
        key: starlings_protocol::ProfileKey {
            manifest_digest: part("model")?,
            execution_profile: execution_profile.into(),
            boundary_profile: boundary_profile.into(),
            backend_build_digest: build.clone(),
            runtime_config_digest: part("limits")?,
            device_digest: part("device")?,
            pool_inventory_digest: part("state")?,
            workload,
        },
        inputs: if first { tokens() } else { pair() },
        outputs: if last { tokens() } else { pair() },
        output_tokens: if last {
            OutputTokens::SampledOne
        } else {
            OutputTokens::SameChunk
        },
    };
    contract.validate(FixtureMode::Reject)?;
    Ok((contract, first, last, vocabulary))
}

pub struct Approval {
    pub descriptor: Digest,
    pub boundary: BoundaryContract,
    /// Engine-owned bytes only. This is not a full GPU pool/memory qualification.
    pub engine_bytes_limit: u64,
    pub payload_bytes_limit: u64,
    pub first: bool,
    pub last: bool,
    pub vocabulary: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Allocation {
    pub weights_bytes: u64,
    pub kv_bytes: u64,
    pub state_bytes: u64,
    pub scratch_bytes: u64,
    pub engine_bytes: u64,
    pub peak_bytes: u64,
    /// The provider's device report at prepare (name, free bytes, total bytes), if it gave one.
    pub device: Option<(String, u64, u64)>,
}

#[derive(Debug)]
pub enum Output {
    Boundary {
        layouts: Vec<TensorLayout>,
        bytes: Vec<u8>,
    },
    Token {
        token: u32,
        /// One greedy token per row of a multi-row decode chunk (GLY-360); else `[token]`.
        tokens: Vec<u32>,
        logits: Option<Vec<u8>>,
    },
}

#[derive(Debug)]
pub enum Event {
    Described,
    Prepared(Allocation),
    Committed,
    /// The engine's handle for the new session; later calls name it.
    Begun(String),
    Chunk {
        step: StepIdentity,
        output: Option<Output>,
    },
    Withdrawn(StepIdentity),
    Status {
        cancelling: bool,
    },
    Cancelled,
    /// The provider's count of positions the ended session held.
    Ended {
        tokens: u64,
    },
    /// A provider allocation receipt, never proof of full device/context cleanup.
    Unloaded {
        retained_bytes: u64,
    },
    Tokens(Vec<u32>),
    Text(String),
}

enum Pending {
    Describe,
    Prepare,
    Commit,
    Begin(DataScope, u32, u32, Option<starlings_protocol::Sampling>),
    Tokenize,
    Detokenize,
    Run {
        session: String,
        step: StepIdentity,
        head: bool,
        logits: bool,
    },
    Status(Option<String>),
    Cancel(String),
    End(String),
    Unload,
}
struct Session {
    handle: String,
    owner: DataScope,
    /// This session's own bounds, within the approved workload maxima.
    prompt: u32,
    output: u32,
    /// The last stage draws with the provider's sampler; None is greedy.
    sampling: Option<starlings_protocol::Sampling>,
    position: u32,
    /// GLY-360: after a multi-row decode chunk from `start`, the next chunk may start from
    /// start + 1 up to `position`; the engine keeps that many of its rows (`keep_previous`).
    rewind: Option<u32>,
    sequence: u64,
    invocations: BTreeSet<InvocationId>,
    last_request: Option<String>,
    cancelled: bool,
}

/// Takes exclusive ownership of the process's protocol pipe halves, not a borrowed shared stream.
/// The process supervisor must supply bounded I/O and retain its group/job/pin/quarantine owner.
/// Hold the existing live StageGrant/lease fence around every begin, dispatch and publication;
/// DataScope and provider handles alone cannot authorize any of them. Raw methods cannot be
/// used as a daemon/native activation path. The authoritative worker owns retired generations.
pub struct Worker<R, W> {
    reader: R,
    writer: W,
    approval: Approval,
    mode: FixtureMode,
    incarnation: Option<String>,
    instance: Option<String>,
    allocation: Option<Allocation>,
    committed: bool,
    /// The open sessions, in the order begun: at most the approved workload's `sessions`
    /// (GLY-190), each a request on the engine's own state.
    sessions: Vec<Session>,
    pending: BTreeMap<String, Pending>,
    next_id: u64,
    head_limit: usize,
    body_limit: usize,
    broken: bool,
}

impl<R: Read, W: Write> Worker<R, W> {
    pub fn new(reader: R, writer: W, approval: Approval, mode: FixtureMode) -> Result<Self> {
        approval.boundary.validate(mode)?;
        if approval.engine_bytes_limit == 0
            || approval.payload_bytes_limit == 0
            || approval.payload_bytes_limit > 2 * BODY_LIMIT as u64
            || approval.vocabulary == 0
            || approval.boundary.key.workload.sessions.get() > 2
        {
            return Err(Error::Invalid);
        }
        // Qwen's documented pair is residual then MLP output (the generic hidden role).
        let pair = |patterns: &[TensorPattern]| {
            patterns.len() == 2
                && patterns[0].role == TensorHeaderTensorRole::Residual
                && patterns[1].role == TensorHeaderTensorRole::Hidden
                && patterns.iter().all(|p| p.dtype == TensorHeaderDtype::Bf16)
        };
        let tokens = |patterns: &[TensorPattern]| {
            patterns.len() == 1 && patterns[0].role == TensorHeaderTensorRole::TokenIds
        };
        if !(if approval.first {
            tokens(&approval.boundary.inputs)
        } else {
            pair(&approval.boundary.inputs)
        }) || !(if approval.last {
            tokens(&approval.boundary.outputs)
        } else {
            pair(&approval.boundary.outputs)
        }) || (approval.boundary.output_tokens == OutputTokens::SampledOne) != approval.last
        {
            return Err(Error::Unsupported);
        }
        approval.boundary.layouts(
            approval.boundary.key.workload.prefill_chunk_tokens.get(),
            approval.payload_bytes_limit,
        )?;
        Ok(Self {
            reader,
            writer,
            approval,
            mode,
            incarnation: None,
            instance: None,
            allocation: None,
            committed: false,
            sessions: Vec::new(),
            pending: BTreeMap::new(),
            next_id: 0,
            head_limit: HEAD_LIMIT,
            body_limit: BODY_LIMIT,
            broken: false,
        })
    }

    /// An I/O/protocol failure permanently closes this mapping. Never retransmit an ambiguous step.
    pub fn is_ambiguous(&self) -> bool {
        self.broken
    }

    fn idle(&self) -> Result<()> {
        if self.broken || !self.pending.is_empty() {
            Err(Error::State)
        } else {
            Ok(())
        }
    }
    fn session(&self, handle: &str) -> Result<&Session> {
        self.sessions
            .iter()
            .find(|s| s.handle == handle)
            .ok_or(Error::State)
    }
    fn session_mut(&mut self, handle: &str) -> Result<&mut Session> {
        self.sessions
            .iter_mut()
            .find(|s| s.handle == handle)
            .ok_or(Error::State)
    }
    fn send(&mut self, mut value: Value, body: &[&[u8]], pending: Pending) -> Result<String> {
        let body_len = body
            .iter()
            .try_fold(0usize, |sum, b| sum.checked_add(b.len()))
            .ok_or(Error::Capacity)?;
        if self.broken || self.pending.len() >= 3 || body_len > self.body_limit {
            return Err(Error::State);
        }
        let next = self.next_id.checked_add(1).ok_or(Error::Capacity)?;
        let id = format!("c{next}");
        value["id"] = id.clone().into();
        let bytes = serde_json::to_vec(&value).map_err(|_| Error::Invalid)?;
        if bytes.is_empty() || bytes.len() > self.head_limit {
            return Err(Error::Capacity);
        }
        self.next_id = next;
        self.pending.insert(id.clone(), pending);
        let written = self
            .writer
            .write_all(&(bytes.len() as u32).to_le_bytes())
            .and_then(|_| self.writer.write_all(&(body_len as u32).to_le_bytes()))
            .and_then(|_| self.writer.write_all(&bytes))
            .and_then(|_| body.iter().try_for_each(|part| self.writer.write_all(part)))
            .and_then(|_| self.writer.flush());
        if written.is_err() {
            self.broken = true;
            return Err(Error::State);
        }
        Ok(id)
    }
    pub fn describe(&mut self) -> Result<String> {
        self.idle()?;
        if self.incarnation.is_some() {
            return Err(Error::State);
        }
        self.send(json!({"op":"describe"}), &[], Pending::Describe)
    }
    pub fn prepare(&mut self) -> Result<String> {
        self.idle()?;
        if self.incarnation.is_none() || self.instance.is_some() {
            return Err(Error::State);
        }
        self.send(json!({"op":"prepare"}), &[], Pending::Prepare)
    }
    pub fn commit(&mut self) -> Result<String> {
        self.idle()?;
        if self.committed {
            return Err(Error::State);
        }
        let instance = self.instance.as_ref().ok_or(Error::State)?;
        self.send(
            json!({"op":"commit","instance":instance}),
            &[],
            Pending::Commit,
        )
    }
    /// The live authoritative worker must approve this generation and reject retired ones first.
    /// `prompt` and `output` are this request's exact counts; the approved workload is their maximum.
    /// A nonzero temperature on the last stage draws each token from (seed, step, logits).
    pub fn begin_session(
        &mut self,
        owner: DataScope,
        prompt: u32,
        output: u32,
        sampling: Option<starlings_protocol::Sampling>,
    ) -> Result<String> {
        self.idle()?;
        let sampling = sampling.filter(|s| s.temperature != 0.0);
        if sampling.as_ref().is_some_and(|s| {
            !self.approval.last
                || !(0.0..=2.0).contains(&s.temperature)
                || !(s.top_p > 0.0 && s.top_p <= 1.0)
        }) {
            return Err(Error::Invalid);
        }
        let workload = &self.approval.boundary.key.workload;
        if prompt == 0
            || output == 0
            || prompt > workload.prompt_tokens.get()
            || output > workload.output_tokens.get()
            || prompt
                .checked_add(output)
                .is_none_or(|n| n > workload.context_tokens.get())
        {
            return Err(Error::Capacity);
        }
        if !self.committed
            || self.sessions.len() >= workload.sessions.get() as usize
            || self
                .sessions
                .iter()
                .any(|s| s.owner.generation_id == owner.generation_id)
            || owner.sequence.0 != 0
            || owner.authority_epoch.0 == 0
            || owner.source_stage == owner.destination_stage
        {
            return Err(Error::State);
        }
        starlings_protocol::encode(
            &starlings_protocol::DataControl::Cancel {
                protocol_version: Version,
                scope: owner.clone(),
                code: starlings_protocol::DataFailure::Cancelled,
            },
            self.mode,
        )
        .map_err(|_| Error::Invalid)?;
        let instance = self.instance.as_ref().ok_or(Error::State)?;
        self.send(
            json!({"op":"begin_session","instance":instance}),
            &[],
            Pending::Begin(owner, prompt, output, sampling),
        )
    }
    /// One user message through the model's chat template, with its generation prompt.
    /// Answered by the provider's control thread; it loads no weights and binds no session.
    pub fn tokenize(&mut self, text: &str) -> Result<String> {
        self.idle()?;
        if self.incarnation.is_none() {
            return Err(Error::State);
        }
        self.send(
            json!({"op":"tokenize","text":text,"chat":true}),
            &[],
            Pending::Tokenize,
        )
    }
    /// A chat (system, user and assistant turns, the last the user's) through the model's chat
    /// template with its generation prompt, as glyd serve renders the same messages (GLY-361).
    pub fn tokenize_chat(&mut self, messages: &[(&str, &str)]) -> Result<String> {
        self.idle()?;
        if self.incarnation.is_none() || messages.is_empty() || messages.len() > 256 {
            return Err(Error::State);
        }
        let messages: Vec<Value> = messages
            .iter()
            .map(|(role, content)| json!({"role": role, "content": content}))
            .collect();
        self.send(
            json!({"op":"tokenize","messages":messages}),
            &[],
            Pending::Tokenize,
        )
    }
    /// The whole text of `ids`, special tokens skipped. Streaming callers send the running ids.
    pub fn detokenize(&mut self, ids: &[u32]) -> Result<String> {
        self.idle()?;
        if self.incarnation.is_none()
            || ids.len() > self.approval.boundary.key.workload.context_tokens.get() as usize
            || ids.iter().any(|&t| t >= self.approval.vocabulary)
        {
            return Err(Error::Invalid);
        }
        self.send(
            json!({"op":"detokenize","ids":ids}),
            &[],
            Pending::Detokenize,
        )
    }
    /// Borrow complete ordered buffers without raw boundary copies; first-stage token JSON
    /// is unchanged. The local owner retains inputs through qualified completion, with a live grant.
    pub fn run_chunk(
        &mut self,
        session: &str,
        call: &RunChunk,
        expected: &StepIdentity,
        inputs: &[impl AsRef<[u8]>],
        head: bool,
        logits: bool,
    ) -> Result<String> {
        self.idle()?;
        let handle = session;
        let session = self.session(handle)?;
        if session.cancelled
            || session.invocations.contains(&call.step.scope.invocation_id)
            || head && !self.approval.last
            || logits && !head
        {
            return Err(Error::State);
        }
        let mut owner = session.owner.clone();
        owner.invocation_id = call.step.scope.invocation_id;
        owner.sequence = Decimal(session.sequence);
        let (start, rows) = (expected.position_start.get(), expected.token_count.get());
        // The rows of the last chunk this one keeps, when it starts before that chunk's end.
        let keep = match session.rewind {
            Some(from) if (from + 1..session.position).contains(&start) => Some(start - from),
            _ if start == session.position => None,
            _ => return Err(Error::Conflict),
        };
        if expected.scope != owner {
            return Err(Error::Conflict);
        }
        let decode = start >= session.prompt;
        if (expected.phase == TensorHeaderPhase::Decode) != decode
            || !decode && rows > session.prompt - start
            || decode && (rows > starlings_protocol::MAX_DECODE_ROWS || start + rows - session.prompt >= session.output)
            || self.approval.last && head != (decode || start + rows == session.prompt)
            // The engine draws only a chunk's last row; drafts are checked greedily.
            || head && rows > 1 && decode && session.sampling.is_some()
        {
            return Err(Error::Conflict);
        }
        call.validate(
            &self.approval.boundary,
            expected,
            self.approval.payload_bytes_limit,
            self.mode,
        )?;
        if inputs.len() != call.inputs.len() {
            return Err(Error::Invalid);
        }
        // Metadata validation bounds this to eight views; resolve each view once before lengths/I/O.
        let inputs: Vec<&[u8]> = inputs.iter().map(AsRef::as_ref).collect();
        if inputs
            .iter()
            .zip(&call.inputs)
            .any(|(b, l)| b.len() as u64 != l.bytes.0)
        {
            return Err(Error::Invalid);
        }
        let (_, outputs) = self.approval.boundary.layouts(
            expected.token_count.get(),
            self.approval.payload_bytes_limit,
        )?;
        let output_bytes = if self.approval.last {
            if logits {
                u64::from(self.approval.vocabulary) * 2
            } else {
                0
            }
        } else {
            outputs.iter().map(|l| l.bytes.0).sum()
        };
        let input_bytes: u64 = inputs.iter().map(|b| b.len() as u64).sum();
        if input_bytes > self.body_limit as u64
            || output_bytes > self.body_limit as u64
            || input_bytes
                .checked_add(output_bytes)
                .is_none_or(|n| n > self.approval.payload_bytes_limit)
        {
            return Err(Error::Capacity);
        }
        let mut request = json!({"op":"run_chunk", "session":session.handle,
            "phase": if decode {"decode"} else {"prefill"}, "position_start":start,
            "head":head, "logits":logits});
        if let Some(keep) = keep {
            request["keep_previous"] = keep.into();
        }
        // A multi-row decode chunk checks drafts: every stage runs it as `verify` (each row as
        // the plain step at its position, rewindable); the last returns each row's token.
        if decode && rows > 1 {
            request["verify"] = true.into();
            if head {
                request["head_rows"] = rows.into();
            }
        }
        if let Some(s) = session.sampling.as_ref().filter(|_| head) {
            // `step` is the answer's tokens before this one: 0 right after the prompt.
            request["sampling"] = json!({"temperature": s.temperature, "top_p": s.top_p,
                "seed": s.seed.get(), "step": start + rows - session.prompt});
        }
        if self.approval.first {
            let tokens: Vec<u32> = inputs[0]
                .chunks_exact(4)
                .map(|bytes| u32::from_le_bytes(bytes.try_into().unwrap()))
                .collect();
            if tokens.iter().any(|&t| t >= self.approval.vocabulary) {
                return Err(Error::Invalid);
            }
            request["tokens"] = json!(tokens);
        } else {
            request["rows"] = expected.token_count.get().into();
        }
        let id = self.send(
            request,
            if self.approval.first { &[] } else { &inputs },
            Pending::Run {
                session: handle.to_owned(),
                step: expected.clone(),
                head,
                logits,
            },
        )?;
        let session = self.session_mut(handle)?;
        session.invocations.insert(call.step.scope.invocation_id);
        session.last_request = Some(id.clone());
        Ok(id)
    }
    /// The engine's state, or one open session's (`session`); with no session named the
    /// engine reports its earliest open one.
    pub fn status(&mut self, session: Option<&str>) -> Result<String> {
        if self.incarnation.is_none()
            || self
                .pending
                .values()
                .any(|p| !matches!(p, Pending::Run { .. } | Pending::Cancel(_)))
        {
            return Err(Error::State);
        }
        let mut request = json!({"op":"status"});
        if let Some(handle) = session {
            self.session(handle)?;
            request["session"] = handle.into();
        }
        self.send(request, &[], Pending::Status(session.map(str::to_owned)))
    }
    pub fn cancel(&mut self, session: &str) -> Result<String> {
        if self
            .pending
            .values()
            .any(|p| !matches!(p, Pending::Run { .. } | Pending::Status(_)))
        {
            return Err(Error::State);
        }
        let session = self.session_mut(session)?;
        if session.cancelled {
            return Err(Error::State);
        }
        session.cancelled = true; // Withdraw output before the first possibly partial write.
        let handle = session.handle.clone();
        self.send(
            json!({"op":"cancel","session":handle}),
            &[],
            Pending::Cancel(handle),
        )
    }
    pub fn end_session(&mut self, session: &str) -> Result<String> {
        self.idle()?;
        let handle = self.session(session)?.handle.clone();
        self.send(
            json!({"op":"end_session","session":handle}),
            &[],
            Pending::End(handle),
        )
    }
    pub fn unload(&mut self) -> Result<String> {
        self.idle()?;
        if !self.sessions.is_empty() {
            return Err(Error::State);
        }
        let instance = self.instance.as_ref().ok_or(Error::State)?;
        self.send(
            json!({"op":"unload","instance":instance}),
            &[],
            Pending::Unload,
        )
    }

    /// Reads one correlated reply; status/cancel may precede an outstanding run reply.
    /// Publication of Chunk still requires the live local StageGrant/output fence.
    pub fn receive(&mut self) -> Result<Event> {
        if self.broken || self.pending.is_empty() {
            return Err(Error::State);
        }
        let result = self.receive_frame();
        if result.is_err() {
            self.broken = true;
        }
        result
    }
    fn receive_frame(&mut self) -> Result<Event> {
        let mut prefix = [0; 8];
        self.reader
            .read_exact(&mut prefix)
            .map_err(|_| Error::State)?;
        let head_len = u32::from_le_bytes(prefix[..4].try_into().unwrap()) as usize;
        let body_len = u32::from_le_bytes(prefix[4..].try_into().unwrap()) as usize;
        if head_len == 0 || head_len > self.head_limit || body_len > self.body_limit {
            return Err(Error::Capacity);
        }
        let mut bytes = vec![0; head_len];
        self.reader
            .read_exact(&mut bytes)
            .map_err(|_| Error::State)?;
        let value = starlings_protocol::parse_json(&bytes).map_err(|_| Error::Invalid)?;
        let id = value["id"].as_str().ok_or(Error::Invalid)?;
        let pending = self.pending.get(id).ok_or(Error::Conflict)?;
        let ok = value["ok"].as_bool().ok_or(Error::Invalid)?;
        let expected_body = match pending {
            Pending::Run {
                step, head, logits, ..
            } if ok => {
                if !self.approval.last {
                    let (_, layouts) = self
                        .approval
                        .boundary
                        .layouts(step.token_count.get(), self.approval.payload_bytes_limit)?;
                    layouts.iter().map(|l| l.bytes.0 as usize).sum()
                } else if *head && *logits {
                    self.approval.vocabulary as usize * 2
                } else {
                    0
                }
            }
            _ => 0,
        };
        if body_len != expected_body {
            return Err(Error::Invalid);
        }
        // Correlate/validate claimed body size before allocating it.
        let mut body = vec![0; body_len];
        self.reader
            .read_exact(&mut body)
            .map_err(|_| Error::State)?;
        let pending = self.pending.remove(id).ok_or(Error::Conflict)?;
        if !ok {
            if let Pending::Run { session, step, .. } = pending
                && self.session(&session).is_ok_and(|s| s.cancelled)
                && value["error"].as_str().is_some()
            {
                return Ok(Event::Withdrawn(step));
            }
            return Err(Error::State);
        }
        self.apply(pending, &value, body)
    }
    fn descriptor(&self, value: &Value) -> Result<String> {
        for field in [
            "build", "model", "stage", "boundary", "state", "limits", "ops", "device",
        ] {
            if value.get(field).is_none_or(Value::is_null) {
                return Err(Error::Invalid);
            }
        }
        // The engine's incarnation is an opaque token (hex), like its handles.
        let incarnation = handle(value, "incarnation")?;
        if value["protocol"] != "glyd-range/1"
            || descriptor_digest(value)? != self.approval.descriptor
            || self.incarnation.as_ref().is_some_and(|n| *n != incarnation)
            || value["stage"]["first"].as_bool() != Some(self.approval.first)
            || value["stage"]["last"].as_bool() != Some(self.approval.last)
            || value["model"]["vocab"].as_u64() != Some(self.approval.vocabulary.into())
            || value["limits"]["sessions"].as_u64()
                != Some(self.approval.boundary.key.workload.sessions.get().into())
            || value["limits"]["context_tokens"].as_u64()
                != Some(
                    self.approval
                        .boundary
                        .key
                        .workload
                        .context_tokens
                        .get()
                        .into(),
                )
            || value["limits"]["chunk_tokens"].as_u64().is_none_or(|n| {
                n < self
                    .approval
                    .boundary
                    .key
                    .workload
                    .prefill_chunk_tokens
                    .get()
                    .into()
            })
        {
            return Err(Error::Conflict);
        }
        let hidden =
            u32::try_from(number(&value["model"], "hidden")?).map_err(|_| Error::Invalid)?;
        if hidden == 0
            || value["boundary"]["dtype"] != "bf16"
            || number(&value["boundary"], "bytes_per_row")? != u64::from(hidden) * 4
        {
            return Err(Error::Unsupported);
        }
        for (patterns, tokens) in [
            (&self.approval.boundary.inputs, self.approval.first),
            (&self.approval.boundary.outputs, self.approval.last),
        ] {
            for pattern in patterns {
                let shape = if tokens {
                    vec![Dimension::Tokens]
                } else {
                    vec![
                        Dimension::Tokens,
                        Dimension::Fixed(BoundedU32::new(hidden).map_err(|_| Error::Invalid)?),
                    ]
                };
                if pattern.shape != shape {
                    return Err(Error::Conflict);
                }
            }
        }
        Ok(incarnation)
    }
    fn apply(&mut self, pending: Pending, value: &Value, body: Vec<u8>) -> Result<Event> {
        match pending {
            Pending::Describe => {
                let incarnation = self.descriptor(value)?;
                let limits = &value["limits"];
                let head = usize::try_from(number(limits, "max_head_bytes")?)
                    .map_err(|_| Error::Capacity)?;
                let body = usize::try_from(number(limits, "max_body_bytes")?)
                    .map_err(|_| Error::Capacity)?;
                if head == 0 || head > HEAD_LIMIT || body == 0 || body > BODY_LIMIT {
                    return Err(Error::Capacity);
                }
                self.incarnation = Some(incarnation);
                self.head_limit = head;
                self.body_limit = body;
                Ok(Event::Described)
            }
            // The engine's `describe` here is a one-line summary for people; the descriptor was
            // approved at `describe` on these same exclusive pipes.
            Pending::Prepare => {
                let a = Allocation {
                    weights_bytes: number(value, "weights_bytes")?,
                    kv_bytes: number(value, "kv_bytes")?,
                    state_bytes: number(value, "state_bytes")?,
                    scratch_bytes: number(value, "scratch_bytes")?,
                    engine_bytes: number(value, "engine_bytes")?,
                    peak_bytes: number(value, "peak_bytes")?,
                    device: value["device"]["name"].as_str().and_then(|name| {
                        let free = value["device"]["free_bytes"].as_u64()?;
                        let total = value["device"]["total_bytes"].as_u64()?;
                        (free <= total && name.len() <= 128).then(|| (name.to_owned(), free, total))
                    }),
                };
                if [a.weights_bytes, a.kv_bytes, a.state_bytes, a.scratch_bytes]
                    .into_iter()
                    .try_fold(0u64, |sum, n| sum.checked_add(n))
                    != Some(a.engine_bytes)
                    || a.engine_bytes > self.approval.engine_bytes_limit
                    || a.peak_bytes < a.engine_bytes
                    || a.peak_bytes > self.approval.engine_bytes_limit
                {
                    return Err(Error::Capacity);
                }
                self.instance = Some(handle(value, "instance")?);
                self.allocation = Some(a.clone());
                Ok(Event::Prepared(a))
            }
            Pending::Commit => {
                if value["committed"] != true || value["ready"] != true {
                    return Err(Error::State);
                }
                self.committed = true;
                Ok(Event::Committed)
            }
            Pending::Detokenize => Ok(Event::Text(
                value["text"].as_str().ok_or(Error::Invalid)?.to_owned(),
            )),
            Pending::Tokenize => {
                let ids = value["ids"].as_array().ok_or(Error::Invalid)?;
                if ids.is_empty()
                    || ids.len() > self.approval.boundary.key.workload.context_tokens.get() as usize
                {
                    return Err(Error::Capacity);
                }
                ids.iter()
                    .map(|id| {
                        id.as_u64()
                            .and_then(|n| u32::try_from(n).ok())
                            .filter(|&n| n < self.approval.vocabulary)
                            .ok_or(Error::Invalid)
                    })
                    .collect::<Result<_>>()
                    .map(Event::Tokens)
            }
            Pending::Begin(owner, prompt, output, sampling) => {
                let allocation = self.allocation.as_ref().ok_or(Error::State)?;
                // Prepare's bytes cover every session the engine was started for.
                let sessions = u64::from(self.approval.boundary.key.workload.sessions.get());
                let handle = handle(value, "session")?;
                if number(value, "tokens")? != 0
                    || number(value, "context_tokens")?
                        != u64::from(self.approval.boundary.key.workload.context_tokens.get())
                    || number(value, "kv_bytes")?.checked_mul(sessions) != Some(allocation.kv_bytes)
                    || number(value, "state_bytes")?.checked_mul(sessions)
                        != Some(allocation.state_bytes)
                    || self.sessions.iter().any(|s| s.handle == handle)
                {
                    return Err(Error::Conflict);
                }
                self.sessions.push(Session {
                    handle: handle.clone(),
                    owner,
                    prompt,
                    output,
                    sampling,
                    position: 0,
                    rewind: None,
                    sequence: 0,
                    invocations: BTreeSet::new(),
                    last_request: None,
                    cancelled: false,
                });
                Ok(Event::Begun(handle))
            }
            Pending::Run {
                session,
                step,
                head,
                logits,
            } => {
                let end = step
                    .position_start
                    .get()
                    .checked_add(step.token_count.get())
                    .ok_or(Error::Capacity)?;
                if number(value, "rows")? != u64::from(step.token_count.get())
                    || number(value, "position_end")? != u64::from(end)
                {
                    return Err(Error::Conflict);
                }
                let session = self.session_mut(&session)?;
                let (start, rows) = (step.position_start.get(), step.token_count.get());
                let decode = step.phase == TensorHeaderPhase::Decode;
                session.position = end;
                session.rewind = (decode && rows > 1).then_some(start);
                session.sequence = session.sequence.checked_add(1).ok_or(Error::Capacity)?;
                if session.cancelled {
                    return Ok(Event::Withdrawn(step));
                }
                let output = if !self.approval.last {
                    if number(&value["body"], "rows")? != u64::from(step.token_count.get())
                        || number(&value["body"], "bytes")? != body.len() as u64
                    {
                        return Err(Error::Invalid);
                    }
                    let (_, layouts) = self
                        .approval
                        .boundary
                        .layouts(step.token_count.get(), self.approval.payload_bytes_limit)?;
                    Some(Output::Boundary {
                        layouts,
                        bytes: body,
                    })
                } else if head {
                    let token =
                        u32::try_from(number(value, "token")?).map_err(|_| Error::Invalid)?;
                    if token >= self.approval.vocabulary || value["logit"].as_f64().is_none() {
                        return Err(Error::Invalid);
                    }
                    // A multi-row decode chunk checks drafts: each row's greedy token, the last
                    // one the chunk's own token.
                    let tokens = if decode && rows > 1 {
                        let tokens: Vec<u32> = value["tokens"]
                            .as_array()
                            .ok_or(Error::Invalid)?
                            .iter()
                            .map(|t| t.as_u64().and_then(|t| u32::try_from(t).ok()))
                            .collect::<Option<_>>()
                            .ok_or(Error::Invalid)?;
                        if tokens.len() != rows as usize
                            || tokens.last() != Some(&token)
                            || tokens.iter().any(|&t| t >= self.approval.vocabulary)
                        {
                            return Err(Error::Invalid);
                        }
                        tokens
                    } else {
                        vec![token]
                    };
                    Some(Output::Token {
                        token,
                        tokens,
                        logits: logits.then_some(body),
                    })
                } else {
                    None
                };
                Ok(Event::Chunk { step, output })
            }
            Pending::Status(_) | Pending::Cancel(_) => {
                // The reply is about the named session, or the earliest open one.
                let named = match &pending {
                    Pending::Status(Some(h)) | Pending::Cancel(h) => Some(h.as_str()),
                    _ => self.sessions.first().map(|s| s.handle.as_str()),
                };
                if handle(value, "incarnation")?
                    != *self.incarnation.as_ref().ok_or(Error::State)?
                    || optional_handle(value, "instance")? != self.instance.as_deref()
                    || optional_handle(value, "session")? != named
                {
                    return Err(Error::Conflict);
                }
                let named = named.map(str::to_owned);
                // The engine's open sessions are exactly ours: after an end, the ended one is
                // gone whichever was earliest (GLY-190). An engine of one session may not list.
                if let Some(open) = value.get("sessions") {
                    let mut theirs = open
                        .as_array()
                        .ok_or(Error::Invalid)?
                        .iter()
                        .map(|s| s["session"].as_str().ok_or(Error::Invalid))
                        .collect::<Result<Vec<_>>>()?;
                    let mut ours: Vec<_> =
                        self.sessions.iter().map(|s| s.handle.as_str()).collect();
                    theirs.sort_unstable();
                    ours.sort_unstable();
                    if theirs != ours {
                        return Err(Error::Conflict);
                    }
                }
                let state = value["state"].as_str().ok_or(Error::Invalid)?;
                if !matches!(
                    state,
                    "empty" | "idle" | "running" | "cancelling" | "cancelled"
                ) {
                    return Err(Error::Invalid);
                }
                let cancelled = value["session_cancelled"].as_bool().ok_or(Error::Invalid)?;
                if matches!(state, "cancelling" | "cancelled") && !cancelled {
                    return Err(Error::Invalid);
                }
                if let Some(named) = &named {
                    if number(value, "tokens")?
                        > u64::from(self.approval.boundary.key.workload.context_tokens.get())
                    {
                        return Err(Error::Invalid);
                    }
                    self.session_mut(named)?.cancelled |= cancelled;
                    // A running request is one of our sessions' last; the engine names its
                    // session (an engine of one session may not).
                    let running = value.get("running").ok_or(Error::Invalid)?;
                    if !running.is_null() {
                        let owner = match running.get("session") {
                            Some(_) => optional_handle(running, "session")?.unwrap_or(named),
                            None => named,
                        };
                        if running["op"] != "run_chunk"
                            || running["id"].as_str()
                                != self.session(owner)?.last_request.as_deref()
                        {
                            return Err(Error::Conflict);
                        }
                    }
                }
                if matches!(pending, Pending::Cancel(_)) {
                    if !matches!(state, "cancelling" | "cancelled") || !cancelled {
                        return Err(Error::State);
                    }
                    Ok(Event::Cancelled)
                } else {
                    Ok(Event::Status {
                        cancelling: matches!(state, "cancelling" | "cancelled"),
                    })
                }
            }
            Pending::End(handle) => {
                let session = self.session(&handle)?;
                let tokens = number(value, "tokens")?;
                if value["ended"] != true
                    || value["cancelled"].as_bool() != Some(session.cancelled)
                    || tokens > u64::from(self.approval.boundary.key.workload.context_tokens.get())
                    || !session.cancelled && tokens != u64::from(session.position)
                {
                    return Err(Error::Conflict);
                }
                self.sessions.retain(|s| s.handle != handle);
                Ok(Event::Ended { tokens })
            }
            Pending::Unload => {
                if value["unloaded"] != true {
                    return Err(Error::State);
                }
                let retained_bytes = number(value, "retained_bytes")?;
                if retained_bytes != 0 {
                    return Err(Error::State);
                }
                self.instance = None;
                self.allocation = None;
                self.committed = false;
                Ok(Event::Unloaded { retained_bytes })
            }
        }
    }
}
#[cfg(unix)]
impl<R: Read + std::os::fd::AsFd, W: Write> Worker<R, W> {
    /// Whether a reply is waiting, within `timeout`; it reads nothing. The owner can then cancel
    /// a chunk that is still running instead of blocking in `receive`. A closed pipe counts as
    /// ready, so the next `receive` fails.
    pub fn reply_ready(&self, timeout: std::time::Duration) -> Result<bool> {
        use rustix::event::{PollFd, PollFlags, Timespec, poll};
        let mut fds = [PollFd::new(&self.reader, PollFlags::IN)];
        let timeout = Timespec {
            tv_sec: timeout.as_secs().try_into().map_err(|_| Error::Invalid)?,
            tv_nsec: timeout.subsec_nanos().into(),
        };
        match poll(&mut fds, Some(&timeout)) {
            Ok(ready) => Ok(ready > 0),
            Err(rustix::io::Errno::INTR) => Ok(false),
            Err(_) => Err(Error::State),
        }
    }
}
fn number(value: &Value, key: &str) -> Result<u64> {
    value[key].as_u64().ok_or(Error::Invalid)
}
fn handle(value: &Value, key: &str) -> Result<String> {
    let s = value[key].as_str().ok_or(Error::Invalid)?;
    if s.is_empty() || s.len() > 256 || !s.bytes().all(|b| b.is_ascii_graphic()) {
        return Err(Error::Invalid);
    }
    Ok(s.to_owned())
}
fn optional_handle<'a>(value: &'a Value, key: &str) -> Result<Option<&'a str>> {
    let field = value.get(key).ok_or(Error::Invalid)?;
    if field.is_null() {
        Ok(None)
    } else {
        field.as_str().map(Some).ok_or(Error::Invalid)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Cursor, ErrorKind};

    fn boundary() -> BoundaryContract {
        let mut b = BoundaryContract::decode(
            include_bytes!("../../../../examples/partial-boundary.json"),
            FixtureMode::Simulation,
        )
        .unwrap();
        for patterns in [&mut b.inputs, &mut b.outputs] {
            patterns[0].role = TensorHeaderTensorRole::Residual;
            patterns[1].role = TensorHeaderTensorRole::Hidden;
            for p in patterns {
                p.shape = vec![
                    Dimension::Tokens,
                    Dimension::Fixed(BoundedU32::new(2).unwrap()),
                ];
            }
        }
        b
    }
    fn describe(b: &BoundaryContract) -> Value {
        json!({"protocol":"glyd-range/1","incarnation":"6649fc7bbb44e0f9","pid":123,
            "build":{"version":"synthetic"}, "model":{"vocab":16,"hidden":2},
            "stage":{"first":b.inputs[0].role == TensorHeaderTensorRole::TokenIds,"last":b.output_tokens == OutputTokens::SampledOne},"boundary":{"dtype":"bf16","bytes_per_row":8},
            "state":{"reset":"synthetic"},"ops":["describe","prepare","commit","begin_session","run_chunk","status","cancel","end_session","unload"],"device":0,
            "limits":{"sessions":1,"context_tokens":b.key.workload.context_tokens.get(),
                "chunk_tokens":b.key.workload.prefill_chunk_tokens.get(),"max_head_bytes":HEAD_LIMIT,"max_body_bytes":BODY_LIMIT}})
    }
    fn reply(id: &str, mut value: Value, body: Vec<u8>) -> (Value, Vec<u8>) {
        value["id"] = id.into();
        value["ok"] = true.into();
        (value, body)
    }
    fn frame(value: &Value, body: &[u8]) -> Vec<u8> {
        let head = serde_json::to_vec(value).unwrap();
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(head.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&(body.len() as u32).to_le_bytes());
        bytes.extend(head);
        bytes.extend(body);
        bytes
    }
    fn initial(b: &BoundaryContract) -> Vec<u8> {
        let d = describe(b);
        let frames = [
            reply("c1", d.clone(), vec![]),
            reply(
                "c2",
                json!({"instance":"opaque-instance","describe":"RTX 4080 SUPER (sm_89): 0..16 of 32 layers",
            "weights_bytes":100,"kv_bytes":20,"state_bytes":30,"scratch_bytes":40,"engine_bytes":190,"peak_bytes":200}),
                vec![],
            ),
            reply("c3", json!({"committed":true,"ready":true}), vec![]),
            reply(
                "c4",
                json!({"session":"opaque-session","context_tokens":b.key.workload.context_tokens.get(),"tokens":0,"kv_bytes":20,"state_bytes":30}),
                vec![],
            ),
        ];
        frames
            .into_iter()
            .flat_map(|(h, b)| frame(&h, &b))
            .collect()
    }
    fn worker(b: BoundaryContract, tail: Vec<u8>) -> Worker<Cursor<Vec<u8>>, Vec<u8>> {
        let mut bytes = initial(&b);
        bytes.extend(tail);
        Worker::new(
            Cursor::new(bytes),
            vec![],
            Approval {
                descriptor: descriptor_digest(&describe(&b)).unwrap(),
                engine_bytes_limit: 1000,
                payload_bytes_limit: 1024,
                first: b.inputs[0].role == TensorHeaderTensorRole::TokenIds,
                last: b.output_tokens == OutputTokens::SampledOne,
                vocabulary: 16,
                boundary: b,
            },
            FixtureMode::Simulation,
        )
        .unwrap()
    }
    fn call(b: &BoundaryContract, n: u32, tokens: u32) -> RunChunk {
        let h: starlings_protocol::TensorHeader = starlings_protocol::decode(
            include_bytes!("../../../../examples/tensor-header.json"),
            FixtureMode::Simulation,
        )
        .unwrap();
        let mut scope: DataScope = (&h).into();
        scope.invocation_id =
            serde_json::from_value(json!(format!("00000000-0000-4000-8000-{n:012x}"))).unwrap();
        scope.sequence = Decimal(n.into());
        RunChunk {
            version: Version,
            step: StepIdentity {
                scope,
                phase: if n < b.key.workload.prompt_tokens.get() {
                    TensorHeaderPhase::Prefill
                } else {
                    TensorHeaderPhase::Decode
                },
                position_start: BoundedU32::new(n).unwrap(),
                token_count: BoundedU32::new(tokens).unwrap(),
            },
            inputs: b.layouts(tokens, 1024).unwrap().0,
        }
    }
    fn open(w: &mut Worker<Cursor<Vec<u8>>, Vec<u8>>, owner: DataScope) {
        w.describe().unwrap();
        assert!(matches!(w.receive(), Ok(Event::Described)));
        w.prepare().unwrap();
        assert!(matches!(w.receive(), Ok(Event::Prepared(_))));
        w.commit().unwrap();
        assert!(matches!(w.receive(), Ok(Event::Committed)));
        let workload = w.approval.boundary.key.workload.clone();
        let (prompt, output) = (workload.prompt_tokens.get(), workload.output_tokens.get());
        for (p, o) in [
            (0, output),
            (prompt, 0),
            (prompt + 1, output),
            (prompt, output + 1),
        ] {
            assert_eq!(
                w.begin_session(owner.clone(), p, o, None),
                Err(Error::Capacity),
                "{p}+{o}"
            );
        }
        w.begin_session(owner, prompt, output, None).unwrap();
        assert!(matches!(w.receive(), Ok(Event::Begun(_))));
    }
    fn run_reply(id: &str, rows: u32, end: u32) -> Vec<u8> {
        let body = vec![0; rows as usize * 8];
        let (h, b) = reply(
            id,
            json!({"rows":rows,"position_end":end,"body":{"what":"boundary","rows":rows,"bytes":body.len()}}),
            body,
        );
        frame(&h, &b)
    }

    #[cfg(unix)]
    #[test]
    fn reply_ready_waits_for_a_frame_without_reading_it() {
        let b = boundary();
        let (reader, mut writer) = std::io::pipe().unwrap();
        let w = Worker::new(
            reader,
            vec![],
            Approval {
                descriptor: descriptor_digest(&describe(&b)).unwrap(),
                engine_bytes_limit: 1000,
                payload_bytes_limit: 1024,
                first: false,
                last: false,
                vocabulary: 16,
                boundary: b,
            },
            FixtureMode::Simulation,
        )
        .unwrap();
        let wait = std::time::Duration::from_millis(20);
        assert_eq!(w.reply_ready(wait), Ok(false));
        writer.write_all(b"x").unwrap();
        assert_eq!(w.reply_ready(wait), Ok(true));
        assert_eq!(w.reply_ready(wait), Ok(true), "nothing was read");
        drop(writer);
    }

    #[test]
    fn range_process_runs_only_the_approved_bytes() {
        // This test binary stands in for the engine (a copied system binary will not run on
        // macOS). File A is approved, then rewritten: the bytes that would run are checked.
        let temp = std::env::temp_dir().join(format!("pin-{}", uuid::Uuid::new_v4().simple()));
        std::fs::create_dir_all(&temp).unwrap();
        let (engine, library, private) = (temp.join("a"), temp.join("lib"), temp.join("private"));
        std::fs::copy(std::env::current_exe().unwrap(), &engine).unwrap();
        std::fs::write(&library, b"library bytes").unwrap();
        let approved = crate::service::executable_digest(&engine).unwrap();
        let library_digest = starlings_protocol::byte_digest(b"library bytes");
        let list = ["--list".to_owned()];
        let env: BTreeMap<_, _> = [(
            "GLYD_GPU_LIB".to_owned(),
            library.to_str().unwrap().to_owned(),
        )]
        .into();
        let launch = |library_digest: Option<&Digest>| {
            RangeProcess::spawn(&engine, &approved, &list, &env, library_digest, &private)
        };
        assert!(launch(None).is_err(), "a library needs its digest");
        assert!(
            RangeProcess::spawn(
                &engine,
                &starlings_protocol::byte_digest(b"other bytes"),
                &list,
                &BTreeMap::new(),
                None,
                &private
            )
            .is_err()
        );
        let (process, mut output, input) = launch(Some(&library_digest)).unwrap();
        drop(input);
        let mut listed = String::new();
        output.read_to_string(&mut listed).unwrap();
        assert!(
            listed.contains("range_process_runs_only_the_approved_bytes"),
            "{listed}"
        );
        // The child got a private copy of the library, not its path.
        let copies: Vec<_> = std::fs::read_dir(&private).unwrap().collect();
        assert_eq!(copies.len(), 1);
        let copy = copies[0].as_ref().unwrap().path().join("gpu-library");
        assert_eq!(std::fs::read(&copy).unwrap(), b"library bytes");
        let clone = process.clone();
        drop(process);
        clone.stop();
        drop(clone);
        assert_eq!(
            std::fs::read_dir(&private).unwrap().count(),
            0,
            "copies removed"
        );
        // After approval, file A and the library change: neither launch may run them.
        std::fs::write(&library, b"swapped library").unwrap();
        assert!(launch(Some(&library_digest)).is_err());
        std::fs::write(&library, b"library bytes").unwrap();
        std::fs::write(&engine, b"#!/bin/sh\nexit 0\n").unwrap();
        assert!(launch(Some(&library_digest)).is_err());
        let _ = std::fs::remove_dir_all(&temp);
    }

    #[test]
    fn each_session_is_bounded_by_its_own_counts_and_text_ops_map_ids() {
        let mut whole = boundary();
        let tokens = TensorPattern {
            role: TensorHeaderTensorRole::TokenIds,
            dtype: TensorHeaderDtype::U32,
            shape: vec![Dimension::Tokens],
        };
        whole.inputs = vec![tokens.clone()];
        whole.outputs = vec![tokens];
        whole.output_tokens = OutputTokens::SampledOne;
        let mut tail = vec![];
        for (id, value) in [
            ("c5", json!({"ids":[1,2]})),
            ("c6", json!({"text":"hi"})),
            (
                "c7",
                json!({"rows":2,"position_end":2,"token":5,"logit":0.5}),
            ),
            (
                "c8",
                json!({"rows":1,"position_end":3,"token":6,"logit":0.5}),
            ),
            ("c9", json!({"ids":[16]})),
        ] {
            let (h, p) = reply(id, value, vec![]);
            tail.extend(frame(&h, &p));
        }
        let mut w = worker(whole.clone(), tail);
        let first = call(&whole, 0, 2);
        w.describe().unwrap();
        assert!(matches!(w.receive(), Ok(Event::Described)));
        w.prepare().unwrap();
        assert!(matches!(w.receive(), Ok(Event::Prepared(_))));
        w.commit().unwrap();
        assert!(matches!(w.receive(), Ok(Event::Committed)));
        // The approved maxima are 4 + 4; this session asks for 2 + 2.
        let sampling = starlings_protocol::Sampling {
            profile: "sample-v1".into(),
            temperature: 0.5,
            top_p: 0.9,
            seed: BoundedU32::new(7).unwrap(),
        };
        let mut hot = sampling.clone();
        hot.temperature = 2.5;
        assert_eq!(
            w.begin_session(first.step.scope.clone(), 2, 2, Some(hot)),
            Err(Error::Invalid)
        );
        w.begin_session(first.step.scope.clone(), 2, 2, Some(sampling))
            .unwrap();
        assert!(matches!(w.receive(), Ok(Event::Begun(_))));
        w.tokenize("hello").unwrap();
        assert!(matches!(w.receive(), Ok(Event::Tokens(ids)) if ids == [1, 2]));
        assert!(w.detokenize(&[16]).is_err(), "id outside vocabulary");
        w.detokenize(&[1, 2]).unwrap();
        assert!(matches!(w.receive(), Ok(Event::Text(t)) if t == "hi"));
        let input = vec![[1u32.to_le_bytes(), 2u32.to_le_bytes()].concat()];
        assert!(
            w.run_chunk("opaque-session", &first, &first.step, &input, false, false)
                .is_err(),
            "the session's last prompt chunk samples"
        );
        w.run_chunk("opaque-session", &first, &first.step, &input, true, false)
            .unwrap();
        assert!(matches!(
            w.receive(),
            Ok(Event::Chunk {
                output: Some(Output::Token { token: 5, .. }),
                ..
            })
        ));
        let mut decode = call(&whole, 2, 1);
        decode.step.phase = TensorHeaderPhase::Decode;
        decode.step.scope.sequence = Decimal(1);
        w.run_chunk(
            "opaque-session",
            &decode,
            &decode.step,
            &[5u32.to_le_bytes().to_vec()],
            true,
            false,
        )
        .unwrap();
        assert!(matches!(w.receive(), Ok(Event::Chunk { .. })));
        let sent = String::from_utf8_lossy(&w.writer);
        assert!(
            sent.contains(r#""seed":7,"step":0,"temperature":0.5"#),
            "{sent}"
        );
        assert!(
            sent.contains(r#""seed":7,"step":1,"temperature":0.5"#),
            "{sent}"
        );
        let mut excess = call(&whole, 3, 1);
        excess.step.phase = TensorHeaderPhase::Decode;
        excess.step.scope.sequence = Decimal(2);
        assert!(
            w.run_chunk(
                "opaque-session",
                &excess,
                &excess.step,
                &[6u32.to_le_bytes().to_vec()],
                true,
                false
            )
            .is_err(),
            "two output tokens end this session below the approved four"
        );
        assert!(!w.is_ambiguous());
        w.tokenize("x").unwrap();
        assert!(w.receive().is_err(), "a returned id outside the vocabulary");
        assert!(w.is_ambiguous());
    }

    /// GLY-360: a multi-row decode chunk goes out as `verify` (and `head_rows` on the last
    /// stage); the next chunk may start only inside the verify chunk's rows, keeping the rows
    /// before it (`keep_previous`); a one-row chunk cannot be rewound; the last stage's reply
    /// must carry one token per row, ending with its own.
    #[cfg(unix)]
    #[test]
    fn drafted_chunks_verify_rewind_inside_their_rows_and_check_row_tokens() {
        let mut whole = boundary();
        let tokens = TensorPattern {
            role: TensorHeaderTensorRole::TokenIds,
            dtype: TensorHeaderDtype::U32,
            shape: vec![Dimension::Tokens],
        };
        whole.inputs = vec![tokens.clone()];
        whole.outputs = vec![tokens];
        whole.output_tokens = OutputTokens::SampledOne;
        let script = |replies: &[Value]| {
            let mut tail = vec![];
            for (i, value) in replies.iter().enumerate() {
                let (h, p) = reply(&format!("c{}", i + 5), value.clone(), vec![]);
                tail.extend(frame(&h, &p));
            }
            let mut w = worker(whole.clone(), tail);
            w.describe().unwrap();
            assert!(matches!(w.receive(), Ok(Event::Described)));
            w.prepare().unwrap();
            assert!(matches!(w.receive(), Ok(Event::Prepared(_))));
            w.commit().unwrap();
            assert!(matches!(w.receive(), Ok(Event::Committed)));
            w.begin_session(call(&whole, 0, 1).step.scope, 1, 4, None)
                .unwrap();
            assert!(matches!(w.receive(), Ok(Event::Begun(_))));
            w
        };
        // Step `sequence` from `start` with these token rows.
        let step = |sequence: u64, start: u32, rows: &[u32]| {
            let mut c = call(&whole, 0, rows.len() as u32);
            c.step.scope.invocation_id = serde_json::from_value(json!(format!(
                "00000000-0000-4000-8000-{:012x}",
                100 + sequence
            )))
            .unwrap();
            c.step.scope.sequence = Decimal(sequence);
            c.step.position_start = BoundedU32::new(start).unwrap();
            if start >= 1 {
                c.step.phase = TensorHeaderPhase::Decode;
            }
            let bytes: Vec<u8> = rows.iter().flat_map(|t| t.to_le_bytes()).collect();
            (c, vec![bytes])
        };
        let mut w = script(&[
            json!({"rows":1,"position_end":1,"token":5,"logit":0.5}),
            json!({"rows":2,"position_end":3,"token":9,"logit":0.5,"tokens":[7,9]}),
            json!({"rows":2,"position_end":4,"token":3,"logit":0.5,"tokens":[8,3]}),
            json!({"rows":1,"position_end":4,"token":4,"logit":0.5}),
        ]);
        let run = |w: &mut Worker<_, _>, (c, input): (RunChunk, Vec<Vec<u8>>)| {
            w.run_chunk("opaque-session", &c, &c.step, &input, true, false)
        };
        run(&mut w, step(0, 0, &[1])).unwrap();
        assert!(matches!(w.receive(), Ok(Event::Chunk { .. })));
        // [5, 7] at 1: a verify chunk; the last stage returns each row's token.
        run(&mut w, step(1, 1, &[5, 7])).unwrap();
        assert!(matches!(
            w.receive(),
            Ok(Event::Chunk { output: Some(Output::Token { token: 9, tokens, .. }), .. })
                if tokens == [7, 9]
        ));
        // The next chunk starts after the verify chunk's first row and up to its end.
        for start in [1, 4] {
            assert_eq!(
                run(&mut w, step(2, start, &[7, 8])),
                Err(Error::Conflict),
                "{start}"
            );
        }
        // 7 held, 9 did not: keep 1 row and run [7, 8] at 2.
        run(&mut w, step(2, 2, &[7, 8])).unwrap();
        assert!(matches!(w.receive(), Ok(Event::Chunk { .. })));
        // Keep 1 of [7, 8]: a one-row step at 3.
        run(&mut w, step(3, 3, &[8])).unwrap();
        assert!(matches!(w.receive(), Ok(Event::Chunk { .. })));
        // A one-row chunk cannot be rewound.
        assert_eq!(run(&mut w, step(4, 3, &[8])), Err(Error::Conflict));
        let sent = String::from_utf8_lossy(&w.writer);
        for expected in [
            r#""head_rows":2,"id":"c6""#,
            r#""id":"c7","keep_previous":1"#,
            r#""id":"c8","keep_previous":1"#,
        ] {
            assert!(sent.contains(expected), "{expected} in {sent}");
        }
        assert_eq!(sent.matches(r#""verify":true"#).count(), 2, "{sent}");
        // A verify reply must carry one token per row, ending with its own.
        for wrong in [json!([9]), json!([7, 8]), json!([7, 16])] {
            let mut w = script(&[
                json!({"rows":1,"position_end":1,"token":5,"logit":0.5}),
                json!({"rows":2,"position_end":3,"token":9,"logit":0.5,"tokens":wrong}),
            ]);
            run(&mut w, step(0, 0, &[1])).unwrap();
            w.receive().unwrap();
            run(&mut w, step(1, 1, &[5, 7])).unwrap();
            assert!(w.receive().is_err(), "{wrong}");
            assert!(w.is_ambiguous());
        }
    }

    #[test]
    fn borrowed_range_inputs_preserve_boxed_payloads_and_refuse_lengths_before_writes() {
        let b = boundary();
        let c = call(&b, 0, 2);
        let mut w = worker(b.clone(), run_reply("c5", 2, 2));
        open(&mut w, c.step.scope.clone());
        let inputs: [Box<[u8]>; 2] = [
            vec![0, 0, 0, 128, 128, 127, 193, 127].into_boxed_slice(),
            vec![255, 255, 1, 0, 0, 128, 85, 170].into_boxed_slice(),
        ];
        let before = w.writer.len();
        for bad in [
            vec![inputs[0].as_ref()],
            vec![inputs[0].as_ref(), &inputs[1][..7]],
            vec![inputs[0].as_ref(), b"ninebytes"],
        ] {
            assert_eq!(
                w.run_chunk("opaque-session", &c, &c.step, &bad, false, false),
                Err(Error::Invalid)
            );
            assert_eq!(w.writer.len(), before);
            assert!(!w.is_ambiguous());
        }
        w.run_chunk("opaque-session", &c, &c.step, &inputs, false, false)
            .unwrap();
        let body: Vec<u8> = inputs.iter().flat_map(|b| b.iter().copied()).collect();
        assert_eq!(
            &w.writer[before..],
            frame(
                &json!({"op":"run_chunk", "session":"opaque-session", "phase":"prefill",
                    "position_start":0, "head":false, "logits":false, "rows":2, "id":"c5"}),
                &body,
            )
        );
        match w.receive().unwrap() {
            Event::Chunk {
                step,
                output: Some(Output::Boundary { layouts, bytes }),
            } => {
                assert_eq!(step, c.step);
                assert_eq!(layouts, b.layouts(2, 1024).unwrap().1);
                assert_eq!(bytes, vec![0; 16]);
            }
            _ => panic!("missing complete ordered boundary"),
        }
    }

    #[test]
    fn owned_range_wire_maps_interleaving_cancellation_fences_and_bounded_failures() {
        let b = boundary();
        let c = call(&b, 0, 2);
        let mut w = worker(b.clone(), run_reply("c5", 2, 2));
        open(&mut w, c.step.scope.clone());
        let input = vec![vec![0; 8], vec![0; 8]];
        for field in [
            "epoch",
            "lease",
            "session",
            "generation",
            "plan",
            "sequence",
            "position",
            "invocation",
        ] {
            let mut bad = c.clone();
            match field {
                "epoch" => bad.step.scope.authority_epoch = Decimal(2),
                "lease" => {
                    bad.step.scope.lease_id =
                        serde_json::from_value(json!("00000000-0000-4000-8000-000000000999"))
                            .unwrap()
                }
                "session" => {
                    bad.step.scope.session_id =
                        serde_json::from_value(json!("00000000-0000-4000-8000-000000000999"))
                            .unwrap()
                }
                "generation" => {
                    bad.step.scope.generation_id =
                        serde_json::from_value(json!("00000000-0000-4000-8000-000000000999"))
                            .unwrap()
                }
                "plan" => {
                    bad.step.scope.plan_digest = starlings_protocol::byte_digest(b"another-plan")
                }
                "sequence" => bad.step.scope.sequence = Decimal(1),
                "position" => bad.step.position_start = BoundedU32::new(1).unwrap(),
                "invocation" => {
                    bad.step.scope.invocation_id =
                        serde_json::from_value(json!("00000000-0000-4000-8000-000000000999"))
                            .unwrap()
                }
                _ => unreachable!(),
            }
            assert!(
                w.run_chunk("opaque-session", &bad, &c.step, &input, false, false)
                    .is_err(),
                "{field}"
            );
            assert!(!w.is_ambiguous());
        }
        w.run_chunk("opaque-session", &c, &c.step, &input, false, false)
            .unwrap();
        match w.receive().unwrap() {
            Event::Chunk {
                step,
                output: Some(Output::Boundary { layouts, bytes }),
            } => {
                assert_eq!(step, c.step);
                assert_eq!(layouts, b.layouts(2, 1024).unwrap().1);
                assert_eq!(bytes, vec![0; 16]);
            }
            _ => panic!("missing complete ordered boundary"),
        }
        assert!(
            w.run_chunk("opaque-session", &c, &c.step, &input, false, false)
                .is_err()
        );

        // Status/cancel replies arrive before the executing chunk, without transferring drain authority.
        for succeeds in [false, true] {
            let mut tail = vec![];
            let status = |state: &str, cancelled: bool| {
                json!({"state":state,"incarnation":"6649fc7bbb44e0f9","instance":"opaque-instance", "session":"opaque-session",
                "session_cancelled":cancelled,"tokens":0,"running":{"id":"c5","op":"run_chunk","ms":1}})
            };
            let (h, p) = reply("c6", status("running", false), vec![]);
            tail.extend(frame(&h, &p));
            let (h, p) = reply("c7", status("cancelling", true), vec![]);
            tail.extend(frame(&h, &p));
            if succeeds {
                tail.extend(run_reply("c5", 2, 2));
            } else {
                tail.extend(frame(
                    &json!({"id":"c5","ok":false,"error":"cancelled during chunk"}),
                    &[],
                ));
            }
            let (h, p) = reply(
                "c8",
                json!({"ended":true,"tokens":2,"cancelled":true}),
                vec![],
            );
            tail.extend(frame(&h, &p));
            let (h, p) = reply("c9", json!({"unloaded":true,"retained_bytes":0}), vec![]);
            tail.extend(frame(&h, &p));
            let mut w = worker(b.clone(), tail);
            open(&mut w, c.step.scope.clone());
            w.run_chunk("opaque-session", &c, &c.step, &input, false, false)
                .unwrap();
            w.status(None).unwrap();
            w.cancel("opaque-session").unwrap();
            assert!(matches!(
                w.receive(),
                Ok(Event::Status { cancelling: false })
            ));
            assert!(matches!(w.receive(), Ok(Event::Cancelled)));
            assert!(matches!(w.receive(), Ok(Event::Withdrawn(_))));
            assert!(
                w.run_chunk("opaque-session", &c, &c.step, &input, false, false)
                    .is_err()
            );
            w.end_session("opaque-session").unwrap();
            assert!(matches!(w.receive(), Ok(Event::Ended { tokens: 2 })));
            w.unload().unwrap();
            assert!(matches!(
                w.receive(),
                Ok(Event::Unloaded { retained_bytes: 0 })
            ));
        }

        for kind in [
            "oversized",
            "wrong-id",
            "wrong-body",
            "duplicate-json",
            "truncated",
            "reincarnated",
        ] {
            let tail = match kind {
                "oversized" => [1u32.to_le_bytes(), (BODY_LIMIT as u32 + 1).to_le_bytes()].concat(),
                "wrong-id" => run_reply("foreign", 2, 2),
                "wrong-body" => frame(&json!({"id":"c5","ok":true}), &[0]),
                "duplicate-json" => {
                    let head = br#"{"id":"c5","ok":true,"ok":true}"#;
                    let mut out = [(head.len() as u32).to_le_bytes(), 16u32.to_le_bytes()].concat();
                    out.extend(head);
                    out
                }
                "truncated" => {
                    let mut bytes = run_reply("c5", 2, 2);
                    bytes.pop();
                    bytes
                }
                "reincarnated" => {
                    let (h, p) = reply(
                        "c5",
                        json!({"state":"idle","incarnation":"6649fc7bbb44e0fa","instance":"opaque-instance", "session":"opaque-session",
                        "session_cancelled":false,"tokens":0,"running":null}),
                        vec![],
                    );
                    frame(&h, &p)
                }
                _ => unreachable!(),
            };
            let mut w = worker(b.clone(), tail);
            open(&mut w, c.step.scope.clone());
            if kind == "reincarnated" {
                w.status(None).unwrap();
            } else {
                w.run_chunk("opaque-session", &c, &c.step, &input, false, false)
                    .unwrap();
            }
            assert!(w.receive().is_err(), "{kind}");
            assert!(w.is_ambiguous());
            assert!(
                w.run_chunk("opaque-session", &c, &c.step, &input, false, false)
                    .is_err()
            );
            if kind == "oversized" {
                assert_eq!(w.reader.position(), initial(&b).len() as u64 + 8);
            }
        }

        struct FailedWrite;
        impl Write for FailedWrite {
            fn write(&mut self, _: &[u8]) -> std::io::Result<usize> {
                Err(ErrorKind::BrokenPipe.into())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        let mut w = Worker::new(
            Cursor::new(vec![]),
            FailedWrite,
            Approval {
                descriptor: descriptor_digest(&describe(&b)).unwrap(),
                boundary: b.clone(),
                engine_bytes_limit: 1000,
                payload_bytes_limit: 1024,
                first: false,
                last: false,
                vocabulary: 16,
            },
            FixtureMode::Simulation,
        )
        .unwrap();
        assert!(w.describe().is_err());
        assert!(w.is_ambiguous());
        assert!(w.describe().is_err());

        // The final prompt chunk already samples the first output token.
        for output_tokens in [1, 3] {
            let mut whole = b.clone();
            whole.key.workload.output_tokens = BoundedU32::new(output_tokens).unwrap();
            let tokens = TensorPattern {
                role: TensorHeaderTensorRole::TokenIds,
                dtype: TensorHeaderDtype::U32,
                shape: vec![Dimension::Tokens],
            };
            whole.inputs = vec![tokens.clone()];
            whole.outputs = vec![tokens];
            whole.output_tokens = OutputTokens::SampledOne;
            let mut tail = vec![];
            let (h, p) = reply("c5", json!({"rows":2,"position_end":2}), vec![]);
            tail.extend(frame(&h, &p));
            let (h, p) = reply(
                "c6",
                json!({"rows":2,"position_end":4,"token":5,"logit":0.25}),
                vec![0; 32],
            );
            tail.extend(frame(&h, &p));
            for n in 0..output_tokens - 1 {
                let (h, p) = reply(
                    &format!("c{}", n + 7),
                    json!({"rows":1,"position_end":n+5,"token":6,"logit":0.5}),
                    vec![],
                );
                tail.extend(frame(&h, &p));
            }
            let mut w = worker(whole.clone(), tail);
            open(&mut w, call(&whole, 0, 2).step.scope);
            let input = vec![[1u32.to_le_bytes(), 2u32.to_le_bytes()].concat()];
            let c = call(&whole, 0, 2);
            assert!(
                w.run_chunk("opaque-session", &c, &c.step, &input, true, false)
                    .is_err(),
                "early prompt sampling"
            );
            w.run_chunk("opaque-session", &c, &c.step, &input, false, false)
                .unwrap();
            assert!(matches!(w.receive(), Ok(Event::Chunk { output: None, .. })));
            let mut c = call(&whole, 2, 2);
            c.step.scope.sequence = Decimal(1);
            assert!(
                w.run_chunk("opaque-session", &c, &c.step, &input, false, false)
                    .is_err(),
                "missing final prompt sampling"
            );
            w.run_chunk("opaque-session", &c, &c.step, &input, true, true)
                .unwrap();
            assert!(matches!(
                w.receive(),
                Ok(Event::Chunk {
                    output: Some(Output::Token {
                        token: 5,
                        logits: Some(_),
                        ..
                    }),
                    ..
                })
            ));
            for n in 0..output_tokens - 1 {
                let mut c = call(&whole, n + 4, 1);
                c.step.scope.sequence = Decimal(u64::from(n + 2));
                assert!(
                    w.run_chunk(
                        "opaque-session",
                        &c,
                        &c.step,
                        &[16u32.to_le_bytes().to_vec()],
                        true,
                        false
                    )
                    .is_err(),
                    "token outside vocabulary"
                );
                w.run_chunk(
                    "opaque-session",
                    &c,
                    &c.step,
                    &[5u32.to_le_bytes().to_vec()],
                    true,
                    false,
                )
                .unwrap();
                assert!(matches!(
                    w.receive(),
                    Ok(Event::Chunk {
                        output: Some(Output::Token {
                            token: 6,
                            logits: None,
                            ..
                        }),
                        ..
                    })
                ));
            }
            let mut c = call(&whole, output_tokens + 3, 1);
            c.step.scope.sequence = Decimal(u64::from(output_tokens + 1));
            assert!(
                w.run_chunk(
                    "opaque-session",
                    &c,
                    &c.step,
                    &[5u32.to_le_bytes().to_vec()],
                    true,
                    false
                )
                .is_err(),
                "excess output after final prompt token"
            );
            assert!(!w.is_ambiguous());
            assert!(String::from_utf8_lossy(&w.writer).contains("\"tokens\":[1,2]"));
        }

        // Provider's bounded reply history must not become a 256-token/session ceiling in Compute.
        let mut b = b;
        b.key.workload.context_tokens = BoundedU32::new(302).unwrap();
        b.key.workload.prompt_tokens = BoundedU32::new(1).unwrap();
        b.key.workload.output_tokens = BoundedU32::new(301).unwrap();
        b.key.workload.prefill_chunk_tokens = BoundedU32::new(1).unwrap();
        let tail = (0..301)
            .flat_map(|n| run_reply(&format!("c{}", n + 5), 1, n + 1))
            .collect();
        let mut w = worker(b.clone(), tail);
        open(&mut w, call(&b, 0, 1).step.scope);
        for n in 0..301 {
            let c = call(&b, n, 1);
            w.run_chunk(
                "opaque-session",
                &c,
                &c.step,
                &[vec![0; 4], vec![0; 4]],
                false,
                false,
            )
            .unwrap();
            assert!(matches!(w.receive(), Ok(Event::Chunk { .. })));
        }
        let c = call(&b, 301, 1);
        assert!(
            w.run_chunk(
                "opaque-session",
                &c,
                &c.step,
                &[vec![0; 4], vec![0; 4]],
                false,
                false
            )
            .is_err()
        );
    }
}
