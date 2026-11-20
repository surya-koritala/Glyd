//! GLY-188: one chat request through two warm `glyd-engine rangeworker` stages held by enrolled
//! worker daemons. The coordinator owns the client, the per-request plan, leases and grants;
//! the first stage tokenizes, embeds and returns text; the last stage samples. Boundaries and
//! tokens cross enrolled TLS directly between the stages under their delegated grants.
use crate::routes::{PipelineRoute, StageRoute};
use serde::{Deserialize, Serialize};
use starlings_adapters::partial::{
    self, BoundaryContract, RunChunk, StepIdentity,
    glyd::{self, Event, Output},
};
use starlings_protocol::{
    BoundedU32, ChatMessageRole, ChatRequest, ControlRequest, DATA_PAYLOAD_LIMIT, DataControl,
    DataScope, Decimal, Digest, FixtureMode, GenerationId, Lease, LeaseId, MAX_DECODE_ROWS,
    Operation, PlacementPlan, ProfileWorkload, RequestId, Sampling, Session, SessionId,
    SessionState, TensorHeader, TensorHeaderDtype, TensorHeaderPhase, TensorHeaderTensorRole,
    Version,
};
use starlings_security::{
    AuthorityEpoch, ClientAdmission, ClientResponse, ClientWatch, DataAdmission, DataStream,
    EnrollmentReader, Error, FlowReceiver, GenerationBinding, GenerationStream, Identity,
    IncomingClient, Invitation, PeerListener, PeerStream, ReceiveBudget, ReceivedTensors,
    RegistryFence, StageBackend, StageChunk, StageGrant, StageIssuer, StageWorker, step_invocation,
};
use std::{
    net::{SocketAddr, TcpStream},
    process::{ChildStdin, ChildStdout},
    str::FromStr,
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    thread,
    time::{Duration, Instant},
};

mod draft;

type Result<T> = std::result::Result<T, Error>;
/// A stage answers a poll within this; the coordinator's record wait is two seconds.
const POLL_WAIT: Duration = Duration::from_millis(150);
/// Grants last two seconds from the request that renewed them.
const RENEW: Duration = Duration::from_millis(900);
const DATA_WAIT: Duration = Duration::from_secs(10);
const PAYLOAD: u64 = 128 << 20;
const TEXT: usize = 8 * 1024;
/// Load bound for the engine; past it, or on stop, the watchdog kills the engine.
// ponytail: a fixed bound; a 27B load from a cold disk may need a route field.
const LOAD_LIMIT: Duration = Duration::from_secs(600);
/// One request ends inside its lease (at most 60 s); this bounds every engine call in it.
const REQUEST_LIMIT: Duration = Duration::from_secs(75);
/// After the control side fails, steps get at least this long (or four times the longest
/// chunk so far) to unwind before the engine is killed.
const GRACE: Duration = Duration::from_secs(3);
/// After a failed request, the coordinator's slot waits this long for the stages to unwind.
const TEARDOWN: Duration = Duration::from_secs(5);

#[derive(Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum Command {
    Count {
        request: Box<ChatRequest>,
        /// Return the prompt's token IDs too: only for a drafted request (GLY-360).
        ids: bool,
    },
    Apply {
        request: Box<ControlRequest>,
    },
    Start {
        leases: Vec<LeaseId>,
        data: Vec<SocketAddr>,
        chunk: u32,
        stop: Vec<u32>,
        /// GLY-360: drafts a greedy decode step may carry, and the prompt's token IDs, which
        /// both stages draft from.
        drafts: u32,
        prompt: Vec<u32>,
        /// GLY-190: the session slot (data listener) this request uses on every stage.
        slot: u32,
    },
    Poll,
}
#[derive(Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum Reply {
    Counted {
        request: Digest,
        tokens: u32,
        /// The prompt's token IDs, for the last stage's drafts (GLY-360).
        ids: Vec<u32>,
    },
    Rejected,
    Applied {
        request: RequestId,
    },
    Pending,
    /// The stage renews both route grants now: two challenges follow.
    Renew,
    Text {
        text: String,
    },
    Finished {
        report: Box<Report>,
    },
}

/// What one stage executed for one request, and the engine's own account of the session it
/// ended. Token IDs stay on the stages; only their counts and digests are reported.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Report {
    pub layers: [u32; 2],
    pub first: bool,
    pub last: bool,
    pub prefill_steps: u32,
    pub decode_steps: u32,
    pub rows: u32,
    pub prompt_tokens: u32,
    pub prompt_sha256: Option<Digest>,
    pub tokens: u32,
    pub tokens_sha256: Digest,
    pub stopped: bool,
    pub engine_ms: f64,
    /// `end_session`'s count of the positions the ended session held.
    pub ended_tokens: u64,
    /// A `status` after `end_session` named no open session.
    pub session_closed: bool,
    /// Data connections dropped while waiting for the adjacent stage.
    pub refused_connections: u32,
    /// GLY-360: drafts sent with decode steps, and how many of them held.
    pub drafted: u32,
    pub accepted: u32,
}

/// The step plan both stages derive from the session: the prompt in chunks, then one token
/// at a time. Step k has sequence k and the derived invocation ID on every stage.
struct Steps {
    plan: Digest,
    epoch: Decimal,
    session: SessionId,
    generation: GenerationId,
    boundary_profile: String,
    leases: [LeaseId; 2],
    hidden: u32,
    prompt: u32,
    chunk: u32,
}
impl Steps {
    fn prefill(&self) -> u64 {
        self.prompt.div_ceil(self.chunk).into()
    }
    /// Prompt chunk k's input (k < prefill()).
    fn header(&self, k: u64, tokens: bool) -> Result<TensorHeader> {
        let position = u32::try_from(k)
            .ok()
            .and_then(|k| k.checked_mul(self.chunk))
            .filter(|&p| p < self.prompt)
            .ok_or(Error::Invalid)?;
        let rows = self.chunk.min(self.prompt - position);
        self.header_at(k, TensorHeaderPhase::Prefill, position, rows, tokens)
    }
    /// Step k's input: token IDs into the first stage (a prompt chunk, or the last stage's
    /// tokens for the step before), the bf16 residual/MLP-output pair, as one
    /// [2, rows, hidden] tensor, into the last.
    fn header_at(
        &self,
        k: u64,
        phase: TensorHeaderPhase,
        position: u32,
        rows: u32,
        tokens: bool,
    ) -> Result<TensorHeader> {
        let bounded = |n: u32| BoundedU32::new(n).map_err(|_| Error::Invalid);
        let (role, dtype, shape) = if tokens {
            (
                TensorHeaderTensorRole::TokenIds,
                TensorHeaderDtype::U32,
                vec![bounded(rows)?],
            )
        } else {
            (
                TensorHeaderTensorRole::Hidden,
                TensorHeaderDtype::Bf16,
                vec![bounded(2)?, bounded(rows)?, bounded(self.hidden)?],
            )
        };
        let total = if tokens {
            4 * u64::from(rows)
        } else {
            4 * u64::from(rows) * u64::from(self.hidden)
        };
        let chunk = total.min(DATA_PAYLOAD_LIMIT as u64);
        let header = TensorHeader {
            kind: "TensorChunk".into(),
            protocol_version: Version,
            lease_id: self.leases[usize::from(!tokens)],
            plan_digest: self.plan.clone(),
            authority_epoch: self.epoch,
            session_id: self.session,
            generation_id: self.generation,
            invocation_id: step_invocation(self.generation, k)?,
            sequence: Decimal(k),
            source_stage: BoundedU32::new(u32::from(tokens)).map_err(|_| Error::Invalid)?,
            destination_stage: BoundedU32::new(u32::from(!tokens)).map_err(|_| Error::Invalid)?,
            phase,
            position_start: BoundedU32::new(position).map_err(|_| Error::Invalid)?,
            token_count: bounded(rows)?,
            boundary_profile: self.boundary_profile.clone(),
            tensor_index: BoundedU32::new(0).map_err(|_| Error::Invalid)?,
            tensor_count: BoundedU32::new(1).map_err(|_| Error::Invalid)?,
            tensor_role: role,
            dtype,
            shape,
            total_tensor_bytes: Decimal(total),
            chunk_offset_bytes: Decimal(0),
            chunk_bytes: Decimal(chunk),
            end_of_tensor: chunk == total,
        };
        starlings_protocol::encode(&header, FixtureMode::Reject)?;
        Ok(header)
    }
}

/// Name the failed step on stderr; errors carry no request data.
fn at<T>(step: &'static str, result: Result<T>) -> Result<T> {
    result.inspect_err(|error| eprintln!("PIPELINE_STEP_FAILED step={step} reason={error:?}"))
}
fn engine_error(error: partial::Error) -> Error {
    match error {
        partial::Error::Invalid => Error::Invalid,
        partial::Error::Capacity => Error::Capacity,
        _ => Error::Transport,
    }
}
fn bounded<const MIN: u32, const MAX: u32>(n: u32) -> Result<BoundedU32<MIN, MAX>> {
    BoundedU32::new(n).map_err(|_| Error::Invalid)
}
fn id<T: FromStr>() -> Result<T> {
    uuid::Uuid::new_v4()
        .to_string()
        .parse()
        .map_err(|_| Error::Invalid)
}
/// Run `work` while a watchdog kills the engine's process group if `deadline` passes or `stop`
/// is set: the engine's pipes close, so a call blocked on them fails and `work` returns.
/// The second value says whether the engine was killed.
/// When a request's current engine call began (GLY-190). The engine is shared, so a watchdog
/// stops it only for a call of this request's that runs too long, never for its wait on another
/// request's call.
struct Calls {
    base: Instant,
    /// ms since `base` + 1 when the call began; 0 between calls.
    since: std::sync::atomic::AtomicU64,
}
struct Call<'a>(&'a Calls);
impl Drop for Call<'_> {
    fn drop(&mut self) {
        self.0.since.store(0, Ordering::Release);
    }
}
impl Calls {
    fn new() -> Self {
        Self {
            base: Instant::now(),
            since: Default::default(),
        }
    }
    /// Held while this request's call (and its reply) is on the engine.
    fn begin(&self) -> Call<'_> {
        let now = self.base.elapsed().as_millis() as u64;
        self.since.store(now + 1, Ordering::Release);
        Call(self)
    }
    /// How long this request's current call has run, if one is running.
    fn running_for(&self) -> Option<Duration> {
        let since = self.since.load(Ordering::Acquire).checked_sub(1)?;
        Some(
            self.base
                .elapsed()
                .saturating_sub(Duration::from_millis(since)),
        )
    }
}
/// Run `work`; past `deadline` (while this request is in an engine call, with `calls`) or on a
/// stop, stop the engine.
fn watched<T>(
    process: &glyd::RangeProcess,
    deadline: Instant,
    stop: &AtomicBool,
    calls: Option<&Calls>,
    work: impl FnOnce() -> T,
) -> (T, bool) {
    let done = AtomicBool::new(false);
    thread::scope(|threads| {
        let watchdog = threads.spawn(|| {
            while !done.load(Ordering::Acquire) {
                let calling = calls.is_none_or(|c| c.running_for().is_some());
                if Instant::now() >= deadline && calling || stop.load(Ordering::Relaxed) {
                    process.stop();
                    return true;
                }
                thread::sleep(Duration::from_millis(10));
            }
            false
        });
        let out = work();
        done.store(true, Ordering::Release);
        (out, watchdog.join().unwrap_or(true))
    })
}
fn digest_ids(ids: &[u32]) -> Digest {
    starlings_protocol::byte_digest(&ids.iter().flat_map(|t| t.to_le_bytes()).collect::<Vec<_>>())
}

/// What a stage's request needs from its daemon.
pub struct Stage<'a> {
    pub route: &'a StageRoute,
    pub identity: &'a Identity,
    pub pin: &'a Invitation,
    pub fence: &'a RegistryFence,
    /// The stage's one runtime for this epoch, shared by its requests (GLY-190).
    pub runtime: &'a starlings_security::StageRuntime,
    /// One data listener per session slot (GLY-190), with its address and whether a request
    /// holds it: a request uses the slot whose address the coordinator gave it.
    pub data: &'a [(SocketAddr, PeerListener, AtomicBool)],
    /// The daemon is stopping: end the request and kill the engine.
    pub stop: &'a AtomicBool,
}
type EngineWorker = glyd::Worker<ChildStdout, ChildStdin>;
/// The stage's one engine; requests take turns, one call and its reply at a time.
fn lock(
    worker: &std::sync::Mutex<EngineWorker>,
) -> Result<std::sync::MutexGuard<'_, EngineWorker>> {
    worker.lock().map_err(|_| Error::Invalid)
}

/// A loaded, committed stage kept across requests; each request opens and ends one session.
/// Any ambiguous engine reply ends this owner: nothing is retried or reattached.
pub struct Warm {
    worker: std::sync::Mutex<EngineWorker>,
    process: glyd::RangeProcess,
    contract: BoundaryContract,
    first: bool,
    last: bool,
    hidden: u32,
    vocabulary: u32,
    manifest: Digest,
    /// The engine's device report at prepare: name, free and total bytes.
    pub device: Option<(String, u64, u64)>,
    pub family: String,
    broken: AtomicBool,
}
impl Warm {
    /// Launch the approved engine, compare its `describe` with the approved digest, then load
    /// and commit the stage. A load past LOAD_LIMIT, or a stop, kills the engine.
    /// `private` is a daemon-owned directory for the engine's checked copies.
    pub fn launch(
        route: &StageRoute,
        private: &std::path::Path,
        stop: &AtomicBool,
    ) -> Result<Self> {
        crate::model_files::verify(&route.model, &route.model_files_sha256)
            .map_err(|_| Error::Denied)?;
        let (process, stdout, stdin) = glyd::RangeProcess::spawn(
            &route.executable,
            &route.executable_sha256,
            &route.arguments(),
            &route.environment,
            route.gpu_library_sha256.as_ref(),
            private,
        )
        .map_err(engine_error)?;
        let (loaded, killed) = watched(&process, Instant::now() + LOAD_LIMIT, stop, None, || {
            Self::load(route, process.clone(), stdout, stdin)
        });
        if killed {
            return Err(Error::Expired);
        }
        loaded
    }
    fn load(
        route: &StageRoute,
        process: glyd::RangeProcess,
        mut stdout: ChildStdout,
        mut stdin: ChildStdin,
    ) -> Result<Self> {
        let d = glyd::describe(&mut stdout, &mut stdin).map_err(engine_error)?;
        let observed = glyd::descriptor_digest(&d).map_err(engine_error)?;
        if observed != route.descriptor {
            eprintln!("STAGE_DESCRIPTOR_UNAPPROVED observed={observed}");
            return Err(Error::Denied);
        }
        if d["stage"]["layers"] != serde_json::json!(route.layers) {
            return Err(Error::Denied);
        }
        let workload = ProfileWorkload {
            context_tokens: bounded(route.context_tokens)?,
            prompt_tokens: bounded(route.max_prompt_tokens)?,
            output_tokens: bounded(route.max_output_tokens)?,
            prefill_chunk_tokens: bounded(route.chunk_tokens)?,
            sessions: bounded(route.sessions)?,
        };
        let (mut contract, first, last, vocabulary) = glyd::qwen_boundary(
            &d,
            &route.executable_sha256,
            &route.execution_profile,
            &route.boundary_profile,
            workload,
        )
        .map_err(engine_error)?;
        contract.key.manifest_digest = route.model_files_sha256.clone();
        let hidden = d["model"]["hidden"]
            .as_u64()
            .and_then(|n| u32::try_from(n).ok())
            .ok_or(Error::Invalid)?;
        let family = d["model"]["family"].as_str().unwrap_or("").to_owned();
        let mut worker = glyd::Worker::new(
            stdout,
            stdin,
            glyd::Approval {
                descriptor: route.descriptor.clone(),
                boundary: contract.clone(),
                engine_bytes_limit: route.engine_bytes_limit.0,
                payload_bytes_limit: PAYLOAD,
                first,
                last,
                vocabulary,
            },
            FixtureMode::Reject,
        )
        .map_err(engine_error)?;
        worker.describe().map_err(engine_error)?;
        let Event::Described = worker.receive().map_err(engine_error)? else {
            return Err(Error::Invalid);
        };
        worker.prepare().map_err(engine_error)?;
        let Event::Prepared(allocation) = worker.receive().map_err(engine_error)? else {
            return Err(Error::Invalid);
        };
        worker.commit().map_err(engine_error)?;
        let Event::Committed = worker.receive().map_err(engine_error)? else {
            return Err(Error::Invalid);
        };
        if first {
            // The first tokenize loads the tokenizer; keep that out of a request's Count.
            worker.tokenize("").map_err(engine_error)?;
            let Event::Tokens(_) = worker.receive().map_err(engine_error)? else {
                return Err(Error::Invalid);
            };
        }
        crate::model_files::verify(&route.model, &route.model_files_sha256)
            .map_err(|_| Error::Denied)?;
        Ok(Self {
            manifest: contract.key.manifest_digest.clone(),
            worker: std::sync::Mutex::new(worker),
            process,
            contract,
            first,
            last,
            hidden,
            vocabulary,
            device: allocation.device,
            family,
            broken: AtomicBool::new(false),
        })
    }
    /// The model the stage serves is bound to its verified checkpoint contents.
    pub fn binding(&self, route: &StageRoute, epoch: AuthorityEpoch) -> GenerationBinding {
        GenerationBinding {
            version: Version,
            epoch,
            model: self.manifest.clone(),
            profile: route.execution_profile.clone(),
        }
    }
    pub fn usable(&self) -> bool {
        !self.broken.load(Ordering::Acquire) && self.process.alive()
    }
    /// One request on one coordinator connection, every engine call in it bounded by
    /// REQUEST_LIMIT and the daemon's stop. An engine reply that leaves its state unknown, or a
    /// killed engine, makes the stage unusable until an operator restarts the daemon.
    pub fn serve(&self, stage: &Stage<'_>, mut channel: GenerationStream) -> Result<Report> {
        let process = self.process.clone();
        let calls = Calls::new();
        let (result, killed) = watched(
            &process,
            Instant::now() + REQUEST_LIMIT,
            stage.stop,
            Some(&calls),
            || self.request(stage, &mut channel, &calls),
        );
        if result.is_err() {
            channel.watch().cancel();
        }
        // A poisoned engine lock is as unknown as an ambiguous reply.
        if killed || lock(&self.worker).map_or(true, |w| w.is_ambiguous()) {
            self.broken.store(true, Ordering::Release);
            self.process.stop();
        }
        result
    }
    fn count(
        &self,
        route: &StageRoute,
        chat: &ChatRequest,
        model: &Digest,
        calls: &Calls,
    ) -> Result<Vec<u32>> {
        starlings_protocol::encode(chat, FixtureMode::Reject)?;
        // The chat's system, user and assistant turns, the last the user's, through the model's
        // template (GLY-361).
        if chat.model != model.as_str()
            || !(1..=256).contains(&chat.messages.len())
            || chat.messages.last().map(|m| m.role) != Some(ChatMessageRole::User)
            || chat.max_tokens.get() > route.max_output_tokens
        {
            return Err(Error::Invalid);
        }
        let turns: Vec<(&str, &str)> = chat
            .messages
            .iter()
            .map(|m| {
                let role = match m.role {
                    ChatMessageRole::System => "system",
                    ChatMessageRole::User => "user",
                    ChatMessageRole::Assistant => "assistant",
                };
                (role, m.content.as_str())
            })
            .collect();
        let mut worker = lock(&self.worker)?;
        let _call = calls.begin();
        worker.tokenize_chat(&turns).map_err(engine_error)?;
        let Event::Tokens(ids) = worker.receive().map_err(engine_error)? else {
            return Err(Error::Invalid);
        };
        drop(worker);
        if ids.len() > route.max_prompt_tokens as usize
            || ids.len() as u64 + u64::from(chat.max_tokens.get()) > u64::from(route.context_tokens)
        {
            return Err(Error::Capacity);
        }
        Ok(ids)
    }
    fn request(
        &self,
        stage: &Stage<'_>,
        channel: &mut GenerationStream,
        calls: &Calls,
    ) -> Result<Report> {
        let route = stage.route;
        let binding = channel.binding().clone();
        if binding.model != self.manifest || binding.profile != route.execution_profile {
            return Err(Error::Denied);
        }
        // Count is read-only: the first stage tokenizes with the model's chat template.
        let prompt = if self.first {
            let Command::Count { request, ids: send } = channel.receive_record()? else {
                return Err(Error::Invalid);
            };
            match self.count(route, &request, &binding.model, calls) {
                Ok(ids) => {
                    channel.send_record(&Reply::Counted {
                        request: starlings_protocol::content_hash(request.as_ref())?,
                        tokens: ids.len() as u32,
                        ids: if send { ids.clone() } else { Vec::new() },
                    })?;
                    ids
                }
                Err(error) => {
                    let _ = channel.send_record(&Reply::Rejected);
                    return Err(error);
                }
            }
        } else {
            Vec::new()
        };
        let node = stage.identity.node();
        // The stage's runtime is for one authority epoch; the request's must be it.
        if stage.runtime.epoch()? != binding.epoch.number.0 {
            return Err(Error::Expired);
        }
        let owner = at("assign", StageWorker::assign(stage.runtime, channel))?;
        let index = usize::from(!self.first);
        let plan = owner.plan().clone();
        let local = plan.stages.get(index).ok_or(Error::Denied)?;
        if plan.stages.len() != 2
            || local.node_id != node
            || [local.layer_start.get(), local.layer_end_exclusive.get()] != route.layers
            || local.backend_build_digest != route.executable_sha256
            || local.backend_id != route.backend_id
            || local.device_id != route.device_id
            || local.memory_pool_id != route.device_id
            || plan.boundary_profile != route.boundary_profile
            || plan.context_limit.get() > route.context_tokens
        {
            return Err(Error::Denied);
        }
        let (mut lease, mut session, mut lease_end) = (None, None, None);
        let hold = Hold::default();
        for phase in 0..3 {
            let Command::Apply { request } = channel.receive_record()? else {
                return Err(Error::Invalid);
            };
            match (phase, &request.operation) {
                (0, Operation::Prepare { lease: l, .. }) => {
                    lease = Some(l.lease_id);
                    lease_end =
                        Some(Instant::now() + Duration::from_millis(l.lifetime_ms.get().into()));
                }
                (1, Operation::Commit { lease_id }) if Some(*lease_id) == lease => {}
                (2, Operation::Begin { session: s }) => {
                    let (p, o) = (s.prompt_token_count.get(), s.max_output_tokens.get());
                    if self.first && p as usize != prompt.len()
                        || p > route.max_prompt_tokens
                        || o > route.max_output_tokens
                        || p + o > route.context_tokens
                    {
                        return Err(Error::Denied);
                    }
                    session = Some(s.as_ref().clone());
                }
                _ => return Err(Error::Invalid),
            }
            if at("stage_apply", owner.apply(channel, &request))? {
                return Err(Error::Replay);
            }
            if phase == 0 {
                // Retain the stage reservation before any engine work for this lease.
                let lease = lease.ok_or(Error::Invalid)?;
                let held = at("hold_backend", owner.hold_backend(channel, lease))?;
                hold.backend.set(held).map_err(|_| Error::Invalid)?;
            }
            channel.send_record(&Reply::Applied {
                request: request.request_id,
            })?;
        }
        let (lease, backend, session, lease_end) = (
            lease.ok_or(Error::Invalid)?,
            hold.backend.get().ok_or(Error::Invalid)?,
            session.ok_or(Error::Invalid)?,
            lease_end.ok_or(Error::Invalid)?,
        );
        let Command::Start {
            leases,
            data,
            chunk,
            stop,
            drafts,
            prompt: ids,
            slot,
        } = channel.receive_record()?
        else {
            return Err(Error::Invalid);
        };
        // A drafted request (greedy only: the engine draws only a chunk's last row) brings the
        // prompt the first stage counted, which the last drafts from and the first checks;
        // any other brings none.
        if drafts >= MAX_DECODE_ROWS
            || if drafts > 0 {
                session.sampling.temperature != 0.0
                    || ids.len() != session.prompt_token_count.get() as usize
                    || self.first && ids != prompt
                    || ids.iter().any(|&t| t >= self.vocabulary)
            } else {
                !ids.is_empty()
            }
        {
            return Err(Error::Invalid);
        }
        let (Ok(leases), Ok(data)) = (
            <[LeaseId; 2]>::try_from(leases),
            <[SocketAddr; 2]>::try_from(data),
        ) else {
            return Err(Error::Invalid);
        };
        if leases[index] != lease || chunk == 0 || chunk > route.chunk_tokens || stop.len() > 64 {
            return Err(Error::Invalid);
        }
        let renewed = Instant::now();
        let (zero, one) = (bounded(0)?, bounded(1)?);
        let grants = [
            owner.refresh_grant(channel, lease, zero, one)?,
            owner.refresh_grant(channel, lease, one, zero)?,
        ];
        let steps = Steps {
            plan: plan.plan_digest.clone(),
            epoch: plan.authority_epoch,
            session: session.session_id,
            generation: session.generation_id,
            boundary_profile: plan.boundary_profile.clone(),
            leases,
            hidden: self.hidden,
            prompt: session.prompt_token_count.get(),
            chunk,
        };
        let output = session.max_output_tokens.get();
        let peer = data[1 - index];
        // This request's slot: its data listener, held until the request ends.
        let (_, listener, held) = stage.data.get(slot as usize).ok_or(Error::Invalid)?;
        if held.swap(true, Ordering::AcqRel) {
            return Err(Error::Busy);
        }
        /// Frees the slot; a peer's connection left over from this request is dropped first.
        struct Release<'a>(&'a AtomicBool, &'a PeerListener);
        impl Drop for Release<'_> {
            fn drop(&mut self) {
                while let Ok(Some(stale)) = self.1.try_accept() {
                    drop(stale);
                }
                self.0.store(false, Ordering::Release);
            }
        }
        let _release = Release(held, listener);
        // From here an engine session may be open.
        hold.live.set(true);
        let (ran, engine_session) = self.generate(
            stage,
            listener,
            channel,
            Grants {
                owner: &owner,
                backend,
                routes: grants,
                lease,
                lease_end,
                renewed,
            },
            Plan {
                steps: &steps,
                output,
                stop: &stop,
                prompt: if self.first { &prompt } else { &ids },
                peer,
                sampling: self.last.then(|| session.sampling.clone()),
                scratch: local.scratch_budget_bytes.0,
                drafts,
            },
            calls,
        );
        // Free the engine's session state whether or not the steps finished.
        let ended = match engine_session.as_deref() {
            Some(session) => self.end(session, calls),
            None => Err(Error::Invalid),
        };
        // A session that may still be open on the engine stays held (an ambiguous engine may
        // hold one it never confirmed): such an engine is stopped rather than kept with a
        // reservation no request will release. Otherwise the hold is released.
        let unknown = lock(&self.worker).map_or(true, |w| w.is_ambiguous());
        let open = engine_session.is_some() && ended.is_err() || unknown;
        hold.live.set(open);
        if open {
            self.broken.store(true, Ordering::Release);
            self.process.stop();
        }
        let ran = ran?;
        let ended_tokens = ended?;
        let report = Report {
            layers: route.layers,
            first: self.first,
            last: self.last,
            prefill_steps: ran.prefill,
            decode_steps: ran.decode,
            rows: ran.rows,
            prompt_tokens: steps.prompt,
            prompt_sha256: self.first.then(|| digest_ids(&prompt)),
            tokens: ran.tokens.len() as u32,
            tokens_sha256: digest_ids(&ran.tokens),
            stopped: ran.stopped,
            engine_ms: ran.engine_ms,
            ended_tokens,
            session_closed: true,
            refused_connections: ran.refused,
            drafted: ran.drafted,
            accepted: ran.accepted,
        };
        channel.send_record(&Reply::Finished {
            report: Box::new(report.clone()),
        })?;
        Ok(report)
    }
    /// End the engine session; a following `status` must no longer list it (the adapter
    /// checks the engine's report against the sessions still open).
    fn end(&self, session: &str, calls: &Calls) -> Result<u64> {
        let mut worker = lock(&self.worker)?;
        let _call = calls.begin();
        worker.end_session(session).map_err(engine_error)?;
        let Event::Ended { tokens } = worker.receive().map_err(engine_error)? else {
            return Err(Error::Invalid);
        };
        worker.status(None).map_err(engine_error)?;
        let Event::Status { .. } = worker.receive().map_err(engine_error)? else {
            return Err(Error::Invalid);
        };
        Ok(tokens)
    }
    /// The steps' outcome, and the engine session they opened (to end it whatever happened).
    fn generate(
        &self,
        stage: &Stage<'_>,
        listener: &PeerListener,
        channel: &mut GenerationStream,
        grants: Grants<'_>,
        plan: Plan<'_>,
        calls: &Calls,
    ) -> (Result<Ran>, Option<String>) {
        let (text, texts) = mpsc::sync_channel::<String>(1024);
        let process = self.process.clone();
        let (abort, longest) = (AtomicBool::new(false), std::sync::atomic::AtomicU64::new(0));
        let mut engine = Engine {
            worker: &self.worker,
            session: None,
            contract: &self.contract,
            first: self.first,
            ms: 0.0,
            rows: 0,
            abort: &abort,
            longest: &longest,
            calls,
        };
        let first = self.first;
        let Grants {
            owner,
            backend,
            routes,
            lease,
            lease_end,
            mut renewed,
        } = grants;
        thread::scope(|threads| {
            let routes = &routes;
            let steps = threads.spawn(move || {
                let data = Data {
                    stage,
                    listener,
                    owner,
                    backend,
                    routes,
                    refused: Default::default(),
                };
                // The sender lives until the steps end: its disconnection is their end.
                let ran = at(
                    "steps",
                    engine.begin(&plan).and_then(|()| {
                        if first {
                            data.first(&mut engine, &plan, &text)
                        } else {
                            data.last(&mut engine, &plan)
                        }
                    }),
                )
                .map(|mut ran| {
                    ran.engine_ms = engine.ms;
                    ran.rows = engine.rows;
                    ran.refused = data.refused.get();
                    ran
                });
                drop(text);
                (ran, engine.session)
            });
            let control = (|| -> Result<()> {
                let zero = bounded(0)?;
                let one = bounded(1)?;
                loop {
                    // The request ends with its lease, or when the daemon stops.
                    if Instant::now() >= lease_end || stage.stop.load(Ordering::Relaxed) {
                        return Err(Error::Expired);
                    }
                    let Command::Poll = at("stage_poll", channel.receive_record())? else {
                        return Err(Error::Invalid);
                    };
                    if renewed.elapsed() >= RENEW {
                        at("renew", channel.send_record(&Reply::Renew))?;
                        let started = Instant::now();
                        at("refresh", owner.refresh_grant(channel, lease, zero, one))?;
                        at("refresh", owner.refresh_grant(channel, lease, one, zero))?;
                        renewed = started;
                        continue;
                    }
                    let reply = match texts.recv_timeout(POLL_WAIT) {
                        Ok(mut text) => {
                            while text.len() < TEXT
                                && let Ok(more) = texts.try_recv()
                            {
                                text.push_str(&more);
                            }
                            Reply::Text { text }
                        }
                        Err(mpsc::RecvTimeoutError::Timeout) => Reply::Pending,
                        // The steps ended; their report answers this poll.
                        Err(mpsc::RecvTimeoutError::Disconnected) => return Ok(()),
                    };
                    at("stage_reply", channel.send_record(&reply))?;
                }
            })();
            drop(texts);
            if control.is_err() {
                // Data waits end with the grants. A running chunk is cancelled, and a call not
                // yet made is not made; only this request's own call still running well past
                // the longest chunk so far gets the (shared) engine killed, never a wait for
                // another request's call.
                channel.watch().cancel();
                abort.store(true, Ordering::Release);
                let longest = Duration::from_millis(longest.load(Ordering::Relaxed));
                let (grace, teardown) = (GRACE.max(longest.saturating_mul(4)), Instant::now());
                while !steps.is_finished() {
                    // From the later of the teardown and the call's start.
                    let running = calls.running_for().map(|d| d.min(teardown.elapsed()));
                    if running.is_some_and(|d| d >= grace) {
                        eprintln!("PIPELINE_STEP_FAILED step=engine_killed reason=Expired");
                        process.stop();
                        break;
                    }
                    thread::sleep(Duration::from_millis(10));
                }
            }
            match steps.join() {
                Ok((ran, session)) => (control.and(ran), session),
                Err(_) => (Err(Error::Transport), None),
            }
        })
    }
}

struct Grants<'a> {
    owner: &'a StageWorker,
    backend: &'a StageBackend,
    /// The 0→1 boundary route and the 1→0 token route, renewed by the control loop.
    routes: [StageGrant; 2],
    lease: LeaseId,
    lease_end: Instant,
    renewed: Instant,
}
struct Plan<'a> {
    steps: &'a Steps,
    output: u32,
    stop: &'a [u32],
    prompt: &'a [u32],
    peer: SocketAddr,
    sampling: Option<Sampling>,
    /// This stage's scratch budget: the received inputs it holds at once must fit in it.
    scratch: u64,
    /// Drafts a decode step may carry (GLY-360; 0 for sampled requests).
    drafts: u32,
}
#[derive(Default)]
struct Ran {
    prefill: u32,
    decode: u32,
    rows: u32,
    tokens: Vec<u32>,
    stopped: bool,
    engine_ms: f64,
    refused: u32,
    drafted: u32,
    accepted: u32,
}
struct Engine<'a> {
    worker: &'a std::sync::Mutex<EngineWorker>,
    /// The engine's handle for this request's session, once begun.
    session: Option<String>,
    contract: &'a BoundaryContract,
    first: bool,
    ms: f64,
    rows: u32,
    /// Set when the request is torn down: a running chunk is cancelled, not waited out.
    #[cfg_attr(not(unix), allow(dead_code))]
    abort: &'a AtomicBool,
    /// The longest chunk so far, in ms: the teardown grace scales with it.
    longest: &'a std::sync::atomic::AtomicU64,
    /// This request's engine call in progress, for the watchdogs.
    calls: &'a Calls,
}
impl Engine<'_> {
    /// Open this request's engine session; on the steps thread, so polls and renewals go on.
    fn begin(&mut self, plan: &Plan<'_>) -> Result<()> {
        let mut worker = lock(self.worker)?;
        if self.abort.load(Ordering::Acquire) {
            return Err(Error::Expired);
        }
        let _call = self.calls.begin();
        worker
            .begin_session(
                DataScope::from(&plan.steps.header(0, self.first)?),
                plan.steps.prompt,
                plan.output,
                plan.sampling.clone(),
            )
            .map_err(engine_error)?;
        let Event::Begun(session) = worker.receive().map_err(engine_error)? else {
            return Err(Error::Invalid);
        };
        self.session = Some(session);
        Ok(())
    }
    fn detokenize(&self, tokens: &[u32]) -> Result<String> {
        let mut worker = lock(self.worker)?;
        let _call = self.calls.begin();
        worker.detokenize(tokens).map_err(engine_error)?;
        let Event::Text(all) = worker.receive().map_err(engine_error)? else {
            return Err(Error::Invalid);
        };
        Ok(all)
    }
    fn run(&mut self, chunk: &StageChunk, head: bool) -> Result<Option<Output>> {
        let h = &chunk.input().headers()[0];
        let rows = h.token_count.get();
        let call = RunChunk {
            version: Version,
            inputs: self
                .contract
                .layouts(rows, PAYLOAD)
                .map_err(engine_error)?
                .0,
            step: StepIdentity {
                scope: DataScope::from(h),
                phase: h.phase,
                position_start: h.position_start,
                token_count: h.token_count,
            },
        };
        let bytes: &[u8] = &chunk.input().tensors()[0];
        let inputs: Vec<&[u8]> = if self.first {
            vec![bytes]
        } else {
            let (residual, mlp_out) = bytes.split_at(bytes.len() / 2);
            vec![residual, mlp_out]
        };
        let session = self.session.clone().ok_or(Error::Invalid)?;
        // One pass at a time on the stage's engine; another request's waits for this one. The
        // step's dispatch check runs once the engine is this request's, and a request torn
        // down meanwhile does not start it.
        let mut worker = lock(self.worker)?;
        if self.abort.load(Ordering::Acquire) {
            return Err(Error::Expired);
        }
        chunk.check_dispatch()?;
        let _call = self.calls.begin();
        let started = Instant::now();
        worker
            .run_chunk(&session, &call, &call.step, &inputs, head, false)
            .map_err(engine_error)?;
        let event = Self::reply(&mut worker, &session, self.abort)?;
        drop(worker);
        let elapsed = started.elapsed();
        self.ms += elapsed.as_secs_f64() * 1e3;
        self.longest.fetch_max(
            u64::try_from(elapsed.as_millis()).unwrap_or(u64::MAX),
            Ordering::Relaxed,
        );
        self.rows += rows;
        match event {
            Event::Chunk { output, .. } => Ok(output),
            Event::Withdrawn(_) => Err(Error::Expired),
            _ => Err(Error::Invalid),
        }
    }
    /// The chunk's reply. If the request is torn down meanwhile, cancel the session: the engine
    /// finishes the chunk and withdraws it, and the warm stage stays usable. Both replies are
    /// read, so nothing is left pending.
    // ponytail: on Windows the pipe cannot be polled; a chunk runs to its end or the kill.
    #[cfg_attr(not(unix), allow(unused_variables))]
    fn reply(worker: &mut EngineWorker, session: &str, abort: &AtomicBool) -> Result<Event> {
        #[cfg_attr(not(unix), allow(unused_mut))]
        let (mut outcome, mut cancelled, mut answered) = (None, false, false);
        loop {
            if let Some(event) = outcome.take_if(|_| !cancelled || answered) {
                return Ok(event);
            }
            #[cfg(unix)]
            {
                if !cancelled && outcome.is_none() && abort.load(Ordering::Acquire) {
                    worker.cancel(session).map_err(engine_error)?;
                    cancelled = true;
                }
                if !worker
                    .reply_ready(Duration::from_millis(20))
                    .map_err(engine_error)?
                {
                    continue;
                }
            }
            match worker.receive().map_err(engine_error)? {
                Event::Cancelled => answered = true,
                event => outcome = Some(event),
            }
        }
    }
}
/// A request's backend hold on the stage's shared runtime (GLY-190). Unless an engine session
/// may still be open (`live`), it is released when the request ends, success or failure: no
/// native work is left for it, and the runtime keeps nothing of the request.
#[derive(Default)]
struct Hold {
    backend: std::cell::OnceCell<StageBackend>,
    live: std::cell::Cell<bool>,
}
impl Drop for Hold {
    fn drop(&mut self) {
        if let Some(backend) = self.backend.get()
            && !self.live.get()
        {
            let _ = at("unloaded", backend.qualified_unloaded());
        }
    }
}
/// A queued step until it drains. Any early return drains it as failed, so the stage's
/// shared runtime (GLY-190) keeps nothing of a failed request: by then the engine has answered
/// for the step (a cancel included) or was killed, or never ran it.
struct Queued(Option<StageChunk>);
impl Queued {
    fn take(queued: &mpsc::Receiver<StageChunk>) -> Result<Self> {
        Ok(Self(Some(queued.try_recv().map_err(|_| Error::Invalid)?)))
    }
    fn drained(mut self) -> Result<()> {
        self.0
            .take()
            .ok_or(Error::Invalid)?
            .qualified_drained(false)
    }
}
impl std::ops::Deref for Queued {
    type Target = StageChunk;
    fn deref(&self) -> &StageChunk {
        self.0.as_ref().expect("a queued step until it drains")
    }
}
impl Drop for Queued {
    fn drop(&mut self) {
        if let Some(chunk) = self.0.take() {
            let _ = chunk.qualified_drained(true);
        }
    }
}
struct Data<'a> {
    stage: &'a Stage<'a>,
    /// This request's slot's data listener.
    listener: &'a PeerListener,
    owner: &'a StageWorker,
    backend: &'a StageBackend,
    routes: &'a [StageGrant; 2],
    /// Data connections dropped while waiting for the adjacent stage.
    refused: std::cell::Cell<u32>,
}
impl<'a> Data<'a> {
    /// The adjacent stage's connection: from its data address, through the stage handshake
    /// for this grant. Anything else is dropped and counted while the wait goes on.
    fn incoming(&self, peer: SocketAddr, grant: &StageGrant) -> Result<PeerStream> {
        let (stream, refused) = self.listener.accept_stage(
            self.stage.identity,
            self.stage.pin,
            grant,
            peer.ip(),
            Instant::now() + Duration::from_secs(5),
        )?;
        self.refused.set(self.refused.get() + refused);
        Ok(stream)
    }
    fn outgoing(&self, address: SocketAddr, grant: &StageGrant) -> Result<PeerStream> {
        PeerStream::connect_stage(
            TcpStream::connect_timeout(&address, Duration::from_secs(2))?,
            self.stage.identity,
            self.stage.pin,
            grant.clone(),
        )
    }
    /// Data connections in route order on both stages: 0→1 first, then 1→0.
    fn connect(&self, steps: &Steps, peer: SocketAddr, first: bool) -> Result<[DataStream; 2]> {
        let [forward, back] = self.routes;
        let (boundary, token) = (
            steps.header(0, false)?,
            steps.header_at(
                steps.prefill(),
                TensorHeaderPhase::Decode,
                steps.prompt,
                1,
                true,
            )?,
        );
        let forward = if first {
            let peer = self.outgoing(peer, forward)?;
            let admission = self.owner.data_admission(&peer, forward, boundary)?;
            DataStream::connect(peer, &admission)?
        } else {
            let peer = self.incoming(peer, forward)?;
            let admission = self.owner.data_admission(&peer, forward, boundary)?;
            DataStream::accept(peer, &admission)?
        };
        let back = if first {
            let peer = self.incoming(peer, back)?;
            let admission = self.owner.data_admission(&peer, back, token)?;
            DataStream::accept(peer, &admission)?
        } else {
            let peer = self.outgoing(peer, back)?;
            let admission = self.owner.data_admission(&peer, back, token)?;
            DataStream::connect(peer, &admission)?
        };
        Ok([forward, back])
    }
    /// Open the next receive on `input`: admit it and grant its whole credit, so the peer can
    /// send as soon as it has the data.
    fn open<'s>(
        &'s self,
        input: &'s mut DataStream,
        slot: &'s mut Option<DataAdmission<'a>>,
        (grant, header, budget): (&StageGrant, TensorHeader, &ReceiveBudget),
    ) -> Result<FlowReceiver<'s, 'a>> {
        let total = header.total_tensor_bytes.0;
        let admission: &'s DataAdmission<'a> = slot.insert(at(
            "data_admission",
            self.owner.data_admission(input.peer(), grant, header),
        )?);
        let mut flow = at(
            "begin_receive",
            input.begin_receive(
                std::slice::from_ref(admission),
                budget,
                Instant::now() + DATA_WAIT,
            ),
        )?;
        at("grant_credit", flow.grant_credit(total))?;
        Ok(flow)
    }
    fn drain(mut flow: FlowReceiver<'_, '_>, total: u64) -> Result<ReceivedTensors> {
        for _ in 0..total.div_ceil(DATA_PAYLOAD_LIMIT as u64) {
            at("receive_chunk", flow.receive_chunk())?;
        }
        at("finish_receive", flow.finish())
    }
    /// Send one output. With `next`, the stage's next receive is opened once the bytes are
    /// on the wire, before their receipt comes back: its credit travels with them, so the
    /// peer's answer never waits a round trip for it.
    #[allow(clippy::too_many_arguments)]
    fn send<'s>(
        &'s self,
        out: &mut DataStream,
        grant: &StageGrant,
        header: TensorHeader,
        chunk: &StageChunk,
        bytes: &[u8],
        next: Option<(&'s mut DataStream, &'s mut Option<DataAdmission<'a>>)>,
        receive: (&StageGrant, TensorHeader, &ReceiveBudget),
    ) -> Result<Option<FlowReceiver<'s, 'a>>> {
        let admission = at(
            "output_admission",
            self.owner
                .output_admission(out.peer(), grant, header, chunk),
        )?;
        let mut flow = at(
            "begin_send",
            out.begin_send(std::slice::from_ref(&admission), Instant::now() + DATA_WAIT),
        )?;
        let DataControl::Credit { .. } = at("credit", flow.receive_control())? else {
            return Err(Error::Invalid);
        };
        let pieces = bytes.chunks(DATA_PAYLOAD_LIMIT);
        let count = pieces.len();
        for piece in pieces {
            at("send_chunk", flow.send_chunk(piece))?;
        }
        let opened = match next {
            Some((input, slot)) => Some(self.open(input, slot, receive)?),
            None => None,
        };
        for _ in 0..count {
            let DataControl::ReceiptAck { .. } = at("receipt", flow.receive_control())? else {
                return Err(Error::Invalid);
            };
        }
        at("finish_send", flow.finish())?;
        Ok(opened)
    }
    /// Run the queued step and send its boundary to the last stage; with `next`, open
    /// `receive` (the tokens that come back) as it goes out.
    #[allow(clippy::too_many_arguments)]
    fn forward<'s>(
        &'s self,
        engine: &mut Engine<'_>,
        out: &mut DataStream,
        queued: &mpsc::Receiver<StageChunk>,
        boundary: TensorHeader,
        receive: TensorHeader,
        budget: &ReceiveBudget,
        next: Option<(&'s mut DataStream, &'s mut Option<DataAdmission<'a>>)>,
    ) -> Result<Option<FlowReceiver<'s, 'a>>> {
        let chunk = Queued::take(queued)?;
        let Some(Output::Boundary { bytes, .. }) = at("run", engine.run(&chunk, false))? else {
            return Err(Error::Invalid);
        };
        let receive = (&self.routes[1], receive, budget);
        let opened = self.send(
            out,
            &self.routes[0],
            boundary,
            &chunk,
            &bytes,
            next,
            receive,
        )?;
        chunk.drained()?;
        Ok(opened)
    }
    /// First stage: prompt chunks start locally; the last one and every decode step open the
    /// receive for the tokens coming back as their boundary goes out. Each decode step is the
    /// last accepted token and its drafts (`Decoder`). Text is detokenized for the client after
    /// the next step has gone out.
    fn first(
        &self,
        engine: &mut Engine<'_>,
        plan: &Plan<'_>,
        text: &mpsc::SyncSender<String>,
    ) -> Result<Ran> {
        let steps = plan.steps;
        let [mut out, mut input] = at("connect", self.connect(steps, plan.peer, true))?;
        let (queue, queued) = mpsc::sync_channel(1);
        // The tokens in hand and the next ones.
        let most = 8 * u64::from(MAX_DECODE_ROWS);
        let budget = ReceiveBudget::new(most, most, most)?;
        let mut ran = Ran::default();
        let mut slot = None;
        let mut prompt = |k: u64| -> Result<()> {
            let header = steps.header(k, true)?;
            let start = header.position_start.get() as usize;
            let ids = plan
                .prompt
                .get(start..start + header.token_count.get() as usize)
                .ok_or(Error::Invalid)?;
            let bytes = ids.iter().flat_map(|t| t.to_le_bytes()).collect();
            ran.prefill += 1;
            self.owner
                .enqueue_prompt(&self.routes[1], self.backend, header, bytes, &queue)
        };
        let (prefill, decode) = (steps.prefill(), TensorHeaderPhase::Decode);
        for k in 0..prefill - 1 {
            prompt(k)?;
            let boundary = steps.header(k, false)?;
            let unused = steps.header(k, true)?;
            self.forward(engine, &mut out, &queued, boundary, unused, &budget, None)?;
        }
        prompt(prefill - 1)?;
        // The last stage's token for the prompt: one row at its end.
        let mut message = steps.header_at(prefill, decode, steps.prompt, 1, true)?;
        let next = Some((&mut input, &mut slot));
        let boundary = steps.header(prefill - 1, false)?;
        let mut flow = self
            .forward(
                engine,
                &mut out,
                &queued,
                boundary,
                message.clone(),
                &budget,
                next,
            )?
            .ok_or(Error::Invalid)?;
        let mut shown = String::new();
        let mut show = |engine: &mut Engine<'_>, tokens: &[u32]| -> Result<()> {
            // ponytail: the whole answer is detokenized for each step; O(n^2) in its length.
            let all = engine.detokenize(tokens)?;
            // Hold back a partial UTF-8 character; never resend text already shown.
            if let Some(new) = all.strip_prefix(shown.as_str())
                && !new.is_empty()
                && !new.ends_with('\u{fffd}')
            {
                text.try_send(new.to_owned())
                    .map_err(|_| Error::Transport)?;
                shown = all;
            }
            Ok(())
        };
        let mut decoder = Decoder::new(plan, plan.prompt);
        loop {
            let rows = message.token_count.get();
            let received = Self::drain(flow, 4 * u64::from(rows))?;
            let tokens: Vec<u32> = received.tensors()[0]
                .chunks_exact(4)
                .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
                .collect();
            let kept = at("accept", decoder.accept(&tokens))?;
            if decoder.done() {
                show(engine, &decoder.tokens)?;
                break;
            }
            // The next step: the last accepted token and its drafts, after the kept rows.
            let k = message.sequence.0;
            let start = message.position_start.get() + kept;
            let drafts = decoder.next();
            at(
                "enqueue_decode",
                self.owner.enqueue_decode(
                    &self.routes[1],
                    self.backend,
                    &received,
                    kept,
                    &drafts,
                    &queue,
                ),
            )?;
            drop(received);
            let rows = 1 + drafts.len() as u32;
            let boundary = steps.header_at(k, decode, start, rows, false)?;
            message = steps.header_at(k + 1, decode, start + 1, rows, true)?;
            let next = Some((&mut input, &mut slot));
            flow = self
                .forward(
                    engine,
                    &mut out,
                    &queued,
                    boundary,
                    message.clone(),
                    &budget,
                    next,
                )?
                .ok_or(Error::Invalid)?;
            ran.decode += 1;
            show(engine, &decoder.tokens)?;
        }
        decoder.finish(&mut ran);
        Ok(ran)
    }
    /// Last stage: run each boundary; a head step's tokens (one per decode row, checking the
    /// drafts) go back to the first stage. The next boundary's receive opens as the tokens go
    /// out (when both inputs fit in scratch), or after a prompt chunk has run and drained.
    fn last(&self, engine: &mut Engine<'_>, plan: &Plan<'_>) -> Result<Ran> {
        let steps = plan.steps;
        let [mut input, mut out] = at("connect", self.connect(steps, plan.peer, false))?;
        let (queue, queued) = mpsc::sync_channel(1);
        // The largest boundary is the first prompt chunk or a drafted step (at most a chunk and
        // MAX_DECODE_ROWS rows); two can be held at once.
        let rows = steps
            .prompt
            .min(steps.chunk)
            .max(MAX_DECODE_ROWS.min(steps.chunk));
        let largest = 2 * 4 * u64::from(rows) * u64::from(steps.hidden);
        let budget = ReceiveBudget::new(largest, largest, largest)?;
        let mut ran = Ran::default();
        let mut slot = None;
        let mut decoder = Decoder::new(plan, plan.prompt);
        let decode = TensorHeaderPhase::Decode;
        let mut header = steps.header(0, false)?;
        let mut flow = self.open(
            &mut input,
            &mut slot,
            (&self.routes[0], header.clone(), &budget),
        )?;
        loop {
            let k = header.sequence.0;
            let (start, rows) = (header.position_start.get(), header.token_count.get());
            let head = header.phase == decode || start + rows == steps.prompt;
            let received = Self::drain(flow, header.total_tensor_bytes.0)?;
            self.owner
                .enqueue_chunk(&self.routes[0], self.backend, received, &queue)?;
            let chunk = Queued::take(&queued)?;
            if !head {
                // The next prompt chunk opens once this one has run and drained: its receive
                // deadline then leaves out this chunk's compute, and two prompt chunks never
                // share the scratch budget.
                engine.run(&chunk, false)?;
                ran.prefill += 1;
                chunk.drained()?;
                header = steps.header(k + 1, false)?;
                flow = self.open(
                    &mut input,
                    &mut slot,
                    (&self.routes[0], header.clone(), &budget),
                )?;
                continue;
            }
            let Some(Output::Token { tokens, .. }) = engine.run(&chunk, true)? else {
                return Err(Error::Invalid);
            };
            if header.phase == decode {
                ran.decode += 1;
            } else {
                ran.prefill += 1;
            }
            let kept = decoder.accept(&tokens)?;
            let done = decoder.done();
            // The tokens are for positions end - len + 1 ..= end.
            let message_start = start + rows + 1 - tokens.len() as u32;
            let back = steps.header_at(k + 1, decode, message_start, tokens.len() as u32, true)?;
            let bytes: Vec<u8> = tokens.iter().flat_map(|t| t.to_le_bytes()).collect();
            let next = if done {
                None
            } else {
                let start = message_start + kept;
                let rows = 1 + decoder.next().len() as u32;
                Some(steps.header_at(k + 1, decode, start, rows, false)?)
            };
            // Early only while this step's input and the next one fit in scratch together.
            let early = next
                .as_ref()
                .filter(|n| header.total_tensor_bytes.0 + n.total_tensor_bytes.0 <= plan.scratch);
            if let Some(next) = early {
                let receive = (&self.routes[0], next.clone(), &budget);
                let into = Some((&mut input, &mut slot));
                flow = self
                    .send(
                        &mut out,
                        &self.routes[1],
                        back,
                        &chunk,
                        &bytes,
                        into,
                        receive,
                    )?
                    .ok_or(Error::Invalid)?;
                chunk.drained()?;
                header = next.clone();
                continue;
            }
            let unused = (&self.routes[0], header.clone(), &budget);
            self.send(
                &mut out,
                &self.routes[1],
                back,
                &chunk,
                &bytes,
                None,
                unused,
            )?;
            chunk.drained()?;
            let Some(next) = next else {
                break;
            };
            header = next;
            flow = self.open(
                &mut input,
                &mut slot,
                (&self.routes[0], header.clone(), &budget),
            )?;
        }
        decoder.finish(&mut ran);
        Ok(ran)
    }
}

/// GLY-360: what both stages derive alike for decode steps, so each predicts the other's next
/// header exactly: the drafts after each step's first token (an n-gram lookup over the prompt
/// and the accepted tokens), the tokens accepted from the last stage's per-row tokens, and
/// where the answer stops.
struct Decoder<'p> {
    plan: &'p Plan<'p>,
    /// The prompt, then the accepted tokens.
    history: Vec<u32>,
    tokens: Vec<u32>,
    stopped: bool,
    /// The drafts of the step in flight (after its first token).
    drafts: Vec<u32>,
    drafted: u32,
    accepted: u32,
}
impl<'p> Decoder<'p> {
    fn new(plan: &'p Plan<'p>, prompt: &[u32]) -> Self {
        Self {
            plan,
            history: prompt.to_vec(),
            tokens: Vec::new(),
            stopped: false,
            drafts: Vec::new(),
            drafted: 0,
            accepted: 0,
        }
    }
    /// Take the last stage's tokens for the step in flight, one per row: the drafts they
    /// confirm, then the one after. Returns how many drafts held.
    fn accept(&mut self, rows: &[u32]) -> Result<u32> {
        if rows.len() != self.drafts.len() + 1 {
            return Err(Error::Invalid);
        }
        let kept = rows
            .iter()
            .zip(&self.drafts)
            .take_while(|(row, draft)| row == draft)
            .count();
        self.accepted += kept as u32;
        for &token in &rows[..=kept] {
            self.tokens.push(token);
            self.history.push(token);
            self.stopped = self.plan.stop.contains(&token);
            if self.done() {
                break;
            }
        }
        Ok(kept as u32)
    }
    fn done(&self) -> bool {
        self.stopped || self.tokens.len() as u32 >= self.plan.output
    }
    /// The next step's drafts: at most `drafts` of them, only as many as the answer still has
    /// room for after the step's own token, and a step is at most one chunk.
    fn next(&mut self) -> Vec<u32> {
        let room = self.plan.output as usize - self.tokens.len() - 1;
        let rows = MAX_DECODE_ROWS.min(self.plan.steps.chunk) as usize;
        let most = (self.plan.drafts as usize).min(room).min(rows - 1);
        self.drafts = draft::draft(&self.history, most);
        self.drafted += self.drafts.len() as u32;
        self.drafts.clone()
    }
    fn finish(self, ran: &mut Ran) {
        ran.tokens = self.tokens;
        ran.stopped = self.stopped;
        ran.drafted = self.drafted;
        ran.accepted = self.accepted;
    }
}

/// Coordinator side: admit the client, count on the first stage, give each stage its lease,
/// session and grants, then relay the first stage's text as SSE (or one JSON answer).
// ponytail: one request at a time through the stages; another is refused, not queued.
pub fn dispatch(
    incoming: IncomingClient,
    admission: &ClientAdmission,
    reader: &EnrollmentReader,
    route: &PipelineRoute,
    epoch: AuthorityEpoch,
    slots: &[AtomicBool],
    sequences: &[std::sync::Mutex<u64>],
) -> Result<()> {
    let binding = GenerationBinding {
        version: Version,
        epoch,
        model: route.plan.manifest_digest.clone(),
        profile: route.plan.execution_profile.clone(),
    };
    let (chat, response) = crate::inference::admit(incoming, admission, &binding)?;
    let request_id: RequestId = id()?;
    let temperature = chat.temperature.unwrap_or(0.0);
    let top_p = chat.top_p.unwrap_or(1.0);
    // A free session slot on both stages (GLY-190): its data listeners carry this request.
    let slot = slots.iter().position(|s| !s.swap(true, Ordering::AcqRel));
    let refused = if chat.max_tokens.get() > route.max_output_tokens
        || !(0.0..=2.0).contains(&temperature)
        || !(top_p > 0.0 && top_p <= 1.0)
    {
        Some(Error::Invalid)
    } else if slot.is_none() {
        Some(Error::Busy)
    } else {
        None
    };
    struct Free<'a>(Option<&'a AtomicBool>);
    impl Drop for Free<'_> {
        fn drop(&mut self) {
            if let Some(slot) = self.0 {
                slot.store(false, Ordering::Release);
            }
        }
    }
    let _free = Free(slot.map(|i| &slots[i]));
    if let Some(error) = refused {
        let _ = response.fail(&crate::inference::failure(error, request_id));
        return Err(error);
    }
    let slot = slot.ok_or(Error::Busy)?;
    let mut started = false;
    let result = relay_pipeline(
        reader,
        route,
        slot,
        sequences,
        binding,
        &chat,
        response,
        request_id,
        &mut started,
    );
    if result.is_err() && started {
        // ponytail: the stages do not say when a failed request has unwound (grants expire,
        // then a stuck engine is cancelled or killed); the slot stays taken for that bound.
        thread::sleep(TEARDOWN);
    }
    result
}

/// Cancels every stage connection when the request ends, on every path.
struct Watches(Vec<starlings_security::GenerationWatch>);
impl Watches {
    fn cancel(&self) {
        for watch in &self.0 {
            watch.cancel();
        }
    }
}
impl Drop for Watches {
    fn drop(&mut self) {
        self.cancel();
    }
}

#[allow(clippy::too_many_arguments)]
fn relay_pipeline(
    reader: &EnrollmentReader,
    route: &PipelineRoute,
    slot: usize,
    sequences: &[std::sync::Mutex<u64>],
    binding: GenerationBinding,
    chat: &ChatRequest,
    response: ClientResponse,
    request_id: RequestId,
    started: &mut bool,
) -> Result<()> {
    let epoch = binding.epoch;
    // The stages end the request with their leases; this side gives up a little later.
    let begun = Instant::now();
    let deadline = begun + Duration::from_millis(route.lifetime_ms.into()) + GRACE;
    let mut watches = Watches(Vec::new());
    let setup = (|| -> Result<_> {
        // Each stage waits at most one record time for its next step: connect the last stage
        // only after the first has counted.
        let mut connect = |i: usize| -> Result<GenerationStream> {
            let socket = at(
                "connect",
                TcpStream::connect_timeout(&route.stages[i].endpoint, Duration::from_secs(2))
                    .map_err(Error::from),
            )?;
            let stream = at(
                "generation",
                reader.connect_generation(socket, route.plan.stages[i].node_id, binding.clone()),
            )?;
            watches.0.push(stream.watch());
            Ok(stream)
        };
        let mut first = connect(0)?;
        // Drafts are for greedy requests; only they send the prompt's IDs on (GLY-360).
        let drafting = route.draft_tokens > 0 && chat.temperature.unwrap_or(0.0) == 0.0;
        first.send_record(&Command::Count {
            request: Box::new(chat.clone()),
            ids: drafting,
        })?;
        let counted = first.receive_record()?;
        let mut streams = [first, connect(1)?];
        let (prompt, ids) = match counted {
            Reply::Counted {
                request,
                tokens,
                ids,
            } if request == starlings_protocol::content_hash(chat)?
                && tokens > 0
                && ids.len() == if drafting { tokens as usize } else { 0 } =>
            {
                (tokens, ids)
            }
            Reply::Rejected => return Err(Error::Capacity),
            _ => return Err(Error::Invalid),
        };
        let (plan, leases, sessions) = records(route, epoch, &response, chat, prompt)?;
        let mut issuers = Vec::new();
        for (((stream, lease), session), counter) in
            streams.iter_mut().zip(&leases).zip(sessions).zip(sequences)
        {
            let issuer = at(
                "issuer",
                reader.stage_issuer_for(&response, plan.clone(), lease.clone()),
            )?;
            at("assign", issuer.assign(stream))?;
            for operation in [
                Operation::Prepare {
                    plan: Box::new(plan.clone()),
                    lease: Box::new(lease.clone()),
                },
                Operation::Commit {
                    lease_id: lease.lease_id,
                },
                Operation::Begin {
                    session: Box::new(session),
                },
            ] {
                // The stage's runtime takes control requests in increasing sequence, from every
                // request in flight (GLY-190): number and apply them one at a time per stage.
                let mut sequence = counter.lock().map_err(|_| Error::Invalid)?;
                *sequence += 1;
                let request = ControlRequest::new(id()?, epoch.number.0, *sequence, operation)?;
                let expected = request.request_id;
                at(
                    "apply",
                    stream.send_record(&Command::Apply {
                        request: Box::new(request),
                    }),
                )?;
                if !matches!(at("applied", stream.receive_record())?, Reply::Applied { request } if request == expected)
                {
                    return Err(Error::Invalid);
                }
                drop(sequence);
            }
            issuers.push(issuer);
        }
        let start = Command::Start {
            leases: leases.iter().map(|l| l.lease_id).collect(),
            data: route
                .stages
                .iter()
                .map(|s| {
                    std::iter::once(&s.data)
                        .chain(&s.more_data)
                        .nth(slot)
                        .copied()
                })
                .collect::<Option<_>>()
                .ok_or(Error::Invalid)?,
            chunk: route.chunk_tokens,
            stop: route.stop_tokens.clone(),
            drafts: if drafting { route.draft_tokens } else { 0 },
            prompt: ids,
            slot: slot as u32,
        };
        *started = true;
        for stream in &mut streams {
            stream.send_record(&start)?;
        }
        // Each stage renews both route grants right away.
        for (stream, issuer) in streams.iter_mut().zip(&issuers) {
            issuer.respond(stream)?;
            issuer.respond(stream)?;
        }
        let issuers = <[StageIssuer; 2]>::try_from(issuers).map_err(|_| Error::Invalid)?;
        Ok((streams, issuers, prompt, plan))
    })();
    let ([mut first, mut last], [first_issuer, last_issuer], prompt, plan) = match setup {
        Ok(setup) => setup,
        Err(error) => {
            let _ = response.fail(&crate::inference::failure(error, request_id));
            return Err(error);
        }
    };
    let setup_ms = begun.elapsed().as_secs_f64() * 1e3;
    let (watch, tail_watch) = (response.watch()?, response.watch()?);
    let session = plan.plan_id.to_string();
    let created = time::OffsetDateTime::now_utc().unix_timestamp();
    let chunk = |delta: serde_json::Value, finish: Option<&str>| {
        serde_json::json!({
            "id": format!("chatcmpl-{session}"), "object": "chat.completion.chunk",
            "created": created, "model": chat.model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        })
    };
    let streaming = chat.stream == Some(true);
    let mut response = Some(response);
    let mut sse = if streaming {
        Some(response.take().ok_or(Error::Invalid)?.stream()?)
    } else {
        None
    };
    let mut content = String::new();
    let mut first_text = None;
    let started = Instant::now();
    let relayed = thread::scope(|threads| {
        let (watches, last) = (&watches, &mut last);
        let tail = threads.spawn(move || {
            let report = poll(last, &last_issuer, &tail_watch, deadline, |_| {
                Err(Error::Invalid)
            });
            if report.is_err() {
                watches.cancel();
            }
            report
        });
        let head = poll(&mut first, &first_issuer, &watch, deadline, |text| {
            first_text.get_or_insert_with(|| begun.elapsed().as_secs_f64() * 1e3);
            if let Some(sse) = &mut sse {
                let role = content.is_empty().then_some("assistant");
                sse.event(&serde_json::to_vec(&chunk(
                    serde_json::json!({"role": role, "content": text}),
                    None,
                ))?)?;
            }
            content.push_str(&text);
            if content.len() > starlings_protocol::MAX_JSON_BYTES / 2 {
                return Err(Error::Capacity);
            }
            Ok(())
        });
        if head.is_err() {
            watches.cancel();
        }
        let tail = tail.join().map_err(|_| Error::Transport)?;
        let (head, tail) = (head?, tail?);
        // Usage and the finish reason come from stage 0; the last stage must agree with it.
        if !head.first
            || !tail.last
            || head.tokens != tail.tokens
            || head.stopped != tail.stopped
            || head.tokens_sha256 != tail.tokens_sha256
            || head.tokens == 0
            || head.tokens > chat.max_tokens.get()
            || head.prompt_tokens != prompt
            || tail.prompt_tokens != prompt
        {
            return Err(Error::Invalid);
        }
        Ok((head, tail))
    });
    let (head, tail) = match relayed {
        Ok(reports) => reports,
        Err(error) => {
            if let Some(sse) = &mut sse {
                let _ = sse.fail(&crate::inference::failure(error, request_id));
            } else if let Some(response) = response {
                let _ = response.fail(&crate::inference::failure(error, request_id));
            }
            return Err(error);
        }
    };
    let finish = if head.stopped { "stop" } else { "length" };
    let usage = serde_json::json!({"prompt_tokens": prompt, "completion_tokens": head.tokens,
        "total_tokens": prompt + head.tokens});
    let mut stages = serde_json::json!([head, tail]);
    // Token digests would confirm a guessed prompt or answer: only test builds log them.
    if !cfg!(feature = "test-backend") {
        for stage in stages.as_array_mut().into_iter().flatten() {
            if let Some(stage) = stage.as_object_mut() {
                stage.remove("prompt_sha256");
                stage.remove("tokens_sha256");
            }
        }
    }
    // Kept before the answer's last chunk goes out, so the web UI finds it with that chunk.
    let report = serde_json::json!({"id": format!("chatcmpl-{session}"),
        "at": (time::OffsetDateTime::now_utc().unix_timestamp_nanos() / 1_000_000) as i64,
        "plan": plan.plan_digest, "ms": started.elapsed().as_secs_f64() * 1e3,
        "setup_ms": setup_ms, "first_text_ms": first_text, "prompt_tokens": prompt,
        "finish": finish, "stages": stages});
    remember(report.clone());
    if let Some(sse) = &mut sse {
        let mut last = chunk(serde_json::json!({}), Some(finish));
        last["usage"] = usage;
        sse.event(&serde_json::to_vec(&last)?)?;
        sse.finish()?;
    } else {
        response.ok_or(Error::Invalid)?.json(&serde_json::to_vec(&serde_json::json!({
            "id": format!("chatcmpl-{session}"), "object": "chat.completion", "created": created,
            "model": chat.model, "usage": usage,
            "choices": [{"index": 0, "finish_reason": finish,
                "message": {"role": "assistant", "content": content}}],
        }))?)?;
    }
    use std::io::Write;
    let _ = writeln!(std::io::stdout(), "PIPELINE_REPORT {report}");
    // Both stages close their connections once their sessions have ended and their owners
    // are gone; hold the slot until then.
    for stream in [&mut first, &mut last] {
        let _ = stream.receive_record::<Reply>();
    }
    let _ = writeln!(
        std::io::stdout(),
        "PIPELINE_CLOSED {}",
        serde_json::json!({"plan": plan.plan_digest, "ms": begun.elapsed().as_secs_f64() * 1e3})
    );
    Ok(())
}

/// The last REPORTS_KEPT request reports, oldest first, for the web UI (one pipeline per
/// coordinator process).
pub static REPORTS: std::sync::Mutex<std::collections::VecDeque<serde_json::Value>> =
    std::sync::Mutex::new(std::collections::VecDeque::new());
const REPORTS_KEPT: usize = 50;
fn remember(report: serde_json::Value) {
    if let Ok(mut reports) = REPORTS.lock() {
        if reports.len() == REPORTS_KEPT {
            reports.pop_front();
        }
        reports.push_back(report);
    }
}

fn poll(
    stream: &mut GenerationStream,
    issuer: &StageIssuer,
    client: &ClientWatch,
    deadline: Instant,
    mut text: impl FnMut(String) -> Result<()>,
) -> Result<Report> {
    loop {
        if Instant::now() >= deadline {
            return at("deadline", Err(Error::Expired));
        }
        at("client", client.check())?;
        at("poll", stream.send_record(&Command::Poll))?;
        match at("reply", stream.receive_record())? {
            Reply::Pending => {}
            Reply::Renew => {
                at("respond", issuer.respond(stream))?;
                at("respond", issuer.respond(stream))?;
            }
            Reply::Text { text: t } if !t.is_empty() && t.len() <= 64 * 1024 => text(t)?,
            Reply::Finished { report } => return Ok(*report),
            _ => return Err(Error::Invalid),
        }
    }
}

/// One request's plan, leases and sessions. The plan keeps the approved stages and gets this
/// coordinator's epoch, a new ID and the lease lifetime.
fn records(
    route: &PipelineRoute,
    epoch: AuthorityEpoch,
    client: &ClientResponse,
    chat: &ChatRequest,
    prompt: u32,
) -> Result<(PlacementPlan, Vec<Lease>, Vec<Session>)> {
    let format = time::format_description::well_known::Rfc3339;
    let now = time::OffsetDateTime::now_utc();
    let mut plan = route.plan.clone();
    plan.authority_epoch = epoch.number;
    plan.plan_id = id()?;
    plan.created_at = now.format(&format).map_err(|_| Error::Invalid)?;
    plan.expires_at = (now + time::Duration::milliseconds(route.lifetime_ms.into()))
        .format(&format)
        .map_err(|_| Error::Invalid)?;
    plan.plan_digest =
        starlings_protocol::canonical_digest(&serde_json::to_value(&plan)?, "plan_digest")?;
    // ponytail: a request must finish inside one 60 s lease; max_output_tokens is set for that.
    let (session_id, generation_id) = (id()?, id()?);
    let mut leases = Vec::new();
    let mut sessions = Vec::new();
    for stage in &plan.stages {
        let lease = Lease {
            kind: "Lease".into(),
            protocol_version: Version,
            lease_id: id()?,
            cluster_id: plan.cluster_id,
            plan_digest: plan.plan_digest.clone(),
            authority_epoch: plan.authority_epoch,
            owner_node_id: stage.node_id,
            issued_at: plan.created_at.clone(),
            lifetime_ms: bounded(route.lifetime_ms)?,
            max_sessions: plan.max_sessions,
            reserved_bytes: Decimal(stage.reserved_bytes()?),
        };
        sessions.push(Session {
            kind: "Session".into(),
            protocol_version: Version,
            session_id,
            generation_id,
            client_id: client
                .client()
                .to_string()
                .parse()
                .map_err(|_| Error::Invalid)?,
            lease_id: lease.lease_id,
            plan_digest: plan.plan_digest.clone(),
            authority_epoch: plan.authority_epoch,
            prompt_token_count: bounded(prompt)?,
            max_output_tokens: chat.max_tokens,
            context_limit: plan.context_limit,
            sampling: Sampling {
                profile: "sample-v1".into(),
                temperature: chat.temperature.unwrap_or(0.0),
                top_p: chat.top_p.unwrap_or(1.0),
                seed: chat.seed.ok_or(Error::Invalid)?,
            },
            state: SessionState::Queued,
        });
        leases.push(lease);
    }
    Ok((plan, leases, sessions))
}

#[cfg(test)]
mod tests {
    use super::*;
    /// GLY-190: a request waiting for the shared engine is not in a call, so its watchdogs
    /// leave the other request's call alone.
    #[test]
    fn only_a_call_on_the_engine_counts_as_running() {
        let calls = Calls::new();
        assert!(calls.running_for().is_none());
        let call = calls.begin();
        thread::sleep(Duration::from_millis(20));
        assert!(calls.running_for().unwrap() >= Duration::from_millis(20));
        drop(call);
        assert!(calls.running_for().is_none());
    }
    #[test]
    fn both_stages_derive_the_same_steps_and_headers() {
        let steps = Steps {
            plan: starlings_protocol::byte_digest(b"plan"),
            epoch: Decimal(3),
            session: id().unwrap(),
            generation: id().unwrap(),
            boundary_profile: "qwen35-residual-mlp-bf16".into(),
            leases: [id().unwrap(), id().unwrap()],
            hidden: 4096,
            prompt: 70,
            chunk: 32,
        };
        assert_eq!(steps.prefill(), 3);
        let at = |k| {
            let h = steps.header(k, true).unwrap();
            (h.phase, h.position_start.get(), h.token_count.get())
        };
        assert_eq!(at(0), (TensorHeaderPhase::Prefill, 0, 32));
        assert_eq!(at(2), (TensorHeaderPhase::Prefill, 64, 6));
        assert!(steps.header(3, true).is_err(), "decode steps are explicit");
        let boundary = steps.header(2, false).unwrap();
        assert_eq!(boundary.lease_id, steps.leases[1]);
        assert_eq!(boundary.total_tensor_bytes.0, 4 * 6 * 4096);
        assert_eq!(
            (
                boundary.source_stage.get(),
                boundary.destination_stage.get()
            ),
            (0, 1)
        );
        // The token after the prompt is one row at its end, step 3's first-stage input.
        let token = steps
            .header_at(3, TensorHeaderPhase::Decode, 70, 1, true)
            .unwrap();
        assert_eq!(token.lease_id, steps.leases[0]);
        assert_eq!(
            (token.source_stage.get(), token.destination_stage.get()),
            (1, 0)
        );
        assert_eq!(token.position_start.get(), 64 + 6);
        assert_eq!(token.sequence.0, 3);
        assert_eq!(
            token.invocation_id,
            step_invocation(steps.generation, 3).unwrap()
        );
        assert_eq!(token.total_tensor_bytes.0, 4);
        // A boundary larger than one wire frame is sent in 8 MiB pieces.
        let mut big = steps;
        big.prompt = 1024;
        big.chunk = 1024;
        let h = big.header(0, false).unwrap();
        assert_eq!(h.total_tensor_bytes.0, 16 << 20);
        assert!(!h.end_of_tensor && h.chunk_bytes.0 == DATA_PAYLOAD_LIMIT as u64);
    }
    #[test]
    fn both_stages_accept_the_same_drafts_and_stop_at_the_same_token() {
        let steps = Steps {
            plan: starlings_protocol::byte_digest(b"plan"),
            epoch: Decimal(3),
            session: id().unwrap(),
            generation: id().unwrap(),
            boundary_profile: "qwen35-residual-mlp-bf16".into(),
            leases: [id().unwrap(), id().unwrap()],
            hidden: 4,
            prompt: 6,
            chunk: 6,
        };
        let plan = |output, drafts| Plan {
            steps: &steps,
            output,
            stop: &[99],
            prompt: &[],
            peer: "127.0.0.1:1".parse().unwrap(),
            sampling: None,
            scratch: 0,
            drafts,
        };
        let plan_10 = plan(10, 4);
        let mut d = Decoder::new(&plan_10, &[1, 2, 3, 4, 1, 2]);
        // The prompt's token: no drafts on it yet.
        assert_eq!(d.accept(&[3]).unwrap(), 0);
        // [3] after "1 2" again: the n-gram guess is what followed before.
        assert_eq!(d.next(), [4, 1, 2, 3]);
        // Two drafts held, the third missed: the new tokens are 4 1 and the row's own 7.
        assert!(d.accept(&[4, 9]).is_err(), "one token per row");
        assert_eq!(d.accept(&[4, 1, 7, 8, 8]).unwrap(), 2);
        assert_eq!(d.tokens, [3, 4, 1, 7]);
        assert!(!d.done());
        // Room for 6 more tokens: the step's own and at most 5 drafts (here what history has).
        assert!(d.next().len() <= 4);
        // A stop token among the accepted ends the answer there.
        let drafts = d.drafts.clone();
        let mut rows = drafts.clone();
        rows.push(0);
        if let Some(first) = rows.first_mut() {
            *first = 99;
        }
        d.accept(&rows).unwrap();
        assert!(d.stopped && d.done());
        assert_eq!(d.tokens.last(), Some(&99));
        // The output bound cuts accepted drafts and leaves no room for drafts at the end.
        let plan_3 = plan(3, 4);
        let mut d = Decoder::new(&plan_3, &[5, 5, 5]);
        d.accept(&[5]).unwrap();
        assert_eq!(d.next(), [5]);
        d.accept(&[5, 5]).unwrap();
        assert_eq!(d.tokens, [5, 5, 5]);
        assert!(d.done() && !d.stopped);
        let plan_2 = plan(2, 4);
        let mut d = Decoder::new(&plan_2, &[5, 5, 5]);
        d.accept(&[5]).unwrap();
        assert!(d.next().is_empty(), "one token left: the step's own");
    }
}
