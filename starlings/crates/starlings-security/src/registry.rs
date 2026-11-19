//! Durable membership fencing and transient, monotonic resource freshness.
use crate::{
    Authority, EnrollmentReader, Error, Identity, Invitation, PendingEnrollment, PrivateDir,
    Result, Role, Trust, decode,
    enrollment::{Member, read_record},
    receive, send, transport,
};
use rusqlite::{Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use starlings_protocol::{
    ClusterId, Decimal, FixtureMode, NodeCapabilities, NodeCapabilitiesState, NodeId, SnapshotId,
    Version,
};
use std::{
    collections::BTreeMap,
    net::{SocketAddr, TcpStream},
    path::Path,
    sync::{Arc, Mutex},
    time::{Duration, Instant, SystemTime},
};
use uuid::Uuid;

pub(crate) const ALPN: &[u8] = b"starlings-registry/1";
const HEARTBEAT_INTERVAL: Duration = Duration::from_secs(2);
pub const REGISTRY_TTL: Duration = Duration::from_secs(6);
const EXCHANGE_TIME: Duration = Duration::from_secs(3);
const SAMPLE_WINDOW: u64 = 256;
/// Spread up to 64 devices across the shared 32-admissions/second listener budget.
pub fn heartbeat_delay() -> Duration {
    HEARTBEAT_INTERVAL + Duration::from_millis(u64::from(Uuid::new_v4().as_bytes()[0]) * 1000 / 255)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NodeStatus {
    Online,
    Offline,
    Draining,
    Revoked,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NodeSummary {
    pub node: NodeId,
    pub role: Role,
    pub status: NodeStatus,
    pub schedulable: bool,
    pub fresh_links: usize,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AuthorityEpoch {
    pub number: Decimal,
    pub incarnation: Uuid,
}
impl AuthorityEpoch {
    fn valid(self) -> bool {
        (self.number.0 == 0) == self.incarnation.is_nil()
    }
    fn initial() -> Self {
        Self {
            number: Decimal(0),
            incarnation: Uuid::nil(),
        }
    }
}
/// One batch from a real probe; replaying a batch never refreshes its receive time.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResourceSample {
    pub sequence: Decimal,
    pub capabilities: NodeCapabilities,
    pub links: Vec<LinkSample>,
}
/// A current diagnostic view, not another observation to replay into a registry.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResourceView {
    pub sample: ResourceSample,
    /// Original sample age plus receiver elapsed time; reading status cannot renew it.
    pub age_ms: u32,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LinkSample {
    pub peer: NodeId,
    pub age_ms: u32,
    pub rtt_us: u32,
    pub bytes_per_second: Decimal,
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Heartbeat {
    version: Version,
    boot: Decimal,
    boot_id: Uuid,
    sequence: Decimal,
    epoch: AuthorityEpoch,
    draining: bool,
    sample: Option<ResourceSample>,
    #[serde(default)]
    keep_alive: bool,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct HeartbeatReply {
    version: Version,
    boot_id: Uuid,
    sequence: Decimal,
    epoch: AuthorityEpoch,
    status: NodeStatus,
    reconciled: bool,
    #[serde(default)]
    certificate: Option<Member>,
    #[serde(default)]
    reenrollment_required: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    sample_sequence_floor: Option<Decimal>,
}
// Instant may pause across OS suspend. The wall clock can only shorten freshness:
// forward jumps expire it; backward jumps refuse it until a new authenticated exchange.
#[derive(Clone, Copy)]
struct Observed {
    monotonic: Instant,
    wall: SystemTime,
}
impl Observed {
    fn at(monotonic: Instant) -> Self {
        Self {
            monotonic,
            wall: SystemTime::now(),
        }
    }
    fn elapsed(self, now: Instant) -> Option<Duration> {
        Some(
            now.checked_duration_since(self.monotonic)?
                .max(self.wall.elapsed().ok()?),
        )
    }
}
struct Observation {
    epoch: AuthorityEpoch,
    sample_enabled: bool,
    boot: u64,
    boot_id: Uuid,
    sequence: u64,
    received: Observed,
    draining: bool,
    sample: Option<(ResourceSample, Observed)>,
}
pub(crate) struct Registry {
    pub epoch: AuthorityEpoch,
    nodes: BTreeMap<NodeId, Observation>,
    last_worker_restart: Option<Instant>,
}
fn advance_epoch(database: &Connection) -> Result<AuthorityEpoch> {
    let previous: Option<String> = database
        .query_row(
            "SELECT value FROM settings WHERE name='registry_epoch'",
            [],
            |r| r.get(0),
        )
        .optional()?;
    let previous: AuthorityEpoch = previous
        .map(|v| decode(v.as_bytes()))
        .transpose()?
        .unwrap_or_else(AuthorityEpoch::initial);
    if !previous.valid() {
        return Err(Error::Storage);
    }
    let epoch = AuthorityEpoch {
        number: Decimal(previous.number.0.checked_add(1).ok_or(Error::Capacity)?),
        incarnation: Uuid::new_v4(),
    };
    database.execute("INSERT INTO settings VALUES ('registry_epoch',?1) ON CONFLICT(name) DO UPDATE SET value=excluded.value", [serde_json::to_string(&epoch)?])?;
    Ok(epoch)
}
impl Registry {
    fn start(database: &Connection) -> Result<Self> {
        database.execute_batch("CREATE TABLE IF NOT EXISTS registry_boots (node TEXT PRIMARY KEY REFERENCES members(node), counter TEXT NOT NULL, incarnation TEXT NOT NULL, sample_counter TEXT NOT NULL);")?;
        let transaction = database.unchecked_transaction()?;
        let epoch = advance_epoch(&transaction)?;
        transaction.commit()?;
        Ok(Self {
            epoch,
            nodes: BTreeMap::new(),
            last_worker_restart: None,
        })
    }
    fn heartbeat(
        &mut self,
        database: &Connection,
        member: &Member,
        request: &Heartbeat,
        now: Instant,
    ) -> Result<HeartbeatReply> {
        if request.boot.0 == 0 || request.boot_id.is_nil() || request.sequence.0 == 0 {
            return Err(Error::Invalid);
        }
        if let Some(sample) = &request.sample {
            validate_sample(sample, member)?;
        }
        let saved: Option<(String, String)> = database
            .query_row(
                "SELECT counter,incarnation FROM registry_boots WHERE node=?1",
                [member.node.to_string()],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .optional()?;
        let previous = saved
            .map(|(counter, id)| {
                Ok::<_, Error>((
                    counter.parse::<u64>().map_err(|_| Error::Storage)?,
                    id.parse::<Uuid>().map_err(|_| Error::Storage)?,
                ))
            })
            .transpose()?;
        if let Some((counter, id)) = previous
            && (request.boot.0 < counter || (request.boot.0 == counter && request.boot_id != id))
        {
            return Ok(HeartbeatReply {
                version: Version,
                boot_id: request.boot_id,
                sequence: request.sequence,
                epoch: self.epoch,
                status: NodeStatus::Offline,
                reconciled: false,
                certificate: None,
                reenrollment_required: true,
                sample_sequence_floor: None,
            });
        }
        if previous.is_some_and(|(counter, _)| request.boot.0 > counter)
            && member.role == Role::Worker
            && self
                .last_worker_restart
                .is_some_and(|at| now.saturating_duration_since(at) < REGISTRY_TTL)
        {
            return Err(Error::Capacity);
        }
        if previous.is_none_or(|(counter, _)| request.boot.0 > counter) {
            // ponytail: a worker restart fences the whole 64-node cluster; use per-worker
            // plan incarnations only if measured restart churn warrants a protocol change.
            let transaction = database.unchecked_transaction()?;
            let epoch = if previous.is_some() && member.role == Role::Worker {
                // Invalidate before attempting durable publication: storage failure stays closed.
                for observation in self.nodes.values_mut() {
                    observation.epoch = AuthorityEpoch::initial();
                }
                advance_epoch(&transaction)?
            } else {
                self.epoch
            };
            transaction.execute("INSERT INTO registry_boots VALUES (?1,?2,?3,'0') ON CONFLICT(node) DO UPDATE SET counter=excluded.counter, incarnation=excluded.incarnation, sample_counter='0'", (member.node.to_string(), request.boot.0.to_string(), request.boot_id.to_string()))?;
            transaction.commit()?;
            if epoch != self.epoch {
                self.last_worker_restart = Some(now);
            }
            self.epoch = epoch;
        }
        let saved_sample: String = database.query_row(
            "SELECT sample_counter FROM registry_boots WHERE node=?1",
            [member.node.to_string()],
            |r| r.get(0),
        )?;
        let saved_sample = saved_sample.parse::<u64>().map_err(|_| Error::Storage)?;
        let old_sample = self
            .nodes
            .get(&member.node)
            .filter(|o| o.boot == request.boot.0)
            .and_then(|o| o.sample.as_ref());
        // After restart, the entire durable reservation is a replay tombstone.
        let sample_sequence_floor = request
            .sample
            .as_ref()
            .filter(|s| old_sample.is_none() && s.sequence.0 <= saved_sample)
            .map(|_| {
                saved_sample
                    .checked_add(1)
                    .map(Decimal)
                    .ok_or(Error::Capacity)
            })
            .transpose()?;
        let reconciled = request.epoch == self.epoch;
        if reconciled {
            if let Some(old) = self.nodes.get(&member.node) {
                if old.boot == request.boot.0
                    && (old.boot_id != request.boot_id || request.sequence.0 <= old.sequence)
                {
                    return Err(Error::Replay);
                }
                if let (Some((old_sample, _)), Some(sample)) = (&old.sample, &request.sample)
                    && old.boot == request.boot.0
                    && (sample.sequence.0 < old_sample.sequence.0
                        || (sample.sequence == old_sample.sequence && sample != old_sample))
                {
                    return Err(Error::Replay);
                }
            }
            if self.nodes.len() >= 64 && !self.nodes.contains_key(&member.node) {
                return Err(Error::Capacity);
            }
            let sample = match &request.sample {
                Some(_) if sample_sequence_floor.is_some() => None,
                Some(new) if old_sample.is_none_or(|(old, _)| new.sequence.0 > old.sequence.0) => {
                    if new.sequence.0 > saved_sample
                        && database.execute(
                            "UPDATE registry_boots SET sample_counter=?1 WHERE node=?2",
                            (
                                new.sequence.0.saturating_add(SAMPLE_WINDOW - 1).to_string(),
                                member.node.to_string(),
                            ),
                        )? != 1
                    {
                        return Err(Error::Storage);
                    }
                    Some((new.clone(), Observed::at(now)))
                }
                // Exact repeats retain their original age, including after omission.
                _ => old_sample.cloned(),
            };
            self.nodes.insert(
                member.node,
                Observation {
                    epoch: self.epoch,
                    sample_enabled: request.sample.is_some(),
                    boot: request.boot.0,
                    boot_id: request.boot_id,
                    sequence: request.sequence.0,
                    received: Observed::at(now),
                    draining: request.draining,
                    sample,
                },
            );
        }
        Ok(HeartbeatReply {
            version: Version,
            boot_id: request.boot_id,
            sequence: request.sequence,
            epoch: self.epoch,
            status: if request.draining {
                NodeStatus::Draining
            } else {
                NodeStatus::Online
            },
            reconciled,
            certificate: None,
            reenrollment_required: false,
            sample_sequence_floor,
        })
    }
    pub(crate) fn online(&self, node: NodeId, epoch: AuthorityEpoch) -> bool {
        epoch == self.epoch
            && self.nodes.get(&node).is_some_and(|o| {
                o.epoch == epoch && !o.draining && fresh(Instant::now(), o.received, 0)
            })
    }
    pub fn eligible(&self, node: NodeId, epoch: Decimal, now: Instant) -> bool {
        epoch == self.epoch.number
            && self.nodes.get(&node).is_some_and(|o| {
                o.epoch == self.epoch
                    && o.sample_enabled
                    && !o.draining
                    && fresh(now, o.received, 0)
                    && o.sample.as_ref().is_some_and(|(s, received)| {
                        s.capabilities.state == NodeCapabilitiesState::Ready
                            && !s.capabilities.backends.is_empty()
                            && fresh(now, *received, s.capabilities.sample_age_ms.get())
                    })
            })
    }
}
fn fresh(now: Instant, received: Observed, age_ms: u32) -> bool {
    received
        .elapsed(now)
        .and_then(|d| d.checked_add(Duration::from_millis(u64::from(age_ms))))
        .is_some_and(|age| age < REGISTRY_TTL)
}
fn validate_sample(sample: &ResourceSample, member: &Member) -> Result<()> {
    if member.role != Role::Worker
        || sample.sequence.0 == 0
        || sample.capabilities.node_id != member.node
        || sample.capabilities.cluster_id != member.cluster
        || sample.links.len() > 64
    {
        return Err(Error::Invalid);
    }
    starlings_protocol::encode(&sample.capabilities, FixtureMode::Reject)?;
    let mut peers = std::collections::BTreeSet::new();
    for link in &sample.links {
        if link.peer == member.node
            || link.age_ms > 60000
            || link.rtt_us == 0
            || link.rtt_us > 60_000_000
            || link.bytes_per_second.0 == 0
            || !peers.insert(link.peer)
        {
            return Err(Error::Invalid);
        }
    }
    Ok(())
}
impl Authority {
    /// Hold the existing exclusive runtime lock, commit a new epoch, then bind listeners.
    /// Capabilities and link freshness are never restored from disk.
    pub fn start_registry(&self) -> Result<AuthorityEpoch> {
        let mut state = self.access.lock()?;
        if state.registry.is_some() {
            return Err(Error::Denied);
        }
        let registry = Registry::start(&state.database)?;
        let epoch = registry.epoch;
        state.registry = Some(registry);
        state.reconcile()?;
        Ok(epoch)
    }
    pub fn registry_epoch(&self) -> Result<AuthorityEpoch> {
        Ok(self.registry_status()?.0)
    }
    /// Current epoch and policy admission withdrawal; neither attests native cleanup.
    pub fn registry_status(&self) -> Result<(AuthorityEpoch, bool)> {
        let state = self.access.lock()?;
        Ok((
            state.registry.as_ref().ok_or(Error::Denied)?.epoch,
            state.draining,
        ))
    }
    pub fn registry_nodes(&self) -> Result<Vec<NodeSummary>> {
        let state = self.access.lock()?;
        let registry = state.registry.as_ref().ok_or(Error::Denied)?;
        let now = Instant::now();
        let mut query = state.database.prepare("SELECT record,EXISTS(SELECT 1 FROM revoked WHERE revoked.node=members.node) FROM members ORDER BY node")?;
        let mut nodes = Vec::new();
        for row in query.query_map([], |r| Ok((r.get::<_, Vec<u8>>(0)?, r.get::<_, bool>(1)?)))? {
            let (record, revoked) = row?;
            let member: Member = decode(&record)?;
            if member.role == Role::Coordinator {
                continue;
            }
            if nodes.len() >= 64 {
                return Err(Error::Capacity);
            }
            let observed = registry
                .nodes
                .get(&member.node)
                .filter(|o| o.epoch == registry.epoch);
            let status = if revoked {
                NodeStatus::Revoked
            } else if state.credential(&member).is_err()
                || observed.is_none_or(|o| !fresh(now, o.received, 0))
            {
                NodeStatus::Offline
            } else if state.draining || observed.is_some_and(|o| o.draining) {
                NodeStatus::Draining
            } else {
                NodeStatus::Online
            };
            nodes.push(NodeSummary {
                node: member.node,
                role: member.role,
                status,
                schedulable: status == NodeStatus::Online
                    && registry.eligible(member.node, registry.epoch.number, now),
                fresh_links: if status == NodeStatus::Online {
                    observed
                        .filter(|o| o.sample_enabled)
                        .and_then(|o| o.sample.as_ref())
                        .map_or(0, |(s, received)| {
                            s.links
                                .iter()
                                .filter(|l| {
                                    fresh(now, *received, l.age_ms) && state.member(l.peer).is_ok()
                                })
                                .count()
                        })
                } else {
                    0
                },
            });
        }
        Ok(nodes)
    }
    /// Local admin diagnostics. No stale, revoked, draining or previous-epoch capacity is exposed.
    pub fn registry_resources(&self, node: NodeId) -> Result<Option<ResourceView>> {
        let state = self.access.lock()?;
        let member = state.member(node)?;
        state.credential(&member)?;
        if state.draining || member.role != Role::Worker {
            return Ok(None);
        }
        let registry = state.registry.as_ref().ok_or(Error::Denied)?;
        let Some(observed) = registry.nodes.get(&node) else {
            return Ok(None);
        };
        let now = Instant::now();
        if observed.epoch != registry.epoch
            || observed.draining
            || !observed.sample_enabled
            || !fresh(now, observed.received, 0)
        {
            return Ok(None);
        }
        let Some((sample, received)) = &observed.sample else {
            return Ok(None);
        };
        let age = received.elapsed(now).and_then(|elapsed| {
            elapsed.checked_add(Duration::from_millis(u64::from(
                sample.capabilities.sample_age_ms.get(),
            )))
        });
        let Some(age) = age.filter(|age| *age < Duration::from_secs(5)) else {
            return Ok(None);
        };
        Ok(Some(ResourceView {
            sample: sample.clone(),
            age_ms: age.as_millis().try_into().map_err(|_| Error::Capacity)?,
        }))
    }
}

pub enum RegistryRequest {
    Enrollment(Box<PendingEnrollment>),
    Heartbeat(Box<PendingHeartbeat>),
}
pub struct PendingHeartbeat {
    pub(crate) permit: Option<crate::listener::Permit>,
    stream: transport::Wire,
    access: crate::authorization::Access,
    member: Member,
    request: Heartbeat,
    received: Instant,
    certificate: Option<Member>,
}
impl EnrollmentReader {
    pub(crate) fn registry(self, socket: TcpStream, deadline: Instant) -> Result<RegistryRequest> {
        let (socket, node) = crate::reconnect::server(socket, &self.access, &self.identity)
            .inspect_err(|e| crate::registry_failure("server_proof", e))?;
        self.registry_admitted(socket, deadline, node)
    }
    pub(crate) fn registry_admitted(
        self,
        socket: TcpStream,
        deadline: Instant,
        node: Option<NodeId>,
    ) -> Result<RegistryRequest> {
        let mut stream = transport::registry_server(socket, &self.identity, &self.ca, deadline)
            .inspect_err(|e| crate::registry_failure("server_tls", e))?;
        if stream.protocol() == Some(transport::BOOTSTRAP) {
            if node.is_some() {
                return Err(Error::Denied);
            }
            return self
                .receive_stream(stream)
                .map(|p| RegistryRequest::Enrollment(Box::new(p)));
        }
        let certificate = transport::peer_certificate(&stream)?;
        let member = {
            let state = self.access.lock()?;
            let mut query = state.database.prepare("SELECT record FROM members UNION ALL SELECT record FROM previous WHERE expires > unixepoch()")?;
            let mut found = None;
            for row in query.query_map([], |r| r.get::<_, Vec<u8>>(0))? {
                let member: Member = decode(&row?)?;
                if member.cluster == self.identity.cluster()
                    && member.certificate == certificate
                    && matches!(member.role, Role::Worker | Role::Controller)
                {
                    found = Some(member);
                    break;
                }
            }
            found.ok_or(Error::Denied)?
        };
        if node != Some(member.node) {
            return Err(Error::Denied);
        }
        // A revoked current certificate may receive only its own revocation tombstone.
        let request =
            receive(&mut stream).inspect_err(|e| crate::registry_failure("server_frame", e))?;
        Ok(RegistryRequest::Heartbeat(Box::new(PendingHeartbeat {
            permit: None,
            stream,
            access: self.access,
            member,
            request,
            received: Instant::now(),
            certificate: None,
        })))
    }
}
type ReplyTrace = Option<Vec<(&'static str, Duration, Option<Error>)>>;

fn reply_step<T>(
    trace: &mut ReplyTrace,
    phase: &'static str,
    action: impl FnOnce() -> Result<T>,
) -> Result<T> {
    let began = trace.as_ref().map(|_| Instant::now());
    let result = action();
    if let (Some(trace), Some(began)) = (trace, began) {
        trace.push((phase, began.elapsed(), result.as_ref().err().copied()));
    }
    result
}
#[cfg(any(debug_assertions, test))]
fn write_reply_trace(trace: &ReplyTrace, failed: bool, mut output: impl std::io::Write) {
    if let Some(trace) = trace
        && (failed
            || trace
                .iter()
                .any(|(_, elapsed, _)| *elapsed >= EXCHANGE_TIME / 3))
    {
        for (phase, elapsed, error) in trace {
            let _ = writeln!(
                output,
                "REGISTRY_REPLY phase={phase} elapsed_ms={} error={error:?}",
                elapsed.as_millis()
            );
        }
    }
}

impl PendingHeartbeat {
    pub(crate) fn node(&self) -> NodeId {
        self.member.node
    }
    /// One connection owns at most one authenticated registry slot. Waiting for the next
    /// complete frame is bounded independently from the completed exchange.
    pub fn respond_next(mut self) -> Result<Option<Self>> {
        let keep = self.request.keep_alive;
        self.reply()?;
        if !keep {
            return Ok(None);
        }
        self.stream.set_deadline(Instant::now() + REGISTRY_TTL);
        self.request = receive(&mut self.stream)?;
        self.received = Instant::now();
        Ok(Some(self))
    }
    pub fn refresh(mut self, authority: &Authority) -> Result<Self> {
        if !self.access.same(&authority.access) {
            return Err(Error::Denied);
        }
        if let Ok(member) = authority.refresh_member(&self.member, false)
            && member.certificate != self.member.certificate
        {
            self.certificate = Some(member);
        }
        Ok(self)
    }
    pub fn respond(mut self) -> Result<()> {
        self.reply()
    }
    fn reply(&mut self) -> Result<()> {
        let mut trace = None;
        #[cfg(debug_assertions)]
        if std::env::var_os("STARLINGS_TEST_REGISTRY_TRACE").is_some() {
            let mut phases = Vec::with_capacity(8);
            phases.push(("dispatch_wait", self.received.elapsed(), None));
            trace = Some(phases);
        }
        let result = self.reply_measured(&mut trace);
        #[cfg(debug_assertions)]
        if let Some(trace) = &mut trace {
            trace.push((
                "receipt_to_reply_total",
                self.received.elapsed(),
                result.as_ref().err().copied(),
            ));
        }
        // The policy guard is gone before diagnostic output can block.
        #[cfg(debug_assertions)]
        write_reply_trace(&trace, result.is_err(), std::io::stderr());
        result
    }
    fn reply_measured(&mut self, trace: &mut ReplyTrace) -> Result<()> {
        let reply = {
            let mut state = reply_step(trace, "policy_lock", || self.access.lock())?;
            let (revoked, epoch) = reply_step(trace, "policy_read", || {
                let revoked: bool = state.database.query_row(
                    "SELECT EXISTS(SELECT 1 FROM revoked WHERE node=?1)",
                    [self.member.node.to_string()],
                    |r| r.get(0),
                )?;
                let epoch = state.registry.as_ref().ok_or(Error::Denied)?.epoch;
                Ok((revoked, epoch))
            })?;
            if revoked || state.draining {
                HeartbeatReply {
                    version: Version,
                    boot_id: self.request.boot_id,
                    sequence: self.request.sequence,
                    epoch,
                    status: if revoked {
                        NodeStatus::Revoked
                    } else {
                        NodeStatus::Draining
                    },
                    reconciled: false,
                    certificate: None,
                    reenrollment_required: false,
                    sample_sequence_floor: None,
                }
            } else {
                reply_step(trace, "credential_check", || state.credential(&self.member))?;
                let crate::authorization::State {
                    database, registry, ..
                } = &mut *state;
                let reply = reply_step(trace, "registry_update", || {
                    registry.as_mut().ok_or(Error::Denied)?.heartbeat(
                        database,
                        &self.member,
                        &self.request,
                        self.received,
                    )
                })?;
                reply_step(trace, "policy_reconcile", || state.reconcile())?;
                reply
            }
        };
        let mut reply = reply;
        if reply.status != NodeStatus::Revoked {
            reply.certificate = self.certificate.take();
        }
        reply_step(trace, "frame_send", || send(&mut self.stream, &reply))?;
        if let Some(member) = reply.certificate {
            self.member = member;
        }
        Ok(())
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SavedFence {
    version: Version,
    node: NodeId,
    cluster: ClusterId,
    coordinator: NodeId,
    boot: Decimal,
    epoch: AuthorityEpoch,
}
#[derive(Clone, Debug, Serialize)]
pub struct RouteSummary {
    pub destination: SocketAddr,
    pub local_address: Option<SocketAddr>,
    pub selected_interface: Option<crate::network::Interface>,
    pub authenticated: bool,
    pub age_ms: Option<u64>,
    pub outcome: String,
}
#[derive(Clone)]
pub struct RegistryFence(Arc<Mutex<FenceState>>);
pub(crate) struct FenceState {
    node: NodeId,
    cluster: ClusterId,
    worker: Option<std::sync::Weak<Mutex<crate::authorization::RuntimeWorker>>>,
    epoch: AuthorityEpoch,
    observed: Option<Observed>,
    status: NodeStatus,
    draining: bool,
    routes: Vec<(RouteSummary, Option<Observed>)>,
}
impl FenceState {
    pub(crate) fn status(&self) -> (AuthorityEpoch, NodeStatus) {
        (
            self.epoch,
            if self.status == NodeStatus::Revoked {
                NodeStatus::Revoked
            } else if self.draining {
                NodeStatus::Draining
            } else if self.observed.is_none_or(|at| !fresh(Instant::now(), at, 0)) {
                NodeStatus::Offline
            } else {
                self.status
            },
        )
    }
}
impl RegistryFence {
    pub(crate) fn same(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
    pub(crate) fn check_identity(&self, node: NodeId, cluster: ClusterId) -> Result<()> {
        let state = self.lock()?;
        if state.node != node || state.cluster != cluster {
            return Err(Error::Denied);
        }
        Ok(())
    }
    pub(crate) fn register(
        &self,
        worker: &Arc<Mutex<crate::authorization::RuntimeWorker>>,
        node: NodeId,
        cluster: ClusterId,
    ) -> Result<()> {
        let mut state = self.0.lock().map_err(|_| Error::Denied)?;
        if state.node != node || state.cluster != cluster {
            return Err(Error::Denied);
        }
        if state.worker.as_ref().is_some_and(|w| w.strong_count() > 0) {
            return Err(Error::Denied);
        }
        state.worker = Some(Arc::downgrade(worker));
        Ok(())
    }
    /// Drive even during a blocked/disconnected heartbeat; in-flight memory remains owned
    /// until the existing trusted local completion path reports it drained.
    /// EnrolledWorker samples its own clock after acquiring the runtime lock; this
    /// injected timestamp remains for the existing coordinator-local worker interface.
    pub fn maintain(&self, monotonic_ms: u64) -> Result<u64> {
        let worker = self
            .lock()?
            .worker
            .as_ref()
            .and_then(std::sync::Weak::upgrade);
        if let Some(worker) = worker {
            let mut runtime = worker.lock().map_err(|_| Error::Denied)?;
            runtime.reconcile_registry(monotonic_ms)?;
            return Ok(runtime.reserved_bytes());
        }
        Ok(0)
    }
    pub(crate) fn lock(&self) -> Result<std::sync::MutexGuard<'_, FenceState>> {
        self.0.lock().map_err(|_| Error::Denied)
    }
    pub fn status(&self) -> Result<(AuthorityEpoch, NodeStatus)> {
        Ok(self.lock()?.status())
    }
    pub fn topology(&self) -> Result<Vec<RouteSummary>> {
        self.topology_using(crate::network::selected_interface)
    }
    fn topology_using(
        &self,
        mut lookup: impl FnMut(SocketAddr) -> Option<crate::network::Interface>,
    ) -> Result<Vec<RouteSummary>> {
        let mut routes = self.lock()?.routes.clone();
        // Interface labels are diagnostic only. No OS enumeration holds the live fence.
        for (route, _) in &mut routes {
            route.selected_interface = route.local_address.and_then(&mut lookup);
        }
        Ok(routes
            .into_iter()
            .map(|(mut route, observed)| {
                // Age after all lookups: a slow diagnostic cannot publish stale authentication.
                route.age_ms = observed
                    .and_then(|at| at.elapsed(Instant::now()))
                    .and_then(|age| age.as_millis().try_into().ok());
                route.authenticated &= observed.is_some_and(|at| fresh(Instant::now(), at, 0));
                if !route.authenticated && route.outcome == "authenticated" {
                    route.outcome = "expired".into();
                }
                route
            })
            .collect())
    }
    pub fn drain(&self) -> Result<()> {
        self.0.lock().map_err(|_| Error::Denied)?.draining = true;
        Ok(())
    }
}
/// One polling worker owns its protected fence file and independent process lock.
/// No credentials, invitation secrets or resource contents are included in status output.
pub struct RegistryClient {
    directory: PrivateDir,
    identity: Identity,
    invitation: Invitation,
    saved: SavedFence,
    boot_id: Uuid,
    sequence: u64,
    sent_sample: Option<(u64, SnapshotId)>,
    sampling_floor: Option<(u64, SnapshotId)>,
    fence: RegistryFence,
    routes: Vec<SocketAddr>,
    source: Option<SocketAddr>,
    next_route: usize,
    persistent: bool,
    stream: Option<transport::Wire>,
    _lock: crate::authorization::FileLock,
}
impl RegistryClient {
    pub fn open(path: &Path, identity: Identity, invitation: Invitation) -> Result<Self> {
        invitation.verify_identity(&identity)?;
        let directory = PrivateDir::open(path)?;
        let lock = match directory.open_file("registry.lock") {
            Ok(file) => file,
            Err(private_fs::Error::Io(e)) if e.kind() == std::io::ErrorKind::NotFound => {
                directory.create("registry.lock")?
            }
            Err(error) => return Err(error.into()),
        };
        let lock = crate::authorization::lock_file(&lock)?;
        lock.0.try_lock().map_err(|_| Error::Denied)?;
        let mut saved: SavedFence = match directory.open_file("registry-fence.json") {
            Ok(_) => read_record(&directory, "registry-fence.json")?,
            Err(private_fs::Error::Io(e)) if e.kind() == std::io::ErrorKind::NotFound => {
                SavedFence {
                    version: Version,
                    node: identity.node(),
                    cluster: identity.cluster(),
                    coordinator: invitation.coordinator.node,
                    boot: Decimal(0),
                    epoch: AuthorityEpoch::initial(),
                }
            }
            Err(error) => return Err(error.into()),
        };
        if saved.node != identity.node()
            || saved.cluster != identity.cluster()
            || saved.coordinator != invitation.coordinator.node
            || !saved.epoch.valid()
        {
            return Err(Error::Denied);
        }
        saved.boot.0 = saved.boot.0.checked_add(1).ok_or(Error::Capacity)?;
        directory.write_atomic_durable("registry-fence.json", &serde_json::to_vec(&saved)?)?;
        let fence = RegistryFence(Arc::new(Mutex::new(FenceState {
            node: identity.node(),
            cluster: identity.cluster(),
            worker: None,
            epoch: saved.epoch,
            observed: None,
            status: NodeStatus::Offline,
            draining: false,
            routes: Vec::new(),
        })));
        Ok(Self {
            directory,
            _lock: lock,
            identity,
            routes: vec![invitation.endpoint],
            source: None,
            next_route: 0,
            persistent: false,
            stream: None,
            invitation,
            saved,
            boot_id: Uuid::new_v4(),
            sequence: 0,
            sent_sample: None,
            sampling_floor: None,
            fence,
        })
    }
    pub fn enable_persistent(&mut self) {
        self.persistent = true;
    }
    pub fn fence(&self) -> RegistryFence {
        self.fence.clone()
    }
    /// Use a new probe capture and a sequence at least this high after a coordinator restart.
    /// The rejected snapshot must not be relabelled as a fresh resource report.
    pub fn sampling_floor(&self) -> Option<(u64, SnapshotId)> {
        self.sampling_floor
    }
    /// Addresses come only from protected local configuration, never an advertisement.
    pub fn configure_routes(
        &mut self,
        routes: Vec<SocketAddr>,
        source: Option<SocketAddr>,
    ) -> Result<()> {
        crate::network::validate_routes(&routes, source)?;
        self.fence.lock()?.routes = routes
            .iter()
            .map(|a| {
                (
                    RouteSummary {
                        destination: *a,
                        local_address: None,
                        selected_interface: None,
                        authenticated: false,
                        age_ms: None,
                        outcome: "untried".into(),
                    },
                    None,
                )
            })
            .collect();
        self.stream = None;
        self.routes = routes;
        self.source = source;
        self.next_route = 0;
        Ok(())
    }
    /// No sample means no schedulable inference capacity. Call at most once every two seconds.
    pub fn heartbeat(&mut self, sample: Option<ResourceSample>) -> Result<()> {
        if self.fence.lock()?.routes.is_empty() {
            self.configure_routes(self.routes.clone(), self.source)?;
        }
        let index = self.next_route;
        let address = self.routes[index];
        let started = Observed::at(Instant::now());
        let mut local = None;
        let result = self.exchange(sample, address, started, &mut local);
        let summary = RouteSummary {
            destination: address,
            local_address: local,
            selected_interface: None,
            authenticated: result.is_ok(),
            age_ms: Some(0),
            outcome: match result {
                Ok(()) => "authenticated".into(),
                Err(error) => format!("refused_{error:?}").to_lowercase(),
            },
        };
        self.fence.lock()?.routes[index] = (summary, Some(started));
        if result.is_err() {
            self.stream = None;
            self.next_route = (index + 1) % self.routes.len();
        }
        result
    }
    fn exchange(
        &mut self,
        sample: Option<ResourceSample>,
        address: SocketAddr,
        started: Observed,
        local: &mut Option<SocketAddr>,
    ) -> Result<()> {
        self.sequence = self.sequence.checked_add(1).ok_or(Error::Capacity)?;
        let deadline = started.monotonic + EXCHANGE_TIME;
        let request = Heartbeat {
            version: Version,
            boot: self.saved.boot,
            boot_id: self.boot_id,
            sequence: Decimal(self.sequence),
            epoch: self.saved.epoch,
            draining: self.fence.0.lock().map_err(|_| Error::Denied)?.draining,
            sample,
            keep_alive: self.persistent,
        };
        self.sent_sample = request
            .sample
            .as_ref()
            .map(|s| (s.sequence.0, s.capabilities.snapshot_id));
        let trust = Trust {
            ca: self.invitation.ca.clone(),
            cluster: self.identity.cluster(),
            members: vec![self.invitation.coordinator.clone()],
            access: None,
        };
        let mut stream = if let Some(mut stream) = self.stream.take() {
            stream.set_deadline(deadline);
            stream
        } else {
            let socket = crate::network::connect(address, self.source, EXCHANGE_TIME)
                .inspect_err(|e| crate::registry_failure("client_connect", e))?;
            transport::client_until(
                socket,
                &trust,
                &self.invitation.coordinator,
                Some(&self.identity),
                ALPN,
                deadline,
            )
            .inspect_err(|e| crate::registry_failure("client_tls", e))?
        };
        *local = Some(stream.local_addr()?);
        send(&mut stream, &request)
            .inspect_err(|e| crate::registry_failure("client_frame_send", e))?;
        let reply: HeartbeatReply = receive(&mut stream)
            .inspect_err(|e| crate::registry_failure("client_frame_receive", e))?;
        if self.persistent && reply.status != NodeStatus::Revoked {
            self.stream = Some(stream);
        }
        self.accept(reply, started)
    }
    fn accept(&mut self, reply: HeartbeatReply, started: Observed) -> Result<()> {
        if reply.boot_id != self.boot_id
            || reply.sequence.0 != self.sequence
            || reply.epoch.number.0 == 0
            || reply.epoch.incarnation.is_nil()
            || reply.epoch.number.0 < self.saved.epoch.number.0
            || (reply.epoch.number == self.saved.epoch.number && reply.epoch != self.saved.epoch)
            || (reply.reconciled && reply.epoch != self.saved.epoch)
            || started
                .elapsed(Instant::now())
                .is_none_or(|d| d >= EXCHANGE_TIME)
        {
            return Err(Error::Replay);
        }
        if reply.reenrollment_required {
            return Err(Error::ReenrollmentRequired);
        }
        let sampling_floor = sample_floor(
            reply.sample_sequence_floor,
            self.sent_sample,
            self.sampling_floor,
        )?;
        if let Some(member) = &reply.certificate {
            if member.node != self.identity.node()
                || member.cluster != self.identity.cluster()
                || member.role != self.identity.role()
            {
                return Err(Error::Denied);
            }
            transport::verify_issued(member, &self.invitation.ca, &self.identity.key)?;
            let old = self.identity.member.clone();
            self.identity.member = member.clone();
            if let Err(error) = self.identity.replace_saved(&self.directory) {
                self.identity.member = old;
                return Err(error);
            }
        }
        let mut fence = self.fence.0.lock().map_err(|_| Error::Denied)?;
        if fence.status == NodeStatus::Revoked {
            return Err(Error::Denied);
        }
        if reply.epoch != self.saved.epoch {
            fence.observed = None;
            drop(fence);
            let previous = self.saved.epoch;
            self.saved.epoch = reply.epoch;
            if let Err(error) = self
                .directory
                .write_atomic_durable("registry-fence.json", &serde_json::to_vec(&self.saved)?)
            {
                self.saved.epoch = previous;
                return Err(error.into());
            }
            fence = self.fence.lock()?;
        }
        fence.epoch = reply.epoch;
        fence.status = reply.status;
        // Require another round trip acknowledging the new epoch before admitting anything.
        fence.observed = if reply.reconciled || reply.status == NodeStatus::Draining {
            Some(started)
        } else {
            None
        };
        self.sampling_floor = sampling_floor;
        Ok(())
    }
}

fn sample_floor(
    offered: Option<Decimal>,
    sent: Option<(u64, SnapshotId)>,
    previous: Option<(u64, SnapshotId)>,
) -> Result<Option<(u64, SnapshotId)>> {
    let Some(floor) = offered else {
        return Ok(previous);
    };
    let (sequence, snapshot) = sent.ok_or(Error::Invalid)?;
    if sequence == 0 || floor.0 <= sequence {
        return Err(Error::Invalid);
    }
    Ok(Some((floor.0.max(previous.map_or(0, |p| p.0)), snapshot)))
}

#[cfg(test)]
impl RegistryFence {
    pub(crate) fn test_online(node: NodeId, cluster: ClusterId) -> Self {
        Self::test_online_epoch(
            node,
            cluster,
            AuthorityEpoch {
                number: Decimal(1),
                incarnation: Uuid::new_v4(),
            },
        )
    }
    pub(crate) fn test_online_epoch(
        node: NodeId,
        cluster: ClusterId,
        epoch: AuthorityEpoch,
    ) -> Self {
        Self(Arc::new(Mutex::new(FenceState {
            node,
            cluster,
            worker: None,
            epoch,
            observed: Some(Observed::at(Instant::now())),
            status: NodeStatus::Online,
            draining: false,
            routes: Vec::new(),
        })))
    }
    pub(crate) fn test_transition(&self, reason: &str) {
        let mut state = self.0.lock().unwrap();
        match reason {
            "registry-epoch" => {
                state.epoch.number = Decimal(2);
                state.epoch.incarnation = Uuid::new_v4();
            }
            "registry-revoked" => state.status = NodeStatus::Revoked,
            "registry-draining" => state.draining = true,
            _ => state.observed = None,
        }
    }
    pub(crate) fn test_heartbeat(&self) {
        self.0.lock().unwrap().observed = Some(Observed::at(Instant::now()));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ListenerBudget, PeerListener,
        enrollment::tests::{enrolled, setup},
    };
    use std::{net::TcpListener, thread};

    fn request(epoch: AuthorityEpoch) -> Heartbeat {
        Heartbeat {
            version: Version,
            boot: Decimal(1),
            boot_id: Uuid::new_v4(),
            sequence: Decimal(1),
            epoch,
            draining: false,
            sample: None,
            keep_alive: false,
        }
    }
    #[test]
    fn reply_phase_output_is_opt_in_bounded_redacted_and_preserves_results() {
        let mut disabled = None;
        let mut calls = 0;
        let result = reply_step(&mut disabled, "policy_lock", || {
            calls += 1;
            Err::<(), _>(Error::Busy)
        });
        assert_eq!(calls, 1);
        assert_eq!(result, Err(Error::Busy));
        let mut bytes = Vec::new();
        write_reply_trace(&disabled, true, &mut bytes);
        assert!(bytes.is_empty());
        let healthy = Some(vec![("dispatch_wait", Duration::from_millis(1), None)]);
        write_reply_trace(&healthy, false, &mut bytes);
        assert!(bytes.is_empty());
        let slow = Some(vec![
            ("dispatch_wait", Duration::from_millis(1), None),
            ("policy_lock", EXCHANGE_TIME, None),
            ("registry_update", Duration::ZERO, Some(Error::Transport)),
        ]);
        write_reply_trace(&slow, false, &mut bytes);
        assert_eq!(
            String::from_utf8(bytes.clone()).unwrap(),
            "REGISTRY_REPLY phase=dispatch_wait elapsed_ms=1 error=None\nREGISTRY_REPLY phase=policy_lock elapsed_ms=3000 error=None\nREGISTRY_REPLY phase=registry_update elapsed_ms=0 error=Some(Transport)\n"
        );
        bytes.clear();
        write_reply_trace(&healthy, true, &mut bytes);
        assert_eq!(
            String::from_utf8(bytes).unwrap(),
            "REGISTRY_REPLY phase=dispatch_wait elapsed_ms=1 error=None\n"
        );
        struct Broken;
        impl std::io::Write for Broken {
            fn write(&mut self, _: &[u8]) -> std::io::Result<usize> {
                Err(std::io::ErrorKind::BrokenPipe.into())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        write_reply_trace(&slow, true, Broken);
    }
    fn sample(member: &Member, peer: NodeId) -> ResourceSample {
        let mut capabilities: NodeCapabilities = starlings_protocol::decode(
            include_bytes!("../../../examples/node-capabilities.json"),
            FixtureMode::Simulation,
        )
        .unwrap();
        // Synthetic values exercise the production validator, never a hardware claim.
        capabilities.fixture = false;
        capabilities.backends[0].build_digest =
            starlings_protocol::byte_digest(b"registry-validator-test");
        capabilities.node_id = member.node;
        capabilities.cluster_id = member.cluster;
        ResourceSample {
            sequence: Decimal(1),
            capabilities,
            links: vec![LinkSample {
                peer,
                age_ms: 0,
                rtt_us: 100,
                bytes_per_second: Decimal(1000),
            }],
        }
    }
    fn memory_registry() -> (Connection, Registry, Member) {
        let database = Connection::open_in_memory().unwrap();
        database.execute_batch("PRAGMA foreign_keys=ON; PRAGMA temp_store=MEMORY; CREATE TABLE members (node TEXT PRIMARY KEY); CREATE TABLE settings (name TEXT PRIMARY KEY, value TEXT NOT NULL);").unwrap();
        let member = Member {
            version: Version,
            node: "00000000-0000-4000-8000-000000000901".parse().unwrap(),
            cluster: "00000000-0000-4000-8000-000000000801".parse().unwrap(),
            role: Role::Worker,
            certificate: vec![],
        };
        database
            .execute("INSERT INTO members VALUES (?1)", [member.node.to_string()])
            .unwrap();
        let registry = Registry::start(&database).unwrap();
        (database, registry, member)
    }
    #[test]
    fn resource_sample_windows_amortize_writes_before_capacity_publication() {
        let (database, mut registry, member) = memory_registry();
        let mut beat = request(registry.epoch);
        beat.sample = Some(sample(
            &member,
            "00000000-0000-4000-8000-000000000902".parse().unwrap(),
        ));
        let now = Instant::now();
        for sequence in 1..=257 {
            beat.sequence = Decimal(sequence);
            beat.sample.as_mut().unwrap().sequence = Decimal(sequence);
            let before = database.total_changes();
            registry.heartbeat(&database, &member, &beat, now).unwrap();
            if matches!(sequence, 1 | 257) {
                let reserved: String = database
                    .query_row("SELECT sample_counter FROM registry_boots", [], |r| {
                        r.get(0)
                    })
                    .unwrap();
                assert_eq!(reserved, (sequence + 255).to_string());
            }
            if sequence > 1 {
                assert_eq!(
                    database.total_changes() - before,
                    u64::from(sequence == 257),
                    "resource report {sequence} must not force a durable write inside its reserved window"
                );
            }
            assert!(registry.eligible(member.node, registry.epoch.number, now));
        }
    }
    #[test]
    fn resource_sample_windows_reject_replay_and_recover_after_lost_floor_replies() {
        let (database, mut registry, member) = memory_registry();
        let mut beat = request(registry.epoch);
        beat.sample = Some(sample(
            &member,
            "00000000-0000-4000-8000-000000000902".parse().unwrap(),
        ));
        let now = Instant::now();
        registry.heartbeat(&database, &member, &beat, now).unwrap();
        let expired = now + REGISTRY_TTL;
        beat.sequence.0 += 1;
        let before = database.total_changes();
        registry
            .heartbeat(&database, &member, &beat, expired)
            .unwrap();
        assert_eq!(database.total_changes(), before);
        assert!(!registry.eligible(member.node, registry.epoch.number, expired));
        let mut mutated = beat.clone();
        mutated.sequence.0 += 1;
        mutated.sample.as_mut().unwrap().links[0].rtt_us += 1;
        assert!(
            registry
                .heartbeat(&database, &member, &mutated, expired)
                .is_err_and(|e| e == Error::Replay)
        );
        // Simulated coordinator restart: persisted range survives, volatile observations do not.
        let mut restarted = Registry::start(&database).unwrap();
        for _ in 0..3 {
            beat.sequence.0 += 1;
            let before = database.total_changes();
            let reply = restarted
                .heartbeat(&database, &member, &beat, expired)
                .unwrap();
            assert_eq!(reply.sample_sequence_floor, Some(Decimal(257)));
            assert_eq!(database.total_changes(), before);
            assert!(!restarted.eligible(member.node, restarted.epoch.number, expired));
            beat.epoch = reply.epoch; // A lost floor response cannot strand resource admission.
        }
        beat.sample.as_mut().unwrap().sequence = Decimal(257);
        beat.sample.as_mut().unwrap().capabilities.snapshot_id =
            "00000000-0000-4000-8000-000000000911".parse().unwrap();
        for failure in ["SELECT RAISE(ABORT, 'injected')", "SELECT RAISE(IGNORE)"] {
            database.execute_batch(&format!("CREATE TEMP TRIGGER refuse_reservation BEFORE UPDATE ON registry_boots BEGIN {failure}; END;")).unwrap();
            beat.sequence.0 += 1;
            assert!(
                restarted
                    .heartbeat(&database, &member, &beat, expired)
                    .is_err_and(|e| e == Error::Storage)
            );
            assert!(!restarted.eligible(member.node, restarted.epoch.number, expired));
            let counter: String = database
                .query_row("SELECT sample_counter FROM registry_boots", [], |r| {
                    r.get(0)
                })
                .unwrap();
            assert_eq!(counter, "256");
            database
                .execute_batch("DROP TRIGGER refuse_reservation")
                .unwrap();
        }
        beat.sequence.0 += 1;
        let reply = restarted
            .heartbeat(&database, &member, &beat, expired)
            .unwrap();
        assert_eq!(reply.sample_sequence_floor, None);
        assert!(restarted.eligible(member.node, restarted.epoch.number, expired));
        let counter: String = database
            .query_row("SELECT sample_counter FROM registry_boots", [], |r| {
                r.get(0)
            })
            .unwrap();
        assert_eq!(counter, "512");
        beat.sample.as_mut().unwrap().sequence = Decimal(256);
        beat.sequence.0 += 1;
        assert!(
            restarted
                .heartbeat(&database, &member, &beat, expired)
                .is_err_and(|e| e == Error::Replay)
        );
    }
    #[test]
    fn resource_sample_windows_floor_validation_and_exhaustion_stay_closed() {
        let snapshot = "00000000-0000-4000-8000-000000000911".parse().unwrap();
        let sent = Some((1, snapshot));
        assert_eq!(sample_floor(None, sent, None).unwrap(), None);
        assert_eq!(
            sample_floor(Some(Decimal(257)), sent, None).unwrap(),
            Some((257, snapshot))
        );
        assert_eq!(
            sample_floor(None, None, Some((257, snapshot))).unwrap(),
            Some((257, snapshot))
        );
        assert_eq!(
            sample_floor(Some(Decimal(2)), sent, Some((257, snapshot))).unwrap(),
            Some((257, snapshot))
        );
        for offered in [0, 1] {
            assert_eq!(
                sample_floor(Some(Decimal(offered)), sent, None),
                Err(Error::Invalid)
            );
        }
        assert_eq!(
            sample_floor(Some(Decimal(257)), None, None),
            Err(Error::Invalid)
        );
        let (database, mut registry, member) = memory_registry();
        let mut beat = request(registry.epoch);
        beat.sample = Some(sample(
            &member,
            "00000000-0000-4000-8000-000000000902".parse().unwrap(),
        ));
        beat.sample.as_mut().unwrap().sequence = Decimal(u64::MAX);
        let now = Instant::now();
        let legacy = registry.heartbeat(&database, &member, &beat, now).unwrap();
        let mut wire = serde_json::to_value(legacy).unwrap();
        assert!(wire.get("sample_sequence_floor").is_none());
        wire["sample_sequence_floor"] = serde_json::json!("257");
        let offered: HeartbeatReply = decode(&serde_json::to_vec(&wire).unwrap()).unwrap();
        assert_eq!(offered.sample_sequence_floor, Some(Decimal(257)));
        wire["sample_sequence_floor"] = serde_json::json!(257);
        assert!(decode::<HeartbeatReply>(&serde_json::to_vec(&wire).unwrap()).is_err());
        wire["sample_sequence_floor"] = serde_json::json!("257");
        wire["unknown"] = serde_json::json!(true);
        assert!(decode::<HeartbeatReply>(&serde_json::to_vec(&wire).unwrap()).is_err());
        let mut restarted = Registry::start(&database).unwrap();
        beat.epoch = restarted.epoch;
        beat.sequence.0 += 1;
        assert!(
            restarted
                .heartbeat(&database, &member, &beat, now)
                .is_err_and(|e| e == Error::Capacity)
        );
        assert!(!restarted.eligible(member.node, restarted.epoch.number, now));
    }
    #[test]
    fn failed_epoch_publication_invalidates_observations_without_advancing_durable_boot() {
        let (_temp, _, mut authority, files) = setup();
        let worker = enrolled(&mut authority, &files, Role::Worker, "storage-failure");
        let epoch = authority.start_registry().unwrap();
        let mut beat = request(epoch);
        beat.sample = Some(sample(&worker.member, authority.identity.node()));
        let mut state = authority.access.lock().unwrap();
        let crate::authorization::State {
            database, registry, ..
        } = &mut *state;
        let registry = registry.as_mut().unwrap();
        let received = Instant::now() - REGISTRY_TTL;
        registry
            .heartbeat(database, &worker.member, &beat, received)
            .unwrap();
        assert!(!registry.eligible(worker.node(), epoch.number, Instant::now()));
        beat.sequence.0 += 1;
        beat.sample.as_mut().unwrap().sequence.0 += 1;
        registry
            .heartbeat(database, &worker.member, &beat, Instant::now())
            .unwrap();
        assert!(registry.eligible(worker.node(), epoch.number, Instant::now()));
        database.execute_batch("CREATE TEMP TRIGGER fail_epoch BEFORE UPDATE ON settings WHEN NEW.name='registry_epoch' BEGIN SELECT RAISE(ABORT, 'injected'); END;").unwrap();
        beat.boot.0 += 1;
        beat.boot_id = Uuid::new_v4();
        assert!(matches!(
            registry.heartbeat(database, &worker.member, &beat, Instant::now()),
            Err(Error::Storage)
        ));
        assert!(!registry.eligible(worker.node(), epoch.number, Instant::now()));
        assert_eq!(registry.epoch, epoch);
        let counter: String = database
            .query_row(
                "SELECT counter FROM registry_boots WHERE node=?1",
                [worker.node().to_string()],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(counter, "1");
    }

    #[test]
    fn reconnect_bypasses_full_untrusted_handshakes_and_idle_probe_queue() {
        use std::{
            io::Write,
            sync::atomic::{AtomicBool, Ordering},
        };
        let (_temp, path, mut authority, files) = setup();
        let worker = enrolled(&mut authority, &files, Role::Worker, "reconnect-worker");
        let budget = ListenerBudget::default();
        let listener = PeerListener::bind("127.0.0.1:0".parse().unwrap(), budget.clone()).unwrap();
        let ipv6 = PeerListener::bind("[::1]:0".parse().unwrap(), budget.clone()).unwrap();
        let address = listener.local_addr().unwrap();
        authority
            .invite(&files, "reconnect-pin", address, Role::Worker)
            .unwrap();
        authority.start_registry().unwrap();
        let profile = path.parent().unwrap().join("reconnect-profile");
        PrivateDir::ensure(&profile).unwrap();
        let mut client = RegistryClient::open(
            &profile,
            worker,
            Invitation::load(&files, "reconnect-pin").unwrap(),
        )
        .unwrap();
        let stopped = AtomicBool::new(false);
        struct Stop<'a>(&'a AtomicBool);
        impl Drop for Stop<'_> {
            fn drop(&mut self) {
                self.0.store(true, Ordering::Relaxed);
            }
        }
        thread::scope(|scope| {
            let _stop = Stop(&stopped);
            scope.spawn(|| {
                while !stopped.load(Ordering::Relaxed) {
                    for listener in [&listener, &ipv6] {
                        if let Some(incoming) = listener.try_accept_registry(&authority).unwrap() {
                            let reader = authority.enrollment_reader().unwrap();
                            scope.spawn(move || {
                                if let Ok(RegistryRequest::Heartbeat(pending)) =
                                    incoming.registry(reader)
                                {
                                    pending.respond().unwrap();
                                }
                            });
                        }
                    }
                    thread::sleep(Duration::from_millis(1));
                }
            });
            client.heartbeat(None).unwrap();
            client.heartbeat(None).unwrap();
            assert_eq!(client.fence.status().unwrap().1, NodeStatus::Online);
            let mut flood = Vec::new();
            for address in [address, ipv6.local_addr().unwrap()] {
                for _ in 0..4 {
                    let mut socket = TcpStream::connect(address).unwrap();
                    socket.write_all(&[22]).unwrap(); // TLS header followed by silence.
                    flood.push(socket);
                }
            }
            let until = Instant::now() + Duration::from_secs(2);
            while budget.usage().unwrap().handshakes != 8 {
                assert!(Instant::now() < until, "untrusted slots were not filled");
                thread::sleep(Duration::from_millis(1));
            }
            // Fill the proof queue too; no new idle peer may pin a proof slot.
            for _ in 0..80 {
                flood.push(TcpStream::connect(address).unwrap());
            }
            for _ in 0..4 {
                client.heartbeat(None).unwrap(); // Each exchange opens a new TLS connection.
                assert_eq!(client.fence.status().unwrap().1, NodeStatus::Online);
                assert_eq!(budget.usage().unwrap().handshakes, 8);
                assert!(budget.usage().unwrap().pending_admissions <= 64);
            }
            drop(flood);
            let until = Instant::now() + Duration::from_secs(2);
            while budget.usage().unwrap().connections != 0
                || budget.usage().unwrap().pending_admissions != 0
            {
                assert!(Instant::now() < until, "admission did not release");
                thread::sleep(Duration::from_millis(1));
            }
        });
    }

    #[test]
    #[ignore = "explicit 63-peer capacity acceptance; run separately from fast CI"]
    fn sixty_three_persistent_peers_keep_authority_under_handshake_pressure() {
        use std::io::Write;
        use std::sync::{
            Mutex,
            atomic::{AtomicBool, AtomicUsize, Ordering},
        };
        let (_temp, path, mut authority, files) = setup();
        let mut identities = Vec::new();
        for n in 0..63 {
            identities.push(enrolled(
                &mut authority,
                &files,
                Role::Worker,
                &format!("load-{n}"),
            ));
        }
        let budget = ListenerBudget::default();
        let listener = PeerListener::bind("127.0.0.1:0".parse().unwrap(), budget.clone()).unwrap();
        let endpoint = listener.local_addr().unwrap();
        let ipv6 = PeerListener::bind("[::1]:0".parse().unwrap(), budget.clone()).unwrap();
        let endpoint6 = ipv6.local_addr().unwrap();
        authority
            .invite(&files, "load-pin", endpoint, Role::Worker)
            .unwrap();
        authority.start_registry().unwrap();
        let mut clients = Vec::new();
        for (n, identity) in identities.into_iter().enumerate() {
            let profile = path.parent().unwrap().join(format!("load-profile-{n}"));
            PrivateDir::ensure(&profile).unwrap();
            let mut client = RegistryClient::open(
                &profile,
                identity,
                Invitation::load(&files, "load-pin").unwrap(),
            )
            .unwrap();
            client.enable_persistent();
            clients.push(client);
        }
        let stopped = AtomicBool::new(false);
        let ready = AtomicUsize::new(0);
        let samples = Mutex::new(Vec::<u128>::new());
        let lost = AtomicUsize::new(0);
        let reconnect = AtomicBool::new(false);
        let reconnected = AtomicUsize::new(0);
        let admin_path = path.parent().unwrap().join("load-admin");
        PrivateDir::ensure(&admin_path).unwrap();
        let admin = crate::AdminListener::bind(&admin_path, ListenerBudget::default()).unwrap();
        struct StopOnDrop<'a>(&'a AtomicBool);
        impl Drop for StopOnDrop<'_> {
            fn drop(&mut self) {
                self.0.store(true, Ordering::Relaxed);
            }
        }
        thread::scope(|scope| {
            let _stop = StopOnDrop(&stopped);
            let server = scope.spawn(|| {
                while !stopped.load(Ordering::Relaxed) {
                    for listener in [&listener, &ipv6] {
                        for _ in 0..8 {
                            match listener.try_accept_registry(&authority) {
                                Ok(Some(incoming)) => {
                                    let reader = authority.enrollment_reader().unwrap();
                                    scope.spawn(|| {
                                        if let Ok(RegistryRequest::Heartbeat(pending)) =
                                            incoming.registry(reader)
                                        {
                                            let mut pending = *pending;
                                            while let Ok(Some(next)) = pending.respond_next() {
                                                pending = next;
                                            }
                                        }
                                    });
                                }
                                Ok(None) | Err(Error::Capacity) => break,
                                Err(error) => panic!("listener failed: {error:?}"),
                            }
                        }
                    }
                    thread::sleep(Duration::from_millis(20));
                }
            });
            for mut client in clients {
                let stopped = &stopped;
                let ready = &ready;
                let samples = &samples;
                let lost = &lost;
                let reconnect = &reconnect;
                let reconnected = &reconnected;
                scope.spawn(move || {
                    let mut online = false;
                    let mut reconnecting = false;
                    let mut resumed = false;
                    while !stopped.load(Ordering::Relaxed) {
                        if reconnect.load(Ordering::Relaxed) && !reconnecting {
                            client.stream = None;
                            reconnecting = true;
                        }
                        let start = Instant::now();
                        let result = client.heartbeat(None);
                        let valid = result.is_ok()
                            && client.fence.status().unwrap().1 == NodeStatus::Online;
                        if valid && !online {
                            online = true;
                            ready.fetch_add(1, Ordering::Relaxed);
                        }
                        if online {
                            samples.lock().unwrap().push(start.elapsed().as_micros());
                            if client.fence.status().unwrap().1 != NodeStatus::Online {
                                lost.fetch_add(1, Ordering::Relaxed);
                            }
                        }
                        if reconnecting && !resumed && valid {
                            resumed = true;
                            reconnected.fetch_add(1, Ordering::Relaxed);
                        }
                        thread::sleep(if result.is_err() {
                            heartbeat_delay() / 8
                        } else if online {
                            Duration::from_millis(500)
                        } else {
                            heartbeat_delay()
                        });
                    }
                });
            }
            let start = Instant::now();
            while ready.load(Ordering::Relaxed) != 63 && start.elapsed() < Duration::from_secs(60) {
                thread::sleep(Duration::from_millis(100));
            }
            let count = ready.load(Ordering::Relaxed);
            if count != 63 {
                stopped.store(true, Ordering::Relaxed);
                panic!("only {count}/63 peers admitted");
            }
            let mut admin_us = Vec::new();
            let flood: Vec<_> = (0..32)
                .flat_map(|_| [endpoint, endpoint6])
                .filter_map(|address| {
                    let mut socket = TcpStream::connect(address).ok()?;
                    socket.write_all(&[22]).ok()?;
                    Some(socket)
                })
                .collect();
            let flood_start = Instant::now();
            while budget.usage().unwrap().handshakes < 8
                && flood_start.elapsed() < Duration::from_secs(2)
            {
                thread::sleep(Duration::from_millis(10));
            }
            assert_eq!(budget.usage().unwrap().handshakes, 8);
            reconnect.store(true, Ordering::Relaxed);
            for _ in 0..20 {
                let start = Instant::now();
                let response = scope.spawn(|| {
                    admin
                        .accept()
                        .unwrap()
                        .respond(|value: serde_json::Value| value)
                });
                let reply: serde_json::Value =
                    crate::admin_call(&admin_path, &serde_json::json!({"status":true})).unwrap();
                response.join().unwrap().unwrap();
                assert_eq!(reply, serde_json::json!({"status":true}));
                admin_us.push(start.elapsed().as_micros());
                thread::sleep(Duration::from_millis(300));
            }
            assert_eq!(
                reconnected.load(Ordering::Relaxed),
                63,
                "all peers must establish fresh TLS under the flood"
            );
            drop(flood);
            stopped.store(true, Ordering::Relaxed);
            server.join().unwrap();
            let mut values = samples.lock().unwrap().clone();
            values.sort_unstable();
            admin_us.sort_unstable();
            println!(
                "63 peers; heartbeat_samples={}; heartbeat_p95_us={}; admin_p95_us={}; lost_heartbeat_authority={}; reconnected={}",
                values.len(),
                values[values.len() * 95 / 100],
                admin_us[admin_us.len() * 95 / 100],
                lost.load(Ordering::Relaxed),
                reconnected.load(Ordering::Relaxed)
            );
            assert_eq!(lost.load(Ordering::Relaxed), 0);
        });
        assert_eq!(budget.usage().unwrap().connections, 0);
    }

    #[test]
    fn automatic_refresh_uses_worker_profile_and_original_coordinator_key_pin() {
        let (_temp, path, mut authority, files) = setup();
        let worker = enrolled(&mut authority, &files, Role::Worker, "refresh-worker");
        let listener =
            PeerListener::bind("127.0.0.1:0".parse().unwrap(), ListenerBudget::default()).unwrap();
        authority
            .invite(
                &files,
                "refresh-pin",
                listener.local_addr().unwrap(),
                Role::Worker,
            )
            .unwrap();
        let invitation = Invitation::load(&files, "refresh-pin").unwrap();
        let worker_path = path.parent().unwrap().join("device");
        let mut client = RegistryClient::open(&worker_path, worker, invitation).unwrap();
        client
            .identity
            .save(&client.directory, "identity.json")
            .unwrap();
        assert!(!worker_path.join("authority.json").exists());
        authority.start_registry().unwrap();
        let pin = authority.identity.request_fingerprint();
        for cycle in 0..4 {
            if cycle == 3 {
                client.enable_persistent();
            }
            // Exercise successive renewal thresholds without waiting seven real days.
            for identity in [&mut client.identity, &mut authority.identity] {
                let mut params = crate::enrollment::certificate_params(&identity.member).unwrap();
                params.not_after = time::OffsetDateTime::now_utc() + time::Duration::hours(1);
                identity.member.certificate = params
                    .signed_by(&identity.key, &authority.issuer)
                    .unwrap()
                    .der()
                    .to_vec();
                authority
                    .access
                    .lock()
                    .unwrap()
                    .database
                    .execute(
                        "UPDATE members SET record=?2 WHERE node=?1",
                        (
                            identity.node().to_string(),
                            serde_json::to_vec(&identity.member).unwrap(),
                        ),
                    )
                    .unwrap();
            }
            authority.refresh_coordinator().unwrap();
            assert_eq!(authority.identity.request_fingerprint(), pin);
            let old_worker_certificate = client.identity.member.certificate.clone();
            // Simulate a committed renewal whose response was lost; retry over real worker TLS.
            let issued = authority
                .refresh_member(&client.identity.member, false)
                .unwrap();
            let previous_expires: i64 = authority
                .access
                .lock()
                .unwrap()
                .database
                .query_row(
                    "SELECT expires FROM previous WHERE node=?1",
                    [client.identity.node().to_string()],
                    |row| row.get(0),
                )
                .unwrap();
            let (_, old_leaf) =
                x509_parser::parse_x509_certificate(&old_worker_certificate).unwrap();
            assert_eq!(previous_expires, old_leaf.validity().not_after.timestamp());
            assert!(previous_expires > crate::now().unwrap() + 300);
            thread::scope(|scope| {
                let reader = authority.enrollment_reader().unwrap();
                let server =
                    scope.spawn(
                        || match listener.accept().unwrap().registry(reader).unwrap() {
                            RegistryRequest::Heartbeat(p) => {
                                let pending = p.refresh(&authority).unwrap();
                                if cycle == 3 {
                                    pending.respond_next().unwrap().unwrap().respond().unwrap();
                                } else {
                                    pending.respond().unwrap();
                                }
                            }
                            _ => panic!("wrong protocol"),
                        },
                    );
                client.heartbeat(None).unwrap();
                if cycle == 3 {
                    client.heartbeat(None).unwrap();
                }
                server.join().unwrap();
            });
            assert_ne!(client.identity.member.certificate, old_worker_certificate);
            assert_eq!(client.identity.member.certificate, issued.certificate);
            assert!(client.identity.certificate_seconds_remaining().unwrap() > 6 * 86400);
            assert_eq!(
                Identity::load(&client.directory, "identity.json")
                    .unwrap()
                    .member
                    .certificate,
                issued.certificate
            );
        }
    }

    #[test]
    fn replay_sample_expiry_boot_fencing_and_durable_restart() {
        let (_temp, path, mut authority, files) = setup();
        let worker = enrolled(&mut authority, &files, Role::Worker, "registry");
        let controller = enrolled(&mut authority, &files, Role::Controller, "observer");
        let first = authority.start_registry().unwrap();
        assert_eq!(first.number.0, 1);
        assert!(authority.start_registry().is_err());
        assert!(Authority::open(&path).unwrap().start_registry().is_err());
        let now = Instant::now();
        let mut beat = request(first);
        beat.sample = Some(sample(&worker.member, authority.identity.node()));
        let mut state = authority.access.lock().unwrap();
        let crate::authorization::State {
            database, registry, ..
        } = &mut *state;
        let registry = registry.as_mut().unwrap();
        let mut client_beat = request(first);
        registry
            .heartbeat(database, &controller.member, &client_beat, now)
            .unwrap();
        client_beat.boot.0 += 1;
        client_beat.boot_id = Uuid::new_v4();
        registry
            .heartbeat(database, &controller.member, &client_beat, now)
            .unwrap();
        assert_eq!(
            registry.epoch, first,
            "client restart must not fence compute workers"
        );
        assert!(
            registry
                .heartbeat(database, &worker.member, &beat, now)
                .unwrap()
                .reconciled
        );
        assert!(registry.eligible(worker.node(), first.number, now));
        assert!(
            registry
                .heartbeat(database, &worker.member, &beat, now + REGISTRY_TTL)
                .is_err()
        );
        beat.sequence.0 += 1;
        registry
            .heartbeat(database, &worker.member, &beat, now + REGISTRY_TTL)
            .unwrap();
        assert!(
            !registry.eligible(worker.node(), first.number, now + REGISTRY_TTL),
            "heartbeats must not freshen samples"
        );
        assert!(!fresh(now + REGISTRY_TTL, Observed::at(now), 0));
        assert!(!fresh(now, Observed::at(now), 6000));
        let last_sample = beat.sample.take();
        beat.sequence.0 += 1;
        registry
            .heartbeat(database, &worker.member, &beat, now + REGISTRY_TTL)
            .unwrap();
        beat.sample = last_sample;
        beat.sequence.0 += 1;
        registry
            .heartbeat(database, &worker.member, &beat, now + REGISTRY_TTL)
            .unwrap();
        assert!(
            !registry.eligible(worker.node(), first.number, now + REGISTRY_TTL),
            "omission cannot reset replay history"
        );
        beat.sample.as_mut().unwrap().sequence.0 += 1;
        beat.sequence.0 += 1;
        registry
            .heartbeat(database, &worker.member, &beat, now + REGISTRY_TTL)
            .unwrap();
        assert!(registry.eligible(worker.node(), first.number, now + REGISTRY_TTL));
        let mut changed = beat.clone();
        changed.sequence.0 += 1;
        changed.sample.as_mut().unwrap().links[0].rtt_us += 1;
        assert!(
            registry
                .heartbeat(database, &worker.member, &changed, now + REGISTRY_TTL)
                .is_err()
        );
        let old_boot = beat.clone();
        beat.boot.0 += 1;
        beat.boot_id = Uuid::new_v4();
        let next = registry
            .heartbeat(database, &worker.member, &beat, now + REGISTRY_TTL)
            .unwrap();
        assert!(!next.reconciled);
        assert_eq!(next.epoch.number.0, 2);
        assert!(!registry.eligible(worker.node(), first.number, now + REGISTRY_TTL));
        assert!(
            registry
                .heartbeat(database, &worker.member, &old_boot, now + REGISTRY_TTL)
                .unwrap()
                .reenrollment_required
        );
        beat.epoch = next.epoch;
        beat.sequence.0 += 1;
        registry
            .heartbeat(database, &worker.member, &beat, now + REGISTRY_TTL)
            .unwrap();
        drop(state);
        drop(authority);
        let restarted = Authority::open(&path).unwrap();
        let third = restarted.start_registry().unwrap();
        assert_eq!(third.number.0, 3);
        assert_ne!(third.incarnation, next.epoch.incarnation);
        let nodes = restarted.registry_nodes().unwrap();
        assert!(
            nodes
                .iter()
                .all(|n| n.status == NodeStatus::Offline && !n.schedulable)
        );
        let mut state = restarted.access.lock().unwrap();
        let crate::authorization::State {
            database, registry, ..
        } = &mut *state;
        let registry = registry.as_mut().unwrap();
        beat.epoch = third;
        beat.sequence.0 += 1;
        let repeated = registry
            .heartbeat(database, &worker.member, &beat, now)
            .unwrap();
        assert!(
            !registry.eligible(worker.node(), third.number, now),
            "old sample survives only as a durable replay tombstone"
        );
        beat.sample.as_mut().unwrap().sequence = repeated.sample_sequence_floor.unwrap();
        beat.sample.as_mut().unwrap().capabilities.snapshot_id =
            Uuid::new_v4().to_string().parse().unwrap();
        beat.sequence.0 += 1;
        registry
            .heartbeat(database, &worker.member, &beat, now)
            .unwrap();
        assert!(registry.eligible(worker.node(), third.number, now));
        beat.draining = true;
        beat.sequence.0 += 1;
        registry
            .heartbeat(database, &worker.member, &beat, now)
            .unwrap();
        assert!(!registry.eligible(worker.node(), third.number, now));
        drop(state);
        assert_eq!(
            restarted
                .registry_nodes()
                .unwrap()
                .into_iter()
                .find(|n| n.node == worker.node())
                .unwrap()
                .status,
            NodeStatus::Draining
        );
        restarted.revoke(worker.node()).unwrap();
        assert_eq!(
            restarted
                .registry_nodes()
                .unwrap()
                .into_iter()
                .find(|n| n.node == worker.node())
                .unwrap()
                .status,
            NodeStatus::Revoked
        );
    }
    #[test]
    fn resource_diagnostics_never_refresh_a_replayed_or_withdrawn_sample() {
        let (_temp, _, mut authority, files) = setup();
        let worker = enrolled(&mut authority, &files, Role::Worker, "resource-view");
        let epoch = authority.start_registry().unwrap();
        let mut beat = request(epoch);
        beat.sample = Some(sample(&worker.member, authority.identity.node()));
        let apply = |beat: &Heartbeat| {
            let mut state = authority.access.lock().unwrap();
            let crate::authorization::State {
                database, registry, ..
            } = &mut *state;
            registry
                .as_mut()
                .unwrap()
                .heartbeat(database, &worker.member, beat, Instant::now())
                .unwrap();
        };
        apply(&beat);
        let first = authority
            .registry_resources(worker.node())
            .unwrap()
            .unwrap();
        assert!(first.age_ms < 5000);
        assert_eq!(first.sample.sequence.0, 1);
        assert!(
            authority
                .registry_resources(worker.node())
                .unwrap()
                .unwrap()
                .age_ms
                >= first.age_ms
        );
        {
            let mut state = authority.access.lock().unwrap();
            let observed = state
                .registry
                .as_mut()
                .unwrap()
                .nodes
                .get_mut(&worker.node())
                .unwrap();
            observed.sample.as_mut().unwrap().1.monotonic -= Duration::from_secs(5);
        }
        beat.sequence.0 += 1;
        apply(&beat); // Live heartbeat, identical resource sample: no renewed resource age.
        assert!(
            authority
                .registry_resources(worker.node())
                .unwrap()
                .is_none()
        );
        beat.sequence.0 += 1;
        beat.sample.as_mut().unwrap().sequence.0 += 1;
        apply(&beat);
        assert!(
            authority
                .registry_resources(worker.node())
                .unwrap()
                .is_some()
        );
        beat.sequence.0 += 1;
        beat.draining = true;
        apply(&beat);
        assert!(
            authority
                .registry_resources(worker.node())
                .unwrap()
                .is_none()
        );
        authority.revoke(worker.node()).unwrap();
        assert!(matches!(
            authority.registry_resources(worker.node()),
            Err(Error::Denied)
        ));
    }
    #[test]
    fn wall_clock_discontinuity_withdraws_fresh_resources_until_a_new_sample() {
        for forward in [false, true] {
            let (_temp, _, mut authority, files) = setup();
            let worker = enrolled(&mut authority, &files, Role::Worker, "clock-control");
            let epoch = authority.start_registry().unwrap();
            let mut beat = request(epoch);
            beat.sample = Some(sample(&worker.member, authority.identity.node()));
            let apply = |beat: &Heartbeat| {
                let mut state = authority.access.lock().unwrap();
                let crate::authorization::State {
                    database, registry, ..
                } = &mut *state;
                registry
                    .as_mut()
                    .unwrap()
                    .heartbeat(database, &worker.member, beat, Instant::now())
                    .unwrap();
            };
            let status = || {
                authority
                    .registry_nodes()
                    .unwrap()
                    .into_iter()
                    .find(|node| node.node == worker.node())
                    .unwrap()
            };
            let resources = || authority.registry_resources(worker.node()).unwrap();
            apply(&beat);
            assert!(status().schedulable);
            assert!(resources().is_some());
            {
                let mut state = authority.access.lock().unwrap();
                let observation = state
                    .registry
                    .as_mut()
                    .unwrap()
                    .nodes
                    .get_mut(&worker.node())
                    .unwrap();
                // Model wall elapsed exceeding TTL or reversing, without touching OS clocks.
                let wall = if forward {
                    SystemTime::now() - REGISTRY_TTL * 2
                } else {
                    SystemTime::now() + REGISTRY_TTL * 2
                };
                observation.received.wall = wall;
                observation.sample.as_mut().unwrap().1.wall = wall;
                assert!(observation.received.monotonic.elapsed() < REGISTRY_TTL);
            }
            assert_eq!(status().status, NodeStatus::Offline);
            assert!(!status().schedulable);
            assert!(resources().is_none());

            beat.sequence.0 += 1;
            apply(&beat); // New contact, same resource batch: membership only recovers.
            assert_eq!(status().status, NodeStatus::Online);
            assert!(!status().schedulable);
            assert!(resources().is_none());

            beat.sequence.0 += 1;
            let new_sample = beat.sample.as_mut().unwrap();
            new_sample.sequence.0 += 1;
            new_sample.capabilities.snapshot_id = Uuid::new_v4().to_string().parse().unwrap();
            new_sample.capabilities.observed_at = time::OffsetDateTime::now_utc()
                .format(&time::format_description::well_known::Rfc3339)
                .unwrap();
            apply(&beat);
            assert_eq!(authority.registry_epoch().unwrap(), epoch);
            assert!(status().schedulable);
            assert!(resources().is_some());
        }
    }

    #[test]
    fn registry_tls_requires_exact_membership_and_revocation_reaches_client() {
        let (_temp, path, mut authority, files) = setup();
        let worker = enrolled(&mut authority, &files, Role::Worker, "member");
        let node = worker.node();
        let listener =
            PeerListener::bind("127.0.0.1:0".parse().unwrap(), ListenerBudget::default()).unwrap();
        authority
            .invite(
                &files,
                "pin.json",
                listener.local_addr().unwrap(),
                Role::Worker,
            )
            .unwrap();
        let invite = Invitation::load(&files, "pin.json").unwrap();
        let worker_path = path.parent().unwrap().join("device");
        let mut client = RegistryClient::open(&worker_path, worker, invite).unwrap();
        authority.start_registry().unwrap();
        let exchange = |client: &mut RegistryClient, sample| {
            thread::scope(|s| {
                let reader = authority.enrollment_reader().unwrap();
                let server =
                    s.spawn(
                        || match listener.accept().unwrap().registry(reader).unwrap() {
                            RegistryRequest::Heartbeat(pending) => pending.respond().unwrap(),
                            _ => panic!("wrong ALPN"),
                        },
                    );
                client.heartbeat(sample).unwrap();
                server.join().unwrap();
            });
        };
        exchange(&mut client, None);
        assert_eq!(client.fence.status().unwrap().1, NodeStatus::Offline);
        exchange(&mut client, None);
        assert_eq!(client.fence.status().unwrap().1, NodeStatus::Online);
        assert_eq!(client.fence.status().unwrap().0.number, Decimal(1));
        assert!(!authority.registry_nodes().unwrap()[0].schedulable);
        let resource = sample(&client.identity.member, authority.identity.node());
        exchange(&mut client, Some(resource));
        let nodes = authority.registry_nodes().unwrap();
        assert!(nodes[0].schedulable);
        assert_eq!(nodes[0].fresh_links, 1);
        // A fresh heartbeat cannot make an invalid current credential schedulable in status.
        let mut invalid = client.identity.member.clone();
        invalid.certificate.clear();
        authority
            .access
            .lock()
            .unwrap()
            .database
            .execute(
                "UPDATE members SET record=?1 WHERE node=?2",
                (serde_json::to_vec(&invalid).unwrap(), node.to_string()),
            )
            .unwrap();
        let nodes = authority.registry_nodes().unwrap();
        assert_eq!(nodes[0].status, NodeStatus::Offline);
        assert!(!nodes[0].schedulable);
        assert!(authority.registry_resources(node).is_err());
        authority
            .access
            .lock()
            .unwrap()
            .database
            .execute(
                "UPDATE members SET record=?1 WHERE node=?2",
                (
                    serde_json::to_vec(&client.identity.member).unwrap(),
                    node.to_string(),
                ),
            )
            .unwrap();
        exchange(&mut client, None);
        let nodes = authority.registry_nodes().unwrap();
        assert!(!nodes[0].schedulable);
        assert_eq!(nodes[0].fresh_links, 0);
        authority.revoke(node).unwrap();
        exchange(&mut client, None);
        assert_eq!(client.fence.status().unwrap().1, NodeStatus::Revoked);
        let trust = authority.trust().unwrap();
        // An anonymous TLS client can use enrollment, but cannot read registry records.
        let socket = TcpListener::bind("127.0.0.1:0").unwrap();
        thread::scope(|s| {
            let reader = authority.enrollment_reader().unwrap();
            let task = s.spawn(|| {
                reader.registry(socket.accept().unwrap().0, Instant::now() + EXCHANGE_TIME)
            });
            let mut wire = transport::client(
                TcpStream::connect(socket.local_addr().unwrap()).unwrap(),
                &trust,
                &authority.identity.member,
                None,
                ALPN,
            )
            .unwrap();
            let _ = send(&mut wire, &request(authority.registry_epoch().unwrap()));
            assert!(task.join().unwrap().is_err());
        });
    }
    #[test]
    fn explicit_routes_fall_back_without_trusting_another_coordinator() {
        let (_temp, path, mut authority, files) = setup();
        let (_other_temp, _other_path, other, _) = setup();
        let worker = enrolled(&mut authority, &files, Role::Worker, "route-worker");
        let good =
            PeerListener::bind("127.0.0.1:0".parse().unwrap(), ListenerBudget::default()).unwrap();
        let bad =
            PeerListener::bind("127.0.0.1:0".parse().unwrap(), ListenerBudget::default()).unwrap();
        authority
            .invite(&files, "routes", good.local_addr().unwrap(), Role::Worker)
            .unwrap();
        let invite = Invitation::load(&files, "routes").unwrap();
        let mut client =
            RegistryClient::open(&path.parent().unwrap().join("device"), worker, invite).unwrap();
        let addresses = vec![bad.local_addr().unwrap(), good.local_addr().unwrap()];
        client
            .configure_routes(addresses.clone(), Some("127.0.0.1:0".parse().unwrap()))
            .unwrap();
        authority.start_registry().unwrap();
        thread::scope(|scope| {
            let reader = other.enrollment_reader().unwrap();
            let task = scope.spawn(|| bad.accept().unwrap().registry(reader));
            assert!(client.heartbeat(None).is_err());
            assert!(task.join().unwrap().is_err());
        });
        assert_eq!(client.fence.status().unwrap().1, NodeStatus::Offline);
        assert!(!client.fence.topology().unwrap()[0].authenticated);
        for _ in 0..2 {
            thread::scope(|scope| {
                let reader = authority.enrollment_reader().unwrap();
                let task = scope.spawn(|| match good.accept().unwrap().registry(reader).unwrap() {
                    RegistryRequest::Heartbeat(p) => p.respond().unwrap(),
                    _ => panic!("wrong protocol"),
                });
                crate::network::INTERFACE_LOOKUP_ALLOWED.with(|allowed| allowed.set(false));
                let heartbeat = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    client.heartbeat(None)
                }));
                crate::network::INTERFACE_LOOKUP_ALLOWED.with(|allowed| allowed.set(true));
                task.join().unwrap();
                heartbeat
                    .expect("heartbeat must not enumerate interfaces")
                    .unwrap();
            });
        }
        assert_eq!(client.fence.status().unwrap().1, NodeStatus::Online);
        let routes = client
            .fence
            .topology_using(|address| {
                assert!(
                    client.fence.0.try_lock().is_ok(),
                    "interface lookup held the registry fence"
                );
                crate::network::selected_interface(address)
            })
            .unwrap();
        assert!(!routes[0].authenticated);
        assert!(routes[1].authenticated);
        assert_eq!(routes[1].destination, addresses[1]);
        assert_eq!(routes[1].local_address.unwrap().ip(), addresses[1].ip());
        assert!(routes[1].selected_interface.is_some());
        client.fence.lock().unwrap().routes[1].1 =
            Some(Observed::at(Instant::now() - REGISTRY_TTL));
        assert!(!client.fence.topology().unwrap()[1].authenticated);
        assert_eq!(client.fence.topology().unwrap()[1].outcome, "expired");
        client.fence.lock().unwrap().routes[1].1 = Some(Observed::at(Instant::now()));
        let delayed = client
            .fence
            .topology_using(|_| {
                assert!(client.fence.0.try_lock().is_ok());
                thread::sleep(REGISTRY_TTL);
                None
            })
            .unwrap();
        assert!(
            !delayed[1].authenticated,
            "slow diagnostics renewed route freshness"
        );
        assert_eq!(delayed[1].outcome, "expired");
        assert!(
            client
                .configure_routes(vec!["8.8.8.8:9741".parse().unwrap()], None)
                .is_err()
        );
    }
    #[test]
    fn worker_reply_fencing_and_monotonic_deadline_cannot_be_replayed() {
        let (_temp, path, mut authority, files) = setup();
        let worker = enrolled(&mut authority, &files, Role::Worker, "fence");
        authority
            .invite(
                &files,
                "pin.json",
                "127.0.0.1:9741".parse().unwrap(),
                Role::Worker,
            )
            .unwrap();
        let invite = Invitation::load(&files, "pin.json").unwrap();
        let mut client =
            RegistryClient::open(&path.parent().unwrap().join("device"), worker, invite).unwrap();
        client.sequence = 1;
        let epoch = AuthorityEpoch {
            number: Decimal(9),
            incarnation: Uuid::new_v4(),
        };
        let reply = |epoch, reconciled, sequence| HeartbeatReply {
            version: Version,
            boot_id: client.boot_id,
            sequence: Decimal(sequence),
            epoch,
            status: NodeStatus::Online,
            reconciled,
            certificate: None,
            reenrollment_required: false,
            sample_sequence_floor: None,
        };
        let boot_id = client.boot_id;
        let first = reply(epoch, false, 1);
        client.accept(first, Observed::at(Instant::now())).unwrap();
        let reply = |epoch, reconciled, sequence| HeartbeatReply {
            version: Version,
            boot_id,
            sequence: Decimal(sequence),
            epoch,
            status: NodeStatus::Online,
            reconciled,
            certificate: None,
            reenrollment_required: false,
            sample_sequence_floor: None,
        };
        assert!(
            client
                .accept(
                    reply(
                        AuthorityEpoch {
                            number: Decimal(8),
                            ..epoch
                        },
                        true,
                        1
                    ),
                    Observed::at(Instant::now())
                )
                .is_err()
        );
        assert!(
            client
                .accept(
                    reply(
                        AuthorityEpoch {
                            incarnation: Uuid::new_v4(),
                            ..epoch
                        },
                        true,
                        1
                    ),
                    Observed::at(Instant::now())
                )
                .is_err()
        );
        assert!(
            client
                .accept(reply(epoch, true, 0), Observed::at(Instant::now()))
                .is_err()
        );
        assert!(
            client
                .accept(
                    reply(epoch, true, 1),
                    Observed::at(Instant::now() - EXCHANGE_TIME)
                )
                .is_err()
        );
        client
            .accept(reply(epoch, true, 1), Observed::at(Instant::now()))
            .unwrap();
        client.fence.0.lock().unwrap().observed = Some(Observed::at(Instant::now() - REGISTRY_TTL));
        assert_eq!(client.fence.status().unwrap().1, NodeStatus::Offline);
        assert_ne!(client.fence.status().unwrap().1, NodeStatus::Online);
        for wall in [
            SystemTime::now() - REGISTRY_TTL,
            SystemTime::now() + Duration::from_secs(1),
        ] {
            let at = Observed {
                monotonic: Instant::now(),
                wall,
            };
            assert!(
                !fresh(Instant::now(), at, 0),
                "suspend/clock jump cannot preserve sample freshness"
            );
            client.fence.0.lock().unwrap().observed = Some(at);
            assert_ne!(client.fence.status().unwrap().1, NodeStatus::Online);
        }
        let mut draining = reply(epoch, false, 1);
        draining.status = NodeStatus::Draining;
        client
            .accept(draining, Observed::at(Instant::now()))
            .unwrap();
        assert_eq!(client.fence.status().unwrap().1, NodeStatus::Draining);
        // A directory at the file name forces durable publication to fail on every OS.
        let fence_path = path.parent().unwrap().join("device/registry-fence.json");
        std::fs::remove_file(&fence_path).unwrap();
        std::fs::create_dir(&fence_path).unwrap();
        let newer = AuthorityEpoch {
            number: Decimal(epoch.number.0 + 1),
            incarnation: Uuid::new_v4(),
        };
        assert!(
            client
                .accept(reply(newer, false, 1), Observed::at(Instant::now()))
                .is_err()
        );
        assert_ne!(client.fence.status().unwrap().1, NodeStatus::Online);
        assert_eq!(client.saved.epoch, epoch);
        std::fs::remove_dir(&fence_path).unwrap();
        client
            .directory
            .write_atomic_durable(
                "registry-fence.json",
                &serde_json::to_vec(&client.saved).unwrap(),
            )
            .unwrap();
        client.fence.drain().unwrap();
        assert_eq!(client.fence.status().unwrap().1, NodeStatus::Draining);
        let saved: SavedFence = read_record(&client.directory, "registry-fence.json").unwrap();
        assert_eq!(saved.epoch, epoch);
        assert_eq!(saved.boot.0, 1);
    }
    #[test]
    fn malformed_reports_and_epoch_overflow_fail_closed() {
        let (_temp, path, mut authority, files) = setup();
        let worker = enrolled(&mut authority, &files, Role::Worker, "invalid");
        let epoch = authority.start_registry().unwrap();
        let mut report = sample(&worker.member, authority.identity.node());
        report.capabilities.fixture = true;
        assert!(validate_sample(&report, &worker.member).is_err());
        report.capabilities.fixture = false;
        report.links.push(report.links[0].clone());
        assert!(validate_sample(&report, &worker.member).is_err());
        report.links.pop();
        report.links[0].age_ms = 60001;
        assert!(validate_sample(&report, &worker.member).is_err());
        report.links[0].age_ms = 0;
        report.capabilities.node_id = authority.identity.node();
        assert!(validate_sample(&report, &worker.member).is_err());
        let mut beat = request(epoch);
        beat.boot_id = Uuid::nil();
        let mut state = authority.access.lock().unwrap();
        let crate::authorization::State {
            database, registry, ..
        } = &mut *state;
        assert!(
            registry
                .as_mut()
                .unwrap()
                .heartbeat(database, &worker.member, &beat, Instant::now())
                .is_err()
        );
        database
            .execute(
                "UPDATE settings SET value=?1 WHERE name='registry_epoch'",
                [serde_json::to_string(&AuthorityEpoch {
                    number: Decimal(u64::MAX),
                    ..epoch
                })
                .unwrap()],
            )
            .unwrap();
        drop(state);
        drop(authority);
        assert!(Authority::open(&path).unwrap().start_registry().is_err());
    }
}
