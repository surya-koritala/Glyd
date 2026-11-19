use crate::enrollment::{Member, create_record, open_database, read_record};
use crate::{Authority, Error, PeerStream, PrivateDir, Result, Role, decode, hash, now};
use ring::rand::{SecureRandom, SystemRandom};
use rusqlite::{Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use starlings_core::{Snapshot, Worker};
use starlings_protocol::{
    ControlRequest, Digest, GenerationId, InvocationId, NodeId, Operation, PlacementPlan, PlanId,
    Version,
};
use std::{
    collections::BTreeMap,
    fs::File,
    net::{Shutdown, TcpStream},
    path::Path,
    sync::{Arc, Mutex, MutexGuard, Weak},
};

const MAX_SCOPES: usize = 32;
pub(crate) const OVERLAP_SECONDS: i64 = 300;
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Action {
    Artifact,
    Infer,
    Control,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Scope {
    pub model: Digest,
    pub action: Action,
}

/// Bearer secret. Persist only in a protected file; never log or put in arguments.
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ClientToken {
    version: Version,
    secret: [u8; 32],
}
impl ClientToken {
    /// Returns the secret for an Authorization header. Never log this value.
    pub fn bearer(&self) -> String {
        self.secret.iter().map(|b| format!("{b:02x}")).collect()
    }
    pub(crate) fn from_bearer(bytes: &[u8]) -> Result<Self> {
        if bytes.len() != 64
            || !bytes
                .iter()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(b))
        {
            return Err(Error::Denied);
        }
        let mut secret = [0; 32];
        for (out, pair) in secret.iter_mut().zip(bytes.chunks_exact(2)) {
            *out = u8::from_str_radix(std::str::from_utf8(pair).map_err(|_| Error::Denied)?, 16)
                .map_err(|_| Error::Denied)?;
        }
        Ok(Self {
            version: Version,
            secret,
        })
    }

    pub fn save(&self, directory: &PrivateDir, name: &str) -> Result<()> {
        create_record(directory, name, self)
    }
    pub fn load(directory: &PrivateDir, name: &str) -> Result<Self> {
        read_record(directory, name)
    }
}
// ponytail: one policy lock for at most 64 members; shard only after measured contention.
#[derive(Clone)]
pub(crate) struct Access(Arc<Mutex<State>>);
pub(crate) struct State {
    pub database: Connection,
    directory: PrivateDir,
    cluster: starlings_protocol::ClusterId,
    connections: Vec<Weak<LiveConnection>>,
    workers: Vec<Weak<Mutex<RuntimeWorker>>>,
    client_grants: Vec<Weak<crate::streaming::Grant>>,
    active: bool,
    clock: i64,
    checkpointed: std::time::Instant,
    pub(crate) draining: bool,
    pub(crate) registry: Option<crate::registry::Registry>,
    memo: std::cell::RefCell<Memo>,
    // Keep one live runtime per authority directory until the other fields close.
    _lock: FileLock,
}
/// Enrollment rows already read, kept only while the database is unchanged: a commit by another
/// connection moves `PRAGMA data_version`, a write on this one moves `total_changes`. Every
/// decision (roles, scopes, certificate and token times) is still made from the rows on each check.
/// Under one `StateGuard` the other connections' version is read once, when the first check (or
/// `Access::lock`) looks; this connection's own writes are seen at once.
#[derive(Default)]
struct Memo {
    key: Option<(i64, u64)>,
    fresh: bool,
    members: BTreeMap<NodeId, Option<Member>>,
    previous: BTreeMap<NodeId, Option<(Member, i64)>>,
    grants: BTreeMap<NodeId, Option<Vec<u8>>>,
    tokens: BTreeMap<String, Option<(String, Vec<u8>, i64)>>,
}
pub(crate) struct LiveConnection {
    pub local: Member,
    pub peer: Member,
    socket: TcpStream,
}
pub(crate) struct RuntimeWorker {
    registry: Option<crate::RegistryFence>,
    generation: Option<crate::GenerationWatch>,
    /// GLY-190: on a stage, each live plan's control channel, which alone may drive it;
    /// `generation` stays the whole-request path's.
    stages: BTreeMap<PlanId, crate::GenerationWatch>,
    /// The plans a stage runtime holds at once.
    slots: usize,
    clock: Option<std::time::Instant>,
    worker: Worker,
    owners: BTreeMap<PlanId, (String, NodeId)>,
}
/// Owns the worker so network callers cannot bypass authorization through a raw Worker handle.
pub struct SecuredWorker {
    access: Access,
    inner: Arc<Mutex<RuntimeWorker>>,
}
/// Shared client policy access without the authority's CA or node private key.
#[derive(Clone)]
pub struct ClientAdmission(Access);
mod data;
mod stage;
pub use stage::{
    StageBackend, StageChunk, StageGrant, StageIssuer, StageRuntime, StageWorker, step_invocation,
};
mod enrolled;
pub use data::DataAdmission;
pub use enrolled::EnrolledWorker;

impl RuntimeWorker {
    fn monotonic_ms(&self, injected: u64) -> u64 {
        self.clock.map_or(injected, |origin| {
            u64::try_from(origin.elapsed().as_millis()).unwrap_or(u64::MAX)
        })
    }
    pub(crate) fn reserved_bytes(&self) -> u64 {
        self.worker.reserved_bytes()
    }
    /// The stage plan's own control channel is `stream`'s.
    pub(crate) fn owns(&self, plan: &PlacementPlan, stream: &crate::GenerationStream) -> bool {
        (self.stages.get(&plan.plan_id)).is_some_and(|w| w.same(&stream.watch()))
    }
    pub(crate) fn reconcile_registry(&mut self, now: u64) -> Result<bool> {
        let fence = self.registry.clone();
        let guard = fence.as_ref().map(crate::RegistryFence::lock).transpose()?;
        self.reconcile_registry_state(now, guard.as_deref())
    }
    fn reconcile_registry_state(
        &mut self,
        now: u64,
        fence: Option<&crate::registry::FenceState>,
    ) -> Result<bool> {
        let now = self.monotonic_ms(now);
        self.worker.advance(now).map_err(|_| Error::Invalid)?;
        if self.worker.backend_is_stopped() {
            return Ok(false);
        }
        if self
            .generation
            .as_ref()
            .is_some_and(crate::GenerationWatch::is_closed)
        {
            self.worker
                .retain_plans(|_| false)
                .map_err(|_| Error::Denied)?;
        }
        // A stage plan whose control channel closed is released; the others go on.
        let closed: Vec<PlanId> = (self.stages.iter())
            .filter(|(_, w)| w.is_closed())
            .map(|(plan, _)| *plan)
            .collect();
        if !closed.is_empty() {
            self.worker
                .retain_plans(|p| !closed.contains(&p.plan_id))
                .map_err(|_| Error::Denied)?;
            self.stages.retain(|plan, _| !closed.contains(plan));
        }
        let Some(fence) = fence else {
            return Ok(true);
        };
        let (epoch, status) = fence.status();
        if status != crate::NodeStatus::Online || epoch.number.0 < self.worker.snapshot().epoch.0 {
            self.worker
                .retain_plans(|_| false)
                .map_err(|_| Error::Denied)?;
            return Ok(false);
        }
        self.worker
            .reconcile_epoch(epoch.number.0)
            .map_err(|_| Error::Denied)
    }
}
impl Access {
    pub fn open(path: &Path, cluster: starlings_protocol::ClusterId) -> Result<Self> {
        let directory = PrivateDir::open(path)?;
        // A fresh enrollment authority may be opened for administration more than once.
        // The runtime lock is acquired lazily when trust/worker handles are published.
        let lock = match directory.open_file("runtime.lock") {
            Ok(file) => file,
            Err(_) => directory.create("runtime.lock")?,
        };
        let database = open_database(path, &directory)?;
        let transaction = database.unchecked_transaction()?;
        transaction.execute_batch("CREATE TABLE IF NOT EXISTS revoked (node TEXT PRIMARY KEY);
            CREATE TABLE IF NOT EXISTS grants (node TEXT PRIMARY KEY, scopes BLOB NOT NULL);
            CREATE TABLE IF NOT EXISTS tokens (hash TEXT PRIMARY KEY, node TEXT NOT NULL, scopes BLOB NOT NULL, expires INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS previous (node TEXT PRIMARY KEY, record BLOB NOT NULL, expires INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS settings (name TEXT PRIMARY KEY, value TEXT NOT NULL);")?;
        let clock =
            transaction.query_row("SELECT observed FROM clock WHERE id=1", [], |r| r.get(0))?;
        transaction.commit()?;
        Ok(Self(Arc::new(Mutex::new(State {
            database,
            clock,
            checkpointed: std::time::Instant::now(),
            directory,
            cluster,
            connections: vec![],
            workers: vec![],
            client_grants: vec![],
            _lock: lock_file(&lock)?,
            active: false,
            draining: false,
            registry: None,
            memo: Default::default(),
        }))))
    }
    pub(crate) fn inspect(&self) -> Result<StateGuard<'_>> {
        self.0.lock().map(StateGuard).map_err(|_| Error::Denied)
    }
    pub fn lock(&self) -> Result<StateGuard<'_>> {
        let mut state = match self.0.lock() {
            Ok(state) => state,
            Err(poisoned) => {
                poisoned.into_inner().fail_closed();
                return Err(Error::Denied);
            }
        };
        if !state.active {
            state._lock.0.try_lock().map_err(|_| Error::Denied)?;
            state.active = true;
        }
        let check = (|| {
            state.directory.open_file("enrollment.sqlite")?;
            private_fs::verify(&state._lock.0)?;
            let current = now()?;
            // (one read for the clock floor and the other connections' version)
            let (persisted, version): (i64, i64) = state.database.query_row(
                "SELECT observed, (SELECT data_version FROM pragma_data_version) FROM clock WHERE id=1",
                [],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )?;
            if current < state.clock.max(persisted) {
                return Err(Error::Expired);
            }
            state.clock = current;
            state.remember(version);
            Ok(())
        })();
        if let Err(error) = check {
            if error == Error::Busy {
                return Err(error);
            }
            state.fail_closed();
            return Err(error);
        }
        Ok(StateGuard(state))
    }
    pub fn checkpoint(&self, force: bool) -> Result<()> {
        let mut state = self.lock()?;
        if !force && state.checkpointed.elapsed() < std::time::Duration::from_secs(60) {
            return Ok(());
        }
        match state.database.execute(
            "UPDATE clock SET observed=MAX(observed,?1) WHERE id=1",
            [state.clock],
        ) {
            Ok(1) => {
                state.checkpointed = std::time::Instant::now();
                Ok(())
            }
            Err(rusqlite::Error::SqliteFailure(error, _))
                if matches!(
                    error.code,
                    rusqlite::ErrorCode::DatabaseBusy | rusqlite::ErrorCode::DatabaseLocked
                ) =>
            {
                Ok(())
            }
            _ => {
                state.fail_closed();
                Err(Error::Storage)
            }
        }
    }
    pub fn same(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
    pub fn register(
        &self,
        local: Member,
        peer: Member,
        socket: TcpStream,
    ) -> Result<Arc<LiveConnection>> {
        let mut state = self.lock()?;
        state.credential(&local)?;
        state.credential(&peer)?;
        state.connections.retain(|c| c.strong_count() > 0);
        // ponytail: bounded linear scan; use indexed connection sets if the 128-stream ceiling changes.
        if state.connections.len() >= 128 {
            return Err(Error::Capacity);
        }
        let live = Arc::new(LiveConnection {
            local,
            peer,
            socket,
        });
        state.connections.push(Arc::downgrade(&live));
        Ok(live)
    }
    pub fn check(&self, live: &LiveConnection) -> Result<()> {
        let state = self.lock()?;
        state.credential(&live.local)?;
        state.credential(&live.peer)
    }
    pub fn members(&self) -> Result<Vec<Member>> {
        let state = self.lock()?;
        let mut members = Vec::new();
        let mut query = state.database.prepare(
            "SELECT record FROM members UNION ALL SELECT record FROM previous WHERE expires>?1",
        )?;
        for record in query.query_map([now()?], |r| r.get::<_, Vec<u8>>(0))? {
            if members.len() >= 128 {
                return Err(Error::Capacity);
            }
            let member: Member = decode(&record?)?;
            if state.credential(&member).is_ok() {
                members.push(member);
            }
        }
        Ok(members)
    }
}
/// The locked enrollment state; releasing it ends the once-per-guard version read (see `Memo`).
pub(crate) struct StateGuard<'a>(MutexGuard<'a, State>);
impl std::ops::Deref for StateGuard<'_> {
    type Target = State;
    fn deref(&self) -> &State {
        &self.0
    }
}
impl std::ops::DerefMut for StateGuard<'_> {
    fn deref_mut(&mut self) -> &mut State {
        &mut self.0
    }
}
impl Drop for StateGuard<'_> {
    fn drop(&mut self) {
        self.0.memo.get_mut().fresh = false;
    }
}
impl State {
    pub(crate) fn clock_floor(&self) -> i64 {
        self.clock
    }
    pub(crate) fn fail_closed(&mut self) {
        self.close_client_grants();
        for live in self.connections.iter().filter_map(Weak::upgrade) {
            let _ = live.socket.shutdown(Shutdown::Both);
        }
        for worker in self.workers.iter().filter_map(Weak::upgrade) {
            // Poisoned worker state is never exposed again; still request cancellation.
            let mut runtime = worker
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let _ = runtime.worker.retain_plans(|_| false);
        }
    }
    fn close_client_grants(&self) {
        for grant in self.client_grants.iter().filter_map(Weak::upgrade) {
            grant.close();
        }
    }
    /// Drops the memo when the enrollment database changed since it was filled.
    fn refresh(&self) -> Result<()> {
        let memo = self.memo.borrow();
        if memo.fresh && memo.key.map(|k| k.1) == Some(self.database.total_changes()) {
            return Ok(());
        }
        drop(memo);
        let version: i64 = self
            .database
            .query_row("PRAGMA data_version", [], |r| r.get(0))?;
        self.remember(version);
        Ok(())
    }
    fn remember(&self, version: i64) {
        let key = Some((version, self.database.total_changes()));
        let mut memo = self.memo.borrow_mut();
        let rows = memo.members.len() + memo.previous.len() + memo.grants.len() + memo.tokens.len();
        // ponytail: one reset for any change, bounded at 1,024 rows; per-table invalidation if churn matters.
        if memo.key != key || rows > 1024 {
            *memo = Memo {
                key,
                ..Memo::default()
            };
        }
        memo.fresh = true;
    }
    fn member_row(&self, node: NodeId) -> Result<Option<Member>> {
        if let Some(found) = self.memo.borrow().members.get(&node) {
            return Ok(found.clone());
        }
        let record: Option<Vec<u8>> = self.database.query_row("SELECT record FROM members WHERE node=?1 AND NOT EXISTS(SELECT 1 FROM revoked WHERE node=?1)", [node.to_string()], |r| r.get(0)).optional()?;
        let member: Option<Member> = record.map(|r| decode(&r)).transpose()?;
        self.memo.borrow_mut().members.insert(node, member.clone());
        Ok(member)
    }
    fn previous_row(&self, node: NodeId) -> Result<Option<(Member, i64)>> {
        if let Some(found) = self.memo.borrow().previous.get(&node) {
            return Ok(found.clone());
        }
        let row: Option<(Vec<u8>, i64)> = self
            .database
            .query_row(
                "SELECT record,expires FROM previous WHERE node=?1",
                [node.to_string()],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .optional()?;
        let previous = row
            .map(|(record, expires)| Ok::<_, Error>((decode(&record)?, expires)))
            .transpose()?;
        self.memo
            .borrow_mut()
            .previous
            .insert(node, previous.clone());
        Ok(previous)
    }
    fn grants_row(&self, node: NodeId) -> Result<Option<Vec<u8>>> {
        if let Some(found) = self.memo.borrow().grants.get(&node) {
            return Ok(found.clone());
        }
        let bytes: Option<Vec<u8>> = self
            .database
            .query_row(
                "SELECT scopes FROM grants WHERE node=?1",
                [node.to_string()],
                |r| r.get(0),
            )
            .optional()?;
        self.memo.borrow_mut().grants.insert(node, bytes.clone());
        Ok(bytes)
    }
    fn token_row(&self, token_hash: &str) -> Result<Option<(String, Vec<u8>, i64)>> {
        if let Some(found) = self.memo.borrow().tokens.get(token_hash) {
            return Ok(found.clone());
        }
        let record: Option<(String, Vec<u8>, i64)> = self
            .database
            .query_row(
                "SELECT node,scopes,expires FROM tokens WHERE hash=?1",
                [token_hash],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
            )
            .optional()?;
        self.memo
            .borrow_mut()
            .tokens
            .insert(token_hash.to_owned(), record.clone());
        Ok(record)
    }
    pub fn member(&self, node: NodeId) -> Result<Member> {
        self.refresh()?;
        self.member_memo(node)
    }
    fn member_memo(&self, node: NodeId) -> Result<Member> {
        let member = self.member_row(node)?.ok_or(Error::Denied)?;
        if member.cluster != self.cluster {
            return Err(Error::Denied);
        }
        Ok(member)
    }
    pub fn credential(&self, member: &Member) -> Result<()> {
        self.refresh()?;
        self.credential_memo(member)
    }
    fn credential_memo(&self, member: &Member) -> Result<()> {
        let current = self.member_memo(member.node)?;
        if member.cluster != current.cluster || member.role != current.role {
            return Err(Error::Denied);
        }
        if member.certificate != current.certificate {
            let (previous, expires) = self.previous_row(member.node)?.ok_or(Error::Denied)?;
            if expires <= now()? || previous.certificate != member.certificate {
                return Err(Error::Expired);
            }
        }
        let (rest, cert) =
            x509_parser::parse_x509_certificate(&member.certificate).map_err(|_| Error::Invalid)?;
        if !rest.is_empty() || !cert.validity().is_valid() {
            return Err(Error::Expired);
        }
        Ok(())
    }
    pub fn allows(&self, node: NodeId, model: &Digest, action: Action) -> Result<()> {
        self.refresh()?;
        self.allows_memo(node, model, action)
    }
    fn allows_memo(&self, node: NodeId, model: &Digest, action: Action) -> Result<()> {
        let member = self.member_memo(node)?;
        // A valid current or overlapping certificate must still exist for this member.
        self.credential_memo(&member)?;
        if action == Action::Control && member.role != Role::Coordinator {
            return Err(Error::Denied);
        }
        let bytes = self.grants_row(node)?;
        let scopes: Vec<Scope> = decode(&bytes.ok_or(Error::Denied)?)?;
        if scopes.contains(&Scope {
            model: model.clone(),
            action,
        }) {
            Ok(())
        } else {
            Err(Error::Denied)
        }
    }
    fn token(&self, token_hash: &str, model: &Digest, action: Action) -> Result<(NodeId, i64)> {
        self.refresh()?;
        self.token_memo(token_hash, model, action)
    }
    fn token_memo(
        &self,
        token_hash: &str,
        model: &Digest,
        action: Action,
    ) -> Result<(NodeId, i64)> {
        if action == Action::Control {
            return Err(Error::Denied);
        }
        let (node, bytes, expires) = self.token_row(token_hash)?.ok_or(Error::Denied)?;
        if expires <= now()? {
            return Err(Error::Expired);
        }
        let node: NodeId = node.parse().map_err(|_| Error::Invalid)?;
        let member = self.member_memo(node)?;
        if member.role != Role::Controller {
            return Err(Error::Denied);
        }
        self.allows_memo(node, model, action)?;
        let scopes: Vec<Scope> = decode(&bytes)?;
        if scopes.contains(&Scope {
            model: model.clone(),
            action,
        }) {
            let (_, certificate) = x509_parser::parse_x509_certificate(&member.certificate)
                .map_err(|_| Error::Invalid)?;
            Ok((
                node,
                expires.min(certificate.validity().not_after.timestamp()),
            ))
        } else {
            Err(Error::Denied)
        }
    }
    fn plan(&self, plan: &PlacementPlan, owner: &(String, NodeId)) -> Result<()> {
        if plan.cluster_id != self.cluster {
            return Err(Error::Denied);
        }
        self.refresh()?;
        self.allows_memo(owner.1, &plan.manifest_digest, Action::Control)?;
        self.token_memo(&owner.0, &plan.manifest_digest, Action::Infer)?;
        for stage in &plan.stages {
            if self.registry.as_ref().is_some_and(|registry| {
                !registry.eligible(
                    stage.node_id,
                    plan.authority_epoch,
                    std::time::Instant::now(),
                )
            }) {
                return Err(Error::Expired);
            }
            if self.member_memo(stage.node_id)?.role != Role::Worker {
                return Err(Error::Denied);
            }
            self.allows_memo(stage.node_id, &plan.manifest_digest, Action::Infer)?;
        }
        Ok(())
    }
    pub fn reconcile(&mut self) -> Result<()> {
        self.client_grants.retain(|grant| grant.strong_count() > 0);
        let mut grant_error = None;
        for grant in self.client_grants.iter().filter_map(Weak::upgrade) {
            // Serializes revocation against a short nonblocking enqueue on this stream only.
            if let Err(error) = grant.refresh(|| {
                if self.draining {
                    Err(Error::Denied)
                } else {
                    self.token(&grant.token_hash, &grant.scope.model, grant.scope.action)
                        .and_then(|(client, expires)| {
                            if client == grant.client {
                                Ok(expires)
                            } else {
                                Err(Error::Denied)
                            }
                        })
                }
            }) {
                grant_error = Some(error);
            }
        }
        if let Some(error) = grant_error {
            return Err(error);
        }
        self.connections.retain(|c| c.strong_count() > 0);
        for live in self.connections.iter().filter_map(Weak::upgrade) {
            match self
                .credential(&live.local)
                .and_then(|()| self.credential(&live.peer))
            {
                Ok(()) => (),
                Err(Error::Busy) => return Err(Error::Busy),
                Err(_) => {
                    let _ = live.socket.shutdown(Shutdown::Both);
                }
            }
        }
        self.workers.retain(|w| w.strong_count() > 0);
        for worker in self.workers.iter().filter_map(Weak::upgrade) {
            let mut runtime = worker.lock().map_err(|_| Error::Denied)?;
            let RuntimeWorker { worker, owners, .. } = &mut *runtime;
            let mut busy = false;
            worker
                .retain_plans(|plan| {
                    match owners
                        .get(&plan.plan_id)
                        .map(|owner| self.plan(plan, owner))
                    {
                        Some(Ok(())) => true,
                        Some(Err(Error::Busy)) => {
                            busy = true;
                            true
                        }
                        _ => false,
                    }
                })
                .map_err(|_| Error::Denied)?;
            if busy {
                return Err(Error::Busy);
            }
        }
        Ok(())
    }
}
impl Authority {
    /// Read-only snapshot: every stored scope must still be usable, never an admission grant.
    pub fn client_token_active(&self, token: &ClientToken) -> Result<bool> {
        let state = self.access.lock()?;
        if state.draining {
            return Ok(false);
        }
        let token_hash = hash(&token.secret);
        let bytes: Option<Vec<u8>> = state
            .database
            .query_row(
                "SELECT scopes FROM tokens WHERE hash=?1 AND typeof(scopes)='blob' AND length(scopes)<=?2",
                (
                    &token_hash,
                    i64::try_from(crate::MAX_RECORD).map_err(|_| Error::Invalid)?,
                ),
                |row| row.get(0),
            )
            .optional()?;
        let Some(bytes) = bytes else {
            return Ok(false);
        };
        let scopes: Vec<Scope> = decode(&bytes)?;
        if scopes.is_empty() || scopes.len() > MAX_SCOPES {
            return Err(Error::Invalid);
        }
        for scope in scopes {
            match state.token(&token_hash, &scope.model, scope.action) {
                Ok(_) => (),
                Err(Error::Denied | Error::Expired) => return Ok(false),
                Err(error) => return Err(error),
            }
        }
        Ok(true)
    }
    /// Read the current committed allowlist; never consume tokens or update policy.
    pub fn permissions(&self, node: NodeId) -> Result<Vec<Scope>> {
        let state = self.access.lock()?;
        state.member(node)?;
        let record: Option<Vec<u8>> = state
            .database
            .query_row(
                "SELECT scopes FROM grants WHERE node=?1",
                [node.to_string()],
                |row| row.get(0),
            )
            .optional()?;
        let scopes: Vec<Scope> = match record {
            Some(bytes) => decode(&bytes)?,
            None => vec![],
        };
        if scopes.len() > MAX_SCOPES {
            return Err(Error::Capacity);
        }
        Ok(scopes)
    }
    /// Replace a member's complete model allowlist. Empty means deny all model operations.
    pub fn set_permissions(&self, node: NodeId, scopes: &[Scope]) -> Result<()> {
        if scopes.len() > MAX_SCOPES {
            return Err(Error::Capacity);
        }
        let mut state = self.access.lock()?;
        let role = state.member(node)?.role;
        if scopes
            .iter()
            .any(|s| s.action == Action::Control && role != Role::Coordinator)
        {
            return Err(Error::Denied);
        }
        state.database.execute("INSERT INTO grants VALUES (?1,?2) ON CONFLICT(node) DO UPDATE SET scopes=excluded.scopes", (node.to_string(), serde_json::to_vec(scopes)?))?;
        state.reconcile()
    }
    pub fn issue_client_token(
        &self,
        controller: NodeId,
        scopes: &[Scope],
        lifetime_seconds: u32,
    ) -> Result<ClientToken> {
        self.client_token_issuer()
            .issue_with(controller, scopes, lifetime_seconds, |_| Ok(()))
    }
    /// Share live policy access for bounded publication work, never authority signing keys.
    pub fn client_token_issuer(&self) -> ClientTokenIssuer {
        ClientTokenIssuer {
            access: self.access.clone(),
        }
    }
    /// Publish durably before activation; a failed commit can leave an inactive file.
    pub fn issue_client_token_saved(
        &self,
        controller: NodeId,
        scopes: &[Scope],
        lifetime_seconds: u32,
        directory: &PrivateDir,
        name: &str,
    ) -> Result<()> {
        self.client_token_issuer().issue_saved(
            controller,
            scopes,
            lifetime_seconds,
            directory,
            name,
        )
    }
}

/// Local privileged publication handle; no CA or device signing key is copied.
pub struct ClientTokenIssuer {
    access: Access,
}
impl ClientTokenIssuer {
    pub fn issue_saved(
        &self,
        controller: NodeId,
        scopes: &[Scope],
        lifetime_seconds: u32,
        directory: &PrivateDir,
        name: &str,
    ) -> Result<()> {
        self.issue_with(controller, scopes, lifetime_seconds, |token| {
            token.save(directory, name)
        })
        .map(|_| ())
    }
    /// The same token as `issue_saved`, kept in memory by its holder and never written out
    /// (GLY-364: the coordinator's local web UI acting for one approved client).
    pub fn issue(
        &self,
        controller: NodeId,
        scopes: &[Scope],
        lifetime_seconds: u32,
    ) -> Result<ClientToken> {
        self.issue_with(controller, scopes, lifetime_seconds, |_| Ok(()))
    }
    fn issue_with(
        &self,
        controller: NodeId,
        scopes: &[Scope],
        lifetime_seconds: u32,
        publish: impl FnOnce(&ClientToken) -> Result<()>,
    ) -> Result<ClientToken> {
        if scopes.is_empty() || scopes.len() > MAX_SCOPES || !(1..=3600).contains(&lifetime_seconds)
        {
            return Err(Error::Invalid);
        }
        let expires = now()?
            .checked_add(i64::from(lifetime_seconds))
            .ok_or(Error::Invalid)?;
        let admission = |state: &State| -> Result<()> {
            if state.draining || state.member(controller)?.role != Role::Controller {
                return Err(Error::Denied);
            }
            if state.clock_floor() >= expires {
                return Err(Error::Expired);
            }
            for scope in scopes {
                if scope.action == Action::Control {
                    return Err(Error::Denied);
                }
                state.allows(controller, &scope.model, scope.action)?;
            }
            let count: i64 = state.database.query_row(
                "SELECT count(*) FROM tokens WHERE expires>?1",
                [state.clock_floor()],
                |r| r.get(0),
            )?;
            if count >= 256 {
                return Err(Error::Capacity);
            }
            Ok(())
        };
        {
            let state = self.access.lock()?;
            admission(&state)?;
        }
        let mut secret = [0; 32];
        SystemRandom::new()
            .fill(&mut secret)
            .map_err(|_| Error::Crypto)?;
        let token = ClientToken {
            version: Version,
            secret,
        };
        // No valid row, policy mutex or SQLite transaction spans file publication.
        publish(&token)?;
        let mut state = self.access.lock()?;
        admission(&state)?;
        let current = state.clock_floor();
        let committed = (|| {
            let transaction = state
                .database
                .transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
            transaction.execute("DELETE FROM tokens WHERE expires<=?1", [current])?;
            transaction.execute(
                "INSERT INTO tokens VALUES (?1,?2,?3,?4)",
                (
                    hash(&token.secret),
                    controller.to_string(),
                    serde_json::to_vec(scopes)?,
                    expires,
                ),
            )?;
            if transaction.execute(
                "UPDATE clock SET observed=MAX(observed,?1) WHERE id=1",
                [current],
            )? != 1
            {
                return Err(Error::Storage);
            }
            transaction.commit()?;
            Ok(())
        })();
        if committed == Err(Error::Storage) {
            state.fail_closed();
        }
        committed?;
        Ok(token)
    }
}
impl Authority {
    pub fn revoke_client_token(&self, token: &ClientToken) -> Result<()> {
        let mut state = self.access.lock()?;
        state
            .database
            .execute("DELETE FROM tokens WHERE hash=?1", [hash(&token.secret)])?;
        state.reconcile()
    }
    pub fn revoke(&self, node: NodeId) -> Result<()> {
        let mut state = self.access.lock()?;
        let exists: bool = state.database.query_row(
            "SELECT EXISTS(SELECT 1 FROM members WHERE node=?1)",
            [node.to_string()],
            |r| r.get(0),
        )?;
        if !exists {
            return Err(Error::Denied);
        }
        // A tombstone also prevents future re-enrollment of this node ID.
        state.database.execute(
            "INSERT OR IGNORE INTO revoked VALUES (?1)",
            [node.to_string()],
        )?;
        state.reconcile()
    }
    /// Call from the daemon's timer (at least once a second), even with no network traffic.
    /// Reaps expired credentials/tokens; worker monotonic lease deadlines are independent.
    pub fn maintain(&self, monotonic_ms: u64) -> Result<()> {
        self.access.checkpoint(false)?;
        let mut state = self.access.lock()?;
        state.reconcile()?;
        for runtime in state.workers.iter().filter_map(Weak::upgrade) {
            runtime
                .lock()
                .map_err(|_| Error::Denied)?
                .worker
                .advance(monotonic_ms)
                .map_err(|_| Error::Invalid)?;
        }
        Ok(())
    }
    /// Stop admission, cancel owned plans, and retain allocations until local backend drain.
    /// Returns reserved bytes still awaiting completion; zero is the safe shutdown gate.
    pub fn drain(&self) -> Result<u64> {
        if !self.access.lock()?.draining {
            self.access.checkpoint(true)?;
        }
        let mut state = self.access.lock()?;
        state.draining = true;
        state.close_client_grants();
        let mut reserved = 0u64;
        for runtime in state.workers.iter().filter_map(Weak::upgrade) {
            let mut runtime = runtime.lock().map_err(|_| Error::Denied)?;
            runtime
                .worker
                .retain_plans(|_| false)
                .map_err(|_| Error::Denied)?;
            reserved = reserved
                .checked_add(runtime.worker.snapshot().reserved_bytes.0)
                .ok_or(Error::Capacity)?;
        }
        Ok(reserved)
    }
    pub fn secure_worker(&self, worker: Worker) -> Result<SecuredWorker> {
        let mut state = self.access.lock()?;
        if state.draining {
            return Err(Error::Denied);
        }
        let identity = worker.recovery();
        let snapshot = worker.snapshot();
        if !snapshot.instances.is_empty() || !snapshot.generations.is_empty() {
            return Err(Error::Denied);
        }
        if identity.cluster_id != state.cluster
            || state.member(identity.node_id)?.role != Role::Worker
        {
            return Err(Error::Denied);
        }
        state.workers.retain(|w| w.strong_count() > 0);
        if state.workers.len() >= 64 {
            return Err(Error::Capacity);
        }
        let inner = Arc::new(Mutex::new(RuntimeWorker {
            worker,
            registry: None,
            generation: None,
            stages: BTreeMap::new(),
            slots: 1,
            clock: None,
            owners: BTreeMap::new(),
        }));
        state.workers.push(Arc::downgrade(&inner));
        Ok(SecuredWorker {
            access: self.access.clone(),
            inner,
        })
    }
    pub fn client_admission(&self) -> ClientAdmission {
        ClientAdmission(self.access.clone())
    }
    /// Execute a short dispatch under policy authorization; never perform blocking I/O.
    pub fn with_client_access<T>(
        &self,
        token: &ClientToken,
        scope: &Scope,
        dispatch: impl FnOnce() -> T,
    ) -> Result<T> {
        self.client_admission().with_access(token, scope, dispatch)
    }
}
impl ClientAdmission {
    pub(crate) fn grant(
        &self,
        token: &ClientToken,
        scope: &Scope,
    ) -> Result<Arc<crate::streaming::Grant>> {
        let mut state = self.0.lock()?;
        if state.draining {
            return Err(Error::Denied);
        }
        state.client_grants.retain(|grant| grant.strong_count() > 0);
        if state.client_grants.len() >= 32 {
            return Err(Error::Capacity);
        }
        let token_hash = hash(&token.secret);
        let (client, expires) = state.token(&token_hash, &scope.model, scope.action)?;
        let grant = Arc::new(crate::streaming::Grant::new(
            token_hash,
            client,
            scope.clone(),
            expires,
        )?);
        state.client_grants.push(Arc::downgrade(&grant));
        Ok(grant)
    }
    /// Execute a short synchronous dispatch while authorization is locked. Do not perform I/O
    /// or reenter Authority methods in this callback. Streaming output must check each chunk.
    pub fn with_access<T>(
        &self,
        token: &ClientToken,
        scope: &Scope,
        dispatch: impl FnOnce() -> T,
    ) -> Result<T> {
        let state = self.0.lock()?;
        if state.draining {
            return Err(Error::Denied);
        }
        state.token(&hash(&token.secret), &scope.model, scope.action)?;
        Ok(dispatch())
    }
}
impl PeerStream {
    pub fn with_access<T>(&self, scope: &Scope, dispatch: impl FnOnce() -> T) -> Result<T> {
        let (access, live) = self.authority()?;
        let state = access.lock()?;
        if state.draining {
            return Err(Error::Denied);
        }
        state.credential(&live.local)?;
        state.credential(&live.peer)?;
        state.allows(self.peer(), &scope.model, scope.action)?;
        Ok(dispatch())
    }
}
impl SecuredWorker {
    /// Bind once, before publishing the worker. All existing authorization and output gates
    /// then also require fresh registry authority; the fence timer reaps orphaned plans.
    pub fn bind_registry(self, fence: crate::RegistryFence) -> Result<Self> {
        {
            let mut runtime = self.inner.lock().map_err(|_| Error::Denied)?;
            if runtime.registry.is_some() {
                return Err(Error::Denied);
            }
            let identity = runtime.worker.recovery();
            fence.register(&self.inner, identity.node_id, identity.cluster_id)?;
            runtime.registry = Some(fence);
        }
        Ok(self)
    }
    pub fn apply(
        &self,
        peer: &PeerStream,
        token: &ClientToken,
        request: &ControlRequest,
        monotonic_ms: u64,
    ) -> Result<bool> {
        let (access, live) = peer.authority()?;
        if !self.access.same(access) || peer.channel != crate::Channel::Control {
            return Err(Error::Denied);
        }
        let state = self.access.lock()?;
        if state.draining {
            return Err(Error::Denied);
        }
        state.credential(&live.local)?;
        state.credential(&live.peer)?;
        let mut runtime = self.inner.lock().map_err(|_| Error::Denied)?;
        // Hold the fence through dispatch/output so a new epoch cannot race this gate.
        let fence = runtime.registry.clone();
        let fence_guard = fence.as_ref().map(crate::RegistryFence::lock).transpose()?;
        if !runtime.reconcile_registry_state(monotonic_ms, fence_guard.as_deref())? {
            return Err(Error::Expired);
        }
        if runtime.worker.recovery().node_id != live.local.node {
            return Err(Error::Denied);
        }
        // Completion is trusted local backend acknowledgement, never a network instruction.
        if matches!(request.operation, Operation::CompleteStep { .. }) {
            return Err(Error::Denied);
        }
        let plan = runtime.worker.request_plan(request).map_err(|error| {
            if error == starlings_core::CoreError::ReplayExpired {
                Error::ReplayExpired
            } else {
                Error::Invalid
            }
        })?;
        let owner = (hash(&token.secret), peer.peer());
        state.plan(plan, &owner)?;
        let id = plan.plan_id;
        if let Some(existing) = runtime.owners.get(&id)
            && existing != &owner
        {
            return Err(Error::Denied);
        }
        let replay = runtime
            .worker
            .apply(request, monotonic_ms)
            .map_err(|error| {
                if error == starlings_core::CoreError::ReplayExpired {
                    Error::ReplayExpired
                } else {
                    Error::Invalid
                }
            })?;
        runtime.owners.insert(id, owner);
        if runtime.owners.len() > starlings_core::MAX_RECORDS + starlings_core::MAX_REQUESTS {
            let retained = runtime.worker.retained_plan_ids();
            runtime.owners.retain(|id, _| retained.contains(id));
        }
        Ok(replay)
    }
    /// Recheck plan membership and client authority for every delivered output chunk.
    /// The callback must be short, synchronous and must not reenter this runtime.
    pub fn with_output<T>(
        &self,
        token: &ClientToken,
        generation: GenerationId,
        monotonic_ms: u64,
        dispatch: impl FnOnce() -> T,
    ) -> Result<T> {
        let state = self.access.lock()?;
        let mut runtime = self.inner.lock().map_err(|_| Error::Denied)?;
        // Hold the fence through dispatch/output so a new epoch cannot race this gate.
        let fence = runtime.registry.clone();
        let fence_guard = fence.as_ref().map(crate::RegistryFence::lock).transpose()?;
        if !runtime.reconcile_registry_state(monotonic_ms, fence_guard.as_deref())? {
            return Err(Error::Expired);
        }
        let operation = Operation::Finish {
            generation_id: generation,
        };
        let plan = runtime
            .worker
            .operation_plan(&operation)
            .map_err(|_| Error::Denied)?;
        let owner = runtime.owners.get(&plan.plan_id).ok_or(Error::Denied)?;
        if owner.0 != hash(&token.secret) {
            return Err(Error::Denied);
        }
        state.plan(plan, owner)?;
        if !runtime.worker.can_publish(generation) {
            return Err(Error::Denied);
        }
        Ok(dispatch())
    }
    /// Local backend only. Discard output for any generation that is Cancelling/terminal.
    pub fn drained(&self, generation: GenerationId, invocation: InvocationId) -> Result<()> {
        self.inner
            .lock()
            .map_err(|_| Error::Denied)?
            .worker
            .drained(generation, invocation)
            .map_err(|_| Error::Invalid)
    }
    /// Local backend only: all process work and buffers must have stopped before this call.
    pub fn failed_and_drained(
        &self,
        generation: GenerationId,
        invocation: InvocationId,
    ) -> Result<()> {
        self.inner
            .lock()
            .map_err(|_| Error::Denied)?
            .worker
            .failed_and_drained(generation, invocation)
            .map_err(|_| Error::Invalid)
    }
    pub fn snapshot(&self) -> Result<Snapshot> {
        Ok(self
            .inner
            .lock()
            .map_err(|_| Error::Denied)?
            .worker
            .snapshot())
    }
}

// Release ownership even if a duplicated/inherited description has not closed yet.
pub(crate) struct FileLock(pub File);
impl Drop for FileLock {
    fn drop(&mut self) {
        let _ = self.0.unlock();
    }
}
#[cfg(unix)]
pub(crate) fn lock_file(file: &private_fs::PrivateFile) -> Result<FileLock> {
    use std::os::fd::AsFd;
    Ok(FileLock(File::from(file.as_fd().try_clone_to_owned()?)))
}
#[cfg(windows)]
pub(crate) fn lock_file(file: &private_fs::PrivateFile) -> Result<FileLock> {
    use std::os::windows::io::AsHandle;
    Ok(FileLock(File::from(file.as_handle().try_clone_to_owned()?)))
}
#[cfg(not(any(unix, windows)))]
pub(crate) fn lock_file(_: &private_fs::PrivateFile) -> Result<FileLock> {
    Err(Error::Storage)
}

#[cfg(test)]
mod review_regressions {
    use super::*;
    #[test]
    fn remembered_rows_never_outlive_a_write_from_this_or_another_connection() {
        let (_temp, path, mut authority, files) = crate::enrollment::tests::setup();
        let worker =
            crate::enrollment::tests::enrolled(&mut authority, &files, Role::Worker, "memo-worker");
        let model = starlings_protocol::byte_digest(b"synthetic memo model");
        let infer = [Scope {
            model: model.clone(),
            action: Action::Infer,
        }];
        let allowed = |authority: &Authority| {
            authority
                .access
                .lock()
                .unwrap()
                .allows(worker.node(), &model, Action::Infer)
        };
        authority.set_permissions(worker.node(), &infer).unwrap();
        assert_eq!(allowed(&authority), Ok(()));
        // A write on the runtime's own connection.
        authority.set_permissions(worker.node(), &[]).unwrap();
        assert_eq!(allowed(&authority), Err(Error::Denied));
        authority.set_permissions(worker.node(), &infer).unwrap();
        assert_eq!(allowed(&authority), Ok(()));
        // Another connection, as an administration command in another process would write.
        let other = Connection::open(path.join("enrollment.sqlite")).unwrap();
        let node = worker.node().to_string();
        other
            .execute("DELETE FROM grants WHERE node=?1", [&node])
            .unwrap();
        assert_eq!(allowed(&authority), Err(Error::Denied));
        other
            .execute("INSERT INTO revoked (node) VALUES (?1)", [&node])
            .unwrap();
        let state = authority.access.lock().unwrap();
        assert_eq!(state.member(worker.node()).map(|_| ()), Err(Error::Denied));
        assert_eq!(state.credential(&worker.member), Err(Error::Denied));
        drop(state);
        // A write under the same held guard is seen by the next check under it.
        other.execute("DELETE FROM revoked", []).unwrap();
        authority.set_permissions(worker.node(), &infer).unwrap();
        let state = authority.access.lock().unwrap();
        assert_eq!(state.allows(worker.node(), &model, Action::Infer), Ok(()));
        state
            .database
            .execute("DELETE FROM grants WHERE node=?1", [&node])
            .unwrap();
        assert_eq!(
            state.allows(worker.node(), &model, Action::Infer),
            Err(Error::Denied)
        );
    }
    #[test]
    fn token_activation_snapshot_is_read_only_and_file_presence_is_not_activation() {
        let (_temp, _path, mut authority, files) = crate::enrollment::tests::setup();
        let controller = crate::enrollment::tests::enrolled(
            &mut authority,
            &files,
            Role::Controller,
            "snapshot-controller",
        );
        let scopes: Vec<_> = [Action::Infer, Action::Artifact]
            .into_iter()
            .map(|action| Scope {
                model: starlings_protocol::byte_digest(b"synthetic activation snapshot"),
                action,
            })
            .collect();
        authority
            .set_permissions(controller.node(), &scopes)
            .unwrap();
        let unknown = ClientToken {
            version: Version,
            secret: [0; 32],
        };
        unknown.save(&files, "uncommitted.json").unwrap();
        assert!(!authority.client_token_active(&unknown).unwrap());
        authority
            .issue_client_token_saved(controller.node(), &scopes, 60, &files, "active.json")
            .unwrap();
        let token = ClientToken::load(&files, "active.json").unwrap();
        let changes = authority.access.lock().unwrap().database.total_changes();
        assert!(authority.client_token_active(&token).unwrap());
        assert_eq!(
            authority.access.lock().unwrap().database.total_changes(),
            changes
        );
        // A token may retain a subset of usable scopes; this snapshot requires all.
        authority
            .set_permissions(controller.node(), &scopes[..1])
            .unwrap();
        assert!(!authority.client_token_active(&token).unwrap());
        authority
            .set_permissions(controller.node(), &scopes)
            .unwrap();
        let token_hash = hash(&token.secret);
        for (bytes, expected) in [
            (vec![0; crate::MAX_RECORD + 1], Ok(false)),
            (
                serde_json::to_vec(&Vec::<Scope>::new()).unwrap(),
                Err(Error::Invalid),
            ),
            (
                serde_json::to_vec(&vec![scopes[0].clone(); MAX_SCOPES + 1]).unwrap(),
                Err(Error::Invalid),
            ),
        ] {
            authority
                .access
                .lock()
                .unwrap()
                .database
                .execute(
                    "UPDATE tokens SET scopes=?1 WHERE hash=?2",
                    (&bytes, &token_hash),
                )
                .unwrap();
            assert_eq!(authority.client_token_active(&token), expected);
        }
        authority
            .access
            .lock()
            .unwrap()
            .database
            .execute(
                "UPDATE tokens SET scopes=CAST(?1 AS TEXT) WHERE hash=?2",
                (serde_json::to_vec(&scopes).unwrap(), &token_hash),
            )
            .unwrap();
        assert!(!authority.client_token_active(&token).unwrap());
        authority
            .access
            .lock()
            .unwrap()
            .database
            .execute(
                "UPDATE tokens SET scopes=?1,expires=0 WHERE hash=?2",
                (serde_json::to_vec(&scopes).unwrap(), &token_hash),
            )
            .unwrap();
        assert!(!authority.client_token_active(&token).unwrap());
        authority
            .access
            .lock()
            .unwrap()
            .database
            .execute(
                "UPDATE tokens SET expires=?1 WHERE hash=?2",
                (now().unwrap() + 60, &token_hash),
            )
            .unwrap();
        assert!(authority.client_token_active(&token).unwrap());
        authority.revoke_client_token(&token).unwrap();
        assert!(!authority.client_token_active(&token).unwrap());
        authority
            .issue_client_token_saved(controller.node(), &scopes, 60, &files, "draining.json")
            .unwrap();
        let token = ClientToken::load(&files, "draining.json").unwrap();
        assert!(authority.client_token_active(&token).unwrap());
        authority.drain().unwrap();
        assert!(!authority.client_token_active(&token).unwrap());
    }
    #[test]
    fn access_schema_failure_rolls_back_every_new_table_and_keeps_membership() {
        for (obstruction, repair) in [
            ("CREATE INDEX tokens ON members(node)", "DROP INDEX tokens"),
            (
                "DROP TABLE clock",
                "CREATE TABLE clock (id INTEGER PRIMARY KEY CHECK(id=1), observed INTEGER NOT NULL); INSERT INTO clock VALUES (1,0)",
            ),
        ] {
            let (_temp, path, authority, _files) = crate::enrollment::tests::setup();
            let node = authority.identity().node();
            let cluster = authority.identity().cluster();
            drop(authority);
            let observer = Connection::open(path.join("enrollment.sqlite")).unwrap();
            observer
                .execute_batch(
                    "DROP TABLE revoked; DROP TABLE grants; DROP TABLE tokens;
            DROP TABLE previous; DROP TABLE settings;",
                )
                .unwrap();
            observer.execute_batch(obstruction).unwrap();
            assert!(matches!(Access::open(&path, cluster), Err(Error::Storage)));
            let created: u32 = observer.query_row(
            "SELECT count(*) FROM sqlite_schema WHERE type='table' AND name IN ('revoked','grants','tokens','previous','settings')",
            [], |r| r.get(0)).unwrap();
            assert_eq!(
                created, 0,
                "failed schema setup must not partially commit tables"
            );
            observer.execute_batch(repair).unwrap();
            let reopened = Authority::open(&path).unwrap();
            assert_eq!(reopened.identity().node(), node);
            assert_eq!(reopened.identity().cluster(), cluster);
            assert_eq!(
                observer
                    .query_row("SELECT count(*) FROM members", [], |r| r.get::<_, u32>(0))
                    .unwrap(),
                1
            );
        }
    }
    #[test]
    fn failed_token_publication_never_activates_even_when_compensation_would_be_busy() {
        let (_temp, path, mut authority, files) = crate::enrollment::tests::setup();
        let controller = crate::enrollment::tests::enrolled(
            &mut authority,
            &files,
            Role::Controller,
            "token-controller",
        );
        let scope = Scope {
            model: starlings_protocol::byte_digest(b"synthetic publication"),
            action: Action::Infer,
        };
        authority
            .set_permissions(controller.node(), std::slice::from_ref(&scope))
            .unwrap();
        use std::io::Write;
        files
            .create("occupied.json")
            .unwrap()
            .write_all(b"retained output")
            .unwrap();
        let mut writer = None;
        let mut active_during_publication = false;
        let result = authority.client_token_issuer().issue_with(
            controller.node(),
            std::slice::from_ref(&scope),
            60,
            |token| {
                let state = authority
                    .access
                    .0
                    .try_lock()
                    .expect("publication holds policy lock");
                active_during_publication = state
                    .token(&hash(&token.secret), &scope.model, scope.action)
                    .is_ok();
                drop(state);
                let database = Connection::open(path.join("enrollment.sqlite")).unwrap();
                database.execute_batch("BEGIN IMMEDIATE").unwrap();
                writer = Some(database);
                token.save(&files, "occupied.json")
            },
        );
        assert!(matches!(result, Err(Error::Storage)));
        let database = writer.as_ref().unwrap();
        let rows: i64 = database
            .query_row("SELECT count(*) FROM tokens", [], |r| r.get(0))
            .unwrap();
        assert_eq!(
            (active_during_publication, rows),
            (false, 0),
            "unpublished token must never be active or require compensating revocation"
        );
        assert_eq!(
            crate::catalog::read_bounded(files.open_file("occupied.json").unwrap()).unwrap(),
            b"retained output"
        );
    }
    #[test]
    fn token_publication_rechecks_policy_capacity_and_successful_commit() {
        for failure in ["busy", "permissions", "capacity", "drain"] {
            let (_temp, path, mut authority, files) = crate::enrollment::tests::setup();
            let controller = crate::enrollment::tests::enrolled(
                &mut authority,
                &files,
                Role::Controller,
                "pending-controller",
            );
            let scope = Scope {
                model: starlings_protocol::byte_digest(b"synthetic token admission"),
                action: Action::Infer,
            };
            authority
                .set_permissions(controller.node(), std::slice::from_ref(&scope))
                .unwrap();
            let mut writer = None;
            let issued = authority.client_token_issuer().issue_with(
                controller.node(), std::slice::from_ref(&scope), 60, |token| {
                    token.save(&files, "candidate.json")?;
                    match failure {
                        "busy" => {
                            let database = Connection::open(path.join("enrollment.sqlite")).unwrap();
                            database.execute_batch("BEGIN IMMEDIATE").unwrap();
                            writer = Some(database);
                        }
                        "permissions" => authority.set_permissions(controller.node(), &[])?,
                        "drain" => { authority.drain()?; },
                        "capacity" => {
                            authority.access.lock()?.database.execute(
                                "WITH RECURSIVE entries(n) AS (VALUES(0) UNION ALL SELECT n+1 FROM entries WHERE n<255)
                                 INSERT INTO tokens SELECT printf('%064x',n),?1,?2,?3 FROM entries",
                                (controller.node().to_string(), serde_json::to_vec(&vec![scope.clone()])?, now()? + 3600),
                            )?;
                        }
                        _ => unreachable!(),
                    }
                    Ok(())
                },
            );
            let expected = match failure {
                "busy" => Error::Busy,
                "permissions" | "drain" => Error::Denied,
                "capacity" => Error::Capacity,
                _ => unreachable!(),
            };
            assert_eq!(issued.err(), Some(expected));
            let candidate = ClientToken::load(&files, "candidate.json").unwrap();
            assert!(
                authority
                    .with_client_access(&candidate, &scope, || ())
                    .is_err()
            );
            let database = Connection::open(path.join("enrollment.sqlite")).unwrap();
            let active: i64 = database
                .query_row(
                    "SELECT count(*) FROM tokens WHERE hash=?1",
                    [hash(&candidate.secret)],
                    |r| r.get(0),
                )
                .unwrap();
            assert_eq!(
                active, 0,
                "published candidate became active after {failure}"
            );
            let rows: i64 = database
                .query_row("SELECT count(*) FROM tokens", [], |r| r.get(0))
                .unwrap();
            assert_eq!(rows, if failure == "capacity" { 256 } else { 0 });
            drop(writer);
        }
    }
    #[test]
    fn coordinator_publication_survives_failed_runtime_reconciliation() {
        let (_temp, path, mut authority, files) = crate::enrollment::tests::setup();
        let device = crate::enrollment::tests::enrolled(
            &mut authority,
            &files,
            Role::Worker,
            "poisoned-worker",
        );
        let worker = authority
            .secure_worker(
                Worker::new(
                    device.node(),
                    device.cluster(),
                    4096,
                    1,
                    starlings_protocol::FixtureMode::Simulation,
                )
                .unwrap(),
            )
            .unwrap();
        let runtime = worker.inner.clone();
        let _ = std::panic::catch_unwind(|| {
            let _guard = runtime.lock().unwrap();
            panic!("simulated backend panic");
        });
        let old = authority.identity.member.certificate.clone();
        authority.renew_coordinator().unwrap();
        assert_ne!(authority.identity.member.certificate, old);
        let current = authority.identity.member.certificate.clone();
        drop(worker);
        drop(runtime);
        drop(authority);
        assert_eq!(
            Authority::open(&path).unwrap().identity.member.certificate,
            current
        );
    }
}
