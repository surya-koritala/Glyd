//! Bounded whole-request placement. Caller supplies authenticated observations and native qualification.
use crate::{CoreError, Registry};
use starlings_protocol::*;
use std::collections::{BTreeMap, BTreeSet, VecDeque};

const MAX_QUEUE: usize = 128;
const MAX_CLIENT_QUEUE: usize = 8;
const MAX_CLIENT_ACTIVE: usize = 2;
const MAX_MODEL_ACTIVE: usize = 32;
const MAX_MODEL_QUEUE: usize = 32;
const TTL_MS: u64 = 5000;

/// Trusted local qualification for an already loaded replica, not a remote capability assertion.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WarmReplica {
    pub node: NodeId,
    pub manifest: Digest,
    pub profile: String,
    pub backend: String,
    pub build: Digest,
    pub pool: String,
    pub evidence: Digest,
    /// Measured additional peak over the loaded idle service, including scratch and safety reserve.
    #[serde(
        serialize_with = "serialize_decimal",
        deserialize_with = "deserialize_decimal"
    )]
    pub request_bytes: u64,
    #[serde(
        serialize_with = "serialize_decimal",
        deserialize_with = "deserialize_decimal"
    )]
    pub measured_request_ms: u64,
    pub context_limit: u32,
    pub max_sessions: usize,
    pub accepting: bool,
}
fn serialize_decimal<S: serde::Serializer>(value: &u64, serializer: S) -> Result<S::Ok, S::Error> {
    serde::Serialize::serialize(&Decimal(*value), serializer)
}
fn deserialize_decimal<'de, D: serde::Deserializer<'de>>(deserializer: D) -> Result<u64, D::Error> {
    serde::Deserialize::deserialize(deserializer).map(|value: Decimal| value.0)
}
impl WarmReplica {
    pub fn validate(&self) -> Result<(), CoreError> {
        if self.request_bytes == 0
            || self.measured_request_ms == 0
            || !(1..=32).contains(&self.max_sessions)
            || !(1..=1_048_576).contains(&self.context_limit)
            || [&self.profile, &self.backend, &self.pool]
                .iter()
                .any(|v| v.is_empty() || v.len() > 256)
            || [&self.manifest, &self.build, &self.evidence]
                .iter()
                .any(|v| v.as_str().bytes().all(|b| b == b'0'))
        {
            return Err(CoreError::Invalid);
        }
        Ok(())
    }
}
#[derive(Clone, Debug)]
pub struct Task {
    pub id: RequestId,
    pub client: NodeId,
    pub manifest: Digest,
    pub profile: String,
    pub context_tokens: u32,
    pub deadline_ms: u64,
}
#[derive(Clone, Debug)]
pub struct Assignment {
    pub task: Task,
    pub node: NodeId,
    pub pool: String,
    pub reserved_bytes: u64,
    pub cancelled: bool,
    pub output_started: bool,
}
/// One owner serializes selection/reservation. No network or backend call occurs inside this object.
pub struct TaskRouter {
    cluster: ClusterId,
    registry: Registry,
    replicas: BTreeMap<NodeId, WarmReplica>,
    queue: VecDeque<Task>,
    active: BTreeMap<RequestId, Assignment>,
    expired: Vec<Task>,
    now: u64,
}
impl TaskRouter {
    pub fn new(cluster: ClusterId, mode: FixtureMode) -> Self {
        Self {
            cluster,
            registry: Registry::new(mode),
            replicas: BTreeMap::new(),
            queue: VecDeque::new(),
            active: BTreeMap::new(),
            expired: Vec::new(),
            now: 0,
        }
    }
    pub fn observe(&mut self, node: NodeCapabilities, now: u64) -> Result<(), CoreError> {
        if node.cluster_id != self.cluster {
            return Err(CoreError::Invalid);
        }
        self.advance(now)?;
        self.registry.observe(node, now)
    }
    /// One warm model/profile per worker for M2. Drain outstanding work before replacing it.
    pub fn qualify(&mut self, replica: WarmReplica) -> Result<(), CoreError> {
        replica.validate()?;
        if self.active.values().any(|a| a.node == replica.node) {
            return Err(CoreError::State);
        }
        if self.replicas.len() == 64 && !self.replicas.contains_key(&replica.node) {
            return Err(CoreError::Limit);
        }
        self.replicas.insert(replica.node, replica);
        Ok(())
    }
    pub fn enqueue(&mut self, task: Task, now: u64) -> Result<(), CoreError> {
        self.advance(now)?;
        if task.deadline_ms <= now
            || task.deadline_ms - now > 600_000
            || !(1..=1_048_576).contains(&task.context_tokens)
            || task.profile.is_empty()
            || task.profile.len() > 256
        {
            return Err(CoreError::Invalid);
        }
        if self.active.contains_key(&task.id)
            || self.queue.iter().any(|t| t.id == task.id)
            || self.expired.iter().any(|t| t.id == task.id)
        {
            return Err(CoreError::Conflict);
        }
        if self.queue.len() + self.expired.len() >= MAX_QUEUE
            || self
                .queue
                .iter()
                .filter(|t| t.manifest == task.manifest)
                .count()
                >= MAX_MODEL_QUEUE
            || self
                .queue
                .iter()
                .filter(|t| t.client == task.client)
                .count()
                >= MAX_CLIENT_QUEUE
        {
            return Err(CoreError::Limit);
        }
        self.queue.push_back(task);
        Ok(())
    }
    /// Oldest runnable task first; incompatible head entries cannot block another model/client.
    pub fn next(&mut self, now: u64) -> Result<Option<Assignment>, CoreError> {
        self.next_excluding(now, &BTreeMap::new())
    }
    /// Per-request connection exclusions never withdraw a healthy worker for another client.
    pub fn next_excluding(
        &mut self,
        now: u64,
        excluded: &BTreeMap<RequestId, BTreeSet<NodeId>>,
    ) -> Result<Option<Assignment>, CoreError> {
        self.advance(now)?;
        if self.active.len() >= 64 {
            return Ok(None);
        }
        // ponytail: bounded scans (128 queued/64 active); index only if measured routing cost warrants it.
        for index in 0..self.queue.len() {
            let task = &self.queue[index];
            if self
                .active
                .values()
                .filter(|a| a.task.client == task.client)
                .count()
                >= MAX_CLIENT_ACTIVE
                || self
                    .active
                    .values()
                    .filter(|a| a.task.manifest == task.manifest)
                    .count()
                    >= MAX_MODEL_ACTIVE
            {
                continue;
            }
            let mut best = None;
            for replica in self.replicas.values() {
                if !replica.accepting
                    || excluded
                        .get(&task.id)
                        .is_some_and(|nodes| nodes.contains(&replica.node))
                    || replica.manifest != task.manifest
                    || replica.profile != task.profile
                    || task.context_tokens > replica.context_limit
                {
                    continue;
                }
                let Ok(node) = self.registry.eligible(replica.node, now, TTL_MS) else {
                    continue;
                };
                let Some(backend) = node.backends.iter().find(|b| {
                    b.backend_id == replica.backend
                        && b.build_digest == replica.build
                        && b.execution_profiles.contains(&replica.profile)
                        && b.task_types.contains(&BackendTaskTypes::Generate)
                        && b.memory_pool_ids.contains(&replica.pool)
                        && task.context_tokens <= b.max_context_tokens.get()
                }) else {
                    continue;
                };
                let Some(pool) = node.memory_pools.iter().find(|p| p.pool_id == replica.pool)
                else {
                    continue;
                };
                let count = self
                    .active
                    .values()
                    .filter(|a| a.node == replica.node)
                    .count();
                if count
                    >= replica
                        .max_sessions
                        .min(backend.max_sessions.get() as usize)
                {
                    continue;
                }
                let reserved = self
                    .active
                    .values()
                    .filter(|a| a.node == replica.node && a.pool == replica.pool)
                    .try_fold(0u64, |sum, a| sum.checked_add(a.reserved_bytes));
                if reserved
                    .and_then(|n| n.checked_add(replica.request_bytes))
                    .is_none_or(|n| n > pool.available_bytes.0)
                {
                    continue;
                }
                let score = replica.measured_request_ms.saturating_mul(count as u64 + 1);
                let candidate = (score, replica.node);
                if best.is_none_or(|old| candidate < old) {
                    best = Some(candidate);
                }
            }
            if let Some((_, node)) = best {
                let task = self.queue.remove(index).ok_or(CoreError::NotFound)?;
                let replica = &self.replicas[&node];
                let assignment = Assignment {
                    task,
                    node,
                    pool: replica.pool.clone(),
                    reserved_bytes: replica.request_bytes,
                    cancelled: false,
                    output_started: false,
                };
                self.active.insert(assignment.task.id, assignment.clone());
                return Ok(Some(assignment));
            }
        }
        Ok(None)
    }
    pub fn withdraw(&mut self, node: NodeId) {
        if let Some(replica) = self.replicas.get_mut(&node) {
            replica.accepting = false;
        }
        for assignment in self.active.values_mut().filter(|a| a.node == node) {
            assignment.cancelled = true;
        }
    }
    pub fn cancel(&mut self, id: RequestId) {
        self.queue.retain(|task| task.id != id);
        if let Some(assignment) = self.active.get_mut(&id) {
            assignment.cancelled = true;
        }
    }
    pub fn output_started(&mut self, id: RequestId, now: u64) -> Result<(), CoreError> {
        self.advance(now)?;
        let assignment = self.active.get_mut(&id).ok_or(CoreError::NotFound)?;
        if assignment.cancelled {
            return Err(CoreError::OutputWithdrawn);
        }
        assignment.output_started = true;
        Ok(())
    }
    /// Local trusted completion only. Cancel/timeout never frees resources without this acknowledgement.
    pub fn drained(&mut self, id: RequestId) -> Result<Assignment, CoreError> {
        self.active.remove(&id).ok_or(CoreError::NotFound)
    }
    pub fn assignments(&self) -> impl Iterator<Item = &Assignment> {
        self.active.values()
    }
    pub fn queued(&self) -> usize {
        self.queue.len()
    }
    /// Deliver terminal queue-expiry results before admitting more work; none are silently evicted.
    pub fn take_expired(&mut self) -> Vec<Task> {
        std::mem::take(&mut self.expired)
    }
    /// Drive expiry/withdrawal even when no request is being enqueued.
    pub fn advance(&mut self, now: u64) -> Result<(), CoreError> {
        if now < self.now {
            return Err(CoreError::Clock);
        }
        self.now = now;
        self.queue.retain(|task| {
            if task.deadline_ms > now {
                true
            } else {
                self.expired.push(task.clone());
                false
            }
        });
        for assignment in self.active.values_mut() {
            let valid = self
                .registry
                .eligible(assignment.node, now, TTL_MS)
                .ok()
                .zip(self.replicas.get(&assignment.node))
                .is_some_and(|(node, replica)| {
                    replica.accepting
                        && node.backends.iter().any(|backend| {
                            backend.backend_id == replica.backend
                                && backend.build_digest == replica.build
                                && backend.execution_profiles.contains(&replica.profile)
                                && backend.task_types.contains(&BackendTaskTypes::Generate)
                                && backend.memory_pool_ids.contains(&replica.pool)
                                && backend.max_context_tokens.get()
                                    >= assignment.task.context_tokens
                        })
                });
            if assignment.task.deadline_ms <= now || !valid {
                assignment.cancelled = true;
            }
        }
        Ok(())
    }
}
