//! Pure admission/lifecycle model. Callers must authenticate authority in M1.
pub mod admission;
pub mod links;
pub mod modes;
pub mod planner;
pub mod profiles;
pub mod routing;
use serde::{Deserialize, Serialize};
use starlings_protocol::*;
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

pub const MAX_RECORDS: usize = 256;
pub const MAX_REQUESTS: usize = 4096;
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CoreError {
    Invalid,
    StaleEpoch,
    ReplayExpired,
    Conflict,
    Capacity,
    Limit,
    NotFound,
    State,
    Expired,
    Clock,
    StaleCapability,
    OutputWithdrawn,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InstanceState {
    Prepared,
    Ready,
    Releasing,
    Released,
}
#[derive(Clone, Debug)]
struct Reservation {
    plan: Arc<PlacementPlan>,
    lease: Lease,
    deadline: u64,
    state: InstanceState,
    backend: Option<BackendOwnership>,
}
#[derive(Clone, Debug)]
struct BackendOwnership {
    identity: Arc<()>,
    retained: bool,
}
/// Local cleanup ownership for one preparation, never a wire grant or native qualification.
/// Cloning or dropping this token never releases the reservation. Only its original owner
/// may acknowledge qualified complete unload; the identity cannot be serialized or recreated.
#[derive(Clone)]
pub struct BackendHold {
    lease: LeaseId,
    identity: Arc<()>,
}
#[derive(Clone, Copy, Debug)]
enum PendingInvocation {
    /// The step's ID, start, end and phase.
    Step(InvocationId, u32, u32, TensorHeaderPhase),
    Request(InvocationId),
}
impl PendingInvocation {
    fn id(&self) -> InvocationId {
        match self {
            Self::Step(id, ..) | Self::Request(id) => *id,
        }
    }
}
#[derive(Clone, Debug)]
struct RunningSession {
    record: Session,
    position: u32,
    decoded: u32,
    pending: Option<PendingInvocation>,
    next_sequence: u64,
    invocations: BTreeSet<InvocationId>,
    terminal_after_drain: Option<SessionState>,
    /// After a multi-row decode step: the lowest start the next step may have (`starts`).
    floor: Option<u32>,
}
impl RunningSession {
    /// A step starts at the session's position or, right after a multi-row decode step,
    /// anywhere from `floor` up to it: the rows past that start were drafts not kept.
    fn starts(start: u32, position: u32, floor: Option<u32>) -> bool {
        start == position || floor.is_some_and(|f| (f..position).contains(&start))
    }
    /// The step's end if its phase fits: prefill only before any decode step and within the
    /// prompt; decode from the prompt's end, 1 to MAX_DECODE_ROWS rows, within the output bound.
    fn step_end(&self, h: &TensorHeader, decoded: u32) -> Result<u32, CoreError> {
        let (start, rows) = (h.position_start.get(), h.token_count.get());
        let end = start.checked_add(rows).ok_or(CoreError::Invalid)?;
        let prompt = self.record.prompt_token_count.get();
        match h.phase {
            TensorHeaderPhase::Prefill if decoded == 0 && end <= prompt => Ok(end),
            TensorHeaderPhase::Decode
                if start >= prompt
                    && (1..=starlings_protocol::MAX_DECODE_ROWS).contains(&rows)
                    && end - prompt <= self.record.max_output_tokens.get() =>
            {
                Ok(end)
            }
            _ => Err(CoreError::State),
        }
    }
}
/// The lowest start allowed after a step from `start` to `end`: set only by a multi-row
/// decode step.
fn floor_after(start: u32, end: u32, phase: TensorHeaderPhase) -> Option<u32> {
    (phase == TensorHeaderPhase::Decode && end - start > 1).then_some(start + 1)
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Recovery {
    pub node_id: NodeId,
    pub cluster_id: ClusterId,
    pub epoch: Decimal,
    pub leases: BTreeSet<LeaseId>,
    pub generations: BTreeMap<GenerationId, SessionState>,
    #[serde(default, skip_serializing_if = "BTreeSet::is_empty")]
    pub retained_backends: BTreeSet<LeaseId>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Snapshot {
    pub epoch: Decimal,
    pub reserved_bytes: Decimal,
    pub instances: BTreeMap<LeaseId, InstanceState>,
    pub generations: BTreeMap<GenerationId, SessionState>,
}

type CachedResult = (
    RequestId,
    Digest,
    Result<(), CoreError>,
    Option<Arc<PlacementPlan>>,
);

pub struct Worker {
    node: NodeId,
    cluster: ClusterId,
    capacity: u64,
    epoch: u64,
    now: u64,
    mode: FixtureMode,
    draining_epoch: bool,
    backend_stopped: bool,
    reservations: BTreeMap<LeaseId, Reservation>,
    sessions: BTreeMap<GenerationId, RunningSession>,
    requests: BTreeMap<u64, CachedResult>,
    last_sequence: u64,
    retired_generations: BTreeMap<GenerationId, SessionState>,
}
fn terminal(s: SessionState) -> bool {
    matches!(
        s,
        SessionState::Completed | SessionState::Cancelled | SessionState::Failed
    )
}
impl Worker {
    pub fn new(
        node: NodeId,
        cluster: ClusterId,
        capacity: u64,
        epoch: u64,
        mode: FixtureMode,
    ) -> Result<Self, CoreError> {
        if capacity == 0 || epoch == 0 {
            return Err(CoreError::Invalid);
        }
        Ok(Self {
            node,
            cluster,
            capacity,
            epoch,
            now: 0,
            mode,
            draining_epoch: false,
            backend_stopped: false,
            reservations: BTreeMap::new(),
            sessions: BTreeMap::new(),
            requests: BTreeMap::new(),
            last_sequence: 0,
            retired_generations: BTreeMap::new(),
        })
    }
    pub fn recover(
        node: NodeId,
        cluster: ClusterId,
        capacity: u64,
        epoch: u64,
        recovery: Recovery,
        mode: FixtureMode,
    ) -> Result<Self, CoreError> {
        if node != recovery.node_id || cluster != recovery.cluster_id {
            return Err(CoreError::Invalid);
        }
        if epoch <= recovery.epoch.0 {
            return Err(CoreError::StaleEpoch);
        }
        if !recovery.retained_backends.is_empty() {
            return Err(CoreError::State);
        }
        if recovery.leases.len() > MAX_RECORDS || recovery.generations.len() > MAX_RECORDS {
            return Err(CoreError::Limit);
        }
        let mut worker = Self::new(node, cluster, capacity, epoch, mode)?;
        // Old-epoch packets are rejected before lookup; lease tombstones need not survive fencing.
        worker.retired_generations = recovery
            .generations
            .into_iter()
            .map(|(g, s)| (g, if terminal(s) { s } else { SessionState::Failed }))
            .collect();
        Ok(worker)
    }
    /// Persist epoch and tombstones before admitting work after a confirmed worker shutdown.
    pub fn recovery(&self) -> Recovery {
        let mut generations = self.snapshot().generations;
        while generations.len() > MAX_RECORDS {
            generations.pop_first();
        }
        Recovery {
            node_id: self.node,
            cluster_id: self.cluster,
            epoch: Decimal(self.epoch),
            leases: self.reservations.keys().copied().collect(),
            generations,
            retained_backends: self
                .reservations
                .iter()
                .filter_map(|(id, r)| {
                    r.backend
                        .as_ref()
                        .is_some_and(|owner| owner.retained)
                        .then_some(*id)
                })
                .collect(),
        }
    }
    /// Cancel the old authority's work and switch only after trusted local drain.
    /// Release old-epoch history only after fencing and trusted local drain.
    pub fn reconcile_epoch(&mut self, epoch: u64) -> Result<bool, CoreError> {
        if self.backend_stopped {
            return Err(CoreError::State);
        }
        if epoch < self.epoch || epoch == 0 {
            return Err(CoreError::StaleEpoch);
        }
        if epoch == self.epoch {
            return Ok(true);
        }
        self.draining_epoch = true;
        self.retain_plans(|_| false)?;
        if self.reserved_bytes() != 0 {
            return Ok(false);
        }
        self.epoch = epoch;
        self.draining_epoch = false;
        self.requests.clear();
        self.last_sequence = 0;
        self.reservations.clear();
        self.sessions.clear();
        self.retired_generations.clear();
        Ok(true)
    }
    pub fn reserved_bytes(&self) -> u64 {
        self.reservations
            .values()
            .filter(|r| r.state != InstanceState::Released)
            .map(|r| r.lease.reserved_bytes.0)
            .sum()
    }
    pub fn backend_is_stopped(&self) -> bool {
        self.backend_stopped
    }
    pub fn snapshot(&self) -> Snapshot {
        let mut generations = self.retired_generations.clone();
        generations.extend(self.sessions.iter().map(|(id, s)| (*id, s.record.state)));
        Snapshot {
            epoch: Decimal(self.epoch),
            reserved_bytes: Decimal(self.reserved_bytes()),
            instances: self
                .reservations
                .iter()
                .map(|(id, r)| (*id, r.state))
                .collect(),
            generations,
        }
    }
    pub fn advance(&mut self, now: u64) -> Result<(), CoreError> {
        if now < self.now {
            return Err(CoreError::Clock);
        }
        self.now = now;
        let expired: Vec<_> = self
            .reservations
            .iter()
            .filter(|(_, r)| r.state != InstanceState::Released && now >= r.deadline)
            .map(|(id, _)| *id)
            .collect();
        for id in expired {
            self.release(id, SessionState::Failed)?;
        }
        Ok(())
    }
    /// Returns true for an exact replay; the backend must not execute it again.
    pub fn apply(&mut self, request: &ControlRequest, now: u64) -> Result<bool, CoreError> {
        if self.backend_stopped {
            return Err(CoreError::State);
        }
        request
            .validate(self.mode)
            .map_err(|_| CoreError::Invalid)?;
        self.advance(now)?;
        if request.authority_epoch.0 != self.epoch {
            return Err(CoreError::StaleEpoch);
        }
        if self.draining_epoch
            && matches!(
                request.operation,
                Operation::Prepare { .. }
                    | Operation::Begin { .. }
                    | Operation::StartStep { .. }
                    | Operation::StartRequest { .. }
                    | Operation::Commit { .. }
            )
        {
            return Err(CoreError::StaleEpoch);
        }
        let sequence = request.request_sequence.0;
        if let Some((id, hash, result, _)) = self.requests.get(&sequence) {
            if id != &request.request_id || hash != &request.request_hash {
                return Err(CoreError::Conflict);
            }
            return result.map(|()| true);
        }
        if sequence <= self.last_sequence {
            return Err(CoreError::ReplayExpired);
        }
        if self
            .requests
            .values()
            .any(|(id, _, _, _)| id == &request.request_id)
        {
            return Err(CoreError::Conflict);
        }
        let plan = match &request.operation {
            Operation::Prepare { plan, .. } => Some(Arc::new(*plan.clone())),
            _ => self.operation_plan(&request.operation).ok().and_then(|p| {
                self.reservations
                    .values()
                    .find(|r| std::ptr::eq(r.plan.as_ref(), p))
                    .map(|r| r.plan.clone())
            }),
        };
        let result = self.execute(&request.operation);
        self.last_sequence = sequence;
        self.requests.insert(
            sequence,
            (
                request.request_id,
                request.request_hash.clone(),
                result,
                plan,
            ),
        );
        if self.requests.len() > MAX_REQUESTS {
            self.requests.pop_first();
        }
        result.map(|()| false)
    }
    /// Cached results retain their authorization plan until the explicit replay window expires.
    pub fn request_plan<'a>(
        &'a self,
        request: &'a ControlRequest,
    ) -> Result<&'a PlacementPlan, CoreError> {
        if let Some((id, hash, _, Some(plan))) = self.requests.get(&request.request_sequence.0) {
            if id != &request.request_id || hash != &request.request_hash {
                return Err(CoreError::Conflict);
            }
            return Ok(plan);
        }
        if request.request_sequence.0 <= self.last_sequence {
            return Err(CoreError::ReplayExpired);
        }
        self.operation_plan(&request.operation)
    }
    pub fn retained_plan_ids(&self) -> BTreeSet<PlanId> {
        self.reservations
            .values()
            .map(|r| r.plan.plan_id)
            .chain(
                self.requests
                    .values()
                    .filter_map(|(_, _, _, plan)| plan.as_ref().map(|p| p.plan_id)),
            )
            .collect()
    }
    pub fn retains_request(&self, id: RequestId) -> bool {
        self.requests.values().any(|(saved, _, _, _)| *saved == id)
    }
    /// Resolve the stored plan, never a caller-supplied model name, for authorization.
    pub fn operation_plan<'a>(
        &'a self,
        operation: &'a Operation,
    ) -> Result<&'a PlacementPlan, CoreError> {
        let lease = match operation {
            Operation::Prepare { plan, .. } => return Ok(plan),
            Operation::Commit { lease_id } | Operation::Release { lease_id } => *lease_id,
            Operation::Begin { session } | Operation::StartRequest { session, .. } => {
                session.lease_id
            }
            Operation::StartStep { header, .. } => {
                self.sessions
                    .get(&header.generation_id)
                    .ok_or(CoreError::NotFound)?
                    .record
                    .lease_id
            }
            Operation::CompleteStep { generation_id, .. }
            | Operation::Cancel { generation_id }
            | Operation::Finish { generation_id } => {
                self.sessions
                    .get(generation_id)
                    .ok_or(CoreError::NotFound)?
                    .record
                    .lease_id
            }
        };
        Ok(&self
            .reservations
            .get(&lease)
            .ok_or(CoreError::NotFound)?
            .plan)
    }
    /// Revoke admission immediately; retain buffers until local backend drain acknowledgement.
    pub fn retain_plans(
        &mut self,
        mut allowed: impl FnMut(&PlacementPlan) -> bool,
    ) -> Result<(), CoreError> {
        let denied: Vec<_> = self
            .reservations
            .iter()
            .filter(|(_, r)| r.state != InstanceState::Released && !allowed(&r.plan))
            .map(|(id, _)| *id)
            .collect();
        for lease in denied {
            self.release(lease, SessionState::Cancelled)?;
        }
        Ok(())
    }
    /// Publication requires both an active lease and a generation that can deliver output.
    pub fn can_publish(&self, generation: GenerationId) -> bool {
        if self.backend_stopped {
            return false;
        }
        self.sessions.get(&generation).is_some_and(|session| {
            matches!(
                session.record.state,
                SessionState::Prefilling | SessionState::Decoding | SessionState::Completed
            ) && self
                .reservations
                .get(&session.record.lease_id)
                .is_some_and(|r| r.state == InstanceState::Ready && self.now < r.deadline)
        })
    }
    /// Read-only local publication gate for the exact invocation still owned by this session.
    /// The caller must also hold current live authority while synchronously enqueuing output.
    pub fn can_publish_step(&self, generation: GenerationId, invocation: InvocationId) -> bool {
        self.can_publish(generation) && self.sessions.get(&generation)
            .is_some_and(|session| matches!(session.pending,Some(PendingInvocation::Step(id,..)) if id == invocation))
    }
    pub fn node_id(&self) -> NodeId {
        self.node
    }
    /// The live session's prompt length; read-only.
    pub fn prompt_tokens(&self, generation: GenerationId) -> Option<u32> {
        self.sessions
            .get(&generation)
            .map(|s| s.record.prompt_token_count.get())
    }
    /// Ready lease authority only; it never asserts native preparation or renews a lease.
    pub fn lease_plan(&self, lease: LeaseId) -> Result<&PlacementPlan, CoreError> {
        let r = self.reservations.get(&lease).ok_or(CoreError::NotFound)?;
        if self.backend_stopped
            || self.draining_epoch
            || r.state != InstanceState::Ready
            || self.now >= r.deadline
            || r.plan.authority_epoch.0 != self.epoch
        {
            return Err(CoreError::Expired);
        }
        Ok(&r.plan)
    }
    /// The plan a lease was prepared for, whatever its state (GLY-190: a stage's runtime
    /// holds several plans; their owners check each lease is theirs).
    pub fn plan_of(&self, lease: LeaseId) -> Option<&PlacementPlan> {
        self.reservations.get(&lease).map(|r| r.plan.as_ref())
    }
    /// Trusted local withdrawal; in-flight reservations remain until local drain.
    pub fn withdraw_lease(&mut self, lease: LeaseId) -> Result<(), CoreError> {
        self.release(lease, SessionState::Cancelled)
    }
    /// Trusted local supervisor only: retain the complete layer-stage reservation before
    /// dispatching native preparation, including when its reply may be lost. Does not load
    /// a backend, authorize dispatch, or turn allocation counters into qualification.
    pub fn hold_backend(&mut self, lease: LeaseId, now: u64) -> Result<BackendHold, CoreError> {
        self.advance(now)?;
        let reservation = self
            .reservations
            .get_mut(&lease)
            .ok_or(CoreError::NotFound)?;
        if self.backend_stopped
            || self.draining_epoch
            || reservation.state != InstanceState::Prepared
            || reservation.plan.mode != PlacementPlanMode::LayerPipeline
            || reservation.plan.authority_epoch.0 != self.epoch
        {
            return Err(CoreError::State);
        }
        if reservation.backend.is_some() {
            return Err(CoreError::Conflict);
        }
        let identity = Arc::new(());
        reservation.backend = Some(BackendOwnership {
            identity: identity.clone(),
            retained: true,
        });
        Ok(BackendHold { lease, identity })
    }
    /// Retained local cleanup ownership only; not readiness, authority or dispatch permission.
    pub fn check_backend_hold(&self, hold: &BackendHold) -> Result<LeaseId, CoreError> {
        let reservation = self
            .reservations
            .get(&hold.lease)
            .ok_or(CoreError::NotFound)?;
        let owner = reservation.backend.as_ref().ok_or(CoreError::Conflict)?;
        if !Arc::ptr_eq(&owner.identity, &hold.identity) {
            return Err(CoreError::Conflict);
        }
        if self.backend_stopped || !owner.retained || reservation.state == InstanceState::Released {
            return Err(CoreError::State);
        }
        Ok(hold.lease)
    }
    /// Trusted local complete-boundary dispatch only. The owner must retain authentic
    /// fully reassembled bytes and first validate every layout against its approved boundary.
    /// These are the original first-frame templates, not fabricated oversized wire chunks.
    /// All metadata/aggregate checks precede the single invocation mutation. Live authority
    /// and bounded enqueue remain the secured local owner's responsibility; no I/O occurs here.
    pub fn start_local_chunk(
        &mut self,
        hold: &BackendHold,
        headers: &[TensorHeader],
        now: u64,
    ) -> Result<(), CoreError> {
        let lease = self.check_backend_hold(hold)?;
        if self.draining_epoch || headers.is_empty() || headers.len() > 8 {
            return Err(CoreError::State);
        }
        let first = &headers[0];
        let reservation = self.reservations.get(&lease).ok_or(CoreError::NotFound)?;
        let stage = reservation
            .plan
            .stages
            .iter()
            .find(|stage| stage.node_id == self.node)
            .ok_or(CoreError::Invalid)?;
        if first.lease_id != lease || first.destination_stage != stage.index {
            return Err(CoreError::Conflict);
        }
        let scope = DataScope::from(first);
        let mut total = 0u64;
        for (index, header) in headers.iter().enumerate() {
            encode(header, self.mode).map_err(|_| CoreError::Invalid)?;
            if DataScope::from(header) != scope
                || header.phase != first.phase
                || header.position_start != first.position_start
                || header.token_count != first.token_count
                || header.boundary_profile != first.boundary_profile
                || header.tensor_index.get() as usize != index
                || header.tensor_count.get() as usize != headers.len()
                || header.chunk_offset_bytes.0 != 0
                || (header.tensor_role == TensorHeaderTensorRole::TokenIds) != stage.owns_embedding
                || header.tensor_role == TensorHeaderTensorRole::TokenIds && headers.len() != 1
            {
                return Err(CoreError::Invalid);
            }
            total = total
                .checked_add(header.total_tensor_bytes.0)
                .ok_or(CoreError::Capacity)?;
        }
        if total > stage.scratch_budget_bytes.0 {
            return Err(CoreError::Capacity);
        }
        self.advance(now)?;
        self.start_invocation(first, true)
    }
    /// Trusted local, qualified full-instance cleanup only. A chunk/status/end/unload
    /// receipt, process exit or dropped handle alone is insufficient. Withdraw authority
    /// first and resolve pending invocations separately; this cannot advance their state.
    pub fn backend_unloaded(&mut self, hold: &BackendHold) -> Result<(), CoreError> {
        let reservation = self
            .reservations
            .get(&hold.lease)
            .ok_or(CoreError::NotFound)?;
        let owner = reservation.backend.as_ref().ok_or(CoreError::Conflict)?;
        if !Arc::ptr_eq(&owner.identity, &hold.identity) {
            return Err(CoreError::Conflict);
        }
        if !owner.retained {
            return Ok(());
        }
        if reservation.state != InstanceState::Releasing
            || self
                .sessions
                .values()
                .any(|s| s.record.lease_id == hold.lease && s.pending.is_some())
        {
            return Err(CoreError::State);
        }
        self.reservations
            .get_mut(&hold.lease)
            .ok_or(CoreError::NotFound)?
            .backend
            .as_mut()
            .ok_or(CoreError::Conflict)?
            .retained = false;
        self.finish_release(hold.lease);
        Ok(())
    }
    /// Read-only data-plane context. A frame never prepares a lease or starts execution.
    /// The receiver proves its own lease; a sender cannot prove a remote lease is live.
    pub fn data_plan(&self, h: &TensorHeader) -> Result<&PlacementPlan, CoreError> {
        h.validate().map_err(|_| CoreError::Invalid)?;
        let s = self
            .sessions
            .get(&h.generation_id)
            .ok_or(CoreError::NotFound)?;
        let r = self
            .reservations
            .get(&s.record.lease_id)
            .ok_or(CoreError::NotFound)?;
        let stage = r
            .plan
            .stages
            .iter()
            .find(|s| s.node_id == self.node)
            .ok_or(CoreError::Invalid)?;
        let source = r
            .plan
            .stages
            .get(h.source_stage.get() as usize)
            .ok_or(CoreError::Invalid)?;
        let destination = r
            .plan
            .stages
            .get(h.destination_stage.get() as usize)
            .ok_or(CoreError::Invalid)?;
        if self.backend_stopped
            || self.draining_epoch
            || r.state != InstanceState::Ready
            || self.now >= r.deadline
            || terminal(s.record.state)
            || s.record.state == SessionState::Cancelling
        {
            return Err(CoreError::State);
        }
        if r.plan.mode != PlacementPlanMode::LayerPipeline
            || h.authority_epoch.0 != self.epoch
            || h.authority_epoch != r.plan.authority_epoch
            || h.plan_digest != r.plan.plan_digest
            || h.session_id != s.record.session_id
            || h.boundary_profile != r.plan.boundary_profile
            || source.node_id == destination.node_id
            || (stage.index != h.source_stage && stage.index != h.destination_stage)
            || (stage.index == h.destination_stage && h.lease_id != r.lease.lease_id)
            || h.total_tensor_bytes.0 > stage.scratch_budget_bytes.0
            || h.position_start
                .get()
                .checked_add(h.token_count.get())
                .is_none_or(|n| n > s.record.context_limit.get())
            || if h.tensor_role == TensorHeaderTensorRole::TokenIds {
                source.index.get() as usize + 1 != r.plan.stages.len()
                    || destination.index.get() != 0
            } else {
                source.index.get() + 1 != destination.index.get()
            }
        {
            return Err(CoreError::Invalid);
        }
        Ok(&r.plan)
    }
    /// The actual local session lease, which can differ from the header's remote lease.
    pub fn data_lease(&self, h: &TensorHeader) -> Result<LeaseId, CoreError> {
        self.data_plan(h)?;
        Ok(self.sessions[&h.generation_id].record.lease_id)
    }
    /// Checked before receiving a chunk. Reassembly/ordering must still precede StartStep.
    pub fn validate_data_input(&self, h: &TensorHeader) -> Result<(), CoreError> {
        self.validate_input(h, false)
    }
    /// Opening a receive only: as `validate_data_input`, or for the step right after a
    /// pending one, checked against the session as that step's completion will leave it. The
    /// data itself is still admitted under `validate_data_input`, after the step drains.
    pub fn validate_data_input_next(&self, h: &TensorHeader) -> Result<(), CoreError> {
        self.validate_input(h, true)
    }
    fn validate_input(&self, h: &TensorHeader, next: bool) -> Result<(), CoreError> {
        let plan = self.data_plan(h)?;
        let s = self
            .sessions
            .get(&h.generation_id)
            .ok_or(CoreError::NotFound)?;
        let (position, floor, sequence, decoded) = match &s.pending {
            None => (s.position, s.floor, s.next_sequence, s.decoded),
            Some(PendingInvocation::Step(_, start, end, phase))
                if next && s.terminal_after_drain.is_none() =>
            {
                (
                    *end,
                    floor_after(*start, *end, *phase),
                    s.next_sequence.checked_add(1).ok_or(CoreError::Limit)?,
                    s.decoded + u32::from(*phase == TensorHeaderPhase::Decode),
                )
            }
            Some(_) => return Err(CoreError::State),
        };
        if plan.stages[h.destination_stage.get() as usize].node_id != self.node
            || s.invocations.contains(&h.invocation_id)
            || !RunningSession::starts(h.position_start.get(), position, floor)
            || h.sequence.0 != sequence
        {
            return Err(CoreError::State);
        }
        s.step_end(h, decoded).map(|_| ())
    }
    /// Trusted local transfer owner only. Abort the admitted generation even after
    /// authority expires; a pending invocation still requires trusted local drain.
    pub fn abort_data(&mut self, h: &TensorHeader) -> Result<(), CoreError> {
        let session = self
            .sessions
            .get_mut(&h.generation_id)
            .ok_or(CoreError::NotFound)?;
        if session.record.session_id != h.session_id
            || session.record.plan_digest != h.plan_digest
            || session.record.authority_epoch != h.authority_epoch
        {
            return Err(CoreError::Conflict);
        }
        Self::terminate(session, SessionState::Failed);
        Ok(())
    }
    /// Trusted local backend completion, including after network authority is revoked.
    pub fn drained(
        &mut self,
        generation: GenerationId,
        invocation: InvocationId,
    ) -> Result<(), CoreError> {
        self.complete(generation, invocation)
    }
    /// A crashed/failed backend is terminal only after its allocations are confirmed drained.
    pub fn failed_and_drained(
        &mut self,
        generation: GenerationId,
        invocation: InvocationId,
    ) -> Result<(), CoreError> {
        let session = self
            .sessions
            .get_mut(&generation)
            .ok_or(CoreError::NotFound)?;
        if session.pending.as_ref().ok_or(CoreError::State)?.id() != invocation {
            return Err(CoreError::Conflict);
        }
        Self::terminate(session, SessionState::Failed);
        self.complete(generation, invocation)
    }
    /// Trusted local acknowledgement that the entire backend and its allocations
    /// have stopped. Preserve history/epoch and permanently refuse this runtime's reuse.
    pub fn backend_shutdown_drained(&mut self) -> Result<(), CoreError> {
        self.backend_stopped = true;
        self.retain_plans(|_| false)?;
        let pending: Vec<_> = self
            .sessions
            .iter()
            .filter_map(|(g, s)| s.pending.as_ref().map(|i| (*g, i.id())))
            .collect();
        for (generation, invocation) in pending {
            self.failed_and_drained(generation, invocation)?;
        }
        for reservation in self.reservations.values_mut() {
            if let Some(owner) = &mut reservation.backend {
                owner.retained = false;
            }
        }
        let leases: Vec<_> = self.reservations.keys().copied().collect();
        for lease in leases {
            self.finish_release(lease);
        }
        Ok(())
    }
    fn execute(&mut self, operation: &Operation) -> Result<(), CoreError> {
        match operation {
            Operation::Prepare { plan, lease } => self.prepare(plan, lease),
            Operation::Commit { lease_id } => {
                let r = self
                    .reservations
                    .get_mut(lease_id)
                    .ok_or(CoreError::NotFound)?;
                match r.state {
                    InstanceState::Prepared => {
                        r.state = InstanceState::Ready;
                        Ok(())
                    }
                    InstanceState::Ready => Ok(()),
                    _ => Err(CoreError::Expired),
                }
            }
            Operation::Release { lease_id } => self.release(*lease_id, SessionState::Cancelled),
            Operation::Begin { session } => self.begin(session),
            Operation::StartRequest {
                session,
                invocation_id,
                request,
            } => self.start_request(session, *invocation_id, request),
            Operation::StartStep { header, .. } => self.start(header),
            Operation::CompleteStep {
                generation_id,
                invocation_id,
            } => self.complete(*generation_id, *invocation_id),
            Operation::Cancel { generation_id } => {
                if self.retired_generations.contains_key(generation_id) {
                    return Ok(());
                }
                let s = self
                    .sessions
                    .get_mut(generation_id)
                    .ok_or(CoreError::NotFound)?;
                Self::terminate(s, SessionState::Cancelled);
                Ok(())
            }
            Operation::Finish { generation_id } => {
                let s = self
                    .sessions
                    .get_mut(generation_id)
                    .ok_or(CoreError::NotFound)?;
                if s.record.state == SessionState::Completed {
                    return Ok(());
                }
                if terminal(s.record.state)
                    || s.pending.is_some()
                    || s.record.state == SessionState::Cancelling
                    || s.position < s.record.prompt_token_count.get()
                {
                    return Err(CoreError::State);
                }
                s.record.state = SessionState::Completed;
                Ok(())
            }
        }
    }
    fn prepare(&mut self, plan: &PlacementPlan, lease: &Lease) -> Result<(), CoreError> {
        if lease.authority_epoch.0 != self.epoch || plan.authority_epoch.0 != self.epoch {
            return Err(CoreError::StaleEpoch);
        }
        if lease.cluster_id != self.cluster
            || plan.cluster_id != self.cluster
            || lease.owner_node_id != self.node
            || lease.plan_digest != plan.plan_digest
            || lease.max_sessions.get() > plan.max_sessions.get()
        {
            return Err(CoreError::Invalid);
        }
        let stages: Vec<_> = plan
            .stages
            .iter()
            .filter(|s| s.node_id == self.node)
            .collect();
        if stages.len() != 1
            || stages[0].reserved_bytes().map_err(|_| CoreError::Invalid)? != lease.reserved_bytes.0
        {
            return Err(CoreError::Invalid);
        }
        if let Some(existing) = self.reservations.get(&lease.lease_id) {
            if existing.plan.as_ref() != plan || &existing.lease != lease {
                return Err(CoreError::Conflict);
            }
            return if matches!(
                existing.state,
                InstanceState::Prepared | InstanceState::Ready
            ) {
                Ok(())
            } else {
                Err(CoreError::Expired)
            };
        }
        if self.reservations.len() >= MAX_RECORDS {
            self.reservations
                .retain(|_, r| r.state != InstanceState::Released);
        }
        if self.reservations.len() >= MAX_RECORDS {
            return Err(CoreError::Capacity);
        }
        let deadline = self
            .now
            .checked_add(lease.lifetime_ms.get().into())
            .ok_or(CoreError::Clock)?;
        if self
            .reserved_bytes()
            .checked_add(lease.reserved_bytes.0)
            .is_none_or(|n| n > self.capacity)
        {
            return Err(CoreError::Capacity);
        }
        self.reservations.insert(
            lease.lease_id,
            Reservation {
                plan: Arc::new(plan.clone()),
                lease: lease.clone(),
                deadline,
                state: InstanceState::Prepared,
                backend: None,
            },
        );
        Ok(())
    }
    fn release(&mut self, id: LeaseId, reason: SessionState) -> Result<(), CoreError> {
        let r = self.reservations.get_mut(&id).ok_or(CoreError::NotFound)?;
        if r.state == InstanceState::Released {
            return Ok(());
        }
        r.state = InstanceState::Releasing;
        for s in self
            .sessions
            .values_mut()
            .filter(|s| s.record.lease_id == id)
        {
            Self::terminate(s, reason);
        }
        self.finish_release(id);
        Ok(())
    }
    fn terminate(s: &mut RunningSession, reason: SessionState) {
        if terminal(s.record.state) {
            return;
        }
        if s.pending.is_some() {
            s.record.state = SessionState::Cancelling;
            if s.terminal_after_drain.is_none() {
                s.terminal_after_drain = Some(reason);
            }
        } else {
            s.record.state = reason;
        }
    }
    fn finish_release(&mut self, id: LeaseId) {
        let pending = self
            .sessions
            .values()
            .any(|s| s.record.lease_id == id && s.pending.is_some());
        if let Some(r) = self.reservations.get_mut(&id)
            && r.state == InstanceState::Releasing
            && !pending
            && r.backend.as_ref().is_none_or(|owner| !owner.retained)
        {
            r.state = InstanceState::Released;
        }
    }
    fn begin(&mut self, session: &Session) -> Result<(), CoreError> {
        if session.authority_epoch.0 != self.epoch {
            return Err(CoreError::StaleEpoch);
        }
        if self.sessions.contains_key(&session.generation_id)
            || self
                .retired_generations
                .contains_key(&session.generation_id)
        {
            return Err(CoreError::State);
        }
        if self.sessions.len() >= MAX_RECORDS {
            self.sessions.retain(|_, s| !terminal(s.record.state));
        }
        if self.sessions.len() >= MAX_RECORDS {
            return Err(CoreError::Capacity);
        }
        let r = self
            .reservations
            .get(&session.lease_id)
            .ok_or(CoreError::NotFound)?;
        if r.state != InstanceState::Ready
            || session.plan_digest != r.plan.plan_digest
            || session.context_limit.get() > r.plan.context_limit.get()
            || session.state != SessionState::Queued
        {
            return Err(CoreError::State);
        }
        if self
            .sessions
            .values()
            .any(|s| s.record.session_id == session.session_id && !terminal(s.record.state))
        {
            return Err(CoreError::Conflict);
        }
        let count = self
            .sessions
            .values()
            .filter(|s| s.record.lease_id == session.lease_id && !terminal(s.record.state))
            .count();
        if count >= r.lease.max_sessions.get() as usize {
            return Err(CoreError::Capacity);
        }
        let mut record = session.clone();
        record.state = SessionState::Admitted;
        self.sessions.insert(
            session.generation_id,
            RunningSession {
                record,
                position: 0,
                decoded: 0,
                pending: None,
                next_sequence: 0,
                invocations: BTreeSet::new(),
                terminal_after_drain: None,
                floor: None,
            },
        );
        Ok(())
    }
    fn start_request(
        &mut self,
        session: &Session,
        invocation: InvocationId,
        request: &ChatRequest,
    ) -> Result<(), CoreError> {
        let reservation = self
            .reservations
            .get(&session.lease_id)
            .ok_or(CoreError::NotFound)?;
        if reservation.plan.mode != PlacementPlanMode::WholeRequest
            || request.model != reservation.plan.manifest_digest.as_str()
        {
            return Err(CoreError::Invalid);
        }
        self.begin(session)?;
        let running = self
            .sessions
            .get_mut(&session.generation_id)
            .ok_or(CoreError::NotFound)?;
        running.pending = Some(PendingInvocation::Request(invocation));
        running.record.state = SessionState::Prefilling;
        Ok(())
    }
    fn start(&mut self, h: &TensorHeader) -> Result<(), CoreError> {
        self.start_invocation(h, false)
    }
    fn start_invocation(
        &mut self,
        h: &TensorHeader,
        local_boundary: bool,
    ) -> Result<(), CoreError> {
        if h.authority_epoch.0 != self.epoch {
            return Err(CoreError::StaleEpoch);
        }
        let s = self
            .sessions
            .get_mut(&h.generation_id)
            .ok_or(CoreError::NotFound)?;
        let r = self
            .reservations
            .get(&s.record.lease_id)
            .ok_or(CoreError::NotFound)?;
        let stage = r
            .plan
            .stages
            .iter()
            .find(|stage| stage.node_id == self.node)
            .ok_or(CoreError::Invalid)?;
        if r.state != InstanceState::Ready
            || r.plan.mode != PlacementPlanMode::LayerPipeline
            || terminal(s.record.state)
            || s.record.state == SessionState::Cancelling
            || s.pending.is_some()
        {
            return Err(CoreError::State);
        }
        let source = if local_boundary && h.tensor_role == TensorHeaderTensorRole::TokenIds {
            r.plan.stages.last().ok_or(CoreError::Invalid)?.index.get()
        } else {
            stage.index.get().saturating_sub(1)
        };
        if h.session_id != s.record.session_id
            || h.lease_id != s.record.lease_id
            || h.plan_digest != s.record.plan_digest
            || h.boundary_profile != r.plan.boundary_profile
            || (h.tensor_role == TensorHeaderTensorRole::TokenIds && stage.index.get() != 0)
            || h.destination_stage.get() != stage.index.get()
            || h.source_stage.get() != source
            || !RunningSession::starts(h.position_start.get(), s.position, s.floor)
            || h.sequence.0 != s.next_sequence
            || s.invocations.contains(&h.invocation_id)
            || h.total_tensor_bytes.0 > stage.scratch_budget_bytes.0
            || !local_boundary
                && (h.tensor_index.get() != 0
                    || h.tensor_count.get() != 1
                    || h.chunk_offset_bytes.0 != 0
                    || h.chunk_bytes != h.total_tensor_bytes
                    || !h.end_of_tensor)
        {
            return Err(CoreError::Invalid);
        }
        let end = s.step_end(h, s.decoded)?;
        s.invocations.insert(h.invocation_id);
        s.pending = Some(PendingInvocation::Step(
            h.invocation_id,
            h.position_start.get(),
            end,
            h.phase,
        ));
        if local_boundary {
            s.record.state = match h.phase {
                TensorHeaderPhase::Prefill => SessionState::Prefilling,
                TensorHeaderPhase::Decode => SessionState::Decoding,
            };
        }
        Ok(())
    }
    fn complete(
        &mut self,
        generation: GenerationId,
        invocation: InvocationId,
    ) -> Result<(), CoreError> {
        let s = self
            .sessions
            .get_mut(&generation)
            .ok_or(CoreError::NotFound)?;
        if s.pending.as_ref().ok_or(CoreError::State)?.id() != invocation {
            return Err(CoreError::Conflict);
        }
        let pending = s.pending.take().ok_or(CoreError::State)?;
        if let Some(reason) = s.terminal_after_drain.take() {
            s.record.state = reason;
        } else if let PendingInvocation::Step(_, start, end, phase) = pending {
            s.position = end;
            s.floor = floor_after(start, end, phase);
            s.next_sequence = s.next_sequence.checked_add(1).ok_or(CoreError::Limit)?;
            if phase == TensorHeaderPhase::Decode {
                s.decoded += 1;
            }
            s.record.state = if s.position < s.record.prompt_token_count.get() {
                SessionState::Prefilling
            } else {
                SessionState::Decoding
            };
        } else {
            s.record.state = SessionState::Completed;
        }
        let lease = s.record.lease_id;
        self.finish_release(lease);
        Ok(())
    }
}

/// Receiver-observed age, not a peer wall clock, decides scheduling freshness.
#[derive(Default)]
pub struct Registry {
    nodes: BTreeMap<NodeId, (NodeCapabilities, u64)>,
    now: u64,
    mode: FixtureMode,
}
impl Registry {
    pub fn new(mode: FixtureMode) -> Self {
        Self {
            mode,
            ..Self::default()
        }
    }
    pub fn observe(&mut self, node: NodeCapabilities, now: u64) -> Result<(), CoreError> {
        if now < self.now {
            return Err(CoreError::Clock);
        }
        self.now = now;
        encode(&node, self.mode).map_err(|_| CoreError::Invalid)?;
        let timestamp = time::OffsetDateTime::parse(
            &node.observed_at,
            &time::format_description::well_known::Rfc3339,
        )
        .map_err(|_| CoreError::Invalid)?;
        if let Some((previous, _)) = self.nodes.get(&node.node_id) {
            if previous.snapshot_id == node.snapshot_id {
                return if previous == &node {
                    Ok(())
                } else {
                    Err(CoreError::Conflict)
                };
            }
            let previous_time = time::OffsetDateTime::parse(
                &previous.observed_at,
                &time::format_description::well_known::Rfc3339,
            )
            .map_err(|_| CoreError::Invalid)?;
            if timestamp <= previous_time {
                return Err(CoreError::StaleCapability);
            }
        }
        if !self.nodes.contains_key(&node.node_id) && self.nodes.len() >= 64 {
            return Err(CoreError::Limit);
        }
        self.nodes.insert(node.node_id, (node, now));
        Ok(())
    }
    pub fn eligible(
        &mut self,
        id: NodeId,
        now: u64,
        ttl_ms: u64,
    ) -> Result<&NodeCapabilities, CoreError> {
        if now < self.now {
            return Err(CoreError::Clock);
        }
        self.now = now;
        let (node, observed) = self.nodes.get(&id).ok_or(CoreError::NotFound)?;
        if node.state != NodeCapabilitiesState::Ready
            || node.backends.is_empty()
            || now
                .checked_sub(*observed)
                .and_then(|age| age.checked_add(node.sample_age_ms.get().into()))
                .is_none_or(|age| age >= ttl_ms)
        {
            return Err(CoreError::StaleCapability);
        }
        Ok(node)
    }
}
