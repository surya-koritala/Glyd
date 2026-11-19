//! Coordinator-delegated, lease-level route authority. No backend execution.
use super::*;
use crate::{AuthorityEpoch, EnrollmentReader, GenerationStream, GenerationWatch, RegistryFence};
use starlings_protocol::{BoundedU32, FixtureMode, LeaseId, PlacementPlanMode, RequestId};
use std::time::{Duration, Instant};

const GRANT_TIME: Duration = Duration::from_secs(2);
const GRANT_RECORD: usize = 16 * 1024;
mod local;
pub use local::{StageBackend, StageChunk, step_invocation};
fn grant_deadline(started: Instant) -> Result<Instant> {
    let until = started + GRANT_TIME;
    if Instant::now() >= until {
        Err(Error::Expired)
    } else {
        Ok(until)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct RouteScope {
    epoch: AuthorityEpoch,
    issuer: RequestId,
    plan: Digest,
    node: NodeId,
    lease: LeaseId,
    source: BoundedU32<0, 7>,
    destination: BoundedU32<0, 7>,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Challenge {
    version: Version,
    nonce: RequestId,
    scope: RouteScope,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Reply {
    version: Version,
    challenge: Challenge,
    peer: Member,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Assignment {
    version: Version,
    plan: PlacementPlan,
    issuer: RequestId,
}

struct Gate {
    until: Instant,
    closed: bool,
}
impl Gate {
    fn check(&mut self) -> Result<()> {
        if self.closed || Instant::now() >= self.until {
            self.closed = true;
            return Err(Error::Expired);
        }
        Ok(())
    }
}
struct Delegation {
    scope: RouteScope,
    peer: Member,
    control: GenerationWatch,
    runtime: Weak<Mutex<RuntimeWorker>>,
    peer_expires: i64,
    gate: Mutex<Gate>,
}
/// A permission receipt, never a tensor admission, backend capability or cleanup proof.
#[derive(Clone)]
pub struct StageGrant(Arc<Delegation>, Option<local::NativeGuard>);
impl StageGrant {
    pub fn check(&self) -> Result<()> {
        self.with_runtime(|_| Ok(()))
    }
    fn with_runtime<T>(&self, dispatch: impl FnOnce(&mut RuntimeWorker) -> Result<T>) -> Result<T> {
        self.with_runtime_checked(|_| Ok(()), dispatch)
    }
    fn with_runtime_checked<T>(
        &self,
        before_authority: impl FnOnce(&RuntimeWorker) -> Result<()>,
        dispatch: impl FnOnce(&mut RuntimeWorker) -> Result<T>,
    ) -> Result<T> {
        let inner = self.0.runtime.upgrade().ok_or(Error::Expired)?;
        let mut runtime = inner.lock().map_err(|_| Error::Denied)?;
        // Stale local ownership must never withdraw a successor with reused wire IDs.
        if let Some(guard) = &self.1 {
            guard.registered(&runtime)?;
        }
        before_authority(&runtime)?;
        let control = self.0.control.check();
        let owns = runtime.stages.values().any(|w| w.same(&self.0.control));
        let result = (|| {
            control?;
            if !owns {
                return Err(Error::Expired);
            }
            let fence = runtime.registry.clone().ok_or(Error::Denied)?;
            let guard = fence.lock()?;
            if guard.status() != (self.0.scope.epoch, crate::NodeStatus::Online)
                || !runtime.reconcile_registry_state(0, Some(&guard))?
                || runtime
                    .worker
                    .lease_plan(self.0.scope.lease)
                    .map_err(|_| Error::Expired)?
                    .plan_digest
                    != self.0.scope.plan
            {
                return Err(Error::Expired);
            }
            let mut gate = self.0.gate.lock().map_err(|_| Error::Denied)?;
            gate.check()?;
            if crate::now()? >= self.0.peer_expires {
                return Err(Error::Expired);
            }
            // Input rejection must not withdraw otherwise valid lease authority.
            Ok((|| {
                if let Some(guard) = &self.1 {
                    guard.pending(&runtime)?;
                }
                dispatch(&mut runtime)
            })())
        })();
        if result.is_err() {
            self.0.gate.lock().map_err(|_| Error::Denied)?.closed = true;
            if owns {
                let _ = runtime.worker.withdraw_lease(self.0.scope.lease);
            }
        }
        result?
    }
    pub(in crate::authorization) fn for_step(&self, step: &StageChunk) -> Self {
        Self(self.0.clone(), Some(step.guard()))
    }
    pub(crate) fn deadline(&self) -> Result<Instant> {
        self.check()?;
        Ok(self.0.gate.lock().map_err(|_| Error::Denied)?.until)
    }
    pub(crate) fn trust(
        &self,
        local: &crate::Identity,
        pin: &crate::Invitation,
    ) -> Result<crate::Trust> {
        self.check()?;
        if local.node() != self.0.scope.node
            || local.cluster() != self.0.peer.cluster
            || local.role() != Role::Worker
            || pin.coordinator.cluster != local.cluster()
        {
            return Err(Error::Denied);
        }
        crate::transport::verify_issued(&local.member, &pin.ca, &local.key)?;
        Ok(crate::Trust {
            ca: pin.ca.clone(),
            cluster: local.cluster(),
            members: vec![local.member.clone(), self.0.peer.clone()],
            access: None,
        })
    }
    pub(crate) fn same(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
    pub fn peer(&self) -> Result<NodeId> {
        self.check()?;
        Ok(self.0.peer.node)
    }
}

/// Pure node-local lifecycle ownership, bound to one explicitly approved layer plan.
/// It grants no native capability and never dispatches RunChunk/StartStep.
pub struct StageWorker {
    inner: Arc<Mutex<RuntimeWorker>>,
    plan: PlacementPlan,
    issuer: RequestId,
    grants: Mutex<BTreeMap<(LeaseId, u32, u32), Arc<Delegation>>>,
    backends: Mutex<BTreeMap<LeaseId, Arc<local::BackendOwner>>>,
    /// The control channel this worker bound its plan to on the shared runtime (GLY-190).
    bound: Mutex<Option<crate::GenerationWatch>>,
}
impl StageWorker {
    /// Receive the approved plan/issuer on the already pinned control connection.
    /// The configured model/profile and current registry epoch still constrain it.
    pub fn accept_assignment(
        worker: Worker,
        fence: RegistryFence,
        stream: &mut GenerationStream,
    ) -> Result<Self> {
        Self::assign(&StageRuntime::new(worker, fence, 1)?, stream)
    }
    /// As `accept_assignment`, on a stage's shared runtime (GLY-190).
    pub fn assign(runtime: &StageRuntime, stream: &mut GenerationStream) -> Result<Self> {
        let fence = runtime.fence()?;
        let result = (|| {
            let assignment: Assignment = stream.receive_record_limit(GRANT_RECORD)?;
            stream.watch().check()?;
            if !fence.same(stream.worker_fence()?)
                || fence.status()? != (stream.binding().epoch, crate::NodeStatus::Online)
                || assignment.plan.manifest_digest != stream.binding().model
                || assignment.plan.execution_profile != stream.binding().profile
                || assignment.plan.authority_epoch != stream.binding().epoch.number
            {
                return Err(Error::Denied);
            }
            Self::planned(runtime, assignment.plan, assignment.issuer)
        })();
        if result.is_err() {
            stream.watch().cancel();
        }
        result
    }
    pub fn new(
        worker: Worker,
        fence: RegistryFence,
        approved: PlacementPlan,
        issuer: RequestId,
    ) -> Result<Self> {
        Self::planned(&StageRuntime::new(worker, fence, 1)?, approved, issuer)
    }
    fn planned(runtime: &StageRuntime, approved: PlacementPlan, issuer: RequestId) -> Result<Self> {
        starlings_protocol::encode(&approved, FixtureMode::Reject)?;
        let (identity, live) = {
            let runtime = runtime.inner.lock().map_err(|_| Error::Denied)?;
            let live = runtime.stages.contains_key(&approved.plan_id)
                || runtime
                    .worker
                    .retained_plan_ids()
                    .contains(&approved.plan_id);
            (runtime.worker.recovery(), live)
        };
        // A plan already live on the runtime is another request's.
        if live
            || approved.mode != PlacementPlanMode::LayerPipeline
            || approved.cluster_id != identity.cluster_id
            || !approved
                .stages
                .iter()
                .any(|s| s.node_id == identity.node_id)
        {
            return Err(Error::Denied);
        }
        Ok(Self {
            inner: runtime.inner.clone(),
            plan: approved,
            issuer,
            grants: Mutex::new(BTreeMap::new()),
            backends: Mutex::new(BTreeMap::new()),
            bound: Mutex::new(None),
        })
    }
    fn context(
        &self,
        runtime: &mut RuntimeWorker,
        stream: &GenerationStream,
    ) -> Result<RegistryFence> {
        stream.watch().check()?;
        let fence = runtime.registry.as_ref().ok_or(Error::Denied)?.clone();
        if !fence.same(stream.worker_fence()?)
            || stream.binding().model != self.plan.manifest_digest
            || stream.binding().profile != self.plan.execution_profile
            || stream.binding().epoch.number != self.plan.authority_epoch
        {
            return Err(Error::Denied);
        }
        let guard = fence.lock()?;
        if guard.status() != (stream.binding().epoch, crate::NodeStatus::Online)
            || !runtime.reconcile_registry_state(0, Some(&guard))?
        {
            return Err(Error::Expired);
        }
        drop(guard);
        Ok(fence)
    }
    pub fn apply(&self, stream: &GenerationStream, request: &ControlRequest) -> Result<bool> {
        request.validate(FixtureMode::Reject)?;
        if matches!(
            request.operation,
            Operation::StartRequest { .. }
                | Operation::StartStep { .. }
                | Operation::CompleteStep { .. }
        ) {
            return Err(Error::Denied);
        }
        let mut runtime = self.inner.lock().map_err(|_| Error::Denied)?;
        let watch = stream.watch();
        let plan = self.plan.plan_id;
        // A plan's first Prepare binds it to this channel; only that channel drives it after,
        // and the runtime holds at most its slots' plans at once.
        if !runtime.stages.get(&plan).is_some_and(|w| w.same(&watch))
            && (runtime.stages.contains_key(&plan)
                || !matches!(request.operation, Operation::Prepare { .. }))
        {
            return Err(Error::Denied);
        }
        if !runtime.stages.contains_key(&plan) && runtime.stages.len() >= runtime.slots {
            return Err(Error::Capacity);
        }
        let fence = self.context(&mut runtime, stream)?;
        let guard = fence.lock()?;
        if guard.status() != (stream.binding().epoch, crate::NodeStatus::Online) {
            return Err(Error::Expired);
        }
        if runtime
            .worker
            .request_plan(request)
            .map_err(|_| Error::Denied)?
            .plan_digest
            != self.plan.plan_digest
        {
            return Err(Error::Denied);
        }
        let now = runtime.monotonic_ms(0);
        let replay = runtime
            .worker
            .apply(request, now)
            .map_err(|_| Error::Invalid)?;
        if !replay && !runtime.stages.contains_key(&plan) {
            runtime.stages.insert(plan, watch.clone());
            *self.bound.lock().map_err(|_| Error::Denied)? = Some(watch);
        }
        Ok(replay)
    }
    /// Renew independently of token traffic. Delay is charged from before request send.
    /// An expired route/lease cannot be revived by a later successful reply.
    pub fn refresh_grant(
        &self,
        stream: &mut GenerationStream,
        lease: LeaseId,
        source: BoundedU32<0, 7>,
        destination: BoundedU32<0, 7>,
    ) -> Result<StageGrant> {
        let (scope, key, old) = {
            let mut runtime = self.inner.lock().map_err(|_| Error::Denied)?;
            self.context(&mut runtime, stream)?;
            if !runtime.owns(&self.plan, stream) {
                return Err(Error::Denied);
            }
            if runtime
                .worker
                .lease_plan(lease)
                .map_err(|_| Error::Denied)?
                .plan_digest
                != self.plan.plan_digest
            {
                return Err(Error::Denied);
            }
            route_peer(&self.plan, runtime.worker.node_id(), source, destination)?;
            let scope = RouteScope {
                epoch: stream.binding().epoch,
                issuer: self.issuer,
                plan: self.plan.plan_digest.clone(),
                node: runtime.worker.node_id(),
                lease,
                source,
                destination,
            };
            let key = (scope.lease, scope.source.get(), scope.destination.get());
            let mut grants = self.grants.lock().map_err(|_| Error::Denied)?;
            let snapshot = runtime.worker.snapshot();
            grants.retain(|(lease, _, _), _| {
                snapshot
                    .instances
                    .get(lease)
                    .is_some_and(|state| *state != starlings_core::InstanceState::Released)
            });
            let old = grants.get(&key).cloned();
            if let Some(grant) = &old {
                if grant
                    .gate
                    .lock()
                    .map_err(|_| Error::Denied)?
                    .check()
                    .is_err()
                {
                    runtime
                        .worker
                        .withdraw_lease(lease)
                        .map_err(|_| Error::Denied)?;
                    return Err(Error::Expired);
                }
            } else if grants.len() >= starlings_core::MAX_RECORDS * 2 {
                return Err(Error::Capacity);
            }
            (scope, key, old)
        };
        let challenge = Challenge {
            version: Version,
            nonce: uuid::Uuid::new_v4()
                .to_string()
                .parse()
                .map_err(|_| Error::Invalid)?,
            scope: scope.clone(),
        };
        let started = Instant::now();
        let result = (|| {
            stream.send_record_limit(&challenge, GRANT_RECORD)?;
            let reply: Reply = stream.receive_record_limit(GRANT_RECORD)?;
            if reply.challenge != challenge {
                return Err(Error::Denied);
            }
            let until = grant_deadline(started)?;
            let mut runtime = self.inner.lock().map_err(|_| Error::Denied)?;
            self.context(&mut runtime, stream)?;
            if !runtime.owns(&self.plan, stream) {
                return Err(Error::Denied);
            }
            if runtime
                .worker
                .lease_plan(lease)
                .map_err(|_| Error::Denied)?
                .plan_digest
                != self.plan.plan_digest
            {
                return Err(Error::Denied);
            }
            let local = runtime.worker.node_id();
            let source = &self.plan.stages[scope.source.get() as usize];
            let destination = &self.plan.stages[scope.destination.get() as usize];
            let peer = if local == source.node_id {
                destination.node_id
            } else {
                source.node_id
            };
            if reply.peer.node != peer
                || reply.peer.cluster != self.plan.cluster_id
                || reply.peer.role != Role::Worker
                || reply.peer.certificate.is_empty()
                || reply.peer.certificate.len() > 4096
            {
                return Err(Error::Denied);
            }
            let grant = if let Some(old) = old {
                if old.scope != scope || old.peer.certificate != reply.peer.certificate {
                    return Err(Error::Denied);
                }
                let mut gate = old.gate.lock().map_err(|_| Error::Denied)?;
                gate.check()?;
                gate.until = until;
                drop(gate);
                old
            } else {
                let peer_expires = crate::generation::certificate_expiry(&reply.peer.certificate)?;
                if crate::now()? >= peer_expires {
                    return Err(Error::Expired);
                }
                Arc::new(Delegation {
                    scope,
                    peer: reply.peer,
                    control: stream.watch(),
                    runtime: Arc::downgrade(&self.inner),
                    peer_expires,
                    gate: Mutex::new(Gate {
                        until,
                        closed: false,
                    }),
                })
            };
            self.grants
                .lock()
                .map_err(|_| Error::Denied)?
                .insert(key, grant.clone());
            Ok(StageGrant(grant, None))
        })();
        if result.is_err() {
            stream.watch().cancel();
            self.inner
                .lock()
                .map_err(|_| Error::Denied)?
                .worker
                .withdraw_lease(lease)
                .map_err(|_| Error::Denied)?;
        }
        result
    }
    pub(super) fn data_context<T>(
        &self,
        peer: &PeerStream,
        grant: &StageGrant,
        expected: &starlings_protocol::TensorHeader,
        input: super::data::Input,
        step: Option<&StageChunk>,
        dispatch: impl FnOnce(&crate::data::DataBinding, u64, &mut RuntimeWorker) -> Result<T>,
    ) -> Result<T> {
        if !grant.same(peer.stage_grant()?)
            || peer.channel != crate::Channel::Data
            || !grant.0.runtime.ptr_eq(&Arc::downgrade(&self.inner))
            || expected.source_stage != grant.0.scope.source
            || expected.destination_stage != grant.0.scope.destination
        {
            return Err(Error::Denied);
        }
        grant.with_runtime_checked(
            |runtime| step.map_or(Ok(()), |step| step.registered(runtime)),
            |runtime| {
                if runtime
                    .worker
                    .data_lease(expected)
                    .map_err(|_| Error::Denied)?
                    != grant.0.scope.lease
                {
                    return Err(Error::Denied);
                }
                let plan = runtime
                    .worker
                    .data_plan(expected)
                    .map_err(|_| Error::Denied)?;
                if plan.plan_digest != grant.0.scope.plan
                    || plan.plan_digest != self.plan.plan_digest
                    || peer.local() != runtime.worker.node_id()
                    || peer.peer() != grant.0.peer.node
                    || peer.role() != Role::Worker
                {
                    return Err(Error::Denied);
                }
                let binding = crate::data::DataBinding {
                    version: Version,
                    epoch: grant.0.scope.epoch,
                    plan: plan.plan_digest.clone(),
                    source: plan.stages[expected.source_stage.get() as usize].node_id,
                    destination: plan.stages[expected.destination_stage.get() as usize].node_id,
                    source_stage: expected.source_stage,
                    destination_stage: expected.destination_stage,
                };
                match input {
                    super::data::Input::No => Ok(()),
                    super::data::Input::Due => runtime.worker.validate_data_input(expected),
                    super::data::Input::Next => runtime.worker.validate_data_input_next(expected),
                }
                .map_err(|_| Error::Denied)?;
                let scratch = plan
                    .stages
                    .iter()
                    .find(|s| s.node_id == peer.local())
                    .ok_or(Error::Denied)?
                    .scratch_budget_bytes
                    .0;
                dispatch(&binding, scratch, runtime)
            },
        )
    }
    pub(super) fn abort_data(&self, expected: &starlings_protocol::TensorHeader) -> Result<()> {
        self.inner
            .lock()
            .map_err(|_| Error::Denied)?
            .worker
            .abort_data(expected)
            .map_err(|_| Error::Denied)
    }
    pub fn snapshot(&self) -> Result<Snapshot> {
        Ok(self
            .inner
            .lock()
            .map_err(|_| Error::Denied)?
            .worker
            .snapshot())
    }
    /// The approved plan this owner was bound to, for the local owner's own stage checks.
    pub fn plan(&self) -> &PlacementPlan {
        &self.plan
    }
}
impl Drop for StageWorker {
    /// Ends this request's plan on the runtime: its channel is cancelled and its leases are
    /// released. Other requests' plans go on.
    fn drop(&mut self) {
        let mut runtime = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let plan = self.plan.plan_id;
        // Only the plan this worker bound, never another's with the same ID.
        let bound = self.bound.get_mut().ok().and_then(|b| b.take());
        let mine = bound
            .as_ref()
            .is_some_and(|b| runtime.stages.get(&plan).is_some_and(|w| w.same(b)));
        if mine {
            if let Some(watch) = runtime.stages.remove(&plan) {
                watch.cancel();
            }
            let _ = runtime.worker.retain_plans(|p| p.plan_id != plan);
        }
    }
}

/// GLY-190: a stage daemon's one runtime, registered with the registry fence once, which the
/// registry maintains and revokes as a whole. Each request's `StageWorker` shares it with its
/// own plan, at most `slots` at once.
// ponytail: each plan reserves its stage's bytes (weights included), so the engine-bytes
// limit must allow `slots` of them; a shared-weights reservation would be exact.
pub struct StageRuntime {
    inner: Arc<Mutex<RuntimeWorker>>,
}
impl StageRuntime {
    /// A fresh runtime (no instances or generations), registered with `fence`.
    pub fn new(worker: Worker, fence: RegistryFence, slots: usize) -> Result<Self> {
        let identity = worker.recovery();
        let snapshot = worker.snapshot();
        fence.check_identity(identity.node_id, identity.cluster_id)?;
        if !snapshot.instances.is_empty() || !snapshot.generations.is_empty() || slots == 0 {
            return Err(Error::Denied);
        }
        let inner = Arc::new(Mutex::new(RuntimeWorker {
            registry: Some(fence.clone()),
            generation: None,
            stages: BTreeMap::new(),
            slots,
            clock: Some(Instant::now()),
            worker,
            owners: BTreeMap::new(),
        }));
        fence.register(&inner, identity.node_id, identity.cluster_id)?;
        Ok(Self { inner })
    }
    fn fence(&self) -> Result<RegistryFence> {
        (self
            .inner
            .lock()
            .map_err(|_| Error::Denied)?
            .registry
            .clone())
        .ok_or(Error::Denied)
    }
    /// The authority epoch the runtime was made for.
    pub fn epoch(&self) -> Result<u64> {
        Ok(self
            .inner
            .lock()
            .map_err(|_| Error::Denied)?
            .worker
            .snapshot()
            .epoch
            .0)
    }
}

/// Coordinator-only issuer pinned to one client token, plan and local lease.
/// The random public ID binds the remote owner; no token/hash leaves this object.
pub struct StageIssuer {
    access: Access,
    coordinator: NodeId,
    token_hash: String,
    plan: PlacementPlan,
    lease: starlings_protocol::Lease,
    id: RequestId,
}
impl EnrollmentReader {
    pub fn stage_issuer(
        &self,
        token: &ClientToken,
        plan: PlacementPlan,
        lease: starlings_protocol::Lease,
    ) -> Result<StageIssuer> {
        self.issuer(hash(&token.secret), plan, lease)
    }
    /// The same issuer for the live admitted HTTP client: its grant must be open and scoped to
    /// Infer on this plan's model. The client's token hash stays inside this crate.
    pub fn stage_issuer_for(
        &self,
        client: &crate::ClientResponse,
        plan: PlacementPlan,
        lease: starlings_protocol::Lease,
    ) -> Result<StageIssuer> {
        if client.scope()
            != &(Scope {
                model: plan.manifest_digest.clone(),
                action: Action::Infer,
            })
        {
            return Err(Error::Denied);
        }
        client.watch()?.check()?;
        self.issuer(client.token_hash().to_owned(), plan, lease)
    }
    fn issuer(
        &self,
        token_hash: String,
        plan: PlacementPlan,
        lease: starlings_protocol::Lease,
    ) -> Result<StageIssuer> {
        starlings_protocol::encode(&plan, FixtureMode::Reject)?;
        starlings_protocol::encode(&lease, FixtureMode::Reject)?;
        if plan.mode != PlacementPlanMode::LayerPipeline
            || lease.plan_digest != plan.plan_digest
            || lease.authority_epoch != plan.authority_epoch
            || lease.cluster_id != plan.cluster_id
            || !plan.stages.iter().any(|s| s.node_id == lease.owner_node_id)
        {
            return Err(Error::Denied);
        }
        self.access
            .lock()?
            .plan(&plan, &(token_hash.clone(), self.identity.node()))?;
        Ok(StageIssuer {
            access: self.access.clone(),
            coordinator: self.identity.node(),
            token_hash,
            plan,
            lease,
            id: uuid::Uuid::new_v4()
                .to_string()
                .parse()
                .map_err(|_| Error::Invalid)?,
        })
    }
}
impl StageIssuer {
    pub fn id(&self) -> RequestId {
        self.id
    }
    pub fn assign(&self, stream: &mut GenerationStream) -> Result<()> {
        let result = (|| {
            self.current(stream)?;
            stream.send_record_limit(
                &Assignment {
                    version: Version,
                    plan: self.plan.clone(),
                    issuer: self.id,
                },
                GRANT_RECORD,
            )
        })();
        if result.is_err() {
            stream.watch().cancel();
        }
        result
    }
    fn current(&self, stream: &GenerationStream) -> Result<NodeId> {
        let (access, node) = stream.coordinator_context()?;
        if !access.same(&self.access)
            || node != self.lease.owner_node_id
            || self.plan.manifest_digest != stream.binding().model
            || self.plan.execution_profile != stream.binding().profile
            || self.plan.authority_epoch != stream.binding().epoch.number
        {
            return Err(Error::Denied);
        }
        let state = self.access.lock()?;
        if state.draining {
            return Err(Error::Denied);
        }
        state.plan(&self.plan, &(self.token_hash.clone(), self.coordinator))?;
        for stage in &self.plan.stages {
            state.credential(&state.member(stage.node_id)?)?;
        }
        Ok(node)
    }
    pub fn respond(&self, stream: &mut GenerationStream) -> Result<()> {
        let result = (|| {
            let challenge: Challenge = stream.receive_record_limit(GRANT_RECORD)?;
            let node = self.current(stream)?;
            let expected = RouteScope {
                epoch: stream.binding().epoch,
                issuer: self.id,
                plan: self.plan.plan_digest.clone(),
                node,
                lease: self.lease.lease_id,
                source: challenge.scope.source,
                destination: challenge.scope.destination,
            };
            if challenge.scope != expected {
                return Err(Error::Denied);
            }
            let peer = {
                let state = self.access.lock()?;
                state.member(route_peer(
                    &self.plan,
                    node,
                    expected.source,
                    expected.destination,
                )?)?
            };
            stream.send_record_limit(
                &Reply {
                    version: Version,
                    challenge,
                    peer,
                },
                GRANT_RECORD,
            )
        })();
        if result.is_err() {
            stream.watch().cancel();
        }
        result
    }
}
fn route_peer(
    plan: &PlacementPlan,
    node: NodeId,
    source: BoundedU32<0, 7>,
    destination: BoundedU32<0, 7>,
) -> Result<NodeId> {
    let src = plan
        .stages
        .get(source.get() as usize)
        .ok_or(Error::Denied)?;
    let dst = plan
        .stages
        .get(destination.get() as usize)
        .ok_or(Error::Denied)?;
    if (source.get() + 1 != destination.get()
        && !(source.get() as usize + 1 == plan.stages.len() && destination.get() == 0))
        || src.node_id == dst.node_id
        || (node != src.node_id && node != dst.node_id)
    {
        return Err(Error::Denied);
    }
    Ok(if node == src.node_id {
        dst.node_id
    } else {
        src.node_id
    })
}
#[cfg(test)]
mod tests;

#[cfg(test)]
mod process_tests;
