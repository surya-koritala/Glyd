//! Whole-request lifecycle on a remote worker with no authority database.
use super::*;
use crate::{GenerationStream, RegistryFence};
use starlings_core::routing::WarmReplica;
use starlings_protocol::{FixtureMode, PlacementPlanMode};

/// The pinned coordinator delegates client authorization; the worker independently
/// enforces its approved profile, single-service capacity, epoch, replay and leases.
/// Its monotonic clock is sampled under the state lock, including during maintenance.
pub struct EnrolledWorker {
    inner: Arc<Mutex<RuntimeWorker>>,
    profile: WarmReplica,
}
impl EnrolledWorker {
    #[cfg(test)]
    pub(crate) fn advance_clock(&self, duration: std::time::Duration) {
        let mut runtime = self.inner.lock().unwrap();
        runtime.clock = Some(runtime.clock.unwrap() - duration);
    }
    pub fn profile(&self) -> &WarmReplica {
        &self.profile
    }
    /// `profile` is trusted local qualification, never a network advertisement.
    /// It is immutable until this runtime has been drained and dropped.
    pub fn new(worker: Worker, fence: RegistryFence, profile: WarmReplica) -> Result<Self> {
        profile.validate().map_err(|_| Error::Invalid)?;
        let identity = worker.recovery();
        let snapshot = worker.snapshot();
        fence.check_identity(identity.node_id, identity.cluster_id)?;
        if profile.node != identity.node_id
            || profile.max_sessions != 1
            || !profile.accepting
            || !snapshot.instances.is_empty()
            || !snapshot.generations.is_empty()
        {
            return Err(Error::Denied);
        }
        let inner = Arc::new(Mutex::new(RuntimeWorker {
            registry: Some(fence.clone()),
            generation: None,
            stages: BTreeMap::new(),
            slots: 1,
            clock: Some(std::time::Instant::now()),
            worker,
            owners: BTreeMap::new(),
        }));
        fence.register(&inner, identity.node_id, identity.cluster_id)?;
        Ok(Self { inner, profile })
    }
    /// Only new accepted StartRequest operations may invoke the backend. `true` is
    /// a historical replay acknowledgement and must never dispatch another attempt.
    pub fn apply(&self, stream: &GenerationStream, request: &ControlRequest) -> Result<bool> {
        request.validate(FixtureMode::Reject)?;
        if matches!(
            request.operation,
            Operation::CompleteStep { .. } | Operation::StartStep { .. } | Operation::Begin { .. }
        ) {
            return Err(Error::Denied);
        }
        let mut runtime = self.inner.lock().map_err(|_| Error::Denied)?;
        let watch = stream.watch();
        let owns = runtime
            .generation
            .as_ref()
            .is_some_and(|owner| owner.same(&watch));
        if !owns
            && (runtime.worker.reserved_bytes() != 0
                || !matches!(request.operation, Operation::Prepare { .. }))
        {
            return Err(Error::Denied);
        }
        let fence = self.fence(&runtime, stream)?;
        let guard = fence.lock()?;
        if guard.status() != (stream.binding().epoch, crate::NodeStatus::Online)
            || !runtime.reconcile_registry_state(0, Some(&guard))?
        {
            runtime.worker.retain_plans(|_| false).map_err(core)?;
            return Err(Error::Expired);
        }
        let plan = runtime.worker.request_plan(request).map_err(core)?;
        self.plan(plan)?;
        if let Operation::Prepare { lease, .. } = &request.operation
            && runtime
                .worker
                .snapshot()
                .instances
                .iter()
                .any(|(id, state)| {
                    *id != lease.lease_id && *state != starlings_core::InstanceState::Released
                })
        {
            return Err(Error::Capacity);
        }
        let now = runtime.monotonic_ms(0);
        let result = runtime.worker.apply(request, now).map_err(core);
        if result == Ok(false) && !owns {
            runtime.generation = Some(watch);
        }
        drop(guard);
        if stream.watch().check().is_err() {
            runtime.worker.retain_plans(|_| false).map_err(core)?;
            return Err(Error::Expired);
        }
        result
    }
    fn fence(&self, runtime: &RuntimeWorker, stream: &GenerationStream) -> Result<RegistryFence> {
        stream.watch().check()?;
        let fence = runtime.registry.as_ref().ok_or(Error::Denied)?;
        if !fence.same(stream.worker_fence()?)
            || stream.binding().model != self.profile.manifest
            || stream.binding().profile != self.profile.profile
        {
            return Err(Error::Denied);
        }
        Ok(fence.clone())
    }
    fn plan(&self, plan: &PlacementPlan) -> Result<()> {
        let [stage] = plan.stages.as_slice() else {
            return Err(Error::Denied);
        };
        if plan.mode != PlacementPlanMode::WholeRequest
            || plan.manifest_digest != self.profile.manifest
            || plan.execution_profile != self.profile.profile
            || plan.context_limit.get() > self.profile.context_limit
            || plan.max_sessions.get() != 1
            || stage.node_id != self.profile.node
            || stage.backend_id != self.profile.backend
            || stage.backend_build_digest != self.profile.build
            || stage.memory_pool_id != self.profile.pool
            || stage.reserved_bytes()? < self.profile.request_bytes
        {
            return Err(Error::Denied);
        }
        Ok(())
    }
    /// A short synchronous enqueue only; never perform network/backend I/O or
    /// reenter this runtime from the callback. The registry fence remains held.
    pub fn with_output<T>(
        &self,
        stream: &GenerationStream,
        generation: GenerationId,
        enqueue: impl FnOnce() -> T,
    ) -> Result<T> {
        let mut runtime = self.inner.lock().map_err(|_| Error::Denied)?;
        if !runtime
            .generation
            .as_ref()
            .is_some_and(|owner| owner.same(&stream.watch()))
        {
            return Err(Error::Denied);
        }
        let fence = self.fence(&runtime, stream)?;
        let guard = fence.lock()?;
        if guard.status() != (stream.binding().epoch, crate::NodeStatus::Online)
            || !runtime.reconcile_registry_state(0, Some(&guard))?
            || !runtime.worker.can_publish(generation)
        {
            return Err(Error::Expired);
        }
        Ok(enqueue())
    }
    /// Connection loss/client cancellation withdraws every lease, never its in-flight buffers.
    pub fn withdraw(&self, stream: &GenerationStream) -> Result<()> {
        let mut runtime = self.inner.lock().map_err(|_| Error::Denied)?;
        if !runtime
            .generation
            .as_ref()
            .is_some_and(|owner| owner.same(&stream.watch()))
        {
            return Err(Error::Denied);
        }
        if let Some(connection) = &runtime.generation {
            connection.cancel();
        }
        runtime.worker.retain_plans(|_| false).map_err(core)
    }
    /// Trusted local completion only, including after authority/connection loss.
    pub fn drained(&self, generation: GenerationId, invocation: InvocationId) -> Result<()> {
        self.inner
            .lock()
            .map_err(|_| Error::Denied)?
            .worker
            .drained(generation, invocation)
            .map_err(core)
    }
    /// Trusted local pressure/shutdown only. Close output and retain in-flight allocations.
    pub fn withdraw_local(&self) -> Result<()> {
        let mut runtime = self.inner.lock().map_err(|_| Error::Denied)?;
        if let Some(connection) = &runtime.generation {
            connection.cancel();
        }
        runtime.worker.retain_plans(|_| false).map_err(core)
    }
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
            .map_err(core)
    }
    pub fn snapshot(&self) -> Result<Snapshot> {
        Ok(self
            .inner
            .lock()
            .map_err(|_| Error::Denied)?
            .worker
            .snapshot())
    }
    /// Only the local supervisor can acknowledge qualified whole-backend shutdown.
    /// No control operation or peer-supplied completion can enter this path.
    pub fn backend_shutdown_drained(&self) -> Result<()> {
        let mut runtime = self.inner.lock().map_err(|_| Error::Denied)?;
        if let Some(connection) = &runtime.generation {
            connection.cancel();
        }
        runtime.worker.backend_shutdown_drained().map_err(core)
    }
}
fn core(error: starlings_core::CoreError) -> Error {
    match error {
        starlings_core::CoreError::ReplayExpired => Error::ReplayExpired,
        starlings_core::CoreError::Capacity | starlings_core::CoreError::Limit => Error::Capacity,
        _ => Error::Invalid,
    }
}
