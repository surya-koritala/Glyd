//! Trusted local ownership only; no provider process, native eligibility or drain proof.
use super::*;
use crate::ReceivedTensors;
use starlings_protocol::{GenerationId, InvocationId, TensorHeader};
use std::sync::mpsc::{SyncSender, TrySendError};

type Step = (GenerationId, InvocationId);
/// The invocation ID of one pipeline step, the same on every stage: SHA-256 of the generation
/// and sequence, shaped as a version-4 UUID because protocol IDs require that form.
pub fn step_invocation(generation: GenerationId, sequence: u64) -> Result<InvocationId> {
    let digest = crate::hash(format!("starlings-step/1 {generation} {sequence}").as_bytes());
    let mut bytes = [0u8; 16];
    for (byte, pair) in bytes.iter_mut().zip(digest.as_bytes().chunks_exact(2)) {
        *byte = u8::from_str_radix(std::str::from_utf8(pair).map_err(|_| Error::Invalid)?, 16)
            .map_err(|_| Error::Invalid)?;
    }
    bytes[6] = bytes[6] & 0x0f | 0x40;
    bytes[8] = bytes[8] & 0x3f | 0x80;
    uuid::Uuid::from_bytes(bytes)
        .to_string()
        .parse()
        .map_err(|_| Error::Invalid)
}
pub(super) struct BackendOwner {
    runtime: Arc<Mutex<RuntimeWorker>>,
    hold: starlings_core::BackendHold,
    lease: LeaseId,
    inputs: Mutex<BTreeMap<Step, Arc<ReceivedTensors>>>,
}
/// The supervisor must retain its separate durable process/catalog owner as well.
/// Dropping this handle cannot acknowledge cleanup or release retained inputs.
pub struct StageBackend(Arc<BackendOwner>);
/// One admitted queue item; dropping it withdraws output but never acknowledges drain.
pub struct StageChunk {
    owner: Arc<BackendOwner>,
    grant: StageGrant,
    input: Arc<ReceivedTensors>,
}
#[derive(Clone)]
pub(in crate::authorization) struct NativeGuard {
    owner: Arc<BackendOwner>,
    input: Arc<ReceivedTensors>,
}
impl NativeGuard {
    pub(super) fn registered(&self, runtime: &RuntimeWorker) -> Result<()> {
        registered(&self.owner, &self.input, runtime)
    }
    pub(super) fn pending(&self, runtime: &RuntimeWorker) -> Result<()> {
        self.registered(runtime)?;
        let scope = self.input.scope();
        if !runtime
            .worker
            .can_publish_step(scope.generation_id, scope.invocation_id)
        {
            return Err(Error::Expired);
        }
        Ok(())
    }
}
fn registered(
    owner: &BackendOwner,
    input: &Arc<ReceivedTensors>,
    runtime: &RuntimeWorker,
) -> Result<()> {
    let scope = input.scope();
    if runtime.worker.check_backend_hold(&owner.hold) != Ok(owner.lease)
        || !owner
            .inputs
            .lock()
            .map_err(|_| Error::Denied)?
            .get(&(scope.generation_id, scope.invocation_id))
            .is_some_and(|held| Arc::ptr_eq(held, input))
    {
        return Err(Error::Expired);
    }
    Ok(())
}
impl StageWorker {
    /// Retain before native preparation; the peer's logical Prepare is insufficient.
    pub fn hold_backend(&self, stream: &GenerationStream, lease: LeaseId) -> Result<StageBackend> {
        let mut runtime = self.inner.lock().map_err(|_| Error::Denied)?;
        let fence = self.context(&mut runtime, stream)?;
        let guard = fence.lock()?;
        if guard.status() != (stream.binding().epoch, crate::NodeStatus::Online)
            || !runtime.owns(&self.plan, stream)
        {
            return Err(Error::Expired);
        }
        // The lease is this worker's plan's, not another request's on the shared runtime.
        if runtime.worker.plan_of(lease).map(|p| &p.plan_digest) != Some(&self.plan.plan_digest) {
            return Err(Error::Denied);
        }
        let mut owners = self.backends.lock().map_err(|_| Error::Denied)?;
        owners.retain(|_, owner| runtime.worker.check_backend_hold(&owner.hold).is_ok());
        if owners.contains_key(&lease) || owners.len() >= starlings_core::MAX_RECORDS {
            return Err(Error::Capacity);
        }
        let now = runtime.monotonic_ms(0);
        let owner = Arc::new(BackendOwner {
            hold: runtime
                .worker
                .hold_backend(lease, now)
                .map_err(|_| Error::Denied)?,
            runtime: self.inner.clone(),
            lease,
            inputs: Mutex::new(BTreeMap::new()),
        });
        owners.insert(lease, owner.clone());
        Ok(StageBackend(owner))
    }

    /// Atomically start one complete local boundary and enqueue without blocking.
    /// The owner validates registered BoundaryContract/RunChunk layouts separately.
    /// Pipe I/O runs outside this method and all runtime/fence/grant locks.
    pub fn enqueue_chunk(
        &self,
        grant: &StageGrant,
        backend: &StageBackend,
        input: ReceivedTensors,
        queue: &SyncSender<StageChunk>,
    ) -> Result<()> {
        self.enqueue(grant, backend, Arc::new(input), queue)
    }
    /// Trusted local first-stage start of one prompt chunk the local owner tokenized itself.
    /// It runs on the last-to-first token route's live grant with the same hold and atomic core
    /// admission as a received token; only the next prefill step with its derived ID is accepted.
    pub fn enqueue_prompt(
        &self,
        grant: &StageGrant,
        backend: &StageBackend,
        header: TensorHeader,
        tokens: Box<[u8]>,
        queue: &SyncSender<StageChunk>,
    ) -> Result<()> {
        starlings_protocol::encode(&header, FixtureMode::Reject)?;
        if header.tensor_role != starlings_protocol::TensorHeaderTensorRole::TokenIds
            || header.phase != starlings_protocol::TensorHeaderPhase::Prefill
            || header.invocation_id != step_invocation(header.generation_id, header.sequence.0)?
            || header.destination_stage.get() != 0
            || header.source_stage.get() as usize + 1 != self.plan.stages.len()
            || header.chunk_offset_bytes.0 != 0
            || !header.end_of_tensor
            || tokens.len() as u64 != header.total_tensor_bytes.0
        {
            return Err(Error::Denied);
        }
        self.enqueue(
            grant,
            backend,
            Arc::new(ReceivedTensors::local(header, tokens)),
            queue,
        )
    }
    /// GLY-360, first stage: the next decode step from the last stage's admitted message (one
    /// greedy token per row of the step before), the `kept` rows of that step whose drafts held,
    /// and this stage's own drafts: [message[kept], drafts..] at the position after the kept rows.
    /// The session checks that position against the rows the step before may leave.
    pub fn enqueue_decode(
        &self,
        grant: &StageGrant,
        backend: &StageBackend,
        received: &ReceivedTensors,
        kept: u32,
        drafts: &[u32],
        queue: &SyncSender<StageChunk>,
    ) -> Result<()> {
        let ([h], [tokens]) = (received.headers(), received.tensors()) else {
            return Err(Error::Denied);
        };
        let rows = 1 + u32::try_from(drafts.len()).map_err(|_| Error::Denied)?;
        if h.tensor_role != starlings_protocol::TensorHeaderTensorRole::TokenIds
            || h.phase != starlings_protocol::TensorHeaderPhase::Decode
            || h.destination_stage.get() != 0
            || kept >= h.token_count.get()
            || tokens.len() as u64 != 4 * u64::from(h.token_count.get())
            || rows > starlings_protocol::MAX_DECODE_ROWS
        {
            return Err(Error::Denied);
        }
        use starlings_protocol::{BoundedU32, Decimal};
        let mut header = h.clone();
        header.position_start =
            BoundedU32::new(h.position_start.get() + kept).map_err(|_| Error::Denied)?;
        header.token_count = BoundedU32::new(rows).map_err(|_| Error::Denied)?;
        header.shape = vec![BoundedU32::new(rows).map_err(|_| Error::Denied)?];
        header.total_tensor_bytes = Decimal(4 * u64::from(rows));
        header.chunk_offset_bytes = Decimal(0);
        header.chunk_bytes = header.total_tensor_bytes;
        header.end_of_tensor = true;
        starlings_protocol::encode(&header, FixtureMode::Reject)?;
        let at = 4 * kept as usize;
        let bytes: Box<[u8]> = tokens[at..at + 4]
            .iter()
            .copied()
            .chain(drafts.iter().flat_map(|d| d.to_le_bytes()))
            .collect();
        self.enqueue(
            grant,
            backend,
            Arc::new(ReceivedTensors::local(header, bytes)),
            queue,
        )
    }
    fn enqueue(
        &self,
        grant: &StageGrant,
        backend: &StageBackend,
        input: Arc<ReceivedTensors>,
        queue: &SyncSender<StageChunk>,
    ) -> Result<()> {
        if !Arc::ptr_eq(&self.inner, &backend.0.runtime)
            || !grant.0.runtime.ptr_eq(&Arc::downgrade(&self.inner))
            // The grant is for this worker's plan (one runtime holds several, GLY-190).
            || grant.0.scope.plan != self.plan.plan_digest
            || input.scope().lease_id != backend.0.lease
            || grant.0.scope.lease != backend.0.lease
            || input.scope().source_stage != grant.0.scope.source
            || input.scope().destination_stage != grant.0.scope.destination
            || input.headers().len() != input.tensors().len()
            || input
                .headers()
                .iter()
                .zip(input.tensors())
                .any(|(h, bytes)| {
                    starlings_protocol::DataScope::from(h) != *input.scope()
                        || h.total_tensor_bytes.0 != bytes.len() as u64
                })
        {
            return Err(Error::Denied);
        }
        let key = (input.scope().generation_id, input.scope().invocation_id);
        let sent = grant.with_runtime_checked(
            |runtime| {
                if runtime.worker.check_backend_hold(&backend.0.hold) != Ok(backend.0.lease) {
                    return Err(Error::Expired);
                }
                Ok(())
            },
            |runtime| {
                let mut held = backend.0.inputs.lock().map_err(|_| Error::Denied)?;
                if held.contains_key(&key) {
                    return Err(Error::Replay);
                }
                let now = runtime.monotonic_ms(0);
                runtime
                    .worker
                    .start_local_chunk(&backend.0.hold, input.headers(), now)
                    .map_err(|_| Error::Denied)?;
                held.insert(key, input.clone());
                // A failed nonblocking send proves this item was never delivered.
                Ok(queue.try_send(StageChunk {
                    owner: backend.0.clone(),
                    grant: grant.clone(),
                    input: input.clone(),
                }))
            },
        )?;
        match sent {
            Ok(()) => Ok(()),
            Err(TrySendError::Full(chunk) | TrySendError::Disconnected(chunk)) => {
                // Drop/cleanup occurs after leaving authority/runtime locks.
                chunk.qualified_drained(true)?;
                Err(Error::Capacity)
            }
        }
    }
}
impl StageBackend {
    /// Qualified full-instance cleanup, including abandoned queue items, never a receipt.
    /// The supervisor proves the complete instance stopped before calling this method.
    pub fn qualified_unloaded(&self) -> Result<()> {
        let mut released = Vec::new();
        {
            let mut runtime = self.0.runtime.lock().map_err(|_| Error::Denied)?;
            let mut held = self.0.inputs.lock().map_err(|_| Error::Denied)?;
            if runtime
                .worker
                .check_backend_hold(&self.0.hold)
                .map_err(|_| Error::Denied)?
                != self.0.lease
            {
                return Err(Error::Denied);
            }
            runtime
                .worker
                .withdraw_lease(self.0.lease)
                .map_err(|_| Error::Denied)?;
            let keys: Vec<_> = held.keys().copied().collect();
            for key in keys {
                runtime
                    .worker
                    .failed_and_drained(key.0, key.1)
                    .map_err(|_| Error::Denied)?;
                released.push(held.remove(&key));
            }
            runtime
                .worker
                .backend_unloaded(&self.0.hold)
                .map_err(|_| Error::Denied)?;
        }
        drop(released);
        Ok(())
    }
}
impl BackendOwner {
    fn drained(&self, key: Step, expected: &Arc<ReceivedTensors>, failed: bool) -> Result<()> {
        let released = {
            let mut runtime = self.runtime.lock().map_err(|_| Error::Denied)?;
            let mut held = self.inputs.lock().map_err(|_| Error::Denied)?;
            if runtime
                .worker
                .check_backend_hold(&self.hold)
                .map_err(|_| Error::Denied)?
                != self.lease
                || !held
                    .get(&key)
                    .is_some_and(|input| Arc::ptr_eq(input, expected))
            {
                return Err(Error::Denied);
            }
            if failed {
                runtime.worker.failed_and_drained(key.0, key.1)
            } else {
                runtime.worker.drained(key.0, key.1)
            }
            .map_err(|_| Error::Denied)?;
            held.remove(&key)
        };
        drop(released);
        Ok(())
    }
}
impl StageChunk {
    pub fn input(&self) -> &ReceivedTensors {
        &self.input
    }
    /// Recheck immediately before native dispatch; no pipe I/O is performed here.
    pub fn check_dispatch(&self) -> Result<()> {
        self.grant.with_runtime_checked(
            |runtime| self.registered(runtime),
            |runtime| self.pending(runtime),
        )
    }
    pub(in crate::authorization) fn guard(&self) -> NativeGuard {
        NativeGuard {
            owner: self.owner.clone(),
            input: self.input.clone(),
        }
    }
    pub(in crate::authorization) fn registered(&self, runtime: &RuntimeWorker) -> Result<()> {
        registered(&self.owner, &self.input, runtime)
    }
    fn pending(&self, runtime: &RuntimeWorker) -> Result<()> {
        self.registered(runtime)?;
        if runtime
            .worker
            .check_backend_hold(&self.owner.hold)
            .map_err(|_| Error::Denied)?
            != self.owner.lease
            || !runtime.worker.can_publish_step(
                self.input.scope().generation_id,
                self.input.scope().invocation_id,
            )
        {
            return Err(Error::Expired);
        }
        Ok(())
    }
    pub(in crate::authorization) fn output(
        &self,
        runtime: &RuntimeWorker,
        h: &TensorHeader,
    ) -> Result<()> {
        self.pending(runtime)?;
        let scope = self.input.scope();
        let input = &self.input.headers()[0];
        if h.plan_digest != scope.plan_digest
            || h.authority_epoch != scope.authority_epoch
            || h.session_id != scope.session_id
            || h.generation_id != scope.generation_id
            || h.source_stage != scope.destination_stage
        {
            return Err(Error::Denied);
        }
        if h.tensor_role == starlings_protocol::TensorHeaderTensorRole::TokenIds {
            // GLY-188: the last stage's sampled token is the first stage's next decode step.
            // Data admission already limits token IDs to the last-to-first route; the receiver's
            // live session still checks its position, sequence and output bound.
            // Only a head step samples: a decode step, or the chunk that ends the prompt.
            // GLY-360: after a decode step, one token per row (each row's greedy token checks
            // the draft after it), for positions start+1..=end; after the prompt's last chunk,
            // one token at its end.
            let next = scope.sequence.0.checked_add(1).ok_or(Error::Denied)?;
            let end = input
                .position_start
                .get()
                .checked_add(input.token_count.get())
                .ok_or(Error::Denied)?;
            let decode = input.phase == starlings_protocol::TensorHeaderPhase::Decode;
            let rows = if decode { input.token_count.get() } else { 1 };
            if h.sequence.0 != next
                || h.invocation_id != step_invocation(h.generation_id, next)?
                || h.destination_stage.get() != 0
                || h.phase != starlings_protocol::TensorHeaderPhase::Decode
                || h.token_count.get() != rows
                || h.dtype != starlings_protocol::TensorHeaderDtype::U32
                || h.shape.len() != 1
                || h.shape[0].get() != rows
                || h.total_tensor_bytes.0 != 4 * u64::from(rows)
                || h.position_start.get() != end + 1 - rows
                || !decode && Some(end) != runtime.worker.prompt_tokens(scope.generation_id)
            {
                return Err(Error::Denied);
            }
            return Ok(());
        }
        if h.invocation_id != scope.invocation_id
            || h.sequence != scope.sequence
            || h.phase != input.phase
            || h.position_start != input.position_start
            || h.token_count != input.token_count
        {
            return Err(Error::Denied);
        }
        Ok(())
    }
    /// Trusted local owner only, after qualified device completion. Provider receipts,
    /// status and allocation counters alone do not prove completion or safe buffer release.
    pub fn qualified_drained(&self, failed: bool) -> Result<()> {
        let scope = self.input.scope();
        self.owner.drained(
            (scope.generation_id, scope.invocation_id),
            &self.input,
            failed,
        )
    }
    pub(in crate::authorization) fn abort(&self) -> Result<()> {
        let mut runtime = self.owner.runtime.lock().map_err(|_| Error::Denied)?;
        let held = self.owner.inputs.lock().map_err(|_| Error::Denied)?;
        let scope = self.input.scope();
        if runtime.worker.check_backend_hold(&self.owner.hold) != Ok(self.owner.lease)
            || !held
                .get(&(scope.generation_id, scope.invocation_id))
                .is_some_and(|input| Arc::ptr_eq(input, &self.input))
        {
            return Err(Error::Expired);
        }
        runtime
            .worker
            .abort_data(self.input.headers().first().ok_or(Error::Invalid)?)
            .map_err(|_| Error::Denied)
    }
}
impl Drop for StageChunk {
    fn drop(&mut self) {
        let _ = self.abort();
        // StageWorker's bounded owner bag retains these bytes until qualified cleanup.
    }
}
