//! Trusted local data admission over the existing secured-worker policy/runtime.
use super::*;
use crate::data::{DataBinding, DataFrame};
use starlings_protocol::{DataControl, DataHeader, TensorHeader};
use std::time::Instant;

/// Created by the local lifecycle owner, never deserialized from a data peer.
/// Use the same monotonic clock origin as control dispatch/maintenance.
enum DataOwner<'a> {
    Authority {
        worker: &'a SecuredWorker,
        token_hash: String,
        clock: Instant,
    },
    Stage {
        worker: &'a StageWorker,
        grant: StageGrant,
        step: Option<&'a StageChunk>,
    },
}
pub struct DataAdmission<'a> {
    owner: DataOwner<'a>,
    expected: TensorHeader,
    binding: DataBinding,
    scratch_budget: u64,
}
/// What a data check admits as input: none, the step now due, or (opening a stage's
/// receive only) the step right after the one it is running.
#[derive(Clone, Copy, PartialEq)]
pub(crate) enum Input {
    No,
    Due,
    Next,
}
impl Input {
    fn due(input: bool) -> Self {
        if input { Self::Due } else { Self::No }
    }
}
impl SecuredWorker {
    /// `expected` is the locally admitted boundary template. Chunk extents may vary;
    /// every other field must match it. This does not reserve or execute anything.
    pub fn data_admission<'a>(
        &'a self,
        peer: &PeerStream,
        token: &ClientToken,
        expected: TensorHeader,
        clock: Instant,
    ) -> Result<DataAdmission<'a>> {
        starlings_protocol::encode(&expected, starlings_protocol::FixtureMode::Reject)?;
        let token_hash = hash(&token.secret);
        let (binding, scratch_budget) =
            self.data_context(peer, &token_hash, &expected, clock, false, |b, scratch| {
                Ok((b.clone(), scratch))
            })?;
        Ok(DataAdmission {
            owner: DataOwner::Authority {
                worker: self,
                token_hash,
                clock,
            },
            expected,
            binding,
            scratch_budget,
        })
    }
    fn data_context<T>(
        &self,
        peer: &PeerStream,
        token_hash: &str,
        expected: &TensorHeader,
        clock: Instant,
        input: bool,
        dispatch: impl FnOnce(&DataBinding, u64) -> Result<T>,
    ) -> Result<T> {
        let (access, live) = peer.authority()?;
        if !self.access.same(access) || peer.channel != crate::Channel::Data {
            return Err(Error::Denied);
        }
        let state = self.access.lock()?;
        if state.draining {
            return Err(Error::Denied);
        }
        state.credential(&live.local)?;
        state.credential(&live.peer)?;
        let mut runtime = self.inner.lock().map_err(|_| Error::Denied)?;
        let fence = runtime.registry.clone().ok_or(Error::Denied)?;
        let guard = fence.lock()?;
        let (epoch, status) = guard.status();
        let now = u64::try_from(clock.elapsed().as_millis()).map_err(|_| Error::Expired)?;
        if status != crate::NodeStatus::Online
            || epoch.number != expected.authority_epoch
            || !runtime.reconcile_registry_state(now, Some(&guard))?
        {
            return Err(Error::Expired);
        }
        let plan = runtime
            .worker
            .data_plan(expected)
            .map_err(|_| Error::Denied)?;
        let owner = runtime.owners.get(&plan.plan_id).ok_or(Error::Denied)?;
        if owner.0 != token_hash {
            return Err(Error::Denied);
        }
        state.plan(plan, owner)?;
        let source = &plan.stages[expected.source_stage.get() as usize];
        let destination = &plan.stages[expected.destination_stage.get() as usize];
        let local = runtime.worker.node_id();
        let remote = if local == source.node_id {
            destination.node_id
        } else {
            source.node_id
        };
        if live.local.node != local || peer.peer() != remote || peer.role() != Role::Worker {
            return Err(Error::Denied);
        }
        if input {
            runtime
                .worker
                .validate_data_input(expected)
                .map_err(|_| Error::Denied)?;
        }
        let binding = DataBinding {
            version: Version,
            epoch,
            plan: plan.plan_digest.clone(),
            source: source.node_id,
            destination: destination.node_id,
            source_stage: expected.source_stage,
            destination_stage: expected.destination_stage,
        };
        let scratch = plan
            .stages
            .iter()
            .find(|s| s.node_id == local)
            .ok_or(Error::Denied)?
            .scratch_budget_bytes
            .0;
        dispatch(&binding, scratch)
    }
}
impl StageWorker {
    /// Caller supplies the approved local boundary, never a template selected by the peer.
    pub fn data_admission<'a>(
        &'a self,
        peer: &PeerStream,
        grant: &StageGrant,
        expected: TensorHeader,
    ) -> Result<DataAdmission<'a>> {
        starlings_protocol::encode(&expected, starlings_protocol::FixtureMode::Reject)?;
        let (binding, scratch_budget) =
            self.data_context(peer, grant, &expected, Input::No, None, |b, scratch, _| {
                Ok((b.clone(), scratch))
            })?;
        Ok(DataAdmission {
            owner: DataOwner::Stage {
                worker: self,
                grant: grant.clone(),
                step: None,
            },
            expected,
            binding,
            scratch_budget,
        })
    }
    /// Bind native output to its exact pending local step and actual output route.
    /// The owner still validates the registered output layout before constructing this.
    pub fn output_admission<'a>(
        &'a self,
        peer: &PeerStream,
        grant: &StageGrant,
        expected: TensorHeader,
        step: &'a StageChunk,
    ) -> Result<DataAdmission<'a>> {
        starlings_protocol::encode(&expected, starlings_protocol::FixtureMode::Reject)?;
        let (binding, scratch_budget) = self.data_context(
            peer,
            grant,
            &expected,
            Input::No,
            Some(step),
            |binding, scratch, runtime| {
                step.output(runtime, &expected)?;
                Ok((binding.clone(), scratch))
            },
        )?;
        Ok(DataAdmission {
            owner: DataOwner::Stage {
                worker: self,
                grant: grant.clone(),
                step: Some(step),
            },
            expected,
            binding,
            scratch_budget,
        })
    }
}
impl DataAdmission<'_> {
    pub(crate) fn transport_grant(&self) -> Option<StageGrant> {
        match &self.owner {
            DataOwner::Stage {
                grant,
                step: Some(step),
                ..
            } => Some(grant.for_step(step)),
            _ => None,
        }
    }
    /// The immutable local admission selects the generation, never a received header.
    /// Cancellation cannot depend on the failed peer or refresh revoked authority.
    pub(crate) fn abort(&self) -> Result<()> {
        match &self.owner {
            DataOwner::Authority { worker, .. } => worker
                .inner
                .lock()
                .map_err(|_| Error::Denied)?
                .worker
                .abort_data(&self.expected)
                .map_err(|_| Error::Denied),
            DataOwner::Stage {
                step: Some(step), ..
            } => step.abort(),
            DataOwner::Stage {
                worker, step: None, ..
            } => worker.abort_data(&self.expected),
        }
    }
    pub(crate) fn scratch_budget(&self) -> u64 {
        self.scratch_budget
    }
    pub(crate) fn expected(&self) -> &TensorHeader {
        &self.expected
    }
    pub(crate) fn binding(&self) -> &DataBinding {
        &self.binding
    }
    pub(crate) fn check<T>(
        &self,
        peer: &PeerStream,
        header: Option<&DataHeader>,
        receiving: bool,
        dispatch: impl FnOnce() -> T,
    ) -> Result<T> {
        let input = self.header_input(peer, header, receiving)?;
        self.run(peer, Input::due(input), dispatch)
    }
    /// Opening a receive: `check` of the expected header, except that a stage may open the
    /// step right after the one it is running, so its credit goes out early. Every chunk is
    /// still read under `check`, which admits only the step now due.
    pub(crate) fn check_open(&self, peer: &PeerStream) -> Result<()> {
        let header = DataHeader::Tensor(self.expected.clone());
        let input = self.header_input(peer, Some(&header), true)?;
        self.run(peer, if input { Input::Next } else { Input::No }, || ())
    }
    fn header_input(
        &self,
        peer: &PeerStream,
        header: Option<&DataHeader>,
        receiving: bool,
    ) -> Result<bool> {
        Ok(match header {
            Some(DataHeader::Tensor(h)) => {
                let mut normalized = h.clone();
                normalized.chunk_offset_bytes = self.expected.chunk_offset_bytes;
                normalized.chunk_bytes = self.expected.chunk_bytes;
                normalized.end_of_tensor = self.expected.end_of_tensor;
                if normalized != self.expected
                    || receiving != (peer.local() == self.binding.destination)
                {
                    return Err(Error::Denied);
                }
                receiving
            }
            Some(DataHeader::Control(h)) => {
                if h.scope() != &starlings_protocol::DataScope::from(&self.expected) {
                    return Err(Error::Denied);
                }
                match h {
                    DataControl::Credit { bytes, .. }
                        if bytes.0 > self.expected.total_tensor_bytes.0 =>
                    {
                        return Err(Error::Capacity);
                    }
                    DataControl::ReceiptAck {
                        tensor_index,
                        received_through_bytes,
                        ..
                    } if *tensor_index != self.expected.tensor_index
                        || received_through_bytes.0 > self.expected.total_tensor_bytes.0 =>
                    {
                        return Err(Error::Invalid);
                    }
                    _ => (),
                }
                if !matches!(h, DataControl::Cancel { .. } | DataControl::Error { .. })
                    && receiving != (peer.local() == self.binding.source)
                {
                    return Err(Error::Denied);
                }
                false
            }
            None => false,
        })
    }
    fn run<T>(&self, peer: &PeerStream, input: Input, dispatch: impl FnOnce() -> T) -> Result<T> {
        let dispatch = |binding: &DataBinding, scratch: u64| {
            if binding != &self.binding || scratch != self.scratch_budget {
                return Err(Error::Expired);
            }
            Ok(dispatch())
        };
        match &self.owner {
            DataOwner::Authority {
                worker,
                token_hash,
                clock,
            } => {
                let input = input != Input::No;
                worker.data_context(peer, token_hash, &self.expected, *clock, input, dispatch)
            }
            DataOwner::Stage {
                worker,
                grant,
                step,
            } => worker.data_context(
                peer,
                grant,
                &self.expected,
                input,
                *step,
                |binding, scratch, runtime| {
                    if let Some(step) = step {
                        step.output(runtime, &self.expected)?;
                    }
                    dispatch(binding, scratch)
                },
            ),
        }
    }
    /// Short synchronous queue/dispatch only; never block or reenter policy/runtime.
    pub(crate) fn with_frame<T>(
        &self,
        peer: &PeerStream,
        frame: DataFrame,
        receiving: bool,
        dispatch: impl FnOnce(DataFrame) -> T,
    ) -> Result<T> {
        let input = self.header_input(peer, Some(frame.header()), receiving)?;
        self.run(peer, Input::due(input), || dispatch(frame))
    }
}
