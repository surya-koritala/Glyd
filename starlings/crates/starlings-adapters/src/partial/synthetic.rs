//! Explicit, bounded CPU byte-echo fixture. Never advertises native inference.
use super::*;
use starlings_protocol::{Digest, GenerationId, InvocationId};
use std::collections::BTreeSet;

struct Session {
    scope: DataScope,
    position: u32,
    decoded: u32,
    sequence: u64,
    seen: BTreeSet<InvocationId>,
}
struct Pending {
    tokens: u32,
    decode: bool,
    bytes: Vec<Vec<u8>>,
}
pub struct SyntheticStage {
    boundary: BoundaryContract,
    plan: Digest,
    budget: u64,
    committed: bool,
    closed: bool,
    cancelled: bool,
    session: Option<Session>,
    pending: Option<Pending>,
    ended: BTreeSet<GenerationId>,
    executions: u64,
}
impl SyntheticStage {
    pub fn prepare(
        boundary: BoundaryContract,
        plan: Digest,
        budget: u64,
        mode: FixtureMode,
    ) -> Result<Self> {
        if mode != FixtureMode::Simulation
            || !boundary.fixture
            || boundary.key.workload.sessions.get() != 1
            || boundary.key.workload.context_tokens.get() > 64
            || boundary.inputs != boundary.outputs
            || boundary.output_tokens != OutputTokens::SameChunk
            || budget == 0
            || budget > 64 * 1024
        {
            return Err(Error::Unsupported);
        }
        boundary.validate(mode)?;
        boundary.layouts(boundary.key.workload.prefill_chunk_tokens.get(), budget)?;
        Ok(Self {
            boundary,
            plan,
            budget,
            committed: false,
            closed: false,
            cancelled: false,
            session: None,
            pending: None,
            ended: BTreeSet::new(),
            executions: 0,
        })
    }
    pub fn describe(&self) -> &BoundaryContract {
        &self.boundary
    }
    pub fn inspect_model(&self) -> ModelCostProfile {
        ModelCostProfile {
            kind: "ModelCostProfile".into(),
            version: Version,
            fixture: true,
            key: self.boundary.key.clone(),
            costs: ModelCosts::Unavailable {
                code: "SYNTHETIC_ONLY".into(),
            },
        }
    }
    pub fn commit(&mut self) -> Result<()> {
        if self.closed {
            return Err(Error::State);
        }
        self.committed = true;
        Ok(())
    }
    pub fn begin(&mut self, scope: DataScope) -> Result<()> {
        if self.closed || !self.committed || self.session.is_some() {
            return Err(Error::State);
        }
        if scope.plan_digest != self.plan
            || scope.sequence.0 != 0
            || scope.authority_epoch.0 == 0
            || scope.source_stage == scope.destination_stage
            || self.ended.contains(&scope.generation_id)
        {
            return Err(Error::Conflict);
        }
        if self.ended.len() == 64 {
            return Err(Error::Capacity);
        }
        self.session = Some(Session {
            scope,
            position: 0,
            decoded: 0,
            sequence: 0,
            seen: BTreeSet::new(),
        });
        self.cancelled = false;
        Ok(())
    }
    pub fn run(&mut self, call: &RunChunk, bytes: Vec<Vec<u8>>) -> Result<()> {
        if self.closed || !self.committed || self.cancelled || self.pending.is_some() {
            return Err(Error::State);
        }
        let session = self.session.as_ref().ok_or(Error::State)?;
        if session.seen.contains(&call.step.scope.invocation_id) {
            return Err(Error::Conflict);
        }
        let workload = &self.boundary.key.workload;
        let decode = session.position >= workload.prompt_tokens.get();
        if (!decode
            && call.step.token_count.get() > workload.prompt_tokens.get() - session.position)
            || decode && session.decoded >= workload.output_tokens.get()
            // One-row decode steps only: this backend models no drafts (GLY-360).
            || decode && call.step.token_count.get() != 1
        {
            return Err(Error::Capacity);
        }
        let mut scope = session.scope.clone();
        scope.invocation_id = call.step.scope.invocation_id;
        scope.sequence = Decimal(session.sequence);
        let expected = StepIdentity {
            scope,
            phase: if decode {
                TensorHeaderPhase::Decode
            } else {
                TensorHeaderPhase::Prefill
            },
            position_start: BoundedU32::new(session.position).map_err(|_| Error::Capacity)?,
            token_count: call.step.token_count,
        };
        call.validate(
            &self.boundary,
            &expected,
            self.budget,
            FixtureMode::Simulation,
        )?;
        if bytes.len() != call.inputs.len()
            || bytes
                .iter()
                .zip(&call.inputs)
                .any(|(b, l)| b.len() as u64 != l.bytes.0)
            || bytes
                .iter()
                .try_fold(0u64, |n, b| n.checked_add(b.capacity() as u64))
                .is_none_or(|n| n > self.budget)
        {
            return Err(Error::Capacity);
        }
        let executions = self.executions.checked_add(1).ok_or(Error::Capacity)?;
        self.session
            .as_mut()
            .ok_or(Error::State)?
            .seen
            .insert(call.step.scope.invocation_id);
        self.pending = Some(Pending {
            tokens: call.step.token_count.get(),
            decode,
            bytes,
        });
        self.executions = executions;
        Ok(())
    }
    /// Only this fixture's local completion releases its held payloads. Not native drain proof.
    pub fn finish_synthetic(&mut self) -> Result<Option<Vec<Vec<u8>>>> {
        let pending = self.pending.as_ref().ok_or(Error::State)?;
        if self.cancelled {
            self.pending = None;
            return Ok(None);
        }
        let session = self.session.as_mut().ok_or(Error::State)?;
        let sequence = session.sequence.checked_add(1).ok_or(Error::Capacity)?;
        let position = session
            .position
            .checked_add(pending.tokens)
            .ok_or(Error::Capacity)?;
        let decoded = session
            .decoded
            .checked_add(u32::from(pending.decode))
            .ok_or(Error::Capacity)?;
        session.sequence = sequence;
        session.position = position;
        session.decoded = decoded;
        Ok(Some(self.pending.take().ok_or(Error::State)?.bytes))
    }
    /// True means no fixture payload remains; a request alone never frees busy buffers.
    pub fn cancel(&mut self) -> Result<bool> {
        if self.session.is_none() || self.closed {
            return Err(Error::State);
        }
        self.cancelled = true;
        Ok(self.pending.is_none())
    }
    pub fn end_session(&mut self) -> Result<()> {
        if self.pending.is_some() {
            return Err(Error::State);
        }
        let session = self.session.take().ok_or(Error::State)?;
        self.ended.insert(session.scope.generation_id);
        Ok(())
    }
    pub fn unload(&mut self) -> Result<()> {
        if self.pending.is_some() || self.session.is_some() {
            return Err(Error::State);
        }
        self.closed = true;
        Ok(())
    }
    pub fn executions(&self) -> u64 {
        self.executions
    }
    /// Payload capacities only; not process RSS, model memory or an allocator measurement.
    pub fn retained_bytes(&self) -> u64 {
        self.pending
            .as_ref()
            .map_or(0, |p| p.bytes.iter().map(|b| b.capacity() as u64).sum())
    }
}
