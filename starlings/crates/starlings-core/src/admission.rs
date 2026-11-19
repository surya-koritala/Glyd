//! Coordinator-owned reservation transaction. No backend calls or native qualification.
use crate::{
    CoreError, MAX_RECORDS,
    planner::{self, Candidate, PreviewInput},
};
use starlings_protocol::*;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum State {
    Preparing,
    Committing,
    Ready,
    Releasing,
}

/// Local owner handle, never a peer-provided grant. Restart requires a new incarnation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReservationId {
    incarnation: InvocationId,
    epoch: u64,
    serial: u64,
}

pub struct Reservation {
    pub input_digest: Digest,
    pub candidate: Candidate,
    pub state: State,
    pub deadline_ms: u64,
    limits: BTreeMap<(NodeId, String), u64>,
    inventories: BTreeMap<NodeId, (Digest, Digest)>,
    prepared: Vec<Option<BTreeMap<String, u64>>>,
    committed: Vec<bool>,
    released: Vec<bool>,
    uncertain_nodes: BTreeSet<NodeId>,
}

/// One owner serializes all transactions. This is not durable or a live registry fence.
pub struct AdmissionBook {
    cluster: ClusterId,
    incarnation: InvocationId,
    epoch: u64,
    mode: FixtureMode,
    now: u64,
    next: u64,
    accepting: bool,
    active: BTreeMap<u64, Reservation>,
}
impl AdmissionBook {
    pub fn new(
        cluster: ClusterId,
        incarnation: InvocationId,
        epoch: u64,
        mode: FixtureMode,
    ) -> Result<Self, CoreError> {
        if epoch == 0 {
            return Err(CoreError::StaleEpoch);
        }
        Ok(Self {
            cluster,
            incarnation,
            epoch,
            mode,
            now: 0,
            next: 1,
            accepting: true,
            active: BTreeMap::new(),
        })
    }

    /// Recompute the bounded preview; never accept a caller-edited Candidate as admission.
    /// The caller supplies locally approved profile digests and receiver-owned capture clocks.
    pub fn begin(
        &mut self,
        input: &PreviewInput,
        candidate_index: usize,
        lifetime_ms: u64,
    ) -> Result<ReservationId, CoreError> {
        if !self.accepting {
            return Err(CoreError::State);
        }
        self.advance(input.now_ms.0)?;
        if !(1..=60_000).contains(&lifetime_ms)
            || input
                .nodes
                .iter()
                .any(|n| n.capabilities.cluster_id != self.cluster)
        {
            return Err(CoreError::Invalid);
        }
        if self.active.len() >= MAX_RECORDS {
            return Err(CoreError::Capacity);
        }
        let report = planner::preview(input, self.mode)?;
        let candidate = report
            .candidates
            .get(candidate_index)
            .ok_or(CoreError::NotFound)?
            .clone();
        let deadline_ms = self.now.checked_add(lifetime_ms).ok_or(CoreError::Clock)?;
        let next = self.next.checked_add(1).ok_or(CoreError::Limit)?;
        let mut limits = BTreeMap::new();
        let mut inventories = BTreeMap::new();
        for stage in &candidate.stages {
            let node = input
                .nodes
                .iter()
                .find(|n| n.capabilities.node_id == stage.node)
                .ok_or(CoreError::Invalid)?;
            let identity = (
                node.expected_profile_key.device_digest.clone(),
                node.expected_profile_key.pool_inventory_digest.clone(),
            );
            for prior in self.active.values() {
                if prior.uncertain_nodes.contains(&stage.node) {
                    return Err(CoreError::State);
                }
                if prior
                    .inventories
                    .get(&stage.node)
                    .is_some_and(|old| old != &identity)
                {
                    return Err(CoreError::Conflict);
                }
            }
            inventories.insert(stage.node, identity);
            for pool in &stage.pools {
                let observed = node
                    .capabilities
                    .memory_pools
                    .iter()
                    .find(|p| p.pool_id == pool.pool_id)
                    .ok_or(CoreError::Invalid)?;
                let key = (stage.node, pool.pool_id.clone());
                let mut limit = observed
                    .available_bytes
                    .0
                    .checked_sub(observed.reserved_bytes.0)
                    .ok_or(CoreError::Capacity)?;
                let mut used = 0u64;
                for prior in self.active.values() {
                    for (index, held) in prior.candidate.stages.iter().enumerate() {
                        if !prior.released[index]
                            && held.node == stage.node
                            && let Some(bytes) =
                                held.pools.iter().find(|p| p.pool_id == pool.pool_id)
                        {
                            used = used
                                .checked_add(bytes.total_bytes.0)
                                .ok_or(CoreError::Capacity)?;
                            limit = limit.min(*prior.limits.get(&key).ok_or(CoreError::Invalid)?);
                        }
                    }
                }
                if used
                    .checked_add(pool.total_bytes.0)
                    .is_none_or(|n| n > limit)
                {
                    return Err(CoreError::Capacity);
                }
                limits.insert(key, limit);
            }
        }
        let stages = candidate.stages.len();
        let id = ReservationId {
            incarnation: self.incarnation,
            epoch: self.epoch,
            serial: self.next,
        };
        self.active.insert(
            self.next,
            Reservation {
                input_digest: report.input_digest,
                candidate,
                state: State::Preparing,
                deadline_ms,
                limits,
                inventories,
                prepared: vec![None; stages],
                committed: vec![false; stages],
                released: vec![false; stages],
                uncertain_nodes: BTreeSet::new(),
            },
        );
        self.next = next;
        Ok(id)
    }

    pub fn reservation(&self, id: ReservationId) -> Result<&Reservation, CoreError> {
        if id.incarnation != self.incarnation || id.epoch != self.epoch {
            return Err(CoreError::Conflict);
        }
        self.active.get(&id.serial).ok_or(CoreError::NotFound)
    }
    fn get_mut(&mut self, id: ReservationId) -> Result<&mut Reservation, CoreError> {
        if id.incarnation != self.incarnation || id.epoch != self.epoch {
            return Err(CoreError::Conflict);
        }
        self.active.get_mut(&id.serial).ok_or(CoreError::NotFound)
    }

    /// Trusted local adapter observation for this exact owner/stage, not a remote readiness ACK.
    /// Include every reserved physical pool, including zero allocations. Uncertain failures fence
    /// the node until all stages of this transaction have trusted cleanup observations.
    pub fn prepared(
        &mut self,
        id: ReservationId,
        stage: usize,
        actual: BTreeMap<String, u64>,
        now: u64,
    ) -> Result<(), CoreError> {
        self.advance(now)?;
        let r = self.get_mut(id)?;
        let expected = r.candidate.stages.get(stage).ok_or(CoreError::Invalid)?;
        let conflict = r.prepared[stage].as_ref().is_some_and(|old| old != &actual);
        if conflict
            || actual.len() != expected.pools.len()
            || expected
                .pools
                .iter()
                .any(|p| actual.get(&p.pool_id).is_none_or(|n| *n > p.total_bytes.0))
        {
            r.state = State::Releasing;
            r.released[stage] = false;
            r.uncertain_nodes.insert(expected.node);
            return Err(if conflict {
                CoreError::Conflict
            } else {
                CoreError::Capacity
            });
        }
        if r.state != State::Preparing {
            return Err(CoreError::State);
        }
        if r.prepared[stage].as_ref() == Some(&actual) {
            return Ok(());
        }
        r.prepared[stage] = Some(actual);
        Ok(())
    }

    pub fn start_commit(&mut self, id: ReservationId, now: u64) -> Result<(), CoreError> {
        self.advance(now)?;
        let r = self.get_mut(id)?;
        match r.state {
            State::Preparing if r.prepared.iter().all(Option::is_some) => {
                r.state = State::Committing
            }
            State::Committing | State::Ready => {}
            _ => return Err(CoreError::State),
        }
        Ok(())
    }
    pub fn committed(
        &mut self,
        id: ReservationId,
        stage: usize,
        now: u64,
    ) -> Result<(), CoreError> {
        self.advance(now)?;
        let r = self.get_mut(id)?;
        if !matches!(r.state, State::Committing | State::Ready) {
            return Err(CoreError::State);
        }
        *r.committed.get_mut(stage).ok_or(CoreError::Invalid)? = true;
        if r.committed.iter().all(|v| *v) {
            r.state = State::Ready;
        }
        Ok(())
    }
    pub fn abort(&mut self, id: ReservationId) -> Result<(), CoreError> {
        self.get_mut(id)?.state = State::Releasing;
        Ok(())
    }
    /// Permanently stop admission; keep existing reservation handles valid for cleanup.
    /// The caller must separately withdraw live dispatch/output authority; this performs no I/O.
    pub fn withdraw(&mut self) {
        self.accepting = false;
        for r in self.active.values_mut() {
            r.state = State::Releasing;
        }
    }
    /// Only the local execution owner may attest cleanup, including never-dispatched stages.
    /// Process/socket exit, timeout, transport receipt and coordinator ACK are not native drain.
    pub fn drained(&mut self, id: ReservationId, stage: usize) -> Result<(), CoreError> {
        let r = self.get_mut(id)?;
        if r.state != State::Releasing {
            return Err(CoreError::State);
        }
        *r.released.get_mut(stage).ok_or(CoreError::Invalid)? = true;
        if r.released.iter().all(|v| *v) {
            self.active.remove(&id.serial);
        }
        Ok(())
    }
    pub fn advance(&mut self, now: u64) -> Result<(), CoreError> {
        if now < self.now {
            return Err(CoreError::Clock);
        }
        self.now = now;
        for r in self.active.values_mut() {
            if now >= r.deadline_ms {
                r.state = State::Releasing;
            }
        }
        Ok(())
    }
}
