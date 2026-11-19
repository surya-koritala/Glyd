//! Invocation-scoped credit and reassembly. No backend dispatch or drain authority.
use super::*;
mod cursor;
use cursor::ReceiveCursor;
use starlings_protocol::{
    DataControl, DataScope, Decimal, GenerationId, InvocationId, LeaseId, SessionId,
};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::{Arc, Mutex},
};

type Key = (NodeId, SessionId, GenerationId, InvocationId);
struct BudgetState {
    total: u64,
    peer: u64,
    session: u64,
    held: u64,
    peers: BTreeMap<NodeId, u64>,
    sessions: BTreeMap<SessionId, u64>,
    live: BTreeSet<Key>,
    leases: BTreeMap<LeaseId, u64>,
}
/// Share one budget across all receiving connections in the local allocation pool.
#[derive(Clone)]
pub struct ReceiveBudget(Arc<Mutex<BudgetState>>);
impl ReceiveBudget {
    pub fn new(total: u64, per_peer: u64, per_session: u64) -> Result<Self> {
        if per_session == 0 || per_session > per_peer || per_peer > total {
            return Err(Error::Invalid);
        }
        Ok(Self(Arc::new(Mutex::new(BudgetState {
            total,
            peer: per_peer,
            session: per_session,
            held: 0,
            peers: BTreeMap::new(),
            sessions: BTreeMap::new(),
            live: BTreeSet::new(),
            leases: BTreeMap::new(),
        }))))
    }
    pub fn held_bytes(&self) -> Result<u64> {
        Ok(self.0.lock().map_err(|_| Error::Denied)?.held)
    }
    fn reserve(
        &self,
        peer: NodeId,
        scope: &DataScope,
        bytes: u64,
        lease_budget: u64,
    ) -> Result<Reservation> {
        let mut s = self.0.lock().map_err(|_| Error::Denied)?;
        let key = (
            peer,
            scope.session_id,
            scope.generation_id,
            scope.invocation_id,
        );
        if s.live.contains(&key) {
            return Err(Error::Replay);
        }
        let total = s.held.checked_add(bytes).ok_or(Error::Capacity)?;
        let peer_bytes = s
            .peers
            .get(&peer)
            .copied()
            .unwrap_or(0)
            .checked_add(bytes)
            .ok_or(Error::Capacity)?;
        let session_bytes = s
            .sessions
            .get(&scope.session_id)
            .copied()
            .unwrap_or(0)
            .checked_add(bytes)
            .ok_or(Error::Capacity)?;
        let lease_bytes = s
            .leases
            .get(&scope.lease_id)
            .copied()
            .unwrap_or(0)
            .checked_add(bytes)
            .ok_or(Error::Capacity)?;
        if bytes == 0
            || s.live.len() >= 128
            || total > s.total
            || peer_bytes > s.peer
            || session_bytes > s.session
            || lease_bytes > lease_budget
        {
            return Err(Error::Capacity);
        }
        s.held = total;
        s.peers.insert(peer, peer_bytes);
        s.sessions.insert(scope.session_id, session_bytes);
        s.live.insert(key);
        s.leases.insert(scope.lease_id, lease_bytes);
        Ok(Reservation {
            budget: self.clone(),
            key,
            lease: scope.lease_id,
            bytes,
        })
    }
}
struct Reservation {
    budget: ReceiveBudget,
    key: Key,
    lease: LeaseId,
    bytes: u64,
}
impl Drop for Reservation {
    fn drop(&mut self) {
        let mut s = self.budget.0.lock().unwrap_or_else(|e| e.into_inner());
        s.held -= self.bytes;
        s.live.remove(&self.key);
        let lease = s.leases.get_mut(&self.lease).unwrap();
        *lease -= self.bytes;
        if *lease == 0 {
            s.leases.remove(&self.lease);
        }
        let peer = s.peers.get_mut(&self.key.0).unwrap();
        *peer -= self.bytes;
        if *peer == 0 {
            s.peers.remove(&self.key.0);
        }
        let key = self.key.1;
        let session = s.sessions.get_mut(&key).unwrap();
        *session -= self.bytes;
        if *session == 0 {
            s.sessions.remove(&key);
        }
    }
}
/// Owned, complete ordered bytes. Moving this value retains its allocation reservation.
/// Neither receipt nor dropping a peer result proves native work has drained.
pub struct ReceivedTensors {
    scope: DataScope,
    headers: Vec<starlings_protocol::TensorHeader>,
    tensors: Vec<Box<[u8]>>,
    _reservation: Option<Reservation>,
}
impl ReceivedTensors {
    /// Trusted local first-stage input (prompt ids); no peer, credit or receive budget.
    pub(crate) fn local(header: starlings_protocol::TensorHeader, bytes: Box<[u8]>) -> Self {
        Self {
            scope: DataScope::from(&header),
            headers: vec![header],
            tensors: vec![bytes],
            _reservation: None,
        }
    }
    pub fn scope(&self) -> &DataScope {
        &self.scope
    }
    pub fn tensors(&self) -> &[Box<[u8]>] {
        &self.tensors
    }
    /// Original admitted first-frame templates; completion does not enlarge wire chunks.
    pub fn headers(&self) -> &[starlings_protocol::TensorHeader] {
        &self.headers
    }
}
struct Contract {
    scope: DataScope,
    total: u64,
    deadline: Instant,
    records: usize,
    header_bytes: usize,
    control: usize,
}
impl Contract {
    fn new(
        stream: &mut DataStream,
        admissions: &[DataAdmission<'_>],
        deadline: Instant,
        receiving: bool,
    ) -> Result<Self> {
        if admissions.is_empty() || admissions.len() > 8 {
            return Err(Error::Invalid);
        }
        let deadline = deadline.min(Instant::now() + transport::TIMEOUT);
        // Refuse a poisoned stream before reserving or allocating any payloads.
        stream.peer.admitted_frame(&admissions[0], deadline, |io| {
            admissions[0].check(io, None, receiving, || ())
        })?;
        let first = admissions[0].expected();
        let scope = DataScope::from(first);
        let mut total = 0u64;
        let mut control = 0;
        for (i, admission) in admissions.iter().enumerate() {
            stream.bound(admission)?;
            let h = admission.expected();
            if receiving {
                admission.check_open(&stream.peer)?;
            } else {
                admission.check(
                    &stream.peer,
                    Some(&DataHeader::Tensor(h.clone())),
                    receiving,
                    || (),
                )?;
            }
            if DataScope::from(h) != scope
                || h.tensor_index.get() as usize != i
                || h.tensor_count.get() as usize != admissions.len()
                || h.chunk_offset_bytes.0 != 0
                || h.phase != first.phase
                || h.position_start != first.position_start
                || h.token_count != first.token_count
                || h.boundary_profile != first.boundary_profile
            {
                return Err(Error::Invalid);
            }
            total = total
                .checked_add(h.total_tensor_bytes.0)
                .ok_or(Error::Capacity)?;
            if h.total_tensor_bytes.0 > admissions[control].expected().total_tensor_bytes.0 {
                control = i;
            }
        }
        if total > admissions[0].scratch_budget()
            || admissions
                .iter()
                .any(|a| a.scratch_budget() != admissions[0].scratch_budget())
        {
            return Err(Error::Capacity);
        }
        Ok(Self {
            scope,
            total,
            deadline,
            records: 256,
            header_bytes: 1024 * 1024,
            control,
        })
    }
    fn record(&mut self, bytes: usize) -> Result<()> {
        self.records = self.records.checked_sub(1).ok_or(Error::Capacity)?;
        self.header_bytes = self
            .header_bytes
            .checked_sub(bytes)
            .ok_or(Error::Capacity)?;
        if Instant::now() >= self.deadline {
            return Err(Error::Expired);
        }
        Ok(())
    }
}
fn poison(stream: &mut DataStream, admission: &DataAdmission<'_>) {
    let _: Result<()> = stream
        .peer
        .admitted_frame(admission, Instant::now(), |_| Err(Error::Transport));
}
/// Borrows the persistent stream exclusively until completion or failure.
pub struct FlowReceiver<'s, 'o> {
    stream: &'s mut DataStream,
    admissions: &'s [DataAdmission<'o>],
    contract: Contract,
    buffers: Vec<Box<[u8]>>,
    reservation: Option<Reservation>,
    cursor: ReceiveCursor,
}
impl DataStream {
    pub fn begin_receive<'s, 'o>(
        &'s mut self,
        admissions: &'s [DataAdmission<'o>],
        budget: &ReceiveBudget,
        deadline: Instant,
    ) -> Result<FlowReceiver<'s, 'o>> {
        let contract = Contract::new(self, admissions, deadline, true)?;
        let reservation = budget.reserve(
            self.peer.peer(),
            &contract.scope,
            contract.total,
            admissions[0].scratch_budget(),
        )?;
        let mut buffers = Vec::new();
        for admission in admissions {
            let h = admission.expected();
            let len = usize::try_from(h.total_tensor_bytes.0).map_err(|_| Error::Capacity)?;
            let mut bytes = Vec::new();
            bytes.try_reserve_exact(len).map_err(|_| Error::Capacity)?;
            bytes.resize(len, 0);
            buffers.push(bytes.into_boxed_slice());
        }
        Ok(FlowReceiver {
            stream: self,
            admissions,
            contract,
            buffers,
            reservation: Some(reservation),
            cursor: ReceiveCursor::default(),
        })
    }
    pub fn begin_send<'s, 'o>(
        &'s mut self,
        admissions: &'s [DataAdmission<'o>],
        deadline: Instant,
    ) -> Result<FlowSender<'s, 'o>> {
        let contract = Contract::new(self, admissions, deadline, false)?;
        let len = admissions.len();
        Ok(FlowSender {
            stream: self,
            admissions,
            contract,
            sent: vec![0; len],
            receipts: vec![0; len],
            tensor: 0,
            credit: 0,
            granted: 0,
            finished: false,
        })
    }
}
impl FlowReceiver<'_, '_> {
    fn abort(&mut self) {
        poison(self.stream, &self.admissions[0]);
        let _ = self.admissions[0].abort();
        self.buffers.clear();
        self.reservation = None;
    }
    pub fn grant_credit(&mut self, bytes: u64) -> Result<()> {
        let result = (|| {
            if self.reservation.is_none() {
                return Err(Error::Capacity);
            }
            let next = self
                .cursor
                .grant(bytes, self.contract.total)
                .ok_or(Error::Capacity)?;
            let h = DataControl::Credit {
                protocol_version: Version,
                scope: self.contract.scope.clone(),
                bytes: Decimal(bytes),
            };
            self.contract.record(serde_json::to_vec(&h)?.len())?;
            self.stream.send_until(
                &self.admissions[self.contract.control],
                &DataHeader::Control(h),
                &[],
                self.contract.deadline,
            )?;
            self.cursor = next;
            Ok(())
        })();
        if result.is_err() {
            self.abort();
        }
        result
    }
    /// Reads directly into reserved storage; no additional payload Vec or partial delivery.
    pub fn receive_chunk(&mut self) -> Result<()> {
        let result = (|| {
            if self.reservation.is_none() || self.cursor.tensor >= self.buffers.len() {
                return Err(Error::Denied);
            }
            let admission = &self.admissions[self.cursor.tensor];
            let deadline = self.contract.deadline;
            let frame = self.stream.peer.admitted_frame(admission, deadline, |io| {
                admission.check(io, None, true, || ())?;
                if self.contract.records == 0 {
                    return Err(Error::Capacity);
                }
                let (header, bytes) = DataStream::read_header(io, self.contract.header_bytes)?;
                self.contract.record(bytes)?;
                admission.check(io, Some(&header), true, || ())?;
                let DataHeader::Tensor(h) = &header else {
                    return Err(Error::Denied);
                };
                let (range, next) = self
                    .cursor
                    .chunk(
                        h.chunk_offset_bytes.0,
                        h.chunk_bytes.0,
                        self.buffers[self.cursor.tensor].len(),
                        h.end_of_tensor,
                    )
                    .ok_or(Error::Capacity)?;
                let slice = self.buffers[self.cursor.tensor]
                    .get_mut(range)
                    .ok_or(Error::Capacity)?;
                io.read_exact(slice)?;
                admission.check(io, Some(&header), true, || {
                    self.cursor = next;
                })?;
                Ok((h.tensor_index, h.end_of_tensor))
            })?;
            let receipt = DataControl::ReceiptAck {
                protocol_version: Version,
                scope: self.contract.scope.clone(),
                tensor_index: frame.0,
                received_through_bytes: Decimal(self.cursor.offset),
            };
            self.contract.record(serde_json::to_vec(&receipt)?.len())?;
            self.stream
                .send_until(admission, &DataHeader::Control(receipt), &[], deadline)?;
            self.cursor.acknowledge(frame.1);
            Ok(())
        })();
        if result.is_err() {
            self.abort();
        }
        result
    }
    pub fn finish(mut self) -> Result<ReceivedTensors> {
        if self.reservation.is_none() || !self.cursor.complete(self.buffers.len()) {
            return Err(Error::Denied);
        }
        let header = DataHeader::Tensor(
            self.admissions
                .last()
                .ok_or(Error::Denied)?
                .expected()
                .clone(),
        );
        let result =
            self.stream
                .peer
                .admitted_frame(&self.admissions[0], self.contract.deadline, |io| {
                    self.admissions.last().ok_or(Error::Denied)?.check(
                        io,
                        Some(&header),
                        true,
                        || {
                            Ok(ReceivedTensors {
                                scope: self.contract.scope.clone(),
                                headers: self
                                    .admissions
                                    .iter()
                                    .map(|a| a.expected().clone())
                                    .collect(),
                                tensors: std::mem::take(&mut self.buffers),
                                _reservation: Some(self.reservation.take().ok_or(Error::Denied)?),
                            })
                        },
                    )?
                });
        if result.is_err() {
            self.abort();
        }
        result
    }
}
impl Drop for FlowReceiver<'_, '_> {
    fn drop(&mut self) {
        if self.reservation.is_some() {
            self.abort();
        }
    }
}
/// Credit/receipt tracking only. Caller retains native output ownership separately.
pub struct FlowSender<'s, 'o> {
    stream: &'s mut DataStream,
    admissions: &'s [DataAdmission<'o>],
    contract: Contract,
    sent: Vec<u64>,
    receipts: Vec<u64>,
    tensor: usize,
    credit: u64,
    granted: u64,
    finished: bool,
}
impl FlowSender<'_, '_> {
    fn abort(&mut self) {
        self.finished = true;
        poison(self.stream, &self.admissions[0]);
        let _ = self.admissions[0].abort();
    }
    pub fn receive_control(&mut self) -> Result<DataControl> {
        let result =
            self.stream
                .peer
                .admitted_frame(&self.admissions[0], self.contract.deadline, |io| {
                    if self.finished || self.contract.records == 0 {
                        return Err(Error::Denied);
                    }
                    self.admissions[0].check(io, None, true, || ())?;
                    let (header, bytes) = DataStream::read_header(io, self.contract.header_bytes)?;
                    self.contract.record(bytes)?;
                    let DataHeader::Control(control) = &header else {
                        return Err(Error::Denied);
                    };
                    let index = match control {
                        DataControl::Credit { .. } => self.contract.control,
                        DataControl::ReceiptAck { tensor_index, .. } => tensor_index.get() as usize,
                        _ => return Err(Error::Denied),
                    };
                    let admission = self.admissions.get(index).ok_or(Error::Denied)?;
                    admission.check(io, Some(&header), true, || ())?;
                    match control {
                        DataControl::Credit { bytes, .. } => {
                            self.granted = self
                                .granted
                                .checked_add(bytes.0)
                                .filter(|n| *n <= self.contract.total)
                                .ok_or(Error::Capacity)?;
                            self.credit =
                                self.credit.checked_add(bytes.0).ok_or(Error::Capacity)?;
                        }
                        DataControl::ReceiptAck {
                            received_through_bytes: bytes,
                            ..
                        } => {
                            if bytes.0 <= self.receipts[index] || bytes.0 > self.sent[index] {
                                return Err(Error::Replay);
                            }
                            self.receipts[index] = bytes.0;
                        }
                        _ => unreachable!(),
                    }
                    Ok(control.clone())
                });
        if result.is_err() {
            self.abort();
        }
        result
    }
    pub fn send_chunk(&mut self, bytes: &[u8]) -> Result<()> {
        let result = (|| {
            if self.finished {
                return Err(Error::Denied);
            }
            let mut h = self
                .admissions
                .get(self.tensor)
                .ok_or(Error::Denied)?
                .expected()
                .clone();
            if bytes.is_empty() || bytes.len() as u64 > self.credit {
                return Err(Error::Capacity);
            }
            h.chunk_offset_bytes = Decimal(self.sent[self.tensor]);
            h.chunk_bytes = Decimal(bytes.len() as u64);
            let end = h
                .chunk_offset_bytes
                .0
                .checked_add(h.chunk_bytes.0)
                .ok_or(Error::Capacity)?;
            h.end_of_tensor = end == h.total_tensor_bytes.0;
            let ended = h.end_of_tensor;
            self.contract.record(serde_json::to_vec(&h)?.len())?;
            let header = DataHeader::Tensor(h);
            self.stream.send_until(
                &self.admissions[self.tensor],
                &header,
                bytes,
                self.contract.deadline,
            )?;
            self.credit -= bytes.len() as u64;
            self.sent[self.tensor] = end;
            if ended {
                self.tensor += 1;
            }
            Ok(())
        })();
        if result.is_err() {
            self.abort();
        }
        result
    }
    pub fn finish(mut self) -> Result<()> {
        if self.finished
            || self.tensor != self.sent.len()
            || self.credit != 0
            || self.receipts != self.sent
        {
            return Err(Error::Denied);
        }
        self.stream
            .peer
            .admitted_frame(&self.admissions[0], self.contract.deadline, |io| {
                self.admissions[0].check(io, None, false, || ())
            })?;
        self.finished = true;
        Ok(())
    }
}
impl Drop for FlowSender<'_, '_> {
    fn drop(&mut self) {
        if !self.finished {
            self.abort();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use starlings_protocol::TensorHeader;
    fn scope() -> DataScope {
        let h: TensorHeader = starlings_protocol::decode(
            include_bytes!("../../../../examples/tensor-header.json"),
            FixtureMode::Simulation,
        )
        .unwrap();
        (&h).into()
    }
    fn id<T: std::str::FromStr>() -> T
    where
        T::Err: std::fmt::Debug,
    {
        uuid::Uuid::new_v4().to_string().parse().unwrap()
    }
    #[test]
    fn shared_allocation_limits_count_live_handoffs_and_release_exactly_once() {
        let budget = ReceiveBudget::new(80, 60, 40).unwrap();
        let peer = id();
        let a = scope();
        let first = budget.reserve(peer, &a, 40, 80).unwrap();
        assert!(matches!(
            budget.reserve(peer, &a, 1, 80),
            Err(Error::Replay)
        ));
        let mut b = a.clone();
        b.invocation_id = id();
        assert!(
            matches!(budget.reserve(peer, &b, 1, 80), Err(Error::Capacity)),
            "same session cannot overcommit"
        );
        b.session_id = id();
        let second = budget.reserve(peer, &b, 20, 80).unwrap();
        b.invocation_id = id();
        assert!(
            matches!(budget.reserve(peer, &b, 1, 80), Err(Error::Capacity)),
            "peer limit counts both sessions"
        );
        let third = budget.reserve(id(), &b, 20, 80).unwrap();
        assert!(
            matches!(budget.reserve(id(), &b, 1, 80), Err(Error::Capacity)),
            "global limit counts all peers"
        );
        assert_eq!(budget.held_bytes().unwrap(), 80);
        drop((first, second, third));
        assert_eq!(budget.held_bytes().unwrap(), 0);
        let s = budget.0.lock().unwrap();
        assert!(
            s.live.is_empty() && s.peers.is_empty() && s.sessions.is_empty() && s.leases.is_empty()
        );
    }
    #[test]
    fn allocation_limits_apply_to_one_session_across_peers_and_one_lease_across_sessions() {
        let budget = ReceiveBudget::new(80, 80, 40).unwrap();
        let a = scope();
        let first = budget.reserve(id(), &a, 40, 40).unwrap();
        let mut other = a.clone();
        other.invocation_id = id();
        other.lease_id = id();
        assert!(
            matches!(budget.reserve(id(), &other, 1, 40), Err(Error::Capacity)),
            "a second peer cannot evade the session limit"
        );
        other.session_id = id();
        other.lease_id = a.lease_id;
        assert!(
            matches!(budget.reserve(id(), &other, 1, 40), Err(Error::Capacity)),
            "a second session cannot evade the actual lease limit"
        );
        drop(first);
        assert_eq!(budget.held_bytes().unwrap(), 0);
    }
    #[test]
    fn metadata_records_bytes_deadlines_and_active_identity_tables_are_bounded() {
        let mut c = Contract {
            scope: scope(),
            total: 1,
            deadline: Instant::now() + transport::TIMEOUT,
            records: 2,
            header_bytes: 5,
            control: 0,
        };
        c.record(3).unwrap();
        assert_eq!(c.record(3), Err(Error::Capacity));
        c.records = 0;
        assert_eq!(c.record(0), Err(Error::Capacity));
        c.records = 1;
        c.deadline = Instant::now();
        assert_eq!(c.record(0), Err(Error::Expired));
        let budget = ReceiveBudget::new(1024, 1024, 1024).unwrap();
        let peer = id();
        let mut scope = scope();
        let mut held = Vec::new();
        for _ in 0..128 {
            scope.invocation_id = id();
            held.push(budget.reserve(peer, &scope, 1, 1024).unwrap());
        }
        scope.invocation_id = id();
        assert!(matches!(
            budget.reserve(peer, &scope, 1, 1024),
            Err(Error::Capacity)
        ));
        drop(held);
        assert_eq!(budget.held_bytes().unwrap(), 0);
    }
}
