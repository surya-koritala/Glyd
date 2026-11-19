//! Explicitly enabled simulation over bounded local pipes. Never binds a network port.
use serde::{Deserialize, Serialize};
use starlings_core::{CoreError, Snapshot, Worker};
use starlings_protocol::*;
use std::{
    collections::BTreeMap,
    io::{self, BufRead, Read, Write},
};
pub const CAPACITY: u64 = 4_194_304;
pub fn id<T: std::str::FromStr>(n: u64) -> T
where
    T::Err: std::fmt::Debug,
{
    format!("00000000-0000-4000-8000-{n:012x}")
        .parse()
        .expect("fixed fixture ID")
}
pub fn plan() -> PlacementPlan {
    let mut plan: PlacementPlan = decode(
        include_bytes!("../../../examples/placement-plan.json"),
        FixtureMode::Simulation,
    )
    .expect("plan fixture");
    plan.plan_digest =
        canonical_digest(&serde_json::to_value(&plan).unwrap(), "plan_digest").unwrap();
    plan
}
pub fn lease(node: u64) -> Lease {
    let mut l: Lease = decode(
        include_bytes!("../../../examples/lease.json"),
        FixtureMode::Simulation,
    )
    .expect("lease fixture");
    l.lease_id = id(0x103 + node);
    l.owner_node_id = id(node);
    l.plan_digest = plan().plan_digest;
    l
}
pub fn session(node: u64) -> Session {
    let mut s: Session = decode(
        include_bytes!("../../../examples/session.json"),
        FixtureMode::Simulation,
    )
    .expect("session fixture");
    s.lease_id = lease(node).lease_id;
    s.plan_digest = plan().plan_digest;
    s
}
pub fn header(node: u64, stage: u32) -> TensorHeader {
    let mut h: TensorHeader = decode(
        include_bytes!("../../../examples/tensor-header.json"),
        FixtureMode::Simulation,
    )
    .expect("tensor fixture");
    h.lease_id = lease(node).lease_id;
    h.plan_digest = plan().plan_digest;
    h.destination_stage = BoundedU32::new(stage).expect("fixture stage");
    h.source_stage = BoundedU32::new(stage.saturating_sub(1)).expect("fixture stage");
    h.shape = vec![
        BoundedU32::new(1).unwrap(),
        BoundedU32::new(2).unwrap(),
        BoundedU32::new(2).unwrap(),
    ];
    h.total_tensor_bytes = Decimal(8);
    h.chunk_bytes = Decimal(8);
    h
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "command", rename_all = "snake_case", deny_unknown_fields)]
pub enum Command {
    Apply {
        request: Box<ControlRequest>,
        now: u64,
        payload: Vec<u8>,
        delay_ms: u32,
    },
    Inspect {
        now: u64,
    },
    Disconnect,
    Crash,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Reply {
    pub request_id: Option<RequestId>,
    pub error: Option<CoreError>,
    pub duplicate: bool,
    pub snapshot: Snapshot,
    pub output: Vec<u8>,
    pub executions: u64,
}
/// Reject before growing the framing buffer beyond the control-message limit.
pub fn read_frame(reader: &mut impl BufRead) -> io::Result<Option<Vec<u8>>> {
    let mut bytes = Vec::new();
    let count = reader
        .take((MAX_JSON_BYTES + 1) as u64)
        .read_until(b'\n', &mut bytes)?;
    if count == 0 {
        return Ok(None);
    }
    if count > MAX_JSON_BYTES || bytes.last() != Some(&b'\n') {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "invalid simulation frame",
        ));
    }
    bytes.pop();
    Ok(Some(bytes))
}
pub fn write_frame(writer: &mut impl Write, value: &impl Serialize) -> io::Result<()> {
    let bytes = serde_json::to_vec(value).map_err(io::Error::other)?;
    if bytes.len() >= MAX_JSON_BYTES {
        return Err(io::Error::other("simulation frame too large"));
    }
    writer.write_all(&bytes)?;
    writer.write_all(b"\n")?;
    writer.flush()
}
struct Pending {
    ready_at: u64,
    output: Vec<u8>,
}
struct SimulationWorker {
    worker: Worker,
    node: u64,
    pending: BTreeMap<(GenerationId, InvocationId), Pending>,
    outputs: BTreeMap<RequestId, (GenerationId, Vec<u8>)>,
    executions: u64,
}
impl SimulationWorker {
    fn discard_cancelled_outputs(&mut self) {
        self.outputs.retain(|id, (generation, _)| {
            self.worker.retains_request(*id) && self.worker.can_publish(*generation)
        });
    }
    fn apply(
        &mut self,
        request: &ControlRequest,
        now: u64,
        payload: Vec<u8>,
        delay_ms: u32,
    ) -> Result<(bool, Vec<u8>), CoreError> {
        match &request.operation {
            Operation::StartStep {
                header,
                input_digest,
            } => {
                if payload.len() > 32768
                    || payload.len() as u64 != header.total_tensor_bytes.0
                    || byte_digest(&payload) != *input_digest
                    || delay_ms > 1000
                    || now.checked_add(delay_ms.into()).is_none()
                {
                    return Err(CoreError::Invalid);
                }
            }
            _ if !payload.is_empty() || delay_ms != 0 => return Err(CoreError::Invalid),
            _ => {}
        }
        if let Operation::CompleteStep {
            generation_id,
            invocation_id,
        } = &request.operation
            && self
                .pending
                .get(&(*generation_id, *invocation_id))
                .is_some_and(|p| now < p.ready_at)
        {
            return Err(CoreError::State);
        }
        let replay = self.worker.apply(request, now)?;
        self.discard_cancelled_outputs();
        if replay {
            if matches!(request.operation, Operation::CompleteStep { .. })
                && !self.outputs.contains_key(&request.request_id)
            {
                return Err(CoreError::OutputWithdrawn);
            }
            return Ok((
                true,
                self.outputs
                    .get(&request.request_id)
                    .map(|(_, bytes)| bytes.clone())
                    .unwrap_or_default(),
            ));
        }
        let mut output = Vec::new();
        match &request.operation {
            Operation::StartRequest {
                session,
                invocation_id,
                ..
            } => {
                self.executions += 1;
                self.pending.insert(
                    (session.generation_id, *invocation_id),
                    Pending {
                        ready_at: now,
                        output: format!("synthetic worker {}", self.node).into_bytes(),
                    },
                );
            }
            Operation::StartStep { header, .. } => {
                self.executions += 1;
                self.pending.insert(
                    (header.generation_id, header.invocation_id),
                    Pending {
                        ready_at: now + u64::from(delay_ms),
                        output: payload
                            .into_iter()
                            .map(|b| {
                                if self.node == 1 {
                                    b.wrapping_add(1)
                                } else {
                                    b.rotate_left(1)
                                }
                            })
                            .collect(),
                    },
                );
            }
            Operation::CompleteStep {
                generation_id,
                invocation_id,
            } => {
                if let Some(p) = self.pending.remove(&(*generation_id, *invocation_id))
                    && self.worker.can_publish(*generation_id)
                {
                    output = p.output;
                    self.outputs
                        .insert(request.request_id, (*generation_id, output.clone()));
                }
            }
            _ => {}
        }
        Ok((false, output))
    }
}
pub fn worker_loop(node: u64, capacity: u64) -> io::Result<()> {
    if !(1..=2).contains(&node) {
        return Err(io::Error::other("invalid fixture worker"));
    }
    let worker = Worker::new(
        id(node),
        plan().cluster_id,
        capacity,
        1,
        FixtureMode::Simulation,
    )
    .map_err(|_| io::Error::other("invalid worker limits"))?;
    let mut simulation = SimulationWorker {
        worker,
        node,
        pending: BTreeMap::new(),
        outputs: BTreeMap::new(),
        executions: 0,
    };
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut input = stdin.lock();
    let mut output = stdout.lock();
    while let Some(bytes) = read_frame(&mut input)? {
        let value = parse_json(&bytes).map_err(|_| io::Error::other("invalid simulation JSON"))?;
        let cmd: Command = serde_json::from_value(value)
            .map_err(|_| io::Error::other("invalid simulation command"))?;
        let request_id = match &cmd {
            Command::Apply { request, .. } => Some(request.request_id),
            _ => None,
        };
        let result = match cmd {
            Command::Disconnect => break,
            Command::Crash => std::process::exit(17),
            Command::Inspect { now } => {
                simulation.worker.advance(now).map(|()| (false, Vec::new()))
            }
            Command::Apply {
                request,
                now,
                payload,
                delay_ms,
            } => simulation.apply(&request, now, payload, delay_ms),
        };
        simulation.discard_cancelled_outputs();
        let (error, duplicate, bytes) = match result {
            Ok((duplicate, bytes)) => (None, duplicate, bytes),
            Err(error) => (Some(error), false, Vec::new()),
        };
        write_frame(
            &mut output,
            &Reply {
                request_id,
                error,
                duplicate,
                snapshot: simulation.worker.snapshot(),
                output: bytes,
                executions: simulation.executions,
            },
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn pipe_frames_are_bounded_and_reject_partial_messages() {
        assert!(read_frame(&mut io::Cursor::new(b"{}".as_slice())).is_err());
        assert!(read_frame(&mut io::Cursor::new(vec![b'x'; MAX_JSON_BYTES + 1])).is_err());
        assert_eq!(
            read_frame(&mut io::Cursor::new(b"{}\n".as_slice())).unwrap(),
            Some(b"{}".to_vec())
        );
        assert!(
            read_frame(&mut io::Cursor::new(Vec::<u8>::new()))
                .unwrap()
                .is_none()
        );
        let mut framed = Vec::new();
        write_frame(&mut framed, &Command::Inspect { now: 5 }).unwrap();
        let value =
            parse_json(&read_frame(&mut io::Cursor::new(framed)).unwrap().unwrap()).unwrap();
        assert!(matches!(
            serde_json::from_value::<Command>(value).unwrap(),
            Command::Inspect { now: 5 }
        ));
    }
}
