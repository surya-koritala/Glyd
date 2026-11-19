use starlings_adapters::fake::{self, Command as WireCommand, Reply};
use starlings_core::{CoreError, Registry};
use starlings_protocol::*;
use std::{
    collections::BTreeMap,
    io::{self, BufReader},
    process::{Child, ChildStdin, Command, Stdio},
    sync::mpsc::{self, Receiver},
    thread::{self, JoinHandle},
    time::Duration,
};
fn failure(message: &str) -> io::Error {
    io::Error::other(message)
}
fn require(ok: bool, message: &str) -> io::Result<()> {
    if ok { Ok(()) } else { Err(failure(message)) }
}
struct Peer {
    child: Child,
    input: Option<ChildStdin>,
    output: Option<Receiver<io::Result<Reply>>>,
    reader: Option<JoinHandle<()>>,
    plan_digest: Digest,
    sequences: BTreeMap<u64, u64>,
}
impl Peer {
    fn spawn(node: u64, capacity: u64) -> io::Result<Self> {
        let mut child = Command::new(std::env::current_exe()?)
            .args(["fake-worker", &node.to_string(), &capacity.to_string()])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()?;
        let input = child.stdin.take();
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| failure("worker stdout unavailable"))?;
        let (tx, rx) = mpsc::sync_channel(1);
        let reader = thread::spawn(move || {
            let mut stream = BufReader::new(stdout);
            loop {
                let result = match fake::read_frame(&mut stream) {
                    Ok(Some(bytes)) => parse_json(&bytes)
                        .map_err(|_| failure("invalid worker reply"))
                        .and_then(|v| {
                            serde_json::from_value(v).map_err(|_| failure("invalid worker reply"))
                        }),
                    Ok(None) => Err(failure("worker disconnected")),
                    Err(e) => Err(e),
                };
                let failed = result.is_err();
                if tx.send(result).is_err() || failed {
                    break;
                }
            }
        });
        Ok(Self {
            child,
            input,
            output: Some(rx),
            reader: Some(reader),
            plan_digest: fake::plan().plan_digest,
            sequences: BTreeMap::new(),
        })
    }
    fn send(&mut self, command: WireCommand) -> io::Result<Reply> {
        self.send_with_timeout(command, Duration::from_secs(10))
    }
    fn send_with_timeout(&mut self, command: WireCommand, timeout: Duration) -> io::Result<Reply> {
        let expected = match &command {
            WireCommand::Apply { request, .. } => Some(request.request_id),
            _ => None,
        };
        let mut input = self
            .input
            .take()
            .ok_or_else(|| failure("worker disconnected"))?;
        let (tx, rx) = mpsc::sync_channel(1);
        let writer = thread::spawn(move || {
            let result = fake::write_frame(&mut input, &command);
            let _ = tx.send((input, result));
        });
        let written = rx.recv_timeout(timeout);
        if written.is_err() {
            let _ = self.child.kill();
        }
        writer.join().map_err(|_| failure("worker writer failed"))?;
        let (input, result) = written.map_err(|_| failure("worker write deadline exceeded"))?;
        self.input = Some(input);
        result?;
        let reply = self
            .output
            .as_ref()
            .ok_or_else(|| failure("worker disconnected"))?
            .recv_timeout(timeout)
            .map_err(|_| {
                let _ = self.child.kill();
                failure("worker response deadline exceeded")
            })??;
        require(
            reply.request_id == expected,
            "worker reply request mismatch",
        )?;
        Ok(reply)
    }

    fn apply(
        &mut self,
        n: u64,
        mut op: Operation,
        now: u64,
        payload: Vec<u8>,
        delay_ms: u32,
    ) -> io::Result<Reply> {
        match &mut op {
            Operation::Prepare { plan, lease } => {
                self.plan_digest = plan.plan_digest.clone();
                lease.plan_digest = self.plan_digest.clone();
            }
            Operation::Begin { session } | Operation::StartRequest { session, .. } => {
                session.plan_digest = self.plan_digest.clone();
            }
            _ => (),
        }
        let next = self.sequences.len() as u64 + 1;
        let sequence = *self.sequences.entry(n).or_insert(next);
        let request = ControlRequest::new(fake::id(n), 1, sequence, op)
            .map_err(|_| failure("invalid fixture request"))?;
        self.send(WireCommand::Apply {
            request: Box::new(request),
            now,
            payload,
            delay_ms,
        })
    }
    fn ok(&mut self, n: u64, op: Operation, now: u64) -> io::Result<Reply> {
        let r = self.apply(n, op, now, vec![], 0)?;
        require(
            r.error.is_none(),
            &format!("control operation {n} failed: {:?}", r.error),
        )?;
        Ok(r)
    }
}
impl Drop for Peer {
    fn drop(&mut self) {
        self.output.take();
        self.input.take();
        let _ = self.child.kill();
        let _ = self.child.wait();
        if let Some(reader) = self.reader.take() {
            let _ = reader.join();
        }
    }
}
fn prepare(node: u64, whole: bool) -> Operation {
    let mut plan = fake::plan();
    if whole {
        plan.mode = PlacementPlanMode::WholeRequest;
        plan.stages.truncate(1);
        plan.stages[0].node_id = fake::id(node);
        plan.stages[0].layer_end_exclusive = BoundedU32::new(32).unwrap();
        plan.stages[0].owns_final_norm_and_head = true;
    }
    plan.plan_digest =
        canonical_digest(&serde_json::to_value(&plan).unwrap(), "plan_digest").unwrap();
    Operation::Prepare {
        plan: Box::new(plan),
        lease: Box::new(fake::lease(node)),
    }
}
fn ready(peer: &mut Peer, node: u64, whole: bool) -> io::Result<()> {
    peer.ok(1, prepare(node, whole), 0)?;
    peer.ok(
        2,
        Operation::Commit {
            lease_id: fake::lease(node).lease_id,
        },
        0,
    )?;
    if whole {
        return Ok(());
    }
    peer.ok(
        3,
        Operation::Begin {
            session: Box::new(fake::session(node)),
        },
        0,
    )?;
    Ok(())
}
fn start(node: u64, stage: u32, payload: &[u8]) -> Operation {
    Operation::StartStep {
        header: Box::new(fake::header(node, stage)),
        input_digest: byte_digest(payload),
    }
}
fn complete(node: u64) -> Operation {
    Operation::CompleteStep {
        generation_id: fake::session(node).generation_id,
        invocation_id: fake::header(node, 0).invocation_id,
    }
}
fn release(node: u64) -> Operation {
    Operation::Release {
        lease_id: fake::lease(node).lease_id,
    }
}
fn cancel(node: u64) -> Operation {
    Operation::Cancel {
        generation_id: fake::session(node).generation_id,
    }
}
fn finish(node: u64) -> Operation {
    Operation::Finish {
        generation_id: fake::session(node).generation_id,
    }
}
fn state(reply: &Reply) -> Option<SessionState> {
    reply
        .snapshot
        .generations
        .get(&fake::session(1).generation_id)
        .copied()
}
fn pair() -> io::Result<(Peer, Peer)> {
    let a = Peer::spawn(1, fake::CAPACITY)?;
    let b = Peer::spawn(2, fake::CAPACITY)?;
    require(
        a.child.id() != b.child.id()
            && a.child.id() != std::process::id()
            && b.child.id() != std::process::id(),
        "workers must be separate processes",
    )?;
    Ok((a, b))
}
fn stage(peer: &mut Peer, node: u64, index: u32, payload: Vec<u8>) -> io::Result<Vec<u8>> {
    let mut op = start(node, index, &payload);
    if let Operation::StartStep { header, .. } = &mut op {
        header.plan_digest = peer.plan_digest.clone();
    }
    let mut wrong_plan = op.clone();
    if let Operation::StartStep { header, .. } = &mut wrong_plan {
        header.plan_digest = byte_digest(b"wrong plan");
    }
    let refused = peer.apply(98, wrong_plan, 0, payload.clone(), 5)?;
    require(
        refused.error.is_some() && refused.executions == 0,
        "wrong plan digest executed",
    )?;
    let mut corrupt = payload.clone();
    corrupt[0] ^= 1;
    let refused = peer.apply(99, op.clone(), 0, corrupt, 5)?;
    require(
        refused.error == Some(CoreError::Invalid) && refused.executions == 0,
        "corrupt payload executed",
    )?;
    let r = peer.apply(10, op.clone(), 0, payload.clone(), 5)?;
    require(
        r.error.is_none() && r.executions == 1,
        "stage did not start",
    )?;
    let duplicate = peer.apply(10, op.clone(), 0, payload.clone(), 5)?;
    require(
        duplicate.duplicate && duplicate.executions == 1,
        "duplicate stage executed twice",
    )?;
    let early = peer.ok_or_error(11, complete(node), 4)?;
    require(
        early.error == Some(CoreError::State),
        "fake delay was ignored",
    )?;
    let r = peer.ok(11, complete(node), 5)?;
    require(
        r.output
            == payload
                .iter()
                .map(|b| {
                    if node == 1 {
                        b.wrapping_add(1)
                    } else {
                        b.rotate_left(1)
                    }
                })
                .collect::<Vec<_>>(),
        "incorrect deterministic tensor output",
    )?;
    let replay = peer.ok(11, complete(node), 5)?;
    require(
        replay.duplicate && replay.output == r.output && replay.executions == 1,
        "completion replay changed output",
    )?;
    let retry = peer.apply(12, op, 5, payload, 5)?;
    require(
        retry.error.is_some() && retry.executions == 1,
        "old invocation executed again",
    )?;
    let finished = peer.ok(13, finish(node), 5)?;
    require(
        state(&finished) == Some(SessionState::Completed),
        "session did not complete",
    )?;
    let late = peer.ok(14, cancel(node), 5)?;
    require(
        state(&late) == Some(SessionState::Completed),
        "late cancellation revived completion",
    )?;
    peer.ok(15, release(node), 5)?;
    require(
        peer.ok_or_error(11, complete(node), 5)?.error == Some(CoreError::OutputWithdrawn),
        "released completion replay exposed output",
    )?;
    Ok(r.output)
}
impl Peer {
    fn ok_or_error(&mut self, n: u64, op: Operation, now: u64) -> io::Result<Reply> {
        self.apply(n, op, now, vec![], 0)
    }
}
pub fn run() -> io::Result<()> {
    // Both independent whole requests execute while the other worker stays alive.
    let (mut a, mut b) = pair()?;
    ready(&mut a, 1, true)?;
    ready(&mut b, 2, true)?;
    let input = vec![0, 1, 2, 3, 252, 253, 254, 255];
    for (node, peer) in [(1, &mut a), (2, &mut b)] {
        let session = fake::session(node);
        let request = ChatRequest {
            model: fake::plan().manifest_digest.to_string(),
            messages: vec![ChatMessage {
                role: ChatMessageRole::User,
                content: "synthetic".into(),
            }],
            max_tokens: session.max_output_tokens,
            temperature: Some(session.sampling.temperature),
            top_p: Some(session.sampling.top_p),
            seed: Some(session.sampling.seed),
            stream: Some(false),
        };
        let operation = Operation::StartRequest {
            session: Box::new(session),
            invocation_id: fake::header(node, 0).invocation_id,
            request: Box::new(request),
        };
        let started = peer.ok(10, operation.clone(), 0)?;
        let duplicate = peer.ok(10, operation, 0)?;
        require(
            started.executions == 1 && duplicate.duplicate && duplicate.executions == 1,
            "whole-request replay executed twice",
        )?;
        let completed = peer.ok(11, complete(node), 0)?;
        require(
            state(&completed) == Some(SessionState::Completed)
                && completed.output == format!("synthetic worker {node}").as_bytes(),
            "whole-request completion missing",
        )?;
        peer.ok(12, release(node), 0)?;
        require(
            peer.ok_or_error(11, complete(node), 0)?.error == Some(CoreError::OutputWithdrawn),
            "released whole-request output survived",
        )?;
    }
    drop((a, b));
    // Commit only after every stage has reserved capacity.
    let (mut a, mut b) = pair()?;
    a.ok(1, prepare(1, false), 0)?;
    b.ok(1, prepare(2, false), 0)?;
    for (node, peer) in [(1, &mut a), (2, &mut b)] {
        peer.ok(
            2,
            Operation::Commit {
                lease_id: fake::lease(node).lease_id,
            },
            0,
        )?;
        peer.ok(
            3,
            Operation::Begin {
                session: Box::new(fake::session(node)),
            },
            0,
        )?;
    }
    let intermediate = stage(&mut a, 1, 0, input.clone())?;
    let result = stage(&mut b, 2, 1, intermediate)?;
    require(
        result
            == input
                .iter()
                .map(|b| b.wrapping_add(1).rotate_left(1))
                .collect::<Vec<_>>(),
        "pipeline order/output mismatch",
    )?;
    drop((a, b));
    // The script requests rollback after a failed second reservation; verify worker cleanup.
    let mut a = Peer::spawn(1, fake::CAPACITY)?;
    let mut b = Peer::spawn(2, 1)?;
    a.ok(1, prepare(1, false), 0)?;
    let r = b.ok_or_error(1, prepare(2, false), 0)?;
    require(
        r.error == Some(CoreError::Capacity),
        "capacity refusal missing",
    )?;
    require(
        a.ok(2, release(1), 0)?.snapshot.reserved_bytes.0 == 0,
        "partial prepare leaked capacity",
    )?;
    drop((a, b));
    // In-flight cancellation and expiry retain capacity until work acknowledges termination.
    for expiry in [false, true] {
        let (mut a, mut b) = pair()?;
        ready(&mut a, 1, false)?;
        ready(&mut b, 2, false)?;
        a.apply(10, start(1, 0, &input), 0, input.clone(), 5)?;
        let now = if expiry { 15_000 } else { 0 };
        let r = if expiry {
            a.send(WireCommand::Inspect { now })?
        } else {
            a.ok(11, release(1), now)?
        };
        require(
            state(&r) == Some(SessionState::Cancelling)
                && r.snapshot.reserved_bytes.0 == fake::CAPACITY,
            "in-flight memory released early",
        )?;
        let now = now.max(5);
        let r = a.ok(12, complete(1), now)?;
        require(
            r.output.is_empty() && r.snapshot.reserved_bytes.0 == 0,
            "cancelled work emitted output or leaked memory",
        )?;
        require(
            state(&r)
                == Some(if expiry {
                    SessionState::Failed
                } else {
                    SessionState::Cancelled
                }),
            "incorrect terminal state",
        )?;
        let started = b.apply(30, start(2, 1, &input), 0, input.clone(), 5)?;
        require(started.error.is_none(), "second stage did not start")?;
        require(
            !b.ok(31, complete(2), 5)?.output.is_empty(),
            "completed output missing",
        )?;
        b.ok(13, release(2), now)?;
        let replay = b.ok_or_error(31, complete(2), now)?;
        require(
            replay.error == Some(CoreError::OutputWithdrawn) && replay.output.is_empty(),
            "cancelled generation replay leaked cached output",
        )?;
    }
    // Observe EOF versus process exit, then explicitly request surviving-worker cleanup.
    for fault in [WireCommand::Disconnect, WireCommand::Crash] {
        let (mut a, mut b) = pair()?;
        ready(&mut a, 1, false)?;
        ready(&mut b, 2, false)?;
        require(
            a.apply(20, start(1, 0, &input), 0, input.clone(), 5)?
                .error
                .is_none(),
            "surviving stage did not start",
        )?;
        require(
            b.apply(20, start(2, 1, &input), 0, input.clone(), 5)?
                .error
                .is_none(),
            "faulted stage did not start",
        )?;
        require(b.send(fault).is_err(), "worker fault was not observed")?;
        let draining = a.ok(10, release(1), 0)?;
        require(
            draining.snapshot.reserved_bytes.0 == fake::CAPACITY,
            "worker loss freed in-flight memory early",
        )?;
        let r = a.ok(21, complete(1), 5)?;
        require(
            r.output.is_empty(),
            "worker loss exposed partial pipeline output",
        )?;
        require(
            r.snapshot.reserved_bytes.0 == 0 && state(&r) == Some(SessionState::Cancelled),
            "disconnect cleanup failed",
        )?;
    }
    let mut registry = Registry::new(FixtureMode::Simulation);
    let node: NodeCapabilities = decode(
        include_bytes!("../../../examples/node-capabilities.json"),
        FixtureMode::Simulation,
    )
    .map_err(|_| failure("invalid capability fixture"))?;
    registry
        .observe(node.clone(), 0)
        .map_err(|_| failure("capability observation failed"))?;
    registry
        .observe(node.clone(), 999)
        .map_err(|_| failure("replay observation failed"))?;
    require(
        registry.eligible(node.node_id, 1000, 1000) == Err(CoreError::StaleCapability),
        "stale telemetry was scheduled",
    )?;
    println!(
        "simulation only: 8 cluster scenarios passed; separate coordinator and two workers; no model execution"
    );
    Ok(())
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    fn stalled() -> Peer {
        let mut child = Command::new("sleep")
            .arg("30")
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .spawn()
            .unwrap();
        let input = child.stdin.take();
        Peer {
            child,
            input,
            output: None,
            reader: None,
            plan_digest: fake::plan().plan_digest,
            sequences: BTreeMap::new(),
        }
    }
    #[test]
    fn stalled_writes_and_mismatched_replies_are_refused() {
        let mut peer = stalled();
        let request = ControlRequest::new(fake::id(700), 1, 700, release(1)).unwrap();
        let start = std::time::Instant::now();
        let result = peer.send_with_timeout(
            WireCommand::Apply {
                request: Box::new(request.clone()),
                now: 0,
                payload: vec![0; MAX_JSON_BYTES / 3],
                delay_ms: 0,
            },
            Duration::from_millis(100),
        );
        assert!(result.unwrap_err().to_string().contains("write deadline"));
        assert!(start.elapsed() < Duration::from_secs(3));
        drop(peer);
        let mut peer = stalled();
        let (tx, rx) = mpsc::sync_channel(1);
        peer.output = Some(rx);
        tx.send(Ok(Reply {
            request_id: Some(fake::id(701)),
            error: None,
            duplicate: false,
            snapshot: starlings_core::Worker::new(
                fake::id(1),
                fake::plan().cluster_id,
                fake::CAPACITY,
                1,
                FixtureMode::Simulation,
            )
            .unwrap()
            .snapshot(),
            output: vec![],
            executions: 0,
        }))
        .unwrap();
        let result = peer.send_with_timeout(
            WireCommand::Apply {
                request: Box::new(request),
                now: 0,
                payload: vec![],
                delay_ms: 0,
            },
            Duration::from_millis(100),
        );
        assert!(result.unwrap_err().to_string().contains("request mismatch"));
    }
}
