use super::*;
use crate::{
    Action, Authority, Channel, ClientToken, Error, Identity, RegistryFence, Scope, SecuredWorker,
    enrollment::tests::{enrolled, pair, setup},
};
use starlings_core::Worker;
use starlings_protocol::*;
use std::{
    net::{TcpListener, TcpStream},
    thread,
    time::Duration,
};
mod benchmark;
mod range;

struct Fixture {
    _temp: tempfile::TempDir,
    authority: Authority,
    identities: [Identity; 2],
    workers: [SecuredWorker; 2],
    fences: [RegistryFence; 2],
    controls: [(PeerStream, PeerStream); 2],
    token: ClientToken,
    plan: PlacementPlan,
    leases: [Lease; 2],
    session: Session,
    clock: Instant,
}
impl Fixture {
    fn new() -> Self {
        Self::with_scratch(1048576)
    }
    fn with_scratch(scratch: u64) -> Self {
        Self::with_workload(scratch, 4, 8)
    }
    fn with_workload(scratch: u64, prompt: u32, output: u32) -> Self {
        let (temp, _, mut authority, files) = setup();
        let identities = [
            enrolled(&mut authority, &files, crate::Role::Worker, "source"),
            enrolled(&mut authority, &files, crate::Role::Worker, "destination"),
        ];
        let controller = enrolled(&mut authority, &files, crate::Role::Controller, "client");
        let mode = FixtureMode::Simulation;
        let mut plan: PlacementPlan = decode(
            include_bytes!("../../../../examples/placement-plan.json"),
            mode,
        )
        .unwrap();
        plan.cluster_id = identities[0].cluster();
        plan.manifest_digest = byte_digest(b"synthetic data fixture");
        plan.boundary_profile = "synthetic-little-endian-v1".into();
        for (stage, node) in plan.stages.iter_mut().zip(&identities) {
            stage.node_id = node.node();
            stage.scratch_budget_bytes = Decimal(scratch);
        }
        plan.plan_digest =
            canonical_digest(&serde_json::to_value(&plan).unwrap(), "plan_digest").unwrap();
        let infer = Scope {
            model: plan.manifest_digest.clone(),
            action: Action::Infer,
        };
        for node in [&identities[0], &identities[1], &controller] {
            authority
                .set_permissions(node.node(), std::slice::from_ref(&infer))
                .unwrap();
        }
        authority
            .set_permissions(
                authority.identity().node(),
                &[Scope {
                    model: plan.manifest_digest.clone(),
                    action: Action::Control,
                }],
            )
            .unwrap();
        let token = authority
            .issue_client_token(controller.node(), &[infer], 60)
            .unwrap();
        let epoch = AuthorityEpoch {
            number: Decimal(1),
            incarnation: uuid::Uuid::new_v4(),
        };
        let fences = std::array::from_fn(|i| {
            RegistryFence::test_online_epoch(identities[i].node(), plan.cluster_id, epoch)
        });
        let workers = std::array::from_fn(|i| {
            authority
                .secure_worker(
                    Worker::new(
                        identities[i].node(),
                        plan.cluster_id,
                        plan.stages[i].reserved_bytes().unwrap(),
                        1,
                        mode,
                    )
                    .unwrap(),
                )
                .unwrap()
                .bind_registry(fences[i].clone())
                .unwrap()
        });
        let leases = std::array::from_fn(|i| {
            let mut lease: Lease =
                decode(include_bytes!("../../../../examples/lease.json"), mode).unwrap();
            lease.lease_id = uuid::Uuid::new_v4().to_string().parse().unwrap();
            lease.owner_node_id = identities[i].node();
            lease.cluster_id = plan.cluster_id;
            lease.plan_digest = plan.plan_digest.clone();
            lease.reserved_bytes = Decimal(plan.stages[i].reserved_bytes().unwrap());
            lease.lifetime_ms = BoundedU32::new(60_000).unwrap();
            lease
        });
        let mut session: Session =
            decode(include_bytes!("../../../../examples/session.json"), mode).unwrap();
        session.plan_digest = plan.plan_digest.clone();
        session.prompt_token_count = BoundedU32::new(prompt).unwrap();
        session.max_output_tokens = BoundedU32::new(output).unwrap();
        let trust = authority.trust().unwrap();
        let controls = std::array::from_fn(|i| pair(&identities[i], authority.identity(), &trust));
        let clock = Instant::now();
        for i in 0..2 {
            let mut local_session = session.clone();
            local_session.lease_id = leases[i].lease_id;
            for (j, op) in [
                Operation::Prepare {
                    plan: Box::new(plan.clone()),
                    lease: Box::new(leases[i].clone()),
                },
                Operation::Commit {
                    lease_id: leases[i].lease_id,
                },
                Operation::Begin {
                    session: Box::new(local_session),
                },
            ]
            .into_iter()
            .enumerate()
            {
                let request = ControlRequest::new(
                    uuid::Uuid::new_v4().to_string().parse().unwrap(),
                    1,
                    j as u64 + 1,
                    op,
                )
                .unwrap();
                workers[i]
                    .apply(
                        &controls[i].0,
                        &token,
                        &request,
                        clock.elapsed().as_millis() as u64,
                    )
                    .unwrap();
            }
        }
        Self {
            _temp: temp,
            authority,
            identities,
            workers,
            fences,
            controls,
            token,
            plan,
            leases,
            session,
            clock,
        }
    }
    fn header(&self, vector: &serde_json::Value) -> TensorHeader {
        let mut h: TensorHeader = decode(
            vector["header_utf8"].as_str().unwrap().as_bytes(),
            FixtureMode::Simulation,
        )
        .unwrap();
        h.plan_digest = self.plan.plan_digest.clone();
        h.boundary_profile = self.plan.boundary_profile.clone();
        h.session_id = self.session.session_id;
        h.generation_id = self.session.generation_id;
        let destination = usize::from(h.dtype != TensorHeaderDtype::U32);
        h.source_stage = BoundedU32::new(1 - destination as u32).unwrap();
        h.destination_stage = BoundedU32::new(destination as u32).unwrap();
        h.lease_id = self.leases[destination].lease_id;
        h
    }
    fn admission<'a>(&'a self, peer: &PeerStream, h: TensorHeader) -> crate::DataAdmission<'a> {
        let i = usize::from(peer.local() == self.identities[1].node());
        self.workers[i]
            .data_admission(peer, &self.token, h, self.clock)
            .unwrap()
    }
    fn connect(&self, h: &TensorHeader) -> (DataStream, DataStream) {
        self.connect_with_window(h, None)
    }
    fn connect_with_window(
        &self,
        h: &TensorHeader,
        window: Option<usize>,
    ) -> (DataStream, DataStream) {
        let source = h.source_stage.get() as usize;
        let destination = h.destination_stage.get() as usize;
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let trust = self.authority.trust().unwrap();
        thread::scope(|scope| {
            let task = scope.spawn(|| {
                let socket = listener.accept().unwrap().0;
                if let Some(bytes) = window {
                    socket2::SockRef::from(&socket)
                        .set_recv_buffer_size(bytes)
                        .unwrap();
                }
                let peer = PeerStream::accept(
                    socket,
                    &self.identities[destination],
                    &trust,
                    Channel::Data,
                )
                .unwrap();
                let admission = self.admission(&peer, h.clone());
                DataStream::accept(peer, &admission).unwrap()
            });
            let socket = TcpStream::connect(address).unwrap();
            if let Some(bytes) = window {
                socket2::SockRef::from(&socket)
                    .set_send_buffer_size(bytes)
                    .unwrap();
            }
            let peer = PeerStream::connect(
                socket,
                &self.identities[source],
                &trust,
                self.identities[destination].node(),
                Channel::Data,
            )
            .unwrap();
            let admission = self.admission(&peer, h.clone());
            (
                DataStream::connect(peer, &admission).unwrap(),
                task.join().unwrap(),
            )
        })
    }
}
fn vectors() -> Vec<serde_json::Value> {
    serde_json::from_slice(include_bytes!("../../../../examples/gcp1-golden.json")).unwrap()
}

#[test]
fn flow_reassembles_a_tensor_larger_than_one_wire_frame_without_a_second_payload_buffer() {
    let size = 16 * 1024 * 1024;
    let f = Fixture::with_scratch(size);
    let mut h = f.header(&vectors()[2]);
    h.shape = vec![
        BoundedU32::new(2).unwrap(),
        BoundedU32::new(2).unwrap(),
        BoundedU32::new(1048576).unwrap(),
    ];
    h.total_tensor_bytes = Decimal(size);
    h.chunk_bytes = Decimal(DATA_PAYLOAD_LIMIT as u64);
    h.end_of_tensor = false;
    let (mut sender, mut receiver) = f.connect(&h);
    let source = [f.admission(&sender.peer, h.clone())];
    let destination = [f.admission(&receiver.peer, h)];
    let budget = ReceiveBudget::new(size, size, size).unwrap();
    let bytes: Vec<u8> = (0..size).map(|i| (i % 251) as u8).collect();
    let tensors = thread::scope(|threads| {
        let receive = threads.spawn(|| {
            let mut rx = receiver
                .begin_receive(
                    &destination,
                    &budget,
                    Instant::now() + Duration::from_secs(10),
                )
                .unwrap();
            rx.grant_credit(size).unwrap();
            rx.receive_chunk().unwrap();
            rx.receive_chunk().unwrap();
            rx.finish().unwrap()
        });
        let mut tx = sender
            .begin_send(&source, Instant::now() + Duration::from_secs(10))
            .unwrap();
        tx.receive_control().unwrap();
        for chunk in bytes.chunks(DATA_PAYLOAD_LIMIT) {
            tx.send_chunk(chunk).unwrap();
            tx.receive_control().unwrap();
        }
        tx.finish().unwrap();
        receive.join().unwrap()
    });
    assert_eq!(budget.held_bytes().unwrap(), size);
    assert!(
        tensors.tensors()[0].as_ref() == bytes.as_slice(),
        "large credited tensor payload changed"
    );
    drop(tensors);
    assert_eq!(budget.held_bytes().unwrap(), 0);
}

#[test]
fn flow_caller_budget_cannot_exceed_the_live_local_lease_scratch_budget() {
    let f = Fixture::new();
    let mut h = f.header(&vectors()[0]);
    h.shape = vec![
        BoundedU32::new(1).unwrap(),
        BoundedU32::new(2).unwrap(),
        BoundedU32::new(262144).unwrap(),
    ];
    h.total_tensor_bytes = Decimal(1048576);
    h.chunk_bytes = h.total_tensor_bytes;
    h.tensor_count = BoundedU32::new(2).unwrap();
    let mut second = h.clone();
    second.tensor_index = BoundedU32::new(1).unwrap();
    second.tensor_role = TensorHeaderTensorRole::Residual;
    let (_sender, mut receiver) = f.connect(&h);
    let admissions = [
        f.admission(&receiver.peer, h),
        f.admission(&receiver.peer, second),
    ];
    let generous = ReceiveBudget::new(2097152, 2097152, 2097152).unwrap();
    assert!(matches!(
        receiver.begin_receive(
            &admissions,
            &generous,
            Instant::now() + Duration::from_secs(10)
        ),
        Err(Error::Capacity)
    ));
    assert_eq!(generous.held_bytes().unwrap(), 0);
}

#[test]
fn flow_preserves_all_four_raw_dtype_vectors_without_numeric_conversion() {
    let f = Fixture::new();
    for vector in vectors() {
        for fence in &f.fences {
            fence.test_heartbeat();
        }
        let h = f.header(&vector);
        let bytes: Vec<u8> = serde_json::from_value(vector["payload"].clone()).unwrap();
        let (mut sender, mut receiver) = f.connect(&h);
        let source = [f.admission(&sender.peer, h.clone())];
        let destination = [f.admission(&receiver.peer, h)];
        let budget =
            ReceiveBudget::new(bytes.len() as u64, bytes.len() as u64, bytes.len() as u64).unwrap();
        let tensors = thread::scope(|threads| {
            let receiving = threads.spawn(|| {
                let mut flow = receiver
                    .begin_receive(
                        &destination,
                        &budget,
                        Instant::now() + Duration::from_secs(10),
                    )
                    .unwrap();
                flow.grant_credit(bytes.len() as u64).unwrap();
                flow.receive_chunk().unwrap();
                flow.finish().unwrap()
            });
            let mut flow = sender
                .begin_send(&source, Instant::now() + Duration::from_secs(10))
                .unwrap();
            flow.receive_control().unwrap();
            flow.send_chunk(&bytes).unwrap();
            flow.receive_control().unwrap();
            flow.finish().unwrap();
            receiving.join().unwrap()
        });
        assert_eq!(tensors.tensors()[0].as_ref(), bytes);
        drop(tensors);
        assert_eq!(budget.held_bytes().unwrap(), 0);
    }
}

#[test]
fn flow_duplicate_chunks_and_receipts_for_unsent_bytes_poison_the_connection() {
    for fake_receipt in [false, true] {
        let f = Fixture::new();
        let h = f.header(&vectors()[0]);
        for fence in &f.fences {
            fence.test_heartbeat();
        }
        let (mut sender, mut receiver) = f.connect(&h);
        let source = [f.admission(&sender.peer, h.clone())];
        let destination = [f.admission(&receiver.peer, h.clone())];
        if fake_receipt {
            let mut tx = sender
                .begin_send(&source, Instant::now() + Duration::from_secs(10))
                .unwrap();
            let header = DataHeader::Control(DataControl::ReceiptAck {
                protocol_version: Version,
                scope: (&h).into(),
                tensor_index: h.tensor_index,
                received_through_bytes: h.total_tensor_bytes,
            });
            receiver.send(&destination[0], &header, &[]).unwrap();
            assert!(matches!(tx.receive_control(), Err(Error::Replay)));
            assert!(tx.send_chunk(&[0; 8]).is_err());
        } else {
            let budget = ReceiveBudget::new(8, 8, 8).unwrap();
            let mut rx = receiver
                .begin_receive(
                    &destination,
                    &budget,
                    Instant::now() + Duration::from_secs(10),
                )
                .unwrap();
            rx.grant_credit(8).unwrap();
            sender.receive(&source[0], |_| ()).unwrap();
            let mut chunk = h.clone();
            chunk.chunk_bytes = Decimal(4);
            chunk.end_of_tensor = false;
            sender
                .send(&source[0], &DataHeader::Tensor(chunk.clone()), &[0; 4])
                .unwrap();
            rx.receive_chunk().unwrap();
            sender.receive(&source[0], |_| ()).unwrap();
            sender
                .send(&source[0], &DataHeader::Tensor(chunk), &[0; 4])
                .unwrap();
            assert_eq!(rx.receive_chunk(), Err(Error::Capacity));
            assert_eq!(budget.held_bytes().unwrap(), 0);
            assert!(rx.finish().is_err());
        }
    }
}

#[test]
fn flow_final_handoff_rechecks_pending_execution_and_never_drains_its_reservation() {
    let f = Fixture::new();
    let h = f.header(&vectors()[0]);
    let (mut sender, mut receiver) = f.connect(&h);
    let source = f.admission(&sender.peer, h.clone());
    let destination = [f.admission(&receiver.peer, h.clone())];
    let budget = ReceiveBudget::new(8, 8, 8).unwrap();
    let mut flow = receiver
        .begin_receive(
            &destination,
            &budget,
            Instant::now() + Duration::from_secs(10),
        )
        .unwrap();
    flow.grant_credit(8).unwrap();
    sender.receive(&source, |_| ()).unwrap();
    sender
        .send(&source, &DataHeader::Tensor(h.clone()), &[0; 8])
        .unwrap();
    flow.receive_chunk().unwrap();
    sender.receive(&source, |_| ()).unwrap();
    let request = ControlRequest::new(
        uuid::Uuid::new_v4().to_string().parse().unwrap(),
        1,
        4,
        Operation::StartStep {
            header: Box::new(h.clone()),
            input_digest: byte_digest(&[0; 8]),
        },
    )
    .unwrap();
    f.workers[1]
        .apply(
            &f.controls[1].0,
            &f.token,
            &request,
            f.clock.elapsed().as_millis() as u64,
        )
        .unwrap();
    let pending = f.workers[1].snapshot().unwrap();
    assert!(
        flow.finish().is_err(),
        "a previously received tensor is no longer current input"
    );
    assert_eq!(
        budget.held_bytes().unwrap(),
        0,
        "only unexecuted reassembly buffers are freed"
    );
    let cancelled = f.workers[1].snapshot().unwrap();
    assert_eq!(
        cancelled.generations[&h.generation_id],
        SessionState::Cancelling
    );
    assert_eq!(cancelled.reserved_bytes, pending.reserved_bytes);
    assert_eq!(cancelled.instances, pending.instances);
    let release = ControlRequest::new(
        uuid::Uuid::new_v4().to_string().parse().unwrap(),
        1,
        5,
        Operation::Release {
            lease_id: f.leases[1].lease_id,
        },
    )
    .unwrap();
    f.workers[1]
        .apply(
            &f.controls[1].0,
            &f.token,
            &release,
            f.clock.elapsed().as_millis() as u64,
        )
        .unwrap();
    assert_eq!(
        f.workers[1].snapshot().unwrap().reserved_bytes,
        pending.reserved_bytes
    );
    f.workers[1]
        .failed_and_drained(h.generation_id, h.invocation_id)
        .unwrap();
    let drained = f.workers[1].snapshot().unwrap();
    assert_eq!(drained.reserved_bytes.0, 0);
    assert_eq!(drained.generations[&h.generation_id], SessionState::Failed);
}

#[test]
fn flow_reassembles_ordered_tensors_once_and_retains_ownership_after_receipts() {
    let f = Fixture::new();
    let mut h = f.header(&vectors()[0]);
    h.tensor_count = BoundedU32::new(2).unwrap();
    let mut residual = h.clone();
    residual.tensor_index = BoundedU32::new(1).unwrap();
    residual.tensor_role = TensorHeaderTensorRole::Residual;
    let headers = [h, residual];
    let (mut sender, mut receiver) = f.connect(&headers[0]);
    let source: Vec<_> = headers
        .iter()
        .map(|h| f.admission(&sender.peer, h.clone()))
        .collect();
    let destination: Vec<_> = headers
        .iter()
        .map(|h| f.admission(&receiver.peer, h.clone()))
        .collect();
    let before = serde_json::to_value(f.workers[1].snapshot().unwrap()).unwrap();
    let budget = ReceiveBudget::new(32, 32, 16).unwrap();
    let payloads = [
        vec![0, 0, 0, 128, 128, 127, 193, 127],
        vec![1, 255, 2, 128, 3, 127, 4, 126],
    ];
    let tensors = thread::scope(|threads| {
        let receive = threads.spawn(|| {
            let mut flow = receiver
                .begin_receive(
                    &destination,
                    &budget,
                    Instant::now() + Duration::from_secs(10),
                )
                .unwrap();
            assert_eq!(budget.held_bytes().unwrap(), 16);
            flow.grant_credit(8).unwrap();
            flow.grant_credit(8).unwrap();
            for _ in 0..4 {
                flow.receive_chunk().unwrap();
            }
            flow.finish().unwrap()
        });
        let mut flow = sender
            .begin_send(&source, Instant::now() + Duration::from_secs(10))
            .unwrap();
        for _ in 0..2 {
            assert!(matches!(
                flow.receive_control().unwrap(),
                DataControl::Credit { .. }
            ));
        }
        for bytes in &payloads {
            for chunk in bytes.chunks(4) {
                flow.send_chunk(chunk).unwrap();
                assert!(matches!(
                    flow.receive_control().unwrap(),
                    DataControl::ReceiptAck { .. }
                ));
            }
        }
        flow.finish().unwrap();
        receive.join().unwrap()
    });
    assert_eq!(tensors.scope(), &DataScope::from(&headers[0]));
    assert_eq!(
        tensors
            .tensors()
            .iter()
            .map(|b| b.as_ref())
            .collect::<Vec<_>>(),
        payloads.iter().map(Vec::as_slice).collect::<Vec<_>>()
    );
    assert_eq!(
        budget.held_bytes().unwrap(),
        16,
        "receipts never release owned input buffers"
    );
    assert_eq!(
        serde_json::to_value(f.workers[1].snapshot().unwrap()).unwrap(),
        before,
        "transport never starts or completes execution"
    );
    assert!(
        receiver
            .begin_receive(
                &destination,
                &budget,
                Instant::now() + Duration::from_secs(10)
            )
            .is_err(),
        "live ownership prevents a second handoff for this invocation"
    );
    drop(tensors);
    assert_eq!(budget.held_bytes().unwrap(), 0);
    // The persistent connection is reusable; output authority remains separate.
    let header = DataHeader::Control(DataControl::ExecutionResult {
        protocol_version: Version,
        scope: (&headers[0]).into(),
    });
    receiver.send(&destination[0], &header, &[]).unwrap();
    sender
        .receive(&source[0], |frame| assert_eq!(frame.header(), &header))
        .unwrap();
    assert_eq!(
        serde_json::to_value(f.workers[1].snapshot().unwrap()).unwrap(),
        before
    );
}

#[test]
fn flow_refuses_uncredited_and_reordered_headers_before_reading_their_payload() {
    for reordered in [false, true] {
        let f = Fixture::new();
        let mut h = f.header(&vectors()[0]);
        h.tensor_count = BoundedU32::new(2).unwrap();
        let mut second = h.clone();
        second.tensor_index = BoundedU32::new(1).unwrap();
        second.tensor_role = TensorHeaderTensorRole::Residual;
        let (mut sender, mut receiver) = f.connect(&h);
        let admissions = [
            f.admission(&receiver.peer, h.clone()),
            f.admission(&receiver.peer, second.clone()),
        ];
        let budget = ReceiveBudget::new(16, 16, 16).unwrap();
        let before = f.workers[1].snapshot().unwrap();
        let mut flow = receiver
            .begin_receive(
                &admissions,
                &budget,
                Instant::now() + Duration::from_secs(10),
            )
            .unwrap();
        let header = DataHeader::Tensor(if reordered { second } else { h });
        // Omit the claimed payload entirely. Capacity/Denied proves the gate
        // refused the header, rather than timing out while reading its body.
        sender
            .peer
            .frame_until(Instant::now() + Duration::from_secs(10), |io| {
                let (prefix, bytes) = header.encode(FixtureMode::Reject)?;
                io.write_all(&prefix.bytes())?;
                io.write_all(&bytes)?;
                io.flush()?;
                Ok(())
            })
            .unwrap();
        assert_eq!(
            flow.receive_chunk(),
            Err(if reordered {
                Error::Denied
            } else {
                Error::Capacity
            })
        );
        assert_eq!(budget.held_bytes().unwrap(), 0);
        drop(flow);
        assert!(receiver.receive(&admissions[0], |_| ()).is_err());
        let after = f.workers[1].snapshot().unwrap();
        assert_eq!(
            after.generations[&admissions[0].expected().generation_id],
            SessionState::Failed
        );
        assert_eq!(after.reserved_bytes, before.reserved_bytes);
        assert_eq!(after.instances, before.instances);
    }
}

#[test]
fn flow_truncation_and_local_cancel_release_only_unexecuted_host_buffers() {
    for truncated in [false, true] {
        let f = Fixture::new();
        let h = f.header(&vectors()[0]);
        let (mut sender, mut receiver) = f.connect(&h);
        let admission = [f.admission(&receiver.peer, h.clone())];
        let source = f.admission(&sender.peer, h.clone());
        let budget = ReceiveBudget::new(8, 8, 8).unwrap();
        let before = f.workers[1].snapshot().unwrap();
        let mut flow = receiver
            .begin_receive(
                &admission,
                &budget,
                Instant::now() + Duration::from_secs(10),
            )
            .unwrap();
        flow.grant_credit(8).unwrap();
        sender.receive(&source, |_| ()).unwrap();
        if truncated {
            sender
                .peer
                .frame_until(Instant::now() + Duration::from_secs(10), |io| {
                    let (prefix, bytes) =
                        DataHeader::Tensor(h.clone()).encode(FixtureMode::Reject)?;
                    io.write_all(&prefix.bytes())?;
                    io.write_all(&bytes)?;
                    io.write_all(&[1, 2])?;
                    io.flush()?;
                    Ok(())
                })
                .unwrap();
            drop(sender);
            assert!(flow.receive_chunk().is_err());
        }
        drop(flow);
        assert_eq!(budget.held_bytes().unwrap(), 0);
        assert_eq!(
            f.workers[1].snapshot().unwrap().generations[&h.generation_id],
            SessionState::Failed,
            "an ambiguous or cancelled transfer must terminate its owned generation"
        );
        assert!(receiver.receive(&admission[0], |_| ()).is_err());
        assert!(matches!(
            receiver.begin_receive(
                &admission,
                &budget,
                Instant::now() + Duration::from_secs(10)
            ),
            Err(Error::Expired)
        ));
        assert_eq!(budget.held_bytes().unwrap(), 0);
        let after = f.workers[1].snapshot().unwrap();
        assert_eq!(after.reserved_bytes, before.reserved_bytes);
        assert_eq!(after.instances, before.instances);
        assert!(
            f.workers[1]
                .data_admission(&receiver.peer, &f.token, h, f.clock)
                .is_err()
        );
    }
}

#[test]
fn flow_expired_final_handoff_aborts_generation_after_releasing_host_buffers() {
    let f = Fixture::new();
    let h = f.header(&vectors()[0]);
    let (mut sender, mut receiver) = f.connect(&h);
    let admissions = [f.admission(&receiver.peer, h.clone())];
    let source = f.admission(&sender.peer, h.clone());
    let budget = ReceiveBudget::new(8, 8, 8).unwrap();
    let deadline = Instant::now() + Duration::from_secs(1);
    let mut flow = receiver
        .begin_receive(&admissions, &budget, deadline)
        .unwrap();
    flow.grant_credit(8).unwrap();
    sender.receive(&source, |_| ()).unwrap();
    let bytes: Vec<u8> = serde_json::from_value(vectors()[0]["payload"].clone()).unwrap();
    sender
        .send(&source, &DataHeader::Tensor(h.clone()), &bytes)
        .unwrap();
    flow.receive_chunk().unwrap();
    sender.receive(&source, |_| ()).unwrap();
    let reserved = f.workers[1].snapshot().unwrap().reserved_bytes;
    let scope = Scope {
        model: f.plan.manifest_digest.clone(),
        action: Action::Infer,
    };
    let result = thread::scope(|threads| {
        f.authority
            .with_client_access(&f.token, &scope, || {
                let (started, waiting) = std::sync::mpsc::sync_channel(0);
                let finish = threads.spawn(move || {
                    started.send(()).unwrap();
                    flow.finish()
                });
                waiting.recv().unwrap();
                // Hold the real policy gate across the handoff's absolute deadline.
                thread::sleep(
                    deadline.saturating_duration_since(Instant::now()) + Duration::from_millis(20),
                );
                finish
            })
            .unwrap()
            .join()
            .unwrap()
    });
    assert!(matches!(result, Err(Error::Expired)));
    assert_eq!(budget.held_bytes().unwrap(), 0);
    let after = f.workers[1].snapshot().unwrap();
    assert_eq!(after.generations[&h.generation_id], SessionState::Failed);
    assert_eq!(after.reserved_bytes, reserved);
}

#[test]
fn flow_slow_reader_aborts_both_generations_and_frees_only_host_reassembly() {
    let size = DATA_PAYLOAD_LIMIT as u64;
    let f = Fixture::with_scratch(size);
    let mut h = f.header(&vectors()[2]);
    h.shape = vec![
        BoundedU32::new(2).unwrap(),
        BoundedU32::new(2).unwrap(),
        BoundedU32::new(524288).unwrap(),
    ];
    h.total_tensor_bytes = Decimal(size);
    h.chunk_bytes = Decimal(size);
    let (mut sender, mut receiver) = f.connect_with_window(&h, Some(8192));
    let source = [f.admission(&sender.peer, h.clone())];
    let destination = [f.admission(&receiver.peer, h.clone())];
    let budget = ReceiveBudget::new(size, size, size).unwrap();
    let reserved = [
        f.workers[0].snapshot().unwrap().reserved_bytes,
        f.workers[1].snapshot().unwrap().reserved_bytes,
    ];
    let mut rx = receiver
        .begin_receive(&destination, &budget, Instant::now() + transport::TIMEOUT)
        .unwrap();
    rx.grant_credit(size).unwrap();
    let started = Instant::now();
    let mut tx = sender
        .begin_send(&source, started + Duration::from_millis(500))
        .unwrap();
    tx.receive_control().unwrap();
    // The receiver stays connected but does not consume its credited payload.
    assert!(matches!(
        tx.send_chunk(&vec![0; size as usize]),
        Err(Error::Transport | Error::Expired)
    ));
    assert!(started.elapsed() < transport::TIMEOUT);
    assert!(tx.send_chunk(&[0; 4]).is_err());
    drop(tx);
    assert_eq!(budget.held_bytes().unwrap(), size);
    assert!(rx.receive_chunk().is_err());
    drop(rx);
    assert_eq!(budget.held_bytes().unwrap(), 0);
    for (worker, bytes) in f.workers.iter().zip(reserved) {
        let snapshot = worker.snapshot().unwrap();
        assert_eq!(snapshot.generations[&h.generation_id], SessionState::Failed);
        assert_eq!(snapshot.reserved_bytes, bytes);
    }
}

#[test]
fn flow_local_abort_needs_no_live_registry_or_peer_authority() {
    let f = Fixture::new();
    let h = f.header(&vectors()[0]);
    let (sender, mut receiver) = f.connect(&h);
    let admissions = [f.admission(&receiver.peer, h.clone())];
    let budget = ReceiveBudget::new(8, 8, 8).unwrap();
    let flow = receiver
        .begin_receive(&admissions, &budget, Instant::now() + transport::TIMEOUT)
        .unwrap();
    f.fences[1].test_transition("registry-offline");
    drop(sender);
    drop(flow);
    assert_eq!(budget.held_bytes().unwrap(), 0);
    assert_eq!(
        f.workers[1].snapshot().unwrap().generations[&h.generation_id],
        SessionState::Failed
    );
}

#[test]
fn persistent_frames_preserve_bits_and_reports_never_execute_or_free_memory() {
    let f = Fixture::new();
    let vectors = vectors();
    let h = f.header(&vectors[0]);
    let (mut sender, mut receiver) = f.connect(&h);
    let before = f.workers[1].snapshot().unwrap();
    for vector in &vectors[..3] {
        let h = f.header(vector);
        let payload: Vec<u8> = serde_json::from_value(vector["payload"].clone()).unwrap();
        let sa = f.admission(&sender.peer, h.clone());
        let ra = f.admission(&receiver.peer, h.clone());
        for (offset, chunk) in payload.chunks(4).enumerate() {
            let mut part = h.clone();
            part.chunk_offset_bytes = Decimal((offset * 4) as u64);
            part.chunk_bytes = Decimal(chunk.len() as u64);
            part.end_of_tensor = offset * 4 + chunk.len() == payload.len();
            sender.send(&sa, &DataHeader::Tensor(part), chunk).unwrap();
            assert_eq!(
                receiver
                    .receive(&ra, |frame| frame.payload().to_vec())
                    .unwrap(),
                chunk
            );
        }
        for report in [
            DataControl::Credit {
                protocol_version: Version,
                scope: (&h).into(),
                bytes: h.total_tensor_bytes,
            },
            DataControl::ReceiptAck {
                protocol_version: Version,
                scope: (&h).into(),
                tensor_index: h.tensor_index,
                received_through_bytes: h.total_tensor_bytes,
            },
            DataControl::ExecutionResult {
                protocol_version: Version,
                scope: (&h).into(),
            },
        ] {
            receiver
                .send(&ra, &DataHeader::Control(report.clone()), &[])
                .unwrap();
            sender
                .receive(&sa, |frame| {
                    assert_eq!(frame.header(), &DataHeader::Control(report))
                })
                .unwrap();
        }
        for report in [
            DataControl::Cancel {
                protocol_version: Version,
                scope: (&h).into(),
                code: DataFailure::Cancelled,
            },
            DataControl::Error {
                protocol_version: Version,
                scope: (&h).into(),
                code: DataFailure::Backend,
            },
        ] {
            sender
                .send(&sa, &DataHeader::Control(report.clone()), &[])
                .unwrap();
            receiver
                .receive(&ra, |frame| {
                    assert_eq!(frame.header(), &DataHeader::Control(report))
                })
                .unwrap();
        }
    }
    let after = f.workers[1].snapshot().unwrap();
    assert_eq!(
        serde_json::to_value(before).unwrap(),
        serde_json::to_value(after).unwrap()
    );
}

#[test]
fn buffered_header_parser_stops_before_payload_and_preserves_io_errors() {
    let mut header: TensorHeader = decode(
        include_bytes!("../../../../examples/tensor-header.json"),
        FixtureMode::Simulation,
    )
    .unwrap();
    header.plan_digest = byte_digest(b"synthetic buffered header");
    let header = DataHeader::Tensor(header);
    let (prefix, bytes) = header.encode(FixtureMode::Reject).unwrap();
    let raw = [prefix.bytes().as_slice(), bytes.as_slice()].concat();
    let mut reader = std::io::Cursor::new(raw.clone());
    assert_eq!(
        DataStream::read_header(&mut reader, DATA_HEADER_LIMIT).unwrap(),
        (header, bytes.len())
    );
    assert_eq!(reader.position() as usize, raw.len());
    let mut denied = std::io::Cursor::new(raw.clone());
    let result =
        DataStream::read_frame(&mut denied, Instant::now() + transport::TIMEOUT, |_, h| {
            if h.is_some() {
                Err(Error::Denied)
            } else {
                Ok(())
            }
        });
    assert_eq!(result.map(|_| ()), Err(Error::Denied));
    assert_eq!(denied.position() as usize, raw.len());
    let mut truncated = std::io::Cursor::new(&raw[..raw.len() - 1]);
    assert_eq!(
        DataStream::read_header(&mut truncated, DATA_HEADER_LIMIT).unwrap_err(),
        Error::Transport
    );
    let mut invalid = prefix.bytes();
    invalid[0] = 0;
    assert_eq!(
        DataStream::read_header(&mut std::io::Cursor::new(invalid), DATA_HEADER_LIMIT).unwrap_err(),
        Error::Invalid
    );
    assert_eq!(
        DataStream::read_header(&mut std::io::Cursor::new(prefix.bytes()), 0).unwrap_err(),
        Error::Capacity
    );
}

#[test]
fn malformed_or_unadmitted_headers_poison_the_stream_before_payload_read() {
    let f = Fixture::new();
    let h = f.header(&vectors()[0]);
    for case in [
        "magic",
        "version",
        "length",
        "shape",
        "session",
        "sequence",
        "dtype",
        "truncated",
    ] {
        let (mut sender, mut receiver) = f.connect(&h);
        let admission = f.admission(&receiver.peer, h.clone());
        let mut bad = h.clone();
        match case {
            "session" => bad.session_id = uuid::Uuid::new_v4().to_string().parse().unwrap(),
            "sequence" => bad.sequence = Decimal(1),
            "dtype" => {
                bad.dtype = TensorHeaderDtype::F16;
            }
            "shape" => {
                bad.shape = vec![BoundedU32::new(1_048_576).unwrap(); 4];
            }
            _ => (),
        }
        let bytes = serde_json::to_vec(&bad).unwrap();
        let mut prefix = DataPrefix::new(DataKind::Tensor, bytes.len(), h.chunk_bytes.0 as usize)
            .unwrap()
            .bytes();
        match case {
            "magic" => prefix[0] = 0,
            "version" => prefix[5] = 2,
            "length" => prefix[8..12].copy_from_slice(&u32::MAX.to_be_bytes()),
            _ => (),
        }
        let expected = match case {
            "session" | "sequence" | "dtype" => Error::Denied,
            "truncated" => Error::Transport,
            _ => Error::Invalid,
        };
        // The exact header gate must be proved without a socket deadline winning first.
        // No payload is available here: an accidental payload read fails this control.
        let mut input = std::io::Cursor::new([prefix.as_slice(), bytes.as_slice()].concat());
        let header_gate = DataStream::read_frame(
            &mut input,
            Instant::now() + transport::TIMEOUT,
            |_, header| admission.check(&receiver.peer, header, true, || ()),
        );
        assert_eq!(
            header_gate.map(|_| ()),
            Err(expected),
            "case={case} header gate"
        );
        sender.peer.write_all(&prefix).unwrap();
        sender.peer.write_all(&bytes).unwrap();
        sender.peer.flush().unwrap();
        let started = Instant::now();
        let deadline = started + Duration::from_millis(100);
        let result = receiver.receive_until(&admission, deadline, |_| panic!("unadmitted frame"));
        let finished = Instant::now();
        let deadline_refusal =
            finished >= deadline && matches!(result, Err(Error::Transport | Error::Expired));
        assert!(
            result == Err(expected) || deadline_refusal,
            "case={case} result={result:?} expected={expected:?} elapsed_us={}",
            finished.duration_since(started).as_micros()
        );
        assert_eq!(receiver.receive(&admission, |_| ()), Err(Error::Expired));
        assert_eq!(
            receiver.send(&admission, &DataHeader::Tensor(h.clone()), &[0; 8]),
            Err(Error::Expired)
        );
    }
}

#[test]
fn live_policy_and_fences_are_rechecked_at_delivery() {
    for reason in [
        "token",
        "permission",
        "node",
        "epoch",
        "offline",
        "lease",
        "cancel",
    ] {
        let f = Fixture::new();
        let h = f.header(&vectors()[0]);
        let (sender, receiver) = f.connect(&h);
        let admission = f.admission(&receiver.peer, h.clone());
        admission
            .check(
                &receiver.peer,
                Some(&DataHeader::Tensor(h.clone())),
                true,
                || (),
            )
            .unwrap();
        match reason {
            "token" => f.authority.revoke_client_token(&f.token).unwrap(),
            "permission" => f
                .authority
                .set_permissions(f.identities[0].node(), &[])
                .unwrap(),
            "node" => f.authority.revoke(f.identities[0].node()).unwrap(),
            "epoch" => f.fences[1].test_transition("registry-epoch"),
            "offline" => f.fences[1].test_transition("registry-offline"),
            "lease" => {
                f.fences[1].maintain(60_001).unwrap();
            }
            "cancel" => {
                let request = ControlRequest::new(
                    uuid::Uuid::new_v4().to_string().parse().unwrap(),
                    1,
                    4,
                    Operation::Cancel {
                        generation_id: h.generation_id,
                    },
                )
                .unwrap();
                f.workers[1]
                    .apply(
                        &f.controls[1].0,
                        &f.token,
                        &request,
                        f.clock.elapsed().as_millis() as u64,
                    )
                    .unwrap();
            }
            _ => unreachable!(),
        }
        let frame = DataFrame {
            header: DataHeader::Tensor(h),
            payload: vec![0; 8],
        };
        assert!(
            admission
                .with_frame(&receiver.peer, frame, true, |_| panic!(
                    "withdrawn frame delivered"
                ))
                .is_err(),
            "{reason}"
        );
        drop(sender);
    }
}

#[derive(Serialize, Deserialize)]
struct ChildBatch {
    source: usize,
    binding: DataBinding,
    headers: Vec<TensorHeader>,
    payloads: Vec<Vec<u8>>,
}
#[derive(Serialize, Deserialize)]
struct ChildInput {
    address: std::net::SocketAddr,
    ca: Vec<u8>,
    cluster: ClusterId,
    members: Vec<crate::enrollment::Member>,
    batches: Vec<ChildBatch>,
}
struct ChildGuard(std::process::Child);
impl Drop for ChildGuard {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

#[test]
fn golden_tensor_bytes_cross_a_child_process_over_enrolled_tls() {
    const CHILD: &str = "STARLINGS_TEST_DATA_CHILD";
    if let Some(path) = std::env::var_os(CHILD) {
        let directory = crate::PrivateDir::ensure(std::path::Path::new(&path)).unwrap();
        let input: ChildInput = serde_json::from_slice(
            &std::fs::read(std::path::Path::new(&path).join("input.json")).unwrap(),
        )
        .unwrap();
        // The byte producer has only enrolled device keys and public trust. It has
        // no authority database, client token, CA key or worker lifecycle handle.
        let trust = crate::Trust {
            ca: input.ca,
            cluster: input.cluster,
            members: input.members,
            access: None,
        };
        for batch in input.batches {
            let identity =
                Identity::load(&directory, &format!("device{}.json", batch.source)).unwrap();
            let destination = trust.member(batch.binding.destination).unwrap();
            let mut wire = transport::client(
                TcpStream::connect(input.address).unwrap(),
                &trust,
                destination,
                Some(&identity),
                b"starlings/1",
            )
            .unwrap();
            crate::send(&mut wire, &batch.binding).unwrap();
            let accepted: DataBinding = crate::receive(&mut wire).unwrap();
            assert_eq!(accepted, batch.binding);
            for (header, payload) in batch.headers.into_iter().zip(batch.payloads) {
                let (prefix, bytes) = DataHeader::Tensor(header)
                    .encode(FixtureMode::Reject)
                    .unwrap();
                assert_eq!(prefix.payload_len(), payload.len());
                wire.write_all(&prefix.bytes()).unwrap();
                wire.write_all(&bytes).unwrap();
                wire.write_all(&payload).unwrap();
                wire.flush().unwrap();
            }
        }
        return;
    }
    let f = Fixture::new();
    let vectors = vectors();
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    listener.set_nonblocking(true).unwrap();
    let trust = f.authority.trust().unwrap();
    let mut batches = Vec::new();
    for (source, indices) in [(0, 0..3), (1, 3..4)] {
        let destination = 1 - source;
        let headers: Vec<_> = indices.clone().map(|i| f.header(&vectors[i])).collect();
        let payloads = indices
            .map(|i| serde_json::from_value(vectors[i]["payload"].clone()).unwrap())
            .collect();
        let binding = DataBinding {
            version: Version,
            epoch: f.fences[destination].status().unwrap().0,
            plan: f.plan.plan_digest.clone(),
            source: f.identities[source].node(),
            destination: f.identities[destination].node(),
            source_stage: BoundedU32::new(source as u32).unwrap(),
            destination_stage: BoundedU32::new(destination as u32).unwrap(),
        };
        batches.push(ChildBatch {
            source,
            binding,
            headers,
            payloads,
        });
    }
    let input = ChildInput {
        address: listener.local_addr().unwrap(),
        ca: trust.ca.clone(),
        cluster: trust.cluster,
        members: trust.members.clone(),
        batches,
    };
    let path = f._temp.path().join("child");
    let directory = crate::PrivateDir::ensure(&path).unwrap();
    for (i, identity) in f.identities.iter().enumerate() {
        identity
            .save(&directory, &format!("device{i}.json"))
            .unwrap();
    }
    directory
        .create("input.json")
        .unwrap()
        .write_all(&serde_json::to_vec(&input).unwrap())
        .unwrap();
    let mut child = ChildGuard(
        std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "data::tests::golden_tensor_bytes_cross_a_child_process_over_enrolled_tls",
                "--nocapture",
            ])
            .env(CHILD, &path)
            .stdout(std::process::Stdio::null())
            .spawn()
            .unwrap(),
    );
    for batch in &input.batches {
        let deadline = Instant::now() + Duration::from_secs(10);
        let socket = loop {
            match listener.accept() {
                Ok((socket, _)) => break socket,
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => (),
                Err(_) => panic!("child listener failed"),
            }
            assert!(
                child.0.try_wait().unwrap().is_none(),
                "byte producer exited before connection"
            );
            assert!(
                Instant::now() < deadline,
                "byte producer connection watchdog"
            );
            thread::park_timeout(Duration::from_millis(5));
        };
        let destination = 1 - batch.source;
        socket.set_nonblocking(false).unwrap();
        let peer =
            PeerStream::accept(socket, &f.identities[destination], &trust, Channel::Data).unwrap();
        f.fences[destination].test_heartbeat();
        let first = f.admission(&peer, batch.headers[0].clone());
        let mut receiver = DataStream::accept(peer, &first).unwrap();
        for (header, payload) in batch.headers.iter().zip(&batch.payloads) {
            let admission = f.admission(&receiver.peer, header.clone());
            receiver
                .receive(&admission, |frame| {
                    assert_eq!(frame.header(), &DataHeader::Tensor(header.clone()));
                    assert_eq!(frame.payload(), payload);
                })
                .unwrap();
        }
    }
    let deadline = Instant::now() + Duration::from_secs(5);
    while child.0.try_wait().unwrap().is_none() {
        assert!(Instant::now() < deadline, "byte producer exit watchdog");
        thread::park_timeout(Duration::from_millis(5));
    }
    assert!(child.0.wait().unwrap().success());
}

#[test]
fn an_absolute_frame_deadline_covers_partial_prefix_header_and_payload() {
    let f = Fixture::new();
    for phase in 0..3 {
        let h = f.header(&vectors()[0]);
        let (mut sender, mut receiver) = f.connect(&h);
        let admission = f.admission(&receiver.peer, h.clone());
        let (prefix, bytes) = DataHeader::Tensor(h).encode(FixtureMode::Reject).unwrap();
        let prefix = prefix.bytes();
        let payload = [0; 8];
        let parts = [prefix.as_slice(), bytes.as_slice(), payload.as_slice()];
        let ready = std::sync::Barrier::new(2);
        let go = std::sync::Barrier::new(2);
        thread::scope(|scope| {
            let writer = scope.spawn(|| {
                ready.wait();
                go.wait();
                for (index, bytes) in parts.iter().enumerate() {
                    let chunk = if index == phase {
                        bytes.len().div_ceil(4)
                    } else {
                        bytes.len()
                    };
                    for part in bytes.chunks(chunk) {
                        if sender
                            .peer
                            .write_all(part)
                            .and_then(|()| sender.peer.flush())
                            .is_err()
                        {
                            return;
                        }
                        if index == phase {
                            thread::sleep(Duration::from_millis(30));
                        }
                    }
                }
            });
            ready.wait();
            let initial_io_deadline = receiver.peer.io_deadline();
            let deadline = Instant::now() + Duration::from_millis(60);
            go.wait();
            let result =
                receiver.receive_until(&admission, deadline, |_| panic!("late frame delivered"));
            assert!(
                Instant::now() >= deadline,
                "phase {phase} failed before its deadline: {result:?}"
            );
            let io_deadline = receiver.peer.io_deadline();
            if io_deadline == initial_io_deadline {
                assert_eq!(
                    result,
                    Err(Error::Expired),
                    "phase {phase} expired before I/O"
                );
            } else {
                assert_eq!(
                    io_deadline, deadline,
                    "phase {phase} renewed an I/O deadline"
                );
                assert!(
                    matches!(result, Err(Error::Transport | Error::Expired)),
                    "phase {phase}: {result:?}"
                );
            }
            assert_eq!(receiver.receive(&admission, |_| ()), Err(Error::Expired));
            writer.join().unwrap();
        });
    }
    let h = f.header(&vectors()[0]);
    let (_, mut receiver) = f.connect(&h);
    assert_eq!(
        receiver.peer.frame_until(
            Instant::now() - Duration::from_millis(1),
            |_| -> std::result::Result<(), Error> { panic!("expired operation started") }
        ),
        Err(Error::Expired)
    );
}

#[test]
fn admission_requires_the_control_owner_and_exact_negotiated_route() {
    let f = Fixture::new();
    let h = f.header(&vectors()[0]);
    let (mut sender, receiver) = f.connect(&h);
    assert!(
        f.workers[1]
            .data_admission(&f.controls[1].0, &f.token, h.clone(), f.clock)
            .is_err()
    );
    let controller = f
        .authority
        .trust()
        .unwrap()
        .members
        .into_iter()
        .find(|m| m.role == crate::Role::Controller)
        .unwrap();
    let other_token = f
        .authority
        .issue_client_token(
            controller.node,
            &[Scope {
                model: f.plan.manifest_digest.clone(),
                action: Action::Infer,
            }],
            60,
        )
        .unwrap();
    assert!(
        f.workers[1]
            .data_admission(&receiver.peer, &other_token, h.clone(), f.clock)
            .is_err()
    );
    for field in ["lease", "session", "epoch", "plan", "peer"] {
        let mut bad = h.clone();
        match field {
            "lease" => bad.lease_id = uuid::Uuid::new_v4().to_string().parse().unwrap(),
            "session" => bad.session_id = uuid::Uuid::new_v4().to_string().parse().unwrap(),
            "epoch" => bad.authority_epoch = Decimal(2),
            "plan" => bad.plan_digest = byte_digest(b"wrong plan"),
            "peer" => {
                bad.source_stage = BoundedU32::new(1).unwrap();
            }
            _ => unreachable!(),
        }
        assert!(
            f.workers[1]
                .data_admission(&receiver.peer, &f.token, bad, f.clock)
                .is_err(),
            "{field}"
        );
    }
    // A CA-enrolled peer's declarations cannot replace the locally derived binding.
    let admission = f.admission(&receiver.peer, h);
    let mut bad = admission.binding().clone();
    bad.epoch.incarnation = uuid::Uuid::new_v4();
    crate::send(&mut sender.peer, &bad).unwrap();
    let peer = receiver.peer;
    assert!(matches!(
        DataStream::accept(peer, &admission),
        Err(Error::Denied)
    ));
}
