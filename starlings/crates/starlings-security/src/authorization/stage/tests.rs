use super::*;
use crate::{
    GenerationBinding, Invitation, ListenerBudget, PeerListener, RegistryClient, RegistryRequest,
    enrollment::tests::{enrolled, setup},
};
use starlings_protocol::{Decimal, Lease, NodeCapabilities, Session, SessionState, TensorHeader};
use std::{net::TcpStream, thread};

struct Fixture {
    temp: tempfile::TempDir,
    authority: Authority,
    nodes: Vec<crate::Identity>,
    pins: Vec<Invitation>,
    registry: Vec<RegistryClient>,
    token: ClientToken,
    controller: NodeId,
    issuers: Vec<StageIssuer>,
    plan: PlacementPlan,
    leases: Vec<Lease>,
    session: Session,
    template: TensorHeader,
}
impl Fixture {
    fn new() -> Self {
        Self::after_preparation(|| ())
    }
    fn after_preparation(preparation: impl FnOnce()) -> Self {
        Self::create(preparation, false)
    }
    fn create(preparation: impl FnOnce(), third: bool) -> Self {
        let (temp, _, mut authority, files) = setup();
        let mut nodes = vec![
            enrolled(&mut authority, &files, Role::Worker, "stage0"),
            enrolled(&mut authority, &files, Role::Worker, "stage1"),
        ];
        if third {
            nodes.push(enrolled(&mut authority, &files, Role::Worker, "stage2"));
        }
        let controller = enrolled(&mut authority, &files, Role::Controller, "client");
        let listener =
            PeerListener::bind("127.0.0.1:0".parse().unwrap(), ListenerBudget::default()).unwrap();
        authority.start_registry().unwrap();
        let mut registry = Vec::new();
        let mut pins = Vec::new();
        for (i, node) in nodes.iter().enumerate() {
            let name = format!("pin{i}.json");
            authority
                .invite(&files, &name, listener.local_addr().unwrap(), Role::Worker)
                .unwrap();
            node.save(&files, &format!("identity{i}.json")).unwrap();
            let registry_path = temp.path().join(format!("registry{i}"));
            PrivateDir::ensure(&registry_path).unwrap();
            let client = RegistryClient::open(
                &registry_path,
                crate::Identity::load(&files, &format!("identity{i}.json")).unwrap(),
                Invitation::load(&files, &name).unwrap(),
            )
            .unwrap();
            registry.push(client);
            pins.push(Invitation::load(&files, &name).unwrap());
        }
        preparation();
        let mode = FixtureMode::Simulation;
        let mut plan: PlacementPlan = starlings_protocol::decode(
            include_bytes!("../../../../../examples/placement-plan.json"),
            mode,
        )
        .unwrap();
        plan.fixture = false;
        plan.cluster_id = nodes[0].cluster();
        plan.authority_epoch = authority.registry_epoch().unwrap().number;
        plan.manifest_digest = starlings_protocol::byte_digest(b"synthetic-stage-model");
        if third {
            plan.stages.push(plan.stages[1].clone());
        }
        for (i, (stage, node)) in plan.stages.iter_mut().zip(&nodes).enumerate() {
            stage.node_id = node.node();
            stage.backend_build_digest = starlings_protocol::byte_digest(b"synthetic-stage-build");
            if third {
                stage.index = BoundedU32::new(i as u32).unwrap();
                stage.layer_start =
                    BoundedU32::new(i as u32 * plan.model_layer_count.get() / 3).unwrap();
                stage.layer_end_exclusive =
                    BoundedU32::new((i as u32 + 1) * plan.model_layer_count.get() / 3).unwrap();
                stage.owns_embedding = i == 0;
                stage.owns_final_norm_and_head = i == 2;
            }
        }
        plan.plan_digest = starlings_protocol::canonical_digest(
            &serde_json::to_value(&plan).unwrap(),
            "plan_digest",
        )
        .unwrap();
        let infer = Scope {
            model: plan.manifest_digest.clone(),
            action: Action::Infer,
        };
        for node in nodes.iter().chain(std::iter::once(&controller)) {
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
        let mut lease: Lease =
            starlings_protocol::decode(include_bytes!("../../../../../examples/lease.json"), mode)
                .unwrap();
        lease.cluster_id = plan.cluster_id;
        lease.owner_node_id = nodes[0].node();
        lease.plan_digest = plan.plan_digest.clone();
        let mut session: Session = starlings_protocol::decode(
            include_bytes!("../../../../../examples/session.json"),
            mode,
        )
        .unwrap();
        session.plan_digest = plan.plan_digest.clone();
        session.lease_id = lease.lease_id;
        let mut template: TensorHeader = starlings_protocol::decode(
            include_bytes!("../../../../../examples/tensor-header.json"),
            mode,
        )
        .unwrap();
        template.plan_digest = plan.plan_digest.clone();
        let mut other_lease = lease.clone();
        other_lease.owner_node_id = nodes[1].node();
        other_lease.lease_id = uuid::Uuid::new_v4().to_string().parse().unwrap();
        let mut leases = vec![lease, other_lease];
        if third {
            let mut extra = leases[1].clone();
            extra.lease_id = uuid::Uuid::new_v4().to_string().parse().unwrap();
            extra.owner_node_id = nodes[2].node();
            leases.push(extra);
        }
        // Publish readiness only after durable policy and fixture preparation. A
        // snapshot is not renewed by constructing a plan or reading its issuer.
        for (node, client) in nodes.iter().zip(&mut registry) {
            for sample_sequence in 1..=2 {
                let mut caps: NodeCapabilities = starlings_protocol::decode(
                    include_bytes!("../../../../../examples/node-capabilities.json"),
                    FixtureMode::Simulation,
                )
                .unwrap();
                caps.fixture = false;
                caps.node_id = node.node();
                caps.cluster_id = node.cluster();
                caps.snapshot_id = uuid::Uuid::new_v4().to_string().parse().unwrap();
                caps.backends[0].build_digest =
                    starlings_protocol::byte_digest(b"synthetic-stage-build");
                let sample = crate::ResourceSample {
                    sequence: Decimal(sample_sequence),
                    capabilities: caps,
                    links: vec![],
                };
                thread::scope(|threads| {
                    let reader = authority.enrollment_reader().unwrap();
                    let server = threads.spawn(|| {
                        match listener.accept().unwrap().registry(reader).unwrap() {
                            RegistryRequest::Heartbeat(p) => p.respond().unwrap(),
                            _ => panic!("expected heartbeat"),
                        }
                    });
                    client.heartbeat(Some(sample)).unwrap();
                    server.join().unwrap();
                });
            }
        }
        let issuers = leases
            .iter()
            .map(|lease| {
                authority
                    .enrollment_reader()
                    .unwrap()
                    .stage_issuer(&token, plan.clone(), lease.clone())
                    .unwrap()
            })
            .collect();
        Self {
            temp,
            authority,
            nodes,
            pins,
            registry,
            token,
            controller: controller.node(),
            issuers,
            plan,
            leases,
            session,
            template,
        }
    }
    fn pair(&self) -> (GenerationStream, GenerationStream) {
        self.pair_at(0)
    }
    fn pair_at(&self, index: usize) -> (GenerationStream, GenerationStream) {
        let binding = GenerationBinding {
            version: Version,
            epoch: self.authority.registry_epoch().unwrap(),
            model: self.plan.manifest_digest.clone(),
            profile: self.plan.execution_profile.clone(),
        };
        let listener =
            PeerListener::bind("127.0.0.1:0".parse().unwrap(), ListenerBudget::default()).unwrap();
        thread::scope(|threads| {
            let server = threads.spawn(|| {
                listener
                    .accept()
                    .unwrap()
                    .generation(
                        &self.nodes[index],
                        &self.pins[index],
                        self.registry[index].fence(),
                        binding.clone(),
                    )
                    .unwrap()
            });
            let coordinator = GenerationStream::connect(
                TcpStream::connect(listener.local_addr().unwrap()).unwrap(),
                self.authority.identity(),
                &self.authority.trust().unwrap(),
                self.nodes[index].node(),
                binding.clone(),
            )
            .unwrap();
            (coordinator, server.join().unwrap())
        })
    }
    fn worker(&self, stream: &GenerationStream) -> StageWorker {
        self.worker_at(stream, 0)
    }
    fn worker_at(&self, stream: &GenerationStream, index: usize) -> StageWorker {
        self.worker_capacity(stream, index, 4_194_304)
    }
    fn worker_capacity(
        &self,
        stream: &GenerationStream,
        index: usize,
        capacity: u64,
    ) -> StageWorker {
        self.worker_owned(stream, index, capacity, false).0
    }
    fn worker_owned(
        &self,
        stream: &GenerationStream,
        index: usize,
        capacity: u64,
        retain: bool,
    ) -> (StageWorker, Option<StageBackend>) {
        let mut session = self.session.clone();
        session.lease_id = self.leases[index].lease_id;
        let worker = StageWorker::new(
            Worker::new(
                self.nodes[index].node(),
                self.plan.cluster_id,
                capacity,
                1,
                FixtureMode::Reject,
            )
            .unwrap(),
            self.registry[index].fence(),
            self.plan.clone(),
            self.issuers[index].id(),
        )
        .unwrap();
        let mut backend = None;
        for (i, operation) in [
            Operation::Prepare {
                plan: Box::new(self.plan.clone()),
                lease: Box::new(self.leases[index].clone()),
            },
            Operation::Commit {
                lease_id: self.leases[index].lease_id,
            },
            Operation::Begin {
                session: Box::new(session),
            },
        ]
        .into_iter()
        .enumerate()
        {
            let request = ControlRequest::new(
                uuid::Uuid::new_v4().to_string().parse().unwrap(),
                1,
                i as u64 + 1,
                operation,
            )
            .unwrap();
            assert!(!worker.apply(stream, &request).unwrap());
            if retain && i == 0 {
                backend = Some(
                    worker
                        .hold_backend(stream, self.leases[index].lease_id)
                        .unwrap(),
                );
            }
        }
        (worker, backend)
    }
    fn refresh(
        &self,
        owner: &StageWorker,
        coordinator: &mut GenerationStream,
        worker: &mut GenerationStream,
    ) -> Result<StageGrant> {
        self.refresh_at(
            owner,
            coordinator,
            worker,
            0,
            self.template.source_stage,
            self.template.destination_stage,
        )
    }
    fn refresh_at(
        &self,
        owner: &StageWorker,
        coordinator: &mut GenerationStream,
        worker: &mut GenerationStream,
        index: usize,
        source: BoundedU32<0, 7>,
        destination: BoundedU32<0, 7>,
    ) -> Result<StageGrant> {
        thread::scope(|threads| {
            let request = threads.spawn(|| {
                owner.refresh_grant(worker, self.leases[index].lease_id, source, destination)
            });
            self.issuers[index].respond(coordinator).unwrap();
            request.join().unwrap()
        })
    }
    fn received_pair(
        &self,
        owners: &[StageWorker; 2],
        grants: &[StageGrant; 2],
    ) -> (crate::ReceivedTensors, crate::ReceiveBudget) {
        let mut first = self.template.clone();
        first.lease_id = self.leases[1].lease_id;
        first.tensor_count = BoundedU32::new(2).unwrap();
        first.tensor_role = starlings_protocol::TensorHeaderTensorRole::Residual;
        let mut second = first.clone();
        second.tensor_index = BoundedU32::new(1).unwrap();
        second.tensor_role = starlings_protocol::TensorHeaderTensorRole::Hidden;
        let headers = [first, second];
        let bytes = headers[0].total_tensor_bytes.0;
        let budget = crate::ReceiveBudget::new(2 * bytes, 2 * bytes, 2 * bytes).unwrap();
        let listener =
            PeerListener::bind("127.0.0.1:0".parse().unwrap(), ListenerBudget::default()).unwrap();
        let received = thread::scope(|threads| {
            let sender = threads.spawn(|| {
                let peer = PeerStream::connect_stage(
                    TcpStream::connect(listener.local_addr().unwrap()).unwrap(),
                    &self.nodes[0],
                    &self.pins[0],
                    grants[0].clone(),
                )
                .unwrap();
                let admissions: Vec<_> = headers
                    .iter()
                    .map(|h| {
                        owners[0]
                            .data_admission(&peer, &grants[0], h.clone())
                            .unwrap()
                    })
                    .collect();
                let mut stream = crate::DataStream::connect(peer, &admissions[0]).unwrap();
                let mut flow = stream
                    .begin_send(&admissions, Instant::now() + Duration::from_secs(2))
                    .unwrap();
                for _ in &headers {
                    assert!(matches!(
                        flow.receive_control().unwrap(),
                        starlings_protocol::DataControl::Credit { .. }
                    ));
                }
                for _ in &headers {
                    flow.send_chunk(&vec![0x80; bytes as usize]).unwrap();
                    assert!(matches!(
                        flow.receive_control().unwrap(),
                        starlings_protocol::DataControl::ReceiptAck { .. }
                    ));
                }
                flow.finish().unwrap();
            });
            let peer = listener
                .accept()
                .unwrap()
                .stage(&self.nodes[1], &self.pins[1], grants[1].clone())
                .unwrap();
            let admissions: Vec<_> = headers
                .iter()
                .map(|h| {
                    owners[1]
                        .data_admission(&peer, &grants[1], h.clone())
                        .unwrap()
                })
                .collect();
            let mut stream = crate::DataStream::accept(peer, &admissions[0]).unwrap();
            let mut flow = stream
                .begin_receive(
                    &admissions,
                    &budget,
                    Instant::now() + Duration::from_secs(2),
                )
                .unwrap();
            for _ in &headers {
                flow.grant_credit(bytes).unwrap();
            }
            for _ in &headers {
                flow.receive_chunk().unwrap();
            }
            let received = flow.finish().unwrap();
            sender.join().unwrap();
            received
        });
        assert_eq!(received.headers(), headers);
        (received, budget)
    }
}

#[test]
fn local_owned_chunks_keep_inputs_until_qualified_drain_and_gate_actual_output_routes() {
    use std::sync::mpsc::sync_channel;
    for case in [
        "complete",
        "drop-owners",
        "full",
        "disconnected",
        "revoked",
        "wrong-lease",
        "lost-item",
    ] {
        let f = Fixture::create(|| (), true);
        let (mut c0, mut w0) = f.pair_at(0);
        let (mut c1, mut w1) = f.pair_at(1);
        let (destination, backend) = f.worker_owned(&w1, 1, 8_388_608, true);
        let backend = backend.unwrap();
        let owners = [f.worker_at(&w0, 0), destination];
        let (mut c2, mut w2) = f.pair_at(2);
        let next_owner = f.worker_at(&w2, 2);
        let other = if case == "wrong-lease" {
            let mut other = f.leases[1].clone();
            other.lease_id = uuid::Uuid::new_v4().to_string().parse().unwrap();
            for (sequence, operation) in [
                (
                    4,
                    Operation::Prepare {
                        plan: Box::new(f.plan.clone()),
                        lease: Box::new(other.clone()),
                    },
                ),
                (
                    5,
                    Operation::Commit {
                        lease_id: other.lease_id,
                    },
                ),
            ] {
                let request = ControlRequest::new(
                    uuid::Uuid::new_v4().to_string().parse().unwrap(),
                    1,
                    sequence,
                    operation,
                )
                .unwrap();
                owners[1].apply(&w1, &request).unwrap();
            }
            Some(other)
        } else {
            None
        };
        let grants = [
            f.refresh_at(
                &owners[0],
                &mut c0,
                &mut w0,
                0,
                f.template.source_stage,
                f.template.destination_stage,
            )
            .unwrap(),
            f.refresh_at(
                &owners[1],
                &mut c1,
                &mut w1,
                1,
                f.template.source_stage,
                f.template.destination_stage,
            )
            .unwrap(),
        ];
        let (input, budget) = f.received_pair(&owners, &grants);
        let scope = input.scope().clone();
        let bytes = budget.held_bytes().unwrap();
        let (tx, rx) = sync_channel(if case == "full" { 0 } else { 1 });
        if let Some(other) = other {
            // Inject a valid local lease on the same plan/route to isolate the dispatch binding.
            let original = &grants[1].0;
            let mut other_scope = original.scope.clone();
            other_scope.lease = other.lease_id;
            let other_grant = StageGrant(
                Arc::new(Delegation {
                    scope: other_scope,
                    peer: original.peer.clone(),
                    control: original.control.clone(),
                    runtime: original.runtime.clone(),
                    peer_expires: original.peer_expires,
                    gate: Mutex::new(Gate {
                        until: Instant::now() + GRANT_TIME,
                        closed: false,
                    }),
                }),
                None,
            );
            other_grant.check().unwrap();
            assert_eq!(
                owners[1].enqueue_chunk(&other_grant, &backend, input, &tx),
                Err(Error::Denied)
            );
            assert!(rx.try_recv().is_err());
            assert_eq!(budget.held_bytes().unwrap(), 0);
            assert_eq!(
                owners[1].snapshot().unwrap().generations[&scope.generation_id],
                SessionState::Admitted
            );
            continue;
        }
        if case == "revoked" {
            w1.watch().cancel();
        }
        if case == "disconnected" {
            drop(rx);
            assert_eq!(
                owners[1].enqueue_chunk(&grants[1], &backend, input, &tx),
                Err(Error::Capacity)
            );
        } else {
            let result = owners[1].enqueue_chunk(&grants[1], &backend, input, &tx);
            if case == "full" || case == "revoked" {
                assert!(result.is_err());
            } else {
                result.unwrap();
                let chunk = rx.try_recv().unwrap();
                assert!(chunk.check_dispatch().is_ok());
                assert_eq!(budget.held_bytes().unwrap(), bytes);
                if case == "drop-owners" {
                    drop(chunk);
                    drop(backend);
                    assert_eq!(budget.held_bytes().unwrap(), bytes);
                    assert_eq!(owners[1].snapshot().unwrap().reserved_bytes.0, 4_194_304);
                    assert_eq!(
                        owners[1].snapshot().unwrap().generations[&scope.generation_id],
                        SessionState::Cancelling
                    );
                    continue;
                }
                if case == "lost-item" {
                    drop(chunk);
                    assert_eq!(budget.held_bytes().unwrap(), bytes);
                    assert_eq!(owners[1].snapshot().unwrap().reserved_bytes.0, 4_194_304);
                    backend.qualified_unloaded().unwrap();
                    assert_eq!(budget.held_bytes().unwrap(), 0);
                    assert_eq!(owners[1].snapshot().unwrap().reserved_bytes.0, 0);
                    continue;
                }
                let source = BoundedU32::new(1).unwrap();
                let destination = BoundedU32::new(2).unwrap();
                let return_grants = [
                    f.refresh_at(&owners[1], &mut c1, &mut w1, 1, source, destination)
                        .unwrap(),
                    f.refresh_at(&next_owner, &mut c2, &mut w2, 2, source, destination)
                        .unwrap(),
                ];
                let listener =
                    PeerListener::bind("127.0.0.1:0".parse().unwrap(), ListenerBudget::default())
                        .unwrap();
                let (mut peer, _receiver) = thread::scope(|threads| {
                    let receive = threads.spawn(|| {
                        listener
                            .accept()
                            .unwrap()
                            .stage(&f.nodes[2], &f.pins[2], return_grants[1].clone())
                            .unwrap()
                    });
                    let peer = PeerStream::connect_stage(
                        TcpStream::connect(listener.local_addr().unwrap()).unwrap(),
                        &f.nodes[1],
                        &f.pins[1],
                        return_grants[0].clone(),
                    )
                    .unwrap();
                    (peer, receive.join().unwrap())
                });
                let mut output = chunk.input().headers()[0].clone();
                output.lease_id = f.leases[2].lease_id;
                output.source_stage = source;
                output.destination_stage = destination;
                assert!(
                    owners[1]
                        .output_admission(&peer, &grants[1], output.clone(), &chunk)
                        .is_err()
                );
                let mut wrong = output.clone();
                wrong.invocation_id = uuid::Uuid::new_v4().to_string().parse().unwrap();
                assert!(
                    owners[1]
                        .output_admission(&peer, &return_grants[0], wrong, &chunk)
                        .is_err()
                );
                let mut wrong_phase = output.clone();
                wrong_phase.phase = starlings_protocol::TensorHeaderPhase::Decode;
                let mut wrong_position = output.clone();
                wrong_position.position_start = BoundedU32::new(1).unwrap();
                let mut wrong_tokens = output.clone();
                wrong_tokens.token_count = BoundedU32::new(1).unwrap();
                wrong_tokens.shape[1] = BoundedU32::new(1).unwrap();
                wrong_tokens.total_tensor_bytes = Decimal(wrong_tokens.total_tensor_bytes.0 / 2);
                wrong_tokens.chunk_bytes = wrong_tokens.total_tensor_bytes;
                for wrong in [wrong_phase, wrong_position, wrong_tokens] {
                    assert!(
                        owners[1]
                            .output_admission(&peer, &return_grants[0], wrong, &chunk)
                            .is_err()
                    );
                }
                let admission = owners[1]
                    .output_admission(&peer, &return_grants[0], output, &chunk)
                    .unwrap();
                admission.check(&peer, None, false, || ()).unwrap();
                let guarded = admission.transport_grant().unwrap();
                // Local cancellation retains ownership but must stop all guarded I/O.
                admission.abort().unwrap();
                assert!(guarded.check().is_err());
                assert!(return_grants[0].0.gate.lock().unwrap().check().is_ok());
                assert_eq!(
                    owners[1].snapshot().unwrap().instances[&scope.lease_id],
                    starlings_core::InstanceState::Ready
                );
                assert_eq!(budget.held_bytes().unwrap(), bytes);
                chunk.qualified_drained(false).unwrap();
                let before = serde_json::to_value(owners[1].snapshot().unwrap()).unwrap();
                return_grants[0].0.gate.lock().unwrap().until =
                    Instant::now() - Duration::from_millis(1);
                // A retired ticket must not let expired grant cleanup withdraw the retained lease.
                assert!(guarded.check().is_err());
                assert!(
                    peer.admitted_frame(&admission, Instant::now() + GRANT_TIME, |io| io
                        .stage_grant()?
                        .check())
                        .is_err()
                );
                assert!(chunk.check_dispatch().is_err());
                assert!(admission.check(&peer, None, false, || ()).is_err());
                assert!(admission.abort().is_err());
                assert_eq!(
                    serde_json::to_value(owners[1].snapshot().unwrap()).unwrap(),
                    before
                );
                drop(admission);
                drop(guarded);
                assert_eq!(budget.held_bytes().unwrap(), bytes);
                drop(chunk);
            }
        }
        assert_eq!(budget.held_bytes().unwrap(), 0);
        if case != "revoked" {
            let release = ControlRequest::new(
                uuid::Uuid::new_v4().to_string().parse().unwrap(),
                1,
                4,
                Operation::Release {
                    lease_id: f.leases[1].lease_id,
                },
            )
            .unwrap();
            owners[1].apply(&w1, &release).unwrap();
        }
        assert_eq!(owners[1].snapshot().unwrap().reserved_bytes.0, 4_194_304);
        backend.qualified_unloaded().unwrap();
        assert_eq!(owners[1].snapshot().unwrap().reserved_bytes.0, 0);
    }
}

#[test]
fn route_grants_use_request_start_age_and_expired_gates_do_not_revive() {
    let start = Instant::now();
    assert_eq!(grant_deadline(start).unwrap(), start + GRANT_TIME);
    assert_eq!(grant_deadline(start - GRANT_TIME), Err(Error::Expired));
    let mut gate = Gate {
        until: start - Duration::from_millis(1),
        closed: false,
    };
    assert_eq!(gate.check(), Err(Error::Expired));
    gate.until = start + GRANT_TIME;
    assert_eq!(gate.check(), Err(Error::Expired));
}

#[test]
fn enrolled_stage_owner_renews_lease_permission_without_authority_storage_or_execution() {
    let f = Fixture::new();
    let (mut c, mut w) = f.pair();
    let owner = f.worker(&w);
    let before = owner.snapshot().unwrap();
    let grant = f.refresh(&owner, &mut c, &mut w).unwrap();
    assert_eq!(grant.peer().unwrap(), f.nodes[1].node());
    let next = f.refresh(&owner, &mut c, &mut w).unwrap();
    assert!(Arc::ptr_eq(&grant.0, &next.0));
    let directory = PrivateDir::open(f.temp.path().join("registry0")).unwrap();
    assert!(directory.open_file("authority.json").is_err());
    assert!(directory.open_file("enrollment.sqlite").is_err());
    assert_eq!(
        serde_json::to_value(before).unwrap(),
        serde_json::to_value(owner.snapshot().unwrap()).unwrap()
    );
    let start = ControlRequest::new(
        uuid::Uuid::new_v4().to_string().parse().unwrap(),
        1,
        4,
        Operation::StartStep {
            header: Box::new(f.template.clone()),
            input_digest: starlings_protocol::byte_digest(b"synthetic"),
        },
    )
    .unwrap();
    assert_eq!(owner.apply(&w, &start), Err(Error::Denied));
    grant.0.gate.lock().unwrap().until = Instant::now() - Duration::from_millis(1);
    assert!(matches!(
        owner.refresh_grant(
            &mut w,
            f.leases[0].lease_id,
            f.template.source_stage,
            f.template.destination_stage
        ),
        Err(Error::Expired)
    ));
    assert!(
        owner
            .refresh_grant(
                &mut w,
                f.leases[0].lease_id,
                f.template.source_stage,
                f.template.destination_stage
            )
            .is_err()
    );
    assert!(grant.check().is_err());
}

#[test]
fn fixture_registers_ready_workers_after_slow_preparation() {
    let f = Fixture::after_preparation(|| thread::sleep(crate::registry::REGISTRY_TTL));
    for node in &f.nodes {
        assert!(
            f.authority
                .registry_resources(node.node())
                .unwrap()
                .is_some()
        );
    }
}

#[test]
fn another_issuer_cannot_replace_the_client_owning_a_live_lease() {
    let f = Fixture::new();
    let (mut c, mut w) = f.pair();
    let owner = f.worker(&w);
    let grant = f.refresh(&owner, &mut c, &mut w).unwrap();
    let other_token = f
        .authority
        .issue_client_token(
            f.controller,
            &[Scope {
                model: f.plan.manifest_digest.clone(),
                action: Action::Infer,
            }],
            60,
        )
        .unwrap();
    let replacement = f
        .authority
        .enrollment_reader()
        .unwrap()
        .stage_issuer(&other_token, f.plan.clone(), f.leases[0].clone())
        .unwrap();
    assert_ne!(replacement.id(), f.issuers[0].id());
    f.authority.revoke_client_token(&f.token).unwrap();
    thread::scope(|threads| {
        let pending = threads.spawn(|| {
            owner.refresh_grant(
                &mut w,
                f.leases[0].lease_id,
                f.template.source_stage,
                f.template.destination_stage,
            )
        });
        assert!(replacement.respond(&mut c).is_err());
        assert!(pending.join().unwrap().is_err());
    });
    assert!(grant.check().is_err());
    assert_eq!(
        owner.snapshot().unwrap().generations[&f.session.generation_id],
        SessionState::Cancelled
    );
}

#[test]
fn any_stage_or_client_permission_withdrawal_refuses_renewal_and_cancels_the_plan() {
    for reason in ["remote-model", "remote-node", "client"] {
        let f = Fixture::new();
        let (mut c, mut w) = f.pair();
        let owner = f.worker(&w);
        let grant = f.refresh(&owner, &mut c, &mut w).unwrap();
        match reason {
            "remote-model" => f.authority.set_permissions(f.nodes[1].node(), &[]).unwrap(),
            "remote-node" => f.authority.revoke(f.nodes[1].node()).unwrap(),
            "client" => f.authority.revoke_client_token(&f.token).unwrap(),
            _ => unreachable!(),
        }
        thread::scope(|threads| {
            let pending = threads.spawn(|| {
                owner.refresh_grant(
                    &mut w,
                    f.leases[0].lease_id,
                    f.template.source_stage,
                    f.template.destination_stage,
                )
            });
            assert!(f.issuers[0].respond(&mut c).is_err());
            assert!(pending.join().unwrap().is_err());
        });
        assert!(grant.check().is_err());
        assert_eq!(
            owner.snapshot().unwrap().generations[&f.session.generation_id],
            SessionState::Cancelled
        );
    }
}

#[test]
fn a_wrong_challenge_or_oversized_reply_cannot_create_a_grant() {
    for case in ["nonce", "scope", "oversized"] {
        let f = Fixture::new();
        let (mut c, mut w) = f.pair();
        let owner = f.worker(&w);
        thread::scope(|threads| {
            let pending = threads.spawn(|| {
                owner.refresh_grant(
                    &mut w,
                    f.leases[0].lease_id,
                    f.template.source_stage,
                    f.template.destination_stage,
                )
            });
            let mut challenge: Challenge = c.receive_record_limit(GRANT_RECORD).unwrap();
            match case {
                "nonce" => challenge.nonce = uuid::Uuid::new_v4().to_string().parse().unwrap(),
                "scope" => {
                    challenge.scope.lease = uuid::Uuid::new_v4().to_string().parse().unwrap()
                }
                "oversized" => (),
                _ => unreachable!(),
            }
            if case == "oversized" {
                let _ = c.send_record(&"x".repeat(GRANT_RECORD + 1));
            } else {
                c.send_record_limit(
                    &Reply {
                        version: Version,
                        challenge,
                        peer: f.nodes[1].member.clone(),
                    },
                    GRANT_RECORD,
                )
                .unwrap();
            }
            assert!(pending.join().unwrap().is_err());
        });
        assert_eq!(
            owner.snapshot().unwrap().generations[&f.session.generation_id],
            SessionState::Cancelled
        );
    }
}

#[test]
fn permission_checks_follow_local_lease_and_owner_and_retain_inflight_work() {
    for reason in [
        "release",
        "lease-expiry",
        "owner-drop",
        "grant-expiry-inflight",
    ] {
        let f = Fixture::new();
        let (mut c, mut w) = f.pair();
        let owner = f.worker(&w);
        let grant = f.refresh(&owner, &mut c, &mut w).unwrap();
        match reason {
            "release" => {
                let request = ControlRequest::new(
                    uuid::Uuid::new_v4().to_string().parse().unwrap(),
                    1,
                    4,
                    Operation::Release {
                        lease_id: f.leases[0].lease_id,
                    },
                )
                .unwrap();
                owner.apply(&w, &request).unwrap();
                assert!(
                    grant.check().is_err(),
                    "released lease retained route permission"
                );
            }
            "lease-expiry" => {
                let mut runtime = owner.inner.lock().unwrap();
                runtime.clock = Some(runtime.clock.unwrap() - Duration::from_secs(60));
                drop(runtime);
                assert!(
                    grant.check().is_err(),
                    "expired local lease retained route permission"
                );
            }
            "owner-drop" => {
                drop(owner);
                assert!(
                    grant.check().is_err(),
                    "permission outlived its runtime owner"
                );
            }
            "grant-expiry-inflight" => {
                let mut h = f.template.clone();
                h.destination_stage = BoundedU32::new(0).unwrap();
                let request = ControlRequest::new(
                    uuid::Uuid::new_v4().to_string().parse().unwrap(),
                    1,
                    4,
                    Operation::StartStep {
                        header: Box::new(h.clone()),
                        input_digest: starlings_protocol::byte_digest(b"synthetic inflight"),
                    },
                )
                .unwrap();
                let mut runtime = owner.inner.lock().unwrap();
                let now = runtime.monotonic_ms(0);
                runtime.worker.apply(&request, now).unwrap();
                drop(runtime);
                grant.0.gate.lock().unwrap().until = Instant::now() - Duration::from_millis(1);
                assert!(grant.check().is_err());
                let snapshot = owner.snapshot().unwrap();
                assert_eq!(
                    snapshot.generations[&h.generation_id],
                    SessionState::Cancelling
                );
                assert_eq!(snapshot.reserved_bytes.0, 4_194_304);
                owner
                    .inner
                    .lock()
                    .unwrap()
                    .worker
                    .drained(h.generation_id, h.invocation_id)
                    .unwrap();
                assert_eq!(owner.snapshot().unwrap().reserved_bytes.0, 0);
            }
            _ => unreachable!(),
        }
    }
}

#[test]
fn node_local_grants_gate_both_enrolled_data_endpoints_and_local_delivery() {
    for abort in [false, true] {
        node_local_transfer(abort);
    }
}
fn node_local_transfer(abort: bool) {
    let f = Fixture::new();
    let (mut c0, mut w0) = f.pair_at(0);
    let (mut c1, mut w1) = f.pair_at(1);
    let owners = [f.worker_at(&w0, 0), f.worker_at(&w1, 1)];
    let grants = [
        f.refresh_at(
            &owners[0],
            &mut c0,
            &mut w0,
            0,
            f.template.source_stage,
            f.template.destination_stage,
        )
        .unwrap(),
        f.refresh_at(
            &owners[1],
            &mut c1,
            &mut w1,
            1,
            f.template.source_stage,
            f.template.destination_stage,
        )
        .unwrap(),
    ];
    let mut h = f.template.clone();
    h.lease_id = f.leases[1].lease_id;
    let payload = vec![0x80; h.chunk_bytes.0 as usize];
    let listener =
        PeerListener::bind("127.0.0.1:0".parse().unwrap(), ListenerBudget::default()).unwrap();
    let before = serde_json::to_value(owners[1].snapshot().unwrap()).unwrap();
    thread::scope(|threads| {
        let receiver = threads.spawn(|| {
            let peer = listener
                .accept()
                .unwrap()
                .stage(&f.nodes[1], &f.pins[1], grants[1].clone())
                .unwrap();
            assert!(peer.authority().is_err());
            assert!(
                peer.with_access(
                    &Scope {
                        model: f.plan.manifest_digest.clone(),
                        action: Action::Control
                    },
                    || ()
                )
                .is_err()
            );
            assert!(
                owners[0]
                    .data_admission(&peer, &grants[1], h.clone())
                    .is_err()
            );
            let admission = owners[1]
                .data_admission(&peer, &grants[1], h.clone())
                .unwrap();
            let mut stream = crate::DataStream::accept(peer, &admission).unwrap();
            stream
                .receive(&admission, |frame| {
                    assert_eq!(
                        frame.header(),
                        &starlings_protocol::DataHeader::Tensor(h.clone())
                    );
                    assert_eq!(frame.payload(), payload);
                })
                .unwrap();
            assert_eq!(
                serde_json::to_value(owners[1].snapshot().unwrap()).unwrap(),
                before
            );
            if abort {
                let budget = crate::ReceiveBudget::new(
                    h.total_tensor_bytes.0,
                    h.total_tensor_bytes.0,
                    h.total_tensor_bytes.0,
                )
                .unwrap();
                let flow = stream
                    .begin_receive(
                        std::slice::from_ref(&admission),
                        &budget,
                        Instant::now() + Duration::from_secs(1),
                    )
                    .unwrap();
                assert!(budget.held_bytes().unwrap() > 0);
                // Local failure must cancel without reauthorizing on the lost control.
                w1.watch().cancel();
                drop(flow);
                assert_eq!(budget.held_bytes().unwrap(), 0);
                assert_eq!(
                    owners[1].snapshot().unwrap().generations[&h.generation_id],
                    SessionState::Failed
                );
            } else {
                let request = ControlRequest::new(
                    uuid::Uuid::new_v4().to_string().parse().unwrap(),
                    1,
                    4,
                    Operation::Cancel {
                        generation_id: h.generation_id,
                    },
                )
                .unwrap();
                owners[1].apply(&w1, &request).unwrap();
            }
            assert!(
                stream
                    .receive(&admission, |_| panic!("cancelled frame delivered"))
                    .is_err()
            );
            assert!(
                stream
                    .receive(&admission, |_| panic!("poisoned stream reused"))
                    .is_err()
            );
        });
        let peer = PeerStream::connect_stage(
            TcpStream::connect(listener.local_addr().unwrap()).unwrap(),
            &f.nodes[0],
            &f.pins[0],
            grants[0].clone(),
        )
        .unwrap();
        assert!(peer.authority().is_err());
        let admission = owners[0]
            .data_admission(&peer, &grants[0], h.clone())
            .unwrap();
        let mut stream = crate::DataStream::connect(peer, &admission).unwrap();
        stream
            .send(
                &admission,
                &starlings_protocol::DataHeader::Tensor(h.clone()),
                &payload,
            )
            .unwrap();
        receiver.join().unwrap();
    });
    assert_eq!(
        owners[1].snapshot().unwrap().generations[&h.generation_id],
        if abort {
            SessionState::Failed
        } else {
            SessionState::Cancelled
        }
    );
}

#[test]
fn assignment_is_bound_to_the_configured_model_and_malformed_plan_closes_control() {
    for reason in ["model", "plan-digest", "issuer"] {
        let f = Fixture::new();
        let (mut c, mut w) = f.pair();
        let mut assignment = serde_json::to_value(Assignment {
            version: Version,
            plan: f.plan.clone(),
            issuer: f.issuers[0].id(),
        })
        .unwrap();
        match reason {
            "model" => {
                assignment["plan"]["manifest_digest"] =
                    starlings_protocol::byte_digest(b"other-model")
                        .as_str()
                        .into()
            }
            "plan-digest" => {
                assignment["plan"]["plan_digest"] =
                    starlings_protocol::byte_digest(b"not-canonical")
                        .as_str()
                        .into()
            }
            "issuer" => assignment["issuer"] = "not-a-uuid".into(),
            _ => unreachable!(),
        }
        c.send_record_limit(&assignment, GRANT_RECORD).unwrap();
        assert!(
            StageWorker::accept_assignment(
                Worker::new(
                    f.nodes[0].node(),
                    f.plan.cluster_id,
                    4_194_304,
                    1,
                    FixtureMode::Reject
                )
                .unwrap(),
                f.registry[0].fence(),
                &mut w,
            )
            .is_err()
        );
        assert!(w.watch().is_closed());
    }
}

#[test]
fn data_authority_cannot_cross_local_leases_or_outlive_permission_expiry() {
    use std::io::Write;
    for reason in ["other-lease", "expiry"] {
        let f = Fixture::new();
        let (mut c0, mut w0) = f.pair_at(0);
        let (mut c1, mut w1) = f.pair_at(1);
        let owner = f.worker_capacity(&w0, 0, 8_388_608);
        let receiver = f.worker_at(&w1, 1);
        let grant = f.refresh(&owner, &mut c0, &mut w0).unwrap();
        let remote = f
            .refresh_at(
                &receiver,
                &mut c1,
                &mut w1,
                1,
                f.template.source_stage,
                f.template.destination_stage,
            )
            .unwrap();
        let listener =
            PeerListener::bind("127.0.0.1:0".parse().unwrap(), ListenerBudget::default()).unwrap();
        let (mut peer, _remote_peer) = thread::scope(|threads| {
            let accepted = threads.spawn(|| {
                listener
                    .accept()
                    .unwrap()
                    .stage(&f.nodes[1], &f.pins[1], remote)
                    .unwrap()
            });
            let peer = PeerStream::connect_stage(
                TcpStream::connect(listener.local_addr().unwrap()).unwrap(),
                &f.nodes[0],
                &f.pins[0],
                grant.clone(),
            )
            .unwrap();
            (peer, accepted.join().unwrap())
        });
        if reason == "expiry" {
            grant.0.gate.lock().unwrap().until = Instant::now() - Duration::from_millis(1);
            assert!(peer.write_all(b"withdrawn").is_err());
        } else {
            let mut other = f.leases[0].clone();
            other.lease_id = uuid::Uuid::new_v4().to_string().parse().unwrap();
            let mut session = f.session.clone();
            session.lease_id = other.lease_id;
            session.session_id = uuid::Uuid::new_v4().to_string().parse().unwrap();
            session.generation_id = uuid::Uuid::new_v4().to_string().parse().unwrap();
            for (sequence, op) in [
                Operation::Prepare {
                    plan: Box::new(f.plan.clone()),
                    lease: Box::new(other),
                },
                Operation::Commit {
                    lease_id: session.lease_id,
                },
                Operation::Begin {
                    session: Box::new(session.clone()),
                },
            ]
            .into_iter()
            .enumerate()
            {
                let request = ControlRequest::new(
                    uuid::Uuid::new_v4().to_string().parse().unwrap(),
                    1,
                    4 + sequence as u64,
                    op,
                )
                .unwrap();
                owner.apply(&w0, &request).unwrap();
            }
            let mut h = f.template.clone();
            h.lease_id = f.leases[1].lease_id;
            h.session_id = session.session_id;
            h.generation_id = session.generation_id;
            assert!(
                owner.data_admission(&peer, &grant, h).is_err(),
                "route grant crossed to another local lease"
            );
            assert!(
                grant.check().is_ok(),
                "invalid input withdrew valid route authority"
            );
            assert_eq!(
                owner.snapshot().unwrap().generations[&f.session.generation_id],
                SessionState::Admitted
            );
            grant.0.gate.lock().unwrap().until = Instant::now() - Duration::from_millis(1);
            assert!(grant.check().is_err());
            assert_eq!(
                owner.snapshot().unwrap().generations[&session.generation_id],
                SessionState::Admitted
            );
        }
        assert!(grant.check().is_err());
        assert_eq!(
            owner.snapshot().unwrap().generations[&f.session.generation_id],
            SessionState::Cancelled
        );
    }
}

#[test]
fn renewed_authority_keeps_an_already_waiting_data_read_live() {
    use std::io::{Read, Write};
    let f = Fixture::new();
    let (mut c, mut w) = f.pair();
    let owner = f.worker(&w);
    let grant = f.refresh(&owner, &mut c, &mut w).unwrap();
    let listener =
        PeerListener::bind("127.0.0.1:0".parse().unwrap(), ListenerBudget::default()).unwrap();
    let trust = f.authority.trust().unwrap();
    let (mut peer, mut server) = thread::scope(|threads| {
        let accepted = threads.spawn(|| {
            listener
                .accept()
                .unwrap()
                .authenticate(&f.nodes[1], &trust, crate::Channel::Data)
                .unwrap()
        });
        let peer = PeerStream::connect_stage(
            TcpStream::connect(listener.local_addr().unwrap()).unwrap(),
            &f.nodes[0],
            &f.pins[0],
            grant,
        )
        .unwrap();
        (peer, accepted.join().unwrap())
    });
    thread::scope(|threads| {
        let renewal = threads.spawn(|| {
            thread::sleep(Duration::from_secs(1));
            f.refresh(&owner, &mut c, &mut w).unwrap();
        });
        let writer = threads.spawn(|| {
            thread::sleep(Duration::from_millis(2400));
            server.write_all(b"x").unwrap();
            server.flush().unwrap();
        });
        let mut byte = [0];
        assert!(
            peer.read_exact(&mut byte).is_ok(),
            "renewed authority did not extend a blocked socket wait"
        );
        assert_eq!(&byte, b"x");
        renewal.join().unwrap();
        writer.join().unwrap();
    });
    f.refresh(&owner, &mut c, &mut w).unwrap();
    let started = Instant::now();
    assert!(
        peer.frame_until(started + Duration::from_millis(100), |io| {
            io.read_exact(&mut [0])?;
            Ok(())
        })
        .is_err()
    );
    assert!(
        started.elapsed() < Duration::from_secs(1),
        "renewal extended the absolute record deadline"
    );
}

#[test]
fn the_sampled_token_returns_to_the_first_stage_as_its_next_decode_step() {
    use starlings_protocol::{
        DataControl, TensorHeaderDtype, TensorHeaderPhase, TensorHeaderTensorRole,
    };
    use std::sync::mpsc::sync_channel;
    let f = Fixture::new();
    let generation = f.session.generation_id;
    let ids = step_invocation(generation, 7).unwrap();
    assert_eq!(ids, step_invocation(generation, 7).unwrap());
    assert_ne!(ids, step_invocation(generation, 8).unwrap());
    let (mut c0, mut w0) = f.pair_at(0);
    let (mut c1, mut w1) = f.pair_at(1);
    let (first, backend0) = f.worker_owned(&w0, 0, 8_388_608, true);
    let (last, backend1) = f.worker_owned(&w1, 1, 8_388_608, true);
    let (backend0, backend1) = (backend0.unwrap(), backend1.unwrap());
    let owners = [first, last];
    let (zero, one) = (f.template.source_stage, f.template.destination_stage);
    let grants = [
        f.refresh_at(&owners[0], &mut c0, &mut w0, 0, zero, one)
            .unwrap(),
        f.refresh_at(&owners[1], &mut c1, &mut w1, 1, zero, one)
            .unwrap(),
    ];
    let returns = [
        f.refresh_at(&owners[0], &mut c0, &mut w0, 0, one, zero)
            .unwrap(),
        f.refresh_at(&owners[1], &mut c1, &mut w1, 1, one, zero)
            .unwrap(),
    ];
    let (tx, rx) = sync_channel(1);

    // The first stage starts its own prompt chunk on the token route's live grant.
    let mut prompt = f.template.clone();
    prompt.lease_id = f.leases[0].lease_id;
    prompt.source_stage = one;
    prompt.destination_stage = zero;
    prompt.tensor_role = TensorHeaderTensorRole::TokenIds;
    prompt.dtype = TensorHeaderDtype::U32;
    prompt.shape = vec![BoundedU32::new(2).unwrap()];
    prompt.total_tensor_bytes = Decimal(8);
    prompt.chunk_bytes = Decimal(8);
    prompt.invocation_id = step_invocation(generation, 0).unwrap();
    let tokens = || vec![1u8; 8].into_boxed_slice();
    let mut random = prompt.clone();
    random.invocation_id = uuid::Uuid::new_v4().to_string().parse().unwrap();
    let mut decode = prompt.clone();
    decode.phase = TensorHeaderPhase::Decode;
    for wrong in [random, decode] {
        assert_eq!(
            owners[0].enqueue_prompt(&returns[0], &backend0, wrong, tokens(), &tx),
            Err(Error::Denied)
        );
    }
    assert!(
        owners[0]
            .enqueue_prompt(&grants[0], &backend0, prompt.clone(), tokens(), &tx)
            .is_err(),
        "only the last-to-first route's grant starts first-stage input"
    );
    owners[0]
        .enqueue_prompt(&returns[0], &backend0, prompt.clone(), tokens(), &tx)
        .unwrap();
    let started = rx.try_recv().unwrap();
    started.check_dispatch().unwrap();
    assert!(
        owners[0]
            .enqueue_prompt(&returns[0], &backend0, prompt.clone(), tokens(), &tx)
            .is_err()
    );
    started.qualified_drained(false).unwrap();
    drop(started);
    let mut past = prompt.clone();
    past.sequence = Decimal(1);
    past.position_start = BoundedU32::new(2).unwrap();
    past.invocation_id = step_invocation(generation, 1).unwrap();
    assert!(
        owners[0]
            .enqueue_prompt(&returns[0], &backend0, past, tokens(), &tx)
            .is_err(),
        "no prefill past the session's prompt"
    );

    // The last stage runs the final prompt chunk; its sampled token is the next decode step.
    let (input, _budget) = f.received_pair(&owners, &grants);
    owners[1]
        .enqueue_chunk(&grants[1], &backend1, input, &tx)
        .unwrap();
    let step = rx.try_recv().unwrap();
    let mut token = prompt.clone();
    token.sequence = Decimal(1);
    token.invocation_id = step_invocation(generation, 1).unwrap();
    token.phase = TensorHeaderPhase::Decode;
    token.position_start = BoundedU32::new(2).unwrap();
    token.token_count = BoundedU32::new(1).unwrap();
    token.shape = vec![BoundedU32::new(1).unwrap()];
    token.total_tensor_bytes = Decimal(4);
    token.chunk_bytes = Decimal(4);
    let listener =
        PeerListener::bind("127.0.0.1:0".parse().unwrap(), ListenerBudget::default()).unwrap();
    let deadline = Instant::now() + Duration::from_secs(5);
    let received = thread::scope(|threads| {
        let receiver = threads.spawn(|| {
            let peer = listener
                .accept()
                .unwrap()
                .stage(&f.nodes[0], &f.pins[0], returns[0].clone())
                .unwrap();
            let admission = owners[0]
                .data_admission(&peer, &returns[0], token.clone())
                .unwrap();
            let mut stream = crate::DataStream::accept(peer, &admission).unwrap();
            let budget = crate::ReceiveBudget::new(4, 4, 4).unwrap();
            let mut flow = stream
                .begin_receive(std::slice::from_ref(&admission), &budget, deadline)
                .unwrap();
            flow.grant_credit(4).unwrap();
            flow.receive_chunk().unwrap();
            flow.finish().unwrap()
        });
        let peer = PeerStream::connect_stage(
            TcpStream::connect(listener.local_addr().unwrap()).unwrap(),
            &f.nodes[1],
            &f.pins[1],
            returns[1].clone(),
        )
        .unwrap();
        let mut same = token.clone();
        same.sequence = Decimal(0);
        let mut random = token.clone();
        random.invocation_id = uuid::Uuid::new_v4().to_string().parse().unwrap();
        let mut early = token.clone();
        early.position_start = BoundedU32::new(1).unwrap();
        let mut prefill = token.clone();
        prefill.phase = TensorHeaderPhase::Prefill;
        // GLY-360: after the prompt's last chunk the message is one token at its end.
        let rows = |n: u32, shape: u32| {
            let mut h = token.clone();
            h.token_count = BoundedU32::new(n).unwrap();
            h.shape = vec![BoundedU32::new(shape).unwrap()];
            h.total_tensor_bytes = Decimal(4 * u64::from(n));
            h.chunk_bytes = h.total_tensor_bytes;
            h
        };
        let (three, misshaped) = (rows(3, 3), rows(1, 2));
        for wrong in [same, random, early, prefill, three, misshaped] {
            assert!(
                owners[1]
                    .output_admission(&peer, &returns[1], wrong, &step)
                    .is_err()
            );
        }
        let admission = owners[1]
            .output_admission(&peer, &returns[1], token.clone(), &step)
            .unwrap();
        let mut stream = crate::DataStream::connect(peer, &admission).unwrap();
        let mut flow = stream
            .begin_send(std::slice::from_ref(&admission), deadline)
            .unwrap();
        assert!(matches!(
            flow.receive_control().unwrap(),
            DataControl::Credit { .. }
        ));
        flow.send_chunk(&7u32.to_le_bytes()).unwrap();
        assert!(matches!(
            flow.receive_control().unwrap(),
            DataControl::ReceiptAck { .. }
        ));
        flow.finish().unwrap();
        receiver.join().unwrap()
    });
    step.qualified_drained(false).unwrap();
    assert_eq!(received.tensors()[0].as_ref(), 7u32.to_le_bytes());
    // GLY-360: the first stage may run the token with its own drafts after it, kept rows only
    // from the message and at most MAX_DECODE_ROWS rows.
    let too_many = vec![9; starlings_protocol::MAX_DECODE_ROWS as usize];
    for (kept, drafts) in [(1, &[9][..]), (0, &too_many[..])] {
        assert_eq!(
            owners[0].enqueue_decode(&returns[0], &backend0, &received, kept, drafts, &tx),
            Err(Error::Denied)
        );
    }
    assert!(
        owners[0]
            .enqueue_decode(&grants[0], &backend0, &received, 0, &[9, 10], &tx)
            .is_err(),
        "only the last-to-first route's grant"
    );
    owners[0]
        .enqueue_decode(&returns[0], &backend0, &received, 0, &[9, 10], &tx)
        .unwrap();
    let drafted = rx.try_recv().unwrap();
    drafted.check_dispatch().unwrap();
    let h = &drafted.input().headers()[0];
    assert_eq!((h.position_start.get(), h.token_count.get()), (2, 3));
    let tokens: Vec<u8> = [7u32, 9, 10].iter().flat_map(|t| t.to_le_bytes()).collect();
    assert_eq!(drafted.input().tensors()[0].as_ref(), tokens);
    // The last stage's answer to that step: one token per row, for positions 3..=5. Kept 1
    // starts the next step at 4, inside the drafted step's rows; kept 3 is past them.
    drafted.qualified_drained(false).unwrap();
    let mut message = token.clone();
    message.sequence = Decimal(2);
    message.invocation_id = step_invocation(generation, 2).unwrap();
    message.position_start = BoundedU32::new(3).unwrap();
    message.token_count = BoundedU32::new(3).unwrap();
    message.shape = vec![BoundedU32::new(3).unwrap()];
    message.total_tensor_bytes = Decimal(12);
    message.chunk_bytes = Decimal(12);
    let rows = crate::ReceivedTensors::local(
        message,
        [9u32, 8, 4].iter().flat_map(|t| t.to_le_bytes()).collect(),
    );
    assert_eq!(
        owners[0].enqueue_decode(&returns[0], &backend0, &rows, 3, &[], &tx),
        Err(Error::Denied)
    );
    owners[0]
        .enqueue_decode(&returns[0], &backend0, &rows, 1, &[11], &tx)
        .unwrap();
    let next = rx.try_recv().unwrap();
    next.check_dispatch().unwrap();
    let h = &next.input().headers()[0];
    assert_eq!((h.position_start.get(), h.token_count.get()), (4, 2));
    let tokens: Vec<u8> = [8u32, 11].iter().flat_map(|t| t.to_le_bytes()).collect();
    assert_eq!(next.input().tensors()[0].as_ref(), tokens);
}

/// While a stage's step runs, it may open the receive for its next step (its credit goes out
/// with the step's output), but that step's data is read only after the running step drains.
#[test]
fn a_receive_opens_one_step_ahead_and_reads_only_after_the_drain() {
    use starlings_protocol::{
        DataControl, TensorHeaderDtype, TensorHeaderPhase, TensorHeaderTensorRole,
    };
    use std::sync::mpsc::{channel, sync_channel};
    for case in ["drain", "early", "drop"] {
        let f = Fixture::new();
        let generation = f.session.generation_id;
        let (mut c0, mut w0) = f.pair_at(0);
        let (mut c1, mut w1) = f.pair_at(1);
        let (first, backend0) = f.worker_owned(&w0, 0, 8_388_608, true);
        let (last, _) = f.worker_owned(&w1, 1, 8_388_608, true);
        let (zero, one) = (f.template.source_stage, f.template.destination_stage);
        let returns = [
            f.refresh_at(&first, &mut c0, &mut w0, 0, one, zero)
                .unwrap(),
            f.refresh_at(&last, &mut c1, &mut w1, 1, one, zero).unwrap(),
        ];
        // The first stage runs its prompt as step 0; the token is step 1.
        let mut prompt = f.template.clone();
        prompt.lease_id = f.leases[0].lease_id;
        prompt.source_stage = one;
        prompt.destination_stage = zero;
        prompt.tensor_role = TensorHeaderTensorRole::TokenIds;
        prompt.dtype = TensorHeaderDtype::U32;
        prompt.shape = vec![BoundedU32::new(2).unwrap()];
        prompt.total_tensor_bytes = Decimal(8);
        prompt.chunk_bytes = Decimal(8);
        prompt.invocation_id = step_invocation(generation, 0).unwrap();
        let (tx, rx) = sync_channel(1);
        first
            .enqueue_prompt(
                &returns[0],
                &backend0.unwrap(),
                prompt.clone(),
                vec![1u8; 8].into_boxed_slice(),
                &tx,
            )
            .unwrap();
        let running = rx.try_recv().unwrap();
        let mut token = prompt.clone();
        token.sequence = Decimal(1);
        token.invocation_id = step_invocation(generation, 1).unwrap();
        token.phase = TensorHeaderPhase::Decode;
        token.position_start = BoundedU32::new(2).unwrap();
        token.token_count = BoundedU32::new(1).unwrap();
        token.shape = vec![BoundedU32::new(1).unwrap()];
        token.total_tensor_bytes = Decimal(4);
        token.chunk_bytes = Decimal(4);
        let mut beyond = token.clone();
        beyond.sequence = Decimal(2);
        beyond.position_start = BoundedU32::new(3).unwrap();
        beyond.invocation_id = step_invocation(generation, 2).unwrap();
        let listener =
            PeerListener::bind("127.0.0.1:0".parse().unwrap(), ListenerBudget::default()).unwrap();
        let deadline = Instant::now() + Duration::from_secs(5);
        let (sent, wait) = channel();
        thread::scope(|threads| {
            let (first, listener, returns, token, f) = (&first, &listener, &returns, &token, &f);
            let receiver = threads.spawn(move || {
                let peer = listener
                    .accept()
                    .unwrap()
                    .stage(&f.nodes[0], &f.pins[0], returns[0].clone())
                    .unwrap();
                let budget = crate::ReceiveBudget::new(8, 8, 8).unwrap();
                let admission = first
                    .data_admission(&peer, &returns[0], token.clone())
                    .unwrap();
                let two = first.data_admission(&peer, &returns[0], beyond).unwrap();
                let mut stream = crate::DataStream::accept(peer, &admission).unwrap();
                // Two steps ahead stays refused.
                assert_eq!(
                    stream
                        .begin_receive(std::slice::from_ref(&two), &budget, deadline)
                        .err(),
                    Some(Error::Denied)
                );
                let mut flow = stream
                    .begin_receive(std::slice::from_ref(&admission), &budget, deadline)
                    .unwrap();
                flow.grant_credit(4).unwrap();
                wait.recv().unwrap();
                match case {
                    "drain" => {
                        running.qualified_drained(false).unwrap();
                        flow.receive_chunk().unwrap();
                        let received = flow.finish().unwrap();
                        assert_eq!(received.tensors()[0].as_ref(), 7u32.to_le_bytes());
                    }
                    "early" => assert_eq!(flow.receive_chunk(), Err(Error::Denied)),
                    _ => {
                        // Dropping the open receive aborts the generation: the running step
                        // may no longer dispatch or publish, and it still drains.
                        drop(flow);
                        assert!(running.check_dispatch().is_err());
                        running.qualified_drained(false).unwrap();
                    }
                }
            });
            let peer = PeerStream::connect_stage(
                TcpStream::connect(listener.local_addr().unwrap()).unwrap(),
                &f.nodes[1],
                &f.pins[1],
                returns[1].clone(),
            )
            .unwrap();
            let admission = last
                .data_admission(&peer, &returns[1], token.clone())
                .unwrap();
            let mut stream = crate::DataStream::connect(peer, &admission).unwrap();
            let mut flow = stream
                .begin_send(std::slice::from_ref(&admission), deadline)
                .unwrap();
            assert!(matches!(
                flow.receive_control().unwrap(),
                DataControl::Credit { .. }
            ));
            flow.send_chunk(&7u32.to_le_bytes()).unwrap();
            sent.send(()).unwrap();
            if case == "drain" {
                assert!(matches!(
                    flow.receive_control().unwrap(),
                    DataControl::ReceiptAck { .. }
                ));
                flow.finish().unwrap();
            }
            receiver.join().unwrap();
        });
    }
}

#[test]
fn stage_accept_drops_junk_and_foreign_connections_and_waits_for_the_granted_peer() {
    use std::io::Write;
    let f = Fixture::new();
    let (mut c0, mut w0) = f.pair_at(0);
    let (mut c1, mut w1) = f.pair_at(1);
    let owners = [f.worker_at(&w0, 0), f.worker_at(&w1, 1)];
    let (zero, one) = (f.template.source_stage, f.template.destination_stage);
    let grants = [
        f.refresh_at(&owners[0], &mut c0, &mut w0, 0, zero, one)
            .unwrap(),
        f.refresh_at(&owners[1], &mut c1, &mut w1, 1, zero, one)
            .unwrap(),
    ];
    let listener =
        PeerListener::bind("127.0.0.1:0".parse().unwrap(), ListenerBudget::default()).unwrap();
    let address = listener.local_addr().unwrap();
    // Junk from the peer's own address arrives first: it fails the handshake and is dropped.
    let mut junk = TcpStream::connect(address).unwrap();
    junk.write_all(b"not a TLS client hello").unwrap();
    drop(junk);
    let silent = TcpStream::connect(address).unwrap();
    thread::scope(|threads| {
        let peer = threads.spawn(|| {
            thread::sleep(Duration::from_millis(50));
            PeerStream::connect_stage(
                TcpStream::connect(address).unwrap(),
                &f.nodes[0],
                &f.pins[0],
                grants[0].clone(),
            )
            .unwrap()
        });
        let (stream, refused) = listener
            .accept_stage(
                &f.nodes[1],
                &f.pins[1],
                &grants[1],
                address.ip(),
                Instant::now() + Duration::from_secs(5),
            )
            .unwrap();
        assert_eq!(stream.peer(), f.nodes[0].node());
        assert!(refused >= 2, "{refused}");
        drop(peer.join().unwrap());
    });
    drop(silent);
    // A connection from any other address is refused before its handshake.
    let _waiting = TcpStream::connect(address).unwrap();
    assert_eq!(
        listener
            .accept_stage(
                &f.nodes[1],
                &f.pins[1],
                &grants[1],
                "192.0.2.1".parse().unwrap(),
                Instant::now() + Duration::from_millis(300),
            )
            .err(),
        Some(Error::Expired)
    );
}

#[test]
fn only_a_head_step_returns_a_token() {
    use starlings_protocol::{TensorHeaderDtype, TensorHeaderPhase, TensorHeaderTensorRole};
    use std::sync::mpsc::sync_channel;
    let mut f = Fixture::new();
    // A four-token prompt: the received chunk (positions 0..2) does not end it.
    f.session.prompt_token_count = BoundedU32::new(4).unwrap();
    let (mut c0, mut w0) = f.pair_at(0);
    let (mut c1, mut w1) = f.pair_at(1);
    let first = f.worker_at(&w0, 0);
    let (last, backend) = f.worker_owned(&w1, 1, 8_388_608, true);
    let owners = [first, last];
    let (zero, one) = (f.template.source_stage, f.template.destination_stage);
    let grants = [
        f.refresh_at(&owners[0], &mut c0, &mut w0, 0, zero, one)
            .unwrap(),
        f.refresh_at(&owners[1], &mut c1, &mut w1, 1, zero, one)
            .unwrap(),
    ];
    let returns = [
        f.refresh_at(&owners[0], &mut c0, &mut w0, 0, one, zero)
            .unwrap(),
        f.refresh_at(&owners[1], &mut c1, &mut w1, 1, one, zero)
            .unwrap(),
    ];
    let (input, _budget) = f.received_pair(&owners, &grants);
    let (tx, rx) = sync_channel(1);
    owners[1]
        .enqueue_chunk(&grants[1], &backend.unwrap(), input, &tx)
        .unwrap();
    let step = rx.try_recv().unwrap();
    let mut token = f.template.clone();
    token.lease_id = f.leases[0].lease_id;
    token.source_stage = one;
    token.destination_stage = zero;
    token.tensor_role = TensorHeaderTensorRole::TokenIds;
    token.dtype = TensorHeaderDtype::U32;
    token.phase = TensorHeaderPhase::Decode;
    token.sequence = Decimal(1);
    token.invocation_id = step_invocation(f.session.generation_id, 1).unwrap();
    token.position_start = BoundedU32::new(2).unwrap();
    token.token_count = BoundedU32::new(1).unwrap();
    token.shape = vec![BoundedU32::new(1).unwrap()];
    token.total_tensor_bytes = Decimal(4);
    token.chunk_bytes = Decimal(4);
    let listener =
        PeerListener::bind("127.0.0.1:0".parse().unwrap(), ListenerBudget::default()).unwrap();
    thread::scope(|threads| {
        let receiver = threads.spawn(|| {
            listener
                .accept()
                .unwrap()
                .stage(&f.nodes[0], &f.pins[0], returns[0].clone())
                .unwrap()
        });
        let peer = PeerStream::connect_stage(
            TcpStream::connect(listener.local_addr().unwrap()).unwrap(),
            &f.nodes[1],
            &f.pins[1],
            returns[1].clone(),
        )
        .unwrap();
        assert!(
            owners[1]
                .output_admission(&peer, &returns[1], token, &step)
                .is_err(),
            "a prompt chunk that does not end the prompt samples nothing"
        );
        drop(receiver.join().unwrap());
    });
}

/// GLY-190: one stage runtime holds two requests' plans; neither can drive or hold the other's,
/// a third waits for a slot, and one ending releases only its own.
#[test]
fn one_runtime_keeps_two_requests_plans_apart() {
    let f = Fixture::new();
    let worker = Worker::new(
        f.nodes[0].node(),
        f.plan.cluster_id,
        // Two requests' 4 MiB reservations.
        2 * 4_194_304,
        1,
        FixtureMode::Reject,
    )
    .unwrap();
    let runtime = StageRuntime::new(worker, f.registry[0].fence(), 2).unwrap();
    let mut plans = vec![(f.plan.clone(), f.leases[0].clone(), f.issuers[0].id())];
    let mut issuers = Vec::new();
    for _ in 0..2 {
        let mut plan = f.plan.clone();
        plan.plan_id = uuid::Uuid::new_v4().to_string().parse().unwrap();
        plan.plan_digest = starlings_protocol::canonical_digest(
            &serde_json::to_value(&plan).unwrap(),
            "plan_digest",
        )
        .unwrap();
        let mut lease = f.leases[0].clone();
        lease.lease_id = uuid::Uuid::new_v4().to_string().parse().unwrap();
        lease.plan_digest = plan.plan_digest.clone();
        let issuer = f
            .authority
            .enrollment_reader()
            .unwrap()
            .stage_issuer(&f.token, plan.clone(), lease.clone())
            .unwrap();
        plans.push((plan, lease, issuer.id()));
        issuers.push(issuer);
    }
    let mut sequence = 0;
    let mut request = |operation| {
        sequence += 1;
        ControlRequest::new(
            uuid::Uuid::new_v4().to_string().parse().unwrap(),
            1,
            sequence,
            operation,
        )
        .unwrap()
    };
    let prepare = |(plan, lease, _): &(PlacementPlan, Lease, RequestId)| Operation::Prepare {
        plan: Box::new(plan.clone()),
        lease: Box::new(lease.clone()),
    };
    let planned = |i: usize| StageWorker::planned(&runtime, plans[i].0.clone(), plans[i].2);
    let streams: Vec<_> = (0..3).map(|_| f.pair()).collect();
    let [a, b, c] = [&streams[0].1, &streams[1].1, &streams[2].1];
    let (first, second, third) = (
        planned(0).unwrap(),
        planned(1).unwrap(),
        planned(2).unwrap(),
    );
    assert!(!first.apply(a, &request(prepare(&plans[0]))).unwrap());
    assert!(!second.apply(b, &request(prepare(&plans[1]))).unwrap());
    // A live plan is not assigned again.
    assert!(planned(0).is_err());
    // Neither drives nor holds the other's plan, on its own channel or the other's.
    assert!(matches!(
        first.apply(a, &request(prepare(&plans[1]))),
        Err(Error::Denied)
    ));
    let commit = |i: usize| Operation::Commit {
        lease_id: plans[i].1.lease_id,
    };
    assert!(matches!(
        second.apply(a, &request(commit(1))),
        Err(Error::Denied)
    ));
    assert!(matches!(
        first.hold_backend(a, plans[1].1.lease_id),
        Err(Error::Denied)
    ));
    // Two slots: a third plan waits.
    assert!(matches!(
        third.apply(c, &request(prepare(&plans[2]))),
        Err(Error::Capacity)
    ));
    // The second ending frees its slot and leaves the first's plan live.
    drop(second);
    assert!(!third.apply(c, &request(prepare(&plans[2]))).unwrap());
    assert!(!first.apply(a, &request(commit(0))).unwrap());
}
