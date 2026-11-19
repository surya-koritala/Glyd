use super::*;
use crate::{
    Authority, ListenerBudget, PeerListener, PrivateDir, RegistryClient, RegistryRequest, Scope,
    enrollment::tests::{enrolled, setup},
};
use std::thread;

#[test]
fn generation_first_refusal_diagnostics_are_opt_in_and_do_not_revive_channels() {
    const CASE: &str = "STARLINGS_TEST_GENERATION_REFUSAL_CHILD";
    const TRACE: &str = "STARLINGS_TEST_GENERATION_GATE_TRACE";
    if std::env::var_os(CASE).is_none() {
        for trace in [false, true] {
            let mut child = std::process::Command::new(std::env::current_exe().unwrap());
            child.args([
                "--exact",
                "generation::tests::generation_first_refusal_diagnostics_are_opt_in_and_do_not_revive_channels",
                "--nocapture",
            ]).env(CASE, "1").env_remove(TRACE);
            if trace {
                child.env(TRACE, "1");
            }
            let output = child.output().unwrap();
            assert!(
                output.status.success(),
                "{}",
                String::from_utf8_lossy(&output.stderr)
            );
            let expected = if trace && cfg!(debug_assertions) {
                "GENERATION_GATE_REFUSED side=worker phase=lifetime reason=Expired\n\
                 GENERATION_GATE_REFUSED side=worker phase=wall_rollback reason=Expired\n\
                 GENERATION_GATE_REFUSED side=worker phase=certificate reason=Expired\n\
                 GENERATION_GATE_REFUSED side=worker phase=worker_registry reason=Expired\n\
                 GENERATION_GATE_REFUSED side=coordinator phase=coordinator_registry reason=Expired\n"
            } else {
                ""
            };
            assert_eq!(
                String::from_utf8(output.stderr)
                    .unwrap()
                    .replace("\r\n", "\n"),
                expected
            );
        }
        return;
    }
    for failure in [
        "lifetime",
        "wall_rollback",
        "certificate",
        "worker_registry",
    ] {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let _peer = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
        let node = "00000000-0000-4000-8000-000000000001".parse().unwrap();
        let cluster = "00000000-0000-4000-8000-000000000002".parse().unwrap();
        let fence = RegistryFence::test_online(node, cluster);
        let mut guard = Guard {
            binding: GenerationBinding {
                version: Version,
                epoch: fence.status().unwrap().0,
                model: starlings_protocol::byte_digest(b"synthetic guard control"),
                profile: "synthetic-profile".into(),
            },
            authorization: Authorization::Worker(fence.clone()),
            socket: listener.accept().unwrap().0,
            closed: AtomicBool::new(false),
            deadline: Instant::now() + LIFETIME,
            certificate_expires: crate::now().unwrap() + 3600,
            wall_high: Mutex::new(crate::now().unwrap()),
        };
        guard.check().unwrap();
        match failure {
            "lifetime" => guard.deadline = Instant::now(),
            "wall_rollback" => *guard.wall_high.lock().unwrap() += 3600,
            "certificate" => guard.certificate_expires = crate::now().unwrap(),
            _ => fence.test_transition("registry-offline"),
        }
        assert_eq!(guard.check(), Err(Error::Expired));
        assert!(guard.closed.load(Ordering::Acquire));
        // Restoring local limits cannot reopen a channel or replace the first refusal.
        guard.deadline = Instant::now() + LIFETIME;
        guard.certificate_expires = crate::now().unwrap() + 3600;
        *guard.wall_high.lock().unwrap() = crate::now().unwrap();
        assert_eq!(guard.check(), Err(Error::Expired));
    }
    // Use a real enrolled TLS pair for the coordinator's current-registry check.
    let f = Fixture::new();
    let (mut coordinator, _worker) = f.pair();
    Arc::get_mut(&mut coordinator.guard)
        .unwrap()
        .binding
        .epoch
        .number
        .0 += 1;
    assert_eq!(coordinator.watch().check(), Err(Error::Expired));
    assert_eq!(coordinator.send_record(&true), Err(Error::Expired));
}

struct Fixture {
    _temp: tempfile::TempDir,
    authority: Authority,
    worker: Identity,
    invitation: Invitation,
    registry: RegistryClient,
    binding: GenerationBinding,
}
impl Fixture {
    fn new() -> Self {
        let (temp, path, mut authority, files) = setup();
        let worker = enrolled(&mut authority, &files, Role::Worker, "member");
        worker.save(&files, "identity.json").unwrap();
        let listener =
            PeerListener::bind("127.0.0.1:0".parse().unwrap(), ListenerBudget::default()).unwrap();
        authority
            .invite(
                &files,
                "pin.json",
                listener.local_addr().unwrap(),
                Role::Worker,
            )
            .unwrap();
        let invitation = Invitation::load(&files, "pin.json").unwrap();
        let mut registry = RegistryClient::open(
            &path.parent().unwrap().join("device"),
            worker,
            Invitation::load(&files, "pin.json").unwrap(),
        )
        .unwrap();
        authority.start_registry().unwrap();
        for _ in 0..2 {
            thread::scope(|threads| {
                let reader = authority.enrollment_reader().unwrap();
                let server =
                    threads.spawn(
                        || match listener.accept().unwrap().registry(reader).unwrap() {
                            RegistryRequest::Heartbeat(pending) => pending.respond().unwrap(),
                            _ => panic!("wrong protocol"),
                        },
                    );
                registry.heartbeat(None).unwrap();
                server.join().unwrap();
            });
        }
        let worker = Identity::load(&files, "identity.json").unwrap();
        let binding = GenerationBinding {
            version: Version,
            epoch: authority.registry_epoch().unwrap(),
            model: starlings_protocol::byte_digest(b"synthetic-generation-model"),
            profile: "synthetic-profile".into(),
        };
        authority
            .set_permissions(
                authority.identity().node(),
                &[Scope {
                    model: binding.model.clone(),
                    action: Action::Control,
                }],
            )
            .unwrap();
        authority
            .set_permissions(
                worker.node(),
                &[Scope {
                    model: binding.model.clone(),
                    action: Action::Infer,
                }],
            )
            .unwrap();
        Self {
            _temp: temp,
            authority,
            worker,
            invitation,
            registry,
            binding,
        }
    }
    fn pair(&self) -> (GenerationStream, GenerationStream) {
        let (client, server) = self.try_pair(self.binding.clone());
        (client.unwrap(), server.unwrap())
    }
    fn try_pair(
        &self,
        requested: GenerationBinding,
    ) -> (Result<GenerationStream>, Result<GenerationStream>) {
        let listener =
            PeerListener::bind("127.0.0.1:0".parse().unwrap(), ListenerBudget::default()).unwrap();
        let address = listener.local_addr().unwrap();
        let local = &self.worker;
        let invitation = &self.invitation;
        let fence = self.registry.fence();
        let binding = self.binding.clone();
        thread::scope(|threads| {
            let server = threads.spawn(|| {
                listener
                    .accept()
                    .unwrap()
                    .generation(local, invitation, fence, binding)
            });
            let client = GenerationStream::connect(
                TcpStream::connect(address).unwrap(),
                self.authority.identity(),
                &self.authority.trust().unwrap(),
                self.worker.node(),
                requested,
            );
            (client, server.join().unwrap())
        })
    }
}

#[test]
fn enrolled_generation_uses_only_worker_identity_and_preserves_record_boundaries() {
    let f = Fixture::new();
    // Worker storage has an identity and fence, not the authority signing key/database.
    let files = PrivateDir::open(f._temp.path().join("device")).unwrap();
    assert!(files.open_file("authority.json").is_err());
    assert!(files.open_file("enrollment.sqlite").is_err());
    let (mut coordinator, mut worker) = f.pair();
    let record = serde_json::json!({"payload":"a".repeat(100_000)});
    thread::scope(|threads| {
        let sender = threads.spawn(|| coordinator.send_record(&record));
        assert_eq!(
            worker.receive_record::<serde_json::Value>().unwrap(),
            record
        );
        sender.join().unwrap().unwrap();
    });
    worker
        .send_record(&serde_json::json!({"complete":true}))
        .unwrap();
    assert_eq!(
        coordinator.receive_record::<serde_json::Value>().unwrap(),
        serde_json::json!({"complete":true})
    );
    let watch = worker.watch();
    drop(worker);
    assert!(watch.check().is_err());
    assert!(coordinator.receive_record::<serde_json::Value>().is_err());
}

#[test]
fn coordinator_key_pin_accepts_renewal_but_not_a_new_name_or_key() {
    let f = Fixture::new();
    let local = f.authority.identity();
    let mut params = crate::enrollment::certificate_params(&local.member).unwrap();
    params.serial_number = Some(42u64.into());
    let renewed = params.signed_by(&local.key, &f.authority.issuer).unwrap();
    assert_ne!(renewed.der().as_ref(), f.invitation.coordinator.certificate);
    verify_coordinator(renewed.der(), &f.invitation).unwrap();
    let other_key = rcgen::KeyPair::generate().unwrap();
    let wrong_key = params.signed_by(&other_key, &f.authority.issuer).unwrap();
    assert!(verify_coordinator(wrong_key.der(), &f.invitation).is_err());
    params.subject_alt_names = vec![rcgen::SanType::DnsName(
        "wrong.glyd.invalid".try_into().unwrap(),
    )];
    let wrong_name = params.signed_by(&local.key, &f.authority.issuer).unwrap();
    assert!(verify_coordinator(wrong_name.der(), &f.invitation).is_err());
}

#[test]
fn generation_refuses_wrong_binding_role_and_unapproved_model() {
    let mut f = Fixture::new();
    for change in ["profile", "model", "epoch"] {
        let mut requested = f.binding.clone();
        match change {
            "profile" => requested.profile = "another-profile".into(),
            "model" => requested.model = starlings_protocol::byte_digest(b"other-model"),
            _ => requested.epoch.incarnation = uuid::Uuid::new_v4(),
        }
        let (client, server) = f.try_pair(requested);
        assert!(client.is_err() && server.is_err());
    }
    let (_temp, _path, _authority, files) = setup();
    let controller = enrolled(&mut f.authority, &files, Role::Controller, "controller");
    assert_eq!(
        verify_coordinator(&controller.member.certificate, &f.invitation),
        Err(Error::Denied)
    );
    f.authority.set_permissions(f.worker.node(), &[]).unwrap();
    assert!(f.try_pair(f.binding.clone()).0.is_err());
}

#[test]
fn generation_fences_permissions_epochs_revocation_and_drain() {
    for transition in [
        "registry-epoch",
        "registry-revoked",
        "registry-draining",
        "registry-offline",
        "permission",
        "revoked",
        "drain",
    ] {
        let f = Fixture::new();
        let (mut coordinator, mut worker) = f.pair();
        let watch = worker.watch();
        match transition {
            "permission" => f.authority.set_permissions(f.worker.node(), &[]).unwrap(),
            "revoked" => f.authority.revoke(f.worker.node()).unwrap(),
            "drain" => {
                f.authority.drain().unwrap();
            }
            _ => f.registry.fence().test_transition(transition),
        }
        if transition.starts_with("registry-") {
            assert!(watch.check().is_err());
            assert!(worker.send_record(&true).is_err());
        } else {
            assert!(coordinator.watch().check().is_err());
            assert!(coordinator.send_record(&true).is_err());
        }
        // A closed handle never revives, even after restoring model permission.
        assert!(coordinator.receive_record::<bool>().is_err());
    }
}

#[test]
fn generation_rejects_bad_frames_and_interrupts_waiting_reads() {
    for bytes in [b"{\"x\":1,\"x\":2}".as_slice(), b"{\"bad\":".as_slice()] {
        let f = Fixture::new();
        let (mut coordinator, mut worker) = f.pair();
        coordinator
            .wire
            .write_all(&(bytes.len() as u32).to_be_bytes())
            .unwrap();
        coordinator.wire.write_all(bytes).unwrap();
        coordinator.wire.flush().unwrap();
        assert!(worker.receive_record::<serde_json::Value>().is_err());
        assert!(worker.watch().check().is_err());
    }
    let f = Fixture::new();
    let (mut coordinator, mut worker) = f.pair();
    coordinator.wire.write_all(&u32::MAX.to_be_bytes()).unwrap();
    coordinator.wire.flush().unwrap();
    assert!(matches!(
        worker.receive_record::<serde_json::Value>(),
        Err(Error::Capacity)
    ));
    let (_coordinator, mut worker) = f.pair();
    let watch = worker.watch();
    let started = Instant::now();
    thread::scope(|threads| {
        let reader = threads.spawn(|| worker.receive_record::<bool>());
        watch.cancel();
        assert!(reader.join().unwrap().is_err());
    });
    assert!(started.elapsed() < Duration::from_secs(1));
    let (_coordinator, mut worker) = f.pair();
    worker.remaining_records = 0;
    assert_eq!(worker.send_record(&true), Err(Error::Capacity));
    assert!(worker.watch().check().is_err());
}

#[test]
fn generation_deadlines_and_byte_budget_cannot_be_extended_by_a_new_record() {
    let f = Fixture::new();
    let (_coordinator, mut worker) = f.pair();
    Arc::get_mut(&mut worker.guard).unwrap().deadline = Instant::now() + Duration::from_millis(50);
    let start = Instant::now();
    assert!(worker.receive_record::<bool>().is_err());
    assert!(start.elapsed() < Duration::from_secs(1));
    assert!(worker.send_record(&true).is_err());
    let (_coordinator, mut worker) = f.pair();
    worker.remaining_bytes = 4;
    assert_eq!(worker.send_record(&true), Err(Error::Capacity));
    assert!(worker.watch().check().is_err());
}

fn lifecycle(
    f: &Fixture,
) -> (
    crate::EnrolledWorker,
    starlings_protocol::PlacementPlan,
    starlings_protocol::Lease,
    starlings_protocol::Session,
) {
    use starlings_protocol::*;
    let mut plan: PlacementPlan = decode(
        include_bytes!("../../../../examples/placement-plan.json"),
        FixtureMode::Simulation,
    )
    .unwrap();
    plan.fixture = false;
    plan.mode = PlacementPlanMode::WholeRequest;
    plan.cluster_id = f.worker.cluster();
    plan.authority_epoch = f.binding.epoch.number;
    plan.manifest_digest = f.binding.model.clone();
    plan.execution_profile = f.binding.profile.clone();
    plan.stages.truncate(1);
    let stage = &mut plan.stages[0];
    stage.node_id = f.worker.node();
    stage.layer_end_exclusive = plan.model_layer_count;
    stage.owns_final_norm_and_head = true;
    stage.backend_build_digest = byte_digest(b"synthetic backend build");
    let profile = starlings_core::routing::WarmReplica {
        node: f.worker.node(),
        manifest: plan.manifest_digest.clone(),
        profile: plan.execution_profile.clone(),
        backend: stage.backend_id.clone(),
        build: stage.backend_build_digest.clone(),
        pool: stage.memory_pool_id.clone(),
        evidence: byte_digest(b"synthetic lifecycle qualification"),
        request_bytes: stage.reserved_bytes().unwrap(),
        measured_request_ms: 1,
        context_limit: plan.context_limit.get(),
        max_sessions: 1,
        accepting: true,
    };
    plan.plan_digest =
        canonical_digest(&serde_json::to_value(&plan).unwrap(), "plan_digest").unwrap();
    let mut lease: Lease = decode(
        include_bytes!("../../../../examples/lease.json"),
        FixtureMode::Simulation,
    )
    .unwrap();
    lease.plan_digest = plan.plan_digest.clone();
    lease.cluster_id = plan.cluster_id;
    lease.owner_node_id = f.worker.node();
    lease.authority_epoch = plan.authority_epoch;
    let mut session: Session = decode(
        include_bytes!("../../../../examples/session.json"),
        FixtureMode::Simulation,
    )
    .unwrap();
    session.plan_digest = plan.plan_digest.clone();
    session.authority_epoch = plan.authority_epoch;
    let worker = starlings_core::Worker::new(
        f.worker.node(),
        f.worker.cluster(),
        2 * profile.request_bytes,
        f.binding.epoch.number.0,
        FixtureMode::Reject,
    )
    .unwrap();
    let worker = crate::EnrolledWorker::new(worker, f.registry.fence(), profile).unwrap();
    (worker, plan, lease, session)
}

#[test]
fn enrolled_worker_applies_whole_requests_once_and_retains_buffers_until_local_drain() {
    use starlings_protocol::*;
    for failure in ["disconnect", "epoch", "expiry", "success"] {
        writeln!(
            std::io::stderr().lock(),
            "ENROLLED_LIFECYCLE_CASE: {failure}"
        )
        .unwrap();
        let f = Fixture::new();
        let (_coordinator, stream) = f.pair();
        let (worker, plan, lease, session) = lifecycle(&f);
        let invocation = "00000000-0000-4000-8000-000000000999".parse().unwrap();
        let request = |sequence, operation| {
            ControlRequest::new(
                uuid::Uuid::new_v4().to_string().parse().unwrap(),
                f.binding.epoch.number.0,
                sequence,
                operation,
            )
            .unwrap()
        };
        worker
            .apply(
                &stream,
                &request(
                    1,
                    Operation::Prepare {
                        plan: Box::new(plan.clone()),
                        lease: Box::new(lease.clone()),
                    },
                ),
            )
            .unwrap();
        worker
            .apply(
                &stream,
                &request(
                    2,
                    Operation::Commit {
                        lease_id: lease.lease_id,
                    },
                ),
            )
            .unwrap();
        let start = request(
            3,
            Operation::StartRequest {
                session: Box::new(session.clone()),
                invocation_id: invocation,
                request: Box::new(ChatRequest {
                    model: plan.manifest_digest.to_string(),
                    messages: vec![ChatMessage {
                        role: ChatMessageRole::User,
                        content: "synthetic input".into(),
                    }],
                    max_tokens: session.max_output_tokens,
                    temperature: Some(session.sampling.temperature),
                    top_p: Some(session.sampling.top_p),
                    seed: Some(session.sampling.seed),
                    stream: Some(true),
                }),
            },
        );
        assert_eq!(worker.apply(&stream, &start), Ok(false));
        assert_eq!(worker.apply(&stream, &start), Ok(true));
        let (_replacement_coordinator, replacement) = f.pair();
        assert_eq!(
            worker.apply(&replacement, &start),
            Err(crate::Error::Denied)
        );
        assert_eq!(worker.withdraw(&replacement), Err(crate::Error::Denied));
        // Concurrent timer samples cannot rewind or prematurely advance a live worker.
        f.registry.fence().maintain(u64::MAX).unwrap();
        f.registry.fence().maintain(0).unwrap();
        assert!(
            worker
                .with_output(&replacement, session.generation_id, || ())
                .is_err()
        );
        assert!(
            worker
                .with_output(&stream, session.generation_id, || ())
                .is_ok()
        );
        assert_eq!(
            worker.apply(
                &stream,
                &request(
                    4,
                    Operation::CompleteStep {
                        generation_id: session.generation_id,
                        invocation_id: invocation
                    }
                ),
            ),
            Err(crate::Error::Denied)
        );
        match failure {
            "disconnect" => {
                stream.watch().cancel();
                f.registry.fence().maintain(1).unwrap();
            }
            "epoch" => {
                f.registry.fence().test_transition("registry-epoch");
                f.registry.fence().maintain(1).unwrap();
            }
            "expiry" => {
                worker.advance_clock(Duration::from_secs(15));
                f.registry.fence().maintain(15_000).unwrap();
            }
            _ => (),
        }
        assert_eq!(
            worker.snapshot().unwrap().reserved_bytes,
            lease.reserved_bytes
        );
        if failure != "success" {
            assert!(
                worker
                    .with_output(&stream, session.generation_id, || ())
                    .is_err()
            );
            worker
                .failed_and_drained(session.generation_id, invocation)
                .unwrap();
            assert_eq!(worker.snapshot().unwrap().reserved_bytes, Decimal(0));
        } else {
            worker.drained(session.generation_id, invocation).unwrap();
            worker
                .apply(
                    &stream,
                    &request(
                        5,
                        Operation::Release {
                            lease_id: lease.lease_id,
                        },
                    ),
                )
                .unwrap();
            assert_eq!(worker.snapshot().unwrap().reserved_bytes, Decimal(0));
        }
    }
}

#[test]
fn enrolled_worker_rejects_unqualified_plans_and_second_lease() {
    use starlings_protocol::*;
    let f = Fixture::new();
    let (coordinator, stream) = f.pair();
    let (worker, plan, lease, _) = lifecycle(&f);
    let prepare = |mut plan: PlacementPlan, mut lease: Lease, sequence| {
        plan.plan_digest =
            canonical_digest(&serde_json::to_value(&plan).unwrap(), "plan_digest").unwrap();
        lease.plan_digest = plan.plan_digest.clone();
        ControlRequest::new(
            uuid::Uuid::new_v4().to_string().parse().unwrap(),
            f.binding.epoch.number.0,
            sequence,
            Operation::Prepare {
                plan: Box::new(plan),
                lease: Box::new(lease),
            },
        )
        .unwrap()
    };
    for change in ["profile", "build", "pool", "memory", "context", "fixture"] {
        let mut bad = plan.clone();
        match change {
            "profile" => bad.execution_profile = "another-profile".into(),
            "build" => bad.stages[0].backend_build_digest = byte_digest(b"unqualified build"),
            "pool" => bad.stages[0].memory_pool_id = "unqualified-pool".into(),
            "memory" => bad.stages[0].resident_bytes = Decimal(0),
            "context" => bad.context_limit = BoundedU32::new(4096).unwrap(),
            _ => bad.fixture = true,
        }
        assert!(
            worker
                .apply(&stream, &prepare(bad, lease.clone(), 1))
                .is_err()
        );
        assert_eq!(worker.snapshot().unwrap().reserved_bytes, Decimal(0));
    }
    assert!(
        worker
            .apply(&coordinator, &prepare(plan.clone(), lease.clone(), 1))
            .is_err()
    );
    worker
        .apply(&stream, &prepare(plan.clone(), lease.clone(), 1))
        .unwrap();
    let mut second = lease.clone();
    second.lease_id = uuid::Uuid::new_v4().to_string().parse().unwrap();
    assert_eq!(
        worker.apply(&stream, &prepare(plan, second, 2)),
        Err(crate::Error::Capacity)
    );
    assert_eq!(
        worker.snapshot().unwrap().reserved_bytes,
        lease.reserved_bytes
    );
}

// Compare the native small-record policy without imposing a flaky timing threshold.
#[test]
fn generation_small_record_tcp_comparison() {
    for trial in 0..5 {
        let f = Fixture::new();
        for nodelay in if trial % 2 == 0 {
            [false, true]
        } else {
            [true, false]
        } {
            let (mut coordinator, mut worker) = f.pair();
            for stream in [&coordinator, &worker] {
                let socket = stream.wire.socket().unwrap();
                socket.set_nodelay(nodelay).unwrap();
                assert_eq!(socket.nodelay().unwrap(), nodelay);
            }
            let ready = std::sync::Barrier::new(2);
            let elapsed = thread::scope(|threads| {
                let server = threads.spawn(|| {
                    ready.wait();
                    for sequence in 0u32..4 {
                        assert_eq!(worker.receive_record::<u32>().unwrap(), sequence);
                        worker.send_record(&sequence).unwrap();
                    }
                });
                ready.wait();
                let began = Instant::now();
                for sequence in 0u32..4 {
                    coordinator.send_record(&sequence).unwrap();
                    assert_eq!(coordinator.receive_record::<u32>().unwrap(), sequence);
                }
                let elapsed = began.elapsed().as_micros();
                server.join().unwrap();
                elapsed
            });
            // Use the raw diagnostic writer so passing CI retains these safe measurements.
            writeln!(std::io::stderr(), "CONTROLLED_TCP os={} arch={} trial={trial} nodelay={nodelay} exchanges=4 us={elapsed}", std::env::consts::OS, std::env::consts::ARCH).unwrap();
        }
    }
}
