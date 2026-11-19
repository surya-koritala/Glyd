//! Both data endpoints use StageWorker, enrolled TLS and live local admission.
use super::*;
use crate::{
    DataStream, GenerationBinding, Identity, Invitation, ListenerBudget, PeerListener, PrivateDir,
    RegistryClient, RegistryRequest,
};
use starlings_protocol::{
    DataControl, DataHeader, Decimal, Lease, NodeCapabilities, Session, TensorHeader,
};
use std::{
    io::Write,
    net::TcpStream,
    path::Path,
    process::{Child, Command, Stdio},
    thread,
};

const CHILD_ENV: &str = "STARLINGS_TEST_STAGE_DEVICE";
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct DeviceConfig {
    index: usize,
    binding: GenerationBinding,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Ready {
    control: std::net::SocketAddr,
    data: std::net::SocketAddr,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Routes {
    data: [std::net::SocketAddr; 2],
    leases: [LeaseId; 2],
    session: Session,
}
struct ChildGuard(Child);
impl Drop for ChildGuard {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}
fn uuid<T: std::str::FromStr>() -> T
where
    T::Err: std::fmt::Debug,
{
    uuid::Uuid::new_v4().to_string().parse().unwrap()
}
fn sample(identity: &Identity, sequence: u64) -> crate::ResourceSample {
    let mut caps: NodeCapabilities = starlings_protocol::decode(
        include_bytes!("../../../../../examples/node-capabilities.json"),
        FixtureMode::Simulation,
    )
    .unwrap();
    caps.fixture = false;
    caps.node_id = identity.node();
    caps.cluster_id = identity.cluster();
    caps.snapshot_id = uuid();
    caps.backends[0].build_digest = starlings_protocol::byte_digest(b"synthetic-stage-build");
    crate::ResourceSample {
        sequence: Decimal(sequence),
        capabilities: caps,
        links: vec![],
    }
}
fn vectors(plan: &PlacementPlan, routes: &Routes) -> Vec<(TensorHeader, Vec<u8>)> {
    let values: Vec<serde_json::Value> =
        serde_json::from_str(include_str!("../../../../../examples/gcp1-golden.json")).unwrap();
    values
        .iter()
        .map(|v| {
            let mut h: TensorHeader = starlings_protocol::decode(
                v["header_utf8"].as_str().unwrap().as_bytes(),
                FixtureMode::Simulation,
            )
            .unwrap();
            let destination = usize::from(v["name"] != "u32");
            h.plan_digest = plan.plan_digest.clone();
            h.boundary_profile = plan.boundary_profile.clone();
            h.authority_epoch = plan.authority_epoch;
            h.session_id = routes.session.session_id;
            h.generation_id = routes.session.generation_id;
            h.source_stage = BoundedU32::new(1 - destination as u32).unwrap();
            h.destination_stage = BoundedU32::new(destination as u32).unwrap();
            h.lease_id = routes.leases[destination];
            (h, serde_json::from_value(v["payload"].clone()).unwrap())
        })
        .collect()
}
fn device(path: &Path) {
    let directory = PrivateDir::open(path).unwrap();
    for absent in ["authority.json", "enrollment.sqlite", "client-token.json"] {
        assert!(directory.open_file(absent).is_err());
    }
    let config: DeviceConfig = crate::enrollment::read_record(&directory, "config.json").unwrap();
    let identity = Identity::load(&directory, "identity.json").unwrap();
    let pin = Invitation::load(&directory, "pin.json").unwrap();
    let registry_path = path.join("registry");
    PrivateDir::ensure(&registry_path).unwrap();
    let mut registry = RegistryClient::open(
        &registry_path,
        Identity::load(&directory, "identity.json").unwrap(),
        Invitation::load(&directory, "pin.json").unwrap(),
    )
    .unwrap();
    for sequence in 1..=2 {
        registry
            .heartbeat(Some(sample(&identity, sequence)))
            .unwrap();
    }
    let control =
        PeerListener::bind("127.0.0.1:0".parse().unwrap(), ListenerBudget::default()).unwrap();
    let data =
        PeerListener::bind("127.0.0.1:0".parse().unwrap(), ListenerBudget::default()).unwrap();
    directory
        .write_atomic_durable(
            "ready.json",
            &serde_json::to_vec(&Ready {
                control: control.local_addr().unwrap(),
                data: data.local_addr().unwrap(),
            })
            .unwrap(),
        )
        .unwrap();
    let mut channel = control
        .accept()
        .unwrap()
        .generation(&identity, &pin, registry.fence(), config.binding)
        .unwrap();
    let runtime = Worker::new(
        identity.node(),
        identity.cluster(),
        4_194_304,
        channel.binding().epoch.number.0,
        FixtureMode::Reject,
    )
    .unwrap();
    let owner = StageWorker::accept_assignment(runtime, registry.fence(), &mut channel).unwrap();
    for _ in 0..3 {
        let request: ControlRequest = channel.receive_record().unwrap();
        assert!(!owner.apply(&channel, &request).unwrap());
        channel.send_record(&true).unwrap();
    }
    let routes: Routes = channel.receive_record().unwrap();
    let grants: [_; 2] = std::array::from_fn(|source| {
        owner
            .refresh_grant(
                &mut channel,
                routes.leases[config.index],
                BoundedU32::new(source as u32).unwrap(),
                BoundedU32::new(1 - source as u32).unwrap(),
            )
            .unwrap()
    });
    let vectors = vectors(&owner.plan, &routes);
    let before = serde_json::to_value(owner.snapshot().unwrap()).unwrap();
    let flow_budget = crate::ReceiveBudget::new(1_048_576, 1_048_576, 1_048_576).unwrap();
    let mut assemblies = 0;
    for (source, grant) in grants.iter().enumerate() {
        let entries: Vec<_> = vectors
            .iter()
            .filter(|(h, _)| h.source_stage.get() as usize == source)
            .collect();
        let sender = source == config.index;
        let peer = if sender {
            PeerStream::connect_stage(
                TcpStream::connect(routes.data[1 - source]).unwrap(),
                &identity,
                &pin,
                grant.clone(),
            )
            .unwrap()
        } else {
            data.accept()
                .unwrap()
                .stage(&identity, &pin, grant.clone())
                .unwrap()
        };
        assert!(peer.authority().is_err());
        let admissions: Vec<_> = entries
            .iter()
            .map(|(h, _)| owner.data_admission(&peer, grant, h.clone()).unwrap())
            .collect();
        let mut stream = if sender {
            DataStream::connect(peer, &admissions[0]).unwrap()
        } else {
            DataStream::accept(peer, &admissions[0]).unwrap()
        };
        for ((h, bytes), admission) in entries.iter().zip(&admissions) {
            let ack = DataHeader::Control(DataControl::ReceiptAck {
                protocol_version: Version,
                scope: h.into(),
                tensor_index: h.tensor_index,
                received_through_bytes: h.total_tensor_bytes,
            });
            if sender {
                stream
                    .send(admission, &DataHeader::Tensor(h.clone()), bytes)
                    .unwrap();
                stream
                    .receive(admission, |frame| {
                        assert_eq!(frame.header(), &ack);
                        assert!(frame.payload().is_empty());
                    })
                    .unwrap();
            } else {
                stream
                    .receive(admission, |frame| {
                        assert_eq!(frame.header(), &DataHeader::Tensor(h.clone()));
                        assert_eq!(frame.payload(), bytes);
                    })
                    .unwrap();
                stream.send(admission, &ack, &[]).unwrap();
            }
            // Keep the original raw framing checks above. The same enrolled
            // endpoints also exercise exclusive credit/reassembly ownership.
            let deadline = Instant::now() + crate::transport::TIMEOUT;
            if sender {
                let mut flow = stream
                    .begin_send(std::slice::from_ref(admission), deadline)
                    .unwrap();
                assert!(matches!(
                    flow.receive_control().unwrap(),
                    DataControl::Credit { .. }
                ));
                for chunk in bytes.chunks(bytes.len() / 2) {
                    flow.send_chunk(chunk).unwrap();
                    assert!(matches!(
                        flow.receive_control().unwrap(),
                        DataControl::ReceiptAck { .. }
                    ));
                }
                flow.finish().unwrap();
            } else {
                let mut flow = stream
                    .begin_receive(std::slice::from_ref(admission), &flow_budget, deadline)
                    .unwrap();
                flow.grant_credit(h.total_tensor_bytes.0).unwrap();
                flow.receive_chunk().unwrap();
                flow.receive_chunk().unwrap();
                let received = flow.finish().unwrap();
                assert_eq!(received.tensors()[0].as_ref(), bytes.as_slice());
                assert_eq!(flow_budget.held_bytes().unwrap(), h.total_tensor_bytes.0);
                assemblies += 1;
                drop(received);
                assert_eq!(flow_budget.held_bytes().unwrap(), 0);
            }
        }
    }
    assert_eq!(assemblies, if config.index == 0 { 1 } else { 3 });
    assert_eq!(
        serde_json::to_value(owner.snapshot().unwrap()).unwrap(),
        before,
        "data/receipt mutated execution state"
    );
    channel.send_record(&true).unwrap();
}
#[test]
fn complete_stage_workers_exchange_all_raw_dtypes_across_processes_without_authority_files() {
    if let Some(path) = std::env::var_os(CHILD_ENV) {
        device(Path::new(&path));
        return;
    }
    use crate::enrollment::tests::{enrolled, setup};
    let (temp, _, mut authority, files) = setup();
    let identities = [
        enrolled(&mut authority, &files, Role::Worker, "process-stage0"),
        enrolled(&mut authority, &files, Role::Worker, "process-stage1"),
    ];
    let controller = enrolled(&mut authority, &files, Role::Controller, "process-client");
    let epoch = authority.start_registry().unwrap();
    let mut plan: PlacementPlan = starlings_protocol::decode(
        include_bytes!("../../../../../examples/placement-plan.json"),
        FixtureMode::Simulation,
    )
    .unwrap();
    plan.fixture = false;
    plan.cluster_id = identities[0].cluster();
    plan.authority_epoch = epoch.number;
    plan.manifest_digest = starlings_protocol::byte_digest(b"synthetic-stage-model");
    for (stage, identity) in plan.stages.iter_mut().zip(&identities) {
        stage.node_id = identity.node();
        stage.backend_build_digest = starlings_protocol::byte_digest(b"synthetic-stage-build");
    }
    plan.plan_digest =
        starlings_protocol::canonical_digest(&serde_json::to_value(&plan).unwrap(), "plan_digest")
            .unwrap();
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
    let binding = GenerationBinding {
        version: Version,
        epoch,
        model: plan.manifest_digest.clone(),
        profile: plan.execution_profile.clone(),
    };
    let registry =
        PeerListener::bind("127.0.0.1:0".parse().unwrap(), ListenerBudget::default()).unwrap();
    let paths: [_; 2] = std::array::from_fn(|i| temp.path().join(format!("device{i}")));
    let mut children: Vec<_> = paths.iter().enumerate().map(|(i, path)| {
        let directory = PrivateDir::ensure(path).unwrap();
        identities[i].save(&directory, "identity.json").unwrap();
        authority.invite(&directory, "pin.json", registry.local_addr().unwrap(), Role::Worker).unwrap();
        directory.create("config.json").unwrap().write_all(&serde_json::to_vec(&DeviceConfig { index: i, binding: binding.clone() }).unwrap()).unwrap();
        ChildGuard(Command::new(std::env::current_exe().unwrap()).args(["--exact", "authorization::stage::process_tests::complete_stage_workers_exchange_all_raw_dtypes_across_processes_without_authority_files", "--nocapture"]).env(CHILD_ENV, path).env("STARLINGS_TEST_REGISTRY_TRACE", "1").stdout(Stdio::null()).spawn().unwrap())
    }).collect();
    let startup = Instant::now() + Duration::from_secs(10);
    for _ in 0..4 {
        let peer = loop {
            if let Some(peer) = registry.try_accept().unwrap() {
                break peer;
            }
            for child in &mut children {
                assert!(
                    child.0.try_wait().unwrap().is_none(),
                    "stage process exited before heartbeat"
                );
            }
            assert!(Instant::now() < startup, "stage heartbeat watchdog");
            thread::park_timeout(Duration::from_millis(5));
        };
        match peer
            .registry(authority.enrollment_reader().unwrap())
            .unwrap()
        {
            RegistryRequest::Heartbeat(pending) => pending.respond().unwrap(),
            _ => panic!("expected enrolled heartbeat"),
        }
    }
    let ready: Vec<Ready> = paths
        .iter()
        .map(|path| {
            loop {
                let directory = PrivateDir::open(path).unwrap();
                if directory.open_file("ready.json").is_ok() {
                    break crate::enrollment::read_record(&directory, "ready.json").unwrap();
                }
                for child in &mut children {
                    assert!(
                        child.0.try_wait().unwrap().is_none(),
                        "stage process exited before readiness"
                    );
                }
                assert!(Instant::now() < startup, "stage readiness watchdog");
                thread::park_timeout(Duration::from_millis(5));
            }
        })
        .collect();
    let leases: [Lease; 2] = std::array::from_fn(|i| {
        let mut lease: Lease = starlings_protocol::decode(
            include_bytes!("../../../../../examples/lease.json"),
            FixtureMode::Simulation,
        )
        .unwrap();
        lease.lease_id = uuid();
        lease.owner_node_id = identities[i].node();
        lease.cluster_id = plan.cluster_id;
        lease.authority_epoch = epoch.number;
        lease.plan_digest = plan.plan_digest.clone();
        lease
    });
    let mut session: Session = starlings_protocol::decode(
        include_bytes!("../../../../../examples/session.json"),
        FixtureMode::Simulation,
    )
    .unwrap();
    session.plan_digest = plan.plan_digest.clone();
    session.authority_epoch = epoch.number;
    session.prompt_token_count = BoundedU32::new(4).unwrap();
    let routes = Routes {
        data: [ready[0].data, ready[1].data],
        leases: leases.each_ref().map(|l| l.lease_id),
        session: session.clone(),
    };
    let mut channels = Vec::new();
    let mut issuers = Vec::new();
    for i in 0..2 {
        let mut channel = GenerationStream::connect(
            TcpStream::connect(ready[i].control).unwrap(),
            authority.identity(),
            &authority.trust().unwrap(),
            identities[i].node(),
            binding.clone(),
        )
        .unwrap();
        let issuer = authority
            .enrollment_reader()
            .unwrap()
            .stage_issuer(&token, plan.clone(), leases[i].clone())
            .unwrap();
        issuer.assign(&mut channel).unwrap();
        let mut local = session.clone();
        local.lease_id = leases[i].lease_id;
        for (sequence, operation) in [
            Operation::Prepare {
                plan: Box::new(plan.clone()),
                lease: Box::new(leases[i].clone()),
            },
            Operation::Commit {
                lease_id: leases[i].lease_id,
            },
            Operation::Begin {
                session: Box::new(local),
            },
        ]
        .into_iter()
        .enumerate()
        {
            let request =
                ControlRequest::new(uuid(), epoch.number.0, sequence as u64 + 1, operation)
                    .unwrap();
            channel.send_record(&request).unwrap();
            assert!(channel.receive_record::<bool>().unwrap());
        }
        channels.push(channel);
        issuers.push(issuer);
    }
    for channel in &mut channels {
        channel.send_record(&routes).unwrap();
    }
    for (issuer, channel) in issuers.iter().zip(&mut channels) {
        for _ in 0..2 {
            issuer.respond(channel).unwrap();
        }
    }
    for channel in &mut channels {
        assert!(channel.receive_record::<bool>().unwrap());
    }
    let end = Instant::now() + Duration::from_secs(5);
    for child in &mut children {
        while child.0.try_wait().unwrap().is_none() {
            assert!(Instant::now() < end, "stage exit watchdog");
            thread::park_timeout(Duration::from_millis(5));
        }
        assert!(child.0.wait().unwrap().success());
    }
}
