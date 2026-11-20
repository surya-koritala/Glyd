use std::{
    fs,
    io::{Read, Seek},
    net::{TcpListener, TcpStream},
    path::{Path, PathBuf},
    process::{Child, Command, Output},
    thread,
    time::{Duration, Instant},
};
const BIN: &str = env!("CARGO_BIN_EXE_starlings");
#[cfg(feature = "test-backend")]
thread_local! {
    static CLI_START_ALLOWED: std::cell::Cell<bool> = const { std::cell::Cell::new(true) };
}
#[cfg(windows)]
static CLI_PROCESS: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[test]
fn directed_probe_is_background_authenticated_and_expires_on_drain_and_revoke() {
    directed_probe(false);
}
#[test]
fn token_link_profile_exports_one_private_snapshot_without_placement_or_native_authority() {
    directed_probe(true);
}
fn directed_probe(token_profile: bool) {
    use starlings_protocol::content_hash;
    use starlings_security::PrivateDir;
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().canonicalize().unwrap();
    let co = root.join("coordinator");
    ok(&co, &["cluster", "init", "--listen", "127.0.0.1:0"]);
    let mut authority = Daemon::start(&co);
    let invite = co.join("probe-worker.json");
    invitation_once(&authority.state, "worker", &invite);
    let device = root.join("worker");
    let prepared = ok(&device, &["cluster", "prepare", invite.to_str().unwrap()]);
    let node = value(&prepared, "node=");
    authority.admin(&[
        "cluster",
        "approve",
        &node,
        &value(&prepared, "fingerprint="),
    ]);
    thread::sleep(Duration::from_millis(220));
    ok(&device, &["cluster", "join", invite.to_str().unwrap()]);
    let mut worker = Daemon::start(&device);
    wait_status(&worker, "registry=online");
    let mut limits = serde_json::json!({"version":1,"payload_bytes":16384,"ping_samples":3,"inter_ping_gap_ms":0,"minimum_interval_ms":30000});
    if token_profile {
        limits["ping_samples"] = 5.into();
        limits["inter_ping_gap_ms"] = 20.into();
        limits["token_profile"] =
            serde_json::json!({"gap_basis":"start_to_start","keepalive_ms":null});
    }
    let server = serde_json::json!({"version":1,"listen":"127.0.0.1:0","limits":limits});
    let files = PrivateDir::open(&co).unwrap();
    files
        .write_atomic_durable("probe-server.json", &serde_json::to_vec(&server).unwrap())
        .unwrap();
    assert_eq!(
        ok(
            &co,
            &[
                "cluster",
                "probe-digest",
                co.join("probe-server.json").to_str().unwrap()
            ]
        )
        .trim(),
        content_hash(&server).unwrap().as_str()
    );
    assert!(
        !command(
            &co,
            &[
                "cluster",
                "probe-listen",
                co.join("probe-server.json").to_str().unwrap(),
                "--approve",
                &"0".repeat(64)
            ]
        )
        .status
        .success()
    );
    let response = authority.admin(&[
        "cluster",
        "probe-listen",
        co.join("probe-server.json").to_str().unwrap(),
        "--approve",
        content_hash(&server).unwrap().as_str(),
    ]);
    let endpoint = response.strip_prefix("PROBE_LISTENING: ").unwrap().trim();
    let route = serde_json::json!({"version":1,"destination":endpoint,"source":"127.0.0.1:0","limits":limits});
    let device_files = PrivateDir::open(&device).unwrap();
    device_files
        .write_atomic_durable("probe-route.json", &serde_json::to_vec(&route).unwrap())
        .unwrap();
    let before = Instant::now();
    assert!(
        worker
            .admin(&[
                "cluster",
                "probe",
                device.join("probe-route.json").to_str().unwrap(),
                "--approve",
                content_hash(&route).unwrap().as_str()
            ])
            .contains("PROBE_STARTED")
    );
    assert!(
        before.elapsed() < Duration::from_secs(3),
        "probe I/O cannot occupy admin response"
    );
    let until = Instant::now() + Duration::from_secs(8);
    let measurement = loop {
        let text = worker.admin(&["cluster", "links", &node]);
        let view: serde_json::Value =
            serde_json::from_str(text.strip_prefix("LINKS: ").unwrap().trim()).unwrap();
        if view["state"]
            == if token_profile {
                "profile_complete"
            } else {
                "measured"
            }
        {
            break view;
        }
        assert!(view["state"] != "failed", "probe failed: {view}");
        assert!(Instant::now() < until, "no measured probe: {view}");
    };
    if token_profile {
        assert!(measurement["measurement"].is_null());
        assert!(measurement["placement_inputs"].is_null());
        let attempt = &measurement["profile_attempt"];
        assert_eq!(attempt["report"]["token_profile"]["completed_samples"], 5);
        assert_eq!(attempt["admission"], false);
        let output = device.join("profile.jsonl");
        assert!(
            ok(
                &device,
                &["cluster", "links-record", &node, output.to_str().unwrap()]
            )
            .contains("PROFILE_SNAPSHOT_SAVED")
        );
        let bytes = fs::read(&output).unwrap();
        assert_eq!(bytes.iter().filter(|b| **b == b'\n').count(), 1);
        let record: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(record["qualification"], "not_assessed");
        assert_eq!(
            record["snapshot"]["profile_attempt"]["report"]["id"],
            attempt["report"]["id"]
        );
        assert_eq!(
            record["snapshot"]["profile_attempt"]["report"]["token_profile"]["native_qualified"],
            false
        );
        assert!(
            !command(
                &device,
                &["cluster", "links-record", &node, output.to_str().unwrap()]
            )
            .status
            .success()
        );
        assert_eq!(fs::read(&output).unwrap(), bytes);
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(
                fs::metadata(&output).unwrap().permissions().mode() & 0o777,
                0o600
            );
        }
    } else {
        assert_eq!(
            measurement["measurement"]["report"]["transfers"][0]["source_node"],
            node
        );
        assert_eq!(
            measurement["measurement"]["report"]["application_payload_bytes"],
            "32768"
        );
        let id = measurement["measurement"]["report"]["id"].clone();
        let inputs = measurement["placement_inputs"].as_array().unwrap();
        assert_eq!(inputs.len(), 2);
        for (index, input) in inputs.iter().enumerate() {
            let transfer = &measurement["measurement"]["report"]["transfers"][index];
            assert_eq!(input["sample_id"], id);
            assert_eq!(input["source"], transfer["source_node"]);
            assert_eq!(input["destination"], transfer["destination_node"]);
            assert_eq!(
                input["acknowledged_bytes_per_second"],
                transfer["acknowledged_bytes_per_second"]
            );
            assert_eq!(
                input["authority_epoch"],
                measurement["measurement"]["report"]["epoch"]["number"]
            );
            assert_eq!(
                input["authority_incarnation"],
                measurement["measurement"]["report"]["epoch"]["incarnation"]
            );
            assert_eq!(input["fixture"], false);
        }
        let age = measurement["measurement"]["age_ms"].as_u64().unwrap();
        let again = worker.admin(&["cluster", "links", &node]);
        let again: serde_json::Value =
            serde_json::from_str(again.strip_prefix("LINKS: ").unwrap().trim()).unwrap();
        assert_eq!(again["measurement"]["report"]["id"], id);
        assert!(again["measurement"]["age_ms"].as_u64().unwrap() > age);
        assert_eq!(
            again["placement_inputs"][0]["age_ms"],
            again["measurement"]["age_ms"]
        );
        assert!(
            authority
                .admin(&["cluster", "links", &node])
                .contains("measured")
        );
    }
    assert!(
        worker
            .admin(&["cluster", "status"])
            .contains("registry=online")
    );
    drain_once(&worker);
    let drained = worker.admin(&["cluster", "links", &node]);
    let drained: serde_json::Value =
        serde_json::from_str(drained.strip_prefix("LINKS: ").unwrap().trim()).unwrap();
    assert!(drained["placement_inputs"].is_null());
    if token_profile {
        assert_eq!(
            drained["profile_attempt"]["report"]["id"],
            measurement["profile_attempt"]["report"]["id"]
        );
        let output = device.join("drained-profile.jsonl");
        ok(
            &device,
            &["cluster", "links-record", &node, output.to_str().unwrap()],
        );
        let record: serde_json::Value = serde_json::from_slice(&fs::read(output).unwrap()).unwrap();
        assert_eq!(record["snapshot"]["state"], "unavailable");
        assert_eq!(record["snapshot"]["profile_attempt"]["admission"], false);
    }
    assert!(
        worker
            .admin(&["cluster", "links", &node])
            .contains("unavailable")
    );
    authority.admin(&["cluster", "revoke", &node]);
    assert!(
        authority
            .admin(&["cluster", "links", &node])
            .contains("unavailable")
    );
    mutation_once(&worker.state, &["cluster", "stop"]);
    worker.wait();
    mutation_once(&authority.state, &["cluster", "stop"]);
    authority.wait();
}

#[cfg(feature = "test-backend")]
#[test]
fn catalog_service_production_route_counts_relays_reuses_slot_and_revokes_client_tokens() {
    // Windows' hosted filesystem/IPC cannot sustain all scenarios concurrently.
    // Keep its serialized production matrix without relaxing deadlines.
    #[cfg(windows)]
    for mode in [
        "normal",
        "crash",
        "partial",
        "disconnect",
        "backpressure",
        "busy-idle",
        "pressure",
        "two",
        "registry-loss",
    ] {
        let started = Instant::now();
        println!("PRODUCTION_MODE_START mode={mode}");
        production_route(mode);
        println!(
            "PRODUCTION_MODE_DONE mode={mode} elapsed_ms={}",
            started.elapsed().as_millis()
        );
    }
    #[cfg(not(windows))]
    production_route("normal");
}

#[cfg(all(feature = "test-backend", not(windows)))]
#[test]
fn catalog_service_two_workers_run_concurrently_through_the_coordinator() {
    production_route("two");
}

#[cfg(all(feature = "test-backend", not(windows)))]
#[test]
fn catalog_service_crashed_backend_retains_ownership_without_restart() {
    production_route("crash");
}

#[cfg(all(feature = "test-backend", not(windows)))]
#[test]
fn catalog_service_revoked_stream_retains_unqualified_ownership() {
    production_route("partial");
}

#[cfg(all(feature = "test-backend", not(windows)))]
#[test]
fn catalog_service_disconnected_stream_cancels_without_replay_or_releasing_ownership() {
    production_route("disconnect");
}

#[cfg(all(feature = "test-backend", not(windows)))]
#[test]
fn catalog_service_slow_reader_cancels_without_replay_or_releasing_ownership() {
    production_route("backpressure");
}

#[cfg(all(feature = "test-backend", not(windows)))]
#[test]
fn catalog_service_busy_slot_cannot_claim_request_drain() {
    production_route("busy-idle");
}

#[cfg(all(feature = "test-backend", not(windows)))]
#[test]
fn catalog_service_registry_loss_survives_cleanup_and_reconnect_without_restart() {
    production_route("registry-loss");
}

#[cfg(all(feature = "test-backend", unix))]
#[test]
fn catalog_service_qualified_recovery_fences_old_work_and_allows_fresh_restart() {
    production_route("qualified");
}
#[cfg(all(feature = "test-backend", not(windows)))]
#[test]
fn catalog_service_pressure_withdraws_output_and_retains_unqualified_ownership() {
    production_route("pressure");
}

#[cfg(feature = "test-backend")]
fn start_execution(
    device: &Path,
    route: &starlings_cli::routes::WorkerRoute,
) -> (Daemon, std::net::SocketAddr) {
    let files = starlings_security::PrivateDir::open(device).unwrap();
    let digest = starlings_protocol::content_hash(route).unwrap();
    files
        .write_atomic_durable("execution.json", &serde_json::to_vec(route).unwrap())
        .unwrap();
    let mut process = Command::new(BIN);
    process
        .args(["daemon", "--execute"])
        .arg(device.join("execution.json"))
        .args(["--approve", digest.as_str()]);
    let worker = Daemon::launch_with_arguments(device, process);
    let deadline = Instant::now() + Duration::from_secs(12);
    let mut observed_text = worker.admin(&["cluster", "service"]);
    let endpoint = loop {
        let text = &observed_text;
        let report: serde_json::Value =
            serde_json::from_str(text.strip_prefix("SERVICE: ").unwrap().trim()).unwrap();
        if report["state"] == "ready" {
            assert_eq!(report["fixture"], true);
            assert_eq!(report["inference_qualified"], false);
            assert_eq!(report["startup_phase"], "complete");
            assert!(report["startup_error"].is_null());
            break report["endpoint"].as_str().unwrap().parse().unwrap();
        }
        assert!(
            report["state"] != "failed_pins_retained" && report["state"] != "failed_cleanup_proven",
            "service startup failed: {report}"
        );
        assert!(
            Instant::now() < deadline,
            "{}: service not ready: {report}; registry: {}; topology: {}",
            route.service.environment["COMPUTE_FIXTURE_MODE"],
            worker.admin(&["cluster", "status"]),
            worker.admin(&["cluster", "topology"])
        );
        observed_text = worker.observe(&["cluster", "service"]);
    };
    (worker, endpoint)
}

#[cfg(feature = "test-backend")]
fn production_route(mode: &str) {
    use starlings_adapters::service::{ServiceConfig, executable_digest};
    use starlings_cli::routes::{CoordinatorRoute, Qualification, WorkerRoute};
    use starlings_core::routing::WarmReplica;
    use starlings_protocol::*;
    use starlings_security::{
        Action, ClientToken, PrivateDir, Scope,
        catalog::{Catalog, seal},
    };
    use std::io::Write;
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().canonicalize().unwrap();
    let coordinator = root.join("coordinator");
    ok(
        &coordinator,
        &["cluster", "init", "--listen", "127.0.0.1:0"],
    );
    let mut authority = Daemon::start(&coordinator);
    let mut nodes = Vec::new();
    let mut members = vec![("worker", "worker"), ("client", "client")];
    if mode == "two" {
        members.push(("worker", "worker2"));
    }
    for (role, name) in members {
        let state = root.join(name);
        let invite = coordinator.join(format!("{name}.json"));
        invitation_once(&authority.state, role, &invite);
        let prepared = ok(&state, &["cluster", "prepare", invite.to_str().unwrap()]);
        let node = value(&prepared, "node=");
        authority.admin(&[
            "cluster",
            "approve",
            &node,
            &value(&prepared, "fingerprint="),
        ]);
        thread::sleep(Duration::from_millis(220));
        ok(&state, &["cluster", "join", invite.to_str().unwrap()]);
        nodes.push(node);
    }
    let device = root.join("worker");
    let saved: serde_json::Value =
        serde_json::from_slice(&fs::read(coordinator.join("config.json")).unwrap()).unwrap();
    let cluster: ClusterId = saved["cluster"].as_str().unwrap().parse().unwrap();
    let source = PrivateDir::ensure(device.join("source")).unwrap();
    let mut draft: ModelManifest = decode(
        include_bytes!("../../../examples/model-manifest.json"),
        FixtureMode::Simulation,
    )
    .unwrap();
    draft.fixture = false;
    draft.source.revision = "a".repeat(40);
    for asset in &mut draft.assets {
        source
            .write_atomic_durable(&asset.path, asset.path.as_bytes())
            .unwrap();
        asset.size_bytes.0 = asset.path.len() as u64;
    }
    let manifest = seal(&serde_json::to_vec(&draft).unwrap(), &source).unwrap();
    drop(source);
    let cache = device.join("cache/models");
    let mut catalog = Catalog::open(&cache).unwrap();
    catalog
        .register(&encode(&manifest, FixtureMode::Reject).unwrap())
        .unwrap();
    catalog.approve(&manifest.manifest_digest, false).unwrap();
    catalog
        .import_directory(&manifest.manifest_digest, &device.join("source"))
        .unwrap();
    let weight = catalog
        .asset_path(&manifest.manifest_digest, "weights.safetensors")
        .unwrap();
    drop(catalog);
    let executable = PathBuf::from(BIN).canonicalize().unwrap();
    let mut service = ServiceConfig {
        executable_sha256: executable_digest(&executable).unwrap(),
        executable,
        arguments: vec!["fixture-service".into(), weight.to_str().unwrap().into()],
        directory: root.clone(),
        environment: [
            (
                "COMPUTE_FIXTURE_MODE".into(),
                match mode {
                    "normal" | "two" | "registry-loss" => "loading",
                    "pressure" => "partial",
                    "qualified" => "partial",
                    "disconnect" => "partial",
                    _ => mode,
                }
                .into(),
            ),
            (
                "COMPUTE_FIXTURE_STARTED".into(),
                root.join("started").to_str().unwrap().into(),
            ),
        ]
        .into(),
        key_environment: "COMPUTE_FIXTURE_KEY".into(),
        endpoint: "127.0.0.1:1".parse().unwrap(),
        backend_version: "synthetic-v1".into(),
        properties: None,
        api: Default::default(),
        glyd_identity: None,
        manifest_digest: manifest.manifest_digest.clone(),
        execution_profile: manifest.execution_profile.clone(),
        backend_model: "synthetic-backend".into(),
        reasoning_content: true,
        request_timeout_ms: if matches!(mode, "disconnect" | "backpressure") {
            // Cancellation must beat both the fixture's stall and its request timeout.
            30_000
        } else if mode == "two" {
            10_000
        } else {
            1000
        },
    };
    if let Some(system) = std::env::var_os("SystemRoot") {
        service
            .environment
            .insert("SYSTEMROOT".into(), system.to_string_lossy().into_owned());
    }
    if mode == "two" {
        service.environment.insert(
            "COMPUTE_FIXTURE_RELEASE".into(),
            root.join("release").to_str().unwrap().into(),
        );
    }
    if mode == "normal" {
        service.properties = Some(starlings_adapters::service::ExpectedProperties {
            build_info: "synthetic-v1".into(),
            context_tokens: 512,
        });
    }
    if mode == "pressure" {
        service.environment.insert(
            "COMPUTE_FIXTURE_PRESSURE".into(),
            root.join("pressure").to_string_lossy().into_owned(),
        );
    }
    let mut plan: PlacementPlan = decode(
        include_bytes!("../../../examples/placement-plan.json"),
        FixtureMode::Simulation,
    )
    .unwrap();
    plan.fixture = false;
    plan.cluster_id = cluster;
    plan.mode = PlacementPlanMode::WholeRequest;
    plan.manifest_digest = manifest.manifest_digest.clone();
    plan.execution_profile = manifest.execution_profile.clone();
    plan.context_limit = BoundedU32::new(8).unwrap();
    plan.stages.truncate(1);
    plan.stages[0].layer_end_exclusive = plan.model_layer_count;
    plan.stages[0].owns_final_norm_and_head = true;
    plan.stages[0].node_id = nodes[0].parse().unwrap();
    plan.stages[0].backend_id = "cpu-service".into();
    plan.stages[0].device_id = "cpu".into();
    plan.stages[0].backend_build_digest = service.executable_sha256.clone();
    plan.stages[0].memory_pool_id = "host".into();
    plan.plan_digest =
        canonical_digest(&serde_json::to_value(&plan).unwrap(), "plan_digest").unwrap();
    let qualification = Qualification {
        version: Version,
        fixture: true,
        os: std::env::consts::OS.into(),
        architecture: std::env::consts::ARCH.into(),
        device_kind: "cpu".into(),
        target: starlings_adapters::resources::HostSnapshot::collect_memory()
            .qualification_target()
            .unwrap(),
        loading_bytes: Decimal(plan.stages[0].reserved_bytes().unwrap() * 2),
        model_layer_count: manifest.layers.len() as u32,
        boundary_profile: manifest.required_boundary_profile.clone(),
        request_drain_profile: "slots-idle-v1".into(),
        configuration: service.digest().unwrap(),
        replica: WarmReplica {
            node: nodes[0].parse().unwrap(),
            manifest: manifest.manifest_digest.clone(),
            profile: manifest.execution_profile.clone(),
            backend: "cpu-service".into(),
            build: service.executable_sha256.clone(),
            pool: "host".into(),
            evidence: byte_digest(b"synthetic qualification; no native proof"),
            request_bytes: plan.stages[0].reserved_bytes().unwrap(),
            measured_request_ms: 1,
            context_limit: plan.context_limit.get(),
            max_sessions: 1,
            accepting: true,
        },
        max_output_tokens: 6,
        memory_qualified: true,
        exact_count_qualified: true,
        request_drain_qualified: true,
        cpu_threads: (mode == "pressure").then_some(1),
    };
    let worker_route = WorkerRoute {
        version: Version,
        listen: "127.0.0.1:0".parse().unwrap(),
        service,
        qualification: qualification.clone(),
        contribution: (mode == "pressure").then(|| starlings_cli::contribution::CpuContribution {
            version: Version,
            max_host_bytes: Decimal(
                qualification.loading_bytes.0 + qualification.replica.request_bytes,
            ),
            min_free_host_bytes: Decimal(0),
            max_cpu_threads: 1,
            max_temperature_c: None,
        }),
        cleanup: if mode == "qualified" {
            Some(starlings_cli::routes::CleanupQualification {
                version: Version,
                configuration: qualification.configuration.clone(),
                target: qualification.target.clone(),
                evidence: byte_digest(b"synthetic CPU/no-escape cleanup policy; no native claim"),
                scope: "posix-cpu-group-v1".into(),
                no_device_work_reviewed: true,
                no_escaped_children_reviewed: true,
            })
        } else {
            None
        },
    };
    let worker_digest = content_hash(&worker_route).unwrap();
    if mode == "normal" {
        worker_route.validate(&worker_digest).unwrap();
        let mut undersized = worker_route.clone();
        undersized
            .service
            .properties
            .as_mut()
            .unwrap()
            .context_tokens = 7;
        undersized.qualification.configuration = undersized.service.digest().unwrap();
        assert!(
            undersized
                .validate(&content_hash(&undersized).unwrap())
                .is_err()
        );
    }
    if mode != "pressure" {
        assert!(
            serde_json::to_value(&worker_route)
                .unwrap()
                .get("contribution")
                .is_none()
        );
    }
    worker_route.validate(&worker_digest).unwrap();
    let mut refused = worker_route.clone();
    refused.qualification.memory_qualified = false;
    assert!(refused.validate(&content_hash(&refused).unwrap()).is_err());
    refused = worker_route.clone();
    refused.qualification.configuration = byte_digest(b"another configuration");
    assert!(refused.validate(&content_hash(&refused).unwrap()).is_err());
    // Complete slow policy/token publication before starting lease-fenced services.
    let cfiles = PrivateDir::open(&coordinator).unwrap();
    for (node, action) in [
        (saved["node"].as_str().unwrap(), Action::Control),
        (nodes[0].as_str(), Action::Infer),
        (nodes[1].as_str(), Action::Infer),
    ] {
        cfiles
            .write_atomic_durable(
                "scopes.json",
                &serde_json::to_vec(&vec![Scope {
                    model: manifest.manifest_digest.clone(),
                    action,
                }])
                .unwrap(),
            )
            .unwrap();
        permissions_once(&authority, node, &coordinator.join("scopes.json"));
    }
    if mode == "two" {
        permissions_once(&authority, &nodes[2], &coordinator.join("scopes.json"));
    }
    let token_file = coordinator.join("token.json");
    token_once(
        &authority.state,
        &nodes[1],
        &coordinator.join("scopes.json"),
        &token_file,
    );
    let token = ClientToken::load(&cfiles, "token.json").unwrap();
    let duplicate = command(
        &coordinator,
        &[
            "cluster",
            "token",
            &nodes[1],
            coordinator.join("scopes.json").to_str().unwrap(),
            "300",
            token_file.to_str().unwrap(),
        ],
    );
    assert!(!duplicate.status.success());
    assert_eq!(
        rusqlite::Connection::open(coordinator.join("enrollment.sqlite"))
            .unwrap()
            .query_row("SELECT count(*) FROM tokens", [], |row| row
                .get::<_, i64>(0))
            .unwrap(),
        1,
        "failed token-file publication must revoke the newly issued token"
    );
    let (mut worker, endpoint) = start_execution(&device, &worker_route);
    let mut secondary = None;
    let mut replicas = vec![];
    if mode == "two" {
        let other = root.join("worker2");
        let mut catalog = Catalog::open(&other.join("cache/models")).unwrap();
        catalog
            .register(&encode(&manifest, FixtureMode::Reject).unwrap())
            .unwrap();
        catalog.approve(&manifest.manifest_digest, false).unwrap();
        catalog
            .import_directory(&manifest.manifest_digest, &device.join("source"))
            .unwrap();
        let weight = catalog
            .asset_path(&manifest.manifest_digest, "weights.safetensors")
            .unwrap();
        drop(catalog);
        let mut second = worker_route.clone();
        second.service.arguments[1] = weight.to_str().unwrap().into();
        second.service.environment.insert(
            "COMPUTE_FIXTURE_STARTED".into(),
            root.join("started2").to_str().unwrap().into(),
        );
        second.qualification.configuration = second.service.digest().unwrap();
        second.qualification.replica.node = nodes[2].parse().unwrap();
        second.qualification.replica.measured_request_ms = 2;
        let (process, endpoint) = start_execution(&other, &second);
        let mut second_plan = plan.clone();
        second_plan.stages[0].node_id = nodes[2].parse().unwrap();
        second_plan.plan_digest =
            canonical_digest(&serde_json::to_value(&second_plan).unwrap(), "plan_digest").unwrap();
        replicas.push(starlings_cli::routes::CoordinatorReplica {
            worker_endpoint: endpoint,
            plan: second_plan,
            qualification: second.qualification,
        });
        secondary = Some(process);
    }
    let coordinator_route = CoordinatorRoute {
        replicas,
        version: Version,
        listen: "127.0.0.1:0".parse().unwrap(),
        worker_endpoint: endpoint,
        plan,
        qualification,
    };
    let route_digest = content_hash(&coordinator_route).unwrap();
    coordinator_route.validate(&route_digest).unwrap();
    cfiles
        .write_atomic_durable(
            "route.json",
            &serde_json::to_vec(&coordinator_route).unwrap(),
        )
        .unwrap();
    authority.admin(&[
        "cluster",
        "route",
        coordinator.join("route.json").to_str().unwrap(),
        "--approve",
        route_digest.as_str(),
    ]);
    wait_status(&worker, "registry=online");
    let address = value(&authority.admin(&["cluster", "status"]), "inference=");
    let mut chat: ChatRequest = decode(
        include_bytes!("../../../examples/chat-request.json"),
        FixtureMode::Reject,
    )
    .unwrap();
    chat.model = manifest.manifest_digest.to_string();
    chat.max_tokens = BoundedU32::new(4).unwrap();
    chat.stream = Some(false);
    let open = |body: &serde_json::Value| {
        thread::sleep(Duration::from_millis(130));
        let body = serde_json::to_vec(body).unwrap();
        let mut socket = TcpStream::connect(&address).unwrap();
        socket
            .set_read_timeout(Some(Duration::from_secs(8)))
            .unwrap();
        write!(socket, "POST /v1/chat/completions HTTP/1.1\r\nHost: {address}\r\nAuthorization: Bearer {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n", token.bearer(), body.len()).unwrap();
        socket.write_all(&body).unwrap();
        socket
    };
    let send = |body: &serde_json::Value| {
        let mut socket = open(body);
        let mut reply = String::new();
        socket.read_to_string(&mut reply).unwrap();
        reply
    };
    let request = |chat: &ChatRequest| send(&serde_json::to_value(chat).unwrap());
    // Wait for a fresh qualified-ready sample, rather than guessing network convergence.
    let deadline = Instant::now() + Duration::from_secs(10);
    let mut observed_text = authority.admin(&["cluster", "resources", &nodes[0]]);
    loop {
        let view = &observed_text;
        let view: serde_json::Value =
            serde_json::from_str(view.strip_prefix("RESOURCES: ").unwrap().trim()).unwrap();
        if view["sample"]["capabilities"]["state"] == "ready" {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "{mode}: no ready sample: {view}; worker: {}",
            worker.admin(&["cluster", "service"])
        );
        observed_text = authority.observe(&["cluster", "resources", &nodes[0]]);
    }
    if mode == "registry-loss" {
        // Crash only this owned fixture: graceful stop can send Draining before expiry.
        authority.child.kill().unwrap();
        authority.child.wait().unwrap();
        wait_status(&worker, "registry=offline");
        let deadline = Instant::now() + Duration::from_secs(8);
        let mut observed_text = worker.admin(&["cluster", "service"]);
        let failed = loop {
            let text = &observed_text;
            let report: serde_json::Value =
                serde_json::from_str(text.strip_prefix("SERVICE: ").unwrap().trim()).unwrap();
            if report["state"] == "failed_pins_retained"
                || report["state"] == "stopped_pins_retained"
            {
                break report;
            }
            assert!(
                Instant::now() < deadline,
                "registry loss not observed: {report}"
            );
            observed_text = worker.observe(&["cluster", "service"]);
        };
        assert_eq!(failed["error"], "REGISTRY_LOST", "{failed}");
        assert_eq!(failed["state"], "failed_pins_retained");
        assert_eq!(failed["registry_failure"], "offline", "{failed}");
        assert_eq!(failed["reserved_bytes"], "0");
        assert_eq!(failed["cleanup_proven"], false);
        #[cfg(windows)]
        assert_eq!(failed["process_group_absent"], false);
        assert!(
            !root.join("started").exists(),
            "no generation was requested"
        );
        authority = Daemon::start(&coordinator);
        wait_status(&worker, "registry=online");
        let text = worker.admin(&["cluster", "service"]);
        let report: serde_json::Value =
            serde_json::from_str(text.strip_prefix("SERVICE: ").unwrap().trim()).unwrap();
        assert_eq!(
            report, failed,
            "reconnect must not revive or relabel the service"
        );
        let text = authority.admin(&["cluster", "resources", &nodes[0]]);
        let view: serde_json::Value =
            serde_json::from_str(text.strip_prefix("RESOURCES: ").unwrap().trim()).unwrap();
        assert!(view["age_ms"].as_u64().unwrap() < 5000);
        assert_eq!(view["sample"]["capabilities"]["state"], "unavailable");
        assert_eq!(
            view["sample"]["capabilities"]["backends"],
            serde_json::json!([])
        );
        assert_eq!(
            view["sample"]["capabilities"]["control_endpoint"],
            "unavailable"
        );
        mutation_once(&worker.state, &["cluster", "stop"]);
        worker.wait();
        let mut catalog = Catalog::open(&cache).unwrap();
        assert!(catalog.ensure_no_service().is_err());
        assert_eq!(catalog.gc().unwrap(), 0);
        assert!(cache.join("service-start.json").exists());
        mutation_once(&authority.state, &["cluster", "stop"]);
        authority.wait();
        return;
    }
    if mode == "two" {
        let deadline = Instant::now() + Duration::from_secs(10);
        let mut observed_text = authority.admin(&["cluster", "resources", &nodes[2]]);
        loop {
            let text = &observed_text;
            let view: serde_json::Value =
                serde_json::from_str(text.strip_prefix("RESOURCES: ").unwrap().trim()).unwrap();
            if view["sample"]["capabilities"]["state"] == "ready" {
                break;
            }
            assert!(
                Instant::now() < deadline,
                "{mode}: secondary capability never ready: {view}; worker: {}",
                secondary.as_ref().unwrap().admin(&["cluster", "service"])
            );
            observed_text = authority.observe(&["cluster", "resources", &nodes[2]]);
        }
        let body = serde_json::to_value(&chat).unwrap();
        // A held backend request must not depend on another OS process starting.
        CLI_START_ALLOWED.set(false);
        let mut first = open(&body);
        let mut second = open(&body);
        let deadline = Instant::now() + Duration::from_secs(3);
        while !root.join("started").exists() || !root.join("started2").exists() {
            assert!(
                Instant::now() < deadline,
                "both actual worker backends must start before release; primary_started={} secondary_started={}",
                root.join("started").exists(),
                root.join("started2").exists(),
            );
            thread::sleep(Duration::from_millis(5));
        }
        let body_bytes = serde_json::to_vec(&chat).unwrap().len();
        let retained_bytes = (10 * body_bytes).to_string();
        let queue_started = Instant::now();
        let mut deadline = queue_started + Duration::from_secs(3);
        let mut abandoned = Vec::new();
        for queued in 1..=8 {
            // Writes are not admission: confirm queue entry before the next paced
            // connection consumes an unauthenticated slot. Share the same 3s
            // observation budget, excluding the existing connection-opening time.
            let opening = Instant::now();
            abandoned.push(open(&body));
            deadline += opening.elapsed();
            let submitted = Instant::now();
            let mut observed_text = authority.query(&["cluster", "status"]);
            loop {
                let route: serde_json::Value =
                    serde_json::from_str(&value(&observed_text, "routing=")).unwrap();
                if route["queued"] == queued {
                    assert_eq!(route["active"], 2);
                    assert_eq!(route["retained_requests"], queued + 2);
                    assert_eq!(
                        route["retained_bytes"],
                        ((queued + 2) * body_bytes).to_string()
                    );
                    break;
                }
                assert!(
                    Instant::now() < deadline,
                    "queue admission {queued} never completed: {route}; since_submit_ms={} queue_phase_ms={} observation_budget_overrun_ms={}",
                    submitted.elapsed().as_millis(),
                    queue_started.elapsed().as_millis(),
                    Instant::now()
                        .saturating_duration_since(deadline)
                        .as_millis(),
                );
                observed_text = authority.observe(&["cluster", "status"]);
            }
        }
        let refused = request(&chat);
        assert!(refused.starts_with("HTTP/1.1 503"), "{refused}");
        let error: WireError = decode(
            refused.split_once("\r\n\r\n").unwrap().1.as_bytes(),
            FixtureMode::Reject,
        )
        .unwrap();
        assert_eq!(error.error.code, ErrorDetailCode::CapacityUnavailable);
        assert!(!error.error.retryable);
        let status = authority.observe(&["cluster", "status"]);
        let route: serde_json::Value = serde_json::from_str(&value(&status, "routing=")).unwrap();
        assert_eq!(route["queued"], 8);
        assert_eq!(route["active"], 2);
        assert_eq!(route["retained_requests"], 10);
        assert_eq!(route["retained_bytes"], retained_bytes);
        drop(abandoned);
        let deadline = Instant::now() + Duration::from_secs(3);
        let mut observed_text = authority.observe(&["cluster", "status"]);
        loop {
            let status = &observed_text;
            let route: serde_json::Value =
                serde_json::from_str(&value(status, "routing=")).unwrap();
            if route["queued"] == 0 && route["retained_requests"] == 2 {
                assert_eq!(route["active"], 2);
                assert_eq!(
                    route["retained_bytes"],
                    (2 * serde_json::to_vec(&chat).unwrap().len()).to_string()
                );
                break;
            }
            assert!(
                Instant::now() < deadline,
                "queued disconnect did not discard unclaimed work: {route}"
            );
            observed_text = authority.observe(&["cluster", "status"]);
        }
        let mut queued = open(&body);
        fs::write(root.join("release"), b"release synthetic fixtures").unwrap();
        CLI_START_ALLOWED.set(true);
        for socket in [&mut first, &mut second, &mut queued] {
            let mut reply = String::new();
            socket.read_to_string(&mut reply).unwrap();
            assert!(
                reply.starts_with("HTTP/1.1 200"),
                "two-worker request failed: {reply}; primary: {}; secondary: {}",
                worker.admin(&["cluster", "service"]),
                secondary.as_ref().unwrap().admin(&["cluster", "service"])
            );
            assert!(reply.contains("synthetic answer"));
            assert!(reply.contains("\"total_tokens\":4"));
        }
        assert_eq!(
            fs::read_to_string(root.join("started"))
                .unwrap()
                .lines()
                .count()
                + fs::read_to_string(root.join("started2"))
                    .unwrap()
                    .lines()
                    .count(),
            3,
            "abandoned and refused requests must never invoke a backend"
        );
        let deadline = Instant::now() + Duration::from_secs(3);
        let mut observed_text = authority.admin(&["cluster", "status"]);
        loop {
            let status = &observed_text;
            let route: serde_json::Value =
                serde_json::from_str(&value(status, "routing=")).unwrap();
            if route["retained_requests"] == 0 {
                assert_eq!(route["queued"], 0);
                assert_eq!(route["active"], 0);
                assert_eq!(route["retained_bytes"], "0");
                break;
            }
            assert!(
                Instant::now() < deadline,
                "trusted completion did not release reservations: {route}"
            );
            observed_text = authority.observe(&["cluster", "status"]);
        }
        for worker in [&mut worker, secondary.as_mut().unwrap()] {
            mutation_once(&worker.state, &["cluster", "stop"]);
            worker.wait();
        }
        mutation_once(&authority.state, &["cluster", "stop"]);
        authority.wait();
        return;
    }
    let mut blocked_client = None;
    if mode != "normal" {
        if mode == "crash" || mode == "busy-idle" {
            assert!(request(&chat).starts_with("HTTP/1.1 503"));
        } else {
            chat.stream = Some(true);
            let body = serde_json::to_vec(&chat).unwrap();
            let mut socket = if mode == "backpressure" {
                let peer: std::net::SocketAddr = address.parse().unwrap();
                let socket = socket2::Socket::new(
                    socket2::Domain::for_address(peer),
                    socket2::Type::STREAM,
                    None,
                )
                .unwrap();
                // Negotiate the small receive window during connect, not afterward.
                socket.set_recv_buffer_size(1024).unwrap();
                socket.connect(&peer.into()).unwrap();
                TcpStream::from(socket)
            } else {
                TcpStream::connect(&address).unwrap()
            };
            socket
                .set_read_timeout(Some(Duration::from_secs(5)))
                .unwrap();
            write!(socket, "POST /v1/chat/completions HTTP/1.1\r\nHost: {address}\r\nAuthorization: Bearer {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n", token.bearer(), body.len()).unwrap();
            socket.write_all(&body).unwrap();
            let mut bytes = Vec::new();
            let deadline = Instant::now() + Duration::from_secs(5);
            while !String::from_utf8_lossy(&bytes).contains("partial") {
                let mut part = [0; 4096];
                let n = socket.read(&mut part).unwrap();
                assert!(n > 0 && Instant::now() < deadline);
                bytes.extend_from_slice(&part[..n]);
            }
            assert!(String::from_utf8_lossy(&bytes).contains("X-Compute-Session-Id:"));
            assert!(!String::from_utf8_lossy(&bytes).contains("data: [DONE]"));
            if mode == "backpressure" {
                // Keep the sink connected and unread until worker cancellation is observed.
                blocked_client = Some(socket);
            } else if mode == "disconnect" {
                socket.shutdown(std::net::Shutdown::Both).unwrap();
                drop(socket);
            } else {
                if mode == "pressure" {
                    fs::write(
                        root.join("pressure"),
                        b"synthetic policy pressure, not native telemetry",
                    )
                    .unwrap();
                } else {
                    mutation_once(
                        &authority.state,
                        &["cluster", "revoke-token", token_file.to_str().unwrap()],
                    );
                    assert!(request(&chat).starts_with("HTTP/1.1 401"));
                }
                socket.read_to_end(&mut bytes).unwrap();
                assert!(!String::from_utf8_lossy(&bytes).contains("data: [DONE]"));
            }
        }
        let deadline = Instant::now() + Duration::from_secs(8);
        let mut observed_text = worker.admin(&["cluster", "service"]);
        loop {
            let report = &observed_text;
            let report: serde_json::Value =
                serde_json::from_str(report.strip_prefix("SERVICE: ").unwrap().trim()).unwrap();
            let expected = if mode == "qualified" {
                "failed_cleanup_proven"
            } else {
                "failed_pins_retained"
            };
            if report["state"] == expected {
                let reserved = report["reserved_bytes"]
                    .as_str()
                    .unwrap()
                    .parse::<u64>()
                    .unwrap();
                if mode == "qualified" {
                    assert_eq!(reserved, 0);
                    assert_eq!(report["cleanup_proven"], true);
                } else {
                    assert!(
                        reserved > 0,
                        "unqualified failure must retain its runtime/reservation: {report}"
                    );
                }
                break;
            }
            assert!(Instant::now() < deadline, "failure not observed: {report}");
            observed_text = worker.observe(&["cluster", "service"]);
        }
        let status = worker.admin(&["cluster", "status"]);
        assert!(status.contains("role=worker"));
        if mode == "backpressure" {
            assert!(
                status.contains("registry=online"),
                "registry expiry cannot satisfy backpressure: {status}"
            );
        }
        drop(blocked_client);
        if matches!(mode, "disconnect" | "backpressure") {
            let status = authority.admin(&["cluster", "status"]);
            assert!(status.contains("role=coordinator"));
            let route: serde_json::Value =
                serde_json::from_str(&value(&status, "routing=")).unwrap();
            assert_eq!(route["queued"], 0);
            assert_eq!(route["active"], 1);
            assert_eq!(route["retained_requests"], 1);
            assert!(
                route["retained_bytes"]
                    .as_str()
                    .unwrap()
                    .parse::<u64>()
                    .unwrap()
                    > 0
            );
        }
        #[cfg(unix)]
        if mode == "qualified" {
            let info = worker.admin(&["cluster", "service"]);
            let info: serde_json::Value =
                serde_json::from_str(info.strip_prefix("SERVICE: ").unwrap().trim()).unwrap();
            let pin = info["pin"].as_str().unwrap();
            let configuration = worker_route.qualification.configuration.as_str();
            assert!(
                !command(
                    &device,
                    &[
                        "cluster",
                        "recover-service",
                        "00000000-0000-4000-8000-000000000999",
                        configuration
                    ]
                )
                .status
                .success()
            );
            assert!(
                worker
                    .admin(&["cluster", "recover-service", pin, configuration])
                    .contains("repeated=false")
            );
            assert!(
                worker
                    .admin(&["cluster", "recover-service", pin, configuration])
                    .contains("repeated=true")
            );
            assert!(worker.admin(&["cluster", "service"]).contains("recovered"));
            mutation_once(&worker.state, &["cluster", "stop"]);
            worker.wait();
            Catalog::open(&cache).unwrap().ensure_no_service().unwrap();
            let mut process = Command::new(BIN);
            process
                .args(["daemon", "--execute"])
                .arg(device.join("execution.json"))
                .args(["--approve", worker_digest.as_str()]);
            let mut restarted = Daemon::launch_with_arguments(&device, process);
            let until = Instant::now() + Duration::from_secs(12);
            let next = loop {
                let info = restarted.admin(&["cluster", "service"]);
                let info: serde_json::Value =
                    serde_json::from_str(info.strip_prefix("SERVICE: ").unwrap().trim()).unwrap();
                if info["state"] == "ready" {
                    break info["pin"].as_str().unwrap().to_owned();
                }
                assert!(
                    Instant::now() < until,
                    "recovered worker did not restart: {info}"
                );
            };
            assert_ne!(next, pin);
            mutation_once(&restarted.state, &["cluster", "stop"]);
            restarted.wait();
            assert!(
                !command(&device, &["model", "recover-service", pin, configuration])
                    .status
                    .success()
            );
            assert!(
                ok(&device, &["model", "recover-service", &next, configuration])
                    .contains("service_recovered")
            );
            let mut catalog = Catalog::open(&cache).unwrap();
            catalog.ensure_no_service().unwrap();
            assert_eq!(catalog.gc().unwrap(), manifest.assets.len() as u64);
            mutation_once(&authority.state, &["cluster", "stop"]);
            authority.wait();
            return;
        }
        assert_eq!(
            fs::read_to_string(root.join("started"))
                .unwrap()
                .lines()
                .count(),
            1,
            "an ambiguous generation must never restart its backend"
        );
        mutation_once(&worker.state, &["cluster", "stop"]);
        let deadline = Instant::now() + Duration::from_secs(12);
        loop {
            if let Some(status) = worker.child.try_wait().unwrap() {
                assert!(
                    !status.success(),
                    "unqualified failure drain must be explicit"
                );
                assert!(
                    fs::read_to_string(&worker.log)
                        .unwrap()
                        .contains("DRAIN_INCOMPLETE")
                );
                break;
            }
            assert!(Instant::now() < deadline);
            thread::sleep(Duration::from_millis(20));
        }
        mutation_once(&authority.state, &["cluster", "stop"]);
        authority.wait();
        assert!(cache.join("service-start.json").exists());
        return;
    }
    let slow_handshake = TcpStream::connect(endpoint).unwrap_or_else(|error| {
        panic!(
            "{mode}: worker endpoint refused: {error}; {}",
            worker.admin(&["cluster", "service"])
        )
    });
    for streaming in [false, false, true] {
        chat.stream = Some(streaming);
        let reply = request(&chat);
        assert!(reply.starts_with("HTTP/1.1 200 OK"), "{reply}");
        if streaming {
            assert!(
                reply.contains("X-Compute-Session-Id:") && reply.contains("data: [DONE]"),
                "{reply}"
            )
        } else {
            let payload = reply.split_once("\r\n\r\n").unwrap().1;
            let body = parse_json(payload.as_bytes()).unwrap();
            assert_eq!(body["usage"]["prompt_tokens"], 3);
            assert_eq!(body["choices"][0]["message"]["content"], "synthetic answer");
            assert_eq!(
                body["choices"][0]["message"]["reasoning_content"],
                "synthetic reasoning"
            );
        }
    }
    chat.stream = Some(false);
    chat.max_tokens = BoundedU32::new(6).unwrap();
    assert!(
        request(&chat).starts_with("HTTP/1.1 400"),
        "exact count plus output must fit context"
    );
    chat.max_tokens = BoundedU32::new(4).unwrap();
    assert!(
        request(&chat).starts_with("HTTP/1.1 200"),
        "a rejected context must leave the service usable"
    );
    chat.max_tokens = BoundedU32::new(7).unwrap();
    assert!(request(&chat).starts_with("HTTP/1.1 400"));
    chat.max_tokens = BoundedU32::new(4).unwrap();
    let mut unsupported = serde_json::to_value(&chat).unwrap();
    unsupported["tools"] = serde_json::json!([]);
    assert!(send(&unsupported).starts_with("HTTP/1.1 400"));
    let mut wrong_model = chat.clone();
    wrong_model.model = byte_digest(b"different model").to_string();
    assert!(request(&wrong_model).starts_with("HTTP/1.1 403"));
    mutation_once(
        &authority.state,
        &["cluster", "revoke-token", token_file.to_str().unwrap()],
    );
    assert!(request(&chat).starts_with("HTTP/1.1 401"));
    assert!(
        authority
            .admin(&["cluster", "status"])
            .contains("role=coordinator")
    );
    drop(slow_handshake);
    assert!(
        !command(
            &device,
            &[
                "cluster",
                "recover-service",
                "00000000-0000-4000-8000-000000000999",
                worker_route.qualification.configuration.as_str()
            ]
        )
        .status
        .success()
    );
    mutation_once(&worker.state, &["cluster", "stop"]);
    worker.wait();
    mutation_once(&authority.state, &["cluster", "stop"]);
    authority.wait();
    assert!(
        cache.join("service-start.json").exists(),
        "unqualified service cleanup must retain its quarantine"
    );
}
fn command(state: &Path, args: &[&str]) -> Output {
    command_with_init_trace(state, args, true)
}
fn command_with_init_trace(state: &Path, args: &[&str], trace_init: bool) -> Output {
    #[cfg(feature = "test-backend")]
    assert!(
        CLI_START_ALLOWED.get(),
        "CLI process startup while backend requests are held: {args:?}"
    );
    // Independent Windows CLI startups contend for hosted filesystem/crypto resources.
    // Daemons, sockets, requests and every scenario remain concurrent; only test clients serialize.
    #[cfg(windows)]
    let _process = CLI_PROCESS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    // Files avoid blocking a child or timeout reporting on a full/inherited pipe.
    let mut stdout = tempfile::tempfile().unwrap();
    let mut stderr = tempfile::tempfile().unwrap();
    let mut process = Command::new(BIN);
    process
        .arg("--state-dir")
        .arg(state)
        .args(args)
        .stdout(stdout.try_clone().unwrap())
        .stderr(stderr.try_clone().unwrap());
    if trace_init
        && cfg!(any(debug_assertions, feature = "test-backend"))
        && args.starts_with(&["cluster", "init"])
    {
        process.env("STARLINGS_TEST_INIT_EXIT_PHASE", "1");
    } else {
        process.env_remove("STARLINGS_TEST_INIT_EXIT_PHASE");
    }
    let mut child = process.spawn().unwrap();
    let start = Instant::now();
    let mut timed_out = false;
    let status = loop {
        if let Some(status) = child.try_wait().unwrap() {
            break status;
        }
        if start.elapsed() > Duration::from_secs(12) {
            child.kill().unwrap();
            timed_out = true;
            break child.wait().unwrap();
        }
        thread::sleep(Duration::from_millis(10));
    };
    stdout.rewind().unwrap();
    stderr.rewind().unwrap();
    let mut out = Vec::new();
    let mut err = Vec::new();
    stdout.read_to_end(&mut out).unwrap();
    stderr.read_to_end(&mut err).unwrap();
    assert!(
        !timed_out,
        "CLI command {args:?} exceeded deadline; stdout: {}; stderr: {}",
        String::from_utf8_lossy(&out),
        String::from_utf8_lossy(&err)
    );
    Output {
        status,
        stdout: out,
        stderr: err,
    }
}

fn mutation_reply_allows_observation(succeeded: bool, stderr: &str) -> bool {
    succeeded || stderr.trim() == "REFUSED: CONTROL_UNAVAILABLE: outcome unknown; check status"
}

#[cfg(any(debug_assertions, feature = "test-backend"))]
#[test]
fn catalog_service_initialization_collisions_report_bounded_phases_and_preserve_state() {
    for collision in ["lock", "enrollment.sqlite", "cache", "authority.json"] {
        eprintln!("INIT_COLLISION_CASE={collision}");
        let temp = tempfile::tempdir().unwrap();
        let state = temp.path().canonicalize().unwrap().join("init-collision");
        let mut lock = Some(starlings_security::StateLock::acquire(&state).unwrap());
        let retained = b"retained authority fixture";
        if collision != "lock" {
            let mut file = lock
                .as_ref()
                .unwrap()
                .directory()
                .create(collision)
                .unwrap();
            std::io::Write::write_all(&mut file, retained).unwrap();
            drop(file);
            drop(lock.take());
        }
        let output = command(&state, &["cluster", "init", "--listen", "127.0.0.1:0"]);
        assert!(!output.status.success());
        assert!(output.stdout.is_empty());
        let stderr = String::from_utf8(output.stderr)
            .unwrap()
            .replace("\r\n", "\n");
        let prefix = match collision {
            "lock" => "INIT_START_PHASE=entry\nINIT_START_PHASE=state_lock\nPROFILE_IN_USE:",
            "cache" => {
                "INIT_START_PHASE=entry\nINIT_START_PHASE=state_lock\nINIT_START_PHASE=cache\nPRIVATE_STATE_REQUIRED:"
            }
            "enrollment.sqlite" => concat!(
                "INIT_START_PHASE=entry\nINIT_START_PHASE=state_lock\n",
                "INIT_START_PHASE=cache\nINIT_START_PHASE=authority_create\n",
                "INIT_AUTHORITY_PHASE=directory_open\nINIT_AUTHORITY_PHASE=enrollment_create\nSECURITY_"
            ),
            "authority.json" => concat!(
                "INIT_START_PHASE=entry\nINIT_START_PHASE=state_lock\n",
                "INIT_START_PHASE=cache\nINIT_START_PHASE=authority_create\n",
                "INIT_AUTHORITY_PHASE=directory_open\nINIT_AUTHORITY_PHASE=enrollment_create\n",
                "INIT_AUTHORITY_PHASE=enrollment_sync\nINIT_AUTHORITY_PHASE=root_key\n",
                "INIT_AUTHORITY_PHASE=root_certificate\nINIT_AUTHORITY_PHASE=authority_publish\nSECURITY_"
            ),
            _ => unreachable!(),
        };
        assert!(
            stderr.starts_with(prefix),
            "unexpected init diagnostic: {stderr}"
        );
        assert!(!stderr.contains(state.to_str().unwrap()));
        assert!(!stderr.contains(std::str::from_utf8(retained).unwrap()));
        assert!(!stderr.contains("INIT_EXIT_PHASE="));
        for name in [
            "config.json",
            "authority.json",
            "coordinator.json",
            "ready.json",
            "identity.json",
        ] {
            if name != collision {
                assert!(!state.join(name).exists(), "refused init published {name}");
            }
        }
        if collision != "lock" {
            assert_eq!(fs::read(state.join(collision)).unwrap(), retained);
            // Refusal must release ownership without deleting the persistent lock file.
            assert!(state.join("daemon.lock").is_file());
            drop(starlings_security::StateLock::acquire(&state).unwrap());
        }
        if collision == "authority.json" {
            assert_eq!(
                fs::metadata(state.join("enrollment.sqlite")).unwrap().len(),
                0
            );
        }
        if collision == "cache" {
            assert!(!state.join("enrollment.sqlite").exists());
            fs::remove_file(state.join("cache")).unwrap();
            eprintln!("INIT_COLLISION_CASE=cache_recovery");
            ok(&state, &["cluster", "init", "--listen", "127.0.0.1:0"]);
            assert!(state.join("cache").is_dir());
            assert!(state.join("config.json").is_file());
            drop(starlings_security::StateLock::acquire(&state).unwrap());
        }
    }
}

#[test]
fn catalog_service_initialization_without_trace_remains_silent() {
    let temp = tempfile::tempdir().unwrap();
    let state = temp.path().canonicalize().unwrap().join("untraced-init");
    let output = command_with_init_trace(
        &state,
        &["cluster", "init", "--listen", "127.0.0.1:0"],
        false,
    );
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(output.stderr.is_empty());
    assert!(
        String::from_utf8(output.stdout)
            .unwrap()
            .starts_with("INITIALIZED:")
    );
    assert!(state.join("config.json").exists());
}

#[cfg(all(windows, feature = "test-backend"))]
#[test]
fn catalog_service_cli_lock_poison_cannot_mask_later_command_results() {
    // This mutex stores no state; a failed client must not mask the next result.
    assert!(
        std::panic::catch_unwind(|| {
            let _guard = CLI_PROCESS
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            panic!("controlled test-client failure");
        })
        .is_err()
    );
    let temp = tempfile::tempdir().unwrap();
    let state = temp.path().canonicalize().unwrap().join("lock-poison");
    let output = command(&state, &["cluster", "status"]);
    assert!(!output.status.success());
    assert!(output.stdout.is_empty());
    assert!(
        String::from_utf8(output.stderr)
            .unwrap()
            .starts_with("DAEMON_UNAVAILABLE:")
    );
}

#[cfg(feature = "test-backend")]
#[test]
fn catalog_service_memory_collection_cannot_block_registry_contact() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().canonicalize().unwrap();
    let co = root.join("coordinator");
    let device = root.join("worker");
    ok(&co, &["cluster", "init", "--listen", "127.0.0.1:0"]);
    let mut coordinator = Daemon::start(&co);
    let invite = co.join("worker.json");
    invitation_once(&coordinator.state, "worker", &invite);
    let prepared = ok(&device, &["cluster", "prepare", invite.to_str().unwrap()]);
    let node = value(&prepared, "node=");
    coordinator.admin(&[
        "cluster",
        "approve",
        &node,
        &value(&prepared, "fingerprint="),
    ]);
    ok(&device, &["cluster", "join", invite.to_str().unwrap()]);
    let release = device.join("memory.release");
    let mut process = Command::new(BIN);
    process.env("STARLINGS_TEST_MEMORY_RELEASE", &release);
    let mut worker = Daemon::launch(&device, process);
    let until = Instant::now() + Duration::from_secs(3);
    while !release.with_extension("entered").exists() {
        assert!(Instant::now() < until, "memory collector did not enter");
        thread::sleep(Duration::from_millis(20));
    }
    wait_status(&worker, "registry=online");
    // Remain online beyond the unchanged six-second fence with no memory result.
    let until = Instant::now() + starlings_security::REGISTRY_TTL + Duration::from_millis(250);
    while Instant::now() < until {
        assert!(
            worker
                .observe(&["cluster", "status"])
                .contains("registry=online")
        );
        assert_eq!(
            coordinator.observe(&["cluster", "resources", &node]).trim(),
            "RESOURCES: null"
        );
    }
    assert!(!release.exists());
    assert!(
        !release.with_extension("expired").exists(),
        "pause expired before release"
    );
    fs::write(&release, b"").unwrap();
    let until = Instant::now() + Duration::from_secs(9);
    loop {
        let text = coordinator.observe(&["cluster", "resources", &node]);
        let view: serde_json::Value =
            serde_json::from_str(text.trim().strip_prefix("RESOURCES: ").unwrap()).unwrap();
        if !view.is_null() {
            assert!(view["age_ms"].as_u64().unwrap() < 5000);
            assert_eq!(view["sample"]["capabilities"]["state"], "unavailable");
            assert_eq!(
                view["sample"]["capabilities"]["backends"],
                serde_json::json!([])
            );
            break;
        }
        assert!(
            Instant::now() < until,
            "collector did not publish after release"
        );
    }
    fs::remove_file(&release).unwrap();
    // A cached capture cannot renew its five-second freshness while collection is paused again.
    let until = Instant::now() + Duration::from_secs(9);
    loop {
        assert!(
            worker
                .observe(&["cluster", "status"])
                .contains("registry=online")
        );
        if coordinator.observe(&["cluster", "resources", &node]).trim() == "RESOURCES: null" {
            break;
        }
        assert!(
            Instant::now() < until,
            "cached memory was refreshed without collection"
        );
    }
    assert!(!release.with_extension("expired").exists());
    fs::write(&release, b"").unwrap();
    mutation_once(&worker.state, &["cluster", "stop"]);
    worker.wait();
    mutation_once(&coordinator.state, &["cluster", "stop"]);
    coordinator.wait();
}

#[cfg(feature = "test-backend")]
#[test]
fn catalog_service_observation_paths_match_and_record_paired_latency() {
    use std::io::Write;
    let temp = tempfile::tempdir().unwrap();
    let state = temp.path().canonicalize().unwrap().join("observation");
    ok(&state, &["cluster", "init", "--listen", "127.0.0.1:0"]);
    let mut daemon = Daemon::start(&state);
    let mut samples = Vec::new();
    for index in 0..5 {
        let mut pair = serde_json::json!({});
        let mut records = Vec::new();
        // Alternate order to expose drift; pacing is the same and excluded from latency.
        for cli in if index % 2 == 0 {
            [true, false]
        } else {
            [false, true]
        } {
            thread::sleep(Duration::from_millis(220));
            let start = Instant::now();
            let text = if cli {
                ok(&state, &["cluster", "status"])
            } else {
                daemon.query(&["cluster", "status"])
            };
            pair[if cli { "cli_us" } else { "ipc_us" }] =
                serde_json::json!(start.elapsed().as_micros());
            let mut stable: Vec<_> = [
                "role=",
                "node=",
                "cluster=",
                "enrollment=",
                "inference=",
                "registry=",
                "epoch=",
            ]
            .iter()
            .map(|field| value(&text, field))
            .collect();
            stable.push(text.split_once(": ").unwrap().0.into());
            records.push(stable);
        }
        assert_eq!(
            records[0], records[1],
            "observation path changed stable status"
        );
        samples.push(pair);
    }
    writeln!(std::io::stdout().lock(),
        "OBSERVATION_TIMING: {}",
        serde_json::json!({"os":std::env::consts::OS,"architecture":std::env::consts::ARCH,"pairs":samples})
    ).unwrap();
    let context = daemon.failure_context();
    assert!(context.contains("\"status\", Ok(("));
    assert!(context.contains("\"service\", Ok(("));
    assert!(context.contains("\"topology\", Ok(("));
    mutation_once(&daemon.state, &["cluster", "stop"]);
    daemon.wait();
    let context = daemon.failure_context();
    assert_eq!(context.matches("Err(Storage)").count(), 3);
    assert!(context.ends_with(&fs::read_to_string(&daemon.log).unwrap()));
    let failed: Result<(), _> = std::panic::catch_unwind(move || {
        let _owned = daemon;
        panic!("controlled daemon unwind");
    });
    assert_eq!(
        failed.unwrap_err().downcast_ref::<&str>(),
        Some(&"controlled daemon unwind")
    );
}

fn fixture_request<T>(
    deadline: Instant,
    mut receive: impl FnMut() -> Result<Option<T>, starlings_security::Error>,
    client_finished: impl Fn() -> bool,
) -> Result<T, &'static str> {
    loop {
        if Instant::now() >= deadline {
            return Err("FIXTURE_REQUEST_DEADLINE");
        }
        if let Some(request) = receive().map_err(|_| "FIXTURE_ACCEPT_FAILED")? {
            return Ok(request);
        }
        if client_finished() {
            return Err("FIXTURE_CLIENT_FINISHED_WITHOUT_REQUEST");
        }
        thread::sleep(Duration::from_millis(10));
    }
}

#[test]
fn catalog_service_fixture_wait_stops_at_client_exit_deadline_and_accept_error() {
    let future = Instant::now() + Duration::from_secs(12);
    assert_eq!(
        fixture_request::<()>(future, || Ok(None), || true),
        Err("FIXTURE_CLIENT_FINISHED_WITHOUT_REQUEST")
    );
    assert_eq!(
        fixture_request::<()>(
            Instant::now(),
            || panic!("expired wait tried to accept"),
            || false
        ),
        Err("FIXTURE_REQUEST_DEADLINE")
    );
    assert_eq!(
        fixture_request::<()>(future, || Err(starlings_security::Error::Storage), || false),
        Err("FIXTURE_ACCEPT_FAILED")
    );
    // An already queued request must not be lost merely because its client finished.
    assert_eq!(fixture_request(future, || Ok(Some(7)), || true), Ok(7));
    let mut attempts = 0;
    assert_eq!(
        fixture_request(
            future,
            || {
                attempts += 1;
                Ok((attempts == 2).then_some(9))
            },
            || false
        ),
        Ok(9)
    );
    assert_eq!(attempts, 2);
}

#[cfg(feature = "test-backend")]
#[test]
fn catalog_service_admin_reply_loss_is_unknown_after_one_authenticated_command() {
    let temp = tempfile::tempdir().unwrap();
    let state = temp.path().canonicalize().unwrap().join("admin-reply-loss");
    starlings_security::PrivateDir::ensure(&state).unwrap();
    let listener = starlings_security::AdminListener::bind(
        &state,
        starlings_security::ListenerBudget::default(),
    )
    .unwrap();
    thread::scope(|threads| {
        let client = threads.spawn(|| command(&state, &["cluster", "leave"]));
        let received: serde_json::Value = fixture_request(
            Instant::now() + Duration::from_secs(12),
            || listener.try_accept(),
            || client.is_finished(),
        )
        .unwrap()
        .receive()
        .unwrap();
        assert_eq!(received, serde_json::json!({"command":"leave"}));
        // receive() closes this one authenticated connection without delivering a reply.
        let output = client.join().unwrap();
        assert!(!output.status.success());
        assert!(output.stdout.is_empty());
        let stderr = String::from_utf8(output.stderr).unwrap();
        assert_eq!(
            stderr.trim(),
            "REFUSED: CONTROL_UNAVAILABLE: outcome unknown; check status"
        );
        assert!(mutation_reply_allows_observation(false, &stderr));
        assert!(
            !state.join("daemon.lock").exists(),
            "unknown leave attempted offline mutation"
        );
        assert!(
            listener.try_accept().unwrap().is_none(),
            "mutation was retried"
        );
    });
    thread::scope(|threads| {
        let client = threads.spawn(|| mutation_once(&state, &["cluster", "stop"]));
        let received: serde_json::Value = fixture_request(
            Instant::now() + Duration::from_secs(12),
            || listener.try_accept(),
            || client.is_finished(),
        )
        .unwrap()
        .receive()
        .unwrap();
        assert_eq!(received, serde_json::json!({"command":"stop"}));
        client
            .join()
            .expect("a lost stop reply permits observation, not a retry");
        assert!(!state.join("daemon.lock").exists());
        assert!(listener.try_accept().unwrap().is_none(), "stop was retried");
    });
}

#[cfg(feature = "test-backend")]
#[test]
fn catalog_service_mutation_reply_requires_success_or_started_unknown() {
    assert!(mutation_reply_allows_observation(true, ""));
    assert!(mutation_reply_allows_observation(
        false,
        "REFUSED: CONTROL_UNAVAILABLE: outcome unknown; check status\n"
    ));
    for refused in [
        "REFUSED: CONTROL_UNAVAILABLE: not executed",
        "REFUSED: PRIVATE_POLICY_UNAVAILABLE",
        "other failure: outcome unknown; check status",
    ] {
        assert!(!mutation_reply_allows_observation(false, refused));
    }
}

#[cfg(feature = "test-backend")]
#[test]
fn catalog_service_publication_reply_loss_requires_active_observation_without_retry() {
    use std::io::Write;
    for (token, active) in [(false, true), (false, false), (true, true), (true, false)] {
        let temp = tempfile::tempdir().unwrap();
        let state = temp
            .path()
            .canonicalize()
            .unwrap()
            .join("publication-reply-loss");
        let files = starlings_security::PrivateDir::ensure(&state).unwrap();
        let scopes = serde_json::json!([{
            "model": starlings_protocol::byte_digest(b"synthetic reply-loss scope"), "action":"infer"
        }]);
        if token {
            files
                .create("scopes.json")
                .unwrap()
                .write_all(&serde_json::to_vec(&scopes).unwrap())
                .unwrap();
        }
        let scopes_file = state.join("scopes.json");
        let node = "00000000-0000-4000-8000-000000000001";
        let listener = starlings_security::AdminListener::bind(
            &state,
            starlings_security::ListenerBudget::default(),
        )
        .unwrap();
        let file = state.join("credential.json");
        thread::scope(|threads| {
            let client = threads.spawn(|| {
                std::panic::catch_unwind(|| {
                    if token {
                        token_once(&state, node, &scopes_file, &file);
                    } else {
                        invitation_once(&state, "worker", &file);
                    }
                })
            });
            eprintln!("PUBLICATION_REPLY_LOSS token={token} active={active} phase=await_mutation");
            let received: serde_json::Value = fixture_request(
                Instant::now() + Duration::from_secs(12),
                || listener.try_accept(),
                || client.is_finished(),
            )
            .unwrap()
            .receive()
            .unwrap();
            assert_eq!(
                received,
                if token {
                    serde_json::json!({"command":"token","node":node,"scopes":scopes,"lifetime":300,"output":file})
                } else {
                    serde_json::json!({"command":"invite","role":"worker","output":file,"via":null})
                }
            );
            eprintln!(
                "PUBLICATION_REPLY_LOSS token={token} active={active} phase=received_mutation"
            );
            // Drop the reply after one authenticated mutation. File presence is no proof.
            files
                .create("credential.json")
                .unwrap()
                .write_all(b"synthetic uncommitted publication")
                .unwrap();
            eprintln!(
                "PUBLICATION_REPLY_LOSS token={token} active={active} phase=await_observation"
            );
            fixture_request(
                Instant::now() + Duration::from_secs(12),
                || listener.try_accept(),
                || client.is_finished(),
            )
            .unwrap()
            .respond(|request: serde_json::Value| {
                assert_eq!(
                    request,
                    if token {
                        serde_json::json!({"command":"token_status","input":file})
                    } else {
                        serde_json::json!({"command":"invite_status","file":file})
                    }
                );
                starlings_cli::AdminReply::ok(
                    if token {
                        "TOKEN_STATUS"
                    } else {
                        "INVITATION_STATUS"
                    },
                    format!("active={active}"),
                )
            })
            .unwrap();
            eprintln!("PUBLICATION_REPLY_LOSS token={token} active={active} phase=response_sent");
            assert_eq!(client.join().unwrap().is_ok(), active);
            eprintln!("PUBLICATION_REPLY_LOSS token={token} active={active} phase=client_joined");
            assert!(
                listener.try_accept().unwrap().is_none(),
                "credential mutation was retried"
            );
        });
    }
}

fn mutation_once(state: &Path, args: &[&str]) {
    thread::sleep(Duration::from_millis(220));
    let outcome = command(state, args);
    // A started mutation can outlive its reply wait. Never resend it.
    assert!(
        mutation_reply_allows_observation(
            outcome.status.success(),
            &String::from_utf8_lossy(&outcome.stderr)
        ),
        "mutation {args:?} failed: {}",
        String::from_utf8_lossy(&outcome.stderr)
    );
}

fn drain_once(daemon: &Daemon) {
    mutation_once(&daemon.state, &["cluster", "drain"]);
    let status = wait_status(daemon, "admission=closed");
    assert!(status.starts_with("DRAINING:"), "{status}");
}

fn invitation_once(state: &Path, role: &str, file: &Path) {
    mutation_once(state, &["cluster", "invite", role, file.to_str().unwrap()]);
    observe_active(
        state,
        serde_json::json!({"command":"invite_status","file":file}),
        "INVITATION_STATUS",
    );
}

#[cfg(feature = "test-backend")]
fn token_once(state: &Path, node: &str, scopes: &Path, file: &Path) {
    assert!(
        !file.try_exists().unwrap(),
        "token helper requires a fresh output file"
    );
    mutation_once(
        state,
        &[
            "cluster",
            "token",
            node,
            scopes.to_str().unwrap(),
            "300",
            file.to_str().unwrap(),
        ],
    );
    observe_active(
        state,
        serde_json::json!({"command":"token_status","input":file}),
        "TOKEN_STATUS",
    );
}

fn observe_active(state: &Path, request: serde_json::Value, expected: &str) {
    thread::sleep(Duration::from_millis(220));
    // One authenticated observation proves activation; file presence cannot.
    let reply: starlings_cli::AdminReply = starlings_security::admin_call(state, &request).unwrap();
    assert!(reply.ok, "{}: {}", reply.code, reply.message);
    assert_eq!(reply.code, expected);
    assert_eq!(
        reply.message, "active=true",
        "{expected} not committed/active"
    );
}

#[cfg(feature = "test-backend")]
fn permissions_once(daemon: &Daemon, node: &str, file: &Path) {
    mutation_once(
        &daemon.state,
        &["cluster", "permissions", node, file.to_str().unwrap()],
    );
    let observation = daemon.admin(&["cluster", "permissions-status", node]);
    let actual: Vec<starlings_security::Scope> = serde_json::from_str(
        observation
            .strip_prefix("PERMISSIONS_STATUS: ")
            .unwrap()
            .trim(),
    )
    .unwrap();
    let expected: Vec<starlings_security::Scope> =
        serde_json::from_slice(&fs::read(file).unwrap()).unwrap();
    assert_eq!(
        actual, expected,
        "read-only inspection must confirm the full committed allowlist"
    );
}

fn ok(state: &Path, args: &[&str]) -> String {
    let output = command(state, args);
    assert!(
        output.status.success(),
        "command {args:?}: {}; daemon log: {}",
        String::from_utf8_lossy(&output.stderr),
        fs::read_to_string(state.join("test-daemon.log")).unwrap_or_default()
    );
    let expected = if cfg!(any(debug_assertions, feature = "test-backend"))
        && args.starts_with(&["cluster", "init"])
    {
        concat!(
            "INIT_START_PHASE=entry\nINIT_START_PHASE=state_lock\n",
            "INIT_START_PHASE=cache\nINIT_START_PHASE=authority_create\n",
            "INIT_AUTHORITY_PHASE=directory_open\nINIT_AUTHORITY_PHASE=enrollment_create\n",
            "INIT_AUTHORITY_PHASE=enrollment_sync\nINIT_AUTHORITY_PHASE=root_key\n",
            "INIT_AUTHORITY_PHASE=root_certificate\nINIT_AUTHORITY_PHASE=authority_publish\n",
            "INIT_AUTHORITY_PHASE=identity_key\nINIT_AUTHORITY_PHASE=coordinator_certificate\n",
            "INIT_AUTHORITY_PHASE=coordinator_publish\nINIT_AUTHORITY_PHASE=database_open\n",
            "INIT_AUTHORITY_PHASE=database_schema\nINIT_AUTHORITY_PHASE=member_insert\n",
            "INIT_AUTHORITY_PHASE=ready_publish\nINIT_AUTHORITY_PHASE=database_close\n",
            "INIT_AUTHORITY_PHASE=access_open\nINIT_AUTHORITY_PHASE=finished\n",
            "INIT_START_PHASE=config_publish\nINIT_EXIT_PHASE=authority_drop\n",
            "INIT_EXIT_PHASE=state_lock_drop\nINIT_EXIT_PHASE=resources_closed\n"
        )
    } else {
        ""
    };
    assert_eq!(
        String::from_utf8_lossy(&output.stderr).replace("\r\n", "\n"),
        expected
    );
    String::from_utf8(output.stdout).unwrap()
}
fn value(text: &str, field: &str) -> String {
    text.split_whitespace()
        .find_map(|word| word.strip_prefix(field))
        .unwrap()
        .trim_end_matches(';')
        .to_owned()
}
struct Daemon {
    child: Child,
    state: PathBuf,
    log: PathBuf,
}
impl Daemon {
    fn start(state: &Path) -> Self {
        Self::launch(state, Command::new(BIN))
    }
    fn launch(state: &Path, mut process: Command) -> Self {
        process.arg("daemon");
        Self::launch_with_arguments(state, process)
    }
    fn launch_with_arguments(state: &Path, mut process: Command) -> Self {
        #[cfg(feature = "test-backend")]
        process
            .env("STARLINGS_TEST_CLIENT_GATE_TRACE", "1")
            .env("STARLINGS_TEST_REGISTRY_TRACE", "1")
            .env("STARLINGS_TEST_GENERATION_GATE_TRACE", "1");
        let log = state.join("test-daemon.log");
        let out = fs::File::create(&log).unwrap();
        let child = process
            .arg("--state-dir")
            .arg(state)
            .stdout(out.try_clone().unwrap())
            .stderr(out)
            .spawn()
            .unwrap();
        let mut daemon = Self {
            child,
            state: state.into(),
            log,
        };
        let start = Instant::now();
        loop {
            let text = fs::read_to_string(&daemon.log).unwrap();
            if state.join("admin.cap").exists()
                && command(state, &["cluster", "status"]).status.success()
            {
                break;
            }
            assert!(daemon.child.try_wait().unwrap().is_none(), "{text}");
            assert!(
                start.elapsed() < Duration::from_secs(8),
                "daemon readiness timed out"
            );
            thread::sleep(Duration::from_millis(20));
        }
        daemon
    }
    fn admin(&self, args: &[&str]) -> String {
        // Stay below the intentional 8 admissions/second local bucket; mutations are never retried.
        thread::sleep(Duration::from_millis(220));
        ok(&self.state, args)
    }
    #[cfg(feature = "test-backend")]
    fn observe(&self, args: &[&str]) -> String {
        thread::sleep(Duration::from_millis(220));
        self.query(args)
    }
    #[cfg(feature = "test-backend")]
    fn query(&self, args: &[&str]) -> String {
        let request = match args {
            ["cluster", kind @ ("status" | "service")] => serde_json::json!({"command": kind}),
            ["cluster", "resources", node] => {
                let node: starlings_protocol::NodeId = node.parse().unwrap();
                serde_json::json!({"command": "resources", "node": node})
            }
            _ => panic!("only read-only fixture observation is allowed: {args:?}"),
        };
        let reply: starlings_cli::AdminReply =
            starlings_security::admin_call(&self.state, &request).unwrap();
        assert!(reply.ok, "{}: {}", reply.code, reply.message);
        format!("{}: {}\n", reply.code, reply.message)
    }
    fn failure_context(&self) -> String {
        let reports = ["status", "service", "topology"].map(|command| {
            let reply: Result<starlings_cli::AdminReply, _> = starlings_security::admin_call(
                &self.state,
                &serde_json::json!({"command":command}),
            );
            (command, reply.map(|reply| (reply.code, reply.message)))
        });
        format!(
            "reports={reports:?}; log={}",
            fs::read_to_string(&self.log).unwrap_or_default()
        )
    }
    fn wait(&mut self) {
        let start = Instant::now();
        loop {
            if let Some(status) = self.child.try_wait().unwrap() {
                assert!(
                    status.success(),
                    "{}",
                    fs::read_to_string(&self.log).unwrap()
                );
                break;
            }
            assert!(
                start.elapsed() < Duration::from_secs(12),
                "shutdown timed out"
            );
            thread::sleep(Duration::from_millis(20));
        }
        assert!(!self.state.join("admin.cap").exists());
        #[cfg(unix)]
        assert!(!self.state.join("admin.sock").exists());
        assert!(self.state.join("config.json").exists());
    }
}
impl Drop for Daemon {
    fn drop(&mut self) {
        if thread::panicking() {
            use std::io::Write;
            let _ = writeln!(
                std::io::stderr(),
                "DAEMON_FAILURE_CONTEXT {}",
                self.failure_context()
            );
        }
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

#[test]
fn real_processes_enroll_roles_drain_leave_restart_and_preserve_data() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().canonicalize().unwrap();
    let coordinator = root.join("coordinator");
    ok(
        &coordinator,
        &["cluster", "init", "--listen", "127.0.0.1:0"],
    );
    assert!(!command(&coordinator, &["cluster", "init"]).status.success());
    let identity = fs::read(coordinator.join("coordinator.json")).unwrap();
    fs::write(
        coordinator.join("cache/retained-model-data"),
        b"retained fixture bytes",
    )
    .unwrap();
    let mut daemon = Daemon::start(&coordinator);
    let duplicate = command(&coordinator, &["daemon"]);
    assert!(!duplicate.status.success());
    assert!(String::from_utf8_lossy(&duplicate.stderr).contains("PROFILE_IN_USE"));
    let ready = fs::read_to_string(&daemon.log).unwrap();
    let endpoint = value(&ready, "enrollment=");
    let slow = TcpStream::connect(&endpoint).unwrap();
    let start = Instant::now();
    assert!(
        daemon
            .admin(&["cluster", "status"])
            .contains("role=coordinator")
    );
    assert!(
        start.elapsed() < Duration::from_secs(5),
        "idle network sender blocked local control"
    );
    drop(slow);
    let diagnostic = daemon.admin(&["cluster", "diagnostics"]);
    let bundle: serde_json::Value = serde_json::from_str(&diagnostic).unwrap();
    assert_eq!(bundle["daemon"]["status"], "available");
    assert_eq!(bundle["daemon"]["data"]["role"], "coordinator");
    assert_eq!(bundle["members"]["data"]["workers"], 0);
    assert_eq!(bundle["service"]["status"], "available");
    assert!(bundle["service"]["data"].is_null());
    assert_eq!(bundle["qualification"], "not_assessed");
    assert!(!diagnostic.contains(&endpoint));
    assert!(!diagnostic.contains(&value(&ready, "node=")));
    for role in ["worker", "client"] {
        let invite = coordinator.join(format!("invite-{role}.json"));
        invitation_once(&daemon.state, role, &invite);
        let device = root.join(role);
        assert!(
            !command(&device, &["cluster", "join", invite.to_str().unwrap()])
                .status
                .success()
        );
        let prepared = ok(&device, &["cluster", "prepare", invite.to_str().unwrap()]);
        let node = value(&prepared, "node=");
        let fingerprint = value(&prepared, "fingerprint=");
        if role == "worker" {
            daemon.admin(&["cluster", "approve", &node, &"0".repeat(64)]);
            assert!(
                !command(&device, &["cluster", "join", invite.to_str().unwrap()])
                    .status
                    .success()
            );
        }
        daemon.admin(&["cluster", "approve", &node, &fingerprint]);
        thread::sleep(Duration::from_millis(220));
        let joined = ok(&device, &["cluster", "join", invite.to_str().unwrap()]);
        assert!(joined.contains(&format!("role={role}")));
        assert_eq!(
            daemon
                .admin(&["cluster", "invite-status", invite.to_str().unwrap()])
                .trim(),
            "INVITATION_STATUS: active=false"
        );
        let saved = fs::read(device.join("identity.json")).unwrap();
        let mut local = Daemon::start(&device);
        let refused = command(
            &device,
            &["cluster", "invite-status", invite.to_str().unwrap()],
        );
        assert!(!refused.status.success());
        assert!(String::from_utf8_lossy(&refused.stderr).contains("COORDINATOR_REQUIRED"));
        assert!(
            local
                .admin(&["cluster", "status"])
                .contains("inference=unavailable")
        );
        assert!(
            !command(&device, &["cluster", "approve", &node, &fingerprint])
                .status
                .success()
        );
        local.admin(&["cluster", "leave"]);
        local.wait();
        assert_eq!(fs::read(device.join("identity.json")).unwrap(), saved);
        let left = command(&device, &["daemon"]);
        assert!(!left.status.success());
        assert!(String::from_utf8_lossy(&left.stderr).contains("LEFT_LOCAL"));
        daemon.admin(&["cluster", "revoke", &node]);
    }
    drain_once(&daemon);
    // The shared port remains available for authenticated drain/revocation heartbeats.
    assert!(TcpStream::connect(&endpoint).is_ok());
    let invite = coordinator.join("denied.json");
    assert!(
        !command(
            &coordinator,
            &["cluster", "invite", "worker", invite.to_str().unwrap()]
        )
        .status
        .success()
    );
    mutation_once(&daemon.state, &["cluster", "stop"]);
    daemon.wait();
    assert_eq!(
        fs::read(coordinator.join("coordinator.json")).unwrap(),
        identity
    );
    let mut restarted = Daemon::start(&coordinator);
    // Abrupt death leaves transient endpoints; exclusive restart must recover only those names.
    restarted.child.kill().unwrap();
    restarted.child.wait().unwrap();
    let mut recovered = Daemon::start(&coordinator);
    mutation_once(&recovered.state, &["cluster", "stop"]);
    recovered.wait();
    assert_eq!(
        fs::read(coordinator.join("coordinator.json")).unwrap(),
        identity
    );
    assert_eq!(
        fs::read(coordinator.join("cache/retained-model-data")).unwrap(),
        b"retained fixture bytes"
    );
}

#[test]
fn invalid_configuration_and_endpoint_collision_fail_without_deleting_data() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().canonicalize().unwrap();
    let state = root.join("invalid");
    assert!(
        !command(&state, &["cluster", "init", "--listen", "0.0.0.0:9741"])
            .status
            .success()
    );
    ok(&state, &["cluster", "init", "--listen", "127.0.0.1:0"]);
    let config_path = state.join("config.json");
    let config = fs::read(&config_path).unwrap();
    // Preserve the protected inode/ACL while injecting an unknown field.
    let mut value: serde_json::Value = serde_json::from_slice(&config).unwrap();
    value["surprise"] = true.into();
    fs::write(&config_path, serde_json::to_vec(&value).unwrap()).unwrap();
    assert!(!command(&state, &["daemon"]).status.success());
    fs::write(&config_path, &config).unwrap();
    #[cfg(unix)]
    {
        fs::write(state.join("admin.sock"), b"not a socket").unwrap();
        assert!(!command(&state, &["daemon"]).status.success());
        assert_eq!(fs::read(state.join("admin.sock")).unwrap(), b"not a socket");
        fs::remove_file(state.join("admin.sock")).unwrap();
    }
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let mut value: serde_json::Value = serde_json::from_slice(&config).unwrap();
    value["listen"] = listener.local_addr().unwrap().to_string().into();
    fs::write(&config_path, serde_json::to_vec(&value).unwrap()).unwrap();
    assert!(!command(&state, &["daemon"]).status.success());
    assert!(!state.join("admin.cap").exists());
    assert!(state.join("coordinator.json").exists());
}

#[cfg(unix)]
#[test]
fn termination_signal_uses_ordered_cleanup() {
    let temp = tempfile::tempdir().unwrap();
    let state = temp.path().canonicalize().unwrap().join("signal");
    ok(&state, &["cluster", "init", "--listen", "127.0.0.1:0"]);
    for signal in ["-TERM", "-INT", "-HUP"] {
        let mut daemon = Daemon::start(&state);
        assert!(
            Command::new("kill")
                .args([signal, &daemon.child.id().to_string()])
                .status()
                .unwrap()
                .success()
        );
        daemon.wait();
        assert!(!state.join("admin.cap").exists());
        assert!(state.join("coordinator.json").exists());
    }
    let mut detached = Command::new("sh");
    detached.args(["-c", "trap '' HUP INT; exec \"$@\" 1>&-", "detached", BIN]);
    let mut daemon = Daemon::launch(&state, detached);
    mutation_once(&daemon.state, &["cluster", "stop"]);
    daemon.wait();
    assert!(!state.join("admin.cap").exists());
}

fn wait_status(daemon: &Daemon, expected: &str) -> String {
    let start = Instant::now();
    loop {
        let status = daemon.admin(&["cluster", "status"]);
        if status.contains(expected) {
            return status;
        }
        assert!(
            start.elapsed() < Duration::from_secs(9),
            "expected {expected}: {status}"
        );
        thread::sleep(Duration::from_millis(150));
    }
}

#[cfg(windows)]
#[test]
fn windows_console_events_preserve_ordered_cleanup() {
    use std::os::windows::process::CommandExt;
    let temp = tempfile::tempdir().unwrap();
    if std::env::var_os("STARLINGS_TEST_CONSOLE_READY").is_none() {
        let log = temp.path().join("console-launch.log");
        let output = fs::File::create(&log).unwrap();
        let mut launcher = Command::new("powershell.exe")
            .args([
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                include_str!("windows-console-launch.ps1"),
            ])
            .env(
                "STARLINGS_TEST_EXECUTABLE",
                std::env::current_exe().unwrap(),
            )
            .stdout(output.try_clone().unwrap())
            .stderr(output)
            .spawn()
            .unwrap();
        let deadline = Instant::now() + Duration::from_secs(75);
        loop {
            if let Some(status) = launcher.try_wait().unwrap() {
                assert!(status.success(), "{}", fs::read_to_string(&log).unwrap());
                return;
            }
            if Instant::now() >= deadline {
                launcher.kill().unwrap();
                launcher.wait().unwrap();
                panic!(
                    "console launcher timed out: {}",
                    fs::read_to_string(&log).unwrap()
                );
            }
            thread::sleep(Duration::from_millis(20));
        }
    }
    let state = temp.path().canonicalize().unwrap().join("console");
    ok(&state, &["cluster", "init", "--listen", "127.0.0.1:0"]);
    for signal in [0, 1] {
        // Native CTRL_C and CTRL_BREAK, not process termination.
        let mut process = Command::new(BIN);
        process.creation_flags(0x00000010); // CREATE_NEW_CONSOLE isolates the test signal.
        let mut daemon = Daemon::launch(&state, process);
        assert!(
            daemon
                .admin(&["cluster", "status"])
                .contains("registry=active")
        );
        let log = temp.path().join("console-helper.log");
        let output = fs::File::create(&log).unwrap();
        let mut helper = Command::new("powershell.exe")
            .args([
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                include_str!("windows-console.ps1"),
            ])
            .env("STARLINGS_TEST_STATE", &state)
            .env("STARLINGS_TEST_PID", daemon.child.id().to_string())
            .env("STARLINGS_TEST_SIGNAL", signal.to_string())
            .stdout(output.try_clone().unwrap())
            .stderr(output)
            .spawn()
            .unwrap();
        let deadline = Instant::now() + Duration::from_secs(15);
        let status = loop {
            if let Some(status) = helper.try_wait().unwrap() {
                break status;
            }
            if Instant::now() >= deadline {
                helper.kill().unwrap();
                helper.wait().unwrap();
                panic!(
                    "console helper exceeded deadline: {}",
                    fs::read_to_string(&log).unwrap()
                );
            }
            thread::sleep(Duration::from_millis(20));
        };
        assert!(status.success(), "{}", fs::read_to_string(&log).unwrap());
        eprintln!("waiting for ordered shutdown after native console signal {signal}");
        daemon.wait();
        assert!(!state.join("admin.cap").exists());
        assert!(state.join("coordinator.json").exists());
    }
}
fn wait_node(daemon: &Daemon, node: &str, status: &str) {
    let start = Instant::now();
    loop {
        let reply = daemon.admin(&["cluster", "nodes"]);
        let records: serde_json::Value =
            serde_json::from_str(reply.trim().strip_prefix("NODES: ").unwrap()).unwrap();
        if let Some(record) = records
            .as_array()
            .unwrap()
            .iter()
            .find(|r| r["node"] == node && r["status"] == status)
        {
            assert_eq!(
                record["schedulable"], false,
                "host-memory observation does not install a qualified backend"
            );
            assert_eq!(record["fresh_links"], 0);
            return;
        }
        assert!(
            start.elapsed() < Duration::from_secs(9),
            "expected {node} {status}: {reply}"
        );
        thread::sleep(Duration::from_millis(150));
    }
}
#[test]
fn registry_process_restart_partition_drain_and_revocation() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().canonicalize().unwrap();
    let coordinator = root.join("coordinator");
    let device = root.join("worker");
    ok(
        &coordinator,
        &[
            "cluster",
            "init",
            "--listen",
            "127.0.0.1:0",
            "--listen",
            "[::1]:0",
        ],
    );
    let mut daemon = Daemon::start(&coordinator);
    let invite = coordinator.join("worker.json");
    invitation_once(&daemon.state, "worker", &invite);
    let prepared = ok(&device, &["cluster", "prepare", invite.to_str().unwrap()]);
    let node = value(&prepared, "node=");
    daemon.admin(&[
        "cluster",
        "approve",
        &node,
        &value(&prepared, "fingerprint="),
    ]);
    ok(&device, &["cluster", "join", invite.to_str().unwrap()]);
    wait_node(&daemon, &node, "offline");
    let mut worker = Daemon::start(&device);
    let initial = wait_status(&worker, "registry=online");
    assert_eq!(value(&initial, "epoch="), "1");
    wait_node(&daemon, &node, "online");
    let resource = daemon.admin(&["cluster", "resources", &node]);
    let view: serde_json::Value =
        serde_json::from_str(resource.trim().strip_prefix("RESOURCES: ").unwrap()).unwrap();
    assert!(view["age_ms"].as_u64().unwrap() < 5000);
    let capabilities = &view["sample"]["capabilities"];
    assert_eq!(capabilities["node_id"], node);
    assert_eq!(capabilities["fixture"], false);
    assert_eq!(capabilities["state"], "unavailable");
    assert_eq!(capabilities["backends"].as_array().unwrap().len(), 0);
    assert_eq!(capabilities["memory_pools"].as_array().unwrap().len(), 1);
    let pool = &capabilities["memory_pools"][0];
    assert!(
        pool["total_bytes"]
            .as_str()
            .unwrap()
            .parse::<u64>()
            .unwrap()
            > 0
    );
    assert!(
        pool["available_bytes"]
            .as_str()
            .unwrap()
            .parse::<u64>()
            .unwrap()
            <= pool["total_bytes"]
                .as_str()
                .unwrap()
                .parse::<u64>()
                .unwrap()
    );
    // Keep both anonymous TLS buckets occupied through worker/coordinator restart,
    // status, drain, revoke and stop. The same source IP also carries legitimate traffic.
    use std::{
        io::Write,
        sync::{
            Arc,
            atomic::{AtomicBool, Ordering},
        },
    };
    struct Flood(Arc<AtomicBool>, Option<thread::JoinHandle<()>>);
    impl Drop for Flood {
        fn drop(&mut self) {
            self.0.store(true, Ordering::Relaxed);
            self.1.take().unwrap().join().unwrap();
        }
    }
    let saved: serde_json::Value =
        serde_json::from_slice(&fs::read(coordinator.join("config.json")).unwrap()).unwrap();
    let addresses: Vec<std::net::SocketAddr> = [
        saved["listen"].as_str().unwrap(),
        saved["additional_listen"][0].as_str().unwrap(),
    ]
    .into_iter()
    .map(|s| s.parse().unwrap())
    .collect();
    let stopped = Arc::new(AtomicBool::new(false));
    let flag = stopped.clone();
    let _flood = Flood(
        stopped,
        Some(thread::spawn(move || {
            let mut sockets = Vec::new();
            while !flag.load(Ordering::Relaxed) {
                sockets
                    .retain(|(_, at): &(TcpStream, Instant)| at.elapsed() < Duration::from_secs(4));
                for address in &addresses {
                    if let Ok(mut socket) =
                        TcpStream::connect_timeout(address, Duration::from_millis(100))
                    {
                        socket
                            .set_write_timeout(Some(Duration::from_millis(100)))
                            .unwrap();
                        if socket.write_all(&[22]).is_ok() {
                            sockets.push((socket, Instant::now()));
                        }
                    }
                }
                thread::sleep(Duration::from_millis(150));
            }
        })),
    );
    wait_status(&daemon, "handshakes=8");
    worker.child.kill().unwrap();
    worker.child.wait().unwrap();
    let mut worker = Daemon::start(&device);
    let restarted = wait_status(&worker, "registry=online");
    assert_eq!(value(&restarted, "epoch="), "2");
    assert!(daemon.admin(&["cluster", "status"]).contains("epoch=2"));
    mutation_once(&daemon.state, &["cluster", "stop"]);
    daemon.wait();
    wait_status(&worker, "registry=offline");
    let mut daemon = Daemon::start(&coordinator);
    assert!(daemon.admin(&["cluster", "status"]).contains("epoch=3"));
    let rejoined = wait_status(&worker, "registry=online");
    assert_eq!(value(&rejoined, "epoch="), "3");
    wait_node(&daemon, &node, "online");
    drain_once(&worker);
    wait_node(&daemon, &node, "draining");
    daemon.admin(&["cluster", "revoke", &node]);
    wait_status(&worker, "registry=revoked");
    wait_node(&daemon, &node, "revoked");
    mutation_once(&worker.state, &["cluster", "stop"]);
    worker.wait();
    mutation_once(&daemon.state, &["cluster", "stop"]);
    daemon.wait();
    assert!(device.join("identity.json").exists());
    assert!(device.join("registry-fence.json").exists());
}

#[test]
fn multiple_listeners_explicit_routes_and_topology_work_without_multicast() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().canonicalize().unwrap();
    let state = root.join("coordinator");
    ok(
        &state,
        &[
            "cluster",
            "init",
            "--listen",
            "127.0.0.1:0",
            "--listen",
            "127.0.0.1:0",
        ],
    );
    let mut coordinator = Daemon::start(&state);
    let saved: serde_json::Value =
        serde_json::from_slice(&fs::read(state.join("config.json")).unwrap()).unwrap();
    let a = saved["listen"].as_str().unwrap().to_owned();
    let b = saved["additional_listen"][0].as_str().unwrap().to_owned();
    let topology = coordinator.admin(&["cluster", "topology"]);
    let record: serde_json::Value =
        serde_json::from_str(topology.trim().strip_prefix("TOPOLOGY: ").unwrap()).unwrap();
    assert_eq!(record["listeners"], serde_json::json!([a, b]));
    assert!(record["mdns_interface"].is_null());
    assert_eq!(record["reverse_reachability"], "unknown");
    assert_eq!(record["discovery_confers_trust"], false);
    let invite = state.join("invite.json");
    assert!(
        !command(
            &state,
            &[
                "cluster",
                "invite",
                "worker",
                invite.to_str().unwrap(),
                "--via",
                "127.0.0.1:1"
            ]
        )
        .status
        .success()
    );
    coordinator.admin(&[
        "cluster",
        "invite",
        "worker",
        invite.to_str().unwrap(),
        "--via",
        &b,
    ]);
    let device = root.join("worker");
    let prepared = ok(&device, &["cluster", "prepare", invite.to_str().unwrap()]);
    coordinator.admin(&[
        "cluster",
        "approve",
        &value(&prepared, "node="),
        &value(&prepared, "fingerprint="),
    ]);
    // Explicit alternate enrollment address keeps the invitation's trust.
    ok(
        &device,
        &["cluster", "join", invite.to_str().unwrap(), "--via", &a],
    );
    assert!(
        !command(&device, &["cluster", "routes", "8.8.8.8:9741"])
            .status
            .success()
    );
    let unused = TcpListener::bind("127.0.0.1:0").unwrap();
    let unavailable = unused.local_addr().unwrap().to_string();
    drop(unused);
    ok(
        &device,
        &[
            "cluster",
            "routes",
            &unavailable,
            &b,
            "--source",
            "127.0.0.1:0",
        ],
    );
    let mut worker = Daemon::start(&device);
    wait_status(&worker, "registry=online");
    let topology = worker.admin(&["cluster", "topology"]);
    let record: serde_json::Value =
        serde_json::from_str(topology.trim().strip_prefix("TOPOLOGY: ").unwrap()).unwrap();
    assert_eq!(record["routes"][0]["authenticated"], false);
    assert_eq!(record["routes"][1]["authenticated"], true);
    assert_eq!(record["routes"][1]["destination"], b);
    assert!(record["routes"][1]["selected_interface"]["name"].is_string());
    assert!(
        !command(&device, &["cluster", "routes", &a])
            .status
            .success(),
        "running configuration must be locked"
    );
    mutation_once(&worker.state, &["cluster", "stop"]);
    worker.wait();
    mutation_once(&coordinator.state, &["cluster", "stop"]);
    coordinator.wait();
}

#[test]
fn offline_leave_and_global_option_forms_preserve_identity() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().canonicalize().unwrap();
    let coordinator = root.join("coordinator");
    ok(
        &coordinator,
        &["cluster", "init", "--listen", "127.0.0.1:0"],
    );
    let mut daemon = Daemon::start(&coordinator);
    let invite = coordinator.join("invite.json");
    invitation_once(&daemon.state, "client", &invite);
    let state = root.join("offline");
    let prepared = ok(&state, &["cluster", "prepare", invite.to_str().unwrap()]);
    daemon.admin(&[
        "cluster",
        "approve",
        &value(&prepared, "node="),
        &value(&prepared, "fingerprint="),
    ]);
    ok(&state, &["cluster", "join", invite.to_str().unwrap()]);
    mutation_once(&daemon.state, &["cluster", "stop"]);
    daemon.wait();
    let identity = fs::read(state.join("identity.json")).unwrap();
    let left = Command::new(BIN)
        .args(["cluster", "leave"])
        .arg(format!("--state-dir={}", state.display()))
        .output()
        .unwrap();
    assert!(
        left.status.success(),
        "{}",
        String::from_utf8_lossy(&left.stderr)
    );
    assert!(String::from_utf8_lossy(&left.stdout).contains("LEFT_LOCAL"));
    assert_eq!(fs::read(state.join("identity.json")).unwrap(), identity);
    let missing = temp.path().join("mistyped");
    let error = command(&missing, &["daemon"]);
    assert!(String::from_utf8_lossy(&error.stderr).contains("STATE_DIR_NOT_FOUND"));
    assert!(!missing.exists());
}

#[test]
fn resets_busy_database_and_broken_stdout_do_not_stop_the_daemon() {
    let temp = tempfile::tempdir().unwrap();
    let state = temp.path().canonicalize().unwrap().join("resilience");
    ok(&state, &["cluster", "init", "--listen", "127.0.0.1:0"]);
    let log = state.join("test-daemon.log");
    let mut child = Command::new(BIN)
        .arg("--state-dir")
        .arg(&state)
        .arg("daemon")
        .stdout(std::process::Stdio::piped())
        .stderr(fs::File::create(&log).unwrap())
        .spawn()
        .unwrap();
    // An actual broken pipe survives process startup; a closed descriptor may be reopened as /dev/null.
    drop(child.stdout.take());
    let mut daemon = Daemon {
        child,
        state: state.clone(),
        log,
    };
    let start = Instant::now();
    while !command(&state, &["cluster", "status"]).status.success() {
        assert!(daemon.child.try_wait().unwrap().is_none());
        assert!(start.elapsed() < Duration::from_secs(8));
        thread::sleep(Duration::from_millis(50));
    }
    let status = daemon.admin(&["cluster", "status"]);
    let endpoint = value(&status, "enrollment=");
    for _ in 0..64 {
        let socket = TcpStream::connect(&endpoint).unwrap();
        socket2::SockRef::from(&socket)
            .set_linger(Some(Duration::ZERO))
            .unwrap();
        drop(socket);
        thread::sleep(Duration::from_millis(5));
    }
    let db = rusqlite::Connection::open(state.join("enrollment.sqlite")).unwrap();
    db.execute_batch("BEGIN EXCLUSIVE").unwrap();
    thread::sleep(Duration::from_millis(2500));
    assert!(
        daemon.child.try_wait().unwrap().is_none(),
        "busy database stopped daemon"
    );
    db.execute_batch("ROLLBACK").unwrap();
    assert!(
        daemon
            .admin(&["cluster", "status"])
            .contains("role=coordinator")
    );
    mutation_once(&daemon.state, &["cluster", "stop"]);
    daemon.wait();
}

#[cfg(feature = "test-backend")]
#[test]
fn catalog_service_survives_failure_and_retains_pins_on_drain_and_restart() {
    use starlings_adapters::service::{ServiceConfig, executable_digest};
    use starlings_protocol::{FixtureMode, ModelManifest, decode, encode};
    use starlings_security::{
        PrivateDir,
        catalog::{Catalog, seal},
    };
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().canonicalize().unwrap();
    let coordinator = root.join("coordinator");
    ok(
        &coordinator,
        &["cluster", "init", "--listen", "127.0.0.1:0"],
    );
    let mut authority = Daemon::start(&coordinator);
    for mode in [
        "normal",
        "startup-crash",
        "unapproved",
        "wrong-path",
        "wrong-profile",
        "wrong-executable",
        "registry-loss",
    ] {
        let device = root.join(mode);
        let invite = coordinator.join(format!("invite-{mode}.json"));
        invitation_once(&authority.state, "worker", &invite);
        let prepared = ok(&device, &["cluster", "prepare", invite.to_str().unwrap()]);
        authority.admin(&[
            "cluster",
            "approve",
            &value(&prepared, "node="),
            &value(&prepared, "fingerprint="),
        ]);
        thread::sleep(Duration::from_millis(220));
        ok(&device, &["cluster", "join", invite.to_str().unwrap()]);
        let source = PrivateDir::ensure(device.join("source")).unwrap();
        let mut draft: ModelManifest = decode(
            include_bytes!("../../../examples/model-manifest.json"),
            FixtureMode::Simulation,
        )
        .unwrap();
        draft.fixture = false;
        draft.source.revision = "a".repeat(40);
        for asset in &mut draft.assets {
            source
                .write_atomic_durable(&asset.path, asset.path.as_bytes())
                .unwrap();
            asset.size_bytes.0 = asset.path.len() as u64;
        }
        let manifest = seal(&serde_json::to_vec(&draft).unwrap(), &source).unwrap();
        drop(source);
        let cache = device.join("cache/models");
        let mut catalog = Catalog::open(&cache).unwrap();
        catalog
            .register(&encode(&manifest, FixtureMode::Reject).unwrap())
            .unwrap();
        catalog.approve(&manifest.manifest_digest, false).unwrap();
        catalog
            .import_directory(&manifest.manifest_digest, &device.join("source"))
            .unwrap();
        let weight = catalog
            .asset_path(&manifest.manifest_digest, "weights.safetensors")
            .unwrap();
        drop(catalog);
        let executable = PathBuf::from(BIN).canonicalize().unwrap();
        let mut config = ServiceConfig {
            executable_sha256: executable_digest(&executable).unwrap(),
            executable,
            arguments: vec!["fixture-service".into(), weight.to_str().unwrap().into()],
            directory: root.clone(),
            environment: [("COMPUTE_FIXTURE_MODE".into(), mode.into())].into(),
            key_environment: "COMPUTE_FIXTURE_KEY".into(),
            endpoint: "127.0.0.1:1".parse().unwrap(),
            backend_version: "synthetic-v1".into(),
            properties: None,
            api: Default::default(),
            glyd_identity: None,
            manifest_digest: manifest.manifest_digest.clone(),
            execution_profile: manifest.execution_profile.clone(),
            backend_model: "synthetic-backend".into(),
            reasoning_content: false,
            request_timeout_ms: 1000,
        };
        if let Some(root) = std::env::var_os("SystemRoot") {
            config
                .environment
                .insert("SYSTEMROOT".into(), root.to_string_lossy().into_owned());
        }
        if mode == "wrong-path" {
            config.arguments[1] = device
                .join("source/weights.safetensors")
                .to_str()
                .unwrap()
                .into();
        }
        if mode == "wrong-profile" {
            config.execution_profile = "another-profile".into();
        }
        if matches!(mode, "normal" | "registry-loss") {
            config.properties = Some(starlings_adapters::service::ExpectedProperties {
                build_info: "synthetic-v1".into(),
                context_tokens: 512,
            });
        }
        if matches!(mode, "wrong-path" | "wrong-profile" | "wrong-executable") {
            // Wrong bindings refuse first; a valid binding still requires the digest check.
            config.executable_sha256 = "0".repeat(64).try_into().unwrap();
        }
        let approved = if mode == "unapproved" {
            "0".repeat(64)
        } else {
            config.digest().unwrap().as_str().into()
        };
        let directory = PrivateDir::open(&device).unwrap();
        directory
            .write_atomic_durable("service.json", &serde_json::to_vec(&config).unwrap())
            .unwrap();
        let start = || {
            // State-directory normalization preserves the service option order.
            let mut process = Command::new(BIN);
            process
                .args(["daemon", "--service"])
                .arg(device.join("service.json"))
                .args(["--approve", &approved]);
            Daemon::launch_with_arguments(&device, process)
        };
        let mut worker = start();
        let deadline = Instant::now() + Duration::from_secs(10);
        let mut observed = loop {
            let value = worker.admin(&["cluster", "service"]);
            let report: serde_json::Value =
                serde_json::from_str(value.strip_prefix("SERVICE: ").unwrap().trim()).unwrap();
            if report["state"] == "ready_unqualified" || report["state"] == "failed_pins_retained" {
                break report;
            }
            assert!(
                Instant::now() < deadline,
                "{mode}: service stalled: {report}"
            );
        };
        assert_eq!(observed["inference_qualified"], false);
        assert_eq!(observed["fixture"], true);
        if mode == "registry-loss" {
            assert_eq!(observed["state"], "ready_unqualified");
            mutation_once(
                &authority.state,
                &["cluster", "revoke", &value(&prepared, "node=")],
            );
            let deadline = Instant::now() + Duration::from_secs(8);
            observed = loop {
                let text = worker.admin(&["cluster", "service"]);
                let report: serde_json::Value =
                    serde_json::from_str(text.strip_prefix("SERVICE: ").unwrap().trim()).unwrap();
                if report["state"] == "failed_pins_retained"
                    || report["state"] == "stopped_pins_retained"
                {
                    break report;
                }
                assert!(
                    Instant::now() < deadline,
                    "warm registry loss not observed: {report}"
                );
            };
            assert_eq!(observed["error"], "REGISTRY_LOST", "{observed}");
            assert_eq!(observed["registry_failure"], "revoked", "{observed}");
            assert_eq!(observed["cleanup_proven"], false);
        }
        if mode == "startup-crash" {
            assert!(
                observed["startup_error"]
                    .as_str()
                    .is_some_and(|error| error.starts_with("BACKEND_")),
                "startup reason must survive cleanup reporting: {observed}"
            );
        }
        let expected = match mode {
            "unapproved" | "wrong-executable" => Some("Unapproved"),
            "wrong-path" => Some("SERVICE_WEIGHTS_NOT_BOUND"),
            "wrong-profile" => Some("SERVICE_PROFILE_MISMATCH"),
            _ => None,
        };
        if let Some(expected) = expected {
            assert!(
                observed["error"].as_str().unwrap().contains(expected),
                "{observed}"
            );
            assert!(observed["pin"].is_null());
            assert_eq!(
                observed["startup_phase"],
                match mode {
                    "unapproved" => "configuration_approval",
                    "wrong-path" => "asset_binding",
                    "wrong-profile" => "profile_binding",
                    "wrong-executable" => "executable_digest",
                    _ => unreachable!(),
                }
            );
        }
        assert!(
            worker
                .admin(&["cluster", "status"])
                .contains("inference=unavailable")
        );
        if mode == "normal" {
            assert_eq!(observed["state"], "ready_unqualified");
            assert_eq!(observed["backend_version_declared"], "synthetic-v1");
            assert_eq!(
                observed["execution_profile_declared"],
                manifest.execution_profile
            );
            assert_eq!(observed["startup_properties"]["build_info"], "synthetic-v1");
            assert_eq!(observed["startup_properties"]["context_tokens"], 512);
            assert!(!observed["inference_qualified"].as_bool().unwrap());
            assert!(!observed.to_string().contains("not-retained"));
            assert!(
                Catalog::open(&cache).is_err(),
                "active service must retain exclusive catalog ownership"
            );
            drain_once(&worker);
            let deadline = Instant::now() + Duration::from_secs(8);
            loop {
                let state = worker.admin(&["cluster", "service"]);
                if state.contains("stopped_pins_retained") || state.contains("failed_pins_retained")
                {
                    break;
                }
                assert!(Instant::now() < deadline, "drain stalled: {state}");
            }
        } else {
            assert_eq!(observed["state"], "failed_pins_retained");
        }
        mutation_once(&worker.state, &["cluster", "stop"]);
        worker.wait();
        let mut catalog = Catalog::open(&cache).unwrap();
        if matches!(mode, "normal" | "startup-crash" | "registry-loss") {
            assert!(catalog.ensure_no_service().is_err());
            assert_eq!(catalog.gc().unwrap(), 0);
            assert!(weight.exists());
            drop(catalog);
            if mode == "registry-loss" {
                // Revocation prevents registry admission before any catalog restart check.
                continue;
            }
            let mut restarted = start();
            let deadline = Instant::now() + Duration::from_secs(8);
            loop {
                let state = restarted.admin(&["cluster", "service"]);
                if state.contains("SERVICE_CLEANUP_REQUIRED") {
                    break;
                }
                assert!(
                    Instant::now() < deadline,
                    "restart not quarantined: {state}"
                );
            }
            mutation_once(&restarted.state, &["cluster", "stop"]);
            restarted.wait();
        } else {
            catalog.ensure_no_service().unwrap();
            assert_eq!(catalog.gc().unwrap(), manifest.assets.len() as u64);
        }
    }
    mutation_once(&authority.state, &["cluster", "stop"]);
    authority.wait();
}

/// One stage route per layer range; the operator approves each stage's actual `describe`
/// (it loads nothing). Returns the routes, the model digest and the model's layer count.
#[cfg(feature = "test-backend")]
fn pipeline_stages(
    engine: &Path,
    model: &Path,
    environment: &std::collections::BTreeMap<String, String>,
    ranges: [[u32; 2]; 2],
    workload: (u32, u32, u32, u32),
) -> (
    Vec<starlings_cli::routes::StageRoute>,
    starlings_protocol::Digest,
    u32,
) {
    pipeline_stages_with(engine, model, environment, ranges, workload, 1)
}
/// As `pipeline_stages`, each stage serving `sessions` requests at once (GLY-190).
#[cfg(feature = "test-backend")]
fn pipeline_stages_with(
    engine: &Path,
    model: &Path,
    environment: &std::collections::BTreeMap<String, String>,
    ranges: [[u32; 2]; 2],
    (context, chunk, max_prompt, max_output): (u32, u32, u32, u32),
    sessions: u32,
) -> (
    Vec<starlings_cli::routes::StageRoute>,
    starlings_protocol::Digest,
    u32,
) {
    use starlings_adapters::partial::glyd;
    let model = if engine == PathBuf::from(BIN).canonicalize().unwrap() {
        let fixture = model.join("fixture-model");
        fs::create_dir_all(&fixture).unwrap();
        for (name, bytes) in [
            ("config.json", "{}"),
            ("tokenizer.json", "{}"),
            ("model.safetensors", "synthetic weights"),
        ] {
            fs::write(fixture.join(name), bytes).unwrap();
        }
        fixture
    } else {
        model.to_path_buf()
    };
    let manifest = starlings_cli::model_files::digest(&model).unwrap();
    let build = starlings_adapters::service::executable_digest(engine).unwrap();
    let gpu = environment
        .get("GLYD_GPU_LIB")
        .map(|lib| starlings_adapters::service::executable_digest(Path::new(lib)).unwrap());
    let port = || {
        TcpListener::bind("127.0.0.1:0")
            .unwrap()
            .local_addr()
            .unwrap()
    };
    let private = std::env::temp_dir().join("starlings-describe-engine");
    let mut descriptors = Vec::new();
    let stages = ranges
        .into_iter()
        .map(|layers| {
            let mut stage = starlings_cli::routes::StageRoute {
                version: starlings_protocol::Version,
                listen: port(),
                data_listen: port(),
                executable: engine.into(),
                executable_sha256: build.clone(),
                model: model.clone(),
                model_files_sha256: manifest.clone(),
                layers,
                context_tokens: context,
                chunk_tokens: chunk,
                max_prompt_tokens: max_prompt,
                max_output_tokens: max_output,
                descriptor: starlings_protocol::byte_digest(b"replaced below"),
                execution_profile: "glyd-range".into(),
                boundary_profile: "qwen35-residual-mlp-bf16".into(),
                backend_id: "glyd-rangeworker".into(),
                device_id: "cuda:0".into(),
                engine_bytes_limit: starlings_protocol::Decimal(15 << 30),
                environment: environment.clone(),
                gpu_library_sha256: gpu.clone(),
                // Room for drafted steps on the last stage (GLY-360).
                head_rows: starlings_protocol::MAX_DECODE_ROWS.min(chunk),
                sessions,
                more_data_listen: (1..sessions).map(|_| port()).collect(),
            };
            let (process, mut stdout, mut stdin) = glyd::RangeProcess::spawn(
                engine,
                &build,
                &stage.arguments(),
                environment,
                gpu.as_ref(),
                &private,
            )
            .unwrap();
            let d = glyd::describe(&mut stdout, &mut stdin).unwrap();
            process.stop();
            stage.descriptor = glyd::descriptor_digest(&d).unwrap();
            descriptors.push(d);
            stage
        })
        .collect();
    let layers = descriptors[0]["model"]["layers"].as_u64().unwrap() as u32;
    (stages, manifest, layers)
}

#[cfg(feature = "test-backend")]
struct PipelineCluster {
    _temp: tempfile::TempDir,
    authority: Daemon,
    workers: Vec<Daemon>,
    nodes: Vec<String>,
    token: starlings_security::ClientToken,
    manifest: starlings_protocol::Digest,
    address: String,
    /// The coordinator's web UI (GLY-364): its address and this launch's secret.
    ui: (String, String),
}
/// A coordinator, two stage workers and a client, enrolled through the CLI. The coordinator
/// restarts with the pipeline route, then both stage daemons start and advertise.
#[cfg(feature = "test-backend")]
fn pipeline_cluster(
    stages: &[starlings_cli::routes::StageRoute],
    manifest: &starlings_protocol::Digest,
    layers: u32,
    lifetime_ms: u32,
    stop_tokens: Vec<u32>,
    link: Option<Duration>,
    draft_tokens: u32,
) -> PipelineCluster {
    use starlings_cli::routes::{PipelineRoute, PipelineStage};
    use starlings_protocol::*;
    use starlings_security::{ClientToken, PrivateDir};
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().canonicalize().unwrap();
    let coordinator = root.join("coordinator");
    ok(
        &coordinator,
        &["cluster", "init", "--listen", "127.0.0.1:0"],
    );
    let mut authority = Daemon::start(&coordinator);
    let mut nodes = Vec::new();
    for (role, name) in [
        ("worker", "stage0"),
        ("worker", "stage1"),
        ("client", "client"),
    ] {
        let state = root.join(name);
        let invite = coordinator.join(format!("{name}.json"));
        invitation_once(&authority.state, role, &invite);
        let prepared = ok(&state, &["cluster", "prepare", invite.to_str().unwrap()]);
        let node = value(&prepared, "node=");
        authority.admin(&[
            "cluster",
            "approve",
            &node,
            &value(&prepared, "fingerprint="),
        ]);
        thread::sleep(Duration::from_millis(220));
        ok(&state, &["cluster", "join", invite.to_str().unwrap()]);
        nodes.push(node);
    }
    let saved: serde_json::Value =
        serde_json::from_slice(&fs::read(coordinator.join("config.json")).unwrap()).unwrap();
    let files = PrivateDir::open(&coordinator).unwrap();
    for (node, action) in [
        (saved["node"].as_str().unwrap(), "control"),
        (nodes[0].as_str(), "infer"),
        (nodes[1].as_str(), "infer"),
        (nodes[2].as_str(), "infer"),
    ] {
        files
            .write_atomic_durable(
                "scopes.json",
                &serde_json::to_vec(&serde_json::json!([{"model": manifest, "action": action}]))
                    .unwrap(),
            )
            .unwrap();
        mutation_once(
            &authority.state,
            &[
                "cluster",
                "permissions",
                node,
                coordinator.join("scopes.json").to_str().unwrap(),
            ],
        );
    }
    let token_file = coordinator.join("token.json");
    mutation_once(
        &coordinator,
        &[
            "cluster",
            "token",
            &nodes[2],
            coordinator.join("scopes.json").to_str().unwrap(),
            "900",
            token_file.to_str().unwrap(),
        ],
    );
    observe_active(
        &coordinator,
        serde_json::json!({"command":"token_status","input":token_file}),
        "TOKEN_STATUS",
    );
    let token = ClientToken::load(&files, "token.json").unwrap();
    mutation_once(&coordinator, &["cluster", "stop"]);
    authority.wait();

    let mut plan: PlacementPlan = decode(
        include_bytes!("../../../examples/placement-plan.json"),
        FixtureMode::Simulation,
    )
    .unwrap();
    plan.fixture = false;
    plan.cluster_id = saved["cluster"].as_str().unwrap().parse().unwrap();
    plan.manifest_digest = manifest.clone();
    plan.execution_profile = "glyd-range".into();
    plan.boundary_profile = "qwen35-residual-mlp-bf16".into();
    plan.context_limit = BoundedU32::new(stages[0].context_tokens).unwrap();
    plan.model_layer_count = BoundedU32::new(layers).unwrap();
    for (i, stage) in plan.stages.iter_mut().enumerate() {
        stage.node_id = nodes[i].parse().unwrap();
        stage.device_id = "cuda:0".into();
        stage.memory_pool_id = "cuda:0".into();
        stage.backend_id = "glyd-rangeworker".into();
        stage.backend_build_digest = stages[i].executable_sha256.clone();
        stage.layer_start = BoundedU32::new(stages[i].layers[0]).unwrap();
        stage.layer_end_exclusive = BoundedU32::new(stages[i].layers[1]).unwrap();
        stage.resident_bytes = Decimal(6 << 30);
        stage.state_budget_bytes = Decimal(256 << 20);
        stage.scratch_budget_bytes = Decimal(64 << 20);
        stage.safety_reserve_bytes = Decimal(512 << 20);
    }
    plan.plan_digest =
        canonical_digest(&serde_json::to_value(&plan).unwrap(), "plan_digest").unwrap();
    let pipeline = PipelineRoute {
        version: Version,
        listen: "127.0.0.1:0".parse().unwrap(),
        plan,
        stages: stages
            .iter()
            .map(|s| PipelineStage {
                endpoint: s.listen,
                data: link.map_or(s.data_listen, |one_way| delay_proxy(s.data_listen, one_way)),
                more_data: s
                    .more_data_listen
                    .iter()
                    .map(|&a| link.map_or(a, |one_way| delay_proxy(a, one_way)))
                    .collect(),
            })
            .collect(),
        chunk_tokens: stages[0].chunk_tokens,
        max_output_tokens: stages[0].max_output_tokens,
        lifetime_ms,
        stop_tokens,
        draft_tokens,
    };
    let launch = |state: &Path, flag: &str, record: serde_json::Value, more: &[&str]| {
        PrivateDir::open(state)
            .unwrap()
            .write_atomic_durable("launch.json", &serde_json::to_vec(&record).unwrap())
            .unwrap();
        let mut process = Command::new(BIN);
        process
            .args(["daemon", flag])
            .arg(state.join("launch.json"))
            .args(["--approve", content_hash(&record).unwrap().as_str()])
            .args(more);
        Daemon::launch_with_arguments(state, process)
    };
    // The coordinator also serves its web UI, whose chat acts as the client.
    let authority = launch(
        &coordinator,
        "--pipeline",
        serde_json::to_value(&pipeline).unwrap(),
        &["--ui", "127.0.0.1:0", "--ui-client", &nodes[2]],
    );
    let workers: Vec<Daemon> = stages
        .iter()
        .zip(["stage0", "stage1"])
        .map(|(stage, name)| {
            launch(
                &root.join(name),
                "--stage",
                serde_json::to_value(stage).unwrap(),
                &[],
            )
        })
        .collect();
    for (worker, node) in workers.iter().zip(&nodes) {
        let deadline = Instant::now() + Duration::from_secs(120);
        while stage_state(worker)["state"] != "ready" {
            assert!(
                Instant::now() < deadline,
                "stage not ready: {}",
                stage_state(worker)
            );
            thread::sleep(Duration::from_millis(500));
        }
        let deadline = Instant::now() + Duration::from_secs(20);
        loop {
            let view = authority.admin(&["cluster", "resources", node]);
            let view: serde_json::Value =
                serde_json::from_str(view.strip_prefix("RESOURCES: ").unwrap().trim()).unwrap();
            if view["sample"]["capabilities"]["state"] == "ready" {
                break;
            }
            assert!(Instant::now() < deadline, "no ready sample: {view}");
        }
    }
    let address = value(&authority.admin(&["cluster", "status"]), "inference=");
    let ui = fs::read_to_string(&authority.log)
        .unwrap()
        .lines()
        .find_map(|l| l.strip_prefix("UI_READY: http://"))
        .and_then(|l| l.split_once("/#k="))
        .map(|(address, key)| (address.to_owned(), key.to_owned()))
        .expect("UI_READY line");
    PipelineCluster {
        _temp: temp,
        authority,
        workers,
        nodes,
        token,
        manifest: manifest.clone(),
        address,
        ui,
    }
}
/// A TCP relay to `target` that holds every byte for `one_way` in each direction: a link with
/// that latency (and no bandwidth limit) for the stages' data connections.
#[cfg(feature = "test-backend")]
fn delay_proxy(target: std::net::SocketAddr, one_way: Duration) -> std::net::SocketAddr {
    use std::io::Write;
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    thread::spawn(move || {
        for client in listener.incoming() {
            let (Ok(client), Ok(server)) = (client, TcpStream::connect(target)) else {
                continue;
            };
            for socket in [&client, &server] {
                socket.set_nodelay(true).unwrap();
            }
            let pairs = [
                (client.try_clone().unwrap(), server.try_clone().unwrap()),
                (server, client),
            ];
            for (mut from, mut to) in pairs {
                let (queue, queued) = std::sync::mpsc::channel::<(Instant, Vec<u8>)>();
                thread::spawn(move || {
                    let mut buffer = vec![0; 64 << 10];
                    while let Ok(n @ 1..) = from.read(&mut buffer) {
                        if queue
                            .send((Instant::now() + one_way, buffer[..n].to_vec()))
                            .is_err()
                        {
                            break;
                        }
                    }
                });
                thread::spawn(move || {
                    for (due, bytes) in queued {
                        thread::sleep(due.saturating_duration_since(Instant::now()));
                        if to.write_all(&bytes).is_err() {
                            break;
                        }
                    }
                    let _ = to.shutdown(std::net::Shutdown::Write);
                });
            }
        }
    });
    address
}
/// STARLINGS_RANGE_LINK_MS: a round trip added to the stages' data connections.
#[cfg(feature = "test-backend")]
fn link_delay() -> Option<Duration> {
    let ms: u64 = std::env::var("STARLINGS_RANGE_LINK_MS")
        .ok()?
        .parse()
        .unwrap();
    Some(Duration::from_micros(ms * 500))
}
#[cfg(feature = "test-backend")]
fn stage_state(worker: &Daemon) -> serde_json::Value {
    let reply = worker.admin(&["cluster", "service"]);
    serde_json::from_str(reply.strip_prefix("SERVICE: ").unwrap().trim()).unwrap()
}
#[cfg(feature = "test-backend")]
impl PipelineCluster {
    /// One chat request of one user message; the whole reply (headers and body) and its time.
    fn chat(
        &self,
        text: &str,
        max_tokens: u32,
        sampling: (f64, f64, u32),
        stream: bool,
    ) -> (String, Duration) {
        let messages = serde_json::json!([{"role": "user", "content": text}]);
        self.chat_turns(messages, max_tokens, sampling, stream)
    }
    /// One chat request of these turns.
    fn chat_turns(
        &self,
        messages: serde_json::Value,
        max_tokens: u32,
        sampling: (f64, f64, u32),
        stream: bool,
    ) -> (String, Duration) {
        use std::io::Write;
        let (temperature, top_p, seed) = sampling;
        let body = serde_json::to_vec(&serde_json::json!({
            "model": self.manifest, "messages": messages,
            "max_tokens": max_tokens, "temperature": temperature, "top_p": top_p, "seed": seed,
            "stream": stream,
        }))
        .unwrap();
        let address = &self.address;
        let mut socket = TcpStream::connect(address).unwrap();
        socket
            .set_read_timeout(Some(Duration::from_secs(90)))
            .unwrap();
        write!(socket, "POST /v1/chat/completions HTTP/1.1\r\nHost: {address}\r\nAuthorization: Bearer {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n", self.token.bearer(), body.len()).unwrap();
        socket.write_all(&body).unwrap();
        let started = Instant::now();
        let mut reply = String::new();
        socket.read_to_string(&mut reply).unwrap();
        (reply, started.elapsed())
    }
    /// Every daemon's log (the stages', then the coordinator's), for failure messages.
    fn logs(&self) -> String {
        self.workers
            .iter()
            .chain(std::iter::once(&self.authority))
            .map(|w| fs::read_to_string(&w.log).unwrap_or_default())
            .collect::<Vec<_>>()
            .join("\n---\n")
    }
    fn reports(&self) -> Vec<serde_json::Value> {
        fs::read_to_string(&self.authority.log)
            .unwrap()
            .lines()
            .filter_map(|l| l.strip_prefix("PIPELINE_REPORT "))
            .map(|l| serde_json::from_str(l).unwrap())
            .collect()
    }
    /// Stop every daemon; each must exit.
    fn stop(self) {
        for daemon in self.workers.iter().chain(std::iter::once(&self.authority)) {
            mutation_once(&daemon.state, &["cluster", "stop"]);
        }
        for mut daemon in self
            .workers
            .into_iter()
            .chain(std::iter::once(self.authority))
        {
            daemon.wait();
        }
    }
}
#[cfg(feature = "test-backend")]
fn sse_text(sse: &str) -> String {
    sse.lines()
        .filter_map(|l| l.strip_prefix("data: "))
        .filter(|l| *l != "[DONE]")
        .filter_map(|l| serde_json::from_str::<serde_json::Value>(l).ok())
        .filter_map(|e| {
            e["choices"][0]["delta"]["content"]
                .as_str()
                .map(str::to_owned)
        })
        .collect()
}

/// GLY-360 with the stand-in engine's history model ("REPEAT"): drafted decode gives the same
/// tokens as plain decode, some drafts hold and some miss (each miss rewinds both stages), and
/// both stages stop at the same token.
#[cfg(feature = "test-backend")]
#[test]
fn pipeline_fake_engine_drafts_give_the_plain_answer() {
    let temp = tempfile::tempdir().unwrap();
    let model = temp.path().canonicalize().unwrap();
    let engine = PathBuf::from(BIN).canonicalize().unwrap();
    let (stages, manifest, layers) = pipeline_stages(
        &engine,
        &model,
        &Default::default(),
        [[0, 1], [1, 2]],
        (128, 4, 32, 40),
    );
    let mut answers = Vec::new();
    for drafts in [0, 4] {
        let cluster = pipeline_cluster(&stages, &manifest, layers, 30_000, vec![], None, drafts);
        let (sse, _) = cluster.chat("REPEAT abc abd", 30, (0.0, 1.0, 1), true);
        assert!(
            sse.trim_end().ends_with("data: [DONE]"),
            "{sse}\n{}",
            cluster.logs()
        );
        let deadline = Instant::now() + Duration::from_secs(5);
        while cluster.reports().is_empty() {
            assert!(Instant::now() < deadline, "no report");
            thread::sleep(Duration::from_millis(20));
        }
        let report = cluster.reports().pop().unwrap();
        let [head, tail] = [&report["stages"][0], &report["stages"][1]];
        assert_eq!(head["tokens"], 30);
        assert_eq!(head["tokens_sha256"], tail["tokens_sha256"]);
        assert_eq!(head["ended_tokens"], tail["ended_tokens"]);
        assert_eq!(
            (&head["drafted"], &head["accepted"]),
            (&tail["drafted"], &tail["accepted"])
        );
        let (drafted, accepted) = (
            head["drafted"].as_u64().unwrap(),
            head["accepted"].as_u64().unwrap(),
        );
        eprintln!(
            "DRAFTS drafts={drafts} decode_steps={} drafted={drafted} accepted={accepted}",
            head["decode_steps"]
        );
        if drafts > 0 {
            assert!(
                accepted > 0 && accepted < drafted,
                "{drafted} drafted, {accepted} held"
            );
            assert!(head["decode_steps"].as_u64().unwrap() < 29);
        }
        answers.push((sse_text(&sse), head["tokens_sha256"].clone()));
        if drafts == 0 {
            // GLY-361: a system prompt and history go through the model's template too, and
            // the history changes the answer.
            let turns = serde_json::json!([
                {"role": "system", "content": "REPEAT"},
                {"role": "user", "content": "ab"},
                {"role": "assistant", "content": "cd"},
                {"role": "user", "content": "abc abd"},
            ]);
            let (sse, _) = cluster.chat_turns(turns, 30, (0.0, 1.0, 1), true);
            assert!(
                sse.trim_end().ends_with("data: [DONE]"),
                "{sse}\n{}",
                cluster.logs()
            );
            assert_ne!(sse_text(&sse), answers[0].0);
            // The last turn must be the user's.
            let turns = serde_json::json!([
                {"role": "user", "content": "REPEAT"},
                {"role": "assistant", "content": "an answer"},
            ]);
            let (refused, _) = cluster.chat_turns(turns, 30, (0.0, 1.0, 1), true);
            assert!(!refused.trim_end().ends_with("data: [DONE]"), "{refused}");
            thread::sleep(Duration::from_secs(6));
        }
        cluster.stop();
    }
    assert_eq!(answers[0], answers[1]);
}

/// GLY-364: the coordinator's web UI over the stand-in pipeline. Its state shows the layer split
/// and both stages' samples; chat through it answers as the client API does for the same prompt,
/// and its last chunk carries that request's report; a wrong Host or secret is refused.
#[cfg(feature = "test-backend")]
#[test]
fn pipeline_fake_engine_web_ui_shows_the_split_and_chats_like_the_api() {
    use std::io::Write;
    let temp = tempfile::tempdir().unwrap();
    let model = temp.path().canonicalize().unwrap();
    let engine = PathBuf::from(BIN).canonicalize().unwrap();
    let (stages, manifest, layers) = pipeline_stages(
        &engine,
        &model,
        &Default::default(),
        [[0, 1], [1, 2]],
        (128, 4, 32, 40),
    );
    let cluster = pipeline_cluster(&stages, &manifest, layers, 30_000, vec![], None, 4);
    let (address, key) = cluster.ui.clone();
    let ui = |request: &str, host: &str, secret: Option<&str>, body: &str| {
        let mut socket = TcpStream::connect(&address).unwrap();
        socket
            .set_read_timeout(Some(Duration::from_secs(90)))
            .unwrap();
        let secret = secret.map_or(String::new(), |k| format!("X-Starlings-UI: {k}\r\n"));
        write!(socket, "{request} HTTP/1.1\r\nHost: {host}\r\n{secret}Content-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}", body.len()).unwrap();
        let mut reply = String::new();
        socket.read_to_string(&mut reply).unwrap();
        reply
    };
    let port = address.rsplit_once(':').unwrap().1;
    for (host, secret, status) in [
        ("attacker.example", Some(key.as_str()), "403"),
        (address.as_str(), None, "401"),
        (address.as_str(), Some("0".repeat(64).as_str()), "401"),
    ] {
        let reply = ui("GET /ui/v1/state", host, secret, "");
        assert!(reply.starts_with(&format!("HTTP/1.1 {status}")), "{reply}");
    }
    let page = ui("GET /", &format!("localhost:{port}"), None, "");
    assert!(page.starts_with("HTTP/1.1 200 OK") && page.contains("<title>Starlings</title>"));

    let state = |reply: String| -> serde_json::Value {
        assert!(reply.starts_with("HTTP/1.1 200 OK"), "{reply}");
        serde_json::from_str(reply.split_once("\r\n\r\n").unwrap().1).unwrap()
    };
    // The control loop refreshes the snapshot once a second: wait for both stages' ready samples.
    let deadline = Instant::now() + Duration::from_secs(10);
    let view = loop {
        let view = state(ui("GET /ui/v1/state", &address, Some(&key), ""));
        let ready = view["nodes"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|n| n["resources"]["sample"]["capabilities"]["state"] == "ready");
        if ready.count() == 2 {
            break view;
        }
        assert!(Instant::now() < deadline, "{view}");
        thread::sleep(Duration::from_millis(200));
    };
    let route = &view["route"];
    assert_eq!(view["chat"], true);
    assert_eq!(route["model"], manifest.as_str());
    assert_eq!(route["draft_tokens"], 4);
    assert_eq!(route["listen"], cluster.address.as_str());
    for (i, stage) in route["stages"].as_array().unwrap().iter().enumerate() {
        assert_eq!(stage["node"], cluster.nodes[i].as_str());
        assert_eq!(
            [&stage["layer_start"], &stage["layer_end"]],
            stages[i].layers.map(serde_json::Value::from).each_ref()
        );
        assert_eq!(
            (stage["embedding"] == true, stage["head"] == true),
            (i == 0, i == 1)
        );
        let node = view["nodes"]
            .as_array()
            .unwrap()
            .iter()
            .find(|n| n["node"] == stage["node"])
            .unwrap();
        assert_eq!(node["status"], "online");
        let sample = &node["resources"]["sample"]["capabilities"];
        assert_eq!(sample["state"], "ready", "{view}");
        assert!(
            sample["memory_pools"]
                .as_array()
                .unwrap()
                .iter()
                .any(|p| p["kind"] == "device")
        );
    }
    assert_eq!(
        view["join"]["state_dir"].as_str().map(PathBuf::from),
        Some(cluster.authority.state.clone())
    );

    let prompt = "REPEAT abc abd";
    let (api, _) = cluster.chat(prompt, 30, (0.0, 1.0, 1), true);
    let body = serde_json::json!({"messages": [{"role": "user", "content": prompt}],
        "max_tokens": 30, "temperature": 0.0, "seed": 1});
    let sse = ui("POST /ui/v1/chat", &address, Some(&key), &body.to_string());
    assert!(
        sse.starts_with("HTTP/1.1 200 OK"),
        "{sse}\n{}",
        cluster.logs()
    );
    assert!(sse.trim_end().ends_with("data: [DONE]"), "{sse}");
    assert_eq!(sse_text(&sse), sse_text(&api));
    assert_eq!(sse_text(&sse).len(), 30);
    let last: serde_json::Value = sse
        .lines()
        .filter_map(|l| l.strip_prefix("data: "))
        .filter_map(|l| serde_json::from_str::<serde_json::Value>(l).ok())
        .find(|e| e["choices"][0]["finish_reason"].is_string())
        .unwrap();
    let report = &last["starlings"];
    assert_eq!(report["id"], last["id"]);
    assert_eq!(report["stages"][0]["tokens"], 30);
    assert_eq!(
        report["stages"][0]["tokens_sha256"],
        report["stages"][1]["tokens_sha256"]
    );
    assert!(report["first_text_ms"].as_f64().unwrap() > 0.0);
    assert!(report["stages"][0]["drafted"].as_u64().unwrap() > 0);
    // Both answers are in the state's reports, the UI's last.
    let view = state(ui("GET /ui/v1/state", &address, Some(&key), ""));
    let reports = view["reports"].as_array().unwrap();
    assert_eq!(reports.len(), 2);
    assert_eq!(reports[1]["id"], last["id"]);
    assert_eq!(
        reports[0]["stages"][0]["tokens_sha256"],
        reports[1]["stages"][0]["tokens_sha256"]
    );
    cluster.stop();
}

/// GLY-190 with the stand-in engine: stages with two sessions serve two requests at once, each
/// in its own engine session and data slot, and each answer equals the same prompt run alone.
#[cfg(feature = "test-backend")]
#[test]
fn pipeline_fake_engine_serves_two_requests_at_once() {
    let temp = tempfile::tempdir().unwrap();
    let model = temp.path().canonicalize().unwrap();
    let engine = PathBuf::from(BIN).canonicalize().unwrap();
    let (stages, manifest, layers) = pipeline_stages_with(
        &engine,
        &model,
        &Default::default(),
        [[0, 1], [1, 2]],
        (128, 4, 32, 40),
        2,
    );
    let cluster = pipeline_cluster(&stages, &manifest, layers, 30_000, vec![], None, 4);
    let prompts = ["REPEAT abc abd", "REPEAT xyz xya"];
    let together: Vec<String> = thread::scope(|threads| {
        let cluster = &cluster;
        let runs: Vec<_> = prompts
            .iter()
            .map(|prompt| threads.spawn(move || cluster.chat(prompt, 30, (0.0, 1.0, 1), true).0))
            .collect();
        runs.into_iter().map(|run| run.join().unwrap()).collect()
    });
    for sse in &together {
        assert!(
            sse.trim_end().ends_with("data: [DONE]"),
            "{sse}\n{}",
            cluster.logs()
        );
    }
    for (prompt, sse) in prompts.iter().zip(&together) {
        thread::sleep(Duration::from_millis(200));
        let (alone, _) = cluster.chat(prompt, 30, (0.0, 1.0, 1), true);
        assert_eq!(sse_text(sse), sse_text(&alone), "{prompt}");
        assert_eq!(sse_text(sse).len(), 30);
    }
    assert_ne!(sse_text(&together[0]), sse_text(&together[1]));
    for worker in &cluster.workers {
        let state = stage_state(worker);
        assert_eq!(
            (&state["requests"], &state["failed_requests"]),
            (&4.into(), &0.into()),
            "{state}"
        );
    }
    cluster.stop();
}

/// The stages' data connections over a 40 ms round trip (a relay that delays every byte): each
/// decode step crosses the link once each way, so a token costs one round trip plus engine time.
#[cfg(feature = "test-backend")]
#[test]
fn pipeline_fake_engine_decodes_at_one_round_trip_a_token() {
    let temp = tempfile::tempdir().unwrap();
    let model = temp.path().canonicalize().unwrap();
    let engine = PathBuf::from(BIN).canonicalize().unwrap();
    let (stages, manifest, layers) = pipeline_stages(
        &engine,
        &model,
        &Default::default(),
        [[0, 1], [1, 2]],
        (128, 4, 32, 40),
    );
    let rtt = Duration::from_millis(40);
    let cluster = pipeline_cluster(&stages, &manifest, layers, 30_000, vec![], Some(rtt / 2), 0);
    let mut ms = Vec::new();
    for tokens in [4, 24] {
        let (reply, client) = cluster.chat("hello pipeline", tokens, (0.0, 1.0, 1), false);
        assert!(
            reply.starts_with("HTTP/1.1 200"),
            "{reply}\n{}",
            cluster.logs()
        );
        // The coordinator logs its report after the answer.
        let deadline = Instant::now() + Duration::from_secs(5);
        while cluster.reports().len() < ms.len() + 1 {
            assert!(Instant::now() < deadline, "no report");
            thread::sleep(Duration::from_millis(20));
        }
        let report = cluster.reports().pop().unwrap();
        assert_eq!(report["stages"][0]["tokens"], tokens);
        let pipeline = report["ms"].as_f64().unwrap();
        eprintln!(
            "LINK tokens={tokens} setup_ms={:.1} pipeline_ms={pipeline:.1} client_ms={:.1}",
            report["setup_ms"].as_f64().unwrap(),
            client.as_secs_f64() * 1e3
        );
        ms.push(pipeline);
        thread::sleep(Duration::from_millis(200));
    }
    let per_token = (ms[1] - ms[0]) / 20.0;
    eprintln!(
        "LINK per_token_ms={per_token:.1} rtt_ms={}",
        rtt.as_millis()
    );
    // Checked on Linux only: a hosted macOS runner measured 249 ms a token here (its sleeps
    // overshoot and the runner is shared), where Linux x64 and ARM64 runners measured 42-45.
    if cfg!(target_os = "linux") {
        assert!(
            per_token < 1.5 * rtt.as_secs_f64() * 1e3,
            "{per_token:.1} ms a token"
        );
    }
    cluster.stop();
}

/// GLY-188 with a stand-in engine (`starlings rangeworker` in test-backend builds): a request
/// streams through both stage daemons; a chunk that outlives the lease is cancelled and the
/// stage keeps serving; a request whose first stage hangs inside run_chunk fails within its
/// lease, the hung engine is killed, the stage stops advertising, and every daemon still stops.
#[cfg(feature = "test-backend")]
#[test]
fn pipeline_fake_engine_streams_then_a_hung_stage_fails_within_its_lease() {
    let temp = tempfile::tempdir().unwrap();
    let model = temp.path().canonicalize().unwrap();
    let engine = PathBuf::from(BIN).canonicalize().unwrap();
    let (stages, manifest, layers) = pipeline_stages(
        &engine,
        &model,
        &Default::default(),
        [[0, 1], [1, 2]],
        (64, 4, 32, 16),
    );
    let lifetime_ms = 5000;
    let cluster = pipeline_cluster(&stages, &manifest, layers, lifetime_ms, vec![], None, 0);
    let (sse, _) = cluster.chat("hello pipeline", 8, (0.0, 1.0, 1), true);
    assert!(sse.starts_with("HTTP/1.1 200"), "{sse}");
    assert!(
        sse.trim_end().ends_with("data: [DONE]"),
        "{sse}\n{}",
        cluster.logs()
    );
    assert_eq!(
        sse_text(&sse),
        "ffffffff",
        "eight copies of the fixture's token 5"
    );
    let report = &cluster.reports()[0];
    let [head, tail] = [&report["stages"][0], &report["stages"][1]];
    assert_eq!(report["finish"], "length");
    // "hello pipeline" is 15 ids; both stages ran the prompt in chunks of 4 and 7 decodes.
    for stage in [head, tail] {
        assert_eq!(stage["prefill_steps"], 4, "{stage}");
        assert_eq!(stage["decode_steps"], 7, "{stage}");
        assert_eq!(stage["rows"], 22, "{stage}");
        assert_eq!(stage["ended_tokens"], 22, "{stage}");
        assert_eq!(stage["session_closed"], true);
        assert_eq!(stage["tokens"], 8);
    }
    assert_eq!(head["tokens_sha256"], tail["tokens_sha256"]);

    // A first-stage chunk outlives the lease: the request fails in time, the chunk is
    // cancelled rather than killed, and the warm stage serves the next request.
    thread::sleep(Duration::from_millis(200));
    let (failed, elapsed) = cluster.chat("SLOW", 8, (0.0, 1.0, 1), true);
    assert!(failed.contains("event: error"), "{failed}");
    assert!(
        elapsed < Duration::from_millis(u64::from(lifetime_ms) + 3000),
        "failed after {elapsed:?}"
    );
    // The coordinator holds its slot for the stages' teardown after a failure.
    thread::sleep(Duration::from_secs(6));
    assert_eq!(
        stage_state(&cluster.workers[0])["state"],
        "ready",
        "{}",
        cluster.logs()
    );
    let (again, _) = cluster.chat("hello again", 8, (0.0, 1.0, 1), true);
    assert!(
        again.trim_end().ends_with("data: [DONE]"),
        "{again}\n{}",
        cluster.logs()
    );
    assert_eq!(sse_text(&again), "ffffffff");

    // The first stage's engine hangs in its prompt chunk.
    thread::sleep(Duration::from_millis(200));
    let (failed, elapsed) = cluster.chat("HANG", 8, (0.0, 1.0, 1), true);
    assert!(failed.contains("event: error"), "{failed}");
    assert!(
        elapsed < Duration::from_millis(u64::from(lifetime_ms) + 3000),
        "failed after {elapsed:?}"
    );
    let deadline = Instant::now() + Duration::from_secs(15);
    while stage_state(&cluster.workers[0])["state"] != "failed" {
        assert!(
            Instant::now() < deadline,
            "{}",
            stage_state(&cluster.workers[0])
        );
        thread::sleep(Duration::from_millis(250));
    }
    assert_eq!(
        stage_state(&cluster.workers[0])["error"],
        "STAGE_ENGINE_UNUSABLE: restart the daemon"
    );
    assert_eq!(stage_state(&cluster.workers[1])["state"], "ready");
    let deadline = Instant::now() + Duration::from_secs(15);
    loop {
        let view = cluster
            .authority
            .admin(&["cluster", "resources", &cluster.nodes[0]]);
        let view: serde_json::Value =
            serde_json::from_str(view.strip_prefix("RESOURCES: ").unwrap().trim()).unwrap();
        if view["sample"]["capabilities"]["state"] != "ready" {
            break;
        }
        assert!(Instant::now() < deadline, "failed stage still advertised");
    }
    cluster.stop();
}

/// GLY-188 on real hardware: a coordinator and two stage daemons serve one chat request through
/// two `glyd-engine rangeworker` stages, then the whole model runs the same prompt directly and
/// must give the same tokens. Runs only with STARLINGS_RANGE_ENGINE and STARLINGS_RANGE_MODEL
/// (GLYD_GPU_LIB is passed to the engine); the caller holds the GPU lock. Token digests are in
/// the coordinator's log only in test-backend builds.
#[cfg(feature = "test-backend")]
#[test]
fn real_pipeline_daemons_generate_through_two_stages_like_the_whole_model() {
    use starlings_adapters::partial::glyd::{self, Event, Output};
    use starlings_protocol::*;
    let (Ok(engine), Ok(model)) = (
        std::env::var("STARLINGS_RANGE_ENGINE"),
        std::env::var("STARLINGS_RANGE_MODEL"),
    ) else {
        return;
    };
    let new: u32 = std::env::var("STARLINGS_RANGE_NEW").map_or(48, |v| v.parse().unwrap());
    // A nonzero temperature draws on the last stage from (seed, step, logits), so the whole
    // model with the same settings must still give the same tokens.
    let temperature: f64 =
        std::env::var("STARLINGS_RANGE_TEMPERATURE").map_or(0.0, |v| v.parse().unwrap());
    let (top_p, seed) = (if temperature == 0.0 { 1.0 } else { 0.9 }, 7);
    let text = std::env::var("STARLINGS_RANGE_TEXT")
        .unwrap_or_else(|_| "Explain in three sentences why the sky is blue.".into());
    let engine = PathBuf::from(engine).canonicalize().unwrap();
    let model = PathBuf::from(model).canonicalize().unwrap();
    let environment: std::collections::BTreeMap<String, String> = std::env::var("GLYD_GPU_LIB")
        .into_iter()
        .map(|lib| ("GLYD_GPU_LIB".to_owned(), lib))
        .collect();
    let (context, chunk, max_prompt, max_output) = (1024, 32, 768, 256);
    // STARLINGS_RANGE_SESSIONS=2: each stage serves two requests at once (GLY-190).
    let sessions: u32 = std::env::var("STARLINGS_RANGE_SESSIONS").map_or(1, |v| v.parse().unwrap());
    let (stages, manifest, layers) = pipeline_stages_with(
        &engine,
        &model,
        &environment,
        [[0, 16], [16, 32]],
        (context, chunk, max_prompt, max_output),
        sessions,
    );
    assert_eq!(layers, 32, "stage ranges are for the 9B");
    let build = stages[0].executable_sha256.clone();
    // The model's end-of-text and end-of-turn IDs end an answer.
    let config: serde_json::Value =
        serde_json::from_slice(&fs::read(model.join("config.json")).unwrap()).unwrap();
    let tokenizer: serde_json::Value =
        serde_json::from_slice(&fs::read(model.join("tokenizer_config.json")).unwrap()).unwrap();
    let mut stop_tokens: Vec<u32> = tokenizer["added_tokens_decoder"]
        .as_object()
        .unwrap()
        .iter()
        .filter(|(_, t)| ["<|endoftext|>", "<|im_end|>"].contains(&t["content"].as_str().unwrap()))
        .map(|(id, _)| id.parse().unwrap())
        .collect();
    if let Some(eos) = config["text_config"]["eos_token_id"].as_u64() {
        stop_tokens.push(eos as u32);
    }
    stop_tokens.sort_unstable();
    stop_tokens.dedup();

    let cluster = pipeline_cluster(
        &stages,
        &manifest,
        layers,
        60_000,
        stop_tokens.clone(),
        link_delay(),
        std::env::var("STARLINGS_RANGE_DRAFTS").map_or(0, |v| v.parse().unwrap()),
    );
    let (sse, elapsed) = cluster.chat(&text, new, (temperature, top_p, seed), true);
    assert!(sse.starts_with("HTTP/1.1 200"), "{sse}");
    assert!(sse.trim_end().ends_with("data: [DONE]"), "{sse}");
    let answer = sse_text(&sse);
    // The non-streaming answer is the same text: the same settings through the same stages.
    thread::sleep(Duration::from_millis(200));
    let (json, _) = cluster.chat(&text, new, (temperature, top_p, seed), false);
    let body: serde_json::Value =
        serde_json::from_str(json.split("\r\n\r\n").nth(1).unwrap()).unwrap();
    assert_eq!(body["choices"][0]["message"]["content"], answer.as_str());
    let reports = cluster.reports();
    assert_eq!(reports.len(), 2, "{reports:?}");
    let report = reports[0].clone();
    let [head, tail] = [&report["stages"][0], &report["stages"][1]];
    for (stage, layers) in [(head, [0, 16]), (tail, [16, 32])] {
        assert_eq!(stage["layers"], serde_json::json!(layers));
        assert_eq!(stage["session_closed"], true, "{stage}");
        assert_eq!(stage["refused_connections"], 0, "{stage}");
        assert!(stage["rows"].as_u64().unwrap() > 0);
    }
    assert_eq!(head["first"], true);
    assert_eq!(tail["last"], true);
    assert_eq!(head["tokens_sha256"], tail["tokens_sha256"]);
    let tokens = head["tokens"].as_u64().unwrap() as u32;
    let prompt_len = head["prompt_tokens"].as_u64().unwrap() as u32;
    // Both stages ran the prompt, then each decode step's token and its drafts (GLY-360).
    let number = |v: &serde_json::Value| v.as_u64().unwrap();
    let (rows, ended) = (number(&head["rows"]), number(&head["ended_tokens"]));
    let drafted = number(&head["drafted"]);
    assert_eq!(
        rows,
        u64::from(prompt_len) + number(&head["decode_steps"]) + drafted
    );
    assert_eq!(
        (&tail["rows"], &tail["ended_tokens"]),
        (&head["rows"], &head["ended_tokens"])
    );
    if drafted == 0 {
        // Every answer token but the last, and the ended session held exactly those.
        assert_eq!(rows, u64::from(prompt_len + tokens - 1));
        assert_eq!(ended, rows);
    } else {
        // The ended session held the kept positions and the last step's rows past them.
        assert!((u64::from(prompt_len + tokens - 1)..=rows).contains(&ended));
    }
    assert_eq!(
        reports[1]["stages"][0]["tokens_sha256"],
        head["tokens_sha256"]
    );
    for worker in &cluster.workers {
        let state = stage_state(worker);
        assert_eq!(state["requests"], 2, "{state}");
        assert_eq!(state["failed_requests"], 0, "{state}");
    }
    // Two requests at once: each answer equals the same prompt run alone afterwards.
    let mut together = serde_json::Value::Null;
    if sessions == 2 {
        let prompts = [
            text.as_str(),
            "Name three rivers in Europe and one fact about each.",
        ];
        let started = Instant::now();
        let both: Vec<String> = thread::scope(|threads| {
            let cluster = &cluster;
            let runs: Vec<_> = prompts
                .iter()
                .map(|p| {
                    threads.spawn(move || cluster.chat(p, new, (temperature, top_p, seed), true).0)
                })
                .collect();
            runs.into_iter().map(|r| r.join().unwrap()).collect()
        });
        let at_once = started.elapsed();
        let started = Instant::now();
        let alone: Vec<String> = prompts
            .iter()
            .map(|p| cluster.chat(p, new, (temperature, top_p, seed), true).0)
            .collect();
        let one_by_one = started.elapsed();
        for (b, a) in both.iter().zip(&alone) {
            assert!(
                b.trim_end().ends_with("data: [DONE]"),
                "{b}\n{}",
                cluster.logs()
            );
            assert_eq!(sse_text(b), sse_text(a));
        }
        eprintln!(
            "TWO_AT_ONCE ms={:.0} one_by_one_ms={:.0}",
            at_once.as_secs_f64() * 1e3,
            one_by_one.as_secs_f64() * 1e3
        );
        together = serde_json::json!({"two_at_once_ms": at_once.as_secs_f64() * 1e3,
            "one_by_one_ms": one_by_one.as_secs_f64() * 1e3, "answers_equal": true});
    }
    // Free the GPU before the whole-model reference.
    cluster.stop();

    let mut whole = stages[0].clone();
    whole.layers = [0, layers];
    // The reference is one session of the whole model.
    (whole.sessions, whole.more_data_listen) = (1, vec![]);
    let (process, mut stdout, mut stdin) = glyd::RangeProcess::spawn(
        &engine,
        &build,
        &whole.arguments(),
        &whole.environment,
        whole.gpu_library_sha256.as_ref(),
        &std::env::temp_dir().join("starlings-reference-engine"),
    )
    .unwrap();
    let d = glyd::describe(&mut stdout, &mut stdin).unwrap();
    let workload = ProfileWorkload {
        context_tokens: BoundedU32::new(context).unwrap(),
        prompt_tokens: BoundedU32::new(max_prompt).unwrap(),
        output_tokens: BoundedU32::new(max_output).unwrap(),
        prefill_chunk_tokens: BoundedU32::new(chunk).unwrap(),
        sessions: BoundedU32::new(1).unwrap(),
    };
    let (contract, first, last, vocabulary) = glyd::qwen_boundary(
        &d,
        &build,
        "glyd-range",
        "qwen35-residual-mlp-bf16",
        workload,
    )
    .unwrap();
    assert!(first && last);
    let mut worker = glyd::Worker::new(
        stdout,
        stdin,
        glyd::Approval {
            descriptor: glyd::descriptor_digest(&d).unwrap(),
            boundary: contract.clone(),
            engine_bytes_limit: 15 << 30,
            payload_bytes_limit: 128 << 20,
            first,
            last,
            vocabulary,
        },
        FixtureMode::Reject,
    )
    .unwrap();
    worker.describe().unwrap();
    assert!(matches!(worker.receive().unwrap(), Event::Described));
    worker.prepare().unwrap();
    assert!(matches!(worker.receive().unwrap(), Event::Prepared(_)));
    worker.commit().unwrap();
    assert!(matches!(worker.receive().unwrap(), Event::Committed));
    worker.tokenize(&text).unwrap();
    let Event::Tokens(prompt) = worker.receive().unwrap() else {
        panic!("tokenize");
    };
    let digest =
        |ids: &[u32]| byte_digest(&ids.iter().flat_map(|t| t.to_le_bytes()).collect::<Vec<_>>());
    assert_eq!(head["prompt_sha256"], serde_json::json!(digest(&prompt)));
    let mut template: TensorHeader = decode(
        include_bytes!("../../../examples/tensor-header.json"),
        FixtureMode::Simulation,
    )
    .unwrap();
    template.plan_digest = byte_digest(b"whole-model reference run");
    worker
        .begin_session(
            (&template).into(),
            prompt.len() as u32,
            new,
            Some(Sampling {
                profile: "sample-v1".into(),
                temperature,
                top_p,
                seed: BoundedU32::new(seed).unwrap(),
            }),
        )
        .unwrap();
    let Event::Begun(session) = worker.receive().unwrap() else {
        panic!("begin_session")
    };
    let mut step = 0;
    let mut run = |start: u32, ids: &[u32], head: bool| {
        let mut scope = DataScope::from(&template);
        scope.invocation_id = uuid::Uuid::new_v4().to_string().parse().unwrap();
        scope.sequence = Decimal(step);
        step += 1;
        let rows = BoundedU32::new(ids.len() as u32).unwrap();
        let call = starlings_adapters::partial::RunChunk {
            version: Version,
            inputs: contract.layouts(rows.get(), 128 << 20).unwrap().0,
            step: starlings_adapters::partial::StepIdentity {
                scope,
                phase: if start < prompt.len() as u32 {
                    TensorHeaderPhase::Prefill
                } else {
                    TensorHeaderPhase::Decode
                },
                position_start: BoundedU32::new(start).unwrap(),
                token_count: rows,
            },
        };
        let bytes: Vec<u8> = ids.iter().flat_map(|t| t.to_le_bytes()).collect();
        worker
            .run_chunk(&session, &call, &call.step, &[bytes], head, false)
            .unwrap();
        match worker.receive().unwrap() {
            Event::Chunk { output, .. } => output.map(|o| match o {
                Output::Token { token, .. } => token,
                other => panic!("{other:?}"),
            }),
            other => panic!("{other:?}"),
        }
    };
    let started = Instant::now();
    let mut reference = Vec::new();
    for (i, ids) in prompt.chunks(chunk as usize).enumerate() {
        let start = i as u32 * chunk;
        let head = start as usize + ids.len() == prompt.len();
        if let Some(token) = run(start, ids, head) {
            reference.push(token);
        }
    }
    while (reference.len() as u32) < new && !stop_tokens.contains(reference.last().unwrap()) {
        let start = prompt.len() as u32 + reference.len() as u32 - 1;
        let token = run(start, &[*reference.last().unwrap()], true).unwrap();
        reference.push(token);
    }
    let whole_ms = started.elapsed().as_secs_f64() * 1e3;
    process.stop();
    let equal = head["tokens_sha256"] == serde_json::json!(digest(&reference))
        && tokens as usize == reference.len();
    let record = serde_json::json!({
        "test": "real_pipeline_daemons_generate_through_two_stages_like_the_whole_model",
        "engine_sha256": build,
        "model": model,
        "manifest": manifest,
        "prompt_text": text,
        "prompt_tokens": prompt_len,
        "chunk_tokens": chunk,
        "max_tokens": new,
        "temperature": temperature,
        "top_p": top_p,
        "seed": seed,
        "stop_tokens": stop_tokens,
        "pipeline_tokens": tokens,
        "pipeline_tokens_sha256": head["tokens_sha256"],
        "whole_tokens": reference.len(),
        "whole_tokens_sha256": digest(&reference),
        "tokens_equal": equal,
        "answer": answer,
        "client_stream_ms": elapsed.as_secs_f64() * 1e3,
        "whole_model_ms": whole_ms,
        "pipeline_reports": reports,
        "two_at_once": together,
    });
    println!("{record}");
    if let Ok(path) = std::env::var("STARLINGS_RANGE_RECORD") {
        fs::write(path, serde_json::to_vec_pretty(&record).unwrap()).unwrap();
    }
    assert!(equal, "pipeline tokens differ from the whole model");
}

/// GLY-366: a stage that dies mid-request. Two requests decode through a 50 ms relay (40 tokens,
/// about 2 s each) when the second stage's daemon is killed: both fail within their lease.
/// Restarted from its approved launch, the stage serves again. Then its engine is killed
/// mid-request: that request fails, the stage reports failed and takes nothing more (it cannot
/// account for that engine), and a restart serves again. The first stage serves throughout.
#[cfg(all(feature = "test-backend", unix))]
#[test]
fn pipeline_fake_engine_a_killed_stage_or_engine_fails_its_requests_and_serves_after_restart() {
    let temp = tempfile::tempdir().unwrap();
    let model = temp.path().canonicalize().unwrap();
    let engine = PathBuf::from(BIN).canonicalize().unwrap();
    let (stages, manifest, layers) = pipeline_stages_with(
        &engine,
        &model,
        &Default::default(),
        [[0, 1], [1, 2]],
        (128, 4, 32, 40),
        2,
    );
    let lifetime_ms = 5000;
    let mut cluster = pipeline_cluster(
        &stages,
        &manifest,
        layers,
        lifetime_ms,
        vec![],
        Some(Duration::from_millis(25)),
        0,
    );
    let kill = |pid: u32| {
        assert!(
            Command::new("kill")
                .args(["-9", &pid.to_string()])
                .status()
                .unwrap()
                .success()
        );
    };
    let failed_in_time = |(reply, elapsed): &(String, Duration)| {
        assert!(
            !reply.trim_end().ends_with("data: [DONE]"),
            "a request through a dead stage finished: {reply}"
        );
        assert!(
            *elapsed < Duration::from_millis(u64::from(lifetime_ms) + 3000),
            "failed after {elapsed:?}"
        );
    };
    let restart = |cluster: &mut PipelineCluster| {
        relaunch(&mut cluster.workers[1], "--stage", &[]);
        wait_stage_ready(cluster, 1);
    };

    // The second stage's daemon dies with two requests in flight.
    let daemon = cluster.workers[1].child.id();
    let replies: Vec<(String, Duration)> = thread::scope(|threads| {
        let c = &cluster;
        let runs: Vec<_> = ["hello one", "hello two"]
            .into_iter()
            .map(|prompt| {
                let run = threads.spawn(move || c.chat(prompt, 40, (0.0, 1.0, 1), true));
                thread::sleep(Duration::from_millis(300));
                run
            })
            .collect();
        thread::sleep(Duration::from_millis(500));
        kill(daemon);
        runs.into_iter().map(|r| r.join().unwrap()).collect()
    });
    replies.iter().for_each(failed_in_time);
    restart(&mut cluster);
    thread::sleep(Duration::from_secs(4));
    let (again, _) = cluster.chat("hello again", 8, (0.0, 1.0, 1), true);
    assert!(
        again.trim_end().ends_with("data: [DONE]"),
        "{again}\n{}",
        cluster.logs()
    );

    // Now its engine dies mid-request.
    let pids = Command::new("pgrep")
        .args(["-P", &cluster.workers[1].child.id().to_string()])
        .output()
        .unwrap();
    let engine: u32 = String::from_utf8_lossy(&pids.stdout)
        .lines()
        .next()
        .expect("the stage's engine")
        .trim()
        .parse()
        .unwrap();
    let reply = thread::scope(|threads| {
        let c = &cluster;
        let run = threads.spawn(move || c.chat("hello three", 40, (0.0, 1.0, 1), true));
        thread::sleep(Duration::from_millis(800));
        kill(engine);
        run.join().unwrap()
    });
    failed_in_time(&reply);
    // The stage takes nothing more with an engine it cannot account for; its operator restarts it.
    let deadline = Instant::now() + Duration::from_secs(20);
    while stage_state(&cluster.workers[1])["state"] != "failed" {
        assert!(Instant::now() < deadline, "the stage kept a dead engine");
        thread::sleep(Duration::from_millis(200));
    }
    mutation_once(&cluster.workers[1].state, &["cluster", "stop"]);
    cluster.workers[1].wait();
    restart(&mut cluster);
    thread::sleep(Duration::from_secs(4));
    let (last, _) = cluster.chat("hello at last", 8, (0.0, 1.0, 1), true);
    assert!(
        last.trim_end().ends_with("data: [DONE]"),
        "{last}\n{}",
        cluster.logs()
    );
    assert_eq!(sse_text(&last), "ffffffff");
    assert_eq!(stage_state(&cluster.workers[0])["state"], "ready");
    cluster.stop();
}

/// Machine-local checkpoint paths do not change the shared model identity.
#[cfg(feature = "test-backend")]
#[test]
fn pipeline_fake_engine_identical_checkpoints_work_at_different_paths() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().canonicalize().unwrap();
    let engine = PathBuf::from(BIN).canonicalize().unwrap();
    let (mut stages, manifest, layers) = pipeline_stages(
        &engine,
        &root.join("first"),
        &Default::default(),
        [[0, 1], [1, 2]],
        (128, 4, 32, 8),
    );
    let (other, other_manifest, _) = pipeline_stages(
        &engine,
        &root.join("second"),
        &Default::default(),
        [[0, 1], [1, 2]],
        (128, 4, 32, 8),
    );
    assert_eq!(manifest, other_manifest);
    stages[1] = other[1].clone();
    assert_ne!(stages[0].model, stages[1].model);
    let cluster = pipeline_cluster(&stages, &manifest, layers, 30_000, vec![], None, 0);
    let (reply, _) = cluster.chat("different paths", 8, (0.0, 1.0, 1), true);
    assert!(
        reply.trim_end().ends_with("data: [DONE]"),
        "{reply}\n{}",
        cluster.logs()
    );
    assert_eq!(sse_text(&reply), "ffffffff");
    cluster.stop();
}

/// Full phase matrix is opt-in so ordinary CI retains the smaller restart regressions.
#[cfg(all(feature = "test-backend", unix))]
#[test]
#[ignore = "explicit local or integration qualification; sixteen process-failure cases"]
fn pipeline_fake_engine_phase_fault_matrix() {
    for sessions in [1, 2] {
        for stage in [0, 1] {
            for target_engine in [false, true] {
                for phase in ["prefill", "decode"] {
                    eprintln!(
                        "FAULT_CASE sessions={sessions} stage={stage} engine={target_engine} phase={phase}"
                    );
                    let temp = tempfile::tempdir().unwrap();
                    let root = temp.path().canonicalize().unwrap();
                    let engine = PathBuf::from(BIN).canonicalize().unwrap();
                    let (mut stages, manifest, layers) = pipeline_stages_with(
                        &engine,
                        &root,
                        &Default::default(),
                        [[0, 1], [1, 2]],
                        (128, 4, 32, 40),
                        sessions,
                    );
                    let gate = root.join("gate");
                    fs::create_dir(&gate).unwrap();
                    stages[stage].environment.insert(
                        "STARLINGS_TEST_PHASE_GATE".into(),
                        gate.to_str().unwrap().into(),
                    );
                    let mut cluster =
                        pipeline_cluster(&stages, &manifest, layers, 5000, vec![], None, 0);
                    fs::write(gate.join("arm"), phase).unwrap();
                    let engine_pid = thread::scope(|threads| {
                        let c = &cluster;
                        let clients: Vec<_> = (0..sessions)
                            .map(|_| {
                                threads
                                    .spawn(move || c.chat("fault barrier", 40, (0.0, 1.0, 1), true))
                            })
                            .collect();
                        let deadline = Instant::now() + Duration::from_secs(4);
                        let engine_pid: u32 = loop {
                            if let Ok(pid) = fs::read_to_string(gate.join("reached"))
                                && let Ok(pid) = pid.parse()
                            {
                                break pid;
                            }
                            assert!(Instant::now() < deadline, "phase not reached\n{}", c.logs());
                            thread::sleep(Duration::from_millis(10));
                        };
                        let pid = if target_engine {
                            engine_pid
                        } else {
                            c.workers[stage].child.id()
                        };
                        assert!(
                            Command::new("kill")
                                .args(["-9", &pid.to_string()])
                                .status()
                                .unwrap()
                                .success()
                        );
                        // Let an orphaned stand-in observe its closed pipes. Native device cleanup
                        // after daemon SIGKILL requires its separate hardware qualification.
                        fs::remove_file(gate.join("arm")).unwrap();
                        for client in clients {
                            let (reply, elapsed) = client.join().unwrap();
                            assert!(!reply.trim_end().ends_with("data: [DONE]"), "{reply}");
                            assert!(elapsed < Duration::from_secs(8), "{elapsed:?}");
                        }
                        engine_pid
                    });
                    if target_engine {
                        let deadline = Instant::now() + Duration::from_secs(8);
                        while stage_state(&cluster.workers[stage])["state"] != "failed" {
                            assert!(Instant::now() < deadline, "dead engine remained eligible");
                            thread::sleep(Duration::from_millis(100));
                        }
                        mutation_once(&cluster.workers[stage].state, &["cluster", "stop"]);
                        cluster.workers[stage].wait();
                    }
                    relaunch(&mut cluster.workers[stage], "--stage", &[]);
                    wait_stage_ready(&cluster, stage);
                    thread::sleep(Duration::from_secs(4));
                    let (reply, _) = cluster.chat("after restart", 8, (0.0, 1.0, 1), true);
                    assert!(
                        reply.trim_end().ends_with("data: [DONE]"),
                        "{reply}\n{}",
                        cluster.logs()
                    );
                    assert_eq!(sse_text(&reply), "ffffffff");
                    // A departed stand-in must not remain alive after recovery.
                    let gone = Command::new("kill")
                        .args(["-0", &engine_pid.to_string()])
                        .output()
                        .unwrap();
                    assert!(
                        !gone.status.success(),
                        "old stand-in engine remains: {engine_pid}"
                    );
                    cluster.stop();
                }
            }
        }
    }
}

/// The actual `up` entry point, with synthetic checkpoint files and GPU inventory.
#[cfg(all(feature = "test-backend", unix))]
#[test]
fn pipeline_fake_engine_up_chats_restarts_and_refuses_changed_saved_setup() {
    onboarding(1);
}

#[cfg(all(feature = "test-backend", unix))]
#[test]
#[ignore = "explicit onboarding qualification; two interactive processes and saved restart"]
fn pipeline_fake_engine_two_machine_up_join_and_saved_restart() {
    onboarding(2);
}

#[cfg(all(feature = "test-backend", unix))]
fn onboarding(machines: u32) {
    use std::{io::Write, os::unix::fs::PermissionsExt};
    static PORTS: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let _ports = PORTS.lock().unwrap(); // `up` uses fixed operator ports.
    // Nested state paths must still fit the Unix socket path bound on macOS.
    let temp = tempfile::tempdir_in("/tmp").unwrap();
    let root = temp.path().canonicalize().unwrap();
    let cache = root.join("hub");
    let repo = cache.join("models--fixture--model");
    let revision = "a".repeat(40);
    let model = repo.join("snapshots").join(&revision);
    fs::create_dir_all(&model).unwrap();
    fs::create_dir(repo.join("refs")).unwrap();
    fs::write(repo.join("refs/main"), revision).unwrap();
    fs::write(
        model.join("config.json"),
        br#"{"num_hidden_layers":2,"num_key_value_heads":1,"head_dim":4}"#,
    )
    .unwrap();
    fs::write(model.join("tokenizer.json"), b"{}").unwrap();
    let header = serde_json::to_vec(&serde_json::json!({
        "model.embed_tokens.weight": {"data_offsets": [0, 1]},
        "model.layers.0.weight": {"data_offsets": [1, 2]},
        "model.layers.1.weight": {"data_offsets": [2, 3]},
        "lm_head.weight": {"data_offsets": [3, 4]}}))
    .unwrap();
    let mut weights = (header.len() as u64).to_le_bytes().to_vec();
    weights.extend(header);
    weights.extend([0; 4]);
    fs::write(model.join("model.safetensors"), weights).unwrap();
    let driver = root.join("nvidia-smi");
    fs::write(&driver, b"#!/bin/sh\nprintf '16384, 580.0\\n'\n").unwrap();
    fs::set_permissions(&driver, fs::Permissions::from_mode(0o755)).unwrap();
    let path = std::env::join_paths(
        std::iter::once(root.clone())
            .chain(std::env::split_paths(&std::env::var_os("PATH").unwrap())),
    )
    .unwrap();
    let state = root.join("missing/parents/state");
    let peer_state = root.join("peer/state");
    let command = || {
        let mut cmd = Command::new(BIN);
        cmd.env("PATH", &path)
            .env("HF_HUB_CACHE", &cache)
            .args(["--state-dir"])
            .arg(&state)
            // The stand-in has no fetch command: this succeeds only by reusing the cached pin.
            .args(["up", "--model", "fixture/model"])
            .args(["--machines", &machines.to_string(), "--host", "127.0.0.1"])
            .args(["--engine", BIN, "--ctx", "128", "--drafts", "0", "--yes"]);
        cmd
    };
    struct Up(Child);
    impl Drop for Up {
        fn drop(&mut self) {
            let _ = Command::new("kill")
                .args(["-INT", &self.0.id().to_string()])
                .output();
            let deadline = Instant::now() + Duration::from_secs(30);
            while self.0.try_wait().ok().flatten().is_none() && Instant::now() < deadline {
                thread::sleep(Duration::from_millis(100));
            }
            let _ = self.0.kill();
            let _ = self.0.wait();
        }
    }
    for _ in 0..2 {
        let log_path = root.join("up.log");
        let log = fs::File::create(&log_path).unwrap();
        let mut up = Up(command()
            .stdin(std::process::Stdio::piped())
            .stdout(log.try_clone().unwrap())
            .stderr(log)
            .spawn()
            .unwrap());
        let peer = if machines == 2 {
            let fresh = !peer_state.join("join.json").exists();
            if fresh {
                let deadline = Instant::now() + Duration::from_secs(30);
                while !state.join("join-code.json").exists() {
                    assert!(
                        up.0.try_wait().unwrap().is_none(),
                        "{}",
                        fs::read_to_string(&log_path).unwrap()
                    );
                    assert!(Instant::now() < deadline, "join code not published");
                    thread::sleep(Duration::from_millis(100));
                }
            }
            let peer_log_path = root.join("join.log");
            let peer_log = fs::File::create(&peer_log_path).unwrap();
            let mut peer = Up(Command::new(BIN)
                .env("PATH", &path)
                .env("HF_HUB_CACHE", &cache)
                .arg("--state-dir")
                .arg(&peer_state)
                .args([
                    "join",
                    "--model",
                    "fixture/model",
                    "--engine",
                    BIN,
                    "--host",
                    "127.0.0.1",
                    "--yes",
                ])
                .stdin(std::process::Stdio::piped())
                .stdout(peer_log.try_clone().unwrap())
                .stderr(peer_log)
                .spawn()
                .unwrap());
            if fresh {
                let code: String =
                    serde_json::from_slice(&fs::read(state.join("join-code.json")).unwrap())
                        .unwrap();
                writeln!(peer.0.stdin.as_mut().unwrap(), "{code}").unwrap();
                let deadline = Instant::now() + Duration::from_secs(30);
                loop {
                    let text = fs::read_to_string(&peer_log_path).unwrap();
                    if let Some(check) = text.lines().find_map(|l| {
                        l.strip_prefix("It then asks for this check; type it there: ")
                    }) {
                        let code = text
                            .split("Paste this on the first machine:")
                            .nth(1)
                            .unwrap()
                            .split_whitespace()
                            .next()
                            .unwrap();
                        writeln!(up.0.stdin.as_mut().unwrap(), "{code}\n{check}").unwrap();
                        break;
                    }
                    assert!(peer.0.try_wait().unwrap().is_none(), "{text}");
                    assert!(
                        Instant::now() < deadline,
                        "peer approval not published: {text}"
                    );
                    thread::sleep(Duration::from_millis(100));
                }
            }
            Some(peer)
        } else {
            None
        };
        let deadline = Instant::now() + Duration::from_secs(45);
        loop {
            let text = fs::read_to_string(&log_path).unwrap();
            if text.contains("READY: open ") {
                break;
            }
            assert!(up.0.try_wait().unwrap().is_none(), "{text}");
            assert!(Instant::now() < deadline, "up did not become ready: {text}");
            thread::sleep(Duration::from_millis(100));
        }
        let coordinator = state.join("coordinator");
        let token = starlings_security::ClientToken::load(
            &starlings_security::PrivateDir::open(&coordinator).unwrap(),
            "token.json",
        )
        .unwrap();
        let saved: serde_json::Value =
            serde_json::from_slice(&fs::read(state.join("up.json")).unwrap()).unwrap();
        let body = serde_json::to_vec(&serde_json::json!({"model": saved["manifest"],
            "messages": [{"role": "user", "content": "hello"}], "max_tokens": 8,
            "temperature": 0.0, "top_p": 1.0, "seed": 1, "stream": true}))
        .unwrap();
        let mut socket = TcpStream::connect("127.0.0.1:9750").unwrap();
        socket
            .set_read_timeout(Some(Duration::from_secs(10)))
            .unwrap();
        write!(socket, "POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1:9750\r\nAuthorization: Bearer {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n", token.bearer(), body.len()).unwrap();
        socket.write_all(&body).unwrap();
        let mut reply = String::new();
        socket.read_to_string(&mut reply).unwrap();
        assert!(reply.trim_end().ends_with("data: [DONE]"), "{reply}");
        assert_eq!(sse_text(&reply), "ffffffff");
        drop(peer);
        drop(up);
        let names: &[&str] = if machines == 1 {
            &["coordinator", "stage0", "stage1"]
        } else {
            &["coordinator", "stage0"]
        };
        for name in names {
            assert!(!state.join(name).join("admin.cap").exists());
            assert!(state.join(name).join("config.json").exists());
        }
        if machines == 2 {
            assert!(!peer_state.join("stage/admin.cap").exists());
            assert!(peer_state.join("stage/config.json").exists());
        }
    }
    let original = fs::read(model.join("model.safetensors")).unwrap();
    let mut changed = original.clone();
    *changed.last_mut().unwrap() = 1;
    fs::write(model.join("model.safetensors"), changed).unwrap();
    let refused = command().output().unwrap();
    assert!(!refused.status.success());
    assert!(String::from_utf8_lossy(&refused.stderr).contains("saved setup does not match"));
    fs::write(model.join("model.safetensors"), original).unwrap();
    let mut saved: serde_json::Value =
        serde_json::from_slice(&fs::read(state.join("up.json")).unwrap()).unwrap();
    saved["cut"] = 99.into();
    fs::write(state.join("up.json"), serde_json::to_vec(&saved).unwrap()).unwrap();
    let refused = command().output().unwrap();
    assert!(!refused.status.success());
    assert!(String::from_utf8_lossy(&refused.stderr).contains("saved split is outside"));
}

/// Starts a killed or stopped daemon again from its approved launch record.
#[cfg(all(feature = "test-backend", unix))]
fn relaunch(daemon: &mut Daemon, flag: &str, extra: &[&str]) {
    let state = daemon.state.clone();
    let _ = daemon.child.wait();
    let record: serde_json::Value =
        serde_json::from_slice(&fs::read(state.join("launch.json")).unwrap()).unwrap();
    let mut process = Command::new(BIN);
    process
        .args(["daemon", flag])
        .arg(state.join("launch.json"))
        .args([
            "--approve",
            starlings_protocol::content_hash(&record).unwrap().as_str(),
        ])
        .args(extra);
    *daemon = Daemon::launch_with_arguments(&state, process);
}
/// Until stage `i` is ready and the coordinator holds its ready sample.
#[cfg(all(feature = "test-backend", unix))]
fn wait_stage_ready(cluster: &PipelineCluster, i: usize) {
    let deadline = Instant::now() + Duration::from_secs(60);
    loop {
        let view = cluster
            .authority
            .admin(&["cluster", "resources", &cluster.nodes[i]]);
        let ready = view
            .strip_prefix("RESOURCES: ")
            .and_then(|v| serde_json::from_str::<serde_json::Value>(v.trim()).ok())
            .is_some_and(|v| v["sample"]["capabilities"]["state"] == "ready");
        if ready && stage_state(&cluster.workers[i])["state"] == "ready" {
            return;
        }
        assert!(
            Instant::now() < deadline,
            "stage {i} not ready after a restart"
        );
        thread::sleep(Duration::from_millis(500));
    }
}

/// GLY-366: the first stage's daemon, then the coordinator, killed with two requests decoding
/// through a 50 ms relay. Each time both requests end within their lease, and the daemon started
/// again from its approved launch serves.
#[cfg(all(feature = "test-backend", unix))]
#[test]
fn pipeline_fake_engine_a_killed_first_stage_or_coordinator_recovers_after_restart() {
    let temp = tempfile::tempdir().unwrap();
    let model = temp.path().canonicalize().unwrap();
    let engine = PathBuf::from(BIN).canonicalize().unwrap();
    let (stages, manifest, layers) = pipeline_stages_with(
        &engine,
        &model,
        &Default::default(),
        [[0, 1], [1, 2]],
        (128, 4, 32, 40),
        2,
    );
    let lifetime_ms = 5000;
    let mut cluster = pipeline_cluster(
        &stages,
        &manifest,
        layers,
        lifetime_ms,
        vec![],
        Some(Duration::from_millis(25)),
        0,
    );
    let kill = |pid: u32| {
        assert!(
            Command::new("kill")
                .args(["-9", &pid.to_string()])
                .status()
                .unwrap()
                .success()
        );
    };
    // Two requests in flight when `pid` dies; each client gets an error or a closed connection
    // within its lease (a dead coordinator cannot answer at all).
    let both_end = |cluster: &PipelineCluster, pid: u32| {
        let ended: Vec<_> = thread::scope(|threads| {
            let runs: Vec<_> = ["hello one", "hello two"]
                .into_iter()
                .map(|prompt| {
                    let run = threads.spawn(move || {
                        let started = Instant::now();
                        let reply = std::panic::catch_unwind(|| {
                            cluster.chat(prompt, 40, (0.0, 1.0, 1), true).0
                        })
                        .unwrap_or_default();
                        (reply, started.elapsed())
                    });
                    thread::sleep(Duration::from_millis(300));
                    run
                })
                .collect();
            thread::sleep(Duration::from_millis(500));
            kill(pid);
            runs.into_iter().map(|r| r.join().unwrap()).collect()
        });
        for (reply, elapsed) in ended {
            assert!(
                !reply.trim_end().ends_with("data: [DONE]"),
                "a request through a dead daemon finished: {reply}"
            );
            assert!(
                elapsed < Duration::from_millis(u64::from(lifetime_ms) + 3000),
                "ended after {elapsed:?}"
            );
        }
    };
    let serves = |cluster: &PipelineCluster| {
        let (reply, _) = cluster.chat("hello again", 8, (0.0, 1.0, 1), true);
        assert!(
            reply.trim_end().ends_with("data: [DONE]"),
            "{reply}\n{}",
            cluster.logs()
        );
        assert_eq!(sse_text(&reply), "ffffffff");
    };

    both_end(&cluster, cluster.workers[0].child.id());
    relaunch(&mut cluster.workers[0], "--stage", &[]);
    wait_stage_ready(&cluster, 0);
    thread::sleep(Duration::from_secs(4));
    serves(&cluster);

    both_end(&cluster, cluster.authority.child.id());
    let client = cluster.nodes[2].clone();
    relaunch(
        &mut cluster.authority,
        "--pipeline",
        &["--ui", "127.0.0.1:0", "--ui-client", &client],
    );
    // The test's client API port is ephemeral; the restarted coordinator says its new one.
    cluster.address = value(
        &cluster.authority.admin(&["cluster", "status"]),
        "inference=",
    );
    // The stages unwind their requests when the leases run out, then take the new epoch's.
    thread::sleep(Duration::from_secs(6));
    wait_stage_ready(&cluster, 0);
    wait_stage_ready(&cluster, 1);
    serves(&cluster);
    cluster.stop();
}
