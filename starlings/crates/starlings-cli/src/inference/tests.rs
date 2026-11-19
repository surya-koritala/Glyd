use super::*;
use starlings_adapters::service::{ServiceConfig, executable_digest};
use starlings_core::{Worker, routing::WarmReplica};
use starlings_protocol::*;
use starlings_security::{
    Authority, ClientListener, Identity, Invitation, ListenerBudget, PeerListener, PrivateDir,
    RegistryClient, RegistryRequest, Role,
};
use std::{
    collections::BTreeMap,
    io::{Read, Write},
    net::TcpStream,
    sync::atomic::{AtomicBool, Ordering},
};

#[test]
fn fixture_process() {
    starlings_adapters::service::fixture::run();
}

#[cfg(unix)]
#[test]
#[ignore = "requires explicit bounded native CPU workload approval and pinned config"]
fn native_relay_acceptance() {
    let path =
        std::env::var_os("COMPUTE_NATIVE_RELAY_CONFIG").expect("explicit native config required");
    let bytes =
        starlings_security::catalog::read_bounded(std::fs::File::open(path).unwrap()).unwrap();
    let config: ServiceConfig = serde_json::from_value(parse_json(&bytes).unwrap()).unwrap();
    let approved =
        std::env::var("COMPUTE_NATIVE_RELAY_DIGEST").expect("explicit approved digest required");
    assert!(
        config.digest().unwrap().as_str() == approved,
        "configuration approval mismatch"
    );
    assert!(config.request_timeout_ms <= 60_000);
    assert_eq!(
        config.manifest_digest.as_str(),
        "cedf89af31c9041b601fa58303285bc46d99c51baee1b13f5e919626ca526ee5"
    );
    println!("{}", relay_acceptance(config));
}
fn relay_acceptance(config: ServiceConfig) -> serde_json::Value {
    let approved = config.digest().unwrap();
    let mut f = Fixture::with_config("normal", false, Some(config));
    f.chat.messages = vec![ChatMessage {
        role: ChatMessageRole::User,
        content: "Reply with the single word ready.".into(),
    }];
    let Operation::Prepare { plan, .. } = &f.controls[0].operation else {
        panic!("prepare")
    };
    let mut dispatch =
        Dispatch::new(f.binding.clone(), *plan.clone(), f.worker.profile(), 60_000).unwrap();
    let start = Instant::now();
    let mut runs = Vec::new();
    let mut reference = None;
    for streaming in [false, true] {
        f.chat.stream = Some(streaming);
        f.while_maintaining(|| {
            let mut service = f.service();
            if !streaming {
                service.generate(&f.chat, |bytes| {
                    reference = Some(semantic(&parse_json(bytes).unwrap()));
                    Ok(())
                }).unwrap();
            }
            let (coordinator, remote) = f.pair();
            let (client, chat, response) = f.client();
            client.set_read_timeout(Some(Duration::from_secs(65))).unwrap();
            let request_start = Instant::now();
            let done = thread::scope(|threads| {
                let worker = threads.spawn(|| serve_worker(&mut service, &f.worker, remote));
                let reader = threads.spawn(move || {
                    let mut bytes = Vec::new();
                    client.take(2 * 1024 * 1024 + 1).read_to_end(&mut bytes).unwrap();
                    assert!(bytes.len() <= 2 * 1024 * 1024);
                    String::from_utf8(bytes).unwrap()
                });
                let result = dispatch.run(response, &chat, coordinator);
                let done = worker.join().unwrap();
                assert!(result.is_ok(), "native relay failed: {result:?}");
                let body = reader.join().unwrap();
                let (header, payload) = body.split_once("\r\n\r\n").unwrap();
                assert!(header.starts_with("HTTP/1.1 200 OK"));
                let actual = if streaming { stream_semantic(payload) } else { semantic(&parse_json(payload.as_bytes()).unwrap()) };
                assert!(reference.as_ref() == Some(&actual), "native semantic parity failed");
                done.unwrap().unwrap()
            });
            let elapsed = request_start.elapsed().as_millis();
            assert!(f.worker.snapshot().unwrap().reserved_bytes.0 > 0);
            #[cfg(unix)]
            service.confirm_process_group_exit().unwrap();
            #[cfg(not(unix))]
            service.stop().unwrap();
            // This isolated CPU acceptance checks original-group exit only. It does
            // not install a production profile or establish GPU/escaped-child drain.
            f.worker.failed_and_drained(done.generation, done.invocation).unwrap();
            assert_eq!(f.worker.snapshot().unwrap().reserved_bytes.0, 0);
            runs.push(serde_json::json!({"streaming":streaming,"relay_ms":elapsed,"semantic_parity":true,"count_matches_usage":true,"owned_group_exit":cfg!(unix)}));
        });
    }
    serde_json::json!({"kind":"RelayAcceptanceOnly","configuration_digest":approved,"executable_sha256":f.config.executable_sha256,"runs":runs,"elapsed_ms":start.elapsed().as_millis(),"usage":reference.unwrap()["usage"],"catalog_qualified":false,"memory_qualified":false,"device_drain_qualified":false,"daemon_activated":false})
}
fn semantic(value: &serde_json::Value) -> serde_json::Value {
    serde_json::json!({"content":value["choices"][0]["message"]["content"].as_str().unwrap_or(""),"reasoning_content":value["choices"][0]["message"]["reasoning_content"].as_str().unwrap_or(""),"finish_reason":value["choices"][0]["finish_reason"],"usage":value["usage"]})
}
fn stream_semantic(payload: &str) -> serde_json::Value {
    let mut content = String::new();
    let mut reasoning = String::new();
    let mut usage = serde_json::Value::Null;
    let mut finish = serde_json::Value::Null;
    let mut done = false;
    for event in payload.split("\n\n").filter(|s| !s.is_empty()) {
        assert!(!done, "output after DONE");
        let data = event
            .strip_prefix("data: ")
            .expect("unexpected stream event");
        if data == "[DONE]" {
            done = true;
            continue;
        }
        let value = parse_json(data.as_bytes()).unwrap();
        content.push_str(
            value["choices"][0]["delta"]["content"]
                .as_str()
                .unwrap_or(""),
        );
        reasoning.push_str(
            value["choices"][0]["delta"]["reasoning_content"]
                .as_str()
                .unwrap_or(""),
        );
        if !value["choices"][0]["finish_reason"].is_null() {
            finish = value["choices"][0]["finish_reason"].clone();
        }
        if !value["usage"].is_null() {
            usage = value["usage"].clone();
        }
    }
    assert!(done);
    serde_json::json!({"content":content,"reasoning_content":reasoning,"finish_reason":finish,"usage":usage})
}

struct Fixture {
    _temp: tempfile::TempDir,
    authority: Authority,
    controller: NodeId,
    identity: Identity,
    invitation: Invitation,
    registry: std::sync::Mutex<RegistryClient>,
    registry_listener: PeerListener,
    fence: starlings_security::RegistryFence,
    config: ServiceConfig,
    binding: GenerationBinding,
    worker: EnrolledWorker,
    controls: [ControlRequest; 3],
    chat: ChatRequest,
    origin: Instant,
}
fn enroll(
    authority: &mut Authority,
    files: &PrivateDir,
    role: Role,
    name: &str,
) -> (Identity, Invitation) {
    let listener =
        PeerListener::bind("127.0.0.1:0".parse().unwrap(), ListenerBudget::default()).unwrap();
    authority
        .invite(files, name, listener.local_addr().unwrap(), role)
        .unwrap();
    let invitation = Invitation::load(files, name).unwrap();
    let mut identity = Identity::generate(invitation.cluster()).unwrap();
    let fingerprint = identity.request_fingerprint();
    thread::scope(|threads| {
        let server = threads.spawn(|| {
            let pending = listener.accept().unwrap().enroll(authority).unwrap();
            authority.approve(pending, &fingerprint).unwrap();
        });
        invitation.enroll(&mut identity).unwrap();
        server.join().unwrap();
    });
    (identity, invitation)
}
impl Fixture {
    fn while_maintaining<T>(&self, run: impl FnOnce() -> T) -> T {
        let stopped = AtomicBool::new(false);
        let heartbeat_done = AtomicBool::new(false);
        let (ready, received) = std::sync::mpsc::sync_channel(1);
        struct Stop<'a>(&'a AtomicBool);
        impl Drop for Stop<'_> {
            fn drop(&mut self) {
                self.0.store(true, Ordering::Release);
            }
        }
        thread::scope(|threads| {
            threads.spawn(|| {
                let _done = Stop(&heartbeat_done);
                let mut ready = Some(ready);
                while !stopped.load(Ordering::Acquire) {
                    let result = self.registry.lock().unwrap().heartbeat(None);
                    assert!(
                        result.is_ok() || stopped.load(Ordering::Acquire),
                        "registry heartbeat failed: {result:?}"
                    );
                    if let Some(ready) = ready.take() {
                        let _ = ready.send(());
                    }
                    let until = Instant::now() + Duration::from_secs(1);
                    while Instant::now() < until && !stopped.load(Ordering::Acquire) {
                        thread::sleep(POLL);
                    }
                }
            });
            threads.spawn(|| {
                // Finish the client exchange before retiring its server. Otherwise a
                // closing TCP handshake can remain queued for the next acceptance run.
                while !stopped.load(Ordering::Acquire) || !heartbeat_done.load(Ordering::Acquire) {
                    let now = self.origin.elapsed().as_millis() as u64;
                    self.authority.maintain(now).unwrap();
                    self.fence.maintain(now).unwrap();
                    if let Some(incoming) = self.registry_listener.try_accept().unwrap() {
                        match incoming
                            .registry(self.authority.enrollment_reader().unwrap())
                            .expect("registry fixture request failed")
                        {
                            RegistryRequest::Heartbeat(pending) => {
                                pending.respond().expect("registry fixture response failed")
                            }
                            _ => panic!("unexpected enrollment"),
                        }
                    }
                    thread::sleep(POLL);
                }
            });
            let _stop = Stop(&stopped);
            received
                .recv_timeout(Duration::from_secs(3))
                .expect("fresh registry heartbeat before relay");
            run()
        })
    }
    fn new(mode: &str, streaming: bool) -> Self {
        Self::with_config(mode, streaming, None)
    }
    fn with_config(mode: &str, streaming: bool, native: Option<ServiceConfig>) -> Self {
        let is_native = native.is_some();
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().canonicalize().unwrap();
        let mut authority = Authority::create(&root.join("authority")).unwrap();
        let files = PrivateDir::ensure(root.join("files")).unwrap();
        let (identity, _) = enroll(&mut authority, &files, Role::Worker, "worker");
        let (controller, _) = enroll(&mut authority, &files, Role::Controller, "controller");
        identity.save(&files, "identity.json").unwrap();
        let listener =
            PeerListener::bind("127.0.0.1:0".parse().unwrap(), ListenerBudget::default()).unwrap();
        authority
            .invite(
                &files,
                "registry",
                listener.local_addr().unwrap(),
                Role::Worker,
            )
            .unwrap();
        let invitation = Invitation::load(&files, "registry").unwrap();
        let mut registry = RegistryClient::open(
            &root.join("files"),
            identity,
            Invitation::load(&files, "registry").unwrap(),
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
                            _ => panic!("expected heartbeat"),
                        },
                    );
                registry.heartbeat(None).unwrap();
                server.join().unwrap();
            });
        }
        let identity = Identity::load(&files, "identity.json").unwrap();
        let endpoint = "127.0.0.1:1".parse::<std::net::SocketAddr>().unwrap();
        let executable = std::env::current_exe().unwrap().canonicalize().unwrap();
        let mut environment = BTreeMap::from([
            ("COMPUTE_FIXTURE_ENDPOINT".into(), endpoint.to_string()),
            ("COMPUTE_FIXTURE_MODE".into(), mode.into()),
            (
                "COMPUTE_FIXTURE_STARTED".into(),
                root.join("started").to_str().unwrap().into(),
            ),
        ]);
        if let Some(system) = std::env::var_os("SystemRoot") {
            environment.insert("SYSTEMROOT".into(), system.to_string_lossy().into_owned());
        }
        let config = native.unwrap_or_else(|| ServiceConfig {
            executable_sha256: executable_digest(&executable).unwrap(),
            executable,
            arguments: vec![
                "--exact".into(),
                "inference::tests::fixture_process".into(),
                "--nocapture".into(),
            ],
            directory: root.clone(),
            environment,
            key_environment: "COMPUTE_FIXTURE_KEY".into(),
            endpoint,
            backend_version: "synthetic-v1".into(),
            properties: None,
            api: Default::default(),
            glyd_identity: None,
            manifest_digest: byte_digest(b"synthetic relay model"),
            execution_profile: "synthetic-whole-request".into(),
            backend_model: "synthetic-backend".into(),
            reasoning_content: true,
            request_timeout_ms: 5_000,
        });
        let binding = GenerationBinding {
            version: Version,
            epoch: authority.registry_epoch().unwrap(),
            model: config.manifest_digest.clone(),
            profile: config.execution_profile.clone(),
        };
        for (node, action) in [
            (authority.identity().node(), Action::Control),
            (identity.node(), Action::Infer),
            (controller.node(), Action::Infer),
        ] {
            authority
                .set_permissions(
                    node,
                    &[Scope {
                        model: binding.model.clone(),
                        action,
                    }],
                )
                .unwrap();
        }
        let mut plan: PlacementPlan = decode(
            include_bytes!("../../../../examples/placement-plan.json"),
            FixtureMode::Simulation,
        )
        .unwrap();
        plan.fixture = false;
        plan.mode = PlacementPlanMode::WholeRequest;
        plan.cluster_id = identity.cluster();
        plan.authority_epoch = binding.epoch.number;
        plan.manifest_digest = binding.model.clone();
        plan.execution_profile = binding.profile.clone();
        plan.stages.truncate(1);
        if is_native {
            plan.model_layer_count = BoundedU32::new(24).unwrap();
            plan.context_limit = BoundedU32::new(512).unwrap();
            // Conservative harness allowance, not a measured production memory profile.
            plan.stages[0].resident_bytes = Decimal(4 * 1024 * 1024 * 1024);
        }
        let stage = &mut plan.stages[0];
        stage.node_id = identity.node();
        stage.layer_end_exclusive = plan.model_layer_count;
        stage.owns_final_norm_and_head = true;
        stage.backend_build_digest = config.executable_sha256.clone();
        if is_native {
            stage.backend_id = "whole-request-service".into();
            stage.device_id = "cpu:0".into();
            stage.memory_pool_id = "harness-reservation-only".into();
        }
        let profile = WarmReplica {
            node: identity.node(),
            manifest: binding.model.clone(),
            profile: binding.profile.clone(),
            backend: stage.backend_id.clone(),
            build: stage.backend_build_digest.clone(),
            pool: stage.memory_pool_id.clone(),
            evidence: byte_digest(b"synthetic qualification, not native"),
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
        lease.owner_node_id = identity.node();
        lease.authority_epoch = plan.authority_epoch;
        let mut session: Session = decode(
            include_bytes!("../../../../examples/session.json"),
            FixtureMode::Simulation,
        )
        .unwrap();
        session.client_id = controller.node().to_string().parse().unwrap();
        session.plan_digest = plan.plan_digest.clone();
        session.authority_epoch = plan.authority_epoch;
        session.prompt_token_count = BoundedU32::new(3).unwrap();
        session.max_output_tokens = BoundedU32::new(if is_native { 8 } else { 1 }).unwrap();
        session.context_limit = plan.context_limit;
        let mut chat: ChatRequest = decode(
            include_bytes!("../../../../examples/chat-request.json"),
            FixtureMode::Reject,
        )
        .unwrap();
        chat.model = binding.model.to_string();
        chat.stream = Some(streaming);
        chat.max_tokens = session.max_output_tokens;
        chat.temperature = Some(session.sampling.temperature);
        chat.top_p = Some(session.sampling.top_p);
        chat.seed = Some(session.sampling.seed);
        let operation = |sequence, operation| {
            ControlRequest::new(
                format!("00000000-0000-4000-8000-{sequence:012}")
                    .parse()
                    .unwrap(),
                binding.epoch.number.0,
                sequence,
                operation,
            )
            .unwrap()
        };
        let controls = [
            operation(
                1,
                Operation::Prepare {
                    plan: Box::new(plan),
                    lease: Box::new(lease.clone()),
                },
            ),
            operation(
                2,
                Operation::Commit {
                    lease_id: lease.lease_id,
                },
            ),
            operation(
                3,
                Operation::StartRequest {
                    session: Box::new(session),
                    invocation_id: "00000000-0000-4000-8000-000000000999".parse().unwrap(),
                    request: Box::new(chat.clone()),
                },
            ),
        ];
        let origin = Instant::now();
        let worker = EnrolledWorker::new(
            Worker::new(
                identity.node(),
                identity.cluster(),
                profile.request_bytes,
                binding.epoch.number.0,
                FixtureMode::Reject,
            )
            .unwrap(),
            registry.fence(),
            profile,
        )
        .unwrap();
        Self {
            _temp: temp,
            authority,
            controller: controller.node(),
            identity,
            invitation,
            fence: registry.fence(),
            registry: std::sync::Mutex::new(registry),
            registry_listener: listener,
            config,
            binding,
            worker,
            controls,
            chat,
            origin,
        }
    }
    fn service(&self) -> Service {
        let key = format!(
            "{}{}",
            uuid::Uuid::new_v4().simple(),
            uuid::Uuid::new_v4().simple()
        );
        let mut service = if self.config.backend_version == "synthetic-v1" {
            starlings_adapters::service::fixture::spawn(self.config.clone(), key)
        } else {
            Service::spawn(self.config.clone(), &self.config.digest().unwrap(), key)
        }
        .unwrap();
        let deadline =
            Instant::now() + Duration::from_millis(self.config.request_timeout_ms.into());
        while service.probe().is_err() {
            assert!(Instant::now() < deadline, "synthetic service did not start");
            thread::sleep(Duration::from_millis(10));
        }
        service
    }
    fn pair(&self) -> (GenerationStream, GenerationStream) {
        let listener =
            PeerListener::bind("127.0.0.1:0".parse().unwrap(), ListenerBudget::default()).unwrap();
        thread::scope(|threads| {
            let server = threads.spawn(|| {
                listener
                    .accept()
                    .unwrap()
                    .generation(
                        &self.identity,
                        &self.invitation,
                        self.fence.clone(),
                        self.binding.clone(),
                    )
                    .unwrap()
            });
            let coordinator = self
                .authority
                .enrollment_reader()
                .unwrap()
                .connect_generation(
                    TcpStream::connect(listener.local_addr().unwrap()).unwrap(),
                    self.identity.node(),
                    self.binding.clone(),
                )
                .unwrap();
            (coordinator, server.join().unwrap())
        })
    }
    fn client(&self) -> (TcpStream, ChatRequest, ClientResponse) {
        self.try_client(encode(&self.chat, FixtureMode::Reject).unwrap())
            .unwrap()
    }
    fn try_client(
        &self,
        body: Vec<u8>,
    ) -> std::result::Result<(TcpStream, ChatRequest, ClientResponse), starlings_security::Error>
    {
        let listener =
            ClientListener::bind("127.0.0.1:0".parse().unwrap(), ListenerBudget::default())
                .unwrap();
        assert!(listener.try_accept().unwrap().is_none());
        let address = listener.local_addr().unwrap();
        let token = self
            .authority
            .issue_client_token(
                self.controller,
                &[Scope {
                    model: self.binding.model.clone(),
                    action: Action::Infer,
                }],
                60,
            )
            .unwrap();
        let mut socket = TcpStream::connect(address).unwrap();
        socket
            .set_read_timeout(Some(Duration::from_secs(5)))
            .unwrap();
        write!(socket, "POST /v1/chat/completions HTTP/1.1\r\nHost: {address}\r\nAuthorization: Bearer {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n", token.bearer(), body.len()).unwrap();
        socket.write_all(&body).unwrap();
        let (chat, response) = admit(
            listener.accept().unwrap(),
            &self.authority.client_admission(),
            &self.binding,
        )?;
        Ok((socket, chat, response))
    }
}

#[test]
fn sustained_registry_maintenance_and_semantic_parser_are_exercised() {
    let f = Fixture::new("normal", false);
    for _ in 0..3 {
        f.while_maintaining(thread::yield_now);
    }
    let record = relay_acceptance(f.config.clone());
    assert_eq!(record["runs"].as_array().unwrap().len(), 2);
    assert_eq!(record["memory_qualified"], false);
    f.while_maintaining(|| {
        assert_eq!(
            f.fence.status().unwrap().1,
            starlings_security::NodeStatus::Online
        );
    });
    let stream = "data: {\"choices\":[{\"delta\":{\"content\":\"ready\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":1,\"total_tokens\":4}}\n\ndata: [DONE]\n\n";
    assert_eq!(stream_semantic(stream)["content"], "ready");
}

#[test]
fn enrolled_relay_preserves_json_sse_and_fragmented_output() {
    for (mode, streaming) in [
        ("normal", false),
        ("normal", true),
        ("large", false),
        ("large", true),
    ] {
        let f = Fixture::new(mode, streaming);
        f.while_maintaining(|| {
            let mut service = f.service();
            let mut direct = Vec::new();
            service
                .generate(&f.chat, |bytes| {
                    direct.extend_from_slice(bytes);
                    Ok(())
                })
                .unwrap();
            let (coordinator, remote) = f.pair();
            let (mut client, chat, response) = f.client();
            assert_eq!(response.client(), f.controller);
            let Operation::Prepare { plan, .. } = &f.controls[0].operation else {
                panic!("prepare")
            };
            let mut dispatch =
                Dispatch::new(f.binding.clone(), *plan.clone(), f.worker.profile(), 5_000).unwrap();
            let done = {
                thread::scope(|threads| {
                    let worker = threads.spawn(|| serve_worker(&mut service, &f.worker, remote));
                    let reader = threads.spawn(move || {
                        let mut body = String::new();
                        client.read_to_string(&mut body).unwrap();
                        body
                    });
                    let relayed = dispatch.run(response, &chat, coordinator);
                    let executed = worker.join().unwrap();
                    assert!(
                        relayed.is_ok(),
                        "{mode} streaming={streaming}: relay={relayed:?}, worker_error={:?}",
                        executed.as_ref().err()
                    );
                    let done = executed.unwrap().unwrap();
                    let body = reader.join().unwrap();
                    let (_, payload) = body.split_once("\r\n\r\n").unwrap();
                    if streaming {
                        assert!(body.contains("X-Compute-Session-Id:"));
                        assert_eq!(
                            payload,
                            format!(
                                "data: {}\n\ndata: [DONE]\n\n",
                                String::from_utf8(direct).unwrap()
                            )
                        );
                    } else {
                        assert_eq!(payload.as_bytes(), direct);
                    }
                    assert!(
                        f.worker.snapshot().unwrap().reserved_bytes.0 > 0,
                        "response completion cannot free native reservations"
                    );
                    done
                })
            };
            service.stop().unwrap();
            // This is the synthetic child fixture only; no native drain qualification.
            f.worker
                .failed_and_drained(done.generation, done.invocation)
                .unwrap();
            assert_eq!(f.worker.snapshot().unwrap().reserved_bytes.0, 0);
        });
    }
}

#[test]
fn counting_failure_returns_http_error_before_any_generation_or_sse_headers() {
    let f = Fixture::new("bad-count", true);
    f.while_maintaining(|| {
        let mut service = f.service();
        let (coordinator, remote) = f.pair();
        let (mut client, chat, response) = f.client();
        let Operation::Prepare { plan, .. } = &f.controls[0].operation else {
            panic!("prepare")
        };
        let mut dispatch =
            Dispatch::new(f.binding.clone(), *plan.clone(), f.worker.profile(), 5000).unwrap();
        thread::scope(|threads| {
            let worker = threads.spawn(|| serve_worker(&mut service, &f.worker, remote));
            assert!(dispatch.run(response, &chat, coordinator).is_err());
            assert!(worker.join().unwrap().is_err());
        });
        let mut body = String::new();
        client.read_to_string(&mut body).unwrap();
        assert!(body.starts_with("HTTP/1.1 503 Service Unavailable"));
        assert!(!body.contains("text/event-stream"));
        assert!(!f.config.directory.join("started").exists());
        assert_eq!(f.worker.snapshot().unwrap().reserved_bytes.0, 0);
        service.stop().unwrap();
    });
}

#[test]
fn backend_failure_emits_one_redacted_terminal_error_without_retry() {
    for streaming in [false, true] {
        let f = Fixture::new("crash", streaming);
        f.while_maintaining(|| {
            let mut service = f.service();
            let (mut coordinator, remote) = f.pair();
            if !streaming {
                thread::sleep(starlings_security::REGISTRY_TTL + Duration::from_millis(100));
                assert_eq!(
                    f.fence.status().unwrap().1,
                    starlings_security::NodeStatus::Online
                );
            }
            let (mut client, chat, response) = f.client();
            thread::scope(|threads| {
                let worker = threads.spawn(|| serve_worker(&mut service, &f.worker, remote));
                assert_eq!(count_prompt(&response, &chat, &mut coordinator).unwrap(), 3);
                assert!(relay(response, &chat, f.controls.clone(), coordinator).is_err());
                assert!(worker.join().unwrap().is_err());
                let mut body = String::new();
                client.read_to_string(&mut body).unwrap();
                if streaming {
                    assert_eq!(body.matches("event: error").count(), 1);
                } else {
                    assert!(body.starts_with("HTTP/1.1 503 Service Unavailable"));
                    let (_, payload) = body.split_once("\r\n\r\n").unwrap();
                    let error: WireError = decode(payload.as_bytes(), FixtureMode::Reject).unwrap();
                    assert!(!error.error.retryable);
                }
                assert!(body.contains("WORKER_LOST"));
                assert!(!body.contains("[DONE]"));
                assert!(!body.contains("synthetic input"));
                assert_eq!(
                    std::fs::read_to_string(f.config.directory.join("started")).unwrap(),
                    "started\n"
                );
                assert!(f.worker.snapshot().unwrap().reserved_bytes.0 > 0);
            });
            service.stop().unwrap();
        });
    }
}

#[test]
fn admission_rejects_model_mismatch_unresolved_sampling_and_unsupported_tools() {
    let f = Fixture::new("normal", false);
    for invalid in ["model", "sampling", "tools"] {
        let mut body = serde_json::to_value(&f.chat).unwrap();
        match invalid {
            "model" => body["model"] = serde_json::json!(byte_digest(b"different model")),
            "sampling" => {
                body.as_object_mut().unwrap().remove("seed");
            }
            _ => body["tools"] = serde_json::json!([]),
        }
        assert!(f.try_client(serde_json::to_vec(&body).unwrap()).is_err());
    }
    assert_eq!(f.worker.snapshot().unwrap().reserved_bytes.0, 0);
    assert!(!f.config.directory.join("started").exists());
}

#[test]
fn worker_rejects_changed_prompt_count_or_context_before_generation() {
    for invalid in ["count", "prompt", "context"] {
        let mut f = Fixture::new("normal", false);
        if invalid == "context" {
            f.chat.max_tokens = BoundedU32::new(2048).unwrap();
        }
        f.while_maintaining(|| {
            let mut service = f.service();
            let (mut coordinator, remote) = f.pair();
            let (_client, chat, response) = f.client();
            thread::scope(|threads| {
                let worker = threads.spawn(|| serve_worker(&mut service, &f.worker, remote));
                if invalid == "context" {
                    assert!(count_prompt(&response, &chat, &mut coordinator).is_err());
                } else {
                    assert_eq!(count_prompt(&response, &chat, &mut coordinator).unwrap(), 3);
                    let mut controls = f.controls.clone();
                    let Operation::StartRequest {
                        session, request, ..
                    } = &mut controls[2].operation
                    else {
                        panic!("wrong test control");
                    };
                    if invalid == "count" {
                        session.prompt_token_count = BoundedU32::new(4).unwrap();
                    } else {
                        request.messages[0].content.push_str("changed");
                    }
                    controls[2] = ControlRequest::new(
                        controls[2].request_id,
                        controls[2].authority_epoch.0,
                        controls[2].request_sequence.0,
                        controls[2].operation.clone(),
                    )
                    .unwrap();
                    for (index, request) in controls.into_iter().enumerate() {
                        coordinator
                            .send_record(&Command::Apply {
                                request: Box::new(request),
                            })
                            .unwrap();
                        let reply = coordinator.receive_record::<Reply>();
                        if index == 2 {
                            assert!(reply.is_err());
                        } else {
                            assert!(matches!(reply, Ok(Reply::Applied { .. })));
                        }
                    }
                }
                assert!(worker.join().unwrap().is_err());
                assert_eq!(f.worker.snapshot().unwrap().reserved_bytes.0, 0);
                assert!(!f.config.directory.join("started").exists());
            });
            service.stop().unwrap();
        });
    }
}

#[test]
fn client_disconnect_revocation_and_registry_drain_cancel_before_first_output() {
    for failure in ["disconnect", "revoke", "drain", "partial"] {
        let f = Fixture::new(
            if failure == "partial" {
                "partial"
            } else {
                "stall"
            },
            true,
        );
        f.while_maintaining(|| {
            let mut service = f.service();
            let (mut coordinator, remote) = f.pair();
            let (mut client, chat, response) = f.client();
            thread::scope(|threads| {
                let worker = threads.spawn(|| serve_worker(&mut service, &f.worker, remote));
                let relay = threads.spawn(|| {
                    count_prompt(&response, &chat, &mut coordinator)?;
                    relay(response, &chat, f.controls.clone(), coordinator)
                });
                let deadline = Instant::now() + Duration::from_secs(3);
                while !f.config.directory.join("started").exists() {
                    assert!(Instant::now() < deadline, "backend was never invoked");
                    thread::sleep(Duration::from_millis(5));
                }
                let cancelled = Instant::now();
                match failure {
                    "partial" => {
                        let mut received = String::new();
                        while !received.ends_with("\n\n") {
                            let mut byte = [0];
                            client.read_exact(&mut byte).unwrap();
                            received.push(char::from(byte[0]));
                        }
                        assert!(received.contains("partial"));
                        assert!(!received.contains("[DONE]"));
                        drop(client);
                    }
                    "disconnect" => drop(client),
                    "revoke" => f.authority.revoke(f.controller).unwrap(),
                    _ => f.fence.drain().unwrap(),
                }
                assert!(relay.join().unwrap().is_err());
                assert!(worker.join().unwrap().is_err());
                assert!(cancelled.elapsed() < Duration::from_secs(4));
                assert!(f.worker.snapshot().unwrap().reserved_bytes.0 > 0);
                assert_eq!(
                    std::fs::read_to_string(f.config.directory.join("started")).unwrap(),
                    "started\n"
                );
            });
            service.stop().unwrap();
        });
    }
}

#[test]
fn connection_closed_before_count_does_not_cancel_the_loaded_service() {
    let f = Fixture::new("normal", false);
    f.while_maintaining(|| {
        let mut service = f.service();
        for invalid_first in [false, true] {
            let (mut coordinator, remote) = f.pair();
            if invalid_first {
                coordinator
                    .send_record(&Command::Poll {
                        generation: "00000000-0000-4000-8000-000000000777".parse().unwrap(),
                    })
                    .unwrap();
            }
            drop(coordinator);
            let result = serve_worker(&mut service, &f.worker, remote);
            assert_eq!(f.worker.snapshot().unwrap().reserved_bytes.0, 0);
            assert!(
                service.check_alive().is_ok(),
                "idle connection cancelled the loaded service"
            );
            assert!(matches!(result, Ok(None)), "no Count was accepted");
            assert!(!f.config.directory.join("started").exists());
        }
        // A fresh request uses that same loaded Service and worker; no restart/recovery.
        let (coordinator, remote) = f.pair();
        let (mut client, chat, response) = f.client();
        let Operation::Prepare { plan, .. } = &f.controls[0].operation else {
            panic!("prepare")
        };
        let mut dispatch =
            Dispatch::new(f.binding.clone(), *plan.clone(), f.worker.profile(), 5_000).unwrap();
        let done = thread::scope(|threads| {
            let worker = threads.spawn(|| serve_worker(&mut service, &f.worker, remote));
            let reader = threads.spawn(move || {
                let mut body = String::new();
                client.read_to_string(&mut body).unwrap();
                body
            });
            dispatch.run(response, &chat, coordinator).unwrap();
            let done = worker.join().unwrap().unwrap().unwrap();
            assert!(reader.join().unwrap().starts_with("HTTP/1.1 200 OK"));
            done
        });
        assert_eq!(
            std::fs::read_to_string(f.config.directory.join("started")).unwrap(),
            "started\n"
        );
        assert!(f.worker.snapshot().unwrap().reserved_bytes.0 > 0);
        service.stop().unwrap();
        f.worker
            .failed_and_drained(done.generation, done.invocation)
            .unwrap();
    });
}

#[test]
fn accepted_count_or_existing_reservation_cannot_be_treated_as_an_idle_disconnect() {
    for counted in [false, true] {
        let f = Fixture::new("normal", false);
        f.while_maintaining(|| {
            let mut service = f.service();
            let (mut coordinator, remote) = f.pair();
            if counted {
                thread::scope(|threads| {
                    let worker = threads.spawn(|| serve_worker(&mut service, &f.worker, remote));
                    coordinator
                        .send_record(&Command::Count {
                            request: Box::new(f.chat.clone()),
                        })
                        .unwrap();
                    assert!(matches!(
                        coordinator.receive_record::<Reply>().unwrap(),
                        Reply::Counted { .. }
                    ));
                    drop(coordinator);
                    assert!(worker.join().unwrap().is_err());
                });
            } else {
                // An older accepted lease prevents the empty-runtime proof, even
                // when the new connection has not supplied its first Count.
                f.worker.apply(&remote, &f.controls[0]).unwrap();
                let (other, incoming) = f.pair();
                drop(other);
                assert!(serve_worker(&mut service, &f.worker, incoming).is_err());
                assert!(f.worker.snapshot().unwrap().reserved_bytes.0 > 0);
            }
            assert!(
                service.check_alive().is_err(),
                "ambiguous/owned work remains conservative"
            );
            service.stop().unwrap();
        });
    }
}
