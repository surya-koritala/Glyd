use super::*;

#[test]
fn fixture_process() {
    super::fixture::run();
}

fn fixture_config(mode: &str) -> ServiceConfig {
    let endpoint = "127.0.0.1:1".parse::<SocketAddr>().unwrap();
    let executable = std::env::current_exe().unwrap().canonicalize().unwrap();
    let mut environment = BTreeMap::from([
        ("COMPUTE_FIXTURE_ENDPOINT".into(), endpoint.to_string()),
        ("COMPUTE_FIXTURE_MODE".into(), mode.into()),
    ]);
    if let Some(root) = std::env::var_os("SystemRoot") {
        environment.insert("SYSTEMROOT".into(), root.to_string_lossy().into_owned());
    }
    ServiceConfig {
        executable_sha256: executable_digest(&executable).unwrap(),
        executable,
        arguments: vec![
            "--exact".into(),
            "service::tests::fixture_process".into(),
            "--nocapture".into(),
        ],
        directory: std::env::current_dir().unwrap().canonicalize().unwrap(),
        environment,
        key_environment: "COMPUTE_FIXTURE_KEY".into(),
        endpoint,
        backend_version: "synthetic-v1".into(),
        properties: None,
        glyd_identity: mode
            .starts_with("glyd-")
            .then(|| GlydIdentity::from_info(&super::fixture::glyd_info()).unwrap()),
        api: if mode.starts_with("glyd-") {
            ServiceApi::GlydServeV1
        } else {
            ServiceApi::InputTokensV1
        },
        manifest_digest: starlings_protocol::byte_digest(b"synthetic manifest"),
        execution_profile: "synthetic-f32".into(),
        backend_model: "synthetic-backend".into(),
        reasoning_content: true,
        request_timeout_ms: 45_000,
    }
}
fn start(config: ServiceConfig) -> Service {
    let mut service =
        super::fixture::spawn(config, "syntheticTestKey00000000000000000000".into()).unwrap();
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        let ready = service.probe();
        if ready.is_ok() {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "fixture service never became ready: {ready:?}"
        );
        std::thread::sleep(Duration::from_millis(10));
    }
    service
}
fn request(config: &ServiceConfig, stream: bool) -> ChatRequest {
    let mut request: ChatRequest = starlings_protocol::decode(
        include_bytes!("../../../../examples/chat-request.json"),
        FixtureMode::Reject,
    )
    .unwrap();
    request.model = config.manifest_digest.to_string();
    request.stream = Some(stream);
    request
}

#[test]
fn glyd_identity_and_request_cancel_preserve_ownership_and_refuse_false_success() {
    let config = fixture_config("glyd-cancel");
    let chat = request(&config, true);
    let mut service = start(config);
    assert!(service.glyd_identity().is_some());
    assert_eq!(service.glyd_memory().unwrap().kv_bytes, Some(20));
    assert_eq!(service.confirm_request_idle(), Err(Error::Unapproved));
    assert_eq!(service.count_prompt(&chat), Ok(3));
    let first = service.request_cancellation().unwrap();
    let mut events = 0;
    assert_eq!(
        service.generate(&chat, |_| {
            events += 1;
            first.cancel()?;
            Err(Error::Cancelled)
        }),
        Err(Error::Cancelled)
    );
    assert_eq!(events, 1);
    service.check_alive().unwrap();
    assert_eq!(service.glyd_memory().unwrap().engine_bytes, 100);
    service.probe().unwrap();
    assert_eq!(service.count_prompt(&chat), Ok(3));
    let next = service.request_cancellation().unwrap();
    assert_ne!(first.id(), next.id());
    first.cancel().unwrap();
    service
        .generate(&chat, |_| {
            first.cancel()?;
            Ok(())
        })
        .unwrap();
    service.check_alive().unwrap();
    service.count_prompt(&chat).unwrap();
    let before_dispatch = service.request_cancellation().unwrap();
    before_dispatch.cancel().unwrap();
    assert_eq!(
        service.generate(&chat, |_| panic!("pre-cancelled request ran")),
        Err(Error::Cancelled)
    );
    service.check_alive().unwrap();
    service.stop().unwrap();
    assert!(service.glyd_identity().is_none());
    assert!(service.glyd_memory().is_none());

    let config = fixture_config("glyd-cancelled-stop");
    let mut service = start(config.clone());
    for stream in [false, true] {
        assert_eq!(
            service.generate(&request(&config, stream), |_| panic!(
                "cancelled stop emitted success"
            )),
            Err(Error::Cancelled)
        );
        service.check_alive().unwrap();
    }
    service.stop().unwrap();

    let config = fixture_config("glyd-unconfirmed");
    let mut service = start(config.clone());
    let chat = request(&config, true);
    service.count_prompt(&chat).unwrap();
    let control = service.request_cancellation().unwrap();
    assert!(
        service
            .generate(&chat, |_| {
                assert!(control.cancel().is_err());
                Err(Error::Cancelled)
            })
            .is_err()
    );
    assert!(service.check_alive().is_err());
    service.stop().unwrap();

    let mut service =
        super::fixture::spawn(fixture_config("glyd-identity-mismatch"), "a".repeat(32)).unwrap();
    assert_eq!(service.probe(), Err(Error::Unapproved));
    assert!(service.glyd_identity().is_none());
    service.stop().unwrap();
}

#[test]
fn glyd_cancel_response_must_settle_before_service_reuse() {
    use std::sync::mpsc::{TryRecvError, channel};
    for fail in [false, true] {
        let root = std::env::temp_dir().join(format!("cancel-settlement-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir(&root).unwrap();
        let started = root.join("started");
        let release = root.join("release");
        let mut config = fixture_config(if fail {
            "glyd-cancel-pending-fail"
        } else {
            "glyd-cancel-pending"
        });
        config.environment.insert(
            "COMPUTE_FIXTURE_CANCEL_STARTED".into(),
            started.to_str().unwrap().into(),
        );
        config.environment.insert(
            "COMPUTE_FIXTURE_RELEASE".into(),
            release.to_str().unwrap().into(),
        );
        let chat = request(&config, true);
        let mut service = start(config);
        service.count_prompt(&chat).unwrap();
        let control = service.request_cancellation().unwrap();
        std::thread::scope(|threads| {
            let (partial, saw_partial) = channel();
            let (finished, returned) = channel();
            let active = &mut service;
            let request = &chat;
            let generation = threads.spawn(move || {
                let result = active.generate(request, |_| {
                    partial.send(()).unwrap();
                    Ok(())
                });
                finished.send(result).unwrap();
            });
            saw_partial.recv_timeout(Duration::from_secs(2)).unwrap();
            let cancellation = threads.spawn(|| control.cancel());
            let deadline = Instant::now() + Duration::from_secs(2);
            while !started.exists() {
                assert!(
                    Instant::now() < deadline,
                    "cancel never reached the fixture"
                );
                std::thread::sleep(Duration::from_millis(5));
            }
            // Normal status sees ended, but the first cancel response is still withheld.
            assert_eq!(control.finish(), Err(Error::Cancelled));
            assert!(!control.cancelled_and_ended());
            assert_eq!(returned.try_recv(), Err(TryRecvError::Empty));
            std::fs::write(&release, b"release").unwrap();
            assert_eq!(cancellation.join().unwrap().is_err(), fail);
            generation.join().unwrap();
            let result = returned.recv().unwrap();
            if fail {
                assert!(result.is_err());
            } else {
                assert_eq!(result, Err(Error::Cancelled));
            }
        });
        if fail {
            assert!(service.control.cancelled.load(Ordering::Acquire));
            assert!(service.check_alive().is_err());
        } else {
            service.check_alive().unwrap();
            service.count_prompt(&chat).unwrap();
            service.generate(&chat, |_| control.cancel()).unwrap();
            service.check_alive().unwrap();
        }
        service.stop().unwrap();
        std::fs::remove_dir_all(root).unwrap();
    }
}

#[test]
fn glyd_completion_requires_ready_identity_and_retained_memory_before_terminal_output() {
    for mode in [
        "glyd-memory-reset",
        "glyd-memory-invalid",
        "glyd-still-running",
    ] {
        for stream in [false, true] {
            let config = fixture_config(mode);
            let mut service = start(config.clone());
            let mut terminal = 0;
            assert!(
                service
                    .generate(&request(&config, stream), |bytes| {
                        let value = parse_json(bytes).unwrap();
                        if !stream || value["choices"][0]["finish_reason"].as_str().is_some() {
                            terminal += 1;
                        }
                        Ok(())
                    })
                    .is_err(),
                "{mode}/{stream}"
            );
            assert_eq!(terminal, 0, "{mode}/{stream}");
            assert!(service.glyd_memory().is_none());
            assert!(service.check_alive().is_err());
            // Allocator/status receipts still cannot authorize the CPU drain path.
            assert_eq!(service.confirm_request_idle(), Err(Error::Unapproved));
            service.stop().unwrap();
        }
    }
}

#[test]
fn release_barrier_uses_the_request_budget_and_expires_without_a_controller() {
    for release in [true, false] {
        let root = std::env::temp_dir().join(format!("fixture-release-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir(&root).unwrap();
        let marker = root.join("started");
        let barrier = root.join("release");
        let mut config = fixture_config("normal");
        config.request_timeout_ms = if release { 10_000 } else { 1000 };
        config.environment.insert(
            "COMPUTE_FIXTURE_STARTED".into(),
            marker.to_str().unwrap().into(),
        );
        config.environment.insert(
            "COMPUTE_FIXTURE_RELEASE".into(),
            barrier.to_str().unwrap().into(),
        );
        let chat = request(&config, false);
        let mut service = start(config);
        let result = std::thread::scope(|threads| {
            let generation = threads.spawn(|| service.generate(&chat, |_| Ok(())));
            let deadline = Instant::now() + Duration::from_secs(3);
            while !marker.exists() {
                assert!(
                    Instant::now() < deadline,
                    "fixture did not enter generation"
                );
                std::thread::sleep(Duration::from_millis(5));
            }
            if release {
                // Outlast the unrelated old five-second barrier, within the existing request budget.
                std::thread::sleep(Duration::from_millis(5500));
                std::fs::write(&barrier, b"release").unwrap();
            }
            generation.join().unwrap()
        });
        if !release {
            let deadline = Instant::now() + Duration::from_secs(3);
            while service.check_alive().is_ok() {
                assert!(
                    Instant::now() < deadline,
                    "unreleased fixture did not expire"
                );
                std::thread::sleep(Duration::from_millis(5));
            }
        }
        service.stop().unwrap();
        std::fs::remove_dir_all(root).unwrap();
        assert_eq!(result.is_ok(), release, "{result:?}");
    }
}
#[test]
fn split_endpoint_report_never_selects_a_valid_partial_port() {
    let report = std::env::temp_dir().join(format!("endpoint-report-{}", uuid::Uuid::new_v4()));
    // A paused write of :49152 leaves a syntactically valid, incorrect :4.
    std::fs::write(&report, b"127.0.0.1:4").unwrap();
    assert_eq!(super::fixture::reported_endpoint(&report), None);
    std::fs::write(&report, b"127.0.0.1:49152\n").unwrap();
    assert_eq!(
        super::fixture::reported_endpoint(&report),
        Some("127.0.0.1:49152".parse().unwrap())
    );
    for invalid in [
        "127.0.0.1:0\n",
        "192.0.2.1:49152\n",
        "127.0.0.1:49152\nextra",
    ] {
        std::fs::write(&report, invalid).unwrap();
        assert_eq!(super::fixture::reported_endpoint(&report), None);
    }
    std::fs::remove_file(report).unwrap();
}

#[test]
fn startup_phases_locate_refusal_without_exposing_configuration_or_skipping_checks() {
    let config = fixture_config("normal");
    let key = "syntheticSecret00000000000000000000".to_string();
    let mut phases = Vec::new();
    assert!(matches!(
        Service::spawn_observed(
            config.clone(),
            &starlings_protocol::byte_digest(b"unapproved"),
            key.clone(),
            |phase| phases.push(phase)
        ),
        Err(Error::Unapproved)
    ));
    assert_eq!(phases, ["configuration_digest"]);
    phases.clear();
    let mut wrong_build = config.clone();
    wrong_build.executable_sha256 = starlings_protocol::byte_digest(b"different executable");
    assert!(matches!(
        Service::spawn_observed(
            wrong_build.clone(),
            &wrong_build.digest().unwrap(),
            key.clone(),
            |phase| phases.push(phase)
        ),
        Err(Error::Unapproved)
    ));
    assert_eq!(
        phases,
        [
            "configuration_digest",
            "configuration_paths",
            "executable_digest"
        ]
    );
    phases.clear();
    let mut service =
        super::fixture::spawn_observed(config, key, |phase| phases.push(phase)).unwrap();
    assert_eq!(
        phases,
        [
            "fixture_directory",
            "configuration_digest",
            "configuration_paths",
            "executable_digest",
            "key_validation",
            "process_spawn",
            "fixture_endpoint"
        ]
    );
    assert!(service.properties().is_none());
    service.probe().unwrap();
    service.stop().unwrap();
}

#[test]
fn an_aborted_probe_connection_does_not_kill_the_fixture() {
    use std::{
        io::Write,
        net::{Shutdown, TcpStream},
    };
    let mut service = start(fixture_config("normal"));
    let mut peer = TcpStream::connect(service.config.endpoint).unwrap();
    peer.set_read_timeout(Some(Duration::from_secs(3))).unwrap();
    peer.write_all(b"GET /health HTTP/1.1\r\n").unwrap();
    peer.shutdown(Shutdown::Write).unwrap();
    // EOF waits for this specific abandoned connection to close; no timing guess or retry.
    peer.read_to_end(&mut Vec::new()).unwrap();
    service.probe().unwrap();
    service.stop().unwrap();
}
#[test]
fn loading_503_never_grants_readiness_or_weakens_authentication() {
    let mut service = super::fixture::spawn(fixture_config("loading"), "a".repeat(32)).unwrap();
    assert_eq!(service.probe(), Err(Error::Unavailable));
    assert!(!service.ready);
    service.probe().unwrap();
    assert!(service.ready);
    service.stop().unwrap();

    let mut service = super::fixture::spawn(fixture_config("unprotected"), "a".repeat(32)).unwrap();
    assert_eq!(service.probe(), Err(Error::Unapproved));
    assert!(!service.ready);
    service.stop().unwrap();
}
#[test]
fn installed_properties_are_approved_bounded_and_cannot_grant_readiness_on_mismatch() {
    let mut config = fixture_config("normal");
    let original = config.digest().unwrap();
    assert!(
        serde_json::to_value(&config)
            .unwrap()
            .get("properties")
            .is_none()
    );
    let decoded: ServiceConfig =
        serde_json::from_value(serde_json::to_value(&config).unwrap()).unwrap();
    assert_eq!(decoded.digest().unwrap(), original);
    config.properties = Some(ExpectedProperties {
        build_info: "synthetic-v1".into(),
        context_tokens: 512,
    });
    assert_eq!(config.validate(&original), Err(Error::Unapproved));
    config.validate(&config.digest().unwrap()).unwrap();
    let mut service = start(config.clone());
    let report = serde_json::to_value(service.properties().unwrap()).unwrap();
    assert_eq!(report["build_info"], "synthetic-v1");
    assert_eq!(report["context_tokens"], 512);
    assert_eq!(report["system_role"], true);
    assert!(!report.to_string().contains("not-retained"));
    service.stop().unwrap();
    assert!(service.properties().is_none());

    config
        .environment
        .insert("COMPUTE_FIXTURE_MODE".into(), "properties-mismatch".into());
    let mut service = super::fixture::spawn(config, "a".repeat(32)).unwrap();
    let mut phase = "";
    assert_eq!(
        service.probe_observed(|p| phase = p),
        Err(Error::Unapproved)
    );
    assert_eq!(phase, "properties");
    assert!(!service.ready);
    assert!(service.properties().is_none());
    service.stop().unwrap();

    let expected = ExpectedProperties {
        build_info: "b1-pinned".into(),
        context_tokens: 512,
    };
    let valid = serde_json::json!({
        "build_info": "b1-pinned", "model_alias": "model", "total_slots": 1,
        "default_generation_settings": {"n_ctx": 512},
        "chat_template_caps": {"supports_system_role": true, "supports_string_content": true, "supports_typed_content": false},
        "model_path": "/private/not-retained", "chat_template": "not-retained"
    });
    parse_properties(&valid, &expected, "model").unwrap();
    for (pointer, value) in [
        ("/build_info", serde_json::json!("wrong")),
        ("/model_alias", serde_json::json!("wrong")),
        ("/total_slots", serde_json::json!(2)),
        ("/default_generation_settings/n_ctx", serde_json::json!(256)),
        (
            "/default_generation_settings/n_ctx",
            serde_json::json!(u64::MAX),
        ),
        (
            "/chat_template_caps/supports_system_role",
            serde_json::json!(false),
        ),
        (
            "/chat_template_caps/supports_string_content",
            serde_json::json!(false),
        ),
        (
            "/chat_template_caps/supports_typed_content",
            serde_json::json!(null),
        ),
    ] {
        let mut invalid = valid.clone();
        *invalid.pointer_mut(pointer).unwrap() = value;
        assert!(
            parse_properties(&invalid, &expected, "model").is_err(),
            "{pointer}"
        );
    }
    let mut typed = valid.clone();
    typed["chat_template_caps"]["supports_string_content"] = false.into();
    typed["chat_template_caps"]["supports_typed_content"] = true.into();
    assert!(parse_properties(&typed, &expected, "model").is_ok());
    assert_eq!(
        read_json(&b"{\"build_info\":1,\"build_info\":2}"[..], 64 * 1024),
        Err(Error::Invalid)
    );
    assert_eq!(
        read_json(&vec![b' '; 64 * 1024 + 1][..], 64 * 1024),
        Err(Error::Capacity)
    );
}
#[test]
fn readiness_attempt_cannot_multiply_its_io_deadline() {
    let mut config = fixture_config("slow-probe");
    config.properties = Some(ExpectedProperties {
        build_info: "synthetic-v1".into(),
        context_tokens: 512,
    });
    let mut service = super::fixture::spawn(config, "a".repeat(32)).unwrap();
    let started = Instant::now();
    let result = service.probe();
    assert!(
        result.is_err(),
        "slow cumulative readiness incorrectly succeeded"
    );
    assert!(!service.ready);
    assert!(service.properties().is_none());
    assert!(started.elapsed() < Duration::from_secs(3));
    service.stop().unwrap();
}
#[test]
fn approved_service_preserves_output_rejects_unsupported_fields_and_cancels() {
    let config = fixture_config("normal");
    let approved = config.digest().unwrap();
    let mut changed = config.clone();
    changed.arguments.push("unapproved".into());
    assert!(matches!(
        Service::spawn(changed, &approved, "a".repeat(32)),
        Err(Error::Unapproved)
    ));
    let mut changed = config.clone();
    changed.endpoint = "192.168.1.1:9000".parse().unwrap();
    assert!(matches!(
        Service::spawn(changed.clone(), &changed.digest().unwrap(), "a".repeat(32)),
        Err(Error::Invalid)
    ));
    let mut service = start(config.clone());
    for stream in [false, true] {
        let mut events = Vec::new();
        service
            .generate(&request(&config, stream), |bytes| {
                events.push(parse_json(bytes).unwrap());
                Ok(())
            })
            .unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(
            events[0]["choices"][0][if stream { "delta" } else { "message" }]["content"],
            "synthetic answer"
        );
        assert_eq!(
            events[0]["choices"][0][if stream { "delta" } else { "message" }]["reasoning_content"],
            "synthetic reasoning"
        );
        assert_eq!(events[0]["usage"]["total_tokens"], 4);
    }
    service.stop().unwrap();
    assert!(service.probe().is_err());
    let mut service = start(config.clone());
    let control = service.cancellation();
    let mut emitted = 0;
    let result = service.generate(&request(&config, true), |_| {
        emitted += 1;
        control.cancel()?;
        Err(Error::Cancelled)
    });
    assert_eq!(result, Err(Error::Cancelled));
    assert_eq!(emitted, 1);
    service.stop().unwrap();
    assert!(service.probe().is_err());
    for mode in ["tools", "crash"] {
        let config = fixture_config(mode);
        let request = request(&config, true);
        let mut service = start(config);
        assert!(
            service
                .generate(&request, |_| panic!("invalid backend emitted output"))
                .is_err()
        );
        service.stop().unwrap();
    }
    let config = fixture_config("stall");
    let request = request(&config, true);
    let mut service = start(config);
    let control = service.cancellation();
    let started = Instant::now();
    std::thread::scope(|threads| {
        let generation = threads
            .spawn(|| service.generate(&request, |_| panic!("cancelled backend emitted output")));
        std::thread::sleep(Duration::from_millis(50));
        control.cancel().unwrap();
        assert!(generation.join().unwrap().is_err());
    });
    service.stop().unwrap();
    assert!(started.elapsed() < Duration::from_secs(3));
}

#[test]
fn output_identity_finish_usage_and_reasoning_are_strict() {
    let mut config = fixture_config("normal");
    let chunk = serde_json::json!({"id":"one", "model":"synthetic-backend", "object":"chat.completion.chunk",
        "choices":[{"index":0,"delta":{"content":"text"},"finish_reason":null}]});
    let mut output = Output::default();
    validate_output(&chunk, &config, true, 8, &mut output).unwrap();
    let mut wrong = chunk.clone();
    wrong["id"] = "two".into();
    assert!(validate_output(&wrong, &config, true, 8, &mut output).is_err());
    let mut final_chunk = chunk.clone();
    final_chunk["choices"][0]["finish_reason"] = "stop".into();
    final_chunk["usage"] =
        serde_json::json!({"prompt_tokens":3,"completion_tokens":1,"total_tokens":4});
    validate_output(&final_chunk, &config, true, 8, &mut output).unwrap();
    assert!(output.finished && output.usage);
    assert!(validate_output(&chunk, &config, true, 8, &mut output).is_err());
    for field in ["total_tokens", "completion_tokens"] {
        let mut invalid = final_chunk.clone();
        invalid["usage"][field] = 9.into();
        assert!(validate_output(&invalid, &config, true, 8, &mut Output::default()).is_err());
    }
    config.reasoning_content = false;
    let mut invalid = final_chunk;
    invalid["choices"][0]["delta"]["reasoning_content"] = "unsupported".into();
    assert!(validate_output(&invalid, &config, true, 8, &mut Output::default()).is_err());
}

#[test]
fn glyd_serve_wire_contract_is_opt_in_strict_and_preserves_legacy_digests() {
    let digest = starlings_protocol::byte_digest(b"wire contract");
    let original = serde_json::json!({"executable":"/not-opened", "executable_sha256":digest,
        "arguments":[], "directory":"/not-opened", "environment":{}, "key_environment":"KEY",
        "endpoint":"127.0.0.1:1", "backend_version":"test", "manifest_digest":digest,
        "execution_profile":"test", "backend_model":"backend", "reasoning_content":true,
        "request_timeout_ms":1000});
    let mut config: ServiceConfig = serde_json::from_value(original.clone()).unwrap();
    assert!(config.api == ServiceApi::InputTokensV1);
    assert!(serde_json::to_value(&config).unwrap().get("api").is_none());
    assert_eq!(config.digest().unwrap(), content_hash(&original).unwrap());
    let legacy_digest = config.digest().unwrap();
    config.api = ServiceApi::GlydServeV1;
    assert_ne!(config.digest().unwrap(), legacy_digest);
    let mut unknown = original;
    unknown["api"] = "unsupported".into();
    assert!(serde_json::from_value::<ServiceConfig>(unknown).is_err());
    assert_eq!(config.api.count_path(), "/v1/glyd/tokenize");
    assert_eq!(
        ServiceApi::InputTokensV1.count_path(),
        "/v1/chat/completions/input_tokens"
    );
    let counted = serde_json::json!({"prompt_tokens":3,"max_model_len":512});
    assert_eq!(parse_prompt_count(counted.clone(), config.api, 509), Ok(3));
    assert!(parse_prompt_count(counted.clone(), config.api, 510).is_err());
    assert!(parse_prompt_count(counted.clone(), config.api, u32::MAX).is_err());
    assert!(parse_prompt_count(counted.clone(), ServiceApi::InputTokensV1, 8).is_err());
    for invalid in [
        serde_json::json!({"prompt_tokens":0,"max_model_len":512}),
        serde_json::json!({"prompt_tokens":3,"max_model_len":0}),
        serde_json::json!({"prompt_tokens":3,"max_model_len":"512"}),
        serde_json::json!({"prompt_tokens":3,"max_model_len":512,"extra":true}),
        serde_json::json!({"object":"response.input_tokens","input_tokens":3}),
    ] {
        assert!(parse_prompt_count(invalid, config.api, 8).is_err());
    }
    for mut invalid in [
        serde_json::Value::Null,
        serde_json::json!("bad"),
        serde_json::json!(3),
        serde_json::json!([]),
        serde_json::json!({"choices":"bad"}),
        serde_json::json!({"choices":[3]}),
        serde_json::json!({"choices":[{}]}),
        serde_json::json!({"choices":[{"delta":3,"message":3}]}),
    ] {
        assert!(normalize_output(&mut invalid, config.api, true).is_err());
    }
    for stream in [false, true] {
        let field = if stream { "delta" } else { "message" };
        let mut value = serde_json::json!({"id":"one", "model":"backend",
            "object":if stream {"chat.completion.chunk"} else {"chat.completion"},
            "choices":[{"index":0,"finish_reason":"stop"}],
            "usage":{"prompt_tokens":3,"completion_tokens":1,"total_tokens":4}});
        value["choices"][0][field] = serde_json::json!({"role":"assistant","content":"answer",
            "reasoning_content":"thought","reasoning":"thought"});
        let mut legacy = config.clone();
        legacy.api = ServiceApi::InputTokensV1;
        assert!(validate_output(&value, &legacy, stream, 8, &mut Output::default()).is_err());
        for alias in [
            serde_json::json!("different"),
            serde_json::json!(3),
            serde_json::Value::Null,
        ] {
            let mut invalid = value.clone();
            invalid["choices"][0][field]["reasoning"] = alias;
            assert!(normalize_output(&mut invalid, config.api, stream).is_err());
        }
        let mut missing = value.clone();
        missing["choices"][0][field]
            .as_object_mut()
            .unwrap()
            .remove("reasoning_content");
        assert!(normalize_output(&mut missing, config.api, stream).is_err());
        normalize_output(&mut value, config.api, stream).unwrap();
        assert!(value["choices"][0][field].get("reasoning").is_none());
        validate_output(&value, &config, stream, 8, &mut Output::default()).unwrap();
        let mut nullable = value.clone();
        nullable["choices"][0][field]["reasoning_content"] = serde_json::Value::Null;
        nullable["choices"][0][field]["reasoning"] = serde_json::Value::Null;
        normalize_output(&mut nullable, config.api, stream).unwrap();
        validate_output(&nullable, &config, stream, 8, &mut Output::default()).unwrap();
        value["choices"][0][field]["tool_calls"] = serde_json::json!([]);
        assert!(validate_output(&value, &config, stream, 8, &mut Output::default()).is_err());
    }
    let mut usage = serde_json::json!({"id":"one","model":"backend","object":"chat.completion.chunk",
        "choices":[],"usage":{"prompt_tokens":3,"completion_tokens":1,"total_tokens":4}});
    normalize_output(&mut usage, config.api, true).unwrap();
    let mut output = Output {
        id: Some("one".into()),
        finished: true,
        expected_prompt: Some(3),
        ..Output::default()
    };
    validate_output(&usage, &config, true, 8, &mut output).unwrap();
    assert!(output.usage);
}

#[cfg(unix)]
#[test]
fn process_group_confirmation_includes_descendants_and_never_signals_after_reaping() {
    for mode in ["normal", "descendants", "crash"] {
        let marker =
            std::env::temp_dir().join(format!("glyd-synthetic-child-{}", uuid::Uuid::new_v4()));
        let mut config = fixture_config(mode);
        config.environment.insert(
            "COMPUTE_FIXTURE_CHILD_MARKER".into(),
            marker.to_string_lossy().into_owned(),
        );
        let request = request(&config, false);
        let mut service = start(config);
        let pid = service.control.process.lock().unwrap().pid().unwrap();
        assert!(rustix::process::test_kill_process_group(pid).is_ok());
        if mode == "descendants" {
            let deadline = Instant::now() + Duration::from_secs(3);
            while !marker.exists() {
                assert!(Instant::now() < deadline);
                std::thread::sleep(Duration::from_millis(5));
            }
        }
        if mode == "crash" {
            assert!(
                service
                    .generate(&request, |_| panic!("crashed fixture emitted output"))
                    .is_err()
            );
        }
        service
            .confirm_process_group_exit()
            .unwrap_or_else(|error| panic!("{mode}: {error:?}"));
        assert_eq!(
            rustix::process::test_kill_process_group(pid),
            Err(rustix::io::Errno::SRCH)
        );
        assert!(!service.control.process.lock().unwrap().signal_allowed);
        // Repeated cleanup only observes completion; it cannot signal a reused PID.
        service
            .confirm_process_group_exit()
            .unwrap_or_else(|error| panic!("{mode}: {error:?}"));
        service.cancellation().cancel().unwrap();
        if mode == "descendants" {
            std::fs::remove_file(marker).unwrap();
        }
    }
}

#[cfg(windows)]
#[test]
fn windows_parent_exit_does_not_claim_group_confirmation() {
    let mut service = start(fixture_config("normal"));
    assert_eq!(
        service.confirm_process_group_exit(),
        Err(Error::Unavailable)
    );
}

#[test]
fn backend_count_binds_exact_request_and_usage_without_estimates() {
    for streaming in [false, true] {
        for mode in ["normal", "glyd-serve"] {
            let config = fixture_config(mode);
            let mut service = start(config.clone());
            let request = request(&config, streaming);
            assert_eq!(service.count_prompt(&request), Ok(3));
            service
                .generate(&request, |bytes| {
                    let value: serde_json::Value = serde_json::from_slice(bytes).unwrap();
                    let message = &value["choices"][0][if streaming { "delta" } else { "message" }];
                    assert_eq!(message["reasoning_content"], "synthetic reasoning");
                    assert!(message.get("reasoning").is_none());
                    Ok(())
                })
                .unwrap();
            assert_eq!(service.count_prompt(&request), Ok(3));
            let mut changed = request;
            changed.messages[0].content.push_str("changed");
            assert_eq!(
                service.generate(&changed, |_| panic!("uncounted prompt emitted")),
                Err(Error::Unapproved)
            );
            service.stop().unwrap();
        }
        let config = fixture_config("wrong-count");
        let mut service = start(config.clone());
        let request = self::request(&config, streaming);
        assert_eq!(service.count_prompt(&request), Ok(4));
        assert_eq!(
            service.generate(&request, |_| panic!("mismatched usage emitted")),
            Err(Error::Invalid)
        );
        service.stop().unwrap();
    }
    for mode in ["bad-count", "oversize-count"] {
        let config = fixture_config(mode);
        let mut service = start(config.clone());
        assert_eq!(
            service.count_prompt(&request(&config, false)),
            Err(Error::Invalid)
        );
        service.stop().unwrap();
    }
}
