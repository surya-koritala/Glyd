use super::*;
use std::{
    net::TcpListener,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    thread,
};

#[test]
fn pinned_exo_stream_object_is_validated_in_process() {
    let endpoint = Endpoint {
        address: "127.0.0.1:1".parse().unwrap(),
        model: "synthetic-shared".into(),
    };
    let chat: ChatRequest = serde_json::from_value(serde_json::json!({
        "model":"synthetic-shared", "messages":[{"role":"user","content":"fixture"}],
        "max_tokens":4, "stream":true
    }))
    .unwrap();
    let workload = Workload {
        _version: Version,
        request_sha256: byte_digest(b"fixture"),
        expected_prompt_tokens: 3,
        concurrency: 1,
    };
    let began = Instant::now();
    let attempt = Attempt {
        stopped: false,
        stream_object: stream_object(System::Exo, "bc412e2230f395e652c23426a5eb88dfa515769b"),
        endpoint: &endpoint,
        key: None,
        body: b"fixture",
        chat: &chat,
        workload: &workload,
        cancel: false,
        deadline: began + Duration::from_secs(1),
        timeout_ms: 1,
    };
    let mut record = RequestRecord {
        request_sha256: byte_digest(b"fixture"),
        attempted: true,
        http_status: Some(200),
        code: "SUCCESS",
        prompt_tokens: None,
        output_tokens: None,
        first_content_ms: None,
        duration_ms: 0.0,
    };
    let value = serde_json::json!({
        "id":"synthetic-response", "object":"chat.completion", "model":"synthetic-shared",
        "choices":[{"index":0,"delta":{"role":"assistant","content":"fixture-output"},"finish_reason":"stop"}],
        "usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}
    });
    let mut output = Output::default();
    assert_eq!(
        output.event(&value, &attempt, true, began, &mut record),
        Ok(())
    );
    assert!(output.finished);
    assert_eq!(record.output_tokens, Some(2));
    assert!(record.first_content_ms.is_some());

    // No sockets or service: test the same parser and pin policy used by collection.
    for (system, commit) in [
        (System::Exo, "bc412e2230f395e652c23426a5eb88dfa515769b"),
        (System::Exo, "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"),
        (System::Glyd, "bc412e2230f395e652c23426a5eb88dfa515769b"),
    ] {
        let attempt = Attempt {
            stream_object: stream_object(system, commit),
            ..attempt
        };
        for tag in ["chat.completion", "chat.completion.chunk", "other"] {
            let mut value = value.clone();
            value["object"] = tag.into();
            let mut output = Output::default();
            record.output_tokens = None;
            assert_eq!(
                output.event(&value, &attempt, true, began, &mut record),
                if tag == attempt.stream_object {
                    Ok(())
                } else {
                    Err("INVALID_RESPONSE")
                }
            );
        }
    }
    for (field, replacement, expected) in [
        ("model", serde_json::json!("wrong"), "INVALID_RESPONSE"),
        ("id", serde_json::json!(""), "INVALID_RESPONSE"),
        ("choices", serde_json::json!([]), "INVALID_USAGE"),
        (
            "usage",
            serde_json::json!({"prompt_tokens":3,"completion_tokens":2,"total_tokens":99}),
            "INVALID_USAGE",
        ),
        (
            "choices",
            serde_json::json!([{"index":0,"delta":{"tool_calls":[]},"finish_reason":"stop"}]),
            "INVALID_RESPONSE",
        ),
    ] {
        let mut value = value.clone();
        value[field] = replacement;
        let mut output = Output::default();
        record.output_tokens = None;
        assert_eq!(
            output.event(&value, &attempt, true, began, &mut record),
            Err(expected)
        );
    }
    let mut output = Output::default();
    record.output_tokens = None;
    let mut no_usage = value.clone();
    no_usage.as_object_mut().unwrap().remove("usage");
    output
        .event(&no_usage, &attempt, true, began, &mut record)
        .unwrap();
    assert!(output.finished);
    assert_eq!(
        record.output_tokens, None,
        "a finish cannot manufacture usage"
    );
    let raw = serde_json::to_string(&record).unwrap();
    assert!(!raw.contains("fixture-output") && !raw.contains("synthetic-response"));
}

struct Mock {
    address: SocketAddr,
    running: Arc<AtomicBool>,
    calls: Arc<AtomicUsize>,
    join: Option<thread::JoinHandle<()>>,
}
impl Mock {
    fn new(mode: &'static str) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        listener.set_nonblocking(true).unwrap();
        let running = Arc::new(AtomicBool::new(true));
        let calls = Arc::new(AtomicUsize::new(0));
        let (live, seen) = (running.clone(), calls.clone());
        let join = thread::spawn(move || {
            let mut handlers = Vec::new();
            while live.load(Ordering::SeqCst) {
                match listener.accept() {
                    Ok((mut stream, _)) => {
                        seen.fetch_add(1, Ordering::SeqCst);
                        stream.set_nonblocking(false).unwrap();
                        handlers.push(thread::spawn(move || {
                            stream.set_read_timeout(Some(Duration::from_secs(2))).unwrap();
                            stream.set_write_timeout(Some(Duration::from_secs(2))).unwrap();
                            stream.set_nodelay(true).unwrap();
                            let mut reader = BufReader::new(stream.try_clone().unwrap());
                            let mut length = 0;
                            loop {
                                let mut line = String::new();
                                if reader.read_line(&mut line).unwrap() == 0 { return; }
                                if line == "\r\n" { break; }
                                if let Some(n) = line.to_ascii_lowercase().strip_prefix("content-length:") { length = n.trim().parse().unwrap(); }
                            }
                            let mut body = vec![0; length];
                            reader.read_exact(&mut body).unwrap();
                            let body: Value = serde_json::from_slice(&body).unwrap();
                            let model = body["model"].as_str().unwrap();
                            let stream_mode = body["stream"] == true;
                            if stream_mode { assert_eq!(body["stream_options"]["include_usage"], true); }
                            if mode == "slow" { thread::sleep(Duration::from_millis(200)); }
                            let usage = serde_json::json!({"prompt_tokens":3,"completion_tokens":2,"total_tokens":5});
                            let mut response = if stream_mode {
                                let event = |choices: Value, usage: Value| format!("data: {}\r\n\r\n", serde_json::json!({"id":"response-private-id","object":"chat.completion.chunk","model":model,"choices":choices,"usage":usage}));
                                let role = event(serde_json::json!([{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]), Value::Null);
                                let content = event(serde_json::json!([{"index":0,"delta":{"content":"private generated text"},"finish_reason":null}]), Value::Null);
                                let finish = event(serde_json::json!([{"index":0,"delta":{},"finish_reason":"stop"}]), Value::Null);
                                let usage = if mode == "missing" { String::new() } else { event(serde_json::json!([]),usage) };
                                format!(": comment\r\n\r\n{role}{content}{finish}{usage}data: [DONE]\r\n\r\n")
                            } else {
                                let mut v = serde_json::json!({"id":"response-private-id","object":"chat.completion","model":model,"choices":[{"index":0,"message":{"role":"assistant","content":"private generated text"},"finish_reason":"stop"}],"usage":usage});
                                if mode == "missing" { v.as_object_mut().unwrap().remove("usage"); }
                                v.to_string()
                            };
                            if mode == "bad_usage" { response = response.replace("\"total_tokens\":5", "\"total_tokens\":99"); }
                            if mode == "bad_json" { response = "{private invalid response".into(); }
                            if mode == "truncated" { response = response.replace("data: [DONE]\r\n\r\n", ""); }
                            if mode == "long_line" { response = format!("data: {}\n\n", "x".repeat(65537)); }
                            let status = if mode == "redirect" { "302 Found\r\nLocation: http://192.0.2.1/never" } else if mode == "failure" { "503 Unavailable" } else { "200 OK" };
                            let kind = if stream_mode { "text/event-stream" } else { "application/json" };
                            let headers = format!("HTTP/1.1 {status}\r\nContent-Type: {kind}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", response.len());
                            let _ = stream.write_all(headers.as_bytes());
                            for bytes in response.as_bytes().chunks(7) {
                                if stream.write_all(bytes).is_err() { break; }
                            }
                        }));
                    }
                    Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(1))
                    }
                    Err(e) => panic!("{e}"),
                }
            }
            for h in handlers {
                h.join().unwrap();
            }
        });
        Self {
            address,
            running,
            calls,
            join: Some(join),
        }
    }
}
impl Drop for Mock {
    fn drop(&mut self) {
        self.running.store(false, Ordering::SeqCst);
        let result = self.join.take().unwrap().join();
        if !thread::panicking() {
            result.unwrap();
        }
    }
}
fn fixture(address: SocketAddr, stream: bool) -> (tempfile::TempDir, Value) {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();
    let request = serde_json::json!({"model":"logical-model","messages":[{"role":"user","content":"private prompt"}],"max_tokens":4,"temperature":0,"top_p":1,"seed":1,"stream":stream});
    let request = serde_json::to_vec(&request).unwrap();
    std::fs::write(root.join("request.json"), &request).unwrap();
    let mut matrix: Value =
        serde_json::from_str(include_str!("../../../../examples/scoreboard-plan.json")).unwrap();
    matrix["fixture"] = true.into();
    let mut artifacts = vec![];
    let mut artifact = |name: &str, bytes: &[u8]| {
        std::fs::write(root.join(name), bytes).unwrap();
        let hash = byte_digest(bytes);
        artifacts.push(serde_json::json!({"path":name,"sha256":hash}));
        hash
    };
    let identity = artifact(
        "identity.txt",
        b"synthetic identities - no hardware/model claim",
    );
    matrix["programs"] = serde_json::json!({"exo":{"repository":"https://github.com/exo-explore/exo","source_commit":"a".repeat(40),"build":identity,"recipe":identity},"glyd":{"repository":"https://github.com/Sigbound/starlings","source_commit":"b".repeat(40),"build":identity,"recipe":identity}});
    let metric = if stream {
        "ttft_milliseconds"
    } else {
        "concurrent_tokens_per_second"
    };
    let boundary = if stream {
        "client_first_nonempty_content_or_reasoning_delta"
    } else {
        "client_batch_usage_tokens_over_wall_seconds"
    };
    let definition = artifact(
        "definition.json",
        &serde_json::to_vec(&serde_json::json!({"version":1,"metric":metric,"boundary":boundary}))
            .unwrap(),
    );
    let workload = artifact("workload.json", &serde_json::to_vec(&serde_json::json!({"version":1,"request_sha256":byte_digest(&request),"expected_prompt_tokens":3,"concurrency":if stream {1} else {2}})).unwrap());
    let cell = matrix["cells"]
        .as_array_mut()
        .unwrap()
        .iter_mut()
        .find(|c| c["metric"] == metric)
        .unwrap();
    cell["matched"] = serde_json::json!({"model":identity,"profile":identity,"hardware":identity,"workload":workload,"definition":definition,"parameters":identity});
    let id = cell["id"].clone();
    matrix["artifacts"] = artifacts.into();
    std::fs::write(
        root.join("matrix.json"),
        serde_json::to_vec(&matrix).unwrap(),
    )
    .unwrap();
    let plan = serde_json::json!({"version":1,"fixture":true,"repeats":5,"timeout_ms":1000,"budget_ms":10000,"exo":{"address":address,"model":"exo-alias"},"glyd":{"address":address,"model":"glyd-alias"},"cells":[{"id":id,"request_path":"request.json","cancel_after_content":false}]});
    (temp, plan)
}
fn collect(root: &Path, plan: &Value) -> Result<scoreboard::Report> {
    std::fs::write(root.join("plan.json"), serde_json::to_vec(plan).unwrap()).unwrap();
    collect_file(
        &root.join("matrix.json"),
        &root.join("plan.json"),
        &root.join("collected.json"),
        FixtureMode::Simulation,
    )
}
#[test]
fn paired_json_and_fragmented_sse_produce_replayable_redacted_records() {
    for stream in [false, true] {
        let server = Mock::new("success");
        let (temp, plan) = fixture(server.address, stream);
        let report = collect(temp.path(), &plan).unwrap();
        assert!(
            report
                .cells
                .iter()
                .any(|c| c.status == scoreboard::Status::Fixture)
        );
        assert!(
            !report
                .cells
                .iter()
                .any(|c| c.status == scoreboard::Status::Won)
        );
        let raw = std::fs::read_to_string(temp.path().join("collected.raw.jsonl")).unwrap();
        let trials: Vec<Value> = raw
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();
        assert_eq!(trials.len(), 10);
        assert_eq!(
            server.calls.load(Ordering::SeqCst),
            if stream { 10 } else { 20 }
        );
        for (i, t) in trials.iter().enumerate() {
            let repeat = i / 2 + 1;
            let first = if repeat % 2 == 1 { "exo" } else { "glyd" };
            assert_eq!(t["repeat"], repeat);
            assert_eq!(t["order"], i % 2);
            assert_eq!(t["system"] == first, i % 2 == 0);
            for r in t["requests"].as_array().unwrap() {
                assert_eq!(r["code"], "SUCCESS");
                assert_eq!(r["prompt_tokens"], 3);
                assert_eq!(r["output_tokens"], 2); // two usage tokens, not chunk count
                assert_eq!(r["first_content_ms"].is_number(), stream);
                assert!(
                    r["duration_ms"].as_f64().unwrap()
                        >= r["first_content_ms"].as_f64().unwrap_or(0.0)
                );
            }
        }
        for secret in [
            "private prompt",
            "private generated text",
            "response-private-id",
            "exo-alias",
            "glyd-alias",
            "logical-model",
        ] {
            assert!(!raw.contains(secret));
        }
        let replay =
            scoreboard::report_file(&temp.path().join("collected.json"), FixtureMode::Simulation)
                .unwrap();
        assert_eq!(replay.matrix_digest, report.matrix_digest);
        assert_eq!(
            collect(temp.path(), &plan).err(),
            Some("COLLECT_OUTPUT_EXISTS_OR_REFUSED")
        );
    }
}
#[test]
fn missing_malformed_http_and_cancelled_trials_are_retained_without_wins() {
    for (stream, mode, code) in [
        (false, "missing", "INCOMPLETE"),
        (true, "missing", "INCOMPLETE"),
        (true, "bad_usage", "INVALID_USAGE"),
        (false, "bad_json", "INVALID_JSON"),
        (true, "truncated", "INCOMPLETE"),
        (true, "long_line", "BYTE_LIMIT"),
        (true, "redirect", "HTTP_STATUS"),
        (false, "failure", "HTTP_STATUS"),
        (true, "success", "CANCELLED"),
    ] {
        let server = Mock::new(mode);
        let (temp, mut plan) = fixture(server.address, stream);
        plan["cells"][0]["cancel_after_content"] = (code == "CANCELLED").into();
        let report = collect(temp.path(), &plan).unwrap();
        let row = report
            .cells
            .iter()
            .find(|c| !c.failures.is_empty())
            .unwrap();
        assert_eq!(row.failures.len(), 10);
        assert_eq!(row.failures[0].code, code);
        assert!(
            row.failures[1..]
                .iter()
                .all(|f| f.code == "NOT_ATTEMPTED_AFTER_FAILURE")
        );
        assert_eq!(
            server.calls.load(Ordering::SeqCst),
            if stream { 1 } else { 2 }
        );
        assert_eq!(row.status, scoreboard::Status::NotMeasured);
        let raw = std::fs::read_to_string(temp.path().join("collected.raw.jsonl")).unwrap();
        assert!(!raw.contains("private"));
    }
}
#[test]
fn admission_and_deadline_controls_send_no_unapproved_request() {
    let server = Mock::new("success");
    let (temp, plan) = fixture(server.address, true);
    for change in [
        serde_json::json!({"repeats":4}),
        serde_json::json!({"exo":{"address":"192.0.2.1:80","model":"x"}}),
        serde_json::json!({"unexpected":true}),
        serde_json::json!({"fixture":false}),
        serde_json::json!({"timeout_ms":30001}),
    ] {
        let mut invalid = plan.clone();
        for (k, v) in change.as_object().unwrap() {
            invalid[k] = v.clone();
        }
        assert!(collect(temp.path(), &invalid).is_err());
        assert!(!temp.path().join("collected.raw.jsonl").exists());
    }
    let mut duplicate = plan.clone();
    duplicate["cells"]
        .as_array_mut()
        .unwrap()
        .push(plan["cells"][0].clone());
    assert_eq!(
        collect(temp.path(), &duplicate).err(),
        Some("COLLECT_DUPLICATE_CELL")
    );
    let mut escape = plan.clone();
    escape["cells"][0]["request_path"] = "../outside.json".into();
    assert_eq!(
        collect(temp.path(), &escape).err(),
        Some("COLLECT_RELATIVE_PATH_REQUIRED")
    );
    std::fs::write(
        temp.path().join("plan.json"),
        b"{\"version\":1,\"version\":1}",
    )
    .unwrap();
    assert_eq!(
        collect_file(
            &temp.path().join("matrix.json"),
            &temp.path().join("plan.json"),
            &temp.path().join("collected.json"),
            FixtureMode::Simulation
        )
        .err(),
        Some("COLLECT_INVALID_JSON")
    );
    let mut matrix: Value =
        serde_json::from_slice(&std::fs::read(temp.path().join("matrix.json")).unwrap()).unwrap();
    let cell = matrix["cells"]
        .as_array_mut()
        .unwrap()
        .iter_mut()
        .find(|c| c["matched"].is_object())
        .unwrap();
    cell["metric"] = "decode_tokens_per_second".into();
    // Another planned TTFT row still exists; add the missing decode row to keep all 18 metrics present.
    matrix["cells"].as_array_mut().unwrap().push(serde_json::json!({"id":"remaining_ttft","metric":"ttft_milliseconds","matched":null,"unavailable":"not measured","observations":[]}));
    std::fs::write(
        temp.path().join("unsupported.json"),
        serde_json::to_vec(&matrix).unwrap(),
    )
    .unwrap();
    std::fs::write(
        temp.path().join("plan.json"),
        serde_json::to_vec(&plan).unwrap(),
    )
    .unwrap();
    assert_eq!(
        collect_file(
            &temp.path().join("unsupported.json"),
            &temp.path().join("plan.json"),
            &temp.path().join("collected.json"),
            FixtureMode::Simulation
        )
        .err(),
        Some("COLLECT_METRIC_UNSUPPORTED")
    );
    std::fs::write(temp.path().join("request.json"), b"changed prompt").unwrap();
    assert!(collect(temp.path(), &plan).is_err());
    assert_eq!(server.calls.load(Ordering::SeqCst), 0);
    let endpoint = Endpoint {
        address: server.address,
        model: "x".into(),
    };
    let chat: ChatRequest = serde_json::from_value(
        serde_json::json!({"model":"x","messages":[{"role":"user","content":"x"}],"max_tokens":4}),
    )
    .unwrap();
    let workload = Workload {
        _version: Version,
        request_sha256: byte_digest(b"x"),
        expected_prompt_tokens: 3,
        concurrency: 1,
    };
    let attempt = Attempt {
        stopped: false,
        stream_object: "chat.completion.chunk",
        endpoint: &endpoint,
        key: None,
        body: b"x",
        chat: &chat,
        workload: &workload,
        cancel: false,
        deadline: Instant::now() - Duration::from_secs(1),
        timeout_ms: 1,
    };
    assert_eq!(attempt.run().code, "BUDGET_EXHAUSTED");
    assert_eq!(server.calls.load(Ordering::SeqCst), 0);
}

#[test]
fn timed_out_batch_stops_later_submissions_and_retains_deadline_failure() {
    let server = Mock::new("slow");
    let (temp, mut plan) = fixture(server.address, true);
    plan["timeout_ms"] = 100.into();
    let report = collect(temp.path(), &plan).unwrap();
    let row = report
        .cells
        .iter()
        .find(|c| !c.failures.is_empty())
        .unwrap();
    assert_eq!(row.failures[0].code, "TIMEOUT");
    assert!(
        row.failures[1..]
            .iter()
            .all(|f| f.code == "NOT_ATTEMPTED_AFTER_FAILURE")
    );
    assert!(server.calls.load(Ordering::SeqCst) <= 1);
}
