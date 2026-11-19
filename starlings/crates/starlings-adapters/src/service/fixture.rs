//! Synthetic HTTP child for software tests only; no model or inference.
use super::*;
use std::{io::Write, net::TcpListener, time::SystemTime};

/// Synthetic children bind their own ephemeral port, then report the bound address.
/// This test-only launcher never runs a native runtime or approves a production config.
pub fn spawn(config: ServiceConfig, key: String) -> Result<Service> {
    spawn_observed(config, key, |_| {})
}
/// Static startup steps only; the observer must not block or reenter this launcher.
pub fn spawn_observed(
    mut config: ServiceConfig,
    key: String,
    mut phase: impl FnMut(&'static str),
) -> Result<Service> {
    if config.backend_version != "synthetic-v1"
        || !config.environment.contains_key("COMPUTE_FIXTURE_MODE")
    {
        return Err(Error::Invalid);
    }
    phase("fixture_directory");
    let root = std::env::temp_dir().join(format!("compute-fixture-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir(&root).map_err(|_| Error::Unavailable)?;
    let report = root.join("endpoint");
    config
        .environment
        .insert("COMPUTE_FIXTURE_ENDPOINT".into(), "127.0.0.1:0".into());
    config.environment.insert(
        "COMPUTE_FIXTURE_REPORT".into(),
        report.to_str().ok_or(Error::Invalid)?.into(),
    );
    config.environment.insert(
        "COMPUTE_FIXTURE_REQUEST_TIMEOUT_MS".into(),
        config.request_timeout_ms.to_string(),
    );
    let result = (|| {
        let mut service =
            Service::spawn_observed(config.clone(), &config.digest()?, key, &mut phase)?;
        phase("fixture_endpoint");
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            service.check_alive()?;
            if let Some(endpoint) = reported_endpoint(&report) {
                service.config.endpoint = endpoint;
                return Ok(service);
            }
            if Instant::now() >= deadline {
                return Err(Error::Unavailable);
            }
            std::thread::sleep(Duration::from_millis(10));
        }
    })();
    let _ = std::fs::remove_dir_all(root);
    result
}

pub(super) fn reported_endpoint(report: &Path) -> Option<SocketAddr> {
    let address = std::fs::read_to_string(report).ok()?;
    let endpoint = address.strip_suffix('\n')?.parse::<SocketAddr>().ok()?;
    (endpoint.ip().is_loopback() && endpoint.port() != 0).then_some(endpoint)
}

pub(super) fn glyd_info() -> serde_json::Value {
    serde_json::json!({"object":"glyd.info", "api":"glyd-serve/1",
        "build":{"version":"synthetic-v1","commit":null,"development":true},
        "model":{"id":"synthetic-backend","repo":null,"revision":null,"source":null,
            "files":[{"name":"not-retained","bytes":1,"sha256":null}]},
        "level":{"requested":null,"effective":"synthetic","lossless":true},
        "precision":{"weights":"f32","activations":"f32","sums":"f32","kv":null,"state":null},
        "limits":{"context_tokens":512,"concurrency":1},
        "fields":{"read":["messages","request_id"],"no_effect":[],"refused":[]},
        "state":"ready","error":null,"requests":{"running":0,"cancelling":0},
        "device":{"ordinal":0},
        "memory":{"gpu":0,"measured_at":2.0,"engine_bytes":100,
            "weights_bytes":60,"kv_bytes":20,"state_bytes":10,"scratch_bytes":10,
            "peak_bytes":120,"peak_at":1.5,"peak_since":1.0,
            "device_total_bytes":1000,"device_free_bytes":700,
            "not_counted":["synthetic; not qualified"]}})
}

fn glyd_status(id: &str, cancelled: bool, running: bool) -> serde_json::Value {
    serde_json::json!({"id":id,"state":if running {"running"} else {"ended"},
        "cancel":if cancelled {Some("asked")} else {None},"began":1.0,
        "ended":if running {None} else {Some(2.0)},
        "how":if running {None} else if cancelled {Some("cancelled")} else {Some("finished")}})
}

pub fn run() {
    if let Ok(marker) = std::env::var("COMPUTE_FIXTURE_DESCENDANT") {
        writeln!(
            std::fs::OpenOptions::new()
                .create_new(true)
                .write(true)
                .open(marker)
                .unwrap(),
            "{}",
            std::process::id()
        )
        .unwrap();
        std::thread::sleep(Duration::from_secs(30));
        return;
    }
    let Ok(endpoint) = std::env::var("COMPUTE_FIXTURE_ENDPOINT") else {
        return;
    };
    let mode = std::env::var("COMPUTE_FIXTURE_MODE").unwrap();
    let key = std::env::var("COMPUTE_FIXTURE_KEY").unwrap();
    let mut descendant = if mode == "descendants" {
        Some(
            std::process::Command::new(std::env::current_exe().unwrap())
                .args(std::env::args_os().skip(1))
                .env(
                    "COMPUTE_FIXTURE_DESCENDANT",
                    std::env::var("COMPUTE_FIXTURE_CHILD_MARKER").unwrap(),
                )
                .spawn()
                .unwrap(),
        )
    } else {
        None
    };
    let listener = TcpListener::bind(endpoint).unwrap();
    if let Ok(report) = std::env::var("COMPUTE_FIXTURE_REPORT") {
        std::fs::write(report, format!("{}\n", listener.local_addr().unwrap())).unwrap();
    }
    let mut loading = mode == "loading";
    let mut cancel_next = matches!(
        mode.as_str(),
        "glyd-cancel" | "glyd-unconfirmed" | "glyd-cancel-pending" | "glyd-cancel-pending-fail"
    );
    let glyd_request = Arc::new(Mutex::new(None::<serde_json::Value>));
    'connections: for incoming in listener.incoming() {
        if mode == "startup-crash" {
            return;
        }
        let Ok(mut socket) = incoming else { continue };
        if socket
            .set_read_timeout(Some(Duration::from_secs(3)))
            .is_err()
        {
            continue;
        }
        let Ok(read) = socket.try_clone() else {
            continue;
        };
        let mut reader = BufReader::new(read);
        let mut first = String::new();
        if !matches!(reader.read_line(&mut first), Ok(n) if n > 0) {
            continue;
        }
        let mut length = 0;
        let mut authorized = false;
        loop {
            let mut line = String::new();
            if !matches!(reader.read_line(&mut line), Ok(n) if n > 0) {
                continue 'connections;
            }
            if line == "\r\n" {
                break;
            }
            let Some((name, value)) = line.split_once(':') else {
                continue 'connections;
            };
            if name.eq_ignore_ascii_case("content-length") {
                let Ok(size) = value.trim().parse::<usize>() else {
                    continue 'connections;
                };
                length = size;
            }
            if name.eq_ignore_ascii_case("authorization") {
                authorized = value.trim() == format!("Bearer {key}");
            }
        }
        if mode == "slow-probe" {
            std::thread::sleep(Duration::from_millis(650));
        }
        if loading {
            loading = false;
            let _ = socket.write_all(
                b"HTTP/1.1 503 Unavailable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
            );
            continue;
        }
        if !authorized && mode == "unprotected" {
            let _ = socket
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\nConnection: close\r\n\r\n");
            continue;
        }
        if !authorized {
            let _ = socket.write_all(
                b"HTTP/1.1 401 Unauthorized\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
            );
            continue;
        }
        let (kind, body) = if first.starts_with("GET /health ") {
            ("application/json", b"{}".to_vec())
        } else if first.starts_with("GET /v1/glyd/info ") {
            let mut info = glyd_info();
            info["memory"]["measured_at"] = SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap()
                .as_secs_f64()
                .into();
            if glyd_request.lock().unwrap().is_some() {
                match mode.as_str() {
                    "glyd-memory-reset" => info["memory"]["peak_since"] = 0.0.into(),
                    "glyd-memory-invalid" => info["memory"]["kv_bytes"] = 1000.into(),
                    "glyd-still-running" => info["requests"]["running"] = 1.into(),
                    _ => (),
                }
            }
            if mode == "glyd-identity-mismatch" {
                info["level"]["effective"] = "different".into();
            }
            ("application/json", serde_json::to_vec(&info).unwrap())
        } else if first.starts_with("GET /v1/glyd/requests/")
            || first.starts_with("POST /v1/glyd/requests/")
        {
            let path = first.split_whitespace().nth(1).unwrap();
            let path = path.strip_prefix("/v1/glyd/requests/").unwrap();
            let id = path.strip_suffix("/cancel").unwrap_or(path);
            let mut status = glyd_request.lock().unwrap();
            let value = status.as_mut().unwrap();
            assert_eq!(value["id"], id);
            if path.ends_with("/cancel") && mode != "glyd-unconfirmed" {
                *value = glyd_status(id, true, false);
            }
            if path.ends_with("/cancel") && mode.starts_with("glyd-cancel-pending") {
                let body = serde_json::to_vec(value).unwrap();
                drop(status);
                let release =
                    std::path::PathBuf::from(std::env::var("COMPUTE_FIXTURE_RELEASE").unwrap());
                let fail = mode == "glyd-cancel-pending-fail";
                std::fs::write(
                    std::env::var("COMPUTE_FIXTURE_CANCEL_STARTED").unwrap(),
                    b"pending",
                )
                .unwrap();
                std::thread::spawn(move || {
                    let deadline = Instant::now() + Duration::from_secs(2);
                    while !release.exists() && Instant::now() < deadline {
                        std::thread::sleep(Duration::from_millis(5));
                    }
                    let code = if fail { "503 Unavailable" } else { "200 OK" };
                    let header = format!(
                        "HTTP/1.1 {code}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                        body.len()
                    );
                    let _ = socket
                        .write_all(header.as_bytes())
                        .and_then(|_| socket.write_all(&body));
                });
                continue;
            }
            ("application/json", serde_json::to_vec(value).unwrap())
        } else if first.starts_with("GET /slots ") {
            (
                "application/json",
                serde_json::to_vec(&serde_json::json!([{"is_processing":mode == "busy-idle"}]))
                    .unwrap(),
            )
        } else if first.starts_with("GET /v1/models ") {
            (
                "application/json",
                br#"{"data":[{"id":"synthetic-backend"}]}"#.to_vec(),
            )
        } else if first.starts_with("GET /props ") {
            (
                "application/json",
                serde_json::to_vec(&serde_json::json!({
                    "build_info": if mode == "properties-mismatch" { "other-build" } else { "synthetic-v1" },
                    "model_alias": "synthetic-backend",
                    "total_slots": 1,
                    "default_generation_settings": {"n_ctx": 512},
                    "chat_template_caps": {
                        "supports_system_role": true,
                        "supports_string_content": true,
                        "supports_typed_content": false
                    },
                    "model_path": "/private/path/not-retained",
                    "chat_template": "not-retained"
                })).unwrap(),
            )
        } else if first.starts_with("POST /v1/chat/completions/input_tokens ")
            || first.starts_with("POST /v1/glyd/tokenize ")
        {
            let glyd = first.starts_with("POST /v1/glyd/tokenize ");
            if glyd != mode.starts_with("glyd-") {
                let _ = socket.write_all(
                    b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
                );
                continue;
            }
            assert!(length <= starlings_protocol::MAX_JSON_BYTES);
            let mut body = vec![0; length];
            if reader.read_exact(&mut body).is_err() {
                continue;
            }
            let request = parse_json(&body).unwrap();
            assert_eq!(request["model"], "synthetic-backend");
            let tokens = match mode.as_str() {
                "bad-count" => 0,
                "wrong-count" => 4,
                "oversize-count" => 2_000_000,
                _ => 3,
            };
            (
                "application/json",
                serde_json::to_vec(&if glyd {
                    serde_json::json!({"prompt_tokens":tokens,"max_model_len":512})
                } else {
                    serde_json::json!({"object":"response.input_tokens", "input_tokens":tokens})
                })
                .unwrap(),
            )
        } else {
            assert!(first.starts_with("POST /v1/chat/completions "));
            assert!(length <= starlings_protocol::MAX_JSON_BYTES);
            let mut body = vec![0; length];
            if reader.read_exact(&mut body).is_err() {
                continue;
            }
            let request = parse_json(&body).unwrap();
            assert_eq!(request["model"], "synthetic-backend");
            let id = if mode.starts_with("glyd-") {
                format!("chatcmpl-{}", request["request_id"].as_str().unwrap())
            } else {
                "synthetic-1".into()
            };
            if mode.starts_with("glyd-") {
                *glyd_request.lock().unwrap() =
                    Some(glyd_status(&id, mode == "glyd-cancelled-stop", false));
            }
            if cancel_next {
                cancel_next = false;
                *glyd_request.lock().unwrap() = Some(glyd_status(&id, false, true));
                let state = glyd_request.clone();
                std::thread::spawn(move || {
                    let value = serde_json::json!({"id":id,"model":"synthetic-backend","object":"chat.completion.chunk",
                        "choices":[{"index":0,"finish_reason":null,"delta":{"content":"partial"}}]});
                    let _ = write!(socket,"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\ndata: {value}\n\n").and_then(|_| socket.flush());
                    let deadline = Instant::now() + Duration::from_secs(5);
                    while Instant::now() < deadline
                        && state.lock().unwrap().as_ref().unwrap()["state"] != "ended"
                    {
                        std::thread::sleep(Duration::from_millis(5));
                    }
                });
                continue;
            }
            if let Ok(marker) = std::env::var("COMPUTE_FIXTURE_STARTED") {
                writeln!(
                    std::fs::OpenOptions::new()
                        .create(true)
                        .append(true)
                        .open(marker)
                        .unwrap(),
                    "started"
                )
                .unwrap();
            }
            if mode == "crash" {
                return;
            }
            if let Ok(release) = std::env::var("COMPUTE_FIXTURE_RELEASE") {
                let timeout = std::env::var("COMPUTE_FIXTURE_REQUEST_TIMEOUT_MS")
                    .unwrap()
                    .parse()
                    .unwrap();
                let deadline = Instant::now() + Duration::from_millis(timeout);
                while !std::path::Path::new(&release).exists() {
                    if Instant::now() >= deadline {
                        return;
                    }
                    std::thread::sleep(Duration::from_millis(5));
                }
            }
            if mode == "stall" {
                std::thread::sleep(Duration::from_secs(30));
                return;
            }
            if mode == "partial" {
                let value = serde_json::json!({"id":"synthetic-1", "model":"synthetic-backend", "object":"chat.completion.chunk", "choices":[{"index":0,"finish_reason":null,"delta":{"role":"assistant","content":"partial"}}]});
                if write!(socket, "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\ndata: {value}\n\n").and_then(|_| socket.flush()).is_err() { continue }
                std::thread::sleep(Duration::from_secs(30));
                return;
            }
            if mode == "backpressure" {
                assert_eq!(request["stream"], true);
                // Bounded nonterminal output below all event/stream/channel byte caps.
                let value = serde_json::json!({"id":"synthetic-1", "model":"synthetic-backend", "object":"chat.completion.chunk", "choices":[{"index":0,"finish_reason":null,"delta":{"role":"assistant","content":format!("partial{}", "x".repeat(16 * 1024))}}]});
                if write!(socket, "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n").is_err() { return }
                for _ in 0..512 {
                    if write!(socket, "data: {value}\n\n")
                        .and_then(|_| socket.flush())
                        .is_err()
                    {
                        return;
                    }
                }
                std::thread::sleep(Duration::from_secs(30));
                return;
            }
            let stream = request["stream"] == true;
            let mut value = serde_json::json!({
                "id":id, "model":"synthetic-backend", "object":if stream {"chat.completion.chunk"} else {"chat.completion"},
                "choices":[{"index":0, "finish_reason":"stop"}],
                "usage":{"prompt_tokens":3,"completion_tokens":1,"total_tokens":4}
            });
            value["choices"][0][if stream { "delta" } else { "message" }] = serde_json::json!({"role":"assistant","content":"synthetic answer", "reasoning_content":"synthetic reasoning"});
            if mode.starts_with("glyd-") {
                value["choices"][0][if stream { "delta" } else { "message" }]["reasoning"] =
                    "synthetic reasoning".into();
            }
            if mode == "large" {
                value["choices"][0][if stream { "delta" } else { "message" }]["content"] =
                    serde_json::json!("x😀".repeat(if stream { 6 * 1024 } else { 50 * 1024 }));
            }
            if mode == "tools" {
                value["choices"][0][if stream { "delta" } else { "message" }]["tool_calls"] =
                    serde_json::json!([]);
            }
            if stream {
                assert_eq!(request["stream_options"]["include_usage"], true);
                let body = format!(
                    ": fixture\r\ndata: {}\r\n\r\ndata: [DONE]\n\n",
                    serde_json::to_string(&value).unwrap()
                );
                ("text/event-stream", body.into_bytes())
            } else {
                ("application/json", serde_json::to_vec(&value).unwrap())
            }
        };
        let header = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: {kind}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            body.len()
        );
        // An abandoned probe is a connection failure, not the intentional crash mode.
        let _ = socket
            .write_all(header.as_bytes())
            .and_then(|_| socket.write_all(&body));
    }
    if let Some(child) = &mut descendant {
        let _ = child.kill();
        let _ = child.wait();
    }
}
