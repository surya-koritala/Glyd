//! The service as the build ships it, with no GPU and no model (CI has none): `/health` and `/ready` answer, `/metrics` is Prometheus's text, and SIGTERM ends it cleanly: it says it is draining, takes no more
//! requests, waits for none (there are none in flight) and exits 0 within a bound. The engine's own `serve` is drained the same way (src/ops.rs); its run on a GPU is the `gpu` CI job's (tests/ops_gpu.sh).

#![cfg(unix)]

use std::io::{Read, Write};
use std::net::TcpStream;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

fn get(port: u16, path: &str) -> Option<(u16, String)> {
    let mut c = TcpStream::connect(("127.0.0.1", port)).ok()?;
    c.set_read_timeout(Some(Duration::from_secs(5))).ok()?;
    c.write_all(format!("GET {path} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n").as_bytes()).ok()?;
    let mut out = String::new();
    c.read_to_string(&mut out).ok()?;
    let (head, body) = out.split_once("\r\n\r\n")?;
    Some((head.split_whitespace().nth(1)?.parse().ok()?, body.to_string()))
}

#[test]
fn the_service_answers_its_probes_and_drains_on_sigterm() {
    let port = std::net::TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port();
    let home = std::env::temp_dir().join(format!("glyd-drain-{}", std::process::id()));
    std::fs::create_dir_all(&home).unwrap();
    let mut child = Command::new(env!("CARGO_BIN_EXE_glyd-engine"))
        .args(["service", "--port", &port.to_string(), "--drain-timeout", "5"])
        .env("HOME", &home)
        .env("XDG_STATE_HOME", home.join("state"))
        .env("HF_HOME", home.join("hf"))
        .env_remove("GLYD_API_KEY")
        .env_remove("VLLM_API_KEY")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let t = Instant::now();
    while get(port, "/ready").is_none_or(|(st, _)| st != 200) {
        assert!(t.elapsed() < Duration::from_secs(20), "the service did not come up");
        std::thread::sleep(Duration::from_millis(100));
    }
    assert_eq!(get(port, "/health").map(|r| r.0), Some(200));
    let (st, text) = get(port, "/metrics").unwrap();
    assert_eq!(st, 200);
    for want in
        ["# TYPE glyd_ready gauge", "glyd_ready 1", "glyd_draining 0", "glyd_requests_in_flight 0", "glyd_service_models_loaded 0", "glyd_build_info{version="]
    {
        assert!(text.contains(want), "{want} in {text}");
    }
    // SIGTERM: the process drains and exits 0 inside the bound
    let kill = Command::new("kill").args(["-TERM", &child.id().to_string()]).status().unwrap();
    assert!(kill.success());
    let t = Instant::now();
    let status = loop {
        if let Some(s) = child.try_wait().unwrap() {
            break s;
        }
        assert!(t.elapsed() < Duration::from_secs(10), "the service did not exit after SIGTERM");
        std::thread::sleep(Duration::from_millis(50));
    };
    assert_eq!(status.code(), Some(0), "exit status");
    let mut err = String::new();
    child.stderr.take().unwrap().read_to_string(&mut err).unwrap();
    assert!(err.contains("SIGTERM: not taking requests") && err.contains("drained: every request ended"), "{err}");
    let _ = std::fs::remove_dir_all(&home);
}
