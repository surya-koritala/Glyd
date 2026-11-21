//! A scan of a default install for listeners that are not on this computer's own address, and the refusals that keep it so: the service as the build ships it (no GPU, no model needed) with no `--host` listens on loopback
//! only, a non-loopback `--host` does not start without a key (or `--insecure-no-auth`), and with a key every /v1 request needs it. The engine's `serve` is scanned the same way by the `gpu` job (tests/ops_check.py, scripts/listener_scan.py).
//! Linux (the scan reads /proc); elsewhere the refusals are tested alone.

use std::io::{Read, Write};
use std::net::TcpStream;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

fn engine(home: &std::path::Path, args: &[&str]) -> Command {
    let mut c = Command::new(env!("CARGO_BIN_EXE_glyd-engine"));
    c.args(args)
        .env("HOME", home)
        .env("XDG_STATE_HOME", home.join("state"))
        .env("HF_HOME", home.join("hf"))
        .env_remove("GLYD_API_KEY")
        .env_remove("VLLM_API_KEY")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped());
    c
}

fn get(port: u16, path: &str, key: Option<&str>) -> Option<u16> {
    let mut c = TcpStream::connect(("127.0.0.1", port)).ok()?;
    c.set_read_timeout(Some(Duration::from_secs(5))).ok()?;
    let auth = key.map_or(String::new(), |k| format!("Authorization: Bearer {k}\r\n"));
    c.write_all(format!("GET {path} HTTP/1.1\r\nHost: localhost\r\n{auth}Connection: close\r\n\r\n").as_bytes()).ok()?;
    let mut out = String::new();
    c.read_to_string(&mut out).ok()?;
    out.split_whitespace().nth(1)?.parse().ok()
}

fn wait_up(port: u16, child: &mut Child) {
    let t = Instant::now();
    while get(port, "/health", None) != Some(200) {
        assert!(child.try_wait().unwrap().is_none(), "the service ended");
        assert!(t.elapsed() < Duration::from_secs(20), "the service did not come up");
        std::thread::sleep(Duration::from_millis(100));
    }
}

fn free_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port()
}

/// (pid, ip, port) of every TCP listener the process and its children hold (Linux: /proc).
#[cfg(target_os = "linux")]
fn listeners(pid: u32) -> Vec<(u32, std::net::IpAddr, u16)> {
    let table = |name: &str, v6: bool| -> Vec<(u64, std::net::IpAddr, u16)> {
        std::fs::read_to_string(format!("/proc/net/{name}"))
            .unwrap_or_default()
            .lines()
            .skip(1)
            .filter_map(|l| {
                let f: Vec<&str> = l.split_whitespace().collect();
                if f.get(3) != Some(&"0A") {
                    return None;
                }
                let (a, p) = f[1].split_once(':')?;
                let ip: std::net::IpAddr = if v6 {
                    let w: Vec<u32> = (0..4).map(|i| u32::from_str_radix(&a[i * 8..i * 8 + 8], 16).unwrap().swap_bytes()).collect();
                    std::net::Ipv6Addr::from(((w[0] as u128) << 96) | ((w[1] as u128) << 64) | ((w[2] as u128) << 32) | w[3] as u128).into()
                } else {
                    std::net::Ipv4Addr::from(u32::from_str_radix(a, 16).unwrap().swap_bytes()).into()
                };
                Some((f[9].parse().ok()?, ip, u16::from_str_radix(p, 16).ok()?))
            })
            .collect()
    };
    let all: Vec<_> = table("tcp", false).into_iter().chain(table("tcp6", true)).collect();
    let mut pids = vec![pid];
    let mut i = 0;
    while i < pids.len() {
        if let Ok(c) = std::fs::read_to_string(format!("/proc/{}/task/{}/children", pids[i], pids[i])) {
            pids.extend(c.split_whitespace().filter_map(|x| x.parse::<u32>().ok()));
        }
        i += 1;
    }
    let mut out = Vec::new();
    for p in pids {
        for fd in std::fs::read_dir(format!("/proc/{p}/fd")).into_iter().flatten().flatten() {
            if let Some(ino) = std::fs::read_link(fd.path()).ok().and_then(|l| l.to_str().and_then(|s| s.strip_prefix("socket:[")?.strip_suffix(']')?.parse::<u64>().ok())) {
                out.extend(all.iter().filter(|(i, _, _)| *i == ino).map(|(_, ip, port)| (p, *ip, *port)));
            }
        }
    }
    out
}

#[test]
fn a_default_install_has_no_listener_off_this_computer() {
    let home = std::env::temp_dir().join(format!("glyd-listeners-{}", std::process::id()));
    std::fs::create_dir_all(&home).unwrap();
    let port = free_port();
    let mut child = engine(&home, &["service", "--port", &port.to_string()]).spawn().unwrap();
    wait_up(port, &mut child);
    #[cfg(target_os = "linux")]
    {
        let found = listeners(child.id());
        assert!(found.iter().any(|(_, _, p)| *p == port), "the scan did not see the service's own port: {found:?}");
        for (pid, ip, p) in &found {
            assert!(ip.is_loopback(), "pid {pid} listens on {ip}:{p}, which is not this computer's own address");
        }
    }
    let _ = child.kill();
    let _ = child.wait();
    let _ = std::fs::remove_dir_all(&home);
}

#[test]
fn an_address_off_this_computer_needs_a_key_and_the_key_guards_v1() {
    let home = std::env::temp_dir().join(format!("glyd-listeners-key-{}", std::process::id()));
    std::fs::create_dir_all(&home).unwrap();
    // no key, an address that is not this computer's own: refused, saying how to have it
    for host in ["0.0.0.0", "::", "192.168.1.5"] {
        let o = engine(&home, &["service", "--host", host, "--port", &free_port().to_string()]).output().unwrap();
        assert!(!o.status.success(), "{host}: the service started with no key");
        let err = String::from_utf8_lossy(&o.stderr);
        assert!(err.contains("--insecure-no-auth") && err.contains("GLYD_API_KEY"), "{host}: {err}");
    }
    // a key: it starts on every address, listens there (that is what was asked), /v1 needs the key, the probes do not
    let port = free_port();
    let mut child = engine(&home, &["service", "--host", "0.0.0.0", "--port", &port.to_string()]).env("GLYD_API_KEY", "scan-key").spawn().unwrap();
    wait_up(port, &mut child);
    assert_eq!(get(port, "/v1/models", None), Some(401));
    assert_eq!(get(port, "/v1/models", Some("wrong")), Some(401));
    assert_eq!(get(port, "/v1/models", Some("scan-key")), Some(200));
    assert_eq!(get(port, "/metrics", None), Some(401));
    assert_eq!(get(port, "/ready", None), Some(200));
    let _ = child.kill();
    let _ = child.wait();
    // the user's word instead of a key: it starts, and says what it is
    let port = free_port();
    let mut child = engine(&home, &["service", "--host", "0.0.0.0", "--port", &port.to_string(), "--insecure-no-auth"]).spawn().unwrap();
    wait_up(port, &mut child);
    assert_eq!(get(port, "/v1/models", None), Some(200));
    let _ = child.kill();
    let _ = child.wait();
    let _ = std::fs::remove_dir_all(&home);
}
