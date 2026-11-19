//! GLY-364: the coordinator's beta web UI, opt-in with `daemon --ui 127.0.0.1:PORT`.
//!
//! Any web page can make a browser send requests to a loopback port, so this listener trusts
//! none of them: it checks `Host` exactly (against DNS rebinding), refuses other origins,
//! answers no CORS preflight and sends no CORS headers, and its API needs this launch's random
//! secret in the `X-Starlings-UI` header. The page reads the secret from the URL fragment the
//! daemon prints; a browser never sends a fragment. The API is read-only apart from chat.
//!
//! Chat goes to the pipeline's own client listener, through the same admission and dispatch as
//! any client, as the client node named by `--ui-client`. Its token is issued in memory for that
//! client's existing permission and never written out: the UI can do what that client can, and
//! a caller that is not on this machine reaches neither the UI nor the token.
use serde_json::{Value, json};
use starlings_protocol::{Digest, NodeId};
use starlings_security::{Action, Authority, ClientToken, ClientTokenIssuer, Scope};
use std::{
    io::{ErrorKind, Read, Write},
    net::{SocketAddr, TcpListener, TcpStream},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

const PAGE: &str = include_str!("../ui/index.html");
const MAX_HEAD: usize = 8 * 1024;
const MAX_BODY: usize = 512 * 1024;
/// Upstream bytes held at once: one SSE event (at most 64 KiB) or an error reply.
const MAX_HELD: usize = 256 * 1024;
const REQUEST_TIME: Duration = Duration::from_secs(5);
const CONNECTIONS: usize = 8;
/// Seconds a UI chat token lives; a new one is issued after half of it.
const TOKEN_LIFETIME: u32 = 600;
const COMMON: &str = "Connection: close\r\nCache-Control: no-store\r\nX-Content-Type-Options: nosniff\r\nX-Frame-Options: DENY\r\nReferrer-Policy: no-referrer\r\nContent-Security-Policy: default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data:; connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'\r\n";

/// `--ui ADDRESS [--ui-client NODE]`.
#[derive(Clone)]
pub struct Config {
    pub listen: SocketAddr,
    pub client: Option<NodeId>,
}

/// Where UI chat goes: the pipeline's client listener, as `client`.
pub struct Chat {
    pub address: SocketAddr,
    pub model: Digest,
    pub lifetime_ms: u32,
    pub client: NodeId,
    pub issuer: ClientTokenIssuer,
}

pub struct Ui {
    listener: TcpListener,
    shared: Arc<Shared>,
}
struct Shared {
    secret: String,
    hosts: [String; 2],
    /// The control loop's latest snapshot, with what never changes (`join`) added on read.
    state: Mutex<Value>,
    join: Value,
    chat: Option<Chat>,
    token: Mutex<Option<(ClientToken, Instant)>>,
    active: AtomicUsize,
    /// The daemon is stopping: a chat in flight ends.
    stop: Arc<AtomicBool>,
}

impl Ui {
    pub fn bind(
        listen: SocketAddr,
        chat: Option<Chat>,
        join: Value,
        stop: Arc<AtomicBool>,
    ) -> Result<Self, &'static str> {
        if !listen.ip().is_loopback() {
            return Err("UI_LOOPBACK_REQUIRED: --ui takes 127.0.0.1:PORT or [::1]:PORT");
        }
        let listener = TcpListener::bind(listen).map_err(|_| "UI_BIND_FAILED")?;
        listener
            .set_nonblocking(true)
            .map_err(|_| "UI_BIND_FAILED")?;
        let bound = listener.local_addr().map_err(|_| "UI_BIND_FAILED")?;
        // Two v4 UUIDs: 244 random bits from the OS.
        let secret = format!(
            "{}{}",
            uuid::Uuid::new_v4().simple(),
            uuid::Uuid::new_v4().simple()
        );
        Ok(Self {
            listener,
            shared: Arc::new(Shared {
                secret,
                hosts: [bound.to_string(), format!("localhost:{}", bound.port())],
                state: Mutex::new(Value::Null),
                join,
                chat,
                token: Mutex::new(None),
                active: AtomicUsize::new(0),
                stop,
            }),
        })
    }
    /// The page's address; the fragment carries the secret and never reaches a server.
    pub fn url(&self) -> String {
        format!("http://{}/#k={}", self.shared.hosts[0], self.shared.secret)
    }
    pub fn publish(&self, state: Value) {
        if let Ok(mut latest) = self.shared.state.lock() {
            *latest = state;
        }
    }
    /// Accept one connection and serve it on its own thread; past CONNECTIONS, drop it.
    pub fn try_accept(&self) -> Option<JoinHandle<()>> {
        let (socket, _) = self.listener.accept().ok()?;
        if self.shared.active.fetch_add(1, Ordering::AcqRel) >= CONNECTIONS {
            self.shared.active.fetch_sub(1, Ordering::AcqRel);
            return None;
        }
        let shared = self.shared.clone();
        let spawned = thread::Builder::new()
            .name("ui-request".into())
            .spawn(move || {
                let _ = shared.serve(socket);
                shared.active.fetch_sub(1, Ordering::AcqRel);
            });
        spawned
            .inspect_err(|_| {
                self.shared.active.fetch_sub(1, Ordering::AcqRel);
            })
            .ok()
    }
}

/// The cluster as the coordinator sees it now: nodes with their latest resource samples and
/// the pipeline route's layer split, with its client listener's bound address. Built on the
/// control loop, which owns the authority.
pub fn snapshot(
    authority: &Authority,
    route: Option<(&crate::routes::PipelineRoute, SocketAddr)>,
) -> Value {
    let nodes: Vec<Value> = authority
        .registry_nodes()
        .unwrap_or_default()
        .into_iter()
        .map(|n| {
            json!({
                "node": n.node, "role": n.role, "status": n.status,
                "schedulable": n.schedulable, "fresh_links": n.fresh_links,
                "resources": authority.registry_resources(n.node).ok().flatten(),
            })
        })
        .collect();
    let route = route.map(|(r, listen)| {
        let stages: Vec<Value> = r
            .plan
            .stages
            .iter()
            .map(|s| {
                json!({
                    "node": s.node_id, "layer_start": s.layer_start.get(),
                    "layer_end": s.layer_end_exclusive.get(), "embedding": s.owns_embedding,
                    "head": s.owns_final_norm_and_head, "resident_bytes": s.resident_bytes.0,
                    "device": s.device_id,
                })
            })
            .collect();
        json!({
            "model": r.plan.manifest_digest, "plan": r.plan.plan_digest,
            "layers": r.plan.model_layer_count.get(), "listen": listen,
            "draft_tokens": r.draft_tokens, "chunk_tokens": r.chunk_tokens,
            "max_output_tokens": r.max_output_tokens, "lifetime_ms": r.lifetime_ms,
            "sessions": r.stages.first().map_or(1, |s| 1 + s.more_data.len()),
            "stages": stages,
        })
    });
    json!({
        "cluster": authority.identity().cluster(), "coordinator": authority.identity().node(),
        "epoch": authority.registry_epoch().ok().map(|e| e.number.0),
        "nodes": nodes, "route": route,
    })
}

struct Request {
    method: String,
    path: String,
    headers: Vec<(String, Vec<u8>)>,
    body: Vec<u8>,
}
impl Request {
    fn header(&self, name: &str) -> Option<&[u8]> {
        self.headers
            .iter()
            .find(|(n, _)| n.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.as_slice())
    }
}

impl Shared {
    fn serve(&self, mut socket: TcpStream) -> std::io::Result<()> {
        socket.set_nonblocking(false)?;
        socket.set_write_timeout(Some(Duration::from_secs(10)))?;
        let request = match read_request(&mut socket) {
            Ok(request) => request,
            Err(status) => return reply(&mut socket, status, "text/plain", b""),
        };
        if let Err(status) = self.check(&request) {
            return reply(&mut socket, status, "text/plain", b"");
        }
        match (request.method.as_str(), request.path.as_str()) {
            ("GET", "/") => reply(
                &mut socket,
                "200 OK",
                "text/html; charset=utf-8",
                PAGE.as_bytes(),
            ),
            ("GET", "/ui/v1/state") => {
                let body = serde_json::to_vec(&self.state()).unwrap_or_default();
                reply(&mut socket, "200 OK", "application/json", &body)
            }
            ("POST", "/ui/v1/chat") => self.chat(&mut socket, &request.body),
            _ => reply(&mut socket, "404 Not Found", "text/plain", b""),
        }
    }
    /// Host exactly this listener's; any Origin this page's own; the API needs the secret.
    fn check(&self, request: &Request) -> Result<(), &'static str> {
        // OPTIONS among the others: no preflight is ever answered.
        if !matches!(request.method.as_str(), "GET" | "POST") {
            return Err("405 Method Not Allowed");
        }
        let host = request
            .header("host")
            .and_then(|h| std::str::from_utf8(h).ok())
            .filter(|h| self.hosts.iter().any(|ours| ours == h))
            .ok_or("403 Forbidden")?;
        if request
            .header("origin")
            .is_some_and(|o| o != format!("http://{host}").as_bytes())
            || request
                .header("sec-fetch-site")
                .is_some_and(|s| s != b"same-origin" && s != b"none")
        {
            return Err("403 Forbidden");
        }
        if request.path.starts_with("/ui/")
            && !request
                .header("x-starlings-ui")
                .is_some_and(|key| same(key, self.secret.as_bytes()))
        {
            return Err("401 Unauthorized");
        }
        Ok(())
    }
    fn state(&self) -> Value {
        let mut state = self.state.lock().map(|s| s.clone()).unwrap_or(Value::Null);
        state["join"] = self.join.clone();
        state["chat"] = json!(self.chat.is_some());
        state["reports"] = crate::pipeline::REPORTS
            .lock()
            .map(|r| r.iter().cloned().collect())
            .unwrap_or_default();
        state["now"] =
            json!((time::OffsetDateTime::now_utc().unix_timestamp_nanos() / 1_000_000) as i64);
        state
    }
    /// One chat: the page's turns and settings, as a streamed request to the pipeline's client
    /// listener. Its SSE comes back as is; the last chunk gains the request's report.
    fn chat(&self, socket: &mut TcpStream, body: &[u8]) -> std::io::Result<()> {
        let Some(chat) = &self.chat else {
            return fail(
                socket,
                "503 Service Unavailable",
                "Chat is off: start the coordinator with --ui-client NODE",
            );
        };
        let Ok(input) = serde_json::from_slice::<Value>(body) else {
            return fail(socket, "400 Bad Request", "Request refused");
        };
        let request = json!({
            "model": chat.model, "messages": input["messages"], "max_tokens": input["max_tokens"],
            "temperature": input["temperature"], "top_p": 1.0, "seed": input["seed"],
            "stream": true,
        });
        let Some(bearer) = self.bearer(chat) else {
            return fail(
                socket,
                "403 Forbidden",
                "The --ui-client node has no infer permission for this model",
            );
        };
        let upstream = TcpStream::connect_timeout(&chat.address, Duration::from_secs(2))
            .and_then(|mut upstream| {
                let request = serde_json::to_vec(&request)?;
                write!(upstream, "POST /v1/chat/completions HTTP/1.1\r\nHost: {}\r\nAuthorization: Bearer {bearer}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n", chat.address, request.len())?;
                upstream.write_all(&request)?;
                Ok(upstream)
            });
        let Ok(upstream) = upstream else {
            return fail(
                socket,
                "502 Bad Gateway",
                "The pipeline's client listener did not answer",
            );
        };
        let deadline = Instant::now()
            + Duration::from_millis(chat.lifetime_ms.into())
            + Duration::from_secs(15);
        relay(socket, upstream, deadline, &self.stop)
    }
    /// This UI's in-memory token for its client; a new one after half the last one's life.
    fn bearer(&self, chat: &Chat) -> Option<String> {
        let mut held = self.token.lock().ok()?;
        if held.as_ref().is_none_or(|(_, issued)| {
            issued.elapsed() >= Duration::from_secs((TOKEN_LIFETIME / 2).into())
        }) {
            let scope = Scope {
                model: chat.model.clone(),
                action: Action::Infer,
            };
            let token = chat
                .issuer
                .issue(chat.client, &[scope], TOKEN_LIFETIME)
                .ok()?;
            *held = Some((token, Instant::now()));
        }
        held.as_ref().map(|(token, _)| token.bearer())
    }
}

/// Copy the pipeline's reply to the browser: an error as it came, SSE event by event. Ends when
/// either side closes or at `deadline`; dropping `upstream` cancels the request.
fn relay(
    browser: &mut TcpStream,
    mut upstream: TcpStream,
    deadline: Instant,
    stop: &AtomicBool,
) -> std::io::Result<()> {
    upstream.set_read_timeout(Some(Duration::from_millis(250)))?;
    let (mut held, mut status, mut piece) = (Vec::new(), None::<String>, [0u8; 16 * 1024]);
    loop {
        if Instant::now() >= deadline || stop.load(Ordering::Relaxed) {
            return Ok(());
        }
        match upstream.read(&mut piece) {
            Ok(0) => break,
            Ok(n) => held.extend_from_slice(&piece[..n]),
            Err(e) if matches!(e.kind(), ErrorKind::WouldBlock | ErrorKind::TimedOut) => {
                if gone(browser) {
                    return Ok(());
                }
                continue;
            }
            Err(e) => return Err(e),
        }
        if held.len() > MAX_HELD {
            return Ok(());
        }
        if status.is_none() {
            let Some(end) = find(&held, b"\r\n\r\n") else {
                continue;
            };
            let line = std::str::from_utf8(&held[..end])
                .ok()
                .and_then(|head| head.lines().next())
                .and_then(|line| line.strip_prefix("HTTP/1.1 "))
                .filter(|s| s.len() < 64 && s.bytes().all(|b| (b' '..=b'~').contains(&b)))
                .unwrap_or("502 Bad Gateway")
                .to_owned();
            held.drain(..end + 4);
            if line.starts_with("200 ") {
                write!(
                    browser,
                    "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n{COMMON}\r\n"
                )?;
            }
            status = Some(line);
        }
        if status.as_deref().is_some_and(|s| s.starts_with("200 ")) {
            while let Some(end) = find(&held, b"\n\n") {
                let event: Vec<u8> = held.drain(..end + 2).collect();
                browser.write_all(&annotate(event))?;
            }
        }
    }
    match status {
        Some(s) if s.starts_with("200 ") => Ok(()),
        Some(s) if starlings_protocol::parse_json(&held).is_ok() => {
            reply(browser, &s, "application/json", &held)
        }
        _ => fail(
            browser,
            "502 Bad Gateway",
            "The pipeline closed the request",
        ),
    }
}

/// The answer's last chunk (the one with a finish reason) gains `starlings`: this request's
/// report, kept by the pipeline before that chunk went out.
fn annotate(event: Vec<u8>) -> Vec<u8> {
    let Some(mut chunk) = event
        .strip_prefix(b"data: ")
        .and_then(|data| serde_json::from_slice::<Value>(data).ok())
    else {
        return event;
    };
    if !chunk["choices"][0]["finish_reason"].is_string() {
        return event;
    }
    let report = crate::pipeline::REPORTS
        .lock()
        .ok()
        .and_then(|r| r.iter().rev().find(|r| r["id"] == chunk["id"]).cloned());
    chunk["starlings"] = report.unwrap_or(Value::Null);
    // Compact JSON has no raw newline, so no SSE field can be injected.
    format!("data: {chunk}\n\n").into_bytes()
}

/// The browser closed its side (the page's Stop, or the tab).
fn gone(socket: &TcpStream) -> bool {
    if socket.set_nonblocking(true).is_err() {
        return true;
    }
    let closed = match socket.peek(&mut [0]) {
        Ok(0) => true,
        Ok(_) => false,
        Err(e) => e.kind() != ErrorKind::WouldBlock,
    };
    socket.set_nonblocking(false).is_err() || closed
}

fn find(bytes: &[u8], needle: &[u8]) -> Option<usize> {
    bytes.windows(needle.len()).position(|w| w == needle)
}

/// Constant time for equal lengths; the secret's length is no secret.
fn same(a: &[u8], b: &[u8]) -> bool {
    a.len() == b.len() && a.iter().zip(b).fold(0, |d, (x, y)| d | (x ^ y)) == 0
}

fn read_request(socket: &mut TcpStream) -> Result<Request, &'static str> {
    const BAD: &str = "400 Bad Request";
    let deadline = Instant::now() + REQUEST_TIME;
    let mut bytes = vec![0; MAX_HEAD];
    let mut len = 0;
    let left = || {
        deadline
            .checked_duration_since(Instant::now())
            .filter(|d| !d.is_zero())
            .ok_or("408 Request Timeout")
    };
    let (method, path, headers, start) = loop {
        if len == bytes.len() {
            return Err("431 Request Header Fields Too Large");
        }
        socket.set_read_timeout(Some(left()?)).map_err(|_| BAD)?;
        match socket.read(&mut bytes[len..]) {
            Ok(0) | Err(_) => return Err(BAD),
            Ok(n) => len += n,
        }
        let mut headers = [httparse::EMPTY_HEADER; 32];
        let mut parsed = httparse::Request::new(&mut headers);
        let Ok(status) = parsed.parse(&bytes[..len]) else {
            return Err(BAD);
        };
        let httparse::Status::Complete(start) = status else {
            continue;
        };
        let (Some(method), Some(path)) = (parsed.method, parsed.path) else {
            return Err(BAD);
        };
        let headers: Vec<(String, Vec<u8>)> = parsed
            .headers
            .iter()
            .map(|h| (h.name.to_ascii_lowercase(), h.value.to_vec()))
            .collect();
        // One value per header name, so a check cannot read one copy and a peer another.
        if headers
            .iter()
            .enumerate()
            .any(|(i, (n, _))| headers[..i].iter().any(|(p, _)| p == n))
        {
            return Err(BAD);
        }
        break (method.to_owned(), path.to_owned(), headers, start);
    };
    let mut request = Request {
        method,
        path,
        headers,
        body: Vec::new(),
    };
    if request.header("transfer-encoding").is_some() {
        return Err(BAD);
    }
    let body_len = match request.header("content-length") {
        Some(v) => std::str::from_utf8(v)
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .ok_or(BAD)?,
        None => 0,
    };
    if body_len > MAX_BODY {
        return Err("413 Content Too Large");
    }
    if len - start > body_len {
        return Err(BAD);
    }
    request.body = bytes[start..len].to_vec();
    request.body.resize(body_len, 0);
    let mut offset = len - start;
    while offset < body_len {
        socket.set_read_timeout(Some(left()?)).map_err(|_| BAD)?;
        match socket.read(&mut request.body[offset..]) {
            Ok(0) | Err(_) => return Err(BAD),
            Ok(n) => offset += n,
        }
    }
    Ok(request)
}

fn reply(socket: &mut TcpStream, status: &str, kind: &str, body: &[u8]) -> std::io::Result<()> {
    write!(
        socket,
        "HTTP/1.1 {status}\r\nContent-Type: {kind}\r\nContent-Length: {}\r\n{COMMON}\r\n",
        body.len()
    )?;
    socket.write_all(body)
}

fn fail(socket: &mut TcpStream, status: &str, message: &str) -> std::io::Result<()> {
    let body = json!({"error": {"message": message}}).to_string();
    reply(socket, status, "application/json", body.as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One request's whole reply from a UI with this snapshot.
    fn exchange(ui: &Ui, request: &str) -> String {
        let address = ui.listener.local_addr().unwrap();
        let mut client = TcpStream::connect(address).unwrap();
        client.write_all(request.as_bytes()).unwrap();
        let deadline = Instant::now() + Duration::from_secs(5);
        let job = loop {
            if let Some(job) = ui.try_accept() {
                break job;
            }
            assert!(Instant::now() < deadline, "not accepted");
            thread::sleep(Duration::from_millis(5));
        };
        let mut reply = String::new();
        client.read_to_string(&mut reply).unwrap();
        job.join().unwrap();
        reply
    }

    #[test]
    fn listener_checks_host_origin_and_secret_and_answers_no_preflight() {
        let stop = Arc::new(AtomicBool::new(false));
        assert!(
            Ui::bind(
                "0.0.0.0:0".parse().unwrap(),
                None,
                Value::Null,
                stop.clone()
            )
            .is_err()
        );
        let join = json!({"state_dir": "/s"});
        let ui = Ui::bind("127.0.0.1:0".parse().unwrap(), None, join, stop).unwrap();
        ui.publish(json!({"nodes": [], "route": null}));
        let host = ui.shared.hosts[0].clone();
        let key = ui.shared.secret.clone();
        assert!(ui.url().ends_with(&format!("/#k={key}")));
        let get =
            |path: &str, extra: &str| exchange(&ui, &format!("GET {path} HTTP/1.1\r\n{extra}\r\n"));
        let status = |reply: String| reply.lines().next().unwrap_or("").to_owned();

        let page = get("/", &format!("Host: {host}\r\n"));
        assert!(page.starts_with("HTTP/1.1 200 OK"), "{page}");
        assert!(page.contains("Content-Security-Policy: default-src 'none'"));
        assert!(page.contains("<title>Starlings</title>"));
        // A rebound name, a missing Host, another site's page or a sibling port: refused.
        for extra in [
            "Host: attacker.example:80\r\n".to_owned(),
            String::new(),
            format!("Host: {host}\r\nOrigin: http://attacker.example\r\n"),
            format!("Host: {host}\r\nSec-Fetch-Site: cross-site\r\n"),
            format!("Host: {host}\r\nHost: {host}\r\n"),
            "Host: 127.0.0.1:1\r\n".to_owned(),
        ] {
            assert_eq!(
                status(get("/", &extra)),
                if extra.matches("Host:").count() == 2 {
                    "HTTP/1.1 400 Bad Request"
                } else {
                    "HTTP/1.1 403 Forbidden"
                },
                "{extra}"
            );
        }
        // The API needs the secret, compared whole.
        for extra in [
            String::new(),
            "X-Starlings-UI: wrong\r\n".to_owned(),
            format!("X-Starlings-UI: {}\r\n", &key[..63]),
            format!("X-Starlings-UI: {key}0\r\n"),
        ] {
            assert_eq!(
                status(get("/ui/v1/state", &format!("Host: {host}\r\n{extra}"))),
                "HTTP/1.1 401 Unauthorized",
                "{extra}"
            );
        }
        let state = get(
            "/ui/v1/state",
            &format!(
                "Host: localhost:{}\r\nX-Starlings-UI: {key}\r\nOrigin: http://localhost:{}\r\nSec-Fetch-Site: same-origin\r\n",
                ui.listener.local_addr().unwrap().port(),
                ui.listener.local_addr().unwrap().port()
            ),
        );
        assert!(state.starts_with("HTTP/1.1 200 OK"), "{state}");
        assert!(!state.to_ascii_lowercase().contains("access-control"));
        let body: Value = serde_json::from_str(state.split("\r\n\r\n").nth(1).unwrap()).unwrap();
        assert_eq!(body["join"]["state_dir"], "/s");
        assert_eq!(body["chat"], false);
        assert!(body["reports"].is_array() && body["now"].is_i64());
        // No preflight: no CORS header, ever.
        let options = exchange(
            &ui,
            &format!(
                "OPTIONS /ui/v1/chat HTTP/1.1\r\nHost: {host}\r\nOrigin: http://{host}\r\nAccess-Control-Request-Method: POST\r\nAccess-Control-Request-Headers: x-starlings-ui\r\n\r\n"
            ),
        );
        assert_eq!(status(options.clone()), "HTTP/1.1 405 Method Not Allowed");
        assert!(!options.to_ascii_lowercase().contains("access-control"));
        // Without --ui-client, chat says how to turn it on.
        let body = "{}";
        let chat = exchange(
            &ui,
            &format!(
                "POST /ui/v1/chat HTTP/1.1\r\nHost: {host}\r\nX-Starlings-UI: {key}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
                body.len()
            ),
        );
        assert!(chat.starts_with("HTTP/1.1 503"), "{chat}");
        assert!(chat.contains("--ui-client"));
        // Bounded: an oversized body is refused before it is read.
        let big = exchange(
            &ui,
            &format!(
                "POST /ui/v1/chat HTTP/1.1\r\nHost: {host}\r\nX-Starlings-UI: {key}\r\nContent-Length: {}\r\n\r\n",
                MAX_BODY + 1
            ),
        );
        assert_eq!(status(big), "HTTP/1.1 413 Content Too Large");
    }

    /// ui/preview.html is the page plus one block of sample state, for viewing without a cluster.
    #[test]
    fn preview_is_the_page_with_sample_state() {
        let preview = include_str!("../ui/preview.html");
        let (head, rest) = preview.split_once("<!-- preview -->\n").unwrap();
        let (_, tail) = rest.split_once("<!-- /preview -->\n").unwrap();
        assert!(
            format!("{head}{tail}") == PAGE,
            "ui/preview.html must be ui/index.html plus its preview block"
        );
    }

    #[test]
    fn last_chunk_gains_its_report() {
        let id = format!("chatcmpl-{}", uuid::Uuid::new_v4());
        crate::pipeline::REPORTS
            .lock()
            .unwrap()
            .push_back(json!({"id": id, "ms": 12.5}));
        let last =
            json!({"id": id, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]});
        let event = annotate(format!("data: {last}\n\n").into_bytes());
        let text = String::from_utf8(event).unwrap();
        let chunk: Value =
            serde_json::from_str(text.strip_prefix("data: ").unwrap().trim()).unwrap();
        assert_eq!(chunk["starlings"]["ms"], 12.5);
        for other in [
            "data: [DONE]\n\n".to_owned(),
            format!(
                "data: {}\n\n",
                json!({"id": id, "choices": [{"delta": {"content": "x"}, "finish_reason": null}]})
            ),
            "event: error\ndata: {}\n\n".to_owned(),
        ] {
            assert_eq!(annotate(other.clone().into_bytes()), other.into_bytes());
        }
        assert!(same(b"abc", b"abc") && !same(b"abc", b"abd") && !same(b"abc", b"ab"));
    }
}
