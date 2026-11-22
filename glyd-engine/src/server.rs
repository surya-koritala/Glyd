//! An OpenAI-compatible server for one user at a time (HTTP/1.1 on a socket of its own: no framework): `GET /` the chat page, `GET /v1/models`, `POST /v1/chat/completions` and
//! `POST /v1/completions` (streamed as server-sent events or whole), `GET /health`. Connections are read on threads of their own (a browser's idle connection waits for no one), the
//! page, the health check and the model's name answered there, and the completions handed to the engine's thread, which runs one reply at a time in the order they come (the
//! terminal's chat sends its replies the same way).

use crate::chat::{Chat, Message};
use crate::choice::{Cfg, Choice, Delta, Done, ToolsCfg};
use crate::conn::Conn;
use crate::constrain::Spec;
use crate::conv::{Gen, Outcome, Request, Talk};
use crate::ffi::{err, Error, Result};
use crate::model::{Engine, Finish, Reply, MAX_ROWS};
use crate::sampler::{fresh_seed, Lp, Params};
use crate::sched::{Ev, ModelStep, Sched, Submit};
use crate::tok::Tok;
use crate::tools::{self, Format, ToolChoice, ToolEv, Tools};
use crate::wire;
use glyd_gpu::json::{self, Value};
use std::io::{BufRead, Read, Write};
#[cfg(test)]
use std::net::TcpStream;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::mpsc::{Receiver, RecvTimeoutError, Sender};
use std::sync::Arc;
use std::time::Duration;

mod anthropic;
mod frontend;
mod ollama;
mod responses;

const PAGE: &str = include_str!("../data/chat.html");
const MAX_BODY: usize = 16 << 20;
/// The requests for the engine that may be in flight at once (a thread each); the next is answered 503 with `Retry-After`.
const MAX_IN_FLIGHT: usize = 8192;
/// The sequences one request may make (its prompts times `n`).
const MAX_CHOICES: usize = 64;

pub struct Req {
    method: String,
    path: String,
    body: Vec<u8>,
    /// The Authorization header's value, as it came.
    auth: Option<String>,
    /// The Host and Origin headers, and a CORS preflight's questions (the method and the headers it asks about).
    host: String,
    origin: String,
    preflight: Option<(String, Option<String>)>,
    /// The headers that go with an answer for a browser's script (CORS), as the server's rules allow them for this request's Origin.
    cors: String,
    /// The request's place in the server's count of requests in flight, for as long as it is answered (it goes with the request to the thread that answers it): a drain waits for these.
    inflight: Option<crate::ops::Inflight>,
}

/// What the engine's thread is asked to do.
pub enum Job {
    /// A completions request, its connection open for the reply.
    Http { s: Box<dyn Conn>, req: Req, chat: bool, id: u64 },
    /// An Anthropic Messages request (`count`: /v1/messages/count_tokens), its connection open for the reply.
    Messages { s: Box<dyn Conn>, req: Req, count: bool, id: u64 },
    /// A request of one of the other APIs (`Endpoint`), its connection open for the reply.
    Api { s: Box<dyn Conn>, req: Req, kind: Endpoint, id: u64 },
    /// A reply for the terminal's chat: its pieces (and, last, how it ended) come back on `out`.
    Term { messages: Vec<Message>, request: Request, out: Sender<Piece>, abort: Arc<AtomicBool> },
    /// A sequence for the scheduler (what a request's thread sends: `Remote`).
    Seq(Submit),
    /// The engine's thread returns.
    Stop,
}

/// The APIs served beside the completions' and Anthropic's, each in a module of its own.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Endpoint {
    /// OpenAI's Responses API (stateless).
    Responses,
    /// Ollama's REST API (`/api/...`): which of its calls is in the request's path.
    Ollama,
}

pub enum Piece {
    Text(String),
    End(std::result::Result<Outcome, String>),
}

/// A request that is not read, with the status it is answered with.
type Rejected = (u16, String);

/// The bounds of the header block, the request line and the header lines together, checked as it is read and before anything in it is looked at, the key included: a client with no credential cannot
/// make the server hold what it likes. vLLM's server stopped a block at 4 MiB and 256 headers (uvicorn's limits, which its launcher sets) and its httptools parser sooner; a client's cookies and tokens
/// are a few KiB. A block past either is answered with 431.
const MAX_HEAD: usize = 64 << 10;
const MAX_HEADERS: usize = 256;

/// What the request line and the headers of a request say that the server uses.
pub struct Head {
    method: String,
    path: String,
    len: usize,
    expect: bool,
    auth: Option<String>,
    /// Anthropic's `x-api-key` header: the key as the Anthropic SDKs and Claude Code send it.
    x_api_key: Option<String>,
    host: String,
    origin: String,
    ask_method: Option<String>,
    ask_headers: Option<String>,
}

/// One line of the header block read into `buf` within what is left of `budget` (a line past it is refused where it is, never read to its end); false at the end of the input.
fn head_line(r: &mut impl BufRead, budget: &mut usize, buf: &mut Vec<u8>) -> std::result::Result<bool, Rejected> {
    buf.clear();
    let n = r.by_ref().take(*budget as u64 + 1).read_until(b'\n', buf).map_err(|e| read_failure(&e))?;
    if n > *budget {
        return Err((431, format!("the request's header block is over {} KiB", MAX_HEAD >> 10)));
    }
    *budget -= n;
    Ok(n > 0)
}

/// The request line and the headers from `r`, within MAX_HEAD bytes and MAX_HEADERS headers; None where the input ends before a request.
pub fn read_head(r: &mut impl BufRead) -> std::result::Result<Option<Head>, Rejected> {
    let (mut budget, mut buf) = (MAX_HEAD, Vec::new());
    if !head_line(r, &mut budget, &mut buf)? {
        return Ok(None);
    }
    let line = String::from_utf8_lossy(&buf).into_owned();
    let mut it = line.split_whitespace();
    let mut h = Head {
        method: it.next().unwrap_or("").to_string(),
        path: it.next().unwrap_or("").to_string(),
        len: 0,
        expect: false,
        auth: None,
        x_api_key: None,
        host: String::new(),
        origin: String::new(),
        ask_method: None,
        ask_headers: None,
    };
    let (mut headers, mut length_seen) = (0, false);
    while head_line(r, &mut budget, &mut buf)? && buf != b"\r\n" && buf != b"\n" {
        headers += 1;
        if headers > MAX_HEADERS {
            return Err((431, format!("the request has more than {MAX_HEADERS} headers")));
        }
        let line = String::from_utf8_lossy(&buf).into_owned();
        let low = line.to_ascii_lowercase();
        let value = line.split_once(':').map(|(_, v)| v.trim().to_string()).unwrap_or_default();
        if let Some(v) = low.strip_prefix("content-length:") {
            // (one Content-Length, all digits: a request whose body's end is in doubt is refused, never read as one thing here and another by what is in front of it)
            let v = v.trim();
            if length_seen || v.is_empty() || !v.bytes().all(|b| b.is_ascii_digit()) {
                return Err((400, "the request's Content-Length is repeated or is not a number".to_string()));
            }
            length_seen = true;
            h.len = v.parse().map_err(|_| (413, format!("the request's body is over {} MiB", MAX_BODY >> 20)))?;
        } else if low.starts_with("transfer-encoding:") {
            return Err((400, "a request body with a Transfer-Encoding is not taken: send it with a Content-Length".to_string()));
        } else if low.starts_with("authorization:") {
            h.auth = Some(value);
        } else if low.starts_with("x-api-key:") {
            h.x_api_key = Some(value);
        } else if low.starts_with("host:") {
            h.host = value;
        } else if low.starts_with("origin:") {
            h.origin = value;
        } else if low.starts_with("access-control-request-method:") {
            h.ask_method = Some(value);
        } else if low.starts_with("access-control-request-headers:") {
            h.ask_headers = Some(value);
        } else if low.starts_with("expect:") && low.contains("100-continue") {
            h.expect = true;
        }
    }
    Ok(Some(h))
}

/// How long a client has to send its request line and headers, and then its body, and how long a write may wait on a client that reads nothing: a connection that sends a byte a minute is not let
/// to hold a connection for ever (the time is of the whole, not of each read).
pub const HEAD_TIMEOUT: Duration = Duration::from_secs(30);
pub const BODY_TIMEOUT: Duration = Duration::from_secs(120);
pub const WRITE_TIMEOUT: Duration = Duration::from_secs(60);

/// A read of `s` that waits no longer than `until`: the stream's timeout is what is left of it, and past it the read fails as a timeout does (the background service's proxy reads its requests so).
pub fn read_until_deadline(s: &mut std::net::TcpStream, until: std::time::Instant, buf: &mut [u8]) -> std::io::Result<usize> {
    let left = until.saturating_duration_since(std::time::Instant::now());
    if left.is_zero() {
        return Err(std::io::Error::new(std::io::ErrorKind::TimedOut, "the client was too slow"));
    }
    s.set_read_timeout(Some(left))?;
    s.read(buf)
}

/// The status a failed read is answered with: 408 where the client was too slow, else 400.
pub fn read_failure(e: &std::io::Error) -> Rejected {
    match e.kind() {
        std::io::ErrorKind::TimedOut | std::io::ErrorKind::WouldBlock => (408, "the request was not sent in time".to_string()),
        _ => (400, e.to_string()),
    }
}

/// The request a head and its body make.
fn req_of(h: Head, body: Vec<u8>) -> Req {
    let preflight = if h.method == "OPTIONS" && !h.origin.is_empty() { h.ask_method.map(|m| (m, h.ask_headers)) } else { None };
    // (the key as `x-api-key` is the same key as a bearer token: the Authorization header, where both come, is the one read)
    let auth = h.auth.or(h.x_api_key.map(|k| format!("Bearer {k}")));
    Req { method: h.method, path: h.path, body, auth, host: h.host, origin: h.origin, preflight, cors: String::new(), inflight: None }
}

fn status_text(code: u16) -> &'static str {
    match code {
        200 => "OK",
        400 => "Bad Request",
        401 => "Unauthorized",
        403 => "Forbidden",
        404 => "Not Found",
        405 => "Method Not Allowed",
        408 => "Request Timeout",
        413 => "Payload Too Large",
        421 => "Misdirected Request",
        431 => "Request Header Fields Too Large",
        503 => "Service Unavailable",
        _ => "Internal Server Error",
    }
}

fn respond(s: &mut dyn Conn, code: u16, ctype: &str, body: &[u8]) {
    respond_with(s, code, ctype, "", body);
}

/// `respond` with more header lines (each ending in CRLF).
fn respond_with(s: &mut dyn Conn, code: u16, ctype: &str, extra: &str, body: &[u8]) {
    // (the head and the body in one write: the answer is one chunk of the pipe to the client)
    s.write_all(&response_bytes(code, ctype, extra, body)).ok();
}

/// A whole answer: its head (the headers of every answer, `extra` more, each ending in CRLF; `Connection: close` is the handler's word for the front end to make `keep-alive` where it keeps the
/// connection) and its body.
fn response_bytes(code: u16, ctype: &str, extra: &str, body: &[u8]) -> Vec<u8> {
    let mut out = format!(
        "HTTP/1.1 {code} {}\r\nContent-Type: {ctype}\r\nContent-Length: {}\r\nCache-Control: no-store\r\n{extra}Connection: close\r\n\r\n",
        status_text(code),
        body.len()
    )
    .into_bytes();
    out.extend_from_slice(body);
    out
}

/// The names this server answers to where it is on this computer's own address: the Host of a request must be one of them (the name a page whose DNS points at this computer has is not).
const LOOPBACK: [&str; 3] = ["localhost", "127.0.0.1", "[::1]"];

/// The names a container uses for the computer it runs on (Docker Desktop's, which its WSL integration and Kubernetes use too, Podman's, Lima's and Minikube's): a program in a container that is set up to reach the
/// server there sends one as its Host. They name this computer to a program inside the container, and resolve nowhere else; the server's address is still its own.
const CONTAINER_HOSTS: [&str; 6] = [
    "host.docker.internal",
    "gateway.docker.internal",
    "kubernetes.docker.internal",
    "host.containers.internal",
    "host.lima.internal",
    "host.minikube.internal",
];

/// What the user let in beyond the program's own (`--allowed-origins`, `--allowed-hosts`; both are empty by default).
#[derive(Default, Debug, Clone, PartialEq, Eq)]
pub struct Access {
    /// Web pages (`http://localhost:5173`) whose scripts may call the API: they are answered with CORS headers for their own origin, and never with `*`.
    pub origins: Vec<String>,
    /// Names a request's Host may be besides the computer's own (a name that DNS gives this computer, a proxy's).
    pub hosts: Vec<String>,
}

static ACCESS: std::sync::OnceLock<Access> = std::sync::OnceLock::new();

fn access() -> &'static Access {
    ACCESS.get_or_init(Access::default)
}

/// `Access` from the two flags' values (comma-separated lists); the reason where one is not a plain origin or name. An origin is scheme, host and port and nothing else; `*` is refused (the way to let every page in is not offered).
pub fn parse_access(origins: Option<&str>, hosts: Option<&str>) -> std::result::Result<Access, String> {
    let list = |v: Option<&str>| v.unwrap_or("").split(',').map(|x| x.trim().to_string()).filter(|x| !x.is_empty()).collect::<Vec<_>>();
    let mut a = Access::default();
    for o in list(origins) {
        let o = o.to_lowercase();
        let rest = o.strip_prefix("http://").or_else(|| o.strip_prefix("https://"));
        if rest.is_none_or(|r| r.is_empty() || r.contains(['/', '?', '#', '*', '@', ' '])) {
            return Err(format!(
                "--allowed-origins {o}: an origin is a scheme, a host and a port, like http://localhost:5173 (no path, and not *: list each page)"
            ));
        }
        a.origins.push(o);
    }
    for h in list(hosts) {
        if h.contains(['/', '*', '@', ' ', '?', '#']) || h.contains("://") {
            return Err(format!("--allowed-hosts {h}: a host is a name or an address, like glyd.example.com (a port is not part of it)"));
        }
        a.hosts.push(hostname(&h));
    }
    Ok(a)
}

/// Sets what the user let in (once, where the program starts its listeners).
pub fn set_access(a: Access) {
    let _ = ACCESS.set(a);
}

/// The name in a Host header: `localhost:8000` is `localhost`, `[::1]:8000` is `[::1]`.
fn hostname(host: &str) -> String {
    let h = host.trim().to_lowercase();
    if h.starts_with('[') {
        if let Some(i) = h.find(']') {
            return h[..=i].to_string();
        }
    }
    h.split(':').next().unwrap_or("").to_string()
}

/// Whether an address the server is bound to (`--host`) is this computer's own: `localhost`, 127.x.x.x, `::1`. 0.0.0.0 and `::` are every interface, not this computer's own.
pub fn is_loopback_addr(host: &str) -> bool {
    let h = host.trim().trim_matches(['[', ']']);
    h.eq_ignore_ascii_case("localhost") || h.parse::<std::net::IpAddr>().is_ok_and(|ip| ip.is_loopback())
}

/// The reason a server must not start on `host` as it is told: an address that is not this computer's own with no API key, and no `--insecure-no-auth` to say that is meant. None where it may.
pub fn bind_refusal(host: &str, has_key: bool, insecure_no_auth: bool) -> Option<String> {
    if is_loopback_addr(host) || has_key || insecure_no_auth {
        return None;
    }
    Some(format!(
        "{host} is not this computer's own address, and there is no API key: anyone who can reach it could send the model prompts. Set a key (GLYD_API_KEY=KEY in the environment), or give --insecure-no-auth if an open server on {host} is what you mean. Without --host the server listens on 127.0.0.1 only."
    ))
}

/// A string as Python's repr writes it (the first `n` characters, in single quotes), as the Python server's refusals did.
fn py_repr(s: &str, n: usize) -> String {
    let t: String = s.chars().take(n).collect();
    let quote = if t.contains('\'') && !t.contains('"') { '"' } else { '\'' };
    let mut o = String::new();
    o.push(quote);
    for c in t.chars() {
        match c {
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            '\r' => o.push_str("\\r"),
            '\t' => o.push_str("\\t"),
            c if c == quote => {
                o.push('\\');
                o.push(c);
            }
            c if (c as u32) < 0x20 || c as u32 == 0x7f => o.push_str(&format!("\\x{:02x}", c as u32)),
            c => o.push(c),
        }
    }
    o.push(quote);
    o
}

/// (status, why) for a request the server does not answer, or None. On this computer's own address (`local_only`) a Host that is not a loopback or container name (or one the user allowed) is a DNS-rebinding page's (421). On every
/// address an Origin that is not the server's own (`http(s)://` and the Host) or one the user allowed is another page's script (403, a preflight too). Where the server is open to the network the Host is any name the computer has,
/// unless the user listed them. A request with no Host or no Origin is not refused for the missing one (curl and the OpenAI libraries send no Origin).
pub fn refusal(host: &str, origin: &str, local_only: bool) -> Option<(u16, String)> {
    refusal_with(host, origin, local_only, access())
}

fn refusal_with(host: &str, origin: &str, local_only: bool, allowed: &Access) -> Option<(u16, String)> {
    let name = hostname(host);
    let own =
        LOOPBACK.contains(&name.as_str()) || CONTAINER_HOSTS.contains(&name.as_str()) || name.parse::<std::net::Ipv4Addr>().is_ok_and(|ip| ip.is_loopback());
    if !host.is_empty() && !allowed.hosts.contains(&name) && if local_only { !own } else { !allowed.hosts.is_empty() && !own } {
        return Some((
            421,
            format!(
                "this server answers to localhost, 127.0.0.1, [::1] and host.docker.internal{}, not to {} (--allowed-hosts NAME lets another name in)",
                if allowed.hosts.is_empty() { String::new() } else { format!(" and {}", allowed.hosts.join(", ")) },
                py_repr(host, 80)
            ),
        ));
    }
    let o = origin.to_lowercase();
    if !origin.is_empty() && !allowed.origins.contains(&o) && !same_origin(&o, host, local_only) {
        return Some((
            403,
            format!(
                "this server answers to its own page, not to a script from {} (--allowed-origins {} lets that page in)",
                py_repr(origin, 80),
                py_repr(origin, 80).trim_matches(['\'', '"'])
            ),
        ));
    }
    None
}

/// Whether `origin` (lower case) is the server's own page: `http://` and the Host as the browser sent it; where the server is open to the network, `https://` too (a proxy that ends TLS in front of it).
fn same_origin(origin: &str, host: &str, local_only: bool) -> bool {
    let h = host.trim().to_lowercase();
    !h.is_empty() && (origin == format!("http://{h}") || (!local_only && origin == format!("https://{h}")))
}

/// The origins this server's own page has (what the CORS answers of a server on this computer's own address allow).
fn own_origins(port: u16) -> Vec<String> {
    LOOPBACK.iter().map(|h| format!("http://{h}:{port}")).collect()
}

/// Whether a script from `origin` is let in with CORS headers: its own page on this computer's own address, or an origin the user listed. Nothing else is, the open server included (there is no `*`).
fn cors_allowed(local_only: bool, port: u16, origin: &str, allowed: &Access) -> bool {
    let o = origin.to_lowercase();
    allowed.origins.contains(&o) || (local_only && own_origins(port).iter().any(|x| *x == o))
}

/// The CORS header lines for an answer to a request from `origin`: none where it sends none, and none where the origin is not its own page's or one the user listed (CORS is off until `--allowed-origins` says otherwise).
fn cors_headers(local_only: bool, port: u16, origin: &str) -> String {
    cors_headers_with(local_only, port, origin, access())
}

fn cors_headers_with(local_only: bool, port: u16, origin: &str, allowed: &Access) -> String {
    if !origin.is_empty() && cors_allowed(local_only, port, origin, allowed) {
        format!("Access-Control-Allow-Origin: {origin}\r\nVary: Origin\r\n")
    } else {
        String::new()
    }
}

/// A CORS preflight's answer as Starlette's CORSMiddleware gives it (vLLM's): 200 `OK` with what is allowed where the origin is, 400 `Disallowed CORS origin` where it is not (a method outside the list too).
fn preflight(s: &mut dyn Conn, local_only: bool, port: u16, origin: &str, method: &str, headers: Option<&str>) {
    const METHODS: [&str; 7] = ["DELETE", "GET", "HEAD", "OPTIONS", "PATCH", "POST", "PUT"];
    let allowed = cors_allowed(local_only, port, origin, access());
    let mut failures = Vec::new();
    if !allowed {
        failures.push("origin");
    }
    if !METHODS.contains(&method) {
        failures.push("method");
    }
    let mut extra = format!("Access-Control-Allow-Methods: {}\r\nAccess-Control-Max-Age: 600\r\n", METHODS.join(", "));
    if allowed {
        extra.push_str(&format!("Access-Control-Allow-Origin: {origin}\r\nVary: Origin\r\n"));
    }
    if let Some(h) = headers {
        extra.push_str(&format!("Access-Control-Allow-Headers: {h}\r\n"));
    }
    if failures.is_empty() {
        respond_with(s, 200, "text/plain; charset=utf-8", &extra, b"OK");
    } else {
        respond_with(s, 400, "text/plain; charset=utf-8", &extra, format!("Disallowed CORS {}", failures.join(", ")).as_bytes());
    }
}

/// Whether an Authorization header's value carries the key, as vLLM's middleware checks it: the scheme `Bearer` (any case), then the token after the first space, compared as SHA-256 digests in constant time.
fn key_matches(digest: &[u8; 32], auth: Option<&str>) -> bool {
    let Some((scheme, token)) = auth.and_then(|a| a.split_once(' ')) else { return false };
    if !scheme.eq_ignore_ascii_case("bearer") {
        return false;
    }
    let got = key_digest(token);
    got.iter().zip(digest).fold(0u8, |d, (a, b)| d | (a ^ b)) == 0
}

fn key_digest(key: &str) -> [u8; 32] {
    use sha2::{Digest, Sha256};
    Sha256::digest(key.as_bytes()).into()
}

/// The page's own headers: what a script in it may load and be framed by (itself only), no sniffing, no referrer.
const PAGE_HEADERS: &str = "Content-Security-Policy: default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'self'; img-src data:; base-uri 'none'; form-action 'none'; frame-ancestors 'none'\r\nX-Content-Type-Options: nosniff\r\nReferrer-Policy: no-referrer\r\n";

/// A JSON string literal.
pub fn jstr(s: &str) -> String {
    let mut o = String::with_capacity(s.len() + 2);
    o.push('"');
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            '\r' => o.push_str("\\r"),
            '\t' => o.push_str("\\t"),
            c if (c as u32) < 0x20 => o.push_str(&format!("\\u{:04x}", c as u32)),
            c => o.push(c),
        }
    }
    o.push('"');
    o
}

/// An error as OpenAI's API writes it: `{"error": {"message", "type", "param", "code"}}`, the type by the status (a client's own mistake `invalid_request_error`, a refused key `authentication_error`, the
/// server's `server_error`) and the code where there is one a client can act on (`invalid_api_key`, `context_length_exceeded`), else null.
fn error_body(msg: &str, code: u16) -> String {
    let msg = said(msg);
    let kind = match code {
        401 => "authentication_error",
        403 => "permission_error",
        429 => "rate_limit_error",
        500..=599 => "server_error",
        _ => "invalid_request_error",
    };
    let slug = match code {
        401 => "invalid_api_key",
        _ if msg.contains("maximum context length") => "context_length_exceeded",
        _ => "",
    };
    let slug = if slug.is_empty() { "null".to_string() } else { jstr(slug) };
    format!("{{\"error\":{{\"message\":{},\"type\":\"{kind}\",\"param\":null,\"code\":{slug}}}}}", jstr(&msg))
}

/// The server runs alone (qwenserve): no vLLM behind it for a refusal to point at, `tool_choice: "none"` keeps the tools out of the prompt (the model would write calls that are not read), and a call
/// required of one tool is a call of that tool.
pub static STANDALONE: AtomicBool = AtomicBool::new(false);

/// A refusal's words; where the server runs alone, without the notes that point at vLLM.
fn said(msg: &str) -> String {
    if STANDALONE.load(Ordering::Relaxed) {
        without_vllm(msg)
    } else {
        msg.to_string()
    }
}

/// A message without its notes in brackets that point at vLLM (`GLYD_ENGINE=0 ...`).
fn without_vllm(msg: &str) -> String {
    let mut m = msg.to_string();
    loop {
        let Some(i) = m.find("GLYD_ENGINE=0") else { break };
        let (Some(a), Some(b)) = (m[..i].rfind('('), m[i..].find(')')) else { break };
        let a = if m[..a].ends_with(' ') { a - 1 } else { a };
        m.replace_range(a..=i + b, "");
    }
    m
}

/// The instruction for JSON mode where the request asks for it (`response_format` json_object, or json_schema with its schema; vLLM's `guided_json` and `structured_outputs` json): the grammar
/// (`constraint_spec`) holds the reply to it, and the prompt says what it is, so that the model writes toward it.
fn json_rule(v: &Value) -> Option<String> {
    let rule = "Reply with one JSON object and nothing else: no code fence, no text before or after it.";
    match constraint_spec(v).ok().flatten()? {
        Spec::JsonObject => Some(rule.to_string()),
        Spec::JsonSchema(schema) => Some(format!("{rule} It must match this JSON schema: {schema}")),
        _ => None,
    }
}

/// What a request asks of the reply's form: OpenAI's `response_format` (`json_object`, `json_schema`, whose schema holds whether or not it says `strict`: vLLM's does), vLLM's `structured_outputs` (one of `json`, `regex`,
/// `choice`, `grammar`, `json_object`) and its older `guided_json`, `guided_regex`, `guided_choice` and `guided_grammar`. An error in words where the request names more than one, or one wrongly.
pub fn constraint_spec(v: &Value) -> std::result::Result<Option<Spec>, String> {
    let mut specs: Vec<(String, Spec)> = Vec::new();
    let text = |x: &Value, what: &str| x.as_str().map(str::to_string).ok_or_else(|| format!("{what} must be a string"));
    let schema = |x: &Value, what: &str| -> std::result::Result<Spec, String> {
        match x {
            Value::Object(_) | Value::Bool(_) => Ok(Spec::JsonSchema(json::to_compact(x))),
            Value::String(t) => json::parse(t).map(|j| Spec::JsonSchema(json::to_compact(&j))).map_err(|e| format!("{what}: not JSON: {e}")),
            _ => Err(format!("{what} must be a JSON schema (an object)")),
        }
    };
    let choice = |x: &Value, what: &str| -> std::result::Result<Spec, String> {
        let l = x.as_array().filter(|l| !l.is_empty() && l.iter().all(|c| c.as_str().is_some()));
        l.map(|l| Spec::Choice(l.iter().filter_map(Value::as_str).map(str::to_string).collect())).ok_or_else(|| format!("{what} must be a list of strings"))
    };
    let set = |v: &Value, k: &str| v.get(k).filter(|x| **x != Value::Null).cloned();
    if let Some(rf) = set(v, "response_format") {
        match rf.get("type").and_then(Value::as_str) {
            Some("text") | None => {}
            Some("json_object") => specs.push(("response_format".into(), Spec::JsonObject)),
            Some("json_schema") => {
                let s = rf.get("json_schema").and_then(|j| j.get("schema")).ok_or("response_format json_schema: no json_schema.schema")?;
                specs.push(("response_format".into(), schema(s, "response_format json_schema.schema")?));
            }
            Some(t) => return Err(format!("response_format of type {t} is not supported: text, json_object and json_schema are")),
        }
    }
    if let Some(so) = set(v, "structured_outputs") {
        let o = match &so {
            Value::Object(_) => &so,
            _ => return Err("structured_outputs must be an object".to_string()),
        };
        for (k, x) in match o {
            Value::Object(m) => m.iter().filter(|(_, x)| *x != Value::Null).collect::<Vec<_>>(),
            _ => Vec::new(),
        } {
            let what = format!("structured_outputs.{k}");
            let spec = match k.as_str() {
                "json" => schema(x, &what)?,
                "regex" => Spec::Regex(text(x, &what)?),
                "choice" => choice(x, &what)?,
                "grammar" => Spec::Grammar(text(x, &what)?),
                "json_object" if x.as_bool() == Some(true) => Spec::JsonObject,
                "json_object" | "disable_fallback" | "disable_any_whitespace" | "disable_additional_properties" | "whitespace_pattern" | "backend" => continue,
                _ => return Err(format!("{what} is not supported")),
            };
            specs.push((what, spec));
        }
    }
    for (k, f) in [
        ("guided_json", &schema as &dyn Fn(&Value, &str) -> std::result::Result<Spec, String>),
        ("guided_regex", &|x: &Value, w: &str| text(x, w).map(Spec::Regex)),
        ("guided_choice", &choice),
        ("guided_grammar", &|x: &Value, w: &str| text(x, w).map(Spec::Grammar)),
    ] {
        if let Some(x) = set(v, k) {
            specs.push((k.to_string(), f(&x, k)?));
        }
    }
    match specs.len() {
        0 => Ok(None),
        1 => Ok(specs.pop().map(|s| s.1)),
        _ => Err(format!("one form of reply at a time: {} are all set", specs.iter().map(|s| s.0.as_str()).collect::<Vec<_>>().join(", "))),
    }
}

fn num(v: &Value, k: &str) -> Option<f64> {
    match v.get(k)? {
        Value::Number(n) => n.parse().ok(),
        _ => None,
    }
}

/// The request's sampling and limits (the OpenAI names, and vLLM's top_k, min_p and repetition_penalty).
pub fn request_of(v: &Value, window: usize, defaults: &Params) -> Request {
    let mut r = Request { max_new: window, ..Request::default() };
    let p = &mut r.params;
    *p = defaults.clone();
    if let Some(x) = num(v, "temperature") {
        // (a temperature above 0 and below 0.01 is 0.01, as vLLM holds it: only 0 is greedy)
        p.temperature = if x > 0.0 && x < 0.01 { 0.01 } else { x as f32 };
    }
    if let Some(x) = num(v, "top_p") {
        p.top_p = x as f32;
    }
    if let Some(x) = num(v, "top_k") {
        p.top_k = x.max(0.0) as usize;
    }
    if let Some(x) = num(v, "min_p") {
        p.min_p = x as f32;
    }
    if let Some(x) = num(v, "presence_penalty") {
        p.presence_penalty = x as f32;
    }
    if let Some(x) = num(v, "frequency_penalty") {
        p.frequency_penalty = x as f32;
    }
    if let Some(x) = num(v, "repetition_penalty") {
        p.repetition_penalty = x as f32;
    }
    p.seed = fresh_seed();
    if let Some(x) = num(v, "seed").filter(|&x| x != -1.0) {
        p.seed = x as i64 as u64;
    }
    p.logit_bias = logit_bias_of(v, usize::MAX).unwrap_or_default();
    if let Some(b) = v.get("ignore_eos").and_then(Value::as_bool) {
        r.ignore_eos = b;
    }
    if let Some(n) = num(v, "max_completion_tokens").or_else(|| num(v, "max_tokens")) {
        r.max_new = (n.max(1.0) as usize).min(window.max(1));
    }
    match v.get("stop") {
        Some(Value::String(s)) => r.stop.push(s.clone()),
        Some(Value::Array(a)) => r.stop.extend(a.iter().filter_map(|x| x.as_str().map(str::to_string))),
        _ => {}
    }
    r.stop_ids = ids_of(v, "stop_token_ids");
    r.include_stop = v.get("include_stop_str_in_output").and_then(Value::as_bool) == Some(true);
    r.skip_special = v.get("skip_special_tokens").and_then(Value::as_bool) != Some(false);
    p.min_tokens = num(v, "min_tokens").map_or(0, |x| x.max(0.0) as usize);
    p.allowed = ids_of(v, "allowed_token_ids");
    // `chat_template_kwargs` as vLLM takes them: each key a variable of the template (`enable_thinking` also sets the engine's own flag)
    if let Some(kw) = v.get("chat_template_kwargs").and_then(Value::as_object) {
        for (k, x) in kw {
            match (k.as_str(), x.as_bool()) {
                ("enable_thinking", Some(b)) => r.thinking = Some(b),
                _ => r.template_kwargs.push((k.clone(), x.clone())),
            }
        }
    }
    // vLLM gives the template `reasoning_effort` and, where `enable_thinking` is not given, thinking on unless the effort is none; `documents` go to the template as they are
    if let Some(e) = v.get("reasoning_effort").and_then(Value::as_str) {
        r.template_kwargs.push(("reasoning_effort".to_string(), Value::from(e)));
        if r.thinking.is_none() {
            r.thinking = Some(e != "none");
        }
    }
    if let Some(d @ Value::Array(_)) = v.get("documents") {
        r.template_kwargs.push(("documents".to_string(), d.clone()));
    }
    r
}

/// A request's list of token ids (absent or null: none).
fn ids_of(v: &Value, k: &str) -> Vec<u32> {
    let mut out: Vec<u32> = Vec::new();
    for t in v.get(k).and_then(Value::as_array).unwrap_or(&[]).iter().filter_map(Value::as_u64) {
        if !out.contains(&(t as u32)) {
            out.push(t as u32);
        }
    }
    out
}

/// The request's `logit_bias` as vLLM takes it: an object whose keys are token ids (as strings) and whose values are numbers, each value held to -100..100; an id must be in the vocabulary. A later key of the
/// same id wins. The message says what is wrong, as vLLM's does.
pub fn logit_bias_of(v: &Value, vocab: usize) -> std::result::Result<Vec<(u32, f32)>, String> {
    let o = match v.get("logit_bias") {
        None | Some(Value::Null) => return Ok(Vec::new()),
        Some(Value::Object(o)) => o,
        Some(_) => return Err("logit_bias must be an object of token ids and numbers".to_string()),
    };
    let (mut out, mut not_ids, mut outside): (Vec<(u32, f32)>, Vec<&str>, Vec<i64>) = (Vec::new(), Vec::new(), Vec::new());
    for (k, x) in o {
        let Value::Number(n) = x else { return Err(format!("logit_bias: the value of {k:?} is not a number")) };
        let Ok(bias) = n.parse::<f64>() else { return Err(format!("logit_bias: the value of {k:?} is not a number")) };
        let Ok(id) = k.trim().parse::<i64>() else {
            not_ids.push(k);
            continue;
        };
        if id < 0 || id as usize >= vocab {
            outside.push(id);
            continue;
        }
        let bias = bias.clamp(-100.0, 100.0) as f32;
        match out.iter_mut().find(|(t, _)| *t == id as u32) {
            Some(e) => e.1 = bias,
            None => out.push((id as u32, bias)),
        }
    }
    if !not_ids.is_empty() {
        return Err(format!("logit_bias contains key(s) that cannot be converted to integer token IDs: {not_ids:?}"));
    }
    if !outside.is_empty() {
        return Err(format!("token_id(s) {outside:?} in logit_bias contain out-of-vocab token ids. Vocabulary size: {vocab}"));
    }
    Ok(out)
}

/// What is wrong in a request's values (a message as vLLM words it), before any work is done for it.
pub fn invalid(v: &Value, chat: bool, vocab: usize) -> Option<String> {
    api_of(v, chat, vocab).err()
}

/// The request fields this server does not do, each with the value that means it was not asked for (null is never a request): where a request has another value it is refused, in words, instead of ignored.
const NOT_DONE: &[(&str, &str)] = &[
    ("use_beam_search", "false"),
    ("bad_words", "[]"),
    ("prompt_logprobs", "null"),
    ("logprob_token_ids", "null"),
    ("thinking_token_budget", "null"),
    ("truncate_prompt_tokens", "null"),
    ("truncation_side", "null"),
    ("return_token_ids", "false"),
    ("return_prompt_text", "false"),
    ("return_token_offsets", "false"),
    ("return_assistant_tokens_mask", "false"),
    ("kv_transfer_params", "null"),
    ("ec_transfer_params", "null"),
    ("vllm_xargs", "null"),
    ("repetition_detection", "null"),
    ("stream_interval", "null"),
    ("priority", "0"),
    ("chat_template", "null"),
    ("media_io_kwargs", "null"),
    ("mm_processor_kwargs", "null"),
    ("continue_final_message", "false"),
    ("prompt_embeds", "null"),
    ("functions", "null"),
    ("function_call", "null"),
    ("audio", "null"),
    ("prediction", "null"),
    ("web_search_options", "null"),
];

/// The fields of constrained decoding that are not done (the grammars' own, `constraint_spec`'s, are): where the server runs alone the grammar's whitespace is llguidance's, and its backend the only one.
const CONSTRAINED: &[&str] = &["guided_whitespace_pattern"];

/// What an engine that runs one sequence at a time (`work_gen`'s: a level's) does not do besides NOT_DONE, as NOT_DONE lists it: it gives no log-probabilities, and no least length (`min_tokens`).
pub const ONE_AT_A_TIME: &[(&str, &str)] = &[("logprobs", "false"), ("min_tokens", "0")];

/// Whether the request sets `name` to something other than null and `default`, the value that means it was not asked for.
fn asked(v: &Value, name: &str, default: &str) -> bool {
    v.get(name).is_some_and(|x| *x != Value::Null && json::parse(default).map_or(true, |d| *x != d))
}

/// What a request asks for that this server does not do, said instead of ignored (a client that asks for a JSON schema, a beam search or the prompt's log-probabilities and gets none of them would go on as if it had).
pub fn unsupported(v: &Value) -> Option<String> {
    let yet = "is not supported by this engine yet (GLYD_ENGINE=0 runs vLLM, which has it)";
    // (the grammars are done, on every model: what is wrong with one is said)
    if let Err(why) = constraint_spec(v) {
        return Some(why);
    }
    for name in CONSTRAINED {
        if v.get(name).is_some_and(|x| *x != Value::Null) {
            return Some(format!("{name} (constrained decoding) {yet}"));
        }
    }
    if let Some((name, _)) = NOT_DONE.iter().find(|(n, d)| asked(v, n, d)) {
        return Some(format!("{name} {yet}"));
    }
    // (a cache scope: vLLM's field, a string; one that is not would be no scope without a word)
    if v.get("cache_salt").is_some_and(|x| *x != Value::Null && x.as_str().is_none_or(str::is_empty)) {
        return Some("cache_salt must be a string that is not empty".to_string());
    }
    if v.get("best_of").and_then(Value::as_u64).is_some_and(|b| b > num(v, "n").map_or(1, |n| n as u64)) {
        return Some(format!("best_of more than n {yet}"));
    }
    None
}

/// The model as OpenAI's list has it, and as Anthropic's does (`type`, `display_name`, `created_at`): the one object serves the clients of both.
fn model_json(name: &str, window: usize) -> String {
    let created = now();
    format!(
        "{{\"id\":{0},\"object\":\"model\",\"created\":{created},\"owned_by\":\"glyd\",\"max_model_len\":{window},\"type\":\"model\",\"display_name\":{0},\"created_at\":\"{1}\"}}",
        jstr(name),
        rfc3339(created)
    )
}

/// A time in seconds since 1970 as RFC 3339 in UTC (`2026-11-20T12:00:00Z`).
fn rfc3339(secs: u64) -> String {
    let (days, rest) = ((secs / 86400) as i64, secs % 86400);
    // (the civil date of a day count: Howard Hinnant's algorithm)
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + (month <= 2) as i64;
    format!("{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}Z", rest / 3600, rest % 3600 / 60, rest % 60)
}

fn now() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs())
}

/// The OpenAI limit on the likeliest tokens a step's log-probabilities list (vLLM's default `--max-logprobs`).
const MAX_LOGPROBS: i64 = 20;

/// The grammar the reply must follow, compiled: its form (`constraint_spec`), or, where the call is forced (`tool_choice` a function or required) and the model's call format is one the grammars know, the call.
/// None where the request asks for neither.
fn constraint_of(sh: &Shared, v: &Value, api: &Api) -> std::result::Result<Option<Arc<crate::constrain::Constraint>>, String> {
    let standalone = STANDALONE.load(Ordering::Relaxed);
    let spec = constraint_spec(v)?;
    let forced = api.tools.as_ref().and_then(|t| {
        let named = match &t.choice {
            ToolChoice::Named(n) => Some(n.clone()),
            // (as `chat_text_of` has it: a call required of one tool is a call of that tool where the server runs alone)
            ToolChoice::Required if standalone && t.list.len() == 1 => {
                t.list[0].get("function").and_then(|f| f.get("name")).and_then(Value::as_str).map(str::to_string)
            }
            ToolChoice::Required => None,
            _ => return None,
        };
        // (Kimi-K3's, Gemma 4's and Muse's own markup: a forced call is opened by the prompt, and goes without a grammar; `constrain::call_grammar` has ATEM's block too, for the named call of Muse, which waits for its end token to be known)
        let c = sh.chat.as_ref().filter(|c| !c.is_xtml() && !c.is_gemma() && !c.is_muse())?;
        crate::constrain::call_grammar(&t.list, named.as_deref(), Format::of(c)).map(Spec::Grammar)
    });
    // (a form the request names is refused where it cannot be held; a forced call is opened by the prompt all the same, and goes without the grammar)
    let call = spec.is_none();
    let Some(spec) = forced.or(spec) else { return Ok(None) };
    let g = match sh.tok.grammars(sh.vocab) {
        Ok(g) => g,
        Err(_) if call => return Ok(None),
        Err(e) => return Err(format!("constrained decoding is not available for this model: {e}")),
    };
    match g.compile(&spec) {
        Ok(c) => Ok(Some(c)),
        Err(_) if call => Ok(None),
        Err(e) => Err(format!("the reply's grammar: {e}")),
    }
}

/// What a request asks for beside its sampling: whether it streams and what the stream carries, how many choices, the log-probabilities.
struct Api {
    stream: bool,
    include_usage: bool,
    n: usize,
    /// The likeliest tokens a token's log-probabilities list (0: the chosen token's alone); None: none asked for.
    logprobs: Option<usize>,
    /// Log-probabilities name a token as `token_id:N` (vLLM's `return_tokens_as_token_ids`).
    as_ids: bool,
    /// The thinking is left out of the reply (vLLM's `include_reasoning: false`).
    hide_reasoning: bool,
    /// The functions the model may call, and how it chooses (a chat's).
    tools: Option<Tools>,
    /// The generation prompt is added (a chat's `add_generation_prompt`, true unless said), and whether the tokenizer adds its special tokens (`add_special_tokens`; unset: a chat's no, a plain prompt's yes).
    add_generation_prompt: bool,
    add_special: Option<bool>,
    /// A plain completion's `echo`: the prompt comes first in each choice's text.
    echo: bool,
    /// The id the reply is given (`request_id`).
    request_id: Option<String>,
    /// How many choices the request makes in all (its prompts times `n`), and the text each prompt's choices begin with where they echo it.
    total: usize,
    echo_texts: Vec<String>,
}

/// A request's integer field: absent or null is None; a number that is not whole is refused.
fn int_field(v: &Value, k: &str) -> std::result::Result<Option<i64>, String> {
    match v.get(k) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Number(n)) => {
            match n.parse::<i64>().ok().or_else(|| n.parse::<f64>().ok().filter(|f| f.fract() == 0.0 && f.abs() < 9e18).map(|f| f as i64)) {
                Some(i) => Ok(Some(i)),
                None => Err(format!("`{k}` must be an integer.")),
            }
        }
        Some(_) => Err(format!("`{k}` must be an integer.")),
    }
}

/// A request's number field: absent or null is None; what is not a number is refused.
fn float_field(v: &Value, k: &str) -> std::result::Result<Option<f64>, String> {
    match v.get(k) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Number(n)) => n.parse::<f64>().map(Some).map_err(|_| format!("`{k}` must be a number.")),
        Some(_) => Err(format!("`{k}` must be a number.")),
    }
}

/// The ranges and types of the sampling fields, in vLLM's words (SamplingParams' own checks).
fn check_values(v: &Value, vocab: usize) -> std::result::Result<(), String> {
    if let Some(t) = float_field(v, "temperature")? {
        if t < 0.0 || t.is_nan() {
            return Err(format!("temperature must be non-negative, got {t}."));
        }
        if t > 2.0 {
            return Err(format!("temperature must be in [0, 2], got {t}."));
        }
    }
    if let Some(p) = float_field(v, "top_p")? {
        if !(p > 0.0 && p <= 1.0) {
            return Err(format!("top_p must be in (0, 1], got {p}."));
        }
    }
    if let Some(k) = int_field(v, "top_k")? {
        if k < -1 {
            return Err(format!("top_k must be 0 (disable), or at least 1, got {k}."));
        }
    }
    if let Some(p) = float_field(v, "min_p")? {
        if !(0.0..=1.0).contains(&p) {
            return Err(format!("min_p must be in [0, 1], got {p}."));
        }
    }
    for k in ["presence_penalty", "frequency_penalty"] {
        if let Some(x) = float_field(v, k)? {
            if !(-2.0..=2.0).contains(&x) {
                return Err(format!("{k} must be in [-2, 2], got {x}."));
            }
        }
    }
    if let Some(x) = float_field(v, "repetition_penalty")? {
        if x.is_nan() || x <= 0.0 {
            return Err(format!("repetition_penalty must be greater than zero, got {x}."));
        }
    }
    let max = match (int_field(v, "max_completion_tokens")?, int_field(v, "max_tokens")?) {
        (Some(m), _) | (None, Some(m)) => Some(m),
        _ => None,
    };
    if let Some(m) = max {
        if m < 1 {
            return Err(format!("max_tokens must be at least 1, got {m}."));
        }
    }
    if let Some(m) = int_field(v, "min_tokens")? {
        if m < 0 {
            return Err(format!("min_tokens must be greater than or equal to 0, got {m}."));
        }
        if let Some(max) = max.filter(|&x| m > x) {
            return Err(format!("min_tokens must be less than or equal to max_tokens={max}, got {m}."));
        }
    }
    int_field(v, "seed")?;
    match v.get("stop") {
        None | Some(Value::Null) => {}
        Some(Value::String(s)) if !s.is_empty() => {}
        Some(Value::Array(a)) if a.iter().all(|x| matches!(x, Value::String(s) if !s.is_empty())) => {}
        Some(Value::String(_)) | Some(Value::Array(_)) => return Err("stop cannot contain an empty string.".to_string()),
        Some(_) => return Err("`stop` must be a string or a list of strings.".to_string()),
    }
    for k in ["stop_token_ids", "allowed_token_ids"] {
        match v.get(k) {
            None | Some(Value::Null) => {}
            Some(Value::Array(a)) => {
                let mut outside = Vec::new();
                for x in a {
                    match x.as_i64() {
                        Some(t) if t >= 0 && (t as usize) < vocab => {}
                        Some(t) => outside.push(t),
                        None => return Err(format!("{k} must contain only integers.")),
                    }
                }
                if !outside.is_empty() {
                    return Err(format!("token_id(s) {outside:?} in {k} contain out-of-vocab token ids. Vocabulary size: {vocab}"));
                }
                if k == "allowed_token_ids" && a.is_empty() {
                    return Err("allowed_token_ids is not None and empty!".to_string());
                }
            }
            Some(_) => return Err(format!("`{k}` must be a list of token ids.")),
        }
    }
    if let Some(e) = v.get("reasoning_effort").filter(|e| **e != Value::Null) {
        if !matches!(e.as_str(), Some("none" | "minimal" | "low" | "medium" | "high" | "xhigh" | "max")) {
            return Err("`reasoning_effort` must be one of none, minimal, low, medium, high, xhigh, max.".to_string());
        }
    }
    match v.get("documents") {
        None | Some(Value::Null) => {}
        Some(Value::Array(a)) if a.iter().all(|d| d.as_object().is_some_and(|o| o.iter().all(|(_, x)| matches!(x, Value::String(_))))) => {}
        Some(_) => return Err("`documents` must be a list of objects whose values are strings.".to_string()),
    }
    Ok(())
}

/// How many likeliest tokens a request asks each token's log-probabilities to list: a chat's `logprobs: true` with `top_logprobs`, a plain completion's `logprobs: N`, as vLLM reads them (and refuses the rest).
fn logprobs_asked(v: &Value, chat: bool, vocab: usize) -> std::result::Result<Option<usize>, String> {
    let (on, k) = if chat {
        let on = match v.get("logprobs") {
            None | Some(Value::Null) | Some(Value::Bool(false)) => false,
            Some(Value::Bool(true)) => true,
            Some(_) => return Err("`logprobs` must be true or false.".to_string()),
        };
        let top = int_field(v, "top_logprobs")?;
        if let Some(t) = top {
            if t < 0 && t != -1 {
                return Err("`top_logprobs` must be a positive value or -1.".to_string());
            }
            if (t == -1 || t > 0) && !on {
                return Err("when using `top_logprobs`, `logprobs` must be set to true.".to_string());
            }
        }
        (on, top.unwrap_or(0))
    } else {
        match int_field(v, "logprobs")? {
            None => (false, 0),
            Some(k) if k < 0 && k != -1 => return Err("`logprobs` must be a positive value or -1.".to_string()),
            Some(k) => (true, k),
        }
    };
    if !on {
        return Ok(None);
    }
    if k == -1 || k > MAX_LOGPROBS {
        return Err(format!("Requested sample logprobs of {}, which is greater than max allowed: {MAX_LOGPROBS}", if k == -1 { vocab as i64 } else { k }));
    }
    Ok(Some(k as usize))
}

fn api_of(v: &Value, chat: bool, vocab: usize) -> std::result::Result<Api, String> {
    logit_bias_of(v, vocab)?;
    let stream = v.get("stream").and_then(Value::as_bool).unwrap_or(false);
    let options = v.get("stream_options").filter(|o| !matches!(o, Value::Null) && !matches!(o, Value::Object(m) if m.is_empty())); // (null and {} are none: vLLM tests the dict's truth)
    if options.is_some() && !stream {
        return Err("Stream options can only be defined when `stream=True`.".to_string());
    }
    if options.and_then(|o| o.get("continuous_usage_stats")).and_then(Value::as_bool) == Some(true) {
        return Err("stream_options.continuous_usage_stats is not supported by this server (GLYD_ENGINE=0 runs vLLM, which has it)".to_string());
    }
    let n = int_field(v, "n")?.unwrap_or(1);
    if n < 1 {
        return Err(format!("n must be at least 1, got {n}."));
    }
    if n > MAX_ROWS as i64 {
        return Err(format!(
            "n must be at most {MAX_ROWS}, got {n}. (This engine runs up to {MAX_ROWS} sequences at once; GLYD_ENGINE=0 runs vLLM, which takes more.)"
        ));
    }
    check_values(v, vocab)?;
    if chat && v.get("echo").and_then(Value::as_bool) == Some(true) {
        return Err("echo of a chat reply is not supported by this engine yet (GLYD_ENGINE=0 runs vLLM, which has it)".to_string());
    }
    if !chat && v.get("echo").and_then(Value::as_bool) == Some(true) && logprobs_asked(v, false, vocab)?.is_some() {
        return Err(
            "logprobs with echo (the prompt's log-probabilities too) is not supported by this engine yet (GLYD_ENGINE=0 runs vLLM, which has it)".to_string()
        );
    }
    let hide_reasoning = v.get("include_reasoning").and_then(Value::as_bool) == Some(false);
    let logprobs = logprobs_asked(v, chat, vocab)?;
    Ok(Api {
        stream,
        include_usage: options.and_then(|o| o.get("include_usage")).and_then(Value::as_bool).unwrap_or(false),
        n: n as usize,
        logprobs: logprobs.filter(|_| !(chat && hide_reasoning)), // (as vLLM: a chat whose reasoning is hidden has no log-probabilities, which would spell it out token by token)
        as_ids: v.get("return_tokens_as_token_ids").and_then(Value::as_bool).unwrap_or(false),
        hide_reasoning,
        tools: if chat { tools::tools_of(v)? } else { None },
        add_generation_prompt: v.get("add_generation_prompt").and_then(Value::as_bool) != Some(false),
        add_special: v.get("add_special_tokens").and_then(Value::as_bool),
        echo: !chat && v.get("echo").and_then(Value::as_bool) == Some(true),
        request_id: v.get("request_id").and_then(Value::as_str).filter(|r| !r.is_empty()).map(str::to_string),
        total: n as usize,
        echo_texts: Vec::new(),
    })
}

/// One prompt of a request: its ids, its text (what an echo says), whether it ends inside the thinking, and the opening of the reply where it is a call that is asked for.
struct Prompt {
    ids: Vec<u32>,
    text: String,
    in_thinking: bool,
    prefix: String,
    /// Its images in order, where the model reads them (`vision::PREP`): each one's place in `ids` its tokens.
    images: Vec<Arc<crate::vision::Image>>,
}

/// The prompts of a request: a chat's one, or a plain completion's `prompt` as a string, a list of strings, a list of token ids or a list of those (each a prompt of its own, as vLLM takes them).
fn prompts_of(sh: &Shared, v: &Value, chat: bool, api: &Api, r: &Request) -> Result<Vec<Prompt>> {
    if chat {
        let (text, in_thinking, prefix, urls) = chat_text_of(sh, v, api, r)?;
        let mut ids = sh.tok.encode_with(&text, api.add_special.unwrap_or(false))?;
        // (the images read here, on the request's thread: each one's place made as long as its tokens)
        let mut images = Vec::new();
        if let (Some(prep), false) = (crate::vision::PREP.get(), urls.is_empty()) {
            images = urls.iter().map(|u| prep.image(&crate::vision::fetch(u)?).map(Arc::new)).collect::<Result<Vec<_>>>()?;
            ids = crate::vision::expand(&ids, prep.image_token, &images)?;
        }
        return Ok(vec![Prompt { ids, text, in_thinking, prefix, images }]);
    }
    let empty = || Error("Either prompt or prompt_embeds must be provided and non-empty.".into());
    let special = api.add_special.unwrap_or(true);
    let ids_of_list = |a: &[Value]| -> Result<Vec<u32>> {
        let ids: Option<Vec<u32>> = a.iter().map(|x| x.as_u64().filter(|&t| (t as usize) < sh.vocab).map(|t| t as u32)).collect();
        ids.ok_or_else(|| Error(format!("a prompt of token ids with one that is not a token of the vocabulary (size {})", sh.vocab)))
    };
    let from_text = |t: &str| -> Result<Prompt> {
        Ok(Prompt { ids: sh.tok.encode_with(t, special)?, text: t.to_string(), in_thinking: false, prefix: String::new(), images: Vec::new() })
    };
    let from_ids =
        |ids: Vec<u32>| -> Result<Prompt> { Ok(Prompt { text: sh.tok.decode(&ids)?, ids, in_thinking: false, prefix: String::new(), images: Vec::new() }) };
    let wrong = || Error("`prompt` must be a string, a list of strings, a list of token ids or a list of lists of token ids".into());
    // `suffix`: the text that comes after the completion, for an editor's fill in the middle: the prompt is the text before it, and the model's own fill-in-the-middle tokens put the two in order
    match v.get("suffix") {
        None | Some(Value::Null) => {}
        Some(Value::String(sfx)) => {
            let Some(Value::String(prefix)) = v.get("prompt") else {
                return err("`suffix` goes with a `prompt` that is one string: the text before the completion");
            };
            let Some([pre, suf, mid]) = fim_tokens(&sh.tok) else {
                return err(
                    "this model has no fill-in-the-middle tokens (<|fim_prefix|>, <fim_prefix> or <｜fim▁begin｜>), so `suffix` cannot be served by it",
                );
            };
            let text = format!("{pre}{prefix}{suf}{sfx}{mid}");
            return Ok(vec![Prompt {
                ids: sh.tok.encode_with(&text, special)?,
                text: prefix.clone(),
                in_thinking: false,
                prefix: String::new(),
                images: Vec::new(),
            }]);
        }
        Some(_) => return Err(Error("`suffix` must be a string".into())),
    }
    let prompts = match v.get("prompt") {
        Some(Value::String(t)) if !t.is_empty() => vec![from_text(t)?],
        Some(Value::Array(a)) if !a.is_empty() && a.iter().all(|x| matches!(x, Value::Number(_))) => vec![from_ids(ids_of_list(a)?)?],
        Some(Value::Array(a)) if !a.is_empty() && a.iter().all(|x| matches!(x, Value::String(_))) => {
            a.iter().filter_map(Value::as_str).map(from_text).collect::<Result<_>>()?
        }
        Some(Value::Array(a)) if !a.is_empty() && a.iter().all(|x| matches!(x, Value::Array(l) if !l.is_empty())) => {
            a.iter().filter_map(Value::as_array).map(|l| from_ids(ids_of_list(l)?)).collect::<Result<_>>()?
        }
        None | Some(Value::Null) | Some(Value::String(_)) => return Err(empty()),
        Some(Value::Array(a)) if a.is_empty() => return Err(empty()),
        _ => return Err(wrong()),
    };
    Ok(prompts)
}

/// The tokens a model writes a fill-in-the-middle prompt with, as (prefix, suffix, middle): `<prefix>TEXT BEFORE<suffix>TEXT AFTER<middle>`, the reply the text between; the first set the tokenizer has as one token each
/// (Qwen's and CodeGemma's `<|fim_prefix|>`, StarCoder's `<fim_prefix>`, DeepSeek-Coder's `<｜fim▁begin｜>`). None where it has none of them.
fn fim_tokens(tok: &Tok) -> Option<[&'static str; 3]> {
    const SETS: [[&str; 3]; 3] = [
        ["<|fim_prefix|>", "<|fim_suffix|>", "<|fim_middle|>"],
        ["<fim_prefix>", "<fim_suffix>", "<fim_middle>"],
        ["<｜fim▁begin｜>", "<｜fim▁hole｜>", "<｜fim▁end｜>"],
    ];
    let added = tok.added_ids();
    SETS.into_iter().find(|set| set.iter().all(|t| tok.encode_with(t, false).is_ok_and(|ids| ids.len() == 1 && added.contains(&ids[0]))))
}

/// The text of a chat prompt (the template's, with the tools, and a forced call's opening at its end), whether it ends inside the thinking, and that opening; and its images' URLs in order
/// where the model reads images (`vision::PREP`).
fn chat_text_of(sh: &Shared, v: &Value, api: &Api, r: &Request) -> Result<(String, bool, String, Vec<String>)> {
    let mut urls = Vec::new();
    let mut messages = tools::template_messages_with(v, crate::vision::PREP.get().map(|_| &mut urls)).map_err(Error)?;
    let standalone = STANDALONE.load(Ordering::Relaxed);
    let Some(c) = sh.chat.as_ref() else {
        // (a model with no template: lines of `role: text`)
        if api.tools.is_some() {
            return err(
                "this model has no chat template, so it has no way to be told of tools (GLYD_ENGINE=0 runs vLLM, which answers the request without them)",
            );
        }
        let mut t = messages
            .iter()
            .map(|m| format!("{}: {}", m.get("role").and_then(Value::as_str).unwrap_or(""), m.get("content").and_then(Value::as_str).unwrap_or("")))
            .collect::<Vec<_>>()
            .join("\n");
        t.push_str("\nassistant:");
        return Ok((t, false, String::new(), urls));
    };
    let mut o = r.template_options();
    o.add_generation_prompt = api.add_generation_prompt;
    let mut prefix = String::new();
    // (Kimi-K3's forced call: its opening, written after the template, by where the prompt ends)
    let mut xtml_call: Option<Option<String>> = None;
    if let Some(t) = api.tools.as_ref().filter(|t| !(standalone && t.choice == ToolChoice::None)) {
        if !(c.mentions("tools") && (c.is_xtml() || c.is_gemma() || c.mentions(Format::of(c).open()))) {
            return err("this model's chat template has no tools, or no tool-call format for them: a request with tools cannot be served here (GLYD_ENGINE=0 runs vLLM, which answers it without them)");
        }
        o.tools = Some(t.list.clone());
        // (a call required of one tool is a call of that tool: else the model may name one it was not given)
        let only = (standalone && t.choice == ToolChoice::Required && t.list.len() == 1)
            .then(|| t.list[0].get("function").and_then(|f| f.get("name")).and_then(Value::as_str))
            .flatten();
        let choice = only.map_or(t.choice.clone(), |n| ToolChoice::Named(n.to_string()));
        // (a call required of several tools opens with `<tool_call>` alone where the server runs alone: cut after `<function=`, the model named a tool it was not given, as `weather` for `get_weather`)
        if c.is_xtml() {
            xtml_call = match &choice {
                ToolChoice::Required => Some(None),
                ToolChoice::Named(n) => Some(Some(n.clone())),
                _ => None,
            };
        }
        // (Gemma 4's own: the call's mark and `call:`, and for a named function its name and the `{` of its arguments)
        let gemma = c.is_gemma().then(|| match &choice {
            ToolChoice::Required => Some("<|tool_call>call:".to_string()),
            ToolChoice::Named(n) => Some(format!("<|tool_call>call:{n}{{")),
            _ => None,
        });
        // (Muse's: the message to the function, and for a named one its ATEM block opened)
        let muse = c.is_muse().then(|| match &choice {
            ToolChoice::Required => Some(" to=".to_string()),
            ToolChoice::Named(n) => tools::forced_prefix(&choice, Format::Atem).map(|p| format!(" to={n}<|message|>{p}")),
            _ => None,
        });
        let forced = gemma.or(muse).unwrap_or_else(|| tools::forced_prefix(&choice, Format::of(c))).filter(|_| !c.is_xtml()).map(|p| {
            if standalone && choice == ToolChoice::Required {
                p.trim_end_matches(tools::FN_OPEN).to_string()
            } else {
                p
            }
        });
        if let Some(p) = forced {
            // a forced call is the reply's first words: the thinking is closed in the prompt, where the template can
            if c.mentions("enable_thinking") {
                o.kwargs.retain(|(k, _)| k != "enable_thinking");
                o.kwargs.push(("enable_thinking".to_string(), Value::Bool(false)));
            }
            prefix = p;
        }
    }
    // (a reply held to a grammar of its own, not a forced call's: it begins at once, so the thinking is closed)
    let grammar_reply = r.params.constraint.is_some() && api.tools.as_ref().map_or(true, |t| t.choice == ToolChoice::None || t.choice == ToolChoice::Auto);
    if let Some(rule) = json_rule(v) {
        if api.tools.is_some() {
            return err("response_format with tools is not supported yet: send one or the other");
        }
        // the rule goes in the system prompt (the template takes a system message first only), the thinking is closed and the answer begins with `{` (where a grammar holds it, the grammar writes the `{`)
        match messages.first_mut().filter(|m| m.get("role").and_then(Value::as_str) == Some("system")) {
            Some(Value::Object(m)) => {
                if let Some((_, Value::String(t))) = m.iter_mut().find(|(k, _)| k == "content") {
                    *t = format!("{t}\n\n{rule}");
                }
            }
            _ => messages.insert(0, tools::obj(vec![("role", Value::from("system")), ("content", Value::String(rule))])),
        }
        if !grammar_reply {
            prefix = "{".to_string();
        }
    }
    if grammar_reply || (!prefix.is_empty() && xtml_call.is_none() && api.tools.is_none()) {
        if c.mentions("enable_thinking") {
            o.kwargs.retain(|(k, _)| k != "enable_thinking");
            o.kwargs.push(("enable_thinking".to_string(), Value::Bool(false)));
        }
    }
    let mut text = c.render_json(&Value::Array(messages), &o)?;
    let mut in_thinking = text.trim_end().ends_with("<think>");
    // (a reply that must begin with a call or a JSON object, where the template has no switch for the thinking and opens it, as GLM-5.3-Flash's: closed
    // empty, as the template writes a turn without reasoning)
    if in_thinking && (!prefix.is_empty() || grammar_reply) && xtml_call.is_none() {
        text.push_str("</think>");
        in_thinking = false;
    }
    if let Some(name) = xtml_call {
        // (the think channel closed empty where the prompt opened it, else the response the prompt opened)
        let in_think = text.ends_with(crate::conv::XTML_THINK);
        if !in_think && !text.ends_with("<|open|>response<|sep|>") {
            return err("this model's prompt does not end where its reply begins, so a call cannot be required of it");
        }
        prefix = crate::xtml::forced(in_think, name.as_deref());
    }
    text.push_str(&prefix);
    Ok((text, in_thinking, prefix, urls))
}

/// A request's cache scope: the digest of its `cache_salt` (vLLM's field: a string a caller, or the gateway before the callers, gives; the same string, the same scope), 0
/// where it has none. Never 0 for a salt, so no salt is the scope of the requests that have none.
fn salt_of(v: &Value) -> u64 {
    match v.get("cache_salt").and_then(Value::as_str).filter(|s| !s.is_empty()) {
        Some(s) => u64::from_le_bytes(key_digest(s)[..8].try_into().expect("8 bytes")) | 1,
        None => 0,
    }
}

/// A sequence's handle as its request may read it: what the sequence found in the cache is told (`Handle::cached`, for the usage) to a request with a cache scope of its
/// own, which can have found only what its scope left. A request with none is told nothing (0, as before these counts were kept) unless the operator has said that every
/// caller is one user (`Context::tell_cached`): the cache serves it all the same, from what any such request left, and a count would let one caller learn how another's
/// prompt begins (docs/serve-api.md).
fn told(sh: &Shared, salt: u64, mut h: Handle) -> Handle {
    if salt == 0 && !sh.rt.tell_cached.load(Ordering::Relaxed) {
        h.cached = Arc::new(AtomicU32::new(0));
    }
    h
}

/// The prompt tokens a request's sequences found in the cache (`Handle::cached` of each prompt's first choice), read when they have ended.
fn cached_tokens(found: &[Arc<AtomicU32>]) -> usize {
    found.iter().map(|c| c.load(Ordering::Relaxed) as usize).sum()
}

/// A choice as a reply that is not streamed has it.
#[derive(Default)]
struct Whole {
    lps: Vec<Lp>,
    ended: Option<(Done, String)>,
}

/// What a choice's thread tells the request's: what its text adds, and how it ended (with its thinking where the reply is not streamed), or why it could not.
enum Out {
    Delta(usize, Delta),
    Done {
        i: usize,
        last: Delta,
        done: Done,
        thinking: Option<String>,
    },
    /// The request cannot go on: why, and whether it is the server's doing (the model's step failed, the engine stopped) and not the request's own (a prompt the cache does not hold).
    Failed(String, bool),
}

/// The sequence of `h` asked to end, and waited for until the engine has let it go (its end comes last; a closed channel is an engine that has stopped): what it held is the engine's to use again
/// when this returns.
fn settle(h: &Handle) {
    h.abort.store(true, Ordering::Relaxed);
    while let Ok(ev) = h.events.recv() {
        if matches!(ev, Ev::End(_)) {
            break;
        }
    }
}

/// A choice's thread: the events of its sequence become text and log-probabilities for the request's thread. `first` is told when the first event has come (the others of the request wait for it: the prompt's blocks
/// are then in the cache for them to find). It returns when the engine has let the sequence go: a choice that ends before the engine ends it (a stop string, its request gone) waits for that (`settle`).
fn drive(i: usize, h: Handle, mut c: Choice, tok: &Tok, whole: bool, out: Sender<Out>, mut first: Option<Sender<()>>) {
    loop {
        let ev = h.events.recv();
        if let Some(f) = first.take() {
            let _ = f.send(());
        }
        let ended = matches!(ev, Ok(Ev::End(_)) | Err(_));
        let step: std::result::Result<Option<Finish>, (String, bool)> = match ev {
            Ok(Ev::Lp(lp)) => {
                c.lp(lp);
                Ok(None)
            }
            Ok(Ev::Token(t)) => match c.token(tok, t) {
                Ok(d) => {
                    if !d.is_empty() && out.send(Out::Delta(i, d)).is_err() {
                        return settle(&h);
                    }
                    if c.stopped() {
                        h.abort.store(true, Ordering::Relaxed);
                        Ok(Some(Finish::Stopped))
                    } else {
                        Ok(None)
                    }
                }
                Err(e) => Err((e.to_string(), false)),
            },
            Ok(Ev::End(Ok(f))) => Ok(Some(f)),
            Ok(Ev::End(Err(e))) => Err((e.msg, e.kind != crate::ffi::Kind::Request)),
            Err(_) => Err(("the engine stopped".to_string(), true)),
        };
        match step {
            Ok(None) => continue,
            Ok(Some(f)) => {
                let _ = match c.end(tok, f) {
                    Ok((last, done)) => out.send(Out::Done { i, last, done, thinking: whole.then(|| c.reasoning.clone()) }),
                    Err(e) => out.send(Out::Failed(e.to_string(), false)),
                };
            }
            Err((e, server)) => {
                let _ = out.send(Out::Failed(e, server));
            }
        }
        if !ended {
            settle(&h);
        }
        return;
    }
}

/// The fields of a chat delta: the thinking (as `reasoning_content`, and `reasoning` as vLLM 0.30 names it), the answer, and the tool calls begun or continued (a call's function with its id, or more of its arguments).
fn delta_fields(d: &Delta) -> String {
    let mut f = Vec::new();
    if !d.reasoning.is_empty() {
        f.push(format!("\"reasoning_content\":{0},\"reasoning\":{0}", jstr(&d.reasoning)));
    }
    if !d.content.is_empty() {
        f.push(format!("\"content\":{}", jstr(&d.content)));
    }
    if !d.calls.is_empty() {
        let calls: Vec<String> = d
            .calls
            .iter()
            .filter_map(|c| match c {
                ToolEv::Start { index, id, name } => {
                    Some(format!("{{\"index\":{index},\"id\":{},\"type\":\"function\",\"function\":{{\"name\":{}}}}}", jstr(id), jstr(name)))
                }
                ToolEv::Args { index, args } => Some(format!("{{\"index\":{index},\"function\":{{\"arguments\":{}}}}}", jstr(args))),
                ToolEv::Content(_) => None,
            })
            .collect();
        f.push(format!("\"tool_calls\":[{}]", calls.join(",")));
    }
    f.join(",")
}

/// A chat request, in OpenAI's chat form, made ready to run as one sequence on the engine: what the other APIs' handlers (a request of theirs is turned into this form first) share with the completions' one,
/// its refusals, its sampling, its grammar (`constraint_of`), its prompt, its tools and its window. `hide_reasoning`: the reply's thinking is left out of what is sent (an API with no place for it).
struct Started {
    h: Handle,
    cfg: Cfg,
    prompt_tokens: usize,
    api: Api,
}

/// What a request is refused with: the status and the words.
type Refused = (u16, String);

/// A chat request read and its prompt made, before anything runs (a count of the prompt's tokens stops here).
struct Prepared {
    pr: Prompt,
    r: Request,
    api: Api,
}

fn prepare(sh: &Shared, o: &Value) -> std::result::Result<Prepared, Refused> {
    prepare_as(sh, o, true)
}

/// `prepare` for a chat (messages) or, where `chat` is false, a plain prompt (`prompt`, and `suffix` for a fill in the middle).
fn prepare_as(sh: &Shared, o: &Value, chat: bool) -> std::result::Result<Prepared, Refused> {
    let bad = |m: String| (400u16, m);
    let not_here = || sh.refused.iter().find(|(n, d)| asked(o, n, d)).map(|(n, _)| format!("{n} is not supported by the engine this model runs on yet"));
    if let Some(why) = unsupported(o).or_else(not_here).or_else(|| invalid(o, chat, sh.vocab)) {
        return Err(bad(why));
    }
    let mut r = request_of(o, sh.window, &sh.defaults);
    r.thinking = r.thinking.or(sh.thinking);
    r.params.logit_bias = logit_bias_of(o, sh.vocab).unwrap_or_default();
    let api = api_of(o, chat, sh.vocab).map_err(bad)?;
    r.params.constraint = constraint_of(sh, o, &api).map_err(bad)?;
    if api.n > 1 {
        return Err(bad("n must be 1: this API makes one choice".to_string()));
    }
    let pr = prompts_of(sh, o, chat, &api, &r).map_err(|e| bad(e.to_string()))?.remove(0);
    if pr.ids.len() + 1 > sh.window {
        return Err(bad(format!(
            "This prompt has {} tokens, longer than the model's context of {} tokens: maximum context length (a longer one: --context N)",
            pr.ids.len(),
            sh.window
        )));
    }
    Ok(Prepared { pr, r, api })
}

fn start_chat(sh: &Shared, tx: &Sender<Job>, o: &Value, hide_reasoning: bool) -> std::result::Result<Started, Refused> {
    start_as(sh, tx, o, true, hide_reasoning)
}

/// `start_chat` for a chat or, where `chat` is false, a plain prompt.
fn start_as(sh: &Shared, tx: &Sender<Job>, o: &Value, chat: bool, hide_reasoning: bool) -> std::result::Result<Started, Refused> {
    let Prepared { pr, r, api } = prepare_as(sh, o, chat)?;
    let reading = api.tools.as_ref().filter(|t| t.choice != ToolChoice::None);
    let tools_cfg = reading.map(|t| ToolsCfg {
        choice: t.choice.clone(),
        parallel: t.parallel,
        format: sh.chat.as_ref().map_or(Format::Hermes, Format::of),
        list: t.list.clone(),
        prefix: pr.prefix.clone(),
        stream: api.stream,
    });
    let reader_kind = sh.chat.as_ref().and_then(|c| crate::reader::ReaderKind::of(c, &pr.text));
    let cfg = Cfg {
        chat,
        in_thinking: pr.in_thinking,
        stops: r.stop.clone(),
        include_stop: r.include_stop,
        skip_special: r.skip_special
            && reading.is_none()
            && !matches!(reader_kind, Some(crate::reader::ReaderKind::Gemma { .. } | crate::reader::ReaderKind::Muse)),
        hide_reasoning: hide_reasoning || api.hide_reasoning,
        tools: tools_cfg,
        lead: if reading.is_none() { pr.prefix.clone() } else { String::new() },
        reader: reader_kind,
    };
    let reply = Reply {
        params: r.params.clone(),
        max_new: r.max_new.min(sh.window - pr.ids.len()).max(1),
        ignore_eos: r.ignore_eos,
        logprobs: None,
        stop_ids: r.stop_ids.clone(),
    };
    let salt = salt_of(o);
    let h = (Remote { tx: tx.clone(), window: sh.window }).start_in(&pr.ids, pr.images.clone(), &reply, salt).map_err(|e| (500, e.to_string()))?;
    Ok(Started { h: told(sh, salt, h), cfg, prompt_tokens: pr.ids.len(), api })
}

/// One completions request on `s`: chat (messages) or plain (a prompt); or, at /v1/glyd/tokenize, the count of its prompt's tokens alone (a chat's where it has messages), made as its reply's would be.
fn handle_completion(sh: &Shared, tx: &Sender<Job>, s: &mut dyn Conn, req: &Req, chat: bool, id: u64) {
    let bad = |s: &mut dyn Conn, msg: &str| respond_with(s, 400, "application/json", &req.cors, error_body(msg, 400).as_bytes());
    let v = match std::str::from_utf8(&req.body).ok().and_then(|t| json::parse(t).ok()) {
        Some(v) => v,
        None => return bad(s, "the body is not JSON"),
    };
    let count = req.path.split('?').next() == Some("/v1/glyd/tokenize");
    let chat = if count { v.get("messages").is_some() } else { chat };
    warn_unknown(&v, chat);
    let not_here = || sh.refused.iter().find(|(n, d)| asked(&v, n, d)).map(|(n, _)| format!("{n} is not supported by the engine this model runs on yet"));
    if let Some(why) = unsupported(&v).or_else(not_here).or_else(|| invalid(&v, chat, sh.vocab)) {
        return bad(s, &why);
    }
    let mut r = request_of(&v, sh.window, &sh.defaults);
    r.thinking = r.thinking.or(sh.thinking);
    // (the whole row is read for a draw with a bias: its tokens are the vocabulary's, which the engine's logits are as wide as)
    r.params.logit_bias = logit_bias_of(&v, sh.vocab).unwrap_or_default();
    if !chat && int_field(&v, "max_tokens").ok().flatten().is_none() {
        r.max_new = 16; // (a plain completion's default, the legacy API's and vLLM's)
    }
    let mut api = match api_of(&v, chat, sh.vocab) {
        Ok(a) => a,
        Err(m) => return bad(s, &m),
    };
    match constraint_of(sh, &v, &api) {
        Ok(c) => r.params.constraint = c,
        Err(m) => return bad(s, &m),
    }
    if api.n > 1 && r.params.temperature <= 0.0 {
        return bad(s, &format!("n must be 1 when using greedy sampling, got {}.", api.n));
    }
    let prompts = match prompts_of(sh, &v, chat, &api, &r) {
        Ok(p) => p,
        Err(e) => return bad(s, &e.to_string()),
    };
    if count {
        let n: usize = prompts.iter().map(|p| p.ids.len()).sum();
        return respond_with(s, 200, "application/json", &req.cors, format!("{{\"prompt_tokens\":{n},\"max_model_len\":{}}}", sh.window).as_bytes());
    }
    if let Some(p) = prompts.iter().find(|p| p.ids.len() + 1 > sh.window) {
        return bad(
            s,
            &format!(
                "This prompt has {} tokens, longer than the model's context of {} tokens: maximum context length (a longer one: --context N)",
                p.ids.len(),
                sh.window
            ),
        );
    }
    // a length asked for that the context has no room for after the prompt is refused, not cut short (a request that asks for none has the room there is)
    if let Some(m) = int_field(&v, "max_completion_tokens").ok().flatten().or(int_field(&v, "max_tokens").ok().flatten()) {
        if let Some(p) = prompts.iter().find(|p| p.ids.len() + m as usize > sh.window) {
            return bad(s, &format!("This prompt has {} tokens and asks for up to {m} more, {} in all: longer than the model's context of {} tokens: maximum context length (a longer one: --context N)", p.ids.len(), p.ids.len() + m as usize, sh.window));
        }
    }
    api.total = prompts.len() * api.n;
    if api.total > MAX_CHOICES {
        return bad(
            s,
            &format!("{} prompts of {} choices each are {} sequences: this engine takes up to {MAX_CHOICES} in a request", prompts.len(), api.n, api.total),
        );
    }
    if api.echo {
        api.echo_texts = (0..api.total).map(|i| prompts[i / api.n].text.clone()).collect();
    }
    let prompt_tokens: usize = prompts.iter().map(|p| p.ids.len()).sum();
    let remote = Remote { tx: tx.clone(), window: sh.window };
    let ident = format!("{}-{}", if chat { "chatcmpl" } else { "cmpl" }, api.request_id.clone().unwrap_or_else(|| id.to_string()));
    if !track_begin(&ident) {
        return bad(s, &format!("a request with the id {ident} is running: an id names one request at a time"));
    }
    let created = now();
    let (out_tx, out_rx) = std::sync::mpsc::channel::<Out>();
    let (first_tx, first_rx) = std::sync::mpsc::channel::<()>();
    let salt = salt_of(&v);
    let mut aborts: Vec<Arc<AtomicBool>> = Vec::new();
    // (each prompt's first choice: what it found in the cache is what the request's prompt tokens did not cost; the other choices of a prompt find the first one's)
    let mut found: Vec<Arc<AtomicU32>> = Vec::new();
    let mut failure: Option<String> = None;
    // (whether the failure is the server's doing: answered 500, where a request's own is 400)
    let mut server_fault = false;
    let (name, tok) = (sh.name.as_str(), &sh.tok);
    // the whole of a reply, where it is not streamed: each choice's log-probabilities, and how it ended with its thinking and its text
    let mut acc: Vec<Whole> = (0..api.total).map(|_| Whole::default()).collect();
    let (kind, stream_kind) = if chat { ("chat.completion", "chat.completion.chunk") } else { ("text_completion", "text_completion") };
    let head = format!("{{\"id\":\"{ident}\",\"object\":\"{stream_kind}\",\"created\":{created},\"model\":{}", jstr(name));
    let (mut started, mut gone) = (false, false);
    let mut offsets = vec![0usize; api.total];
    let mut completion_tokens = 0usize;
    // a tool is read for calls where there are tools and the choice is not none; a call is one of the model's own words, the special tokens kept
    let reading = api.tools.as_ref().filter(|t| t.choice != ToolChoice::None);
    std::thread::scope(|sc| {
        'prompts: for (pi, pr) in prompts.iter().enumerate() {
            let max_new = r.max_new.min(sh.window - pr.ids.len()).max(1);
            let tools_cfg = reading.map(|t| ToolsCfg {
                choice: t.choice.clone(),
                parallel: t.parallel,
                format: sh.chat.as_ref().map_or(Format::Hermes, Format::of),
                list: t.list.clone(),
                prefix: pr.prefix.clone(),
                stream: api.stream,
            });
            let lead = if reading.is_none() { pr.prefix.clone() } else { String::new() };
            let reader_kind = sh.chat.as_ref().and_then(|c| crate::reader::ReaderKind::of(c, &pr.text));
            let cfg = Cfg {
                chat,
                in_thinking: pr.in_thinking,
                stops: r.stop.clone(),
                include_stop: r.include_stop,
                skip_special: r.skip_special
                    && reading.is_none()
                    && !matches!(reader_kind, Some(crate::reader::ReaderKind::Gemma { .. } | crate::reader::ReaderKind::Muse)),
                hide_reasoning: api.hide_reasoning,
                tools: tools_cfg,
                lead,
                reader: reader_kind,
            };
            for i in 0..api.n {
                let idx = pi * api.n + i;
                let mut reply = Reply { params: r.params.clone(), max_new, ignore_eos: r.ignore_eos, logprobs: api.logprobs, stop_ids: r.stop_ids.clone() };
                reply.params.seed = r.params.seed.wrapping_add(i as u64);
                let h = match remote.start_in(&pr.ids, pr.images.clone(), &reply, salt) {
                    Ok(h) => told(sh, salt, h),
                    Err(e) => {
                        failure = Some(e.to_string());
                        break 'prompts;
                    }
                };
                aborts.push(h.abort.clone());
                if i == 0 {
                    found.push(h.cached.clone());
                }
                track_seq(&ident, &h.abort);
                let (c, o, whole) = (Choice::new(cfg.clone()), out_tx.clone(), !api.stream);
                let announce = (i == 0 && api.n > 1).then(|| first_tx.clone());
                sc.spawn(move || drive(idx, h, c, tok, whole, o, announce));
                if i == 0 && api.n > 1 {
                    // (the others begin as this one does: they wait until its prompt has run and its blocks are in the cache)
                    let _ = first_rx.recv();
                }
            }
        }
        drop(out_tx);
        let mut ended = 0;
        while ended < aborts.len() && failure.is_none() {
            let out = match out_rx.recv_timeout(WAIT_TICK) {
                Ok(out) => out,
                Err(RecvTimeoutError::Timeout) => {
                    // (nothing to send yet: the request waits for its turn or its prompt is read. A streamed reply's headers go out, then a comment each tick, so a client's read timeout does
                    // not fire while it waits; a client that has gone ends the request before its prompt runs)
                    if api.stream {
                        gone = !if started {
                            sse_comment(s)
                        } else {
                            started = true;
                            start_stream(s, &head, chat, &api, &req.cors)
                        };
                    }
                    gone = gone || client_gone(s);
                    if gone {
                        break;
                    }
                    continue;
                }
                Err(RecvTimeoutError::Disconnected) => break,
            };
            match out {
                Out::Failed(e, server) => {
                    failure = Some(e);
                    server_fault = server;
                }
                Out::Delta(i, d) => {
                    if !api.stream {
                        acc[i].lps.extend(d.lps.iter().cloned());
                    }
                    if api.stream && !gone {
                        let (lp, text) = (&d.lps, delta_fields(&d));
                        gone = !stream_chunk(s, &req.cors, &mut started, &head, chat, &api, tok, i, &text, lp, &mut offsets[i], None);
                    }
                }
                Out::Done { i, last, done, thinking } => {
                    ended += 1;
                    completion_tokens += done.tokens;
                    if !api.stream {
                        acc[i].lps.extend(last.lps.iter().cloned());
                    }
                    if api.stream && !gone {
                        let text = delta_fields(&last);
                        gone =
                            !stream_chunk(s, &req.cors, &mut started, &head, chat, &api, tok, i, &text, &last.lps, &mut offsets[i], Some(done.finish_reason));
                    }
                    if let Some(reasoning) = thinking {
                        acc[i].ended = Some((done, reasoning));
                    }
                }
            }
            if gone {
                break;
            }
        }
        if gone {
            track_cancel(&ident, "client gone");
        }
        if gone || failure.is_some() {
            for a in &aborts {
                a.store(true, Ordering::Relaxed);
            }
        }
        drop(out_rx);
    });
    // (each choice's thread has seen its sequence end: the engine has let go of them all)
    track_end(&ident, failure.as_deref());
    if let Some(msg) = failure {
        eprintln!("glyd: a request failed: {msg}");
        return if started {
            let _ = sse(s, &format!("{{\"error\":{{\"message\":{},\"type\":\"server_error\"}}}}", jstr(&msg)));
            let _ = sse(s, "[DONE]");
            let _ = s.write_all(b"0\r\n\r\n");
        } else if server_fault {
            respond_with(s, 500, "application/json", &req.cors, error_body(&msg, 500).as_bytes())
        } else {
            bad(s, &msg)
        };
    }
    if gone {
        return;
    }
    if api.stream {
        if !started {
            start_stream(s, &head, chat, &api, &req.cors);
        }
        if api.include_usage {
            let _ = sse(s, &format!("{head},\"choices\":[],\"usage\":{}}}", wire::usage_with(prompt_tokens, completion_tokens, cached_tokens(&found))));
        }
        let _ = sse(s, "[DONE]");
        let _ = s.write_all(b"0\r\n\r\n");
        return;
    }
    let choices: Vec<String> = acc
        .iter()
        .enumerate()
        .map(|(i, w)| {
            let (lps, (done, reasoning)) = (&w.lps, w.ended.as_ref().expect("a choice that ended"));
            let logprobs = if api.logprobs.is_none() {
                "null".to_string()
            } else if chat {
                wire::chat_logprobs(tok, lps, api.as_ids)
            } else {
                wire::text_logprobs(tok, lps, api.as_ids, &mut 0)
            };
            if chat {
                let think = if reasoning.is_empty() || api.hide_reasoning { String::new() } else { format!(",\"reasoning_content\":{0},\"reasoning\":{0}", jstr(reasoning)) };
                let calls = if done.calls.is_empty() {
                    String::new()
                } else {
                    let list: Vec<String> = done.calls.iter().map(|c| format!("{{\"id\":{},\"type\":\"function\",\"function\":{{\"name\":{},\"arguments\":{}}}}}", jstr(&tools::new_id()), jstr(&c.name), jstr(&c.arguments))).collect();
                    format!(",\"tool_calls\":[{}]", list.join(","))
                };
                let content = done.content.as_deref().map_or("null".to_string(), jstr);
                format!("{{\"index\":{i},\"message\":{{\"role\":\"assistant\",\"content\":{content}{think}{calls}}},\"logprobs\":{logprobs},\"finish_reason\":\"{}\"}}", done.finish_reason)
            } else {
                let text = format!("{}{}", api.echo_texts.get(i).map_or("", String::as_str), done.content.as_deref().unwrap_or(""));
                format!("{{\"index\":{i},\"text\":{},\"logprobs\":{logprobs},\"finish_reason\":\"{}\"}}", jstr(&text), done.finish_reason)
            }
        })
        .collect();
    let body = format!(
        "{{\"id\":\"{ident}\",\"object\":\"{kind}\",\"created\":{created},\"model\":{},\"choices\":[{}],\"usage\":{}}}",
        jstr(name),
        choices.join(","),
        wire::usage_with(prompt_tokens, completion_tokens, cached_tokens(&found))
    );
    respond_with(s, 200, "application/json", &req.cors, body.as_bytes());
}

/// The fields a request may carry that this server reads (a value of some is refused, in words: `unsupported`, `invalid`), with NO_EFFECT's, NOT_DONE's and CONSTRAINED's the ones it knows: another is said once on the
/// program's standard error where it comes, not taken for something it is not.
const DONE: &[&str] = &[
    "messages",
    "prompt",
    "model",
    "frequency_penalty",
    "logit_bias",
    "logprobs",
    "top_logprobs",
    "max_tokens",
    "max_completion_tokens",
    "n",
    "presence_penalty",
    "response_format",
    "structured_outputs",
    "guided_json",
    "guided_regex",
    "guided_choice",
    "guided_grammar",
    "guided_decoding_backend",
    "seed",
    "stop",
    "stream",
    "stream_options",
    "temperature",
    "top_p",
    "tools",
    "tool_choice",
    "reasoning_effort",
    "include_reasoning",
    "parallel_tool_calls",
    "top_k",
    "min_p",
    "repetition_penalty",
    "stop_token_ids",
    "include_stop_str_in_output",
    "ignore_eos",
    "min_tokens",
    "skip_special_tokens",
    "allowed_token_ids",
    "echo",
    "add_generation_prompt",
    "add_special_tokens",
    "documents",
    "chat_template_kwargs",
    "request_id",
    "return_tokens_as_token_ids",
    "best_of",
    "suffix",
    "cache_salt",
];

/// The fields that change nothing on a server of one model on this computer.
const NO_EFFECT: &[&str] = &[
    "user",
    "watermarking",
    "length_penalty",
    "spaces_between_special_tokens",
    "session_id",
    "store",
    "metadata",
    "safety_identifier",
    "prompt_cache_key",
    "verbosity",
    "modalities",
    "routed_experts_prompt_start",
];

/// A field of a request that the server does not know, said once.
fn warn_unknown(v: &Value, chat: bool) {
    static SAID: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());
    for (k, _) in v.as_object().unwrap_or(&[]) {
        if DONE.contains(&k.as_str()) || NO_EFFECT.contains(&k.as_str()) || NOT_DONE.iter().any(|(n, _)| n == k) || CONSTRAINED.contains(&k.as_str()) {
            continue;
        }
        if let Ok(mut said) = SAID.lock() {
            if said.len() < 64 && !said.contains(k) {
                // (a client that sends names without end is told of the first few: the list and its search stay small)
                said.push(k.clone());
                eprintln!("glyd: a {} request has a field this server does not know, {k:?}: it has no effect", if chat { "chat" } else { "completions" });
            }
        }
    }
}

/// The headers of a streamed reply, and (a chat's) the first chunk of every choice, which carries the role.
fn start_stream(s: &mut dyn Conn, head: &str, chat: bool, api: &Api, cors: &str) -> bool {
    let h = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-store\r\n{cors}Transfer-Encoding: chunked\r\nConnection: close\r\n\r\n"
    );
    if s.write_all(h.as_bytes()).is_err() {
        return false;
    }
    // (OpenAI's chunks carry `"usage": null` where the stream's last chunk has the usage: `stream_options.include_usage`)
    let usage = if api.include_usage { ",\"usage\":null" } else { "" };
    for i in 0..api.total {
        let first = if chat {
            format!("{head},\"choices\":[{{\"index\":{i},\"delta\":{{\"role\":\"assistant\",\"content\":\"\"}},\"logprobs\":null,\"finish_reason\":null}}]{usage}}}")
        } else if let Some(t) = api.echo_texts.get(i) {
            // (a plain completion that echoes its prompt: the first chunk of each choice is the prompt)
            format!("{head},\"choices\":[{{\"index\":{i},\"text\":{},\"logprobs\":null,\"finish_reason\":null}}]{usage}}}", jstr(t))
        } else {
            continue;
        };
        if sse(s, &first).is_err() {
            return false;
        }
    }
    true
}

/// A chunk of a streamed reply for choice `i`: its delta (`fields`, the inside of a chat's delta object, or a plain completion's text) with the log-probabilities that go with it, and the finish reason where it ends.
/// False where the client has gone.
#[allow(clippy::too_many_arguments)]
fn stream_chunk(
    s: &mut dyn Conn,
    cors: &str,
    started: &mut bool,
    head: &str,
    chat: bool,
    api: &Api,
    tok: &Tok,
    i: usize,
    fields: &str,
    lps: &[Lp],
    offset: &mut usize,
    finish: Option<&str>,
) -> bool {
    if !*started {
        *started = true;
        if !start_stream(s, head, chat, api, cors) {
            return false;
        }
    }
    let logprobs = if api.logprobs.is_none() || (lps.is_empty() && finish.is_none()) {
        "null".to_string()
    } else if chat {
        wire::chat_logprobs(tok, lps, api.as_ids)
    } else {
        wire::text_logprobs(tok, lps, api.as_ids, offset)
    };
    let finish = finish.map_or("null".to_string(), |f| format!("\"{f}\""));
    let choice = if chat {
        format!("{{\"index\":{i},\"delta\":{{{fields}}},\"logprobs\":{logprobs},\"finish_reason\":{finish}}}")
    } else {
        // (a plain completion's text is in `fields` as the chat's content key: its value is the text)
        let text = fields.strip_prefix("\"content\":").unwrap_or("\"\"");
        format!("{{\"index\":{i},\"text\":{text},\"logprobs\":{logprobs},\"finish_reason\":{finish}}}")
    };
    sse(s, &format!("{head},\"choices\":[{choice}]{}}}", if api.include_usage { ",\"usage\":null" } else { "" })).is_ok()
}

/// How long a request's thread waits for its reply's next piece before it looks at its client.
const WAIT_TICK: Duration = Duration::from_secs(1);

/// A comment of a server-sent stream (clients skip it): a stream that waits shows it is alive.
fn sse_comment(s: &mut dyn Conn) -> bool {
    s.write_all(b"b\r\n: waiting\n\n\r\n").and_then(|_| s.flush()).is_ok()
}

/// Whether the client of a connection has closed it (a request that waited in line for a reply nobody reads).
fn client_gone(s: &dyn Conn) -> bool {
    s.gone()
}

/// A server-sent event as one chunk of the chunked body.
fn sse(s: &mut dyn Conn, data: &str) -> std::io::Result<()> {
    let ev = format!("data: {data}\n\n");
    // one write for the chunk (its size, the event, the line end): three small ones would wait on each other's acknowledgements
    s.write_all(format!("{:x}\r\n{ev}\r\n", ev.len()).as_bytes())?;
    s.flush()
}

/// Listens on `addr` (host:port): a thread accepts, a thread of its own reads each connection. The page, the health check and the model list are answered there; a completions request is
/// sent as a `Job` to `tx`. Returns the address it is listening on.
pub fn listen(addr: &str, name: &str, window: usize, key: Option<String>, local_only: bool, tx: Sender<Job>) -> Result<std::net::SocketAddr> {
    listen_loading(addr, name, window, key, local_only, tx, Arc::new(AtomicBool::new(false)))
}

/// `listen` for a server that listens while its model loads (qwenserve): `/health` answers 503 `{"status":"loading"}` for as long as `loading` is set (the program clears it when the model can serve), so
/// a script that waits for a 200 sends its first request after the load and not before. Nothing else answers differently: the model list, the page and the completions requests (which wait for their turn) are as they were.
pub fn listen_loading(
    addr: &str,
    name: &str,
    window: usize,
    key: Option<String>,
    local_only: bool,
    tx: Sender<Job>,
    loading: Arc<AtomicBool>,
) -> Result<std::net::SocketAddr> {
    listen_with_life(addr, name, window, key, local_only, tx, loading, crate::ops::Life::global())
}

/// `listen_loading` with the life (draining and the requests in flight) it answers to: the process's own for a server, a test's own for a test.
#[allow(clippy::too_many_arguments)]
pub fn listen_with_life(
    addr: &str,
    name: &str,
    window: usize,
    key: Option<String>,
    local_only: bool,
    tx: Sender<Job>,
    loading: Arc<AtomicBool>,
    life: Arc<crate::ops::Life>,
) -> Result<std::net::SocketAddr> {
    listen_in(addr, name, window, key, local_only, tx, loading, life, crate::runtime::context::Context::process().clone())
}

/// `listen_with_life` for the model of the runtime context `rt`: the process's own for a server of one model, its own for each model of a process that serves several (and for a test's).
#[allow(clippy::too_many_arguments)]
pub fn listen_in(
    addr: &str,
    name: &str,
    window: usize,
    key: Option<String>,
    local_only: bool,
    tx: Sender<Job>,
    loading: Arc<AtomicBool>,
    life: Arc<crate::ops::Life>,
    rt: Arc<crate::runtime::context::Context>,
) -> Result<std::net::SocketAddr> {
    crate::ops::init();
    crate::otel::set_model(name);
    let key = Arc::new(key.map(|k| key_digest(&k)));
    let name = Arc::new(name.to_string());
    // (the connections are the front end's: tasks of its own runtime, with no cap on how many; what answers a request runs on threads, as before)
    let bound = frontend::serve(addr, move |port| Ctx { name, window, key, local_only, port, tx, loading, life, times: Times::default(), rt })
        .map_err(|e| Error(format!("{addr}: {e}")))?;
    Ok(bound)
}

/// A refusal of the Host or the Origin as the Python server's was: a JSON error of type `Forbidden`.
fn forbidden_body(why: &str, code: u16) -> String {
    format!("{{\"error\":{{\"message\":{},\"type\":\"Forbidden\",\"code\":{code}}}}}", jstr(why))
}

/// One connection's request. The checks come in the order of the Python server's layers: the Host and the Origin where the server is on this computer's own address (everything is refused for them, the page too), a CORS
/// preflight, the key for the API.
/// What the server answers a request under: its model's name and window, its key, whether it is on its own address only, the engine's queue, and its life.
#[derive(Clone)]
pub(crate) struct Ctx {
    pub name: Arc<String>,
    pub window: usize,
    pub key: Arc<Option<[u8; 32]>>,
    pub local_only: bool,
    pub port: u16,
    pub tx: Sender<Job>,
    pub loading: Arc<AtomicBool>,
    pub life: Arc<crate::ops::Life>,
    pub times: Times,
    /// The model's runtime context: what /v1/glyd/info, /metrics and /stats read, and where the requests are counted.
    pub rt: Arc<crate::runtime::context::Context>,
}

/// The times the front end gives a connection: for the head of a request (of the whole of it), for its body, and for the next request of a connection that has been answered.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Times {
    pub head: Duration,
    pub body: Duration,
    pub idle: Duration,
}

impl Default for Times {
    fn default() -> Times {
        Times { head: HEAD_TIMEOUT, body: BODY_TIMEOUT, idle: frontend::IDLE }
    }
}

/// The job sent to the engine's thread, and where it fails to go (the engine is not running) the connection given back, to say so.
fn send_job(tx: &Sender<Job>, job: Job) -> std::result::Result<(), Box<dyn Conn>> {
    tx.send(job).map_err(|e| match e.0 {
        Job::Http { s, .. } | Job::Messages { s, .. } | Job::Api { s, .. } => s,
        _ => unreachable!("only a request with a connection is sent here"),
    })
}

/// A request that was read, answered: the checks come in the order of the Python server's layers (the Host and the Origin where the server is on this computer's own address, everything is
/// refused for them, a preflight too; the key for the API), then the request is answered here (the page, the health checks, the model list) or sent to the engine's thread with its connection.
pub(crate) fn dispatch(mut s: Box<dyn Conn>, mut req: Req, ctx: &Ctx, id: u64) {
    let (name, window, key, (local_only, port), tx, loading, life) =
        (ctx.name.as_str(), ctx.window, &*ctx.key, (ctx.local_only, ctx.port), &ctx.tx, &*ctx.loading, &ctx.life);
    if let Some((code, why)) = refusal(&req.host, &req.origin, local_only) {
        ctx.rt.figures.refused(code);
        return respond(&mut *s, code, "application/json", forbidden_body(&why, code).as_bytes());
    }
    if let Some((method, headers)) = &req.preflight {
        return preflight(&mut *s, local_only, port, &req.origin, method, headers.as_deref());
    }
    req.cors = cors_headers(local_only, port, &req.origin);
    let path = req.path.split('?').next().unwrap_or("").to_string();
    let cors = req.cors.clone();
    // the API wants the key where there is one (the page and the health check do not; nor does an OPTIONS)
    if (path.starts_with("/v1") || path.starts_with("/api/") || path == "/metrics")
        && req.method != "OPTIONS"
        && key.as_ref().is_some_and(|k| !key_matches(k, req.auth.as_deref()))
    {
        // (the Messages API's clients, the Anthropic SDKs and Claude Code, read Anthropic's error: it says `invalid x-api-key`, and the key is `x-api-key` or a bearer token)
        let body = if path.starts_with("/v1/messages") {
            anthropic::anthropic_body(401, "invalid x-api-key: the server's key is needed as `x-api-key: <key>` or `Authorization: Bearer <key>`")
        } else if path.starts_with("/api/") {
            "{\"error\":\"unauthorized: the server's key is needed as `Authorization: Bearer <key>`\"}".to_string()
        } else {
            error_body("Incorrect API key provided: the server's key is needed as `Authorization: Bearer <key>`.", 401)
        };
        ctx.rt.figures.refused(401);
        return respond_with(&mut *s, 401, "application/json", &format!("{cors}WWW-Authenticate: Bearer\r\n"), body.as_bytes());
    }
    // a request for an API that goes to the engine is counted as in flight until its answer ends, and is not taken while the server drains (a 503 that says so: the load balancer's next try goes elsewhere)
    let to_engine = (req.method == "POST"
        && matches!(path.as_str(), "/v1/chat/completions" | "/v1/completions" | "/v1/responses" | "/v1/messages" | "/v1/messages/count_tokens"))
        || path.starts_with("/api/");
    if to_engine {
        ctx.rt.figures.request(&path);
        // (requests being answered are bounded as the connections are: each holds a thread of its own while the engine makes its reply)
        if life.inflight() >= MAX_IN_FLIGHT {
            ctx.rt.figures.refused(503);
            let body = error_body("the server is answering as many requests as it takes: try again", 503);
            return respond_with(&mut *s, 503, "application/json", &format!("{cors}Retry-After: 1\r\n"), body.as_bytes());
        }
        match life.enter() {
            Some(g) => req.inflight = Some(g),
            None => {
                ctx.rt.figures.refused(503);
                let body = error_body("the server is shutting down: it takes no new requests", 503);
                return respond_with(&mut *s, 503, "application/json", &format!("{cors}Retry-After: 1\r\n"), body.as_bytes());
            }
        }
    }
    match (req.method.as_str(), path.as_str()) {
        ("GET" | "HEAD", "/") | ("GET" | "HEAD", "/index.html") => {
            // (a page given with the server, read for each request: its own headers, since it loads its files from /web/)
            let (page, headers) = match PAGE_DIR.get() {
                Some(d) => {
                    (std::fs::read(d.join("chat.html")).unwrap_or_else(|e| format!("<p>{}: {e}</p>", d.join("chat.html").display()).into_bytes()).into(), "")
                }
                None => (std::borrow::Cow::Borrowed(PAGE.as_bytes()), PAGE_HEADERS),
            };
            let head = format!("HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {}\r\nCache-Control: no-store\r\n{headers}{cors}Connection: close\r\n\r\n", page.len());
            if s.write_all(head.as_bytes()).is_ok() && req.method == "GET" {
                s.write_all(&page).ok();
            }
        }
        ("GET", p) if p.starts_with("/web/") && PAGE_DIR.get().is_some() => {
            // (a file of the page's: a plain name, no path)
            let f = &p[5..];
            let file = PAGE_DIR
                .get()
                .map(|d| d.join("web").join(f))
                .filter(|_| !f.is_empty() && f.bytes().all(|c| c.is_ascii_alphanumeric() || b"._-".contains(&c)) && !f.starts_with('.'));
            match file.and_then(|f| std::fs::read(f).ok()) {
                Some(b) => {
                    let kind = match f.rsplit('.').next() {
                        Some("js") => "text/javascript; charset=utf-8",
                        Some("css") => "text/css; charset=utf-8",
                        Some("svg") => "image/svg+xml",
                        Some("woff2") => "font/woff2",
                        _ => "application/octet-stream",
                    };
                    respond_with(&mut *s, 200, kind, &cors, &b);
                }
                None => respond_with(&mut *s, 404, "application/json", &cors, error_body("not found", 404).as_bytes()),
            }
        }
        ("GET", "/v1/glyd/status") if crate::monitor::on() => respond_with(&mut *s, 200, "application/json", &cors, crate::monitor::json(name).as_bytes()),
        ("GET", "/v1/glyd/info") => match ctx.rt.info.get() {
            Some(i) => respond_with(&mut *s, 200, "application/json", &cors, info_json(i, &ctx.rt).as_bytes()),
            None => respond_with(&mut *s, 404, "application/json", &cors, error_body("this server has no description (glyd serve gives one)", 404).as_bytes()),
        },
        (m @ ("GET" | "POST"), p) if p.starts_with("/v1/glyd/requests/") => {
            let rest = &p["/v1/glyd/requests/".len()..];
            let answer = match (m, rest.strip_suffix("/cancel")) {
                ("POST", Some(id)) => track_cancel(id, "asked"),
                ("GET", None) => track_status(rest),
                _ => return respond_with(&mut *s, 405, "application/json", &cors, error_body("method not allowed", 405).as_bytes()),
            };
            match answer {
                Some(b) => respond_with(&mut *s, 200, "application/json", &cors, b.as_bytes()),
                None => respond_with(
                    &mut *s,
                    404,
                    "application/json",
                    &cors,
                    error_body("no request has this id (the last 64 that ended are kept)", 404).as_bytes(),
                ),
            }
        }
        ("GET", "/health" | "/v1/health") if loading.load(Ordering::Relaxed) => {
            respond_with(&mut *s, 503, "application/json", &cors, b"{\"status\":\"loading\"}")
        }
        ("GET", "/health" | "/v1/health") => respond_with(&mut *s, 200, "application/json", &cors, b"{\"status\":\"ok\"}"),
        // (ready: the model is loaded and the server is not draining; `/health` says only that the process is up)
        ("GET" | "HEAD", "/ready") if loading.load(Ordering::Relaxed) => respond_with(&mut *s, 503, "application/json", &cors, b"{\"status\":\"loading\"}"),
        ("GET" | "HEAD", "/ready") if life.draining() => respond_with(&mut *s, 503, "application/json", &cors, b"{\"status\":\"draining\"}"),
        ("GET" | "HEAD", "/ready") => respond_with(&mut *s, 200, "application/json", &cors, b"{\"status\":\"ready\"}"),
        ("GET", "/metrics") => {
            let ready = !loading.load(Ordering::Relaxed);
            let text = crate::ops::render_of(&ctx.rt, &crate::ops::State { model: name, ready, life, engine: true });
            respond_with(&mut *s, 200, crate::ops::CONTENT_TYPE, &cors, text.as_bytes())
        }
        ("GET", "/stats") => respond_with(&mut *s, 200, "application/json", &cors, crate::sched::stats_json_of(&ctx.rt).as_bytes()),
        ("GET", "/v1/models") => {
            let body =
                format!("{{\"object\":\"list\",\"data\":[{}],\"has_more\":false,\"first_id\":{1},\"last_id\":{1}}}", model_json(name, window), jstr(name));
            respond_with(&mut *s, 200, "application/json", &cors, body.as_bytes());
        }
        ("POST", p @ ("/v1/chat/completions" | "/v1/completions" | "/v1/glyd/tokenize")) => {
            let chat = p == "/v1/chat/completions";
            if let Err(mut s) = send_job(tx, Job::Http { s, req, chat, id }) {
                respond(&mut *s, 503, "application/json", error_body("the engine is not running", 503).as_bytes());
            }
        }
        ("GET", p) if p.starts_with("/v1/models/") => {
            // (one model, as the OpenAI SDKs' `models.retrieve` asks: the model's name may hold slashes)
            let asked = p["/v1/models/".len()..].replace("%2F", "/").replace("%2f", "/").replace("%3A", ":").replace("%3a", ":");
            if asked == name {
                respond_with(&mut *s, 200, "application/json", &cors, model_json(name, window).as_bytes());
            } else {
                respond_with(&mut *s, 404, "application/json", &cors, error_body(&format!("The model `{asked}` does not exist"), 404).as_bytes());
            }
        }
        // (Ollama's API: the key guards it as it guards /v1; its errors are its own, `{"error": "..."}`)
        (_, p) if p.starts_with("/api/") => {
            if let Err(mut s) = send_job(tx, Job::Api { s, req, kind: Endpoint::Ollama, id }) {
                respond_with(&mut *s, 503, "application/json", &cors, b"{\"error\":\"the engine is not running\"}");
            }
        }
        ("POST", "/v1/responses") => {
            if let Err(mut s) = send_job(tx, Job::Api { s, req, kind: Endpoint::Responses, id }) {
                respond(&mut *s, 503, "application/json", error_body("the engine is not running", 503).as_bytes());
            }
        }
        ("POST", p @ ("/v1/messages" | "/v1/messages/count_tokens")) => {
            let count = p.ends_with("count_tokens");
            if let Err(mut s) = send_job(tx, Job::Messages { s, req, count, id }) {
                respond(&mut *s, 503, "application/json", error_body("the engine is not running", 503).as_bytes());
            }
        }
        (_, "/v1/messages" | "/v1/messages/count_tokens") => {
            respond_with(&mut *s, 405, "application/json", &cors, anthropic::anthropic_body(405, "method not allowed").as_bytes())
        }
        (_, "/v1/chat/completions") | (_, "/v1/completions") | (_, "/v1/models") | (_, "/v1/responses") | (_, "/v1/glyd/tokenize") | (_, "/v1/glyd/info") => {
            respond_with(&mut *s, 405, "application/json", &cors, error_body("method not allowed", 405).as_bytes())
        }
        _ => respond_with(&mut *s, 404, "application/json", &cors, error_body(&format!("Invalid URL ({} {path})", req.method), 404).as_bytes()),
    }
}

/// What a request's thread needs besides its connection: the tokenizer and the chat template, the model's name and sampling defaults, the window.
pub struct Shared {
    pub tok: Tok,
    pub chat: Option<Chat>,
    pub name: String,
    pub defaults: Params,
    pub window: usize,
    /// The tokens the model has a logit for (the width of a row of logits).
    pub vocab: usize,
    /// Thinking where a request does not say (None: the template's own default).
    pub thinking: Option<bool>,
    /// The fields the engine does not do besides NOT_DONE's (ONE_AT_A_TIME, or none): refused, in words, where a request sets one.
    pub refused: &'static [(&'static str, &'static str)],
    /// The model's runtime context (src/runtime/context.rs): its figures, its engine's first failure, what /v1/glyd/info says of it. The process's own where one model is served.
    pub rt: Arc<crate::runtime::context::Context>,
}

/// The chat page's directory where the server was given one (`chat.html`, and its files in `web/`), in place of the page built in.
pub static PAGE_DIR: std::sync::OnceLock<std::path::PathBuf> = std::sync::OnceLock::new();

/// What `GET /v1/glyd/info` says that does not change while the server runs, set by `serve` once the model is loaded: the fields `serve` knows (the model, its levels, the precision, the GPU: cli.rs
/// `about`), the GPU to ask for its memory now, what the model holds there, the sequences it runs at once, its context, the fields its engine does not do, and when the load began.
pub struct Info {
    pub about: Vec<(&'static str, Value)>,
    pub gpu: Option<i32>,
    pub held: Option<crate::dev::Held>,
    pub concurrency: usize,
    pub window: usize,
    pub refused: &'static [(&'static str, &'static str)],
    pub started: f64,
}

/// The version of what /v1/glyd/info, /v1/glyd/requests/ID and /v1/glyd/tokenize say (docs/serve-api.md).
pub const API: &str = "glyd-serve/1";

fn names<'a>(l: impl Iterator<Item = &'a str>) -> Value {
    Value::Array(l.map(Value::from).collect())
}

/// /v1/glyd/info's answer: `i`, the build, the limits and the fields, and what is so now: whether the engine has failed, the requests, and the memory (the engine's buffers, and the GPU's as the driver has it:
/// every program's on it).
fn info_json(i: &Info, rt: &crate::runtime::context::Context) -> String {
    use tools::obj;
    let (num, secs) = (|n: usize| Value::from(n as u64), |t: f64| Value::Number(format!("{t:.3}")));
    let (live, peak, peak_at) = crate::dev::held_now();
    let gpu = i.gpu.and_then(|o| {
        let g = crate::dev::Dev::probe(o).ok();
        crate::dev::Dev::release(o);
        g
    });
    let failed = rt.failed.lock().unwrap_or_else(|e| e.into_inner()).clone();
    let (running, cancelling) = track_counts();
    let here: Vec<&str> = i.refused.iter().map(|(n, _)| *n).collect();
    let held = |f: fn(&crate::dev::Held) -> u64| i.held.as_ref().map_or(Value::Null, |h| Value::from(f(h)));
    let mut o = vec![("object", Value::from("glyd.info")), ("api", Value::from(API))];
    o.push((
        "build",
        obj(vec![
            ("version", Value::from(env!("CARGO_PKG_VERSION"))),
            ("commit", option_env!("GLYD_SOURCE_COMMIT").filter(|c| !c.is_empty()).map_or(Value::Null, Value::from)),
            ("development", Value::Bool(cfg!(feature = "dev"))),
        ]),
    ));
    o.extend(i.about.iter().cloned());
    o.push((
        "limits",
        obj(vec![
            ("context_tokens", num(i.window)),
            ("concurrency", num(i.concurrency)),
            ("max_choices", num(MAX_CHOICES)),
            ("max_body_bytes", num(MAX_BODY)),
            ("max_header_bytes", num(MAX_HEAD)),
            ("max_headers", num(MAX_HEADERS)),
            ("max_connections", Value::Null), // (no cap: the connections are the front end's tasks)
            ("read_timeout_seconds", num(30)),
        ]),
    ));
    o.push((
        "fields",
        obj(vec![
            ("read", names(DONE.iter().copied().filter(|f| !here.contains(f)))),
            ("no_effect", names(NO_EFFECT.iter().copied())),
            ("refused", names(NOT_DONE.iter().map(|(n, _)| *n).chain(CONSTRAINED.iter().copied()).chain(here.iter().copied()))),
        ]),
    ));
    o.push(("state", Value::from(if failed.is_some() { "failed" } else { "ready" })));
    o.push(("error", failed.map_or(Value::Null, |(t, e)| obj(vec![("at", secs(t)), ("message", Value::String(e))]))));
    o.push(("requests", obj(vec![("running", num(running)), ("cancelling", num(cancelling))])));
    let scratch = i.held.as_ref().map_or(Value::Null, |h| Value::from(live.saturating_sub(h.weights + h.kv + h.state)));
    o.push((
        "memory",
        obj(vec![
            ("gpu", i.gpu.map_or(Value::Null, |g| num(g as usize))),
            ("measured_at", secs(crate::monitor::clock())),
            ("engine_bytes", Value::from(live)),
            ("weights_bytes", held(|h| h.weights)),
            ("kv_bytes", held(|h| h.kv)),
            ("state_bytes", held(|h| h.state)),
            ("scratch_bytes", scratch),
            ("peak_bytes", Value::from(peak)),
            ("peak_at", secs(peak_at as f64 / 1000.0)),
            ("peak_since", secs(i.started)),
            ("device_total_bytes", gpu.as_ref().map_or(Value::Null, |g| Value::from(g.total))),
            ("device_free_bytes", gpu.as_ref().map_or(Value::Null, |g| Value::from(g.free))),
            ("not_counted", names(["the CUDA context and the kernels' code", "cuBLAS's own workspace", "the computer's memory"].into_iter())),
        ]),
    ));
    json::to_compact(&obj(o))
}

/// The scheduler's sequences seen from a request's thread: a reply is a sequence sent to the engine's thread, its tokens read as they come.
pub struct Remote {
    pub tx: Sender<Job>,
    pub window: usize,
}

/// A sequence sent to the scheduler: its events come back on `events`; `abort` ends it.
pub struct Handle {
    pub events: Receiver<Ev>,
    pub abort: Arc<AtomicBool>,
    /// The prompt tokens the scheduler found in its cache as it let the sequence in (`sched::Submit::cached`): read for the request's usage once the sequence has ended.
    pub cached: Arc<AtomicU32>,
}

impl Remote {
    /// `prompt` and its reply sent to the scheduler as a sequence.
    pub fn start(&self, prompt: &[u32], o: &Reply) -> Result<Handle> {
        self.start_with(prompt, Vec::new(), o)
    }

    /// `start` with the prompt's images (their places in it their tokens).
    pub fn start_with(&self, prompt: &[u32], images: Vec<Arc<crate::vision::Image>>, o: &Reply) -> Result<Handle> {
        self.start_in(prompt, images, o, 0)
    }

    /// `start_with` in the cache scope `salt` (`sched::Submit::salt`; 0: none).
    pub fn start_in(&self, prompt: &[u32], images: Vec<Arc<crate::vision::Image>>, o: &Reply, salt: u64) -> Result<Handle> {
        if prompt.is_empty() {
            return err("an empty prompt");
        }
        if prompt.len() + 1 > self.window {
            return err(format!("{} tokens, where the context holds {}", prompt.len(), self.window));
        }
        let (out, events) = std::sync::mpsc::channel();
        let (abort, cached) = (Arc::new(AtomicBool::new(false)), Arc::new(AtomicU32::new(0)));
        self.tx
            .send(Job::Seq(Submit {
                prompt: prompt.to_vec(),
                reply: o.clone(),
                out,
                abort: abort.clone(),
                resumed: Default::default(),
                cached: cached.clone(),
                salt,
                images,
            }))
            .map_err(|_| Error("the engine is not running".into()))?;
        Ok(Handle { events, abort, cached })
    }
}

impl Gen for Remote {
    fn capacity(&self) -> usize {
        self.window
    }

    fn reply(&mut self, prompt: &[u32], o: &Reply, each: &mut dyn FnMut(u32) -> bool) -> Result<(Vec<u32>, Finish)> {
        let h = self.start(prompt, o)?;
        let mut tokens = Vec::new();
        loop {
            match h.events.recv() {
                Ok(Ev::Token(t)) => {
                    tokens.push(t);
                    if !each(t) {
                        h.abort.store(true, Ordering::Relaxed);
                        return Ok((tokens, Finish::Stopped));
                    }
                }
                Ok(Ev::Lp(_)) => {}
                Ok(Ev::End(Ok(f))) => return Ok((tokens, f)),
                Ok(Ev::End(Err(e))) => return Err(e),
                Err(_) => return err("the engine stopped"),
            }
        }
    }
}

/// The engine's thread: the scheduler's rounds (a step for every sequence that runs, a chunk of a prompt), and between them the jobs that come, until `Job::Stop`. A completions request and the terminal's chat
/// each run on a thread of their own (their text, their connection) and ask the scheduler for a sequence through `tx`; the engine's thread never waits on a connection.
pub fn work(e: &mut Engine, shared: Arc<Shared>, rx: &Receiver<Job>, tx: &Sender<Job>, users: usize) {
    work_with(&mut Sched::new(e, users), shared, rx, tx);
}

/// A request's thread or the terminal's reply, started for `job` (the engine's thread never waits on a connection); any other job is not one.
/// A request's handler on a thread of its own, for as long as the request is answered (it waits for the engine's events). Where the system will not give another thread, the client is told so
/// (503, `Retry-After`) and the engine's own thread goes on: a refused spawn must not take the server down with it.
fn spawn_handler(s: Box<dyn Conn>, run: impl FnOnce(&mut dyn Conn) + Send + 'static) {
    let cell = Arc::new(std::sync::Mutex::new(Some(s)));
    let inner = cell.clone();
    let spawned = std::thread::Builder::new().name("glyd-request".into()).spawn(move || {
        let taken = inner.lock().ok().and_then(|mut g| g.take());
        if let Some(mut s) = taken {
            if !s.gone() {
                run(&mut *s);
            }
        }
    });
    if spawned.is_err() {
        crate::ops::refused(503);
        if let Some(mut s) = cell.lock().ok().and_then(|mut g| g.take()) {
            respond_with(
                &mut *s,
                503,
                "application/json",
                "Retry-After: 1\r\n",
                error_body("the server cannot start another thread for this request: try again", 503).as_bytes(),
            );
        }
    }
}

pub fn detach(job: Job, shared: &Arc<Shared>, tx: &Sender<Job>) {
    let (sh, tx) = (shared.clone(), tx.clone());
    match job {
        Job::Http { s, req, chat, id } => spawn_handler(s, move |s| handle_completion(&sh, &tx, s, &req, chat, id)),
        Job::Messages { s, req, count, id } => spawn_handler(s, move |s| anthropic::handle(&sh, &tx, s, &req, count, id)),
        Job::Api { s, req, kind, id } => spawn_handler(s, move |s| match kind {
            Endpoint::Responses => responses::handle(&sh, &tx, s, &req, id),
            Endpoint::Ollama => ollama::handle(&sh, &tx, s, &req),
        }),
        Job::Term { messages, request, out, abort } => {
            std::thread::spawn(move || {
                let mut remote = Remote { tx, window: sh.window };
                let mut talk = Talk { e: &mut remote, tok: &sh.tok, chat: sh.chat.as_ref() };
                let sink = out.clone();
                let r = talk.reply(&messages, &request, &mut |t| sink.send(Piece::Text(t.to_string())).is_ok() && !abort.load(Ordering::Relaxed));
                let _ = out.send(Piece::End(r.map_err(|e| e.to_string())));
            });
        }
        _ => {}
    }
}

/// `work` for an engine that has no scheduler (a lossy level's): the sequences are run one at a time, in the order they come; one that is asked to end before its turn (a cancel, its client gone) is not run.
pub fn work_gen(e: &mut dyn Gen, shared: Arc<Shared>, rx: &Receiver<Job>, tx: &Sender<Job>) {
    while let Ok(job) = rx.recv() {
        match job {
            Job::Stop => return,
            Job::Seq(sub) if sub.abort.load(Ordering::Relaxed) => {
                let _ = sub.out.send(Ev::End(Ok(Finish::Stopped)));
            }
            Job::Seq(sub) => {
                let mut timer = crate::ops::Timer::start(sub.prompt.len());
                let r = e.reply(&sub.prompt, &sub.reply, &mut |t| {
                    timer.token();
                    sub.out.send(Ev::Token(t)).is_ok() && !sub.abort.load(Ordering::Relaxed)
                });
                timer.end(match &r {
                    Ok((_, Finish::Stop)) => "stop",
                    Ok((_, Finish::Length)) => "length",
                    Ok((_, Finish::Stopped)) => "stopped",
                    Err(_) => "error",
                });
                if let Err(err) = &r {
                    shared.rt.fail(&err.to_string());
                }
                let lost = r.as_ref().err().filter(|e| e.sticky()).cloned();
                let _ = sub.out.send(Ev::End(r.map(|x| x.1)));
                if let Some(err) = lost {
                    device_lost(&err);
                    return;
                }
            }
            _ => detach(job, &shared, tx),
        }
    }
}

/// `work` on any scheduler (the tests run it on a scheduler of a made-up model).
pub fn work_with<B: ModelStep>(sched: &mut Sched<B>, shared: Arc<Shared>, rx: &Receiver<Job>, tx: &Sender<Job>) {
    // (the scheduler's figures and its sequences' counts go where the server's own do: the model's runtime context)
    let rt = shared.rt.clone();
    sched.set_context(rt.clone());
    // A thread of its own takes the jobs off the queue as they come: the requests' threads (which parse, tokenize and apply the chat template) start at once, and not at the end of the prompt pass or decode
    // step the scheduler is in (20 ms of every first token with eight requests at once), and the sequences they make are passed on; the scheduler takes them in at its next round.
    let (wtx, wrx) = std::sync::mpsc::channel::<Job>();
    /// The queue's receiver for the one thread that reads it from here on (this function's own thread does not touch it until the scope ends).
    struct OnlyReader<'a>(&'a Receiver<Job>);
    // SAFETY: a Receiver may move to another thread (its jobs are Send); only the thread given this reads it while the scope runs, and the scope ends before `rx` is used again.
    unsafe impl Send for OnlyReader<'_> {}
    let reader = OnlyReader(rx);
    std::thread::scope(|scope| {
        scope.spawn(move || {
            let reader = reader;
            while let Ok(job) = reader.0.recv() {
                match job {
                    Job::Http { .. } | Job::Messages { .. } | Job::Api { .. } | Job::Term { .. } => detach(job, &shared, tx),
                    Job::Stop => {
                        let _ = wtx.send(Job::Stop);
                        return;
                    }
                    job => {
                        if wtx.send(job).is_err() {
                            return;
                        }
                    }
                }
            }
            // (every sender is gone: the scheduler's queue closes, and it ends when it has nothing left to run)
            drop(wtx);
        });
        let mut next: Option<Job> = None;
        loop {
            if next.is_none() {
                next = if sched.idle() { wrx.recv().ok() } else { wrx.try_recv().ok() };
                if next.is_none() && sched.idle() {
                    return;
                }
            }
            while let Some(job) = next.take() {
                match job {
                    Job::Stop => {
                        sched.fail(&crate::ffi::Error("the server is stopping".to_string()));
                        return;
                    }
                    Job::Seq(sub) => sched.submit(sub),
                    _ => {} // (the requests' own jobs went to their threads at the dispatcher)
                }
                next = wrx.try_recv().ok();
            }
            if let Err(err) = sched.tick() {
                eprintln!("glyd: {err}");
                rt.fail(&err.to_string());
                sched.fail(&err);
                if err.sticky() {
                    device_lost(&err);
                    return;
                }
            }
        }
    });
}

// ---- a request by its id: how it stands, and a cancel ---------------------------------------------------------------------------------------------------

/// A completions request by the id its reply carries: its sequences' stop flags, why it was asked to end before its time (a cancel, its client gone), when it began, and when the engine had let go of every
/// sequence of it and how it ended.
struct Track {
    id: String,
    aborts: Vec<Arc<AtomicBool>>,
    cancel: Option<&'static str>,
    began: f64,
    ended: Option<(f64, String)>,
}

/// The requests that run, and the last KEPT that ended. One server a process: the ids are the process's (`listen_loading`'s count).
static TRACKS: std::sync::Mutex<std::collections::VecDeque<Track>> = std::sync::Mutex::new(std::collections::VecDeque::new());
// (the tests' servers share the process, so each test's ended requests are kept while the others end theirs)
const KEPT: usize = if cfg!(test) { 1 << 16 } else { 64 };

fn tracks() -> std::sync::MutexGuard<'static, std::collections::VecDeque<Track>> {
    TRACKS.lock().unwrap_or_else(|e| e.into_inner())
}

/// A request begun under `id`; false where one of that id runs (an id names one request at a time). One that ended under it is forgotten.
fn track_begin(id: &str) -> bool {
    let mut t = tracks();
    if t.iter().any(|x| x.id == id && x.ended.is_none()) {
        return false;
    }
    t.retain(|x| x.id != id);
    while t.iter().filter(|x| x.ended.is_some()).count() >= KEPT {
        let first = t.iter().position(|x| x.ended.is_some()).expect("an ended request");
        t.remove(first);
    }
    t.push_back(Track { id: id.to_string(), aborts: Vec::new(), cancel: None, began: crate::monitor::clock(), ended: None });
    true
}

/// A sequence of request `id` begun: its stop flag, set at once where the request is asked to end already.
fn track_seq(id: &str, abort: &Arc<AtomicBool>) {
    if let Some(x) = tracks().iter_mut().find(|x| x.id == id && x.ended.is_none()) {
        if x.cancel.is_some() {
            abort.store(true, Ordering::Relaxed);
        }
        x.aborts.push(abort.clone());
    }
}

/// Request `id` asked to end before its time, `why`: every sequence of it stops at its next step (or is skipped where its turn has not come). Its status, or None where no request has the id.
fn track_cancel(id: &str, why: &'static str) -> Option<String> {
    let mut t = tracks();
    let x = t.iter_mut().find(|x| x.id == id)?;
    if x.ended.is_none() {
        x.cancel = x.cancel.or(Some(why));
        for a in &x.aborts {
            a.store(true, Ordering::Relaxed);
        }
    }
    Some(track_json(x))
}

/// Request `id` over: the engine has let go of every sequence of it (each choice's thread has seen its end), so what they held is the engine's to use again. `failure`: the error it ended with.
fn track_end(id: &str, failure: Option<&str>) {
    if let Some(x) = tracks().iter_mut().find(|x| x.id == id && x.ended.is_none()) {
        let how = match (failure, x.cancel) {
            (Some(e), _) => format!("failed: {e}"),
            (None, Some(_)) => "cancelled".to_string(),
            (None, None) => "finished".to_string(),
        };
        x.ended = Some((crate::monitor::clock(), how));
        x.aborts.clear();
    }
}

/// How request `id` stands, or None where no request has the id.
fn track_status(id: &str) -> Option<String> {
    tracks().iter().find(|x| x.id == id).map(track_json)
}

/// (running, asked to end and not ended yet) of the requests.
fn track_counts() -> (usize, usize) {
    let t = tracks();
    let running = t.iter().filter(|x| x.ended.is_none() && x.cancel.is_none()).count();
    (running, t.iter().filter(|x| x.ended.is_none() && x.cancel.is_some()).count())
}

/// A request's status: `running`; `cancelling`, asked to end (a cancel, or its client gone) and the engine not done with it yet; `ended`, the engine has let go of it, with how (finished, cancelled, failed: ...).
fn track_json(x: &Track) -> String {
    let state = match (&x.ended, x.cancel) {
        (Some(_), _) => "ended",
        (None, Some(_)) => "cancelling",
        (None, None) => "running",
    };
    let (ended, how) = x.ended.as_ref().map_or(("null".to_string(), "null".to_string()), |(t, h)| (format!("{t:.3}"), jstr(h)));
    format!(
        "{{\"id\":{},\"state\":\"{state}\",\"cancel\":{},\"began\":{:.3},\"ended\":{ended},\"how\":{how}}}",
        jstr(&x.id),
        x.cancel.map_or("null".to_string(), jstr),
        x.began
    )
}

/// The sticky device errors this process's servers ended for (a test's server does not end the process: it says so here).
#[cfg(test)]
static LOST: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());

/// The engine's GPU context is lost (`ffi::Error::sticky`): every later step would fail as this one did, so the server does not go on. Its requests have had the error (`Sched::fail`, or the
/// reply's end) and /v1/glyd/info says the server failed; after a moment for those replies to be sent the process ends with `ui::DEVICE_LOST`. The service sees the program gone and starts
/// the model again when it is next asked for, and `glyd serve` alone ends with the error said.
fn device_lost(err: &crate::ffi::Error) {
    eprintln!("glyd: the GPU cannot be used after this error; the server is stopping ({})", err.msg);
    #[cfg(test)]
    LOST.lock().unwrap_or_else(|e| e.into_inner()).push(err.msg.clone());
    #[cfg(not(test))]
    {
        std::thread::sleep(std::time::Duration::from_millis(500));
        std::process::exit(crate::ui::DEVICE_LOST);
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::io::BufReader;

    #[test]
    fn json_strings() {
        assert_eq!(jstr("a\"b\\c\nd\u{1}é"), "\"a\\\"b\\\\c\\nd\\u0001é\"");
    }

    fn get(addr: std::net::SocketAddr, path: &str, auth: Option<&str>) -> String {
        let mut c = TcpStream::connect(addr).unwrap();
        let a = auth.map_or(String::new(), |k| format!("Authorization: Bearer {k}\r\n"));
        c.write_all(format!("GET {path} HTTP/1.1\r\nHost: x\r\n{a}\r\n").as_bytes()).unwrap();
        let mut out = String::new();
        c.read_to_string(&mut out).unwrap();
        out
    }

    #[test]
    fn the_api_key_guards_the_api_and_not_the_page() {
        let (tx, _rx) = std::sync::mpsc::channel();
        let addr = listen("127.0.0.1:0", "m", 4096, Some("secret".to_string()), false, tx).unwrap();
        assert!(get(addr, "/v1/models", None).starts_with("HTTP/1.1 401"));
        assert!(get(addr, "/v1/models", Some("wrong")).starts_with("HTTP/1.1 401"));
        let ok = get(addr, "/v1/models", Some("secret"));
        assert!(ok.starts_with("HTTP/1.1 200") && ok.contains("\"max_model_len\":4096"), "{ok}");
        assert!(get(addr, "/health", None).starts_with("HTTP/1.1 200"));
        assert!(get(addr, "/", None).starts_with("HTTP/1.1 200"));
        assert!(get(addr, "/nothing", None).starts_with("HTTP/1.1 404"));
        // no key set: the API is open
        let (tx2, _rx2) = std::sync::mpsc::channel();
        let open = listen("127.0.0.1:0", "m", 4096, None, false, tx2).unwrap();
        assert!(get(open, "/v1/models", None).starts_with("HTTP/1.1 200"));
    }

    /// A server that listens while its model loads: /health is 503 `loading` until the flag is cleared, then 200 `ok`; the model list answers throughout, and the key still guards /v1/health only.
    #[test]
    fn health_says_loading_until_the_model_can_serve() {
        let (tx, _rx) = std::sync::mpsc::channel();
        let loading = Arc::new(AtomicBool::new(true));
        let addr = listen_loading("127.0.0.1:0", "m", 4096, Some("secret".to_string()), false, tx, loading.clone()).unwrap();
        let h = get(addr, "/health", None);
        assert!(h.starts_with("HTTP/1.1 503 Service Unavailable") && h.ends_with("{\"status\":\"loading\"}"), "{h}");
        assert!(get(addr, "/v1/health", None).starts_with("HTTP/1.1 401"));
        assert!(get(addr, "/v1/health", Some("secret")).starts_with("HTTP/1.1 503"));
        assert!(get(addr, "/v1/models", Some("secret")).starts_with("HTTP/1.1 200"), "the model list is not changed");
        loading.store(false, Ordering::Relaxed);
        let h = get(addr, "/health", None);
        assert!(h.starts_with("HTTP/1.1 200 OK") && h.ends_with("{\"status\":\"ok\"}"), "{h}");
        assert!(get(addr, "/v1/health", Some("secret")).starts_with("HTTP/1.1 200"));
    }

    #[test]
    fn the_standalone_servers_words() {
        assert_eq!(
            without_vllm("echo of a chat reply is not supported by this engine yet (GLYD_ENGINE=0 runs vLLM, which has it)"),
            "echo of a chat reply is not supported by this engine yet"
        );
        assert_eq!(
            without_vllm("n must be at most 8, got 9. (This engine runs up to 8 sequences at once; GLYD_ENGINE=0 runs vLLM, which takes more.)"),
            "n must be at most 8, got 9."
        );
        assert_eq!(without_vllm("a (b) c"), "a (b) c");
        let v = |t: &str| json::parse(t).unwrap();
        assert!(json_rule(&v(r#"{"response_format":{"type":"json_object"}}"#)).unwrap().starts_with("Reply with one JSON object"));
        assert!(json_rule(&v(r#"{"response_format":{"type":"json_schema","json_schema":{"name":"x","schema":{"type":"object"}}}}"#))
            .unwrap()
            .ends_with(r#"schema: {"type":"object"}"#));
        assert_eq!(json_rule(&v(r#"{"response_format":{"type":"text"}}"#)), None);
    }

    /// The forms of a reply a request asks for, as OpenAI's and vLLM's names give them.
    #[test]
    fn the_forms_of_a_reply_a_request_asks_for() {
        let spec = |t: &str| constraint_spec(&json::parse(t).unwrap());
        assert_eq!(spec(r#"{"messages":[]}"#), Ok(None));
        assert_eq!(spec(r#"{"response_format":{"type":"text"}}"#), Ok(None));
        assert_eq!(spec(r#"{"response_format":null,"guided_json":null}"#), Ok(None));
        assert_eq!(spec(r#"{"response_format":{"type":"json_object"}}"#), Ok(Some(Spec::JsonObject)));
        let schema = r#"{"type":"object","properties":{"a":{"type":"integer"}}}"#;
        let want = Ok(Some(Spec::JsonSchema(schema.to_string())));
        assert_eq!(spec(&format!(r#"{{"response_format":{{"type":"json_schema","json_schema":{{"name":"x","strict":true,"schema":{schema}}}}}}}"#)), want);
        assert_eq!(spec(&format!(r#"{{"response_format":{{"type":"json_schema","json_schema":{{"name":"x","schema":{schema}}}}}}}"#)), want);
        assert_eq!(spec(&format!(r#"{{"guided_json":{schema}}}"#)), want);
        assert_eq!(spec(&format!(r#"{{"structured_outputs":{{"json":{schema}}}}}"#)), want);
        assert_eq!(spec(r#"{"guided_json":"{\"type\":\"object\"}"}"#), Ok(Some(Spec::JsonSchema(r#"{"type":"object"}"#.to_string()))));
        assert_eq!(spec(r#"{"guided_regex":"[0-9]+"}"#), Ok(Some(Spec::Regex("[0-9]+".to_string()))));
        assert_eq!(spec(r#"{"structured_outputs":{"regex":"a+"}}"#), Ok(Some(Spec::Regex("a+".to_string()))));
        assert_eq!(spec(r#"{"guided_choice":["yes","no"]}"#), Ok(Some(Spec::Choice(vec!["yes".to_string(), "no".to_string()]))));
        assert_eq!(spec(r#"{"guided_grammar":"start: \"a\""}"#), Ok(Some(Spec::Grammar("start: \"a\"".to_string()))));
        assert_eq!(spec(r#"{"structured_outputs":{"json_object":true,"disable_fallback":true}}"#), Ok(Some(Spec::JsonObject)));
        // what is wrong is said
        let bad = |t: &str| spec(t).unwrap_err();
        assert!(bad(r#"{"response_format":{"type":"json_schema"}}"#).contains("json_schema.schema"));
        assert!(bad(r#"{"response_format":{"type":"xml"}}"#).contains("xml"));
        assert!(bad(r#"{"guided_choice":[]}"#).contains("guided_choice"));
        assert!(bad(r#"{"guided_choice":[1]}"#).contains("list of strings"));
        assert!(bad(r#"{"guided_json":5}"#).contains("guided_json"));
        assert!(bad(r#"{"guided_regex":5}"#).contains("string"));
        assert!(bad(r#"{"structured_outputs":{"nonsense":1}}"#).contains("structured_outputs.nonsense"));
        assert!(bad(r#"{"guided_regex":"a","guided_choice":["b"]}"#).contains("one form"));
        // the grammars are done, and what is wrong with one is said
        let u = |t: &str| unsupported(&json::parse(t).unwrap());
        for t in [r#"{"response_format":{"type":"json_object"}}"#, r#"{"guided_json":{}}"#, r#"{"structured_outputs":{"regex":"a"}}"#] {
            assert_eq!(u(t), None, "{t}");
        }
        assert!(u(r#"{"guided_choice":[]}"#).unwrap().contains("guided_choice"));
        assert!(u(r#"{"guided_whitespace_pattern":"x"}"#).is_some());
        // (the prompt says what the reply is to be)
        let r = |t: &str| json_rule(&json::parse(t).unwrap());
        assert!(r(&format!(r#"{{"guided_json":{schema}}}"#)).unwrap().ends_with(schema));
        assert_eq!(r(r#"{"guided_regex":"a"}"#), None);
    }

    #[test]
    fn what_is_not_done_is_said() {
        let u = |t: &str| unsupported(&json::parse(t).unwrap());
        assert_eq!(u(r#"{"messages":[]}"#), None);
        assert_eq!(u(r#"{"n":1,"logprobs":false,"tools":[],"response_format":{"type":"text"},"logit_bias":{}}"#), None);
        assert_eq!(u(r#"{"n":2}"#), None); // (several choices are done: see `several_choices_from_one_prompt`)
        assert_eq!(u(r#"{"tools":[{"type":"function"}]}"#), None); // (tools are done: see `tool_calls_through_the_api`; `invalid` says what is wrong in them)
        assert_eq!(u(r#"{"response_format":{"type":"json_object"}}"#), None); // (grammars are done: see `the_forms_of_a_reply_a_request_asks_for`)
                                                                              // (log-probabilities are done: see `log_probabilities_through_the_api`)
        assert_eq!(u(r#"{"logprobs":true}"#), None);
        // (a logit bias is done now: see `logit_bias_as_vllm_takes_it`)
        assert_eq!(u(r#"{"logit_bias":{"50256":-100}}"#), None);
    }

    /// vLLM's logit_bias: the keys are token ids as strings, a value is held to -100..100 (clamped, not refused), an id must be in the vocabulary, a later key of the same id wins; what is not a number or an id is refused in words.
    #[test]
    fn logit_bias_as_vllm_takes_it() {
        let bias = |t: &str, vocab: usize| logit_bias_of(&json::parse(t).unwrap(), vocab);
        assert_eq!(bias(r#"{"logit_bias":{"15":5,"7":-250,"3":1e3,"+9":0.5}}"#, 100).unwrap(), vec![(15, 5.0), (7, -100.0), (3, 100.0), (9, 0.5)]);
        assert_eq!(bias(r#"{"logit_bias":{"15":5,"+15":-5}}"#, 100).unwrap(), vec![(15, -5.0)]);
        assert_eq!(bias(r#"{"logit_bias":null}"#, 100).unwrap(), vec![]);
        assert_eq!(bias(r#"{}"#, 100).unwrap(), vec![]);
        assert_eq!(bias(r#"{"logit_bias":{}}"#, 100).unwrap(), vec![]);
        let e = |t: &str| bias(t, 100).unwrap_err();
        assert!(e(r#"{"logit_bias":{"abc":1}}"#).contains("cannot be converted to integer token IDs") && e(r#"{"logit_bias":{"abc":1}}"#).contains("abc"));
        assert!(e(r#"{"logit_bias":{"100":1}}"#).contains("out-of-vocab") && e(r#"{"logit_bias":{"100":1}}"#).contains("Vocabulary size: 100"));
        assert!(e(r#"{"logit_bias":{"-1":1}}"#).contains("out-of-vocab"));
        assert!(e(r#"{"logit_bias":{"5":"x"}}"#).contains("logit_bias"));
        assert!(e(r#"{"logit_bias":[1,2]}"#).contains("logit_bias"));
        // the request carries it as the sampler wants it, and `invalid` says what is wrong before any work is done
        let v = json::parse(r#"{"logit_bias":{"15":5,"7":-250}}"#).unwrap();
        assert_eq!(request_of(&v, 4096, &Params::default()).params.logit_bias, vec![(15, 5.0), (7, -100.0)]);
        assert_eq!(invalid(&v, true, 100), None);
        assert!(invalid(&json::parse(r#"{"logit_bias":{"500":5}}"#).unwrap(), true, 100).unwrap().contains("out-of-vocab"));
    }

    /// P5e: the prompt tokens a completion found in the cache are in its usage as OpenAI's API has them (`prompt_tokens_details.cached_tokens`, among `prompt_tokens`), whole
    /// and streamed, for a request with a cache scope of its own (`cache_salt`); the first time a prompt is seen in a scope there are none and the usage is as it was. A scope
    /// finds nothing of another's. A request with no scope is served from the cache and told nothing of it.
    #[test]
    fn the_usage_counts_the_prompt_tokens_found_in_the_cache() {
        let rt = crate::runtime::context::Context::new();
        let addr = serve_fake_in(Fake::new(64, 32), None, 2, false, None, &[], rt.clone());
        let q = |stream: bool, salt: &str| {
            format!(
                r#"{{"model":"m","max_tokens":20,"temperature":0,"stream":{stream},"stream_options":{}{salt},"messages":[{{"role":"user","content":"LONG"}}]}}"#,
                if stream { r#"{"include_usage":true}"# } else { "null" }
            )
            .replace("LONG", "the quick brown fox jumps over the lazy dog and keeps on running through the field until the sun goes down behind the hills the quick brown fox jumps over the lazy dog and keeps on running through the field until the sun goes down behind the hills the quick brown fox jumps over the lazy dog and keeps on running through the field until the sun goes down behind the hills")
        };
        let found = |body: &str| -> Option<u64> {
            at(&json::parse(body).unwrap(), &["usage"]).get("prompt_tokens_details").map(|d| at(d, &["cached_tokens"]).as_u64().unwrap())
        };
        // (the server's own count of the prompt tokens found, which /stats and /metrics do not tell either: null, and no such metric)
        let hits = || rt.sched[1].load(Ordering::Relaxed);
        // no scope, twice: the second is served from the cache (the server's own count says so) and its usage does not say it
        let (st, first) = call(addr, "/v1/chat/completions", &q(false, ""));
        assert_eq!(st, 200, "{first}");
        let prompt = at(&json::parse(&first).unwrap(), &["usage", "prompt_tokens"]).as_u64().unwrap();
        let again = call(addr, "/v1/chat/completions", &q(false, "")).1;
        assert!(hits() >= 32 && found(&first).is_none() && found(&again).is_none(), "{again}");
        let stats = get(addr, "/stats", None);
        assert!(
            stats.contains("\"prefix_hit_tokens\":null,\"prefix_hit_rate_percent\":null")
                && !get(addr, "/metrics", None).contains("glyd_prefix_cache_hit_tokens_total"),
            "{stats}"
        );
        // a scope: nothing of the requests with none; then its own
        let a = r#","cache_salt":"tenant-a""#;
        let before = hits();
        let first = call(addr, "/v1/chat/completions", &q(false, a)).1;
        assert!(found(&first).is_none() && hits() == before, "{first}");
        let again = call(addr, "/v1/chat/completions", &q(false, a)).1;
        let n = found(&again).expect("cached tokens");
        assert!(
            at(&json::parse(&again).unwrap(), &["usage", "prompt_tokens"]).as_u64() == Some(prompt) && n >= 32 && n < prompt && n % crate::kv::PAGE as u64 == 0,
            "{n} of {prompt}"
        );
        // another scope finds nothing of it
        assert!(found(&call(addr, "/v1/chat/completions", &q(false, r#","cache_salt":"tenant-b""#)).1).is_none());
        // streamed, with the usage asked for: the last chunk's
        let (st, body) = call(addr, "/v1/chat/completions", &q(true, a));
        assert_eq!(st, 200, "{body}");
        let last = body.lines().filter_map(|l| l.strip_prefix("data: ")).rfind(|d| d.contains("\"usage\":{")).expect("a chunk with the usage");
        assert_eq!(found(last), Some(n), "{body}");
    }

    /// A server whose operator has said every caller is one user (GLYD_ENGINE_CACHED_USAGE=1, `Context::tell_cached`) tells a request with no scope what it found. It is
    /// not the listener's address that decides: this one is on a loopback address like the others, which tell nothing.
    #[test]
    fn a_server_told_its_callers_are_one_user_tells_what_was_found_without_a_scope() {
        let body = format!(r#"{{"model":"fake","max_tokens":8,"temperature":0,"prompt":"{}"}}"#, "the quick brown fox jumps over the lazy dog and keeps on running through the field until the sun goes down behind the hills the quick brown fox jumps over the lazy dog and keeps on running through the field until the sun goes down behind the hills the quick brown fox jumps over the lazy dog and keeps on running through the field until the sun goes down behind the hills");
        let found = |addr| -> (Option<u64>, Option<u64>) {
            let ask = || json::parse(&call(addr, "/v1/completions", &body).1).unwrap();
            let details = |d: &Value| at(d, &["usage"]).get("prompt_tokens_details").map(|x| at(x, &["cached_tokens"]).as_u64().unwrap());
            (details(&ask()), details(&ask()))
        };
        let rt = crate::runtime::context::Context::new();
        rt.tell_cached.store(true, Ordering::Relaxed);
        let addr = serve_fake_in(Fake::new(64, 32), None, 2, false, None, &[], rt);
        let (first, again) = found(addr);
        assert!(first.is_none() && again.is_some_and(|n| n >= 32), "{first:?} {again:?}");
        let (stats, metrics) = (get(addr, "/stats", None), get(addr, "/metrics", None));
        assert!(stats.contains(&format!("\"prefix_hit_tokens\":{}", again.unwrap())) && metrics.contains("glyd_prefix_cache_hit_tokens_total"), "{stats}");
        // the same server without the operator's word
        assert_eq!(found(serve_fake_in(Fake::new(64, 32), None, 2, false, None, &[], crate::runtime::context::Context::new())), (None, None));
    }

    #[test]
    fn requests_as_the_openai_api_names_them() {
        let v = json::parse(r#"{"messages":[{"role":"system","content":"s"},{"role":"user","content":[{"type":"text","text":"a"},{"type":"text","text":"b"}]}],"temperature":0.5,"top_p":0.9,"top_k":40,"max_tokens":64,"stop":["x","y"],"seed":7,"chat_template_kwargs":{"enable_thinking":false}}"#).unwrap();
        let m = tools::template_messages(&v).unwrap();
        assert_eq!(
            (at(&m[0], &["role"]).as_str(), at(&m[0], &["content"]).as_str(), at(&m[1], &["content"]).as_str()),
            (Some("system"), Some("s"), Some("a\nb"))
        ); // (text parts are joined by new lines, as vLLM's chat path joins them)
        let r = request_of(&v, 4096, &Params { temperature: 1.0, ..Params::default() });
        assert_eq!((r.params.temperature, r.params.top_p, r.params.top_k, r.params.seed, r.max_new), (0.5, 0.9, 40, 7, 64));
        assert_eq!((r.stop.clone(), r.thinking), (vec!["x".to_string(), "y".to_string()], Some(false)));
        assert!(r.template_kwargs.is_empty());
        // every other key of chat_template_kwargs is a variable of the template, as given (vLLM's own rule: reasoning_effort, date_string ...)
        let k =
            request_of(&json::parse(r#"{"chat_template_kwargs":{"reasoning_effort":"low","enable_thinking":true,"n":3}}"#).unwrap(), 4096, &Params::default());
        assert_eq!(k.thinking, Some(true));
        assert_eq!(k.template_kwargs, vec![("reasoning_effort".to_string(), json::Value::from("low")), ("n".to_string(), json::Value::from(3u64))]);
        let o = k.template_options();
        assert!(o.add_generation_prompt && o.tools.is_none());
        assert_eq!(o.kwargs.iter().map(|(k, _)| k.as_str()).collect::<Vec<_>>(), ["enable_thinking", "reasoning_effort", "n"]);
        // no limit asked: the window; no temperature: 1 (the API's default)
        let d = request_of(&json::parse("{}").unwrap(), 4096, &Params { temperature: 1.0, ..Params::default() });
        assert_eq!((d.max_new, d.params.temperature, d.ignore_eos), (4096, 1.0, false));
        // vLLM's ignore_eos: a reply of the length asked for
        assert!(request_of(&json::parse(r#"{"prompt":"p","ignore_eos":true,"max_tokens":8}"#).unwrap(), 4096, &Params::default()).ignore_eos);
        assert!(tools::template_messages(&json::parse(r#"{"messages":[]}"#).unwrap()).is_err());
    }

    pub(crate) use crate::sched::tests::Fake;
    use std::sync::atomic::Ordering::Relaxed;

    /// A server on a made-up model (`Fake`: a token after tokens is a hash of them, its row of logits has 10.0 for that token and 5.0 for the next) with a tokenizer whose words are its tokens: what a client of the API sees,
    /// through the real handler, scheduler and sockets, with no GPU.
    pub(crate) fn serve_fake(f: Fake, template: Option<&str>, users: usize) -> std::net::SocketAddr {
        serve_fake_with(f, template, users, false, None, &[])
    }

    pub(crate) fn serve_fake_with(
        f: Fake,
        template: Option<&str>,
        users: usize,
        local_only: bool,
        key: Option<&str>,
        refused: &'static [(&'static str, &'static str)],
    ) -> std::net::SocketAddr {
        serve_fake_in(f, template, users, local_only, key, refused, crate::runtime::context::Context::new())
    }

    /// `serve_fake_with` as the model of the runtime context `rt` (each made-up server has its own: what one counts or fails is not another's).
    pub(crate) fn serve_fake_in(
        f: Fake,
        template: Option<&str>,
        users: usize,
        local_only: bool,
        key: Option<&str>,
        refused: &'static [(&'static str, &'static str)],
        rt: Arc<crate::runtime::context::Context>,
    ) -> std::net::SocketAddr {
        let words: Vec<String> = (1..=1100).map(|i| format!(" w{i}")).collect();
        let refs: Vec<&str> = words.iter().map(String::as_str).collect();
        let window = f.store.len() * crate::kv::PAGE;
        let chat = template.map(|t| Chat::from_template(t).unwrap());
        let shared = Arc::new(Shared {
            tok: crate::tok::tests::toy(&refs),
            chat,
            name: "fake".to_string(),
            defaults: Params { temperature: 1.0, ..Params::default() },
            window,
            vocab: 2048,
            thinking: None,
            refused,
            rt: rt.clone(),
        });
        let (tx, rx) = std::sync::mpsc::channel();
        let life = crate::ops::Life::global();
        let addr =
            listen_in("127.0.0.1:0", "fake", window, key.map(str::to_string), local_only, tx.clone(), Arc::new(AtomicBool::new(false)), life, rt).unwrap();
        std::thread::spawn(move || work_with(&mut Sched::new(f, users), shared, &rx, &tx));
        addr
    }

    /// A request and its whole answer: (status, the body, with a chunked body put together).
    pub(crate) fn call(addr: std::net::SocketAddr, path: &str, body: &str) -> (u16, String) {
        let mut c = TcpStream::connect(addr).unwrap();
        c.write_all(format!("POST {path} HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}", body.len()).as_bytes())
            .unwrap();
        let mut raw = Vec::new();
        c.read_to_end(&mut raw).unwrap();
        let raw = String::from_utf8(raw).unwrap();
        let (head, rest) = raw.split_once("\r\n\r\n").unwrap();
        let status: u16 = head.split_whitespace().nth(1).unwrap().parse().unwrap();
        if !head.to_ascii_lowercase().contains("transfer-encoding: chunked") {
            return (status, rest.to_string());
        }
        let (mut out, mut at) = (String::new(), rest);
        loop {
            let (size, tail) = at.split_once("\r\n").unwrap();
            let n = usize::from_str_radix(size.trim(), 16).unwrap();
            if n == 0 {
                break;
            }
            out.push_str(&tail[..n]);
            at = &tail[n + 2..];
        }
        (status, out)
    }

    /// The JSON of each `data:` event of a streamed answer (`[DONE]` last, as the string it is).
    pub(crate) fn events(body: &str) -> Vec<json::Value> {
        body.split("\n\n")
            .filter_map(|e| e.strip_prefix("data: "))
            .map(|d| if d == "[DONE]" { Value::String("[DONE]".to_string()) } else { json::parse(d).unwrap() })
            .collect()
    }

    pub(crate) fn f64_of(v: &Value) -> f64 {
        match v {
            Value::Number(n) => n.parse().unwrap(),
            other => panic!("{other:?} is not a number"),
        }
    }

    pub(crate) fn at<'a>(v: &'a Value, path: &[&str]) -> &'a Value {
        path.iter().fold(v, |v, k| match k.parse::<usize>() {
            Ok(i) => &v.as_array().unwrap_or_else(|| panic!("no list at {k} in {v:?}"))[i],
            Err(_) => v.get(k).unwrap_or_else(|| panic!("no {k} in {v:?}")),
        })
    }

    /// Log-probabilities as vLLM gives them (`raw_logprobs`): of the model's own row, whatever the draw; a chat's list per token with the likeliest tokens, a plain completion's in the legacy form; the same streamed and not; the end-of-sequence
    /// token's among them and counted; and the refusals in vLLM's words.
    #[test]
    fn log_probabilities_through_the_api() {
        let lse = (10f64.exp() + 5f64.exp() + 2046.0 * (-5f64).exp()).ln();
        for gpu_top in [true, false] {
            let mut f = Fake::new(64, 32);
            f.gpu_top = gpu_top;
            let addr = serve_fake(f, None, 4);
            let q = r#""messages":[{"role":"user","content":"hello there"}],"max_tokens":5,"temperature":0"#;
            let (st, body) = call(addr, "/v1/chat/completions", &format!("{{{q},\"logprobs\":true,\"top_logprobs\":2}}"));
            assert_eq!(st, 200, "{body}");
            let d = json::parse(&body).unwrap();
            let content = at(&d, &["choices", "0", "logprobs", "content"]).as_array().unwrap().to_vec();
            assert_eq!(content.len(), 5, "{body}");
            assert_eq!(at(&d, &["usage", "completion_tokens"]), &Value::Number("5".into()));
            for e in &content {
                // the likeliest token is the chosen one (greedy), 10.0 of the row; the second is the token after it, 5.0
                assert!((f64_of(at(e, &["logprob"])) - (10.0 - lse)).abs() < 1e-6, "gpu_top {gpu_top}: {e:?}");
                let top = at(e, &["top_logprobs"]).as_array().unwrap();
                assert_eq!(top.len(), 2);
                assert_eq!(at(&top[0], &["token"]), at(e, &["token"]));
                assert!((f64_of(at(&top[1], &["logprob"])) - (5.0 - lse)).abs() < 1e-6);
                assert!(at(e, &["bytes"]).as_array().unwrap().len() >= 2);
            }
            let tokens: Vec<String> = content.iter().map(|e| at(e, &["token"]).as_str().unwrap().to_string()).collect();
            assert_eq!(at(&d, &["choices", "0", "message", "content"]).as_str().unwrap(), tokens.concat(), "the text is the tokens'");
            // `logprobs: true` alone lists the chosen token and no others
            let (_, body) = call(addr, "/v1/chat/completions", &format!("{{{q},\"logprobs\":true}}"));
            let d = json::parse(&body).unwrap();
            assert_eq!(at(&d, &["choices", "0", "logprobs", "content", "0", "top_logprobs"]), &Value::Array(vec![]));
            // none asked for: none sent
            let (_, body) = call(addr, "/v1/chat/completions", &format!("{{{q}}}"));
            assert_eq!(at(&json::parse(&body).unwrap(), &["choices", "0", "logprobs"]), &Value::Null);
            // streamed: the same entries, in the chunks that carry their text, the last with the finish reason
            let (st, body) = call(addr, "/v1/chat/completions", &format!("{{{q},\"logprobs\":true,\"top_logprobs\":2,\"stream\":true}}"));
            assert_eq!(st, 200);
            let ev = events(&body);
            assert_eq!(ev.last().unwrap(), &Value::String("[DONE]".into()));
            let mut streamed = Vec::new();
            for e in &ev[..ev.len() - 1] {
                if let Some(l) =
                    e.get("choices").and_then(|c| c.as_array()).and_then(|c| c.first()).and_then(|c| c.get("logprobs")).and_then(|l| l.get("content"))
                {
                    streamed.extend(l.as_array().unwrap().iter().map(|x| at(x, &["token"]).as_str().unwrap().to_string()));
                }
            }
            assert_eq!(streamed, tokens, "gpu_top {gpu_top}");
            // a plain completion: the legacy lists
            let (st, body) = call(addr, "/v1/completions", r#"{"prompt":"hello there","max_tokens":4,"temperature":0,"logprobs":2}"#);
            assert_eq!(st, 200, "{body}");
            let d = json::parse(&body).unwrap();
            let l = at(&d, &["choices", "0", "logprobs"]);
            assert_eq!(at(l, &["tokens"]).as_array().unwrap().len(), 4);
            let offsets: Vec<f64> = at(l, &["text_offset"]).as_array().unwrap().iter().map(f64_of).collect();
            assert_eq!(offsets[0], 0.0);
            assert!(offsets.windows(2).all(|w| w[1] > w[0]));
            assert!((f64_of(at(l, &["token_logprobs", "0"])) - (10.0 - lse)).abs() < 1e-6);
            assert_eq!(at(l, &["top_logprobs", "0"]).as_object().unwrap().len(), 2);
            // the refusals, in vLLM's words
            for (path, extra, want) in [
                ("/v1/chat/completions", r#""logprobs":true,"top_logprobs":21"#, "greater than max allowed: 20"),
                ("/v1/chat/completions", r#""top_logprobs":2"#, "`logprobs` must be set to true"),
                ("/v1/chat/completions", r#""logprobs":true,"top_logprobs":-3"#, "must be a positive value or -1"),
                ("/v1/chat/completions", r#""logprobs":true,"top_logprobs":"two""#, "must be an integer"),
                ("/v1/completions", r#""logprobs":21,"prompt":"x""#, "greater than max allowed: 20"),
            ] {
                let (st, body) = call(addr, path, &format!("{{\"messages\":[{{\"role\":\"user\",\"content\":\"x\"}}],{extra}}}"));
                assert!(st == 400 && body.contains(want), "{extra}: {st} {body}");
            }
        }
    }

    /// The end-of-sequence token is a token of the reply with a log-probability of its own, and counted in the usage, as vLLM counts it; the text has none of it.
    #[test]
    fn the_end_token_is_counted_and_has_log_probabilities() {
        let mut probe = Fake::new(64, 32);
        probe.eos = vec![0];
        let addr = serve_fake(probe, None, 4);
        let q = r#""messages":[{"role":"user","content":"hello there"}],"temperature":0"#;
        let (_, body) = call(addr, "/v1/chat/completions", &format!("{{{q},\"max_tokens\":4,\"logprobs\":true}}"));
        let d = json::parse(&body).unwrap();
        let first_four: Vec<String> =
            at(&d, &["choices", "0", "logprobs", "content"]).as_array().unwrap().iter().map(|e| at(e, &["token"]).as_str().unwrap().to_string()).collect();
        // the model's fourth token is an end token for a server whose model ends there: the same prompt, the end token among the model's
        let fourth_id: usize = first_four[3].trim().trim_start_matches('w').parse().unwrap();
        let mut f = Fake::new(64, 32);
        f.eos = vec![fourth_id as i64];
        let addr = serve_fake(f, None, 4);
        let (st, body) = call(addr, "/v1/chat/completions", &format!("{{{q},\"max_tokens\":50,\"logprobs\":true}}"));
        assert_eq!(st, 200, "{body}");
        let d = json::parse(&body).unwrap();
        assert_eq!(at(&d, &["choices", "0", "finish_reason"]).as_str(), Some("stop"));
        let list = at(&d, &["choices", "0", "logprobs", "content"]).as_array().unwrap();
        let text = at(&d, &["choices", "0", "message", "content"]).as_str().unwrap();
        // (the first three tokens, or fewer where the end token shows earlier)
        let n_text = text.matches('w').count();
        assert_eq!(list.len(), n_text + 1, "the end token's own entry comes last: {body}");
        assert_eq!(at(&d, &["usage", "completion_tokens"]), &Value::Number((n_text + 1).to_string()), "the end token is counted");
        assert!(at(list.last().unwrap(), &["token"]).as_str().unwrap().trim() == format!("w{fourth_id}"));
    }

    /// Several choices from one prompt: each with its own index and finish reason and its own draws (the seed of choice i is the request's plus i, as vLLM's; none given, each its own), the prompt run once and its
    /// blocks found in the cache by the others, the usage counting the prompt once and every choice's tokens, the same streamed and whole, and the limits in vLLM's words.
    #[test]
    fn several_choices_from_one_prompt() {
        let f = Fake::new(64, 32);
        let ran = f.prefilled.clone();
        let addr = serve_fake(f, None, 4);
        let prompt = "w ".repeat(100); // 100 tokens
        let q = |extra: &str| format!("{{\"prompt\":\"{prompt}\",\"max_tokens\":12,\"temperature\":2.0,\"top_k\":20{extra}}}");
        let texts = |body: &str| -> Vec<String> {
            at(&json::parse(body).unwrap(), &["choices"]).as_array().unwrap().iter().map(|c| at(c, &["text"]).as_str().unwrap().to_string()).collect()
        };
        let before = ran.load(Relaxed);
        let (st, body) = call(addr, "/v1/completions", &q(",\"n\":4,\"seed\":7"));
        assert_eq!(st, 200, "{body}");
        // the first choice runs the prompt (100 tokens); each of the three others finds its six full blocks (96 tokens) and runs the last four
        assert_eq!(ran.load(Relaxed) - before, 100 + 3 * 4, "the prompt was not shared");
        let d = json::parse(&body).unwrap();
        let cs = at(&d, &["choices"]).as_array().unwrap();
        assert_eq!(cs.len(), 4);
        for (i, c) in cs.iter().enumerate() {
            assert_eq!(at(c, &["index"]), &Value::Number(i.to_string()));
            assert_eq!(at(c, &["finish_reason"]).as_str(), Some("length"));
        }
        assert_eq!(
            (at(&d, &["usage", "prompt_tokens"]), at(&d, &["usage", "completion_tokens"]), at(&d, &["usage", "total_tokens"])),
            (&Value::Number("100".into()), &Value::Number("48".into()), &Value::Number("148".into())),
            "{body}"
        );
        let t = texts(&body);
        assert!(t.iter().any(|x| x != &t[0]), "four choices of one text: {t:?}");
        // the same seed, the same four texts; another seed, other texts
        assert_eq!(texts(&call(addr, "/v1/completions", &q(",\"n\":4,\"seed\":7")).1), t);
        assert_ne!(texts(&call(addr, "/v1/completions", &q(",\"n\":4,\"seed\":8")).1), t);
        // choice i alone, with the seed 7 + i, is the choice i (the seeds are what makes the choices apart)
        for i in 0..4u64 {
            assert_eq!(texts(&call(addr, "/v1/completions", &q(&format!(",\"seed\":{}", 7 + i))).1), vec![t[i as usize].clone()], "choice {i}");
        }
        // no seed: the choices differ from each other, and a second request from the first (a draw: here a choice is all the likeliest
        // tokens about one time in three, so three choices alike come about one request in 25; up to 5 requests are made for one that differs)
        let draw = || texts(&call(addr, "/v1/completions", &q(",\"n\":3")).1);
        let ta = (0..5).map(|_| draw()).find(|t| t.iter().any(|x| x != &t[0]));
        let tb = draw();
        assert!(ta.as_ref().is_some_and(|ta| ta != &tb), "{ta:?} {tb:?}");
        // streamed: a role chunk for each choice first, then deltas that carry their index, each choice ended once, the usage, [DONE]; each choice's text is the whole answer's
        let chat = r#""messages":[{"role":"user","content":"hello there"}],"max_tokens":12,"temperature":2.0,"top_k":20,"n":3,"seed":11"#;
        let (st, body) = call(addr, "/v1/chat/completions", &format!("{{{chat},\"stream\":true,\"stream_options\":{{\"include_usage\":true}}}}"));
        assert_eq!(st, 200, "{body}");
        let ev = events(&body);
        assert_eq!(ev.last().unwrap(), &Value::String("[DONE]".into()));
        let first: Vec<String> = ev[..3]
            .iter()
            .map(|e| format!("{}:{}", f64_of(at(e, &["choices", "0", "index"])), at(e, &["choices", "0", "delta", "role"]).as_str().unwrap()))
            .collect();
        assert_eq!(first, vec!["0:assistant", "1:assistant", "2:assistant"]);
        let (mut built, mut finishes) = (vec![String::new(); 3], vec![0; 3]);
        for e in &ev[3..ev.len() - 2] {
            let c = at(e, &["choices", "0"]);
            let i: usize = match at(c, &["index"]) {
                Value::Number(n) => n.parse().unwrap(),
                _ => unreachable!(),
            };
            if let Some(t) = c.get("delta").and_then(|d| d.get("content")).and_then(Value::as_str) {
                built[i].push_str(t);
            }
            if c.get("finish_reason").and_then(Value::as_str).is_some() {
                finishes[i] += 1;
            }
        }
        assert_eq!(finishes, vec![1, 1, 1]);
        let usage = ev[ev.len() - 2].get("usage").expect("a usage chunk before [DONE]");
        assert_eq!((at(usage, &["completion_tokens"]), ev[ev.len() - 2].get("choices").unwrap()), (&Value::Number("36".into()), &Value::Array(vec![])));
        let (_, whole) = call(addr, "/v1/chat/completions", &format!("{{{chat}}}"));
        let whole: Vec<String> = at(&json::parse(&whole).unwrap(), &["choices"])
            .as_array()
            .unwrap()
            .iter()
            .map(|c| at(c, &["message", "content"]).as_str().unwrap().to_string())
            .collect();
        assert_eq!(built, whole, "streamed and whole choices are the same texts");
        // the limits: 8 at the most (and 8 are served by 4 users, 4 waiting), none or fewer than one is wrong, and greedy draws are all one text
        let (st, body) = call(addr, "/v1/completions", &q(",\"n\":8"));
        assert_eq!((st, texts(&body).len()), (200, 8), "{body}");
        for (extra, want) in [
            (",\"n\":9", "n must be at most 8, got 9"),
            (",\"n\":0", "n must be at least 1, got 0"),
            (",\"n\":2.5", "`n` must be an integer"),
            (",\"n\":2,\"temperature\":0", "n must be 1 when using greedy sampling, got 2"),
        ] {
            let (st, body) = call(addr, "/v1/completions", &format!("{{\"prompt\":\"{prompt}\",\"max_tokens\":4{extra}}}"));
            assert!(st == 400 && body.contains(want), "{extra}: {st} {body}");
        }
    }

    /// A request as text, and the whole answer: (status, the header block in lower case, the body).
    fn ask(addr: std::net::SocketAddr, method: &str, path: &str, headers: &[(&str, &str)]) -> (u16, String, String) {
        let mut c = TcpStream::connect(addr).unwrap();
        let h: String = headers.iter().map(|(k, v)| format!("{k}: {v}\r\n")).collect();
        c.write_all(format!("{method} {path} HTTP/1.1\r\n{h}Connection: close\r\n\r\n").as_bytes()).unwrap();
        let mut raw = Vec::new();
        c.read_to_end(&mut raw).unwrap();
        let raw = String::from_utf8(raw).unwrap();
        let (head, body) = raw.split_once("\r\n\r\n").unwrap();
        (head.split_whitespace().nth(1).unwrap().parse().unwrap(), head.to_lowercase(), body.to_string())
    }

    /// The server on this computer's own address, as the Python server's page guard was (glyd-gpu-py/glyd_gpu/page.py, its test in test_onboard.py): the Host must be localhost, 127.0.0.1 or [::1] (421 for another name:
    /// a page whose DNS points at this computer, the page too), the Origin the server's own, `http://` and the Host (403, another page's script, a preflight too); programs that send no Origin and an address as
    /// their Host (curl, the OpenAI libraries, Open WebUI's server) are answered, and so is the server's own page.
    #[test]
    fn this_computers_own_address_answers_its_own_page_and_programs() {
        let (tx, _rx) = std::sync::mpsc::channel();
        let addr = listen("127.0.0.1:0", "m", 4096, None, true, tx).unwrap();
        let status = |method: &str, path: &str, h: &[(&str, &str)]| ask(addr, method, path, h).0;
        for host in ["localhost:8000", "127.0.0.1:8000", "[::1]:8000", "LocalHost:8000", "localhost", "[::1]", "localhost:"] {
            assert_eq!(status("GET", "/v1/models", &[("Host", host)]), 200, "{host}");
            assert_eq!(status("GET", "/health", &[("Host", host)]), 200, "{host}");
        }
        // the names a container has for this computer (Open WebUI in Docker Desktop reaches the server by one): answered; the Origin is judged as for any Host
        for host in [
            "host.docker.internal:8000",
            "host.containers.internal:8000",
            "Host.Docker.Internal",
            "host.docker.internal.evil.example:8000",
            "xhost.docker.internal:8000",
        ] {
            let want = if host.to_ascii_lowercase().contains("evil") || host.starts_with('x') { 421 } else { 200 };
            assert_eq!(status("GET", "/v1/models", &[("Host", host)]), want, "{host}");
        }
        assert_eq!(status("GET", "/v1/models", &[("Host", "host.docker.internal:8000"), ("Origin", "http://host.docker.internal:8000")]), 200);
        assert_eq!(status("GET", "/v1/models", &[("Host", "host.docker.internal:8000"), ("Origin", "http://evil.example")]), 403);
        // (HTTP/1.0 and a client with no Host: nothing to refuse for it)
        assert_eq!(status("GET", "/v1/models", &[]), 200);
        // the page's own requests: the Origin is `http://` and the Host, in any case
        assert_eq!(status("GET", "/v1/models", &[("Host", "localhost:8000"), ("Origin", "http://localhost:8000")]), 200);
        assert_eq!(status("GET", "/v1/models", &[("Host", "127.0.0.1:8000"), ("Origin", "http://127.0.0.1:8000")]), 200);
        assert_eq!(status("GET", "/v1/models", &[("Host", "localhost:8000"), ("Origin", "HTTP://LOCALHOST:8000")]), 200);
        assert_eq!(status("GET", "/", &[("Host", "localhost:8000")]), 200);
        // a name that is not this computer's: the rebinding page's own, whatever the path (the page is not answered to it either)
        for host in [
            "evil.example",
            "evil.example:8000",
            "localhost.evil.example:8000",
            "127.0.0.1.evil.example",
            "192.168.1.5:8000",
            "0.0.0.0:8000",
            "localhost.:8000",
            "[::1",
            "[::2]:8000",
            "[fe80::1]:8000",
        ] {
            for path in ["/v1/models", "/", "/health", "/stats"] {
                let (st, _, body) = ask(addr, "GET", path, &[("Host", host)]);
                assert_eq!(st, 421, "{host} {path}");
                let e = json::parse(&body).unwrap();
                assert_eq!((at(&e, &["error", "type"]).as_str(), f64_of(at(&e, &["error", "code"]))), (Some("Forbidden"), 421.0));
                assert!(at(&e, &["error", "message"]).as_str().unwrap().contains(&format!("{:?}", host).replace('"', "")), "{body}");
            }
        }
        // an Origin that is not the server's own: another site's script, another port, another scheme, another way of writing the address, `null`, a name that begins as the Host
        for origin in [
            "http://evil.example",
            "https://localhost:8000",
            "http://localhost:5173",
            "http://127.0.0.1:8000",
            "null",
            "http://localhost:8000.evil.example",
            "http://",
        ] {
            let (st, _, body) = ask(addr, "POST", "/v1/chat/completions", &[("Host", "localhost:8000"), ("Origin", origin), ("Content-Length", "0")]);
            assert_eq!(st, 403, "{origin}");
            assert_eq!(at(&json::parse(&body).unwrap(), &["error", "type"]).as_str(), Some("Forbidden"));
            assert_eq!(status("GET", "/v1/models", &[("Host", "localhost:8000"), ("Origin", origin)]), 403, "{origin}");
        }
        // an Origin with no Host to be the origin of
        assert_eq!(status("GET", "/v1/models", &[("Origin", "http://localhost:8000")]), 403);
        // the Host is judged first: a name that is not ours is 421 whatever the Origin
        assert_eq!(status("GET", "/v1/models", &[("Host", "evil.example"), ("Origin", "http://evil.example")]), 421);
        // the preflight of another site's script is refused (the page's own requests have none)
        let pre = [("Host", "localhost:8000"), ("Origin", "http://evil.example"), ("Access-Control-Request-Method", "POST")];
        assert_eq!(status("OPTIONS", "/v1/chat/completions", &pre), 403);
        assert_eq!(
            status(
                "OPTIONS",
                "/v1/chat/completions",
                &[("Host", "evil.example"), ("Origin", "http://evil.example"), ("Access-Control-Request-Method", "POST")]
            ),
            421
        );
        // a preflight of the server's own origin, as the CORS middleware of vLLM answers it (the origins are the three addresses at the server's port); one of the right form for the Host but not this port, or for a method
        // there is none of, is 400
        let own = format!("http://localhost:{}", addr.port());
        let host = format!("localhost:{}", addr.port());
        let (st, head, body) = ask(
            addr,
            "OPTIONS",
            "/v1/chat/completions",
            &[("Host", &host), ("Origin", &own), ("Access-Control-Request-Method", "POST"), ("Access-Control-Request-Headers", "content-type,authorization")],
        );
        assert_eq!((st, body.as_str()), (200, "OK"));
        for line in [
            format!("access-control-allow-origin: {}", own.to_lowercase()),
            "access-control-allow-methods: delete, get, head, options, patch, post, put".to_string(),
            "access-control-allow-headers: content-type,authorization".to_string(),
            "access-control-max-age: 600".to_string(),
            "vary: origin".to_string(),
        ] {
            assert!(head.contains(&line), "{line} in {head}");
        }
        let (st, head, body) = ask(
            addr,
            "OPTIONS",
            "/v1/chat/completions",
            &[("Host", "localhost:8000"), ("Origin", "http://localhost:8000"), ("Access-Control-Request-Method", "POST")],
        );
        assert_eq!((st, body.as_str(), head.contains("access-control-allow-origin")), (400, "Disallowed CORS origin", false));
        let (st, _, body) = ask(addr, "OPTIONS", "/v1/chat/completions", &[("Host", &host), ("Origin", &own), ("Access-Control-Request-Method", "TRACE")]);
        assert_eq!((st, body.as_str()), (400, "Disallowed CORS method"));
        // an OPTIONS with no question is an ordinary request (an API path does not take it)
        assert_eq!(status("OPTIONS", "/v1/chat/completions", &[("Host", &host)]), 405);
        // the answers to the server's own origin carry its CORS line; those to a program that sends no Origin carry none
        let (_, head, _) = ask(addr, "GET", "/v1/models", &[("Host", &host), ("Origin", &own)]);
        assert!(head.contains(&format!("access-control-allow-origin: {}", own.to_lowercase())), "{head}");
        assert!(!ask(addr, "GET", "/v1/models", &[("Host", &host)]).1.contains("access-control"));
        // the page: its own headers (it may talk to its own origin only, is not framed, not sniffed), and HEAD answers them with no body
        let (st, head, body) = ask(addr, "GET", "/", &[("Host", &host)]);
        assert_eq!(st, 200);
        assert!(body.contains("<html") || body.contains("<!DOCTYPE") || body.contains("<!doctype"));
        for want in ["content-security-policy: default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'self'; img-src data:; base-uri 'none'; form-action 'none'; frame-ancestors 'none'", "x-content-type-options: nosniff", "referrer-policy: no-referrer"] {
            assert!(head.contains(want), "{want} in {head}");
        }
        let (st, head, body) = ask(addr, "HEAD", "/", &[("Host", &host)]);
        assert_eq!((st, body.as_str(), head.contains("content-length")), (200, "", true));
    }

    /// A server that is open to the network (a bound address that is not this computer's own, which it needs a key or `--insecure-no-auth` to start on): the Host is any name the computer has, but CORS is off, there is
    /// no `*` anywhere: a script of another origin is refused (403, a preflight too), and only the page's own origin (`http(s)://` and the Host) is answered.
    #[test]
    fn an_open_server_has_cors_off() {
        let (tx, _rx) = std::sync::mpsc::channel();
        let addr = listen("127.0.0.1:0", "m", 4096, None, false, tx).unwrap();
        let (st, head, _) = ask(addr, "GET", "/v1/models", &[("Host", "192.168.1.5:8000")]);
        assert_eq!(st, 200);
        assert!(!head.contains("access-control"), "{head}");
        assert_eq!(ask(addr, "GET", "/v1/models", &[("Host", "glyd.example.com"), ("Origin", "http://glyd.example.com")]).0, 200);
        assert_eq!(ask(addr, "GET", "/v1/models", &[("Host", "glyd.example.com"), ("Origin", "https://glyd.example.com")]).0, 200); // (behind a proxy that ends TLS)
        for origin in ["http://evil.example", "null", "http://localhost:5173"] {
            let (st, head, _) = ask(addr, "GET", "/v1/models", &[("Host", "192.168.1.5:8000"), ("Origin", origin)]);
            assert_eq!(st, 403, "{origin}");
            assert!(!head.contains("access-control-allow-origin"), "{head}");
        }
        let pre = [
            ("Host", "x:1"),
            ("Origin", "http://localhost:5173"),
            ("Access-Control-Request-Method", "POST"),
            ("Access-Control-Request-Headers", "authorization"),
        ];
        assert_eq!(ask(addr, "OPTIONS", "/v1/chat/completions", &pre).0, 403);
    }

    /// What the user lets in is exactly what they list: a page's origin gets CORS headers for itself (never `*`), a name gets past the Host check, and nothing else changes.
    #[test]
    fn what_the_user_lets_in_is_what_they_list() {
        let allowed = parse_access(Some("http://localhost:5173, HTTPS://App.Example.com"), Some("glyd.example.com, GLYD.lan:8000")).unwrap();
        assert_eq!(allowed.origins, ["http://localhost:5173", "https://app.example.com"]);
        assert_eq!(allowed.hosts, ["glyd.example.com", "glyd.lan"]);
        // on this computer's own address: the listed origin passes the Origin check and is answered for itself; an unlisted one is not
        assert_eq!(refusal_with("localhost:8000", "http://localhost:5173", true, &allowed), None);
        assert_eq!(refusal_with("localhost:8000", "https://app.example.com", true, &allowed), None);
        assert_eq!(refusal_with("localhost:8000", "http://evil.example", true, &allowed).unwrap().0, 403);
        assert_eq!(cors_headers_with(true, 8000, "http://localhost:5173", &allowed), "Access-Control-Allow-Origin: http://localhost:5173\r\nVary: Origin\r\n");
        assert_eq!(cors_headers_with(true, 8000, "http://evil.example", &allowed), "");
        assert_eq!(cors_headers_with(false, 8000, "http://evil.example", &allowed), "");
        assert_eq!(cors_headers_with(false, 8000, "http://localhost:5173", &allowed), "Access-Control-Allow-Origin: http://localhost:5173\r\nVary: Origin\r\n");
        // the Host: the listed names join the program's own; a name nobody listed is still 421
        assert_eq!(refusal_with("glyd.example.com:8000", "", true, &allowed), None);
        assert_eq!(refusal_with("glyd.lan", "", true, &allowed), None);
        assert_eq!(refusal_with("evil.example", "", true, &allowed).unwrap().0, 421);
        // the open server: any Host where none are listed; where some are, those and the computer's own
        assert_eq!(refusal_with("192.168.1.5:8000", "", false, &Access::default()), None);
        assert_eq!(refusal_with("glyd.example.com", "", false, &allowed), None);
        assert_eq!(refusal_with("localhost", "", false, &allowed), None);
        assert_eq!(refusal_with("192.168.1.5:8000", "", false, &allowed).unwrap().0, 421);
        // `*`, a path and a bare scheme are not origins
        for bad in ["*", "http://*", "http://a/b", "localhost:5173", "http://", "ftp://a", "http://a b"] {
            assert!(parse_access(Some(bad), None).is_err(), "{bad}");
        }
        assert!(parse_access(Some("http://localhost:3000,*"), None).is_err());
        for bad in ["a/b", "*.example.com", "http://a", "u@h"] {
            assert!(parse_access(None, Some(bad)).is_err(), "{bad}");
        }
        assert_eq!(parse_access(None, None).unwrap(), Access::default());
    }

    /// The defaults themselves: nothing is let in unless it is listed (no origin, no extra name), a server starts on this computer's own address, and an address that is not needs a key or the user's word.
    #[test]
    fn the_defaults_let_nothing_in() {
        assert_eq!(*access(), Access::default());
        for own in ["127.0.0.1", "localhost", "LOCALHOST", "[::1]", "::1", "127.1.2.3"] {
            assert!(is_loopback_addr(own), "{own}");
            assert_eq!(bind_refusal(own, false, false), None, "{own}");
        }
        for open in ["0.0.0.0", "::", "[::]", "192.168.1.5", "10.0.0.2", "example.com", "fe80::1", ""] {
            assert!(!is_loopback_addr(open), "{open}");
            let why = bind_refusal(open, false, false).unwrap_or_else(|| panic!("{open} was let start with no key"));
            assert!(why.contains("--insecure-no-auth") && why.contains("GLYD_API_KEY"), "{why}");
            assert_eq!(bind_refusal(open, true, false), None, "{open} with a key");
            assert_eq!(bind_refusal(open, false, true), None, "{open} with the user's word");
        }
        // the names a container has for this computer and its programs (Open WebUI in Docker Desktop, Kubernetes, WSL's Docker integration) are answered at the default address
        for host in [
            "host.docker.internal:3000",
            "gateway.docker.internal",
            "kubernetes.docker.internal",
            "host.containers.internal",
            "host.lima.internal",
            "host.minikube.internal",
        ] {
            assert_eq!(refusal_with(host, "", true, &Access::default()), None, "{host}");
        }
        assert_eq!(refusal_with("127.0.0.2:1", "", true, &Access::default()), None);
        // no CORS answer to any origin by default, the server's own page aside
        assert_eq!(cors_headers_with(true, 8000, "http://evil.example", &Access::default()), "");
        assert_eq!(cors_headers_with(false, 8000, "http://localhost:8000", &Access::default()), "");
        assert!(cors_headers_with(true, 8000, "http://localhost:8000", &Access::default()).contains("http://localhost:8000"));
    }

    /// The process's limit of open files raised (each test connection is two: a macOS default of 256 holds no thousand connections).
    #[cfg(unix)]
    pub(crate) fn raise_fd_limit(want: u64) {
        #[repr(C)]
        struct Rlimit {
            cur: u64,
            max: u64,
        }
        extern "C" {
            fn getrlimit(resource: i32, r: *mut Rlimit) -> i32;
            fn setrlimit(resource: i32, r: *const Rlimit) -> i32;
        }
        #[cfg(target_os = "macos")]
        const NOFILE: i32 = 8;
        #[cfg(not(target_os = "macos"))]
        const NOFILE: i32 = 7;
        let mut r = Rlimit { cur: 0, max: 0 };
        // SAFETY: getrlimit and setrlimit read and write the struct, which has the layout of rlimit on the 64-bit Unixes the tests run on.
        unsafe {
            if getrlimit(NOFILE, &mut r) == 0 && r.cur < want {
                r.cur = want.min(if r.max == 0 { want } else { r.max });
                setrlimit(NOFILE, &r);
            }
        }
    }

    #[cfg(not(unix))]
    pub(crate) fn raise_fd_limit(_: u64) {}

    /// A server on a free port with the times given, for the tests of what the front end does with a slow client.
    fn listen_timed(times: Times, tx: Sender<Job>) -> std::net::SocketAddr {
        frontend::serve("127.0.0.1:0", move |port| Ctx {
            name: Arc::new("m".into()),
            window: 4096,
            key: Arc::new(None),
            local_only: true,
            port,
            tx,
            loading: Arc::new(AtomicBool::new(false)),
            life: Arc::default(),
            times,
            rt: crate::runtime::context::Context::new(),
        })
        .unwrap()
    }

    /// What a server answers a request written to it in pieces: (the status, the whole of what it sent), read to its end.
    fn raw_status(c: &mut TcpStream) -> (u16, String) {
        c.set_read_timeout(Some(Duration::from_secs(10))).unwrap();
        let mut out = String::new();
        let _ = c.read_to_string(&mut out);
        (out.split_whitespace().nth(1).and_then(|c| c.parse().ok()).unwrap_or(0), out)
    }

    /// A step that fails as a call of the driver does. With a status after which the GPU's context is lost (700: an illegal address) the request is told, the server says it failed, and it
    /// ends itself (`device_lost`: a test's server says so in `LOST`, a real one ends the process with `ui::DEVICE_LOST` and the service starts the model again). With a status that leaves the
    /// context whole (1: an invalid value) the request is told and the server goes on: the next request is answered.
    #[test]
    fn a_sticky_device_error_ends_the_server_and_another_error_does_not() {
        let ask = |addr: std::net::SocketAddr| {
            let body = r#"{"messages":[{"role":"user","content":"hello there"}],"max_tokens":6,"temperature":0}"#;
            let mut c = TcpStream::connect(addr).unwrap();
            c.write_all(
                format!(
                    "POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nConnection: close\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
                    body.len()
                )
                .as_bytes(),
            )
            .unwrap();
            raw_status(&mut c)
        };
        let lost = |status: i32| LOST.lock().unwrap_or_else(|e| e.into_inner()).iter().filter(|m| m.ends_with(&format!("failed ({status})"))).count();
        // an invalid value: this request fails, the server lives
        let mut f = Fake::new(512, 64);
        f.fault = Some((2, 1));
        let addr = serve_fake(f, None, 4);
        let (status, text) = ask(addr);
        assert!(status >= 500 && text.contains("failed (1)"), "{status}: {text}");
        assert_eq!(lost(1), 0);
        assert_eq!(ask(addr).0, 200, "the server answers the next request");
        // an illegal address: the request fails, and the server ends itself
        let mut f = Fake::new(512, 64);
        f.fault = Some((2, 700));
        let addr = serve_fake(f, None, 4);
        let (status, text) = ask(addr);
        assert!(status >= 500 && text.contains("failed (700)"), "{status}: {text}");
        let t = std::time::Instant::now();
        while lost(700) == 0 && t.elapsed() < Duration::from_secs(5) {
            std::thread::sleep(Duration::from_millis(10));
        }
        assert_eq!(lost(700), 1, "the server ended itself for the lost context, once");
    }

    /// Two models in one process (R16): each server has its own runtime context, so its scheduler's figures, its counts and its engine's failure are its own. Three requests to one and a
    /// request that fails to the other: the first one's /stats and /metrics count its three and not the other's, the second one's /v1/glyd/info says it failed and the first one's that it is
    /// ready, and the process's own context has seen none of it.
    #[test]
    fn two_models_in_one_process_have_their_own_figures_and_failures() {
        use crate::runtime::context::Context;
        let ask = |addr: std::net::SocketAddr| {
            let body = r#"{"messages":[{"role":"user","content":"hello there"}],"max_tokens":6,"temperature":0}"#;
            let mut c = TcpStream::connect(addr).unwrap();
            c.write_all(
                format!(
                    "POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nConnection: close\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
                    body.len()
                )
                .as_bytes(),
            )
            .unwrap();
            raw_status(&mut c).0
        };
        let (rt_a, rt_b) = (Context::new(), Context::new());
        let mut failing = Fake::new(512, 64);
        failing.fault = Some((2, 1));
        let a = serve_fake_in(Fake::new(512, 64), None, 4, false, None, &[], rt_a.clone());
        let b = serve_fake_in(failing, None, 4, false, None, &[], rt_b.clone());
        for rt in [&rt_a, &rt_b] {
            assert!(rt
                .info
                .set(Info { about: Vec::new(), gpu: None, held: None, concurrency: 4, window: 512 * crate::kv::PAGE, refused: &[], started: 0.0 })
                .is_ok());
        }
        let before = Context::process().sched[0].load(Ordering::Relaxed);
        assert_eq!((ask(a), ask(a), ask(a)), (200, 200, 200));
        assert_eq!(ask(b), 500);
        let num = |text: &str, key: &str| -> f64 {
            let v = json::parse(text.split_once("\r\n\r\n").map_or(text, |x| x.1)).unwrap();
            f64_of(at(&v, &[key]))
        };
        // the schedulers' figures: prompt tokens run
        let (sa, sb) = (get(a, "/stats", None), get(b, "/stats", None));
        let (pa, pb) = (num(&sa, "prompt_tokens"), num(&sb, "prompt_tokens"));
        assert!(pb > 0.0 && pa == 3.0 * pb, "{pa} prompt tokens of three requests, {pb} of one");
        // the counts: tokens written to replies (six a reply of the first; the second's reply failed at its second step)
        let made = |text: &str| -> f64 {
            text.lines().find_map(|l| l.strip_prefix("glyd_generated_tokens_total ")).and_then(|v| v.trim().parse().ok()).expect("glyd_generated_tokens_total")
        };
        let (ma, mb) = (made(&get(a, "/metrics", None)), made(&get(b, "/metrics", None)));
        assert!(ma == 18.0 && mb < 6.0, "{ma} tokens of the first model's replies, {mb} of the second's");
        // the failure is the second model's alone
        let state = |addr| {
            let t = get(addr, "/v1/glyd/info", None);
            let v = json::parse(t.split_once("\r\n\r\n").map_or(t.as_str(), |x| x.1)).unwrap();
            at(&v, &["state"]).as_str().unwrap_or("").to_string()
        };
        assert_eq!((state(a), state(b)), ("ready".to_string(), "failed".to_string()));
        assert!(rt_a.failed.lock().unwrap().is_none() && rt_b.failed.lock().unwrap().is_some());
        assert_eq!(Context::process().sched[0].load(Ordering::Relaxed), before, "the process's own context counts none of their prompts");
    }

    /// A client that sends its request slowly is let wait for the time that is given the whole of it, not for each read: a request that stops, or goes a byte at a time, is answered 408 and its
    /// connection is let go; a whole request, sent at once, is answered.
    #[test]
    fn a_slow_request_is_timed_out_as_a_whole() {
        let (tx, _rx) = std::sync::mpsc::channel();
        let addr = listen_timed(Times { head: Duration::from_millis(300), body: Duration::from_millis(300), idle: Duration::from_secs(5) }, tx);
        // a head that never ends
        let mut c = TcpStream::connect(addr).unwrap();
        c.write_all(b"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\n").unwrap();
        let t = std::time::Instant::now();
        assert_eq!(raw_status(&mut c).0, 408);
        assert!(t.elapsed() < Duration::from_secs(3), "{:?}", t.elapsed());
        // a head that goes a byte every 100 ms: each read is in time, the whole is not
        let mut c = TcpStream::connect(addr).unwrap();
        let mut w = c.try_clone().unwrap();
        let drip = std::thread::spawn(move || {
            for b in b"GET /v1/models HTTP/1.1\r\nHost: localhost\r\nX-Slow: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".iter() {
                if w.write_all(&[*b]).is_err() {
                    break;
                }
                std::thread::sleep(Duration::from_millis(100));
            }
        });
        let t = std::time::Instant::now();
        assert_eq!(raw_status(&mut c).0, 408);
        assert!(t.elapsed() < Duration::from_secs(3), "{:?}", t.elapsed());
        drop(c);
        drip.join().unwrap();
        // a body that stops short of its Content-Length
        let mut c = TcpStream::connect(addr).unwrap();
        c.write_all(b"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Length: 100\r\n\r\n{\"a\"").unwrap();
        assert_eq!(raw_status(&mut c).0, 408);
        // a whole request, sent at once, is answered
        let mut c = TcpStream::connect(addr).unwrap();
        c.write_all(b"GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n").unwrap();
        let (st, out) = raw_status(&mut c);
        assert!(st == 200 && out.contains("{\"status\":\"ok\"}"), "{out}");
    }

    /// There is no cap on connections: a thousand that send nothing are held (a task each, and no thread), the server answers a request from another meanwhile, and each of them is answered 408 and
    /// let go by its time.
    #[test]
    fn idle_connections_are_held_without_a_cap_and_closed_by_their_time() {
        raise_fd_limit(8192);
        let (tx, _rx) = std::sync::mpsc::channel();
        let addr = listen_timed(Times { head: Duration::from_secs(2), body: Duration::from_secs(2), idle: Duration::from_secs(2) }, tx);
        let idle: Vec<TcpStream> = (0..1000).map(|_| TcpStream::connect(addr).unwrap()).collect();
        let mut c = TcpStream::connect(addr).unwrap();
        c.write_all(b"GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n").unwrap();
        assert_eq!(raw_status(&mut c).0, 200, "answered beside a thousand idle connections");
        // none of them was refused: the time ends them (a connection that sent nothing is let go without an answer)
        let mut closed = 0;
        for mut i in idle {
            i.set_read_timeout(Some(Duration::from_secs(10))).unwrap();
            let mut out = Vec::new();
            let _ = i.read_to_end(&mut out);
            // (answered 408 and let go, as a connection that sent a part of a request is; one that was kept after an answer and sent nothing is let go with no word)
            if out.is_empty() || out.starts_with(b"HTTP/1.1 408") {
                closed += 1;
            }
        }
        assert_eq!(closed, 1000);
    }

    /// R11's load test: a thousand streaming connections at once, on a scheduler over a CPU fake model, and every one of them is answered 200 with its events and its `[DONE]`, none refused, none
    /// cut. Every client has sent its request before any is read: all of them are in flight together (the scheduler runs eight at a time; the rest wait in its queue, and their connections are
    /// held by the front end as tasks, with no cap on how many). The clients are eight threads that poll their connections, not a thread each: where a system gives a process few threads (a Mac
    /// runner's), the thousand handlers of the replies are threads enough.
    #[test]
    fn a_thousand_streaming_connections_all_end_without_an_error() {
        const N: usize = 1000;
        const WORKERS: usize = 8;
        raise_fd_limit(8192);
        let mut f = Fake::new(512, 64);
        f.decode_delay = Duration::from_millis(1);
        let addr = serve_fake(f, None, 8);
        let body = r#"{"messages":[{"role":"user","content":"hello there"}],"max_tokens":6,"temperature":0,"stream":true}"#;
        let sent = Arc::new(std::sync::Barrier::new(WORKERS));
        let workers: Vec<_> = (0..WORKERS)
            .map(|w| {
                let sent = sent.clone();
                std::thread::spawn(move || -> Vec<String> {
                    let mut errors = Vec::new();
                    let mut open: Vec<(usize, TcpStream, Vec<u8>)> = Vec::new();
                    for i in (w..N).step_by(WORKERS) {
                        // (made in waves: a burst of a thousand SYNs is longer than a system's listen queue, which is no concern of the server's; once made, all are held open together)
                        std::thread::sleep(Duration::from_millis((i / 50) as u64 * 40 / WORKERS as u64));
                        let connected = TcpStream::connect(addr).and_then(|mut c| {
                            c.write_all(
                                format!("POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}", body.len()).as_bytes(),
                            )?;
                            c.set_nonblocking(true)?;
                            Ok(c)
                        });
                        match connected {
                            Ok(c) => open.push((i, c, Vec::new())),
                            Err(e) => errors.push(format!("{i}: {e}")),
                        }
                    }
                    sent.wait(); // (every connection is open and asked before any answer is read)
                    let deadline = std::time::Instant::now() + Duration::from_secs(120);
                    while !open.is_empty() && std::time::Instant::now() < deadline {
                        let mut buf = [0u8; 8192];
                        let mut progressed = false;
                        open.retain_mut(|(i, c, got)| match c.read(&mut buf) {
                            Ok(0) => {
                                let text = String::from_utf8_lossy(got);
                                if !text.starts_with("HTTP/1.1 200 OK") {
                                    errors.push(format!("{i}: {}", text.lines().next().unwrap_or("")));
                                } else if !text.contains("data: [DONE]") {
                                    errors.push(format!("{i}: the stream did not end with [DONE]"));
                                }
                                progressed = true;
                                false
                            }
                            Ok(n) => {
                                got.extend_from_slice(&buf[..n]);
                                progressed = true;
                                true
                            }
                            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => true,
                            Err(e) => {
                                errors.push(format!("{i}: read: {e}"));
                                progressed = true;
                                false
                            }
                        });
                        if !progressed {
                            std::thread::sleep(Duration::from_millis(5));
                        }
                    }
                    errors.extend(open.iter().map(|(i, ..)| format!("{i}: no end in 120 s")));
                    errors
                })
            })
            .collect();
        let errors: Vec<String> = workers.into_iter().flat_map(|w| w.join().unwrap()).collect();
        assert!(errors.is_empty(), "{} of {N} failed; the first: {:?}", errors.len(), errors.first());
    }

    /// A client that goes while its reply is being made ends the sequence: the pipe says it is gone, the handler stops writing, and the scheduler is not left running a reply nobody reads (the
    /// request's record, `/v1/glyd/requests/ID`, says it ended).
    #[test]
    fn a_client_that_disconnects_ends_its_sequence() {
        let mut f = Fake::new(64, 32);
        f.decode_delay = Duration::from_millis(20);
        let addr = serve_fake(f, None, 4);
        let body = r#"{"messages":[{"role":"user","content":"hello there"}],"max_tokens":400,"temperature":0,"stream":true}"#;
        let (r, id) = first_event(addr, body);
        let state = |id: &str| -> String {
            let (st, _, b) = ask(addr, "GET", &format!("/v1/glyd/requests/{id}"), &[("Host", "x")]);
            assert_eq!(st, 200, "{b}");
            at(&json::parse(&b).unwrap(), &["state"]).as_str().unwrap_or("").to_string()
        };
        assert_eq!(state(&id), "running", "the reply is being made");
        drop(r); // the client goes
        let t = std::time::Instant::now();
        while state(&id) != "ended" {
            assert!(t.elapsed() < Duration::from_secs(10), "the sequence of a client that went is still running");
            std::thread::sleep(Duration::from_millis(50));
        }
    }

    /// A request whose body's end is in doubt is refused (400), on a connection that is kept or not: a repeated Content-Length, one that is not a number, and a Transfer-Encoding (which this server
    /// does not read) would each be read one way here and another by a proxy in front of it.
    #[test]
    fn an_ambiguous_body_length_is_refused() {
        let head = |t: &str| read_head(&mut BufReader::new(t.as_bytes())).err().map(|e| e.0);
        assert_eq!(head("POST / HTTP/1.1\r\nContent-Length: 5\r\nContent-Length: 5\r\n\r\n"), Some(400));
        assert_eq!(head("POST / HTTP/1.1\r\nContent-Length: 5\r\ncontent-length: 6\r\n\r\n"), Some(400));
        assert_eq!(head("POST / HTTP/1.1\r\nContent-Length: +5\r\n\r\n"), Some(400));
        assert_eq!(head("POST / HTTP/1.1\r\nContent-Length: 5, 5\r\n\r\n"), Some(400));
        assert_eq!(head("POST / HTTP/1.1\r\nContent-Length:\r\n\r\n"), Some(400));
        assert_eq!(head("POST / HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n"), Some(400));
        assert_eq!(head("POST / HTTP/1.1\r\nContent-Length: 99999999999999999999\r\n\r\n"), Some(413));
        assert_eq!(head("POST / HTTP/1.1\r\nContent-Length: 5\r\n\r\n"), None, "one is fine");
        // and over the wire, on a connection that asked to be kept
        let (tx, _rx) = std::sync::mpsc::channel();
        let addr = listen_timed(Times::default(), tx);
        for bad in ["Content-Length: 2\r\nContent-Length: 2", "Transfer-Encoding: chunked"] {
            let mut c = TcpStream::connect(addr).unwrap();
            c.write_all(format!("POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nConnection: keep-alive\r\n{bad}\r\n\r\n{{}}").as_bytes()).unwrap();
            assert_eq!(raw_status(&mut c).0, 400, "{bad}");
        }
    }

    /// A client that sends more than the beginning of a next request while its answer is being made is let go, and its sequence with it (what it sent is not held without bound).
    #[test]
    fn a_client_that_floods_while_it_is_answered_is_let_go() {
        let mut f = Fake::new(64, 32);
        f.decode_delay = Duration::from_millis(20);
        let addr = serve_fake(f, None, 4);
        let body = r#"{"messages":[{"role":"user","content":"hello there"}],"max_tokens":400,"temperature":0,"stream":true}"#;
        let (r, id) = first_event(addr, body);
        let mut c = r.into_inner();
        let junk = vec![b'x'; 256 << 10];
        let _ = c.write_all(&junk);
        let state = |id: &str| -> String {
            let (st, _, b) = ask(addr, "GET", &format!("/v1/glyd/requests/{id}"), &[("Host", "x")]);
            assert_eq!(st, 200, "{b}");
            at(&json::parse(&b).unwrap(), &["state"]).as_str().unwrap_or("").to_string()
        };
        let t = std::time::Instant::now();
        while state(&id) != "ended" {
            assert!(t.elapsed() < Duration::from_secs(10), "the sequence of a client that flooded is still running");
            std::thread::sleep(Duration::from_millis(50));
        }
    }

    /// One connection carries as many requests as the client sends while it asks for `Connection: keep-alive` and each answer has a length, and ends with one that does not (a stream); a client that
    /// asks for nothing, or for `close`, gets the connection closed after its answer.
    #[test]
    fn a_connection_is_kept_for_the_next_request() {
        let (tx, _rx) = std::sync::mpsc::channel();
        let addr = listen_timed(Times::default(), tx);
        let mut c = TcpStream::connect(addr).unwrap();
        c.set_read_timeout(Some(Duration::from_secs(10))).unwrap();
        let mut r = std::io::BufReader::new(c.try_clone().unwrap());
        // reads one answer with a Content-Length: (status, its head, its body)
        let mut one = || {
            let mut head = String::new();
            loop {
                let mut l = String::new();
                r.read_line(&mut l).unwrap();
                head.push_str(&l);
                if l == "\r\n" || l.is_empty() {
                    break;
                }
            }
            let len: usize = head.to_ascii_lowercase().lines().find_map(|l| l.strip_prefix("content-length:").map(|v| v.trim().parse().unwrap())).unwrap_or(0);
            let mut body = vec![0u8; len];
            r.read_exact(&mut body).unwrap();
            (head.split_whitespace().nth(1).unwrap().to_string(), head, String::from_utf8(body).unwrap())
        };
        for i in 0..5 {
            c.write_all(b"GET /health HTTP/1.1\r\nHost: localhost\r\nConnection: keep-alive\r\n\r\n").unwrap();
            let (st, head, body) = one();
            assert_eq!((st.as_str(), body.as_str()), ("200", "{\"status\":\"ok\"}"), "request {i}");
            assert!(head.contains("Connection: keep-alive"), "{head}");
        }
        // two requests sent together are answered in turn
        c.write_all(b"GET /health HTTP/1.1\r\nHost: localhost\r\nConnection: keep-alive\r\n\r\nGET /ready HTTP/1.1\r\nHost: localhost\r\nConnection: keep-alive\r\n\r\n").unwrap();
        assert_eq!(one().2, "{\"status\":\"ok\"}");
        assert_eq!(one().2, "{\"status\":\"ready\"}");
        // `Connection: close` is answered with the connection closed
        c.write_all(b"GET /health HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n").unwrap();
        let (_, head, _) = one();
        assert!(head.contains("Connection: close"), "{head}");
        let mut b = [0u8; 4];
        assert_eq!(r.read(&mut b).unwrap(), 0, "the server closed it");
    }

    /// `/health` says the process is up, `/ready` that the model is loaded (200 `ready`; 503 `loading` where the server listens while it loads), `/metrics` is Prometheus's text format with the families an operator
    /// needs; with a key set the key guards `/metrics` and not the two probes (a load balancer has no key).
    #[test]
    fn health_ready_and_metrics_answer_as_an_operator_needs() {
        let (tx, _rx) = std::sync::mpsc::channel();
        let loading = Arc::new(AtomicBool::new(true));
        let addr = listen_with_life("127.0.0.1:0", "Qwen/Qwen3-8B", 4096, Some("k3y".into()), true, tx, loading.clone(), Arc::default()).unwrap();
        // loading: alive, not ready
        assert_eq!(ask(addr, "GET", "/ready", &[]).0, 503);
        assert!(ask(addr, "GET", "/ready", &[]).2.contains("loading"));
        loading.store(false, Ordering::Relaxed);
        let (st, _, body) = ask(addr, "GET", "/ready", &[]);
        assert_eq!((st, body.as_str()), (200, "{\"status\":\"ready\"}"));
        assert_eq!(ask(addr, "GET", "/health", &[]).0, 200);
        // the metrics: the key guards them
        assert_eq!(ask(addr, "GET", "/metrics", &[]).0, 401);
        assert_eq!(ask(addr, "GET", "/metrics", &[("Authorization", "Bearer wrong")]).0, 401);
        let (st, head, text) = ask(addr, "GET", "/metrics", &[("Authorization", "Bearer k3y")]);
        assert_eq!(st, 200);
        assert!(head.contains("content-type: text/plain; version=0.0.4"), "{head}");
        for name in [
            "glyd_requests_total",
            "glyd_generated_tokens_total",
            "glyd_prompt_tokens_total",
            "glyd_time_to_first_token_seconds_bucket",
            "glyd_time_per_output_token_seconds_bucket",
            "glyd_queue_depth",
            "glyd_requests_in_flight",
            "glyd_ready 1",
            "glyd_draining 0",
            "glyd_build_info{version=",
        ] {
            assert!(text.contains(name), "{name} in {text}");
        }
        // a refusal is counted: the 401 above
        assert!(text.contains("glyd_requests_refused_total{code=\"401\"} "), "{text}");
        // without a key the metrics are open (this computer's own address, no key)
        let (tx, _rx2) = std::sync::mpsc::channel();
        let open = listen("127.0.0.1:0", "m", 4096, None, true, tx).unwrap();
        assert_eq!(ask(open, "GET", "/metrics", &[]).0, 200);
        // and the Host check applies to them as to every path
        assert_eq!(ask(open, "GET", "/metrics", &[("Host", "evil.example")]).0, 421);
    }

    /// A drain: `/ready` turns 503 `draining`, the requests for the engine in flight are let end, a new one is refused (503, `Retry-After`), the probes and the model list still answer, and the count is 0 when the last ends.
    #[test]
    fn a_drain_stops_taking_requests_and_waits_for_those_in_flight() {
        let (tx, rx) = std::sync::mpsc::channel();
        let life = Arc::new(crate::ops::Life::default());
        let addr = listen_with_life("127.0.0.1:0", "m", 4096, None, true, tx, Arc::new(AtomicBool::new(false)), life.clone()).unwrap();
        // a request that is taken and waits in the engine's queue (nobody runs it here): in flight
        let body = "{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}";
        let mut c = TcpStream::connect(addr).unwrap();
        c.write_all(
            format!("POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}", body.len())
                .as_bytes(),
        )
        .unwrap();
        let t = std::time::Instant::now();
        while life.inflight() == 0 {
            assert!(t.elapsed() < Duration::from_secs(5), "the request was not taken");
            std::thread::sleep(Duration::from_millis(10));
        }
        assert_eq!(ask(addr, "GET", "/ready", &[]).0, 200);
        life.begin_drain();
        let (st, _, b) = ask(addr, "GET", "/ready", &[]);
        assert_eq!((st, b.as_str()), (503, "{\"status\":\"draining\"}"));
        // a new request for the engine is refused and not counted; the rest answers
        let (st, head, b) = ask(addr, "POST", "/v1/chat/completions", &[("Content-Length", "0")]);
        assert_eq!(st, 503, "{b}");
        assert!(head.contains("retry-after: 1") && b.contains("shutting down"), "{head} {b}");
        assert_eq!(ask(addr, "GET", "/v1/models", &[]).0, 200);
        assert_eq!(ask(addr, "GET", "/health", &[]).0, 200);
        assert!(ask(addr, "GET", "/metrics", &[]).2.contains("glyd_draining 1"));
        assert_eq!(life.inflight(), 1);
        assert_eq!(life.wait_idle(Duration::from_millis(100)), 1, "the bound passed with the request still in flight");
        // the request ends (here: the engine's queue is dropped with the request in it): the drain is done
        drop(rx);
        assert_eq!(life.wait_idle(Duration::from_secs(5)), 0);
        drop(c);
    }

    /// The API key as vLLM's middleware checks it: `Authorization: Bearer KEY` (the scheme in any case; the token is what follows the first space), by the digests of both in constant time; for every path of /v1, and
    /// not for an OPTIONS, the page, /health or /stats; the Host and Origin checks come before it.
    #[test]
    fn the_api_key_as_vllm_checks_it() {
        let (tx, _rx) = std::sync::mpsc::channel();
        let addr = listen("127.0.0.1:0", "m", 4096, Some("s3cret key".to_string()), true, tx).unwrap();
        let get = |path: &str, auth: Option<&str>| {
            let mut h = vec![("Host", "localhost:8000")];
            if let Some(a) = auth {
                h.push(("Authorization", a));
            }
            ask(addr, "GET", path, &h)
        };
        for ok in ["Bearer s3cret key", "bearer s3cret key", "BEARER s3cret key", "Bearer s3cret key  "] {
            assert_eq!(get("/v1/models", Some(ok)).0, 200, "{ok:?}");
        }
        for bad in [
            "Bearer s3cret",
            "Bearer s3cret keyx",
            "Bearer  s3cret key",
            "Basic s3cret key",
            "s3cret key",
            "Bearer",
            "Bearer ",
            "",
            "Bearer S3CRET KEY",
            "Token s3cret key",
        ] {
            let (st, _, body) = get("/v1/models", Some(bad));
            assert!(
                st == 401 && body.starts_with("{\"error\":{\"message\":\"Incorrect API key provided") && body.contains("\"code\":\"invalid_api_key\""),
                "{bad:?}: {st} {body}"
            );
        }
        assert_eq!(get("/v1/models", None).0, 401);
        // every path under /v1 asks for it, one that does not exist too (the key is asked before the route is looked for)
        assert_eq!(get("/v1/nothing", None).0, 401);
        assert_eq!(get("/v1", None).0, 401);
        assert_eq!(get("/v1/nothing", Some("Bearer s3cret key")).0, 404);
        assert_eq!(ask(addr, "POST", "/v1/chat/completions", &[("Host", "localhost:8000"), ("Content-Length", "0")]).0, 401);
        // what is open
        for open in ["/", "/health", "/stats", "/nothing"] {
            assert_ne!(get(open, None).0, 401, "{open}");
        }
        // an OPTIONS is not asked for the key (it is the browser's question before it sends one)
        assert_eq!(ask(addr, "OPTIONS", "/v1/models", &[("Host", "localhost:8000")]).0, 405);
        // the Host is judged before the key
        assert_eq!(ask(addr, "GET", "/v1/models", &[("Host", "evil.example")]).0, 421);
        // the comparison: digests of the same length for any key
        let d = key_digest("s3cret key");
        assert!(key_matches(&d, Some("Bearer s3cret key")) && !key_matches(&d, Some("Bearer s3cret kez")) && !key_matches(&d, None));
        assert!(key_matches(&key_digest("é"), Some("bearer é")));
        // every byte of the digest counts: a digest that differs in any one byte, the last included, is another key
        for i in 0..32 {
            let mut other = d;
            other[i] ^= 1;
            assert!(!key_matches(&other, Some("Bearer s3cret key")), "byte {i}");
        }
    }

    /// A connection whose request is written on a thread of its own while this one reads the answer (a server that answers before it has read it all must be read from at once); what came back.
    fn send_raw(addr: std::net::SocketAddr, write: impl FnOnce(&mut dyn Conn) + Send + 'static) -> String {
        let mut c = TcpStream::connect(addr).unwrap();
        let mut w = c.try_clone().unwrap();
        let t = std::thread::spawn(move || write(&mut w));
        let mut out = Vec::new();
        let _ = c.read_to_end(&mut out);
        drop(c);
        t.join().unwrap();
        String::from_utf8_lossy(&out).into_owned()
    }

    /// The header block has bounds, 64 KiB and 256 headers, which are held before anything in it is looked at, the key included: a client with no credential that sends a header of 96 MiB, or 300 headers, is answered
    /// 431 where the server reads no more than the bound and not 401 after reading it all; a block within the bounds is judged as before.
    #[test]
    fn a_header_block_is_bounded_before_the_key_is_asked_for() {
        let (tx, _rx) = std::sync::mpsc::channel();
        let addr = listen("127.0.0.1:0", "m", 4096, Some("secret".to_string()), false, tx).unwrap();
        let first = |t: &str| t.lines().next().unwrap_or("").to_string();
        // a header line of 96 MiB, written while the server answers
        let t0 = std::time::Instant::now();
        let out = send_raw(addr, |w| {
            let _ = w.write_all(b"GET /v1/models HTTP/1.1\r\nHost: x\r\nX-Pad: ");
            let chunk = vec![b'a'; 1 << 20];
            for _ in 0..96 {
                if w.write_all(&chunk).is_err() {
                    return;
                }
            }
            let _ = w.write_all(b"\r\n\r\n");
        });
        assert_eq!(first(&out), "HTTP/1.1 431 Request Header Fields Too Large", "{}", &out[..out.len().min(300)]);
        assert!(out.contains("\"type\":\"invalid_request_error\"") && out.contains("header block"), "{out}");
        assert!(t0.elapsed() < Duration::from_secs(20), "{:?}", t0.elapsed());
        // a request line of 96 MiB is the same block
        let out = send_raw(addr, |w| {
            let _ = w.write_all(b"GET /");
            let chunk = vec![b'a'; 1 << 20];
            for _ in 0..96 {
                if w.write_all(&chunk).is_err() {
                    return;
                }
            }
            let _ = w.write_all(b" HTTP/1.1\r\n\r\n");
        });
        assert!(first(&out).starts_with("HTTP/1.1 431"), "{}", &out[..out.len().min(300)]);
        // 300 headers, 257 of them too many; 256 are within the bound
        let many = |n: usize| {
            move |w: &mut dyn Conn| {
                let h: String = (0..n).map(|i| format!("X-N{i}: 1\r\n")).collect();
                let _ = w.write_all(format!("GET /v1/models HTTP/1.1\r\nHost: x\r\n{h}\r\n").as_bytes());
            }
        };
        assert!(first(&send_raw(addr, many(300))).starts_with("HTTP/1.1 431"));
        assert!(first(&send_raw(addr, many(257))).starts_with("HTTP/1.1 431"));
        // (Host is one of the 256 here: 255 more are the most)
        assert!(first(&send_raw(addr, many(255))).starts_with("HTTP/1.1 401"), "within the count the key is asked for");
        // 60 KiB in one header is within the bound: the key is asked for, and with it the request is answered
        let big = |auth: &'static str| {
            move |w: &mut dyn Conn| {
                let _ = w.write_all(format!("GET /v1/models HTTP/1.1\r\nHost: x\r\nX-Pad: {}\r\n{auth}\r\n", "a".repeat(60 << 10)).as_bytes());
            }
        };
        assert!(first(&send_raw(addr, big(""))).starts_with("HTTP/1.1 401"));
        assert!(first(&send_raw(addr, big("Authorization: Bearer secret\r\n"))).starts_with("HTTP/1.1 200"));
        // the bound is the block's, not a line's: two headers of 40 KiB are 80 KiB
        let two = |w: &mut dyn Conn| {
            let pad = "a".repeat(40 << 10);
            let _ = w.write_all(format!("GET /v1/models HTTP/1.1\r\nHost: x\r\nX-A: {pad}\r\nX-B: {pad}\r\n\r\n").as_bytes());
        };
        assert!(first(&send_raw(addr, two)).starts_with("HTTP/1.1 431"));
        // the Host and Origin checks of an address of this computer's own come after the bound, as the key does
        let (tx2, _rx2) = std::sync::mpsc::channel();
        let local = listen("127.0.0.1:0", "m", 4096, None, true, tx2).unwrap();
        let out = send_raw(local, many(300));
        assert!(first(&out).starts_with("HTTP/1.1 431"), "{out}");
        // and the server still answers
        assert_eq!(ask(addr, "GET", "/v1/models", &[("Host", "x"), ("Authorization", "Bearer secret")]).0, 200);
    }

    /// What is read of an input that never ends is no more than the bound and a buffer: the header block is refused where it passes it, whatever the line, and is not read to its end.
    #[test]
    fn a_header_block_is_read_no_further_than_its_bound() {
        struct Endless(usize, u8);
        impl Read for Endless {
            fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
                buf.fill(self.1);
                self.0 += buf.len();
                Ok(buf.len())
            }
        }
        for byte in *b"a\n" {
            let mut r = BufReader::new(Endless(0, byte));
            let e = read_head(&mut r);
            // (a stream of newlines is blank lines: the first one ends the block with no request; a stream of letters is a line with no end)
            if byte == b'a' {
                assert_eq!(e.err().map(|e| e.0), Some(431));
            } else {
                assert!(matches!(e, Ok(Some(_)) | Ok(None)));
            }
            assert!(r.get_ref().0 <= MAX_HEAD + (16 << 10), "read {} bytes", r.get_ref().0);
        }
        // the lines that fit are read as they were: the fields the server uses
        let text = "POST /v1/completions HTTP/1.1\r\nHost: localhost:8000\r\nOrigin: http://localhost:8000\r\nAuthorization: Bearer k\r\nContent-Length: 12\r\nExpect: 100-continue\r\nAccess-Control-Request-Method: POST\r\n\r\nbody";
        let mut r = BufReader::new(text.as_bytes());
        let h = read_head(&mut r).unwrap().unwrap();
        assert_eq!(
            (h.method.as_str(), h.path.as_str(), h.len, h.expect, h.host.as_str(), h.origin.as_str()),
            ("POST", "/v1/completions", 12, true, "localhost:8000", "http://localhost:8000")
        );
        assert_eq!((h.auth.as_deref(), h.ask_method.as_deref(), h.ask_headers), (Some("Bearer k"), Some("POST"), None));
        // (the body is left in the reader, for the body's own read)
        let mut rest = String::new();
        r.read_to_string(&mut rest).unwrap();
        assert_eq!(rest, "body");
        // a header that is not text is read, not refused: its bytes are what they are
        let mut r = BufReader::new(&b"GET / HTTP/1.1\r\nHost: x\r\nX-Name: caf\xe9\r\n\r\n"[..]);
        assert_eq!(read_head(&mut r).unwrap().unwrap().host, "x");
        assert!(read_head(&mut BufReader::new(&b""[..])).unwrap().is_none());
    }

    /// Qwen3's own chat template, from the fixture of its tests (as transformers renders it).
    fn qwen3_template() -> String {
        let f = json::parse(include_str!("../tests/fixtures/chat_qwen3.json")).unwrap();
        f.get("template").unwrap().as_str().unwrap().to_string()
    }

    const WEATHER: &str = r#"{"type":"function","function":{"name":"get_weather","description":"Get the weather","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}"#;

    /// A reply held to a grammar begins at once: the prompt closes the thinking and puts no `{` or call opening of its own (the grammar writes it).
    #[test]
    fn the_prompt_of_a_request_held_to_a_grammar() {
        let shared = Shared {
            tok: crate::tok::tests::toy(&["x"]),
            chat: Some(Chat::from_template(&qwen3_template()).unwrap()),
            name: "m".into(),
            defaults: Params::default(),
            window: 4096,
            vocab: 2048,
            thinking: None,
            refused: &[],
            rt: crate::runtime::context::Context::process().clone(),
        };
        let mut bytes: Vec<Vec<u8>> = (0..=255u8).map(|b| vec![b]).collect();
        bytes.push([&[toktrie::TokTrie::SPECIAL_TOKEN_MARKER][..], b"<|im_end|>"].concat());
        let c = crate::constrain::Grammars::new(bytes, &[256], 300).unwrap().compile(&Spec::Regex("[0-9]+".into())).unwrap();
        let msgs = r#""messages":[{"role":"user","content":"A number?"}]"#;
        let text = |body: &str, constraint: bool| {
            let v = json::parse(body).unwrap();
            let mut r = request_of(&v, 4096, &Params::default());
            r.params.constraint = constraint.then(|| c.clone());
            let api = api_of(&v, true, 2048).unwrap();
            chat_text_of(&shared, &v, &api, &r).map(|(t, i, p, _)| (t, i, p)).unwrap()
        };
        let body = format!("{{{msgs},\"guided_regex\":\"[0-9]+\",\"chat_template_kwargs\":{{\"enable_thinking\":true}}}}");
        let (t, thinking, prefix) = text(&body, true);
        assert!(t.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"), "{t}");
        assert_eq!((thinking, prefix.as_str()), (false, ""));
        // no grammar: the model's own generation prompt
        let (t, _, prefix) = text(&body, false);
        assert!(t.ends_with("<|im_start|>assistant\n"), "{t}");
        assert_eq!(prefix, "");
        // (the grammar of a request is the one the request names: forms that name none, and calls where the tools' format has no grammar, have none)
        let api = |b: &str| api_of(&json::parse(b).unwrap(), true, 2048).unwrap();
        let g = |b: &str| constraint_of(&shared, &json::parse(b).unwrap(), &api(b)).unwrap().is_some();
        assert!(!g(&format!("{{{msgs}}}")));
        assert!(!g(&format!("{{{msgs},\"tools\":[{WEATHER}]}}")));
    }

    /// What the model is shown: the tools in the template as vLLM's chat path gives them (`tojson` as Python writes it), a conversation with a call and its result, and the opening of a forced call at the end of the
    /// prompt, the thinking closed before it.
    #[test]
    fn the_prompt_of_a_request_with_tools() {
        let shared = Shared {
            tok: crate::tok::tests::toy(&["x"]),
            chat: Some(Chat::from_template(&qwen3_template()).unwrap()),
            name: "m".into(),
            defaults: Params::default(),
            window: 4096,
            vocab: 2048,
            thinking: None,
            refused: &[],
            rt: crate::runtime::context::Context::process().clone(),
        };
        let text = |body: &str| {
            let v = json::parse(body).unwrap();
            let r = request_of(&v, 4096, &Params::default());
            let api = api_of(&v, true, 2048).unwrap();
            chat_text_of(&shared, &v, &api, &r).map(|(t, i, p, _)| (t, i, p)).unwrap()
        };
        let head = "<|im_start|>system\nBe brief.\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n{\"type\": \"function\", \"function\": {\"name\": \"get_weather\", \"description\": \"Get the weather\", \"parameters\": {\"type\": \"object\", \"properties\": {\"city\": {\"type\": \"string\"}}, \"required\": [\"city\"]}}}\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n";
        let msgs = r#""messages":[{"role":"system","content":"Be brief."},{"role":"user","content":"Weather in Paris?"}]"#;
        let (t, thinking, prefix) = text(&format!("{{{msgs},\"tools\":[{WEATHER}]}}"));
        assert_eq!(
            (t.as_str(), thinking, prefix.as_str()),
            (format!("{head}<|im_start|>user\nWeather in Paris?<|im_end|>\n<|im_start|>assistant\n").as_str(), false, "")
        );
        // none: the tools are shown all the same (vLLM does), and no call is read
        assert_eq!(text(&format!("{{{msgs},\"tools\":[{WEATHER}],\"tool_choice\":\"none\"}}")).0, t);
        // no tools: no tools block
        let (plain, _, _) = text(&format!("{{{msgs}}}"));
        assert!(plain.starts_with("<|im_start|>system\nBe brief.<|im_end|>\n") && !plain.contains("# Tools"), "{plain}");
        // a call and its result in the conversation (as Open WebUI sends them: the arguments a string, the content null)
        let conv = r#""messages":[{"role":"system","content":"Be brief."},{"role":"user","content":"Weather in Paris?"},{"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"{\"city\":\"Paris\"}"}}]},{"role":"tool","tool_call_id":"call_1","content":"sunny, 20C"}]"#;
        let (t, _, _) = text(&format!("{{{conv},\"tools\":[{WEATHER}]}}"));
        assert_eq!(t, format!("{head}<|im_start|>user\nWeather in Paris?<|im_end|>\n<|im_start|>assistant\n<tool_call>\n{{\"name\": \"get_weather\", \"arguments\": {{\"city\": \"Paris\"}}}}\n</tool_call><|im_end|>\n<|im_start|>user\n<tool_response>\nsunny, 20C\n</tool_response><|im_end|>\n<|im_start|>assistant\n"));
        // a forced call: the thinking is closed in the prompt and the call opened (for a named function, to its arguments)
        let (t, thinking, prefix) =
            text(&format!("{{{msgs},\"tools\":[{WEATHER}],\"tool_choice\":{{\"type\":\"function\",\"function\":{{\"name\":\"get_weather\"}}}}}}"));
        assert!(t.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n<tool_call>\n{\"name\": \"get_weather\", \"arguments\": "), "{t}");
        assert_eq!((thinking, prefix.as_str()), (false, "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": "));
        let (t, _, prefix) =
            text(&format!("{{{msgs},\"tools\":[{WEATHER}],\"tool_choice\":\"required\",\"chat_template_kwargs\":{{\"enable_thinking\":true}}}}"));
        assert!(t.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n<tool_call>\n{\"name\": \""), "{t}");
        assert_eq!(prefix, "<tool_call>\n{\"name\": \"");
        // the thinking of the model, left on: the generation prompt as it is
        let (t, _, _) = text(&format!("{{{msgs},\"tools\":[{WEATHER}],\"chat_template_kwargs\":{{\"enable_thinking\":false}}}}"));
        assert!(t.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"), "{t}");
    }

    /// A server whose model says `reply` (a few characters a token) after any prompt, with Qwen3's template.
    /// Kimi-K3's forced call: its own opening at the prompt's end (the think channel closed empty, an empty response, the call opened; a named function's to its
    /// arguments), and none where the call is not forced.
    #[test]
    fn a_kimi_prompt_ends_with_its_forced_call() {
        let shared = Shared {
            tok: crate::tok::tests::toy(&["x"]),
            chat: Some(Chat::from_template(crate::chat::KIMI_CHAT).unwrap()),
            name: "m".into(),
            defaults: Params::default(),
            window: 4096,
            vocab: 2048,
            thinking: None,
            refused: &[],
            rt: crate::runtime::context::Context::process().clone(),
        };
        let text = |body: &str| {
            let v = json::parse(body).unwrap();
            let r = request_of(&v, 4096, &Params::default());
            let api = api_of(&v, true, 2048).unwrap();
            chat_text_of(&shared, &v, &api, &r).map(|(t, _, p, _)| (t, p)).unwrap()
        };
        let msgs = r#""messages":[{"role":"user","content":"Weather in Paris?"}]"#;
        let (t, p) = text(&format!("{{{msgs},\"tools\":[{WEATHER}]}}"));
        assert!(t.ends_with(crate::conv::XTML_THINK) && p.is_empty(), "{t}");
        let (t, p) = text(&format!("{{{msgs},\"tools\":[{WEATHER}],\"tool_choice\":{{\"type\":\"function\",\"function\":{{\"name\":\"get_weather\"}}}}}}"));
        assert_eq!(p, crate::xtml::forced(true, Some("get_weather")));
        assert!(t.ends_with(&format!("{}{p}", crate::conv::XTML_THINK)), "{t}");
        let (_, p) = text(&format!("{{{msgs},\"tools\":[{WEATHER}, {}],\"tool_choice\":\"required\"}}", WEATHER.replace("get_weather", "get_time")));
        assert_eq!(p, crate::xtml::forced(true, None));
    }

    /// A template that opens the thinking with no switch for it (GLM-5.3-Flash's): a forced call closes it empty before the call's opening.
    #[test]
    fn a_forced_call_closes_a_thinking_the_template_opens() {
        let template = "{%- for m in messages %}<|{{ m.role }}|>{{ m.content }}{% endfor %}{% if tools %}<tools>{% for t in tools %}{{ t | tojson }}{% endfor %}</tools><tool_call>NAME<arg_key>K</arg_key><arg_value>V</arg_value></tool_call>{% endif %}{%- if add_generation_prompt %}<|assistant|><think>{% endif %}";
        let shared = Shared {
            tok: crate::tok::tests::toy(&["x"]),
            chat: Some(Chat::from_template(template).unwrap()),
            name: "m".into(),
            defaults: Params::default(),
            window: 4096,
            vocab: 2048,
            thinking: None,
            refused: &[],
            rt: crate::runtime::context::Context::process().clone(),
        };
        let text = |body: &str| {
            let v = json::parse(body).unwrap();
            let r = request_of(&v, 4096, &Params::default());
            let api = api_of(&v, true, 2048).unwrap();
            chat_text_of(&shared, &v, &api, &r).map(|(t, i, p, _)| (t, i, p)).unwrap()
        };
        let msgs = r#""messages":[{"role":"user","content":"Weather in Paris?"}]"#;
        let (t, thinking, _) = text(&format!("{{{msgs},\"tools\":[{WEATHER}]}}"));
        assert!(t.ends_with("<|assistant|><think>") && thinking, "{t}");
        let (t, thinking, p) =
            text(&format!("{{{msgs},\"tools\":[{WEATHER}],\"tool_choice\":{{\"type\":\"function\",\"function\":{{\"name\":\"get_weather\"}}}}}}"));
        assert!(!thinking && t.ends_with(&format!("<|assistant|><think></think>{p}")) && p.starts_with("<tool_call>get_weather"), "{t}");
    }

    pub(crate) fn serve_script(reply: &str) -> std::net::SocketAddr {
        serve_script_with(reply, &qwen3_template())
    }

    /// `serve_script` with the chat template given.
    pub(crate) fn serve_script_with(reply: &str, template: &str) -> std::net::SocketAddr {
        // (a special token of the reply is a piece of its own)
        let mut pieces: Vec<String> = Vec::new();
        for (k, part) in reply.split("<|im_end|>").enumerate() {
            if k > 0 {
                pieces.push("<|im_end|>".to_string());
            }
            let chars: Vec<char> = part.chars().collect();
            pieces.extend(chars.chunks(4).map(|c| c.iter().collect::<String>()));
        }
        let (mut words, mut script): (Vec<String>, Vec<u32>) = (Vec::new(), Vec::new());
        for w in pieces {
            let id = words.iter().position(|x| *x == w).unwrap_or_else(|| {
                words.push(w.clone());
                words.len() - 1
            });
            script.push(id as u32 + 1);
        }
        let mut f = Fake::new(64, 32);
        f.script = Some(script);
        let refs: Vec<&str> = words.iter().map(String::as_str).collect();
        let window = f.store.len() * crate::kv::PAGE;
        let shared = Arc::new(Shared {
            tok: crate::tok::tests::toy(&refs),
            chat: Some(Chat::from_template(template).unwrap()),
            name: "fake".into(),
            defaults: Params { temperature: 0.0, ..Params::default() },
            window,
            vocab: 2048,
            thinking: None,
            refused: &[],
            rt: crate::runtime::context::Context::process().clone(),
        });
        let (tx, rx) = std::sync::mpsc::channel();
        let addr = listen("127.0.0.1:0", "fake", window, None, false, tx.clone()).unwrap();
        std::thread::spawn(move || work_with(&mut Sched::new(f, 4), shared, &rx, &tx));
        addr
    }

    /// A streamed chat answer put together: (thinking, content, the calls as (id, name, arguments) by index, the finish reasons, the usage's completion tokens).
    #[allow(clippy::type_complexity)]
    pub(crate) fn put_together(body: &str) -> (String, String, Vec<(String, String, String)>, Vec<String>, Option<f64>) {
        let (mut think, mut content, mut calls, mut finishes, mut usage) =
            (String::new(), String::new(), Vec::<(String, String, String)>::new(), Vec::new(), None);
        for e in events(body) {
            let Some(c) = e.get("choices").and_then(|c| c.as_array()).and_then(|c| c.first()) else {
                if let Some(u) = e.get("usage") {
                    usage = Some(f64_of(at(u, &["completion_tokens"])));
                }
                continue;
            };
            if let Some(d) = c.get("delta") {
                think.push_str(d.get("reasoning_content").and_then(Value::as_str).unwrap_or(""));
                content.push_str(d.get("content").and_then(Value::as_str).unwrap_or(""));
                for t in d.get("tool_calls").and_then(Value::as_array).unwrap_or(&[]) {
                    let i = f64_of(at(t, &["index"])) as usize;
                    while calls.len() <= i {
                        calls.push(Default::default());
                    }
                    if let Some(id) = t.get("id").and_then(Value::as_str) {
                        assert_eq!(t.get("type").and_then(Value::as_str), Some("function"));
                        calls[i].0 = id.to_string();
                        calls[i].1 = at(t, &["function", "name"]).as_str().unwrap().to_string();
                    }
                    calls[i].2.push_str(t.get("function").and_then(|f| f.get("arguments")).and_then(Value::as_str).unwrap_or(""));
                }
            }
            if let Some(f) = c.get("finish_reason").and_then(Value::as_str) {
                finishes.push(f.to_string());
            }
        }
        (think, content, calls, finishes, usage)
    }

    /// A chat whose reasoning is hidden (`include_reasoning: false`) has no log-probabilities, as vLLM's: they would spell the reasoning out token by token; a chat that shows its reasoning has them, and a plain completion's are as asked.
    #[test]
    fn hidden_reasoning_is_not_spelled_out_by_log_probabilities() {
        let addr = serve_script("<think>\nsecret thought\n</think>\n\nthe answer");
        let q =
            |extra: &str| format!("{{\"messages\":[{{\"role\":\"user\",\"content\":\"hi\"}}],\"max_tokens\":200,\"logprobs\":true,\"top_logprobs\":1{extra}}}");
        let (st, body) = call(addr, "/v1/chat/completions", &q(",\"include_reasoning\":false"));
        assert_eq!(st, 200, "{body}");
        let d = json::parse(&body).unwrap();
        assert_eq!(
            (at(&d, &["choices", "0", "message", "content"]).as_str().map(str::trim), at(&d, &["choices", "0", "logprobs"])),
            (Some("the answer"), &Value::Null),
            "{body}"
        );
        assert!(!body.contains("secr") && !body.contains("thought"), "{body}");
        let (_, body) = call(addr, "/v1/chat/completions", &q(",\"include_reasoning\":false,\"stream\":true"));
        let (think, content, _, _, _) = put_together(&body);
        assert!(!body.contains("secr") && !body.contains("thought") && think.is_empty() && content.trim() == "the answer", "{body}");
        for e in events(&body) {
            for c in e.get("choices").and_then(Value::as_array).unwrap_or(&[]) {
                assert_eq!(c.get("logprobs"), Some(&Value::Null), "{body}");
            }
        }
        // the reasoning shown: the log-probabilities are there, and so is its text
        let (_, body) = call(addr, "/v1/chat/completions", &q(""));
        let d = json::parse(&body).unwrap();
        assert!(at(&d, &["choices", "0", "logprobs", "content"]).as_array().is_some_and(|l| l.len() > 5), "{body}");
        assert_eq!(at(&d, &["choices", "0", "message", "reasoning_content"]).as_str().map(str::trim), Some("secret thought"), "{body}");
        // a plain completion has no reasoning to hide: the field changes nothing for it
        let (st, body) = call(addr, "/v1/completions", r#"{"prompt":"hi","max_tokens":4,"logprobs":1,"include_reasoning":false}"#);
        assert_eq!(st, 200, "{body}");
        assert!(at(&json::parse(&body).unwrap(), &["choices", "0", "logprobs", "tokens"]).as_array().is_some_and(|t| !t.is_empty()), "{body}");
    }

    /// `stream_options` with nothing in it is none, as in vLLM (an empty dict is false there): a request that always sends it, whether it streams or not, is answered; one with an option and no stream is not.
    #[test]
    fn an_empty_stream_options_is_none() {
        let a = |t: &str| api_of(&json::parse(t).unwrap(), true, 100).map(|a| (a.stream, a.include_usage));
        assert_eq!(a(r#"{"stream_options":{}}"#), Ok((false, false)));
        assert_eq!(a(r#"{"stream_options":null}"#), Ok((false, false)));
        assert_eq!(a(r#"{"stream":true,"stream_options":{}}"#), Ok((true, false)));
        assert_eq!(a(r#"{"stream":true,"stream_options":{"include_usage":true}}"#), Ok((true, true)));
        assert!(a(r#"{"stream_options":{"include_usage":true}}"#).unwrap_err().contains("only be defined when `stream=True`"));
    }

    /// Fill in the middle: the prompt is the text before, `suffix` the text after, and the model's own tokens put them in order (`<prefix>BEFORE<suffix>AFTER<middle>`); a model with no such tokens refuses `suffix`, in words.
    #[test]
    fn a_suffix_is_a_fill_in_the_middle() {
        let shared = |words: &[&str]| Shared {
            tok: crate::tok::tests::toy(words),
            chat: None,
            name: "m".into(),
            defaults: Params::default(),
            window: 4096,
            vocab: 2048,
            thinking: None,
            refused: &[],
            rt: crate::runtime::context::Context::process().clone(),
        };
        let ask = |sh: &Shared, body: &str| {
            let v = json::parse(body).unwrap();
            let api = api_of(&v, false, sh.vocab).unwrap();
            prompts_of(sh, &v, false, &api, &request_of(&v, sh.window, &sh.defaults))
        };
        let qwen = shared(&["<|fim_prefix|>", "<|fim_suffix|>", "<|fim_middle|>", "def", "add", "return", "a"]);
        let p = ask(&qwen, r#"{"prompt":"def add","suffix":"return a"}"#).unwrap();
        // (the ids the toy gives: the three tokens are 1, 2, 3 and the words after them)
        assert_eq!(p[0].ids, [1, 4, 5, 2, 6, 7, 3]);
        assert_eq!(p[0].text, "def add");
        // an empty suffix is the end of the file (still a fill in the middle); none, or null, is a plain prompt
        assert_eq!(ask(&qwen, r#"{"prompt":"def add","suffix":""}"#).unwrap()[0].ids, [1, 4, 5, 2, 3]);
        assert_eq!(ask(&qwen, r#"{"prompt":"def add"}"#).unwrap()[0].ids, [4, 5]);
        assert_eq!(ask(&qwen, r#"{"prompt":"def add","suffix":null}"#).unwrap()[0].ids, [4, 5]);
        let e = |sh: &Shared, b: &str| ask(sh, b).err().map(|e| e.to_string()).unwrap_or_default();
        assert!(e(&qwen, r#"{"prompt":["a","b"],"suffix":"x"}"#).contains("`prompt` that is one string"));
        assert!(e(&qwen, r#"{"prompt":"a","suffix":5}"#).contains("`suffix` must be a string"));
        assert!(e(&shared(&["def", "add"]), r#"{"prompt":"def","suffix":"add"}"#).contains("no fill-in-the-middle tokens"));
        // (a model that has only part of the set has none)
        assert!(e(&shared(&["<|fim_prefix|>", "<|fim_suffix|>", "def"]), r#"{"prompt":"def","suffix":"def"}"#).contains("no fill-in-the-middle tokens"));
    }

    /// OpenAI's error form: `{"error": {"message", "type", "param", "code"}}` with the status a client keys on, for what the server itself refuses as for what a request does; a streamed reply's chunks carry `usage: null` where the usage is asked for.
    #[test]
    fn errors_and_streams_as_openai_writes_them() {
        let addr = serve_fake_with(Fake::new(64, 32), None, 2, false, Some("k"), &[]);
        let raw = |method: &str, path: &str, auth: bool| {
            let mut c = TcpStream::connect(addr).unwrap();
            let a = if auth { "Authorization: Bearer k\r\n" } else { "" };
            c.write_all(format!("{method} {path} HTTP/1.1\r\nHost: x\r\n{a}Content-Length: 0\r\n\r\n").as_bytes()).unwrap();
            let mut out = String::new();
            c.read_to_string(&mut out).unwrap();
            let (head, body) = out.split_once("\r\n\r\n").unwrap();
            (head.to_string(), json::parse(body).unwrap_or_else(|e| panic!("{body}: {e}")))
        };
        let shape = |b: &Value, kind: &str, code: Option<&str>| {
            assert_eq!(at(b, &["error", "type"]).as_str(), Some(kind), "{b:?}");
            assert_eq!(at(b, &["error", "param"]), &Value::Null);
            assert_eq!(at(b, &["error", "code"]).as_str(), code, "{b:?}");
            assert!(at(b, &["error", "message"]).as_str().is_some_and(|m| !m.is_empty()));
        };
        let (h, b) = raw("GET", "/v1/models", false);
        assert!(h.starts_with("HTTP/1.1 401") && h.contains("WWW-Authenticate: Bearer"), "{h}");
        shape(&b, "authentication_error", Some("invalid_api_key"));
        let (h, b) = raw("GET", "/v1/nothing", true);
        assert!(h.starts_with("HTTP/1.1 404"), "{h}");
        shape(&b, "invalid_request_error", None);
        let (h, b) = raw("PUT", "/v1/chat/completions", true);
        assert!(h.starts_with("HTTP/1.1 405"), "{h}");
        shape(&b, "invalid_request_error", None);
        // a body past the limit is 413, before any of it is read
        let mut c = TcpStream::connect(addr).unwrap();
        c.write_all(b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nAuthorization: Bearer k\r\nContent-Length: 99999999999\r\n\r\n").unwrap();
        let mut out = String::new();
        c.read_to_string(&mut out).unwrap();
        assert!(out.starts_with("HTTP/1.1 413 Payload Too Large"), "{out}");
        shape(&json::parse(out.split_once("\r\n\r\n").unwrap().1).unwrap(), "invalid_request_error", None);
        let (_, b) = raw("GET", "/v1/nothing", true);
        assert_eq!(at(&b, &["error", "message"]).as_str(), Some("Invalid URL (GET /v1/nothing)"));
        // one model by its name, as `models.retrieve` asks (the slash of an org's model name may be written %2F)
        let (h, b) = raw("GET", "/v1/models/fake", true);
        assert!(h.starts_with("HTTP/1.1 200"), "{h}");
        assert_eq!((at(&b, &["id"]).as_str(), at(&b, &["object"]).as_str()), (Some("fake"), Some("model")));
        let (h, b) = raw("GET", "/v1/models/gpt-4", true);
        assert!(h.starts_with("HTTP/1.1 404"), "{h}");
        assert!(at(&b, &["error", "message"]).as_str().unwrap().contains("`gpt-4` does not exist"));
        // a refused request: the type, and no code where there is none to act on
        let mut c = TcpStream::connect(addr).unwrap();
        let body = r#"{"messages":[{"role":"user","content":"hi"}],"n":0}"#;
        c.write_all(
            format!("POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nAuthorization: Bearer k\r\nContent-Length: {}\r\n\r\n{body}", body.len()).as_bytes(),
        )
        .unwrap();
        let mut out = String::new();
        c.read_to_string(&mut out).unwrap();
        assert!(out.starts_with("HTTP/1.1 400"), "{out}");
        shape(&json::parse(out.split_once("\r\n\r\n").unwrap().1).unwrap(), "invalid_request_error", None);
        // a stream that asks for its usage: every chunk has `usage` (null), the last has it and no choices
        let open = serve_fake(Fake::new(64, 32), None, 2);
        let chat = r#""messages":[{"role":"user","content":"hello there"}],"max_tokens":3,"temperature":0"#;
        let (st, raw_out) = call(open, "/v1/chat/completions", &format!("{{{chat},\"stream\":true,\"stream_options\":{{\"include_usage\":true}}}}"));
        assert_eq!(st, 200, "{raw_out}");
        let evs = events(&raw_out);
        let chunks: Vec<&Value> = evs.iter().filter(|e| e.get("object").is_some()).collect();
        assert!(chunks.len() > 2, "{raw_out}");
        let (last, rest) = chunks.split_last().unwrap();
        assert!(rest.iter().all(|e| e.get("usage") == Some(&Value::Null)), "{raw_out}");
        assert_eq!((at(last, &["choices"]).as_array().map(<[Value]>::len), f64_of(at(last, &["usage", "completion_tokens"]))), (Some(0), 3.0));
    }

    const CALL: &str = "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>";

    fn chat_body(extra: &str) -> String {
        chat_body_with(WEATHER, extra)
    }

    fn chat_body_with(tool: &str, extra: &str) -> String {
        format!("{{\"messages\":[{{\"role\":\"user\",\"content\":\"Weather in Paris?\"}}],\"max_tokens\":200,\"tools\":[{tool}]{extra}}}")
    }

    /// Tool calls through the API as vLLM's hermes parser gives them: the call (an id, the function, its arguments as Python writes them) in `tool_calls`, the content null and the finish reason `tool_calls`, the same
    /// streamed (the function when its name is complete, then the arguments) with the usage counting the end token; an ordinary answer where tools are offered is an ordinary reply; a block that is not a call is text;
    /// the thinking apart from the call; only the first of several where `parallel_tool_calls` is false; none read where the choice is none.
    #[test]
    fn tool_calls_through_the_api() {
        let addr = serve_script(CALL);
        let (st, body) = call(addr, "/v1/chat/completions", &chat_body(""));
        assert_eq!(st, 200, "{body}");
        let d = json::parse(&body).unwrap();
        let c = at(&d, &["choices", "0"]);
        assert_eq!(at(c, &["finish_reason"]).as_str(), Some("tool_calls"));
        assert_eq!(at(c, &["message", "content"]), &Value::Null);
        let tc = at(c, &["message", "tool_calls"]).as_array().unwrap();
        assert_eq!(tc.len(), 1);
        let id = at(&tc[0], &["id"]).as_str().unwrap();
        assert!(id.starts_with("chatcmpl-tool-") && id.len() == 30, "{id}");
        assert_eq!(
            (at(&tc[0], &["type"]).as_str(), at(&tc[0], &["function", "name"]).as_str(), at(&tc[0], &["function", "arguments"]).as_str()),
            (Some("function"), Some("get_weather"), Some("{\"city\": \"Paris\"}"))
        );
        assert_eq!(f64_of(at(&d, &["usage", "completion_tokens"])), (CALL.chars().count().div_ceil(4) + 1) as f64, "the pieces of the call and the end token");
        // streamed: a chunk with the function (its id), chunks of arguments, the finish; nothing as content; the same call
        let (st, body) = call(addr, "/v1/chat/completions", &chat_body(",\"stream\":true,\"stream_options\":{\"include_usage\":true}"));
        assert_eq!(st, 200);
        let (think, content, calls, finishes, usage) = put_together(&body);
        assert_eq!((think.as_str(), content.as_str(), finishes), ("", "", vec!["tool_calls".to_string()]));
        assert_eq!(calls.len(), 1);
        assert!(calls[0].0.starts_with("chatcmpl-tool-") && calls[0].1 == "get_weather" && calls[0].2 == "{\"city\": \"Paris\"}", "{calls:?}");
        assert_eq!(usage, Some((CALL.chars().count().div_ceil(4) + 1) as f64));
        // the first chunk of the call names the function and has no arguments; the rest have no id
        let ev = events(&body);
        let starts: Vec<&Value> = ev
            .iter()
            .filter_map(|e| {
                e.get("choices")
                    .and_then(|c| c.as_array())
                    .and_then(|c| c.first())
                    .and_then(|c| c.get("delta"))
                    .and_then(|d| d.get("tool_calls"))
                    .and_then(|t| t.as_array())
                    .and_then(|t| t.first())
            })
            .collect();
        assert!(
            starts[0].get("id").is_some() && at(starts[0], &["function"]).get("arguments").is_none() && starts[1..].iter().all(|t| t.get("id").is_none()),
            "{starts:?}"
        );
        // an Open WebUI chat: every request carries its tools, and the model answers in text (a normal reply, whole and streamed)
        let talk = serve_script("The weather in Paris is sunny.");
        let (st, body) = call(talk, "/v1/chat/completions", &chat_body(""));
        let d = json::parse(&body).unwrap();
        assert_eq!(
            (st, at(&d, &["choices", "0", "message", "content"]).as_str(), at(&d, &["choices", "0", "finish_reason"]).as_str()),
            (200, Some("The weather in Paris is sunny."), Some("stop")),
            "{body}"
        );
        assert!(at(&d, &["choices", "0", "message"]).get("tool_calls").is_none());
        let (_, body) = call(talk, "/v1/chat/completions", &chat_body(",\"stream\":true"));
        let (_, content, calls, finishes, _) = put_together(&body);
        assert_eq!((content.as_str(), calls.len(), finishes), ("The weather in Paris is sunny.", 0, vec!["stop".to_string()]));
        // text before the call is content; the call's own text is not
        let both = serve_script(&format!("Let me look.\n{CALL}"));
        let (_, body) = call(both, "/v1/chat/completions", &chat_body(""));
        let d = json::parse(&body).unwrap();
        assert_eq!(
            (at(&d, &["choices", "0", "message", "content"]).as_str(), at(&d, &["choices", "0", "finish_reason"]).as_str()),
            (Some("Let me look.\n"), Some("tool_calls"))
        );
        let (_, body) = call(both, "/v1/chat/completions", &chat_body(",\"stream\":true"));
        let (_, content, calls, finishes, _) = put_together(&body);
        assert_eq!((content.as_str(), calls.len(), finishes), ("Let me look.\n", 1, vec!["tool_calls".to_string()]));
        // the thinking apart from the call
        let thought = serve_script(&format!("<think>\nParis, so the weather tool.\n</think>\n\n{CALL}"));
        let (_, body) = call(thought, "/v1/chat/completions", &chat_body(""));
        let d = json::parse(&body).unwrap();
        assert_eq!(
            (
                at(&d, &["choices", "0", "message", "reasoning_content"]).as_str(),
                at(&d, &["choices", "0", "message", "reasoning"]).as_str(),
                at(&d, &["choices", "0", "message", "content"])
            ),
            (Some("Paris, so the weather tool."), Some("Paris, so the weather tool."), &Value::Null),
            "{body}"
        );
        assert_eq!(at(&d, &["choices", "0", "message", "tool_calls", "0", "function", "name"]).as_str(), Some("get_weather"));
        let (_, body) = call(thought, "/v1/chat/completions", &chat_body(",\"stream\":true"));
        let (think, content, calls, _, _) = put_together(&body);
        assert_eq!((think.as_str(), content.as_str(), calls.len()), ("Paris, so the weather tool.", "", 1), "{body}");
        // a block that is not a call is text, never an error
        for text in [
            "<tool_call>\nnot json\n</tool_call>",
            "<tool_call>\n{\"name\": \"get_weather\"}\n</tool_call>",
            "Ok <tool_call>\n{\"name\": \"f\", \"arguments\": {",
        ] {
            let bad = serve_script(text);
            let (st, body) = call(bad, "/v1/chat/completions", &chat_body(""));
            let d = json::parse(&body).unwrap();
            assert_eq!(
                (st, at(&d, &["choices", "0", "message", "content"]).as_str(), at(&d, &["choices", "0", "finish_reason"]).as_str()),
                (200, Some(text), Some("stop")),
                "{text}: {body}"
            );
            assert!(at(&d, &["choices", "0", "message"]).get("tool_calls").is_none());
        }
        let bad = serve_script("<tool_call>\nnot json\n</tool_call>");
        let (_, body) = call(bad, "/v1/chat/completions", &chat_body(",\"stream\":true"));
        let (_, content, calls, finishes, _) = put_together(&body);
        assert_eq!((content.as_str(), calls.len(), finishes), ("<tool_call>\nnot json\n</tool_call>", 0, vec!["stop".to_string()]));
        // two calls: both, or the first where parallel calls are off
        let two = serve_script(&format!("{CALL}\n{}", CALL.replace("Paris", "Rome")));
        let (_, body) = call(two, "/v1/chat/completions", &chat_body(""));
        assert_eq!(at(&json::parse(&body).unwrap(), &["choices", "0", "message", "tool_calls"]).as_array().unwrap().len(), 2);
        let (_, body) = call(two, "/v1/chat/completions", &chat_body(",\"parallel_tool_calls\":false"));
        let one = json::parse(&body).unwrap();
        assert_eq!(at(&one, &["choices", "0", "message", "tool_calls"]).as_array().unwrap().len(), 1);
        assert!(at(&one, &["choices", "0", "message", "tool_calls", "0", "function", "arguments"]).as_str().unwrap().contains("Paris"));
        let (_, body) = call(two, "/v1/chat/completions", &chat_body(",\"parallel_tool_calls\":false,\"stream\":true"));
        let (_, _, calls, _, _) = put_together(&body);
        assert_eq!(calls.len(), 1);
        let (_, body) = call(two, "/v1/chat/completions", &chat_body(",\"stream\":true"));
        assert_eq!(put_together(&body).2.len(), 2);
        // several choices, each its own call (top_k 1 so that every draw is the model's one token)
        let pair = ",\"n\":2,\"temperature\":1,\"top_k\":1";
        let (_, body) = call(addr, "/v1/chat/completions", &chat_body(pair));
        let d = json::parse(&body).unwrap();
        for i in 0..2 {
            let c = at(&d, &["choices", &i.to_string()]);
            assert_eq!(
                (
                    at(c, &["finish_reason"]).as_str(),
                    at(c, &["message", "tool_calls"]).as_array().unwrap().len(),
                    at(c, &["message", "tool_calls", "0", "function", "name"]).as_str()
                ),
                (Some("tool_calls"), 1, Some("get_weather")),
                "{body}"
            );
        }
        assert_ne!(
            at(&d, &["choices", "0", "message", "tool_calls", "0", "id"]),
            at(&d, &["choices", "1", "message", "tool_calls", "0", "id"]),
            "each call has an id of its own"
        );
        let (_, body) = call(addr, "/v1/chat/completions", &chat_body(&format!("{pair},\"stream\":true,\"stream_options\":{{\"include_usage\":true}}")));
        let (mut per, mut finishes) = (vec![0usize; 2], vec![0usize; 2]);
        for e in events(&body) {
            if let Some(c) = e.get("choices").and_then(|c| c.as_array()).and_then(|c| c.first()) {
                let i = f64_of(at(c, &["index"])) as usize;
                per[i] += c
                    .get("delta")
                    .and_then(|d| d.get("tool_calls"))
                    .and_then(|t| t.as_array())
                    .map_or(0, |t| t.iter().filter(|x| x.get("id").is_some()).count());
                finishes[i] += (c.get("finish_reason").and_then(Value::as_str) == Some("tool_calls")) as usize;
            }
        }
        assert_eq!((per, finishes), (vec![1, 1], vec![1, 1]), "{body}");
        // tool_choice none: the tools were shown, the text stays text
        let (_, body) = call(addr, "/v1/chat/completions", &chat_body(",\"tool_choice\":\"none\""));
        let d = json::parse(&body).unwrap();
        assert_eq!((at(&d, &["choices", "0", "message", "content"]).as_str(), at(&d, &["choices", "0", "finish_reason"]).as_str()), (Some(CALL), Some("stop")));
        // refused, in vLLM's words
        for (extra, want) in [
            (",\"tool_choice\":\"sometimes\"", "Invalid value for `tool_choice`"),
            (",\"tool_choice\":{\"type\":\"function\",\"function\":{\"name\":\"nope\"}}", "does not match any of the specified `tools`"),
        ] {
            let (st, body) = call(addr, "/v1/chat/completions", &chat_body(extra));
            assert!(st == 400 && body.contains(want), "{extra}: {st} {body}");
        }
        let (st, body) = call(addr, "/v1/chat/completions", "{\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"tools\":[]}");
        assert!(st == 400 && body.contains("must not be an empty array"), "{body}");
        let (st, body) = call(addr, "/v1/chat/completions", "{\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"tool_choice\":\"auto\"}");
        assert!(st == 400 && body.contains("`tools` must be set"), "{body}");
    }

    /// A forced call: the reply begins in the prompt with the call's opening and the model writes the rest, which is read as a call (a named function's finish reason is the reply's own, as vLLM's: `stop`; a required one's is
    /// `tool_calls`).
    #[test]
    fn a_call_that_is_asked_for() {
        let named = r#","tool_choice":{"type":"function","function":{"name":"get_weather"}}"#;
        let addr = serve_script("{\"city\": \"Paris\"}}\n</tool_call>");
        let (st, body) = call(addr, "/v1/chat/completions", &chat_body(named));
        assert_eq!(st, 200, "{body}");
        let d = json::parse(&body).unwrap();
        assert_eq!(
            (at(&d, &["choices", "0", "finish_reason"]).as_str(), at(&d, &["choices", "0", "message", "content"]).as_str()),
            (Some("stop"), Some("")),
            "{body}"
        );
        assert_eq!(
            (
                at(&d, &["choices", "0", "message", "tool_calls", "0", "function", "name"]).as_str(),
                at(&d, &["choices", "0", "message", "tool_calls", "0", "function", "arguments"]).as_str()
            ),
            (Some("get_weather"), Some("{\"city\": \"Paris\"}"))
        );
        let (_, body) = call(addr, "/v1/chat/completions", &chat_body(&format!("{named},\"stream\":true")));
        let (_, content, calls, finishes, _) = put_together(&body);
        assert_eq!((content.as_str(), finishes), ("", vec!["stop".to_string()]));
        assert!(calls[0].1 == "get_weather" && calls[0].2 == "{\"city\": \"Paris\"}", "{calls:?}");
        // required: the model names the function
        let any = serve_script("get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>");
        let (_, body) = call(any, "/v1/chat/completions", &chat_body(",\"tool_choice\":\"required\""));
        let d = json::parse(&body).unwrap();
        assert_eq!(
            (at(&d, &["choices", "0", "finish_reason"]).as_str(), at(&d, &["choices", "0", "message", "tool_calls", "0", "function", "name"]).as_str()),
            (Some("tool_calls"), Some("get_weather")),
            "{body}"
        );
        let (_, body) = call(any, "/v1/chat/completions", &chat_body(",\"tool_choice\":\"required\",\"stream\":true"));
        let (_, _, calls, finishes, _) = put_together(&body);
        assert_eq!((finishes, calls[0].1.as_str(), calls[0].2.as_str()), (vec!["tool_calls".to_string()], "get_weather", "{\"city\": \"Paris\"}"));
        // a forced call the model does not complete is text, and no failure
        let cut = serve_script("{\"city\": ");
        let (st, body) = call(cut, "/v1/chat/completions", &chat_body(named));
        let d = json::parse(&body).unwrap();
        assert_eq!((st, at(&d, &["choices", "0", "finish_reason"]).as_str()), (200, Some("stop")), "{body}");
        assert!(at(&d, &["choices", "0", "message"]).get("tool_calls").is_none());
    }

    /// The tools parts of Qwen3.5's own chat template (tests/fixtures), which writes its calls as XML.
    fn qwen35_template() -> &'static str {
        include_str!("../tests/fixtures/chat_qwen35_tools.jinja")
    }

    pub(crate) fn gemma_template() -> &'static str {
        include_str!("../tests/fixtures/chat_gemma4_tools.jinja")
    }

    const GEMMA_CALL: &str = "<|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>}<tool_call|>";

    /// Gemma 4 (its template, its markup): the prompt has the tools as the template declares them and the thought channel closed; a call written in its own marks is an OpenAI tool call, whole and streamed, whether
    /// the model chose it, was asked for a function or for a call; the thought channel is the reasoning, not text.
    #[test]
    fn gemma_4_tool_calls_and_thinking() {
        // the prompt
        let shared = Shared {
            tok: crate::tok::tests::toy(&["x"]),
            chat: Some(Chat::from_template(gemma_template()).unwrap()),
            name: "m".into(),
            defaults: Params::default(),
            window: 4096,
            vocab: 2048,
            thinking: None,
            refused: &[],
            rt: crate::runtime::context::Context::process().clone(),
        };
        let prompt = |extra: &str| {
            let v = json::parse(&chat_body(extra)).unwrap();
            let api = api_of(&v, true, 100).unwrap();
            let r = request_of(&v, 4096, &Params::default());
            chat_text_of(&shared, &v, &api, &r).unwrap()
        };
        let (t, thinking, p, _) = prompt("");
        assert!(
            t.contains("<|tool>declaration:get_weather{description:<|\"|>Get the weather<|\"|>") && t.ends_with("<|turn>model\n<|channel>thought\n<channel|>"),
            "{t}"
        );
        assert!(!thinking && p.is_empty());
        // a function asked for: the call opened in the prompt, to the arguments' brace; a call: to `call:`
        let (t, _, p, _) = prompt(r#","tool_choice":{"type":"function","function":{"name":"get_weather"}}"#);
        assert_eq!(p, "<|tool_call>call:get_weather{");
        assert!(t.ends_with(&format!("<|channel>thought\n<channel|>{p}")), "{t}");
        let (_, _, p, _) = prompt(r#","tool_choice":"required""#);
        assert_eq!(p, "<|tool_call>call:");
        // (a call and its answer in the history: the arguments an object, the answer after it)
        let v = json::parse(r#"{"messages":[{"role":"user","content":"Weather?"},{"role":"assistant","content":"","tool_calls":[{"id":"c1","type":"function","function":{"name":"get_weather","arguments":"{\"city\":\"Paris\"}"}}]},{"role":"tool","tool_call_id":"c1","content":"18 C"}],"tools":[{"type":"function","function":{"name":"get_weather","parameters":{"type":"object"}}}]}"#).unwrap();
        let api = api_of(&v, true, 100).unwrap();
        let (t, ..) = chat_text_of(&shared, &v, &api, &request_of(&v, 4096, &Params::default())).unwrap();
        assert!(
            t.contains(
                "<|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>}<tool_call|><|tool_response>response:get_weather{value:<|\"|>18 C<|\"|>}<tool_response|>"
            ),
            "{t}"
        );

        // the reply
        let addr = serve_script_with(GEMMA_CALL, gemma_template());
        let (st, body) = call(addr, "/v1/chat/completions", &chat_body(""));
        assert_eq!(st, 200, "{body}");
        let d = json::parse(&body).unwrap();
        let c = at(&d, &["choices", "0"]);
        assert_eq!((at(c, &["finish_reason"]).as_str(), at(c, &["message", "content"])), (Some("tool_calls"), &Value::Null), "{body}");
        assert_eq!(
            (
                at(c, &["message", "tool_calls", "0", "function", "name"]).as_str(),
                json::parse(at(c, &["message", "tool_calls", "0", "function", "arguments"]).as_str().unwrap()).unwrap()
            ),
            (Some("get_weather"), json::parse(r#"{"city":"Paris"}"#).unwrap())
        );
        let (_, body) = call(addr, "/v1/chat/completions", &chat_body(",\"stream\":true"));
        let (_, content, calls, finishes, _) = put_together(&body);
        assert_eq!((content.trim(), finishes), ("", vec!["tool_calls".to_string()]), "{body}");
        assert_eq!(calls[0].1, "get_weather");
        assert_eq!(json::parse(&calls[0].2).unwrap(), json::parse(r#"{"city":"Paris"}"#).unwrap(), "{calls:?}");
        assert!(body.matches("\"arguments\"").count() > 2, "the arguments come in pieces: {body}");
        // asked for a function: the model writes the arguments on from the brace
        let named = r#","tool_choice":{"type":"function","function":{"name":"get_weather"}}"#;
        let addr = serve_script_with("city:<|\"|>Paris<|\"|>}<tool_call|>", gemma_template());
        for extra in [named.to_string(), format!("{named},\"stream\":true")] {
            let (_, body) = call(addr, "/v1/chat/completions", &chat_body(&extra));
            let calls = if extra.contains("stream") {
                put_together(&body).2.into_iter().map(|c| (c.1, c.2)).collect::<Vec<_>>()
            } else {
                let d = json::parse(&body).unwrap();
                vec![(
                    at(&d, &["choices", "0", "message", "tool_calls", "0", "function", "name"]).as_str().unwrap().to_string(),
                    at(&d, &["choices", "0", "message", "tool_calls", "0", "function", "arguments"]).as_str().unwrap().to_string(),
                )]
            };
            assert_eq!(calls[0].0, "get_weather", "{body}");
            assert_eq!(json::parse(&calls[0].1).unwrap(), json::parse(r#"{"city":"Paris"}"#).unwrap(), "{body}");
        }
        // asked for a call: the model names the function
        let addr = serve_script_with("get_weather{city:<|\"|>Paris<|\"|>}<tool_call|>", gemma_template());
        let (_, body) = call(addr, "/v1/chat/completions", &chat_body(r#","tool_choice":"required""#));
        let d = json::parse(&body).unwrap();
        assert_eq!(
            (at(&d, &["choices", "0", "finish_reason"]).as_str(), at(&d, &["choices", "0", "message", "tool_calls", "0", "function", "name"]).as_str()),
            (Some("tool_calls"), Some("get_weather")),
            "{body}"
        );

        // the thought channel is the reasoning, the answer the content (the tokens are special ones in the reply: the reader sees them)
        let addr = serve_script_with("<|channel>thought\nLet me look.\n<channel|>The answer.", gemma_template());
        let ask = |extra: &str| format!("{{\"messages\":[{{\"role\":\"user\",\"content\":\"hi\"}}],\"max_tokens\":200{extra}}}");
        let (_, body) = call(addr, "/v1/chat/completions", &ask(""));
        let d = json::parse(&body).unwrap();
        assert_eq!(
            (
                at(&d, &["choices", "0", "message", "reasoning_content"]).as_str().map(str::trim),
                at(&d, &["choices", "0", "message", "content"]).as_str().map(str::trim)
            ),
            (Some("Let me look."), Some("The answer.")),
            "{body}"
        );
        assert!(!body.contains("channel"), "{body}");
        let (_, body) = call(addr, "/v1/chat/completions", &ask(",\"stream\":true"));
        let (think, content, _, _, _) = put_together(&body);
        assert_eq!((think.trim(), content.trim()), ("Let me look.", "The answer."), "{body}");
    }

    pub(crate) fn muse_template() -> &'static str {
        include_str!("../tests/fixtures/chat_muse_atem.jinja")
    }

    const MUSE_CALL: &str = " to=self<|message|>Need the weather.<|eom|><|start|>assistant to=get_weather<|message|><atem:function_calls>\n<atem:invoke name=\"get_weather\">\n<atem:parameter name=\"city\">Paris</atem:parameter>\n</atem:invoke>\n</atem:function_calls><|eom|>";

    /// Muse (its ATEM protocol): the prompt has the tools and the recipients; a message to `self` is the reasoning, one to a function an ATEM block that is an OpenAI tool call (whole and streamed), a function or a call asked for
    /// opened in the prompt, and a message to `user` the answer.
    #[test]
    fn muse_tool_calls_and_thinking() {
        let shared = Shared {
            tok: crate::tok::tests::toy(&["x"]),
            chat: Some(Chat::from_template(muse_template()).unwrap()),
            name: "m".into(),
            defaults: Params::default(),
            window: 4096,
            vocab: 2048,
            thinking: None,
            refused: &[],
            rt: crate::runtime::context::Context::process().clone(),
        };
        let prompt = |extra: &str| {
            let v = json::parse(&chat_body(extra)).unwrap();
            let api = api_of(&v, true, 100).unwrap();
            let r = request_of(&v, 4096, &Params::default());
            chat_text_of(&shared, &v, &api, &r).unwrap()
        };
        let (t, thinking, p, _) = prompt("");
        assert!(t.contains("<atem:function_calls>") && t.contains("\"name\": \"get_weather\"") && t.ends_with("<|start|>assistant"), "{t}");
        assert!(!thinking && p.is_empty());
        let (t, _, p, _) = prompt(r#","tool_choice":{"type":"function","function":{"name":"get_weather"}}"#);
        assert_eq!(p, " to=get_weather<|message|><atem:function_calls>\n<atem:invoke name=\"get_weather\">\n");
        assert!(t.ends_with(&format!("<|start|>assistant{p}")), "{t}");
        let (_, _, p, _) = prompt(r#","tool_choice":"required""#);
        assert_eq!(p, " to=");

        let addr = serve_script_with(MUSE_CALL, muse_template());
        let (st, body) = call(addr, "/v1/chat/completions", &chat_body(""));
        assert_eq!(st, 200, "{body}");
        let d = json::parse(&body).unwrap();
        let c = at(&d, &["choices", "0"]);
        assert_eq!(
            (at(c, &["finish_reason"]).as_str(), at(c, &["message", "reasoning_content"]).as_str().map(str::trim)),
            (Some("tool_calls"), Some("Need the weather.")),
            "{body}"
        );
        assert_eq!(
            (at(c, &["message", "tool_calls", "0", "function", "name"]).as_str(), at(c, &["message", "tool_calls", "0", "function", "arguments"]).as_str()),
            (Some("get_weather"), Some("{\"city\": \"Paris\"}")),
            "{body}"
        );
        assert!(!body.contains("atem") && !body.contains("<|"), "{body}");
        let (_, body) = call(addr, "/v1/chat/completions", &chat_body(",\"stream\":true"));
        let (think, content, calls, finishes, _) = put_together(&body);
        assert_eq!((think.trim(), content.trim(), finishes), ("Need the weather.", "", vec!["tool_calls".to_string()]), "{body}");
        assert_eq!((calls[0].1.as_str(), calls[0].2.as_str()), ("get_weather", "{\"city\": \"Paris\"}"), "{calls:?}");
        // asked for a function: the model writes the parameters on
        let addr = serve_script_with("<atem:parameter name=\"city\">Paris</atem:parameter>\n</atem:invoke>\n</atem:function_calls><|eom|>", muse_template());
        let named = r#","tool_choice":{"type":"function","function":{"name":"get_weather"}}"#;
        let (_, body) = call(addr, "/v1/chat/completions", &chat_body(named));
        let d = json::parse(&body).unwrap();
        assert_eq!(
            (
                at(&d, &["choices", "0", "message", "tool_calls", "0", "function", "name"]).as_str(),
                at(&d, &["choices", "0", "message", "tool_calls", "0", "function", "arguments"]).as_str()
            ),
            (Some("get_weather"), Some("{\"city\": \"Paris\"}")),
            "{body}"
        );
        // the answer to the user
        let addr = serve_script_with(" to=self<|message|>Easy.<|eom|><|start|>assistant to=user<|message|>It is sunny.<|eot|>", muse_template());
        let ask = "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":200}";
        let (_, body) = call(addr, "/v1/chat/completions", ask);
        let d = json::parse(&body).unwrap();
        assert_eq!(
            (
                at(&d, &["choices", "0", "message", "reasoning_content"]).as_str().map(str::trim),
                at(&d, &["choices", "0", "message", "content"]).as_str().map(str::trim)
            ),
            (Some("Easy."), Some("It is sunny.")),
            "{body}"
        );
    }

    /// Every family's tool call through both APIs, as its own template and markup have it: the OpenAI chat (whole and streamed) and Anthropic's Messages (whole and streamed) each give the call of `get_weather` with
    /// `{"city": "Paris"}`, and the finish reason or stop reason of a call.
    #[test]
    fn every_family_calls_a_tool_through_both_apis() {
        let weather_anthropic = r#"{"name":"get_weather","description":"Get the weather","input_schema":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}"#;
        // (a template that opens the thinking itself: the model closes it and then calls)
        let after_thinking = |call: &str| -> &'static str { Box::leak(format!("Need it.\n</think>\n\n{call}").into_boxed_str()) };
        let families: Vec<(&str, String, &str)> = vec![
            ("Qwen3 (hermes)", qwen3_template(), CALL),
            ("Qwen3.5 (XML)", qwen35_template().to_string(), XML_CALL),
            ("Qwen3.8 (XML)", include_str!("../tests/fixtures/chat_qwen38_tools.jinja").to_string(), after_thinking(XML_CALL)),
            ("GLM-5.3-Flash", include_str!("../tests/fixtures/chat_glm53_tools.jinja").to_string(), after_thinking("<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call>")),
            ("DeepSeek-V4 (DSML)", crate::chat::DSV4_CHAT.to_string(), "\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"get_weather\">\n<｜DSML｜parameter name=\"city\" string=\"true\">Paris</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>"),
            ("Gemma 4", gemma_template().to_string(), GEMMA_CALL),
            ("Muse (ATEM)", muse_template().to_string(), MUSE_CALL),
        ];
        let paris = json::parse(r#"{"city":"Paris"}"#).unwrap();
        for (name, template, reply) in families {
            let addr = serve_script_with(reply, &template);
            // OpenAI: whole
            let (st, body) = call(addr, "/v1/chat/completions", &chat_body(""));
            assert_eq!(st, 200, "{name}: {body}");
            let d = json::parse(&body).unwrap();
            let c = at(&d, &["choices", "0"]);
            assert_eq!(at(c, &["finish_reason"]).as_str(), Some("tool_calls"), "{name}: {body}");
            assert_eq!(at(c, &["message", "tool_calls", "0", "function", "name"]).as_str(), Some("get_weather"), "{name}: {body}");
            assert_eq!(json::parse(at(c, &["message", "tool_calls", "0", "function", "arguments"]).as_str().unwrap()).unwrap(), paris, "{name}: {body}");
            // OpenAI: streamed
            let (_, body) = call(addr, "/v1/chat/completions", &chat_body(",\"stream\":true"));
            let (_, _, calls, finishes, _) = put_together(&body);
            assert_eq!(finishes, vec!["tool_calls".to_string()], "{name}: {body}");
            assert_eq!((calls[0].1.as_str(), json::parse(&calls[0].2).unwrap()), ("get_weather", paris.clone()), "{name}: {body}");
            // Anthropic: whole and streamed
            let q = |extra: &str| {
                format!(r#"{{"max_tokens":200,"tools":[{weather_anthropic}],"messages":[{{"role":"user","content":"Weather in Paris?"}}]{extra}}}"#)
            };
            let (st, body) = call(addr, "/v1/messages", &q(""));
            assert_eq!(st, 200, "{name}: {body}");
            let d = json::parse(&body).unwrap();
            let tu = d
                .get("content")
                .and_then(Value::as_array)
                .unwrap()
                .iter()
                .find(|b| b.get("type").and_then(Value::as_str) == Some("tool_use"))
                .unwrap_or_else(|| panic!("{name}: {body}"));
            assert_eq!(
                (at(tu, &["name"]).as_str(), at(tu, &["input"]), at(&d, &["stop_reason"]).as_str()),
                (Some("get_weather"), &paris, Some("tool_use")),
                "{name}: {body}"
            );
            let (_, body) = call(addr, "/v1/messages", &q(r#","stream":true"#));
            let args: String = body
                .lines()
                .filter_map(|l| l.strip_prefix("data: "))
                .filter_map(|d| json::parse(d).ok())
                .filter_map(|e| e.get("delta").and_then(|d| d.get("partial_json")).and_then(Value::as_str).map(str::to_string))
                .collect();
            assert_eq!(json::parse(&args).unwrap(), paris, "{name}: {body}");
            assert!(body.contains("\"stop_reason\":\"tool_use\""), "{name}: {body}");
        }
    }

    /// A request for thinking gets a reply that begins with a thinking block, whole and streamed, empty where the model closes its thinking at once and writes
    /// no answer (the GPU job's random model did: a reply with no content block at all, which a client reading the first block fails on).
    #[test]
    fn a_thinking_reply_begins_with_its_thinking_block() {
        let addr = serve_script_with("</think>\n\n", include_str!("../tests/fixtures/chat_qwen38_tools.jinja"));
        let q = |extra: &str| {
            format!(r#"{{"max_tokens":50,"thinking":{{"type":"enabled","budget_tokens":20}},"messages":[{{"role":"user","content":"Hi"}}]{extra}}}"#)
        };
        let (st, body) = call(addr, "/v1/messages", &q(""));
        assert_eq!(st, 200, "{body}");
        let d = json::parse(&body).unwrap();
        assert_eq!((at(&d, &["content", "0", "type"]).as_str(), at(&d, &["content", "0", "thinking"]).as_str()), (Some("thinking"), Some("")), "{body}");
        let (_, body) = call(addr, "/v1/messages", &q(r#","stream":true"#));
        let first = body
            .lines()
            .filter_map(|l| l.strip_prefix("data: "))
            .filter_map(|d| json::parse(d).ok())
            .find(|e| e.get("type").and_then(Value::as_str) == Some("content_block_start"));
        assert_eq!(first.as_ref().and_then(|e| at(e, &["content_block", "type"]).as_str()), Some("thinking"), "{body}");
        // (without thinking asked for, no thinking block)
        let (_, body) = call(addr, "/v1/messages", r#"{"max_tokens":50,"messages":[{"role":"user","content":"Hi"}]}"#);
        assert!(!body.contains("\"type\":\"thinking\""), "{body}");
    }

    /// The random model that `logitsmodel qwen35 DIR --servable` makes serves a client: its tokenizer reads the template's text and the characters of JSON, its template takes tools, its end token is named, and the
    /// grammars of a JSON schema and of a forced call (Qwen3.8's XML) compile on its tokens, so that a client gets valid structured output and a valid call from weights that write random characters.
    #[cfg(feature = "dev")]
    #[test]
    fn a_servable_random_model_takes_grammars() {
        let dir = std::env::temp_dir().join(format!("glyd-servable-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("config.json"), r#"{"model_type":"qwen3_5_text","eos_token_id":2,"vocab_size":4096}"#).unwrap();
        crate::logits::write_serving_files(&dir, include_str!("../tests/fixtures/chat_qwen38_tools.jinja")).unwrap();
        let tok = Tok::load(&dir).unwrap();
        let sh = Shared {
            tok,
            chat: Chat::load(&dir).unwrap(),
            name: "tiny".into(),
            defaults: Params::default(),
            window: 4096,
            vocab: 4096,
            thinking: None,
            refused: &[],
            rt: crate::runtime::context::Context::process().clone(),
        };
        assert!(sh.chat.is_some());
        // the text: every character known, the markers one token each, the end token the config's
        let ids = sh.tok.encode_with("<|im_start|>user\nHi {\"a\": [1, 2.5, true]}<|im_end|>", false).unwrap();
        assert_eq!(ids[0], 1);
        assert_eq!(*ids.last().unwrap(), 2);
        assert!(!ids.contains(&0), "{ids:?}");
        assert_eq!(sh.tok.decode(&ids).unwrap(), "<|im_start|>user\nHi {\"a\": [1, 2.5, true]}<|im_end|>");
        let chat = |extra: &str| json::parse(&format!(r#"{{"messages":[{{"role":"user","content":"Weather in Paris?"}}],"max_tokens":40{extra}}}"#)).unwrap();
        let tool = r#","tools":[{"type":"function","function":{"name":"get_weather","description":"Get the weather","parameters":{"type":"object","properties":{"city":{"type":"string"},"days":{"type":"integer"}},"required":["city","days"]}}}]"#;
        // a plain chat, the thinking left to the template (it opens `<think>`)
        let p = prepare(&sh, &chat("")).unwrap();
        assert!(p.pr.ids.len() > 5 && p.pr.in_thinking && p.r.params.constraint.is_none());
        // a JSON schema: a grammar
        let p = prepare(&sh, &chat(r#","response_format":{"type":"json_schema","json_schema":{"name":"w","schema":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"],"additionalProperties":false}}}"#)).unwrap();
        assert!(p.r.params.constraint.is_some());
        // a forced call: its opening in the prompt, its arguments a grammar
        for choice in [r#","tool_choice":{"type":"function","function":{"name":"get_weather"}}"#, r#","tool_choice":"required""#] {
            let p = prepare(&sh, &chat(&format!("{tool}{choice}"))).unwrap();
            assert!(p.pr.text.contains("<tool_call>") && p.r.params.constraint.is_some(), "{choice}: {}", p.pr.text);
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    const XML_CALL: &str = "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>";

    /// What a model whose template writes XML calls is shown: the tools as that template gives them, a call of the conversation and its result in XML, and the opening of a forced call (a named function's through the line after
    /// its name, a required one's through `<function=`) at the end of the prompt, the thinking closed before it.
    #[test]
    fn the_prompt_of_a_request_with_xml_tools() {
        let shared = Shared {
            tok: crate::tok::tests::toy(&["x"]),
            chat: Some(Chat::from_template(qwen35_template()).unwrap()),
            name: "m".into(),
            defaults: Params::default(),
            window: 4096,
            vocab: 2048,
            thinking: None,
            refused: &[],
            rt: crate::runtime::context::Context::process().clone(),
        };
        let text = |body: &str| {
            let v = json::parse(body).unwrap();
            let r = request_of(&v, 4096, &Params::default());
            let api = api_of(&v, true, 2048).unwrap();
            chat_text_of(&shared, &v, &api, &r).map(|(t, i, p, _)| (t, i, p)).unwrap()
        };
        let head = "<|im_start|>system\n# Tools\n\nYou have access to the following functions:\n\n<tools>\n{\"type\": \"function\", \"function\": {\"name\": \"get_weather\", \"description\": \"Get the weather\", \"parameters\": {\"type\": \"object\", \"properties\": {\"city\": {\"type\": \"string\"}}, \"required\": [\"city\"]}}}\n</tools>\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n";
        let msgs = r#""messages":[{"role":"system","content":"Be brief."},{"role":"user","content":"Weather in Paris?"}]"#;
        let (t, thinking, prefix) = text(&format!("{{{msgs},\"tools\":[{WEATHER}]}}"));
        assert!(
            t.starts_with(head)
                && t.ends_with(
                    "</IMPORTANT>\n\nBe brief.<|im_end|>\n<|im_start|>user\nWeather in Paris?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
                ),
            "{t}"
        );
        assert_eq!((thinking, prefix.as_str()), (false, ""));
        // a call and its result in the conversation (the arguments a string, as Open WebUI sends them): the call as the template writes it, the arguments an object
        let conv = r#""messages":[{"role":"user","content":"Weather in Paris?"},{"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"{\"city\":\"Paris\"}"}}]},{"role":"tool","tool_call_id":"call_1","content":"sunny, 20C"}]"#;
        let (t, _, _) = text(&format!("{{{conv},\"tools\":[{WEATHER}]}}"));
        assert!(t.contains("<|im_start|>assistant\n<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call><|im_end|>\n<|im_start|>user\n<tool_response>\nsunny, 20C\n</tool_response><|im_end|>\n"), "{t}");
        // a forced call: the thinking closed in the prompt and the call opened
        let (t, thinking, prefix) =
            text(&format!("{{{msgs},\"tools\":[{WEATHER}],\"tool_choice\":{{\"type\":\"function\",\"function\":{{\"name\":\"get_weather\"}}}}}}"));
        assert!(t.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n<tool_call>\n<function=get_weather>\n"), "{t}");
        assert_eq!((thinking, prefix.as_str()), (false, "<tool_call>\n<function=get_weather>\n"));
        let (t, _, prefix) =
            text(&format!("{{{msgs},\"tools\":[{WEATHER}],\"tool_choice\":\"required\",\"chat_template_kwargs\":{{\"enable_thinking\":true}}}}"));
        assert!(t.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n<tool_call>\n<function="), "{t}");
        assert_eq!(prefix, "<tool_call>\n<function=");
    }

    /// Tool calls through the API as vLLM's `qwen3_coder` parser gives them: the call (an id, the function, its arguments as Python writes them, each parameter typed by the tool's schema) in `tool_calls`, the content null and the
    /// finish reason `tool_calls`; an ordinary answer where tools are offered is an ordinary reply; what is not a call is text; the text before the call is the content; only the first of several calls where
    /// `parallel_tool_calls` is false; none read where the choice is none.
    #[test]
    fn xml_tool_calls_through_the_api() {
        let addr = serve_script_with(XML_CALL, qwen35_template());
        let (st, body) = call(addr, "/v1/chat/completions", &chat_body(""));
        assert_eq!(st, 200, "{body}");
        let d = json::parse(&body).unwrap();
        let c = at(&d, &["choices", "0"]);
        assert_eq!((at(c, &["finish_reason"]).as_str(), at(c, &["message", "content"])), (Some("tool_calls"), &Value::Null));
        let tc = at(c, &["message", "tool_calls"]).as_array().unwrap();
        assert_eq!(tc.len(), 1);
        let id = at(&tc[0], &["id"]).as_str().unwrap();
        assert!(id.starts_with("chatcmpl-tool-") && id.len() == 30, "{id}");
        assert_eq!(
            (at(&tc[0], &["type"]).as_str(), at(&tc[0], &["function", "name"]).as_str(), at(&tc[0], &["function", "arguments"]).as_str()),
            (Some("function"), Some("get_weather"), Some("{\"city\": \"Paris\"}"))
        );
        assert_eq!(
            f64_of(at(&d, &["usage", "completion_tokens"])),
            (XML_CALL.chars().count().div_ceil(4) + 1) as f64,
            "the pieces of the call and the end token"
        );
        // the types of the tool's schema
        let forecast = r#"{"type":"function","function":{"name":"forecast","parameters":{"type":"object","properties":{"city":{"type":"string"},"days":{"type":"integer"},"metric":{"type":"boolean"},"tags":{"type":"array"}}}}}"#;
        let typed = serve_script_with("<tool_call>\n<function=forecast>\n<parameter=city>\n12\n</parameter>\n<parameter=days>\n3\n</parameter>\n<parameter=metric>\nFalse\n</parameter>\n<parameter=tags>\n[\"a\", \"é\"]\n</parameter>\n</function>\n</tool_call>", qwen35_template());
        let (_, body) = call(typed, "/v1/chat/completions", &chat_body_with(forecast, ""));
        assert_eq!(
            at(&json::parse(&body).unwrap(), &["choices", "0", "message", "tool_calls", "0", "function", "arguments"]).as_str(),
            Some("{\"city\": \"12\", \"days\": 3, \"metric\": false, \"tags\": [\"a\", \"é\"]}"),
            "{body}"
        );
        // an ordinary answer, with the tools offered
        let talk = serve_script_with("The weather in Paris is sunny.", qwen35_template());
        let (_, body) = call(talk, "/v1/chat/completions", &chat_body(""));
        let d = json::parse(&body).unwrap();
        assert_eq!(
            (at(&d, &["choices", "0", "message", "content"]).as_str(), at(&d, &["choices", "0", "finish_reason"]).as_str()),
            (Some("The weather in Paris is sunny."), Some("stop")),
            "{body}"
        );
        assert!(at(&d, &["choices", "0", "message"]).get("tool_calls").is_none());
        // text before the call is content
        let both = serve_script_with(&format!("Let me look.\n\n{XML_CALL}"), qwen35_template());
        let (_, body) = call(both, "/v1/chat/completions", &chat_body(""));
        let d = json::parse(&body).unwrap();
        assert_eq!(
            (at(&d, &["choices", "0", "message", "content"]).as_str(), at(&d, &["choices", "0", "finish_reason"]).as_str()),
            (Some("Let me look.\n\n"), Some("tool_calls"))
        );
        // what is not a call is text, never an error: a JSON call (Qwen3's) in a block, a function with no `>`
        for text in [
            "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>",
            "Ok <tool_call>\n<function=get_weather\n</tool_call>",
        ] {
            let bad = serve_script_with(text, qwen35_template());
            let (st, body) = call(bad, "/v1/chat/completions", &chat_body(""));
            let d = json::parse(&body).unwrap();
            assert_eq!(
                (st, at(&d, &["choices", "0", "message", "content"]).as_str(), at(&d, &["choices", "0", "finish_reason"]).as_str()),
                (200, Some(text), Some("stop")),
                "{text}: {body}"
            );
            assert!(at(&d, &["choices", "0", "message"]).get("tool_calls").is_none());
        }
        // two calls: both, or the first where parallel calls are off
        let two = serve_script_with(&format!("{XML_CALL}\n{}", XML_CALL.replace("Paris", "Rome")), qwen35_template());
        let (_, body) = call(two, "/v1/chat/completions", &chat_body(""));
        assert_eq!(at(&json::parse(&body).unwrap(), &["choices", "0", "message", "tool_calls"]).as_array().unwrap().len(), 2);
        let (_, body) = call(two, "/v1/chat/completions", &chat_body(",\"parallel_tool_calls\":false"));
        let one = json::parse(&body).unwrap();
        assert_eq!(at(&one, &["choices", "0", "message", "tool_calls"]).as_array().unwrap().len(), 1);
        assert!(at(&one, &["choices", "0", "message", "tool_calls", "0", "function", "arguments"]).as_str().unwrap().contains("Paris"));
        // tool_choice none: the tools were shown, the text stays text
        let (_, body) = call(addr, "/v1/chat/completions", &chat_body(",\"tool_choice\":\"none\""));
        let d = json::parse(&body).unwrap();
        assert_eq!(
            (at(&d, &["choices", "0", "message", "content"]).as_str(), at(&d, &["choices", "0", "finish_reason"]).as_str()),
            (Some(XML_CALL), Some("stop"))
        );
    }

    /// The same calls streamed: a chunk with the function (its id), chunks of arguments (the `{`, each parameter when its end is written, the `}`), the finish; the text before the call as content; the thinking apart; what is not a
    /// call as text; the first call only where parallel calls are off; each choice its own calls.
    #[test]
    fn xml_tool_calls_streamed() {
        let addr = serve_script_with(XML_CALL, qwen35_template());
        let (st, body) = call(addr, "/v1/chat/completions", &chat_body(",\"stream\":true,\"stream_options\":{\"include_usage\":true}"));
        assert_eq!(st, 200);
        let (think, content, calls, finishes, usage) = put_together(&body);
        assert_eq!((think.as_str(), content.as_str(), finishes), ("", "", vec!["tool_calls".to_string()]));
        assert_eq!(calls.len(), 1);
        assert!(calls[0].0.starts_with("chatcmpl-tool-") && calls[0].1 == "get_weather" && calls[0].2 == "{\"city\": \"Paris\"}", "{calls:?}");
        assert_eq!(usage, Some((XML_CALL.chars().count().div_ceil(4) + 1) as f64));
        // the first chunk of the call names the function and has no arguments; the rest have no id
        let ev = events(&body);
        let starts: Vec<&Value> = ev
            .iter()
            .filter_map(|e| {
                e.get("choices")
                    .and_then(|c| c.as_array())
                    .and_then(|c| c.first())
                    .and_then(|c| c.get("delta"))
                    .and_then(|d| d.get("tool_calls"))
                    .and_then(|t| t.as_array())
                    .and_then(|t| t.first())
            })
            .collect();
        assert!(
            starts[0].get("id").is_some() && at(starts[0], &["function"]).get("arguments").is_none() && starts[1..].iter().all(|t| t.get("id").is_none()),
            "{starts:?}"
        );
        // the arguments come as the parameters are written, a chunk each: the `{`, then `"city": "Paris"`, then the `}`
        let pieces: Vec<&str> = starts.iter().filter_map(|t| t.get("function").and_then(|f| f.get("arguments")).and_then(Value::as_str)).collect();
        assert_eq!(pieces, vec!["{", "\"city\": \"Paris\"", "}"]);
        for e in &ev {
            assert!(
                e.get("choices")
                    .and_then(|c| c.as_array())
                    .and_then(|c| c.first())
                    .and_then(|c| c.get("delta"))
                    .and_then(|d| d.get("tool_calls"))
                    .and_then(|t| t.as_array())
                    .map_or(true, |t| t.len() == 1),
                "a chunk of one update: {e:?}"
            );
        }
        // the types of the tool's schema, as in the whole reply
        let forecast = r#"{"type":"function","function":{"name":"forecast","parameters":{"type":"object","properties":{"city":{"type":"string"},"days":{"type":"integer"},"metric":{"type":"boolean"},"tags":{"type":"array"}}}}}"#;
        let typed = serve_script_with("<tool_call>\n<function=forecast>\n<parameter=city>\n12\n</parameter>\n<parameter=days>\n3\n</parameter>\n<parameter=metric>\nFalse\n</parameter>\n<parameter=tags>\n[\"a\", \"é\"]\n</parameter>\n</function>\n</tool_call>", qwen35_template());
        let (_, body) = call(typed, "/v1/chat/completions", &chat_body_with(forecast, ",\"stream\":true"));
        assert_eq!(put_together(&body).2[0].2, "{\"city\": \"12\", \"days\": 3, \"metric\": false, \"tags\": [\"a\", \"é\"]}", "{body}");
        // an ordinary answer, with the tools offered
        let talk = serve_script_with("The weather in Paris is sunny.", qwen35_template());
        let (_, body) = call(talk, "/v1/chat/completions", &chat_body(",\"stream\":true"));
        let (_, content, calls, finishes, _) = put_together(&body);
        assert_eq!((content.as_str(), calls.len(), finishes), ("The weather in Paris is sunny.", 0, vec!["stop".to_string()]));
        // text before the call is content, and the thinking is apart from both
        let both = serve_script_with(&format!("<think>\nParis, so the weather tool.\n</think>\n\nLet me look.\n\n{XML_CALL}"), qwen35_template());
        let (_, body) = call(both, "/v1/chat/completions", &chat_body(",\"stream\":true"));
        let (think, content, calls, finishes, _) = put_together(&body);
        assert_eq!(
            (think.as_str(), content.trim(), calls.len(), finishes),
            ("Paris, so the weather tool.", "Let me look.", 1, vec!["tool_calls".to_string()]),
            "{body}"
        );
        // what is not a call is text, never an error
        for text in [
            "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>",
            "Ok <tool_call>\n<function=get_weather\n</tool_call>",
        ] {
            let bad = serve_script_with(text, qwen35_template());
            let (_, body) = call(bad, "/v1/chat/completions", &chat_body(",\"stream\":true"));
            let (_, content, calls, finishes, _) = put_together(&body);
            assert_eq!((content.as_str(), calls.len(), finishes), (text, 0, vec!["stop".to_string()]), "{text}");
        }
        // two calls: both, or the first where parallel calls are off
        let two = serve_script_with(&format!("{XML_CALL}\n{}", XML_CALL.replace("Paris", "Rome")), qwen35_template());
        let (_, body) = call(two, "/v1/chat/completions", &chat_body(",\"stream\":true"));
        let (_, _, calls, _, _) = put_together(&body);
        assert_eq!(calls.iter().map(|c| c.2.as_str()).collect::<Vec<_>>(), vec!["{\"city\": \"Paris\"}", "{\"city\": \"Rome\"}"]);
        let (_, body) = call(two, "/v1/chat/completions", &chat_body(",\"parallel_tool_calls\":false,\"stream\":true"));
        assert_eq!(put_together(&body).2.len(), 1);
        // several choices, each its own call (top_k 1 so that every draw is the model's one token)
        let (_, body) = call(addr, "/v1/chat/completions", &chat_body(",\"n\":2,\"temperature\":1,\"top_k\":1,\"stream\":true"));
        let (mut per, mut finishes) = (vec![0usize; 2], vec![0usize; 2]);
        for e in events(&body) {
            if let Some(c) = e.get("choices").and_then(|c| c.as_array()).and_then(|c| c.first()) {
                let i = f64_of(at(c, &["index"])) as usize;
                per[i] += c
                    .get("delta")
                    .and_then(|d| d.get("tool_calls"))
                    .and_then(|t| t.as_array())
                    .map_or(0, |t| t.iter().filter(|x| x.get("id").is_some()).count());
                finishes[i] += (c.get("finish_reason").and_then(Value::as_str) == Some("tool_calls")) as usize;
            }
        }
        assert_eq!((per, finishes), (vec![1, 1], vec![1, 1]), "{body}");
    }

    /// A forced call in the XML format: the reply begins in the prompt with the call's opening and the model writes the rest, which is read as a call (a named function's finish reason is the reply's own, `stop`; a required
    /// one's is `tool_calls`).
    #[test]
    fn a_call_that_is_asked_for_in_xml() {
        let named = r#","tool_choice":{"type":"function","function":{"name":"get_weather"}}"#;
        let addr = serve_script_with("<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>", qwen35_template());
        let (st, body) = call(addr, "/v1/chat/completions", &chat_body(named));
        assert_eq!(st, 200, "{body}");
        let d = json::parse(&body).unwrap();
        assert_eq!(
            (at(&d, &["choices", "0", "finish_reason"]).as_str(), at(&d, &["choices", "0", "message", "content"]).as_str()),
            (Some("stop"), Some("")),
            "{body}"
        );
        assert_eq!(
            (
                at(&d, &["choices", "0", "message", "tool_calls", "0", "function", "name"]).as_str(),
                at(&d, &["choices", "0", "message", "tool_calls", "0", "function", "arguments"]).as_str()
            ),
            (Some("get_weather"), Some("{\"city\": \"Paris\"}"))
        );
        let (_, body) = call(addr, "/v1/chat/completions", &chat_body(&format!("{named},\"stream\":true")));
        let (_, content, calls, finishes, _) = put_together(&body);
        assert_eq!((content.as_str(), finishes), ("", vec!["stop".to_string()]));
        assert!(calls[0].1 == "get_weather" && calls[0].2 == "{\"city\": \"Paris\"}", "{calls:?}");
        // required: the model names the function
        let any = serve_script_with("get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>", qwen35_template());
        let (_, body) = call(any, "/v1/chat/completions", &chat_body(",\"tool_choice\":\"required\""));
        let d = json::parse(&body).unwrap();
        assert_eq!(
            (
                at(&d, &["choices", "0", "finish_reason"]).as_str(),
                at(&d, &["choices", "0", "message", "tool_calls", "0", "function", "name"]).as_str(),
                at(&d, &["choices", "0", "message", "tool_calls", "0", "function", "arguments"]).as_str()
            ),
            (Some("tool_calls"), Some("get_weather"), Some("{\"city\": \"Paris\"}")),
            "{body}"
        );
        let (_, body) = call(any, "/v1/chat/completions", &chat_body(",\"tool_choice\":\"required\",\"stream\":true"));
        let (_, _, calls, finishes, _) = put_together(&body);
        assert_eq!((finishes, calls[0].1.as_str(), calls[0].2.as_str()), (vec!["tool_calls".to_string()], "get_weather", "{\"city\": \"Paris\"}"));
    }

    /// What Open WebUI sends: a chat with its built-in tools (and the usage asked for in the stream), the follow-up after a tool call (the call in the history with its arguments as a string, its result as a tool message), and
    /// the title it asks for in a request of its own: each is served, and the template of a model that has no tools is told so.
    #[test]
    fn open_webui_requests() {
        let tools = r#"[{"type":"function","function":{"name":"search_web","description":"Search the web for information.","parameters":{"type":"object","properties":{"query":{"type":"string","description":"The search query"},"count":{"type":"integer","description":"Results to return","default":5}},"required":["query"]}}},{"type":"function","function":{"name":"get_current_timestamp","description":"Get the current time.","parameters":{"type":"object","properties":{}}}}]"#;
        let addr = serve_script("It is the capital.");
        let chat = format!(
            r#"{{"stream":true,"model":"Qwen/Qwen3-8B","messages":[{{"role":"system","content":"You are a helpful assistant."}},{{"role":"user","content":"What is the capital of France?"}}],"tools":{tools},"stream_options":{{"include_usage":true}},"temperature":0.7,"top_p":0.9,"max_tokens":64}}"#
        );
        let (st, body) = call(addr, "/v1/chat/completions", &chat);
        assert_eq!(st, 200, "{body}");
        let (_, content, calls, finishes, usage) = put_together(&body);
        assert_eq!((content.as_str(), calls.len(), finishes, usage.is_some()), ("It is the capital.", 0, vec!["stop".to_string()], true));
        // the follow-up, with the call and its result in the history
        let follow = format!(
            r#"{{"stream":true,"model":"Qwen/Qwen3-8B","messages":[{{"role":"user","content":"Search for glyd"}},{{"role":"assistant","content":"","tool_calls":[{{"id":"call_ab12","type":"function","function":{{"name":"search_web","arguments":"{{\"query\": \"glyd\"}}"}}}}]}},{{"role":"tool","tool_call_id":"call_ab12","content":"{{\"results\": [\"a\", \"b\"]}}"}}],"tools":{tools},"stream_options":{{"include_usage":true}}}}"#
        );
        let (st, body) = call(addr, "/v1/chat/completions", &follow);
        assert_eq!(st, 200, "{body}");
        assert_eq!(put_together(&body).1, "It is the capital.");
        // its title request: no tools, not streamed, its own limits (it asks for 1,000 tokens, which a model's context of 4,096 has room for; this model's is 1,024, where 1,000 after this prompt
        // is refused, as vLLM refuses it: 900 here)
        let title = r####"{"model":"Qwen/Qwen3-8B","messages":[{"role":"user","content":"### Task:\nGenerate a concise, 3-5 word title with an emoji summarizing the chat history.\n### Output:\nJSON format: { \"title\": \"your concise title here\" }\n### Chat History:\n<chat_history>\nUSER: hi\n</chat_history>"}],"stream":false,"max_tokens":900,"temperature":0.1}"####;
        let (st, body) = call(addr, "/v1/chat/completions", title);
        assert_eq!(st, 200, "{body}");
        assert_eq!(at(&json::parse(&body).unwrap(), &["choices", "0", "message", "content"]).as_str(), Some("It is the capital."));
        // the native function-calling flow: the same chat with the model's call
        let calling = serve_script("<tool_call>\n{\"name\": \"search_web\", \"arguments\": {\"query\": \"glyd\", \"count\": 3}}\n</tool_call>");
        let (_, body) = call(calling, "/v1/chat/completions", &chat);
        let (_, content, calls, finishes, _) = put_together(&body);
        assert_eq!((content.as_str(), calls.len(), finishes), ("", 1, vec!["tool_calls".to_string()]));
        assert_eq!((calls[0].1.as_str(), calls[0].2.as_str()), ("search_web", "{\"query\": \"glyd\", \"count\": 3}"));
    }

    /// The ranges of the sampling fields and what is refused, in vLLM's words, before any work is done: a value that is not a number, out of its range or out of the vocabulary, and each field of vLLM's that this
    /// engine does not do (constrained decoding names the way to vLLM), where another value than the one that means it was not asked for.
    #[test]
    fn values_and_what_is_not_done() {
        let addr = serve_script("x");
        let ask = |extra: &str| call(addr, "/v1/completions", &format!("{{\"prompt\":\"hi\"{extra}}}"));
        for (extra, want) in [
            (",\"temperature\":-1", "temperature must be non-negative, got -1."),
            (",\"temperature\":2.5", "temperature must be in [0, 2], got 2.5."),
            (",\"temperature\":\"hot\"", "`temperature` must be a number."),
            (",\"top_p\":0", "top_p must be in (0, 1], got 0."),
            (",\"top_p\":1.5", "top_p must be in (0, 1], got 1.5."),
            (",\"top_k\":-5", "top_k must be 0 (disable), or at least 1, got -5."),
            (",\"min_p\":2", "min_p must be in [0, 1], got 2."),
            (",\"presence_penalty\":3", "presence_penalty must be in [-2, 2], got 3."),
            (",\"frequency_penalty\":-2.5", "frequency_penalty must be in [-2, 2], got -2.5."),
            (",\"repetition_penalty\":0", "repetition_penalty must be greater than zero, got 0."),
            (",\"max_tokens\":0", "max_tokens must be at least 1, got 0."),
            (",\"min_tokens\":-1", "min_tokens must be greater than or equal to 0, got -1."),
            (",\"max_tokens\":4,\"min_tokens\":10", "min_tokens must be less than or equal to max_tokens=4, got 10."),
            (",\"stop\":\"\"", "stop cannot contain an empty string."),
            (",\"stop\":[\"a\",\"\"]", "stop cannot contain an empty string."),
            (",\"stop\":5", "`stop` must be a string or a list of strings."),
            (",\"stop_token_ids\":[999999]", "token_id(s) [999999] in stop_token_ids contain out-of-vocab token ids. Vocabulary size: 2048"),
            (",\"allowed_token_ids\":[]", "allowed_token_ids is not None and empty!"),
            (",\"allowed_token_ids\":[3000]", "in allowed_token_ids contain out-of-vocab"),
            (",\"seed\":1.5", "`seed` must be an integer."),
            (",\"guided_whitespace_pattern\":\"x\"", "guided_whitespace_pattern (constrained decoding) is not supported"),
            (",\"guided_choice\":[]", "guided_choice must be a list of strings"),
            (",\"response_format\":{\"type\":\"xml\"}", "response_format of type xml is not supported"),
            (",\"bad_words\":[\"x\"]", "bad_words is not supported by this engine yet (GLYD_ENGINE=0 runs vLLM, which has it)"),
            (",\"prompt_logprobs\":1", "prompt_logprobs is not supported"),
            (",\"suffix\":\"x\"", "no fill-in-the-middle tokens"),
            (",\"best_of\":3", "best_of more than n is not supported"),
            (",\"use_beam_search\":true", "use_beam_search is not supported"),
            (",\"priority\":2", "priority is not supported"),
            (",\"cache_salt\":7", "cache_salt must be a string that is not empty"),
            (",\"cache_salt\":\"\"", "cache_salt must be a string that is not empty"),
            (",\"logprobs\":1,\"echo\":true", "logprobs with echo"),
        ] {
            let (st, body) = ask(extra);
            assert!(st == 400 && body.contains(want), "{extra}: {st} {body}");
        }
        // what is not asked for is not refused: null, and the value that means it
        for ok in [
            ",\"response_format\":{\"type\":\"text\"}",
            ",\"response_format\":null",
            ",\"use_beam_search\":false",
            ",\"priority\":0",
            ",\"bad_words\":[]",
            ",\"suffix\":null",
            ",\"best_of\":1",
            ",\"user\":\"u\"",
            ",\"temperature\":0.001",
            ",\"top_k\":-1",
            ",\"stop\":null",
            ",\"stop\":[\"zz\"]",
            ",\"n\":1",
        ] {
            let (st, body) = ask(ok);
            assert_eq!(st, 200, "{ok}: {body}");
        }
        for (extra, want) in [
            (",\"echo\":true", "echo of a chat reply is not supported"),
            (",\"continue_final_message\":true", "continue_final_message is not supported"),
            (",\"functions\":[{\"name\":\"f\"}]", "functions is not supported"),
            (",\"tool_choice\":\"auto\"", "`tools` must be set"),
            (",\"reasoning_effort\":\"extreme\"", "reasoning_effort"),
            (",\"documents\":[1]", "`documents`"),
        ] {
            let (st, body) =
                call(addr, "/v1/chat/completions", &format!("{{\"messages\":[{{\"role\":\"user\",\"content\":\"hi\"}}],\"max_tokens\":4{extra}}}"));
            assert!(st == 400 && body.contains(want), "{extra}: {st} {body}");
        }
        // a temperature above 0 and below 0.01 is 0.01 (it is not greedy: n of several is allowed)
        assert_eq!(request_of(&json::parse(r#"{"temperature":0.001}"#).unwrap(), 4096, &Params::default()).params.temperature, 0.01);
        assert_eq!(request_of(&json::parse(r#"{"temperature":0}"#).unwrap(), 4096, &Params::default()).params.temperature, 0.0);
    }

    /// The fields that shape a reply, honoured as vLLM's are: the stop string kept in the text, the special tokens left out of it unless asked for, the stop tokens, the tokens allowed, a plain completion that echoes its
    /// prompt, its prompts given in each way vLLM takes them (each a choice of its own, with its index), its default length, the id of the request, and a template that is not given the generation prompt.
    #[test]
    fn fields_that_shape_a_reply() {
        let addr = serve_script("Hello world END and more<|im_end|> tail");
        let text = |extra: &str| -> String {
            let (st, body) = call(addr, "/v1/completions", &format!("{{\"prompt\":\"hi\",\"max_tokens\":100,\"temperature\":0{extra}}}"));
            assert_eq!(st, 200, "{body}");
            at(&json::parse(&body).unwrap(), &["choices", "0", "text"]).as_str().unwrap().to_string()
        };
        assert_eq!(text(""), "Hello world END and more tail", "the special token is left out of the text");
        assert_eq!(text(",\"skip_special_tokens\":false"), "Hello world END and more<|im_end|> tail");
        assert_eq!(text(",\"stop\":[\"END\"]"), "Hello world ");
        assert_eq!(text(",\"stop\":[\"END\"],\"include_stop_str_in_output\":true"), "Hello world END");
        assert_eq!(text(",\"echo\":true,\"stop\":[\"END\"]"), "hiHello world ");
        // streamed with the prompt echoed: its text first
        let (_, body) =
            call(addr, "/v1/completions", "{\"prompt\":\"hi\",\"max_tokens\":100,\"temperature\":0,\"echo\":true,\"stream\":true,\"stop\":[\"END\"]}");
        let built: String = events(&body)
            .iter()
            .filter_map(|e| {
                e.get("choices").and_then(|c| c.as_array()).and_then(|c| c.first()).and_then(|c| c.get("text")).and_then(Value::as_str).map(str::to_string)
            })
            .collect();
        assert_eq!(built, "hiHello world ");
        // the length a plain completion has where it gives none (16 tokens: the script is longer), and the id it asked for
        let long = serve_script(&"ab".repeat(40));
        let (_, body) = call(long, "/v1/completions", "{\"prompt\":\"hi\",\"temperature\":0,\"request_id\":\"mine-7\"}");
        let d = json::parse(&body).unwrap();
        assert_eq!(
            (f64_of(at(&d, &["usage", "completion_tokens"])), at(&d, &["id"]).as_str(), at(&d, &["choices", "0", "finish_reason"]).as_str()),
            (16.0, Some("cmpl-mine-7"), Some("length")),
            "{body}"
        );
        let (_, body) = call(long, "/v1/chat/completions", "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"temperature\":0}");
        assert!(f64_of(at(&json::parse(&body).unwrap(), &["usage", "completion_tokens"])) > 16.0, "a chat has the context for its length");
        // the prompts of a plain completion: a string, a list of strings, token ids, a list of them: each its own choice, the prompt tokens added up
        let tokens = |prompt: &str| -> (Vec<String>, f64) {
            let (st, body) = call(addr, "/v1/completions", &format!("{{\"prompt\":{prompt},\"max_tokens\":3,\"temperature\":0}}"));
            assert_eq!(st, 200, "{prompt}: {body}");
            let d = json::parse(&body).unwrap();
            (
                at(&d, &["choices"])
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|c| format!("{}:{}", f64_of(at(c, &["index"])), at(c, &["text"]).as_str().unwrap().chars().take(5).collect::<String>()))
                    .collect(),
                f64_of(at(&d, &["usage", "prompt_tokens"])),
            )
        };
        let (one, p1) = tokens(r#""one two three""#);
        assert_eq!((one.len(), p1), (1, 3.0));
        let (two, p2) = tokens(r#"["a b", "c d e", "f"]"#);
        assert_eq!((two.iter().map(|c| &c[..2]).collect::<Vec<_>>(), p2), (vec!["0:", "1:", "2:"], 6.0));
        let (ids, p3) = tokens("[5, 6, 7, 8]");
        assert_eq!((ids.len(), p3), (1, 4.0));
        let (batch, p4) = tokens("[[1, 2], [3, 4, 5]]");
        assert_eq!((batch.len(), p4), (2, 5.0));
        // two prompts of two choices: four, in order
        let (_, body) = call(addr, "/v1/completions", r#"{"prompt":["a b","c d"],"n":2,"max_tokens":2,"temperature":1.0,"top_k":5,"seed":3}"#);
        let idx: Vec<f64> = at(&json::parse(&body).unwrap(), &["choices"]).as_array().unwrap().iter().map(|c| f64_of(at(c, &["index"]))).collect();
        assert_eq!(idx, vec![0.0, 1.0, 2.0, 3.0]);
        for (prompt, want) in [
            ("\"\"", "Either prompt or prompt_embeds must be provided and non-empty."),
            ("[]", "Either prompt or prompt_embeds must be provided and non-empty."),
            ("[1, \"a\"]", "`prompt` must be a string"),
            ("[[1], \"a\"]", "`prompt` must be a string"),
            ("[99999]", "not a token of the vocabulary"),
            ("5", "`prompt` must be a string"),
        ] {
            let (st, body) = call(addr, "/v1/completions", &format!("{{\"prompt\":{prompt}}}"));
            assert!(st == 400 && body.contains(want), "{prompt}: {st} {body}");
        }
        let (st, body) = call(addr, "/v1/completions", "{\"max_tokens\":3}");
        assert!(st == 400 && body.contains("Either prompt or prompt_embeds"), "{body}");
        // too many sequences for one request
        let (st, body) = call(addr, "/v1/completions", &format!("{{\"prompt\":[{}],\"n\":2,\"max_tokens\":2,\"temperature\":1}}", vec!["\"a\""; 40].join(",")));
        assert!(st == 400 && body.contains("takes up to 64"), "{body}");
        // the tokens allowed are the only ones: a model that would say others
        let free = serve_fake(Fake::new(64, 32), None, 4);
        let (_, body) = call(free, "/v1/completions", r#"{"prompt":"hi","max_tokens":12,"temperature":0,"allowed_token_ids":[5,6]}"#);
        let said = at(&json::parse(&body).unwrap(), &["choices", "0", "text"]).as_str().unwrap().to_string();
        assert!(said.split_whitespace().count() == 12 && said.split_whitespace().all(|w| w == "w5" || w == "w6"), "{said:?}");
        // a stop token ends the reply before it, counted
        let (_, body) = call(free, "/v1/completions", r#"{"prompt":"hi","max_tokens":12,"temperature":0}"#);
        let plain = at(&json::parse(&body).unwrap(), &["choices", "0", "text"]).as_str().unwrap().to_string();
        let words: Vec<&str> = plain.split_whitespace().collect();
        let stop: u32 = words[3].trim_start_matches('w').parse().unwrap();
        let (_, body) = call(free, "/v1/completions", &format!(r#"{{"prompt":"hi","max_tokens":12,"temperature":0,"stop_token_ids":[{stop}]}}"#));
        let d = json::parse(&body).unwrap();
        let cut = at(&d, &["choices", "0", "text"]).as_str().unwrap().to_string();
        let first = words.iter().position(|w| *w == format!("w{stop}")).unwrap();
        assert_eq!(
            (cut.split_whitespace().count(), at(&d, &["choices", "0", "finish_reason"]).as_str(), f64_of(at(&d, &["usage", "completion_tokens"]))),
            (first, Some("stop"), (first + 1) as f64),
            "{cut:?} {body}"
        );
        // a chat that is not given the generation prompt, and effort none that closes the thinking
        let shared = Shared {
            tok: crate::tok::tests::toy(&["x"]),
            chat: Some(Chat::from_template(&qwen3_template()).unwrap()),
            name: "m".into(),
            defaults: Params::default(),
            window: 4096,
            vocab: 2048,
            thinking: None,
            refused: &[],
            rt: crate::runtime::context::Context::process().clone(),
        };
        let prompt_text = |body: &str| {
            let v = json::parse(body).unwrap();
            chat_text_of(&shared, &v, &api_of(&v, true, 2048).unwrap(), &request_of(&v, 4096, &Params::default())).unwrap().0
        };
        let msgs = r#""messages":[{"role":"user","content":"hi"}]"#;
        assert!(prompt_text(&format!("{{{msgs}}}")).ends_with("<|im_start|>assistant\n"));
        assert!(prompt_text(&format!("{{{msgs},\"add_generation_prompt\":false}}")).ends_with("hi<|im_end|>\n"));
        assert!(prompt_text(&format!("{{{msgs},\"reasoning_effort\":\"none\"}}")).ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"));
        assert!(prompt_text(&format!("{{{msgs},\"reasoning_effort\":\"high\"}}")).ends_with("<|im_start|>assistant\n"));
        assert!(
            prompt_text(&format!("{{{msgs},\"reasoning_effort\":\"none\",\"chat_template_kwargs\":{{\"enable_thinking\":true}}}}"))
                .ends_with("<|im_start|>assistant\n"),
            "a variable the request names wins over the effort"
        );
    }

    /// /v1/glyd/info, behind the key: what `serve` set (the model, its levels), the build, the limits, the fields read, of no effect and refused (the engine's own refusals among them), whether the engine has
    /// failed, the requests, and the memory: the model's parts as it holds them, the rest of the engine's buffers as its working memory, the peak since the load began; the GPU's own numbers null where no GPU is named.
    #[test]
    fn the_server_says_what_it_is_and_what_it_holds() {
        let held = crate::dev::Held { weights: 5 << 30, kv: 1 << 28, state: 1 << 20, kv_type: "bf16", state_type: "fp32" };
        let about = vec![
            ("model", tools::obj(vec![("id", Value::from("Qwen/Qwen3.5-9B:swift"))])),
            ("level", tools::obj(vec![("requested", Value::from("swift")), ("effective", Value::from("swift"))])),
        ];
        assert!(crate::runtime::context::Context::process()
            .info
            .set(Info { about, gpu: None, held: Some(held), concurrency: 1, window: 4096, refused: ONE_AT_A_TIME, started: 1.5 })
            .is_ok());
        let (tx, _rx) = std::sync::mpsc::channel();
        let addr = listen("127.0.0.1:0", "Qwen/Qwen3.5-9B:swift", 4096, Some("k".into()), false, tx).unwrap();
        assert_eq!(ask(addr, "GET", "/v1/glyd/info", &[("Host", "x")]).0, 401);
        let info = || json::parse(&ask(addr, "GET", "/v1/glyd/info", &[("Host", "x"), ("Authorization", "Bearer k")]).2).unwrap();
        let d = info();
        assert_eq!(
            (at(&d, &["object"]).as_str(), at(&d, &["api"]).as_str(), at(&d, &["model", "id"]).as_str(), at(&d, &["level", "effective"]).as_str()),
            (Some("glyd.info"), Some(API), Some("Qwen/Qwen3.5-9B:swift"), Some("swift"))
        );
        assert_eq!(at(&d, &["build", "version"]).as_str(), Some(env!("CARGO_PKG_VERSION")));
        assert_eq!((f64_of(at(&d, &["limits", "context_tokens"])), f64_of(at(&d, &["limits", "concurrency"]))), (4096.0, 1.0));
        let list = |k: &str| at(&d, &["fields", k]).as_array().unwrap().iter().map(|x| x.as_str().unwrap().to_string()).collect::<Vec<_>>();
        for f in ["logprobs", "min_tokens", "guided_whitespace_pattern", "use_beam_search"] {
            assert!(list("refused").contains(&f.to_string()), "{f}");
        }
        assert!(
            !list("read").contains(&"logprobs".to_string())
                && list("read").contains(&"max_tokens".to_string())
                && list("no_effect").contains(&"user".to_string())
        );
        assert!(matches!(at(&d, &["state"]).as_str(), Some("ready" | "failed")));
        let m = at(&d, &["memory"]);
        assert_eq!(
            (f64_of(at(m, &["weights_bytes"])), f64_of(at(m, &["kv_bytes"])), f64_of(at(m, &["state_bytes"]))),
            ((5u64 << 30) as f64, (1u64 << 28) as f64, (1u64 << 20) as f64)
        );
        let parts = ((5u64 << 30) + (1 << 28) + (1 << 20)) as f64;
        assert_eq!(
            f64_of(at(m, &["scratch_bytes"])),
            (f64_of(at(m, &["engine_bytes"])) - parts).max(0.0),
            "the working memory is what the engine's buffers hold besides the model's parts"
        );
        assert!(f64_of(at(m, &["peak_bytes"])) >= f64_of(at(m, &["engine_bytes"])) && f64_of(at(m, &["peak_since"])) == 1.5);
        assert_eq!((at(m, &["gpu"]), at(m, &["device_total_bytes"]), at(m, &["device_free_bytes"])), (&Value::Null, &Value::Null, &Value::Null));
        // after a failure of the engine it says so (nothing is sure of the GPU after one), with when and what
        crate::runtime::context::Context::process().fail("cuLaunchKernel: an illegal memory access was encountered (700)");
        let d = info();
        assert!(at(&d, &["state"]).as_str() == Some("failed") && at(&d, &["error", "message"]).as_str().is_some(), "{d:?}");
    }

    /// A streamed request written, and its answer read up to its first event: the reader, kept open, and the reply's id.
    fn first_event(addr: std::net::SocketAddr, body: &str) -> (BufReader<TcpStream>, String) {
        let mut c = TcpStream::connect(addr).unwrap();
        c.write_all(
            format!("POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}", body.len())
                .as_bytes(),
        )
        .unwrap();
        let mut r = BufReader::new(c);
        let mut line = String::new();
        while r.read_line(&mut line).unwrap() > 0 {
            if let Some(d) = line.trim().strip_prefix("data: ") {
                return (r, at(&json::parse(d).unwrap(), &["id"]).as_str().unwrap().to_string());
            }
            line.clear();
        }
        panic!("the stream ended before its first event");
    }

    /// A request by the id its reply carries: running while it runs; a cancel (POST /v1/glyd/requests/ID/cancel) makes it cancelling, then ended, cancelled, once the engine has let go of its sequences, its stream
    /// ended as a client's stop ends one (a finish reason, [DONE]); a client that goes away ends it the same way (`client gone`); a running request's id is not taken by another; a request that runs to its end is
    /// ended, finished, before its answer is sent; an id no request has is not found.
    #[test]
    fn a_request_by_its_id_its_cancel_and_its_end() {
        let mut f = Fake::new(64, 32);
        f.decode_delay = std::time::Duration::from_millis(20);
        let addr = serve_fake(f, None, 4);
        let status = |id: &str| -> Value {
            let (st, _, body) = ask(addr, "GET", &format!("/v1/glyd/requests/{id}"), &[("Host", "x")]);
            assert_eq!(st, 200, "{body}");
            json::parse(&body).unwrap()
        };
        let ended = |id: &str| -> Value {
            for _ in 0..500 {
                let s = status(id);
                if at(&s, &["state"]).as_str() == Some("ended") {
                    return s;
                }
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
            panic!("{id} did not end");
        };
        let body = r#"{"messages":[{"role":"user","content":"hello there"}],"max_tokens":400,"temperature":0,"stream":true,"request_id":"job-1"}"#;
        let (mut r, id) = first_event(addr, body);
        assert_eq!(id, "chatcmpl-job-1");
        let s = status(&id);
        assert_eq!((at(&s, &["state"]).as_str(), at(&s, &["cancel"]), at(&s, &["ended"])), (Some("running"), &Value::Null, &Value::Null), "{s:?}");
        let (st, b) = call(addr, "/v1/chat/completions", &body.replace(",\"stream\":true", ""));
        assert!(st == 400 && b.contains("chatcmpl-job-1 is running"), "{st} {b}");
        let (st, _, b) = ask(addr, "POST", &format!("/v1/glyd/requests/{id}/cancel"), &[("Host", "x"), ("Content-Length", "0")]);
        assert!(st == 200 && matches!(at(&json::parse(&b).unwrap(), &["state"]).as_str(), Some("cancelling" | "ended")), "{st} {b}");
        let s = ended(&id);
        assert_eq!((at(&s, &["cancel"]).as_str(), at(&s, &["how"]).as_str()), (Some("asked"), Some("cancelled")), "{s:?}");
        let mut rest = String::new();
        r.read_to_string(&mut rest).unwrap();
        assert!(rest.contains("\"finish_reason\":\"stop\"") && rest.contains("data: [DONE]"), "{rest}");
        // ended, its id may be given again; a client that goes away ends its request as a cancel does
        let (r, id) = first_event(addr, body);
        drop(r);
        let s = ended(&id);
        assert_eq!((at(&s, &["cancel"]).as_str(), at(&s, &["how"]).as_str()), (Some("client gone"), Some("cancelled")), "{s:?}");
        let (st, b) = call(addr, "/v1/chat/completions", r#"{"messages":[{"role":"user","content":"hi"}],"max_tokens":3,"temperature":0}"#);
        assert_eq!(st, 200, "{b}");
        let s = status(at(&json::parse(&b).unwrap(), &["id"]).as_str().unwrap());
        assert_eq!((at(&s, &["state"]).as_str(), at(&s, &["how"]).as_str()), (Some("ended"), Some("finished")), "{s:?}");
        assert_eq!(ask(addr, "GET", "/v1/glyd/requests/nothing", &[("Host", "x")]).0, 404);
        assert_eq!(ask(addr, "POST", "/v1/glyd/requests/nothing/cancel", &[("Host", "x"), ("Content-Length", "0")]).0, 404);
        assert_eq!(ask(addr, "GET", "/v1/glyd/requests/nothing/cancel", &[("Host", "x")]).0, 405);
    }

    /// What a level's engine does not do (ONE_AT_A_TIME: log-probabilities, a least length) is refused in words where it is asked for, and not where its value means it is not; a length the context has no room
    /// for after the prompt is refused, not cut short (one that fits to the last token is served); and /v1/glyd/tokenize counts a prompt as its reply's usage does, a chat's or a plain completion's, and runs nothing.
    #[test]
    fn what_the_engine_does_not_do_is_refused_and_a_prompt_is_counted() {
        let f = Fake::new(64, 32);
        let (ran, window) = (f.prefilled.clone(), f.store.len() * crate::kv::PAGE);
        let addr = serve_fake_with(f, None, 4, false, None, ONE_AT_A_TIME);
        let chat = |extra: &str| {
            call(addr, "/v1/chat/completions", &format!("{{\"messages\":[{{\"role\":\"user\",\"content\":\"hello there\"}}],\"temperature\":0{extra}}}"))
        };
        for (extra, want) in [
            (",\"logprobs\":true", "logprobs is not supported by the engine this model runs on yet"),
            (",\"min_tokens\":2", "min_tokens is not supported by the engine this model runs on yet"),
        ] {
            let (st, body) = chat(extra);
            assert!(st == 400 && body.contains(want), "{extra}: {st} {body}");
        }
        for ok in [",\"logprobs\":false", ",\"logprobs\":null", ",\"min_tokens\":0", ""] {
            let (st, body) = chat(&format!(",\"max_tokens\":4{ok}"));
            assert_eq!(st, 200, "{ok}: {body}");
        }
        let (st, body) = call(addr, "/v1/completions", r#"{"prompt":"hi","max_tokens":2,"logprobs":0}"#);
        assert!(st == 400 && body.contains("logprobs is not supported by the engine"), "{body}");
        let (_, body) = chat(",\"max_tokens\":4");
        let used = f64_of(at(&json::parse(&body).unwrap(), &["usage", "prompt_tokens"])) as usize;
        let before = ran.load(Relaxed);
        let (st, body) = call(addr, "/v1/glyd/tokenize", r#"{"messages":[{"role":"user","content":"hello there"}]}"#);
        assert_eq!(st, 200, "{body}");
        let d = json::parse(&body).unwrap();
        assert_eq!((f64_of(at(&d, &["prompt_tokens"])) as usize, f64_of(at(&d, &["max_model_len"])) as usize), (used, window));
        let (_, body) = call(addr, "/v1/glyd/tokenize", r#"{"prompt":["a b", "c d e"]}"#);
        assert_eq!(f64_of(at(&json::parse(&body).unwrap(), &["prompt_tokens"])), 5.0);
        assert_eq!(ran.load(Relaxed), before, "a count runs nothing");
        let (st, body) = call(addr, "/v1/glyd/tokenize", r#"{"messages":[{"role":"user","content":"x"}],"logprobs":true}"#);
        assert!(st == 400 && body.contains("logprobs is not supported"), "a count refuses what the request would be refused for: {body}");
        // a length the context has no room for: refused with the numbers, whichever field asks for it; one that fills the context to its last token is served
        for field in ["max_tokens", "max_completion_tokens"] {
            let (st, body) = chat(&format!(",\"{field}\":{}", window - used + 1));
            assert!(
                st == 400 && body.contains(&format!("This prompt has {used} tokens and asks for up to {} more, {} in all", window - used + 1, window + 1)),
                "{field}: {st} {body}"
            );
        }
        let (st, body) = chat(&format!(",\"max_tokens\":{}", window - used));
        assert_eq!(st, 200, "{body}");
    }

    /// A level's engine (`work_gen`: one sequence at a time) does not run a sequence asked to end before its turn (a cancel, its client gone): it ends at once, stopped; the next runs.
    #[test]
    fn a_sequence_asked_to_end_before_its_turn_is_not_run() {
        struct Count(usize);
        impl Gen for Count {
            fn capacity(&self) -> usize {
                64
            }
            fn reply(&mut self, _: &[u32], _: &Reply, each: &mut dyn FnMut(u32) -> bool) -> Result<(Vec<u32>, Finish)> {
                self.0 += 1;
                each(7);
                Ok((vec![7], Finish::Length))
            }
        }
        let shared = Arc::new(Shared {
            tok: crate::tok::tests::toy(&["x"]),
            chat: None,
            name: "m".into(),
            defaults: Params::default(),
            window: 64,
            vocab: 2048,
            thinking: None,
            refused: ONE_AT_A_TIME,
            rt: crate::runtime::context::Context::process().clone(),
        });
        let (tx, rx) = std::sync::mpsc::channel();
        let seq = |abort: bool| {
            let (out, events) = std::sync::mpsc::channel();
            let reply = Reply { params: Params::default(), max_new: 4, ignore_eos: false, logprobs: None, stop_ids: Vec::new() };
            tx.send(Job::Seq(Submit {
                prompt: vec![1, 2],
                reply,
                out,
                abort: Arc::new(AtomicBool::new(abort)),
                resumed: Default::default(),
                cached: Default::default(),
                salt: 0,
                images: Vec::new(),
            }))
            .unwrap();
            events
        };
        let (gone, kept) = (seq(true), seq(false));
        tx.send(Job::Stop).unwrap();
        let mut c = Count(0);
        work_gen(&mut c, shared, &rx, &tx);
        assert!(matches!(gone.recv().unwrap(), Ev::End(Ok(Finish::Stopped))));
        assert!(matches!(kept.recv().unwrap(), Ev::Token(7)) && matches!(kept.recv().unwrap(), Ev::End(Ok(Finish::Length))));
        assert_eq!(c.0, 1, "the sequence asked to end was not run");
    }
}

/// The translations of the other APIs' request bodies into the completions form, as the fuzz targets (fuzz/) call them: Anthropic's Messages, OpenAI's Responses, Ollama's chat and generate.
#[doc(hidden)]
pub mod fuzzing {
    use glyd_gpu::json::Value;

    pub fn anthropic(v: &Value) -> Result<Value, String> {
        super::anthropic::openai_of_with(v, false)
    }

    pub fn responses(v: &Value) -> Result<Value, String> {
        super::responses::chat_of(v)
    }

    pub fn ollama_chat(v: &Value) -> Result<Value, String> {
        super::ollama::chat_of(v)
    }

    pub fn ollama_generate(v: &Value) -> Result<(Value, bool), String> {
        super::ollama::generate_of(v)
    }
}
