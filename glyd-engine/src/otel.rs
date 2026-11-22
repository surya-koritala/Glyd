//! OpenTelemetry traces per request: a span for the sequence and one each for its wait in the queue, its prompt pass and its decode, sent to an OTLP collector over HTTP (JSON, `POST /v1/traces`).
//! Off unless an endpoint is configured; nothing is sent, started or allocated beyond a clock for each sequence where it is off. Configuration is OpenTelemetry's own variables:
//!
//! - `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` (the full URL) or `OTEL_EXPORTER_OTLP_ENDPOINT` (a base: `/v1/traces` is added), `OTEL_EXPORTER_OTLP_HEADERS` (`k=v,k2=v2`, for a collector's key),
//!   `OTEL_SERVICE_NAME` (default `glyd`), `OTEL_SDK_DISABLED=true` (off whatever else is set). Only `http/json` is sent: `OTEL_EXPORTER_OTLP_PROTOCOL=grpc` or `http/protobuf` is said and not honoured.
//!
//! The spans of a sequence: `glyd.request` (the whole, from the scheduler getting it to its end) with the children `glyd.queue` (waiting for room), `glyd.prefill` (the prompt, to the first token) and `glyd.decode` (first token to
//! the end). Attributes: the model, the prompt and generated tokens, why it ended. Spans are batched by a thread of their own (up to 100 or a second), a full queue drops spans (counted, `glyd_trace_spans_dropped_total`), and an
//! export that fails is counted and said at most every 30 seconds: a collector that is down never slows or fails a request.

use serde_json::{json, Value};
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::mpsc::{sync_channel, Receiver, RecvTimeoutError, SyncSender};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// The clock of one sequence: when it began (the scheduler got it), when it was first taken in (its queue ends), when its first token was written, and its prompt's length.
#[derive(Clone, Copy, Debug)]
pub struct Clock {
    began: Instant,
    began_unix_ns: u64,
    admitted: Option<Instant>,
    first: Option<Instant>,
    prompt: usize,
}

impl Clock {
    pub fn new(prompt: usize) -> Clock {
        Clock { began: Instant::now(), began_unix_ns: unix_ns(), admitted: None, first: None, prompt }
    }
    /// The sequence is taken in (the first time only: one that was sent back to wait keeps its first).
    pub fn admit(&mut self) {
        self.admitted.get_or_insert_with(Instant::now);
    }
    /// A token is written (the first one only is kept).
    pub fn token(&mut self) {
        self.first.get_or_insert_with(Instant::now);
    }
}

fn unix_ns() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_nanos() as u64)
}

/// One finished span, as it is sent.
#[derive(Clone, Debug, PartialEq)]
pub struct Span {
    pub trace: u128,
    pub id: u64,
    pub parent: Option<u64>,
    pub name: &'static str,
    pub start_ns: u64,
    pub end_ns: u64,
    pub attrs: Vec<(&'static str, Attr)>,
    pub error: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Attr {
    Str(String),
    Int(i64),
}

/// The spans of a finished sequence: the whole and its three parts (a part that did not happen, a queue the engine has no clock for, a decode of a sequence that made no token, is left out).
pub fn spans_of(c: &Clock, now: Instant, how: &str, generated: usize, model: &str) -> Vec<Span> {
    let at = |t: Instant| c.began_unix_ns + t.saturating_duration_since(c.began).as_nanos() as u64;
    let end = at(now);
    let (trace, root) = (new_trace(), new_span());
    let mut v = vec![Span {
        trace,
        id: root,
        parent: None,
        name: "glyd.request",
        start_ns: c.began_unix_ns,
        end_ns: end,
        attrs: vec![
            ("gen_ai.request.model", Attr::Str(model.to_string())),
            ("glyd.prompt_tokens", Attr::Int(c.prompt as i64)),
            ("glyd.generated_tokens", Attr::Int(generated as i64)),
            ("glyd.finish_reason", Attr::Str(how.to_string())),
        ],
        error: how == "error",
    }];
    let child = |name: &'static str, a: u64, b: u64| Span {
        trace,
        id: new_span(),
        parent: Some(root),
        name,
        start_ns: a,
        end_ns: b.max(a),
        attrs: Vec::new(),
        error: false,
    };
    let admitted = c.admitted.map(at);
    let first = c.first.map(at);
    if let Some(a) = admitted {
        v.push(child("glyd.queue", c.began_unix_ns, a));
    }
    // (the prompt pass: from the sequence being taken in, or from its start where the engine has no queue, to its first token; a sequence that ended before one has the pass to its end)
    let pass_from = admitted.unwrap_or(c.began_unix_ns);
    v.push(child("glyd.prefill", pass_from, first.unwrap_or(end)));
    if let Some(f) = first {
        v.push(child("glyd.decode", f, end));
    }
    v
}

// ---- ids ------------------------------------------------------------------------------------------------------------------------------------------

fn mix() -> u128 {
    use sha2::{Digest, Sha256};
    static N: AtomicU64 = AtomicU64::new(0);
    let mut h = Sha256::new();
    h.update(N.fetch_add(1, Relaxed).to_le_bytes());
    h.update(unix_ns().to_le_bytes());
    h.update(std::process::id().to_le_bytes());
    h.update(format!("{:?}", std::thread::current().id()).as_bytes());
    let d = h.finalize();
    u128::from_le_bytes(d[..16].try_into().expect("16 bytes"))
}

fn new_trace() -> u128 {
    mix() | 1
}

fn new_span() -> u64 {
    (mix() as u64) | 1
}

// ---- the OTLP JSON ----------------------------------------------------------------------------------------------------------------------------

fn value(a: &Attr) -> Value {
    match a {
        Attr::Str(s) => json!({"stringValue": s}),
        Attr::Int(i) => json!({"intValue": i.to_string()}),
    }
}

/// The request body of `POST /v1/traces` for these spans (OTLP/HTTP JSON: ids as hex, times and integers as strings).
pub fn otlp_json(service: &str, spans: &[Span]) -> String {
    let spans: Vec<Value> = spans
        .iter()
        .map(|s| {
            let mut o = json!({
                "traceId": format!("{:032x}", s.trace),
                "spanId": format!("{:016x}", s.id),
                "name": s.name,
                "kind": if s.parent.is_none() { 2 } else { 1 },
                "startTimeUnixNano": s.start_ns.to_string(),
                "endTimeUnixNano": s.end_ns.to_string(),
                "attributes": s.attrs.iter().map(|(k, v)| json!({"key": k, "value": value(v)})).collect::<Vec<_>>(),
                "status": {"code": if s.error { 2 } else { 1 }},
            });
            if let Some(p) = s.parent {
                o["parentSpanId"] = json!(format!("{p:016x}"));
            }
            o
        })
        .collect();
    json!({"resourceSpans": [{
        "resource": {"attributes": [
            {"key": "service.name", "value": {"stringValue": service}},
            {"key": "service.version", "value": {"stringValue": env!("CARGO_PKG_VERSION")}},
        ]},
        "scopeSpans": [{"scope": {"name": "glyd-engine", "version": env!("CARGO_PKG_VERSION")}, "spans": spans}],
    }]})
    .to_string()
}

// ---- the exporter ---------------------------------------------------------------------------------------------------------------------------------

enum Msg {
    Spans(Vec<Span>),
    Flush(SyncSender<()>),
}

/// Sends spans to a collector from a thread of its own.
pub struct Exporter {
    tx: SyncSender<Msg>,
    pub exported: Arc<AtomicU64>,
    pub dropped: Arc<AtomicU64>,
}

/// Where and how to send: the traces URL, the headers, the service's name.
#[derive(Clone, Debug, PartialEq)]
pub struct Config {
    pub url: String,
    pub headers: Vec<(String, String)>,
    pub service: String,
}

/// The configuration the environment gives (`get` reads a variable), or None where tracing is off.
pub fn config_from(get: &dyn Fn(&str) -> Option<String>) -> Option<Config> {
    let var = |k: &str| get(k).map(|v| v.trim().to_string()).filter(|v| !v.is_empty());
    if var("OTEL_SDK_DISABLED").is_some_and(|v| v.eq_ignore_ascii_case("true")) {
        return None;
    }
    let url = match (var("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"), var("OTEL_EXPORTER_OTLP_ENDPOINT")) {
        (Some(u), _) => u,
        (None, Some(b)) => format!("{}/v1/traces", b.trim_end_matches('/')),
        (None, None) => return None,
    };
    let headers = var("OTEL_EXPORTER_OTLP_TRACES_HEADERS")
        .or_else(|| var("OTEL_EXPORTER_OTLP_HEADERS"))
        .map(|h| h.split(',').filter_map(|kv| kv.split_once('=').map(|(k, v)| (k.trim().to_string(), v.trim().to_string()))).collect())
        .unwrap_or_default();
    Some(Config { url, headers, service: var("OTEL_SERVICE_NAME").unwrap_or_else(|| "glyd".to_string()) })
}

/// A URL as it may be said in a log: scheme, host and port only (the authority up to its last `@`, so no user name or password; no path, query or fragment, any of which may hold a collector's key).
pub fn shown(url: &str) -> String {
    let (scheme, rest) = url.split_once("://").unwrap_or(("", url));
    let authority = rest.split(['/', '?', '#']).next().unwrap_or("");
    let host = authority.rsplit_once('@').map_or(authority, |(_, h)| h);
    if scheme.is_empty() {
        host.to_string()
    } else {
        format!("{scheme}://{host}")
    }
}

/// Why an export failed, in words that carry nothing of the request (the error's own text may name the URL).
fn why(e: &ureq::Error) -> String {
    match e {
        ureq::Error::StatusCode(c) => format!("the collector answered {c}"),
        ureq::Error::Timeout(_) => "timed out".to_string(),
        ureq::Error::HostNotFound => "host not found".to_string(),
        _ => "connection failed".to_string(),
    }
}

impl Exporter {
    pub fn start(cfg: Config) -> Exporter {
        let (tx, rx) = sync_channel::<Msg>(4096);
        let (exported, dropped) = (Arc::new(AtomicU64::new(0)), Arc::new(AtomicU64::new(0)));
        let (e, d) = (exported.clone(), dropped.clone());
        std::thread::Builder::new().name("glyd-otlp".into()).spawn(move || run(cfg, rx, e, d)).expect("a thread");
        Exporter { tx, exported, dropped }
    }

    /// Queues spans for sending: dropped (and counted) where the queue is full, never waited for.
    pub fn send(&self, spans: Vec<Span>) {
        let n = spans.len() as u64;
        if self.tx.try_send(Msg::Spans(spans)).is_err() {
            self.dropped.fetch_add(n, Relaxed);
        }
    }

    /// Waits up to `bound` for what is queued to be sent (at a drain, before the process ends); whether it was.
    pub fn flush(&self, bound: Duration) -> bool {
        let (ack, done) = sync_channel(1);
        self.tx.send_timeout_compat(Msg::Flush(ack), bound) && done.recv_timeout(bound).is_ok()
    }
}

trait SendTimeout {
    fn send_timeout_compat(&self, m: Msg, bound: Duration) -> bool;
}

impl SendTimeout for SyncSender<Msg> {
    fn send_timeout_compat(&self, mut m: Msg, bound: Duration) -> bool {
        let end = Instant::now() + bound;
        loop {
            match self.try_send(m) {
                Ok(()) => return true,
                Err(std::sync::mpsc::TrySendError::Full(back)) if Instant::now() < end => {
                    m = back;
                    std::thread::sleep(Duration::from_millis(10));
                }
                Err(_) => return false,
            }
        }
    }
}

const BATCH: usize = 100;
const EVERY: Duration = Duration::from_secs(1);

fn run(cfg: Config, rx: Receiver<Msg>, exported: Arc<AtomicU64>, dropped: Arc<AtomicU64>) {
    let agent = ureq::Agent::new_with_config(ureq::Agent::config_builder().timeout_global(Some(Duration::from_secs(5))).build());
    let mut batch: Vec<Span> = Vec::new();
    let (mut last_said, mut deadline) = (None::<Instant>, Instant::now() + EVERY);
    let mut send = |batch: &mut Vec<Span>| {
        if batch.is_empty() {
            return;
        }
        let body = otlp_json(&cfg.service, batch);
        let mut req = agent.post(&cfg.url).header("Content-Type", "application/json");
        for (k, v) in &cfg.headers {
            req = req.header(k.as_str(), v.as_str());
        }
        match req.send(body.as_str()) {
            Ok(_) => {
                exported.fetch_add(batch.len() as u64, Relaxed);
            }
            Err(e) => {
                dropped.fetch_add(batch.len() as u64, Relaxed);
                if last_said.is_none_or(|t| t.elapsed() > Duration::from_secs(30)) {
                    last_said = Some(Instant::now());
                    eprintln!("glyd: traces: could not send {} spans to {}: {}", batch.len(), shown(&cfg.url), why(&e));
                }
            }
        }
        batch.clear();
    };
    loop {
        match rx.recv_timeout(deadline.saturating_duration_since(Instant::now())) {
            Ok(Msg::Spans(s)) => {
                batch.extend(s);
                if batch.len() >= BATCH {
                    send(&mut batch);
                    deadline = Instant::now() + EVERY;
                }
            }
            Ok(Msg::Flush(ack)) => {
                send(&mut batch);
                let _ = ack.send(());
            }
            Err(RecvTimeoutError::Timeout) => {
                send(&mut batch);
                deadline = Instant::now() + EVERY;
            }
            Err(RecvTimeoutError::Disconnected) => {
                send(&mut batch);
                return;
            }
        }
    }
}

// ---- the process's exporter -------------------------------------------------------------------------------------------------------------------

static EXPORTER: Mutex<Option<Arc<Exporter>>> = Mutex::new(None);

fn exporter() -> Option<Arc<Exporter>> {
    EXPORTER.lock().unwrap_or_else(|e| e.into_inner()).clone()
}

/// Starts the exporter from the environment (once): where OpenTelemetry's variables name an endpoint. Said on stderr where it starts, and where the protocol asked for is not the one sent.
pub fn init_from_env() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let get = |k: &str| std::env::var(k).ok();
        let Some(cfg) = config_from(&get) else { return };
        if let Some(p) = get("OTEL_EXPORTER_OTLP_PROTOCOL").filter(|p| p.trim() != "http/json" && !p.trim().is_empty()) {
            eprintln!("glyd: traces: OTEL_EXPORTER_OTLP_PROTOCOL={p}: only http/json is sent (to {})", shown(&cfg.url));
        }
        eprintln!("glyd: traces: spans per request (queue, prefill, decode) go to {} as service {}", shown(&cfg.url), cfg.service);
        install(Exporter::start(cfg));
    });
}

/// The process's exporter (a test's own collector, or the environment's).
pub fn install(e: Exporter) {
    *EXPORTER.lock().unwrap_or_else(|e| e.into_inner()) = Some(Arc::new(e));
}

/// Whether traces are being sent.
pub fn on() -> bool {
    exporter().is_some()
}

/// (spans sent, spans dropped) so far, where traces are on.
pub fn counts() -> Option<(u64, u64)> {
    exporter().map(|e| (e.exported.load(Relaxed), e.dropped.load(Relaxed)))
}

/// A sequence of the served model `model` ended (`how`: stop, length, stopped or error) after `generated` tokens: its spans go to the collector, where there is one.
pub fn record(c: &Clock, how: &str, generated: usize, model: &str) {
    if let Some(e) = exporter() {
        e.send(spans_of(c, Instant::now(), how, generated, model));
    }
}

/// What is queued is sent (a drain, before the process ends), within `bound`.
pub fn flush(bound: Duration) {
    if let Some(e) = exporter() {
        e.flush(bound);
    }
}

#[cfg(test)]
pub(crate) mod testing {
    use std::io::{BufRead, BufReader, Read, Write};
    use std::net::TcpListener;
    use std::sync::{Arc, Mutex};

    /// A collector on this computer's own port: it takes `POST /v1/traces` and keeps the bodies (and the headers it saw), and answers 200 `{}`.
    pub struct Collector {
        pub url: String,
        pub bodies: Arc<Mutex<Vec<(String, Vec<String>)>>>,
    }

    pub fn collector() -> Collector {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}/v1/traces", l.local_addr().unwrap());
        let bodies = Arc::new(Mutex::new(Vec::new()));
        let kept = bodies.clone();
        std::thread::spawn(move || {
            for c in l.incoming() {
                let Ok(mut c) = c else { continue };
                let kept = kept.clone();
                std::thread::spawn(move || {
                    let mut r = BufReader::new(c.try_clone().unwrap());
                    let (mut len, mut head) = (0usize, Vec::new());
                    let mut line = String::new();
                    while r.read_line(&mut line).unwrap_or(0) > 0 && line != "\r\n" {
                        if let Some(v) = line.to_ascii_lowercase().strip_prefix("content-length:") {
                            len = v.trim().parse().unwrap_or(0);
                        }
                        head.push(line.trim().to_string());
                        line.clear();
                    }
                    let mut body = vec![0; len];
                    r.read_exact(&mut body).unwrap();
                    kept.lock().unwrap().push((String::from_utf8(body).unwrap(), head));
                    let _ = c.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{}");
                });
            }
        });
        Collector { url, bodies }
    }
}

#[cfg(test)]
mod tests {
    use super::testing::collector;
    use super::*;

    fn at(c: &Clock, ms: u64) -> Instant {
        c.began + Duration::from_millis(ms)
    }

    #[test]
    fn a_sequence_makes_a_request_span_with_queue_prefill_and_decode_beneath() {
        let mut c = Clock::new(12);
        c.admitted = Some(at(&c, 30));
        c.first = Some(at(&c, 80));
        let spans = spans_of(&c, at(&c, 200), "stop", 7, "m");
        let names: Vec<_> = spans.iter().map(|s| s.name).collect();
        assert_eq!(names, ["glyd.request", "glyd.queue", "glyd.prefill", "glyd.decode"]);
        let root = &spans[0];
        assert!(spans[1..].iter().all(|s| s.parent == Some(root.id) && s.trace == root.trace));
        let ms = |s: &Span| ((s.end_ns - s.start_ns) as f64 / 1e6).round() as u64;
        assert_eq!((ms(root), ms(&spans[1]), ms(&spans[2]), ms(&spans[3])), (200, 30, 50, 120));
        // the parts are in order and inside the whole
        assert_eq!((spans[1].end_ns, spans[2].end_ns), (spans[2].start_ns, spans[3].start_ns));
        assert!(root.attrs.contains(&("glyd.prompt_tokens", Attr::Int(12))) && root.attrs.contains(&("glyd.finish_reason", Attr::Str("stop".into()))));
        // no queue clock (an engine with no scheduler): no queue span; no token: no decode, and the pass runs to the end; an error is a span in error
        let c = Clock::new(3);
        let s = spans_of(&c, at(&c, 40), "error", 0, "m");
        assert_eq!(s.iter().map(|s| s.name).collect::<Vec<_>>(), ["glyd.request", "glyd.prefill"]);
        assert!(s[0].error && !s[1].error && (s[1].end_ns - s[1].start_ns) / 1_000_000 == 40);
    }

    #[test]
    fn the_otlp_json_has_the_shape_a_collector_reads() {
        let mut c = Clock::new(5);
        c.admitted = Some(at(&c, 1));
        c.first = Some(at(&c, 2));
        let body: Value = serde_json::from_str(&otlp_json("glyd-test", &spans_of(&c, at(&c, 3), "length", 2, "Qwen/Qwen3-8B"))).unwrap();
        let rs = &body["resourceSpans"][0];
        assert_eq!(rs["resource"]["attributes"][0], json!({"key": "service.name", "value": {"stringValue": "glyd-test"}}));
        let spans = rs["scopeSpans"][0]["spans"].as_array().unwrap();
        assert_eq!(spans.len(), 4);
        let root = &spans[0];
        assert_eq!(root["traceId"].as_str().unwrap().len(), 32);
        assert_eq!(root["spanId"].as_str().unwrap().len(), 16);
        assert!(root.get("parentSpanId").is_none() && root["kind"] == 2 && root["status"]["code"] == 1);
        assert!(root["startTimeUnixNano"].as_str().unwrap().parse::<u64>().unwrap() > 1_700_000_000_000_000_000);
        assert_eq!(root["attributes"][1], json!({"key": "glyd.prompt_tokens", "value": {"intValue": "5"}}));
        for child in &spans[1..] {
            assert_eq!(child["parentSpanId"], root["spanId"]);
            assert_eq!(child["traceId"], root["traceId"]);
        }
        // ids are different for every span and every trace
        let ids: std::collections::HashSet<_> = spans.iter().map(|s| s["spanId"].as_str().unwrap()).collect();
        assert_eq!(ids.len(), 4);
    }

    /// What is said of the collector's URL carries no secret: no user name or password, no query, no fragment.
    #[test]
    fn a_collectors_url_is_said_without_its_secrets() {
        assert_eq!(shown("https://user:pass@collector.example:4318/v1/traces?api_key=abc#frag"), "https://collector.example:4318");
        assert_eq!(shown("http://127.0.0.1:4318/v1/traces"), "http://127.0.0.1:4318");
        assert_eq!(shown("http://token@host"), "http://host");
        assert_eq!(shown("http://host?k=v"), "http://host");
        assert_eq!(shown("http://a@b@host:1/p"), "http://host:1");
        assert_eq!(shown("http://host/p@secret"), "http://host");
        assert_eq!(shown("http://host#u@secret"), "http://host");
    }

    #[test]
    fn tracing_is_off_unless_an_endpoint_is_configured() {
        let env = |pairs: &'static [(&'static str, &'static str)]| move |k: &str| pairs.iter().find(|(n, _)| *n == k).map(|(_, v)| v.to_string());
        assert_eq!(config_from(&env(&[])), None);
        assert_eq!(config_from(&env(&[("OTEL_SERVICE_NAME", "x")])), None);
        assert_eq!(config_from(&env(&[("OTEL_EXPORTER_OTLP_ENDPOINT", "  ")])), None);
        assert_eq!(config_from(&env(&[("OTEL_EXPORTER_OTLP_ENDPOINT", "http://c:4318"), ("OTEL_SDK_DISABLED", "true")])), None);
        let c = config_from(&env(&[
            ("OTEL_EXPORTER_OTLP_ENDPOINT", "http://collector:4318/"),
            ("OTEL_EXPORTER_OTLP_HEADERS", "api-key=abc, x=y"),
            ("OTEL_SERVICE_NAME", "glyd-prod"),
        ]))
        .unwrap();
        assert_eq!(c.url, "http://collector:4318/v1/traces");
        assert_eq!(c.headers, [("api-key".to_string(), "abc".to_string()), ("x".to_string(), "y".to_string())]);
        assert_eq!(c.service, "glyd-prod");
        // the traces endpoint is the whole URL and wins
        let c = config_from(&env(&[("OTEL_EXPORTER_OTLP_ENDPOINT", "http://a:1"), ("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "http://b:2/custom")])).unwrap();
        assert_eq!((c.url.as_str(), c.service.as_str()), ("http://b:2/custom", "glyd"));
    }

    /// Spans reach a collector: batched, with the headers configured, and counted; a flush waits for them; a collector that is not there drops them and counts, and the sender is not held up.
    #[test]
    fn spans_reach_a_local_collector_and_a_dead_one_costs_nothing() {
        let col = collector();
        let ex = Exporter::start(Config { url: col.url.clone(), headers: vec![("x-key".into(), "k1".into())], service: "glyd-test".into() });
        let mut c = Clock::new(9);
        c.admit();
        c.token();
        for _ in 0..3 {
            ex.send(spans_of(&c, Instant::now(), "stop", 4, "m"));
        }
        assert!(ex.flush(Duration::from_secs(5)));
        assert_eq!(ex.exported.load(Relaxed), 12);
        let got = col.bodies.lock().unwrap().clone();
        let n: usize =
            got.iter().map(|(b, _)| serde_json::from_str::<Value>(b).unwrap()["resourceSpans"][0]["scopeSpans"][0]["spans"].as_array().unwrap().len()).sum();
        assert_eq!(n, 12);
        assert!(
            got[0].1.iter().any(|h| h.to_lowercase().starts_with("post /v1/traces")) && got[0].1.iter().any(|h| h.to_lowercase() == "x-key: k1"),
            "{:?}",
            got[0].1
        );
        assert!(got[0].1.iter().any(|h| h.to_lowercase() == "content-type: application/json"));
        // a collector that is not there
        let dead = Exporter::start(Config { url: "http://127.0.0.1:1/v1/traces".into(), headers: vec![], service: "x".into() });
        let t = Instant::now();
        for _ in 0..50 {
            dead.send(spans_of(&c, Instant::now(), "stop", 1, "m"));
        }
        assert!(t.elapsed() < Duration::from_millis(500), "a dead collector held the sender up for {:?}", t.elapsed());
        assert!(dead.flush(Duration::from_secs(10)));
        assert_eq!((dead.exported.load(Relaxed), dead.dropped.load(Relaxed)), (0, 200));
    }
}
