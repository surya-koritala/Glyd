//! What an operator needs of a running server: Prometheus metrics (`GET /metrics`), what the server is doing for a load balancer (`/health` alive, `/ready` able to serve) and a graceful drain on SIGTERM (stop taking
//! requests, let the ones in flight finish, exit within a bound). The figures are atomics the server's threads add to as they go (no lock on a request's path), written out in the text format of Prometheus
//! (version 0.0.4) when a scrape comes. Every server path (`glyd serve`, the lossy levels', the dev servers, `glyd service`) answers the endpoints; where an engine has no scheduler (a lossy level's) the figures of the scheduler
//! are left out, not made up.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering::Relaxed};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

// ---- histograms -----------------------------------------------------------------------------------------------------------------------------------

/// Buckets (seconds) of the time to a request's first token, between two tokens of a reply, and a whole request: what a chat user waits, what a reader sees as speed, what a gateway's timeout is set against.
const FIRST_TOKEN: &[f64] = &[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0];
const BETWEEN_TOKENS: &[f64] = &[0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 5.0];
const WHOLE: &[f64] = &[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0];

/// A histogram of seconds: the count of observations in each bucket (not cumulative; `render` adds them up), their number and their sum.
pub struct Histogram {
    bounds: &'static [f64],
    /// One count for each bound and one past the last.
    counts: [AtomicU64; 16],
    sum_ns: AtomicU64,
}

impl Histogram {
    pub const fn new(bounds: &'static [f64]) -> Histogram {
        Histogram { bounds, counts: [const { AtomicU64::new(0) }; 16], sum_ns: AtomicU64::new(0) }
    }

    pub fn observe(&self, secs: f64) {
        let i = self.bounds.iter().position(|&b| secs <= b).unwrap_or(self.bounds.len());
        self.counts[i].fetch_add(1, Relaxed);
        self.sum_ns.fetch_add((secs.max(0.0) * 1e9) as u64, Relaxed);
    }

    pub fn count(&self) -> u64 {
        self.counts.iter().map(|c| c.load(Relaxed)).sum()
    }

    fn render(&self, out: &mut String, name: &str, help: &str) {
        let _ = writeln!(out, "# HELP {name} {help}\n# TYPE {name} histogram");
        let mut cum = 0;
        for (i, b) in self.bounds.iter().enumerate() {
            cum += self.counts[i].load(Relaxed);
            let _ = writeln!(out, "{name}_bucket{{le=\"{b}\"}} {cum}");
        }
        cum += self.counts[self.bounds.len()].load(Relaxed);
        let _ = writeln!(out, "{name}_bucket{{le=\"+Inf\"}} {cum}\n{name}_sum {}\n{name}_count {cum}", self.sum_ns.load(Relaxed) as f64 / 1e9);
    }
}

// ---- counters and gauges --------------------------------------------------------------------------------------------------------------------------

const ENDS: [&str; 4] = ["stop", "length", "stopped", "error"];
const ROUTES: [&str; 6] = ["chat_completions", "completions", "messages", "responses", "ollama", "other"];

/// What is counted of one served model's requests and sequences (a runtime context's: `runtime::context::Context::figures`).
pub struct Figures {
    pub ttft: Histogram,
    pub token_gap: Histogram,
    pub request_time: Histogram,
    /// Tokens written to replies, sequences ended by how they ended (stop, length, stopped by the client or a cancel, error).
    tokens_out: AtomicU64,
    ended: [AtomicU64; 4],
    /// Requests taken, by the API they are for, and requests refused before any work, by status.
    requests: [AtomicU64; 6],
    refused: Mutex<BTreeMap<u16, u64>>,
    /// The cache's pages: all, and those a running sequence holds (set by the scheduler each round; 0 where the engine has no paged cache).
    pub kv_pages: AtomicU64,
    pub kv_pages_held: AtomicU64,
    /// The prompt tokens of the requests an engine with no scheduler ran (the scheduler counts its own, the context's `sched`).
    prompt_tokens: AtomicU64,
    /// The served model's name, which its sequences' spans carry (`otel::record`): set where its server starts.
    pub model: std::sync::OnceLock<String>,
}

impl Default for Figures {
    fn default() -> Figures {
        Figures::new()
    }
}

impl Figures {
    pub const fn new() -> Figures {
        Figures {
            ttft: Histogram::new(FIRST_TOKEN),
            token_gap: Histogram::new(BETWEEN_TOKENS),
            request_time: Histogram::new(WHOLE),
            tokens_out: AtomicU64::new(0),
            ended: [const { AtomicU64::new(0) }; 4],
            requests: [const { AtomicU64::new(0) }; 6],
            refused: Mutex::new(BTreeMap::new()),
            kv_pages: AtomicU64::new(0),
            kv_pages_held: AtomicU64::new(0),
            prompt_tokens: AtomicU64::new(0),
            model: std::sync::OnceLock::new(),
        }
    }

    /// A request for one of the APIs is taken (`path`: its path); the label of its API.
    pub fn request(&self, path: &str) {
        let i = match path {
            "/v1/chat/completions" => 0,
            "/v1/completions" => 1,
            p if p.starts_with("/v1/messages") => 2,
            "/v1/responses" => 3,
            p if p.starts_with("/api/") => 4,
            _ => 5,
        };
        self.requests[i].fetch_add(1, Relaxed);
    }

    /// A request is refused before any work is done for it (the key, the Host, the Origin, a limit, a drain): by its status.
    pub fn refused(&self, code: u16) {
        *self.refused.lock().unwrap_or_else(|e| e.into_inner()).entry(code).or_insert(0) += 1;
    }

    /// A token of a reply written at `now`: the first of a sequence (its `last` is None) is the time since `began`, a later one the time since the last.
    pub fn token(&self, began: Instant, last: &mut Option<Instant>) {
        let now = Instant::now();
        match last {
            None => self.ttft.observe(now.duration_since(began).as_secs_f64()),
            Some(t) => self.token_gap.observe(now.duration_since(*t).as_secs_f64()),
        }
        *last = Some(now);
        self.tokens_out.fetch_add(1, Relaxed);
    }

    /// A sequence ended (`how`: stop, length, stopped or error) `began` ago.
    pub fn ended(&self, how: &str, began: Instant) {
        if let Some(i) = ENDS.iter().position(|e| *e == how) {
            self.ended[i].fetch_add(1, Relaxed);
        }
        self.request_time.observe(began.elapsed().as_secs_f64());
    }

    /// A sequence ended: counted (`ended`) and, where traces are on, its spans sent (`otel::record`).
    pub fn finished(&self, how: &str, began: Instant, clock: &crate::otel::Clock, generated: usize) {
        self.ended(how, began);
        crate::otel::record(clock, how, generated, self.model.get().map_or("", String::as_str));
    }
}

/// The process's own figures (`Context::process`): what the functions below count in, for the callers that name no context.
pub fn figures() -> &'static Figures {
    &crate::runtime::context::Context::process().figures
}

fn start() -> Instant {
    static START: OnceLock<Instant> = OnceLock::new();
    *START.get_or_init(Instant::now)
}

/// Remembers when the process began (the first call is the start of `glyd_uptime_seconds`).
pub fn init() {
    let _ = start();
    crate::otel::init_from_env();
}

/// `Figures::request` of the process's.
pub fn request(path: &str) {
    figures().request(path)
}

/// `Figures::refused` of the process's.
pub fn refused(code: u16) {
    figures().refused(code)
}

/// `Figures::token` of the process's.
pub fn token(began: Instant, last: &mut Option<Instant>) {
    figures().token(began, last)
}

/// `Figures::ended` of the process's.
pub fn ended(how: &str, began: Instant) {
    figures().ended(how, began)
}

/// `Figures::finished` of the process's.
pub fn finished(how: &str, began: Instant, clock: &crate::otel::Clock, generated: usize) {
    figures().finished(how, began, clock, generated)
}

/// One request of an engine that has no scheduler (a lossy level's): its tokens, first and gaps, and its end, as the scheduler's sequences have them.
pub struct Timer {
    began: Instant,
    first: Option<Instant>,
    last: Option<Instant>,
    clock: crate::otel::Clock,
    made: usize,
    prompt: usize,
}

impl Timer {
    /// A request of `prompt` tokens begins.
    pub fn start(prompt: usize) -> Timer {
        figures().prompt_tokens.fetch_add(prompt as u64, Relaxed);
        Timer { began: Instant::now(), first: None, last: None, clock: crate::otel::Clock::new(prompt), made: 0, prompt }
    }
    pub fn token(&mut self) {
        token(self.began, &mut self.last);
        self.first.get_or_insert_with(Instant::now);
        self.clock.token();
        self.made += 1;
    }
    /// The request ended: its figures, and its speed kept with the plan that loaded (src/plan.rs `observe`; a request run alone, the only kind a `Timer` times).
    pub fn end(self, how: &str) {
        finished(how, self.began, &self.clock, self.made);
        if let (Some(first), Some(last)) = (self.first, self.last) {
            crate::plan::observe(self.prompt, (first - self.began).as_secs_f64(), self.made, (last - first).as_secs_f64());
        }
    }
}

// ---- the GPU's memory ----------------------------------------------------------------------------------------------------------------------------

/// The GPUs' memory, (index, used, total) in bytes, as `nvidia-smi` says it: asked at most every 5 seconds, where there is one (empty where there is not, or it does not answer).
fn gpu_memory() -> Vec<(u32, u64, u64)> {
    static CACHE: Mutex<Option<(Instant, Vec<(u32, u64, u64)>)>> = Mutex::new(None);
    let mut g = CACHE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some((t, v)) = g.as_ref() {
        if t.elapsed() < Duration::from_secs(5) {
            return v.clone();
        }
    }
    let v = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=index,memory.used,memory.total", "--format=csv,noheader,nounits"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| parse_smi(&String::from_utf8_lossy(&o.stdout)))
        .unwrap_or_default();
    *g = Some((Instant::now(), v.clone()));
    v
}

/// `index, used MiB, total MiB` lines as (index, used bytes, total bytes).
pub fn parse_smi(text: &str) -> Vec<(u32, u64, u64)> {
    text.lines()
        .filter_map(|l| {
            let mut f = l.split(',').map(str::trim);
            let (i, u, t) = (f.next()?.parse().ok()?, f.next()?.parse::<u64>().ok()?, f.next()?.parse::<u64>().ok()?);
            Some((i, u << 20, t << 20))
        })
        .collect()
}

// ---- readiness and drain -------------------------------------------------------------------------------------------------------------------------

/// What a listener knows of its own life: whether it is draining (it takes no new request) and how many requests are in flight (taken, not yet answered to their end).
#[derive(Default)]
pub struct Life {
    draining: AtomicBool,
    inflight: AtomicUsize,
}

/// One request in flight: it counts for as long as this is held (it goes with the request to the thread that answers it).
pub struct Inflight(Arc<Life>);

impl Drop for Inflight {
    fn drop(&mut self) {
        self.0.inflight.fetch_sub(1, Relaxed);
    }
}

impl Life {
    /// The process's own: what SIGTERM drains.
    pub fn global() -> Arc<Life> {
        static G: OnceLock<Arc<Life>> = OnceLock::new();
        G.get_or_init(Arc::default).clone()
    }

    /// A request is taken: its place in the count, or None where the server is draining (it takes no new one).
    pub fn enter(self: &Arc<Life>) -> Option<Inflight> {
        self.inflight.fetch_add(1, Relaxed);
        if self.draining.load(Relaxed) {
            self.inflight.fetch_sub(1, Relaxed);
            return None;
        }
        Some(Inflight(self.clone()))
    }

    pub fn draining(&self) -> bool {
        self.draining.load(Relaxed)
    }

    pub fn inflight(&self) -> usize {
        self.inflight.load(Relaxed)
    }

    /// Stops taking requests: `/ready` says 503 and every new request for an API is answered 503.
    pub fn begin_drain(&self) {
        self.draining.store(true, Relaxed);
    }

    /// Waits for the requests in flight to end, up to `bound`: how many are left (0: all ended).
    pub fn wait_idle(&self, bound: Duration) -> usize {
        let end = Instant::now() + bound;
        while self.inflight() > 0 && Instant::now() < end {
            std::thread::sleep(Duration::from_millis(20));
        }
        self.inflight()
    }

    /// `begin_drain`, then `wait_idle`: the number of requests that did not end within `bound`.
    pub fn drain(&self, bound: Duration) -> usize {
        self.begin_drain();
        self.wait_idle(bound)
    }
}

/// The time a drain waits for the requests in flight by default (`--drain-timeout SECS` says another); a request that is still running then is ended with the process.
pub const DRAIN_SECONDS: u64 = 30;

static SIGNALS: AtomicUsize = AtomicUsize::new(0);

mod sig {
    use super::SIGNALS;
    use std::sync::atomic::Ordering::Relaxed;
    extern "C" {
        fn signal(sig: i32, handler: usize) -> usize;
        fn _exit(code: i32) -> !;
    }
    extern "C" fn on_term(_: i32) {
        // the first SIGTERM starts the drain (the watcher thread sees the count); a second leaves at once
        if SIGNALS.fetch_add(1, Relaxed) >= 1 {
            // SAFETY: _exit is async-signal-safe.
            unsafe { _exit(143) }
        }
    }
    pub fn install() {
        // SAFETY: SIGTERM's handler does nothing but an atomic add and _exit.
        unsafe {
            signal(15, on_term as *const () as usize);
        }
    }
}

/// SIGTERM drains the process: a thread waits for it, stops the listeners taking requests (`life`), waits up to `bound` for the requests in flight, says what it did and ends the process with `then` (0 where every request
/// ended, 1 where some were still running at the bound). `then` is where the caller closes what it holds first (a service stops its models) before the process ends.
pub fn drain_on_sigterm(life: Arc<Life>, bound: Duration, then: impl Fn(usize) + Send + 'static) {
    // (once for the process: the first listener's is the one that counts)
    static ONCE: std::sync::Once = std::sync::Once::new();
    let mut first = false;
    ONCE.call_once(|| first = true);
    if !first {
        return;
    }
    sig::install();
    std::thread::spawn(move || {
        while SIGNALS.load(Relaxed) == 0 {
            std::thread::sleep(Duration::from_millis(50));
        }
        life.begin_drain();
        eprintln!("glyd: SIGTERM: not taking requests; {} in flight, waited for up to {}s", life.inflight(), bound.as_secs());
        let left = life.wait_idle(bound);
        if left == 0 {
            eprintln!("glyd: drained: every request ended");
        } else {
            eprintln!("glyd: drain bound reached: {left} request(s) still running are ended with the process");
        }
        then(left);
        crate::otel::flush(Duration::from_secs(5));
        if left == 0 {
            // (the last request's answer is written; the connection closes on its own before the process does)
            std::thread::sleep(Duration::from_millis(200));
        }
        std::process::exit(if left == 0 { 0 } else { 1 });
    });
}

// ---- the text of a scrape ------------------------------------------------------------------------------------------------------------------------

/// What a server's own state adds to a scrape: its model's name, and whether it is ready and draining.
pub struct State<'a> {
    pub model: &'a str,
    pub ready: bool,
    pub life: &'a Life,
    /// Whether this process runs a model (an engine's server) and so has tokens, latencies, a queue and a cache to tell of; the service that supervises them has not.
    pub engine: bool,
}

fn family(out: &mut String, name: &str, kind: &str, help: &str, lines: &[(String, String)]) {
    let _ = writeln!(out, "# HELP {name} {help}\n# TYPE {name} {kind}");
    for (labels, v) in lines {
        let _ = writeln!(out, "{name}{labels} {v}");
    }
}

fn one(v: impl std::fmt::Display) -> Vec<(String, String)> {
    vec![(String::new(), v.to_string())]
}

/// `render_of` the process's context.
pub fn render(s: &State) -> String {
    render_of(crate::runtime::context::Context::process(), s)
}

/// The metrics of the model of `rt` in Prometheus's text format (0.0.4): requests, tokens, the histograms of the first token and of the time between tokens, the scheduler's queue and what runs, the cache's
/// pages, the GPUs' memory.
pub fn render_of(rt: &crate::runtime::context::Context, s: &State) -> String {
    let f = &rt.figures;
    let mut o = String::with_capacity(4096);
    family(
        &mut o,
        "glyd_build_info",
        "gauge",
        "The version of this program and the model it serves.",
        &[(format!("{{version=\"{}\",model=\"{}\"}}", env!("CARGO_PKG_VERSION"), s.model.replace('\\', "\\\\").replace('"', "\\\"")), "1".into())],
    );
    family(&mut o, "glyd_uptime_seconds", "gauge", "Seconds since the server started.", &one(start().elapsed().as_secs_f64()));
    family(
        &mut o,
        "glyd_ready",
        "gauge",
        "1 where the model is loaded and the server takes requests (what /ready says), else 0.",
        &one(u8::from(s.ready && !s.life.draining())),
    );
    family(
        &mut o,
        "glyd_draining",
        "gauge",
        "1 while the server drains (SIGTERM): it takes no new request and lets those in flight end.",
        &one(u8::from(s.life.draining())),
    );
    family(&mut o, "glyd_requests_in_flight", "gauge", "Requests taken and not yet answered to their end.", &one(s.life.inflight()));
    let reqs: Vec<_> = ROUTES.iter().zip(&f.requests).map(|(r, c)| (format!("{{api=\"{r}\"}}"), c.load(Relaxed).to_string())).collect();
    family(&mut o, "glyd_requests_total", "counter", "Requests taken for an API (chat_completions, completions, messages, responses, ollama).", &reqs);
    let refused: Vec<_> = f.refused.lock().unwrap_or_else(|e| e.into_inner()).iter().map(|(c, n)| (format!("{{code=\"{c}\"}}"), n.to_string())).collect();
    family(&mut o, "glyd_requests_refused_total", "counter", "Requests refused before any work (key, Host, Origin, size, time, drain), by status.", &refused);
    let ends: Vec<_> = ENDS.iter().zip(&f.ended).map(|(r, c)| (format!("{{reason=\"{r}\"}}"), c.load(Relaxed).to_string())).collect();
    family(
        &mut o,
        "glyd_sequences_finished_total",
        "counter",
        "Sequences ended, by how: stop, length, stopped (the client left or cancelled) or error.",
        &ends,
    );
    if !s.engine {
        return o;
    }
    let v: Vec<u64> = rt.sched.iter().take(7).map(|a| a.load(Relaxed)).collect();
    family(&mut o, "glyd_prompt_tokens_total", "counter", "Prompt tokens run or found in the prefix cache.", &one(v[0] + f.prompt_tokens.load(Relaxed)));
    // (told only where the server tells its requests what they found: `Context::tell_cached`, as /stats)
    if rt.tell_cached.load(Relaxed) {
        family(&mut o, "glyd_prefix_cache_hit_tokens_total", "counter", "Prompt tokens found in the prefix cache.", &one(v[1]));
    }
    family(&mut o, "glyd_generated_tokens_total", "counter", "Tokens written to replies.", &one(f.tokens_out.load(Relaxed)));
    f.ttft.render(
        &mut o,
        "glyd_time_to_first_token_seconds",
        "Seconds from a sequence being sent to the scheduler to its first token (the queue, the prompt and the first step).",
    );
    f.token_gap.render(&mut o, "glyd_time_per_output_token_seconds", "Seconds between two tokens of one reply.");
    f.request_time.render(&mut o, "glyd_request_duration_seconds", "Seconds from a sequence being sent to the scheduler to its end.");
    family(&mut o, "glyd_queue_depth", "gauge", "Sequences waiting for room in the scheduler.", &one(v[6]));
    family(&mut o, "glyd_sequences_running", "gauge", "Sequences the scheduler is running.", &one(v[5]));
    let pages = f.kv_pages.load(Relaxed);
    if pages > 0 {
        let held = f.kv_pages_held.load(Relaxed);
        family(&mut o, "glyd_kv_cache_pages", "gauge", "Pages of the KV cache.", &one(pages));
        family(&mut o, "glyd_kv_cache_pages_used", "gauge", "Pages of the KV cache that running sequences hold.", &one(held));
        family(&mut o, "glyd_kv_cache_usage_ratio", "gauge", "Used pages of the KV cache over all of them (0 to 1).", &one(held as f64 / pages as f64));
    }
    if let Some((sent, dropped)) = crate::otel::counts() {
        family(&mut o, "glyd_trace_spans_exported_total", "counter", "Spans sent to the OTLP collector.", &one(sent));
        family(&mut o, "glyd_trace_spans_dropped_total", "counter", "Spans not sent: the collector did not take them, or the queue was full.", &one(dropped));
    }
    let gpus = gpu_memory();
    if !gpus.is_empty() {
        let used: Vec<_> = gpus.iter().map(|(i, u, _)| (format!("{{gpu=\"{i}\"}}"), u.to_string())).collect();
        let total: Vec<_> = gpus.iter().map(|(i, _, t)| (format!("{{gpu=\"{i}\"}}"), t.to_string())).collect();
        family(&mut o, "glyd_gpu_memory_used_bytes", "gauge", "GPU memory in use on the computer (nvidia-smi), by GPU.", &used);
        family(&mut o, "glyd_gpu_memory_total_bytes", "gauge", "GPU memory, by GPU.", &total);
    }
    o
}

/// The Content-Type of a scrape.
pub const CONTENT_TYPE: &str = "text/plain; version=0.0.4; charset=utf-8";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_histogram_is_cumulative_and_has_its_sum_and_count() {
        static H: Histogram = Histogram::new(&[0.1, 1.0]);
        for s in [0.05, 0.1, 0.5, 2.0, 30.0] {
            H.observe(s);
        }
        let mut o = String::new();
        H.render(&mut o, "x_seconds", "help");
        assert!(o.starts_with("# HELP x_seconds help\n# TYPE x_seconds histogram\n"), "{o}");
        for want in ["x_seconds_bucket{le=\"0.1\"} 2", "x_seconds_bucket{le=\"1\"} 3", "x_seconds_bucket{le=\"+Inf\"} 5", "x_seconds_count 5"] {
            assert!(o.contains(want), "{want} in {o}");
        }
        let sum: f64 = o.lines().find_map(|l| l.strip_prefix("x_seconds_sum ")).unwrap().parse().unwrap();
        assert!((sum - 32.65).abs() < 1e-6, "{sum}");
        assert_eq!(H.count(), 5);
    }

    /// What `nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits` prints (MiB).
    #[test]
    fn nvidia_smi_lines_are_read_as_bytes() {
        assert_eq!(parse_smi("0, 1024, 16376\n1, 0, 24564\n"), [(0, 1024 << 20, 16376 << 20), (1, 0, 24564 << 20)]);
        assert!(parse_smi("No devices were found\n").is_empty());
        assert!(parse_smi("").is_empty());
    }

    /// A drain takes no new request, lets the ones in flight end, and says how many did not within its bound.
    #[test]
    fn a_drain_refuses_new_requests_and_waits_for_the_ones_in_flight() {
        let life = Arc::new(Life::default());
        let a = life.enter().expect("a request is taken");
        let b = life.enter().expect("another");
        assert_eq!((life.inflight(), life.draining()), (2, false));
        life.begin_drain();
        assert!(life.enter().is_none(), "a request was taken while draining");
        assert_eq!(life.inflight(), 2, "a refused request is not counted");
        // the bound: two are still in flight
        assert_eq!(life.wait_idle(Duration::from_millis(100)), 2);
        drop(a);
        let l2 = life.clone();
        let t = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(150));
            drop(b);
            l2.inflight()
        });
        let t0 = Instant::now();
        assert_eq!(life.wait_idle(Duration::from_secs(10)), 0);
        assert!(t0.elapsed() < Duration::from_secs(5), "waited for the bound and not for the request");
        assert_eq!(t.join().unwrap(), 0);
        assert_eq!(life.drain(Duration::from_millis(10)), 0);
    }

    /// The scrape has every family the operator is promised, each with its HELP and TYPE, and its lines read as Prometheus's text format.
    #[test]
    fn a_scrape_has_the_promised_families_in_the_text_format() {
        let life = Life::default();
        request("/v1/chat/completions");
        refused(401);
        figures().ttft.observe(0.2);
        figures().token_gap.observe(0.01);
        figures().kv_pages.store(100, Relaxed);
        figures().kv_pages_held.store(25, Relaxed);
        let text = render(&State { model: "Qwen/Qwen3-8B", ready: true, life: &life, engine: true });
        for name in [
            "glyd_build_info",
            "glyd_ready",
            "glyd_draining",
            "glyd_requests_in_flight",
            "glyd_requests_total",
            "glyd_requests_refused_total",
            "glyd_sequences_finished_total",
            "glyd_prompt_tokens_total",
            "glyd_generated_tokens_total",
            "glyd_time_to_first_token_seconds",
            "glyd_time_per_output_token_seconds",
            "glyd_request_duration_seconds",
            "glyd_queue_depth",
            "glyd_sequences_running",
            "glyd_kv_cache_pages_used",
            "glyd_kv_cache_usage_ratio",
        ] {
            assert!(text.contains(&format!("# TYPE {name} ")) && text.contains(&format!("# HELP {name} ")), "{name} in {text}");
        }
        assert!(text.contains("glyd_requests_total{api=\"chat_completions\"} ") && text.contains("glyd_requests_refused_total{code=\"401\"} "), "{text}");
        assert!(text.contains("glyd_build_info{version=\"") && text.contains("model=\"Qwen/Qwen3-8B\""), "{text}");
        assert!(text.contains("glyd_ready 1\n") && text.contains("glyd_draining 0\n"), "{text}");
        life.begin_drain();
        let text = render(&State { model: "m", ready: true, life: &life, engine: true });
        assert!(text.contains("glyd_ready 0\n") && text.contains("glyd_draining 1\n"), "{text}");
        // every line is a comment or `name{labels} number`
        for l in text.lines().filter(|l| !l.starts_with('#')) {
            let (name, v) = l.rsplit_once(' ').unwrap_or_else(|| panic!("{l}"));
            assert!(v.parse::<f64>().is_ok(), "{l}");
            assert!(name.chars().next().is_some_and(|c| c.is_ascii_lowercase()) && !name.contains(' ') || name.contains('{'), "{l}");
        }
    }

    #[test]
    fn a_timer_counts_the_first_token_the_gaps_and_the_end() {
        let f = figures();
        let (t0, g0, r0) = (f.ttft.count(), f.token_gap.count(), f.request_time.count());
        let mut t = Timer::start(3);
        t.token();
        t.token();
        t.token();
        t.end("stop");
        // (the counts are the process's: other tests add to them, so at least ours)
        assert!(f.ttft.count() > t0 && f.token_gap.count() >= g0 + 2 && f.request_time.count() > r0);
    }
}
