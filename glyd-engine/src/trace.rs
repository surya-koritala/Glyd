//! A development build's timeline of the steps of a session: markers recorded on the engine's stream (a CUDA event each: the stream's own clock) with the host's clock beside them, for the steps `qwenserve` is asked to trace
//! (`{"set": {"trace": 1}}`, or 2 for every stage marker of the engine). The decode step of the expert cache is host work and GPU work in turn (a layer's router choices are read on the host before its experts run); this is
//! where the time of one step is split per layer into the parts: the GPU's always-on work, the readback of the router's choices and the wait for it, the host's plan and copies, the experts on the GPU, the experts on the host
//! and the sums sent back, the sum and what follows.

use crate::dev::Ev;
use crate::ffi::Stream;
use std::cell::{Cell, RefCell};
use std::time::Instant;

/// One marker: its label, the event recorded on the stream where it was queued, and the host's clock then.
pub struct Mark {
    pub label: &'static str,
    pub ev: Ev,
    pub at: Instant,
}

thread_local! {
    static MARKS: RefCell<Option<Vec<Mark>>> = const { RefCell::new(None) };
    static ALL: Cell<bool> = const { Cell::new(false) };
}

/// The labels of the markers a coarse trace (level 1) keeps; level 2 keeps every one.
const COARSE: [&str; 16] = [
    "mixer half start",
    "mHC update (mixer)",
    "feed-forward half start",
    "mHC mix (feed-forward)",
    "router",
    "shared expert",
    "store: choices requested",
    "store: choices read",
    "store: planned",
    "store: products queued",
    "store: host sums sent",
    "store: sum queued",
    "experts or MLP",
    "mHC update (feed-forward)",
    "segment: launched",
    "segment: graph ran",
];

/// Starts a trace: markers are kept from now on (all of them where `all`).
pub fn start(all: bool) {
    ALL.with(|a| a.set(all));
    MARKS.with(|m| *m.borrow_mut() = Some(Vec::new()));
}

/// The markers kept so far dropped, the trace going on (a measurement's own beginning).
pub fn restart() {
    MARKS.with(|m| {
        if let Some(v) = m.borrow_mut().as_mut() {
            v.clear();
        }
    });
}

/// Ends the trace and hands back its markers, in the order they were queued.
pub fn stop() -> Vec<Mark> {
    MARKS.with(|m| m.borrow_mut().take().unwrap_or_default())
}

/// Records a marker on `s` if a trace is on.
pub fn mark(s: Stream, label: &'static str) {
    MARKS.with(|m| {
        if let Some(v) = m.borrow_mut().as_mut() {
            if ALL.with(Cell::get) || COARSE.contains(&label) {
                if let Ok(ev) = Ev::new() {
                    if ev.record(s).is_ok() {
                        v.push(Mark { label, ev, at: Instant::now() });
                    }
                }
            }
        }
    });
}

/// One layer's parts of a step, in microseconds on the stream's clock unless said: from the marker "mixer half start" of the layer to the next layer's (or the last marker).
#[derive(Clone, Copy, Debug, Default)]
pub struct Layer {
    /// the mixer half, then the feed-forward's collapse and norm: the GPU's work that does not wait for the router
    pub always: f64,
    /// the router and the shared expert and what the experts' input needs (until the choices are asked for)
    pub router: f64,
    /// the choices' copy to the host and the host seeing them (the stream idle after its last kernel)
    pub readback: f64,
    /// the plan and the copies of the missed experts: the host's work, the stream waiting
    pub plan: f64,
    /// the experts' products on the GPU
    pub products: f64,
    /// the host's experts (the stream waits for their sums), and the sums' copy
    pub host_experts: f64,
    /// the sum of the pairs, R^T and the shared expert's addition
    pub sum: f64,
    /// the streams' update after the experts and the next layer's start, if any
    pub post: f64,
    /// the host's time from the layer's first marker to the choices being requested: what it takes to queue the GPU's always-on work
    pub host_queue: f64,
    /// the host's time from the choices being requested to their being read (the host blocked)
    pub host_wait: f64,
    /// the layer in all
    pub total: f64,
    /// whether the layer has experts in the store (else only `always` and `total` are set)
    pub store: bool,
}

/// The parts of every layer of a traced step from its markers (the stream must be idle: every event complete).
pub fn layers(marks: &[Mark]) -> Vec<Layer> {
    if marks.is_empty() {
        return Vec::new();
    }
    let t0 = &marks[0].ev;
    let at: Vec<f64> = marks.iter().map(|m| m.ev.since(t0).map_or(0.0, |ms| ms as f64 * 1e3)).collect();
    let host: Vec<f64> = marks.iter().map(|m| m.at.duration_since(marks[0].at).as_secs_f64() * 1e6).collect();
    let starts: Vec<usize> = marks.iter().enumerate().filter(|(_, m)| m.label == "mixer half start").map(|(i, _)| i).collect();
    let mut out = Vec::new();
    for (k, &a) in starts.iter().enumerate() {
        let end = starts.get(k + 1).copied().unwrap_or(marks.len() - 1);
        let find = |label: &str| (a..=end.min(marks.len() - 1)).find(|&i| marks[i].label == label);
        let (ffn, req, read, planned, prod, sums, sumq, last) = (
            find("feed-forward half start"),
            find("store: choices requested"),
            find("store: choices read"),
            find("store: planned"),
            find("store: products queued"),
            find("store: host sums sent"),
            find("store: sum queued"),
            find("mHC update (feed-forward)"),
        );
        let mut l = Layer { total: at[end] - at[a], ..Layer::default() };
        let (Some(ffn), Some(last)) = (ffn, last) else {
            l.always = l.total;
            out.push(l);
            continue;
        };
        l.always = at[ffn] - at[a];
        if let (Some(req), Some(read), Some(planned), Some(prod), Some(sumq)) = (req, read, planned, prod, sumq) {
            l.store = true;
            l.router = at[req] - at[ffn];
            l.readback = at[read] - at[req];
            l.plan = at[planned] - at[read];
            l.products = at[prod] - at[planned];
            l.host_experts = sums.map_or(0.0, |s| at[s] - at[prod]);
            l.sum = at[sumq] - sums.map_or(at[prod], |s| at[s]);
            l.post = at[end.max(last)] - at[sumq];
            l.host_queue = host[req] - host[a];
            l.host_wait = host[read] - host[req];
        } else {
            l.router = at[last] - at[ffn];
        }
        out.push(l);
    }
    out
}

/// The time from each marker's predecessor to it, in microseconds on the stream's clock, with the marker's label (the first marker has none).
pub fn stages(marks: &[Mark]) -> Vec<(&'static str, f64)> {
    marks.windows(2).map(|w| (w[1].label, w[1].ev.since(&w[0].ev).map_or(0.0, |ms| ms as f64 * 1e3))).collect()
}

/// A traced decode step: the parts of each layer, the stages, the step's milliseconds.
pub type TracedStep = (Vec<Layer>, Vec<(&'static str, f64)>, f64);

#[cfg(feature = "dev")]
/// The traced steps of a request (all but the first, which finds the cache as the prompt left it, where there are others): the mean of each layer's parts in microseconds on the stream's clock, the totals by part, and with
/// every stage marker the stages' totals.
pub(crate) fn print_trace(steps: &[TracedStep]) {
    let used = if steps.len() > 1 { &steps[1..] } else { steps };
    let n = used.len() as f64;
    let layers = used.iter().map(|s| s.0.len()).min().unwrap_or(0);
    let mean = |f: &dyn Fn(&Layer) -> f64, l: usize| used.iter().map(|s| f(&s.0[l])).sum::<f64>() / n;
    let step_ms = used.iter().map(|s| s.2).sum::<f64>() / n;
    let mut tot = [0f64; 11];
    let mut lines = vec![format!(
        "trace: {} decode steps ({} of them averaged), a step {step_ms:.1} ms; per layer, microseconds on the stream's clock",
        steps.len(),
        used.len()
    )];
    lines.push("  layer   always  router readback     plan products hostexp      sum     post |    total | host: queue   wait".into());
    for l in 0..layers {
        let f: [&dyn Fn(&Layer) -> f64; 11] = [
            &|x| x.always,
            &|x| x.router,
            &|x| x.readback,
            &|x| x.plan,
            &|x| x.products,
            &|x| x.host_experts,
            &|x| x.sum,
            &|x| x.post,
            &|x| x.total,
            &|x| x.host_queue,
            &|x| x.host_wait,
        ];
        let v: Vec<f64> = f.iter().map(|g| mean(g, l)).collect();
        for (t, x) in tot.iter_mut().zip(&v) {
            *t += x;
        }
        lines.push(format!(
            "  {l:>5} {:>8.0} {:>7.0} {:>8.0} {:>8.0} {:>8.0} {:>7.0} {:>8.0} {:>8.0} | {:>8.0} | {:>11.0} {:>6.0}",
            v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8], v[9], v[10]
        ));
    }
    lines.push(format!("  total {:>8.0} {:>7.0} {:>8.0} {:>8.0} {:>8.0} {:>7.0} {:>8.0} {:>8.0} | {:>8.0} | {:>11.0} {:>6.0}   (the layers; a step is {:.0} us with the head and the embedding)", tot[0], tot[1], tot[2], tot[3], tot[4], tot[5], tot[6], tot[7], tot[8], tot[9], tot[10], step_ms * 1e3));
    let sum = tot[8].max(1e-9);
    lines.push(format!(
        "  share of the layers: always-on {:.0}%, router and shared expert {:.0}%, readback {:.0}%, plan and copies {:.0}%, experts on the GPU {:.0}%, experts on the host {:.0}%, sum {:.0}%, update {:.0}%",
        100.0 * tot[0] / sum, 100.0 * tot[1] / sum, 100.0 * tot[2] / sum, 100.0 * tot[3] / sum, 100.0 * tot[4] / sum, 100.0 * tot[5] / sum, 100.0 * tot[6] / sum, 100.0 * tot[7] / sum
    ));
    // the stages (level 2): the time from the marker before to each marker, summed by its label over the layers, averaged over the steps
    if used.iter().any(|s| s.1.len() > 64) {
        let mut by: std::collections::BTreeMap<&str, f64> = Default::default();
        for s in used {
            for (label, us) in &s.1 {
                *by.entry(label).or_default() += us / n;
            }
        }
        let mut v: Vec<(&str, f64)> = by.into_iter().collect();
        v.sort_by(|a, b| b.1.total_cmp(&a.1));
        lines.push("  stages, microseconds a step (the time since the marker before), largest first:".into());
        for (label, us) in v.iter().take(40) {
            lines.push(format!("    {label:<34} {us:>8.0}"));
        }
    }
    // (each line on the standard error, with the seconds since the first line, as the server's log lines)
    static START: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();
    for l in lines {
        eprintln!("[{:7.1} s] {l}", START.get_or_init(Instant::now).elapsed().as_secs_f64());
    }
}
