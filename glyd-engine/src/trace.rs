//! A development build's timeline of the steps of a session: markers recorded on the engine's stream (a CUDA event each: the stream's own clock) with the host's clock beside them, for the steps `glmserve` is asked to trace
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
        let (ffn, req, read, planned, prod, sums, sumq, last) = (find("feed-forward half start"), find("store: choices requested"), find("store: choices read"), find("store: planned"), find("store: products queued"), find("store: host sums sent"), find("store: sum queued"), find("mHC update (feed-forward)"));
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
