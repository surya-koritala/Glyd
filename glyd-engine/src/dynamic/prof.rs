//! GLYD_PROF=1 (a development build's): the executor's own per-operation GPU timing, with no external profiler. Every pass records a CUDA event after each
//! operation (`Dyn::mark`: inside the step's graph where the pass is a graph's replay, each graph with its own events), and after the pass the times between
//! them are added here by the operation's label, per kind of pass: decode rows alone, a prompt's chunks alone, or both in one pass. Each call of the scheduler's
//! step (`ModelStep::decode`, `round`, `prefill`, `prefill_many`; `Dyn::step` and `Dyn::prefill` where no scheduler runs, as `run`) is timed on the host
//! too: its wall time, the GPU time of its passes, the rest (the host's work, the copies and the waits around the passes) and the time between one call's
//! end and the next one's start (the scheduler, the sampler, the server).
//!
//! What it reads: an eager pass's operation is timed from the event after the operation before it, so its time holds the GPU waiting for the host to queue
//! it where the host is slower than the GPU (`queued_ms`, the host's time to queue the pass, says when that is so); a graph's are the GPU's own. The events
//! add a little to every pass (a graph's event nodes, an eager pass's records). `serve` hands the figures out at `GET /v1/glyd/prof` (`?reset=1` starts
//! again); `bench` reads them around each load's measured rounds and prints them under the load; `run` prints them when it ends.

use serde_json::{json, Value};
use std::sync::Mutex;
use std::time::Instant;

/// Whether GLYD_PROF=1 asks for the profile (a development build's; read once).
pub fn on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::tune!("GLYD_PROF").is_some_and(|v| v == "1"))
}

/// A kind of pass, or of step: the rows that decode in it and its prompt tokens (rounded up to a power of two, at least 64: chunks of about one size together).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct Key {
    pub rows: usize,
    pub prompt: usize,
}

impl Key {
    pub fn of(rows: usize, prompt: usize) -> Key {
        Key { rows, prompt: if prompt == 0 { 0 } else { prompt.next_power_of_two().max(64) } }
    }

    fn words(&self) -> String {
        let rows = |r: usize| format!("{r} row{}", if r == 1 { "" } else { "s" });
        let prompt = |p: usize| if p <= 64 { "up to 64 prompt tokens".to_string() } else { format!("{}-{p} prompt tokens", p / 2 + 1) };
        match (self.rows, self.prompt) {
            (r, 0) => format!("decode, {}", rows(r)),
            (0, p) => format!("prompt, {}", prompt(p)),
            (r, p) => format!("decode {} with {}", rows(r), prompt(p)),
        }
    }
}

/// One kind of pass's sums: passes, those that were a graph's replay, the pieces' graphs replayed (src/dynamic/pieces.rs), prompt tokens, the attention's decode calls (a launch
/// of the parts kernel or a call of the library's decode; a graph's as it was captured), the GPU's time, the host's time to queue the ones not a whole graph's, and each
/// operation's time (ms).
#[derive(Default)]
struct Passes {
    n: u64,
    graphs: u64,
    pieces: u64,
    tokens: u64,
    calls: u64,
    gpu: f64,
    queued: f64,
    ops: Vec<(&'static str, f64)>,
}

/// One kind of step's sums (ms): calls, their wall time, their passes' GPU time, and the time before each since the call before it ended (`gaps` of them).
#[derive(Default)]
struct Steps {
    n: u64,
    wall: f64,
    gpu: f64,
    between: f64,
    gaps: u64,
}

#[derive(Default)]
struct Acc {
    passes: Vec<(Key, Passes)>,
    steps: Vec<(Key, Steps)>,
    /// The GPU time of the passes of the step being timed, the end of the last step, and whether a step is being timed (a step within a step is not timed again).
    step_gpu: f64,
    last_end: Option<Instant>,
    in_step: bool,
}

static ACC: Mutex<Option<Acc>> = Mutex::new(None);

fn with<R>(f: impl FnOnce(&mut Acc) -> R) -> R {
    let mut g = ACC.lock().unwrap_or_else(|e| e.into_inner());
    f(g.get_or_insert_with(Acc::default))
}

fn entry<T: Default>(v: &mut Vec<(Key, T)>, k: Key) -> &mut T {
    let i = match v.iter().position(|(x, _)| *x == k) {
        Some(i) => i,
        None => {
            v.push((k, T::default()));
            v.len() - 1
        }
    };
    &mut v[i].1
}

/// A pass of `rows` decode rows and `prompt` prompt tokens: its operations' times (ms, in order, a label more than once summed), whether it was a graph's replay,
/// the host's time to queue it (ms; None for a replay), its attention's decode calls, and the pieces' graphs it replayed.
pub fn pass(rows: usize, prompt: usize, ops: &[(&'static str, f64)], graph: bool, queued: Option<f64>, calls: usize, pieces: usize) {
    let gpu: f64 = ops.iter().map(|x| x.1).sum();
    with(|a| {
        if a.in_step {
            a.step_gpu += gpu;
        }
        let p = entry(&mut a.passes, Key::of(rows, prompt));
        (p.n, p.graphs, p.tokens, p.gpu, p.queued, p.calls) =
            (p.n + 1, p.graphs + graph as u64, p.tokens + prompt as u64, p.gpu + gpu, p.queued + queued.unwrap_or(0.0), p.calls + calls as u64);
        p.pieces += pieces as u64;
        for &(k, ms) in ops {
            match p.ops.iter_mut().find(|x| x.0 == k) {
                Some(x) => x.1 += ms,
                None => p.ops.push((k, ms)),
            }
        }
    });
}

/// A step being timed (`begin`): when it began. None where the profile is off or a step is being timed already. One that ends without `end` (an error) is not counted.
pub struct Step(Instant);

impl Drop for Step {
    fn drop(&mut self) {
        with(|a| a.in_step = false);
    }
}

/// The start of a step's call (None: not timed).
pub fn begin() -> Option<Step> {
    if !on() {
        return None;
    }
    with(|a| {
        if a.in_step {
            return None;
        }
        (a.in_step, a.step_gpu) = (true, 0.0);
        Some(Step(Instant::now()))
    })
}

/// The end of the step `s` began, of `rows` decode rows and `prompt` prompt tokens.
pub fn end(s: Option<Step>, rows: usize, prompt: usize) {
    let Some(t0) = s.as_ref().map(|s| s.0) else { return };
    let now = Instant::now();
    with(|a| {
        let (gpu, last) = (a.step_gpu, a.last_end);
        (a.in_step, a.step_gpu, a.last_end) = (false, 0.0, Some(now));
        let st = entry(&mut a.steps, Key::of(rows, prompt));
        (st.n, st.wall, st.gpu) = (st.n + 1, st.wall + now.duration_since(t0).as_secs_f64() * 1e3, st.gpu + gpu);
        if let Some(l) = last.filter(|&l| l <= t0) {
            (st.between, st.gaps) = (st.between + t0.duration_since(l).as_secs_f64() * 1e3, st.gaps + 1);
        }
    });
}

/// The figures so far, as JSON (every time a mean: a pass's or a call's, ms); with `reset` they start again.
pub fn snapshot(reset: bool) -> Value {
    let r3 = |x: f64| (x * 1000.0).round() / 1000.0;
    with(|a| {
        let mut passes: Vec<&(Key, Passes)> = a.passes.iter().collect();
        passes.sort_by_key(|x| x.0);
        let mut steps: Vec<&(Key, Steps)> = a.steps.iter().collect();
        steps.sort_by_key(|x| x.0);
        let v = json!({
            "passes": passes.iter().map(|(k, p)| {
                let n = p.n.max(1) as f64;
                let eager = (p.n - p.graphs).max(1) as f64;
                json!({"kind": k.words(), "rows": k.rows, "prompt_tokens": r3(p.tokens as f64 / n), "passes": p.n, "graphs": p.graphs, "pieces": r3(p.pieces as f64 / n),
                    "decode_calls": r3(p.calls as f64 / n),
                    "gpu_ms": r3(p.gpu / n), "queued_ms": if p.graphs < p.n { json!(r3(p.queued / eager)) } else { Value::Null },
                    "ops": p.ops.iter().map(|(op, ms)| json!({"op": op, "ms": r3(ms / n), "share": r3(if p.gpu > 0.0 { ms / p.gpu } else { 0.0 })})).collect::<Vec<_>>()})
            }).collect::<Vec<_>>(),
            "steps": steps.iter().map(|(k, s)| {
                let n = s.n.max(1) as f64;
                json!({"kind": k.words(), "calls": s.n, "wall_ms": r3(s.wall / n), "gpu_ms": r3(s.gpu / n), "host_ms": r3((s.wall - s.gpu) / n),
                    "between_ms": if s.gaps > 0 { json!(r3(s.between / s.gaps as f64)) } else { Value::Null }})
            }).collect::<Vec<_>>(),
        });
        if reset {
            *a = Acc { last_end: a.last_end, ..Acc::default() };
        }
        v
    })
}

/// The figures of `snapshot` as lines to read: each kind of pass with its GPU time and its operations' (the largest first), then each kind of step.
pub fn text(v: &Value) -> String {
    let f = |x: &Value| x.as_f64().unwrap_or(0.0);
    let mut out = String::new();
    let passes = v["passes"].as_array().map_or(&[][..], |a| a.as_slice());
    if passes.is_empty() {
        return "GPU profile: no pass ran\n".to_string();
    }
    out.push_str("GPU profile (GLYD_PROF=1), ms a pass by operation (events after each; a graph's own where the pass replays one):\n");
    for p in passes {
        let (n, graphs) = (p["passes"].as_u64().unwrap_or(0), p["graphs"].as_u64().unwrap_or(0));
        let how = match graphs {
            0 => "eager".to_string(),
            g if g == n => "its graph".to_string(),
            g => format!("{g} in a graph"),
        };
        let (prompt, calls, pieces) = (f(&p["prompt_tokens"]), f(&p["decode_calls"]), f(&p["pieces"]));
        out.push_str(&format!(
            "{}: {n} passes ({how}){}; GPU {:.3} ms a pass{}{}{}\n",
            p["kind"].as_str().unwrap_or(""),
            if prompt > 0.0 { format!(", {prompt:.0} prompt tokens on average") } else { String::new() },
            f(&p["gpu_ms"]),
            p["queued_ms"].as_f64().map_or(String::new(), |q| format!(" (the host queued one not a graph's in {q:.3} ms)")),
            if calls > 0.0 { format!("; {calls:.0} attention decode calls a pass") } else { String::new() },
            if pieces > 0.0 { format!("; {pieces:.1} piece graphs replayed a pass") } else { String::new() }
        ));
        let mut ops: Vec<&Value> = p["ops"].as_array().map_or(Vec::new(), |a| a.iter().collect());
        ops.sort_by(|a, b| f(&b["ms"]).total_cmp(&f(&a["ms"])));
        for o in ops {
            out.push_str(&format!("  {:>9.3} {:>5.1}%  {}\n", f(&o["ms"]), 100.0 * f(&o["share"]), o["op"].as_str().unwrap_or("")));
        }
    }
    let steps = v["steps"].as_array().map_or(&[][..], |a| a.as_slice());
    if !steps.is_empty() {
        out.push_str(
            "steps (a call of the scheduler's step, or of a step where none runs), ms a call: wall = GPU (its passes) + the rest (host work, copies, waits)\n",
        );
        for s in steps {
            out.push_str(&format!(
                "{}: {} calls, wall {:.3} = GPU {:.3} + {:.3}{}\n",
                s["kind"].as_str().unwrap_or(""),
                s["calls"].as_u64().unwrap_or(0),
                f(&s["wall_ms"]),
                f(&s["gpu_ms"]),
                f(&s["host_ms"]),
                s["between_ms"].as_f64().map_or(String::new(), |b| format!("; {b:.3} between one call's end and the next one's start"))
            ));
        }
    }
    out
}

/// The times between consecutive events (ms), each by the later one's label: `evs` recorded in order on one stream, all of them complete.
pub(crate) fn between(evs: &[(&'static str, &crate::dev::Ev)]) -> crate::ffi::Result<Vec<(&'static str, f64)>> {
    let mut out: Vec<(&'static str, f64)> = Vec::new();
    for w in evs.windows(2) {
        let ms = w[1].1.since(w[0].1)? as f64;
        match out.iter_mut().find(|(k, _)| *k == w[1].0) {
            Some(x) => x.1 += ms,
            None => out.push((w[1].0, ms)),
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One test holds the accumulator at a time (it is the process's).
    static SERIAL: Mutex<()> = Mutex::new(());

    #[test]
    fn passes_are_summed_by_kind_and_label_and_reset() {
        let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        snapshot(true);
        pass(1, 0, &[("norms", 0.5), ("attention", 1.0), ("norms", 0.25)], true, None, 36, 0);
        pass(1, 0, &[("norms", 0.25), ("attention", 3.0)], true, None, 36, 0);
        pass(0, 1000, &[("attention", 40.0)], false, Some(2.0), 0, 37);
        pass(8, 0, &[("attention", 8.0)], false, Some(6.0), 288, 0);
        let v = snapshot(true);
        let p = v["passes"].as_array().unwrap();
        // (ordered by kind: prompt alone (0 rows), then 1 row, then 8)
        assert_eq!(p.iter().map(|x| x["kind"].as_str().unwrap()).collect::<Vec<_>>(), ["prompt, 513-1024 prompt tokens", "decode, 1 row", "decode, 8 rows"]);
        assert_eq!((p[1]["passes"].as_u64(), p[1]["graphs"].as_u64(), p[1]["gpu_ms"].as_f64()), (Some(2), Some(2), Some(2.5)));
        assert_eq!(p[1]["ops"][0]["op"], "norms");
        assert_eq!((p[1]["ops"][0]["ms"].as_f64(), p[1]["ops"][1]["ms"].as_f64(), p[1]["ops"][1]["share"].as_f64()), (Some(0.5), Some(2.0), Some(0.8)));
        assert!(p[1]["queued_ms"].is_null());
        assert_eq!((p[0]["prompt_tokens"].as_f64(), p[0]["queued_ms"].as_f64()), (Some(1000.0), Some(2.0)));
        assert!(snapshot(false)["passes"].as_array().unwrap().is_empty(), "a reset starts again");
        let t = text(&v);
        assert!(t.contains("decode, 1 row: 2 passes (its graph); GPU 2.500 ms a pass; 36 attention decode calls a pass"), "{t}");
        assert!(
            t.contains(
                "decode, 8 rows: 1 passes (eager); GPU 8.000 ms a pass (the host queued one not a graph's in 6.000 ms); 288 attention decode calls a pass"
            ),
            "{t}"
        );
        // (a pass in pieces: its piece graphs counted, the host's time to queue it kept)
        assert!(
            t.contains(
                "prompt, 513-1024 prompt tokens: 1 passes (eager), 1000 prompt tokens on average; GPU 40.000 ms a pass (the host queued one not a graph's in 2.000 ms); 37.0 piece graphs replayed a pass\n"
            ),
            "{t}"
        );
        assert_eq!(p[0]["pieces"].as_f64(), Some(37.0));
        // (the largest operation first)
        assert!(t.find("2.000  80.0%  attention").unwrap() < t.find("0.500  20.0%  norms").unwrap(), "{t}");
    }

    #[test]
    fn a_step_counts_its_passes_gpu_time_and_the_time_between_steps() {
        let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        snapshot(true);
        // (the profile is off unless GLYD_PROF=1: a step's timing made by hand, as `begin` makes it)
        let mark = || {
            with(|a| (a.in_step, a.step_gpu) = (true, 0.0));
            Some(Step(Instant::now()))
        };
        let s = mark();
        pass(1, 0, &[("a", 0.5)], true, None, 0, 0);
        std::thread::sleep(std::time::Duration::from_millis(2));
        end(s, 1, 0);
        std::thread::sleep(std::time::Duration::from_millis(1));
        let s = mark();
        pass(1, 0, &[("a", 0.5)], true, None, 0, 0);
        end(s, 1, 0);
        // (a pass outside a step: counted as a pass, not in a step)
        pass(1, 0, &[("a", 0.5)], true, None, 0, 0);
        let v = snapshot(true);
        let st = &v["steps"][0];
        assert_eq!((st["kind"].as_str(), st["calls"].as_u64(), st["gpu_ms"].as_f64()), (Some("decode, 1 row"), Some(2), Some(0.5)));
        assert!(st["wall_ms"].as_f64().unwrap() >= 1.0 && st["host_ms"].as_f64().unwrap() >= 0.5);
        assert!(st["between_ms"].as_f64().unwrap() >= 1.0, "{st}");
        assert_eq!(v["passes"][0]["passes"].as_u64(), Some(3));
    }

    #[test]
    fn prompts_are_grouped_by_the_power_of_two_above() {
        assert_eq!(Key::of(0, 4011), Key { rows: 0, prompt: 4096 });
        assert_eq!(Key::of(0, 4097), Key { rows: 0, prompt: 8192 });
        assert_eq!(Key::of(3, 10), Key { rows: 3, prompt: 64 });
        assert_eq!(Key::of(0, 4011).words(), "prompt, 2049-4096 prompt tokens");
        assert_eq!(Key::of(2, 300).words(), "decode 2 rows with 257-512 prompt tokens");
    }
}
