//! The numbers of the chat page's Monitor tab (`GET /v1/glyd/status`): what the engine is doing (loading, reading a prompt, writing, idle), the last request's numbers and the last 30 requests, and the computer's
//! numbers once a second (the GPU's through nvidia-smi, the CPU's, the RAM's and the disk's through /proc) with the last two minutes of each. The engine's thread and the requests' threads write it, the page reads it.

use crate::server::jstr;
use std::collections::VecDeque;
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

const KEYS: [&str; 8] = ["gpu_load", "vram_used", "gpu_temp", "power", "cpu", "ram_used", "disk_read", "tok_s"];

#[derive(Default)]
struct Mon {
    mode: String,
    ctx: usize,
    state: &'static str,
    since: f64,
    prompt_tokens: usize,
    reused: usize,
    made: usize,
    writing_since: f64,
    live_tok_s: f64,
    /// The last request's engine numbers and the GPU's places (JSON objects), the last 30 requests (newest first).
    stats: String,
    places: String,
    requests: VecDeque<String>,
    /// The computer's numbers now, the ones that do not change (JSON fields), and two minutes of each of KEYS.
    now: Vec<Option<f64>>,
    fixed: String,
    hist: Vec<VecDeque<Option<f64>>>,
}

static MON: Mutex<Option<Mon>> = Mutex::new(None);

pub(crate) fn clock() -> f64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0.0, |d| d.as_secs_f64())
}

fn with<T>(f: impl FnOnce(&mut Mon) -> T) -> T {
    let mut g = MON.lock().unwrap_or_else(|e| e.into_inner());
    f(g.get_or_insert_with(|| Mon { state: "loading", since: clock(), stats: "{}".into(), places: "null".into(), now: vec![None; KEYS.len()], hist: vec![VecDeque::new(); KEYS.len()], fixed: String::new(), ..Mon::default() }))
}

/// Whether a server keeps these numbers (`start` was called).
pub fn on() -> bool {
    MON.lock().unwrap_or_else(|e| e.into_inner()).as_ref().is_some_and(|m| !m.mode.is_empty())
}

/// The server is up (the model loading): its mode and context, and the computer's numbers from now on.
pub fn start(mode: &str, ctx: usize) {
    with(|m| (m.mode, m.ctx) = (mode.to_string(), ctx));
    std::thread::spawn(|| {
        let mut prev = Machine::default();
        loop {
            let (now, fixed) = prev.sample();
            with(|m| {
                for (i, k) in KEYS.iter().enumerate() {
                    let v = if *k == "tok_s" { Some(m.live_tok_s) } else { now[i] };
                    m.now[i] = v;
                    m.hist[i].push_back(v);
                    if m.hist[i].len() > 120 {
                        m.hist[i].pop_front();
                    }
                }
                if !fixed.is_empty() {
                    m.fixed = fixed;
                }
            });
            std::thread::sleep(std::time::Duration::from_secs(1));
        }
    });
}

/// The GPU's places for experts as the load made them (GB, a layer).
pub fn places(gb: f64, per_layer: usize) {
    with(|m| m.places = format!("{{\"gb\":{gb:.2},\"per_layer\":{per_layer}}}"));
}

/// The model is loaded and waits for requests.
pub fn ready() {
    with(|m| (m.state, m.since) = ("idle", clock()));
}

/// A request begins: `prompt` tokens, `reused` of them the engine's state holds already.
pub fn begin(prompt: usize, reused: usize) {
    with(|m| (m.state, m.since, m.prompt_tokens, m.reused, m.made, m.live_tok_s) = ("reading", clock(), prompt, reused, 0, 0.0));
}

/// `n` more tokens written.
pub fn wrote(n: usize) {
    with(|m| {
        let t = clock();
        if m.state == "reading" {
            (m.state, m.writing_since) = ("writing", t);
        }
        m.made += n;
        m.live_tok_s = (m.made as f64 / (t - m.writing_since).max(1e-3) * 10.0).round() / 10.0;
    });
}

/// A request ended: `stats`, its engine numbers (the fields of a JSON object), and its row of the requests' table.
pub fn end(stats: &str, row: &str) {
    with(|m| {
        m.stats = format!("{{{stats}}}");
        push(m, format!("{{\"t\":{:.0},\"status\":\"done\",{row}}}", clock()));
    });
}

pub fn failed(prompt: usize, why: &str) {
    with(|m| push(m, format!("{{\"t\":{:.0},\"status\":\"error\",\"prompt\":{prompt},\"error\":{}}}", clock(), jstr(&why.chars().take(200).collect::<String>()))));
}

fn push(m: &mut Mon, row: String) {
    m.requests.push_front(row);
    m.requests.truncate(30);
    (m.state, m.since, m.live_tok_s) = ("idle", clock(), 0.0);
}

fn num(x: Option<f64>) -> String {
    x.map_or("null".into(), |v| format!("{v}"))
}

/// The status as the page reads it.
pub fn json(name: &str) -> String {
    with(|m| {
        let t = clock();
        let secs = ((t - m.since) * 10.0).round() / 10.0;
        // (a prompt's progress guessed from the time it has taken: about 4,200 tokens a second)
        let progress = if m.state == "reading" { (secs * 4200.0 / (m.prompt_tokens.saturating_sub(m.reused)).max(1) as f64).min(0.99) } else { 0.0 };
        let now: Vec<String> = KEYS.iter().zip(&m.now).map(|(k, v)| format!("\"{k}\":{}", num(*v))).collect();
        let fixed = if m.fixed.is_empty() { String::new() } else { format!("{},", m.fixed) };
        let hist: Vec<String> = KEYS.iter().zip(&m.hist).map(|(k, h)| format!("\"{k}\":[{}]", h.iter().map(|v| num(*v)).collect::<Vec<_>>().join(","))).collect();
        format!(
            "{{\"model\":{},\"mode\":{},\"ctx\":{},\"engine\":{{\"state\":\"{}\",\"seconds\":{secs},\"progress\":{progress:.3},\"prompt_tokens\":{},\"made\":{}}},\"waiting\":0,\"stats\":{},\"places\":{},\"machine\":{{{fixed}{}}},\"history\":{{{}}},\"requests\":[{}],\"live_tok_s\":{}}}",
            jstr(name),
            jstr(&m.mode),
            m.ctx,
            m.state,
            m.prompt_tokens,
            m.made,
            m.stats,
            m.places,
            now.join(","),
            hist.join(","),
            m.requests.iter().cloned().collect::<Vec<_>>().join(","),
            m.live_tok_s
        )
    })
}

/// The counters a sample's rates come from: the CPU's (idle, total) and the disks' sectors read with the time.
#[derive(Default)]
struct Machine {
    cpu: Option<(u64, u64)>,
    disk: Option<(u64, f64)>,
}

impl Machine {
    /// KEYS' values now (tok_s aside) and the fields that do not change (empty where nvidia-smi did not answer).
    fn sample(&mut self) -> (Vec<Option<f64>>, String) {
        let mut v = vec![None; KEYS.len()];
        let mut fixed = Vec::new();
        let q = "name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit,pcie.link.gen.current,pcie.link.width.current";
        if let Ok(o) = std::process::Command::new("nvidia-smi").args([&format!("--query-gpu={q}"), "--format=csv,noheader,nounits"]).output() {
            let text = String::from_utf8_lossy(&o.stdout);
            let f: Vec<&str> = text.lines().next().unwrap_or("").split(',').map(str::trim).collect();
            if f.len() == 9 {
                let n = |i: usize| f[i].parse::<f64>().ok();
                (v[0], v[1], v[2], v[3]) = (n(1), n(2), n(4), n(5));
                fixed.push(format!("\"gpu\":{},\"vram_total\":{},\"power_limit\":{},\"pcie\":{}", jstr(f[0]), num(n(3)), num(n(6)), jstr(&format!("Gen{} x{}", f[7], f[8]))));
            }
        }
        if let Some(line) = std::fs::read_to_string("/proc/stat").ok().and_then(|t| t.lines().next().map(str::to_string)) {
            let x: Vec<u64> = line.split_whitespace().skip(1).filter_map(|s| s.parse().ok()).collect();
            if x.len() > 4 {
                let (idle, total) = (x[3] + x[4], x.iter().sum::<u64>());
                if let Some((i0, t0)) = self.cpu {
                    v[4] = Some((1000.0 * (1.0 - (idle - i0) as f64 / (total - t0).max(1) as f64)).round() / 10.0);
                }
                self.cpu = Some((idle, total));
            }
        }
        if let Some(t) = crate::os::meminfo() {
            let kb = |k: &str| t.lines().find_map(|l| l.strip_prefix(k)?.trim().trim_end_matches("kB").trim().parse::<f64>().ok());
            if let (Some(total), Some(avail)) = (kb("MemTotal:"), kb("MemAvailable:")) {
                v[5] = Some(((total - avail) / 1048576.0 * 10.0).round() / 10.0);
                fixed.push(format!("\"ram_total\":{:.1}", total / 1048576.0));
            }
        }
        if let Ok(t) = std::fs::read_to_string("/proc/diskstats") {
            // (whole disks: nvme0n1, sda, vda; not their partitions)
            let whole = |d: &str| {
                let nvme = d.strip_prefix("nvme").is_some_and(|r| r.split_once('n').is_some_and(|(a, b)| !a.is_empty() && !b.is_empty() && a.bytes().chain(b.bytes()).all(|c| c.is_ascii_digit())));
                nvme || (d.len() == 3 && (d.starts_with("sd") || d.starts_with("vd")) && d.as_bytes()[2].is_ascii_lowercase())
            };
            let sectors: u64 = t.lines().filter_map(|l| {
                let f: Vec<&str> = l.split_whitespace().collect();
                (f.len() > 5 && whole(f[2])).then(|| f[5].parse::<u64>().unwrap_or(0))
            }).sum();
            let now = clock();
            if let Some((s0, t0)) = self.disk {
                v[6] = Some((sectors.saturating_sub(s0) as f64 * 512.0 / 1e6 / (now - t0).max(1e-3) * 10.0).round() / 10.0);
            }
            self.disk = Some((sectors, now));
        }
        fixed.push(format!("\"cores\":{}", std::thread::available_parallelism().map_or(0, |n| n.get())));
        (v, fixed.join(","))
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn the_status_is_json_with_what_the_page_reads() {
        super::start("fast", 1000);
        super::begin(100, 10);
        super::wrote(3);
        super::ready();
        super::end("\"prompt_tokens\":100", "\"prompt\":100,\"output\":3");
        let v = glyd_gpu::json::parse(&super::json("m")).unwrap();
        assert_eq!(v.get("engine").and_then(|e| e.get("state")).and_then(|s| s.as_str()), Some("idle"));
        assert_eq!(v.get("requests").and_then(|r| r.as_array()).map(|r| r.len()), Some(1));
        assert!(v.get("history").and_then(|h| h.get("tok_s")).is_some());
    }
}
