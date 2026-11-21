//! `glyd bench`: one benchmark harness (R8). It puts fixed prompts at fixed concurrencies on an OpenAI-compatible server over HTTP and writes the speed as JSON, `glyd-bench/1` (docs/bench.md).
//!
//!   glyd-engine bench MODEL[:LEVEL] [--suite gate|bar|full | --loads short:1,4k:8,...] [--runs 5] [--warmup 1] [--max-tokens 128] [--ctx N] [--out FILE] [--label TEXT]
//!                     [--against URL] [--weights-gib G] [--bandwidth-gbs B] [--port P] [--no-draft] [--eval gsm8k[:N]] [--quality [--quality-ref RESULT.json]]
//!       starts `serve MODEL` of this program (or measures the server at --against, whatever program it is: vLLM, llama.cpp, SGLang) and measures each load: per load, one warm-up round and `runs` rounds
//!       of N simultaneous requests, every prompt different from its first token on (no prefix to share). Time to first token, inter-token latency, decode tokens/s a user and in all, prefill tokens/s,
//!       each as the runs' mean, standard deviation, median and the runs themselves; the peak GPU memory; and the decode rate as a share of the ceiling (the weights' bytes a token over the memory's
//!       bandwidth: the engine's own weights from /v1/glyd/info, else --weights-gib).
//!   glyd-engine bench --compare BASE.json CAND.json [--history DIR] [--z 3] [--floor 0.003] [--kl-max 0.005] [--synthetic F]
//!       the regression gate: each metric of each load of CAND against BASE's, a regression where it is worse by more than the noise band: z times the measured spread of the runs of both files
//!       (the standard error of two medians) and, where DIR holds earlier results on the same machine, of the medians from one session to another. Exit 0 within the band, 1 a regression, 2 not
//!       decided (a result taken while the GPU was busy). --synthetic F: CAND's speeds scaled by F first (the gate's own test: 0.9 is a 10% slowdown).
//!   glyd-engine bench --bar ENGINE.json[,MORE] OTHER.json[,MORE] [--gpu-name NAME --cc M.m] [--family NAME] [--released DATE] [--stale-ok] [--noise 0.03]
//!       the speed bar of src/machine.rs (`GPUS`): at every load the first server's decode tokens/s (a user and in all) and its time to first token at least the second's (within the noise), its peak
//!       memory lower; a pass prints the line for `GPUS` (per (GPU, family): `--family`, default the file's, else Qwen3). The model measured must have been released within 6 months (`--released`, or the file's):
//!       exit 2 for an older one, unless `--stale-ok`. Exit 0, 1 or 2 (docs/adding-a-gpu.md).

use crate::args::Args;
use crate::ffi::{err, Error, Result};
use serde_json::{json, Value};
use std::io::{BufRead, BufReader};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

pub const SCHEMA: &str = "glyd-bench/1";

// ---- the fixed workload

const WORDS: &str = "river stone market letter garden window engine paper silver winter harbor signal ladder copper meadow lantern bridge orchard compass thunder village canvas anchor forest \
ribbon quartz pepper violet cabin marble kettle falcon summit harvest tunnel mirror saddle velvet crystal ember plains lagoon pigeon ceiling puzzle bottle shadow carpet planet button candle valley hammer \
island jacket cotton rocket temple orange basket";

const TOPICS: [&str; 8] = [
    "how a heat pump works",
    "why the sky is blue",
    "how a compiler turns source into machine code",
    "what makes bread rise",
    "how vaccines train the immune system",
    "why tides have two highs a day",
    "how public-key encryption works",
    "how a bicycle stays upright",
];

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

/// The tokens a prompt kind is about (None: a short question).
fn kind_tokens(kind: &str) -> Result<Option<usize>> {
    Ok(match kind {
        "short" => None,
        "1k" => Some(1024),
        "2k" => Some(2048),
        "4k" => Some(4096),
        "8k" => Some(8192),
        "16k" => Some(16384),
        "32k" => Some(32768),
        k => return err(format!("prompt {k}: short, 1k, 2k, 4k, 8k, 16k or 32k")),
    })
}

/// A prompt of the kind, different from every other seed's from the first token on. The long ones are random words (about 1.09 tokens a word) and a question that ignores them.
pub fn prompt(kind: &str, seed: u64) -> Result<String> {
    Ok(match kind_tokens(kind)? {
        None => format!("[{seed}] Explain {}, in detail, in about 300 words.", TOPICS[seed as usize % TOPICS.len()]),
        Some(n) => {
            let words: Vec<&str> = WORDS.split_whitespace().collect();
            let mut r = Rng(0x9E37_79B9_7F4A_7C15 ^ seed.wrapping_mul(0x2545_F491_4F6C_DD1D) | 1);
            let body: Vec<&str> = (0..((n.saturating_sub(50)) as f64 / 1.09) as usize).map(|_| words[(r.next() % words.len() as u64) as usize]).collect();
            format!("[{seed}] Here are some notes:\n{}\n\nIgnoring the notes, write a long, detailed essay about the history of bridges.", body.join(" "))
        }
    })
}

/// A suite's loads, (prompt kind, simultaneous requests).
pub fn suite(name: &str) -> Result<Vec<(String, usize)>> {
    let mk = |kinds: &[&str], concs: &[usize]| kinds.iter().flat_map(|k| concs.iter().map(move |&c| (k.to_string(), c))).collect::<Vec<_>>();
    Ok(match name {
        // what a pull request is held to: decode at 1 and 8 at once, prefill at 1K and 4K
        "gate" => [mk(&["short"], &[1, 8]), mk(&["1k", "4k"], &[1])].concat(),
        // the speed bar's matrix (docs/adding-a-gpu.md)
        "bar" => mk(&["short", "2k", "4k"], &[1, 4, 8]),
        // everything the issue lists: decode at 1, 8, 64 and 256 at once, prefill at 1K, 4K and 32K
        "full" => [mk(&["short"], &[1, 8, 64, 256]), mk(&["1k", "4k", "32k"], &[1])].concat(),
        s => return err(format!("--suite {s}: gate, bar or full")),
    })
}

fn parse_loads(s: &str) -> Result<Vec<(String, usize)>> {
    s.split(',')
        .map(|l| {
            let (k, c) = l.split_once(':').ok_or_else(|| Error(format!("--loads {l}: PROMPT:CONCURRENCY, as short:8")))?;
            kind_tokens(k)?;
            Ok((k.to_string(), c.parse::<usize>().ok().filter(|&c| c > 0).ok_or_else(|| Error(format!("--loads {l}: {c} is not a number of requests")))?))
        })
        .collect()
}

// ---- the client

#[derive(Default)]
struct Req {
    ttft: Option<f64>,
    tokens: u64,
    prompt_tokens: Option<u64>,
    first: Option<Instant>,
    last: Option<Instant>,
    gaps: Vec<f64>,
    error: Option<String>,
}

fn agent() -> ureq::Agent {
    ureq::Agent::new_with_config(ureq::Agent::config_builder().timeout_global(Some(Duration::from_secs(1800))).build())
}

/// One streamed chat completion: the time to the first content, each gap between chunks, the tokens (the server's usage, else the chunks).
fn one(agent: &ureq::Agent, url: &str, model: &str, kind: &str, seed: u64, max_tokens: usize) -> Req {
    let mut out = Req::default();
    let content = match prompt(kind, seed) {
        Ok(c) => c,
        Err(e) => {
            out.error = Some(e.0);
            return out;
        }
    };
    let body = json!({"model": model, "messages": [{"role": "user", "content": content}], "max_tokens": max_tokens, "temperature": 0, "stream": true,
        "stream_options": {"include_usage": true}, "chat_template_kwargs": {"enable_thinking": false}});
    let t0 = Instant::now();
    let mut resp = match agent.post(&format!("{url}/v1/chat/completions")).header("Content-Type", "application/json").send(body.to_string()) {
        Ok(r) => r,
        Err(e) => {
            out.error = Some(e.to_string());
            return out;
        }
    };
    let (mut chunks, mut usage_tokens) = (0u64, None);
    for line in BufReader::new(resp.body_mut().as_reader()).lines() {
        let Ok(line) = line else {
            out.error = Some("the stream broke".into());
            break;
        };
        let Some(data) = line.trim().strip_prefix("data:") else { continue };
        let data = data.trim();
        if data == "[DONE]" || data.is_empty() {
            continue;
        }
        let Ok(d) = serde_json::from_str::<Value>(data) else { continue };
        if let Some(u) = d.get("usage").filter(|u| !u.is_null()) {
            usage_tokens = u["completion_tokens"].as_u64();
            out.prompt_tokens = u["prompt_tokens"].as_u64();
        }
        let delta = &d["choices"][0]["delta"];
        if ["content", "reasoning_content", "reasoning"].iter().any(|k| delta[*k].as_str().is_some_and(|s| !s.is_empty())) {
            let now = Instant::now();
            if let Some(l) = out.last {
                out.gaps.push(now.duration_since(l).as_secs_f64());
            }
            out.first.get_or_insert(now);
            out.last = Some(now);
            chunks += 1;
        }
    }
    out.tokens = usage_tokens.unwrap_or(chunks);
    out.ttft = out.first.map(|f| f.duration_since(t0).as_secs_f64());
    if out.first.is_none() && out.error.is_none() {
        out.error = Some("no content came".into());
    }
    out
}

fn median(xs: &[f64]) -> Option<f64> {
    let mut v: Vec<f64> = xs.iter().copied().filter(|x| x.is_finite()).collect();
    if v.is_empty() {
        return None;
    }
    v.sort_by(f64::total_cmp);
    Some(if v.len() % 2 == 1 { v[v.len() / 2] } else { (v[v.len() / 2 - 1] + v[v.len() / 2]) / 2.0 })
}

fn percentile(xs: &[f64], p: f64) -> Option<f64> {
    let mut v: Vec<f64> = xs.iter().copied().filter(|x| x.is_finite()).collect();
    if v.is_empty() {
        return None;
    }
    v.sort_by(f64::total_cmp);
    Some(v[((v.len() - 1) as f64 * p).round() as usize])
}

/// The runs' figures as the file keeps them: mean, sample standard deviation, median, smallest, largest and the runs.
pub fn stats(runs: &[f64]) -> Value {
    let n = runs.len() as f64;
    let mean = runs.iter().sum::<f64>() / n.max(1.0);
    let sd = if runs.len() > 1 { (runs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt() } else { 0.0 };
    let round = |x: f64| (x * 1e6).round() / 1e6;
    json!({"mean": round(mean), "sd": round(sd), "median": round(median(runs).unwrap_or(f64::NAN)),
        "min": round(runs.iter().cloned().fold(f64::INFINITY, f64::min)), "max": round(runs.iter().cloned().fold(f64::NEG_INFINITY, f64::max)),
        "runs": runs.iter().map(|&x| round(x)).collect::<Vec<_>>()})
}

/// One round of `n` simultaneous requests: (time to first token, decode tokens/s, mean gap in ms, the gaps' 99th percentile in ms, all users' tokens after the first over the span, prompt tokens/s, requests that failed, prompt tokens).
struct Round {
    ttft: Option<f64>,
    decode: Option<f64>,
    itl: Option<f64>,
    itl_p99: Option<f64>,
    agg: Option<f64>,
    prefill: Option<f64>,
    errors: usize,
    prompt_tokens: Option<f64>,
}

fn round(agent: &ureq::Agent, url: &str, model: &str, kind: &str, n: usize, seed: &mut u64, max_tokens: usize) -> Round {
    let seeds: Vec<u64> = (0..n)
        .map(|_| {
            *seed += 1;
            *seed
        })
        .collect();
    let reqs: Vec<Req> = std::thread::scope(|s| {
        let hs: Vec<_> = seeds.iter().map(|&sd| s.spawn(move || one(agent, url, model, kind, sd, max_tokens))).collect();
        hs.into_iter().map(|h| h.join().unwrap_or_default()).collect()
    });
    let ok: Vec<&Req> = reqs.iter().filter(|r| r.error.is_none() && r.ttft.is_some()).collect();
    let tps = |r: &Req| match (r.first, r.last) {
        (Some(f), Some(l)) if l > f && r.tokens > 1 => Some((r.tokens - 1) as f64 / l.duration_since(f).as_secs_f64()),
        _ => None,
    };
    let decode: Vec<f64> = ok.iter().filter_map(|r| tps(r)).collect();
    let gaps: Vec<f64> = ok.iter().flat_map(|r| r.gaps.iter().copied()).collect();
    let span = match (ok.iter().filter_map(|r| r.first).min(), ok.iter().filter_map(|r| r.last).max()) {
        (Some(f), Some(l)) if l > f => Some(l.duration_since(f).as_secs_f64()),
        _ => None,
    };
    let after_first: u64 = ok.iter().map(|r| r.tokens.saturating_sub(1)).sum();
    let prompt_total: Option<u64> = ok.iter().map(|r| r.prompt_tokens).sum();
    let slowest_first = ok.iter().filter_map(|r| r.ttft).fold(f64::NEG_INFINITY, f64::max);
    Round {
        ttft: median(&ok.iter().filter_map(|r| r.ttft).collect::<Vec<_>>()),
        decode: median(&decode),
        itl: median(&decode).map(|t| 1000.0 / t),
        itl_p99: percentile(&gaps, 0.99).map(|g| g * 1000.0),
        agg: span.map(|s| after_first as f64 / s),
        prefill: prompt_total.filter(|_| slowest_first.is_finite() && slowest_first > 0.0).map(|p| p as f64 / slowest_first),
        errors: reqs.len() - ok.len(),
        prompt_tokens: median(&ok.iter().filter_map(|r| r.prompt_tokens.map(|p| p as f64)).collect::<Vec<_>>()),
    }
}

// ---- the machine

fn smi(query: &str) -> Option<String> {
    let o = Command::new("nvidia-smi").args([&format!("--query-gpu={query}"), "--format=csv,noheader,nounits"]).stderr(Stdio::null()).output().ok()?;
    o.status.success().then(|| String::from_utf8_lossy(&o.stdout).lines().next().unwrap_or("").trim().to_string())
}

fn num_of(s: &str) -> Option<f64> {
    s.trim().parse::<f64>().ok()
}

/// A GPU's memory read rate by its spec sheet, GB/s, by its name and memory (None where the table does not know it: --bandwidth-gbs).
pub fn spec_bandwidth(name: &str, memory_mib: f64) -> Option<f64> {
    let n = name.to_uppercase();
    let has = |w: &str| n.split(|c: char| !c.is_alphanumeric()).any(|x| x == w);
    let pair = |a: &str, b: &str| has(a) && has(b);
    Some(if pair("4080", "SUPER") {
        736.0
    } else if has("4090") {
        1008.0
    } else if has("4080") {
        716.8
    } else if has("5090") {
        1792.0
    } else if has("3090") {
        936.0
    } else if has("L40S") {
        864.0
    } else if has("L4") {
        300.0
    } else if has("A10") {
        600.0
    } else if has("A100") {
        if memory_mib > 50_000.0 {
            2039.0
        } else {
            1555.0
        }
    } else {
        return None;
    })
}

fn machine() -> Value {
    let q = smi("name,driver_version,memory.total,clocks.max.sm,clocks.max.memory,power.limit,compute_cap").unwrap_or_default();
    let f: Vec<&str> = q.split(',').map(str::trim).collect();
    let get = |i: usize| f.get(i).copied().unwrap_or("");
    let cpu = std::fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|t| t.lines().find(|l| l.starts_with("model name")).map(|l| l.split(':').nth(1).unwrap_or("").trim().to_string()));
    json!({"gpu": get(0), "driver": get(1), "memory_mib": num_of(get(2)), "max_sm_mhz": num_of(get(3)), "max_mem_mhz": num_of(get(4)), "power_limit_w": num_of(get(5)), "compute_capability": get(6),
        "cpu": cpu, "cpu_threads": std::thread::available_parallelism().map(|n| n.get()).ok(), "os": std::env::consts::OS, "arch": std::env::consts::ARCH})
}

/// The GPU as it is now: used memory (MiB), utilization (%), the SM and memory clocks (MHz), the computer's one-minute load.
fn gpu_now() -> Value {
    let q = smi("memory.used,utilization.gpu,clocks.sm,clocks.mem").unwrap_or_default();
    let f: Vec<Option<f64>> = q.split(',').map(num_of).collect();
    let g = |i: usize| f.get(i).copied().flatten();
    let load = std::fs::read_to_string("/proc/loadavg").ok().and_then(|t| t.split_whitespace().next().and_then(num_of));
    json!({"memory_used_mib": g(0), "utilization_pct": g(1), "sm_mhz": g(2), "mem_mhz": g(3), "load_1m": load})
}

/// Peak GPU memory (MiB) while it runs.
struct Peak {
    stop: std::sync::Arc<std::sync::atomic::AtomicBool>,
    peak: std::sync::Arc<std::sync::atomic::AtomicU64>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl Peak {
    fn start() -> Peak {
        use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
        let (stop, peak) = (std::sync::Arc::new(AtomicBool::new(false)), std::sync::Arc::new(AtomicU64::new(0)));
        let (s, p) = (stop.clone(), peak.clone());
        let handle = std::thread::spawn(move || {
            while !s.load(Ordering::Relaxed) {
                if let Some(m) = smi("memory.used").and_then(|x| num_of(&x)) {
                    p.fetch_max(m as u64, Ordering::Relaxed);
                }
                std::thread::sleep(Duration::from_millis(100));
            }
        });
        Peak { stop, peak, handle: Some(handle) }
    }

    fn finish(mut self) -> u64 {
        self.stop.store(true, std::sync::atomic::Ordering::Relaxed);
        if let Some(h) = self.handle.take() {
            h.join().ok();
        }
        self.peak.load(std::sync::atomic::Ordering::Relaxed)
    }
}

// ---- the server

struct Server {
    child: Child,
    url: String,
    /// The directory of this run's server log (private to this run, removed with the server).
    dir: std::path::PathBuf,
}

impl Drop for Server {
    fn drop(&mut self) {
        self.child.kill().ok();
        self.child.wait().ok();
        std::fs::remove_dir_all(&self.dir).ok();
    }
}

/// A directory of this run alone (the process and the time in its name, made new, 0700 where the system has modes): a name another user could have made first is an error, not a place to write.
fn private_dir() -> Result<std::path::PathBuf> {
    let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_nanos());
    let dir = std::env::temp_dir().join(format!("glyd-bench-{}-{nanos}", std::process::id()));
    std::fs::create_dir(&dir).map_err(|e| Error(format!("{}: {e}", dir.display())))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o700)).map_err(|e| Error(e.to_string()))?;
    }
    Ok(dir)
}

fn free_port() -> Result<u16> {
    let l = std::net::TcpListener::bind("127.0.0.1:0").map_err(|e| Error(e.to_string()))?;
    l.local_addr().map(|a| a.port()).map_err(|e| Error(e.to_string()))
}

/// `serve MODEL` of this program on a port of its own, up when /v1/models answers (the first start downloads and packs the model).
fn start_server(model: &str, a: &Args, users: usize) -> Result<Server> {
    let port = match a.flag("port") {
        Some(p) => p.parse::<u16>().map_err(|_| Error(format!("--port {p}: not a port")))?,
        None => free_port()?,
    };
    let exe = std::env::current_exe().map_err(|e| Error(e.to_string()))?;
    let dir = private_dir()?;
    let log = dir.join("server.log");
    let out = std::fs::OpenOptions::new().write(true).create_new(true).open(&log).map_err(|e| Error(format!("{}: {e}", log.display())))?;
    let mut cmd = Command::new(exe);
    cmd.args(["serve", model, "--host", "127.0.0.1", "--port", &port.to_string()]);
    for f in ["ctx", "layout", "chunk"] {
        if let Some(v) = a.flag(f) {
            cmd.args([format!("--{f}"), v.to_string()]);
        }
    }
    // (the server's places for requests at once: its default of 4 would make 8 at once 4 and 4 waiting; the most the loads ask for, at most the 8 it takes)
    cmd.args(["--users".to_string(), a.flag("users").map_or(users.to_string(), str::to_string)]);
    if a.has("no-draft") {
        // (the draft model's speculative decoding off: the decode rate is then the weights' bytes over the bandwidth, which the ceiling is)
        cmd.arg("--no-draft");
    }
    let child = cmd
        .stdout(Stdio::from(out.try_clone().map_err(|e| Error(e.to_string()))?))
        .stderr(Stdio::from(out))
        .spawn()
        .map_err(|e| Error(format!("serve: {e}")))?;
    let mut server = Server { child, url: format!("http://127.0.0.1:{port}"), dir };
    let up = Instant::now();
    loop {
        if let Ok(Some(status)) = server.child.try_wait() {
            let tail: String =
                std::fs::read_to_string(&log).unwrap_or_default().lines().rev().take(8).collect::<Vec<_>>().into_iter().rev().collect::<Vec<_>>().join("\n");
            return err(format!("serve {model} stopped ({status}):\n{tail}"));
        }
        if ureq::get(&format!("{}/v1/models", server.url)).call().is_ok() {
            return Ok(server);
        }
        if up.elapsed() > Duration::from_secs(1800) {
            return err(format!("serve {model}: not up after 30 minutes (log: {})", log.display()));
        }
        std::thread::sleep(Duration::from_millis(500));
    }
}

fn get_json(url: &str) -> Option<Value> {
    let mut r = ureq::get(url).call().ok()?;
    serde_json::from_str(&r.body_mut().read_to_string().ok()?).ok()
}

// ---- the measurement

fn git_sha() -> Option<String> {
    std::env::var("GITHUB_SHA").ok().map(|s| s.chars().take(10).collect()).or_else(|| {
        Command::new("git")
            .args(["rev-parse", "--short=10", "HEAD"])
            .stderr(Stdio::null())
            .output()
            .ok()
            .filter(|o| o.status.success())
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
    })
}

/// `bench MODEL ...`: see the top of this file. Prints the table, writes --out.
pub fn measure(a: &Args) -> Result<()> {
    let against = a.flag("against").map(|u| u.trim_end_matches('/').to_string());
    let model_arg = a.pos.get(1).map(String::as_str);
    if against.is_none() && model_arg.is_none() {
        return err("bench MODEL[:LEVEL] (this program's `serve` is measured), or bench --against URL (any OpenAI-compatible server)");
    }
    let loads = match a.flag("loads") {
        Some(l) => parse_loads(l)?,
        None => suite(a.flag("suite").unwrap_or("gate"))?,
    };
    let (runs, warmup, max_tokens): (usize, usize, usize) = (a.num("runs", 5)?, a.num("warmup", 1)?, a.num("max-tokens", 128)?);
    if runs == 0 {
        return err("--runs 0");
    }
    let machine = machine();
    // (the GPU as it is before the server starts, five looks 0.3 s apart: busy where its median utilization is past 10%, or more than a desktop's memory is used: another process on it)
    let mut looks: Vec<Value> = Vec::new();
    for i in 0..5 {
        if i > 0 {
            std::thread::sleep(Duration::from_millis(300));
        }
        looks.push(gpu_now());
    }
    let util = median(&looks.iter().filter_map(|l| l["utilization_pct"].as_f64()).collect::<Vec<_>>());
    let before = looks.last().cloned().unwrap_or(Value::Null);
    let idle = before["memory_used_mib"].as_f64().unwrap_or(0.0);
    let busy = util.is_some_and(|u| u > 10.0) || idle > 2500.0;
    let peak = Peak::start();
    let server = match &against {
        Some(_) => None,
        None => Some(start_server(model_arg.unwrap_or(""), a, loads.iter().map(|l| l.1).max().unwrap_or(1).min(8))?),
    };
    let url = against.clone().unwrap_or_else(|| server.as_ref().map(|s| s.url.clone()).unwrap_or_default());
    let models = get_json(&format!("{url}/v1/models")).ok_or_else(|| Error(format!("{url}/v1/models: no answer")))?;
    let model = a
        .flag("model")
        .map(str::to_string)
        .or_else(|| models["data"][0]["id"].as_str().map(str::to_string))
        .ok_or_else(|| Error("the server lists no model".into()))?;
    let info = get_json(&format!("{url}/v1/glyd/info"));
    let weights = a
        .flag("weights-gib")
        .map(|g| g.parse::<f64>().map(|g| g * (1u64 << 30) as f64))
        .transpose()
        .map_err(|_| Error("--weights-gib: not a number".into()))?
        .or_else(|| info.as_ref().and_then(|i| i["memory"]["weights_bytes"].as_f64()));
    let bandwidth = a
        .flag("bandwidth-gbs")
        .and_then(num_of)
        .or_else(|| spec_bandwidth(machine["gpu"].as_str().unwrap_or(""), machine["memory_mib"].as_f64().unwrap_or(0.0)));
    let ceiling = weights.zip(bandwidth).map(|(w, b)| b * 1e9 / w);
    let agent = agent();
    let quiet = a.has("quiet");
    let mut seed = 1000u64;
    // (a warm-up that is not a load of the matrix: kernels, graphs and the server's own caches)
    let _ = one(&agent, &url, &model, "short", 0, 16);
    let _ = one(&agent, &url, &model, "1k", 1, 16);
    if !quiet {
        eprintln!(
            "{:<12} {:>5} {:>9} {:>14} {:>16} {:>12} {:>12} {:>10}",
            "load", "runs", "prompt", "first token s", "decode tok/s user", "total tok/s", "prefill tok/s", "of ceiling"
        );
    }
    let mut cells = Vec::new();
    for (kind, n) in &loads {
        for _ in 0..warmup {
            round(&agent, &url, &model, kind, *n, &mut seed, max_tokens);
        }
        let rs: Vec<Round> = (0..runs).map(|_| round(&agent, &url, &model, kind, *n, &mut seed, max_tokens)).collect();
        let col = |f: &dyn Fn(&Round) -> Option<f64>| rs.iter().filter_map(f).collect::<Vec<f64>>();
        let (ttft, decode, itl, itl99, agg, prefill) =
            (col(&|r| r.ttft), col(&|r| r.decode), col(&|r| r.itl), col(&|r| r.itl_p99), col(&|r| r.agg), col(&|r| r.prefill));
        let errors: usize = rs.iter().map(|r| r.errors).sum();
        let pt = median(&col(&|r| r.prompt_tokens));
        let cell_ceiling = ceiling;
        let pct = |x: &[f64]| cell_ceiling.zip(median(x)).map(|(c, d)| d / c);
        if !quiet {
            let m = |x: &[f64]| median(x).map_or("-".to_string(), |v| if v < 10.0 { format!("{v:.3}") } else { format!("{v:.1}") });
            eprintln!(
                "{:<12} {:>5} {:>9} {:>14} {:>16} {:>12} {:>12} {:>10}{}",
                format!("{kind} x{n}"),
                runs,
                pt.map_or("-".into(), |p| format!("{p:.0}")),
                m(&ttft),
                m(&decode),
                m(&agg),
                m(&prefill),
                pct(&decode).map_or("-".into(), |p| format!("{:.0}%", p * 100.0)),
                if errors > 0 { format!("  ({errors} requests failed)") } else { String::new() }
            );
        }
        let mut cell = json!({"prompt": kind, "conc": n, "prompt_tokens": pt, "errors": errors,
            "ttft_s": stats(&ttft), "decode_tps_per_user": stats(&decode), "itl_ms": stats(&itl), "itl_p99_ms": stats(&itl99), "agg_tps": stats(&agg), "prefill_tps": stats(&prefill)});
        if let (Some(c), Some(p)) = (cell_ceiling, pct(&decode)) {
            cell["ceiling_tps_per_user"] = json!((c * 100.0).round() / 100.0);
            cell["pct_of_ceiling"] = json!((p * 10000.0).round() / 10000.0);
        }
        cells.push(cell);
    }
    let mut quality = a.flag("eval").map(|e| match e.strip_prefix("gsm8k") {
        Some(rest) => {
            let n = rest.strip_prefix(':').and_then(|n| n.parse::<usize>().ok()).unwrap_or(30);
            let r = gsm8k(&agent, &url, &model, n);
            if !quiet {
                eprintln!("gsm8k: {} of {} correct", r["correct"], r["problems"]);
            }
            json!({"gsm8k": r})
        }
        None => json!({"error": format!("--eval {e}: gsm8k or gsm8k:N")}),
    });
    if a.has("quality") {
        // (the next-token distribution after fixed prefixes of one text: what a reference server gave, and how far this one is from it)
        let rows = next_token_rows(&agent, &url, &model);
        let mut q = json!({"next_token": rows});
        if let Some(rf) = a.flag("quality-ref") {
            let reference = load(rf)?;
            q["kl_vs_ref"] = kl_vs_ref(&reference["quality"]["next_token"], &q["next_token"]);
            if !quiet {
                eprintln!(
                    "quality: KL against {rf}: mean {} max {} (best token the reference's at {} of {})",
                    q["kl_vs_ref"]["mean"], q["kl_vs_ref"]["max"], q["kl_vs_ref"]["top1_agree"], q["kl_vs_ref"]["prefixes"]
                );
            }
        }
        match quality.as_mut() {
            Some(o) => {
                o["next_token"] = q["next_token"].take();
                if !q["kl_vs_ref"].is_null() {
                    o["kl_vs_ref"] = q["kl_vs_ref"].take();
                }
            }
            None => quality = Some(q),
        }
    }
    let peak_mib = peak.finish();
    let after = gpu_now();
    let result = json!({
        "schema": SCHEMA,
        "label": a.flag("label").unwrap_or(""),
        "recorded_unix": std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs()),
        "machine": machine,
        "model": {"id": model, "argument": model_arg, "level": info.as_ref().and_then(|i| i["level"]["effective"].as_str()), "weights_bytes": weights,
            "weights_source": if a.flag("weights-gib").is_some() { "flag" } else if weights.is_some() { "/v1/glyd/info" } else { "unknown" }},
        "bandwidth": {"gbs": bandwidth, "source": if a.flag("bandwidth-gbs").is_some() { "flag" } else if bandwidth.is_some() { "spec sheet" } else { "unknown" }},
        "versions": {"glyd_engine": env!("CARGO_PKG_VERSION"), "git": git_sha(), "server": info.as_ref().and_then(|i| i["build"]["version"].as_str())},
        "server": {"against": against, "started_by_bench": server.is_some(), "users": server.as_ref().map(|_| a.flag("users").and_then(|u| u.parse::<usize>().ok()).unwrap_or_else(|| loads.iter().map(|l| l.1).max().unwrap_or(1).min(8)))},
        "config": {"suite": a.flag("suite").unwrap_or(if a.flag("loads").is_some() { "loads" } else { "gate" }), "loads": loads.iter().map(|(k, n)| format!("{k}:{n}")).collect::<Vec<_>>(),
            "runs": runs, "warmup": warmup, "max_tokens": max_tokens},
        "contention": {"busy_before": busy, "utilization_median_pct": util, "before": before, "after": after},
        "idle_mib": idle,
        "peak_mib": peak_mib,
        "quality": quality,
        "cells": cells,
    });
    drop(server);
    match a.flag("out") {
        Some(f) => std::fs::write(f, serde_json::to_string_pretty(&result).unwrap_or_default() + "\n").map_err(|e| Error(format!("{f}: {e}")))?,
        None => println!("{}", serde_json::to_string_pretty(&result).unwrap_or_default()),
    }
    if busy {
        eprintln!(
            "bench: the GPU was busy before the run began ({idle} MiB used, {}% utilized): the result is marked contended and the gate will not decide on it",
            util.unwrap_or(0.0)
        );
    }
    Ok(())
}

// ---- a small eval

/// The last number in a reply (commas out of it), as GSM8K's answers are compared.
pub fn last_number(text: &str) -> Option<String> {
    let mut best = None;
    let mut cur = String::new();
    let flush = |cur: &mut String, best: &mut Option<String>| {
        let t = cur.trim_end_matches('.').to_string();
        if t.chars().any(|c| c.is_ascii_digit()) {
            *best = Some(t);
        }
        cur.clear();
    };
    for c in text.chars() {
        if c.is_ascii_digit() || (c == '.' && !cur.is_empty()) || (c == '-' && cur.is_empty()) {
            cur.push(c);
        } else if c == ',' && !cur.is_empty() {
            // (a thousands separator)
        } else {
            flush(&mut cur, &mut best);
        }
    }
    flush(&mut cur, &mut best);
    best
}

fn same_number(a: &str, b: &str) -> bool {
    match (a.parse::<f64>(), b.parse::<f64>()) {
        (Ok(x), Ok(y)) => (x - y).abs() < 1e-6,
        _ => a == b,
    }
}

/// `--eval gsm8k[:N]`: the first N (30) problems of GSM8K's test split (read from the Hub's datasets server, MIT license), each asked of the server at temperature 0 with eight at once; the share whose last number is the
/// answer after "####". Not a benchmark of the model's reasoning, a check that a change did not make it worse: the same problems every time.
fn gsm8k(agent: &ureq::Agent, url: &str, model: &str, n: usize) -> Value {
    let rows = ureq::get(&format!("https://datasets-server.huggingface.co/rows?dataset=openai/gsm8k&config=main&split=test&offset=0&length={n}"))
        .call()
        .ok()
        .and_then(|mut r| r.body_mut().read_to_string().ok())
        .and_then(|t| serde_json::from_str::<Value>(&t).ok());
    let Some(rows) = rows.as_ref().and_then(|r| r["rows"].as_array()) else {
        return json!({"error": "the problems could not be read from datasets-server.huggingface.co"});
    };
    let problems: Vec<(String, String)> = rows
        .iter()
        .filter_map(|r| Some((r["row"]["question"].as_str()?.to_string(), r["row"]["answer"].as_str()?.rsplit("####").next()?.trim().replace(',', ""))))
        .collect();
    let done = std::sync::atomic::AtomicUsize::new(0);
    let correct = std::sync::atomic::AtomicUsize::new(0);
    let next = std::sync::atomic::AtomicUsize::new(0);
    std::thread::scope(|s| {
        for _ in 0..8 {
            s.spawn(|| loop {
                let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let Some((q, want)) = problems.get(i) else { break };
                let body = json!({"model": model, "messages": [{"role": "user", "content": format!("{q}\nThink step by step, then end with the final answer as a number alone on the last line.")}],
                    "max_tokens": 768, "temperature": 0, "stream": false, "chat_template_kwargs": {"enable_thinking": false}});
                let reply = agent
                    .post(&format!("{url}/v1/chat/completions"))
                    .header("Content-Type", "application/json")
                    .send(body.to_string())
                    .ok()
                    .and_then(|mut r| r.body_mut().read_to_string().ok())
                    .and_then(|t| serde_json::from_str::<Value>(&t).ok())
                    .and_then(|v| v["choices"][0]["message"]["content"].as_str().map(str::to_string));
                done.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if reply.and_then(|t| last_number(&t)).is_some_and(|got| same_number(&got, want)) {
                    correct.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
            });
        }
    });
    let (c, d) = (correct.into_inner(), done.into_inner());
    json!({"source": "openai/gsm8k test, first problems", "problems": d, "correct": c, "accuracy": if d > 0 { c as f64 / d as f64 } else { f64::NAN }})
}

// ---- quality: the next token after fixed prefixes of one text

/// A text of about 330 words; its prefixes are the contexts the next-token distributions are taken at.
const PASSAGE: &str = "The lighthouse keeper climbed the spiral stairs every evening at dusk, a lantern swinging from one hand and a notebook tucked under the other arm. \
From the gallery he could see the whole harbour: the fishing boats nosing back to their moorings, the ferry drawing a long white line across the grey water, \
and beyond the breakwater the open sea, which that night was calm enough to hold the first stars like coins in a dish. He wrote down the wind, the swell and the \
visibility in a small neat hand, then trimmed the wick, wound the clockwork and listened for the soft click that told him the great lens had begun to turn. \
Somewhere below a door closed, a kettle began to sing, and the beam swept out over the dark water, steady as a promise, reaching for ships that would never know his name. \
In the morning the harbour master would telephone for the report, and the keeper would read it out in the same even voice he used for the weather and the tides. \
Nobody asked how the night had been, and he did not offer; the work was the lamp, and the lamp had burned. Years earlier an engineer had arrived with a crate of \
instruments and a plan to automate the light, and for a week the two men had argued politely over tea about whether a machine could be trusted with something so quiet. \
The engineer left with his measurements and a jar of the keeper's plum jam, and the light stayed as it was, though the keeper sometimes thought about the plan on long nights. \
When the storms came in autumn the whole tower hummed like a struck bell, and he slept in the lamp room with a blanket over his knees, waking at every change in the sound. \
In spring the gulls returned to the rail of the gallery and argued with him about the cleaning, and he gave up each year a little sooner than the last.";

/// A prefix count, and the words the first prefix has: the prefixes grow by the same number of words to the whole text.
const PREFIXES: usize = 48;
const TOP: usize = 20;

/// Each prefix of PASSAGE as a raw completion of one token at temperature 0, its top 20 log-probabilities: rows of [token, logprob] pairs. A prefix that fails is an empty row.
fn next_token_rows(agent: &ureq::Agent, url: &str, model: &str) -> Value {
    let words: Vec<&str> = PASSAGE.split_whitespace().collect();
    let first = 24usize.min(words.len());
    let step = (words.len() - first) / (PREFIXES - 1);
    let mut rows = Vec::new();
    for i in 0..PREFIXES {
        let n = (first + i * step).min(words.len());
        let body = json!({"model": model, "prompt": words[..n].join(" "), "max_tokens": 1, "temperature": 0, "logprobs": TOP});
        let pairs: Vec<Value> = agent
            .post(&format!("{url}/v1/completions"))
            .header("Content-Type", "application/json")
            .send(body.to_string())
            .ok()
            .and_then(|mut r| r.body_mut().read_to_string().ok())
            .and_then(|t| serde_json::from_str::<Value>(&t).ok())
            .and_then(|v| v["choices"][0]["logprobs"]["top_logprobs"][0].as_object().map(|o| o.iter().map(|(k, x)| json!([k, x])).collect()))
            .unwrap_or_default();
        rows.push(Value::Array(pairs));
    }
    json!({"prefixes": PREFIXES, "top": TOP, "words_first": first, "words_step": step, "rows": rows})
}

/// KL(reference || candidate) over the reference's top tokens and the rest as one bin, for one prefix; the candidate's probability of a token it did not list is its smallest listed one.
fn kl_row(reference: &[(String, f64)], cand: &[(String, f64)]) -> Option<(f64, bool)> {
    if reference.is_empty() || cand.is_empty() {
        return None;
    }
    let floor = cand.iter().map(|c| c.1).fold(f64::INFINITY, f64::min);
    let (mut kl, mut p_sum, mut q_sum) = (0f64, 0f64, 0f64);
    for (tok, lp) in reference {
        let lq = cand.iter().find(|c| &c.0 == tok).map_or(floor - 1.0, |c| c.1);
        let p = lp.exp();
        kl += p * (lp - lq);
        p_sum += p;
        q_sum += lq.exp();
    }
    let (pt, qt) = ((1.0 - p_sum).max(0.0), (1.0 - q_sum).max(0.0));
    if pt > 1e-9 {
        kl += pt * (pt / qt.max(1e-300)).ln();
    }
    let best = |v: &[(String, f64)]| v.iter().max_by(|a, b| a.1.total_cmp(&b.1)).map(|x| x.0.clone());
    Some((kl.max(0.0), best(reference) == best(cand)))
}

fn pairs_of(row: &Value) -> Vec<(String, f64)> {
    row.as_array().into_iter().flatten().filter_map(|p| Some((p[0].as_str()?.to_string(), p[1].as_f64()?))).collect()
}

/// The candidate's next-token rows against the reference's: mean and largest KL, and at how many prefixes the best token is the reference's.
pub fn kl_vs_ref(reference: &Value, cand: &Value) -> Value {
    let (r, c) = (reference["rows"].as_array().cloned().unwrap_or_default(), cand["rows"].as_array().cloned().unwrap_or_default());
    let per: Vec<(f64, bool)> = r.iter().zip(&c).filter_map(|(a, b)| kl_row(&pairs_of(a), &pairs_of(b))).collect();
    if per.is_empty() {
        return json!({"error": "no prefix has a distribution in both", "prefixes": 0});
    }
    let mean = per.iter().map(|x| x.0).sum::<f64>() / per.len() as f64;
    let max = per.iter().map(|x| x.0).fold(0.0, f64::max);
    json!({"prefixes": per.len(), "mean": (mean * 1e6).round() / 1e6, "max": (max * 1e6).round() / 1e6, "top1_agree": per.iter().filter(|x| x.1).count()})
}

// ---- the gate

fn load(path: &str) -> Result<Value> {
    let v: Value =
        serde_json::from_str(&std::fs::read_to_string(path).map_err(|e| Error(format!("{path}: {e}")))?).map_err(|e| Error(format!("{path}: {e}")))?;
    if v.get("summary_of_runs").is_some() {
        return Ok(from_summary(&v));
    }
    if v["schema"] != SCHEMA {
        return err(format!("{path}: schema {}, where {SCHEMA} is read", v["schema"]));
    }
    Ok(v)
}

/// The means and spreads of the summary file the earlier `load.py` and `gpu_bar.py --summarize` kept (benchmarks/gpu/rtx4080s-bar-2026-11-20): as a result, each figure the one run of its mean, with the spread it had.
fn from_summary(v: &Value) -> Value {
    let n = v["summary_of_runs"].as_u64().unwrap_or(1) as usize;
    let one = |c: &Value, key: &str| {
        let x = c[key].as_f64().unwrap_or(f64::NAN);
        let sd = c[format!("{key}_sd")].as_f64().unwrap_or(0.0) * x;
        // (n runs at the mean, the standard deviation kept in the object as the file had it)
        let mut st = stats(&vec![x; n.max(1)]);
        st["sd"] = json!(sd);
        st
    };
    let cells: Vec<Value> = v["cells"]
        .as_array()
        .into_iter()
        .flatten()
        .map(|c| json!({"prompt": c["prompt"], "conc": c["conc"], "errors": c["errors"].as_u64().unwrap_or(0), "decode_tps_per_user": one(c, "decode_tps_per_user"), "agg_tps": one(c, "agg_tps"), "ttft_s": one(c, "ttft_s")}))
        .collect();
    // (the model it was measured on, when that model was released and its family, where the file says: `gpu_bar.py --summarize ... --model ID --released DATE --family NAME`)
    json!({"schema": SCHEMA, "peak_mib": v["peak_mib"], "config": {"runs": n}, "legacy_summary": true, "cells": cells, "machine": {}, "model": {"id": v["model"], "released": v["released"], "family": v["family"]}, "contention": {"busy_before": false}})
}

/// (metric, higher is better)
const METRICS: [(&str, bool); 4] = [("decode_tps_per_user", true), ("agg_tps", true), ("prefill_tps", true), ("ttft_s", false)];

fn cell<'a>(v: &'a Value, prompt: &str, conc: u64) -> Option<&'a Value> {
    v["cells"].as_array()?.iter().find(|c| c["prompt"] == prompt && c["conc"] == conc)
}

fn f(v: &Value, k: &str) -> Option<f64> {
    v[k].as_f64().filter(|x| x.is_finite())
}

/// One metric's change and the band it is held to. Returns (base median, candidate median, change that is worse as a fraction, band as a fraction, the band's source).
pub fn judge(base: &Value, cand: &Value, hist: &[f64], z: f64, floor: f64, higher: bool) -> Option<(f64, f64, f64, f64, &'static str)> {
    let (mb, mc) = (f(base, "median")?, f(cand, "median")?);
    let n = |v: &Value| v["runs"].as_array().map_or(1, |r| r.len().max(1)) as f64;
    let rel_sd = |v: &Value| match (f(v, "sd"), f(v, "mean")) {
        (Some(s), Some(m)) if m != 0.0 => s / m,
        _ => 0.0,
    };
    // the standard error of the difference of two medians of n runs: 1.25 sd / sqrt(n) each
    let within = z * 1.25 * ((rel_sd(base).powi(2) / n(base)) + (rel_sd(cand).powi(2) / n(cand))).sqrt();
    let mut band = within.max(floor);
    let mut source = if within >= floor { "the runs' spread" } else { "the floor" };
    if hist.len() >= 3 {
        let mean = hist.iter().sum::<f64>() / hist.len() as f64;
        let sd = (hist.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (hist.len() - 1) as f64).sqrt();
        let across = z * (sd / mean.abs().max(1e-12)) * (1.0 + 1.0 / hist.len() as f64).sqrt();
        if across > band {
            band = across;
            source = "the sessions' spread";
        }
    }
    let worse = if higher { (mb - mc) / mb } else { (mc - mb) / mb };
    Some((mb, mc, worse, band, source))
}

/// `bench --compare BASE CAND`: see the top of this file.
pub fn compare(a: &Args) -> Result<i32> {
    let (bp, cp) = (a.need(1, "BASE.json")?, a.need(2, "CAND.json")?);
    let (base, mut cand) = (load(bp)?, load(cp)?);
    let (z, floor): (f64, f64) = (a.num("z", 3.0)?, a.num("floor", 0.003)?);
    if let Some(syn) = a.flag("synthetic").map(|s| s.parse::<f64>().map_err(|_| Error("--synthetic: not a number".into()))).transpose()? {
        // (the candidate's speeds scaled, its times the other way: a slowdown of 1 - F)
        if let Some(cells) = cand["cells"].as_array_mut() {
            for c in cells {
                for (m, higher) in METRICS {
                    if let Some(o) = c[m].as_object_mut() {
                        for k in ["mean", "median", "min", "max"] {
                            if let Some(x) = o.get(k).and_then(Value::as_f64) {
                                o.insert(k.into(), json!(if higher { x * syn } else { x / syn }));
                            }
                        }
                    }
                }
            }
        }
    }
    let (gb, gc) = (base["machine"]["gpu"].as_str().unwrap_or(""), cand["machine"]["gpu"].as_str().unwrap_or(""));
    if gb != gc {
        return err(format!("the results are of different GPUs ({gb:?}, {gc:?}): a gate compares one GPU's with its own"));
    }
    if base["model"]["id"] != cand["model"]["id"] {
        return err(format!("the results are of different models ({}, {})", base["model"]["id"], cand["model"]["id"]));
    }
    let history: Vec<Value> = match a.flag("history") {
        Some(d) => {
            let mut files: Vec<_> = std::fs::read_dir(d)
                .map_err(|e| Error(format!("{d}: {e}")))?
                .filter_map(|e| e.ok())
                .map(|e| e.path())
                .filter(|p| p.extension().is_some_and(|x| x == "json"))
                .collect();
            files.sort();
            files
                .iter()
                .filter_map(|p| load(&p.to_string_lossy()).ok())
                .filter(|h| h["machine"]["gpu"] == base["machine"]["gpu"] && h["model"]["id"] == base["model"]["id"] && h["contention"]["busy_before"] != true)
                .collect()
        }
        None => Vec::new(),
    };
    let contended = cand["contention"]["busy_before"] == true;
    let mut regressions = Vec::new();
    println!("{:<12} {:<20} {:>10} {:>10} {:>9} {:>9}  {}", "load", "metric", "base", "candidate", "change", "band", "");
    for c in cand["cells"].as_array().into_iter().flatten() {
        let (kind, conc) = (c["prompt"].as_str().unwrap_or(""), c["conc"].as_u64().unwrap_or(0));
        let Some(b) = cell(&base, kind, conc) else {
            println!("{:<12} not in the base result", format!("{kind} x{conc}"));
            continue;
        };
        if c["errors"].as_u64().unwrap_or(0) > 0 {
            regressions.push(format!("{kind} x{conc}: {} requests failed", c["errors"]));
        }
        for (m, higher) in METRICS {
            let hist: Vec<f64> = history.iter().filter_map(|h| cell(h, kind, conc).and_then(|hc| f(&hc[m], "median"))).collect();
            let Some((mb, mc, worse, band, source)) = judge(&b[m], &c[m], &hist, z, floor, higher) else { continue };
            let verdict = if worse > band {
                regressions.push(format!(
                    "{kind} x{conc}: {m} {mc:.4} against {mb:.4}: {:.1}% worse, the band is {:.1}% ({source})",
                    worse * 100.0,
                    band * 100.0
                ));
                "REGRESSION"
            } else if -worse > band {
                "faster"
            } else {
                ""
            };
            println!("{:<12} {:<20} {:>10.3} {:>10.3} {:>+8.1}% {:>8.1}%  {verdict}", format!("{kind} x{conc}"), m, mb, mc, -worse * 100.0, band * 100.0);
        }
    }
    // quality: the candidate's next-token distributions against the base's (the same prefixes of the same text); a regression past --kl-max (0.005 mean) or a best token that changes at more than 5% of the prefixes
    let (bq, cq) = (&base["quality"]["next_token"], &cand["quality"]["next_token"]);
    if bq["rows"].is_array() && cq["rows"].is_array() {
        let k = kl_vs_ref(bq, cq);
        let (mean, n, agree) = (k["mean"].as_f64().unwrap_or(f64::NAN), k["prefixes"].as_u64().unwrap_or(0), k["top1_agree"].as_u64().unwrap_or(0));
        let kl_max: f64 = a.num("kl-max", 0.005)?;
        let bad = mean > kl_max || (n > 0 && (agree as f64) < 0.95 * n as f64);
        println!(
            "{:<12} {:<20} KL mean {mean:.2e} max {}, best token the base's at {agree} of {n}{}",
            "quality",
            "next token",
            k["max"],
            if bad { "  REGRESSION" } else { "" }
        );
        if bad {
            regressions.push(format!("quality: next-token KL {mean:.2e} (bound {kl_max:.0e}), best token kept at {agree} of {n} prefixes"));
        }
    }
    println!("{} earlier results of this GPU and model in the history", history.len());
    if !regressions.is_empty() {
        println!("\nREGRESSION:\n  {}", regressions.join("\n  "));
        if contended {
            println!("(the candidate was measured on a busy GPU: not decided)");
            return Ok(2);
        }
        return Ok(1);
    }
    if contended {
        println!("the candidate was measured on a busy GPU: within the band, but not decided");
        return Ok(2);
    }
    println!("\nwithin the noise band");
    Ok(0)
}

// ---- the speed bar

/// The words of the driver's name for the GPU that tell it from its neighbours (src/machine.rs matches them as whole words): "NVIDIA GeForce RTX 4080 SUPER" is "RTX 4080 SUPER", "NVIDIA L4" is "L4",
/// "NVIDIA A100-SXM4-80GB" is "A100".
pub fn gpu_word(name: &str) -> String {
    let n = name.trim();
    let n = n.strip_prefix("NVIDIA").map_or(n, str::trim_start);
    let n = n.strip_prefix("GeForce").map_or(n, str::trim_start);
    let words: Vec<&str> = n.split_whitespace().collect();
    if words.first() == Some(&"RTX") && words.get(1).is_some_and(|w| w.chars().next().is_some_and(|c| c.is_ascii_digit())) {
        return words.join(" ");
    }
    let first = n.split(|c: char| !c.is_alphanumeric()).next().unwrap_or("");
    let letters = first.chars().take_while(|c| c.is_ascii_alphabetic()).count();
    if letters > 0 && first.chars().nth(letters).is_some_and(|c| c.is_ascii_digit()) {
        first.to_string()
    } else {
        n.to_string()
    }
}

/// Several result files of one server as one: each load's runs together.
fn merged(paths: &str) -> Result<Value> {
    let mut all: Vec<Value> = paths.split(',').map(load).collect::<Result<_>>()?;
    let mut first = all.remove(0);
    for other in &all {
        for oc in other["cells"].as_array().into_iter().flatten() {
            let (kind, conc) = (oc["prompt"].as_str().unwrap_or(""), oc["conc"].as_u64().unwrap_or(0));
            let Some(cells) = first["cells"].as_array_mut() else { continue };
            let Some(c) = cells.iter_mut().find(|c| c["prompt"] == kind && c["conc"] == conc) else { continue };
            for (m, _) in METRICS {
                let mut runs: Vec<f64> = c[m]["runs"].as_array().into_iter().flatten().filter_map(Value::as_f64).collect();
                runs.extend(oc[m]["runs"].as_array().into_iter().flatten().filter_map(Value::as_f64));
                c[m] = stats(&runs);
            }
            c["errors"] = json!(c["errors"].as_u64().unwrap_or(0) + oc["errors"].as_u64().unwrap_or(0));
        }
        first["peak_mib"] = json!(first["peak_mib"].as_u64().unwrap_or(0).max(other["peak_mib"].as_u64().unwrap_or(0)));
    }
    Ok(first)
}

/// Days since 1970-01-01 of a civil date (Howard Hinnant's algorithm).
fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + d - 1;
    era * 146_097 + yoe * 365 + yoe / 4 - yoe / 100 + doy - 719_468
}

/// The days between a release date ("2025-04-27", or the Hub's "2025-04-27T03:42:21.000Z") and `today` (days since the epoch); None where it does not parse.
pub fn age_days(released: &str, today: i64) -> Option<i64> {
    let mut p = released.get(..10)?.split('-').map(|x| x.parse::<i64>());
    let (y, m, d) = (p.next()?.ok()?, p.next()?.ok()?, p.next()?.ok()?);
    ((1..=12).contains(&m) && (1..=31).contains(&d)).then(|| today - days_from_civil(y, m, d))
}

/// The owner's rule: a measurement's model was released within the last 6 months.
pub const MAX_MODEL_AGE_DAYS: i64 = 183;

/// `bench --bar ENGINE OTHER`: see the top of this file.
pub fn bar(a: &Args) -> Result<i32> {
    let (e, v) = (merged(a.need(1, "ENGINE.json")?)?, merged(a.need(2, "OTHER.json")?)?);
    let noise: f64 = a.num("noise", 0.03)?;
    let name = a.flag("gpu-name").map(str::to_string).or_else(|| e["machine"]["gpu"].as_str().map(str::to_string)).unwrap_or_default();
    let cc =
        a.flag("cc").map(str::to_string).or_else(|| e["machine"]["compute_capability"].as_str().map(str::to_string)).ok_or_else(|| Error("--cc M.m".into()))?;
    let family =
        a.flag("family").map(str::to_string).or_else(|| e["model"]["family"].as_str().map(str::to_string)).unwrap_or_else(|| crate::machine::QWEN3.to_string());
    let released = a.flag("released").map(str::to_string).or_else(|| e["model"]["released"].as_str().map(str::to_string));
    let model = e["model"]["id"].as_str().map(str::to_string);
    let today = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs() as i64 / 86_400);
    let old = match released.as_deref().and_then(|r| age_days(r, today)) {
        Some(age) => {
            println!("measured on {} (family {family}), released {} days ago", model.as_deref().unwrap_or("a model the file does not name"), age);
            age > MAX_MODEL_AGE_DAYS
        }
        None => {
            println!("NOTE: the file does not say when its model was released (--released DATE, or `--summarize ... --released DATE` / the Hub's createdAt)");
            !a.has("stale-ok")
        }
    };
    let n = |x: &Value| x["config"]["runs"].as_u64().unwrap_or(0);
    println!("{} runs of the first server, {} of the second: medians (the first's standard deviation as a percent of its mean in brackets)", n(&e), n(&v));
    println!("{:<14}{:>30}{:>30}{:>30}", "load", "decode/user", "decode total", "first token (s)");
    let mut fails = Vec::new();
    for ec in e["cells"].as_array().into_iter().flatten() {
        let (kind, conc) = (ec["prompt"].as_str().unwrap_or(""), ec["conc"].as_u64().unwrap_or(0));
        let Some(vc) = cell(&v, kind, conc) else { continue };
        let mut row = format!("{:<14}", format!("{kind} x{conc}"));
        for (m, higher) in [("decode_tps_per_user", true), ("agg_tps", true), ("ttft_s", false)] {
            let (x, y) = (f(&ec[m], "mean"), f(&vc[m], "mean"));
            let (Some(x), Some(y)) = (x, y) else {
                row += &format!("{:>30}", "n/a");
                fails.push(format!("{kind} x{conc}: {m} missing"));
                continue;
            };
            let ratio = if higher { x / y } else { y / x };
            let sd = f(&ec[m], "sd").map_or(0.0, |s| s / x * 100.0);
            row += &format!("{:>30}", format!("{x:.3} / {y:.3} {ratio:.2} ({sd:.1}%)"));
            if ratio < 1.0 - noise {
                fails.push(format!("{kind} x{conc}: {m} {x:.3} against {y:.3} ({:.0}% behind)", (1.0 - ratio) * 100.0));
            }
        }
        if ec["errors"].as_u64().unwrap_or(0) + vc["errors"].as_u64().unwrap_or(0) > 0 {
            fails.push(format!("{kind} x{conc}: requests failed (first {}, second {})", ec["errors"], vc["errors"]));
        }
        println!("{row}");
    }
    let (pe, pv) = (e["peak_mib"].as_u64(), v["peak_mib"].as_u64());
    println!("peak GPU memory: first {} MiB, second {} MiB", pe.unwrap_or(0), pv.unwrap_or(0));
    if pe.zip(pv).is_none_or(|(x, y)| x >= y) {
        fails.push(format!("memory: {} MiB is not less than {} MiB", pe.unwrap_or(0), pv.unwrap_or(0)));
    }
    if !fails.is_empty() {
        println!("\nFAILS the bar:\n  {}", fails.join("\n  "));
        return Ok(1);
    }
    if old && !a.has("stale-ok") {
        println!("\nPasses the bar, but its model is older than {MAX_MODEL_AGE_DAYS} days (6 months): no line. --stale-ok keeps it, for a family with no newer model.");
        return Ok(2);
    }
    if old {
        println!("\nNOTE: --stale-ok: the model is older than 6 months (or its date is not given); the family has no newer one.");
    }
    let (major, minor) = cc.split_once('.').unwrap_or((&cc, "0"));
    println!("\nPASSES the bar for {family} models. For GPUS in glyd-engine/src/machine.rs (a GPU already there: add the family to its line):\n    (\"{}\", ({major}, {minor}), &[\"{family}\"]),", gpu_word(&name));
    Ok(0)
}

/// The command: `bench MODEL`, `bench --compare`, `bench --bar`.
pub fn cmd(a: &Args) -> i32 {
    let r = if a.has("compare") {
        compare(a)
    } else if a.has("bar") {
        bar(a)
    } else {
        measure(a).map(|_| 0)
    };
    match r {
        Ok(code) => code,
        Err(e) => {
            eprintln!("glyd-engine bench: {e}");
            1
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn runs(base: f64, rel_sd: f64, n: usize) -> Value {
        // (a spread of rel_sd around base, alternating: the sample sd is about rel_sd * base)
        let r: Vec<f64> = (0..n).map(|i| base * (1.0 + if i % 2 == 0 { rel_sd } else { -rel_sd })).collect();
        stats(&r)
    }

    #[test]
    fn prompts_differ_from_the_first_token_and_are_the_same_every_time() {
        assert_ne!(prompt("short", 1).unwrap(), prompt("short", 2).unwrap());
        assert_eq!(prompt("4k", 7).unwrap(), prompt("4k", 7).unwrap());
        let (a, b) = (prompt("1k", 1).unwrap(), prompt("1k", 2).unwrap());
        assert!(a.len() > 4000 && a != b && a[..3] != b[..3]);
        assert!(prompt("9k", 1).is_err());
    }

    #[test]
    fn kl_of_next_token_rows_is_zero_for_the_same_server_and_grows_with_a_shifted_distribution() {
        let row = |lps: &[f64]| json!(lps.iter().enumerate().map(|(i, l)| json!([format!("t{i}"), l])).collect::<Vec<_>>());
        let lp = |ps: &[f64]| ps.iter().map(|p| p.ln()).collect::<Vec<f64>>();
        let a = json!({"rows": [row(&lp(&[0.5, 0.2, 0.1, 0.05])), row(&lp(&[0.4, 0.3, 0.1, 0.1]))]});
        let same = kl_vs_ref(&a, &a);
        assert!(same["mean"].as_f64().unwrap() < 1e-9 && same["top1_agree"] == 2, "{same}");
        // the best token swapped: the distribution moved
        let b = json!({"rows": [row(&lp(&[0.1, 0.5, 0.2, 0.05])), row(&lp(&[0.4, 0.3, 0.1, 0.1]))]});
        let moved = kl_vs_ref(&a, &b);
        assert!(moved["mean"].as_f64().unwrap() > 0.05 && moved["top1_agree"] == 1, "{moved}");
        // a failed prefix (an empty row) is left out, not counted as agreement
        let c = json!({"rows": [json!([]), row(&lp(&[0.4, 0.3, 0.1, 0.1]))]});
        assert_eq!(kl_vs_ref(&a, &c)["prefixes"], 1);
    }

    #[test]
    fn the_next_token_rows_are_read_from_a_completions_answer() {
        use std::io::{Read, Write};
        let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = l.local_addr().unwrap().port();
        std::thread::spawn(move || {
            for s in l.incoming().flatten().take(PREFIXES) {
                let mut s = s;
                // (the whole request, headers and body, is read before the answer: a socket closed with a request unread is reset on Windows, and the answer is lost with it)
                let (mut req, mut buf) = (Vec::new(), [0u8; 4096]);
                loop {
                    let n = s.read(&mut buf).unwrap_or(0);
                    req.extend_from_slice(&buf[..n]);
                    let text = String::from_utf8_lossy(&req).to_string();
                    let Some(end) = text.find("\r\n\r\n") else {
                        if n == 0 {
                            break;
                        }
                        continue;
                    };
                    let len = text[..end]
                        .lines()
                        .find_map(|l| l.to_ascii_lowercase().strip_prefix("content-length:").and_then(|v| v.trim().parse::<usize>().ok()))
                        .unwrap_or(0);
                    if req.len() >= end + 4 + len || n == 0 {
                        break;
                    }
                }
                let body = r#"{"choices":[{"logprobs":{"top_logprobs":[{" the":-0.5,"a":-1.5}]}}]}"#;
                let _ = write!(s, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len());
            }
        });
        let rows = next_token_rows(&agent(), &format!("http://127.0.0.1:{port}"), "m");
        assert_eq!(rows["rows"].as_array().unwrap().len(), PREFIXES);
        assert_eq!(rows["rows"][0].as_array().unwrap().len(), 2);
        assert!(kl_vs_ref(&rows, &rows)["mean"].as_f64().unwrap() < 1e-9);
    }

    #[test]
    fn the_prompt_prefixes_grow_and_end_at_the_whole_text() {
        let words = PASSAGE.split_whitespace().count();
        assert!(words >= 300, "{words}");
        let first = 24usize;
        let step = (words - first) / (PREFIXES - 1);
        assert!(first + (PREFIXES - 1) * step <= words && first + (PREFIXES - 1) * step + PREFIXES > words);
    }

    #[test]
    fn the_last_number_of_a_reply_is_the_answer() {
        assert_eq!(last_number("So she has 1,234 apples.\n\n18").as_deref(), Some("18"));
        assert_eq!(last_number("The answer is $-5.50.").as_deref(), Some("-5.50"));
        assert_eq!(last_number("no digits"), None);
        assert!(same_number("18.0", "18") && !same_number("19", "18"));
    }

    #[test]
    fn suites_and_loads_are_read() {
        assert_eq!(suite("gate").unwrap().len(), 4);
        assert!(suite("full").unwrap().contains(&("short".to_string(), 256)));
        assert_eq!(parse_loads("short:1,4k:8").unwrap(), vec![("short".to_string(), 1), ("4k".to_string(), 8)]);
        assert!(parse_loads("short").is_err() && parse_loads("short:0").is_err());
    }

    #[test]
    fn the_band_is_the_measured_spread_and_a_slowdown_past_it_is_a_regression() {
        let base = runs(60.0, 0.004, 5);
        // the same speed: within the band
        let (_, _, worse, band, _) = judge(&base, &runs(60.0, 0.004, 5), &[], 3.0, 0.003, true).unwrap();
        assert!(worse.abs() < band, "{worse} {band}");
        // a 2% slowdown on a spread of 0.4%: past the band (3 * 1.25 * 0.004 * sqrt(2/5) = 0.95%)
        let (_, _, worse, band, _) = judge(&base, &runs(58.8, 0.004, 5), &[], 3.0, 0.003, true).unwrap();
        assert!(worse > band, "{worse} {band}");
        // the same 2% where the runs spread 3%: within the band (a noisy machine is held to less)
        let (_, _, worse, band, _) = judge(&runs(60.0, 0.03, 5), &runs(58.8, 0.03, 5), &[], 3.0, 0.003, true).unwrap();
        assert!(worse < band, "{worse} {band}");
        // a time is worse where it is longer
        let (_, _, worse, _, _) = judge(&runs(0.1, 0.004, 5), &runs(0.11, 0.004, 5), &[], 3.0, 0.003, false).unwrap();
        assert!(worse > 0.09);
    }

    #[test]
    fn sessions_that_differ_widen_the_band() {
        let (b, c) = (runs(60.0, 0.002, 5), runs(59.0, 0.002, 5));
        let (_, _, worse, tight, _) = judge(&b, &c, &[], 3.0, 0.003, true).unwrap();
        assert!(worse > tight);
        let (_, _, _, wide, source) = judge(&b, &c, &[60.0, 58.5, 61.5, 59.0, 60.5], 3.0, 0.003, true).unwrap();
        assert!(wide > tight && source == "the sessions' spread", "{wide}");
    }

    /// The owner's rule: a measurement's model was released within the last 6 months; dates are read as the Hub's createdAt gives them.
    #[test]
    fn a_models_age_is_counted_in_days_from_its_release() {
        let today = days_from_civil(2026, 11, 20);
        assert_eq!(age_days("2026-11-20", today), Some(0));
        assert_eq!(age_days("2026-05-23T01:17:15.000Z", today), Some(181));
        assert_eq!(age_days("2025-04-27T03:42:21.000Z", today), Some(572));
        assert!(age_days("2026-05-23", today).unwrap() <= MAX_MODEL_AGE_DAYS && age_days("2026-05-19", today).unwrap() > MAX_MODEL_AGE_DAYS);
        assert_eq!(age_days("not a date", today), None);
        assert_eq!(age_days("2026-13-01", today), None);
        assert_eq!(days_from_civil(1970, 1, 1), 0);
        assert_eq!(days_from_civil(2000, 3, 1), 11_017);
    }

    #[test]
    fn gpu_words_are_the_names_that_tell_neighbours_apart() {
        assert_eq!(gpu_word("NVIDIA GeForce RTX 4080 SUPER"), "RTX 4080 SUPER");
        assert_eq!(gpu_word("NVIDIA L4"), "L4");
        assert_eq!(gpu_word("NVIDIA A100-SXM4-80GB"), "A100");
        assert_eq!(spec_bandwidth("NVIDIA GeForce RTX 4080 SUPER", 16376.0), Some(736.0));
        assert_eq!(spec_bandwidth("NVIDIA GeForce RTX 4080", 16376.0), Some(716.8));
    }

    #[test]
    fn the_compare_command_fails_a_synthetic_slowdown_and_passes_the_same_result() {
        let cell = |scale: f64| {
            json!({"prompt": "short", "conc": 1, "errors": 0, "decode_tps_per_user": runs(60.0 * scale, 0.004, 5), "agg_tps": runs(60.0 * scale, 0.004, 5),
            "prefill_tps": runs(5000.0 * scale, 0.004, 5), "ttft_s": runs(0.02 / scale, 0.004, 5)})
        };
        let file = |n: &str, scale: f64| {
            let p = std::env::temp_dir().join(format!("glyd-bench-test-{}-{n}.json", std::process::id()));
            let v = json!({"schema": SCHEMA, "machine": {"gpu": "g"}, "model": {"id": "m"}, "contention": {"busy_before": false}, "cells": [cell(scale)]});
            std::fs::write(&p, v.to_string()).unwrap();
            p.to_string_lossy().to_string()
        };
        let (base, same) = (file("base", 1.0), file("same", 1.0));
        let args = |extra: &[&str]| Args::from(["bench", "--compare", &base, &same].iter().chain(extra).map(|s| s.to_string()));
        // (--compare takes no value: the two files are the positionals)
        assert_eq!(compare(&args(&[])).unwrap(), 0);
        assert_eq!(compare(&args(&["--synthetic", "0.95"])).unwrap(), 1);
        assert_eq!(compare(&args(&["--synthetic", "0.999"])).unwrap(), 0);
        for n in ["base", "same"] {
            std::fs::remove_file(std::env::temp_dir().join(format!("glyd-bench-test-{}-{n}.json", std::process::id()))).ok();
        }
    }
}
