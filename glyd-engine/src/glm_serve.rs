//! FROZEN (R13, #234): see src/model_glm.rs; the dynamic route does not tier GLM's experts through the codes pack yet, nor draft with MTP.
//! `glyd-engine glmserve CKPT PACK`: GLM-5.3-Flash with its routed experts from a glyd-hbr-codes/2 directory (src/moe_glm_cb.rs), loaded once, then one request a line on the standard input: `{"ids":[...], "max_new":N,
//! "stop":[...]}`, one answer a line on the standard output: `{"ids":[...], "prefill_ms":x, "decode_ms":y, "token_ms":[...]}` (greedy decoding; the model's own end-of-sequence tokens stop it too; the stop token is in the
//! answer). Flags: `--layout i5|i6|i4` (the always-on matrices, packed from the checkpoint as it loads; default i5), `--cache-gb G` (the GPU memory for the experts in places, in all; default 7), `--chunk N` (prompt tokens a
//! pass, default 1024; 2048 where GLYD_CB2_PREFILL=1 turns the streaming prefill on), `--spec` (room for drafts verified in one pass: a request with "spec": K runs rounds of K guesses, for the development checks), `--ctx N` (positions a sequence may reach, default 33000), `--dump DIR` (the first request's layers' streams and final hidden state are written there: DIR/layer{L}.bin, bf16 [rows, 4, 4096], and
//! DIR/final_hidden.bin, bf16 [rows, 4096]; the request must fit one chunk).

use crate::args::Args;
use crate::ffi::{err, Error, Result};
use crate::model_glm::{EngineGlm, OptsGlm};
use crate::moe_glm_cb::{factory, CbPack, CbShared, BOOKS, CODES};
use crate::offload::{ExpertStore, Mode, Offload};
use crate::weights::Layout;
use glyd_gpu::json::{self, Value};
use std::io::{BufRead, Write};
use std::path::Path;
use std::time::Instant;

fn stderr_line(s: &str) {
    eprintln!("[{}] {s}", chrono_free_clock());
}

/// A clock for the log lines: seconds since the process started.
fn chrono_free_clock() -> String {
    static START: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();
    format!("{:7.1} s", START.get_or_init(Instant::now).elapsed().as_secs_f64())
}

pub fn serve(snap: &Path, pack_dir: &Path, a: &Args) -> Result<()> {
    chrono_free_clock();
    let layout = a.layout()?.unwrap_or(Layout::Int5);
    let gb: f64 = a.num("cache-gb", 7.0f64)?;
    let (chunk, ctx): (usize, usize) = (a.num("chunk", if crate::moe_glm_cb::stream_default() { 2048usize } else { 1024 })?, a.num("ctx", 33000usize)?);
    let dump = a.flag("dump").map(std::path::PathBuf::from);
    // (the experts are read from the page cache and the engine packs the always-on matrices in memory: refuse to start where the computer has less than the headroom to spare)
    let need_gib: f64 = a.num("min-free-gb", 8.0f64)?;
    // (offload::host_memory: MemAvailable, or less where a cgroup's limit is lower, the planner's own probe)
    let avail_gib = crate::offload::host_memory().map(|b| b as f64 / (1u64 << 30) as f64);
    match avail_gib {
        Some(g) if g < need_gib => return err(format!("MemAvailable is {g:.1} GiB, where {need_gib:.1} GiB are wanted free to start (--min-free-gb)")),
        Some(g) => stderr_line(&format!("MemAvailable {g:.1} GiB")),
        None => {}
    }
    let pack = CbPack::open(pack_dir)?;
    // (the pack's files are asked to be read ahead as far as the computer's memory holds them beside the process: a pack bigger than that is read as the experts are asked for, and the page cache keeps what it can)
    let (pack_gib, limit_gib) = (pack.file_bytes() as f64 / (1u64 << 30) as f64, cgroup_limit_gib());
    let room_gib = avail_gib.unwrap_or(f64::MAX).min(limit_gib.unwrap_or(f64::MAX)) - a.num("headroom-gb", 8.0f64)?;
    let ahead = if a.has("no-read-ahead") { 0.0 } else { room_gib.max(0.0).min(pack_gib) };
    stderr_line(&format!("the pack's files: {pack_gib:.1} GiB; the computer's memory for them (MemAvailable{} less the headroom): {:.1} GiB; read ahead at the start: {ahead:.1} GiB", limit_gib.map_or(String::new(), |l| format!(" and the control group's limit {l:.1} GiB")), room_gib.max(0.0)));
    // (the layers of the model; the multi-token-prediction layer after them only where it is asked for: `--mtp`)
    let model_layers = crate::glm::ConfigGlm::load(snap)?.layers;
    let layers: Vec<usize> = (0..64).filter(|&l| pack.has(l) && (l < model_layers || a.has("mtp"))).collect();
    let first = *layers.first().ok_or_else(|| Error(format!("{}: no layers", pack_dir.display())))?;
    let kept = pack.kept(first).map_or(0, <[usize]>::len);
    // (what the GPU keeps beside the cache: the always-on matrices at the layout, the caches and scratch of a chunk and a context, the library's workspace; the cache takes the rest, at most what was asked)
    let gib = (1u64 << 30) as f64;
    let free_gib = crate::dev::Dev::new(0).and_then(|d| d.mem_info()).map(|(f, _)| f as f64 / gib).unwrap_or(15.0);
    let always = match layout {
        Layout::Int6 => 7.3,
        Layout::Int4 => 5.3,
        _ => 6.3,
    };
    let reserve = always + 0.3 + 0.6 + 0.14 + chunk as f64 * 0.0007 + ctx as f64 * 20e-6 + 0.2 + 0.45;
    let fit_gb = ((free_gib - reserve) * gib / 1e9).max(0.5);
    let gb = if gb > fit_gb {
        stderr_line(&format!("--cache-gb {gb:.1} does not fit {free_gib:.1} GiB free beside {reserve:.1} GiB of weights, caches and scratch: {fit_gb:.1} GB"));
        fit_gb
    } else {
        gb
    };
    let place_bytes = layers.iter().map(|&l| pack.place_bytes(l)).max().unwrap_or(CODES + BOOKS);
    let slots = (((gb * 1e9) / (layers.len() as f64 * place_bytes as f64)) as usize).clamp(4, kept.max(4));
    stderr_line(&format!(
        "{} layers of experts in {}, {kept} kept a layer; {slots} places a layer on the GPU ({:.2} GB)",
        layers.len(),
        pack_dir.display(),
        (slots * layers.len() * place_bytes) as f64 / 1e9
    ));
    let store = ExpertStore::new();
    // (the host computes the misses: `--mode copy` copies them all instead; `--cpu-threads N` its crew, 16 pinned by default)
    let mode = a.flag("mode").map_or(Some(Mode::Auto), Mode::parse).ok_or_else(|| Error("--mode: copy, cpu, split or auto".into()))?;
    let od = Offload::default();
    store.set_offload(Offload {
        mode,
        threads: a.num("cpu-threads", 16usize)?.max(1),
        pin: !a.has("no-pin"),
        window: a.num("window", od.window)?,
        promote: a.num("promote", od.promote)?,
        skip: a.num("skip", 0.0f32)?,
        async_copies: a.has("async-copies"),
        ..od
    });
    let common = CbShared::new();
    common.read_ahead.set((ahead * (1u64 << 30) as f64) as u64);
    let make = factory(&pack, slots, 0, &store, &common);
    common.topk.set(a.num("topk", 0usize)?);
    let make_mla = crate::mla_glm::factory();
    let o =
        OptsGlm { ctx, chunk, graph: false, layout: Some(layout), chunked: true, spec: a.has("spec") || a.has("mtp"), mtp: a.has("mtp"), ..Default::default() };
    let t0 = Instant::now();
    let mut e = EngineGlm::new(snap, &o, Some(&make), Some(&make_mla))?;
    stderr_line(&format!("loaded in {:.1} s: {}", t0.elapsed().as_secs_f64(), e.describe()));
    if a.has("inventory") {
        inventory(&e);
    }
    e.segmented.set(a.has("segmented"));
    e.skinny_on.set(a.has("skinny"));
    let eos: Vec<u32> = e.w.cfg.eos.iter().map(|&x| x as u32).collect();
    if a.has("lookup") {
        return err("--lookup was taken out (drafting from the text before); --mtp drafts from the model's own head");
    }
    let mut drafting = crate::draft_loop::Drafting::new();
    let mut first_request = true;
    let stdin = std::io::stdin();
    let mut out = std::io::stdout();
    for line in stdin.lock().lines() {
        let line = line.map_err(|x| Error(x.to_string()))?;
        if line.trim().is_empty() {
            continue;
        }
        // (a request that fails is answered with its error and the session goes on)
        let reply = one_request(&mut e, &store, &common, &mut drafting, &line, &eos, dump.as_deref(), &mut first_request).unwrap_or_else(|x| {
            stderr_line(&format!("a request failed: {}", x.msg));
            format!("{{\"error\":{}}}", json_string(&x.msg))
        });
        writeln!(out, "{reply}").map_err(|x| Error(x.to_string()))?;
        out.flush().map_err(|x| Error(x.to_string()))?;
    }
    Ok(())
}

/// The `k` most probable entries of the logits as (token, probability) by the softmax over all of them.
fn top_of(logits: &[f32], k: usize) -> Vec<(u32, f32)> {
    let m = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let z: f64 = logits.iter().map(|&x| ((x - m) as f64).exp()).sum();
    let mut idx: Vec<u32> = (0..logits.len() as u32).collect();
    let k = k.min(idx.len());
    idx.select_nth_unstable_by(k.saturating_sub(1), |&a, &b| logits[b as usize].total_cmp(&logits[a as usize]));
    let mut best: Vec<u32> = idx[..k].to_vec();
    best.sort_by(|&a, &b| logits[b as usize].total_cmp(&logits[a as usize]));
    best.into_iter().map(|i| (i, (((logits[i as usize] - m) as f64).exp() / z) as f32)).collect()
}

/// The engine's packed matrices by shape, with how many there are and their layout (`--inventory`), and the ones it holds as they are.
fn inventory(e: &EngineGlm) {
    use crate::kern35::Mat;
    let mut packed: std::collections::BTreeMap<(i64, i64, bool), usize> = Default::default();
    let mut dense: std::collections::BTreeMap<(usize, usize), usize> = Default::default();
    for ly in e.w.layers.iter().flatten() {
        for m in &ly.mats {
            match m {
                Mat::Packed(p) => *packed.entry((p.pack.rows, p.pack.cols, p.pack.is_twelve())).or_default() += 1,
                Mat::Dense(_, r, c) => *dense.entry((*r, *c)).or_default() += 1,
            }
        }
        for sh in ly.mla.iter().flat_map(|b| b.pack_shapes()).chain(ly.moe.iter().flat_map(|b| b.pack_shapes())) {
            *packed.entry(sh).or_default() += 1;
        }
    }
    if let Some(h) = e.w.lm_head() {
        *packed.entry((h.pack.rows, h.pack.cols, h.pack.is_twelve())).or_default() += 1;
    }
    stderr_line(&format!("packed matrices (layout {:?}; rows x columns, 12-bit tiers or not, how many):", e.w.layout));
    for ((r, c, tw), n) in &packed {
        stderr_line(&format!("   {r:>6} x {c:>6}  {}  x {n}", if *tw { "twelve" } else { "tiered" }));
    }
    for ((r, c), n) in &dense {
        stderr_line(&format!("   {r:>6} x {c:>6}  bf16 as it is  x {n}"));
    }
}

/// The traced steps of a request (all but the first, which finds the cache as the prompt left it, where there are others): the mean of each layer's parts in microseconds on the stream's clock, the totals by part, and with
/// every stage marker the stages' totals.
pub(crate) fn print_trace(steps: &[(Vec<crate::trace::Layer>, Vec<(&'static str, f64)>, f64)]) {
    let used = if steps.len() > 1 { &steps[1..] } else { steps };
    let n = used.len() as f64;
    let layers = used.iter().map(|s| s.0.len()).min().unwrap_or(0);
    let mean = |f: &dyn Fn(&crate::trace::Layer) -> f64, l: usize| used.iter().map(|s| f(&s.0[l])).sum::<f64>() / n;
    let step_ms = used.iter().map(|s| s.2).sum::<f64>() / n;
    let mut tot = [0f64; 11];
    let mut lines = vec![format!(
        "trace: {} decode steps ({} of them averaged), a step {step_ms:.1} ms; per layer, microseconds on the stream's clock",
        steps.len(),
        used.len()
    )];
    lines.push("  layer   always  router readback     plan products hostexp      sum     post |    total | host: queue   wait".into());
    for l in 0..layers {
        let f: [&dyn Fn(&crate::trace::Layer) -> f64; 11] = [
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
    for l in lines {
        stderr_line(&l);
    }
}

/// The memory limit of the process's control group in GiB (cgroup v2's memory.max; None: none or not known). A scope with a limit holds the page cache the process fills too.
fn cgroup_limit_gib() -> Option<f64> {
    let path = std::fs::read_to_string("/proc/self/cgroup").ok()?.lines().find_map(|l| l.strip_prefix("0::").map(String::from))?;
    let v = std::fs::read_to_string(format!("/sys/fs/cgroup{}/memory.max", path.trim())).ok()?;
    v.trim().parse::<f64>().ok().map(|b| b / (1u64 << 30) as f64)
}

/// Bytes this process has had read from storage so far (/proc/self/io read_bytes: the page faults of a mapped file and its read-ahead count; a read served from the page cache does not).
fn read_bytes() -> u64 {
    std::fs::read_to_string("/proc/self/io").ok().and_then(|t| t.lines().find_map(|l| l.strip_prefix("read_bytes:")?.trim().parse().ok())).unwrap_or(0)
}

fn json_string(s: &str) -> String {
    let mut o = String::from("\"");
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            c if (c as u32) < 32 => o.push(' '),
            c => o.push(c),
        }
    }
    o.push('"');
    o
}

/// One request line: the prompt through the model, then greedy decoding to `max_new` tokens or a stop token; the reply line.
#[allow(clippy::too_many_arguments)]
fn one_request(
    e: &mut EngineGlm,
    store: &std::rc::Rc<ExpertStore>,
    common: &std::rc::Rc<CbShared>,
    draft: &mut crate::draft_loop::Drafting,
    line: &str,
    eos: &[u32],
    dump: Option<&Path>,
    first_request: &mut bool,
) -> Result<String> {
    let v = json::parse(line).map_err(|x| Error(format!("a request: {x}")))?;
    // a setting of the session between requests: {"set": {"mode": "auto", "promote": 1, "window": 16, "cpu_threads": 16, "topk": 6}}
    if let Some(set) = v.get("set") {
        let mut o = store.offload();
        if let Some(m) = set.get("mode").and_then(Value::as_str) {
            o.mode = Mode::parse(m).ok_or_else(|| Error(format!("mode {m}: copy, cpu, split or auto")))?;
        }
        if let Some(x) = set.get("promote").and_then(Value::as_u64) {
            o.promote = x as usize;
        }
        if let Some(x) = set.get("window").and_then(Value::as_u64) {
            o.window = x;
        }
        if let Some(x) = set.get("cpu_threads").and_then(Value::as_u64) {
            o.threads = (x as usize).max(1);
        }
        if let Some(Value::Bool(b)) = set.get("async") {
            o.async_copies = *b;
        }
        if let Some(Value::Bool(b)) = set.get("skinny") {
            e.skinny_on.set(*b);
            e.forget_segments();
        }
        if let Some(x) = set.get("spin_us").and_then(Value::as_u64) {
            crate::cpu_experts::set_spin_us(x);
        }
        if let Some(Value::Bool(b)) = set.get("segmented") {
            e.segmented.set(*b);
        }
        e.forget_segments();
        if let Some(Value::Bool(true)) = set.get("lookup") {
            return err("drafting from the text before was taken out; use \"mtp\": K");
        }
        if let Some(x) = set.get("skip").and_then(|v| match v {
            Value::Number(n) => n.parse::<f32>().ok(),
            _ => None,
        }) {
            o.skip = x.max(0.0);
        }
        store.set_offload(o);
        if let Some(x) = set.get("topk").and_then(Value::as_u64) {
            common.topk.set(x as usize);
        }
        if let Some(Value::Bool(b)) = set.get("stream") {
            common.stream.set(*b);
        }
        if let Some(x) = set.get("trace").and_then(Value::as_u64) {
            common.trace.set(x.min(2) as u8);
        }
        return Ok(format!("{{\"ok\":true,\"mode\":\"{}\",\"promote\":{},\"window\":{},\"cpu_threads\":{},\"skip\":{},\"topk\":{},\"stream\":{},\"async\":{},\"lookup\":{},\"segmented\":{}}}", o.mode.name(), o.promote, o.window, o.threads, o.skip, common.topk.get(), common.stream.get(), o.async_copies, draft.on, e.segmented.get()));
    }
    let ids: Vec<u32> = v.get("ids").and_then(Value::as_u64s).ok_or_else(|| Error("a request with no ids".into()))?.into_iter().map(|x| x as u32).collect();
    let max_new = v.get("max_new").and_then(Value::as_u64).unwrap_or(1) as usize;
    let mut stop: Vec<u32> = v.get("stop").and_then(Value::as_u64s).unwrap_or_default().into_iter().map(|x| x as u32).collect();
    stop.extend(eos);
    if ids.is_empty() || ids.len() + max_new + 1 > e.ctx() {
        return err(format!("{} prompt tokens and {max_new} to make, where a sequence may reach {}", ids.len(), e.ctx()));
    }
    e.reset()?;
    e.trace = None;
    // drafts from the model's multi-token-prediction head (`--mtp`): {"ids": [...], "max_new": N, "mtp": K}
    if let Some(k) = v.get("mtp").and_then(Value::as_u64) {
        return crate::draft_loop::mtp_request(e, store, &ids, max_new, k as usize, &stop);
    }
    // a development check of the draft path: {"ids": [...], "max_new": N, "spec": K}: rounds of K guesses (the last token repeated, mostly wrong) verified in one pass each and settled (`--spec` makes room for them)
    if let Some(k) = v.get("spec").and_then(Value::as_u64) {
        let (k, t1) = (k as usize, Instant::now());
        let (out, st) = e.generate_spec(&ids, max_new.max(1), &mut |all: &[u32]| vec![all[all.len() - 1]; k])?;
        let list = out.iter().map(|x| x.token.to_string()).collect::<Vec<_>>().join(",");
        return Ok(format!(
            "{{\"ids\":[{list}],\"passes\":{},\"drafted\":{},\"accepted\":{},\"ms\":{:.1}}}",
            st.passes,
            st.drafted,
            st.accepted,
            t1.elapsed().as_secs_f64() * 1e3
        ));
    }
    let dumping = *first_request && dump.is_some();
    if let (true, Some(dir)) = (dumping, dump) {
        if ids.len() > e.chunk() {
            return err(format!("--dump: the first request has {} tokens, where a chunk is {}", ids.len(), e.chunk()));
        }
        std::fs::create_dir_all(dir).map_err(|x| Error(format!("{}: {x}", dir.display())))?;
        let dir = dir.to_path_buf();
        let hid = e.w.cfg.hidden;
        e.trace = Some(Box::new(move |e: &EngineGlm, l: usize, n: usize, _| {
            let mut v = vec![0u16; n * 4 * hid];
            if e.dev.sync().is_ok() && e.x.read(0, &mut v).is_ok() {
                let bytes: Vec<u8> = v.iter().flat_map(|w| w.to_le_bytes()).collect();
                if let Err(x) = std::fs::write(dir.join(format!("layer{l}.bin")), bytes) {
                    eprintln!("--dump: layer {l}: {x}");
                }
            }
        }));
    }
    let (t1, io0) = (Instant::now(), read_bytes());
    let s = e.prefill(&ids)?;
    let prefill_ms = t1.elapsed().as_secs_f64() * 1e3;
    // `"top": K`: the K most probable tokens after the prompt with their probabilities (softmax of the logits the prefill ended in), in the reply
    let top: Option<Vec<(u32, f32)>> = match v.get("top").and_then(Value::as_u64) {
        Some(kk) if kk > 0 => Some(top_of(&e.logits()?, kk as usize)),
        _ => None,
    };
    let (io1, prefill_io) = (read_bytes(), read_bytes() - io0);
    if let (true, Some(dir)) = (dumping, dump) {
        let h = e.final_hidden_rows(ids.len())?;
        std::fs::write(dir.join("final_hidden.bin"), h.iter().flat_map(|w| w.to_le_bytes()).collect::<Vec<u8>>()).map_err(|x| Error(x.to_string()))?;
        e.trace = None;
    }
    *first_request = false;
    let st0 = store.stats().unwrap_or_default();
    let t2 = Instant::now();
    let level = common.trace.get();
    let dec = crate::draft_loop::decode(e, draft, level, &ids, s, max_new, &stop)?;
    let (made, times, traced) = (dec.made, dec.times, dec.traced);
    if !traced.is_empty() {
        print_trace(&traced);
    }
    let decode_ms = t2.elapsed().as_secs_f64() * 1e3;
    let decode_io = read_bytes() - io1;
    let st1 = store.stats().unwrap_or_default();
    let steps = made.len().saturating_sub(1).max(1) as f64;
    stderr_line(&format!(
        "{} prompt tokens in {prefill_ms:.0} ms; decode, a token: {:.1} ms; the cache: {:.1} expert reads, {:.1}% found in places, {:.2} pairs on the host in {:.2} ms ({:.2} left out), {:.2} copied ({:.2} ms of copying), the host waited {:.2} ms for the GPU; {:.2} promoted; read from storage: {:.0} MB in the prompt, {:.1} MB a token ({:.2} GB/s of the decode)",
        ids.len(),
        decode_ms / steps,
        (st1.reads - st0.reads) as f64 / steps,
        100.0 * (st1.hits - st0.hits) as f64 / ((st1.reads - st0.reads) as f64).max(1.0),
        (st1.cpu_pairs - st0.cpu_pairs) as f64 / steps,
        (st1.cpu_ms - st0.cpu_ms) / steps,
        (st1.skipped - st0.skipped) as f64 / steps,
        (st1.copied - st0.copied) as f64 / steps,
        (st1.copy_ms - st0.copy_ms) / steps,
        (st1.wait_ms - st0.wait_ms) / steps,
        (st1.promoted - st0.promoted) as f64 / steps,
        prefill_io as f64 / 1e6,
        decode_io as f64 / 1e6 / steps,
        decode_io as f64 / 1e9 / (decode_ms / 1e3).max(1e-9)
    ));
    let list = |v: &[String]| v.join(",");
    let top_json = top.map_or(String::new(), |t| format!(",\"top\":[{}]", t.iter().map(|(i, p)| format!("[{i},{p:.6}]")).collect::<Vec<_>>().join(",")));
    Ok(format!(
        "{{\"ids\":[{}],\"prefill_ms\":{prefill_ms:.1},\"decode_ms\":{decode_ms:.1},\"token_ms\":[{}]{top_json}}}",
        list(&made.iter().map(|x| x.to_string()).collect::<Vec<_>>()),
        list(&times.iter().map(|x| format!("{x:.2}")).collect::<Vec<_>>())
    ))
}

/// `glmcbtest PACK TESTVEC`: the kernels of kernels/glm_cb.cu on the weights work's test vector (an expert of layer 3 and a real input of the layer: x, x' = R x, gate and up, the activation, the expert's output in the rotated
/// basis and in the plain one, all fp32 from the file's dequantised weights) against what they printed: each stage's largest difference from the file's as a share of the file's largest value.
pub fn cb_test(pack_dir: &Path, vec: &Path) -> Result<()> {
    use crate::dev::Dev;
    use crate::moe_glm_cb::KernCb;
    let pack = CbPack::open(pack_dir)?;
    let tv = glyd_gpu::safetensors::File::open(vec).map_err(|e| Error(format!("{}: {e}", vec.display())))?;
    let get = |name: &str| -> Result<Vec<u8>> {
        let t = tv.get(name).ok_or_else(|| Error(format!("{}: no tensor {name}", vec.display())))?;
        tv.read(t).map_err(|e| Error(format!("{}: {name}: {e}", vec.display())))
    };
    let f32s = |b: Vec<u8>| -> Vec<f32> { b.as_chunks::<4>().0.iter().map(|x| f32::from_le_bytes(*x)).collect() };
    let (layer, expert) = (3usize, 0usize);
    let kept = pack.kept(layer).ok_or_else(|| Error("no layer 3".into()))?;
    let i = kept.iter().position(|&e| e == expert).ok_or_else(|| Error("expert 0 of layer 3 is pruned".into()))?;
    let crate::moe_glm_cb::LayerMap { map, starts: (coff, boff), .. } = pack.map_layer(layer)?;
    let dev = Dev::new(0)?;
    let kern = KernCb::load(&dev)?;
    // SAFETY: the mapping holds the kept experts end to end (map_layer checked it); the slices are inside it.
    let (codes, books) = unsafe {
        (
            std::slice::from_raw_parts((map.ptr as *const u8).add(coff + i * CODES), CODES),
            std::slice::from_raw_parts((map.ptr as *const u8).add(boff + i * BOOKS), BOOKS),
        )
    };
    let place = [crate::dev::Buf::new(CODES)?, crate::dev::Buf::new(BOOKS)?];
    place[0].write(0, codes)?;
    place[1].write(0, books)?;
    let x = get("x")?;
    let sg = get("signs")?;
    let sgb = crate::dev::Buf::new(4096)?;
    sgb.write(0, &sg)?;
    let (xb, xr, gu, part, wts, yb, way) = (
        crate::dev::Buf::new(4096 * 2)?,
        crate::dev::Buf::new(4096 * 4)?,
        crate::dev::Buf::new(4096 * 4)?,
        crate::dev::Buf::new(4 * 4096 * 4)?,
        crate::dev::Buf::new(4)?,
        crate::dev::Buf::new(4096 * 2)?,
        crate::dev::Buf::new(8)?,
    );
    xb.write(0, &x)?;
    wts.write(0, &[1.0f32])?;
    way.write(0, &[0i64])?;
    let s = dev.stream;
    // SAFETY: live buffers of the sizes the kernels read and write; the stream is synchronized before the reads.
    unsafe {
        kern.rot_in(s, xb.ptr, 4096, sgb.ptr, xr.ptr, 0, 1)?;
        kern.up(s, place[0].ptr, CODES as u64, place[1].ptr, BOOKS as u64, way.ptr, xr.ptr, gu.ptr, 1, 1, false)?;
        kern.down(s, place[0].ptr, CODES as u64, place[1].ptr, BOOKS as u64, way.ptr, gu.ptr, part.ptr, 4, 10.0, 1, false)?;
        kern.sum_rot(s, part.ptr, wts.ptr, 0, 0, sgb.ptr, yb.ptr, 1, 4, 1)?;
    }
    dev.sync()?;
    let rd = |b: &crate::dev::Buf, n: usize| -> Result<Vec<f32>> {
        let mut v = vec![0f32; n];
        b.read(0, &mut v)?;
        Ok(v)
    };
    let (xr_g, gu_g) = (rd(&xr, 4096)?, rd(&gu, 4096)?);
    let part_g = rd(&part, 4 * 4096)?;
    let y_rot_g: Vec<f32> = (0..4096).map(|o| (0..4).map(|s| part_g[s * 4096 + o]).sum()).collect();
    let mut yv = vec![0u16; 4096];
    yb.read(0, &mut yv)?;
    let y_g: Vec<f32> = yv.iter().map(|&w| f32::from_bits((w as u32) << 16)).collect();
    let cmp = |name: &str, got: &[f32], want: Vec<f32>| {
        let big = want.iter().fold(0f32, |a, &b| a.max(b.abs()));
        let worst = got.iter().zip(&want).fold(0f32, |a, (g, w)| a.max((g - w).abs()));
        let rms = (got.iter().zip(&want).map(|(g, w)| ((g - w) as f64).powi(2)).sum::<f64>() / want.iter().map(|w| (*w as f64).powi(2)).sum::<f64>()).sqrt();
        println!("{name}: largest difference {:.3e} of the largest value {big:.3e} ({:.2e}), relative norm {rms:.2e}", worst, worst / big.max(1e-30));
    };
    cmp("x' = R x", &xr_g, f32s(get("x_rot")?));
    cmp("gate and up", &gu_g, f32s(get("gate_up")?));
    cmp("y' (rotated basis)", &y_rot_g, f32s(get("y_rot")?));
    cmp("y (plain basis, bf16)", &y_g, f32s(get("y")?));
    Ok(())
}

/// `glmcbblock CKPT PACK LAYER [--tokens N] [--mode copy|cpu|auto] [--cache-places N]`: one layer of the model (its latent attention or Kimi delta attention and its experts from the pack) on N random rows of streams, twice
/// (a cold cache and a warm one), to see that it runs and what comes out: the streams' norm and whether every value is finite.
pub fn cb_block(snap: &Path, pack_dir: &Path, a: &Args) -> Result<()> {
    let layer: usize = a.need(3, "LAYER")?.parse().map_err(|_| Error("LAYER: not a number".into()))?;
    let n: usize = a.num("tokens", 128usize)?;
    let places: usize = a.num("cache-places", 20usize)?;
    let pack = CbPack::open(pack_dir)?;
    let store = ExpertStore::new();
    let mode = a.flag("mode").map_or(Some(Mode::Auto), Mode::parse).ok_or_else(|| Error("--mode: copy, cpu, split or auto".into()))?;
    store.set_offload(Offload { mode, threads: a.num("cpu-threads", 16usize)?.max(1), pin: !a.has("no-pin"), ..Offload::default() });
    let common = CbShared::new();
    let make = factory(&pack, places, 0, &store, &common);
    common.topk.set(a.num("topk", 0usize)?);
    let make_mla = crate::mla_glm::factory();
    let o = OptsGlm { ctx: n + 64, chunk: n, graph: false, layout: Some(Layout::Int5), chunked: true, only: Some(vec![layer]), ..Default::default() };
    let e = EngineGlm::new(snap, &o, Some(&make), Some(&make_mla))?;
    let hid = e.w.cfg.hidden;
    let ly = e.w.layers[layer].as_ref().ok_or_else(|| Error("the layer is not loaded".into()))?;
    let mut rng = 0x9E37_79B9_7F4A_7C15u64;
    let mut next = || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        ((rng >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 2.0
    };
    let x: Vec<u16> = (0..n * 4 * hid).map(|_| crate::glm_cb::f32_bf16(next())).collect();
    for pass in 0..2 {
        e.x.write(0, &x)?;
        e.pos.write(0, &(0..n as i32).collect::<Vec<_>>())?;
        let t = Instant::now();
        // SAFETY: live buffers of the engine, sized for a chunk of n rows; the stream orders every step.
        unsafe {
            e.reset_for_block()?;
            e.mixer_half(ly, n, e.pos.ptr, 0)?;
            e.mlp_half(ly, n)?;
        }
        e.dev.sync()?;
        let mut out = vec![0u16; n * 4 * hid];
        e.x.read(0, &mut out)?;
        let f: Vec<f32> = out.iter().map(|&w| f32::from_bits((w as u32) << 16)).collect();
        let finite = f.iter().filter(|v| v.is_finite()).count();
        let norm = (f.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / f.len() as f64).sqrt();
        println!(
            "pass {pass}: {n} rows in {:.1} ms; {finite} of {} values finite, rms {norm:.4}; {}",
            t.elapsed().as_secs_f64() * 1e3,
            f.len(),
            store.report().unwrap_or_default().replace('\n', " | ")
        );
    }
    Ok(())
}
