//! `glyd-engine qwenserve CKPT PACK`: Qwen3.8-Flash-Next with its routed experts from a glyd-hbr-codes/2 directory (src/moe_qwen4x.rs), loaded once, then one request a line on the standard input, one answer a line on the standard
//! output, glmserve's protocol: `{"ids":[...], "max_new":N, "stop":[...], "top":K}` -> `{"ids":[...], "prefill_ms":x, "decode_ms":y, "token_ms":[...], "top":[[id,p],...]}` (greedy; the model's end tokens stop it too, and the
//! stop token is in the answer); `{"set": {"mode": "auto", "cpu_threads": 16, "skip": 0.3, "async": true, "topk": 8}}` changes the session's settings. Flags: `--layout i6|i5|i4` (the always-on matrices, packed as they load;
//! default i6), `--cache-gb G` (the GPU memory for the experts' places; default: what fits), `--chunk N` (prompt tokens a pass, default 1024), `--ctx N` (positions a sequence may reach, default 33000), `--mode`,
//! `--cpu-threads`, `--skip`, `--async-copies` (the expert cache's, as glmserve's). `--gpus 0,1` splits the layers over those GPUs (`--split K`: the second GPU's first layer, by default where the places a layer come
//! out most alike; `--p2p`: the streams GPU to GPU, not through pinned host memory; `--gpus 0,0`: the split's check on one GPU).

use crate::args::Args;
use crate::ffi::{err, Error, Result};
use crate::model_qwen4x::{EngineQx, MtpRun, OptsQx, SeqQx, SPEC_ROWS};
use crate::moe_glm_cb::CbPack;
use crate::moe_qwen4x::{factory, QxCbShared};
use crate::offload::{ExpertStore, Mode, Offload, Stats};
use crate::qwen4x::ConfigQx;
use crate::weights::Layout;
use glyd_gpu::json::{self, Value};
use crate::model::{Finish, Reply};
use crate::sched::{Ev, Submit};
use crate::server::{self, Job, Shared};
use std::collections::VecDeque;
use std::io::{BufRead, Write};
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::path::Path;
use std::rc::Rc;
use std::time::Instant;

thread_local! {
    /// `{"set": {"trace": 1}}` (2: every stage marker): the decode steps' timeline per layer (src/trace.rs).
    static TRACE: std::cell::Cell<u8> = const { std::cell::Cell::new(0) };
    /// `{"set": {"kprof": true}}`: the next plain decode's kernel census.
    static KPROF_NEXT: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

fn log(s: &str) {
    static START: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();
    eprintln!("[{:7.1} s] {s}", START.get_or_init(Instant::now).elapsed().as_secs_f64());
}

fn mem_available_gib() -> Option<f64> {
    // (GLYD_MEMAVAIL_GB: a test's stand-in for a smaller computer's MemAvailable, with a memory cgroup of that size around the process)
    if let Some(g) = crate::tune!("GLYD_MEMAVAIL_GB").and_then(|v| v.parse::<f64>().ok()) {
        return Some(g);
    }
    // (in a container the memory cgroup's limit can be far below the host's MemAvailable, which the container still shows: a Runpod pod of
    // 30 GB showed the host's 62; the cgroup counts the page cache, which it can reclaim, so its room is the limit less what is not cache)
    let cg = cgroup_room_gib();
    let host = mem_info_available_gib();
    match (host, cg) {
        (Some(h), Some(c)) => Some(h.min(c)),
        (h, c) => h.or(c),
    }
}

/// The memory cgroup's room in GiB (v2 `memory.max`, else v1 `memory.limit_in_bytes`, less the usage that is not page cache), if it has a limit.
fn cgroup_room_gib() -> Option<f64> {
    let num = |p: &str| std::fs::read_to_string(p).ok()?.trim().parse::<f64>().ok();
    let stat = |p: &str, key: &str| std::fs::read_to_string(p).ok()?.lines().find_map(|l| l.strip_prefix(key)?.trim().parse::<f64>().ok());
    // (v2: this process's cgroup, /proc/self/cgroup's "0::/path", and the nearest of it and its parents with a limit; in a container "/")
    let own = std::fs::read_to_string("/proc/self/cgroup").ok().and_then(|t| t.lines().find_map(|l| l.strip_prefix("0::").map(str::to_string))).unwrap_or_default();
    let v2 = std::iter::successors(Some(Path::new("/sys/fs/cgroup").join(own.trim_start_matches('/'))), |d| d.parent().filter(|p| p.starts_with("/sys/fs/cgroup")).map(Path::to_path_buf))
        .find_map(|d| num(&d.join("memory.max").to_string_lossy()).map(|m| (m, d)));
    let (limit, used, cache) = if let Some((max, d)) = v2 {
        (max, num(&d.join("memory.current").to_string_lossy())?, stat(&d.join("memory.stat").to_string_lossy(), "file ").unwrap_or(0.0))
    } else {
        let max = num("/sys/fs/cgroup/memory/memory.limit_in_bytes")?;
        (max, num("/sys/fs/cgroup/memory/memory.usage_in_bytes")?, stat("/sys/fs/cgroup/memory/memory.stat", "total_cache ").unwrap_or(0.0))
    };
    // (no limit: "max" does not parse; v1's is a huge number)
    (limit < 1e15).then(|| (limit - (used - cache).max(0.0)).max(0.0) / (1u64 << 30) as f64)
}

fn mem_info_available_gib() -> Option<f64> {
    crate::os::meminfo().and_then(|t| t.lines().find_map(|l| l.strip_prefix("MemAvailable:")?.trim().trim_end_matches("kB").trim().parse::<f64>().ok())).map(|kb| kb / (1024.0 * 1024.0))
}

/// The decode's ceiling on this computer: the bytes a pass reads over the rates they can be read at, the GPU's streaming read for the always-on
/// matrices and the experts found in places, the host's for the experts it computes, the two at once. A reply's time against it says how much of
/// the computer the engine uses (the rest is launches, waits and kernels below the memory's rate, not the hardware).
#[derive(Clone, Copy)]
struct Ceiling {
    gpu_gbs: f64,
    ram_gbs: f64,
    /// bytes a pass reads on the GPU beside the experts: the always-on matrices (the weights less the places)
    always: f64,
    /// bytes of an expert read in a place, and of one the host computes (the pack's mean)
    place: f64,
    mean: f64,
}

static CEILING: std::sync::OnceLock<Ceiling> = std::sync::OnceLock::new();

impl Ceiling {
    /// For a decode of `tokens` tokens in `passes` passes and `ms`, with `hits` experts read in places and `host` on the host: the ceiling in
    /// tokens a second, the share of it reached (%), and the GB a pass on the GPU and from memory.
    fn of(&self, tokens: usize, passes: usize, ms: f64, hits: u64, host: u64) -> Option<(f64, f64, f64, f64)> {
        if tokens == 0 || passes == 0 || ms <= 0.0 || self.gpu_gbs <= 0.0 || self.ram_gbs <= 0.0 {
            return None;
        }
        let (gpu, ram) = (passes as f64 * self.always + hits as f64 * self.place, host as f64 * self.mean);
        let floor_ms = (gpu / self.gpu_gbs).max(ram / self.ram_gbs) / 1e6;
        Some((tokens as f64 * 1e3 / floor_ms, 100.0 * floor_ms / ms, gpu / passes as f64 / 1e9, ram / passes as f64 / 1e9))
    }
}

/// The GPU's streaming read in GB/s: `mib` MiB read `reps` times by the read-only kernel (report::bandwidth's, which dev builds alone have).
fn gpu_read_gbs(mib: usize, reps: usize) -> Result<f64> {
    use crate::dev::{Buf, Dev, Timer};
    let dev = Dev::new(0)?;
    let k = crate::kern::Kern::load(&dev)?;
    let bytes = mib << 20;
    let (buf, out) = (Buf::zeroed(bytes)?, Buf::zeroed(4)?);
    let blocks = dev.sms as u32 * 16;
    // SAFETY: live buffers of the sizes given; the timer synchronizes the stream.
    unsafe {
        k.read_stream(dev.stream, buf.ptr, (bytes / 16) as u64, out.ptr, blocks)?;
        dev.sync()?;
        let t = Timer::new()?;
        t.start(dev.stream)?;
        for _ in 0..reps {
            k.read_stream(dev.stream, buf.ptr, (bytes / 16) as u64, out.ptr, blocks)?;
        }
        let ms = t.stop(dev.stream)? as f64;
        Ok(bytes as f64 * reps as f64 / (ms / 1e3) / 1e9)
    }
}

/// The host's streaming read in GB/s with `threads` threads, best of three over 512 MiB (the rate the misses' products read their experts at, at most).
fn ram_read_gbs(threads: usize) -> f64 {
    let buf = vec![1u64; 64 << 20];
    let each = buf.len().div_ceil(threads.max(1));
    (0..3)
        .map(|_| {
            let t = Instant::now();
            let s = std::thread::scope(|sc| {
                let parts: Vec<_> = buf.chunks(each).map(|c| sc.spawn(move || c.iter().fold(0u64, |a, &x| a.wrapping_add(x)))).collect();
                parts.into_iter().map(|p| p.join().unwrap_or(0)).fold(0u64, u64::wrapping_add)
            });
            std::hint::black_box(s);
            (buf.len() * 8) as f64 / t.elapsed().as_secs_f64() / 1e9
        })
        .fold(0.0, f64::max)
}

/// The expert caches of the model's parts (one a GPU where it is split over GPUs, `--gpus`): their stores and their blocks' shared settings, the first part's first.
pub struct Caches {
    pub stores: Vec<Rc<ExpertStore>>,
    pub commons: Vec<Rc<QxCbShared>>,
}

impl Caches {
    /// The stores' counts added up (the ones the replies' lines read: reads, hits, the host's experts, pairs and time, copies and their time).
    pub fn stats(&self) -> Stats {
        let mut t = Stats::default();
        for s in self.stores.iter().filter_map(|s| s.stats().ok()) {
            (t.reads, t.hits, t.cpu, t.cpu_pairs, t.cpu_ms, t.copied, t.copy_ms) = (t.reads + s.reads, t.hits + s.hits, t.cpu + s.cpu, t.cpu_pairs + s.cpu_pairs, t.cpu_ms + s.cpu_ms, t.copied + s.copied, t.copy_ms + s.copy_ms);
        }
        t
    }
}

/// The parts of a model of `layers` layers split over GPUs (`free`: the GiB free of each part's GPU, a GPU's own shared among the parts on it): each part's first layer after the first part's (`cuts`) as `--split` gives
/// them, else the split that leaves the most places a layer on the part with the fewest (`room(part, layers)`: the places a layer part `part` has room for with that many layers; the first part holds `least` layers at
/// least, the n-gram embedding's being there). None where no split leaves a part a place a layer.
fn split_layers(layers: usize, parts: usize, least: usize, room: &dyn Fn(usize, usize) -> f64) -> Option<Vec<usize>> {
    // (at least `s` places a layer on every part: each part takes as many layers as leave it that, the last part the rest)
    let fit = |s: f64| -> Option<Vec<usize>> {
        let (mut cuts, mut at) = (Vec::new(), 0);
        for p in 0..parts - 1 {
            let k = (1..=layers - at - (parts - 1 - p)).rev().find(|&k| room(p, k) >= s)?;
            if p == 0 && k < least {
                return None;
            }
            at += k;
            cuts.push(at);
        }
        (room(parts - 1, layers - at) >= s).then_some(cuts)
    };
    let (mut lo, mut hi) = (1.0, 1e5);
    fit(lo)?;
    for _ in 0..40 {
        let mid = (lo + hi) / 2.0;
        if fit(mid).is_some() {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    fit(lo)
}

/// The engine with its experts in the cache, as `qwenserve` loads it (and the development commands that run the whole model).
pub fn load(snap: &Path, pack_dir: &Path, a: &Args) -> Result<(EngineQx, Caches)> {
    // (the always-on matrices at four bits by default: gated on the reference's acceptance and GSM8K as i6 was, and a token's bytes fewer)
    let layout = a.layout()?.unwrap_or(Layout::Int4);
    // (the engine's chunk: a decode step's, a verification's and a few-token prompt's rows; a longer prompt makes the chunk it takes for itself, up to 4,096 rows, and frees it after)
    let (chunk, ctx): (usize, usize) = (a.num("chunk", 128usize)?, a.num("ctx", 33536usize)?);
    let avail = mem_available_gib();
    if let Some(g) = avail {
        log(&format!("MemAvailable {g:.1} GiB"));
    }
    let c = ConfigQx::load(snap)?;
    let mut pack = CbPack::open_dims(pack_dir, c.hidden, c.inter)?;
    // (--side DIR[,DIR]: sidecars' copies of some experts in place of the pack's; by default the pack's cb2side directory (every cb3 and cb4 expert at two bits: smaller places, more of them; MMLU-Pro 124/150 as
    // with the pack's own), else its cb3hot (the hottest at three bits), else none; --no-side for none)
    // (--side A,B: several, a later one's copy of an expert over an earlier one's)
    let sides: Vec<std::path::PathBuf> = match a.flag("side") {
        Some(list) => list.split(',').filter(|x| !x.is_empty()).map(std::path::PathBuf::from).collect(),
        // (--precise: the pack's cb3hot (its cb3 and cb4 experts as they are, the hottest at three bits; KL to the bf16 model as the pack's own), fewer places, slower; else the fast mode's cb2side)
        None => (if a.has("precise") { &["hot"][..] } else { &["fast", "hot"][..] }).iter().map(|x| pack_dir.join(x)).find(|d| d.join("meta.json").exists() && !a.has("no-side")).into_iter().collect(),
    };
    for d in sides {
        log(&format!("the GPU copies {}", d.display()));
        pack = pack.with_side(&d)?;
    }
    let layers: Vec<usize> = (0..c.layers).filter(|&l| pack.has(l)).collect();
    if layers.len() != c.layers {
        return err(format!("{}: {} of the model's {} layers of experts", pack_dir.display(), layers.len(), c.layers));
    }
    // (what the GPU keeps beside the places: the always-on matrices at the layout, the caches of the context and the scratch of a chunk; the places take the rest)
    let gib = (1u64 << 30) as f64;
    // (--gpus 0,1: the layers split over those GPUs, each with its layers' matrices, states, caches and places, the streams of each pass going from one to the next through pinned host memory (--p2p: GPU to GPU); a GPU named
    // twice: the split's check on one GPU, each part after the first in a context of its own)
    let gpus: Vec<i32> = a.flag("gpus").unwrap_or("0").split(',').map(|x| x.trim().parse::<i32>()).collect::<std::result::Result<_, _>>().map_err(|_| Error("--gpus: GPU numbers, comma-separated (0,1)".into()))?;
    if gpus.is_empty() || gpus.len() > c.layers / 2 {
        return err("--gpus: one GPU or more, at most one for every two layers");
    }
    let free = if gpus.len() == 1 { crate::dev::Dev::new(gpus[0]).and_then(|d| d.mem_info()).map(|(f, _)| f as f64 / gib).unwrap_or(15.0) } else { 0.0 };
    let frees: Vec<f64> = if gpus.len() > 1 { gpus.iter().map(|&g| crate::dev::Dev::probe(g).map(|p| p.free as f64 / gib / gpus.iter().filter(|&&h| h == g).count() as f64)).collect::<Result<_>>()? } else { Vec::new() };
    // (the rates the decode's ceiling uses, measured before the load takes the GPU's memory: about a second)
    let threads = a.num("cpu-threads", 16usize)?.max(1);
    let (gpu_gbs, ram_gbs) = (gpu_read_gbs(512, 20).unwrap_or(0.0), ram_read_gbs(threads));
    log(&format!("this computer reads {gpu_gbs:.0} GB/s on the GPU and {ram_gbs:.0} GB/s from memory ({threads} threads)"));
    // (--hi i6: the attention layers and the LM head at six bits beside --layout's four)
    let hi = match a.flag("hi") {
        Some("i6") => Some(Layout::Int6),
        Some("i5") => Some(Layout::Int5),
        Some("i4") => Some(Layout::Int4),
        Some("none") => None,
        // (by default with i4 the attention layers and the LM head at five bits: i4 alone moved ref/long's first token to the reference's second, i4 with them at i5 keeps it)
        None => (layout == Layout::Int4).then_some(Layout::Int5),
        Some(x) => return err(format!("--hi {x}: i6, i5, i4 or none")),
    };
    // (the always-on matrices at the layout and what the first calls make: measured at i6 with 0.28 GiB left free after the load, i5 and i4 short of that; i6 0.15 more since the drafts' graphs
    // ran out at 0.11-0.15 GiB free on the 4080 (2026-10-19: a trellis pack's two more places a layer were enough to fail cuGraphInstantiate))
    let always = match layout {
        Layout::Int4 => 2.6,
        Layout::Int5 => 3.05,
        _ => 3.55,
    } + if hi.is_some_and(|h| h != layout) { 0.35 } else { 0.0 };
    let place = layers.iter().map(|&l| pack.place_bytes(l)).max().unwrap_or(1);
    // (a prompt layer by layer, `--whole` rows at most (0: in chunks through all the layers): each row's streams, position, n-gram row and stream weights, and the experts' set; the streaming prefill's ring and buffers)
    // (a prompt layer by layer by default, up to 32,768 rows a part: 32K tokens in under 10 s; --whole 0 for chunks through all the layers)
    let whole = a.num("whole", 32768usize)?.min(ctx);
    let whole_gb = (whole * (c.hc * c.hidden * 2 + 4 + c.ple.as_ref().map_or(8, |p| p.dim) * 2 + c.hc * 4) + crate::moe_qwen4x::whole_bytes(whole, c.top_k)) as f64 / gib;
    // (a prompt's buffers taken layer by layer, and the streaming prefill's, are made for the prompt and freed after it, the last layers' places lent to them where the GPU has not the room: no reserve; a chunk's
    // streamed call (--whole 0) keeps its ring)
    let _ = whole_gb;
    let stream = if !a.has("no-stream") && whole == 0 { crate::cb2_prefill::Cb2Kern::device_bytes(chunk * c.top_k, place) as f64 / gib } else { 0.0 };
    // (measured on the 4080 at chunk 1024, ctx 33000, i6: 1.5 GiB was left free with 1.2 more reserved than this; chunk 128 against 1024: 0.28 GiB more left free after a prompt, so a row of the chunk
    // holds 0.335 MB and the rest of the old 1.2 GB is the room the first prompt and decode use)
    let headroom = a.num("headroom-gb", 0.0f64)?;
    let reserve = always + 0.886 + chunk as f64 * 0.000335 + ctx as f64 * 28e-6 + 0.3 + stream + headroom;
    // (a part's own of it, where the model is split over GPUs: the scratch of a chunk and the room the first prompt and decode use)
    let fixed = 0.886 + chunk as f64 * 0.000335 + 0.3 + stream + headroom;
    // (the MTP head by default, --no-mtp for none)
    let mtp = !a.has("no-mtp") && c.mtp_layers > 0 && pack.has(c.layers);
    let per_gb = |gb: f64, n: usize| ((gb * 1e9) / (n as f64 * place as f64)) as usize;
    // the parts of the model: (GPU, its layers, places a layer, GB of places), one a GPU of --gpus
    let parts: Vec<(i32, (usize, usize), usize, f64)> = if gpus.len() == 1 {
        let fit = (free - reserve).max(0.5) * gib / 1e9;
        let gb: f64 = a.num("cache-gb", fit)?.min(fit);
        let slots = per_gb(gb, layers.len() + mtp as usize);
        log(&format!("the pack: {:.1} GiB in {} layers; the GPU's places: {gb:.2} GB, {slots} a layer of {:.2} MB", pack.file_bytes() as f64 / gib, layers.len(), place as f64 / 1e6));
        vec![(gpus[0], (0, c.layers), slots, gb)]
    } else {
        // (a part's room for places: its GPU's free memory (its share, where parts share a GPU) less the scratch of a part (a context of its own more), its layers' share of the always-on matrices and of the context's
        // caches, and on the last the LM head (at ~6 bits a weight); --cache-gb: the GB of places of all the parts, the same places a layer as on one GPU)
        let head = c.vocab as f64 * c.hidden as f64 * 0.75 / gib;
        let n_all = c.layers as f64;
        let last = gpus.len() - 1;
        let room_gib = |p: usize, n: usize| frees[p] - fixed - if gpus[..p].contains(&gpus[p]) { 0.25 } else { 0.0 } - (always - head + ctx as f64 * 28e-6) * n as f64 / n_all - if p == last { head } else { 0.0 };
        let places = |p: usize, n: usize| room_gib(p, n).max(0.5) * gib / ((n + (p == last && mtp) as usize) as f64 * place as f64);
        let least = c.ple.as_ref().map_or(1, |x| x.layer + 1);
        let cuts: Vec<usize> = match a.flag("split") {
            Some(s) => s.split(',').map(|x| x.trim().parse::<usize>()).collect::<std::result::Result<_, _>>().map_err(|_| Error("--split: the first layer of each GPU's part after the first, comma-separated".into()))?,
            None => split_layers(c.layers, gpus.len(), least, &places).ok_or_else(|| Error(format!("--gpus {}: no split of the {} layers leaves every GPU a place a layer", a.flag("gpus").unwrap_or(""), c.layers)))?,
        };
        let bounds: Vec<usize> = std::iter::once(0).chain(cuts.iter().copied()).chain(std::iter::once(c.layers)).collect();
        if cuts.len() != last || bounds.windows(2).any(|w| w[0] >= w[1]) || bounds[1] < least {
            return err(format!("--split {cuts:?}: {last} first layers in order between {least} and {}", c.layers - 1));
        }
        let all = a.num("cache-gb", 0.0f64)?;
        let parts: Vec<(i32, (usize, usize), usize, f64)> = (0..gpus.len())
            .map(|p| {
                let (lo, hi) = (bounds[p], bounds[p + 1]);
                let held = hi - lo + (p == last && mtp) as usize;
                let fit = (places(p, hi - lo) as usize).min(c.experts);
                let slots = if all > 0.0 { per_gb(all, layers.len() + mtp as usize).min(fit) } else { fit };
                (gpus[p], (lo, hi), slots, (slots * held * place) as f64 / 1e9)
            })
            .collect();
        for &(g, (lo, hi), slots, gb) in &parts {
            log(&format!("GPU {g}: layers {lo}-{}{}, {gb:.2} GB of places, {slots} a layer of {:.2} MB", hi - 1, if hi == c.layers { " and the head" } else { "" }, place as f64 / 1e6));
        }
        parts
    };
    let (gb, slots) = (parts.iter().map(|p| p.3).sum::<f64>(), parts.iter().map(|p| p.2).min().unwrap_or(0));
    crate::monitor::places(gb, slots);
    // (the host's experts for the misses by default, the copies on a thread of their own: --mode, --no-async-copies)
    let mode = a.flag("mode").map_or(Some(Mode::Cpu), Mode::parse).ok_or_else(|| Error("--mode: copy, cpu, split or auto".into()))?;
    let od = Offload::default();
    // (--skip-rank: 1 in precise mode, the most skipping that keeps its pack's numbers against the bf16 model (Strata's set: top-64 KL 0.137, top-1 88.1%, against
    // 0.152 and 87.0% at 3); 0 with best's experts (the folder qwenget --best makes): none skipped, its pack's 0.099 and 89.6% (0.105 at 1, 0.117 at 3); 3 elsewhere)
    let best = pack_dir.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.starts_with("best"));
    let skip_rank = a.num("skip-rank", if a.has("precise") { 1usize } else if best { 0 } else { 3 })?;
    let offload = Offload { mode, threads: a.num("cpu-threads", 16usize)?.max(1), pin: !a.has("no-pin"), skip: a.num("skip", 0.0f32)?, skip_rank, skip_rank_later: a.num("skip-rank-later", 6usize)?, async_copies: !a.has("no-async-copies"), ..od };
    // (--pin-pack on|off|auto: the experts copied into pinned host memory at load, so that the prompt's copies are DMA; auto: where the computer has the pack and 12 GiB more free; off by default: 22 s more load
    // for 0.18 s at 2K and nothing at 8K or 32K, measured)
    let pack_gib = pack.expert_bytes() as f64 / gib;
    let pin = match a.flag("pin-pack").unwrap_or("off") {
        "on" => true,
        "off" => false,
        "auto" => {
            let ok = avail.is_some_and(|g| g >= pack_gib + 12.0);
            if !ok {
                log(&format!("the pack stays mapped, not pinned: {:.1} GiB free, where its {pack_gib:.1} GiB and 12 more are wanted (--pin-pack on to pin anyway)", avail.unwrap_or(0.0)));
            }
            ok
        }
        x => return err(format!("--pin-pack {x}: on, off or auto")),
    };
    if pin {
        log(&format!("the pack's experts pinned in host memory: {pack_gib:.1} GiB"));
    }
    let room = avail.unwrap_or(f64::MAX) - a.num("headroom-gb", 8.0f64)?;
    let t0 = Instant::now();
    let (mut e, mut caches): (Option<EngineQx>, Caches) = (None, Caches { stores: Vec::new(), commons: Vec::new() });
    for (p, &(g, (l0, l1), slots, _)) in parts.iter().enumerate() {
        let (store, common) = (ExpertStore::new(), QxCbShared::new());
        store.set_offload(offload);
        common.pin_pack.set(pin);
        // (the experts of a prompt streamed by default: --no-stream for the cache's per-pair products)
        common.stream.set(!a.has("no-stream"));
        // (the read-ahead's room shared by the parts as their layers)
        common.read_ahead.set((room.max(0.0) * gib * (l1 - l0) as f64 / c.layers as f64) as u64);
        let o = OptsQx { ctx, chunk, layout, chunked: true, mtp, spec: mtp, whole, hi, device: g, span: (parts.len() > 1).then_some((l0, l1)), own_ctx: gpus[..p].contains(&g), peer: a.has("p2p"), ..Default::default() };
        let part = {
            let make = factory(&pack, slots, &store, &common);
            EngineQx::new(snap, &o, Some(&make))?
        };
        e = Some(match e {
            None => part,
            Some(front) => part.with_front(front)?,
        });
        caches.stores.push(store);
        caches.commons.push(common);
    }
    let e = e.ok_or_else(|| Error("no GPU".into()))?;
    for s in &caches.stores[1..] {
        s.share_host_thread(&caches.stores[0])?;
    }
    log(&format!("loaded in {:.1} s: {}", t0.elapsed().as_secs_f64(), e.describe()));
    let mut weights = 0;
    let mut at = Some(&e);
    while let Some(x) = at {
        weights += x.w.bytes;
        at = x.front.as_deref();
    }
    let always = weights.saturating_sub(caches.commons.iter().map(|c| c.places_bytes.get()).sum()) as f64;
    let mean = pack.expert_bytes() as f64 / ((layers.len() + mtp as usize) * c.experts).max(1) as f64;
    let _ = CEILING.set(Ceiling { gpu_gbs, ram_gbs, always, place: place as f64, mean });
    log(&format!("a pass reads {:.2} GB of always-on matrices on the GPU; an expert is {:.2} MB in a place, {:.2} MB on average", always / 1e9, place as f64 / 1e6, mean / 1e6));
    for common in &caches.commons {
        common.fill_page_tables();
    }
    Ok((e, caches))
}

/// The serving settings of a loaded engine (the flags of `qwenserve`).
fn tune(e: &EngineQx, a: &Args) -> Result<()> {
    // (the decode in parts and the skinny products by default: --no-segmented, --no-skinny)
    e.segmented.set(!a.has("no-segmented"));
    // (--draft-vocab N: the MTP head's drafts from the LM head's first N rows and its special tokens' alone, a multiple of 64, 0 for all; 98304 by default: 99.96% of the tokens of the bench's replies are among them)
    e.draft_vocab.set(a.num("draft-vocab", 98304usize)?);
    // (--draft-min-p P: a draft beyond a pass's first kept only while the MTP head's probabilities of the pass's drafts multiplied are at least P; 0: always the request's number; 0.7 by default: drafts of
    // up to 6 at 0.7 made 171 tok/s of speed.py and 6.7 ms a token of the 12 prompts, against 153 and 7.2 for drafts of 1)
    e.draft_min_p.set(a.num("draft-min-p", 0.7f32)?);
    e.skinny_on.set(!a.has("no-skinny"));
    Ok(())
}

pub fn serve(snap: &Path, pack_dir: &Path, a: &Args) -> Result<()> {
    let (mut e, caches) = load(snap, pack_dir, a)?;
    tune(&e, a)?;
    let tower = vision(snap, &e, a)?;
    let eos: Vec<u32> = e.w.cfg.eos.iter().map(|&x| x as u32).collect();
    let stdin = std::io::stdin();
    let mut out = std::io::stdout();
    // (the tokens the engine's state holds after the last request, for the next one's reuse, and the keys of the images among them)
    let mut hist: Vec<u32> = Vec::new();
    let mut held: Vec<u64> = Vec::new();
    let mut last_free = e.dev.mem_info().map_or(0.0, |(f, _)| f as f64 / (1u64 << 30) as f64);
    log(&format!("the GPU's free memory after the load: {last_free:.3} GiB"));
    for line in stdin.lock().lines() {
        let line = line.map_err(|x| Error(x.to_string()))?;
        if line.trim().is_empty() {
            continue;
        }
        #[cfg(feature = "dev")]
        let lock = crate::kcheck_qwen4x::gpu_lock(a)?;  // (`--lock PATH`: the machine's GPU lock held for each request alone)
        #[cfg(not(feature = "dev"))]
        let lock = ();
        let reply = one(&mut e, &caches, &line, &eos, &mut hist, tower.as_ref(), &mut held).unwrap_or_else(|x| {
            hist.clear();
            held.clear();
            log(&format!("a request failed: {}", x.0));
            format!("{{\"error\":{}}}", json_string(&x.0))
        });
        *e.emit.borrow_mut() = None;
        *e.sampler.borrow_mut() = None;
        drop(lock);
        // (the GPU's free memory after each request where it changed: what the requests made beside the load, for the reserve)
        if let Ok((free, _)) = e.dev.mem_info() {
            let gib = free as f64 / (1u64 << 30) as f64;
            if (gib - last_free).abs() > 0.005 {
                log(&format!("the GPU's free memory: {gib:.3} GiB"));
                last_free = gib;
            }
        }
        writeln!(out, "{reply}").map_err(|x| Error(x.to_string()))?;
        out.flush().map_err(|x| Error(x.to_string()))?;
    }
    Ok(())
}

/// `qwenserve CKPT PACK --port P [--host H] [--page DIR] [--name NAME] [--users N]`: the HTTP server (src/server.rs) on the engine: the chat page (DIR's chat.html and web/), the OpenAI API (/v1/chat/completions,
/// /v1/completions), Anthropic's (/v1/messages, /v1/messages/count_tokens), and the Monitor's numbers (/v1/glyd/status). It listens while the model loads (the page says so; requests wait; /health is 503 until it is loaded), then answers up
/// to N requests at once (4 by default, 1 to 8; `Qx`), the others waiting in the order they came: each request beside the first holds its states and caches on the GPU (Qwen3.8-Flash-Next: 116 MB and 30 KB a position of
/// its prompt and max_tokens, 4,096 positions at least: 1.12 GB at 32,768), taken from the experts' places while it runs. Thinking is off where a request does not ask for it; the sampling defaults are the model's, at a
/// temperature of 0.5 at most with min-p 0.1.
/// Images (PNG, JPEG) come as OpenAI image_url parts (a data: URL or an http(s) one) and Anthropic image blocks: read by the model's vision tower (src/vision.rs), `--image-pixels N` the most
/// pixels each is resized to.
pub fn http(snap: &Path, pack_dir: &Path, a: &Args) -> Result<()> {
    let name = a.flag("name").unwrap_or("qwen3.8-flash-next").to_string();
    let ctx = a.num("ctx", 33536usize)?;
    let users = a.num("users", 4usize)?;
    if !(1..=crate::model::MAX_ROWS).contains(&users) {
        return err(format!("--users {users}: 1 to {} requests at once", crate::model::MAX_ROWS));
    }
    let window = ctx - 2;
    // (the mode the Monitor and qwenbench name: best's experts are the folder qwenget --best makes)
    let mode = if a.has("precise") { "precise" } else if pack_dir.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.starts_with("best")) { "best" } else { "fast" };
    crate::monitor::start(mode, ctx);
    server::STANDALONE.store(true, std::sync::atomic::Ordering::Relaxed);
    if let Some(d) = a.flag("page") {
        let _ = server::PAGE_DIR.set(d.into());
    }
    let tok = crate::tok::Tok::load(snap)?;
    let chat = crate::chat::Chat::load(snap)?;
    let mut defaults = crate::sampler::generation_defaults(snap);
    if defaults.temperature > 0.0 {
        // (one-shot games of ~15K tokens in precise mode, browser-tested: the model's own temperature (1.0) ended about one answer in four early; 0.7 with
        // min-p 0.05 made 1 playable in 4; 0.5 with min-p 0.1 made 3 in 6, with no loop or early end; 0.4 with min-p 0.15 looped. A request's own values win.)
        defaults.temperature = defaults.temperature.min(0.5);
        defaults.min_p = 0.1;
    }
    let host = a.flag("host").unwrap_or("127.0.0.1");
    let port: u16 = a.num("port", 8080u16)?;
    let key = std::env::var("GLYD_API_KEY").ok().filter(|k| !k.is_empty());
    let (tx, rx) = std::sync::mpsc::channel();
    // (the address the program chooses is this computer's own: the checks that keep other web pages' scripts away are on; an address the user names is theirs)
    // (/health says "loading" until the model can serve: a script that waits for it does not send its first request into the load)
    let loading = Arc::new(std::sync::atomic::AtomicBool::new(true));
    let addr = server::listen_loading(&format!("{host}:{port}"), &name, window, key, !a.has("host"), tx.clone(), loading.clone())?;
    println!("Qwen3.8-Flash-Next on Glyd: the chat at http://{addr}/, the API at http://{addr}/v1 (OpenAI and Anthropic). Loading the model...");
    let (e, caches) = load(snap, pack_dir, a)?;
    tune(&e, a)?;
    let tower = vision(snap, &e, a)?;
    let vocab = e.w.cfg.vocab;
    let eos: Vec<u32> = e.w.cfg.eos.iter().map(|&x| x as u32).collect();
    let shared = Arc::new(Shared { tok, chat, name, defaults, window, vocab, thinking: Some(false), refused: &[] });
    crate::monitor::ready();
    loading.store(false, Ordering::Relaxed);
    println!("Ready.");
    // (a model split over GPUs answers one request at a time: each part would need its share of a sequence)
    let users = if e.takes_seqs() { users } else { 1 };
    let mut q = Qx::new(e, caches, eos, users, tower);
    // a request's thread starts as it comes (not when the reply before it ends): one that waits for its turn streams its headers, and is skipped where its client has gone. The engine's
    // thread takes the sequences in the order they come, up to `users` at once, a pass each in turn.
    let (seq_tx, seq_rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        while let Ok(job) = rx.recv() {
            match job {
                Job::Seq(_) | Job::Stop => {
                    if seq_tx.send(job).is_err() {
                        return;
                    }
                }
                job => server::detach(job, &shared, &tx),
            }
        }
    });
    loop {
        // (the jobs that have come; where nothing runs or waits, the next one waited for, or the end of the idle slots' wait)
        let mut next = match (q.idle(), q.linger()) {
            (false, _) => seq_rx.try_recv().ok(),
            (true, None) => match seq_rx.recv() {
                Ok(j) => Some(j),
                Err(_) => break,
            },
            (true, Some(d)) => match seq_rx.recv_timeout(d) {
                Ok(j) => Some(j),
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => None,
                Err(_) => break,
            },
        };
        while let Some(job) = next {
            match job {
                Job::Stop => return Ok(()),
                Job::Seq(sub) => q.waiting.push_back(sub),
                _ => {}
            }
            next = seq_rx.try_recv().ok();
        }
        q.round();
    }
    Ok(())
}

/// `qwenget REPO DIR [--precise | --best]`: the model from the Hugging Face Hub (src/hub.rs: large files as ranges on several connections, resumed where they stopped, checked by SHA-256) into the Hub's cache, which
/// every tool on the computer shares, and DIR/model and DIR/pack (with --precise, DIR/precise in its place; with --best, DIR/best) linked to its folders.
pub fn get(repo: &str, dir: &Path, a: &Args) -> Result<()> {
    use crate::hub::{disk_free, Hub, HubError};
    let hub = Hub::from_env();
    let e = |x: HubError| match x {
        HubError::Denied(code) => Error(format!("{repo}: the Hub answered {code} (no such repository, or a private one: set HF_TOKEN)")),
        x => Error(format!("{repo}: {x}")),
    };
    let remote = hub.list(repo, "main").map_err(e)?;
    // (fast mode's experts are fast/ since v0.2.2 (format q); pack/ is the earlier engines' fast. precise-v2/ and best-v2/ (format q at 3 and 4 bits) are
    // this engine's precise and best, precise/ and best/ the earlier engines' (and ours where the repo has no -v2 yet))
    let has = |p: &str| remote.files.iter().any(|f| f.path.starts_with(&format!("{p}/")));
    let experts = if a.has("best") { ["best-v2", "best"] } else if a.has("precise") { ["precise-v2", "precise"] } else { ["fast", "fast"] };
    let parts: &[&str] = &["model", experts.into_iter().find(|p| has(p)).unwrap_or(experts[1])];
    let wanted: Vec<crate::hub::Entry> = remote.files.iter().filter(|f| parts.iter().any(|p| f.path.starts_with(&format!("{p}/")))).cloned().collect();
    for p in parts {
        if !wanted.iter().any(|f| f.path.starts_with(&format!("{p}/"))) {
            return err(format!("{repo} has no {p}/ folder"));
        }
    }
    let need = hub.missing(&remote, &wanted);
    let at = crate::store::existing_ancestor(hub.cache());
    if let Some(free) = disk_free(&at) {
        if free < need + (2 << 30) {
            return err(format!("{:.0} GB more to download and {:.0} GB free in {}: free some space (or set HF_HUB_CACHE to a disk with room)", need as f64 / 1e9, free as f64 / 1e9, at.display()));
        }
    }
    let total: u64 = wanted.iter().map(|f| f.size).sum();
    println!("{repo}: {:.1} GB, {:.1} GB of it to download, into {}", total as f64 / 1e9, need as f64 / 1e9, hub.cache().display());
    let t0 = Instant::now();
    let shown = std::sync::Mutex::new((0.0f64, 0u64));
    let progress = |done: u64, total: u64| {
        let t = t0.elapsed().as_secs_f64();
        let Ok(mut s) = shown.lock() else { return };
        if t - s.0 < 1.0 && done < total {
            return;
        }
        if s.1 == 0 {
            s.1 = done.max(1); // (what was there before this run, not counted in the speed)
        }
        let rate = (done.saturating_sub(s.1)) as f64 / t.max(1e-3);
        let left = if rate > 0.0 { format!(", about {:.0} min left", (total - done) as f64 / rate / 60.0) } else { String::new() };
        print!("\r  {:.1} of {:.1} GB  {:.0} MB/s{left}      ", done as f64 / 1e9, total as f64 / 1e9, rate / 1e6);
        let _ = std::io::stdout().flush();
        s.0 = t;
    };
    let snap = hub.fetch(&remote, &wanted, &progress).map_err(e)?;
    hub.pin(&remote).map_err(e)?;
    println!();
    std::fs::create_dir_all(dir).map_err(|x| Error(format!("{}: {x}", dir.display())))?;
    for p in parts {
        let link = dir.join(p);
        #[cfg(unix)]
        if link.is_symlink() {
            let _ = std::fs::remove_file(&link);
        }
        #[cfg(unix)]
        std::os::unix::fs::symlink(snap.join(p), &link).map_err(|x| Error(format!("{}: {x}", link.display())))?;
        // (Windows: a junction, the folder link it makes without Developer Mode or administrator rights)
        #[cfg(windows)]
        {
            if link.is_symlink() {
                let _ = std::fs::remove_dir(&link);
            }
            let made = std::process::Command::new("cmd").args(["/C", "mklink", "/J"]).arg(&link).arg(snap.join(p)).output().map_err(|x| Error(format!("{}: {x}", link.display())))?;
            if !made.status.success() {
                return err(format!("{}: {}", link.display(), String::from_utf8_lossy(&made.stderr).trim()));
            }
        }
    }
    // (a download counts on the Hub only through its query files (meta.yaml at the root): one small read per install, none when nothing was downloaded)
    if need > 0 {
        let _ = hub.text(repo, "main", "meta.yaml");
    }
    println!("the model is in {} ({})", dir.display(), parts.join(", "));
    Ok(())
}

/// The vision tower of the checkpoint at `snap` where it has one, the images' bounds set for the requests' threads (`--image-pixels N`: the most pixels an image is resized to, 1,048,576
/// by default: up to 1,024 tokens an image; the processor's own top is 16.8 M, 16,384 tokens).
fn vision(snap: &Path, e: &EngineQx, a: &Args) -> Result<Option<crate::vision::Tower>> {
    let pixels = a.num("image-pixels", 1usize << 20)?;
    // (a checkpoint whose tower this engine cannot read is served for text: the reason logged)
    let loaded = crate::vision::Prep::load(snap, Some(pixels)).and_then(|p| Ok(p.zip(crate::vision::Tower::open(snap, &e.dev)?)));
    match loaded {
        Ok(Some((prep, tower))) => {
            log(&format!("images: up to {} pixels each ({} tokens)", prep.max_pixels, prep.max_pixels / 1024));
            let _ = crate::vision::PREP.set(prep);
            Ok(Some(tower))
        }
        Ok(None) => Ok(None),
        Err(x) => {
            log(&format!("images are off: {}", x.0));
            Ok(None)
        }
    }
}

/// A prompt's images into the engine (`EngineQx::set_images`): the first `held` of them the state holds already (where `keep` reuses it), the rest through the tower, the last layers'
/// places lent to it where the GPU has not the room.
fn see(e: &mut EngineQx, tower: Option<&crate::vision::Tower>, prompt: &[u32], imgs: &[Arc<crate::vision::Image>], keep: usize, held: usize) -> Result<()> {
    let token = crate::vision::PREP.get().map_or(u32::MAX, |p| p.image_token);
    let mut list = Vec::with_capacity(imgs.len());
    for (i, img) in imgs.iter().enumerate() {
        let rows = if keep > 0 && i < held {
            None
        } else {
            let tw = tower.ok_or_else(|| Error("this model has no vision tower loaded".into()))?;
            let need = crate::vision::Tower::bytes(img.gh * img.gw);
            let t0 = Instant::now();
            let lent = if e.dev.mem_info()?.0 < need + (64 << 20) { e.borrow_places(need)? } else { Vec::new() };
            let r = tw.run(&e.dev, &e.kern, img);
            e.return_places(&lent)?;
            log(&format!("an image of {} x {} patches, {} tokens: the tower {:.0} ms in {:.0} MB of GPU memory{}", img.gh, img.gw, img.tokens(), t0.elapsed().as_secs_f64() * 1e3, need as f64 / 1e6, if lent.is_empty() { String::new() } else { format!(" (the places of {} layers lent)", lent.len()) }));
            Some(r?)
        };
        list.push((img.grid(), rows));
    }
    e.set_images(prompt, token, list)
}


/// A request being answered: its slot, its generation, its stop tokens, and what its log line needs (`shared`: another request ran beside it, so its times are not its alone).
struct Live {
    slot: usize,
    sub: Submit,
    run: MtpRun,
    stop: Vec<u32>,
    keep: usize,
    t0: Instant,
    st0: crate::offload::Stats,
    shared: bool,
}

/// GPU memory a sequence made for a request takes beside its states and caches: its decode step's graphs, of every count of rows a verification has (the load's reserve has 0.11-0.15 GiB for the first
/// sequence's; a graph that finds no room is not made: `EngineQx::seg`).
const GRAPHS: usize = 160 << 20;

/// How long sequences made for requests are kept after the last of them ended (the next requests take them, with their graphs) before their memory goes back to the experts' places.
const LINGER: std::time::Duration = std::time::Duration::from_secs(3);

/// The engine as the server's backend: up to `slots.len()` requests at once, a pass of each in turn, a request in a slot of its own: slot 0 is the engine's own sequence (a conversation's earlier turns reused
/// there), the others are made for requests (as long as a prompt and max_tokens need) in memory the experts' places give up (`EngineQx::resize_places`), and freed `LINGER` after the last of them ended.
/// The MTP head's drafts, the request's sampling.
struct Qx {
    e: EngineQx,
    caches: Caches,
    eos: Vec<u32>,
    /// The tokens slot 0's state holds after its last reply.
    hist: Vec<u32>,
    /// Each slot's sequence while another is in the engine (`active`'s is the engine's own; None: not made).
    slots: Vec<Option<SeqQx>>,
    active: usize,
    live: Vec<Live>,
    waiting: VecDeque<Submit>,
    /// The GPU's free memory with every place made and slot 0 alone (the room the engine's own work keeps), as it was when a slot was last made from there.
    headroom: usize,
    /// When a request in a slot made for it last ended.
    extra_at: Instant,
    tower: Option<crate::vision::Tower>,
    /// The keys of the images among `hist`'s tokens, in order (slot 0's).
    held: Vec<u64>,
}

impl Qx {
    fn new(e: EngineQx, caches: Caches, eos: Vec<u32>, users: usize, tower: Option<crate::vision::Tower>) -> Qx {
        Qx { e, caches, eos, hist: Vec::new(), tower, held: Vec::new(), slots: (0..users).map(|_| None).collect(), active: 0, live: Vec::new(), waiting: VecDeque::new(), headroom: 0, extra_at: Instant::now() }
    }

    fn idle(&self) -> bool {
        self.live.is_empty() && self.waiting.is_empty()
    }

    fn busy(&self, j: usize) -> bool {
        self.live.iter().any(|l| l.slot == j)
    }

    /// Positions slot `j`'s sequence may reach (0: none made).
    fn cap(&self, j: usize) -> usize {
        if j == self.active {
            self.e.ctx()
        } else {
            self.slots[j].as_ref().map_or(0, SeqQx::ctx)
        }
    }

    /// The slots made for requests that none runs in now.
    fn idle_made(&self) -> Vec<usize> {
        (1..self.slots.len()).filter(|&j| self.cap(j) > 0 && !self.busy(j)).collect()
    }

    /// How long until the idle slots' memory goes back (None: none waits for that).
    fn linger(&self) -> Option<std::time::Duration> {
        let busy = (1..self.slots.len()).any(|j| self.busy(j));
        (!busy && !self.idle_made().is_empty()).then(|| LINGER.saturating_sub(self.extra_at.elapsed()))
    }

    /// Slot `j`'s sequence put in the engine, the one there kept in its slot.
    fn activate(&mut self, j: usize) -> Result<()> {
        if j == self.active {
            return Ok(());
        }
        let mut s = self.slots[j].take().ok_or_else(|| Error(format!("sequence {j} is not made")))?;
        if let Err(x) = self.e.swap_seq(&mut s) {
            self.slots[j] = Some(s);
            return Err(x);
        }
        self.slots[self.active] = Some(s);
        self.active = j;
        Ok(())
    }

    /// A sequence of `ctx` positions made in slot `j`, the places of the experts made fewer where the GPU has not the room for it and the engine's own (half of them kept at least): false where it cannot be had now.
    fn make(&mut self, j: usize, ctx: usize) -> Result<bool> {
        let t = Instant::now();
        let free = self.e.dev.mem_info()?.0;
        let (now, full, row) = self.e.place_slots();
        if now == full && self.active == 0 && self.slots[1..].iter().all(Option::is_none) {
            self.headroom = free;
        }
        let need = self.e.seq_bytes(ctx) + GRAPHS + self.headroom;
        let fewer = need.saturating_sub(free).div_ceil(row.max(1));
        // ponytail: a fixed floor of half the places; a request past it waits for a slot to end
        if fewer > 0 && now < full / 2 + fewer {
            return Ok(false);
        }
        if fewer > 0 {
            self.e.resize_places(now - fewer)?;
        }
        self.slots[j] = Some(self.e.new_seq(ctx)?);
        log(&format!("sequence {j}: {ctx} positions ({:.2} GB), {:.2} GB free before; the experts' places {now} -> {} a layer, {:.2} GB free now ({:.0} ms)", self.e.seq_bytes(ctx) as f64 / 1e9, free as f64 / 1e9, now - fewer, self.e.dev.mem_info()?.0 as f64 / 1e9, t.elapsed().as_secs_f64() * 1e3));
        Ok(true)
    }

    /// The slots made for requests that none runs in now freed, and the experts' places made as many again as the memory allows (to the nearest place: half a place of the engine's room may go).
    fn release_idle(&mut self) -> Result<()> {
        let idle = self.idle_made();
        if idle.is_empty() {
            return Ok(());
        }
        let t = Instant::now();
        if idle.contains(&self.active) {
            self.activate(0)?;
        }
        self.e.dev.sync()?;
        let before = self.e.dev.mem_info()?.0;
        for &j in &idle {
            self.slots[j] = None;
        }
        let (now, full, row) = self.e.place_slots();
        let free = self.e.dev.mem_info()?.0;
        let more = ((free + row / 2).saturating_sub(self.headroom) / row.max(1)).min(full - now);
        if more > 0 {
            self.e.resize_places(now + more)?;
        }
        log(&format!("sequences {idle:?} freed: {:.2} GB free before, {:.2} after; the experts' places {now} -> {} a layer, {:.2} GB free now ({:.0} ms)", before as f64 / 1e9, free as f64 / 1e9, now + more, self.e.dev.mem_info()?.0 as f64 / 1e9, t.elapsed().as_secs_f64() * 1e3));
        Ok(())
    }

    /// A round: the waiting requests whose client left ended, the others taken in while there are slots (each one's prompt read and its first tokens sent), then a pass of each request in turn; the slots
    /// made for requests freed once none has run in them for `LINGER`.
    fn round(&mut self) {
        if self.linger().is_some_and(|d| d.is_zero()) && self.waiting.is_empty() {
            if let Err(x) = self.release_idle() {
                log(&format!("the idle sequences not freed: {}", x.0));
            }
        }
        self.waiting.retain(|s| {
            let gone = s.abort.load(Ordering::Relaxed);
            if gone {
                // (its client left while it waited)
                let _ = s.out.send(Ev::End(Ok(Finish::Stopped)));
            }
            !gone
        });
        // (prompts of up to 4,096 tokens in all a round, one at least: a long one does not hold the others' streams up for more than its own prompt)
        let mut read = 0;
        while self.live.len() < self.slots.len() && self.waiting.front().is_some_and(|s| read == 0 || read + s.prompt.len() <= 4096) {
            let Some(sub) = self.waiting.pop_front() else { break };
            read += sub.prompt.len();
            if let Some(sub) = self.admit(sub) {
                self.waiting.push_front(sub);
                break;
            }
        }
        let mut i = 0;
        while i < self.live.len() {
            let together = self.live.len() > 1;
            let r = match self.activate(self.live[i].slot) {
                Ok(()) => self.e.mtp_pass(&mut self.live[i].run),
                Err(x) => Err(x),
            };
            self.live[i].shared |= together;
            match r {
                Ok(true) => i += 1,
                Ok(false) => {
                    let l = self.live.remove(i);
                    self.finish(l, Ok(()));
                }
                Err(x) => {
                    let l = self.live.remove(i);
                    self.finish(l, Err(x));
                }
            }
        }
    }

    /// A request taken in: slot 0 where it is free, else an idle slot made for an earlier request that is long enough, else one made for it (the idle ones freed first); its prompt read and its first tokens
    /// sent. The request back where no slot can be had now (it waits).
    fn admit(&mut self, sub: Submit) -> Option<Submit> {
        let max_new = sub.reply.max_new.min(self.e.ctx_max().saturating_sub(sub.prompt.len() + 2)).max(1);
        let j = if !self.busy(0) {
            0
        } else {
            // (as long as the prompt and max_tokens need, in steps of 1,024; at least 4,096: past the indexer's budget, as the engine's own)
            let ctx = (sub.prompt.len() + max_new + 2).next_multiple_of(1024).max(4096).min(self.e.ctx_max());
            let free: Vec<usize> = (1..self.slots.len()).filter(|&j| !self.busy(j)).collect();
            match free.iter().copied().find(|&j| self.cap(j) >= ctx) {
                Some(j) => j,
                None => {
                    let Some(&j) = free.first() else { return Some(sub) };
                    match self.release_idle().and_then(|()| self.make(j, ctx)) {
                        Ok(true) => j,
                        Ok(false) => return Some(sub),
                        Err(x) => {
                            log(&format!("a request failed: {}", x.0));
                            let _ = sub.out.send(Ev::End(Err(x.0)));
                            return None;
                        }
                    }
                }
            }
        };
        match self.begin(j, &sub, max_new) {
            Ok((run, stop, keep, t0, st0)) => self.live.push(Live { slot: j, sub, run, stop, keep, t0, st0, shared: false }),
            Err(x) => {
                let l = self.live.len();
                *self.e.emit.borrow_mut() = None;
                *self.e.sampler.borrow_mut() = None;
                if j == 0 {
                    self.hist.clear();
                    self.held.clear();
                }
                crate::monitor::failed(sub.prompt.len(), &x.0);
                log(&format!("a request failed: {} ({l} others running)", x.0));
                let _ = sub.out.send(Ev::End(Err(x.0)));
                self.extra_at = Instant::now();
            }
        }
        None
    }

    /// Slot `j`'s sequence in the engine, the request's sampling and sink set, its prompt read (after the tokens slot 0 holds of it already) and its first tokens sent: its generation, stop tokens, reused tokens
    /// and start.
    fn begin(&mut self, j: usize, sub: &Submit, max_new: usize) -> Result<(MtpRun, Vec<u32>, usize, Instant, crate::offload::Stats)> {
        self.activate(j)?;
        let (e, prompt, o): (&mut EngineQx, &[u32], &Reply) = (&mut self.e, &sub.prompt, &sub.reply);
        if !e.has_mtp() {
            return err("the pack has no MTP head");
        }
        // (slot 0's state reused where the prompt goes on from it and the images it holds are the prompt's first)
        let same = j == 0 && self.held.len() <= sub.images.len() && self.held.iter().zip(&sub.images).all(|(k, i)| *k == i.key);
        let keep = if same && !self.hist.is_empty() && self.hist.len() == e.n && prompt.len() > self.hist.len() && prompt.starts_with(&self.hist) { self.hist.len() } else { 0 };
        let held = if j == 0 {
            self.hist.clear();
            std::mem::take(&mut self.held).len()
        } else {
            0
        };
        see(e, self.tower.as_ref(), prompt, &sub.images, keep, held)?;
        let mut stop = o.stop_ids.clone();
        if !o.ignore_eos {
            stop.extend(&self.eos);
        }
        *e.sampler.borrow_mut() = (!o.params.greedy()).then(|| crate::sampler::Sampler::new(o.params.clone(), prompt));
        crate::monitor::begin(prompt.len(), keep);
        let halt = e.halt.clone();
        halt.set(false);
        let (out, abort, ends) = (sub.out.clone(), sub.abort.clone(), stop.clone());
        *e.emit.borrow_mut() = Some(Box::new(move |ids: &[u32]| {
            for &t in ids {
                // (an end or stop token is not a token of the reply, as the scheduler has it)
                if halt.get() || ends.contains(&t) {
                    return;
                }
                if out.send(Ev::Token(t)).is_err() || abort.load(Ordering::Relaxed) {
                    halt.set(true);
                }
            }
            crate::monitor::wrote(ids.len());
        }));
        let (t0, st0) = (Instant::now(), self.caches.stats());
        e.keep.set(keep);
        let run = e.mtp_begin(prompt, max_new, 6, &stop)?;
        Ok((run, stop, keep, t0, st0))
    }

    /// A request's end (its sequence is the engine's: its last pass just ran): the sink and the sampling cleared, the log line and the Monitor's numbers, how it ended sent; slot 0's tokens kept for the next
    /// request's reuse (another slot waits `LINGER` for one).
    fn finish(&mut self, l: Live, r: Result<()>) {
        if l.slot != 0 {
            self.extra_at = Instant::now();
        }
        let e = &mut self.e;
        *e.emit.borrow_mut() = None;
        *e.sampler.borrow_mut() = None;
        let (prompt, made, st) = (&l.sub.prompt, &l.run.out, &l.run.stats);
        let end = match r {
            Err(x) => {
                if l.slot == 0 {
                    self.hist.clear();
                    self.held.clear();
                }
                crate::monitor::failed(prompt.len(), &x.0);
                log(&format!("a request failed: {}", x.0));
                Err(x.0)
            }
            Ok(()) => {
                // (the tokens the engine's state now holds: the prompt and every token made but the last)
                if l.slot == 0 && !made.is_empty() && e.n == prompt.len() + made.len() - 1 {
                    self.hist.extend_from_slice(prompt);
                    self.hist.extend(made[..made.len() - 1].iter().map(|x| x.token));
                    self.held = l.sub.images.iter().map(|i| i.key).collect();
                }
                let ms = l.t0.elapsed().as_secs_f64() * 1e3;
                let st1 = self.caches.stats();
                let st0 = &l.st0;
                let ms_tok = (ms - st.prompt_ms as f64) / (made.len().max(2) - 1) as f64;
                let hits = 100.0 * (st1.hits - st0.hits) as f64 / ((st1.reads - st0.reads) as f64).max(1.0);
                let per_pass = (made.len().max(1) - 1) as f64 / st.passes.max(1) as f64;
                // (the ceiling of a request that ran alone: the store's counts are of every request's passes)
                let ceil = CEILING.get().filter(|_| !l.shared).and_then(|c| c.of(made.len().saturating_sub(1), st.passes, ms - st.prompt_ms as f64, st1.hits - st0.hits, st1.cpu - st0.cpu));
                if let Some((tok_s, pct, gpu, ram)) = ceil {
                    log(&format!("the ceiling here: {gpu:.2} GB a pass on the GPU, {ram:.2} GB from memory: {tok_s:.0} tokens a second, {pct:.0}% of it reached"));
                }
                let beside = if l.shared { format!(" (sequence {}, beside others)", l.slot) } else { String::new() };
                log(&format!("{} prompt tokens ({} reused), {} made in {ms:.0} ms (the prompt {} ms, then {ms_tok:.1} ms a token): {} of {} guesses right, {per_pass:.2} tokens a pass; the cache: {hits:.1}% in places{beside}", prompt.len(), l.keep, made.len(), st.prompt_ms, st.accepted, st.drafted));
                crate::monitor::end(
                    &format!("\"prompt_tokens\":{},\"made\":{},\"prompt_ms\":{},\"ms_a_token\":{ms_tok:.2},\"guesses_right\":{},\"guesses\":{},\"tokens_a_pass\":{per_pass:.2},\"cache_hits\":{hits:.1},\"ceiling_tok_s\":{:.0},\"of_ceiling\":{:.0}", prompt.len(), made.len(), st.prompt_ms, st.accepted, st.drafted, ceil.map_or(0.0, |c| c.0), ceil.map_or(0.0, |c| c.1)),
                    &format!("\"prompt\":{},\"reused\":{},\"output\":{},\"tok_s\":{:.1},\"prompt_s\":{:.2},\"seconds\":{:.2},\"guesses\":{}", prompt.len(), l.keep, made.len(), 1000.0 / ms_tok.max(1e-3), st.prompt_ms as f64 / 1e3, ms / 1e3, (100 * st.accepted / st.drafted.max(1))),
                );
                Ok(if e.halt.get() {
                    Finish::Stopped
                } else if made.last().is_some_and(|s| l.stop.contains(&s.token)) {
                    Finish::Stop
                } else {
                    Finish::Length
                })
            }
        };
        let _ = l.sub.out.send(Ev::End(end));
    }
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

/// The `k` most probable entries of the logits as (token, probability).
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

fn one(e: &mut EngineQx, caches: &Caches, line: &str, eos: &[u32], hist: &mut Vec<u32>, tower: Option<&crate::vision::Tower>, held: &mut Vec<u64>) -> Result<String> {
    let v = json::parse(line).map_err(|x| Error(format!("a request: {x}")))?;
    // ("stream": true: each pass's new tokens as a line {"part": [ids]} before the reply's own line)
    if matches!(v.get("stream"), Some(Value::Bool(true))) {
        *e.emit.borrow_mut() = Some(Box::new(|ids: &[u32]| {
            let mut o = std::io::stdout();
            let list = ids.iter().map(u32::to_string).collect::<Vec<_>>().join(",");
            let _ = writeln!(o, "{{\"part\":[{list}]}}");
            let _ = o.flush();
        }));
    }
    if let Some(set) = v.get("set") {
        // (every part's cache alike, where the model is split over GPUs)
        let common = &caches.commons[0];
        let mut o = caches.stores[0].offload();
        if let Some(m) = set.get("mode").and_then(Value::as_str) {
            o.mode = Mode::parse(m).ok_or_else(|| Error(format!("mode {m}: copy, cpu, split or auto")))?;
        }
        if let Some(x) = set.get("cpu_threads").and_then(Value::as_u64) {
            o.threads = (x as usize).max(1);
        }
        if let Some(x) = set.get("promote").and_then(Value::as_u64) {
            o.promote = x as usize;
        }
        if let Some(x) = set.get("window").and_then(Value::as_u64) {
            o.window = x;
        }
        if let Some(x) = set.get("spin_us").and_then(Value::as_u64) {
            crate::cpu_experts::set_spin_us(x);
        }
        if let Some(Value::Bool(b)) = set.get("async") {
            o.async_copies = *b;
        }
        if let Some(x) = set.get("skip").and_then(|v| match v {
            Value::Number(n) => n.parse::<f32>().ok(),
            _ => None,
        }) {
            o.skip = x.max(0.0);
        }
        if let Some(x) = set.get("skip_rank").and_then(Value::as_u64) {
            o.skip_rank = x as usize;
        }
        if let Some(x) = set.get("skip_rank_later").and_then(Value::as_u64) {
            o.skip_rank_later = x as usize;
        }
        if let Some(Value::Bool(b)) = set.get("skinny") {
            e.skinny_on.set(*b);
        }
        if let Some(Value::Bool(b)) = set.get("segmented") {
            e.segmented.set(*b);
        }
        e.forget_segments();
        caches.stores.iter().for_each(|s| s.set_offload(o));
        if let Some(x) = set.get("topk").and_then(Value::as_u64) {
            caches.commons.iter().for_each(|c| c.topk.set(x as usize));
        }
        // (a development check of `borrow_places`: {"set": {"borrow_gb": N}} frees the last layers' places until N GiB are free, logs it, and makes them again)
        if let Some(gb) = set.get("borrow_gb").and_then(|v| match v {
            Value::Number(n) => n.parse::<f64>().ok(),
            _ => None,
        }) {
            let before = e.dev.mem_info()?.0;
            let layers = e.borrow_places((gb * (1u64 << 30) as f64) as usize)?;
            let during = e.dev.mem_info()?.0;
            e.return_places(&layers)?;
            let after = e.dev.mem_info()?.0;
            let g = |b: usize| b as f64 / (1u64 << 30) as f64;
            log(&format!("borrowed the places of layers {layers:?}: {:.3} GiB free before, {:.3} while lent, {:.3} after", g(before), g(during), g(after)));
        }
        if let Some(Value::Bool(b)) = set.get("whole") {
            e.whole_on.set(*b);
        }
        if let Some(Value::Bool(b)) = set.get("stream") {
            caches.commons.iter().for_each(|c| c.stream.set(*b));
        }
        if let Some(x) = set.get("trace").and_then(Value::as_u64) {
            TRACE.with(|t| t.set(x.min(3) as u8));
        }
        // ("kprof": true: the next plain decode's kernels timed one launch at a time, a table of them by kernel (src/dev.rs); development only, the steps far slower)
        if let Some(Value::Number(x)) = set.get("draft_min_p") {
            e.draft_min_p.set(x.parse::<f32>().map_err(|_| Error(format!("draft_min_p {x}")))?);
        }
        if let Some(Value::Bool(b)) = set.get("kprof") {
            KPROF_NEXT.with(|t| t.set(*b));
        }
        return Ok(format!("{{\"ok\":true,\"promote\":{},\"window\":{},\"mode\":\"{}\",\"cpu_threads\":{},\"skip\":{},\"skip_rank\":{},\"async\":{},\"topk\":{},\"skinny\":{},\"segmented\":{},\"stream\":{}}}", o.promote, o.window, o.mode.name(), o.threads, o.skip, o.skip_rank, o.async_copies, common.topk.get(), e.skinny_on.get(), e.segmented.get(), common.stream.get()));
    }
    let ids: Vec<u32> = v.get("ids").and_then(Value::as_u64s).ok_or_else(|| Error("a request with no ids".into()))?.into_iter().map(|x| x as u32).collect();
    // ("images": [a path or URL, ...]: the prompt's images in order, each one's place in "ids" one image token (made as long as its tokens) or already as many as its tokens; the
    // reply has the prompt's ids then, "prompt": [...])
    let imgs: Vec<Arc<crate::vision::Image>> = match v.get("images").and_then(Value::as_array) {
        Some(list) => {
            let prep = crate::vision::PREP.get().ok_or_else(|| Error("this model has no vision tower loaded".into()))?;
            let read = |s: &str| if s.starts_with("data:") || s.contains("://") { crate::vision::fetch(s) } else { std::fs::read(s).map_err(|x| Error(format!("{s}: {x}"))) };
            list.iter().map(|s| prep.image(&read(s.as_str().ok_or_else(|| Error("an image that is not a string".into()))?)?).map(Arc::new)).collect::<Result<_>>()?
        }
        None => Vec::new(),
    };
    let token = crate::vision::PREP.get().map_or(u32::MAX, |p| p.image_token);
    let ids = if !imgs.is_empty() && ids.iter().filter(|&&t| t == token).count() == imgs.len() { crate::vision::expand(&ids, token, &imgs)? } else { ids };
    let prompt_json = if imgs.is_empty() { String::new() } else { format!(",\"prompt\":[{}]", ids.iter().map(u32::to_string).collect::<Vec<_>>().join(",")) };
    // ("score_from": N: the ids after the first N read one at a time, teacher-forced, and each one's log-probability under the model returned with how many were not
    // its top pick and how many it gave less than 10%: how close a build stays to a known text, code most of all)
    if let Some(from) = v.get("score_from").and_then(Value::as_u64).map(|x| x as usize).filter(|&f| f > 0 && f < ids.len()) {
        hist.clear();
        held.clear();
        *e.sampler.borrow_mut() = None;
        e.reset()?;
        see(e, tower, &ids, &imgs, 0, 0)?;
        // ("prefill_steps": true: the first ids read one at a time too, for a pack whose experts the GPU's prompt kernels do not read (trellis-coded))
        if matches!(v.get("prefill_steps"), Some(Value::Bool(true))) {
            e.prefill(&ids[..1])?;
            for &t in &ids[1..from] {
                e.step(t)?;
            }
        } else {
            e.prefill(&ids[..from])?;
        }
        let (mut lps, mut miss, mut low) = (Vec::new(), 0usize, 0usize);
        // ("dump": PATH: each scored position's whole row of logits appended to PATH as little-endian f32, for KL divergence against a reference: research/qwen-kld)
        let mut dump = match v.get("dump").and_then(Value::as_str) {
            Some(p) => Some(std::fs::OpenOptions::new().create(true).append(true).open(p).map_err(|x| Error(format!("{p}: {x}")))?),
            None => None,
        };
        let mut put = |lg: &[f32], t: u32| -> Result<()> {
            if let Some(f) = dump.as_mut() {
                let bytes: Vec<u8> = lg.iter().flat_map(|x| x.to_le_bytes()).collect();
                f.write_all(&bytes).map_err(|x| Error(x.to_string()))?;
            }
            let m = lg.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let lse = m as f64 + lg.iter().map(|&x| ((x - m) as f64).exp()).sum::<f64>().ln();
            let t = t as usize;
            let lp = lg[t] as f64 - lse;
            miss += (lg[t] < m) as usize;
            low += (lp < 0.1f64.ln()) as usize;
            lps.push(format!("{lp:.3}"));
            Ok(())
        };
        // ("rows": K: the ids read K a pass the way drafts are verified, all kept, so the multi-row kernels drafting runs are the ones scored)
        let rows = v.get("rows").and_then(Value::as_u64).map_or(1, |x| (x as usize).clamp(1, SPEC_ROWS));
        put(&e.logits()?, ids[from])?;
        let (mut i, voc) = (from, e.w.cfg.vocab);
        while i + 1 < ids.len() {
            if rows > 1 {
                let k = rows.min(ids.len() - 1 - i);
                e.verify(&ids[i..i + k])?;
                let lg = e.verify_logits(k)?;
                e.commit(k, k)?;
                for j in 0..k {
                    put(&lg[j * voc..(j + 1) * voc], ids[i + 1 + j])?;
                }
                i += k;
            } else {
                e.step(ids[i])?;
                put(&e.logits()?, ids[i + 1])?;
                i += 1;
            }
        }
        drop(put);
        let nll = -lps.iter().map(|x| x.parse::<f64>().unwrap_or(0.0)).sum::<f64>() / lps.len() as f64;
        return Ok(format!("{{\"n\":{},\"nll\":{nll:.4},\"miss\":{miss},\"low\":{low},\"lps\":[{}]{prompt_json}}}", lps.len(), lps.join(",")));
    }
    let max_new = v.get("max_new").and_then(Value::as_u64).unwrap_or(1) as usize;
    let mut stop: Vec<u32> = v.get("stop").and_then(Value::as_u64s).unwrap_or_default().into_iter().map(|x| x as u32).collect();
    // ("ignore_eos": true: the reply runs to max_new, an end token or not, as a speed measurement's)
    if !matches!(v.get("ignore_eos"), Some(Value::Bool(true))) {
        stop.extend(eos);
    }
    if ids.is_empty() || ids.len() + max_new + 1 > e.ctx() {
        return err(format!("{} prompt tokens and {max_new} to make, where a sequence may reach {}", ids.len(), e.ctx()));
    }
    // (a conversation's earlier turns reused: where these ids extend the tokens the engine's state holds, only the rest is read; "reuse": false: a fresh start)
    let same = held.len() <= imgs.len() && held.iter().zip(&imgs).all(|(k, i)| *k == i.key);
    let keep = if same && !matches!(v.get("reuse"), Some(Value::Bool(false))) && !hist.is_empty() && hist.len() == e.n && ids.len() > hist.len() && ids.starts_with(hist) { hist.len() } else { 0 };
    hist.clear();
    let n_held = std::mem::take(held).len();
    see(e, tower, &ids, &imgs, keep, n_held)?;
    // (sampling: "temperature", "top_k", "top_p", "min_p", "repetition_penalty", "presence_penalty", "frequency_penalty", "seed"; greedy where none asks for it)
    let num = |k: &str| match v.get(k) {
        Some(Value::Number(n)) => n.parse::<f64>().ok(),
        _ => None,
    };
    let d = crate::sampler::Params::default();
    let params = crate::sampler::Params {
        temperature: num("temperature").map_or(d.temperature, |x| x as f32),
        top_k: num("top_k").map_or(d.top_k, |x| x.max(0.0) as usize),
        top_p: num("top_p").map_or(d.top_p, |x| x.clamp(1e-6, 1.0) as f32),
        min_p: num("min_p").map_or(d.min_p, |x| x.clamp(0.0, 1.0) as f32),
        repetition_penalty: num("repetition_penalty").map_or(d.repetition_penalty, |x| x as f32),
        presence_penalty: num("presence_penalty").map_or(d.presence_penalty, |x| x as f32),
        frequency_penalty: num("frequency_penalty").map_or(d.frequency_penalty, |x| x as f32),
        seed: num("seed").map_or_else(crate::sampler::fresh_seed, |x| x as u64),
        ..d
    };
    *e.sampler.borrow_mut() = (!params.greedy()).then(|| crate::sampler::Sampler::new(params, &ids));
    // drafts from the model's MTP head: {"ids": [...], "max_new": N, "mtp": K}, at most K a pass (beyond the first, while the head is sure enough: --draft-min-p); 6 by default where the head is loaded, "mtp": 0 for none
    let drafts = v.get("mtp").and_then(Value::as_u64).unwrap_or(if e.has_mtp() { 6 } else { 0 });
    if drafts > 0 {
        let k = drafts;
        // (trace 3: the whole generation's stages, every marker, as one step)
        let level = TRACE.with(|t| t.get());
        if level > 2 {
            crate::trace::start(true);
        }
        let (t1, st0) = (Instant::now(), caches.stats());
        // (the kernel census of a drafted generation: every launch of the prompt and the passes, by pass)
        let kprof = KPROF_NEXT.with(|t| t.replace(false));
        e.kprof_passes.set(kprof);
        e.keep.set(keep);
        let (out, st) = e.generate_mtp(&ids, max_new.max(1), k as usize, &stop)?;
        // (the tokens the engine's state now holds: the prompt and every token made but the last, which has not been read)
        if !out.is_empty() && e.n == ids.len() + out.len() - 1 {
            hist.extend_from_slice(&ids);
            hist.extend(out[..out.len() - 1].iter().map(|x| x.token));
            *held = imgs.iter().map(|i| i.key).collect();
        }
        if kprof {
            let tab = crate::dev::kprof_stop();
            let passes = st.passes.max(1) as f64;
            let total: f64 = tab.iter().map(|x| x.2).sum::<f64>() / passes;
            log(&format!("kernel census: {} passes ({:.2} tokens a pass), {total:.0} us a pass in {:.0} launches a pass (each timed alone)", st.passes, out.len() as f64 / passes, tab.iter().map(|x| x.1).sum::<u64>() as f64 / passes));
            for (name, calls, us) in tab.iter().take(30) {
                log(&format!("  {name:<28} {:>6.1} a pass {:>7.2} us each {:>7.1} us a pass", *calls as f64 / passes, us / *calls as f64, us / passes));
            }
        }
        let ms = t1.elapsed().as_secs_f64() * 1e3;
        if level > 2 {
            let marks = crate::trace::stop();
            print_trace(&[(crate::trace::layers(&marks), crate::trace::stages(&marks), ms)]);
        }
        let st1 = caches.stats();
        log(&format!(
            "{} prompt tokens, {} made in {ms:.0} ms with drafts of {k} (the prompt {} ms, then {:.1} ms a token): {} passes, {} of {} guesses right ({:.0}%), {:.2} tokens a pass; the cache: {:.1}% in places, {:.1} pairs on the host a pass ({:.1} experts, {:.2} ms)",
            ids.len(),
            out.len(),
            st.prompt_ms,
            (ms - st.prompt_ms as f64) / (out.len().max(2) - 1) as f64,
            st.passes,
            st.accepted,
            st.drafted,
            100.0 * st.accepted as f64 / st.drafted.max(1) as f64,
            (out.len() - 1) as f64 / st.passes.max(1) as f64,
            100.0 * (st1.hits - st0.hits) as f64 / ((st1.reads - st0.reads) as f64).max(1.0),
            (st1.cpu_pairs - st0.cpu_pairs) as f64 / st.passes.max(1) as f64,
            (st1.cpu - st0.cpu) as f64 / st.passes.max(1) as f64,
            (st1.cpu_ms - st0.cpu_ms) / st.passes.max(1) as f64
        ));
        let list = out.iter().map(|x| x.token.to_string()).collect::<Vec<_>>().join(",");
        return Ok(format!("{{\"ids\":[{list}],\"passes\":{},\"drafted\":{},\"accepted\":{},\"ms\":{ms:.1},\"reused\":{keep}{prompt_json}}}", st.passes, st.drafted, st.accepted));
    }
    if keep == 0 {
        e.reset()?;
    }
    let level = TRACE.with(|t| t.get());
    // (trace 3: the prompt's stages too, every marker, as one step)
    if level > 2 {
        crate::trace::start(true);
    }
    // (a development check's `"dump": PATH, "dump_at": [positions]`: the streams at those positions after every layer of a prompt taken layer by layer, then the last position's logits, written to PATH as bf16 rows and f32)
    #[cfg(feature = "dev")]
    let dump = v.get("dump").and_then(Value::as_str).map(String::from);
    #[cfg(feature = "dev")]
    if dump.is_some() {
        let at: Vec<usize> = v.get("dump_at").and_then(Value::as_u64s).unwrap_or_default().into_iter().map(|x| x as usize).collect();
        crate::model_qwen4x::PEEK.with(|p| *p.borrow_mut() = Some((at, Vec::new())));
    }
    let t1 = Instant::now();
    let mut s = e.prefill(&ids[keep..])?;
    let prefill_ms = t1.elapsed().as_secs_f64() * 1e3;
    #[cfg(feature = "dev")]
    if let Some(path) = dump {
        let rows = crate::model_qwen4x::PEEK.with(|p| p.borrow_mut().take()).map(|x| x.1).unwrap_or_default();
        let mut bytes: Vec<u8> = rows.iter().flat_map(|x| x.to_le_bytes()).collect();
        bytes.extend(e.logits()?.iter().flat_map(|x| x.to_le_bytes()));
        std::fs::write(&path, bytes).map_err(|x| Error(format!("{path}: {x}")))?;
    }
    if level > 2 {
        let marks = crate::trace::stop();
        print_trace(&[(crate::trace::layers(&marks), crate::trace::stages(&marks), prefill_ms)]);
    }
    let top = match v.get("top").and_then(Value::as_u64) {
        Some(k) if k > 0 => Some(top_of(&e.logits()?, k as usize)),
        _ => None,
    };
    let st0 = caches.stats();
    let mut made = vec![s.token];
    e.emit_tokens(std::slice::from_ref(&s));
    let mut times = Vec::new();
    let t2 = Instant::now();
    let mut traced = Vec::new();
    let kprof = KPROF_NEXT.with(|t| t.replace(false));
    if kprof {
        crate::dev::kprof_start();
    }
    while made.len() < max_new && !stop.contains(&s.token) {
        let ts = Instant::now();
        if level > 0 {
            crate::trace::start(level > 1);
        }
        s = e.step(s.token)?;
        let ms = ts.elapsed().as_secs_f64() * 1e3;
        if level > 0 {
            let marks = crate::trace::stop();
            traced.push((crate::trace::layers(&marks), crate::trace::stages(&marks), ms));
        }
        times.push(ms);
        made.push(s.token);
        e.emit_tokens(std::slice::from_ref(&s));
    }
    // (the tokens the engine's state now holds, for the next request's reuse: the prompt and every token made but the last)
    if e.n == ids.len() + made.len() - 1 {
        hist.extend_from_slice(&ids);
        hist.extend_from_slice(&made[..made.len() - 1]);
        *held = imgs.iter().map(|i| i.key).collect();
    }
    if !traced.is_empty() {
        print_trace(&traced);
    }
    if kprof {
        let tab = crate::dev::kprof_stop();
        let steps = times.len().max(1) as f64;
        let total: f64 = tab.iter().map(|x| x.2).sum::<f64>() / steps;
        log(&format!("kernel census: {} steps, {total:.0} us a step in {} launches a step (each timed alone)", times.len(), tab.iter().map(|x| x.1).sum::<u64>() as f64 / steps));
        for (name, calls, us) in &tab {
            log(&format!("  {name:<28} {:>6.1} a step {:>7.2} us each {:>7.1} us a step", *calls as f64 / steps, us / *calls as f64, us / steps));
        }
    }
    let decode_ms = t2.elapsed().as_secs_f64() * 1e3;
    let st1 = caches.stats();
    let steps = times.len().max(1) as f64;
    log(&format!(
        "{} prompt tokens in {prefill_ms:.0} ms; decode {:.1} ms a token; the cache: {:.1} expert reads a token, {:.1}% in places, {:.2} pairs on the host ({:.2} ms), {:.2} copied ({:.2} ms)",
        ids.len(),
        decode_ms / steps,
        (st1.reads - st0.reads) as f64 / steps,
        100.0 * (st1.hits - st0.hits) as f64 / ((st1.reads - st0.reads) as f64).max(1.0),
        (st1.cpu_pairs - st0.cpu_pairs) as f64 / steps,
        (st1.cpu_ms - st0.cpu_ms) / steps,
        (st1.copied - st0.copied) as f64 / steps,
        (st1.copy_ms - st0.copy_ms) / steps
    ));
    let list = |v: Vec<String>| v.join(",");
    let top_json = top.map_or(String::new(), |t| format!(",\"top\":[{}]", t.iter().map(|(i, p)| format!("[{i},{p:.6}]")).collect::<Vec<_>>().join(",")));
    Ok(format!(
        "{{\"ids\":[{}],\"prefill_ms\":{prefill_ms:.1},\"decode_ms\":{decode_ms:.1},\"token_ms\":[{}]{top_json},\"reused\":{keep}{prompt_json}}}",
        list(made.iter().map(|x| x.to_string()).collect()),
        list(times.iter().map(|x| format!("{x:.2}")).collect())
    ))
}

/// A traced generation's timeline, in a development build (src/glm_serve.rs); a release build has no tracing to print.
fn print_trace(steps: &[crate::draft_loop::TracedStep]) {
    #[cfg(feature = "dev")]
    crate::glm_serve::print_trace(steps);
    #[cfg(not(feature = "dev"))]
    let _ = steps;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The split leaves the most places a layer on the part with the fewest: GPUs alike share the layers evenly, one with less room takes fewer, and a first part that cannot hold its least is no split.
    #[test]
    fn the_split_evens_the_places() {
        // (a part's places a layer: its room less a share for each layer, over its layers)
        let room = |rooms: Vec<f64>| move |p: usize, n: usize| (rooms[p] - 70.0 * n as f64) / n as f64;
        assert_eq!(split_layers(48, 2, 2, &room(vec![12000.0, 12000.0])), Some(vec![24]));
        assert_eq!(split_layers(48, 3, 2, &room(vec![12000.0, 12000.0, 12000.0])), Some(vec![16, 32]));
        assert!(split_layers(48, 2, 2, &room(vec![12000.0, 10000.0])).is_some_and(|c| c[0] > 24));
        assert_eq!(split_layers(48, 2, 40, &room(vec![1000.0, 12000.0])), None);
    }
}
