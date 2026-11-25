//! Qwen3.8-Flash-Next (`Qwen4ExpForConditionalGeneration`, Condor's model) on the executor (R3d, #140): its config's layer kinds lowered to the shared operations
//! (`lower::qwen4x`) and loaded here: the matrices packed at the level asked (penguin's lossless packs where none is), the ones the library does not pack and the table of n-grams
//! held as saved (bf16), the small tensors as saved but A_log in fp32 (the delta rule reads it so). The experts either all on the GPU at the level asked (`load`:
//! `moe_qwen4x::QxMoe`, handed over as a `ResidentMoe`), or from a pack of codebook rows in the shared expert store (`load_cached`: the codebook block,
//! `QxMoeCb`, the misses on the host). The MTP layer is a draft head of the model (`load_drafting`: `lower::qwen4x_head`, src/dynamic/head.rs). Not on the
//! executor yet: the vision tower, and the table of n-grams read from the disk in place of the GPU (the family's own engine did both; R3d deleted it).

use super::{Dyn, ModelDesc, Op, Opts, ResidentMoe, Stage};
use crate::cb_pack::CbPack;
use crate::dev::Dev;
use crate::ffi::{err, Result};
use crate::kern_nh::MoeBlock;
use crate::moe_qwen4x::{QxCbShared, QxMoeCb};
use crate::offload::{ExpertStore, Offload};
pub use crate::qwen4x::PackMode;
use crate::weights::Layout;
use glyd_gpu::safetensors::Checkpoint;
use std::path::Path;

/// The executor description's family name.
pub const FAMILY: &str = "Qwen3.8-Flash-Next";

/// The executor's description of the model in `dir` for sequences of up to `ctx` positions: its kinds lowered (`lower::qwen4x`), held to Condor's own reading of
/// the config (`ConfigQx::agrees`).
pub fn lowered(dir: &Path, ctx: usize) -> Result<ModelDesc> {
    let (kinds, label) = crate::runtime::describe::describe_dir(dir)?;
    let c = crate::qwen4x::ConfigQx::agrees(crate::qwen4x::ConfigQx::load(dir)?, &kinds).map_err(|e| e.context(&label))?;
    super::lower::qwen4x(&kinds, c.eos.clone(), ctx.min(c.max_pos)).map_err(|e| e.context(&label))
}

/// The model in `dir` on the executor's loader (`Dyn::new_with`: its matrices at the level asked or this GPU's, the table of n-grams and the shapes the library
/// does not pack as saved, A_log in fp32), its experts all on the GPU (`QxMoe`'s block, handed over as a `ResidentMoe`) at the level asked: a lossy level's codes
/// where `o.layout` is one, else lossless (penguin), as every family's.
pub fn load(dir: &Path, o: &Opts) -> Result<Dyn> {
    load_level(dir, o, experts_at(o.layout))
}

/// The experts' layout for the matrices' `layout`: a lossy level's block codes (six, five or four bits) where it is one, else the lossless layout.
pub fn experts_at(layout: Option<Layout>) -> Layout {
    match layout {
        Some(l @ (Layout::Int6 | Layout::Int5 | Layout::Int4)) => l,
        _ => Layout::Tiered,
    }
}

/// The dynamic route's stage loader of the family (`split::loader`: the gated residuals' streams): the whole model on one GPU, its experts all on it at the level
/// `o.layout` names (`load`). A stage of a model split across GPUs, and its experts in the expert cache as a hummingbird directory's caching says, are refused in
/// words: the family's cache is its pack of codebook rows (`load_cached`, `qwenserve CKPT PACK`).
pub fn stage(dir: &Path, o: &Opts, st: Stage, caching: Option<&crate::hbg_store::HbCaching>) -> Result<Dyn> {
    if !(st.first && st.last) {
        return err(format!("{FAMILY}: a stage of layers {}..{} of a model split across GPUs: the family loads on one GPU yet", st.layers.0, st.layers.1));
    }
    if caching.is_some() {
        return err(format!(
            "{FAMILY}: its experts in the expert cache are a pack of codebook rows (qwenserve CKPT PACK), not a hummingbird directory's places"
        ));
    }
    load(dir, o)
}

/// The draft head of the model in `dir` (its MTP layers: `lower::qwen4x_head`) for sequences of up to `ctx` positions; None where its config has none.
pub fn head_lowered(dir: &Path, ctx: usize) -> Result<Option<ModelDesc>> {
    let (kinds, label) = crate::runtime::describe::describe_dir(dir)?;
    if kinds.desc.mtp.is_empty() {
        return Ok(None);
    }
    let c = crate::qwen4x::ConfigQx::agrees(crate::qwen4x::ConfigQx::load(dir)?, &kinds).map_err(|e| e.context(&label))?;
    super::lower::qwen4x_head(&kinds, c.eos.clone(), ctx.min(c.max_pos)).map(Some).map_err(|e| e.context(&label))
}

/// `load` with the experts at `level` (lossless, or a lossy level's codes: six, five or four bits).
pub fn load_level(dir: &Path, o: &Opts, level: Layout) -> Result<Dyn> {
    load_with(dir, o, level, None)
}

/// `load_level` with the model's draft head (`head_lowered`) drafting `k` tokens a round (at most `VERIFY_ROWS` - 1), its experts held as the model's: each round's
/// drafts checked in one verify pass whose rows are the steps' (src/dynamic/head.rs).
pub fn load_drafting(dir: &Path, o: &Opts, level: Layout, k: usize) -> Result<Dyn> {
    load_with(dir, o, level, Some(k))
}

fn load_with(dir: &Path, o: &Opts, level: Layout, draft: Option<usize>) -> Result<Dyn> {
    let c = crate::qwen4x::ConfigQx::load(dir)?;
    let d = lowered(dir, o.ctx + o.chunk + 16)?;
    // (what the GPU would hold: the experts at the level (lossless: two bytes a weight at most) and the table of n-grams as saved; past that, `load_cached`)
    let per = match level {
        Layout::Int6 => 6,
        Layout::Int5 => 5,
        Layout::Int4 => 4,
        _ => 16,
    };
    let experts = (c.layers * c.experts * 3 * c.inter * c.hidden) as u64 * per / 8;
    let table: u64 = held_as_saved(&d).iter().map(|&k| (d.mats[k].rows * d.mats[k].cols * 2) as u64).sum();
    let free = Dev::new(o.device)?.mem_info()?.0 as u64;
    if experts + table > free {
        return err(format!(
            "{FAMILY}: its experts ({:.1} GB at {per} bits) and its table of n-grams ({:.1} GB) do not fit this GPU's {:.1} GB free: a pack of its experts serves it (`load_cached`)",
            experts as f64 / 1e9,
            table as f64 / 1e9,
            free as f64 / 1e9
        ));
    }
    let st = Stage::whole(d.layers.len());
    // (layer l's experts, the MTP layers' after the model's)
    let resident = |ck: &Checkpoint, ls: std::ops::Range<usize>| -> Result<Vec<Box<dyn MoeBlock>>> {
        ls.map(|l| Ok(Box::new(ResidentMoe(crate::moe_qwen4x::QxMoe::load(l, &c, ck, level)?.into_layer())) as Box<dyn MoeBlock>)).collect()
    };
    let blocks = |ck: &Checkpoint, _: Layout| resident(ck, 0..c.layers);
    let head = |ck: &Checkpoint, _: Layout| resident(ck, c.layers..c.layers + c.mtp_layers);
    let spec = match draft {
        None => None,
        Some(k) => {
            let h = head_lowered(dir, o.ctx + o.chunk + 16)?
                .ok_or_else(|| crate::ffi::Error(format!("{FAMILY}: {} has no MTP layer to draft with", dir.display())))?;
            Some(super::DraftSpec { dir, desc: h, k, experts: Some(&head) })
        }
    };
    Dyn::new_with(dir, d, spec, st, o, &blocks)
}

/// Condor's routed experts from a pack of codebook rows (glyd-hbr-codes/2: each expert cb2, cb3, cb4 or ternary, the pack's mode's sidecar over them) in the
/// shared expert store (src/offload.rs): a layer's places on the GPU, the pack's file mapped as their host tier, the misses computed by the host or copied in as
/// `offload` says.
pub struct Caching {
    pub pack: std::path::PathBuf,
    pub mode: PackMode,
    /// places a layer (None: what the GPU has free after the always-on matrices, less `reserve`, over the layers)
    pub places: Option<usize>,
    pub offload: Offload,
    /// GPU bytes the places leave for the sequences' keys, values and states and the executor's scratch (`reserve_for`)
    pub reserve: u64,
}

impl Caching {
    /// The pack in `pack` served as `mode`, its places sized by the room the GPU has, its misses computed by the host, the mode's skipping, room kept for `users`
    /// sequences of the description `d`.
    pub fn new(pack: &Path, mode: PackMode, d: &ModelDesc, users: usize) -> Caching {
        let offload = Offload { mode: crate::offload::Mode::Cpu, skip_rank: mode.skip_rank(), async_copies: true, ..Offload::default() };
        Caching { pack: pack.to_path_buf(), mode, places: None, offload, reserve: reserve_for(d, users) }
    }
}

/// The GPU memory a model's places leave: each of `users` sequences' keys, values and states for the description's context, and the executor's slots and scratch
/// (1.5 GiB, not measured for this model: the final pass measures it).
pub fn reserve_for(d: &ModelDesc, users: usize) -> u64 {
    let ring: u64 = d.rings.iter().map(|r| (d.max_pos * r.nk * r.d * 2 * (1 + r.values as usize)) as u64).sum();
    let states: u64 = d.states.iter().map(|&b| b as u64).sum();
    users.max(1) as u64 * (ring + states) + (3 << 29)
}

/// Places a layer: the GPU's `free` bytes less `reserve`, over `layers` layers of places of `place` bytes; at least one, at most `kept` (all of a layer's).
pub fn places_for(free: u64, reserve: u64, layers: usize, place: usize, kept: usize) -> usize {
    ((free.saturating_sub(reserve) / (layers.max(1) * place.max(1)) as u64) as usize).clamp(1, kept.max(1))
}

/// The model in `dir` on the executor's loader with its routed experts in the expert cache (`Caching`): the always-on matrices as `load` loads them, each layer's
/// block Condor's codebook block (`QxMoeCb`) over one store shared by the layers.
pub fn load_cached(dir: &Path, o: &Opts, k: &Caching) -> Result<Dyn> {
    let c = crate::qwen4x::ConfigQx::load(dir)?;
    // (a width the codebook kernels do not take is refused before anything is read or put on the GPU)
    crate::moe_qwen4x::takes(c.hidden, c.inter).map_err(|e| e.context(FAMILY))?;
    let d = lowered(dir, o.ctx + o.chunk + 16)?;
    let mut pack = CbPack::open_dims(&k.pack, c.hidden, c.inter)?;
    if let Some(side) = k.mode.side(&k.pack) {
        pack = pack.with_side(&side)?;
    }
    if let Some(l) = (0..c.layers).find(|&l| !pack.has(l)) {
        return err(format!("{}: no experts of layer {l} of the model's {}", k.pack.display(), c.layers));
    }
    let (store, common) = (ExpertStore::new(), QxCbShared::new());
    store.set_offload(k.offload);
    common.stream.set(true);
    let place = (0..c.layers).map(|l| pack.place_bytes(l)).max().unwrap_or(1);
    let kept = (0..c.layers).filter_map(|l| pack.kept(l).map(<[usize]>::len)).min().unwrap_or(1);
    let dev = o.device;
    let blocks = |ck: &Checkpoint, layout: Layout| -> Result<Vec<Box<dyn MoeBlock>>> {
        // (read after the always-on matrices are on the GPU: what is free now, less the reserve, is the places')
        let slots = match k.places {
            Some(p) => p,
            None => places_for(Dev::new(dev)?.mem_info()?.0 as u64, k.reserve, c.layers, place, kept),
        };
        (0..c.layers)
            .map(|l| {
                crate::status::STATUS.beat();
                Ok(Box::new(QxMoeCb::load(l, &c, ck, layout, &pack, &store, slots, &common)?) as Box<dyn MoeBlock>)
            })
            .collect()
    };
    let e = Dyn::new_with(dir, d, None, Stage::whole(c.layers), o, &blocks)?;
    common.fill_page_tables();
    Ok(e)
}

/// `load_cached` of the pack in `pack` served as its mode (`PackMode::of`), room kept for `users` sequences.
pub fn load_pack(dir: &Path, pack: &Path, precise: bool, users: usize, o: &Opts) -> Result<Dyn> {
    let d = lowered(dir, o.ctx + o.chunk + 16)?;
    load_cached(dir, o, &Caching::new(pack, PackMode::of(pack, precise), &d, users))
}

/// `qwenserve CKPT PACK --port P [--host H] [--page DIR] [--name NAME] [--users N] [--ctx N] [--chunk N] [--places N] [--precise] [--layout L]`: the HTTP server
/// (src/server.rs: the chat page, the OpenAI and Anthropic APIs, the Monitor's numbers) on the description executor, the model's routed experts from its pack of
/// codebook rows in the expert cache (`load_cached`: `--places N` a layer, else what the GPU has free past the reserve), the always-on matrices at the level asked
/// (four bits where none is), its requests through the common scheduler (`server::work_with`), up to N at once. It listens while the model loads (/health is 503
/// until it is loaded). Thinking is off where a request does not ask for it; the sampling defaults are the model's, at a temperature of 0.5 at most with min-p
/// 0.1. Not on the executor yet: the vision tower, and the draft head over the expert cache (its verify pass takes experts held whole on the GPU: src/dynamic/head.rs).
pub fn http(snap: &Path, pack: &Path, a: &crate::args::Args) -> Result<()> {
    use crate::server;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;
    let name = a.flag("name").unwrap_or("qwen3.8-flash-next").to_string();
    let ctx = a.num("ctx", 33536usize)?;
    let users = a.num("users", 4usize)?;
    if !(1..=crate::sched::MAX_ROWS).contains(&users) {
        return err(format!("--users {users}: 1 to {} requests at once", crate::sched::MAX_ROWS));
    }
    let mode = PackMode::of(pack, a.has("precise"));
    crate::monitor::start(mode.name(), ctx);
    server::STANDALONE.store(true, Ordering::Relaxed);
    if let Some(d) = a.flag("page") {
        let _ = server::PAGE_DIR.set(d.into());
    }
    let tok = crate::tok::Tok::load(snap)?;
    let chat = crate::chat::Chat::load(snap)?;
    let mut defaults = crate::sampler::generation_defaults(snap);
    if defaults.temperature > 0.0 {
        // (one-shot games of ~15K tokens in precise mode, browser-tested: the model's own temperature (1.0) ended about one answer in four early; 0.5 with min-p 0.1 made
        // 3 in 6, with no loop or early end. A request's own values win.)
        defaults.temperature = defaults.temperature.min(0.5);
        defaults.min_p = 0.1;
    }
    let host = a.flag("host").unwrap_or("127.0.0.1");
    let port: u16 = a.num("port", 8080u16)?;
    let key = std::env::var("GLYD_API_KEY").ok().filter(|k| !k.is_empty());
    if let Some(why) = server::bind_refusal(host, key.is_some(), a.has("insecure-no-auth")) {
        return err(why);
    }
    server::set_access(server::parse_access(a.flag("allowed-origins"), a.flag("allowed-hosts")).map_err(crate::ffi::Error)?);
    let (tx, rx) = std::sync::mpsc::channel();
    // (/health says "loading" until the model can serve: a script that waits for it does not send its first request into the load)
    let loading = Arc::new(AtomicBool::new(true));
    crate::ops::drain_on_sigterm(crate::ops::Life::global(), std::time::Duration::from_secs(a.num("drain-timeout", crate::ops::DRAIN_SECONDS)?), |_| {});
    let addr = server::listen_loading(&format!("{host}:{port}"), &name, ctx - 2, key, server::is_loopback_addr(host), tx.clone(), loading.clone())?;
    println!("Qwen3.8-Flash-Next on Glyd: the chat at http://{addr}/, the API at http://{addr}/v1 (OpenAI and Anthropic). Loading the model...");
    // (the always-on matrices at four bits by default, as the release has been served; `--layout` another level's)
    let o = Opts { ctx, chunk: a.num("chunk", 1024usize)?, graph: true, device: 0, layout: a.layout()?.or(Some(Layout::Int4)) };
    let mut k = Caching::new(pack, mode, &lowered(snap, o.ctx + o.chunk + 16)?, users);
    if let Some(p) = a.flag("places") {
        k.places = Some(p.parse().map_err(|_| crate::ffi::Error(format!("--places {p}: a number of places a layer")))?);
    }
    let mut e = load_cached(snap, &o, &k)?;
    let vocab = e.desc().vocab;
    let Some((n, step)) = crate::conv::Gen::as_step(&mut e, users)? else { return err(format!("{FAMILY}: the engine is not a scheduler's step")) };
    let window = crate::sched::ModelStep::window(step).min(ctx - 2);
    let rt = crate::runtime::context::Context::process().clone();
    let shared = Arc::new(server::Shared { tok, chat, name, defaults, window, vocab, thinking: Some(false), refused: &[], rt });
    crate::monitor::ready();
    loading.store(false, Ordering::Relaxed);
    println!("Ready: up to {n} requests at once.");
    server::work_with(&mut crate::sched::Sched::new(step, n), shared, &rx, &tx);
    Ok(())
}

/// `qwenget REPO DIR [--precise | --best]`: the model from the Hugging Face Hub (src/hub.rs: large files as ranges on several connections, resumed where they stopped, checked by SHA-256) into the Hub's cache, which
/// every tool on the computer shares, and DIR/model and DIR/pack (with --precise, DIR/precise in its place; with --best, DIR/best) linked to its folders.
pub fn get(repo: &str, dir: &Path, a: &crate::args::Args) -> Result<()> {
    use crate::ffi::Error;
    use crate::hub::{disk_free, Hub, HubError};
    use std::io::Write;
    use std::time::Instant;
    let hub = Hub::from_env();
    let e = |x: HubError| match x {
        HubError::Denied(code) => Error(format!("{repo}: the Hub answered {code} (no such repository, or a private one: set HF_TOKEN)")),
        x => Error(format!("{repo}: {x}")),
    };
    let remote = hub.list(repo, "main").map_err(e)?;
    // (fast mode's experts are fast/ since v0.2.2 (format q); pack/ is the earlier engines' fast. precise-v2/ and best-v2/ (format q at 3 and 4 bits) are
    // this engine's precise and best, precise/ and best/ the earlier engines' (and ours where the repo has no -v2 yet))
    let has = |p: &str| remote.files.iter().any(|f| f.path.starts_with(&format!("{p}/")));
    let experts = if a.has("best") {
        ["best-v2", "best"]
    } else if a.has("precise") {
        ["precise-v2", "precise"]
    } else {
        ["fast", "fast"]
    };
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
            return err(format!(
                "{:.0} GB more to download and {:.0} GB free in {}: free some space (or set HF_HUB_CACHE to a disk with room)",
                need as f64 / 1e9,
                free as f64 / 1e9,
                at.display()
            ));
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
            let made = std::process::Command::new("cmd")
                .args(["/C", "mklink", "/J"])
                .arg(&link)
                .arg(snap.join(p))
                .output()
                .map_err(|x| Error(format!("{}: {x}", link.display())))?;
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

/// `glyd-engine servecheck CASE [--tokens N]`: the family's HTTP server (`http`) on the expert cache's random case (`logits::qwen4x_cb_random`, CASE `qwen4x-cb`:
/// its pack, a tokenizer and chat template made for it) against the common scheduler in this process on the same model, pack and places (`load_cached` as
/// `http` loads it): for a short prompt and one past a chunk, the served completion's tokens (greedy, N of them, by their ids) the scheduler's for the same
/// tokens. An error where one differs.
#[cfg(feature = "dev")]
pub fn serve_check(a: &crate::args::Args) -> Result<()> {
    use crate::sched::{Ev, Finish, Reply, Sched, Submit};
    use std::io::{Read, Write};
    let case = a.need(1, "CASE")?;
    if case != "qwen4x-cb" {
        return err(format!("servecheck {case}: the served case is qwen4x-cb (the family's random model with its pack of experts)"));
    }
    let tokens = a.num("tokens", 16usize)?;
    let dir = std::env::temp_dir().join(format!("glyd-servecheck-{}", std::process::id()));
    std::fs::create_dir_all(&dir).map_err(|e| crate::ffi::Error(e.to_string()))?;
    crate::logits::qwen4x_cb_random(&dir)?;
    crate::logits::write_serving_files(&dir, include_str!("../../tests/fixtures/chat_qwen38_tools.jinja"))?;
    let pack = dir.join("pack");
    let (ctx, chunk, places, users) = (1024usize, 64usize, 3usize, 2usize);
    let port = std::net::TcpListener::bind("127.0.0.1:0").and_then(|l| l.local_addr()).map_err(|e| crate::ffi::Error(e.to_string()))?.port();
    let words: Vec<String> = ["qwenserve", &dir.display().to_string(), &pack.display().to_string(), "--port", &port.to_string()]
        .into_iter()
        .map(String::from)
        .chain(["--ctx", &ctx.to_string(), "--chunk", &chunk.to_string(), "--places", &places.to_string(), "--users", &users.to_string()].map(String::from))
        .collect();
    let served = crate::args::Args::from(words);
    let (d2, p2) = (dir.clone(), pack.clone());
    std::thread::spawn(move || {
        if let Err(e) = http(&d2, &p2, &served) {
            eprintln!("the server: {}", e.msg);
        }
    });
    // (a request on its own connection, the whole reply read: its status and body; the host this computer's own address, which the server's checks take)
    let ask = |method: &str, path: &str, body: &str| -> Option<(u16, String)> {
        let mut c = std::net::TcpStream::connect(("127.0.0.1", port)).ok()?;
        let head = format!(
            "{method} {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n",
            body.len()
        );
        c.write_all(head.as_bytes()).ok()?;
        c.write_all(body.as_bytes()).ok()?;
        let mut raw = String::new();
        c.read_to_string(&mut raw).ok()?;
        let (h, rest) = raw.split_once("\r\n\r\n")?;
        Some((h.split_whitespace().nth(1)?.parse().ok()?, rest.to_string()))
    };
    let t0 = std::time::Instant::now();
    while ask("GET", "/health", "").is_none_or(|r| r.0 != 200) {
        if t0.elapsed().as_secs() > 600 {
            return err("servecheck: the server did not say it was ready in 600 s");
        }
        std::thread::sleep(std::time::Duration::from_millis(500));
    }
    println!("the server is ready at port {port} after {:.1} s", t0.elapsed().as_secs_f64());
    // (the prompts as ids: a short one, and one past the chunk of 64)
    let prompts: Vec<Vec<u32>> = vec![(0..9).map(|i| 40 + 3 * i).collect(), (0..150).map(|i| (37 * i + 11) % 2000).collect()];
    let mut got = Vec::new();
    for p in &prompts {
        let list = p.iter().map(u32::to_string).collect::<Vec<_>>().join(",");
        let body = format!(
            r#"{{"model":"m","prompt":[{list}],"max_tokens":{tokens},"temperature":0,"ignore_eos":true,"logprobs":0,"return_tokens_as_token_ids":true}}"#
        );
        let (status, reply) = ask("POST", "/v1/completions", &body).ok_or_else(|| crate::ffi::Error("servecheck: the completion was not read".into()))?;
        let v =
            glyd_gpu::json::parse(&reply).map_err(|e| crate::ffi::Error(format!("servecheck: a reply of status {status} that is not JSON ({e}): {reply}")))?;
        let ids: Vec<u32> = v
            .get("choices")
            .and_then(|c| c.as_array())
            .and_then(|c| c.first())
            .and_then(|c| c.get("logprobs"))
            .and_then(|l| l.get("tokens"))
            .and_then(|t| t.as_array())
            .map(|t| t.iter().filter_map(|x| x.as_str()?.strip_prefix("token_id:")?.parse().ok()).collect())
            .unwrap_or_default();
        if status != 200 || ids.len() != tokens {
            return err(format!("servecheck: status {status}, {} token ids where {tokens} were asked: {reply}", ids.len()));
        }
        got.push(ids);
    }
    // the same model, pack and places on the scheduler in this process, the same prompts one after another
    let o = Opts { ctx, chunk, graph: true, device: 0, layout: Some(Layout::Int4) };
    let mut k = Caching::new(&pack, PackMode::of(&pack, false), &lowered(&dir, o.ctx + o.chunk + 16)?, users);
    k.places = Some(places);
    let mut e = load_cached(&dir, &o, &k)?;
    let Some((n, step)) = crate::conv::Gen::as_step(&mut e, users)? else { return err("servecheck: no scheduler's step") };
    let mut s = Sched::new(step, n);
    let reply = Reply {
        params: crate::sampler::Params { temperature: 0.0, ..Default::default() },
        max_new: tokens,
        ignore_eos: true,
        logprobs: None,
        stop_ids: Vec::new(),
    };
    let mut ok = true;
    for (p, served) in prompts.iter().zip(&got) {
        let (tx, rx) = std::sync::mpsc::channel();
        s.submit(Submit {
            prompt: p.clone(),
            reply: reply.clone(),
            out: tx,
            abort: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
            resumed: Default::default(),
            cached: Default::default(),
            created: Default::default(),
            salt: 0,
            pin: 0,
            images: Vec::new(),
        });
        while !s.idle() {
            s.tick()?;
        }
        let mut mine = Vec::new();
        while let Ok(ev) = rx.recv() {
            match ev {
                Ev::Token(t) => mine.push(t),
                Ev::Lp(_) => {}
                Ev::End(Ok(Finish::Length)) => break,
                Ev::End(f) => return err(format!("servecheck: the scheduler's sequence ended {f:?} before its {tokens} tokens")),
            }
        }
        let same = mine == *served;
        ok &= same;
        println!(
            "{} a served prompt of {} tokens: its {tokens} tokens {} the scheduler's here{}",
            if same { "PASS" } else { "FAIL" },
            p.len(),
            if same { "are" } else { "are NOT" },
            if same { String::new() } else { format!(" (served {served:?}, here {mine:?})") }
        );
    }
    std::fs::remove_dir_all(&dir).ok();
    if !ok {
        return err("servecheck qwen4x-cb: a served reply is not the scheduler's");
    }
    println!("PASS servecheck {case}");
    Ok(())
}

/// The matrices held as saved: the tables of n-grams (`Op::NgramRows` reads their rows).
fn held_as_saved(d: &ModelDesc) -> Vec<usize> {
    d.layers.iter().flatten().filter_map(|op| if let Op::NgramRows(x) = op { Some(x.table) } else { None }).collect()
}

#[cfg(test)]
mod tests {
    use super::super::{HeadMix, IdxKeys, Op, Slot, StreamsMix};
    use crate::runtime::describe::describe;

    fn tiny() -> serde_json::Value {
        let m: serde_json::Value = serde_json::from_str(include_str!("../../tests/fixtures/qwen4x_tiny_manifest.json")).unwrap();
        serde_json::from_str(m["config"].as_str().unwrap()).unwrap()
    }

    /// The executor's loader takes every matrix of the tiny model (a whole model's stage wants each one), holds as saved the table of n-grams and the shapes the
    /// library does not pack (the inject weights' 4 rows, the b and a gates' 2 value heads), packs the rest, and widens A_log alone to fp32.
    #[test]
    fn the_loader_holds_each_matrix_and_small_tensor_as_its_operations_read_it() {
        let m = super::super::lower::qwen4x(&describe(&tiny()).unwrap(), vec![2000], 1100).unwrap();
        assert!(m.wanted(&super::super::Stage::whole(m.layers.len())).iter().all(|&w| w), "a matrix no operation multiplies by");
        let man: serde_json::Value = serde_json::from_str(include_str!("../../tests/fixtures/qwen4x_tiny_manifest.json")).unwrap();
        let rows: std::collections::HashMap<String, usize> =
            man["tensors"].as_array().unwrap().iter().map(|t| (t[0].as_str().unwrap().to_string(), t[2][0].as_u64().unwrap() as usize)).collect();
        let rows_of = |n: &str| rows.get(n).copied();
        let saved: Vec<usize> = (0..m.mats.len()).filter(|&k| super::super::held_as_saved(&m, k, &rows_of)).collect();
        let table = super::held_as_saved(&m);
        assert!(!table.is_empty() && table.iter().all(|k| saved.contains(k)), "{table:?} of {saved:?}");
        for &k in &saved {
            let it = &m.mats[k];
            assert!(
                table.contains(&k) || it.members.iter().any(|n| rows_of(n).unwrap() % 64 != 0) || it.cols % 16 != 0,
                "{}: {}x{}",
                it.members[0],
                it.rows,
                it.cols
            );
        }
        // (and packs the rest: a matrix outside the table is held as saved only where its rows or its columns are not a multiple of 64)
        for k in 0..m.mats.len() {
            let it = &m.mats[k];
            assert_eq!(saved.contains(&k), table.contains(&k) || it.rows % 64 != 0 || it.cols % 64 != 0, "{}", it.members[0]);
        }
        assert!(saved.len() > table.len(), "the unpacked shapes");
        let f32s = super::super::read_f32(&m);
        assert!(!f32s.is_empty() && f32s.iter().all(|&i| m.small[i].ends_with("A_log")), "{:?}", f32s.iter().map(|&i| &m.small[i]).collect::<Vec<_>>());
    }

    /// The tiny model of the `qwen4x` logits case, lowered: eight layers of gated residuals around each block (a gated mix before it, the gated update after),
    /// the n-gram embedding's three operations before the second layer's, Gated DeltaNet with the sigmoid gate, the sparse attention's index product, its
    /// queries and pools, its unweighted pick of 2 pools of 4 and the tail, the attention over the picks with its keys and values in states, the experts, and
    /// the final mixer on the last layer; every matrix and small tensor named is the checkpoint's, and every slot an operation reads is sized.
    #[test]
    fn condors_tiny_model_lowers_to_the_shared_operations() {
        let d = describe(&tiny()).unwrap();
        let m = super::super::lower::qwen4x(&d, vec![2000], 1100).unwrap();
        assert!(matches!(m.first, Op::Streams));
        // (each sparse attention layer's keys and values, its indexer's raw keys and its pools' keys: three rings, paged as every attention's, R4b)
        assert_eq!((m.layers.len(), m.hidden, m.rings.len(), m.ropes.len()), (8, 256, 6, 1));
        for three in m.rings.chunks(3) {
            let shapes: Vec<_> = three.iter().map(|r| (r.d, r.nk, r.window, r.values, r.ratio)).collect();
            assert_eq!(shapes, [(256, 2, 0, true, 1), (128, 1, 0, false, 1), (128, 1, 0, false, 4)]);
        }
        assert!(super::super::rows::pages_all(&m));
        let kinds: Vec<Vec<&'static str>> = m
            .layers
            .iter()
            .map(|ops| {
                ops.iter()
                    .map(|op| match op {
                        Op::StreamsPre { mix: StreamsMix::Gated { .. }, .. } => "pre",
                        Op::StreamsPost { gated: true, .. } => "post",
                        Op::StreamsHead { mix: HeadMix::Gated { .. }, .. } => "head",
                        Op::NgramRows(_) => "ngram rows",
                        Op::NgramAdd(_) => "ngram add",
                        Op::Linear { .. } => "linear",
                        Op::GdnConv { .. } => "conv",
                        Op::GdnDelta { .. } => "delta",
                        Op::GatedNorm { sigmoid: true, .. } => "sigmoid norm",
                        Op::IdxMean { .. } => "index",
                        Op::IdxPick(x) if !x.weighted && matches!(x.keys, IdxKeys::Pools(_)) && (x.ratio, x.top_k, x.heads) == (4, 2, 4) => "pick",
                        Op::Attn(a) if a.picked && (a.nq, a.nk, a.d) == (24, 2, 256) => "picked attention",
                        Op::Moe(_) => "experts",
                        o => panic!("{o:?}"),
                    })
                    .collect()
            })
            .collect();
        let gdn = ["pre", "linear", "linear", "conv", "delta", "sigmoid norm", "linear", "post", "pre", "experts", "post"];
        let attn = ["pre", "linear", "linear", "index", "pick", "picked attention", "linear", "post", "pre", "experts", "post"];
        assert_eq!(kinds[0], gdn);
        assert_eq!(kinds[1][..3], ["ngram rows", "linear", "ngram add"]);
        assert_eq!(kinds[1][3..], gdn);
        assert_eq!(kinds[3], attn);
        assert_eq!(kinds[7][..attn.len()], attn);
        assert_eq!(kinds[7][attn.len()..], ["head"]);
        // (each matrix's members are the checkpoint's tensors, its rows theirs stacked)
        let names = d.names();
        assert!(m.mats.iter().all(|it| it.members.iter().all(|n| names.contains(n))) && m.small.iter().all(|n| names.contains(n)));
        let rows: Vec<(usize, usize)> = m.mats.iter().map(|it| (it.rows, it.cols)).collect();
        // (layer 0: the gated residual's three, qkvz, ba, out, the experts' gated residual's three)
        assert_eq!(rows[..9], [(64, 1024), (1024, 64), (4, 1024), (2048, 256), (12, 256), (256, 768), (64, 1024), (1024, 64), (4, 1024)]);
        let table = m.layers[1].iter().find_map(|op| if let Op::NgramRows(x) = op { Some(*x) } else { None }).unwrap();
        assert_eq!((rows[table.table], table.per, table.row, table.eos), ((128 * 132, 16), 8, 16, 2000));
        // (the states: per GDN layer its convolution's and recurrence's; the n-gram's tokens and convolution; none of an attention layer's, which are rings)
        assert_eq!(m.states.iter().filter(|&&b| b == 1100 * 2 * 256 * 2 || b == 1100 * 128 * 2).count(), 0);
        assert_eq!(m.states.iter().filter(|&&b| b == 8).count(), 1);
        for (s, w) in
            [(Slot::S, 1024), (Slot::Hq, 1024), (Slot::Mid, 64), (Slot::Mix, 12288 + 1024), (Slot::Qr, 640), (Slot::Iq, 512), (Slot::Gu, 1280), (Slot::Kv, 4)]
        {
            assert_eq!(m.width(s), w, "{s:?}");
        }
        // (A_log the delta rule's, read in fp32)
        assert_eq!(super::super::read_f32(&m).len(), 6);
        assert!(super::super::read_f32(&m).iter().all(|&i| m.small[i].ends_with("linear_attn.A_log")));
    }

    /// The places take what the GPU has free past the reserve, a layer's share of it: at least one, at most all of a layer's experts; the reserve holds each
    /// sequence's keys, values and states beside the scratch, and grows with the users by them.
    #[test]
    fn the_places_take_the_room_past_the_reserve() {
        assert_eq!(super::places_for(10 << 30, 2 << 30, 48, 1 << 20, 512), 170);
        assert_eq!(super::places_for(1 << 30, 2 << 30, 48, 1 << 20, 512), 1);
        assert_eq!(super::places_for(1 << 40, 0, 48, 1 << 20, 512), 512);
        let m = super::super::lower::qwen4x(&describe(&tiny()).unwrap(), vec![2000], 1100).unwrap();
        let (one, four) = (super::reserve_for(&m, 1), super::reserve_for(&m, 4));
        let states: u64 = m.states.iter().map(|&b| b as u64).sum();
        let seq = one - (3 << 29);
        assert!(states > 0 && seq >= states && four - one == 3 * seq, "{one} {four} {states}");
    }

    /// The experts take a lossy level's codes where the matrices' layout is one, else the lossless layout (penguin's, as every family's).
    #[test]
    fn the_experts_are_lossless_unless_a_lossy_level_is_asked() {
        use crate::weights::Layout;
        assert_eq!(super::experts_at(None), Layout::Tiered);
        assert_eq!(super::experts_at(Some(Layout::Twelve)), Layout::Tiered);
        for l in [Layout::Int6, Layout::Int5, Layout::Int4] {
            assert_eq!(super::experts_at(Some(l)), l);
        }
    }

    /// The indexer's head size and pool are the config's: heads of 64 over pools of 2 lower to the raw keys' ring of 64, the pools' ring of entries of 2 tokens,
    /// the index product's rows of 64 a head and the pick's scores over heads of 64.
    #[test]
    fn the_indexers_heads_and_pools_are_the_configs() {
        let mut c = tiny();
        c["text_config"]["indexer_head_dim"] = 64.into();
        c["text_config"]["indexer_compress_ratio"] = 2.into();
        let m = super::super::lower::qwen4x(&describe(&c).unwrap(), vec![2000], 1100).unwrap();
        let ix = m.layers.iter().flatten().find_map(|op| if let Op::IdxMean { heads, dim, block, .. } = *op { Some((heads, dim, block)) } else { None });
        assert_eq!(ix, Some((4, 64, 2)));
        let pick = m.layers.iter().flatten().find_map(|op| if let Op::IdxPick(x) = op { Some(x.clone()) } else { None }).unwrap();
        assert_eq!((pick.dim(), pick.ratio, pick.top_k), (64, 2, 4));
        assert_eq!((m.width(Slot::Qr), m.width(Slot::Iq)), (5 * 64, 4 * 64));
        let mut c = tiny();
        c["text_config"]["indexer_head_dim"] = 100.into();
        assert!(super::super::lower::qwen4x(&describe(&c).unwrap(), vec![2000], 1100).unwrap_err().msg.contains("multiple of 32"));
    }

    /// The family's second random case (`logits::qwen4x_s2_random`: 2 streams, index heads of 64 over pools of 2, no n-gram embedding) lowers to the shared
    /// operations with those counts, and every matrix it multiplies by is a tensor of its manifest of the shape the lowering reads.
    #[cfg(feature = "dev")]
    #[test]
    fn the_second_random_case_lowers_with_its_counts() {
        let m: serde_json::Value = serde_json::from_str(&crate::logits::qwen4x_s2_manifest().unwrap()).unwrap();
        let c: serde_json::Value = serde_json::from_str(m["config"].as_str().unwrap()).unwrap();
        let d = super::super::lower::qwen4x(&describe(&c).unwrap(), vec![2000], 1100).unwrap();
        assert_eq!((d.streams(), d.width(Slot::S)), (2, 512));
        let ix = d.layers.iter().flatten().find_map(|op| if let Op::IdxMean { dim, block, .. } = *op { Some((dim, block)) } else { None });
        assert_eq!(ix, Some((64, 2)));
        assert!(!d.layers.iter().flatten().any(|op| matches!(op, Op::NgramRows(_))));
        let shapes: std::collections::HashMap<String, Vec<u64>> = m["tensors"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| (t[0].as_str().unwrap().to_string(), t[2].as_array().unwrap().iter().map(|v| v.as_u64().unwrap()).collect()))
            .collect();
        for it in &d.mats {
            let rows: u64 = it.members.iter().map(|n| shapes.get(n).unwrap_or_else(|| panic!("{n}: not in the manifest"))[0]).sum();
            let cols = *shapes[&it.members[0]].last().unwrap();
            assert_eq!((rows as usize, cols as usize), (it.rows, it.cols), "{:?}", it.members);
        }
    }

    /// The draft head of the family's random case with its MTP layer (`logits::qwen4x_mtp_manifest`) lowers to a draft part: `Op::MtpIn` (the head's two norms
    /// and two products of its manifest), one layer of the model's attention layer's operations then the experts' and the head's own final mixer, three rings of
    /// its own (its keys and values, its indexer's raw keys and pools), the model's embedding and LM head named; beside the model it fits, and its rings move
    /// after the model's six.
    #[cfg(feature = "dev")]
    #[test]
    fn the_draft_head_lowers_to_a_draft_part() {
        let m: serde_json::Value = serde_json::from_str(&crate::logits::qwen4x_mtp_manifest().unwrap()).unwrap();
        let c: serde_json::Value = serde_json::from_str(m["config"].as_str().unwrap()).unwrap();
        let d = describe(&c).unwrap();
        let big = super::super::lower::qwen4x(&d, vec![2000], 1100).unwrap();
        let mut h = super::super::lower::qwen4x_head(&d, vec![2000], 1100).unwrap();
        let Op::MtpIn { norm_embed, embed, norm_hidden, hidden } = h.first else { panic!("{:?}", h.first) };
        assert_eq!((h.small[norm_embed].as_str(), h.small[norm_hidden].as_str()), ("mtp.pre_fc_norm_embedding.weight", "mtp.pre_fc_norm_hidden.weight"));
        assert_eq!((h.mats[embed].members[0].as_str(), h.mats[hidden].members[0].as_str()), ("mtp.fc_embedding.weight", "mtp.fc_hidden.weight"));
        assert_eq!((h.layers.len(), h.rings.len(), h.width(Slot::Hid), h.width(Slot::S)), (1, 3, 1024, 1024));
        let names: Vec<String> = h.layers[0].iter().map(super::super::rows::op_name).collect();
        assert_eq!(
            names,
            ["StreamsPre", "Linear", "Linear", "IdxMean", "IdxPick", "Attn", "Linear", "StreamsPost", "StreamsPre", "Moe", "StreamsPost", "StreamsHead"]
        );
        assert!(h
            .mats
            .iter()
            .flat_map(|it| &it.members)
            .filter(|n| !n.is_empty())
            .all(|n| n.starts_with("mtp.") || [&big.mats[big.embed].members[0], &big.mats[big.lm].members[0]].contains(&n)));
        assert_eq!((h.mats[h.embed].members.clone(), h.mats[h.lm].members.clone()), (big.mats[big.embed].members.clone(), big.mats[big.lm].members.clone()));
        let Some(Op::StreamsHead { norm, .. }) = h.layers[0].last() else { panic!() };
        assert_eq!(h.small[*norm], "mtp.hyper_connection_mixer.hc_norm.weight");
        super::super::head::fits(&big, &h).unwrap();
        super::super::head::shift_rings(&mut h, big.rings.len()).unwrap();
        let rings: Vec<usize> = h.layers[0].iter().flat_map(|op| op.rings()).collect();
        assert_eq!(rings, [7, 8, 8, 6]);
        // (a model without the MTP layer has no head)
        let mut plain = c.clone();
        plain["text_config"].as_object_mut().unwrap().remove("mtp_num_hidden_layers");
        assert!(super::super::lower::qwen4x_head(&describe(&plain).unwrap(), vec![2000], 1100).unwrap_err().msg.contains("names no draft head"));
    }

    /// A model whose experts the codebook kernels do not take (the tiny one: 128 neurons over 256) is refused by the expert cache's loader before it reads
    /// anything else, in words that name its widths and the kernels'.
    #[cfg(feature = "dev")]
    #[test]
    fn a_width_the_codebook_kernels_do_not_take_is_refused() {
        let dir = std::env::temp_dir().join(format!("glyd-qwen4x-cb-width-{}", std::process::id()));
        crate::logits::qwen4x_random(&dir).unwrap();
        let k =
            super::Caching { pack: dir.join("pack"), mode: super::PackMode::Fast, places: Some(1), offload: crate::offload::Offload::default(), reserve: 0 };
        let o = super::super::Opts { ctx: 256, chunk: 64, graph: false, device: 0, layout: None };
        let e = super::load_cached(&dir, &o, &k).err().map(|e| e.msg).unwrap_or_default();
        let _ = std::fs::remove_dir_all(&dir);
        assert!(e.contains("128 neurons over a width of 256") && e.contains("exactly 640 neurons over a width of 2560"), "{e}");
        assert!(crate::moe_qwen4x::takes(2560, 640).is_ok() && crate::moe_qwen4x::takes(2560, 1280).is_err());
    }

    /// What the shared operations do not take is refused in words: a linear attention's silu gate, gated residuals over more streams than the kernels take.
    #[test]
    fn what_the_kernels_do_not_take_is_refused() {
        let mut c = tiny();
        c["text_config"]["output_gate_type"] = "silu".into();
        let e = super::super::lower::qwen4x(&describe(&c).unwrap(), vec![2000], 1100).unwrap_err().msg;
        assert!(e.contains("sigmoid gate"), "{e}");
        let mut c = tiny();
        c["text_config"]["hc_count"] = 5.into();
        let e = super::super::lower::qwen4x(&describe(&c).unwrap(), vec![2000], 1100).unwrap_err().msg;
        assert!(e.contains("streams"), "{e}");
    }

    /// The family's stream count is its config's: two streams lower to residual slots of two rows of the width and the inject's two weights a row (a model with
    /// no n-gram embedding: that block's kernels take four streams yet).
    #[test]
    fn the_streams_are_the_configs() {
        let mut c = tiny();
        c["text_config"]["hc_count"] = 2.into();
        let e = super::super::lower::qwen4x(&describe(&c).unwrap(), vec![2000], 1100).unwrap_err().msg;
        assert!(e.contains("four streams"), "{e}");
        (c["text_config"]["ple_layer_ids"], c["text_config"]["number_of_conv_states"]) = (serde_json::json!([]), 1.into());
        let m = super::super::lower::qwen4x(&describe(&c).unwrap(), vec![2000], 1100).unwrap();
        assert!(!m.layers.iter().flatten().any(|op| matches!(op, Op::NgramRows(_))));
        assert_eq!((m.streams(), m.width(Slot::S), m.width(Slot::Kv)), (2, 512, 2));
        let four = super::super::lower::qwen4x(&describe(&tiny()).unwrap(), vec![2000], 1100).unwrap();
        assert_eq!((four.streams(), four.width(Slot::S), four.width(Slot::Kv)), (4, 1024, 4));
    }
}
