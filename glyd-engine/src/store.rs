//! A model ready to run, made on first use: the Hub checkpoint packed into the engine's cache (its layout the GPU's), and the KV cache's tables calibrated for it, each once.
//!
//! The cache is `$GLYD_CACHE`, else `$XDG_CACHE_HOME/glyd`, else `~/.cache/glyd`; a packed model lives in `engine/<model>-<revision>-<layout>/` and its tables in that directory
//! (`kv-tables-<positions>.safetensors`, the least tier that covers the context asked for). Both are made beside the final name and renamed into it, so a run that stops leaves nothing half made.

use crate::calib;
use crate::chat::Chat;
use crate::ffi::{err, Error, Result};
use crate::hub::{self, Entry, Hub, HubError, Remote};
use crate::kvcal;
use crate::model::{Engine, Opts};
use crate::tok::Tok;
use crate::ui::{fmt_time, gb, Bar, Done, Refusal, Ui};
use crate::weights::{choose_layout, Config, Layout};
use glyd_gpu::save::Source;
use std::cell::RefCell;
use std::path::{Path, PathBuf};

pub fn cache_root() -> PathBuf {
    if let Some(p) = std::env::var_os("GLYD_CACHE") {
        return p.into();
    }
    if let Some(p) = std::env::var_os("XDG_CACHE_HOME") {
        return Path::new(&p).join("glyd");
    }
    Path::new(&crate::os::home()).join(".cache").join("glyd")
}

/// A model made ready: its packed directory and its KV tables' file.
pub struct Prepared {
    pub dir: PathBuf,
    pub tables: PathBuf,
}

fn io<T>(r: std::io::Result<T>) -> Result<T> {
    r.map_err(|e| Error(e.to_string()))
}

/// The directory name of a packed model: the repo (or directory) name, the revision's first eight characters (where it has one) and the layout (the compact one, which glyd.json names as the tiered, `a`).
pub fn packed_name(source: &glyd_gpu::save::Source, layout: Layout) -> String {
    let base: String = source.repo.trim_end_matches('/').rsplit('/').take(2).collect::<Vec<_>>().into_iter().rev().collect::<Vec<_>>().join("--");
    let rev = source.revision.as_deref().map_or(String::new(), |r| format!("-{}", &r[..r.len().min(8)]));
    format!("{base}{rev}-{}", if layout == Layout::Compact { "a" } else { layout.name() })
}

/// The packed directory `dir` of a model, made where it is not there, and made again where it is of the tiered layout of glyd 0.27 and before (which the library no longer reads): `make` packs into the directory it is given,
/// which is renamed into `dir`; the KV tables of the old directory (made on the engine's own forward pass, and the weights unpack to the same bits in the new layout) are carried over to the new one.
fn ensure_packed(root: &Path, dir: &Path, model: &str, layout: &str, log: &dyn Fn(&str), make: &dyn Fn(&Path) -> Result<()>) -> Result<()> {
    let state = |d: &Path| if !d.join("engine.safetensors").exists() { None } else { Some(crate::weights::earlier_format(d).is_some()) };
    if state(dir) == Some(false) {
        return Ok(());
    }
    // one process makes it at a time
    let lock = io(std::fs::File::create(root.join(".lock")))?;
    io(lock.lock())?;
    let old = match state(dir) {
        Some(false) => return Ok(()),
        Some(true) => true,
        None => false,
    };
    if old {
        log(&format!("glyd: the packed copy of {model} is of the layout of glyd 0.27 and before, which this glyd does not read: packing it again for this GPU ({layout}), once: {}", dir.display()));
    } else {
        log(&format!("glyd: packing {model} for this GPU ({layout}), once: {}", dir.display()));
    }
    let tmp = root.join(format!(".packing-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    make(&tmp)?;
    if old {
        replace_old(dir, &tmp)?;
    } else {
        io(std::fs::rename(&tmp, dir))?;
    }
    Ok(())
}

/// The new packed directory `tmp` put in the place of the old one `dir`, the old one's KV tables carried over.
fn replace_old(dir: &Path, tmp: &Path) -> Result<()> {
    for e in io(std::fs::read_dir(dir))? {
        let e = io(e)?;
        let name = e.file_name().to_string_lossy().into_owned();
        if name.starts_with("kv-tables-") && name.ends_with(".safetensors") {
            io(std::fs::copy(e.path(), tmp.join(&name)))?;
        }
    }
    io(std::fs::remove_dir_all(dir))?;
    io(std::fs::rename(tmp, dir))
}

/// The least tables file in `dir` that covers `positions` (kv-tables-<tier>.safetensors), and the tier a new one would be made for.
pub fn tables_for(dir: &Path, positions: usize) -> Result<(Option<PathBuf>, usize)> {
    let want = kvcal::check_positions(positions)?;
    for t in kvcal::TIERS.iter().copied().filter(|&t| t >= want) {
        let p = dir.join(format!("kv-tables-{t}.safetensors"));
        if p.exists() {
            return Ok((Some(p), want));
        }
    }
    Ok((None, want))
}

/// `tables_for`, but a file of another version of the tables' rule (an earlier engine's) is deleted and not taken: the tables are made again.
fn current_tables_for(dir: &Path, positions: usize) -> Result<(Option<PathBuf>, usize)> {
    loop {
        let (found, tier) = tables_for(dir, positions)?;
        match found {
            Some(p) if !crate::kv::KvTables::is_current(&p) => {
                let _ = std::fs::remove_file(&p);
            }
            other => return Ok((other, tier)),
        }
    }
}

/// The model `model` (a packed directory, a Hub-format directory, or a Hub repo id in the local cache) packed and calibrated for contexts of `ctx` tokens: the cache's files where they are,
/// else made now (`log` says what is happening; a first run of an 8B model takes about two minutes).
pub fn prepare(model: &str, ctx: usize, layout: Option<Layout>, log: &dyn Fn(&str)) -> Result<Prepared> {
    let direct = Path::new(model);
    let packed = if direct.join("glyd.json").exists() {
        if let Some(why) = crate::weights::earlier_format(direct) {
            // (a directory the user keeps: not packed over)
            return err(format!("{model}: {why}: pack the model again (glyd pack MODEL DIR)"));
        }
        direct.to_path_buf()
    } else {
        let source = glyd_gpu::save::Source::find(model).map_err(|e| Error(e.to_string()))?;
        let cfg = Config::load(&source.dir)?;
        let layout = match layout {
            Some(l) => l,
            None => {
                let dev = crate::dev::Dev::new(0)?;
                let (l, why) = choose_layout(None, dev.cc_pair(), dev.total_mem as u64, cfg.matrix_bytes(), 0);
                log(&format!("glyd: layout {}: {why}", l.name()));
                l
            }
        };
        let root = cache_root().join("engine");
        io(std::fs::create_dir_all(&root))?;
        let dir = root.join(packed_name(&source, layout));
        ensure_packed(&root, &dir, model, layout.name(), log, &|tmp| crate::pack::run(model, tmp, 16, layout))?;
        dir
    };
    let (found, tier) = current_tables_for(&packed, ctx)?;
    let tables = match found {
        Some(p) => p,
        None => {
            let lock = io(std::fs::File::create(packed.join(".lock")))?;
            io(lock.lock())?;
            let (found, tier) = current_tables_for(&packed, ctx)?;
            match found {
                Some(p) => p,
                None => {
                    log(&format!("glyd: the lossless KV cache takes tables for {tier} positions of {model}, made once on the engine itself"));
                    let out = packed.join(format!("kv-tables-{tier}.safetensors"));
                    let o = Opts { ctx: tier, chunk: 4096, graph: false, ..Opts::default() };
                    let e = Engine::for_calibration(&packed, &o)?;
                    let tok = Tok::load(&packed)?;
                    let chat = Chat::load(&packed)?;
                    let t = calib::calibrate(&e, &tok, chat.as_ref(), &calib::Options { positions: tier, model: model.to_string() }, log)?;
                    let tmp = packed.join(format!(".tables-{}", std::process::id()));
                    calib::save(&tmp, &t)?;
                    io(std::fs::rename(&tmp, &out))?;
                    if let Some(why) = t.refused {
                        let _ = std::fs::remove_file(&out);
                        return err(why);
                    }
                    out
                }
            }
        }
    };
    let _ = tier;
    Ok(Prepared { dir: packed, tables })
}

// ---- the pieces of a first run, in the order `glyd run` takes them ------------------------------------------------------------------------------------------

const HUB: &str = "https://huggingface.co";

/// A Hub repo id: ORG/NAME, no more.
pub fn looks_like_repo(model: &str) -> bool {
    let ok = |p: &str| !p.is_empty() && p.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'));
    let mut it = model.split('/');
    matches!((it.next(), it.next(), it.next()), (Some(a), Some(b), None) if ok(a) && ok(b))
}

/// The Hub's refusal in words.
pub fn hub_refusal(model: &str) -> Refusal {
    Refusal::new(
        format!("the Hugging Face Hub would not show {model}: the name is wrong, or it is a gated or private model"),
        format!("Check the name (ORG/NAME, as in Qwen/Qwen3.5-9B). For a gated model, accept its licence at {HUB}/{model}, then run: glyd login   (it takes a token from {HUB}/settings/tokens)"),
    )
}

/// A checkpoint that is not bf16 (the Hub counts the parameters of each dtype; `weights` is the safetensors' bytes), refused in words: Glyd packs bf16 weights. `packed`: a repo whose glyd.json says
/// glyd-v7, whose buffers are bytes and words of packs, not weights.
pub fn check_dtypes(model: &str, weights: u64, dtypes: &[(String, u64)], packed: bool) -> Option<Refusal> {
    if packed || dtypes.is_empty() {
        return None;
    }
    let n = |k: &str| dtypes.iter().find(|(d, _)| d == k).map_or(0, |(_, n)| *n);
    let bf16 = 2 * (n("BF16") + n("F16"));
    if weights > bf16 && (weights - bf16) as f64 >= 0.05 * weights as f64 {
        if n("F32") * 4 > weights / 2 {
            return Some(Refusal::unsupported(format!("{model} is stored in 32-bit floats, and Glyd packs bf16 weights"), "Run a bf16 version of the model if there is one (most models publish one)"));
        }
        let kind = if n("F8_E4M3") > n("U8").max(n("I8")) { "FP8" } else { "4-bit" };
        return Some(Refusal::unsupported(format!("{model} is already quantized ({kind}), and Glyd packs bf16 weights"), "Run the model's bf16 version: usually the same name without -FP8, -AWQ or -GPTQ"));
    }
    if n("F16") > n("BF16") {
        return Some(Refusal::unsupported(format!("{model} is stored in float16, and Glyd packs bfloat16 weights"), "Use the model's bf16 version if it has one"));
    }
    None
}

/// The model's files on the Hub, or why not: the repo listed, its weights counted by dtype (a checkpoint that is not bf16 is refused here, before any download).
pub fn hub_plan(model: &str, ui: &RefCell<Ui>) -> Done<(Hub, Remote, Vec<Entry>)> {
    hub_plan_as(model, ui, false)
}

/// `hub_plan` of a release the dynamic engine runs as it is (`exact`: an FP8 or FP4 checkpoint is not refused; GLY-354).
pub fn hub_plan_as(model: &str, ui: &RefCell<Ui>, exact: bool) -> Done<(Hub, Remote, Vec<Entry>)> {
    if !looks_like_repo(model) {
        return Err(Refusal::new(format!("{model}: no such directory, and not a Hugging Face name"), "Give a directory with the model, or its Hub name (ORG/NAME, as in Qwen/Qwen3.5-9B)"));
    }
    if hub::offline() {
        return Err(Refusal::new(format!("{model} is not downloaded, and HF_HUB_OFFLINE is set"), "Unset HF_HUB_OFFLINE to download it"));
    }
    let hub = Hub::from_env();
    ui.borrow_mut().status("Reading the model's details...");
    let remote = match hub.list(model, "main") {
        Ok(r) => r,
        Err(HubError::Denied(_)) => return Err(hub_refusal(model)),
        Err(e) => return Err(Refusal::new(format!("cannot read {model} from the Hugging Face Hub ({e})"), "Check the network connection and run this again")),
    };
    let wanted = hub::wanted(&remote);
    let weights: u64 = wanted.iter().filter(|e| e.path.ends_with(".safetensors")).map(|e| e.size).sum();
    if weights == 0 {
        return Err(Refusal::unsupported(format!("{model} has no safetensors weights"), "Glyd reads .safetensors checkpoints, which most models on the Hub have"));
    }
    let packed = remote.files.iter().any(|e| e.path == "glyd.json") && hub.text(model, "main", "glyd.json").ok().is_some_and(|t| glyd_v7_level(&t).is_some());
    if let Some(r) = check_dtypes(model, weights, &remote.dtypes, packed || exact) {
        return Err(r);
    }
    Ok((hub, remote, wanted))
}

/// The level a glyd.json (text) names where it is a glyd-v7 save's.
fn glyd_v7_level(glyd_json: &str) -> Option<String> {
    let m = glyd_gpu::json::parse(glyd_json).ok()?;
    (m.get("format")?.as_str()? == "glyd-v7").then(|| m.get("level")?.as_str().map(str::to_string))?
}

/// The pre-packed repo `repo` of a model at `level` (kestrel or swift), planned as `hub_plan` plans a model, or None where the Hub has no such repo or it is not a glyd-v7 save of that level
/// (the caller packs the bf16 model here instead). A repo the Hub would not show is none: it answers 401 for a name that is not there as for a private one.
pub fn hub_plan_packed(repo: &str, level: &str, ui: &RefCell<Ui>) -> Done<Option<(Hub, Remote, Vec<Entry>)>> {
    if !looks_like_repo(repo) || hub::offline() {
        return Ok(None);
    }
    let hub = Hub::from_env();
    ui.borrow_mut().status("Looking for a pre-packed download...");
    let remote = match hub.list(repo, "main") {
        Ok(r) => r,
        Err(HubError::Denied(_)) => return Ok(None),
        Err(e) => return Err(Refusal::new(format!("cannot read {repo} from the Hugging Face Hub ({e})"), "Check the network connection and run this again")),
    };
    let wanted = packed_files(&remote, hub.text(repo, "main", "glyd.json").ok().as_deref(), level);
    Ok(wanted.map(|w| (hub, remote, w)))
}

/// The files to download of a repo whose glyd.json (text, where it has one) is a glyd-v7 save of `level`, else None.
fn packed_files(remote: &Remote, glyd_json: Option<&str>, level: &str) -> Option<Vec<Entry>> {
    (remote.files.iter().any(|e| e.path == "glyd.json") && glyd_v7_level(glyd_json?).as_deref() == Some(level)).then(|| hub::wanted(remote))
}

/// `entries` of `remote` in the cache, with one bar; where they are all there already, nothing is read.
pub fn download(hub: &Hub, remote: &Remote, entries: &[Entry], model: &str, ui: &RefCell<Ui>) -> Done<()> {
    let need = hub.missing(remote, entries);
    let total: u64 = entries.iter().map(|e| e.size).sum();
    if need > 0 {
        let mut probe = hub.cache().to_path_buf();
        while !probe.exists() && probe.pop() {}
        if let Some(free) = hub::disk_free(&probe) {
            if free < need + (1 << 30) {
                return Err(Refusal::new(
                    format!("the disk holding {} has {} left, and {} are to be downloaded", hub.cache().display(), gb(free), gb(need)),
                    format!("Free some space, or keep the models on another drive: HF_HOME=/path/on/that/drive glyd run {model}"),
                ));
            }
        }
    }
    let name = model.rsplit('/').next().unwrap_or(model);
    let bar = RefCell::new(Bar::new(&format!("Downloading {name}"), total));
    let shown = std::cell::Cell::new(false);
    let r = hub.fetch(remote, entries, &|done, _| {
        if need > 0 {
            shown.set(true);
            bar.borrow_mut().update(&mut ui.borrow_mut(), done, false);
        }
    });
    match r {
        Ok(_) => {}
        Err(HubError::Denied(_)) => return Err(hub_refusal(model)),
        Err(HubError::Network(e)) => return Err(Refusal::new(format!("the download of {model} stopped ({e})"), "Check the network connection and run the same command again: it resumes where it stopped")),
        Err(HubError::Other(e)) => return Err(Refusal::new(e, "")),
    }
    hub.pin(remote)?;
    if shown.get() {
        bar.borrow_mut().update(&mut ui.borrow_mut(), total, true);
        ui.borrow_mut().line(&format!("Downloaded {name} ({}).", gb(total)));
    } else {
        ui.borrow_mut().line(&format!("{name} is downloaded already."));
    }
    Ok(())
}

/// What a run that was stopped left in `dir` (the lock is held, so no other run is making anything): the entries whose names start with `prefix`.
fn remove_stale(dir: &Path, prefix: &str) {
    for e in std::fs::read_dir(dir).into_iter().flatten().flatten() {
        if e.file_name().to_string_lossy().starts_with(prefix) {
            let p = e.path();
            let _ = if p.is_dir() { std::fs::remove_dir_all(&p) } else { std::fs::remove_file(&p) };
        }
    }
}

/// The bytes in the files of a directory (not below it).
fn dir_bytes(dir: &Path) -> u64 {
    std::fs::read_dir(dir).map_or(0, |d| d.flatten().filter_map(|e| e.metadata().ok()).filter(|m| m.is_file()).map(|m| m.len()).sum())
}

/// The engine's cache of packed models.
pub fn packed_root() -> PathBuf {
    cache_root().join("engine")
}

/// Where `source` packed in `layout` is (or would be) in the cache.
pub fn packed_dir(source: &Source, layout: Layout) -> PathBuf {
    packed_root().join(packed_name(source, layout))
}

/// Whether `dir` holds a packed model of the layout of this glyd (one packed by glyd 0.27 and before is made again, once).
pub fn is_packed(dir: &Path) -> bool {
    dir.join("engine.safetensors").exists() && crate::weights::earlier_format(dir).is_none()
}

/// Whether `dir` holds a glyd-v7 save in `layout` (a lossy level's pack in the cache: it is there whole or not at all).
pub fn cached_pack(dir: &Path, layout: Layout) -> bool {
    matches!(crate::weights::v7_manifest(dir), Ok(Some((l, _))) if l == layout)
}

/// A failure of what the engine does for itself (its cache, its packing, its tables), which the other runner does not need: the status that hands the model to it.
fn cannot(e: impl std::fmt::Display) -> Refusal {
    Refusal::unsupported(e.to_string(), "")
}

/// `source` packed for this GPU into the cache (once; a run that stops leaves nothing half made), with a bar by the bytes written. `expect`: about what it comes to.
pub fn pack(source: &Source, model: &str, layout: Layout, expect: u64, ui: &RefCell<Ui>) -> Done<PathBuf> {
    let root = packed_root();
    let io = |r: std::io::Result<()>| r.map_err(|e| Refusal::unsupported(format!("{}: {e}", root.display()), "Check that the folder can be written, or set GLYD_CACHE to another"));
    io(std::fs::create_dir_all(&root))?;
    let dir = packed_dir(source, layout);
    // (a packed copy of the layout of glyd 0.27 and before is made again, once; a lossy level's is a glyd-v7 directory, which a rename puts in its place whole)
    let lossy = matches!(layout, Layout::Int6 | Layout::Int5 | Layout::Int4 | Layout::Int3);
    let have = |d: &Path| if lossy { cached_pack(d, layout) } else { is_packed(d) };
    if have(&dir) {
        return Ok(dir);
    }
    if let Some(free) = hub::disk_free(&root) {
        if free < expect + (1 << 30) {
            return Err(Refusal::unsupported(format!("the disk holding {} has {} left, and the packed model takes {}", root.display(), gb(free), gb(expect)), "Free some space, or put the cache on another drive: GLYD_CACHE=/path/on/that/drive"));
        }
    }
    // one process makes it at a time
    let lock = std::fs::File::create(root.join(".lock")).map_err(cannot)?;
    lock.lock().map_err(cannot)?;
    if have(&dir) {
        return Ok(dir);
    }
    let old = !lossy && dir.join("engine.safetensors").exists();
    let name = model.rsplit('/').next().unwrap_or(model);
    if old {
        ui.borrow_mut().line(&format!("The packed copy of {name} is of the layout of glyd 0.27 and before, which this glyd does not read: packing it again for this GPU ({}), once: {}", layout.name(), dir.display()));
    } else {
        ui.borrow_mut().line(&format!("Packing {name} for this GPU ({}), once: {}", layout.name(), dir.display()));
    }
    remove_stale(&root, ".packing-");
    let tmp = root.join(format!(".packing-{}", std::process::id()));
    let t0 = std::time::Instant::now();
    let mut bar = Bar::new(&format!("Packing {name}"), expect);
    let result = std::thread::scope(|s| {
        let work = s.spawn(|| crate::pack::run_with(model, &tmp, 16, layout, true));
        while !work.is_finished() {
            bar.update(&mut ui.borrow_mut(), dir_bytes(&tmp), false);
            std::thread::sleep(std::time::Duration::from_millis(250));
        }
        work.join().unwrap_or_else(|_| err("the packing thread panicked"))
    });
    if let Err(e) = result {
        let _ = std::fs::remove_dir_all(&tmp);
        return Err(cannot(format!("packing {model} stopped: {e}")));
    }
    bar.update(&mut ui.borrow_mut(), expect, true);
    if old {
        replace_old(&dir, &tmp).map_err(cannot)?;
    } else {
        io(std::fs::rename(&tmp, &dir))?;
    }
    ui.borrow_mut().line(&format!("Packed in {} ({}).", fmt_time(t0.elapsed().as_secs_f64()), gb(dir_bytes(&dir))));
    Ok(dir)
}

/// What the tables' rule said of the model packed in `dir`, where it refused it for a context of `tier` positions (its values compress too poorly for the lossless KV cache), as it was written when
/// the tables were made: the reason. A mark written under another version of the rule is not taken.
pub fn known_refusal(dir: &Path, tier: usize) -> Option<String> {
    let text = std::fs::read_to_string(dir.join(format!("kv-refused-{tier}"))).ok()?;
    let (version, why) = text.split_once('\n')?;
    (version.trim() == crate::kvcal::VERSION.to_string()).then(|| why.trim().to_string())
}

pub(crate) fn note_refusal(dir: &Path, tier: usize, why: &str) {
    let _ = std::fs::write(dir.join(format!("kv-refused-{tier}")), format!("{}\n{why}\n", crate::kvcal::VERSION));
}

/// The KV cache's tables for `packed` that cover `ctx` positions: the cache's file where there is one, else made now on the engine itself (once), on GPU `device`. Where the rule refuses the model
/// (its values compress too poorly) that is written down beside the packed copy, so that the next run knows before it loads anything; every failure here is a refusal of status 3, for the engine
/// is not the only way to run the model.
pub fn tables(packed: &Path, model: &str, ctx: usize, device: i32, ui: &RefCell<Ui>) -> Done<PathBuf> {
    let (found, tier) = current_tables_for(packed, ctx).map_err(cannot)?;
    if let Some(p) = found {
        return Ok(p);
    }
    if let Some(why) = known_refusal(packed, tier) {
        return Err(cannot(why));
    }
    let lock = std::fs::File::create(packed.join(".lock")).map_err(cannot)?;
    lock.lock().map_err(cannot)?;
    if let Some(p) = current_tables_for(packed, ctx).map_err(cannot)?.0 {
        return Ok(p);
    }
    let name = model.rsplit('/').next().unwrap_or(model);
    remove_stale(packed, ".tables-");
    ui.borrow_mut().line(&format!("Making the lossless KV cache's tables for {name} ({tier} positions), once: a few minutes at most."));
    let t0 = std::time::Instant::now();
    let out = packed.join(format!("kv-tables-{tier}.safetensors"));
    let o = Opts { ctx: tier, chunk: 4096, graph: false, device, ..Opts::default() };
    let e = Engine::for_calibration(packed, &o).map_err(cannot)?;
    let tok = Tok::load(packed).map_err(cannot)?;
    let chat = Chat::load(packed).map_err(cannot)?;
    let t = calib::calibrate(&e, &tok, chat.as_ref(), &calib::Options { positions: tier, model: model.to_string() }, &|s| ui.borrow_mut().status(s)).map_err(cannot)?;
    let tmp = packed.join(format!(".tables-{}", std::process::id()));
    calib::save(&tmp, &t).map_err(cannot)?;
    if let Some(why) = t.refused {
        let _ = std::fs::remove_file(&tmp);
        note_refusal(packed, tier, &why);
        return Err(cannot(why));
    }
    std::fs::rename(&tmp, &out).map_err(cannot)?;
    ui.borrow_mut().line(&format!("Tables made in {}.", fmt_time(t0.elapsed().as_secs_f64())));
    Ok(out)
}

/// The nearest folder of `path` that exists (a disk is asked about that one).
pub fn existing_ancestor(path: &Path) -> PathBuf {
    let mut p = path.to_path_buf();
    while !p.exists() && p.pop() {}
    p
}

/// Whether two folders that exist are on one filesystem.
#[cfg(unix)]
pub fn same_filesystem(a: &Path, b: &Path) -> bool {
    use std::os::unix::fs::MetadataExt;
    matches!((std::fs::metadata(a), std::fs::metadata(b)), (Ok(x), Ok(y)) if x.dev() == y.dev())
}

/// Whether two folders that exist are on one volume (Windows: the same drive or share at the start of their full paths).
#[cfg(windows)]
pub fn same_filesystem(a: &Path, b: &Path) -> bool {
    let root = |p: &Path| std::fs::canonicalize(p).ok().and_then(|c| c.components().next().map(|x| x.as_os_str().to_ascii_lowercase()));
    matches!((root(a), root(b)), (Some(x), Some(y)) if x == y)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checkpoints_that_are_not_bf16_are_refused() {
        let d = |v: &[(&str, u64)]| v.iter().map(|(k, n)| (k.to_string(), *n)).collect::<Vec<_>>();
        // Qwen3-8B: 8.19 B parameters in bf16, 16.4 GB
        assert!(check_dtypes("Qwen/Qwen3-8B", 16_381_470_720, &d(&[("BF16", 8_190_735_360)]), false).is_none());
        assert!(check_dtypes("x", 1, &[], false).is_none());
        let fp8 = check_dtypes("Qwen/Qwen3-8B-FP8", 8_900_000_000, &d(&[("F8_E4M3", 6_950_000_000), ("BF16", 1_240_000_000)]), false).unwrap();
        assert!(fp8.what.contains("FP8"), "{}", fp8.what);
        let q4 = check_dtypes("x-AWQ", 5_700_000_000, &d(&[("I32", 1_100_000_000), ("F16", 600_000_000)]), false).unwrap();
        assert!(q4.what.contains("4-bit"), "{}", q4.what);
        let f32 = check_dtypes("x", 32_000_000_000, &d(&[("F32", 8_000_000_000)]), false).unwrap();
        assert!(f32.what.contains("32-bit"), "{}", f32.what);
        let f16 = check_dtypes("x", 16_000_000_000, &d(&[("F16", 8_000_000_000)]), false).unwrap();
        assert!(f16.what.contains("float16"), "{}", f16.what);
    }

    #[test]
    fn what_a_stopped_run_left_goes() {
        let dir = std::env::temp_dir().join(format!("glyd-engine-stale-{}", std::process::id()));
        std::fs::create_dir_all(dir.join(".packing-1234")).unwrap();
        std::fs::write(dir.join(".packing-1234").join("shard"), b"x").unwrap();
        std::fs::write(dir.join(".tables-77"), b"x").unwrap();
        std::fs::create_dir_all(dir.join("Qwen--Qwen3-8B-mma")).unwrap();
        remove_stale(&dir, ".packing-");
        assert!(!dir.join(".packing-1234").exists() && dir.join(".tables-77").exists() && dir.join("Qwen--Qwen3-8B-mma").exists());
        remove_stale(&dir, ".tables-");
        assert!(!dir.join(".tables-77").exists());
        std::fs::remove_dir_all(&dir).unwrap();
        assert!(looks_like_repo("Qwen/Qwen3-8B") && looks_like_repo("a-b/c_d.e") && !looks_like_repo("Qwen3-8B") && !looks_like_repo("a/b/c") && !looks_like_repo("/models/x"));
    }

    #[test]
    fn names_of_packed_models() {
        let s = glyd_gpu::save::Source { dir: "/x".into(), repo: "Qwen/Qwen3-8B".into(), revision: Some("b968826d9c46dd6066d109eabc6255188de91218".into()) };
        assert_eq!(packed_name(&s, Layout::Tiered), "Qwen--Qwen3-8B-b968826d-mma");
        let d = glyd_gpu::save::Source { dir: "/models/mine".into(), repo: "/models/mine".into(), revision: None };
        assert_eq!(packed_name(&d, Layout::Twelve), "models--mine-mma12");
        // each layout its own directory: the compact one is not the tiered one's, and the lossy levels have theirs
        let names: Vec<String> = [Layout::Tiered, Layout::Twelve, Layout::Compact, Layout::Int6, Layout::Int5, Layout::Int4, Layout::Int3].iter().map(|&l| packed_name(&s, l)).collect();
        let want = ["mma", "mma12", "a", "i6", "i5", "i4", "i3"].map(|l| format!("Qwen--Qwen3-8B-b968826d-{l}"));
        assert_eq!(names, want);
    }

    /// A glyd-v7 repo is taken whole (its buffers are not weights); the same counts of an unknown 4-bit repo are still refused.
    #[test]
    fn a_pre_packed_repo_is_planned_and_its_buffers_pass() {
        let d = |v: &[(&str, u64)]| v.iter().map(|(k, n)| (k.to_string(), *n)).collect::<Vec<_>>();
        let u8s = d(&[("U8", 7_000_000_000), ("I32", 300_000_000), ("BF16", 1_000_000_000)]);
        assert!(check_dtypes("glyd/Qwen3.5-9B-swift", 6_000_000_000, &u8s, true).is_none());
        let q4 = check_dtypes("x-AWQ", 5_700_000_000, &d(&[("I32", 1_100_000_000), ("F16", 600_000_000)]), false).unwrap();
        assert!(q4.what.contains("4-bit"), "{}", q4.what);
        let e = |p: &str, size| Entry { path: p.to_string(), size, blob: p.to_string(), sha256: None };
        let remote = |files: Vec<Entry>| Remote { repo: "glyd/m-swift".into(), revision: "main".into(), commit: "c".repeat(40), files, dtypes: vec![] };
        let files = vec![e("glyd.json", 9), e("config.json", 9), e("model-00001-of-00002.safetensors", 100), e("sub/x.safetensors", 5)];
        let v7 = |level: &str| format!("{{\"format\": \"glyd-v7\", \"layout\": \"i5\", \"level\": \"{level}\"}}");
        let paths = |w: Option<Vec<Entry>>| w.map(|w| w.iter().map(|e| e.path.clone()).collect::<Vec<_>>());
        assert_eq!(paths(packed_files(&remote(files.clone()), Some(&v7("swift")), "swift")), Some(vec!["glyd.json".to_string(), "config.json".into(), "model-00001-of-00002.safetensors".into()]));
        // the Hub's repo falls back to the bf16 model: another level, another format, no glyd.json, one that cannot be read
        assert!(packed_files(&remote(files.clone()), Some(&v7("kestrel")), "swift").is_none());
        assert!(packed_files(&remote(files.clone()), Some("{\"format\": \"glyd-v4\", \"level\": \"swift\"}"), "swift").is_none());
        assert!(packed_files(&remote(files[1..].to_vec()), Some(&v7("swift")), "swift").is_none());
        assert!(packed_files(&remote(files), None, "swift").is_none());
    }

    /// A lossy level's pack is in the cache when its directory is a glyd-v7 save of that layout, and not for the other level's.
    #[test]
    fn a_cached_lossy_pack_is_found_by_its_layout() {
        let dir = std::env::temp_dir().join(format!("glyd-engine-cached-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        assert!(!cached_pack(&dir, Layout::Int5));
        std::fs::write(dir.join("glyd.json"), "{\"format\": \"glyd-v7\", \"layout\": \"i5\", \"level\": \"swift\", \"packs\": {}}").unwrap();
        assert!(cached_pack(&dir, Layout::Int5) && !cached_pack(&dir, Layout::Int6) && !cached_pack(&dir, Layout::Int4));
        std::fs::write(dir.join("glyd.json"), "{\"format\": \"glyd-v7\", \"layout\": \"i4\", \"level\": \"wren\", \"packs\": {}}").unwrap();
        assert!(cached_pack(&dir, Layout::Int4) && !cached_pack(&dir, Layout::Int5) && !cached_pack(&dir, Layout::Int3));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn tiers_found_and_wanted() {
        let dir = std::env::temp_dir().join(format!("glyd-engine-store-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        assert_eq!(tables_for(&dir, 3000).unwrap(), (None, 4096));
        assert_eq!(tables_for(&dir, 5000).unwrap(), (None, 8192));
        std::fs::write(dir.join("kv-tables-16384.safetensors"), b"x").unwrap();
        // a table for more positions serves a smaller context
        assert_eq!(tables_for(&dir, 5000).unwrap(), (Some(dir.join("kv-tables-16384.safetensors")), 8192));
        assert!(tables_for(&dir, 99999).is_err());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// A model the tables' rule refused is known so, beside its packed copy, for the context's tier it was refused at, and under the version of the rule that refused it: the next run says so
    /// before it makes anything, and its status hands the model to vLLM.
    #[test]
    fn a_refused_model_is_known_before_its_tables_are_made_again() {
        let dir = std::env::temp_dir().join(format!("glyd-engine-refused-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        assert_eq!(known_refusal(&dir, 4096), None);
        let why = "the model's values compress too poorly (worst kind of text left out: K 0.210%, V 0.050%, the limit 0.15%)";
        note_refusal(&dir, 4096, why);
        assert_eq!(known_refusal(&dir, 4096).as_deref(), Some(why));
        assert_eq!(known_refusal(&dir, 8192), None); // (another tier is a table of its own)
        // tables the rule would make now: the model is not known by what an earlier version said
        std::fs::write(dir.join("kv-refused-16384"), format!("{}\n{why}\n", crate::kvcal::VERSION - 1)).unwrap();
        assert_eq!(known_refusal(&dir, 16384), None);
        // asking for the tables: the refusal, with no GPU touched
        let ui = RefCell::new(Ui::silent());
        let r = tables(&dir, "org/model", 3000, 0, &ui).unwrap_err();
        assert!(r.code == crate::ui::CANNOT && r.what == why, "{r}");
        // where the tables exist, they are the answer
        std::fs::remove_file(dir.join("kv-refused-4096")).unwrap();
        std::fs::write(dir.join("kv-tables-4096.safetensors"), b"x").unwrap();
        crate::pack::write_safetensors(&dir.join("kv-tables-4096.safetensors"), &[("format", "glyd-kv-tables-v0".to_string()), ("meta", format!("{{\"version\":{}}}", crate::kvcal::VERSION))], &[("k", "U8", vec![1, 1, 1], &[0u8][..])]).unwrap();
        assert_eq!(tables(&dir, "org/model", 3000, 0, &ui).unwrap(), dir.join("kv-tables-4096.safetensors"));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn folders_and_filesystems() {
        let dir = std::env::temp_dir().join(format!("glyd-engine-fs-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        assert_eq!(existing_ancestor(&dir.join("a").join("b")), dir);
        assert_eq!(existing_ancestor(&dir), dir);
        assert!(same_filesystem(&dir, &std::env::temp_dir()));
        assert!(!same_filesystem(&dir, &dir.join("missing")));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_checkpoint_that_does_not_run_is_for_the_other_runner() {
        let d = |v: &[(&str, u64)]| v.iter().map(|(k, n)| (k.to_string(), *n)).collect::<Vec<_>>();
        for r in [
            check_dtypes("x-FP8", 8_900_000_000, &d(&[("F8_E4M3", 6_950_000_000), ("BF16", 1_240_000_000)]), false).unwrap(),
            check_dtypes("x", 32_000_000_000, &d(&[("F32", 8_000_000_000)]), false).unwrap(),
            check_dtypes("x", 16_000_000_000, &d(&[("F16", 8_000_000_000)]), false).unwrap(),
        ] {
            assert_eq!(r.code, crate::ui::CANNOT, "{r}");
        }
    }

    #[test]
    fn tables_of_another_version_are_made_again() {
        let dir = std::env::temp_dir().join(format!("glyd-engine-store-v-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let write = |name: &str, version: u32| {
            let meta = format!("{{\"positions\":4096,\"model\":\"m\",\"version\":{version}}}");
            crate::pack::write_safetensors(&dir.join(name), &[("format", "glyd-kv-tables-v0".to_string()), ("meta", meta)], &[("k", "U8", vec![1, 1, 1], &[0u8][..])]).unwrap();
        };
        write("kv-tables-4096.safetensors", crate::kvcal::VERSION - 1);
        write("kv-tables-16384.safetensors", crate::kvcal::VERSION);
        // the earlier engine's file is deleted, and the one that is current serves the context
        assert_eq!(current_tables_for(&dir, 3000).unwrap(), (Some(dir.join("kv-tables-16384.safetensors")), 4096));
        assert!(!dir.join("kv-tables-4096.safetensors").exists());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// A packed directory as a few files hold it: glyd.json (one pack, its tiers' first word as given), the sidecar with the embedding's metadata, and any more files.
    fn fake_packed(dir: &Path, tiers0: u32, extra: &[(&str, &str)]) {
        std::fs::create_dir_all(dir).unwrap();
        let format = if tiers0 >> 24 == 0xFF { "glyd-v1" } else { "glyd-v4" };
        std::fs::write(dir.join("glyd.json"), format!("{{\"format\": \"{format}\", \"layout\": \"mma\", \"packs\": {{\"model.layers.0.self_attn.q_proj\": {{\"layout\": \"mma\", \"shape\": [64, 16], \"tiers\": [{tiers0}, 0, 0]}}}}}}")).unwrap();
        let (a, b): (Vec<u8>, Vec<u8>) = (vec![0u8; 4], vec![0u8; 4]);
        let meta = format!("{{\"shape\":[64,16],\"tiers\":[{tiers0},0,0]}}");
        crate::pack::write_safetensors(&dir.join("engine.safetensors"), &[("format", "x".to_string()), ("embed", meta)], &[("embed.glyd_data", "U8", vec![4], &a[..]), ("embed.glyd_blocks", "U8", vec![4], &b[..])]).unwrap();
        for (name, text) in extra {
            std::fs::write(dir.join(name), text).unwrap();
        }
    }

    /// A model packed in the layout of glyd 0.27 and before is packed again, once, where it was packed for an earlier run: the new directory takes the old one's place, with the old one's KV tables; one that is of the
    /// layout of this glyd is left as it is, and one that is not there is made.
    #[test]
    fn a_packed_copy_of_the_earlier_layout_is_made_again_once_and_keeps_its_tables() {
        let root = std::env::temp_dir().join(format!("glyd-engine-repack-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        let dir = root.join("m-mma");
        let said = RefCell::new(Vec::<String>::new());
        let calls = std::cell::Cell::new(0);
        let make = |tmp: &Path| -> Result<()> {
            calls.set(calls.get() + 1);
            fake_packed(tmp, 2 << 24 | 5, &[]);
            Ok(())
        };
        let log = |s: &str| said.borrow_mut().push(s.to_string());
        // an earlier run's copy, of the earlier layout, with the tables it made
        fake_packed(&dir, 0xFF << 24 | 5, &[("kv-tables-4096.safetensors", "the tables"), ("kv-tables-16384.safetensors", "more tables"), ("notes.txt", "not carried")]);
        assert!(crate::weights::earlier_format(&dir).is_some());
        ensure_packed(&root, &dir, "m", "mma", &log, &make).unwrap();
        assert_eq!(calls.get(), 1);
        assert!(said.borrow()[0].contains("0.27 and before") && said.borrow()[0].contains("packing it again"), "{:?}", said.borrow());
        assert_eq!(crate::weights::earlier_format(&dir), None, "the new directory is the one in its place");
        assert_eq!(std::fs::read_to_string(dir.join("kv-tables-4096.safetensors")).unwrap(), "the tables");
        assert_eq!(std::fs::read_to_string(dir.join("kv-tables-16384.safetensors")).unwrap(), "more tables");
        assert!(!dir.join("notes.txt").exists());
        assert!(!root.join(format!(".packing-{}", std::process::id())).exists());
        // a second run finds it made
        ensure_packed(&root, &dir, "m", "mma", &log, &make).unwrap();
        assert_eq!(calls.get(), 1, "packed once");
        // a model that was never packed is made, with no tables
        let fresh = root.join("n-mma");
        ensure_packed(&root, &fresh, "n", "mma", &log, &make).unwrap();
        assert_eq!(calls.get(), 2);
        assert!(said.borrow()[1].contains("packing n for this GPU") && fresh.join("glyd.json").exists());
        // a failed packing leaves the old directory where it was
        fake_packed(&root.join("o-mma"), 0xFF << 24 | 5, &[("kv-tables-4096.safetensors", "t")]);
        let failing = |_: &Path| -> Result<()> { err("the disk is full") };
        assert!(ensure_packed(&root, &root.join("o-mma"), "o", "mma", &log, &failing).is_err());
        assert!(crate::weights::earlier_format(&root.join("o-mma")).is_some() && root.join("o-mma").join("kv-tables-4096.safetensors").exists());
        std::fs::remove_dir_all(&root).unwrap();
    }
}
