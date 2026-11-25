//! The commands a person runs: `glyd-engine run MODEL` (the first run makes the model ready, then a chat in the terminal and at one URL, or one answer with --prompt), `serve MODEL` (the API
//! and the page), `doctor`, `login` and `supports MODEL`. `glyd run` and `glyd serve` start these.

use crate::args::Args;
use crate::chat::{Chat, Message};
use crate::conv::{Gen, Part, Request, Splitter, Talk};
use crate::dev::{Dev, Gpu};
use crate::gpu_ops::GpuOps;
use crate::hub::{self, Hub};
use crate::levels::{self, is_qat, layout_for, QAT_WREN};
pub use crate::levels::{packed_repo, ship_layout, short_name};
use crate::machine;
use crate::offload::ExpertStore;
use crate::plan::sizing::{bits, lossy_plan, qwen3_plan, rooms, size, size_dynamic, static_family, target};
use crate::plan::{Kind, LoadPlan, Target, HOST_MARGIN, LOSSY_CTX};
use crate::sampler::{self, Params};
use crate::server::{self, Job, Piece};
use crate::status::{Status, STATUS};
use crate::store;
use crate::tok::Tok;
use crate::ui::{fmt_time, gb, gib, Done, Refusal, Ui};
use crate::weights::{v7_manifest, Config, Layout};
use glyd_gpu::json::{self, Value};
use glyd_gpu::save::Source;
use std::cell::RefCell;
use std::io::{BufRead, IsTerminal, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::Arc;
use std::time::Instant;

const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Ctrl-C: it stops the answer being written (and returns to the prompt) where a reply is running, else it leaves.
mod sig {
    use std::sync::atomic::{AtomicBool, Ordering};
    pub static BUSY: AtomicBool = AtomicBool::new(false);
    pub static INTERRUPT: AtomicBool = AtomicBool::new(false);
    extern "C" {
        fn signal(sig: i32, handler: usize) -> usize;
        fn _exit(code: i32) -> !;
        #[cfg_attr(windows, link_name = "_write")]
        fn write(fd: i32, buf: *const u8, n: usize) -> isize;
    }
    extern "C" fn on_int(_: i32) {
        if BUSY.load(Ordering::Relaxed) {
            INTERRUPT.store(true, Ordering::Relaxed);
        } else {
            // SAFETY: async-signal-safe calls only (write and _exit).
            unsafe {
                write(2, b"\n".as_ptr(), 1);
                _exit(130)
            }
        }
    }
    pub fn install() {
        // SAFETY: SIGINT's handler, a function that does nothing but the calls above.
        unsafe {
            signal(2, on_int as *const () as usize);
        }
    }
}

fn out(s: &str) {
    let mut o = std::io::stdout();
    let _ = o.write_all(s.as_bytes());
    let _ = o.flush();
}

fn dim(s: &str, on: bool) -> String {
    if on {
        format!("\x1b[2m{s}\x1b[0m")
    } else {
        s.to_string()
    }
}

/// The level MODEL asks for, as a refusal where its tag is no level this build runs (src/levels.rs has the table and the rules).
pub(crate) fn level(model: &str, lossy: bool) -> Done<(&str, Option<Layout>)> {
    levels::parse(model, lossy).map_err(|e| Refusal::new(e.what, e.fix))
}

/// Whether MODEL is a Gemma 4 model: a directory whose config says so, or a Hub name of one (google/gemma-4-..., glyd/gemma-4-...).
fn is_gemma4_model(model: &str) -> bool {
    match std::fs::read_to_string(Path::new(model).join("config.json")) {
        Ok(text) => crate::gemma4::is_gemma4(&text),
        Err(_) => short_name(model).to_ascii_lowercase().starts_with("gemma-4"),
    }
}

/// A level's name from its layout.
pub(crate) fn level_name(layout: Layout) -> &'static str {
    levels::name_of(layout)
}

/// The wren repo of a Gemma 4 model that has one (the instruction-tuned 12B and 31B, packed exact from Google's QAT checkpoint), from its Hub name or its penguin repo's.
pub(crate) fn gemma_wren(model: &str) -> Option<String> {
    let repo = match model.strip_prefix("glyd/").and_then(|n| n.strip_suffix("-penguin")) {
        Some(n) => format!("glyd/{n}-wren"),
        None if is_qat(model) => packed_repo(model, "wren"),
        None => return None,
    };
    QAT_WREN.iter().any(|m| packed_repo(m, "wren") == repo).then_some(repo)
}

/// The layout of MODEL where it is a local glyd-v7 directory (a lossy level's packs, saved).
fn v7_layout(model: &str) -> Done<Option<Layout>> {
    Ok(v7_manifest(Path::new(model)).map_err(cannot)?.map(|m| m.0))
}

/// The level a run takes where MODEL is a glyd-v7 directory (`have`: its layout): its own where none is asked, else the one asked if it is the same, or refused naming both.
fn v7_level(model: &str, have: Option<Layout>, asked: Option<Layout>) -> Done<Option<Layout>> {
    match (have, asked) {
        (Some(h), Some(a)) if h != a && !(h == Layout::Int4H && a == Layout::Int4) => {
            Err(Refusal::new(format!("{model} is packed at the {} level, not :{}", level_name(h), level_name(a)), format!("Run it as it is: glyd run {model}")))
        }
        (Some(h), _) => Ok(Some(h)),
        (None, a) => Ok(a),
    }
}

/// Whether MODEL is a plain dense decoder (the dense kind on the executor: src/dynamic/dense.rs), by its config.json alone (`config_text`: the directory's, the
/// cache's, else the Hub's). A name says nothing of the kind: where the config is not to be had the model has no level here, and the path that reads it says why.
fn is_dense_model(model: &str) -> bool {
    config_text(model).is_ok_and(|text| crate::dynamic::dense::is_dense(&text))
}

/// The level run, serve and preflight take MODEL at: a glyd-v7 directory's own, else the one asked; a Gemma 4 model, a plain dense decoder or a family of the dynamic route
/// without one is penguin (each runs on the description executor, its packs lossless: the levels' route with the lossless layout).
fn run_level(model: &str, asked: Option<Layout>) -> Done<Option<Layout>> {
    Ok(v7_level(model, v7_layout(model)?, asked)?
        .or_else(|| (is_gemma4_model(model) || dynamic_family(model).is_some() || is_dense_model(model)).then_some(Layout::Tiered)))
}

/// The refusal of MODEL where no engine of this build takes it (`run_level` gives it no level): what `supported` says of its config, in the words the program that hands
/// a model to another runner reads.
fn unrun(model: &str) -> Refusal {
    let name = short_name(model);
    match config_text(model).and_then(|text| supported(&name, &text)) {
        Err(r) => r,
        Ok(()) => Refusal::unsupported(format!("{name}: no engine of this build runs it"), unrun_fix()),
    }
}

/// The level `fit` plans for MODEL on this machine (plan.rs `planned`, with this command's own flags: --users, --ctx), as a layout: penguin's is the lossless one. None where the planner
/// chooses nothing (the run's own refusal says why) or cannot read the model now.
#[cfg(test)]
fn planner_level(a: &Args, model: &str) -> Option<Layout> {
    let chosen = planner(a, model)?.plan.chosen?;
    Some(levels::by_name(chosen.level)?.layout.unwrap_or(Layout::Tiered))
}

/// `fit`'s plan of MODEL with this command's flags, or None where it cannot be made now (no config to read: the run's own paths say why).
fn planner(a: &Args, model: &str) -> Option<crate::plan::Planned> {
    let mut q = a.clone();
    if let Some(m) = q.pos.get_mut(1) {
        *m = model.to_string();
    }
    crate::plan::planned(&q).ok()
}

/// The level run, serve and preflight load MODEL at (`run_level`), where nothing is asked the one `fit` printed: a Gemma 4 model or a family of the dynamic route takes
/// penguin by default (`run_level`), which a 16 GB card cannot hold for Gemma 4 12B; the planner's level is what the printed plan said would load (P16, #221). A tag, --lossy,
/// a glyd-v7 directory and a pre-packed repo are their own level.
/// A family whose default is penguin takes the planner's level where it is a lossy one: Qwen3.8-27B on a 24 GB card is swift (the M1 gate on Windows, an A10G), its penguin
/// needing two such cards or a larger one; a model the planner keeps lossless stays at penguin.
fn load_level(a: &Args, model: &str, asked: Option<Layout>) -> Done<Option<Layout>> {
    let layout = run_level(model, asked)?;
    // (a tag is its own level, `:penguin` too, whose layout is the untagged one's: the ladder, not the planner, settles what a tagged model that does not fit loads at)
    let tagged = a.pos.get(1).is_some_and(|m| levels::split_tag(m).1.is_some());
    if asked.is_none()
        && !tagged
        && !a.has("lossy")
        && (layout.is_none() || layout == Some(Layout::Tiered))
        && v7_layout(model)?.is_none()
        && levels::repo_level(model).is_none()
    {
        if let Some(p) = planner(a, model) {
            match p.plan.chosen.as_ref().and_then(|c| levels::by_name(c.level)) {
                Some(l) => {
                    let planned = l.layout.unwrap_or(Layout::Tiered);
                    if layout.is_some() || planned != Layout::Tiered {
                        return Ok(Some(planned));
                    }
                }
                // nothing fits this machine at any level the family runs: fit's refusal, before anything is downloaded (on a disk large enough, GLM-5.3-Flash's 328 GB
                // was downloaded and refused after; on Windows, whose free space was not read, until the disk was full)
                // (a problem of the machine itself, no GPU or a driver too old, is left to the run's own checks, which name its fix: "no NVIDIA driver", the driver to
                // install; the planner words them otherwise, and glyd-gpu's install check reads run's)
                None => {
                    if let Some(r) = p.refusal.filter(|_| crate::support::machine(&p.machine).is_none()) {
                        return Err(r);
                    }
                }
            }
        }
    }
    Ok(layout)
}

/// The family of a model the dynamic engine runs from its release (DeepSeek-V4, GLM-5.3-Flash, Muse Spark, Nemotron-H, the Qwen3.5 layout), by
/// its config; None for another path's, a glyd-v7 directory's, or a config not to be had (offline).
fn dynamic_family(model: &str) -> Option<&'static str> {
    if v7_layout(model).ok().flatten().is_some() {
        return None;
    }
    crate::dynamic::split::family(&config_text(model).ok()?)
}

/// Whether a family of the dynamic route takes `layout` on the one-GPU path of the lossy levels instead (`load_lossy`'s own): every level but penguin of a
/// family whose row has lossy levels (Nemotron-H, the Qwen3.5 layout); the others run penguin and hummingbird on the dynamic route alone.
fn lossy_route(family: &str, layout: Layout) -> bool {
    layout != Layout::Tiered && crate::support::row_named(family).is_some_and(|r| r.allows("kestrel"))
}

// ---- the GPU --------------------------------------------------------------------------------------------------------------------------------------------

/// A failure of what this engine does that vLLM does not need: the status that hands the model to it.
pub(crate) fn cannot(e: impl std::fmt::Display) -> Refusal {
    Refusal::unsupported(e.to_string(), "")
}

/// The GPU a run uses: the one with the most free memory of those the driver shows (CUDA_VISIBLE_DEVICES applied), as the earlier release chose; the contexts the look at the others made are let go.
fn gpu_pick() -> Done<Gpu> {
    let gpus = Dev::probe_all().map_err(|e| {
        if e.kind == crate::ffi::Kind::NoDriver {
            Refusal::unsupported(
                "no NVIDIA driver was found on this computer",
                "glyd run needs an NVIDIA GPU of the Ampere generation or newer and its driver (https://www.nvidia.com/drivers). On a computer without one, glyd compresses files: glyd --help",
            )
        } else {
            Refusal::unsupported(format!("the GPU cannot be used: {}", e.msg), "Check that nvidia-smi shows it")
        }
    })?;
    let g = machine::pick(&gpus).cloned().ok_or_else(|| Refusal::unsupported("the driver shows no GPU", "Check that nvidia-smi shows it"))?;
    for other in gpus.iter().map(|x| x.ordinal).filter(|&i| i != g.ordinal) {
        Dev::release(other);
    }
    if g.cc.0 < 8 {
        return Err(Refusal::unsupported(
            format!("{} (compute capability {}.{}) is older than the engine runs on", g.name, g.cc.0, g.cc.1),
            "glyd run needs the Ampere generation or newer: RTX 30 and 40 series, A10, A100, L4, L40S, H100",
        ));
    }
    Ok(g)
}

/// The machine code this build carries (`sm_80 sm_86 ...`, build.rs), or "none" for a build without kernels.
pub fn build_code() -> &'static str {
    match env!("GLYD_ENGINE_SASS") {
        "" => "none",
        s => s,
    }
}

/// What is said of a GPU the build has no machine code for: its compute capability and the code the build carries (the driver is new enough, so the driver is not the fix).
pub fn no_code_for(gpu: &Gpu) -> String {
    format!(
        "this build has no code for {} (compute capability {}.{}): it carries {} and the PTX of compute_89, which the driver compiles for a newer GPU, and it failed here",
        gpu.name,
        gpu.cc.0,
        gpu.cc.1,
        build_code()
    )
}

/// The way out of `no_code_for`: a build for this GPU, or vLLM.
pub fn no_code_fix(gpu: &Gpu) -> String {
    let arch = format!("{}{}", gpu.cc.0, gpu.cc.1);
    machine::instead(
        &format!("Build the engine for this GPU (GLYD_ENGINE_ARCHS={arch} cargo build --release -p glyd-engine, CUDA toolkit 13), or use the vLLM-based glyd"),
        &format!("Build the engine for this GPU (GLYD_ENGINE_ARCHS={arch} cargo build --release -p glyd-engine, CUDA toolkit 13): {}", machine::FULL_TOOL),
    )
}

/// An error from loading the engine in words (memory, a driver too old for the kernels), by its kind and the status of the call that failed (`ffi::Error`: "cuMemAlloc: out of memory (2)" is
/// the call, the driver's text and its status). Whatever it is, the engine is not the only way to run the model: status 3.
fn engine_refusal(e: crate::ffi::Error, name: &str, fit: &LoadPlan, gpu: &Gpu) -> Refusal {
    let status = e.status();
    if e.out_of_memory() {
        return Refusal::unsupported(
            format!("the GPU ran out of memory loading {name} (about {} are needed; {} were free at the start)", gib(fit.gpu_total()), gib(gpu.free)),
            format!(
                "Close programs that use the GPU (nvidia-smi lists them), or ask for less context: --context {}, or run a smaller model",
                (fit.context / 2).max(2048) / 256 * 256
            ),
        );
    }
    // (209: no machine code for this GPU, where the driver is new enough for the build's kernels; the images the driver cannot read: 200 and 218; a toolchain newer than the driver: 222)
    if status == Some(209) && gpu.driver >= machine::driver_min(gpu.cc) {
        return Refusal::unsupported(no_code_for(gpu), no_code_fix(gpu));
    }
    if matches!(status, Some(200 | 209 | 218 | 222)) {
        let need = machine::driver_need(gpu.cc);
        let series = machine::driver_series(machine::driver_min(gpu.cc));
        return Refusal::unsupported(
            format!("the GPU's driver cannot run the engine's kernels ({})", e.msg),
            machine::instead(
                &format!("Update the NVIDIA driver (the engine's kernels need {need}), or use the vLLM-based glyd"),
                &format!("Update the NVIDIA driver to {series} or newer: on Ubuntu `sudo apt install nvidia-driver-{series}` and reboot, elsewhere the installer at https://www.nvidia.com/drivers"),
            ),
        );
    }
    cannot(e.msg)
}

// ---- a model made ready -------------------------------------------------------------------------------------------------------------------------------

/// What a plain dense decoder's description holds that the dense kind's lowering refuses, in the words a refusal says it in: biases in the attention, a rope with
/// scaling, an activation other than silu. None for the rest, which the lowering's own sentence names.
fn dense_lacks(d: &crate::runtime::describe::Described) -> Option<&'static str> {
    use crate::runtime::kinds::{Act, Body, Ffn, Mixer, RopeScaling};
    d.desc.blocks.iter().find_map(|b| match &b.body {
        Body::Mixer(Mixer::Attention(a)) if a.bias => Some("biases in the attention"),
        Body::Mixer(Mixer::Attention(a)) if a.rope.as_ref().is_some_and(|r| r.scaling != RopeScaling::None) => Some("rope scaling"),
        Body::Ffn(Ffn::Mlp(m)) if m.act != Act::Silu => Some("an activation other than silu"),
        _ => None,
    })
}

/// What to do with a model the engine does not run (yet): vLLM runs it where it is installed, else the whole tool does.
pub(crate) fn unrun_fix() -> String {
    machine::instead("Run it with the vLLM-based glyd, which is installed here", &format!("The engine does not run it yet: {}", machine::FULL_TOOL))
}

/// A model's config.json against what the description executor runs of its kind, with no file read but the config. A plain dense decoder (`dynamic::dense::is_dense`)
/// is held to the dense kind's own lowering (`runtime::describe`, then `dynamic::lower::dense`): what that takes runs, and what it refuses is refused here in the
/// same words, before anything is downloaded. A family of another route is that route's to check; a config the table has no family for is refused (exit 3) with what
/// is missing: the description of its architecture (src/support.rs `family`).
pub(crate) fn supported(name: &str, cfg_text: &str) -> Done<()> {
    // (config.json parsed once: src/runtime/config.rs)
    let c = crate::runtime::config::ModelConfig::parse(cfg_text).map_err(|e| Refusal::unsupported(format!("{name}: {}", e.msg), unrun_fix()))?;
    if !crate::runtime::describe::LLAMA.contains(&c.text_type.as_str()) {
        return crate::support::family(cfg_text, name).map(|_| ());
    }
    let d = crate::runtime::describe::describe_config(&c).map_err(|e| Refusal::unsupported(format!("{name}: {}", e.msg), unrun_fix()))?;
    let max_pos = c.shared.max_position_embeddings.unwrap_or(crate::support::DEFAULT_CTX);
    match crate::dynamic::lower::dense(&d, Vec::new(), max_pos) {
        Ok(_) => Ok(()),
        Err(e) => Err(Refusal::unsupported(
            match dense_lacks(&d) {
                Some(what) => format!("{name} has {what}, which this engine does not run yet"),
                None => format!("{name}: {}", e.msg),
            },
            unrun_fix(),
        )),
    }
}

/// The model's shape against what the engine's KV cache takes, and its context against what the tables cover: refused before anything is downloaded.
pub(crate) fn shape_checks(name: &str, cfg: &Config, fit: &LoadPlan) -> Done<()> {
    match machine::config_refusal(name, cfg).or_else(|| machine::context_refusal(name, fit.context)) {
        Some(r) => Err(r),
        None => Ok(()),
    }
}

/// The bytes of the packed copy a first run writes into the engine's cache: `pack` (the plan's weights), and none where the copy is made already (a glyd-v7 save at
/// the layout in the model's place in the cache).
fn first_run_pack_bytes(pack: u64, layout: Layout, source: Option<&Source>) -> u64 {
    let made = source.is_some_and(|s| store::cached_pack(&store::packed_dir(s, layout), layout));
    if made {
        0
    } else {
        pack
    }
}

/// What a first run writes (the download, and the packed copy of `pack` bytes: the plan's weights) against the disks: refused before anything is downloaded. `source`: the model
/// where it is on this computer already; `download`: the bytes still to fetch.
fn first_run_checks(name: &str, pack: u64, layout: Layout, source: Option<&Source>, download: u64) -> Done<()> {
    let pack = first_run_pack_bytes(pack, layout, source);
    let (models, cache) = (store::existing_ancestor(&hub::hub_cache()), store::existing_ancestor(&store::cache_root()));
    if let Some(r) = machine::disk_refusal(name, hub::disk_free(&models), hub::disk_free(&cache), store::same_filesystem(&models, &cache), download, pack) {
        return Err(r);
    }
    Ok(())
}

/// The directory a level's engine reads MODEL from at `layout` on `gpu`, made ready (a run's steps before its load, and all of `glyd prepare`): a glyd-v7 directory as
/// it is; else the pack in the cache or the pre-packed repo (`pre_packed`); else the model, downloaded where it is not here and packed once into the cache, its plan
/// made (`plan_level`) and the disks checked before anything is fetched. The directory, the glyd-v7 directory it is, and the layout as packed (wren of a QAT
/// checkpoint is the exact four-bit one).
fn level_dir(a: &Args, model: &str, layout: Layout, gpu: &Gpu, ui: &RefCell<Ui>) -> Done<(PathBuf, Option<PathBuf>, Layout)> {
    let name = short_name(model);
    let mut packed = v7_layout(model)?.map(|_| PathBuf::from(model));
    let first = match packed {
        None => Some(plan_level(a, model, layout, &config_text(model)?, None, gpu, None)?),
        Some(_) => None,
    };
    if packed.is_none() {
        packed = pre_packed(model, layout, ui)?;
    }
    let pack_bytes = first.as_ref().map_or(0, |p| p.weights);
    let dir = match &packed {
        Some(d) => d.clone(),
        None if Path::new(model).join("config.json").exists() => PathBuf::from(model),
        None => match Source::find(model) {
            Ok(s) => s.dir,
            Err(_) => {
                let (hub, remote, wanted) = store::hub_plan(model, ui)?;
                first_run_checks(&name, pack_bytes, layout, None, hub.missing(&remote, &wanted))?;
                store::download(&hub, &remote, &wanted, model, ui)?;
                Source::find(model).map_err(cannot)?.dir
            }
        },
    };
    // (a pre-packed repo given by its own name is a glyd-v7 directory once downloaded; a bf16 model is packed into the cache once, and read from there)
    if packed.is_none() {
        packed = v7_manifest(&dir).map_err(cannot)?.map(|_| dir.clone());
    }
    if let Some(d) = &packed {
        // (the layout as packed: wren of a QAT checkpoint is the exact four-bit one)
        let layout = v7_level(model, v7_manifest(d).map_err(cannot)?.map(|m| m.0), Some(layout))?.unwrap_or(layout);
        return Ok((dir, packed, layout));
    }
    if layout == Layout::Hummingbird {
        return Err(Refusal::unsupported(
            format!("{name} at the hummingbird level comes as a finished download or directory only: its experts' codes are made apart"),
            format!("Download the pre-packed model: glyd/{name}-hummingbird"),
        ));
    }
    let source = Source::find(model).map_err(cannot)?;
    first_run_checks(&name, pack_bytes, layout, Some(&source), 0)?;
    let dir = store::pack(&source, model, layout, pack_bytes, ui)?;
    Ok((dir.clone(), Some(dir), layout))
}

/// MODEL made ready at its level, the directory the engine reads (`glyd prepare`): a run's own steps up to its load, for every family this build runs. The level is
/// the one `serve` would load (`served`: the tag's, a packed directory's own, else the planner's; a step the ladder asks about needs --yes or GLYD_DEGRADE=yes), held to the
/// table as a run's is. A family of the dynamic route is its release's directory (downloaded where it is not here; its hummingbird directory at that level) sized
/// over this machine's GPUs as its load sizes it (`dynamic_dir`, `dynamic_plan`); every other is its level's glyd-v7 directory, pre-packed or packed here once, and
/// planned as the load plans it (`level_dir`, `plan_level`). What cannot be made ready is refused by the step that lacks something, in that step's own words.
fn prepare(a: &Args, ui: &RefCell<Ui>) -> Done<PathBuf> {
    a.need(1, "MODEL")?;
    let (a, layout) = served(a, ui)?;
    let model = a.need(1, "MODEL")?.to_string();
    let Some(layout) = layout else { return Err(unrun(&model)) };
    let name = short_name(&model);
    if let Some(family) = dynamic_family(&model).filter(|f| !lossy_route(f, layout)) {
        let layout = hb_level(&model, layout);
        if !matches!(layout, Layout::Tiered | Layout::Hummingbird) {
            return Err(penguin_only(&name, family, &model));
        }
        let dir = dynamic_dir(&model, &name, family, layout == Layout::Hummingbird, ui)?;
        dynamic_plan(&a, &name, family, &dir)?;
        return Ok(dir);
    }
    let gpu = gpu_pick()?;
    if let Some(r) = machine::gpu_refusal(&gpu, std::env::consts::ARCH, true, machine::ANY_FAMILY) {
        return Err(r);
    }
    let (dir, packed, layout) = level_dir(&a, &model, layout_for(&model, layout), &gpu, ui)?;
    // (the plan of the directory as it now is, the one the load makes: a refusal of it is said here, not at the run)
    let read = |f: &str| std::fs::read_to_string(dir.join(f)).ok();
    let text = read("config.json").ok_or_else(|| cannot(format!("{}: no config.json", dir.display())))?;
    plan_level(&a, &model, layout, &text, read("generation_config.json").as_deref(), &gpu, packed.as_deref())?;
    Ok(dir)
}

/// Sequences a server runs at once: --users, else 4 (1 to MAX_USERS; a level's engine as many as its memory holds).
pub(crate) fn users_of(a: &Args) -> Done<usize> {
    use crate::sched::MAX_USERS;
    let n: usize = a.num("users", 4usize).map_err(|e| Refusal::new(e.msg, format!("Give a number from 1 to {MAX_USERS}: --users 4")))?;
    if !(1..=MAX_USERS).contains(&n) {
        return Err(Refusal::new(format!("--users {n}: 1 to {MAX_USERS} sequences at once"), format!("Give a number from 1 to {MAX_USERS}: --users 4")));
    }
    Ok(n)
}

/// What the load took of the GPU against the plan: the weights the engine counts and the memory the GPU lost since the plan was made. A development build
/// writes it as JSON to `GLYD_PLAN_REPORT` (the GPU check reads it) and says it with `GLYD_ENGINE_VERBOSE`: the numbers the planner is calibrated on.
fn loaded_report(model: &str, plan: &LoadPlan, gpus: &[&Gpu], weights: u64) {
    let used = gpus.iter().map(|gpu| Dev::probe(gpu.ordinal).map_or(0, |g| gpu.free.saturating_sub(g.free))).sum::<u64>();
    crate::plan::remember_in(&store::cache_root(), model, plan, weights, used);
    // (the replies that follow are measured against the plan's prediction and kept with it: src/plan.rs `observe`)
    if let Some(sh) = config_text(model).ok().and_then(|c| crate::plan::shape_of(model, &c).ok()) {
        crate::plan::watch_in(&store::cache_root(), model, plan, sh, crate::plan::Machine::probe());
    }
    if let Some(path) = crate::tune!("GLYD_PLAN_REPORT") {
        let _ = std::fs::write(path, plan.report(model, weights, used));
    }
    if crate::tune!("GLYD_ENGINE_VERBOSE").is_some() {
        eprintln!(
            "planned {} of weights and {} of GPU memory; loaded {} of weights, the GPU lost {}",
            gib(plan.weights_on_gpu()),
            gib(plan.gpu_total() - plan.reserve),
            gib(weights),
            gib(used)
        );
    }
}

fn fail(ui: &RefCell<Ui>, r: &Refusal) -> i32 {
    refuse_to(ui, r, &STATUS)
}

/// A refusal said, and its status: to the terminal, or, where the program that started the engine named a status file and the engine is not up yet (it runs the model another way then), there, in a
/// line of its own, and nothing on the terminal.
fn refuse_to(ui: &RefCell<Ui>, r: &Refusal, status: &Status) -> i32 {
    if status.refuse(&r.what) {
        ui.borrow_mut().clear_live();
        return r.code;
    }
    ui.borrow_mut().refuse(r);
    r.code
}

/// `--status-file PATH`, which the program that starts `run` or `serve` gives: see status.rs.
fn status_file(a: &Args) {
    if let Some(p) = a.flag("status-file").filter(|p| !p.is_empty()) {
        STATUS.set(PathBuf::from(p));
    }
}

// ---- run and serve --------------------------------------------------------------------------------------------------------------------------------

/// The API key: --api-key, else GLYD_API_KEY or VLLM_API_KEY (the variable is the safer: a flag is in the process list).
fn api_key(a: &Args) -> Option<String> {
    key_of(a.flag("api-key"), &|k| std::env::var(k).ok())
}

/// `api_key` with the environment given (the flag first, then the variables in this order; an empty one is none).
fn key_of(flag: Option<&str>, env: &dyn Fn(&str) -> Option<String>) -> Option<String> {
    flag.map(str::to_string).or_else(|| ["GLYD_API_KEY", "VLLM_API_KEY"].iter().find_map(|k| env(k))).filter(|k| !k.is_empty())
}

/// What `--host`, the key and `--insecure-no-auth` allow, and what the user let in (`--allowed-origins`, `--allowed-hosts`): the refusal where the server must not start. Called before the model is loaded, and again where the
/// server listens (`listen`); it also sets the lists the server's checks read.
fn access_check(a: &Args) -> std::result::Result<(), Refusal> {
    let host = a.flag("host").unwrap_or("127.0.0.1");
    if let Some(why) = server::bind_refusal(host, api_key(a).is_some(), a.has("insecure-no-auth")) {
        return Err(Refusal::new(why, "GLYD_API_KEY=KEY glyd serve MODEL --host ADDRESS"));
    }
    server::set_access(server::parse_access(a.flag("allowed-origins"), a.flag("allowed-hosts")).map_err(|e| Refusal::new(e, ""))?);
    Ok(())
}

/// The server of a model on --port (8000 where none is given, else the first free one after it), or where no --port is given the port after `after` (the first model's of
/// `serve --with`).
fn listen_after(a: &Args, name: &str, window: usize, tx: mpsc::Sender<Job>, after: Option<u16>) -> Done<std::net::SocketAddr> {
    access_check(a)?;
    let host = a.flag("host").unwrap_or("127.0.0.1").to_string();
    let port: u16 = a.num("port", after.map_or(8000u16, |p| p.saturating_add(1)))?;
    let explicit = a.has("port");
    // SIGTERM stops the server taking requests and lets those in flight end (up to --drain-timeout seconds), then ends the process
    crate::ops::drain_on_sigterm(crate::ops::Life::global(), std::time::Duration::from_secs(a.num("drain-timeout", crate::ops::DRAIN_SECONDS)?), |_| {});
    // the checks of the Host and the Origin are on for every address of this computer's own, a loopback one the user names included; an address that is not is open to the network, with the key (or the user's word) it was let start with
    let local_only = server::is_loopback_addr(&host);
    for p in port..=port.saturating_add(if explicit { 0 } else { 20 }) {
        match server::listen(&format!("{host}:{p}"), name, window, api_key(a), local_only, tx.clone()) {
            Ok(addr) => {
                also_on_ollamas_port(a, &host, name, window, local_only, &tx);
                return Ok(addr);
            }
            Err(e) if e.msg.to_lowercase().contains("in use") && !explicit => continue,
            Err(e) if e.msg.to_lowercase().contains("in use") => {
                return Err(Refusal::new(format!("port {p} is in use by another program"), format!("Choose another: --port {}", p + 1)))
            }
            Err(e) => return Err(Refusal::new(e.msg, "")),
        }
    }
    Err(Refusal::new(format!("ports {port} to {} are all in use", port + 20), "Choose another: --port N"))
}

/// `--ollama-port PORT` (Ollama's own is 11434): the same server also listens there, for the programs that look for Ollama at its port. Only where the port is free: where it is not (Ollama runs there), that is said and
/// the server goes on at its own port, where Ollama's API is too (`/api`). The address, the key and the checks of the Host and the Origin are the server's own.
fn also_on_ollamas_port(a: &Args, host: &str, name: &str, window: usize, local_only: bool, tx: &mpsc::Sender<Job>) {
    let Some(p) = a.flag("ollama-port") else { return };
    let Some(port) = p.parse::<u16>().ok().filter(|&p| p > 0) else {
        eprintln!("glyd: --ollama-port {p}: not a port (Ollama's is 11434)");
        return;
    };
    match server::listen(&format!("{host}:{port}"), name, window, api_key(a), local_only, tx.clone()) {
        Ok(addr) => eprintln!("glyd: Ollama's API is also at http://{addr}"),
        Err(e) => eprintln!("glyd: --ollama-port {port}: {}: Ollama's API stays at the server's own port", e.msg),
    }
}

/// What the address and the key leave open, said where the server starts: an address that is not this computer's own opens the model to the network, and the origins the user let in are named.
fn access_notes(ui: &RefCell<Ui>, a: &Args) {
    let key = api_key(a).is_some();
    if let Some(h) = a.flag("host").filter(|h| !server::is_loopback_addr(h)) {
        ui.borrow_mut().note(&format!(
            "listening on {h}: anyone who can reach this computer's network address can send the model prompts, and the traffic is plain HTTP, not encrypted (use a VPN or an SSH tunnel). {} The chat page, /health and /stats stay open without a key.",
            if key { "The API key guards /v1 only." } else { "There is NO API key (--insecure-no-auth): set one with  GLYD_API_KEY=KEY glyd serve ...  (the key goes in the environment, not on the command line, where ps and a log show it)." }
        ));
    }
    if let Some(o) = a.flag("allowed-origins").filter(|o| !o.is_empty()) {
        ui.borrow_mut().note(&format!("scripts of these web pages may call the API: {o}"));
    }
    if a.has("api-key") {
        ui.borrow_mut().note("the API key is on this program's command line, which ps shows: next time set GLYD_API_KEY=KEY in the environment (glyd run and glyd serve move a key given as --api-key there before they start the engine).");
    }
}

/// The level `given` (MODEL as typed) asks for: its tag's, --lossy's (kestrel), a pre-packed repo's own (glyd/NAME-swift), penguin for `:penguin`; None where it names none.
fn requested(given: &str, lossy: bool) -> Option<&'static str> {
    level(given, lossy).ok().and_then(|(rest, l)| l.map(level_name).or_else(|| (rest != given).then_some("penguin")))
}

/// The NVIDIA driver's version (595.91.07) where the system says it (/proc/driver/nvidia/version, Linux).
pub(crate) fn driver_version() -> Option<String> {
    nv_version(&std::fs::read_to_string("/proc/driver/nvidia/version").ok()?)
}

/// The version in /proc/driver/nvidia/version's first line (`NVRM version: NVIDIA UNIX Open Kernel Module for x86_64  595.91.07  Release Build ...`): its first word of numbers and dots.
fn nv_version(text: &str) -> Option<String> {
    let line = text.lines().next()?;
    line.split_whitespace().find(|w| w.contains('.') && w.split('.').all(|p| !p.is_empty() && p.bytes().all(|b| b.is_ascii_digit()))).map(str::to_string)
}

/// What /v1/glyd/info says of the model served and the GPU (server::Info's `about`): the name it is served as; the repo and revision where the Hub's cache holds the directory (`models--ORG--NAME/snapshots/REV`);
/// the model it was made from and the Glyd that made it (glyd.json); each file's size and SHA-256 (a large one's the cache's blob name, which the download checked; one of up to 64 MiB read now; else null); the level
/// asked for and the one that runs; the precision of the weights, the activations, the sums, the keys and values and the states; and the GPU.
#[allow(clippy::too_many_arguments)]
fn about(
    served: &str,
    dir: &Path,
    asked: Option<&str>,
    level: &str,
    lossless: bool,
    weights: &str,
    held: Option<&crate::dev::Held>,
    gpu: &Gpu,
) -> Vec<(&'static str, Value)> {
    use crate::tools::obj;
    let (s, opt) = (|t: &str| Value::from(t), |t: Option<String>| t.map_or(Value::Null, Value::String));
    let manifest = std::fs::read_to_string(dir.join("glyd.json")).ok().and_then(|t| json::parse(&t).ok());
    let repo = dir
        .parent()
        .filter(|p| p.file_name().is_some_and(|n| n == "snapshots"))
        .and_then(Path::parent)
        .and_then(|p| p.file_name()?.to_str()?.strip_prefix("models--").map(|r| r.replacen("--", "/", 1)));
    let revision = repo.as_ref().and(dir.file_name().and_then(|n| n.to_str()).map(str::to_string));
    let mut paths: Vec<PathBuf> =
        std::fs::read_dir(dir).map(|r| r.filter_map(|e| e.ok().map(|e| e.path())).filter(|p| p.is_file()).collect()).unwrap_or_default();
    paths.sort();
    let files = paths.iter().map(|p| {
        let bytes = p.metadata().map_or(0, |m| m.len());
        let blob = std::fs::read_link(p)
            .ok()
            .and_then(|t| t.file_name()?.to_str().map(str::to_string))
            .filter(|b| b.len() == 64 && b.bytes().all(|c| c.is_ascii_hexdigit()));
        let sha256 = blob.or_else(|| (bytes <= 64 << 20).then(|| hub::sha256_of(p).ok()).flatten());
        obj(vec![
            ("name", Value::String(p.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default())),
            ("bytes", Value::from(bytes)),
            ("sha256", opt(sha256)),
        ])
    });
    let made_by = manifest.as_ref().and_then(|m| m.get("glyd")).and_then(Value::as_str).map(|v| format!("glyd {v}"));
    vec![
        (
            "model",
            obj(vec![
                ("id", s(served)),
                ("repo", opt(repo)),
                ("revision", opt(revision)),
                ("source", manifest.as_ref().and_then(|m| m.get("source")).cloned().unwrap_or(Value::Null)),
                ("made_by", opt(made_by)),
                ("files", Value::Array(files.collect())),
            ]),
        ),
        ("level", obj(vec![("requested", asked.map_or(Value::Null, s)), ("effective", s(level)), ("lossless", Value::Bool(lossless))])),
        (
            "precision",
            obj(vec![
                ("weights", s(weights)),
                ("activations", s("bf16")),
                ("sums", s("fp32")),
                ("kv", held.map_or(Value::Null, |h| s(h.kv_type))),
                ("state", held.map_or(Value::Null, |h| s(h.state_type))),
            ]),
        ),
        (
            "device",
            obj(vec![
                ("ordinal", Value::from(gpu.ordinal.max(0) as u64)),
                ("name", s(&gpu.name)),
                ("compute_capability", s(&format!("{}.{}", gpu.cc.0, gpu.cc.1))),
                ("cuda", s(&format!("{}.{}", gpu.driver / 1000, gpu.driver % 1000 / 10))),
                ("driver", opt(driver_version())),
            ]),
        ),
    ]
}

/// MODEL's level (its tag, or --lossy), before anything reads the model; the args are given back with the model without its tag. A glyd-v7 directory is a lossy level's packs: it runs at its own level, or the one asked if that is the same.
fn tagged(a: &Args) -> Done<(Args, Option<Layout>)> {
    let (model, layout) = level(a.pos.get(1).map_or("", String::as_str), a.has("lossy")).map(|(m, l)| (m.to_string(), l))?;
    let layout = load_level(a, &model, layout)?;
    table_check(&model, layout)?;
    let mut a = a.clone();
    if let Some(m) = a.pos.get_mut(1) {
        *m = model;
    }
    Ok((a, layout))
}

/// The family and the level against the table `fit` reads (src/support.rs), which `run`, `serve`, `preflight` and `fetch` all check first; a glyd-v7 directory or a
/// pre-packed repo is at its own level, and a config not to be had now is the paths' own to say.
fn table_check(model: &str, layout: Option<Layout>) -> Done<()> {
    if v7_layout(model)?.is_none() && levels::repo_level(model).is_none() {
        if let Ok(text) = config_text(model) {
            crate::support::check(&text, model, &short_name(model), layout.map_or("penguin", level_name))?;
        }
    }
    Ok(())
}

/// What a command does with the degrade steps the ladder asks about (src/plan.rs `ladder`): take them (`--yes`, or `GLYD_DEGRADE=yes`), ask (a terminal, `run` alone;
/// the default), or refuse (`GLYD_DEGRADE=no`, or no one to ask: `serve`, the service, a run with no terminal).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Degrade {
    Take,
    Ask,
    Refuse,
}

/// `yes`: the flag `--yes`; `config`: GLYD_DEGRADE's value (`yes`, `ask`, `no`); `can_ask`: this command may stop for an answer and has a terminal for it.
pub(crate) fn degrade_decision(yes: bool, config: Option<&str>, can_ask: bool) -> Degrade {
    match (yes, config.map(str::to_ascii_lowercase).as_deref()) {
        (true, _) | (_, Some("yes" | "y" | "true" | "1")) => Degrade::Take,
        (_, Some("no" | "n" | "false" | "0" | "refuse")) => Degrade::Refuse,
        _ if can_ask => Degrade::Ask,
        _ => Degrade::Refuse,
    }
}

/// Before the model is loaded: the ladder from the level asked for (src/plan.rs), said in the first lines where a step is taken, and the steps the user must agree to
/// asked about (a lower level, or a step over a third of the speed): taken with `--yes`, asked in `run` at a terminal, else refused with the steps named (the same
/// status `run` gave before the ladder, so a program that started the engine can run the model another way). `interactive`: `run`, which may ask.
/// Returns the arguments and level the load goes on with: the lower level a step took, else those it was given.
pub(crate) fn degrade_gate(a: &Args, model: &str, layout: Option<Layout>, interactive: bool, ui: &RefCell<Ui>) -> Done<(Args, Option<Layout>)> {
    let same = || Ok((a.clone(), layout));
    // (a glyd-v7 directory or a pre-packed repo is at its own level; a model whose config is not to be had is the paths' own to say)
    if v7_layout(model)?.is_some() || levels::repo_level(model).is_some() {
        return same();
    }
    let Ok(text) = config_text(model) else { return same() };
    let name = short_name(model);
    let Ok(family) = crate::support::family(&text, &name) else { return same() };
    let row = crate::support::row(family);
    // (the machine the planner that picks an untagged run's level plans for: --machine's preset, else this one)
    let machine = a.flag("machine").and_then(crate::plan::Machine::preset).unwrap_or_else(crate::plan::Machine::probe);
    // (the shape `fit` prices the steps on: the model's tensors where their headers can be had, so the run asks where fit says it will)
    let Ok(sh) = crate::plan::shape_of(model, &text) else { return same() };
    let asked = layout.map_or("penguin", level_name);
    if !row.levels_now(model).contains(&asked) {
        return same();
    }
    let users = if a.pos.first().is_some_and(|c| c == "serve") { users_of(a).unwrap_or(1) } else { 1 };
    let Some((_, target, devs, reach)) = crate::plan::sizing_inputs(&machine, true) else { return same() };
    let want = a.flag("context").and_then(|c| c.parse().ok()).unwrap_or_else(|| row.default_ctx.unwrap_or(LOSSY_CTX).min(sh.max_ctx));
    let size = |l: &'static str, ctx: usize| {
        let q = if ctx == want { a.clone() } else { a.with(&["context", "ctx"], ctx.to_string()) };
        size(&q, model, &text, l, &devs, &target, &*reach, users)
    };
    let ladder = crate::plan::ladder(&sh, &machine, row, model, asked, want, &size);
    if ladder.is_as_asked() || ladder.plan.is_none() || ladder.steps.is_empty() {
        return same();
    }
    ui.borrow_mut().line(ladder.report().trim_end());
    // (the load goes on at the context the ladder found: a shorter one is a step it takes)
    let at = ladder.plan.as_ref().map_or(want, |p| p.context);
    let args = if at < want { a.with(&["context", "ctx"], at.to_string()) } else { a.clone() };
    if !ladder.needs_confirmation() {
        return Ok((args, layout));
    }
    let chosen = ladder.plan.as_ref().map(|p| p.level).unwrap_or(asked);
    let can_ask = interactive && std::io::stdin().is_terminal() && std::io::stderr().is_terminal();
    let take = match degrade_decision(a.has("yes"), std::env::var("GLYD_DEGRADE").ok().as_deref(), can_ask) {
        Degrade::Take => true,
        Degrade::Ask => {
            eprint!("Take these steps? [y/N] ");
            let mut answer = String::new();
            let _ = std::io::stdin().read_line(&mut answer);
            matches!(answer.trim().to_ascii_lowercase().as_str(), "y" | "yes")
        }
        Degrade::Refuse => false,
    };
    if !take {
        return Err(Refusal::unsupported(
            format!("{name} does not load at the {asked} level as asked on this machine; it loads at {chosen} with the steps above, which need your word"),
            format!(
                "Take them: add --yes (or set GLYD_DEGRADE=yes); or choose the level yourself: glyd run {}:{chosen}",
                level(model, false).map_or(model, |m| m.0)
            ),
        ));
    }
    let layout = if chosen == asked { layout } else { levels::layout_of(chosen).map(|l| layout_for(model, l)) };
    Ok((args, layout))
}

pub fn serve(a: &Args) -> i32 {
    if let Some(with) = a.flag("with") {
        return serve_two(a, with.to_string());
    }
    let ui = RefCell::new(Ui::new());
    status_file(a);
    // (before the GPU is looked at and the model is read: an address that may not be served is refused first)
    if let Err(r) = access_check(a) {
        return fail(&ui, &r);
    }
    // the level is run's, read before the GPU is looked at; the name served keeps the tag
    let given = a.pos.get(1).cloned().unwrap_or_default();
    // (a server never stops for an answer: a step it asks about is taken only with --yes)
    let (a, layout) = match served(a, &ui) {
        Ok(x) => x,
        Err(r) => return fail(&ui, &r),
    };
    let a = &a;
    // (every model this build runs is the description executor's, at a level; one with none is refused in the words the other runner reads)
    match layout {
        Some(layout) => serve_lossy(a, layout, given, &ui, None),
        None => fail(&ui, &unrun(a.pos.get(1).map_or("", String::as_str))),
    }
}

/// What `serve` serves: the arguments with the level's tag read and its degrade steps taken, and the level's layout (None: no engine of this build takes the model).
fn served(a: &Args, ui: &RefCell<Ui>) -> Done<(Args, Option<Layout>)> {
    tagged(a).and_then(|(a, l)| {
        let model = a.pos.get(1).cloned().unwrap_or_default();
        degrade_gate(&a, &model, l, false, ui)
    })
}

/// The two models of `serve --with`, each on its own thread, in step: the first loads and makes its sequences' memory, then tells the second, which loads in what is left (its plan
/// sees what the first holds) and makes its own of the rest; each tells the other when its server listens (its port). A model that fails says so and its thread ends, its end of the pair
/// with it: the other's wait ends there too (`hear` is None), so neither waits for a model that is not coming.
struct Pair {
    first: bool,
    tell: mpsc::Sender<u16>,
    hear: mpsc::Receiver<u16>,
}

impl Pair {
    fn tell(&self, v: u16) {
        let _ = self.tell.send(v);
    }

    fn hear(&self) -> Option<u16> {
        self.hear.recv().ok()
    }
}

/// `serve MODEL --with MODEL2 [--with-port P]`: two models in one process, each its own server (MODEL on --port, MODEL2 on --with-port, else the first free port after MODEL's),
/// scheduler and runtime context (its figures, requests, failure and /v1/glyd/info its own) on the one GPU. The first loads, as it would alone, and takes up to half of the memory left
/// for its sequences; the second then loads in what is left and takes the rest. Both are models of the description executor (a lossy level, or a family it runs at every level). A sticky
/// device error of either ends the process, as it ends a server of one model (the service starts the models again).
fn serve_two(a: &Args, with: String) -> i32 {
    let ui = RefCell::new(Ui::new());
    status_file(a);
    if let Err(r) = access_check(a) {
        return fail(&ui, &r);
    }
    let mut b = a.clone();
    if b.pos.len() < 2 {
        return fail(&ui, &Refusal::new("serve --with: the first model is not named", "glyd serve MODEL --with MODEL2"));
    }
    b.pos[1] = with.clone();
    b.flags.retain(|(k, _)| !matches!(k.as_str(), "with" | "with-port" | "port" | "status-file" | "ollama-port"));
    if let Some(p) = a.flag("with-port") {
        b.flags.push(("port".into(), Some(p.into())));
    }
    let ((to_two, from_one), (to_one, from_two)) = (mpsc::channel(), mpsc::channel());
    let two = std::thread::Builder::new().name("glyd-serve-with".into()).spawn(move || {
        crate::runtime::context::Context::serve_here(crate::runtime::context::Context::new());
        let ui = RefCell::new(Ui::new());
        let pair = Pair { first: false, tell: to_one, hear: from_one };
        match served(&b, &ui) {
            Ok((b, Some(layout))) => serve_lossy(&b, layout, with, &ui, Some(&pair)),
            Ok(_) => fail(&ui, &Refusal::new(format!("{with}: --with serves a model of the description executor"), "Give it a lossy level, as MODEL:wren")),
            Err(r) => fail(&ui, &r),
        }
    });
    if let Err(e) = two {
        return fail(&ui, &Refusal::new(format!("a thread for the second model: {e}"), ""));
    }
    let pair = Pair { first: true, tell: to_two, hear: from_two };
    let given = a.pos.get(1).cloned().unwrap_or_default();
    match served(a, &ui) {
        Ok((a, Some(layout))) => serve_lossy(&a, layout, given, &ui, Some(&pair)),
        Ok(_) => fail(
            &ui,
            &Refusal::new(format!("{given}: --with serves a model of the description executor beside another"), "Give it a lossy level, as MODEL:wren"),
        ),
        Err(r) => fail(&ui, &r),
    }
}

/// `serve MODEL:kestrel` (or `:swift`): the lossy level loaded as run loads it, served by the scheduler to up to --users sequences at once where the engine is a step of several (src/dynamic/rows.rs: Gemma 4),
/// else to one user at a time. `pair`: one of the two models of `serve --with`.
fn serve_lossy(a: &Args, layout: Layout, name: String, ui: &RefCell<Ui>, pair: Option<&Pair>) -> i32 {
    let gone = |what: &str| Refusal::new(format!("the other model of serve --with {what}"), "Its own lines say why");
    // (the second model loads once the first has its memory: its plan sees what the first holds)
    if let Some(p) = pair.filter(|p| !p.first) {
        if p.hear().is_none() {
            return fail(ui, &gone("did not load"));
        }
    }
    let (mut e, tok, chat, defaults, vocab, line, mut info) = match load_lossy(a, layout, &name, ui) {
        Ok(x) => x,
        Err(r) => return fail(ui, &r),
    };
    ui.borrow_mut().line(&line);
    let users = match users_of(a) {
        Ok(u) => u,
        Err(r) => return fail(ui, &r),
    };
    // (the first takes up to half of what is free for its sequences, leaving the rest for the second)
    if pair.is_some_and(|p| p.first) {
        e.share_memory(0.5);
    }
    // (the step's window: the context, or less where its paged memory holds less for one sequence alone)
    let (several, step_window) = match e.as_step(users) {
        Ok(s) => (s.as_ref().map(|(k, _)| *k), s.map(|(_, st)| st.window())),
        Err(err) => return fail(ui, &cannot(err)),
    };
    e.share_memory(1.0);
    if let Some(p) = pair.filter(|p| p.first) {
        p.tell(0);
    }
    let window = step_window.map_or(e.capacity(), |w| w.min(e.capacity()));
    if let Some(k) = several {
        // (one where more were asked for: the GPU's memory holds one, or the model runs one sequence at a time, across GPUs or with a draft model)
        ui.borrow_mut().line(&match k {
            1 if users > 1 => format!("One user at a time, of the {users} asked for."),
            _ if k < users => format!("Up to {k} users at once (the GPU's free memory holds {k} of the {users} asked for)."),
            _ => format!("Up to {k} users at once."),
        });
        if window < e.capacity() {
            ui.borrow_mut().line(&format!("A request holds up to {window} tokens of the {} of context: the GPU's memory holds no more for one.", e.capacity()));
        }
        if let Some(i) = &mut info {
            (i.concurrency, i.refused, i.held, i.window) = (k, &[], e.held(), window);
        }
    }
    let (tx, rx) = mpsc::channel();
    // (the second model's server after the first's: its port the next free one where it names none)
    let after = match pair {
        Some(p) if !p.first => match p.hear() {
            Some(port) => Some(port),
            None => return fail(ui, &gone("did not start its server")),
        },
        _ => None,
    };
    let addr = match listen_after(a, &name, window, tx.clone(), after) {
        Ok(x) => x,
        Err(r) => return fail(ui, &r),
    };
    if let Some(i) = info {
        let _ = crate::runtime::context::Context::current().info.set(i);
    }
    if let Some(p) = pair {
        p.tell(addr.port());
    }
    // (ready, for the status file, when every model this process serves is)
    match pair {
        Some(p) if p.first => {
            if p.hear().is_none() {
                return fail(ui, &gone("did not start its server"));
            }
            STATUS.ready();
        }
        Some(_) => {}
        None => STATUS.ready(),
    }
    access_notes(ui, a);
    ui.borrow_mut().line(&format!("{name} is served at http://{addr}: the chat page at /, the API at /v1 (OpenAI-compatible). Ctrl-C stops it."));
    let shared =
        Arc::new(server::Shared { vocab, tok, chat, name, defaults, window, thinking: None, refused: &[], rt: crate::runtime::context::Context::current() });
    match e.as_step(users) {
        Ok(Some((k, step))) => server::work_with(&mut crate::sched::Sched::new(step, k), shared, &rx, &tx),
        Ok(None) => return fail(ui, &cannot(crate::ffi::Error("this engine is not a scheduler's step".into()))),
        Err(err) => return fail(ui, &cannot(err)),
    }
    0
}

/// A prompt that is not there.
fn no_prompt() -> Refusal {
    Refusal::new("no prompt: standard input is empty", "Give one with --prompt TEXT, or run glyd in a terminal to chat")
}

/// Whether a file is known to hold nothing without reading it: no bytes, or /dev/null (a pipe is not asked: reading it is not undone).
fn is_empty_input(f: &std::fs::File) -> bool {
    #[cfg(unix)]
    use std::os::fd::AsRawFd;
    #[cfg(unix)]
    use std::os::unix::fs::FileTypeExt;
    match f.metadata() {
        Ok(m) if m.is_file() => m.len() == 0,
        #[cfg(unix)]
        Ok(m) if m.file_type().is_char_device() => std::fs::read_link(format!("/proc/self/fd/{}", f.as_raw_fd())).is_ok_and(|p| p == Path::new("/dev/null")),
        _ => false,
    }
}

#[cfg(unix)]
fn stdin_is_empty() -> bool {
    use std::os::fd::AsFd;
    std::io::stdin().as_fd().try_clone_to_owned().is_ok_and(|fd| is_empty_input(&std::fs::File::from(fd)))
}

#[cfg(windows)]
fn stdin_is_empty() -> bool {
    use std::os::windows::io::AsHandle;
    std::io::stdin().as_handle().try_clone_to_owned().is_ok_and(|h| is_empty_input(&std::fs::File::from(h)))
}

pub fn run(a: &Args) -> i32 {
    let ui = RefCell::new(Ui::new());
    status_file(a);
    if let Err(r) = access_check(a) {
        return fail(&ui, &r);
    }
    let (a, layout) = match tagged(a).and_then(|(a, l)| {
        let model = a.pos.get(1).cloned().unwrap_or_default();
        degrade_gate(&a, &model, l, true, &ui)
    }) {
        Ok(x) => x,
        Err(r) => return fail(&ui, &r),
    };
    let a = &a;
    let thinking = if a.has("no-think") { Some(false) } else { None };
    // one answer for a prompt given, or piped in: a pipe is read once the engine is up, for where it stops before that the program that runs the model another way finds standard input as it was
    // (a pipe that is known to hold nothing, a file of no bytes or /dev/null, is refused now)
    let prompt = a.flag("prompt").map(str::to_string);
    let piped = prompt.is_none() && !a.has("chat") && !std::io::stdin().is_terminal();
    if piped && stdin_is_empty() {
        return fail(&ui, &no_prompt());
    }
    if let Some(p) = &prompt {
        if p.trim().is_empty() {
            return fail(&ui, &Refusal::new("the prompt is empty", "Give one: --prompt \"Say hello\""));
        }
    }
    sig::install();
    match layout {
        Some(layout) => run_lossy(a, layout, &ui, prompt, piped, thinking),
        None => fail(&ui, &unrun(a.pos.get(1).map_or("", String::as_str))),
    }
}

/// One answer on stdout (the thinking on stderr), and the exit status.
fn once(talk: &mut Talk, defaults: &Params, prompt: &str, thinking: Option<bool>, ui: &RefCell<Ui>) -> i32 {
    let messages = vec![Message::new("user", prompt)];
    let mut request = Request { params: defaults.clone(), thinking, ..Request::default() };
    request.params.seed = sampler::fresh_seed();
    let mut split = Splitter::new(talk.opens_thinking(&messages, &request));
    let (mut thought, mut answered, mut ended) = (false, false, true);
    let tty_err = std::io::stderr().is_terminal();
    let mut show = |parts: Vec<(Part, String)>| {
        for (part, t) in parts {
            match part {
                Part::Reasoning => {
                    thought = true;
                    ended = t.ends_with('\n');
                    eprint!("{}", dim(&t, tty_err));
                }
                Part::Content => {
                    // (the thinking ends its line before the answer: it ran into the answer's first words where it did not end in one)
                    if thought && !answered && !ended {
                        eprintln!();
                    }
                    answered = true;
                    out(&t);
                }
            }
        }
    };
    let r = talk.reply(&messages, &request, &mut |t| {
        show(split.feed(t));
        true
    });
    show(split.finish());
    match r {
        Ok(o) => {
            out("\n");
            if !answered {
                let why = if o.finish == crate::sched::Finish::Length && thought {
                    "The model was still thinking when it ran out of room: ask again with --no-think"
                } else {
                    "The model sent no answer"
                };
                ui.borrow_mut().line(why);
                return 1;
            }
            0
        }
        Err(e) => {
            ui.borrow_mut().refuse(&Refusal::new(e.msg, ""));
            1
        }
    }
}

/// The terminal's chat: lines read, each sent to the engine's thread as a job, the pieces of the reply printed as they come (the thinking dimmed). Returns (and stops the engine's thread) at
/// /bye or the end of input.
fn chat_loop(tx: mpsc::Sender<Job>, defaults: Params, mut thinking: Option<bool>, tty: bool) {
    let stdin = std::io::stdin();
    let mut messages: Vec<Message> = Vec::new();
    let mut input = stdin.lock();
    'chat: loop {
        eprint!("> ");
        let _ = std::io::stderr().flush();
        let mut line = String::new();
        if input.read_line(&mut line).unwrap_or(0) == 0 {
            break;
        }
        let mut text = line.trim().to_string();
        match text.as_str() {
            "" => continue,
            "/bye" => break,
            "/clear" => {
                messages.clear();
                eprintln!("(a new chat)");
                continue;
            }
            "/think" => {
                thinking = if thinking == Some(false) { None } else { Some(false) };
                eprintln!("thinking {}", if thinking == Some(false) { "off" } else { "on" });
                continue;
            }
            "/help" => {
                eprintln!("  /bye           leave (or Ctrl-D)\n  /clear         start a new chat\n  /think         turn the model's thinking on or off\n  \"\"\"            start and end a message of several lines");
                continue;
            }
            "\"\"\"" => {
                text.clear();
                loop {
                    let mut l = String::new();
                    if input.read_line(&mut l).unwrap_or(0) == 0 {
                        break 'chat;
                    }
                    if l.trim() == "\"\"\"" {
                        break;
                    }
                    text.push_str(&l);
                }
                if text.trim().is_empty() {
                    continue;
                }
            }
            _ => {}
        }
        messages.push(Message::new("user", &text));
        let (otx, orx) = mpsc::channel();
        let abort = Arc::new(AtomicBool::new(false));
        sig::INTERRUPT.store(false, Ordering::Relaxed);
        sig::BUSY.store(true, Ordering::Relaxed);
        let mut request = Request { params: defaults.clone(), thinking, ..Request::default() };
        request.params.seed = sampler::fresh_seed();
        if tx.send(Job::Term { messages: messages.clone(), request, out: otx, abort: abort.clone() }).is_err() {
            break;
        }
        let t0 = Instant::now();
        let mut split = Splitter::new(false);
        let (mut said, mut thought_now) = (String::new(), false);
        let mut ended = None;
        let show = |parts: Vec<(Part, String)>, said: &mut String, thought_now: &mut bool| {
            for (part, t) in parts {
                match part {
                    Part::Reasoning => {
                        *thought_now = true;
                        out(&dim(&t, tty));
                    }
                    Part::Content => {
                        if *thought_now {
                            out(&format!("\n{}\n", dim("...done thinking.", tty)));
                            *thought_now = false;
                        }
                        said.push_str(&t);
                        out(&t);
                    }
                }
            }
        };
        loop {
            if sig::INTERRUPT.swap(false, Ordering::Relaxed) {
                abort.store(true, Ordering::Relaxed);
            }
            match orx.recv_timeout(std::time::Duration::from_millis(100)) {
                Ok(Piece::Text(t)) => show(split.feed(&t), &mut said, &mut thought_now),
                Ok(Piece::End(r)) => {
                    ended = Some(r);
                    break;
                }
                Err(mpsc::RecvTimeoutError::Timeout) => {}
                Err(mpsc::RecvTimeoutError::Disconnected) => break,
            }
        }
        sig::BUSY.store(false, Ordering::Relaxed);
        show(split.finish(), &mut said, &mut thought_now);
        println!();
        match ended {
            Some(Ok(o)) => {
                let secs = t0.elapsed().as_secs_f64();
                eprintln!(
                    "({} tokens, {:.1} tokens/s{})",
                    o.completion_tokens,
                    o.completion_tokens as f64 / secs,
                    if abort.load(Ordering::Relaxed) { ", stopped" } else { "" }
                );
                if said.trim().is_empty() {
                    messages.pop();
                    eprintln!(
                        "{}",
                        if o.finish == crate::sched::Finish::Length {
                            "The model was still thinking when it ran out of room: turn thinking off with /think, or start a new chat with /clear"
                        } else {
                            "The model sent no answer"
                        }
                    );
                } else {
                    messages.push(Message::new("assistant", said.trim()));
                }
            }
            Some(Err(e)) => {
                messages.pop();
                eprintln!(
                    "{}",
                    if e.contains("maximum context length") {
                        "This conversation is longer than the model's window. Start a new chat with /clear.".to_string()
                    } else {
                        e
                    }
                );
            }
            None => break,
        }
    }
    let _ = tx.send(Job::Stop);
}

// ---- the lossy levels: a model of a family whose row has them (the plain dense kind, Gemma 4, the Qwen3.5 layout, the Nemotron-H layout) at about 6.5 bits per weight (:kestrel, which --lossy is: six-bit block codes), 5.5 (:swift: five-bit) or 4.5 (:wren: four-bit; the Nemotron-H layout's experts alone, the rest six-bit), one user at a time ------

// (the lossy paths' work, context bounds, host margin and Gemma's draft and chunk are src/plan.rs's constants, with the one reserve)

/// The Gemma 4 draft model ("-assistant") to draft with: `--draft DIR`, else the model directory's `draft/` where it has a config.json; none with `--no-draft`.
pub(crate) fn gemma_draft(a: &Args, dir: Option<&Path>) -> Option<PathBuf> {
    if a.has("no-draft") {
        return None;
    }
    a.flag("draft").map(PathBuf::from).or_else(|| dir.map(|d| d.join("draft")).filter(|d| d.join("config.json").exists()))
}

/// The terminal's replies on the one sequence of `e` (chat_loop's jobs, on this thread): the jobs a server would send never come, as no port is held.
fn work_lossy(e: &mut dyn Gen, tok: &Tok, chat: Option<&Chat>, rx: &mpsc::Receiver<Job>) {
    while let Ok(job) = rx.recv() {
        match job {
            Job::Term { messages, request, out, abort } => {
                let mut talk = Talk { e: &mut *e, tok, chat };
                let sink = out.clone();
                let r = talk.reply(&messages, &request, &mut |t| sink.send(Piece::Text(t.to_string())).is_ok() && !abort.load(Ordering::Relaxed));
                let _ = out.send(Piece::End(r.map_err(|e| e.to_string())));
            }
            Job::Stop => return,
            _ => {}
        }
    }
}

/// The glyd-v7 directory of MODEL at `layout` that needs no packing here, or None (said in a line where the Hub has none): the pack in the cache, else the pre-packed repo (`packed_repo`) in the Hub cache or downloaded.
/// A local directory, and a pre-packed repo given by its own name, are not looked up.
fn pre_packed(model: &str, layout: Layout, ui: &RefCell<Ui>) -> Done<Option<PathBuf>> {
    if Path::new(model).join("config.json").exists() || levels::repo_level(model).is_some() {
        return Ok(None);
    }
    let level = level_name(layout);
    if let Ok(s) = Source::find(model) {
        let dir = store::packed_dir(&s, layout);
        if store::cached_pack(&dir, layout) {
            return Ok(Some(dir));
        }
    }
    let repo = packed_repo(model, level);
    let held = |r: &str| Source::find(r).ok().map(|s| s.dir).filter(|d| matches!(v7_manifest(d), Ok(Some((l, _))) if l == layout));
    if let Some(d) = held(&repo) {
        return Ok(Some(d));
    }
    let Some((hub, remote, wanted)) = store::hub_plan_packed(&repo, level, ui)? else {
        ui.borrow_mut()
            .line(&format!("There is no pre-packed download of {} at the {level} level: downloading the model and packing it here, once.", short_name(model)));
        return Ok(None);
    };
    store::download(&hub, &remote, &wanted, &repo, ui)?;
    Ok(held(&repo))
}

/// The plan of MODEL (its config.json `text`, its generation_config.json's `gen`) at `layout` on `gpu`, as `load_lossy`, `check_lossy` and `prepare` make it, by the
/// kind its config describes: a plain dense decoder's on the executor (`dynamic::dense::is_dense`; sized as `fit` sizes it: penguin at the layout this GPU is fastest
/// in, a lossy level at its own; a server's sequences in shared pages), else the lossy levels' (`lossy_plan`; `packed`: the glyd-v7 directory it is).
fn plan_level(a: &Args, model: &str, layout: Layout, text: &str, gen: Option<&str>, gpu: &Gpu, packed: Option<&Path>) -> Done<LoadPlan> {
    if !crate::dynamic::dense::is_dense(text) {
        return lossy_plan(a, model, layout, text, gen, gpu.free, &gpu.name, packed);
    }
    let name = short_name(model);
    supported(&name, text)?;
    let cfg = Config::parse(text, "config.json", gen).map_err(cannot)?;
    let users = if a.pos.first().is_some_and(|c| c == "serve") { users_of(a)? } else { 1 };
    let (_, _, p) = qwen3_plan(a, &cfg, &name, gpu, Some(layout).filter(|l| *l != Layout::Tiered), users)?;
    shape_checks(&name, &cfg, &p)?;
    Ok(p)
}

/// The level's engine as `run` loads it (Gemma 4's, the plain dense kind's, the Qwen3.5 layout's or the Nemotron-H layout's), with its tokenizer, chat template and sampling defaults, the size of its vocabulary, the line that says it is ready, and what
/// /v1/glyd/info says of it served as `served` (MODEL as typed).
#[allow(clippy::type_complexity)]
fn load_lossy(
    a: &Args,
    layout: Layout,
    served: &str,
    ui: &RefCell<Ui>,
) -> Done<(Box<dyn Gen>, Tok, Option<Chat>, Params, usize, String, Option<server::Info>)> {
    let (t0, started) = (Instant::now(), crate::monitor::clock());
    let model = a.need(1, "MODEL")?.to_string();
    match dynamic_family(&model) {
        Some(family) if lossy_route(family, layout) => {}
        Some(family) => return load_dynamic(a, &model, family, hb_level(&model, layout), served, ui),
        None => {}
    }
    let name = short_name(&model);
    let gpu = gpu_pick()?;
    if let Some(r) = machine::gpu_refusal(&gpu, std::env::consts::ARCH, true, machine::ANY_FAMILY) {
        return Err(r);
    }
    // (the directory made ready, as `glyd prepare` makes it: downloaded, packed once)
    let (dir, packed, layout) = level_dir(a, &model, layout_for(&model, layout), &gpu, ui)?;
    let read = |f: &str| std::fs::read_to_string(dir.join(f)).ok();
    let plan = plan_level(
        a,
        &model,
        layout,
        &read("config.json").ok_or_else(|| cannot(format!("{}: no config.json", dir.display())))?,
        read("generation_config.json").as_deref(),
        &gpu,
        packed.as_deref(),
    )?;
    // (a Nemotron-H model's layouts differ by pack: the bits a weight are what its bytes come to)
    let bits = if plan.kind == Kind::NemotronH { plan.weights as f64 * 16.0 / plan.bf16 as f64 } else { bits(layout) };
    let on_gpu = plan.weights_on_gpu();
    let exact = match layout {
        Layout::Tiered => "lossless: every weight is the original's",
        Layout::Int4H => "the QAT checkpoint's weights bit for bit",
        _ => "not lossless",
    };
    ui.borrow_mut().line(&format!(
        "{name}: about {} on the GPU at the {} level (about {bits:.1} bits per weight, {exact}), in place of {} of bf16.",
        gib(on_gpu),
        level_name(layout),
        gib(plan.bf16)
    ));
    if let Some(c) = plan.experts {
        let held = if c.ram < c.host_bytes() {
            format!("{} of experts in the computer's memory, {} read from the disk as needed", gib(c.ram), gib(c.disk_bytes()))
        } else {
            format!("all of them are in the computer's memory ({})", gib(c.host_bytes()))
        };
        ui.borrow_mut().line(&format!("{} experts of {} a layer are on the GPU, chosen as the router asks for them; {held}.", c.slots, c.experts));
    }
    ui.borrow_mut().line(&format!("Plan: {}.", plan.text()));
    if let Some(n) = machine::cuda12_note(gpu.driver, crate::dev::has_kernels12()) {
        ui.borrow_mut().line(&format!("Note: {n}"));
    }
    ui.borrow_mut().status(&format!("{} {name} and loading it onto the GPU...", if packed.is_some() { "Reading" } else { "Packing" }));
    let (mut e, bytes, vocab) = open_lossy(a, &plan, &dir, &gpu, &name, ui)?;
    loaded_report(&model, &plan, &[&gpu], bytes as u64);
    // (the model's first load on this GPU: a reply timed, its speeds measured here from then on)
    crate::plan::first_use(&mut *e, vocab);
    let tok = Tok::load(&dir).map_err(cannot)?;
    let chat = Chat::load(&dir).map_err(cannot)?;
    let mut defaults = sampler::generation_defaults(&dir);
    if let Some(t) = a.flag("temp") {
        defaults.temperature = t.parse().map_err(|_| Refusal::new(format!("--temp {t}: not a number"), ""))?;
    }
    let line = format!(
        "Ready in {}: {name} on the {}, {} of weights, {} tokens of context, one user at a time.",
        fmt_time(t0.elapsed().as_secs_f64()),
        gpu.name,
        gib(bytes as u64),
        plan.context
    );
    // (what /v1/glyd/info says of it, where it is served)
    let info = (!served.is_empty()).then(|| {
        let (held, lossless) = (e.held(), matches!(layout, Layout::Tiered | Layout::Int4H));
        let weights = format!("{} level: about {bits:.1} bits per weight, {exact}", level_name(layout));
        let mut about = about(served, &dir, requested(served, a.has("lossy")), level_name(layout), lossless, &weights, held.as_ref(), &gpu);
        about.push(("costs", crate::costs::served(&dir, served, requested(served, a.has("lossy")), e.capacity(), plan.chunk, held)));
        server::Info { about, gpu: Some(gpu.ordinal), held, concurrency: 1, window: e.capacity(), refused: &[], started }
    });
    Ok((e, tok, chat, defaults, vocab, line, info))
}

/// The engine of a level's plan (Gemma 4's, Qwen3's, Nemotron-H's or Qwen3.5's description on the executor) from the directory `dir`, on `gpu`: everything it holds is the
/// plan's (its context, its prompt chunk, its layout, its draft model and its expert cache); nothing is sized here. The engine, its weights' bytes and its vocabulary.
fn open_lossy(a: &Args, plan: &LoadPlan, dir: &Path, gpu: &Gpu, name: &str, ui: &RefCell<Ui>) -> Done<(Box<dyn Gen>, usize, usize)> {
    let o = crate::dynamic::Opts { ctx: plan.context, chunk: plan.chunk, graph: true, device: gpu.ordinal, layout: Some(plan.layout) };
    let oom = |e: crate::ffi::Error| {
        if e.out_of_memory() {
            Refusal::unsupported(
                format!("the GPU ran out of memory loading {name} ({} were free at the start)", gib(gpu.free)),
                "Close programs that use the GPU (nvidia-smi lists them), or ask for less context: --context 4096",
            )
        } else {
            cannot(e.msg)
        }
    };
    Ok(if plan.kind == Kind::Gemma4 {
        // (drafted where a draft model is found: the big model checks every draft token, so the tokens are its own; greedy replies draft, sampled ones step)
        let draft = gemma_draft(a, Some(dir)).filter(|_| plan.draft > 0).map(|d| (d, a.num("draft-tokens", 8usize).unwrap_or(8).clamp(1, plan.chunk - 1)));
        if let Some((d, k)) = &draft {
            ui.borrow_mut().line(&format!("Drafting with {} ({k} tokens a round; --no-draft runs without it).", d.display()));
        }
        // the dynamic engine (GLY-326): Gemma 4's description and, where one is found, its draft model's
        let c = crate::gemma4::ConfigG4::load(dir).map_err(cannot)?;
        let spec = match &draft {
            Some((d, k)) => {
                let dc = crate::gemma4::ConfigG4::load_for(d, true).map_err(cannot)?;
                Some(crate::dynamic::DraftSpec { dir: d, desc: crate::dynamic::gemma4::describe_draft(&dc, &c), k: *k })
            }
            None => None,
        };
        let mut e = crate::dynamic::Dyn::new(dir, crate::dynamic::gemma4::lowered(dir, &c).map_err(cannot)?, spec, &o).map_err(oom)?;
        if draft.is_none() {
            twelve_bit_pages(&mut e, dir, name, ui)?;
        }
        let (bytes, vocab) = (e.weights_bytes(), e.desc().vocab);
        (Box::new(e), bytes, vocab)
    } else if plan.kind == Kind::Qwen3 {
        // the executor (R3c, #139): a Llama-style dense model's description from its config, its products at the plan's level
        let mut e = crate::dynamic::dense::load(dir, &o).map_err(|e| engine_refusal(e, name, plan, gpu))?;
        twelve_bit_pages(&mut e, dir, name, ui)?;
        let (bytes, vocab) = (e.weights_bytes(), e.desc().vocab);
        (Box::new(e), bytes, vocab)
    } else if plan.kind == Kind::NemotronH {
        let store = ExpertStore::new();
        // (the experts the computer's memory does not hold stay in the packed directory: src/offload.rs `Tier`)
        if let Some(c) = plan.experts.filter(|c| c.ram < c.host_bytes()) {
            store.set_host_memory(Some(crate::offload::HostMemory { bytes: c.ram, total: c.host_bytes() }));
        }
        // an expert that is not on the GPU is computed by the CPU (--cpu-threads) where the computer's memory holds a second copy of the experts' first matrices besides the experts; else it is copied
        let mut offload = crate::offload::offload_args(a).map_err(|e| Refusal::new(e.msg, "--cpu-threads takes a number of threads: --cpu-threads 8"))?;
        if let Some(c) = plan.experts {
            if offload.mode != crate::offload::Mode::Copy && !c.host_half_fits(crate::offload::host_memory(), HOST_MARGIN) {
                offload.mode = crate::offload::Mode::Copy;
                ui.borrow_mut().line(&format!("The computer's memory holds the experts but not the {} more that the CPU needs to compute the experts that are not on the GPU: they are copied in as the router asks for them.", gib(c.host_half_bytes())));
            } else if offload.mode != crate::offload::Mode::Copy {
                let more = if c.host_half_bytes() == 0 {
                    if c.ram < c.host_bytes() {
                        "from the computer's memory (read from the disk first where they are not in it)".to_string()
                    } else {
                        "from the computer's memory they are held in".to_string()
                    }
                } else {
                    format!("{} more of the computer's memory", gib(c.host_half_bytes()))
                };
                ui.borrow_mut().line(&format!(
                    "The experts that are not on the GPU are computed by the CPU on {} threads (--cpu-threads N changes the count), {more}.",
                    offload.threads
                ));
            }
        }
        store.set_offload(offload);
        let (plain, cached) = (crate::moe_nh::factory(plan.layout), plan.experts.map(|c| crate::moe_nh::factory_cached(plan.layout, c.slots, &store)));
        let make: crate::kern_nh::MoeFactory<'_> = match &cached {
            Some(f) => f,
            None => &plain,
        };
        // the executor (GLY-354): Nemotron-H's description, its blocks of experts (an expert cache's) as `make` makes them
        let layers = crate::nemotron::ConfigNemotronH::load(dir).map_err(cannot)?.layers;
        let e = crate::dynamic::nemotron::load_with(dir, &o, make, crate::dynamic::Stage::whole(layers)).map_err(oom)?;
        let (bytes, vocab) = (e.weights_bytes(), e.desc().vocab);
        (Box::new(e), bytes, vocab)
    } else {
        // the dynamic engine (GLY-326): Qwen3.5's description
        let e = crate::dynamic::qwen35::load(dir, &o).map_err(oom)?;
        let (bytes, vocab) = (e.weights_bytes(), e.desc().vocab);
        (Box::new(e), bytes, vocab)
    })
}

/// The 12-bit lossless KV pages for `e` where the scheduler pages its cache and GLYD_KV=lossless asks for them (off unless asked for: admission does not reserve their room): the tables made on
/// the model's first run and kept in the cache.
fn twelve_bit_pages(e: &mut crate::dynamic::Dyn, dir: &Path, name: &str, ui: &RefCell<Ui>) -> Done<()> {
    if std::env::var("GLYD_KV").as_deref() != Ok("lossless") {
        return Ok(());
    }
    let t0 = std::time::Instant::now();
    let (t, made) = crate::dynamic::kv12::tables_for(e, dir, name).map_err(cannot)?;
    if made {
        ui.borrow_mut().line(&format!("Made the 12-bit KV tables of {} in {:.0} s (once: kept in the cache).", name, t0.elapsed().as_secs_f64()));
    }
    e.set_kv12(Some(t));
    Ok(())
}

/// The GPUs the dynamic route uses (Ampere or newer, as the driver lists them) and the first, which the machine's checks are of.
fn dynamic_gpus() -> Done<(Vec<Gpu>, Gpu)> {
    let gpus: Vec<Gpu> = Dev::probe_all()
        .map_err(|e| Refusal::unsupported(format!("the GPU cannot be used: {}", e.msg), "Check that nvidia-smi shows it"))?
        .into_iter()
        .filter(|g| g.cc.0 >= 8)
        .collect();
    let first = gpus
        .first()
        .cloned()
        .ok_or_else(|| Refusal::unsupported("the driver shows no GPU of the Ampere generation or newer", "Check that nvidia-smi shows it"))?;
    if let Some(r) = machine::gpu_refusal(&first, std::env::consts::ARCH, true, machine::ANY_FAMILY) {
        return Err(r);
    }
    Ok((gpus, first))
}

// ponytail: penguin only on the dynamic route; GLM's and Nemotron-H's other levels when their packs run here
fn penguin_only(name: &str, family: &str, model: &str) -> Refusal {
    let base = model.split_once(':').map_or(model, |m| m.0);
    Refusal::new(format!("{name} ({family}) runs at the penguin level only: the release's weights exactly"), format!("glyd run {base}"))
}

/// The hummingbird directory of a model on the dynamic route (src/hb_pack.rs): MODEL itself where it is one, else its pre-packed repo
/// (`glyd/NAME-hummingbird`) where it is here, else downloaded where `ui` is given and the Hub has it; or the refusal that says how to pack it.
fn hb_dir_for(model: &str, name: &str, family: &str, ui: Option<&RefCell<Ui>>) -> Done<PathBuf> {
    let is_hb = |d: &Path| matches!(crate::hb_pack::HbDir::open(d), Ok(Some(_)));
    if Path::new(model).join("glyd.json").exists() && is_hb(Path::new(model)) {
        return Ok(PathBuf::from(model));
    }
    let repo = packed_repo(model, "hummingbird");
    let held = |r: &str| Source::find(r).ok().map(|s| s.dir).filter(|d| is_hb(d));
    if let Some(d) = held(&repo) {
        return Ok(d);
    }
    if let Some(ui) = ui {
        if let Some((hub, remote, wanted)) = store::hub_plan_packed(&repo, "hummingbird", ui)? {
            store::download(&hub, &remote, &wanted, &repo, ui)?;
            if let Some(d) = held(&repo) {
                return Ok(d);
            }
        }
    }
    Err(Refusal::unsupported(
        format!("{name} ({family}) at the hummingbird level is packed ahead, from a count of each expert's use, and there is no pre-packed {repo}"),
        format!(
            "Pack it: glyd-engine calibrate-experts {model} usage.json, then glyd-engine pack {model} DIR --layout hummingbird --usage usage.json, and run DIR"
        ),
    ))
}

/// The level a dynamic-route MODEL is run at: a hummingbird directory's own where no other level is asked.
fn hb_level(model: &str, layout: Layout) -> Layout {
    if layout == Layout::Tiered && matches!(crate::hb_pack::HbDir::open(Path::new(model)), Ok(Some(_))) {
        Layout::Hummingbird
    } else {
        layout
    }
}

/// The words a dynamic-route plan says its level in.
fn dynamic_level_words(layout: Layout) -> &'static str {
    if layout == Layout::Hummingbird {
        "the hummingbird level (the routed experts at 3 and 2 bits by use, the rest the release's weights exactly)"
    } else {
        "the penguin level (lossless: the release's weights exactly)"
    }
}

/// `preflight` of a model on the dynamic route: whether the GPUs here hold it (the cut `run` makes, where the model is downloaded; its download's
/// size against all their room, where not) and the disk its download.
/// What `preflight` says of the plan of a model of the dynamic route: the bytes on the GPUs (and the experts the computer's memory holds).
fn dynamic_line(name: &str, family: &str, p: &LoadPlan) -> String {
    let sp = p.spread.as_ref().expect("the dynamic route's plan spreads the model");
    let level = p.level;
    if sp.host_bytes > 0 && level == "hummingbird" {
        return format!(
            "{name} ({family}): about {} on {} GPUs at the hummingbird level, {} of experts in the computer's memory and {} on the disk",
            gib(p.weights_on_gpu()),
            sp.devices.len(),
            gib(p.ram_bytes()),
            gib(p.disk_bytes())
        );
    }
    if sp.host_bytes > 0 {
        return format!(
            "{name} ({family}): about {} on {} GPUs at the penguin level, {} of experts in the computer's memory",
            gib(p.weights_on_gpu()),
            sp.devices.len(),
            gib(sp.host_bytes)
        );
    }
    let n = sp.devices.len();
    format!("{name} ({family}): about {} on {} at the {level} level", gib(p.weights_on_gpu()), if n == 1 { "one GPU".to_string() } else { format!("{n} GPUs") })
}

fn check_dynamic(a: &Args, model: &str, family: &str, layout: Layout, ui: &RefCell<Ui>) -> Done<String> {
    let name = short_name(model);
    if !matches!(layout, Layout::Tiered | Layout::Hummingbird) {
        return Err(penguin_only(&name, family, model));
    }
    let hb = layout == Layout::Hummingbird;
    let level = if hb { "hummingbird" } else { "penguin" };
    let (gpus, _) = dynamic_gpus()?;
    crate::ffi::Lib::find().map_err(cannot)?;
    let room = rooms(&gpus);
    let all = room.iter().map(|r| r.1).sum::<u64>();
    let local = if hb {
        Some(hb_dir_for(model, &name, family, None)?)
    } else if Path::new(model).join("config.json").exists() {
        Some(PathBuf::from(model))
    } else {
        Source::find(model).ok().map(|s| s.dir)
    };
    let (bytes, gpus_used) = match &local {
        Some(dir) => {
            let text = std::fs::read_to_string(dir.join("config.json")).map_err(|e| cannot(format!("{}: {e}", dir.display())))?;
            let targets: Vec<Target> = gpus.iter().map(target).collect();
            // (the plan `run` loads, or the refusal it gives, with Kimi-K3's expert cache where the GPUs do not hold its experts)
            let table = crate::sizes::Table::from_dir(dir).map_err(cannot)?;
            let p =
                size_dynamic(a, &name, &table, Some(dir), &text, static_family(family), &targets, crate::offload::host_memory(), &crate::dynamic::tp::reach)?;
            return Ok(dynamic_line(&name, family, &p));
        }
        None => {
            let text = config_text(model)?;
            let (hub, remote, wanted) = store::hub_plan_as(model, ui, true)?;
            let need = hub.missing(&remote, &wanted);
            let models = store::existing_ancestor(&hub::hub_cache());
            if let Some(free) = hub::disk_free(&models) {
                if free < need + (1 << 30) {
                    return Err(Refusal::unsupported(
                        format!("the disk holding {} has {} left, and {} are to be downloaded", models.display(), gb(free), gb(need)),
                        "Free some space, or set HF_HOME to a disk with room",
                    ));
                }
            }
            // the exact plan from the files' headers by ranges, nothing downloaded (src/sizes.rs)
            if let Some(table) = crate::sizes::Table::find(model) {
                let targets: Vec<Target> = gpus.iter().map(target).collect();
                let p =
                    size_dynamic(a, &name, &table, None, &text, static_family(family), &targets, crate::offload::host_memory(), &crate::dynamic::tp::reach)?;
                return Ok(dynamic_line(&name, family, &p));
            }
            // (headers not to be had: a bf16 release at penguin's 72% at most; an FP8 or FP4 one as it is; the cut is made once it is here)
            let bytes = if text.contains("\"quantization_config\"") { need } else { need * 72 / 100 };
            // ponytail: Kimi-K3 not yet here, its experts (about 93% of it) in the computer's memory and the rest on the GPUs; the exact plan is made once it is here
            if family == "Kimi-K3" && bytes > all {
                let host = crate::offload::host_memory().unwrap_or(0);
                if bytes * 93 / 100 + HOST_MARGIN > host || bytes * 7 / 100 > all {
                    return Err(Refusal::unsupported(format!("{name} keeps the experts its GPUs do not hold in the computer's memory: about {} of experts, {} free, and about {} for the GPUs, {} free", gib(bytes * 93 / 100), gib(host), gib(bytes * 7 / 100), gib(all)), "A machine with that much memory runs it"));
                }
                return Ok(format!(
                    "{name} ({family}): about {} on the GPUs at the penguin level, about {} of experts in the computer's memory",
                    gib(bytes * 7 / 100),
                    gib(bytes * 93 / 100)
                ));
            }
            (bytes, (bytes <= all).then_some(gpus.len()))
        }
    };
    match gpus_used {
        Some(n) => {
            Ok(format!("{name} ({family}): about {} on {} at the {level} level", gib(bytes), if n == 1 { "one GPU".to_string() } else { format!("{n} GPUs") }))
        }
        None => Err(Refusal::unsupported(
            format!(
                "{name} needs about {} of GPU memory at {level}, and the {} GPU{} here have {} free",
                gib(bytes),
                gpus.len(),
                if gpus.len() == 1 { "" } else { "s" },
                gib(gpus.iter().map(|g| g.free).sum::<u64>())
            ),
            "A machine whose GPUs have that much memory free runs it (nvidia-smi lists what uses them)",
        )),
    }
}

/// The directory the dynamic route reads MODEL from, made ready (a run's first step, and `glyd prepare`'s): its hummingbird directory where `hb` (`hb_dir_for`),
/// else the model's own, the cache's copy, else downloaded.
fn dynamic_dir(model: &str, name: &str, family: &str, hb: bool, ui: &RefCell<Ui>) -> Done<PathBuf> {
    if hb {
        return hb_dir_for(model, name, family, Some(ui));
    }
    if Path::new(model).join("config.json").exists() {
        return Ok(PathBuf::from(model));
    }
    match Source::find(model) {
        Ok(s) => Ok(s.dir),
        Err(_) => {
            let (hub, remote, wanted) = store::hub_plan_as(model, ui, true)?;
            store::download(&hub, &remote, &wanted, model, ui)?;
            Ok(Source::find(model).map_err(cannot)?.dir)
        }
    }
}

/// The plan of the dynamic route's model in `dir` over this machine's GPUs (`size_dynamic` on its files' headers), or its refusal: the directory's config.json, the
/// GPUs, the first of them and the plan.
fn dynamic_plan(a: &Args, name: &str, family: &str, dir: &Path) -> Done<(String, Vec<Gpu>, Gpu, LoadPlan)> {
    let text = std::fs::read_to_string(dir.join("config.json")).map_err(|e| cannot(format!("{}: {e}", dir.display())))?;
    let (gpus, first) = dynamic_gpus()?;
    let targets: Vec<Target> = gpus.iter().map(target).collect();
    let table = crate::sizes::Table::from_dir(dir).map_err(cannot)?;
    let plan = size_dynamic(a, name, &table, Some(dir), &text, static_family(family), &targets, crate::offload::host_memory(), &crate::dynamic::tp::reach)?;
    Ok((text, gpus, first, plan))
}

/// `run`/`serve` of a model the dynamic engine runs from its release's directory (`dynamic_family`), at penguin: the release's weights exactly (an FP8 or
/// FP4 release's as it has them, a bf16 one's lossless). The directory is the model's own, the cache's copy, else downloaded; its layers are cut over the
/// machine's GPUs as their free memory holds them (`--split CUTS --devices GPUS` chooses), and `--context N` (8,192) is the KV's. What load_lossy gives.
fn load_dynamic(
    a: &Args,
    model: &str,
    family: &str,
    layout: Layout,
    served: &str,
    ui: &RefCell<Ui>,
) -> Done<(Box<dyn Gen>, Tok, Option<Chat>, Params, usize, String, Option<server::Info>)> {
    let (t0, started) = (Instant::now(), crate::monitor::clock());
    let name = short_name(model);
    if !matches!(layout, Layout::Tiered | Layout::Hummingbird) {
        return Err(penguin_only(&name, family, model));
    }
    let hb = layout == Layout::Hummingbird;
    let dir = dynamic_dir(model, &name, family, hb, ui)?;
    let (text, gpus, first, plan) = dynamic_plan(a, &name, family, &dir)?;
    let nl = crate::dynamic::split::config_num(&text, "num_hidden_layers").unwrap_or(0);
    let sp = plan.spread.clone().expect("the dynamic route's plan spreads the model");
    let (total, coded, across) = (plan.weights_on_gpu(), sp.coded, sp.across);
    let (cuts, devices) = (sp.cuts.clone(), sp.devices.clone());
    let cache = (sp.host_bytes > 0).then_some(&sp);
    let gpu_name = |d: i32| gpus.iter().find(|g| g.ordinal == d).map_or_else(|| format!("GPU {d}"), |g| g.name.clone());
    let mut bounds: Vec<usize> = std::iter::once(0).chain(cuts.iter().copied()).chain(std::iter::once(nl)).collect();
    bounds.dedup();
    let stages: Vec<String> = if across {
        vec![format!("every layer across GPUs {devices:?}")]
    } else {
        devices.iter().zip(bounds.windows(2)).map(|(&d, w)| format!("layers {}..{} on GPU {d} ({})", w[0], w[1], gpu_name(d))).collect()
    };
    ui.borrow_mut().line(&format!(
        "{name} ({family}): about {} on the GPU{} at {}; {}.",
        gib(total),
        if devices.len() == 1 { "" } else { "s" },
        dynamic_level_words(layout),
        stages.join(", ")
    ));
    ui.borrow_mut().line(&format!("Plan: {}.", plan.text()));
    if let Some(n) = machine::cuda12_note(first.driver, crate::dev::has_kernels12()) {
        ui.borrow_mut().line(&format!("Note: {n}"));
    }
    if let Some(k) = &cache {
        let (lo, hi) = (k.places.iter().min().copied().unwrap_or(0), k.places.iter().max().copied().unwrap_or(0));
        let n = if lo == hi { lo.to_string() } else { format!("{lo} to {hi}") };
        let held = if plan.disk_bytes() > 0 {
            format!("{} of them in the computer's memory and {} read from the disk as the router asks for them", gib(plan.ram_bytes()), gib(plan.disk_bytes()))
        } else {
            format!("all of them are in the computer's memory ({})", gib(k.host_bytes))
        };
        ui.borrow_mut().line(&format!("{n} experts of {} a layer are on the GPUs, chosen as the router asks for them; {held}.", k.experts_per_layer));
    }
    if sp.dense > 0 {
        // (P2d's rung after the experts: every step reads these layers whole over the link)
        let ms = sp.dense_bytes as f64 / (crate::plan::LINK_GBS * 1e9) * 1e3;
        ui.borrow_mut().line(&format!(
            "The last {} layers' dense matrices ({}) are in the computer's memory too, each copied to the GPU before it runs: about {ms:.0} ms more a token.",
            sp.dense,
            gib(sp.dense_bytes)
        ));
    }
    if coded {
        // (the release's FP8 bytes plain do not fit these GPUs: coded, exactly the same values, in about 7 of 8 bits)
        crate::ffi::Lib::find().and_then(|l| l.f8h_plain(Some(false))).map_err(cannot)?;
        ui.borrow_mut().line("Its FP8 matrices are held coded (the same values in about 7 bits of 8) for it to fit; as they are, they do not.");
    }
    ui.borrow_mut().status(&format!("Reading {name} and loading it onto the GPU{}...", if devices.len() == 1 { "" } else { "s" }));
    let mut e = open_dynamic(a, &plan, &dir)?.map_err(|e| {
        if e.out_of_memory() {
            Refusal::unsupported(
                format!("a GPU ran out of memory loading {name} ({})", stages.join(", ")),
                "Ask for less context (--context 4096), or cut the layers yourself: --split CUTS --devices GPUS",
            )
        } else {
            cannot(e.msg)
        }
    })?;
    // (the weights the engine holds on one GPU; stages and layers across GPUs hold theirs on their own threads and say none: 0)
    let weights = crate::conv::Gen::held(&e).map_or(0, |h| h.weights);
    loaded_report(model, &plan, &gpus.iter().filter(|g| devices.contains(&g.ordinal)).collect::<Vec<_>>(), weights);
    crate::plan::first_use(&mut e, crate::dynamic::split::config_num(&text, "vocab_size").unwrap_or(0));
    let tok = Tok::load(&dir).map_err(cannot)?;
    // (DeepSeek-V4's and Kimi-K3's releases have their chat formats in Python alone: ours as templates)
    let own = match family {
        "DeepSeek-V4" => Some(crate::chat::DSV4_CHAT),
        "Kimi-K3" => Some(crate::chat::KIMI_CHAT),
        _ => None,
    };
    let chat = Chat::load_or(&dir, own).map_err(cannot)?;
    let mut defaults = sampler::generation_defaults(&dir);
    if let Some(t) = a.flag("temp") {
        defaults.temperature = t.parse().map_err(|_| Refusal::new(format!("--temp {t}: not a number"), ""))?;
    }
    let distinct = devices.iter().collect::<std::collections::BTreeSet<_>>().len();
    let on = if distinct == 1 { format!("the {}", gpu_name(devices[0])) } else { format!("{distinct} GPUs") };
    let line = format!(
        "Ready in {}: {name} on {on}, {} of weights, {} tokens of context, one user at a time.",
        fmt_time(t0.elapsed().as_secs_f64()),
        gib(total),
        e.capacity()
    );
    let info = (!served.is_empty()).then(|| {
        let held = e.held();
        let gpu = gpus.iter().find(|g| g.ordinal == devices[0]).unwrap_or(&first);
        let (level, exact, words) = if hb {
            ("hummingbird", false, "hummingbird level: the routed experts at 3 and 2 bits by use, the rest the release's weights exactly")
        } else {
            ("penguin", true, "penguin level: the release's weights exactly")
        };
        let about = about(served, &dir, requested(served, a.has("lossy")), level, exact, words, held.as_ref(), gpu);
        server::Info { about, gpu: Some(devices[0]), held, concurrency: 1, window: e.capacity(), refused: &[], started }
    });
    let vocab = crate::dynamic::split::config_num(&text, "vocab_size").unwrap_or(0);
    Ok((Box::new(e), tok, chat, defaults, vocab, line, info))
}

/// The engine of a dynamic-route plan from the directory `dir` (src/dynamic/split.rs `open_plan`: it loads what the plan says), the flags' copy-or-compute policy
/// of its expert cache.
fn open_dynamic(a: &Args, plan: &LoadPlan, dir: &Path) -> Done<std::result::Result<crate::dynamic::split::Run, crate::ffi::Error>> {
    let offload = crate::offload::offload_args(a).map_err(|e| Refusal::new(e.msg, "--cpu-threads takes a number of threads: --cpu-threads 8"))?;
    Ok(crate::dynamic::split::open_plan(dir, plan, offload))
}

/// `run MODEL:kestrel` (or `:swift`, or `--lossy`): the model (a directory in the Hub's layout, the cache's copy, else downloaded), every weight rounded and packed in `layout` once into the cache and read from there (a glyd-v7 directory: read
/// as it is, at its level; a model with a pre-packed repo on the Hub, `glyd/NAME-LEVEL`: that is downloaded in place of the bf16 model), then one answer or the terminal's chat.
fn run_lossy(a: &Args, layout: Layout, ui: &RefCell<Ui>, mut prompt: Option<String>, piped: bool, thinking: Option<bool>) -> i32 {
    let started = load_lossy(a, layout, "", ui);
    let (mut e, tok, chat, defaults, _vocab, line, _) = match started {
        Ok(x) => x,
        Err(r) => return fail(ui, &r),
    };
    ui.borrow_mut().line(&line);
    STATUS.ready();
    if piped {
        let mut t = String::new();
        let _ = std::io::stdin().read_to_string(&mut t);
        if t.trim().is_empty() {
            return fail(ui, &no_prompt());
        }
        prompt = Some(t);
    }
    // (a development build's GLYD_PROF=1: the executor's GPU profile of the run, on stderr at its end: src/dynamic/prof.rs)
    let profile = || {
        if crate::dynamic::prof::on() {
            eprint!("{}", crate::dynamic::prof::text(&crate::dynamic::prof::snapshot(false)));
        }
    };
    if let Some(p) = prompt {
        let code = once(&mut Talk { e: &mut *e, tok: &tok, chat: chat.as_ref() }, &defaults, p.trim(), thinking, ui);
        profile();
        return code;
    }
    ui.borrow_mut().line("Chat here. /bye leaves, /clear starts again, /think turns thinking on or off.");
    let (tx, rx) = mpsc::channel();
    let tty = std::io::stdout().is_terminal();
    let chat_defaults = defaults.clone();
    let t = std::thread::spawn(move || chat_loop(tx, chat_defaults, thinking, tty));
    work_lossy(&mut *e, &tok, chat.as_ref(), &rx);
    let _ = t.join();
    profile();
    0
}

// ---- the other commands ---------------------------------------------------------------------------------------------------------------------------

/// The model's config.json as text: from the directory it is, or its snapshot in the cache, else from the Hub (that one file: nothing is downloaded).
pub(crate) fn config_text(model: &str) -> Done<String> {
    let read = |p: PathBuf| std::fs::read_to_string(&p).map_err(|e| Refusal::new(format!("{}: {e}", p.display()), ""));
    if Path::new(model).join("config.json").exists() {
        return read(Path::new(model).join("config.json"));
    }
    if let Ok(s) = Source::find(model) {
        return read(s.dir.join("config.json"));
    }
    if store::looks_like_repo(model) && !hub::offline() {
        return Hub::from_env().text(model, "main", "config.json").map_err(|e| match e {
            hub::HubError::Denied(_) => store::hub_refusal(model),
            e => Refusal::new(format!("cannot read {model} from the Hugging Face Hub ({e})"), "Check the network connection"),
        });
    }
    Err(Refusal::new(
        format!("{model}: no such directory, and not a model on the Hugging Face Hub"),
        "Give a directory with the model, or its Hub name (ORG/NAME)",
    ))
}

/// Whether the engine runs MODEL: exit 0 where it does, 3 (with the reason on stderr) where it does not, 1 where that cannot be told. The kind of model, and its level where one is asked (`MODEL:LEVEL`, or
/// `--lossy`), against the table `fit` and `run` read (src/support.rs): no GPU is looked at (`preflight` looks at the computer).
pub fn supports(a: &Args) -> i32 {
    let ui = RefCell::new(Ui::new());
    let Ok(model) = a.need(1, "MODEL") else { return 1 };
    match supports_check(model, a.has("lossy")) {
        Ok(()) if a.has("json") => {
            println!("{}", serde_json::json!({"model": model, "ok": true, "exit": 0, "supported": true}));
            0
        }
        Ok(()) => 0,
        Err(r) if a.has("json") => json_refusal(&ui, model, &r),
        Err(r) => fail(&ui, &r),
    }
}

/// `supports`' answer for `tagged` (MODEL with its level, if one is asked): the family, at the level where there is one, and for a plain dense decoder what the dense kind's lowering does not take
/// (`supported`). A model packed at a level (a glyd-v7 directory, a pre-packed repo) is at its own.
fn supports_check(tagged: &str, lossy: bool) -> Done<()> {
    let (model, layout) = level(tagged, lossy)?;
    // (a level is asked by a tag or --lossy; the lossless one has no layout, so a tag's presence says it)
    let asked = (model.len() != tagged.len() || layout.is_some()).then(|| layout.map_or("penguin", level_name));
    let name = short_name(model);
    let text = config_text(model)?;
    if v7_layout(model)?.is_some() || levels::repo_level(model).is_some() {
        return Ok(());
    }
    crate::support::family(&text, &name)?;
    if let Some(level) = asked {
        crate::support::check(&text, model, &name, level)?;
    }
    if crate::dynamic::dense::is_dense(&text) {
        supported(&name, &text)?;
    }
    Ok(())
}

/// A refusal as `--json` says it: on the standard output, with the exit status it has (3: not supported, 1: not told).
fn json_refusal(ui: &RefCell<Ui>, model: &str, r: &Refusal) -> i32 {
    ui.borrow_mut().clear_live();
    println!("{}", serde_json::json!({"model": model, "ok": false, "exit": r.code, "reason": r.what, "fix": r.fix}));
    r.code
}

/// Whether the engine will certainly run MODEL on this computer, said before anything is downloaded: exit 0 where it will (one line on stdout says on what), 3 where it will not (the reason on
/// stderr), 1 where that could not be told (the Hub did not answer). The checks are `supports`' (the kind of model), the GPU a run would use and its driver, the model's shape and the memory it
/// takes there at the context asked for (the other flags are `run`'s), what a first run writes to the disks, and the KV cache's tables. Without a MODEL: this computer's GPU alone.
pub fn preflight(a: &Args) -> i32 {
    let ui = RefCell::new(Ui::new());
    if a.has("costs") {
        // (the model's cost profile at the level, from its files: what glyd run's planner and the fit page read; docs/cost-profile.md)
        return match a.need(1, "MODEL[:LEVEL]").and_then(|m| crate::costs::costs_json(m, a)) {
            Ok(v) => {
                println!("{}", json::to_python(&v, 2));
                0
            }
            Err(e) => fail(&ui, &cannot(e.msg)),
        };
    }
    match check(a, &ui) {
        Ok(line) => {
            ui.borrow_mut().clear_live();
            if a.has("json") {
                println!("{}", serde_json::json!({"model": a.pos.get(1), "ok": true, "exit": 0, "line": line}));
            } else {
                println!("{line}");
            }
            0
        }
        Err(r) if a.has("json") => json_refusal(&ui, a.pos.get(1).map_or("", String::as_str), &r),
        Err(r) => fail(&ui, &r),
    }
}

fn check(a: &Args, ui: &RefCell<Ui>) -> Done<String> {
    // MODEL's level (its tag, or --lossy) before anything reads the model; below, the model is without its tag
    let (model, layout) = match a.pos.get(1) {
        Some(m) => level(m, a.has("lossy")).map(|(m, l)| (Some(m), l))?,
        None => (None, None),
    };
    let layout = match model {
        Some(m) => load_level(a, m, layout)?,
        None => layout,
    };
    if let Some(m) = model {
        table_check(m, layout)?;
    }
    match (model, layout) {
        (Some(m), Some(layout)) => check_lossy(a, m, layout, ui),
        (Some(m), None) => Err(unrun(m)),
        // the GPU a run would use and its driver, with no model
        (None, _) => {
            let gpu = gpu_pick()?;
            if let Some(r) = machine::gpu_refusal(&gpu, std::env::consts::ARCH, true, machine::ANY_FAMILY) {
                return Err(r);
            }
            Ok(format!("{} ({} free)", gpu.name, gib(gpu.free)))
        }
    }
}

/// `preflight MODEL:kestrel` (or `:swift`, or `--lossy`): the kind of model, the GPU and its driver (the list of GPUs measured against vLLM is not looked at: vLLM has no lossy levels), the memory, the disk a
/// download takes.
fn check_lossy(a: &Args, model: &str, layout: Layout, ui: &RefCell<Ui>) -> Done<String> {
    match dynamic_family(model) {
        Some(family) if lossy_route(family, layout) => {}
        Some(family) => return check_dynamic(a, model, family, hb_level(model, layout), ui),
        None => {}
    }
    let text = config_text(model)?;
    let gpu = gpu_pick()?;
    // (a plain dense decoder at penguin goes to the engine by the measured list, `machine::GPUS`, where vLLM is here to run it otherwise: GLYD_ENGINE=1, passed on by glyd run
    // and glyd serve, asks for the engine by name; every other path is the engine's whatever the list)
    let dense = crate::dynamic::dense::is_dense(&text) && layout == Layout::Tiered;
    let (asked, family) = match dense {
        true => (machine::engine_chosen(std::env::var("GLYD_ENGINE_ASKED").is_ok_and(|v| v.trim() == "1")), machine::QWEN3),
        false => (true, machine::ANY_FAMILY),
    };
    if let Some(r) = machine::gpu_refusal(&gpu, std::env::consts::ARCH, asked, family) {
        return Err(r);
    }
    let plan = plan_level(a, model, layout, &text, None, &gpu, v7_layout(model)?.map(|_| Path::new(model)))?;
    crate::ffi::Lib::find().map_err(cannot)?;
    Dev::new(gpu.ordinal).map_err(|e| cannot(e.msg))?;
    let (level, repo) = (level_name(layout), packed_repo(model, level_name(layout)));
    if !Path::new(model).join("config.json").exists() && Source::find(model).is_err() && Source::find(&repo).is_err() {
        // (a model with a pre-packed repo downloads that one, which is smaller)
        let planned = if levels::repo_level(model).is_some() { None } else { store::hub_plan_packed(&repo, level, ui)? };
        let (hub, remote, wanted) = match planned {
            Some(p) => p,
            None => store::hub_plan(model, ui)?,
        };
        let need = hub.missing(&remote, &wanted);
        let models = store::existing_ancestor(&hub::hub_cache());
        if let Some(free) = hub::disk_free(&models) {
            if free < need + (1 << 30) {
                return Err(Refusal::unsupported(
                    format!("the disk holding {} has {} left, and {} are to be downloaded", models.display(), gb(free), gb(need)),
                    format!("Free some space, or keep the models on another drive: HF_HOME=/path/on/that/drive glyd run {model}:{}", level_name(layout)),
                ));
            }
        }
    }
    Ok(format!("{}: {}, {} tokens of context, {} level", gpu.name, short_name(model), plan.context, level_name(layout)))
}

/// MODEL made ready at its level (downloaded, packed for this GPU): what `run` does first, for any model it runs. The directory is printed.
pub fn prepare_cmd(a: &Args) -> i32 {
    let ui = RefCell::new(Ui::new());
    match prepare(a, &ui) {
        Ok(dir) => {
            println!("{}", dir.display());
            0
        }
        Err(r) => fail(&ui, &r),
    }
}

/// MODEL's files downloaded into the Hub cache, no GPU needed (what `run` does first).
pub fn fetch(a: &Args) -> i32 {
    let ui = RefCell::new(Ui::new());
    let Ok(tagged) = a.need(1, "MODEL") else { return 1 };
    let r = (|| -> Done<()> {
        // MODEL:LEVEL fetches the level's pre-packed repo (google/gemma-4-12B-it:wren is glyd/gemma-4-12B-wren)
        let (rest, layout) = level(tagged, false)?;
        table_check(rest, layout)?;
        let dynamic = dynamic_family(rest).filter(|f| !layout.is_some_and(|l| lossy_route(f, l)));
        if let (Some(family), Some(l)) = (dynamic, layout) {
            if l != Layout::Tiered {
                return Err(penguin_only(&short_name(rest), family, rest));
            }
        }
        let packed = if dynamic.is_some() { None } else { layout.map(|l| packed_repo(rest, level_name(layout_for(rest, l)))) };
        let model = packed.as_deref().unwrap_or(rest);
        let (hub, remote, wanted) = store::hub_plan_as(model, &ui, dynamic.is_some())?;
        let text = hub.text(model, "main", "config.json").map_err(|e| match e {
            hub::HubError::Denied(_) => store::hub_refusal(model),
            e => Refusal::new(format!("cannot read {model} from the Hugging Face Hub ({e})"), "Check the network connection and run this again"),
        })?;
        // (a pre-packed level's model is one the levels' engines run, which the run checks; so is a Gemma 4 model, its draft model too, which runs penguin without a tag)
        if packed.is_none() && dynamic.is_none() && !crate::gemma4::is_gemma4(&text) {
            supported(&short_name(model), &text)?;
        }
        store::download(&hub, &remote, &wanted, model, &ui)
    })();
    match r {
        Ok(()) => 0,
        Err(r) => fail(&ui, &r),
    }
}

/// MODEL's glyd save (a directory, or a Hub repo id in the local cache) checked bit for bit: every pack decoded on the CPU and
/// every tensor against the sha256 glyd.json holds. Lossless (penguin) and wren of a checkpoint trained to four bits hold their
/// source's sha256 (the pack refuses a matrix it cannot give back exactly), so a pass there is the source's weights, every bit.
pub fn verify(a: &Args) -> i32 {
    let Ok(model) = a.need(1, "MODEL") else { return 1 };
    let dir = if Path::new(model).join("glyd.json").exists() {
        PathBuf::from(model)
    } else {
        match Source::find(model) {
            Ok(s) => s.dir,
            Err(e) => {
                eprintln!("glyd verify: {e}");
                return 1;
            }
        }
    };
    let threads = std::thread::available_parallelism().map_or(8, |n| n.get());
    let t = Instant::now();
    match glyd_gpu::save::verify(&dir, threads, &glyd_gpu::save::Decoder::Cpu) {
        Ok(v) => {
            let m = std::fs::read_to_string(dir.join("glyd.json")).ok().and_then(|t| json::parse(&t).ok());
            let field = |k: &str| m.as_ref().and_then(|m| m.get(k)).and_then(Value::as_str).map(str::to_string);
            let source = m.as_ref().and_then(|m| m.get("source")).and_then(|s| s.get("repo")).and_then(Value::as_str).unwrap_or("its source").to_string();
            let level = field("level").unwrap_or_else(|| "penguin".into());
            // (the exact four-bit layout is wren of a QAT checkpoint: its packs refuse anything but its source's weights)
            let exact = level == "penguin" || field("layout").as_deref() == Some(Layout::Int4H.name());
            println!(
                "{}: {} packed tensors decode to their sha256, {} kept as they are match theirs{} ({:.0} s, {threads} threads)",
                dir.display(),
                v.packed,
                v.hashed,
                if v.unchecked > 0 { format!(", {} with no sha256 (saved before glyd 0.25)", v.unchecked) } else { String::new() },
                t.elapsed().as_secs_f64()
            );
            println!(
                "{}",
                if exact {
                    format!("level {level}: every weight is {source}'s, bit for bit")
                } else {
                    format!("level {level}: the save is intact (its weights are rounded from {source}'s, as the level says)")
                }
            );
            for k in &v.skipped {
                eprintln!("glyd verify: {k:?}: a key of a newer glyd's save than this one ({VERSION}), skipped");
            }
            0
        }
        Err(e) => {
            eprintln!("glyd verify: FAILED: {}: {e}", dir.display());
            1
        }
    }
}

/// What `doctor` says of a driver below `machine::driver_min` (this build's kernels are CUDA 13's, and CUDA 12's where it has them): what it runs, what the engine needs and the exact update; None where it is new enough.
pub fn driver_problem(g: &Gpu) -> Option<String> {
    (g.driver < machine::driver_min(g.cc)).then(|| format!("too old: {}", machine::driver_fix(g.driver, g.cc, "run the model with vLLM")))
}

/// `glyd doctor [--json] [--bundle [FILE]]`: src/doctor.rs.
pub fn doctor(a: &Args) -> i32 {
    crate::doctor::run(a)
}

/// The terminal's echo off while a secret is typed, and back on however the read ends (Ctrl-C included: the handler puts the terminal back before it leaves).
#[cfg(unix)]
mod echo {
    use std::sync::atomic::{AtomicBool, Ordering};
    // glibc's struct termios on Linux (x86-64 and aarch64): 60 bytes, c_lflag the fourth 32-bit word; ECHO is octal 10.
    static mut SAVED: [u32; 32] = [0; 32];
    static OFF: AtomicBool = AtomicBool::new(false);
    extern "C" {
        fn tcgetattr(fd: i32, t: *mut u32) -> i32;
        fn tcsetattr(fd: i32, how: i32, t: *const u32) -> i32;
        fn signal(sig: i32, handler: usize) -> usize;
        fn _exit(code: i32) -> !;
    }
    extern "C" fn on_int(_: i32) {
        restore();
        // SAFETY: _exit is async-signal-safe.
        unsafe { _exit(130) }
    }
    /// Echo off on standard input; false where it cannot be (not a terminal, or not Linux, whose termios this is).
    pub fn off() -> bool {
        if !cfg!(target_os = "linux") {
            return false;
        }
        // SAFETY: SAVED is only touched here and in `restore`, on this thread or in the handler this thread's read is waiting under; the buffer is larger than any platform's termios.
        unsafe {
            let saved = std::ptr::addr_of_mut!(SAVED) as *mut u32;
            if tcgetattr(0, saved) != 0 {
                return false;
            }
            let mut t = [0u32; 32];
            std::ptr::copy_nonoverlapping(saved, t.as_mut_ptr(), 32);
            t[3] &= !0o10;
            if tcsetattr(0, 0, t.as_ptr()) != 0 {
                return false;
            }
            OFF.store(true, Ordering::SeqCst);
            signal(2, on_int as *const () as usize);
        }
        true
    }
    pub fn restore() {
        if OFF.swap(false, Ordering::SeqCst) {
            // SAFETY: the terminal settings `off` saved.
            unsafe {
                tcsetattr(0, 0, std::ptr::addr_of!(SAVED) as *const u32);
            }
        }
    }
}

/// Windows: the console keeps its echo (`off` is false: the caller says the token will show).
// ponytail: SetConsoleMode without ENABLE_ECHO_INPUT would hide it, if a Windows user asks
#[cfg(not(unix))]
mod echo {
    pub fn off() -> bool {
        false
    }
    pub fn restore() {}
}

/// A Hugging Face token read from a line and saved where `hf login` keeps it.
pub fn login(_a: &Args) -> i32 {
    let ui = RefCell::new(Ui::new());
    let tty = std::io::stdin().is_terminal();
    if tty {
        eprintln!("Paste a Hugging Face token (a read token from https://huggingface.co/settings/tokens), then press Enter. It is not shown.");
    }
    let hidden = tty && echo::off();
    let mut line = String::new();
    let _ = std::io::stdin().lock().read_line(&mut line);
    if hidden {
        echo::restore();
        eprintln!();
    }
    let token = line.trim().to_string();
    if token.is_empty() {
        return fail(&ui, &Refusal::new("no token was given", "Run glyd login again and paste one"));
    }
    let h = Hub::new(h_endpoint().as_str(), Some(token.clone()), hub::hub_cache(), 1 << 20, 1);
    match h.whoami() {
        Ok(who) => match hub::save_token(&token) {
            Ok(p) => {
                println!("Logged in{}; the token is saved in {}.", if who.is_empty() { String::new() } else { format!(" as {who}") }, p.display());
                0
            }
            Err(e) => fail(&ui, &Refusal::new(format!("cannot save the token: {e}"), "Set HF_TOKEN=TOKEN in the environment instead")),
        },
        Err(hub::HubError::Denied(_)) => {
            fail(&ui, &Refusal::new("the Hub did not accept that token", "Make a new one at https://huggingface.co/settings/tokens and run glyd login again"))
        }
        Err(e) => fail(&ui, &Refusal::new(format!("cannot reach the Hugging Face Hub ({e})"), "Check the network connection and run this again")),
    }
}

fn h_endpoint() -> String {
    std::env::var("HF_ENDPOINT").ok().filter(|v| !v.is_empty()).unwrap_or_else(|| "https://huggingface.co".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::sizing::{expert_cache_for, hb_cache, held_in_memory, lossy_plan_with, release_cache};
    use crate::plan::{GEMMA_CHUNK, LOSSY_MIN_CTX, LOSSY_WORK};

    /// `serve --with`'s two models in step: what one tells the other arrives in order, and a model that fails (its thread ended, its end of the pair gone) ends the other's wait at
    /// once (None) instead of leaving it waiting for a model that is not coming.
    #[test]
    fn a_pair_hears_its_partner_in_order_and_hears_none_once_it_is_gone() {
        let ((to_two, from_one), (to_one, from_two)) = (mpsc::channel(), mpsc::channel());
        let one = Pair { first: true, tell: to_two, hear: from_two };
        let two = std::thread::spawn(move || {
            let two = Pair { first: false, tell: to_one, hear: from_one };
            let heard = (two.hear(), two.hear());
            two.tell(8001);
            heard
            // (its end of the pair goes with the thread)
        });
        one.tell(0);
        one.tell(8000);
        assert_eq!(two.join().unwrap(), (Some(0), Some(8000)), "the first's memory made, then its port");
        assert_eq!(one.hear(), Some(8001));
        let t = Instant::now();
        assert_eq!(one.hear(), None, "a partner that is gone ends the wait");
        assert!(t.elapsed() < std::time::Duration::from_secs(1));
    }

    /// R12: a release's expert cache on the dynamic route (`release_cache`), a tiny DeepSeek-V4's: with the GPU's room for its weights besides the experts and a third of the experts, places for some of each layer's on one
    /// GPU and every expert in memory; with less memory than its experts, refused (the disk tier is hummingbird's).
    #[test]
    fn a_releases_experts_are_cached_in_memory() {
        let dir = std::env::temp_dir().join(format!("glyd-cli-relcache-{}", std::process::id()));
        crate::hb_pack::tests::release(&dir, "dsv4_model");
        let text = std::fs::read_to_string(dir.join("config.json")).unwrap();
        let nl = crate::dynamic::split::config_num(&text, "num_hidden_layers").unwrap();
        let expert = |n: &str| n.contains(".experts.");
        let (per, _) = crate::dynamic::split::layer_bytes_where(&dir, nl, 100, 100, &expert).unwrap();
        let (dense, base) =
            crate::dynamic::split::layer_bytes_where(&dir, nl, crate::dynamic::split::CODED_PCT, crate::dynamic::split::TIERED_PCT, &|n| !expert(n)).unwrap();
        let all: u64 = per.iter().sum();
        assert!(all > 0);
        let sh = crate::plan::Shape::from_config(&text).unwrap();
        let kv = ((sh.kv_full * 16.0 + sh.kv_sliding * 16f64.min(sh.window.max(1) as f64) + sh.state) / nl as f64) as u64 * nl as u64;
        let room = vec![(0, dense.iter().sum::<u64>() + base + kv + all / 3)];
        let table = crate::sizes::Table::from_dir(&dir).unwrap();
        let (cuts, devices, places, on_gpus, got_all, experts, dense) =
            release_cache(&table, &text, nl, &room, "tiny", 16, false, Some(all + HOST_MARGIN + (1 << 30))).unwrap();
        // (room for a token's experts beside the layers' weights: no dense layer in the computer's memory; DeepSeek-V4's loader keeps none there)
        assert_eq!(dense, (0, 0));
        assert!(cuts.is_empty() && devices == vec![0] && places[0] > 0 && places[0] < experts, "{places:?} of {experts}");
        assert_eq!(got_all, all);
        assert!(on_gpus <= room[0].1);
        let e = release_cache(&table, &text, nl, &room, "tiny", 16, false, Some(all / 2 + HOST_MARGIN)).unwrap_err();
        assert!(e.what.contains("keeps the experts its GPUs do not hold in the computer's memory"), "{}", e.what);
        // (the plan `run` loads and `fit` prints, on a GPU of that room: the cache's layers are the model's layers of experts, and its bytes all the experts')
        let free = (0..40).fold((0u64, 64u64 << 30), |(lo, hi), _| {
            let mid = lo + (hi - lo) / 2;
            if crate::plan::dynamic_room_of(mid) >= room[0].1 {
                (lo, mid)
            } else {
                (mid + 1, hi)
            }
        });
        let gpu = Target { ordinal: 0, name: "NVIDIA GeForce RTX 4080 SUPER".into(), cc: (8, 9), total: free.1, free: free.1, host: None, cublas: Some(false) };
        let host = Some(all + HOST_MARGIN + (1 << 30));
        let p = size_dynamic(&args(&["run", "tiny", "--context", "16"]), "tiny", &table, Some(&dir), &text, "DeepSeek-V4", &[gpu], host, &|_| false).unwrap();
        let c = p.experts.expect("a release's expert cache is in the plan");
        assert_eq!((c.layers, c.experts), (per.iter().filter(|&&b| b > 0).count(), experts));
        assert_eq!((c.host_bytes(), p.ram_bytes(), p.disk_bytes()), (all, all, 0));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// A hummingbird directory's expert cache on the dynamic route (`hb_cache`), a tiny GLM-5.3-Flash's: with the GPU's room for its weights besides the experts and a third of the experts, places for some of each layer's
    /// on one GPU and every expert in memory where the computer has room for them; its memory less than all of them, the rest on the disk; less than a token's experts in every layer, refused.
    #[test]
    fn a_hummingbird_directory_is_cached_in_memory_and_on_the_disk() {
        let dir = std::env::temp_dir().join(format!("glyd-cli-hbcache-{}", std::process::id()));
        let src = dir.join("src");
        let d = crate::hb_pack::tests::release(&src, "glm53flash_model");
        let u = crate::hb_codec::Usage {
            model: "tiny".into(),
            tokens: 1,
            layers: crate::hb_pack::moe_layers(&d)
                .unwrap()
                .iter()
                .map(|m| crate::hb_codec::LayerUse { block: m.block, counts: (0..m.experts as u64).collect() })
                .collect(),
        };
        std::fs::write(dir.join("usage.json"), u.to_json()).unwrap();
        let out = dir.join("out");
        crate::hb_pack::run(src.to_str().unwrap(), &out, 2, &dir.join("usage.json")).unwrap();
        let text = std::fs::read_to_string(out.join("config.json")).unwrap();
        let nl = crate::dynamic::split::config_num(&text, "num_hidden_layers").unwrap();
        let hb = crate::hb_pack::HbDir::open(&out).unwrap().unwrap();
        let all: u64 = hb.layers.iter().map(|r| crate::hbg_store::layer_store_bytes(r).unwrap()).sum();
        let (dense, base) =
            crate::dynamic::split::layer_bytes_with(&out, nl, crate::dynamic::split::CODED_PCT, crate::dynamic::split::TIERED_PCT, &|_| true, false).unwrap();
        let sh = crate::plan::Shape::from_config(&text).unwrap();
        let kv = ((sh.kv_full * 16.0 + sh.kv_sliding * 16f64.min(sh.window.max(1) as f64) + sh.state) / nl as f64) as u64 * nl as u64;
        let room = vec![(0, dense.iter().sum::<u64>() + base + kv + all / 3)];
        let (cuts, devices, places, on_gpus, got_all, experts, ram, dense) =
            hb_cache(&out, &text, nl, &room, "tiny", 16, false, Some(all + HOST_MARGIN + (1 << 30))).unwrap();
        assert_eq!(dense, (0, 0));
        assert!(cuts.is_empty() && devices == vec![0] && places.len() == 1, "{cuts:?} {devices:?} {places:?}");
        assert!(places[0] > 0 && places[0] < experts, "{places:?} of {experts}");
        assert_eq!((got_all, ram), (all, all));
        assert!(on_gpus <= room[0].1);
        // memory for half of them: the rest on the disk
        let (.., ram, _) = hb_cache(&out, &text, nl, &room, "tiny", 16, false, Some(all / 2 + HOST_MARGIN)).unwrap();
        assert_eq!(ram, all / 2);
        // none to give: refused
        let e = hb_cache(&out, &text, nl, &room, "tiny", 16, false, Some(HOST_MARGIN)).unwrap_err();
        assert!(e.what.contains("in the computer's memory and on the disk"), "{}", e.what);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// wren on a QAT checkpoint: the exact four-bit layout, its pre-packed repo by its short name, and a saved i4x directory taken as wren.

    #[test]
    fn wren_on_a_qat_checkpoint_is_exact() {
        let m = "google/gemma-4-31B-it-qat-q4_0-unquantized";
        let tagged = format!("{m}:wren");
        let (rest, layout) = level(&tagged, false).unwrap();
        assert_eq!((rest, layout), (m, Some(Layout::Int4)));
        assert_eq!(layout_for(m, Layout::Int4), Layout::Int4H);
        assert_eq!(layout_for("google/gemma-4-31B-it", Layout::Int4), Layout::Int4H);
        assert_eq!(layout_for("Qwen/Qwen3.8-27B", Layout::Int4), Layout::Int4);
        assert_eq!(layout_for(m, Layout::Int6), Layout::Int6);
        assert_eq!(packed_repo(m, level_name(Layout::Int4H)), "glyd/gemma-4-31B-wren");
        assert_eq!(packed_repo("google/gemma-4-12B-it", "penguin"), "glyd/gemma-4-12B-penguin");
        assert_eq!(packed_repo("glyd/gemma-4-12B-wren", "wren"), "glyd/gemma-4-12B-wren");
        assert!(level("google/gemma-4-12B-it:wren", false).is_ok() && is_qat("google/gemma-4-12B-it") && !is_qat("google/gemma-4-12B"));
        assert_eq!(packed_repo("google/gemma-4-12B", "penguin"), "glyd/gemma-4-12B-base-penguin");
        assert_eq!(v7_level(m, Some(Layout::Int4H), Some(Layout::Int4)).unwrap(), Some(Layout::Int4H));
        assert!(v7_level(m, Some(Layout::Int4H), Some(Layout::Int6)).is_err());
    }
    use crate::ui::CANNOT;

    /// The plain dense kind's sizing case (a pre-norm dense decoder of 36 layers of 4,096, 32 query heads over 8 of 128: no release's by name), and a config of a
    /// kind this engine has no description of. `architectures` and `model_type` are what the description builder reads: its input, not a subject.
    const OTHER: &str =
        r#"{"architectures": ["OtherForCausalLM"], "model_type": "other", "hidden_size": 4096, "num_hidden_layers": 32, "num_attention_heads": 32}"#;
    const DENSE: &str = r#"{"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3", "hidden_size": 4096, "num_hidden_layers": 36, "num_attention_heads": 32, "num_key_value_heads": 8, "head_dim": 128, "intermediate_size": 12288,
        "vocab_size": 151936, "rms_norm_eps": 1e-06, "rope_theta": 1000000, "max_position_embeddings": 40960, "tie_word_embeddings": false, "rope_scaling": null, "sliding_window": null, "use_sliding_window": false}"#;

    fn dir_with(name: &str, config: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("glyd-engine-cli-{name}-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        std::fs::write(d.join("config.json"), config).unwrap();
        d
    }

    fn gpu(name: &str, cc: (i32, i32), free_gib: u64) -> Gpu {
        Gpu { name: name.to_string(), cc, sms: 58, total: (free_gib + 1) << 30, free: free_gib << 30, driver: 13000, count: 1, ordinal: 0 }
    }

    fn args(v: &[&str]) -> Args {
        Args::from(v.iter().map(|s| s.to_string()))
    }

    /// The checks that keep another web page's script, and a name that DNS points at this computer, away from the server are on for every address of this computer's own, the one the program chooses (no `--host`) and a
    /// loopback one the user names alike; the server listens on 127.0.0.1 unless told another address, and is not told another without a key (or `--insecure-no-auth`).
    #[test]
    fn the_host_checks_are_on_for_this_computers_own_address_and_a_server_opens_to_the_network_only_with_a_key() {
        let status = |addr: std::net::SocketAddr, host: &str| -> u16 {
            let mut c = std::net::TcpStream::connect(("127.0.0.1", addr.port())).unwrap(); // (a server on every address is reached by this computer's own)
            c.write_all(format!("GET /health HTTP/1.1\r\nHost: {host}\r\nConnection: close\r\n\r\n").as_bytes()).unwrap();
            let mut out = String::new();
            c.read_to_string(&mut out).unwrap();
            out.split_whitespace().nth(1).unwrap().parse().unwrap()
        };
        let (tx, _rx) = mpsc::channel();
        let own = listen_after(&args(&["serve", "m", "--port", "0"]), "m", 4096, tx.clone(), None).map_err(|r| r.to_string()).unwrap();
        assert!(own.ip().is_loopback(), "the default address is {own}");
        assert_eq!((status(own, "evil.example"), status(own, "localhost:8000"), status(own, "127.0.0.1")), (421, 200, 200));
        let chosen =
            listen_after(&args(&["serve", "m", "--host", "127.0.0.1", "--port", "0"]), "m", 4096, tx.clone(), None).map_err(|r| r.to_string()).unwrap();
        assert_eq!((status(chosen, "evil.example"), status(chosen, "localhost:8000")), (421, 200));
        // an address that is not this computer's own: refused with no key, answered with one or with the user's word (and then any Host is, there being no list of names)
        for host in ["0.0.0.0", "::", "192.168.1.5"] {
            let refused = listen_after(&args(&["serve", "m", "--host", host, "--port", "0"]), "m", 4096, tx.clone(), None).map(|_| ()).unwrap_err();
            assert!(refused.to_string().contains("--insecure-no-auth") && refused.to_string().contains("GLYD_API_KEY"), "{host}: {refused}");
        }
        let keyed = listen_after(&args(&["serve", "m", "--host", "0.0.0.0", "--port", "0", "--api-key", "k"]), "m", 4096, tx.clone(), None)
            .map_err(|r| r.to_string())
            .unwrap();
        assert_eq!((keyed.ip().is_unspecified(), status(keyed, "192.168.1.5:8000")), (true, 200));
        let said = listen_after(&args(&["serve", "m", "--host", "0.0.0.0", "--port", "0", "--insecure-no-auth"]), "m", 4096, tx.clone(), None)
            .map_err(|r| r.to_string())
            .unwrap();
        assert_eq!(status(said, "glyd.example.com"), 200);
        // the lists of what the user lets in are checked before the server starts
        for bad in ["*", "localhost:5173", "http://a/b"] {
            let e = listen_after(&args(&["serve", "m", "--port", "0", "--allowed-origins", bad]), "m", 4096, tx.clone(), None).map(|_| ()).unwrap_err();
            assert!(e.to_string().contains("--allowed-origins"), "{bad}: {e}");
        }
        // the refusal comes before the model is loaded, from `serve` and `run` themselves
        let a = args(&["serve", "m", "--host", "0.0.0.0"]);
        assert!(access_check(&a).is_err());
        assert_ne!(serve(&a), 0, "an open server with no key started");
    }

    /// `--ollama-port N` makes the server listen on N too, where N is free, with the checks the server's own address has; a port in use is said and does not stop the server.
    #[test]
    fn ollamas_port_is_taken_only_where_it_is_free() {
        let free = || std::net::TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port();
        let version = |port: u16, host: &str| -> String {
            let mut c = std::net::TcpStream::connect(("127.0.0.1", port)).unwrap();
            c.write_all(format!("GET /api/version HTTP/1.1\r\nHost: {host}\r\nConnection: close\r\n\r\n").as_bytes()).unwrap();
            let mut out = String::new();
            c.read_to_string(&mut out).unwrap();
            out.lines().next().unwrap_or("").to_string()
        };
        let (tx, _rx) = mpsc::channel();
        let p = free();
        let own = listen_after(&args(&["serve", "m", "--port", "0", "--ollama-port", &p.to_string()]), "m", 4096, tx.clone(), None)
            .map_err(|r| r.to_string())
            .unwrap();
        assert_ne!(own.port(), p);
        // (the engine's thread is not running here: a connection that is answered by the listener alone, the Host check, shows the port is the server's)
        assert!(version(p, "evil.example").contains("421"), "the Host check is the server's own");
        // a port in use: the server starts all the same
        let busy = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let q = busy.local_addr().unwrap().port();
        assert!(listen_after(&args(&["serve", "m", "--port", "0", "--ollama-port", &q.to_string()]), "m", 4096, tx.clone(), None).is_ok());
        // not a port: said, and the server starts
        assert!(listen_after(&args(&["serve", "m", "--port", "0", "--ollama-port", "x"]), "m", 4096, tx, None).is_ok());
    }

    #[test]
    fn the_api_key_from_the_flag_or_the_environment() {
        let env = |pairs: &'static [(&'static str, &'static str)]| move |k: &str| pairs.iter().find(|(n, _)| *n == k).map(|(_, v)| v.to_string());
        // the flag first, then GLYD_API_KEY, then VLLM_API_KEY; an empty one is none
        assert_eq!(key_of(Some("flag"), &env(&[("GLYD_API_KEY", "g"), ("VLLM_API_KEY", "v")])), Some("flag".to_string()));
        assert_eq!(key_of(None, &env(&[("GLYD_API_KEY", "g"), ("VLLM_API_KEY", "v")])), Some("g".to_string()));
        assert_eq!(key_of(None, &env(&[("VLLM_API_KEY", "v")])), Some("v".to_string()));
        assert_eq!(key_of(None, &env(&[])), None);
        assert_eq!(key_of(Some(""), &env(&[])), None);
        assert_eq!(key_of(None, &env(&[("GLYD_API_KEY", "")])), None);
    }

    /// The config decides, by its kind: a plain dense decoder runs whatever its architecture is called, and a kind with no description is refused with that, not
    /// with a list of families.
    #[test]
    fn what_the_engine_runs() {
        assert!(supported("dense-36l", DENSE).is_ok());
        assert!(supported("dense-36l", &DENSE.replace("Qwen3ForCausalLM", "AnotherNameForCausalLM")).is_ok(), "the architecture's name is not read");
        for other_type in ["mistral", "llama"] {
            assert!(supported("dense-36l", &DENSE.replace(r#""model_type": "qwen3""#, &format!(r#""model_type": "{other_type}""#))).is_ok(), "{other_type}");
        }
        let other = supported("other", OTHER).unwrap_err();
        assert_eq!(other.code, 3);
        assert_eq!(other.what, "other is a OtherForCausalLM model, which this engine has no description of yet");
        assert!(!other.what.contains("Qwen3") && !other.fix.contains("Qwen3 model") && !other.fix.contains("Qwen3-"), "{other:?}");
        assert_eq!(supported("x", "{}").unwrap_err().code, 3);
        assert_eq!(short_name("google/gemma-4-12B-it"), "gemma-4-12B-it");
        assert_eq!(short_name("/models/mine/"), "mine");
    }

    /// The dense route is taken by the config's kind alone: a directory whose name begins as an older release's did and whose config is another architecture has no
    /// level; a plain dense decoder under any name, of any of the builder's model types, is penguin; a name with no config to read says nothing.
    #[test]
    fn the_dense_route_is_the_configs_not_the_names() {
        let named = dir_with("qwen3-named-other", OTHER).join("qwen3-8b");
        std::fs::create_dir_all(&named).unwrap();
        std::fs::write(named.join("config.json"), OTHER).unwrap();
        let n = named.to_str().unwrap();
        assert!(short_name(n).starts_with("qwen3-") && !is_dense_model(n));
        assert_eq!(run_level(n, None).unwrap(), None);
        let ui = RefCell::new(Ui::silent());
        let r = check(&args(&["preflight", n]), &ui).unwrap_err();
        assert!(r.code == 3 && r.what.contains("OtherForCausalLM") && r.what.contains("no description"), "{r}");
        for model_type in ["qwen3", "mistral", "llama"] {
            let d = dir_with(&format!("any-name-{model_type}"), &DENSE.replace(r#""model_type": "qwen3""#, &format!(r#""model_type": "{model_type}""#)));
            let m = d.to_str().unwrap();
            assert!(is_dense_model(m), "{model_type}");
            assert_eq!(run_level(m, None).unwrap(), Some(Layout::Tiered), "{model_type}");
            assert!(table_check(m, None).is_ok() && table_check(m, Some(Layout::Int5)).is_ok(), "{model_type}");
            // (and its plan is the dense kind's, by the config: an A100's free memory holds it)
            let p = plan_level(&args(&["run", m]), m, Layout::Tiered, &config_text(m).unwrap(), None, &gpu("NVIDIA A100-SXM4-80GB", (8, 0), 70), None).unwrap();
            assert!(p.kind == Kind::Qwen3 && p.family == crate::dynamic::dense::FAMILY, "{model_type}: {}", p.family);
            std::fs::remove_dir_all(&d).unwrap();
        }
        assert!(!is_dense_model("/nonexistent/qwen3-8b"));
        assert_eq!(run_level("/nonexistent/qwen3-8b", None).unwrap(), None);
        std::fs::remove_dir_all(named.parent().unwrap()).unwrap();
    }

    /// `prepare` takes the path `run` and `serve` take, whatever the family: the level and the table are read first, from the config, and a refusal is a run's own
    /// (a kind with no description; a level the family does not run; a tag that is no level), said before a GPU is looked at. No family is told it is not the one
    /// the command serves.
    #[test]
    fn prepare_reads_the_level_and_the_table_as_a_run_does() {
        let ui = RefCell::new(Ui::silent());
        let other = dir_with("prepare-other", OTHER);
        let r = prepare(&args(&["prepare", other.to_str().unwrap()]), &ui).unwrap_err();
        assert!(r.code == CANNOT && r.what.contains("is a OtherForCausalLM model, which this engine has no description of yet"), "{r}");
        let g = fixture_dir("prepare-g4", "gemma4_12b_config.json");
        let gp = g.to_str().unwrap();
        let r = prepare(&args(&["prepare", &format!("{gp}:hummingbird")]), &ui).unwrap_err();
        assert!(r.code == CANNOT && r.what.ends_with("(Gemma 4) runs at the penguin, kestrel, swift and wren levels in this engine, not hummingbird"), "{r}");
        let r = prepare(&args(&["prepare", &format!("{gp}:sparrow")]), &ui).unwrap_err();
        assert!(r.code == 1 && r.what == ":sparrow is not a level", "{r}");
        let dense = dir_with("prepare-dense", DENSE);
        let r = prepare(&args(&["prepare", &format!("{}:hummingbird", dense.to_str().unwrap())]), &ui).unwrap_err();
        assert!(r.code == CANNOT && r.what.contains("(plain dense) runs at the penguin, kestrel, swift and wren levels"), "{r}");
        assert_eq!(prepare(&args(&["prepare"]), &ui).unwrap_err().code, 1);
        for d in [other, g, dense] {
            std::fs::remove_dir_all(d).unwrap();
        }
    }

    /// A plain dense decoder that sets what the dense kind's lowering does not take is refused before anything is downloaded, with the setting named: biases in the
    /// attention, a rope with scaling, another activation. A sliding window is the attention kind's parameter and runs; what a config sets off, or null, is nothing.
    #[test]
    fn a_dense_config_the_lowering_refuses_is_refused_with_what_it_sets() {
        let with = |extra: &str| DENSE.replacen('{', &format!("{{{extra}, "), 1);
        for fine in [r#""attention_bias": false, "hidden_act": "silu""#, r#""layer_types": null"#] {
            assert!(supported("m", &with(fine)).is_ok(), "{fine}");
        }
        for (setting, said) in [
            (r#""attention_bias": true"#, "m has biases in the attention, which this engine does not run yet"),
            (r#""hidden_act": "gelu""#, "m has an activation other than silu, which this engine does not run yet"),
        ] {
            let r = supported("m", &with(setting)).unwrap_err();
            assert!(r.code == CANNOT && r.what == said, "{setting}: {r}");
        }
        let scaled = DENSE.replace(r#""rope_scaling": null"#, r#""rope_scaling": {"rope_type": "linear", "factor": 2.0}"#);
        let r = supported("m", &scaled).unwrap_err();
        assert!(r.code == CANNOT && r.what == "m has rope scaling, which this engine does not run yet", "{r}");
        // (a window is a parameter of the attention kind: the lowering takes it)
        let windowed = DENSE
            .replace(r#""model_type": "qwen3""#, r#""model_type": "mistral""#)
            .replace(r#""sliding_window": null, "use_sliding_window": false"#, r#""sliding_window": 4096"#);
        assert!(supported("m", &windowed).is_ok());
    }

    fn fixture_dir(name: &str, file: &str) -> PathBuf {
        dir_with(name, &std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/").to_string() + file).unwrap())
    }

    /// `supports` takes MODEL:LEVEL and every family the table has: a recent family at a lossy level runs, and a refusal says the levels the family does run.
    #[test]
    fn supports_takes_a_level_and_every_family_of_the_table() {
        let g = fixture_dir("supports-g4", "gemma4_12b_config.json");
        let gp = g.to_str().unwrap();
        // Gemma 4 12B: no level, a lossy one, --lossy, every level of its row
        assert!(supports_check(gp, false).is_ok());
        for l in ["penguin", "kestrel", "swift", "wren"] {
            assert!(supports_check(&format!("{gp}:{l}"), false).is_ok(), "{l}");
        }
        assert!(supports_check(gp, true).is_ok());
        let e = supports_check(&format!("{gp}:hummingbird"), false).unwrap_err();
        assert!(
            e.code == 3
                && e.what.ends_with("(Gemma 4) runs at the penguin, kestrel, swift and wren levels in this engine, not hummingbird")
                && e.fix.starts_with("Run it lossless: glyd run "),
            "{e:?}"
        );
        // a tag that is no level is an error of the command line, not "not supported"
        let e = supports_check(&format!("{gp}:sparrow"), false).unwrap_err();
        assert!(e.code == 1 && e.what.contains(":sparrow is not a level"), "{e:?}");
        let e = supports_check(&format!("{gp}:swift"), true).unwrap_err();
        assert!(e.what.contains("--lossy is :kestrel"), "{e:?}");
        // Qwen3.5 to 3.8 runs lossless and at the lossy levels; not at hummingbird, which is for experts
        let q35: Value = json::parse(&std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/qwen35_configs.json")).unwrap()).unwrap();
        let first = match &q35 {
            Value::Object(m) => m[0].1.clone(),
            Value::Array(a) => a[0].clone(),
            other => other.clone(),
        };
        let q = dir_with("supports-q35", &json::to_compact(&first));
        let qp = q.to_str().unwrap();
        assert!(supports_check(qp, false).is_ok());
        assert!(supports_check(&format!("{qp}:swift"), false).is_ok());
        assert!(supports_check(&format!("{qp}:penguin"), false).is_ok());
        let e = supports_check(&format!("{qp}:hummingbird"), false).unwrap_err();
        assert!(
            e.code == 3
                && e.what.ends_with("(Qwen3.5) runs at the penguin, kestrel, swift and wren levels in this engine, not hummingbird")
                && e.fix == format!("Run it lossless: glyd run {qp}"),
            "{e:?}"
        );
        // a plain dense decoder runs at penguin and the block levels, and keeps the dense kind's own refusals; a kind with no description keeps its words at any level
        let dense = dir_with("supports-dense", DENSE);
        let dp = dense.to_str().unwrap();
        assert!(
            supports_check(dp, false).is_ok()
                && supports_check(&format!("{dp}:penguin"), false).is_ok()
                && supports_check(&format!("{dp}:swift"), false).is_ok()
        );
        let e = supports_check(&format!("{dp}:hummingbird"), false).unwrap_err();
        assert!(e.code == 3 && e.what.ends_with("(plain dense) runs at the penguin, kestrel, swift and wren levels in this engine, not hummingbird"), "{e:?}");
        let biased = dir_with("supports-biased", &DENSE.replacen('{', r#"{"attention_bias": true, "#, 1));
        assert!(supports_check(biased.to_str().unwrap(), false).unwrap_err().what.contains("biases in the attention"));
        let l = dir_with("supports-llama", OTHER);
        for tag in ["", ":kestrel"] {
            let e = supports_check(&format!("{}{tag}", l.to_str().unwrap()), false).unwrap_err();
            assert!(e.code == 3 && e.what.contains("is a OtherForCausalLM model, which this engine has no description of yet"), "{e:?}");
        }
        // a model that cannot be read is "not told", whatever the level
        assert_eq!(supports_check("/nonexistent/model:kestrel", false).unwrap_err().code, 1);
        for d in [g, q, dense, biased, l] {
            std::fs::remove_dir_all(d).unwrap();
        }
    }

    /// The dynamic route takes the Qwen3.5 layout at penguin; its lossy levels stay on the one-GPU lossy path; Kimi-K3's hummingbird stays on the dynamic route.
    #[test]
    fn the_route_of_each_level() {
        let q35: Value = json::parse(&std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/qwen35_configs.json")).unwrap()).unwrap();
        let q38 = json::to_compact(q35.get("Qwen/Qwen3.8-27B").unwrap());
        assert_eq!(crate::dynamic::split::family(&q38), Some("Qwen3.5"));
        for (family, layout, lossy) in [
            ("Qwen3.5", Layout::Tiered, false),
            ("Qwen3.5", Layout::Int6, true),
            ("Qwen3.5", Layout::Int4, true),
            ("Kimi-K3", Layout::Hummingbird, false),
            ("GLM-5.3-Flash", Layout::Tiered, false),
        ] {
            assert_eq!(lossy_route(family, layout), lossy, "{family} {layout:?}");
        }
    }

    #[test]
    fn supports_is_the_kind_of_model_alone() {
        let q = dir_with("supports-q", DENSE);
        let l = dir_with("supports-l", OTHER);
        assert_eq!(supports(&args(&["supports", q.to_str().unwrap()])), 0);
        assert_eq!(supports(&args(&["supports", l.to_str().unwrap()])), 3);
        assert_eq!(supports(&args(&["supports", "/nonexistent/model"])), 1); // (cannot be told: the engine's own refusal comes with the run)
        assert_eq!(supports(&args(&["supports"])), 1);
        // the preflight says the same of a model of another kind before it looks at a GPU
        let ui = RefCell::new(Ui::silent());
        let r = check(&args(&["preflight", l.to_str().unwrap()]), &ui).unwrap_err();
        assert!(r.code == 3 && r.what.contains("OtherForCausalLM"), "{r}");
        assert_eq!(config_text(q.to_str().unwrap()).unwrap(), DENSE);
        std::fs::remove_dir_all(&q).unwrap();
        std::fs::remove_dir_all(&l).unwrap();
    }

    /// A level is the tag after the model's last ':' when that is all ASCII letters, in any case, and it comes off; --lossy is kestrel; a name that is no level, and --lossy with another level, are
    /// refused (status 1); a model with no tag (a Hub name, a path whose text after a ':' has a '/' or more) is as it is.
    /// The Hub repo of a model at a level is `glyd/NAME-LEVEL` (glyd-gpu-py's test_engine_packed.py pins the same examples), and a pre-packed repo's own name is its level.
    #[test]
    fn a_pre_packed_repo_is_named_by_its_model_and_level() {
        for (model, level, repo) in [
            ("Qwen/Qwen3.5-9B", "swift", "glyd/Qwen3.5-9B-swift"),
            ("Qwen/Qwen3.8-27B", "kestrel", "glyd/Qwen3.8-27B-kestrel"),
            ("Qwen/Qwen3.5-9B", "penguin", "glyd/Qwen3.5-9B-penguin"),
            ("Qwen/Qwen3.5-9B", "wren", "glyd/Qwen3.5-9B-wren"),
            ("nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16", "hummingbird", "glyd/nemotron-3-hummingbird"),
        ] {
            assert_eq!(packed_repo(model, level), repo);
        }
        assert_eq!(level("glyd/Qwen3.5-9B-swift", false).unwrap(), ("glyd/Qwen3.5-9B-swift", Some(Layout::Int5)));
        assert_eq!(level("glyd/Qwen3.8-27B-kestrel", false).unwrap().1, Some(Layout::Int6));
        assert_eq!(level("glyd/Qwen3.5-9B-wren", false).unwrap(), ("glyd/Qwen3.5-9B-wren", Some(Layout::Int4)));
        for plain in ["glyd/Qwen3.5-9B", "glyd/Qwen3.5-9B-penguin", "Qwen/Qwen3.5-9B-swift", "glyd/a/b-swift"] {
            assert_eq!(level(plain, false).unwrap().1, None, "{plain}");
        }
    }

    #[test]
    fn a_level_is_the_tag_after_the_models_last_colon() {
        let m = "Qwen/Qwen3.8-27B";
        // the tags, in any case: the model without them, and the layout the level packs in (none: lossless)
        for (tag, layout) in [
            ("penguin", None),
            ("kestrel", Some(Layout::Int6)),
            ("swift", Some(Layout::Int5)),
            ("wren", Some(Layout::Int4)),
            ("hummingbird", Some(Layout::Hummingbird)),
            ("Penguin", None),
            ("KESTREL", Some(Layout::Int6)),
            ("sWiFt", Some(Layout::Int5)),
            ("Wren", Some(Layout::Int4)),
            ("HUMMINGBIRD", Some(Layout::Hummingbird)),
        ] {
            assert_eq!(level(&format!("{m}:{tag}"), false).unwrap(), (m, layout), "{tag}");
        }
        assert_eq!(level("/models/mine:swift", false).unwrap(), ("/models/mine", Some(Layout::Int5)));
        assert_eq!(level("a:b:kestrel", false).unwrap(), ("a:b", Some(Layout::Int6))); // (the last ':' only)
                                                                                       // no tag: as it was; --lossy is kestrel, with its tag or without
        assert_eq!(level(m, false).unwrap(), (m, None));
        assert_eq!(level(m, true).unwrap(), (m, Some(Layout::Int6)));
        assert_eq!(level(&format!("{m}:Kestrel"), true).unwrap(), (m, Some(Layout::Int6)));
        // a name that is no level: refused with the levels there are (sparrow, once proposed, is no level)
        for (tag, said) in [("sparrow", "not a level"), ("falcon", "not a level"), ("x", "not a level"), ("mma", "not a level")] {
            let r = level(&format!("{m}:{tag}"), false).unwrap_err();
            assert!(
                r.code == 1
                    && r.what == format!(":{} is {said}", tag.to_lowercase())
                    && r.fix == format!("The levels available now: {}", levels::names_now().join(", ")),
                "{tag}: {r}"
            );
        }
        // what is not all letters after the last ':' is no tag: the model is left alone, --lossy or not
        for plain in [
            "org/dense-36l",
            "/models/mine/",
            "mine",
            "/mnt/disk:1/model",
            "./a:b/c",
            "/data/x:kestrel/model",
            "C:\\models\\Qwen",
            "org/dense-36l:",
            "m:v2",
            "m:4bit",
            "m:ké",
            "m: swift",
            ":",
            "",
        ] {
            assert_eq!(level(plain, false).unwrap(), (plain, None), "{plain}");
            assert_eq!(level(plain, true).unwrap(), (plain, Some(Layout::Int6)), "{plain}");
        }
        // --lossy is :kestrel: with another level it is refused; a name that is no level is refused as it is
        for tag in ["penguin", "swift", "SWIFT", "wren", "hummingbird"] {
            let r = level(&format!("{m}:{tag}"), true).unwrap_err();
            assert!(r.code == 1 && r.what == format!("--lossy is :kestrel, not :{}", tag.to_lowercase()), "{tag}: {r}");
        }
        assert_eq!(level(&format!("{m}:sparrow"), true).unwrap_err().what, ":sparrow is not a level");
        assert_eq!(level(&format!("{m}:hello"), true).unwrap_err().what, ":hello is not a level");
        // the names the messages use, and the bits they say
        assert_eq!(
            (level_name(Layout::Int6), level_name(Layout::Int5), level_name(Layout::Int4), level_name(Layout::Int3), level_name(Layout::Hummingbird)),
            ("kestrel", "swift", "wren", "i3", "hummingbird")
        );
        assert_eq!(
            (format!("{:.1}", bits(Layout::Int6)), format!("{:.1}", bits(Layout::Int5)), format!("{:.1}", bits(Layout::Int4))),
            ("6.5".to_string(), "5.5".to_string(), "4.5".to_string())
        );
    }

    /// `run` and `preflight` read the level before they look at the model, the GPU or standard input; the model they look up is the one without its tag. `serve` reads it the same way.
    #[test]
    fn run_preflight_and_serve_read_the_level_first() {
        let ui = RefCell::new(Ui::silent());
        let r = check(&args(&["preflight", "Qwen/Qwen3.8-27B:sparrow"]), &ui).unwrap_err();
        assert!(r.code == 1 && r.what == ":sparrow is not a level", "{r}");
        let r = check(&args(&["preflight", "Qwen/Qwen3.8-27B:swift", "--lossy"]), &ui).unwrap_err();
        assert!(r.code == 1 && r.what.starts_with("--lossy is :kestrel"), "{r}");
        assert_eq!(run(&args(&["run", "Qwen/Qwen3.8-27B:sparrow", "--prompt", "hi"])), 1);
        assert_eq!(run(&args(&["run", "Qwen/Qwen3.8-27B:penguin", "--lossy", "--prompt", "hi"])), 1);
        // (serve: the tags run refuses are refused, before the GPU is looked at)
        for tag in ["sparrow", "falcon"] {
            assert_eq!(serve(&args(&["serve", &format!("Qwen/Qwen3.8-27B:{tag}")])), 1, "{tag}");
        }
        // penguin's tag is off the model that is looked up: the directory it names is the one read (a kind with no description: status 3), whatever the tag's case
        let l = dir_with("level-l", OTHER);
        let m = l.to_str().unwrap();
        for tagged in [m.to_string(), format!("{m}:penguin"), format!("{m}:PENGUIN")] {
            let r = check(&args(&["preflight", &tagged]), &ui).unwrap_err();
            assert!(r.code == 3 && r.what.contains("OtherForCausalLM"), "{tagged}: {r}");
        }
        std::fs::remove_dir_all(&l).unwrap();
    }

    /// A glyd-v7 directory runs at its own level where none is asked, and is refused, naming both, at another.
    #[test]
    fn a_packed_directory_runs_at_its_own_level() {
        let m = "/models/q-swift";
        assert_eq!(v7_level(m, Some(Layout::Int5), None).unwrap(), Some(Layout::Int5));
        assert_eq!(v7_level(m, Some(Layout::Int5), Some(Layout::Int5)).unwrap(), Some(Layout::Int5));
        let r = v7_level(m, Some(Layout::Int5), Some(Layout::Int6)).unwrap_err();
        assert!(r.code == 1 && r.what == "/models/q-swift is packed at the swift level, not :kestrel", "{r}");
        assert_eq!(v7_level(m, None, Some(Layout::Int6)).unwrap(), Some(Layout::Int6));
        assert_eq!(v7_level(m, None, None).unwrap(), None);
        let r = v7_level("/models/q-wren", Some(Layout::Int4), Some(Layout::Int5)).unwrap_err();
        assert!(r.code == 1 && r.what == "/models/q-wren is packed at the wren level, not :swift", "{r}");
    }

    /// A lossy level's plan sizes the weights in its layout (plan.rs `sizing::bits`), and its refusals name the level.
    #[test]
    fn a_lossy_plan_is_sized_in_its_levels_layout() {
        let text = include_str!("../../research/lossy-2026/qwen38-config.json");
        let (a, model) = (args(&["run", "Qwen/Qwen3.8-27B"]), "Qwen/Qwen3.8-27B");
        let plan = |layout, free_gib: u64| lossy_plan(&a, model, layout, text, None, free_gib << 30, "NVIDIA A100", None);
        let (six, five, four) = (plan(Layout::Int6, 80).unwrap(), plan(Layout::Int5, 80).unwrap(), plan(Layout::Int4, 80).unwrap());
        assert_eq!(six.bf16, five.bf16);
        assert_eq!(five.bf16, four.bf16);
        // (what is packed is bits/16 of its bf16 bytes, the rest as it is: the ratio is a little above the layouts')
        let ratio = five.weights as f64 / six.weights as f64;
        assert!(ratio > bits(Layout::Int5) / bits(Layout::Int6) && ratio < 0.9, "{ratio}");
        let ratio = four.weights as f64 / five.weights as f64;
        assert!(ratio > bits(Layout::Int4) / bits(Layout::Int5) && ratio < 0.9, "{ratio}");
        for (layout, level) in [(Layout::Int6, "kestrel"), (Layout::Int5, "swift"), (Layout::Int4, "wren")] {
            let r = plan(layout, 8).err().unwrap();
            assert!(r.code == CANNOT && r.what.starts_with(&format!("Qwen3.8-27B at the {level} level takes about ")), "{r}");
            let r = lossy_plan(&a, "org/dense-36l", layout, DENSE, None, 80 << 30, "NVIDIA A100", None).err().unwrap();
            assert!(
                r.code == CANNOT
                    && r.what.starts_with(&format!("dense-36l: the {level} level on this path takes the Qwen3.5 and Nemotron-H layouts"))
                    && r.fix == "Run it lossless: glyd run org/dense-36l",
                "{r}"
            );
        }
    }

    /// Nemotron 3 Super at the kestrel, swift and wren levels is sized by its packs in their layouts (120.58 billion matrix weights, 241 GB of bf16), with 8 KiB of KV a token; it is refused with the memory it needs and
    /// the GPU's where that is short (the 80 GB cards hold wren alone, a 96 GB card swift too). At wren the 112.7 billion weights of its experts are four-bit codes (4.53 bits a weight) and the other 7.84 billion six-bit ones
    /// (6.53), 70.4 GB with the small tensors: it fits the 79 GiB an A100 or H100 80 GB reports free, with room for the longest context the plan takes, and not 60 GiB.
    #[test]
    fn gemma4_plans_at_penguin_and_the_block_levels() {
        // (Google's gemma-4-12B-it config.json: penguin, a lossless save of it, is planned as the block levels are; i3 is not built for Gemma 4)
        let text = include_str!("../tests/fixtures/gemma4_12b_config.json");
        let model = "google/gemma-4-12B-it";
        let plan = |layout| lossy_plan_with(&args(&["run", model]), model, layout, text, None, 22 << 30, "NVIDIA RTX 4090", None, None);
        let (penguin, kestrel) = (plan(Layout::Tiered).unwrap(), plan(Layout::Int6).unwrap());
        assert!((penguin.kind == Kind::Gemma4) && (kestrel.kind == Kind::Gemma4) && penguin.bf16 == kestrel.bf16);
        let ratio = penguin.weights as f64 / penguin.bf16 as f64;
        assert!((0.66..0.69).contains(&ratio), "penguin at {ratio} of bf16");
        assert!(penguin.weights > kestrel.weights && penguin.context >= LOSSY_MIN_CTX);
        let r = plan(Layout::Int3).err().unwrap();
        assert!(r.code == CANNOT && r.what.contains("the i3 level is not built for Gemma 4"), "{r}");
        // (penguin short of room on a 16 GB card: the wren repo, which fits, is named first; at 6 GiB nothing fits and the fix is the plain one; the base model has no wren)
        let small = |m: &str, free_gib: u64| {
            lossy_plan_with(&args(&["run", m]), m, Layout::Tiered, text, None, free_gib << 30, "NVIDIA GeForce RTX 4080 SUPER", None, None).err().unwrap()
        };
        for m in [model, "google/gemma-4-12B-it:penguin", "glyd/gemma-4-12B-penguin"] {
            let r = small(m, 15);
            assert!(
                r.code == CANNOT && r.fix.starts_with("glyd run glyd/gemma-4-12B-wren fits: the wren level, about ") && r.fix.contains("\nOr close programs"),
                "{m}: {r}"
            );
        }
        assert!(small(model, 6).fix.starts_with("Close programs") && small("google/gemma-4-12B", 15).fix.starts_with("Close programs"));
        assert_eq!(
            (gemma_wren("google/gemma-4-31B-it").as_deref(), gemma_wren("glyd/gemma-4-31B-penguin").as_deref()),
            (Some("glyd/gemma-4-31B-wren"), Some("glyd/gemma-4-31B-wren"))
        );
        assert_eq!((gemma_wren("glyd/gemma-4-12B-base-penguin"), gemma_wren("Qwen/Qwen3.8-27B")), (None, None));
        // (served to --users at once with no draft model: each user's room and context fit beside the others'; run, and a context asked for, keep one user's)
        let free = 14u64 << 30;
        let at = |argv: &[&str]| lossy_plan_with(&args(argv), model, Layout::Int4, text, None, free, "NVIDIA GeForce RTX 4080 SUPER", None, None).unwrap();
        let cfg = crate::gemma4::ConfigG4::parse(text, None, false).unwrap();
        let (one, eight) = (at(&["run", model, "--no-draft"]), at(&["serve", model, "--users", "8", "--no-draft"]));
        assert_eq!((one.context, one.users), (LOSSY_CTX, 1));
        assert!(eight.context >= LOSSY_MIN_CTX && eight.context < one.context && eight.users == 8, "{}", eight.context);
        let (rings, per) = cfg.kv_bytes(GEMMA_CHUNK, eight.context, false);
        assert_eq!(eight.kv, 8 * (rings + eight.context as u64 * per));
        assert!(eight.weights + LOSSY_WORK + eight.reserve + eight.kv <= free, "8 users at {} tokens", eight.context);
        assert_eq!(at(&["serve", model, "--users", "1", "--no-draft"]).context, LOSSY_CTX);
        assert_eq!(at(&["serve", model, "--users", "8", "--no-draft", "--ctx", "8192"]).context, 8192);
        // (a draft model to come with the download: drafted, one user at a time, the context of one)
        assert_eq!(at(&["serve", model, "--users", "8"]).users, 1);
        // (run, serve and preflight alike: Gemma 4 without a tag is penguin, with one the level asked; so is a plain dense decoder, on the executor too)
        assert_eq!(run_level(model, None).unwrap(), Some(Layout::Tiered));
        assert_eq!(run_level(model, Some(Layout::Int4)).unwrap(), Some(Layout::Int4));
        let dense = dir_with("run-level-dense", DENSE);
        assert_eq!(run_level(dense.to_str().unwrap(), None).unwrap(), Some(Layout::Tiered));
    }

    /// A run whose context does not fit at the level asked loads at the shorter one the ladder takes (on a 16 GB card, gemma-4-12B at kestrel with 262144 tokens loads
    /// at 131072): the gate's arguments carry it, where the load refused at the context asked for after the ladder said it would not.
    #[test]
    fn the_gate_hands_the_ladders_shorter_context_to_the_load() {
        let dir = std::env::temp_dir().join(format!("glyd-gate-ctx-{}", std::process::id())).join("gemma-4-12B-it");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("config.json"), include_str!("../tests/fixtures/gemma4_12b_config.json")).unwrap();
        let m = dir.to_str().unwrap();
        let tagged = format!("{m}:kestrel");
        let ui = RefCell::new(Ui::new());
        let a = Args::from(["run", tagged.as_str(), "--context", "262144", "--machine", "rtx4080s"].iter().map(|s| s.to_string()));
        let (b, layout) = degrade_gate(&a, m, Some(Layout::Int6), false, &ui).unwrap();
        assert_eq!((b.flag("context"), layout), (Some("131072"), Some(Layout::Int6)));
        // (one that fits as asked keeps its arguments)
        let a = Args::from(["run", tagged.as_str(), "--context", "32768", "--machine", "rtx4080s"].iter().map(|s| s.to_string()));
        assert_eq!(degrade_gate(&a, m, Some(Layout::Int6), false, &ui).unwrap().0.flag("context"), Some("32768"));
        let _ = std::fs::remove_dir_all(dir.parent().unwrap());
    }

    /// Run, serve and preflight without a tag load the level `fit` plans (P16, the M1 gate on an 8 GB card): Gemma 4 12B on a 16 GB card is kestrel, which the card holds,
    /// not penguin, which it does not; on a 24 GB card penguin. A tag is its own level.
    #[test]
    fn an_untagged_model_loads_the_level_fit_plans() {
        let dir = std::env::temp_dir().join(format!("glyd-load-level-{}", std::process::id())).join("gemma-4-12B-it");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("config.json"), include_str!("../tests/fixtures/gemma4_12b_config.json")).unwrap();
        let m = dir.to_str().unwrap();
        let on = |machine: &str| Args::from(["serve", m, "--machine", machine].iter().map(|s| s.to_string()));
        assert_eq!(planner_level(&on("rtx4080s"), m), Some(Layout::Int6));
        assert_eq!(load_level(&on("rtx4080s"), m, None).unwrap(), Some(Layout::Int6));
        assert_eq!(load_level(&on("rtx4090"), m, None).unwrap(), Some(Layout::Tiered));
        assert_eq!(load_level(&on("rtx4080s"), m, Some(Layout::Int4)).unwrap(), Some(Layout::Int4));
        // `:penguin` is asked for, not planned (its layout is the untagged one's): the 16 GB card's run goes to the ladder, which asks before kestrel
        let tagged = format!("{m}:penguin");
        let on_tag = Args::from(["run", tagged.as_str(), "--machine", "rtx4080s"].iter().map(|s| s.to_string()));
        assert_eq!(load_level(&on_tag, m, None).unwrap(), Some(Layout::Tiered));
        let _ = std::fs::remove_dir_all(dir.parent().unwrap());
        // Qwen3.8-27B untagged loads the planner's level: a lossy one on a 24 GB card (its penguin does not fit one), which the table takes, as it takes penguin
        let q35: Value = json::parse(&std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/qwen35_configs.json")).unwrap()).unwrap();
        let q38 = match &q35 {
            Value::Object(m) => m.iter().find(|(k, _)| k.as_str() == "Qwen/Qwen3.8-27B").map(|(_, v)| v.clone()).unwrap(),
            _ => panic!("qwen35_configs.json is an object of configs"),
        };
        let q = dir_with("load-level-q38", &json::to_compact(&q38));
        let qp = q.to_str().unwrap();
        let onq = |machine: &str| Args::from(["serve", qp, "--machine", machine].iter().map(|s| s.to_string()));
        let l = load_level(&onq("rtx4090"), qp, None).unwrap();
        assert!(matches!(l, Some(Layout::Int6 | Layout::Int5 | Layout::Int4)), "{l:?}");
        assert!(table_check(qp, l).is_ok() && table_check(qp, None).is_ok());
        assert_eq!(run_level(qp, None).unwrap(), Some(Layout::Tiered), "penguin where nothing is asked, as the dynamic route's families");
        assert_eq!(l, planner_level(&onq("rtx4090"), qp));
        // on a 16 GB card nothing fits: fit's refusal (run's own sizing, in its words), before anything is downloaded
        let e = load_level(&onq("rtx4080s"), qp, None).unwrap_err();
        assert!(e.what.contains("does not fit") || e.what.contains("GiB of GPU memory"), "{e:?}");
        // a machine the planner refuses for itself (this preset's driver is CUDA 12.7) is left to the run's own checks, which say it in their words
        if crate::support::machine(&crate::plan::Machine::preset("rtx4060-laptop").unwrap()).is_some() {
            assert!(load_level(&onq("rtx4060-laptop"), qp, None).is_ok());
        }
        // a plain dense decoder the planner keeps lossless loads at penguin, on the executor
        let dense = dir_with("load-level-dense", DENSE);
        let dp = dense.to_str().unwrap();
        assert_eq!(load_level(&Args::from(["serve", dp, "--machine", "rtx4090"].iter().map(|s| s.to_string())), dp, None).unwrap(), Some(Layout::Tiered));
    }

    #[test]
    fn a_lossy_plan_of_nemotron_is_sized_by_its_packs() {
        let text = include_str!("../tests/fixtures/nemotron_h_config.json");
        let (a, model) = (args(&["run", "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16"]), "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16");
        // (a computer of 8 GiB cannot take the experts in its memory: the GPU has to hold the model whole)
        let lossy_plan = |a: &Args, model: &str, layout: Layout, text: &str, gen: Option<&str>, free: u64, gpu: &str, packed: Option<&Path>| {
            lossy_plan_with(a, model, layout, text, gen, free, gpu, packed, Some(8 << 30))
        };
        let plan = |layout, free_gib: u64| lossy_plan(&a, model, layout, text, None, free_gib << 30, "NVIDIA H200", None);
        let (six, five, four) = (plan(Layout::Int6, 140).unwrap(), plan(Layout::Int5, 140).unwrap(), plan(Layout::Int4, 140).unwrap());
        assert!(
            (six.kind == Kind::NemotronH)
                && (five.kind == Kind::NemotronH)
                && (four.kind == Kind::NemotronH)
                && six.bf16 == five.bf16
                && five.bf16 == four.bf16
        );
        assert!((241_300_000_000..241_400_000_000).contains(&six.bf16), "{}", six.bf16);
        // (6.53 and 5.53 bits a weight of the 120.58 B, and the small tensors as they are)
        assert!((98_500_000_000..98_700_000_000).contains(&six.weights), "{}", six.weights);
        assert!((83_400_000_000..83_600_000_000).contains(&five.weights), "{}", five.weights);
        // (wren: the experts' 112,742,891,520 weights at 4.53 bits, the other 7,839,154,176 at 6.53, and the small tensors, 173 MB, as they are)
        let wren = (112_742_891_520f64 * 4.53 + 7_839_154_176f64 * 6.53) / 8.0 + 173_324_288.0;
        assert!((four.weights as f64 - wren).abs() < 2e6, "{} against {wren}", four.weights);
        assert!((70_300_000_000..70_500_000_000).contains(&four.weights), "{}", four.weights);
        assert_eq!((six.context, five.context, four.context), (LOSSY_CTX, LOSSY_CTX, LOSSY_CTX));
        for (layout, level) in [(Layout::Int6, "kestrel"), (Layout::Int5, "swift")] {
            let r = plan(layout, 70).err().unwrap();
            assert!(
                r.code == CANNOT
                    && r.what.starts_with(&format!("NVIDIA-Nemotron-3-Super-120B-A12B-BF16 at the {level} level takes about "))
                    && r.what.contains("NVIDIA H200 has 70.0 GiB free"),
                "{r}"
            );
        }
        assert!(plan(Layout::Int5, 78).is_err() && plan(Layout::Int5, 82).is_ok(), "swift takes about 80 GiB");
        assert!(plan(Layout::Int6, 92).is_err() && plan(Layout::Int6, 96).is_ok(), "kestrel takes about 94 GiB");
        // wren on the 80 GB cards (79 GiB free): the weights 65.6 GiB, and the context the plan allows at most; refused at 60 GiB, with the memory it needs and the card's
        let a100 = lossy_plan(&a, model, Layout::Int4, text, None, 79 << 30, "NVIDIA A100-SXM4-80GB", None).unwrap();
        assert!((a100.kind == Kind::NemotronH) && a100.context >= LOSSY_MIN_CTX.max(4096) && a100.context == LOSSY_CTX, "{}", a100.context);
        let r = lossy_plan(&a, model, Layout::Int4, text, None, 60 << 30, "NVIDIA A100-SXM4-80GB", None).err().unwrap();
        assert!(
            r.code == CANNOT
                && r.what.starts_with("NVIDIA-Nemotron-3-Super-120B-A12B-BF16 at the wren level takes about 6")
                && r.what.contains("NVIDIA A100-SXM4-80GB has 60.0 GiB free"),
            "{r}"
        );
        assert!(plan(Layout::Int4, 66).is_err() && plan(Layout::Int4, 68).is_ok(), "wren takes about 67 GiB");
        // (a context asked for is taken where it fits and refused where it does not: 4,096 tokens at 68 GiB, not the model's 262,144)
        let asked = |ctx: &str| lossy_plan(&args(&["run", model, "--context", ctx]), model, Layout::Int4, text, None, 68 << 30, "NVIDIA A100-SXM4-80GB", None);
        assert_eq!(asked("4096").unwrap().context, 4096);
        assert!(asked("262144").is_err());
        // the three-bit layout is no level of it
        let r = plan(Layout::Int3, 140).err().unwrap();
        assert!(r.code == CANNOT && r.what.contains("the i3 level is not built for the Nemotron-H layout") && r.fix.ends_with(&format!("{model}:wren")), "{r}");
        // a config the engine does not run (here one with a convolution without its bias) is said in the config's own words
        let r = plan_of(&a, model, text.replace("\"use_conv_bias\": true", "\"use_conv_bias\": false")).err().unwrap();
        assert!(r.code == CANNOT && r.what.contains("a convolution without its bias"), "{r}");
    }

    /// Nemotron 3 Super at swift on a GPU that cannot hold it whole runs with its experts in the computer's memory (77.5 GB of them pinned, and a margin) and a cache of them on the GPU: the cache is what the GPU has left
    /// besides the rest of the model (5.6 GiB), the states, the workspaces and the 32,768 tokens' keys and values: 56 experts a layer of 512 on a 16 GB card, 106 on a 24 GB card (23 GiB free), 163 on a 32 GB card
    /// (31 GiB), 276 on 47; the least is two steps' choices (44 a layer). The computer's memory decides too: it is refused, in words, with 70 GiB where the experts and the margin want 78.
    #[test]
    fn a_nemotron_that_does_not_fit_the_gpu_keeps_its_experts_in_the_computers_memory() {
        let text = include_str!("../tests/fixtures/nemotron_h_config.json");
        let (a, model) = (args(&["run", "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16"]), "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16");
        let plan = |layout, free_gib: f64, host_gib: u64| {
            lossy_plan_with(&a, model, layout, text, None, (free_gib * (1u64 << 30) as f64) as u64, "NVIDIA RTX 4090", None, Some(host_gib << 30))
        };
        for (free, slots) in [(16.0, 56), (23.0, 106), (31.0, 163), (47.0, 276)] {
            let p = plan(Layout::Int5, free, 128).unwrap();
            let c = p.experts.expect("a cache");
            assert_eq!((c.slots, c.layers, c.experts, c.expert_bytes, p.context), (slots, 40, 512, 3_784_704, LOSSY_CTX), "{free} GiB free");
            assert_eq!(c.host_bytes(), 40 * 512 * 3_784_704);
            assert!(c.gpu_bytes() < (free * (1u64 << 30) as f64) as u64 - (5 << 30), "the cache and the rest of the model are inside what is free");
        }
        // kestrel: 4,472,832 bytes an expert
        let c = plan(Layout::Int6, 23.0, 128).unwrap().experts.unwrap();
        assert_eq!((c.slots, c.expert_bytes), (84, 4_472_832));
        // a GPU that holds it whole needs no cache; a GPU too small for the rest of the model and the least cache is refused, with both memories
        assert!(plan(Layout::Int5, 140.0, 128).unwrap().experts.is_none());
        let r = plan(Layout::Int5, 12.0, 128).err().unwrap();
        assert!(r.code == CANNOT && r.what.contains("it needs at least 14.2 GiB of GPU memory (and 78.2 GiB of the computer's)"), "{r}");
        assert!(plan(Layout::Int6, 16.0, 128).is_err() && plan(Layout::Int6, 17.0, 128).is_ok());
        // the computer: 78 GiB of memory (72.2 of experts and 6 of margin) is the least
        let r = plan(Layout::Int5, 23.0, 64).err().unwrap();
        assert!(
            r.code == CANNOT
                && r.what.contains("it needs 78.2 GiB of the computer's memory besides at least 14.2 GiB of GPU memory, and this computer has 64.0 GiB free"),
            "{r}"
        );
        assert!(plan(Layout::Int5, 23.0, 78).is_err() && plan(Layout::Int5, 23.0, 79).is_ok());
        // a computer that does not say how much it has is taken to have enough (the allocation says if it has not)
        assert!(lossy_plan_with(&a, model, Layout::Int5, text, None, 23 << 30, "NVIDIA RTX 4090", None, None).unwrap().experts.is_some());
    }

    /// `expert_cache_for` is what `run` would give the directory: the plan of its own config on the memory the GPU has free (a directory of a config alone is sized by the plan's bits a weight), the places of 23 GiB
    /// free as the plan test above has them, none where the GPU holds the model whole, and a refusal where the computer has too little memory.
    #[test]
    fn the_cache_a_directory_gets_is_the_plans() {
        let dir = std::env::temp_dir().join(format!("glyd-engine-cache-for-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("config.json"), include_str!("../tests/fixtures/nemotron_h_config.json")).unwrap();
        let gib = 1u64 << 30;
        assert_eq!(expert_cache_for(&dir, Layout::Int5, 23 * gib, "a GPU", Some(128 * gib)).unwrap().unwrap().slots, 106);
        assert_eq!(expert_cache_for(&dir, Layout::Int5, 31 * gib, "a GPU", Some(128 * gib)).unwrap().unwrap().slots, 163);
        // wren: 3,096,576 bytes an expert, and the rest of the model (six-bit) a little over 7 GiB
        let wren = expert_cache_for(&dir, Layout::Int4, 23 * gib, "a GPU", Some(128 * gib)).unwrap().unwrap();
        assert_eq!((wren.slots, wren.expert_bytes, wren.host_bytes()), (122, 3_096_576, 40 * 512 * 3_096_576), "a 24 GB card");
        assert_eq!(expert_cache_for(&dir, Layout::Int4, 31 * gib, "a GPU", Some(128 * gib)).unwrap().unwrap().slots, 191, "a 32 GB card");
        assert!(expert_cache_for(&dir, Layout::Int4, 140 * gib, "a GPU", Some(128 * gib)).unwrap().is_none());
        let r = expert_cache_for(&dir, Layout::Int4, 23 * gib, "a GPU", Some(40 * gib)).err().unwrap();
        assert!(r.code == CANNOT && r.what.contains("this computer has 40.0 GiB free"), "{r}");
        // Hummingbird: 2,056,068 bytes an expert (the mean of the recipe hb40), no memory for the host's half; the 15 GiB a 16 GB card (RTX 4080) leaves hold 84 experts a layer of 512 (6.43 GiB of cache), a 24 GB card's 23 GiB
        // 188 and a 32 GB card's 31 GiB 293 (the figures of a config alone, sized at the plan's bits a weight; a directory's own bytes are what `run` uses)
        let hb = |f: u64| expert_cache_for(&dir, Layout::Hummingbird, f * gib, "a GPU", Some(128 * gib)).unwrap().unwrap();
        assert_eq!((hb(15).slots, hb(23).slots, hb(31).slots), (84, 188, 293));
        let c = hb(15);
        assert_eq!((c.expert_bytes, c.host_bytes(), c.host_half_bytes(), c.in_place), (2_056_068, 40 * 512 * 2_056_068, 0, true));
        // the pod job asks for these (tools/cards/nh_offload_job.sh): a 48 GB card at swift holds the caches of a 24 GB and a 32 GB card, the 24 and 32 GB cards their own at wren
        let job = include_str!("../../tools/cards/nh_offload_job.sh");
        assert!(
            job.contains("CAPS=\"24g:auto:23 32g:auto:31 48g:auto\"") && job.contains("CAPS=\"card:auto\"") && job.contains("CAP16=\"16g:auto:15\""),
            "the job's caches"
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// `--expert-cache GIB` gives the places of the cache that much of the GPU's memory (and a cache to a GPU that would hold the model whole): 12 GiB are 85 experts a layer of 512; more than the GPU has free, or
    /// fewer than a step's choices a layer, are refused; what is not a number is the user's (status 1), and a model with no experts to keep is told so.
    #[test]
    fn an_expert_cache_is_asked_for_in_gib() {
        let text = include_str!("../tests/fixtures/nemotron_h_config.json");
        let model = "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16";
        let plan = |cache: &str, free_gib: u64| {
            lossy_plan_with(
                &args(&["run", model, "--expert-cache", cache]),
                model,
                Layout::Int5,
                text,
                None,
                free_gib << 30,
                "NVIDIA L40S",
                None,
                Some(128 << 30),
            )
        };
        let c = plan("12", 46).unwrap().experts.unwrap();
        assert_eq!(c.slots, 85);
        assert_eq!(
            plan("12", 140).unwrap().experts.unwrap().slots,
            85,
            "a cache is asked for of a GPU that would hold the model: the experts are in the computer's memory all the same"
        );
        let r = plan("40", 46).err().unwrap();
        assert!(
            r.code == CANNOT
                && r.what
                    .contains("a cache of 40.0 GiB for the experts and the 8.0 GiB the rest of the model takes are more than the NVIDIA L40S's 46.0 GiB free"),
            "{r}"
        );
        assert!(plan("3", 46).is_err() && plan("3.5", 46).is_ok(), "22 experts a layer are 3.1 GiB");
        for bad in ["abc", "0", "-2", "nan"] {
            let r = plan(bad, 46).err().unwrap();
            assert!(r.code == 1 && r.what.contains(&format!("--expert-cache {bad}: not a number of GiB")), "{r}");
        }
        let qwen = include_str!("../../research/lossy-2026/qwen38-config.json");
        let r = lossy_plan_with(
            &args(&["run", "Qwen/Qwen3.8-27B", "--expert-cache", "8"]),
            "Qwen/Qwen3.8-27B",
            Layout::Int5,
            qwen,
            None,
            80 << 30,
            "NVIDIA A100",
            None,
            Some(128 << 30),
        )
        .err()
        .unwrap();
        assert!(r.code == 1 && r.what.contains("has no experts to keep in the computer's memory"), "{r}");
    }

    /// `--host-memory GIB|auto` is the computer's memory the experts may take: a number of GiB or `auto` (the default: what the computer has free, less the 6 GiB reserve), at most all the experts. Nemotron 3 Super at the
    /// Hummingbird level on a computer of 32 GB (26 GiB to give, 39.2 GiB of experts) keeps 26 of them in memory and reads the rest from the packed directory; with 128 GiB all of them are in memory, as ever; `--host-memory 20`
    /// holds 20 GiB of them whatever the computer has (and 100 is all of them); more than the computer has free less the reserve, and less than a step's choices in each width of each layer (3.4 GiB), are refused with the
    /// figures; the wren level, which has no packed directory to read experts from, keeps its refusal; a model with no experts is told so.
    #[test]
    fn the_computers_memory_for_the_experts_is_asked_for_in_gib() {
        let text = include_str!("../tests/fixtures/nemotron_h_config.json");
        let model = "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16";
        let gib = 1u64 << 30;
        let plan = |flag: &[&str], layout, host_gib: f64| {
            let mut v = vec!["run", model];
            v.extend_from_slice(flag);
            lossy_plan_with(&args(&v), model, layout, text, None, 23 * gib, "NVIDIA RTX 4090", None, Some((host_gib * gib as f64) as u64))
        };
        let hb = Layout::Hummingbird;
        let all = 40 * 512 * 2_056_068u64;
        // (auto: a computer of 128 GiB holds the experts, one of 32 GB gives 26 GiB of them)
        let c = plan(&[], hb, 128.0).unwrap().experts.unwrap();
        assert_eq!((c.ram, c.disk_bytes(), c.host_bytes()), (all, 0, all));
        let c = plan(&[], hb, 32.0).unwrap().experts.unwrap();
        assert_eq!((c.ram, c.disk_bytes(), c.slots), (26 * gib, all - 26 * gib, 188), "a computer of 32 GiB: the places on the GPU are the same");
        assert_eq!(plan(&["--host-memory", "auto"], hb, 32.0).unwrap().experts.unwrap().ram, 26 * gib);
        // (asked for)
        assert_eq!(plan(&["--host-memory", "20"], hb, 128.0).unwrap().experts.unwrap().ram, 20 * gib);
        assert_eq!(plan(&["--host-memory", "19.5"], hb, 32.0).unwrap().experts.unwrap().ram, 19 * gib + gib / 2);
        assert_eq!(plan(&["--host-memory", "100"], hb, 128.0).unwrap().experts.unwrap().ram, all);
        // (refused: more than the computer has free less the reserve; fewer than a step's choices in each width of each layer; not a number; the wren level; a model with no experts)
        let r = plan(&["--host-memory", "30"], hb, 32.0).err().unwrap();
        assert!(r.code == CANNOT && r.what.contains("--host-memory 30.0 GiB: the computer has 32.0 GiB free and keeps 6.0 GiB of them for the system"), "{r}");
        let r = plan(&["--host-memory", "2"], hb, 128.0).err().unwrap();
        assert!(r.code == CANNOT && r.what.contains("with its experts read from the disk it needs at least 9.4 GiB of the computer's memory for them (3.4 GiB for a step's choices in each width of each layer), and 2.0 GiB are to be given"), "{r}");
        assert!(plan(&["--host-memory", "3.5"], hb, 128.0).is_ok());
        let r = plan(&[], hb, 8.0).err().unwrap();
        assert!(r.code == CANNOT && r.what.contains("with its experts read from the disk it needs at least"), "a computer that gives 2 GiB: {r}");
        for bad in ["abc", "0", "-2", "nan"] {
            let r = plan(&["--host-memory", bad], hb, 128.0).err().unwrap();
            assert!(r.code == 1 && r.what.contains(&format!("--host-memory {bad}: not a number of GiB")), "{r}");
        }
        let r = plan(&[], Layout::Int4, 40.0).err().unwrap();
        assert!(r.code == CANNOT && r.what.contains("this computer has 40.0 GiB free"), "wren keeps its experts in memory or is refused: {r}");
        let qwen = include_str!("../../research/lossy-2026/qwen38-config.json");
        let r = lossy_plan_with(
            &args(&["run", "Qwen/Qwen3.8-27B", "--host-memory", "8"]),
            "Qwen/Qwen3.8-27B",
            Layout::Int5,
            qwen,
            None,
            80 << 30,
            "NVIDIA A100",
            None,
            Some(128 << 30),
        )
        .err()
        .unwrap();
        assert!(r.code == 1 && r.what.contains("has no experts to keep in the computer's memory"), "{r}");
        // (the share of the experts held, in bytes, for a 128 GiB computer and 32 GiB: what `held_in_memory` says)
        assert_eq!(
            (
                held_in_memory(all, None, Some(128 * gib)),
                held_in_memory(all, None, Some(32 * gib)),
                held_in_memory(all, None, None),
                held_in_memory(all, Some(gib), None)
            ),
            (all, 26 * gib, all, gib)
        );
    }

    fn plan_of(a: &Args, model: &str, text: String) -> Done<LoadPlan> {
        lossy_plan(a, model, Layout::Int5, &text, None, 140 << 30, "NVIDIA H200", None)
    }

    /// `run`, `serve`, `preflight` and `fetch` all check the family and the level against the table first (src/support.rs), in its words: a kind the engine does not run, a
    /// level its family does not run at, and nothing for a model at a level it runs.
    #[test]
    fn the_table_is_read_before_anything_else() {
        let dir = std::env::temp_dir().join(format!("glyd-table-check-{}", std::process::id()));
        let model = |name: &str, config: &str| {
            let d = dir.join(name);
            std::fs::create_dir_all(&d).unwrap();
            std::fs::write(d.join("config.json"), config).unwrap();
            d.to_string_lossy().into_owned()
        };
        let other = model("other", OTHER);
        let r = table_check(&other, None).unwrap_err();
        assert!(r.code == CANNOT && r.what.contains("is a OtherForCausalLM model"), "{r}");
        let dense = model("dense-36l", DENSE);
        assert!(table_check(&dense, None).is_ok() && table_check(&dense, Some(Layout::Tiered)).is_ok());
        assert!(table_check(&dense, Some(Layout::Int5)).is_ok());
        let r = table_check(&dense, Some(Layout::Hummingbird)).unwrap_err();
        assert!(r.code == CANNOT && r.what.contains("runs at the penguin, kestrel, swift and wren levels in this engine, not hummingbird"), "{r}");
        // (a model whose config is not to be had is the paths' own to say)
        assert!(table_check(&dir.join("nowhere").to_string_lossy(), Some(Layout::Int5)).is_ok());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// What a command does with a step it asks about: `--yes` and `GLYD_DEGRADE=yes` take it; a terminal in `run` is asked; `serve`, the service, a pipe and `GLYD_DEGRADE=no` refuse.
    #[test]
    fn a_step_that_asks_is_taken_asked_or_refused_by_who_can_answer() {
        use Degrade::*;
        assert_eq!(degrade_decision(true, None, false), Take, "--yes, whoever runs");
        assert_eq!(degrade_decision(false, Some("yes"), false), Take);
        assert_eq!(degrade_decision(false, Some("YES"), true), Take);
        assert_eq!(degrade_decision(false, None, true), Ask, "a terminal in run");
        assert_eq!(degrade_decision(false, Some("ask"), true), Ask);
        assert_eq!(degrade_decision(false, None, false), Refuse, "serve, the service, a pipe: nobody to ask");
        assert_eq!(degrade_decision(false, Some("ask"), false), Refuse);
        assert_eq!(degrade_decision(false, Some("no"), true), Refuse, "the config says never");
        assert_eq!(degrade_decision(true, Some("no"), true), Take, "the flag is this run's word");
    }

    /// The plan of a model on a GPU is a status 3 wherever it fails for the computer or the model (not for what the user typed): the memory, the context, the shape.
    #[test]
    fn what_a_model_needs_of_a_gpu_is_decided_before_anything_is_downloaded() {
        let cfg = Config::parse(DENSE, "config.json", None).unwrap();
        let l4 = gpu("NVIDIA L4", (8, 9), 22);
        let (layout, _, fit) = qwen3_plan(&args(&["run", "org/dense-36l"]), &cfg, "dense-36l", &l4, None, 1).unwrap();
        assert!(fit.context == 8192 && layout == Layout::Tiered && shape_checks("dense-36l", &cfg, &fit).is_ok());
        // a card with too little room: the vLLM way, with the numbers
        let small = qwen3_plan(&args(&["run", "org/dense-36l"]), &cfg, "dense-36l", &gpu("NVIDIA L4", (8, 9), 8), None, 1).unwrap_err();
        assert!(small.code == CANNOT && small.what.contains("takes about") && small.what.contains("has 8.0 GiB free"), "{small}");
        // a context that does not fit
        let big = qwen3_plan(&args(&["run", "org/dense-36l", "--context", "40960"]), &cfg, "dense-36l", &gpu("NVIDIA L4", (8, 9), 13), None, 1).unwrap_err();
        assert!(big.code == CANNOT && big.fix.contains("--context"), "{big}");
        // a context past the tables' reach on a model whose own window is longer
        let long = Config { max_pos: 262_144, ..cfg.clone() };
        // (the table's limit, which `fit` and `preflight` give in the same words)
        let r = qwen3_plan(&args(&["run", "m", "--context", "100000"]), &long, "m", &gpu("NVIDIA A100-SXM4-80GB", (8, 0), 70), None, 1).unwrap_err();
        assert!(r.code == CANNOT && r.what.contains("100000") && r.what.contains("40960"), "{r}");
        let (_, _, fit) = qwen3_plan(&args(&["run", "m"]), &long, "m", &l4, None, 1).unwrap();
        assert_eq!(fit.context, 8192); // (the default context is the engine's, whatever the window)
        assert!(shape_checks("m", &long, &fit).is_ok());
        // what the user typed is the user's to fix: status 1
        assert_eq!(qwen3_plan(&args(&["run", "m", "--layout", "tiny"]), &cfg, "m", &l4, None, 1).unwrap_err().code, 1);
        assert_eq!(qwen3_plan(&args(&["run", "m", "--context", "abc"]), &cfg, "m", &l4, None, 1).unwrap_err().code, 1);
    }

    /// What a first run writes is held to the disks before anything is downloaded: a packed copy that is made already (a glyd-v7 save at the level) is not written again.
    /// (This test is the only one that reads GLYD_CACHE.)
    #[test]
    fn a_first_run_counts_the_pack_it_has_not_made() {
        let root = std::env::temp_dir().join(format!("glyd-engine-first-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        std::env::set_var("GLYD_CACHE", &root);
        // (the packed copy's bytes: the plan's weights at the level)
        let pack = (Config::parse(DENSE, "config.json", None).unwrap().matrix_bytes() as f64 * bits(Layout::Tiered) / 16.0) as u64;
        let source = Source { dir: "/x".into(), repo: "org/model".into(), revision: Some("0123456789abcdef0123456789abcdef01234567".into()) };
        // (not made yet: the copy's bytes are counted, and a disk that has not that much left over refuses the run)
        assert!(pack > 0 && first_run_pack_bytes(pack, Layout::Tiered, Some(&source)) == pack && first_run_pack_bytes(pack, Layout::Tiered, None) == pack);
        assert!(machine::disk_refusal("model", Some(pack), Some(pack), true, 0, pack).is_some());
        assert!(first_run_checks("model", pack, Layout::Tiered, Some(&source), 0).is_ok());
        let d = store::packed_dir(&source, Layout::Tiered);
        std::fs::create_dir_all(&d).unwrap();
        std::fs::write(d.join("glyd.json"), r#"{"format": "glyd-v7", "layout": "mma", "level": "penguin", "packs": {}}"#).unwrap();
        // (made: nothing is counted for it, at that layout alone, and the same disk takes the run)
        assert!(store::cached_pack(&d, Layout::Tiered));
        assert_eq!(first_run_pack_bytes(pack, Layout::Tiered, Some(&source)), 0);
        assert_eq!(first_run_pack_bytes(pack, Layout::Int6, Some(&source)), pack);
        assert!(machine::disk_refusal("model", Some(pack), Some(pack), true, 0, 0).is_none());
        assert!(first_run_checks("model", pack, Layout::Tiered, Some(&source), 0).is_ok());
        std::fs::remove_dir_all(&root).unwrap();
    }

    /// Every way of loading the engine failing says it in words, and hands the model to the other runner.
    #[test]
    fn a_failure_to_load_the_engine_hands_the_model_to_vllm() {
        let cfg = Config::parse(DENSE, "config.json", None).unwrap();
        let g = gpu("NVIDIA L4", (8, 9), 22);
        let f = crate::plan::sizing::qwen3_fit(&cfg, "dense-36l", &g.name, g.free, Layout::Tiered, None, 1, true).unwrap();
        let said = |e: &str| engine_refusal(crate::ffi::Error(e.to_string()), "dense-36l", &f, &g);
        // (an error of the kind, whatever its words; then errors that come as words alone, as from another process: the status their words end in)
        let r = engine_refusal(crate::ffi::Error::of(crate::ffi::Kind::OutOfMemory, "no room for a KV cache"), "dense-36l", &f, &g);
        assert!(r.code == CANNOT && r.what.contains("ran out of memory"), "{r}");
        let r = engine_refusal(crate::ffi::Error::of(crate::ffi::Kind::of_status(2), "cuMemAlloc: no words for it"), "dense-36l", &f, &g);
        assert!(r.what.contains("ran out of memory"), "{r}");
        for (e, text) in [
            ("cuMemAlloc: out of memory (2)", "ran out of memory"),
            ("cuMemAlloc: CUDA error 2 (2)", "ran out of memory"),
            ("cuModuleLoadData: device kernel image is invalid (200)", "driver cannot run the engine's kernels"),
            // (no machine code for this GPU, on a driver the build runs on: the GPU is what the build lacks, not the driver)
            ("cuModuleLoadData: no kernel image is available for execution on the device (209)", "this build has no code for NVIDIA L4"),
            ("cuModuleLoadData: the provided PTX was compiled with an unsupported toolchain. (222)", "driver cannot run"),
            ("cuModuleLoadData: CUDA error 222 (222)", "driver cannot run"),
            ("Glyd's GPU library not found (set GLYD_GPU_LIB): nothing", "library not found"),
            ("cuLaunchKernel: an illegal memory access was encountered (700)", "illegal memory access"),
        ] {
            let r = said(e);
            assert!(r.code == CANNOT && r.what.contains(text), "{e}: {r}");
        }
        assert!(said("cuMemAlloc: out of memory (2)").what.contains("22.0 GiB were free"));
        // the same error on a driver below the build's minimum is the driver's
        let old = Gpu { driver: machine::driver_min(g.cc) - 1, ..g.clone() };
        let r =
            engine_refusal(crate::ffi::Error("cuModuleLoadData: no kernel image is available for execution on the device (209)".into()), "dense-36l", &f, &old);
        assert!(r.what.contains("driver cannot run the engine's kernels") && r.fix.contains("Update the NVIDIA driver"), "{r}");
    }

    /// A refusal before the engine is up goes into the file `--status-file` names, as one line, and not to the terminal, whatever its status (the program that started the engine runs the model
    /// another way); once the engine is up it is said where it always was. With no file named, nothing is written.
    #[test]
    fn the_reason_goes_where_the_caller_asked() {
        let dir = std::env::temp_dir().join(format!("glyd-engine-reason-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("why");
        let ui = RefCell::new(Ui::silent());
        let cannot = Refusal::unsupported("dense-36l does not fit this GPU: it needs about 14.2 GiB.", "Close programs that use the GPU");
        let own = Refusal::new("port 8000 is in use by another program", "Choose another: --port 8001");
        // no file named: said on the terminal, nothing written
        let none = Status::new();
        assert_eq!(refuse_to(&ui, &cannot, &none), 3);
        // a file named, the engine not up: the reason there, for either status, the status as it was
        let st = Status::new();
        st.set(file.clone());
        assert_eq!(refuse_to(&ui, &cannot, &st), 3);
        assert_eq!(std::fs::read_to_string(&file).unwrap(), "dense-36l does not fit this GPU: it needs about 14.2 GiB\n");
        assert_eq!(refuse_to(&ui, &own, &st), 1);
        assert_eq!(std::fs::read_to_string(&file).unwrap(), "port 8000 is in use by another program\n");
        // up: the file says so, and a refusal leaves it so
        st.ready();
        assert_eq!(refuse_to(&ui, &own, &st), 1);
        assert_eq!(std::fs::read_to_string(&file).unwrap(), "ready\n");
        // the flag, as the program that starts the engine gives it, among the user's own
        let a = args(&["run", "--status-file", file.to_str().unwrap(), "org/dense-36l", "--prompt", "hi"]);
        assert_eq!(
            (a.pos.clone(), a.flag("status-file"), a.flag("prompt")),
            (vec!["run".to_string(), "org/dense-36l".to_string()], Some(file.to_str().unwrap()), Some("hi"))
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// Standard input is read once the engine is up, so it is not asked before: only what is known to be empty is refused now.
    #[test]
    fn what_is_known_to_hold_nothing_is_not_read() {
        let dir = std::env::temp_dir().join(format!("glyd-engine-stdin-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let (empty, full) = (dir.join("empty"), dir.join("full"));
        std::fs::write(&empty, b"").unwrap();
        std::fs::write(&full, b"Say hello\n").unwrap();
        assert!(is_empty_input(&std::fs::File::open(&empty).unwrap()));
        assert!(!is_empty_input(&std::fs::File::open(&full).unwrap()));
        if cfg!(target_os = "linux") {
            assert!(is_empty_input(&std::fs::File::open("/dev/null").unwrap()) && !is_empty_input(&std::fs::File::open("/dev/zero").unwrap()));
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// What /v1/glyd/info says of a model served from the Hub's cache (`models--ORG--NAME/snapshots/REV`): the repo and the revision; the model it was made from and the Glyd that made it (glyd.json); each file's
    /// SHA-256, a blob link's from its name (the download checked it) and a small file's read; the level asked for (a tag, --lossy, a pre-packed repo's name; none) and the one that runs; the GPU; and the driver's
    /// version as /proc says it.
    #[test]
    fn a_served_model_is_described_by_its_files_its_levels_and_its_gpu() {
        let root = std::env::temp_dir().join(format!("glyd-engine-cli-about-{}", std::process::id()));
        let snap = root.join("models--glyd--Qwen3.5-9B-swift").join("snapshots").join("073a2132");
        let blob = "ea1b678c4787351b59de96915373a4a2aecc9eea49df70a64baf707b3e6f3830";
        std::fs::create_dir_all(&snap).unwrap();
        std::fs::create_dir_all(root.join("models--glyd--Qwen3.5-9B-swift").join("blobs")).unwrap();
        std::fs::write(root.join("models--glyd--Qwen3.5-9B-swift").join("blobs").join(blob), b"weights").unwrap();
        #[cfg(unix)]
        std::os::unix::fs::symlink(format!("../../blobs/{blob}"), snap.join("model-00001-of-00001.safetensors")).unwrap();
        std::fs::write(
            snap.join("glyd.json"),
            r#"{"format":"glyd-v7","glyd":"0.29.3","source":{"repo":"Qwen/Qwen3.5-9B","revision":"c202236"},"layout":"i5","level":"swift"}"#,
        )
        .unwrap();
        std::fs::write(snap.join("config.json"), "{}").unwrap();
        let held = crate::dev::Held { kv_type: "bf16", state_type: "fp32", ..Default::default() };
        let v = crate::tools::obj(about(
            "Qwen/Qwen3.5-9B:swift",
            &snap,
            Some("swift"),
            "swift",
            false,
            "swift level: about 5.5 bits per weight, not lossless",
            Some(&held),
            &gpu("NVIDIA GeForce RTX 4080 SUPER", (8, 9), 15),
        ));
        let get = |path: &[&str]| path.iter().fold(&v, |v, k| v.get(k).unwrap_or_else(|| panic!("no {k}"))).clone();
        assert_eq!(
            (get(&["model", "id"]), get(&["model", "repo"]), get(&["model", "revision"])),
            (Value::from("Qwen/Qwen3.5-9B:swift"), Value::from("glyd/Qwen3.5-9B-swift"), Value::from("073a2132"))
        );
        assert_eq!((get(&["model", "source", "revision"]), get(&["model", "made_by"])), (Value::from("c202236"), Value::from("glyd 0.29.3")));
        let files = get(&["model", "files"]);
        let sha =
            |name: &str| files.as_array().unwrap().iter().find(|f| f.get("name").and_then(Value::as_str) == Some(name)).and_then(|f| f.get("sha256")).cloned();
        assert_eq!(sha("config.json"), Some(Value::from("44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a"))); // (SHA-256 of "{}")
        #[cfg(unix)]
        assert_eq!(sha("model-00001-of-00001.safetensors"), Some(Value::from(blob)));
        assert_eq!(
            (get(&["level", "requested"]), get(&["level", "effective"]), get(&["level", "lossless"])),
            (Value::from("swift"), Value::from("swift"), Value::Bool(false))
        );
        assert_eq!(
            (get(&["precision", "kv"]), get(&["precision", "state"]), get(&["device", "compute_capability"]), get(&["device", "cuda"])),
            (Value::from("bf16"), Value::from("fp32"), Value::from("8.9"), Value::from("13.0"))
        );
        // a directory out of the Hub's cache has no repo or revision
        let plain = crate::tools::obj(about("m", &root, None, "swift", false, "w", None, &gpu("g", (8, 9), 15)));
        assert_eq!(
            (plain.get("model").and_then(|m| m.get("repo")), plain.get("level").and_then(|l| l.get("requested"))),
            (Some(&Value::Null), Some(&Value::Null))
        );
        std::fs::remove_dir_all(&root).unwrap();
        // the level asked for
        for (given, lossy, want) in [
            ("Qwen/Qwen3.5-9B:swift", false, Some("swift")),
            ("Qwen/Qwen3.5-9B:PENGUIN", false, Some("penguin")),
            ("Qwen/Qwen3.5-9B", true, Some("kestrel")),
            ("glyd/Qwen3.5-9B-swift", false, Some("swift")),
            ("/models/q-swift", false, None),
            ("Qwen/Qwen3.5-9B:sparrow", false, None),
        ] {
            assert_eq!(requested(given, lossy), want, "{given}");
        }
        // the driver's version, from /proc/driver/nvidia/version's first line
        assert_eq!(nv_version("NVRM version: NVIDIA UNIX Open Kernel Module for x86_64  595.91.07  Release Build  (dvs-builder@U22-I3-AF03-09-1)  Thu Sep 11 22:53:16 UTC 2025\nGCC version:  gcc version 13.2.0"), Some("595.91.07".to_string()));
        assert_eq!(nv_version("NVRM version: NVIDIA UNIX x86_64 Kernel Module  580.65.06  Sun Jul 27 07:09:24 UTC 2025"), Some("580.65.06".to_string()));
        assert_eq!(nv_version(""), None);
    }
}
