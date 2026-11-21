//! The commands a person runs: `glyd-engine run MODEL` (the first run makes the model ready, then a chat in the terminal and at one URL, or one answer with --prompt), `serve MODEL` (the API
//! and the page), `doctor`, `login` and `supports MODEL`. `glyd run` and `glyd serve` start these.

use crate::args::Args;
use crate::chat::{Chat, Message};
use crate::conv::{Gen, Part, Request, Splitter, Talk};
use crate::dev::{Dev, Gpu};
use crate::ffi::Blas;
use crate::fit;
use crate::hub::{self, Hub};
use crate::levels::{self, is_qat, layout_for, QAT_WREN};
pub use crate::levels::{packed_repo, ship_layout, short_name};
use crate::machine;
use crate::model::{Engine, Opts};
use crate::model_nh::{EngineNh, OptsNh};
use crate::nemotron::{pack_layout, ConfigNemotronH, NhKind};
use crate::offload::{CachePlan, ExpertStore};
use crate::plan::{reserve_of, Kind, LoadPlan, Target, GEMMA_CHUNK, GEMMA_DRAFT_BYTES, HOST_MARGIN, LOSSY_CTX, LOSSY_MIN_CTX, LOSSY_WORK};
use crate::qwen35::Config35;
use crate::qwen35::Pack35;
use crate::sampler::{self, Params};
use crate::server::{self, Job, Piece};
use crate::status::{Status, STATUS};
use crate::store;
use crate::tok::Tok;
use crate::ui::{fmt_time, gb, gib, Done, Refusal, Ui};
use crate::weights::{choose_layout, packed_bytes, packed_layout, v7_manifest, Config, Layout};
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
fn level(model: &str, lossy: bool) -> Done<(&str, Option<Layout>)> {
    levels::parse(model, lossy).map_err(|e| Refusal::new(e.what, e.fix))
}

/// Whether MODEL is a Gemma 4 model: a directory whose config says so, or a Hub name of one (google/gemma-4-..., glyd/gemma-4-...).
fn is_gemma4_model(model: &str) -> bool {
    match std::fs::read_to_string(Path::new(model).join("config.json")) {
        Ok(text) => crate::model_gemma4::is_gemma4(&text),
        Err(_) => short_name(model).to_ascii_lowercase().starts_with("gemma-4"),
    }
}

/// A level's name from its layout.
fn level_name(layout: Layout) -> &'static str {
    levels::name_of(layout)
}

/// The wren repo of a Gemma 4 model that has one (the instruction-tuned 12B and 31B, packed exact from Google's QAT checkpoint), from its Hub name or its penguin repo's.
fn gemma_wren(model: &str) -> Option<String> {
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

/// The level run, serve and preflight take MODEL at: a glyd-v7 directory's own, else the one asked; a Gemma 4 model without one is penguin (it runs on the engine's Gemma path, its packs lossless:
/// the levels' route with the lossless layout).
fn run_level(model: &str, asked: Option<Layout>) -> Done<Option<Layout>> {
    Ok(v7_level(model, v7_layout(model)?, asked)?.or_else(|| (is_gemma4_model(model) || dynamic_family(model).is_some()).then_some(Layout::Tiered)))
}

/// The family of a model the dynamic engine runs from its release (DeepSeek-V4, GLM-5.3-Flash, Muse Spark, Nemotron-H; GLY-354), by its config; None for
/// another path's, a glyd-v7 directory's, or a config not to be had (offline).
fn dynamic_family(model: &str) -> Option<&'static str> {
    if v7_layout(model).ok().flatten().is_some() {
        return None;
    }
    crate::dynamic::split::family(&config_text(model).ok()?)
}

// ---- the GPU --------------------------------------------------------------------------------------------------------------------------------------------

/// A failure of what this engine does that vLLM does not need: the status that hands the model to it.
fn cannot(e: impl std::fmt::Display) -> Refusal {
    Refusal::unsupported(e.to_string(), "")
}

/// The GPU a run uses: the one with the most free memory of those the driver shows (CUDA_VISIBLE_DEVICES applied), as the earlier release chose; the contexts the look at the others made are let go.
fn gpu_pick() -> Done<Gpu> {
    let gpus = Dev::probe_all().map_err(|e| {
        if e.0.contains("no CUDA driver") || e.0.contains("libcuda") {
            Refusal::unsupported(
                "no NVIDIA driver was found on this computer",
                "glyd run needs an NVIDIA GPU of the Ampere generation or newer and its driver (https://www.nvidia.com/drivers). On a computer without one, glyd compresses files: glyd --help",
            )
        } else {
            Refusal::unsupported(format!("the GPU cannot be used: {}", e.0), "Check that nvidia-smi shows it")
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

/// An error from loading the engine in words (memory, a driver too old for the kernels), as the driver words it ("cuMemAlloc: out of memory (2)": the call, its text and its status). Whatever it
/// is, the engine is not the only way to run the model: status 3.
fn engine_refusal(e: crate::ffi::Error, name: &str, fit: &LoadPlan, gpu: &Gpu) -> Refusal {
    let t = e.0.to_lowercase();
    let status: Option<i32> = t.rsplit_once('(').and_then(|(_, n)| n.strip_suffix(')')).and_then(|n| n.parse().ok());
    if t.contains("out of memory") || t.contains("no room for a kv cache") || status == Some(2) {
        return Refusal::unsupported(
            format!("the GPU ran out of memory loading {name} (about {} are needed; {} were free at the start)", gib(fit.gpu_total()), gib(gpu.free)),
            format!(
                "Close programs that use the GPU (nvidia-smi lists them), or ask for less context: --context {}, or run a smaller model",
                (fit.context / 2).max(2048) / 256 * 256
            ),
        );
    }
    // (209: no machine code for this GPU, where the driver is new enough for the build's kernels; the images the driver cannot read: 200 and 218; a toolchain newer than the driver: 222)
    let no_image = t.contains("no kernel image") || status == Some(209);
    if no_image && gpu.driver >= machine::driver_min(gpu.cc) {
        return Refusal::unsupported(no_code_for(gpu), no_code_fix(gpu));
    }
    if ["no kernel image", "kernel image is invalid", "unsupported toolchain", "unsupported_ptx", "invalid_image"].iter().any(|w| t.contains(w))
        || matches!(status, Some(200 | 209 | 218 | 222))
    {
        let need = machine::driver_need(gpu.cc);
        let series = machine::driver_series(machine::driver_min(gpu.cc));
        return Refusal::unsupported(
            format!("the GPU's driver cannot run the engine's kernels ({})", e.0),
            machine::instead(
                &format!("Update the NVIDIA driver (the engine's kernels need {need}), or use the vLLM-based glyd"),
                &format!("Update the NVIDIA driver to {series} or newer: on Ubuntu `sudo apt install nvidia-driver-{series}` and reboot, elsewhere the installer at https://www.nvidia.com/drivers"),
            ),
        );
    }
    cannot(e.0)
}

// ---- a model made ready -------------------------------------------------------------------------------------------------------------------------------

struct Prepared {
    dir: PathBuf,
    tables: PathBuf,
    ctx: usize,
    fit: LoadPlan,
}

fn architecture(cfg_text: &str) -> Option<(String, bool)> {
    let v = json::parse(cfg_text).ok()?;
    let arch = v.get("architectures")?.as_array()?.first()?.as_str()?.to_string();
    let scaled = v.get("rope_scaling").is_some_and(|r| *r != Value::Null);
    Some((arch, scaled))
}

/// What a Qwen3 config.json sets that the engine does not do (it would run such a model wrongly, and without a word): a sliding window or layers that are not full attention, biases in the attention,
/// an activation other than silu.
fn unsupported_setting(cfg_text: &str) -> Option<&'static str> {
    let v = json::parse(cfg_text).ok()?;
    let on = |k: &str| v.get(k).is_some_and(|x| !matches!(x, Value::Null | Value::Bool(false)));
    if on("sliding_window") && v.get("use_sliding_window").is_some_and(|x| *x != Value::Bool(false)) {
        return Some("a sliding window");
    }
    if v.get("layer_types").and_then(Value::as_array).is_some_and(|t| t.iter().any(|l| l.as_str().is_some_and(|l| l != "full_attention"))) {
        return Some("layers that are not full attention");
    }
    if v.get("attention_bias") == Some(&Value::Bool(true)) {
        return Some("biases in the attention");
    }
    if v.get("hidden_act").and_then(Value::as_str).is_some_and(|a| a != "silu") {
        return Some("an activation other than silu");
    }
    None
}

/// What to do with a model the engine does not run (yet): vLLM runs it where it is installed, else the whole tool does.
fn unrun_fix() -> String {
    machine::instead("Run it with the vLLM-based glyd, which is installed here", &format!("The engine does not run it yet: {}", machine::FULL_TOOL))
}

/// The engine runs Qwen3 dense models (Qwen3ForCausalLM) with no rope scaling: a Refusal (exit 3) for the rest.
fn supported(name: &str, cfg_text: &str) -> Done<()> {
    let Some((arch, scaled)) = architecture(cfg_text) else {
        return Err(Refusal::unsupported(format!("{name}: its config.json does not say what kind of model it is"), "The engine runs Qwen3 models so far"));
    };
    if crate::nemotron::is_nemotron_h(cfg_text) {
        return Err(Refusal::unsupported(
            format!("{name} is a Nemotron-H model: this engine runs it at the hummingbird, swift and kestrel levels, not lossless"),
            machine::instead(
                "Run it at a level: glyd run MODEL:hummingbird on a 16 to 32 GB card (or :swift, :kestrel on a 96 GB one or larger); lossless it runs with the vLLM-based glyd, which is installed here",
                &format!("Run it at a level: glyd run MODEL:hummingbird on a 16 to 32 GB card (or :swift, :kestrel on a 96 GB one or larger); lossless, {}", machine::FULL_TOOL),
            ),
        ));
    }
    if arch != "Qwen3ForCausalLM" {
        return Err(Refusal::unsupported(
            format!("{name} is a {arch} model, and this engine runs Qwen3 models so far"),
            machine::instead(
                "Other models run with the vLLM-based glyd, which is installed here",
                &format!("Pick a Qwen3 model (glyd run Qwen/Qwen3-8B), or {}", machine::FULL_TOOL),
            ),
        ));
    }
    if scaled {
        return Err(Refusal::unsupported(format!("{name} uses rope scaling, which this engine does not run yet"), unrun_fix()));
    }
    if let Some(what) = unsupported_setting(cfg_text) {
        return Err(Refusal::unsupported(format!("{name} has {what}, which this engine does not run yet"), unrun_fix()));
    }
    Ok(())
}

/// A dense product for the long prompts (cuBLAS where it is, else the engine's own on Ampere, Ada and Hopper): the decoded matrix and the larger chunk that goes with it.
fn dense_product(cc: (i32, i32)) -> bool {
    Blas::present() || (matches!(cc.0, 8 | 9) && crate::tune!("GLYD_ENGINE_GEMM").map_or(true, |v| v != "lib"))
}

/// The GPU `run` picked, as a plan sees it, and the computer's memory the experts may take.
fn target(gpu: &Gpu) -> Target {
    Target { ordinal: gpu.ordinal, name: gpu.name.clone(), cc: gpu.cc, total: gpu.total, free: gpu.free, host: crate::offload::host_memory() }
}

/// The layout and the fit of a model on a GPU: what `prepare` settles before it downloads a thing, and `preflight` before it says the engine will run the model. `packed`: the layout of a
/// directory `pack` made, which is the layout there.
pub(crate) fn plan(a: &Args, cfg: &Config, name: &str, gpu: &Gpu, packed: Option<Layout>, users: usize) -> Done<(Layout, String, LoadPlan)> {
    plan_for(a, cfg, name, &target(gpu), packed, users)
}

fn plan_for(a: &Args, cfg: &Config, name: &str, t: &Target, packed: Option<Layout>, users: usize) -> Done<(Layout, String, LoadPlan)> {
    let asked = if a.has("context") || a.has("ctx") { Some(a.either("context", "ctx", 0usize)?) } else { None };
    let (layout, why) = match packed {
        Some(l) => (l, "as packed".to_string()),
        None => choose_layout(a.layout()?, t.cc, t.total, cfg.matrix_bytes(), 0),
    };
    // (what the last load of this model here took beyond its plan is planned as already taken: `calibrated`)
    let extra = calibration(name, levels::name_of(layout), &t.name);
    let mut fit = fit::plan(cfg, name, &t.name, t.free.saturating_sub(extra), layout, asked, users, dense_product(t.cc))?;
    calibrated(&mut fit, t.free, extra);
    Ok((layout, why, fit))
}

/// The bytes of this GPU's free memory the planner takes as already used for `model` at `level`: what the last load took beyond its plan (src/plan.rs
/// `calibration_in`; the one-GPU paths' only: the others' loads do not count their weights apart).
fn calibration(model: &str, level: &str, gpu: &str) -> u64 {
    crate::plan::calibration_in(&store::cache_root(), model, level, gpu)
}

/// `plan` made for the GPU's free memory less `extra`, said as it is: the free memory as it was, and `extra` a part of the reserve.
pub(crate) fn calibrated(p: &mut LoadPlan, free: u64, extra: u64) {
    p.free = free;
    p.reserve += extra;
}

/// The plan `run` loads MODEL (its config.json `text`) at `level` on the GPU `t`, with the flags of `a`, or the refusal `run` gives: what `fit` prints
/// and `run`, `serve` and `preflight` size with. `users`: the sequences a server holds at once (the Qwen3 path's; the lossy paths hold one). Ok(None): this
/// family's path is not sized by its loader yet, and the planner's estimate stands (src/plan.rs).
pub(crate) fn size(
    a: &Args,
    model: &str,
    text: &str,
    level: &'static str,
    devs: &[Target],
    t: &Target,
    reach: &dyn Fn(&[i32]) -> bool,
    users: usize,
) -> Done<Option<LoadPlan>> {
    use crate::support::Family;
    let name = short_name(model);
    match crate::support::row(crate::support::family(text, &name)?).family {
        Family::Qwen3 => {
            supported(&name, text)?;
            let cfg = Config::parse(text, "config.json", None).map_err(cannot)?;
            let (_, _, p) = plan_for(a, &cfg, &name, t, None, users)?;
            shape_checks(&name, &cfg, &p)?;
            Ok(Some(p))
        }
        Family::Qwen35 | Family::Gemma4 | Family::NemotronH => {
            let Some(layout) = levels::layout_of(level) else { return Ok(None) };
            if layout == Layout::Tiered && crate::nemotron::is_nemotron_h(text) {
                return size_local_dynamic(a, model, text, "Nemotron-H", devs, reach);
            }
            lossy_plan_with(a, model, layout_for(model, layout), text, None, t.free, &t.name, None, t.host).map(Some)
        }
        Family::Kimi => size_local_dynamic(a, model, text, "Kimi-K3", devs, reach),
        Family::Glm => size_local_dynamic(a, model, text, "GLM-5.3-Flash", devs, reach),
        Family::DeepSeekV4 => size_local_dynamic(a, model, text, "DeepSeek-V4", devs, reach),
        Family::Muse => size_local_dynamic(a, model, text, "Muse Spark", devs, reach),
        Family::FlashNext => Ok(None),
    }
}

/// `size_dynamic` of a model whose files are on this computer (a directory, or the Hub's cache); Ok(None) where they are not yet: its bytes are the files'
/// headers' (ROADMAP P2b), and the planner's estimate stands until then.
fn size_local_dynamic(a: &Args, model: &str, text: &str, family: &'static str, devs: &[Target], reach: &dyn Fn(&[i32]) -> bool) -> Done<Option<LoadPlan>> {
    let dir = if Path::new(model).join("config.json").exists() { Some(PathBuf::from(model)) } else { Source::find(model).ok().map(|s| s.dir) };
    let Some(dir) = dir else { return Ok(None) };
    let host = devs.first().and_then(|t| t.host);
    size_dynamic(a, &dir, text, family, devs, host, reach).map(Some)
}

/// The model's shape against what the engine's KV cache takes, and its context against what the tables cover: refused before anything is downloaded.
fn shape_checks(name: &str, cfg: &Config, fit: &LoadPlan) -> Done<()> {
    match machine::config_refusal(name, cfg).or_else(|| machine::context_refusal(name, fit.context)) {
        Some(r) => Err(r),
        None => Ok(()),
    }
}

/// What a first run writes (the download, and the packed copy) against the disks, and a refusal the tables' rule gave this model before: refused before anything is downloaded. `source`: the model
/// where it is on this computer already; `download`: the bytes still to fetch.
fn first_run_checks(name: &str, cfg: &Config, fit: &LoadPlan, layout: Layout, source: Option<&Source>, download: u64) -> Done<()> {
    let made = source.is_some_and(|s| store::is_packed(&store::packed_dir(s, layout)));
    let pack = if made { 0 } else { (cfg.matrix_bytes() as f64 * fit::bits(layout) / 16.0) as u64 };
    let (models, cache) = (store::existing_ancestor(&hub::hub_cache()), store::existing_ancestor(&store::cache_root()));
    if let Some(r) = machine::disk_refusal(name, hub::disk_free(&models), hub::disk_free(&cache), store::same_filesystem(&models, &cache), download, pack) {
        return Err(r);
    }
    if let (Some(s), Ok(tier)) = (source, crate::kvcal::check_positions(fit.context)) {
        for l in [Layout::Tiered, Layout::Twelve] {
            if let Some(why) = store::known_refusal(&store::packed_dir(s, l), tier) {
                return Err(Refusal::unsupported(why, ""));
            }
        }
    }
    Ok(())
}

fn prepare(a: &Args, gpu: &Gpu, ui: &RefCell<Ui>, with_tables: bool, users: usize) -> Done<Prepared> {
    let model = a.need(1, "MODEL")?;
    let name = short_name(model);
    let direct = Path::new(model);
    // a directory `pack` made
    if direct.join("glyd.json").exists() {
        if let Some(why) = crate::weights::earlier_format(direct) {
            return Err(Refusal::unsupported(format!("{model} is {why}"), "Pack the model again: glyd pack MODEL DIR"));
        }
        supported(&name, &std::fs::read_to_string(direct.join("config.json")).map_err(|e| Refusal::unsupported(format!("{model}/config.json: {e}"), ""))?)?;
        let cfg = Config::load(direct).map_err(cannot)?;
        let (_, _, fit) = plan(a, &cfg, &name, gpu, Some(packed_layout(direct).unwrap_or(Layout::Tiered)), users)?;
        shape_checks(&name, &cfg, &fit)?;
        let tables = match a.flag("tables") {
            Some(t) => PathBuf::from(t),
            None if direct.join("kv-tables.safetensors").exists() => direct.join("kv-tables.safetensors"),
            None if !with_tables => PathBuf::new(),
            None => store::tables(direct, model, fit.context, gpu.ordinal, ui)?,
        };
        return Ok(Prepared { dir: direct.to_path_buf(), tables, ctx: fit.context, fit });
    }
    // a Hub-format directory, or a repo in the cache, else the Hub
    let source: Source = match Source::find(model) {
        Ok(s) => s,
        Err(_) => {
            let (hub, remote, wanted) = store::hub_plan(model, ui)?;
            // what kind of model it is, and whether it fits, before its weights are fetched
            let text = hub.text(model, "main", "config.json").map_err(|e| match e {
                hub::HubError::Denied(_) => store::hub_refusal(model),
                e => Refusal::new(format!("cannot read {model} from the Hugging Face Hub ({e})"), "Check the network connection and run this again"),
            })?;
            supported(&name, &text)?;
            let small: Vec<hub::Entry> = wanted.iter().filter(|e| e.path == "config.json" || e.path == "generation_config.json").cloned().collect();
            hub.fetch(&remote, &small, &|_, _| {})?;
            let cfg = Config::load(&hub.snapshot(&remote)).map_err(cannot)?;
            let (layout, _, fit) = plan(a, &cfg, &name, gpu, None, users)?;
            shape_checks(&name, &cfg, &fit)?;
            first_run_checks(&name, &cfg, &fit, layout, None, hub.missing(&remote, &wanted))?;
            store::download(&hub, &remote, &wanted, model, ui)?;
            Source::find(model).map_err(cannot)?
        }
    };
    supported(
        &name,
        &std::fs::read_to_string(source.dir.join("config.json")).map_err(|e| Refusal::unsupported(format!("{}: {e}", source.dir.display()), ""))?,
    )?;
    let cfg = Config::load(&source.dir).map_err(cannot)?;
    let (layout, why, fit) = plan(a, &cfg, &name, gpu, None, users)?;
    shape_checks(&name, &cfg, &fit)?;
    first_run_checks(&name, &cfg, &fit, layout, Some(&source), 0)?;
    ui.borrow_mut().line(&format!(
        "{name}: {} on the GPU with Glyd, instead of {}; layout {}: {why}.",
        gib(fit.weights),
        gib(cfg.matrix_bytes()),
        layout.name()
    ));
    let expect = (cfg.matrix_bytes() as f64 * fit::bits(layout) / 16.0) as u64;
    let dir = store::pack(&source, model, layout, expect, ui)?;
    let tables = if with_tables { store::tables(&dir, model, fit.context, gpu.ordinal, ui)? } else { PathBuf::new() };
    Ok(Prepared { dir, tables, ctx: fit.context, fit })
}

struct Ready {
    engine: Engine,
    tok: Tok,
    chat: Option<Chat>,
    defaults: Params,
    name: String,
    /// Sequences the engine runs at once (1: the one-reply way).
    users: usize,
    /// The directory the model is read from, and the GPU it runs on.
    dir: PathBuf,
    gpu: Gpu,
}

/// Sequences a server runs at once: --users, else 4 (1 to MAX_USERS; the dense Qwen3 engine takes at most MAX_ROWS of them, a level's engine as many as its memory holds).
fn users_of(a: &Args) -> Done<usize> {
    use crate::sched::MAX_USERS;
    let n: usize = a.num("users", 4usize).map_err(|e| Refusal::new(e.0, format!("Give a number from 1 to {MAX_USERS}: --users 4")))?;
    if !(1..=MAX_USERS).contains(&n) {
        return Err(Refusal::new(format!("--users {n}: 1 to {MAX_USERS} sequences at once"), format!("Give a number from 1 to {MAX_USERS}: --users 4")));
    }
    Ok(n)
}

fn start(a: &Args, ui: &RefCell<Ui>, several: bool) -> Done<Ready> {
    // (the dense engine's decode step takes at most MAX_ROWS rows)
    let users = if several { users_of(a)?.min(crate::model::MAX_ROWS) } else { 1 };
    let t0 = Instant::now();
    let gpu = gpu_pick()?;
    // (the machine checks of every other path: the CPU architecture, the driver, a MIG slice)
    if let Some(r) = machine::gpu_refusal(&gpu, std::env::consts::ARCH, true, machine::ANY_FAMILY) {
        return Err(r);
    }
    let model = a.need(1, "MODEL")?.to_string();
    let name = short_name(&model);
    let p = prepare(a, &gpu, ui, true, users)?;
    ui.borrow_mut().line(&format!("Plan: {}.", p.fit.text()));
    ui.borrow_mut().status(&format!("Loading {name} onto the GPU..."));
    let o = Opts {
        ctx: p.ctx,
        chunk: p.fit.chunk,
        tables: Some(p.tables.clone()),
        graph: !(cfg!(feature = "dev") && a.has("no-graph")),
        pool: if several { crate::tune!("GLYD_ENGINE_POOL").and_then(|v| v.parse().ok()).unwrap_or(p.ctx * users) } else { 0 },
        users: if several { users } else { 0 },
        device: gpu.ordinal,
        ..Opts::default()
    };
    let engine = Engine::new(&p.dir, &o).map_err(|e| engine_refusal(e, &name, &p.fit, &gpu))?;
    loaded_report(&model, &p.fit, &[&gpu], engine.w.bytes as u64);
    let tok = Tok::load(&p.dir).map_err(cannot)?;
    let chat = Chat::load(&p.dir).map_err(cannot)?;
    let mut defaults = sampler::generation_defaults(&p.dir);
    if let Some(t) = a.flag("temp") {
        defaults.temperature = t.parse().map_err(|_| Refusal::new(format!("--temp {t}: not a number"), ""))?;
    }
    ui.borrow_mut().line(&format!(
        "Ready in {}: {name} on the {}, {} of weights, {} tokens of context{}.",
        fmt_time(t0.elapsed().as_secs_f64()),
        gpu.name,
        gib(engine.w.bytes as u64),
        engine.capacity(),
        if several { format!(", {users} users at once") } else { String::new() }
    ));
    if crate::tune!("GLYD_ENGINE_VERBOSE").is_some() {
        ui.borrow_mut().line(&engine.describe());
    }
    Ok(Ready { engine, tok, chat, defaults, name: model, users, dir: p.dir, gpu })
}

/// What the load took of the GPU against the plan: the weights the engine counts and the memory the GPU lost since the plan was made. A development build
/// writes it as JSON to `GLYD_PLAN_REPORT` (the GPU check reads it) and says it with `GLYD_ENGINE_VERBOSE`: the numbers the planner is calibrated on.
fn loaded_report(model: &str, plan: &LoadPlan, gpus: &[&Gpu], weights: u64) {
    let used = gpus.iter().map(|gpu| Dev::probe(gpu.ordinal).map_or(0, |g| gpu.free.saturating_sub(g.free))).sum::<u64>();
    crate::plan::remember_in(&store::cache_root(), model, plan, weights, used);
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

fn listen(a: &Args, name: &str, window: usize, tx: mpsc::Sender<Job>) -> Done<std::net::SocketAddr> {
    access_check(a)?;
    let host = a.flag("host").unwrap_or("127.0.0.1").to_string();
    let port: u16 = a.num("port", 8000u16)?;
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
            Err(e) if e.0.to_lowercase().contains("in use") && !explicit => continue,
            Err(e) if e.0.to_lowercase().contains("in use") => {
                return Err(Refusal::new(format!("port {p} is in use by another program"), format!("Choose another: --port {}", p + 1)))
            }
            Err(e) => return Err(Refusal::new(e.0, "")),
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
        Err(e) => eprintln!("glyd: --ollama-port {port}: {}: Ollama's API stays at the server's own port", e.0),
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
    let layout = run_level(&model, layout)?;
    // the family and the level against the table fit reads (src/support.rs); a glyd-v7 directory or a pre-packed repo is at its own
    // level, and a config not to be had now is the paths' own to say
    if v7_layout(&model)?.is_none() && levels::repo_level(&model).is_none() {
        if let Ok(text) = config_text(&model) {
            crate::support::check(&text, &model, &short_name(&model), layout.map_or("penguin", level_name))?;
        }
    }
    let mut a = a.clone();
    if let Some(m) = a.pos.get_mut(1) {
        *m = model;
    }
    Ok((a, layout))
}

pub fn serve(a: &Args) -> i32 {
    let ui = RefCell::new(Ui::new());
    let started = crate::monitor::clock();
    status_file(a);
    // (before the GPU is looked at and the model is read: an address that may not be served is refused first)
    if let Err(r) = access_check(a) {
        return fail(&ui, &r);
    }
    // the level is run's, read before the GPU is looked at; the name served keeps the tag
    let given = a.pos.get(1).cloned().unwrap_or_default();
    let asked = requested(&given, a.has("lossy"));
    let (a, layout) = match tagged(a) {
        Ok(x) => x,
        Err(r) => return fail(&ui, &r),
    };
    let a = &a;
    if let Some(layout) = layout {
        return serve_lossy(a, layout, given, &ui);
    }
    let mut ready = match start(a, &ui, true) {
        Ok(r) => r,
        Err(r) => return fail(&ui, &r),
    };
    ready.name = given;
    let (tx, rx) = mpsc::channel();
    let window = ready.engine.capacity();
    let addr = match listen(a, &ready.name, window, tx.clone()) {
        Ok(x) => x,
        Err(r) => return fail(&ui, &r),
    };
    let mut about = about(&ready.name, &ready.dir, asked, "penguin", true, "penguin level: lossless, every weight the original's", None, &ready.gpu);
    about.push(("costs", crate::costs::served(&ready.dir, &ready.name, asked, window, ready.engine.chunk(), None)));
    let _ = server::INFO.set(server::Info { about, gpu: Some(ready.gpu.ordinal), held: None, concurrency: ready.users, window, refused: &[], started });
    STATUS.ready(); // (serving: from here a failure is the engine's to say, and nothing is started again)
    access_notes(&ui, a);
    ui.borrow_mut().line(&format!("{} is served at http://{addr}: the chat page at /, the API at /v1 (OpenAI-compatible). Ctrl-C stops it.", ready.name));
    let Ready { mut engine, tok, chat, defaults, name, users, .. } = ready;
    let vocab = engine.w.cfg.vocab;
    server::work(&mut engine, Arc::new(server::Shared { vocab, tok, chat, name, defaults, window, thinking: None, refused: &[] }), &rx, &tx, users);
    0
}

/// `serve MODEL:kestrel` (or `:swift`): the lossy level loaded as run loads it, served by the scheduler to up to --users sequences at once where the engine is a step of several (src/dynamic/rows.rs: Gemma 4),
/// else to one user at a time.
fn serve_lossy(a: &Args, layout: Layout, name: String, ui: &RefCell<Ui>) -> i32 {
    let (mut e, tok, chat, defaults, vocab, line, mut info) = match load_lossy(a, layout, &name, ui) {
        Ok(x) => x,
        Err(r) => return fail(ui, &r),
    };
    ui.borrow_mut().line(&line);
    let users = match users_of(a) {
        Ok(u) => u,
        Err(r) => return fail(ui, &r),
    };
    let several = match e.as_step(users) {
        Ok(s) => s.map(|(k, _)| k),
        Err(err) => return fail(ui, &cannot(err)),
    };
    if let Some(k) = several {
        ui.borrow_mut().line(&format!(
            "Up to {k} users at once{}.",
            if k < users { format!(" (the GPU's free memory holds {k} of the {users} asked for)") } else { String::new() }
        ));
        if let Some(i) = &mut info {
            (i.concurrency, i.refused, i.held) = (k, &[], e.held());
        }
    }
    let (tx, rx) = mpsc::channel();
    let window = e.capacity();
    let addr = match listen(a, &name, window, tx.clone()) {
        Ok(x) => x,
        Err(r) => return fail(ui, &r),
    };
    if let Some(i) = info {
        let _ = server::INFO.set(i);
    }
    STATUS.ready();
    access_notes(ui, a);
    ui.borrow_mut().line(&format!("{name} is served at http://{addr}: the chat page at /, the API at /v1 (OpenAI-compatible). Ctrl-C stops it."));
    let refused = if several.is_some() { &[][..] } else { server::ONE_AT_A_TIME };
    let shared = Arc::new(server::Shared { vocab, tok, chat, name, defaults, window, thinking: None, refused });
    match e.as_step(users) {
        Ok(Some((k, step))) => server::work_with(&mut crate::sched::Sched::new(step, k), shared, &rx, &tx),
        _ => server::work_gen(&mut *e, shared, &rx, &tx),
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
    let (a, layout) = match tagged(a) {
        Ok(x) => x,
        Err(r) => return fail(&ui, &r),
    };
    let a = &a;
    let thinking = if a.has("no-think") { Some(false) } else { None };
    // one answer for a prompt given, or piped in: a pipe is read once the engine is up, for where it stops before that the program that runs the model another way finds standard input as it was
    // (a pipe that is known to hold nothing, a file of no bytes or /dev/null, is refused now)
    let mut prompt = a.flag("prompt").map(str::to_string);
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
    if let Some(layout) = layout {
        return run_lossy(a, layout, &ui, prompt, piped, thinking);
    }
    let mut ready = match start(a, &ui, prompt.is_none() && !piped) {
        Ok(r) => r,
        Err(r) => return fail(&ui, &r),
    };
    STATUS.ready(); // (up: from here a failure is the engine's to say, and nothing is started again)
    if piped {
        let mut t = String::new();
        let _ = std::io::stdin().read_to_string(&mut t);
        if t.trim().is_empty() {
            return fail(&ui, &no_prompt());
        }
        prompt = Some(t);
    }
    if let Some(p) = prompt {
        let defaults = ready.defaults.clone();
        return once(&mut Talk { e: &mut ready.engine, tok: &ready.tok, chat: ready.chat.as_ref() }, &defaults, p.trim(), thinking, &ui);
    }
    let window = ready.engine.capacity();
    let (tx, rx) = mpsc::channel();
    let addr = match listen(a, &ready.name, window, tx.clone()) {
        Ok(x) => Some(x),
        Err(r) => {
            ui.borrow_mut().note(&r.to_string());
            None
        }
    };
    match addr {
        Some(addr) => {
            access_notes(&ui, a);
            ui.borrow_mut().line(&format!("Chat here, or open http://{addr} in a browser. /bye leaves, /clear starts again, /think turns thinking on or off."));
        }
        None => ui.borrow_mut().line("Chat here. /bye leaves, /clear starts again, /think turns thinking on or off."),
    }
    let defaults = ready.defaults.clone();
    let chat_tx = tx.clone();
    let tty = std::io::stdout().is_terminal();
    let t = std::thread::spawn(move || chat_loop(chat_tx, defaults, thinking, tty));
    let Ready { mut engine, tok, chat, defaults, name, users, .. } = ready;
    let vocab = engine.w.cfg.vocab;
    server::work(&mut engine, Arc::new(server::Shared { vocab, tok, chat, name, defaults, window, thinking: None, refused: &[] }), &rx, &tx, users);
    let _ = t.join();
    0
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
                let why = if o.finish == crate::model::Finish::Length && thought {
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
            ui.borrow_mut().refuse(&Refusal::new(e.0, ""));
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
                        if o.finish == crate::model::Finish::Length {
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

// ---- the lossy levels: a Qwen3.5-family or Nemotron 3 Super model at about 6.5 bits per weight (:kestrel, which --lossy is: six-bit block codes), 5.5 (:swift: five-bit) or 4.5 (:wren: four-bit; Nemotron's experts alone, the rest six-bit), one user at a time ------

// (the lossy paths' work, context bounds, host margin and Gemma's draft and chunk are src/plan.rs's constants, with the one reserve)

/// The Gemma 4 draft model ("-assistant") to draft with: `--draft DIR`, else the model directory's `draft/` where it has a config.json; none with `--no-draft`.
fn gemma_draft(a: &Args, dir: Option<&Path>) -> Option<PathBuf> {
    if a.has("no-draft") {
        return None;
    }
    a.flag("draft").map(PathBuf::from).or_else(|| dir.map(|d| d.join("draft")).filter(|d| d.join("config.json").exists()))
}

/// `--expert-cache GIB`: the GPU memory to give the places of the expert cache, in GiB.
fn cache_flag(a: &Args) -> Done<Option<u64>> {
    let Some(v) = a.flag("expert-cache") else { return Ok(None) };
    match v.parse::<f64>() {
        Ok(g) if g.is_finite() && g > 0.0 => Ok(Some((g * (1u64 << 30) as f64) as u64)),
        _ => Err(Refusal::new(format!("--expert-cache {v}: not a number of GiB"), "Give the GPU memory for the experts, in GiB: --expert-cache 12")),
    }
}

/// `--host-memory GIB|auto`: the computer's memory the experts may take, in GiB; None for `auto` (and where it is not given): what the computer has free less a reserve (`HOST_MARGIN`), at most all the experts. Less than all
/// of them, the Hummingbird level keeps the rest in its packed directory and reads them as the router asks for them (src/offload.rs `Tier`).
pub fn host_flag(a: &Args) -> Done<Option<u64>> {
    match a.flag("host-memory") {
        None | Some("auto") => Ok(None),
        Some(v) => match v.parse::<f64>() {
            Ok(g) if g.is_finite() && g > 0.0 => Ok(Some((g * (1u64 << 30) as f64) as u64)),
            _ => Err(Refusal::new(
                format!("--host-memory {v}: not a number of GiB"),
                "Give the computer's memory the experts may take, in GiB: --host-memory 20 (auto: what the computer has free, less 6 GiB for the system)",
            )),
        },
    }
}

/// The bytes of `experts_bytes` of experts the computer's memory holds (`CachePlan::ram`): what `asked` is (`--host-memory`), else what the computer has free (`host`; None: it does not say, and is taken to have enough)
/// less the reserve; at most all of them.
pub fn held_in_memory(experts_bytes: u64, asked: Option<u64>, host: Option<u64>) -> u64 {
    match (asked, host) {
        (Some(b), _) => b.min(experts_bytes),
        (None, Some(h)) => h.saturating_sub(HOST_MARGIN).min(experts_bytes),
        (None, None) => experts_bytes,
    }
}

/// The expert cache a Nemotron model in the glyd-v7 directory `dir` (at `layout`) gets on a GPU of `free` bytes free, in a computer with `host` bytes of memory to give: what `run` gives it (`lossy_plan_with` on the
/// directory's own config and packs); None where the GPU holds the model whole. For `glyd-engine generate --expert-cache auto`.
pub fn expert_cache_for(dir: &Path, layout: Layout, free: u64, gpu: &str, host: Option<u64>) -> Done<Option<CachePlan>> {
    let file = |f: &str| std::fs::read_to_string(dir.join(f));
    let text = file("config.json").map_err(|e| Refusal::new(format!("{}: {e}", dir.join("config.json").display()), ""))?;
    let (generation, name) = (file("generation_config.json").ok(), dir.to_string_lossy().into_owned());
    let packed = dir.join("glyd.json").exists().then_some(dir);
    Ok(lossy_plan_with(&Args::from(["run".to_string(), name.clone()]), &name, layout, &text, generation.as_deref(), free, gpu, packed, host)?.experts)
}

/// Nemotron 3 Super with its experts in the computer's memory (pinned) and a cache of them on the GPU: the context, and the cache, which takes what the GPU has left besides the rest of the model, the states, the
/// workspaces and the context's keys and values (or `asked` bytes of it); why not, in words, where the GPU cannot hold even that rest and the least cache (a step's choices and as many again; the choices alone for
/// an asked cache), or the computer cannot hold the experts. At the Hummingbird level a computer that cannot hold them all (or `--host-memory`, `asked_host`, that gives it less) keeps the rest in the packed directory:
/// `CachePlan::ram` is what it holds, refused where that is less than a step's choices in each width of each layer.
#[allow(clippy::too_many_arguments)]
fn offloaded(
    cfg: &ConfigNemotronH,
    layout: Layout,
    weights: u64,
    state: u64,
    per: u64,
    free: u64,
    host: Option<u64>,
    ctx: Option<usize>,
    asked: Option<u64>,
    asked_host: Option<u64>,
    gpu: &str,
) -> Result<(usize, CachePlan), String> {
    let (layers, experts, k) = (cfg.layers_of(NhKind::Moe), cfg.moe.experts, cfg.moe.top_k);
    let each = crate::moe_nh::expert_bytes(cfg, layout).map_err(|e| e.0)?;
    let experts_bytes = (layers * experts) as u64 * each;
    let ctx = ctx.unwrap_or(LOSSY_CTX).min(cfg.max_pos.max(LOSSY_MIN_CTX));
    // (the plan's weights are at fit::bits a weight, a little over the experts' exact bytes: the rest of the model is taken as what is left)
    let rest = weights.saturating_sub(experts_bytes) + state + LOSSY_WORK + reserve_of(free) + ctx as u64 * per.max(1);
    let least = if asked.is_some() { k } else { 2 * k };
    let need = rest + (layers * least) as u64 * each;
    let budget = asked.unwrap_or_else(|| free.saturating_sub(rest));
    if asked.is_some_and(|b| rest.saturating_add(b) > free) {
        return Err(format!(
            "a cache of {} for the experts and the {} the rest of the model takes are more than the {gpu}'s {} free",
            gib(budget),
            gib(rest),
            gib(free)
        ));
    }
    let host_need = experts_bytes + HOST_MARGIN;
    // (the Hummingbird level's experts can stay in the packed directory: it needs only the memory for what it holds, and a step's choices in each of a layer's two widths)
    let disk = layout == Layout::Hummingbird;
    let ram = held_in_memory(experts_bytes, asked_host, host);
    let least_ram = if disk { (layers * 2 * k) as u64 * each } else { experts_bytes };
    let Some(plan) = crate::moe_nh::cache_plan(cfg, layout, budget, least).map_err(|e| e.0)?.filter(|p| p.gpu_bytes() + rest <= free) else {
        return Err(format!(
            "with its experts in the computer's memory it needs at least {} of GPU memory (and {} of the computer's)",
            gib(need),
            gib(if disk { least_ram + HOST_MARGIN } else { host_need })
        ));
    };
    if !disk {
        if let Some(h) = host.filter(|&h| h < host_need) {
            return Err(format!("with its experts in the computer's memory it needs {} of the computer's memory besides at least {} of GPU memory, and this computer has {} free", gib(host_need), gib(need), gib(h)));
        }
        return Ok((ctx, plan));
    }
    if let Some(h) = host.filter(|&h| asked_host.is_some() && ram.saturating_add(HOST_MARGIN) > h) {
        return Err(format!("--host-memory {}: the computer has {} free and keeps {} of them for the system", gib(ram), gib(h), gib(HOST_MARGIN)));
    }
    if ram < least_ram {
        return Err(format!("with its experts read from the disk it needs at least {} of the computer's memory for them ({} for a step's choices in each width of each layer), and {} are to be given", gib(least_ram + HOST_MARGIN), gib(least_ram), gib(ram)));
    }
    Ok((ctx, CachePlan { ram, ..plan }))
}

/// The model's config as a lossy level reads it (Qwen3.5, Qwen3.6 and Qwen3.8 dense: their config says `qwen3_5`; Nemotron 3 Super: `nemotron_h`, at the kestrel, swift and wren levels), and what it takes in `layout` of
/// `free` bytes of GPU memory at the context asked for (else up to LOSSY_CTX, as much as fits): refused where it does not fit at LOSSY_MIN_CTX. A pack is sized in its own layout (`pack_layout`: Nemotron's experts
/// alone are four-bit at wren). `packed`: the glyd-v7 directory the model is, where its packs' bytes are the ones it holds.
pub(crate) fn lossy_plan(a: &Args, model: &str, layout: Layout, text: &str, gen: Option<&str>, free: u64, gpu: &str, packed: Option<&Path>) -> Done<LoadPlan> {
    lossy_plan_with(a, model, layout, text, gen, free, gpu, packed, crate::offload::host_memory())
}

/// `lossy_plan` for a computer that has `host` bytes of memory to give (None: it does not say). A Nemotron model that does not fit the GPU whole, or that `--expert-cache GIB` asks for, keeps its experts in the
/// computer's memory and a cache of them on the GPU (`offloaded`); it is refused, with the memory it needs of each, where the GPU or the computer has too little for that.
#[allow(clippy::too_many_arguments)]
pub(crate) fn lossy_plan_with(
    a: &Args,
    model: &str,
    layout: Layout,
    text: &str,
    gen: Option<&str>,
    free: u64,
    gpu: &str,
    packed: Option<&Path>,
    host: Option<u64>,
) -> Done<LoadPlan> {
    // (what the last load of this model here took beyond its plan is planned as already taken: `calibrated`)
    let extra = calibration(&short_name(model), level_name(layout), gpu);
    let mut p = lossy_plan_inner(a, model, layout, text, gen, free.saturating_sub(extra), gpu, packed, host)?;
    calibrated(&mut p, free, extra);
    Ok(p)
}

#[allow(clippy::too_many_arguments)]
fn lossy_plan_inner(
    a: &Args,
    model: &str,
    layout: Layout,
    text: &str,
    gen: Option<&str>,
    free: u64,
    gpu: &str,
    packed: Option<&Path>,
    host: Option<u64>,
) -> Done<LoadPlan> {
    let name = short_name(model);
    let tag = level_name(layout);
    if crate::model_gemma4::is_gemma4(text) {
        return gemma_plan(a, &name, tag, layout, text, gen, free, gpu, packed);
    }
    // the matrices and the tensors kept as they are, what a sequence holds (its states, a token's KV) and the longest context
    let nemotron = crate::nemotron::is_nemotron_h(text);
    let (packs, kept, state, per, max_pos) = if nemotron {
        if !matches!(layout, Layout::Int6 | Layout::Int5 | Layout::Int4 | Layout::Hummingbird) {
            return Err(Refusal::unsupported(format!("{name}: the {tag} level is not built for Nemotron 3 Super, whose experts are held at the kestrel, swift and wren levels, or at the hummingbird level"), format!("Run it at the wren, swift or kestrel level: glyd run {model}:wren")));
        }
        let cfg = ConfigNemotronH::parse(text, "config.json", gen).map_err(|e| Refusal::unsupported(format!("{name}: {}", e.0), unrun_fix()))?;
        (cfg.packs(), cfg.kept(), cfg.state_bytes(), cfg.kv_bytes_per_token(), cfg.max_pos)
    } else {
        if layout == Layout::Hummingbird {
            return Err(Refusal::unsupported(
                format!("{name}: the hummingbird level is built for Nemotron 3 Super alone"),
                "Run it at the kestrel, swift or wren level: glyd run MODEL:swift",
            ));
        }
        let cfg = Config35::parse(text, "config.json", gen).map_err(|e| {
            Refusal::unsupported(
                format!("{name}: the {tag} level runs {} models so far ({})", "Qwen3.5, Qwen3.6, Qwen3.8 and Nemotron 3 Super", e.0),
                format!("Run it lossless: glyd run {model}"),
            )
        })?;
        let cfg = match packed {
            Some(d) => cfg.as_saved(d),
            None => cfg,
        };
        (cfg.packs(), cfg.kept(), cfg.state_bytes(), cfg.kv_bytes_per_token(), cfg.max_pos)
    };
    let ck = packed.map(|d| glyd_gpu::safetensors::Checkpoint::open(d).map_err(cannot)).transpose()?;
    let (mut weights, mut bf16) = (0u64, 0u64);
    for p in packs {
        let b = (p.rows() * p.cols * 2) as u64;
        bf16 += b;
        // (a matrix the directory holds as bf16 is packed as it loads: sized as without one, in the layout it is packed in)
        let saved = ck.as_ref().map_or(0, |ck| p.parts.iter().filter_map(|x| packed_bytes(ck, Pack35::saved(&x.0))).sum::<u64>());
        let own = if nemotron { pack_layout(layout, &p.key) } else { layout };
        weights += if saved > 0 {
            saved
        } else if p.packable() {
            (b as f64 * fit::bits(own) / 16.0) as u64
        } else {
            b
        };
    }
    for t in kept {
        let b = t.shape.iter().product::<u64>() * if t.dtype == "F32" { 4 } else { 2 };
        bf16 += b;
        weights += b;
    }
    let reserve = reserve_of(free);
    let fixed = weights + state + LOSSY_WORK + reserve;
    let per = per.max(1);
    let family = if nemotron { "Nemotron-H" } else { "Qwen3.5" };
    let kind = if nemotron { Kind::NemotronH } else { Kind::Qwen35 };
    let mk = |ctx: usize, cache: Option<CachePlan>| LoadPlan {
        kind,
        family,
        level: level_name(layout),
        layout,
        context: ctx,
        users: 1,
        chunk: 2048,
        weights,
        bf16,
        kv: ctx as u64 * per + state,
        work: LOSSY_WORK,
        draft: 0,
        reserve,
        free,
        gpu: gpu.to_string(),
        experts: cache,
        spread: None,
    };
    let room = (free.saturating_sub(fixed) / per) as usize;
    let asked = if a.has("context") || a.has("ctx") { Some(a.either("context", "ctx", 0usize)?) } else { None };
    let ctx = asked.unwrap_or_else(|| room.min(LOSSY_CTX)).min(max_pos.max(LOSSY_MIN_CTX));
    let fits = !(ctx < LOSSY_MIN_CTX.min(asked.unwrap_or(LOSSY_MIN_CTX)) || ctx > room);
    let cache = cache_flag(a)?;
    if cache.is_some() && !nemotron {
        return Err(Refusal::new(format!("--expert-cache: {name} has no experts to keep in the computer's memory"), "It is for Nemotron 3 Super"));
    }
    let host_asked = host_flag(a)?;
    if host_asked.is_some() && !nemotron {
        return Err(Refusal::new(format!("--host-memory: {name} has no experts to keep in the computer's memory"), "It is for Nemotron 3 Super"));
    }
    let mut also = String::new();
    if nemotron && (!fits || cache.is_some()) {
        let cfg = ConfigNemotronH::parse(text, "config.json", gen).map_err(|e| Refusal::unsupported(format!("{name}: {}", e.0), unrun_fix()))?;
        match offloaded(&cfg, layout, weights, state, per, free, host, asked, cache, host_asked, gpu) {
            Ok((ctx, plan)) => return Ok(mk(ctx, Some(plan))),
            Err(why) if cache.is_some() => {
                return Err(Refusal::unsupported(
                    format!("{name} at the {tag} level with --expert-cache: {why}"),
                    "Close programs that use the GPU (nvidia-smi lists them), ask for a smaller cache, or use a GPU or a computer with more memory",
                ))
            }
            Err(why) => also = format!("; {why}"),
        }
    }
    if !fits {
        return Err(Refusal::unsupported(
            format!(
                "{name} at the {tag} level takes about {} of GPU memory at {} tokens of context, and the {gpu} has {} free{also}",
                gib(fixed + ctx.max(LOSSY_MIN_CTX) as u64 * per),
                ctx.max(LOSSY_MIN_CTX),
                gib(free)
            ),
            if also.is_empty() {
                "Close programs that use the GPU (nvidia-smi lists them), ask for less context (--context 4096), or use a GPU with more memory"
            } else {
                "Close programs that use the GPU (nvidia-smi lists them) or the computer's memory, or use a GPU or a computer with more memory"
            },
        ));
    }
    Ok(mk(ctx, None))
}

/// `lossy_plan` for a Gemma 4 dense model: its packed weights (as a glyd-v7 directory holds them, else at `layout`), the sliding layers' rings, a draft model where one is found, and the
/// context the rest of the GPU holds in the global layers' keys and values.
#[allow(clippy::too_many_arguments)]
fn gemma_plan(a: &Args, name: &str, tag: &str, layout: Layout, text: &str, gen: Option<&str>, free: u64, gpu: &str, packed: Option<&Path>) -> Done<LoadPlan> {
    if !matches!(layout, Layout::Tiered | Layout::Int6 | Layout::Int5 | Layout::Int4 | Layout::Int4H) {
        return Err(Refusal::unsupported(
            format!("{name}: the {tag} level is not built for Gemma 4"),
            format!("Run it lossless (penguin) or at the kestrel or swift level: glyd run {name}:kestrel"),
        ));
    }
    let cfg = crate::model_gemma4::ConfigG4::parse(text, gen, false).map_err(|e| Refusal::unsupported(format!("{name}: {}", e.0), ""))?;
    let ck = packed.map(|d| glyd_gpu::safetensors::Checkpoint::open(d).map_err(cannot)).transpose()?;
    let (weights, bf16) = cfg.weight_bytes(layout, ck.as_ref());
    // (a model not on this computer yet comes as its pre-packed repo, which holds its draft model: planned with it, the larger of the two plans)
    let typed = a.pos.get(1).map_or(name, |m| level(m, false).map_or(m.as_str(), |(m, _)| m));
    let draft_to_come = packed.is_none() && !a.has("no-draft") && short_name(typed).to_ascii_lowercase().starts_with("gemma-4-"); // (ship's `draft_repo`: Google's "-assistant" model of each Gemma 4)
    let draft = if gemma_draft(a, packed).is_some() || draft_to_come { GEMMA_DRAFT_BYTES } else { 0 };
    let asked = if a.has("context") || a.has("ctx") { Some(a.either("context", "ctx", 0usize)?) } else { None };
    let (rings, per) = cfg.kv_bytes(GEMMA_CHUNK, asked.unwrap_or(LOSSY_CTX).max(LOSSY_MIN_CTX));
    let reserve = reserve_of(free);
    let fixed = weights + rings + draft + LOSSY_WORK + reserve;
    let room = (free.saturating_sub(fixed) / per.max(1)) as usize;
    // (served with no draft model, the scheduler gives each of --users a room of its own, its sliding layers' rings and its context (src/dynamic/rows.rs): the context not asked for is what the room gives
    // each of them, at least LOSSY_MIN_CTX (where that leaves room for fewer, the server takes fewer))
    let users = if draft == 0 && a.pos.first().is_some_and(|c| c == "serve") { users_of(a)? as u64 } else { 1 };
    let each = (free.saturating_sub(fixed + rings * (users - 1)) / (per.max(1) * users)) as usize;
    let ctx = asked.unwrap_or_else(|| each.max(LOSSY_MIN_CTX).min(room).min(LOSSY_CTX)).min(cfg.max_pos.max(LOSSY_MIN_CTX)) / 16 * 16;
    if ctx < LOSSY_MIN_CTX.min(asked.unwrap_or(LOSSY_MIN_CTX)) || ctx > room {
        let fix = "lose programs that use the GPU (nvidia-smi lists them), ask for less context (--context 4096), run without the draft model (--no-draft), or use a GPU with more memory";
        // (penguin that does not fit, of a model with a wren repo that does: that repo first, with its draft model, at the shortest context a plan takes)
        let wren = gemma_wren(typed).filter(|_| layout == Layout::Tiered).and_then(|repo| {
            let need = cfg.weight_bytes(Layout::Int4H, None).0 + rings + GEMMA_DRAFT_BYTES + LOSSY_WORK + reserve + LOSSY_MIN_CTX as u64 * per;
            (need <= free).then(|| {
                format!(
                    "glyd run {repo} fits: the wren level, about {} at {LOSSY_MIN_CTX} tokens of context (Google's QAT checkpoint, bit for bit)\nOr c{fix}",
                    gib(need)
                )
            })
        });
        return Err(Refusal::unsupported(
            format!(
                "{name} at the {tag} level takes about {} of GPU memory at {} tokens of context, and the {gpu} has {} free",
                gib(fixed + ctx.max(LOSSY_MIN_CTX) as u64 * per),
                ctx.max(LOSSY_MIN_CTX),
                gib(free)
            ),
            wren.unwrap_or_else(|| format!("C{fix}")),
        ));
    }
    Ok(LoadPlan {
        kind: Kind::Gemma4,
        family: "Gemma 4",
        level: level_name(layout),
        layout,
        context: ctx,
        users: users as usize,
        chunk: GEMMA_CHUNK,
        weights,
        bf16,
        kv: users * (rings + ctx as u64 * per),
        work: LOSSY_WORK,
        draft,
        reserve,
        free,
        gpu: gpu.to_string(),
        experts: None,
        spread: None,
    })
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

/// The lossy level's engine as `run` loads it (Qwen3.5's or Nemotron 3 Super's), with its tokenizer, chat template and sampling defaults, the size of its vocabulary, the line that says it is ready, and what
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
        Some("Nemotron-H") if layout != Layout::Tiered => {}
        Some(family) => return load_dynamic(a, &model, family, layout, served, ui),
        None => {}
    }
    let mut layout = layout_for(&model, layout);
    let name = short_name(&model);
    let mut packed = v7_layout(&model)?.map(|_| PathBuf::from(&model));
    let gpu = gpu_pick()?;
    if let Some(r) = machine::gpu_refusal(&gpu, std::env::consts::ARCH, true, machine::ANY_FAMILY) {
        return Err(r);
    }
    let first = match packed {
        None => Some(lossy_plan(a, &model, layout, &config_text(&model)?, None, gpu.free, &gpu.name, None)?),
        Some(_) => None,
    };
    if packed.is_none() {
        packed = pre_packed(&model, layout, ui)?;
    }
    let mut dir = match &packed {
        Some(d) => d.clone(),
        None if Path::new(&model).join("config.json").exists() => PathBuf::from(&model),
        None => match Source::find(&model) {
            Ok(s) => s.dir,
            Err(_) => {
                let (hub, remote, wanted) = store::hub_plan(&model, ui)?;
                store::download(&hub, &remote, &wanted, &model, ui)?;
                Source::find(&model).map_err(cannot)?.dir
            }
        },
    };
    // (a pre-packed repo given by its own name is a glyd-v7 directory once downloaded; a bf16 model is packed into the cache once, and read from there)
    if packed.is_none() {
        packed = v7_manifest(&dir).map_err(cannot)?.map(|_| dir.clone());
    }
    if let Some(d) = &packed {
        // (the layout as packed: wren of a QAT checkpoint is the exact four-bit one)
        layout = v7_level(&model, v7_manifest(d).map_err(cannot)?.map(|m| m.0), Some(layout))?.unwrap_or(layout);
    } else if let Some(first) = first {
        if layout == Layout::Hummingbird {
            return Err(Refusal::unsupported(
                format!("{name} at the hummingbird level comes as a finished download or directory only: its experts' codes are made apart"),
                format!("Download the pre-packed model: glyd/{name}-hummingbird"),
            ));
        }
        dir = store::pack(&Source::find(&model).map_err(cannot)?, &model, layout, first.weights, ui)?;
        packed = Some(dir.clone());
    }
    let read = |f: &str| std::fs::read_to_string(dir.join(f)).ok();
    let plan = lossy_plan(
        a,
        &model,
        layout,
        &read("config.json").ok_or_else(|| cannot(format!("{}: no config.json", dir.display())))?,
        read("generation_config.json").as_deref(),
        gpu.free,
        &gpu.name,
        packed.as_deref(),
    )?;
    // (a Nemotron-H model's layouts differ by pack: the bits a weight are what its bytes come to)
    let bits = if plan.kind == Kind::NemotronH { plan.weights as f64 * 16.0 / plan.bf16 as f64 } else { fit::bits(layout) };
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
    ui.borrow_mut().status(&format!("{} {name} and loading it onto the GPU...", if packed.is_some() { "Reading" } else { "Packing" }));
    let oom = |e: crate::ffi::Error| {
        if e.0.to_lowercase().contains("out of memory") {
            Refusal::unsupported(
                format!("the GPU ran out of memory loading {name} ({} were free at the start)", gib(gpu.free)),
                "Close programs that use the GPU (nvidia-smi lists them), or ask for less context: --context 4096",
            )
        } else {
            cannot(e.0)
        }
    };
    let (e, bytes, vocab): (Box<dyn Gen>, usize, usize) = if plan.kind == Kind::Gemma4 {
        // (drafted where a draft model is found: the big model checks every draft token, so the tokens are its own; greedy replies draft, sampled ones step)
        let draft = gemma_draft(a, Some(&dir)).map(|d| (d, a.num("draft-tokens", 8usize).unwrap_or(8).clamp(1, GEMMA_CHUNK - 1)));
        if let Some((d, k)) = &draft {
            ui.borrow_mut().line(&format!("Drafting with {} ({k} tokens a round; --no-draft runs without it).", d.display()));
        }
        if crate::tune!("GLYD_G4_DYNAMIC").is_some_and(|v| v == "0") {
            // (a development build's way back to the hand-written engine, to compare)
            let o = crate::model_gemma4::OptsG4 { ctx: plan.context, chunk: GEMMA_CHUNK, graph: true, device: gpu.ordinal, layout: Some(layout), draft };
            let e = crate::model_gemma4::EngineG4::new(&dir, &o).map_err(oom)?;
            let (bytes, vocab) = (e.w.bytes, e.w.cfg.vocab);
            (Box::new(e), bytes, vocab)
        } else {
            // the dynamic engine (GLY-326): Gemma 4's description and, where one is found, its draft model's
            let c = crate::model_gemma4::ConfigG4::load(&dir).map_err(cannot)?;
            let spec = match &draft {
                Some((d, k)) => {
                    let dc = crate::model_gemma4::ConfigG4::load_for(d, true).map_err(cannot)?;
                    Some(crate::dynamic::DraftSpec { dir: d, desc: crate::dynamic::gemma4::describe_draft(&dc, &c), k: *k })
                }
                None => None,
            };
            let o = crate::dynamic::Opts { ctx: plan.context, chunk: GEMMA_CHUNK, graph: true, device: gpu.ordinal, layout: Some(layout) };
            let e = crate::dynamic::Dyn::new(&dir, crate::dynamic::gemma4::describe(&c), spec, &o).map_err(oom)?;
            let (bytes, vocab) = (e.weights_bytes(), e.desc().vocab);
            (Box::new(e), bytes, vocab)
        }
    } else if plan.kind == Kind::NemotronH {
        let o = OptsNh { ctx: plan.context, chunk: 2048, graph: true, device: gpu.ordinal, layout: Some(layout), chunked: true, only: None };
        let store = ExpertStore::new();
        // (the experts the computer's memory does not hold stay in the packed directory: src/offload.rs `Tier`)
        if let Some(c) = plan.experts.filter(|c| c.ram < c.host_bytes()) {
            store.set_host_memory(Some(crate::offload::HostMemory { bytes: c.ram, total: c.host_bytes() }));
        }
        // an expert that is not on the GPU is computed by the CPU (--cpu-threads) where the computer's memory holds a second copy of the experts' first matrices besides the experts; else it is copied
        let mut offload = crate::offload::offload_args(a).map_err(|e| Refusal::new(e.0, "--cpu-threads takes a number of threads: --cpu-threads 8"))?;
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
        let (plain, cached) = (crate::moe_nh::factory(layout), plan.experts.map(|c| crate::moe_nh::factory_cached(layout, c.slots, &store)));
        let make: crate::model_nh::MoeFactory<'_> = match &cached {
            Some(f) => f,
            None => &plain,
        };
        if crate::tune!("GLYD_NH_DYNAMIC").is_some_and(|v| v == "0") {
            // (a development build's way back to the hand-written engine, to compare)
            let e = EngineNh::new(&dir, &o, Some(make)).map_err(oom)?;
            let (bytes, vocab) = (e.w.bytes, e.w.cfg.vocab);
            (Box::new(e), bytes, vocab)
        } else {
            // the dynamic engine (GLY-354): Nemotron-H's description, its blocks of experts (an expert cache's) as for the hand-written one
            let layers = crate::nemotron::ConfigNemotronH::load(&dir).map_err(cannot)?.layers;
            let o = crate::dynamic::Opts { ctx: o.ctx, chunk: o.chunk, graph: o.graph, device: o.device, layout: o.layout };
            let e = crate::dynamic::nemotron::load_with(&dir, &o, make, crate::dynamic::Stage::whole(layers)).map_err(oom)?;
            let (bytes, vocab) = (e.weights_bytes(), e.desc().vocab);
            (Box::new(e), bytes, vocab)
        }
    } else {
        // the dynamic engine (GLY-326): Qwen3.5's description
        let o = crate::dynamic::Opts { ctx: plan.context, chunk: 2048, graph: true, device: gpu.ordinal, layout: Some(layout) };
        let e = crate::dynamic::qwen35::load(&dir, &o).map_err(oom)?;
        let (bytes, vocab) = (e.weights_bytes(), e.desc().vocab);
        (Box::new(e), bytes, vocab)
    };
    loaded_report(&model, &plan, &[&gpu], bytes as u64);
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
        about.push((
            "costs",
            crate::costs::served(
                &dir,
                served,
                requested(served, a.has("lossy")),
                e.capacity(),
                if plan.kind == Kind::Gemma4 { GEMMA_CHUNK } else { 2048 },
                held,
            ),
        ));
        server::Info { about, gpu: Some(gpu.ordinal), held, concurrency: 1, window: e.capacity(), refused: server::ONE_AT_A_TIME, started }
    });
    Ok((e, tok, chat, defaults, vocab, line, info))
}

/// The GPUs the dynamic route uses (Ampere or newer, as the driver lists them) and the first, which the machine's checks are of.
fn dynamic_gpus() -> Done<(Vec<Gpu>, Gpu)> {
    let gpus: Vec<Gpu> = Dev::probe_all()
        .map_err(|e| Refusal::unsupported(format!("the GPU cannot be used: {}", e.0), "Check that nvidia-smi shows it"))?
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

/// Kimi-K3's expert cache over these GPUs (`split::places_plan`): the cuts, their GPUs, each stage's places a layer, the bytes on the GPUs, the experts' bytes in the computer's
/// memory, the experts a layer has; refused with the numbers where the computer's memory does not hold every expert or a GPU has room for fewer than a token's experts.
fn kimi_cache(
    dir: &Path,
    nl: usize,
    room: &[(i32, u64)],
    name: &str,
    ctx: usize,
    twelve: bool,
    host: Option<u64>,
) -> Done<(Vec<usize>, Vec<i32>, Vec<usize>, u64, u64, usize)> {
    let c = crate::kimi::ConfigKimi::load(dir).map_err(cannot)?;
    // (its bf16 matrices in the 12-bit layout where `twelve`: an HBM GPU's faster product, a few hundred places fewer)
    let bf16 = if twelve { crate::dynamic::split::TWELVE_PCT } else { crate::dynamic::split::TIERED_PCT };
    let (mut dense, base) =
        crate::dynamic::split::layer_bytes_where(dir, nl, crate::dynamic::split::CODED_PCT, bf16, &|n| !n.contains(".block_sparse_moe.experts."))
            .map_err(cannot)?;
    // (each layer's cache for the context too: a latent attention's latents, ctx of kv_rank + rope bf16; a delta attention's state, heads 128 x 128 fp32, and its convolution's tail)
    let qd = c.kda.heads * c.kda.dim;
    for (l, d) in dense.iter_mut().enumerate() {
        *d += match c.kinds.get(l) {
            Some(crate::glm::GlmKind::Mla) => (ctx * (c.mla.kv_rank + c.mla.rope) * 2) as u64,
            _ => (c.kda.heads * c.kda.dim * c.kda.dim * 4 + 3 * 3 * qd * 2) as u64,
        };
    }
    let expert = crate::dynamic::kimi::expert_bytes(&c).map_err(cannot)?;
    let e = &c.experts;
    let layers = c.moe.iter().filter(|&&m| m).count();
    let all = (layers * e.experts) as u64 * expert;
    let host = host.unwrap_or(0);
    if all + HOST_MARGIN > host {
        return Err(Refusal::unsupported(
            format!(
                "{name} keeps the {} experts the GPUs do not hold in the computer's memory: they take {}, and {} is free",
                layers * e.experts,
                gib(all),
                gib(host)
            ),
            "A machine with that much memory runs it",
        ));
    }
    let Some((cuts, devices, places)) = crate::dynamic::split::places_plan(&dense, base, &c.moe, room, expert, e.experts, e.top_k) else {
        return Err(Refusal::unsupported(
            format!(
                "{name}: its weights besides the experts take {}, and the GPUs here leave too little room for its experts",
                gib(dense.iter().sum::<u64>() + base)
            ),
            "A machine whose GPUs have more memory free runs it",
        ));
    };
    let bounds: Vec<usize> = std::iter::once(0).chain(cuts.iter().copied()).chain(std::iter::once(nl)).collect();
    let on_gpus = dense.iter().sum::<u64>()
        + base
        + bounds.windows(2).zip(&places).map(|(w, &p)| c.moe[w[0]..w[1]].iter().filter(|&&m| m).count() as u64 * p as u64 * expert).sum::<u64>();
    Ok((cuts, devices, places, on_gpus, all, e.experts))
}

/// `kimi_cache` with every layer across all the GPUs of `room` (src/dynamic/tp.rs): each GPU holds its slice of the rows of each big matrix (the embedding
/// whole, and every matrix too small or of rows that do not divide), the caches for the context whole, and the same places a layer, each a slice of an
/// expert's; the least room of the GPUs sets the places.
fn kimi_cache_tp(
    dir: &Path,
    room: &[(i32, u64)],
    name: &str,
    ctx: usize,
    twelve: bool,
    host: Option<u64>,
) -> Done<(Vec<usize>, Vec<i32>, Vec<usize>, u64, u64, usize)> {
    let c = crate::kimi::ConfigKimi::load(dir).map_err(cannot)?;
    let n = room.len().max(1) as u64;
    let bf16 = if twelve { crate::dynamic::split::TWELVE_PCT } else { crate::dynamic::split::TIERED_PCT };
    let ck = glyd_gpu::safetensors::Checkpoint::open(dir).map_err(|e| cannot(format!("{}: {e}", dir.display())))?;
    let mut held = 0u64;
    for f in &ck.files {
        for (t_name, t) in &f.tensors {
            if t_name.contains(".block_sparse_moe.experts.") || ["mtp", "vision", "visual", "audio", "multi_modal"].iter().any(|w| t_name.contains(w)) {
                continue;
            }
            let matrix = t.shape.len() == 2 && t.shape.iter().all(|&d| d >= 256);
            let b = if t.dtype == "BF16" && matrix { t.bytes * bf16 / 100 } else { t.bytes };
            let sliced = matrix && !t_name.contains("embed_tokens") && t.shape[0] % (64 * n) == 0 && t.bytes >= 16 << 20;
            held += if sliced { b / n } else { b };
        }
    }
    let qd = c.kda.heads * c.kda.dim;
    held += c
        .kinds
        .iter()
        .map(|k| match k {
            crate::glm::GlmKind::Mla => (ctx * (c.mla.kv_rank + c.mla.rope) * 2) as u64,
            _ => (c.kda.heads * c.kda.dim * c.kda.dim * 4 + 3 * 3 * qd * 2) as u64,
        })
        .sum::<u64>();
    let e = &c.experts;
    let layers = c.moe.iter().filter(|&&m| m).count() as u64;
    let expert = crate::dynamic::kimi::expert_bytes(&c).map_err(cannot)?;
    let all = layers * e.experts as u64 * expert;
    let host = host.unwrap_or(0);
    if all + HOST_MARGIN > host {
        return Err(Refusal::unsupported(
            format!(
                "{name} keeps the {} experts the GPUs do not hold in the computer's memory: they take {}, and {} is free",
                layers as usize * e.experts,
                gib(all),
                gib(host)
            ),
            "A machine with that much memory runs it",
        ));
    }
    let least = room.iter().map(|r| r.1).min().unwrap_or(0);
    let places = ((least.saturating_sub(held)) / (layers.max(1) * (expert / n)).max(1)).min(e.experts as u64) as usize;
    if places < e.top_k {
        return Err(Refusal::unsupported(
            format!("{name}: its weights besides the experts take {} a GPU across {n} GPUs, and they leave too little room for its experts", gib(held)),
            "A machine whose GPUs have more memory free runs it",
        ));
    }
    let on_gpus = n * (held + layers * places as u64 * (expert / n));
    Ok((Vec::new(), room.iter().map(|r| r.0).collect(), vec![places; room.len()], on_gpus, all, e.experts))
}

/// Each GPU's room for weights on the dynamic route: its free memory less the reserve and the route's work (the KV, the scratch and the graphs together;
// ponytail: that work is a fixed margin, 3 GiB and 5% with the reserve; the description's own numbers where it proves too little or too much)
fn rooms(gpus: &[Gpu]) -> Vec<(i32, u64)> {
    gpus.iter().map(|g| (g.ordinal, crate::plan::dynamic_room_of(g.free))).collect()
}

/// The table's name of a dynamic family (`dynamic_family`'s words are these).
fn static_family(family: &str) -> &'static str {
    ["Kimi-K3", "GLM-5.3-Flash", "DeepSeek-V4", "Muse Spark", "Nemotron-H"].into_iter().find(|f| *f == family).unwrap_or("DeepSeek-V4")
}

/// Whether the GPUs of `devs` are HBM ones (A100, H100, B200: compute capability 8.0, 9.0, 10.0), whose FP8 matrices take layout H (src/f8.rs `layout_h`).
fn hbm_of(devs: &[Target]) -> bool {
    devs.iter().any(|t| t.cc.1 == 0 && (8..=10).contains(&t.cc.0))
}

/// The plan `run` loads a model of the dynamic route (`dynamic_family`) from `dir` at penguin on the GPUs `devs`: the layers cut over them as their room holds
/// them (`--split CUTS --devices GPUS` chooses), or every layer across them (src/dynamic/tp.rs; `reach`: whether they reach each other directly), the FP8
/// matrices coded where only that fits, Kimi-K3's experts kept in the computer's memory (`host` bytes) with places on the GPUs where they do not all fit; or
/// the refusal `run` gives. What `load_dynamic` loads, `check_dynamic` checks and `fit` prints.
#[allow(clippy::too_many_arguments)]
pub(crate) fn size_dynamic(
    a: &Args,
    dir: &Path,
    text: &str,
    family: &'static str,
    devs: &[Target],
    host: Option<u64>,
    reach: &dyn Fn(&[i32]) -> bool,
) -> Done<LoadPlan> {
    let name = short_name(&dir.to_string_lossy());
    let nl = crate::dynamic::split::config_num(text, "num_hidden_layers").unwrap_or(0);
    let room: Vec<(i32, u64)> = devs.iter().map(|t| (t.ordinal, crate::plan::dynamic_room_of(t.free))).collect();
    let hbm = hbm_of(devs);
    let kimi = family == "Kimi-K3";
    let ((per, base), cut, coded, mut twelve) = crate::dynamic::split::fit(dir, nl, &room, hbm, kimi).map_err(cannot)?;
    let mut total = per.iter().sum::<u64>() + base;
    let chosen = a.flag("split").is_some() || a.flag("devices").is_some();
    let ctx = a.num("context", 8192usize).map_err(|e| Refusal::new(e.0, "--context 8192"))?;
    // (every layer across the GPUs, src/dynamic/tp.rs: by default on datacenter GPUs that reach each other directly, where no split is asked for; a
    // development build's GLYD_TP=1 or 0 says so itself, GLYD_TP_GPUS which GPUs)
    let tp_room: Vec<(i32, u64)> = match crate::tune!("GLYD_TP_GPUS") {
        Some(v) => room.iter().copied().filter(|r| v.split(',').any(|x| x.trim().parse::<i32>().ok() == Some(r.0))).collect(),
        None => room.clone(),
    };
    let tp_devs: Vec<i32> = tp_room.iter().map(|r| r.0).collect();
    let across = matches!(family, "Kimi-K3" | "DeepSeek-V4" | "GLM-5.3-Flash" | "Muse Spark")
        && tp_room.len() > 1
        && !chosen
        && match crate::tune!("GLYD_TP").as_deref() {
            Some("1") => true,
            Some(_) => false,
            None => matches!(family, "Kimi-K3" | "DeepSeek-V4") && hbm && reach(&tp_devs),
        };
    // Kimi-K3 where the GPUs do not hold its experts: the expert cache (each GPU's room after the other weights as places for its layers' experts, every
    // expert in the computer's memory)
    let cache = if kimi && cut.is_none() && !chosen {
        Some(if across { kimi_cache_tp(dir, &tp_room, &name, ctx, hbm, host)? } else { kimi_cache(dir, nl, &room, &name, ctx, hbm, host)? })
    } else {
        None
    };
    if let Some(k) = &cache {
        total = k.3;
        twelve = hbm;
    }
    let (cuts, devices) = if let Some(k) = &cache {
        (k.0.clone(), k.1.clone())
    } else if across {
        // (each GPU its slice of the big matrices and the rest whole: about its share of the bytes and the embedding's and head's)
        // ponytail: the share is an estimate; a GPU that runs out of memory says so at the load
        (Vec::new(), tp_devs.clone())
    } else if chosen {
        crate::dynamic::split::split_args(a).map_err(|e| Refusal::new(e.0, "--split 31 --devices 0,1: the layers before 31 on GPU 0, the rest on GPU 1"))?
    } else {
        cut.ok_or_else(|| {
            Refusal::unsupported(
                format!(
                    "{name} needs about {} of GPU memory at penguin, and the {} GPU{} here have {} free",
                    gib(total),
                    devs.len(),
                    if devs.len() == 1 { "" } else { "s" },
                    gib(devs.iter().map(|t| t.free).sum::<u64>())
                ),
                "A machine whose GPUs have that much memory free runs it (nvidia-smi lists what uses them)",
            )
        })?
    };
    let used: Vec<&Target> = devs.iter().filter(|t| devices.contains(&t.ordinal)).collect();
    let (free, work) = (used.iter().map(|t| t.free).sum::<u64>(), used.iter().map(|t| crate::plan::dynamic_work_of(t.free)).sum::<u64>());
    let host_bytes = cache.as_ref().map_or(0, |k| k.4);
    Ok(LoadPlan {
        kind: Kind::Dynamic,
        family,
        level: "penguin",
        layout: if twelve { Layout::Twelve } else { Layout::Tiered },
        context: ctx,
        users: 1,
        chunk: 256,
        weights: total + host_bytes,
        bf16: 0,
        kv: 0,
        work,
        draft: 0,
        reserve: used.len() as u64 * crate::plan::CONTEXT_FLOOR,
        free,
        gpu: used.iter().map(|t| t.name.as_str()).collect::<Vec<_>>().join(", "),
        experts: None,
        spread: Some(crate::plan::Spread {
            cuts,
            devices,
            across,
            coded,
            twelve,
            places: cache.as_ref().map_or(Vec::new(), |k| k.2.clone()),
            experts_per_layer: cache.as_ref().map_or(0, |k| k.5),
            host_bytes,
            gpu_bytes: total,
        }),
    })
}

// ponytail: penguin only on the dynamic route; GLM's and Nemotron-H's other levels when their packs run here
fn penguin_only(name: &str, family: &str, model: &str) -> Refusal {
    let base = model.split_once(':').map_or(model, |m| m.0);
    Refusal::new(format!("{name} ({family}) runs at the penguin level only: the release's weights exactly"), format!("glyd run {base}"))
}

/// `preflight` of a model on the dynamic route: whether the GPUs here hold it (the cut `run` makes, where the model is downloaded; its download's
/// size against all their room, where not) and the disk its download.
fn check_dynamic(a: &Args, model: &str, family: &str, layout: Layout, ui: &RefCell<Ui>) -> Done<String> {
    let name = short_name(model);
    if layout != Layout::Tiered {
        return Err(penguin_only(&name, family, model));
    }
    let (gpus, _) = dynamic_gpus()?;
    crate::ffi::Lib::find().map_err(cannot)?;
    let room = rooms(&gpus);
    let all = room.iter().map(|r| r.1).sum::<u64>();
    let local = if Path::new(model).join("config.json").exists() { Some(PathBuf::from(model)) } else { Source::find(model).ok().map(|s| s.dir) };
    let (bytes, gpus_used) = match &local {
        Some(dir) => {
            let text = std::fs::read_to_string(dir.join("config.json")).map_err(|e| cannot(format!("{}: {e}", dir.display())))?;
            let targets: Vec<Target> = gpus.iter().map(target).collect();
            // (the plan `run` loads, or the refusal it gives, with Kimi-K3's expert cache where the GPUs do not hold its experts)
            let p = size_dynamic(a, dir, &text, static_family(family), &targets, crate::offload::host_memory(), &crate::dynamic::tp::reach)?;
            let sp = p.spread.as_ref().expect("the dynamic route's plan spreads the model");
            if sp.host_bytes > 0 {
                return Ok(format!(
                    "{name} ({family}): about {} on {} GPUs at the penguin level, {} of experts in the computer's memory",
                    gib(p.weights_on_gpu()),
                    sp.devices.len(),
                    gib(sp.host_bytes)
                ));
            }
            (p.weights_on_gpu(), Some(sp.devices.len()))
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
            // (a bf16 release at penguin's 72% at most; an FP8 or FP4 one as it is; the cut is made once it is here)
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
            Ok(format!("{name} ({family}): about {} on {} at the penguin level", gib(bytes), if n == 1 { "one GPU".to_string() } else { format!("{n} GPUs") }))
        }
        None => Err(Refusal::unsupported(
            format!(
                "{name} needs about {} of GPU memory at penguin, and the {} GPU{} here have {} free",
                gib(bytes),
                gpus.len(),
                if gpus.len() == 1 { "" } else { "s" },
                gib(gpus.iter().map(|g| g.free).sum::<u64>())
            ),
            "A machine whose GPUs have that much memory free runs it (nvidia-smi lists what uses them)",
        )),
    }
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
    if layout != Layout::Tiered {
        return Err(penguin_only(&name, family, model));
    }
    let dir = if Path::new(model).join("config.json").exists() {
        PathBuf::from(model)
    } else {
        match Source::find(model) {
            Ok(s) => s.dir,
            Err(_) => {
                let (hub, remote, wanted) = store::hub_plan_as(model, ui, true)?;
                store::download(&hub, &remote, &wanted, model, ui)?;
                Source::find(model).map_err(cannot)?.dir
            }
        }
    };
    let text = std::fs::read_to_string(dir.join("config.json")).map_err(|e| cannot(format!("{}: {e}", dir.display())))?;
    let (gpus, first) = dynamic_gpus()?;
    let nl = crate::dynamic::split::config_num(&text, "num_hidden_layers").unwrap_or(0);
    let targets: Vec<Target> = gpus.iter().map(target).collect();
    let plan = size_dynamic(a, &dir, &text, static_family(family), &targets, crate::offload::host_memory(), &crate::dynamic::tp::reach)?;
    let sp = plan.spread.clone().expect("the dynamic route's plan spreads the model");
    let (total, coded, twelve, across, ctx) = (plan.weights_on_gpu(), sp.coded, sp.twelve, sp.across, plan.context);
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
        "{name} ({family}): about {} on the GPU{} at the penguin level (lossless: the release's weights exactly); {}.",
        gib(total),
        if devices.len() == 1 { "" } else { "s" },
        stages.join(", ")
    ));
    ui.borrow_mut().line(&format!("Plan: {}.", plan.text()));
    if let Some(k) = &cache {
        let (lo, hi) = (k.places.iter().min().copied().unwrap_or(0), k.places.iter().max().copied().unwrap_or(0));
        let n = if lo == hi { lo.to_string() } else { format!("{lo} to {hi}") };
        ui.borrow_mut().line(&format!(
            "{n} experts of {} a layer are on the GPUs, chosen as the router asks for them; all of them are in the computer's memory ({}).",
            k.experts_per_layer,
            gib(k.host_bytes)
        ));
    }
    if coded {
        // (the release's FP8 bytes plain do not fit these GPUs: coded, exactly the same values, in about 7 of 8 bits)
        crate::ffi::Lib::find().and_then(|l| l.f8h_plain(Some(false))).map_err(cannot)?;
        ui.borrow_mut().line("Its FP8 matrices are held coded (the same values in about 7 bits of 8) for it to fit; as they are, they do not.");
    }
    // (penguin's bf16 matrices in its 12-bit layout where it fits an HBM GPU: lossless as the smallest, a faster one-token product there)
    let o = crate::dynamic::Opts { ctx, chunk: 256, graph: true, device: devices[0], layout: Some(if twelve { Layout::Twelve } else { layout }) };
    ui.borrow_mut().status(&format!("Reading {name} and loading it onto the GPU{}...", if devices.len() == 1 { "" } else { "s" }));
    let opened = match &cache {
        Some(k) if across => crate::dynamic::split::open_kimi_tp(&dir, &o, &devices, k.places[0]),
        Some(k) => crate::dynamic::split::open_kimi(&dir, &o, &cuts, &devices, k.places.clone()),
        None if across => crate::dynamic::split::open_tp(&dir, &o, &devices),
        None => crate::dynamic::split::open(&dir, &o, &cuts, &devices),
    };
    let e = opened.map_err(|e| {
        if e.0.to_lowercase().contains("out of memory") {
            Refusal::unsupported(
                format!("a GPU ran out of memory loading {name} ({})", stages.join(", ")),
                "Ask for less context (--context 4096), or cut the layers yourself: --split CUTS --devices GPUS",
            )
        } else {
            cannot(e.0)
        }
    })?;
    // (the weights the route holds are not counted apart: 0)
    loaded_report(model, &plan, &gpus.iter().filter(|g| devices.contains(&g.ordinal)).collect::<Vec<_>>(), 0);
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
        let about = about(served, &dir, requested(served, a.has("lossy")), "penguin", true, "penguin level: the release's weights exactly", held.as_ref(), gpu);
        server::Info { about, gpu: Some(devices[0]), held, concurrency: 1, window: e.capacity(), refused: server::ONE_AT_A_TIME, started }
    });
    let vocab = crate::dynamic::split::config_num(&text, "vocab_size").unwrap_or(0);
    Ok((Box::new(e), tok, chat, defaults, vocab, line, info))
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
    if let Some(p) = prompt {
        return once(&mut Talk { e: &mut *e, tok: &tok, chat: chat.as_ref() }, &defaults, p.trim(), thinking, ui);
    }
    ui.borrow_mut().line("Chat here. /bye leaves, /clear starts again, /think turns thinking on or off.");
    let (tx, rx) = mpsc::channel();
    let tty = std::io::stdout().is_terminal();
    let chat_defaults = defaults.clone();
    let t = std::thread::spawn(move || chat_loop(tx, chat_defaults, thinking, tty));
    work_lossy(&mut *e, &tok, chat.as_ref(), &rx);
    let _ = t.join();
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

/// Whether the engine runs MODEL: exit 0 where it does, 3 (with the reason on stderr) where it does not, 1 where that cannot be told. The kind of model alone: no GPU is looked at (`preflight` looks at the computer).
pub fn supports(a: &Args) -> i32 {
    let ui = RefCell::new(Ui::new());
    let Ok(model) = a.need(1, "MODEL") else { return 1 };
    match config_text(model).and_then(|t| supported(&short_name(model), &t)) {
        Ok(()) if a.has("json") => {
            println!("{}", serde_json::json!({"model": model, "ok": true, "exit": 0, "supported": true}));
            0
        }
        Ok(()) => 0,
        Err(r) if a.has("json") => json_refusal(&ui, model, &r),
        Err(r) => fail(&ui, &r),
    }
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
            Err(e) => fail(&ui, &cannot(e.0)),
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
        Some(m) => run_level(m, layout)?,
        None => layout,
    };
    if let (Some(m), Some(layout)) = (model, layout) {
        return check_lossy(a, m, layout, ui);
    }
    // the kind of model, from its config.json alone
    let cfg = match model {
        Some(m) => {
            let text = config_text(m)?;
            supported(&short_name(m), &text)?;
            Some(Config::parse(&text, "config.json", None).map_err(cannot)?)
        }
        None => None,
    };
    // the GPU a run would use, its driver, and the engine's own files on it
    let gpu = gpu_pick()?;
    let asked = std::env::var("GLYD_ENGINE_ASKED").map_or(false, |v| v.trim() == "1"); // (GLYD_ENGINE=1, passed on by glyd run and glyd serve)
    if let Some(r) = machine::gpu_refusal(&gpu, std::env::consts::ARCH, asked, machine::QWEN3) {
        return Err(r);
    }
    let (Some(model), Some(cfg)) = (model, cfg) else {
        return Ok(format!("{} ({} free)", gpu.name, gib(gpu.free)));
    };
    let name = short_name(model);
    let dir = Path::new(model);
    let packed = dir.join("glyd.json").exists();
    if let (true, Some(why)) = (packed, crate::weights::earlier_format(dir)) {
        return Err(Refusal::unsupported(format!("{model} is {why}"), "Pack the model again: glyd pack MODEL DIR"));
    }
    // the model on this GPU: its memory at the context asked for, its shape, the tables' reach
    let users = a.num("users", 1usize).map_err(|e| Refusal::new(e.0, "Give a number from 1 to 8: --users 4"))?.clamp(1, crate::model::MAX_ROWS);
    let (layout, _, fit) = plan(a, &cfg, &name, &gpu, packed.then(|| packed_layout(dir).unwrap_or(Layout::Tiered)), users)?;
    shape_checks(&name, &cfg, &fit)?;
    crate::ffi::Lib::find().map_err(cannot)?;
    Dev::new(gpu.ordinal).map_err(|e| engine_refusal(e, &name, &fit, &gpu))?;
    let line = format!("{}: {name}, {} tokens of context", gpu.name, fit.context);
    if packed {
        return Ok(line); // (nothing to download or pack)
    }
    // what a first run writes (the Hub says what is still to be downloaded, and refuses a checkpoint that is not bf16 here, before anything is), and a refusal given before
    let source = Source::find(model).ok();
    let download = match &source {
        Some(_) => 0,
        None => {
            let (hub, remote, wanted) = store::hub_plan(model, ui)?;
            hub.missing(&remote, &wanted)
        }
    };
    first_run_checks(&name, &cfg, &fit, layout, source.as_ref(), download)?;
    Ok(line)
}

/// `preflight MODEL:kestrel` (or `:swift`, or `--lossy`): the kind of model, the GPU and its driver (the list of GPUs measured against vLLM is not looked at: vLLM has no lossy levels), the memory, the disk a
/// download takes.
fn check_lossy(a: &Args, model: &str, layout: Layout, ui: &RefCell<Ui>) -> Done<String> {
    match dynamic_family(model) {
        Some("Nemotron-H") if layout != Layout::Tiered => {}
        Some(family) => return check_dynamic(a, model, family, layout, ui),
        None => {}
    }
    let text = config_text(model)?;
    let gpu = gpu_pick()?;
    if let Some(r) = machine::gpu_refusal(&gpu, std::env::consts::ARCH, true, machine::ANY_FAMILY) {
        return Err(r);
    }
    let plan = lossy_plan(a, model, layout, &text, None, gpu.free, &gpu.name, v7_layout(model)?.map(|_| Path::new(model)))?;
    crate::ffi::Lib::find().map_err(cannot)?;
    Dev::new(gpu.ordinal).map_err(|e| cannot(e.0))?;
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

/// MODEL made ready but for the KV tables (downloaded, packed for this GPU), which are the part that needs the GPU's time: what `run` does first.
pub fn prepare_cmd(a: &Args) -> i32 {
    let ui = RefCell::new(Ui::new());
    let r = gpu_pick().and_then(|g| prepare(a, &g, &ui, false, 1));
    match r {
        Ok(p) => {
            println!("{}", p.dir.display());
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
        let dynamic = dynamic_family(rest).filter(|f| *f != "Nemotron-H" || layout.map_or(true, |l| l == Layout::Tiered));
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
        if packed.is_none() && dynamic.is_none() && !crate::model_gemma4::is_gemma4(&text) {
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

    const QWEN3: &str = r#"{"architectures": ["Qwen3ForCausalLM"], "hidden_size": 4096, "num_hidden_layers": 36, "num_attention_heads": 32, "num_key_value_heads": 8, "head_dim": 128, "intermediate_size": 12288,
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
        let own = listen(&args(&["serve", "m", "--port", "0"]), "m", 4096, tx.clone()).map_err(|r| r.to_string()).unwrap();
        assert!(own.ip().is_loopback(), "the default address is {own}");
        assert_eq!((status(own, "evil.example"), status(own, "localhost:8000"), status(own, "127.0.0.1")), (421, 200, 200));
        let chosen = listen(&args(&["serve", "m", "--host", "127.0.0.1", "--port", "0"]), "m", 4096, tx.clone()).map_err(|r| r.to_string()).unwrap();
        assert_eq!((status(chosen, "evil.example"), status(chosen, "localhost:8000")), (421, 200));
        // an address that is not this computer's own: refused with no key, answered with one or with the user's word (and then any Host is, there being no list of names)
        for host in ["0.0.0.0", "::", "192.168.1.5"] {
            let refused = listen(&args(&["serve", "m", "--host", host, "--port", "0"]), "m", 4096, tx.clone()).map(|_| ()).unwrap_err();
            assert!(refused.to_string().contains("--insecure-no-auth") && refused.to_string().contains("GLYD_API_KEY"), "{host}: {refused}");
        }
        let keyed =
            listen(&args(&["serve", "m", "--host", "0.0.0.0", "--port", "0", "--api-key", "k"]), "m", 4096, tx.clone()).map_err(|r| r.to_string()).unwrap();
        assert_eq!((keyed.ip().is_unspecified(), status(keyed, "192.168.1.5:8000")), (true, 200));
        let said =
            listen(&args(&["serve", "m", "--host", "0.0.0.0", "--port", "0", "--insecure-no-auth"]), "m", 4096, tx.clone()).map_err(|r| r.to_string()).unwrap();
        assert_eq!(status(said, "glyd.example.com"), 200);
        // the lists of what the user lets in are checked before the server starts
        for bad in ["*", "localhost:5173", "http://a/b"] {
            let e = listen(&args(&["serve", "m", "--port", "0", "--allowed-origins", bad]), "m", 4096, tx.clone()).map(|_| ()).unwrap_err();
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
        let own = listen(&args(&["serve", "m", "--port", "0", "--ollama-port", &p.to_string()]), "m", 4096, tx.clone()).map_err(|r| r.to_string()).unwrap();
        assert_ne!(own.port(), p);
        // (the engine's thread is not running here: a connection that is answered by the listener alone, the Host check, shows the port is the server's)
        assert!(version(p, "evil.example").contains("421"), "the Host check is the server's own");
        // a port in use: the server starts all the same
        let busy = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let q = busy.local_addr().unwrap().port();
        assert!(listen(&args(&["serve", "m", "--port", "0", "--ollama-port", &q.to_string()]), "m", 4096, tx.clone()).is_ok());
        // not a port: said, and the server starts
        assert!(listen(&args(&["serve", "m", "--port", "0", "--ollama-port", "x"]), "m", 4096, tx).is_ok());
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

    #[test]
    fn what_the_engine_runs() {
        let qwen3 = r#"{"architectures": ["Qwen3ForCausalLM"], "rope_scaling": null}"#;
        assert!(supported("Qwen3-8B", qwen3).is_ok());
        assert!(supported("Qwen3-8B", QWEN3).is_ok());
        let llama = supported("Llama-3.1-8B", r#"{"architectures": ["LlamaForCausalLM"]}"#).unwrap_err();
        assert_eq!(llama.code, 3);
        assert!(llama.what.contains("LlamaForCausalLM"));
        assert_eq!(supported("x", r#"{"architectures": ["Qwen3ForCausalLM"], "rope_scaling": {"type": "yarn"}}"#).unwrap_err().code, 3);
        assert_eq!(supported("x", "{}").unwrap_err().code, 3);
        // Nemotron-H is known by name and refused lossless with its own reason (exit 3, as any other model the engine does not run), in a release build as in a development one: its levels are named
        let nemo = supported("NVIDIA-Nemotron-3-Super-120B-A12B-BF16", r#"{"architectures": ["NemotronHForCausalLM"]}"#).unwrap_err();
        assert_eq!(nemo.code, 3);
        assert!(
            nemo.what.contains("Nemotron-H model")
                && nemo.what.contains("hummingbird, swift and kestrel levels, not lossless")
                && nemo.fix.contains("MODEL:hummingbird on a 16 to 32 GB card")
                && !nemo.fix.contains("wren"),
            "{}",
            nemo.what
        );
        assert_eq!(short_name("Qwen/Qwen3-8B"), "Qwen3-8B");
        assert_eq!(short_name("/models/mine/"), "mine");
    }

    /// A Qwen3 config that sets what the engine does not do is not run by it: a sliding window that is on, layers of another kind, biases, another activation. What it has set off, or null, is nothing.
    #[test]
    fn a_qwen3_that_sets_what_the_engine_does_not_do_goes_to_vllm() {
        let with = |extra: &str| format!(r#"{{"architectures": ["Qwen3ForCausalLM"], {extra}}}"#);
        for fine in [
            r#""sliding_window": null, "use_sliding_window": false"#,
            r#""sliding_window": 4096, "use_sliding_window": false"#,
            r#""sliding_window": 4096"#,
            r#""use_sliding_window": true"#,
            r#""layer_types": ["full_attention", "full_attention"]"#,
            r#""attention_bias": false, "hidden_act": "silu""#,
        ] {
            assert!(supported("m", &with(fine)).is_ok(), "{fine}");
        }
        for (setting, said) in [
            (r#""sliding_window": 4096, "use_sliding_window": true"#, "a sliding window"),
            (r#""layer_types": ["full_attention", "sliding_attention"]"#, "layers that are not full attention"),
            (r#""layer_types": ["linear_attention"]"#, "layers that are not full attention"),
            (r#""attention_bias": true"#, "biases in the attention"),
            (r#""hidden_act": "gelu""#, "an activation other than silu"),
        ] {
            let r = supported("m", &with(setting)).unwrap_err();
            assert!(r.code == CANNOT && r.what.contains(said), "{setting}: {r}");
        }
    }

    #[test]
    fn supports_is_the_kind_of_model_alone() {
        let q = dir_with("supports-q", QWEN3);
        let l = dir_with("supports-l", r#"{"architectures": ["LlamaForCausalLM"]}"#);
        assert_eq!(supports(&args(&["supports", q.to_str().unwrap()])), 0);
        assert_eq!(supports(&args(&["supports", l.to_str().unwrap()])), 3);
        assert_eq!(supports(&args(&["supports", "/nonexistent/model"])), 1); // (cannot be told: the engine's own refusal comes with the run)
        assert_eq!(supports(&args(&["supports"])), 1);
        // the preflight says the same of a model of another kind before it looks at a GPU
        let ui = RefCell::new(Ui::silent());
        let r = check(&args(&["preflight", l.to_str().unwrap()]), &ui).unwrap_err();
        assert!(r.code == 3 && r.what.contains("LlamaForCausalLM"), "{r}");
        assert_eq!(config_text(q.to_str().unwrap()).unwrap(), QWEN3);
        std::fs::remove_dir_all(&q).unwrap();
        std::fs::remove_dir_all(&l).unwrap();
    }

    /// A level is the tag after the model's last ':' when that is all ASCII letters, in any case, and it comes off; --lossy is kestrel; a name that is no level built, and --lossy with another level, are
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
            if !levels::PREVIEW && tag.eq_ignore_ascii_case("wren") {
                // (a release build: wren is not built; hummingbird is)
                let r = level(&format!("{m}:{tag}"), false).unwrap_err();
                assert!(
                    r.code == 1
                        && r.what == format!(":{} is not built yet", tag.to_lowercase())
                        && r.fix == "The levels available now: penguin, kestrel, swift, hummingbird",
                    "{tag}: {r}"
                );
                continue;
            }
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
            "Qwen/Qwen3-8B",
            "/models/mine/",
            "mine",
            "/mnt/disk:1/Qwen3",
            "./a:b/c",
            "/data/x:kestrel/model",
            "C:\\models\\Qwen",
            "Qwen/Qwen3-8B:",
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
        // --lossy is :kestrel: with another level it is refused; a name that is no level built is refused as it is
        for tag in ["penguin", "swift", "SWIFT", "wren", "hummingbird"].into_iter().filter(|t| levels::PREVIEW || *t != "wren") {
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
            (format!("{:.1}", fit::bits(Layout::Int6)), format!("{:.1}", fit::bits(Layout::Int5)), format!("{:.1}", fit::bits(Layout::Int4))),
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
        // penguin's tag is off the model that is looked up: the directory it names is the one read (a Llama: status 3), whatever the tag's case
        let l = dir_with("level-l", r#"{"architectures": ["LlamaForCausalLM"]}"#);
        let m = l.to_str().unwrap();
        for tagged in [m.to_string(), format!("{m}:penguin"), format!("{m}:PENGUIN")] {
            let r = check(&args(&["preflight", &tagged]), &ui).unwrap_err();
            assert!(r.code == 3 && r.what.contains("LlamaForCausalLM"), "{tagged}: {r}");
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

    /// A lossy level's plan sizes the weights in its layout (fit::bits), and its refusals name the level.
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
        assert!(ratio > fit::bits(Layout::Int5) / fit::bits(Layout::Int6) && ratio < 0.9, "{ratio}");
        let ratio = four.weights as f64 / five.weights as f64;
        assert!(ratio > fit::bits(Layout::Int4) / fit::bits(Layout::Int5) && ratio < 0.9, "{ratio}");
        for (layout, level) in [(Layout::Int6, "kestrel"), (Layout::Int5, "swift"), (Layout::Int4, "wren")] {
            let r = plan(layout, 8).err().unwrap();
            assert!(r.code == CANNOT && r.what.starts_with(&format!("Qwen3.8-27B at the {level} level takes about ")), "{r}");
            let r = lossy_plan(&a, "Qwen/Qwen3-8B", layout, QWEN3, None, 80 << 30, "NVIDIA A100", None).err().unwrap();
            assert!(
                r.code == CANNOT
                    && r.what.starts_with(&format!("Qwen3-8B: the {level} level runs Qwen3.5"))
                    && r.fix == "Run it lossless: glyd run Qwen/Qwen3-8B",
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
        let cfg = crate::model_gemma4::ConfigG4::parse(text, None, false).unwrap();
        let (one, eight) = (at(&["run", model, "--no-draft"]), at(&["serve", model, "--users", "8", "--no-draft"]));
        assert_eq!((one.context, one.users), (LOSSY_CTX, 1));
        assert!(eight.context >= LOSSY_MIN_CTX && eight.context < one.context && eight.users == 8, "{}", eight.context);
        let (rings, per) = cfg.kv_bytes(GEMMA_CHUNK, eight.context);
        assert_eq!(eight.kv, 8 * (rings + eight.context as u64 * per));
        assert!(eight.weights + LOSSY_WORK + eight.reserve + eight.kv <= free, "8 users at {} tokens", eight.context);
        assert_eq!(at(&["serve", model, "--users", "1", "--no-draft"]).context, LOSSY_CTX);
        assert_eq!(at(&["serve", model, "--users", "8", "--no-draft", "--ctx", "8192"]).context, 8192);
        // (a draft model to come with the download: drafted, one user at a time, the context of one)
        assert_eq!(at(&["serve", model, "--users", "8"]).users, 1);
        // (run, serve and preflight alike: Gemma 4 without a tag is penguin, with one the level asked; other models without one take the lossless route of old)
        assert_eq!(run_level(model, None).unwrap(), Some(Layout::Tiered));
        assert_eq!(run_level(model, Some(Layout::Int4)).unwrap(), Some(Layout::Int4));
        assert_eq!(run_level("Qwen/Qwen3-8B", None).unwrap(), None);
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
        assert!(r.code == CANNOT && r.what.contains("the i3 level is not built for Nemotron 3 Super") && r.fix.ends_with(&format!("{model}:wren")), "{r}");
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
    #[cfg(feature = "dev")] // (Nemotron 3 Super runs in a development build only: levels::PREVIEW)
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

    /// The plan of a model on a GPU is a status 3 wherever it fails for the computer or the model (not for what the user typed): the memory, the context, the shape.
    #[test]
    fn what_a_model_needs_of_a_gpu_is_decided_before_anything_is_downloaded() {
        let cfg = Config::parse(QWEN3, "config.json", None).unwrap();
        let l4 = gpu("NVIDIA L4", (8, 9), 22);
        let (layout, _, fit) = plan(&args(&["run", "Qwen/Qwen3-8B"]), &cfg, "Qwen3-8B", &l4, None, 1).unwrap();
        assert!(fit.context == 8192 && layout == Layout::Tiered && shape_checks("Qwen3-8B", &cfg, &fit).is_ok());
        // a card with too little room: the vLLM way, with the numbers
        let small = plan(&args(&["run", "Qwen/Qwen3-8B"]), &cfg, "Qwen3-8B", &gpu("NVIDIA L4", (8, 9), 8), None, 1).unwrap_err();
        assert!(small.code == CANNOT && small.what.contains("does not fit this GPU"), "{small}");
        // a context that does not fit
        let big = plan(&args(&["run", "Qwen/Qwen3-8B", "--context", "40960"]), &cfg, "Qwen3-8B", &gpu("NVIDIA L4", (8, 9), 13), None, 1).unwrap_err();
        assert!(big.code == CANNOT && big.fix.contains("--context"), "{big}");
        // a context past the tables' reach on a model whose own window is longer
        let long = Config { max_pos: 262_144, ..cfg.clone() };
        let (_, _, fit) = plan(&args(&["run", "m", "--context", "100000"]), &long, "m", &gpu("NVIDIA A100-SXM4-80GB", (8, 0), 70), None, 1).unwrap();
        let r = shape_checks("m", &long, &fit).unwrap_err();
        assert!(r.code == CANNOT && r.what.contains("100000") && r.what.contains("40960"), "{r}");
        let (_, _, fit) = plan(&args(&["run", "m"]), &long, "m", &l4, None, 1).unwrap();
        assert_eq!(fit.context, 8192); // (the default context is the engine's, whatever the window)
        assert!(shape_checks("m", &long, &fit).is_ok());
        // what the user typed is the user's to fix: status 1
        assert_eq!(plan(&args(&["run", "m", "--layout", "tiny"]), &cfg, "m", &l4, None, 1).unwrap_err().code, 1);
        assert_eq!(plan(&args(&["run", "m", "--context", "abc"]), &cfg, "m", &l4, None, 1).unwrap_err().code, 1);
    }

    /// What the cache's tables said of a model the last time is said again before anything is downloaded or made: for the context's tier, under whichever layout the packed copy is in. (This test
    /// is the only one that reads GLYD_CACHE.)
    #[test]
    fn a_model_the_tables_refused_is_refused_again_before_anything_is_made() {
        let root = std::env::temp_dir().join(format!("glyd-engine-first-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        std::env::set_var("GLYD_CACHE", &root);
        let cfg = Config::parse(QWEN3, "config.json", None).unwrap();
        let source = Source { dir: "/x".into(), repo: "org/model".into(), revision: Some("0123456789abcdef0123456789abcdef01234567".into()) };
        let fit = fit::plan(&cfg, "model", "NVIDIA L4", 22 << 30, Layout::Tiered, None, 1, true).unwrap();
        // a packed copy that is made already: a first run writes no more than the tables
        for l in [Layout::Tiered, Layout::Twelve] {
            let d = store::packed_dir(&source, l);
            std::fs::create_dir_all(&d).unwrap();
            std::fs::write(d.join("engine.safetensors"), b"x").unwrap();
        }
        assert!(first_run_checks("model", &cfg, &fit, Layout::Tiered, Some(&source), 0).is_ok());
        let why = "the model's values compress too poorly (worst kind of text left out: K 0.210%, V 0.050%, the limit 0.15%)";
        // refused at another tier: nothing to say of this context; at this one, under the layout it was packed in, or the other's: refused, with the reason, for the other runner
        store::note_refusal(&store::packed_dir(&source, Layout::Twelve), 16384, why);
        assert!(first_run_checks("model", &cfg, &fit, Layout::Tiered, Some(&source), 0).is_ok());
        store::note_refusal(&store::packed_dir(&source, Layout::Twelve), 8192, why);
        let r = first_run_checks("model", &cfg, &fit, Layout::Tiered, Some(&source), 0).unwrap_err();
        assert!(r.code == CANNOT && r.what == why, "{r}");
        std::fs::remove_dir_all(&root).unwrap();
    }

    /// Every way of loading the engine failing says it in words, and hands the model to the other runner.
    #[test]
    fn a_failure_to_load_the_engine_hands_the_model_to_vllm() {
        let cfg = Config::parse(QWEN3, "config.json", None).unwrap();
        let g = gpu("NVIDIA L4", (8, 9), 22);
        let f = fit::plan(&cfg, "Qwen3-8B", &g.name, g.free, Layout::Tiered, None, 1, true).unwrap();
        let said = |e: &str| engine_refusal(crate::ffi::Error(e.to_string()), "Qwen3-8B", &f, &g);
        for (e, text) in [
            ("cuMemAlloc: out of memory (2)", "ran out of memory"),
            ("cuMemAlloc: CUDA error 2 (2)", "ran out of memory"),
            ("no room for a KV cache", "ran out of memory"),
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
            engine_refusal(crate::ffi::Error("cuModuleLoadData: no kernel image is available for execution on the device (209)".into()), "Qwen3-8B", &f, &old);
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
        let cannot = Refusal::unsupported("Qwen3-8B does not fit this GPU: it needs about 14.2 GiB.", "Close programs that use the GPU");
        let own = Refusal::new("port 8000 is in use by another program", "Choose another: --port 8001");
        // no file named: said on the terminal, nothing written
        let none = Status::new();
        assert_eq!(refuse_to(&ui, &cannot, &none), 3);
        // a file named, the engine not up: the reason there, for either status, the status as it was
        let st = Status::new();
        st.set(file.clone());
        assert_eq!(refuse_to(&ui, &cannot, &st), 3);
        assert_eq!(std::fs::read_to_string(&file).unwrap(), "Qwen3-8B does not fit this GPU: it needs about 14.2 GiB\n");
        assert_eq!(refuse_to(&ui, &own, &st), 1);
        assert_eq!(std::fs::read_to_string(&file).unwrap(), "port 8000 is in use by another program\n");
        // up: the file says so, and a refusal leaves it so
        st.ready();
        assert_eq!(refuse_to(&ui, &own, &st), 1);
        assert_eq!(std::fs::read_to_string(&file).unwrap(), "ready\n");
        // the flag, as the program that starts the engine gives it, among the user's own
        let a = args(&["run", "--status-file", file.to_str().unwrap(), "Qwen/Qwen3-8B", "--prompt", "hi"]);
        assert_eq!(
            (a.pos.clone(), a.flag("status-file"), a.flag("prompt")),
            (vec!["run".to_string(), "Qwen/Qwen3-8B".to_string()], Some(file.to_str().unwrap()), Some("hi"))
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
