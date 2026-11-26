//! Qwen3.8-Flash-Next (`Qwen4ExpForConditionalGeneration`, Condor's model) on the executor (R3d, #140): its config's layer kinds lowered to the shared operations
//! (`lower::qwen4x`) and loaded here: the matrices packed at the level asked (penguin's lossless packs where none is), the ones the library does not pack and the table of n-grams
//! held as saved (bf16), the small tensors as saved but A_log in fp32 (the delta rule reads it so). Each layer's experts are the Moe kind's block of its roles
//! (src/dynamic/experts.rs `block`): either all on the GPU at the level asked (`load`), or at the hummingbird level of a codes pack (src/cb_pack.rs: each expert
//! codebook rows or format q's rows of its own width) in the expert cache (`load_cached`: src/moe_cb.rs, the places on the GPU, the pack's files mapped as their host
//! tier, the misses copied in or computed by the host). The MTP layer is a draft head of the model (`load_drafting`: `lower::draft_head`, src/dynamic/head.rs). Across GPUs
//! the layers are cut by the generic split (src/dynamic/split.rs): a stage's experts all on its GPU (`load_stage`, `open_split`; the head on the last), or in
//! an expert cache of its own (`load_cached_stage`, `open_planned`; `qwenserve --gpus 0,1`). The vision tower is the description's (src/dynamic/images.rs: every
//! loader of the executor opens it, on the first stage of a model cut across GPUs, whose every stage turns its rows by the images' positions). The table of
//! n-grams and the embedding are read on the host
//! from the checkpoint's files mapped, their rows copied up with each pass, as the description holds them (`kinds::TableAt::Host`: src/dynamic/tables.rs).

use super::{Dyn, ModelDesc, Op, Opts, Stage};
use crate::cb_pack::CbPack;
use crate::dev::Dev;
use crate::dynamic::experts::MoeBlock;
use crate::ffi::{err, Result};
use crate::moe_cb::{CbShared, Codes};
use crate::offload::{ExpertStore, Offload};
pub use crate::qwen4x::PackMode;
use crate::runtime::config::ModelConfig;
use crate::weights::Layout;
use glyd_gpu::safetensors::Checkpoint;
use std::path::Path;

/// The executor description's family name.
pub const FAMILY: &str = "Qwen3.8-Flash-Next";

/// The executor's description of the model in `dir` for sequences of up to `ctx` positions: its kinds lowered (`lower::qwen4x`), held to Condor's own reading of
/// the config (`ConfigQx::agrees`).
pub fn lowered(dir: &Path, ctx: usize) -> Result<ModelDesc> {
    lowered_of(dir, &ModelConfig::load(dir)?, ctx)
}

/// `lowered` of a config.json parsed once (`m`: `dir`'s).
pub fn lowered_of(dir: &Path, m: &ModelConfig, ctx: usize) -> Result<ModelDesc> {
    let (kinds, label) = crate::runtime::describe::describe_dir_of(dir, m)?;
    let c = crate::qwen4x::ConfigQx::agrees(m.read_with(dir, crate::qwen4x::ConfigQx::of)?, &kinds).map_err(|e| e.context(&label))?;
    super::lower::qwen4x(&kinds, c.eos.clone(), ctx.min(c.max_pos)).map_err(|e| e.context(&label))
}

/// The model in `dir` on the executor's loader (`Dyn::new_with`: its matrices at the level asked or this GPU's, the table of n-grams and the shapes the library
/// does not pack as saved, A_log in fp32), its experts all on the GPU (the Moe kind's block, `experts::block`) at the level asked: a lossy level's codes where
/// `o.layout` is one, else lossless (penguin), as every family's.
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

/// The dynamic route's stage loader of the family (`split::loader`: the gated residuals' streams): stage `st` (the whole model on one GPU, or its layers
/// `st.layers` of a model split across GPUs: src/dynamic/split.rs), its experts all on its GPU at the level `o.layout` names (`load_stage`); with `caching`
/// (the hummingbird level's plan, `plan::sizing::codes_plan`), the codes pack beside its weights (`DIR/pack`) or `--pack`'s (`codes_of`) in the expert cache's places,
/// `caching.places` a layer, its misses as `caching.offload` says (`load_cached_stage`). A caching that asks for layers' matrices in the computer's memory is
/// refused in words: the family's stay on the GPU.
pub fn stage(dir: &Path, m: &ModelConfig, o: &Opts, st: Stage, caching: Option<&crate::hbg_store::HbCaching>) -> Result<Dyn> {
    let Some(c) = caching else { return load_stage(dir, m, o, experts_at(o.layout), st, None) };
    if c.dense > 0 {
        return err(format!(
            "{FAMILY}: its layers' matrices stay on the GPU beside its expert cache ({} layers of them in the computer's memory are asked)",
            c.dense
        ));
    }
    let pack = crate::plan::sizing::codes_of(dir).ok_or_else(|| {
        crate::ffi::Error(format!(
            "{FAMILY}: its experts in the expert cache are a codes pack beside its weights ({}/pack) or named by --pack, and there is none",
            dir.display()
        ))
    })?;
    // (the pack's experts exact: none of a call's pairs skipped unless `caching.offload` says so)
    let k = Caching { places: Some(c.places), offload: c.offload, ..Caching::new(&pack, PackMode::of(&pack, false), 1) };
    load_cached_stage(dir, m, o, &k, st)
}

/// The draft head of the model in `dir` (its MTP layers: `lower::draft_head` of the family's lowering) for sequences of up to `ctx` positions; None where its
/// config has none.
pub fn head_lowered(dir: &Path, ctx: usize) -> Result<Option<ModelDesc>> {
    head_lowered_of(dir, &ModelConfig::load(dir)?, ctx)
}

/// `head_lowered` of a config.json parsed once (`m`: `dir`'s).
pub fn head_lowered_of(dir: &Path, m: &ModelConfig, ctx: usize) -> Result<Option<ModelDesc>> {
    let (kinds, label) = crate::runtime::describe::describe_dir_of(dir, m)?;
    if kinds.desc.mtp.is_empty() {
        return Ok(None);
    }
    let c = crate::qwen4x::ConfigQx::agrees(m.read_with(dir, crate::qwen4x::ConfigQx::of)?, &kinds).map_err(|e| e.context(&label))?;
    super::lower::draft_head(&kinds, c.eos.clone(), ctx.min(c.max_pos), super::lower::qwen4x).map(Some).map_err(|e| e.context(&label))
}

/// `load` with the experts at `level` (lossless, or a lossy level's codes: six, five or four bits).
pub fn load_level(dir: &Path, o: &Opts, level: Layout) -> Result<Dyn> {
    load_stage(dir, &ModelConfig::load(dir)?, o, level, None, None)
}

/// `load_level` with the model's draft head (`head_lowered`) drafting `k` tokens a round (at most `VERIFY_ROWS` - 1), its experts held as the model's: each round's
/// drafts checked in one verify pass whose rows are the steps' (src/dynamic/head.rs).
pub fn load_drafting(dir: &Path, o: &Opts, level: Layout, k: usize) -> Result<Dyn> {
    load_stage(dir, &ModelConfig::load(dir)?, o, level, None, Some(k))
}

/// Stage `st` of the model in `dir` (None: the whole model) on the executor's loader (`Dyn::new_placed`): the stage's layers' matrices and their experts all on
/// its GPU at `level`, the embedding on the first, the final mixer and the LM head on the last; where `draft` is given, the last stage holds the draft head too
/// (`load_drafting`'s, drafting k tokens a round: src/dynamic/split.rs runs its rounds across the stages).
pub fn load_stage(dir: &Path, m: &ModelConfig, o: &Opts, level: Layout, st: impl Into<Option<Stage>>, draft: Option<usize>) -> Result<Dyn> {
    let c = m.read_with(dir, crate::qwen4x::ConfigQx::of)?;
    let st = st.into().unwrap_or(Stage::whole(c.layers));
    let d = lowered_of(dir, m, o.ctx + o.chunk + 16)?.staged(&st);
    // (what the GPU would hold of the stage: its layers' experts at the level (lossless: two bytes a weight at most) and their table of n-grams as saved where
    // it is on the GPU (`ModelDesc::host`: the release's is on the host); past that, `load_cached`)
    let per = match level {
        Layout::Int6 => 6,
        Layout::Int5 => 5,
        Layout::Int4 => 4,
        _ => 16,
    };
    let experts = ((st.layers.1 - st.layers.0) * c.experts * 3 * c.inter * c.hidden) as u64 * per / 8;
    let table: u64 = held_as_saved(&d, &st).iter().filter(|k| !d.host.contains(k)).map(|&k| (d.mats[k].rows * d.mats[k].cols * 2) as u64).sum();
    // (a plan put the experts here where its GPU holds them (`plan::sizing::size_dynamic`), else in the expert cache: the free memory is asked only by a
    // development check that loads with no plan (`free_sized`), which is refused past it)
    let free = if super::free_sized() { Dev::new(o.device)?.mem_info()?.0 as u64 } else { u64::MAX };
    if !crate::plan::sizing::resident_fits(experts + table, free) {
        return err(format!(
            "{FAMILY}: the experts of layers {}..{} ({:.1} GB at {per} bits) and their table of n-grams ({:.1} GB) do not fit this GPU's {:.1} GB free: a pack of its experts serves it (`load_cached`)",
            st.layers.0,
            st.layers.1,
            experts as f64 / 1e9,
            table as f64 / 1e9,
            free as f64 / 1e9
        ));
    }
    // (each layer's experts the Moe kind's block of its roles, every expert on the GPU at the level; the MTP layers' after the model's, each the expert cache's
    // next layer)
    let (kinds, _) = crate::runtime::describe::describe_dir_of(dir, m)?;
    let held = super::experts::Held { hb: None, cache: None, ordinal: o.device, f8: None, dense: None, codebooks: None, codes: None };
    let hidden = kinds.desc.hidden;
    let blocks = |ck: &Checkpoint, _: Layout| -> Result<Vec<Box<dyn MoeBlock>>> {
        super::experts::of_ffn_layers(&kinds, st.layers).map(|(ix, m, roles)| super::experts::block(m, roles, hidden, ck, level, (ix, &held))).collect()
    };
    let model_moe = super::experts::of_ffn_layers(&kinds, (0, c.layers)).count();
    let head = |ck: &Checkpoint, _: Layout| -> Result<Vec<Box<dyn MoeBlock>>> {
        super::experts::of_head(&kinds, 0)
            .map(|(b, m, roles)| {
                let ix = super::experts::Ix { layer: c.layers + b / 2, moe: model_moe + b / 2 };
                super::experts::block(m, roles, hidden, ck, level, (ix, &held))
            })
            .collect()
    };
    let spec = match draft.filter(|_| st.last) {
        None => None,
        Some(k) => {
            let h = head_lowered_of(dir, m, o.ctx + o.chunk + 16)?
                .ok_or_else(|| crate::ffi::Error(format!("{FAMILY}: {} has no MTP layer to draft with", dir.display())))?;
            Some(super::DraftSpec { dir, desc: h, k, experts: Some(&head) })
        }
    };
    Dyn::new_placed(dir, m, d, spec, st, o, &blocks, &super::Placing::default())
}

/// The model in `dir` at `level` with its layers cut before each of `cuts` over `devices` (one more than the cuts; a GPU named twice: two stages on it, the
/// split's check on one GPU), through the generic split (src/dynamic/split.rs: a stage a thread, the rows handed over through host memory), each stage's
/// experts all on its GPU (`load_stage`); the draft head on the last stage drafting `draft` tokens a round where given. One GPU where `cuts` is empty.
pub fn open_split(dir: &Path, o: &Opts, level: Layout, cuts: &[usize], devices: &[i32], draft: Option<usize>) -> Result<super::split::Run> {
    let m = ModelConfig::load(dir)?;
    let c = m.read_with(dir, crate::qwen4x::ConfigQx::of)?;
    let (owned, m) = (dir.to_path_buf(), std::sync::Arc::new(m));
    super::split::Run::new(c.layers, c.eos, cuts, devices, o, std::sync::Arc::new(move |st, o: &Opts| load_stage(&owned, &m, o, level, st, draft)))
}

/// The routed experts from a codes pack (glyd-hbr-codes/2: each expert cb2, cb3, cb4, ternary, q, q3 or q4, sidecars' copies over some of them) in the
/// shared expert store (src/offload.rs): a layer's places on the GPU, the pack's file mapped as their host tier, the misses computed by the host or copied in as
/// `offload` says. A model split across GPUs has an expert cache a stage, each of these settings (`open_cached`).
#[derive(Clone, Debug)]
pub struct Caching {
    pub pack: std::path::PathBuf,
    pub mode: PackMode,
    /// the sidecars whose copies of some experts the places hold in place of the pack's, a later one's over an earlier one's (`PackMode::side`'s by
    /// default; Condor's server's `--side A,B`, none with `--no-side`: `Form`)
    pub sides: Vec<std::path::PathBuf>,
    /// the layout of the attention blocks' own matrices and the LM head where it is not the stage's (`Placing::hi`: Condor's server's `--hi`)
    pub hi: Option<Layout>,
    /// places a layer (the plan's; None: what the GPU has free after the always-on matrices, less the reserve, over the layers, a development check's
    /// alone: `free_sized`)
    pub places: Option<usize>,
    pub offload: Offload,
    /// the sequences the GPU keeps room for beside the places: their keys, values and states, and the executor's scratch (`reserve_for`, the planner's)
    pub users: usize,
}

impl Caching {
    /// The pack in `pack` served as `mode` (its sidecar, `PackMode::side`), its places sized by the room the GPU has, its misses computed by the host, the
    /// mode's skipping, room kept for `users` sequences; the matrices all at the stage's layout.
    pub fn new(pack: &Path, mode: PackMode, users: usize) -> Caching {
        let offload = Offload { mode: crate::offload::Mode::Cpu, skip_rank: mode.skip_rank(), async_copies: true, ..Offload::default() };
        Caching { pack: pack.to_path_buf(), mode, sides: mode.side(pack).into_iter().collect(), hi: None, places: None, offload, users }
    }

    /// `new` as Condor's server's flags select it (`Form`): their sidecars and second layout.
    pub fn of_form(pack: &Path, f: &Form, users: usize) -> Caching {
        Caching { sides: f.sides.clone(), hi: f.hi, ..Caching::new(pack, f.mode, users) }
    }
}

/// What Condor's server's flags select, flag for flag as v0.34.2's engine read them (its `qwen4x_serve::load`): the pack's mode (`--precise`; else `best` where
/// the pack's folder is named so, else `fast`: its skipping and the Monitor's name), the sidecars over the pack's experts (`--side A,B`; none with `--no-side`;
/// else the mode's, `PackMode::side`), the always-on matrices' layout (`--layout`, four bits where none is given) and the layout of the attention blocks' own
/// matrices and the LM head (`--hi i6 | i5 | i4 | none`; where none is given five bits beside a layout of four, else the layout's), None where it is the
/// layout's. tools/serve.sh: precise `--precise --layout i6 --hi i6`, best `--no-side --layout i6 --hi i6`, fast none of them.
#[derive(Clone, Debug, PartialEq)]
pub struct Form {
    pub mode: PackMode,
    pub sides: Vec<std::path::PathBuf>,
    pub layout: Layout,
    pub hi: Option<Layout>,
}

/// The flags `qwenserve` reads (`http`, `serve_plan`, `form_of`, `users_of`, `gpus`, `plan::sizing::host_flag`), and `--hf-token`, which every command takes.
pub const SERVE_FLAGS: &[&str] = &[
    "port",
    "host",
    "page",
    "name",
    "users",
    "gpus",
    "split",
    "ctx",
    "context",
    "chunk",
    "places",
    "host-memory",
    "image-pixels",
    "precise",
    "layout",
    "hi",
    "side",
    "no-side",
    "insecure-no-auth",
    "allowed-origins",
    "allowed-hosts",
    "drain-timeout",
    "hf-token",
];

/// v0.34.2's `qwenserve` flags that set its own engine, which the executor does not have: each refused with what holds its place now.
const GONE: &[(&str, &str)] = &[
    ("skip", "the misses' skipping is the mode's (--precise, or the pack's folder: best, fast)"),
    ("skip-rank", "the misses' skipping is the mode's (--precise, or the pack's folder: best, fast)"),
    ("skip-rank-later", "the misses' skipping is the mode's (--precise, or the pack's folder: best, fast)"),
    ("mode", "the misses are computed by the host"),
    ("cpu-threads", "the host's threads are the computer's"),
    ("no-async-copies", "the copies are made on a thread of their own"),
    ("no-pin", "the pack's files are mapped as the expert cache's host tier"),
    ("pin-pack", "the pack's files are mapped as the expert cache's host tier"),
    ("whole", "a prompt is read in chunks of --chunk rows"),
    ("cache-gb", "the places are the plan's (--places N a layer)"),
    ("headroom-gb", "the places are the plan's (--places N a layer)"),
    ("no-stream", "a prompt's experts are the expert cache's"),
    ("no-mtp", "the draft head does not draft over the expert cache on the executor"),
    ("draft-vocab", "the draft head does not draft over the expert cache on the executor"),
    ("draft-min-p", "the draft head does not draft over the expert cache on the executor"),
    ("no-skinny", "the products' kernels are the executor's"),
    ("no-segmented", "the products' kernels are the executor's"),
    ("p2p", "the layers cut over GPUs hand their rows over through the computer's memory"),
    ("lock", "a development build's GPU lock of the JSON-lines protocol, which is gone"),
];

/// Every flag `a` gives that `qwenserve` does not read is refused in words (none is ignored): one of v0.34.2's engine's own with what holds its place
/// now, any other with the flags the server takes.
pub fn check_flags(a: &crate::args::Args) -> Result<()> {
    for (f, _) in &a.flags {
        if SERVE_FLAGS.contains(&f.as_str()) {
            continue;
        }
        return match GONE.iter().find(|g| g.0 == f) {
            Some((_, now)) => err(format!("qwenserve --{f}: a setting of v0.34.2's own engine for this model, which is gone; {now}")),
            None => err(format!("qwenserve --{f}: not a flag of the server (its flags: --{})", SERVE_FLAGS.join(", --"))),
        };
    }
    Ok(())
}

/// The `Form` `a`'s flags select of the pack in `pack`; a layout `--hi` does not name, and `--side` beside `--no-side`, refused in words.
pub fn form_of(a: &crate::args::Args, pack: &Path) -> Result<Form> {
    let mode = PackMode::of(pack, a.has("precise"));
    let layout = a.layout()?.unwrap_or(Layout::Int4);
    let hi = match a.flag("hi") {
        Some("i6") => Some(Layout::Int6),
        Some("i5") => Some(Layout::Int5),
        Some("i4") => Some(Layout::Int4),
        Some("none") => None,
        // (by default with i4 the attention layers and the LM head at five bits, as v0.34.2 served them: i4 alone moved a long prompt's first token)
        None => (layout == Layout::Int4).then_some(Layout::Int5),
        Some(x) => return err(format!("--hi {x}: i6, i5, i4 or none")),
    };
    let sides = match (a.flag("side"), a.has("no-side")) {
        (Some(_), true) => return err("--side and --no-side: sidecars named, or none"),
        (Some(list), false) => list.split(',').filter(|x| !x.is_empty()).map(std::path::PathBuf::from).collect(),
        (None, true) => Vec::new(),
        (None, false) => mode.side(pack).into_iter().collect(),
    };
    Ok(Form { mode, sides, layout, hi: hi.filter(|&h| h != layout) })
}

/// The GPU memory the places of stage `st` of a model leave for its sequences and the executor's scratch (`plan::sizing::cache_reserve`, the planner's: #239).
pub fn reserve_for(d: &ModelDesc, users: usize, st: &Stage) -> u64 {
    crate::plan::sizing::cache_reserve(d, users, st)
}

/// The model in `dir` on the executor's loader with its routed experts in the expert cache (`Caching`): the always-on matrices as `load` loads them, each layer's
/// experts the Moe kind's block at the codes pack's hummingbird level (src/moe_cb.rs `CbMoe`) over one store shared by the layers.
pub fn load_cached(dir: &Path, o: &Opts, k: &Caching) -> Result<Dyn> {
    load_cached_stage(dir, &ModelConfig::load(dir)?, o, k, None)
}

/// Stage `st` of `load_cached`'s model (None: the whole model), with an expert cache of its own: its layers' blocks over one store and one set of the codebook
/// block's shared settings, made on this stage's GPU (src/dynamic/split.rs makes each stage on a thread of its own), the places a layer `k.places` or what the
/// GPU has free after the stage's always-on matrices, less the reserve its own layers' sequences take (`reserve_for`), over its layers.
pub fn load_cached_stage(dir: &Path, m: &ModelConfig, o: &Opts, k: &Caching, st: impl Into<Option<Stage>>) -> Result<Dyn> {
    let c = m.read_with(dir, crate::qwen4x::ConfigQx::of)?;
    // (a width the codebook kernels do not take is refused before anything is read or put on the GPU)
    crate::moe_cb::takes(c.hidden, c.inter).map_err(|e| e.context(FAMILY))?;
    let st = st.into().unwrap_or(Stage::whole(c.layers));
    let d = lowered_of(dir, m, o.ctx + o.chunk + 16)?.staged(&st);
    let mut pack = CbPack::open_dims(&k.pack, c.hidden, c.inter)?;
    for side in &k.sides {
        pack = pack.with_side(side)?;
    }
    let layers = st.layers.0..st.layers.1;
    if let Some(l) = layers.clone().find(|&l| !pack.has(l)) {
        return err(format!("{}: no experts of layer {l} of the model's {}", k.pack.display(), c.layers));
    }
    let (store, common) = (ExpertStore::new(), CbShared::new());
    store.set_offload(k.offload);
    common.stream.set(true);
    let place = layers.clone().map(|l| pack.place_bytes(l)).max().unwrap_or(1);
    let kept = layers.clone().filter_map(|l| pack.kept(l).map(<[usize]>::len)).min().unwrap_or(1);
    let (dev, reserve) = (o.device, reserve_for(&d, k.users, &st));
    let (kinds, _) = crate::runtime::describe::describe_dir_of(dir, m)?;
    let hidden = kinds.desc.hidden;
    let blocks = |ck: &Checkpoint, _: Layout| -> Result<Vec<Box<dyn MoeBlock>>> {
        // (the plan's places (`plan::sizing::codes_plan_with`); a development check's with none, read after the always-on matrices are on the GPU:
        // what is free now, less the reserve, is the places')
        let places = match k.places {
            Some(p) => p,
            None if super::free_sized() => crate::plan::sizing::cache_places(Dev::new(dev)?.mem_info()?.0 as u64, reserve, layers.len(), place, kept),
            None => return super::unplanned("the expert cache's places"),
        };
        // (each layer's experts the Moe kind's block at the codes pack's hummingbird level: its places in the stage's one store)
        let codes = Codes { pack: &pack, places, store: &store, common: &common };
        let held = super::experts::Held { hb: None, cache: None, ordinal: dev, f8: None, dense: None, codebooks: None, codes: Some(codes) };
        super::experts::of_ffn_layers(&kinds, st.layers)
            .map(|(ix, m, roles)| {
                crate::status::STATUS.beat();
                super::experts::block(m, roles, hidden, ck, Layout::Hummingbird, (ix, &held))
            })
            .collect()
    };
    let e = Dyn::new_placed(dir, m, d, None, st, o, &blocks, &super::Placing { hi: k.hi, ..Default::default() })?;
    common.fill_page_tables();
    Ok(e)
}

/// The model in `dir` with its routed experts in the expert cache (`k`), its layers cut before each of `cuts` over `devices` (one more than the cuts; a GPU
/// named twice: two stages on it, the split's check on one GPU), through the generic split (src/dynamic/split.rs): each stage's experts in a cache of its own
/// (`load_cached_stage`: `k.places` a layer, else its GPU's room). One GPU where `cuts` is empty.
pub fn open_cached(dir: &Path, o: &Opts, k: &Caching, cuts: &[usize], devices: &[i32]) -> Result<super::split::Run> {
    let m = ModelConfig::load(dir)?;
    let c = m.read_with(dir, crate::qwen4x::ConfigQx::of)?;
    let (owned, k, m) = (dir.to_path_buf(), k.clone(), std::sync::Arc::new(m));
    super::split::Run::new(c.layers, c.eos, cuts, devices, o, std::sync::Arc::new(move |st, o: &Opts| load_cached_stage(&owned, &m, o, &k, st)))
}

/// `load_cached` of the pack in `pack` served as its mode (`PackMode::of`), room kept for `users` sequences.
pub fn load_pack(dir: &Path, pack: &Path, precise: bool, users: usize, o: &Opts) -> Result<Dyn> {
    load_cached(dir, o, &Caching::new(pack, PackMode::of(pack, precise), users))
}

/// The GPUs `--gpus` names (`GLYD_GPUS` of tools/serve.sh), GPU 0 where it names none: one, or several to cut the layers over (a GPU named twice: two parts on
/// it, the split's check on one GPU). Anything else is refused in words.
pub fn gpus(a: &crate::args::Args) -> Result<Vec<i32>> {
    let Some(asked) = a.flag("gpus") else { return Ok(vec![0]) };
    let named: Option<Vec<i32>> = asked.split(',').map(|x| x.trim().parse::<i32>().ok().filter(|&n| n >= 0)).collect();
    named.ok_or_else(|| {
        crate::ffi::Error(format!("--gpus {asked}: GPU numbers, comma-separated (--gpus 1 for the second card, --gpus 0,1 to cut the layers over two)"))
    })
}

/// The requests `qwenserve` answers at once, `--users N` (4 where none is given): 1 to `sched::MAX_USERS`, the most a step runs; the plan then holds them or
/// refuses them in its words, with the most the GPU holds (`serve_plan`). No other bound: the release's 8 was its engine's rows a step.
pub fn users_of(a: &crate::args::Args) -> Result<usize> {
    let users = a.num("users", 4usize)?;
    if !(1..=crate::sched::MAX_USERS).contains(&users) {
        return err(format!("--users {users}: 1 to {} requests at once", crate::sched::MAX_USERS));
    }
    Ok(users)
}

/// The model in `dir` with its routed experts in the expert cache (`k`) as a plan spreads it: its layers cut before each of `cuts` over `devices`, part i's
/// places a layer `places[i]` (`load_cached_stage`; a part the plan gives none is refused where no development check said to size it from free memory).
/// `m`: the directory's config.json as its caller parsed it, once.
pub fn open_planned(
    dir: &Path,
    m: std::sync::Arc<ModelConfig>,
    o: &Opts,
    k: &Caching,
    cuts: &[usize],
    devices: &[i32],
    places: &[usize],
) -> Result<super::split::Run> {
    let c = m.read_with(dir, crate::qwen4x::ConfigQx::of)?;
    let (owned, k, at, places) = (dir.to_path_buf(), k.clone(), cuts.to_vec(), places.to_vec());
    let make = move |st: Stage, o: &Opts| {
        let part = at.iter().take_while(|&&x| x <= st.layers.0).count();
        load_cached_stage(&owned, &m, o, &Caching { places: places.get(part).copied(), ..k.clone() }, st)
    };
    super::split::Run::new(c.layers, c.eos, cuts, devices, o, std::sync::Arc::new(make))
}

/// Condor's server's plan (`plan::sizing::codes_plan_with`, the planned route's rule): the model in `snap` with the pack `pack` served as the flags select
/// (`Form`: its sidecars, the always-on matrices' layout and the attention's and the LM head's), `ctx` tokens of context for `users` at once in prompt chunks
/// of `chunk`, on the GPUs `devs` (the layers cut over them in their order where there are several, at `--split CUT,..` where given), `--places N` a layer
/// where given, `--host-memory` the experts' share of the computer's memory. Refused in the planner's words where it does not fit.
#[allow(clippy::too_many_arguments)]
pub(crate) fn serve_plan(
    snap: &Path,
    c: &ModelConfig,
    pack: &Path,
    form: &Form,
    a: &crate::args::Args,
    (ctx, users, chunk): (usize, usize, usize),
    devs: &[crate::plan::Target],
    host: Option<u64>,
) -> Result<crate::plan::LoadPlan> {
    use crate::ffi::Error;
    let table = crate::sizes::Table::from_dir(snap).map_err(Error)?;
    let row = crate::support::row_named(FAMILY).ok_or_else(|| Error(format!("no family named {FAMILY}")))?;
    let nl = c.num("num_hidden_layers").unwrap_or(0);
    let split = match a.flag("split") {
        None => None,
        Some(s) => {
            let cuts: Vec<usize> = s
                .split(',')
                .map(|x| x.trim().parse::<usize>().ok())
                .collect::<Option<_>>()
                .ok_or_else(|| Error(format!("--split {s}: the first layer of each GPU's part after the first, comma-separated")))?;
            if cuts.len() + 1 != devs.len() || cuts.windows(2).any(|w| w[0] >= w[1]) || cuts.iter().any(|&x| x == 0 || x >= nl) {
                return err(format!(
                    "--split {s}: {} first layers in order between 1 and {}, one for each GPU after the first",
                    devs.len().saturating_sub(1),
                    nl.saturating_sub(1)
                ));
            }
            Some(cuts)
        }
    };
    let places = match a.flag("places") {
        None => None,
        Some(p) => Some(p.parse::<usize>().ok().filter(|&n| n > 0).ok_or_else(|| Error(format!("--places {p}: a number of places a layer")))?),
    };
    let host_memory = crate::plan::sizing::host_flag(a).map_err(|r| Error(r.to_string()))?;
    let ask = crate::plan::sizing::CodesAsk {
        ctx,
        users,
        served: true,
        chunk,
        layout: form.layout,
        hi: form.hi,
        sides: form.sides.clone(),
        cut: devs.len() > 1,
        split,
        places,
        host_memory,
    };
    crate::plan::sizing::codes_plan_with(FAMILY, &table, &c, row, pack, devs, host, &ask).map_err(|r| Error(r.to_string()))
}

/// `qwenserve CKPT PACK --port P [--host H] [--page DIR] [--name NAME] [--users N] [--gpus G,..] [--split CUT,..] [--ctx N | --context N] [--chunk N] [--places N] [--precise]
/// [--layout L] [--hi L] [--side A,B | --no-side] [--image-pixels N]`: the HTTP server (src/server.rs: the chat page, the OpenAI and Anthropic APIs, the Monitor's numbers) on the description
/// executor, the model's routed experts from its pack of codebook rows in the expert cache (the sidecars the flags select over them: `Form`), the always-on
/// matrices at the level asked (four bits where none is; the attention blocks' own and the LM head at `--hi`'s, five bits beside four where none is given),
/// any flag it does not read refused in words (`check_flags`), loaded by its plan (`serve_plan`, the planned route's rule: the users' keys, values and states at the context first, the places a layer what is left, or
/// `--places N`; refused in the planner's words where it does not fit), its requests through the common scheduler (`server::work_with`), up to N (1 to 64:
/// the plan's) at once. It listens while the model loads (/health is 503 until it is loaded). Thinking is off where a request does not ask for it; the sampling defaults are
/// the model's, at a temperature of 0.5 at most with min-p 0.1. `--gpus 0,1`: the layers cut over those GPUs through the generic split (`open_planned`: each
/// part's experts in a cache of its own, the cut the plan's), one request at a time (each part would hold its share of every sequence). Images (PNG, JPEG)
/// come as OpenAI image_url parts (a data: URL or an http(s) one) and Anthropic image blocks, read by the model's vision tower on one GPU (src/dynamic/images.rs),
/// `--image-pixels N` the most pixels each is resized to (1,048,576 by default); across GPUs the first part's GPU runs the tower. Not on the executor yet:
/// the draft head over the expert cache (its verify pass takes experts held whole on the GPU: src/dynamic/head.rs).
pub fn http(snap: &Path, pack: &Path, a: &crate::args::Args) -> Result<()> {
    use crate::server;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;
    check_flags(a)?;
    let name = a.flag("name").unwrap_or("qwen3.8-flash-next").to_string();
    let ctx = a.either("context", "ctx", 33536usize)?;
    let users = users_of(a)?;
    let gpus = gpus(a)?;
    // (the plan, before anything listens or loads: the GPUs named, the pack's sidecars and the matrices' layouts as the flags select them (`Form`: four bits
    // with the attention and the LM head at five by default, as the release has been served))
    let form = form_of(a, pack)?;
    let (mode, layout) = (form.mode, form.layout);
    let chunk = a.num("chunk", 1024usize)?;
    let host_free = crate::offload::host_memory();
    let devs = gpus.iter().map(|&g| Ok(crate::plan::sizing::target(&Dev::probe(g)?))).collect::<Result<Vec<_>>>()?;
    // (config.json parsed once: the plan's and the loader's)
    let m = std::sync::Arc::new(ModelConfig::load(snap)?);
    let plan = serve_plan(snap, &m, pack, &form, a, (ctx, users, chunk), &devs, host_free)?;
    println!("Plan: {}.", plan.text());
    let sides: Vec<String> = form.sides.iter().map(|p| p.display().to_string()).collect();
    println!(
        "Mode: {}; the experts' sidecars: {}; the matrices at {layout:?}{}.",
        mode.name(),
        if sides.is_empty() { "none".to_string() } else { sides.join(", ") },
        form.hi.map_or(String::new(), |h| format!(", the attention's and the LM head at {h:?}"))
    );
    if let Some(p) = a.flag("image-pixels") {
        let n = p.parse::<usize>().ok().filter(|&n| n > 0).ok_or_else(|| crate::ffi::Error(format!("--image-pixels {p}: a number of pixels")))?;
        let _ = super::images::MAX_PIXELS.set(n);
    }
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
    let sp = plan.spread.clone().ok_or_else(|| crate::ffi::Error(format!("{FAMILY}: a codes pack's plan spreads the model")))?;
    // (the Monitor's places, as v0.34.2's server gave them: the chat page's "Experts kept on the GPU")
    let (gb, per_layer) = monitor_places(&sp, &m, pack, &form)?;
    crate::monitor::places(gb, per_layer);
    let o = Opts { ctx, chunk, graph: true, device: sp.devices[0], layout: Some(layout) };
    if gpus.len() > 1 {
        println!(
            "{}",
            match sp.cuts.as_slice() {
                [] => format!("GPU {} holds the model and the places it plans: one part", sp.devices[0]),
                cuts => format!("the layers cut before {cuts:?} over GPUs {:?}", sp.devices),
            }
        );
    }
    let mut e = open_planned(snap, m, &o, &Caching::of_form(pack, &form, plan.users), &sp.cuts, &sp.devices, &sp.places)?;
    // (its sequences' keys, values and states the plan's: on one GPU the pools take its bytes, no more)
    crate::conv::Gen::take_plan(&mut e, plan.users, plan.kv);
    let vocab = crate::qwen4x::ConfigQx::load(snap)?.vocab;
    let weights = crate::conv::Gen::held(&e).map_or(0, |h| h.weights);
    let Some((n, step)) = crate::conv::Gen::as_step(&mut e, plan.users)? else { return err(format!("{FAMILY}: the engine is not a scheduler's step")) };
    // (what the load took of the GPUs against the plan, its pools made: a development build's GLYD_PLAN_REPORT, as `run` and `serve` report theirs)
    let mut seen: Vec<&crate::plan::Target> = Vec::new();
    for t in &devs {
        if !seen.iter().any(|x| x.ordinal == t.ordinal) {
            seen.push(t);
        }
    }
    let used: u64 = seen.iter().map(|t| Dev::probe(t.ordinal).map_or(0, |g| t.free.saturating_sub(g.free))).sum();
    if let Some(path) = crate::tune!("GLYD_PLAN_REPORT") {
        let _ = std::fs::write(path, plan.report(FAMILY, weights, used));
    }
    println!(
        "Loaded: the GPU{} lost {:.2} GiB, of the plan's {:.2} GiB.",
        if seen.len() == 1 { "" } else { "s" },
        used as f64 / (1u64 << 30) as f64,
        (plan.gpu_total() - plan.reserve) as f64 / (1u64 << 30) as f64
    );
    let window = crate::sched::ModelStep::window(step).min(ctx - 2);
    let rt = crate::runtime::context::Context::process().clone();
    let shared = Arc::new(server::Shared { tok, chat, name, defaults, window, vocab, thinking: Some(false), refused: &[], rt });
    crate::monitor::ready();
    loading.store(false, Ordering::Relaxed);
    if n < users {
        let why = if sp.cuts.is_empty() { "" } else { " (its layers cut over GPUs run one sequence at a time)" };
        println!("Ready: up to {n} requests at once, of the {users} asked for{why}.");
    } else {
        println!("Ready: up to {n} requests at once.");
    }
    server::work_with(&mut crate::sched::Sched::new(step, n), shared, &rx, &tx);
    Ok(())
}

/// The places a plan `sp` gives the experts of the pack in `pack` as the form `f` opens it (its sidecars), as v0.34.2's server told its Monitor (`monitor::places`,
/// `/v1/glyd/status`'s `places`, the chat page's "Experts kept on the GPU"): their GB on the GPUs (each part's places a layer times its layers times a place's
/// bytes, the largest expert's of the pack: `CbPack::place_bytes`), and the fewest places a layer of any part.
pub(crate) fn monitor_places(sp: &crate::plan::Spread, m: &ModelConfig, pack: &Path, f: &Form) -> Result<(f64, usize)> {
    let (nl, hidden, inter) = (m.num("num_hidden_layers").unwrap_or(0), m.num("hidden_size").unwrap_or(0), m.num("moe_intermediate_size").unwrap_or(0));
    let mut cb = CbPack::open_dims(pack, hidden, inter)?;
    for side in &f.sides {
        cb = cb.with_side(side)?;
    }
    let place = (0..nl).filter(|&l| cb.has(l)).map(|l| cb.place_bytes(l)).max().unwrap_or(0);
    let bounds: Vec<usize> = std::iter::once(0).chain(sp.cuts.iter().copied()).chain(std::iter::once(nl)).collect();
    let bytes: usize = bounds.windows(2).zip(&sp.places).map(|(w, &p)| (w[0]..w[1]).filter(|&l| cb.has(l)).count() * p * place).sum();
    Ok((bytes as f64 / 1e9, sp.places.iter().copied().min().unwrap_or(0)))
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

/// `glyd-engine servecheck CASE [--tokens N] [--gpus G,.. --split CUT,..] [--serve-mode precise | best | fast] [--sidecar]`: the family's HTTP server (`http`) on the
/// expert cache's random case (`logits::qwen4x_cb_random`, CASE `qwen4x-cb`: its pack, a tokenizer and chat template made for it; or `qwen4x-rel`, the release's
/// form: a slim checkpoint, its n-gram table at 4 bits, `logits::qwen4x_release_random`; `--sidecar`: a sidecar `hot/` over its pack, `logits::codes_sidecar`)
/// against the common scheduler in this process on the same model, pack, form and places on one GPU (`load_cached` as `http` loads it, `form_of` the server's
/// flags): for a short prompt and one past a chunk, the served completion's tokens (greedy, N of them, by their ids) the scheduler's for the same tokens.
/// `--serve-mode`: tools/serve.sh's words for that mode and its pack's folder name (precise `--precise --layout i6 --hi i6` of `precise-v2`, best `--no-side
/// --layout i6 --hi i6` of `best-v2`, fast none of `fast`); none: the pack as made, no flags. With `--gpus` and `--split` the server cuts the layers over those
/// GPUs (`open_cached`), which the one-GPU scheduler's tokens then hold to. An error where one differs.
#[cfg(feature = "dev")]
pub fn serve_check(a: &crate::args::Args) -> Result<()> {
    use crate::sched::{Ev, Finish, Reply, Sched, Submit};
    use std::io::{Read, Write};
    let case = a.need(1, "CASE")?;
    if !matches!(case, "qwen4x-cb" | "qwen4x-rel") {
        return err(format!(
            "servecheck {case}: the served cases are qwen4x-cb (the family's random model with its pack of experts) and qwen4x-rel (the release's form: a slim checkpoint, its n-gram table at 4 bits, its pack beside it)"
        ));
    }
    let tokens = a.num("tokens", 16usize)?;
    let dir = std::env::temp_dir().join(format!("glyd-servecheck-{}", std::process::id()));
    std::fs::create_dir_all(&dir).map_err(|e| crate::ffi::Error(e.to_string()))?;
    crate::logits::random_model(case, &dir)?;
    crate::logits::write_serving_files(&dir, include_str!("../../tests/fixtures/chat_qwen38_tools.jinja"))?;
    if a.has("sidecar") {
        crate::logits::codes_sidecar(&dir)?;
    }
    // (tools/serve.sh's words for the mode asked, and its pack's folder by the mode's name: the pack as made renamed so)
    let (folder, mode_words): (&str, &[&str]) = match a.flag("serve-mode") {
        None => ("pack", &[]),
        Some("precise") => ("precise-v2", &["--precise", "--layout", "i6", "--hi", "i6"]),
        Some("best") => ("best-v2", &["--no-side", "--layout", "i6", "--hi", "i6"]),
        Some("fast") => ("fast", &[]),
        Some(m) => return err(format!("--serve-mode {m}: precise, best or fast (tools/serve.sh's modes)")),
    };
    let pack = dir.join(folder);
    if folder != "pack" {
        std::fs::rename(dir.join("pack"), &pack).map_err(|e| crate::ffi::Error(format!("{}: {e}", pack.display())))?;
    }
    let (ctx, chunk, places, users) = (1024usize, 64usize, 3usize, 2usize);
    let port = std::net::TcpListener::bind("127.0.0.1:0").and_then(|l| l.local_addr()).map_err(|e| crate::ffi::Error(e.to_string()))?.port();
    let words: Vec<String> = ["qwenserve", &dir.display().to_string(), &pack.display().to_string(), "--port", &port.to_string()]
        .into_iter()
        .map(String::from)
        .chain(["--ctx", &ctx.to_string(), "--chunk", &chunk.to_string(), "--places", &places.to_string(), "--users", &users.to_string()].map(String::from))
        .chain(["gpus", "split"].into_iter().filter_map(|f| a.flag(f).map(|v| [format!("--{f}"), v.to_string()])).flatten())
        .chain(mode_words.iter().map(|w| w.to_string()))
        .collect();
    let served = crate::args::Args::from(words);
    let form = form_of(&served, &pack)?;
    println!("servecheck {case}: the server's form {form:?}");
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
    let o = Opts { ctx, chunk, graph: true, device: 0, layout: Some(form.layout) };
    let mut k = Caching::of_form(&pack, &form, users);
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

/// `glyd-engine servescore CKPT PACK [--precise] [--layout L] [--hi L] [--side A,B | --no-side] --ids A,B,.. --from N --out FILE [--places N] [--ctx N] [--chunk N] [--passes N]`:
/// the family's model as Condor's server loads it with those flags (`form_of`, `Caching::of_form`; its places `--places` a layer, else what the GPU has free),
/// the ids before the N-th as a prompt, then each id after it a step: every scored position's whole row of logits (the prompt's last, then each step's) appended
/// to FILE as little-endian f32, as v0.34.2's `qwenserve` wrote them for a request's `score_from` and `dump` (the rows a check holds the two engines to).
#[cfg(feature = "dev")]
pub fn serve_score(a: &crate::args::Args) -> Result<()> {
    use crate::ffi::Error;
    let (dir, pack) = (Path::new(a.need(1, "CKPT")?), Path::new(a.need(2, "PACK")?));
    let form = form_of(a, pack)?;
    let list = a.flag("ids").ok_or_else(|| Error("servescore: --ids A,B,.. the tokens".into()))?;
    let ids: Vec<u32> =
        list.split(',').map(|x| x.trim().parse()).collect::<std::result::Result<_, _>>().map_err(|_| Error(format!("--ids {list}: token ids")))?;
    let from = a.num("from", 1usize)?;
    if from == 0 || from >= ids.len() {
        return err(format!("--from {from}: 1 to {} (a prompt of the ids before it, a step of each after)", ids.len().saturating_sub(1)));
    }
    let out = a.flag("out").ok_or_else(|| Error("servescore: --out FILE".into()))?;
    let o = Opts { ctx: a.either("context", "ctx", 4096usize)?, chunk: a.num("chunk", 1024usize)?, graph: true, device: 0, layout: Some(form.layout) };
    let mut k = Caching::of_form(pack, &form, 1);
    k.places = a.flag("places").map(|p| p.parse::<usize>().map_err(|_| Error(format!("--places {p}: a number")))).transpose()?;
    let mut e = load_cached(dir, &o, &k)?;
    // (the sequence `--passes` times (3 by default), the last one's rows written: a fresh cache's places fill as the router asks, the first pass's misses the
    // host's, as a server's first request; the last two passes' rows compared)
    let passes = a.num("passes", 3usize)?.max(1);
    let (mut rows, mut last): (Vec<Vec<f32>>, Vec<Vec<f32>>) = (Vec::new(), Vec::new());
    for _ in 0..passes {
        last = std::mem::take(&mut rows);
        e.reset()?;
        e.prefill(&ids[..from])?;
        rows.push(e.logits()?);
        for &t in &ids[from..ids.len() - 1] {
            e.step(t)?;
            rows.push(e.logits()?);
        }
    }
    let bits = |r: &[Vec<f32>]| r.iter().flatten().map(|x| x.to_bits()).collect::<Vec<u32>>();
    let bytes: Vec<u8> = rows.iter().flatten().flat_map(|x| x.to_le_bytes()).collect();
    std::fs::write(out, bytes).map_err(|x| Error(format!("{out}: {x}")))?;
    println!(
        "servescore: {} rows of {} logits to {out} after {passes} passes (the last two {}; {:?}, sidecars {:?}, the matrices at {:?}, the attention's and head's at {:?})",
        rows.len(),
        rows[0].len(),
        if passes < 2 { "not compared" } else if bits(&rows) == bits(&last) { "the same bits" } else { "APART" },
        form.mode,
        form.sides,
        form.layout,
        form.hi
    );
    Ok(())
}

/// The matrices stage `st` holds as saved: its layers' tables of n-grams (`Op::NgramRows` reads their rows).
fn held_as_saved(d: &ModelDesc, st: &Stage) -> Vec<usize> {
    d.layers[st.layers.0..st.layers.1].iter().flatten().filter_map(|op| if let Op::NgramRows(x) = op { Some(x.table) } else { None }).collect()
}

#[cfg(test)]
mod tests {
    use super::super::{HeadMix, IdxKeys, Op, Slot, StreamsMix};
    use crate::runtime::describe::describe;

    fn tiny() -> serde_json::Value {
        let m: serde_json::Value = serde_json::from_str(include_str!("../../tests/fixtures/qwen4x_tiny_manifest.json")).unwrap();
        serde_json::from_str(m["config"].as_str().unwrap()).unwrap()
    }

    /// `--gpus` (`GLYD_GPUS` of tools/serve.sh) names the GPUs the family runs on: GPU 0 where none is named, one, or a list to cut the layers over (a GPU named
    /// twice: two parts on it); what is not a list of GPU numbers is refused in words.
    #[test]
    fn gpus_names_the_gpus_or_is_refused_in_words() {
        let gpus = |g: &[&str]| super::gpus(&crate::args::Args::from(["qwenserve", "ckpt", "pack"].iter().chain(g).map(|w| w.to_string())));
        assert_eq!(gpus(&[]).unwrap(), [0]);
        assert_eq!((gpus(&["--gpus", "1"]).unwrap(), gpus(&["--gpus", " 2 "]).unwrap()), (vec![1], vec![2]));
        assert_eq!((gpus(&["--gpus", "0,1"]).unwrap(), gpus(&["--gpus", "0, 0"]).unwrap()), (vec![0, 1], vec![0, 0]));
        for bad in ["", "x", "0,", "1.5", "-1"] {
            let e = gpus(&["--gpus", bad]).unwrap_err().to_string();
            assert!(e.contains(&format!("--gpus {bad}: GPU numbers")) && e.contains("--gpus 0,1 to cut the layers over two"), "{bad:?}: {e}");
        }
    }

    /// `--users` (`GLYD_USERS` of tools/serve.sh) is 1 to the most a step runs (64), 4 where none is given: no 8-row bound (the plan holds them or refuses).
    #[test]
    fn the_users_are_one_to_the_most_a_step_runs() {
        let users = |u: &[&str]| super::users_of(&crate::args::Args::from(["qwenserve", "ckpt", "pack"].iter().chain(u).map(|w| w.to_string())));
        assert_eq!((users(&[]).unwrap(), users(&["--users", "16"]).unwrap(), users(&["--users", "64"]).unwrap()), (4, 16, 64));
        for bad in ["0", "65"] {
            assert!(users(&["--users", bad]).unwrap_err().to_string().contains(&format!("--users {bad}: 1 to 64 requests at once")), "{bad}");
        }
    }

    /// tools/serve.sh's modes select what v0.34.2's engine served (`form_of`): precise its pack's `hot` sidecar and every matrix at six bits; best no sidecar
    /// (`--no-side`) where the pack has one, six bits; fast the pack's `fast` sidecar, four bits with the attention blocks' own matrices and the LM head at five
    /// (`--hi`'s default beside four). `--hi none` / `--hi` of the layout's own bits are the layout; `--side` names the sidecars; a layout `--hi` does not take,
    /// and `--side` beside `--no-side`, are refused in words.
    #[test]
    fn the_servers_modes_select_the_releases_sidecars_and_layouts() {
        use crate::weights::Layout::{Int4, Int5, Int6};
        let root = std::env::temp_dir().join(format!("glyd-form-{}", std::process::id()));
        let (precise, best, fast) = (root.join("precise-v2"), root.join("best-v2"), root.join("fast"));
        for d in [precise.join("hot"), best.join("hot"), fast.join("fast"), fast.join("hot")] {
            std::fs::create_dir_all(&d).unwrap();
            std::fs::write(d.join("meta.json"), "{}").unwrap();
        }
        let form = |pack: &std::path::Path, w: &[&str]| {
            let a = crate::args::Args::from(["qwenserve", "ckpt", pack.to_str().unwrap(), "--port", "8080"].iter().chain(w).map(|x| x.to_string()));
            super::check_flags(&a).and_then(|_| super::form_of(&a, pack))
        };
        let f = |mode, sides: Vec<std::path::PathBuf>, layout, hi| super::Form { mode, sides, layout, hi };
        // (serve.sh's words, mode by mode)
        assert_eq!(form(&precise, &["--precise", "--layout", "i6", "--hi", "i6"]).unwrap(), f(super::PackMode::Precise, vec![precise.join("hot")], Int6, None));
        assert_eq!(form(&best, &["--no-side", "--layout", "i6", "--hi", "i6"]).unwrap(), f(super::PackMode::Best, vec![], Int6, None));
        assert_eq!(form(&fast, &[]).unwrap(), f(super::PackMode::Fast, vec![fast.join("fast")], Int4, Some(Int5)));
        // (each flag moves what it names: best's sidecar without --no-side, the attention at the layout with --hi none or i4, at six with --hi i6)
        assert_eq!(form(&best, &["--layout", "i6"]).unwrap().sides, [best.join("hot")]);
        assert_eq!([&["--hi", "none"][..], &["--hi", "i4"], &["--hi", "i6"]].map(|w| form(&fast, w).unwrap().hi), [None, None, Some(Int6)]);
        assert_eq!(form(&fast, &["--layout", "i5"]).unwrap().hi, None);
        assert_eq!(form(&fast, &["--side", "/a,/b"]).unwrap().sides, [std::path::PathBuf::from("/a"), std::path::PathBuf::from("/b")]);
        let e = form(&fast, &["--hi", "i7"]).unwrap_err().to_string();
        assert!(e.contains("--hi i7: i6, i5, i4 or none"), "{e}");
        let e = form(&best, &["--side", "/a", "--no-side"]).unwrap_err().to_string();
        assert!(e.contains("--side and --no-side"), "{e}");
        let _ = std::fs::remove_dir_all(&root);
    }

    /// `qwenserve` refuses every flag it does not read, in words (none is accepted and ignored): v0.34.2's engine's own settings with what holds their place,
    /// any other with the flags it takes; the flags tools/serve.sh, servecheck and the stage scripts pass are its own.
    #[test]
    fn the_server_refuses_a_flag_it_does_not_read() {
        let check = |w: &[&str]| super::check_flags(&crate::args::Args::from(["qwenserve", "ckpt", "pack"].iter().chain(w).map(|x| x.to_string())));
        let serve_sh = ["--no-side", "--layout", "i6", "--hi", "i6", "--ctx", "32768", "--port", "8080", "--page", "p", "--users", "4", "--gpus", "0,1"];
        check(&serve_sh).unwrap();
        check(&["--precise", "--side", "a", "--context", "4096", "--chunk", "64", "--places", "3", "--split", "1", "--host", "h", "--name", "n"]).unwrap();
        check(&["--host-memory", "20", "--image-pixels", "9", "--insecure-no-auth", "--allowed-origins", "x", "--allowed-hosts", "y", "--drain-timeout", "3"])
            .unwrap();
        let e = check(&["--port", "1", "--draft-min-p", "0.5"]).unwrap_err().to_string();
        assert!(e.contains("qwenserve --draft-min-p: a setting of v0.34.2's own engine") && e.contains("draft head"), "{e}");
        let e = check(&["--skip-rank", "2"]).unwrap_err().to_string();
        assert!(e.contains("skipping is the mode's"), "{e}");
        let e = check(&["--hi6"]).unwrap_err().to_string();
        assert!(e.contains("qwenserve --hi6: not a flag of the server") && e.contains("--hi, --side, --no-side"), "{e}");
    }

    /// `--hi` takes the attention blocks' own matrices and the LM head (`Described::attention_and_head`, `at_hi`), as v0.34.2's engine did: in the tiny model, each
    /// indexed attention layer's q, k and v product, its indexer's product and its o product, and the LM head; none of a Gated DeltaNet layer's, none of the gated
    /// residuals' bottlenecks or inject, not the embedding or the n-gram block's (the shared expert is the codes block's, as int8 rows at any level).
    #[test]
    fn the_second_layout_takes_the_attentions_own_matrices_and_the_head() {
        let d = describe(&tiny()).unwrap();
        let m = super::super::lower::qwen4x(&d, vec![2000], 1100).unwrap();
        let hi = super::super::at_hi(&m, &d.attention_and_head());
        let attn = d.desc.blocks.iter().filter(|b| matches!(b.body, crate::runtime::kinds::Body::Mixer(crate::runtime::kinds::Mixer::Attention(_)))).count();
        let picked: Vec<&str> = (0..m.mats.len()).filter(|&k| hi[k]).map(|k| m.mats[k].members[0].as_str()).collect();
        assert!(attn > 0 && picked.len() == 3 * attn + 1, "{attn} attention layers: {picked:?}");
        assert!(hi[m.lm] && !hi[m.embed], "the LM head, not the embedding");
        for k in (0..m.mats.len()).filter(|&k| hi[k] && k != m.lm) {
            assert!(m.mats[k].members.iter().all(|n| n.contains(".self_attn.")), "{:?}", m.mats[k].members);
        }
        let others = (0..m.mats.len()).filter(|&k| !hi[k]).flat_map(|k| m.mats[k].members.iter());
        assert!(others.clone().all(|n| !n.contains(".self_attn.")), "an attention matrix left at the layout");
        assert!(others.clone().any(|n| n.contains("hyper_connection")) && others.clone().any(|n| n.contains("linear_attn")));
    }

    /// A block layout's share of a matrix's bf16 bytes is its step's: four bits about 28%, five 35%, six 41%; penguin's are its own.
    #[test]
    fn a_layouts_share_of_the_bf16_bytes_is_its_steps() {
        use crate::plan::sizing::layout_pct;
        use crate::weights::Layout;
        assert_eq!([Layout::Int4, Layout::Int5, Layout::Int6].map(layout_pct), [29, 35, 41]);
        assert_eq!((layout_pct(Layout::Tiered), layout_pct(Layout::Twelve)), (super::super::split::TIERED_PCT, super::super::split::TWELVE_PCT));
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
        let table = super::held_as_saved(&m, &super::super::Stage::whole(m.layers.len()));
        assert!(!table.is_empty() && table.iter().all(|k| saved.contains(k)), "{table:?} of {saved:?}");
        // (the family's description holds its tables on the host, the n-gram table and the embedding, which is not its head: neither is on the GPU)
        assert!(m.embed != m.lm && m.host.contains(&m.embed) && table.iter().all(|k| m.host.contains(k)) && m.host.len() == table.len() + 1, "{:?}", m.host);
        // (the table is the n-gram layer's: a stage without that layer holds none)
        assert!(super::held_as_saved(&m, &super::super::Stage { layers: (2, 8), first: false, last: true }).is_empty());
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

    /// A development check's places with no plan take what the GPU has free past the reserve, a layer's share of it: at least one, at most all of a layer's
    /// experts; the reserve holds each sequence's keys, values and states beside the scratch, and grows with the users by them.
    #[test]
    fn the_places_take_the_room_past_the_reserve() {
        use crate::plan::sizing::cache_places;
        assert_eq!(cache_places(10 << 30, 2 << 30, 48, 1 << 20, 512), 170);
        assert_eq!(cache_places(1 << 30, 2 << 30, 48, 1 << 20, 512), 1);
        assert_eq!(cache_places(1 << 40, 0, 48, 1 << 20, 512), 512);
        let m = super::super::lower::qwen4x(&describe(&tiny()).unwrap(), vec![2000], 1100).unwrap();
        let whole = super::super::Stage::whole(m.layers.len());
        let (one, four) = (super::reserve_for(&m, 1, &whole), super::reserve_for(&m, 4, &whole));
        let states: u64 = m.states.iter().map(|&b| b as u64).sum();
        let seq = one - (3 << 29);
        assert!(states > 0 && seq >= states && four - one == 3 * seq, "{one} {four} {states}");
        // (a stage's reserve is its own layers' rings and states: the two halves of the model hold the whole one's between them)
        let half = |layers| super::reserve_for(&m, 4, &super::super::Stage { layers, first: false, last: false }) - (3 << 29);
        assert!(half((0, 4)) > 0 && half((4, 8)) > 0 && half((0, 4)) + half((4, 8)) == four - (3 << 29));
    }

    /// A stage of the model numbers its experts' blocks as its loader reads them (the stage's k-th layer of experts, block k), the whole model's the layers'.
    #[test]
    fn a_stage_numbers_its_experts_from_its_first_layer() {
        use super::super::{Op, Stage};
        let m = super::super::lower::qwen4x(&describe(&tiny()).unwrap(), vec![2000], 1100).unwrap();
        let blocks = |d: &super::super::ModelDesc, ls: std::ops::Range<usize>| -> Vec<usize> {
            d.layers[ls].iter().flatten().filter_map(|op| if let Op::Moe(x) = op { Some(x.block) } else { None }).collect()
        };
        assert_eq!(blocks(&m.clone().staged(&Stage::whole(8)), 0..8), (0..8).collect::<Vec<_>>());
        let late = m.staged(&Stage { layers: (5, 8), first: false, last: true });
        assert_eq!(blocks(&late, 5..8), [0, 1, 2]);
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
        let mut h = super::super::lower::draft_head(&d, vec![2000], 1100, super::super::lower::qwen4x).unwrap();
        let Op::MtpIn { norm_embed, norm_hidden, front: super::super::MtpFront::Streams { embed, hidden } } = h.first else { panic!("{:?}", h.first) };
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
        assert!(super::super::lower::draft_head(&describe(&plain).unwrap(), vec![2000], 1100, super::super::lower::qwen4x)
            .unwrap_err()
            .msg
            .contains("names no draft head"));
    }

    /// A model whose experts the codebook kernels do not take (the tiny one: 128 neurons over 256) is refused by the expert cache's loader before it reads
    /// anything else, in words that name its widths and the kernels'.
    #[cfg(feature = "dev")]
    #[test]
    fn a_width_the_codebook_kernels_do_not_take_is_refused() {
        let dir = std::env::temp_dir().join(format!("glyd-qwen4x-cb-width-{}", std::process::id()));
        crate::logits::qwen4x_random(&dir).unwrap();
        let k =
            super::Caching { places: Some(1), offload: crate::offload::Offload::default(), ..super::Caching::new(&dir.join("pack"), super::PackMode::Fast, 1) };
        let o = super::super::Opts { ctx: 256, chunk: 64, graph: false, device: 0, layout: None };
        let e = super::load_cached(&dir, &o, &k).err().map(|e| e.msg).unwrap_or_default();
        let _ = std::fs::remove_dir_all(&dir);
        assert!(e.contains("128 neurons over a width of 256") && e.contains("exactly 640 neurons over a width of 2560"), "{e}");
        assert!(crate::moe_cb::takes(2560, 640).is_ok() && crate::moe_cb::takes(2560, 1280).is_err());
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
