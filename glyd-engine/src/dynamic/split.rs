//! A model split across several GPUs in one machine (GLY-300 part 1): each GPU runs a stage of the layers (`Stage`) on a thread of its own, which
//! holds that stage's engine (`Dyn::new_stage`: its layers' matrices, caches and states, the embedding on the first, the head on the last). A pass
//! goes stage by stage; the rows' slots (the residual and what the next layer reads) are handed over through host memory, not copied between the
//! GPUs (some machines' direct copies are not to be trusted).

use super::{Dyn, ModelDesc, Opts, Stage};
use crate::ffi::{err, Error, Result};
use crate::model::Sample;
use std::path::Path;
use std::sync::mpsc::{channel, Receiver, Sender};

enum Cmd {
    Reset,
    Rows { tokens: Vec<u32>, at: usize, io: Vec<u16>, head: bool },
    Sample,
    Logits,
    Logprob(f32),
    Describe,
}

enum Out {
    Done(Result<()>),
    Rows(Result<Vec<u16>>),
    Sample(Result<Sample>),
    Logits(Result<Vec<f32>>),
    Logprob(Result<f32>),
    Text(String),
}

struct Worker {
    tx: Option<Sender<Cmd>>,
    rx: Receiver<Out>,
    h: Option<std::thread::JoinHandle<()>>,
}

/// Builds a stage's engine on the GPU of the options it is given.
pub type MakeStage = std::sync::Arc<dyn Fn(Stage, &Opts) -> Result<Dyn> + Send + Sync>;

pub struct Split {
    w: Vec<Worker>,
    /// Tokens in the cache.
    pub n: usize,
    ctx: usize,
    chunk: usize,
    eos: Vec<i64>,
}

fn gone() -> Error {
    Error("a GPU stage's thread ended".into())
}

impl Split {
    /// The model of `dir` described by `d`, its layers cut before each of `cuts` (ascending, inside 1..layers), stage k on GPU `devices[k]`
    /// (one more device than cuts; a device may repeat: two stages on one GPU check the hand-over on a machine with one).
    pub fn new(dir: &Path, d: ModelDesc, cuts: &[usize], devices: &[i32], o: &Opts) -> Result<Split> {
        let (nl, eos, dir) = (d.layers.len(), d.eos.clone(), dir.to_path_buf());
        Split::with(nl, eos, cuts, devices, o, std::sync::Arc::new(move |st: Stage, o: &Opts| Dyn::new_stage(&dir, d.clone(), None, st, o)))
    }

    /// A model of `nl` layers whose stage `make` builds on the GPU of the options it is given (a family's own loader: `qwen35::load_stage`,
    /// `nemotron::load_stage`).
    pub fn with(nl: usize, eos: Vec<i64>, cuts: &[usize], devices: &[i32], o: &Opts, make: MakeStage) -> Result<Split> {
        if devices.len() != cuts.len() + 1 || cuts.windows(2).any(|w| w[0] >= w[1]) || cuts.iter().any(|&c| c == 0 || c >= nl) {
            return err(format!("cuts {cuts:?} over {nl} layers on GPUs {devices:?}: ascending cuts inside the layers, one GPU more than cuts"));
        }
        let bounds: Vec<usize> = std::iter::once(0).chain(cuts.iter().copied()).chain(std::iter::once(nl)).collect();
        let mut w = Vec::new();
        for (k, &device) in devices.iter().enumerate() {
            let st = Stage { layers: (bounds[k], bounds[k + 1]), first: k == 0, last: k + 1 == devices.len() };
            let (tx, crx) = channel::<Cmd>();
            let (otx, rx) = channel::<Out>();
            let (make, mut o) = (make.clone(), o.clone());
            o.device = device;
            let h = std::thread::Builder::new()
                .name(format!("glyd-stage-{k}"))
                .spawn(move || {
                    // (the stage's engine lives and dies on this thread: its GPU context is this thread's)
                    let mut e = match make(st, &o) {
                        Ok(e) => e,
                        Err(x) => {
                            let _ = otx.send(Out::Done(Err(x)));
                            return;
                        }
                    };
                    let _ = otx.send(Out::Done(Ok(())));
                    while let Ok(c) = crx.recv() {
                        let out = match c {
                            Cmd::Reset => Out::Done(e.reset()),
                            Cmd::Rows { tokens, at, mut io, head } => Out::Rows(e.stage_rows(&tokens, at, &mut io, head as usize).map(|_| io)),
                            Cmd::Sample => Out::Sample(e.sample()),
                            Cmd::Logits => Out::Logits(e.logits()),
                            Cmd::Logprob(l) => Out::Logprob(e.logprob(l)),
                            Cmd::Describe => Out::Text(e.describe()),
                        };
                        if otx.send(out).is_err() {
                            break;
                        }
                    }
                })
                .map_err(|x| Error(format!("a GPU stage's thread: {x}")))?;
            w.push(Worker { tx: Some(tx), rx, h: Some(h) });
        }
        // (the stages load at once, a thread each)
        for wk in &w {
            match wk.rx.recv() {
                Ok(Out::Done(r)) => r?,
                _ => return Err(gone()),
            }
        }
        Ok(Split { w, n: 0, ctx: o.ctx.div_ceil(16) * 16, chunk: o.chunk.max(1), eos })
    }

    fn ask(&self, k: usize, c: Cmd) -> Result<Out> {
        self.w[k].tx.as_ref().ok_or_else(gone)?.send(c).map_err(|_| gone())?;
        self.w[k].rx.recv().map_err(|_| gone())
    }

    /// The rows through every stage in turn, the first at `at`; the head on the last row where `head`.
    fn rows(&mut self, tokens: &[u32], at: usize, head: bool) -> Result<()> {
        let mut io = Vec::new();
        for k in 0..self.w.len() {
            match self.ask(k, Cmd::Rows { tokens: tokens.to_vec(), at, io, head })? {
                Out::Rows(r) => io = r?,
                _ => return Err(gone()),
            }
        }
        Ok(())
    }

    pub fn reset(&mut self) -> Result<()> {
        for k in 0..self.w.len() {
            match self.ask(k, Cmd::Reset)? {
                Out::Done(r) => r?,
                _ => return Err(gone()),
            }
        }
        self.n = 0;
        Ok(())
    }

    /// The prompt after the tokens already in the cache, in chunks; the token that follows it.
    pub fn prefill(&mut self, tokens: &[u32]) -> Result<Sample> {
        if tokens.is_empty() {
            return err("an empty prompt");
        }
        if self.n + tokens.len() + 1 > self.ctx {
            return err(format!("{} tokens and {} more, where the cache holds {}", self.n, tokens.len(), self.ctx));
        }
        let mut at = 0;
        while at < tokens.len() {
            let n = self.chunk.min(tokens.len() - at);
            self.rows(&tokens[at..at + n], self.n, at + n == tokens.len())?;
            self.n += n;
            at += n;
        }
        self.sample()
    }

    /// One token through the model; the token after it.
    pub fn step(&mut self, token: u32) -> Result<Sample> {
        if self.n + 1 > self.ctx {
            return err("the cache is full");
        }
        self.rows(&[token], self.n, true)?;
        self.n += 1;
        self.sample()
    }

    fn sample(&self) -> Result<Sample> {
        match self.ask(self.w.len() - 1, Cmd::Sample)? {
            Out::Sample(r) => r,
            _ => Err(gone()),
        }
    }

    /// The last pass's logits (the last stage's).
    pub fn logits(&self) -> Result<Vec<f32>> {
        match self.ask(self.w.len() - 1, Cmd::Logits)? {
            Out::Logits(r) => r,
            _ => Err(gone()),
        }
    }

    /// The log-probability of the last pass's `logit` (the last stage's, on its GPU).
    pub fn logprob(&self, logit: f32) -> Result<f32> {
        match self.ask(self.w.len() - 1, Cmd::Logprob(logit))? {
            Out::Logprob(r) => r,
            _ => Err(gone()),
        }
    }

    pub fn describe(&self) -> String {
        (0..self.w.len())
            .map(|k| match self.ask(k, Cmd::Describe) {
                Ok(Out::Text(t)) => format!("stage {k}: {t}"),
                _ => format!("stage {k}: gone"),
            })
            .collect::<Vec<_>>()
            .join("; ")
    }

    pub fn capacity(&self) -> usize {
        self.ctx
    }

    pub fn eos(&self) -> &[i64] {
        &self.eos
    }
}

impl Drop for Split {
    fn drop(&mut self) {
        // (closing a stage's channel ends its loop; its engine is freed on its own thread)
        for wk in &mut self.w {
            wk.tx = None;
        }
        for wk in &mut self.w {
            if let Some(h) = wk.h.take() {
                let _ = h.join();
            }
        }
    }
}

/// A model on one GPU, or split across several: what a check or a caller runs either way.
pub enum Run {
    One(Dyn),
    Split(Split),
    /// Every layer across the GPUs (src/dynamic/tp.rs).
    Tp(super::tp::Tp),
}

impl Run {
    /// One GPU where `cuts` is empty, else `Split::with`'s stages (`devices`: one more than cuts; a device may repeat), each made by `make`.
    pub fn new(nl: usize, eos: Vec<i64>, cuts: &[usize], devices: &[i32], o: &Opts, make: MakeStage) -> Result<Run> {
        if cuts.is_empty() {
            let mut o = o.clone();
            o.device = devices.first().copied().unwrap_or(o.device);
            return Ok(Run::One(make(Stage::whole(nl), &o)?));
        }
        Ok(Run::Split(Split::with(nl, eos, cuts, devices, o, make)?))
    }
    pub fn describe(&self) -> String {
        match self {
            Run::One(e) => e.describe(),
            Run::Split(e) => e.describe(),
            Run::Tp(e) => e.describe(),
        }
    }
    pub fn prefill(&mut self, t: &[u32]) -> Result<Sample> {
        match self {
            Run::One(e) => e.prefill(t),
            Run::Split(e) => e.prefill(t),
            Run::Tp(e) => e.prefill(t),
        }
    }
    pub fn step(&mut self, t: u32) -> Result<Sample> {
        match self {
            Run::One(e) => e.step(t),
            Run::Split(e) => e.step(t),
            Run::Tp(e) => e.step(t),
        }
    }
    pub fn logits(&self) -> Result<Vec<f32>> {
        match self {
            Run::One(e) => e.logits(),
            Run::Split(e) => e.logits(),
            Run::Tp(e) => e.logits(),
        }
    }
    pub fn reset(&mut self) -> Result<()> {
        match self {
            Run::One(e) => e.reset(),
            Run::Split(e) => e.reset(),
            Run::Tp(e) => e.reset(),
        }
    }
}

/// `--tp GPU,..` of a command line: every layer across those GPUs (a GPU may repeat: places on one GPU check the gathers), or none.
pub fn tp_args(a: &crate::args::Args) -> Result<Option<Vec<i32>>> {
    a.flag("tp").map(|v| v.split(',').map(|x| x.trim().parse::<i32>().map_err(|_| Error(format!("--tp {v}: GPU numbers")))).collect()).transpose()
}

/// `--split CUT,..` and `--devices GPU,..` of a command line: the cuts (none: one GPU) and the devices (one more than the cuts; GPU 0 for each where not given).
pub fn split_args(a: &crate::args::Args) -> Result<(Vec<usize>, Vec<i32>)> {
    let num = |v: &str, what: &str| -> Result<Vec<i64>> {
        v.split(',').map(|x| x.trim().parse::<i64>().map_err(|_| Error(format!("--{what} {v}: numbers")))).collect()
    };
    let cuts: Vec<usize> = match a.flag("split") {
        Some(v) => num(v, "split")?.into_iter().map(|x| x as usize).collect(),
        None => Vec::new(),
    };
    let devices: Vec<i32> = match a.flag("devices") {
        Some(v) => num(v, "devices")?.into_iter().map(|x| x as i32).collect(),
        None => vec![0; cuts.len() + 1],
    };
    Ok((cuts, devices))
}

impl Split {
    /// A reply as `Dyn::reply` gives one, without drafting; a sampled token from the last stage's logits row on the host.
    fn reply(&mut self, prompt: &[u32], o: &crate::model::Reply, each: &mut dyn FnMut(u32) -> bool) -> Result<(Vec<u32>, crate::model::Finish)> {
        use crate::model::Finish;
        if prompt.is_empty() || prompt.len() + 1 > self.ctx {
            return err(format!("{} prompt tokens, where the context holds {}", prompt.len(), self.ctx));
        }
        self.reset()?;
        let greedy = o.params.greedy();
        let mut sampler = crate::sampler::Sampler::new(o.params.clone(), prompt);
        let mut s = self.prefill(prompt)?;
        let mut out = Vec::new();
        loop {
            let tok = if greedy { s.token } else { sampler.sample_with(None, || self.logits())? };
            if !o.ignore_eos && (self.eos.contains(&(tok as i64)) || o.stop_ids.contains(&tok)) {
                return Ok((out, Finish::Stop));
            }
            out.push(tok);
            if !each(tok) {
                return Ok((out, Finish::Stopped));
            }
            if out.len() >= o.max_new || self.n + 1 > self.ctx {
                return Ok((out, Finish::Length));
            }
            s = self.step(tok)?;
        }
    }
}

impl crate::conv::Gen for Run {
    fn capacity(&self) -> usize {
        match self {
            Run::One(e) => crate::conv::Gen::capacity(e),
            Run::Split(e) => e.capacity(),
            Run::Tp(e) => e.capacity(),
        }
    }
    fn reply(&mut self, prompt: &[u32], o: &crate::model::Reply, each: &mut dyn FnMut(u32) -> bool) -> Result<(Vec<u32>, crate::model::Finish)> {
        match self {
            Run::One(e) => crate::conv::Gen::reply(e, prompt, o, each),
            Run::Split(e) => e.reply(prompt, o, each),
            Run::Tp(e) => e.reply(prompt, o, each),
        }
    }
    fn held(&self) -> Option<crate::dev::Held> {
        match self {
            Run::One(e) => crate::conv::Gen::held(e),
            Run::Split(_) | Run::Tp(_) => None,
        }
    }
    /// One GPU: the engine's own step (src/dynamic/rows.rs); stages and layers across GPUs run one sequence at a time.
    fn as_step(&mut self, users: usize) -> Result<Option<(usize, &mut dyn crate::sched::ModelStep)>> {
        match self {
            Run::One(e) => crate::conv::Gen::as_step(e, users),
            Run::Split(_) | Run::Tp(_) => Ok(None),
        }
    }
}

/// The families the dynamic engine runs from their release's directory through `open`, by the config's text (None: another path's model).
pub fn family(text: &str) -> Option<&'static str> {
    if crate::muse::is_muse(text) {
        Some("Muse Spark")
    } else if crate::kimi::is_kimi(text) {
        Some("Kimi-K3")
    } else if crate::glm::is_glm(text) {
        Some("GLM-5.3-Flash")
    } else if crate::nemotron::is_nemotron_h(text) {
        Some("Nemotron-H")
    } else if glyd_gpu::json::parse(text).ok().and_then(|v| v.get("model_type").and_then(|t| t.as_str()).map(|t| t == "deepseek_v4")).unwrap_or(false) {
        Some("DeepSeek-V4")
    } else {
        None
    }
}

/// The model of `dir` on the dynamic engine, by its config's family: on one GPU where `cuts` is empty, else in stages (`devices`: one more than `cuts`).
/// GLM's and Nemotron-H's experts at `o.layout`'s level (penguin's lossless where none is given); DeepSeek-V4's and Muse's as the release has them, exact.
pub fn open(dir: &Path, o: &Opts, cuts: &[usize], devices: &[i32]) -> Result<Run> {
    let (nl, eos, make) = maker(dir, o)?;
    Run::new(nl, eos, cuts, devices, o, make)
}

/// The model of `dir` with every layer across `devices` (src/dynamic/tp.rs).
pub fn open_tp(dir: &Path, o: &Opts, devices: &[i32]) -> Result<Run> {
    let (nl, eos, make) = maker(dir, o)?;
    Ok(Run::Tp(super::tp::Tp::with(nl, eos, devices, o, make)?))
}

/// The family's layers, end tokens and loader of a stage, by the config of `dir`.
fn maker(dir: &Path, o: &Opts) -> Result<(usize, Vec<i64>, MakeStage)> {
    let text = std::fs::read_to_string(dir.join("config.json")).map_err(|e| Error(format!("{}: {e}", dir.display())))?;
    let owned = dir.to_path_buf();
    let level = o.layout.unwrap_or(crate::weights::Layout::Tiered);
    Ok(match family(&text) {
        Some("Muse Spark") => {
            let c = crate::muse::ConfigMuse::load(dir)?;
            (c.layers, c.eos.clone(), std::sync::Arc::new(move |st, o: &Opts| super::muse::load_stage(&owned, o, st)))
        }
        Some("Kimi-K3") => {
            let c = crate::kimi::ConfigKimi::load(dir)?;
            // (a development build's GLYD_KIMI_PLACES: the experts of a layer on the GPU, the rest in the expert cache's host tier)
            let places = crate::tune!("GLYD_KIMI_PLACES").and_then(|v| v.parse::<usize>().ok());
            (c.layers, c.eos.clone(), std::sync::Arc::new(move |st, o: &Opts| super::kimi::load_with_places(&owned, o, st, places)))
        }
        Some("GLM-5.3-Flash") => {
            let c = crate::glm::ConfigGlm::load(dir)?;
            (c.layers, c.eos.clone(), std::sync::Arc::new(move |st, o: &Opts| super::glm::load_stage(&owned, o, level, st)))
        }
        Some("Nemotron-H") => {
            let (nl, eos) = (config_num(&text, "num_hidden_layers").unwrap_or(0), config_eos(&text));
            (nl, eos, std::sync::Arc::new(move |st, o: &Opts| super::nemotron::load_stage(&owned, o, level, st)))
        }
        _ => {
            let c = crate::deepseek_v4::ConfigV4::load(dir)?;
            (c.layers, c.eos.clone(), std::sync::Arc::new(move |st, o: &Opts| super::deepseek_v4::load_stage(&owned, o, st)))
        }
    })
}

/// A number of the config (its `text_config`'s for a model that nests one).
pub fn config_num(text: &str, key: &str) -> Option<usize> {
    let v = glyd_gpu::json::parse(text).ok()?;
    let n = v.get(key).or_else(|| v.get("text_config").and_then(|t| t.get(key)))?;
    n.as_u64().map(|x| x as usize)
}

fn config_eos(text: &str) -> Vec<i64> {
    let Some(v) = glyd_gpu::json::parse(text).ok() else { return Vec::new() };
    match v.get("eos_token_id") {
        Some(glyd_gpu::json::Value::Array(a)) => a.iter().filter_map(|x| x.as_i64()).collect(),
        Some(x) => x.as_i64().map(|x| vec![x]).unwrap_or_default(),
        None => Vec::new(),
    }
}

/// An FP8 matrix's bytes on the GPU, in hundredths of the release's: layout H's plain packs (an HBM GPU's default) 9 bits a value (a patch's row blocks 0-2
/// in its 7-bit core, the 4th in var: 112.5%); coded packs about 7.1 bits (layout H; layout c 6.75), 90 a bound. ponytail: fixed shares; the matrices' own
/// sizes (`f8_size`) where a release codes worse.
pub const PLAIN_PCT: u64 = 113;
pub const CODED_PCT: u64 = 90;
/// A bf16 matrix's bytes on the GPU at the penguin level, in hundredths of the release's: the smallest lossless layout (tiered, about 10.8 bits a value)
/// 72, the 12-bit one (about 12.0 and its exceptions) 78. The 12-bit layout's one-token product reads at about 80% of an H100's rate where the tiered
/// one's reads at 30-45% (research/fp4-one-h100-2026-10-22), so an HBM GPU holds bf16 matrices in it where they fit.
pub const TIERED_PCT: u64 = 72;
pub const TWELVE_PCT: u64 = 78;

/// The bytes each of the `nl` layers holds on a GPU, and the rest's (embedding, head, final norm): the release's tensors as they are, an FP8 matrix at
/// `f8_pct` hundredths of its bytes, a bf16 matrix at penguin's 72% (about 11.5 bits a weight, an upper bound). Not counted: the tensors no description
/// loads (a multi-token-prediction layer, a vision or audio tower).
pub fn layer_bytes(dir: &Path, nl: usize, f8_pct: u64) -> Result<(Vec<u64>, u64)> {
    layer_bytes_where(dir, nl, f8_pct, TIERED_PCT, &|_| true)
}

/// As `layer_bytes`, of the tensors whose names `keep` keeps.
pub fn layer_bytes_where(dir: &Path, nl: usize, f8_pct: u64, bf16_pct: u64, keep: &dyn Fn(&str) -> bool) -> Result<(Vec<u64>, u64)> {
    layer_bytes_with(dir, nl, f8_pct, bf16_pct, keep, true)
}

/// `layer_bytes_where`, a hummingbird directory's experts (their files of their own) counted in each layer's bytes only where `experts`.
pub fn layer_bytes_with(dir: &Path, nl: usize, f8_pct: u64, bf16_pct: u64, keep: &dyn Fn(&str) -> bool, experts: bool) -> Result<(Vec<u64>, u64)> {
    let ck = glyd_gpu::safetensors::Checkpoint::open(dir).map_err(|e| Error(format!("{}: {e}", dir.display())))?;
    let (mut per, mut base) = (vec![0u64; nl], 0u64);
    for f in &ck.files {
        for (name, t) in &f.tensors {
            if ["mtp", "vision", "visual", "audio", "multi_modal"].iter().any(|w| name.contains(w)) || !keep(name) {
                continue;
            }
            let matrix = t.shape.len() == 2 && t.shape.iter().all(|&d| d >= 256);
            let b = match t.dtype.as_str() {
                "BF16" if matrix => t.bytes * bf16_pct / 100,
                "F8_E4M3" if matrix => t.bytes * f8_pct / 100,
                _ => t.bytes,
            };
            match layer_of(name) {
                Some(l) if l < nl => per[l] += b,
                Some(_) => {} // (a layer past the model's: a multi-token-prediction one)
                None => base += b,
            }
        }
    }
    // (a hummingbird directory's routed experts are in files of their own, by the description's block: decoder layer l's are block 2 l + 1)
    if let Some(hb) = crate::hb_pack::HbDir::open(dir)?.filter(|_| experts) {
        for (l, bytes) in hb.layer_bytes() {
            if l < nl {
                per[l] += bytes;
            }
        }
    }
    Ok((per, base))
}

/// The bytes `layer_bytes` gives and the cut `plan_cuts` makes over `room`: on HBM GPUs (`hbm`) the FP8 matrices in layout H's plain packs and, for a loader
/// that takes it (`twelve`: Kimi-K3's, whose `Opts::layout` is its bf16 matrices' alone), the bf16 ones in the 12-bit layout where they fit (their faster
/// products), else the bf16 ones tiered, then the FP8 ones coded too (coded: true third, `Lib::f8h_plain(Some(false))` before the load; 12-bit: true fourth,
/// `Layout::Twelve` the load's); elsewhere coded and tiered, their only layouts. No cut where nothing fits.
#[allow(clippy::type_complexity)]
pub fn fit(dir: &Path, nl: usize, room: &[(i32, u64)], hbm: bool, twelve: bool) -> Result<((Vec<u64>, u64), Option<(Vec<usize>, Vec<i32>)>, bool, bool)> {
    if hbm {
        for (f8, twelve) in [(PLAIN_PCT, twelve), (PLAIN_PCT, false), (CODED_PCT, twelve)] {
            let b = layer_bytes_where(dir, nl, f8, if twelve { TWELVE_PCT } else { TIERED_PCT }, &|_| true)?;
            if let Some(c) = plan_cuts(&b.0, b.1, room) {
                return Ok((b, Some(c), f8 == CODED_PCT, twelve));
            }
        }
    }
    let coded = layer_bytes(dir, nl, CODED_PCT)?;
    let cut = plan_cuts(&coded.0, coded.1, room);
    let switched = hbm && cut.is_some();
    Ok((coded, cut, switched, false))
}

/// A model whose experts the GPUs do not all hold (Kimi-K3's 1.4 TB on 8x 80 GB): its layers over the GPUs of `room` in order, each layer taking its bytes
/// besides its routed experts (`dense`) and, where it has experts (`moe`), places for the same number of them, `expert` bytes each, the most that lets every
/// layer fit (`base`, the rest, half on the first GPU and half on the last); then each GPU's room left over as more places for its own layers, as many as a
/// layer has at most. The cuts, their GPUs and each stage's places a layer; None where fewer than `least` a layer fit.
pub fn places_plan(
    dense: &[u64],
    base: u64,
    moe: &[bool],
    room: &[(i32, u64)],
    expert: u64,
    experts: usize,
    least: usize,
) -> Option<(Vec<usize>, Vec<i32>, Vec<usize>)> {
    let n = dense.len();
    let foot = |l: usize, p: u64| dense[l] + if moe[l] { p * expert } else { 0 };
    // (the layers in order, each GPU filled to its room, with p places a layer: where each stage starts; None where the GPUs do not hold them)
    let assign = |p: u64| -> Option<Vec<usize>> {
        let (mut starts, mut g, mut used) = (vec![0usize], 0usize, base / 2);
        for l in 0..n {
            if used + foot(l, p) > room.get(g)?.1 {
                g += 1;
                starts.push(l);
                used = 0;
            }
            used += foot(l, p);
            if used > room.get(g)?.1 {
                return None;
            }
        }
        (used + base - base / 2 <= room[g].1).then_some(starts)
    };
    let (mut lo, mut hi) = (least as u64, experts as u64);
    assign(lo)?;
    while lo < hi {
        let mid = (lo + hi).div_ceil(2);
        if assign(mid).is_some() {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    let starts = assign(lo)?;
    let bounds: Vec<usize> = starts.iter().copied().chain(std::iter::once(n)).collect();
    let stages = starts.len();
    let mut places = Vec::new();
    for (k, w) in bounds.windows(2).enumerate() {
        let held = dense[w[0]..w[1]].iter().sum::<u64>() + if k == 0 { base / 2 } else { 0 } + if k + 1 == stages { base - base / 2 } else { 0 };
        let layers = moe[w[0]..w[1]].iter().filter(|&&m| m).count().max(1) as u64;
        places.push(((room[k].1.saturating_sub(held) / (layers * expert.max(1))) as usize).clamp(lo as usize, experts));
    }
    Some((starts[1..].to_vec(), room[..stages].iter().map(|r| r.0).collect(), places))
}

/// Kimi-K3 in stages (`open`'s) with `places[k]` of a layer's experts on stage k's GPU, the rest in the expert cache's host tier.
pub fn open_kimi(dir: &Path, o: &Opts, cuts: &[usize], devices: &[i32], places: Vec<usize>, dense: usize) -> Result<Run> {
    let c = crate::kimi::ConfigKimi::load(dir)?;
    let owned = dir.to_path_buf();
    let firsts: Vec<usize> = std::iter::once(0).chain(cuts.iter().copied()).collect();
    Run::new(
        c.layers,
        c.eos.clone(),
        cuts,
        devices,
        o,
        std::sync::Arc::new(move |st, o: &Opts| {
            let k = firsts.iter().position(|&f| f == st.layers.0).unwrap_or(0);
            // (the stage's last `dense` layers' matrices in the computer's memory, src/dense_ram.rs: one stage alone)
            let caching =
                (dense > 0).then(|| crate::hbg_store::HbCaching { places: places.get(k).copied().unwrap_or(1), ram: None, offload: Default::default(), dense });
            super::kimi::load_with_caching(&owned, o, st, places.get(k).copied(), caching.as_ref())
        }),
    )
}

/// Each of the model's `nl` layers' bytes that its family's loader can keep in the computer's memory (src/dense_ram.rs: its packed linears there), at the packing `layer_bytes_where` counts: Muse Spark's attention and
/// latent projections, Kimi-K3's delta attention's three; zeros for the others (their attention is in blocks of their own, or FP8).
pub fn ram_bytes(dir: &Path, nl: usize, family: &str, f8_pct: u64, bf16_pct: u64) -> Result<Vec<u64>> {
    let kda: Vec<bool> = match family {
        "Kimi-K3" => crate::kimi::ConfigKimi::load(dir)?.kinds.iter().map(|k| matches!(k, crate::glm::GlmKind::Kda)).collect(),
        "Muse Spark" => Vec::new(),
        _ => return Ok(vec![0; nl]),
    };
    Ok(layer_bytes_where(dir, nl, f8_pct, bf16_pct, &|n| in_ram(family, &kda, n))?.0)
}

/// Whether the tensor `name` is one of the matrices `family`'s loader keeps in the computer's memory for a layer there (src/dense_ram.rs; `kda`: Kimi-K3's layers of delta attention).
fn in_ram(family: &str, kda: &[bool], name: &str) -> bool {
    let layer = name.split("layers.").nth(1).and_then(|r| r.split('.').next()).and_then(|x| x.parse::<usize>().ok());
    match family {
        "Muse Spark" => {
            (name.contains(".self_attn.") && name.ends_with("_proj.weight")) || name.contains(".pre_expert_proj.") || name.contains(".post_expert_proj.")
        }
        "Kimi-K3" => {
            layer.is_some_and(|l| kda.get(l) == Some(&true))
                && name.contains(".self_attn.")
                && ["q_proj", "k_proj", "v_proj", "g_proj", "f_a_proj", "b_proj", "o_proj", "f_b_proj"].iter().any(|x| name.ends_with(&format!(".{x}.weight")))
        }
        _ => false,
    }
}

/// `places_plan`, and where it finds no room on one GPU, the last layers' dense matrices in the computer's memory (src/dense_ram.rs; `ram`: each layer's bytes that can be there), as few layers as fit: their bytes off the GPU,
/// one slot of the largest of them on it. The cuts, their GPUs, each stage's places, how many layers are in the computer's memory, and the bytes that leaves off the GPU (theirs less the slot).
#[allow(clippy::type_complexity)]
pub fn places_plan_dense(
    dense: &[u64],
    base: u64,
    moe: &[bool],
    room: &[(i32, u64)],
    expert: u64,
    experts: usize,
    least: usize,
    ram: &[u64],
) -> Option<(Vec<usize>, Vec<i32>, Vec<usize>, usize, u64)> {
    if let Some((c, d, p)) = places_plan(dense, base, moe, room, expert, experts, least) {
        return Some((c, d, p, 0, 0));
    }
    if room.len() != 1 || ram.len() != dense.len() {
        return None;
    }
    let n = dense.len();
    (1..=n).find_map(|k| {
        let from = n - k;
        let moved: Vec<u64> = (from..n).map(|l| ram[l].min(dense[l])).collect();
        let slot = moved.iter().copied().max().unwrap_or(0);
        if slot == 0 {
            return None;
        }
        let mut d = dense.to_vec();
        d[from..].iter_mut().zip(&moved).for_each(|(x, m)| *x -= m);
        places_plan(&d, base + slot, moe, room, expert, experts, least).map(|(c, dv, p)| (c, dv, p, k, moved.iter().sum::<u64>() - slot))
    })
}

/// A model of the dynamic route in stages (`open`'s), stage k's layers with their experts in the expert cache as `cachings[k]` says: a hummingbird directory's (src/hbg_store.rs: places on its GPU, the computer's memory, the disk),
/// or the release's where its experts are each the same bytes (src/tiering.rs, R12: DeepSeek-V4's FP4).
pub fn open_cached(dir: &Path, o: &Opts, cuts: &[usize], devices: &[i32], cachings: Vec<crate::hbg_store::HbCaching>) -> Result<Run> {
    let text = std::fs::read_to_string(dir.join("config.json")).map_err(|e| Error(format!("{}: {e}", dir.display())))?;
    let (nl, eos) = match family(&text) {
        Some("Muse Spark") => {
            let c = crate::muse::ConfigMuse::load(dir)?;
            (c.layers, c.eos.clone())
        }
        Some("Kimi-K3") => {
            let c = crate::kimi::ConfigKimi::load(dir)?;
            (c.layers, c.eos.clone())
        }
        Some("GLM-5.3-Flash") => {
            let c = crate::glm::ConfigGlm::load(dir)?;
            (c.layers, c.eos.clone())
        }
        Some("Nemotron-H") | None => {
            return err(format!("{}: not a family whose experts the expert cache takes (Kimi-K3, DeepSeek-V4, GLM-5.3-Flash, Muse Spark)", dir.display()))
        }
        Some(_) => {
            let c = crate::deepseek_v4::ConfigV4::load(dir)?;
            (c.layers, c.eos.clone())
        }
    };
    let fam = family(&text).unwrap_or("");
    let level = o.layout.unwrap_or(crate::weights::Layout::Tiered);
    let owned = dir.to_path_buf();
    let firsts: Vec<usize> = std::iter::once(0).chain(cuts.iter().copied()).collect();
    let make: MakeStage = std::sync::Arc::new(move |st, o: &Opts| {
        let k = firsts.iter().position(|&f| f == st.layers.0).unwrap_or(0);
        let c = cachings.get(k).or(cachings.last());
        match fam {
            "Muse Spark" => super::muse::load_stage_hb(&owned, o, st, c),
            "Kimi-K3" => super::kimi::load_with_caching(&owned, o, st, None, c),
            "GLM-5.3-Flash" => super::glm::load_stage_hb(&owned, o, level, st, c),
            _ => super::deepseek_v4::load_stage_hb(&owned, o, st, c),
        }
    });
    Run::new(nl, eos, cuts, devices, o, make)
}

/// Kimi-K3 with every layer across `devices` (src/dynamic/tp.rs), `places` of a layer's experts in each GPU's slices of its places, the rest in the expert
/// cache's host tier (src/dynamic/tp_cache.rs).
pub fn open_kimi_tp(dir: &Path, o: &Opts, devices: &[i32], places: usize) -> Result<Run> {
    let c = crate::kimi::ConfigKimi::load(dir)?;
    let owned = dir.to_path_buf();
    Ok(Run::Tp(super::tp::Tp::with(
        c.layers,
        c.eos.clone(),
        devices,
        o,
        std::sync::Arc::new(move |st, o: &Opts| super::kimi::load_with_places(&owned, o, st, Some(places))),
    )?))
}

/// The layer a tensor's name is of (`...layers.12....`).
pub(crate) fn layer_of(name: &str) -> Option<usize> {
    let rest = &name[name.find("layers.")? + 7..];
    rest.split('.').next()?.parse().ok()
}

/// Where to cut layers of `per` bytes (the rest, `base`, half with the first stage and half with the last) over GPUs with `room` bytes each (ordinal, room):
/// the one with the most room where it holds the whole model, else consecutive layers over the GPUs in order, each GPU's share of the bytes as its
/// share of the room. None: they do not hold it.
pub fn plan_cuts(per: &[u64], base: u64, room: &[(i32, u64)]) -> Option<(Vec<usize>, Vec<i32>)> {
    let total = per.iter().sum::<u64>() + base;
    let best = room.iter().max_by_key(|r| r.1)?;
    if total <= best.1 {
        return Some((Vec::new(), vec![best.0]));
    }
    cut_by_share(per, base, room)
}

/// The layers over every GPU of `room`, in order, each GPU's share of the bytes as its share of the room (`plan_cuts` where one GPU does not hold them).
fn cut_by_share(per: &[u64], base: u64, room: &[(i32, u64)]) -> Option<(Vec<usize>, Vec<i32>)> {
    let total = per.iter().sum::<u64>() + base;
    let all = room.iter().map(|r| r.1).sum::<u64>().max(1);
    let (head, tail) = (base / 2, base - base / 2);
    let (mut cuts, mut devices, mut l) = (Vec::new(), Vec::new(), 0usize);
    for (k, &(dev, cap)) in room.iter().enumerate() {
        if l == per.len() {
            break;
        }
        let (start, last) = (l, k + 1 == room.len());
        let target = (total as u128 * cap as u128 / all as u128) as u64;
        let mut held = if devices.is_empty() { head } else { 0 };
        while l < per.len() {
            let add = per[l] + if l + 1 == per.len() { tail } else { 0 };
            if held + add > cap || (!last && held >= target) {
                break;
            }
            held += add;
            l += 1;
        }
        if l > start {
            if !devices.is_empty() {
                cuts.push(start);
            }
            devices.push(dev);
        }
    }
    (l == per.len()).then_some((cuts, devices))
}

#[cfg(test)]
mod tests {
    use super::{fit, layer_bytes, layer_of, plan_cuts, TWELVE_PCT};

    #[test]
    fn cuts_follow_the_room() {
        let per = vec![10u64; 8];
        assert_eq!(plan_cuts(&per, 4, &[(0, 50), (1, 100)]), Some((vec![], vec![1]))); // (the freest holds it all)
        assert_eq!(plan_cuts(&per, 4, &[(0, 50), (1, 50)]), Some((vec![4], vec![0, 1]))); // (even: 2 + 40 and 40 + 2)
        assert_eq!(plan_cuts(&per, 4, &[(0, 30), (1, 30)]), None); // (84 bytes, 60 of room)
        assert_eq!(plan_cuts(&per, 4, &[(0, 5), (1, 45), (2, 45)]), Some((vec![4], vec![1, 2]))); // (no layer fits beside the embedding on GPU 0)
        assert_eq!(layer_of("model.layers.12.mlp.experts.3.w1.weight"), Some(12));
        assert_eq!(layer_of("lm_head.weight"), None);
    }

    /// The matrices a layer in the computer's memory keeps there (src/dense_ram.rs) are the ones its family's loader puts there: Muse Spark's attention and latent projections, Kimi-K3's delta attention's (not its latent
    /// attention's, whose blocks keep their own), none of DeepSeek-V4's.
    #[test]
    fn the_matrices_a_layer_keeps_in_memory_are_its_loaders() {
        let m = |n: &str| super::in_ram("Muse Spark", &[], n);
        assert!(m("model.language_model.layers.3.self_attn.q_proj.weight") && m("model.language_model.layers.3.self_attn.o_proj.weight"));
        assert!(m("model.language_model.layers.0.mlp.pre_expert_proj.weight") && m("model.language_model.layers.0.mlp.post_expert_proj.weight"));
        assert!(!m("model.language_model.layers.0.self_attn.q_norm.weight") && !m("model.language_model.layers.0.mlp.experts.gate_up_proj"));
        assert!(!m("model.language_model.layers.0.mlp.gate.weight") && !m("model.language_model.embed_tokens.weight"));
        let kda = [true, true, true, false];
        let k = |n: &str| super::in_ram("Kimi-K3", &kda, n);
        assert!(k("language_model.model.layers.2.self_attn.f_a_proj.weight") && k("language_model.model.layers.0.self_attn.o_proj.weight"));
        assert!(!k("language_model.model.layers.3.self_attn.o_proj.weight"), "a latent attention layer's block keeps its own");
        assert!(
            !k("language_model.model.layers.1.self_attn.q_conv1d.weight") && !k("language_model.model.layers.1.block_sparse_moe.shared_experts.up_proj.weight")
        );
        assert!(!super::in_ram("DeepSeek-V4", &[], "model.layers.0.self_attn.wo.weight"));
    }

    /// The expert cache's plan: the dense bytes cut over every GPU (even where one would hold them), each stage's room after them as places for its experts.
    #[test]
    fn places_follow_the_room_left() {
        use super::{places_plan, places_plan_dense};
        // (8 layers of 10 dense bytes, the first with no experts, 4 bytes besides; 2 GPUs of 100: 42 and 42 dense, 58 left each for 3 and 4 expert layers of 2-byte experts)
        let (dense, moe) = (vec![10u64; 8], (0..8).map(|l| l > 0).collect::<Vec<_>>());
        assert_eq!(places_plan(&dense, 4, &moe, &[(0, 100), (1, 100)], 2, 64, 4), Some((vec![4], vec![0, 1], vec![9, 7])));
        // (at most the experts a layer has; fewer than `least` places: none)
        assert_eq!(places_plan(&dense, 4, &moe, &[(0, 100), (1, 100)], 2, 5, 4).map(|p| p.2), Some(vec![5, 5]));
        assert_eq!(places_plan(&dense, 4, &moe, &[(0, 50), (1, 50)], 2, 64, 4), None);
        // (the same places for every layer first: 3 GPUs of 100 for 8 layers of 10, 7 of them with experts; 11 places a layer (32 bytes) fit them as 3, 3 and 2
        // layers, 12 do not; then each GPU's room left over: 68 for 2 expert layers, 70 for 3, 78 for 2)
        assert_eq!(places_plan(&dense, 4, &moe, &[(0, 100), (1, 100), (2, 100)], 2, 64, 4), Some((vec![3, 6], vec![0, 1, 2], vec![17, 11, 19])));
        // (one GPU too small for a token's experts besides the layers' weights: the last layers' dense matrices in the computer's memory, as few as fit, and one slot of the largest on the GPU)
        let ram = vec![3u64; dense.len()];
        assert_eq!(
            places_plan_dense(&dense, 4, &moe, &[(0, 100)], 2, 64, 4, &ram).map(|p| p.3),
            places_plan(&dense, 4, &moe, &[(0, 100)], 2, 64, 4).map(|_| 0)
        );
        let tight = dense.iter().sum::<u64>() + 4 + moe.iter().filter(|&&m| m).count() as u64 * 4 * 2 - 7;
        let got = places_plan_dense(&dense, 4, &moe, &[(0, tight)], 2, 64, 4, &ram).expect("fits with dense layers in the computer's memory");
        assert!(places_plan(&dense, 4, &moe, &[(0, tight)], 2, 64, 4).is_none());
        assert!(got.3 >= 1 && got.4 == 3 * got.3 as u64 - 3, "{got:?}");
        assert_eq!(places_plan_dense(&dense, 4, &moe, &[(0, 10)], 2, 64, 4, &ram), None);
    }

    /// An FP8 release on HBM GPUs: plain where that fits, coded where only that does, no cut where neither; coded alone on other GPUs.
    #[test]
    fn fp8_is_plain_where_it_fits_and_coded_where_only_that_does() {
        let dir = std::env::temp_dir().join(format!("glyd-fit-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        // (two layers of one FP8 matrix each, 256 x 1,024 bytes, and a bf16 head of 256 x 256)
        let header = r#"{"model.layers.0.mlp.w.weight":{"dtype":"F8_E4M3","shape":[256,1024],"data_offsets":[0,262144]},"model.layers.1.mlp.w.weight":{"dtype":"F8_E4M3","shape":[256,1024],"data_offsets":[262144,524288]},"lm_head.weight":{"dtype":"BF16","shape":[256,256],"data_offsets":[524288,655360]}}"#;
        let mut f = (header.len() as u64).to_le_bytes().to_vec();
        f.extend(header.as_bytes());
        f.resize(f.len() + 655360, 0);
        std::fs::write(dir.join("model.safetensors"), f).unwrap();
        let (raw, base) = layer_bytes(&dir, 2, 100).unwrap();
        assert_eq!((raw, base), (vec![262144, 262144], 131072 * 72 / 100));
        let (plain, coded) = (2 * (262144 * 113 / 100) + base, 2 * (262144 * 90 / 100) + base);
        let one = |room: u64, hbm: bool| {
            fit(&dir, 2, &[(0, room)], hbm, false).map(|(b, c, switched, _)| (b.0.iter().sum::<u64>() + b.1, c.is_some(), switched)).unwrap()
        };
        assert_eq!(one(plain, true), (plain, true, false));
        assert_eq!(one(plain - 1, true), (coded, true, true));
        assert_eq!(one(coded - 1, true), (coded, false, false));
        assert_eq!(one(coded, false), (coded, true, false));
        // (a loader that takes the 12-bit layout: the bf16 head in it where it fits, else tiered, plain FP8 first)
        let base12 = 131072 * TWELVE_PCT / 100;
        let (plain12, coded12) = (plain - base + base12, coded - base + base12);
        let one12 = |room: u64| {
            fit(&dir, 2, &[(0, room)], true, true).map(|(b, c, switched, twelve)| (b.0.iter().sum::<u64>() + b.1, c.is_some(), switched, twelve)).unwrap()
        };
        assert_eq!(one12(plain12), (plain12, true, false, true));
        assert_eq!(one12(plain12 - 1), (plain, true, false, false));
        assert_eq!(one12(plain - 1), (coded12, true, true, true));
        assert_eq!(one12(coded12 - 1), (coded, true, true, false));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
