//! Cost profiles (GLY-236, docs/cost-profile.md): per model and level, what each layer holds on the GPU and what a decoded token reads, from the packed
//! directory's own files (config.json, glyd.json and the safetensors headers: the bytes the loader uploads, pack by pack), a measurement of this
//! machine's memory tiers (`glyd-engine rates`: the GPU's memory, the link to it, the computer's memory, a disk), and a predictor of decode speed
//! over them: a token's time is the sum, over the tiers, of the bytes it reads from each over that tier's rate. `glyd-engine costs MODEL[:level]`
//! prints the profile; `glyd serve` puts it in /v1/glyd/info as `costs`.

use crate::ffi::{err, Error, Result};
use crate::nemotron::{ConfigNemotronH, NhKind};
use crate::qwen35::{Config35, Kind, Pack35, Tensor35};
use crate::tools::obj;
use crate::weights::Layout;
use glyd_gpu::json::{self, Value};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

/// What the JSON says it is (a change that breaks a reader gets a new number).
pub const SCHEMA: &str = "glyd-costs/1";

// ---- the levels and layouts ------------------------------------------------------------------------------------------------------------------------------------------------------

pub use crate::levels::name_of as level_of;
use crate::levels::{layout_of, split_tag};

// ---- where the files are ---------------------------------------------------------------------------------------------------------------------------------------------------------

/// A model's files: a directory (given, or the Hub cache's snapshot of the repo), else the repo on the Hub, read by ranges (its small files and the safetensors headers alone).
struct Found {
    id: String,
    tag: Option<String>,
    dir: Option<PathBuf>,
    repo: Option<String>,
    revision: Option<String>,
    remote: Option<(crate::hub::Hub, crate::hub::Remote)>,
}

fn find(model: &str) -> Result<Found> {
    let (name, tag) = split_tag(model);
    let snapshot = |dir: &Path| -> (Option<String>, Option<String>) {
        let repo = dir
            .parent()
            .filter(|p| p.file_name().is_some_and(|n| n == "snapshots"))
            .and_then(Path::parent)
            .and_then(|p| p.file_name()?.to_str()?.strip_prefix("models--").map(|r| r.replacen("--", "/", 1)));
        let rev = repo.as_ref().and(dir.file_name().and_then(|n| n.to_str()).map(str::to_string));
        (repo, rev)
    };
    if Path::new(name).is_dir() {
        let dir = PathBuf::from(name);
        let (repo, revision) = snapshot(&dir);
        return Ok(Found { id: model.to_string(), tag, dir: Some(dir), repo, revision, remote: None });
    }
    // a Hub name: its pre-packed repo at the level (glyd/NAME-LEVEL), else the model itself
    let repo = match &tag {
        Some(t) => crate::levels::packed_repo(name, t),
        None => name.to_string(),
    };
    if let Ok(s) = glyd_gpu::save::Source::find(&repo) {
        return Ok(Found { id: model.to_string(), tag, dir: Some(s.dir), repo: Some(repo), revision: s.revision, remote: None });
    }
    let hub = crate::hub::Hub::from_env();
    let (repo, remote) = match hub.list(&repo, "main") {
        Ok(r) => (repo, r),
        // (no pre-packed repo at the level: the model itself, its sizes estimated as the loader would pack it)
        Err(e) if tag.is_some() && repo != name => (name.to_string(), hub.list(name, "main").map_err(|e2| Error(format!("{repo}: {e}; {name}: {e2}")))?),
        Err(e) => return err(format!("{repo}: {e}")),
    };
    let revision = Some(remote.commit.clone());
    Ok(Found { id: model.to_string(), tag, dir: None, repo: Some(repo), revision, remote: Some((hub, remote)) })
}

impl Found {
    fn text(&self, file: &str) -> Option<String> {
        match (&self.dir, &self.remote) {
            (Some(d), _) => std::fs::read_to_string(d.join(file)).ok(),
            (None, Some((hub, r))) => r.files.iter().any(|e| e.path == file).then(|| hub.text(&r.repo, &r.commit, file).ok()).flatten(),
            _ => None,
        }
    }

    /// Every tensor of the checkpoint's safetensors files and its bytes in the file (None: the files cannot be read, or there are none).
    fn table(&self) -> Option<HashMap<String, u64>> {
        let mut t = HashMap::new();
        match (&self.dir, &self.remote) {
            (Some(d), _) => {
                let ck = glyd_gpu::safetensors::Checkpoint::open(d).ok()?;
                for f in &ck.files {
                    for (name, info) in &f.tensors {
                        t.insert(name.clone(), info.bytes);
                    }
                }
            }
            (None, Some((hub, r))) => {
                for e in r.files.iter().filter(|e| e.path.ends_with(".safetensors") && !e.path.contains('/')) {
                    let (h, _) = hub.header(r, e).ok()?;
                    for (name, _, _, a, b) in h {
                        t.insert(name, b.saturating_sub(a));
                    }
                }
            }
            _ => return None,
        }
        (!t.is_empty()).then_some(t)
    }
}

// ---- the families -----------------------------------------------------------------------------------------------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum LayerKind {
    FullAttention,
    LinearAttention,
    Mamba,
    Moe,
}

impl LayerKind {
    pub fn name(self) -> &'static str {
        match self {
            LayerKind::FullAttention => "full_attention",
            LayerKind::LinearAttention => "linear_attention",
            LayerKind::Mamba => "mamba",
            LayerKind::Moe => "moe",
        }
    }
}

/// The families this reads: what each says of its layers, its packs and the tensors it keeps as they are (the engine's own lists: src/qwen35.rs, src/nemotron.rs).
enum Family {
    Qwen(Config35),
    Nemotron(ConfigNemotronH),
}

impl Family {
    fn parse(text: &str, generation: Option<&str>, saved_causal: bool) -> Result<Family> {
        if crate::nemotron::is_nemotron_h(text) {
            return Ok(Family::Nemotron(ConfigNemotronH::parse(text, "config.json", generation)?));
        }
        if crate::gemma4::is_gemma4(text) {
            return err("a Gemma 4 model: its cost profile is not described yet (Qwen3.5-family and Nemotron-H models are)");
        }
        let mut c = Config35::parse(text, "config.json", generation)
            .map_err(|e| Error(format!("costs: {} (Qwen3.5-family and Nemotron-H models are described)", e.msg)))?;
        if saved_causal {
            // (a glyd-v7 save holds the text model under the causal LM's names, whatever its config.json says of the checkpoint it came from, and no vision tower)
            (c.prefix, c.vision) = ("model.", false);
        }
        Ok(Family::Qwen(c))
    }

    fn name(&self) -> &'static str {
        match self {
            Family::Qwen(_) => "Qwen3.5",
            Family::Nemotron(_) => "Nemotron-H",
        }
    }

    fn layers(&self) -> Vec<LayerKind> {
        match self {
            Family::Qwen(c) => c.kinds.iter().map(|k| if *k == Kind::Full { LayerKind::FullAttention } else { LayerKind::LinearAttention }).collect(),
            Family::Nemotron(c) => c
                .kinds
                .iter()
                .map(|k| match k {
                    NhKind::Mamba => LayerKind::Mamba,
                    NhKind::Attention => LayerKind::FullAttention,
                    NhKind::Moe => LayerKind::Moe,
                })
                .collect(),
        }
    }

    fn packs(&self) -> Vec<Pack35> {
        match self {
            Family::Qwen(c) => c.packs(),
            Family::Nemotron(c) => c.packs(),
        }
    }

    fn kept(&self) -> Vec<Tensor35> {
        match self {
            Family::Qwen(c) => c.kept(),
            Family::Nemotron(c) => c.kept(),
        }
    }

    /// (hidden, vocab, max_pos, tied head)
    fn dims(&self) -> (usize, usize, usize, bool) {
        match self {
            Family::Qwen(c) => (c.hidden, c.vocab, c.max_pos, c.tie),
            Family::Nemotron(c) => (c.hidden, c.vocab, c.max_pos, c.tie),
        }
    }

    /// A token's keys and values in one full-attention layer (bf16), and one recurrent layer's state a sequence holds.
    fn per_layer(&self) -> (u64, u64) {
        match self {
            Family::Qwen(c) => ((2 * c.kv_heads * c.head_dim * 2) as u64, c.state_bytes() / c.linear_layers().max(1) as u64),
            Family::Nemotron(c) => ((2 * c.kv_heads * c.head_dim * 2) as u64, c.state_bytes() / c.layers_of(NhKind::Mamba).max(1) as u64),
        }
    }

    /// (experts a layer, the experts a token takes) where the model has routed experts.
    fn moe(&self) -> Option<(usize, usize)> {
        match self {
            Family::Nemotron(c) if c.layers_of(NhKind::Moe) > 0 => Some((c.moe.experts, c.moe.top_k)),
            _ => None,
        }
    }

    /// The bytes a pack takes at `level` where the files do not say (packed as the loader would pack it: plan.rs `sizing::bits` a weight in its layout).
    fn estimate(&self, level: Layout, p: &Pack35) -> Result<u64> {
        let weights = (p.rows() * p.cols) as u64;
        match self {
            Family::Nemotron(c) if level == Layout::Hummingbird && p.key.contains(".mixer.experts.") => Ok(crate::moe_hb::expert_bytes_mean(c)? / 2),
            Family::Nemotron(_) => Ok((weights as f64 * crate::plan::sizing::bits(crate::nemotron::pack_layout(level, &p.key)) / 8.0) as u64),
            Family::Qwen(_) => Ok((weights as f64 * crate::plan::sizing::bits(level) / 8.0) as u64),
        }
    }

    fn state_type(&self) -> &'static str {
        match self {
            Family::Qwen(_) => "fp32 (the delta rule's), bf16 (the convolution's)",
            Family::Nemotron(_) => "fp32 (the scan's), bf16 (the convolution's)",
        }
    }
}

// ---- what a tensor is -------------------------------------------------------------------------------------------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Comp {
    /// A layer's mixer matrices: attention's q/k/v and o, linear attention's and Mamba-2's in and out projections, a mixture layer's latent projections.
    Proj,
    /// The layer's norms (input, post-attention, q and k).
    Norm,
    /// A dense MLP.
    Mlp,
    /// One routed expert (its number).
    Expert(usize),
    /// A shared expert.
    Shared,
    /// A router and its bias.
    Router,
    /// A recurrence's parameters: the convolution, A_log, D, dt_bias, the gated norm.
    Recurrent,
    /// A layer's tensor of no class above (a rotation's signs).
    Other,
    Embed,
    FinalNorm,
    Head,
    /// Not loaded by the engine: the vision tower, the MTP head.
    Excluded,
}

/// A tensor's or pack's layer and class, from its name.
fn classify(name: &str) -> (Option<usize>, Comp) {
    if name.starts_with("mtp.") || name.contains("visual.") {
        return (None, Comp::Excluded);
    }
    let Some(rest) = name.split("layers.").nth(1) else {
        let c = if name == "embed" || name.contains("embed_tokens") || name.contains("embeddings") {
            Comp::Embed
        } else if name.starts_with("lm_head") {
            Comp::Head
        } else if name.ends_with("norm.weight") || name.ends_with("norm_f.weight") {
            Comp::FinalNorm
        } else {
            Comp::Other
        };
        return (None, c);
    };
    let Some((l, rest)) = rest.split_once('.') else { return (None, Comp::Other) };
    let Ok(layer) = l.parse::<usize>() else { return (None, Comp::Other) };
    let c = if let Some(e) = rest.strip_prefix("mixer.experts.") {
        match e.split('.').next().and_then(|n| n.parse::<usize>().ok()) {
            Some(n) => Comp::Expert(n),
            None => Comp::Other,
        }
    } else if rest.contains("glyd_") {
        // (a save's own tensor of the layer: the Hummingbird rotation's signs)
        Comp::Other
    } else if rest.starts_with("mixer.shared_experts.") {
        Comp::Shared
    } else if rest.starts_with("mixer.gate.") {
        Comp::Router
    } else if rest.starts_with("mlp.") {
        Comp::Mlp
    } else if rest.contains("conv1d")
        || rest.ends_with("A_log")
        || rest.ends_with("dt_bias")
        || rest == "mixer.D"
        || rest == "linear_attn.norm.weight"
        || rest == "mixer.norm.weight"
    {
        Comp::Recurrent
    } else if rest.ends_with("layernorm.weight") || rest == "norm.weight" || rest.ends_with("q_norm.weight") || rest.ends_with("k_norm.weight") {
        Comp::Norm
    } else {
        Comp::Proj
    };
    (Some(layer), c)
}

// ---- the profile -------------------------------------------------------------------------------------------------------------------------------------------------------------------

#[derive(Clone, Debug, Default)]
pub struct LayerCost {
    pub kind: Option<LayerKind>,
    pub proj: u64,
    pub norm: u64,
    pub mlp: u64,
    pub shared: u64,
    pub router: u64,
    pub recurrent: u64,
    pub other: u64,
    /// Each routed expert's bytes (both its matrices), by its number.
    pub experts: Vec<u64>,
    /// The experts a token takes of them.
    pub top_k: usize,
    pub kv_per_token: u64,
    pub state: u64,
}

impl LayerCost {
    pub fn dense(&self) -> u64 {
        self.proj + self.norm + self.mlp + self.shared + self.router + self.recurrent + self.other
    }

    pub fn experts_all(&self) -> u64 {
        self.experts.iter().sum()
    }

    pub fn expert_mean(&self) -> u64 {
        if self.experts.is_empty() {
            0
        } else {
            self.experts_all() / self.experts.len() as u64
        }
    }

    pub fn resident(&self) -> u64 {
        self.dense() + self.experts_all()
    }

    /// What a decoded token reads of the layer: everything dense, and `top_k` experts of the mean size.
    pub fn active(&self) -> u64 {
        self.dense() + self.top_k as u64 * self.expert_mean()
    }

    fn json(&self, index: usize) -> Value {
        let (min, max) = (self.experts.iter().copied().min().unwrap_or(0), self.experts.iter().copied().max().unwrap_or(0));
        obj(vec![
            ("index", num(index as u64)),
            ("kind", self.kind.map_or(Value::Null, |k| Value::from(k.name()))),
            (
                "resident_bytes",
                obj(vec![
                    ("projections", num(self.proj)),
                    ("norms", num(self.norm)),
                    ("mlp", num(self.mlp)),
                    ("experts", num(self.experts_all())),
                    ("experts_count", num(self.experts.len() as u64)),
                    ("expert_mean", num(self.expert_mean())),
                    ("expert_min", num(min)),
                    ("expert_max", num(max)),
                    ("shared_experts", num(self.shared)),
                    ("router", num(self.router)),
                    ("recurrent", num(self.recurrent)),
                    ("other", num(self.other)),
                    ("total", num(self.resident())),
                ]),
            ),
            ("top_k", num(self.top_k as u64)),
            ("active_bytes_per_token", num(self.active())),
            ("kv_bytes_per_token", num(self.kv_per_token)),
            ("state_bytes_per_session", num(self.state)),
        ])
    }
}

/// A model's costs at a level, and where they came from.
pub struct Profile {
    pub id: String,
    pub repo: Option<String>,
    pub revision: Option<String>,
    pub format: Option<String>,
    pub source: Value,
    pub made_by: Option<String>,
    pub family: &'static str,
    pub requested: Option<String>,
    pub level: &'static str,
    pub layout: Layout,
    pub hidden: usize,
    pub vocab: usize,
    pub max_pos: usize,
    pub layers: Vec<LayerCost>,
    pub embed: u64,
    pub final_norm: u64,
    pub head: u64,
    pub tied: bool,
    pub other: u64,
    pub excluded: Vec<(String, u64)>,
    /// The packs' weights in bf16, and every kept tensor's bytes, as the checkpoint would hold them.
    pub bf16_bytes: u64,
    pub weights_count: u64,
    /// How many packs were sized from the files, and how many estimated from their shapes.
    pub from_files: usize,
    pub estimated: usize,
    pub state_type: &'static str,
}

impl Profile {
    /// The profile of MODEL (a directory, a Hub name with a level tag, a pre-packed repo's name): its files where they are, else the Hub's by ranges.
    pub fn of(model: &str) -> Result<Profile> {
        let f = find(model)?;
        Profile::from_found(&f)
    }

    fn from_found(f: &Found) -> Result<Profile> {
        let config = f.text("config.json").ok_or_else(|| Error(format!("{}: no config.json", f.id)))?;
        let generation = f.text("generation_config.json");
        let manifest = f.text("glyd.json").and_then(|t| json::parse(&t).ok());
        let m_str = |k: &str| manifest.as_ref().and_then(|m| m.get(k)).and_then(Value::as_str).map(str::to_string);
        let causal = |k: &str| {
            manifest.as_ref().and_then(|m| m.get(k)).and_then(Value::as_object).is_some_and(|o| o.iter().any(|(n, _)| n.starts_with("model.layers.")))
        };
        let saved = m_str("format").as_deref().is_some_and(|x| x.starts_with("glyd-v")) && (causal("packs") || causal("tensors"));
        let fam = Family::parse(&config, generation.as_deref(), saved)?;
        // the level: the save's, else the tag's, else lossless
        let layout = match (m_str("layout").as_deref().and_then(layout_of), f.tag.as_deref().and_then(layout_of)) {
            (Some(l), _) => l,
            (None, Some(l)) => l,
            (None, None) => Layout::Tiered,
        };
        if let Some(t) = f.tag.as_deref().filter(|t| layout_of(t).is_none()) {
            return err(format!(":{t} is not a level"));
        }
        let level = m_str("level").map(|s| crate::levels::by_name(&s).map_or("other", |l| l.name)).unwrap_or(level_of(layout));
        let table = f.table();
        // the packs' bytes by the name they are saved under (NAME.glyd_data, .glyd_blocks, .glyd_block_base or .glyd_codebook)
        let mut by_pack: HashMap<String, u64> = HashMap::new();
        if let Some(t) = &table {
            for (name, bytes) in t {
                if let Some((pack, _)) = name.split_once(".glyd_") {
                    *by_pack.entry(pack.to_string()).or_default() += bytes;
                }
            }
        }
        let kinds = fam.layers();
        let (hidden, vocab, max_pos, tied) = fam.dims();
        let (kv_layer, state_layer) = fam.per_layer();
        let moe = fam.moe();
        let mut p = Profile {
            id: f.id.clone(),
            repo: f.repo.clone(),
            revision: f.revision.clone(),
            format: m_str("format"),
            source: manifest.as_ref().and_then(|m| m.get("source")).cloned().unwrap_or(Value::Null),
            made_by: m_str("glyd").map(|v| format!("glyd {v}")),
            family: fam.name(),
            requested: f.tag.clone(),
            level,
            layout,
            hidden,
            vocab,
            max_pos,
            layers: kinds
                .iter()
                .map(|&k| LayerCost {
                    kind: Some(k),
                    experts: if k == LayerKind::Moe { vec![0; moe.map_or(0, |m| m.0)] } else { Vec::new() },
                    top_k: if k == LayerKind::Moe { moe.map_or(0, |m| m.1) } else { 0 },
                    kv_per_token: if k == LayerKind::FullAttention { kv_layer } else { 0 },
                    state: if matches!(k, LayerKind::LinearAttention | LayerKind::Mamba) { state_layer } else { 0 },
                    ..Default::default()
                })
                .collect(),
            embed: 0,
            final_norm: 0,
            head: 0,
            tied,
            other: 0,
            excluded: Vec::new(),
            bf16_bytes: 0,
            weights_count: 0,
            from_files: 0,
            estimated: 0,
            state_type: fam.state_type(),
        };
        let mut seen: HashSet<String> = HashSet::new();
        for pk in fam.packs() {
            let saved: u64 = pk.parts.iter().map(|(n, _)| by_pack.get(Pack35::saved(n)).copied().unwrap_or(0)).sum();
            // (a packed directory that holds a matrix as it is, bf16: glyd-v4's embedding; the engine uploads it as it is)
            let dense: u64 = pk.parts.iter().map(|(n, _)| table.as_ref().and_then(|t| t.get(n)).copied().unwrap_or(0)).sum();
            for (n, _) in &pk.parts {
                seen.insert(Pack35::saved(n).to_string());
            }
            let bytes = if saved > 0 {
                p.from_files += 1;
                saved
            } else if dense > 0 && !by_pack.is_empty() {
                p.from_files += 1;
                dense
            } else {
                // (a model not packed yet: as the loader would pack it at load)
                p.estimated += 1;
                fam.estimate(layout, &pk)?
            };
            p.weights_count += (pk.rows() * pk.cols) as u64;
            p.bf16_bytes += (pk.rows() * pk.cols * 2) as u64;
            p.add(&pk.key, bytes);
        }
        let known: HashSet<String> = fam.kept().iter().map(|t| t.name.clone()).collect();
        for t in fam.kept() {
            // (as the engine holds it: an fp32 parameter four bytes wide whatever the file holds)
            let n: u64 = t.shape.iter().product();
            let runtime = n * if t.dtype == "F32" { 4 } else { 2 };
            if table.as_ref().is_some_and(|tb| !tb.contains_key(&t.name)) && table.as_ref().is_some_and(|tb| tb.keys().any(|k| k.contains(".glyd_"))) {
                // (a tensor the files do not have in a packed directory: a part of a pack saved fused, counted with it)
                continue;
            }
            p.bf16_bytes += n * 2;
            p.add(&t.name, runtime);
        }
        // the files' tensors neither list names: a save's own (a rotation's signs), and what the engine does not load (counted under `excluded`)
        if let Some(t) = &table {
            let mut packs: Vec<&String> = by_pack.keys().filter(|k| !seen.contains(k.as_str())).collect();
            packs.sort();
            for k in packs {
                p.add(k, by_pack[k]);
            }
            let mut names: Vec<&String> =
                t.keys().filter(|n| !n.contains(".glyd_") && !known.contains(n.as_str()) && !seen.contains(Pack35::saved(n))).collect();
            names.sort();
            for n in names {
                p.add(n, t[n]);
            }
        }
        Ok(p)
    }

    /// `bytes` of the tensor or pack `name` counted where it belongs.
    fn add(&mut self, name: &str, bytes: u64) {
        match classify(name) {
            (Some(l), c) if l < self.layers.len() => {
                let ly = &mut self.layers[l];
                match c {
                    Comp::Proj => ly.proj += bytes,
                    Comp::Norm => ly.norm += bytes,
                    Comp::Mlp => ly.mlp += bytes,
                    Comp::Shared => ly.shared += bytes,
                    Comp::Router => ly.router += bytes,
                    Comp::Recurrent => ly.recurrent += bytes,
                    Comp::Expert(e) => {
                        if ly.experts.len() <= e {
                            ly.experts.resize(e + 1, 0);
                        }
                        ly.experts[e] += bytes;
                    }
                    _ => ly.other += bytes,
                }
            }
            (_, Comp::Embed) => self.embed += bytes,
            (_, Comp::Head) => self.head += bytes,
            (_, Comp::FinalNorm) => self.final_norm += bytes,
            (_, Comp::Excluded) => self.excluded.push((name.to_string(), bytes)),
            _ => self.other += bytes,
        }
    }

    pub fn resident(&self) -> u64 {
        self.layers.iter().map(LayerCost::resident).sum::<u64>() + self.non_layer()
    }

    pub fn non_layer(&self) -> u64 {
        self.embed + self.final_norm + self.head + self.other
    }

    /// What a decoded token reads of the whole model: every layer's active bytes, the head (the embedding where they are tied), one row of the embedding and the final norm.
    pub fn active(&self) -> u64 {
        self.layers.iter().map(LayerCost::active).sum::<u64>() + self.non_layer_active()
    }

    pub fn non_layer_active(&self) -> u64 {
        (if self.tied { self.embed } else { self.head }) + self.embed / self.vocab.max(1) as u64 + self.final_norm
    }

    pub fn kv_per_token(&self) -> u64 {
        self.layers.iter().map(|l| l.kv_per_token).sum()
    }

    pub fn state(&self) -> u64 {
        self.layers.iter().map(|l| l.state).sum()
    }

    /// The bytes a token reads from where the routed experts are (the mean expert, `top_k` of them a layer), and from everything else.
    pub fn active_split(&self) -> (u64, u64) {
        let experts: u64 = self.layers.iter().map(|l| l.top_k as u64 * l.expert_mean()).sum();
        (experts, self.active() - experts)
    }

    pub fn experts_resident(&self) -> u64 {
        self.layers.iter().map(LayerCost::experts_all).sum()
    }

    pub fn bits_per_weight(&self) -> f64 {
        if self.weights_count == 0 {
            return 0.0;
        }
        // (the packs alone: the kept tensors are not weights of a matrix)
        let packs: u64 = self.resident() - self.layers.iter().map(|l| l.norm + l.recurrent + l.router + l.other).sum::<u64>() - self.final_norm - self.other;
        packs as f64 * 8.0 / self.weights_count as f64
    }

    /// The whole profile as JSON. `work`: the workload the numbers that depend on one are given at; `alloc`: what a loaded engine holds (measured), where one is loaded;
    /// `predictions`: the predictor's rows and the tiers they used, where asked.
    pub fn json(&self, work: &Workload, alloc: Option<&Allocated>, predictions: Option<(Value, Value)>) -> Value {
        let s = |t: &str| Value::from(t);
        let opt = |t: &Option<String>| t.as_deref().map_or(Value::Null, Value::from);
        let (kv_at_ctx, state) = (self.kv_per_token() * work.ctx as u64, self.state());
        let working = alloc.map_or(Value::Null, |a| num(a.working));
        let engine_at_ctx = self.resident() + kv_at_ctx + state + alloc.map_or(0, |a| a.working);
        let sizes_basis = if self.estimated == 0 {
            "file"
        } else if self.from_files == 0 {
            "estimated"
        } else {
            "mixed"
        };
        let (prediction_rows, tiers) = predictions.map_or((Value::Null, Value::Null), |(p, t)| (p, t));
        let decode = match &prediction_rows {
            Value::Array(rows) => rows.first().and_then(|r| r.get("tok_s")).cloned().map_or(Value::Null, |v| {
                obj(vec![("value", v), ("basis", s("estimated")), ("placement", rows[0].get("placement").cloned().unwrap_or(Value::Null))])
            }),
            _ => Value::Null,
        };
        obj(vec![
            ("object", s("glyd.costs")),
            ("schema", s(SCHEMA)),
            ("build", obj(vec![("version", s(env!("CARGO_PKG_VERSION"))), ("commit", option_env!("GLYD_SOURCE_COMMIT").filter(|c| !c.is_empty()).map_or(Value::Null, Value::from)), ("development", Value::Bool(cfg!(feature = "dev")))])),
            (
                "model",
                obj(vec![
                    ("id", s(&self.id)),
                    ("repo", opt(&self.repo)),
                    ("revision", opt(&self.revision)),
                    ("family", s(self.family)),
                    ("format", opt(&self.format)),
                    ("layout", s(self.layout.name())),
                    ("source", self.source.clone()),
                    ("made_by", opt(&self.made_by)),
                    ("hidden", num(self.hidden as u64)),
                    ("vocab", num(self.vocab as u64)),
                    ("max_context", num(self.max_pos as u64)),
                    ("layers", num(self.layers.len() as u64)),
                ]),
            ),
            ("level", obj(vec![("requested", opt(&self.requested)), ("effective", s(self.level)), ("lossless", Value::Bool(matches!(self.layout, Layout::Tiered | Layout::Twelve | Layout::Int4H)))])),
            (
                "basis",
                obj(vec![
                    ("sizes", s(sizes_basis)),
                    ("packs_from_files", num(self.from_files as u64)),
                    ("packs_estimated", num(self.estimated as u64)),
                    ("kv_and_state", s("formula (the engine allocates the same: checked against /v1/glyd/info's counters on the record)")),
                    ("working_memory", s(if alloc.is_some() { "measured" } else { "unavailable" })),
                    ("note", s("sizes are the bytes the loader uploads: the packs as the files hold them, the kept tensors as the kernels take them (fp32 ones four bytes wide)")),
                ]),
            ),
            (
                "precision",
                obj(vec![
                    ("weights", s(&format!("{} level, layout {}: {:.2} bits per weight over the packed matrices", self.level, self.layout.name(), self.bits_per_weight()))),
                    ("bits_per_weight", numf(self.bits_per_weight(), 3)),
                    ("activations", s("bf16")),
                    ("sums", s("fp32")),
                    ("kv", s("bf16")),
                    ("state", s(self.state_type)),
                ]),
            ),
            ("device", device_json()),
            ("host", host_json()),
            ("workload", work.json()),
            ("observed_at", numf(crate::monitor::clock(), 3)),
            ("samples", num(0)),
            ("layers", Value::Array(self.layers.iter().enumerate().map(|(i, l)| l.json(i)).collect())),
            (
                "non_layer",
                obj(vec![
                    ("embeddings_bytes", num(self.embed)),
                    ("final_norm_bytes", num(self.final_norm)),
                    ("head_bytes", num(self.head)),
                    ("head_tied", Value::Bool(self.tied)),
                    ("other_bytes", num(self.other)),
                    ("active_bytes_per_token", num(self.non_layer_active())),
                ]),
            ),
            (
                "totals",
                obj(vec![
                    ("resident_bytes", num(self.resident())),
                    ("resident_layer_bytes", num(self.resident() - self.non_layer())),
                    ("resident_expert_bytes", num(self.experts_resident())),
                    ("active_bytes_per_token", num(self.active())),
                    ("active_expert_bytes_per_token", num(self.active_split().0)),
                    ("active_dense_bytes_per_token", num(self.active_split().1)),
                    ("kv_bytes_per_token", num(self.kv_per_token())),
                    ("kv_bytes_at_context", num(kv_at_ctx)),
                    ("state_bytes_per_session", num(state)),
                    ("working_bytes", working),
                    ("overhead_bytes", Value::Null),
                    ("engine_bytes_at_context", num(engine_at_ctx)),
                    ("bf16_bytes", num(self.bf16_bytes)),
                    ("weights", num(self.weights_count)),
                ]),
            ),
            ("allocated", alloc.map_or(Value::Null, Allocated::json)),
            (
                "excluded",
                Value::Array(
                    [("vision tower", "model.visual.*: not loaded by the engine"), ("MTP head", "mtp.*: not loaded by the engine"), ("draft model", "none for this family")]
                        .into_iter()
                        .map(|(what, note)| {
                            let bytes: u64 = self.excluded.iter().filter(|(n, _)| (what == "vision tower" && n.contains("visual.")) || (what == "MTP head" && n.starts_with("mtp."))).map(|x| x.1).sum();
                            obj(vec![("what", s(what)), ("bytes_in_files", num(bytes)), ("note", s(note))])
                        })
                        .collect(),
                ),
            ),
            ("timing", obj(vec![("prefill_tok_s", Value::Null), ("decode_tok_s", decode)])),
            ("predictions", prediction_rows),
            ("tiers", tiers),
        ])
    }
}

/// The workload the context-dependent numbers are given at.
pub struct Workload {
    pub ctx: usize,
    pub prompt: usize,
    pub output: usize,
    pub chunk: usize,
}

impl Workload {
    fn json(&self) -> Value {
        obj(vec![
            ("context_tokens", num(self.ctx as u64)),
            ("prompt_tokens", num(self.prompt as u64)),
            ("output_tokens", num(self.output as u64)),
            ("prefill_chunk_tokens", num(self.chunk as u64)),
            ("sessions", num(1)),
        ])
    }

    /// The positions a decode step's attention reads, on average over the reply: the prompt and half the output.
    pub fn positions(&self) -> u64 {
        (self.prompt + self.output / 2) as u64
    }
}

/// What a loaded engine holds on the GPU (src/dev.rs `Held`, and the live count): the measured side of the profile.
#[derive(Clone, Copy, Debug)]
pub struct Allocated {
    pub weights: u64,
    pub kv: u64,
    pub state: u64,
    pub working: u64,
    pub engine: u64,
    pub ctx: usize,
}

impl Allocated {
    pub fn now(h: crate::dev::Held, ctx: usize) -> Allocated {
        let live = crate::dev::held_now().0;
        Allocated { weights: h.weights, kv: h.kv, state: h.state, working: live.saturating_sub(h.weights + h.kv + h.state), engine: live, ctx }
    }

    fn json(&self) -> Value {
        obj(vec![
            ("basis", Value::from("measured (the engine's own counters at load)")),
            ("weights_bytes", num(self.weights)),
            ("kv_bytes", num(self.kv)),
            ("kv_bytes_per_token", num(self.kv / self.ctx.max(1) as u64)),
            ("state_bytes", num(self.state)),
            ("working_bytes", num(self.working)),
            ("engine_bytes", num(self.engine)),
            ("context_tokens", num(self.ctx as u64)),
        ])
    }
}

fn num(n: u64) -> Value {
    Value::from(n)
}

fn numf(x: f64, places: usize) -> Value {
    if !x.is_finite() {
        return Value::Null;
    }
    Value::Number(format!("{x:.places$}"))
}

fn device_json() -> Value {
    if NO_GPU.load(std::sync::atomic::Ordering::Relaxed) {
        return Value::Null;
    }
    let Ok(g) = crate::dev::Dev::probe(0) else { return Value::Null };
    crate::dev::Dev::release(0);
    obj(vec![
        ("ordinal", num(g.ordinal.max(0) as u64)),
        ("name", Value::from(g.name.as_str())),
        ("compute_capability", Value::from(format!("{}.{}", g.cc.0, g.cc.1).as_str())),
        ("cuda", Value::from(format!("{}.{}", g.driver / 1000, g.driver % 1000 / 10).as_str())),
        ("driver", crate::cli::driver_version().map_or(Value::Null, |v| Value::from(v.as_str()))),
        ("total_bytes", num(g.total)),
        ("free_bytes", num(g.free)),
    ])
}

fn host_json() -> Value {
    let mem = crate::os::meminfo();
    let kb = |key: &str| {
        mem.as_ref()
            .and_then(|m| m.lines().find(|l| l.starts_with(key)))
            .and_then(|l| l.split_whitespace().nth(1))
            .and_then(|v| v.parse::<u64>().ok())
            .map(|v| v * 1024)
    };
    let cpu = std::fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|t| t.lines().find(|l| l.starts_with("model name")).and_then(|l| l.split_once(':')).map(|x| x.1.trim().to_string()));
    obj(vec![
        ("cpu", cpu.map_or(Value::Null, |c| Value::from(c.as_str()))),
        ("threads", num(std::thread::available_parallelism().map_or(0, |n| n.get() as u64))),
        ("ram_total_bytes", kb("MemTotal:").map_or(Value::Null, num)),
        ("ram_available_bytes", kb("MemAvailable:").map_or(Value::Null, num)),
    ])
}

// ---- the tiers --------------------------------------------------------------------------------------------------------------------------------------------------------------------

/// Bytes a second each tier gives (0: not measured): the GPU's memory as a kernel streams it, the link to the GPU (pinned host memory), the computer's memory as threads stream
/// it, a disk as the SSD tier reads it (direct, 16 reads at once).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Rates {
    pub vram: f64,
    pub link: f64,
    pub ram: f64,
    pub ssd: f64,
}

impl Rates {
    /// The rates of a `tiers` JSON (its `rates` object: the GPU's memory as the engine's products read it where that was measured, else the streaming read).
    pub fn from_json(v: &Value) -> Rates {
        let r = v.get("rates").unwrap_or(v);
        let gbs = |k: &str| {
            r.get(k).and_then(|x| x.get("gbs")).and_then(|x| if let Value::Number(n) = x { n.parse::<f64>().ok() } else { None }).unwrap_or(0.0) * 1e9
        };
        let vram = match gbs("vram_product") {
            x if x > 0.0 => x,
            _ => gbs("vram_read"),
        };
        Rates { vram, link: gbs("link_h2d_pinned"), ram: gbs("ram_read"), ssd: gbs("ssd_read_16") }
    }

    fn json(&self) -> Value {
        obj(vec![
            ("vram_gbs", numf(self.vram / 1e9, 1)),
            ("link_gbs", numf(self.link / 1e9, 1)),
            ("ram_gbs", numf(self.ram / 1e9, 1)),
            ("ssd_gbs", numf(self.ssd / 1e9, 2)),
        ])
    }
}

/// `tiers [--mib N] [--reps N] [--dir DIR] [--no-gpu]`: this machine's tiers measured (N MiB buffers, `reps` passes each): the GPU's memory streamed by a kernel and copied within it,
/// the link to the GPU from pinned and from pageable memory and back, the computer's memory streamed by all its threads and by one, and (with a directory of safetensors files) random
/// 2 MiB reads of them as the SSD tier makes them, 16 at once and one at a time; with a glyd-v7 directory, the engine's own products over its first layers' packs at one row (the rate a
/// decode step reads weights at). One JSON document on stdout.
pub fn tiers_cmd(a: &crate::args::Args) -> Result<()> {
    let (mib, reps) = (a.num("mib", 512usize)?.max(1), a.num("reps", 10usize)?.max(1));
    let dir = a.flag("dir").map(PathBuf::from);
    let bytes = mib << 20;
    let mut rates = Vec::new();
    let row = |gbs: f64, moved: u64, secs: f64, samples: usize, how: &str| {
        obj(vec![("gbs", numf(gbs, 2)), ("bytes", num(moved)), ("seconds", numf(secs, 4)), ("samples", num(samples as u64)), ("how", Value::from(how))])
    };
    if !a.has("no-gpu") {
        use crate::dev::{copy_d2d_async, copy_d2h_async, copy_h2d_async, Buf, Dev, Pinned};
        let dev = Dev::new(0)?;
        let k = crate::kern::Kern::load(&dev)?;
        let buf = Buf::zeroed(bytes)?;
        let out = Buf::zeroed(4)?;
        let blocks = (dev.sms as u32) * 16;
        let timed = |f: &dyn Fn() -> Result<()>| -> Result<f64> {
            f()?;
            dev.sync()?;
            let t0 = std::time::Instant::now();
            for _ in 0..reps {
                f()?;
            }
            dev.sync()?;
            Ok(t0.elapsed().as_secs_f64())
        };
        // SAFETY: live buffers of the sizes given, on the device's stream, synchronized by `timed`.
        let secs = timed(&|| unsafe { k.read_stream(dev.stream, buf.ptr, (bytes / 16) as u64, out.ptr, blocks) })?;
        rates.push((
            "vram_read",
            row(
                bytes as f64 * reps as f64 / secs / 1e9,
                (bytes * reps) as u64,
                secs,
                reps,
                "a kernel that reads the buffer and sums it (the ceiling a memory-bound decode step has)",
            ),
        ));
        let buf2 = Buf::zeroed(bytes)?;
        let secs = timed(&|| unsafe { copy_d2d_async(buf2.ptr, buf.ptr, bytes, dev.stream) })?;
        rates.push((
            "vram_copy",
            row(
                bytes as f64 * reps as f64 / secs / 1e9,
                (bytes * reps) as u64,
                secs,
                reps,
                "cuMemcpyDtoD of the buffer: bytes copied a second (read and written both)",
            ),
        ));
        let pinned = Pinned::new(bytes)?;
        let secs = timed(&|| unsafe { copy_h2d_async(buf.ptr, pinned.ptr, bytes, dev.stream) })?;
        rates.push((
            "link_h2d_pinned",
            row(
                bytes as f64 * reps as f64 / secs / 1e9,
                (bytes * reps) as u64,
                secs,
                reps,
                "cuMemcpyHtoD from pinned host memory (what the expert cache's copies use)",
            ),
        ));
        let secs = timed(&|| unsafe { copy_d2h_async(pinned.ptr, buf.ptr, bytes, dev.stream) })?;
        rates
            .push(("link_d2h_pinned", row(bytes as f64 * reps as f64 / secs / 1e9, (bytes * reps) as u64, secs, reps, "cuMemcpyDtoH into pinned host memory")));
        let pageable = vec![1u8; bytes];
        let secs = timed(&|| unsafe { copy_h2d_async(buf.ptr, pageable.as_ptr() as *const std::ffi::c_void, bytes, dev.stream) })?;
        rates.push((
            "link_h2d_pageable",
            row(bytes as f64 * reps as f64 / secs / 1e9, (bytes * reps) as u64, secs, reps, "cuMemcpyHtoD from pageable memory (the driver stages it)"),
        ));
        drop(pageable);
        if let Some(d) = dir.as_deref().filter(|d| d.join("glyd.json").exists()) {
            match product_rate(&dev, d, reps) {
                Ok(Some((gbs, moved, secs, packs))) => rates.push(("vram_product", row(gbs, moved, secs, reps, &format!("the engine's product at one row over {packs} packs of the directory's first layers, back to back (what a decode step reads weights at)")))),
                Ok(None) => {}
                Err(e) => eprintln!("rates: the products over {}: {e}", d.display()),
            }
        }
    }
    // the computer's memory: every thread streams its share of a buffer, then one thread alone
    let words = bytes / 8;
    let host: Vec<u64> = (0..words as u64).collect();
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    for (name, t) in [("ram_read", threads), ("ram_read_1_thread", 1)] {
        let sum_once = |t: usize| -> u64 {
            let per = words.div_ceil(t);
            std::thread::scope(|s| {
                let hs: Vec<_> = host.chunks(per.max(1)).map(|c| s.spawn(move || c.iter().fold(0u64, |a, &x| a.wrapping_add(x)))).collect();
                hs.into_iter().map(|h| h.join().unwrap_or(0)).fold(0u64, |a, x| a.wrapping_add(x))
            })
        };
        std::hint::black_box(sum_once(t));
        let t0 = std::time::Instant::now();
        for _ in 0..reps {
            std::hint::black_box(sum_once(t));
        }
        let secs = t0.elapsed().as_secs_f64();
        rates.push((name, row(bytes as f64 * reps as f64 / secs / 1e9, (bytes * reps) as u64, secs, reps, &format!("{t} thread(s) summing a buffer of u64"))));
        // (the sum keeps the reads honest)
    }
    drop(host);
    if let Some(d) = &dir {
        for (name, depth, reads, r) in [("ssd_read_16", 16usize, 256usize, reps), ("ssd_read_1", 1, 32, 1)] {
            match disk_rate(d, depth, reads, r) {
                Ok((gbs, moved, secs, mode)) => rates.push((
                    name,
                    row(
                        gbs,
                        moved,
                        secs,
                        r,
                        &format!("random 2 MiB spans of the directory's safetensors files, {depth} at a time, {mode} reads (src/disk.rs, the disk's reader)"),
                    ),
                )),
                Err(e) => eprintln!("rates: the disk under {}: {e}", d.display()),
            }
        }
    }
    let v = obj(vec![
        ("object", Value::from("glyd.tiers")),
        ("schema", Value::from(SCHEMA)),
        ("device", device_json()),
        ("host", host_json()),
        ("directory", dir.as_ref().map_or(Value::Null, |d| Value::from(d.to_string_lossy().as_ref()))),
        ("buffer_bytes", num(bytes as u64)),
        ("observed_at", numf(crate::monitor::clock(), 3)),
        ("rates", obj(rates)),
    ]);
    println!("{}", json::to_python(&v, 2));
    Ok(())
}

/// The engine's product at one row over the first layers' packs of a glyd-v7 directory, back to back, so that the GPU's cache holds none of them: GB/s of weights read; None where the directory's
/// packs are not in a block layout the product takes.
fn product_rate(dev: &crate::dev::Dev, dir: &Path, reps: usize) -> Result<Option<(f64, u64, f64, usize)>> {
    use crate::weights::{read_v7, upload_packed, v7_manifest, PackBuf};
    let Some((layout, manifest)) = v7_manifest(dir)? else { return Ok(None) };
    if !matches!(layout, Layout::Int6 | Layout::Int5 | Layout::Int4 | Layout::Int4H | Layout::Int3) {
        return Ok(None);
    }
    let ck = glyd_gpu::safetensors::Checkpoint::open(dir).map_err(|e| Error(e.to_string()))?;
    let lib = crate::ffi::Lib::find()?;
    let packs = manifest.get("packs").and_then(Value::as_object).ok_or_else(|| Error("glyd.json: no packs".into()))?;
    // (the first four layers' packs, in the save's order; the embedding and the head left out)
    let mut keys: Vec<(&str, u64, u64)> = packs
        .iter()
        .filter_map(|(k, e)| {
            let l = k.split("layers.").nth(1)?.split('.').next()?.parse::<usize>().ok()?;
            let sh = e.get("shape").and_then(Value::as_u64s)?;
            (l < 4 && sh.len() == 2 && e.get("layout").and_then(Value::as_str).and_then(layout_of) == Some(layout)).then_some((k.as_str(), sh[0], sh[1]))
        })
        .collect();
    keys.sort();
    if keys.is_empty() {
        return Ok(None);
    }
    let mats: Vec<PackBuf> =
        keys.iter().map(|&(k, rows, cols)| upload_packed(read_v7(&ck, &manifest, layout, &[(k, rows as usize)], cols as usize)?)).collect::<Result<_>>()?;
    let bytes: u64 = mats.iter().map(|m| m.bytes as u64).sum();
    let (rows, cols) = (keys.iter().map(|k| k.1).max().unwrap_or(1) as usize, keys.iter().map(|k| k.2).max().unwrap_or(1) as usize);
    let (x, y) = (crate::dev::Buf::zeroed(cols * 2)?, crate::dev::Buf::zeroed(rows * 2)?);
    let need =
        mats.iter().map(|m| lib.linear_workspace(m.pack.is_twelve(), m.pack.rows, m.pack.cols, 1)).collect::<Result<Vec<_>>>()?.into_iter().max().unwrap_or(16);
    let (ws, done) = (crate::dev::Buf::zeroed(need.max(16))?, crate::dev::Buf::zeroed(4 << 18)?);
    let run = || -> Result<()> {
        for m in &mats {
            // SAFETY: live buffers of the sizes the product reads and writes.
            unsafe { lib.linear(&m.pack, x.ptr, 1, y.ptr, ws.ptr, ws.bytes, done.ptr, dev.stream)? };
        }
        Ok(())
    };
    run()?;
    dev.sync()?;
    let t0 = std::time::Instant::now();
    for _ in 0..reps {
        run()?;
    }
    dev.sync()?;
    let secs = t0.elapsed().as_secs_f64();
    Ok(Some((bytes as f64 * reps as f64 / secs / 1e9, bytes * reps as u64, secs, mats.len())))
}

/// Random 2 MiB spans of the safetensors files under `dir`, `depth` reads at once by the disk's reader (direct where the file system takes it), `reads` a pass and `reps` passes: GB/s, the bytes,
/// the seconds and the reads' mode.
fn disk_rate(dir: &Path, depth: usize, reads: usize, reps: usize) -> Result<(f64, u64, f64, &'static str)> {
    use crate::disk::{Aligned, DirectFile, Job, Pool};
    use std::cell::RefCell;
    use std::sync::Arc;
    const SPAN: u64 = 2 << 20;
    let mut files: Vec<(Arc<DirectFile>, u64)> = Vec::new();
    let mut names: Vec<PathBuf> = std::fs::read_dir(dir)
        .map_err(|e| Error(format!("{}: {e}", dir.display())))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
        .collect();
    names.sort();
    for p in names {
        let len = std::fs::metadata(&p).map_err(|e| Error(e.to_string()))?.len();
        if len >= 2 * SPAN {
            files.push((Arc::new(DirectFile::open(&p).map_err(|e| Error(format!("{}: {e}", p.display())))?), len));
        }
    }
    if files.is_empty() {
        return err(format!("{}: no safetensors file of 4 MiB or more", dir.display()));
    }
    thread_local! {
        static BOUNCE: RefCell<Aligned> = const { RefCell::new(Aligned::new()) };
    }
    let mut x = 0x9E37_79B9_7F4A_7C15u64;
    let mut next = move || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        x
    };
    let pool = Pool::new(depth.max(1));
    let (mut moved, mut secs) = (0u64, 0.0);
    for _ in 0..reps.max(1) {
        let jobs: Vec<Job> = (0..reads)
            .map(|_| {
                let (f, len) = &files[(next() % files.len() as u64) as usize];
                let at = (next() % ((len - SPAN) / 4096)) * 4096;
                let f = f.clone();
                Box::new(move || BOUNCE.with(|b| f.read_span(at, SPAN as usize, &mut b.borrow_mut()).map(|s| std::hint::black_box(s.len())).map(|_| ()))) as Job
            })
            .collect();
        let t0 = std::time::Instant::now();
        pool.run(jobs).map_err(|e| Error(e.to_string()))?;
        secs += t0.elapsed().as_secs_f64();
        moved += reads as u64 * SPAN;
    }
    Ok((moved as f64 / secs / 1e9, moved, secs, files[0].0.mode()))
}

// ---- the predictor ------------------------------------------------------------------------------------------------------------------------------------------------------------------

/// The bytes a decoded token reads from each tier.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Traffic {
    pub vram: f64,
    pub link: f64,
    pub ram: f64,
    pub ssd: f64,
}

/// Each tier's seconds a token: its bytes over its rate; None where a tier read has no rate.
fn tier_seconds(t: Traffic, r: Rates) -> Option<[f64; 4]> {
    let mut out = [0.0; 4];
    for (i, (bytes, rate)) in [(t.vram, r.vram), (t.link, r.link), (t.ram, r.ram), (t.ssd, r.ssd)].into_iter().enumerate() {
        if bytes > 0.0 {
            if rate <= 0.0 {
                return None;
            }
            out[i] = bytes / rate;
        }
    }
    Some(out)
}

/// Tokens a second where a token's time is the sum over the tiers of its bytes there over the tier's rate (the tiers one after another, nothing overlapped: the issue's model); None where a tier it reads
/// has no rate, or nothing is read.
pub fn predict(t: Traffic, r: Rates) -> Option<f64> {
    let s: f64 = tier_seconds(t, r)?.iter().sum();
    (s > 0.0).then(|| 1.0 / s)
}

/// The same with the GPU's reads and the computer's memory's overlapped (the expert cache's CPU mode computes the missed experts while the GPU runs the layer's others): the longer of the two, then the
/// link's and the disk's copies, which a layer waits for.
pub fn predict_overlapped(t: Traffic, r: Rates) -> Option<f64> {
    let [v, l, m, d] = tier_seconds(t, r)?;
    let s = v.max(m) + l + d;
    (s > 0.0).then(|| 1.0 / s)
}

/// Where the weights are: every tier's bytes a token for a profile, by placement.
#[derive(Clone, Copy, Debug)]
pub struct Placement {
    /// The share of the routed experts the GPU holds (an expert cache's places over the experts), and the share of a token's expert reads found there (the cache's hit rate; the share itself where no better is known).
    pub gpu_share: f64,
    pub hit: f64,
    /// The share of the routed experts the computer's memory holds (the rest stay on the disk).
    pub ram_share: f64,
}

/// (name, what a token reads from each tier) for each way of placing the model: all of it in the GPU's memory; the routed experts in the computer's memory with a cache of them on the GPU, a missed
/// expert computed by the CPU where it is (`cpu`) or copied in (`copy`); the same with the experts the memory does not hold read from the disk (`ssd`); and every weight streamed from the disk each token.
pub fn placements(p: &Profile, w: &Workload, pl: Placement) -> Vec<(&'static str, Traffic)> {
    let (experts, dense) = p.active_split();
    let (experts, dense) = (experts as f64, dense as f64);
    let fixed = dense + (p.kv_per_token() * w.positions()) as f64 + 2.0 * p.state() as f64;
    let mut out = vec![("vram", Traffic { vram: fixed + experts, ..Default::default() })];
    if experts > 0.0 {
        let hit = pl.hit.clamp(0.0, 1.0);
        let miss = experts * (1.0 - hit);
        out.push(("experts_ram_cpu", Traffic { vram: fixed + experts * hit, ram: miss, ..Default::default() }));
        out.push(("experts_ram_copy", Traffic { vram: fixed + experts, link: miss, ..Default::default() }));
        // (of the misses, the ones the memory does not hold come from the disk: the experts off the GPU are in memory in the share the memory holds of them, less the GPU's own)
        let off_gpu = (1.0 - pl.gpu_share).max(1e-9);
        let on_disk = ((1.0 - pl.ram_share) / off_gpu).clamp(0.0, 1.0);
        out.push(("experts_ssd_cpu", Traffic { vram: fixed + experts * hit, ram: miss * (1.0 - on_disk), ssd: miss * on_disk, ..Default::default() }));
    }
    out.push(("stream_ssd", Traffic { vram: fixed + experts, link: dense + experts, ssd: dense + experts, ram: 0.0 }));
    out
}

/// The prediction rows for the JSON: each placement's tokens a second from `rates`, the bytes a token by tier, and what was assumed.
pub fn prediction_json(p: &Profile, w: &Workload, pl: Placement, rates: Rates, first: &str) -> Value {
    let mut rows: Vec<(&str, Traffic)> = placements(p, w, pl);
    if let Some(i) = rows.iter().position(|r| r.0 == first) {
        rows.rotate_left(i);
    }
    Value::Array(
        rows.into_iter()
            .map(|(name, t)| {
                let (tok_s, over) = (predict(t, rates), predict_overlapped(t, rates));
                obj(vec![
                    ("placement", Value::from(name)),
                    ("tok_s", tok_s.map_or(Value::Null, |x| numf(x, 2))),
                    ("ms_per_token", tok_s.map_or(Value::Null, |x| numf(1000.0 / x, 3))),
                    ("tok_s_overlapped", over.map_or(Value::Null, |x| numf(x, 2))),
                    (
                        "bytes_per_token",
                        obj(vec![("vram", num(t.vram as u64)), ("link", num(t.link as u64)), ("ram", num(t.ram as u64)), ("ssd", num(t.ssd as u64))]),
                    ),
                    ("basis", Value::from("estimated")),
                    (
                        "assumes",
                        obj(vec![
                            ("positions_read", num(w.positions())),
                            ("gpu_share_of_experts", numf(pl.gpu_share, 3)),
                            ("hit", numf(pl.hit, 3)),
                            ("ram_share_of_experts", numf(pl.ram_share, 3)),
                        ]),
                    ),
                ])
            })
            .collect(),
    )
}

/// `costs MODEL[:level] [--ctx N] [--prompt N] [--new N] [--chunk N] [--predict [--tiers FILE] [--placement NAME] [--gpu-experts F] [--hit F] [--ram-experts F]] [--load] [--no-gpu]`: the profile of the model
/// at the level, one JSON document on stdout. `--predict`: the predictor's rows from the tiers of `--tiers FILE` (a `tiers` document), else the GPU's memory, its link and the computer's memory measured now (256 MiB,
/// 5 passes). `--load`: the model loaded on the GPU (a Qwen3.5-family one), its allocation measured and set beside the profile.
pub fn costs_cmd(a: &crate::args::Args) -> Result<()> {
    let model = a.need(1, "MODEL[:level]")?;
    let v = costs_json(model, a)?;
    println!("{}", json::to_python(&v, 2));
    Ok(())
}

/// `--no-gpu`: the GPU is not asked anything (the `device` field is null): a profile from the files alone, on a machine whose GPU is busy or absent.
static NO_GPU: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// What `costs` prints, as a value (`preflight --costs` prints it too).
pub fn costs_json(model: &str, a: &crate::args::Args) -> Result<Value> {
    NO_GPU.store(a.has("no-gpu"), std::sync::atomic::Ordering::Relaxed);
    let p = Profile::of(model)?;
    let w = Workload {
        ctx: a.either("context", "ctx", 4096usize)?,
        prompt: a.num("prompt", 1024usize)?,
        output: a.num("new", 256usize)?,
        chunk: a.num("chunk", 2048usize)?,
    };
    let alloc = if a.has("load") {
        let f = find(model)?;
        let dir = f.dir.ok_or_else(|| Error(format!("{model}: --load needs the model's directory")))?;
        if p.family != "Qwen3.5" {
            return err(format!("--load: {} is loaded by `glyd-engine serve`, whose /v1/glyd/info has the allocation (the costs object)", p.family));
        }
        let o = crate::dynamic::Opts { ctx: w.ctx, chunk: w.chunk, graph: true, device: 0, layout: None };
        let e = crate::dynamic::qwen35::load(&dir, &o)?;
        let held = crate::conv::Gen::held(&e).ok_or_else(|| Error("the engine did not say what it holds".into()))?;
        Some(Allocated::now(held, crate::conv::Gen::capacity(&e)))
    } else {
        None
    };
    let predictions = if a.has("predict") {
        let rates = match a.flag("rates") {
            Some(f) => {
                Rates::from_json(&json::parse(&std::fs::read_to_string(f).map_err(|e| Error(format!("{f}: {e}")))?).map_err(|e| Error(format!("{f}: {e}")))?)
            }
            None => quick_rates()?,
        };
        let gpu_share = a.num("gpu-experts", 1.0f64)?;
        let pl = Placement { gpu_share, hit: a.num("hit", gpu_share)?, ram_share: a.num("ram-experts", 1.0f64)? };
        Some((prediction_json(&p, &w, pl, rates, a.flag("placement").unwrap_or("vram")), rates.json()))
    } else {
        None
    };
    Ok(p.json(&w, alloc.as_ref(), predictions))
}

/// The GPU's memory, its link and the computer's memory measured now, briefly (256 MiB, 5 passes), for a prediction without a tiers file.
fn quick_rates() -> Result<Rates> {
    use crate::dev::{copy_h2d_async, Buf, Dev, Pinned};
    let bytes = 256usize << 20;
    let reps = 5;
    let dev = Dev::new(0)?;
    let k = crate::kern::Kern::load(&dev)?;
    let (buf, out) = (Buf::zeroed(bytes)?, Buf::zeroed(4)?);
    let blocks = (dev.sms as u32) * 16;
    let timed = |f: &dyn Fn() -> Result<()>| -> Result<f64> {
        f()?;
        dev.sync()?;
        let t0 = std::time::Instant::now();
        for _ in 0..reps {
            f()?;
        }
        dev.sync()?;
        Ok(bytes as f64 * reps as f64 / t0.elapsed().as_secs_f64())
    };
    // SAFETY: live buffers of the sizes given, synchronized by `timed`.
    let vram = timed(&|| unsafe { k.read_stream(dev.stream, buf.ptr, (bytes / 16) as u64, out.ptr, blocks) })?;
    let pinned = Pinned::new(bytes)?;
    let link = timed(&|| unsafe { copy_h2d_async(buf.ptr, pinned.ptr, bytes, dev.stream) })?;
    let host: Vec<u64> = (0..(bytes / 8) as u64).collect();
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    let sum = || {
        std::thread::scope(|s| {
            host.chunks((bytes / 8).div_ceil(threads).max(1))
                .map(|c| s.spawn(move || c.iter().fold(0u64, |a, &x| a.wrapping_add(x))))
                .collect::<Vec<_>>()
                .into_iter()
                .map(|h| h.join().unwrap_or(0))
                .fold(0u64, |a, x| a.wrapping_add(x))
        })
    };
    std::hint::black_box(sum());
    let t0 = std::time::Instant::now();
    for _ in 0..reps {
        std::hint::black_box(sum());
    }
    let ram = bytes as f64 * reps as f64 / t0.elapsed().as_secs_f64();
    Ok(Rates { vram, link, ram, ssd: 0.0 })
}

/// The costs object of /v1/glyd/info for the model served from `dir` (never an error: what went wrong, where it did).
pub fn served(dir: &Path, id: &str, requested: Option<&str>, ctx: usize, chunk: usize, held: Option<crate::dev::Held>) -> Value {
    let f = Found { id: id.to_string(), tag: requested.map(str::to_string), dir: Some(dir.to_path_buf()), repo: None, revision: None, remote: None };
    let f = match find(&dir.to_string_lossy()) {
        Ok(found) => Found { id: f.id, tag: f.tag, ..found },
        Err(_) => f,
    };
    match Profile::from_found(&f) {
        Ok(p) => p.json(&Workload { ctx, prompt: 0, output: 0, chunk }, held.map(|h| Allocated::now(h, ctx)).as_ref(), None),
        Err(e) => obj(vec![("object", Value::from("glyd.costs")), ("status", Value::from("unavailable")), ("error", Value::from(e.msg.as_str()))]),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tiny model's names and bytes, as the files would give them: two dense layers (one full attention, one linear attention) and a mixture layer of four experts a token takes two of,
    /// with a shared expert and a router; the embedding, the final norm and a head of its own.
    fn tiny() -> Profile {
        let mut p = Profile {
            id: "tiny".into(),
            repo: None,
            revision: None,
            format: None,
            source: Value::Null,
            made_by: None,
            family: "test",
            requested: None,
            level: "swift",
            layout: Layout::Int5,
            hidden: 8,
            vocab: 100,
            max_pos: 64,
            layers: vec![
                LayerCost { kind: Some(LayerKind::FullAttention), kv_per_token: 16, ..Default::default() },
                LayerCost { kind: Some(LayerKind::LinearAttention), state: 1000, ..Default::default() },
                LayerCost { kind: Some(LayerKind::Moe), experts: vec![0; 4], top_k: 2, ..Default::default() },
            ],
            embed: 0,
            final_norm: 0,
            head: 0,
            tied: false,
            other: 0,
            excluded: Vec::new(),
            bf16_bytes: 0,
            weights_count: 0,
            from_files: 0,
            estimated: 0,
            state_type: "none",
        };
        for (name, bytes) in [
            ("model.layers.0.self_attn.q_proj", 300u64),
            ("model.layers.0.self_attn.o_proj", 100),
            ("model.layers.0.input_layernorm.weight", 2),
            ("model.layers.0.post_attention_layernorm.weight", 2),
            ("model.layers.0.self_attn.q_norm.weight", 1),
            ("model.layers.0.mlp.gate_proj", 400),
            ("model.layers.0.mlp.down_proj", 200),
            ("model.layers.1.linear_attn.in_proj_qkv", 300),
            ("model.layers.1.linear_attn.out_proj", 100),
            ("model.layers.1.linear_attn.conv1d.weight", 10),
            ("model.layers.1.linear_attn.A_log", 4),
            ("model.layers.1.linear_attn.norm.weight", 4),
            ("model.layers.1.mlp.gate_proj", 400),
            ("model.layers.1.mlp.down_proj", 200),
            ("backbone.layers.2.mixer.experts.0.up_proj", 50),
            ("backbone.layers.2.mixer.experts.0.down_proj", 50),
            ("backbone.layers.2.mixer.experts.1.up_proj", 60),
            ("backbone.layers.2.mixer.experts.1.down_proj", 60),
            ("backbone.layers.2.mixer.experts.2.up_proj", 70),
            ("backbone.layers.2.mixer.experts.2.down_proj", 70),
            ("backbone.layers.2.mixer.experts.3.up_proj", 80),
            ("backbone.layers.2.mixer.experts.3.down_proj", 80),
            ("backbone.layers.2.mixer.shared_experts.up_proj", 30),
            ("backbone.layers.2.mixer.gate.weight", 8),
            ("backbone.layers.2.mixer.gate.e_score_correction_bias", 2),
            ("backbone.layers.2.mixer.fc1_latent_proj", 20),
            ("backbone.layers.2.norm.weight", 2),
            ("model.embed_tokens", 1000),
            ("lm_head", 1000),
            ("model.norm.weight", 2),
            ("mtp.layers.0.mlp.down_proj.weight", 777),
        ] {
            p.add(name, bytes);
        }
        p
    }

    #[test]
    fn the_sums_by_layer_and_component() {
        let p = tiny();
        let l0 = &p.layers[0];
        assert_eq!((l0.proj, l0.norm, l0.mlp, l0.resident(), l0.active()), (400, 5, 600, 1005, 1005));
        let l1 = &p.layers[1];
        assert_eq!((l1.proj, l1.recurrent, l1.mlp, l1.resident()), (400, 18, 600, 1018));
        let l2 = &p.layers[2];
        assert_eq!(
            (l2.experts.clone(), l2.experts_all(), l2.expert_mean(), l2.shared, l2.router, l2.proj, l2.norm),
            (vec![100, 120, 140, 160], 520, 130, 30, 10, 20, 2)
        );
        // a token takes two experts of the mean size, and everything dense
        assert_eq!((l2.dense(), l2.active(), l2.resident()), (62, 62 + 2 * 130, 582));
        assert_eq!((p.embed, p.head, p.final_norm, p.non_layer()), (1000, 1000, 2, 2002));
        // the head and one row of the embedding (1000 / 100) are read each token; the MTP head is excluded, not counted
        assert_eq!((p.non_layer_active(), p.excluded.len(), p.excluded[0].1), (1012, 1, 777));
        assert_eq!((p.resident(), p.active(), p.kv_per_token(), p.state()), (1005 + 1018 + 582 + 2002, 1005 + 1018 + 322 + 1012, 16, 1000));
        assert_eq!(p.active_split(), (260, p.active() - 260));
        // tied: the embedding is the head
        let tied = Profile { tied: true, ..tiny() };
        assert_eq!(tied.non_layer_active(), 1012);
    }

    #[test]
    fn names_are_classed_by_what_they_are() {
        let c = |n: &str| classify(n);
        assert_eq!(c("model.layers.3.self_attn.q_proj"), (Some(3), Comp::Proj));
        assert_eq!(c("model.layers.12.linear_attn.in_proj_b.weight"), (Some(12), Comp::Proj));
        assert_eq!(c("backbone.layers.7.mixer.in_proj"), (Some(7), Comp::Proj));
        assert_eq!(c("backbone.layers.7.mixer.fc2_latent_proj"), (Some(7), Comp::Proj));
        assert_eq!(c("backbone.layers.7.mixer.conv1d.bias"), (Some(7), Comp::Recurrent));
        assert_eq!(c("backbone.layers.7.mixer.D"), (Some(7), Comp::Recurrent));
        assert_eq!(c("backbone.layers.7.mixer.norm.weight"), (Some(7), Comp::Recurrent));
        assert_eq!(c("model.layers.0.linear_attn.dt_bias"), (Some(0), Comp::Recurrent));
        assert_eq!(c("model.layers.0.self_attn.k_norm.weight"), (Some(0), Comp::Norm));
        assert_eq!(c("backbone.layers.9.norm.weight"), (Some(9), Comp::Norm));
        assert_eq!(c("backbone.layers.9.mixer.experts.511.down_proj"), (Some(9), Comp::Expert(511)));
        assert_eq!(c("backbone.layers.9.mixer.experts.glyd_rot_signs"), (Some(9), Comp::Other));
        assert_eq!(c("backbone.layers.9.mixer.glyd_rot_signs"), (Some(9), Comp::Other));
        assert_eq!(c("embed"), (None, Comp::Embed));
        assert_eq!(c("backbone.layers.9.mixer.shared_experts.down_proj"), (Some(9), Comp::Shared));
        assert_eq!(c("backbone.layers.9.mixer.gate.weight"), (Some(9), Comp::Router));
        assert_eq!(c("model.language_model.embed_tokens"), (None, Comp::Embed));
        assert_eq!(c("backbone.embeddings"), (None, Comp::Embed));
        assert_eq!(c("lm_head"), (None, Comp::Head));
        assert_eq!(c("backbone.norm_f.weight"), (None, Comp::FinalNorm));
        assert_eq!(c("model.visual.blocks.0.attn.qkv.weight"), (None, Comp::Excluded));
        assert_eq!(c("mtp.layers.0.mlp.down_proj.weight"), (None, Comp::Excluded));
    }

    #[test]
    fn the_predictor_sums_the_tiers() {
        let r = Rates { vram: 1000e9, link: 25e9, ram: 40e9, ssd: 5e9 };
        // 8 GB a token from the GPU's memory alone: 8 ms, 125 tokens a second
        let t = predict(Traffic { vram: 8e9, ..Default::default() }, r).unwrap();
        assert!((t - 125.0).abs() < 1e-9, "{t}");
        // 8 GB from the GPU and 0.5 GB computed from the computer's memory: 8 + 12.5 ms
        let t = predict(Traffic { vram: 8e9, ram: 0.5e9, ..Default::default() }, r).unwrap();
        assert!((1000.0 / t - 20.5).abs() < 1e-9, "{t}");
        // the same 0.5 GB copied over the link, then read on the GPU: 8.5 + 20 ms
        let t = predict(Traffic { vram: 8.5e9, link: 0.5e9, ..Default::default() }, r).unwrap();
        assert!((1000.0 / t - 28.5).abs() < 1e-9, "{t}");
        // a tier without a rate: no prediction; nothing read: none either
        assert!(predict(Traffic { ssd: 1e9, ..Default::default() }, Rates { ssd: 0.0, ..r }).is_none());
        assert!(predict(Traffic::default(), r).is_none());
        // the record's Nemotron case (benchmarks/gpu/nh-hb-cpu-4090-2026-10-10): 244 experts of 2.1 MB a token by the CPU at the kernel's 36 GB/s, the rest on a 4090 at 1,008 GB/s
        let t = predict(Traffic { vram: 8.0e9, ram: 244.0 * 2.1e6, ..Default::default() }, Rates { vram: 1008e9, ram: 36e9, ..r }).unwrap();
        assert!((44.0..47.0).contains(&t), "{t}");
        // overlapped: the GPU's 8 ms under the CPU's 12.5 ms, then the link's 20 ms
        let t = predict_overlapped(Traffic { vram: 8e9, ram: 0.5e9, link: 0.5e9, ..Default::default() }, r).unwrap();
        assert!((1000.0 / t - 32.5).abs() < 1e-9, "{t}");
        assert!(predict_overlapped(Traffic { ram: 1e9, ..Default::default() }, Rates { ram: 0.0, ..r }).is_none());
    }

    #[test]
    fn placements_of_a_mixture_and_of_a_dense_model() {
        let p = tiny();
        let w = Workload { ctx: 64, prompt: 32, output: 16, chunk: 8 };
        let rows = placements(&p, &w, Placement { gpu_share: 0.5, hit: 0.75, ram_share: 0.75 });
        let names: Vec<&str> = rows.iter().map(|r| r.0).collect();
        assert_eq!(names, ["vram", "experts_ram_cpu", "experts_ram_copy", "experts_ssd_cpu", "stream_ssd"]);
        let (experts, dense) = p.active_split();
        let fixed = dense as f64 + (16 * 40) as f64 + 2000.0; // (the keys and values of 32 + 8 positions, the state read and written)
        assert_eq!(rows[0].1, Traffic { vram: fixed + experts as f64, ..Default::default() });
        let miss = experts as f64 * 0.25;
        assert_eq!(rows[1].1, Traffic { vram: fixed + experts as f64 * 0.75, ram: miss, ..Default::default() });
        assert_eq!(rows[2].1, Traffic { vram: fixed + experts as f64, link: miss, ..Default::default() });
        // half the experts are off the GPU; the memory holds three quarters of all of them: half of the misses are on the disk
        assert_eq!(rows[3].1, Traffic { vram: fixed + experts as f64 * 0.75, ram: miss * 0.5, ssd: miss * 0.5, ..Default::default() });
        assert_eq!(rows[4].1, Traffic { vram: fixed + experts as f64, link: p.active() as f64, ssd: p.active() as f64, ram: 0.0 });
        // a dense model has the GPU and the disk placements alone
        let dense_only = Profile { layers: p.layers[..2].to_vec(), ..tiny() };
        assert_eq!(
            placements(&dense_only, &w, Placement { gpu_share: 1.0, hit: 1.0, ram_share: 1.0 }).iter().map(|r| r.0).collect::<Vec<_>>(),
            ["vram", "stream_ssd"]
        );
    }

    #[test]
    fn levels_tags_and_rates() {
        assert_eq!(
            (level_of(Layout::Int5), level_of(Layout::Int6), level_of(Layout::Int4H), level_of(Layout::Hummingbird), level_of(Layout::Tiered)),
            ("swift", "kestrel", "wren", "hummingbird", "penguin")
        );
        assert_eq!(
            (layout_of("swift"), layout_of("i6"), layout_of("hb"), layout_of("mma"), layout_of("x")),
            (Some(Layout::Int5), Some(Layout::Int6), Some(Layout::Hummingbird), Some(Layout::Tiered), None)
        );
        assert_eq!(split_tag("Qwen/Qwen3.5-9B:swift"), ("Qwen/Qwen3.5-9B", Some("swift".to_string())));
        assert_eq!(split_tag("/a/b:c/d"), ("/a/b:c/d", None));
        assert_eq!(split_tag("glyd/Qwen3.5-9B-swift"), ("glyd/Qwen3.5-9B-swift", None));
        let v = json::parse(r#"{"rates":{"vram_read":{"gbs":700.5},"vram_product":{"gbs":600},"link_h2d_pinned":{"gbs":25.0},"ram_read":{"gbs":40},"ssd_read_16":{"gbs":6.5}}}"#).unwrap();
        assert_eq!(Rates::from_json(&v), Rates { vram: 600e9, link: 25e9, ram: 40e9, ssd: 6.5e9 });
        let v = json::parse(r#"{"rates":{"vram_read":{"gbs":700.5}}}"#).unwrap();
        assert_eq!(Rates::from_json(&v), Rates { vram: 700.5e9, link: 0.0, ram: 0.0, ssd: 0.0 });
    }

    #[test]
    fn the_json_has_the_three_fields_and_nulls_where_nothing_is_measured() {
        let p = tiny();
        let v = p.json(&Workload { ctx: 64, prompt: 32, output: 16, chunk: 8 }, None, None);
        let at = |path: &[&str]| path.iter().try_fold(&v, |x, k| x.get(k)).cloned();
        assert_eq!(at(&["totals", "resident_bytes"]).and_then(|x| x.as_u64()), Some(p.resident()));
        assert_eq!(at(&["totals", "active_bytes_per_token"]).and_then(|x| x.as_u64()), Some(p.active()));
        assert_eq!(at(&["totals", "kv_bytes_per_token"]).and_then(|x| x.as_u64()), Some(16));
        assert_eq!(at(&["totals", "kv_bytes_at_context"]).and_then(|x| x.as_u64()), Some(16 * 64));
        assert_eq!(at(&["totals", "state_bytes_per_session"]).and_then(|x| x.as_u64()), Some(1000));
        assert_eq!(at(&["totals", "working_bytes"]), Some(Value::Null));
        assert_eq!(at(&["timing", "decode_tok_s"]), Some(Value::Null));
        assert_eq!(at(&["timing", "prefill_tok_s"]), Some(Value::Null));
        assert_eq!(at(&["basis", "sizes"]).and_then(|x| x.as_str().map(str::to_string)), Some("file".to_string()));
        let layers = at(&["layers"]).and_then(|x| x.as_array().map(|a| a.len()));
        assert_eq!(layers, Some(3));
        assert_eq!(
            v.get("layers").and_then(Value::as_array).and_then(|a| a[2].get("resident_bytes")).and_then(|r| r.get("expert_mean")).and_then(Value::as_u64),
            Some(130)
        );
        // with an allocation: the working memory is measured and the engine's bytes at the context include it
        let a = Allocated { weights: 4607, kv: 1024, state: 1000, working: 500, engine: 7131, ctx: 64 };
        let v = p.json(&Workload { ctx: 64, prompt: 32, output: 16, chunk: 8 }, Some(&a), None);
        assert_eq!(v.get("totals").and_then(|t| t.get("working_bytes")).and_then(Value::as_u64), Some(500));
        assert_eq!(v.get("allocated").and_then(|t| t.get("kv_bytes_per_token")).and_then(Value::as_u64), Some(16));
    }
}
