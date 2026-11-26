//! Model descriptions from `config.json` (R3a, #136): a `ModelDesc` of layer kinds and the weight map that names each block role's
//! checkpoint tensors, with no per-model code. Families: Qwen3.8 dense and MoE (the Qwen3.5 architecture), Qwen3.8-Flash-Next (Condor's
//! model: gated residuals, the indexer on GQA, the hashed n-gram embedding), gemma-4 text, GLM-5.3 and
//! GLM-5.3-Flash (transformers' keys or the release's, FP8 block-scaled matrices), Kimi-K3, Muse Spark, DeepSeek-V4, Nemotron 3
//! (`nemotron_h`), and the Llama-style standard architectures (Llama, Mistral, Qwen2, Qwen3). Weight maps name the released tensors,
//! through the tables of their naming convention (`names`); a format detail such as Kimi-K3's MXFP4 experts (`.weight_packed` for
//! `.weight`) is the loader's. The config comes parsed once (`config::ModelConfig`).
//!
//! A config key the builder does not know fails with the key named; keys that cannot change the math (ids, dtypes, training settings)
//! are listed in `INERT`. A decoder layer is two blocks: the mixer's, then the feed-forward's (a Nemotron layer is one block: its
//! pattern names a mixer or a feed-forward per layer).

use super::config::ModelConfig;
use super::kinds::*;
use super::names::{self, all, part, Convention};
use crate::ffi::{err, Error, Result};
use serde_json::{Map, Value};
use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};

/// Where a role's tensor comes from in the checkpoint.
#[derive(Clone, Debug, PartialEq)]
pub enum Source {
    Tensor(String),
    /// Matrices stacked along their rows (a gate and an up projection as one `gate_up`).
    Rows(Vec<String>),
    /// One entry per expert (each its rows stacked), as [experts, rows, cols].
    Experts(Vec<Vec<String>>),
}

impl Source {
    /// The checkpoint's tensors the source reads.
    pub fn names(&self) -> Vec<String> {
        match self {
            Source::Tensor(n) => vec![n.clone()],
            Source::Rows(v) => v.clone(),
            Source::Experts(v) => v.iter().flatten().cloned().collect(),
        }
    }
}

/// A description and its weight map.
#[derive(Clone, Debug)]
pub struct Described {
    pub desc: ModelDesc,
    pub embed: String,
    pub final_norm: String,
    /// None: the head is the embedding (tied).
    pub head: Option<String>,
    /// Per block, its roles' sources.
    pub blocks: Vec<Vec<(String, Source)>>,
    /// The model's own roles besides the embedding, final norm and head (Kimi-K3's `out_mix_norm`, `out_mix_proj`).
    pub extra: Vec<(String, Source)>,
    /// Draft layers in the checkpoint that the description does not use yet (`mtp_num_hidden_layers`).
    pub mtp_layers: usize,
    /// Matrices stored quantized (FP8 blocks as released, FP4 experts), each with the tensor of its scales. The weight map names them;
    /// decoding a stored format is the tensor source's job.
    pub scales: std::collections::BTreeMap<String, String>,
    /// The FP8 matrices' blocks (the quantization_config's weight_block_size, square), where the config says the release's matrices are FP8 and the
    /// builder reads it: the format a release's inference quantizes its activations to as well (DeepSeek-V4's 128, V4.1's 32).
    pub fp8_block: Option<usize>,
    /// Each tensor's shape as the config makes it, where the builder says it (Gemma 4's: its matrices, norms and scalars; empty for the builders that do not yet):
    /// what a lowering to the executor sizes its matrices by, for a packed directory too, whose tensors are not stored at their shapes.
    pub shapes: std::collections::BTreeMap<String, Vec<usize>>,
    /// The naming convention the weight map is in (`names::Convention::name`).
    pub convention: &'static str,
    /// Parts of the checkpoint the description does not read: the towers and the draft layers, as prefixes of their tensors' names.
    pub unread: Vec<String>,
    /// Per draft head of `desc.mtp` whose tensors are mapped: its blocks' roles' sources, and its own roles' (DSpark's `main_proj`, `markov_embed`, ...).
    pub mtp: Vec<(Vec<Vec<(String, Source)>>, Vec<(String, Source)>)>,
    /// The vision tower's roles (`desc.vision`), where the description reads one.
    pub vision: Option<VisionRoles>,
}

/// A vision tower's roles (`kinds::VisionDesc`): its own (`names::part::VISION`: the patch product, the position table, the merger), then each block's
/// (`VISION_BLOCK`).
#[derive(Clone, Debug, Default)]
pub struct VisionRoles {
    pub own: Vec<(String, Source)>,
    pub blocks: Vec<Vec<(String, Source)>>,
}

impl VisionRoles {
    /// Role `r`'s tensor: the tower's own (`block` None) or block `b`'s.
    pub fn tensor(&self, block: Option<usize>, r: &str) -> Result<&str> {
        let roles = match block {
            None => &self.own,
            Some(b) => self.blocks.get(b).ok_or_else(|| Error(format!("the vision tower has no block {b}")))?,
        };
        match roles.iter().find(|(x, _)| x == r) {
            Some((_, Source::Tensor(n))) => Ok(n),
            _ => err(format!("the vision tower has no tensor for its role {r}")),
        }
    }
}

impl Described {
    /// Every checkpoint tensor the description reads.
    pub fn names(&self) -> BTreeSet<String> {
        let mut s: BTreeSet<String> = [self.embed.clone(), self.final_norm.clone()].into_iter().chain(self.head.clone()).collect();
        for (_, src) in &self.extra {
            s.extend(src.names());
        }
        s.extend(self.scales.values().cloned());
        for b in &self.blocks {
            for (_, src) in b {
                s.extend(src.names());
            }
        }
        for (blocks, own) in &self.mtp {
            for (_, src) in blocks.iter().flatten().chain(own) {
                s.extend(src.names());
            }
        }
        if let Some(v) = &self.vision {
            for (_, src) in v.own.iter().chain(v.blocks.iter().flatten()) {
                s.extend(src.names());
            }
        }
        s
    }

    /// The tensors of its attention blocks' own matrices (a block whose mixer attends over the context: `Mixer::Attention`, `Mla`, `Compressed`), its residual's
    /// and its norms' roles aside, and the LM head where it is not the embedding: the matrices a second layout holds (Condor's server's `--hi`, v0.34.2's
    /// engine's: the long context's precision kept where the rest is at fewer bits).
    pub fn attention_and_head(&self) -> BTreeSet<String> {
        let mut s: BTreeSet<String> = self.head.iter().cloned().collect();
        for (b, roles) in self.desc.blocks.iter().zip(&self.blocks) {
            if !matches!(b.body, Body::Mixer(Mixer::Attention(_) | Mixer::Mla(_) | Mixer::Compressed(_))) {
                continue;
            }
            let aside = |r: &str| r == "norm_in" || r == "norm_out" || b.residual.roles().contains(&r);
            s.extend(roles.iter().filter(|(r, _)| !aside(r)).flat_map(|(_, src)| src.names()));
        }
        s
    }

    /// The routed experts' tensors it reads (the Moe kind's `experts_*` roles, the draft heads' too).
    pub fn routed_experts(&self) -> BTreeSet<String> {
        let heads = self.mtp.iter().flat_map(|(blocks, _)| blocks.iter());
        self.blocks.iter().chain(heads).flatten().filter(|(r, _)| r.starts_with("experts_")).flat_map(|(_, s)| s.names()).collect()
    }

    /// Whether a checkpoint of the tensor names `have` is slim: it holds none of the routed experts the description reads (in any stored form, `stored`),
    /// where it reads some; a codes pack holds them (Glyd's release of Qwen3.8-Flash-Next: `model/` beside its packs).
    pub fn slim(&self, have: &BTreeSet<String>) -> bool {
        let routed = self.routed_experts();
        !routed.is_empty() && !routed.iter().any(|n| stored(have, n, false))
    }

    /// The weight map against a checkpoint's safetensors index (`have`): every tensor it reads is there; with `all`, as the config says it (a quantized
    /// matrix with its scales), and every tensor there is read but the parts the description leaves (`unread`). Without `all`, which a loader asks, a
    /// matrix is there in any stored form (`stored`), its scales are not asked (the format is the loader's to read from the checkpoint), and a checkpoint
    /// may hold none of the routed experts (`slim`: a codes pack holds them, which its loader is given; a loader that reads them from the checkpoint names
    /// them missing). An error names the convention, how many differ and the first of them.
    pub fn check(&self, have: &BTreeSet<String>, all: bool) -> Result<()> {
        let few = |v: &[String]| {
            let (n, first) = (v.len(), v.iter().take(4).cloned().collect::<Vec<_>>().join(", "));
            format!("{n} tensor{}: {first}{}", if n == 1 { "" } else { "s" }, if n > 4 { ", ..." } else { "" })
        };
        let scales: BTreeSet<&String> = self.scales.values().collect();
        let routed = if !all && self.slim(have) { self.routed_experts() } else { BTreeSet::new() };
        let missing: Vec<String> = self.names().into_iter().filter(|n| !stored(have, n, all) && (all || !scales.contains(n)) && !routed.contains(n)).collect();
        if !missing.is_empty() {
            return err(format!("the description reads tensors that are not in the checkpoint ({} names), {}", self.convention, few(&missing)));
        }
        let names = self.names();
        let unread = |n: &str| self.unread.iter().any(|p| p.strip_prefix('*').map_or(n.starts_with(p.as_str()), |sfx| n.ends_with(sfx)));
        let left: Vec<String> = have.iter().filter(|n| !names.contains(*n) && !unread(n)).cloned().collect();
        if all && !left.is_empty() {
            return err(format!("the checkpoint holds tensors not read by its description ({} names), {}", self.convention, few(&left)));
        }
        Ok(())
    }
}

/// Whether `have` holds the tensor `n` a description reads: as named, or (`!all`, a loader's check) in another stored form of a matrix: `.weight`,
/// compressed-tensors' `.weight_packed`, or a table read by rows at 4 bits (`X.q4`, `formats::q4_name`).
fn stored(have: &BTreeSet<String>, n: &str, all: bool) -> bool {
    let base = n.strip_suffix(".weight_packed").or(n.strip_suffix(".weight"));
    have.contains(n)
        || (!all
            && base.is_some_and(|b| have.contains(&format!("{b}.weight")) || have.contains(&format!("{b}.weight_packed")) || have.contains(&format!("{b}.q4"))))
}

/// The description of the model in `dir`, from its config.json alone, and the file's path for the words of an error. A directory in the Hub's format (its
/// safetensors, no glyd.json) is held to its index: a tensor the description reads and the checkpoint lacks is named here, with all of its kind.
pub fn describe_dir(dir: &std::path::Path) -> Result<(Described, String)> {
    describe_dir_of(dir, &ModelConfig::load(dir)?)
}

/// `describe_dir` of a config.json parsed once.
pub fn describe_dir_of(dir: &std::path::Path, c: &ModelConfig) -> Result<(Described, String)> {
    let label = dir.join("config.json").display().to_string();
    let hub = !dir.join("glyd.json").exists() && (dir.join("model.safetensors.index.json").exists() || dir.join("model.safetensors").exists());
    let have = if hub { Some(names::index_names(dir)?) } else { None };
    let d = describe_config_in(c, have.as_ref())?;
    if let Some(have) = &have {
        d.check(have, false).map_err(|e| e.context(&label))?;
    }
    Ok((d, label))
}

/// Keys that cannot change what a step computes.
const INERT: &[&str] = &[
    "architectures",
    "model_type",
    "dtype",
    "torch_dtype",
    "transformers_version",
    "initializer_range",
    "use_cache",
    "pad_token_id",
    "bos_token_id",
    "eos_token_id",
    "attention_dropout",
    "output_router_logits",
    "router_aux_loss_coef",
    "pretraining_tp",
    "_name_or_path",
    "output_attentions",
    "output_hidden_states",
    "return_dict",
    "chunk_size_feed_forward",
    "id2label",
    "label2id",
    "problem_type",
    "tokenizer_class",
    "is_encoder_decoder",
    "tie_encoder_decoder",
    "vocab_size_per_layer_input",
    "max_position_embeddings",
    "auto_map",
    "hidden_dropout",
    // (transformers' PretrainedConfig writes its generation and training defaults into some releases' configs: Kimi-K3's text config has them all; none is the model's shape)
    "add_cross_attention",
    "bad_words_ids",
    "begin_suppress_tokens",
    "cross_attention_hidden_size",
    "diversity_penalty",
    "do_sample",
    "early_stopping",
    "encoder_no_repeat_ngram_size",
    "exponential_decay_length_penalty",
    "finetuning_task",
    "is_decoder",
    "length_penalty",
    "max_length",
    "min_length",
    "no_repeat_ngram_size",
    "num_beam_groups",
    "num_beams",
    "num_return_sequences",
    "output_scores",
    "prefix",
    "pruned_heads",
    "remove_invalid_values",
    "repetition_penalty",
    "return_dict_in_generate",
    "suppress_tokens",
    "task_specific_params",
    "temperature",
    "tf_legacy_loss",
    "top_k",
    "top_p",
    "torchscript",
    "typical_p",
    "use_bfloat16",
    // (the loss's ignored label, and the text a prompt marks an image with: the training's and the processor's, not the language model's)
    "ignore_index",
    "image_placeholder",
];

/// A key that says nothing of the model's shape: one of INERT, or a special token's number (`decoder_start_token_id`, `image_token_index`).
fn inert(k: &str) -> bool {
    INERT.contains(&k) || k.ends_with("_token_id") || k.ends_with("_token_index")
}

/// A config's keys, each read at most through these getters; `finish` names a key nobody read.
struct Keys<'a> {
    m: &'a Map<String, Value>,
    used: RefCell<BTreeSet<String>>,
    what: String,
}

impl<'a> Keys<'a> {
    fn new(m: &'a Map<String, Value>, what: &str) -> Keys<'a> {
        Keys { m, used: RefCell::new(BTreeSet::new()), what: what.to_string() }
    }
    fn raw(&self, k: &str) -> Option<&'a Value> {
        self.used.borrow_mut().insert(k.to_string());
        self.m.get(k).filter(|v| !v.is_null())
    }
    fn usize(&self, k: &str) -> Result<usize> {
        self.opt_usize(k)?.ok_or_else(|| Error(format!("{}: no {k}", self.what)))
    }
    fn opt_usize(&self, k: &str) -> Result<Option<usize>> {
        self.raw(k).map(|v| v.as_u64().map(|n| n as usize).ok_or_else(|| Error(format!("{}: {k} is not a count", self.what)))).transpose()
    }
    fn f64(&self, k: &str) -> Result<f64> {
        self.opt_f64(k)?.ok_or_else(|| Error(format!("{}: no {k}", self.what)))
    }
    fn opt_f64(&self, k: &str) -> Result<Option<f64>> {
        self.raw(k).map(|v| v.as_f64().ok_or_else(|| Error(format!("{}: {k} is not a number", self.what)))).transpose()
    }
    fn bool(&self, k: &str, default: bool) -> Result<bool> {
        self.raw(k).map_or(Ok(default), |v| v.as_bool().ok_or_else(|| Error(format!("{}: {k} is not true or false", self.what))))
    }
    fn str(&self, k: &str) -> Result<Option<&'a str>> {
        self.raw(k).map(|v| v.as_str().ok_or_else(|| Error(format!("{}: {k} is not a string", self.what)))).transpose()
    }
    /// The key must be absent, null, or `want` (a setting the builder does not support otherwise).
    fn expect(&self, k: &str, want: &Value) -> Result<()> {
        match self.raw(k) {
            Some(v) if v != want => err(format!("{}: {k} = {v} is not supported (only {want})", self.what)),
            _ => Ok(()),
        }
    }
    fn finish(&self) -> Result<()> {
        let used = self.used.borrow();
        match self.m.keys().find(|k| !used.contains(*k) && !inert(k)) {
            Some(k) => err(format!("{}: config key {k} is not understood by the description builder", self.what)),
            None => Ok(()),
        }
    }
}

fn act(name: &str) -> Result<Act> {
    Ok(match name {
        "silu" | "swish" => Act::Silu,
        "gelu_pytorch_tanh" | "gelu_tanh" => Act::GeluTanh,
        "gelu" => Act::Gelu,
        "relu2" => Act::Relu2,
        "sigmoid" => Act::Sigmoid,
        a => return err(format!("activation {a} is not supported")),
    })
}

/// A rope from transformers' `rope_parameters` (or the older `rope_theta` + `rope_scaling`) for heads of `head_dim`.
fn rope(p: &Map<String, Value>, head_dim: usize, max_pos: Option<usize>, what: &str) -> Result<Rope> {
    let k = Keys::new(p, what);
    let theta = k.f64("rope_theta")?;
    let partial = k.opt_f64("partial_rotary_factor")?.unwrap_or(1.0);
    let dim = (head_dim as f64 * partial) as usize;
    let ty = k.str("rope_type")?.or(k.str("type")?).unwrap_or("default");
    let orig = || -> Result<u32> { Ok(k.usize("original_max_position_embeddings")? as u32) };
    let mut r = Rope::plain(theta, dim);
    let mrope = match k.raw("mrope_section") {
        Some(Value::Array(a)) if a.len() == 3 => Some([0, 1, 2].map(|i| a[i].as_u64().unwrap_or(0) as usize)),
        Some(v) => return err(format!("{what}: mrope_section {v}")),
        None => None,
    };
    k.bool("mrope_interleaved", false)?;
    r.mrope = mrope;
    r.scaling = match ty {
        "default" => RopeScaling::None,
        "proportional" => {
            r = Rope { mrope, ..Rope::proportional(theta, head_dim, partial) };
            match k.opt_f64("factor")? {
                Some(f) if f != 1.0 => RopeScaling::Linear { factor: f },
                _ => RopeScaling::None,
            }
        }
        "linear" => RopeScaling::Linear { factor: k.f64("factor")? },
        "dynamic" => RopeScaling::DynamicNtk {
            factor: k.f64("factor")?,
            max_positions: max_pos.ok_or_else(|| Error(format!("{what}: dynamic rope without max_position_embeddings")))? as u32,
        },
        "yarn" => RopeScaling::Yarn {
            factor: k.f64("factor")?,
            original_max: orig()?,
            beta_fast: k.opt_f64("beta_fast")?.unwrap_or(32.0),
            beta_slow: k.opt_f64("beta_slow")?.unwrap_or(1.0),
            mscale: k.opt_f64("mscale")?,
            mscale_all_dim: k.opt_f64("mscale_all_dim")?,
            attention_factor: k.opt_f64("attention_factor")?,
            truncate: k.bool("truncate", true)?,
        },
        "llama3" => RopeScaling::Llama3 {
            factor: k.f64("factor")?,
            low_freq_factor: k.f64("low_freq_factor")?,
            high_freq_factor: k.f64("high_freq_factor")?,
            original_max: orig()?,
        },
        "longrope" => {
            let fs = |n: &str| -> Result<Vec<f64>> {
                k.raw(n)
                    .and_then(|v| v.as_array())
                    .map(|a| a.iter().map(|x| x.as_f64().unwrap_or(1.0)).collect())
                    .ok_or_else(|| Error(format!("{what}: no {n}")))
            };
            let max = max_pos.ok_or_else(|| Error(format!("{what}: longrope without max_position_embeddings")))? as u32;
            k.opt_f64("factor")?;
            RopeScaling::LongRope {
                short: fs("short_factor")?,
                long: fs("long_factor")?,
                original_max: orig()?,
                max_positions: max,
                attention_factor: k.opt_f64("attention_factor")?,
            }
        }
        t => return err(format!("{what}: rope_type {t} is not supported")),
    };
    k.finish()?;
    Ok(r)
}

/// The rope of a config without `rope_parameters`: `rope_theta` and the older `rope_scaling`.
fn legacy_rope(k: &Keys, head_dim: usize, max_pos: Option<usize>) -> Result<Rope> {
    let mut m = match k.raw("rope_scaling") {
        Some(Value::Object(o)) => o.clone(),
        _ => Map::new(),
    };
    m.insert("rope_theta".into(), Value::from(k.opt_f64("rope_theta")?.unwrap_or(10_000.0)));
    if let Some(p) = k.opt_f64("partial_rotary_factor")? {
        m.insert("partial_rotary_factor".into(), Value::from(p));
    }
    rope(&m, head_dim, max_pos, &format!("{} rope", k.what))
}

fn gated(act: Act) -> Mlp {
    Mlp { act, gated: true, bias: false, clamp: None }
}

/// The roles of `table` but those in `drop`'s list, under `at` (`experts` of them where a role is each expert's).
fn but(table: names::Roles, at: &str, experts: usize, drop: &[&str]) -> Vec<(String, Source)> {
    names::roles(table, at, experts, |r| !drop.contains(&r))
}

/// Shapes by role: each of `want`'s roles that `r` names as one tensor, that tensor's shape.
fn shape_roles(shapes: &mut std::collections::BTreeMap<String, Vec<usize>>, r: &[(String, Source)], want: &[(&str, Vec<usize>)]) {
    for (role, s) in want {
        match r.iter().find(|(x, _)| x == role) {
            Some((_, Source::Tensor(n))) => {
                shapes.insert(n.clone(), s.clone());
            }
            // (an expert's matrices, each its own tensor of the shape given: a gate and an up projection alike)
            Some((_, Source::Experts(e))) => shapes.extend(e.iter().flatten().map(|n| (n.clone(), s.clone()))),
            _ => {}
        }
    }
}

/// The model types the `llama` builder reads: the pre-norm dense decoders of one attention and one gated MLP a layer.
pub const LLAMA: &[&str] = &["llama", "mistral", "qwen2", "qwen3"];

/// The description of the model whose `config.json` is `cfg`.
pub fn describe(cfg: &Value) -> Result<Described> {
    describe_config(&ModelConfig::from_value(cfg)?)
}

/// The description of the model whose `config.json` is `cfg`, for a checkpoint whose tensor names are `have` (`describe_config_in`).
pub fn describe_held(cfg: &Value, have: &BTreeSet<String>) -> Result<Described> {
    describe_config_in(&ModelConfig::from_value(cfg)?, Some(have))
}

/// Where a checkpoint holds a role's experts as one tensor of them all, [experts, rows, cols] (`held`: its description held to its index, `describe_held`),
/// and `named` (the same model's description from its config alone) names each expert's matrix a tensor (or its rows stacked: an expert's gate and up):
/// each such tensor's name, with the stacked tensor and its rows there (the stacked tensor seen as [experts rows, cols]), expert e's members its rows
/// `e rows..(e + 1) rows` in order, each as many as its own shape says. Every other role, and a checkpoint that holds a tensor an expert's matrix, adds
/// nothing. A shape the two descriptions do not agree on is refused.
pub fn stacked_rows(named: &Described, held: &Described) -> Result<BTreeMap<String, (String, usize, usize)>> {
    if named.blocks.len() != held.blocks.len() {
        return err(format!("{} blocks from the config, {} as the checkpoint holds them", named.blocks.len(), held.blocks.len()));
    }
    let shape = |d: &Described, n: &str| d.shapes.get(n).cloned().ok_or_else(|| Error(format!("the description has no shape for {n}")));
    let mut out = BTreeMap::new();
    for (b, (nr, hr)) in named.blocks.iter().zip(&held.blocks).enumerate() {
        for (role, hs) in hr {
            let (Source::Tensor(t), Some((_, Source::Experts(each)))) = (hs, nr.iter().find(|(r, _)| r == role)) else { continue };
            let [n, rows, cols] = shape(held, t)?[..] else {
                return err(format!("block {b}: {t}: {:?}, where experts stacked are three-dimensional", shape(held, t)?));
            };
            if n != each.len() {
                return err(format!("block {b}: {t} holds {n} experts, where the config has {}", each.len()));
            }
            for (e, members) in each.iter().enumerate() {
                let mut at = e * rows;
                for m in members {
                    match shape(named, m)?[..] {
                        [r, c] if c == cols => {
                            out.insert(m.clone(), (t.clone(), at, at + r));
                            at += r;
                        }
                        ref s => return err(format!("block {b}: {m}: {s:?}, where rows of {t}'s {cols} columns are")),
                    }
                }
                if at != (e + 1) * rows {
                    return err(format!("block {b}: {t}: expert {e}'s matrices take {} of its {rows} rows", at - e * rows));
                }
            }
        }
    }
    Ok(out)
}

/// The description of a config.json parsed once.
pub fn describe_config(c: &ModelConfig) -> Result<Described> {
    describe_config_in(c, None)
}

/// The description of a config.json parsed once as a checkpoint that stacks each layer's experts holds it, where the layout has that form besides a tensor
/// an expert's matrix (the Qwen3.5 layout's: transformers' form, the one Qwen3.8-2.4T-A95B is released in); else the config's own.
pub fn describe_stacked(c: &ModelConfig) -> Result<Described> {
    let first = names::HF.layer(c.prefix(), 0);
    let have: BTreeSet<String> = names::roles(part::QWEN_MOE_STACKED, &first, 0, |r| r == "experts_gate_up").iter().flat_map(|(_, s)| s.names()).collect();
    describe_config_in(c, Some(&have))
}

/// The description of a config.json parsed once, in the stored form of the checkpoint whose tensor names are `have` where its config does not say how it
/// stores them: the Qwen3.5 layout's routed experts are one tensor an expert's matrix (as its earlier releases are saved) unless the checkpoint holds each
/// layer's experts as two stacked tensors (`mlp.experts.gate_up_proj`, `mlp.experts.down_proj`: the form transformers keeps them in, and the one
/// Qwen3.8-2.4T-A95B is released in). With no `have`, the form the config alone gives.
pub fn describe_config_in(c: &ModelConfig, have: Option<&BTreeSet<String>>) -> Result<Described> {
    let (tc, prefix) = (&c.text, c.prefix());
    // (the Kimi-K3 layout by its text model's, not the wrapper's release name: Kimi-K3's wrapper is `kimi_k3`, Kimi-K3.1's `kimi_k31`)
    if c.nested && c.text_type == "kimi_linear" {
        return kimi(c);
    }
    match c.model_type.as_str() {
        "muse_spark" => return muse(tc, prefix),
        "deepseek_v4" => return deepseek_v4(tc),
        "deepseek_v41" => return deepseek_v41(c),
        "nemotron_h" => return nemotron_h(tc),
        _ => {}
    }
    let mt = c.text_type.as_str();
    // (a multimodal config's tie flag can sit at the top)
    let tied_top = c.top.get("tie_word_embeddings").and_then(|v| v.as_bool());
    match mt {
        t if LLAMA.contains(&t) => llama(tc, prefix, mt, tied_top),
        "qwen3_5_text" | "qwen3_5_moe_text" | "qwen3_5" | "qwen3_5_moe" => qwen35(tc, prefix, mt, tied_top, have),
        "qwen4_exp_text" => qwen4x(tc, prefix, mt, tied_top).and_then(|d| with_tower(d, c)),
        "gemma4_text" | "gemma4_unified_text" => gemma4(tc, prefix, mt, tied_top),
        "glm_moe_dsa" | "glm5_next_text" | "glm5_next" => glm(tc, prefix, mt, tied_top, c.top.get("quantization_config").or(tc.get("quantization_config"))),
        "" => err("config.json names no model_type"),
        m => err(format!("model_type {m} has no description builder yet")),
    }
}

fn common(k: &Keys, tied_top: Option<bool>) -> Result<(usize, usize, usize, f32, bool)> {
    let (d, layers, vocab) = (k.usize("hidden_size")?, k.usize("num_hidden_layers")?, k.usize("vocab_size")?);
    let eps = k.opt_f64("rms_norm_eps")?.unwrap_or(1e-6) as f32;
    let tied = k.bool("tie_word_embeddings", tied_top.unwrap_or(false))?;
    Ok((d, layers, vocab, eps, tied))
}

/// The description, its blocks' roles and its own tensors named by `cv` (the text model's under `prefix`), once every key of `k` is read.
fn finish(k: &Keys, desc: ModelDesc, cv: &Convention, prefix: &str, blocks: Vec<Vec<(String, Source)>>, mtp_layers: usize) -> Result<Described> {
    k.finish()?;
    let tied = desc.head.tied;
    Ok(Described {
        desc,
        embed: cv.embed(prefix),
        final_norm: cv.final_norm(prefix),
        head: (!tied).then(|| cv.head(prefix)),
        blocks,
        extra: cv.extra(prefix),
        mtp_layers,
        scales: Default::default(),
        fp8_block: None,
        shapes: Default::default(),
        convention: cv.name,
        unread: cv.unread(),
        mtp: Vec::new(),
        vision: None,
    })
}

fn llama(tc: &Map<String, Value>, prefix: &str, mt: &str, tied_top: Option<bool>) -> Result<Described> {
    let k = Keys::new(tc, mt);
    let (d, layers, vocab, eps, tied) = common(&k, tied_top)?;
    let (heads, kv) = (k.usize("num_attention_heads")?, k.opt_usize("num_key_value_heads")?.unwrap_or(k.usize("num_attention_heads")?));
    let hd = k.opt_usize("head_dim")?.unwrap_or(d / heads);
    let max_pos = k.opt_usize("max_position_embeddings")?;
    let rp = match k.raw("rope_parameters") {
        Some(Value::Object(o)) => rope(o, hd, max_pos, &format!("{mt} rope_parameters"))?,
        _ => legacy_rope(&k, hd, max_pos)?,
    };
    let bias = k.bool("attention_bias", mt == "qwen2")?;
    k.expect("mlp_bias", &Value::Bool(false))?;
    let act = act(k.str("hidden_act")?.unwrap_or("silu"))?;
    let inter = k.usize("intermediate_size")?;
    // sliding windows: Mistral's `sliding_window` for every layer; Qwen2's past `max_window_layers` when `use_sliding_window`
    let window = k.opt_usize("sliding_window")?;
    let qwen_sliding = k.bool("use_sliding_window", mt == "mistral")?;
    let max_window_layers = k.opt_usize("max_window_layers")?.unwrap_or(0);
    let layer_types: Option<Vec<String>> =
        k.raw("layer_types").and_then(|v| v.as_array()).map(|a| a.iter().filter_map(|x| x.as_str().map(String::from)).collect());
    let qk_norm = mt == "qwen3";
    let (mut blocks, mut roles, mut shapes) = (Vec::new(), Vec::new(), std::collections::BTreeMap::new());
    for l in 0..layers {
        let sliding = match &layer_types {
            Some(lt) => lt.get(l).map(|s| s == "sliding_attention").unwrap_or(false),
            None => qwen_sliding && window.is_some() && (mt == "mistral" || l >= max_window_layers),
        };
        let a = Attn {
            heads,
            kv_heads: kv,
            head_dim: hd,
            v_head_dim: hd,
            window: if sliding { window } else { None },
            sinks: false,
            rope: Some(rp.clone()),
            q_norm: qk_norm.then(|| Norm::rms(eps)),
            k_norm: qk_norm.then(|| Norm::rms(eps)),
            v_norm: None,
            gate: None,
            softcap: None,
            bias,
            scale: (hd as f32).powf(-0.5),
            k_eq_v: false,
            indexer: None,
            rounding: AttnRounding::F32,
            kv_from: None,
        };
        let p = names::HF.layer(prefix, l);
        let (ar, fr) = (
            names::roles(part::ATTN, &p, 0, |r| (bias || !r.ends_with("_bias")) && (qk_norm || !matches!(r, "q_norm" | "k_norm"))),
            names::roles(part::MLP, &p, 0, all),
        );
        // (the shapes: each matrix [rows, columns] as the heads and widths make it, each norm and bias its width)
        let (q, kvw) = (heads * hd, kv * hd);
        shape_roles(
            &mut shapes,
            &ar,
            &[
                ("norm_in", vec![d]),
                ("q", vec![q, d]),
                ("q_bias", vec![q]),
                ("k", vec![kvw, d]),
                ("k_bias", vec![kvw]),
                ("v", vec![kvw, d]),
                ("v_bias", vec![kvw]),
                ("o", vec![d, q]),
                ("q_norm", vec![hd]),
                ("k_norm", vec![hd]),
            ],
        );
        shape_roles(&mut shapes, &fr, &[("norm_in", vec![d]), ("gate", vec![inter, d]), ("up", vec![inter, d]), ("down", vec![d, inter])]);
        blocks.push(Block { norm_in: Norm::rms(eps), body: Body::Mixer(Mixer::Attention(a)), norm_out: None, residual: Residual::Add });
        roles.push(ar);
        blocks.push(Block { norm_in: Norm::rms(eps), body: Body::Ffn(Ffn::Mlp(gated(act))), norm_out: None, residual: Residual::Add });
        roles.push(fr);
    }
    let desc = ModelDesc {
        hidden: d,
        vocab,
        embed: Embed { scale: 1.0, norm: None, streams: 1, per_layer: None, table: TableAt::Gpu },
        blocks,
        final_norm: Norm::rms(eps),
        head: Head { tied, softcap: None, scale: 1.0, collapse: Collapse::None, rounding: Rounding::F32 },
        mtp: Vec::new(),
        vision: None,
        rope: Some(rp),
    };
    let mut out = finish(&k, desc, &names::HF, prefix, roles, 0)?;
    shapes.insert(out.embed.clone(), vec![vocab, d]);
    shapes.insert(out.final_norm.clone(), vec![d]);
    if let Some(h) = &out.head {
        shapes.insert(h.clone(), vec![vocab, d]);
    }
    out.shapes = shapes;
    Ok(out)
}

fn qwen35(tc: &Map<String, Value>, prefix: &str, mt: &str, tied_top: Option<bool>, have: Option<&BTreeSet<String>>) -> Result<Described> {
    let k = Keys::new(tc, mt);
    let (d, layers, vocab, eps, tied) = common(&k, tied_top)?;
    let (heads, kv, hd) = (k.usize("num_attention_heads")?, k.usize("num_key_value_heads")?, k.usize("head_dim")?);
    let max_pos = k.opt_usize("max_position_embeddings")?;
    k.opt_f64("partial_rotary_factor")?;
    let rp = match k.raw("rope_parameters") {
        Some(Value::Object(o)) => rope(o, hd, max_pos, &format!("{mt} rope_parameters"))?,
        _ => legacy_rope(&k, hd, max_pos)?,
    };
    // (the routed experts' stored form, which the config does not say: each expert's matrices a tensor, unless the checkpoint holds the layer's experts stacked)
    let first = names::HF.layer(prefix, 0);
    let stacked = have.is_some_and(|h| {
        names::roles(part::QWEN_MOE_STACKED, &first, 0, |r| r == "experts_gate_up").iter().all(|(_, s)| s.names().iter().all(|n| h.contains(n)))
    });
    k.expect("attention_bias", &Value::Bool(false))?;
    let gate = k.bool("attn_output_gate", true)?;
    let lin_gate = act(k.str("output_gate_type")?.unwrap_or("swish"))?;
    let act = act(k.str("hidden_act")?.unwrap_or("silu"))?;
    k.expect("mamba_ssm_dtype", &Value::from("float32"))?;
    k.opt_usize("full_attention_interval")?;
    let types: Vec<String> = k
        .raw("layer_types")
        .and_then(|v| v.as_array())
        .ok_or_else(|| Error(format!("{mt}: no layer_types")))?
        .iter()
        .map(|x| x.as_str().unwrap_or("").to_string())
        .collect();
    if types.len() != layers {
        return err(format!("{mt}: {} layer_types for {layers} layers", types.len()));
    }
    let lin = Linear {
        kind: LinearKind::GatedDeltaNet,
        k_heads: k.usize("linear_num_key_heads")?,
        v_heads: k.usize("linear_num_value_heads")?,
        k_dim: k.usize("linear_key_head_dim")?,
        v_dim: k.usize("linear_value_head_dim")?,
        conv: k.usize("linear_conv_kernel_dim")?,
        decay: Decay::Softplus,
        gate_act: lin_gate,
        norm_eps: eps,
        state_f32: true,
    };
    let mut dense_inter = None;
    let (mut moe_inter, mut shared_inter) = (0, 0);
    let moe = if k.raw("num_experts").is_some() {
        let ne = k.usize("num_experts")?;
        moe_inter = k.usize("moe_intermediate_size")?;
        shared_inter = k.opt_usize("shared_expert_intermediate_size")?.unwrap_or(0);
        let shared = shared_inter > 0;
        Some(Moe {
            router: Router {
                score: RouterScore::Softmax,
                logit_scale: 1.0,
                bias: false,
                groups: None,
                top_k: k.usize("num_experts_per_tok")?,
                renorm: k.bool("norm_topk_prob", true)?,
                scale: 1.0,
                hash: false,
            },
            experts: ne,
            expert: gated(act),
            shared: shared.then(|| gated(act)),
            shared_gate: shared,
            latent: None,
        })
    } else {
        dense_inter = Some(k.usize("intermediate_size")?);
        None
    };
    k.expect("mlp_only_layers", &Value::Array(Vec::new()))?;
    let mtp_layers = k.opt_usize("mtp_num_hidden_layers")?.unwrap_or(0);
    k.bool("mtp_use_dedicated_embeddings", false)?;
    let pn = Norm::rms_plus_one(eps);
    let (mut blocks, mut roles) = (Vec::new(), Vec::new());
    // (the shapes of the tensors, as the heads and widths make them: a mixture of experts' each expert's, the router's and the gated shared expert's too)
    let mut shapes = std::collections::BTreeMap::new();
    let (kh, vh, kd, vd) = (lin.k_heads, lin.v_heads, lin.k_dim, lin.v_dim);
    let qkv = 2 * kh * kd + vh * vd;
    for (l, ty) in types.iter().enumerate() {
        let p = names::HF.layer(prefix, l);
        let (mixer, r, sh) = match ty.as_str() {
            "linear_attention" => (
                Mixer::Linear(lin.clone()),
                names::roles(part::QWEN_LINEAR, &p, 0, all),
                vec![
                    ("qkv", vec![qkv, d]),
                    ("gate", vec![vh * vd, d]),
                    ("beta", vec![vh, d]),
                    ("decay", vec![vh, d]),
                    ("conv", vec![qkv, 1, lin.conv]),
                    ("a_log", vec![vh]),
                    ("dt_bias", vec![vh]),
                    ("norm", vec![vd]),
                    ("o", vec![d, vh * vd]),
                ],
            ),
            "full_attention" => {
                let a = Attn {
                    heads,
                    kv_heads: kv,
                    head_dim: hd,
                    v_head_dim: hd,
                    window: None,
                    sinks: false,
                    rope: Some(rp.clone()),
                    q_norm: Some(pn),
                    k_norm: Some(pn),
                    v_norm: None,
                    gate: gate.then_some(OutGate { act: Act::Sigmoid, from: GateFrom::InQuery, norm: None }),
                    softcap: None,
                    bias: false,
                    scale: (hd as f32).powf(-0.5),
                    k_eq_v: false,
                    indexer: None,
                    rounding: AttnRounding::Flash,
                    kv_from: None,
                };
                let sh = vec![
                    ("q", vec![heads * hd * if gate { 2 } else { 1 }, d]),
                    ("k", vec![kv * hd, d]),
                    ("v", vec![kv * hd, d]),
                    ("o", vec![d, heads * hd]),
                    ("q_norm", vec![hd]),
                    ("k_norm", vec![hd]),
                ];
                (Mixer::Attention(a), names::roles(part::ATTN, &p, 0, |r| !r.ends_with("_bias")), sh)
            }
            other => return err(format!("{mt}: layer type {other} at layer {l}")),
        };
        blocks.push(Block { norm_in: pn, body: Body::Mixer(mixer), norm_out: None, residual: Residual::Add });
        let f = match &moe {
            None => {
                blocks.push(Block { norm_in: pn, body: Body::Ffn(Ffn::Mlp(gated(act))), norm_out: None, residual: Residual::Add });
                names::roles(part::MLP, &p, 0, all)
            }
            Some(m) => {
                blocks.push(Block { norm_in: pn, body: Body::Ffn(Ffn::Moe(m.clone())), norm_out: None, residual: Residual::Add });
                names::roles(if stacked { part::QWEN_MOE_STACKED } else { part::QWEN_MOE }, &p, m.experts, |r| m.shared.is_some() || !r.starts_with("shared_"))
            }
        };
        shape_roles(&mut shapes, &r, &[sh, vec![("norm_in", vec![d])]].concat());
        match (dense_inter, &moe) {
            (Some(inter), _) => {
                shape_roles(&mut shapes, &f, &[("norm_in", vec![d]), ("gate", vec![inter, d]), ("up", vec![inter, d]), ("down", vec![d, inter])])
            }
            (None, Some(m)) => shape_roles(
                &mut shapes,
                &f,
                &[
                    ("norm_in", vec![d]),
                    ("router", vec![m.experts, d]),
                    // (a tensor an expert's matrix, or the stacked [experts, 2 inter, in] and [experts, out, inter])
                    ("experts_gate_up", if stacked { vec![m.experts, 2 * moe_inter, d] } else { vec![moe_inter, d] }),
                    ("experts_down", if stacked { vec![m.experts, d, moe_inter] } else { vec![d, moe_inter] }),
                    ("shared_gate", vec![shared_inter, d]),
                    ("shared_up", vec![shared_inter, d]),
                    ("shared_down", vec![d, shared_inter]),
                    ("shared_router", vec![1, d]),
                ],
            ),
            (None, None) => unreachable!("a dense width or experts"),
        }
        roles.push(r);
        roles.push(f);
    }
    let desc = ModelDesc {
        hidden: d,
        vocab,
        embed: Embed { scale: 1.0, norm: None, streams: 1, per_layer: None, table: TableAt::Gpu },
        blocks,
        final_norm: pn,
        head: Head { tied, softcap: None, scale: 1.0, collapse: Collapse::None, rounding: Rounding::F32 },
        mtp: Vec::new(),
        vision: None,
        rope: Some(rp),
    };
    let mut out = finish(&k, desc, &names::HF, prefix, roles, mtp_layers)?;
    for n in [Some(&out.embed), out.head.as_ref()].into_iter().flatten() {
        shapes.insert(n.clone(), vec![vocab, d]);
    }
    shapes.insert(out.final_norm.clone(), vec![d]);
    out.shapes = shapes;
    Ok(out)
}

/// Qwen3.8-Flash-Next (`qwen4_exp_text`, Condor's model): the Qwen3.5 hybrid's Gated DeltaNet and gated attention, the attention's Qwen Sparse
/// Attention indexer (`IndexForm::Mean`), the softmax-routed experts with the gated shared expert, every block inside gated residuals over
/// `hc_count` streams (`Residual::GatedStreams`), the final mixer (`Collapse::GatedMix`), and the hashed n-gram embedding (`Body::Ngram`)
/// before each layer `ple_layer_ids` names. A layer is two blocks, three with the n-gram embedding. The MTP layers are the draft head
/// (`MtpInput::Streams`: each an indexed attention and its experts in gated residuals, as the model's layers, its front's norms and products
/// and its own final mixer, `names::part::QX_MTP`). The vision tower is the multimodal config's (`with_tower`). The defaults of a key a config leaves
/// out are transformers' (`Qwen4ExpTextConfig`). Each tensor's shape is said.
fn qwen4x(tc: &Map<String, Value>, prefix: &str, mt: &str, tied_top: Option<bool>) -> Result<Described> {
    let k = Keys::new(tc, mt);
    let (d, layers, vocab, eps, tied) = common(&k, tied_top)?;
    let (heads, kv, hd) = (k.usize("num_attention_heads")?, k.usize("num_key_value_heads")?, k.usize("head_dim")?);
    let max_pos = k.opt_usize("max_position_embeddings")?;
    k.opt_f64("partial_rotary_factor")?;
    let rp = match k.raw("rope_parameters") {
        Some(Value::Object(o)) => rope(o, hd, max_pos, &format!("{mt} rope_parameters"))?,
        _ => legacy_rope(&k, hd, max_pos)?,
    };
    k.expect("attention_bias", &Value::Bool(false))?;
    let gate = k.bool("attn_output_gate", true)?;
    let act = act(k.str("hidden_act")?.unwrap_or("silu"))?;
    let lin_gate = match k.str("output_gate_type")? {
        Some(g) => self::act(g)?,
        None => act,
    };
    k.expect("mamba_ssm_dtype", &Value::from("float32"))?;
    let interval = k.opt_usize("full_attention_interval")?.unwrap_or(4).max(1);
    // (the release writes "full_attention" for the layers transformers calls "indexed_attention")
    let types: Vec<String> = match k.raw("layer_types").and_then(|v| v.as_array()) {
        Some(a) => a.iter().map(|x| x.as_str().unwrap_or("").to_string()).collect(),
        None => (0..layers).map(|l| if (l + 1) % interval == 0 { "indexed_attention" } else { "linear_attention" }.to_string()).collect(),
    };
    if types.len() != layers {
        return err(format!("{mt}: {} layer_types for {layers} layers", types.len()));
    }
    let lin = Linear {
        kind: LinearKind::GatedDeltaNet,
        k_heads: k.usize("linear_num_key_heads")?,
        v_heads: k.usize("linear_num_value_heads")?,
        k_dim: k.usize("linear_key_head_dim")?,
        v_dim: k.usize("linear_value_head_dim")?,
        conv: k.usize("linear_conv_kernel_dim")?,
        decay: Decay::Softplus,
        gate_act: lin_gate,
        norm_eps: eps,
        state_f32: true,
    };
    let pn = Norm::rms_plus_one(eps);
    // the indexer: all five keys or none
    const IX: [&str; 5] = ["indexer_n_heads", "indexer_kv_heads", "indexer_head_dim", "indexer_budget", "indexer_compress_ratio"];
    let ix: Vec<Option<usize>> = IX.iter().map(|x| k.opt_usize(x)).collect::<Result<_>>()?;
    let indexer = match ix[..] {
        [Some(ih), Some(ikv), Some(idim), Some(budget), Some(ratio)] => {
            if ikv != 1 {
                return err(format!("{mt}: indexer_kv_heads = {ikv} is not supported (only 1: one key head)"));
            }
            if ih == 0 || idim == 0 || ratio == 0 || budget == 0 || budget % ratio != 0 || rp.dim > idim {
                return err(format!("{mt}: an indexer of {ih} heads of {idim} keeping {budget} tokens in pools of {ratio}, under a rotary of {}", rp.dim));
            }
            Some(Indexer { heads: ih, head_dim: idim, top_k: budget, block: ratio, shared: false, form: IndexForm::Mean { norm: pn }, rope: 0 })
        }
        [None, None, None, None, None] => None,
        _ => return err(format!("{mt}: the indexer's keys come together ({}), and some are missing", IX.join(", "))),
    };
    let ne = k.opt_usize("num_experts")?.unwrap_or(512);
    let (inter, shared_inter) = (k.usize("moe_intermediate_size")?, k.usize("shared_expert_intermediate_size")?);
    let moe = Moe {
        router: Router {
            score: RouterScore::Softmax,
            logit_scale: 1.0,
            bias: false,
            groups: None,
            top_k: k.opt_usize("num_experts_per_tok")?.unwrap_or(10),
            renorm: k.bool("norm_topk_prob", true)?,
            scale: 1.0,
            hash: false,
        },
        experts: ne,
        expert: gated(act),
        shared: (shared_inter > 0).then(|| gated(act)),
        shared_gate: shared_inter > 0,
        latent: None,
    };
    let (hc, rank) = (k.opt_usize("hc_count")?.unwrap_or(4), k.opt_usize("hc_lowrank")?.unwrap_or(320));
    if hc < 2 || rank == 0 {
        return err(format!("{mt}: gated residuals over {hc} streams of a rank of {rank}"));
    }
    let gr = Residual::GatedStreams(GatedStreamsP { streams: hc, lowrank: rank });
    // the n-gram embedding: one-based layer ids, each a linear-attention layer
    let mut ple: Vec<usize> = match k.raw("ple_layer_ids") {
        Some(Value::Array(a)) => {
            a.iter().map(|x| x.as_u64().map(|v| v as usize).ok_or_else(|| Error(format!("{mt}: ple_layer_ids {x}")))).collect::<Result<_>>()?
        }
        Some(v) => return err(format!("{mt}: ple_layer_ids {v}")),
        None => Vec::new(),
    };
    ple.sort_unstable();
    ple.dedup();
    let (order, per) = (k.opt_usize("ngram_size")?.unwrap_or(3), k.opt_usize("heads_per_ngram")?.unwrap_or(8));
    let ple_dim = k.opt_usize("ple_embed_dim")?.unwrap_or(d);
    let base = k.opt_usize("ngram_vocab_size_base")?.unwrap_or(20_000_000) as u64;
    let pad = k.opt_usize("make_ngram_vocab_size_divisible_by")?.unwrap_or(128);
    let seed = k.opt_usize("seed")?.unwrap_or(1234) as u64;
    let parts = k.opt_usize("split_ngram_parts")?.unwrap_or(512);
    let taps = k.opt_usize("ple_conv_kernel_size")?.unwrap_or(4);
    let states = if ple.is_empty() { 1 } else { 3 };
    k.expect("number_of_conv_states", &Value::from(states))?;
    let eos = match k.raw("eos_token_id") {
        Some(Value::Array(a)) => a.first().and_then(Value::as_u64),
        Some(v) => v.as_u64(),
        None => None,
    };
    let tables = (order.max(1) - 1).saturating_mul(per);
    let ngram = |index: usize| -> Result<Ngram> {
        let eos = eos.ok_or_else(|| Error(format!("{mt}: an n-gram embedding and no eos_token_id")))? as u32;
        // (sizes past these are no model's, and would keep the builder finding primes or naming shards for long: 2e7 and 512 are transformers' defaults)
        let huge = base > 1 << 40 || tables.saturating_mul(index + 1) > 4096 || parts > 1 << 16 || order > 64;
        if order < 2 || per == 0 || ple_dim == 0 || ple_dim % tables != 0 || taps == 0 || parts == 0 || huge {
            return err(format!(
                "{mt}: an n-gram embedding of {ple_dim} over {per} tables for each order up to {order}, a convolution of {taps}, {parts} shards"
            ));
        }
        Ok(Ngram {
            order,
            heads: per,
            dim: ple_dim,
            base,
            vocab,
            seed,
            index,
            eos,
            streams: hc,
            conv: taps,
            dilation: order,
            norm: pn,
            form: NgramForm::Segmented,
            table: TableAt::Host,
        })
    };
    for &l in &ple {
        if l == 0 || l > layers || types[l - 1] != "linear_attention" {
            return err(format!("{mt}: ple_layer_ids {l} is not a linear_attention layer (one-based)"));
        }
    }
    // (the draft head's config: its layers, each an indexed attention with its experts on the model's last streams, at the model's rotary)
    let mtp = match k.raw("mtp") {
        Some(Value::Object(o)) => {
            let types = o.get("layer_types").and_then(Value::as_array).map(|a| a.iter().map(|x| x.as_str().unwrap_or("").to_string()).collect::<Vec<_>>());
            if let Some(t) = types.iter().flatten().find(|t| !["full_attention", "indexed_attention"].contains(&t.as_str())) {
                return err(format!("{mt}: an MTP layer of type {t}: the draft head's layers are indexed attentions"));
            }
            if o.get("mtp_use_hidden_state_from_layer").is_some_and(|v| !v.is_null()) {
                return err(format!("{mt}: an MTP head on the hidden state of a layer before the last ({})", o["mtp_use_hidden_state_from_layer"]));
            }
            if let Some(t) = o.get("rope_theta").and_then(Value::as_f64).filter(|&t| t != rp.theta as f64) {
                return err(format!("{mt}: an MTP head's rotary of theta {t}, where the model's is {}", rp.theta));
            }
            o.get("num_hidden_layers").and_then(Value::as_u64)
        }
        Some(v) => return err(format!("{mt}: mtp {v}")),
        None => None,
    };
    let mtp_layers = k.opt_usize("mtp_num_hidden_layers")?.or(mtp.map(|n| n as usize)).unwrap_or(0);
    k.bool("mtp_use_dedicated_embeddings", false)?;
    let (mut blocks, mut roles) = (Vec::new(), Vec::new());
    let mut shapes = std::collections::BTreeMap::new();
    let w = hc * d;
    let gr_shapes = vec![("norm_in", vec![w]), ("mix_down", vec![rank, w]), ("mix_up", vec![w, rank]), ("inject", vec![hc, w])];
    let (kh, vh, kd, vd) = (lin.k_heads, lin.v_heads, lin.k_dim, lin.v_dim);
    let qkv = 2 * kh * kd + vh * vd;
    // (an indexed attention block's mixer, roles and shapes under the path `p`: the model's attention layers' and the draft head's)
    let attention = |p: &str, at: &str| -> Result<(Mixer, Vec<(String, Source)>, Vec<(&'static str, Vec<usize>)>)> {
        let ix = indexer.clone().ok_or_else(|| Error(format!("{mt}: {at} is an indexed attention, and the config has no indexer_* keys")))?;
        let a = Attn {
            heads,
            kv_heads: kv,
            head_dim: hd,
            v_head_dim: hd,
            window: None,
            sinks: false,
            rope: Some(rp.clone()),
            q_norm: Some(pn),
            k_norm: Some(pn),
            v_norm: None,
            gate: gate.then_some(OutGate { act: Act::Sigmoid, from: GateFrom::InQuery, norm: None }),
            softcap: None,
            bias: false,
            scale: (hd as f32).powf(-0.5),
            k_eq_v: false,
            rounding: AttnRounding::Flash,
            kv_from: None,
            indexer: Some(ix.clone()),
        };
        let mut r = names::roles(part::ATTN, p, 0, |r| r != "norm_in" && !r.ends_with("_bias"));
        r.extend(names::roles(part::QX_INDEXER, p, 0, all));
        let sh = vec![
            ("q", vec![heads * hd * if gate { 2 } else { 1 }, d]),
            ("k", vec![kv * hd, d]),
            ("v", vec![kv * hd, d]),
            ("o", vec![d, heads * hd]),
            ("q_norm", vec![hd]),
            ("k_norm", vec![hd]),
            ("index_qk", vec![(ix.heads + 1) * ix.head_dim, d]),
            ("index_q_norm", vec![ix.head_dim]),
            ("index_k_norm", vec![ix.head_dim]),
        ];
        Ok((Mixer::Attention(a), r, sh))
    };
    // (the experts' block in its gated residual under the path `p`, its roles, their shapes said)
    let experts = |p: &str, shapes: &mut std::collections::BTreeMap<String, Vec<usize>>| -> (Block, Vec<(String, Source)>) {
        let mut f = names::roles(part::QX_MOE, p, 0, |r| moe.shared.is_some() || !r.starts_with("shared_"));
        f.extend(names::roles(part::QX_GR_MLP, p, 0, all));
        let fs = vec![
            ("router", vec![ne, d]),
            ("experts_gate_up", vec![ne, 2 * inter, d]),
            ("experts_down", vec![ne, d, inter]),
            ("shared_gate", vec![shared_inter, d]),
            ("shared_up", vec![shared_inter, d]),
            ("shared_down", vec![d, shared_inter]),
            ("shared_router", vec![1, d]),
        ];
        shape_roles(shapes, &f, &[fs, gr_shapes.clone()].concat());
        (Block { norm_in: pn, body: Body::Ffn(Ffn::Moe(moe.clone())), norm_out: None, residual: gr.clone() }, f)
    };
    for (l, ty) in types.iter().enumerate() {
        let p = names::QWEN4X.layer(prefix, l);
        if let Some(index) = ple.iter().position(|&x| x == l + 1) {
            let n = ngram(index)?;
            let rows = super::reference::ngram::rows(&n, pad);
            if rows % parts != 0 {
                return err(format!("{mt}: an n-gram table of {rows} rows in {parts} shards"));
            }
            let mut r = names::roles(part::QX_NGRAM, &p, 0, all);
            let sh = vec![
                ("norm_in", vec![w]),
                ("key", vec![w, ple_dim]),
                ("value", vec![d, ple_dim]),
                ("norm_key", vec![w]),
                ("norm_conv", vec![w]),
                ("conv", vec![w, 1, taps]),
                ("ngram_mult", vec![order]),
                ("ngram_sizes", vec![tables]),
                ("ngram_offsets", vec![tables]),
            ];
            shape_roles(&mut shapes, &r, &sh);
            let table = names::shards(part::QX_NGRAM_SHARD, &p, parts);
            if let Source::Rows(v) = &table {
                shapes.extend(v.iter().map(|s| (s.clone(), vec![rows / parts, ple_dim / tables])));
            }
            r.push(("ngram".into(), table));
            blocks.push(Block { norm_in: pn, body: Body::Ngram(n), norm_out: None, residual: Residual::Add });
            roles.push(r);
        }
        let (mixer, mut r, sh) = match ty.as_str() {
            "linear_attention" => (
                Mixer::Linear(lin.clone()),
                names::roles(part::QWEN_LINEAR, &p, 0, |r| r != "norm_in"),
                vec![
                    ("qkv", vec![qkv, d]),
                    ("gate", vec![vh * vd, d]),
                    ("beta", vec![vh, d]),
                    ("decay", vec![vh, d]),
                    ("conv", vec![qkv, 1, lin.conv]),
                    ("a_log", vec![vh]),
                    ("dt_bias", vec![vh]),
                    ("norm", vec![vd]),
                    ("o", vec![d, vh * vd]),
                ],
            ),
            "full_attention" | "indexed_attention" => {
                let (a, r, sh) = attention(&p, &format!("layer {l}"))?;
                (a, r, sh)
            }
            other => return err(format!("{mt}: layer type {other} at layer {l}")),
        };
        r.extend(names::roles(part::QX_GR_ATTN, &p, 0, all));
        shape_roles(&mut shapes, &r, &[sh, gr_shapes.clone()].concat());
        blocks.push(Block { norm_in: pn, body: Body::Mixer(mixer), norm_out: None, residual: gr.clone() });
        roles.push(r);
        let (b, f) = experts(&p, &mut shapes);
        blocks.push(b);
        roles.push(f);
    }
    // the draft head (`MtpInput::Streams`): each MTP layer an indexed attention and its experts in gated residuals, as the model's layers; its own front
    // (the two norms and the two products) and its own final mixer
    let (mut head_blocks, mut head_roles) = (Vec::new(), Vec::new());
    for j in 0..mtp_layers {
        let p = format!("mtp.layers.{j}");
        let (mixer, mut r, sh) = attention(&p, &format!("MTP layer {j}"))?;
        r.extend(names::roles(part::QX_GR_ATTN, &p, 0, all));
        shape_roles(&mut shapes, &r, &[sh, gr_shapes.clone()].concat());
        head_blocks.push(Block { norm_in: pn, body: Body::Mixer(mixer), norm_out: None, residual: gr.clone() });
        head_roles.push(r);
        let (b, f) = experts(&p, &mut shapes);
        head_blocks.push(b);
        head_roles.push(f);
    }
    let head_own = if mtp_layers > 0 { names::roles(part::QX_MTP, "mtp", 0, all) } else { Vec::new() };
    shape_roles(
        &mut shapes,
        &head_own,
        &[
            ("norm_embed", vec![d]),
            ("norm_hidden", vec![w]),
            ("fc_embedding", vec![d, d]),
            ("fc_hidden", vec![d, d]),
            ("final_norm", vec![w]),
            ("out_mix_down", vec![rank, w]),
            ("out_mix_up", vec![w, rank]),
        ],
    );
    let head =
        (mtp_layers > 0).then_some(MtpHead { blocks: head_blocks, norm_embed: pn, norm_hidden: pn, final_norm: pn, dspark: None, input: MtpInput::Streams });
    let desc = ModelDesc {
        hidden: d,
        vocab,
        embed: Embed { scale: 1.0, norm: None, streams: hc, per_layer: None, table: if tied { TableAt::Gpu } else { TableAt::Host } },
        blocks,
        final_norm: pn,
        head: Head { tied, softcap: None, scale: 1.0, collapse: Collapse::GatedMix, rounding: Rounding::F32 },
        mtp: head.into_iter().collect(),
        vision: None,
        rope: Some(rp),
    };
    let mut out = finish(&k, desc, &names::QWEN4X, prefix, roles, mtp_layers)?;
    for n in [Some(&out.embed), out.head.as_ref()].into_iter().flatten() {
        shapes.insert(n.clone(), vec![vocab, d]);
    }
    shapes.insert(out.final_norm.clone(), vec![w]);
    shape_roles(&mut shapes, &out.extra, &[("out_mix_down", vec![rank, w]), ("out_mix_up", vec![w, rank])]);
    out.shapes = shapes;
    // (the draft head named: its tensors are read, no longer a part left)
    if mtp_layers > 0 {
        out.mtp = vec![(head_roles, head_own)];
        out.unread.retain(|p| p != "mtp.");
    }
    Ok(out)
}

/// The Qwen-VL vision tower of a multimodal config (`kinds::VisionDesc`, `vision_config`) added to its text model's description `d`: the kind's
/// parameters, the image token (the top level's `image_token_id`), its roles under `model.visual` (`names::part::VISION`, `VISION_BLOCK`) and each
/// tensor's shape; its tensors then read, no longer a part left. `d` as it is where the config has no tower, or is the language model alone
/// (`language_model_only`). The tower's tokens take M-RoPE positions in the text: the text's rotary must have interleaved sections. A tower with
/// deepstack layers (its features added into the text's layers too) is refused, and so is a key the builder does not know.
fn with_tower(mut d: Described, c: &ModelConfig) -> Result<Described> {
    let Some(v) = c.top.get("vision_config").filter(|v| !v.is_null()) else { return Ok(d) };
    if c.top.get("language_model_only").and_then(Value::as_bool) == Some(true) {
        return Ok(d);
    }
    let Value::Object(v) = v else { return err(format!("vision_config {v}: not an object")) };
    let k = Keys::new(v, "vision_config");
    let (blocks, hidden, heads, inter) = (k.usize("depth")?, k.usize("hidden_size")?, k.usize("num_heads")?, k.usize("intermediate_size")?);
    let act = act(k.str("hidden_act")?.unwrap_or("gelu_pytorch_tanh"))?;
    let (channels, patch) = (k.opt_usize("in_channels")?.unwrap_or(3), k.usize("patch_size")?);
    let (frames, merge) = (k.opt_usize("temporal_patch_size")?.unwrap_or(2), k.opt_usize("spatial_merge_size")?.unwrap_or(2));
    let (out, positions) = (k.usize("out_hidden_size")?, k.usize("num_position_embeddings")?);
    match k.raw("deepstack_visual_indexes") {
        None => {}
        Some(Value::Array(a)) if a.is_empty() => {}
        Some(x) => return err(format!("vision_config: deepstack_visual_indexes {x}: a tower whose features go into the text's layers too is not supported")),
    }
    // (transformers' newer configs say the tower's rotary: the axial one, its theta; the release's says none, the default's 10,000)
    let theta = match k.raw("rope_parameters") {
        None => 10_000.0,
        Some(Value::Object(o)) => {
            let r = Keys::new(o, "vision_config rope_parameters");
            if r.str("rope_type")?.unwrap_or("axial") != "axial" {
                return err(format!("vision_config: rope_parameters {}: only the axial rotary is supported", Value::Object(o.clone())));
            }
            let t = r.opt_f64("rope_theta")?.unwrap_or(10_000.0);
            r.finish()?;
            t
        }
        Some(x) => return err(format!("vision_config: rope_parameters {x}")),
    };
    k.finish()?;
    let table = (positions as f64).sqrt().round() as usize;
    if [blocks, hidden, heads, inter, channels, patch, frames, merge, out].contains(&0)
        || hidden % heads != 0
        || (hidden / heads) % 4 != 0
        || table * table != positions
    {
        return err(format!(
            "vision_config: a tower of {blocks} blocks of {heads} heads over {hidden}, {inter} neurons, {positions} positions (a square table), patches of {channels} x {frames} x {patch} x {patch}, merged {merge} x {merge}: not a tower of this kind"
        ));
    }
    if out != d.desc.hidden {
        return err(format!("vision_config: out_hidden_size {out}, where the text model's rows are {}", d.desc.hidden));
    }
    let image_token = match c.top.get("image_token_id").and_then(Value::as_u64) {
        Some(t) if (t as usize) < d.desc.vocab => t as u32,
        t => return err(format!("image_token_id {t:?}: the image token must be one of the vocabulary's {}", d.desc.vocab)),
    };
    // (the image's tokens take M-RoPE positions: the text's rotary has sections, interleaved)
    let rope = d.desc.blocks.iter().find_map(|b| if let Body::Mixer(Mixer::Attention(a)) = &b.body { a.rope.clone() } else { None }).or(d.desc.rope.clone());
    let interleaved = ["rope_parameters", "rope_scaling"].iter().find_map(|r| c.text.get(*r)?.get("mrope_interleaved")?.as_bool()) == Some(true);
    if !rope.as_ref().is_some_and(|r| r.mrope.is_some()) || !interleaved {
        return err(format!("vision_config: a tower whose tokens' positions the text's rotary ({rope:?}) does not turn as interleaved M-RoPE sections"));
    }
    let at = names::part::VISION_AT;
    let own = names::roles(part::VISION, at, 0, all);
    let m = hidden * merge * merge;
    let mut shapes = std::mem::take(&mut d.shapes);
    shape_roles(
        &mut shapes,
        &own,
        &[
            ("patch", vec![hidden, channels, frames, patch, patch]),
            ("patch_bias", vec![hidden]),
            ("pos", vec![positions, hidden]),
            ("merge_norm", vec![hidden]),
            ("merge_norm_bias", vec![hidden]),
            ("merge_up", vec![m, m]),
            ("merge_up_bias", vec![m]),
            ("merge_down", vec![out, m]),
            ("merge_down_bias", vec![out]),
        ],
    );
    let roles: Vec<Vec<(String, Source)>> = (0..blocks)
        .map(|l| {
            let r = names::roles(part::VISION_BLOCK, &format!("{at}.blocks.{l}"), 0, all);
            shape_roles(
                &mut shapes,
                &r,
                &[
                    ("norm1", vec![hidden]),
                    ("norm1_bias", vec![hidden]),
                    ("qkv", vec![3 * hidden, hidden]),
                    ("qkv_bias", vec![3 * hidden]),
                    ("o", vec![hidden, hidden]),
                    ("o_bias", vec![hidden]),
                    ("norm2", vec![hidden]),
                    ("norm2_bias", vec![hidden]),
                    ("up", vec![inter, hidden]),
                    ("up_bias", vec![inter]),
                    ("down", vec![hidden, inter]),
                    ("down_bias", vec![hidden]),
                ],
            );
            r
        })
        .collect();
    d.shapes = shapes;
    d.desc.vision = Some(VisionDesc { blocks, hidden, heads, inter, act, channels, patch, frames, merge, out, table, theta, eps: 1e-6, image_token });
    d.vision = Some(VisionRoles { own, blocks: roles });
    d.unread.retain(|p| p != &format!("{at}."));
    Ok(d)
}

fn gemma4(tc: &Map<String, Value>, prefix: &str, mt: &str, tied_top: Option<bool>) -> Result<Described> {
    let k = Keys::new(tc, mt);
    // (transformers' Gemma 4 ties the head to the embedding unless the config says otherwise, where Llama and Qwen default to a head of their own)
    let (d, layers, vocab, eps, tied) = common(&k, tied_top.or(Some(true)))?;
    let (heads, kv, hd) = (k.usize("num_attention_heads")?, k.usize("num_key_value_heads")?, k.usize("head_dim")?);
    // the global layers' head dim and KV heads: the release's keys, or (transformers 5.18 writes these instead) `per_layer_config`
    let per = match k.raw("per_layer_config") {
        Some(Value::Object(o)) => Some(o),
        Some(v) => return err(format!("{mt}: per_layer_config {v}")),
        None => None,
    };
    let per_first = |key: &str| per.and_then(|o| o.values().find_map(|v| v.get(key).and_then(|x| x.as_u64()))).map(|x| x as usize);
    let ghd = k.opt_usize("global_head_dim")?.or(per_first("head_dim")).unwrap_or(hd);
    let k_eq_v = k.bool("attention_k_eq_v", false)?;
    let gkv = if k_eq_v { k.opt_usize("num_global_key_value_heads")?.or(per_first("num_key_value_heads")).unwrap_or(kv) } else { kv };
    k.opt_usize("num_global_key_value_heads")?;
    let window = k.usize("sliding_window")?;
    let max_pos = k.opt_usize("max_position_embeddings")?;
    k.expect("attention_bias", &Value::Bool(false))?;
    k.expect("enable_moe_block", &Value::Bool(false))?;
    // per-layer inputs (E2B, E4B): `ple` values a layer for each token, from a second table of `vocab_size_per_layer_input` rows
    let ple = k.opt_usize("hidden_size_per_layer_input")?.unwrap_or(0);
    let ple_vocab = k.opt_usize("vocab_size_per_layer_input")?;
    // KV sharing (E2B, E4B): the last `shared` layers' queries attend to the keys and values of the last earlier layer of their type; with the
    // double-wide MLP those layers' intermediate width is twice the config's
    let shared = k.opt_usize("num_kv_shared_layers")?.unwrap_or(0);
    let wide = k.bool("use_double_wide_mlp", false)?;
    for x in ["num_experts", "top_k_experts", "moe_intermediate_size"] {
        k.expect(x, &Value::Null)?;
    }
    if k.str("use_bidirectional_attention")? == Some("all") {
        return err(format!("{mt}: bidirectional attention for text is not supported"));
    }
    let inter = k.usize("intermediate_size")?;
    let act = act(k.str("hidden_activation")?.or(k.str("hidden_act")?).unwrap_or("gelu_pytorch_tanh"))?;
    let softcap = k.opt_f64("final_logit_softcapping")?.map(|v| v as f32);
    let rps = match k.raw("rope_parameters") {
        Some(Value::Object(o)) => o,
        _ => return err(format!("{mt}: no rope_parameters per layer type")),
    };
    let rope_of = |ty: &str, head: usize| -> Result<Rope> {
        match rps.get(ty) {
            Some(Value::Object(o)) => rope(o, head, max_pos, &format!("{mt} rope_parameters.{ty}")),
            _ => err(format!("{mt}: no rope_parameters for {ty}")),
        }
    };
    let types: Vec<String> = k
        .raw("layer_types")
        .and_then(|v| v.as_array())
        .ok_or_else(|| Error(format!("{mt}: no layer_types")))?
        .iter()
        .map(|x| x.as_str().unwrap_or("").to_string())
        .collect();
    if types.len() != layers {
        return err(format!("{mt}: {} layer_types for {layers} layers", types.len()));
    }
    let (sl, gl) = (rope_of("sliding_attention", hd)?, rope_of("full_attention", ghd)?);
    // transformers writes the global layers' head dim and KV heads again per layer; they must agree
    if let Some(per) = per {
        for (l, o) in per {
            let i: usize = l.parse().map_err(|_| Error(format!("{mt}: per_layer_config key {l}")))?;
            let (h, n) = (o.get("head_dim").and_then(|v| v.as_u64()), o.get("num_key_value_heads").and_then(|v| v.as_u64()));
            if types.get(i).map(String::as_str) != Some("full_attention")
                || h.is_some_and(|h| h as usize != ghd)
                || n.is_some_and(|n| n as usize != gkv)
                || o.as_object().is_none_or(|m| m.len() > 2)
            {
                return err(format!("{mt}: per_layer_config for layer {l} ({o:?}) is not the global layers' shape"));
            }
        }
    }
    if shared >= layers {
        return err(format!("{mt}: {shared} of {layers} layers share keys and values, where an earlier layer of each type must make them"));
    }
    let first_shared = layers - shared;
    // (each shared layer's store: the last layer before the shared ones of its type; its attention block is twice its index)
    let store = |l: usize| -> Result<Option<usize>> {
        if l < first_shared {
            return Ok(None);
        }
        match types[..first_shared].iter().rposition(|t| *t == types[l]) {
            Some(j) => Ok(Some(2 * j)),
            None => err(format!("{mt}: layer {l} ({}) shares keys and values, and no earlier layer of its type makes them", types[l])),
        }
    };
    let n = Norm::rms(eps);
    let (mut blocks, mut roles, mut shapes) = (Vec::new(), Vec::new(), std::collections::BTreeMap::new());
    let mut unread = Vec::new();
    for (l, ty) in types.iter().enumerate() {
        let sliding = match ty.as_str() {
            "sliding_attention" => true,
            "full_attention" => false,
            o => return err(format!("{mt}: layer type {o} at layer {l}")),
        };
        let kv_from = store(l)?;
        let inter = if wide && kv_from.is_some() { 2 * inter } else { inter };
        let (h, kvh, kev) = if sliding { (hd, kv, false) } else { (ghd, gkv, k_eq_v) };
        let a = Attn {
            heads,
            kv_heads: kvh,
            head_dim: h,
            v_head_dim: h,
            window: sliding.then_some(window),
            sinks: false,
            rope: Some(if sliding { sl.clone() } else { gl.clone() }),
            q_norm: Some(n),
            k_norm: Some(n),
            v_norm: Some(Norm::rms_bare(eps)),
            gate: None,
            softcap: None,
            bias: false,
            scale: 1.0,
            k_eq_v: kev,
            indexer: None,
            rounding: AttnRounding::Eager,
            kv_from,
        };
        let p = names::GEMMA.layer(prefix, l);
        let own_kv = kv_from.is_none();
        let r = [
            names::roles(part::ATTN, &p, 0, |r| !r.ends_with("_bias") && !(kev && r == "v") && (own_kv || !matches!(r, "k" | "v" | "k_norm"))),
            names::roles(part::GEMMA_ATTN_OUT, &p, 0, all),
        ]
        .concat();
        // (a shared layer's checkpoint may still hold its key and value products and k norm: not read)
        if !own_kv {
            unread.extend(["k_proj.", "v_proj.", "k_norm."].map(|t| format!("{p}.self_attn.{t}")));
        }
        // (with per-layer inputs the layer scalar follows their branch, the MLP's residual a plain add)
        let f = names::roles(part::GEMMA_MLP, &p, 0, |r| ple == 0 || r != "res_scale");
        // (the shapes: each matrix [rows, columns] as the layer's heads and widths make it, each norm and scalar its width)
        shape_roles(
            &mut shapes,
            &r,
            &[
                ("norm_in", vec![d]),
                ("q", vec![heads * h, d]),
                ("k", vec![kvh * h, d]),
                ("v", vec![kvh * h, d]),
                ("o", vec![d, heads * h]),
                ("q_norm", vec![h]),
                ("k_norm", vec![h]),
                ("norm_out", vec![d]),
            ],
        );
        shape_roles(
            &mut shapes,
            &f,
            &[("norm_in", vec![d]), ("gate", vec![inter, d]), ("up", vec![inter, d]), ("down", vec![d, inter]), ("norm_out", vec![d]), ("res_scale", vec![1])],
        );
        blocks.push(Block { norm_in: n, body: Body::Mixer(Mixer::Attention(a)), norm_out: Some(n), residual: Residual::Add });
        roles.push(r);
        let mlp_res = if ple == 0 { Residual::AddThenScale } else { Residual::Add };
        blocks.push(Block { norm_in: n, body: Body::Ffn(Ffn::Mlp(gated(act))), norm_out: Some(n), residual: mlp_res });
        roles.push(f);
        if ple > 0 {
            let g = names::roles(part::GEMMA_PER_LAYER, &p, 0, all);
            shape_roles(&mut shapes, &g, &[("gate", vec![ple, d]), ("proj", vec![d, ple]), ("norm_out", vec![d]), ("res_scale", vec![1])]);
            let norm_in = Norm { kind: NormKind::Identity, weight: NormWeight::None, eps, rounding: Rounding::F32 };
            blocks.push(Block { norm_in, body: Body::PerLayer(PerLayer { layer: l, act }), norm_out: Some(n), residual: Residual::AddThenScale });
            roles.push(g);
        }
    }
    let per_layer = (ple > 0).then(|| PerLayerIn { dim: ple, layers, vocab: ple_vocab.unwrap_or(vocab), norm: n });
    let desc = ModelDesc {
        hidden: d,
        vocab,
        embed: Embed { scale: (d as f32).sqrt(), norm: None, streams: 1, per_layer, table: TableAt::Gpu },
        blocks,
        final_norm: n,
        // (the release's head is bf16: transformers soft-caps `lm_head`'s bf16 logits by three bf16 operations)
        head: Head { tied, softcap, scale: 1.0, collapse: Collapse::None, rounding: Rounding::Twice },
        mtp: Vec::new(),
        vision: None,
        rope: None,
    };
    let mut out = finish(&k, desc, &names::GEMMA, prefix, roles, 0)?;
    shapes.insert(out.embed.clone(), vec![vocab, d]);
    shapes.insert(out.final_norm.clone(), vec![d]);
    if let Some(pl) = per_layer {
        let x = names::roles(part::GEMMA_PER_LAYER_IN, prefix.trim_end_matches('.'), 0, all);
        shape_roles(&mut shapes, &x, &[("ple_embed", vec![pl.vocab, layers * ple]), ("ple_proj", vec![layers * ple, d]), ("ple_norm", vec![ple])]);
        out.extra.extend(x);
    }
    out.shapes = shapes;
    out.unread.extend(unread);
    Ok(out)
}

/// FP8 block quantization as released (`quantization_config`): which matrices hold FP8 and their block.
struct Fp8 {
    block: usize,
    skip: Vec<String>,
}

impl Fp8 {
    fn parse(q: Option<&Value>, what: &str) -> Result<Option<Fp8>> {
        let Some(q) = q.and_then(|v| v.as_object()) else { return Ok(None) };
        let method = q.get("quant_method").and_then(|v| v.as_str()).unwrap_or("fp8");
        let fmt = q.get("fmt").and_then(|v| v.as_str()).unwrap_or("e4m3");
        if method != "fp8" || fmt != "e4m3" {
            return err(format!("{what}: quantization {method} {fmt} is not supported"));
        }
        let block = match q.get("weight_block_size") {
            Some(Value::Array(a)) if a.len() == 2 && a[0] == a[1] => a[0].as_u64().unwrap_or(128) as usize,
            Some(v) => return err(format!("{what}: weight_block_size {v}")),
            // (no block size: one scale a tensor or a channel, not the block scales the loaders read; refused, as the GLM loader refuses it)
            None => return err(format!("{what}: FP8 with no weight_block_size (scales a tensor or a channel) is not supported: block scales are")),
        };
        let skip = q
            .get("modules_to_not_convert")
            .and_then(|v| v.as_array())
            .map(|a| a.iter().filter_map(|x| x.as_str().map(String::from)).collect())
            .unwrap_or_default();
        Ok(Some(Fp8 { block, skip }))
    }

    /// Whether the matrix `w` (`x.weight`, the language model's prefix dropped as the config lists modules) is FP8.
    fn holds(&self, w: &str) -> bool {
        let m = w.strip_suffix(".weight").unwrap_or(w).replace("language_model.", "");
        !self.skip.iter().any(|s| m == *s || m.ends_with(&format!(".{s}")) || m.starts_with(&format!("{s}.")))
    }
}

/// The layer kinds of a GLM config, from transformers' keys (`layer_types`) or the release's (`linear_attn_config`).
fn glm_types(k: &Keys, layers: usize, mt: &str) -> Result<Vec<bool>> {
    if let Some(Value::Array(a)) = k.raw("layer_types") {
        let t: Vec<bool> = a
            .iter()
            .map(|x| match x.as_str() {
                Some("linear_attention") => Ok(true),
                Some("indexed_attention" | "deepseek_sparse_attention" | "full_attention") => Ok(false),
                o => err(format!("{mt}: layer type {o:?}")),
            })
            .collect::<Result<_>>()?;
        if t.len() < layers {
            return err(format!("{mt}: {} layer_types for {layers} layers", t.len()));
        }
        return Ok(t[..layers].to_vec());
    }
    Ok(vec![false; layers])
}

/// Each layer's indexer, `full` (its own) or `shared` (the picks of the layer before), as transformers' GlmMoeDsaConfig reads them: `indexer_types` where the
/// config lists them; else the schedule they are made from, `index_topk_pattern` (a string of F and S, or a list), else `index_topk_freq` and
/// `index_skip_topk_offset` (layer i's own where max(i - offset + 1, 0) is a multiple of freq; 1 and 2 where absent). A config with neither: every one full.
fn indexer_types(k: &Keys, layers: usize, mt: &str) -> Result<Vec<String>> {
    let pattern = k.raw("index_topk_pattern").filter(|v| !v.is_null());
    let (freq, offset) = (k.opt_usize("index_topk_freq")?, k.opt_usize("index_skip_topk_offset")?);
    let kind = |x: &Value| match x.as_str() {
        Some(t @ ("full" | "shared")) => Ok(t.to_string()),
        Some("F") => Ok("full".to_string()),
        Some("S") => Ok("shared".to_string()),
        o => err(format!("{mt}: an indexer of type {o:?}")),
    };
    let types: Vec<String> = match (k.raw("indexer_types").filter(|v| !v.is_null()), pattern) {
        (Some(Value::Array(a)), _) => a.iter().map(kind).collect::<Result<_>>()?,
        (Some(v), _) => return err(format!("{mt}: indexer_types {v}")),
        (None, Some(Value::String(p))) => p.chars().map(|c| kind(&Value::from(c.to_string()))).collect::<Result<_>>()?,
        (None, Some(Value::Array(a))) => a.iter().map(kind).collect::<Result<_>>()?,
        (None, Some(v)) => return err(format!("{mt}: index_topk_pattern {v}")),
        (None, None) if freq.is_none() && offset.is_none() => return Ok(Vec::new()),
        (None, None) => {
            let (f, o) = (freq.unwrap_or(1).max(1), offset.unwrap_or(2));
            (0..layers).map(|i| if (i + 1).saturating_sub(o) % f == 0 { "full" } else { "shared" }.to_string()).collect()
        }
    };
    if types.first().is_some_and(|t| t == "shared") {
        return err(format!("{mt}: the first layer's indexer is shared, with no layer before it to take picks from"));
    }
    Ok(types)
}

fn glm(tc: &Map<String, Value>, prefix: &str, mt: &str, tied_top: Option<bool>, quant: Option<&Value>) -> Result<Described> {
    let k = Keys::new(tc, mt);
    let flash = mt != "glm_moe_dsa";
    let (d, layers, vocab, eps, tied) = common(&k, tied_top)?;
    let max_pos = k.opt_usize("max_position_embeddings")?;
    k.expect("attention_bias", &Value::Bool(false))?;
    k.expect("mlp_bias", &Value::Bool(false))?;
    k.raw("quantization_config");
    let fp8 = Fp8::parse(quant, mt)?;
    let act = act(k.str("hidden_act")?.unwrap_or("silu"))?;
    if let Some(sf) = k.str("scoring_func")? {
        if sf != "sigmoid" {
            return err(format!("{mt}: scoring_func {sf}"));
        }
    }
    if let Some(tm) = k.str("topk_method")? {
        if tm != "noaux_tc" {
            return err(format!("{mt}: topk_method {tm}"));
        }
    }
    k.expect("moe_router_dtype", &Value::from("float32"))?;
    k.expect("moe_layer_freq", &Value::from(1))?;
    k.opt_usize("ep_size")?;
    let types = glm_types(&k, layers, mt)?;
    // the release's linear-attention block (else transformers' keys)
    let lac = match k.raw("linear_attn_config") {
        Some(Value::Object(o)) => Some(o),
        _ => None,
    };
    let lin_num = |a: &str, b: &str| -> Result<Option<usize>> {
        match lac.and_then(|o| o.get(a)).and_then(|v| v.as_u64()) {
            Some(v) => Ok(Some(v as usize)),
            None => k.opt_usize(b),
        }
    };
    let types = if lac.is_some() && k.m.get("layer_types").is_none() {
        let kda: BTreeSet<u64> =
            lac.and_then(|o| o.get("kda_layers")).and_then(|v| v.as_array()).map(|a| a.iter().filter_map(|x| x.as_u64()).collect()).unwrap_or_default();
        (0..layers).map(|l| kda.contains(&(l as u64))).collect()
    } else {
        types
    };
    let bound = match lac.and_then(|o| o.get("gate_lower_bound")) {
        Some(v) => v.as_f64(),
        None => k.opt_f64("linear_lower_bound")?,
    };
    let lin = Linear {
        kind: LinearKind::KimiDelta,
        k_heads: lin_num("num_heads", "linear_num_heads")?.unwrap_or(0),
        v_heads: lin_num("num_heads", "linear_num_heads")?.unwrap_or(0),
        k_dim: lin_num("head_dim", "linear_head_dim")?.unwrap_or(0),
        v_dim: lin_num("head_dim", "linear_head_dim")?.unwrap_or(0),
        conv: lin_num("short_conv_kernel_size", "linear_conv_kernel_dim")?.unwrap_or(4),
        decay: match bound {
            Some(b) => Decay::Bounded(b as f32),
            None => Decay::Softplus,
        },
        gate_act: Act::Sigmoid,
        norm_eps: eps,
        state_f32: true,
    };
    if types.iter().any(|&t| t) && lin.k_heads == 0 {
        return err(format!("{mt}: linear-attention layers without their heads"));
    }
    // the latent attention
    let (heads, nope, rd, vd, kvl) = (
        k.usize("num_attention_heads")?,
        k.usize("qk_nope_head_dim")?,
        k.opt_usize("qk_rope_head_dim")?.unwrap_or(0),
        k.usize("v_head_dim")?,
        k.usize("kv_lora_rank")?,
    );
    k.opt_usize("num_key_value_heads")?;
    k.opt_usize("qk_head_dim")?;
    k.opt_usize("head_dim")?;
    let q_lora = k.opt_usize("q_lora_rank")?;
    let nope_only = k.bool("mla_use_nope", rd == 0)?;
    let rope = if rd > 0 && !nope_only {
        let mut r = match k.raw("rope_parameters") {
            Some(Value::Object(o)) => rope(o, rd, max_pos, &format!("{mt} rope_parameters"))?,
            _ => legacy_rope(&k, rd, max_pos)?,
        };
        r.style = RopeStyle::Interleaved;
        Some(r)
    } else {
        k.raw("rope_parameters");
        k.raw("rope_theta");
        None
    };
    let mut scale = ((nope + rd) as f32).powf(-0.5);
    if let Some(Rope { scaling: RopeScaling::Yarn { factor, mscale_all_dim: Some(m), .. }, .. }) = &rope {
        let ms = if *factor <= 1.0 { 1.0 } else { 0.1 * m * factor.ln() + 1.0 };
        scale *= (ms * ms) as f32;
    }
    let mla_eps = if flash { eps } else { 1e-6 };
    let (ih, idim, itop) = (k.opt_usize("index_n_heads")?, k.opt_usize("index_head_dim")?, k.opt_usize("index_topk")?);
    let pool = k.opt_usize("index_kpool")?.unwrap_or(1);
    // (the release's flags that its model's code has no other way of: transformers' GlmMoeDsaIndexer and attention turn interleaved pairs whatever they say)
    for x in ["index_kpool_always_select_tail", "index_kpool_compress", "index_share_for_mtp_iteration", "indexer_rope_interleave", "rope_interleave"] {
        k.bool(x, true)?;
    }
    let itypes = indexer_types(&k, layers, mt)?;
    // the feed-forward
    let first_dense = k.opt_usize("first_k_dense_replace")?;
    let mlp_types: Option<Vec<bool>> = k.raw("mlp_layer_types").and_then(|v| v.as_array()).map(|a| a.iter().map(|x| x.as_str() == Some("sparse")).collect());
    let dense_inter = k.usize("intermediate_size")?;
    let moe_inter = k.usize("moe_intermediate_size")?;
    let clamp = if flash { k.opt_f64("swiglu_limit")?.map(|v| v as f32) } else { None };
    let mlp = Mlp { act, gated: true, bias: false, clamp };
    let ne = k.usize("n_routed_experts")?;
    let ng = k.opt_usize("n_group")?.unwrap_or(1);
    let tg = k.opt_usize("topk_group")?.unwrap_or(1);
    let moe = Moe {
        router: Router {
            score: RouterScore::Sigmoid,
            logit_scale: 1.0,
            bias: true,
            groups: (ng > 1).then_some((ng, tg)),
            top_k: k.usize("num_experts_per_tok")?,
            renorm: k.bool("norm_topk_prob", true)?,
            scale: k.opt_f64("routed_scaling_factor")?.unwrap_or(1.0) as f32,
            hash: false,
        },
        experts: ne,
        expert: mlp,
        shared: (k.opt_usize("n_shared_experts")?.unwrap_or(0) > 0).then_some(mlp),
        shared_gate: false,
        latent: None,
    };
    let shared_inter = moe_inter * k.opt_usize("n_shared_experts")?.unwrap_or(0);
    // the residual streams (GLM-5.3-Flash)
    let streams = if flash && k.bool("mhc", true)? {
        Some(StreamsP {
            lagged: false,
            streams: k.opt_usize("hc_mult")?.unwrap_or(4),
            sinkhorn_iters: k.opt_usize("hc_sinkhorn_iters")?.unwrap_or(20),
            eps: k.opt_f64("hc_eps")?.unwrap_or(1e-6) as f32,
            norm_eps: eps,
            rounding: Rounding::Twice,
        })
    } else {
        None
    };
    let mtp_layers = k.opt_usize("num_nextn_predict_layers")?.unwrap_or(0);
    // (the RMS norms round as the release's do, Glm5NextTextRMSNorm and GlmMoeDsaRMSNorm in bf16: the normed value to bf16, times the bf16 weight, rounded:
    // the blocks', the final one, the latent attention's q and key-value latents')
    let n = Norm { rounding: Rounding::Twice, ..Norm::rms(eps) };
    // (the matrices a release holds in FP8: those of these roles that its quantization_config does not leave out; a layer's experts as its first expert's)
    let held = |w: &str| fp8.as_ref().is_some_and(|f| f.holds(w));
    let mark = |r: &[(String, Source)], out: &mut BTreeSet<String>| {
        for (role, src) in r {
            if !FP8_ROLES.contains(&role.as_str()) {
                continue;
            }
            match src {
                Source::Tensor(w) if held(w) => {
                    out.insert(w.clone());
                }
                Source::Experts(e) => out.extend(e.first().into_iter().flatten().filter(|w| held(w)).cloned()),
                _ => {}
            }
        }
        let first = r.iter().find_map(|(role, s)| match s {
            Source::Experts(e) if role == "experts_gate_up" => e.first().and_then(|x| x.first()),
            _ => None,
        });
        if first.is_some_and(|w| held(w)) {
            out.extend(r.iter().filter(|(role, _)| role.starts_with("experts_")).flat_map(|(_, s)| s.names()));
        }
    };
    let mut fp8set = BTreeSet::new();
    // (the shapes of the tensors the full-size layout reads, which a loader by the kinds and a random model take: the norms, the latent attention and its
    // indexer, the experts, the dense MLP; the streams' too. Not yet the delta attention's)
    let mut shapes = std::collections::BTreeMap::new();
    let hc = streams.map(|s| s.streams).unwrap_or(0);
    let hc_shapes = [("hc_fn", vec![(2 + hc) * hc, hc * d]), ("hc_base", vec![(2 + hc) * hc]), ("hc_scale", vec![3])];
    let (mut blocks, mut roles) = (Vec::new(), Vec::new());
    // (the draft layers after the model's, GLM-5.3's: each a draft head of DeepSeek-V3's form (`MtpInput::Joined`), its block a latent attention with an
    // indexer of its own, then the experts, on a plain residual; GLM-5.3-Flash's sits in a model of streams and is left unread)
    let drafts = if streams.is_none() { mtp_layers } else { 0 };
    let (mut head_blocks, mut head_roles) = (Vec::new(), Vec::new());
    for (l, is_lin) in types.iter().copied().chain(std::iter::repeat_n(false, drafts)).enumerate() {
        let (blocks, roles) = if l < layers { (&mut blocks, &mut roles) } else { (&mut head_blocks, &mut head_roles) };
        let p = names::HF.layer(prefix, l);
        let residual = |table: names::Roles, r: &mut Vec<(String, Source)>| -> Residual {
            match streams {
                Some(s) => {
                    r.extend(names::roles(table, &p, 0, all));
                    Residual::Streams(s)
                }
                None => Residual::Add,
            }
        };
        let mut r = names::roles(part::NORM_IN, &p, 0, all);
        let mixer = if is_lin {
            r.extend(but(part::KDA, &p, 0, &["gate"]));
            Mixer::Linear(lin.clone())
        } else {
            let q = if q_lora.is_some() { "q" } else { "q_a q_a_norm q_b" };
            r.extend(names::roles(part::MLA, &p, 0, |x| x != "g" && !q.split(' ').any(|y| y == x)));
            let shared = itypes.get(l).is_some_and(|s| s == "shared");
            let indexer = match (ih, idim, itop) {
                (Some(h), Some(dim), Some(top)) => {
                    if !shared {
                        r.extend(names::roles(part::MLA_INDEXER, &p, 0, |x| flash || !x.starts_with("indexer_pool")));
                    }
                    // (with a rotary the indexer's queries and keys turn their first channels by it too: transformers' GlmMoeDsaIndexer)
                    let turned = if rope.is_some() { rd } else { 0 };
                    if turned > dim {
                        return err(format!("{mt}: an indexer of heads of {dim} under a rotary of {turned} channels"));
                    }
                    Some(Indexer { heads: h, head_dim: dim, top_k: top, block: pool, shared, form: IndexForm::LAYER_KEYS, rope: turned })
                }
                _ => None,
            };
            let qw = heads * (nope + rd);
            let mut sh = vec![("kv_a", vec![kvl + rd, d]), ("kv_a_norm", vec![kvl]), ("kv_b", vec![heads * (nope + vd), kvl]), ("o", vec![d, heads * vd])];
            match q_lora {
                Some(ql) => sh.extend([("q_a", vec![ql, d]), ("q_a_norm", vec![ql]), ("q_b", vec![qw, ql])]),
                None => sh.push(("q", vec![qw, d])),
            }
            if let (Some(h), Some(dim)) = (ih, idim) {
                sh.extend([
                    ("indexer_wq_b", vec![h * dim, q_lora.unwrap_or(d)]),
                    ("indexer_wk", vec![dim, d]),
                    ("indexer_k_norm", vec![dim]),
                    ("indexer_k_norm_bias", vec![dim]),
                    ("indexer_weights", vec![h, d]),
                    ("indexer_pool_ape", vec![pool, dim]),
                    ("indexer_pool_gate", vec![dim, d]),
                ]);
            }
            shape_roles(&mut shapes, &r, &sh);
            Mixer::Mla(Mla {
                heads,
                q_lora,
                kv_lora: kvl,
                nope,
                rope_dim: rd,
                v_dim: vd,
                rope: rope.clone(),
                q_norm: Norm { rounding: Rounding::Twice, ..Norm::rms(mla_eps) },
                kv_norm: Norm { rounding: Rounding::Twice, ..Norm::rms(mla_eps) },
                gate: None,
                scale,
                absorbed: true,
                indexer,
            })
        };
        mark(&r, &mut fp8set);
        let res = residual(part::HC_ATTN, &mut r);
        shape_roles(&mut shapes, &r, &[vec![("norm_in", vec![d])], hc_shapes.to_vec()].concat());
        blocks.push(Block { norm_in: n, body: Body::Mixer(mixer), norm_out: None, residual: res });
        roles.push(r);
        let sparse = match (&mlp_types, first_dense) {
            (Some(t), _) => t.get(l).copied().unwrap_or(true),
            (None, Some(f)) => l >= f,
            (None, None) => true,
        };
        let mut r = names::roles(part::NORM_POST, &p, 0, all);
        let body = if sparse {
            r.extend(names::roles(part::MOE, &p, ne, |x| moe.shared.is_some() || !x.starts_with("shared_")));
            shape_roles(
                &mut shapes,
                &r,
                &[
                    ("router", vec![ne, d]),
                    ("choice_bias", vec![ne]),
                    ("experts_gate_up", vec![moe_inter, d]),
                    ("experts_down", vec![d, moe_inter]),
                    ("shared_gate", vec![shared_inter, d]),
                    ("shared_up", vec![shared_inter, d]),
                    ("shared_down", vec![d, shared_inter]),
                ],
            );
            Body::Ffn(Ffn::Moe(moe.clone()))
        } else {
            r.extend(names::roles(part::MLP_BODY, &p, 0, all));
            // (a dense MLP's shapes, which the lowering to the executor reads its width from)
            shape_roles(&mut shapes, &r, &[("gate", vec![dense_inter, d]), ("up", vec![dense_inter, d]), ("down", vec![d, dense_inter])]);
            Body::Ffn(Ffn::Mlp(mlp))
        };
        mark(&r, &mut fp8set);
        let res = residual(part::HC_FFN, &mut r);
        shape_roles(&mut shapes, &r, &[vec![("norm_in", vec![d])], hc_shapes.to_vec()].concat());
        blocks.push(Block { norm_in: n, body, norm_out: None, residual: res });
        roles.push(r);
    }
    // (each draft head: its layer's two blocks, its own roles beside them (vLLM deepseek_mtp.py, SGLang deepseek_nextn.py: enorm, hnorm, eh_proj,
    // shared_head.norm; the model's embedding and LM head))
    let (mut mtp, mut mtp_roles) = (Vec::new(), Vec::new());
    for (j, (bs, rs)) in head_blocks.chunks(2).zip(head_roles.chunks(2)).enumerate() {
        let own = names::roles(part::MTP_LAYER, &names::HF.layer(prefix, layers + j), 0, all);
        shape_roles(&mut shapes, &own, &[("eh_proj", vec![d, 2 * d]), ("norm_embed", vec![d]), ("norm_hidden", vec![d]), ("final_norm", vec![d])]);
        mtp.push(MtpHead { blocks: bs.to_vec(), norm_embed: n, norm_hidden: n, final_norm: n, dspark: None, input: MtpInput::Joined });
        mtp_roles.push((rs.to_vec(), own));
    }
    let desc = ModelDesc {
        hidden: d,
        vocab,
        embed: Embed { scale: 1.0, norm: None, streams: streams.map_or(1, |s| s.streams), per_layer: None, table: TableAt::Gpu },
        blocks,
        final_norm: n,
        // (the logits in bf16: the release's lm_head is a bf16 product, its output the logits (modeling_glm5_next.py, modeling_glm_moe_dsa.py))
        head: Head { tied, softcap: None, scale: 1.0, collapse: if streams.is_some() { Collapse::Mean } else { Collapse::None }, rounding: Rounding::Once },
        mtp,
        vision: None,
        rope,
    };
    let mut out = finish(&k, desc, &names::HF, prefix, roles, mtp_layers)?;
    out.scales = fp8set.into_iter().map(|w| (w.clone(), format!("{w}_scale_inv"))).collect();
    shapes.insert(out.embed.clone(), vec![vocab, d]);
    shapes.insert(out.final_norm.clone(), vec![d]);
    if let Some(h) = &out.head {
        shapes.insert(h.clone(), vec![vocab, d]);
    }
    out.shapes = shapes;
    // (the draft layers follow the model's, under the same names: the ones described as heads are read)
    out.unread.extend((drafts..mtp_layers).map(|i| format!("{}.", names::HF.layer(prefix, layers + i))));
    out.mtp = mtp_roles;
    let _ = fp8.map(|f| f.block);
    Ok(out)
}

/// The roles whose matrices a release may hold in FP8 (GLM's `quantization_config`): the products of the attention and the feed-forward, and the indexer's
/// query and key products (GLM-5.3 holds them so; GLM-5.3-Flash's list leaves its attention out); not the routers, the norms, the convolutions or the
/// indexer's head weights (bf16 in both releases, fp32 in transformers: GlmMoeDsaPreTrainedModel's `_keep_in_fp32_modules`).
const FP8_ROLES: &[&str] = &[
    "indexer_wq_b",
    "indexer_wk",
    "q",
    "k",
    "v",
    "beta",
    "decay_a",
    "decay_b",
    "gate_a",
    "gate_b",
    "o",
    "q_a",
    "q_b",
    "kv_a",
    "kv_b",
    "gate",
    "up",
    "down",
    "shared_gate",
    "shared_up",
    "shared_down",
    "experts_gate_up",
    "experts_down",
];

/// The top-level keys of a multimodal config that the text model does not read.
fn top_keys(top: &Map<String, Value>, what: &str, known: &[&str]) -> Result<()> {
    for k in top.keys() {
        if !known.contains(&k.as_str()) && !inert(k) {
            return err(format!("{what}: config key {k} is not understood by the description builder"));
        }
    }
    Ok(())
}

/// Kimi-K3 (`kimi_k3`, a `kimi_linear` text model; Kimi-K3.1's `kimi_k31` is the same layout with the latent attention's sparse indexer, `dsa_config`): Kimi
/// delta attention and gated NoPE MLA, the depth mix over groups of `attn_res_block_size` layers, a dense first layer then the latent MoE, situ activations,
/// KimiRMSNorm (two roundings).
fn kimi(c: &ModelConfig) -> Result<Described> {
    let (mt, tc) = ("kimi_linear", &c.text);
    top_keys(&c.top, &c.model_type, &["text_config", "vision_config", "quantization_config", "tie_word_embeddings"])?;
    let k = Keys::new(tc, mt);
    let (d, layers, vocab, eps, tied) = common(&k, None)?;
    if k.str("hidden_act")? != Some("situ") {
        return err(format!("{mt}: only situ is supported"));
    }
    let situ = Act::Situ { beta: k.f64("activation_situ_beta")? as f32, linear: k.f64("activation_situ_linear_beta")? as f32 };
    let mlp = Mlp { act: situ, gated: true, bias: false, clamp: None };
    let twice = |e: f32| Norm { rounding: Rounding::Twice, ..Norm::rms(e) };
    for (x, want) in [("mla_use_nope", true), ("mla_use_output_gate", true), ("latent_moe_use_norm", true), ("moe_renormalize", true)] {
        if k.bool(x, want)? != want {
            return err(format!("{mt}: {x} = {} is not supported", !want));
        }
    }
    k.expect("moe_router_activation_func", &Value::from("sigmoid"))?;
    k.expect("num_expert_group", &Value::from(1))?;
    k.expect("topk_group", &Value::from(1))?;
    k.expect("topk_method", &Value::from("noaux_tc"))?;
    // (grouped top-k over one group of all the experts, which the two above make it, is the plain top-k)
    let _ = k.raw("use_grouped_topk");
    k.expect("moe_layer_freq", &Value::from(1))?;
    k.expect("rope_scaling", &Value::Null)?;
    k.opt_usize("max_position_embeddings")?;
    let la = match k.raw("linear_attn_config") {
        Some(Value::Object(o)) => o,
        _ => return err(format!("{mt}: no linear_attn_config")),
    };
    let lk = Keys::new(la, "linear_attn_config");
    let list = |x: &str| -> Result<BTreeSet<u64>> {
        lk.raw(x)
            .and_then(|v| v.as_array())
            .map(|a| a.iter().filter_map(|v| v.as_u64()).collect())
            .ok_or_else(|| Error(format!("{mt}: no linear_attn_config.{x}")))
    };
    let (kda_l, full_l) = (list("kda_layers")?, list("full_attn_layers")?);
    if (1..=layers as u64).any(|n| kda_l.contains(&n) == full_l.contains(&n)) {
        return err(format!("{mt}: kda_layers and full_attn_layers are not the {layers} layers (1-based) split in two"));
    }
    if !lk.bool("use_full_rank_gate", false)? {
        return err(format!("{mt}: a low-rank output gate in the delta attention is not supported"));
    }
    let (kh, kd) = (lk.usize("num_heads")?, lk.usize("head_dim")?);
    let lin = Linear {
        kind: LinearKind::KimiDelta,
        k_heads: kh,
        v_heads: kh,
        k_dim: kd,
        v_dim: kd,
        conv: lk.usize("short_conv_kernel_size")?,
        decay: Decay::Bounded(lk.f64("gate_lower_bound")? as f32),
        gate_act: Act::Sigmoid,
        norm_eps: 1e-5,
        state_f32: true,
    };
    lk.finish()?;
    let heads = k.usize("num_attention_heads")?;
    k.expect("num_key_value_heads", &Value::from(heads))?;
    let (nope, rd) = (k.usize("qk_nope_head_dim")?, k.usize("qk_rope_head_dim")?);
    let mla = Mla {
        heads,
        q_lora: Some(k.usize("q_lora_rank")?),
        kv_lora: k.usize("kv_lora_rank")?,
        nope,
        rope_dim: rd,
        v_dim: k.usize("v_head_dim")?,
        rope: None,
        q_norm: twice(1e-6),
        kv_norm: twice(1e-6),
        gate: Some(OutGate { act: Act::Sigmoid, from: GateFrom::Own, norm: None }),
        scale: ((nope + rd) as f32).powf(-0.5),
        absorbed: true,
        indexer: None,
    };
    // Kimi-K3.1's sparse attention (`dsa_config`): on the latent attention layers it names (1-based), the sparse attention's indexer of a key a token, the
    // gated form, its keys RMS-normed at the model's eps or LayerNorm-ed with a bias (`index_k_norm_type`); the indexer's query from the q latent
    let dsa = match k.raw("dsa_config") {
        None | Some(Value::Null) => None,
        Some(Value::Object(o)) => {
            let dk = Keys::new(o, "kimi_linear: dsa_config");
            let named = if dk.raw("dsa_layers").is_some() { "dsa_layers" } else { "layers" };
            let on: BTreeSet<u64> = dk
                .raw(named)
                .and_then(|v| v.as_array())
                .map(|a| a.iter().filter_map(Value::as_u64).collect())
                .ok_or_else(|| Error(format!("{mt}: dsa_config names no layers")))?;
            if let Some(l) = on.iter().find(|&&l| l == 0 || l > layers as u64 || kda_l.contains(&l)) {
                return err(format!("{mt}: dsa_config names layer {l}, which is not one of the latent attention's (1-based)"));
            }
            let key = match dk.str("index_k_norm_type")?.unwrap_or("rmsnorm") {
                "rmsnorm" => Norm::rms(eps),
                "layernorm" => Norm { kind: NormKind::Layer, ..Norm::rms(eps) },
                other => return err(format!("{mt}: dsa_config.index_k_norm_type {other}: rmsnorm or layernorm")),
            };
            // (a rotary on the indexer's first dims, picks shared across layers and an FP8 indexer are not described)
            for (x, want) in [("index_rope_dim", 0), ("index_topk_freq", 1), ("index_skip_topk_offset", 1)] {
                dk.expect(x, &Value::from(want))?;
            }
            dk.expect("indexer_use_fp8", &Value::Bool(false))?;
            let ix = Indexer {
                heads: dk.usize("index_num_heads")?,
                head_dim: dk.usize("index_head_dim")?,
                top_k: dk.usize("index_topk")?,
                block: 1,
                shared: false,
                form: IndexForm::Gated { key },
                rope: 0,
            };
            dk.finish()?;
            Some((on, ix))
        }
        Some(v) => return err(format!("{mt}: dsa_config {v}: an object")),
    };
    let dense = k.usize("first_k_dense_replace")?;
    let inter = k.usize("intermediate_size")?;
    let moe_inter = k.usize("moe_intermediate_size")?;
    let ne = k.usize("num_experts")?;
    // (the routed experts as released: MXFP4, 4-bit floats in groups of 32 with a byte's scale each, as compressed-tensors packs them; every other matrix bf16)
    let mxfp4 = match k.raw("quantization_config") {
        None => false,
        Some(q) => {
            kimi_mxfp4(q)?;
            true
        }
    };
    let mut scales = std::collections::BTreeMap::new();
    let n_shared = k.usize("num_shared_experts")?;
    let shared = n_shared > 0;
    let lat = k.usize("routed_expert_hidden_size")?;
    let moe = Moe {
        router: Router {
            score: RouterScore::Sigmoid,
            logit_scale: 1.0,
            bias: true,
            groups: None,
            top_k: k.usize("num_experts_per_token")?,
            renorm: true,
            scale: k.f64("routed_scaling_factor")? as f32,
            hash: false,
        },
        experts: ne,
        expert: mlp,
        shared: shared.then_some(mlp),
        shared_gate: false,
        latent: Some(Latent { dim: lat, norm_in: None, expert_norm: None, norm_out: Some(twice(eps)) }),
    };
    let group = k.usize("attn_res_block_size")?;
    let prefix = "language_model.model.";
    let n = twice(eps);
    let (mut blocks, mut roles, mut shapes) = (Vec::new(), Vec::new(), std::collections::BTreeMap::new());
    for l in 0..layers {
        let p = names::KIMI.layer(prefix, l);
        let mut r = names::roles(part::KIMI_ATTN_MIX, &p, 0, all);
        let mixer = if kda_l.contains(&(l as u64 + 1)) {
            let x = but(part::KDA, &p, 0, &["gate_a", "gate_b"]);
            // (each matrix's shape, which the lowering to the executor sizes its products by and names its tensors with; the forget gate's low rank the delta
            // heads' size)
            let qd = kh * kd;
            shape_roles(
                &mut shapes,
                &x,
                &[
                    ("q", vec![qd, d]),
                    ("k", vec![qd, d]),
                    ("v", vec![qd, d]),
                    ("gate", vec![qd, d]),
                    ("decay_a", vec![kd, d]),
                    ("beta", vec![kh, d]),
                    ("o", vec![d, qd]),
                    ("decay_b", vec![qd, kd]),
                ],
            );
            r.extend(x);
            Mixer::Linear(lin.clone())
        } else {
            let x = but(part::MLA, &p, 0, &["q"]);
            let (ql, hv) = (mla.q_lora.unwrap_or(0), heads * mla.v_dim);
            shape_roles(
                &mut shapes,
                &x,
                &[
                    ("q_a", vec![ql, d]),
                    ("g", vec![hv, d]),
                    ("kv_a", vec![mla.kv_lora + rd, d]),
                    ("q_b", vec![heads * (nope + rd), ql]),
                    ("kv_b", vec![heads * (nope + mla.v_dim), mla.kv_lora]),
                    ("o", vec![d, hv]),
                ],
            );
            r.extend(x);
            let indexer = dsa.as_ref().filter(|(on, _)| on.contains(&(l as u64 + 1))).map(|(_, ix)| ix.clone());
            if let Some((IndexForm::Gated { key }, ix)) = indexer.as_ref().map(|ix| (ix.form, ix)) {
                let x =
                    names::roles(part::MLA_INDEXER, &p, 0, |x| !x.starts_with("indexer_pool") && (key.kind == NormKind::Layer || x != "indexer_k_norm_bias"));
                let (ih, id) = (ix.heads, ix.head_dim);
                shape_roles(&mut shapes, &x, &[("indexer_wq_b", vec![ih * id, ql]), ("indexer_wk", vec![id, d]), ("indexer_weights", vec![ih, d])]);
                r.extend(x);
            }
            Mixer::Mla(Mla { indexer, ..mla.clone() })
        };
        blocks.push(Block { norm_in: n, body: Body::Mixer(mixer), norm_out: None, residual: Residual::DepthMix(DepthMixP { starts: l % group == 0 }) });
        roles.push(r);
        let mut r = names::roles(part::KIMI_FFN_MIX, &p, 0, all);
        let body = if l < dense {
            r.extend(names::roles(part::MLP_BODY, &p, 0, all));
            shape_roles(&mut shapes, &r, &[("gate", vec![inter, d]), ("up", vec![inter, d]), ("down", vec![d, inter])]);
            Body::Ffn(Ffn::Mlp(mlp))
        } else {
            let mut x = names::roles(part::KIMI_MOE, &p, ne, |x| shared || !x.starts_with("shared_"));
            // (MXFP4 experts as released: each matrix its codes, `.weight_packed`, and its scales)
            for (_, src) in x.iter_mut().filter(|(role, _)| mxfp4 && role.starts_with("experts_")) {
                if let Source::Experts(e) = src {
                    for w in e.iter_mut().flatten() {
                        w.push_str("_packed");
                        scales.insert(w.clone(), w.replace(".weight_packed", ".weight_scale"));
                    }
                }
            }
            let si = n_shared * moe_inter;
            // (an expert's matrices at their rows and columns, on the latent: its codes as released are two to a byte, their scales `scales`')
            shape_roles(
                &mut shapes,
                &x,
                &[
                    ("router", vec![ne, d]),
                    ("latent_down", vec![lat, d]),
                    ("latent_up", vec![d, lat]),
                    ("experts_gate_up", vec![moe_inter, lat]),
                    ("experts_down", vec![lat, moe_inter]),
                    ("shared_gate", vec![si, d]),
                    ("shared_up", vec![si, d]),
                    ("shared_down", vec![d, si]),
                ],
            );
            r.extend(x);
            Body::Ffn(Ffn::Moe(moe.clone()))
        };
        blocks.push(Block { norm_in: n, body, norm_out: None, residual: Residual::DepthMix(DepthMixP { starts: false }) });
        roles.push(r);
    }
    // (the checkpoint's next-token-prediction layers, which the description does not use yet)
    let mtp_layers = k.opt_usize("num_nextn_predict_layers")?.unwrap_or(0);
    let desc = ModelDesc {
        hidden: d,
        vocab,
        embed: Embed { scale: 1.0, norm: None, streams: 1, per_layer: None, table: TableAt::Gpu },
        blocks,
        final_norm: n,
        head: Head { tied, softcap: None, scale: 1.0, collapse: Collapse::DepthMix, rounding: Rounding::F32 },
        mtp: Vec::new(),
        vision: None,
        rope: None,
    };
    let mut out = finish(&k, desc, &names::KIMI, prefix, roles, mtp_layers)?;
    // (the embedding's and the head's too: every matrix a product multiplies by or a token's row is read from)
    shapes.extend(std::iter::once(&out.embed).chain(&out.head).map(|n| (n.clone(), vec![vocab, d])));
    (out.scales, out.shapes) = (scales, shapes);
    Ok(out)
}

/// Kimi-K3's `quantization_config` as released: compressed-tensors' `mxfp4-pack-quantized`, 4-bit floats in symmetric groups of 32, the attention, the shared experts, the
/// dense layer's MLP and the head left out (so only the routed experts are packed). Anything else is refused with what differs.
fn kimi_mxfp4(q: &Value) -> Result<()> {
    let s = |v: &Value, k: &str| v.get(k).and_then(Value::as_str).map(str::to_string);
    let what = "kimi_linear: quantization_config";
    if s(q, "quant_method").as_deref() != Some("compressed-tensors") || s(q, "format").as_deref() != Some("mxfp4-pack-quantized") {
        return err(format!("{what}: {:?} {:?}, where the description reads compressed-tensors' mxfp4-pack-quantized", s(q, "quant_method"), s(q, "format")));
    }
    let groups = q.get("config_groups").and_then(Value::as_object).ok_or_else(|| Error(format!("{what}: no config_groups")))?;
    if groups.len() != 1 {
        return err(format!("{what}: {} config groups, where the description reads one", groups.len()));
    }
    let w = groups.values().next().and_then(|g| g.get("weights")).ok_or_else(|| Error(format!("{what}: a config group with no weights")))?;
    let n = |k: &str| w.get(k).and_then(Value::as_u64);
    if n("num_bits") != Some(4)
        || n("group_size") != Some(32)
        || s(w, "type").as_deref() != Some("float")
        || s(w, "strategy").as_deref() != Some("group")
        || w.get("symmetric").and_then(Value::as_bool) != Some(true)
    {
        return err(format!("{what}: weights {w}, where the description reads 4-bit floats in symmetric groups of 32"));
    }
    let ignore: Vec<String> =
        q.get("ignore").and_then(Value::as_array).map(|a| a.iter().filter_map(|x| x.as_str().map(str::to_string)).collect()).unwrap_or_default();
    for part in ["self_attn", "shared_experts", "mlp", "lm_head"] {
        if !ignore.iter().any(|x| x.contains(part)) {
            return err(format!("{what}: the {part} matrices are packed too, where the description reads them in bf16"));
        }
    }
    Ok(())
}

/// Muse Spark (`muse_spark`): GQA with weightless q/k norms, rope and a window on the sliding layers only, a gated output with a per-head
/// norm; a normed embedding; sandwich norms with Muse's gated residual; the latent MoE in every layer; a soft-capped head.
fn muse(tc: &Map<String, Value>, prefix: &str) -> Result<Described> {
    let mt = "muse_spark_text";
    let k = Keys::new(tc, mt);
    let (d, layers, vocab, eps, tied) = common(&k, None)?;
    let pe = k.f64("post_norm_eps")? as f32;
    k.expect("hidden_act", &Value::from("silu"))?;
    k.expect("attention_bias", &Value::Bool(false))?;
    k.expect("num_shared_experts", &Value::from(0))?;
    k.opt_usize("max_position_embeddings")?;
    k.usize("intermediate_size")?;
    let (heads, kv, hd) = (k.usize("num_attention_heads")?, k.usize("num_key_value_heads")?, k.usize("head_dim")?);
    let window = k.usize("sliding_window")?;
    let strs = |x: &str| -> Result<Vec<String>> {
        Ok(k.raw(x).and_then(|v| v.as_array()).ok_or_else(|| Error(format!("{mt}: no {x}")))?.iter().map(|v| v.as_str().unwrap_or("").to_string()).collect())
    };
    let (lt, mlt) = (strs("layer_types")?, strs("mlp_layer_types")?);
    if lt.len() != layers || mlt.iter().any(|m| m != "sparse") {
        return err(format!("{mt}: layer types other than one per layer, every MLP sparse"));
    }
    let thetas: Vec<f64> = k
        .raw("layer_rope_theta")
        .and_then(|v| v.as_array())
        .ok_or_else(|| Error(format!("{mt}: no layer_rope_theta")))?
        .iter()
        .map(|v| v.as_f64().unwrap_or(0.0))
        .collect();
    if let Some(Value::Object(rp)) = k.raw("rope_parameters") {
        if rp.get("rope_type").and_then(|v| v.as_str()).is_some_and(|t| t != "default") {
            return err(format!("{mt}: rope scaling is not supported"));
        }
    }
    let qk = k.f64("qk_scale_factor")? as f32;
    let out_mult = k.f64("output_multiplier")? as f32;
    let softcap = k.opt_f64("final_logit_softcapping")?.map(|v| v as f32);
    let temperature = k.f64("residual_gate_temperature")? as f32;
    let (ne, lat) = (k.usize("num_local_experts")?, k.usize("moe_hidden_size")?);
    k.usize("moe_intermediate_size")?;
    let moe = Moe {
        router: Router {
            score: RouterScore::Sigmoid,
            logit_scale: out_mult,
            bias: true,
            groups: None,
            top_k: k.usize("num_experts_per_tok")?,
            renorm: true,
            scale: k.f64("routed_scaling_factor")? as f32,
            hash: false,
        },
        experts: ne,
        expert: Mlp { act: Act::Silu, gated: true, bias: false, clamp: None },
        shared: None,
        shared_gate: false,
        latent: Some(Latent {
            dim: lat,
            norm_in: Some(Norm::rms_plus_one(eps)),
            expert_norm: Some(Norm { weight: NormWeight::InputPlusOne, ..Norm::rms(pe) }),
            norm_out: None,
        }),
    };
    let bare = Norm::rms_bare(eps);
    let (mut blocks, mut roles) = (Vec::new(), Vec::new());
    for l in 0..layers {
        let sliding = lt[l] == "sliding_attention";
        let theta = thetas.get(l).copied().unwrap_or(0.0);
        if sliding != (theta > 0.0) {
            return err(format!("{mt}: layer {l}: a rotary that is not on exactly the window's layers"));
        }
        let a = Attn {
            heads,
            kv_heads: kv,
            head_dim: hd,
            v_head_dim: hd,
            window: sliding.then_some(window),
            sinks: false,
            rope: sliding.then(|| Rope::plain(theta, hd)),
            q_norm: Some(bare),
            k_norm: Some(bare),
            v_norm: None,
            gate: Some(OutGate { act: Act::Sigmoid, from: GateFrom::Own, norm: Some(bare) }),
            softcap: None,
            bias: false,
            scale: qk / (hd as f32).sqrt(),
            k_eq_v: false,
            indexer: None,
            rounding: AttnRounding::Sdpa,
            kv_from: None,
        };
        let p = names::HF.layer(prefix, l);
        blocks.push(Block {
            norm_in: Norm::rms_plus_one(eps),
            body: Body::Mixer(Mixer::Attention(a)),
            norm_out: Some(Norm::rms_bare(pe)),
            residual: Residual::Gated(GateP { temperature }),
        });
        roles.push(names::roles(part::MUSE_ATTN, &p, 0, all));
        blocks.push(Block {
            norm_in: Norm::rms_plus_one(eps),
            body: Body::Ffn(Ffn::Moe(moe.clone())),
            norm_out: Some(Norm { weight: NormWeight::InputPlusOne, ..Norm::rms(pe) }),
            residual: Residual::Gated(GateP { temperature }),
        });
        roles.push(names::roles(part::MUSE_MOE, &p, 0, all));
    }
    let desc = ModelDesc {
        hidden: d,
        vocab,
        embed: Embed { scale: 1.0, norm: Some(bare), streams: 1, per_layer: None, table: TableAt::Gpu },
        blocks,
        final_norm: Norm::rms(eps),
        // (the release's head is bf16: its product, the output multiplier's product and the soft-cap's three operations each rounded, as muse_ref.rs and the
        // engine's muse_softcap compute them)
        head: Head { tied, softcap, scale: out_mult, collapse: Collapse::None, rounding: Rounding::Twice },
        mtp: Vec::new(),
        vision: None,
        rope: None,
    };
    finish(&k, desc, &names::HF, prefix, roles, 0)
}

/// DeepSeek-V4 (`deepseek_v4`): the compressed attention (`kinds::Comp`), MoE with hash routing in the first layers then sqrt-softplus
/// with a choice bias, SwiGLU limits, 4 mHC streams and the weighted hyper-head. Two config flavours: the release's (`compress_ratios`
/// per layer, a ratio of 4 being the overlapping windows with the indexer; `num_hash_layers`; YaRN `rope_scaling` and
/// `compress_rope_theta` on the compressing layers; FP8 matrices and FP4 experts with `.scale` tensors; names without a prefix) and
/// transformers' (`layer_types`, `compress_rates`, `rope_parameters` main and compress, `mlp_layer_types`; names under `model.`).
fn deepseek_v4(tc: &Map<String, Value>) -> Result<Described> {
    let mt = "deepseek_v4";
    let k = Keys::new(tc, mt);
    let (d, layers, vocab, eps, tied) = common(&k, None)?;
    let max_pos = k.opt_usize("max_position_embeddings")?;
    k.expect("hidden_act", &Value::from("silu"))?;
    k.expect("attention_bias", &Value::Bool(false))?;
    k.expect("mlp_bias", &Value::Bool(false))?;
    k.expect("scoring_func", &Value::from("sqrtsoftplus"))?;
    k.expect("topk_method", &Value::from("noaux_tc"))?;
    k.expect("num_key_value_heads", &Value::from(1))?;
    k.expect("n_shared_experts", &Value::from(1))?;
    k.expect("norm_topk_prob", &Value::Bool(true))?;
    k.expect("router_jitter_noise", &Value::from(0.0))?;
    k.raw("architectures");
    let released = k.m.contains_key("compress_ratios");
    let fp4 = k.str("expert_dtype")? == Some("fp4");
    let fp8_block = Fp8::parse(k.raw("quantization_config"), mt)?.map(|f| f.block);
    let fp8 = fp8_block.is_some();
    let (heads, hd, rd) = (k.usize("num_attention_heads")?, k.usize("head_dim")?, k.usize("qk_rope_head_dim")?);
    k.opt_f64("partial_rotary_factor")?;
    let (q_lora, o_lora, o_groups, window) =
        (k.usize("q_lora_rank")?, k.usize("o_lora_rank")?, k.usize("o_groups")?, k.opt_usize("sliding_window")?.unwrap_or(128));
    let (ih, idim, itop) = (k.usize("index_n_heads")?, k.usize("index_head_dim")?, k.usize("index_topk")?);
    let (ne, top_k) = (k.usize("n_routed_experts")?, k.usize("num_experts_per_tok")?);
    k.usize("moe_intermediate_size")?;
    let limit = k.opt_f64("swiglu_limit")?.filter(|&l| l > 0.0).map(|l| l as f32);
    let scale = k.opt_f64("routed_scaling_factor")?.unwrap_or(1.0) as f32;
    let streams = StreamsP {
        lagged: false,
        streams: k.usize("hc_mult")?,
        sinkhorn_iters: k.opt_usize("hc_sinkhorn_iters")?.unwrap_or(20),
        eps: k.opt_f64("hc_eps")?.unwrap_or(1e-6) as f32,
        norm_eps: eps,
        rounding: Rounding::Once,
    };
    let mtp_layers = k.opt_usize("num_nextn_predict_layers")?.unwrap_or(0);
    // each layer's compression ratio and whether it is CSA (overlap and indexer), its rope, and whether its MoE is hash-routed
    let (ratios, csa, hash, main_rope, comp_rope): (Vec<usize>, Vec<bool>, Vec<bool>, Rope, Rope) = if released {
        let r: Vec<usize> =
            k.raw("compress_ratios").and_then(|v| v.as_array()).map(|a| a.iter().map(|x| x.as_u64().unwrap_or(0) as usize).collect()).unwrap_or_default();
        if r.len() < layers {
            return err(format!("{mt}: {} compress_ratios for {layers} layers", r.len()));
        }
        let r = r[..layers].to_vec();
        let nh = k.opt_usize("num_hash_layers")?.unwrap_or(0);
        let mut main = Rope::plain(k.opt_f64("rope_theta")?.unwrap_or(10_000.0), rd);
        main.style = RopeStyle::Interleaved;
        let mut comp_m = Map::new();
        if let Some(Value::Object(o)) = k.raw("rope_scaling") {
            comp_m = o.clone();
        }
        comp_m.insert("rope_theta".into(), Value::from(k.opt_f64("compress_rope_theta")?.unwrap_or(160_000.0)));
        // (the release's model.py stretches the frequencies only: YaRN without the attention factor transformers puts on cos and sin)
        if comp_m.contains_key("factor") && !comp_m.contains_key("attention_factor") {
            comp_m.insert("attention_factor".into(), Value::from(1.0));
        }
        let mut comp = rope(&comp_m, rd, max_pos, &format!("{mt} rope_scaling"))?;
        comp.style = RopeStyle::Interleaved;
        (r.clone(), r.iter().map(|&x| x == 4).collect(), (0..layers).map(|l| l < nh).collect(), main, comp)
    } else {
        let rates = match k.raw("compress_rates") {
            Some(Value::Object(o)) => o.clone(),
            _ => Map::new(),
        };
        let rate = |t: &str| rates.get(t).and_then(|v| v.as_u64()).map(|x| x as usize);
        let types: Vec<String> = k
            .raw("layer_types")
            .and_then(|v| v.as_array())
            .ok_or_else(|| Error(format!("{mt}: no layer_types")))?
            .iter()
            .map(|x| x.as_str().unwrap_or("").to_string())
            .collect();
        let mut r = Vec::new();
        for t in types.iter().take(layers) {
            r.push(match t.as_str() {
                "sliding_attention" => 0,
                "compressed_sparse_attention" | "heavily_compressed_attention" => rate(t).ok_or_else(|| Error(format!("{mt}: no compress_rates.{t}")))?,
                o => return err(format!("{mt}: layer type {o}")),
            });
        }
        let mlp: Vec<String> =
            k.raw("mlp_layer_types").and_then(|v| v.as_array()).map(|a| a.iter().map(|x| x.as_str().unwrap_or("").to_string()).collect()).unwrap_or_default();
        let rp = match k.raw("rope_parameters") {
            Some(Value::Object(o)) => o.clone(),
            _ => return err(format!("{mt}: no rope_parameters")),
        };
        let one = |t: &str| -> Result<Rope> {
            let o = rp.get(t).and_then(|v| v.as_object()).ok_or_else(|| Error(format!("{mt}: no rope_parameters.{t}")))?;
            let mut o = o.clone();
            o.insert("partial_rotary_factor".into(), Value::from(1.0));
            let mut r = rope(&o, rd, max_pos, &format!("{mt} rope_parameters.{t}"))?;
            r.style = RopeStyle::Interleaved;
            Ok(r)
        };
        k.opt_f64("rope_theta")?;
        k.opt_f64("compress_rope_theta")?;
        (
            r,
            types.iter().take(layers).map(|t| t == "compressed_sparse_attention").collect(),
            (0..layers).map(|l| mlp.get(l).is_some_and(|m| m == "hash_moe")).collect(),
            one("main")?,
            one("compress")?,
        )
    };
    let cv = if released { &names::DEEPSEEK } else { &names::DEEPSEEK_HF };
    // (the release's quantized matrices, each with its `.scale`: the FP8 products of the attention and the shared experts, the FP4 routed experts)
    let mut scales = std::collections::BTreeMap::new();
    let q8 = ["q_a", "q_b", "kv", "o_a", "o_b", "idx_q_b", "shared_gate", "shared_up", "shared_down", "main_proj"];
    let mark = |r: &[(String, Source)], scales: &mut std::collections::BTreeMap<String, String>| {
        for (_, src) in r.iter().filter(|(role, _)| released && if role.starts_with("experts_") { fp4 } else { fp8 && q8.contains(&role.as_str()) }) {
            for w in src.names() {
                scales.insert(w.clone(), w.replace(".weight", ".scale"));
            }
        }
    };
    let mlp = Mlp { act: Act::Silu, gated: true, bias: false, clamp: limit };
    let (mut blocks, mut roles) = (Vec::new(), Vec::new());
    // (the last layer's attention and feed-forward, of which a DSpark drafter's are made)
    let (mut comp_template, mut moe_template) = (None, None);
    for l in 0..layers {
        let p = cv.layer("", l);
        let (ratio, is_csa) = (ratios[l], csa[l]);
        let mut r = names::roles(part::V4_ATTN, &p, 0, all);
        r.extend(names::roles(if released { part::V4_KV_NORM } else { part::V4_KV_NORM_HF }, &p, 0, all));
        r.extend(names::roles(part::V4_SINKS, &p, 0, all));
        if ratio > 0 {
            r.extend(names::roles(part::V4_COMP, &p, 0, all));
        }
        if ratio > 0 && is_csa {
            r.extend(names::roles(part::V4_INDEXER, &p, 0, all));
        }
        r.extend(names::roles(part::HC_ATTN, &p, 0, all));
        mark(&r, &mut scales);
        let c = Comp {
            heads,
            head_dim: hd,
            q_lora,
            o_groups,
            o_lora,
            window,
            ratio,
            overlap: ratio > 0 && is_csa,
            rope: if ratio > 0 { comp_rope.clone() } else { main_rope.clone() },
            // (its compressed entries' keys normed by the compressor's RMS norm, `idx_norm`)
            indexer: (ratio > 0 && is_csa).then_some(Indexer {
                heads: ih,
                head_dim: idim,
                top_k: itop,
                block: ratio,
                shared: false,
                form: IndexForm::Gated { key: Norm::rms(eps) },
                rope: 0,
            }),
            norm_eps: eps,
            ape: true,
            q_head_norm: true,
            share: None,
        };
        comp_template = Some(c.clone());
        blocks.push(Block { norm_in: Norm::rms(eps), body: Body::Mixer(Mixer::Compressed(c)), norm_out: None, residual: Residual::Streams(streams) });
        roles.push(r);
        // (a hash-routed layer's table of experts by token, else the choice bias)
        let mut r = names::roles(part::V4_MOE, &p, ne, |x| x != if hash[l] { "choice_bias" } else { "hash" });
        r.extend(names::roles(part::HC_FFN, &p, 0, all));
        mark(&r, &mut scales);
        let moe = Moe {
            router: Router { score: RouterScore::SqrtSoftplus, logit_scale: 1.0, bias: !hash[l], groups: None, top_k, renorm: true, scale, hash: hash[l] },
            experts: ne,
            expert: mlp,
            shared: Some(mlp),
            shared_gate: false,
            latent: None,
        };
        moe_template = Some(moe.clone());
        blocks.push(Block { norm_in: Norm::rms(eps), body: Body::Ffn(Ffn::Moe(moe)), norm_out: None, residual: Residual::Streams(streams) });
        roles.push(r);
    }
    for x in [
        "expert_dtype",
        "quantization_config",
        "rope_scaling",
        "num_hash_layers",
        "compress_ratios",
        "rope_theta",
        "compress_rope_theta",
        "layer_types",
        "mlp_layer_types",
        "compress_rates",
        "rope_parameters",
    ] {
        k.raw(x);
    }
    // (a DSpark drafter: `n_mtp_layers` of the decoder's layers with the window alone, the format's default where the config does not say)
    let draft = match (comp_template, moe_template) {
        (Some(c), Some(mut m)) => {
            let window = Comp { ratio: 0, overlap: false, rope: main_rope.clone(), indexer: None, ..c };
            (m.router.hash, m.router.bias) = (false, true);
            dspark_head(&k, k.opt_usize("n_mtp_layers")?.unwrap_or(DSPARK_LAYERS), window, m, Residual::Streams(streams), Norm::rms(eps))?
        }
        _ => None,
    };
    let desc = ModelDesc {
        hidden: d,
        vocab,
        embed: Embed { scale: 1.0, norm: None, streams: streams.streams, per_layer: None, table: TableAt::Gpu },
        blocks,
        final_norm: Norm::rms(eps),
        head: Head { tied, softcap: None, scale: 1.0, collapse: Collapse::Weighted { eps: streams.eps, norm_eps: eps }, rounding: Rounding::F32 },
        mtp: draft.into_iter().collect(),
        vision: None,
        rope: None,
    };
    let mut out = finish(&k, desc, cv, "", roles, mtp_layers)?;
    // (the 0731 release's drafter's tensors, under DeepSeek's names)
    if let (Some(h), true) = (out.desc.mtp.first(), released) {
        let (blocks, own) = dspark_roles(h.blocks.len() / 2, ne, false);
        for r in blocks.iter().chain(std::iter::once(&own)) {
            mark(r, &mut scales);
        }
        out.mtp = vec![(blocks, own)];
    }
    out.scales = scales;
    out.fp8_block = fp8_block;
    Ok(out)
}

/// A DSpark drafter's layers where its config has the drafter's keys and no `n_mtp_layers`: the format's default, which its readers define (vLLM's
/// `models/deepseek_v4/nvidia/dspark.py`: `n_mtp_layers`, else 3). A directory is held to its index (`describe_dir`): a checkpoint that lacks a draft layer
/// the count reads is refused, its tensors named.
const DSPARK_LAYERS: usize = 3;

/// A DeepSeek release's DSpark drafter (`dspark_*` keys; `kinds::DSpark`), where the config has one: `layers` draft layers, each the model's
/// decoder layer with the window alone (the attention `window` with the model's own form, the feed-forward `moe`) on the model's residual rule,
/// drafting `dspark_block_size` tokens from main_x of `dspark_target_layer_ids`; a Markov head of `dspark_markov_rank` and a confidence head.
fn dspark_head(k: &Keys, layers: usize, window: Comp, moe: Moe, residual: Residual, norm: Norm) -> Result<Option<MtpHead>> {
    let Some(block) = k.opt_usize("dspark_block_size")? else { return Ok(None) };
    let targets: Vec<usize> = k
        .raw("dspark_target_layer_ids")
        .and_then(|v| v.as_array())
        .map(|a| a.iter().filter_map(|x| x.as_u64().map(|n| n as usize)).collect())
        .ok_or_else(|| Error("dspark_target_layer_ids: a list of layers".into()))?;
    let dspark = DSpark { block, noise: k.usize("dspark_noise_token_id")? as u32, targets, markov_rank: k.usize("dspark_markov_rank")?, confidence: true };
    let b = |body: Body| Block { norm_in: norm, body, norm_out: None, residual: residual.clone() };
    let blocks = (0..layers).flat_map(|_| [b(Body::Mixer(Mixer::Compressed(window.clone()))), b(Body::Ffn(Ffn::Moe(moe.clone())))]).collect();
    Ok(Some(MtpHead { blocks, norm_embed: norm, norm_hidden: norm, final_norm: norm, dspark: Some(dspark), input: MtpInput::Joined }))
}

/// A DSpark drafter's tensors as DeepSeek releases name them (`mtp.{i}.`): each draft layer's attention (the window's: no compressor) and its
/// experts as the model's layers name them; the head's own on its first module (`main_proj`, `main_norm`) and its last (the Markov head's two factors,
/// `markov_w1` and `markov_w2`, V4.1's `embed` and `head`; the confidence head, the final norm, and V4's hyper-head `hc_head_*`).
fn dspark_roles(layers: usize, experts: usize, v41: bool) -> (Vec<Vec<(String, Source)>>, Vec<(String, Source)>) {
    let mut blocks = Vec::new();
    for i in 0..layers {
        let p = format!("mtp.{i}");
        let mut r = names::roles(part::V4_ATTN, &p, 0, all);
        for t in [part::V4_KV_NORM, part::V4_SINKS, part::HC_ATTN] {
            r.extend(names::roles(t, &p, 0, all));
        }
        blocks.push(r);
        let mut r = names::roles(part::V4_MOE, &p, experts, |x| x != "hash");
        r.extend(names::roles(part::HC_FFN, &p, 0, all));
        blocks.push(r);
    }
    let (first, last) = ("mtp.0".to_string(), format!("mtp.{}", layers.saturating_sub(1)));
    let t = |at: &str, n: &str| Source::Tensor(format!("{at}.{n}"));
    let (w1, w2) = if v41 { ("markov_head.embed.weight", "markov_head.head.weight") } else { ("markov_head.markov_w1.weight", "markov_head.markov_w2.weight") };
    let mut own = vec![
        ("main_proj".to_string(), t(&first, "main_proj.weight")),
        ("main_norm".to_string(), t(&first, "main_norm.weight")),
        ("markov_embed".to_string(), t(&last, w1)),
        ("markov_head".to_string(), t(&last, w2)),
        ("confidence".to_string(), t(&last, "confidence_head.proj.weight")),
        ("final_norm".to_string(), t(&last, "norm.weight")),
    ];
    if !v41 {
        own.extend(
            [("out_hc_fn", "hc_head_fn"), ("out_hc_base", "hc_head_base"), ("out_hc_scale", "hc_head_scale")].map(|(r, n)| (r.to_string(), t(&last, n))),
        );
    }
    (blocks, own)
}

/// DeepSeek-V4.1 (`deepseek_v41`, its text model `deepseek_v41_text`, a vision tower beside it): V4's decoder with V4.1's three mechanisms, each a
/// parameter of a kind V4 has. Its attention is CSA2 (`Comp` with no window bias and no per-head query norm): `compress_ratios` 0 (the window
/// alone), 1 or 2, every compressing layer reading the entries of the most recent kv source at or below it (`kv_source_layer_ids`) and the picks of
/// the most recent index source (`index_source_layer_ids`), the candidate source (`candidate_source_layer_id`) publishing blocks of
/// `candidate_block_size` entries, `candidate_topk_blocks` kept, which the index sources after it score inside (`CompShare`). Its residual is lagged
/// mHC (`StreamsP::lagged`, the head's collapse `Collapse::LastPre`: no hyper-head). On `engram_layer_ids` an Engram block (`NgramForm::Engram`)
/// comes before the layer's attention, its tables read by rows on the host (`TableAt::Host`). The MoE is V4's sqrt-softplus router (no hash layers;
/// `gate_temp` 1 alone). Not read: the DSpark draft layers (counted), the vision tower, the routers' image-token bias.
fn deepseek_v41(c: &ModelConfig) -> Result<Described> {
    let (mt, tc) = ("deepseek_v41", &c.text);
    // (the image token beside the vision tower's config: the tower is not read)
    top_keys(&c.top, mt, &["text_config", "vision_config", "quantization_config", "architectures", "tie_word_embeddings", "image_token_id"])?;
    let k = Keys::new(tc, mt);
    let (d, layers, vocab, eps, tied) = common(&k, c.top.get("tie_word_embeddings").and_then(|v| v.as_bool()))?;
    let max_pos = k.opt_usize("max_position_embeddings")?;
    k.expect("hidden_act", &Value::from("silu"))?;
    k.expect("attention_bias", &Value::Bool(false))?;
    k.expect("scoring_func", &Value::from("sqrtsoftplus"))?;
    k.expect("topk_method", &Value::from("noaux_tc"))?;
    k.expect("num_key_value_heads", &Value::from(1))?;
    k.expect("n_shared_experts", &Value::from(1))?;
    k.expect("norm_topk_prob", &Value::Bool(true))?;
    // (the router's logits divided by the gate temperature before the score: as their factor 1 / it, the same at the release's 1)
    let temperature = match (k.opt_f64("gate_temperature")?, k.opt_f64("gate_temp")?) {
        (Some(a), Some(b)) if a != b => return err(format!("{mt}: gate_temperature {a} and gate_temp {b} differ")),
        (a, b) => a.or(b).unwrap_or(1.0),
    };
    if temperature <= 0.0 {
        return err(format!("{mt}: a gate temperature of {temperature}"));
    }
    // (the Engram tables' scale blocks: one a 32 values, which the scales' shapes say)
    k.expect("engram_block_size", &Value::from(32))?;
    k.expect("num_hash_layers", &Value::from(0))?;
    let q = c.top.get("quantization_config").or(k.raw("quantization_config"));
    let fp8_block = Fp8::parse(
        q.map(|v| {
            let mut o = v.as_object().cloned().unwrap_or_default();
            // (the experts' and the Engram tables' formats, said beside the dense matrices' FP8)
            o.retain(|key, _| !key.starts_with("engram_") && !matches!(key.as_str(), "expert_dtype" | "scale_fmt" | "activation_scheme"));
            Value::Object(o)
        })
        .as_ref(),
        mt,
    )?
    .map(|f| f.block);
    let fp8 = fp8_block.is_some();
    let fp4 = q.and_then(|v| v.get("expert_dtype")).or(k.raw("expert_dtype")).and_then(|v| v.as_str()) == Some("fp4");
    let (heads, hd, rd) = (k.usize("num_attention_heads")?, k.usize("head_dim")?, k.usize("qk_rope_head_dim")?);
    let (q_lora, o_lora, o_groups, window) =
        (k.usize("q_lora_rank")?, k.usize("o_lora_rank")?, k.usize("o_groups")?, k.opt_usize("sliding_window")?.unwrap_or(128));
    let (ih, idim, itop) = (k.usize("index_n_heads")?, k.usize("index_head_dim")?, k.usize("index_topk")?);
    let (ne, top_k) = (k.usize("n_routed_experts")?, k.usize("num_experts_per_tok")?);
    k.usize("moe_intermediate_size")?;
    let limit = k.opt_f64("swiglu_limit")?.filter(|&l| l > 0.0).map(|l| l as f32);
    let scale = k.opt_f64("routed_scaling_factor")?.unwrap_or(1.0) as f32;
    let streams = StreamsP {
        streams: k.usize("hc_mult")?,
        sinkhorn_iters: k.opt_usize("hc_sinkhorn_iters")?.unwrap_or(20),
        eps: k.opt_f64("hc_eps")?.unwrap_or(1e-6) as f32,
        norm_eps: eps,
        rounding: Rounding::Once,
        lagged: true,
    };
    // (the release's RMSNorm: fp32, the weight times the normed value cast to bf16 once; Engram's gate normalises in fp32 and keeps it)
    let rn = Norm { rounding: Rounding::Once, ..Norm::rms(eps) };
    let mtp_layers = k.opt_usize("num_nextn_predict_layers")?.unwrap_or(0);
    // (a list of counts: one that is not a list, or holds an entry that is not a count, is refused by name, never read as the entries that are)
    let list = |x: &str| -> Result<Vec<usize>> {
        let v = k.raw(x).ok_or_else(|| Error(format!("{mt}: no {x}")))?;
        let a = v.as_array().ok_or_else(|| Error(format!("{mt}: {x} = {v} is not a list")))?;
        a.iter().map(|e| e.as_u64().map(|n| n as usize).ok_or_else(|| Error(format!("{mt}: {x} = {v}: {e} is not a count")))).collect()
    };
    let ratios = list("compress_ratios")?;
    if ratios.len() < layers || ratios[..layers].iter().any(|r| *r > 2) {
        return err(format!("{mt}: compress_ratios {ratios:?} for {layers} layers: 0, 1 or 2 each"));
    }
    let (kv_src, idx_src) = (list("kv_source_layer_ids")?, list("index_source_layer_ids")?);
    let cand = k.raw("candidate_source_layer_id").and_then(|v| v.as_i64()).filter(|&c| c >= 0).map(|c| c as usize);
    let cand_shape = match cand {
        Some(_) => Some((k.usize("candidate_block_size")?, k.usize("candidate_topk_blocks")?)),
        None => None,
    };
    // (each compressing layer's sources: the most recent at or below it, which compresses by the same ratio)
    let source = |ids: &[usize], l: usize| ids.iter().copied().filter(|&s| s <= l).max().filter(|&s| ratios[s] == ratios[l]);
    for l in (0..layers).filter(|&l| ratios[l] > 0) {
        if source(&kv_src, l).is_none() || source(&idx_src, l).is_none() {
            return err(format!("{mt}: layer {l} compresses by {}, and the most recent kv or index source at or below it does not", ratios[l]));
        }
        // (a candidate block is a span of positions of the candidate source's ratio: an index source of another ratio is not described)
        if let Some(c) = cand.filter(|&c| c < l && idx_src.contains(&l) && ratios[c] != ratios[l]) {
            return err(format!("{mt}: index source {l} compresses by {}, the candidate source {c} by {}", ratios[l], ratios[c]));
        }
    }
    let mut main = Rope::plain(k.opt_f64("rope_theta")?.unwrap_or(10_000.0), rd);
    main.style = RopeStyle::Interleaved;
    // (the compressing layers' YaRN: rope_scaling, else the rope_yarn_* keys, else the release's defaults: 16, 65,536, 32, 1)
    let mut comp_m = match k.raw("rope_scaling") {
        Some(Value::Object(o)) => o.clone(),
        _ => {
            let mut o = Map::new();
            o.insert("type".into(), Value::from("yarn"));
            for (key, own, dflt) in [
                ("factor", "rope_yarn_factor", 16.0),
                ("original_max_position_embeddings", "rope_yarn_original_max_position_embeddings", 65536.0),
                ("beta_fast", "rope_yarn_beta_fast", 32.0),
                ("beta_slow", "rope_yarn_beta_slow", 1.0),
            ] {
                let v = k.opt_f64(own)?.unwrap_or(dflt);
                o.insert(key.into(), if key == "original_max_position_embeddings" { Value::from(v as u64) } else { Value::from(v) });
            }
            o
        }
    };
    comp_m.insert("rope_theta".into(), Value::from(k.opt_f64("compress_rope_theta")?.unwrap_or(160_000.0)));
    if comp_m.contains_key("factor") && !comp_m.contains_key("attention_factor") {
        comp_m.insert("attention_factor".into(), Value::from(1.0));
    }
    let mut comp_rope = rope(&comp_m, rd, max_pos, &format!("{mt} rope_scaling"))?;
    comp_rope.style = RopeStyle::Interleaved;
    // Engram's layers, each a block before the layer's attention: its tables the next primes past engram_vocab_size, their rows engram_num_embeddings
    // (no engram_layer_ids: a model with no Engram layers; one that is there and is not a list of layers is refused, not read as none)
    let engram: Vec<usize> = match k.raw("engram_layer_ids") {
        None => Vec::new(),
        Some(_) => list("engram_layer_ids")?,
    };
    let engram_kind = |place: usize, layer: usize| -> Result<Ngram> {
        let (order, heads, head_dim) = (k.usize("engram_max_ngram_size")?, k.usize("engram_n_heads")?, k.usize("engram_head_dim")?);
        let (base, vocab) = (k.usize("engram_vocab_size")? as u64, k.usize("engram_compressed_vocab_size")?);
        // (sizes that are no model's: an order under 2 has no table, no heads or head width divides by zero, and past these the builder would keep finding
        // primes; the bounds are the other n-gram form's, above)
        let tables = order.saturating_sub(1).saturating_mul(heads);
        if order < 2 || order > 64 || heads == 0 || head_dim == 0 || vocab == 0 || base == 0 || base > 1 << 40 || tables.saturating_mul(place + 1) > 4096 {
            return err(format!(
                "{mt}: Engram of engram_max_ngram_size {order}, engram_n_heads {heads}, engram_head_dim {head_dim}, engram_vocab_size {base}, engram_compressed_vocab_size {vocab}: an order of 2 to 64, heads and widths above 0, a vocabulary up to 2^40 and at most 4096 tables"
            ));
        }
        let n = Ngram {
            order,
            heads,
            dim: (order - 1) * heads * head_dim,
            base,
            vocab,
            seed: 10007 * layer as u64,
            index: place,
            eos: k.usize("engram_pad_token_id")? as u32,
            streams: streams.streams,
            conv: 0,
            dilation: 1,
            norm: Norm::rms(eps),
            form: NgramForm::Engram,
            // (its tables read by rows on the host: the release's are 384M rows of 256 a layer, FP8 with a scale a run of 32, about 98 GB each)
            table: TableAt::Host,
        };
        let rows = list("engram_num_embeddings")?;
        let want = super::reference::ngram::tables(&n).0.iter().sum::<u64>();
        if rows.get(place).map(|&r| r as u64) != Some(want) {
            return err(format!("{mt}: engram_num_embeddings {rows:?}: Engram layer {layer}'s tables hold {want} rows"));
        }
        Ok(n)
    };
    let cv = &names::DEEPSEEK_V41;
    let mut scales = std::collections::BTreeMap::new();
    let q8 = ["q_a", "q_b", "kv", "o_a", "o_b", "idx_q_b", "shared_gate", "shared_up", "shared_down", "ngram", "main_proj"];
    let mark = |r: &[(String, Source)], scales: &mut std::collections::BTreeMap<String, String>| {
        for (_, src) in r.iter().filter(|(role, _)| if role.starts_with("experts_") { fp4 } else { fp8 && q8.contains(&role.as_str()) }) {
            for w in src.names() {
                scales.insert(w.clone(), w.replace(".weight", ".scale"));
            }
        }
    };
    let mlp = Mlp { act: Act::Silu, gated: true, bias: false, clamp: limit };
    let (mut blocks, mut roles, mut attn_block) = (Vec::new(), Vec::new(), vec![0usize; layers]);
    // (the last layer's attention and feed-forward, of which the drafter's are made)
    let (mut comp_template, mut moe_template) = (None, None);
    for l in 0..layers {
        let p = cv.layer("", l);
        if let Some(place) = engram.iter().position(|&e| e == l) {
            let r = names::roles(part::ENGRAM, &p, 0, all);
            mark(&r, &mut scales);
            blocks.push(Block { norm_in: Norm::rms(eps), body: Body::Ngram(engram_kind(place, l)?), norm_out: None, residual: Residual::Add });
            roles.push(r);
        }
        attn_block[l] = blocks.len();
        let ratio = ratios[l];
        let (is_kv, is_idx) = (kv_src.contains(&l), idx_src.contains(&l));
        let mut r = names::roles(part::V4_ATTN, &p, 0, all);
        r.extend(names::roles(part::V4_KV_NORM, &p, 0, all));
        r.extend(names::roles(part::V4_SINKS, &p, 0, all));
        if ratio > 0 && is_kv {
            r.extend(names::roles(part::V41_COMP, &p, 0, |x| ratio > 1 || x != "comp_gate"));
        }
        if ratio > 0 && is_idx {
            r.extend(names::roles(part::V41_INDEXER, &p, 0, |x| is_kv || !x.starts_with("idx_k")));
        }
        r.extend(names::roles(part::HC_ATTN, &p, 0, all));
        mark(&r, &mut scales);
        let share = (ratio > 0).then(|| CompShare {
            entries: (!is_kv).then(|| attn_block[source(&kv_src, l).expect("checked above")]),
            picks: (!is_idx).then(|| attn_block[source(&idx_src, l).expect("checked above")]),
            candidates: cand.filter(|&c| is_idx && c < l).map(|c| attn_block[c]),
            publish: cand_shape.filter(|_| cand == Some(l)),
        });
        let comp = Comp {
            heads,
            head_dim: hd,
            q_lora,
            o_groups,
            o_lora,
            window,
            ratio,
            overlap: false,
            rope: if ratio > 0 { comp_rope.clone() } else { main.clone() },
            indexer: (ratio > 0).then_some(Indexer {
                heads: ih,
                head_dim: idim,
                top_k: itop,
                block: ratio,
                shared: false,
                form: IndexForm::Gated { key: Norm::rms(eps) },
                rope: 0,
            }),
            norm_eps: eps,
            ape: false,
            q_head_norm: false,
            share,
        };
        comp_template = Some(comp.clone());
        blocks.push(Block { norm_in: rn, body: Body::Mixer(Mixer::Compressed(comp)), norm_out: None, residual: Residual::Streams(streams) });
        roles.push(r);
        let mut r = names::roles(part::V4_MOE, &p, ne, |x| x != "hash");
        r.extend(names::roles(part::HC_FFN, &p, 0, all));
        mark(&r, &mut scales);
        let moe = Moe {
            router: Router {
                score: RouterScore::SqrtSoftplus,
                logit_scale: (1.0 / temperature) as f32,
                bias: true,
                groups: None,
                top_k,
                renorm: true,
                scale,
                hash: false,
            },
            experts: ne,
            expert: mlp,
            shared: Some(mlp),
            shared_gate: false,
            latent: None,
        };
        moe_template = Some(moe.clone());
        blocks.push(Block { norm_in: rn, body: Body::Ffn(Ffn::Moe(moe)), norm_out: None, residual: Residual::Streams(streams) });
        roles.push(r);
    }
    for x in ["rope_scaling", "rope_theta", "compress_rope_theta", "candidate_source_layer_id", "candidate_block_size", "candidate_topk_blocks", "expert_dtype"]
    {
        k.raw(x);
    }
    for x in [
        "engram_max_ngram_size",
        "engram_n_heads",
        "engram_head_dim",
        "engram_vocab_size",
        "engram_compressed_vocab_size",
        "engram_pad_token_id",
        "engram_num_embeddings",
    ] {
        k.raw(x);
    }
    // (the DSpark drafter: its layers the decoder's with the window alone, their experts its own count)
    let window_attn = Comp { ratio: 0, rope: main.clone(), indexer: None, share: None, ..comp_template.ok_or_else(|| Error(format!("{mt}: no layers")))? };
    let mut dmoe = moe_template.ok_or_else(|| Error(format!("{mt}: no layers")))?;
    dmoe.router.top_k = k.opt_usize("dspark_num_experts_per_tok")?.unwrap_or(top_k);
    dmoe.experts = k.opt_usize("dspark_n_routed_experts")?.unwrap_or(ne);
    let n_draft = k.opt_usize("n_mtp_layers")?.unwrap_or(mtp_layers);
    let draft = dspark_head(&k, n_draft, window_attn, dmoe, Residual::Streams(streams), rn)?;
    let desc = ModelDesc {
        hidden: d,
        vocab,
        embed: Embed { scale: 1.0, norm: None, streams: streams.streams, per_layer: None, table: TableAt::Gpu },
        blocks,
        final_norm: rn,
        head: Head { tied, softcap: None, scale: 1.0, collapse: Collapse::LastPre, rounding: Rounding::F32 },
        mtp: draft.into_iter().collect(),
        vision: None,
        rope: None,
    };
    let mut out = finish(&k, desc, cv, "", roles, mtp_layers)?;
    if let Some(h) = out.desc.mtp.first() {
        let experts = h.blocks.iter().find_map(|b| if let Body::Ffn(Ffn::Moe(m)) = &b.body { Some(m.experts) } else { None }).unwrap_or(0);
        let (blocks, own) = dspark_roles(h.blocks.len() / 2, experts, true);
        for r in blocks.iter().chain(std::iter::once(&own)) {
            mark(r, &mut scales);
        }
        out.mtp = vec![(blocks, own)];
    }
    out.scales = scales;
    out.fp8_block = fp8_block;
    out.shapes = v41_shapes(&out, k.usize("moe_intermediate_size")?);
    Ok(out)
}

/// Each tensor's shape in a DeepSeek-V4.1 description (its blocks', its drafter's), from the kinds' widths and the experts' `inter`: what a lowering sizes
/// its matrices by and what a random model of the config is made at.
fn v41_shapes(out: &Described, inter: usize) -> std::collections::BTreeMap<String, Vec<usize>> {
    let (d, v) = (out.desc.hidden, out.desc.vocab);
    let mut sh = std::collections::BTreeMap::new();
    let mut put = |b: &Block, r: &[(String, Source)]| {
        let n = match b.residual {
            Residual::Streams(s) => s.streams,
            _ => 1,
        };
        let hc = [("hc_fn", vec![(2 + n) * n, n * d]), ("hc_base", vec![(2 + n) * n]), ("hc_scale", vec![3])];
        let want: Vec<(&str, Vec<usize>)> = match &b.body {
            Body::Mixer(Mixer::Compressed(c)) => {
                let (q, hd, (ih, idim)) = (c.q_lora, c.head_dim, c.indexer.as_ref().map_or((0, 0), |x| (x.heads, x.head_dim)));
                vec![
                    ("norm_in", vec![d]),
                    ("q_a", vec![q, d]),
                    ("q_a_norm", vec![q]),
                    ("q_b", vec![c.heads * hd, q]),
                    ("kv", vec![hd, d]),
                    ("kv_norm", vec![hd]),
                    ("o_a", vec![c.o_groups * c.o_lora, c.heads * hd / c.o_groups.max(1)]),
                    ("o_b", vec![d, c.o_groups * c.o_lora]),
                    ("sinks", vec![c.heads]),
                    ("comp_kv", vec![hd, d]),
                    ("comp_gate", vec![hd, d]),
                    ("comp_norm", vec![hd]),
                    ("idx_q_b", vec![ih * idim, q]),
                    ("idx_weights", vec![ih, d]),
                    ("idx_k", vec![idim, hd]),
                    ("idx_k_norm", vec![idim]),
                ]
            }
            Body::Ffn(Ffn::Moe(m)) => vec![
                ("norm_in", vec![d]),
                ("router", vec![m.experts, d]),
                ("choice_bias", vec![m.experts]),
                ("experts_gate_up", vec![inter, d]),
                ("experts_down", vec![d, inter]),
                ("shared_gate", vec![inter, d]),
                ("shared_up", vec![inter, d]),
                ("shared_down", vec![d, inter]),
            ],
            Body::Ngram(g) => {
                let rows = super::reference::ngram::tables(g).0.iter().sum::<u64>() as usize;
                vec![
                    ("ngram", vec![rows, g.dim / ((g.order - 1) * g.heads)]),
                    ("kv", vec![(g.streams + 1) * d, g.dim]),
                    ("norm_in", vec![g.streams, d]),
                    ("norm_key", vec![g.streams, d]),
                ]
            }
            _ => Vec::new(),
        };
        shape_roles(&mut sh, r, &[want, hc.to_vec()].concat());
    };
    for (b, r) in out.desc.blocks.iter().zip(&out.blocks) {
        put(b, r);
    }
    if let (Some(h), Some((blocks, own))) = (out.desc.mtp.first(), out.mtp.first()) {
        for (b, r) in h.blocks.iter().zip(blocks) {
            put(b, r);
        }
        if let Some(ds) = &h.dspark {
            let own_shapes = [
                ("main_proj", vec![d, ds.targets.len() * d]),
                ("main_norm", vec![d]),
                ("markov_embed", vec![v, ds.markov_rank]),
                ("markov_head", vec![v, ds.markov_rank]),
                ("confidence", vec![1, d + ds.markov_rank]),
                ("final_norm", vec![d]),
            ];
            shape_roles(&mut sh, own, &own_shapes);
        }
    }
    for n in [Some(&out.embed), out.head.as_ref()].into_iter().flatten() {
        sh.insert(n.clone(), vec![v, d]);
    }
    sh.insert(out.final_norm.clone(), vec![d]);
    sh
}

/// Nemotron 3 (`nemotron_h`, `NemotronHForCausalLM`): a layer is one block, `x + mixer(norm(x))`, its mixer named by the pattern:
/// Mamba-2 (`M`), attention without any position embedding (`*`), or a mixture of experts (`E`) whose routed experts run in a latent
/// (two projections around them; none where `moe_latent_size` is null) with a shared expert on the full vector. The router is
/// DeepSeek-V3's: sigmoid scores, a bias for the choice, the chosen scores normed and times `routed_scaling_factor`. The pattern is
/// `hybrid_override_pattern`, or a list of types (`layers_block_type`). The draft head (`mtp.*`) is counted and not described yet.
fn nemotron_h(tc: &Map<String, Value>) -> Result<Described> {
    let k = Keys::new(tc, "nemotron_h");
    let (d, vocab) = (k.usize("hidden_size")?, k.usize("vocab_size")?);
    let tied = k.bool("tie_word_embeddings", false)?;
    // (one eps for the layers' norms, the final norm and the Mamba-2 gated norm; a config with two that differ does not say which is whose)
    let eps = match (k.opt_f64("layer_norm_epsilon")?, k.opt_f64("norm_eps")?) {
        (Some(a), Some(b)) if a != b => return err(format!("nemotron_h: layer_norm_epsilon {a} and norm_eps {b} differ")),
        (a, b) => a.or(b).unwrap_or(1e-5) as f32,
    };
    let kinds: Vec<char> = if let Some(p) = k.str("hybrid_override_pattern")? {
        p.chars().collect()
    } else if let Some(a) = k.raw("layers_block_type").or(k.raw("layer_types")).and_then(|v| v.as_array()) {
        a.iter()
            .map(|x| match x.as_str() {
                Some("mamba" | "linear_attention") => Ok('M'),
                Some("attention" | "full_attention") => Ok('*'),
                Some("moe") => Ok('E'),
                other => err(format!("nemotron_h: a layer of type {other:?} is not supported (mamba, attention, moe)")),
            })
            .collect::<Result<_>>()?
    } else {
        return err("nemotron_h: no hybrid_override_pattern");
    };
    (k.raw("layers_block_type"), k.raw("layer_types"));
    if let Some(c) = kinds.iter().find(|c| !"M*E".contains(**c)) {
        return err(format!("nemotron_h: a layer {c:?} in hybrid_override_pattern is not supported (M, * and E)"));
    }
    if kinds.is_empty() || k.opt_usize("num_hidden_layers")?.is_some_and(|n| n != kinds.len()) {
        return err(format!("nemotron_h: {} layers in the pattern, num_hidden_layers {:?}", kinds.len(), tc.get("num_hidden_layers")));
    }
    // settings the description does not take otherwise
    for (key, want) in [
        ("use_bias", Value::Bool(false)),
        ("mlp_bias", Value::Bool(false)),
        ("attention_bias", Value::Bool(false)),
        ("mamba_proj_bias", Value::Bool(false)),
        ("use_conv_bias", Value::Bool(true)),
        ("residual_in_fp32", Value::Bool(false)),
        ("mamba_hidden_act", Value::from("silu")),
        ("mamba_ssm_cache_dtype", Value::from("float32")),
    ] {
        k.expect(key, &want)?;
    }
    // (when the shared experts' work overlaps the routed experts' exchange between GPUs: a schedule, not arithmetic; the releases say true, the transformers
    // default; either is read)
    k.bool("moe_shared_expert_overlap", true)?;
    if k.raw("sliding_window").is_some() {
        return err("nemotron_h: a sliding window is not supported");
    }
    // (read and without effect: the attention has no position embedding; the time steps and the rescaling are how the weights were
    // first set; the kernels and the logits kept are transformers' own choices; no layer of the pattern is a plain MLP)
    for x in [
        "rope_theta",
        "partial_rotary_factor",
        "time_step_floor",
        "time_step_max",
        "time_step_min",
        "time_step_rank",
        "time_step_scale",
        "time_step_limit",
        "rescale_prenorm_residual",
        "use_mamba_kernels",
        "num_logits_to_keep",
        "intermediate_size",
        "mtp_hybrid_override_pattern",
    ] {
        k.raw(x);
    }
    let has = |c: char| kinds.contains(&c);
    let mamba = if has('M') {
        let (heads, head_dim, groups) = (k.usize("mamba_num_heads")?, k.usize("mamba_head_dim")?, k.usize("n_groups")?);
        if heads == 0 || groups == 0 || !heads.is_multiple_of(groups) {
            return err(format!("nemotron_h: {heads} Mamba-2 heads in {groups} groups"));
        }
        // (the inner width is the heads' channels; `expand` is not what sets it: the toy model of the GPU check has 64 heads of 64 on a width of 256)
        k.raw("expand");
        Some(Mamba2 {
            heads,
            head_dim,
            state: k.usize("ssm_state_size")?,
            groups,
            conv: k.usize("conv_kernel")?,
            chunk: k.opt_usize("chunk_size")?.unwrap_or(128),
            norm_eps: eps,
        })
    } else {
        for x in ["mamba_num_heads", "mamba_head_dim", "n_groups", "expand", "ssm_state_size", "conv_kernel", "chunk_size"] {
            k.raw(x);
        }
        None
    };
    let attn = {
        let (heads, kv) = (k.usize("num_attention_heads")?, k.usize("num_key_value_heads")?);
        if heads == 0 || kv == 0 || !heads.is_multiple_of(kv) {
            return err(format!("nemotron_h: {heads} query heads to {kv} KV heads"));
        }
        let hd = k.opt_usize("head_dim")?.unwrap_or(d / heads);
        Attn {
            heads,
            kv_heads: kv,
            head_dim: hd,
            v_head_dim: hd,
            window: None,
            sinks: false,
            rope: None,
            q_norm: None,
            k_norm: None,
            v_norm: None,
            gate: None,
            softcap: None,
            bias: false,
            scale: (hd as f32).powf(-0.5),
            k_eq_v: false,
            indexer: None,
            rounding: AttnRounding::Flash,
            kv_from: None,
        }
    };
    let moe_keys =
        ["n_routed_experts", "num_experts_per_tok", "moe_intermediate_size", "moe_shared_expert_intermediate_size", "moe_latent_size", "mlp_hidden_act"];
    // (an expert's width inside and the shared expert's)
    let mut moe_widths = (0, 0);
    let moe = if has('E') {
        let (experts, top_k) = (k.usize("n_routed_experts")?, k.usize("num_experts_per_tok")?);
        if top_k == 0 || top_k > experts {
            return err(format!("nemotron_h: {top_k} experts a token of {experts}"));
        }
        for g in ["n_group", "topk_group", "n_shared_experts"] {
            k.expect(g, &Value::from(1))?;
        }
        moe_widths = (k.usize("moe_intermediate_size")?, k.usize("moe_shared_expert_intermediate_size")?);
        let mlp = Mlp { act: act(k.str("mlp_hidden_act")?.unwrap_or("relu2"))?, gated: false, bias: false, clamp: None };
        Some(Moe {
            router: Router {
                score: RouterScore::Sigmoid,
                logit_scale: 1.0,
                bias: true,
                groups: None,
                top_k,
                renorm: k.bool("norm_topk_prob", true)?,
                scale: k.opt_f64("routed_scaling_factor")?.unwrap_or(1.0) as f32,
                hash: false,
            },
            experts,
            expert: mlp,
            shared: Some(mlp),
            shared_gate: false,
            latent: k.opt_usize("moe_latent_size")?.filter(|&l| l > 0).map(|dim| Latent { dim, norm_in: None, expert_norm: None, norm_out: None }),
        })
    } else {
        for x in moe_keys.iter().chain(&["n_group", "topk_group", "n_shared_experts", "norm_topk_prob", "routed_scaling_factor"]) {
            k.raw(x);
        }
        None
    };
    let mtp_layers = k.opt_usize("num_nextn_predict_layers")?.unwrap_or(0);
    let n = Norm::rms(eps);
    let (mut blocks, mut roles, mut shapes) = (Vec::new(), Vec::new(), std::collections::BTreeMap::new());
    for (l, c) in kinds.iter().enumerate() {
        let p = names::NEMOTRON.layer("", l);
        // (the shapes: each matrix [rows, columns] as the heads and widths make it, each norm, bias and per-head value its width)
        let (body, r, want) = match (c, &mamba, &moe) {
            ('M', Some(mb), _) => {
                let (nh, inner) = (mb.heads, mb.heads * mb.head_dim);
                let cd = inner + 2 * mb.groups * mb.state;
                let want = vec![
                    ("in", vec![inner + cd + nh, d]),
                    ("conv", vec![cd, 1, mb.conv]),
                    ("conv_bias", vec![cd]),
                    ("a_log", vec![nh]),
                    ("d", vec![nh]),
                    ("dt_bias", vec![nh]),
                    ("norm", vec![inner]),
                    ("o", vec![d, inner]),
                ];
                (Body::Mixer(Mixer::Mamba2(mb.clone())), names::roles(part::NH_MAMBA, &p, 0, all), want)
            }
            // (no latent: the experts read the model's width, and there are no projections around them)
            ('E', _, Some(me)) => {
                let ((inter, shared), lat) = (moe_widths, me.latent.as_ref().map_or(0, |x| x.dim));
                let w = if lat > 0 { lat } else { d };
                let want = vec![
                    ("router", vec![me.experts, d]),
                    ("choice_bias", vec![me.experts]),
                    ("latent_down", vec![lat, d]),
                    ("latent_up", vec![d, lat]),
                    ("shared_up", vec![shared, d]),
                    ("shared_down", vec![d, shared]),
                    ("experts_up", vec![inter, w]),
                    ("experts_down", vec![w, inter]),
                ];
                let r = names::roles(part::NH_MOE, &p, me.experts, |x| me.latent.is_some() || !x.starts_with("latent_"));
                (Body::Ffn(Ffn::Moe(me.clone())), r, want)
            }
            _ => {
                let (q, kv) = (attn.heads * attn.head_dim, attn.kv_heads * attn.head_dim);
                let want = vec![("q", vec![q, d]), ("k", vec![kv, d]), ("v", vec![kv, d]), ("o", vec![d, q])];
                (Body::Mixer(Mixer::Attention(attn.clone())), names::roles(part::NH_ATTN, &p, 0, all), want)
            }
        };
        shape_roles(&mut shapes, &r, &want);
        shape_roles(&mut shapes, &r, &[("norm_in", vec![d])]);
        blocks.push(Block { norm_in: n, body, norm_out: None, residual: Residual::Add });
        roles.push(r);
    }
    let desc = ModelDesc {
        hidden: d,
        vocab,
        embed: Embed { scale: 1.0, norm: None, streams: 1, per_layer: None, table: TableAt::Gpu },
        blocks,
        final_norm: n,
        head: Head { tied, softcap: None, scale: 1.0, collapse: Collapse::None, rounding: Rounding::F32 },
        mtp: Vec::new(),
        vision: None,
        rope: None,
    };
    let mut out = finish(&k, desc, &names::NEMOTRON, "", roles, mtp_layers)?;
    shapes.insert(out.embed.clone(), vec![vocab, d]);
    shapes.insert(out.final_norm.clone(), vec![d]);
    if let Some(h) = &out.head {
        shapes.insert(h.clone(), vec![vocab, d]);
    }
    out.shapes = shapes;
    Ok(out)
}
