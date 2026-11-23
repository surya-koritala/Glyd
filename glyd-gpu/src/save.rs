//! A bf16 checkpoint packed on the CPU and saved as glyd-v4 (glyd-v3 in the
//! 12-bit layout; glyd-v1 and glyd-v2, the tiered layout of glyd 0.20 to 0.27,
//! are read, not written), byte for byte as
//! `python -m glyd.gpu pack` saves it (bindings/python/glyd/gpu/format.py's
//! save_pretrained over the model hf.py's from_pretrained loads): the same
//! tensors, names, shards, glyd.json and index; and a saved one checked
//! (`verify`).
//!
//! Python saves what transformers' model holds, in its module order; here the
//! families' layouts are written out (`Family`), each checked against
//! Python's saves: Qwen3 (Qwen3-0.6B to 8B) and GraniteMoe
//! (granite-3.1-3b-a800m), and tiny random checkpoints of Llama, Qwen2,
//! Mistral and Granite. Anything else is refused, with Python's command.
//!
//! A planned save ([`save_planned`]) also takes FP8 matrices (E4M3 with block scales, [`crate::fp8`]), read as the bf16 they dequantise to and packed as a bf16 one is; the
//! families above are bf16 checkpoints alone. In a save in a block layout (glyd-v7) a plan's linear may name a row layout (`i6r`, `i5r`, `i4r`: [`Layout::Int6Row`]), which packs the
//! weight's transpose a row at a time; [`verify`] decodes those on the CPU.
//!
//! Memory: a save holds the shard its writer writes, the one it fills (about
//! 5 GB each) and at most one shard's worth of weights read or packed past the
//! one it waits for (and that one), whatever the threads: about 15 GB at most
//! for a model of several shards, less for one of fewer. A verify on the CPU
//! holds a pack and its matrix decoded for each of its threads, and one more.

use crate::codebook;
use crate::hbexport::{HbExport, Matrix};
use crate::json::{self, Value};
use crate::pack;
use crate::safetensors::{Checkpoint, TensorInfo};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::{Condvar, Mutex};

/// A shard is closed once it holds this many bytes (save_pretrained's shard_bytes).
pub const SHARD_BYTES: u64 = 5_000_000_000;
/// The files copied from the source beside the weights (format.py's FILES).
const FILES: &[&str] = &[
    "config.json",
    "generation_config.json",
    "tokenizer*",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab*",
    "merges.txt",
    "*.model",
    "chat_template*",
    "preprocessor_config.json",
    "processor_config.json",
];

fn bad(why: impl std::fmt::Display) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, why.to_string())
}

/// An io error with the path it came from.
fn at(path: &Path) -> impl FnOnce(io::Error) -> io::Error + '_ {
    move |e| io::Error::new(e.kind(), format!("{}: {e}", path.display()))
}

/// A model's checkpoint: its directory, and the repo and revision glyd.json names as its source (transformers'
/// name_or_path and commit hash).
#[derive(Clone, Debug)]
pub struct Source {
    pub dir: PathBuf,
    pub repo: String,
    pub revision: Option<String>,
}

/// The Hugging Face hub cache: $HF_HUB_CACHE, else $HF_HOME/hub, else ~/.cache/huggingface/hub.
fn hub_cache() -> PathBuf {
    if let Some(p) = std::env::var_os("HF_HUB_CACHE") {
        return p.into();
    }
    if let Some(p) = std::env::var_os("HF_HOME") {
        return Path::new(&p).join("hub");
    }
    let cache =
        std::env::var_os("XDG_CACHE_HOME").map(PathBuf::from).unwrap_or_else(|| Path::new(&std::env::var_os("HOME").unwrap_or_default()).join(".cache"));
    cache.join("huggingface").join("hub")
}

fn is_hash(s: &str) -> bool {
    s.len() == 40 && s.bytes().all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
}

impl Source {
    /// `model`: a directory, or a Hub repo id already in the local cache (its main revision; nothing is downloaded).
    pub fn find(model: &str) -> io::Result<Source> {
        let dir = Path::new(model);
        if dir.is_dir() {
            // transformers' commit hash for a directory: its snapshot's, where it is one
            let p = dir.join("config.json").to_string_lossy().replace('\\', "/");
            let revision = p
                .split("snapshots/")
                .nth(1)
                .and_then(|r| r.split('/').next())
                .filter(|r| is_hash(r) && p.contains(&format!("snapshots/{r}/")))
                .map(str::to_string);
            return Ok(Source { dir: dir.to_path_buf(), repo: model.to_string(), revision });
        }
        let repo = hub_cache().join(format!("models--{}", model.replace('/', "--")));
        let revision = std::fs::read_to_string(repo.join("refs").join("main")).map(|r| r.trim().to_string()).map_err(|_| {
            io::Error::new(
                io::ErrorKind::NotFound,
                format!(
                    "{model}: no such directory, nor in the Hugging Face cache ({}): download it first (glyd fetch {model}), or give its directory",
                    hub_cache().display()
                ),
            )
        })?;
        let dir = repo.join("snapshots").join(&revision);
        if !dir.is_dir() || !is_hash(&revision) {
            return Err(bad(format!("{model}: the cache's main revision {revision} has no snapshot there")));
        }
        Ok(Source { dir, repo: model.to_string(), revision: Some(revision) })
    }
}

/// What one save writes, in save_pretrained's order.
#[derive(Clone, Debug)]
enum Put {
    /// A bf16 tensor as it is: saved as `name` from the source's `from` (cast: an f32 one there, made bf16 as
    /// transformers loads it into a bf16 model).
    Copy { name: String, from: String, cast: bool },
    /// Linears' weights stacked by rows (q, k, v; gate, up; or one) as one pack under the first's module path, then
    /// their biases; read from the source's `from` (the members' paths there). `layout`: the block layout this pack is
    /// in where it is not the save's (a plan's `"layout"`: a save in one block layout may hold a pack in another, its
    /// entry in glyd.json saying which).
    Linear { members: Vec<String>, from: Vec<String>, layout: Option<Layout> },
    /// A mixture of experts' weight (its E matrices, held [E, out, in]) as one pack of them stacked, [E out, in]:
    /// saved under its module (`module.glyd_WEIGHT_...`) from the source's `from`.
    Experts { module: String, weight: String, from: String, experts: usize },
    /// A routed expert's matrix from a calibration export (hbexport), packed as it is in a codebook row layout (c3r or c2r by the expert's bits): the pack of `module` (its tensor `module.weight`, which the source has with
    /// the shape the plan says), the export's MoE layer `layer` (its number among the MoE layers), expert and matrix (`down`: the export's downT, the pack of the transpose of the tensor).
    Codebook { module: String, layer: usize, expert: usize, down: bool },
    /// The signs (+1 and -1, I8) of the rotation of the export's MoE layer `layer`, saved as `name`.
    Signs { name: String, layer: usize },
}

/// A decoder layer's module in its family's order (transformers' registration order); each in the checkpoint.
#[derive(Clone, Copy)]
enum Child {
    /// An nn.Linear (its weight; its bias where the checkpoint has one).
    Linear(&'static str),
    /// A module whose parameters are saved as they are (a norm; a router of its own class): its parameter's name,
    /// and the source's where it is another.
    Plain(&'static str, &'static str),
    /// A mixture of experts' Experts module: its weights and the source's for them.
    Experts(&'static str, &'static [(&'static str, &'static str)]),
}

/// A family's decoder layer: its blocks (a module path within the layer, its children, the Linears run as one
/// product there), in order.
struct Family {
    architectures: &'static [&'static str],
    layer: &'static [(&'static str, &'static [Child], &'static [&'static str])],
    /// The experts' count in config.json (a mixture of experts).
    experts: &'static str,
}

use Child::{Experts, Linear, Plain};

const ATTENTION: &[Child] = &[Linear("q_proj"), Linear("k_proj"), Linear("v_proj"), Linear("o_proj")];
const ATTENTION_QK_NORMS: &[Child] =
    &[Linear("q_proj"), Linear("k_proj"), Linear("v_proj"), Linear("o_proj"), Plain("q_norm.weight", ""), Plain("k_norm.weight", "")];
const QKV: &[&str] = &["q_proj", "k_proj", "v_proj"];
const MLP: &[Child] = &[Linear("gate_proj"), Linear("up_proj"), Linear("down_proj")];
const GATE_UP: &[&str] = &["gate_proj", "up_proj"];
const NORMS: &[Child] = &[Plain("input_layernorm.weight", ""), Plain("post_attention_layernorm.weight", "")];

const FAMILIES: &[Family] = &[
    Family { architectures: &["Qwen3ForCausalLM"], layer: &[("self_attn", ATTENTION_QK_NORMS, QKV), ("mlp", MLP, GATE_UP), ("", NORMS, &[])], experts: "" },
    Family {
        architectures: &["Qwen2ForCausalLM", "LlamaForCausalLM", "MistralForCausalLM", "GraniteForCausalLM"],
        layer: &[("self_attn", ATTENTION, QKV), ("mlp", MLP, GATE_UP), ("", NORMS, &[])],
        experts: "",
    },
    Family {
        architectures: &["GraniteMoeForCausalLM"],
        layer: &[
            ("self_attn", ATTENTION, QKV),
            ("", NORMS, &[]),
            (
                "block_sparse_moe",
                &[
                    Plain("router.weight", "router.layer.weight"),
                    Experts("experts", &[("gate_up_proj", "input_linear.weight"), ("down_proj", "output_linear.weight")]),
                ],
                &[],
            ),
        ],
        experts: "num_local_experts",
    },
];

/// The layout packs are saved in.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Layout {
    /// The tiered layout (glyd-v4), the 12-bit one for a matrix that takes fewer bytes in it ([`pack::pack_smallest`]): the smallest.
    Tiered,
    /// The 12-bit layout (glyd-v3): what an A10, A100 or H100 runs, loaded there without packing again.
    Twelve,
    /// Candidate A, lossy (glyd-v6): every bf16 weight rounded to three fraction bits (pack::round_a, as the quality
    /// screens' candidate-a arm rounds every bf16 parameter; f32 tensors as they are), each Linear's matrix in the compact
    /// tiered layout (pack::pack_compact_checked; its entry "mma" with tiers[0]'s tag COMPACT_TAG).
    Compact,
    /// Six-bit block codes, lossy (glyd-v7, the level kestrel): each run of 32 values of a row as six-bit codes and a bf16 scale
    /// (pack::pack_int6_checked, tiers[0]'s tag INT6_TAG).
    Int6,
    /// Five-bit block codes, lossy (glyd-v7, the level swift): each run of 32 values of a row as five-bit codes and a bf16 scale
    /// (pack::pack_int5_checked, tiers[0]'s tag INT5_TAG).
    Int5,
    /// Four-bit block codes, lossy (glyd-v7, the level wren): each run of 32 values of a row as four-bit codes and a bf16 scale
    /// (pack::pack_int4_checked, tiers[0]'s tag INT4_TAG).
    Int4,
    /// Exact four-bit block codes, lossless where it is made (glyd-v7, "i4x"): a checkpoint trained to four bits (quantization-aware) held bit for bit,
    /// each run of 32 values of a row as four-bit codes and a scale of its own 16-bit form (pack::pack_int4h_checked, tiers[0]'s tag INT4H_TAG).
    Int4H,
    /// Three-bit block codes, lossy (glyd-v7, no level: a model alone is too coarse in them; a mixture of experts' experts may be): each run of 32
    /// values of a row as three-bit codes and a bf16 scale (pack::pack_int3_checked, tiers[0]'s tag INT3_TAG).
    Int3,
    /// The six-, five- and four-bit row layouts, lossy (glyd-v7, entries "i6r", "i5r", "i4r"; a pack's own layout in a save in a block layout, never a save's): the
    /// codes and scales the block layout of the same bits makes of a matrix, placed a row at a time (pack::pack_rows, tiers[0]'s tag INT6_ROW_TAG, INT5_ROW_TAG or
    /// INT4_ROW_TAG). The pack holds the transpose of its tensor (a linear's weight [out, in] is the pack [in, out], each row one input's weights to all the outputs, in runs of 32
    /// outputs): an experts' down matrix, whose input is mostly zeros, is read for the inputs that are not.
    Int6Row,
    Int5Row,
    Int4Row,
    /// The three- and two-bit codebook row layouts, lossy (glyd-v7, entries "c3r", "c2r"; a pack's own layout, never a save's): a matrix of neuron rows held as indices into a codebook of the row's own (codebook::CbRows,
    /// tiers[0]'s tag CB3_ROW_TAG or CB2_ROW_TAG), its three buffers data (the codes), blocks (the scales) and codebook; the codes come from a calibration export (hbexport), never from the weights.
    Cb3Row,
    Cb2Row,
    /// The Hummingbird level (glyd-v7, "hb", the level hummingbird; a save's layout, never a pack's): the routed experts of a mixture of experts in codebook rows (c3r or c2r by the bits of each expert, from a calibration
    /// export) and every other matrix in the six-bit block layout (each such linear says so in its plan), and a tensor `glyd_rot_signs` for each layer of experts, the signs of its rotation.
    Hummingbird,
}

/// A layout is shown by its name in glyd.json (`i6`, `hb`, ...), in messages and in `{:?}` alike: the program of a release holds no other word for one.
impl std::fmt::Debug for Layout {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

impl Layout {
    /// Its name in glyd.json (the compact layout's is the tiered layout's: its packs are told apart by the tag).
    pub fn name(self) -> &'static str {
        match self {
            Layout::Tiered | Layout::Compact => "mma",
            Layout::Twelve => "mma12",
            Layout::Int6 => "i6",
            Layout::Int5 => "i5",
            Layout::Int4 => "i4",
            Layout::Int4H => "i4x",
            Layout::Int3 => "i3",
            Layout::Int6Row => "i6r",
            Layout::Int5Row => "i5r",
            Layout::Int4Row => "i4r",
            Layout::Cb3Row => "c3r",
            Layout::Cb2Row => "c2r",
            Layout::Hummingbird => "hb",
        }
    }

    pub fn by_name(name: &str) -> Option<Layout> {
        [
            Layout::Tiered,
            Layout::Twelve,
            Layout::Int6,
            Layout::Int5,
            Layout::Int4,
            Layout::Int4H,
            Layout::Int3,
            Layout::Int6Row,
            Layout::Int5Row,
            Layout::Int4Row,
            Layout::Cb3Row,
            Layout::Cb2Row,
            Layout::Hummingbird,
        ]
        .into_iter()
        .find(|l| l.name() == name)
    }

    /// The bits of a code in a codebook row layout; None for the others.
    pub fn cb_bits(self) -> Option<u32> {
        match self {
            Layout::Cb3Row => Some(3),
            Layout::Cb2Row => Some(2),
            _ => None,
        }
    }

    /// The bits of a code in a row layout; None for the others.
    pub fn row_bits(self) -> Option<u32> {
        match self {
            Layout::Int6Row => Some(6),
            Layout::Int5Row => Some(5),
            Layout::Int4Row => Some(4),
            _ => None,
        }
    }

    /// The row layout of a block layout's bits (six, five and four); None for the others.
    pub fn row_of(self) -> Option<Layout> {
        match self {
            Layout::Int6 => Some(Layout::Int6Row),
            Layout::Int5 => Some(Layout::Int5Row),
            Layout::Int4 => Some(Layout::Int4Row),
            _ => None,
        }
    }

    /// A pack's words' name in glyd.json: the tiered layout's three tiers, the 12-bit layout's base hb (an int).
    fn words(self) -> &'static str {
        match self {
            Layout::Tiered
            | Layout::Compact
            | Layout::Int6
            | Layout::Int5
            | Layout::Int4
            | Layout::Int4H
            | Layout::Int3
            | Layout::Int6Row
            | Layout::Int5Row
            | Layout::Int4Row
            | Layout::Cb3Row
            | Layout::Cb2Row
            | Layout::Hummingbird => "tiers",
            Layout::Twelve => "hb",
        }
    }

    /// A pack's buffers' names (the key's suffixes), dtypes, and bytes a step of its data.
    fn buffers(self) -> ([(&'static str, &'static str); 3], usize) {
        match self {
            Layout::Tiered => ([("data", "U8"), ("blocks", "U8"), ("block_base", "I32")], 1280),
            Layout::Compact => ([("data", "U8"), ("blocks", "U8"), ("block_base", "I32")], pack::COMPACT_STEP_BYTES),
            Layout::Int6 => ([("data", "U8"), ("blocks", "U8"), ("block_base", "I32")], pack::INT6_STEP_BYTES),
            Layout::Int5 => ([("data", "U8"), ("blocks", "U8"), ("block_base", "I32")], pack::INT5_STEP_BYTES),
            Layout::Int4 | Layout::Int4H => ([("data", "U8"), ("blocks", "U8"), ("block_base", "I32")], pack::INT4_STEP_BYTES),
            Layout::Int3 => ([("data", "U8"), ("blocks", "U8"), ("block_base", "I32")], pack::INT3_STEP_BYTES),
            Layout::Int6Row | Layout::Int5Row | Layout::Int4Row => ([("data", "U8"), ("blocks", "U8"), ("block_base", "I32")], 0), // (a row layout's data is rows of `rows_stride` bytes, not steps)
            Layout::Cb3Row | Layout::Cb2Row => ([("data", "U8"), ("blocks", "U8"), ("codebook", "U8")], 0), // (the codes, the scales, the codebooks: rows of `cb_strides` bytes)
            Layout::Hummingbird => ([("data", "U8"), ("blocks", "U8"), ("block_base", "I32")], pack::INT6_STEP_BYTES), // (a save's layout, no pack's: a damaged entry naming it is refused where it is decoded)
            Layout::Twelve => ([("data", "U8"), ("exc", "I32"), ("exc_base", "I32")], 1536),
        }
    }
}

/// A save's plan: what it puts, in order.
struct Plan {
    puts: Vec<Put>,
}

fn dims(t: &TensorInfo) -> (usize, usize) {
    (t.shape.first().copied().unwrap_or(0) as usize, t.shape.get(1).copied().unwrap_or(0) as usize)
}

/// A Linear's weight packed: bf16, 2-D, rows a multiple of 64 and columns of 16.
fn packable(t: &TensorInfo) -> bool {
    let (o, k) = dims(t);
    t.dtype == "BF16" && t.shape.len() == 2 && o > 0 && o % 64 == 0 && k % 16 == 0
}

/// A Linear's weight [out, in] packed in a row layout, as the transpose [in, out]: bf16, 2-D, the outputs in runs of 32, the weights whole thousands.
fn packable_rows(t: &TensorInfo) -> bool {
    let (o, k) = dims(t);
    t.dtype == "BF16" && t.shape.len() == 2 && o > 0 && k > 0 && o % 32 == 0 && (o * k) % pack::STEP == 0
}

impl Plan {
    /// What save_pretrained writes for this checkpoint, in its order: first as the model's modules come (the
    /// embedding as a GEmbedding; each layer's packs, a merged group's after its block's other Linears; experts';
    /// the output layer packed; lossy: the embedding too, a pack as a Linear's matrix is), then what its state_dict holds that is not packed. A tensor the family has that the
    /// checkpoint lacks is refused (transformers would make it up).
    fn new(src: &Checkpoint, config: &Value, merge: bool, lossy: bool) -> io::Result<Plan> {
        let arch = config.get("architectures").and_then(Value::as_array).and_then(|a| a.first()).and_then(Value::as_str).unwrap_or("");
        let family = FAMILIES.iter().find(|f| f.architectures.contains(&arch)).ok_or_else(|| {
            let known: Vec<&str> = FAMILIES.iter().flat_map(|f| f.architectures.iter().copied()).collect();
            bad(format!("{arch}: not a family this packer knows ({}); python -m glyd.gpu pack packs any", known.join(", ")))
        })?;
        let layers = config.get("num_hidden_layers").and_then(Value::as_u64).ok_or_else(|| bad("config.json: no num_hidden_layers"))? as usize;
        let get = |name: &str| src.get(name).map(|(_, t)| t.clone());
        let need = |name: &str| get(name).ok_or_else(|| bad(format!("{}: no {name}, which a {arch} has", src.dir.display())));
        for f in &src.files {
            if let Some((n, t)) = f.tensors.iter().find(|(_, t)| t.dtype != "BF16") {
                return Err(bad(format!("{n}: {}, where glyd packs a bf16 checkpoint", t.dtype)));
            }
        }
        let embed = need("model.embed_tokens.weight")?;
        // an output layer tied to the embedding: where the config says so, else where the checkpoint has none
        let tied = config.get("tie_word_embeddings").and_then(Value::as_bool).unwrap_or(get("lm_head.weight").is_none());
        let head = if tied { embed.clone() } else { need("lm_head.weight")? };
        let gembedding = embed.shape.len() == 2 && embed.shape[1] % 128 == 0;
        let (mut puts, mut rest) = (Vec::new(), Vec::new());
        if gembedding && lossy && embed.shape[0] % 64 == 0 {
            // (a block layout's: the embedding a pack as the engine packs it, the output layer tied to it reading it)
            puts.push(Put::Linear { members: vec!["model.embed_tokens".into()], from: vec!["model.embed_tokens".into()], layout: None });
        } else if gembedding {
            puts.push(Put::Copy { name: "model.embed_tokens.weight".into(), from: "model.embed_tokens.weight".into(), cast: false });
        } else {
            rest.push(Put::Copy { name: "model.embed_tokens.weight".into(), from: "model.embed_tokens.weight".into(), cast: false });
        }
        let copy = |name: String| Put::Copy { from: name.clone(), name, cast: false };
        for i in 0..layers {
            for &(block, children, group) in family.layer {
                let path = |c: &str| if block.is_empty() { format!("model.layers.{i}.{c}") } else { format!("model.layers.{i}.{block}.{c}") };
                for &child in children {
                    if let Linear(c) = child {
                        need(&format!("{}.weight", path(c)))?;
                    }
                }
                // the group runs as one product where every member is packed and all or none have a bias
                let has_bias = |c: &str| get(&format!("{}.bias", path(c))).is_some();
                let merged = merge
                    && !group.is_empty()
                    && group.iter().all(|c| get(&format!("{}.weight", path(c))).is_some_and(|t| packable(&t)))
                    && group.iter().all(|c| has_bias(c) == has_bias(group[0]));
                for &child in children {
                    match child {
                        Linear(c) => {
                            let t = need(&format!("{}.weight", path(c)))?;
                            if merged && group.contains(&c) {
                                continue;
                            }
                            if packable(&t) {
                                puts.push(Put::Linear { members: vec![path(c)], from: vec![path(c)], layout: None });
                            } else {
                                rest.push(copy(format!("{}.weight", path(c))));
                                if has_bias(c) {
                                    rest.push(copy(format!("{}.bias", path(c))));
                                }
                            }
                        }
                        Plain(name, from) => {
                            let from = if from.is_empty() { path(name) } else { path(from) };
                            need(&from)?;
                            rest.push(Put::Copy { name: path(name), from, cast: false });
                        }
                        Experts(module, weights) => {
                            let e =
                                config.get(family.experts).and_then(Value::as_u64).ok_or_else(|| bad(format!("config.json: no {}", family.experts)))? as usize;
                            let ts: Vec<(&str, String, TensorInfo)> =
                                weights.iter().map(|&(w, from)| need(&path(from)).map(|t| (w, path(from), t))).collect::<io::Result<_>>()?;
                            // packed where every matrix [out, in] has rows a multiple of 64 and columns of 16
                            let all = ts.iter().all(|(_, _, t)| t.shape.len() == 3 && t.shape[0] as usize == e && t.shape[1] % 64 == 0 && t.shape[2] % 16 == 0);
                            for (w, from, _) in ts {
                                if all {
                                    puts.push(Put::Experts { module: path(module), weight: w.to_string(), from, experts: e });
                                } else {
                                    rest.push(Put::Copy { name: format!("{}.{w}", path(module)), from, cast: false });
                                }
                            }
                        }
                    }
                }
                if merged {
                    let members: Vec<String> = group.iter().map(|c| path(c)).collect();
                    puts.push(Put::Linear { from: members.clone(), members, layout: None });
                }
            }
        }
        need("model.norm.weight")?;
        rest.push(copy("model.norm.weight".into()));
        if packable(&head) && !tied {
            puts.push(Put::Linear { members: vec!["lm_head".into()], from: vec!["lm_head".into()], layout: None });
        } else if !packable(&head) {
            rest.push(Put::Copy {
                name: "lm_head.weight".into(),
                from: if tied { "model.embed_tokens.weight".into() } else { "lm_head.weight".into() },
                cast: false,
            });
        }
        puts.extend(rest);
        Ok(Plan { puts })
    }
}

impl Plan {
    /// A plan as JSON: {"puts": [{"linear": [module path, ...], "from": [its path in the source, ...], "layout": "i6" | "i5" |
    /// "i4" | "i3" | "i6r" | "i5r" | "i4r"} | {"copy": name, "from": its name in the source}, ...]} in save_pretrained's order ("from" defaults to the
    /// names; a linear's "layout", in a save in a block layout, puts that pack in another layout than the save's: another block layout, or a row
    /// layout, which packs the one weight's transpose); each tensor in the checkpoint, a linear's weights bf16 or FP8 with their block scales
    /// ([`Checkpoint::matrix`]), packable and of one width, a copy's BF16 or F32 (saved as it is, F32 too, unless it is cast).
    fn from_json(text: &str, src: &Checkpoint) -> io::Result<Plan> {
        let v = json::parse(text).map_err(|e| bad(format!("plan: {e}")))?;
        let items = v.get("puts").and_then(Value::as_array).ok_or_else(|| bad("plan: no \"puts\""))?;
        let found = |n: &str| src.get(n).map(|(_, t)| t.clone()).ok_or_else(|| bad(format!("plan: {n} is not in {}", src.dir.display())));
        let tensor = |n: &str| -> io::Result<TensorInfo> {
            let t = found(n)?;
            if t.dtype != "BF16" {
                return Err(bad(format!("{n}: {}, where glyd packs a bf16 checkpoint", t.dtype)));
            }
            Ok(t)
        };
        let weight = |n: &str| -> io::Result<TensorInfo> {
            found(n)?;
            src.matrix(n)
        };
        let mut puts = Vec::with_capacity(items.len());
        for it in items {
            if let Some(ms) = it.get("linear").and_then(Value::as_array) {
                let names = |v: &[Value]| v.iter().map(|m| m.as_str().map(str::to_string)).collect::<Option<Vec<_>>>().filter(|m| !m.is_empty());
                let members = names(ms).ok_or_else(|| bad("plan: a linear's members are not names"))?;
                let from = match it.get("from").and_then(Value::as_array) {
                    Some(f) => names(f).filter(|f| f.len() == members.len()).ok_or_else(|| bad("plan: a linear's \"from\" is not a name for each member"))?,
                    None => members.clone(),
                };
                let ts = from.iter().map(|m| weight(&format!("{m}.weight"))).collect::<io::Result<Vec<_>>>()?;
                let layout = match it.get("layout") {
                    None => None,
                    Some(l) => Some(
                        l.as_str()
                            .and_then(Layout::by_name)
                            .filter(|l| {
                                matches!(
                                    l,
                                    Layout::Int6
                                        | Layout::Int5
                                        | Layout::Int4
                                        | Layout::Int4H
                                        | Layout::Int3
                                        | Layout::Int6Row
                                        | Layout::Int5Row
                                        | Layout::Int4Row
                                )
                            })
                            .ok_or_else(|| bad(format!("plan: {}: a layout other than i6, i5, i4, i3, i6r, i5r or i4r", members.join(" + "))))?,
                    ),
                };
                // (a row layout packs one weight's transpose: its outputs runs of 32, its weights whole thousands)
                let rows_pack = layout.is_some_and(|l| l.row_bits().is_some());
                if rows_pack && (ts.len() != 1 || !packable_rows(&ts[0])) {
                    return Err(bad(format!(
                        "plan: {} is not packable in a row layout (one weight, its outputs runs of 32 and its size whole thousands)",
                        members.join(" + ")
                    )));
                }
                if !rows_pack && (!ts.iter().all(packable) || ts.iter().any(|t| dims(t).1 != dims(&ts[0]).1)) {
                    return Err(bad(format!("plan: {} is not packable as one matrix", members.join(" + "))));
                }
                for m in &from {
                    if src.get(&format!("{m}.bias")).is_some() {
                        tensor(&format!("{m}.bias"))?;
                    }
                }
                puts.push(Put::Linear { members, from, layout });
            } else if let Some(module) = it.get("codebook").and_then(Value::as_str) {
                let n = |k: &str| it.get(k).and_then(Value::as_u64).map(|x| x as usize).ok_or_else(|| bad(format!("plan: {module}: no {k}")));
                let down = match it.get("matrix").and_then(Value::as_str) {
                    Some("up") => false,
                    Some("downT") => true,
                    other => return Err(bad(format!("plan: {module}: a matrix {other:?}, where `up` and `downT` are the export's"))),
                };
                tensor(&format!("{module}.weight"))?;
                puts.push(Put::Codebook { module: module.into(), layer: n("layer")?, expert: n("expert")?, down });
            } else if let Some(name) = it.get("signs").and_then(Value::as_str) {
                let layer = it.get("layer").and_then(Value::as_u64).ok_or_else(|| bad(format!("plan: {name}: no layer")))? as usize;
                puts.push(Put::Signs { name: name.into(), layer });
            } else if let Some(name) = it.get("copy").and_then(Value::as_str) {
                let from = it.get("from").and_then(Value::as_str).unwrap_or(name);
                let cast = match it.get("cast").and_then(Value::as_str) {
                    None => false,
                    Some("BF16") => true,
                    Some(c) => return Err(bad(format!("plan: {name}: a cast to {c}, where only BF16 is made"))),
                };
                if cast {
                    let t = found(from)?;
                    if t.dtype != "F32" {
                        return Err(bad(format!("plan: {from}: {}, where a cast to BF16 takes F32", t.dtype)));
                    }
                } else if found(from)?.dtype != "F32" {
                    tensor(from)?;
                }
                puts.push(Put::Copy { name: name.into(), from: from.into(), cast });
            } else {
                return Err(bad("plan: a put is neither a linear, a codebook matrix, signs nor a copy"));
            }
        }
        Ok(Plan { puts })
    }
}

/// A tensor to save: its name, dtype, shape, bytes; the sha256 of a tensor saved as it is (not a pack's buffer).
pub(crate) struct Saved {
    name: String,
    dtype: &'static str,
    shape: Vec<u64>,
    bytes: Vec<u8>,
    sha256: Option<String>,
}

impl Saved {
    /// A tensor saved as it is (BF16 or F32), its sha256 with it.
    fn plain(name: String, dtype: &'static str, shape: Vec<u64>, bytes: Vec<u8>) -> Saved {
        let sha256 = Some(hex(&Sha256::digest(&bytes)));
        Saved { name, dtype, shape, bytes, sha256 }
    }
}

/// A put done: its tensors in order, its pack's glyd.json entry, the tensors it checked, those of them packed from calibrated codes.
struct Done {
    tensors: Vec<Saved>,
    entry: Option<(String, Value)>,
    checked: usize,
    coded: usize,
}

/// The calibrated codes (`NAME.codes`, I8, rows x cols) and bf16 scales (`NAME.scales`, BF16, rows x cols / 32) of the matrices `from` in `codes`, stacked by rows; None where none of them has any. An error where
/// one has only some, or has a dtype or shape other than its matrix's.
fn read_codes(codes: &Checkpoint, from: &[String], rows: &[usize], cols: usize) -> io::Result<Option<(Vec<i8>, Vec<u16>)>> {
    if !from.iter().any(|m| codes.get(&format!("{m}.codes")).is_some() || codes.get(&format!("{m}.scales")).is_some()) {
        return Ok(None);
    }
    let (mut c, mut s) = (Vec::new(), Vec::new());
    for (m, &r) in from.iter().zip(rows) {
        let get = |suffix: &str, dtype: &str, shape: [u64; 2]| -> io::Result<Vec<u8>> {
            let name = format!("{m}.{suffix}");
            let (f, t) = codes.get(&name).ok_or_else(|| bad(format!("no {name} in {}", codes.dir.display())))?;
            if t.dtype != dtype || t.shape != shape {
                return Err(bad(format!("{name}: {} {:?}, where {dtype} {shape:?} is wanted", t.dtype, t.shape)));
            }
            f.read(t)
        };
        c.extend(get("codes", "I8", [r as u64, cols as u64])?.into_iter().map(|b| b as i8));
        s.extend(get("scales", "BF16", [r as u64, cols as u64 / 32])?.chunks_exact(2).map(|b| u16::from_le_bytes([b[0], b[1]])));
    }
    Ok(Some((c, s)))
}

fn le_bytes<T: Copy>(v: &[T], f: impl Fn(T) -> [u8; 4]) -> Vec<u8> {
    v.iter().flat_map(|&x| f(x)).collect()
}

/// f32 bytes (little-endian) as bf16's, rounded to nearest even as torch's .to(torch.bfloat16) rounds (a NaN: 0x7FC0).
fn bf16_from_f32(b: &[u8]) -> Vec<u8> {
    b.chunks_exact(4)
        .flat_map(|c| {
            let u = u32::from_le_bytes([c[0], c[1], c[2], c[3]]);
            let h = if f32::from_bits(u).is_nan() { 0x7FC0 } else { (u.wrapping_add(0x7FFF + ((u >> 16) & 1)) >> 16) as u16 };
            h.to_le_bytes()
        })
        .collect()
}

fn u16_bytes(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn hex(d: &[u8]) -> String {
    d.iter().map(|b| format!("{b:02x}")).collect()
}

/// The sha256 of bf16 weights' bytes, little-endian as safetensors holds them.
fn sha256_u16(v: &[u16]) -> String {
    if cfg!(target_endian = "little") {
        // SAFETY: u16s' bytes, the host's order safetensors'.
        return hex(&Sha256::digest(unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 2) }));
    }
    let mut h = Sha256::new();
    for c in v.chunks(1 << 20) {
        h.update(u16_bytes(c));
    }
    hex(&h.finalize())
}

fn shape_value(s: &[u64]) -> Value {
    Value::Array(s.iter().map(|&d| Value::from(d)).collect())
}

/// A pack's glyd.json entry (format.py's entry): its layout, its matrix's shape, its words (the tiers; the 12-bit
/// layout's base hb, an int: words[0]), its tensors (name, shape, sha256) in their order in its rows; an experts'
/// weight's E (held as [E, out, in]).
fn entry(layout: Layout, shape: [usize; 2], words: &[u32], tensors: Vec<(String, Vec<u64>, String)>, experts: Option<usize>) -> Value {
    let words = match layout {
        Layout::Tiered
        | Layout::Compact
        | Layout::Int6
        | Layout::Int5
        | Layout::Int4
        | Layout::Int4H
        | Layout::Int3
        | Layout::Int6Row
        | Layout::Int5Row
        | Layout::Int4Row
        | Layout::Cb3Row
        | Layout::Cb2Row
        | Layout::Hummingbird => Value::Array(words.iter().map(|&t| Value::from(t as u64)).collect()),
        Layout::Twelve => Value::from(words[0] as u64),
    };
    let mut e = vec![
        ("layout".to_string(), Value::from(layout.name())),
        ("shape".to_string(), shape_value(&[shape[0] as u64, shape[1] as u64])),
        (layout.words().to_string(), words),
        (
            "tensors".to_string(),
            Value::Array(
                tensors
                    .into_iter()
                    .map(|(n, s, h)| {
                        Value::Object(vec![("name".into(), Value::String(n)), ("shape".into(), shape_value(&s)), ("sha256".into(), Value::String(h))])
                    })
                    .collect(),
            ),
        ),
    ];
    if let Some(n) = experts {
        e.push(("experts".into(), Value::from(n as u64)));
        e.push(("transposed".into(), Value::Bool(false)));
    }
    Value::Object(e)
}

/// A matrix packed in `layout` (the tiered one: the smaller of its two layouts, [`pack::pack_smallest_checked`]), each
/// step decoded back and compared with its weights ("checked"), its three buffers as saved under `key` (`key`data
/// ...), its words (the tiers; the 12-bit layout's base hb) and the layout it is in. A lossy block layout leaves `w` the
/// weights it decodes to (what the GPU computes with, and what glyd.json hashes). `coded`: a block layout's codes and
/// scales (a calibration's) to pack as they are, instead of the fit of `w`'s.
fn packed(
    layout: Layout,
    w: &mut [u16],
    rows: usize,
    cols: usize,
    key: &str,
    what: &str,
    coded: Option<(&[i8], &[u16])>,
) -> io::Result<(Vec<Saved>, Vec<u32>, Layout)> {
    let differ = || bad(format!("{what}: decoded to other bits than its weights"));
    let steps = (rows * cols / pack::STEP) as u64;
    let buffer = |name: String, dtype, shape, bytes| Saved { name, dtype, shape: vec![shape], bytes, sha256: None };
    let tiered = |p: pack::Tiered| {
        let (blocks, data) = (p.blocks.len() as u64, p.data.len() as u64); // (1280 bytes a step; the compact layout's 768)
        let base = le_bytes(&p.block_base, i32::to_le_bytes);
        let lay = match p.tiers[0] >> 24 {
            pack::COMPACT_TAG => Layout::Compact,
            pack::INT6_TAG => Layout::Int6,
            pack::INT5_TAG => Layout::Int5,
            pack::INT4_TAG => Layout::Int4,
            pack::INT4H_TAG => Layout::Int4H,
            pack::INT3_TAG => Layout::Int3,
            pack::INT6_ROW_TAG => Layout::Int6Row,
            pack::INT5_ROW_TAG => Layout::Int5Row,
            pack::INT4_ROW_TAG => Layout::Int4Row,
            _ => Layout::Tiered,
        };
        (
            vec![
                buffer(format!("{key}data"), "U8", data, p.data),
                buffer(format!("{key}blocks"), "U8", blocks, p.blocks),
                buffer(format!("{key}block_base"), "I32", steps + 1, base),
            ],
            p.tiers.to_vec(),
            lay,
        )
    };
    let twelve = |p: pack::Twelve| {
        let (exc, base) = (le_bytes(&p.exc, i32::to_le_bytes), le_bytes(&p.exc_base, i32::to_le_bytes));
        let hb = p.hb() as u32;
        (
            vec![
                buffer(format!("{key}data"), "U8", steps * 1536, p.data),
                buffer(format!("{key}exc"), "I32", p.exc.len() as u64, exc),
                buffer(format!("{key}exc_base"), "I32", steps + 1, base),
            ],
            vec![hb],
            Layout::Twelve,
        )
    };
    Ok(match layout {
        Layout::Tiered => match pack::pack_smallest_checked(w, rows, cols).ok_or_else(differ)? {
            pack::Packed::Tiered(p) => tiered(p),
            pack::Packed::Twelve(p) => twelve(p),
        },
        Layout::Twelve => twelve(pack::pack_twelve_checked(w, rows, cols).ok_or_else(differ)?),
        Layout::Compact => tiered(pack::pack_compact_checked(w, rows, cols).ok_or_else(differ)?), // (w rounded by its caller)
        Layout::Int6 => {
            let p = match coded {
                Some((c, s)) => pack::pack_int6_runs_checked(c, s, rows, cols),
                None => pack::pack_int6_checked(w, rows, cols),
            };
            let p = p.ok_or_else(|| bad(format!("{what}: not a six-bit layout's matrix, or decoded to other bits than its codes")))?;
            pack::unpack_int6(&p, w);
            tiered(p)
        }
        Layout::Int5 => {
            let p = match coded {
                Some((c, s)) => pack::pack_int5_runs_checked(c, s, rows, cols),
                None => pack::pack_int5_checked(w, rows, cols),
            };
            let p = p.ok_or_else(|| bad(format!("{what}: not a five-bit layout's matrix, or decoded to other bits than its codes")))?;
            pack::unpack_int5(&p, w);
            tiered(p)
        }
        Layout::Int4 => {
            let p = pack::pack_int4_checked(w, rows, cols)
                .ok_or_else(|| bad(format!("{what}: not a four-bit layout's matrix, or decoded to other bits than its codes")))?;
            pack::unpack_int4(&p, w);
            tiered(p)
        }
        Layout::Int4H => tiered(
            pack::pack_int4h_checked(w, rows, cols)
                .ok_or_else(|| bad(format!("{what}: not a checkpoint trained to four bits (no codes times a scale give its weights bit for bit)")))?,
        ),
        Layout::Int3 => {
            let p = pack::pack_int3_checked(w, rows, cols)
                .ok_or_else(|| bad(format!("{what}: not a three-bit layout's matrix, or decoded to other bits than its codes")))?;
            pack::unpack_int3(&p, w);
            tiered(p)
        }
        // (`w` is the matrix to pack, already the transpose of the tensor's: `Put::run`)
        Layout::Int6Row | Layout::Int5Row | Layout::Int4Row => {
            let bits = layout.row_bits().expect("a row layout");
            let p = pack::pack_rows_checked(bits, w, rows, cols).ok_or_else(|| {
                bad(format!(
                    "{what}: not a {bits}-bit row layout's matrix (columns of 32 and whole thousands of weights), or decoded to other bits than its codes"
                ))
            })?;
            pack::unpack_rows(&p, w);
            tiered(p)
        }
        Layout::Cb3Row | Layout::Cb2Row => {
            return Err(bad(format!("{what}: a codebook row layout holds codes made apart (an export), not a fit of the weights")))
        }
        Layout::Hummingbird => return Err(bad(format!("{what}: a Hummingbird save packs each of its linears in a layout of their own (the plan says which)"))),
    })
}

impl Put {
    /// Its bytes of bf16 read (the work it is; an FP8 matrix's the bytes of the bf16 it is read as).
    fn weight(&self, src: &Checkpoint) -> u64 {
        let b = |n: &str| src.get(n).map_or(0, |(_, t)| t.bytes);
        match self {
            Put::Copy { from, .. } => b(from),
            Put::Linear { from, .. } => from.iter().map(|m| src.matrix(&format!("{m}.weight")).map_or(0, |t| t.bytes)).sum(),
            Put::Experts { from, .. } => b(from),
            Put::Codebook { module, .. } => b(&format!("{module}.weight")),
            Put::Signs { .. } => 0,
        }
    }

    fn run(&self, src: &Checkpoint, layout: Layout, codes: Option<&Checkpoint>, hb: Option<&HbExport>) -> io::Result<Done> {
        match self {
            Put::Copy { name, from, cast } => {
                let (f, t) = src.get(from).ok_or_else(|| bad(format!("no {from}")))?;
                let mut bytes = if *cast { bf16_from_f32(&f.read(t)?) } else { f.read(t)? };
                if layout == Layout::Compact && !*cast && t.dtype == "BF16" {
                    for b in bytes.chunks_exact_mut(2) {
                        // (Candidate A: every bf16 parameter rounded; an f32 one, cast here, as it is)
                        let r = pack::round_a(u16::from_le_bytes([b[0], b[1]]));
                        b.copy_from_slice(&r.to_le_bytes());
                    }
                }
                let dtype = if !*cast && t.dtype == "F32" { "F32" } else { "BF16" };
                Ok(Done { tensors: vec![Saved::plain(name.clone(), dtype, t.shape.clone(), bytes)], entry: None, checked: 0, coded: 0 })
            }
            Put::Linear { members, from, layout: own } => {
                let layout = own.unwrap_or(layout);
                let ts: Vec<TensorInfo> = from.iter().map(|m| src.matrix(&format!("{m}.weight"))).collect::<io::Result<_>>()?;
                let cols = dims(&ts[0]).1;
                let rows: Vec<usize> = ts.iter().map(|t| dims(t).0).collect();
                let mut w = vec![0u16; rows.iter().sum::<usize>() * cols];
                let coded = codes.map(|c| read_codes(c, from, &rows, cols)).transpose()?.flatten(); // (calibrated: the codes and scales are packed as they are, the weights not read)
                let mut at = 0;
                for (m, &r) in from.iter().zip(&rows) {
                    if coded.is_none() {
                        src.read_bf16(&format!("{m}.weight"), &mut w[at..at + r * cols])?;
                    }
                    at += r * cols;
                }
                if layout == Layout::Compact {
                    w.iter_mut().for_each(|x| *x = pack::round_a(*x)); // (the values packed and hashed are Candidate A's)
                }
                let all = w.len() / cols;
                // a row layout packs the transpose of the weight [out, in]: [in, out], each row one input's weights; the weight it decodes to goes back to [out, in] (what glyd.json hashes)
                let rows_of = layout.row_bits().is_some();
                if rows_of && (members.len() != 1 || coded.is_some()) {
                    return Err(bad(format!("{}: a row layout holds one matrix, not codes made apart", members.join(" + "))));
                }
                let (mut tensors, words, lay) = if rows_of {
                    let mut t = pack::transpose(&w, all, cols);
                    let packed_rows = packed(layout, &mut t, cols, all, &format!("{}.glyd_", members[0]), &members.join(" + "), None)?;
                    w = pack::transpose(&t, cols, all);
                    packed_rows
                } else {
                    packed(layout, &mut w, all, cols, &format!("{}.glyd_", members[0]), &members.join(" + "), coded.as_ref().map(|(c, s)| (&c[..], &s[..])))?
                };
                let mut hashed = Vec::new();
                let mut at = 0;
                for (m, &r) in members.iter().zip(&rows) {
                    hashed.push((format!("{m}.weight"), vec![r as u64, cols as u64], sha256_u16(&w[at..at + r * cols])));
                    at += r * cols;
                }
                for (m, s) in members.iter().zip(from) {
                    if let Some((f, t)) = src.get(&format!("{s}.bias")) {
                        let mut bytes = f.read(t)?;
                        if layout == Layout::Compact {
                            for b in bytes.chunks_exact_mut(2) {
                                let r = pack::round_a(u16::from_le_bytes([b[0], b[1]]));
                                b.copy_from_slice(&r.to_le_bytes());
                            }
                        }
                        tensors.push(Saved::plain(format!("{m}.bias"), "BF16", t.shape.clone(), bytes));
                    }
                }
                let n = hashed.len();
                let shape = if rows_of { [cols, all] } else { [all, cols] };
                Ok(Done {
                    tensors,
                    entry: Some((members[0].clone(), entry(lay, shape, &words, hashed, None))),
                    checked: n,
                    coded: if coded.is_some() { n } else { 0 },
                })
            }
            Put::Codebook { module, layer, expert, down } => {
                let export = hb.ok_or_else(|| bad(format!("{module}: a matrix of an export of codes, and none is given")))?;
                let tensor = format!("{module}.weight");
                let (_, t) = src.get(&tensor).ok_or_else(|| bad(format!("no {tensor}")))?;
                let p = export.matrix(*layer, *expert, if *down { Matrix::DownT } else { Matrix::Up })?;
                // (the tensor is [neurons, row] for an up matrix, [row, neurons] for a down one: the pack holds its neurons' rows either way)
                let want = if *down { vec![p.cols as u64, p.rows as u64] } else { vec![p.rows as u64, p.cols as u64] };
                if t.shape != want {
                    return Err(bad(format!(
                        "{tensor}: {:?} in the source, where the export's {} neuron rows of {} values are {want:?}",
                        t.shape, p.rows, p.cols
                    )));
                }
                let mut w = vec![0u16; p.rows * p.cols];
                if !codebook::unpack_codebook_rows(&p, &mut w) {
                    return Err(bad(format!("{module}: the export's arrays do not make a matrix")));
                }
                if *down {
                    w = pack::transpose(&w, p.rows, p.cols);
                }
                let lay = if p.bits == 3 { Layout::Cb3Row } else { Layout::Cb2Row };
                let key = format!("{module}.glyd_");
                let buffer =
                    |suffix: &str, bytes: Vec<u8>| Saved { name: format!("{key}{suffix}"), dtype: "U8", shape: vec![bytes.len() as u64], bytes, sha256: None };
                let mut e = entry(lay, [p.rows, p.cols], &p.tiers(), vec![(tensor, t.shape.clone(), sha256_u16(&w))], None);
                if let Value::Object(o) = &mut e {
                    o.push(("transposed".into(), Value::Bool(*down)));
                }
                Ok(Done {
                    tensors: vec![buffer("data", p.codes), buffer("blocks", p.scales), buffer("codebook", p.codebook)],
                    entry: Some((module.clone(), e)),
                    checked: 1,
                    coded: 1,
                })
            }
            Put::Signs { name, layer } => {
                let export = hb.ok_or_else(|| bad(format!("{name}: the signs of an export of codes, and none is given")))?;
                let l = export.layers.get(*layer).ok_or_else(|| bad(format!("{name}: the export has no MoE layer {layer}")))?;
                Ok(Done {
                    tensors: vec![Saved::plain(name.clone(), "I8", vec![l.signs.len() as u64], l.signs.iter().map(|&s| s as u8).collect())],
                    entry: None,
                    checked: 0,
                    coded: 0,
                })
            }
            Put::Experts { module, weight, from, experts } => {
                let (_, t) = src.get(from).ok_or_else(|| bad(format!("no {from}")))?;
                let (e, a, b) = (*experts, t.shape[1] as usize, t.shape[2] as usize);
                let mut w = vec![0u16; e * a * b];
                src.read_bf16(from, &mut w)?;
                let (tensors, words, lay) = packed(layout, &mut w, e * a, b, &format!("{module}.glyd_{weight}_"), &format!("{module}.{weight}"), None)?;
                let sha = sha256_u16(&w);
                let name = format!("{module}.{weight}");
                Ok(Done {
                    tensors,
                    entry: Some((name.clone(), entry(lay, [e * a, b], &words, vec![(name, t.shape.clone(), sha)], Some(e)))),
                    checked: 1,
                    coded: 0,
                })
            }
        }
    }
}

/// items run on `threads` threads, their results handed to `done` in their order. A thread takes the item `done`
/// waits for, else a `big` one (a large matrix's pack runs beside the others, not after them), else the next in order,
/// but for the one `done` waits for only while the items taken and not yet handed to `done` weigh at most `budget`
/// with it (their weights: bytes held, or 1 an item). The results wait for their turn in a map.
fn in_order<T: Sync, R: Send>(
    items: &[T],
    weights: &[u64],
    big: &[bool],
    threads: usize,
    budget: u64,
    run: impl Fn(&T) -> io::Result<R> + Sync,
    mut done: impl FnMut(R) -> io::Result<()>,
) -> io::Result<()> {
    struct State<R> {
        taken: Vec<bool>,
        next: usize,
        waits: usize,
        held: u64,
        results: BTreeMap<usize, io::Result<R>>,
        stop: bool,
    }
    let n = items.len();
    assert!(weights.len() == n && big.len() == n, "a weight and a flag an item");
    let state = Mutex::new(State { taken: vec![false; n], next: 0, waits: 0, held: 0, results: BTreeMap::new(), stop: false });
    let cv = Condvar::new();
    std::thread::scope(|s| {
        for _ in 0..threads.max(1) {
            s.spawn(|| loop {
                let i = {
                    let mut st = state.lock().unwrap();
                    loop {
                        if st.stop {
                            return;
                        }
                        while st.next < n && st.taken[st.next] {
                            st.next += 1;
                        }
                        if st.next >= n {
                            return;
                        }
                        let fits = |i: usize, st: &State<R>| st.held + weights[i] <= budget;
                        let pick = if !st.taken[st.waits] {
                            Some(st.waits)
                        } else {
                            (0..n).find(|&i| big[i] && !st.taken[i] && fits(i, &st)).or_else(|| fits(st.next, &st).then_some(st.next))
                        };
                        if let Some(i) = pick {
                            st.taken[i] = true;
                            st.held += weights[i];
                            break i;
                        }
                        st = cv.wait(st).unwrap();
                    }
                };
                let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run(&items[i]))).unwrap_or_else(|_| Err(bad("a packing thread panicked")));
                state.lock().unwrap().results.insert(i, r);
                cv.notify_all();
            });
        }
        let mut result = Ok(());
        for (i, &w) in weights.iter().enumerate() {
            let r = {
                let mut st = state.lock().unwrap();
                loop {
                    if let Some(r) = st.results.remove(&i) {
                        st.waits = (i + 1).min(n - 1);
                        st.held -= w;
                        break r;
                    }
                    st = cv.wait(st).unwrap();
                }
            };
            cv.notify_all();
            if let Err(e) = r.and_then(&mut done) {
                result = Err(e);
                break;
            }
        }
        state.lock().unwrap().stop = true;
        cv.notify_all();
        result
    })
}

/// A safetensors file of `tensors`, as the safetensors library writes one: the header's tensors by dtype (the
/// larger first, as its Dtype orders them) then name, `__metadata__` {"format": "pt"} first, padded with spaces to
/// a multiple of 8 bytes; the data in the header's order.
pub(crate) fn write_safetensors(path: &Path, tensors: &mut [Saved]) -> io::Result<()> {
    // safetensors' Dtype, in its order
    const DTYPES: &[&str] = &[
        "BOOL", "F4", "F6_E2M3", "F6_E3M2", "U8", "I8", "F8_E5M2", "F8_E4M3", "F8_E8M0", "I16", "U16", "F16", "BF16", "I32", "U32", "F32", "C64", "F64", "I64",
        "U64",
    ];
    let rank = |d: &str| DTYPES.iter().position(|x| *x == d).expect("a dtype safetensors has");
    tensors.sort_by(|a, b| rank(b.dtype).cmp(&rank(a.dtype)).then(a.name.cmp(&b.name)));
    let mut header = vec![("__metadata__".to_string(), Value::Object(vec![("format".into(), Value::from("pt"))]))];
    let mut pos = 0u64;
    for t in tensors.iter() {
        let n = t.bytes.len() as u64;
        header.push((
            t.name.clone(),
            Value::Object(vec![
                ("dtype".into(), Value::from(t.dtype)),
                ("shape".into(), shape_value(&t.shape)),
                ("data_offsets".into(), Value::Array(vec![Value::from(pos), Value::from(pos + n)])),
            ]),
        ));
        pos += n;
    }
    let mut h = json::to_compact(&Value::Object(header)).into_bytes();
    h.resize(h.len().next_multiple_of(8), b' ');
    let mut f = io::BufWriter::with_capacity(8 << 20, std::fs::File::create(path).map_err(at(path))?);
    f.write_all(&(h.len() as u64).to_le_bytes()).map_err(at(path))?;
    f.write_all(&h).map_err(at(path))?;
    for t in tensors.iter() {
        f.write_all(&t.bytes).map_err(at(path))?;
    }
    f.flush().map_err(at(path))
}

fn matches(pattern: &str, name: &str) -> bool {
    match pattern.split_once('*') {
        Some((a, b)) => name.len() >= a.len() + b.len() && name.starts_with(a) && name.ends_with(b),
        None => pattern == name,
    }
}

/// What a save did.
pub struct Saving {
    /// Tensors packed and checked (decoded back to their weights' bits).
    pub checked: usize,
    /// Tensors saved as they are, their sha256 in glyd.json.
    pub hashed: usize,
    /// Bytes of bf16 read (an FP8 matrix as the bf16 it is read as).
    pub bytes: u64,
    pub shards: usize,
    /// Tensors packed from calibrated codes ([`save_calibrated`]) among the checked.
    pub calibrated: usize,
}

/// The checkpoint at `source` packed on `threads` threads and saved in `out` as glyd-v1 (glyd-v2 with a mixture of
/// experts' packs; glyd-v3 in the 12-bit layout), as save_pretrained saves it: shards of about `shard_bytes`,
/// glyd.json (with the sha256 of every tensor, packed or saved as it is), the index (several shards), the source's
/// config, generation config and tokenizer files. merge: q, k, v and gate, up as one pack each (from_pretrained's
/// merge). A shard is written by a thread of its own as the next is made; the host holds two and at most a shard's
/// worth of weights in flight past the one the save waits for (the module's docs). A directory holding another
/// checkpoint is refused.
pub fn save(source: &Source, out: &Path, threads: usize, shard_bytes: u64, merge: bool, layout: Layout) -> io::Result<Saving> {
    save_as::<false>(source, out, threads, shard_bytes, layout, None, None, false, |src, config| {
        Plan::new(src, config, merge, matches!(layout, Layout::Int6 | Layout::Int5 | Layout::Int4 | Layout::Int4H | Layout::Int3))
    })
}

/// `save` with a plan made elsewhere (huggingface/plan.py: what save_pretrained writes for the model transformers
/// builds, for a family `Plan::new` does not know), as JSON text: [`Plan::from_json`].
pub fn save_planned(source: &Source, out: &Path, threads: usize, shard_bytes: u64, layout: Layout, plan: &str) -> io::Result<Saving> {
    save_as::<false>(source, out, threads, shard_bytes, layout, None, None, false, |src, _| Plan::from_json(plan, src))
}

/// `save_planned` in the lossless layout for Gemma 4's penguin: a glyd-v7 directory of level penguin, which its engine reads as saved.
/// Every other lossless save stays glyd-v4, as the readers of those expect.
pub fn save_penguin(source: &Source, out: &Path, threads: usize, shard_bytes: u64, plan: &str) -> io::Result<Saving> {
    save_as::<false>(source, out, threads, shard_bytes, Layout::Tiered, None, None, true, |src, _| Plan::from_json(plan, src))
}

/// `save_planned` in a block layout (Int6, Int5) with calibrated codes: `codes` a directory of safetensors in a checkpoint's form (model.safetensors, or shards named by its index) holding, for each source matrix
/// calibrated (research/lossy-2026/calib.py quantize), `NAME.codes` (I8, [rows, cols], -q to q) and `NAME.scales` (BF16, [rows, cols / 32]), NAME its path in the source (the plan's "from"), and calibration.json
/// (an object with a "method"), which glyd.json records as "calibration". A linear whose matrices have codes is packed from them, bit for bit; one that has none (the embedding, the output layer) is packed from its
/// weights as `save_planned` does. Codes the plan does not use are refused.
pub fn save_calibrated(source: &Source, out: &Path, threads: usize, shard_bytes: u64, layout: Layout, plan: &str, codes: &Path) -> io::Result<Saving> {
    save_as::<true>(source, out, threads, shard_bytes, layout, Some(codes), None, false, |src, _| Plan::from_json(plan, src))
}

/// `save_planned` at the Hummingbird level ([`Layout::Hummingbird`]) from a calibration export (`export`: a directory of hbexport's format): the plan's `{"codebook": MODULE, "layer": L, "expert": E, "matrix": "up" | "downT"}`
/// puts are the routed experts' matrices, packed as they are in c3r or c2r by the bits the export gives the expert (the tensor MODULE.weight of the source is held to the shape the matrix has), and each `{"signs": NAME, "layer": L}`
/// is the signs of a layer's rotation saved as the tensor NAME (I8); the linears the plan keeps carry their own block layout (`"layout": "i6"`) as at the other levels. glyd.json records the export's meta.json (without the
/// signs and the experts' bits) as its "calibration". The export is held to the plan before anything is packed.
pub fn save_hummingbird(source: &Source, out: &Path, threads: usize, shard_bytes: u64, plan: &str, export: &Path) -> io::Result<Saving> {
    let ex = HbExport::open(export)?;
    save_as::<true>(source, out, threads, shard_bytes, Layout::Hummingbird, None, Some(&ex), false, |src, _| Plan::from_json(plan, src))
}

/// `EXTRAS`: whether this save may have calibrated codes or an export (`save_calibrated`, `save_hummingbird`); `save_planned`'s is false, so the program of a
/// build that only packs plainly (a release engine) holds none of their code.
#[allow(clippy::too_many_arguments)]
fn save_as<const EXTRAS: bool>(
    source: &Source,
    out: &Path,
    threads: usize,
    shard_bytes: u64,
    layout: Layout,
    codes: Option<&Path>,
    hb: Option<&HbExport>,
    penguin: bool,
    plan: impl FnOnce(&Checkpoint, &Value) -> io::Result<Plan>,
) -> io::Result<Saving> {
    let cfg = source.dir.join("config.json");
    let config = json::parse(&std::fs::read_to_string(&cfg).map_err(at(&cfg))?).map_err(|e| bad(format!("{}: {e}", cfg.display())))?;
    let src = Checkpoint::open(&source.dir)?;
    if layout.row_bits().is_some() || layout.cb_bits().is_some() {
        return Err(bad(format!(
            "a save in the {} layout: a row layout is a pack's own, a linear's \"layout\" in the plan of a save in a block layout (i6, i5, i4, i3)",
            layout.name()
        )));
    }
    let plan = plan(&src, &config)?;
    if !matches!(layout, Layout::Int6 | Layout::Int5 | Layout::Int4 | Layout::Int4H | Layout::Int3 | Layout::Hummingbird)
        && plan.puts.iter().any(|p| matches!(p, Put::Linear { layout: Some(_), .. }))
    {
        return Err(bad("plan: a linear's own layout is for a save in a block layout (i6, i5, i4, i3)"));
    }
    // a calibration export belongs to the Hummingbird level alone, which is nothing without one; its matrices and signs are held to what the export has before the first pack
    let from_export = plan.puts.iter().any(|p| matches!(p, Put::Codebook { .. } | Put::Signs { .. }));
    if (layout == Layout::Hummingbird) != hb.is_some() || (layout == Layout::Hummingbird) != from_export {
        return Err(bad("a Hummingbird save is made from an export of codes (and from it alone), and its plan holds the export's matrices and signs"));
    }
    if let Some(ex) = hb {
        for p in &plan.puts {
            match p {
                Put::Codebook { module, layer, expert, .. } if *layer >= ex.layers.len() || *expert >= ex.experts => {
                    return Err(bad(format!(
                        "plan: {module}: layer {layer} expert {expert}, where the export has {} MoE layers of {} experts",
                        ex.layers.len(),
                        ex.experts
                    )));
                }
                Put::Signs { name, layer } if *layer >= ex.layers.len() => {
                    return Err(bad(format!("plan: {name}: layer {layer}, where the export has {} MoE layers", ex.layers.len())))
                }
                _ => {}
            }
        }
    }
    let codes = codes
        .map(|dir| -> io::Result<(Checkpoint, Value)> {
            if !matches!(layout, Layout::Int6 | Layout::Int5) {
                return Err(bad("calibrated codes are for the block layouts (i6, i5)"));
            }
            let file = dir.join("calibration.json");
            let cal = json::parse(&std::fs::read_to_string(&file).map_err(at(&file))?).map_err(|e| bad(format!("{}: {e}", file.display())))?;
            if cal.get("method").and_then(Value::as_str).is_none() {
                return Err(bad(format!("{}: no \"method\"", file.display())));
            }
            let c = Checkpoint::open(dir)?;
            if c.names().next().is_none() {
                return Err(bad(format!("{}: no codes", dir.display())));
            }
            let packed: HashSet<&str> = plan
                .puts
                .iter()
                .filter_map(|p| match p {
                    Put::Linear { from, .. } => Some(from),
                    _ => None,
                })
                .flatten()
                .map(String::as_str)
                .collect();
            if let Some(n) = c.names().find(|n| n.strip_suffix(".codes").or_else(|| n.strip_suffix(".scales")).is_none_or(|m| !packed.contains(m))) {
                return Err(bad(format!("{}: {n} is not a matrix the plan packs (codes of another model, or another plan?)", dir.display())));
            }
            Ok((c, cal))
        })
        .transpose()?;
    let listing = |pattern: &dyn Fn(&str) -> bool| -> io::Result<Vec<PathBuf>> {
        Ok(match std::fs::read_dir(out) {
            Ok(d) => d.filter_map(|e| e.ok()).map(|e| e.path()).filter(|p| p.file_name().and_then(|n| n.to_str()).is_some_and(pattern)).collect(),
            Err(_) => Vec::new(),
        })
    };
    let old = listing(&|n| (n.starts_with("model") && n.ends_with(".safetensors")) || n == "model.safetensors.index.json")?;
    if !old.is_empty() && !out.join("glyd.json").exists() {
        let held = old
            .iter()
            .filter(|p| p.extension().is_some_and(|e| e == "safetensors"))
            .any(|p| crate::safetensors::File::open(p).is_ok_and(|f| f.tensors.iter().any(|(n, _)| n.contains(".glyd_"))));
        if !held {
            return Err(bad(format!("{} holds another checkpoint; save into a directory of its own", out.display())));
        }
    }
    for p in listing(&|n| n.starts_with("glyd-part-") && n.ends_with(".safetensors"))? {
        std::fs::remove_file(&p).map_err(at(&p))?; // a save cut short's
    }
    std::fs::create_dir_all(out).map_err(at(out))?;

    let (mut part, mut size, mut shards) = (Vec::<Saved>::new(), 0u64, Vec::<(String, Vec<String>, u64)>::new());
    let (mut packs, mut hashes) = (Vec::new(), Vec::new());
    let (mut checked, mut calibrated) = (0, 0);
    let weights: Vec<u64> = plan.puts.iter().map(|p| p.weight(&src)).collect();
    let bytes: u64 = weights.iter().sum();
    let big: Vec<bool> = weights.iter().map(|&w| w >= bytes / 16).collect(); // (a model's output layer, its embedding)
                                                                             // a shard written by a thread of its own as the next is made (the host holds two at most)
    let (to_writer, shards_in) = std::sync::mpsc::sync_channel::<(PathBuf, Vec<Saved>)>(0);
    std::thread::scope(|s| {
        let writer = s.spawn(move || -> io::Result<()> {
            for (path, mut part) in shards_in {
                write_safetensors(&path, &mut part)?;
            }
            Ok(())
        });
        let made = {
            let mut flush = |part: &mut Vec<Saved>, size: &mut u64| -> io::Result<()> {
                if !part.is_empty() {
                    let name = format!("glyd-part-{:05}.safetensors", shards.len() + 1); // renamed once the shards are known
                    shards.push((name.clone(), part.iter().map(|t| t.name.clone()).collect(), *size));
                    to_writer.send((out.join(name), std::mem::take(part))).map_err(|_| bad("the shards' writer stopped"))?;
                    *size = 0;
                }
                Ok(())
            };
            in_order(
                &plan.puts,
                &weights,
                &big,
                threads,
                shard_bytes,
                |p| p.run(&src, layout, codes.as_ref().map(|c| &c.0), hb),
                |d: Done| {
                    checked += d.checked;
                    calibrated += d.coded;
                    if let Some((name, e)) = d.entry {
                        packs.push((name, e));
                    }
                    for mut t in d.tensors {
                        if let Some(h) = t.sha256.take() {
                            hashes.push((t.name.clone(), Value::String(h)));
                        }
                        size += t.bytes.len() as u64;
                        part.push(t);
                        if size >= shard_bytes {
                            flush(&mut part, &mut size)?;
                        }
                    }
                    Ok(())
                },
            )
            .and_then(|_| flush(&mut part, &mut size))
        };
        drop(to_writer);
        let written = writer.join().unwrap_or_else(|_| Err(bad("the shards' writer panicked")));
        written.and(made) // (the writer's error first: the save's own is then its channel closed)
    })?;

    // glyd.json first out (a save cut short from here on has shards and no manifest, which a load refuses), the
    // earlier save's files, the shards renamed, the index
    let manifest = out.join("glyd.json");
    if manifest.exists() {
        std::fs::remove_file(&manifest).map_err(at(&manifest))?;
    }
    for p in old {
        std::fs::remove_file(&p).map_err(at(&p))?;
    }
    let names: Vec<String> = if shards.len() == 1 {
        vec!["model.safetensors".into()]
    } else {
        (1..=shards.len()).map(|i| format!("model-{i:05}-of-{:05}.safetensors", shards.len())).collect()
    };
    for ((tmp, _, _), name) in shards.iter().zip(&names) {
        std::fs::rename(out.join(tmp), out.join(name)).map_err(at(&out.join(name)))?;
    }
    if shards.len() > 1 {
        let map = shards.iter().zip(&names).flat_map(|((_, ks, _), name)| ks.iter().map(move |k| (k.clone(), Value::from(name.as_str())))).collect();
        let total: u64 = shards.iter().map(|s| s.2).sum();
        let index =
            Value::Object(vec![("metadata".into(), Value::Object(vec![("total_size".into(), Value::from(total))])), ("weight_map".into(), Value::Object(map))]);
        let p = out.join("model.safetensors.index.json");
        std::fs::write(&p, json::to_python(&index, 2)).map_err(at(&p))?;
    }
    let source_v = Value::Object(vec![
        ("repo".into(), Value::String(source.repo.clone())),
        ("revision".into(), source.revision.clone().map_or(Value::Null, Value::String)),
    ]);
    let hashed = hashes.len();
    let mut m = Value::Object(vec![
        (
            "format".into(),
            Value::from(match layout {
                Layout::Twelve => "glyd-v3",
                Layout::Compact => "glyd-v6",
                Layout::Tiered if penguin => "glyd-v7",
                Layout::Tiered => "glyd-v4",
                Layout::Int6
                | Layout::Int5
                | Layout::Int4
                | Layout::Int4H
                | Layout::Int3
                | Layout::Int6Row
                | Layout::Int5Row
                | Layout::Int4Row
                | Layout::Cb3Row
                | Layout::Cb2Row
                | Layout::Hummingbird => "glyd-v7",
            }),
        ),
        ("glyd".into(), Value::from(env!("CARGO_PKG_VERSION"))),
        ("source".into(), source_v),
        ("layout".into(), Value::from(layout.name())),
        ("packs".into(), Value::Object(packs)),
        ("tensors".into(), Value::Object(hashes)),
    ]);
    if let (Value::Object(o), Some(level)) = (
        &mut m,
        match layout {
            Layout::Tiered if penguin => Some("penguin"),
            Layout::Int6 => Some("kestrel"),
            Layout::Int5 => Some("swift"),
            Layout::Int4 | Layout::Int4H => Some("wren"),
            Layout::Hummingbird => Some("hummingbird"),
            _ => None,
        },
    ) {
        o.push(("level".into(), Value::from(level))); // (the level the lossy layout serves)
    }
    if let (true, Value::Object(o), Some((_, cal))) = (EXTRAS, &mut m, &codes) {
        o.push(("calibration".into(), cal.clone())); // (how the codes were made: calibration.json as it is)
    }
    if let (true, Value::Object(o), Some(ex)) = (EXTRAS, &mut m, hb) {
        let (twos, threes) = ex.bit_counts();
        let cal = Value::Object(vec![
            ("method".into(), Value::from("hummingbird")),
            ("recipe".into(), Value::from(ex.recipe.as_str())),
            ("experts_at_2_bits".into(), Value::from(twos as u64)),
            ("experts_at_3_bits".into(), Value::from(threes as u64)),
            ("export".into(), ex.meta_brief()),
        ]);
        o.push(("calibration".into(), cal)); // (how the codes were made: the export's meta.json without the signs and the experts' bits)
    }
    std::fs::write(&manifest, json::to_python(&m, 1)).map_err(at(&manifest))?;
    // the source's config, generation config and tokenizer files, each once, its content alone
    if std::fs::canonicalize(&source.dir)? != std::fs::canonicalize(out)? {
        let mut files: Vec<PathBuf> = std::fs::read_dir(&source.dir)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.is_file() && p.file_name().and_then(|n| n.to_str()).is_some_and(|n| !n.starts_with('.') && FILES.iter().any(|f| matches(f, n))))
            .collect();
        files.sort();
        for f in files {
            let to = out.join(f.file_name().unwrap());
            std::fs::write(&to, std::fs::read(&f).map_err(at(&f))?).map_err(at(&to))?;
            // (the Hub's cache keeps them read-only: not the mode)
        }
    }
    Ok(Saving { checked, hashed, bytes, shards: shards.len(), calibrated })
}

/// The formats `verify` reads (format.py's FORMATS; glyd-v5, its TABLES: a save whose packs hold a table too, as a Linear's), and those of them whose tiered packs are the layout of glyd 0.27 and
/// before (format.py's OLD_FORMATS: [`crate::v1`]).
const FORMATS: &[&str] = &["glyd-v1", "glyd-v2", "glyd-v3", "glyd-v4", "glyd-v5", "glyd-v6", "glyd-v7"];
const OLD_FORMATS: &[&str] = &["glyd-v1", "glyd-v2"];
/// glyd.json's keys (glyd 0.21 on; "tensors" from 0.25), format.py's KEYS: verify refuses any other in a save of this
/// glyd or an older one (a key damaged), and skips it in a newer one's.
const KEYS: &[&str] = &["format", "glyd", "source", "layout", "packs", "tensors", "level", "calibration"];
/// The first glyd whose saves carry "tensors" (format.py's MAP_FROM): a glyd-v1 or v2 saved by it or later without
/// them is refused, as a glyd-v3 or v4 is.
const MAP_FROM: (u64, u64) = (0, 25);

/// Where a pack's matrix is decoded for `verify`: on the CPU, or on a GPU by the library.
pub enum Decoder<'a> {
    Cpu,
    Gpu(&'a crate::Library, &'a crate::cuda::Context),
}

fn u64s(v: Option<&Value>) -> Option<Vec<u64>> {
    v.and_then(Value::as_u64s)
}

/// A pack's buffers' key (`NAME.glyd_`; an experts' weight's `MODULE.glyd_WEIGHT_`).
fn pack_key(name: &str, e: &Value) -> String {
    match name.rsplit_once('.') {
        Some((module, weight)) if e.get("experts").is_some() => format!("{module}.glyd_{weight}_"),
        _ => format!("{name}.glyd_"),
    }
}

/// Whether a pack's entry in glyd.json names a row layout.
fn is_rows(e: &Value) -> bool {
    e.get("layout").and_then(Value::as_str).and_then(Layout::by_name).is_some_and(|l| l.row_bits().is_some() || l.cb_bits().is_some())
}

/// A pack of `dir` read and decoded: its entry in glyd.json, whether its save is of a format whose tiered packs are the
/// earlier layout (`old`), the matrix [rows, cols], bf16. Its buffers are read and held to its shape before its matrix is
/// made, and (on a GPU) to where the decode reads by them.
fn decode(dir: &Checkpoint, name: &str, e: &Value, old: bool, decoder: &Decoder) -> io::Result<(usize, usize, Vec<u16>)> {
    let what = |why: &str| bad(format!("{name}: {why}"));
    let layout = e.get("layout").and_then(Value::as_str).and_then(Layout::by_name).ok_or_else(|| what("a layout this glyd does not read"))?;
    let shape = u64s(e.get("shape")).filter(|s| s.len() == 2).ok_or_else(|| what("no shape in glyd.json"))?;
    if layout == Layout::Hummingbird {
        return Err(what("the layout of a whole save, not of a pack"));
    }
    if let Some(bits) = layout.row_bits() {
        return decode_rows(dir, name, e, layout, bits, &shape);
    }
    if let Some(bits) = layout.cb_bits() {
        return decode_cb(dir, name, e, bits, &shape);
    }
    // the C API's words: the tiered layout's three tiers; the 12-bit layout's base hb (0-120) in the first's bytes
    let words: Vec<u32> = match layout {
        Layout::Tiered | Layout::Compact | Layout::Int6 | Layout::Int5 | Layout::Int4 | Layout::Int4H | Layout::Int3 => {
            let t: Vec<u32> = u64s(e.get("tiers"))
                .filter(|t| t.len() == 3 && t.iter().all(|&x| x <= u32::MAX as u64))
                .ok_or_else(|| what("a pack's sizes are missing from glyd.json"))?
                .into_iter()
                .map(|x| x as u32)
                .collect();
            let tag = t[0] >> 24;
            let ok = match layout {
                Layout::Int6 => tag == pack::INT6_TAG,
                Layout::Int5 => tag == pack::INT5_TAG,
                Layout::Int4 => tag == pack::INT4_TAG,
                Layout::Int4H => tag == pack::INT4H_TAG,
                Layout::Int3 => tag == pack::INT3_TAG,
                _ => tag == if old { crate::v1::TAG } else { pack::TIERED_TAG } || (!old && tag == pack::COMPACT_TAG),
            };
            if !ok {
                return Err(what("a pack's sizes are not its layout's"));
            }
            t
        }
        Layout::Twelve => match e.get("hb") {
            None => return Err(what("a pack an unreleased glyd saved: save it again")),
            Some(h) => pack::twelve_words(h.as_u64().filter(|&h| h <= 120).ok_or_else(|| what("its hb is not an int 0-120"))? as u8).to_vec(),
        },
        Layout::Int6Row | Layout::Int5Row | Layout::Int4Row | Layout::Cb3Row | Layout::Cb2Row | Layout::Hummingbird => {
            unreachable!("a row layout's pack is decoded above")
        }
    };
    let (rows, cols) = (shape[0] as usize, shape[1] as usize);
    if rows == 0 || rows % 64 != 0 || cols == 0 || cols % 16 != 0 || rows > (1 << 40) / cols {
        return Err(what("not an mma layout's shape"));
    }
    let key = pack_key(name, e);
    let compact = layout == Layout::Tiered && !old && words.first().is_some_and(|&t| t >> 24 == pack::COMPACT_TAG);
    let (bufs, step_bytes) = if compact { Layout::Compact.buffers() } else { layout.buffers() };
    let mut read = bufs.iter().map(|&(b, dtype)| -> io::Result<Vec<u8>> {
        let (f, t) = dir.get(&format!("{key}{b}")).ok_or_else(|| what(&format!("no {key}{b} in its safetensors")))?;
        if t.dtype != dtype {
            return Err(what(&format!("{key}{b} is {}, not {dtype}", t.dtype)));
        }
        f.read(t)
    });
    let (data, a, base) = (read.next().unwrap()?, read.next().unwrap()?, read.next().unwrap()?);
    let steps = rows * cols / pack::STEP;
    if data.len() != steps * step_bytes || base.len() != (steps + 1) * 4 {
        return Err(what("its buffers are not its shape's"));
    }
    let i32s = |b: &[u8]| -> Vec<i32> { b.as_chunks::<4>().0.iter().map(|x| i32::from_le_bytes(*x)).collect() };
    let mut w = vec![0u16; rows * cols]; // (its size held to the data read)
    match (layout, decoder) {
        (Layout::Tiered | Layout::Compact | Layout::Int6 | Layout::Int5 | Layout::Int4 | Layout::Int4H | Layout::Int3, Decoder::Cpu) => {
            let (block_base, tiers) = (i32s(&base), [words[0], words[1], words[2]]);
            let p = pack::Tiered { rows, cols, data, blocks: a, block_base, tiers };
            let int = matches!(layout, Layout::Int6 | Layout::Int5 | Layout::Int4 | Layout::Int4H | Layout::Int3);
            let ok = if layout == Layout::Int6 {
                pack::unpack_int6(&p, &mut w)
            } else if layout == Layout::Int5 {
                pack::unpack_int5(&p, &mut w)
            } else if matches!(layout, Layout::Int4 | Layout::Int4H) {
                pack::unpack_int4(&p, &mut w)
            } else if layout == Layout::Int3 {
                pack::unpack_int3(&p, &mut w)
            } else if old {
                crate::v1::unpack(&crate::v1::Tiered { rows, cols, data: p.data, blocks: p.blocks, block_base: p.block_base, tiers }, &mut w)
            } else if compact {
                pack::expand_compact(&p).is_some_and(|t| pack::unpack_tiered(&t, &mut w))
            } else {
                pack::unpack_tiered(&p, &mut w)
            };
            if !ok {
                return Err(what(if int { "its scales are not its shape's" } else { "its blocks do not hold its escapes" }));
            }
        }
        (Layout::Int6Row | Layout::Int5Row | Layout::Int4Row | Layout::Cb3Row | Layout::Cb2Row | Layout::Hummingbird, _) => {
            unreachable!("a row layout's pack is decoded above")
        }
        (Layout::Twelve, Decoder::Cpu) => {
            if a.len() % 16 != 0 {
                return Err(what("its exceptions not a multiple of 4"));
            }
            let p = pack::Twelve { rows, cols, data, exc: i32s(&a), exc_base: i32s(&base), sym: [words[0], words[1], words[2], words[3]] };
            if !pack::unpack_twelve(&p, &mut w) {
                return Err(what("its exceptions are not within its exc"));
            }
        }
        (_, Decoder::Gpu(lib, ctx)) => {
            // Held first to where the decode reads by them: a tiered step's escapes within its block (the counts in
            // its digits), the 12-bit exceptions' runs within exc (a multiple of 4).
            let base = i32s(&base);
            // (a compact pack's escapes checked in its bytes restored: the tiered layout's)
            let restored = compact
                .then(|| {
                    pack::expand_compact(&pack::Tiered {
                        rows,
                        cols,
                        data: data.clone(),
                        blocks: Vec::new(),
                        block_base: Vec::new(),
                        tiers: [words[0], words[1], words[2]],
                    })
                    .map(|t| t.data)
                })
                .flatten();
            let ok = match layout {
                Layout::Tiered if old => crate::v1::blocks_hold(&data, &a, &base),
                Layout::Int6 | Layout::Int5 | Layout::Int4 | Layout::Int4H | Layout::Int3 => cols % 32 == 0 && a.len() == steps / 2 * 128, // (a scale for each run of 32 of a row, 128 bytes a pair of steps)
                Layout::Tiered | Layout::Compact => {
                    (!compact || restored.is_some()) && pack::tiered_blocks_hold(restored.as_deref().unwrap_or(&data), &a, &base)
                }
                Layout::Twelve => a.len() % 16 == 0 && base[0] == 0 && base.windows(2).all(|x| x[0] <= x[1]) && base[steps] as usize <= a.len() / 4,
                Layout::Int6Row | Layout::Int5Row | Layout::Int4Row | Layout::Cb3Row | Layout::Cb2Row | Layout::Hummingbird => {
                    unreachable!("a row layout's pack is decoded above")
                }
            };
            if !ok {
                return Err(what("its buffers do not hold what its data says (the GPU's decode would read past them)"));
            }
            let gpu = |r: crate::Result<()>| r.map_err(|e| bad(format!("{name}: {e}")));
            let (d, x, b) = (ctx.upload(&data).map_err(bad)?, ctx.upload(&a).map_err(bad)?, ctx.upload(&base).map_err(bad)?);
            let out = ctx.alloc(rows * cols * 2).map_err(bad)?;
            let p = match layout {
                Layout::Tiered | Layout::Compact | Layout::Int6 | Layout::Int5 | Layout::Int4 | Layout::Int4H | Layout::Int3 => {
                    crate::Pack::Tiered(crate::Tiered { data: d.ptr(), blocks: x.ptr(), block_base: b.ptr(), tiers: [words[0], words[1], words[2]] })
                }
                Layout::Twelve => {
                    crate::Pack::Twelve(crate::Twelve { data: d.ptr(), exc: x.ptr(), exc_base: b.ptr(), sym: [words[0], words[1], words[2], words[3]] })
                }
                Layout::Int6Row | Layout::Int5Row | Layout::Int4Row | Layout::Cb3Row | Layout::Cb2Row | Layout::Hummingbird => {
                    unreachable!("a row layout's pack is decoded above")
                }
            };
            let m = crate::Matrix { pack: p, rows: rows as i64, cols: cols as i64 };
            // SAFETY: the pack's buffers uploaded whole, their sizes and every place the decode reads by them held to
            // its shape above; out holds rows x cols.
            gpu(unsafe { lib.unpack(&m, 0, rows as i64, out.ptr(), 0, crate::Stream::DEFAULT) })?;
            gpu(ctx.synchronize())?;
            gpu(out.read(&mut w))?;
        }
    }
    Ok((rows, cols, w))
}

/// A row-layout pack of `dir` read and decoded, on the CPU (the library has no kernel for it, so a GPU decode of a save reads these packs here): its entry's shape is the pack's [in, out], its tensor's [out, in], and it
/// is returned as the tensor, [out, in] bf16, the weights it decodes to.
fn decode_rows(dir: &Checkpoint, name: &str, e: &Value, layout: Layout, bits: u32, shape: &[u64]) -> io::Result<(usize, usize, Vec<u16>)> {
    let what = |why: &str| bad(format!("{name}: {why}"));
    let t: Vec<u32> = u64s(e.get("tiers"))
        .filter(|t| t.len() == 3 && t.iter().all(|&x| x <= u32::MAX as u64))
        .ok_or_else(|| what("a pack's sizes are missing from glyd.json"))?
        .into_iter()
        .map(|x| x as u32)
        .collect();
    if pack::rows_bits(t[0] >> 24) != Some(bits) {
        return Err(what("a pack's sizes are not its layout's"));
    }
    let (rows, cols) = (shape[0] as usize, shape[1] as usize);
    if rows == 0 || cols == 0 || cols % 32 != 0 || rows > (1 << 40) / cols || (rows * cols) % pack::STEP != 0 {
        return Err(what("not a row layout's shape"));
    }
    let key = pack_key(name, e);
    let (bufs, _) = layout.buffers();
    let mut read = bufs.iter().map(|&(b, dtype)| -> io::Result<Vec<u8>> {
        let (f, t) = dir.get(&format!("{key}{b}")).ok_or_else(|| what(&format!("no {key}{b} in its safetensors")))?;
        if t.dtype != dtype {
            return Err(what(&format!("{key}{b} is {}, not {dtype}", t.dtype)));
        }
        f.read(t)
    });
    let (data, blocks, base) = (read.next().unwrap()?, read.next().unwrap()?, read.next().unwrap()?);
    if data.len() != rows * pack::rows_stride(bits, cols) || blocks.len() != rows * cols / 32 * 2 || base.len() != (rows * cols / pack::STEP + 1) * 4 {
        return Err(what("its buffers are not its shape's"));
    }
    let p = pack::Tiered { rows, cols, data, blocks, block_base: Vec::new(), tiers: [t[0], t[1], t[2]] };
    let mut w = vec![0u16; rows * cols];
    if !pack::unpack_rows(&p, &mut w) {
        return Err(what("its buffers are not its shape's"));
    }
    Ok((cols, rows, pack::transpose(&w, rows, cols)))
}

/// A codebook-row pack of `dir` read and decoded, on the CPU: its entry's shape is the pack's [neurons, row] (`shape`), and it is returned in its tensor's orientation, the pack as it is (an up matrix) or its transpose
/// (the entry's "transposed": a down matrix, whose tensor is [out, in]), the bf16 each weight's exact product rounds to.
fn decode_cb(dir: &Checkpoint, name: &str, e: &Value, bits: u32, shape: &[u64]) -> io::Result<(usize, usize, Vec<u16>)> {
    let what = |why: &str| bad(format!("{name}: {why}"));
    let layout = if bits == 3 { Layout::Cb3Row } else { Layout::Cb2Row };
    let t: Vec<u32> = u64s(e.get("tiers"))
        .filter(|t| t.len() == 3 && t.iter().all(|&x| x <= u32::MAX as u64))
        .ok_or_else(|| what("a pack's sizes are missing from glyd.json"))?
        .into_iter()
        .map(|x| x as u32)
        .collect();
    if codebook::cb_rows_bits(t[0] >> 24) != Some(bits) || t[1] != 0 || t[2] != 0 {
        return Err(what("a pack's sizes are not its layout's"));
    }
    let (rows, cols) = (shape[0] as usize, shape[1] as usize);
    if rows == 0 || cols == 0 || !cols.is_multiple_of(32) || rows > (1 << 40) / cols {
        return Err(what("not a codebook row layout's shape"));
    }
    let key = pack_key(name, e);
    let (bufs, _) = layout.buffers();
    let mut read = bufs.iter().map(|&(b, dtype)| -> io::Result<Vec<u8>> {
        let (f, t) = dir.get(&format!("{key}{b}")).ok_or_else(|| what(&format!("no {key}{b} in its safetensors")))?;
        if t.dtype != dtype {
            return Err(what(&format!("{key}{b} is {}, not {dtype}", t.dtype)));
        }
        f.read(t)
    });
    let (data, blocks, book) = (read.next().unwrap()?, read.next().unwrap()?, read.next().unwrap()?);
    let p = codebook::CbRows::new(bits, rows, cols, data, blocks, book).ok_or_else(|| what("its buffers are not its shape's"))?;
    let mut w = vec![0u16; rows * cols];
    if !codebook::unpack_codebook_rows(&p, &mut w) {
        return Err(what("its buffers are not its shape's"));
    }
    match e.get("transposed").and_then(Value::as_bool) {
        Some(true) => Ok((cols, rows, pack::transpose(&w, rows, cols))),
        Some(false) => Ok((rows, cols, w)),
        None => Err(what("no \"transposed\" in glyd.json: a codebook row pack is its tensor or its transpose")),
    }
}

/// A tensor a pack holds, as the model holds it: its name, its shape and sha256 in glyd.json, its bf16.
struct Held<'a> {
    name: &'a str,
    shape: Vec<u64>,
    sha256: &'a str,
    w: std::borrow::Cow<'a, [u16]>,
}

/// The tensors of a pack whose matrix decoded to `w` [rows, cols]: a merged pack's in their rows' order, an experts' weight as the model holds
/// it ([E, out, in], or [E, in, out] where the entry says "transposed").
fn held<'a>(name: &'a str, e: &'a Value, rows: usize, cols: usize, w: &'a [u16]) -> io::Result<Vec<Held<'a>>> {
    let tensors = e.get("tensors").and_then(Value::as_array).filter(|t| !t.is_empty()).ok_or_else(|| bad(format!("{name}: no tensors in glyd.json")))?;
    let mut row = 0;
    let mut out = Vec::with_capacity(tensors.len());
    for t in tensors {
        let tn = t.get("name").and_then(Value::as_str).unwrap_or(name);
        let sha256 = t.get("sha256").and_then(Value::as_str).ok_or_else(|| bad(format!("{tn}: no sha256 in glyd.json")))?;
        let shape = u64s(t.get("shape")).ok_or_else(|| bad(format!("{tn}: no shape in glyd.json")))?;
        let got: std::borrow::Cow<[u16]> = match (e.get("experts").and_then(Value::as_u64), shape.as_slice()) {
            (Some(n), &[en, a, b]) if en == n && n.checked_mul(a).and_then(|x| x.checked_mul(b)) == Some((rows * cols) as u64) => {
                if e.get("transposed").and_then(Value::as_bool).unwrap_or(false) {
                    // held [E, in, out]: each expert's [out, in] matrix transposed
                    let (o, k) = (rows / n as usize, cols);
                    let mut held = vec![0u16; w.len()];
                    for x in 0..n as usize {
                        for r in 0..o {
                            for c in 0..k {
                                held[x * o * k + c * o + r] = w[x * o * k + r * k + c];
                            }
                        }
                    }
                    held.into()
                } else {
                    w.into()
                }
            }
            (None, &[r, c]) if c as usize == cols && r <= (rows - row) as u64 => {
                row += r as usize;
                w[(row - r as usize) * cols..row * cols].into()
            }
            _ => return Err(bad(format!("{tn}: its shape {shape:?} in glyd.json is not its pack's"))),
        };
        out.push(Held { name: tn, shape, sha256, w: got });
    }
    if e.get("experts").is_none() && row != rows {
        return Err(bad(format!("{name}: its tensors' rows are not its own")));
    }
    Ok(out)
}

/// A pack's tensors against glyd.json's sha256: how many.
fn hashes_match(name: &str, e: &Value, rows: usize, cols: usize, w: &[u16]) -> io::Result<usize> {
    let ts = held(name, e, rows, cols, w)?;
    if let Some(t) = ts.iter().find(|t| sha256_u16(&t.w) != t.sha256) {
        return Err(bad(format!("{} decodes to other bytes than glyd.json's sha256", t.name)));
    }
    Ok(ts.len())
}

/// Every tensor of the saved checkpoint `dir` as the model holds it, in f32, once [`verify`] has held the whole save to its glyd.json: each pack decoded (on the
/// CPU, or by the library on a GPU; a row or codebook layout's on the CPU, as verify does) and cut into the tensors it holds, and each tensor saved as it is (BF16, F32
/// or I64). The weights a reader of the save multiplies by: what a CPU reference runs a saved model on. Tensors of another type (a Hummingbird layer's I8 signs) are
/// left out.
pub fn decoded(dir: &Path, threads: usize, decoder: &Decoder) -> io::Result<Vec<(String, Vec<u64>, Vec<f32>)>> {
    verify(dir, threads, decoder)?;
    let m = json::parse(&std::fs::read_to_string(dir.join("glyd.json"))?).map_err(bad)?;
    let old = OLD_FORMATS.contains(&m.get("format").and_then(Value::as_str).unwrap_or(""));
    let saved = Checkpoint::open(dir)?;
    let f32_of = |w: &[u16]| w.iter().map(|&x| f32::from_bits((x as u32) << 16)).collect::<Vec<f32>>();
    let mut out = Vec::new();
    let mut buffers = HashSet::new();
    for (name, e) in m.get("packs").and_then(Value::as_object).unwrap_or(&[]) {
        let layout =
            e.get("layout").and_then(Value::as_str).and_then(Layout::by_name).ok_or_else(|| bad(format!("{name}: a layout this glyd does not read")))?;
        let key = pack_key(name, e);
        buffers.extend(layout.buffers().0.iter().map(|(b, _)| format!("{key}{b}")));
        let on = if is_rows(e) { &Decoder::Cpu } else { decoder };
        let (rows, cols, w) = decode(&saved, name, e, old, on)?;
        out.extend(held(name, e, rows, cols, &w)?.into_iter().map(|t| (t.name.to_string(), t.shape, f32_of(&t.w))));
    }
    for name in saved.names().filter(|n| !buffers.contains(*n)) {
        let (f, t) = saved.get(name).ok_or_else(|| bad(format!("{name}: not in the safetensors")))?;
        let b = f.read(t)?;
        let v: Vec<f32> = match t.dtype.as_str() {
            "BF16" => b.chunks_exact(2).map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16)).collect(),
            "F32" => b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
            "I64" => b.chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap_or_default()) as f32).collect(),
            _ => continue,
        };
        out.push((name.to_string(), t.shape.clone(), v));
    }
    Ok(out)
}

/// What a verify found.
pub struct Verified {
    /// Packed tensors decoded to glyd.json's sha256.
    pub packed: usize,
    /// Tensors saved as they are at glyd.json's sha256.
    pub hashed: usize,
    /// Tensors saved as they are with no sha256 in glyd.json to check (a save of glyd 0.24 or before).
    pub unchecked: usize,
    /// glyd.json's keys this glyd does not have, of a newer glyd's save: skipped (the rest checked).
    pub skipped: Vec<String>,
}

/// glyd.json's glyd ("0.25.0") as (0, 25, 0), its patch 0 where there is none, as format.py's version reads it from
/// its start: ASCII digits, a point, digits (then a point and digits, or anything).
fn version(v: &str) -> Option<(u64, u64, u64)> {
    fn digits(s: &str) -> (Option<u64>, &str) {
        let n = s.bytes().take_while(u8::is_ascii_digit).count();
        (s[..n].parse().ok(), &s[n..])
    }
    let (major, rest) = digits(v);
    let (minor, rest) = digits(rest.strip_prefix('.')?);
    let patch = rest.strip_prefix('.').and_then(|r| digits(r).0).unwrap_or(0);
    Some((major?, minor?, patch))
}

/// A saved checkpoint (glyd-v1, glyd-v2, glyd-v3) checked, as `python -m glyd.gpu verify` checks it: glyd.json's
/// keys its own (another refused where this glyd or an older one saved it: damaged; skipped where a newer one did,
/// named in `skipped`), its glyd a version, its map of sha256 an object, and there wherever its format
/// or its glyd says it is (glyd-v3; glyd-v1 and v2 saved by glyd 0.25 on); each file's tensors back to back to its
/// end (the reader); the index, where there are shards, naming each tensor's shard; every tensor a pack's buffer or
/// one whose sha256 glyd.json holds (a save of glyd 0.24 or before has none: those are counted as unchecked); each
/// pack's tensors its own (its module's .weight, a merged group's
/// q, k, v or gate, up under the first's path, an experts' weight's own name), in no other pack and none also saved
/// as it is; every pack decoded (on `threads` threads of the CPU, or on a GPU) and each of its tensors' sha256, and
/// each tensor saved as it is, against glyd.json's.
pub fn verify(dir: &Path, threads: usize, decoder: &Decoder) -> io::Result<Verified> {
    let path = dir.join("glyd.json");
    let text = std::fs::read_to_string(&path).map_err(at(&path))?;
    let m = json::parse(&text).map_err(|e| bad(format!("{}: {e}", path.display())))?;
    let top = m.as_object().ok_or_else(|| bad(format!("{}: not an object", path.display())))?;
    // the glyd that saved it; a key glyd.json does not have: damaged where this glyd or an older one saved it, a newer
    // one's (skipped, the rest checked) where a newer one did
    let glyd = m.get("glyd").and_then(Value::as_str).unwrap_or("");
    let by = version(glyd).ok_or_else(|| bad(format!("{}: glyd {glyd:?}, not a version", path.display())))?;
    let this = version(env!("CARGO_PKG_VERSION")).expect("the crate's version");
    let mut skipped = Vec::new();
    for (k, _) in top.iter().filter(|(k, _)| !KEYS.contains(&k.as_str())) {
        if by <= this {
            return Err(bad(format!("{}: {k:?}, a key glyd.json does not have (damaged?)", path.display())));
        }
        skipped.push(k.clone());
    }
    let format = m.get("format").and_then(Value::as_str).unwrap_or("");
    if !FORMATS.contains(&format) {
        return Err(bad(format!("{}: format {format:?}; this glyd reads {}", path.display(), FORMATS.join(" and "))));
    }
    let hashes: Option<Vec<(String, Value)>> = match m.get("tensors") {
        None => None,
        Some(Value::Object(o)) => Some(o.to_vec()),
        Some(_) => return Err(bad(format!("{}: \"tensors\" is not an object of sha256", path.display()))),
    };
    if hashes.is_none() && (!OLD_FORMATS.contains(&format) || (by.0, by.1) >= MAP_FROM) {
        return Err(bad(format!("{}: no sha256 for the tensors saved as they are, which a {format} of glyd {glyd} has", path.display())));
    }
    let saved = Checkpoint::open(dir)?;
    if let Some(index) = &saved.index {
        for (name, shard) in index {
            if saved.get(name).is_none_or(|(f, _)| f.path != dir.join(shard)) {
                return Err(bad(format!("model.safetensors.index.json: {name} not in {shard}, its shard")));
            }
        }
        if index.len() != saved.names().count() {
            return Err(bad("model.safetensors.index.json: not every tensor of the shards named, or one named twice"));
        }
    }
    let packs: Vec<(String, Value)> = m.get("packs").and_then(Value::as_object).ok_or_else(|| bad(format!("{}: no packs", path.display())))?.to_vec();
    let hashed: HashMap<&str, &str> = hashes.iter().flatten().map(|(k, v)| (k.as_str(), v.as_str().unwrap_or(""))).collect();
    let (mut buffers, mut members) = (HashSet::new(), HashSet::new());
    for (name, e) in &packs {
        let layout =
            e.get("layout").and_then(Value::as_str).and_then(Layout::by_name).ok_or_else(|| bad(format!("{name}: a layout this glyd does not read")))?;
        let key = pack_key(name, e);
        buffers.extend(layout.buffers().0.iter().map(|(b, _)| format!("{key}{b}")));
        let ts: Vec<&str> =
            e.get("tensors").and_then(Value::as_array).unwrap_or(&[]).iter().map(|t| t.get("name").and_then(Value::as_str).unwrap_or("")).collect();
        // its own: an experts' weight's name, its module's .weight, a merged group's (q, k, v; gate, up: every name
        // the key's; q, k alone where a layer's values are its keys, Gemma 4's global layers)
        let (parent, first) = name.rsplit_once('.').unwrap_or(("", name));
        let own: Vec<String> = match (e.get("experts"), ts.len()) {
            (Some(_), _) => vec![name.clone()],
            (None, 1) => vec![format!("{name}.weight")],
            _ => [QKV, GATE_UP]
                .iter()
                .filter(|g| g[0] == first && g.len() >= ts.len())
                .flat_map(|g| g[..ts.len()].iter().map(|c| format!("{parent}.{c}.weight")))
                .collect(),
        };
        if ts != own {
            return Err(bad(format!("glyd.json: {name}'s tensors are not its own: {:?}", &ts[..ts.len().min(3)])));
        }
        for t in ts {
            if !members.insert(t.to_string()) {
                return Err(bad("glyd.json: a tensor held by two packs, or twice by one"));
            }
            if hashed.contains_key(t) {
                return Err(bad(format!("glyd.json: {t} both packed and saved as it is")));
            }
        }
    }
    let mut unchecked = 0;
    for name in saved.names() {
        if buffers.contains(name) {
            continue;
        }
        if hashes.is_none() {
            unchecked += 1;
        } else if !hashed.contains_key(name) {
            return Err(bad(format!("{name}: in the safetensors, but glyd.json neither packs it nor holds its sha256")));
        }
    }
    if let Some(b) = buffers.iter().find(|b| saved.get(b).is_none()) {
        return Err(bad(format!("{b}: a pack's buffer, not in the safetensors")));
    }
    for (name, want) in &hashed {
        let (f, t) = saved.get(name).ok_or_else(|| bad(format!("{name}: glyd.json's sha256, but not in the safetensors")))?;
        if hex(&Sha256::digest(f.read(t)?)) != *want {
            return Err(bad(format!("{name} is other bytes than glyd.json's sha256")));
        }
    }
    let mut n = 0;
    let old = OLD_FORMATS.contains(&format);
    let one = |(name, e): &(String, Value), decoder: &Decoder| decode(&saved, name, e, old, decoder).and_then(|(r, c, w)| hashes_match(name, e, r, c, &w));
    match decoder {
        // a pack and its matrix a thread, and one more waiting its turn
        Decoder::Cpu => in_order(
            &packs,
            &vec![1; packs.len()],
            &vec![false; packs.len()],
            threads,
            threads as u64 + 1,
            |p| one(p, &Decoder::Cpu),
            |k| {
                n += k;
                Ok(())
            },
        )?,
        Decoder::Gpu(..) => {
            // (the library has no kernel for a row layout: its packs are decoded on the CPU, on the threads, and the others by the GPU one after another)
            let (rows, others): (Vec<_>, Vec<_>) = packs.iter().cloned().partition(|(_, e)| is_rows(e));
            in_order(
                &rows,
                &vec![1; rows.len()],
                &vec![false; rows.len()],
                threads,
                threads as u64 + 1,
                |p| one(p, &Decoder::Cpu),
                |k| {
                    n += k;
                    Ok(())
                },
            )?;
            for p in &others {
                n += one(p, decoder)?;
            }
        }
    }
    Ok(Verified { packed: n, hashed: hashed.len(), unchecked, skipped })
}

#[cfg(test)]
mod pack_compat;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn file_patterns() {
        assert!(matches("tokenizer*", "tokenizer.json") && matches("*.model", "spm.model") && matches("vocab*", "vocab.json"));
        assert!(!matches("*.model", "model") && !matches("config.json", "config.json.bak"));
    }

    /// glyd.json's glyd read as format.py's version reads it (its test holds the same list): from its start, ASCII
    /// digits, a point, digits, then a point and digits or anything; else not a version (a separator one bit off).
    #[test]
    fn versions_as_python_reads_them() {
        for (v, want) in [
            ("0.25.0", Some((0, 25, 0))),
            ("0.25", Some((0, 25, 0))),
            ("0.25rc1.3", Some((0, 25, 0))),
            ("0.25.0.dev0", Some((0, 25, 0))),
            ("0.25.", Some((0, 25, 0))),
            ("12.3.4", Some((12, 3, 4))),
        ] {
            assert_eq!(version(v), want, "{v}");
        }
        for v in ["0/25.0", "0,25.0", "0x.25.0", ".25.0", "0.x", "", "v0.25", "0", "0.\u{0663}"] {
            assert_eq!(version(v), None, "{v}");
        }
    }

    /// in_order hands every result to `done` in order, whatever the threads, the weights and the budget (one item
    /// heavier than the whole budget among them), the big ones taken first.
    #[test]
    fn in_order_keeps_its_order() {
        let items: Vec<usize> = (0..200).collect();
        for (threads, budget) in [(1, 1), (4, 3), (8, 1000), (3, 10)] {
            let weights: Vec<u64> = items.iter().map(|&i| if i == 7 { 5000 } else { 1 + (i % 5) as u64 }).collect();
            let big: Vec<bool> = items.iter().map(|&i| i % 50 == 49).collect();
            let mut got = Vec::new();
            in_order(
                &items,
                &weights,
                &big,
                threads,
                budget,
                |&i| Ok(i * 3),
                |r| {
                    got.push(r);
                    Ok(())
                },
            )
            .unwrap();
            assert_eq!(got, items.iter().map(|i| i * 3).collect::<Vec<_>>(), "{threads} threads, budget {budget}");
        }
    }

    /// A glyd-v1 save (the tiered layout of glyd 0.20 to 0.27: one pack, as that glyd's pack_mma made it) verified on the
    /// CPU: its tensors against glyd.json's sha256; a save of glyd 0.24, which has no map of them for the tensors saved as
    /// they are, passes with them unchecked; and refused: the pack's words as the new layout's, the same pack in a
    /// glyd-v4 save, a block cut short.
    #[test]
    fn legacy_saves_are_read() {
        let dir = std::env::temp_dir().join(format!("glyd-gpu-legacy-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let (rows, cols) = (128, 192);
        let w = pack::tests::weights(rows * cols, 0.01, 17);
        let p = crate::v1::pack(&w, rows, cols);
        let steps = (rows * cols / pack::STEP) as u64;
        let saved = |name: &str, dtype: &'static str, n: u64, bytes: Vec<u8>| Saved { name: name.into(), dtype, shape: vec![n], bytes, sha256: None };
        let mut ts = vec![
            saved("m.o_proj.glyd_data", "U8", steps * 1280, p.data.clone()),
            saved("m.o_proj.glyd_blocks", "U8", p.blocks.len() as u64, p.blocks.clone()),
            saved("m.o_proj.glyd_block_base", "I32", steps + 1, le_bytes(&p.block_base, i32::to_le_bytes)),
        ];
        write_safetensors(&dir.join("model.safetensors"), &mut ts).unwrap();
        let manifest = |format: &str, glyd: &str, tiers: [u32; 3], map: bool| {
            let tiers = tiers.iter().map(u32::to_string).collect::<Vec<_>>().join(", ");
            let tensors = if map { ", \"tensors\": {}" } else { "" };
            std::fs::write(
                dir.join("glyd.json"),
                format!(
                    "{{\"format\": \"{format}\", \"glyd\": \"{glyd}\", \"source\": {{\"repo\": null, \"revision\": null}}, \"layout\": \"mma\", \"packs\": {{\"m.o_proj\": {{\"layout\": \"mma\", \"shape\": [{rows}, {cols}], \"tiers\": [{tiers}], \"tensors\": [{{\"name\": \"m.o_proj.weight\", \"shape\": [{rows}, {cols}], \"sha256\": \"{}\"}}]}}}}{tensors}}}",
                    sha256_u16(&w)
                ),
            )
            .unwrap();
        };
        manifest("glyd-v1", "0.27.0", p.tiers, true);
        let v = verify(&dir, 1, &Decoder::Cpu).unwrap();
        assert_eq!((v.packed, v.hashed, v.unchecked), (1, 0, 0));
        manifest("glyd-v1", "0.24.0", p.tiers, false);
        assert_eq!(verify(&dir, 1, &Decoder::Cpu).unwrap().packed, 1);
        let refused = |why: &str| {
            let e = verify(&dir, 1, &Decoder::Cpu).err().map(|e| e.to_string()).unwrap_or_default();
            assert!(e.contains(why), "not refused for {why:?}: {e:?}");
        };
        manifest("glyd-v1", "0.27.0", [p.tiers[0] & 0xFFFFFF | pack::TIERED_TAG << 24, p.tiers[1], p.tiers[2]], true);
        refused("a pack's sizes are not its layout's");
        manifest("glyd-v4", "0.28.0", p.tiers, true);
        refused("a pack's sizes are not its layout's");
        let mut cut = p.blocks.clone();
        cut.truncate(cut.len() - 300); // the last blocks gone: a step's block past the end
        let mut ts = vec![
            saved("m.o_proj.glyd_data", "U8", steps * 1280, p.data.clone()),
            saved("m.o_proj.glyd_blocks", "U8", cut.len() as u64, cut),
            saved("m.o_proj.glyd_block_base", "I32", steps + 1, le_bytes(&p.block_base, i32::to_le_bytes)),
        ];
        write_safetensors(&dir.join("model.safetensors"), &mut ts).unwrap();
        manifest("glyd-v1", "0.27.0", p.tiers, true);
        refused("do not hold its escapes");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A matrix packed in a block layout: its entry in that layout's name (the lossless layouts keep "mma" and "mma12"), `w` left the
    /// weights it decodes to, the shape refused.
    #[test]
    fn int_packs_are_named_and_leave_their_decoded_weights() {
        for (layout, name, tag) in [
            (Layout::Int6, "i6", pack::INT6_TAG),
            (Layout::Int5, "i5", pack::INT5_TAG),
            (Layout::Int4, "i4", pack::INT4_TAG),
            (Layout::Int3, "i3", pack::INT3_TAG),
        ] {
            let (rows, cols) = (128, 96);
            let orig = pack::tests::weights(rows * cols, 0.0, 9);
            let mut w = orig.clone();
            let (ts, words, lay) = packed(layout, &mut w, rows, cols, "x.glyd_", "x", None).unwrap();
            assert!(lay == layout && words[0] >> 24 == tag && ts.len() == 3 && w != orig);
            let mut back = vec![0u16; w.len()];
            let p = pack::Tiered {
                rows,
                cols,
                data: ts[0].bytes.clone(),
                blocks: ts[1].bytes.clone(),
                block_base: vec![0; rows * cols / pack::STEP + 1],
                tiers: [words[0], words[1], words[2]],
            };
            let unpack = match layout {
                Layout::Int6 => pack::unpack_int6,
                Layout::Int5 => pack::unpack_int5,
                Layout::Int4 => pack::unpack_int4,
                _ => pack::unpack_int3,
            };
            assert!(unpack(&p, &mut back) && back == w);
            let e = entry(lay, [rows, cols], &words, vec![], None);
            assert_eq!(e.get("layout").and_then(Value::as_str), Some(name));
            assert_eq!(Layout::by_name(name), Some(layout));
            assert!(packed(layout, &mut orig[..32 * 96].to_vec(), 32, 96, "x.glyd_", "x", None).is_err());
        }
        assert_eq!([Layout::Tiered, Layout::Compact, Layout::Twelve].map(Layout::name), ["mma", "mma", "mma12"]);
    }

    /// The tiny Qwen3 saved in each block layout (glyd-v7, its level in glyd.json: none for the three-bit one) and verified on the CPU: every pack decoded bit
    /// for bit to the weights glyd.json hashes (the decoded ones, not the checkpoint's); then refused: one byte of a pack changed.
    #[test]
    fn int_saves_are_verified() {
        let dir = std::env::temp_dir().join(format!("glyd-gpu-int-{}", std::process::id()));
        let src = dir.join("src");
        tiny_qwen3(&src);
        let source = Source::find(src.to_str().unwrap()).unwrap();
        for (layout, name, level) in
            [(Layout::Int5, "i5", Some("swift")), (Layout::Int6, "i6", Some("kestrel")), (Layout::Int4, "i4", Some("wren")), (Layout::Int3, "i3", None)]
        {
            let out = dir.join(name);
            save(&source, &out, 2, 1 << 30, true, layout).unwrap();
            let m = json::parse(&std::fs::read_to_string(out.join("glyd.json")).unwrap()).unwrap();
            let s = |k: &str| m.get(k).and_then(Value::as_str);
            assert_eq!((s("format"), s("layout"), s("level")), (Some("glyd-v7"), Some(name), level));
            let v = verify(&out, 2, &Decoder::Cpu).unwrap();
            assert_eq!((v.packed, v.hashed, v.unchecked), (15, 9, 0), "{name}"); // (the embedding a pack, not a tensor saved as it is)
            let c = Checkpoint::open(&out).unwrap();
            let e = m.get("packs").unwrap().get("model.embed_tokens").expect("the embedding is packed");
            assert_eq!((e.get("layout").and_then(Value::as_str), u64s(e.get("shape"))), (Some(name), Some(vec![256, 128])));
            assert!(c.get("model.embed_tokens.weight").is_none() && c.get("model.embed_tokens.glyd_blocks").is_some());
            let packed_bytes: u64 = ["data", "blocks", "block_base"].iter().map(|b| c.get(&format!("model.embed_tokens.glyd_{b}")).unwrap().1.bytes).sum();
            assert!(packed_bytes < 256 * 128 * 2, "{name}: {packed_bytes} bytes, the bf16 embedding's {}", 256 * 128 * 2);
            let (f, info) = c.get("model.layers.0.self_attn.o_proj.glyd_data").unwrap();
            let mut bytes = std::fs::read(&f.path).unwrap();
            bytes[info.offset as usize + 5] ^= 0x10;
            std::fs::write(&f.path, bytes).unwrap();
            let e = verify(&out, 2, &Decoder::Cpu).err().map(|e| e.to_string()).unwrap_or_default();
            assert!(e.contains("decodes to other bytes"), "{name}: {e:?}");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The tiny Qwen3 saved at the four-bit layout with its MLPs and its embedding put in the six-bit one by the plan (a linear's "layout"): glyd.json names the save's layout (i4, the level wren) and each pack's own, with
    /// the layout's tag in the pack's words, and verify decodes each pack in its own on the CPU to the sha256 glyd.json holds; refused: a pack whose entry names the other block layout than its bytes are in, a
    /// plan's layout that is no block layout, and one in a save that is not in a block layout.
    #[test]
    fn a_save_may_hold_packs_in_more_than_one_block_layout() {
        let dir = std::env::temp_dir().join(format!("glyd-gpu-mixed-{}", std::process::id()));
        let src = dir.join("src");
        tiny_qwen3(&src);
        let source = Source::find(src.to_str().unwrap()).unwrap();
        let config = json::parse(&std::fs::read_to_string(src.join("config.json")).unwrap()).unwrap();
        let puts = Plan::new(&Checkpoint::open(&src).unwrap(), &config, true, true).unwrap().puts;
        let six = |members: &[String]| members.iter().any(|m| m.contains(".mlp.") || m.contains("embed_tokens"));
        let plan = |own: &str| {
            let items: Vec<String> = puts
                .iter()
                .map(|p| match p {
                    Put::Linear { members, .. } => format!(
                        "{{\"linear\": [{}]{}}}",
                        members.iter().map(|m| format!("\"{m}\"")).collect::<Vec<_>>().join(", "),
                        if six(members) { format!(", \"layout\": \"{own}\"") } else { String::new() }
                    ),
                    Put::Copy { name, from, .. } => format!("{{\"copy\": \"{name}\", \"from\": \"{from}\"}}"),
                    Put::Experts { .. } | Put::Codebook { .. } | Put::Signs { .. } => unreachable!(),
                })
                .collect();
            format!("{{\"puts\": [{}]}}", items.join(", "))
        };
        let out = dir.join("mixed");
        save_planned(&source, &out, 2, 1 << 30, Layout::Int4, &plan("i6")).unwrap();
        let m = json::parse(&std::fs::read_to_string(out.join("glyd.json")).unwrap()).unwrap();
        let s = |k: &str| m.get(k).and_then(Value::as_str);
        assert_eq!((s("format"), s("layout"), s("level")), (Some("glyd-v7"), Some("i4"), Some("wren")));
        let (mut four, mut six_bit) = (0, 0);
        for (name, e) in m.get("packs").and_then(Value::as_object).unwrap() {
            let (layout, tag) = if name.contains(".mlp.") || name.contains("embed_tokens") {
                six_bit += 1;
                ("i6", pack::INT6_TAG)
            } else {
                four += 1;
                ("i4", pack::INT4_TAG)
            };
            assert_eq!(e.get("layout").and_then(Value::as_str), Some(layout), "{name}");
            assert_eq!(u64s(e.get("tiers")).map(|t| (t[0] >> 24) as u32), Some(tag), "{name}");
        }
        assert_eq!(
            (four, six_bit),
            (4, 5),
            "two layers' attention packs (q, k, v as one, and o); their MLPs' (gate and up as one, and down) and the embedding"
        );
        let v = verify(&out, 2, &Decoder::Cpu).unwrap();
        assert_eq!((v.packed, v.hashed, v.unchecked), (15, 9, 0));
        // a pack saved in one layout and named as the other: its words are not that layout's
        let text = std::fs::read_to_string(out.join("glyd.json")).unwrap();
        std::fs::write(out.join("glyd.json"), text.replacen("\"layout\": \"i6\"", "\"layout\": \"i4\"", 1)).unwrap();
        let e = verify(&out, 2, &Decoder::Cpu).err().map(|e| e.to_string()).unwrap_or_default();
        assert!(e.contains("a pack's sizes are not its layout's"), "{e:?}");
        // the plan's layouts
        let refused = |why: &str, plan: String, layout: Layout| {
            let e = save_planned(&source, &dir.join("refused"), 2, 1 << 30, layout, &plan).err().map(|e| e.to_string()).unwrap_or_default();
            assert!(e.contains(why), "not refused for {why:?}: {e:?}");
        };
        refused("a layout other than i6, i5, i4, i3, i6r, i5r or i4r", plan("mma"), Layout::Int4);
        refused("a layout other than i6, i5, i4, i3, i6r, i5r or i4r", plan("i7"), Layout::Int4);
        refused("for a save in a block layout", plan("i6"), Layout::Tiered);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The tiny Qwen3 saved at the swift layout with its down matrices in the five-bit row layout (a linear's "layout": "i5r"): glyd.json names each down pack "i5r", its shape the transpose of its tensor's ([in, out]
    /// against [out, in]) and its words the row layout's tag; the pack's bytes are those pack_rows makes of the weight's transpose (the packer's, here made apart from the save), and verify decodes each on the CPU, back to
    /// the tensor's orientation, to the sha256 glyd.json holds; refused: a row pack named as a block layout, as the row layout of other bits, with its shape or its buffers cut, a row layout for several
    /// weights as one or for a weight whose outputs are not runs of 32, one in a save that is not in a block layout, and a row layout as the save's own.
    #[test]
    fn a_save_may_hold_a_pack_in_a_row_layout() {
        let dir = std::env::temp_dir().join(format!("glyd-gpu-rows-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let src = dir.join("src");
        tiny_qwen3(&src);
        let source = Source::find(src.to_str().unwrap()).unwrap();
        let config = json::parse(&std::fs::read_to_string(src.join("config.json")).unwrap()).unwrap();
        let puts = Plan::new(&Checkpoint::open(&src).unwrap(), &config, true, true).unwrap().puts;
        let plan = |down: &str, merged: &str| {
            let items: Vec<String> = puts
                .iter()
                .map(|p| match p {
                    Put::Linear { members, .. } => {
                        let own = if members.len() == 1 && members[0].ends_with(".down_proj") {
                            format!(", \"layout\": \"{down}\"")
                        } else if members.len() > 1 && !merged.is_empty() {
                            format!(", \"layout\": \"{merged}\"")
                        } else {
                            String::new()
                        };
                        format!("{{\"linear\": [{}]{own}}}", members.iter().map(|m| format!("\"{m}\"")).collect::<Vec<_>>().join(", "))
                    }
                    Put::Copy { name, from, .. } => format!("{{\"copy\": \"{name}\", \"from\": \"{from}\"}}"),
                    Put::Experts { .. } | Put::Codebook { .. } | Put::Signs { .. } => unreachable!(),
                })
                .collect();
            format!("{{\"puts\": [{}]}}", items.join(", "))
        };
        // (the names, the bits and the block layout each row layout is the rows of)
        assert_eq!([Layout::Int6Row, Layout::Int5Row, Layout::Int4Row].map(Layout::name), ["i6r", "i5r", "i4r"]);
        for (block, row, bits) in [(Layout::Int6, Layout::Int6Row, 6), (Layout::Int5, Layout::Int5Row, 5), (Layout::Int4, Layout::Int4Row, 4)] {
            assert_eq!((Layout::by_name(row.name()), block.row_of(), row.row_bits(), block.row_bits()), (Some(row), Some(row), Some(bits), None));
        }
        assert_eq!((Layout::Int3.row_of(), Layout::Tiered.row_of(), Layout::Int5Row.row_of()), (None, None, None));
        // (the entries a GPU's verify decodes on the CPU)
        let entry_of = |name: &str| json::parse(&format!("{{\"layout\": \"{name}\"}}")).unwrap();
        assert_eq!(
            ["i6r", "i5r", "i4r", "i6", "i5", "i4", "i3", "mma", "mma12", "i7r"].map(|n| is_rows(&entry_of(n))),
            [true, true, true, false, false, false, false, false, false, false]
        );
        assert!(!is_rows(&json::parse("{}").unwrap()));
        let out = dir.join("rows");
        save_planned(&source, &out, 2, 1 << 30, Layout::Int5, &plan("i5r", "")).unwrap();
        let m = json::parse(&std::fs::read_to_string(out.join("glyd.json")).unwrap()).unwrap();
        let s = |k: &str| m.get(k).and_then(Value::as_str);
        assert_eq!((s("format"), s("layout"), s("level")), (Some("glyd-v7"), Some("i5"), Some("swift")));
        let (saved, original) = (Checkpoint::open(&out).unwrap(), Checkpoint::open(&src).unwrap());
        let mut rows_packs = 0;
        for (name, e) in m.get("packs").and_then(Value::as_object).unwrap() {
            let (layout, tag) = if name.ends_with(".down_proj") {
                rows_packs += 1;
                ("i5r", pack::INT5_ROW_TAG)
            } else {
                ("i5", pack::INT5_TAG)
            };
            assert_eq!(e.get("layout").and_then(Value::as_str), Some(layout), "{name}");
            assert_eq!(u64s(e.get("tiers")).map(|t| (t[0] >> 24) as u32), Some(tag), "{name}");
            if layout == "i5r" {
                // the pack is the transpose [in, out], its tensor the weight [out, in]; its bytes the packer's own of the transpose
                assert_eq!(
                    (u64s(e.get("shape")), u64s(e.get("tensors").and_then(Value::as_array).map(|t| &t[0]).and_then(|t| t.get("shape")))),
                    (Some(vec![256, 128]), Some(vec![128, 256])),
                    "{name}"
                );
                let mut w = vec![0u16; 128 * 256];
                original.read_bf16(&format!("{name}.weight"), &mut w).unwrap();
                let want = pack::pack_rows(5, &pack::transpose(&w, 128, 256), 256, 128).unwrap();
                assert_eq!(saved.read(&format!("{name}.glyd_data")).unwrap(), want.data, "{name}: the codes");
                assert_eq!(saved.read(&format!("{name}.glyd_blocks")).unwrap(), want.blocks, "{name}: the scales");
                assert_eq!(saved.read(&format!("{name}.glyd_block_base")).unwrap(), vec![0u8; (256 * 128 / pack::STEP + 1) * 4], "{name}: the base, not read");
            }
        }
        assert_eq!(rows_packs, 2, "a down matrix a layer");
        let v = verify(&out, 2, &Decoder::Cpu).unwrap();
        assert_eq!((v.packed, v.hashed, v.unchecked), (15, 9, 0));
        // an entry that names the pack another layout than its bytes are in
        let text = std::fs::read_to_string(out.join("glyd.json")).unwrap();
        for (name, why) in
            [("i5", "a pack's sizes are not its layout's"), ("i4r", "a pack's sizes are not its layout's"), ("i6r", "a pack's sizes are not its layout's")]
        {
            std::fs::write(out.join("glyd.json"), text.replacen("\"layout\": \"i5r\"", &format!("\"layout\": \"{name}\""), 1)).unwrap();
            let e = verify(&out, 2, &Decoder::Cpu).err().map(|e| e.to_string()).unwrap_or_default();
            assert!(e.contains(why), "{name}: {e:?}");
        }
        // a layout no reader of this glyd knows is a name it does not read (an older glyd refuses the row layouts so)
        std::fs::write(out.join("glyd.json"), text.replacen("\"layout\": \"i5r\"", "\"layout\": \"i5x\"", 1)).unwrap();
        let e = verify(&out, 2, &Decoder::Cpu).err().map(|e| e.to_string()).unwrap_or_default();
        assert!(e.contains("a layout this glyd does not read"), "{e:?}");
        // its shape the weight's, not the transpose (the same bytes, read in rows of 256: the tensor's shape is then not the pack's), and one cut to a shape the bytes are not
        let reshaped = |shape: [u64; 2]| {
            let mut v = json::parse(&text).unwrap();
            let Value::Object(top) = &mut v else { unreachable!() };
            let Some((_, Value::Object(packs))) = top.iter_mut().find(|(k, _)| k == "packs") else { unreachable!() };
            let Some((_, Value::Object(entry))) = packs.iter_mut().find(|(k, _)| k.ends_with(".down_proj")) else { unreachable!() };
            entry.iter_mut().find(|(k, _)| k == "shape").unwrap().1 = shape_value(&shape);
            std::fs::write(out.join("glyd.json"), json::to_python(&v, 2)).unwrap();
            verify(&out, 2, &Decoder::Cpu).err().map(|e| e.to_string()).unwrap_or_default()
        };
        let e = reshaped([128, 256]);
        assert!(e.contains("is not its pack's"), "{e:?}");
        let e = reshaped([256, 96]);
        assert!(e.contains("its buffers are not its shape's"), "{e:?}");
        let e = reshaped([250, 128]);
        assert!(e.contains("not a row layout's shape"), "{e:?}");
        std::fs::write(out.join("glyd.json"), &text).unwrap();
        assert_eq!(verify(&out, 2, &Decoder::Cpu).unwrap().packed, 15);
        // the plan's row layouts
        let refused = |why: &str, plan: String, layout: Layout| {
            let e = save_planned(&source, &dir.join("refused"), 2, 1 << 30, layout, &plan).err().map(|e| e.to_string()).unwrap_or_default();
            assert!(e.contains(why), "not refused for {why:?}: {e:?}");
        };
        refused("not packable in a row layout", plan("i5r", "i5r"), Layout::Int5); // (q, k, v as one pack, gate and up as one)
        refused("for a save in a block layout", plan("i5r", ""), Layout::Tiered);
        refused("a row layout is a pack's own", plan("i5r", ""), Layout::Int5Row);
        // a weight whose outputs are not runs of 32 (the tiny model's norms are vectors, its q_norm 32 long: a matrix of 16 outputs)
        let odd = dir.join("odd");
        std::fs::create_dir_all(&odd).unwrap();
        std::fs::write(odd.join("config.json"), "{}").unwrap();
        let w: Vec<u16> = (0..16 * 64).map(|i| 0x3C00 + i as u16 % 64).collect();
        let mut ts = vec![Saved { name: "d.weight".into(), dtype: "BF16", shape: vec![16, 64], bytes: u16_bytes(&w), sha256: None }];
        write_safetensors(&odd.join("model.safetensors"), &mut ts).unwrap();
        let e = save_planned(
            &Source::find(odd.to_str().unwrap()).unwrap(),
            &dir.join("odd-out"),
            1,
            1 << 30,
            Layout::Int5,
            r#"{"puts": [{"linear": ["d"], "layout": "i5r"}]}"#,
        )
        .err()
        .map(|e| e.to_string())
        .unwrap_or_default();
        assert!(e.contains("not packable in a row layout"), "{e:?}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The tiny Qwen3 saved in each block layout from calibrated codes and scales (none a fit makes: any code, scales of both signs) for every linear but the embedding, and verified on the CPU: each calibrated
    /// matrix decodes to its codes times its scales, bit for bit (glyd.json's sha256 of it is that of the values the codes and scales make, here computed apart from the packer), the embedding is packed from its
    /// weights as the plain save packs it, glyd.json records the calibration; refused: codes the plan does not use, some of a merged group's alone, a shape or a dtype other than the matrix's, a code past q, a
    /// scale not finite, a layout that is not a block one, no calibration.json.
    #[test]
    fn calibrated_codes_are_saved_as_given() {
        let dir = std::env::temp_dir().join(format!("glyd-gpu-codes-{}", std::process::id()));
        let src = dir.join("src");
        tiny_qwen3(&src);
        let source = Source::find(src.to_str().unwrap()).unwrap();
        let config = json::parse(&std::fs::read_to_string(src.join("config.json")).unwrap()).unwrap();
        let puts = Plan::new(&Checkpoint::open(&src).unwrap(), &config, true, true).unwrap().puts;
        let plan = format!(
            "{{\"puts\": [{}]}}",
            puts.iter()
                .map(|p| match p {
                    Put::Linear { members, .. } => format!("{{\"linear\": [{}]}}", members.iter().map(|m| format!("\"{m}\"")).collect::<Vec<_>>().join(", ")),
                    Put::Copy { name, from, .. } => format!("{{\"copy\": \"{name}\", \"from\": \"{from}\"}}"),
                    Put::Experts { .. } | Put::Codebook { .. } | Put::Signs { .. } => unreachable!(),
                })
                .collect::<Vec<_>>()
                .join(", ")
        );
        let calibration = r#"{"method": "gptq-act", "data": [{"dataset": "Salesforce/wikitext", "revision": "b08601e0"}], "seqs": 512, "tokens": 2048}"#;
        let mut seed = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        let linears: Vec<(String, usize, usize)> = puts
            .iter()
            .filter_map(|p| match p {
                Put::Linear { from, .. } if from[0] != "model.embed_tokens" => Some(from),
                _ => None,
            })
            .flatten()
            .map(|m| {
                let (r, c) = dims(Checkpoint::open(&src).unwrap().get(&format!("{m}.weight")).unwrap().1);
                (m.clone(), r, c)
            })
            .collect();
        for (layout, name, q, level) in [(Layout::Int5, "i5", 15i8, "swift"), (Layout::Int6, "i6", 31, "kestrel")] {
            let value = |c: i8, s: u16| if q == 15 { pack::int5_value((c + 16) as u8, s) } else { pack::int6_value((c + 32) as u8, s) };
            let codes = dir.join(format!("codes-{name}"));
            std::fs::create_dir_all(&codes).unwrap();
            std::fs::write(codes.join("calibration.json"), calibration).unwrap();
            let (mut tensors, mut want) = (Vec::new(), HashMap::new());
            for (m, r, c) in &linears {
                let code: Vec<i8> = (0..r * c).map(|_| (next() % (2 * q as u64 + 1)) as i8 - q).collect();
                let scale: Vec<u16> = (0..r * c / 32).map(|_| next() as u16 & 0xBF7F | 0x3000).collect();
                let values: Vec<u16> = code.iter().zip(scale.iter().flat_map(|s| [s; 32])).map(|(&c, &s)| value(c, s)).collect();
                want.insert(format!("{m}.weight"), sha256_u16(&values));
                tensors.push(Saved {
                    name: format!("{m}.codes"),
                    dtype: "I8",
                    shape: vec![*r as u64, *c as u64],
                    bytes: code.iter().map(|&c| c as u8).collect(),
                    sha256: None,
                });
                tensors.push(Saved {
                    name: format!("{m}.scales"),
                    dtype: "BF16",
                    shape: vec![*r as u64, *c as u64 / 32],
                    bytes: u16_bytes(&scale),
                    sha256: None,
                });
            }
            write_safetensors(&codes.join("model.safetensors"), &mut tensors).unwrap();
            let (out, plain) = (dir.join(name), dir.join(format!("plain-{name}")));
            let s = save_calibrated(&source, &out, 2, 1 << 30, layout, &plan, &codes).unwrap();
            assert_eq!((s.checked, s.calibrated), (15, 14), "{name}");
            save_planned(&source, &plain, 2, 1 << 30, layout, &plan).unwrap();
            let m = json::parse(&std::fs::read_to_string(out.join("glyd.json")).unwrap()).unwrap();
            let sv = |k: &str| m.get(k).and_then(Value::as_str);
            assert_eq!((sv("format"), sv("layout"), sv("level")), (Some("glyd-v7"), Some(name), Some(level)));
            assert_eq!(m.get("calibration"), Some(&json::parse(calibration).unwrap()));
            assert!(!std::fs::read_to_string(plain.join("glyd.json")).unwrap().contains("calibration"));
            let v = verify(&out, 2, &Decoder::Cpu).unwrap();
            assert_eq!((v.packed, v.hashed, v.unchecked, v.skipped.len()), (15, 9, 0, 0), "{name}");
            let mut got = HashMap::new();
            for (_, e) in m.get("packs").unwrap().as_object().unwrap() {
                for t in e.get("tensors").and_then(Value::as_array).unwrap() {
                    got.insert(t.get("name").and_then(Value::as_str).unwrap().to_string(), t.get("sha256").and_then(Value::as_str).unwrap().to_string());
                }
            }
            for (n, h) in &want {
                assert_eq!(got.get(n), Some(h), "{name}: {n} does not decode to its codes times its scales");
            }
            assert_eq!(got.len(), want.len() + 1); // (and the embedding, from its weights)
            let (a, b, c) = (Checkpoint::open(&out).unwrap(), Checkpoint::open(&plain).unwrap(), "model.layers.0.self_attn.o_proj.glyd_blocks");
            assert_eq!(a.read("model.embed_tokens.glyd_blocks").unwrap(), b.read("model.embed_tokens.glyd_blocks").unwrap());
            assert_ne!(a.read(c).unwrap(), b.read(c).unwrap());
            // refused
            let refused = |why: &str, edit: &dyn Fn(&mut Vec<Saved>), layout: Layout| {
                let mut ts = Vec::new();
                for (n, t) in Checkpoint::open(&codes).unwrap().files[0].tensors.iter() {
                    ts.push(Saved {
                        name: n.clone(),
                        dtype: if n.ends_with(".codes") { "I8" } else { "BF16" },
                        shape: t.shape.clone(),
                        bytes: std::fs::read(&codes.join("model.safetensors")).unwrap()[t.offset as usize..(t.offset + t.bytes) as usize].to_vec(),
                        sha256: None,
                    });
                }
                edit(&mut ts);
                let bad = dir.join("bad-codes");
                let _ = std::fs::remove_dir_all(&bad);
                std::fs::create_dir_all(&bad).unwrap();
                write_safetensors(&bad.join("model.safetensors"), &mut ts).unwrap();
                std::fs::write(bad.join("calibration.json"), calibration).unwrap();
                let e = save_calibrated(&source, &dir.join("refused"), 2, 1 << 30, layout, &plan, &bad).err().map(|e| e.to_string()).unwrap_or_default();
                assert!(e.contains(why), "{name}: not refused for {why:?}: {e:?}");
            };
            refused("no codes", &|ts| ts.clear(), layout);
            refused(
                "is not a matrix the plan packs",
                &|ts| ts.push(Saved { name: "model.layers.0.self_attn.nope.codes".into(), dtype: "I8", shape: vec![1], bytes: vec![0], sha256: None }),
                layout,
            );
            refused(
                "is not a matrix the plan packs",
                &|ts| ts.push(Saved { name: "model.embed_tokens.stray".into(), dtype: "I8", shape: vec![1], bytes: vec![0], sha256: None }),
                layout,
            );
            refused("no model.layers.0.self_attn.k_proj.codes", &|ts| ts.retain(|t| t.name != "model.layers.0.self_attn.k_proj.codes"), layout);
            refused("no model.layers.0.self_attn.k_proj.scales", &|ts| ts.retain(|t| t.name != "model.layers.0.self_attn.k_proj.scales"), layout);
            refused(
                "where I8 [128, 128] is wanted",
                &|ts| {
                    let t = ts.iter_mut().find(|t| t.name == "model.layers.0.self_attn.o_proj.codes").unwrap();
                    t.shape = vec![64, 256];
                },
                layout,
            );
            refused(
                "where BF16 [128, 4] is wanted",
                &|ts| {
                    let t = ts.iter_mut().find(|t| t.name == "model.layers.0.self_attn.o_proj.scales").unwrap();
                    t.shape = vec![64, 8];
                },
                layout,
            );
            let not_block = if q == 15 { "not a five-bit layout's matrix" } else { "not a six-bit layout's matrix" };
            refused(not_block, &|ts| ts.iter_mut().find(|t| t.name == "model.layers.1.mlp.down_proj.codes").unwrap().bytes[9] = q as u8 + 1, layout);
            refused(
                not_block,
                &|ts| ts.iter_mut().find(|t| t.name == "model.layers.1.mlp.down_proj.scales").unwrap().bytes[..2].copy_from_slice(&0x7F80u16.to_le_bytes()),
                layout,
            );
            refused("calibrated codes are for the block layouts", &|_| {}, Layout::Tiered);
            std::fs::remove_file(codes.join("calibration.json")).unwrap();
            let e = save_calibrated(&source, &dir.join("refused"), 2, 1 << 30, layout, &plan, &codes).err().map(|e| e.to_string()).unwrap_or_default();
            assert!(e.contains("calibration.json"), "{name}: {e:?}");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A glyd-v5 save (an embedding's table packed as a Linear's matrix is, its entry marked "table", as format.py's
    /// save_pretrained(tables=True) writes it) verified on the CPU in either layout: its tensor against glyd.json's
    /// sha256; then refused: another sha256 for it.
    /// f32 made bf16 as torch rounds it: to nearest, ties to even, up to infinity past the largest; NaN 0x7FC0.
    #[test]
    fn bf16_from_f32_rounds_as_torch() {
        let cases: [(u32, u16); 9] = [
            (0x3F80_0000, 0x3F80),
            (0x3F80_8000, 0x3F80),
            (0x3F81_8000, 0x3F82),
            (0x3F80_8001, 0x3F81),
            (0x8000_0000, 0x8000),
            (0x7F80_0000, 0x7F80),
            (0x7F7F_FFFF, 0x7F80),
            (0x7FC0_0001, 0x7FC0),
            (0x0000_8000, 0x0000),
        ];
        let bytes: Vec<u8> = cases.iter().flat_map(|c| c.0.to_le_bytes()).collect();
        let got: Vec<u16> = bf16_from_f32(&bytes).chunks_exact(2).map(|h| u16::from_le_bytes([h[0], h[1]])).collect();
        assert_eq!(got, cases.iter().map(|c| c.1).collect::<Vec<_>>());
    }

    #[test]
    fn table_saves_are_read() {
        let dir = std::env::temp_dir().join(format!("glyd-gpu-table-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let (rows, cols) = (256, 128);
        let w = pack::tests::weights(rows * cols, 0.01, 23);
        for layout in [Layout::Tiered, Layout::Twelve] {
            let (mut ts, words, lay) = packed(layout, &mut w.clone(), rows, cols, "model.embed_tokens.glyd_", "model.embed_tokens", None).unwrap();
            write_safetensors(&dir.join("model.safetensors"), &mut ts).unwrap();
            let manifest = |sha: String| {
                let mut e = entry(lay, [rows, cols], &words, vec![("model.embed_tokens.weight".into(), vec![rows as u64, cols as u64], sha)], None);
                if let Value::Object(o) = &mut e {
                    o.push(("table".into(), Value::Bool(true)));
                }
                let source = Value::Object(vec![("repo".into(), Value::Null), ("revision".into(), Value::Null)]);
                let m = Value::Object(vec![
                    ("format".into(), Value::from("glyd-v5")),
                    ("glyd".into(), Value::from(env!("CARGO_PKG_VERSION"))),
                    ("source".into(), source),
                    ("layout".into(), Value::from(layout.name())),
                    ("packs".into(), Value::Object(vec![("model.embed_tokens".into(), e)])),
                    ("tensors".into(), Value::Object(vec![])),
                ]);
                std::fs::write(dir.join("glyd.json"), json::to_python(&m, 1)).unwrap();
            };
            manifest(sha256_u16(&w));
            let v = verify(&dir, 1, &Decoder::Cpu).unwrap();
            assert_eq!((v.packed, v.hashed, v.unchecked), (1, 0, 0), "{layout:?}");
            manifest(sha256_u16(&w[1..]));
            let e = verify(&dir, 1, &Decoder::Cpu).err().map(|e| e.to_string()).unwrap_or_default();
            assert!(e.contains("model.embed_tokens.weight decodes to other bytes"), "{layout:?}: {e:?}");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A tensor of a checkpoint as a `Saved` (its dtype one of the three these tests use).
    fn saved_of(c: &Checkpoint) -> Vec<Saved> {
        let dtype = |d: &str| ["BF16", "F32", "F8_E4M3"].into_iter().find(|x| *x == d).expect("a dtype of the tests");
        c.files
            .iter()
            .flat_map(|f| {
                f.tensors.iter().map(|(n, t)| Saved {
                    name: n.clone(),
                    dtype: dtype(&t.dtype),
                    shape: t.shape.clone(),
                    bytes: f.read(t).unwrap(),
                    sha256: None,
                })
            })
            .collect()
    }

    /// FP8 matrices (E4M3 with block scales of 128 x 128, as GLM-5.3-Flash holds its experts, MLPs and latent-attention projections) beside bf16 and F32 tensors, written to `dir`: a gate and an up of 192 x 160
    /// (a merged pack), a down of 64 x 192, every size short of a whole number of blocks (rows 192 and 64, columns 160 and 192), a bf16 matrix `m`, a bf16 norm `n` and an F32 parameter `p`. `dir16` holds the
    /// same checkpoint with each FP8 matrix replaced by the bf16 it dequantises to, worked out here a weight at a time. Returns the FP8 matrices' names.
    fn tiny_fp8(dir: &Path, dir16: &Path) -> Vec<String> {
        let config = r#"{"architectures": ["Tiny"], "quantization_config": {"quant_method": "fp8", "fmt": "e4m3", "activation_scheme": "dynamic", "weight_block_size": [128, 128]}}"#;
        for d in [dir, dir16] {
            std::fs::create_dir_all(d).unwrap();
            std::fs::write(d.join("config.json"), config).unwrap();
        }
        let mut seed = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        let (mut fp8, mut dequantised, mut names) = (Vec::new(), Vec::new(), Vec::new());
        for (name, rows, cols) in [("e.gate_proj", 192usize, 160usize), ("e.up_proj", 192, 160), ("e.down_proj", 64, 192)] {
            let [gr, gc] = crate::fp8::grid(rows, cols, [128, 128]);
            let q: Vec<u8> = (0..rows * cols)
                .map(|_| match (next() >> 24) as u8 {
                    0x7F => 0x7E,
                    0xFF => 0xFE,
                    b => b,
                })
                .collect();
            // (scales of a trained matrix's size: about 2^-10 to 2^-9)
            let scales: Vec<f32> = (0..gr * gc).map(|_| f32::from_bits(0x3A80_0000 + ((next() >> 40) as u32) % 0x0100_0000)).collect();
            let w: Vec<u16> = (0..rows * cols)
                .map(|i| {
                    let (r, c) = (i / cols, i % cols);
                    let h = bf16_from_f32(&(crate::fp8::E4M3[q[i] as usize] * scales[(r / 128) * gc + c / 128]).to_le_bytes());
                    u16::from_le_bytes([h[0], h[1]])
                })
                .collect();
            let shape = vec![rows as u64, cols as u64];
            fp8.push(Saved { name: format!("{name}.weight"), dtype: "F8_E4M3", shape: shape.clone(), bytes: q, sha256: None });
            fp8.push(Saved {
                name: format!("{name}.weight_scale_inv"),
                dtype: "F32",
                shape: vec![gr as u64, gc as u64],
                bytes: le_bytes(&scales, f32::to_le_bytes),
                sha256: None,
            });
            dequantised.push(Saved { name: format!("{name}.weight"), dtype: "BF16", shape, bytes: u16_bytes(&w), sha256: None });
            names.push(format!("{name}.weight"));
        }
        let mut same = Vec::new();
        for (name, dtype, shape) in [("m.weight", "BF16", vec![128u64, 96]), ("n.weight", "BF16", vec![96]), ("p", "F32", vec![64])] {
            let n: u64 = shape.iter().product();
            let bytes = if dtype == "F32" {
                le_bytes(&(0..n).map(|_| f32::from_bits(0x3F00_0000 | (next() >> 41) as u32)).collect::<Vec<f32>>(), f32::to_le_bytes)
            } else {
                u16_bytes(
                    &(0..n)
                        .map(|_| {
                            let x = next();
                            ((x >> 20) as u16 & 0x807F) | ((116 + (x >> 40) % 14) as u16) << 7
                        })
                        .collect::<Vec<u16>>(),
                )
            };
            same.push(Saved { name: name.into(), dtype, shape, bytes, sha256: None });
        }
        let copy = |ts: &[Saved]| {
            ts.iter().map(|t| Saved { name: t.name.clone(), dtype: t.dtype, shape: t.shape.clone(), bytes: t.bytes.clone(), sha256: None }).collect::<Vec<_>>()
        };
        let (mut a, mut b) = (fp8, dequantised);
        a.extend(copy(&same));
        b.extend(same);
        write_safetensors(&dir.join("model.safetensors"), &mut a).unwrap();
        write_safetensors(&dir16.join("model.safetensors"), &mut b).unwrap();
        names
    }

    /// A checkpoint with FP8 matrices saved in each of the four block layouts by a plan (a gate and an up as one pack, a down, a bf16 matrix, a bf16 norm and an F32 parameter copied as they are) holds exactly
    /// what the same checkpoint with those matrices dequantised to bf16 first holds: every tensor of the two saves the same bytes, every pack's entry in glyd.json the same, and a verify of the first decodes
    /// each pack to the sha256 of the dequantised weights. The F32 parameter is saved F32, byte for byte.
    #[test]
    fn fp8_matrices_save_as_the_bf16_they_dequantise_to() {
        let dir = std::env::temp_dir().join(format!("glyd-gpu-fp8-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let (src, src16) = (dir.join("src"), dir.join("src16"));
        let names = tiny_fp8(&src, &src16);
        let plan =
            r#"{"puts": [{"linear": ["e.gate_proj", "e.up_proj"]}, {"linear": ["e.down_proj"]}, {"linear": ["m"]}, {"copy": "n.weight"}, {"copy": "p"}]}"#;
        let (source, source16) = (Source::find(src.to_str().unwrap()).unwrap(), Source::find(src16.to_str().unwrap()).unwrap());
        let (c, c16) = (Checkpoint::open(&src).unwrap(), Checkpoint::open(&src16).unwrap());
        assert_eq!((c.fp8_block, c16.fp8_block), (Some([128, 128]), Some([128, 128])));
        // the matrix as the save reads it: the shape and the bytes of its bf16 words (a bf16 tensor's own info), and the words the dequantised checkpoint has
        for n in &names {
            let (t, t16) = (c.matrix(n).unwrap(), c16.matrix(n).unwrap());
            assert_eq!((t.dtype.as_str(), &t.shape, t.bytes), ("BF16", &t16.shape, t16.bytes), "{n}");
            let (mut a, mut b) = (vec![0u16; (t.bytes / 2) as usize], vec![0u16; (t.bytes / 2) as usize]);
            c.read_bf16(n, &mut a).unwrap();
            c16.read_bf16(n, &mut b).unwrap();
            assert!(a == b, "{n}");
        }
        for layout in [Layout::Int6, Layout::Int5, Layout::Int4, Layout::Int3] {
            let (out, out16) = (dir.join(format!("out-{}", layout.name())), dir.join(format!("out16-{}", layout.name())));
            let s = save_planned(&source, &out, 2, 1 << 30, layout, plan).unwrap();
            let s16 = save_planned(&source16, &out16, 2, 1 << 30, layout, plan).unwrap();
            // (the bytes of bf16 read are those of the bf16 matrices, whichever the source holds)
            assert_eq!((s.checked, s.hashed, s.bytes), (s16.checked, s16.hashed, s16.bytes), "{layout:?}");
            assert_eq!((s.checked, s.hashed), (4, 2));
            let (a, b) = (Checkpoint::open(&out).unwrap(), Checkpoint::open(&out16).unwrap());
            let (mut ours, mut theirs): (Vec<&str>, Vec<&str>) = (a.names().collect(), b.names().collect());
            ours.sort();
            theirs.sort();
            assert_eq!(ours, theirs, "{layout:?}");
            assert!(ours.iter().all(|n| a.read(n).unwrap() == b.read(n).unwrap()), "{layout:?}");
            let (ma, mb) = (
                json::parse(&std::fs::read_to_string(out.join("glyd.json")).unwrap()).unwrap(),
                json::parse(&std::fs::read_to_string(out16.join("glyd.json")).unwrap()).unwrap(),
            );
            assert!(ma.get("packs").is_some() && ma.get("packs") == mb.get("packs") && ma.get("tensors") == mb.get("tensors"), "{layout:?}");
            // the gate and the up are one pack under the gate's name; the F32 parameter is F32 in the save and is not a pack; no scale is saved
            assert!(ma.get("packs").unwrap().get("e.gate_proj").is_some() && ma.get("packs").unwrap().get("e.up_proj").is_none());
            assert_eq!(a.get("p").map(|(_, t)| (t.dtype.clone(), t.shape.clone())), Some(("F32".to_string(), vec![64])));
            assert!(a.read("p").unwrap() == c.read("p").unwrap());
            assert!(a.get("e.down_proj.weight").is_none() && a.get("e.down_proj.glyd_data").is_some() && a.names().all(|n| !n.contains("scale_inv")));
            let v = verify(&out, 2, &Decoder::Cpu).unwrap();
            assert_eq!((v.packed, v.hashed, v.unchecked), (4, 2, 0), "{layout:?}");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// What an FP8 matrix needs, and what is refused: its `NAME_scale_inv` (F32, the grid its blocks make), config.json's block size; another dtype than BF16 or FP8 is no matrix; a plan that copies an FP8 tensor
    /// as it is; the lossless families, which take a bf16 checkpoint alone.
    #[test]
    fn fp8_matrices_need_their_scales() {
        let dir = std::env::temp_dir().join(format!("glyd-gpu-fp8-refused-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let src = dir.join("src");
        tiny_fp8(&src, &dir.join("src16"));
        let base = saved_of(&Checkpoint::open(&src).unwrap());
        // the checkpoint with `edit` applied to its tensors and `config` its config.json, in a directory of its own
        let variant = |name: &str, config: &str, edit: &dyn Fn(&mut Vec<Saved>)| -> Checkpoint {
            let d = dir.join(name);
            std::fs::create_dir_all(&d).unwrap();
            std::fs::write(d.join("config.json"), config).unwrap();
            let mut ts: Vec<Saved> =
                base.iter().map(|t| Saved { name: t.name.clone(), dtype: t.dtype, shape: t.shape.clone(), bytes: t.bytes.clone(), sha256: None }).collect();
            edit(&mut ts);
            write_safetensors(&d.join("model.safetensors"), &mut ts).unwrap();
            Checkpoint::open(&d).unwrap()
        };
        let config = |block: &str| format!(r#"{{"architectures": ["Tiny"], "quantization_config": {{"quant_method": "fp8", "weight_block_size": {block}}}}}"#);
        let wanted = |c: &Checkpoint, why: &str| {
            let e = c.matrix("e.up_proj.weight").err().map(|e| e.to_string()).unwrap_or_default();
            assert!(e.contains(why), "{e:?} does not say {why:?}");
            let e = c.read_bf16("e.up_proj.weight", &mut vec![0u16; 192 * 160]).err().map(|e| e.to_string()).unwrap_or_default();
            assert!(e.contains(why), "{e:?} does not say {why:?}");
        };
        let scale = "e.up_proj.weight_scale_inv";
        // the checkpoint as it is is read; its scales gone, of another dtype, of another grid; no block size in config.json, a block size other than the scales' grid
        let ok = variant("ok", &config("[128, 128]"), &|_| {});
        ok.matrix("e.up_proj.weight").unwrap();
        wanted(&variant("noscale", &config("[128, 128]"), &|ts| ts.retain(|t| t.name != scale)), "there is no e.up_proj.weight_scale_inv, its block scales");
        wanted(
            &variant("dtype", &config("[128, 128]"), &|ts| {
                ts.iter_mut().filter(|t| t.name == scale).for_each(|t| {
                    t.dtype = "BF16";
                    t.bytes.truncate(8);
                })
            }),
            "are BF16 [2, 2], where F32 [2, 2] is wanted for blocks of 128 x 128",
        );
        wanted(
            &variant("grid", &config("[128, 128]"), &|ts| ts.iter_mut().filter(|t| t.name == scale).for_each(|t| t.shape = vec![4, 1])),
            "are F32 [4, 1], where F32 [2, 2] is wanted",
        );
        wanted(&variant("noblock", r#"{"architectures": ["Tiny"]}"#, &|_| {}), "gives no quantization_config.weight_block_size");
        wanted(&variant("badblock", &config("[128]"), &|_| {}), "gives no quantization_config.weight_block_size");
        wanted(&variant("block", &config("[64, 32]"), &|_| {}), "where F32 [3, 5] is wanted for blocks of 64 x 32");
        // another dtype: no matrix; a rank of 1: no FP8 matrix
        let e = ok.matrix("p").err().map(|e| e.to_string()).unwrap_or_default();
        assert!(e.contains("p: F32, where a bf16 matrix, or an FP8 one with its block scales, is wanted"), "{e:?}");
        let flat = variant("flat", &config("[128, 128]"), &|ts| ts.iter_mut().filter(|t| t.name == "e.up_proj.weight").for_each(|t| t.shape = vec![192 * 160]));
        assert!(flat
            .matrix("e.up_proj.weight")
            .err()
            .map(|e| e.to_string())
            .unwrap_or_default()
            .contains("F8_E4M3 of shape [30720], where a matrix is wanted"));
        // a plan: a linear needs the scales; a copy of an FP8 tensor is refused, and so is a copy of a scale, which is F32 and may be saved but is no tensor of a plan's
        let source = Source::find(dir.join("noscale").to_str().unwrap()).unwrap();
        let e = save_planned(&source, &dir.join("o1"), 1, 1 << 30, Layout::Int6, r#"{"puts": [{"linear": ["e.up_proj"]}]}"#)
            .err()
            .map(|e| e.to_string())
            .unwrap_or_default();
        assert!(e.contains("there is no e.up_proj.weight_scale_inv"), "{e:?}");
        let source = Source::find(src.to_str().unwrap()).unwrap();
        let e = save_planned(&source, &dir.join("o2"), 1, 1 << 30, Layout::Int6, r#"{"puts": [{"copy": "e.up_proj.weight"}]}"#)
            .err()
            .map(|e| e.to_string())
            .unwrap_or_default();
        assert!(e.contains("e.up_proj.weight: F8_E4M3, where glyd packs a bf16 checkpoint"), "{e:?}");
        // the lossless families take a bf16 checkpoint alone, an FP8 tensor in it or not
        std::fs::write(src.join("config.json"), r#"{"architectures": ["Qwen3ForCausalLM"], "num_hidden_layers": 0, "quantization_config": {"quant_method": "fp8", "weight_block_size": [128, 128]}}"#).unwrap();
        let e = save(&source, &dir.join("o3"), 1, 1 << 30, true, Layout::Tiered).err().map(|e| e.to_string()).unwrap_or_default();
        assert!(e.contains("where glyd packs a bf16 checkpoint"), "{e:?}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A tiny Qwen3 checkpoint (random bf16 weights, two layers, the output layer tied) written to `src`.
    fn tiny_qwen3(src: &Path) {
        std::fs::create_dir_all(&src).unwrap();
        std::fs::write(src.join("config.json"), r#"{"architectures": ["Qwen3ForCausalLM"], "num_hidden_layers": 2, "tie_word_embeddings": true}"#).unwrap();
        std::fs::write(src.join("tokenizer.json"), "{}").unwrap();
        let mut seed = 7u64;
        let mut t = |name: &str, shape: &[u64]| {
            let n: u64 = shape.iter().product();
            let w: Vec<u16> = (0..n)
                .map(|_| {
                    seed ^= seed << 13;
                    seed ^= seed >> 7;
                    seed ^= seed << 17;
                    ((seed >> 20) as u16 & 0x807F) | ((116 + (seed >> 40) % 14) as u16) << 7
                })
                .collect();
            Saved { name: name.into(), dtype: "BF16", shape: shape.to_vec(), bytes: u16_bytes(&w), sha256: None }
        };
        let mut ts = vec![t("model.embed_tokens.weight", &[256, 128]), t("model.norm.weight", &[128])];
        for i in 0..2 {
            for (n, s) in [
                ("self_attn.q_proj.weight", &[128, 128][..]),
                ("self_attn.k_proj.weight", &[64, 128]),
                ("self_attn.v_proj.weight", &[64, 128]),
                ("self_attn.o_proj.weight", &[128, 128]),
                ("self_attn.q_norm.weight", &[32]),
                ("self_attn.k_norm.weight", &[32]),
                ("mlp.gate_proj.weight", &[256, 128]),
                ("mlp.up_proj.weight", &[256, 128]),
                ("mlp.down_proj.weight", &[128, 256]),
                ("input_layernorm.weight", &[128]),
                ("post_attention_layernorm.weight", &[128]),
            ] {
                ts.push(t(&format!("model.layers.{i}.{n}"), s));
            }
        }
        write_safetensors(&src.join("model.safetensors"), &mut ts).unwrap();
    }

    /// A tiny Qwen3 checkpoint (random bf16 weights, two layers, the output layer tied), saved in shards of 64 KB
    /// and verified: glyd.json's packs in save_pretrained's order (a layer's o_proj, then q, k, v as one; down_proj,
    /// then gate, up), every tensor's sha256 the checkpoint's; then refused: one byte of a pack changed, one of a
    /// tensor saved as it is, bytes appended to a shard, a merged pack's member renamed (to another's, and by one
    /// letter), a member of one pack another's, a tensor neither packed nor hashed, glyd.json's map damaged or gone
    /// where its glyd says it has one, a member's rows past any matrix's; and the same save in the 12-bit layout.
    #[test]
    fn save_and_verify_a_tiny_qwen3() {
        let dir = std::env::temp_dir().join(format!("glyd-gpu-test-{}", std::process::id()));
        let (src, out) = (dir.join("src"), dir.join("out"));
        tiny_qwen3(&src);
        let source = Source::find(src.to_str().unwrap()).unwrap();
        let s = save(&source, &out, 3, 64 << 10, true, Layout::Tiered).unwrap();
        // the family's plan written as JSON (as huggingface/plan.py writes one) saves the same files, byte for byte
        let config = json::parse(&std::fs::read_to_string(src.join("config.json")).unwrap()).unwrap();
        let puts: Vec<String> = Plan::new(&Checkpoint::open(&src).unwrap(), &config, true, false)
            .unwrap()
            .puts
            .iter()
            .map(|p| match p {
                Put::Linear { members, .. } => format!("{{\"linear\": [{}]}}", members.iter().map(|m| format!("\"{m}\"")).collect::<Vec<_>>().join(", ")),
                Put::Copy { name, from, .. } => format!("{{\"copy\": \"{name}\", \"from\": \"{from}\"}}"),
                Put::Experts { .. } | Put::Codebook { .. } | Put::Signs { .. } => unreachable!(),
            })
            .collect();
        let planned = dir.join("planned");
        save_planned(&source, &planned, 2, 64 << 10, Layout::Tiered, &format!("{{\"puts\": [{}]}}", puts.join(", "))).unwrap();
        let mut names: Vec<_> = std::fs::read_dir(&out).unwrap().map(|e| e.unwrap().file_name()).collect();
        names.sort();
        assert_eq!(names, {
            let mut n: Vec<_> = std::fs::read_dir(&planned).unwrap().map(|e| e.unwrap().file_name()).collect();
            n.sort();
            n
        });
        for n in &names {
            assert!(std::fs::read(out.join(n)).unwrap() == std::fs::read(planned.join(n)).unwrap(), "{n:?} differs");
        }
        for (plan, why) in [
            ("{\"puts\": [{\"copy\": \"model.nope\"}]}", "model.nope is not in"),
            ("{\"puts\": [{\"linear\": [\"model.layers.0.self_attn.nope\"]}]}", "nope.weight is not in"),
            ("{\"puts\": [{\"linear\": [\"model.layers.0.self_attn.q_norm\"]}]}", "q_norm is not packable"),
            ("{\"puts\": [{\"copy\": \"model.norm.weight\", \"cast\": \"BF16\"}]}", "where a cast to BF16 takes F32"),
            ("{\"puts\": [{\"linear\": [\"model.layers.0.mlp.gate_proj\", \"model.layers.0.mlp.down_proj\"]}]}", "not packable as one matrix"),
            ("{\"puts\": [{\"x\": 1}]}", "neither a linear, a codebook matrix, signs nor a copy"),
        ] {
            let e = save_planned(&source, &dir.join("refused"), 2, 64 << 10, Layout::Tiered, plan).err().map(|e| e.to_string()).unwrap_or_default();
            assert!(e.contains(why), "not refused for {why:?}: {e:?}");
        }
        assert_eq!((s.checked, s.hashed, s.shards > 1), (14, 10, true));
        let m = json::parse(&std::fs::read_to_string(out.join("glyd.json")).unwrap()).unwrap();
        let packs: Vec<&str> = m.get("packs").unwrap().as_object().unwrap().iter().map(|(k, _)| k.as_str()).collect();
        assert_eq!(
            &packs[..4],
            &["model.layers.0.self_attn.o_proj", "model.layers.0.self_attn.q_proj", "model.layers.0.mlp.down_proj", "model.layers.0.mlp.gate_proj"]
        );
        assert_eq!(m.get("format").and_then(Value::as_str), Some("glyd-v4"));
        let hashed: Vec<&str> = m.get("tensors").unwrap().as_object().unwrap().iter().map(|(k, _)| k.as_str()).collect();
        assert_eq!(&hashed[..3], &["model.embed_tokens.weight", "model.layers.0.self_attn.q_norm.weight", "model.layers.0.self_attn.k_norm.weight"]);
        assert!(out.join("model.safetensors.index.json").exists() && out.join("tokenizer.json").exists());
        let v = verify(&out, 2, &Decoder::Cpu).unwrap();
        assert_eq!((v.packed, v.hashed, v.unchecked), (14, 10, 0));
        let fresh = |to: &Path| {
            let _ = std::fs::remove_dir_all(to);
            std::fs::create_dir_all(to).unwrap();
            for e in std::fs::read_dir(&out).unwrap() {
                let p = e.unwrap().path();
                std::fs::copy(&p, to.join(p.file_name().unwrap())).unwrap();
            }
            Checkpoint::open(to).unwrap()
        };
        let flip = |to: &Path, name: &str| {
            let c = fresh(to);
            let (f, info) = c.get(name).unwrap();
            let mut bytes = std::fs::read(&f.path).unwrap();
            bytes[info.offset as usize + 5] ^= 0x10;
            std::fs::write(&f.path, bytes).unwrap();
        };
        let bad = dir.join("bad");
        let refused = |why: &str| {
            let e = verify(&bad, 2, &Decoder::Cpu).err().map(|e| e.to_string()).unwrap_or_default();
            assert!(e.contains(why), "not refused for {why:?}: {e:?}");
        };
        flip(&bad, "model.layers.1.mlp.down_proj.glyd_data");
        refused("down_proj.weight decodes to other bytes");
        flip(&bad, "model.layers.1.post_attention_layernorm.weight");
        refused("post_attention_layernorm.weight is other bytes");
        let c = fresh(&bad);
        std::fs::OpenOptions::new().append(true).open(&c.files[0].path).unwrap().write_all(&[0; 8]).unwrap();
        refused("8 bytes past its last tensor");
        for (from, to, why) in [
            ("self_attn.k_proj.weight", "self_attn.v_proj.weight", "self_attn.q_proj's tensors are not its own"),
            ("self_attn.k_proj.weight", "self_attn.k_prok.weight", "self_attn.q_proj's tensors are not its own"),
            ("self_attn.o_proj.weight", "self_attn.q_proj.weight", "self_attn.o_proj's tensors are not its own"),
        ] {
            fresh(&bad);
            let text =
                std::fs::read_to_string(bad.join("glyd.json")).unwrap().replacen(&format!("\"model.layers.0.{from}\""), &format!("\"model.layers.0.{to}\""), 1);
            std::fs::write(bad.join("glyd.json"), text).unwrap();
            refused(why);
        }
        // a member of one pack another's: down_proj's pack renamed up_proj's in glyd.json (its tensor then its own)
        fresh(&bad);
        let text = std::fs::read_to_string(bad.join("glyd.json"))
            .unwrap()
            .replace("layers.0.mlp.down_proj\"", "layers.0.mlp.up_proj\"")
            .replace("layers.0.mlp.down_proj.weight\"", "layers.0.mlp.up_proj.weight\"");
        std::fs::write(bad.join("glyd.json"), text).unwrap();
        refused("a tensor held by two packs, or twice by one");
        fresh(&bad);
        let text = std::fs::read_to_string(bad.join("glyd.json")).unwrap().replace("\"model.norm.weight\"", "\"model.norm.weight2\"");
        std::fs::write(bad.join("glyd.json"), text).unwrap();
        refused("model.norm.weight: in the safetensors, but glyd.json neither packs it");
        // glyd.json's map damaged or gone: its key renamed "densors" (one bit) in this glyd's save or an older one's, its
        // value not an object, gone from a save of glyd 0.25 (and its glyd not a version): each refused; gone from a
        // save of glyd 0.24, whose saves have none, its tensors unchecked. A key of a newer glyd's save: skipped, the
        // rest checked (a byte of a tensor saved as it is flipped: refused)
        type Top = Vec<(String, Value)>; // glyd.json's entries
        let edit = |f: &dyn Fn(&mut Top)| {
            let mut m = json::parse(&std::fs::read_to_string(bad.join("glyd.json")).unwrap()).unwrap();
            if let Value::Object(top) = &mut m {
                f(top);
            }
            std::fs::write(bad.join("glyd.json"), json::to_python(&m, 1)).unwrap();
        };
        let set = |top: &mut Top, k: &str, v: Value| top.iter_mut().find(|(n, _)| n == k).unwrap().1 = v;
        let densors = |top: &mut Top| top.iter_mut().find(|(k, _)| k == "tensors").unwrap().0 = "densors".into();
        assert_eq!(m.get("glyd").and_then(Value::as_str), Some(env!("CARGO_PKG_VERSION"))); // (this glyd's save)
        fresh(&bad);
        edit(&densors);
        refused("\"densors\", a key glyd.json does not have");
        fresh(&bad);
        edit(&|top| {
            densors(top);
            set(top, "glyd", Value::from("0.21.0"));
        });
        refused("\"densors\", a key glyd.json does not have");
        fresh(&bad);
        edit(&|top| set(top, "tensors", Value::Null));
        refused("\"tensors\" is not an object of sha256");
        fresh(&bad);
        edit(&|top| {
            top.retain(|(k, _)| k != "tensors");
            set(top, "glyd", Value::from("0.25.0"));
        });
        refused("no sha256 for the tensors saved as they are, which a glyd-v4 of glyd 0.25.0 has");
        for v in ["0.x", "0/25.0"] {
            fresh(&bad);
            edit(&|top| set(top, "glyd", Value::from(v)));
            refused(&format!("glyd {v:?}, not a version"));
        }
        fresh(&bad);
        edit(&|top| {
            top.retain(|(k, _)| k != "tensors");
            set(top, "glyd", Value::from("0.24.0"));
        });
        refused("which a glyd-v4 of glyd 0.24.0 has"); // (a save of glyd 0.24 without the map is a glyd-v1's: legacy_saves_are_read)
        let newer = |top: &mut Top| {
            set(top, "glyd", Value::from("99.0.0"));
            top.push(("future".into(), Value::Object(vec![("x".into(), Value::from(1u64))])));
        };
        fresh(&bad);
        edit(&newer);
        let v = verify(&bad, 2, &Decoder::Cpu).unwrap();
        assert_eq!((v.packed, v.hashed, v.unchecked, v.skipped), (14, 10, 0, vec!["future".to_string()]));
        flip(&bad, "model.layers.1.post_attention_layernorm.weight");
        edit(&newer);
        refused("post_attention_layernorm.weight is other bytes");
        // k_proj's rows (a merged pack's second member) past any matrix's: no arithmetic on glyd.json's numbers overflows
        fresh(&bad);
        let text = std::fs::read_to_string(bad.join("glyd.json")).unwrap().replacen("\"shape\": [\n      64,", "\"shape\": [\n      18446744073709551615,", 1);
        assert!(text.contains("18446744073709551615"));
        std::fs::write(bad.join("glyd.json"), text).unwrap();
        refused("is not its pack's");
        // the 12-bit layout: glyd-v3, a pack's base hb (an int, in the place of the tiers) and its data, exc and
        // exc_base; then refused: a pack with no hb (as saved before split byte, with its four words "sym"), and one
        // whose hb is past 120
        let out12 = dir.join("out12");
        assert_eq!(save(&source, &out12, 2, 1 << 30, true, Layout::Twelve).unwrap().checked, 14);
        let text12 = std::fs::read_to_string(out12.join("glyd.json")).unwrap();
        let m = json::parse(&text12).unwrap();
        assert_eq!((m.get("format").and_then(Value::as_str), m.get("layout").and_then(Value::as_str)), (Some("glyd-v3"), Some("mma12")));
        let o = m.get("packs").unwrap().get("model.layers.1.self_attn.o_proj").unwrap();
        let keys: Vec<&str> = o.as_object().unwrap().iter().map(|(k, _)| k.as_str()).collect();
        let hb = o.get("hb").and_then(Value::as_u64).unwrap();
        assert!(keys == ["layout", "shape", "hb", "tensors"] && hb <= 120 && text12.contains(&format!("\"hb\": {hb},\n")), "{keys:?} {hb}");
        assert!(Checkpoint::open(&out12).unwrap().get("model.layers.1.self_attn.o_proj.glyd_exc_base").is_some());
        assert_eq!(verify(&out12, 2, &Decoder::Cpu).unwrap().packed, 14);
        let bad12 = dir.join("bad12");
        for (to, why) in [
            ("\"sym\": [\n    1,\n    2,\n    3,\n    4\n   ]".to_string(), "a pack an unreleased glyd saved"),
            ("\"hb\": 121".to_string(), "its hb is not an int 0-120"),
        ] {
            let _ = std::fs::remove_dir_all(&bad12);
            std::fs::create_dir_all(&bad12).unwrap();
            for e in std::fs::read_dir(&out12).unwrap() {
                let p = e.unwrap().path();
                std::fs::copy(&p, bad12.join(p.file_name().unwrap())).unwrap();
            }
            std::fs::write(bad12.join("glyd.json"), text12.replacen(&format!("\"hb\": {hb}"), &to, 1)).unwrap();
            let e = verify(&bad12, 2, &Decoder::Cpu).err().map(|e| e.to_string()).unwrap_or_default();
            assert!(e.contains(why), "not refused for {why:?}: {e:?}");
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// A tiny model saved at the Hummingbird level from a synthetic calibration export (nemotron_hbr.py's format; mixed 3- and 2-bit experts): each expert's up and down matrices are packs in c3r or c2r (tag, shape and words in glyd.json,
    /// the down matrix's tensor the transpose of its pack and its entry "transposed"), their three buffers the export's bytes as they are (data, blocks, codebook), the signs a tensor of the layer, the dense linear the six-bit block layout; the
    /// directory is glyd-v7 at layout `hb`, level `hummingbird`, with the export's meta.json recorded; `verify` decodes every pack on the CPU to the sha256 glyd.json holds; refused: the Hummingbird level without an export or with
    /// a plan that holds none of its matrices, an export for another level, a layer or an expert the export does not have, a tensor of the source of another shape than the export's matrix, a codebook layout as a save's own
    /// or a linear's, a pack whose bytes are changed.
    #[test]
    fn a_save_at_the_hummingbird_level_from_an_export() {
        let dir = std::env::temp_dir().join(format!("glyd-gpu-hb-save-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let (rows, cols, experts) = (64usize, 128usize, 2usize);
        let exp = dir.join("export");
        let bits = |_l: usize, e: usize| if e == 0 { 3 } else { 2 };
        let held = crate::hbexport::tests::synthetic(&exp, 1, experts, rows, cols, &bits, 77);
        // the source: a dense linear, the experts' bf16 matrices (not read: the export replaces them), a norm
        let src = dir.join("src");
        std::fs::create_dir_all(&src).unwrap();
        std::fs::write(src.join("config.json"), "{}").unwrap();
        let mut x = 99u64;
        let mut bf = |n: usize| -> Vec<u8> {
            (0..n)
                .flat_map(|_| {
                    {
                        x ^= x << 13;
                        x ^= x >> 7;
                        x ^= x << 17;
                        ((((x >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.1).to_bits() >> 16) as u16
                    }
                    .to_le_bytes()
                })
                .collect()
        };
        let m = "m.layers.1.mixer";
        let t = |name: String, shape: Vec<u64>, bytes: Vec<u8>| Saved { name, dtype: "BF16", shape, bytes, sha256: None };
        let mut ts = vec![
            t("m.fc.weight".into(), vec![64, 128], bf(64 * 128)),
            t(format!("{m}.experts.0.up_proj.weight"), vec![rows as u64, cols as u64], bf(rows * cols)),
            t(format!("{m}.experts.0.down_proj.weight"), vec![cols as u64, rows as u64], bf(rows * cols)),
            t(format!("{m}.experts.1.up_proj.weight"), vec![rows as u64, cols as u64], bf(rows * cols)),
            t(format!("{m}.experts.1.down_proj.weight"), vec![cols as u64, rows as u64], bf(rows * cols)),
            t("m.norm.weight".into(), vec![64], bf(64)),
        ];
        write_safetensors(&src.join("model.safetensors"), &mut ts).unwrap();
        let source = Source::find(src.to_str().unwrap()).unwrap();
        let plan = |extra: &str| {
            let mut puts = vec!["{\"linear\": [\"m.fc\"], \"layout\": \"i6\"}".to_string()];
            for e in 0..experts {
                puts.push(format!("{{\"codebook\": \"{m}.experts.{e}.up_proj\", \"layer\": 0, \"expert\": {e}, \"matrix\": \"up\"}}"));
                puts.push(format!("{{\"codebook\": \"{m}.experts.{e}.down_proj\", \"layer\": 0, \"expert\": {e}, \"matrix\": \"downT\"}}"));
            }
            puts.push(format!("{{\"signs\": \"{m}.glyd_rot_signs\", \"layer\": 0}}"));
            puts.push("{\"copy\": \"m.norm.weight\"}".to_string());
            puts.push(extra.to_string());
            format!("{{\"puts\": [{}]}}", puts.into_iter().filter(|p| !p.is_empty()).collect::<Vec<_>>().join(", "))
        };
        let out = dir.join("out");
        let s = save_hummingbird(&source, &out, 2, 1 << 30, &plan(""), &exp).unwrap();
        assert_eq!((s.checked, s.calibrated, s.hashed), (5, 4, 2));
        let j = json::parse(&std::fs::read_to_string(out.join("glyd.json")).unwrap()).unwrap();
        let s = |k: &str| j.get(k).and_then(Value::as_str);
        assert_eq!((s("format"), s("layout"), s("level")), (Some("glyd-v7"), Some("hb"), Some("hummingbird")));
        let cal = j.get("calibration").unwrap();
        assert_eq!(
            (
                cal.get("method").and_then(Value::as_str),
                cal.get("experts_at_3_bits").and_then(Value::as_u64),
                cal.get("experts_at_2_bits").and_then(Value::as_u64)
            ),
            (Some("hummingbird"), Some(1), Some(1))
        );
        assert!(cal.get("export").unwrap().get("layers").and_then(Value::as_array).unwrap()[0].get("signs_u8").is_none());
        let saved = Checkpoint::open(&out).unwrap();
        let packs = j.get("packs").unwrap();
        for e in 0..experts {
            for (name, down, mat) in [("up_proj", false, 0), ("down_proj", true, 1)] {
                let key = format!("{m}.experts.{e}.{name}");
                let ent = packs.get(&key).unwrap();
                let (lay, tag) = if e == 0 { ("c3r", pack::CB3_ROW_TAG) } else { ("c2r", pack::CB2_ROW_TAG) };
                assert_eq!(ent.get("layout").and_then(Value::as_str), Some(lay), "{key}");
                assert_eq!(u64s(ent.get("tiers")), Some(vec![(tag as u64) << 24, 0, 0]), "{key}");
                assert_eq!(u64s(ent.get("shape")), Some(vec![rows as u64, cols as u64]), "{key}: the pack holds the neurons' rows");
                assert_eq!(ent.get("transposed").and_then(Value::as_bool), Some(down), "{key}");
                let tensor = ent.get("tensors").and_then(Value::as_array).unwrap();
                assert_eq!(
                    u64s(tensor[0].get("shape")),
                    Some(if down { vec![cols as u64, rows as u64] } else { vec![rows as u64, cols as u64] }),
                    "{key}: its tensor"
                );
                let p = &held[0][e][mat];
                assert_eq!(saved.read(&format!("{key}.glyd_data")).unwrap(), p.codes, "{key}: the export's codes");
                assert_eq!(saved.read(&format!("{key}.glyd_blocks")).unwrap(), p.scales);
                assert_eq!(saved.read(&format!("{key}.glyd_codebook")).unwrap(), p.codebook);
            }
        }
        assert_eq!(packs.get("m.fc").and_then(|p| p.get("layout")).and_then(Value::as_str), Some("i6"));
        let (_, signs) = saved.get(&format!("{m}.glyd_rot_signs")).unwrap();
        assert_eq!((signs.dtype.as_str(), signs.shape.clone()), ("I8", vec![cols as u64]));
        let on_disk: Vec<i8> = saved.read(&format!("{m}.glyd_rot_signs")).unwrap().into_iter().map(|b| b as i8).collect();
        assert!(on_disk.iter().all(|&s| s == 1 || s == -1) && on_disk.contains(&1) && on_disk.contains(&-1));
        let v = verify(&out, 2, &Decoder::Cpu).unwrap();
        assert_eq!((v.packed, v.hashed, v.unchecked), (5, 2, 0));
        // the decode the hash is of: the exact products rounded to bf16, the down matrix transposed to its tensor
        let mut w = vec![0u16; rows * cols];
        assert!(codebook::unpack_codebook_rows(&held[0][0][1], &mut w));
        assert_eq!(hex(&Sha256::digest(&u16_bytes(&pack::transpose(&w, rows, cols)))), tensor_hash(&j, &format!("{m}.experts.0.down_proj")));
        // refused
        let refused = |why: &str, r: io::Result<Saving>| {
            let e = r.err().map(|e| e.to_string()).unwrap_or_default();
            assert!(e.contains(why), "not refused for {why:?}: {e:?}");
        };
        let o2 = dir.join("refused");
        refused("made from an export of codes", save_planned(&source, &o2, 2, 1 << 30, Layout::Hummingbird, &plan("")));
        refused("made from an export of codes", save_planned(&source, &o2, 2, 1 << 30, Layout::Int6, &plan("")));
        refused("made from an export of codes", save_hummingbird(&source, &o2, 2, 1 << 30, "{\"puts\": [{\"copy\": \"m.norm.weight\"}]}", &exp).map(|s| s));
        refused(
            "layer 3 expert 0",
            save_hummingbird(
                &source,
                &o2,
                2,
                1 << 30,
                &plan(&format!("{{\"codebook\": \"{m}.experts.0.up_proj\", \"layer\": 3, \"expert\": 0, \"matrix\": \"up\"}}")),
                &exp,
            ),
        );
        refused(
            "expert 2",
            save_hummingbird(
                &source,
                &o2,
                2,
                1 << 30,
                &plan(&format!("{{\"codebook\": \"{m}.experts.1.up_proj\", \"layer\": 0, \"expert\": 2, \"matrix\": \"up\"}}")),
                &exp,
            ),
        );
        refused(
            "a matrix Some(\"down\")",
            save_hummingbird(
                &source,
                &o2,
                2,
                1 << 30,
                &plan(&format!("{{\"codebook\": \"{m}.experts.1.up_proj\", \"layer\": 0, \"expert\": 1, \"matrix\": \"down\"}}")),
                &exp,
            ),
        );
        refused(
            "layer 2, where the export has 1 MoE layers",
            save_hummingbird(&source, &o2, 2, 1 << 30, &plan(&format!("{{\"signs\": \"{m}.glyd_rot_signs2\", \"layer\": 2}}")), &exp),
        );
        // an up matrix named as a down one: the source's tensor is [rows, cols], the export's downT would be [cols, rows]
        refused(
            "in the source, where the export's",
            save_hummingbird(
                &source,
                &o2,
                2,
                1 << 30,
                &plan(&format!("{{\"codebook\": \"{m}.experts.1.up_proj\", \"layer\": 0, \"expert\": 1, \"matrix\": \"downT\"}}")),
                &exp,
            ),
        );
        refused("a layout other than", save_planned(&source, &o2, 2, 1 << 30, Layout::Int6, "{\"puts\": [{\"linear\": [\"m.fc\"], \"layout\": \"c3r\"}]}"));
        refused("a row layout is a pack's own", save_planned(&source, &o2, 2, 1 << 30, Layout::Cb3Row, "{\"puts\": []}"));
        // a changed byte of a pack's codes, and a pack named another layout
        let shard = out.join("model.safetensors");
        let mut bytes = std::fs::read(&shard).unwrap();
        let (_, info) = saved.get(&format!("{m}.experts.1.down_proj.glyd_data")).unwrap();
        bytes[info.offset as usize + 5] ^= 0x10;
        std::fs::write(&shard, &bytes).unwrap();
        let e = verify(&out, 2, &Decoder::Cpu).err().map(|e| e.to_string()).unwrap_or_default();
        assert!(e.contains("down_proj.glyd_data") || e.contains("down_proj"), "{e:?}");
        bytes[info.offset as usize + 5] ^= 0x10;
        std::fs::write(&shard, &bytes).unwrap();
        assert_eq!(verify(&out, 2, &Decoder::Cpu).unwrap().packed, 5);
        let text = std::fs::read_to_string(out.join("glyd.json")).unwrap();
        std::fs::write(out.join("glyd.json"), text.replacen("\"layout\": \"c2r\"", "\"layout\": \"c3r\"", 1)).unwrap();
        let e = verify(&out, 2, &Decoder::Cpu).err().map(|e| e.to_string()).unwrap_or_default();
        assert!(e.contains("a pack's sizes are not its layout's"), "{e:?}");
        std::fs::write(out.join("glyd.json"), text.replacen("\"layout\": \"c2r\"", "\"layout\": \"hb\"", 1)).unwrap();
        let e = verify(&out, 2, &Decoder::Cpu).err().map(|e| e.to_string()).unwrap_or_default();
        assert!(e.contains("the layout of a whole save, not of a pack") || e.contains("glyd.json neither packs it nor holds its sha256"), "{e:?}");
        std::fs::write(out.join("glyd.json"), text.replacen("\"transposed\": true", "\"transposed\": false", 1)).unwrap();
        let e = verify(&out, 2, &Decoder::Cpu).err().map(|e| e.to_string()).unwrap_or_default();
        assert!(e.contains("is not its pack's") || e.contains("decodes to other bytes"), "{e:?}");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// The sha256 glyd.json holds for the tensor `module.weight`.
    fn tensor_hash(j: &Value, module: &str) -> String {
        let e = j.get("packs").unwrap().get(module).unwrap();
        e.get("tensors").and_then(Value::as_array).unwrap()[0].get("sha256").and_then(Value::as_str).unwrap().to_string()
    }
}
