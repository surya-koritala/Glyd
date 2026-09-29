//! A bf16 checkpoint packed on the CPU and saved as glyd-v1 (glyd-v2 with a
//! mixture of experts' packs, glyd-v3 in the 12-bit layout), byte for byte as
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
//! Memory: a save holds the shard its writer writes, the one it fills (about
//! 5 GB each) and at most one shard's worth of weights read or packed past the
//! one it waits for (and that one), whatever the threads: about 15 GB at most
//! for a model of several shards, less for one of fewer. A verify on the CPU
//! holds a pack and its matrix decoded for each of its threads, and one more.

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
const FILES: &[&str] = &["config.json", "generation_config.json", "tokenizer*", "special_tokens_map.json", "added_tokens.json", "vocab*", "merges.txt", "*.model", "chat_template*", "preprocessor_config.json", "processor_config.json"];

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
    let cache = std::env::var_os("XDG_CACHE_HOME").map(PathBuf::from).unwrap_or_else(|| Path::new(&std::env::var_os("HOME").unwrap_or_default()).join(".cache"));
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
            let revision = p.split("snapshots/").nth(1).and_then(|r| r.split('/').next()).filter(|r| is_hash(r) && p.contains(&format!("snapshots/{r}/"))).map(str::to_string);
            return Ok(Source { dir: dir.to_path_buf(), repo: model.to_string(), revision });
        }
        let repo = hub_cache().join(format!("models--{}", model.replace('/', "--")));
        let revision = std::fs::read_to_string(repo.join("refs").join("main")).map(|r| r.trim().to_string()).map_err(|_| io::Error::new(io::ErrorKind::NotFound, format!("{model}: no such directory, nor in the Hugging Face cache ({}): download it first (hf download {model}), or give its directory", hub_cache().display())))?;
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
    /// A bf16 tensor as it is: saved as `name` from the source's `from`.
    Copy { name: String, from: String },
    /// Linears' weights stacked by rows (q, k, v; gate, up; or one) as one pack under the first's module path, then
    /// their biases.
    Linear { members: Vec<String> },
    /// A mixture of experts' weight (its E matrices, held [E, out, in]) as one pack of them stacked, [E out, in]:
    /// saved under its module (`module.glyd_WEIGHT_...`) from the source's `from`.
    Experts { module: String, weight: String, from: String, experts: usize },
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
const ATTENTION_QK_NORMS: &[Child] = &[Linear("q_proj"), Linear("k_proj"), Linear("v_proj"), Linear("o_proj"), Plain("q_norm.weight", ""), Plain("k_norm.weight", "")];
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
            ("block_sparse_moe", &[Plain("router.weight", "router.layer.weight"), Experts("experts", &[("gate_up_proj", "input_linear.weight"), ("down_proj", "output_linear.weight")])], &[]),
        ],
        experts: "num_local_experts",
    },
];

/// The layout packs are saved in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Layout {
    /// The tiered layout (glyd-v1; glyd-v2 with a mixture of experts' packs): the smallest, what glyd 0.20 on reads.
    Tiered,
    /// The 12-bit layout (glyd-v3): what an A10, A100 or H100 runs, loaded there without packing again.
    Twelve,
}

impl Layout {
    /// Its name in glyd.json.
    pub fn name(self) -> &'static str {
        match self {
            Layout::Tiered => "mma",
            Layout::Twelve => "mma12",
        }
    }

    fn by_name(name: &str) -> Option<Layout> {
        [Layout::Tiered, Layout::Twelve].into_iter().find(|l| l.name() == name)
    }

    /// A pack's words' name in glyd.json.
    fn words(self) -> &'static str {
        match self {
            Layout::Tiered => "tiers",
            Layout::Twelve => "sym",
        }
    }

    /// A pack's buffers' names (the key's suffixes), dtypes, and bytes a step of its data.
    fn buffers(self) -> ([(&'static str, &'static str); 3], usize) {
        match self {
            Layout::Tiered => ([("data", "U8"), ("blocks", "U8"), ("block_base", "I32")], 1280),
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

impl Plan {
    /// What save_pretrained writes for this checkpoint, in its order: first as the model's modules come (the
    /// embedding as a GEmbedding; each layer's packs, a merged group's after its block's other Linears; experts';
    /// the output layer packed), then what its state_dict holds that is not packed. A tensor the family has that the
    /// checkpoint lacks is refused (transformers would make it up).
    fn new(src: &Checkpoint, config: &Value, merge: bool) -> io::Result<Plan> {
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
        if gembedding {
            puts.push(Put::Copy { name: "model.embed_tokens.weight".into(), from: "model.embed_tokens.weight".into() });
        } else {
            rest.push(Put::Copy { name: "model.embed_tokens.weight".into(), from: "model.embed_tokens.weight".into() });
        }
        let copy = |name: String| Put::Copy { from: name.clone(), name };
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
                let merged = merge && !group.is_empty() && group.iter().all(|c| get(&format!("{}.weight", path(c))).is_some_and(|t| packable(&t))) && group.iter().all(|c| has_bias(c) == has_bias(group[0]));
                for &child in children {
                    match child {
                        Linear(c) => {
                            let t = need(&format!("{}.weight", path(c)))?;
                            if merged && group.contains(&c) {
                                continue;
                            }
                            if packable(&t) {
                                puts.push(Put::Linear { members: vec![path(c)] });
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
                            rest.push(Put::Copy { name: path(name), from });
                        }
                        Experts(module, weights) => {
                            let e = config.get(family.experts).and_then(Value::as_u64).ok_or_else(|| bad(format!("config.json: no {}", family.experts)))? as usize;
                            let ts: Vec<(&str, String, TensorInfo)> = weights.iter().map(|&(w, from)| need(&path(from)).map(|t| (w, path(from), t))).collect::<io::Result<_>>()?;
                            // packed where every matrix [out, in] has rows a multiple of 64 and columns of 16
                            let all = ts.iter().all(|(_, _, t)| t.shape.len() == 3 && t.shape[0] as usize == e && t.shape[1] % 64 == 0 && t.shape[2] % 16 == 0);
                            for (w, from, _) in ts {
                                if all {
                                    puts.push(Put::Experts { module: path(module), weight: w.to_string(), from, experts: e });
                                } else {
                                    rest.push(Put::Copy { name: format!("{}.{w}", path(module)), from });
                                }
                            }
                        }
                    }
                }
                if merged {
                    puts.push(Put::Linear { members: group.iter().map(|c| path(c)).collect() });
                }
            }
        }
        need("model.norm.weight")?;
        rest.push(copy("model.norm.weight".into()));
        if packable(&head) && !tied {
            puts.push(Put::Linear { members: vec!["lm_head".into()] });
        } else if !packable(&head) {
            rest.push(Put::Copy { name: "lm_head.weight".into(), from: if tied { "model.embed_tokens.weight".into() } else { "lm_head.weight".into() } });
        }
        puts.extend(rest);
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
    /// A tensor saved as it is, its sha256 with it.
    fn plain(name: String, shape: Vec<u64>, bytes: Vec<u8>) -> Saved {
        let sha256 = Some(hex(&Sha256::digest(&bytes)));
        Saved { name, dtype: "BF16", shape, bytes, sha256 }
    }
}

/// A put done: its tensors in order, its pack's glyd.json entry, the tensors it checked.
struct Done {
    tensors: Vec<Saved>,
    entry: Option<(String, Value)>,
    checked: usize,
}

fn read_u16(src: &Checkpoint, name: &str, out: &mut [u16]) -> io::Result<()> {
    let (f, t) = src.get(name).ok_or_else(|| bad(format!("no {name}")))?;
    if t.bytes as usize != out.len() * 2 {
        return Err(bad(format!("{name}: {} bytes, where {} are wanted", t.bytes, out.len() * 2)));
    }
    // SAFETY: u16's bytes, as many as its elements take; the file's are little-endian, as the host's.
    let bytes = unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, out.len() * 2) };
    f.read_into(t, bytes)?;
    if cfg!(target_endian = "big") {
        out.iter_mut().for_each(|v| *v = u16::from_le(*v));
    }
    Ok(())
}

fn le_bytes<T: Copy>(v: &[T], f: impl Fn(T) -> [u8; 4]) -> Vec<u8> {
    v.iter().flat_map(|&x| f(x)).collect()
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

/// A pack's glyd.json entry (format.py's entry): its layout, its matrix's shape, its words (tiers, sym), its
/// tensors (name, shape, sha256) in their order in its rows; an experts' weight's E (held as [E, out, in]).
fn entry(layout: Layout, shape: [usize; 2], words: &[u32], tensors: Vec<(String, Vec<u64>, String)>, experts: Option<usize>) -> Value {
    let mut e = vec![
        ("layout".to_string(), Value::from(layout.name())),
        ("shape".to_string(), shape_value(&[shape[0] as u64, shape[1] as u64])),
        (layout.words().to_string(), Value::Array(words.iter().map(|&t| Value::from(t as u64)).collect())),
        ("tensors".to_string(), Value::Array(tensors.into_iter().map(|(n, s, h)| Value::Object(vec![("name".into(), Value::String(n)), ("shape".into(), shape_value(&s)), ("sha256".into(), Value::String(h))])).collect())),
    ];
    if let Some(n) = experts {
        e.push(("experts".into(), Value::from(n as u64)));
        e.push(("transposed".into(), Value::Bool(false)));
    }
    Value::Object(e)
}

/// A matrix packed in `layout`, each step decoded back and compared with its weights ("checked"), its three buffers
/// as saved under `key` (`key`data ...), and its words.
fn packed(layout: Layout, w: &[u16], rows: usize, cols: usize, key: &str, what: &str) -> io::Result<(Vec<Saved>, Vec<u32>)> {
    let differ = || bad(format!("{what}: decoded to other bits than its weights"));
    let steps = (rows * cols / pack::STEP) as u64;
    let buffer = |name: String, dtype, shape, bytes| Saved { name, dtype, shape: vec![shape], bytes, sha256: None };
    Ok(match layout {
        Layout::Tiered => {
            let p = pack::pack_tiered_checked(w, rows, cols).ok_or_else(differ)?;
            let blocks = p.blocks.len() as u64;
            let base = le_bytes(&p.block_base, i32::to_le_bytes);
            (vec![buffer(format!("{key}data"), "U8", steps * 1280, p.data), buffer(format!("{key}blocks"), "U8", blocks, p.blocks), buffer(format!("{key}block_base"), "I32", steps + 1, base)], p.tiers.to_vec())
        }
        Layout::Twelve => {
            let p = pack::pack_twelve_checked(w, rows, cols).ok_or_else(differ)?;
            let (exc, base) = (le_bytes(&p.exc, i32::to_le_bytes), le_bytes(&p.exc_base, i32::to_le_bytes));
            (vec![buffer(format!("{key}data"), "U8", steps * 1536, p.data), buffer(format!("{key}exc"), "I32", p.exc.len() as u64, exc), buffer(format!("{key}exc_base"), "I32", steps + 1, base)], p.sym.to_vec())
        }
    })
}

impl Put {
    /// Its bytes of bf16 read (the work it is).
    fn weight(&self, src: &Checkpoint) -> u64 {
        let b = |n: &str| src.get(n).map_or(0, |(_, t)| t.bytes);
        match self {
            Put::Copy { from, .. } => b(from),
            Put::Linear { members } => members.iter().map(|m| b(&format!("{m}.weight"))).sum(),
            Put::Experts { from, .. } => b(from),
        }
    }

    fn run(&self, src: &Checkpoint, layout: Layout) -> io::Result<Done> {
        match self {
            Put::Copy { name, from } => {
                let (f, t) = src.get(from).ok_or_else(|| bad(format!("no {from}")))?;
                Ok(Done { tensors: vec![Saved::plain(name.clone(), t.shape.clone(), f.read(t)?)], entry: None, checked: 0 })
            }
            Put::Linear { members } => {
                let ts: Vec<TensorInfo> = members.iter().map(|m| src.get(&format!("{m}.weight")).map(|(_, t)| t.clone()).ok_or_else(|| bad(format!("no {m}.weight")))).collect::<io::Result<_>>()?;
                let cols = dims(&ts[0]).1;
                let rows: Vec<usize> = ts.iter().map(|t| dims(t).0).collect();
                let mut w = vec![0u16; rows.iter().sum::<usize>() * cols];
                let mut at = 0;
                for (m, &r) in members.iter().zip(&rows) {
                    read_u16(src, &format!("{m}.weight"), &mut w[at..at + r * cols])?;
                    at += r * cols;
                }
                let (mut tensors, words) = packed(layout, &w, w.len() / cols, cols, &format!("{}.glyd_", members[0]), &members.join(" + "))?;
                let mut hashed = Vec::new();
                let mut at = 0;
                for (m, &r) in members.iter().zip(&rows) {
                    hashed.push((format!("{m}.weight"), vec![r as u64, cols as u64], sha256_u16(&w[at..at + r * cols])));
                    at += r * cols;
                }
                for m in members {
                    if let Some((f, t)) = src.get(&format!("{m}.bias")) {
                        tensors.push(Saved::plain(format!("{m}.bias"), t.shape.clone(), f.read(t)?));
                    }
                }
                let n = hashed.len();
                Ok(Done { tensors, entry: Some((members[0].clone(), entry(layout, [w.len() / cols, cols], &words, hashed, None))), checked: n })
            }
            Put::Experts { module, weight, from, experts } => {
                let (_, t) = src.get(from).ok_or_else(|| bad(format!("no {from}")))?;
                let (e, a, b) = (*experts, t.shape[1] as usize, t.shape[2] as usize);
                let mut w = vec![0u16; e * a * b];
                read_u16(src, from, &mut w)?;
                let sha = sha256_u16(&w);
                let (tensors, words) = packed(layout, &w, e * a, b, &format!("{module}.glyd_{weight}_"), &format!("{module}.{weight}"))?;
                let name = format!("{module}.{weight}");
                Ok(Done { tensors, entry: Some((name.clone(), entry(layout, [e * a, b], &words, vec![(name, t.shape.clone(), sha)], Some(e)))), checked: 1 })
            }
        }
    }
}

/// items run on `threads` threads, their results handed to `done` in their order. A thread takes the item `done`
/// waits for, else a `big` one (a large matrix's pack runs beside the others, not after them), else the next in order,
/// but for the one `done` waits for only while the items taken and not yet handed to `done` weigh at most `budget`
/// with it (their weights: bytes held, or 1 an item). The results wait for their turn in a map.
fn in_order<T: Sync, R: Send>(items: &[T], weights: &[u64], big: &[bool], threads: usize, budget: u64, run: impl Fn(&T) -> io::Result<R> + Sync, mut done: impl FnMut(R) -> io::Result<()>) -> io::Result<()> {
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
    const DTYPES: &[&str] = &["BOOL", "F4", "F6_E2M3", "F6_E3M2", "U8", "I8", "F8_E5M2", "F8_E4M3", "F8_E8M0", "I16", "U16", "F16", "BF16", "I32", "U32", "F32", "C64", "F64", "I64", "U64"];
    let rank = |d: &str| DTYPES.iter().position(|x| *x == d).expect("a dtype safetensors has");
    tensors.sort_by(|a, b| rank(b.dtype).cmp(&rank(a.dtype)).then(a.name.cmp(&b.name)));
    let mut header = vec![("__metadata__".to_string(), Value::Object(vec![("format".into(), Value::from("pt"))]))];
    let mut pos = 0u64;
    for t in tensors.iter() {
        let n = t.bytes.len() as u64;
        header.push((t.name.clone(), Value::Object(vec![("dtype".into(), Value::from(t.dtype)), ("shape".into(), shape_value(&t.shape)), ("data_offsets".into(), Value::Array(vec![Value::from(pos), Value::from(pos + n)]))])));
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
    /// Bytes of bf16 read.
    pub bytes: u64,
    pub shards: usize,
}

/// The checkpoint at `source` packed on `threads` threads and saved in `out` as glyd-v1 (glyd-v2 with a mixture of
/// experts' packs; glyd-v3 in the 12-bit layout), as save_pretrained saves it: shards of about `shard_bytes`,
/// glyd.json (with the sha256 of every tensor, packed or saved as it is), the index (several shards), the source's
/// config, generation config and tokenizer files. merge: q, k, v and gate, up as one pack each (from_pretrained's
/// merge). A shard is written by a thread of its own as the next is made; the host holds two and at most a shard's
/// worth of weights in flight past the one the save waits for (the module's docs). A directory holding another
/// checkpoint is refused.
pub fn save(source: &Source, out: &Path, threads: usize, shard_bytes: u64, merge: bool, layout: Layout) -> io::Result<Saving> {
    let cfg = source.dir.join("config.json");
    let config = json::parse(&std::fs::read_to_string(&cfg).map_err(at(&cfg))?).map_err(|e| bad(format!("{}: {e}", cfg.display())))?;
    let src = Checkpoint::open(&source.dir)?;
    let plan = Plan::new(&src, &config, merge)?;
    let listing = |pattern: &dyn Fn(&str) -> bool| -> io::Result<Vec<PathBuf>> {
        Ok(match std::fs::read_dir(out) {
            Ok(d) => d.filter_map(|e| e.ok()).map(|e| e.path()).filter(|p| p.file_name().and_then(|n| n.to_str()).is_some_and(pattern)).collect(),
            Err(_) => Vec::new(),
        })
    };
    let old = listing(&|n| (n.starts_with("model") && n.ends_with(".safetensors")) || n == "model.safetensors.index.json")?;
    if !old.is_empty() && !out.join("glyd.json").exists() {
        let held = old.iter().filter(|p| p.extension().is_some_and(|e| e == "safetensors")).any(|p| crate::safetensors::File::open(p).is_ok_and(|f| f.tensors.iter().any(|(n, _)| n.contains(".glyd_"))));
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
    let (mut checked, mut experts) = (0, false);
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
            in_order(&plan.puts, &weights, &big, threads, shard_bytes, |p| p.run(&src, layout), |d: Done| {
                checked += d.checked;
                if let Some((name, e)) = d.entry {
                    experts |= e.get("experts").is_some();
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
            })
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
    let names: Vec<String> = if shards.len() == 1 { vec!["model.safetensors".into()] } else { (1..=shards.len()).map(|i| format!("model-{i:05}-of-{:05}.safetensors", shards.len())).collect() };
    for ((tmp, _, _), name) in shards.iter().zip(&names) {
        std::fs::rename(out.join(tmp), out.join(name)).map_err(at(&out.join(name)))?;
    }
    if shards.len() > 1 {
        let map = shards.iter().zip(&names).flat_map(|((_, ks, _), name)| ks.iter().map(move |k| (k.clone(), Value::from(name.as_str())))).collect();
        let total: u64 = shards.iter().map(|s| s.2).sum();
        let index = Value::Object(vec![("metadata".into(), Value::Object(vec![("total_size".into(), Value::from(total))])), ("weight_map".into(), Value::Object(map))]);
        let p = out.join("model.safetensors.index.json");
        std::fs::write(&p, json::to_python(&index, 2)).map_err(at(&p))?;
    }
    let source_v = Value::Object(vec![("repo".into(), Value::String(source.repo.clone())), ("revision".into(), source.revision.clone().map_or(Value::Null, Value::String))]);
    let hashed = hashes.len();
    let m = Value::Object(vec![
        ("format".into(), Value::from(if layout == Layout::Twelve { "glyd-v3" } else if experts { "glyd-v2" } else { "glyd-v1" })),
        ("glyd".into(), Value::from(env!("CARGO_PKG_VERSION"))),
        ("source".into(), source_v),
        ("layout".into(), Value::from(layout.name())),
        ("packs".into(), Value::Object(packs)),
        ("tensors".into(), Value::Object(hashes)),
    ]);
    std::fs::write(&manifest, json::to_python(&m, 1)).map_err(at(&manifest))?;
    // the source's config, generation config and tokenizer files, each once, its content alone
    if std::fs::canonicalize(&source.dir)? != std::fs::canonicalize(out)? {
        let mut files: Vec<PathBuf> = std::fs::read_dir(&source.dir)?.filter_map(|e| e.ok()).map(|e| e.path()).filter(|p| p.is_file() && p.file_name().and_then(|n| n.to_str()).is_some_and(|n| !n.starts_with('.') && FILES.iter().any(|f| matches(f, n)))).collect();
        files.sort();
        for f in files {
            let to = out.join(f.file_name().unwrap());
            std::fs::write(&to, std::fs::read(&f).map_err(at(&f))?).map_err(at(&to))?; // (the Hub's cache keeps them read-only: not the mode)
        }
    }
    Ok(Saving { checked, hashed, bytes, shards: shards.len() })
}

/// The formats `verify` reads (format.py's FORMATS).
const FORMATS: &[&str] = &["glyd-v1", "glyd-v2", "glyd-v3"];

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

/// A pack of `dir` read and decoded: its entry in glyd.json, the matrix [rows, cols], bf16. Its buffers are read and
/// held to its shape before its matrix is made, and (on a GPU) to where the decode reads by them.
fn decode(dir: &Checkpoint, name: &str, e: &Value, decoder: &Decoder) -> io::Result<(usize, usize, Vec<u16>)> {
    let what = |why: &str| bad(format!("{name}: {why}"));
    let layout = e.get("layout").and_then(Value::as_str).and_then(Layout::by_name).ok_or_else(|| what("a layout this glyd does not read"))?;
    let shape = u64s(e.get("shape")).filter(|s| s.len() == 2).ok_or_else(|| what("no shape in glyd.json"))?;
    let n_words = if layout == Layout::Tiered { 3 } else { 4 };
    let words = u64s(e.get(layout.words())).filter(|t| t.len() == n_words && t.iter().all(|&x| x <= u32::MAX as u64)).ok_or_else(|| what(&format!("no {} in glyd.json", layout.words())))?;
    let words: Vec<u32> = words.into_iter().map(|x| x as u32).collect();
    let (rows, cols) = (shape[0] as usize, shape[1] as usize);
    if rows == 0 || rows % 64 != 0 || cols == 0 || cols % 16 != 0 || rows > (1 << 40) / cols {
        return Err(what("not an mma layout's shape"));
    }
    let key = pack_key(name, e);
    let (bufs, step_bytes) = layout.buffers();
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
        (Layout::Tiered, Decoder::Cpu) => {
            let p = pack::Tiered { rows, cols, data, blocks: a, block_base: i32s(&base), tiers: [words[0], words[1], words[2]] };
            if !pack::unpack_tiered(&p, &mut w) {
                return Err(what("its blocks do not hold its escapes"));
            }
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
            let ok = match layout {
                Layout::Tiered => pack::tiered_blocks_hold(&data, &a, &base),
                Layout::Twelve => a.len() % 16 == 0 && base[0] == 0 && base.windows(2).all(|x| x[0] <= x[1]) && base[steps] as usize <= a.len() / 4,
            };
            if !ok {
                return Err(what("its buffers do not hold what its data says (the GPU's decode would read past them)"));
            }
            let gpu = |r: crate::Result<()>| r.map_err(|e| bad(format!("{name}: {e}")));
            let (d, x, b) = (ctx.upload(&data).map_err(bad)?, ctx.upload(&a).map_err(bad)?, ctx.upload(&base).map_err(bad)?);
            let out = ctx.alloc(rows * cols * 2).map_err(bad)?;
            let p = match layout {
                Layout::Tiered => crate::Pack::Tiered(crate::Tiered { data: d.ptr(), blocks: x.ptr(), block_base: b.ptr(), tiers: [words[0], words[1], words[2]] }),
                Layout::Twelve => crate::Pack::Twelve(crate::Twelve { data: d.ptr(), exc: x.ptr(), exc_base: b.ptr(), sym: [words[0], words[1], words[2], words[3]] }),
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

/// A pack's tensors against glyd.json's sha256 (a merged pack's in their rows' order; an experts' weight as the
/// model holds it, [E, out, in]): how many.
fn hashes_match(name: &str, e: &Value, rows: usize, cols: usize, w: &[u16]) -> io::Result<usize> {
    let tensors = e.get("tensors").and_then(Value::as_array).filter(|t| !t.is_empty()).ok_or_else(|| bad(format!("{name}: no tensors in glyd.json")))?;
    let mut row = 0;
    for t in tensors {
        let tn = t.get("name").and_then(Value::as_str).unwrap_or(name);
        let want = t.get("sha256").and_then(Value::as_str).ok_or_else(|| bad(format!("{tn}: no sha256 in glyd.json")))?;
        let shape = u64s(t.get("shape")).ok_or_else(|| bad(format!("{tn}: no shape in glyd.json")))?;
        let got = match (e.get("experts").and_then(Value::as_u64), shape.as_slice()) {
            (Some(n), &[en, a, b]) if en == n && (n * a) as usize * b as usize == rows * cols => {
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
                    sha256_u16(&held)
                } else {
                    sha256_u16(w)
                }
            }
            (None, &[r, c]) if c as usize == cols && row + r as usize <= rows => {
                row += r as usize;
                sha256_u16(&w[(row - r as usize) * cols..row * cols])
            }
            _ => return Err(bad(format!("{tn}: its shape {shape:?} in glyd.json is not its pack's"))),
        };
        if got != want {
            return Err(bad(format!("{tn} decodes to other bytes than glyd.json's sha256")));
        }
    }
    if e.get("experts").is_none() && row != rows {
        return Err(bad(format!("{name}: its tensors' rows are not its own")));
    }
    Ok(tensors.len())
}

/// What a verify found.
pub struct Verified {
    /// Packed tensors decoded to glyd.json's sha256.
    pub packed: usize,
    /// Tensors saved as they are at glyd.json's sha256.
    pub hashed: usize,
    /// Tensors saved as they are with no sha256 in glyd.json to check (a save of glyd 0.24 or before).
    pub unchecked: usize,
}

/// A saved checkpoint (glyd-v1, glyd-v2, glyd-v3) checked, as `python -m glyd.gpu verify` checks it: each file's
/// tensors back to back to its end (the reader); the index, where there are shards, naming each tensor's shard;
/// every tensor a pack's buffer or one whose sha256 glyd.json holds (glyd-v3, and glyd-v1 and v2 saved by glyd 0.25
/// on; before, those are counted as unchecked); each pack's tensors its own (its module's .weight, a merged group's
/// q, k, v or gate, up under the first's path, an experts' weight's own name), in no other pack and none also saved
/// as it is; every pack decoded (on `threads` threads of the CPU, or on a GPU) and each of its tensors' sha256, and
/// each tensor saved as it is, against glyd.json's.
pub fn verify(dir: &Path, threads: usize, decoder: &Decoder) -> io::Result<Verified> {
    let path = dir.join("glyd.json");
    let text = std::fs::read_to_string(&path).map_err(at(&path))?;
    let m = json::parse(&text).map_err(|e| bad(format!("{}: {e}", path.display())))?;
    let format = m.get("format").and_then(Value::as_str).unwrap_or("");
    if !FORMATS.contains(&format) {
        return Err(bad(format!("{}: format {format:?}; this glyd reads {}", path.display(), FORMATS.join(" and "))));
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
    let hashes: Option<Vec<(String, Value)>> = m.get("tensors").and_then(Value::as_object).map(<[_]>::to_vec);
    if hashes.is_none() && format == "glyd-v3" {
        return Err(bad("glyd.json: no sha256 for the tensors saved as they are (glyd-v3 has them)"));
    }
    let hashed: HashMap<&str, &str> = hashes.iter().flatten().map(|(k, v)| (k.as_str(), v.as_str().unwrap_or(""))).collect();
    let (mut buffers, mut members) = (HashSet::new(), HashSet::new());
    for (name, e) in &packs {
        let layout = e.get("layout").and_then(Value::as_str).and_then(Layout::by_name).ok_or_else(|| bad(format!("{name}: a layout this glyd does not read")))?;
        let key = pack_key(name, e);
        buffers.extend(layout.buffers().0.iter().map(|(b, _)| format!("{key}{b}")));
        let ts: Vec<&str> = e.get("tensors").and_then(Value::as_array).unwrap_or(&[]).iter().map(|t| t.get("name").and_then(Value::as_str).unwrap_or("")).collect();
        // its own: an experts' weight's name, its module's .weight, a merged group's (q, k, v; gate, up: every name
        // the key's)
        let (parent, first) = name.rsplit_once('.').unwrap_or(("", name));
        let own: Vec<String> = match (e.get("experts"), ts.len()) {
            (Some(_), _) => vec![name.clone()],
            (None, 1) => vec![format!("{name}.weight")],
            _ => [QKV, GATE_UP].iter().filter(|g| g[0] == first && g.len() == ts.len()).flat_map(|g| g.iter().map(|c| format!("{parent}.{c}.weight"))).collect(),
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
    let one = |(name, e): &(String, Value), decoder: &Decoder| decode(&saved, name, e, decoder).and_then(|(r, c, w)| hashes_match(name, e, r, c, &w));
    match decoder {
        // a pack and its matrix a thread, and one more waiting its turn
        Decoder::Cpu => in_order(&packs, &vec![1; packs.len()], &vec![false; packs.len()], threads, threads as u64 + 1, |p| one(p, &Decoder::Cpu), |k| {
            n += k;
            Ok(())
        })?,
        Decoder::Gpu(..) => {
            for p in &packs {
                n += one(p, decoder)?;
            }
        }
    }
    Ok(Verified { packed: n, hashed: hashed.len(), unchecked })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn file_patterns() {
        assert!(matches("tokenizer*", "tokenizer.json") && matches("*.model", "spm.model") && matches("vocab*", "vocab.json"));
        assert!(!matches("*.model", "model") && !matches("config.json", "config.json.bak"));
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
            in_order(&items, &weights, &big, threads, budget, |&i| Ok(i * 3), |r| {
                got.push(r);
                Ok(())
            })
            .unwrap();
            assert_eq!(got, items.iter().map(|i| i * 3).collect::<Vec<_>>(), "{threads} threads, budget {budget}");
        }
    }

    /// A tiny Qwen3 checkpoint (random bf16 weights, two layers, the output layer tied), saved in shards of 64 KB
    /// and verified: glyd.json's packs in save_pretrained's order (a layer's o_proj, then q, k, v as one; down_proj,
    /// then gate, up), every tensor's sha256 the checkpoint's; then refused: one byte of a pack changed, one of a
    /// tensor saved as it is, bytes appended to a shard, a merged pack's member renamed (to another's, and by one
    /// letter), a member of one pack another's, a tensor neither packed nor hashed; and the same save in the 12-bit
    /// layout.
    #[test]
    fn save_and_verify_a_tiny_qwen3() {
        let dir = std::env::temp_dir().join(format!("glyd-gpu-test-{}", std::process::id()));
        let (src, out) = (dir.join("src"), dir.join("out"));
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
            for (n, s) in [("self_attn.q_proj.weight", &[128, 128][..]), ("self_attn.k_proj.weight", &[64, 128]), ("self_attn.v_proj.weight", &[64, 128]), ("self_attn.o_proj.weight", &[128, 128]), ("self_attn.q_norm.weight", &[32]), ("self_attn.k_norm.weight", &[32]), ("mlp.gate_proj.weight", &[256, 128]), ("mlp.up_proj.weight", &[256, 128]), ("mlp.down_proj.weight", &[128, 256]), ("input_layernorm.weight", &[128]), ("post_attention_layernorm.weight", &[128])] {
                ts.push(t(&format!("model.layers.{i}.{n}"), s));
            }
        }
        write_safetensors(&src.join("model.safetensors"), &mut ts).unwrap();
        let source = Source::find(src.to_str().unwrap()).unwrap();
        let s = save(&source, &out, 3, 64 << 10, true, Layout::Tiered).unwrap();
        assert_eq!((s.checked, s.hashed, s.shards > 1), (14, 10, true));
        let m = json::parse(&std::fs::read_to_string(out.join("glyd.json")).unwrap()).unwrap();
        let packs: Vec<&str> = m.get("packs").unwrap().as_object().unwrap().iter().map(|(k, _)| k.as_str()).collect();
        assert_eq!(&packs[..4], &["model.layers.0.self_attn.o_proj", "model.layers.0.self_attn.q_proj", "model.layers.0.mlp.down_proj", "model.layers.0.mlp.gate_proj"]);
        assert_eq!(m.get("format").and_then(Value::as_str), Some("glyd-v1"));
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
        for (from, to, why) in [("self_attn.k_proj.weight", "self_attn.v_proj.weight", "self_attn.q_proj's tensors are not its own"), ("self_attn.k_proj.weight", "self_attn.k_prok.weight", "self_attn.q_proj's tensors are not its own"), ("self_attn.o_proj.weight", "self_attn.q_proj.weight", "self_attn.o_proj's tensors are not its own")] {
            fresh(&bad);
            let text = std::fs::read_to_string(bad.join("glyd.json")).unwrap().replacen(&format!("\"model.layers.0.{from}\""), &format!("\"model.layers.0.{to}\""), 1);
            std::fs::write(bad.join("glyd.json"), text).unwrap();
            refused(why);
        }
        // a member of one pack another's: down_proj's pack renamed up_proj's in glyd.json (its tensor then its own)
        fresh(&bad);
        let text = std::fs::read_to_string(bad.join("glyd.json")).unwrap().replace("layers.0.mlp.down_proj\"", "layers.0.mlp.up_proj\"").replace("layers.0.mlp.down_proj.weight\"", "layers.0.mlp.up_proj.weight\"");
        std::fs::write(bad.join("glyd.json"), text).unwrap();
        refused("a tensor held by two packs, or twice by one");
        fresh(&bad);
        let text = std::fs::read_to_string(bad.join("glyd.json")).unwrap().replace("\"model.norm.weight\"", "\"model.norm.weight2\"");
        std::fs::write(bad.join("glyd.json"), text).unwrap();
        refused("model.norm.weight: in the safetensors, but glyd.json neither packs it");
        // the 12-bit layout: glyd-v3, a pack's sym and its data, exc and exc_base
        let out12 = dir.join("out12");
        assert_eq!(save(&source, &out12, 2, 1 << 30, true, Layout::Twelve).unwrap().checked, 14);
        let m = json::parse(&std::fs::read_to_string(out12.join("glyd.json")).unwrap()).unwrap();
        assert_eq!((m.get("format").and_then(Value::as_str), m.get("layout").and_then(Value::as_str)), (Some("glyd-v3"), Some("mma12")));
        let o = m.get("packs").unwrap().get("model.layers.1.self_attn.o_proj").unwrap();
        assert!(o.get("sym").is_some() && o.get("tiers").is_none());
        assert!(Checkpoint::open(&out12).unwrap().get("model.layers.1.self_attn.o_proj.glyd_exc_base").is_some());
        assert_eq!(verify(&out12, 2, &Decoder::Cpu).unwrap().packed, 14);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
