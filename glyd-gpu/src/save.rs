//! A bf16 checkpoint packed on the CPU and saved as glyd-v1 (glyd-v2 with a
//! mixture of experts' packs), byte for byte as `python -m glyd.gpu pack`
//! saves it (bindings/python/glyd/gpu/format.py's save_pretrained over the
//! model hf.py's from_pretrained loads): the same tensors, names, shards,
//! glyd.json and index; and a saved one checked (`verify`).
//!
//! Python saves what transformers' model holds, in its module order; here the
//! families' layouts are written out (`Family`), each checked against
//! Python's saves: Qwen3 (Qwen3-0.6B to 8B) and GraniteMoe
//! (granite-3.1-3b-a800m). Anything else is refused, with Python's command.

use crate::json::{self, Value};
use crate::pack;
use crate::safetensors::{Checkpoint, TensorInfo};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
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
    /// A mixture of experts' weight (its E matrices, held [E, out, in], or [E, in, out] transposed) as one pack of
    /// them stacked, [E out, in]: saved under its module (`module.glyd_WEIGHT_...`) from the source's `from`.
    Experts { module: String, weight: String, from: String, experts: usize, transposed: bool },
}

/// A decoder layer's module in its family's order (transformers' registration order).
#[derive(Clone, Copy)]
enum Child {
    /// An nn.Linear (weight, bias where there is one).
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

const ATTENTION: &[Child] = &[Linear("q_proj"), Linear("k_proj"), Linear("v_proj"), Linear("o_proj"), Plain("q_norm.weight", ""), Plain("k_norm.weight", "")];
const QKV: &[&str] = &["q_proj", "k_proj", "v_proj"];
const NORMS: &[Child] = &[Plain("input_layernorm.weight", ""), Plain("post_attention_layernorm.weight", "")];

const FAMILIES: &[Family] = &[
    Family {
        architectures: &["Qwen3ForCausalLM"],
        layer: &[("self_attn", ATTENTION, QKV), ("mlp", &[Linear("gate_proj"), Linear("up_proj"), Linear("down_proj")], &["gate_proj", "up_proj"]), ("", NORMS, &[])],
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
    /// the output layer packed), then what its state_dict holds that is not packed.
    fn new(src: &Checkpoint, config: &Value, merge: bool) -> io::Result<Plan> {
        let arch = config.get("architectures").and_then(Value::as_array).and_then(|a| a.first()).and_then(Value::as_str).unwrap_or("");
        let family = FAMILIES.iter().find(|f| f.architectures.contains(&arch)).ok_or_else(|| bad(format!("{arch}: not a family this packer knows (Qwen3ForCausalLM, GraniteMoeForCausalLM); python -m glyd.gpu pack packs any")))?;
        let layers = config.get("num_hidden_layers").and_then(Value::as_u64).ok_or_else(|| bad("config.json: no num_hidden_layers"))? as usize;
        let get = |name: &str| src.get(name).map(|(_, t)| t.clone());
        let need = |name: &str| get(name).ok_or_else(|| bad(format!("{}: no {name}", src.dir.display())));
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
                // the group runs as one product where every member is packed and all or none have a bias
                let has_bias = |c: &str| get(&format!("{}.bias", path(c))).is_some();
                let merged = merge && !group.is_empty() && group.iter().all(|c| get(&format!("{}.weight", path(c))).is_some_and(|t| packable(&t))) && group.iter().all(|c| has_bias(c) == has_bias(group[0]));
                for &child in children {
                    match child {
                        Linear(c) => {
                            let Some(t) = get(&format!("{}.weight", path(c))) else { continue };
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
                            if get(&from).is_some() {
                                rest.push(Put::Copy { name: path(name), from });
                            }
                        }
                        Experts(module, weights) => {
                            let e = config.get(family.experts).and_then(Value::as_u64).ok_or_else(|| bad(format!("config.json: no {}", family.experts)))? as usize;
                            let ts: Vec<(&str, String, TensorInfo)> = weights.iter().map(|&(w, from)| need(&path(from)).map(|t| (w, path(from), t))).collect::<io::Result<_>>()?;
                            // packed where every matrix [out, in] has rows a multiple of 64 and columns of 16
                            let all = ts.iter().all(|(_, _, t)| t.shape.len() == 3 && t.shape[0] as usize == e && t.shape[1] % 64 == 0 && t.shape[2] % 16 == 0);
                            for (w, from, _) in ts {
                                if all {
                                    puts.push(Put::Experts { module: path(module), weight: w.to_string(), from, experts: e, transposed: false });
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

/// A tensor to save: its name, dtype, shape, bytes.
pub(crate) struct Saved {
    name: String,
    dtype: &'static str,
    shape: Vec<u64>,
    bytes: Vec<u8>,
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

/// A pack's glyd.json entry (format.py's entry): its matrix's shape, its tiers, its tensors (name, shape, sha256)
/// in their order in its rows; an experts' weight's E and whether it is held transposed.
fn entry(shape: [usize; 2], tiers: [u32; 3], tensors: Vec<(String, Vec<u64>, String)>, experts: Option<(usize, bool)>) -> Value {
    let mut e = vec![
        ("layout".to_string(), Value::from("mma")),
        ("shape".to_string(), shape_value(&[shape[0] as u64, shape[1] as u64])),
        ("tiers".to_string(), Value::Array(tiers.iter().map(|&t| Value::from(t as u64)).collect())),
        ("tensors".to_string(), Value::Array(tensors.into_iter().map(|(n, s, h)| Value::Object(vec![("name".into(), Value::String(n)), ("shape".into(), shape_value(&s)), ("sha256".into(), Value::String(h))])).collect())),
    ];
    if let Some((n, tr)) = experts {
        e.push(("experts".into(), Value::from(n as u64)));
        e.push(("transposed".into(), Value::Bool(tr)));
    }
    Value::Object(e)
}

/// A matrix packed, decoded back and compared with its weights ("checked"), its three buffers as saved under `key`
/// (`key`.glyd_data ...).
fn packed(w: &[u16], rows: usize, cols: usize, key: &str, what: &str) -> io::Result<(Vec<Saved>, [u32; 3])> {
    let p = pack::pack_tiered_checked(w, rows, cols).ok_or_else(|| bad(format!("{what}: decoded to other bits than its weights")))?;
    let steps = (rows * cols / pack::STEP) as u64;
    let blocks = p.blocks.len() as u64;
    let tensors = vec![
        Saved { name: format!("{key}data"), dtype: "U8", shape: vec![steps * 1280], bytes: p.data },
        Saved { name: format!("{key}blocks"), dtype: "U8", shape: vec![blocks], bytes: p.blocks },
        Saved { name: format!("{key}block_base"), dtype: "I32", shape: vec![steps + 1], bytes: le_bytes(&p.block_base, i32::to_le_bytes) },
    ];
    Ok((tensors, p.tiers))
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

    fn run(&self, src: &Checkpoint) -> io::Result<Done> {
        match self {
            Put::Copy { name, from } => {
                let (f, t) = src.get(from).ok_or_else(|| bad(format!("no {from}")))?;
                Ok(Done { tensors: vec![Saved { name: name.clone(), dtype: "BF16", shape: t.shape.clone(), bytes: f.read(t)? }], entry: None, checked: 0 })
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
                let (mut tensors, tiers) = packed(&w, w.len() / cols, cols, &format!("{}.glyd_", members[0]), &members.join(" + "))?;
                let mut hashed = Vec::new();
                let mut at = 0;
                for (m, &r) in members.iter().zip(&rows) {
                    hashed.push((format!("{m}.weight"), vec![r as u64, cols as u64], sha256_u16(&w[at..at + r * cols])));
                    at += r * cols;
                }
                for m in members {
                    if let Some((f, t)) = src.get(&format!("{m}.bias")) {
                        tensors.push(Saved { name: format!("{m}.bias"), dtype: "BF16", shape: t.shape.clone(), bytes: f.read(t)? });
                    }
                }
                let n = hashed.len();
                Ok(Done { tensors, entry: Some((members[0].clone(), entry([w.len() / cols, cols], tiers, hashed, None))), checked: n })
            }
            Put::Experts { module, weight, from, experts, transposed } => {
                let (_, t) = src.get(from).ok_or_else(|| bad(format!("no {from}")))?;
                let (e, a, b) = (*experts, t.shape[1] as usize, t.shape[2] as usize);
                let mut w = vec![0u16; e * a * b];
                read_u16(src, from, &mut w)?;
                let sha = sha256_u16(&w);
                if *transposed {
                    return Err(bad(format!("{module}.{weight}: experts held transposed are not packed here")));
                }
                let (tensors, tiers) = packed(&w, e * a, b, &format!("{module}.glyd_{weight}_"), &format!("{module}.{weight}"))?;
                let name = format!("{module}.{weight}");
                Ok(Done { tensors, entry: Some((name.clone(), entry([e * a, b], tiers, vec![(name, t.shape.clone(), sha)], Some((e, *transposed))))), checked: 1 })
            }
        }
    }
}

/// items run on `threads` threads, their results handed to `done` in their order: the `big` ones taken first (a
/// large matrix's pack runs beside the others, not after them), the rest in order, at most `window` past the one
/// `done` waits for (the results held until their turn).
fn in_order<T: Sync, R: Send>(items: &[T], big: &[bool], threads: usize, window: usize, run: impl Fn(&T) -> io::Result<R> + Sync, mut done: impl FnMut(R) -> io::Result<()>) -> io::Result<()> {
    struct State<R> {
        pos: usize,
        taken: usize,
        results: BTreeMap<usize, io::Result<R>>,
        stop: bool,
    }
    let order: Vec<usize> = (0..items.len()).filter(|&i| big[i]).chain((0..items.len()).filter(|&i| !big[i])).collect();
    let state = Mutex::new(State { pos: 0, taken: 0, results: BTreeMap::new(), stop: false });
    let cv = Condvar::new();
    std::thread::scope(|s| {
        for _ in 0..threads.max(1) {
            s.spawn(|| loop {
                let i = {
                    let mut st = state.lock().unwrap();
                    while !st.stop && st.pos < order.len() && !big[order[st.pos]] && order[st.pos] >= st.taken + window {
                        st = cv.wait(st).unwrap();
                    }
                    if st.stop || st.pos >= order.len() {
                        return;
                    }
                    st.pos += 1;
                    order[st.pos - 1]
                };
                let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run(&items[i]))).unwrap_or_else(|_| Err(bad("a packing thread panicked")));
                state.lock().unwrap().results.insert(i, r);
                cv.notify_all();
            });
        }
        let mut result = Ok(());
        for i in 0..items.len() {
            let r = {
                let mut st = state.lock().unwrap();
                loop {
                    if let Some(r) = st.results.remove(&i) {
                        st.taken = i + 1;
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
    let mut at = 0u64;
    for t in tensors.iter() {
        let n = t.bytes.len() as u64;
        header.push((t.name.clone(), Value::Object(vec![("dtype".into(), Value::from(t.dtype)), ("shape".into(), shape_value(&t.shape)), ("data_offsets".into(), Value::Array(vec![Value::from(at), Value::from(at + n)]))])));
        at += n;
    }
    let mut h = json::to_compact(&Value::Object(header)).into_bytes();
    h.resize(h.len().next_multiple_of(8), b' ');
    let mut f = io::BufWriter::with_capacity(8 << 20, std::fs::File::create(path)?);
    f.write_all(&(h.len() as u64).to_le_bytes())?;
    f.write_all(&h)?;
    for t in tensors.iter() {
        f.write_all(&t.bytes)?;
    }
    f.flush()
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
    /// Bytes of bf16 read.
    pub bytes: u64,
    pub shards: usize,
}

/// The checkpoint at `source` packed on `threads` threads and saved in `out` as glyd-v1 (glyd-v2 with a mixture of
/// experts' packs), as save_pretrained saves it: shards of about `shard_bytes`, glyd.json, the index (several
/// shards), the source's config, generation config and tokenizer files. merge: q, k, v and gate, up as one pack
/// each (from_pretrained's merge). A shard is written once whole (the host holds a shard and the packs in flight). A
/// directory holding another checkpoint is refused.
pub fn save(source: &Source, out: &Path, threads: usize, shard_bytes: u64, merge: bool) -> io::Result<Saving> {
    let config = json::parse(&std::fs::read_to_string(source.dir.join("config.json"))?).map_err(bad)?;
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
        std::fs::remove_file(p)?; // a save cut short's
    }
    std::fs::create_dir_all(out)?;

    let (mut part, mut size, mut shards) = (Vec::<Saved>::new(), 0u64, Vec::<(String, Vec<String>, u64)>::new());
    let mut packs = Vec::new();
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
            in_order(&plan.puts, &big, threads, 4 * threads.max(1), |p| p.run(&src), |d: Done| {
                checked += d.checked;
                if let Some((name, e)) = d.entry {
                    experts |= e.get("experts").is_some();
                    packs.push((name, e));
                }
                for t in d.tensors {
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
        made.and(written)
    })?;

    // glyd.json first out (a save cut short from here on has shards and no manifest, which a load refuses), the
    // earlier save's files, the shards renamed, the index
    let manifest = out.join("glyd.json");
    if manifest.exists() {
        std::fs::remove_file(&manifest)?;
    }
    for p in old {
        std::fs::remove_file(p)?;
    }
    let names: Vec<String> = if shards.len() == 1 { vec!["model.safetensors".into()] } else { (1..=shards.len()).map(|i| format!("model-{i:05}-of-{:05}.safetensors", shards.len())).collect() };
    for ((tmp, _, _), name) in shards.iter().zip(&names) {
        std::fs::rename(out.join(tmp), out.join(name))?;
    }
    if shards.len() > 1 {
        let map = shards.iter().zip(&names).flat_map(|((_, ks, _), name)| ks.iter().map(move |k| (k.clone(), Value::from(name.as_str())))).collect();
        let total: u64 = shards.iter().map(|s| s.2).sum();
        let index = Value::Object(vec![("metadata".into(), Value::Object(vec![("total_size".into(), Value::from(total))])), ("weight_map".into(), Value::Object(map))]);
        std::fs::write(out.join("model.safetensors.index.json"), json::to_python(&index, 2))?;
    }
    let source_v = Value::Object(vec![("repo".into(), Value::String(source.repo.clone())), ("revision".into(), source.revision.clone().map_or(Value::Null, Value::String))]);
    let m = Value::Object(vec![
        ("format".into(), Value::from(if experts { "glyd-v2" } else { "glyd-v1" })),
        ("glyd".into(), Value::from(env!("CARGO_PKG_VERSION"))),
        ("source".into(), source_v),
        ("layout".into(), Value::from("mma")),
        ("packs".into(), Value::Object(packs)),
    ]);
    std::fs::write(&manifest, json::to_python(&m, 1))?;
    // the source's config, generation config and tokenizer files, each once, its content alone
    if std::fs::canonicalize(&source.dir)? != std::fs::canonicalize(out)? {
        let mut files: Vec<PathBuf> = std::fs::read_dir(&source.dir)?.filter_map(|e| e.ok()).map(|e| e.path()).filter(|p| p.is_file() && p.file_name().and_then(|n| n.to_str()).is_some_and(|n| !n.starts_with('.') && FILES.iter().any(|f| matches(f, n)))).collect();
        files.sort();
        for f in files {
            std::fs::write(out.join(f.file_name().unwrap()), std::fs::read(&f)?)?; // (the Hub's cache keeps them read-only: not the mode)
        }
    }
    Ok(Saving { checked, bytes, shards: shards.len() })
}

/// The formats `verify` reads (format.py's FORMATS).
const FORMATS: &[&str] = &["glyd-v1", "glyd-v2"];

/// Where a pack's matrix is decoded for `verify`: on the CPU, or on a GPU by the library.
pub enum Decoder<'a> {
    Cpu,
    Gpu(&'a crate::Library, &'a crate::cuda::Context),
}

fn u64s(v: Option<&Value>) -> Option<Vec<u64>> {
    v.and_then(Value::as_u64s)
}

/// A pack of `dir` read and decoded: its entry in glyd.json, the matrix [rows, cols], bf16.
fn decode(dir: &Checkpoint, name: &str, e: &Value, decoder: &Decoder) -> io::Result<(usize, usize, Vec<u16>)> {
    let what = |why: &str| bad(format!("{name}: {why}"));
    let shape = u64s(e.get("shape")).filter(|s| s.len() == 2).ok_or_else(|| what("no shape in glyd.json"))?;
    let tiers = u64s(e.get("tiers")).filter(|t| t.len() == 3 && t.iter().all(|&x| x <= u32::MAX as u64)).ok_or_else(|| what("no tiers in glyd.json"))?;
    let (rows, cols) = (shape[0] as usize, shape[1] as usize);
    if rows == 0 || rows % 64 != 0 || cols == 0 || cols % 16 != 0 || rows > (1 << 40) / cols {
        return Err(what("not a tiered pack's shape"));
    }
    if e.get("layout").and_then(Value::as_str) != Some("mma") {
        return Err(what("a layout this glyd does not read"));
    }
    let key = match name.rsplit_once('.') {
        Some((module, weight)) if e.get("experts").is_some() => format!("{module}.glyd_{weight}_"),
        _ => format!("{name}.glyd_"),
    };
    let read = |b: &str, dtype: &str| -> io::Result<Vec<u8>> {
        let (f, t) = dir.get(&format!("{key}{b}")).ok_or_else(|| what(&format!("no {key}{b} in its safetensors")))?;
        if t.dtype != dtype {
            return Err(what(&format!("{key}{b} is {}, not {dtype}", t.dtype)));
        }
        f.read(t)
    };
    let steps = rows * cols / pack::STEP;
    let (data, blocks, base) = (read("data", "U8")?, read("blocks", "U8")?, read("block_base", "I32")?);
    if data.len() != steps * 1280 || base.len() != (steps + 1) * 4 {
        return Err(what("its buffers are not its shape's"));
    }
    let block_base: Vec<i32> = base.as_chunks::<4>().0.iter().map(|b| i32::from_le_bytes(*b)).collect();
    let p = pack::Tiered { rows, cols, data, blocks, block_base, tiers: [tiers[0] as u32, tiers[1] as u32, tiers[2] as u32] };
    let mut w = vec![0u16; rows * cols];
    match decoder {
        Decoder::Cpu => {
            if !pack::unpack_tiered(&p, &mut w) {
                return Err(what("its blocks do not hold its escapes"));
            }
        }
        Decoder::Gpu(lib, ctx) => {
            // (block_base checked first: the kernel reads 128 bytes before a block, 256 past the last)
            let b = &p.block_base;
            if b.iter().enumerate().any(|(s, &x)| x < 128 || (s > 0 && x < b[s - 1]) || x as usize + 256 > p.blocks.len()) {
                return Err(what("its block_base is not within its blocks"));
            }
            let gpu = |r: crate::Result<()>| r.map_err(|e| bad(format!("{name}: {e}")));
            let (d, bl, bb) = (ctx.upload(&p.data).map_err(bad)?, ctx.upload(&p.blocks).map_err(bad)?, ctx.upload(&p.block_base).map_err(bad)?);
            let out = ctx.alloc(rows * cols * 2).map_err(bad)?;
            let m = crate::Matrix { pack: crate::Pack::Tiered(crate::Tiered { data: d.ptr(), blocks: bl.ptr(), block_base: bb.ptr(), tiers: p.tiers }), rows: rows as i64, cols: cols as i64 };
            // SAFETY: the pack's buffers uploaded whole, their sizes and block_base checked; out holds rows x cols.
            gpu(unsafe { lib.unpack(&m, 0, rows as i64, out.ptr(), 0, crate::Stream::DEFAULT) })?;
            gpu(ctx.synchronize())?;
            gpu(out.read(&mut w))?;
        }
    }
    Ok((rows, cols, w))
}

/// A pack's tensors against glyd.json's sha256 (a merged pack's in their rows' order; an experts' weight as the
/// model holds it, [E, out, in] or transposed [E, in, out]): how many.
fn hashes_match(name: &str, e: &Value, rows: usize, cols: usize, w: &[u16]) -> io::Result<usize> {
    let tensors = e.get("tensors").and_then(Value::as_array).filter(|t| !t.is_empty()).ok_or_else(|| bad(format!("{name}: no tensors in glyd.json")))?;
    let mut row = 0;
    for t in tensors {
        let tn = t.get("name").and_then(Value::as_str).unwrap_or(name);
        let want = t.get("sha256").and_then(Value::as_str).ok_or_else(|| bad(format!("{tn}: no sha256 in glyd.json")))?;
        let shape = u64s(t.get("shape")).ok_or_else(|| bad(format!("{tn}: no shape in glyd.json")))?;
        let got = match (e.get("experts").and_then(Value::as_u64), shape.as_slice()) {
            (Some(n), &[en, a, b]) if en == n && (n * a) as usize * b as usize == rows * cols => {
                let transposed = e.get("transposed").and_then(Value::as_bool).unwrap_or(false);
                if !transposed {
                    sha256_u16(w)
                } else {
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

/// A saved checkpoint (glyd-v1, glyd-v2) checked as `python -m glyd.gpu verify` checks it: every pack decoded (on
/// `threads` threads of the CPU, or on a GPU) and each of its tensors' sha256 compared with glyd.json's. The
/// tensors checked.
pub fn verify(dir: &Path, threads: usize, decoder: &Decoder) -> io::Result<usize> {
    let path = dir.join("glyd.json");
    let text = std::fs::read_to_string(&path).map_err(|e| io::Error::new(e.kind(), format!("{}: {e}", path.display())))?;
    let m = json::parse(&text).map_err(|e| bad(format!("{}: {e}", path.display())))?;
    let format = m.get("format").and_then(Value::as_str).unwrap_or("");
    if !FORMATS.contains(&format) {
        return Err(bad(format!("{}: format {format:?}; this glyd reads {}", path.display(), FORMATS.join(" and "))));
    }
    let saved = Checkpoint::open(dir)?;
    let packs: Vec<(String, Value)> = m.get("packs").and_then(Value::as_object).ok_or_else(|| bad(format!("{}: no packs", path.display())))?.to_vec();
    let mut n = 0;
    let one = |(name, e): &(String, Value)| decode(&saved, name, e, decoder).and_then(|(r, c, w)| hashes_match(name, e, r, c, &w));
    match decoder {
        Decoder::Cpu => in_order(&packs, &vec![false; packs.len()], threads, 2 * threads.max(1), one, |k| {
            n += k;
            Ok(())
        })?,
        Decoder::Gpu(..) => {
            for p in &packs {
                n += one(p)?;
            }
        }
    }
    Ok(n)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn file_patterns() {
        assert!(matches("tokenizer*", "tokenizer.json") && matches("*.model", "spm.model") && matches("vocab*", "vocab.json"));
        assert!(!matches("*.model", "model") && !matches("config.json", "config.json.bak"));
    }

    /// A tiny Qwen3 checkpoint (random bf16 weights, two layers, the output layer tied), saved in shards of 64 KB
    /// and verified: glyd.json's packs in save_pretrained's order (a layer's o_proj, then q, k, v as one; down_proj,
    /// then gate, up), every tensor's sha256 the checkpoint's; one byte of a pack changed, refused.
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
            Saved { name: name.into(), dtype: "BF16", shape: shape.to_vec(), bytes: u16_bytes(&w) }
        };
        let mut ts = vec![t("model.embed_tokens.weight", &[256, 128]), t("model.norm.weight", &[128])];
        for i in 0..2 {
            for (n, s) in [("self_attn.q_proj.weight", &[128, 128][..]), ("self_attn.k_proj.weight", &[64, 128]), ("self_attn.v_proj.weight", &[64, 128]), ("self_attn.o_proj.weight", &[128, 128]), ("self_attn.q_norm.weight", &[32]), ("self_attn.k_norm.weight", &[32]), ("mlp.gate_proj.weight", &[256, 128]), ("mlp.up_proj.weight", &[256, 128]), ("mlp.down_proj.weight", &[128, 256]), ("input_layernorm.weight", &[128]), ("post_attention_layernorm.weight", &[128])] {
                ts.push(t(&format!("model.layers.{i}.{n}"), s));
            }
        }
        write_safetensors(&src.join("model.safetensors"), &mut ts).unwrap();
        let source = Source::find(src.to_str().unwrap()).unwrap();
        let s = save(&source, &out, 3, 64 << 10, true).unwrap();
        assert_eq!((s.checked, s.shards > 1), (14, true));
        let m = json::parse(&std::fs::read_to_string(out.join("glyd.json")).unwrap()).unwrap();
        let packs: Vec<&str> = m.get("packs").unwrap().as_object().unwrap().iter().map(|(k, _)| k.as_str()).collect();
        assert_eq!(&packs[..4], &["model.layers.0.self_attn.o_proj", "model.layers.0.self_attn.q_proj", "model.layers.0.mlp.down_proj", "model.layers.0.mlp.gate_proj"]);
        assert_eq!(m.get("format").and_then(Value::as_str), Some("glyd-v1"));
        assert!(out.join("model.safetensors.index.json").exists() && out.join("tokenizer.json").exists());
        assert_eq!(verify(&out, 2, &Decoder::Cpu).unwrap(), 14);
        // one byte of a pack's data changed: its tensor's sha256 is not glyd.json's
        let c = Checkpoint::open(&out).unwrap();
        let (f, info) = c.get("model.layers.1.mlp.down_proj.glyd_data").unwrap();
        let mut bytes = std::fs::read(&f.path).unwrap();
        bytes[info.offset as usize + 300] ^= 0x10;
        std::fs::write(&f.path, bytes).unwrap();
        let e = verify(&out, 2, &Decoder::Cpu).unwrap_err().to_string();
        assert!(e.contains("down_proj.weight decodes to other bytes"), "{e}");
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
