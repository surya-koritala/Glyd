//! A Qwen3 checkpoint packed by Glyd, in GPU memory: every matrix a pack in the smaller of Glyd's two layouts (the tiered `mma` one, or the 12-bit
//! `mma12` one for a matrix that takes fewer bytes in it: each pack has its own, as `glyd-gpu pack`'s glyd-v4 directory says), the embedding too (and
//! the LM head, the same pack where the model ties them), the norms bf16, the rotary tables. The files are that directory plus `engine.safetensors`,
//! which `pack.rs` writes: the embedding packed.

use crate::dev::{copy_h2d_async, Buf, Pinned};
use crate::ffi::Words;
use crate::ffi::{err, Error, Pack, Result, Stream};
use crate::qwen35::Pack35;
use glyd_gpu::json::{self, Value};
use glyd_gpu::safetensors::Checkpoint;
pub use glyd_gpu::save::Layout;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::mpsc;

fn io<T>(r: std::io::Result<T>) -> Result<T> {
    r.map_err(|e| Error(e.to_string()))
}

pub(crate) fn num(v: &Value, what: &str) -> Result<f64> {
    match v {
        Value::Number(n) => n.parse().map_err(|_| Error(format!("config.json: {what}: not a number"))),
        _ => err(format!("config.json: {what}: not a number")),
    }
}

/// The end-of-sequence ids: generation_config.json's (a number or a list) where it gives any, else the config's.
pub fn eos_ids(config: &Value, generation: Option<&str>) -> Vec<i64> {
    let mut eos = Vec::new();
    if let Some(Ok(g)) = generation.map(json::parse) {
        match g.get("eos_token_id") {
            Some(Value::Number(n)) => eos.extend(n.parse::<i64>().ok()),
            Some(Value::Array(a)) => eos.extend(a.iter().filter_map(Value::as_i64)),
            _ => {}
        }
    }
    if eos.is_empty() {
        eos.extend(config.get("eos_token_id").and_then(Value::as_i64));
    }
    eos
}

/// What the engine reads of a Qwen3 config.json.
#[derive(Clone, Debug)]
pub struct Config {
    pub hidden: usize,
    pub layers: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub inter: usize,
    pub vocab: usize,
    pub eps: f32,
    pub rope_theta: f64,
    pub max_pos: usize,
    pub tie: bool,
    pub eos: Vec<i64>,
}

impl Config {
    pub fn load(dir: &Path) -> Result<Config> {
        let p = dir.join("config.json");
        let text = io(std::fs::read_to_string(&p))?;
        let generation = std::fs::read_to_string(dir.join("generation_config.json")).ok();
        Config::parse(&text, &p.display().to_string(), generation.as_deref())
    }

    /// A config.json's text (and a generation_config.json's, where there is one) read as `load` reads the files: `label` names the file in an error.
    pub fn parse(text: &str, label: &str, generation: Option<&str>) -> Result<Config> {
        let v = json::parse(text).map_err(|e| Error(format!("{label}: {e}")))?;
        let get = |k: &str| v.get(k).ok_or_else(|| Error(format!("{label}: no {k}")));
        let arch = get("architectures")?.as_array().and_then(|a| a.first()).and_then(Value::as_str).unwrap_or("");
        if arch != "Qwen3ForCausalLM" {
            return err(format!("{label}: {arch}: the engine runs Qwen3 dense (Qwen3ForCausalLM) so far"));
        }
        let u = |k: &str| -> Result<usize> { get(k)?.as_u64().map(|x| x as usize).ok_or_else(|| Error(format!("{label}: {k}: not an integer"))) };
        let heads = u("num_attention_heads")?;
        if heads == 0 {
            return err(format!("{label}: num_attention_heads: 0"));
        }
        let head_dim = v.get("head_dim").and_then(Value::as_u64).map(|x| x as usize).unwrap_or(u("hidden_size")? / heads);
        if v.get("rope_scaling").is_some_and(|r| *r != Value::Null) {
            return err(format!("{label}: rope_scaling: not supported yet"));
        }
        let eos = eos_ids(&v, generation);
        let layers = u("num_hidden_layers")?;
        if layers > crate::plan::MAX_LAYERS {
            return err(format!("{label}: num_hidden_layers {layers}, where no model has more than {}", crate::plan::MAX_LAYERS));
        }
        Ok(Config {
            hidden: u("hidden_size")?,
            layers,
            heads,
            kv_heads: u("num_key_value_heads")?,
            head_dim,
            inter: u("intermediate_size")?,
            vocab: u("vocab_size")?,
            eps: num(get("rms_norm_eps")?, "rms_norm_eps")? as f32,
            rope_theta: num(get("rope_theta")?, "rope_theta")?,
            max_pos: u("max_position_embeddings")?,
            tie: v.get("tie_word_embeddings").and_then(Value::as_bool).unwrap_or(false),
            eos,
        })
    }
}

impl Config {
    /// What the model's matrices take in bf16 (the layers' and the embedding, and the LM head where it is its own): the Linears the layouts pack.
    pub fn matrix_bytes(&self) -> u64 {
        let (h, d, i, kv) = (self.hidden as u64, self.head_dim as u64, self.inter as u64, self.kv_heads as u64);
        let layer = (self.heads as u64 + 2 * kv) * d * h + h * self.heads as u64 * d + 2 * i * h + h * i;
        2 * (self.layers as u64 * layer + self.vocab as u64 * h * if self.tie { 1 } else { 2 })
    }
}

/// The layout a packed directory (`glyd.json` beside its shards) was packed for: its own label ("mma", the smallest: each matrix in the smaller of
/// the two layouts; "mma12": every matrix in the 12-bit one), else its first matrix's.
pub fn packed_layout(dir: &Path) -> Option<Layout> {
    let v = json::parse(&std::fs::read_to_string(dir.join("glyd.json")).ok()?).ok()?;
    let name = |s: &str| match s {
        "mma" => Some(Layout::Tiered),
        "mma12" => Some(Layout::Twelve),
        _ => None,
    };
    v.get("layout").and_then(Value::as_str).and_then(name).or_else(|| name(v.get("packs")?.get("model.layers.0.self_attn.q_proj")?.get("layout")?.as_str()?))
}

/// How a directory that is not read is said to be: made by a glyd whose tiered layout was the earlier one (0.27 and before), which the library no longer multiplies by.
pub const EARLIER_FORMAT: &str = "packed by glyd 0.27 or before, in a layout the library no longer reads";

/// The key of glyd.json (and of the embedding's sidecar) that holds the words of a pack of the smaller layout: a name of the format.
pub const WORDS_KEY: &str = "tiers";

/// Whether the packed directory `dir` holds packs of the earlier tiered layout (its `glyd.json`'s tiered packs, or the embedding's in `engine.safetensors`, have 0xFF in the top byte of their first tier): why, or None.
pub fn earlier_format(dir: &Path) -> Option<String> {
    let old = |tiers: Option<Vec<u64>>| tiers.is_some_and(|t| t.first().is_some_and(|&t0| (t0 >> 24) as u32 == glyd_gpu::v1::TAG));
    if let Ok(text) = std::fs::read_to_string(dir.join("glyd.json")) {
        if let Ok(v) = json::parse(&text) {
            if let Some(Value::Object(packs)) = v.get("packs") {
                if packs.iter().any(|(_, e)| old(e.get(WORDS_KEY).and_then(Value::as_u64s))) {
                    return Some(EARLIER_FORMAT.to_string());
                }
            }
        }
    }
    if let Ok(f) = glyd_gpu::safetensors::File::open(&dir.join("engine.safetensors")) {
        if let Some((_, m)) = f.metadata.iter().find(|(k, _)| k == "embed") {
            if json::parse(m).ok().is_some_and(|m| old(m.get(WORDS_KEY).and_then(Value::as_u64s))) {
                return Some(EARLIER_FORMAT.to_string());
            }
        }
    }
    None
}

/// Whether the smallest (tiered, 10.8 bits a weight) layout also decodes the fastest on this GPU: measured on Ada (RTX 40, L4, L40S, 2026-09-26), where it is as fast as the 12-bit one and a third
/// smaller. On every other GPU measured (A10, A100, H100, GH200) the 12-bit layout decodes faster (Blackwell is treated as those until it is measured). This is the one line to change when a smaller
/// layout that is fast on the data-center GPUs lands.
pub fn smallest_is_fastest(cc: (i32, i32)) -> bool {
    cc == (8, 9)
}

/// The layout for this GPU, and why: the plugin's `best_layout` for a dense model. The tiered layout where it is the fastest (`smallest_is_fastest`) or the only one that fits (2 GiB of the GPU kept
/// for activations and the cache), else the 12-bit one.
pub fn best_layout(cc: (i32, i32), total_mem: u64, matrix_bytes: u64, other_bytes: u64) -> (Layout, String) {
    let room = total_mem.saturating_sub(2 << 30);
    let (tiered, twelve) = ((matrix_bytes as f64 * 10.80 / 16.0) as u64 + other_bytes, (matrix_bytes as f64 * 12.04 / 16.0) as u64 + other_bytes);
    if twelve > room && room >= tiered {
        return (
            Layout::Tiered,
            format!(
                "only the mma layout fits (the smallest: {:.1} GB; mma12 {:.1} GB, room {:.1} GB)",
                tiered as f64 / 1e9,
                twelve as f64 / 1e9,
                room as f64 / 1e9
            ),
        );
    }
    if smallest_is_fastest(cc) {
        return (Layout::Tiered, "Ada: mma, the smallest layout, is as fast as mma12 (measured)".to_string());
    }
    (Layout::Twelve, "mma12, the faster layout, on this GPU (measured)".to_string())
}

/// The layout of a run: the one asked for with `--layout` (what the user wants wins, whether or not it fits: the caller's fit check says so), else this GPU's.
pub fn choose_layout(asked: Option<Layout>, cc: (i32, i32), total_mem: u64, matrix_bytes: u64, other_bytes: u64) -> (Layout, String) {
    match asked {
        Some(l) => (l, "as asked".to_string()),
        None => best_layout(cc, total_mem, matrix_bytes, other_bytes),
    }
}

/// A pack and the device buffers it points into.
pub struct PackBuf {
    pub pack: Pack,
    pub bytes: usize,
    /// The whole matrix's rows where this pack is this GPU's slice of them (src/dynamic/tp.rs: a product is gathered after it); 0: the whole matrix.
    pub tp_rows: usize,
    _bufs: [Buf; 3],
}

pub struct Layer {
    pub qkv: PackBuf,
    pub o: PackBuf,
    pub gate_up: PackBuf,
    pub down: PackBuf,
    pub in_norm: Buf,
    pub post_norm: Buf,
    pub q_norm: Buf,
    pub k_norm: Buf,
}

pub struct Weights {
    pub cfg: Config,
    /// The layout the directory was packed for (the tiered one means each matrix is in the smaller of the two: see each pack's `is_twelve`).
    pub layout: Layout,
    pub layers: Vec<Layer>,
    pub embed: PackBuf,
    /// The LM head where the model has its own; none where it is the embedding's pack.
    pub head: Option<PackBuf>,
    pub norm: Buf,
    pub cos: Buf,
    pub sin: Buf,
    /// Device bytes of every weight buffer (packs, norms, rotary tables).
    pub bytes: usize,
    /// Bytes bf16 would take for the same matrices and norms (the checkpoint's own size).
    pub bf16_bytes: usize,
    /// Per kind of matrix: (name, packed bytes, weights).
    pub kinds: Vec<(&'static str, usize, usize)>,
    pub dir: PathBuf,
}

impl Weights {
    pub fn lm_head(&self) -> &PackBuf {
        self.head.as_ref().unwrap_or(&self.embed)
    }
}

/// A tensor to load: its name, and where its bytes are (file, offset, length).
#[derive(Clone)]
struct Job {
    name: String,
    path: PathBuf,
    offset: u64,
    bytes: usize,
}

/// Bytes a pinned staging buffer holds: a tensor is read and copied in pieces of this size.
const PIECE: usize = 128 << 20;

struct Raw(*mut u8);
// SAFETY: a pointer into a pinned buffer that the reader thread alone writes while the main thread does not read it (a buffer is handed over by a channel).
unsafe impl Send for Raw {}

/// The tensors of `jobs` read from their files and copied to device memory, in order: a reader thread fills pinned buffers while this thread
/// copies the one before to the GPU, so a load goes at the slower of the file and the bus. Returns each tensor's device buffer, by name.
fn load_all(jobs: &[Job], stream: Stream) -> Result<std::vec::IntoIter<(String, Buf)>> {
    const NBUF: usize = 3;
    let pieces: Vec<(usize, usize, usize)> =
        jobs.iter().enumerate().flat_map(|(i, j)| (0..j.bytes.div_ceil(PIECE)).map(move |k| (i, k * PIECE, PIECE.min(j.bytes - k * PIECE)))).collect();
    let mut pins = Vec::new();
    for _ in 0..NBUF {
        pins.push(Pinned::new(PIECE)?);
    }
    let raw: Vec<Raw> = pins.iter().map(|p| Raw(p.ptr as *mut u8)).collect();
    let (ready_tx, ready_rx) = mpsc::channel::<std::result::Result<(usize, usize), String>>();
    let (free_tx, free_rx) = mpsc::channel::<usize>();
    for b in 0..NBUF {
        free_tx.send(b).ok();
    }
    let (jobs_t, pieces_t) = (jobs.to_vec(), pieces.clone());
    let reader = std::thread::spawn(move || {
        use crate::os::FileExt;
        let raw = raw;
        let mut files: HashMap<PathBuf, std::fs::File> = HashMap::new();
        for (pi, &(ji, at, len)) in pieces_t.iter().enumerate() {
            let Ok(b) = free_rx.recv() else { return };
            let j = &jobs_t[ji];
            if !files.contains_key(&j.path) {
                match std::fs::File::open(&j.path) {
                    Ok(f) => {
                        files.insert(j.path.clone(), f);
                    }
                    Err(e) => {
                        ready_tx.send(Err(format!("{}: {e}", j.path.display()))).ok();
                        return;
                    }
                }
            }
            // SAFETY: the buffer is PIECE bytes, free (its last copy done), and read by nobody until it is sent.
            let buf = unsafe { std::slice::from_raw_parts_mut(raw[b].0, len) };
            if let Err(e) = files[&j.path].read_exact_at(buf, j.offset + at as u64) {
                ready_tx.send(Err(format!("{} ({}): {e}", j.path.display(), j.name))).ok();
                return;
            }
            if ready_tx.send(Ok((pi, b))).is_err() {
                return;
            }
        }
    });
    let mut out: Vec<(String, Buf)> = Vec::with_capacity(jobs.len());
    let mut cur: Option<(usize, Buf)> = None;
    let mut result = Ok(());
    for &(ji, at, len) in &pieces {
        let (_, b) = match ready_rx.recv() {
            Ok(Ok(x)) => x,
            Ok(Err(e)) => {
                result = err(e);
                break;
            }
            Err(_) => {
                result = err("the weights' reader stopped");
                break;
            }
        };
        if at == 0 {
            if let Some((cj, done)) = cur.take() {
                out.push((jobs[cj].name.clone(), done));
            }
            match Buf::new(jobs[ji].bytes) {
                Ok(x) => cur = Some((ji, x)),
                Err(e) => {
                    result = Err(e);
                    break;
                }
            }
        }
        let dst = cur.as_ref().map_or(0, |c| c.1.ptr) + at as u64;
        // SAFETY: the pinned buffer holds `len` bytes just read; the copy completes at the synchronize, before the buffer is handed back.
        let r = unsafe { copy_h2d_async(dst, pins[b].ptr, len, stream) }.and_then(|_| {
            // SAFETY: a stream of this device.
            unsafe { crate::ffi::cu("cuStreamSynchronize", (crate::ffi::driver()?.cuStreamSynchronize)(stream)) }
        });
        if let Err(e) = r {
            result = Err(e);
            break;
        }
        free_tx.send(b).ok();
    }
    drop(free_tx);
    drop(ready_rx);
    reader.join().ok();
    result?;
    if let Some((cj, done)) = cur.take() {
        out.push((jobs[cj].name.clone(), done));
    }
    // (a tensor of no bytes has no piece: it is an empty buffer)
    let mut full: Vec<(String, Buf)> = Vec::with_capacity(jobs.len());
    let mut it = out.into_iter();
    for j in jobs {
        if j.bytes == 0 {
            full.push((j.name.clone(), Buf::new(0)?));
        } else {
            match it.next() {
                Some(x) if x.0 == j.name => full.push(x),
                _ => return err(format!("{}: not loaded", j.name)),
            }
        }
    }
    Ok(full.into_iter())
}

/// The loaded buffers in the order the loader asked for them.
struct Feed(std::vec::IntoIter<(String, Buf)>);

impl Feed {
    fn take(&mut self, name: &str) -> Result<Buf> {
        match self.0.next() {
            Some((n, b)) if n == name => Ok(b),
            Some((n, _)) => err(format!("loading {name}, the next loaded is {n}")),
            None => err(format!("loading {name}: nothing left")),
        }
    }
}

/// The suffixes of a pack's three buffers in a layout.
fn suffixes(layout: Layout) -> [&'static str; 3] {
    match layout {
        Layout::Cb3Row | Layout::Cb2Row => ["data", "blocks", "codebook"],
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
        | Layout::Hummingbird => ["data", "blocks", "block_base"],
        Layout::Twelve => ["data", "exc", "exc_base"],
    }
}

/// A pack's words from its manifest entry: the tiered layout's three tiers, or the 12-bit layout's base (the library's four words).
fn words_of(layout: Layout, e: &Value, name: &str) -> Result<([u64; 3], [u32; 4])> {
    match layout {
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
        | Layout::Hummingbird => {
            let t = e
                .get(WORDS_KEY)
                .and_then(Value::as_u64s)
                .filter(|t| t.len() == 3)
                .ok_or_else(|| Error(format!("glyd.json: {name}: the pack's words are missing")))?;
            let tag = (t[0] >> 24) as u32;
            if tag == glyd_gpu::v1::TAG {
                return err(format!("glyd.json: {name}: {EARLIER_FORMAT}: pack the model again"));
            }
            if tag != glyd_gpu::pack::TIERED_TAG {
                return err(format!("glyd.json: {name}: words of a layout this engine does not read (number {tag})"));
            }
            Ok(([t[0], t[1], t[2]], [0; 4]))
        }
        Layout::Twelve => {
            let hb = e.get("hb").and_then(Value::as_u64).filter(|&h| h <= 120).ok_or_else(|| Error(format!("glyd.json: {name}: no base (hb)")))?;
            Ok(([0; 3], glyd_gpu::pack::twelve_words(hb as u8)))
        }
    }
}

/// A layout's name in a manifest.
fn layout_named(name: Option<&str>, what: &str) -> Result<Layout> {
    match name {
        Some("mma") => Ok(Layout::Tiered),
        Some("mma12") => Ok(Layout::Twelve),
        other => err(format!("{what}: layout {other:?}, where the engine reads the layouts `mma` and `mma12`")),
    }
}

/// One pack of a packed directory as its manifest says it is stored: its key, its own layout, its matrix's shape and its words.
#[derive(Clone, Debug)]
pub struct PackPlan {
    pub key: String,
    pub layout: Layout,
    pub rows: i64,
    pub cols: i64,
    words: ([u64; 3], [u32; 4]),
}

/// What a packed directory holds for the engine, read from its manifest and its files' headers alone (no GPU): its packs in the order they load (every layer's q/k/v, o, gate/up and down, then the
/// embedding and the LM head where the model has its own), each in its own layout, and the tensors that make the load (the packs' three buffers, the norms) in order.
pub struct Plan {
    /// The layout the directory was packed for.
    pub layout: Layout,
    pub packs: Vec<PackPlan>,
    jobs: Vec<Job>,
}

impl Plan {
    /// The tensors' names in load order.
    pub fn tensors(&self) -> Vec<String> {
        self.jobs.iter().map(|j| j.name.clone()).collect()
    }
}

/// The directory `dir` as the engine loads it (`Weights::load`): errors where it is not a packed Qwen3 of this engine's formats, or holds packs of the earlier tiered layout.
pub fn plan(dir: &Path) -> Result<Plan> {
    let cfg = Config::load(dir)?;
    let gj = dir.join("glyd.json");
    let manifest = json::parse(&io(std::fs::read_to_string(&gj))?).map_err(|e| Error(format!("{}: {e}", gj.display())))?;
    let packs = manifest.get("packs").ok_or_else(|| Error(format!("{}: no packs", gj.display())))?;
    let ck = io(Checkpoint::open(dir))?;
    let ep = dir.join("engine.safetensors");
    let ef = io(glyd_gpu::safetensors::File::open(&ep)).map_err(|e| Error(format!("{e} (run `glyd pack` first)")))?;
    let (h, kv, d, i) = (cfg.hidden, cfg.kv_heads, cfg.head_dim, cfg.inter);
    let layout = packed_layout(dir).ok_or_else(|| Error(format!("{}: no layout of the packs", gj.display())))?;
    let mut jobs: Vec<Job> = Vec::new();
    let mut plans: Vec<PackPlan> = Vec::new();
    let mut want = |sidecar: bool, name: String| -> Result<()> {
        let (path, t) = if sidecar {
            (ef.path.clone(), ef.get(&name).ok_or_else(|| Error(format!("{}: no tensor {name}", ep.display())))?)
        } else {
            let (f, t) = ck.get(&name).ok_or_else(|| Error(format!("{}: no tensor {name}", dir.display())))?;
            (f.path.clone(), t)
        };
        jobs.push(Job { name, path, offset: t.offset, bytes: t.bytes as usize });
        Ok(())
    };
    // a pack of glyd.json: its own layout, its shape and words; its three buffers wanted by that layout's names
    let pack = |key: &str, want: &mut dyn FnMut(bool, String) -> Result<()>, expect: Option<(usize, usize)>| -> Result<PackPlan> {
        let e = packs.get(key).ok_or_else(|| Error(format!("{}: no pack {key}", gj.display())))?;
        let lay = layout_named(e.get("layout").and_then(Value::as_str), &format!("{}: {key}", gj.display()))?;
        let shape = e.get("shape").and_then(Value::as_u64s).filter(|s| s.len() == 2).ok_or_else(|| Error(format!("{}: {key}: no shape", gj.display())))?;
        if let Some((rows, cols)) = expect {
            if shape[0] as usize != rows || shape[1] as usize != cols {
                return err(format!(
                    "{key}: a pack of {} x {}, where Qwen3 with this config has {rows} x {cols} (merged q, k, v and gate, up?)",
                    shape[0], shape[1]
                ));
            }
        }
        for sf in suffixes(lay) {
            want(false, format!("{key}.glyd_{sf}"))?;
        }
        Ok(PackPlan { key: key.to_string(), layout: lay, rows: shape[0] as i64, cols: shape[1] as i64, words: words_of(lay, e, key)? })
    };
    for l in 0..cfg.layers {
        let p = format!("model.layers.{l}");
        plans.push(pack(&format!("{p}.self_attn.q_proj"), &mut want, Some(((cfg.heads + 2 * kv) * d, h)))?);
        plans.push(pack(&format!("{p}.self_attn.o_proj"), &mut want, Some((h, cfg.heads * d)))?);
        plans.push(pack(&format!("{p}.mlp.gate_proj"), &mut want, Some((2 * i, h)))?);
        plans.push(pack(&format!("{p}.mlp.down_proj"), &mut want, Some((h, i)))?);
        for nm in ["input_layernorm", "post_attention_layernorm", "self_attn.q_norm", "self_attn.k_norm"] {
            want(false, format!("{p}.{nm}.weight"))?;
        }
    }
    want(false, "model.norm.weight".to_string())?;
    // the embedding: engine.safetensors (its metadata names its layout; an earlier engine's, which does not, is by its words)
    let meta = ef.metadata.iter().find(|(k, _)| k == "embed").map(|(_, v)| v.clone()).ok_or_else(|| Error(format!("{}: no embed entry", ep.display())))?;
    let m = json::parse(&meta).map_err(|e| Error(format!("{}: embed: {e}", ep.display())))?;
    let shape = m.get("shape").and_then(Value::as_u64s).filter(|s| s.len() == 2).ok_or_else(|| Error(format!("{}: embed: no shape", ep.display())))?;
    if shape[0] as usize != cfg.vocab || shape[1] as usize != h {
        return err(format!("{}: the embedding is {} x {}, where the config says {} x {h}", ep.display(), shape[0], shape[1], cfg.vocab));
    }
    let elay = match m.get("layout").and_then(Value::as_str) {
        Some(n) => layout_named(Some(n), &format!("{}: embed", ep.display()))?,
        None if m.get("hb").is_some() => Layout::Twelve,
        None => Layout::Tiered,
    };
    for sf in suffixes(elay) {
        want(true, format!("embed.glyd_{sf}"))?;
    }
    plans.push(PackPlan { key: "embed".to_string(), layout: elay, rows: shape[0] as i64, cols: shape[1] as i64, words: words_of(elay, &m, "embed")? });
    if !cfg.tie {
        plans.push(pack("lm_head", &mut want, Some((cfg.vocab, h)))?);
    }
    Ok(Plan { layout, packs: plans, jobs })
}

fn pack_from(feed: &mut Feed, key: &str, rows: i64, cols: i64, layout: Layout, words: ([u64; 3], [u32; 4])) -> Result<PackBuf> {
    let sfx = suffixes(layout);
    let (data, b1, b2) =
        (feed.take(&format!("{key}.glyd_{}", sfx[0]))?, feed.take(&format!("{key}.glyd_{}", sfx[1]))?, feed.take(&format!("{key}.glyd_{}", sfx[2]))?);
    let bytes = data.bytes + b1.bytes + b2.bytes;
    let w = match layout {
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
        | Layout::Hummingbird => Words::Tiered { blocks: b1.ptr, block_base: b2.ptr, tiers: [words.0[0] as u32, words.0[1] as u32, words.0[2] as u32] },
        Layout::Twelve => Words::Twelve { exc: b1.ptr, exc_base: b2.ptr, sym: words.1 },
    };
    let pack = Pack { data: data.ptr, words: w, rows, cols };
    Ok(PackBuf { pack, bytes, tp_rows: 0, _bufs: [data, b1, b2] })
}

/// The rotary tables as torch computes them, in fp32, then rounded to bf16 as vLLM's cache is: cos and sin [max_pos, dim / 2].
pub fn rope_tables(theta: f64, dim: usize, max_pos: usize) -> (Vec<u16>, Vec<u16>) {
    let half = dim / 2;
    let inv: Vec<f32> = (0..half).map(|i| 1.0f32 / (theta as f32).powf((2 * i) as f32 / dim as f32)).collect();
    let bf = |x: f32| -> u16 {
        // round to nearest even
        let b = x.to_bits();
        let r = b.wrapping_add(0x7fff + ((b >> 16) & 1));
        (r >> 16) as u16
    };
    let mut cos = Vec::with_capacity(max_pos * half);
    let mut sin = Vec::with_capacity(max_pos * half);
    for p in 0..max_pos {
        for &f in &inv {
            let a = p as f32 * f;
            cos.push(bf(a.cos()));
            sin.push(bf(a.sin()));
        }
    }
    (cos, sin)
}

impl Weights {
    /// `dir` is a directory `glyd-engine pack` made (`plan` says what it holds). `max_pos` caps the rotary tables (positions served). The weights are read by a
    /// thread of their own and copied to the GPU on `stream`.
    pub fn load(dir: &Path, max_pos: Option<usize>, stream: Stream) -> Result<Weights> {
        let cfg = Config::load(dir)?;
        let plan = plan(dir)?;
        let mut feed = Feed(load_all(&plan.jobs, stream)?);
        let mut packs = plan.packs.iter();
        let mut next_pack = |feed: &mut Feed| -> Result<PackBuf> {
            let pp = packs.next().ok_or_else(|| Error("the plan has no pack left".into()))?;
            pack_from(feed, &pp.key, pp.rows, pp.cols, pp.layout, pp.words)
        };
        let mut layers = Vec::with_capacity(cfg.layers);
        for l in 0..cfg.layers {
            let p = format!("model.layers.{l}");
            let qkv = next_pack(&mut feed)?;
            let o = next_pack(&mut feed)?;
            let gate_up = next_pack(&mut feed)?;
            let down = next_pack(&mut feed)?;
            let (in_norm, post_norm) = (feed.take(&format!("{p}.input_layernorm.weight"))?, feed.take(&format!("{p}.post_attention_layernorm.weight"))?);
            let (q_norm, k_norm) = (feed.take(&format!("{p}.self_attn.q_norm.weight"))?, feed.take(&format!("{p}.self_attn.k_norm.weight"))?);
            layers.push(Layer { qkv, o, gate_up, down, in_norm, post_norm, q_norm, k_norm });
        }
        let norm = feed.take("model.norm.weight")?;
        let embed = next_pack(&mut feed)?;
        let head = if cfg.tie { None } else { Some(next_pack(&mut feed)?) };
        Weights::finish(cfg, plan.layout, layers, norm, embed, head, max_pos, dir)
    }

    /// The loaded parts as a `Weights`: the bytes they take (and bf16 would), per kind of matrix, and the rotary tables.
    #[allow(clippy::too_many_arguments)]
    fn finish(
        cfg: Config,
        layout: Layout,
        layers: Vec<Layer>,
        norm: Buf,
        embed: PackBuf,
        head: Option<PackBuf>,
        max_pos: Option<usize>,
        dir: &Path,
    ) -> Result<Weights> {
        let (h, d) = (cfg.hidden, cfg.head_dim);
        let (mut bytes, mut bf16_bytes) = (0usize, 0usize);
        let mut kinds: Vec<(&'static str, usize, usize)> = ["qkv", "o", "gate_up", "down", "embed", "lm_head"].iter().map(|&n| (n, 0, 0)).collect();
        for ly in &layers {
            for (k, pk) in [&ly.qkv, &ly.o, &ly.gate_up, &ly.down].into_iter().enumerate() {
                bytes += pk.bytes;
                bf16_bytes += (pk.pack.rows * pk.pack.cols) as usize * 2;
                kinds[k].1 += pk.bytes;
                kinds[k].2 += (pk.pack.rows * pk.pack.cols) as usize;
            }
            for n in [&ly.in_norm, &ly.post_norm, &ly.q_norm, &ly.k_norm] {
                bytes += n.bytes;
                bf16_bytes += n.bytes;
            }
        }
        bytes += norm.bytes;
        bf16_bytes += norm.bytes;
        bytes += embed.bytes;
        bf16_bytes += cfg.vocab * h * 2;
        kinds[4].1 += embed.bytes;
        kinds[4].2 += cfg.vocab * h;
        if let Some(hp) = &head {
            bytes += hp.bytes;
            bf16_bytes += cfg.vocab * h * 2;
            kinds[5].1 += hp.bytes;
            kinds[5].2 += cfg.vocab * h;
        }
        let mp = max_pos.unwrap_or(cfg.max_pos).min(cfg.max_pos);
        let (c, s) = rope_tables(cfg.rope_theta, d, mp);
        let (cos, sin) = (upload_bytes(as_bytes(&c))?, upload_bytes(as_bytes(&s))?);
        bytes += cos.bytes + sin.bytes;
        Ok(Weights { cfg, layout, layers, embed, head, norm, cos, sin, bytes, bf16_bytes, kinds, dir: dir.to_path_buf() })
    }
}

/// A matrix to pack: the checkpoint's tensors stacked by rows (q, k, v; gate, up; or one).
#[derive(Clone)]
pub(crate) struct Item {
    pub members: Vec<String>,
    pub rows: usize,
    pub cols: usize,
}

/// A packed matrix in host memory: its data and the layout's other two buffers (the tiered one's blocks and block base, the 12-bit one's exceptions and their base), as bytes.
pub(crate) struct Packed {
    pub(crate) data: Vec<u8>,
    pub(crate) b1: Vec<u8>,
    pub(crate) b2: Vec<u8>,
    pub(crate) words: Words,
    pub(crate) rows: usize,
    pub(crate) cols: usize,
}

fn i32_bytes(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

pub(crate) fn pack_item(ck: &Checkpoint, it: &Item, layout: Layout) -> Result<Packed> {
    pack_item_map(ck, it, layout, &|_| {})
}

/// Apply an explicit development study transform before the unchanged checked lossless packer.
pub(crate) fn pack_item_map(ck: &Checkpoint, it: &Item, layout: Layout, transform: &(dyn Fn(&mut [u16]) + Sync)) -> Result<Packed> {
    let mut w = item_words(ck, it)?;
    transform(&mut w);
    pack_words(&w, it.rows, it.cols, layout, &it.members.join(" + "))
}

/// An item's members stacked, as bf16 words (an FP8 member as the bf16 it dequantises to).
fn item_words(ck: &Checkpoint, it: &Item) -> Result<Vec<u16>> {
    let mut w = vec![0u16; it.rows * it.cols];
    let mut at = 0usize;
    for m in &it.members {
        let (_, t) = ck.get(m).ok_or_else(|| Error(format!("{}: no tensor {m}", ck.dir.display())))?;
        // (a bf16 matrix as it is; an FP8 one with its block scales as the bf16 it dequantises to)
        if !matches!(t.dtype.as_str(), "BF16" | "F8_E4M3") || t.shape.len() != 2 || t.shape[1] as usize != it.cols {
            return err(format!("{m}: {} {:?}, where a bf16 matrix of {} columns is wanted", t.dtype, t.shape, it.cols));
        }
        let n = t.shape[0] as usize * it.cols;
        io(ck.read_bf16(m, &mut w[at..at + n]))?;
        at += n;
    }
    if at != w.len() {
        return err(format!("{}: {at} weights, where {} x {} are wanted", it.members.join(" + "), it.rows, it.cols));
    }
    Ok(w)
}

/// A bf16 matrix [rows, cols] (its words) packed at `layout`; `name` names it in an error.
pub(crate) fn pack_words(w: &[u16], rows: usize, cols: usize, layout: Layout, name: &str) -> Result<Packed> {
    let what = || Error(format!("{name}: decoded to other bits than its weights"));
    match layout {
        // Candidate A (its values made by the caller's transform, pack::round_a): the compact tiered layout, checked by decoding
        Layout::Compact => {
            let p = glyd_gpu::pack::pack_compact_checked(w, rows, cols).ok_or_else(what)?;
            Ok(Packed {
                data: p.data,
                b1: p.blocks,
                b2: i32_bytes(&p.block_base),
                words: Words::Tiered { blocks: 0, block_base: 0, tiers: p.tiers },
                rows,
                cols,
            })
        }
        // Six-bit block codes (lossy: the packer quantizes; its decode is the reference the GPU's matches)
        Layout::Int6 => {
            let p = glyd_gpu::pack::pack_int6(w, rows, cols).ok_or_else(|| {
                Error(format!("{}: not packable in the six-bit layout (a weight that is not finite, or {} columns not a multiple of 32)", name, cols))
            })?;
            Ok(Packed {
                data: p.data,
                b1: p.blocks,
                b2: i32_bytes(&p.block_base),
                words: Words::Tiered { blocks: 0, block_base: 0, tiers: p.tiers },
                rows,
                cols,
            })
        }
        // Five-bit block codes (lossy: the packer quantizes; its decode is the reference the GPU's matches)
        Layout::Int5 => {
            let p = glyd_gpu::pack::pack_int5(w, rows, cols).ok_or_else(|| {
                Error(format!("{}: not packable in the five-bit layout (a weight that is not finite, or {} columns not a multiple of 32)", name, cols))
            })?;
            Ok(Packed {
                data: p.data,
                b1: p.blocks,
                b2: i32_bytes(&p.block_base),
                words: Words::Tiered { blocks: 0, block_base: 0, tiers: p.tiers },
                rows,
                cols,
            })
        }
        // Four-bit block codes (lossy: the packer quantizes; its decode is the reference the GPU's matches)
        Layout::Int4 => {
            let p = glyd_gpu::pack::pack_int4(w, rows, cols).ok_or_else(|| {
                Error(format!("{}: not packable at the wren level (a weight that is not finite, or {} columns not a multiple of 32)", name, cols))
            })?;
            Ok(Packed {
                data: p.data,
                b1: p.blocks,
                b2: i32_bytes(&p.block_base),
                words: Words::Tiered { blocks: 0, block_base: 0, tiers: p.tiers },
                rows,
                cols,
            })
        }
        // Exact four-bit block codes (lossless: a checkpoint trained to four bits, held bit for bit); a matrix that is not one keeps the smallest lossless layout
        Layout::Int4H => match glyd_gpu::pack::pack_int4h_checked(w, rows, cols) {
            Some(p) => Ok(Packed {
                data: p.data,
                b1: p.blocks,
                b2: i32_bytes(&p.block_base),
                words: Words::Tiered { blocks: 0, block_base: 0, tiers: p.tiers },
                rows,
                cols,
            }),
            None => pack_words(w, rows, cols, Layout::Tiered, name),
        },
        // A row layout (lossy): the weight [out, in] transposed, [in, out], one input's weights a row (an experts' down matrix, read for the inputs that are not zero); the pack's rows are the weight's columns
        Layout::Int6Row | Layout::Int5Row | Layout::Int4Row => {
            let bits = layout.row_bits().expect("a row layout");
            let p = glyd_gpu::pack::pack_rows(bits, &glyd_gpu::pack::transpose(w, rows, cols), cols, rows).ok_or_else(|| {
                Error(format!("{}: not packable in a {bits}-bit row layout (a weight that is not finite, or {} outputs not a multiple of 32)", name, rows))
            })?;
            Ok(Packed {
                data: p.data,
                b1: p.blocks,
                b2: i32_bytes(&p.block_base),
                words: Words::Tiered { blocks: 0, block_base: 0, tiers: p.tiers },
                rows: cols,
                cols: rows,
            })
        }
        // The codebook rows and the Hummingbird level hold codes made apart (src/moe_hb.rs reads them from a directory), which no fit of bf16 weights makes
        Layout::Cb3Row | Layout::Cb2Row | Layout::Hummingbird => {
            err(format!("{}: the {} layout holds codes made apart, not a fit of the weights", name, layout.name()))
        }
        // Three-bit block codes (lossy: the packer quantizes; its decode is the reference the GPU's matches)
        Layout::Int3 => {
            let p = glyd_gpu::pack::pack_int3(w, rows, cols).ok_or_else(|| {
                Error(format!("{}: not packable at the hummingbird level (a weight that is not finite, or {} columns not a multiple of 32)", name, cols))
            })?;
            Ok(Packed {
                data: p.data,
                b1: p.blocks,
                b2: i32_bytes(&p.block_base),
                words: Words::Tiered { blocks: 0, block_base: 0, tiers: p.tiers },
                rows,
                cols,
            })
        }
        Layout::Tiered => match glyd_gpu::pack::pack_smallest_checked(w, rows, cols).ok_or_else(what)? {
            // (the smallest layout: a matrix that takes fewer bytes in the 12-bit layout is held in it, as `glyd-gpu pack` saves it)
            glyd_gpu::pack::Packed::Tiered(p) => Ok(Packed {
                data: p.data,
                b1: p.blocks,
                b2: i32_bytes(&p.block_base),
                words: Words::Tiered { blocks: 0, block_base: 0, tiers: p.tiers },
                rows,
                cols,
            }),
            glyd_gpu::pack::Packed::Twelve(p) => Ok(Packed {
                data: p.data,
                b1: i32_bytes(&p.exc),
                b2: i32_bytes(&p.exc_base),
                words: Words::Twelve { exc: 0, exc_base: 0, sym: p.sym },
                rows,
                cols,
            }),
        },
        Layout::Twelve => {
            let p = glyd_gpu::pack::pack_twelve_checked(w, rows, cols).ok_or_else(what)?;
            Ok(Packed { data: p.data, b1: i32_bytes(&p.exc), b2: i32_bytes(&p.exc_base), words: Words::Twelve { exc: 0, exc_base: 0, sym: p.sym }, rows, cols })
        }
    }
}

/// A block layout by its name in glyd.json.
fn block_layout(name: &str) -> Option<Layout> {
    match name {
        "i6" => Some(Layout::Int6),
        "i5" => Some(Layout::Int5),
        "i4" => Some(Layout::Int4),
        "i4x" => Some(Layout::Int4H),
        "i3" => Some(Layout::Int3),
        _ => None,
    }
}

/// A pack's layout by its name in a glyd-v7 entry: a block layout, or a row layout (`i6r`, `i5r`, `i4r`: the weight's transpose, a row for each of its inputs).
fn entry_layout(name: &str) -> Option<Layout> {
    block_layout(name).or_else(|| Layout::by_name(name).filter(|l| l.row_bits().is_some()))
}

/// A glyd-v7 directory (a lossy level's packs, saved): its level's layout and its glyd.json. None where `dir` has no glyd.json or it is another format. The level's layout is the directory's `layout`; a pack's
/// own, its entry's (`read_v7`), is the same but where the level holds some in another (Nemotron 3 Super at wren: the experts in four-bit codes, every other matrix in six-bit ones).
pub fn v7_manifest(dir: &Path) -> Result<Option<(Layout, Value)>> {
    let Ok(text) = std::fs::read_to_string(dir.join("glyd.json")) else { return Ok(None) };
    let m = json::parse(&text).map_err(|e| Error(format!("{}: glyd.json: {e}", dir.display())))?;
    if m.get("format").and_then(Value::as_str) != Some("glyd-v7") {
        return Ok(None);
    }
    let name = m.get("layout").and_then(Value::as_str);
    // (the Hummingbird level, `hb`: the experts in codebook rows, every other matrix in six-bit codes, a level of its own and no block layout's)
    // (penguin, `mma`: lossless packs a planned save made for an engine that reads them itself, Gemma 4's; read_lossless)
    match name.and_then(|n| {
        block_layout(n)
            .or_else(|| (n == Layout::Hummingbird.name()).then_some(Layout::Hummingbird))
            .or_else(|| (n == Layout::Tiered.name()).then_some(Layout::Tiered))
    }) {
        Some(layout) => Ok(Some((layout, m))),
        None => err(format!("{}: glyd.json: layout {name:?}, where a glyd-v7 directory has `i6`, `i5`, `i4`, `i3`, `hb` or `mma`", dir.display())),
    }
}

/// A saved block pack (its glyd.json `shape` and `tiers`, its three buffers' lengths in bytes) against `layout` and the `rows` x `cols` the config gives its matrix: the tag, the shape and the
/// lengths pack_int6, pack_int5, pack_int4 and pack_int3 make (a step of 64 rows and 16 columns takes the layout's step bytes of data, a pair of steps 128 bytes of scales, and one more base than steps takes 4 bytes each).
pub(crate) fn check_v7(key: &str, layout: Layout, rows: usize, cols: usize, shape: &[u64], tiers: &[u64], lens: [u64; 3]) -> Result<()> {
    use glyd_gpu::pack::{
        rows_stride, INT3_STEP_BYTES, INT3_TAG, INT4H_TAG, INT4_ROW_TAG, INT4_STEP_BYTES, INT4_TAG, INT5_ROW_TAG, INT5_STEP_BYTES, INT5_TAG, INT6_ROW_TAG,
        INT6_STEP_BYTES, INT6_TAG,
    };
    // a row layout's pack is the weight's transpose [cols, rows]: a row for each of the weight's inputs, `rows_stride` bytes of codes and a scale for each run of 32 outputs
    if let Some(bits) = layout.row_bits() {
        let tag = [INT4_ROW_TAG, INT5_ROW_TAG, INT6_ROW_TAG][bits as usize - 4];
        if tiers.first().map(|t| (t >> 24) as u32) != Some(tag) {
            return err(format!("glyd.json: {key}: words of tag {:?}, where the {} layout's is {tag}", tiers.first().map(|t| t >> 24), layout.name()));
        }
        if shape != [cols as u64, rows as u64] || rows == 0 || cols == 0 || !rows.is_multiple_of(32) || !(rows * cols).is_multiple_of(1024) {
            return err(format!(
                "glyd.json: {key}: shape {shape:?}, where the config has {rows} x {cols} (a row layout holds {cols} x {rows}, its outputs runs of 32)"
            ));
        }
        let want = [(cols * rows_stride(bits, rows)) as u64, (cols * rows / 32 * 2) as u64, ((rows * cols / 1024 + 1) * 4) as u64];
        if lens != want {
            return err(format!("{key}: buffers of {lens:?} bytes (data, blocks, block_base), where {want:?} are wanted"));
        }
        return Ok(());
    }
    let (tag, step) = match layout {
        Layout::Int6 => (INT6_TAG, INT6_STEP_BYTES),
        Layout::Int5 => (INT5_TAG, INT5_STEP_BYTES),
        Layout::Int4 => (INT4_TAG, INT4_STEP_BYTES),
        Layout::Int4H => (INT4H_TAG, INT4_STEP_BYTES),
        Layout::Int3 => (INT3_TAG, INT3_STEP_BYTES),
        _ => return err(format!("{key}: {} is not a layout glyd-v7 holds", layout.name())),
    };
    if tiers.first().map(|t| (t >> 24) as u32) != Some(tag) {
        return err(format!("glyd.json: {key}: words of tag {:?}, where the {} layout's is {tag}", tiers.first().map(|t| t >> 24), layout.name()));
    }
    if shape != [rows as u64, cols as u64] || rows % 64 != 0 || cols % 32 != 0 {
        return err(format!("glyd.json: {key}: shape {shape:?}, where the config has {rows} x {cols}"));
    }
    let steps = (rows / 64 * (cols / 16)) as u64;
    let want = [steps * step as u64, steps / 2 * 128, (steps + 1) * 4];
    if lens != want {
        return err(format!("{key}: buffers of {lens:?} bytes (data, blocks, block_base), where {want:?} are wanted"));
    }
    Ok(())
}

/// The bytes of the pack saved as `key` in `ck` (its three buffers), None where it is not saved.
pub(crate) fn packed_bytes(ck: &Checkpoint, key: &str) -> Option<u64> {
    // (a codebook row pack's third buffer is its codebook, the others' its block base)
    let third = if ck.get(&format!("{key}.glyd_codebook")).is_some() { "codebook" } else { "block_base" };
    ["data", "blocks", third].iter().map(|s| ck.get(&format!("{key}.glyd_{s}")).map(|t| t.1.bytes)).sum()
}

/// A matrix read from a glyd-v7 directory, each pack checked against its layout and its shape in the config first (`check_v7`): the layout its entry in glyd.json names, else `layout` (the directory's), so that a
/// directory with packs in several block layouts is read as it is saved. The packs saved as `parts` (each its pack key and rows; one where the save fused them, as it does q, k, v and gate, up; else each part of a fused
/// matrix is its own, joined here by rows: a pack is steps of 64 rows, whole row blocks in order, so its buffers join end to end, and in one layout). A pack in a row layout (an experts' down matrix: the
/// weight's transpose, a row for each of its inputs) is one weight alone, and comes back as its pack: `rows` the weight's columns and `cols` its rows.
pub(crate) fn read_v7(ck: &Checkpoint, manifest: &Value, layout: Layout, parts: &[(&str, usize)], cols: usize) -> Result<Packed> {
    let (mut data, mut b1, mut b2, mut tiers, mut rows) = (Vec::new(), Vec::new(), Vec::new(), [0u32; 3], 0);
    let mut joined: Option<Layout> = None;
    for &(key, r) in parts {
        let e = manifest.get("packs").and_then(|p| p.get(key)).ok_or_else(|| Error(format!("glyd.json: no pack {key}")))?;
        let own = match e.get("layout").and_then(Value::as_str) {
            None => layout,
            Some(name) => entry_layout(name)
                .ok_or_else(|| Error(format!("glyd.json: {key}: layout {name:?}, where a glyd-v7 pack has `i6`, `i5`, `i4`, `i3`, `i6r`, `i5r` or `i4r`")))?,
        };
        if *joined.get_or_insert(own) != own {
            return err(format!(
                "glyd.json: {key}: packed in the {} layout, where the part of its matrix before it is in {}",
                own.name(),
                joined.map_or("none", Layout::name)
            ));
        }
        let t = e
            .get(WORDS_KEY)
            .and_then(Value::as_u64s)
            .filter(|t| t.len() == 3)
            .ok_or_else(|| Error(format!("glyd.json: {key}: the pack's words are missing")))?;
        let names = ["data", "blocks", "block_base"].map(|s| format!("{key}.glyd_{s}"));
        let mut lens = [0u64; 3];
        for (l, n) in lens.iter_mut().zip(&names) {
            *l = ck.get(n).ok_or_else(|| Error(format!("{}: no tensor {n}", ck.dir.display())))?.1.bytes;
        }
        check_v7(key, own, r, cols, &e.get("shape").and_then(Value::as_u64s).unwrap_or_default(), &t, lens)?;
        let [d, b, x] = names.map(|n| io(ck.read(&n)));
        data.extend(d?);
        b1.extend(b?);
        // (the base is steps + 1 zeros, not read: the joined pack's has the same count)
        let x = x?;
        b2.extend(&x[if b2.is_empty() { 0 } else { 4.min(x.len()) }..]);
        tiers = [t[0] as u32, t[1] as u32, t[2] as u32];
        rows += r;
    }
    // (a row layout's pack is its weight's transpose, held alone: its rows are the weight's columns, its columns the weight's rows)
    if joined.is_some_and(|l| l.row_bits().is_some()) {
        if parts.len() != 1 {
            return err(format!("glyd.json: {}: a matrix of {} parts in a row layout, which holds one weight", parts[0].0, parts.len()));
        }
        return Ok(Packed { data, b1, b2, words: Words::Tiered { blocks: 0, block_base: 0, tiers }, rows: cols, cols: rows });
    }
    Ok(Packed { data, b1, b2, words: Words::Tiered { blocks: 0, block_base: 0, tiers }, rows, cols })
}

/// A matrix of a lossless (penguin) glyd-v7 directory, saved as one pack under `key`: tiered (`mma`) or 12-bit (`mma12`), whichever the
/// save found smaller, its buffers held to where the GPU's decode reads by them before they are used (the checks `glyd verify` makes).
pub(crate) fn read_lossless(ck: &Checkpoint, manifest: &Value, key: &str, rows: usize, cols: usize) -> Result<Packed> {
    let e = manifest.get("packs").and_then(|p| p.get(key)).ok_or_else(|| Error(format!("glyd.json: no pack {key}")))?;
    let layout = match e.get("layout").and_then(Value::as_str) {
        Some("mma") => Layout::Tiered,
        Some("mma12") => Layout::Twelve,
        other => return err(format!("glyd.json: {key}: layout {other:?}, where a penguin pack is `mma` or `mma12`")),
    };
    if e.get("shape").and_then(Value::as_u64s).unwrap_or_default() != [rows as u64, cols as u64] || rows % 64 != 0 || cols % 16 != 0 {
        return err(format!("glyd.json: {key}: its shape is not the config's {rows} x {cols}"));
    }
    let words = words_of(layout, e, key)?;
    let [data, b1, b2] = suffixes(layout).map(|s| io(ck.read(&format!("{key}.glyd_{s}"))));
    let (data, b1, b2) = (data?, b1?, b2?);
    let steps = rows / 64 * (cols / 16);
    let base: Vec<i32> = b2.as_chunks::<4>().0.iter().map(|b| i32::from_le_bytes(*b)).collect();
    let holds = match layout {
        Layout::Tiered => data.len() == steps * 1280 && glyd_gpu::pack::tiered_blocks_hold(&data, &b1, &base),
        _ => {
            data.len() == steps * 1536
                && b1.len() % 16 == 0
                && base.len() == steps + 1
                && base[0] == 0
                && base.windows(2).all(|x| x[0] <= x[1])
                && base[steps] as usize <= b1.len() / 4
        }
    };
    if !holds {
        return err(format!("{key}: its buffers do not hold what its data says (damaged? glyd verify checks the whole save)"));
    }
    let words = match layout {
        Layout::Tiered => Words::Tiered { blocks: 0, block_base: 0, tiers: [words.0[0] as u32, words.0[1] as u32, words.0[2] as u32] },
        _ => Words::Twelve { exc: 0, exc_base: 0, sym: words.1 },
    };
    Ok(Packed { data, b1, b2, words, rows, cols })
}

/// A lossless directory saved as glyd-v4 in the tiered layout (`glyd-gpu pack`'s save; the glyd/NAME-penguin repos): its glyd.json. None for any other directory (a glyd-v7 `mma` save is `v7_manifest`'s).
pub fn lossless_manifest(dir: &Path) -> Result<Option<Value>> {
    let Ok(text) = std::fs::read_to_string(dir.join("glyd.json")) else { return Ok(None) };
    let m = json::parse(&text).map_err(|e| Error(format!("{}: glyd.json: {e}", dir.display())))?;
    Ok((m.get("format").and_then(Value::as_str) == Some("glyd-v4") && m.get("layout").and_then(Value::as_str) == Some("mma")).then_some(m))
}

/// A matrix of a lossless directory held as the saved packs of `parts` (each its rows): the one pack as saved where the save fused the parts; else each part decoded on the CPU (bit for bit) and the
/// stacked matrix packed here in the tiered layout, checked by decoding. The tiered layout's words are one matrix's own, so two packs cannot be joined as they are (Qwen3.5's in_proj_qkv and in_proj_z).
pub(crate) fn read_lossless_parts(ck: &Checkpoint, manifest: &Value, parts: &[(&str, usize)], cols: usize) -> Result<Packed> {
    if let [(key, rows)] = parts {
        return read_lossless(ck, manifest, key, *rows, cols);
    }
    let i32s = |b: &[u8]| -> Vec<i32> { b.as_chunks::<4>().0.iter().map(|x| i32::from_le_bytes(*x)).collect() };
    let rows: usize = parts.iter().map(|p| p.1).sum();
    let mut w = vec![0u16; rows * cols];
    let mut at = 0;
    for &(key, r) in parts {
        let p = read_lossless(ck, manifest, key, r, cols)?;
        let out = &mut w[at..at + r * cols];
        let ok = match p.words {
            Words::Tiered { tiers, .. } => {
                glyd_gpu::pack::unpack_tiered(&glyd_gpu::pack::Tiered { rows: r, cols, data: p.data, blocks: p.b1, block_base: i32s(&p.b2), tiers }, out)
            }
            Words::Twelve { sym, .. } => {
                glyd_gpu::pack::unpack_twelve(&glyd_gpu::pack::Twelve { rows: r, cols, data: p.data, exc: i32s(&p.b1), exc_base: i32s(&p.b2), sym }, out)
            }
        };
        if !ok {
            return err(format!("{key}: its pack does not decode (damaged? glyd verify checks the whole save)"));
        }
        at += r * cols;
    }
    pack_words(&w, rows, cols, Layout::Tiered, &parts.iter().map(|p| p.0).collect::<Vec<_>>().join(" + "))
}

/// How a glyd-v7 directory holds the pack `pk`: the packs it has saved for its parts, each with its rows. A save fuses a pack's parts under the first one's name (q, k, v), or keeps a pack for each part; none where
/// `ck` holds the pack in neither way (the parts are bf16 there).
pub(crate) fn saved_parts<'a>(ck: &Checkpoint, pk: &'a Pack35) -> Vec<(&'a str, usize)> {
    let each: Vec<(&str, usize)> = pk.parts.iter().map(|x| (Pack35::saved(&x.0), x.1)).collect();
    let has = |k: &str| ck.get(&format!("{k}.glyd_data")).is_some();
    if has(each[0].0) && (each.len() == 1 || !has(each[1].0)) {
        vec![(each[0].0, pk.rows())]
    } else if each.iter().all(|x| has(x.0)) {
        each
    } else {
        Vec::new()
    }
}

/// The pack `pk` in `level`, ready to upload: read from a glyd-v7 directory (`v7`: its level and glyd.json) where it holds the pack saved (in the layout its entry names, else `level`), else packed from the bf16 tensors
/// in `level`.
/// `read_pack` of rows r0..r0 + rn of the matrix alone (a GPU's slice of it, src/dynamic/tp.rs), from the checkpoint (a glyd-v7 directory's saved packs are
/// whole matrices: not sliced); the whole where `rows` is None.
pub(crate) fn read_pack_rows(ck: &Checkpoint, v7: Option<&(Layout, Value)>, pk: &Pack35, level: Layout, rows: Option<(usize, usize)>) -> Result<Packed> {
    let Some((r0, rn)) = rows else { return read_pack(ck, v7, pk, level) };
    if v7.is_some() && !saved_parts(ck, pk).is_empty() {
        return err(format!("{}: a saved pack, which is not sliced across GPUs", pk.key));
    }
    let it = Item { members: pk.parts.iter().map(|x| x.0.clone()).collect(), rows: pk.rows(), cols: pk.cols };
    let w = item_words(ck, &it)?;
    pack_words(&w[r0 * it.cols..(r0 + rn) * it.cols], rn, it.cols, level, &it.members.join(" + "))
}

pub(crate) fn read_pack(ck: &Checkpoint, v7: Option<&(Layout, Value)>, pk: &Pack35, level: Layout) -> Result<Packed> {
    let saved = saved_parts(ck, pk);
    match v7 {
        Some((_, manifest)) if !saved.is_empty() => read_v7(ck, manifest, level, &saved, pk.cols),
        _ => pack_item_map(ck, &Item { members: pk.parts.iter().map(|x| x.0.clone()).collect(), rows: pk.rows(), cols: pk.cols }, level, &|_| {}),
    }
}

/// `n` jobs run by `threads` workers, the results handed to `each` in order as they are ready (a worker waits where the next `window` results are not yet taken).
pub(crate) fn ordered<T: Send>(
    n: usize,
    threads: usize,
    window: usize,
    work: &(dyn Fn(usize) -> Result<T> + Sync),
    mut each: impl FnMut(usize, T) -> Result<()>,
) -> Result<()> {
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::{Condvar, Mutex};
    let slots: Mutex<Vec<Option<Result<T>>>> = Mutex::new((0..n).map(|_| None).collect());
    let cv = Condvar::new();
    let next = AtomicUsize::new(0);
    let taken = AtomicUsize::new(0);
    let stop = AtomicBool::new(false);
    std::thread::scope(|sc| {
        for _ in 0..threads.max(1) {
            sc.spawn(|| loop {
                let i = next.fetch_add(1, Ordering::SeqCst);
                if i >= n {
                    return;
                }
                {
                    let mut g = slots.lock().unwrap();
                    while i >= taken.load(Ordering::SeqCst) + window && !stop.load(Ordering::SeqCst) {
                        g = cv.wait(g).unwrap();
                    }
                }
                if stop.load(Ordering::SeqCst) {
                    return;
                }
                let r = work(i);
                slots.lock().unwrap()[i] = Some(r);
                cv.notify_all();
            });
        }
        let mut result = Ok(());
        for i in 0..n {
            let r = {
                let mut g = slots.lock().unwrap();
                while g[i].is_none() {
                    g = cv.wait(g).unwrap();
                }
                g[i].take().unwrap()
            };
            taken.store(i + 1, Ordering::SeqCst);
            cv.notify_all();
            match r.and_then(|v| each(i, v)) {
                Ok(()) => {}
                Err(e) => {
                    result = Err(e);
                    stop.store(true, Ordering::SeqCst);
                    cv.notify_all();
                    break;
                }
            }
        }
        result
    })
}

impl Weights {
    /// A Hub-format directory (config.json, the bf16 safetensors): every matrix packed here, in memory, by `threads` workers, each copied to the GPU as it is done, nothing written
    /// (the same packs `glyd-engine pack` writes; about 20 s for an 8B model on 12 threads).
    pub fn load_bf16(dir: &Path, max_pos: Option<usize>, threads: usize, layout: Layout) -> Result<Weights> {
        let cfg = Config::load(dir)?;
        let ck = io(Checkpoint::open(dir))?;
        let (h, kv, d, i) = (cfg.hidden, cfg.kv_heads, cfg.head_dim, cfg.inter);
        let mut items: Vec<Item> = vec![Item { members: vec!["model.embed_tokens.weight".into()], rows: cfg.vocab, cols: h }];
        if !cfg.tie {
            items.push(Item { members: vec!["lm_head.weight".into()], rows: cfg.vocab, cols: h });
        }
        let first_layer = items.len();
        for l in 0..cfg.layers {
            let p = format!("model.layers.{l}");
            let m = |names: &[&str]| names.iter().map(|n| format!("{p}.{n}.weight")).collect::<Vec<_>>();
            items.push(Item { members: m(&["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]), rows: (cfg.heads + 2 * kv) * d, cols: h });
            items.push(Item { members: m(&["self_attn.o_proj"]), rows: h, cols: cfg.heads * d });
            items.push(Item { members: m(&["mlp.gate_proj", "mlp.up_proj"]), rows: 2 * i, cols: h });
            items.push(Item { members: m(&["mlp.down_proj"]), rows: h, cols: i });
        }
        let norm_buf = |name: &str| -> Result<Buf> { upload_bytes(&io(ck.read(name))?) };
        let mut packs: Vec<PackBuf> = Vec::with_capacity(items.len());
        ordered(items.len(), threads, 2 * threads.max(1), &|k| pack_item(&ck, &items[k], layout), |_, p| {
            packs.push(upload_packed(p)?);
            Ok(())
        })?;
        let mut it = packs.into_iter();
        let embed = it.next().ok_or_else(|| Error("no embedding".into()))?;
        let head = if cfg.tie { None } else { it.next() };
        debug_assert_eq!(it.len(), 4 * cfg.layers);
        let _ = first_layer;
        let mut layers = Vec::with_capacity(cfg.layers);
        for l in 0..cfg.layers {
            let p = format!("model.layers.{l}");
            let (qkv, o, gate_up, down) = (it.next().unwrap(), it.next().unwrap(), it.next().unwrap(), it.next().unwrap());
            layers.push(Layer {
                qkv,
                o,
                gate_up,
                down,
                in_norm: norm_buf(&format!("{p}.input_layernorm.weight"))?,
                post_norm: norm_buf(&format!("{p}.post_attention_layernorm.weight"))?,
                q_norm: norm_buf(&format!("{p}.self_attn.q_norm.weight"))?,
                k_norm: norm_buf(&format!("{p}.self_attn.k_norm.weight"))?,
            });
        }
        let norm = norm_buf("model.norm.weight")?;
        Weights::finish(cfg, layout, layers, norm, embed, head, max_pos, dir)
    }
}

/// A matrix packed in host memory, copied to the GPU.
pub(crate) fn upload_packed(p: Packed) -> Result<PackBuf> {
    let (data, b1, b2) = (upload_bytes(&p.data)?, upload_bytes(&p.b1)?, upload_bytes(&p.b2)?);
    let bytes = data.bytes + b1.bytes + b2.bytes;
    let words = match p.words {
        Words::Tiered { tiers, .. } => Words::Tiered { blocks: b1.ptr, block_base: b2.ptr, tiers },
        Words::Twelve { sym, .. } => Words::Twelve { exc: b1.ptr, exc_base: b2.ptr, sym },
    };
    Ok(PackBuf { pack: Pack { data: data.ptr, words, rows: p.rows as i64, cols: p.cols as i64 }, bytes, tp_rows: 0, _bufs: [data, b1, b2] })
}

impl PackBuf {
    /// The whole matrix's rows: `tp_rows` where it is a slice of them, else its own.
    pub fn whole_rows(&self) -> usize {
        if self.tp_rows > 0 {
            self.tp_rows
        } else {
            self.pack.rows as usize
        }
    }

    /// A matrix whose bytes are elsewhere on the GPU (src/dense_ram.rs: the slot a layer in the computer's memory is copied into before it runs): its pack, no buffers of its own, no bytes counted.
    pub(crate) fn view(pack: Pack) -> Result<PackBuf> {
        Ok(PackBuf { pack, bytes: 0, tp_rows: 0, _bufs: [Buf::new(16)?, Buf::new(16)?, Buf::new(16)?] })
    }

    /// A matrix's shape and no weights (a matrix held in another format beside it, as `MoeLayer::shared8` holds the shared expert's): never multiplied by.
    pub fn shape_only(rows: usize, cols: usize) -> Result<PackBuf> {
        let bufs = [Buf::new(16)?, Buf::new(16)?, Buf::new(16)?];
        Ok(PackBuf {
            pack: Pack {
                data: bufs[0].ptr,
                words: Words::Tiered { blocks: bufs[1].ptr, block_base: bufs[2].ptr, tiers: [0; 3] },
                rows: rows as i64,
                cols: cols as i64,
            },
            bytes: 0,
            tp_rows: 0,
            _bufs: bufs,
        })
    }
}

/// A pack of glyd-gpu's packer (the tiered layout, Int6, Int5) in host memory, copied to the GPU.
pub fn upload_tiered(p: &glyd_gpu::pack::Tiered) -> Result<PackBuf> {
    upload_packed(Packed {
        data: p.data.clone(),
        b1: p.blocks.clone(),
        b2: i32_bytes(&p.block_base),
        words: Words::Tiered { blocks: 0, block_base: 0, tiers: p.tiers },
        rows: p.rows,
        cols: p.cols,
    })
}

pub(crate) fn upload_bytes(bytes: &[u8]) -> Result<Buf> {
    let b = Buf::new(bytes.len())?;
    b.write(0, bytes)?;
    Ok(b)
}

pub fn as_bytes<T: Copy>(v: &[T]) -> &[u8] {
    // SAFETY: plain data (u16 here), no padding; the bytes of the slice.
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bf16(x: f32) -> u16 {
        let b = x.to_bits();
        (b.wrapping_add(0x7fff + ((b >> 16) & 1)) >> 16) as u16
    }

    /// Position 0 rotates by nothing; position 1's first frequency is 1 radian.
    #[test]
    fn rope_tables_by_position() {
        let (c, s) = rope_tables(1e6, 128, 3);
        assert_eq!(c.len(), 3 * 64);
        assert!(c[..64].iter().all(|&x| x == 0x3F80) && s[..64].iter().all(|&x| x == 0));
        assert_eq!(c[64], bf16(1f32.cos()));
        assert_eq!(s[64], bf16(1f32.sin()));
        // the last channel pair turns slowest: its angle at position 2 is tiny, cos rounds to 1
        assert_eq!(c[2 * 64 + 63], 0x3F80);
    }

    /// The workers' results reach `each` in order, however they finish; a failure stops the run with its error.
    #[test]
    fn ordered_results() {
        let mut got = Vec::new();
        ordered(
            40,
            4,
            6,
            &|i| {
                std::thread::sleep(std::time::Duration::from_micros(((40 - i) * 50) as u64));
                Ok(i * i)
            },
            |i, v| {
                assert_eq!(i, got.len());
                got.push(v);
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(got, (0..40).map(|i| i * i).collect::<Vec<_>>());
        let e = ordered(40, 4, 6, &|i| if i == 17 { err("seventeen") } else { Ok(i) }, |_, _| Ok(())).unwrap_err();
        assert_eq!(e.to_string(), "seventeen");
    }

    /// The layout a GPU is given (the plugin's best_layout): Ada the tiered one, every other GPU the 12-bit one, the tiered one where only it fits; `--layout` wins.
    #[test]
    fn layout_by_gpu() {
        let gib = 1u64 << 30;
        let m = 16_380_000_000u64; // Qwen3-8B's matrices in bf16
        let (t, w) = (Layout::Tiered, Layout::Twelve);
        // (the GPU, its class, its memory, the layout it is given for an 8B model)
        for (gpu, cc, mem, want) in [
            ("L4", (8, 9), 22, t),
            ("RTX 4090", (8, 9), 24, t),
            ("L40S", (8, 9), 48, t),
            ("A10", (8, 6), 24, w),
            ("RTX 3090", (8, 6), 24, w),
            ("A100 40 GB", (8, 0), 40, w),
            ("A100 80 GB", (8, 0), 80, w),
            ("H100", (9, 0), 80, w),
            ("GH200", (9, 0), 96, w),
            ("B200", (10, 0), 180, w),
            ("RTX 5090", (12, 0), 32, w),
        ] {
            assert_eq!(best_layout(cc, mem * gib, m, 1_000_000).0, want, "{gpu}");
            assert_eq!(choose_layout(None, cc, mem * gib, m, 1_000_000).0, want, "{gpu} with no flag");
            // --layout wins on every GPU, either way
            assert_eq!(
                choose_layout(Some(Layout::Tiered), cc, mem * gib, m, 1_000_000),
                (Layout::Tiered, "as asked".to_string()),
                "{gpu} asked for the tiered layout"
            );
            assert_eq!(choose_layout(Some(Layout::Twelve), cc, mem * gib, m, 1_000_000).0, Layout::Twelve, "{gpu} asked for the 12-bit layout");
        }
        // a 24 GiB A10 for a 14B-class model: 12.04 bits do not fit, 10.8 do
        let (l, why) = best_layout((8, 6), 24 * gib, 33_000_000_000, 1_000_000);
        assert_eq!(l, Layout::Tiered);
        assert!(why.contains("only the mma layout fits"), "{why}");
        // neither fits (a 32B model on that A10): the GPU's own layout, and the fit check refuses the model
        assert_eq!(best_layout((8, 6), 24 * gib, 64_000_000_000, 1_000_000).0, Layout::Twelve);
        // the knob: only Ada has the smallest layout as the fastest
        assert!(smallest_is_fastest((8, 9)) && !smallest_is_fastest((8, 0)) && !smallest_is_fastest((9, 0)) && !smallest_is_fastest((8, 6)));
    }

    #[test]
    fn config_of_a_qwen3() {
        let dir = std::env::temp_dir().join(format!("glyd-engine-cfg-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("config.json"),
            r#"{"architectures":["Qwen3ForCausalLM"],"hidden_size":1024,"num_hidden_layers":28,"num_attention_heads":16,"num_key_value_heads":8,"head_dim":128,"intermediate_size":3072,"vocab_size":151936,"rms_norm_eps":1e-06,"rope_theta":1000000.0,"max_position_embeddings":40960,"tie_word_embeddings":true,"rope_scaling":null}"#,
        )
        .unwrap();
        std::fs::write(dir.join("generation_config.json"), r#"{"eos_token_id":[151645,151643]}"#).unwrap();
        let c = Config::load(&dir).unwrap();
        assert_eq!((c.hidden, c.layers, c.heads, c.kv_heads, c.head_dim, c.inter, c.vocab), (1024, 28, 16, 8, 128, 3072, 151936));
        assert!(c.tie && c.eps == 1e-6 && c.rope_theta == 1e6 && c.eos == vec![151645, 151643]);
        std::fs::write(dir.join("config.json"), r#"{"architectures":["LlamaForCausalLM"]}"#).unwrap();
        assert!(Config::load(&dir).unwrap_err().to_string().contains("Qwen3 dense"));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// A config.json with no attention heads was a division by zero (the head's width is the hidden size over them).
    #[test]
    fn a_qwen3_config_with_no_heads_is_an_error() {
        let c = |heads: &str| {
            format!(
                r#"{{"architectures":["Qwen3ForCausalLM"],"hidden_size":1024,"num_hidden_layers":28,"num_attention_heads":{heads},"num_key_value_heads":8,"intermediate_size":3072,"vocab_size":151936,"rms_norm_eps":1e-06,"rope_theta":1000000.0,"max_position_embeddings":40960}}"#
            )
        };
        assert!(Config::parse(&c("0"), "config.json", None).unwrap_err().to_string().contains("num_attention_heads"));
        assert_eq!(Config::parse(&c("16"), "config.json", None).unwrap().head_dim, 64);
    }

    /// bf16 weights from a fixed seed: most near 2^-6 (a trained matrix's exponents), `wild` of them (a share 0 to 1) any bit pattern at all.
    fn weights_of_spread(n: usize, wild: f64, seed: u64) -> Vec<u16> {
        let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        let mut next = move || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        (0..n)
            .map(|_| {
                let r = next();
                if (r >> 40) as f64 / (1u64 << 24) as f64 <= wild {
                    (next() >> 20) as u16
                } else {
                    // sign, an exponent of 119 to 123 (2^-8 to 2^-4), any mantissa
                    let e = 119 + (r % 5) as u16;
                    (((r >> 8) & 1) as u16) << 15 | e << 7 | ((r >> 16) & 0x7f) as u16
                }
            })
            .collect()
    }

    fn bytes_of(w: &[u16]) -> Vec<u8> {
        w.iter().flat_map(|x| x.to_le_bytes()).collect()
    }

    /// A small Qwen3 (two layers, tied embedding) as a Hub-format directory of bf16 tensors: every matrix of a trained matrix's spread but the second layer's gate and up, whose bits are any at all (a matrix the 12-bit
    /// layout holds in fewer bytes than the tiered one). Returns the tensors by name, for comparing what comes back.
    fn tiny_checkpoint(dir: &Path) -> HashMap<String, (Vec<u64>, Vec<u16>)> {
        std::fs::create_dir_all(dir).unwrap();
        std::fs::write(
            dir.join("config.json"),
            r#"{"architectures":["Qwen3ForCausalLM"],"hidden_size":128,"num_hidden_layers":2,"num_attention_heads":2,"num_key_value_heads":1,"head_dim":64,"intermediate_size":256,"vocab_size":256,"rms_norm_eps":1e-06,"rope_theta":1000000.0,"max_position_embeddings":512,"tie_word_embeddings":true}"#,
        )
        .unwrap();
        let mut t: Vec<(String, Vec<u64>, Vec<u16>)> = Vec::new();
        let mut seed = 1u64;
        let mut add = |name: &str, rows: u64, cols: u64, wild: f64| {
            seed += 1;
            let shape = if cols == 0 { vec![rows] } else { vec![rows, cols] };
            t.push((name.to_string(), shape, weights_of_spread((rows * cols.max(1)) as usize, wild, seed)));
        };
        add("model.embed_tokens.weight", 256, 128, 0.0);
        for l in 0..2 {
            let p = format!("model.layers.{l}");
            add(&format!("{p}.self_attn.q_proj.weight"), 128, 128, 0.0);
            add(&format!("{p}.self_attn.k_proj.weight"), 64, 128, 0.0);
            add(&format!("{p}.self_attn.v_proj.weight"), 64, 128, 0.0);
            add(&format!("{p}.self_attn.o_proj.weight"), 128, 128, 0.0);
            add(&format!("{p}.self_attn.q_norm.weight"), 64, 0, 0.0);
            add(&format!("{p}.self_attn.k_norm.weight"), 64, 0, 0.0);
            let wild = if l == 1 { 1.0 } else { 0.0 };
            add(&format!("{p}.mlp.gate_proj.weight"), 256, 128, wild);
            add(&format!("{p}.mlp.up_proj.weight"), 256, 128, wild);
            add(&format!("{p}.mlp.down_proj.weight"), 128, 256, 0.0);
            add(&format!("{p}.input_layernorm.weight"), 128, 0, 0.0);
            add(&format!("{p}.post_attention_layernorm.weight"), 128, 0, 0.0);
        }
        add("model.norm.weight", 128, 0, 0.0);
        let tensors: Vec<(&str, &str, Vec<u64>, Vec<u8>)> = t.iter().map(|(n, s, w)| (n.as_str(), "BF16", s.clone(), bytes_of(w))).collect();
        let refs: Vec<(&str, &str, Vec<u64>, &[u8])> = tensors.iter().map(|(n, d, s, b)| (*n, *d, s.clone(), &b[..])).collect();
        crate::pack::write_safetensors(&dir.join("model.safetensors"), &[("format", "pt".to_string())], &refs).unwrap();
        t.into_iter().map(|(n, s, w)| (n, (s, w))).collect()
    }

    fn le_i32(b: &[u8]) -> Vec<i32> {
        b.chunks_exact(4).map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
    }

    /// A lossless glyd-v4 save that holds a matrix in two packs (Qwen3.5's in_proj_qkv and in_proj_z, as `glyd-gpu pack` saves them): `lossless_manifest` names the save, `read_lossless_parts` reads a pack saved
    /// whole as it is and makes the two-part matrix again from the packs' decode, the stacked weights bit for bit; a lossy save is not named.
    #[test]
    fn a_lossless_save_in_two_packs_is_read_as_one_matrix() {
        let dir = std::env::temp_dir().join(format!("glyd-engine-v4parts-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let (cols, ra, rb) = (128usize, 128usize, 64usize);
        let (wa, wb) = (weights_of_spread(ra * cols, 0.0, 3), weights_of_spread(rb * cols, 1.0, 4));
        let mut entries = Vec::new();
        let mut tensors: Vec<(String, &'static str, Vec<u64>, Vec<u8>)> = Vec::new();
        for (key, w, rows) in [("m.a", &wa, ra), ("m.b", &wb, rb)] {
            match glyd_gpu::pack::pack_smallest_checked(w, rows, cols).unwrap() {
                glyd_gpu::pack::Packed::Tiered(p) => {
                    entries.push(format!(r#""{key}":{{"layout":"mma","shape":[{rows},{cols}],"tiers":[{},{},{}]}}"#, p.tiers[0], p.tiers[1], p.tiers[2]));
                    tensors.push((format!("{key}.glyd_data"), "U8", vec![p.data.len() as u64], p.data));
                    tensors.push((format!("{key}.glyd_blocks"), "U8", vec![p.blocks.len() as u64], p.blocks));
                    tensors.push((format!("{key}.glyd_block_base"), "I32", vec![p.block_base.len() as u64], i32_bytes(&p.block_base)));
                }
                glyd_gpu::pack::Packed::Twelve(p) => {
                    entries.push(format!(r#""{key}":{{"layout":"mma12","shape":[{rows},{cols}],"hb":{}}}"#, p.hb()));
                    tensors.push((format!("{key}.glyd_data"), "U8", vec![p.data.len() as u64], p.data));
                    tensors.push((format!("{key}.glyd_exc"), "I32", vec![p.exc.len() as u64], i32_bytes(&p.exc)));
                    tensors.push((format!("{key}.glyd_exc_base"), "I32", vec![p.exc_base.len() as u64], i32_bytes(&p.exc_base)));
                }
            }
        }
        std::fs::write(dir.join("config.json"), r#"{"model_type":"test"}"#).unwrap();
        std::fs::write(dir.join("glyd.json"), format!(r#"{{"format":"glyd-v4","glyd":"0.29.4","layout":"mma","packs":{{{}}}}}"#, entries.join(","))).unwrap();
        let refs: Vec<(&str, &str, Vec<u64>, &[u8])> = tensors.iter().map(|(n, d, s, b)| (n.as_str(), *d, s.clone(), &b[..])).collect();
        crate::pack::write_safetensors(&dir.join("model.safetensors"), &[], &refs).unwrap();
        let m = lossless_manifest(&dir).unwrap().expect("a lossless glyd-v4 save");
        assert!(v7_manifest(&dir).unwrap().is_none());
        let ck = Checkpoint::open(&dir).unwrap();
        let decode = |p: &Packed| -> Vec<u16> {
            let mut out = vec![0u16; p.rows * p.cols];
            let ok = match p.words {
                Words::Tiered { tiers, .. } => glyd_gpu::pack::unpack_tiered(
                    &glyd_gpu::pack::Tiered { rows: p.rows, cols: p.cols, data: p.data.clone(), blocks: p.b1.clone(), block_base: le_i32(&p.b2), tiers },
                    &mut out,
                ),
                Words::Twelve { sym, .. } => glyd_gpu::pack::unpack_twelve(
                    &glyd_gpu::pack::Twelve { rows: p.rows, cols: p.cols, data: p.data.clone(), exc: le_i32(&p.b1), exc_base: le_i32(&p.b2), sym },
                    &mut out,
                ),
            };
            assert!(ok);
            out
        };
        // one pack as saved: its decode is its weights
        let one = read_lossless_parts(&ck, &m, &[("m.a", ra)], cols).unwrap();
        assert_eq!(decode(&one), wa);
        let other = read_lossless_parts(&ck, &m, &[("m.b", rb)], cols).unwrap();
        assert_eq!(decode(&other), wb);
        // two packs: one matrix of their rows, packed again here, its decode the stacked weights
        let both = read_lossless_parts(&ck, &m, &[("m.a", ra), ("m.b", rb)], cols).unwrap();
        assert_eq!((both.rows, both.cols), (ra + rb, cols));
        assert_eq!(decode(&both), [wa.clone(), wb.clone()].concat());
        assert!(matches!(read_lossless_parts(&ck, &m, &[("m.a", ra), ("m.c", rb)], cols), Err(e) if e.0.contains("no pack m.c")));
        // a lossy save is not a lossless one
        std::fs::write(dir.join("glyd.json"), r#"{"format":"glyd-v4","layout":"i5","packs":{}}"#).unwrap();
        assert!(lossless_manifest(&dir).unwrap().is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A glyd-v4 save that mixes both layouts, as the engine loads it: each pack in the layout its manifest names (the second layer's gate and up in the 12-bit one, the rest in the tiered one, the embedding in the
    /// sidecar by its own metadata), the tensors to load named by those layouts, and every pack's buffers, read as the plan reads them, decoding to the bits of the checkpoint they came from.
    #[test]
    fn a_save_that_mixes_the_two_layouts_loads_each_pack_in_its_own() {
        let dir = std::env::temp_dir().join(format!("glyd-engine-mix-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let (src, out) = (dir.join("src"), dir.join("packed"));
        let w = tiny_checkpoint(&src);
        crate::pack::run_with(src.to_str().unwrap(), &out, 2, Layout::Tiered, true).unwrap();
        let manifest = json::parse(&std::fs::read_to_string(out.join("glyd.json")).unwrap()).unwrap();
        assert_eq!(manifest.get("format").and_then(Value::as_str), Some("glyd-v4"));
        assert_eq!(earlier_format(&out), None);
        assert_eq!(packed_layout(&out), Some(Layout::Tiered));

        let planned = plan(&out).unwrap();
        let layout_of = |k: &str| planned.packs.iter().find(|p| p.key == k).unwrap_or_else(|| panic!("no pack {k}")).layout;
        assert_eq!(layout_of("model.layers.1.mlp.gate_proj"), Layout::Twelve, "the matrix of any bits is held in the 12-bit layout");
        for k in [
            "model.layers.0.self_attn.q_proj",
            "model.layers.0.self_attn.o_proj",
            "model.layers.0.mlp.gate_proj",
            "model.layers.0.mlp.down_proj",
            "model.layers.1.self_attn.q_proj",
            "model.layers.1.mlp.down_proj",
            "embed",
        ] {
            assert_eq!(layout_of(k), Layout::Tiered, "{k}");
        }
        assert_eq!(planned.packs.len(), 9, "two layers of four packs and the embedding (the head is the embedding's)");
        // the tensors to load: each pack's three buffers by its layout's names, in the order the loader takes them
        let names = planned.tensors();
        assert!(
            names.contains(&"model.layers.1.mlp.gate_proj.glyd_exc_base".to_string())
                && !names.contains(&"model.layers.1.mlp.gate_proj.glyd_blocks".to_string())
        );
        assert!(
            names.contains(&"model.layers.0.mlp.gate_proj.glyd_blocks".to_string()) && !names.contains(&"model.layers.0.mlp.gate_proj.glyd_exc".to_string())
        );
        assert_eq!(names.len(), 9 * 3 + 2 * 4 + 1);

        // every pack as the library holds it, decoded on the CPU, against the weights it was made from
        let ck = Checkpoint::open(&out).unwrap();
        let ef = glyd_gpu::safetensors::File::open(&out.join("engine.safetensors")).unwrap();
        let read = |name: &str, sidecar: bool| if sidecar { ef.read(ef.get(name).unwrap()).unwrap() } else { ck.read(name).unwrap() };
        let members = |key: &str| -> Vec<u16> {
            let names: Vec<String> = if key == "embed" {
                vec!["model.embed_tokens.weight".to_string()]
            } else {
                let (module, last) = key.rsplit_once('.').unwrap();
                match last {
                    "q_proj" => vec![format!("{module}.q_proj.weight"), format!("{module}.k_proj.weight"), format!("{module}.v_proj.weight")],
                    "gate_proj" => vec![format!("{module}.gate_proj.weight"), format!("{module}.up_proj.weight")],
                    _ => vec![format!("{key}.weight")],
                }
            };
            names.iter().flat_map(|n| w[n].1.clone()).collect()
        };
        for pp in &planned.packs {
            let key = pp.key.as_str();
            let sidecar = key == "embed";
            let sfx = suffixes(pp.layout);
            let (rows, cols) = (pp.rows as usize, pp.cols as usize);
            let buf = |i: usize| read(&format!("{key}.glyd_{}", sfx[i]), sidecar);
            let mut out = vec![0u16; rows * cols];
            let ok = match pp.layout {
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
                | Layout::Hummingbird => {
                    assert_eq!(pp.words.0[0] >> 24, glyd_gpu::pack::TIERED_TAG as u64, "{key}: the new layout's number");
                    let p = glyd_gpu::pack::Tiered {
                        rows,
                        cols,
                        data: buf(0),
                        blocks: buf(1),
                        block_base: le_i32(&buf(2)),
                        tiers: [pp.words.0[0] as u32, pp.words.0[1] as u32, pp.words.0[2] as u32],
                    };
                    glyd_gpu::pack::unpack_tiered(&p, &mut out)
                }
                Layout::Twelve => {
                    let p = glyd_gpu::pack::Twelve { rows, cols, data: buf(0), exc: le_i32(&buf(1)), exc_base: le_i32(&buf(2)), sym: pp.words.1 };
                    glyd_gpu::pack::unpack_twelve(&p, &mut out)
                }
            };
            assert!(ok, "{key}: its buffers do not decode");
            assert!(out == members(key), "{key}: decoded to other bits than the checkpoint's");
        }

        // a directory of the earlier layout (glyd 0.27's: the tiers' top byte 0xFF) is named as such and not loaded
        let old = dir.join("old");
        std::fs::create_dir_all(&old).unwrap();
        for f in std::fs::read_dir(&out).unwrap() {
            let f = f.unwrap();
            std::fs::copy(f.path(), old.join(f.file_name())).unwrap();
        }
        let text = std::fs::read_to_string(old.join("glyd.json")).unwrap();
        let mut v = json::parse(&text).unwrap();
        if let Value::Object(top) = &mut v {
            for (k, val) in top.iter_mut() {
                if k == "format" {
                    *val = Value::from("glyd-v1");
                }
                if k == "packs" {
                    if let Value::Object(packs) = val {
                        let (_, e) = packs.iter_mut().find(|(k, _)| k == "model.layers.0.self_attn.q_proj").unwrap();
                        if let Value::Object(fields) = e {
                            let (_, tiers) = fields.iter_mut().find(|(k, _)| k == "tiers").unwrap();
                            if let Value::Array(a) = tiers {
                                let t0 = a[0].as_u64().unwrap();
                                a[0] = Value::from((t0 & 0xFF_FFFF) | 0xFF << 24);
                            }
                        }
                    }
                }
            }
        }
        std::fs::write(old.join("glyd.json"), json::to_python(&v, 1)).unwrap();
        assert_eq!(earlier_format(&old).as_deref(), Some(EARLIER_FORMAT));
        let e = plan_error(&old);
        assert!(e.contains("glyd 0.27") && e.contains("pack the model again"), "{e}");
        // `glyd-engine pack` into that directory makes its packs again (its other files stay), and the result loads
        std::fs::write(old.join("kv-tables-4096.safetensors"), "tables").unwrap();
        crate::pack::run_with(src.to_str().unwrap(), &old, 2, Layout::Tiered, true).unwrap();
        assert_eq!(earlier_format(&old), None);
        assert_eq!(plan(&old).unwrap().packs.len(), 9);
        assert_eq!(std::fs::read_to_string(old.join("kv-tables-4096.safetensors")).unwrap(), "tables");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    fn plan_error(dir: &Path) -> String {
        match plan(dir) {
            Ok(_) => String::from("loaded"),
            Err(e) => e.to_string(),
        }
    }

    /// A matrix of a checkpoint packed as the loader packs it in the four- and three-bit layouts (q, k and v stacked by rows): the packer's own pack, the words its tag.
    #[test]
    fn a_matrix_is_packed_in_a_block_layout_as_it_loads() {
        let dir = std::env::temp_dir().join(format!("glyd-engine-blocks-{}", std::process::id()));
        let w = tiny_checkpoint(&dir);
        let ck = Checkpoint::open(&dir).unwrap();
        let members: Vec<String> = ["q", "k", "v"].iter().map(|p| format!("model.layers.0.self_attn.{p}_proj.weight")).collect();
        let item = Item { members: members.clone(), rows: 256, cols: 128 };
        let stacked: Vec<u16> = members.iter().flat_map(|n| w[n].1.clone()).collect();
        type Pack = fn(&[u16], usize, usize) -> Option<glyd_gpu::pack::Tiered>;
        for (layout, pack, tag) in
            [(Layout::Int4, glyd_gpu::pack::pack_int4 as Pack, glyd_gpu::pack::INT4_TAG), (Layout::Int3, glyd_gpu::pack::pack_int3, glyd_gpu::pack::INT3_TAG)]
        {
            let got = pack_item(&ck, &item, layout).unwrap();
            let want = pack(&stacked, 256, 128).unwrap();
            assert!(
                got.data == want.data && got.b1 == want.blocks && got.b2 == i32_bytes(&want.block_base) && (got.rows, got.cols) == (256, 128),
                "{layout:?}"
            );
            assert!(matches!(got.words, Words::Tiered { tiers, .. } if tiers == want.tiers && tiers[0] >> 24 == tag), "{layout:?}");
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// A glyd-v7 directory whose packs are in two block layouts (four matrices of 64 columns saved at the four-bit layout, the plan putting two of them in the six-bit one): `v7_manifest` names the level's layout, `read_v7`
    /// and `read_pack` read each pack in the layout its entry names, to the very bytes the packer makes of the bf16 matrix (the parts of a fused matrix joined end to end where they are in one layout); a reader that takes
    /// the level's layout for every pack (`check_v7` with it, as every reader did before a directory could hold two) refuses a six-bit pack by its tag, not reading it as four-bit codes; refused too: the parts of one matrix in
    /// two layouts, and an entry that names no block layout.
    #[test]
    fn a_directory_with_packs_in_two_block_layouts_is_read_pack_by_pack() {
        use glyd_gpu::pack::{pack_int4, pack_int6, INT4_TAG, INT6_TAG};
        let dir = std::env::temp_dir().join(format!("glyd-engine-two-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let (src, out) = (dir.join("src"), dir.join("out"));
        std::fs::create_dir_all(&src).unwrap();
        std::fs::write(src.join("config.json"), "{}").unwrap();
        // a (128 rows) and d (64) in six-bit codes, b and c (64) in the level's four-bit ones
        let (names, rows) = (["m.a", "m.b", "m.c", "m.d"], [128usize, 64, 64, 64]);
        let w: Vec<Vec<u16>> = rows.iter().enumerate().map(|(i, &r)| weights_of_spread(r * 64, 0.0, 10 + i as u64)).collect();
        let (files, data): (Vec<String>, Vec<Vec<u8>>) = names.iter().zip(&w).map(|(n, w)| (format!("{n}.weight"), bytes_of(w))).unzip();
        let tensors: Vec<(&str, &str, Vec<u64>, &[u8])> =
            files.iter().zip(&rows).zip(&data).map(|((n, &r), b)| (n.as_str(), "BF16", vec![r as u64, 64], &b[..])).collect();
        crate::pack::write_safetensors(&src.join("model.safetensors"), &[("format", "pt".to_string())], &tensors).unwrap();
        let plan = r#"{"puts": [{"linear": ["m.a"], "layout": "i6"}, {"linear": ["m.b"]}, {"linear": ["m.c"]}, {"linear": ["m.d"], "layout": "i6"}]}"#;
        glyd_gpu::save::save_planned(&glyd_gpu::save::Source::find(src.to_str().unwrap()).unwrap(), &out, 2, 1 << 30, Layout::Int4, plan).unwrap();
        let (level, m) = v7_manifest(&out).unwrap().unwrap();
        assert_eq!(level, Layout::Int4);
        let ck = Checkpoint::open(&out).unwrap();
        type Pack = fn(&[u16], usize, usize) -> Option<glyd_gpu::pack::Tiered>;
        for (k, own, pack, tag) in [
            (0, Layout::Int6, pack_int6 as Pack, INT6_TAG),
            (1, Layout::Int4, pack_int4 as Pack, INT4_TAG),
            (2, Layout::Int4, pack_int4 as Pack, INT4_TAG),
            (3, Layout::Int6, pack_int6 as Pack, INT6_TAG),
        ] {
            let key = names[k];
            let got = read_v7(&ck, &m, level, &[(key, rows[k])], 64).unwrap();
            let want = pack(&w[k], rows[k], 64).unwrap();
            assert!(got.data == want.data && got.b1 == want.blocks && got.b2 == i32_bytes(&want.block_base), "{key}");
            assert!(matches!(got.words, Words::Tiered { tiers, .. } if tiers == want.tiers && tiers[0] >> 24 == tag), "{key}");
            // (read_pack, which the Nemotron loader calls, reads it so too: the level it is given is the directory's, here)
            let pk = Pack35 { key: key.to_string(), parts: vec![(format!("{key}.weight"), rows[k])], cols: 64 };
            let read = read_pack(&ck, Some(&(level, m.clone())), &pk, level).unwrap();
            assert!(read.data == got.data && read.b1 == got.b1, "{key}");
            // a reader that holds the level's layout for every pack
            let e = m.get("packs").and_then(|p| p.get(key)).unwrap();
            assert_eq!(e.get("layout").and_then(Value::as_str), Some(own.name()), "{key}");
            let lens = ["data", "blocks", "block_base"].map(|s| ck.get(&format!("{key}.glyd_{s}")).unwrap().1.bytes);
            let said =
                check_v7(key, level, rows[k], 64, &e.get("shape").and_then(Value::as_u64s).unwrap(), &e.get(WORDS_KEY).and_then(Value::as_u64s).unwrap(), lens);
            match own == level {
                true => assert!(said.is_ok(), "{key}"),
                false => assert!(said.unwrap_err().0.contains("where the i4 layout's is 6"), "{key}"),
            }
        }
        // the parts of one matrix, joined: two in the level's layout are the packer's pack of the two stacked; one in each layout are refused
        let joined = read_v7(&ck, &m, level, &[("m.b", 64), ("m.c", 64)], 64).unwrap();
        let want = pack_int4(&[w[1].clone(), w[2].clone()].concat(), 128, 64).unwrap();
        assert!(joined.data == want.data && joined.b1 == want.blocks && joined.rows == 128);
        let e = read_v7(&ck, &m, level, &[("m.c", 64), ("m.d", 64)], 64).err().unwrap().0;
        assert!(e.contains("m.d: packed in the i6 layout, where the part of its matrix before it is in i4"), "{e}");
        // an entry that names no block layout, and one that names none (the directory's layout is then its pack's)
        let edited = |layout: Option<&str>| -> Value {
            let mut v = m.clone();
            let Value::Object(top) = &mut v else { unreachable!() };
            let Some((_, Value::Object(packs))) = top.iter_mut().find(|(k, _)| k == "packs") else { unreachable!() };
            let Some((_, Value::Object(entry))) = packs.iter_mut().find(|(k, _)| k == "m.b") else { unreachable!() };
            entry.retain(|(k, _)| k != "layout");
            if let Some(l) = layout {
                entry.push(("layout".into(), Value::from(l)));
            }
            v
        };
        let e = read_v7(&ck, &edited(Some("mma")), level, &[("m.b", 64)], 64).err().unwrap().0;
        assert!(e.contains("m.b: layout \"mma\", where a glyd-v7 pack has"), "{e}");
        assert!(read_v7(&ck, &edited(None), level, &[("m.b", 64)], 64).unwrap().data == pack_int4(&w[1], 64, 64).unwrap().data);
        assert!(
            read_v7(&ck, &edited(None), Layout::Int6, &[("m.b", 64)], 64).err().unwrap().0.contains("where the i6 layout's is 4"),
            "no entry layout: the directory's, checked as before"
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// A saved pack is checked against what pack_int6, pack_int5, pack_int4 and pack_int3 make of the config's shape: the tag, the shape and each buffer's length.
    #[test]
    fn a_saved_pack_is_checked_before_it_is_read() {
        let w: Vec<u16> = (0..128 * 64).map(|i| bf16(((i * 7919) % 201) as f32 / 100.0 - 1.0)).collect();
        let six = glyd_gpu::pack::pack_int6(&w, 128, 64).unwrap();
        let five = glyd_gpu::pack::pack_int5(&w, 128, 64).unwrap();
        let (four, three) = (glyd_gpu::pack::pack_int4(&w, 128, 64).unwrap(), glyd_gpu::pack::pack_int3(&w, 128, 64).unwrap());
        let lens = |p: &glyd_gpu::pack::Tiered| [p.data.len() as u64, p.blocks.len() as u64, 4 * p.block_base.len() as u64];
        let tiers = |p: &glyd_gpu::pack::Tiered| p.tiers.map(u64::from);
        let ok = |l, rows, shape: &[u64], t: &[u64], n: [u64; 3]| check_v7("k", l, rows, 64, shape, t, n).map_err(|e| e.0);
        assert!(ok(Layout::Int6, 128, &[128, 64], &tiers(&six), lens(&six)).is_ok());
        assert!(ok(Layout::Int5, 128, &[128, 64], &tiers(&five), lens(&five)).is_ok());
        assert!(ok(Layout::Int4, 128, &[128, 64], &tiers(&four), lens(&four)).is_ok());
        assert!(ok(Layout::Int3, 128, &[128, 64], &tiers(&three), lens(&three)).is_ok());
        // the tag of the other level
        assert!(ok(Layout::Int6, 128, &[128, 64], &tiers(&five), lens(&five)).unwrap_err().contains("tag Some(5), where the i6 layout's is 4"));
        assert!(ok(Layout::Int4, 128, &[128, 64], &tiers(&three), lens(&three)).unwrap_err().contains("tag Some(7), where the i4 layout's is 6"));
        assert!(ok(Layout::Int3, 128, &[128, 64], &tiers(&four), lens(&four)).unwrap_err().contains("tag Some(6), where the i3 layout's is 7"));
        // (the right tag, the other layout's step: the buffers' lengths say so)
        assert!(ok(Layout::Int4, 128, &[128, 64], &tiers(&four), lens(&three)).unwrap_err().contains("where"));
        // a shape other than the config's
        assert!(ok(Layout::Int6, 128, &[64, 128], &tiers(&six), lens(&six)).unwrap_err().contains("shape [64, 128], where the config has 128 x 64"));
        assert!(ok(Layout::Int6, 192, &[128, 64], &tiers(&six), lens(&six)).is_err());
        // a buffer cut short
        for i in 0..3 {
            let mut n = lens(&six);
            n[i] -= 4;
            assert!(ok(Layout::Int6, 128, &[128, 64], &tiers(&six), n).unwrap_err().contains("where"), "{i}");
            let mut n = lens(&four);
            n[i] -= 4;
            assert!(ok(Layout::Int4, 128, &[128, 64], &tiers(&four), n).unwrap_err().contains("where"), "{i}");
        }
        // (a layout that is not a block one)
        assert!(ok(Layout::Tiered, 128, &[128, 64], &tiers(&six), lens(&six)).is_err());
    }
}
