//! Packed matrices on the GPU (`PackBuf`) and how a checkpoint's matrices become them: packed as they load in a level's layout (`pack_item_map`), or read as a glyd-v7
//! or a lossless glyd-v4 directory saved them (`read_v7`, `read_lossless`); a Qwen3 config as the planner reads it (`Config`), the rotary tables, the layouts' choice.

use crate::dev::Buf;
use crate::ffi::Words;
use crate::ffi::{err, Error, Pack, Result};
use crate::qwen35::Pack35;
use crate::runtime::config::{ModelConfig, Width};
use glyd_gpu::json::{self, Value};
use glyd_gpu::safetensors::Checkpoint;
pub use glyd_gpu::save::Layout;
use std::path::Path;

fn io<T>(r: std::io::Result<T>) -> Result<T> {
    r.map_err(|e| Error(e.to_string()))
}

pub use crate::runtime::config::{eos_ids, num};

/// What the engine reads of a plain dense decoder's config.json (the dense kind's sizes: `runtime::describe::LLAMA`).
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
        ModelConfig::read(dir, Config::of)
    }

    /// A config.json's text (and a generation_config.json's, where there is one) read as `load` reads the files: `label` names the file in an error.
    pub fn parse(text: &str, label: &str, generation: Option<&str>) -> Result<Config> {
        Config::of(&ModelConfig::parse(text).map_err(|e| e.context(label))?, label, generation)
    }

    /// What the engine reads of a config.json parsed once (`ModelConfig`): its shared keys typed, `rms_norm_eps` and `rope_theta` its own.
    pub fn of(c: &ModelConfig, label: &str, generation: Option<&str>) -> Result<Config> {
        let s = &c.shared;
        if !crate::runtime::describe::LLAMA.contains(&c.text_type.as_str()) {
            return err(format!("{label}: model_type {:?}: not a plain dense decoder (the dense kind's sizes are read here)", c.text_type));
        }
        let need = |x: Option<usize>, k: &str| x.ok_or_else(|| Error(format!("{label}: no {k}")));
        let heads = need(s.num_attention_heads, "num_attention_heads")?;
        if heads == 0 {
            return err(format!("{label}: num_attention_heads: 0"));
        }
        let hidden = need(s.hidden_size, "hidden_size")?;
        let head_dim = s.head_dim.unwrap_or(hidden / heads);
        if c.top.get("rope_scaling").is_some_and(|r| !r.is_null()) {
            return err(format!("{label}: rope_scaling: not supported yet"));
        }
        let layers = need(s.num_hidden_layers, "num_hidden_layers")?;
        if layers > crate::plan::MAX_LAYERS {
            return err(format!("{label}: num_hidden_layers {layers}, where no model has more than {}", crate::plan::MAX_LAYERS));
        }
        let inter = match &s.intermediate_size {
            Some(Width::One(i)) => *i,
            Some(Width::PerLayer(_)) => return err(format!("{label}: intermediate_size: a width a layer, where Qwen3 has one")),
            None => return err(format!("{label}: no intermediate_size")),
        };
        let get = |k: &str| c.top.get(k).ok_or_else(|| Error(format!("{label}: no {k}")));
        Ok(Config {
            hidden,
            layers,
            heads,
            kv_heads: need(s.num_key_value_heads, "num_key_value_heads")?,
            head_dim,
            inter,
            vocab: need(s.vocab_size, "vocab_size")?,
            eps: num(get("rms_norm_eps")?, "rms_norm_eps")? as f32,
            rope_theta: num(get("rope_theta")?, "rope_theta")?,
            max_pos: need(s.max_position_embeddings, "max_position_embeddings")?,
            tie: s.tie_word_embeddings.unwrap_or(false),
            eos: eos_ids(&c.top, generation),
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

/// The layout for a GPU, chosen by the planner (src/plan.rs `sizing`).
pub use crate::plan::sizing::{best_layout, choose_layout};

/// A pack and the device buffers it points into.
pub struct PackBuf {
    pub pack: Pack,
    pub bytes: usize,
    /// The whole matrix's rows where this pack is this GPU's slice of them (src/dynamic/tp.rs: a product is gathered after it); 0: the whole matrix.
    pub tp_rows: usize,
    _bufs: [Buf; 3],
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

/// `x` rounded to bf16 (to nearest, ties to even), its bits.
pub(crate) fn bf16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    (b.wrapping_add(0x7fff + ((b >> 16) & 1)) >> 16) as u16
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

/// The last member of an item whose rows are more than its tensors': it names no tensor, and the rows after them are zeros (a fused product padded so that
/// its rows pack, and slice across a group of GPUs whatever the group: no product reads them).
pub(crate) const ZERO_ROWS: &str = "";

/// A matrix to pack: the checkpoint's tensors stacked by rows (q, k, v; gate, up; or one), zero rows after them where its last member is `ZERO_ROWS`.
#[derive(Clone, Debug)]
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

pub(crate) fn i32_bytes(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// Apply an explicit development study transform before the unchanged checked lossless packer.
pub(crate) fn pack_item_map(ck: &Checkpoint, it: &Item, layout: Layout, transform: &(dyn Fn(&mut [u16]) + Sync)) -> Result<Packed> {
    let mut w = item_words(ck, it)?;
    transform(&mut w);
    pack_words(&w, it.rows, it.cols, layout, &it.members.join(" + "))
}

/// `pack_item_map` with no transform, of an item whose members the checkpoint may hold quantized as its description says (`item_words_as`).
pub(crate) fn pack_item_as(ck: &Checkpoint, it: &Item, layout: Layout, stored: &crate::formats::Stored) -> Result<Packed> {
    pack_words(&item_words_as(ck, it, stored)?, it.rows, it.cols, layout, &it.members.join(" + "))
}

/// An item's members stacked, as bf16 words (an FP8 member as the bf16 it dequantises to), zeros after them where the item says so (`ZERO_ROWS`, its last
/// member).
pub(crate) fn item_words(ck: &Checkpoint, it: &Item) -> Result<Vec<u16>> {
    item_words_as(ck, it, &crate::formats::Stored::default())
}

/// `item_words` of a model whose description names its stored formats (`stored`: `Described::stored`, which the dense lowering hands its loader). A
/// member the checkpoint holds quantized with scales the description names (FP8 with block scales under any name, in blocks of any size, a grid a shard
/// where the description cuts the matrix in shards; 4-bit codes two a byte with their runs' scales) is DEQUANTISED HERE to bf16 (`Stored::words`) and
/// goes on as a bf16 matrix does: packed at the plan's level, or at penguin held lossless in the engine's own layout. A release stored quantized is so
/// not held as released: at penguin its weights are the bf16 the release's codes and scales decode to (the release's values exactly where a scale is a
/// power of two; the bf16 nearest each product where a scale is fp32), in packs the size bf16 weights take, not the release's bytes.
pub(crate) fn item_words_as(ck: &Checkpoint, it: &Item, stored: &crate::formats::Stored) -> Result<Vec<u16>> {
    let mut w = vec![0u16; it.rows * it.cols];
    let mut at = 0usize;
    for (i, m) in it.members.iter().enumerate() {
        if m == ZERO_ROWS {
            if i + 1 != it.members.len() || at > w.len() {
                return err(format!("{}: zero rows that are not after its {at} weights, within {} x {}", it.members.join(" + "), it.rows, it.cols));
            }
            at = w.len();
            continue;
        }
        // (a table stored at 4 bits beside its name: the bf16 it decodes to, every row as the host's gather decodes it)
        if let Some(q) = crate::formats::q4_beside(ck, m) {
            let v = crate::formats::nf4_words(ck, &q, it.cols)?;
            if at + v.len() > w.len() {
                return err(format!("{}: {} weights past its {} x {}", it.members.join(" + "), at + v.len(), it.rows, it.cols));
            }
            w[at..at + v.len()].copy_from_slice(&v);
            at += v.len();
            continue;
        }
        // (a matrix its description names scales for, held quantized: the bf16 its codes and scales decode to)
        if let Some((_, cols, v)) = stored.words(ck, m)? {
            if cols != it.cols || at + v.len() > w.len() {
                return err(format!("{m}: {} weights of {cols} columns, where {} x {} holds {at} before them", v.len(), it.rows, it.cols));
            }
            w[at..at + v.len()].copy_from_slice(&v);
            at += v.len();
            continue;
        }
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
pub(crate) fn entry_layout(name: &str) -> Option<Layout> {
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
#[cfg(any(test, feature = "dev"))]
pub(crate) fn read_pack(ck: &Checkpoint, v7: Option<&(Layout, Value)>, pk: &Pack35, level: Layout) -> Result<Packed> {
    read_pack_as(ck, v7, pk, level, &crate::formats::Stored::default())
}

/// `read_pack`, a part the checkpoint holds quantized as its description says (`stored`) dequantised as it is packed (`item_words_as`).
pub(crate) fn read_pack_as(ck: &Checkpoint, v7: Option<&(Layout, Value)>, pk: &Pack35, level: Layout, stored: &crate::formats::Stored) -> Result<Packed> {
    let saved = saved_parts(ck, pk);
    match v7 {
        Some((_, manifest)) if !saved.is_empty() => read_v7(ck, manifest, level, &saved, pk.cols),
        _ => pack_item_as(ck, &Item { members: pk.parts.iter().map(|x| x.0.clone()).collect(), rows: pk.rows(), cols: pk.cols }, level, stored),
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
            // (under the lock: a worker reads `taken` and `stop` under it before it waits, so a change it has not seen cannot fall between its
            // reading and its waiting, where the wake-up would be lost and the worker, and with it this loop, would wait for ever)
            {
                let _g = slots.lock().unwrap();
                taken.store(i + 1, Ordering::SeqCst);
            }
            cv.notify_all();
            match r.and_then(|v| each(i, v)) {
                Ok(()) => {}
                Err(e) => {
                    result = Err(e);
                    {
                        let _g = slots.lock().unwrap();
                        stop.store(true, Ordering::SeqCst);
                    }
                    cv.notify_all();
                    break;
                }
            }
        }
        result
    })
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
    /// The matrix as the GPU holds it: its buffers read back and decoded on the host ([rows, cols], bf16). A lossless layout's alone (the tiered one by its
    /// tag, the 12-bit one): a check's, which holds what a lossless level loaded to the checkpoint's bits.
    pub fn words(&self) -> Result<Vec<u16>> {
        use glyd_gpu::pack::{unpack_tiered, unpack_twelve, Tiered, Twelve, TIERED_TAG};
        let read = |b: &Buf| -> Result<Vec<u8>> {
            let mut v = vec![0u8; b.bytes];
            b.read(0, &mut v)?;
            Ok(v)
        };
        let i32s = |b: &[u8]| -> Vec<i32> { b.as_chunks::<4>().0.iter().map(|x| i32::from_le_bytes(*x)).collect() };
        let (rows, cols) = (self.pack.rows as usize, self.pack.cols as usize);
        let (data, b1, b2) = (read(&self._bufs[0])?, read(&self._bufs[1])?, read(&self._bufs[2])?);
        let mut out = vec![0u16; rows * cols];
        let ok = match self.pack.words {
            Words::Tiered { tiers, .. } if tiers[0] >> 24 == TIERED_TAG => {
                unpack_tiered(&Tiered { rows, cols, data, blocks: b1, block_base: i32s(&b2), tiers }, &mut out)
            }
            Words::Tiered { tiers, .. } => return err(format!("a pack of layout tag {}: not a lossless layout's", tiers[0] >> 24)),
            Words::Twelve { sym, .. } => unpack_twelve(&Twelve { rows, cols, data, exc: i32s(&b1), exc_base: i32s(&b2), sym }, &mut out),
        };
        if !ok {
            return err(format!("a pack of {rows} x {cols} read back from the GPU does not decode"));
        }
        Ok(out)
    }

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
        // (many short runs with a narrow window: a wake-up lost between a worker's reading of what is taken and its waiting left the run waiting for ever)
        for round in 0..2000 {
            let mut sum = 0usize;
            ordered(24, 8, 1, &|i| Ok(i), |_, v| {
                sum += v;
                Ok(())
            })
            .unwrap_or_else(|e| panic!("round {round}: {e}"));
            assert_eq!(sum, 276);
        }
    }

    #[test]
    fn config_of_a_dense_decoder() {
        let dir = std::env::temp_dir().join(format!("glyd-engine-cfg-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("config.json"),
            r#"{"architectures":["Qwen3ForCausalLM"],"model_type":"qwen3","hidden_size":1024,"num_hidden_layers":28,"num_attention_heads":16,"num_key_value_heads":8,"head_dim":128,"intermediate_size":3072,"vocab_size":151936,"rms_norm_eps":1e-06,"rope_theta":1000000.0,"max_position_embeddings":40960,"tie_word_embeddings":true,"rope_scaling":null}"#,
        )
        .unwrap();
        std::fs::write(dir.join("generation_config.json"), r#"{"eos_token_id":[151645,151643]}"#).unwrap();
        let c = Config::load(&dir).unwrap();
        assert_eq!((c.hidden, c.layers, c.heads, c.kv_heads, c.head_dim, c.inter, c.vocab), (1024, 28, 16, 8, 128, 3072, 151936));
        assert!(c.tie && c.eps == 1e-6 && c.rope_theta == 1e6 && c.eos == vec![151645, 151643]);
        // (the kind is the model type's, as the description builder reads it: an architecture's name alone is not it)
        std::fs::write(dir.join("config.json"), r#"{"architectures":["Qwen3ForCausalLM"],"model_type":"other"}"#).unwrap();
        assert!(Config::load(&dir).unwrap_err().to_string().contains("not a plain dense decoder"));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// A config.json with no attention heads was a division by zero (the head's width is the hidden size over them).
    #[test]
    fn a_dense_config_with_no_heads_is_an_error() {
        let c = |heads: &str| {
            format!(
                r#"{{"architectures":["Qwen3ForCausalLM"],"model_type":"qwen3","hidden_size":1024,"num_hidden_layers":28,"num_attention_heads":{heads},"num_key_value_heads":8,"intermediate_size":3072,"vocab_size":151936,"rms_norm_eps":1e-06,"rope_theta":1000000.0,"max_position_embeddings":40960}}"#
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
    fn tiny_checkpoint(dir: &Path) -> std::collections::HashMap<String, (Vec<u64>, Vec<u16>)> {
        std::fs::create_dir_all(dir).unwrap();
        std::fs::write(
            dir.join("config.json"),
            r#"{"architectures":["Qwen3ForCausalLM"],"model_type":"qwen3","hidden_size":128,"num_hidden_layers":2,"num_attention_heads":2,"num_key_value_heads":1,"head_dim":64,"intermediate_size":256,"vocab_size":256,"rms_norm_eps":1e-06,"rope_theta":1000000.0,"max_position_embeddings":512,"tie_word_embeddings":true}"#,
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

    /// An item whose last member is `ZERO_ROWS`: its tensors stacked, then zero rows to its own count (a fused product padded to rows the library packs,
    /// packed here and decoded to the same words). Zero rows anywhere else, or tensors of more rows than the item's, are refused; an item without the member
    /// still takes exactly its tensors' rows.
    #[test]
    fn an_item_says_the_zero_rows_after_its_tensors() {
        let dir = std::env::temp_dir().join(format!("glyd-engine-zero-rows-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let (cols, ra, rb) = (32usize, 40usize, 8usize);
        let words = |rows: usize, k: u16| -> Vec<u16> { (0..rows * cols).map(|i| 0x3c00 + ((i as u16).wrapping_mul(7) ^ k) % 0x300).collect() };
        let (a, b) = (words(ra, 1), words(rb, 2));
        let (ba, bb): (Vec<u8>, Vec<u8>) = (a.iter().flat_map(|w| w.to_le_bytes()).collect(), b.iter().flat_map(|w| w.to_le_bytes()).collect());
        let refs: Vec<(&str, &str, Vec<u64>, &[u8])> =
            vec![("a.weight", "BF16", vec![ra as u64, cols as u64], &ba[..]), ("b.weight", "BF16", vec![rb as u64, cols as u64], &bb[..])];
        crate::pack::write_safetensors(&dir.join("model.safetensors"), &[("format", "pt".to_string())], &refs).unwrap();
        let ck = Checkpoint::open(&dir).unwrap();
        let item = |members: &[&str], rows: usize| Item { members: members.iter().map(|m| m.to_string()).collect(), rows, cols };
        let w = item_words(&ck, &item(&["a.weight", "b.weight", ZERO_ROWS], 64)).unwrap();
        assert_eq!((&w[..ra * cols], &w[ra * cols..(ra + rb) * cols]), (&a[..], &b[..]));
        assert!(w[(ra + rb) * cols..].iter().all(|&x| x == 0) && w.len() == 64 * cols);
        // (no zero rows to add: the member is the tensors' rows alone)
        assert_eq!(item_words(&ck, &item(&["a.weight", "b.weight", ZERO_ROWS], ra + rb)).unwrap().len(), (ra + rb) * cols);
        assert!(item_words(&ck, &item(&["a.weight", ZERO_ROWS, "b.weight"], 64)).unwrap_err().msg.contains("zero rows"));
        assert!(item_words(&ck, &item(&["a.weight", "b.weight"], 64)).unwrap_err().msg.contains("are wanted"));
        // (packed as one matrix of its own rows: the lossless pack decodes to the padded words)
        let p = pack_item_map(&ck, &item(&["a.weight", "b.weight", ZERO_ROWS], 64), Layout::Tiered, &|_| {}).unwrap();
        assert_eq!((p.rows, p.cols), (64, cols));
        let _ = std::fs::remove_dir_all(&dir);
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
        assert!(matches!(read_lossless_parts(&ck, &m, &[("m.a", ra), ("m.c", rb)], cols), Err(e) if e.msg.contains("no pack m.c")));
        // a lossy save is not a lossless one
        std::fs::write(dir.join("glyd.json"), r#"{"format":"glyd-v4","layout":"i5","packs":{}}"#).unwrap();
        assert!(lossless_manifest(&dir).unwrap().is_none());
        let _ = std::fs::remove_dir_all(&dir);
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
            let got = pack_item_map(&ck, &item, layout, &|_| {}).unwrap();
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
                false => assert!(said.unwrap_err().msg.contains("where the i4 layout's is 6"), "{key}"),
            }
        }
        // the parts of one matrix, joined: two in the level's layout are the packer's pack of the two stacked; one in each layout are refused
        let joined = read_v7(&ck, &m, level, &[("m.b", 64), ("m.c", 64)], 64).unwrap();
        let want = pack_int4(&[w[1].clone(), w[2].clone()].concat(), 128, 64).unwrap();
        assert!(joined.data == want.data && joined.b1 == want.blocks && joined.rows == 128);
        let e = read_v7(&ck, &m, level, &[("m.c", 64), ("m.d", 64)], 64).err().unwrap().msg;
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
        let e = read_v7(&ck, &edited(Some("mma")), level, &[("m.b", 64)], 64).err().unwrap().msg;
        assert!(e.contains("m.b: layout \"mma\", where a glyd-v7 pack has"), "{e}");
        assert!(read_v7(&ck, &edited(None), level, &[("m.b", 64)], 64).unwrap().data == pack_int4(&w[1], 64, 64).unwrap().data);
        assert!(
            read_v7(&ck, &edited(None), Layout::Int6, &[("m.b", 64)], 64).err().unwrap().msg.contains("where the i6 layout's is 4"),
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
        let ok = |l, rows, shape: &[u64], t: &[u64], n: [u64; 3]| check_v7("k", l, rows, 64, shape, t, n).map_err(|e| e.msg);
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
