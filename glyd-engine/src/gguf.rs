//! GGUF files, read in place: the header, the key-value metadata, the tensor table and the data (memory-mapped), and what the engine's loaders need of them, so that a model given as a `.gguf` file
//! (or a Hub repo that holds one) runs where its types are ones the engine has.
//!
//! The parser (`Gguf::parse`) works on bytes, checks every count and offset it takes from them against the bytes there are (so a corrupt or hostile file is an error, never a panic or a huge
//! allocation), and keeps a tensor of a type it does not know (its size unknown) so that a file with one is still listed. `File::open` maps a file and parses it.
//!
//! What a GGUF Qwen3 becomes is a Hugging Face directory in the cache (`$GLYD_CACHE/gguf/<name>-<key>/`: config.json from the metadata, the tokenizer and chat template from the vocabulary, merges
//! and template there, generation_config.json, and model.safetensors holding the tensors under their Hugging Face names), which the engine then packs as it packs any checkpoint. Nothing is
//! converted silently: BF16 tensors are copied as they are; F32 and F16 ones become bf16 only where every value is exactly a bf16 value (llama.cpp keeps a BF16 model's norms in F32: those are);
//! every other type (the quantized ones, whose kernels the engine does not have yet) is refused, naming the types the file holds.

use crate::args::Args;
use crate::ffi::{err, Error, Result};
use crate::hub::{self, Hub};
use crate::mapped::Mapping;
use crate::ui::{Done, Refusal, Ui};
use glyd_gpu::json::{self, Value as J};
use sha2::{Digest, Sha256};
use std::io::Write;
use std::path::{Path, PathBuf};

/// The file's first bytes.
pub const MAGIC: &[u8; 4] = b"GGUF";

/// Metadata arrays are not read past this many elements (a vocabulary is about 10^5).
const MAX_ARRAY: u64 = 1 << 26;
/// Dimensions a tensor may have (ggml's limit is 4).
const MAX_DIMS: u32 = 8;

/// ggml's tensor types: id, name, elements in a block, bytes in a block. (31 to 33 were the online-repacked Q4_0 types, since removed from ggml.)
const TYPES: &[(u32, &str, u64, u64)] = &[
    (0, "F32", 1, 4),
    (1, "F16", 1, 2),
    (2, "Q4_0", 32, 18),
    (3, "Q4_1", 32, 20),
    (6, "Q5_0", 32, 22),
    (7, "Q5_1", 32, 24),
    (8, "Q8_0", 32, 34),
    (9, "Q8_1", 32, 36),
    (10, "Q2_K", 256, 84),
    (11, "Q3_K", 256, 110),
    (12, "Q4_K", 256, 144),
    (13, "Q5_K", 256, 176),
    (14, "Q6_K", 256, 210),
    (15, "Q8_K", 256, 292),
    (16, "IQ2_XXS", 256, 66),
    (17, "IQ2_XS", 256, 74),
    (18, "IQ3_XXS", 256, 98),
    (19, "IQ1_S", 256, 50),
    (20, "IQ4_NL", 32, 18),
    (21, "IQ3_S", 256, 110),
    (22, "IQ2_S", 256, 82),
    (23, "IQ4_XS", 256, 136),
    (24, "I8", 1, 1),
    (25, "I16", 1, 2),
    (26, "I32", 1, 4),
    (27, "I64", 1, 8),
    (28, "F64", 1, 8),
    (29, "IQ1_M", 256, 56),
    (30, "BF16", 1, 2),
    (34, "TQ1_0", 256, 54),
    (35, "TQ2_0", 256, 66),
    (39, "MXFP4", 32, 17),
];

pub const F32: u32 = 0;
pub const F16: u32 = 1;
pub const BF16: u32 = 30;

/// The name of ggml type `t`, its elements a block and its bytes a block; None for a type this does not know.
pub fn type_info(t: u32) -> Option<(&'static str, u64, u64)> {
    TYPES.iter().find(|x| x.0 == t).map(|&(_, n, b, s)| (n, b, s))
}

/// A metadata value.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    U(u64),
    I(i64),
    F(f64),
    Bool(bool),
    Str(String),
    Array(Vec<Value>),
}

impl Value {
    pub fn as_u64(&self) -> Option<u64> {
        match *self {
            Value::U(u) => Some(u),
            Value::I(i) => u64::try_from(i).ok(),
            _ => None,
        }
    }
    pub fn as_f64(&self) -> Option<f64> {
        match *self {
            Value::F(f) => Some(f),
            Value::U(u) => Some(u as f64),
            Value::I(i) => Some(i as f64),
            _ => None,
        }
    }
    pub fn as_bool(&self) -> Option<bool> {
        match *self {
            Value::Bool(b) => Some(b),
            _ => None,
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::Str(s) => Some(s),
            _ => None,
        }
    }
    pub fn as_array(&self) -> Option<&[Value]> {
        match self {
            Value::Array(a) => Some(a),
            _ => None,
        }
    }
}

/// A tensor of the file's table.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Tensor {
    pub name: String,
    /// ggml's `ne`: the elements along each dimension, the fastest-varying (a matrix's columns, its inputs) first.
    pub dims: Vec<u64>,
    /// The ggml type's id (`type_info`).
    pub ty: u32,
    /// Where its bytes start, from the start of the data (a multiple of the alignment).
    pub offset: u64,
    /// What its bytes take, where the type is known.
    pub bytes: Option<u64>,
    /// The part of a split file it is in (0: the file itself).
    pub part: usize,
}

impl Tensor {
    /// The shape as row-major (and Hugging Face) lists it: the slowest-varying dimension first.
    pub fn shape(&self) -> Vec<u64> {
        self.dims.iter().rev().copied().collect()
    }
    pub fn elements(&self) -> u64 {
        self.dims.iter().product()
    }
    /// The type's name, or `type N` for one this does not know.
    pub fn type_name(&self) -> String {
        type_info(self.ty).map_or_else(|| format!("type {}", self.ty), |t| t.0.to_string())
    }
}

/// A GGUF file's header: its metadata and its tensor table.
#[derive(Clone, Debug)]
pub struct Gguf {
    pub version: u32,
    pub meta: Vec<(String, Value)>,
    pub tensors: Vec<Tensor>,
    /// `general.alignment`, else 32.
    pub alignment: u64,
    /// Where the data starts in the file.
    pub data_offset: u64,
    pub file_len: u64,
}

struct Rd<'a> {
    b: &'a [u8],
    p: usize,
}

impl<'a> Rd<'a> {
    fn left(&self) -> u64 {
        (self.b.len() - self.p) as u64
    }
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self.p.checked_add(n).filter(|&e| e <= self.b.len()).ok_or_else(|| Error("the file ends inside its header".into()))?;
        let s = &self.b[self.p..end];
        self.p = end;
        Ok(s)
    }
    fn arr<const N: usize>(&mut self) -> Result<[u8; N]> {
        Ok(self.take(N)?.try_into().unwrap())
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.arr()?))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.arr()?))
    }
    fn string(&mut self) -> Result<String> {
        let n = self.u64()?;
        let n = usize::try_from(n).ok().filter(|&n| n as u64 <= self.left()).ok_or_else(|| Error(format!("a string of {n} bytes, past the file's header")))?;
        Ok(String::from_utf8_lossy(self.take(n)?).into_owned())
    }

    /// A value of metadata type `ty`; `depth` the arrays it is inside.
    fn value(&mut self, ty: u32, depth: u32) -> Result<Value> {
        Ok(match ty {
            0 => Value::U(u8::from_le_bytes(self.arr()?) as u64),
            1 => Value::I(i8::from_le_bytes(self.arr()?) as i64),
            2 => Value::U(u16::from_le_bytes(self.arr()?) as u64),
            3 => Value::I(i16::from_le_bytes(self.arr()?) as i64),
            4 => Value::U(self.u32()? as u64),
            5 => Value::I(i32::from_le_bytes(self.arr()?) as i64),
            6 => Value::F(f32::from_le_bytes(self.arr()?) as f64),
            7 => match self.take(1)?[0] {
                0 => Value::Bool(false),
                1 => Value::Bool(true),
                b => return err(format!("a boolean of {b}")),
            },
            8 => Value::Str(self.string()?),
            9 => {
                if depth >= 4 {
                    return err("arrays nested past 4 deep");
                }
                let et = self.u32()?;
                let n = self.u64()?;
                // (every element takes at least this: a string its length, an array its header)
                let least: u64 = match et {
                    0 | 1 | 7 => 1,
                    2 | 3 => 2,
                    4 | 5 | 6 => 4,
                    10 | 11 | 12 | 8 => 8,
                    9 => 12,
                    t => return err(format!("an array of metadata type {t}")),
                };
                if n > MAX_ARRAY || n > self.left() / least {
                    return err(format!("an array of {n} elements, past the file's header"));
                }
                let mut v = Vec::with_capacity(n as usize);
                for _ in 0..n {
                    v.push(self.value(et, depth + 1)?);
                }
                Value::Array(v)
            }
            10 => Value::U(self.u64()?),
            11 => Value::I(i64::from_le_bytes(self.arr()?)),
            12 => Value::F(f64::from_le_bytes(self.arr()?)),
            t => return err(format!("metadata of type {t}")),
        })
    }
}

fn align_up(x: u64, a: u64) -> Option<u64> {
    x.checked_add(a - 1).map(|y| y / a * a)
}

impl Gguf {
    /// The header of `bytes`, a whole file.
    pub fn parse(bytes: &[u8]) -> Result<Gguf> {
        Gguf::parse_header(bytes, bytes.len() as u64)
    }

    /// The header read from `bytes`, the start of a file of `file_len` bytes (the tensors are checked against that length: a header fetched alone gives the model's sizes, `bytes` need hold
    /// no more than the header and the padding up to the data).
    pub fn parse_header(bytes: &[u8], file_len: u64) -> Result<Gguf> {
        let mut r = Rd { b: bytes, p: 0 };
        if r.take(4).ok() != Some(&MAGIC[..]) {
            return err("not a GGUF file");
        }
        let version = r.u32()?;
        if version == 1 {
            return err("GGUF version 1 (the file is older than any llama.cpp has read for a year)");
        }
        if version.swap_bytes() == 3 || version.swap_bytes() == 2 {
            return err("a big-endian GGUF file");
        }
        if !matches!(version, 2 | 3) {
            return err(format!("GGUF version {version}: this reads 2 and 3"));
        }
        let (n_tensors, n_kv) = (r.u64()?, r.u64()?);
        // (a key-value pair takes at least 12 bytes, a tensor's entry 24)
        if n_kv > r.left() / 12 || n_tensors > r.left() / 24 {
            return err(format!("{n_tensors} tensors and {n_kv} metadata entries, more than the header's bytes hold"));
        }
        let mut meta: Vec<(String, Value)> = Vec::with_capacity(n_kv as usize);
        for _ in 0..n_kv {
            let key = r.string()?;
            let ty = r.u32()?;
            let v = r.value(ty, 0).map_err(|e| Error(format!("{key}: {e}")))?;
            meta.push((key, v));
        }
        let alignment = match meta.iter().find(|(k, _)| k == "general.alignment") {
            Some((_, v)) => v.as_u64().filter(|a| a.is_power_of_two() && *a <= 1 << 20).ok_or_else(|| Error("general.alignment: not a power of two".into()))?,
            None => 32,
        };
        let mut tensors: Vec<Tensor> = Vec::with_capacity(n_tensors as usize);
        let mut seen = std::collections::HashSet::with_capacity(n_tensors as usize);
        for _ in 0..n_tensors {
            let name = r.string()?;
            let nd = r.u32()?;
            if nd == 0 || nd > MAX_DIMS {
                return err(format!("{name}: {nd} dimensions"));
            }
            let mut dims = Vec::with_capacity(nd as usize);
            for _ in 0..nd {
                dims.push(r.u64()?);
            }
            let (ty, offset) = (r.u32()?, r.u64()?);
            let elements = dims.iter().try_fold(1u64, |a, &d| a.checked_mul(d)).ok_or_else(|| Error(format!("{name}: more elements than 2^64")))?;
            let bytes = match type_info(ty) {
                Some((tn, block, size)) => {
                    if dims[0] % block != 0 {
                        return err(format!("{name}: {} elements in a row, not a multiple of {tn}'s block of {block}", dims[0]));
                    }
                    Some((elements / block).checked_mul(size).ok_or_else(|| Error(format!("{name}: more bytes than 2^64")))?)
                }
                None => None,
            };
            if !seen.insert(name.clone()) {
                return err(format!("{name}: twice in the tensor table"));
            }
            if offset % alignment != 0 {
                return err(format!("{name}: its data at {offset}, not a multiple of the alignment {alignment}"));
            }
            tensors.push(Tensor { name, dims, ty, offset, bytes, part: 0 });
        }
        // (a part of a split that holds the metadata alone ends where its header does, with no padding to the alignment)
        let data_offset = match align_up(r.p as u64, alignment) {
            Some(d) if d <= file_len => d,
            _ if tensors.is_empty() && r.p as u64 <= file_len => r.p as u64,
            _ => return err("the file ends before its data"),
        };
        let data = file_len - data_offset;
        for t in &tensors {
            if let Some(b) = t.bytes {
                if t.offset.checked_add(b).is_none_or(|e| e > data) {
                    return err(format!("{}: {b} bytes at {}, past the file's {data} bytes of data", t.name, t.offset));
                }
            }
        }
        Ok(Gguf { version, meta, tensors, alignment, data_offset, file_len })
    }

    pub fn get(&self, key: &str) -> Option<&Value> {
        self.meta.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }
    pub fn str(&self, key: &str) -> Option<&str> {
        self.get(key)?.as_str()
    }
    pub fn u64(&self, key: &str) -> Option<u64> {
        self.get(key)?.as_u64()
    }
    pub fn f64(&self, key: &str) -> Option<f64> {
        self.get(key)?.as_f64()
    }
    pub fn tensor(&self, name: &str) -> Option<&Tensor> {
        self.tensors.iter().find(|t| t.name == name)
    }

    /// The tensors' types and how many of each, in the order they first appear.
    pub fn types(&self) -> Vec<(String, usize)> {
        let mut v: Vec<(String, usize)> = Vec::new();
        for t in &self.tensors {
            let n = t.type_name();
            match v.iter_mut().find(|(k, _)| *k == n) {
                Some(e) => e.1 += 1,
                None => v.push((n, 1)),
            }
        }
        v
    }

    /// What the metadata says the model is.
    pub fn description(&self) -> Result<Description> {
        let arch = self.str("general.architecture").ok_or_else(|| Error("no general.architecture".into()))?.to_string();
        let k = |s: &str| self.u64(&format!("{arch}.{s}"));
        let tokens = self.get("tokenizer.ggml.tokens").and_then(Value::as_array).map(|a| a.len() as u64);
        let embedding = self.tensor("token_embd.weight").and_then(|t| t.dims.get(1).copied());
        let hidden = k("embedding_length");
        let heads = k("attention.head_count");
        // (head_count_kv is a list in models whose layers differ: not read as one number)
        let kv_heads = k("attention.head_count_kv");
        Ok(Description {
            name: self.str("general.name").map(str::to_string),
            context: k("context_length"),
            layers: k("block_count"),
            ffn: k("feed_forward_length"),
            head_dim: k("attention.key_length").or_else(|| hidden?.checked_div(heads?)),
            eps: self.f64(&format!("{arch}.attention.layer_norm_rms_epsilon")),
            rope_theta: self.f64(&format!("{arch}.rope.freq_base")),
            rope_scaling: self.str(&format!("{arch}.rope.scaling.type")).filter(|s| *s != "none").map(str::to_string),
            experts: k("expert_count"),
            vocab: tokens,
            tokenizer: self.str("tokenizer.ggml.model").map(str::to_string),
            tokenizer_pre: self.str("tokenizer.ggml.pre").map(str::to_string),
            chat_template: self.str("tokenizer.chat_template").is_some(),
            file_type: self.u64("general.file_type"),
            hidden: hidden.or(embedding),
            heads,
            kv_heads,
            arch,
        })
    }
}

/// The model the metadata describes, in the fields the engine's configs have. A field the file does not give is None.
#[derive(Clone, Debug, PartialEq)]
pub struct Description {
    /// `general.architecture` ("qwen3", "llama", "gemma3" ...).
    pub arch: String,
    pub name: Option<String>,
    pub context: Option<u64>,
    pub hidden: Option<u64>,
    pub layers: Option<u64>,
    pub heads: Option<u64>,
    pub kv_heads: Option<u64>,
    pub head_dim: Option<u64>,
    pub ffn: Option<u64>,
    pub eps: Option<f64>,
    pub rope_theta: Option<f64>,
    /// The rope scaling's kind, where the model has one.
    pub rope_scaling: Option<String>,
    pub experts: Option<u64>,
    /// The vocabulary's size where the file lists the tokens.
    pub vocab: Option<u64>,
    /// `tokenizer.ggml.model` ("gpt2" is byte-level BPE) and `tokenizer.ggml.pre` (the pre-tokenizer's name).
    pub tokenizer: Option<String>,
    pub tokenizer_pre: Option<String>,
    pub chat_template: bool,
    /// llama.cpp's file type (`general.file_type`: 32 is BF16, 1 F16, 7 Q8_0 ...).
    pub file_type: Option<u64>,
}

/// One file of a model, mapped.
struct Part {
    path: PathBuf,
    map: Mapping,
    data_offset: u64,
    file_len: u64,
}

impl Part {
    fn bytes(&self) -> &[u8] {
        // SAFETY: a read-only mapping of `len` bytes, alive as long as the part is.
        unsafe { std::slice::from_raw_parts(self.map.ptr as *const u8, self.map.len) }
    }
}

/// `NAME-00002-of-00005.gguf`: the part's name without its numbers, its number (from 1) and the parts there are.
pub fn split_of(name: &str) -> Option<(&str, u32, u32)> {
    let (head, of) = name.strip_suffix(".gguf")?.rsplit_once("-of-")?;
    let (prefix, n) = head.rsplit_once('-')?;
    let digits = |s: &str| (s.len() == 5 && s.bytes().all(|c| c.is_ascii_digit())).then(|| s.parse::<u32>().ok()).flatten();
    let (n, m) = (digits(n)?, digits(of)?);
    (n >= 1 && n <= m).then_some((prefix, n, m))
}

/// The name of part `n` of `m` of the split `prefix`.
pub fn split_name(prefix: &str, n: u32, m: u32) -> String {
    format!("{prefix}-{n:05}-of-{m:05}.gguf")
}

/// The headers of a model's files (`label` names each in an error), in the order of its parts, as one: the first file's metadata and the tensors of all, each saying which file it is in. A split's
/// parts must say they are its parts (llama.cpp writes `split.no`, from 0, and `split.count` in each part's metadata, and the model's tensors' number in the first's).
pub fn merge(headers: Vec<(String, Gguf)>) -> Result<Gguf> {
    let n = headers.len();
    let mut merged: Option<Gguf> = None;
    for (i, (label, g)) in headers.into_iter().enumerate() {
        if n > 1 {
            if let Some(c) = g.u64("split.count").filter(|&c| c != n as u64) {
                return err(format!("{label}: split.count is {c}, and its name says {n} parts"));
            }
            if let Some(no) = g.u64("split.no").filter(|&no| no != i as u64) {
                return err(format!("{label}: it is part {} of the split by its metadata (split.no {no}), and part {} by its name", no + 1, i + 1));
            }
        }
        match merged.as_mut() {
            None => merged = Some(g),
            Some(m) => {
                for mut t in g.tensors {
                    t.part = i;
                    if m.tensor(&t.name).is_some() {
                        return err(format!("{label}: {} is in two parts of the split", t.name));
                    }
                    m.tensors.push(t);
                }
            }
        }
    }
    let gguf = merged.ok_or_else(|| Error("no file".into()))?;
    if let Some(c) = gguf.u64("split.tensors.count").filter(|&c| c != gguf.tensors.len() as u64) {
        return err(format!("split.tensors.count is {c}, and the parts have {} tensors", gguf.tensors.len()));
    }
    Ok(gguf)
}

/// A model's file or files, mapped and parsed: `gguf` has the first file's metadata and the tensors of all the files (each says which file it is in), and the tensors' bytes are read in place.
pub struct File {
    /// The first file's path.
    pub path: PathBuf,
    pub gguf: Gguf,
    parts: Vec<Part>,
}

impl File {
    /// The model at `path`: a file, or any one file of a split (`NAME-00002-of-00005.gguf`), whose others are found beside it and must all be there.
    pub fn open(path: &Path) -> Result<File> {
        let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        let dir = path.parent().unwrap_or(Path::new("."));
        let paths: Vec<PathBuf> = match split_of(&name) {
            Some((prefix, _, m)) => (1..=m).map(|n| dir.join(split_name(prefix, n, m))).collect(),
            None => vec![path.to_path_buf()],
        };
        let mut parts = Vec::new();
        let mut headers = Vec::new();
        for (i, p) in paths.iter().enumerate() {
            if !p.is_file() {
                return err(format!(
                    "{}: {}",
                    p.display(),
                    if paths.len() > 1 { format!("part {} of {} of a split GGUF is not there", i + 1, paths.len()) } else { "no such file".into() }
                ));
            }
            let map = Mapping::open(p)?;
            // SAFETY: as in `Part::bytes`.
            let bytes = unsafe { std::slice::from_raw_parts(map.ptr as *const u8, map.len) };
            let g = Gguf::parse(bytes).map_err(|e| Error(format!("{}: {e}", p.display())))?;
            parts.push(Part { path: p.clone(), map, data_offset: g.data_offset, file_len: g.file_len });
            headers.push((p.display().to_string(), g));
        }
        let gguf = merge(headers)?;
        Ok(File { path: paths[0].clone(), gguf, parts })
    }

    /// The files' headers (up to their data): what names the model's contents.
    fn header_bytes(&self) -> Vec<u8> {
        self.parts.iter().flat_map(|p| p.file_len.to_le_bytes().into_iter().chain(p.bytes()[..p.data_offset as usize].iter().copied())).collect()
    }

    /// The files' lengths added, and each file's length and modification time as text.
    fn stamp(&self) -> (u64, String) {
        let one = |p: &Part| {
            let t = std::fs::metadata(&p.path)
                .ok()
                .and_then(|m| m.modified().ok())
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map_or(0, |d| d.as_nanos());
            format!("{}:{t}", p.file_len)
        };
        (self.parts.iter().map(|p| p.file_len).sum(), self.parts.iter().map(one).collect::<Vec<_>>().join(","))
    }

    /// Tensor `t`'s bytes, in the mapping; an error for a type whose size is not known.
    pub fn data(&self, t: &Tensor) -> Result<&[u8]> {
        let n = t.bytes.ok_or_else(|| Error(format!("{}: {}, a type this does not know", t.name, t.type_name())))?;
        let p = &self.parts[t.part];
        let start = (p.data_offset + t.offset) as usize;
        Ok(&p.bytes()[start..start + n as usize])
    }
}

// ---- GGUF names and types as the engine's loaders have them ----------------------------------------------------------------------------------------

/// The Hugging Face name of a GGUF tensor of a Qwen3 (dense) model, else None.
pub fn hf_name(name: &str) -> Option<String> {
    let top = match name {
        "token_embd.weight" => Some("model.embed_tokens.weight"),
        "output.weight" => Some("lm_head.weight"),
        "output_norm.weight" => Some("model.norm.weight"),
        _ => None,
    };
    if let Some(t) = top {
        return Some(t.to_string());
    }
    let rest = name.strip_prefix("blk.")?;
    let (n, what) = rest.split_once('.')?;
    n.parse::<u32>().ok()?;
    let to = match what {
        "attn_norm.weight" => "input_layernorm.weight",
        "ffn_norm.weight" => "post_attention_layernorm.weight",
        "attn_q.weight" => "self_attn.q_proj.weight",
        "attn_k.weight" => "self_attn.k_proj.weight",
        "attn_v.weight" => "self_attn.v_proj.weight",
        "attn_output.weight" => "self_attn.o_proj.weight",
        "attn_q_norm.weight" => "self_attn.q_norm.weight",
        "attn_k_norm.weight" => "self_attn.k_norm.weight",
        "ffn_gate.weight" => "mlp.gate_proj.weight",
        "ffn_up.weight" => "mlp.up_proj.weight",
        "ffn_down.weight" => "mlp.down_proj.weight",
        _ => return None,
    };
    Some(format!("model.layers.{n}.{to}"))
}

/// What llama.cpp's converter (conversion/*.py) does to the tensors of the recent families' checkpoints on its way to a GGUF, by GGUF architecture: the family, and what is changed. None of it
/// can be undone bit for bit (a bf16 weight plus one is rounded again, an exponential is not a logarithm's inverse in 32 bits), or it is a layout the release's loader does not have, so a
/// GGUF of these is not the release's checkpoint and the engine, which loads the checkpoint, does not run it.
const CONVERTED: &[(&[&str], &str, &str)] = &[
    (
        &["qwen35", "qwen35moe", "qwen3next"],
        "Qwen3.5, 3.6 and 3.8",
        "norm weights stored as w+1, A_log as -exp(A_log), the linear attention's V heads reordered and its projections merged",
    ),
    (&["qwen4exp"], "Qwen3.8 Flash-Next", "norm gammas stored as w+1, projections split and merged"),
    (&["gemma2", "gemma3", "gemma3n", "gemma4", "gemma4-assistant"], "Gemma", "norm weights stored as w+1"),
    (
        &["muse-spark", "muse-glimmer"],
        "Muse Spark",
        "norm weights stored as w+1, the output multiplier folded into the output weights, RoPE rows un-permuted, the QK norms replaced",
    ),
    (&["kimi-k3", "kimi-k31"], "Kimi-K3", "A_log stored as -exp(A_log), the attention residuals' norms fused into projections, experts stacked"),
    (&["deepseek41"], "DeepSeek-V4", "engram gates folded with their scales, experts stacked"),
    (&["glm5-next", "glm-dsa", "glm4moe"], "GLM", "experts stacked, kv_b split into k_b and v_b"),
];

/// Whether the engine reads this architecture from GGUF: Qwen3 dense, with no rope scaling.
fn check_arch(d: &Description) -> Done<()> {
    if let Some((_, family, what)) = CONVERTED.iter().find(|(archs, _, _)| archs.contains(&d.arch.as_str())) {
        return Err(Refusal::unsupported(
            format!(
                "GGUF architecture {} ({family}): the file holds llama.cpp's converted tensors ({what}), not the release's checkpoint, and the engine runs the checkpoint",
                d.arch
            ),
            "Run the model's Hugging Face checkpoint (safetensors) instead",
        ));
    }
    if d.arch != "qwen3" {
        return Err(Refusal::unsupported(
            format!("GGUF architecture {}: the engine reads Qwen3 dense from GGUF so far", d.arch),
            "Run the model's Hugging Face checkpoint (safetensors) instead",
        ));
    }
    if let Some(s) = &d.rope_scaling {
        return Err(Refusal::unsupported(format!("rope scaling ({s}): not supported yet"), ""));
    }
    Ok(())
}

/// The config.json of a Qwen3 (dense) GGUF: what the metadata says, in the fields `weights::Config::parse` reads. Another architecture is refused.
pub fn config_json(g: &Gguf, d: &Description) -> Done<String> {
    check_arch(d)?;
    let need = |v: Option<u64>, k: &str| v.ok_or_else(|| Refusal::new(format!("the GGUF's metadata has no qwen3.{k}"), ""));
    let embed = g.tensor("token_embd.weight").ok_or_else(|| Refusal::new("the GGUF has no token_embd.weight", ""))?;
    let (hidden, vocab) = (embed.dims[0], embed.dims.get(1).copied().unwrap_or(0));
    let num = |x: u64| J::Number(x.to_string());
    let float = |x: f64| J::Number(format!("{x:?}"));
    let eps = d.eps.ok_or_else(|| Refusal::new("the GGUF's metadata has no qwen3.attention.layer_norm_rms_epsilon", ""))?;
    let theta = d.rope_theta.ok_or_else(|| Refusal::new("the GGUF's metadata has no qwen3.rope.freq_base", ""))?;
    let mut o: Vec<(String, J)> = vec![
        ("architectures".into(), J::Array(vec![J::String("Qwen3ForCausalLM".into())])),
        ("model_type".into(), J::String("qwen3".into())),
        ("hidden_size".into(), num(hidden)),
        ("num_hidden_layers".into(), num(need(d.layers, "block_count")?)),
        ("num_attention_heads".into(), num(need(d.heads, "attention.head_count")?)),
        ("num_key_value_heads".into(), num(need(d.kv_heads, "attention.head_count_kv")?)),
        ("head_dim".into(), num(need(d.head_dim, "attention.key_length")?)),
        ("intermediate_size".into(), num(need(d.ffn, "feed_forward_length")?)),
        ("vocab_size".into(), num(vocab)),
        ("rms_norm_eps".into(), float(eps)),
        ("rope_theta".into(), float(theta)),
        ("rope_scaling".into(), J::Null),
        ("max_position_embeddings".into(), num(need(d.context, "context_length")?)),
        ("tie_word_embeddings".into(), J::Bool(g.tensor("output.weight").is_none())),
        ("hidden_act".into(), J::String("silu".into())),
        ("attention_bias".into(), J::Bool(false)),
        ("torch_dtype".into(), J::String("bfloat16".into())),
    ];
    if let Some(id) = g.u64("tokenizer.ggml.bos_token_id") {
        o.push(("bos_token_id".into(), num(id)));
    }
    if let Some(id) = g.u64("tokenizer.ggml.eos_token_id") {
        o.push(("eos_token_id".into(), num(id)));
    }
    Ok(json::to_python(&J::Object(o), 2))
}

/// generation_config.json: the end-of-sequence ids the metadata gives (`eos`, and `eot` where it has one).
pub fn generation_json(g: &Gguf) -> String {
    let ids: Vec<J> =
        ["tokenizer.ggml.eos_token_id", "tokenizer.ggml.eot_token_id"].iter().filter_map(|k| g.u64(k)).map(|i| J::Number(i.to_string())).collect();
    json::to_python(&J::Object(vec![("eos_token_id".into(), J::Array(ids))]), 2)
}

/// tokenizer.json and tokenizer_config.json from the vocabulary of `g` (src/gguf_tok.rs).
pub fn tokenizer_files(g: &Gguf) -> Done<(String, String)> {
    crate::gguf_tok::files(g)
}

// ---- the weights as a safetensors file ------------------------------------------------------------------------------------------------------------

/// A half-precision float as an f32.
pub fn f16_to_f32(h: u16) -> f32 {
    let (s, e, m) = (((h >> 15) as u32) << 31, ((h >> 10) & 0x1f) as u32, (h & 0x3ff) as u32);
    f32::from_bits(match (e, m) {
        (0, 0) => s,
        (0, _) => {
            // a subnormal: normalised
            let (mut e, mut m) = (113u32, m);
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            s | (e << 23) | ((m & 0x3ff) << 13)
        }
        (31, _) => s | 0x7f80_0000 | (m << 13),
        _ => s | ((e + 112) << 23) | (m << 13),
    })
}

/// How a tensor's bytes become the BF16 ones the checkpoint holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Cast {
    Copy,
    F32,
    F16,
}

/// The tensors of `g` that are not stored as BF16, F16 or F32 (the types, with how many tensors of each), or the ones with a type this does not know.
fn unsupported_types(g: &Gguf) -> Vec<(String, usize)> {
    let mut v: Vec<(String, usize)> = Vec::new();
    for t in &g.tensors {
        if !matches!(t.ty, F32 | F16 | BF16) {
            let n = t.type_name();
            match v.iter_mut().find(|(k, _)| *k == n) {
                Some(e) => e.1 += 1,
                None => v.push((n, 1)),
            }
        }
    }
    v
}

fn quantized_refusal(file: &str, types: &[(String, usize)]) -> Refusal {
    let list: Vec<String> = types.iter().map(|(t, n)| format!("{t} ({n} tensors)")).collect();
    Refusal::unsupported(
        format!("{file}: the GGUF holds {}; the engine has no kernels for quantized GGUF types yet", list.join(", ")),
        "Run a BF16 GGUF of the model (or the model's Hugging Face checkpoint)",
    )
}

/// Writes `g`'s tensors from `f` to `out` as a safetensors file of BF16 tensors under their Hugging Face names. Every F32 or F16 value must be a bf16 value exactly.
fn write_safetensors(f: &File, out: &Path) -> Done<()> {
    let io = |e: std::io::Error| Refusal::new(format!("{}: {e}", out.display()), "");
    let mut entries = Vec::new();
    let mut plan: Vec<(&Tensor, Cast)> = Vec::new();
    let mut pos = 0u64;
    // (in the order of their names: the file is the same whatever the parts the tensors were in)
    let mut sorted: Vec<&Tensor> = f.gguf.tensors.iter().collect();
    sorted.sort_by(|a, b| a.name.cmp(&b.name));
    for t in sorted {
        let name = hf_name(&t.name).ok_or_else(|| Refusal::unsupported(format!("{}: a GGUF tensor the engine's Qwen3 loader has no name for", t.name), ""))?;
        let cast = match t.ty {
            BF16 => Cast::Copy,
            F32 => Cast::F32,
            F16 => Cast::F16,
            _ => unreachable!("the types were checked"),
        };
        let bytes = t.elements() * 2;
        entries.push((
            name,
            J::Object(vec![
                ("dtype".into(), J::String("BF16".into())),
                ("shape".into(), J::Array(t.shape().iter().map(|d| J::Number(d.to_string())).collect())),
                ("data_offsets".into(), J::Array(vec![J::Number(pos.to_string()), J::Number((pos + bytes).to_string())])),
            ]),
        ));
        pos += bytes;
        plan.push((t, cast));
    }
    entries.insert(0, ("__metadata__".into(), J::Object(vec![("format".into(), J::String("pt".into()))])));
    let mut header = json::to_compact(&J::Object(entries)).into_bytes();
    while header.len() % 8 != 0 {
        header.push(b' ');
    }
    let mut w = std::io::BufWriter::with_capacity(8 << 20, std::fs::File::create(out).map_err(io)?);
    w.write_all(&(header.len() as u64).to_le_bytes()).map_err(io)?;
    w.write_all(&header).map_err(io)?;
    let mut buf: Vec<u8> = Vec::new();
    for (t, cast) in plan {
        let data = f.data(t)?;
        if cast == Cast::Copy {
            w.write_all(data).map_err(io)?;
            continue;
        }
        let width = if cast == Cast::F32 { 4 } else { 2 };
        for (c, chunk) in data.chunks(1 << 22).enumerate() {
            buf.clear();
            for (i, x) in chunk.chunks_exact(width).enumerate() {
                let v = if cast == Cast::F32 { f32::from_le_bytes([x[0], x[1], x[2], x[3]]) } else { f16_to_f32(u16::from_le_bytes([x[0], x[1]])) };
                let bits = v.to_bits();
                if bits & 0xffff != 0 && !v.is_nan() {
                    let at = c * ((1 << 22) / width) + i;
                    return Err(Refusal::unsupported(
                        format!(
                            "{}: {} tensor {} is not exactly bf16 (its value {v:e} at element {at} has more than 8 bits of significand), and the engine does not round a GGUF's weights",
                            f.path.display(),
                            t.type_name(),
                            t.name
                        ),
                        "Run the BF16 GGUF of the model (or the model's Hugging Face checkpoint)",
                    ));
                }
                buf.extend_from_slice(&((bits >> 16) as u16).to_le_bytes());
            }
            w.write_all(&buf).map_err(io)?;
        }
    }
    w.flush().map_err(io)
}

/// The Hugging Face directory of the GGUF file at `path`, made under `root` where it is not there (named by the file's name and a key of its header, so that a changed file is a new
/// directory): config.json, generation_config.json, tokenizer.json, tokenizer_config.json and model.safetensors. A model of another architecture, or a quantized one, is refused before
/// anything is written.
pub fn materialize(path: &Path, root: &Path, log: &dyn Fn(&str)) -> Done<PathBuf> {
    let f = File::open(path)?;
    let g = &f.gguf;
    let d = g.description()?;
    let first = f.path.file_name().map_or_else(|| path.display().to_string(), |n| n.to_string_lossy().into_owned());
    // (a split's name is its parts' without their numbers)
    let file = split_of(&first).map_or(first.clone(), |(prefix, _, m)| format!("{prefix}.gguf ({m} parts)"));
    check_arch(&d)?;
    let q = unsupported_types(g);
    if !q.is_empty() {
        return Err(quantized_refusal(&file, &q));
    }
    let config = config_json(g, &d)?;
    let (tokenizer, tokenizer_config) = crate::gguf_tok::files(g)?;
    let mut h = Sha256::new();
    h.update(f.header_bytes());
    let key: String = h.finalize().iter().take(6).map(|b| format!("{b:02x}")).collect();
    let stem = split_of(&first).map_or(first.strip_suffix(".gguf").unwrap_or(&first), |(prefix, _, _)| prefix);
    let stem: String = stem.chars().map(|c| if c.is_ascii_alphanumeric() || matches!(c, '-' | '.') { c } else { '_' }).collect();
    let dir = root.join(format!("{stem}-{key}"));
    // (the directory is the file's own while its length and modification time are the ones it was made from: the key is the header's, which a rewrite of the weights in place leaves as it was)
    let (size, stamp) = f.stamp();
    let source = json::to_python(
        &J::Object(vec![("file".into(), J::String(file.clone())), ("size".into(), J::Number(size.to_string())), ("files".into(), J::String(stamp))]),
        2,
    );
    if dir.join("model.safetensors").exists() && dir.join("config.json").exists() {
        if std::fs::read_to_string(dir.join("gguf-source.json")).is_ok_and(|t| t == source) {
            return Ok(dir);
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
    let io = |e: std::io::Error| Refusal::new(format!("{}: {e}", root.display()), "");
    std::fs::create_dir_all(root).map_err(io)?;
    let tmp = root.join(format!(".tmp-{}-{stem}-{key}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).map_err(io)?;
    log(&format!(
        "glyd: reading {file} as a Hugging Face checkpoint ({:.1} GB of bf16 weights) in {}",
        g.tensors.iter().map(|t| t.elements() * 2).sum::<u64>() as f64 / 1e9,
        dir.display()
    ));
    let made = (|| -> Done<()> {
        for (name, text) in [
            ("config.json", &config),
            ("generation_config.json", &generation_json(g)),
            ("tokenizer.json", &tokenizer),
            ("tokenizer_config.json", &tokenizer_config),
        ] {
            std::fs::write(tmp.join(name), text).map_err(io)?;
        }
        std::fs::write(tmp.join("gguf-source.json"), &source).map_err(io)?;
        write_safetensors(&f, &tmp.join("model.safetensors"))
    })();
    if let Err(e) = made {
        let _ = std::fs::remove_dir_all(&tmp);
        return Err(e);
    }
    if std::fs::rename(&tmp, &dir).is_err() {
        // (another run made it first)
        let _ = std::fs::remove_dir_all(&tmp);
        if !dir.join("model.safetensors").exists() {
            return Err(Refusal::new(format!("{}: cannot be made", dir.display()), ""));
        }
    }
    Ok(dir)
}

// ---- a model given as a GGUF ----------------------------------------------------------------------------------------------------------------------

fn base(n: &str) -> &str {
    n.rsplit('/').next().unwrap_or(n)
}

/// The names of the parts of the split whose first part is `first` (all of them, `first` the first), or the one file `first`: an error naming the parts that are missing.
fn parts_of<'a>(names: &[&'a str], first: &'a str) -> Done<Vec<&'a str>> {
    let Some((prefix, _, m)) = split_of(base(first)) else { return Ok(vec![first]) };
    let dir = &first[..first.len() - base(first).len()];
    let mut out = Vec::new();
    for n in 1..=m {
        let want = format!("{dir}{}", split_name(prefix, n, m));
        match names.iter().find(|x| **x == want) {
            Some(x) => out.push(*x),
            None => {
                return Err(Refusal::new(
                    format!("part {n} of {m} of the split GGUF {first} is missing ({want})"),
                    "Download every part of the split, into one directory",
                ))
            }
        }
    }
    Ok(out)
}

/// Which of the GGUF files `names` (a directory's or a repo's) to run: the BF16 one, else F32, else F16 (as the names say; the file's own header decides what it is); a split model is its first
/// part (`parts_of` has the rest). An error, with the names said, where there is no unquantized one.
fn choose<'a>(names: &[&'a str]) -> Done<&'a str> {
    let ggufs: Vec<&str> = names.iter().copied().filter(|n| n.ends_with(".gguf") && !base(n).starts_with("mmproj")).collect();
    // one candidate for a model: its file, or its split's first part (a model in one file before the same in parts)
    let mut cands: Vec<&str> = ggufs.iter().copied().filter(|n| split_of(base(n)).is_none_or(|(_, k, _)| k == 1)).collect();
    cands.sort_by_key(|n| split_of(base(n)).is_some());
    for want in ["bf16", "f32", "f16"] {
        if let Some(n) = cands.iter().find(|n| base(n).to_ascii_lowercase().contains(want)) {
            parts_of(&ggufs, n)?;
            return Ok(n);
        }
    }
    if let [only] = cands[..] {
        parts_of(&ggufs, only)?;
        return Ok(only);
    }
    let shown: Vec<&str> = ggufs.iter().copied().take(8).collect();
    Err(Refusal::unsupported(
        format!("none of the GGUF files here is BF16, F32 or F16 ({}): the engine has no kernels for quantized GGUF types yet", shown.join(", ")),
        "Run a BF16 GGUF of the model (or the model's Hugging Face checkpoint)",
    ))
}

/// The `.gguf` files of `dir` and of its folders (as their paths from `dir`).
fn ggufs_in(dir: &Path) -> Vec<String> {
    let mut v = Vec::new();
    for e in std::fs::read_dir(dir).into_iter().flatten().flatten() {
        let Ok(name) = e.file_name().into_string() else { continue };
        if name.ends_with(".gguf") {
            v.push(name);
        } else if e.path().is_dir() {
            for f in std::fs::read_dir(e.path()).into_iter().flatten().flatten() {
                if let Some(f) = f.file_name().to_str().filter(|f| f.ends_with(".gguf")) {
                    v.push(format!("{name}/{f}"));
                }
            }
        }
    }
    v.sort();
    v
}

/// `model:level`'s model and its `:level` (an ASCII-lowercase word after the last colon), apart.
fn split_level(model: &str) -> (&str, &str) {
    match model.rfind(':') {
        Some(i) if !model[i + 1..].is_empty() && model[i + 1..].bytes().all(|c| c.is_ascii_lowercase()) && !model[..i].ends_with(':') => {
            (&model[..i], &model[i..])
        }
        _ => (model, ""),
    }
}

/// A planning command's directory for a GGUF's headers (what `fit`, `supports`, `preflight` and `costs` read of a model: its config.json and generation_config.json), made under `root` where it is
/// not there: the same refusals as a run's (the architecture, the quantized types), and no weights, so that nothing is read of the data. `label` names the model in a refusal.
pub fn plan_dir(g: &Gguf, label: &str, root: &Path) -> Done<PathBuf> {
    let d = g.description()?;
    check_arch(&d)?;
    let q = unsupported_types(g);
    if !q.is_empty() {
        return Err(quantized_refusal(label, &q));
    }
    let (config, generation) = (config_json(g, &d)?, generation_json(g));
    let mut h = Sha256::new();
    h.update(&config);
    h.update(&generation);
    let key: String = h.finalize().iter().take(6).map(|b| format!("{b:02x}")).collect();
    let stem: String =
        label.strip_suffix(".gguf").unwrap_or(label).chars().map(|c| if c.is_ascii_alphanumeric() || matches!(c, '-' | '.') { c } else { '_' }).collect();
    let dir = root.join(format!("{stem}-{key}"));
    if !dir.join("config.json").exists() {
        let io = |e: std::io::Error| Refusal::new(format!("{}: {e}", dir.display()), "");
        let tmp = root.join(format!(".tmp-{}-{stem}-{key}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).map_err(io)?;
        std::fs::write(tmp.join("generation_config.json"), &generation).map_err(io)?;
        std::fs::write(tmp.join("config.json"), &config).map_err(io)?;
        // the file's tensors by their headers, as the planner reads them (src/sizes.rs: the Hugging Face names the engine loads, the bytes the file holds)
        crate::sizes::Table::from_gguf(g).write(&tmp.join(crate::sizes::GGUF_TENSORS));
        if std::fs::rename(&tmp, &dir).is_err() {
            let _ = std::fs::remove_dir_all(&tmp);
        }
    }
    Ok(dir)
}

/// The headers of the GGUF files `parts` of a repo, read by ranges (nothing of the data is downloaded): each file's first bytes, doubled until its header parses.
fn hub_headers(hub: &Hub, remote: &hub::Remote, parts: &[hub::Entry]) -> Done<Gguf> {
    let mut headers = Vec::new();
    for e in parts {
        let mut n = 4u64 << 20;
        let g = loop {
            let bytes = hub.prefix(remote, e, n).map_err(Refusal::from)?;
            match Gguf::parse_header(&bytes, e.size) {
                Ok(g) => break g,
                // (a header the bytes do not reach the end of fails where they end)
                Err(_) if (bytes.len() as u64) < e.size && n < 1 << 30 => n *= 4,
                Err(err) => return Err(Refusal::new(format!("{}: {err}", e.path), "")),
            }
        };
        headers.push((e.path.clone(), g));
    }
    Ok(merge(headers)?)
}

/// MODEL as the engine's commands take it, made a Hugging Face directory where it is a GGUF: a `.gguf` file (or a part of a split), a directory that holds `.gguf` files and no config.json, or a Hub
/// repo (in the local cache, else on the Hub) that has `.gguf` files and no safetensors. Some(the directory, with MODEL's `:level` kept), or None where MODEL is not one of these. `plan`: for a
/// planning command (`plan_dir`): the directory has the config alone, and a repo that is not downloaded is read by its files' headers, not downloaded.
pub fn resolve(model: &str, log: &dyn Fn(&str), plan: bool) -> Done<Option<String>> {
    let (path, level) = split_level(model);
    let cache = crate::store::cache_root();
    let made = |p: &Path| -> Done<Option<String>> {
        let dir = if plan {
            let f = File::open(p)?;
            plan_dir(&f.gguf, &f.path.file_name().map_or_else(|| "model".into(), |n| n.to_string_lossy().into_owned()), &cache.join("gguf-plan"))?
        } else {
            materialize(p, &cache.join("gguf"), log)?
        };
        Ok(Some(format!("{}{level}", dir.display())))
    };
    let p = Path::new(path);
    if p.is_file() && path.to_ascii_lowercase().ends_with(".gguf") {
        return made(p);
    }
    if p.is_dir() {
        if p.join("config.json").exists() {
            return Ok(None);
        }
        let names = ggufs_in(p);
        if names.is_empty() {
            return Ok(None);
        }
        let pick = choose(&names.iter().map(String::as_str).collect::<Vec<_>>())?;
        return made(&p.join(pick));
    }
    if !crate::store::looks_like_repo(path) {
        return Ok(None);
    }
    // a repo in the Hub cache (a GGUF-only one has no config.json in its snapshot)
    if let Ok(s) = glyd_gpu::save::Source::find(path) {
        if s.dir.join("config.json").exists() {
            return Ok(None);
        }
        let names = ggufs_in(&s.dir);
        if names.is_empty() {
            return Ok(None);
        }
        let pick = choose(&names.iter().map(String::as_str).collect::<Vec<_>>())?;
        return made(&s.dir.join(pick));
    }
    if hub::offline() {
        return Ok(None);
    }
    // not downloaded: the Hub's listing says whether it is GGUF files (a Hub that does not answer is the usual path's to report)
    let hub = Hub::from_env();
    let Ok(remote) = hub.list(path, "main") else { return Ok(None) };
    let has = |ext: &str| remote.files.iter().any(|e| e.path.ends_with(ext));
    if has(".safetensors") || !has(".gguf") {
        return Ok(None);
    }
    let names: Vec<&str> = remote.files.iter().map(|e| e.path.as_str()).collect();
    let pick = choose(&names)?;
    let wanted: Vec<hub::Entry> = parts_of(&names, pick)?.iter().filter_map(|n| remote.files.iter().find(|e| e.path == *n).cloned()).collect();
    if plan {
        let g = hub_headers(&hub, &remote, &wanted)?;
        let label = format!("{path}/{}", base(pick));
        let dir = plan_dir(&g, &label, &cache.join("gguf-plan"))?;
        return Ok(Some(format!("{}{level}", dir.display())));
    }
    log(&format!(
        "glyd: {path} is GGUF files: downloading {pick}{} ({:.1} GB)",
        if wanted.len() > 1 { format!(" and its {} parts", wanted.len() - 1) } else { String::new() },
        wanted.iter().map(|e| e.size).sum::<u64>() as f64 / 1e9
    ));
    let last = std::cell::Cell::new(std::time::Instant::now());
    let snap = hub
        .fetch(&remote, &wanted, &|done, total| {
            if last.get().elapsed().as_secs() >= 5 {
                last.set(std::time::Instant::now());
                log(&format!("glyd: {pick}: {:.1} of {:.1} GB", done as f64 / 1e9, total as f64 / 1e9));
            }
        })
        .map_err(Refusal::from)?;
    made(&snap.join(pick))
}

/// `glyd-engine run|serve|prepare|verify|fit|supports|preflight|costs MODEL` with MODEL a GGUF (see `resolve`): MODEL replaced by its directory (a planning command's has the config alone). Some(exit
/// status) where it is refused, the reason said.
pub fn resolve_args(a: &mut Args) -> Option<i32> {
    let plan = match a.pos.first().map(String::as_str) {
        Some("run" | "serve" | "prepare" | "verify") => false,
        Some("fit" | "supports" | "preflight" | "costs") => true,
        _ => return None,
    };
    let model = a.pos.get(1)?.clone();
    match resolve(&model, &|s| eprintln!("{s}"), plan) {
        Ok(Some(dir)) => {
            a.pos[1] = dir;
            None
        }
        Ok(None) => None,
        Err(r) => {
            Ui::new().refuse(&r);
            Some(r.code)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A GGUF file's bytes: `meta` the key-value pairs (already encoded), `tensors` (name, ne, type, data) with their data after the header.
    struct Build {
        version: u32,
        meta: Vec<u8>,
        n_meta: u64,
        infos: Vec<u8>,
        n_tensors: u64,
        data: Vec<u8>,
        alignment: u64,
    }

    fn string(out: &mut Vec<u8>, s: &str) {
        out.extend((s.len() as u64).to_le_bytes());
        out.extend(s.as_bytes());
    }

    impl Build {
        fn new() -> Build {
            Build { version: 3, meta: Vec::new(), n_meta: 0, infos: Vec::new(), n_tensors: 0, data: Vec::new(), alignment: 32 }
        }
        fn kv(&mut self, key: &str, ty: u32, value: Vec<u8>) -> &mut Build {
            string(&mut self.meta, key);
            self.meta.extend(ty.to_le_bytes());
            self.meta.extend(value);
            self.n_meta += 1;
            self
        }
        fn s(&mut self, key: &str, v: &str) -> &mut Build {
            let mut b = Vec::new();
            string(&mut b, v);
            self.kv(key, 8, b)
        }
        fn u32(&mut self, key: &str, v: u32) -> &mut Build {
            self.kv(key, 4, v.to_le_bytes().to_vec())
        }
        fn f32(&mut self, key: &str, v: f32) -> &mut Build {
            self.kv(key, 6, v.to_le_bytes().to_vec())
        }
        fn strings(&mut self, key: &str, v: &[&str]) -> &mut Build {
            let mut b = 8u32.to_le_bytes().to_vec();
            b.extend((v.len() as u64).to_le_bytes());
            for s in v {
                string(&mut b, s);
            }
            self.kv(key, 9, b)
        }
        fn ints(&mut self, key: &str, v: &[i32]) -> &mut Build {
            let mut b = 5u32.to_le_bytes().to_vec();
            b.extend((v.len() as u64).to_le_bytes());
            for x in v {
                b.extend(x.to_le_bytes());
            }
            self.kv(key, 9, b)
        }
        fn tensor(&mut self, name: &str, ne: &[u64], ty: u32, data: &[u8]) -> &mut Build {
            while self.data.len() as u64 % self.alignment != 0 {
                self.data.push(0);
            }
            string(&mut self.infos, name);
            self.infos.extend((ne.len() as u32).to_le_bytes());
            for d in ne {
                self.infos.extend(d.to_le_bytes());
            }
            self.infos.extend(ty.to_le_bytes());
            self.infos.extend((self.data.len() as u64).to_le_bytes());
            self.data.extend(data);
            self.n_tensors += 1;
            self
        }
        fn bytes(&self) -> Vec<u8> {
            let mut b = b"GGUF".to_vec();
            b.extend(self.version.to_le_bytes());
            b.extend(self.n_tensors.to_le_bytes());
            b.extend(self.n_meta.to_le_bytes());
            b.extend(&self.meta);
            b.extend(&self.infos);
            while b.len() as u64 % self.alignment != 0 {
                b.push(0);
            }
            b.extend(&self.data);
            b
        }
    }

    fn bf16s(v: &[f32]) -> Vec<u8> {
        v.iter().flat_map(|x| ((x.to_bits() >> 16) as u16).to_le_bytes()).collect()
    }

    /// A one-layer Qwen3 of hidden 4 (2 heads of 2, one kv head), a vocabulary of the 256 bytes (as the GPT-2 byte-to-unicode map spells them), `<|im_start|>`, `<|im_end|>` and `<think>`.
    fn tiny_qwen3(weights: u32, norm: u32) -> Vec<u8> {
        tiny_parts(weights, norm, 1).remove(0)
    }

    /// `tiny_qwen3` as the `parts` files of a split: the metadata in the first (which holds no tensors where there are more files), the tensors dealt out to the others.
    fn tiny_parts(weights: u32, norm: u32, parts: usize) -> Vec<Vec<u8>> {
        let mut vocab: Vec<String> = Vec::new();
        // the printable ASCII is its own token; the rest are not used here
        for b in 0u8..=255 {
            let c = match b {
                b'!'..=b'~' => b as char,
                _ => char::from_u32(256 + b as u32).unwrap(),
            };
            vocab.push(c.to_string());
        }
        // (the merges' results)
        vocab.extend(["he", "hel"].map(String::from));
        let n_plain = vocab.len();
        vocab.extend(["<|im_start|>", "<|im_end|>", "<think>"].map(String::from));
        let v: Vec<&str> = vocab.iter().map(String::as_str).collect();
        let mut types = vec![1i32; n_plain];
        types.extend([3, 3, 4]);
        let mut b = Build::new();
        let mut tensors: Vec<(String, Vec<u64>, u32, Vec<u8>)> = Vec::new();
        b.s("general.architecture", "qwen3")
            .s("general.name", "Tiny")
            .u32("qwen3.context_length", 128)
            .u32("qwen3.embedding_length", 4)
            .u32("qwen3.block_count", 1)
            .u32("qwen3.feed_forward_length", 8)
            .u32("qwen3.attention.head_count", 2)
            .u32("qwen3.attention.head_count_kv", 1)
            .u32("qwen3.attention.key_length", 2)
            .f32("qwen3.attention.layer_norm_rms_epsilon", 1e-6)
            .f32("qwen3.rope.freq_base", 1_000_000.0)
            .s("tokenizer.ggml.model", "gpt2")
            .s("tokenizer.ggml.pre", "qwen2")
            .strings("tokenizer.ggml.tokens", &v)
            .ints("tokenizer.ggml.token_type", &types)
            .strings("tokenizer.ggml.merges", &["h e", "he l"])
            .u32("tokenizer.ggml.eos_token_id", n_plain as u32 + 1)
            .s("tokenizer.chat_template", "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}");
        let n = vocab.len() as u64;
        // (values that are bf16 values, as a BF16 model's weights are)
        let w = |count: usize, scale: f32| -> Vec<f32> { (0..count).map(|i| f32::from_bits(((i as f32 + 1.0) * scale).to_bits() >> 16 << 16)).collect() };
        let ty_bytes = |ty: u32, x: &[f32]| -> Vec<u8> {
            match ty {
                BF16 => bf16s(x),
                F32 => x.iter().flat_map(|v| v.to_le_bytes()).collect(),
                _ => unreachable!(),
            }
        };
        tensors.push(("token_embd.weight".to_string(), vec![4, n], weights, ty_bytes(weights, &w(4 * n as usize, 0.125))));
        tensors.push(("output_norm.weight".to_string(), vec![4], norm, ty_bytes(norm, &[1.0, 2.0, 0.5, 4.0])));
        tensors.push(("blk.0.attn_norm.weight".to_string(), vec![4], norm, ty_bytes(norm, &[1.0; 4])));
        tensors.push(("blk.0.ffn_norm.weight".to_string(), vec![4], norm, ty_bytes(norm, &[1.0; 4])));
        tensors.push(("blk.0.attn_q_norm.weight".to_string(), vec![2], norm, ty_bytes(norm, &[1.0; 2])));
        tensors.push(("blk.0.attn_k_norm.weight".to_string(), vec![2], norm, ty_bytes(norm, &[1.0; 2])));
        tensors.push(("blk.0.attn_q.weight".to_string(), vec![4, 4], weights, ty_bytes(weights, &w(16, 0.25))));
        tensors.push(("blk.0.attn_k.weight".to_string(), vec![4, 2], weights, ty_bytes(weights, &w(8, 0.25))));
        tensors.push(("blk.0.attn_v.weight".to_string(), vec![4, 2], weights, ty_bytes(weights, &w(8, 0.5))));
        tensors.push(("blk.0.attn_output.weight".to_string(), vec![4, 4], weights, ty_bytes(weights, &w(16, 0.5))));
        tensors.push(("blk.0.ffn_gate.weight".to_string(), vec![4, 8], weights, ty_bytes(weights, &w(32, 0.25))));
        tensors.push(("blk.0.ffn_up.weight".to_string(), vec![4, 8], weights, ty_bytes(weights, &w(32, 0.5))));
        tensors.push(("blk.0.ffn_down.weight".to_string(), vec![8, 4], weights, ty_bytes(weights, &w(32, 0.125))));
        let total = tensors.len();
        let mut out: Vec<Build> = (0..parts).map(|_| Build::new()).collect();
        for (i, o) in out.iter_mut().enumerate() {
            if i == 0 {
                o.meta = b.meta.clone();
                o.n_meta = b.n_meta;
            } else {
                o.s("general.architecture", "qwen3");
            }
            if parts > 1 {
                o.u32("split.no", i as u32).u32("split.count", parts as u32).u32("split.tensors.count", total as u32);
            }
        }
        for (k, (name, ne, ty, data)) in tensors.into_iter().enumerate() {
            let at = if parts == 1 { 0 } else { 1 + k % (parts - 1) };
            out[at].tensor(&name, &ne, ty, &data);
        }
        out.iter().map(Build::bytes).collect()
    }

    fn temp(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("glyd-gguf-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn a_header_is_read_with_its_metadata_and_tensors() {
        let mut b = Build::new();
        b.s("general.architecture", "llama").u32("llama.block_count", 7).f32("x.f", 0.5).ints("x.list", &[-1, 2, 3]).strings("x.names", &["a", "bc"]);
        b.tensor("a.weight", &[64, 3], 1, &[0; 64 * 3 * 2]);
        b.tensor("b.weight", &[32], 2, &[7; 18]);
        let bytes = b.bytes();
        let g = Gguf::parse(&bytes).unwrap();
        assert_eq!((g.version, g.alignment), (3, 32));
        assert_eq!(g.str("general.architecture"), Some("llama"));
        assert_eq!(g.u64("llama.block_count"), Some(7));
        assert_eq!(g.f64("x.f"), Some(0.5));
        assert_eq!(g.get("x.list"), Some(&Value::Array(vec![Value::I(-1), Value::I(2), Value::I(3)])));
        assert_eq!(g.get("x.names").and_then(Value::as_array).map(|a| a.len()), Some(2));
        assert_eq!(g.tensors.len(), 2);
        let a = g.tensor("a.weight").unwrap();
        // (ne is fastest first: 3 rows of 64)
        assert_eq!((a.shape(), a.bytes, a.type_name(), a.offset), (vec![3, 64], Some(384), "F16".to_string(), 0));
        let q = g.tensor("b.weight").unwrap();
        assert_eq!((q.bytes, q.type_name(), q.offset), (Some(18), "Q4_0".to_string(), 384));
        assert_eq!(g.types(), vec![("F16".to_string(), 1), ("Q4_0".to_string(), 1)]);
        assert_eq!(g.data_offset % 32, 0);
        assert_eq!(g.data_offset + 384 + 18, bytes.len() as u64);
    }

    #[test]
    fn a_file_of_metadata_alone_may_end_with_its_header() {
        let mut b = Build::new();
        b.s("general.architecture", "llama").u32("split.no", 0);
        let bytes = b.bytes();
        // (the header ends off the alignment, and no padding follows it)
        let header = 24 + b.meta.len() + b.infos.len();
        assert!(header % 32 != 0 && header < bytes.len());
        let g = Gguf::parse(&bytes[..header]).unwrap();
        assert_eq!((g.tensors.len(), g.data_offset, g.u64("split.no")), (0, header as u64, Some(0)));
        // with a tensor the data must be there
        let mut t = Build::new();
        t.tensor("a", &[2], 0, &[0; 8]);
        let bytes = t.bytes();
        assert!(Gguf::parse(&bytes[..bytes.len() - 8 - 1]).is_err());
    }

    #[test]
    fn an_alignment_the_file_names_is_kept() {
        let mut b = Build::new();
        b.alignment = 64;
        b.u32("general.alignment", 64);
        b.tensor("a", &[2], 0, &[0; 8]).tensor("b", &[2], 0, &[0; 8]);
        let g = Gguf::parse(&b.bytes()).unwrap();
        assert_eq!((g.alignment, g.tensor("b").unwrap().offset, g.data_offset % 64), (64, 64, 0));
    }

    #[test]
    fn a_tensor_of_a_type_not_known_is_listed_and_not_read() {
        let mut b = Build::new();
        b.tensor("a", &[2], 99, &[0; 8]);
        let bytes = b.bytes();
        let g = Gguf::parse(&bytes).unwrap();
        assert_eq!((g.tensors[0].bytes, g.tensors[0].type_name()), (None, "type 99".to_string()));
        assert_eq!(unsupported_types(&g), vec![("type 99".to_string(), 1)]);
    }

    #[test]
    fn what_cannot_be_read_is_an_error() {
        let mut b = Build::new();
        b.s("general.architecture", "llama").strings("t", &["a", "b"]);
        b.tensor("a", &[64], 0, &[0; 256]);
        let good = b.bytes();
        assert!(Gguf::parse(&good).is_ok());
        // every cut of the file inside its header (and the data short) is an error, none a panic
        let data = Gguf::parse(&good).unwrap().data_offset as usize;
        for n in 0..data + 255 {
            assert!(Gguf::parse(&good[..n]).is_err(), "{n}");
        }
        let with = |f: &dyn Fn(&mut Vec<u8>)| {
            let mut v = good.clone();
            f(&mut v);
            Gguf::parse(&v).unwrap_err().to_string()
        };
        assert!(with(&|v| v[0] = b'X').contains("not a GGUF"));
        assert!(with(&|v| v[4..8].copy_from_slice(&1u32.to_le_bytes())).contains("version 1"));
        assert!(with(&|v| v[4..8].copy_from_slice(&3u32.to_be_bytes())).contains("big-endian"));
        assert!(with(&|v| v[4..8].copy_from_slice(&9u32.to_le_bytes())).contains("version 9"));
        // counts past what the bytes could hold
        assert!(with(&|v| v[8..16].copy_from_slice(&u64::MAX.to_le_bytes())).contains("more than the header's bytes"));
        assert!(with(&|v| v[16..24].copy_from_slice(&u64::MAX.to_le_bytes())).contains("more than the header's bytes"));
        // a string, or an array, longer than the file
        let mut c = Build::new();
        c.kv("k", 8, u64::MAX.to_le_bytes().to_vec());
        assert!(Gguf::parse(&c.bytes()).unwrap_err().to_string().contains("string"));
        let mut c = Build::new();
        let mut arr = 4u32.to_le_bytes().to_vec();
        arr.extend(u64::MAX.to_le_bytes());
        c.kv("k", 9, arr);
        assert!(Gguf::parse(&c.bytes()).unwrap_err().to_string().contains("array"));
        // a tensor past the data, one off the alignment, a row that is not whole blocks, two of a name
        let mut c = Build::new();
        c.tensor("a", &[64], 0, &[0; 256]);
        let mut bytes = c.bytes();
        bytes.truncate(bytes.len() - 4);
        assert!(Gguf::parse(&bytes).unwrap_err().to_string().contains("past the file's"));
        let mut c = Build::new();
        c.tensor("a", &[31], 2, &[0; 18]);
        assert!(Gguf::parse(&c.bytes()).unwrap_err().to_string().contains("not a multiple of Q4_0's block of 32"));
        let mut c = Build::new();
        c.tensor("a", &[2], 0, &[0; 8]).tensor("a", &[2], 0, &[0; 8]);
        assert!(Gguf::parse(&c.bytes()).unwrap_err().to_string().contains("twice"));
        let mut c = Build::new();
        c.tensor("a", &[2], 0, &[0; 8]).tensor("b", &[2], 0, &[0; 8]);
        // (the second tensor's offset, 32, made 33: its entry ends with the offset)
        let mut bytes = c.bytes();
        let end = Gguf::parse(&bytes).unwrap().data_offset as usize;
        let at = (0..end - 8).rev().find(|&i| bytes[i..i + 8] == 32u64.to_le_bytes()).unwrap();
        bytes[at] = 33;
        assert!(Gguf::parse(&bytes).unwrap_err().to_string().contains("alignment"));
    }

    #[test]
    fn a_header_alone_is_checked_against_the_files_length() {
        let mut b = Build::new();
        b.tensor("a", &[64], 0, &[0; 256]);
        let bytes = b.bytes();
        let data = Gguf::parse(&bytes).unwrap().data_offset as usize;
        assert_eq!(Gguf::parse_header(&bytes[..data], bytes.len() as u64).unwrap().tensors[0].bytes, Some(256));
        assert!(Gguf::parse_header(&bytes[..data], bytes.len() as u64 - 1).is_err());
    }

    #[test]
    fn every_type_of_the_table_has_a_size_that_divides_its_block() {
        for &(id, name, block, size) in TYPES {
            assert!(block >= 1 && size >= 1, "{name}");
            assert_eq!(type_info(id), Some((name, block, size)));
        }
        assert_eq!(type_info(30), Some(("BF16", 1, 2)));
        assert_eq!(type_info(12), Some(("Q4_K", 256, 144)));
        assert_eq!(type_info(31), None);
    }

    #[test]
    fn half_floats_become_f32() {
        for (h, f) in [
            (0x3c00u16, 1.0f32),
            (0xc000, -2.0),
            (0x0000, 0.0),
            (0x7bff, 65504.0),
            (0x0001, 2f32.powi(-24)),
            (0x0400, 2f32.powi(-14)),
            (0x03ff, 1023.0 * 2f32.powi(-24)),
        ] {
            assert_eq!(f16_to_f32(h), f, "{h:#x}");
        }
        assert!(f16_to_f32(0x7c00).is_infinite() && f16_to_f32(0x7e00).is_nan());
    }

    #[test]
    fn tensors_are_named_as_hugging_face_names_them() {
        assert_eq!(hf_name("token_embd.weight").as_deref(), Some("model.embed_tokens.weight"));
        assert_eq!(hf_name("blk.12.attn_q_norm.weight").as_deref(), Some("model.layers.12.self_attn.q_norm.weight"));
        assert_eq!(hf_name("blk.3.ffn_down.weight").as_deref(), Some("model.layers.3.mlp.down_proj.weight"));
        assert_eq!(hf_name("blk.3.attn_output.weight").as_deref(), Some("model.layers.3.self_attn.o_proj.weight"));
        assert_eq!(hf_name("rope_freqs.weight"), None);
        assert_eq!(hf_name("blk.x.attn_q.weight"), None);
    }

    /// A Qwen3 GGUF becomes a directory the engine's own readers read: the config, the tokenizer, the chat template and the weights, each as it is in the file.
    #[test]
    fn a_qwen3_gguf_becomes_a_checkpoint_directory() {
        let dir = temp("qwen3");
        let file = dir.join("tiny-BF16.gguf");
        std::fs::write(&file, tiny_qwen3(BF16, F32)).unwrap();
        let g = File::open(&file).unwrap();
        let d = g.gguf.description().unwrap();
        assert_eq!(
            (d.arch.as_str(), d.layers, d.heads, d.kv_heads, d.head_dim, d.ffn, d.context, d.hidden),
            ("qwen3", Some(1), Some(2), Some(1), Some(2), Some(8), Some(128), Some(4))
        );
        drop(g);
        let out = materialize(&file, &dir.join("cache"), &|_| {}).unwrap();
        let c = crate::weights::Config::load(&out).unwrap();
        assert_eq!(
            (c.hidden, c.layers, c.heads, c.kv_heads, c.head_dim, c.inter, c.vocab, c.max_pos, c.tie, c.eos.clone()),
            (4, 1, 2, 1, 2, 8, 261, 128, true, vec![259])
        );
        assert_eq!((c.rope_theta, c.eps), (1_000_000.0, 1e-6));
        // the weights, under Hugging Face's names, in bf16 as the file has them (the norms were F32 holding bf16 values)
        let ck = glyd_gpu::safetensors::Checkpoint::open(&out).unwrap();
        let mut norm = [0u16; 4];
        ck.read_bf16("model.norm.weight", &mut norm).unwrap();
        assert_eq!(norm.map(|b| f32::from_bits((b as u32) << 16)), [1.0, 2.0, 0.5, 4.0]);
        let mut q = [0u16; 16];
        ck.read_bf16("model.layers.0.self_attn.q_proj.weight", &mut q).unwrap();
        assert_eq!(q.map(|b| f32::from_bits((b as u32) << 16)), core::array::from_fn(|i| (i as f32 + 1.0) * 0.25));
        assert_eq!(ck.get("model.embed_tokens.weight").unwrap().1.shape, vec![261, 4]);
        assert!(ck.get("lm_head.weight").is_none());
        // the tokenizer and the chat template
        let tok = crate::tok::Tok::load(&out).unwrap();
        // ("he" and "hel" are the file's merges "h e" and "he l")
        assert_eq!(tok.encode("hel").unwrap(), vec![257]);
        let ids = tok.encode("<|im_start|>hel").unwrap();
        assert_eq!(ids, vec![258, 257]);
        assert_eq!(tok.decode(&ids).unwrap(), "<|im_start|>hel");
        let chat = crate::chat::Chat::load(&out).unwrap().unwrap();
        assert_eq!(chat.render(&[crate::chat::Message::new("user", "hi")], false, None).unwrap(), "<|im_start|>user\nhi<|im_end|>\n");
        // a second call finds the directory
        assert_eq!(materialize(&file, &dir.join("cache"), &|_| {}).unwrap(), out);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn an_f32_weight_that_is_not_a_bf16_value_is_refused_and_nothing_is_left() {
        let dir = temp("f32");
        let file = dir.join("tiny-F32.gguf");
        // (F32 weights: 0.1 has more than 8 bits of significand)
        let mut bytes = tiny_qwen3(F32, F32);
        let g = Gguf::parse(&bytes).unwrap();
        let at = (g.data_offset + g.tensor("blk.0.attn_q.weight").unwrap().offset) as usize;
        bytes[at..at + 4].copy_from_slice(&0.1f32.to_le_bytes());
        std::fs::write(&file, bytes).unwrap();
        let e = materialize(&file, &dir.join("cache"), &|_| {}).unwrap_err().to_string();
        assert!(e.contains("F32 tensor blk.0.attn_q.weight is not exactly bf16"), "{e}");
        assert_eq!(std::fs::read_dir(dir.join("cache")).unwrap().count(), 0);
        // the same file with bf16 values in its F32 weights is read
        std::fs::write(&file, tiny_qwen3(F32, F32)).unwrap();
        assert!(materialize(&file, &dir.join("cache"), &|_| {}).is_ok());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_quantized_gguf_is_refused_naming_its_types() {
        let dir = temp("quant");
        let file = dir.join("tiny-Q8_0.gguf");
        let mut b = Build::new();
        b.s("general.architecture", "qwen3");
        b.tensor("token_embd.weight", &[32, 2], 8, &[0; 68]).tensor("blk.0.attn_norm.weight", &[4], 0, &[0; 16]).tensor(
            "blk.0.attn_q.weight",
            &[256],
            12,
            &[0; 144],
        );
        b.tensor("blk.0.attn_k.weight", &[256], 12, &[0; 144]).tensor("x", &[4], 77, &[0; 4]);
        std::fs::write(&file, b.bytes()).unwrap();
        let e = materialize(&file, &dir.join("cache"), &|_| {}).unwrap_err();
        assert_eq!(e.code, crate::ui::CANNOT);
        let e = e.to_string();
        assert!(e.contains("Q8_0 (1 tensors), Q4_K (2 tensors), type 77 (1 tensors)") && e.contains("no kernels for quantized GGUF types"), "{e}");
        assert!(!dir.join("cache").exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// R3a (#136): a GGUF maps to the same description as a Hugging Face checkpoint. The config.json its metadata makes describes (the description builder of
    /// src/runtime/describe.rs, as a Hub download's does), and the tensors that description reads are the GGUF's own under their Hugging Face names, all of them.
    #[test]
    fn a_gguf_has_the_description_of_its_hugging_face_twin() {
        let g = Gguf::parse(&tiny_qwen3(BF16, F32)).unwrap();
        let config: serde_json::Value = serde_json::from_str(&config_json(&g, &g.description().unwrap()).unwrap()).unwrap();
        let d = crate::runtime::describe::describe(&config).unwrap_or_else(|e| panic!("{}", e.msg));
        let mapped: std::collections::BTreeSet<String> =
            g.tensors.iter().map(|t| hf_name(&t.name).unwrap_or_else(|| panic!("{} has no Hugging Face name", t.name))).collect();
        assert_eq!(d.names(), mapped);
        assert_eq!(d.desc.blocks.len(), 2 * config["num_hidden_layers"].as_u64().unwrap() as usize);
    }

    #[test]
    fn another_architecture_is_described_and_refused_by_name() {
        let mut b = Build::new();
        b.s("general.architecture", "gemma3").u32("gemma3.block_count", 26).tensor("token_embd.weight", &[8, 4], 0, &[0; 128]);
        let g = Gguf::parse(&b.bytes()).unwrap();
        let d = g.description().unwrap();
        assert_eq!((d.arch.as_str(), d.layers), ("gemma3", Some(26)));
        assert!(config_json(&g, &d).unwrap_err().to_string().contains("GGUF architecture gemma3"));
    }

    #[test]
    fn the_file_to_run_is_the_unquantized_one() {
        assert_eq!(choose(&["m-Q4_K_M.gguf", "m-BF16.gguf", "m-F16.gguf", "README.md"]).unwrap(), "m-BF16.gguf");
        assert_eq!(choose(&["m-Q4_K_M.gguf", "m-F16.gguf", "m-F32.gguf"]).unwrap(), "m-F32.gguf");
        assert_eq!(choose(&["only.gguf"]).unwrap(), "only.gguf");
        assert_eq!(choose(&["mmproj-F16.gguf", "m-BF16.gguf"]).unwrap(), "m-BF16.gguf");
        let e = choose(&["m-Q4_K_M.gguf", "m-Q8_0.gguf"]).unwrap_err().to_string();
        assert!(e.contains("m-Q4_K_M.gguf, m-Q8_0.gguf") && e.contains("no kernels for quantized"), "{e}");
        // a split is its first part, and every part must be there
        let split = ["BF16/m-BF16-00002-of-00002.gguf", "BF16/m-BF16-00001-of-00002.gguf", "m-Q4_K_M.gguf"];
        assert_eq!(choose(&split).unwrap(), "BF16/m-BF16-00001-of-00002.gguf");
        assert_eq!(parts_of(&split, "BF16/m-BF16-00001-of-00002.gguf").unwrap(), vec!["BF16/m-BF16-00001-of-00002.gguf", "BF16/m-BF16-00002-of-00002.gguf"]);
        let e = choose(&["BF16/m-BF16-00001-of-00003.gguf", "BF16/m-BF16-00003-of-00003.gguf"]).unwrap_err().to_string();
        assert!(e.contains("part 2 of 3 of the split GGUF") && e.contains("is missing"), "{e}");
        // (a model in one file before the same in parts)
        assert_eq!(choose(&["m-BF16-00001-of-00002.gguf", "m-BF16-00002-of-00002.gguf", "x-BF16.gguf"]).unwrap(), "x-BF16.gguf");
        assert_eq!(split_of("a-b-00002-of-00004.gguf"), Some(("a-b", 2, 4)));
        assert_eq!(split_of("a-00005-of-00004.gguf"), None);
        assert_eq!(split_of("a-of-b.gguf"), None);
        assert_eq!(split_of("model-0001-of-0004.gguf"), None);
        assert_eq!(split_name("a-b", 2, 4), "a-b-00002-of-00004.gguf");
    }

    /// A real GGUF of a Qwen3 against the model's own checkpoint: `GLYD_TEST_GGUF` a BF16 GGUF file, `GLYD_TEST_CKPT` the Hugging Face directory it was made from (`cargo test gguf_against_the_checkpoint
    /// -- --ignored`). Every tensor's bytes are the checkpoint's; the config is the checkpoint's; the tokenizer encodes and decodes as the checkpoint's does; the chat template renders as its.
    #[test]
    #[ignore = "needs GLYD_TEST_GGUF and GLYD_TEST_CKPT"]
    fn gguf_against_the_checkpoint() {
        let (Some(gguf), Some(ckpt)) = (std::env::var_os("GLYD_TEST_GGUF"), std::env::var_os("GLYD_TEST_CKPT")) else {
            panic!("set GLYD_TEST_GGUF and GLYD_TEST_CKPT")
        };
        let (ckpt, root) = (PathBuf::from(ckpt), temp("real"));
        let out = materialize(Path::new(&gguf), &root, &|m| eprintln!("{m}")).unwrap();
        let (a, b) = (glyd_gpu::safetensors::Checkpoint::open(&out).unwrap(), glyd_gpu::safetensors::Checkpoint::open(&ckpt).unwrap());
        let mut names: Vec<&str> = a.names().collect();
        names.sort();
        let mut want: Vec<&str> = b.names().collect();
        // (a tied model's checkpoint may carry lm_head.weight as a copy)
        want.retain(|n| *n != "lm_head.weight" || a.get(n).is_some());
        want.sort();
        assert_eq!(names, want);
        for n in &names {
            let ((_, x), (_, y)) = (a.get(n).unwrap(), b.get(n).unwrap());
            assert_eq!((&x.dtype, &x.shape), (&y.dtype, &y.shape), "{n}");
            assert!(a.read(n).unwrap() == b.read(n).unwrap(), "{n}: the bytes differ");
        }
        eprintln!("{} tensors, every byte the checkpoint's", names.len());
        let (c, d) = (crate::weights::Config::load(&out).unwrap(), crate::weights::Config::load(&ckpt).unwrap());
        let shape = |c: &crate::weights::Config| {
            format!("{:?}", (c.hidden, c.layers, c.heads, c.kv_heads, c.head_dim, c.inter, c.vocab, c.eps, c.rope_theta, c.max_pos, c.tie))
        };
        assert_eq!(shape(&c), shape(&d));
        let (t, u) = (crate::tok::Tok::load(&out).unwrap(), crate::tok::Tok::load(&ckpt).unwrap());
        let texts = [
            "Hello, world!",
            "  leading and trailing  ",
            "The quick brown fox jumps over the lazy dog. It's 3.14159 and 1,000,000!",
            "fn main() {\n    println!(\"hi\");\n}\n\n\n",
            "日本語のテキスト、中文,한국어 and emoji 🎉🚀",
            "e\u{301} vs é, tabs\t\tand\r\nCRLF",
            "<|im_start|>user\nhi<|im_end|>\n<think>\n\n</think>",
            "1234567890 ab12cd34",
        ];
        for text in texts {
            let ids = u.encode(text).unwrap();
            assert_eq!(t.encode(text).unwrap(), ids, "{text:?}");
            assert_eq!(t.decode(&ids).unwrap(), u.decode(&ids).unwrap(), "{text:?}");
        }
        assert_eq!(t.added_ids(), u.added_ids());
        let (x, y) = (crate::chat::Chat::load(&out).unwrap().unwrap(), crate::chat::Chat::load(&ckpt).unwrap().unwrap());
        let conv = [
            crate::chat::Message::new("system", "Be brief."),
            crate::chat::Message::new("user", "Hi"),
            crate::chat::Message::new("assistant", "Hello."),
            crate::chat::Message::new("user", "Bye"),
        ];
        for think in [None, Some(false)] {
            assert_eq!(x.render(&conv, true, think).unwrap(), y.render(&conv, true, think).unwrap());
        }
        let _ = std::fs::remove_dir_all(&root);
    }

    /// The headers of real GGUF files (`GLYD_TEST_GGUF_HEADERS`: a directory of `NAME.head`, the file's first bytes, with `NAME.size`, its length): each parses and is described.
    #[test]
    #[ignore = "needs GLYD_TEST_GGUF_HEADERS"]
    fn real_headers_parse_and_are_described() {
        let dir = PathBuf::from(std::env::var_os("GLYD_TEST_GGUF_HEADERS").expect("set GLYD_TEST_GGUF_HEADERS"));
        let mut heads: Vec<PathBuf> =
            std::fs::read_dir(&dir).unwrap().map(|e| e.unwrap().path()).filter(|p| p.extension().is_some_and(|e| e == "head")).collect();
        heads.sort();
        assert!(!heads.is_empty());
        for p in &heads {
            let len: u64 = std::fs::read_to_string(p.with_extension("size")).unwrap().trim().parse().unwrap();
            let g = Gguf::parse_header(&std::fs::read(p).unwrap(), len).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
            let d = g.description().unwrap_or_else(|e| panic!("{}: {e}", p.display()));
            let types: Vec<String> = g.types().iter().map(|(t, n)| format!("{t}x{n}")).collect();
            eprintln!(
                "{:<70} v{} {:<10} layers {:?} hidden {:?} heads {:?}/{:?} experts {:?} vocab {:?} {:?}/{:?} tensors {} [{}]",
                p.file_stem().unwrap().to_string_lossy().chars().take(70).collect::<String>(),
                g.version,
                d.arch,
                d.layers,
                d.hidden,
                d.heads,
                d.kv_heads,
                d.experts,
                d.vocab,
                d.tokenizer,
                d.tokenizer_pre,
                g.tensors.len(),
                types.join(" ")
            );
            assert!(d.layers.is_some() && d.hidden.is_some() && d.context.is_some(), "{}", p.display());
        }
    }

    #[test]
    fn hostile_metadata_and_tables_are_errors_or_nothing() {
        // a head count of 0 with no key length: no head dimension, not a division by zero
        let mut b = Build::new();
        b.s("general.architecture", "llama").u32("llama.embedding_length", 64).u32("llama.attention.head_count", 0);
        assert_eq!(Gguf::parse(&b.bytes()).unwrap().description().unwrap().head_dim, None);
        // many tensors of the same name: refused at the second, in time linear in their number
        let mut c = Build::new();
        for _ in 0..20_000 {
            c.tensor("same", &[1], 0, &[0; 4]);
        }
        let t = std::time::Instant::now();
        assert!(Gguf::parse(&c.bytes()).unwrap_err().to_string().contains("twice"));
        assert!(t.elapsed().as_secs() < 5);
        let mut d = Build::new();
        for i in 0..20_000 {
            d.tensor(&format!("t{i}"), &[1], 0, &[0; 4]);
        }
        let t = std::time::Instant::now();
        assert_eq!(Gguf::parse(&d.bytes()).unwrap().tensors.len(), 20_000);
        assert!(t.elapsed().as_secs() < 5);
    }

    /// The directory made from a file is made again where the file changed under its header (same length and key, other weights).
    #[test]
    fn a_file_changed_in_place_is_read_again() {
        let dir = temp("again");
        let file = dir.join("tiny-BF16.gguf");
        let bytes = tiny_qwen3(BF16, F32);
        std::fs::write(&file, &bytes).unwrap();
        let first = materialize(&file, &dir.join("cache"), &|_| {}).unwrap();
        let weights = |d: &Path| glyd_gpu::safetensors::Checkpoint::open(d).unwrap().read("model.layers.0.self_attn.q_proj.weight").unwrap();
        let before = weights(&first);
        let g = Gguf::parse(&bytes).unwrap();
        let mut changed = bytes.clone();
        let at = (g.data_offset + g.tensor("blk.0.attn_q.weight").unwrap().offset) as usize;
        changed[at..at + 2].copy_from_slice(&[0x80, 0x3f]);
        std::fs::write(&file, &changed).unwrap();
        // (a later modification time)
        let t = std::fs::File::options().write(true).open(&file).unwrap();
        t.set_modified(std::time::SystemTime::now() + std::time::Duration::from_secs(5)).unwrap();
        let second = materialize(&file, &dir.join("cache"), &|_| {}).unwrap();
        assert_eq!(second, first);
        assert_ne!(weights(&second), before);
        assert_eq!(&weights(&second)[..2], &[0x80, 0x3f]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A file with bytes of its header changed at random is read or refused, whatever it holds: never a panic, and what is read is checked against the bytes there are.
    #[test]
    fn a_header_with_random_bytes_changed_never_panics() {
        let good = tiny_qwen3(BF16, F32);
        let header = Gguf::parse(&good).unwrap().data_offset as usize;
        let mut x = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = move || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        for _ in 0..20_000 {
            let mut v = good.clone();
            for _ in 0..1 + next() % 4 {
                let at = (next() % header as u64) as usize;
                v[at] = next() as u8;
            }
            if let Ok(g) = Gguf::parse(&v) {
                let _ = g.description();
                for t in &g.tensors {
                    assert!(t.bytes.is_none_or(|b| g.data_offset + t.offset + b <= v.len() as u64));
                }
            }
        }
    }

    /// A model in several files is one model: the metadata of the first, the tensors of all, in whatever order the parts are named.
    #[test]
    fn a_split_gguf_becomes_the_same_checkpoint_as_one_file() {
        let dir = temp("split");
        let whole = dir.join("one").join("tiny-BF16.gguf");
        std::fs::create_dir_all(whole.parent().unwrap()).unwrap();
        std::fs::write(&whole, tiny_qwen3(BF16, F32)).unwrap();
        let parts = tiny_parts(BF16, F32, 3);
        assert_eq!(parts.len(), 3);
        // (the first part has the metadata and no tensors)
        assert_eq!(Gguf::parse(&parts[0]).unwrap().tensors.len(), 0);
        let split_dir = dir.join("three");
        std::fs::create_dir_all(&split_dir).unwrap();
        for (i, b) in parts.iter().enumerate() {
            std::fs::write(split_dir.join(split_name("tiny-BF16", i as u32 + 1, 3)), b).unwrap();
        }
        let f = File::open(&split_dir.join(split_name("tiny-BF16", 2, 3))).unwrap();
        assert_eq!((f.gguf.tensors.len(), f.gguf.str("general.name")), (13, Some("Tiny")));
        assert!(f.gguf.tensors.iter().any(|t| t.part == 2) && f.gguf.tensors.iter().any(|t| t.part == 1));
        let cache = dir.join("cache");
        let (one, three) =
            (materialize(&whole, &cache, &|_| {}).unwrap(), materialize(&split_dir.join(split_name("tiny-BF16", 1, 3)), &cache, &|_| {}).unwrap());
        assert!(three.file_name().unwrap().to_string_lossy().starts_with("tiny-BF16-"));
        for name in ["model.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json"] {
            assert!(std::fs::read(one.join(name)).unwrap() == std::fs::read(three.join(name)).unwrap(), "{name}");
        }
        // a part missing, a part that says it is another, a count that is not the parts'
        std::fs::remove_file(split_dir.join(split_name("tiny-BF16", 3, 3))).unwrap();
        let e = File::open(&split_dir.join(split_name("tiny-BF16", 1, 3))).err().unwrap().to_string();
        assert!(e.contains("part 3 of 3 of a split GGUF is not there"), "{e}");
        let bad = dir.join("bad");
        std::fs::create_dir_all(&bad).unwrap();
        let put = |i: u32, bytes: &[u8]| std::fs::write(bad.join(split_name("t", i, 3)), bytes).unwrap();
        put(1, &parts[0]);
        put(2, &parts[2]);
        put(3, &parts[2]);
        let e = File::open(&bad.join(split_name("t", 1, 3))).err().unwrap().to_string();
        assert!(e.contains("part 3 of the split by its metadata (split.no 2), and part 2 by its name"), "{e}");
        put(2, &parts[1]);
        put(3, &parts[1]);
        let e = File::open(&bad.join(split_name("t", 1, 3))).err().unwrap().to_string();
        assert!(e.contains("part 2 of the split by its metadata (split.no 1), and part 3 by its name"), "{e}");
        put(3, &tiny_parts(BF16, F32, 4)[3]);
        let e = File::open(&bad.join(split_name("t", 1, 3))).err().unwrap().to_string();
        assert!(e.contains("split.count is 4"), "{e}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// What a planning command reads of a GGUF is its config and nothing of the data; a quantized one, or another architecture, is refused as a run refuses it.
    #[test]
    fn a_planning_directory_has_the_config_alone() {
        let dir = temp("plan");
        let g = Gguf::parse(&tiny_qwen3(BF16, F32)).unwrap();
        let out = plan_dir(&g, "tiny-BF16.gguf", &dir).unwrap();
        let mut names: Vec<String> = std::fs::read_dir(&out).unwrap().map(|e| e.unwrap().file_name().into_string().unwrap()).collect();
        names.sort();
        assert_eq!(names, ["config.json", "generation_config.json", "glyd-tensors.json"]);
        let c = crate::weights::Config::load(&out).unwrap();
        assert_eq!((c.layers, c.hidden, c.vocab, c.eos.clone()), (1, 4, 261, vec![259]));
        assert_eq!(plan_dir(&g, "tiny-BF16.gguf", &dir).unwrap(), out);
        // the same config as a run's directory has
        let file = dir.join("tiny-BF16.gguf");
        std::fs::write(&file, tiny_qwen3(BF16, F32)).unwrap();
        let run = materialize(&file, &dir.join("run"), &|_| {}).unwrap();
        assert_eq!(std::fs::read(run.join("config.json")).unwrap(), std::fs::read(out.join("config.json")).unwrap());
        // the planner's table of the file's headers is the bytes the run loads (P2b: the tensors of the directory the file becomes, by their Hugging Face names)
        let (planned, loaded) = (crate::sizes::Table::from_dir(&out).unwrap(), crate::sizes::Table::from_dir(&run).unwrap());
        let sum = |t: &crate::sizes::Table, keep: &dyn Fn(&str) -> bool| t.rows.iter().filter(|r| keep(&r.name)).map(|r| r.bytes).sum::<u64>();
        let weights = |n: &str| n.starts_with("model.") || n == "lm_head.weight";
        assert!(sum(&planned, &weights) > 0 && (sum(&planned, &weights) as f64 / sum(&loaded, &weights) as f64 - 1.0).abs() < 0.01);
        // refusals, with nothing made
        let before = std::fs::read_dir(&dir).unwrap().count();
        let mut q = Build::new();
        q.s("general.architecture", "qwen3").tensor("blk.0.attn_q.weight", &[256], 12, &[0; 144]);
        let e = plan_dir(&Gguf::parse(&q.bytes()).unwrap(), "q.gguf", &dir).unwrap_err();
        assert!(e.code == crate::ui::CANNOT && e.to_string().contains("Q4_K (1 tensors)"), "{e}");
        assert_eq!(std::fs::read_dir(&dir).unwrap().count(), before);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The architectures of the recent families (the keys and tensors their GGUFs have, as the files' headers show) are described, and refused with what llama.cpp's converter did to them.
    #[test]
    fn recent_architectures_are_described_and_refused_with_what_the_converter_did() {
        let dir = temp("recent");
        let cases: [(&str, &str, &str); 11] = [
            ("qwen35", "Qwen3.5, 3.6 and 3.8", "A_log as -exp(A_log)"),
            ("qwen35moe", "Qwen3.5, 3.6 and 3.8", "norm weights stored as w+1"),
            ("qwen4exp", "Qwen3.8 Flash-Next", "norm gammas stored as w+1"),
            ("gemma4", "Gemma", "norm weights stored as w+1"),
            ("gemma4-assistant", "Gemma", "norm weights stored as w+1"),
            ("muse-spark", "Muse Spark", "output multiplier folded"),
            ("kimi-k3", "Kimi-K3", "experts stacked"),
            ("kimi-k31", "Kimi-K3", "experts stacked"),
            ("deepseek41", "DeepSeek-V4", "engram gates"),
            ("glm5-next", "GLM", "kv_b split"),
            ("glm-dsa", "GLM", "kv_b split"),
        ];
        for (arch, family, what) in cases {
            let mut b = Build::new();
            b.s("general.architecture", arch)
                .u32(&format!("{arch}.block_count"), 65)
                .u32(&format!("{arch}.context_length"), 262144)
                .u32(&format!("{arch}.embedding_length"), 5120)
                .u32(&format!("{arch}.attention.head_count"), 24)
                .u32(&format!("{arch}.attention.head_count_kv"), 4)
                .u32(&format!("{arch}.expert_count"), 128)
                .s("tokenizer.ggml.model", "gpt2")
                .s("tokenizer.ggml.pre", "qwen35")
                .tensor("token_embd.weight", &[4, 3], BF16, &[0; 24])
                .tensor("blk.0.ssm_a", &[4], F32, &[0; 16]);
            let g = Gguf::parse(&b.bytes()).unwrap();
            let d = g.description().unwrap();
            assert_eq!(
                (d.arch.as_str(), d.layers, d.hidden, d.heads, d.kv_heads, d.experts, d.context),
                (arch, Some(65), Some(5120), Some(24), Some(4), Some(128), Some(262144))
            );
            let e = plan_dir(&g, "m.gguf", &dir).unwrap_err();
            let text = e.to_string();
            assert!(
                e.code == crate::ui::CANNOT
                    && text.contains(&format!("GGUF architecture {arch} ({family})"))
                    && text.contains(what)
                    && text.contains("not the release's checkpoint"),
                "{arch}: {text}"
            );
            let file = dir.join(format!("{arch}.gguf"));
            std::fs::write(&file, b.bytes()).unwrap();
            assert_eq!(materialize(&file, &dir.join("run"), &|_| {}).unwrap_err().to_string(), text);
        }
        assert!(!dir.join("run").exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The tokenizers of real GGUFs against the models' own (`GLYD_TEST_TOKENIZERS`: a directory of `NAME.head` and `NAME.size` as for `real_headers_parse_and_are_described`, with the model's
    /// `NAME.tokenizer.json`): the same token ids and the same text back, for a battery of texts, with the special tokens.
    #[test]
    #[ignore = "needs GLYD_TEST_TOKENIZERS"]
    fn tokenizers_against_the_models_own() {
        let dir = PathBuf::from(std::env::var_os("GLYD_TEST_TOKENIZERS").expect("set GLYD_TEST_TOKENIZERS"));
        let only = std::env::var("GLYD_TEST_ONLY").ok();
        let mut heads: Vec<PathBuf> =
            std::fs::read_dir(&dir).unwrap().map(|e| e.unwrap().path()).filter(|p| p.extension().is_some_and(|e| e == "head")).collect();
        heads.sort();
        let texts = [
            "Hello, world!",
            "  leading and trailing  ",
            " Hello",
            "The quick brown fox jumps over the lazy dog. It's 3.14159 and 1,000,000!",
            "fn main() {\n    println!(\"hi\");\n}\n\n\n",
            "def f(x):\n\treturn x**2  # square\n",
            "日本語のテキスト、中文,한국어 and emoji 🎉🚀 and Ünïcödé",
            "e\u{301} vs é, tabs\t\tand\r\nCRLF",
            "1234567890 ab12cd34 0.5e-3",
            "<s>[INST] hi [/INST]</s> <|im_start|>user\nhi<|im_end|>\n<think>\n\n</think> <start_of_turn>user",
            "a  b   c    d",
            "x\n\n\ny \n z",
            "",
            "\u{a0}non-breaking\u{200b}zero",
            "I'M YOU'RE we've THEY'LL",
        ];
        let mut bad: Vec<String> = Vec::new();
        for p in &heads {
            let name = p.file_stem().unwrap().to_string_lossy().into_owned();
            if only.as_ref().is_some_and(|o| *o != name) {
                continue;
            }
            // (TinyLlama's GGUF has every score 0: its merges' order is not in the file)
            if name == "tinyllama" {
                continue;
            }
            let len: u64 = std::fs::read_to_string(p.with_extension("size")).unwrap().trim().parse().unwrap();
            let g = Gguf::parse_header(&std::fs::read(p).unwrap(), len).unwrap();
            let out = temp(&format!("tok-{name}"));
            let (tj, tc) = crate::gguf_tok::files(&g).unwrap_or_else(|e| panic!("{name}: {e}"));
            std::fs::write(out.join("tokenizer.json"), tj).unwrap();
            std::fs::write(out.join("tokenizer_config.json"), tc).unwrap();
            let theirs = dir.join(format!("{name}.tokenizer.json"));
            let ref_dir = temp(&format!("ref-{name}"));
            std::fs::copy(&theirs, ref_dir.join("tokenizer.json")).unwrap();
            let (mine, want) = (crate::tok::Tok::load(&out).unwrap(), crate::tok::Tok::load(&ref_dir).unwrap());
            let mut wrong = 0;
            let bos = g.u64("tokenizer.ggml.bos_token_id").map(|b| b as u32);
            for text in texts {
                // (what a GGUF does not say: the added tokens' lstrip and rstrip; whether a
                // tokenizer that does not add the beginning-of-sequence token in its tokenizer.json has the file's say-so)
                if name == "phi4" && text.contains("<|im_start|>") {
                    continue;
                }
                for special in [true, false] {
                    let (mut a, mut b) = (mine.encode_with(text, special).unwrap(), want.encode_with(text, special).unwrap());
                    for v in [&mut a, &mut b] {
                        if special && v.first().copied() == bos && bos.is_some() {
                            v.remove(0);
                        }
                    }
                    if a != b {
                        wrong += 1;
                        if std::env::var_os("GLYD_TEST_VERBOSE").is_some() {
                            eprintln!("  {name}: {text:?} special={special}\n     mine {a:?}\n     want {b:?}");
                        }
                        if wrong <= 3 {
                            bad.push(format!("{name}: {text:?} special={special}\n   mine {a:?}\n   want {b:?}"));
                        }
                    } else if mine.decode(&a).unwrap() != want.decode(&b).unwrap() {
                        wrong += 1;
                        bad.push(format!("{name}: {text:?} decodes {:?}, wanted {:?}", mine.decode(&a).unwrap(), want.decode(&b).unwrap()));
                    }
                }
            }
            eprintln!("{name}: {} texts x 2, {wrong} different", texts.len());
            let _ = std::fs::remove_dir_all(&out);
            let _ = std::fs::remove_dir_all(&ref_dir);
        }
        assert!(bad.is_empty(), "\n{}", bad.join("\n"));
    }

    fn tokenizer_of(build: &Build) -> (crate::tok::Tok, PathBuf) {
        let g = Gguf::parse(&build.bytes()).unwrap();
        static N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let dir = temp(&format!("toks-{}", N.fetch_add(1, std::sync::atomic::Ordering::Relaxed)));
        let (tj, tc) = crate::gguf_tok::files(&g).unwrap_or_else(|e| panic!("{e}"));
        std::fs::write(dir.join("tokenizer.json"), tj).unwrap();
        std::fs::write(dir.join("tokenizer_config.json"), tc).unwrap();
        (crate::tok::Tok::load(&dir).unwrap(), dir)
    }

    /// SentencePiece merges come out as transformers' converter makes them: by the merged piece's score, the pieces' lengths, the ties in the order of the pieces' ids.
    #[test]
    fn sentencepiece_merges_are_derived_from_the_scores() {
        let pieces = ["<unk>", "<s>", "</s>", "▁", "a", "b", "▁a", "ab", "▁ab"];
        let scores = [0.0, 0.0, 0.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0];
        let m = |l: &str, r: &str| (l.to_string(), r.to_string());
        // ("▁ab" splits two ways, at one score: the longer left piece first)
        assert_eq!(crate::gguf_tok::spm_merges(&pieces, &scores), vec![m("▁", "a"), m("a", "b"), m("▁a", "b"), m("▁", "ab")]);
        // a higher score first, whatever the ids
        let scores = [0.0, 0.0, 0.0, -1.0, -2.0, -3.0, -6.0, -5.0, -4.0];
        assert_eq!(crate::gguf_tok::spm_merges(&pieces, &scores), vec![m("▁a", "b"), m("▁", "ab"), m("a", "b"), m("▁", "a")]);
        // scores that were not kept (all 0): the order of the pieces' ids
        assert_eq!(crate::gguf_tok::spm_merges(&pieces, &[0.0; 9]), vec![m("▁", "a"), m("a", "b"), m("▁", "ab"), m("▁a", "b")]);
    }

    /// A SentencePiece GGUF becomes a tokenizer that puts a ▁ before the text, falls back to bytes, adds the beginning-of-sequence token and reads the text back.
    #[test]
    fn a_sentencepiece_gguf_becomes_a_tokenizer() {
        let mut pieces: Vec<String> = ["<unk>", "<s>", "</s>", "▁", "a", "b", "▁a", "ab", "▁ab"].map(String::from).to_vec();
        pieces.extend((0..=255).map(|b| format!("<0x{b:02X}>")));
        let v: Vec<&str> = pieces.iter().map(String::as_str).collect();
        let mut scores = vec![0.0f32, 0.0, 0.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0];
        scores.extend(std::iter::repeat_n(0.0, 256));
        let mut types = vec![2i32, 3, 3, 1, 1, 1, 1, 1, 1];
        types.extend(std::iter::repeat_n(6, 256));
        let mut b = Build::new();
        b.s("general.architecture", "llama").s("tokenizer.ggml.model", "llama").strings("tokenizer.ggml.tokens", &v).ints("tokenizer.ggml.token_type", &types);
        let mut sc = 6u32.to_le_bytes().to_vec();
        sc.extend((scores.len() as u64).to_le_bytes());
        for x in &scores {
            sc.extend(x.to_le_bytes());
        }
        b.kv("tokenizer.ggml.scores", 9, sc)
            .u32("tokenizer.ggml.bos_token_id", 1)
            .u32("tokenizer.ggml.eos_token_id", 2)
            .u32("tokenizer.ggml.unknown_token_id", 0);
        let (tok, dir) = tokenizer_of(&b);
        assert_eq!(tok.encode("ab ab").unwrap(), vec![1, 8, 8]);
        assert_eq!(tok.encode_with("ab ab", false).unwrap(), vec![8, 8]);
        // (a byte no piece has is its byte's piece: "é" is the bytes 0xC3 0xA9)
        assert_eq!(tok.encode_with("a\u{e9}", false).unwrap(), vec![6, 9 + 0xC3, 9 + 0xA9]);
        assert_eq!(tok.decode(&tok.encode_with("ab \u{e9}a", false).unwrap()).unwrap(), "ab \u{e9}a");
        // (the text after a special token takes no ▁ of its own)
        assert_eq!(tok.encode_with("</s>ab", true).unwrap(), vec![1, 2, 7]);
        let cfg = std::fs::read_to_string(dir.join("tokenizer_config.json")).unwrap();
        assert!(cfg.contains("\"bos_token\": \"<s>\"") && cfg.contains("\"unk_token\": \"<unk>\""), "{cfg}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Every pre-tokenizer the table names makes a tokenizer the engine loads (the expressions compile), and one it does not have is refused by name, as is a kind of tokenizer.
    #[test]
    fn the_pre_tokenizers_load_and_the_others_are_refused_by_name() {
        for (names, _, _, _) in crate::gguf_pre::PRE {
            for name in *names {
                let mut b = Build::new();
                b.s("general.architecture", "llama")
                    .s("tokenizer.ggml.model", "gpt2")
                    .s("tokenizer.ggml.pre", name)
                    .strings("tokenizer.ggml.tokens", &["a", "b", "ab"])
                    .strings("tokenizer.ggml.merges", &["a b"]);
                let (tok, dir) = tokenizer_of(&b);
                assert_eq!(tok.encode_with("ab", false).unwrap(), vec![2], "{name}");
                let _ = std::fs::remove_dir_all(&dir);
            }
        }
        let refused = |model: &str, pre: &str| {
            let mut b = Build::new();
            b.s("general.architecture", "llama")
                .s("tokenizer.ggml.model", model)
                .strings("tokenizer.ggml.tokens", &["a"])
                .strings("tokenizer.ggml.merges", &[]);
            if !pre.is_empty() {
                b.s("tokenizer.ggml.pre", pre);
            }
            crate::gguf_tok::files(&Gguf::parse(&b.bytes()).unwrap()).unwrap_err()
        };
        let e = refused("gpt2", "kimi-k2");
        assert!(e.code == crate::ui::CANNOT && e.to_string().contains("pre-tokenizer \"kimi-k2\""), "{e}");
        assert!(refused("bert", "").to_string().contains("tokenizer is \"bert\""));
        assert!(refused("llama", "").to_string().contains("no scores"));
    }

    #[test]
    fn a_level_after_the_model_is_kept_apart() {
        assert_eq!(split_level("m.gguf:kestrel"), ("m.gguf", ":kestrel"));
        assert_eq!(split_level("org/name"), ("org/name", ""));
        assert_eq!(split_level("/a:b/m.gguf"), ("/a:b/m.gguf", ""));
    }

    #[test]
    fn a_directory_or_name_that_is_not_a_gguf_is_left_alone() {
        let dir = temp("plain");
        std::fs::write(dir.join("config.json"), "{}").unwrap();
        std::fs::write(dir.join("m.gguf"), "x").unwrap();
        // (a checkpoint directory with a GGUF beside its safetensors is the checkpoint)
        assert_eq!(resolve(dir.to_str().unwrap(), &|_| {}, false).unwrap(), None);
        assert_eq!(resolve("not/a/repo/at/all", &|_| {}, false).unwrap(), None);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
