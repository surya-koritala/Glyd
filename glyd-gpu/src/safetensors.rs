//! safetensors files: an 8-byte little-endian length, a JSON header naming
//! each tensor's dtype, shape and byte range in the data after it; and a
//! checkpoint's directory of them (model.safetensors, or shards named by
//! model.safetensors.index.json). Every number taken from a file is checked
//! before it is used.

use crate::json::{self, Value};
use std::collections::HashMap;
use std::io::{self, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

/// Headers past this are refused (safetensors' own limit).
const HEADER_LIMIT: u64 = 100_000_000;

fn bad(path: &Path, why: impl std::fmt::Display) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, format!("{}: {why}", path.display()))
}

/// A dtype's bytes an element (the whole-byte ones), else None.
pub fn dtype_bytes(dtype: &str) -> Option<u64> {
    Some(match dtype {
        "BOOL" | "U8" | "I8" | "F8_E5M2" | "F8_E4M3" | "F8_E8M0" => 1,
        "I16" | "U16" | "F16" | "BF16" => 2,
        "I32" | "U32" | "F32" => 4,
        "I64" | "U64" | "F64" => 8,
        _ => return None,
    })
}

/// A tensor of a file: its dtype, shape, and bytes [offset, offset + bytes) of the file.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorInfo {
    pub dtype: String,
    pub shape: Vec<u64>,
    pub offset: u64,
    pub bytes: u64,
}

/// A safetensors file's header, its tensors in the header's order.
#[derive(Clone, Debug)]
pub struct File {
    pub path: PathBuf,
    pub tensors: Vec<(String, TensorInfo)>,
    pub metadata: Vec<(String, String)>,
}

impl File {
    /// The header of the file at `path`, checked: within the file (and 100
    /// MB); each tensor's dtype whole bytes, its data_offsets in order and
    /// within the data, their bytes its dtype by its shape.
    pub fn open(path: &Path) -> io::Result<File> {
        let mut f = std::fs::File::open(path)?;
        let size = f.metadata()?.len();
        let mut n8 = [0u8; 8];
        f.read_exact(&mut n8).map_err(|_| bad(path, "not a safetensors file"))?;
        let n = u64::from_le_bytes(n8);
        if n > HEADER_LIMIT || n > size - 8 {
            return Err(bad(path, format!("a header of {n} bytes, past the file's {size} (or 100 MB)")));
        }
        let mut h = vec![0u8; n as usize];
        f.read_exact(&mut h)?;
        let h = std::str::from_utf8(&h).map_err(|_| bad(path, "a header not UTF-8"))?;
        let header = json::parse(h).map_err(|e| bad(path, e))?;
        let data = size - 8 - n;
        let mut out = File { path: path.to_path_buf(), tensors: Vec::new(), metadata: Vec::new() };
        for (name, t) in header.as_object().ok_or_else(|| bad(path, "a header not an object"))? {
            if name == "__metadata__" {
                for (k, v) in t.as_object().unwrap_or(&[]) {
                    out.metadata.push((k.clone(), v.as_str().unwrap_or_default().to_string()));
                }
                continue;
            }
            let what = |why: &str| bad(path, format!("{name}: {why}"));
            let dtype = t.get("dtype").and_then(Value::as_str).ok_or_else(|| what("no dtype"))?;
            let shape = t.get("shape").and_then(Value::as_u64s).ok_or_else(|| what("no shape"))?;
            let offs = t.get("data_offsets").and_then(Value::as_u64s).filter(|o| o.len() == 2).ok_or_else(|| what("no data_offsets"))?;
            let width = dtype_bytes(dtype).ok_or_else(|| what(&format!("dtype {dtype}")))?;
            let bytes = shape.iter().try_fold(width, |a, &d| a.checked_mul(d)).ok_or_else(|| what("a shape past 2^64 bytes"))?;
            if offs[0] > offs[1] || offs[1] > data {
                return Err(what(&format!("data_offsets {offs:?} out of order or past the data's {data} bytes")));
            }
            if offs[1] - offs[0] != bytes {
                return Err(what(&format!("{} bytes, where its dtype and shape take {bytes}", offs[1] - offs[0])));
            }
            out.tensors.push((name.clone(), TensorInfo { dtype: dtype.to_string(), shape, offset: 8 + n + offs[0], bytes }));
        }
        Ok(out)
    }

    pub fn get(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.iter().find(|(n, _)| n == name).map(|(_, t)| t)
    }

    /// A tensor's bytes, read from the file.
    pub fn read(&self, t: &TensorInfo) -> io::Result<Vec<u8>> {
        let mut out = vec![0u8; t.bytes as usize];
        self.read_into(t, &mut out)?;
        Ok(out)
    }

    /// A tensor's bytes into `out` (its length).
    pub fn read_into(&self, t: &TensorInfo, out: &mut [u8]) -> io::Result<()> {
        let mut f = std::fs::File::open(&self.path)?;
        f.seek(SeekFrom::Start(t.offset))?;
        f.read_exact(out)
    }
}

/// A checkpoint's safetensors: model.safetensors, or the shards
/// model.safetensors.index.json names (each a file of the directory).
pub struct Checkpoint {
    pub dir: PathBuf,
    pub files: Vec<File>,
    names: HashMap<String, (usize, usize)>,
}

impl Checkpoint {
    pub fn open(dir: &Path) -> io::Result<Checkpoint> {
        let index = dir.join("model.safetensors.index.json");
        let mut shards: Vec<String> = Vec::new();
        if index.exists() {
            let v = json::parse(&std::fs::read_to_string(&index)?).map_err(|e| bad(&index, e))?;
            for (_, f) in v.get("weight_map").and_then(Value::as_object).ok_or_else(|| bad(&index, "no weight_map"))? {
                let f = f.as_str().filter(|f| !f.is_empty() && !f.contains('/') && *f != "..").ok_or_else(|| bad(&index, "a shard not a file of its directory"))?;
                if !shards.iter().any(|s| s == f) {
                    shards.push(f.to_string());
                }
            }
        } else {
            shards.push("model.safetensors".into());
        }
        let mut c = Checkpoint { dir: dir.to_path_buf(), files: Vec::new(), names: HashMap::new() };
        for s in shards {
            let f = File::open(&dir.join(&s))?;
            for (j, (name, _)) in f.tensors.iter().enumerate() {
                c.names.insert(name.clone(), (c.files.len(), j));
            }
            c.files.push(f);
        }
        Ok(c)
    }

    /// Tensor `name`: its file and where it is there.
    pub fn get(&self, name: &str) -> Option<(&File, &TensorInfo)> {
        let &(f, t) = self.names.get(name)?;
        Some((&self.files[f], &self.files[f].tensors[t].1))
    }

    /// Tensor `name`'s bytes.
    pub fn read(&self, name: &str) -> io::Result<Vec<u8>> {
        let (f, t) = self.get(name).ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, format!("{}: no tensor {name}", self.dir.display())))?;
        f.read(t)
    }
}
