//! safetensors files: an 8-byte little-endian length, a JSON header naming
//! each tensor's dtype, shape and byte range in the data after it; and a
//! checkpoint's directory of them (model.safetensors, or shards named by
//! model.safetensors.index.json). Every number taken from a file is checked
//! before it is used, and a file's tensors run back to back from its data's
//! start to its end, as the safetensors library requires to read one. A matrix
//! is read as bf16 words ([`Checkpoint::read_bf16`]): a BF16 tensor as it is, an
//! FP8 (E4M3) one with its block scales dequantised ([`crate::fp8`]).

use crate::fp8;
use crate::json::{self, Value};
use std::collections::HashMap;
use std::io::{self, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

/// Headers past this are refused (safetensors' own limit).
const HEADER_LIMIT: u64 = 100_000_000;

fn bad(path: &Path, why: impl std::fmt::Display) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, format!("{}: {why}", path.display()))
}

/// An io error with the path it came from.
fn at(path: &Path) -> impl FnOnce(io::Error) -> io::Error + '_ {
    move |e| io::Error::new(e.kind(), format!("{}: {e}", path.display()))
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
    /// MB); each tensor's dtype whole bytes, its bytes its dtype by its
    /// shape; the tensors back to back from the data's start to the file's
    /// end (no gap, overlap or byte past the last).
    pub fn open(path: &Path) -> io::Result<File> {
        let mut f = std::fs::File::open(path).map_err(at(path))?;
        let size = f.metadata().map_err(at(path))?.len();
        let mut n8 = [0u8; 8];
        f.read_exact(&mut n8).map_err(|_| bad(path, "not a safetensors file"))?;
        let n = u64::from_le_bytes(n8);
        if n > HEADER_LIMIT || n > size - 8 {
            return Err(bad(path, format!("a header of {n} bytes, past the file's {size} (or 100 MB)")));
        }
        let mut h = vec![0u8; n as usize];
        f.read_exact(&mut h).map_err(at(path))?;
        let h = std::str::from_utf8(&h).map_err(|_| bad(path, "a header not UTF-8"))?;
        let header = json::parse(h).map_err(|e| bad(path, e))?;
        let data = size - 8 - n;
        let mut out = File { path: path.to_path_buf(), tensors: Vec::new(), metadata: Vec::new() };
        let mut spans = Vec::new();
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
            spans.push((offs[0], offs[1], name.clone()));
            out.tensors.push((name.clone(), TensorInfo { dtype: dtype.to_string(), shape, offset: 8 + n + offs[0], bytes }));
        }
        spans.sort();
        let mut pos = 0;
        for (a, b, name) in spans {
            if a != pos {
                return Err(bad(path, format!("{name}'s bytes at {a}, where the tensor before ends at {pos}")));
            }
            pos = b;
        }
        if pos != data {
            return Err(bad(path, format!("{} bytes past its last tensor", data - pos)));
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
        let mut f = std::fs::File::open(&self.path).map_err(at(&self.path))?;
        f.seek(SeekFrom::Start(t.offset)).map_err(at(&self.path))?;
        f.read_exact(out).map_err(at(&self.path))
    }

    /// `read_into`, then its bytes dropped from the page cache: a load that pins most of the computer's memory (Kimi-K3's 1.35 TB of experts) keeps the
    /// cache from holding the rest, which every pinned allocation would otherwise have to reclaim and compact first.
    pub fn read_into_uncached(&self, t: &TensorInfo, out: &mut [u8]) -> io::Result<()> {
        let mut f = std::fs::File::open(&self.path).map_err(at(&self.path))?;
        f.seek(SeekFrom::Start(t.offset)).map_err(at(&self.path))?;
        f.read_exact(out).map_err(at(&self.path))?;
        drop_range(&f, t.offset, out.len() as u64);
        Ok(())
    }

    /// The file's pages dropped from the page cache (the clean ones; see `read_into_uncached`).
    pub fn drop_cached(&self) -> io::Result<()> {
        let f = std::fs::File::open(&self.path).map_err(at(&self.path))?;
        drop_range(&f, 0, 0);
        Ok(())
    }
}

/// Linux's posix_fadvise(POSIX_FADV_DONTNEED) on `len` bytes of `f` from `at` (0: to the end); elsewhere nothing. Advice: a failure changes nothing.
fn drop_range(f: &std::fs::File, at: u64, len: u64) {
    #[cfg(target_os = "linux")]
    {
        use std::os::unix::io::AsRawFd;
        extern "C" {
            fn posix_fadvise(fd: i32, offset: i64, len: i64, advice: i32) -> i32;
        }
        // SAFETY: an open file's descriptor; 4 is POSIX_FADV_DONTNEED.
        unsafe { posix_fadvise(f.as_raw_fd(), at as i64, len as i64, 4) };
    }
    #[cfg(not(target_os = "linux"))]
    let _ = (f, at, len);
}

/// A checkpoint's safetensors: model.safetensors, or the shards
/// model.safetensors.index.json names (each a file of the directory).
pub struct Checkpoint {
    pub dir: PathBuf,
    pub files: Vec<File>,
    /// The index's weight_map, where there are shards: each tensor's name and its shard's.
    pub index: Option<Vec<(String, String)>>,
    /// config.json's `quantization_config.weight_block_size`, where it gives one: the rows and columns of a block of an FP8 matrix's scales.
    pub fp8_block: Option<[usize; 2]>,
    names: HashMap<String, (usize, usize)>,
}

/// config.json's `quantization_config.weight_block_size` in `dir`: two positive integers, else None.
fn fp8_block(dir: &Path) -> Option<[usize; 2]> {
    let config = json::parse(&std::fs::read_to_string(dir.join("config.json")).ok()?).ok()?;
    match config.get("quantization_config")?.get("weight_block_size")?.as_u64s()?[..] {
        [m, n] if m > 0 && n > 0 => Some([m as usize, n as usize]),
        _ => None,
    }
}

impl Checkpoint {
    pub fn open(dir: &Path) -> io::Result<Checkpoint> {
        let index = dir.join("model.safetensors.index.json");
        let mut shards: Vec<String> = Vec::new();
        let mut map = None;
        if index.exists() {
            let v = json::parse(&std::fs::read_to_string(&index).map_err(at(&index))?).map_err(|e| bad(&index, e))?;
            let mut m = Vec::new();
            for (name, f) in v.get("weight_map").and_then(Value::as_object).ok_or_else(|| bad(&index, "no weight_map"))? {
                let f = f.as_str().filter(|f| !f.is_empty() && !f.contains('/') && *f != "..").ok_or_else(|| bad(&index, "a shard not a file of its directory"))?;
                if !shards.iter().any(|s| s == f) {
                    shards.push(f.to_string());
                }
                m.push((name.clone(), f.to_string()));
            }
            map = Some(m);
        } else {
            shards.push("model.safetensors".into());
        }
        let mut c = Checkpoint { dir: dir.to_path_buf(), files: Vec::new(), index: map, fp8_block: fp8_block(dir), names: HashMap::new() };
        for s in shards {
            let f = File::open(&dir.join(&s))?;
            for (j, (name, _)) in f.tensors.iter().enumerate() {
                if c.names.insert(name.clone(), (c.files.len(), j)).is_some() {
                    return Err(bad(&f.path, format!("{name}, in two of the shards")));
                }
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

    /// Every tensor's name, the shards' in turn.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.files.iter().flat_map(|f| f.tensors.iter().map(|(n, _)| n.as_str()))
    }

    fn missing(&self, name: &str) -> io::Error {
        io::Error::new(io::ErrorKind::NotFound, format!("{}: no tensor {name}", self.dir.display()))
    }

    /// An F8_E4M3 matrix `name` (`t` its info): its blocks, and its scales' tensor `NAME_scale_inv` (so `X.weight`'s are `X.weight_scale_inv`), checked to be F32 of the grid the blocks make.
    fn fp8_scales(&self, name: &str, t: &TensorInfo) -> io::Result<([usize; 2], &File, &TensorInfo)> {
        let what = |why: String| bad(&self.dir, format!("{name}: {why}"));
        if t.shape.len() != 2 {
            return Err(what(format!("F8_E4M3 of shape {:?}, where a matrix is wanted", t.shape)));
        }
        let block = self.fp8_block.ok_or_else(|| what("F8_E4M3, and config.json gives no quantization_config.weight_block_size for its scales".into()))?;
        let scale = format!("{name}_scale_inv");
        let (f, st) = self.get(&scale).ok_or_else(|| what(format!("F8_E4M3, and there is no {scale}, its block scales")))?;
        let grid = fp8::grid(t.shape[0] as usize, t.shape[1] as usize, block).map(|g| g as u64);
        if st.dtype != "F32" || st.shape != grid {
            return Err(what(format!("its block scales {scale} are {} {:?}, where F32 {grid:?} is wanted for blocks of {} x {}", st.dtype, st.shape, block[0], block[1])));
        }
        Ok((block, f, st))
    }

    /// Tensor `name` as the bf16 matrix [`Checkpoint::read_bf16`] reads: a BF16 tensor's own info; an F8_E4M3 matrix's with the dtype BF16 and the bytes its words take (its block scales checked); any other
    /// dtype is refused.
    pub fn matrix(&self, name: &str) -> io::Result<TensorInfo> {
        let (_, t) = self.get(name).ok_or_else(|| self.missing(name))?;
        match t.dtype.as_str() {
            "BF16" => Ok(t.clone()),
            "F8_E4M3" => {
                self.fp8_scales(name, t)?;
                Ok(TensorInfo { dtype: "BF16".into(), shape: t.shape.clone(), offset: t.offset, bytes: t.bytes * 2 })
            }
            other => Err(bad(&self.dir, format!("{name}: {other}, where a bf16 matrix, or an FP8 one with its block scales, is wanted"))),
        }
    }

    /// Tensor `name` as bf16 words into `out` (its elements): a BF16 tensor as it is; an F8_E4M3 matrix times its block scales ([`crate::fp8::dequant`]: each weight's f32 product rounded to bf16 once), its scales
    /// being `NAME_scale_inv` and its blocks config.json's `weight_block_size`. Any other dtype is refused.
    pub fn read_bf16(&self, name: &str, out: &mut [u16]) -> io::Result<()> {
        let (f, t) = self.get(name).ok_or_else(|| self.missing(name))?;
        match t.dtype.as_str() {
            "BF16" => {
                if t.bytes as usize != out.len() * 2 {
                    return Err(bad(&self.dir, format!("{name}: {} bytes, where {} are wanted", t.bytes, out.len() * 2)));
                }
                // SAFETY: u16's bytes, as many as its elements take; the file's are little-endian, as the host's.
                let bytes = unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, out.len() * 2) };
                f.read_into(t, bytes)?;
                if cfg!(target_endian = "big") {
                    out.iter_mut().for_each(|v| *v = u16::from_le(*v));
                }
                Ok(())
            }
            "F8_E4M3" => {
                let (block, sf, st) = self.fp8_scales(name, t)?;
                let (rows, cols) = (t.shape[0] as usize, t.shape[1] as usize);
                if out.len() != rows * cols {
                    return Err(bad(&self.dir, format!("{name}: {} weights, where {} are wanted", rows * cols, out.len())));
                }
                let scales: Vec<f32> = sf.read(st)?.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
                let mut q = vec![0u8; rows * cols];
                f.read_into(t, &mut q)?;
                fp8::dequant(&q, &scales, rows, cols, block, out).map_err(|e| bad(&self.dir, format!("{name}: {e}")))
            }
            other => Err(bad(&self.dir, format!("{name}: {other}, where a bf16 matrix, or an FP8 one with its block scales, is wanted"))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A file of two tensors, then the same with a gap between them, overlapping, and with bytes past the last: the
    /// first read, the others refused.
    #[test]
    fn tensors_back_to_back() {
        let dir = std::env::temp_dir().join(format!("glyd-gpu-st-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = |offs: [(u64, u64); 2], data: usize| {
            let h = format!(r#"{{"a":{{"dtype":"U8","shape":[{}],"data_offsets":[{},{}]}},"b":{{"dtype":"U8","shape":[{}],"data_offsets":[{},{}]}}}}"#, offs[0].1 - offs[0].0, offs[0].0, offs[0].1, offs[1].1 - offs[1].0, offs[1].0, offs[1].1);
            let p = dir.join("model.safetensors");
            let mut b = (h.len() as u64).to_le_bytes().to_vec();
            b.extend(h.as_bytes());
            b.extend(vec![7u8; data]);
            std::fs::write(&p, b).unwrap();
            File::open(&p).map(|f| f.tensors.len())
        };
        assert_eq!(file([(0, 4), (4, 10)], 10).unwrap(), 2);
        assert!(file([(0, 4), (5, 10)], 10).unwrap_err().to_string().contains("b's bytes at 5"));
        assert!(file([(0, 5), (4, 10)], 10).unwrap_err().to_string().contains("where the tensor before ends at 5"));
        assert!(file([(0, 4), (4, 10)], 12).unwrap_err().to_string().contains("2 bytes past its last tensor"));
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
