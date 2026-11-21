//! GLM-5.3-Flash's mixture-of-experts layer with the routed experts as rows of 2-bit codebook codes in a rotated basis (glyd-hbr-codes/2, written by research/glm-data/glm_pack.py; kernels/glm_cb.cu has the arithmetic),
//! `GlmMoeCb`, the `MoeBlock` that model_glm.rs's layers call: the router (a pruned expert never chosen), the rotation of the layer's input, the experts' two products over the cache's places, the weighted sum and the
//! rotation back, the shared expert (the engine's block layouts, as moe_glm.rs holds it). The experts of every layer are the expert cache's host tier (src/offload.rs): the layer's file mapped into memory as it is (the
//! pack's tensors are two runs of bytes there, the codebooks of the kept experts and then their codes, an expert's downT before its up; the mapping is unpinned page cache, which the system takes back when it needs the
//! memory); the GPU holds `slots` of a layer's experts in places, copied in as the router chooses them (least recently used out), and the misses are copied over PCIe. The router's choices are read on the host to place
//! them, so a decode step is not captured in a CUDA graph.

use crate::cb2_prefill::{cb2_prefill_experts, Cb2Kern, ExpertDesc};
use crate::cpu_cb::Fmt;
use crate::dev::Buf;
use crate::ffi::{err, Error, Result};
use crate::glm::ConfigGlm;
use crate::launch;
use crate::mapped::Mapping;
use crate::model_glm::{f32_bytes, EngineGlm, MoeBlock};
use crate::moe::{route_with, shared_add_with, shared_mlp_with, MoeCfg, MoeScratch, Shape, Shared};
use crate::moe_glm::moe_cfg;
use crate::offload::{Call, ExpertStore, HostMem, Placement, Stage, CPU_TOKENS};
use crate::weights::{read_pack, upload_bytes, upload_packed, v7_manifest, Layout};
use glyd_gpu::json::{self, Value};
use glyd_gpu::safetensors::Checkpoint;
use std::cell::OnceCell;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::Arc;

// The kernels' names: those the .cu sources give them, in every build (kname!).
macro_rules! kname {
    ($n:literal) => {
        $n
    };
}

/// The hidden width, the neurons of an expert and the bytes of a row of codes (2 bits a code, 4,096 codes).
const HIDDEN: usize = 4096;
const INTER: usize = 2048;
const ROW: usize = 1024;
/// Bytes of an expert's codes (up's 4,096 rows, then downT's 2,048) and of its codebooks (four bf16 a row).
pub const CODES: usize = (4096 + 2048) * ROW;
pub const BOOKS: usize = (4096 + 2048) * 8;
/// The most pair-runs (pairs times runs of neurons) the partial sums of a call hold.
const PART_ROWS: usize = 4096;
/// The most tokens the per-pair products take in one pass (their buffers are made for this many); a longer chunk is cut into passes of this many, or goes through the streaming prefill whole.
const PAIR_TOKENS: usize = 1024;

/// The glyd-hbr-codes/2 directory's `meta.json`: for each layer its file, its rotation's signs and the format of each of the model's experts.
pub struct CbPack {
    pub dir: PathBuf,
    layers: Vec<(usize, LayerMeta)>,
    /// Sidecar directories of other copies of some experts (`with_side`, in the order given: a later one's copy of an expert stands over an earlier one's).
    side_dirs: Vec<PathBuf>,
    /// The experts' width (the rows' length, the rotation's) and neurons: GLM's 4,096 and 2,048, Qwen3.8-Flash-Next's 2,560 and 640.
    pub hidden: usize,
    pub inter: usize,
}

struct LayerMeta {
    file: String,
    signs: Vec<i8>,
    /// the model's experts that are kept (ascending), in the order of the layer's experts in the host tier
    kept: Vec<usize>,
    /// the formats of each kept expert's up and its down
    fmts: Vec<(Fmt, Fmt)>,
    experts: usize,
    /// each sidecar's file for the layer (by the sidecar's number), and for each kept expert the sidecar its bytes are in (`CbPack::with_side`; none: all the layer's file's)
    side: Option<(Vec<Option<String>>, Vec<Option<u8>>)>,
}

impl LayerMeta {
    /// Whether every kept expert is two-bit in both matrices (the layout `glm_cb_up` and `glm_cb_down` are for; any other layer is run by `glm_cb_upx` and `glm_cb_downx`).
    fn uniform(&self) -> bool {
        // (a layer with a sidecar's experts is placed expert by expert, whatever their formats: they are in two files)
        self.side.is_none() && self.fmts.iter().all(|f| *f == (Fmt::Cb2, Fmt::Cb2))
    }

    /// The bytes of the biggest expert's codes and codebooks as the host tier holds them (experts of `inter` neurons over `hidden`).
    fn biggest(&self, hidden: usize, inter: usize) -> (usize, usize) {
        let bytes = |&(u, d): &(Fmt, Fmt)| (inter * d.row_bytes(hidden) + 2 * inter * u.row_bytes(hidden), (inter * d.entries() + 2 * inter * u.entries()) * 2);
        self.fmts.iter().map(bytes).fold((0, 0), |a, b| (a.0.max(b.0), a.1.max(b.1)))
    }
}

/// The formats of an expert's matrices from the pack's name for them: "cb2", "cb3", "cb4", "tern", or "X+Y" (X for the up, Y for the down).
fn parse_fmts(s: &str) -> Option<(Fmt, Fmt)> {
    match s.split_once('+') {
        Some((u, d)) => Some((Fmt::parse(u)?, Fmt::parse(d)?)),
        None => Fmt::parse(s).map(|f| (f, f)),
    }
}

/// The byte a place carries for the kernels (`glm_cb_upx`): up's format in the low nibble, down's in the high one.
fn fmt_nibble(f: Fmt) -> u8 {
    match f {
        Fmt::Cb2 => 0,
        Fmt::Cb3 => 1,
        Fmt::Cb4 => 2,
        Fmt::Tern => 3,
        Fmt::Tq => 4,
        Fmt::Tq3 => 5,
        Fmt::Tq4 => 6,
    }
}

/// A layer's file mapped and where its experts are: the host tier's offsets (for a layer that is not all two-bit), the strides of the places, and the experts as the streaming prefill is told of them.
pub struct LayerMap {
    pub map: Arc<Mapping>,
    /// where each kept expert's codes and codebooks are in the file, their lengths and formats (None: all two-bit and end to end from `starts`, `CODES` and `BOOKS` bytes each)
    pub place: Option<Placement>,
    /// the strides of the places on the GPU (codes, codebooks)
    pub strides: [usize; 2],
    /// where the codes and the codebooks of the first kept expert are in the file (a layer that is all two-bit; else 0)
    pub starts: (usize, usize),
    /// the kept experts as the streaming prefill takes them (addresses in the mapping)
    pub descs: Vec<ExpertDesc>,
}

impl CbPack {
    /// GLM-5.3-Flash's pack (experts of 2,048 neurons over 4,096).
    pub fn open(dir: &Path) -> Result<CbPack> {
        CbPack::open_dims(dir, HIDDEN, INTER)
    }

    /// A pack of experts of `inter` neurons over `hidden` (the signs of its rotation `hidden` of them).
    pub fn open_dims(dir: &Path, hidden: usize, inter: usize) -> Result<CbPack> {
        let path = dir.join("meta.json");
        let text = std::fs::read_to_string(&path).map_err(|e| Error(format!("{}: {e}", path.display())))?;
        let v = json::parse(&text).map_err(|e| Error(format!("{}: {e}", path.display())))?;
        if v.get("format").and_then(Value::as_str) != Some("glyd-x/1") {
            return err(format!("{}: not a glyd-x/1 directory", dir.display()));
        }
        let entries: Vec<(usize, &Value)> = match v.get("layers") {
            Some(Value::Object(o)) => o.iter().filter_map(|(k, v)| Some((k.parse().ok()?, v))).collect(),
            Some(Value::Array(a)) => a
                .iter()
                .filter_map(|v| {
                    let n = v.get("layer")?;
                    Some((n.as_u64().map(|x| x as usize).or_else(|| n.as_str()?.parse().ok())?, v))
                })
                .collect(),
            _ => return err(format!("{}: no layers", path.display())),
        };
        let mut layers = Vec::new();
        for (l, m) in entries {
            let file = m.get("file").and_then(Value::as_str).map(String::from).unwrap_or_else(|| format!("layer-{l:03}.safetensors"));
            let signs: Vec<i8> = m
                .get("s")
                .and_then(Value::as_u64s)
                .ok_or_else(|| Error(format!("{}: layer {l}: no s", path.display())))?
                .into_iter()
                .map(|b| if b == 1 { 1 } else { -1 })
                .collect();
            let fmt = m.get("f").and_then(Value::as_array).ok_or_else(|| Error(format!("{}: layer {l}: no f", path.display())))?;
            let (mut kept, mut fmts) = (Vec::new(), Vec::new());
            for (e, f) in fmt.iter().enumerate() {
                match f.as_str() {
                    Some("pruned") => {}
                    Some(name) => match parse_fmts(name) {
                        Some(pair) => {
                            kept.push(e);
                            fmts.push(pair);
                        }
                        None => return err(format!("{}: layer {l}: expert {e} is {name:?}; a format this engine does not read", path.display())),
                    },
                    None => return err(format!("{}: layer {l}: expert {e}'s format is not a string", path.display())),
                }
            }
            if signs.len() != hidden {
                return err(format!("{}: layer {l}: {} signs, where {hidden} are wanted", path.display(), signs.len()));
            }
            layers.push((l, LayerMeta { file, signs, kept, fmts, experts: fmt.len(), side: None }));
        }
        layers.sort_by_key(|(l, _)| *l);
        // (format q's rows are read with this engine's constants: the pack says which version of them wrote it, 1 for 2-bit rows alone, 2 with 3- and 4-bit ones)
        let q = v.get("q").and_then(Value::as_u64);
        let has = |f: Fmt| layers.iter().any(|(_, m)| m.fmts.iter().any(|&(u, d)| u == f || d == f));
        if (has(Fmt::Tq3) || has(Fmt::Tq4)) && q != Some(2) || has(Fmt::Tq) && !matches!(q, Some(1 | 2)) {
            return err(format!("{}: format q of a version this engine does not read", path.display()));
        }
        Ok(CbPack { dir: dir.to_path_buf(), layers, hidden, inter, side_dirs: Vec::new() })
    }

    /// The pack with the experts of a sidecar directory in place of its own copies of them (the packer's `sidecar`: its `meta.json` says the format, every one of its experts', and each layer's file and experts; the
    /// rotation is the pack's): e.g. the hottest experts at three bits instead of four, for more places on the GPU.
    ///
    /// A layer's entry may give each expert's format (`fmts`, beside `experts`) in place of the sidecar's one `fmt`.
    pub fn with_side(mut self, side: &Path) -> Result<CbPack> {
        let which = self.side_dirs.len();
        if which >= 255 {
            return err("too many GPU copies");
        }
        let path = side.join("meta.json");
        let text = std::fs::read_to_string(&path).map_err(|e| Error(format!("{}: {e}", path.display())))?;
        let v = json::parse(&text).map_err(|e| Error(format!("{}: {e}", path.display())))?;
        if v.get("format").and_then(Value::as_str) != Some("glyd-x/1") {
            return err(format!("{}: not a glyd-x/1 copy", side.display()));
        }
        let fmt = v.get("fmt").and_then(Value::as_str).and_then(parse_fmts);
        let Some(Value::Object(layers)) = v.get("layers") else { return err(format!("{}: no layers", path.display())) };
        for (k, m) in layers.iter() {
            let l: usize = k.parse().map_err(|_| Error(format!("{}: layer {k:?}", path.display())))?;
            let file = m.get("file").and_then(Value::as_str).ok_or_else(|| Error(format!("{}: layer {l}: no file", path.display())))?.to_string();
            let experts = m.get("experts").and_then(Value::as_u64s).ok_or_else(|| Error(format!("{}: layer {l}: no experts", path.display())))?;
            let fmts: Option<Vec<&str>> = m.get("fmts").and_then(Value::as_array).map(|a| a.iter().filter_map(Value::as_str).collect());
            let Some((_, meta)) = self.layers.iter_mut().find(|(x, _)| *x == l) else {
                return err(format!("{}: layer {l}, which the pack does not hold", path.display()));
            };
            let kept = meta.kept.len();
            let (files, of) = meta.side.get_or_insert_with(|| (Vec::new(), vec![None; kept]));
            files.resize(which + 1, None);
            files[which] = Some(file);
            for (j, e) in experts.into_iter().enumerate() {
                let i = meta
                    .kept
                    .iter()
                    .position(|&x| x == e as usize)
                    .ok_or_else(|| Error(format!("{}: layer {l}: expert {e}, which the pack does not keep", path.display())))?;
                let f = match &fmts {
                    Some(list) => list.get(j).and_then(|x| parse_fmts(x)),
                    None => fmt,
                }
                .ok_or_else(|| Error(format!("{}: layer {l}: expert {e} has no format (fmt, or fmts beside experts)", path.display())))?;
                of[i] = Some(which as u8);
                meta.fmts[i] = f;
            }
        }
        self.side_dirs.push(side.to_path_buf());
        Ok(self)
    }

    fn layer(&self, l: usize) -> Option<&LayerMeta> {
        self.layers.iter().find(|(x, _)| *x == l).map(|(_, m)| m)
    }

    /// Whether any expert of the pack has a matrix of format `f`.
    pub fn has_fmt(&self, f: Fmt) -> bool {
        self.layers.iter().any(|(_, m)| m.fmts.iter().any(|&(u, d)| u == f || d == f))
    }

    /// The bytes of the layers' files together.
    pub fn file_bytes(&self) -> u64 {
        self.layers.iter().filter_map(|(_, m)| std::fs::metadata(self.dir.join(&m.file)).ok()).map(|m| m.len()).sum()
    }

    /// The bytes of a place of layer `l` on the GPU (codes and codebooks together): the biggest of its experts' (two-bit experts alone: `CODES + BOOKS`), and a byte for the formats where the layer is not all two-bit.
    pub fn place_bytes(&self, l: usize) -> usize {
        let (codes, books) = self.uniform_bytes();
        self.layer(l).map_or(codes + books, |m| {
            if m.uniform() {
                codes + books
            } else {
                let (c, b) = m.biggest(self.hidden, self.inter);
                c.next_multiple_of(256) + (b + 16).next_multiple_of(16)
            }
        })
    }

    /// The bytes of a two-bit expert's codes and codebooks (GLM's `CODES` and `BOOKS`).
    pub fn uniform_bytes(&self) -> (usize, usize) {
        (3 * self.inter * self.hidden / 4, 3 * self.inter * 8)
    }

    /// Whether the directory holds layer `l`'s experts.
    pub fn has(&self, l: usize) -> bool {
        self.layer(l).is_some()
    }

    /// The signs of layer `l`'s rotation (+1 or -1).
    pub fn signs(&self, l: usize) -> Option<&[i8]> {
        self.layer(l).map(|m| m.signs.as_slice())
    }

    /// The kept experts of layer `l`.
    pub fn kept(&self, l: usize) -> Option<&[usize]> {
        self.layer(l).map(|m| m.kept.as_slice())
    }

    /// Layer `l`'s file mapped and where its kept experts are. A layer of two-bit experts only has the layout the cache was first made for: each run (the codebooks of the kept experts, the codes) holds them end to end, an
    /// expert's downT then its up, `BOOKS` and `CODES` bytes each. A layer of other formats, or of mixed ones, is placed expert by expert from the file's header: the offset and length of each expert's codes (its downT's
    /// rows, then its up's: the checks are that they follow each other) and codebooks, the places on the GPU as big as the biggest expert (a byte more, for the formats). The layout is checked for every expert.
    pub fn map_layer(&self, l: usize) -> Result<LayerMap> {
        let m = self.layer(l).ok_or_else(|| Error(format!("{}: no layer {l}", self.dir.display())))?;
        let path = self.dir.join(&m.file);
        let f = glyd_gpu::safetensors::File::open(&path).map_err(|e| Error(format!("{}: {e}", path.display())))?;
        let first = *m.kept.first().ok_or_else(|| Error(format!("{}: layer {l} keeps no expert", self.dir.display())))?;
        let off = |e: usize, what: &str, dtype: &str, shape: [u64; 2]| -> Result<u64> {
            // (GLM's pack names an expert in three digits, Qwen3.8-Flash-Next's in four)
            let name = format!("e{e:03}.{what}");
            let t = f.get(&name).or_else(|| f.get(&format!("e{e:04}.{what}"))).ok_or_else(|| Error(format!("{}: no tensor {name}", path.display())))?;
            if t.dtype != dtype || t.shape != shape {
                return err(format!("{}: {name} is {} {:?}, where {dtype} {shape:?} is wanted", path.display(), t.dtype, t.shape));
            }
            Ok(t.offset)
        };
        let (hid, inter) = (self.hidden, self.inter);
        let (n, n2, row) = (inter as u64, 2 * inter as u64, (hid / 4) as u64);
        let (codes_b, books_b) = self.uniform_bytes();
        if m.uniform() {
            let (b0, c0) = (off(first, "d.1", "BF16", [n, 4])?, off(first, "d.0", "U8", [n, row])?);
            for (i, &e) in m.kept.iter().enumerate() {
                let (b, c) = (b0 + (i * books_b) as u64, c0 + (i * codes_b) as u64);
                let ok = off(e, "d.1", "BF16", [n, 4])? == b
                    && off(e, "u.1", "BF16", [n2, 4])? == b + n * 8
                    && off(e, "d.0", "U8", [n, row])? == c
                    && off(e, "u.0", "U8", [n2, row])? == c + n * row;
                if !ok {
                    return err(format!(
                        "{}: expert {e}'s tensors are not where the {}th expert's are in a run of the kept experts end to end",
                        path.display(),
                        i
                    ));
                }
            }
            let map = Arc::new(Mapping::open(&path)?);
            if (c0 as usize) + m.kept.len() * codes_b > map.len || (b0 as usize) + m.kept.len() * books_b > map.len {
                return err(format!("{}: the tensors reach past the end of the file", path.display()));
            }
            // SAFETY: pointer arithmetic inside the mapping (checked above).
            let descs = (0..m.kept.len())
                .map(|i| unsafe {
                    ExpertDesc {
                        codes: (map.ptr as *const u8).add(c0 as usize + i * codes_b),
                        books: (map.ptr as *const u8).add(b0 as usize + i * books_b),
                        up: Fmt::Cb2,
                        down: Fmt::Cb2,
                        resident: None,
                        hid,
                        inter,
                    }
                })
                .collect();
            return Ok(LayerMap { map, place: None, strides: [codes_b, books_b], starts: (c0 as usize, b0 as usize), descs });
        }
        let map = Arc::new(Mapping::open(&path)?);
        // (a sidecar's experts from its own file: the placement then holds every expert's address, not offsets in the layer's file)
        // (each sidecar's file for the layer, mapped: (path, header, mapping) by the sidecar's number)
        let mut sides: Vec<Option<(PathBuf, glyd_gpu::safetensors::File, Arc<Mapping>)>> = Vec::new();
        let of: Vec<Option<u8>> = m.side.as_ref().map_or(vec![None; m.kept.len()], |s| s.1.clone());
        if let Some((files, _)) = &m.side {
            for (j, file) in files.iter().enumerate() {
                let opened = match file {
                    Some(file) if of.contains(&Some(j as u8)) => {
                        let sp = self.side_dirs[j].join(file);
                        let sf = glyd_gpu::safetensors::File::open(&sp).map_err(|e| Error(format!("{}: {e}", sp.display())))?;
                        let sm = Arc::new(Mapping::open(&sp)?);
                        Some((sp, sf, sm))
                    }
                    _ => None,
                };
                sides.push(opened);
            }
        }
        let side = !sides.iter().all(Option::is_none);
        let (mut at, mut len) = ([Vec::new(), Vec::new()], [Vec::new(), Vec::new()]);
        let mut descs = Vec::new();
        for (i, &e) in m.kept.iter().enumerate() {
            let (fu, fd) = m.fmts[i];
            let (rbu, rbd, ku, kd) = (fu.row_bytes(hid) as u64, fd.row_bytes(hid) as u64, fu.entries() as u64, fd.entries() as u64);
            let theirs = of[i].and_then(|j| sides.get(j as usize).and_then(Option::as_ref));
            let (file, fpath, fmap) = match theirs {
                Some((sp, sf, sm)) => (sf, sp, sm),
                None => (&f, &path, &map),
            };
            let off = |what: &str, dtype: &str, shape: [u64; 2]| -> Result<u64> {
                let name = format!("e{e:03}.{what}");
                let t =
                    file.get(&name).or_else(|| file.get(&format!("e{e:04}.{what}"))).ok_or_else(|| Error(format!("{}: no tensor {name}", fpath.display())))?;
                if t.dtype != dtype || t.shape != shape {
                    return err(format!("{}: {name} is {} {:?}, where {dtype} {shape:?} is wanted", fpath.display(), t.dtype, t.shape));
                }
                Ok(t.offset)
            };
            let (dc, uc) = (off("d.0", "U8", [n, rbd])?, off("u.0", "U8", [n2, rbu])?);
            let (db, ub) = (off("d.1", "BF16", [n, kd])?, off("u.1", "BF16", [n2, ku])?);
            if uc != dc + n * rbd || ub != db + n * kd * 2 {
                return err(format!(
                    "{}: expert {e}: the up's tensors do not follow the down's ({} + {}): an expert's parts are each copied as one range",
                    fpath.display(),
                    fd.name(),
                    fu.name()
                ));
            }
            let (cl, bl) = (n * rbd + n2 * rbu, (n * kd + n2 * ku) * 2);
            if (dc + cl) as usize > fmap.len || (db + bl) as usize > fmap.len {
                return err(format!("{}: expert {e}'s tensors reach past the end of the file", fpath.display()));
            }
            let base = if side { fmap.ptr as u64 } else { 0 };
            at[0].push(base + dc);
            at[1].push(base + db);
            len[0].push(cl as u32);
            len[1].push(bl as u32);
            // SAFETY: inside the mapping (checked above).
            descs.push(unsafe {
                ExpertDesc {
                    codes: (fmap.ptr as *const u8).add(dc as usize),
                    books: (fmap.ptr as *const u8).add(db as usize),
                    up: fu,
                    down: fd,
                    resident: None,
                    hid,
                    inter,
                }
            });
        }
        let (cmax, bmax) = (len[0].iter().copied().max().unwrap_or(0) as usize, len[1].iter().copied().max().unwrap_or(0) as usize);
        let strides = [cmax.next_multiple_of(256), (bmax + 16).next_multiple_of(16)];
        let marks: Vec<u8> = m.fmts.iter().map(|&(u, d)| fmt_nibble(u) | fmt_nibble(d) << 4).collect();
        let maps = if side { std::iter::once(map.clone()).chain(sides.iter().flatten().map(|x| x.2.clone())).collect() } else { Vec::new() };
        Ok(LayerMap { map, place: Some(Placement { at, len, fmts: m.fmts.clone(), marks, maps }), strides, starts: (0, 0), descs })
    }
}

impl CbPack {
    /// Layer `l`'s experts (as `map_layer` placed them, `lm`) copied into pinned host memory once: each expert's codes end to end in expert order, then each one's codebooks (64-byte aligned), so that the GPU's copies
    /// of them are DMA and the host's reads see the same bytes; the files' pages then left to the system (DONTNEED) and their mappings dropped. A layer without a placement (all two-bit) is left mapped.
    pub fn pin_layer(&self, l: usize, lm: LayerMap) -> Result<LayerMap> {
        let Some(place) = lm.place.as_ref() else { return Ok(lm) };
        let n = place.len[0].len();
        let up = |x: usize| x.next_multiple_of(64);
        let (codes_total, books_total): (usize, usize) =
            (place.len[0].iter().map(|&x| up(x as usize)).sum(), place.len[1].iter().map(|&x| up(x as usize)).sum());
        let blob = Arc::new(Mapping::pinned(codes_total + books_total)?);
        // each expert's source (an offset in the layer's file, or an address where the placement holds addresses) and destination
        let abs = !place.maps.is_empty();
        let base = lm.map.ptr as usize;
        let mut jobs: Vec<(usize, usize, usize)> = Vec::with_capacity(2 * n);
        let mut at = [Vec::with_capacity(n), Vec::with_capacity(n)];
        let (mut co, mut bo) = (0usize, codes_total);
        for i in 0..n {
            for (k, off) in [(0usize, &mut co), (1, &mut bo)] {
                let src = if abs { place.at[k][i] as usize } else { base + place.at[k][i] as usize };
                let dst = blob.ptr as usize + *off;
                jobs.push((src, dst, place.len[k][i] as usize));
                at[k].push(dst as u64);
                *off += up(place.len[k][i] as usize);
            }
        }
        // (the copies on 8 threads: a layer is about 800 MB, from the page cache or the drive)
        let per = jobs.len().div_ceil(8);
        std::thread::scope(|sc| {
            for part in jobs.chunks(per.max(1)) {
                sc.spawn(move || {
                    for &(src, dst, len) in part {
                        // SAFETY: src is inside a mapping `lm` holds for this scope (checked by `map_layer`), dst inside the pinned blob (sized above); the ranges do not overlap.
                        unsafe { std::ptr::copy_nonoverlapping(src as *const u8, dst as *mut u8, len) };
                    }
                });
            }
        });
        let descs = lm
            .descs
            .iter()
            .enumerate()
            .map(|(i, d)| ExpertDesc { codes: at[0][i] as usize as *const u8, books: at[1][i] as usize as *const u8, ..*d })
            .collect();
        let placed = Placement { at, len: place.len.clone(), fmts: place.fmts.clone(), marks: place.marks.clone(), maps: vec![blob.clone()] };
        let strides = lm.strides;
        // the files' pages given back (the bytes are in the blob now)
        drop(lm);
        if let Some(m) = self.layer(l) {
            Mapping::drop_cache(&self.dir.join(&m.file));
            if let Some((files, _)) = &m.side {
                for (j, file) in files.iter().enumerate() {
                    if let (Some(file), Some(dir)) = (file, self.side_dirs.get(j)) {
                        Mapping::drop_cache(&dir.join(file));
                    }
                }
            }
        }
        Ok(LayerMap { map: blob, place: Some(placed), strides, starts: (0, 0), descs })
    }

    /// The bytes the experts of every layer take as `pin_layer` copies them (each expert's codes and codebooks, 64-byte aligned), without mapping a file.
    pub fn expert_bytes(&self) -> usize {
        let (hid, inter) = (self.hidden, self.inter);
        self.layers
            .iter()
            .map(|(_, m)| {
                m.fmts
                    .iter()
                    .map(|&(u, d)| {
                        (inter * d.row_bytes(hid) + 2 * inter * u.row_bytes(hid)).next_multiple_of(64)
                            + ((inter * d.entries() + 2 * inter * u.entries()) * 2).next_multiple_of(64)
                    })
                    .sum::<usize>()
            })
            .sum()
    }
}

/// The kernels of kernels/glm_cb.cu with typed launches.
pub struct KernCb {
    rot_in: crate::dev::Kernel,
    local: crate::dev::Kernel,
    up: crate::dev::Kernel,
    down: crate::dev::Kernel,
    /// the same for places of any format (a layer that is not all two-bit)
    upx: crate::dev::Kernel,
    downx: crate::dev::Kernel,
    sum_rot: crate::dev::Kernel,
}

impl KernCb {
    pub fn load(dev: &crate::dev::Dev) -> Result<KernCb> {
        Ok(KernCb {
            rot_in: dev.kernel(kname!("glm_cb_rot_in"))?,
            local: dev.kernel(kname!("glm_cb_local"))?,
            up: dev.kernel(kname!("glm_cb_up"))?,
            down: dev.kernel(kname!("glm_cb_down"))?,
            upx: dev.kernel(kname!("glm_cb_upx"))?,
            downx: dev.kernel(kname!("glm_cb_downx"))?,
            sum_rot: dev.kernel(kname!("glm_cb_sum_rot"))?,
        })
    }

    /// x' = R x for n rows of x (bf16, rows `xs` apart).
    ///
    /// # Safety
    /// Live device buffers of those sizes, used on the stream.
    pub unsafe fn rot_in(&self, s: crate::ffi::Stream, x: u64, xs: i32, sg: u64, xr: u64, xrb: u64, n: usize) -> Result<()> {
        launch!(self.rot_in, [n as u32, 1, 1], [1024, 1, 1], s; x, xs, sg, xr, xrb)
    }

    /// The model's expert numbers of the pairs' choices as the layer's numbers.
    ///
    /// # Safety
    /// As `rot_in`.
    pub unsafe fn local(&self, s: crate::ffi::Stream, ids: u64, local: u64, out: u64, pairs: i32, experts: i32) -> Result<()> {
        launch!(self.local, [(pairs as u32).div_ceil(256), 1, 1], [256, 1, 1], s; ids, local, out, pairs, experts)
    }

    /// The first products of the pairs that have a place (`way`: the place of each pair, -1 for none).
    ///
    /// # Safety
    /// As `rot_in`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn up(
        &self,
        s: crate::ffi::Stream,
        codes: u64,
        cstride: u64,
        books: u64,
        bstride: u64,
        way: u64,
        xr: u64,
        gu: u64,
        k: i32,
        pairs: usize,
        mixed: bool,
    ) -> Result<()> {
        if mixed {
            return launch!(self.upx, [128, pairs as u32, 1], [256, 1, 1], s; codes, cstride, books, bstride, way, xr, gu, k);
        }
        launch!(self.up, [128, pairs as u32, 1], [256, 1, 1], s; codes, cstride, books, bstride, way, xr, gu, k)
    }

    /// The activation and the second products, `split` runs of neurons a pair, partial sums into `part`.
    ///
    /// # Safety
    /// As `rot_in`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn down(
        &self,
        s: crate::ffi::Stream,
        codes: u64,
        cstride: u64,
        books: u64,
        bstride: u64,
        way: u64,
        gu: u64,
        part: u64,
        split: i32,
        limit: f32,
        pairs: usize,
        mixed: bool,
    ) -> Result<()> {
        if mixed {
            return launch!(self.downx, [4, split as u32, pairs as u32], [256, 1, 1], s; codes, cstride, books, bstride, way, gu, part, split, limit);
        }
        launch!(self.down, [4, split as u32, pairs as u32], [256, 1, 1], s; codes, cstride, books, bstride, way, gu, part, split, limit)
    }

    /// The weighted sum of each token's pairs, the runs added, R^T, one rounding to bf16 into `y`.
    ///
    /// # Safety
    /// As `rot_in`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn sum_rot(
        &self,
        s: crate::ffi::Stream,
        part: u64,
        wts: u64,
        extra: u64,
        skip: u64,
        sg: u64,
        y: u64,
        k: i32,
        split: i32,
        n: usize,
    ) -> Result<()> {
        launch!(self.sum_rot, [n as u32, 1, 1], [1024, 1, 1], s; part, wts, extra, skip, sg, y, k, split)
    }
}

/// The buffers all the model's layers share (a layer's calls follow each other on the stream), made by the first call.
pub struct CbShared {
    scratch: OnceCell<CbScratch>,
    /// Experts a token takes now, at most the model's (0: the model's): a development session may lower it between requests.
    pub topk: std::cell::Cell<usize>,
    /// Whether a prompt chunk of more than `CPU_TOKENS` tokens goes through the streaming prefill (src/cb2_prefill.rs) instead of the cache's per-pair products.
    pub stream: std::cell::Cell<bool>,
    /// The bytes of the pack's files that loading may still ask the system to read ahead (u64::MAX: all): a pack bigger than the computer's memory is not read ahead whole, the page cache would only be turned over.
    pub read_ahead: std::cell::Cell<u64>,
    /// A development session's timeline of the decode steps (src/trace.rs): 0 off, 1 the key markers, 2 every stage marker.
    pub trace: std::cell::Cell<u8>,
}

struct CbScratch {
    kern: KernCb,
    xr: Buf,
    xrb: Buf,
    gids: Buf,
    gu: Buf,
    part: Buf,
    /// -1 for every pair of a chunk (the sum that takes the streaming prefill's result leaves all the pairs' own sums out)
    neg: Buf,
    pre: OnceCell<Cb2Kern>,
}

/// Whether the streaming prefill is on at the start: not yet by default (GLYD_CB2_PREFILL=1 turns it on, and a `set` request may), until it has been checked on the real model.
pub fn stream_default() -> bool {
    crate::tune!("GLYD_CB2_PREFILL").as_deref() == Some("1")
}

impl CbShared {
    pub fn new() -> Rc<CbShared> {
        Rc::new(CbShared {
            scratch: OnceCell::new(),
            topk: std::cell::Cell::new(0),
            stream: std::cell::Cell::new(stream_default()),
            read_ahead: std::cell::Cell::new(u64::MAX),
            trace: std::cell::Cell::new(0),
        })
    }
}

/// A mixture-of-experts layer on the GPU with its routed experts in the expert cache.
pub struct GlmMoeCb {
    cfg: MoeCfg,
    router: Buf,
    bias: Buf,
    shared: Shared,
    signs: Buf,
    local_of: Buf,
    experts: usize,
    /// the places of the cache on the GPU: the codes, the codebooks
    places: [Buf; 2],
    /// the layer's file mapped (kept alive), the kept experts as the streaming prefill takes them, the strides of the places, and whether the layer is not all two-bit (the kernels read the formats from the places)
    #[allow(dead_code)]
    map: Arc<Mapping>,
    descs: Vec<ExpertDesc>,
    strides: [usize; 2],
    mixed: bool,
    #[allow(dead_code)] // the GLM development engine (R13 retires or moves it)
    kept: usize,
    store: Rc<ExpertStore>,
    layer: usize,
    common: Rc<CbShared>,
}

impl GlmMoeCb {
    /// Layer `l`: the router (a pruned expert's bias -1e30, so that it is never among the best eight), the shared expert at `level`, the layer's rotation, and the experts: read from `pack` into the cache's host tier, with
    /// `slots` places on the GPU.
    pub fn load(
        l: usize,
        c: &ConfigGlm,
        ck: &Checkpoint,
        level: Layout,
        pack: &CbPack,
        store: &Rc<ExpertStore>,
        slots: usize,
        topk: usize,
        common: &Rc<CbShared>,
    ) -> Result<GlmMoeCb> {
        if !c.moe_of(l) {
            return err(format!("layer {l} is not a mixture-of-experts layer"));
        }
        if (c.hidden, c.experts.inter) != (HIDDEN, INTER) {
            return err(format!("experts of {} neurons over {}: the kernels take {INTER} over {HIDDEN}", c.experts.inter, c.hidden));
        }
        let meta = pack.layer(l).ok_or_else(|| Error(format!("{}: no experts of layer {l}", pack.dir.display())))?;
        if meta.experts != c.experts.experts {
            return err(format!("{}: layer {l}: {} experts, where the model has {}", pack.dir.display(), meta.experts, c.experts.experts));
        }
        let p = format!("{}layers.{l}.mlp", c.prefix);
        let name = format!("{p}.gate.weight");
        let (_, info) = ck.get(&name).ok_or_else(|| Error(format!("no tensor {name}")))?;
        if info.dtype != "BF16" {
            return err(format!("{name} is {}, where the router is read as BF16", info.dtype));
        }
        let router = ck.read(&name).map_err(|e| Error(format!("{name}: {e}")))?;
        let mut bias = f32_bytes(ck, &format!("{p}.gate.e_score_correction_bias"))?;
        let mut local = vec![-1i32; meta.experts];
        for (i, &e) in meta.kept.iter().enumerate() {
            local[e] = i as i32;
        }
        for (e, b) in bias.chunks_exact_mut(4).enumerate() {
            if local[e] < 0 {
                b.copy_from_slice(&(-1e30f32).to_le_bytes());
            }
        }
        let v7 = v7_manifest(&ck.dir)?;
        let packs = c.layer_packs(l);
        let [s_up, s_down] = &packs[packs.len() - 2..] else {
            return err(format!("layer {l}: a mixture-of-experts layer has two matrices besides its experts and its attention's"));
        };
        let (s_up, s_down) = (read_pack(ck, v7.as_ref(), s_up, level)?, read_pack(ck, v7.as_ref(), s_down, level)?);
        let shared = Shared { inter: c.experts.shared_inter, up: upload_packed(s_up)?, down: upload_packed(s_down)?, gate: None };
        let kept = meta.kept.len();
        let slots = slots.clamp(1, kept);
        let lm = pack.map_layer(l)?;
        let (mixed, strides) = (lm.place.is_some(), lm.strides);
        let places = [Buf::new(slots * strides[0])?, Buf::new(slots * strides[1])?];
        let LayerMap { map, place, starts, descs, .. } = lm;
        {
            // (the file is asked to be read ahead, in the background, while the rest of the model loads, as far as the budget goes: the page cache is warm when the first request comes)
            let ahead = (map.len as u64).min(common.read_ahead.get());
            common.read_ahead.set(common.read_ahead.get().saturating_sub(ahead));
            map.will_need(0, ahead as usize);
            let limit = match moe_cfg(c).act {
                crate::moe::Act::SwiGlu { limit: Some(l) } => l,
                _ => return err("the experts take a clamped SwiGLU"),
            };
            store.add_layer_mapped(
                l,
                kept,
                slots,
                [HostMem::mapped(&map, starts.0), HostMem::mapped(&map, starts.1)],
                place,
                [strides[0] as u64, strides[1] as u64],
                [places[0].ptr, places[1].ptr],
                Some((limit, HIDDEN, INTER)),
            )?;
        }
        let mut cfg = moe_cfg(c);
        if topk != 0 {
            if topk > cfg.router.top_k {
                return err(format!("{topk} experts a token, where the model picks {}", cfg.router.top_k));
            }
            cfg.router.top_k = topk;
        }
        Ok(GlmMoeCb {
            cfg,
            router: upload_bytes(&router)?,
            bias: upload_bytes(&bias)?,
            shared,
            signs: upload_bytes(&meta.signs.iter().map(|&s| s as u8).collect::<Vec<u8>>())?,
            local_of: upload_bytes(&local.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>())?,
            experts: meta.experts,
            places,
            map,
            descs,
            strides,
            mixed,
            kept,
            store: store.clone(),
            layer: l,
            common: common.clone(),
        })
    }

    /// The MoE scratch every layer shares, and this module's beside it, made by the first call.
    fn scratch<'a>(&'a self, e: &'a EngineGlm) -> Result<(&'a MoeScratch, &'a CbScratch)> {
        if e.moe_scratch.get().is_none() {
            let shape = Shape {
                cfg: self.cfg,
                twelve: false,
                shared: Some((self.shared.inter, self.shared.up.pack.cols as usize, self.shared.down.pack.rows as usize)),
                expert_rows: self.cfg.inter,
                sparse: true,
                slice: 1,
                hb: 0,
            };
            let _ = e.moe_scratch.set(MoeScratch::for_shapes(&e.dev, &e.lib, &[shape], e.chunk())?);
        }
        let sc = e.moe_scratch.get().ok_or_else(|| Error("the mixture-of-experts scratch is not made".into()))?;
        if self.common.scratch.get().is_none() {
            let (n, k) = (e.chunk(), self.cfg.router.top_k);
            let z = CbBytes::of(n, k);
            // (the per-pair products' buffers: for a pass of PAIR_TOKENS tokens at most; partial sums for every pair of a pass, and for the most pairs that are cut into runs of neurons)
            let made = CbScratch {
                kern: KernCb::load(&e.dev)?,
                xr: Buf::new(z.xr)?,
                xrb: Buf::new(z.xrb)?,
                gids: Buf::new(z.gids)?,
                gu: Buf::new(z.gu)?,
                part: Buf::new(z.part)?,
                neg: {
                    let b = Buf::new(z.neg)?;
                    b.write(0, &vec![-1i64; n * k])?;
                    b
                },
                pre: OnceCell::new(),
            };
            for (b, need, what) in [
                (&made.xr, z.xr, "x' in f32"),
                (&made.xrb, z.xrb, "x' in bf16"),
                (&made.gids, z.gids, "ids"),
                (&made.gu, z.gu, "first products"),
                (&made.part, z.part, "partial sums"),
                (&made.neg, z.neg, "skip ids"),
            ] {
                holds(b, need, what, n, k)?;
            }
            let _ = self.common.scratch.set(made);
        }
        Ok((sc, self.common.scratch.get().ok_or_else(|| Error("the scratch is not made".into()))?))
    }
}

/// Bytes of the codebook block's scratch for calls of up to `n` rows of `k` pairs each, from the chunk and the top-k alone: x' in f32 and in bf16 (a row of `HIDDEN` each), the pairs' local ids (int64), the first products
/// of a pass of the per-pair path (`PAIR_TOKENS` rows at most; fp32 `HIDDEN` a pair), the partial sums of a pass (`runs_for` runs of every pair, within `PART_ROWS` rows of `HIDDEN` fp32 unless the pairs alone are more) and
/// the all -1 ids of the streaming prefill's sum.
struct CbBytes {
    xr: usize,
    xrb: usize,
    gids: usize,
    gu: usize,
    part: usize,
    neg: usize,
}

impl CbBytes {
    fn of(n: usize, k: usize) -> CbBytes {
        let pass = n.min(PAIR_TOKENS) * k;
        CbBytes { xr: n * HIDDEN * 4, xrb: n * HIDDEN * 2, gids: n * k * 8, gu: pass * HIDDEN * 4, part: PART_ROWS.max(pass) * HIDDEN * 4, neg: n * k * 8 }
    }
}

/// An error where a buffer holds fewer bytes than a call needs (a kernel would write past it).
fn holds(b: &Buf, need: usize, what: &str, n: usize, k: usize) -> Result<()> {
    if b.bytes < need {
        return err(format!("the block's {what} holds {} bytes, where {n} rows of {k} pairs take {need}", b.bytes));
    }
    Ok(())
}

/// The runs the neurons are cut into for `pairs` pairs: enough blocks to fill the GPU, and the partial sums within `PART_ROWS`.
fn runs_for(pairs: usize) -> usize {
    let s = match pairs {
        0..=32 => 16,
        33..=64 => 8,
        65..=128 => 4,
        129..=256 => 2,
        _ => 1,
    };
    s.min((PART_ROWS / pairs.max(1)).max(1))
}

impl GlmMoeCb {
    /// `n` rows, each pair's products through the cache (`stream`: false) or the whole chunk through the streaming prefill (src/cb2_prefill.rs: the layer's experts that have pairs streamed from the mapped file through a ring
    /// on the GPU, the tensor cores' products), the weighted sum, R^T, the shared expert.
    ///
    /// # Safety
    /// As `MoeBlock::forward`.
    unsafe fn run_rows(&self, e: &EngineGlm, h: u64, n: usize, out: u64, stream: bool) -> Result<()> {
        let s = e.dev.stream;
        let mut cur = self.cfg;
        let want = self.common.topk.get();
        if want != 0 {
            cur.router.top_k = want.min(self.cfg.router.top_k);
        }
        let c = &cur;
        let (sc, cs) = self.scratch(e)?;
        let (k, pairs) = (c.router.top_k, n * c.router.top_k);
        // (a call is within what the buffers were made for: a chunk's rows, and for the per-pair products a pass of PAIR_TOKENS rows)
        if n == 0 || n > e.chunk() || (!stream && n > PAIR_TOKENS) || k > self.cfg.router.top_k {
            return err(format!(
                "the block on {n} rows of {k} pairs, where a chunk is {} rows of {} and a pass of the per-pair products {PAIR_TOKENS}",
                e.chunk(),
                self.cfg.router.top_k
            ));
        }
        let z = CbBytes::of(n, k);
        holds(&cs.xr, z.xr, "x' in f32", n, k)?;
        holds(&cs.xrb, z.xrb, "x' in bf16", n, k)?;
        holds(&cs.gids, z.gids, "ids", n, k)?;
        route_with(c, &self.router, Some(&self.bias), None, sc, s, h, c.router_hidden, n)?;
        crate::tmark!(s, "router");
        cs.kern.rot_in(s, h, HIDDEN as i32, self.signs.ptr, cs.xr.ptr, cs.xrb.ptr, n)?;
        cs.kern.local(s, sc.ids.ptr, self.local_of.ptr, cs.gids.ptr, pairs as i32, self.experts as i32)?;
        if stream {
            shared_mlp_with(c, Some(&self.shared), &e.lib, sc, s, h, n)?;
            crate::tmark!(s, "shared expert");
        }
        if stream {
            if cs.pre.get().is_none() {
                let _ = cs.pre.set(Cb2Kern::load(&e.dev)?);
            }
            let kern = cs.pre.get().ok_or_else(|| Error("the streaming prefill is not made".into()))?;
            holds(&cs.neg, pairs * 8, "skip ids", n, k)?;
            // (the experts the cache holds are read where they are, and first: a place has the pack's own bytes at its start; a ternary expert is made two-bit rows from a copy, so it is left to be copied)
            let at = self.store.resident_places(self.layer, s)?;
            let descs: Vec<ExpertDesc> = self
                .descs
                .iter()
                .zip(&at)
                .map(|(d, &pl)| {
                    let mut d = *d;
                    if pl >= 0 && d.up != Fmt::Tern && d.down != Fmt::Tern {
                        d.resident = Some((self.places[0].ptr + pl as u64 * self.strides[0] as u64, self.places[1].ptr + pl as u64 * self.strides[1] as u64));
                    }
                    d
                })
                .collect();
            // (x' in f32 is not wanted any more: its buffer takes the sums in the rotated basis)
            cb2_prefill_experts(&e.dev, kern, &descs, n, k, cs.xrb.ptr, cs.gids.ptr, sc.wts.ptr, cs.xr.ptr, s)?;
            cs.kern.sum_rot(s, cs.part.ptr, sc.wts.ptr, cs.xr.ptr, cs.neg.ptr, self.signs.ptr, out, k as i32, 1, n)?;
            return shared_add_with(Some(&self.shared), sc, s, n, out);
        }
        let split = runs_for(pairs);
        holds(&cs.gu, pairs * HIDDEN * 4, "first products", n, k)?;
        holds(&cs.part, pairs * split * HIDDEN * 4, "partial sums", n, k)?;
        let call = Call { ids: cs.gids.ptr, wts: sc.wts.ptr, z: cs.xrb.ptr, way: sc.way.ptr, part: sc.cpu.ptr, tokens: n, pairs };
        let limit = match c.act {
            crate::moe::Act::SwiGlu { limit: Some(l) } => l,
            _ => return err("the experts take a clamped SwiGLU"),
        };
        // (the shared expert is queued by the store once the choices are on their way to the host: it runs while the host plans)
        let mut early = || {
            shared_mlp_with(c, Some(&self.shared), &e.lib, sc, s, h, n)?;
            crate::tmark!(s, "shared expert");
            Ok(())
        };
        self.store.run_early(self.layer, &e.dev, &call, &mut early, &mut |stage, wave, waves, hy| match stage {
            Stage::Products => {
                if !hy.gpu {
                    return Ok(());
                }
                cs.kern.up(
                    s,
                    self.places[0].ptr,
                    self.strides[0] as u64,
                    self.places[1].ptr,
                    self.strides[1] as u64,
                    sc.way.ptr,
                    cs.xr.ptr,
                    cs.gu.ptr,
                    k as i32,
                    pairs,
                    self.mixed,
                )?;
                cs.kern.down(
                    s,
                    self.places[0].ptr,
                    self.strides[0] as u64,
                    self.places[1].ptr,
                    self.strides[1] as u64,
                    sc.way.ptr,
                    cs.gu.ptr,
                    cs.part.ptr,
                    split as i32,
                    limit,
                    pairs,
                    self.mixed,
                )
            }
            Stage::Sum => {
                if wave + 1 == waves {
                    cs.kern.sum_rot(s, cs.part.ptr, sc.wts.ptr, hy.extra, hy.ids, self.signs.ptr, out, k as i32, split as i32, n)?;
                }
                Ok(())
            }
        })?;
        shared_add_with(Some(&self.shared), sc, s, n, out)
    }
}

impl MoeBlock for GlmMoeCb {
    /// The router on `h` (a pruned expert never chosen), x' = R x, the pairs' experts made resident (`ExpertStore::run`: the choices are read on the host, the missing experts copied or computed there, in waves where a
    /// call has more experts than the layer has places), their products, the weighted sum of the pairs with R^T into `out`, the shared expert added. A prompt chunk of more than `CPU_TOKENS` tokens goes through the
    /// streaming prefill whole; without it, in passes of `PAIR_TOKENS`.
    unsafe fn forward(&self, e: &EngineGlm, h: u64, n: usize, out: u64) -> Result<()> {
        let stream = n > CPU_TOKENS && self.common.stream.get();
        if stream || n <= PAIR_TOKENS {
            return self.run_rows(e, h, n, out, stream);
        }
        let mut r = 0;
        while r < n {
            let m = PAIR_TOKENS.min(n - r);
            self.run_rows(e, h + (r * HIDDEN * 2) as u64, m, out + (r * HIDDEN * 2) as u64, false)?;
            r += m;
        }
        Ok(())
    }

    fn segmented(&self) -> bool {
        self.common.topk.get() == 0 && self.store.supports_parts(self.layer)
    }

    unsafe fn prepare(&self, e: &EngineGlm) -> Result<()> {
        self.scratch(e)?;
        // (for the biggest call of any kind, so that no later call makes them again: the graphs' copies name their addresses)
        self.store.prepare_parts(self.layer, &e.dev, CPU_TOKENS, e.chunk() * self.cfg.router.top_k)
    }

    unsafe fn front(&self, e: &EngineGlm, h: u64, n: usize) -> Result<()> {
        let (sc, cs) = self.scratch(e)?;
        let (c, k) = (&self.cfg, self.cfg.router.top_k);
        let s = e.dev.stream;
        route_with(c, &self.router, Some(&self.bias), None, sc, s, h, c.router_hidden, n)?;
        cs.kern.rot_in(s, h, HIDDEN as i32, self.signs.ptr, cs.xr.ptr, cs.xrb.ptr, n)?;
        cs.kern.local(s, sc.ids.ptr, self.local_of.ptr, cs.gids.ptr, (n * k) as i32, self.experts as i32)?;
        let call = Call { ids: cs.gids.ptr, wts: sc.wts.ptr, z: cs.xrb.ptr, way: sc.way.ptr, part: sc.cpu.ptr, tokens: n, pairs: n * k };
        self.store.request_parts(self.layer, &e.dev, &call)
    }

    unsafe fn host(&self, e: &EngineGlm) -> Result<()> {
        let (sc, cs) = self.scratch(e)?;
        let (c, k) = (&self.cfg, self.cfg.router.top_k);
        let (s, n, pairs) = (e.dev.stream, 1usize, k);
        let h = e.h.ptr;
        let split = runs_for(pairs);
        let call = Call { ids: cs.gids.ptr, wts: sc.wts.ptr, z: cs.xrb.ptr, way: sc.way.ptr, part: sc.cpu.ptr, tokens: n, pairs };
        let limit = match c.act {
            crate::moe::Act::SwiGlu { limit: Some(l) } => l,
            _ => return err("the experts take a clamped SwiGLU"),
        };
        // (the shared expert runs while the host plans)
        let mut early = || shared_mlp_with(c, Some(&self.shared), &e.lib, sc, s, h, n);
        self.store.run_parts(self.layer, &e.dev, &call, &mut early, &mut |hy| {
            if !hy.gpu {
                return Ok(());
            }
            cs.kern.up(
                s,
                self.places[0].ptr,
                self.strides[0] as u64,
                self.places[1].ptr,
                self.strides[1] as u64,
                sc.way.ptr,
                cs.xr.ptr,
                cs.gu.ptr,
                k as i32,
                pairs,
                self.mixed,
            )?;
            cs.kern.down(
                s,
                self.places[0].ptr,
                self.strides[0] as u64,
                self.places[1].ptr,
                self.strides[1] as u64,
                sc.way.ptr,
                cs.gu.ptr,
                cs.part.ptr,
                split as i32,
                limit,
                pairs,
                self.mixed,
            )
        })
    }

    unsafe fn back(&self, e: &EngineGlm, out: u64, n: usize) -> Result<()> {
        let (sc, cs) = self.scratch(e)?;
        let (k, s) = (self.cfg.router.top_k, e.dev.stream);
        cs.kern.sum_rot(s, cs.part.ptr, sc.wts.ptr, sc.cpu.ptr, sc.way.ptr, self.signs.ptr, out, k as i32, runs_for(n * k) as i32, n)?;
        shared_add_with(Some(&self.shared), sc, s, n, out)
    }

    fn bytes(&self) -> usize {
        self.router.bytes
            + self.bias.bytes
            + self.shared.up.bytes
            + self.shared.down.bytes
            + self.signs.bytes
            + self.local_of.bytes
            + self.places[0].bytes
            + self.places[1].bytes
    }
}

/// The factory `EngineGlm::new` takes for a model whose routed experts are in `pack`: each mixture-of-experts layer the directory holds is a `GlmMoeCb` with `slots` places on the GPU, all of them sharing `store`; the
/// shared expert and the matrices of the router are read from the checkpoint at the layout the engine chose. `topk`: experts a token (0: the model's).
pub fn factory<'a>(
    pack: &'a CbPack,
    slots: usize,
    topk: usize,
    store: &'a Rc<ExpertStore>,
    common: &'a Rc<CbShared>,
) -> impl Fn(usize, &ConfigGlm, &Checkpoint, Layout) -> Result<Box<dyn MoeBlock>> + 'a {
    move |l, c, ck, level| {
        crate::status::STATUS.beat();
        Ok(Box::new(GlmMoeCb::load(l, c, ck, level, pack, store, slots, topk, common)?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Whatever the rows of a pass and the experts a token takes (the model's top-k or fewer), the first products and the partial sums of the per-pair products fit the buffers made for the chunk and the top-k.
    #[test]
    fn the_per_pair_buffers_fit_every_call() {
        for (chunk, k) in [(1usize, 8usize), (64, 8), (512, 8), (513, 8), (1024, 8), (2048, 8), (4096, 8), (1024, 16)] {
            let z = CbBytes::of(chunk, k);
            for n in 1..=chunk.min(PAIR_TOKENS) {
                for kk in 1..=k {
                    let pairs = n * kk;
                    assert!(pairs * HIDDEN * 4 <= z.gu, "first products: chunk {chunk} top-k {k}, {n} rows of {kk}");
                    assert!(
                        pairs * runs_for(pairs) * HIDDEN * 4 <= z.part,
                        "partial sums: chunk {chunk} top-k {k}, {n} rows of {kk}: {} runs",
                        runs_for(pairs)
                    );
                }
            }
        }
    }

    /// The size the buffers had before (4,096 rows) was too small for 513 rows of 8 pairs: the sizes now follow the pairs.
    #[test]
    fn a_pass_of_a_thousand_tokens_has_room_for_its_pairs() {
        assert!(CbBytes::of(1024, 8).part >= 8192 * HIDDEN * 4);
        assert!(CbBytes::of(2048, 8).part >= 8192 * HIDDEN * 4);
        assert_eq!(CbBytes::of(256, 8).part, PART_ROWS * HIDDEN * 4);
    }
}
