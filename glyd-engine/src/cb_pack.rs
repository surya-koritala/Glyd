//! The codes pack (glyd-hbr-codes/2: a model's routed experts as rows of codebook codes in a rotated basis, written by research/glm-data/glm_pack.py; kernels/glm_cb.cu has the arithmetic)
//! as Condor's engine reads it (src/moe_qwen4x.rs, src/qwen4x_serve.rs): its `meta.json`, each layer's file mapped into memory as it is (the pack's tensors are two runs of bytes there,
//! the codebooks of the kept experts and then their codes, an expert's downT before its up; unpinned page cache, which the system takes back when it needs the memory), and where each
//! kept expert's bytes are (`LayerMap`) for the expert cache's places and the streaming prefill. GLM-5.3-Flash's own block over these packs went with its development engine's serving
//! (R13, #234): the executor tiers GLM's experts by hummingbird and the block levels (R12).

use crate::cb2_prefill::ExpertDesc;
use crate::cpu_cb::Fmt;
use crate::ffi::{err, Error, Result};
use crate::mapped::Mapping;
use crate::offload::Placement;
use glyd_gpu::json::{self, Value};
use std::path::{Path, PathBuf};
use std::sync::Arc;

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
    /// where each kept expert's codes and codebooks are in the file, their lengths and formats (None: all two-bit and end to end from `starts`, `uniform_bytes` each)
    pub place: Option<Placement>,
    /// the strides of the places on the GPU (codes, codebooks)
    pub strides: [usize; 2],
    /// where the codes and the codebooks of the first kept expert are in the file (a layer that is all two-bit; else 0)
    pub starts: (usize, usize),
    /// the kept experts as the streaming prefill takes them (addresses in the mapping)
    pub descs: Vec<ExpertDesc>,
}

impl CbPack {
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
            layers.push((l, LayerMeta { file, signs, kept, fmts, side: None }));
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

    /// The bytes of the layers' files together.
    pub fn file_bytes(&self) -> u64 {
        self.layers.iter().filter_map(|(_, m)| std::fs::metadata(self.dir.join(&m.file)).ok()).map(|m| m.len()).sum()
    }

    /// The bytes of a place of layer `l` on the GPU (codes and codebooks together): the biggest of its experts' (two-bit experts alone: `uniform_bytes`), and a byte for the formats where the layer is not all two-bit.
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

    /// The bytes of a two-bit expert's codes and codebooks (its up's and downT's rows of 2-bit codes, four bf16 entries of a codebook a row).
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
    /// expert's downT then its up, `uniform_bytes` each. A layer of other formats, or of mixed ones, is placed expert by expert from the file's header: the offset and length of each expert's codes (its downT's
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
