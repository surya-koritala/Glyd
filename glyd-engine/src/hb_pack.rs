//! `glyd-engine pack MODEL OUT --layout hummingbird --usage FILE`: a mixture-of-experts model at the hummingbird level for every family the
//! description builder maps (`runtime::describe`: Kimi-K3, DeepSeek-V4, GLM-5.3 and -Flash, Muse Spark, Qwen3.5/3.8 MoE), with no export
//! made apart (docs/design/hummingbird.md, #429). Each MoE layer's routed experts are read from the release exactly (bf16, FP8 with its
//! block scales, FP4 or MXFP4 with their E8M0 scales), given their bits by use (the corpus pass's counts, `FILE`; the recipe hb40), rotated
//! and rounded row by row (`hb_codec`), and written as codebook rows; every other tensor of the release is copied byte for byte, so the
//! family's loader reads the rest as it reads the release.
//!
//! The directory: `glyd.json` (format `glyd-hbg/1`, level hummingbird: not a glyd-v7 directory, whose readers take every matrix as packed at a
//! level; here the rest is the release's, read as the release is; an `experts` record: the format, the recipe, the usage file's
//! tokens and hash, each layer's block, file, width, rotation block, sizes, experts' bits, bytes and sha256); `experts-BBBB.safetensors`
//! for the MoE layer at the description's block BBBB, holding `glyd_rot_signs` (I8 [W]) and for each expert e and matrix m (`gate_up`,
//! `down_t`) `eEEEE.m.codes` (U8 [rows, W bits / 8]), `eEEEE.m.scales` (BF16 [rows, W / 32]) and `eEEEE.m.codebook` (BF16 [rows, 2^bits]);
//! `model-NNNNN-of-MMMMM.safetensors` and `model.safetensors.index.json`, the release's other tensors; and the release's other files
//! (config, tokenizer, templates).

use crate::ffi::{err, Error, Result};
use crate::hb_codec::{self, bits_by_use, block_of, code_expert, signs, ExpertRows, LayerRows, Usage, HB40_PERCENT};
use crate::runtime::describe::{describe, Described, Source as Src};
use crate::runtime::kinds::{Body, Ffn};
use glyd_gpu::codebook::{cb_strides, CbRows};
use glyd_gpu::json::{self, Value};
use glyd_gpu::safetensors::{Checkpoint, File, TensorInfo};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::time::Instant;

/// glyd.json's `experts.format`.
pub const FORMAT: &str = "glyd-hbg/1";
/// The largest shard of the copied tensors.
const SHARD_BYTES: u64 = 5 << 30;

fn io<T>(r: std::io::Result<T>) -> Result<T> {
    r.map_err(|e| Error(e.to_string()))
}

/// Where a MoE layer's experts are in the release: one entry an expert (its tensors stacked by rows), or one tensor [experts, rows, cols].
#[derive(Clone, Debug, PartialEq)]
pub enum Experts {
    Each(Vec<Vec<String>>),
    Stacked(String),
}

/// A MoE layer of a description: its block, whether its experts are gated, how many, and where their two matrices are.
#[derive(Clone, Debug, PartialEq)]
pub struct MoeLayer {
    pub block: usize,
    pub gated: bool,
    pub experts: usize,
    pub gate_up: Experts,
    pub down: Experts,
}

fn experts_of(src: &Src, n: usize, what: &str) -> Result<Experts> {
    match src {
        Src::Experts(v) if v.len() == n && v.iter().all(|e| !e.is_empty()) => Ok(Experts::Each(v.clone())),
        Src::Tensor(t) => Ok(Experts::Stacked(t.clone())),
        _ => err(format!("{what}: not one source an expert for {n} experts")),
    }
}

/// The MoE layers of a description, in order.
pub fn moe_layers(d: &Described) -> Result<Vec<MoeLayer>> {
    let mut out = Vec::new();
    for (b, block) in d.desc.blocks.iter().enumerate() {
        let Body::Ffn(Ffn::Moe(m)) = &block.body else { continue };
        let role = |name: &str| d.blocks[b].iter().find(|(r, _)| r == name).map(|(_, s)| s);
        let first = if m.expert.gated { "experts_gate_up" } else { "experts_up" };
        let gu = role(first).ok_or_else(|| Error(format!("block {b}: no {first} in the weight map")))?;
        let dn = role("experts_down").ok_or_else(|| Error(format!("block {b}: no experts_down in the weight map")))?;
        out.push(MoeLayer {
            block: b,
            gated: m.expert.gated,
            experts: m.experts,
            gate_up: experts_of(gu, m.experts, &format!("block {b}'s {first}"))?,
            down: experts_of(dn, m.experts, &format!("block {b}'s experts_down"))?,
        });
    }
    Ok(out)
}

fn le_u16(b: &[u8]) -> Vec<u16> {
    b.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect()
}

/// E2M1's values (the low three bits the magnitude, the fourth the sign).
const E2M1: [f32; 16] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0];

/// FP4 codes [rows, cols] two a byte (the first in the low four bits) times their run's E8M0 scale (a byte a run of 32 of a row), exactly.
pub fn fp4_f32(q: &[u8], scales: &[u8], rows: usize, cols: usize) -> Result<Vec<f32>> {
    if !cols.is_multiple_of(32) || q.len() * 2 != rows * cols || scales.len() * 32 != rows * cols {
        return err(format!("{} bytes and {} scales for an FP4 matrix of {rows} x {cols}", q.len(), scales.len()));
    }
    let mut out = Vec::with_capacity(rows * cols);
    for (i, &b) in q.iter().enumerate() {
        let s = crate::formats::e8m0(scales[i * 2 / 32]);
        if s.is_nan() {
            return err(format!("an FP4 run's scale is NaN (run {})", i * 2 / 32));
        }
        out.push(E2M1[(b & 15) as usize] * s);
        out.push(E2M1[(b >> 4) as usize] * s);
    }
    Ok(out)
}

/// FP8 (E4M3) [rows, cols] times an E8M0 scale a block of 128 x 128 (DeepSeek-V4's `.scale`), exactly.
fn fp8_e8m0_f32(q: &[u8], scales: &[u8], rows: usize, cols: usize) -> Result<Vec<f32>> {
    let (gr, gc) = (rows.div_ceil(128), cols.div_ceil(128));
    if q.len() != rows * cols || scales.len() != gr * gc {
        return err(format!("{} bytes and {} scales for an FP8 matrix of {rows} x {cols} in blocks of 128", q.len(), scales.len()));
    }
    Ok((0..rows * cols).map(|i| glyd_gpu::fp8::e4m3(q[i]) * crate::formats::e8m0(scales[(i / cols / 128) * gc + (i % cols) / 128])).collect())
}

/// A release matrix as f32 [rows, cols]: BF16 or F32 as it is; F8_E4M3 with its `_scale_inv` blocks (as transformers dequantizes it, to
/// bf16) or with an E8M0 scale a block (`scales`' entry); FP4 two a byte with an E8M0 scale a run of 32 (`scales`' entry); or, where the
/// tensor is not there, compressed-tensors' MXFP4 (`.weight_packed`, `.weight_scale`).
pub fn read_f32(ck: &Checkpoint, name: &str, scales: &BTreeMap<String, String>) -> Result<(Vec<f32>, usize, usize)> {
    let Some((_, t)) = ck.get(name) else {
        let base = name.strip_suffix(".weight").ok_or_else(|| Error(format!("no tensor {name}")))?;
        let (pk, sk) = (format!("{base}.weight_packed"), format!("{base}.weight_scale"));
        let (_, t) = ck.get(&pk).ok_or_else(|| Error(format!("no tensor {name}, nor {pk}")))?;
        let (rows, cols) = dims(t, 2, &pk)?;
        let v = fp4_f32(&io(ck.read(&pk))?, &io(ck.read(&sk))?, rows, cols)?;
        return Ok((v, rows, cols));
    };
    match (t.dtype.as_str(), scales.get(name)) {
        ("BF16", _) | ("F8_E4M3", None) => {
            let (rows, cols) = dims(t, 1, name)?;
            let mut w = vec![0u16; rows * cols];
            io(ck.read_bf16(name, &mut w))?;
            Ok((w.into_iter().map(hb_codec::f32_of).collect(), rows, cols))
        }
        ("F8_E4M3", Some(s)) if ck.get(s).is_some_and(|(_, st)| st.dtype != "F32") => {
            let (rows, cols) = dims(t, 1, name)?;
            Ok((fp8_e8m0_f32(&io(ck.read(name))?, &io(ck.read(s))?, rows, cols)?, rows, cols))
        }
        ("F8_E4M3", Some(_)) => {
            let (rows, cols) = dims(t, 1, name)?;
            let mut w = vec![0u16; rows * cols];
            io(ck.read_bf16(name, &mut w))?;
            Ok((w.into_iter().map(hb_codec::f32_of).collect(), rows, cols))
        }
        ("F32", _) => {
            let (rows, cols) = dims(t, 1, name)?;
            Ok((io(ck.read(name))?.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect(), rows, cols))
        }
        ("I8" | "U8", Some(s)) => {
            let (rows, cols) = dims(t, 2, name)?;
            Ok((fp4_f32(&io(ck.read(name))?, &io(ck.read(s))?, rows, cols)?, rows, cols))
        }
        (d, _) => err(format!("{name}: {d}, where BF16, F32, FP8 with its scales or FP4 with its scales is wanted")),
    }
}

/// A matrix's rows and columns from its shape (`per`: values a column of the stored tensor holds, 2 for FP4 two a byte).
fn dims(t: &TensorInfo, per: usize, name: &str) -> Result<(usize, usize)> {
    match t.shape[..] {
        [r, c] if r > 0 && c > 0 => Ok((r as usize, c as usize * per)),
        _ => err(format!("{name}: shape {:?}, where a matrix is wanted", t.shape)),
    }
}

/// Expert `e`'s rows of a stacked tensor [experts, rows, cols] (BF16 or F32) as f32.
fn stacked_f32(ck: &Checkpoint, name: &str, e: usize) -> Result<(Vec<f32>, usize, usize)> {
    let (f, t): (&File, &TensorInfo) = ck.get(name).ok_or_else(|| Error(format!("no tensor {name}")))?;
    let [n, rows, cols] = t.shape[..] else { return err(format!("{name}: shape {:?}, where [experts, rows, cols] is wanted", t.shape)) };
    let size: u64 = match t.dtype.as_str() {
        "BF16" => 2,
        "F32" => 4,
        d => return err(format!("{name}: {d}, where stacked experts are BF16 or F32")),
    };
    if e as u64 >= n {
        return err(format!("{name}: expert {e} of {n}"));
    }
    let bytes = rows * cols * size;
    let sub = TensorInfo { dtype: t.dtype.clone(), shape: vec![rows, cols], offset: t.offset + e as u64 * bytes, bytes };
    let raw = io(f.read(&sub))?;
    let v = if size == 2 {
        le_u16(&raw).into_iter().map(hb_codec::f32_of).collect()
    } else {
        raw.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect()
    };
    Ok((v, rows as usize, cols as usize))
}

/// Expert `e`'s matrix of `x`: its tensors stacked by rows.
fn expert_f32(ck: &Checkpoint, x: &Experts, e: usize, scales: &BTreeMap<String, String>) -> Result<(Vec<f32>, usize, usize)> {
    match x {
        Experts::Stacked(n) => stacked_f32(ck, n, e),
        Experts::Each(v) => {
            let (mut out, mut rows, mut cols) = (Vec::new(), 0, 0);
            for n in &v[e] {
                let (w, r, c) = read_f32(ck, n, scales)?;
                if cols != 0 && c != cols {
                    return err(format!("{n}: {c} columns, where the expert's other rows have {cols}"));
                }
                out.extend(w);
                (rows, cols) = (rows + r, c);
            }
            Ok((out, rows, cols))
        }
    }
}

/// Every release tensor the experts of `layers` are (with their scales, and MXFP4's packed names): what the copy leaves out.
fn expert_tensors(ck: &Checkpoint, layers: &[MoeLayer], scales: &BTreeMap<String, String>) -> BTreeSet<String> {
    let mut modules = BTreeSet::new();
    let mut whole = BTreeSet::new();
    for l in layers {
        for x in [&l.gate_up, &l.down] {
            match x {
                Experts::Stacked(n) => {
                    whole.insert(n.clone());
                }
                Experts::Each(v) => {
                    for n in v.iter().flatten() {
                        whole.insert(n.clone());
                        if let Some(s) = scales.get(n) {
                            whole.insert(s.clone());
                        }
                        modules.insert(n.strip_suffix(".weight").unwrap_or(n).to_string() + ".");
                    }
                }
            }
        }
    }
    ck.names()
        .filter(|n| whole.contains(*n) || modules.range(..=n.to_string()).next_back().is_some_and(|m| n.starts_with(m.as_str())))
        .map(str::to_string)
        .collect()
}

/// A layer's experts coded on `threads` threads: each expert's two matrices read, rotated by the layer's signs and rounded at its bits.
fn code_layer(ck: &Checkpoint, l: &MoeLayer, scales: &BTreeMap<String, String>, bits: &[u32], threads: usize) -> Result<LayerRows> {
    let (gu0, u, w) = expert_f32(ck, &l.gate_up, 0, scales)?;
    let (_, w2, inter) = expert_f32(ck, &l.down, 0, scales)?;
    drop(gu0);
    if w2 != w || u != if l.gated { 2 * inter } else { inter } {
        return err(format!(
            "block {}: experts of {u} x {w} and {w2} x {inter}, where a{} expert of width {w} and {inter} neurons is {} x {w} and {w} x {inter}",
            l.block,
            if l.gated { " gated" } else { "n ungated" },
            if l.gated { 2 * inter } else { inter }
        ));
    }
    let rot = block_of(w)
        .ok_or_else(|| Error(format!("block {}: experts of width {w}: hummingbird's rotation takes widths in runs of {}", l.block, hb_codec::MIN_BLOCK)))?;
    let s = signs(l.block, w);
    let next = std::sync::atomic::AtomicUsize::new(0);
    let done: std::sync::Mutex<Vec<Option<ExpertRows>>> = std::sync::Mutex::new(vec![None; l.experts]);
    let failed: std::sync::Mutex<Option<Error>> = std::sync::Mutex::new(None);
    std::thread::scope(|sc| {
        for _ in 0..threads.clamp(1, l.experts) {
            sc.spawn(|| loop {
                let e = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if e >= l.experts || failed.lock().is_ok_and(|f| f.is_some()) {
                    return;
                }
                let r = (|| -> Result<ExpertRows> {
                    let (gu, _, _) = expert_f32(ck, &l.gate_up, e, scales)?;
                    let (dn, _, _) = expert_f32(ck, &l.down, e, scales)?;
                    Ok(code_expert(&gu, &dn, w, inter, &s, rot, bits[e], None)?.0)
                })();
                match r {
                    Ok(x) => done.lock().expect("a coder panicked")[e] = Some(x),
                    Err(x) => {
                        *failed.lock().expect("a coder panicked") = Some(Error(format!("block {} expert {e}: {}", l.block, x.msg)));
                        return;
                    }
                }
            });
        }
    });
    if let Some(e) = failed.into_inner().expect("a coder panicked") {
        return Err(e);
    }
    let experts = done.into_inner().expect("a coder panicked").into_iter().map(|x| x.expect("every expert coded")).collect();
    Ok(LayerRows { block: l.block, width: w, rot, inter, gated: l.gated, signs: s, experts })
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

fn sha256_file(p: &Path) -> Result<String> {
    use std::io::Read;
    let mut f = io(std::fs::File::open(p))?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 22];
    loop {
        let n = io(f.read(&mut buf))?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(hex(&h.finalize()))
}

fn layer_file(block: usize) -> String {
    format!("experts-{block:04}.safetensors")
}

/// A layer's experts written to `dir`: its file's name, bytes and sha256.
fn write_layer(dir: &Path, l: &LayerRows) -> Result<(String, u64, String)> {
    let name = layer_file(l.block);
    let signs: Vec<u8> = l.signs.iter().map(|&s| s as u8).collect();
    let mut owned: Vec<(String, &str, Vec<u64>, &[u8])> = vec![("glyd_rot_signs".into(), "I8", vec![l.width as u64], &signs[..])];
    for (e, x) in l.experts.iter().enumerate() {
        for (m, p) in [("gate_up", &x.gate_up), ("down_t", &x.down_t)] {
            let (cs, _, _) = cb_strides(p.bits, p.cols);
            owned.push((format!("e{e:04}.{m}.codes"), "U8", vec![p.rows as u64, cs as u64], &p.codes[..]));
            owned.push((format!("e{e:04}.{m}.scales"), "BF16", vec![p.rows as u64, (p.cols / 32) as u64], &p.scales[..]));
            owned.push((format!("e{e:04}.{m}.codebook"), "BF16", vec![p.rows as u64, 1 << p.bits], &p.codebook[..]));
        }
    }
    let tensors: Vec<(&str, &str, Vec<u64>, &[u8])> = owned.iter().map(|(n, d, s, b)| (n.as_str(), *d, s.clone(), *b)).collect();
    let path = dir.join(&name);
    crate::pack::write_safetensors(&path, &[("format", "pt".to_string()), ("glyd", FORMAT.to_string()), ("block", l.block.to_string())], &tensors)?;
    let bytes = io(std::fs::metadata(&path))?.len();
    Ok((name, bytes, sha256_file(&path)?))
}

/// The release's tensors but `skip` copied byte for byte into `dir`'s shards, with an index. Returns the tensors and bytes copied.
fn copy_rest(ck: &Checkpoint, dir: &Path, skip: &BTreeSet<String>) -> Result<(usize, u64)> {
    let names: Vec<String> = ck.names().filter(|n| !skip.contains(*n)).map(str::to_string).collect();
    let mut shards: Vec<Vec<String>> = vec![Vec::new()];
    let mut size = 0u64;
    for n in &names {
        let b = ck.get(n).expect("a name of the checkpoint").1.bytes;
        if size > 0 && size + b > SHARD_BYTES {
            shards.push(Vec::new());
            size = 0;
        }
        shards.last_mut().expect("a shard").push(n.clone());
        size += b;
    }
    let total = shards.len();
    let mut map = Vec::new();
    let mut bytes = 0u64;
    for (i, s) in shards.iter().enumerate() {
        let file = format!("model-{:05}-of-{total:05}.safetensors", i + 1);
        let data: Vec<(String, String, Vec<u64>, Vec<u8>)> = s
            .iter()
            .map(|n| {
                let (_, t) = ck.get(n).expect("a name of the checkpoint");
                Ok((n.clone(), t.dtype.clone(), t.shape.clone(), io(ck.read(n))?))
            })
            .collect::<Result<_>>()?;
        let tensors: Vec<(&str, &str, Vec<u64>, &[u8])> = data.iter().map(|(n, d, sh, b)| (n.as_str(), d.as_str(), sh.clone(), &b[..])).collect();
        crate::pack::write_safetensors(&dir.join(&file), &[("format", "pt".to_string())], &tensors)?;
        bytes += data.iter().map(|d| d.3.len() as u64).sum::<u64>();
        map.extend(s.iter().map(|n| (n.clone(), Value::from(file.as_str()))));
    }
    let index =
        Value::Object(vec![("metadata".into(), Value::Object(vec![("total_size".into(), Value::from(bytes))])), ("weight_map".into(), Value::Object(map))]);
    io(std::fs::write(dir.join("model.safetensors.index.json"), json::to_compact(&index)))?;
    Ok((names.len(), bytes))
}

/// The release's other files (config, tokenizer, templates: every file of its directory that is no safetensors and no index) copied.
fn copy_files(src: &Path, dir: &Path) -> Result<usize> {
    let mut n = 0;
    for e in io(std::fs::read_dir(src))? {
        let p = io(e)?.path();
        let name = p.file_name().and_then(|s| s.to_str()).unwrap_or("").to_string();
        if !p.is_file() || name.ends_with(".safetensors") || name == "model.safetensors.index.json" || name == "glyd.json" {
            continue;
        }
        io(std::fs::copy(&p, dir.join(&name)))?;
        n += 1;
    }
    Ok(n)
}

/// `model` (a mixture of experts the description builder maps, as released) at the hummingbird level into `out`, each expert's bits from
/// `usage` (the corpus pass's counts), on `threads` threads. A directory already holding a pack is left as it is.
pub fn run(model: &str, out: &Path, threads: usize, usage: &Path) -> Result<()> {
    let t0 = Instant::now();
    let source = io(glyd_gpu::save::Source::find(model))?;
    if out.join("glyd.json").exists() {
        eprintln!("{}: already packed, keeping its files", out.display());
        return Ok(());
    }
    let text = io(std::fs::read_to_string(source.dir.join("config.json")))?;
    let cfg: serde_json::Value = serde_json::from_str(&text).map_err(|e| Error(format!("config.json: {e}")))?;
    let d = describe(&cfg)?;
    let layers = moe_layers(&d)?;
    if layers.is_empty() {
        return err(format!("{model}: no mixture of experts: the hummingbird level is for the experts of one"));
    }
    let use_text = io(std::fs::read_to_string(usage))?;
    let used = Usage::parse(&use_text)?;
    for l in &layers {
        match used.layer(l.block) {
            Some(u) if u.counts.len() == l.experts => {}
            Some(u) => return err(format!("{}: block {} has {} counts, where the layer has {} experts", usage.display(), l.block, u.counts.len(), l.experts)),
            None => {
                return err(format!(
                    "{}: no counts for the MoE layer at block {} (the corpus pass of this model: glyd-engine calibrate-experts)",
                    usage.display(),
                    l.block
                ))
            }
        }
    }
    let ck = io(Checkpoint::open(&source.dir))?;
    let tmp = out.with_extension("partial");
    let _ = std::fs::remove_dir_all(&tmp);
    io(std::fs::create_dir_all(&tmp))?;
    let mut recs = Vec::new();
    let (mut twos, mut threes) = (0, 0);
    for l in &layers {
        let bits = bits_by_use(&used.layer(l.block).expect("checked").counts, HB40_PERCENT);
        let rows = code_layer(&ck, l, &d.scales, &bits, threads)?;
        let (file, bytes, sha) = write_layer(&tmp, &rows)?;
        twos += bits.iter().filter(|&&b| b == 2).count();
        threes += bits.iter().filter(|&&b| b == 3).count();
        recs.push(Value::Object(vec![
            ("block".into(), Value::from(l.block as u64)),
            ("file".into(), Value::from(file.as_str())),
            ("width".into(), Value::from(rows.width as u64)),
            ("rot".into(), Value::from(rows.rot as u64)),
            ("inter".into(), Value::from(rows.inter as u64)),
            ("gated".into(), Value::Bool(rows.gated)),
            ("expert_bits".into(), Value::Array(bits.iter().map(|&b| Value::from(b as u64)).collect())),
            ("bytes".into(), Value::from(bytes)),
            ("sha256".into(), Value::from(sha.as_str())),
        ]));
        eprintln!("block {}: {} experts of {} x {} coded ({:.0} s)", l.block, l.experts, rows.width, rows.inter, t0.elapsed().as_secs_f64());
    }
    let skip = expert_tensors(&ck, &layers, &d.scales);
    let (copied, copied_bytes) = copy_rest(&ck, &tmp, &skip)?;
    let files = copy_files(&source.dir, &tmp)?;
    let manifest = Value::Object(vec![
        ("format".into(), Value::from(FORMAT)),
        ("level".into(), Value::from("hummingbird")),
        (
            "source".into(),
            Value::Object(vec![
                ("repo".into(), Value::from(source.repo.as_str())),
                ("revision".into(), source.revision.as_deref().map_or(Value::Null, Value::from)),
            ]),
        ),
        (
            "experts".into(),
            Value::Object(vec![
                ("format".into(), Value::from(FORMAT)),
                ("recipe".into(), Value::from("hb40")),
                ("rounding".into(), Value::from("nearest")),
                (
                    "usage".into(),
                    Value::Object(vec![
                        ("tokens".into(), Value::from(used.tokens)),
                        ("sha256".into(), Value::from(hex(&Sha256::digest(use_text.as_bytes())).as_str())),
                    ]),
                ),
                ("layers".into(), Value::Array(recs)),
            ]),
        ),
    ]);
    io(std::fs::write(tmp.join("glyd.json"), json::to_compact(&manifest)))?;
    if out.exists() {
        io(std::fs::remove_dir(out)).map_err(|e| Error(format!("{}: {} (it should be empty or absent)", out.display(), e.msg)))?;
    }
    io(std::fs::rename(&tmp, out))?;
    eprintln!(
        "{FORMAT} at the hummingbird level: {} MoE layers ({twos} experts at 2 bits, {threes} at 3), {copied} other tensors copied ({:.2} GB), {files} files ({:.1} s)",
        layers.len(),
        copied_bytes as f64 / 1e9,
        t0.elapsed().as_secs_f64()
    );
    Ok(())
}

/// A hummingbird directory of this format opened: its MoE layers' records.
pub struct HbDir {
    pub dir: PathBuf,
    pub layers: Vec<LayerRec>,
}

/// A MoE layer's record in glyd.json.
#[derive(Clone, Debug, PartialEq)]
pub struct LayerRec {
    pub block: usize,
    pub file: String,
    pub width: usize,
    pub rot: usize,
    pub inter: usize,
    pub gated: bool,
    pub bits: Vec<u32>,
}

impl HbDir {
    /// `dir` where it is a hummingbird directory of this format, else None.
    pub fn open(dir: &Path) -> Result<Option<HbDir>> {
        let Ok(text) = std::fs::read_to_string(dir.join("glyd.json")) else { return Ok(None) };
        let m = json::parse(&text).map_err(|e| Error(format!("{}: glyd.json: {e}", dir.display())))?;
        if m.get("format").and_then(Value::as_str) != Some(FORMAT) {
            return Ok(None);
        }
        let x = m
            .get("experts")
            .filter(|x| x.get("format").and_then(Value::as_str) == Some(FORMAT))
            .ok_or_else(|| Error(format!("{}: glyd.json: no experts record", dir.display())))?;
        let what = |why: String| Error(format!("{}: glyd.json: {why}", dir.display()));
        let mut layers = Vec::new();
        for (i, r) in x.get("layers").and_then(Value::as_array).ok_or_else(|| what("no experts.layers".into()))?.iter().enumerate() {
            let n = |k: &str| r.get(k).and_then(Value::as_u64).map(|v| v as usize).ok_or_else(|| what(format!("layer {i}: no {k}")));
            let bits: Vec<u32> = r
                .get("expert_bits")
                .and_then(Value::as_u64s)
                .filter(|b| b.iter().all(|&x| x == 2 || x == 3))
                .ok_or_else(|| what(format!("layer {i}: no expert_bits of 2 and 3")))?
                .into_iter()
                .map(|b| b as u32)
                .collect();
            let file = r
                .get("file")
                .and_then(Value::as_str)
                .filter(|f| !f.is_empty() && !f.contains('/') && *f != "..")
                .ok_or_else(|| what(format!("layer {i}: no file")))?;
            let (width, rot) = (n("width")?, n("rot")?);
            if block_of(width) != Some(rot) {
                return Err(what(format!("layer {i}: a rotation of {rot} for a width of {width}")));
            }
            layers.push(LayerRec {
                block: n("block")?,
                file: file.to_string(),
                width,
                rot,
                inter: n("inter")?,
                gated: r.get("gated").and_then(Value::as_bool).unwrap_or(true),
                bits,
            });
        }
        Ok(Some(HbDir { dir: dir.to_path_buf(), layers }))
    }

    /// The record of the MoE layer at block `block`.
    pub fn layer(&self, block: usize) -> Option<&LayerRec> {
        self.layers.iter().find(|l| l.block == block)
    }

    /// The signs of the MoE layer at block `block`'s rotation (its file's `glyd_rot_signs`), held to the record.
    pub fn read_signs(&self, block: usize) -> Result<Vec<i8>> {
        let r = self.layer(block).ok_or_else(|| Error(format!("{}: no experts for block {block}", self.dir.display())))?;
        let f = io(File::open(&self.dir.join(&r.file)))?;
        let t = f.get("glyd_rot_signs").ok_or_else(|| Error(format!("{}: no glyd_rot_signs", r.file)))?;
        let signs: Vec<i8> = io(f.read(t))?.into_iter().map(|b| b as i8).collect();
        if signs.len() != r.width || signs.iter().any(|&s| s != 1 && s != -1) {
            return err(format!("{}: glyd_rot_signs are not {} signs", r.file, r.width));
        }
        Ok(signs)
    }

    /// The MoE layer at block `block` read whole: its signs and each expert's two matrices, held to the record.
    pub fn read_layer(&self, block: usize) -> Result<LayerRows> {
        let r = self.layer(block).ok_or_else(|| Error(format!("{}: no experts for block {block}", self.dir.display())))?;
        let f = io(File::open(&self.dir.join(&r.file)))?;
        let read = |name: &str| -> Result<Vec<u8>> {
            let t = f.get(name).ok_or_else(|| Error(format!("{}: no {name}", r.file)))?;
            io(f.read(t))
        };
        let signs: Vec<i8> = read("glyd_rot_signs")?.into_iter().map(|b| b as i8).collect();
        if signs.len() != r.width || signs.iter().any(|&s| s != 1 && s != -1) {
            return err(format!("{}: glyd_rot_signs are not {} signs", r.file, r.width));
        }
        let up_rows = if r.gated { 2 * r.inter } else { r.inter };
        let mut experts = Vec::with_capacity(r.bits.len());
        for (e, &bits) in r.bits.iter().enumerate() {
            let m = |name: &str, rows: usize| -> Result<CbRows> {
                let p = |k: &str| read(&format!("e{e:04}.{name}.{k}"));
                CbRows::new(bits, rows, r.width, p("codes")?, p("scales")?, p("codebook")?)
                    .ok_or_else(|| Error(format!("{}: expert {e}'s {name} is not {rows} rows of {} at {bits} bits", r.file, r.width)))
            };
            experts.push(ExpertRows { bits, gate_up: m("gate_up", up_rows)?, down_t: m("down_t", r.inter)? });
        }
        Ok(LayerRows { block, width: r.width, rot: r.rot, inter: r.inter, gated: r.gated, signs, experts })
    }

    /// The routed experts of decoder layer `l` (the description's block 2 l + 1) on the GPU, for `moe::MoeLayer::new_hb`; None where the
    /// directory has no experts of that layer. A family's loader asks this of a hummingbird directory in place of the release's experts.
    pub fn upload_layer(&self, l: usize) -> Result<Option<crate::moe_hbg::HbgExperts>> {
        if self.layer(2 * l + 1).is_none() {
            return Ok(None);
        }
        crate::moe_hbg::HbgExperts::upload(&self.read_layer(2 * l + 1)?).map(Some)
    }

    /// The directory's experts for decoder layer `l` where `hb` is a hummingbird directory, refused where it is one and the layer is
    /// missing from it (a model's every MoE layer is there), and where the GPUs share each layer's rows (every layer across them).
    pub fn experts_for(hb: Option<&HbDir>, l: usize, across: bool) -> Result<Option<crate::moe_hbg::HbgExperts>> {
        let Some(d) = hb else { return Ok(None) };
        if across {
            return err("a hummingbird directory's experts are whole on a GPU: every layer across GPUs takes the penguin level so far (cut the layers over the GPUs: --split)");
        }
        match d.upload_layer(l)? {
            Some(x) => Ok(Some(x)),
            None => err(format!("{}: no experts for decoder layer {l} (block {})", d.dir.display(), 2 * l + 1)),
        }
    }

    /// Bytes of a layer's routed experts at their bits (codes, scales and codebooks).
    pub fn bytes_of(r: &LayerRec) -> u64 {
        let up = if r.gated { 2 * r.inter } else { r.inter };
        r.bits
            .iter()
            .map(|&b| {
                let (c, s, k) = cb_strides(b, r.width);
                ((up + r.inter) * (c + s + k)) as u64
            })
            .sum()
    }

    /// Bytes of the routed experts at their bits.
    pub fn experts_bytes(&self) -> u64 {
        self.layers.iter().map(HbDir::bytes_of).sum()
    }

    /// Each MoE layer's experts' bytes by its decoder layer (the description's block over 2: a decoder layer is the mixer's block, then the
    /// feed-forward's).
    pub fn layer_bytes(&self) -> Vec<(usize, u64)> {
        self.layers.iter().map(|r| (r.block / 2, HbDir::bytes_of(r))).collect()
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::runtime::reference::testing::Fixture;

    fn bf16_bytes(v: &[f32]) -> Vec<u8> {
        v.iter().flat_map(|&x| hb_codec::bf16(x).to_le_bytes()).collect()
    }

    /// A tiny release from a transformers fixture (`tests/runtime/NAME.json`): its config, and every tensor of its weight map in bf16
    /// with the fixture's values. Returns the directory and the description.
    pub(crate) fn release(dir: &Path, fixture: &str) -> Described {
        let _ = std::fs::remove_dir_all(dir);
        std::fs::create_dir_all(dir).unwrap();
        let f = Fixture::load(fixture);
        std::fs::write(dir.join("config.json"), serde_json::to_string(&f.j["config"]).unwrap()).unwrap();
        std::fs::write(dir.join("tokenizer_config.json"), "{}").unwrap();
        let d = describe(&f.j["config"]).unwrap();
        let owned: Vec<(String, Vec<u64>, Vec<u8>)> = d
            .names()
            .into_iter()
            .map(|n| {
                let (shape, v) = f.tensor(&n);
                (n, shape.iter().map(|&s| s as u64).collect(), bf16_bytes(&v))
            })
            .collect();
        let tensors: Vec<(&str, &str, Vec<u64>, &[u8])> = owned.iter().map(|(n, s, b)| (n.as_str(), "BF16", s.clone(), &b[..])).collect();
        crate::pack::write_safetensors(&dir.join("model.safetensors"), &[("format", "pt".to_string())], &tensors).unwrap();
        d
    }

    fn usage_for(d: &Described, seed: u64) -> Usage {
        let mut x = seed | 1;
        Usage {
            model: "tiny".into(),
            tokens: 4096,
            layers: moe_layers(d)
                .unwrap()
                .iter()
                .map(|l| hb_codec::LayerUse {
                    block: l.block,
                    counts: (0..l.experts)
                        .map(|_| {
                            x ^= x << 13;
                            x ^= x >> 7;
                            x ^= x << 17;
                            x % 1000
                        })
                        .collect(),
                })
                .collect(),
        }
    }

    /// A hummingbird directory as the engine's loaders see it: no glyd-v7 directory (the rest is read as the release's), its index holds the
    /// copied tensors, the dynamic route sizes each decoder layer with its experts' bytes, and the experts a loader asks for are refused where
    /// a layer is missing or the GPUs would share each layer's rows.
    #[test]
    fn the_loaders_read_a_hummingbird_directory_as_the_release_with_its_experts() {
        let dir = std::env::temp_dir().join(format!("glyd-hbdir-{}", std::process::id()));
        let src = dir.join("src");
        let d = release(&src, "glm53flash_model");
        let up = dir.join("usage.json");
        std::fs::write(&up, usage_for(&d, 3).to_json()).unwrap();
        let out = dir.join("out");
        run(src.to_str().unwrap(), &out, 2, &up).unwrap();
        assert!(crate::weights::v7_manifest(&out).unwrap().is_none(), "not a glyd-v7 directory");
        let hb = HbDir::open(&out).unwrap().unwrap();
        let nl = d.desc.blocks.len() / 2;
        let (with, _) = crate::dynamic::split::layer_bytes_where(&out, nl, 100, 100, &|_| true).unwrap();
        let (without, _) = crate::dynamic::split::layer_bytes_where(&src, nl, 100, 100, &|n| !n.contains(".experts.")).unwrap();
        for r in &hb.layers {
            assert_eq!(with[r.block / 2], without[r.block / 2] + HbDir::bytes_of(r), "decoder layer {}", r.block / 2);
        }
        let first = hb.layers[0].block / 2;
        assert!(HbDir::experts_for(None, first, false).unwrap().is_none());
        assert!(HbDir::experts_for(Some(&hb), first, true).err().expect("refused").msg.contains("every layer across GPUs"));
        assert!(HbDir::experts_for(Some(&hb), 0, false).err().expect("refused").msg.contains("no experts for decoder layer 0"));
        assert!(HbDir::open(&src).unwrap().is_none());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn fp4_and_mxfp4_decode_to_the_release_values() {
        // (two weights a byte, the first low; a scale a run of 32: 2^(e - 127))
        let q: Vec<u8> = (0..32u8).map(|i| (i % 16) | ((15 - i % 16) << 4)).collect();
        let scales = [127u8, 130];
        let v = fp4_f32(&q, &scales, 2, 32).unwrap();
        assert_eq!(v[0], 0.0);
        assert_eq!(v[1], -6.0);
        assert_eq!(v[2], 0.5);
        assert_eq!(v[32 + 6], E2M1[3] * 8.0);
        assert!(fp4_f32(&q, &[127], 2, 32).is_err() && fp4_f32(&q, &[255, 127], 2, 32).is_err());
        // FP8 with an E8M0 scale a block of 128
        let q8 = vec![0x38u8; 128 * 130]; // (1.0 in E4M3)
        let v8 = fp8_e8m0_f32(&q8, &[128, 126], 128, 130).unwrap();
        assert_eq!((v8[0], v8[129]), (2.0, 0.5));
    }

    /// A tiny GLM-5.3-Flash and a tiny DeepSeek-V4 (transformers' fixtures, bf16) packed at the hummingbird level from a usage file and read
    /// back: each expert's bits are the recipe's, its rows decode, unrotated, to the release's matrix within the rounding, the signs are the
    /// layer's, the other tensors are the release's byte for byte, the config is there; refused: counts of another layer's size, a model
    /// with no experts' counts; a directory already packed is kept.
    #[test]
    fn a_tiny_release_is_packed_and_read_back() {
        for fixture in ["glm53flash_model", "dsv4_model", "qwen38_moe_model"] {
            let dir = std::env::temp_dir().join(format!("glyd-hbpack-{fixture}-{}", std::process::id()));
            let src = dir.join("src");
            let d = release(&src, fixture);
            let layers = moe_layers(&d).unwrap();
            assert!(!layers.is_empty(), "{fixture}");
            let u = usage_for(&d, 7);
            let up = dir.join("usage.json");
            std::fs::write(&up, u.to_json()).unwrap();
            let out = dir.join("out");
            run(src.to_str().unwrap(), &out, 3, &up).unwrap();
            let hb = HbDir::open(&out).unwrap().expect("a hummingbird directory");
            assert_eq!(hb.layers.len(), layers.len());
            let ck = Checkpoint::open(&src).unwrap();
            for l in &layers {
                let rows = hb.read_layer(l.block).unwrap();
                assert_eq!(rows.signs, signs(l.block, rows.width));
                let want_bits = bits_by_use(&u.layer(l.block).unwrap().counts, HB40_PERCENT);
                assert_eq!(rows.experts.iter().map(|e| e.bits).collect::<Vec<_>>(), want_bits, "{fixture} block {}", l.block);
                for e in [0, l.experts - 1] {
                    let (gu, _, _) = expert_f32(&ck, &l.gate_up, e, &d.scales).unwrap();
                    let mut back = hb_codec::decode(&rows.experts[e].gate_up).unwrap();
                    hb_codec::unrotate(&mut back, &rows.signs, rows.rot);
                    let err = back.iter().zip(&gu).map(|(a, b)| ((a - b) as f64).powi(2)).sum::<f64>() / gu.iter().map(|b| (*b as f64).powi(2)).sum::<f64>();
                    assert!(err < 0.2, "{fixture} block {} expert {e}: {err}", l.block);
                }
            }
            // the rest is the release's, byte for byte, and no expert tensor is copied
            let packed = Checkpoint::open(&out).unwrap();
            let skip = expert_tensors(&ck, &layers, &d.scales);
            assert!(!skip.is_empty());
            for n in ck.names() {
                match packed.get(n) {
                    Some(_) => assert_eq!(packed.read(n).unwrap(), ck.read(n).unwrap(), "{fixture}: {n}"),
                    None => assert!(skip.contains(n), "{fixture}: {n} missing"),
                }
            }
            assert!(packed.names().all(|n| !skip.contains(n)));
            assert!(out.join("config.json").exists() && out.join("tokenizer_config.json").exists());
            assert!(hb.experts_bytes() > 0);
            // a directory already packed is left as it is
            let before = std::fs::metadata(out.join("glyd.json")).unwrap().modified().unwrap();
            run(src.to_str().unwrap(), &out, 3, &up).unwrap();
            assert_eq!(std::fs::metadata(out.join("glyd.json")).unwrap().modified().unwrap(), before);
            // refused: counts of the wrong size, no counts for a layer
            let mut bad = u.clone();
            bad.layers[0].counts.pop();
            std::fs::write(&up, bad.to_json()).unwrap();
            let e = run(src.to_str().unwrap(), &dir.join("out2"), 2, &up).unwrap_err().msg;
            assert!(e.contains("counts, where the layer has"), "{e}");
            let mut none = u.clone();
            none.layers.remove(0);
            std::fs::write(&up, none.to_json()).unwrap();
            let e = run(src.to_str().unwrap(), &dir.join("out2"), 2, &up).unwrap_err().msg;
            assert!(e.contains("calibrate-experts"), "{e}");
            std::fs::remove_dir_all(&dir).unwrap();
        }
    }
}
