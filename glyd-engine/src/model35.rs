//! Qwen3.5 (2026) on the GPU, one sequence: the packed weights (every matrix the library packs; the small tensors as the checkpoint has them), the forward pass of its two
//! kinds of layer (kernels/qwen35.cu), a prompt in chunks and a decode step as one CUDA graph, greedy tokens. A development build's until the engine serves the model
//! (docs/superpowers/plans/2026-10-06-engine-qwen35.md, tasks 10 to 12).
//!
//! A layer: x += mixer(norm1(x)); x += mlp(norm2(x)), the norms zero-centered. Full attention (1 in 4): the qkv product (each query head with its gate), the q/k norms and
//! the rotary on 64 of 256 dims, the keys and values into a bf16 cache (contiguous: one sequence), the causal attention times sigmoid(gate), the o product. Linear attention:
//! the qkvz and ba products, the causal convolution with SiLU, the gated delta rule (a recurrence, or in chunks for a prompt), the gated norm, the out product.

use crate::dev::{copy_d2h_async, copy_h2d_async, memset_async, Buf, Dev, Ev, Graph, Kernel, Pinned};
use std::cell::RefCell;
use crate::ffi::{err, Error, Lib, Result, Stream};
use crate::kern::{norm_threads, Kern};
use crate::launch;
use crate::model::{embed_plan, Sample, TopGpu};
use crate::qwen35::{Config35, Kind, Pack35};
use crate::weights::{choose_layout, lossless_manifest, ordered, pack_item_map, read_lossless_parts, read_v7, rope_tables, upload_bytes, upload_packed, v7_manifest, Item, Layout, PackBuf};
use glyd_gpu::safetensors::Checkpoint;
use std::collections::{HashMap, HashSet};
use std::path::Path;

// The kernels' names: those the .cu sources give them, in every build (kname!).
macro_rules! kname {
    ($n:literal) => {
        $n
    };
}

/// Bytes of dynamic shared memory gdn_chunk takes (kernels/qwen35.cu).
const GDN_CHUNK_SMEM: u32 = (2 * 64 * 65 + 64 * 64 + 64 * 64 + 128 * 64 + 2 * 64) * 4;
/// Keys a tile and the padded row of attn256.
const AT_KB: u32 = 32;
const AT_KS: u32 = 129;
/// The parts a decode step's attention splits each row's keys into (attn256_split), and the floats a part's result takes.
pub const ATTN_SPLITS: usize = 16;
pub(crate) const ATTN_PART: usize = 258;

/// Qwen3.5's kernels with typed launches.
pub struct Kern35 {
    rmsnorm: Kernel,
    rmsnorm_add: Kernel,
    prep: Kernel,
    attn: Kernel,
    attn_split: Kernel,
    attn_merge: Kernel,
    conv: Kernel,
    rec: Kernel,
    chunk: Kernel,
    gnorm: Kernel,
    small: Kernel,
}

impl Kern35 {
    pub fn load(dev: &Dev) -> Result<Kern35> {
        let chunk = dev.kernel(kname!("gdn_chunk"))?;
        chunk.allow_smem(GDN_CHUNK_SMEM)?;
        Ok(Kern35 {
            rmsnorm: dev.kernel(kname!("rmsnorm1p"))?,
            rmsnorm_add: dev.kernel(kname!("rmsnorm1p_add"))?,
            prep: dev.kernel(kname!("q35_prep"))?,
            attn: dev.kernel(kname!("attn256"))?,
            attn_split: dev.kernel(kname!("attn256_parts"))?,
            attn_merge: dev.kernel(kname!("attn256_merge"))?,
            conv: dev.kernel(kname!("conv_silu"))?,
            rec: dev.kernel(kname!("gdn_recurrent"))?,
            chunk,
            gnorm: dev.kernel(kname!("gated_norm"))?,
            small: dev.kernel(kname!("small_linear"))?,
        })
    }

    /// out [rows, k] = rmsnorm(x) * (1 + w).
    ///
    /// # Safety
    /// Live device buffers of the sizes the kernels read and write (their comments in kernels/qwen35.cu).
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn rmsnorm(&self, s: Stream, x: u64, w: u64, out: u64, rows: u32, k: i32, eps: f32) -> Result<()> {
        launch!(self.rmsnorm, [rows, 1, 1], [norm_threads(k)?, 1, 1], s; x, w, out, k, eps)
    }

    /// x += y, then out = rmsnorm(x) * (1 + w).
    ///
    /// # Safety
    /// As `rmsnorm`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn rmsnorm_add(&self, s: Stream, x: u64, y: u64, w: u64, out: u64, rows: u32, k: i32, eps: f32) -> Result<()> {
        launch!(self.rmsnorm_add, [rows, 1, 1], [norm_threads(k)?, 1, 1], s; x, y, w, out, k, eps)
    }

    /// The n qkv rows of a full-attention layer: queries normed and turned into qo, keys normed and turned and values copied into the cache at each row's position.
    ///
    /// # Safety
    /// As `rmsnorm`; the cache holds every position in `pos`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn prep(&self, s: Stream, qkv: u64, stride: i32, qw: u64, kw: u64, cos: u64, sin: u64, half: i32, pos: u64, qo: u64, kc: u64, vc: u64, n: u32, nq: i32, nk: i32, eps: f32) -> Result<()> {
        if !(1..=64).contains(&half) {
            return err(format!("a rotary of {} dims: not supported", 2 * half));
        }
        launch!(self.prep, [n, (nq + 2 * nk) as u32, 1], [128, 1, 1], s; qkv, stride, qw, kw, cos, sin, half, pos, qo, kc, vc, nq, nk, eps)
    }

    /// The causal attention of n query rows over the cache, times sigmoid(gate), into out.
    ///
    /// # Safety
    /// As `rmsnorm`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn attn(&self, s: Stream, q: u64, qs: i32, gate: u64, gs: i32, kc: u64, vc: u64, pos: u64, out: u64, os: i32, n: i32, nq: i32, nk: i32) -> Result<()> {
        if nk == 0 || nq % nk != 0 || nq / nk > 16 {
            return err(format!("{nq} query heads over {nk} key-value heads: not supported"));
        }
        let g = nq / nk;
        let r = if n == 1 { 1 } else { (16 / g).max(1) };
        let smem = (AT_KB * (AT_KS + 128) + 128 * (r * g) as u32) * 4;
        let sl2 = (1.0f64 / 16.0 * std::f64::consts::LOG2_E) as f32;
        launch!(self.attn, [((n + r - 1) / r) as u32, nk as u32, 1], [(32 * r * g) as u32, 1, 1], smem smem, s; q, qs, gate, gs, kc, vc, pos, out, os, n, nq, nk, r, sl2)
    }

    /// A decode step's attention (n rows, each its own sequence position), the keys in ATTN_SPLITS parts at once, then combined: `part` holds n nq ATTN_SPLITS ATTN_PART floats.
    ///
    /// # Safety
    /// As `rmsnorm`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn attn_decode(&self, s: Stream, q: u64, qs: i32, gate: u64, gs: i32, kc: u64, vc: u64, pos: u64, out: u64, os: i32, n: i32, nq: i32, nk: i32, part: u64) -> Result<()> {
        if nk == 0 || nq % nk != 0 || nq / nk > 16 {
            return err(format!("{nq} query heads over {nk} key-value heads: not supported"));
        }
        let g = (nq / nk) as u32;
        let smem = (AT_KB * (AT_KS + 128) + 128 * g) * 4;
        let sl2 = (1.0f64 / 16.0 * std::f64::consts::LOG2_E) as f32;
        let splits = ATTN_SPLITS as i32;
        launch!(self.attn_split, [n as u32, nk as u32, ATTN_SPLITS as u32], [32 * g, 1, 1], smem smem, s; q, qs, kc, vc, pos, n, nq, nk, sl2, part, splits)?;
        launch!(self.attn_merge, [n as u32, nq as u32, 1], [256, 1, 1], s; part, splits, gate, gs, out, os, nq)
    }

    /// The causal convolution and SiLU of n rows (the first c channels of rows xs apart), the state read and written.
    ///
    /// # Safety
    /// As `rmsnorm`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn conv(&self, s: Stream, x: u64, xs: i32, w: u64, st: u64, y: u64, n: i32, c: i32) -> Result<()> {
        launch!(self.conv, [(c as u32).div_ceil(256), 1, 1], [256, 1, 1], s; x, xs, w, st, y, n, c)
    }

    /// The gated delta rule over n tokens: one after another, or in chunks of 64 (`chunked`).
    ///
    /// # Safety
    /// As `rmsnorm`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn delta(&self, s: Stream, chunked: bool, xc: u64, c: i32, ba: u64, sba: i32, a_log: u64, dt_bias: u64, state: u64, core: u64, n: i32, nk: i32, nv: i32) -> Result<()> {
        if chunked {
            launch!(self.chunk, [nv as u32, 2, 1], [256, 1, 1], smem GDN_CHUNK_SMEM, s; xc, c, ba, sba, a_log, dt_bias, state, core, n, nk, nv)
        } else {
            launch!(self.rec, [nv as u32, 1, 1], [128, 1, 1], s; xc, c, ba, sba, a_log, dt_bias, state, core, n, nk, nv)
        }
    }

    /// The gated norm of n rows of nv value heads, in place.
    ///
    /// # Safety
    /// As `rmsnorm`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn gated_norm(&self, s: Stream, core: u64, z: u64, zs: i32, w: u64, n: u32, nv: i32, eps: f32) -> Result<()> {
        launch!(self.gnorm, [n, nv as u32, 1], [128, 1, 1], s; core, z, zs, w, nv, eps)
    }

    /// y [m, rows] = x [m, cols] W^T for a bf16 W the library does not pack.
    ///
    /// # Safety
    /// As `rmsnorm`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn small_linear(&self, s: Stream, x: u64, xs: i32, w: u64, y: u64, ys: i32, m: u32, rows: i32, cols: i32) -> Result<()> {
        launch!(self.small, [m, (rows.max(1) as u32).div_ceil(8), 1], [256, 1, 1], s; x, xs, w, y, ys, rows, cols)
    }
}

/// A matrix of the model: packed, or bf16 as the checkpoint has it (rows, cols) where the library does not pack it.
pub enum Mat {
    Packed(PackBuf),
    Dense(Buf, usize, usize),
}

impl Mat {
    pub fn bytes(&self) -> usize {
        match self {
            Mat::Packed(p) => p.bytes,
            Mat::Dense(b, _, _) => b.bytes,
        }
    }

    pub fn rows(&self) -> usize {
        match self {
            Mat::Packed(p) => p.pack.rows as usize,
            Mat::Dense(_, r, _) => *r,
        }
    }

    /// Its product's rows: the whole matrix's where this GPU holds a slice of them.
    pub fn whole_rows(&self) -> usize {
        match self {
            Mat::Packed(p) => p.whole_rows(),
            Mat::Dense(_, r, _) => *r,
        }
    }

    pub(crate) fn packed_opt(&self) -> Option<&PackBuf> {
        match self {
            Mat::Packed(p) => Some(p),
            Mat::Dense(..) => None,
        }
    }

    fn packed(&self) -> Result<&PackBuf> {
        match self {
            Mat::Packed(p) => Ok(p),
            Mat::Dense(..) => err("a matrix the library does not pack, where a packed one is needed"),
        }
    }
}

/// A layer's buffers, by kind: its matrices (full: qkv, o; linear: qkvz, ba, out; then gate_up, down) and small tensors.
pub struct Layer35 {
    pub kind: Kind,
    /// Its index among the layers of its kind (a linear layer's state, a full layer's cache).
    pub idx: usize,
    pub mats: Vec<Mat>,
    pub in_norm: Buf,
    pub post_norm: Buf,
    /// Full: q_norm, k_norm. Linear: conv1d, A_log, dt_bias, the gated norm's weight.
    pub small: Vec<Buf>,
}

/// Where a matrix of a layer comes from: a pack of `Config35::packs` (by its key), or the tensors kept as the checkpoint has them, stacked by rows (rows, cols).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Src {
    Pack(String),
    Dense(Vec<String>, usize, usize),
}

/// Each layer's matrices in the forward pass's order: full attention qkv, o; linear attention qkvz, ba, out; then gate_up, down. A matrix the packer does not pack (b and a
/// of each layer, 16 to 48 rows a part on the 2026 dense models) is the parts it keeps, stacked.
pub fn layer_mats(cfg: &Config35) -> Vec<Vec<Src>> {
    let keys: std::collections::HashSet<String> = cfg.packs().into_iter().map(|p| p.key).collect();
    let h = cfg.hidden;
    let one = |key: String, parts: &[(&str, usize)], module: &str| {
        if keys.contains(&key) {
            Src::Pack(key)
        } else {
            Src::Dense(parts.iter().map(|(n, _)| format!("{module}.{n}.weight")).collect(), parts.iter().map(|p| p.1).sum(), h)
        }
    };
    cfg.kinds
        .iter()
        .enumerate()
        .map(|(l, kind)| {
            let p = format!("{}layers.{l}", cfg.prefix);
            let mut v = match kind {
                Kind::Full => vec![Src::Pack(format!("{p}.self_attn.q_proj")), Src::Pack(format!("{p}.self_attn.o_proj"))],
                Kind::Linear => {
                    let m = format!("{p}.linear_attn");
                    vec![Src::Pack(format!("{m}.in_proj_qkv")), one(format!("{m}.in_proj_b"), &[("in_proj_b", cfg.lin_v_heads), ("in_proj_a", cfg.lin_v_heads)], &m), Src::Pack(format!("{m}.out_proj"))]
                }
            };
            v.push(Src::Pack(format!("{p}.mlp.gate_proj")));
            v.push(Src::Pack(format!("{p}.mlp.down_proj")));
            v
        })
        .collect()
}

/// The model's weights in GPU memory.
pub struct Weights35 {
    pub cfg: Config35,
    pub layout: Layout,
    pub layers: Vec<Layer35>,
    pub embed: PackBuf,
    pub head: Option<PackBuf>,
    pub norm: Buf,
    pub cos: Buf,
    pub sin: Buf,
    /// Device bytes of the weights (packs, the tensors kept as they are, the rotary tables).
    pub bytes: usize,
    /// The checkpoint's own bytes for the same tensors.
    pub ckpt_bytes: usize,
    /// The vision tower's and MTP head's tensors, not loaded.
    pub skipped: usize,
}

/// `x` rounded to bf16 (to nearest, ties to even), as an f32.
fn bf16_round(x: f32) -> f32 {
    if x.is_nan() {
        return x;
    }
    let b = x.to_bits();
    f32::from_bits(b.wrapping_add(0x7fff + ((b >> 16) & 1)) & 0xffff_0000)
}

/// `--lossy`'s weights: every bf16 value rounded to three fraction bits (glyd_gpu::pack::round_a), the matrices then held in the compact layout.
pub fn round_a_all(w: &mut [u16]) {
    w.iter_mut().for_each(|x| *x = glyd_gpu::pack::round_a(*x));
}

fn transform_bf16_bytes(bytes: &mut [u8], transform: Option<fn(&mut [u16])>) -> Result<()> {
    if bytes.len() % 2 != 0 { return err("odd BF16 byte count"); }
    if let Some(f) = transform {
        let mut words: Vec<u16> = bytes.chunks_exact(2).map(|b| u16::from_le_bytes([b[0], b[1]])).collect();
        f(&mut words);
        for (b, w) in bytes.chunks_exact_mut(2).zip(words) { b.copy_from_slice(&w.to_le_bytes()); }
    }
    Ok(())
}

fn runtime_parameter(bytes: Vec<u8>, source: &str, wanted: &str) -> Result<Vec<u8>> {
    if source == wanted { return Ok(bytes); }
    if source == "BF16" && wanted == "F32" && bytes.len() % 2 == 0 {
        return Ok(bytes.chunks_exact(2).flat_map(|b| ((u16::from_le_bytes([b[0], b[1]]) as u32) << 16).to_le_bytes()).collect());
    }
    err(format!("unsupported parameter storage {source} -> {wanted}"))
}

impl Weights35 {
    pub fn lm_head(&self) -> &PackBuf {
        self.head.as_ref().unwrap_or(&self.embed)
    }

    /// A Hub-format directory (config.json, the safetensors): checked against the weight map, every packable matrix packed here by `threads` workers in `layout` (None: the GPU's,
    /// as `glyd-engine run` chooses), each checked to decode to its weights bit for bit (the packer's own check) and copied to the GPU as it is done.
    pub fn load(dir: &Path, max_pos: usize, threads: usize, layout: Option<Layout>, cc: (i32, i32), total_mem: u64, transform: Option<fn(&mut [u16])>) -> Result<Weights35> {
        Weights35::load_stage(dir, max_pos, threads, layout, cc, total_mem, transform, None)
    }

    /// As `load`, of a stage alone where `stage` is (layers a..b, whether it holds the embedding and the head: a model split across GPUs): another
    /// stage's matrices and tensors are not read, and stand here as empty buffers (never run).
    #[allow(clippy::too_many_arguments)]
    pub fn load_stage(dir: &Path, max_pos: usize, threads: usize, layout: Option<Layout>, cc: (i32, i32), total_mem: u64, transform: Option<fn(&mut [u16])>, stage: Option<((usize, usize), bool, bool)>) -> Result<Weights35> {
        // (the layer a tensor or pack belongs to: the number after "layers.")
        let layer_of = |name: &str| name.split("layers.").nth(1).and_then(|r| r.split('.').next()).and_then(|n| n.parse::<usize>().ok());
        let ((la, lb), first, last) = stage.unwrap_or(((0, usize::MAX), true, true));
        let in_stage = |name: &str| layer_of(name).is_none_or(|l| (la..lb).contains(&l));
        let cfg = Config35::load(dir)?;
        let ck = Checkpoint::open(dir).map_err(|e| Error(e.to_string()))?;
        let have: Vec<(String, String, Vec<u64>)> = ck.names().map(|n| {
            let (_, t) = ck.get(n).expect("a tensor the checkpoint names");
            (n.to_string(), t.dtype.clone(), t.shape.clone())
        }).collect();
        // a glyd-v7 directory (a lossy level's packs, saved): its matrices are read as they are, in its level's layout; one it holds as bf16 (the embedding, the LM head) is packed here
        // (a lossless glyd-v4 save, glyd/NAME-penguin's: the tiered layout, its packs read as saved; a matrix it holds in two packs is decoded and packed again, read_lossless_parts)
        let v7 = match v7_manifest(dir)? {
            Some(x) => Some(x),
            None => lossless_manifest(dir)?.map(|m| (Layout::Tiered, m)),
        };
        if let (Some((got, _)), Some(asked)) = (&v7, layout) {
            if *got != asked {
                return err(format!("{}: packed in the {got:?} layout, where {asked:?} is asked", dir.display()));
            }
        }
        let packs = cfg.packs();
        // (each pack's saved parts: its own name where the save fused them, else each part's)
        let has = |k: &str| ck.get(&format!("{k}.glyd_data")).is_some();
        let saved: Vec<Vec<(&str, usize)>> = packs.iter().map(|p| {
            let each: Vec<(&str, usize)> = p.parts.iter().map(|x| (Pack35::saved(&x.0), x.1)).collect();
            if v7.is_none() { vec![] } else if has(each[0].0) && (each.len() == 1 || !has(each[1].0)) { vec![(each[0].0, p.rows())] } else if each.iter().all(|x| has(x.0)) { each } else { vec![] }
        }).collect();
        let held: HashSet<String> = saved.iter().flatten().map(|x| x.0.to_string()).chain(packs.iter().zip(&saved).filter(|(_, s)| s.len() == 1).flat_map(|(p, _)| p.parts.iter().map(|x| Pack35::saved(&x.0).to_string()))).collect();
        let skipped = cfg.check(&have, &held)?;
        let matrix_bytes: u64 = packs.iter().map(|p| (p.rows() * p.cols * 2) as u64).sum();
        let kept = cfg.kept();
        let other: u64 = kept.iter().map(|t| ck.get(&t.name).expect("validated tensor").1.bytes).sum();
        let ckpt_bytes = (matrix_bytes + other) as usize;
        let runtime_other: u64 = kept.iter().map(|t| t.shape.iter().product::<u64>() * if t.dtype == "F32" { 4 } else { 2 }).sum();
        let layout = match &v7 {
            Some((l, _)) => *l,
            None => choose_layout(layout, cc, total_mem, matrix_bytes, runtime_other).0,
        };
        let items: Vec<Item> = packs.iter().map(|p| Item { members: p.parts.iter().map(|x| x.0.clone()).collect(), rows: p.rows(), cols: p.cols }).collect();
        let tie = cfg.tie;
        let wanted = |key: &str| match key {
            "embed" => first || (last && tie),
            "lm_head" => last,
            k => in_stage(k),
        };
        let ks: Vec<usize> = (0..packs.len()).filter(|&k| wanted(&packs[k].key)).collect();
        let mut packed: HashMap<String, PackBuf> = HashMap::new();
        ordered(ks.len(), threads, 2 * threads.max(1), &|i| { let k = ks[i]; match &v7 {
            Some((Layout::Tiered, m)) if !saved[k].is_empty() => read_lossless_parts(&ck, m, &saved[k], items[k].cols),
            Some((_, m)) if !saved[k].is_empty() => read_v7(&ck, m, layout, &saved[k], items[k].cols),
            _ => pack_item_map(&ck, &items[k], layout, &|w| { if let Some(f) = transform { f(w); } }),
        } }, |i, p| {
            crate::status::STATUS.beat(); // (a sign of life for the program that started the engine)
            packed.insert(packs[ks[i]].key.clone(), upload_packed(p)?);
            Ok(())
        })?;
        // (an empty pack: where another stage holds the matrix)
        let empty = || upload_packed(crate::weights::Packed { data: vec![0; 16], b1: vec![0; 16], b2: vec![0; 16], words: crate::ffi::Words::Tiered { blocks: 0, block_base: 0, tiers: [0; 3] }, rows: 0, cols: 0 });
        let read = |name: &str| -> Result<Vec<u8>> {
            let mut b = ck.read(name).map_err(|e| Error(e.to_string()))?;
            if ck.get(name).expect("validated tensor").1.dtype == "BF16" {
                transform_bf16_bytes(&mut b, transform)?;
            }
            Ok(b)
        };
        let mut mats_of = |srcs: Vec<Src>| -> Result<Vec<Mat>> {
            srcs.into_iter()
                .map(|src| match src {
                    Src::Pack(key) => packed.remove(&key).map(Mat::Packed).ok_or_else(|| Error(format!("no pack {key}"))),
                    Src::Dense(parts, rows, cols) => {
                        let mut b = Vec::with_capacity(rows * cols * 2);
                        for n in &parts {
                            b.extend(read(n)?);
                        }
                        Ok(Mat::Dense(upload_bytes(&b)?, rows, cols))
                    }
                })
                .collect()
        };
        let plan = layer_mats(&cfg);
        let dense: std::collections::HashSet<String> = plan.iter().flatten().filter_map(|s| if let Src::Dense(p, ..) = s { Some(p.clone()) } else { None }).flatten().collect();
        let mut small: HashMap<String, Buf> = HashMap::new();
        for t in kept.iter().filter(|t| !dense.contains(&t.name) && in_stage(&t.name) && (last || layer_of(&t.name).is_some())) {
            let source_dtype = &ck.get(&t.name).expect("validated tensor").1.dtype;
            let mut b = runtime_parameter(read(&t.name)?, source_dtype, t.dtype)?;
            if t.name.ends_with("linear_attn.norm.weight") && t.dtype == "F32" {
                // the gated norm's weight as vLLM 0.30 holds it, in the model's dtype: the checkpoint's fp32 rounded to bf16 (kept 4 bytes wide for the kernel); A_log stays fp32, as there
                for v in b.as_chunks_mut::<4>().0 {
                    *v = bf16_round(f32::from_le_bytes(*v)).to_le_bytes();
                }
            }
            small.insert(t.name.clone(), upload_bytes(&b)?);
        }
        let mut take = |name: String| small.remove(&name).ok_or_else(|| Error(format!("no tensor {name}")));
        let mut layers = Vec::with_capacity(cfg.layers);
        for ((l, (kind, idx)), srcs) in cfg.schedule().into_iter().enumerate().zip(plan) {
            let p = format!("{}layers.{l}", cfg.prefix);
            if !(la..lb).contains(&l) {
                // (another stage's layer: empty buffers, never run)
                let n_small = if kind == Kind::Full { 2 } else { 4 };
                layers.push(Layer35 { kind, idx, mats: (0..srcs.len()).map(|_| Ok(Mat::Dense(Buf::new(2)?, 0, 0))).collect::<Result<_>>()?, in_norm: Buf::new(4)?, post_norm: Buf::new(4)?, small: (0..n_small).map(|_| Buf::new(4)).collect::<Result<_>>()? });
                continue;
            }
            let ms = mats_of(srcs)?;
            let sm = match kind {
                Kind::Full => vec![take(format!("{p}.self_attn.q_norm.weight"))?, take(format!("{p}.self_attn.k_norm.weight"))?],
                Kind::Linear => vec![
                    take(format!("{p}.linear_attn.conv1d.weight"))?,
                    take(format!("{p}.linear_attn.A_log"))?,
                    take(format!("{p}.linear_attn.dt_bias"))?,
                    take(format!("{p}.linear_attn.norm.weight"))?,
                ],
            };
            layers.push(Layer35 { kind, idx, mats: ms, in_norm: take(format!("{p}.input_layernorm.weight"))?, post_norm: take(format!("{p}.post_attention_layernorm.weight"))?, small: sm });
        }
        let embed = match packed.remove("embed") {
            Some(e) => e,
            None if !(first || (last && cfg.tie)) => empty()?,
            None => return err("the embedding is not packed"),
        };
        let head = if cfg.tie { None } else if last { Some(packed.remove("lm_head").ok_or_else(|| Error("the LM head is not packed".into()))?) } else { Some(empty()?) };
        if let Some(k) = packed.keys().next() {
            return err(format!("a pack the forward pass does not take: {k}"));
        }
        let norm = if last { take(format!("{}norm.weight", cfg.prefix))? } else { Buf::new(4)? };
        let (c, s) = rope_tables(cfg.rope_theta, cfg.rotary_dim, max_pos.min(cfg.max_pos));
        let (cos, sin) = (upload_bytes(crate::weights::as_bytes(&c))?, upload_bytes(crate::weights::as_bytes(&s))?);
        let mut bytes = norm.bytes + cos.bytes + sin.bytes + embed.bytes + head.as_ref().map_or(0, |h| h.bytes);
        for ly in &layers {
            bytes += ly.mats.iter().map(Mat::bytes).sum::<usize>() + ly.in_norm.bytes + ly.post_norm.bytes + ly.small.iter().map(|b| b.bytes).sum::<usize>();
        }
        Ok(Weights35 { cfg, layout, layers, embed, head, norm, cos, sin, bytes, ckpt_bytes, skipped })
    }
}

/// How a run is set up.
pub struct Opts35 {
    /// Explicit research-only BF16 approximation, before the checked lossless GPU packing.
    /// This measures quality; it does not execute the research format on the GPU.
    pub weight_transform: Option<fn(&mut [u16])>,
    /// Positions the cache holds (prompt and reply).
    pub ctx: usize,
    /// Tokens a prompt chunk takes at most.
    pub chunk: usize,
    pub graph: bool,
    pub device: i32,
    pub layout: Option<Layout>,
    /// A prompt's delta rule in chunks of 64 (else one token after another, as a decode step).
    pub chunked: bool,
}

impl Default for Opts35 {
    fn default() -> Opts35 {
        Opts35 { weight_transform: None, ctx: 4096, chunk: 2048, graph: true, device: 0, layout: None, chunked: true }
    }
}

pub struct Engine35 {
    pub dev: Dev,
    pub lib: Lib,
    kern: Kern,
    k35: Kern35,
    pub w: Weights35,
    chunk: usize,
    ctx: usize,
    chunked: bool,
    // activations, `chunk` rows each
    x: Buf,
    h: Buf,
    mix: Buf,
    ba: Buf,
    xc: Buf,
    core: Buf,
    qo: Buf,
    att: Buf,
    o: Buf,
    gu: Buf,
    m: Buf,
    d: Buf,
    hlast: Buf,
    logits: Buf,
    scratch: Buf,
    embed_groups: Buf,
    embed_pairs: Buf,
    pos: Buf,
    st: Buf,
    tok: Buf,
    val: Buf,
    lse: Buf,
    /// The sampler's candidates for the last step, where its request draws (`Gen::reply`).
    pub top: TopGpu,
    /// A decode step's attention parts.
    attn_part: Buf,
    // the products
    ws: Buf,
    done: Buf,
    blas: Option<crate::ffi::Blas>,
    own: bool,
    scratch_w: Buf,
    gpu_code: i64,
    // what a sequence holds
    pub kc: Buf,
    pub vc: Buf,
    pub ssm: Buf,
    pub convst: Buf,
    graph: Option<Graph>,
    host: Pinned,
    /// Tokens in the cache.
    pub n: usize,
    /// Device bytes of the activations and workspaces.
    pub work_bytes: usize,
    /// A step's profile: an event after each kind of work, labelled (`profile_steps`).
    marks: RefCell<Option<Vec<(&'static str, Ev)>>>,
}

/// 64-row groups of the embedding decoded at once for a prompt chunk.
const EMBED_GROUPS: usize = 32;

impl Engine35 {
    pub fn new(dir: &Path, o: &Opts35) -> Result<Engine35> {
        let dev = Dev::new(o.device)?;
        let lib = Lib::find()?;
        let kern = Kern::load(&dev)?;
        let k35 = Kern35::load(&dev)?;
        let (free0, _) = dev.mem_info()?;
        let threads = std::thread::available_parallelism().map_or(8, |n| n.get().min(16));
        let w = Weights35::load(dir, o.ctx + 16, threads, o.layout, dev.cc_pair(), dev.total_mem as u64, o.weight_transform)?;
        let c = &w.cfg;
        if c.head_dim != 256 || c.lin_k_dim != 128 || c.lin_v_dim != 128 || c.conv_kernel != 4 {
            return err(format!("head {}, linear heads {}/{}, convolution {}: this engine runs head 256, linear heads of 128 and a convolution of 4", c.head_dim, c.lin_k_dim, c.lin_v_dim, c.conv_kernel));
        }
        let (hid, nq, nk, nv, inter) = (c.hidden, c.heads, c.kv_heads, c.lin_v_heads, c.inter);
        let (kd, vd, cd) = (c.key_dim(), c.value_dim(), c.conv_dim());
        let ch = o.chunk.max(1);
        let full_w = 2 * nq * 256 + 2 * nk * 256;
        let act = |cols: usize, rows: usize| Buf::new(rows * cols * 2);
        let (x, h) = (act(hid, ch)?, act(hid, ch)?);
        let mix = act((cd + vd).max(full_w), ch)?;
        let ba = act(2 * nv, ch)?;
        let xc = act(cd, ch)?;
        let core = act(vd, ch)?;
        let (qo, att) = (act(nq * 256, ch)?, act(nq * 256, ch)?);
        let (o_, gu, m, d) = (act(hid, ch)?, act(2 * inter, ch)?, act(inter, ch)?, act(hid, ch)?);
        let (hlast, logits) = (act(hid, 1)?, act(c.vocab, 1)?);
        let scratch = Buf::new(64 * hid * 2)?;
        let embed_groups = Buf::new(EMBED_GROUPS * 64 * hid * 2)?;
        let embed_pairs = Buf::new(ch * 8)?;
        let (pos, st) = (Buf::zeroed(4 * ch)?, Buf::zeroed(16)?);
        let (tok, val, lse) = (Buf::zeroed(4)?, Buf::zeroed(4)?, Buf::zeroed(4)?);
        let top = TopGpu::new(c.vocab)?;
        let attn_part = Buf::zeroed(nq * ATTN_SPLITS * ATTN_PART * 4)?;
        let want = crate::tune!("GLYD_ENGINE_GEMM").unwrap_or_default();
        let blas = if ch > 64 && (want == "blas" || (want.is_empty() && dev.cc != (8, 9))) { crate::ffi::Blas::find() } else { None };
        let own = want == "own" || (want.is_empty() && blas.is_none() && matches!(dev.cc.0, 8 | 9));
        let gpu_code = lib.gpu()? as i64;
        let packed: Vec<&PackBuf> = w.layers.iter().flat_map(|l| l.mats.iter()).filter_map(|m| m.packed().ok()).collect();
        let scratch_w = if blas.is_some() || own { Buf::new(packed.iter().map(|p| (p.pack.rows * p.pack.cols) as usize * 2).max().unwrap_or(16))? } else { Buf::new(16)? };
        let mut shapes: Vec<(i64, i64, bool)> = packed.iter().map(|p| (p.pack.rows, p.pack.cols, p.pack.is_twelve())).collect();
        shapes.sort_unstable();
        shapes.dedup();
        let mut need = 0usize;
        for m_ in 1..=ch as i64 {
            for &(oo, kk, twelve) in &shapes {
                need = need.max(lib.linear_workspace(twelve, oo, kk, m_)?);
            }
        }
        let head = w.lm_head();
        need = need.max(lib.linear_workspace(head.pack.is_twelve(), head.pack.rows, head.pack.cols, 1)?);
        let ws = Buf::zeroed(need)?;
        let done = Buf::zeroed(4 << 18)?;
        let ctx = o.ctx.div_ceil(16) * 16;
        let kv_bytes = c.full_layers() * ctx * nk * 256 * 2;
        let (kc, vc) = (Buf::zeroed(kv_bytes)?, Buf::zeroed(kv_bytes)?);
        let ssm = Buf::zeroed(c.linear_layers() * nv * 128 * 128 * 4)?;
        let convst = Buf::zeroed(c.linear_layers() * 3 * cd * 2)?;
        let _ = kd;
        let host = Pinned::new(4096)?;
        let used = free0.saturating_sub(dev.mem_info()?.0);
        let work_bytes = used.saturating_sub(w.bytes + kc.bytes + vc.bytes + ssm.bytes + convst.bytes);
        let mut e = Engine35 {
            dev, lib, kern, k35, w, chunk: ch, ctx, chunked: o.chunked, x, h, mix, ba, xc, core, qo, att, o: o_, gu, m, d, hlast, logits, scratch, embed_groups, embed_pairs, pos, st, tok, val, lse, top, attn_part, ws, done, blas, own, scratch_w, gpu_code, kc, vc, ssm, convst, graph: None, host, n: 0, work_bytes, marks: RefCell::new(None),
        };
        if o.graph {
            e.capture_graph()?;
        }
        Ok(e)
    }

    pub fn describe(&self) -> String {
        let gib = |b: usize| b as f64 / (1u64 << 30) as f64;
        let (free, total) = self.dev.mem_info().unwrap_or((0, 0));
        format!(
            "{} (sm_{}{}): weights {:.3} GiB ({:.1}% of the checkpoint's {:.3} GiB for the same tensors; {} vision and MTP tensors not loaded; layout {:?}), states {:.3} GiB, KV cache {} positions {:.3} GiB, workspaces {:.3} GiB, {:.2} of {:.2} GiB in use",
            self.dev.name, self.dev.cc.0, self.dev.cc.1, gib(self.w.bytes), 100.0 * self.w.bytes as f64 / self.w.ckpt_bytes as f64, gib(self.w.ckpt_bytes), self.w.skipped, self.w.layout,
            gib(self.ssm.bytes + self.convst.bytes), self.ctx, gib(self.kc.bytes + self.vc.bytes), gib(self.work_bytes), gib(total - free), gib(total)
        )
    }

    /// GPU memory in use now (bytes): what nvidia-smi counts for the process, less what the driver holds before the engine starts.
    pub fn mem_used(&self) -> usize {
        let (free, total) = self.dev.mem_info().unwrap_or((0, 0));
        total - free
    }

    /// Last forward pass's BF16 logits, widened exactly for development scoring.
    pub fn logits(&self) -> Result<Vec<f32>> {
        self.dev.sync()?;
        let mut words = vec![0u16; self.w.cfg.vocab];
        self.logits.read(0, &mut words)?;
        Ok(words.into_iter().map(|w| f32::from_bits((w as u32) << 16)).collect())
    }

    pub fn has_graph(&self) -> bool {
        self.graph.is_some()
    }

    pub fn chunk(&self) -> usize {
        self.chunk
    }

    /// A new sequence: the states zeroed (the cache's old positions are never read: a row reads positions up to its own).
    pub fn reset(&mut self) -> Result<()> {
        self.n = 0;
        self.top.ask = None;
        memset_async(self.ssm.ptr, 0, self.ssm.bytes, self.dev.stream)?;
        memset_async(self.convst.ptr, 0, self.convst.bytes, self.dev.stream)?;
        self.dev.sync()
    }

    /// Whether a product of `m` rows by `p` decodes the matrix and multiplies by a dense product (the library's route says DECODE or AHEAD), as the Qwen3 engine's `lin`.
    unsafe fn decoded_way(&self, p: &PackBuf, m: usize) -> Result<bool> {
        if (self.blas.is_none() && !self.own) || m <= 64 || p.pack.cols % 64 != 0 || p.pack.rows % 128 != 0 {
            return Ok(false);
        }
        let route = self.lib.route(self.gpu_code, p.pack.is_twelve(), p.pack.rows, p.pack.cols, m as i64)?;
        Ok(route == 0 || route == 5)
    }

    unsafe fn lin_packed(&self, p: &PackBuf, x: u64, m: usize, y: u64) -> Result<()> {
        if self.decoded_way(p, m)? {
            self.lib.unpack(&p.pack, 0, p.pack.rows, self.scratch_w.ptr, self.dev.stream)?;
            if let Some(b) = &self.blas {
                return b.gemm_nt(self.dev.stream, self.scratch_w.ptr, x, y, p.pack.rows as i32, p.pack.cols as i32, m as i32);
            }
            let tile = self.kern.pick_tile(m, p.pack.rows as usize, p.pack.cols as usize);
            return self.kern.gemm(self.dev.stream, tile, x, self.scratch_w.ptr, y, m as i32, p.pack.rows as i32, p.pack.cols as i32);
        }
        self.lib.linear(&p.pack, x, m as i64, y, self.ws.ptr, self.ws.bytes, self.done.ptr, self.dev.stream)
    }

    /// Y [m, rows] = X [m, cols] W^T, X and Y dense.
    unsafe fn lin(&self, w: &Mat, x: u64, m: usize, y: u64) -> Result<()> {
        match w {
            Mat::Packed(p) => self.lin_packed(p, x, m, y),
            Mat::Dense(b, rows, cols) => self.k35.small_linear(self.dev.stream, x, *cols as i32, b.ptr, y, *rows as i32, m as u32, *rows as i32, *cols as i32),
        }
    }

    /// silu(gate) * up of the gate-and-up product into `m`, as the Qwen3 engine's.
    unsafe fn gate_up_silu(&self, w: &Mat, n: usize) -> Result<()> {
        let s = self.dev.stream;
        let inter = self.w.cfg.inter;
        if let Mat::Packed(p) = w {
            if self.blas.is_none() && self.own && self.decoded_way(p, n)? && p.pack.rows as usize == 2 * inter && inter.is_multiple_of(128) {
                self.lib.unpack(&p.pack, 0, p.pack.rows, self.scratch_w.ptr, s)?;
                return self.kern.gemm_silu(s, self.h.ptr, self.scratch_w.ptr, self.m.ptr, n as i32, p.pack.rows as i32, p.pack.cols as i32);
            }
        }
        self.lin(w, self.h.ptr, n, self.gu.ptr)?;
        self.kern.silu_mul(s, self.gu.ptr, self.m.ptr, n as u32, inter as i32)
    }

    /// Where a profile is being taken: an event now, labelled with the work queued since the last.
    fn mark(&self, label: &'static str) -> Result<()> {
        if let Some(v) = self.marks.borrow_mut().as_mut() {
            let e = Ev::new()?;
            e.record(self.dev.stream)?;
            v.push((label, e));
        }
        Ok(())
    }

    /// `steps` decode steps run eagerly with an event after each kind of work: the GPU time of each kind a step (ms), the launches a step, and the total.
    pub fn profile_steps(&mut self, mut token: u32, steps: usize) -> Result<Vec<(String, f64, usize)>> {
        let mut acc: Vec<(String, f64, usize)> = Vec::new();
        let mut total = 0.0;
        for _ in 0..steps {
            self.dev.sync()?;
            let p = [self.n as i32];
            let start = Ev::new()?;
            // SAFETY: as `step`.
            unsafe {
                let hp = (self.host.ptr as *mut u8).add(64);
                std::ptr::copy_nonoverlapping(p.as_ptr() as *const u8, hp, 4);
                copy_h2d_async(self.st.ptr, hp as *const _, 4, self.dev.stream)?;
                *self.marks.borrow_mut() = Some(Vec::new());
                start.record(self.dev.stream)?;
                self.embed(&[token])?;
                self.mark("embedding row")?;
                self.forward(1, self.st.ptr, true, None)?;
            }
            self.dev.sync()?;
            self.n += 1;
            let marks = self.marks.borrow_mut().take().unwrap_or_default();
            let mut prev = &start;
            for (label, e) in &marks {
                let ms = e.since(prev)? as f64;
                prev = e;
                total += ms;
                match acc.iter_mut().find(|a| a.0 == *label) {
                    Some(a) => {
                        a.1 += ms;
                        a.2 += 1;
                    }
                    None => acc.push((label.to_string(), ms, 1)),
                }
            }
            let mut t = [0i32; 1];
            self.tok.read(0, &mut t)?;
            token = t[0] as u32;
        }
        let mut out: Vec<(String, f64, usize)> = acc.into_iter().map(|(l, ms, n)| (l, ms / steps as f64, n / steps)).collect();
        out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        out.push(("total".to_string(), total / steps as f64, 0));
        Ok(out)
    }

    /// Token rows into x (as the Qwen3 engine's `embed`).
    unsafe fn embed(&self, tokens: &[u32]) -> Result<()> {
        let hid = self.w.cfg.hidden;
        let s = self.dev.stream;
        if let Some(&t) = tokens.iter().find(|&&t| t as usize >= self.w.cfg.vocab) {
            return err(format!("token {t}: the vocabulary has {}", self.w.cfg.vocab));
        }
        if tokens.len() == 1 {
            let t = tokens[0] as i64;
            let row0 = (t / 64) * 64;
            self.lib.unpack(&self.w.embed.pack, row0, 64, self.scratch.ptr, s)?;
            return self.kern.copy_row(s, self.scratch.ptr, (t - row0) as i32, hid as i32, self.x.ptr);
        }
        let (groups, by_batch) = embed_plan(tokens, EMBED_GROUPS);
        let counts: Vec<usize> = by_batch.iter().map(Vec::len).collect();
        let pairs: Vec<i32> = by_batch.iter().flatten().flat_map(|&(d, s_)| [d, s_]).collect();
        self.embed_pairs.write(0, &pairs)?;
        let mut done = 0usize;
        for (b, batch) in groups.chunks(EMBED_GROUPS).enumerate() {
            for (j, &g) in batch.iter().enumerate() {
                self.lib.unpack(&self.w.embed.pack, g * 64, 64, self.embed_groups.ptr + (j * 64 * hid * 2) as u64, s)?;
            }
            self.kern.copy_rows(s, self.embed_groups.ptr, self.embed_pairs.ptr + (done * 8) as u64, counts[b] as i32, hid as i32, self.x.ptr)?;
            done += counts[b];
        }
        Ok(())
    }

    /// The layers on the n rows in x (positions at `pos`, int32 on the device); where `head`, the final norm, the LM head and the argmax on the last row. `dump`: the last row
    /// of the residual stream after every layer but the last, then the final norm's output of it, copied out (a reference's hidden states, layer by layer).
    unsafe fn forward(&self, n: usize, pos: u64, head: bool, mut dump: Option<&mut Vec<Vec<u16>>>) -> Result<()> {
        let c = &self.w.cfg;
        let s = self.dev.stream;
        let (hid, nq, nk, nv) = (c.hidden, c.heads as i32, c.kv_heads as i32, c.lin_v_heads as i32);
        let (cd, vd) = (c.conv_dim(), c.value_dim());
        let (nu, hid_i, eps) = (n as u32, hid as i32, c.eps);
        let full_w = (2 * c.heads * 256 + 2 * c.kv_heads * 256) as i32;
        let kv_layer = (self.ctx * c.kv_heads * 256 * 2) as u64;
        let ssm_layer = (c.lin_v_heads * 128 * 128 * 4) as u64;
        let conv_layer = (3 * cd * 2) as u64;
        let chunked = self.chunked && n > 1;
        for (l, ly) in self.w.layers.iter().enumerate() {
            if l == 0 {
                self.k35.rmsnorm(s, self.x.ptr, ly.in_norm.ptr, self.h.ptr, nu, hid_i, eps)?;
                self.mark("norms")?;
            } else {
                self.k35.rmsnorm_add(s, self.x.ptr, self.d.ptr, ly.in_norm.ptr, self.h.ptr, nu, hid_i, eps)?;
                self.mark("norms")?;
                if let Some(out) = dump.as_mut() {
                    self.dev.sync()?;
                    let mut v = vec![0u16; hid];
                    self.x.read((n - 1) * hid * 2, &mut v)?;
                    out.push(v);
                }
            }
            match ly.kind {
                Kind::Linear => {
                    let (sw, sa, sd, sn) = (ly.small[0].ptr, ly.small[1].ptr, ly.small[2].ptr, ly.small[3].ptr);
                    let row = (cd + vd) as i32;
                    self.lin(&ly.mats[0], self.h.ptr, n, self.mix.ptr)?;
                    self.mark("linear attention: qkvz product")?;
                    self.lin(&ly.mats[1], self.h.ptr, n, self.ba.ptr)?;
                    self.mark("linear attention: ba product")?;
                    self.k35.conv(s, self.mix.ptr, row, sw, self.convst.ptr + ly.idx as u64 * conv_layer, self.xc.ptr, n as i32, cd as i32)?;
                    self.mark("linear attention: convolution")?;
                    self.k35.delta(s, chunked, self.xc.ptr, cd as i32, self.ba.ptr, 2 * nv, sa, sd, self.ssm.ptr + ly.idx as u64 * ssm_layer, self.core.ptr, n as i32, c.lin_k_heads as i32, nv)?;
                    self.mark("linear attention: delta rule")?;
                    self.k35.gated_norm(s, self.core.ptr, self.mix.ptr + (cd * 2) as u64, row, sn, nu, nv, eps)?;
                    self.mark("linear attention: gated norm")?;
                    self.lin(&ly.mats[2], self.core.ptr, n, self.o.ptr)?;
                    self.mark("linear attention: out product")?;
                }
                Kind::Full => {
                    let (kc, vc) = (self.kc.ptr + ly.idx as u64 * kv_layer, self.vc.ptr + ly.idx as u64 * kv_layer);
                    self.lin(&ly.mats[0], self.h.ptr, n, self.mix.ptr)?;
                    self.mark("attention: qkv product")?;
                    self.k35.prep(s, self.mix.ptr, full_w, ly.small[0].ptr, ly.small[1].ptr, self.w.cos.ptr, self.w.sin.ptr, (c.rotary_dim / 2) as i32, pos, self.qo.ptr, kc, vc, nu, nq, nk, eps)?;
                    self.mark("attention: q/k norm, rotary, cache write")?;
                    if n == 1 {
                        self.k35.attn_decode(s, self.qo.ptr, nq * 256, self.mix.ptr, full_w, kc, vc, pos, self.att.ptr, nq * 256, 1, nq, nk, self.attn_part.ptr)?;
                    } else {
                        self.k35.attn(s, self.qo.ptr, nq * 256, self.mix.ptr, full_w, kc, vc, pos, self.att.ptr, nq * 256, n as i32, nq, nk)?;
                    }
                    self.mark("attention: attention")?;
                    self.lin(&ly.mats[1], self.att.ptr, n, self.o.ptr)?;
                    self.mark("attention: o product")?;
                }
            }
            let mlp = if ly.kind == Kind::Full { 2 } else { 3 };
            self.k35.rmsnorm_add(s, self.x.ptr, self.o.ptr, ly.post_norm.ptr, self.h.ptr, nu, hid_i, eps)?;
            self.mark("norms")?;
            self.gate_up_silu(&ly.mats[mlp], n)?;
            self.mark("MLP: gate and up product, SiLU")?;
            self.lin(&ly.mats[mlp + 1], self.m.ptr, n, self.d.ptr)?;
            self.mark("MLP: down product")?;
        }
        if head || dump.is_some() {
            let last = ((n - 1) * hid * 2) as u64;
            self.k35.rmsnorm_add(s, self.x.ptr + last, self.d.ptr + last, self.w.norm.ptr, self.hlast.ptr, 1, hid_i, eps)?;
            if let Some(out) = dump.as_mut() {
                self.dev.sync()?;
                let mut v = vec![0u16; hid];
                self.hlast.read(0, &mut v)?;
                out.push(v);
            }
            self.mark("norms")?;
            self.lin_packed(self.w.lm_head(), self.hlast.ptr, 1, self.logits.ptr)?;
            self.mark("LM head product")?;
            self.kern.argmax_rows(s, self.logits.ptr, c.vocab as i32, 1, self.tok.ptr, self.val.ptr)?;
            self.mark("argmax")?;
        }
        Ok(())
    }

    /// The decode step as a graph (the position read from `st`).
    fn capture_graph(&mut self) -> Result<()> {
        self.st.write(0, &[0i32])?;
        // one eager step first (lazy initializations out of the capture); it writes position 0 and the states, which `reset` and a prompt rewrite
        // SAFETY: live buffers of the engine; the stream is synchronized after.
        unsafe {
            self.embed(&[0u32])?;
            self.forward(1, self.st.ptr, true, None)?;
        }
        self.dev.sync()?;
        let g = self.dev.capture(|| {
            // SAFETY: as above; nothing runs during a capture.
            unsafe { self.forward(1, self.st.ptr, true, None) }
        })?;
        self.graph = Some(g);
        self.reset()
    }

    /// The token chosen after the work queued (the largest logit); with `logprob`, its log-probability.
    fn read_sample(&mut self, logprob: bool) -> Result<Sample> {
        // SAFETY: the pinned buffer holds 4096 bytes; the copies complete at the synchronize.
        unsafe {
            copy_d2h_async(self.host.ptr, self.tok.ptr, 4, self.dev.stream)?;
            copy_d2h_async((self.host.ptr as *mut u8).add(8) as *mut _, self.val.ptr, 4, self.dev.stream)?;
            self.top.queue(&self.kern, &self.dev, self.logits.ptr, self.val.ptr)?;
        }
        self.dev.sync()?;
        // SAFETY: the pinned bytes were just written.
        let (token, logit) = unsafe { (*(self.host.ptr as *const i32), *((self.host.ptr as *const u8).add(8) as *const f32)) };
        let mut s = Sample { token: token as u32, logit, logprob: None };
        if logprob {
            // SAFETY: live buffers; the lse is read after the synchronize.
            unsafe { self.kern.logsumexp(self.dev.stream, self.logits.ptr, self.w.cfg.vocab as i32, logit, self.lse.ptr)? };
            self.dev.sync()?;
            let mut l = [0f32; 1];
            self.lse.read(0, &mut l)?;
            s.logprob = Some(logit - l[0]);
        }
        Ok(s)
    }

    /// The prompt through the model after the tokens already in the cache; the token that follows it. `dump`: as `forward`'s, on the last chunk.
    pub fn prefill_dump(&mut self, tokens: &[u32], logprob: bool, mut dump: Option<&mut Vec<Vec<u16>>>) -> Result<Sample> {
        if tokens.is_empty() {
            return err("an empty prompt");
        }
        if self.n + tokens.len() + 1 > self.ctx {
            return err(format!("{} tokens and {} more, where the cache holds {}", self.n, tokens.len(), self.ctx));
        }
        let mut at = 0;
        while at < tokens.len() {
            let n = self.chunk.min(tokens.len() - at);
            let pos: Vec<i32> = (0..n).map(|i| (self.n + i) as i32).collect();
            self.dev.sync()?; // (the previous chunk's kernels read the positions until then)
            self.pos.write(0, &pos)?;
            let last = at + n == tokens.len();
            // SAFETY: live buffers of the engine, sized for a chunk; the stream orders every step.
            unsafe {
                self.embed(&tokens[at..at + n])?;
                self.forward(n, self.pos.ptr, last, if last { dump.take() } else { None })?;
            }
            self.n += n;
            at += n;
        }
        self.read_sample(logprob)
    }

    pub fn prefill(&mut self, tokens: &[u32], logprob: bool) -> Result<Sample> {
        self.prefill_dump(tokens, logprob, None)
    }

    /// One decode step: `token` at the next position; the token that follows it.
    pub fn step(&mut self, token: u32, logprob: bool) -> Result<Sample> {
        if self.n + 1 > self.ctx {
            return err(format!("the cache holds {} positions", self.ctx));
        }
        let p = [self.n as i32];
        // SAFETY: pinned staging copied before the launches in stream order; live buffers.
        unsafe {
            let hp = (self.host.ptr as *mut u8).add(64);
            std::ptr::copy_nonoverlapping(p.as_ptr() as *const u8, hp, 4);
            copy_h2d_async(self.st.ptr, hp as *const _, 4, self.dev.stream)?;
            self.embed(&[token])?;
            match &self.graph {
                Some(g) => g.launch(self.dev.stream)?,
                None => self.forward(1, self.st.ptr, true, None)?,
            }
        }
        self.n += 1;
        self.read_sample(logprob)
    }

    /// Greedy tokens: the prompt, then up to `max_new` tokens (stopping at an end-of-sequence token unless `ignore_eos`).
    pub fn generate(&mut self, prompt: &[u32], max_new: usize, ignore_eos: bool, logprob: bool) -> Result<Vec<Sample>> {
        self.reset()?;
        let mut out = Vec::with_capacity(max_new);
        let mut s = self.prefill(prompt, logprob)?;
        out.push(s);
        while out.len() < max_new {
            if !ignore_eos && self.w.cfg.eos.contains(&(s.token as i64)) {
                break;
            }
            s = self.step(s.token, logprob)?;
            out.push(s);
        }
        Ok(out)
    }
}

/// One user's replies (conv.rs's `Talk`): the prompt from a new sequence, then each token greedy or sampled on the host, from the 64 largest logits the GPU makes behind the step where they are all the draw looks at (`TopGpu`),
/// else from the step's logits, as `Engine::reply` does.
impl crate::conv::Gen for Engine35 {
    fn capacity(&self) -> usize {
        self.ctx
    }

    fn reply(&mut self, prompt: &[u32], o: &crate::model::Reply, each: &mut dyn FnMut(u32) -> bool) -> Result<(Vec<u32>, crate::model::Finish)> {
        use crate::model::Finish;
        if prompt.is_empty() {
            return err("an empty prompt");
        }
        if prompt.len() + 1 > self.ctx {
            return err(format!("{} tokens, where the context holds {}", prompt.len(), self.ctx));
        }
        self.reset()?;
        let greedy = o.params.greedy();
        let mut sampler = crate::sampler::Sampler::new(o.params.clone(), prompt);
        self.top.ask = sampler.top_need();
        let mut s = self.prefill(prompt, false)?;
        let mut out = Vec::new();
        loop {
            let tok = if greedy { s.token } else { sampler.sample_with(self.top.read().as_ref(), || self.logits())? };
            if !o.ignore_eos && (self.w.cfg.eos.contains(&(tok as i64)) || o.stop_ids.contains(&tok)) {
                return Ok((out, Finish::Stop));
            }
            out.push(tok);
            if !each(tok) {
                return Ok((out, Finish::Stopped));
            }
            if out.len() >= o.max_new || self.n + 1 > self.ctx {
                return Ok((out, Finish::Length));
            }
            self.top.ask = sampler.top_need();
            s = self.step(tok, false)?;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every layer's matrices are packs the packer makes or tensors it keeps, each pack taken once (the embedding and the LM head apart): the 2026 dense configs.
    #[test]
    fn the_layers_take_every_pack_once() {
        let all = glyd_gpu::json::parse(include_str!("../tests/fixtures/qwen35_configs.json")).unwrap();
        for name in ["Qwen/Qwen3.5-0.8B", "Qwen/Qwen3.5-2B", "Qwen/Qwen3.5-4B", "Qwen/Qwen3.5-9B", "Qwen/Qwen3.5-27B", "Qwen/Qwen3.6-27B", "Qwen/Qwen3.8-27B"] {
            let c = Config35::parse(&glyd_gpu::json::to_compact(all.get(name).unwrap()), name, None).unwrap();
            let plan = layer_mats(&c);
            assert_eq!(plan.len(), c.layers);
            let packs = c.packs();
            let kept: std::collections::HashSet<String> = c.kept().into_iter().map(|t| t.name).collect();
            let mut used = std::collections::HashSet::new();
            for (l, (srcs, (kind, _))) in plan.iter().zip(c.schedule()).enumerate() {
                assert_eq!(srcs.len(), if kind == Kind::Full { 4 } else { 5 }, "{name} layer {l}");
                for s in srcs {
                    match s {
                        Src::Pack(k) => {
                            assert!(packs.iter().any(|p| &p.key == k), "{name}: {k} is not a pack");
                            assert!(used.insert(k.clone()), "{name}: {k} twice");
                        }
                        Src::Dense(parts, rows, cols) => {
                            assert!(parts.iter().all(|p| kept.contains(p)), "{name}: {parts:?} not kept");
                            assert_eq!((*rows, *cols), (2 * c.lin_v_heads, c.hidden));
                        }
                    }
                }
            }
            let left: Vec<&str> = packs.iter().map(|p| p.key.as_str()).filter(|k| !used.contains(*k)).collect();
            assert_eq!(left, if c.tie { vec!["embed"] } else { vec!["embed", "lm_head"] }, "{name}");
        }
    }
    #[test]
    fn bf16_runtime_parameter_widening_preserves_every_pattern() {
        let bytes: Vec<u8> = (0..=u16::MAX).flat_map(u16::to_le_bytes).collect();
        let out = runtime_parameter(bytes.clone(), "BF16", "F32").unwrap();
        for (w, b) in (0..=u16::MAX).zip(out.chunks_exact(4)) {
            assert_eq!(u32::from_le_bytes(b.try_into().unwrap()), (w as u32) << 16);
        }
        assert_eq!(runtime_parameter(bytes.clone(), "BF16", "BF16").unwrap(), bytes);
        assert!(runtime_parameter(vec![0], "BF16", "F32").is_err());
        assert!(runtime_parameter(vec![0; 4], "F16", "F32").is_err());
    }

    #[test]
    fn development_transform_is_explicit_and_precedes_widening() {
        let mut b = 0x3f8fu16.to_le_bytes().to_vec();
        transform_bf16_bytes(&mut b, None).unwrap();
        assert_eq!(b, 0x3f8fu16.to_le_bytes());
        transform_bf16_bytes(&mut b, Some(|w| w.fill(0x3f80))).unwrap();
        assert_eq!(runtime_parameter(b, "BF16", "F32").unwrap(), 1.0f32.to_le_bytes());
        assert!(transform_bf16_bytes(&mut [0], None).is_err());
    }

}
