//! The Qwen3.5 layout's GPU pieces that the description executor (src/dynamic) and Condor (src/model_qwen4x.rs) run (its kernel table, `Kern35`, is src/kern/qwen35.rs's):
//! a packed matrix (`Mat`) and the model's packed weights (`Weights35`: every matrix the library packs, the small tensors as the checkpoint has them). The hand-written
//! engine that ran them was removed in R3b (#137), after `paritycheck` found the description executor's rows the same bits at every level.
//!
//! A layer: x += mixer(norm1(x)); x += mlp(norm2(x)), the norms zero-centered. Full attention (1 in 4): the qkv product (each query head with its gate), the q/k norms and
//! the rotary on 64 of 256 dims, the causal attention times sigmoid(gate) (the attention kind's kernels, src/dynamic/attn.rs), the o product. Linear attention: the qkvz and ba products, the causal convolution with SiLU,
//! the gated delta rule (a recurrence, or in chunks for a prompt), the gated norm, the out product.

use crate::dev::Buf;
use crate::ffi::{err, Error, Result};
use crate::qwen35::{Config35, Kind, Pack35};
use crate::weights::{
    choose_layout, lossless_manifest, ordered, pack_item_map, read_lossless_parts, read_v7, rope_tables, upload_bytes, upload_packed, v7_manifest, Item,
    Layout, PackBuf,
};
use glyd_gpu::safetensors::Checkpoint;
use std::collections::{HashMap, HashSet};
use std::path::Path;

pub use crate::kern::qwen35::Kern35;

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
                    vec![
                        Src::Pack(format!("{m}.in_proj_qkv")),
                        one(format!("{m}.in_proj_b"), &[("in_proj_b", cfg.lin_v_heads), ("in_proj_a", cfg.lin_v_heads)], &m),
                        Src::Pack(format!("{m}.out_proj")),
                    ]
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
pub(crate) fn bf16_round(x: f32) -> f32 {
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
    if bytes.len() % 2 != 0 {
        return err("odd BF16 byte count");
    }
    if let Some(f) = transform {
        let mut words: Vec<u16> = bytes.chunks_exact(2).map(|b| u16::from_le_bytes([b[0], b[1]])).collect();
        f(&mut words);
        for (b, w) in bytes.chunks_exact_mut(2).zip(words) {
            b.copy_from_slice(&w.to_le_bytes());
        }
    }
    Ok(())
}

fn runtime_parameter(bytes: Vec<u8>, source: &str, wanted: &str) -> Result<Vec<u8>> {
    if source == wanted {
        return Ok(bytes);
    }
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
    pub fn load(
        dir: &Path,
        max_pos: usize,
        threads: usize,
        layout: Option<Layout>,
        cc: (i32, i32),
        total_mem: u64,
        transform: Option<fn(&mut [u16])>,
    ) -> Result<Weights35> {
        Weights35::load_stage(dir, max_pos, threads, layout, cc, total_mem, transform, None)
    }

    /// As `load`, of a stage alone where `stage` is (layers a..b, whether it holds the embedding and the head: a model split across GPUs): another
    /// stage's matrices and tensors are not read, and stand here as empty buffers (never run).
    #[allow(clippy::too_many_arguments)]
    pub fn load_stage(
        dir: &Path,
        max_pos: usize,
        threads: usize,
        layout: Option<Layout>,
        cc: (i32, i32),
        total_mem: u64,
        transform: Option<fn(&mut [u16])>,
        stage: Option<((usize, usize), bool, bool)>,
    ) -> Result<Weights35> {
        // (the layer a tensor or pack belongs to: the number after "layers.")
        let layer_of = |name: &str| name.split("layers.").nth(1).and_then(|r| r.split('.').next()).and_then(|n| n.parse::<usize>().ok());
        let ((la, lb), first, last) = stage.unwrap_or(((0, usize::MAX), true, true));
        let in_stage = |name: &str| layer_of(name).is_none_or(|l| (la..lb).contains(&l));
        let cfg = Config35::load(dir)?;
        let ck = Checkpoint::open(dir).map_err(|e| Error(e.to_string()))?;
        let have: Vec<(String, String, Vec<u64>)> = ck
            .names()
            .map(|n| {
                let (_, t) = ck.get(n).expect("a tensor the checkpoint names");
                (n.to_string(), t.dtype.clone(), t.shape.clone())
            })
            .collect();
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
        let saved: Vec<Vec<(&str, usize)>> = packs
            .iter()
            .map(|p| {
                let each: Vec<(&str, usize)> = p.parts.iter().map(|x| (Pack35::saved(&x.0), x.1)).collect();
                if v7.is_none() {
                    vec![]
                } else if has(each[0].0) && (each.len() == 1 || !has(each[1].0)) {
                    vec![(each[0].0, p.rows())]
                } else if each.iter().all(|x| has(x.0)) {
                    each
                } else {
                    vec![]
                }
            })
            .collect();
        let held: HashSet<String> = saved
            .iter()
            .flatten()
            .map(|x| x.0.to_string())
            .chain(packs.iter().zip(&saved).filter(|(_, s)| s.len() == 1).flat_map(|(p, _)| p.parts.iter().map(|x| Pack35::saved(&x.0).to_string())))
            .collect();
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
        ordered(
            ks.len(),
            threads,
            2 * threads.max(1),
            &|i| {
                let k = ks[i];
                match &v7 {
                    Some((Layout::Tiered, m)) if !saved[k].is_empty() => read_lossless_parts(&ck, m, &saved[k], items[k].cols),
                    Some((_, m)) if !saved[k].is_empty() => read_v7(&ck, m, layout, &saved[k], items[k].cols),
                    _ => pack_item_map(&ck, &items[k], layout, &|w| {
                        if let Some(f) = transform {
                            f(w);
                        }
                    }),
                }
            },
            |i, p| {
                crate::status::STATUS.beat(); // (a sign of life for the program that started the engine)
                packed.insert(packs[ks[i]].key.clone(), upload_packed(p)?);
                Ok(())
            },
        )?;
        // (an empty pack: where another stage holds the matrix)
        let empty = || {
            upload_packed(crate::weights::Packed {
                data: vec![0; 16],
                b1: vec![0; 16],
                b2: vec![0; 16],
                words: crate::ffi::Words::Tiered { blocks: 0, block_base: 0, tiers: [0; 3] },
                rows: 0,
                cols: 0,
            })
        };
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
        let dense: std::collections::HashSet<String> =
            plan.iter().flatten().filter_map(|s| if let Src::Dense(p, ..) = s { Some(p.clone()) } else { None }).flatten().collect();
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
                layers.push(Layer35 {
                    kind,
                    idx,
                    mats: (0..srcs.len()).map(|_| Ok(Mat::Dense(Buf::new(2)?, 0, 0))).collect::<Result<_>>()?,
                    in_norm: Buf::new(4)?,
                    post_norm: Buf::new(4)?,
                    small: (0..n_small).map(|_| Buf::new(4)).collect::<Result<_>>()?,
                });
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
            layers.push(Layer35 {
                kind,
                idx,
                mats: ms,
                in_norm: take(format!("{p}.input_layernorm.weight"))?,
                post_norm: take(format!("{p}.post_attention_layernorm.weight"))?,
                small: sm,
            });
        }
        let embed = match packed.remove("embed") {
            Some(e) => e,
            None if !(first || (last && cfg.tie)) => empty()?,
            None => return err("the embedding is not packed"),
        };
        let head = if cfg.tie {
            None
        } else if last {
            Some(packed.remove("lm_head").ok_or_else(|| Error("the LM head is not packed".into()))?)
        } else {
            Some(empty()?)
        };
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
