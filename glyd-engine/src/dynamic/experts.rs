//! A mixture of experts' block from its layer kind (`runtime::kinds::Moe`) and its roles' tensors: the core's layer (src/moe.rs: the router, the experts, the
//! shared expert and its gate), which `Op::Moe` runs. For any family whose description names its experts: the kind says the router's scores, the experts'
//! activation, whether they work in a latent (with projections of the block's own where the latent has no norm) and whether each one's output is normed, and
//! whether the shared expert is gated; the roles say where each matrix is, a tensor (or its rows) an expert or one tensor of them all stacked. The experts and
//! the shared expert are held at the level asked: a lossless level's stacked and packed as one matrix each (the packer's check: it decodes to the weights bit
//! for bit), a block level's (kestrel, swift, wren) an expert at a time, relu squared experts' second matrices in the level's row layout (the sparse product);
//! a glyd-v7 directory's packs as saved. A stage can hold them otherwise (`Held`): this GPU's slice of each expert's rows where every layer is across a group
//! of GPUs, a hummingbird directory's codebook rows, the expert cache's places.

use super::{MoeOp, ResidentMoe, Slot};
use crate::dev::{Buf, Dev};
use crate::ffi::{err, Error, Result, Stream, Words};
use crate::hb_pack::HbDir;
use crate::hbg_store::HbCaching;
use crate::kern35::Mat;
use crate::kern_nh::{MoeBlock, NhHost};
use crate::launch;
use crate::moe::{Act, ExpertStack, Joined, MoeCfg, MoeLayer, MoeScratch, MoeWeights, Router, Scoring, Shared};
use crate::moe_rows::{Down, DownHost, RowsJoined};
use crate::offload::{ExpertStore, HostHalf, PinnedPack};
use crate::qwen35::Pack35;
use crate::runtime::describe::{Described, Source};
use crate::runtime::kinds::{self, Body, Ffn, NormKind, NormWeight, RouterScore};
use crate::weights::{as_bytes, item_words, ordered, pack_words, read_pack, upload_bytes, upload_packed, Item, Layout, PackBuf, Packed};
use glyd_gpu::json::Value;
use glyd_gpu::pack::Tiered;
use glyd_gpu::safetensors::Checkpoint;
use std::rc::Rc;

/// The norm of each expert's output before it is weighed, where the kind has one (`Latent::expert_norm`).
fn normed(m: &kinds::Moe) -> Option<kinds::Norm> {
    m.latent.as_ref().and_then(|l| l.expert_norm)
}

/// The core layer's shape and arithmetic for the kind `m` over a model `hidden` wide, its experts `inter` wide: the router reads the model's width, the experts
/// the latent's where the kind has one. Experts whose outputs are normed take each pair's activation from a kernel of its own, rounded (never the first
/// product's), and their router's logits times the kind's scale. What the core's layer does not run is refused in words: hash routing and grouped choices are
/// other operations' (`MoeOp`'s variants, a family's own block), a logit scale the normed experts' alone.
pub fn cfg(m: &kinds::Moe, hidden: usize, inter: usize) -> Result<MoeCfg> {
    let r = &m.router;
    if r.hash || r.groups.is_some() {
        return err(format!("experts of {m:?}: hash routing and grouped choices are not the resident block's"));
    }
    if r.logit_scale != 1.0 && normed(m).is_none() {
        return err(format!("experts of {m:?}: a logit scale is the router's of experts whose outputs are normed"));
    }
    let scoring = match r.score {
        RouterScore::Softmax => Scoring::Softmax,
        RouterScore::Sigmoid => Scoring::Sigmoid,
        RouterScore::SqrtSoftplus => Scoring::SqrtSoftplus,
    };
    if r.bias && scoring == Scoring::Softmax {
        return err("a softmax router with a choice bias: the kernels add the bias to sigmoid and sqrt-softplus scores");
    }
    if m.expert.bias || m.shared.is_some_and(|s| s != m.expert) {
        return err(format!("experts of {:?} beside a shared expert of {:?}: the kernels take one activation and no biases", m.expert, m.shared));
    }
    let act = match (m.expert.act, m.expert.gated, normed(m)) {
        (kinds::Act::Silu, true, Some(n)) if n.kind == NormKind::Rms && n.weight == NormWeight::InputPlusOne && m.shared.is_none() => {
            Act::SwiGlu { limit: Some(m.expert.clamp.unwrap_or(f32::INFINITY)) }
        }
        (_, _, Some(n)) => {
            return err(format!(
                "experts of {:?} whose outputs take {n:?}: the kernels norm SwiGLU experts' outputs, input-scaled, with no shared expert",
                m.expert
            ))
        }
        (kinds::Act::Silu, true, None) => Act::SwiGlu { limit: m.expert.clamp },
        (kinds::Act::GeluTanh, true, None) if m.expert.clamp.is_none() => Act::GeGlu,
        (kinds::Act::Relu2, false, None) if m.expert.clamp.is_none() => Act::Relu2,
        (kinds::Act::Situ { beta, linear }, true, None) if m.expert.clamp.is_none() => Act::Situ { beta, linear },
        _ => return err(format!("experts of {:?}: the kernels take SwiGLU, GeGLU, relu squared and SiTU", m.expert)),
    };
    // (a softmax router is a bf16 product, its logits rounded before the scores; the others' scores are made in fp32)
    let router = Router { scoring, top_k: r.top_k, renorm: r.renorm, scale: r.scale, bf16_logits: scoring == Scoring::Softmax };
    Ok(MoeCfg { experts: m.experts, router_hidden: hidden, hidden: m.latent.as_ref().map_or(hidden, |l| l.dim), inter, router, act })
}

/// The operation of block `block` whose experts work in a latent and whose outputs are normed: the router reads H by its fp32 weights, its logits times `mult`
/// (the kind's `Router::logit_scale`), the experts the normed latent Xc, each expert's output normed (input-scaled by the small `post_w`, eps) before it is
/// weighed, their sum into M (`MoeLayer::forward_normed`).
pub(crate) fn latent_normed(block: usize, post_w: usize, eps: f32, mult: f32) -> MoeOp {
    MoeOp { x: Slot::Xc, y: Slot::M, logit_scale: mult, expert_norm: Some((post_w, eps)), ..MoeOp::of(block) }
}

/// A block of experts' place in its model: its decoder layer (a hummingbird directory's record is its, src/hb_pack.rs) and its number among the model's blocks
/// of experts (its layer in the expert cache: the next block of experts is the next layer there, which a predictor hints).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Ix {
    pub layer: usize,
    pub moe: usize,
}

/// The blocks of experts of the decoder layers `layers` of a model of `per` blocks a layer (two: a mixer, then its feed-forward block; one: each block a layer
/// of its own), in `Op::Moe`'s order: each one's place, its kind and its roles.
pub fn of_layers(d: &Described, layers: (usize, usize), per: usize) -> impl Iterator<Item = (Ix, &kinds::Moe, &[(String, Source)])> {
    let per = per.max(1);
    d.desc
        .blocks
        .iter()
        .zip(&d.blocks)
        .enumerate()
        .filter_map(move |(b, (blk, roles))| match &blk.body {
            Body::Ffn(Ffn::Moe(m)) => Some((b / per, m, roles.as_slice())),
            _ => None,
        })
        .enumerate()
        .filter_map(move |(moe, (layer, m, roles))| (layers.0..layers.1).contains(&layer).then_some((Ix { layer, moe }, m, roles)))
}

/// The blocks of experts of draft head `i` of `d` (`Described::mtp`), in order: each one's block number in the head, its kind and its roles.
pub fn of_head(d: &Described, i: usize) -> impl Iterator<Item = (usize, &kinds::Moe, &[(String, Source)])> {
    let (blocks, roles) = (d.desc.mtp.get(i).map_or(&[][..], |h| &h.blocks[..]), d.mtp.get(i).map_or(&[][..], |r| &r.0[..]));
    blocks.iter().zip(roles).enumerate().filter_map(|(b, (blk, r))| match &blk.body {
        Body::Ffn(Ffn::Moe(m)) => Some((b, m, r.as_slice())),
        _ => None,
    })
}

/// How a stage holds its blocks of experts beyond their level: a hummingbird directory's (src/hb_pack.rs; None: the release's), the expert cache (the stage's
/// caching and its store: src/hbg_store.rs, src/tiering.rs; a cache of as many places as a layer has experts holds them all on the GPU), on the GPU `ordinal`;
/// FP8 matrices as released where `f8` is given (`F8`); the block's matrices besides the routed experts' (the shared expert's, a latent's own projections) at
/// `dense` where it is given, else at the experts' level; a glyd-v7 directory's codebook rows at its hummingbird level sharing `codebooks`' scratch (src/moe_hb.rs).
/// A layer of a stage so held is across the thread's group of GPUs where it has one (src/dynamic/tp.rs): this GPU's rows of each expert's.
#[derive(Clone, Copy)]
pub struct Held<'a> {
    pub hb: Option<&'a HbDir>,
    pub cache: Option<(&'a HbCaching, &'a Rc<ExpertStore>)>,
    pub ordinal: i32,
    pub f8: Option<F8<'a>>,
    pub dense: Option<Layout>,
    pub codebooks: Option<&'a Rc<crate::moe_hb::HbShared>>,
}

/// The experts and the shared expert held in the library's FP8 format as released (src/f8.rs: E4M3 bytes bit for bit, their 128 x 128 block scales), where the
/// description names each matrix's scales (`Described::scales`) and the level is the lossless one; packed on `dev` by `lib`. Elsewhere (no such scales, a
/// block level, none given) an FP8 matrix is held as the bf16 it dequantises to.
#[derive(Clone, Copy)]
pub struct F8<'a> {
    pub scales: &'a std::collections::BTreeMap<String, String>,
    pub dev: &'a Dev,
    pub lib: &'a crate::ffi::Lib,
}

impl F8<'_> {
    /// The matrix of `members` stacked by rows as the FP8 format takes it: its rows and columns, where every member is E4M3 with block scales of 128 x 128 that
    /// the description names, and the rows of each and the columns are 128s; None otherwise.
    fn shape(&self, ck: &Checkpoint, members: &[String]) -> Result<Option<(usize, usize)>> {
        let (mut rows, mut cols) = (0, 0);
        for n in members {
            let (Some((_, t)), Some(s)) = (ck.get(n), self.scales.get(n)) else { return Ok(None) };
            let Some((_, st)) = ck.get(s) else { return Ok(None) };
            let (r, c) = match t.shape[..] {
                [r, c] => (r as usize, c as usize),
                _ => return Ok(None),
            };
            if t.dtype != "F8_E4M3"
                || !r.is_multiple_of(128)
                || !c.is_multiple_of(128)
                || (cols != 0 && c != cols)
                || st.shape != [r as u64 / 128, c as u64 / 128]
            {
                return Ok(None);
            }
            (rows, cols) = (rows + r, c);
        }
        Ok((rows > 0).then_some((rows, cols)))
    }

    /// `members`' E4M3 bytes stacked by rows and their fp32 block scales ([rows / 128, cols / 128] each, stacked alike), as `shape` takes them.
    fn read(&self, ck: &Checkpoint, members: &[String]) -> Result<(Vec<u8>, Vec<f32>)> {
        let (mut q, mut sc) = (Vec::new(), Vec::new());
        for n in members {
            let s = self.scales.get(n).ok_or_else(|| Error(format!("{n}: no block scales named")))?;
            q.extend(io(ck.read(n))?);
            sc.extend(crate::kern_glm::f32_bytes(ck, s)?.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)));
        }
        Ok((q, sc))
    }

    /// The matrix of `members` stacked by rows ([rows, cols], `shape`'s) packed: this GPU's slice of its rows, whole blocks of 128 for the scales, where every layer
    /// is across a group of GPUs (`tp_rows`).
    fn mat(&self, ck: &Checkpoint, members: &[String], rows: usize, cols: usize) -> Result<crate::f8::F8Mat> {
        let (q, sc) = self.read(ck, members)?;
        let (r0, rn) = f8_tp_rows(rows)?;
        let gc = cols.div_ceil(128);
        let mut m = crate::f8::F8Mat::pack(
            self.dev,
            self.lib,
            &q[r0 * cols..(r0 + rn) * cols],
            &sc[r0 / 128 * gc..(r0 + rn) / 128 * gc],
            rn,
            cols,
            cfg!(feature = "dev"),
        )?;
        if rn < rows {
            m.tp_rows = rows;
        }
        Ok(m)
    }

    /// Every expert's matrix (`each`: an expert's members, [rows, cols] stacked) stacked and packed as one (`F8Mat::moe`): this GPU's slice of each expert's
    /// rows where every layer is across a group of GPUs.
    fn stack(&self, ck: &Checkpoint, each: &[Vec<String>], rows: usize, cols: usize) -> Result<crate::f8::F8Mat> {
        let (r0, rn) = f8_tp_rows(rows)?;
        let gc = cols.div_ceil(128);
        let (mut q, mut sc) = (Vec::with_capacity(each.len() * rn * cols), Vec::with_capacity(each.len() * rn / 128 * gc));
        for m in each {
            let (eq, es) = self.read(ck, m)?;
            q.extend_from_slice(&eq[r0 * cols..(r0 + rn) * cols]);
            sc.extend_from_slice(&es[r0 / 128 * gc..(r0 + rn) / 128 * gc]);
        }
        let mut m = crate::f8::F8Mat::pack(self.dev, self.lib, &q, &sc, each.len() * rn, cols, cfg!(feature = "dev"))?;
        if rn < rows {
            m.tp_rows = rows;
        }
        Ok(m)
    }

    /// Every expert's matrix packed on its own for the expert cache's places (`F8Places`, src/tiering.rs), on one GPU.
    fn places(&self, ck: &Checkpoint, each: &[Vec<String>], rows: usize, cols: usize) -> Result<crate::f8::F8Places> {
        if f8_tp_rows(rows)?.1 < rows {
            return err("FP8 experts in the expert cache: on one GPU a layer, not sliced across GPUs");
        }
        crate::f8::F8Places::pack(
            self.dev,
            self.lib,
            each.len(),
            rows,
            cols,
            &mut |a, b| {
                let (mut q, mut sc) = (Vec::new(), Vec::new());
                for m in &each[a..b] {
                    let (eq, es) = self.read(ck, m)?;
                    q.extend(eq);
                    sc.extend(es);
                }
                Ok((q, sc))
            },
            cfg!(feature = "dev"),
        )
    }
}

/// This GPU's rows of an FP8 matrix of `rows` where it is a GPU of a group (src/dynamic/tp.rs): its slice in whole blocks of 128 rows (the scales'); all of
/// them on a GPU on its own.
fn f8_tp_rows(rows: usize) -> Result<(usize, usize)> {
    match super::tp::slicer() {
        None => Ok((0, rows)),
        Some(s) if rows.is_multiple_of(128 * s.n) => Ok(s.rows(rows)),
        Some(s) => err(format!("an FP8 matrix of {rows} rows across {} GPUs: each GPU's rows whole blocks of 128 are wanted", s.n)),
    }
}

/// A role's experts as the checkpoint holds them: a tensor (or its rows stacked) an expert, or one tensor of them all [experts, rows, cols].
enum Stacked<'a> {
    Each(&'a [Vec<String>]),
    One(&'a str),
}

impl Stacked<'_> {
    fn of<'a>(src: Option<&'a Source>, experts: usize, role: &str) -> Result<Stacked<'a>> {
        match src {
            Some(Source::Experts(v)) if v.len() == experts && experts > 0 => Ok(Stacked::Each(v)),
            Some(Source::Tensor(n)) => Ok(Stacked::One(n)),
            other => {
                err(format!("the experts' role {role} is {other:?}, where a tensor (or its rows) an expert is, {experts} of them, or one tensor of them all"))
            }
        }
    }

    /// The name in an error.
    fn name(&self) -> String {
        match self {
            Stacked::Each(v) => v.iter().flatten().cloned().collect::<Vec<_>>().join(" + "),
            Stacked::One(n) => n.to_string(),
        }
    }

    /// Expert `e`'s name in an error.
    fn name_of(&self, e: usize) -> String {
        match self {
            Stacked::Each(v) => v[e].join(" + "),
            Stacked::One(n) => format!("{n}: expert {e}"),
        }
    }

    /// An expert's rows and its columns (where a glyd-v7 directory holds it packed, `v7`, its pack's shape).
    fn dims(&self, ck: &Checkpoint, v7: V7, experts: usize) -> Result<(usize, usize)> {
        match self {
            Stacked::Each(v) => rows_cols(ck, v7, &v[0]),
            Stacked::One(n) => {
                let (_, t) = ck.get(n).ok_or_else(|| Error(format!("no tensor {n}")))?;
                if t.dtype != "BF16" || t.shape.len() != 3 || t.shape[0] as usize != experts {
                    return err(format!("{n}: {} {:?}, where the bf16 matrices of {experts} experts are stacked", t.dtype, t.shape));
                }
                Ok((t.shape[1] as usize, t.shape[2] as usize))
            }
        }
    }

    /// Every expert's matrix, stacked [experts rows, cols]: bf16 words.
    fn words(&self, ck: &Checkpoint, experts: usize) -> Result<Vec<u16>> {
        let (rows, cols) = self.dims(ck, None, experts)?;
        match self {
            Stacked::Each(v) => item_words(ck, &Item { members: v.iter().flatten().cloned().collect(), rows: experts * rows, cols }),
            Stacked::One(n) => {
                let mut w = vec![0u16; experts * rows * cols];
                ck.read_bf16(n, &mut w).map_err(|e| Error(format!("{n}: {e}")))?;
                Ok(w)
            }
        }
    }

    /// Expert `e`'s matrix [rows, cols] (`all`: every expert's, where the checkpoint holds them as one tensor).
    fn expert(&self, ck: &Checkpoint, e: usize, all: &[u16], rows: usize, cols: usize) -> Result<Vec<u16>> {
        match self {
            Stacked::Each(v) => item_words(ck, &Item { members: v[e].clone(), rows, cols }),
            Stacked::One(_) => Ok(all[e * rows * cols..(e + 1) * rows * cols].to_vec()),
        }
    }
}

/// A glyd-v7 directory's level and glyd.json, where the checkpoint is one (`weights::v7_manifest`): the packs it holds are read as saved.
type V7<'a> = Option<&'a (Layout, Value)>;

/// Tensor `n`'s [rows, cols]: its shape, or where a glyd-v7 directory holds it packed (`v7`) its weight's as its pack's entry says (a row layout's pack is the
/// weight's transpose).
fn dims_of(ck: &Checkpoint, v7: V7, n: &str) -> Result<(usize, usize)> {
    if let Some((_, t)) = ck.get(n) {
        return match t.shape[..] {
            [r, c] => Ok((r as usize, c as usize)),
            _ => err(format!("{n}: {:?}, where a matrix is read", t.shape)),
        };
    }
    let key = Pack35::saved(n);
    let e = v7.and_then(|(_, m)| m.get("packs")?.get(key)).ok_or_else(|| Error(format!("{}: no tensor {n} and no pack {key}", ck.dir.display())))?;
    let s = e.get("shape").and_then(Value::as_u64s).filter(|s| s.len() == 2).ok_or_else(|| Error(format!("glyd.json: {key}: no shape")))?;
    let transposed = e.get("layout").and_then(Value::as_str).and_then(crate::weights::entry_layout).is_some_and(|l| l.row_bits().is_some());
    Ok(if transposed { (s[1] as usize, s[0] as usize) } else { (s[0] as usize, s[1] as usize) })
}

/// Each of `members`' [rows, cols] (`dims_of`), where a glyd-v7 directory holds them as one pack under the first's name (a save fuses an expert's gate and
/// up so, `pack::described`): each its tensor's shape as that pack's entry names it.
fn dims_all(ck: &Checkpoint, v7: V7, members: &[String]) -> Result<Vec<(usize, usize)>> {
    let fused = match (v7, members) {
        (Some((_, m)), [first, _, ..]) if ck.get(first).is_none() => m.get("packs").and_then(|p| p.get(Pack35::saved(first))),
        _ => None,
    };
    let Some(e) = fused else { return members.iter().map(|n| dims_of(ck, v7, n)).collect() };
    let tensors = e.get("tensors").and_then(Value::as_array).unwrap_or(&[]);
    let shape = |n: &str| tensors.iter().find(|t| t.get("name").and_then(Value::as_str) == Some(n)).and_then(|t| t.get("shape")).and_then(Value::as_u64s);
    members
        .iter()
        .map(|n| match shape(n).as_deref() {
            Some(&[r, c]) => Ok((r as usize, c as usize)),
            _ => err(format!("glyd.json: the pack {} names no matrix {n}", Pack35::saved(&members[0]))),
        })
        .collect()
}

/// Matrices stacked along their rows: their rows' sum and their columns.
fn rows_cols(ck: &Checkpoint, v7: V7, members: &[String]) -> Result<(usize, usize)> {
    let (mut rows, mut cols) = (0, 0);
    for (n, (r, c)) in members.iter().zip(dims_all(ck, v7, members)?) {
        if cols != 0 && cols != c {
            return err(format!("{n}: {r} x {c}, where a matrix of {cols} columns is stacked"));
        }
        (rows, cols) = (rows + r, c);
    }
    Ok((rows, cols))
}

/// The matrix of `members` stacked by rows at `layout`, as the checkpoint holds it: a glyd-v7 directory's pack as saved, in the layout its entry names (`v7`),
/// else packed here from its weights.
pub(crate) fn pack_of(ck: &Checkpoint, v7: V7, members: &[String], layout: Layout) -> Result<Packed> {
    let (_, cols) = rows_cols(ck, v7, members)?;
    let parts = members.iter().cloned().zip(dims_all(ck, v7, members)?.into_iter().map(|d| d.0)).collect();
    read_pack(ck, v7, &Pack35 { key: Pack35::saved(&members[0]).to_string(), parts, cols }, layout)
}

/// Expert `e`'s matrix of `s` ([rows, cols]; `all` every expert's, where the checkpoint holds them as one tensor) as a pack of `layout` (a row layout: its
/// transpose, a row a neuron), as the checkpoint holds it (`pack_of`).
#[allow(clippy::too_many_arguments)]
fn expert_pack(ck: &Checkpoint, v7: V7, s: &Stacked, e: usize, all: &[u16], rows: usize, cols: usize, layout: Layout) -> Result<Tiered> {
    let p = match s {
        Stacked::Each(v) => pack_of(ck, v7, &v[e], layout),
        Stacked::One(_) => pack_words(&s.expert(ck, e, all, rows, cols)?, rows, cols, layout, &s.name_of(e)),
    }?;
    let Words::Tiered { tiers, .. } = p.words else { return err(format!("{}: an expert that is not in a block or row layout", s.name_of(e))) };
    Ok(Tiered { rows: p.rows, cols: p.cols, data: p.data, blocks: p.b1, block_base: Vec::new(), tiers })
}

/// A block level's experts (`up`, `down`: their roles' tensors, `e` of them), each expert's two packs at `level` joined (`Joined`, `DownHost`), as the checkpoint
/// holds them (`pack_of`): where the experts' activation is relu squared (`act`), their second matrices in the level's row layout, the transpose a row a neuron
/// (the sparse product skips the neurons relu squared leaves at zero, src/moe_rows.rs; a development build's GLYD_ENGINE_DENSE_DOWN keeps the block layout).
/// Where the checkpoint holds a matrix of every expert as one tensor, it is read once and cut per expert.
fn block_packs(up: &Stacked, down: &Stacked, ck: &Checkpoint, v7: V7, e: usize, level: Layout, act: kinds::Act) -> Result<(Joined, DownHost)> {
    let threads = std::thread::available_parallelism().map_or(8, |n| n.get().min(16));
    let all = |s: &Stacked| match s {
        Stacked::One(_) => s.words(ck, e),
        Stacked::Each(_) => Ok(Vec::new()),
    };
    let (all_up, all_down) = (all(up)?, all(down)?);
    let (du, dd) = (up.dims(ck, v7, e)?, down.dims(ck, v7, e)?);
    let rows = act == kinds::Act::Relu2 && crate::tune!("GLYD_ENGINE_DENSE_DOWN").is_none();
    let down_layout = if rows { level.row_of().unwrap_or(level) } else { level };
    let (mut ups, mut downs): (Vec<Tiered>, Vec<Tiered>) = (Vec::new(), Vec::new());
    ordered(
        2 * e,
        threads,
        2 * threads,
        &|k| {
            let (s, all, (r, c), layout) = if k % 2 == 0 { (up, &all_up, du, level) } else { (down, &all_down, dd, down_layout) };
            expert_pack(ck, v7, s, k / 2, all, r, c, layout)
        },
        |k, t| {
            (if k % 2 == 0 { &mut ups } else { &mut downs }).push(t);
            Ok(())
        },
    )?;
    Ok((Joined::of(&ups.iter().collect::<Vec<_>>())?, DownHost::of(&downs)?))
}

/// The host's half of a layer in the expert cache whose second matrices are in a row layout (`down`): the host computes the pairs of the experts that have no
/// place from the rows of both matrices (src/cpu_experts.rs), the first's (`up`) a copy in rows the store holds.
fn rows_half(up: &Joined, down: &RowsJoined) -> Result<HostHalf> {
    let (data, scales) = crate::cpu_experts::rows_of_experts(up.tag, up.experts, up.rows, up.cols, &up.data, &up.scales, 0)?;
    let bits = glyd_gpu::pack::rows_bits(down.tag).ok_or_else(|| Error(format!("a second matrix of tag {}: not a row layout", down.tag)))?;
    Ok(HostHalf::Rows(crate::offload::CpuRows { bits, inter: up.rows, hidden: up.cols, data, scales }))
}

/// A router's vector read as fp32 bytes (the choice bias; the router of experts whose outputs are normed): from an fp32 tensor as it is, from a bf16 one (a
/// glyd-v7 directory's) widened exactly.
pub(crate) fn bias_f32(dtype: &str, bytes: &[u8]) -> Result<Vec<u8>> {
    match dtype {
        "F32" => Ok(bytes.to_vec()),
        "BF16" => Ok(bytes.chunks_exact(2).flat_map(|b| ((u16::from_le_bytes([b[0], b[1]]) as u32) << 16).to_le_bytes()).collect()),
        other => err(format!("a router's vector of {other}, where F32 or BF16 is read")),
    }
}

/// The block of the kind `m` whose tensors `roles` names, read from `ck`, its experts and shared expert at `level`, every expert on the GPU.
pub fn resident(m: &kinds::Moe, roles: &[(String, Source)], hidden: usize, ck: &Checkpoint, level: Layout) -> Result<MoeLayer> {
    held(m, roles, hidden, ck, level, None)
}

/// `resident`, the experts of the block at `at.0` held as the stage holds them (`at.1`): a hummingbird directory's in place of the release's (in the expert
/// cache where the stage has one), the release's in the expert cache's places (the cache's layer the block's number among the blocks of experts), or on the
/// GPU, this GPU's rows of each expert's where the layer is across a group of GPUs. At the lossless level, experts released in FP8 whose scales the description
/// names are held as released where the stage asks (`Held::f8`: in the cache's places each packed on its own), and the shared expert too where its matrices are
/// FP8 (as it is beside a hummingbird directory's experts); other experts in the cache are each stack's one lossless pack in the tiered layout cut per expert,
/// or at a block level each expert's packs (`block_packs`: relu squared experts' second matrices in a row layout, which the host computes misses from where the
/// store's mode has it compute). A glyd-v7 directory's packs are read as saved, at its own level (no lossless level of its experts is read).
pub fn held(m: &kinds::Moe, roles: &[(String, Source)], hidden: usize, ck: &Checkpoint, level: Layout, at: Option<(Ix, &Held)>) -> Result<MoeLayer> {
    let lossless = matches!(level, Layout::Tiered | Layout::Twelve);
    if !lossless && !matches!(level, Layout::Int6 | Layout::Int5 | Layout::Int4) {
        return err(format!("the experts are held at the penguin, kestrel, swift and wren levels: {} is asked", level.name()));
    }
    let v7 = crate::weights::v7_manifest(&ck.dir)?;
    if let Some((got, _)) = &v7 {
        if *got != level || lossless {
            return err(format!(
                "{}: packed in the {} layout, where the experts are asked at {} (a glyd-v7 directory's experts are read at its own block level)",
                ck.dir.display(),
                got.name(),
                level.name()
            ));
        }
    }
    let v7 = v7.as_ref();
    let find = |r: &str| roles.iter().find(|(n, _)| n == r).map(|(_, s)| s);
    let one = |r: &str| match find(r) {
        Some(Source::Tensor(n)) => Ok(n.clone()),
        other => err(format!("the experts' role {r} is {other:?}, where one tensor is")),
    };
    let read = |n: &str| ck.read(n).map_err(|e| Error(format!("{n}: {e}")));
    let dtype = |n: &str| ck.get(n).map(|(_, t)| t.dtype.clone()).ok_or_else(|| Error(format!("no tensor {n}")));
    let bf16 = |n: &str| -> Result<Vec<u8>> {
        if dtype(n)? != "BF16" {
            return err(format!("{n} is {}, where the router and the shared expert's gate are read as BF16", dtype(n)?));
        }
        read(n)
    };
    let (e, gated, norm) = (m.experts, m.expert.gated, normed(m));
    let up_role = if gated { "experts_gate_up" } else { "experts_up" };
    let (up, down) = (Stacked::of(find(up_role), e, up_role)?, Stacked::of(find("experts_down"), e, "experts_down")?);
    // (the experts' width: a hummingbird directory's record's, which holds no matrix of theirs; else their first matrix's rows)
    let inter = match at.and_then(|(x, h)| h.hb.map(|d| (d, x.layer))) {
        Some((d, l)) => d.layer(2 * l + 1).ok_or_else(|| Error(format!("{}: no experts for decoder layer {l}", d.dir.display())))?.inter,
        None => {
            let (rows, _) = up.dims(ck, v7, e)?;
            if gated {
                rows / 2
            } else {
                rows
            }
        }
    };
    let c = cfg(m, hidden, inter)?;
    // (the shared expert's matrices at the stage's layout of the block's other matrices, where it gives one)
    let dense = at.and_then(|(_, h)| h.dense).unwrap_or(level);
    let packed = |members: Vec<String>| -> Result<PackBuf> { upload_packed(pack_of(ck, v7, &members, dense)?) };
    // (the normed experts' router is read by its fp32 weights, `MoeLayer::forward_normed`; another's is a bf16 product)
    let router = {
        let n = one("router")?;
        match norm {
            Some(_) => upload_bytes(&bias_f32(&dtype(&n)?, &read(&n)?)?)?,
            None => upload_bytes(&bf16(&n)?)?,
        }
    };
    let bias = if m.router.bias {
        let n = one("choice_bias")?;
        Some(upload_bytes(&bias_f32(&dtype(&n)?, &read(&n)?)?)?)
    } else {
        None
    };
    // (the experts as released in FP8, where the stage asks and the description names their scales: at the lossless level, each expert's [rows, cols] a
    // multiple of 128s; `F8`)
    let f8 = at.and_then(|(_, h)| h.f8);
    let f8x = match (f8, &up, &down) {
        (Some(f), Stacked::Each(u), Stacked::Each(d)) if lossless => match (f.shape(ck, &u[0])?, f.shape(ck, &d[0])?) {
            (Some(a), Some(b)) => Some((f, *u, *d, a, b)),
            _ => None,
        },
        _ => None,
    };
    // (the shared expert: FP8 as released where the experts are, or are a hummingbird directory's codes, and its own matrices are; else packed at the level)
    let mut shared8 = None;
    let shared = match m.shared {
        None => None,
        Some(_) => {
            let first = if gated { vec![one("shared_gate")?, one("shared_up")?] } else { vec![one("shared_up")?] };
            // (its width: its first matrices' rows, a gate's and an up's where gated, as a glyd-v7 directory holds them too: one pack)
            let (rows, _) = rows_cols(ck, v7, &first)?;
            let inter = if gated { rows / 2 } else { rows };
            let second = vec![one("shared_down")?];
            let gate = if m.shared_gate { Some(upload_bytes(&bf16(&one("shared_router")?)?)?) } else { None };
            let as_f8 = f8.filter(|_| f8x.is_some() || at.is_some_and(|(_, h)| h.hb.is_some()));
            let shapes = match as_f8 {
                Some(f) => Some((f, f.shape(ck, &first)?, f.shape(ck, &second)?)),
                None => None,
            };
            match shapes {
                Some((f, Some((ur, uc)), Some((dr, dc)))) => {
                    shared8 = Some((f.mat(ck, &first, ur, uc)?, f.mat(ck, &second, dr, dc)?));
                    Some(Shared { inter, up: PackBuf::shape_only(ur, uc)?, down: PackBuf::shape_only(dr, dc)?, gate })
                }
                _ => Some(Shared { inter, up: packed(first)?, down: packed(second)?, gate }),
            }
        }
    };
    let with8 = |l: MoeLayer, s: Option<(crate::f8::F8Mat, crate::f8::F8Mat)>| match s {
        Some((u, d)) => l.with_shared8(u, d),
        None => Ok(l),
    };
    let blocks = || block_packs(&up, &down, ck, v7, e, level, m.expert.act);
    // (a stage that holds its experts its own way has its layers across the thread's group of GPUs, where it is one's)
    let across = at.and_then(|_| super::tp::slicer());
    if let Some((ix, h)) = at {
        if h.hb.is_some() {
            let finish = match norm {
                Some(n) => {
                    // (the host's pairs of the cache normed as the sum norms each expert's output: the norm's input scale as the GPU holds it, bf16)
                    let w: Vec<f32> = match h.cache {
                        Some(_) => {
                            let b = bf16(&one("expert_norm")?)?;
                            let w: Vec<f32> = b.as_chunks::<2>().0.iter().map(|x| crate::hb_codec::f32_of(u16::from_le_bytes(*x))).collect();
                            crate::muse::plus_one(&w).iter().map(|&v| crate::hb_codec::f32_of(crate::hb_codec::bf16(v))).collect()
                        }
                        None => Vec::new(),
                    };
                    crate::cpu_hbg::Finish::Normed { signs: Vec::new(), rot: 0, w, eps: n.eps }
                }
                // (outputs not normed: each pair's rotation undone as its row is summed)
                None => crate::cpu_hbg::Finish::Rotated,
            };
            if let Some(x) = crate::hbg_store::hb_layer(h.hb, ix.layer, across.is_some(), h.cache, c.act, finish, h.ordinal, c.router.top_k)? {
                return with8(x.into_layer(c, router, bias, None, shared)?, shared8);
            }
        }
        // (R12: the release's experts in the expert cache, every one in the computer's memory; places for every expert of the layer hold them all on the GPU)
        if let Some((cg, store)) = h.cache.filter(|(cg, _)| cg.places < e) {
            if across.is_some() || cg.ram.is_some() {
                return err("the release's experts in the expert cache: a layer on one GPU, every expert in the computer's memory (the disk holds hummingbird directories' experts alone)");
            }
            // (FP8 as released: each expert packed on its own, places for some on the GPU, the heads made the place's after the store's copies)
            if let Some((f, u, d, (ur, uc), (dr, dc))) = f8x {
                let (pu, pd) = (f.places(ck, u, ur, uc)?, f.places(ck, d, dr, dc)?);
                let (x, cache) = crate::tiering::cached_f8(&pu, &pd, cg.places, store, ix.moe, h.ordinal)?;
                return with8(MoeLayer::new_f8(c, router, bias, x, shared)?, shared8)?.with_cache(cache);
            }
            // (a block level's: each expert's packs in places of the level's layout; the host computes the misses from rows where the second matrices are in
            // rows and the store's mode has it compute)
            if !lossless {
                let (ju, jd) = blocks()?;
                let cpu = match &jd {
                    DownHost::Rows(r) if store.wants_cpu() => Some(rows_half(&ju, r)?),
                    _ => None,
                };
                let (up, down, cache) = crate::tiering::cached_stacks(&ju, &jd, cg.places, store, ix.moe, h.ordinal, cpu)?;
                return with8(MoeLayer::new(c, MoeWeights { router, bias, escale: None, up, down }, shared)?, shared8)?.with_cache(cache);
            }
            // (lossless bf16: each stack's one lossless pack in the tiered layout, whatever the level of the rest, cut per expert)
            let places = |s: &Stacked| -> Result<crate::tiering::TieredPlaces> {
                let (rows, cols) = s.dims(ck, v7, e)?;
                let p = glyd_gpu::pack::pack_tiered_checked(&s.words(ck, e)?, e * rows, cols)
                    .ok_or_else(|| Error(format!("{}: decoded to other bits than its weights", s.name())))?;
                crate::tiering::TieredPlaces::of(&p, e)
            };
            let (pu, pd) = (places(&up)?, places(&down)?);
            let (up, down, cache) = crate::tiering::cached_tiered(&pu, &pd, cg.places, store, ix.moe, h.ordinal)?;
            return with8(MoeLayer::new(c, MoeWeights { router, bias, escale: None, up, down: Down::Block(down) }, shared)?, shared8)?.with_cache(cache);
        }
    }
    // (FP8 as released, every expert on the GPU: each of the two matrices of all of them stacked and packed as one, this GPU's rows of each expert's)
    if let Some((f, u, d, (ur, uc), (dr, dc))) = f8x {
        let x = crate::f8::F8Experts { up: f.stack(ck, u, ur, uc)?, down: f.stack(ck, d, dr, dc)?, experts: e };
        return with8(MoeLayer::new_f8(c, router, bias, x, shared)?, shared8);
    }
    let (up, down) = if lossless {
        let stack = |s: &Stacked| -> Result<ExpertStack> {
            let (rows, cols) = s.dims(ck, v7, e)?;
            let w = s.words(ck, e)?;
            let Some(t) = across else {
                return ExpertStack::lossless(upload_packed(pack_words(&w, e * rows, cols, level, &s.name())?)?, e);
            };
            // (a group of GPUs: this GPU's rows of each expert's, stacked)
            if rows % (64 * t.n) != 0 {
                return err(format!("{}: experts of {rows} rows across {} GPUs: each GPU's rows in 64s are wanted", s.name(), t.n));
            }
            let (r0, rn) = t.rows(rows);
            let mine: Vec<u16> = (0..e).flat_map(|x| w[(x * rows + r0) * cols..(x * rows + r0 + rn) * cols].iter().copied()).collect();
            let mut st = ExpertStack::lossless(upload_packed(pack_words(&mine, e * rn, cols, level, &s.name())?)?, e)?;
            st.tp_rows = rows;
            Ok(st)
        };
        (stack(&up)?, Down::Block(stack(&down)?))
    } else {
        let (ju, jd) = blocks()?;
        (ExpertStack::upload(&ju)?, jd.upload()?)
    };
    with8(MoeLayer::new(c, MoeWeights { router, bias, escale: None, up, down }, shared)?, shared8)
}

/// A latent of the block's own projections (`kinds::Latent` with no norm): the down projection of the layer's input the experts read, the up projection of
/// their sum back (`latent_down`, `latent_up`), which `Op::Moe` runs around the experts (src/dynamic/mod.rs `moe`). None where the kind has no latent, or
/// one with a norm: its operations are the lowering's (`latent_normed`) or the latent block's (`latent_block`).
fn own_latent(m: &kinds::Moe) -> Option<&kinds::Latent> {
    m.latent.as_ref().filter(|l| l.norm_in.is_none() && l.expert_norm.is_none() && l.norm_out.is_none())
}

/// The Moe kind's block of the layer at `at.0` as `Op::Moe` runs it, its experts held as the stage holds them (`held`): the core's layer (`ResidentMoe`), and
/// around it the latent's own projections where the kind's latent has no norm (`own_latent`: `Projected`), packed at the stage's layout of the block's other
/// matrices (`Held::dense`) as the checkpoint holds them; at the hummingbird level of a glyd-v7 directory, its experts' codebook rows (src/moe_hb.rs `HbMoe`,
/// in the expert cache where the stage has one).
pub fn block(m: &kinds::Moe, roles: &[(String, Source)], hidden: usize, ck: &Checkpoint, level: Layout, at: (Ix, &Held)) -> Result<Box<dyn MoeBlock>> {
    let (ix, h) = at;
    let dense = h.dense.unwrap_or(level);
    if level == Layout::Hummingbird {
        let common = h.codebooks.ok_or_else(|| Error("a glyd-v7 directory's codebook rows without the stage's scratch for them".into()))?;
        let cache = h.cache.filter(|(cg, _)| cg.places < m.experts).map(|(cg, st)| (cg.places, st));
        return Ok(Box::new(crate::moe_hb::HbMoe::load(m, roles, hidden, ck, dense, ix.moe, cache, common)?));
    }
    let fc = match own_latent(m) {
        Some(_) => {
            let v7 = crate::weights::v7_manifest(&ck.dir)?;
            let mat = |r: &str| -> Result<Mat> {
                let n = match roles.iter().find(|(x, _)| x == r).map(|(_, s)| s) {
                    Some(Source::Tensor(n)) => n.clone(),
                    other => return err(format!("the latent's role {r} is {other:?}, where one tensor is")),
                };
                Ok(Mat::Packed(upload_packed(pack_of(ck, v7.as_ref(), &[n], dense)?)?))
            };
            Some((mat("latent_down")?, mat("latent_up")?))
        }
        None => None,
    };
    let layer = held(m, roles, hidden, ck, level, Some(at))?;
    Ok(match fc {
        Some(fc) => Box::new(Projected { layer, fc }),
        None => Box::new(ResidentMoe(layer)),
    })
}

/// A block whose experts work in a latent of its own (`own_latent`): the core's layer and the latent's two projections, which `Op::Moe` runs: the router and the
/// shared expert on the layer's input, the experts on its down projection, the up projection of their sum (src/dynamic/mod.rs `moe`).
struct Projected {
    layer: MoeLayer,
    fc: (Mat, Mat),
}

impl MoeBlock for Projected {
    unsafe fn forward(&self, _: &dyn NhHost, _: u64, _: usize, _: u64) -> Result<()> {
        err("these experts run on the dynamic engine")
    }

    fn graphable(&self) -> bool {
        true
    }

    fn bytes(&self) -> usize {
        self.layer.weights_bytes() + self.fc.0.bytes() + self.fc.1.bytes()
    }

    fn pack_shapes(&self) -> Vec<(i64, i64, bool)> {
        [&self.fc.0, &self.fc.1]
            .iter()
            .filter_map(|m| if let Mat::Packed(p) = m { Some((p.pack.rows, p.pack.cols, p.pack.is_twelve())) } else { None })
            .collect()
    }

    fn resident(&self) -> Option<(&MoeLayer, Option<(&Mat, &Mat)>)> {
        Some((&self.layer, Some((&self.fc.0, &self.fc.1))))
    }
}

fn io<T>(r: std::io::Result<T>) -> Result<T> {
    r.map_err(|e| Error(e.to_string()))
}

/// The latent block's kernels: the latent's norm of the experts' sum with two roundings (KimiRMSNorm: kernels/mla.cu's `rms2_rows512`), engine.cu's copy of a
/// fused product's part, and kernels/kimi.cu's plan and copies of the expert cache on the GPU.
pub struct KernLatent {
    rms2: crate::kt::rms2_rows512,
    rows: crate::kt::parts_rows,
    /// the expert cache's plan and copies on the GPU (src/dynamic/tp_cache.rs's device path)
    pub(crate) dc_plan: crate::kt::dc_plan,
    pub(crate) dc_fill: crate::kt::dc_fill,
}

impl KernLatent {
    pub fn load(dev: &Dev) -> Result<KernLatent> {
        Ok(KernLatent { rms2: dev.typed()?, rows: dev.typed()?, dc_plan: dev.typed()?, dc_fill: dev.typed()? })
    }

    /// y = rmsnorm(x) w, the normed value rounded, then times w and rounded (rows xs and ys apart, k values).
    ///
    /// # Safety
    /// Live device buffers of those sizes, on stream `s`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn rms2(&self, s: Stream, x: u64, xs: usize, w: u64, y: u64, ys: usize, k: usize, eps: f32, n: u32) -> Result<()> {
        launch!(self.rms2, [n, 1, 1], [512, 1, 1], s; x, xs as i32, w, y, ys as i32, k as i32, eps)
    }

    /// dst [n, w] (rows ds apart) = n rows of w values of src (rows ss apart).
    ///
    /// # Safety
    /// As `rms2`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn rows(&self, s: Stream, src: u64, ss: usize, dst: u64, ds: usize, w: usize, n: usize) -> Result<()> {
        launch!(self.rows, [((n * w) as u32).div_ceil(256), 1, 1], [256, 1, 1], s; src, ss as i64, dst, ds as i64, w as i32, n as i32)
    }
}

/// An fp32 vector (A_log, dt_bias, the router's bias), from fp32 or widened exactly from bf16.
fn f32s(ck: &Checkpoint, name: &str) -> Result<Buf> {
    let (_, t) = ck.get(name).ok_or_else(|| Error(format!("no tensor {name}")))?;
    let b = io(ck.read(name))?;
    match t.dtype.as_str() {
        "F32" => upload_bytes(&b),
        "BF16" => upload_bytes(&b.chunks(2).flat_map(|c| ((u16::from_le_bytes([c[0], c[1]]) as u32) << 16).to_le_bytes()).collect::<Vec<u8>>()),
        d => err(format!("{name}: {d}, where fp32 or bf16 is read")),
    }
}

/// The Moe kind's block of experts that work in a latent (`kinds::Latent`) with its own projections: the core's layer (the router over the model's width, the
/// experts on the latent, the shared experts), the latent's down projection (the shared experts' gate and up riding in its product), its norm of the experts'
/// sum (`Latent::norm_out`) and its up projection.
struct LatentMoe {
    layer: MoeLayer,
    /// the latent's down projection and the shared experts' gate and up: one product of the layer's input (a row's latent values, then the gate's and
    /// the up's; zeros after, to a multiple of a GPU's slice); the shared expert's own first matrix a shape alone
    dg: Mat,
    up: Mat,
    norm: Buf,
    latent: usize,
    eps: f32,
    k: Rc<KernLatent>,
}

impl LatentMoe {
    /// The fused product of h into Mix, and where the experts read z, where their sum goes and where the shared expert reads its gate and up: one token's
    /// in place (z at Mix's start, the sum into Xc); a prompt's copied out first (z into Xc, the gate and up into the scratch), the sum into Mix.
    unsafe fn down(&self, e: &dyn NhHost, sc: &MoeScratch, h: u64, n: usize) -> Result<(u64, u64, u64)> {
        let (mix, xc, s) = (e.mix().ptr, e.xc().ptr, e.dev().stream);
        let (row, gw) = (self.dg.whole_rows(), self.layer.shared.as_ref().map_or(0, |x| 2 * x.inter));
        if e.mix().bytes < n * row * 2 || e.xc().bytes < n * self.latent * 2 {
            return err(format!("{n} rows of {row} values do not fit the engine's buffers"));
        }
        e.lin(&self.dg, h, n, mix)?;
        let gu = mix + (self.latent * 2) as u64;
        if n == 1 {
            return Ok((mix, xc, gu));
        }
        let k = &self.k;
        k.rows(s, mix, row, xc, self.latent, self.latent, n)?;
        if gw > 0 {
            k.rows(s, gu, row, sc.sg.ptr, gw, gw, n)?;
        }
        Ok((xc, mix, sc.sg.ptr))
    }
}

impl MoeBlock for LatentMoe {
    /// The router on h; z, and the shared experts' gate and up, of h (`down`); the experts' weighted sum of z; its norm; up of that into out; the shared
    /// MLP's rest added.
    unsafe fn forward(&self, e: &dyn NhHost, h: u64, n: usize, out: u64) -> Result<()> {
        let (c, s) = (&self.layer.cfg, e.dev().stream);
        if e.mix().bytes < n * c.router_hidden * 2 || e.xc().bytes < n * c.router_hidden * 2 {
            return err(format!("{n} rows of {} values do not fit the engine's buffers", c.router_hidden));
        }
        if e.moe_scratch().get().is_none() {
            let _ = e.moe_scratch().set(MoeScratch::new(e.dev(), e.lib(), &[&self.layer], e.chunk())?);
        }
        let sc = e.moe_scratch().get().ok_or_else(|| Error("the mixture-of-experts scratch is not made".into()))?;
        self.layer.route(sc, s, h, c.router_hidden, n)?;
        e.mark("MoE: router");
        let (z, y, gu) = self.down(e, sc, h, n)?;
        e.mark("MoE: latent down and shared gate, up product");
        self.layer.experts(e.lib(), sc, s, z, n, y)?;
        e.mark("MoE: experts");
        self.k.rms2(s, y, self.latent, self.norm.ptr, z, self.latent, self.latent, self.eps, n as u32)?;
        e.mark("MoE: latent norm");
        e.lin(&self.up, z, n, out)?;
        e.mark("MoE: latent up product");
        self.layer.shared_from(e.lib(), sc, s, gu, n)?;
        self.layer.shared_add(sc, s, n, out)
    }

    fn graphable(&self) -> bool {
        !self.layer.reads_host()
    }

    fn reads_host(&self) -> bool {
        self.layer.reads_host()
    }

    fn bytes(&self) -> usize {
        self.layer.weights_bytes() + self.dg.bytes() + self.up.bytes() + self.norm.bytes
    }

    /// Its core layer, its router's choices in the scratch after a call (`forward` routes first), where every expert is on the GPU.
    fn core(&self) -> Option<&MoeLayer> {
        (!self.layer.reads_host()).then_some(&self.layer)
    }

    fn pack_shapes(&self) -> Vec<(i64, i64, bool)> {
        [&self.dg, &self.up].iter().filter_map(|m| if let Mat::Packed(p) = m { Some((p.pack.rows, p.pack.cols, p.pack.is_twelve())) } else { None }).collect()
    }
}

/// A mixture-of-experts layer whose experts are not all on the GPU: the core's layer over `slots` places for them (the expert cache's, src/offload.rs: every expert in
/// the computer's memory, the ones the router chooses copied into the places), the latent's projections and norm. What it computes is `LatentMoe`'s to the bit; a
/// decode step is not captured in a graph (the choices are read on the host to place the experts).
struct LatentCached {
    moe: LatentMoe,
    store: Rc<ExpertStore>,
    index: usize,
    /// (a decode step's guess of the experts, `Op::MoeEarly`: none where no copies are made for it)
    guess: Option<Guess>,
    /// (a development build's GLYD_KIMI_STATS: the stage's first cached layer reports its store every 64 one-token calls; their count and the time of the last report)
    report: Option<(std::cell::Cell<u32>, std::cell::Cell<std::time::Instant>)>,
}

/// A decode step's early guess of a cached layer's experts: at most `copies` of the ones not in places copied for it, its choices read back to `ids` (pinned) after
/// `ev`, and whether one is queued and not yet taken up. A development build's GLYD_KIMI_EARLY of several counts ("0,1,2") takes them in turn, the stage's
/// layers together, a turn for each of GLYD_KIMI_STATS's reports (`turn`, shared by the stage's layers), to compare them in one run.
pub(crate) struct Guess {
    pub(crate) copies: Rc<[usize]>,
    pub(crate) turn: Rc<std::cell::Cell<usize>>,
    pub(crate) ids: crate::dev::Pinned,
    pub(crate) ev: crate::dev::Ev,
    pub(crate) pending: std::cell::Cell<bool>,
}

/// Copies a layer makes for its early guess at most: none. On 8x A100 (research/kimi-k3-session3-2026-10-22) the guess's copies were mostly experts the router then
/// did not choose, and took the PCIe time of its misses: 64 tokens in 14.7 s with no guess, 15.1-15.6 s with one or two copies a layer (Kimi-K3's attention output
/// weighs too much in its experts' input for the input without it to tell them). A development build's GLYD_KIMI_EARLY sets it, for a better guess.
pub(crate) const GUESS_COPIES: usize = 0;

impl Guess {
    fn copies(&self) -> usize {
        self.copies[self.turn.get() % self.copies.len()]
    }
}

impl MoeBlock for LatentCached {
    fn early_wanted(&self) -> bool {
        self.guess.as_ref().is_some_and(|g| g.copies() > 0)
    }

    /// The router on `h`, its choices read back for `forward` to place before the layer's call (`ExpertStore::guess`).
    unsafe fn early(&self, e: &dyn NhHost, h: u64) -> Result<()> {
        let Some(g) = &self.guess else { return Ok(()) };
        let m = &self.moe;
        let (c, s) = (&m.layer.cfg, e.dev().stream);
        let Some(sc) = e.moe_scratch().get() else { return Ok(()) };
        m.layer.route(sc, s, h, c.router_hidden, 1)?;
        crate::dev::copy_d2h_async(g.ids.ptr, sc.ids.ptr, c.router.top_k * 8, s)?;
        g.ev.record(s)?;
        g.pending.set(true);
        Ok(())
    }

    /// `LatentMoe::forward`'s launches with the experts' products over the places the router's choices were put in; the shared MLP queued before them.
    unsafe fn forward(&self, e: &dyn NhHost, h: u64, n: usize, out: u64) -> Result<()> {
        let m = &self.moe;
        let (c, s) = (&m.layer.cfg, e.dev().stream);
        if e.mix().bytes < n * c.router_hidden * 2 || e.xc().bytes < n * c.router_hidden * 2 {
            return err(format!("{n} rows of {} values do not fit the engine's buffers", c.router_hidden));
        }
        if e.moe_scratch().get().is_none() {
            let _ = e.moe_scratch().set(MoeScratch::new(e.dev(), e.lib(), &[&m.layer], e.chunk())?);
        }
        let sc = e.moe_scratch().get().ok_or_else(|| Error("the mixture-of-experts scratch is not made".into()))?;
        // (the early guess's copies queued: its choices were read back before the attention ran)
        if let Some(g) = self.guess.as_ref().filter(|g| g.pending.replace(false) && n == 1) {
            g.ev.wait_host()?;
            let ids: Vec<u16> = std::slice::from_raw_parts(g.ids.ptr as *const i64, c.router.top_k).iter().map(|&x| x as u16).collect();
            self.store.guess(self.index, &ids, g.copies())?;
        }
        let (z, y, gu) = m.down(e, sc, h, n)?;
        m.layer.route(sc, s, h, c.router_hidden, n)?;
        // (the shared experts queued while the host reads the choices and plans, beside the copies, instead of after them; a decode's call planned on the GPU where the store does: src/devplan.rs)
        m.layer.via_store(
            &self.store,
            self.index,
            crate::offload::At::of(e.dev()),
            sc,
            z,
            n,
            y,
            &mut || m.layer.shared_from(e.lib(), sc, s, gu, n),
            &mut |waves| m.layer.experts_products(e.lib(), sc, s, z, n, y, Some(sc.way.ptr), waves),
        )?;
        m.k.rms2(s, y, m.latent, m.norm.ptr, z, m.latent, m.latent, m.eps, n as u32)?;
        e.lin(&m.up, z, n, out)?;
        if let Some((c, at)) = self.report.as_ref().filter(|_| n == 1) {
            c.set(c.get() + 1);
            if c.get() % 64 == 0 {
                let early = self.guess.as_ref().map_or(0, Guess::copies);
                eprintln!(
                    "layers from {}: 64 steps in {:.1} ms, {early} early copies a layer; {}",
                    self.index,
                    at.get().elapsed().as_secs_f64() * 1e3,
                    self.store.report()?
                );
                at.set(std::time::Instant::now());
                self.store.reset_stats()?;
                if let Some(g) = &self.guess {
                    g.turn.set(g.turn.get() + 1);
                }
            }
        }
        m.layer.shared_add(sc, s, n, out)
    }

    /// A decode step's calls planned on the GPU (src/devplan.rs) need no host: the step is one graph.
    fn graphable(&self) -> bool {
        self.store.device_on(self.index)
    }

    fn reads_host(&self) -> bool {
        !self.store.device_on(self.index)
    }

    /// The experts a token takes: `forward` routes into the experts' scratch, whose choices stay there after the call for a check to read (`Dyn::trace_routes`).
    fn top_k(&self) -> Option<usize> {
        Some(self.moe.layer.cfg.router.top_k)
    }

    fn bytes(&self) -> usize {
        self.moe.bytes()
    }

    fn pack_shapes(&self) -> Vec<(i64, i64, bool)> {
        self.moe.pack_shapes()
    }

    fn store(&self) -> Option<Rc<ExpertStore>> {
        Some(self.store.clone())
    }
}

/// A mixture-of-experts layer of the expert cache across a group of GPUs (src/dynamic/tp_cache.rs): `LatentCached`'s, the layer's stacks this GPU's slices
/// of the places, the plan the first GPU's.
struct LatentTp {
    moe: LatentMoe,
    rank: Rc<super::tp_cache::Rank>,
    index: usize,
    places: super::tp_cache::Places,
    report: Option<(std::cell::Cell<u32>, std::cell::Cell<std::time::Instant>)>,
    /// The device path (`Places::dev`'s table): a decode step's plan and copies on the GPU, its fills and their count; None: every call planned by the
    /// first GPU's thread.
    dev: Option<(crate::dev::Buf, crate::dev::Buf)>,
}

/// The cache's decode steps planned on the GPU (kernels/kimi.cu's dc_plan and dc_fill: no host in a step, so the step is a graph): a development
/// build's GLYD_KIMI_DEVICE=1.
pub fn device_path() -> bool {
    crate::tune!("GLYD_KIMI_DEVICE").as_deref() == Some("1")
}

impl MoeBlock for LatentTp {
    unsafe fn forward(&self, e: &dyn NhHost, h: u64, n: usize, out: u64) -> Result<()> {
        use super::tp_cache::{Call as TpCall, Rank, Stage as TpStage};
        let m = &self.moe;
        let (c, s) = (&m.layer.cfg, e.dev().stream);
        if e.mix().bytes < n * c.router_hidden * 2 || e.xc().bytes < n * c.router_hidden * 2 {
            return err(format!("{n} rows of {} values do not fit the engine's buffers", c.router_hidden));
        }
        if e.moe_scratch().get().is_none() {
            let _ = e.moe_scratch().set(MoeScratch::new(e.dev(), e.lib(), &[&m.layer], e.chunk())?);
        }
        let sc = e.moe_scratch().get().ok_or_else(|| Error("the mixture-of-experts scratch is not made".into()))?;
        let (z, y, gu) = m.down(e, sc, h, n)?;
        m.layer.route(sc, s, h, c.router_hidden, n)?;
        // (a decode step on the device path: the places looked up and the misses copied in by the GPU, the products over them; nothing waits for the host)
        if let (1, Some((fills, nfill)), Some(t)) = (n, &self.dev, self.places.dev.as_ref()) {
            let pairs = c.router.top_k;
            let (po, eo, la, rf, me) = t.at();
            let k = &m.k;
            crate::launch!(k.dc_plan, [1, 1, 1], [256, 1, 1], s; sc.ids.ptr, pairs as i32, po, eo, la, rf, me, t.experts as i32, t.slots as i32, sc.way.ptr, fills.ptr, nfill.ptr)?;
            crate::launch!(k.dc_fill, [16, pairs as u32, 1], [256, 1, 1], s; fills.ptr, nfill.ptr, t.reg.ptr)?;
            m.layer.shared_from(e.lib(), sc, s, gu, n)?;
            m.layer.experts_products(e.lib(), sc, s, z, n, y, Some(sc.way.ptr), 1)?;
            m.layer.experts_sum(sc, s, n, y, 0, 0)?;
            m.k.rms2(s, y, m.latent, m.norm.ptr, z, m.latent, m.latent, m.eps, n as u32)?;
            e.lin(&m.up, z, n, out)?;
            return m.layer.shared_add(sc, s, n, out);
        }
        let call = TpCall { ids: sc.ids.ptr, wts: sc.wts.ptr, z, way: sc.way.ptr, part: sc.cpu.ptr, tokens: n, pairs: n * c.router.top_k };
        self.rank.run(self.index, &self.places, &call, s, &mut || m.layer.shared_from(e.lib(), sc, s, gu, n), &mut |stage| match stage {
            TpStage::Products(_, waves, true) => m.layer.experts_products(e.lib(), sc, s, z, n, y, Some(sc.way.ptr), waves),
            TpStage::Products(..) => Rank::zero(y, n * c.hidden * 2, s),
            TpStage::Sum(extra) => m.layer.experts_sum(sc, s, n, y, 0, extra),
        })?;
        m.k.rms2(s, y, m.latent, m.norm.ptr, z, m.latent, m.latent, m.eps, n as u32)?;
        e.lin(&m.up, z, n, out)?;
        if let Some((cnt, at)) = self.report.as_ref().filter(|_| n == 1) {
            cnt.set(cnt.get() + 1);
            if cnt.get() % 64 == 0 {
                eprintln!("64 steps in {:.1} ms; {}", at.get().elapsed().as_secs_f64() * 1e3, self.rank.report());
                at.set(std::time::Instant::now());
            }
        }
        m.layer.shared_add(sc, s, n, out)
    }

    fn graphable(&self) -> bool {
        self.dev.is_some()
    }

    /// As `LatentCached`'s: `forward` routes into the experts' scratch.
    fn top_k(&self) -> Option<usize> {
        Some(self.moe.layer.cfg.router.top_k)
    }

    fn bytes(&self) -> usize {
        self.moe.bytes()
    }

    fn pack_shapes(&self) -> Vec<(i64, i64, bool)> {
        self.moe.pack_shapes()
    }
}

/// A layer's experts packed straight into pinned memory, the expert cache's host tier, from the checkpoint (`names`: each expert's gate, up and down, FP4 as
/// released, their codes and their scales; `f` the layer's experts in an error): each expert's three matrices read with each thread's own buffers, gate and
/// up packed into the two halves of its first matrix's place and down into its second's, checked there (glyd_gpu::pack's `_into`s); in the layout with
/// packed scales where every expert's scales fit it, else as released. Whether packed, and the two matrices' strides (codes, scales) and buffers. No copy
/// of the layer is made: a load of Kimi-K3 joined each layer's 15 GB, copied it into pinned memory and freed it, and spent about half its time in the TLB
/// shootdowns of that.
fn pack_into_host(
    ck: &Checkpoint,
    names: &[[(String, String); 3]],
    f: &str,
    inter: usize,
    latent: usize,
    threads: usize,
    portable: bool,
) -> Result<(bool, [(u64, u64, crate::dev::Pinned, crate::dev::Pinned); 2])> {
    use glyd_gpu::pack::{fp4_bytes, pack_fp4_into, pack_fp4r_into};
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    let experts = names.len();
    for packed in [true, false] {
        let ((gd, gs), (dd, ds)) = (fp4_bytes(inter, latent, packed), fp4_bytes(latent, inter, packed));
        let (ud, us) = (2 * gd, 2 * gs);
        // (portable: pinned for every GPU's copies, where every GPU of a group copies its slices from it)
        let pin = |b: usize| if portable { crate::dev::Pinned::portable(b) } else { crate::dev::Pinned::new(b) };
        let bufs = [pin(experts * ud)?, pin(experts * us)?, pin(experts * dd)?, pin(experts * ds)?];
        let ptrs: Vec<usize> = bufs.iter().map(|b| b.ptr as usize).collect();
        let (next, bad, failed) = (AtomicUsize::new(0), AtomicBool::new(false), std::sync::Mutex::new(None::<Error>));
        let pk = if packed { pack_fp4r_into } else { pack_fp4_into };
        std::thread::scope(|sc| {
            for _ in 0..threads.min(experts).max(1) {
                sc.spawn(|| {
                    let mut raw: [Vec<u8>; 6] = Default::default();
                    loop {
                        let x = next.fetch_add(1, Ordering::Relaxed);
                        if x >= experts || bad.load(Ordering::Relaxed) || failed.lock().map_or(true, |g| g.is_some()) {
                            break;
                        }
                        let read = |n: &str, buf: &mut Vec<u8>| -> Result<()> {
                            let (file, t) = ck.get(n).ok_or_else(|| Error(format!("{f}: no tensor {n}")))?;
                            buf.resize(t.bytes as usize, 0);
                            io(file.read_into_uncached(t, buf))
                        };
                        let r = (|| -> Result<bool> {
                            // (gate, up, down: each one's codes, then its scales)
                            for (k, (codes, scales)) in names[x].iter().enumerate() {
                                read(codes, &mut raw[2 * k])?;
                                read(scales, &mut raw[2 * k + 1])?;
                            }
                            // SAFETY: expert x's place in each buffer (x times its stride, its stride long) is written by this item alone, and the buffers live for the scope.
                            let place = |b: usize, stride: usize| unsafe { std::slice::from_raw_parts_mut((ptrs[b] as *mut u8).add(x * stride), stride) };
                            let (ud_, us_, dd_, ds_) = (place(0, ud), place(1, us), place(2, dd), place(3, ds));
                            let (g_d, u_d) = ud_.split_at_mut(gd);
                            let (g_s, u_s) = us_.split_at_mut(gs);
                            Ok(pk(&raw[0], &raw[1], inter, latent, g_d, g_s)
                                && pk(&raw[2], &raw[3], inter, latent, u_d, u_s)
                                && pk(&raw[4], &raw[5], latent, inter, dd_, ds_))
                        })();
                        match r {
                            Ok(true) => {}
                            Ok(false) => bad.store(true, Ordering::Relaxed),
                            Err(e) => {
                                if let Ok(mut g) = failed.lock() {
                                    g.get_or_insert(e);
                                }
                            }
                        }
                    }
                });
            }
        });
        if let Some(e) = failed.into_inner().ok().flatten() {
            return Err(e);
        }
        if !bad.into_inner() {
            let [a, b, c, d] = bufs;
            return Ok((packed, [(ud as u64, us as u64, a, b), (dd as u64, ds as u64, c, d)]));
        }
    }
    err(format!("{f}: experts the FP4 layouts do not hold exactly"))
}

/// How a stage holds its blocks of experts: a hummingbird directory's (None: the release's), the expert cache's store and caching, the penguin level's places
/// (None, or as many as a layer's experts: all on the GPU), the group of GPUs' shared cache and this GPU's place in the group, the threads that pack, on the
/// GPU `ordinal`.
pub(crate) struct LatentHeld<'a> {
    pub(crate) hb: Option<&'a crate::hb_pack::HbDir>,
    pub(crate) hb_store: Option<&'a Rc<ExpertStore>>,
    pub(crate) caching: Option<&'a crate::hbg_store::HbCaching>,
    pub(crate) places: Option<usize>,
    pub(crate) tp_rank: Option<&'a Rc<super::tp_cache::Rank>>,
    pub(crate) tpr: Option<(usize, usize)>,
    pub(crate) store: &'a Rc<ExpertStore>,
    pub(crate) threads: usize,
    pub(crate) ordinal: i32,
}

/// The matrix of `names` stacked by rows (several: zero rows after them to `lower::FUSED_ROWS`s, never read), packed at `layout`: this GPU's slice of its rows where
/// every layer is across a group of GPUs and the group takes it; a shape the library does not pack (rows not a multiple of 64, columns of 16) held as saved, whole.
fn packed_mat(ck: &Checkpoint, names: &[String], layout: Layout) -> Result<Mat> {
    let (mut rows, mut cols) = (0usize, 0usize);
    for n in names {
        let (_, t) = ck.get(n).ok_or_else(|| Error(format!("no tensor {n}")))?;
        if t.shape.len() != 2 || t.dtype != "BF16" || (cols != 0 && t.shape[1] as usize != cols) {
            return err(format!("{n}: {} {:?}, not a bf16 matrix of {cols} columns", t.dtype, t.shape));
        }
        (rows, cols) = (rows + t.shape[0] as usize, t.shape[1] as usize);
    }
    let mut members = names.to_vec();
    if names.len() > 1 {
        rows = super::lower::fused(rows);
        members.push(crate::weights::ZERO_ROWS.to_string());
    }
    let w = crate::weights::item_words(ck, &crate::weights::Item { members, rows, cols })?;
    if rows % 64 != 0 || cols % 16 != 0 {
        return Ok(Mat::Dense(upload_bytes(as_bytes(&w))?, rows, cols));
    }
    let sliced = super::tp::slicer().filter(|t| t.slices(rows, cols));
    let (r0, rn) = sliced.map_or((0, rows), |t| t.rows(rows));
    let mut pk = upload_packed(pack_words(&w[r0 * cols..(r0 + rn) * cols], rn, cols, layout, &names[0])?)?;
    if sliced.is_some() {
        pk.tp_rows = rows;
    }
    Ok(Mat::Packed(pk))
}

/// What the latent block takes of its kind `m`: its latent (`kinds::Latent`) and the epsilon of the latent's norm of the experts' sum (`Latent::norm_out`, RMS
/// with a weight and two roundings, KimiRMSNorm's). What the block does not run is refused in words: no latent, another norm of the sum or none, a norm of
/// the latent's input or of each expert's output (the executor's latent operations and `latent_normed`'s), a gated shared expert.
fn latent_of(m: &kinds::Moe) -> Result<(&kinds::Latent, f32)> {
    let lat = m.latent.as_ref().ok_or_else(|| Error("no latent, where the latent block's experts work in one".into()))?;
    let eps = match lat.norm_out {
        Some(n) if n.kind == NormKind::Rms && n.weight == NormWeight::Plain && n.rounding == kinds::Rounding::Twice => n.eps,
        n => return err(format!("a latent whose sum takes {n:?}, where the latent block norms it with two roundings (RMS, a weight)")),
    };
    if lat.norm_in.is_some() || lat.expert_norm.is_some() || m.shared_gate {
        return err(format!(
            "a latent of {lat:?} and a shared expert's gate {}: the latent block's latent has a norm of the experts' sum alone, its shared experts no gate",
            m.shared_gate
        ));
    }
    Ok((lat, eps))
}

/// Decoder layer `l`'s block of experts of the kind `m` in a latent (`kinds::Latent`) with its own projections, from its roles' tensors (`roles`; the experts as
/// released in FP4, each matrix's codes with the scales `scales` names), over a model `hidden` wide: the router over the model's width and its choice bias, the
/// latent's down projection with the shared experts' gate and up riding in its product, the experts on the latent, the latent's norm (`Latent::norm_out`) and
/// up projection, the shared experts' down; held as the stage holds them (`held`). What the block does not run is refused in words: experts not in FP4 as
/// released, a latent whose sum is not normed with two roundings, normed experts (`latent_normed`'s), a gated shared expert, an activation the host tier
/// does not compute where it computes the misses.
#[allow(clippy::too_many_arguments)]
pub(crate) fn latent_block(
    m: &kinds::Moe,
    roles: &[(String, Source)],
    hidden: usize,
    scales: &std::collections::BTreeMap<String, String>,
    ck: &Checkpoint,
    layout: Layout,
    l: usize,
    held: &LatentHeld,
    k: &Rc<KernLatent>,
    first: bool,
    guess: &dyn Fn(usize) -> Result<Option<Guess>>,
) -> Result<Box<dyn MoeBlock>> {
    let h = hidden;
    let (lat, eps) = latent_of(m).map_err(|e| Error(format!("layer {l}'s experts: {}", e.msg)))?;
    let find = |r: &str| roles.iter().find(|(n, _)| n == r).map(|(_, s)| s);
    let one = |r: &str| match find(r) {
        Some(Source::Tensor(n)) => Ok(n.clone()),
        x => err(format!("layer {l}'s experts' role {r} is {x:?}, where one tensor is")),
    };
    // (each expert's matrices as released, FP4: codes and scales, its gate and up (`experts_gate_up`) then its down)
    let fp4 = |r: &str, each: usize| -> Result<Vec<Vec<(String, String)>>> {
        let Some(Source::Experts(v)) = find(r) else {
            return err(format!("layer {l}'s experts' role {r} is {:?}, where a tensor of each expert's is", find(r)));
        };
        if v.len() != m.experts || v.iter().any(|x| x.len() != each) {
            return err(format!(
                "layer {l}'s experts' role {r}: {} experts of {:?} tensors, where {} of {each} are",
                v.len(),
                v.first().map(Vec::len),
                m.experts
            ));
        }
        v.iter()
            .map(|m| {
                m.iter()
                    .map(|n| {
                        scales.get(n).map(|s| (n.clone(), s.clone())).ok_or_else(|| Error(format!("{n}: not FP4 with its scales, as the experts are read")))
                    })
                    .collect()
            })
            .collect()
    };
    let (gu, dn) = (fp4("experts_gate_up", 2)?, fp4("experts_down", 1)?);
    let names: Vec<[(String, String); 3]> = gu.iter().zip(&dn).map(|(g, d)| [g[0].clone(), g[1].clone(), d[0].clone()]).collect();
    let f = format!("layer {l}'s experts");
    let small = |n: &str| -> Result<Buf> { upload_bytes(&io(ck.read(n))?) };
    let dims = |n: &str| ck.get(n).map(|(_, t)| t.shape.clone()).ok_or_else(|| Error(format!("no tensor {n}")));
    let (latent, experts, top_k) = (lat.dim, m.experts, m.router.top_k);
    // (the experts' width: a hummingbird directory's record's, which holds no matrix of theirs; else their gate's rows as released, its codes two a byte
    // [inter, latent / 2])
    let inter = match held.hb.map(|d| (d, d.layer(2 * l + 1))) {
        Some((_, Some(r))) => r.inter,
        Some((d, None)) => return err(format!("{}: no experts for decoder layer {l}", d.dir.display())),
        None => {
            let g = dims(&names.first().ok_or_else(|| Error(format!("{f}: no experts")))?[0].0)?;
            if g.len() != 2 || 2 * g[1] as usize != latent {
                return err(format!("{f}: experts' codes of {g:?}, where the kind's latent is {latent} (two a byte)"));
            }
            g[0] as usize
        }
    };
    let cfg = cfg(m, h, inter)?;
    // (the activation as the host tier computes a miss: SiTU's two betas)
    let situ = match cfg.act {
        Act::Situ { beta, linear } => Ok((beta, linear)),
        a => err(format!("{f}: experts of {a:?} in the computer's memory: the host tier computes SiTU's")),
    };
    // (the shared experts' width, all of them as one: their gate's rows)
    let si = match m.shared {
        Some(_) => dims(&one("shared_gate")?)?[0] as usize,
        None => 0,
    };
    // the experts as released, packed exactly in the smallest FP4 layout that holds them all (DeepSeek-V4's: packed scales, else as released): gate and up stacked as the first matrix;
    // all on the GPU, or places for `p` of them and every expert in the computer's memory, packed there straight from the checkpoint (pack_into_host)
    let slots = held.places.filter(|&p| p < experts);
    let mut tp_places = None;
    let hbx = crate::hbg_store::hb_layer(
        held.hb,
        l,
        held.tpr.is_some(),
        held.caching.zip(held.hb_store),
        cfg.act,
        crate::cpu_hbg::Finish::Rotated,
        held.ordinal,
        top_k,
    )?;
    if hbx.is_some() && slots.is_some() {
        return err(format!("{f}: a hummingbird directory's experts are in the hummingbird expert cache (src/hbg_store.rs), not the penguin level's places"));
    }
    let (up, down) = if hbx.is_some() {
        (ExpertStack::shape_only(experts, 2 * inter, latent), ExpertStack::shape_only(experts, latent, inter))
    } else {
        match slots {
            Some(p) if held.tp_rank.is_some() => {
                let (rank, n) = held.tpr.ok_or_else(|| Error("a group of GPUs without its place".into()))?;
                let cache = held.tp_rank.expect("a group's cache").cache_ref();
                // (the layers' host tiers packed by the GPUs' threads in turn, each the layers of its number, and shared)
                let host = if l % n == rank {
                    let (packed, packs) = pack_into_host(ck, &names, &f, inter, latent, held.threads, true)?;
                    let [(ud, us, a, b), (dd, ds, c2, d2)] = packs;
                    use super::tp_cache::HostBuf;
                    let h = std::sync::Arc::new(super::tp_cache::HostLayer {
                        mats: [[HostBuf(a), HostBuf(b)], [HostBuf(c2), HostBuf(d2)]],
                        stride: [[ud, us], [dd, ds]],
                        packed,
                        inter,
                        latent,
                        act: situ?,
                    });
                    cache.put_host(l, h.clone())?;
                    h
                } else {
                    cache.host(l)?
                };
                let tag = if host.packed { glyd_gpu::pack::FP4R_TAG } else { glyd_gpu::pack::FP4_TAG };
                let mut up = ExpertStack::pooled(tag, [tag << 24, 0, 0], experts, p, 2 * inter / n, latent)?;
                let mut down = ExpertStack::pooled(tag, [tag << 24, 0, 0], experts, p, latent / n, inter)?;
                let slice = [[up.data_stride, up.scale_stride], [down.data_stride, down.scale_stride]];
                if (0..2).any(|m| (0..2).any(|k| slice[m][k] * n as u64 != host.stride[m][k])) {
                    return err(format!("{f}: experts of {:?} bytes in the computer's memory, where {n} GPUs' places hold {slice:?}", host.stride));
                }
                (up.tp_rows, down.tp_rows) = (2 * inter, latent);
                let mut pl = super::tp_cache::Places::new([up.buffers(), down.buffers()], slice, experts, p)?;
                // (the device path places a step's experts at once: a layer of fewer places than a token's choices is the host's, in waves)
                if device_path() && p >= top_k {
                    pl.attach_device(&host, rank)?;
                }
                tp_places = Some(pl);
                (up, down)
            }
            None => {
                // (read by the packing threads: what they need of the holding alone)
                let tpr = held.tpr;
                let fp4 = |x: usize, packed_scales: bool| -> Result<Option<(glyd_gpu::pack::Tiered, glyd_gpu::pack::Tiered)>> {
                    let raw = |w: &(String, String)| -> Result<(Vec<u8>, Vec<u8>)> { Ok((io(ck.read(&w.0))?, io(ck.read(&w.1))?)) };
                    let pk = if packed_scales { glyd_gpu::pack::pack_fp4r_checked } else { glyd_gpu::pack::pack_fp4_checked };
                    let ((mut uq, mut ue), (gq, ge), (dq, de)) = (raw(&names[x][0])?, raw(&names[x][1])?, raw(&names[x][2])?);
                    uq.extend(gq);
                    ue.extend(ge);
                    // (a group of GPUs: this GPU's rows of each, codes half a byte a value and a scale a run of 32)
                    let rows_of = |q: &[u8], sc: &[u8], rows: usize, cols: usize| -> (Vec<u8>, Vec<u8>, usize) {
                        match tpr {
                            Some((rank, n)) => {
                                let (r0, rn) = (rank * rows / n, rows / n);
                                (q[r0 * cols / 2..(r0 + rn) * cols / 2].to_vec(), sc[r0 * cols / 32..(r0 + rn) * cols / 32].to_vec(), rn)
                            }
                            None => (q.to_vec(), sc.to_vec(), rows),
                        }
                    };
                    let (uq, ue, ur) = rows_of(&uq, &ue, 2 * inter, latent);
                    let (dq, de, dr) = rows_of(&dq, &de, latent, inter);
                    Ok(pk(&uq, &ue, ur, latent).zip(pk(&dq, &de, dr, inter)))
                };
                let (mut ups, mut downs) = (Vec::with_capacity(experts), Vec::with_capacity(experts));
                for packed_scales in [true, false] {
                    (ups.clear(), downs.clear());
                    let mut all = true;
                    crate::weights::ordered(experts, held.threads, 2 * held.threads, &|x| fp4(x, packed_scales), |_, pr| {
                        match pr {
                            Some((u, d)) => (ups.push(u), downs.push(d)),
                            None => (all = false, ()),
                        };
                        Ok(())
                    })?;
                    if all {
                        break;
                    }
                }
                if ups.len() != experts {
                    return err(format!("{f}: experts the FP4 layouts do not hold exactly"));
                }
                let (ju, jd) = (Joined::of(&ups.iter().collect::<Vec<_>>())?, Joined::of(&downs.iter().collect::<Vec<_>>())?);
                drop((ups, downs));
                let (mut su, mut sd) = (ExpertStack::upload(&ju)?, ExpertStack::upload(&jd)?);
                if let Some((_, n)) = held.tpr {
                    (su.tp_rows, sd.tp_rows) = (su.rows * n, sd.rows * n);
                }
                (su, sd)
            }
            Some(p) => {
                let (packed, packs) = pack_into_host(ck, &names, &f, inter, latent, held.threads, false)?;
                let tag = if packed { glyd_gpu::pack::FP4R_TAG } else { glyd_gpu::pack::FP4_TAG };
                let up = ExpertStack::pooled(tag, [tag << 24, 0, 0], experts, p, 2 * inter, latent)?;
                let down = ExpertStack::pooled(tag, [tag << 24, 0, 0], experts, p, latent, inter)?;
                let [(ud, us, a, b), (dd, ds, c2, d2)] = packs;
                if [ud, us, dd, ds] != [up.data_stride, up.scale_stride, down.data_stride, down.scale_stride] {
                    return err(format!(
                        "{f}: experts of {ud} + {us} and {dd} + {ds} bytes, where the places hold {} + {} and {} + {}",
                        up.data_stride, up.scale_stride, down.data_stride, down.scale_stride
                    ));
                }
                let host = vec![
                    PinnedPack { data: a, scales: b, stride: [ud, us], dev: up.buffers() },
                    PinnedPack { data: c2, scales: d2, stride: [dd, ds], dev: down.buffers() },
                ];
                // (the host computes the misses it is better at from the host tier's bytes as they are: src/cpu_fp4.rs)
                let (beta, linear) = situ?;
                let half = crate::offload::HostHalf::Fp4 { inter, latent, packed, act: crate::cpu_fp4::Act::Situ { beta, linear } };
                held.store.add_layer_pinned(l, experts, p, host, Some(half))?;
                (up, down)
            }
        }
    };
    // (the latent's down projection and the shared experts' gate and up, one product of the layer's input: one fewer a layer, and one gather fewer across
    // GPUs; the shared expert's first matrix a shape there, never multiplied by)
    let first_product: Vec<String> = match m.shared {
        Some(_) => vec![one("latent_down")?, one("shared_gate")?, one("shared_up")?],
        None => vec![one("latent_down")?],
    };
    let dg = packed_mat(ck, &first_product, layout)?;
    let sh = match m.shared {
        Some(_) => {
            let Mat::Packed(sd) = packed_mat(ck, &[one("shared_down")?], layout)? else {
                return err(format!("{f}: the shared experts' down projection is not packed"));
            };
            Some(Shared { inter: si, up: PackBuf::shape_only(2 * si, h)?, down: sd, gate: None })
        }
        None => None,
    };
    // (a router's choice bias where the kind has one)
    let bias = if m.router.bias { Some(f32s(ck, &one("choice_bias")?)?) } else { None };
    let w = MoeWeights { router: small(&one("router")?)?, bias, escale: None, up, down: Down::Block(down) };
    let layer = match hbx {
        Some(x) => x.into_layer(cfg, w.router, w.bias, None, sh)?,
        None => MoeLayer::new(cfg, w, sh)?,
    };
    let b = LatentMoe { layer, dg, up: packed_mat(ck, &[one("latent_up")?], layout)?, norm: small(&one("latent_norm_out")?)?, latent, eps, k: k.clone() };
    Ok(match slots {
        None => Box::new(b),
        Some(_) if tp_places.is_some() => {
            let report = (crate::tune!("GLYD_KIMI_STATS").is_some() && held.tpr.is_some_and(|r| r.0 == 0) && first)
                .then(|| (std::cell::Cell::new(0), std::cell::Cell::new(std::time::Instant::now())));
            let dev = if tp_places.as_ref().is_some_and(|pl| pl.dev.is_some()) {
                Some((crate::dev::Buf::new(top_k * 8)?, crate::dev::Buf::zeroed(4)?))
            } else {
                None
            };
            Box::new(LatentTp {
                moe: b,
                rank: held.tp_rank.cloned().expect("a group's cache"),
                index: l,
                places: tp_places.take().expect("the places"),
                report,
                dev,
            })
        }
        Some(_) => {
            let report =
                (crate::tune!("GLYD_KIMI_STATS").is_some() && first).then(|| (std::cell::Cell::new(0), std::cell::Cell::new(std::time::Instant::now())));
            Box::new(LatentCached { moe: b, store: held.store.clone(), index: l, guess: guess(top_k)?, report })
        }
    })
}

#[cfg(test)]
mod tests {
    use crate::moe::{Act, Scoring};
    use crate::runtime::kinds::{self, Mlp, Moe, Router, RouterScore};

    fn qwen() -> Moe {
        let mlp = Mlp { act: kinds::Act::Silu, gated: true, bias: false, clamp: None };
        Moe {
            router: Router { score: RouterScore::Softmax, logit_scale: 1.0, bias: false, groups: None, top_k: 10, renorm: true, scale: 1.0, hash: false },
            experts: 512,
            expert: mlp,
            shared: Some(mlp),
            shared_gate: true,
            latent: None,
        }
    }

    /// The Qwen3.5 layout with experts saved at a block level (`pack::run_with`: each expert's gate and up one pack under the gate's name, its down another, the
    /// shared expert's two) is read back as the same packs the loader makes of the release's weights at that level (`block_packs`, `pack_of`), and its widths
    /// as the release's: the experts' and the shared expert's first matrices' rows, a gate's and an up's.
    #[cfg(feature = "dev")]
    #[test]
    fn a_saved_directorys_fused_experts_are_the_packs_of_the_release() {
        use super::{block_packs, pack_of, rows_cols, Source, Stacked};
        use crate::weights::Layout;
        let dir = std::env::temp_dir().join(format!("glyd-experts-v7-{}", std::process::id()));
        let (src, out) = (dir.join("src"), dir.join("i5"));
        let _ = std::fs::remove_dir_all(&dir);
        crate::logits::qwen35moe_random(&src).unwrap();
        crate::pack::run_with(src.to_str().unwrap(), &out, 4, glyd_gpu::save::Layout::Int5, true).unwrap();
        let (d, _) = crate::runtime::describe::describe_dir(&src).unwrap();
        let (a, b) = (glyd_gpu::safetensors::Checkpoint::open(&src).unwrap(), glyd_gpu::safetensors::Checkpoint::open(&out).unwrap());
        let v7 = crate::weights::v7_manifest(&out).unwrap();
        assert_eq!(v7.as_ref().map(|x| x.0), Some(Layout::Int5));
        let mut blocks = 0;
        for (_, m, roles) in super::of_layers(&d, (0, 4), 2) {
            let find = |r: &str| roles.iter().find(|(n, _)| n == r).map(|(_, s)| s);
            let (up, down) = (Stacked::of(find("experts_gate_up"), m.experts, "up").unwrap(), Stacked::of(find("experts_down"), m.experts, "down").unwrap());
            let (ra, rb) = (
                rows_cols(
                    &a,
                    None,
                    match find("experts_gate_up") {
                        Some(Source::Experts(v)) => &v[0],
                        _ => panic!("an expert's tensors"),
                    },
                )
                .unwrap(),
                up.dims(&b, v7.as_ref(), m.experts).unwrap(),
            );
            assert_eq!((ra, rb), ((256, 1024), (256, 1024)), "the experts' first matrices: a gate's and an up's rows");
            let ((ja, da), (jb, db)) = (
                block_packs(&up, &down, &a, None, m.experts, Layout::Int5, m.expert.act).unwrap(),
                block_packs(&up, &down, &b, v7.as_ref(), m.experts, Layout::Int5, m.expert.act).unwrap(),
            );
            assert!(ja.data == jb.data && ja.scales == jb.scales && (ja.rows, ja.cols, ja.experts) == (jb.rows, jb.cols, jb.experts));
            assert!(da.bytes() == db.bytes(), "the experts' second matrices");
            let one = |r: &str| match find(r) {
                Some(Source::Tensor(n)) => n.clone(),
                _ => panic!("{r}"),
            };
            let first = vec![one("shared_gate"), one("shared_up")];
            assert_eq!((rows_cols(&a, None, &first).unwrap(), rows_cols(&b, v7.as_ref(), &first).unwrap()), ((256, 1024), (256, 1024)));
            for members in [first, vec![one("shared_down")]] {
                let (pa, pb) = (pack_of(&a, None, &members, Layout::Int5).unwrap(), pack_of(&b, v7.as_ref(), &members, Layout::Int5).unwrap());
                assert!(pa.data == pb.data && pa.b1 == pb.b1 && pa.b2 == pb.b2 && (pa.rows, pa.cols) == (pb.rows, pb.cols), "{members:?}");
            }
            blocks += 1;
        }
        assert_eq!(blocks, 4);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The kind's router and activation as the core's layer takes them: a softmax router's logits bf16, a sigmoid's scores fp32 with its scale, the SwiGLU clamp.
    #[test]
    fn a_kind_becomes_the_core_layers_shape() {
        let c = super::cfg(&qwen(), 8192, 2048).unwrap();
        assert_eq!((c.experts, c.router_hidden, c.hidden, c.inter, c.up_rows()), (512, 8192, 8192, 2048, 4096));
        assert_eq!((c.router.scoring, c.router.top_k, c.router.renorm, c.router.scale, c.router.bf16_logits), (Scoring::Softmax, 10, true, 1.0, true));
        assert_eq!(c.act, Act::SwiGlu { limit: None });
        c.check().unwrap();
        let mut g = qwen();
        g.router = Router { score: RouterScore::Sigmoid, bias: true, scale: 2.5, top_k: 8, ..g.router };
        (g.expert.clamp, g.shared, g.shared_gate) = (Some(10.0), None, false);
        let c = super::cfg(&g, 4096, 2048).unwrap();
        assert_eq!((c.router.scoring, c.router.scale, c.router.bf16_logits, c.act), (Scoring::Sigmoid, 2.5, false, Act::SwiGlu { limit: Some(10.0) }));
    }

    /// Experts that work in a latent and whose outputs are normed: the experts' width is the latent's, the router's the model's; each pair's activation is a
    /// kernel's of its own (an infinite clamp: never the first product's, so a stack's rows can be sliced across GPUs), and the router's logit scale is
    /// theirs, the operation's to apply. A latent whose outputs are not normed keeps the first product's activation.
    #[test]
    fn experts_in_a_latent_with_normed_outputs() {
        use crate::runtime::kinds::{Latent, Norm, NormWeight};
        let mut m = qwen();
        m.router = Router { score: RouterScore::Sigmoid, logit_scale: 0.5, bias: true, top_k: 4, scale: 1.0, ..m.router };
        (m.experts, m.shared, m.shared_gate) = (16, None, false);
        let normed = Norm { weight: NormWeight::InputPlusOne, ..Norm::rms(1e-8) };
        m.latent = Some(Latent { dim: 512, norm_in: Some(Norm::rms_plus_one(1e-5)), expert_norm: Some(normed), norm_out: None });
        let c = super::cfg(&m, 1024, 512).unwrap();
        assert_eq!((c.experts, c.router_hidden, c.hidden, c.inter, c.fused()), (16, 1024, 512, 512, 0));
        assert_eq!((c.router.scoring, c.router.bf16_logits, c.act), (Scoring::Sigmoid, false, Act::SwiGlu { limit: Some(f32::INFINITY) }));
        c.check().unwrap();
        let op = super::latent_normed(3, 7, 1e-8, 0.5);
        assert_eq!((op.block, op.x, op.y, op.logit_scale, op.expert_norm), (3, super::Slot::Xc, super::Slot::M, 0.5, Some((7, 1e-8))));
        // (a plain weight on the outputs' norm, or a shared expert beside them, is not the kernels')
        let mut plain = m.clone();
        plain.latent.as_mut().unwrap().expert_norm = Some(Norm::rms(1e-8));
        assert!(super::cfg(&plain, 1024, 512).unwrap_err().msg.contains("input-scaled"));
        let mut unnormed = m.clone();
        unnormed.latent.as_mut().unwrap().expert_norm = None;
        unnormed.router.logit_scale = 1.0;
        let c = super::cfg(&unnormed, 1024, 512).unwrap();
        assert_eq!((c.router_hidden, c.hidden, c.fused(), c.act), (1024, 512, 1, Act::SwiGlu { limit: None }));
    }

    /// The latent block takes a kind whose experts work in a latent with a norm of their sum of two roundings (Kimi-K3's), with shared experts or without them;
    /// what it does not run is refused in words: no latent, another norm of the sum or none, a norm of the latent's input or of each expert's output, a gated
    /// shared expert.
    #[test]
    fn the_latent_blocks_kind() {
        use crate::runtime::kinds::{Latent, Norm, NormKind, Rounding};
        let twice = Norm { rounding: Rounding::Twice, ..Norm::rms(1e-5) };
        let mut m = qwen();
        m.router = Router { score: RouterScore::Sigmoid, bias: true, top_k: 4, ..m.router };
        (m.shared_gate, m.latent) = (false, Some(Latent { dim: 128, norm_in: None, expert_norm: None, norm_out: Some(twice) }));
        assert_eq!(super::latent_of(&m).map(|(l, e)| (l.dim, e)).unwrap(), (128, 1e-5));
        let unshared = Moe { shared: None, ..m.clone() };
        assert!(super::latent_of(&unshared).is_ok());
        let refused = |edit: &dyn Fn(&mut Moe), word: &str| {
            let mut x = m.clone();
            edit(&mut x);
            let e = super::latent_of(&x).unwrap_err().msg;
            assert!(e.contains(word), "{e}");
        };
        refused(&|x| x.latent = None, "no latent");
        refused(&|x| x.latent.as_mut().unwrap().norm_out = None, "two roundings");
        refused(&|x| x.latent.as_mut().unwrap().norm_out = Some(Norm::rms(1e-5)), "two roundings");
        refused(&|x| x.latent.as_mut().unwrap().norm_out = Some(Norm { kind: NormKind::Layer, ..twice }), "two roundings");
        refused(&|x| x.latent.as_mut().unwrap().norm_in = Some(Norm::rms_plus_one(1e-5)), "alone");
        refused(&|x| x.latent.as_mut().unwrap().expert_norm = Some(Norm::rms(1e-5)), "alone");
        refused(&|x| x.shared_gate = true, "no gate");
    }

    /// What the resident block does not run is refused in words.
    #[test]
    fn what_the_block_does_not_run_is_refused() {
        let refused = |edit: &dyn Fn(&mut Moe), word: &str| {
            let mut m = qwen();
            edit(&mut m);
            let e = super::cfg(&m, 1024, 256).unwrap_err().msg;
            assert!(e.contains(word), "{e}");
        };
        refused(&|m| m.router.hash = true, "hash routing");
        refused(&|m| m.router.groups = Some((8, 4)), "grouped choices");
        refused(&|m| m.router.logit_scale = 2.0, "logit scale");
        refused(&|m| m.router.bias = true, "choice bias");
        refused(&|m| m.expert.bias = true, "no biases");
        refused(
            &|m| {
                m.expert.act = kinds::Act::Gelu;
                m.shared = Some(m.expert);
            },
            "SwiGLU",
        );
    }

    /// A model of one mixture-of-experts layer with a latent (8 experts of 3 a token, a latent of 64, a model width of 128), small enough to pack in a test and
    /// with every matrix in whole blocks of the block layouts.
    const SMALL: &str = r#"{"architectures": ["NemotronHForCausalLM"], "model_type": "nemotron_h", "hidden_size": 128, "num_hidden_layers": 1, "hybrid_override_pattern": "E", "vocab_size": 64,
        "layer_norm_epsilon": 1e-5, "max_position_embeddings": 256, "mamba_num_heads": 64, "mamba_head_dim": 64, "n_groups": 1, "ssm_state_size": 128, "conv_kernel": 4, "num_attention_heads": 2,
        "num_key_value_heads": 1, "head_dim": 128, "n_routed_experts": 8, "num_experts_per_tok": 3, "moe_intermediate_size": 128, "moe_shared_expert_intermediate_size": 64, "moe_latent_size": 64,
        "routed_scaling_factor": 5.0, "norm_topk_prob": true, "n_group": 1, "topk_group": 1, "n_shared_experts": 1, "mlp_hidden_act": "relu2", "tie_word_embeddings": false, "eos_token_id": 2}"#;

    /// A block of relu squared experts in a latent of its own, read from the bf16 checkpoint as the packers make it at the three block levels (each expert's first
    /// matrix packed alone, joined: the stacked pack of the packer; each second matrix as the row layout of its transpose, joined; the latent's projections and the
    /// shared expert at the stage's layout of them, six-bit at wren); the same block saved as a glyd-v7 directory read to the same bytes; a directory saved with the
    /// second matrices in the block layout (a directory of before the row layouts) read in it, as the stacked pack of the packer; a directory of another level, and
    /// a level the experts are not held at, refused before anything is read.
    #[test]
    fn a_block_is_read_from_bf16_and_from_a_saved_directory_alike() {
        use crate::moe_rows::{DownHost, RowsJoined};
        use crate::nemotron::tests::checkpoint;
        use crate::nemotron::ConfigNemotronH;
        use crate::pack::nemotron_plan;
        use crate::runtime::describe::Source;
        use crate::weights::{Layout, Packed};
        use glyd_gpu::pack::{pack_int4, pack_int5, pack_int6, pack_rows, transpose, Tiered, INT4_TAG, INT5_TAG, INT6_TAG};
        use glyd_gpu::safetensors::Checkpoint;
        use glyd_gpu::save::{self, Source as Src};
        let dir = std::env::temp_dir().join(format!("glyd-engine-experts-block-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let c = ConfigNemotronH::parse(SMALL, "small", None).unwrap();
        let src = dir.join("src");
        let w = checkpoint(&src, SMALL, &c);
        let ck = Checkpoint::open(&src).unwrap();
        let kinds = crate::runtime::describe::describe(&serde_json::from_str(SMALL).unwrap()).unwrap();
        let (ix, m, roles) = super::of_layers(&kinds, (0, 1), 1).next().unwrap();
        assert_eq!((ix, super::own_latent(m).map(|l| l.dim)), (super::Ix { layer: 0, moe: 0 }, Some(64)));
        let find = |r: &str| roles.iter().find(|(n, _)| n == r).map(|(_, s)| s);
        let one = |r: &str| match find(r) {
            Some(Source::Tensor(n)) => n.clone(),
            x => panic!("{r}: {x:?}"),
        };
        let (up, down) =
            (super::Stacked::of(find("experts_up"), 8, "experts_up").unwrap(), super::Stacked::of(find("experts_down"), 8, "experts_down").unwrap());
        let p = "backbone.layers.0.mixer";
        let (e, lat, inter, hid, sh) = (8usize, 64usize, 128usize, 128usize, 64usize);
        let stacked = |name: &dyn Fn(usize) -> String| -> Vec<u16> { (0..e).flat_map(|i| w[&name(i)].clone()).collect() };
        let (up_w, down_w) = (stacked(&|i| format!("{p}.experts.{i}.up_proj.weight")), stacked(&|i| format!("{p}.experts.{i}.down_proj.weight")));
        let parts = |d: &DownHost| match d {
            DownHost::Block(j) => (j.tag, j.experts, j.rows, j.cols, j.data.clone(), j.scales.clone()),
            DownHost::Rows(j) => (j.tag, j.experts, j.rows, j.cols, j.data.clone(), j.scales.clone()),
        };
        let tag_of = |p: &Packed| match p.words {
            crate::ffi::Words::Tiered { tiers, .. } => tiers[0] >> 24,
            crate::ffi::Words::Twelve { .. } => 0,
        };
        type Pack = fn(&[u16], usize, usize) -> Option<Tiered>;
        for (level, pack, dense_pack, tags) in [
            (Layout::Int6, pack_int6 as Pack, pack_int6 as Pack, (INT6_TAG, INT6_TAG)),
            (Layout::Int5, pack_int5 as Pack, pack_int5 as Pack, (INT5_TAG, INT5_TAG)),
            (Layout::Int4, pack_int4 as Pack, pack_int6 as Pack, (INT4_TAG, INT6_TAG)),
        ] {
            let dense = crate::nemotron::dense_layout(level);
            let read = |ck: &Checkpoint| {
                let v7 = crate::weights::v7_manifest(&ck.dir).unwrap();
                let (ju, jd) = super::block_packs(&up, &down, ck, v7.as_ref(), e, level, m.expert.act).unwrap();
                let mats: Vec<Packed> = ["latent_down", "latent_up", "shared_up", "shared_down"]
                    .iter()
                    .map(|r| super::pack_of(ck, v7.as_ref(), &[one(r)], dense).unwrap())
                    .collect();
                (ju, jd, mats)
            };
            let (ju, jd, mats) = read(&ck);
            // the experts: the stacked pack of the packer, the codes and the scales; the second matrices as the row layout of each one's transpose
            let bits = level.row_of().unwrap().row_bits().unwrap();
            let want_up = pack(&up_w, e * inter, lat).unwrap();
            let rows_packs: Vec<Tiered> =
                (0..e).map(|i| pack_rows(bits, &transpose(&down_w[i * lat * inter..(i + 1) * lat * inter], lat, inter), inter, lat).unwrap()).collect();
            let want_down = RowsJoined::of(&rows_packs.iter().collect::<Vec<_>>()).unwrap();
            assert_eq!((ju.tag, tag_of(&mats[0]), tag_of(&mats[2])), (tags.0, tags.1, tags.1), "{level:?}: the layouts");
            assert!(
                ju.data == want_up.data && ju.scales == want_up.blocks && (ju.experts, ju.rows, ju.cols) == (e, inter, lat),
                "{level:?}: the first matrices"
            );
            assert!(matches!(jd, DownHost::Rows(_)), "{level:?}: relu squared experts' second matrices are in the row layout");
            assert_eq!(parts(&jd), (want_down.tag, e, inter, lat, want_down.data.clone(), want_down.scales.clone()), "{level:?}: the second matrices");
            for (got, name, rows, cols) in [
                (&mats[0], "fc1_latent_proj", lat, hid),
                (&mats[1], "fc2_latent_proj", hid, lat),
                (&mats[2], "shared_experts.up_proj", sh, hid),
                (&mats[3], "shared_experts.down_proj", hid, sh),
            ] {
                let want = dense_pack(&w[&format!("{p}.{name}.weight")], rows, cols).unwrap();
                assert!(got.data == want.data && got.b1 == want.blocks && (got.rows, got.cols) == (rows, cols), "{level:?}: {name}");
            }
            // saved as a glyd-v7 directory, read back: the same bytes
            let out = dir.join(level.name());
            let have = &|n: &str| ck.get(n).is_some_and(|(_, t)| t.dtype == "F32");
            save::save_planned(&Src::find(src.to_str().unwrap()).unwrap(), &out, 2, 5 << 30, level, &nemotron_plan(&c, level, have)).unwrap();
            let saved = Checkpoint::open(&out).unwrap();
            assert!(saved.get(&format!("{p}.experts.7.down_proj.glyd_data")).is_some() && saved.get(&format!("{p}.experts.7.down_proj.weight")).is_none());
            let (su, sd, smats) = read(&saved);
            assert!(su.data == ju.data && su.scales == ju.scales && parts(&sd) == parts(&jd), "{level:?}: the experts of the saved directory");
            for (a, b) in smats.iter().zip(&mats) {
                assert!(a.data == b.data && a.b1 == b.b1 && (a.rows, a.cols) == (b.rows, b.cols), "{level:?}: a matrix of the saved directory");
            }
            // a directory saved before the row layouts (the second matrices in the level's block layout): read in it, as the stacked pack of the packer
            let plan = nemotron_plan(&c, level, have);
            let old_plan = plan.replace(&format!(", \"layout\": \"{}\"", level.row_of().unwrap().name()), "");
            assert!(plan.contains(&format!("\"layout\": \"{}\"", level.row_of().unwrap().name())) && old_plan != plan);
            let before = dir.join(format!("{}-before", level.name()));
            save::save_planned(&Src::find(src.to_str().unwrap()).unwrap(), &before, 2, 5 << 30, level, &old_plan).unwrap();
            let (bu, bd, _) = read(&Checkpoint::open(&before).unwrap());
            let want_down = pack(&down_w, e * lat, inter).unwrap();
            assert!(matches!(bd, DownHost::Block(_)), "{level:?}: the second matrices of a directory of before are in the block layout");
            assert_eq!(parts(&bd), (tags.0, e, lat, inter, want_down.data.clone(), want_down.blocks.clone()), "{level:?}: a directory of before");
            assert!(bu.data == ju.data, "{level:?}: and the first");
            // (a directory of another level is refused, before anything is read)
            let other = if level == Layout::Int6 { Layout::Int5 } else { Layout::Int6 };
            let said = super::held(m, roles, hid, &saved, other, None).err().unwrap().msg;
            assert!(said.contains("packed in"), "{said}");
        }
        let said = super::held(m, roles, hid, &ck, Layout::Int3, None).err().unwrap().msg;
        assert!(said.contains("penguin, kestrel, swift and wren levels"), "{said}");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// What a tiny model of grid weights relies on of the packers (all three block layouts): weights that are integers of at most the layout's largest code
    /// times a power of two, with the largest code in every run of 32, come back exactly (the fit of the scale lands on the power of two).
    #[test]
    fn weights_on_the_grid_are_held_exactly() {
        use glyd_gpu::pack::{pack_int4, pack_int5, pack_int6, unpack_int4, unpack_int5, unpack_int6, Tiered};
        let (rows, cols) = (128usize, 96usize);
        let mut x = 7u64;
        let mut next = move || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        type Pack = fn(&[u16], usize, usize) -> Option<Tiered>;
        type Unpack = fn(&Tiered, &mut [u16]) -> bool;
        for (kmax, pack, unpack, power) in [
            (31i32, pack_int6 as Pack, unpack_int6 as Unpack, -8),
            (15, pack_int5 as Pack, unpack_int5 as Unpack, -7),
            (7, pack_int4 as Pack, unpack_int4 as Unpack, -6),
        ] {
            let w: Vec<u16> = (0..rows * cols)
                .map(|i| {
                    let k = if i % 32 == 0 {
                        if next() & 1 == 0 {
                            kmax
                        } else {
                            -kmax
                        }
                    } else {
                        (next() % (2 * kmax as u64 + 1)) as i32 - kmax
                    };
                    let v = k as f32 * 2f32.powi(power);
                    (v.to_bits() >> 16) as u16 // (a code of at most 5 bits times a power of two: bf16 holds it)
                })
                .collect();
            let p = pack(&w, rows, cols).unwrap();
            let mut back = vec![0u16; rows * cols];
            assert!(unpack(&p, &mut back));
            assert!(back == w, "kmax {kmax}: the grid's weights did not come back");
        }
    }

    #[test]
    fn a_bias_is_read_as_fp32_from_fp32_or_bf16() {
        let w = [0x3f80u16, 0xbf00, 0x3dcd];
        let widened: Vec<u8> = w.iter().flat_map(|&x| ((x as u32) << 16).to_le_bytes()).collect();
        let bf16: Vec<u8> = w.iter().flat_map(|x| x.to_le_bytes()).collect();
        assert_eq!(super::bias_f32("F32", &widened).unwrap(), widened);
        assert_eq!(super::bias_f32("BF16", &bf16).unwrap(), widened);
        assert!(super::bias_f32("F16", &[0; 4]).unwrap_err().msg.contains("F32 or BF16"));
    }
}
