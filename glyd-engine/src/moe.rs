//! A mixture-of-experts layer on the GPU (Linear GLY-101), the block Nemotron 3 Super, GLM-5.3-Flash, Gemma-4-26B-A4B and Qwen3.6-35B-A3B share: the router (kernels/moe.cu), the pairs of a token's k experts sorted by expert
//! (the library's `moe_way`), the experts' two products over E experts' packs stacked as one matrix (the library's `moe`: one grouped launch each, at 1 token and at a prompt's thousands), the activation between them,
//! the weighted sum, and a shared expert. Every step is a launch on the stream and none reads a result on the host, so a decode step's layers capture into the CUDA graph.
//!
//! What differs between the models is data: `Router` (softmax or sigmoid scores, a selection bias, renormalizing, a scale, a per-expert scale, logits in bf16 or fp32), `Act` (SwiGLU with a clamp or without, GELU,
//! relu squared without a gate), the shared expert (a gate or none) and the widths (Nemotron 3's router reads the model's width, its experts a latent). The experts are held in a block layout (`ExpertStack`: Int6, Int5
//! and Int4; the layout's tag is the pack's, and `block_step_bytes` is the one place the step size of a layout is, with the library's dispatch of the grouped products by that tag, `glyd_gpu_mma_moe`: a new block
//! layout, GLY-134's Int3, goes in both); `Lib::moe` takes the tiered and 12-bit layouts' stacks as well (the lossless levels), which no type here holds yet.
//!
//! A relu squared layer's second matrix may be held in a row layout instead (`moe_rows::Down::Rows`): its second product then reads only the rows of the neurons that are not zero, whatever the count of tokens
//! (src/moe_rows.rs, kernels/moe_rows.cu); the first product, the activation, the router and the weighted sum's rounding are the same.
//!
//! The arithmetic is fp32 with the roundings of vLLM's fused MoE: the router in fp32 (the logits of a bf16 gate rounded to bf16, as its Linear does), the first product stored in bf16 when the activation is not fused
//! into it (a clamp, relu squared) and the activation rounded once, each pair's down sums times its weight added in fp32 and rounded once.

use crate::dev::{Buf, Dev};
use crate::ffi::{err, Lib, Pack, Result, Stream, Words};
use crate::gpu_ops::GpuOps;
use crate::launch;
use crate::moe_hbg::{self, HbgExperts, HbgKern};
use crate::moe_rows::{Down, RowsKern};
use crate::offload::CPU_TOKENS;
use crate::weights::PackBuf;
use glyd_gpu::pack::{Tiered, FP4R_TAG, FP4_TAG, INT4_STEP_BYTES, INT4_TAG, INT5_STEP_BYTES, INT5_TAG, INT6_STEP_BYTES, INT6_TAG};

// The kernels' names: those the .cu sources give them, in every build (kname!).
/// Experts a router takes (moe_select8, 16 and 32: 256, 512 and 1,024), and choices a token has.
pub const MAX_EXPERTS: usize = 1024;
pub const MAX_TOPK: usize = 32;

// moe_select's flags and moe_act's modes (kernels/moe.cu defines the same: a test holds them equal).
const FLAG_SIGMOID: i32 = 1;
const FLAG_RENORM: i32 = 2;
const FLAG_BF16: i32 = 4;
const FLAG_SQRTSP: i32 = 8;
const ACT_SILU: i32 = 0;
const ACT_GELU: i32 = 1;
const ACT_RELU2: i32 = 2;
const ACT_SITU: i32 = 3;
const ACT_SILU_R: i32 = 4;

/// How a router scores the experts of a token.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scoring {
    /// softmax over all the experts (Qwen3.5 and 3.6 MoE, Gemma 4)
    Softmax,
    /// sigmoid of each expert's logit (DeepSeek's noaux_tc: GLM-5.x, Nemotron 3)
    Sigmoid,
    /// sqrt(softplus) of each expert's logit (DeepSeek-V4's noaux_tc)
    SqrtSoftplus,
}

/// What turns a token's router logits into its k experts and their weights (`moe_select`): the k largest scores (the bias added to a sigmoid's, to pick and not to weigh), the chosen scores divided by their
/// sum, times the per-expert scale of the chosen experts where there is one, times `scale`. Equal scores: the smaller expert first.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Router {
    pub scoring: Scoring,
    pub top_k: usize,
    /// the chosen scores divided by their sum (norm_topk_prob; Gemma 4 always)
    pub renorm: bool,
    /// routed_scaling_factor: 2.5 for GLM-5.x, 5.0 for Nemotron 3, else 1
    pub scale: f32,
    /// the logits rounded to bf16 before the scores (the gate is a bf16 Linear: Qwen3.5 and 3.6, Gemma 4); else fp32 (GLM-5.x, Nemotron 3 compute the router in fp32)
    pub bf16_logits: bool,
}

impl Router {
    /// Qwen3.5 and 3.6 MoE (8 of 256), Gemma 4 (8 of 128; with its per-expert scale in `MoeWeights::escale`): softmax, the k largest, renormalized.
    pub const fn softmax(top_k: usize) -> Router {
        Router { scoring: Scoring::Softmax, top_k, renorm: true, scale: 1.0, bf16_logits: true }
    }

    /// DeepSeek's noaux_tc as GLM-5.3-Flash (8 of 288, `scale` 2.5) and Nemotron 3 Super (22 of 512, `scale` 5.0) run it: sigmoid scores in fp32, the selection bias in `MoeWeights::bias`, renormalized, times `scale`.
    pub const fn sigmoid(top_k: usize, scale: f32) -> Router {
        Router { scoring: Scoring::Sigmoid, top_k, renorm: true, scale, bf16_logits: false }
    }

    /// moe_select's flags.
    pub fn flags(&self) -> i32 {
        (match self.scoring {
            Scoring::Sigmoid => FLAG_SIGMOID,
            Scoring::SqrtSoftplus => FLAG_SQRTSP,
            Scoring::Softmax => 0,
        }) | (if self.renorm { FLAG_RENORM } else { 0 })
            | (if self.bf16_logits { FLAG_BF16 } else { 0 })
    }
}

/// The activation between an expert's two products.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Act {
    /// silu(gate) up, the gate rows then the up rows of the first matrix; `limit`: the gate at most it and the up within it of zero (GLM-5.x: 10)
    SwiGlu { limit: Option<f32> },
    /// gelu_tanh(gate) up (Gemma 4)
    GeGlu,
    /// relu(x) squared, the first matrix without a gate (Nemotron 3)
    Relu2,
    /// situ: beta tanh(gate / beta) sigmoid(gate) x linear tanh(up / linear) (Kimi-K3: 4 and 25)
    Situ { beta: f32, linear: f32 },
}

impl Act {
    /// Whether the first matrix holds a gate and an up (twice the intermediate size in rows).
    pub fn gated(&self) -> bool {
        !matches!(self, Act::Relu2)
    }

    /// moe_act's mode and its two numbers (the clamp; situ's two caps).
    fn kernel(&self) -> (i32, f32, f32) {
        match *self {
            Act::SwiGlu { limit } => (ACT_SILU, limit.unwrap_or(f32::INFINITY), 0.0),
            Act::GeGlu => (ACT_GELU, f32::INFINITY, 0.0),
            Act::Relu2 => (ACT_RELU2, f32::INFINITY, 0.0),
            Act::Situ { beta, linear } => (ACT_SITU, beta, linear),
        }
    }
}

/// A layer's shape: E experts of a hidden size (the width the experts work in; the router reads `router_hidden`, the same but for Nemotron 3, whose experts work on a latent a quarter as wide) and an intermediate size.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MoeCfg {
    pub experts: usize,
    pub router_hidden: usize,
    pub hidden: usize,
    pub inter: usize,
    pub router: Router,
    pub act: Act,
}

impl MoeCfg {
    /// Rows of an expert's first matrix: gate and up, or up alone.
    pub fn up_rows(&self) -> usize {
        if self.act.gated() {
            2 * self.inter
        } else {
            self.inter
        }
    }

    /// The library's activation of the first product: 1 (SiLU) or 2 (GELU) where the product applies it as its sums are written (no clamp; the intermediate size a multiple of 64), else 0 and `moe_act` applies it.
    pub fn fused(&self) -> i64 {
        match self.act {
            Act::SwiGlu { limit: None } if self.inter.is_multiple_of(64) => 1,
            Act::GeGlu if self.inter.is_multiple_of(64) => 2,
            _ => 0,
        }
    }

    /// Whether the layer's shapes are ones the kernels and the library's products take (the block layouts: rows a multiple of 64, columns of 32): why not, else Ok.
    pub fn check(&self) -> Result<()> {
        let (e, k) = (self.experts, self.router.top_k);
        if e == 0 || e > MAX_EXPERTS {
            return err(format!("{e} experts: the router takes 1 to {MAX_EXPERTS}"));
        }
        if k == 0 || k > e.min(MAX_TOPK) {
            return err(format!("{k} experts a token of {e}: 1 to {} are taken", e.min(MAX_TOPK)));
        }
        if !self.router_hidden.is_multiple_of(8) || self.router_hidden == 0 {
            return err(format!("a router over {} values: not divisible by 8", self.router_hidden));
        }
        if !self.hidden.is_multiple_of(64) || !self.inter.is_multiple_of(32) || !self.up_rows().is_multiple_of(64) {
            return err(format!(
                "experts of hidden size {} and intermediate size {}: the products take rows divisible by 64 ({} and {}) and columns divisible by 32",
                self.hidden,
                self.inter,
                self.hidden,
                self.up_rows()
            ));
        }
        if let Act::SwiGlu { limit: Some(l) } = self.act {
            if l.is_nan() || l <= 0.0 {
                return err(format!("a SwiGLU clamp of {l}"));
            }
        }
        if let Act::Situ { beta, linear } = self.act {
            if !(beta > 0.0 && linear > 0.0) {
                return err(format!("situ's caps {beta} and {linear}"));
            }
        }
        Ok(())
    }
}

/// A block layout's bytes a step of 1,024 weights (64 rows by 16 columns) takes, by the tag in the top byte of its first word: the one table a new block layout (GLY-134's Int3) is added to, its scales being
/// 128 bytes a pair of steps as Int6's, Int5's and Int4's are (the FP4 layouts differ: `scale_bytes`).
pub fn block_step_bytes(tag: u32) -> Option<u64> {
    match tag {
        INT6_TAG => Some(INT6_STEP_BYTES as u64),
        INT5_TAG => Some(INT5_STEP_BYTES as u64),
        INT4_TAG | FP4_TAG | FP4R_TAG => Some(INT4_STEP_BYTES as u64),
        _ => None,
    }
}

/// A block layout's scale bytes for a matrix of `steps` steps of `ks` a row block: two bytes a run, one in the exact FP4 layout, and in the one with packed scales 2 bits a run and a byte a row.
pub fn scale_bytes(tag: u32, steps: u64, ks: u64) -> u64 {
    match tag {
        FP4_TAG => steps / 2 * 64,
        FP4R_TAG => steps / ks * (64 + ks * 8),
        _ => steps / 2 * 128,
    }
}

/// An expert's bytes in a block layout of `tag`: its codes and its scales (a matrix of whole row blocks of 64 rows, columns a multiple of 32).
pub fn expert_strides(tag: u32, rows: usize, cols: usize) -> Result<(u64, u64)> {
    let Some(step) = block_step_bytes(tag) else {
        return err(format!("a layout of tag {tag}: experts are held end to end in the block layouts (Int6, Int5, Int4) alone"));
    };
    if rows == 0 || !rows.is_multiple_of(64) || cols == 0 || !cols.is_multiple_of(32) {
        return err(format!("an expert of {rows} x {cols}: rows divisible by 64 and columns divisible by 32 are wanted"));
    }
    let steps = (rows / 64 * (cols / 16)) as u64;
    Ok((steps * step, scale_bytes(tag, steps, (cols / 16) as u64)))
}

/// E experts' packs end to end on the host: the codes and the scales as the library's grouped products read them (E packs of [rows, cols] are the pack of the E rows stacked).
pub struct Joined {
    pub tag: u32,
    pub tiers: [u32; 3],
    pub experts: usize,
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<u8>,
    pub scales: Vec<u8>,
}

impl Joined {
    /// The packs of the experts, one each, joined: all in one layout, of one shape.
    pub fn of(packs: &[&Tiered]) -> Result<Joined> {
        let Some(p0) = packs.first() else { return err("no experts") };
        let tag = p0.tiers[0] >> 24;
        let (data_bytes, scale_bytes) = expert_strides(tag, p0.rows, p0.cols)?;
        let (mut data, mut scales) = (Vec::with_capacity(packs.len() * data_bytes as usize), Vec::with_capacity(packs.len() * scale_bytes as usize));
        for (i, p) in packs.iter().enumerate() {
            if p.tiers != p0.tiers || (p.rows, p.cols) != (p0.rows, p0.cols) || p.data.len() as u64 != data_bytes || p.blocks.len() as u64 != scale_bytes {
                return err(format!(
                    "expert {i}: a pack of {} x {} (words {:x?}, {} + {} bytes), where expert 0's is {} x {} ({:x?}, {data_bytes} + {scale_bytes})",
                    p.rows,
                    p.cols,
                    p.tiers,
                    p.data.len(),
                    p.blocks.len(),
                    p0.rows,
                    p0.cols,
                    p0.tiers
                ));
            }
            data.extend_from_slice(&p.data);
            scales.extend_from_slice(&p.blocks);
        }
        Ok(Joined { tag, tiers: p0.tiers, experts: packs.len(), rows: p0.rows, cols: p0.cols, data, scales })
    }

    /// The pack of the stacked matrix [experts rows, cols] split back into `experts` experts (it is the same bytes).
    pub fn stacked(p: &Tiered, experts: usize) -> Result<Joined> {
        if experts == 0 || !p.rows.is_multiple_of(experts) {
            return err(format!("{} rows are not {experts} experts'", p.rows));
        }
        let rows = p.rows / experts;
        let tag = p.tiers[0] >> 24;
        let (data_bytes, scale_bytes) = expert_strides(tag, rows, p.cols)?;
        if p.data.len() as u64 != data_bytes * experts as u64 || p.blocks.len() as u64 != scale_bytes * experts as u64 {
            return err(format!(
                "a pack of {} + {} bytes, where {experts} experts of {rows} x {} take {} + {}",
                p.data.len(),
                p.blocks.len(),
                p.cols,
                data_bytes * experts as u64,
                scale_bytes * experts as u64
            ));
        }
        Ok(Joined { tag, tiers: p.tiers, experts, rows, cols: p.cols, data: p.data.clone(), scales: p.blocks.clone() })
    }
}

/// E experts' matrices [rows, cols] in one block layout, end to end in device memory: `pack` is the one matrix [E rows, cols] the library's grouped products take, and an expert's own pack (`expert`) is the
/// bytes at `e * data_stride` and `e * scale_stride`, for a single-expert product or a cache's slot. The scales' base (`block_base` of the tiered layout) is not read in these layouts: none is held.
pub struct ExpertStack {
    pub pack: Pack,
    pub experts: usize,
    /// The experts the buffers have room for: `experts`, or fewer in a stack made by `pooled`.
    pub slots: usize,
    pub rows: usize,
    pub cols: usize,
    pub tag: u32,
    pub data_stride: u64,
    pub scale_stride: u64,
    pub bytes: usize,
    /// An expert's whole rows where the stack holds this GPU's slice of each expert's (src/dynamic/tp.rs: a product is gathered after it); 0: whole experts.
    pub tp_rows: usize,
    /// Where the stack is an expert cache's places of a tiered pack (`placed`, src/tiering.rs `TieredPlaces`): a place's bytes of blocks; 0 otherwise.
    pub placed: u64,
    _bufs: Keep,
}

/// What an `ExpertStack`'s pack reads: its codes and scales (a block layout), or a lossless pack's buffers.
#[allow(dead_code)]
enum Keep {
    Bufs([Buf; 2]),
    Placed([Buf; 3]),
    Pack(crate::weights::PackBuf),
    None,
}

impl ExpertStack {
    /// The experts of a lossless level (the tiered layout, or the 12-bit one where it is smaller): the stacked matrix [experts rows, cols] packed as one, which the library's grouped products take
    /// as they take a block layout's stack. Its experts are not at fixed strides (the codes' lengths are the weights'), so no expert's own pack is taken of it (`expert`; a cache's places are in block layouts).
    pub fn lossless(p: crate::weights::PackBuf, experts: usize) -> Result<ExpertStack> {
        let (rows_all, cols) = (p.pack.rows as usize, p.pack.cols as usize);
        if experts == 0 || !rows_all.is_multiple_of(experts) {
            return err(format!("{rows_all} rows are not {experts} experts'"));
        }
        let tag = match p.pack.words {
            Words::Tiered { tiers, .. } => tiers[0] >> 24,
            Words::Twelve { .. } => 0,
        };
        Ok(ExpertStack {
            pack: p.pack,
            experts,
            slots: experts,
            rows: rows_all / experts,
            cols,
            tag,
            data_stride: 0,
            scale_stride: 0,
            bytes: p.bytes,
            tp_rows: 0,
            placed: 0,
            _bufs: Keep::Pack(p),
        })
    }

    /// No experts' buffers, the shape alone: a layer whose experts are held in another format (`MoeLayer::f8`).
    pub fn shape_only(experts: usize, rows: usize, cols: usize) -> ExpertStack {
        let pack = Pack { data: 0, words: Words::Tiered { blocks: 0, block_base: 0, tiers: [0; 3] }, rows: (experts * rows) as i64, cols: cols as i64 };
        ExpertStack { pack, experts, slots: experts, rows, cols, tag: 0, data_stride: 0, scale_stride: 0, bytes: 0, tp_rows: 0, placed: 0, _bufs: Keep::None }
    }

    /// The experts of `j` copied to the GPU.
    pub fn upload(j: &Joined) -> Result<ExpertStack> {
        let (data, scales) = (Buf::new(j.data.len())?, Buf::new(j.scales.len())?);
        data.write(0, &j.data)?;
        scales.write(0, &j.scales)?;
        ExpertStack::from_buffers(j.tag, j.tiers, j.experts, j.rows, j.cols, data, scales)
    }

    /// The stack of experts whose codes and scales are in device memory already (`experts` strides each: `expert_strides`).
    pub fn from_buffers(tag: u32, tiers: [u32; 3], experts: usize, rows: usize, cols: usize, data: Buf, scales: Buf) -> Result<ExpertStack> {
        let (data_stride, scale_stride) = expert_strides(tag, rows, cols)?;
        if tiers[0] >> 24 != tag || data.bytes as u64 != data_stride * experts as u64 || scales.bytes as u64 != scale_stride * experts as u64 {
            return err(format!(
                "{experts} experts of {rows} x {cols} in the layout of tag {tag} take {} + {} bytes: {} + {} are given",
                data_stride * experts as u64,
                scale_stride * experts as u64,
                data.bytes,
                scales.bytes
            ));
        }
        let pack = Pack { data: data.ptr, words: Words::Tiered { blocks: scales.ptr, block_base: 0, tiers }, rows: (experts * rows) as i64, cols: cols as i64 };
        let bytes = data.bytes + scales.bytes;
        Ok(ExpertStack {
            pack,
            experts,
            slots: experts,
            rows,
            cols,
            tag,
            data_stride,
            scale_stride,
            bytes,
            tp_rows: 0,
            placed: 0,
            _bufs: Keep::Bufs([data, scales]),
        })
    }

    /// Room for `slots` experts' matrices [rows, cols] in a block layout, standing for a layer's `experts` (zeroed): the pack the library's grouped products read has `experts` experts' rows, and a plan made of
    /// ids below `slots` (the expert cache's: the place each chosen expert is in, `MoeLayer::experts_via`) never reaches past the room.
    pub fn pooled(tag: u32, tiers: [u32; 3], experts: usize, slots: usize, rows: usize, cols: usize) -> Result<ExpertStack> {
        let (data_stride, scale_stride) = expert_strides(tag, rows, cols)?;
        if slots == 0 || slots > experts || tiers[0] >> 24 != tag {
            return err(format!("room for {slots} experts standing for {experts}, in the layout of tag {tag} (words {tiers:x?})"));
        }
        let (data, scales) = (Buf::zeroed(slots * data_stride as usize)?, Buf::zeroed(slots * scale_stride as usize)?);
        let pack = Pack { data: data.ptr, words: Words::Tiered { blocks: scales.ptr, block_base: 0, tiers }, rows: (experts * rows) as i64, cols: cols as i64 };
        let bytes = data.bytes + scales.bytes;
        Ok(ExpertStack { pack, experts, slots, rows, cols, tag, data_stride, scale_stride, bytes, tp_rows: 0, placed: 0, _bufs: Keep::Bufs([data, scales]) })
    }

    /// `slots` places for `experts` experts [rows, cols] of a stack in the tiered layout (`tiers` its pack's), each place an expert's steps (`data_stride` bytes), its blocks (`stride` bytes, `scale_stride`) and its own
    /// block entries (steps + 1): the store copies an expert's bytes of each into a place (src/tiering.rs `TieredPlaces`), the products read them by glyd_gpu_mma_moe_placed (`moe_tp`). Zeros until then.
    pub fn placed(tiers: [u32; 3], experts: usize, slots: usize, rows: usize, cols: usize, stride: u64) -> Result<ExpertStack> {
        if slots == 0 || slots > experts || !rows.is_multiple_of(64) || !cols.is_multiple_of(16) || !stride.is_multiple_of(4) || stride == 0 {
            return err(format!("{slots} places for {experts} tiered experts of {rows} x {cols}, blocks of {stride} bytes a place"));
        }
        let steps = rows / 64 * (cols / 16);
        let (data, blocks, entries) = (Buf::zeroed(slots * steps * 1280)?, Buf::zeroed(slots * stride as usize)?, Buf::zeroed(slots * (steps + 1) * 4)?);
        let pack = Pack {
            data: data.ptr,
            words: Words::Tiered { blocks: blocks.ptr, block_base: entries.ptr, tiers },
            rows: (experts * rows) as i64,
            cols: cols as i64,
        };
        let bytes = data.bytes + blocks.bytes + entries.bytes;
        Ok(ExpertStack {
            pack,
            experts,
            slots,
            rows,
            cols,
            tag: tiers[0] >> 24,
            data_stride: (steps * 1280) as u64,
            scale_stride: stride,
            bytes,
            tp_rows: 0,
            placed: stride,
            _bufs: Keep::Placed([data, blocks, entries]),
        })
    }

    /// An expert's whole rows: `tp_rows` where the stack holds slices of them, else its own.
    pub fn whole_rows(&self) -> usize {
        if self.tp_rows > 0 {
            self.tp_rows
        } else {
            self.rows
        }
    }

    /// The addresses of the codes and of the scales (of the first expert's, or of the first place).
    pub fn buffers(&self) -> [u64; 2] {
        let Words::Tiered { blocks, .. } = self.pack.words else { unreachable!("an ExpertStack holds a block layout") };
        [self.pack.data, blocks]
    }

    /// Expert `e`'s own pack [rows, cols].
    pub fn expert(&self, e: usize) -> Pack {
        assert!(self.data_stride > 0, "a lossless stack's experts are not at fixed strides");
        let Words::Tiered { blocks, block_base, tiers } = self.pack.words else { unreachable!("an ExpertStack holds a block layout") };
        Pack {
            data: self.pack.data + e as u64 * self.data_stride,
            words: Words::Tiered { blocks: blocks + e as u64 * self.scale_stride, block_base, tiers },
            rows: self.rows as i64,
            cols: self.cols as i64,
        }
    }
}

/// The shared expert: a dense gated (or relu squared, per the layer's `Act`) MLP every token goes through, added to the routed experts' sum: `up` [up rows, in] (gate rows then up rows), `down` [out, inter], and
/// the gate where there is one (Qwen3.5 and 3.6: sigmoid(x . gate) scales its output), [in] bf16.
pub struct Shared {
    pub inter: usize,
    pub up: PackBuf,
    pub down: PackBuf,
    pub gate: Option<Buf>,
}

/// The router's and the experts' weights.
pub struct MoeWeights {
    /// [experts, router_hidden] bf16
    pub router: Buf,
    /// [experts] fp32: added to the sigmoid scores to pick the experts (not to weigh them)
    pub bias: Option<Buf>,
    /// [experts] fp32: each chosen expert's weight times its entry (Gemma 4's per_expert_scale)
    pub escale: Option<Buf>,
    pub up: ExpertStack,
    pub down: Down,
}

/// The layer's kernels (kernels/moe.cu) with typed launches. Device addresses are `u64`.
pub struct MoeKern {
    logits: crate::kt::moe_logits,
    select: [crate::dev::ByArgs<crate::dev::ArgsOf<crate::kt::moe_select8>>; 3],
    act: crate::kt::moe_act,
    add: crate::kt::moe_add,
    add_f32: crate::kt::moe_add_f32,
    // the weighed experts' activation, each pair's weight in it, and the routing by token (kernels/dsv4.cu: the release's, DeepSeek-V4's)
    act_w: crate::kt::v4_moe_act,
    hash: crate::kt::v4_hash_select,
    // the FP8 experts' sum (each pair's own row): the pairs' rows, the weighted sum
    inv: crate::kt::moe_inv,
    wsum: crate::kt::moe_wsum,
    // the FP8 experts' places in the expert cache: their heads made their own (src/f8.rs F8Places)
    f8_rebase: crate::kt::moe_f8_rebase,
    // the normed experts' router of fp32 weights and their normed weighted sum (kernels/muse.cu: transformers', Muse Spark's)
    router_f32: crate::kt::muse_router,
    norm_sum: crate::kt::muse_moe_sum,
    // the expert cache planned on the GPU (src/devplan.rs): the plan, the copies into places, the wait for the host's pairs, their sums brought to the GPU
    xp_plan: crate::kt::xp_plan,
    xp_fill: crate::kt::xp_fill,
    xp_wait: crate::kt::xp_wait,
    xp_take: crate::kt::xp_take,
}

/// The select kernel for `experts` (8, 16 or 32 registers a lane: 256, 512 or 1,024 experts).
fn select_index(experts: usize) -> Result<usize> {
    match experts {
        1..=256 => Ok(0),
        257..=512 => Ok(1),
        513..=MAX_EXPERTS => Ok(2),
        _ => err(format!("{experts} experts: the router takes 1 to {MAX_EXPERTS}")),
    }
}

fn aligned(p: u64, what: &str) -> Result<()> {
    if !p.is_multiple_of(16) {
        return err(format!("{what}: {p:#x} is not 16-byte aligned"));
    }
    Ok(())
}

impl MoeKern {
    pub fn load(dev: &Dev) -> Result<MoeKern> {
        Ok(MoeKern {
            logits: dev.typed()?,
            select: [dev.by_args::<crate::kt::moe_select8>()?, dev.by_args::<crate::kt::moe_select16>()?, dev.by_args::<crate::kt::moe_select32>()?],
            act: dev.typed()?,
            add: dev.typed()?,
            add_f32: dev.typed()?,
            act_w: dev.typed()?,
            hash: dev.typed()?,
            inv: dev.typed()?,
            wsum: dev.typed()?,
            f8_rebase: dev.typed()?,
            router_f32: dev.typed()?,
            norm_sum: dev.typed()?,
            xp_plan: dev.typed()?,
            xp_fill: dev.typed()?,
            xp_wait: dev.typed()?,
            xp_take: dev.typed()?,
        })
    }

    /// logits [t, e] (fp32) = x [t, h] W^T for a bf16 W [e, h], the rows of x `xs` elements apart: the router's logits in fp32 (and the shared expert's gate's, with e = 1). h a multiple of 8; x, W and xs 16-byte aligned.
    ///
    /// # Safety
    /// Live device buffers of the sizes the kernel reads and writes (kernels/moe.cu).
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn logits(&self, s: Stream, x: u64, xs: usize, w: u64, logits: u64, t: usize, e: usize, h: usize) -> Result<()> {
        if !h.is_multiple_of(8) || !xs.is_multiple_of(8) || xs < h {
            return err(format!("a router over {h} values in rows {xs} apart: both divisible by 8, the rows at least {h} apart"));
        }
        aligned(x, "the router's input")?;
        aligned(w, "the router's weights")?;
        if t == 0 || e == 0 {
            return Ok(());
        }
        launch!(self.logits, [(e as u32).div_ceil(2), (t as u32).div_ceil(8), 1], [256, 1, 1], s; x, xs as i32, w, logits, t as i32, e as i32, h as i32)
    }

    /// For each of `t` tokens, its `k` experts into ids [t, k] (int64, as `Lib::moe_way` reads them) and their weights into wts [t, k] (fp32) from logits [t, e] (fp32): see `Router`. `bias` and `escale` (fp32 [e]): 0 for none.
    ///
    /// # Safety
    /// As `logits`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn select(&self, s: Stream, logits: u64, bias: u64, escale: u64, ids: u64, wts: u64, t: usize, e: usize, k: usize, r: &Router) -> Result<()> {
        let kern = &self.select[select_index(e)?];
        if k == 0 || k > MAX_TOPK || k > e {
            return err(format!("{k} experts a token of {e}"));
        }
        if t == 0 {
            return Ok(());
        }
        launch!(kern, [(t as u32).div_ceil(4), 1, 1], [128, 1, 1], s; logits, bias, escale, ids, wts, t as i32, e as i32, k as i32, r.flags(), r.scale)
    }

    /// out [rows, inter] = the activation of gu [rows, 2 inter] (gate then up) or, for relu squared, of gu [rows, inter] (out = gu is fine): see `Act`. inter a multiple of 8.
    ///
    /// # Safety
    /// As `logits`.
    pub unsafe fn act(&self, s: Stream, gu: u64, out: u64, rows: usize, inter: usize, a: &Act) -> Result<()> {
        if !inter.is_multiple_of(8) {
            return err(format!("intermediate size {inter}: not divisible by 8"));
        }
        aligned(gu, "the activation's input")?;
        aligned(out, "the activation's output")?;
        if rows == 0 {
            return Ok(());
        }
        let (mode, lim, lin) = a.kernel();
        launch!(self.act, [rows as u32, (inter as u32 / 8).div_ceil(256), 1], [256, 1, 1], s; gu, out, inter as i32, mode, lim, lin)
    }

    /// `act`'s SwiGLU as transformers computes it on bf16 tensors: out = bf16(bf16(silu(gate)) up), no clamp (the normed experts').
    ///
    /// # Safety
    /// As `logits`.
    pub unsafe fn act_rounded(&self, s: Stream, gu: u64, out: u64, rows: usize, inter: usize) -> Result<()> {
        if !inter.is_multiple_of(8) {
            return err(format!("intermediate size {inter}: not divisible by 8"));
        }
        aligned(gu, "the activation's input")?;
        aligned(out, "the activation's output")?;
        if rows == 0 {
            return Ok(());
        }
        launch!(self.act, [rows as u32, (inter as u32 / 8).div_ceil(256), 1], [256, 1, 1], s; gu, out, inter as i32, ACT_SILU_R, f32::INFINITY, 0f32)
    }

    /// y [rows, h] += g sh [rows, h], g = sigmoid(gate[row]) for an fp32 `gate` [rows] (0: g = 1).
    ///
    /// # Safety
    /// As `logits`.
    /// y [t, h] = each token's rows of `rows` [t k, h] (each pair's own, in the plan's order) times its weights `wts` [t, k], summed in fp32 in its choices' order; `inv` int32 [t k] scratch.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn wsum(&self, s: Stream, rows: u64, plan: u64, inv: u64, wts: u64, y: u64, t: usize, k: usize, e: usize, h: usize) -> Result<()> {
        let p = t * k;
        launch!(self.inv, [(p as u32).div_ceil(256), 1, 1], [256, 1, 1], s; plan, inv, p as i32, e as i32)?;
        launch!(self.wsum, [t as u32, (h as u32 / 8).div_ceil(256), 1], [256, 1, 1], s; rows, inv, wts, y, k as i32, h as i32)
    }

    /// The heads of an FP8 stack's places (`F8Mat::place_heads`: `n` heads at `heads`, `per` a place, a place's var `vmax` words) made their own place's, after the store's copies.
    ///
    /// # Safety
    /// `heads` a live device buffer of `n` heads of 8 words.
    pub unsafe fn f8_rebase(&self, s: Stream, heads: u64, n: usize, per: usize, vmax: usize) -> Result<()> {
        if n == 0 || per == 0 || vmax == 0 {
            return Ok(());
        }
        launch!(self.f8_rebase, [(n as u32).div_ceil(256), 1, 1], [256, 1, 1], s; heads, n as i64, per as i64, vmax as i64)
    }

    /// y [n] bf16 += x [n] fp32 (the host's sums into a block layout's routed sum).
    ///
    /// # Safety
    /// y and x live device buffers of n values.
    pub unsafe fn add_f32(&self, s: Stream, y: u64, x: u64, n: usize) -> Result<()> {
        if n == 0 {
            return Ok(());
        }
        launch!(self.add_f32, [(n as u32).div_ceil(256), 1, 1], [256, 1, 1], s; y, x, n as i64)
    }

    pub unsafe fn add(&self, s: Stream, y: u64, sh: u64, gate: u64, rows: usize, h: usize) -> Result<()> {
        if !h.is_multiple_of(8) {
            return err(format!("width {h}: not divisible by 8"));
        }
        aligned(y, "the routed sum")?;
        aligned(sh, "the shared expert's output")?;
        if rows == 0 {
            return Ok(());
        }
        launch!(self.add, [rows as u32, (h as u32 / 8).div_ceil(256), 1], [256, 1, 1], s; y, sh, gate, h as i32)
    }
}

/// A hummingbird layer's experts in the expert cache (src/offload.rs `ExpertStore`, docs/design/hummingbird.md): the store, the layer's two groups' numbers there (one a bit width; None where the layer has no expert of that width), each
/// expert's group and number in it (the store places a call's pairs by them), the GPU the layer runs on, and the buffers a call's places, the sum's ids and the host's sums go through (a call of at most `offload::CPU_TOKENS` tokens).
pub struct HbCache {
    pub store: std::rc::Rc<crate::offload::ExpertStore>,
    pub layers: [Option<usize>; 2],
    pub group_of: Vec<u8>,
    pub local_of: Vec<u16>,
    pub ordinal: i32,
    ways: [Buf; 2],
    sids: Buf,
    part: Buf,
}

impl HbCache {
    /// The cache of a layer of `top_k` choices a token and experts of `width`, its groups `layers` of `store` on the GPU `ordinal`. The places of a group the layer does not have stay -1 (no pair of it is on the GPU).
    pub fn new(
        store: std::rc::Rc<crate::offload::ExpertStore>,
        layers: [Option<usize>; 2],
        group_of: Vec<u8>,
        local_of: Vec<u16>,
        ordinal: i32,
        top_k: usize,
        width: usize,
    ) -> Result<HbCache> {
        let pairs = crate::offload::CPU_TOKENS * top_k.max(1);
        // (-1 in every place: no pair of a group the store does not fill is on the GPU)
        let way = || crate::weights::upload_bytes(&vec![0xFFu8; pairs * 8]);
        let (ways, sids, part) = ([way()?, way()?], Buf::new(pairs * 8)?, Buf::new(crate::offload::CPU_TOKENS * width * 4)?);
        Ok(HbCache { store, layers, group_of, local_of, ordinal, ways, sids, part })
    }

    /// Device bytes of the cache's own buffers.
    pub fn bytes(&self) -> usize {
        self.ways[0].bytes + self.ways[1].bytes + self.sids.bytes + self.part.bytes
    }
}

/// A layer's experts in the expert cache (src/offload.rs `ExpertStore`, docs/design/runtime.md §6): the store, the layer's number there, and the GPU the layer runs on. The layer's stacks are the cache's places.
pub struct ExpertCache {
    pub store: std::rc::Rc<crate::offload::ExpertStore>,
    pub layer: usize,
    pub ordinal: i32,
}

/// How a hummingbird layer in the cache sums its pairs (`MoeLayer::hb_cached`): linear in the experts' outputs with the router's weights (`experts_via`), with the weights in the activation (`forward_weighed`, `act_w`:
/// `ones` fp32 [t k] for the sum), or each expert's output normed before it is weighed (`forward_normed`: the input scale `post_w`, `eps`, the pairs' rows `inv`).
#[derive(Clone, Copy, Debug)]
pub enum HbSum {
    Linear,
    Weighed { ones: u64, lim: f32, fp8: bool },
    Normed { post_w: u64, eps: f32, inv: u64 },
}

/// A layer: its config, its weights and its shared expert. The buffers its calls work in are the model's (`MoeScratch`, one for all its layers).
pub struct MoeLayer {
    pub cfg: MoeCfg,
    pub w: MoeWeights,
    pub shared: Option<Shared>,
    /// The experts in the FP8 format, as an FP8 checkpoint has them (`w.up` and `w.down` their shapes alone): the products `F8Mat::moe`'s, the weighted sum of each pair's own row.
    pub f8: Option<crate::f8::F8Experts>,
    /// The shared expert's two matrices in the FP8 format (its gate and up as one, its down), as the release has them; `shared` then holds their shapes alone (`PackBuf::shape_only`).
    pub shared8: Option<(crate::f8::F8Mat, crate::f8::F8Mat)>,
    /// The experts in the hummingbird format (src/moe_hbg.rs: codebook rows of 2 and 3 bits in the basis of a rotation, any width, gated or not): the products are `HbgKern`'s and the weighted sum is in the rotated basis, rotated back once
    /// (`w.up` and `w.down` their shapes alone).
    pub hb: Option<HbgExperts>,
    /// Where `hb`'s experts are in the expert cache's places, the computer's memory holding them all (or some, the rest read from the packed directory): the store and the layer's groups there (`with_hb_cache`).
    pub hb_cache: Option<HbCache>,
    /// Where the experts of the stacks (`ExpertStack::pooled`: the cache's places) are in the expert cache, every one in the computer's memory: the store, the layer's number there, the GPU (`with_cache`; R12, #232).
    pub cache: Option<ExpertCache>,
}

impl MoeLayer {
    /// The layer, the weights' shapes held to the config's (and the shared expert's to its intermediate size and the layer's activation).
    pub fn new(cfg: MoeCfg, w: MoeWeights, shared: Option<Shared>) -> Result<MoeLayer> {
        cfg.check()?;
        let (e, hidden, inter, up_rows) = (cfg.experts, cfg.hidden, cfg.inter, cfg.up_rows());
        let shape = |s: &ExpertStack, what: &str, rows: usize, cols: usize| -> Result<()> {
            if (s.experts, s.whole_rows(), s.cols) != (e, rows, cols) {
                return err(format!(
                    "the experts' {what}: {} experts of {} x {}, where the layer has {e} of {rows} x {cols}",
                    s.experts,
                    s.whole_rows(),
                    s.cols
                ));
            }
            Ok(())
        };
        if w.up.tp_rows > 0 && cfg.fused() != 0 {
            return err("experts sliced across GPUs with the activation in the first product: its gate and up rows are on different GPUs");
        }
        shape(&w.up, "first matrix", up_rows, hidden)?;
        match &w.down {
            Down::Block(d) => shape(d, "second matrix", hidden, inter)?,
            rows => rows.check(&cfg)?,
        }
        if w.router.bytes < e * cfg.router_hidden * 2 || w.bias.as_ref().is_some_and(|b| b.bytes < e * 4) || w.escale.as_ref().is_some_and(|b| b.bytes < e * 4)
        {
            return err("the router's weights, bias or per-expert scale are smaller than the layer's experts take");
        }
        if cfg.router.scoring == Scoring::Softmax && w.bias.is_some() {
            return err("a selection bias with softmax scores: only the sigmoid scores take one");
        }
        if let Some(sx) = &shared {
            let (rows, in_, out) = (if cfg.act.gated() { 2 * sx.inter } else { sx.inter }, sx.up.pack.cols as usize, sx.down.whole_rows());
            if sx.up.whole_rows() != rows
                || sx.down.pack.cols as usize != sx.inter
                || !in_.is_multiple_of(8)
                || !out.is_multiple_of(8)
                || !sx.inter.is_multiple_of(8)
            {
                return err(format!("the shared expert's matrices {} x {} and {} x {}, where its intermediate size {} wants {rows} x (a width divisible by 8) and (a width divisible by 8) x {}", sx.up.pack.rows, in_, sx.down.pack.rows, sx.down.pack.cols, sx.inter, sx.inter));
            }
            if sx.gate.as_ref().is_some_and(|g| g.bytes < in_ * 2) {
                return err("the shared expert's gate is smaller than its input");
            }
        }
        Ok(MoeLayer { cfg, w, shared, f8: None, shared8: None, hb: None, hb_cache: None, cache: None })
    }

    /// The shared expert's matrices in the FP8 format (`shared8`): the shapes of `shared`'s, which holds them alone.
    pub fn with_shared8(mut self, up: crate::f8::F8Mat, down: crate::f8::F8Mat) -> Result<MoeLayer> {
        let Some(sx) = &self.shared else { return err("FP8 shared expert matrices for a layer without a shared expert") };
        if (up.whole_n(), up.k, down.whole_n(), down.k)
            != (sx.up.pack.rows as usize, sx.up.pack.cols as usize, sx.down.pack.rows as usize, sx.down.pack.cols as usize)
            || sx.up.bytes + sx.down.bytes != 0
        {
            return err("the FP8 shared expert's matrices and the shapes held beside them differ");
        }
        self.shared8 = Some((up, down));
        Ok(self)
    }

    /// y = x times the shared expert's first (`up`) or second matrix, in whichever format it is held.
    ///
    /// # Safety
    /// As `route`.
    #[allow(clippy::too_many_arguments)]
    unsafe fn shared_lin(&self, lib: &Lib, sc: &MoeScratch, s: Stream, up: bool, x: u64, t: usize, y: u64) -> Result<()> {
        self.shared_lin_in(lib, (&sc.ws, &sc.done), s, up, x, t, y)
    }

    /// `shared_lin` with the library's workspace and counters `wd` (the scratch's, or `Beside`'s for the products on its stream).
    #[allow(clippy::too_many_arguments)]
    unsafe fn shared_lin_in(&self, lib: &Lib, wd: (&Buf, &Buf), s: Stream, up: bool, x: u64, t: usize, y: u64) -> Result<()> {
        let (ws, done) = wd;
        match (&self.shared8, &self.shared) {
            (Some((u, d)), _) => (if up { u } else { d }).linear(lib, x, t, y, ws.ptr, ws.bytes, done.ptr, s),
            (None, Some(sx)) => linear_tp(lib, if up { &sx.up } else { &sx.down }, x, t, y, ws, done, s),
            (None, None) => err("no shared expert"),
        }
    }

    /// The layer with its experts in the FP8 format (`f8`): the router and its bias, the shared expert, as `new`'s. The first matrices' rows must hold the second's outputs (the second product's rows go
    /// where the first's were).
    pub fn new_f8(cfg: MoeCfg, router: Buf, bias: Option<Buf>, f8: crate::f8::F8Experts, shared: Option<Shared>) -> Result<MoeLayer> {
        let (e, up_rows) = (cfg.experts, cfg.up_rows());
        if !cfg.act.gated() || cfg.fused() != 0 || up_rows < cfg.hidden || !cfg.hidden.is_multiple_of(8) {
            return err("FP8 experts: a gated activation of its own kernel, the first matrices' rows at least the width, a width divisible by 8");
        }
        // (a stack of this GPU's slices of each expert's rows: the whole experts' shapes, src/dynamic/tp.rs; the expert cache's places: an expert's rows)
        let whole = |m: &crate::f8::F8Mat| match m.place {
            Some(p) => e * p.rows,
            None if m.tp_rows > 0 => e * m.tp_rows,
            None => m.n,
        };
        if f8.up.places() != f8.down.places() {
            return err(format!("FP8 experts in {} places of the first matrices and {} of the second", f8.up.places(), f8.down.places()));
        }
        if f8.experts != e || (whole(&f8.up), f8.up.k, whole(&f8.down), f8.down.k) != (e * up_rows, cfg.hidden, e * cfg.hidden, cfg.inter) {
            return err(format!(
                "FP8 experts of {} x {} and {} x {}, where the layer's {e} take {} x {} and {} x {}",
                whole(&f8.up),
                f8.up.k,
                whole(&f8.down),
                f8.down.k,
                e * up_rows,
                cfg.hidden,
                e * cfg.hidden,
                cfg.inter
            ));
        }
        let w = MoeWeights {
            router,
            bias,
            escale: None,
            up: ExpertStack::shape_only(e, up_rows, cfg.hidden),
            down: Down::Block(ExpertStack::shape_only(e, cfg.hidden, cfg.inter)),
        };
        let mut l = MoeLayer::new(cfg, w, shared)?;
        l.f8 = Some(f8);
        Ok(l)
    }

    /// The layer with its routed experts in the hummingbird format (`hb`): the router and its selection bias and per-expert scale where there are any, the shared expert, as `new`'s; the stacks are shapes alone. The experts must be the
    /// config's (their number, the width they work in, the intermediate size, a gate where the activation has one). The layer's forwards (`experts_products` and `experts_sum`, `forward_normed`, `forward_weighed`) run `HbgKern`'s products in
    /// place of the library's: the router, the activation and the sum of each family as they are, R once a token before the first product, R^T once a token after the weighted sum (once a pair for Muse Spark, whose sum is not linear).
    pub fn new_hb(cfg: MoeCfg, router: Buf, bias: Option<Buf>, escale: Option<Buf>, hb: HbgExperts, shared: Option<Shared>) -> Result<MoeLayer> {
        hb.shape.check(&cfg)?;
        let (e, up_rows) = (cfg.experts, cfg.up_rows());
        let w = MoeWeights {
            router,
            bias,
            escale,
            up: ExpertStack::shape_only(e, up_rows, cfg.hidden),
            down: Down::Block(ExpertStack::shape_only(e, cfg.hidden, cfg.inter)),
        };
        let mut l = MoeLayer::new(cfg, w, shared)?;
        l.hb = Some(hb);
        Ok(l)
    }

    /// The hummingbird layer with its experts in the expert cache's places (`HbgExperts::pooled`; the store holds the layer's groups as `cache.layers`): each call reads the router's choices on the host to place them (`hb_cached`),
    /// so a step with it is not captured in a graph.
    pub fn with_hb_cache(mut self, cache: HbCache) -> Result<MoeLayer> {
        if self.hb.is_none() {
            return err("an expert cache of hummingbird experts for a layer whose experts are not hummingbird's");
        }
        if cache.group_of.len() != self.cfg.experts || cache.local_of.len() != self.cfg.experts {
            return err(format!(
                "an expert cache's tables of {} and {} experts, for a layer of {}",
                cache.group_of.len(),
                cache.local_of.len(),
                self.cfg.experts
            ));
        }
        self.hb_cache = Some(cache);
        Ok(self)
    }

    /// The layer with its experts in the expert cache (R12, #232): its stacks are places (`ExpertStack::pooled`) the store copies the experts the router asks for into, every expert being in the computer's memory (or on
    /// the disk); each call reads the router's choices on the host, so a step with it is not captured in a graph. The families' forwards (`experts_via`, `forward_weighed`, `forward_normed`) run their products over the places.
    pub fn with_cache(mut self, cache: ExpertCache) -> Result<MoeLayer> {
        // (FP8 experts: their matrices the places, `F8Places::places`)
        let slots = self.f8.as_ref().map_or(self.w.up.slots, |f| f.up.places());
        if self.hb.is_some() || slots >= self.cfg.experts || slots == 0 {
            return err(format!(
                "an expert cache for a layer whose experts are not in places of its own ({slots} places for {} experts{})",
                self.cfg.experts,
                if self.f8.is_some() { ", FP8 as released" } else { "" }
            ));
        }
        self.cache = Some(cache);
        Ok(self)
    }

    /// The expert cache's store the layer's experts are in (none: all on the GPU).
    pub fn store(&self) -> Option<&std::rc::Rc<crate::offload::ExpertStore>> {
        self.cache.as_ref().map(|c| &c.store).or(self.hb_cache.as_ref().map(|c| &c.store))
    }

    /// Whether a call reads the router's choices on the host (the experts in the expert cache): no graph captures it.
    pub fn reads_host(&self) -> bool {
        self.hb_cache.is_some() || self.cache.as_ref().is_some_and(|c| !c.store.device_on(c.layer))
    }

    /// The call of the expert cache for `t` tokens whose choices the router made (`ids`, `wts` of the scratch; `z` the experts' input as the host reads it): the store reads the choices, copies the misses into places (in waves
    /// where a call has more experts than places) or keeps them for the host where it has a half, `products` runs the products over the places of a wave (`way`, the waves), and `experts_sum` adds the host's sums.
    ///
    /// # Safety
    /// As `route`.
    #[allow(clippy::too_many_arguments)]
    unsafe fn cached(&self, sc: &MoeScratch, s: Stream, z: u64, t: usize, y: u64, products: &mut dyn FnMut(usize) -> Result<()>) -> Result<()> {
        let c = self.cache.as_ref().ok_or_else(|| crate::ffi::Error("no expert cache".into()))?;
        let at = crate::offload::At { ordinal: c.ordinal, stream: s };
        self.via_store(&c.store, c.layer, at, sc, z, t, y, &mut || Ok(()), products)
    }

    /// A call of `t` tokens through layer `layer` of the expert cache `store` (the router's choices `ids` and `wts` of the scratch, `z` the experts' input [t, hidden]): where the store plans the layer on
    /// the GPU (src/devplan.rs; a decode's call), the plan, the misses' copies into places, `early`, the products over the places (`products`, one wave), the promotions' copies behind them, the wait for
    /// the host's pairs and the experts' sum with the host's sums, all queued on the stream with no host in them (a graph holds them); else the store plans the call on the host
    /// (`ExpertStore::run_early_at`, `early` queued while it reads the choices).
    ///
    /// # Safety
    /// As `route`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn via_store(
        &self,
        store: &crate::offload::ExpertStore,
        layer: usize,
        at: crate::offload::At,
        sc: &MoeScratch,
        z: u64,
        t: usize,
        y: u64,
        early: &mut dyn FnMut() -> Result<()>,
        products: &mut dyn FnMut(usize) -> Result<()>,
    ) -> Result<()> {
        let (s, k, width) = (at.stream, self.cfg.router.top_k, self.cfg.hidden);
        let call = crate::offload::Call { ids: sc.ids.ptr, wts: sc.wts.ptr, z, way: sc.way.ptr, part: sc.cpu.ptr, tokens: t, pairs: t * k };
        if let Some(d) = store.device_args(layer, t, call.pairs) {
            if sc.cpu.bytes < t * width * 4 || sc.way.bytes < call.pairs * 8 {
                return err(format!("{t} tokens of {width} values: the scratch has no room for the host's sums and the pairs' places"));
            }
            let kk = &sc.k;
            launch!(kk.xp_plan, [1, 1, 1], [256, 1, 1], s; call.ids, call.wts, z, t as i32, k as i32, width as i32, d.place_of, d.expert_of, d.last, d.hostlast, d.refb, d.meta, d.params, d.experts as i32, d.slots as i32, sc.way.ptr, d.fills, d.promos, d.mail, d.seq, d.layer as i32, d.max_pairs as i32)?;
            launch!(kk.xp_fill, [16, d.slots as u32, 1], [256, 1, 1], s; d.fills, d.meta, 8i32, d.reg, d.nreg as i32)?;
            early()?;
            crate::dev::memset_async(y, 0, t * width * 2, s)?;
            products(1)?;
            launch!(kk.xp_fill, [16, d.slots as u32, 1], [256, 1, 1], s; d.promos, d.meta, 9i32, d.reg, d.nreg as i32)?;
            launch!(kk.xp_wait, [1, 1, 1], [32, 1, 1], s; d.meta, d.ans, crate::devplan::XP_LIMIT)?;
            let n = t * width;
            launch!(kk.xp_take, [(n as u32).div_ceil(256), 1, 1], [256, 1, 1], s; d.meta, d.sums, sc.cpu.ptr, n as i64)?;
            return self.experts_sum(sc, s, t, y, sc.way.ptr, sc.cpu.ptr);
        }
        store.run_early_at(layer, at, &call, early, &mut |stage, _, waves, hy| match stage {
            crate::offload::Stage::Products if hy.gpu => products(waves),
            // (every pair the host's: the sum is its sums alone)
            crate::offload::Stage::Products => crate::dev::memset_async(y, 0, t * width * 2, s),
            crate::offload::Stage::Sum => self.experts_sum(sc, s, t, y, hy.ids, hy.extra),
        })
    }

    /// `cached` for FP8 experts in places (`F8Places`, copied only): in calls of at most `f8::MOE_TOKENS` tokens whose pairs are no more than the places (each call's experts in one wave, every row of the sum the call's
    /// own), the places' heads made their own (`MoeKern::f8_rebase`) after the store's copies, then the products and the weighted sum as `experts_products` runs them, over the places.
    ///
    /// # Safety
    /// As `route`.
    #[allow(clippy::too_many_arguments)]
    unsafe fn f8_cached(&self, lib: &Lib, sc: &MoeScratch, s: Stream, f: &crate::f8::F8Experts, x: u64, t: usize, y: u64) -> Result<()> {
        let cache = self.cache.as_ref().ok_or_else(|| crate::ffi::Error("no expert cache".into()))?;
        let (c, e, k) = (&self.cfg, self.cfg.experts, self.cfg.router.top_k);
        let tn_max = (f.up.places() / k).clamp(1, crate::f8::MOE_TOKENS);
        for t0 in (0..t).step_by(tn_max) {
            let tn = (t - t0).min(tn_max);
            let (ids, wts, xs, ys) =
                (sc.ids.ptr + (t0 * k * 8) as u64, sc.wts.ptr + (t0 * k * 4) as u64, x + (t0 * c.hidden * 2) as u64, y + (t0 * c.hidden * 2) as u64);
            let pairs = tn * k;
            let call = crate::offload::Call { ids, wts, z: xs, way: sc.way.ptr, part: sc.cpu.ptr, tokens: tn, pairs };
            let at = crate::offload::At { ordinal: cache.ordinal, stream: s };
            cache.store.run_at(cache.layer, at, &call, &mut |stage, _, waves, hy| {
                if stage == crate::offload::Stage::Sum {
                    return if hy.ids != 0 || hy.extra != 0 {
                        err("FP8 experts in the cache: a pair not on the GPU (they are copied, never computed on the host)")
                    } else {
                        Ok(())
                    };
                }
                if waves > 1 || !hy.gpu {
                    return err("FP8 experts in the cache: every pair of a call on the GPU in one wave");
                }
                for m in [&f.up, &f.down] {
                    if let Some((heads, n, per, vmax)) = m.place_heads() {
                        sc.k.f8_rebase(s, heads, n, per, vmax)?;
                    }
                }
                lib.moe_way(sc.way.ptr, pairs as i64, e as i64, sc.plan.ptr, s)?;
                f.up.moe(lib, e, xs, sc.plan.ptr, tn, k, true, sc.a.ptr, sc.ws.ptr, sc.ws.bytes, sc.done.ptr, s)?;
                sc.k.act(s, sc.a.ptr, sc.h.ptr, pairs, c.inter, &c.act)?;
                f.down.moe(lib, e, sc.h.ptr, sc.plan.ptr, tn, k, false, sc.a.ptr, sc.ws.ptr, sc.ws.bytes, sc.done.ptr, s)?;
                sc.k.wsum(s, sc.a.ptr, sc.plan.ptr, sc.inv.ptr, wts, ys, tn, k, e, c.hidden)
            })?;
        }
        Ok(())
    }

    /// Device bytes of the layer's weights.
    pub fn weights_bytes(&self) -> usize {
        self.w.router.bytes
            + self.w.bias.as_ref().map_or(0, |b| b.bytes)
            + self.w.escale.as_ref().map_or(0, |b| b.bytes)
            + self.w.up.bytes
            + self.w.down.bytes()
            + self.shared.as_ref().map_or(0, |s| s.up.bytes + s.down.bytes + s.gate.as_ref().map_or(0, |g| g.bytes))
            + self.f8.as_ref().map_or(0, |f| f.up.bytes + f.down.bytes)
            + self.hb.as_ref().map_or(0, HbgExperts::bytes)
            + self.hb_cache.as_ref().map_or(0, HbCache::bytes)
            + self.shared8.as_ref().map_or(0, |(u, d)| u.bytes + d.bytes)
    }

    /// Whether the activation of the experts' first product is a kernel's, into a buffer of its own (`MoeScratch::h`); else it is the product's (fused) or in place (relu squared).
    fn separate_h(&self) -> bool {
        self.cfg.act.gated() && (self.cfg.fused() == 0 || self.hb.is_some())
    }

    /// The scratch buffer that holds the experts' intermediate rows [pairs, inter] after `experts`, in the plan's order.
    pub fn rows<'a>(&self, sc: &'a MoeScratch) -> &'a Buf {
        if self.separate_h() {
            &sc.h
        } else {
            &sc.a
        }
    }

    fn check_tokens(&self, sc: &MoeScratch, t: usize) -> Result<()> {
        if t > sc.max_tokens {
            return err(format!("{t} tokens, where the scratch is made for {}", sc.max_tokens));
        }
        Ok(())
    }

    /// The router: the experts and weights of `t` tokens from x [t, router_hidden] (rows `xs` elements apart) into the scratch's `ids` and `wts`.
    ///
    /// # Safety
    /// Live device buffers of the sizes the kernels read and write; the stream is the one the layer's other calls are on.
    pub unsafe fn route(&self, sc: &MoeScratch, s: Stream, x: u64, xs: usize, t: usize) -> Result<()> {
        route_with(&self.cfg, &self.w.router, self.w.bias.as_ref(), self.w.escale.as_ref(), sc, s, x, xs, t)
    }

    /// The experts' sum of `t` tokens into y [t, hidden]: the pairs sorted by expert, the first product (with the activation where the library applies it), the activation, the second product with the weights.
    /// x [t, hidden] is the experts' input; `route` has made `ids` and `wts`.
    ///
    /// # Safety
    /// As `route`.
    pub unsafe fn experts(&self, lib: &Lib, sc: &MoeScratch, s: Stream, x: u64, t: usize, y: u64) -> Result<()> {
        self.experts_via(lib, sc, s, x, t, y, None, 1)
    }

    /// `experts` for a layer whose stacks are `ExpertStack::pooled` (the expert cache's): the plan is made of `way` (int64 [t k] on the device) in place of the router's ids, each pair's place in the stacks, or -1 for a pair
    /// computed in another of the `calls` calls that share the layer's pairs between them (a prompt that chooses more experts than the stacks hold); the weighted sum is of every pair the router chose, so the last of
    /// the calls writes the layer's sum. Each pair's rows are the same whatever the order of the plan, and the kernels are the ones `experts` runs (the experts, tokens and choices they are asked for are the layer's).
    /// A second matrix in a row layout (`RowStack::pooled`) is read by the sparse product at the places of `way`, and the parts of the earlier calls' pairs stay in its scratch (`MoeScratch::part`) for the last call's sum.
    ///
    /// # Safety
    /// As `route`; `way` is a live device buffer of `t k` int64 whose ids are below the places the stacks have.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn experts_via(&self, lib: &Lib, sc: &MoeScratch, s: Stream, x: u64, t: usize, y: u64, way: Option<u64>, calls: usize) -> Result<()> {
        if self.hb_cache.is_some() && way.is_none() && calls <= 1 {
            return self.hb_cached(lib, sc, s, x, t, y, HbSum::Linear);
        }
        if self.cache.is_some() && way.is_none() && calls <= 1 {
            self.check_tokens(sc, t)?;
            if let Some(f) = &self.f8 {
                return self.f8_cached(lib, sc, s, f, x, t, y);
            }
            return self.cached(sc, s, x, t, y, &mut |waves| self.experts_products(lib, sc, s, x, t, y, Some(sc.way.ptr), waves));
        }
        self.experts_products(lib, sc, s, x, t, y, way, calls)?;
        self.experts_sum(sc, s, t, y, 0, 0)
    }

    /// A call computed in `waves` (calls that share a layer's pairs) keeps the sums of the earlier ones in the scratch's workspace, the weighted second product's rows, where a first product (activation `act`) whose K
    /// is split over blocks would write its parts: the first products of such a layer take a workspace of their own (`MoeScratch::up_ws`), and a scratch that has none refuses the call that would need it. (The sparse
    /// second product of a row layout keeps its parts in its own buffer.)
    fn waves_keep(&self, lib: &Lib, sc: &MoeScratch, t: usize, waves: usize, act: i64) -> Result<()> {
        let c = &self.cfg;
        if waves > 1
            && sc.ws1.is_none()
            && matches!(self.w.down, Down::Block(_))
            && lib.moe_workspace(
                self.w.up.pack.is_twelve(),
                c.experts as i64,
                self.w.up.rows as i64,
                c.hidden as i64,
                t as i64,
                c.router.top_k as i64,
                act,
                false,
            )? > 0
        {
            return err(format!(
                "{t} tokens of {} choices: the first product needs a workspace, which the sums of a layer's earlier calls are kept in",
                c.router.top_k
            ));
        }
        Ok(())
    }

    /// `experts_via` up to the second product: the plan, the first product, the activation, and the second (a block layout's, which writes y, its weighted sum; a row layout's parts, which `experts_sum` adds up). A
    /// pair whose place in `way` is -1 is in no part of this call: the experts the host computes are, for the cache, pairs with no place.
    ///
    /// # Safety
    /// As `experts_via`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn experts_products(&self, lib: &Lib, sc: &MoeScratch, s: Stream, x: u64, t: usize, y: u64, way: Option<u64>, calls: usize) -> Result<()> {
        self.check_tokens(sc, t)?;
        if t == 0 {
            return Ok(());
        }
        let (c, e, k, tt) = (&self.cfg, self.cfg.experts as i64, self.cfg.router.top_k as i64, t as i64);
        let pairs = t * c.router.top_k;
        if let Some(hb) = &self.hb {
            // (the experts all on the GPU: no cache's places, no waves; the parts of every pair are `experts_sum`'s, as a row layout's; a layer in the expert cache is `hb_cached`, its products and sum one call)
            if way.is_some() || calls > 1 || self.hb_cache.is_some() {
                return err("hummingbird experts are all on the GPU: the expert cache's places and waves of pairs are not taken");
            }
            lib.moe_way(sc.ids.ptr, pairs as i64, e, sc.plan.ptr, s)?;
            self.hb_up(hb, sc, s, x, t, sc.ids.ptr)?;
            let rows = self.hb_act(sc, s, pairs)?;
            return self.hb_down(hb, sc, s, rows, t, sc.ids.ptr);
        }
        if let Some(f) = &self.f8 {
            // (in parts of up to 64 tokens: the plan, the first product of each pair's token, the activation, the second product of each pair's own row into `a`, free after the activation, the weighted sum)
            let (kk, ee) = (c.router.top_k, c.experts);
            for t0 in (0..t).step_by(crate::f8::MOE_TOKENS) {
                let (tn, ids) = ((t - t0).min(crate::f8::MOE_TOKENS), way.unwrap_or(sc.ids.ptr) + (t0 * kk * 8) as u64);
                lib.moe_way(ids, (tn * kk) as i64, e, sc.plan.ptr, s)?;
                f.up.moe(lib, ee, x + (t0 * c.hidden * 2) as u64, sc.plan.ptr, tn, kk, true, sc.a.ptr, sc.ws.ptr, sc.ws.bytes, sc.done.ptr, s)?;
                sc.k.act(s, sc.a.ptr, sc.h.ptr, tn * kk, c.inter, &c.act)?;
                f.down.moe(lib, ee, sc.h.ptr, sc.plan.ptr, tn, kk, false, sc.a.ptr, sc.ws.ptr, sc.ws.bytes, sc.done.ptr, s)?;
                sc.k.wsum(s, sc.a.ptr, sc.plan.ptr, sc.inv.ptr, sc.wts.ptr + (t0 * kk * 4) as u64, y + (t0 * c.hidden * 2) as u64, tn, kk, ee, c.hidden)?;
            }
            return Ok(());
        }
        self.waves_keep(lib, sc, t, calls, c.fused())?;
        lib.moe_way(way.unwrap_or(sc.ids.ptr), pairs as i64, e, sc.plan.ptr, s)?;
        moe_tp(lib, &self.w.up, e, x, tt, k, true, sc.plan.ptr, c.fused(), 0, 0, sc.a.ptr, pairs, sc, sc.up_ws(), s)?;
        let rows = if c.fused() != 0 {
            sc.a.ptr
        } else if self.separate_h() {
            sc.k.act(s, sc.a.ptr, sc.h.ptr, pairs, c.inter, &c.act)?;
            sc.h.ptr
        } else {
            sc.k.act(s, sc.a.ptr, sc.a.ptr, pairs, c.inter, &c.act)?;
            sc.a.ptr
        };
        match &self.w.down {
            // (the sum's pairs by their places where the store gives them in one call: a pair the host computes or leaves out, -1, is left out of it; over
            // several calls, the waves, by the router's choices: each call sums the rows of the earlier ones too)
            Down::Block(d) => moe_tp(
                lib,
                d,
                e,
                rows,
                tt,
                k,
                false,
                sc.plan.ptr,
                0,
                sc.wts.ptr,
                if calls > 1 { sc.ids.ptr } else { way.unwrap_or(sc.ids.ptr) },
                y,
                t,
                sc,
                &sc.ws,
                s,
            ),
            Down::Rows(_) if self.w.up.tp_rows > 0 => err("experts sliced across GPUs with a second matrix in a row layout"),
            Down::Rows(r) => sc.rk.down(s, c, r, rows, sc.plan.ptr, way.unwrap_or(sc.ids.ptr), sc.part.ptr, sc.part.bytes, t),
        }
    }

    /// The routed sum of a row layout's parts into y [t, hidden]: the sum is of the pairs whose id in `ids` (int64 [t, k] on the device; 0: the router's) is an expert, and `extra` (fp32 [t, hidden] on the device; 0:
    /// none) is added before the one rounding. A block layout's `experts_products` wrote y already: `extra`, where given, is added to it (a second rounding).
    ///
    /// # Safety
    /// As `experts_via`.
    pub unsafe fn experts_sum(&self, sc: &MoeScratch, s: Stream, t: usize, y: u64, ids: u64, extra: u64) -> Result<()> {
        self.check_tokens(sc, t)?;
        if let Some(hb) = &self.hb {
            return self.hb_sum(hb, sc, s, t, y, if ids == 0 { sc.ids.ptr } else { ids }, sc.wts.ptr, extra);
        }
        match &self.w.down {
            Down::Block(_) if extra != 0 => sc.k.add_f32(s, y, extra, t * self.cfg.hidden),
            Down::Block(_) => Ok(()),
            Down::Rows(_) => sc.rk.sum(s, &self.cfg, if ids == 0 { sc.ids.ptr } else { ids }, sc.wts.ptr, sc.part.ptr, sc.part.bytes, y, t, extra),
        }
    }

    /// A hummingbird layer in the expert cache (`hb_cache`), for `t` tokens of x [t, hidden] (bf16) whose choices `route` (or the family's router) made: in calls of at most `offload::CPU_TOKENS` tokens, each R x (the host reads it
    /// back, fp32), then the store's one wait for the layer (`ExpertStore::run_groups_at`): the router's choices read on the host, each group's misses copied into its places or kept for the host, the tables that send each pair to
    /// its place written (`HbgKern::tables`: a pair the host keeps has no group, and the products skip it), the plan, the two products with the family's activation between them (`kind`), the host's pairs computed meanwhile from the
    /// host tier (src/cpu_hbg.rs), and the sum: the GPU's pairs (the host's left out by their ids) and the host's sums (`extra`) together before R^T, or for Muse Spark the GPU's normed sum and the host's normed sums added after it.
    ///
    /// # Safety
    /// As `route`; `ids` and `wts` of the scratch are the choices of the `t` tokens.
    #[allow(clippy::too_many_arguments)]
    unsafe fn hb_cached(&self, lib: &Lib, sc: &MoeScratch, s: Stream, x: u64, t: usize, y: u64, kind: HbSum) -> Result<()> {
        use crate::offload::{At, Groups, GroupsCall, Stage, CPU_TOKENS};
        self.check_tokens(sc, t)?;
        let (Some(hb), Some(cache)) = (&self.hb, &self.hb_cache) else { return err("a layer with no hummingbird experts in an expert cache") };
        let hk = sc.hbk.as_ref().ok_or_else(|| crate::ffi::Error("the scratch was made without hummingbird layers' room".into()))?;
        let (c, k, e, w) = (&self.cfg, self.cfg.router.top_k, self.cfg.experts, self.cfg.hidden);
        if sc.hb_xr.bytes < moe_hbg::xr_bytes(t.min(CPU_TOKENS), w) || sc.hb_part.bytes < moe_hbg::part_bytes(t.min(CPU_TOKENS) * k, w) {
            return err(format!("{t} tokens of hummingbird experts of width {w}: the scratch has no room for their rotated inputs and parts"));
        }
        let groups = Groups { layers: cache.layers, group_of: &cache.group_of, local_of: &cache.local_of };
        let at = At { ordinal: cache.ordinal, stream: s };
        for t0 in (0..t).step_by(CPU_TOKENS) {
            let (tn, pairs) = ((t - t0).min(CPU_TOKENS), (t - t0).min(CPU_TOKENS) * k);
            let (ids, wts) = (sc.ids.ptr + (t0 * k * 8) as u64, sc.wts.ptr + (t0 * k * 4) as u64);
            let (xs, ys) = (x + (t0 * w * 2) as u64, y + (t0 * w * 2) as u64);
            hk.rot_in(s, hb, xs, w, sc.hb_xr.ptr, tn)?;
            let call = GroupsCall {
                ids,
                wts,
                z: sc.hb_xr.ptr,
                ways: [cache.ways[0].ptr, cache.ways[1].ptr],
                sum_ids: cache.sids.ptr,
                part: cache.part.ptr,
                tokens: tn,
                pairs,
            };
            let mut queued = false;
            cache.store.run_groups_at(&groups, at, &call, &mut |stage, _, hy| match stage {
                Stage::Products => {
                    // (both groups' pairs at once, on the first group's turn: the activation is between the two products of every pair)
                    if std::mem::replace(&mut queued, true) {
                        return Ok(());
                    }
                    hk.tables(s, hb, ids, call.ways[0], call.ways[1], pairs)?;
                    lib.moe_way(ids, pairs as i64, e as i64, sc.plan.ptr, s)?;
                    for g in 0..2 {
                        hk.up(s, g, hb, sc.hb_xr.ptr, sc.plan.ptr, ids, sc.a.ptr, pairs, k)?;
                    }
                    let rows = match kind {
                        HbSum::Linear => self.hb_act(sc, s, pairs)?,
                        HbSum::Weighed { lim, fp8, .. } => {
                            launch!(sc.k.act_w, [pairs as u32, (c.inter as u32 / 128).div_ceil(4), 1], [128, 1, 1], s; sc.a.ptr, sc.h.ptr, sc.plan.ptr, wts, e as i32, c.inter as i32, lim, fp8 as i32)?;
                            sc.h.ptr
                        }
                        HbSum::Normed { .. } => {
                            sc.k.act_rounded(s, sc.a.ptr, sc.h.ptr, pairs, c.inter)?;
                            sc.h.ptr
                        }
                    };
                    for g in 0..2 {
                        hk.down(s, g, hb, rows, sc.plan.ptr, ids, sc.hb_part.ptr, pairs)?;
                    }
                    Ok(())
                }
                Stage::Sum => {
                    let sum_ids = if hy.ids != 0 { hy.ids } else { ids };
                    match kind {
                        HbSum::Linear => self.hb_sum(hb, sc, s, tn, ys, sum_ids, wts, hy.extra),
                        HbSum::Weighed { ones, .. } => self.hb_sum(hb, sc, s, tn, ys, sum_ids, ones + (t0 * k * 4) as u64, hy.extra),
                        HbSum::Normed { post_w, eps, inv } => {
                            hk.unrotate(s, hb, sc.hb_part.ptr, sc.plan.ptr, sc.hb_rows.ptr, pairs)?;
                            launch!(sc.k.inv, [(pairs as u32).div_ceil(256), 1, 1], [256, 1, 1], s; sc.plan.ptr, inv, pairs as i32, e as i32)?;
                            launch!(sc.k.norm_sum, [tn as u32, 1, 1], [256, 1, 1], s; sc.hb_rows.ptr, inv, sum_ids, wts, post_w, ys, k as i32, w as i32, eps)?;
                            // (the host's pairs normed and weighed: added after the sum, a second rounding)
                            if hy.extra != 0 {
                                sc.k.add_f32(s, ys, hy.extra, tn * w)?;
                            }
                            Ok(())
                        }
                    }
                }
            })?;
        }
        Ok(())
    }

    /// A hummingbird layer's first product for `t` tokens of x [t, hidden] (bf16): R x once a token (the scratch's `hb_xr`), then each group's pairs of the plan (`moe_way` over `ids`, made by the caller) into `a` [pairs, up rows] in the plan's
    /// order: the rows the library's first product writes, for the family's activation.
    ///
    /// # Safety
    /// As `route`; the plan of `ids` is in the scratch.
    unsafe fn hb_up(&self, hb: &HbgExperts, sc: &MoeScratch, s: Stream, x: u64, t: usize, ids: u64) -> Result<()> {
        let (c, hk) = (&self.cfg, sc.hbk.as_ref().ok_or_else(|| crate::ffi::Error("the scratch was made without hummingbird layers' room".into()))?);
        let pairs = t * c.router.top_k;
        if sc.hb_xr.bytes < moe_hbg::xr_bytes(t, c.hidden)
            || sc.hb_part.bytes < moe_hbg::part_bytes(pairs, c.hidden)
            || sc.hb_rows.bytes < moe_hbg::rows_bytes(pairs, c.hidden)
        {
            return err(format!("{t} tokens of hummingbird experts of width {}: the scratch has no room for their rotated inputs and parts", c.hidden));
        }
        hk.rot_in(s, hb, x, c.hidden, sc.hb_xr.ptr, t)?;
        for g in 0..2 {
            hk.up(s, g, hb, sc.hb_xr.ptr, sc.plan.ptr, ids, sc.a.ptr, pairs, c.router.top_k)?;
        }
        Ok(())
    }

    /// The activation between a hummingbird layer's two products, the family's own kernel (`MoeKern::act`) on the first product's rows `a` [pairs, up rows]: a gated activation into `h`, relu squared in place. Returns the buffer that holds the
    /// intermediate rows [pairs, inter].
    ///
    /// # Safety
    /// As `hb_up`.
    unsafe fn hb_act(&self, sc: &MoeScratch, s: Stream, pairs: usize) -> Result<u64> {
        let c = &self.cfg;
        let out = self.rows(sc).ptr;
        sc.k.act(s, sc.a.ptr, out, pairs, c.inter, &c.act)?;
        Ok(out)
    }

    /// A hummingbird layer's second product for the plan's pairs: the parts [pairs, hidden] (fp32, the rotated basis, the plan's order) of `rows` [pairs, inter] (bf16, the activation's, in the plan's order), each group's pairs by its
    /// own kernel.
    ///
    /// # Safety
    /// As `hb_up`.
    unsafe fn hb_down(&self, hb: &HbgExperts, sc: &MoeScratch, s: Stream, rows: u64, t: usize, ids: u64) -> Result<()> {
        let (c, hk) = (&self.cfg, sc.hbk.as_ref().ok_or_else(|| crate::ffi::Error("the scratch was made without hummingbird layers' room".into()))?);
        for g in 0..2 {
            hk.down(s, g, hb, rows, sc.plan.ptr, ids, sc.hb_part.ptr, t * c.router.top_k)?;
        }
        Ok(())
    }

    /// A hummingbird layer's routed sum into y [t, hidden]: each token's pairs (the choices in `ids` that are experts) found through the plan's inverse, their parts weighed by `wts` (fp32 [t, k]) and added (and `extra`: fp32 [t, hidden],
    /// the rotated basis; 0: none), R^T once, one rounding to bf16.
    ///
    /// # Safety
    /// As `hb_up`.
    #[allow(clippy::too_many_arguments)]
    unsafe fn hb_sum(&self, hb: &HbgExperts, sc: &MoeScratch, s: Stream, t: usize, y: u64, ids: u64, wts: u64, extra: u64) -> Result<()> {
        let (k, hk) =
            (self.cfg.router.top_k, sc.hbk.as_ref().ok_or_else(|| crate::ffi::Error("the scratch was made without hummingbird layers' room".into()))?);
        hk.inverse(s, hb, sc.plan.ptr, sc.inv.ptr, t * k)?;
        hk.sum_rot(s, hb, sc.hb_part.ptr, sc.inv.ptr, wts, ids, y, extra, t, k)
    }

    /// Expert `e`'s first product alone, for `t` tokens of x [t, hidden]: y [t, up rows] (bf16), by the expert's own pack at its offset in the stack (`ExpertStack::expert`) and the library's dense products: what a
    /// slot of a cache of single experts would run, and what the stack's strides are checked by.
    ///
    /// # Safety
    /// As `route`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn expert_up(&self, lib: &Lib, sc: &MoeScratch, s: Stream, e: usize, x: u64, t: usize, y: u64) -> Result<()> {
        self.check_tokens(sc, t)?;
        if e >= self.cfg.experts {
            return err(format!("expert {e} of {}", self.cfg.experts));
        }
        lib.linear(&self.w.up.expert(e), x, t as i64, y, sc.ws.ptr, sc.ws.bytes, sc.done.ptr, s)
    }

    /// The shared expert's output for x [t, in] added into y [t, out] (the layer's output width): its MLP (the layer's activation), times sigmoid(x . gate) where it has a gate. Nothing where the layer has none.
    ///
    /// # Safety
    /// As `route`.
    pub unsafe fn add_shared(&self, lib: &Lib, sc: &MoeScratch, s: Stream, x: u64, t: usize, y: u64) -> Result<()> {
        self.shared_mlp(lib, sc, s, x, t)?;
        self.shared_add(sc, s, t, y)
    }

    /// Experts whose outputs are normed before they are weighed (the kind's `Latent::expert_norm`), as transformers computes them on bf16 tensors (Muse Spark's), for `t` tokens: the router reads
    /// h [t, router_hidden] by its fp32 weights `router` (times `mult`, the kind's `Router::logit_scale`; the layer's scores, `bias` to pick, renormalized, times the layer's scale), the experts read
    /// x [t, hidden] (a latent's, normed); each pair's activation silu(gate) up (each rounded), its second product unweighted, normed (input-scaled with `post_w`, eps), times its weight, added in the order
    /// of the experts' ids into y [t, hidden]. `inv`: int32 [t k] (the pairs' rows).
    ///
    /// # Safety
    /// As `route`; `router`, `bias`, `post_w` and `inv` live device buffers of those sizes.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn forward_normed(
        &self,
        lib: &Lib,
        sc: &MoeScratch,
        s: Stream,
        h: u64,
        x: u64,
        t: usize,
        y: u64,
        router: u64,
        mult: f32,
        post_w: u64,
        eps: f32,
        inv: u64,
    ) -> Result<()> {
        self.check_tokens(sc, t)?;
        if t == 0 {
            return Ok(());
        }
        let (c, e, k) = (&self.cfg, self.cfg.experts, self.cfg.router.top_k);
        let Down::Block(down) = &self.w.down else { return err("normed experts' second matrix in a row layout") };
        launch!(sc.k.router_f32, [t as u32, (e as u32).div_ceil(8), 1], [256, 1, 1], s; h, router, sc.logits.ptr, e as i32, c.router_hidden as i32, mult)?;
        sc.k.select(s, sc.logits.ptr, self.w.bias.as_ref().map_or(0, |b| b.ptr), 0, sc.ids.ptr, sc.wts.ptr, t, e, k, &c.router)?;
        if self.hb_cache.is_some() {
            return self.hb_cached(lib, sc, s, x, t, y, HbSum::Normed { post_w, eps, inv });
        }
        if let Some(cache) = &self.cache {
            // (in calls whose pairs are no more than the places: each one's experts in one wave, every row of the sum the call's own)
            let tn_max = (self.w.up.slots / k).max(1);
            for t0 in (0..t).step_by(tn_max) {
                let tn = (t - t0).min(tn_max);
                let (ids, wts, xs, ys) =
                    (sc.ids.ptr + (t0 * k * 8) as u64, sc.wts.ptr + (t0 * k * 4) as u64, x + (t0 * c.hidden * 2) as u64, y + (t0 * c.hidden * 2) as u64);
                let pairs = tn * k;
                let call = crate::offload::Call { ids, wts, z: xs, way: sc.way.ptr, part: sc.cpu.ptr, tokens: tn, pairs };
                let at = crate::offload::At { ordinal: cache.ordinal, stream: s };
                cache.store.run_at(cache.layer, at, &call, &mut |stage, _, waves, hy| {
                    if stage != crate::offload::Stage::Products {
                        return Ok(());
                    }
                    if waves > 1 || !hy.gpu || hy.ids != 0 {
                        return err("normed experts in the cache: every pair of a call on the GPU in one wave");
                    }
                    let (ei, ti, ki) = (e as i64, tn as i64, k as i64);
                    lib.moe_way(sc.way.ptr, pairs as i64, ei, sc.plan.ptr, s)?;
                    moe_tp(lib, &self.w.up, ei, xs, ti, ki, true, sc.plan.ptr, 0, 0, 0, sc.a.ptr, pairs, sc, &sc.ws, s)?;
                    sc.k.act_rounded(s, sc.a.ptr, sc.h.ptr, pairs, c.inter)?;
                    moe_tp(lib, down, ei, sc.h.ptr, ti, ki, false, sc.plan.ptr, 0, 0, 0, sc.a.ptr, pairs, sc, &sc.ws, s)?;
                    launch!(sc.k.inv, [(pairs as u32).div_ceil(256), 1, 1], [256, 1, 1], s; sc.plan.ptr, inv, pairs as i32, e as i32)?;
                    launch!(sc.k.norm_sum, [tn as u32, 1, 1], [256, 1, 1], s; sc.a.ptr, inv, ids, wts, post_w, ys, k as i32, c.hidden as i32, eps)
                })?;
            }
            return Ok(());
        }
        let (ei, tt, ki, pairs) = (e as i64, t as i64, k as i64, t * k);
        lib.moe_way(sc.ids.ptr, pairs as i64, ei, sc.plan.ptr, s)?;
        if let Some(hb) = &self.hb {
            // (hummingbird: the first product over the rotated input, the second unweighted in the rotated basis, each pair's row rotated back and rounded to bf16 where the sum norms it)
            self.hb_up(hb, sc, s, x, t, sc.ids.ptr)?;
            sc.k.act_rounded(s, sc.a.ptr, sc.h.ptr, pairs, c.inter)?;
            self.hb_down(hb, sc, s, sc.h.ptr, t, sc.ids.ptr)?;
            let hk = sc.hbk.as_ref().ok_or_else(|| crate::ffi::Error("the scratch was made without hummingbird layers' room".into()))?;
            hk.unrotate(s, hb, sc.hb_part.ptr, sc.plan.ptr, sc.hb_rows.ptr, pairs)?;
            launch!(sc.k.inv, [(pairs as u32).div_ceil(256), 1, 1], [256, 1, 1], s; sc.plan.ptr, inv, pairs as i32, e as i32)?;
            return launch!(sc.k.norm_sum, [t as u32, 1, 1], [256, 1, 1], s; sc.hb_rows.ptr, inv, sc.ids.ptr, sc.wts.ptr, post_w, y, k as i32, c.hidden as i32, eps);
        }
        moe_tp(lib, &self.w.up, ei, x, tt, ki, true, sc.plan.ptr, 0, 0, 0, sc.a.ptr, pairs, sc, &sc.ws, s)?;
        sc.k.act_rounded(s, sc.a.ptr, sc.h.ptr, pairs, c.inter)?;
        // (the second product unweighted, the pairs' rows in the plan's order: into `a`, free after the activation)
        moe_tp(lib, down, ei, sc.h.ptr, tt, ki, false, sc.plan.ptr, 0, 0, 0, sc.a.ptr, pairs, sc, &sc.ws, s)?;
        launch!(sc.k.inv, [(pairs as u32).div_ceil(256), 1, 1], [256, 1, 1], s; sc.plan.ptr, inv, pairs as i32, e as i32)?;
        launch!(sc.k.norm_sum, [t as u32, 1, 1], [256, 1, 1], s; sc.a.ptr, inv, sc.ids.ptr, sc.wts.ptr, post_w, y, k as i32, c.hidden as i32, eps)
    }

    /// Experts whose pairs' weights are applied to their activations before the second product, as a release computes them (DeepSeek-V4's), for `t` tokens: the router reads h [t, hidden] (the layer's
    /// scores and the bias; or, where `hash` is given (the kind's `Router::hash`: the table int64 [vocab, k] and the tokens int32 [t] on the device), the token's own experts); the experts and the shared expert
    /// read hq (h as the release's products quantize it). Each pair's activation is weighed and rounded to FP8 where `fp8` (kernels/dsv4.cu's v4_moe_act), the second product then summing a token's pairs
    /// with weights of one (`ones`, fp32 [t k]); y [t, hidden] = that sum + the shared expert's output.
    ///
    /// # Safety
    /// As `route`; `ones` and `hash`'s buffers live device buffers of those sizes.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn forward_weighed(
        &self,
        lib: &Lib,
        sc: &MoeScratch,
        s: Stream,
        h: u64,
        hq: u64,
        t: usize,
        y: u64,
        hash: Option<(u64, u64)>,
        ones: u64,
        fp8: bool,
    ) -> Result<()> {
        self.check_tokens(sc, t)?;
        if t == 0 {
            return Ok(());
        }
        let (c, e, k) = (&self.cfg, self.cfg.experts, self.cfg.router.top_k);
        let Act::SwiGlu { limit } = c.act else { return err("weighed experts are SwiGLU") };
        let Down::Block(down) = &self.w.down else { return err("weighed experts' second matrix in a row layout") };
        let lim = limit.unwrap_or(f32::INFINITY);
        crate::tmark!(s, "moe: start");
        // (a decode step's shared expert on the scratch's second stream, beside the routed experts: it reads hq alone, and its output is added at the end)
        let beside = self.shared.as_ref().zip(sc.beside.as_ref()).filter(|_| t <= BESIDE_TOKENS);
        if let Some((sx, b)) = beside {
            let s2 = b.side.stream;
            b.fork.record(s)?;
            b.fork.wait(s2)?;
            self.shared_lin_in(lib, (&b.ws, &b.done), s2, true, hq, t, sc.sg.ptr)?;
            launch!(sc.k.act_w, [t as u32, (sx.inter as u32 / 128).div_ceil(4), 1], [128, 1, 1], s2; sc.sg.ptr, sc.sh.ptr, 0u64, 0u64, 0i32, sx.inter as i32, lim, fp8 as i32)?;
            self.shared_lin_in(lib, (&b.ws, &b.done), s2, false, sc.sh.ptr, t, sc.so.ptr)?;
            b.join.record(s2)?;
        }
        sc.k.logits(s, h, c.hidden, self.w.router.ptr, sc.logits.ptr, t, e, c.hidden)?;
        match hash {
            Some((table, tok)) => {
                launch!(sc.k.hash, [(t as u32).div_ceil(128), 1, 1], [128, 1, 1], s; sc.logits.ptr, table, tok, sc.ids.ptr, sc.wts.ptr, t as i32, e as i32, k as i32, c.router.scale)?
            }
            None => sc.k.select(s, sc.logits.ptr, self.w.bias.as_ref().map_or(0, |b| b.ptr), 0, sc.ids.ptr, sc.wts.ptr, t, e, k, &c.router)?,
        }
        crate::tmark!(s, "moe: router");
        if self.hb_cache.is_some() {
            self.hb_cached(lib, sc, s, hq, t, y, HbSum::Weighed { ones, lim, fp8 })?;
        } else if self.cache.is_some() {
            // (the expert cache: the products over the places of each wave, the weights in the activation, the sum's pairs the call's)
            self.cached(sc, s, hq, t, y, &mut |waves| {
                let (ei, tt, ki, pairs) = (e as i64, t as i64, k as i64, t * k);
                self.waves_keep(lib, sc, t, waves, 0)?;
                lib.moe_way(sc.way.ptr, pairs as i64, ei, sc.plan.ptr, s)?;
                moe_tp(lib, &self.w.up, ei, hq, tt, ki, true, sc.plan.ptr, 0, 0, 0, sc.a.ptr, pairs, sc, sc.up_ws(), s)?;
                launch!(sc.k.act_w, [pairs as u32, (c.inter as u32 / 128).div_ceil(4), 1], [128, 1, 1], s; sc.a.ptr, sc.h.ptr, sc.plan.ptr, sc.wts.ptr, e as i32, c.inter as i32, lim, fp8 as i32)?;
                moe_tp(lib, down, ei, sc.h.ptr, tt, ki, false, sc.plan.ptr, 0, ones, if waves > 1 { sc.ids.ptr } else { sc.way.ptr }, y, t, sc, &sc.ws, s)
            })?;
        } else {
            let (ei, tt, ki, pairs) = (e as i64, t as i64, k as i64, t * k);
            lib.moe_way(sc.ids.ptr, pairs as i64, ei, sc.plan.ptr, s)?;
            crate::tmark!(s, "moe: plan");
            match &self.hb {
                Some(hb) => self.hb_up(hb, sc, s, hq, t, sc.ids.ptr)?,
                None => moe_tp(lib, &self.w.up, ei, hq, tt, ki, true, sc.plan.ptr, 0, 0, 0, sc.a.ptr, pairs, sc, &sc.ws, s)?,
            }
            crate::tmark!(s, "moe: up");
            launch!(sc.k.act_w, [pairs as u32, (c.inter as u32 / 128).div_ceil(4), 1], [128, 1, 1], s; sc.a.ptr, sc.h.ptr, sc.plan.ptr, sc.wts.ptr, e as i32, c.inter as i32, lim, fp8 as i32)?;
            crate::tmark!(s, "moe: act");
            match &self.hb {
                // (the weights are in the activation: the sum of a token's parts is with weights of one, R^T once)
                Some(hb) => {
                    self.hb_down(hb, sc, s, sc.h.ptr, t, sc.ids.ptr)?;
                    self.hb_sum(hb, sc, s, t, y, sc.ids.ptr, ones, 0)?;
                }
                None => moe_tp(lib, down, ei, sc.h.ptr, tt, ki, false, sc.plan.ptr, 0, ones, sc.ids.ptr, y, t, sc, &sc.ws, s)?,
            }
        }
        crate::tmark!(s, "moe: down");
        if let Some((_, b)) = beside {
            b.join.wait(s)?;
            crate::tmark!(s, "moe: shared joined");
            sc.k.add(s, y, sc.so.ptr, 0, t, c.hidden)?;
            crate::tmark!(s, "moe: add");
            return Ok(());
        }
        let Some(sx) = &self.shared else { return Ok(()) };
        self.shared_lin(lib, sc, s, true, hq, t, sc.sg.ptr)?;
        crate::tmark!(s, "moe: shared up");
        launch!(sc.k.act_w, [t as u32, (sx.inter as u32 / 128).div_ceil(4), 1], [128, 1, 1], s; sc.sg.ptr, sc.sh.ptr, 0u64, 0u64, 0i32, sx.inter as i32, lim, fp8 as i32)?;
        crate::tmark!(s, "moe: shared act");
        self.shared_lin(lib, sc, s, false, sc.sh.ptr, t, sc.so.ptr)?;
        crate::tmark!(s, "moe: shared down");
        sc.k.add(s, y, sc.so.ptr, 0, t, c.hidden)?;
        crate::tmark!(s, "moe: add");
        Ok(())
    }

    /// `shared_mlp` after its first product, given: gu [t, 2 inter] its gate's and up's values (a product the shared expert's first matrix was packed into
    /// with others: Kimi-K3's latent projection's), the activation and the second product into the scratch's `so`. A shared expert of a gated activation
    /// and no gate of its own.
    ///
    /// # Safety
    /// As `route`; gu a live device buffer of that size.
    pub unsafe fn shared_from(&self, lib: &Lib, sc: &MoeScratch, s: Stream, gu: u64, t: usize) -> Result<()> {
        self.check_tokens(sc, t)?;
        let Some(sx) = self.shared.as_ref().filter(|_| t > 0) else { return Ok(()) };
        if !self.cfg.act.gated() || sx.gate.is_some() || self.shared8.is_some() {
            return err("a shared expert's first product given apart: one of a gated activation, no gate of its own, packed");
        }
        sc.k.act(s, gu, sc.sh.ptr, t, sx.inter, &self.cfg.act)?;
        linear_tp(lib, &sx.down, sc.sh.ptr, t, sc.so.ptr, &sc.ws, &sc.done, s)
    }

    /// The shared expert's output for x [t, in] into the scratch (`so`, and its gate's logits where it has one): `add_shared` up to the add, which does not depend on the routed experts and can be queued before them.
    ///
    /// # Safety
    /// As `route`.
    pub unsafe fn shared_mlp(&self, lib: &Lib, sc: &MoeScratch, s: Stream, x: u64, t: usize) -> Result<()> {
        let (Some(_), Some(sx)) = (&self.shared8, &self.shared) else { return shared_mlp_with(&self.cfg, self.shared.as_ref(), lib, sc, s, x, t) };
        // (shared_mlp_with's steps, the products the FP8 format's)
        self.check_tokens(sc, t)?;
        if t == 0 {
            return Ok(());
        }
        let c = &self.cfg;
        self.shared_lin(lib, sc, s, true, x, t, sc.sg.ptr)?;
        let h = if c.act.gated() {
            sc.k.act(s, sc.sg.ptr, sc.sh.ptr, t, sx.inter, &c.act)?;
            sc.sh.ptr
        } else {
            sc.k.act(s, sc.sg.ptr, sc.sg.ptr, t, sx.inter, &c.act)?;
            sc.sg.ptr
        };
        self.shared_lin(lib, sc, s, false, h, t, sc.so.ptr)?;
        if let Some(g) = &sx.gate {
            let inp = sx.up.pack.cols as usize;
            sc.k.logits(s, x, inp, g.ptr, sc.gate.ptr, t, 1, inp)?;
        }
        Ok(())
    }

    /// y [t, out] += the shared expert's output (`shared_mlp`'s), times its gate where it has one.
    ///
    /// # Safety
    /// As `route`.
    pub unsafe fn shared_add(&self, sc: &MoeScratch, s: Stream, t: usize, y: u64) -> Result<()> {
        shared_add_with(self.shared.as_ref(), sc, s, t, y)
    }

    /// The whole layer for `t` tokens whose x [t, hidden] is the router's, the experts' and the shared expert's input alike (Qwen3.5 and 3.6, GLM-5.x): y [t, hidden] = the routed experts' sum + the shared expert.
    /// (Nemotron 3 puts a projection between the routed sum (in its latent width) and the shared expert's add; Gemma 4 a norm on each of its two branches, the experts' and the dense MLP, before their sum. They call
    /// `route` and `experts` themselves: Nemotron `add_shared` after its projection, Gemma 4 its dense MLP in its own path, where `add_shared`'s add of the two un-normed outputs would be wrong.)
    ///
    /// # Safety
    /// As `route`.
    pub unsafe fn forward(&self, lib: &Lib, sc: &MoeScratch, s: Stream, x: u64, t: usize, y: u64) -> Result<()> {
        if self.cfg.router_hidden != self.cfg.hidden {
            return err("a layer whose router and experts read different widths: its router, experts and shared expert are three calls");
        }
        self.route(sc, s, x, self.cfg.hidden, t)?;
        self.experts(lib, sc, s, x, t, y)?;
        self.add_shared(lib, sc, s, x, t, y)
    }
}

/// y [t, rows] = x [t, cols] by a pack, its slices gathered across the GPUs where it is this GPU's slice of the rows (src/dynamic/tp.rs).
///
/// # Safety
/// As `MoeLayer::route`.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn linear_tp(lib: &Lib, p: &PackBuf, x: u64, t: usize, y: u64, ws: &Buf, done: &Buf, s: Stream) -> Result<()> {
    if p.tp_rows == 0 {
        return lib.linear(&p.pack, x, t as i64, y, ws.ptr, ws.bytes, done.ptr, s);
    }
    let c = crate::dynamic::tp::comm().ok_or_else(|| crate::ffi::Error("a matrix sliced across GPUs, on a thread of no group of them".into()))?;
    let sl = p.pack.rows as usize;
    let tmp = c.temp(s, t * sl * 2)?;
    lib.linear(&p.pack, x, t as i64, tmp, ws.ptr, ws.bytes, done.ptr, s)?;
    c.gather(s, tmp, t, sl, y, p.tp_rows)
}

/// The library's grouped product (`Lib::moe`) by a stack that may hold this GPU's slice of each expert's rows: into this GPU's room, then every GPU's
/// slices gathered into y, `rows` rows of an expert's whole rows (the pairs' of a first product, or a weighted second's tokens'). `ws`: the workspace it takes
/// (the scratch's, or the first products' own: `MoeScratch::up_ws`).
///
/// # Safety
/// As `MoeLayer::route`.
#[allow(clippy::too_many_arguments)]
unsafe fn moe_tp(
    lib: &Lib,
    st: &ExpertStack,
    e: i64,
    x: u64,
    t: i64,
    k: i64,
    first: bool,
    plan: u64,
    act: i64,
    wts: u64,
    ids: u64,
    y: u64,
    rows: usize,
    sc: &MoeScratch,
    ws: &Buf,
    s: Stream,
) -> Result<()> {
    if st.placed > 0 {
        return lib.moe_placed(&st.pack, st.placed, e, x, t, k, first, plan, act, wts, ids, y, ws.ptr, ws.bytes, sc.done.ptr, s);
    }
    if st.tp_rows == 0 {
        return lib.moe(&st.pack, e, x, t, k, first, plan, act, wts, ids, y, ws.ptr, ws.bytes, sc.done.ptr, s);
    }
    let c = crate::dynamic::tp::comm().ok_or_else(|| crate::ffi::Error("experts sliced across GPUs, on a thread of no group of them".into()))?;
    let tmp = c.temp(s, rows * st.rows * 2)?;
    lib.moe(&st.pack, e, x, t, k, first, plan, act, wts, ids, tmp, ws.ptr, ws.bytes, sc.done.ptr, s)?;
    c.gather(s, tmp, rows, st.rows, y, st.tp_rows)
}

/// `MoeLayer::route` for a layer whose router is given apart from its experts (the Hummingbird layer's, src/moe_hb.rs): the router `router` [experts, router_hidden] bf16 and its selection bias and per-expert scale where there are
/// any, the experts and weights of `t` tokens from x [t, router_hidden] (rows `xs` elements apart) into the scratch's `ids` and `wts`.
///
/// # Safety
/// As `MoeLayer::route`.
#[allow(clippy::too_many_arguments)]
pub unsafe fn route_with(
    c: &MoeCfg,
    router: &Buf,
    bias: Option<&Buf>,
    escale: Option<&Buf>,
    sc: &MoeScratch,
    s: Stream,
    x: u64,
    xs: usize,
    t: usize,
) -> Result<()> {
    if t > sc.max_tokens {
        return err(format!("{t} tokens, where the scratch is made for {}", sc.max_tokens));
    }
    sc.k.logits(s, x, xs, router.ptr, sc.logits.ptr, t, c.experts, c.router_hidden)?;
    sc.k.select(s, sc.logits.ptr, bias.map_or(0, |b| b.ptr), escale.map_or(0, |b| b.ptr), sc.ids.ptr, sc.wts.ptr, t, c.experts, c.router.top_k, &c.router)
}

/// `route_with`'s choices from logits a caller has put in the scratch's `logits` (t rows): the experts and their weights into `ids` and `wts`.
///
/// # Safety
/// As `route_with`.
pub unsafe fn select_with(c: &MoeCfg, bias: Option<&Buf>, sc: &MoeScratch, s: Stream, t: usize) -> Result<()> {
    if t > sc.max_tokens {
        return err(format!("{t} tokens, where the scratch is made for {}", sc.max_tokens));
    }
    sc.k.select(s, sc.logits.ptr, bias.map_or(0, |b| b.ptr), 0, sc.ids.ptr, sc.wts.ptr, t, c.experts, c.router.top_k, &c.router)
}

/// `MoeLayer::add_shared` for a layer whose shared expert is given apart from its experts: the shared expert's output for x [t, in] added into y [t, out], nothing where it has none.
///
/// # Safety
/// As `MoeLayer::route`.
#[allow(clippy::too_many_arguments)]
pub unsafe fn add_shared_with(c: &MoeCfg, shared: Option<&Shared>, lib: &Lib, sc: &MoeScratch, s: Stream, x: u64, t: usize, y: u64) -> Result<()> {
    shared_mlp_with(c, shared, lib, sc, s, x, t)?;
    shared_add_with(shared, sc, s, t, y)
}

/// `MoeLayer::shared_mlp` for a shared expert given apart from its experts: its output for x [t, in] into the scratch (`so`, and its gate's logits where it has one).
///
/// # Safety
/// As `MoeLayer::route`.
#[allow(clippy::too_many_arguments)]
pub unsafe fn shared_mlp_with(c: &MoeCfg, shared: Option<&Shared>, lib: &Lib, sc: &MoeScratch, s: Stream, x: u64, t: usize) -> Result<()> {
    if t > sc.max_tokens {
        return err(format!("{t} tokens, where the scratch is made for {}", sc.max_tokens));
    }
    let Some(sx) = shared else { return Ok(()) };
    if t == 0 {
        return Ok(());
    }
    let inp = sx.up.pack.cols as usize;
    linear_tp(lib, &sx.up, x, t, sc.sg.ptr, &sc.ws, &sc.done, s)?;
    let h = if c.act.gated() {
        sc.k.act(s, sc.sg.ptr, sc.sh.ptr, t, sx.inter, &c.act)?;
        sc.sh.ptr
    } else {
        sc.k.act(s, sc.sg.ptr, sc.sg.ptr, t, sx.inter, &c.act)?;
        sc.sg.ptr
    };
    linear_tp(lib, &sx.down, h, t, sc.so.ptr, &sc.ws, &sc.done, s)?;
    if let Some(g) = &sx.gate {
        sc.k.logits(s, x, inp, g.ptr, sc.gate.ptr, t, 1, inp)?;
    }
    Ok(())
}

/// `MoeLayer::shared_add` for a shared expert given apart from its experts: y [t, out] += its output (`shared_mlp_with`'s), times its gate where it has one.
///
/// # Safety
/// As `MoeLayer::route`.
pub unsafe fn shared_add_with(shared: Option<&Shared>, sc: &MoeScratch, s: Stream, t: usize, y: u64) -> Result<()> {
    if t > sc.max_tokens {
        return err(format!("{t} tokens, where the scratch is made for {}", sc.max_tokens));
    }
    let Some(sx) = shared else { return Ok(()) };
    if t == 0 {
        return Ok(());
    }
    sc.k.add(s, y, sc.so.ptr, sx.gate.as_ref().map_or(0, |_| sc.gate.ptr), t, sx.down.whole_rows())
}

/// The buffers the layers' calls work in, for up to `max_tokens` tokens a call: one set for all the layers of a model (a layer's calls follow each other on the stream), sized by the largest of them, with the kernels
/// and the library's workspace and counters. The stages' results stay in it until the next call: a check or a profile reads them.
pub struct MoeScratch {
    k: MoeKern,
    pub max_tokens: usize,
    /// logits [tokens, experts] fp32
    pub logits: Buf,
    /// the chosen experts [tokens, k] int64 and their weights [tokens, k] fp32
    pub ids: Buf,
    pub wts: Buf,
    /// the plan of `Lib::moe_way`, int32 [2 + 2 experts + pairs]
    pub plan: Buf,
    /// the ids a plan is made of where they are not the router's (`MoeLayer::experts_via`), int64 [tokens, k]
    pub way: Buf,
    /// the first product's rows in the plan's order: [pairs, inter] where the activation is fused into it or is relu squared (applied in place), else [pairs, 2 inter]
    pub a: Buf,
    /// the activation's rows [pairs, inter] where it is a kernel's and not in place (`MoeLayer::rows` says which of the two holds the intermediate rows)
    pub h: Buf,
    /// each pair's row in the plan's order, int32 [pairs] (the FP8 experts' sum)
    pub inv: Buf,
    ws: Buf,
    done: Buf,
    /// The first products' workspace where a layer's experts are in the expert cache (`MoeLayer::cache`, a second matrix in a block layout): a call computed in waves keeps the sums of its earlier waves in `ws` (the
    /// weighted second product's rows, fp32), where a first product whose K is split over blocks (a call of few pairs: a decode step's) would write its parts.
    ws1: Option<Buf>,
    // the sparse second product (`moe_rows`): its kernels and the partial sums of its blocks
    rk: RowsKern,
    pub part: Buf,
    /// fp32 [tokens, hidden] for up to `CPU_TOKENS` tokens: the sums that the host made of the pairs of experts it computed (the expert cache's), added to the routed sum
    pub cpu: Buf,
    // the hummingbird experts' (src/moe_hbg.rs): the kernels, the tokens' rotated inputs [tokens, width] fp32, the second product's parts [pairs, width] fp32 and the pairs' rows rotated back [pairs, width] bf16 (Muse Spark's), all in the plan's order
    hbk: Option<HbgKern>,
    pub hb_xr: Buf,
    pub hb_part: Buf,
    pub hb_rows: Buf,
    // the shared expert: its first product, its activation, its second product, its gate's logits
    pub(crate) sg: Buf,
    sh: Buf,
    so: Buf,
    gate: Buf,
    /// the shared expert's products beside the routed experts' (`forward_weighed`'s calls of up to `BESIDE_TOKENS` tokens), where the layers have one
    beside: Option<Beside>,
}

/// Tokens a call has at most for `forward_weighed` to run its shared expert beside its routed experts (a decode step's, or a draft's few): past them the GPU is full anyway.
pub const BESIDE_TOKENS: usize = 64;

/// A second stream for the shared expert's products, the events that fork it from the engine's stream and join it back, and its own workspace and counters for the
/// library (the routed experts' products use the scratch's at the same time). Fork and join are graph edges in a capture.
struct Beside {
    side: crate::dev::Side,
    fork: crate::dev::Ev,
    join: crate::dev::Ev,
    ws: Buf,
    done: Buf,
}

/// What a layer asks of the scratch: its config and its shared expert's intermediate size, input and output widths.
#[derive(Clone, Copy, Debug)]
pub struct Shape {
    pub cfg: MoeCfg,
    pub twelve: bool,
    pub shared: Option<(usize, usize, usize)>,
    /// a single expert's first product runs (`MoeLayer::expert_up`)
    pub expert_rows: usize,
    /// the second matrix is in a row layout (`Down::Rows`): the sparse product, not the library's
    pub sparse: bool,
    /// the GPUs the experts' and shared expert's rows are sliced across (src/dynamic/tp.rs; 1: none): the products' rows are theirs over this
    pub slice: usize,
    /// the width of the rows of hummingbird experts (src/moe_hbg.rs: their products are kernels of their own, not the library's: the first product's rows are never fused with the activation, and the scratch holds the rotated inputs and
    /// the parts); 0: the experts are in another format
    pub hb: usize,
}

/// The scratch's buffers in bytes, and the library's calls whose workspace it must hold (asked for every count of tokens): the largest of what the layers ask, not their sum.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Sizes {
    pub logits: usize,
    pub pairs: usize,
    pub plan: usize,
    pub a: usize,
    pub h: usize,
    pub sg: usize,
    pub sh: usize,
    pub so: usize,
    /// the sparse second product's partial sums (`RowsKern::part_bytes`)
    pub part: usize,
    /// the sums that the host makes of the pairs of experts it computes, fp32 for up to `CPU_TOKENS` tokens (src/offload.rs)
    pub cpu: usize,
    /// the hummingbird experts': the rotated inputs (fp32), the second product's parts (fp32) and the pairs' rows rotated back (bf16)
    pub hb_xr: usize,
    pub hb_part: usize,
    pub hb_rows: usize,
    pub units: usize,
    /// (twelve, experts, rows, cols, k, act, weighted) of the grouped products; (twelve, rows, cols) of the dense ones
    pub grouped: Vec<[i64; 7]>,
    pub dense: Vec<[i64; 3]>,
}

impl Sizes {
    pub fn of(shapes: &[Shape], max_tokens: usize) -> Sizes {
        let mut z = Sizes { sg: 16, sh: 16, so: 16, h: 16, units: 1 << 18, ..Sizes::default() }; // (the counters of the dense products too: 1 << 18, as the engine's)
        for sh in shapes {
            let c = &sh.cfg;
            // (hummingbird's first product writes the rows the activation's kernel reads, whatever the activation)
            let (hb, fused) = (sh.hb > 0, if sh.hb > 0 { 0 } else { sh.cfg.fused() });
            let pairs = max_tokens * c.router.top_k;
            z.logits = z.logits.max(max_tokens * c.experts * 4);
            z.pairs = z.pairs.max(pairs);
            z.plan = z.plan.max((2 + 2 * c.experts + pairs) * 4);
            z.a = z.a.max(pairs * (if fused != 0 { c.inter } else { c.up_rows() }) * 2);
            if c.act.gated() && fused == 0 {
                z.h = z.h.max(pairs * c.inter * 2);
            }
            if sh.sparse {
                z.part = z.part.max(RowsKern::part_bytes(pairs, c.hidden));
            }
            if hb {
                z.hb_xr = z.hb_xr.max(crate::moe_hbg::xr_bytes(max_tokens, sh.hb));
                z.hb_part = z.hb_part.max(crate::moe_hbg::part_bytes(pairs, sh.hb));
                z.hb_rows = z.hb_rows.max(crate::moe_hbg::rows_bytes(pairs, sh.hb));
            }
            z.cpu = z.cpu.max(max_tokens.min(CPU_TOKENS) * c.hidden * 4);
            // (the second product is the library's unless it is sparse)
            let sl = sh.slice.max(1);
            for (rows, cols, act, weighted) in [(c.up_rows() / sl, c.hidden, fused, 0), (c.hidden / sl, c.inter, 0, 1)].into_iter().take(if hb {
                0
            } else if sh.sparse {
                1
            } else {
                2
            }) {
                let key = [sh.twelve as i64, c.experts as i64, rows as i64, cols as i64, c.router.top_k as i64, act, weighted];
                if !z.grouped.contains(&key) {
                    z.grouped.push(key);
                }
                z.units = z.units.max((if act != 0 { rows / 128 } else { rows / 64 }) * c.experts.min(pairs));
            }
            // (the library's single-expert product is not asked of experts the library does not hold)
            let mut dense = if hb { Vec::new() } else { vec![[sh.twelve as i64, sh.expert_rows as i64, c.hidden as i64]] };
            if let Some((inter, inp, out)) = sh.shared {
                let rows = if c.act.gated() { 2 * inter } else { inter };
                // (whole and sliced both: the shared expert's matrices are sliced where they are big enough, the experts' stacks where theirs are)
                for sl in [1, sl] {
                    dense.extend([[sh.twelve as i64, (rows / sl) as i64, inp as i64], [sh.twelve as i64, (out / sl) as i64, inter as i64]]);
                }
                (z.sg, z.sh, z.so) = (z.sg.max(max_tokens * rows * 2), z.sh.max(max_tokens * inter * 2), z.so.max(max_tokens * out * 2));
            }
            for d in dense {
                if !z.dense.contains(&d) {
                    z.dense.push(d);
                }
            }
        }
        z
    }
}

impl MoeScratch {
    /// The workspace a first product takes: its own where the scratch has one (`ws1`: a layer in the expert cache), else the scratch's.
    fn up_ws(&self) -> &Buf {
        self.ws1.as_ref().unwrap_or(&self.ws)
    }

    /// The scratch for `layers` and calls of up to `max_tokens` tokens: the library's workspace asked for every count of tokens up to it (it is not monotonic in it), for each distinct shape of the layers' products.
    pub fn new(dev: &Dev, lib: &Lib, layers: &[&MoeLayer], max_tokens: usize) -> Result<MoeScratch> {
        if layers.is_empty() || max_tokens == 0 {
            return err("a scratch for no layer or no tokens");
        }
        let shapes: Vec<Shape> = layers
            .iter()
            .map(|l| Shape {
                cfg: l.cfg,
                twelve: l.w.up.pack.is_twelve(),
                shared: l.shared.as_ref().map(|x| (x.inter, x.up.pack.cols as usize, x.down.whole_rows())),
                expert_rows: l.w.up.rows,
                sparse: matches!(l.w.down, Down::Rows(_)),
                slice: if l.w.up.tp_rows > 0 {
                    l.w.up.tp_rows / l.w.up.rows
                } else {
                    l.shared.as_ref().filter(|x| x.up.tp_rows > 0).map_or(1, |x| x.up.tp_rows / x.up.pack.rows as usize)
                },
                hb: l.hb.as_ref().map_or(0, |h| h.shape.width),
            })
            .collect();
        let (mut f8, mut beside) = (0, 0);
        for l in layers {
            if let Some(f) = &l.f8 {
                f8 = f8.max(f.up.moe_workspace(lib, f.experts, max_tokens, l.cfg.router.top_k)?).max(f.down.moe_workspace(
                    lib,
                    f.experts,
                    max_tokens,
                    l.cfg.router.top_k,
                )?);
            }
            if let Some((u, d)) = &l.shared8 {
                for m in 1..=max_tokens.min(64) {
                    f8 = f8.max(u.workspace(lib, m)?).max(d.workspace(lib, m)?);
                }
            }
            // (the release's layers, `forward_weighed`'s: their shared expert's products at up to BESIDE_TOKENS tokens)
            if let (Some(sx), Act::SwiGlu { limit: Some(_) }, Down::Block(_)) = (&l.shared, l.cfg.act, &l.w.down) {
                for m in 1..=max_tokens.min(BESIDE_TOKENS) {
                    beside = beside.max(match &l.shared8 {
                        Some((u, d)) => u.workspace(lib, m)?.max(d.workspace(lib, m)?),
                        None => lib.linear_workspace(sx.up.pack.is_twelve(), sx.up.pack.rows, sx.up.pack.cols, m as i64)?.max(lib.linear_workspace(
                            sx.down.pack.is_twelve(),
                            sx.down.pack.rows,
                            sx.down.pack.cols,
                            m as i64,
                        )?),
                    });
                }
            }
        }
        let mut sc = MoeScratch::sized(dev, lib, &shapes, max_tokens, f8)?;
        if layers.iter().any(|l| l.cache.is_some() && matches!(l.w.down, Down::Block(_))) {
            let mut need = 0;
            for t in 1..=max_tokens as i64 {
                for g in Sizes::of(&shapes, max_tokens).grouped.iter().filter(|g| g[6] == 0) {
                    // (with and without the activation: the release's layers ask for the plain product of a stack whose config fuses it)
                    need = need.max(lib.moe_workspace(g[0] != 0, g[1], g[2], g[3], t, g[4], g[5], false)?).max(lib.moe_workspace(
                        g[0] != 0,
                        g[1],
                        g[2],
                        g[3],
                        t,
                        g[4],
                        0,
                        false,
                    )?);
                }
            }
            sc.ws1 = Some(Buf::zeroed(need.max(16))?);
        }
        // (a development build's GLYD_MOE_BESIDE=0: the shared expert after the routed ones, on the engine's stream)
        if layers.iter().any(|l| l.shared.is_some() && matches!((l.cfg.act, &l.w.down), (Act::SwiGlu { limit: Some(_) }, Down::Block(_))))
            && crate::tune!("GLYD_MOE_BESIDE").as_deref() != Some("0")
        {
            sc.beside = Some(Beside {
                side: crate::dev::Side::new()?,
                fork: crate::dev::Ev::new()?,
                join: crate::dev::Ev::new()?,
                ws: Buf::zeroed(beside.max(16))?,
                done: Buf::zeroed(4 << 18)?,
            });
        }
        Ok(sc)
    }

    /// The scratch for layers of the given shapes (what `new` makes of its layers': a layer whose weights are held apart from a `MoeLayer`, the Hummingbird layer, asks for its own).
    pub fn for_shapes(dev: &Dev, lib: &Lib, shapes: &[Shape], max_tokens: usize) -> Result<MoeScratch> {
        MoeScratch::sized(dev, lib, shapes, max_tokens, 0)
    }

    /// `for_shapes` with at least `ws` bytes of workspace (the FP8 experts' products').
    fn sized(dev: &Dev, lib: &Lib, shapes: &[Shape], max_tokens: usize, ws: usize) -> Result<MoeScratch> {
        if shapes.is_empty() || max_tokens == 0 {
            return err("a scratch for no layer or no tokens");
        }
        let z = Sizes::of(shapes, max_tokens);
        let mut need = ws;
        for t in 1..=max_tokens as i64 {
            for g in &z.grouped {
                need = need.max(lib.moe_workspace(g[0] != 0, g[1], g[2], g[3], t, g[4], g[5], g[6] != 0)?);
            }
            // (in either layout: the shared expert's matrices may be in another than the experts' stacks, the 12-bit one on an HBM GPU)
            for d in &z.dense {
                need = need.max(lib.linear_workspace(false, d[1], d[2], t)?).max(lib.linear_workspace(true, d[1], d[2], t)?);
            }
        }
        Ok(MoeScratch {
            k: MoeKern::load(dev)?,
            max_tokens,
            logits: Buf::new(z.logits)?,
            ids: Buf::new(z.pairs * 8)?,
            wts: Buf::new(z.pairs * 4)?,
            plan: Buf::new(z.plan)?,
            way: Buf::new(z.pairs * 8)?,
            a: Buf::new(z.a)?,
            h: Buf::new(z.h)?,
            inv: Buf::new(z.pairs * 4)?,
            ws: Buf::zeroed(need.max(16))?,
            done: Buf::zeroed(z.units * 4)?,
            rk: RowsKern::load(dev)?,
            part: Buf::new(z.part.max(16))?,
            cpu: Buf::new(z.cpu.max(16))?,
            hbk: if shapes.iter().any(|sh| sh.hb > 0) { Some(HbgKern::load(dev)?) } else { None },
            hb_xr: Buf::new(z.hb_xr.max(16))?,
            hb_part: Buf::new(z.hb_part.max(16))?,
            hb_rows: Buf::new(z.hb_rows.max(16))?,
            sg: Buf::new(z.sg)?,
            sh: Buf::new(z.sh)?,
            so: Buf::new(z.so)?,
            gate: Buf::new(max_tokens * 4)?,
            beside: None,
            ws1: None,
        })
    }

    /// A scratch for the router and the expert cache's calls alone (a codebook block's, whose products have buffers of their own): the logits, the choices and their ids for calls of up to `max_tokens` tokens, the host's
    /// sums for `CPU_TOKENS` of them; nothing for the library's products.
    pub fn routing(dev: &Dev, experts: usize, top_k: usize, hidden: usize, max_tokens: usize) -> Result<MoeScratch> {
        if max_tokens == 0 {
            return err("a scratch for no tokens");
        }
        let pairs = max_tokens * top_k;
        Ok(MoeScratch {
            k: MoeKern::load(dev)?,
            max_tokens,
            logits: Buf::new(max_tokens * experts * 4)?,
            ids: Buf::new(pairs * 8)?,
            wts: Buf::new(pairs * 4)?,
            plan: Buf::new((2 + 2 * experts + pairs) * 4)?,
            way: Buf::new(pairs * 8)?,
            a: Buf::new(16)?,
            h: Buf::new(16)?,
            inv: Buf::new(pairs * 4)?,
            ws: Buf::zeroed(16)?,
            done: Buf::zeroed(16)?,
            rk: RowsKern::load(dev)?,
            part: Buf::new(16)?,
            cpu: Buf::new(max_tokens.min(CPU_TOKENS) * hidden * 4)?,
            hbk: None,
            hb_xr: Buf::new(16)?,
            hb_part: Buf::new(16)?,
            hb_rows: Buf::new(16)?,
            sg: Buf::new(16)?,
            sh: Buf::new(16)?,
            so: Buf::new(16)?,
            gate: Buf::new(max_tokens * 4)?,
            beside: None,
            ws1: None,
        })
    }

    /// Device bytes of the scratch.
    pub fn bytes(&self) -> usize {
        [
            &self.logits,
            &self.ids,
            &self.wts,
            &self.plan,
            &self.way,
            &self.a,
            &self.h,
            &self.ws,
            &self.done,
            &self.part,
            &self.cpu,
            &self.hb_xr,
            &self.hb_part,
            &self.hb_rows,
            &self.sg,
            &self.sh,
            &self.so,
            &self.gate,
        ]
        .iter()
        .map(|b| b.bytes)
        .sum::<usize>()
            + self.beside.as_ref().map_or(0, |b| b.ws.bytes + b.done.bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glyd_gpu::pack::{pack_int4, pack_int5, pack_int6};

    const SRC: &str = include_str!("../kernels/moe.cu");
    const LIBRARY: &str = include_str!("../../gpu/glyd_gpu.cu");

    /// The four models' layers: their configs pass `check`, and the activation is fused into the first product where the library applies it.
    #[test]
    fn the_four_models_layers() {
        // Qwen3.6-35B-A3B: 8 of 256, hidden 2048, intermediate 512, softmax
        let qwen = MoeCfg { experts: 256, router_hidden: 2048, hidden: 2048, inter: 512, router: Router::softmax(8), act: Act::SwiGlu { limit: None } };
        // Gemma-4-26B-A4B: 8 of 128, hidden 2816, intermediate 704, GELU
        let gemma = MoeCfg { experts: 128, router_hidden: 2816, hidden: 2816, inter: 704, router: Router::softmax(8), act: Act::GeGlu };
        // GLM-5.3-Flash: 8 of 288, hidden 4096, intermediate 2048, sigmoid x 2.5, SwiGLU clamped at 10
        let glm =
            MoeCfg { experts: 288, router_hidden: 4096, hidden: 4096, inter: 2048, router: Router::sigmoid(8, 2.5), act: Act::SwiGlu { limit: Some(10.0) } };
        // Nemotron 3 Super: 22 of 512, the router over 4096, the experts over a latent of 1024, intermediate 2688, sigmoid x 5, relu squared without a gate
        let nemo = MoeCfg { experts: 512, router_hidden: 4096, hidden: 1024, inter: 2688, router: Router::sigmoid(22, 5.0), act: Act::Relu2 };
        for c in [qwen, gemma, glm, nemo] {
            c.check().unwrap();
        }
        assert_eq!((qwen.fused(), gemma.fused(), glm.fused(), nemo.fused()), (1, 2, 0, 0));
        assert_eq!((qwen.up_rows(), gemma.up_rows(), glm.up_rows(), nemo.up_rows()), (1024, 1408, 4096, 2688));
        assert_eq!(qwen.router.flags(), FLAG_RENORM | FLAG_BF16);
        assert_eq!(glm.router.flags(), FLAG_SIGMOID | FLAG_RENORM);
        // an intermediate size that is not a multiple of 64 cannot be fused (the gate's half must be rows of 64): the activation is a kernel's
        assert_eq!(MoeCfg { inter: 480, ..qwen }.fused(), 0);
    }

    #[test]
    fn shapes_the_products_do_not_take_are_refused() {
        let ok = MoeCfg { experts: 128, router_hidden: 512, hidden: 512, inter: 256, router: Router::softmax(8), act: Act::SwiGlu { limit: None } };
        ok.check().unwrap();
        let bad = |c: MoeCfg, why: &str| {
            let e = c.check().unwrap_err().to_string();
            assert!(e.contains(why), "{e} does not say {why:?}");
        };
        bad(MoeCfg { experts: 0, ..ok }, "0 experts");
        bad(MoeCfg { experts: 2000, ..ok }, "2000 experts");
        bad(MoeCfg { router: Router::softmax(0), ..ok }, "0 experts a token");
        bad(MoeCfg { router: Router::softmax(33), ..ok }, "33 experts a token");
        bad(MoeCfg { experts: 4, router: Router::softmax(5), ..ok }, "5 experts a token of 4");
        bad(MoeCfg { router_hidden: 100, ..ok }, "divisible by 8");
        bad(MoeCfg { hidden: 500, ..ok }, "divisible by 64");
        bad(MoeCfg { inter: 100, ..ok }, "columns divisible by 32");
        // relu squared has no gate: its rows are the intermediate size, which must itself be a multiple of 64
        bad(MoeCfg { inter: 96, act: Act::Relu2, ..ok }, "divisible by 64");
        bad(MoeCfg { act: Act::SwiGlu { limit: Some(0.0) }, ..ok }, "clamp");
        MoeCfg { inter: 96, ..ok }.check().unwrap(); // (gated: 192 rows)
    }

    /// The select kernel by the number of experts (256, 512, 1,024 a lane's 8, 16, 32 registers a step).
    #[test]
    fn the_select_kernel_by_experts() {
        for (e, i) in [(1, 0), (128, 0), (256, 0), (257, 1), (288, 1), (512, 1), (513, 2), (1024, 2)] {
            assert_eq!(select_index(e).unwrap(), i, "{e} experts");
        }
        assert!(select_index(0).is_err() && select_index(1025).is_err());
    }

    fn weights_of(n: usize, seed: u64) -> Vec<u16> {
        let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        (0..n)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                let v = ((x >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.06;
                ((v.to_bits().wrapping_add(0x7fff + ((v.to_bits() >> 16) & 1))) >> 16) as u16
            })
            .collect()
    }

    /// E packs of [rows, cols] end to end are the pack of the E matrices stacked, byte for byte, in the three block layouts (the library's grouped products and a cache of single experts read the same bytes); and an
    /// expert's strides are the bytes of its own pack.
    #[test]
    fn experts_end_to_end_are_the_stacked_pack() {
        let (e, rows, cols) = (3usize, 128usize, 96usize);
        let w = weights_of(e * rows * cols, 7);
        type Pack = fn(&[u16], usize, usize) -> Option<Tiered>;
        for (tag, pack) in [(INT6_TAG, pack_int6 as Pack), (INT5_TAG, pack_int5 as Pack), (INT4_TAG, pack_int4 as Pack)] {
            let each: Vec<Tiered> = (0..e).map(|i| pack(&w[i * rows * cols..(i + 1) * rows * cols], rows, cols).unwrap()).collect();
            let stacked = pack(&w, e * rows, cols).unwrap();
            let j = Joined::of(&each.iter().collect::<Vec<_>>()).unwrap();
            assert_eq!((j.tag, j.experts, j.rows, j.cols), (tag, e, rows, cols));
            assert_eq!(j.data, stacked.data, "tag {tag}: the codes");
            assert_eq!(j.scales, stacked.blocks, "tag {tag}: the scales");
            let (db, sb) = expert_strides(tag, rows, cols).unwrap();
            assert_eq!((db as usize, sb as usize), (each[0].data.len(), each[0].blocks.len()));
            let s = Joined::stacked(&stacked, e).unwrap();
            assert_eq!((s.data, s.scales, s.rows), (j.data, j.scales, rows));
        }
    }

    #[test]
    fn packs_that_are_not_one_layout_and_shape_are_refused() {
        let (rows, cols) = (64usize, 64usize);
        let w = weights_of(2 * rows * cols, 3);
        let a = pack_int5(&w[..rows * cols], rows, cols).unwrap();
        let b = pack_int6(&w[rows * cols..], rows, cols).unwrap();
        assert!(Joined::of(&[&a, &b]).is_err(), "Int5 and Int6 packs together");
        let four = pack_int4(&w[..rows * cols], rows, cols).unwrap();
        assert!(Joined::of(&[&four, &a]).is_err() && Joined::of(&[&four, &four]).is_ok(), "Int4 packs alone");
        let c = pack_int5(&w[..32 * 64 * 2], 64, 64).unwrap();
        let d = pack_int5(&w[..64 * 32], 64, 32).unwrap();
        assert!(Joined::of(&[&c, &d]).is_err(), "two shapes");
        assert!(Joined::of(&[]).is_err());
        assert!(Joined::stacked(&a, 3).is_err() && Joined::stacked(&a, 2).is_err(), "a pack that is not 3 or 2 experts' of that shape");
        // the tiered layout holds its steps in blocks of any length: its experts are not at strides
        assert!(expert_strides(2, 64, 64).is_err());
        assert!(expert_strides(INT5_TAG, 100, 64).is_err() && expert_strides(INT5_TAG, 64, 48).is_err());
        assert!(
            expert_strides(INT4_TAG, 100, 64).is_err()
                && expert_strides(INT4_TAG, 64, 48).is_err()
                && expert_strides(glyd_gpu::pack::INT3_TAG, 64, 64).is_err(),
            "the three-bit layout is not held yet"
        );
    }

    /// The library's grouped products take the layouts this table holds: `glyd_gpu_mma_moe` (the products) and `glyd_gpu_mma_moe_unpack` (the experts back to bf16) each dispatch on every tag `block_step_bytes` knows
    /// (a tag the table took and the library did not would be an error on the GPU alone), and on no block layout it does not (the library's three-bit layout is not in the grouped products yet).
    #[test]
    fn the_librarys_grouped_products_take_the_layouts_of_the_table() {
        let body = |head: &str| -> &str {
            let at = LIBRARY.find(head).unwrap_or_else(|| panic!("no {head} in the library"));
            &LIBRARY[at..at + LIBRARY[at..].find("\n}\n").expect("the function's end")]
        };
        for head in ["GLYD_GPU_INTERNAL int glyd_gpu_mma_moe(", "GLYD_GPU_INTERNAL int glyd_gpu_mma_moe_unpack("] {
            let f = body(head);
            for (tag, bits) in [(INT6_TAG, 6), (INT5_TAG, 5), (INT4_TAG, 4), (glyd_gpu::pack::INT3_TAG, 3)] {
                let held = block_step_bytes(tag).is_some();
                assert_eq!(f.contains(&format!("int{bits}_words(tiers, K, data, blocks, bad)")), held, "{head} and the {bits}-bit layout");
                assert_eq!(f.contains(&format!("Int{bits}{{data, blocks}}")), held, "{head} and the {bits}-bit layout");
            }
            assert!(
                block_step_bytes(FP4_TAG).is_some() && f.contains("fp4_words(tiers, K, data, blocks, bad)") && f.contains("Fp4{data, blocks}"),
                "{head} and the FP4 layout"
            );
            assert!(
                block_step_bytes(FP4R_TAG).is_some() && f.contains("fp4r_words(tiers, K, data, blocks, bad)") && f.contains("fp4r_fmt(data, blocks, K)"),
                "{head} and the FP4 layout with packed scales"
            );
        }
    }

    /// Room for fewer experts than a layer has (the expert cache's places) is asked for what the stacks hold: places for 1 to all of the layer's experts, in the layout the words name, of a shape the layouts take.
    #[test]
    fn places_are_refused_what_they_cannot_hold() {
        let words = [INT5_TAG << 24, 0, 0];
        let why = |r: Result<ExpertStack>| r.err().expect("refused").to_string();
        assert!(why(ExpertStack::pooled(INT5_TAG, words, 8, 0, 128, 96)).contains("room for 0 experts standing for 8"));
        assert!(why(ExpertStack::pooled(INT5_TAG, words, 8, 9, 128, 96)).contains("room for 9 experts standing for 8"));
        assert!(why(ExpertStack::pooled(INT6_TAG, words, 8, 4, 128, 96)).contains("layout of tag"));
        assert!(why(ExpertStack::pooled(INT5_TAG, words, 8, 4, 100, 96)).contains("divisible by 64"));
        assert!(why(ExpertStack::pooled(2, words, 8, 4, 128, 96)).contains("block layouts"));
    }

    /// What the Rust side names in the kernels' source is there: each kernel, and the flags and modes it passes (a name a kernel lost, or a constant that moved, would show on the GPU alone).
    #[test]
    fn kernels_and_constants_are_in_the_source() {
        for name in ["moe_logits", "moe_select8", "moe_select16", "moe_select32", "moe_act", "moe_add"] {
            assert!(
                SRC.contains(&format!("extern \"C\" __global__ void {name}(")) || SRC.contains(&format!(") {name}(")),
                "no kernel {name} in kernels/moe.cu"
            );
        }
        let define = |n: &str| -> Option<i32> {
            SRC.lines().find_map(|l| l.strip_prefix(&format!("#define {n} "))).and_then(|v| v.split_whitespace().next()).and_then(|v| v.parse().ok())
        };
        for (n, v) in [
            ("MOE_SIGMOID", FLAG_SIGMOID),
            ("MOE_RENORM", FLAG_RENORM),
            ("MOE_BF16", FLAG_BF16),
            ("MOE_SILU", ACT_SILU),
            ("MOE_GELU", ACT_GELU),
            ("MOE_RELU2", ACT_RELU2),
            ("MOE_SITU", ACT_SITU),
        ] {
            assert_eq!(define(n), Some(v), "{n}");
        }
        // the select kernels' registers a lane hold the experts they are chosen for
        assert!(SRC.contains("moe_select_t<8>") && SRC.contains("moe_select_t<16>") && SRC.contains("moe_select_t<32>"));
    }

    /// The scratch is the largest layer's, not the layers' sum: forty layers of a model take what one takes; and the sizes of the four models' layers at a prompt chunk are what their shapes give.
    #[test]
    fn the_scratch_is_the_largest_layers_not_the_sum() {
        let glm = Shape {
            cfg: MoeCfg {
                experts: 288,
                router_hidden: 4096,
                hidden: 4096,
                inter: 2048,
                router: Router::sigmoid(8, 2.5),
                act: Act::SwiGlu { limit: Some(10.0) },
            },
            twelve: false,
            shared: Some((2048, 4096, 4096)),
            expert_rows: 4096,
            sparse: false,
            slice: 1,
            hb: 0,
        };
        let nemo = Shape {
            cfg: MoeCfg { experts: 512, router_hidden: 4096, hidden: 1024, inter: 2688, router: Router::sigmoid(22, 5.0), act: Act::Relu2 },
            twelve: false,
            shared: Some((5376, 4096, 4096)),
            expert_rows: 2688,
            sparse: false,
            slice: 1,
            hb: 0,
        };
        let one = Sizes::of(&[glm], 2048);
        assert_eq!(Sizes::of(&[glm; 42], 2048), one, "42 layers of one shape");
        // GLM-5.3-Flash at 2,048 tokens: 16,384 pairs; the clamp is a kernel's: the first product's rows are 4,096 wide and the activation's 2,048
        assert_eq!((one.pairs, one.a, one.h, one.plan), (16384, 16384 * 4096 * 2, 16384 * 2048 * 2, (2 + 2 * 288 + 16384) * 4));
        assert_eq!((one.sg, one.sh, one.so), (2048 * 4096 * 2, 2048 * 2048 * 2, 2048 * 4096 * 2));
        assert_eq!(one.grouped, vec![[0, 288, 4096, 4096, 8, 0, 0], [0, 288, 4096, 2048, 8, 0, 1]]);
        // Nemotron 3: relu squared in place (no buffer of its own for the activation), the shared expert reading and writing the model's width
        let n = Sizes::of(&[nemo], 2048);
        assert_eq!((n.a, n.h), (2048 * 22 * 2688 * 2, 16), "{n:?}");
        assert_eq!(n.so, 2048 * 4096 * 2);
        // the two together: the maxima of each, and both layers' products asked for
        let both = Sizes::of(&[glm, nemo], 2048);
        assert_eq!((both.a, both.h, both.pairs, both.logits), (one.a.max(n.a), one.h, one.pairs.max(n.pairs), one.logits.max(n.logits)));
        assert_eq!(both.grouped.len(), 4);
        // the counters: a unit a row block of the first product (128 rows with a fused activation), by the experts a pair can hit
        assert_eq!(both.units, 1 << 18); // (the dense products' floor: the grouped ones need 42 * 512 and 64 * 288)
    }

    /// A hummingbird layer's scratch: the first product's rows are never fused with the activation (a SwiGLU layer whose library product would fuse them has rows of 2 inter and a buffer for the activation), the rotated inputs, the parts and the
    /// pairs' rows have their room, and the library's grouped and single-expert products are not asked for.
    #[test]
    fn the_scratch_of_hummingbird_experts() {
        // Qwen3.5-like: 256 experts of 512 neurons over 2,048 values, SwiGLU with no clamp (fused in the library's product)
        let cfg = MoeCfg { experts: 256, router_hidden: 2048, hidden: 2048, inter: 512, router: Router::softmax(8), act: Act::SwiGlu { limit: None } };
        let plain = Shape { cfg, twelve: false, shared: None, expert_rows: 1024, sparse: false, slice: 1, hb: 0 };
        let hb = Shape { hb: 2048, ..plain };
        let (a, b) = (Sizes::of(&[plain], 2048), Sizes::of(&[hb], 2048));
        let pairs = 2048 * 8;
        assert_eq!((a.a, a.h, a.hb_xr, a.hb_part, a.hb_rows), (pairs * 512 * 2, 16, 0, 0, 0));
        assert_eq!((b.a, b.h), (pairs * 1024 * 2, pairs * 512 * 2));
        assert_eq!((b.hb_xr, b.hb_part, b.hb_rows), (2048 * 2048 * 4, pairs * 2048 * 4, pairs * 2048 * 2));
        assert!(a.grouped.len() == 2 && a.dense.len() == 1 && b.grouped.is_empty() && b.dense.is_empty());
        // relu squared stays in place; the rows of a layer of two widths (the router's and the experts' latent) are the experts' width
        let nemo = Shape {
            cfg: MoeCfg { experts: 512, router_hidden: 4096, hidden: 1024, inter: 2688, router: Router::sigmoid(22, 5.0), act: Act::Relu2 },
            twelve: false,
            shared: Some((5376, 4096, 4096)),
            expert_rows: 2688,
            sparse: false,
            slice: 1,
            hb: 1024,
        };
        let n = Sizes::of(&[nemo], 64);
        assert_eq!((n.a, n.h, n.hb_xr, n.hb_part), (64 * 22 * 2688 * 2, 16, 64 * 1024 * 4, 64 * 22 * 1024 * 4));
        // (the shared expert's products are still asked for)
        assert_eq!(n.dense.len(), 2);
        // the largest of the layers' (the wider one's), as the rest of the scratch
        let both = Sizes::of(&[hb, nemo], 64);
        assert_eq!((both.hb_xr, both.hb_part), (64 * 2048 * 4, 64 * 22 * 1024 * 4));
    }

    /// `new_hb` holds the layer's config to the experts: another number of experts, width, intermediate size or a gate where the activation has none is refused (what the layer is built from is checked before any buffer is taken).
    #[test]
    fn new_hb_refuses_experts_that_are_not_the_layers() {
        use crate::moe_hbg::{testing::*, HbgHost};
        let mut rng = Rng(5);
        let rows = random_layer(&mut rng, 256, 256, 64, true, &[2, 3, 2, 3], 1);
        let host = HbgHost::of(&rows).unwrap();
        let cfg = MoeCfg { experts: 4, router_hidden: 256, hidden: 256, inter: 64, router: Router::softmax(2), act: Act::SwiGlu { limit: None } };
        let none = Buf::none;
        for (c, why) in [
            (MoeCfg { experts: 8, ..cfg }, "4 of width 256"),
            (MoeCfg { hidden: 512, router_hidden: 512, ..cfg }, "width 256"),
            (MoeCfg { inter: 128, ..cfg }, "64 neurons"),
            (MoeCfg { act: Act::Relu2, ..cfg }, "not gated"),
        ] {
            let e = MoeLayer::new_hb(c, none(), None, None, crate::moe_hbg::HbgExperts::without_device(&host), None).err().expect("refused").to_string();
            assert!(e.contains(why), "{e} does not say {why:?}");
        }
        // (the experts' shape is the config's: the check the layer makes passes)
        crate::moe_hbg::HbgExperts::without_device(&host).shape.check(&cfg).unwrap();
        cfg.check().unwrap();
    }

    #[test]
    fn the_activation_kernels_modes() {
        assert_eq!(Act::SwiGlu { limit: None }.kernel(), (ACT_SILU, f32::INFINITY, 0.0));
        assert_eq!(Act::SwiGlu { limit: Some(10.0) }.kernel(), (ACT_SILU, 10.0, 0.0));
        assert_eq!(Act::Situ { beta: 4.0, linear: 25.0 }.kernel(), (ACT_SITU, 4.0, 25.0));
        assert_eq!(Act::GeGlu.kernel().0, ACT_GELU);
        assert_eq!(Act::Relu2.kernel().0, ACT_RELU2);
        assert!(Act::GeGlu.gated() && !Act::Relu2.gated());
    }
}
