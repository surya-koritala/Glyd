//! The dynamic engine (GLY-326, research/dynamic-engine/DESIGN.md): a model is a description (`ModelDesc`: its matrices, its small tensors, its
//! rotary tables, its key/value rings and its layers as lists of operations), and one executor (`Dyn`) runs any description on the shared kernels.
//! A family adds a description (`gemma4.rs`), not an engine; an operation is added once, when a family needs one no other has.

pub mod attn;
pub mod deepseek_v4;
pub mod gemma4;
pub mod glm;
pub mod kimi;
pub mod kv12;
mod kvpages;
#[cfg(feature = "dev")]
pub use kvpages::check as kvpages_check;
pub mod lower;
pub mod muse;
pub mod nemotron;
pub mod pages;
pub mod qwen35;
pub mod range;
mod rows;
pub(crate) mod snaps;
pub mod split;
pub mod tp;
pub mod tp_cache;

use crate::dev::{copy_d2d_async, copy_d2h_async, copy_h2d_async, Buf, Dev, Ev, Graph, Pinned};
use crate::ffi::{err, Error, Lib, Result};
use crate::gemma4::{pf_kib, rotary, G4Pf, KernG4, EMBED_GROUPS};
use crate::kern::{norm_threads, Kern};
use crate::kern35::{Kern35, Mat};
use crate::kern_nh::{KernNh, MoeBlock, SsdWs};
use crate::launch;
use crate::model::{embed_plan, Sample, TopGpu};
use crate::weights::{choose_layout, ordered, pack_item_map, upload_bytes, upload_packed, Item, Layout, PackBuf};
pub use attn::AttnOp;
use glyd_gpu::safetensors::Checkpoint;
use std::path::Path;

/// An activation a pass reads and writes, [rows, width] bf16: the residual, the normed input of the next product, the q/k/v projection, the prepared
/// queries, the attention's output, o's, gate and up, the gated product, down's; a draft model's input and the big model's hidden state it drafts from;
/// a linear-attention layer's projection (q, k, v, z), its b and a, its convolved channels and its core output; the last row's final norm.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Slot {
    X,
    H,
    Proj,
    Qo,
    Att,
    O,
    Gu,
    M,
    D,
    In,
    Hid,
    Mix,
    Ba,
    Xc,
    Core,
    Hlast,
    /// DeepSeek-V4's: the four residual streams [4 hidden]; H as its products quantize it; the query's low rank normed; the key-value row; a compressor's two products (fp32: twice the width in bf16 words);
    /// the indexer's queries and its weights' product; the grouped low rank's products, a group's rows together
    S,
    Hq,
    Qr,
    Kv,
    Cp,
    Iq,
    Iw,
    Mid,
    /// Kimi-K3's completed blocks of layers (the depth mix's sources but the partial sum, X): a row's blocks side by side
    Blk,
}
const SLOTS: usize = 25;

/// Where an operation reads part of a fused product's rows: its slot, or for one token (`at` given) Mix's row at that column, the product's own output read
/// in place (a prompt's rows copied into the slot first: `Op::KimiParts`).
#[derive(Clone, Copy, Debug)]
pub struct View {
    pub slot: Slot,
    pub at: Option<usize>,
}

impl View {
    pub const fn slot(slot: Slot) -> View {
        View { slot, at: None }
    }

    pub const fn at(slot: Slot, at: usize) -> View {
        View { slot, at: Some(at) }
    }
}

/// One step of a layer. Matrices, small tensors, rotary tables and rings are indices into the description's lists.
#[derive(Clone, Debug)]
pub enum Op {
    /// x = bf16(x scale) (the embedding's scale), h = bf16(rmsnorm(x) norm).
    EmbedNorm {
        scale: f32,
        norm: usize,
    },
    /// y = x W^T for matrix `w` (any layout: the skinny products for a few rows, the library's otherwise); `name` labels the stage in a profile.
    Linear {
        w: usize,
        x: Slot,
        y: Slot,
        name: &'static str,
    },
    /// y = x W^T for the part's FP8 matrix `w` (the library's lossless FP8 format, an FP8 checkpoint's matrix as released).
    F8Lin {
        w: usize,
        x: Slot,
        y: Slot,
        name: &'static str,
    },
    /// The attention layer kind, every family's (src/dynamic/attn.rs): the preparation of the projection's heads (norms, rotary, the keys and values into
    /// their ring), then the causal attention of the queries over the ring into Att, gated where the kind is.
    Attn(AttnOp),
    /// The sandwich residual: x = bf16(x + bf16(rmsnorm(y) post)) (then bf16(x scalar) where `scalar`), and h = bf16(rmsnorm(x) next).
    AddNorm {
        y: Slot,
        post: usize,
        scalar: Option<usize>,
        next: usize,
    },
    /// m = bf16(bf16(gelu_tanh(gate)) up), gate and up the halves of gu.
    GeluMul {
        inter: usize,
    },
    /// y = bf16(rmsnorm(x) w).
    Norm {
        x: Slot,
        w: usize,
        y: Slot,
    },
    /// A draft model's input: In = [the big model's raw embedding row (its X) times `scale`, Hid].
    DraftIn {
        scale: f32,
    },
    /// The pre-norm models' norms: y = bf16(rmsnorm(x) w), times (1 + w) where `plus1` (Qwen3.5's); and the pre-norm residual, x += y, h = the norm of x.
    RmsNorm {
        x: Slot,
        w: usize,
        y: Slot,
        plus1: bool,
    },
    NormAdd {
        y: Slot,
        w: usize,
        plus1: bool,
    },
    /// A linear-attention (Gated DeltaNet) layer: the causal convolution of Mix's first `cd` channels (rows `row` wide) with weights `w` and its state into Xc;
    /// the delta rule over Xc and Ba (`nk` key heads, `nv` value heads; A_log, dt_bias) with its state into Core; the gated norm of Core by Mix's z (at `z_off`).
    GdnConv {
        w: usize,
        state: usize,
        cd: usize,
        row: usize,
    },
    GdnDelta {
        a_log: usize,
        dt_bias: usize,
        state: usize,
        cd: usize,
        nk: usize,
        nv: usize,
    },
    GatedNorm {
        w: usize,
        row: usize,
        nv: usize,
        z_off: usize,
    },
    /// m = silu(gate) up of matrix `w`'s product (gate and up stacked) of h.
    GateUpSilu {
        w: usize,
        inter: usize,
    },
    /// A Mamba-2 layer (Nemotron-H): the causal convolution (weights `w`, `bias`) of Mix's xBC (at `x_off`, rows `row` wide) with its state into Xc; the scan over Xc with
    /// Mix's dt (at `dt_off`) into Core (chunked for a prompt); the gated norm of Core by Mix's z in `ng` groups.
    MambaConv {
        w: usize,
        bias: usize,
        state: usize,
        cd: usize,
        row: usize,
        x_off: usize,
    },
    MambaScan {
        dt_bias: usize,
        a_log: usize,
        d: usize,
        state: usize,
        cd: usize,
        row: usize,
        dt_off: usize,
        nh: usize,
        ng: usize,
    },
    MambaNorm {
        w: usize,
        row: usize,
        di: usize,
        ng: usize,
    },
    /// A mixture-of-experts layer (the part's block `block`) of H into O.
    Moe {
        block: usize,
    },
    /// DeepSeek-V4's (src/dynamic/deepseek_v4.rs, kernels/dsv4.cu): the streams S from the embedding row X, four times.
    V4Streams,
    /// hc_pre: the mixing weights of S by the fp32 projection `f` (small, [24, 4 hidden]), `base` and `scale`, made doubly stochastic, and the collapse normed by `norm` into H; the weights kept for V4Post.
    V4Pre {
        f: usize,
        base: usize,
        scale: usize,
        norm: usize,
    },
    /// hc_post: S from the block's output O and the weights V4Pre kept.
    V4Post,
    /// hc_head and the final norm into H (the LM head reads its last rows).
    V4Head {
        f: usize,
        base: usize,
        scale: usize,
        norm: usize,
    },
    /// y = x as the release's products quantize their input: E4M3 in blocks of `block`.
    Fp8 {
        x: Slot,
        y: Slot,
        block: usize,
    },
    /// The queries Qo (`heads` of 512): each normed in bf16 arithmetic, the rotary of table `rope`.
    V4Q {
        rope: usize,
        heads: usize,
    },
    /// The key-value row Kv: normed by `norm`, the rotary of `rope`, its first 448 channels to E4M3; into ring `ring`.
    V4Kv {
        norm: usize,
        rope: usize,
        ring: usize,
    },
    /// A compressor of H: its two products by `w` (small: [2 W, hidden] bf16, kv's rows then the gate's) in fp32 into Cp, the pooling of every `ratio` rows with `ape` (small, fp32) and its state (states
    /// `state` and `state` + 1: the kv rows and the scores), each completed window's entry normed by `norm`, the rotary of `rope`, quantized (`indexer`: rotated, E2M1; else E4M3) into state `cache` [entries, d].
    V4Comp {
        w: usize,
        ape: usize,
        norm: usize,
        d: usize,
        ratio: usize,
        rope: usize,
        cache: usize,
        state: usize,
        indexer: bool,
    },
    /// The indexer's queries Iq (`heads` of 128): the rotary of `rope`, the rotation, E2M1.
    V4IdxQ {
        rope: usize,
        heads: usize,
    },
    /// The indexer's picks: Iq and Iw (its weights' product, times `wscale`) scored against its cache (state `cache`), each row's `topk` best entries.
    V4IdxPick {
        cache: usize,
        ratio: usize,
        heads: usize,
        topk: usize,
        wscale: f32,
    },
    /// The attention of Qo over the window (ring `ring`) and the cache (state `cache`: ratio 4, the indexer's picks; another, every complete entry) with the sink (small, fp32), the rotary of `rope` taken off
    /// its output; into Att in the grouped low rank's order (`hpg` heads a group).
    V4Attn {
        ring: usize,
        cache: Option<usize>,
        ratio: usize,
        sink: usize,
        rope: usize,
        heads: usize,
        hpg: usize,
    },
    /// The grouped low rank: group g's rows of Att (hpg 512 wide) by matrix `w` + g (an FP8 matrix where `in_f8`) into Mid (`ol` wide), then back in rows into `y` (E4M3 where `fp8`).
    V4OLora {
        w: usize,
        in_f8: bool,
        groups: usize,
        hpg: usize,
        ol: usize,
        y: Slot,
        fp8: bool,
    },
    /// The same with the groups' matrices one FP8 stack (`f8[w]`, [groups ol, hpg 512]): every group's product in one launch, the groups as the experts' product's experts.
    V4OLoraStack {
        w: usize,
        groups: usize,
        ol: usize,
        y: Slot,
        fp8: bool,
    },
    /// The mixture of experts as the release computes it (block `block`; the first layers' table of experts by token, small int64 [vocab, k], `hash`): the router reads H, the experts Hq; into O.
    V4Moe {
        block: usize,
        hash: Option<usize>,
    },
    /// GLM-5.3-Flash's (src/dynamic/glm.rs; kernels/glm.cu, transformers' rounding): the streams S from X; a block's mixing weights (the projection `f`, bf16 [24, 4 hidden], `base`, `scale`) and its normed
    /// collapse into H; the streams updated from O; the mean of the streams normed by `norm` into H.
    GlmStreams,
    GlmPre {
        f: usize,
        base: usize,
        scale: usize,
        norm: usize,
    },
    GlmPost,
    GlmHead {
        norm: usize,
    },
    /// Kimi delta attention: the causal convolution of Mix's first `c` channels (q, k and v; Mix's rows `row` apart) with weights `w` and its state into Xc; the delta rule over Xc, the forget gate's product
    /// Qo and the input gate's `ba` (`heads` heads, A_log, dt_bias, the forget gate's lower `bound`) with its state into Core; the gated norm of Core by the output gate's product `iq` into Att.
    KdaConv {
        w: usize,
        state: usize,
        c: usize,
        row: usize,
        w32: bool,
    },
    KdaDelta {
        a_log: usize,
        dt_bias: usize,
        state: usize,
        heads: usize,
        bound: f32,
        ba: View,
    },
    KdaOut {
        w: usize,
        heads: usize,
        w32: bool,
        iq: View,
    },
    /// y = w x, x a part of a fused product's rows (`View`)
    LinearAt {
        w: usize,
        x: View,
        y: Slot,
        name: &'static str,
    },
    /// A prompt's parts of the fused product in Mix (rows `row` apart): each (column, values, slot) copied into its slot, rows as wide as the part (one
    /// token's are read in place: `View`)
    KimiParts {
        row: usize,
        parts: [(usize, usize, Slot); 3],
    },
    /// GLM's latent attention over the keys its indexer picks (the part's block `block`, src/mla_glm.rs) of H into O.
    Mla {
        block: usize,
    },
    /// Kimi-K3's half-layer start, one kernel: X = bf16(X + O) where `add`; the depth mix of the first `nb` completed blocks (Blk) and X, by its norm
    /// weight and projection (`small`'s); X into block `push` of Blk, then X = 0, where given (the next block's partial sum starts empty); KimiRMSNorm (two
    /// roundings) of the mix by `w` into H
    KimiMixNorm {
        add: bool,
        nb: usize,
        w_n: usize,
        w_p: usize,
        push: Option<usize>,
        w: usize,
        eps: f32,
    },
    /// X = bf16(X + x)
    KimiAdd {
        x: Slot,
    },
    /// Kimi-K3's guess of a cached layer's experts in a decode step: the depth mix and norm of its experts' input (the mix's `nb`, `w_n`, `w_p`, the norm's `w`)
    /// over X before the layer's attention adds to it, into Att (written before it is read in the layer), and the block's router on that
    /// (`MoeBlock::early`), so that the copies of the experts it guesses run beside the attention.
    KimiEarly {
        block: usize,
        nb: usize,
        w_n: usize,
        w_p: usize,
        w: usize,
        eps: f32,
    },
    /// The MLP's situ (Gu [gate, up] into M)
    Situ {
        inter: usize,
        beta: f32,
        linear: f32,
    },
    /// Nothing (a model whose first operation is none)
    Nop,
    /// M = silu(min(gate, limit)) clamp(up, -limit, limit) of Gu (the gate's `inter` columns, then the up's), rounded once.
    SwiGluClamp {
        inter: usize,
        limit: f32,
    },
    /// Muse Spark's (src/dynamic/muse.rs; kernels/muse.cu): an RMS norm with eps (y = bf16(rmsnorm(x) w), the weight a centered norm's bf16(1 + w), or ones for none); the input-scaled norm
    /// (y = RMS(bf16(x w))); the residual gate (X = alpha X + beta O, small `alpha`, `beta`).
    MuseRms {
        x: Slot,
        w: usize,
        y: Slot,
        eps: f32,
    },
    MuseRmsIn {
        x: Slot,
        w: usize,
        y: Slot,
        eps: f32,
    },
    MuseGate {
        alpha: usize,
        beta: usize,
    },
    /// The experts in the latent (block `block`): the router reads H, the experts Xc; each expert's output normed (small `post_w`, eps); their weighted sum into M.
    MuseMoe {
        block: usize,
        post_w: usize,
        eps: f32,
    },
}

/// What DeepSeek-V4's operations need beyond the slots (its scratch is sized from these and the context).
#[derive(Clone, Debug)]
pub struct V4Dims {
    pub hc_eps: f32,
    pub iters: i32,
    pub heads: usize,
    pub idx_heads: usize,
    pub topk: usize,
    pub window: usize,
    pub experts_k: usize,
    /// the smallest compression whose entries a query reads all of (0: none)
    pub all_ratio: usize,
    /// the release's quantizations (the products' inputs to FP8, the caches' and the indexer's simulated FP8 and FP4); off: a development check's arithmetic without them
    pub sim: bool,
    /// the attention output's low-rank groups
    pub o_groups: usize,
}

/// cos and sin [positions, pairs] of pairs i < `pairs` of a head of `dim`: inv_freq_i = theta^(-2i / dim).
#[derive(Clone, Debug)]
pub struct Rope {
    pub theta: f64,
    pub dim: usize,
    pub pairs: usize,
}

/// A layer's keys and values, `nk` heads of `d`: a ring of the window and a chunk where `window` > 0, else the whole context.
#[derive(Clone, Debug)]
pub struct Ring {
    pub d: usize,
    pub nk: usize,
    pub window: usize,
}

#[derive(Clone, Debug)]
pub struct ModelDesc {
    pub family: &'static str,
    pub hidden: usize,
    pub vocab: usize,
    pub eps: f32,
    /// The longest context the model takes.
    pub max_pos: usize,
    /// Each slot's width a row (index: `Slot as usize`).
    pub widths: [usize; SLOTS],
    /// The matrices to pack, in load order.
    pub(crate) mats: Vec<Item>,
    /// The small tensors (norms, scalars), bf16 as saved.
    pub small: Vec<String>,
    pub ropes: Vec<Rope>,
    pub rings: Vec<Ring>,
    pub first: Op,
    pub layers: Vec<Vec<Op>>,
    /// Per-sequence states (bytes each, zeroed at a new sequence): a linear-attention layer's convolution and recurrence.
    pub states: Vec<usize>,
    /// Where the final norm is on the last rows after their residual's add (Qwen3.5, Nemotron-H): its weight, the slot added, whether a (1 + w) norm. The LM head then reads Hlast.
    pub final_norm: Option<(usize, Slot, bool)>,
    /// Long prompt chunks decode a matrix and multiply it dense (where the library's route says so), as Qwen3.5's engine does.
    pub decoded: bool,
    /// Products of up to 16 rows take the skinny kernels where they take the layout (else the library's: Qwen3.5's engine's, to the digit).
    pub skinny: bool,
    /// The embedding's matrix (its rows read by token), and the LM head's (the same where they are tied).
    pub embed: usize,
    pub lm: usize,
    /// The logits' soft-cap (0: none).
    pub softcap: f32,
    pub eos: Vec<i64>,
    /// A draft model's: the matrix that takes its last hidden state back to the big model's width (into Hid) for the next draft.
    pub post: Option<usize>,
    /// DeepSeek-V4's dimensions where the model is one.
    pub v4: Option<V4Dims>,
    /// The hyper-connections' eps and Sinkhorn rounds (GLM's operations; 0s where none).
    pub hc: (f32, i32),
    /// The logits times this before the soft-cap (Muse Spark's: cap tanh(x mult / cap), each step rounded); 0: the soft-cap alone.
    pub logit_mult: f32,
}

impl ModelDesc {
    pub fn width(&self, s: Slot) -> usize {
        self.widths[s as usize]
    }

    /// The most query heads, and the widest head, of any attention.
    fn heads(&self) -> (usize, usize) {
        let mut hd = (0, 0);
        for op in self.layers.iter().flatten() {
            if let Op::Attn(a) = op {
                hd = (hd.0.max(a.nq), hd.1.max(a.d));
            }
        }
        hd
    }
}

#[derive(Clone)]
pub struct Opts {
    pub ctx: usize,
    pub chunk: usize,
    pub graph: bool,
    pub device: i32,
    pub layout: Option<Layout>,
}

/// The layers one GPU runs of a model split across several (`split::Split`): layers `layers.0..layers.1`, the embedding where `first`, the final norm
/// and the LM head where `last`. One stage of the whole model is the model on one GPU.
#[derive(Clone, Copy, Debug)]
pub struct Stage {
    pub layers: (usize, usize),
    pub first: bool,
    pub last: bool,
}

impl Stage {
    pub fn whole(layers: usize) -> Stage {
        Stage { layers: (0, layers), first: true, last: true }
    }
}

impl Op {
    /// The matrix the operation multiplies by, the ring it reads or writes, the state it carries.
    fn mat(&self) -> Option<usize> {
        match *self {
            Op::Linear { w, .. } | Op::GateUpSilu { w, .. } | Op::V4OLora { w, in_f8: false, .. } => Some(w),
            _ => None,
        }
    }

    fn ring(&self) -> Option<usize> {
        match *self {
            Op::Attn(ref a) => Some(a.ring),
            Op::V4Kv { ring, .. } | Op::V4Attn { ring, .. } => Some(ring),
            _ => None,
        }
    }

    fn states(&self) -> Vec<usize> {
        match *self {
            Op::GdnConv { state, .. } | Op::GdnDelta { state, .. } | Op::MambaConv { state, .. } | Op::MambaScan { state, .. } => vec![state],
            Op::V4Comp { cache, state, .. } => vec![cache, state, state + 1],
            Op::KdaConv { state, .. } | Op::KdaDelta { state, .. } => vec![state],
            _ => Vec::new(),
        }
    }
}

impl ModelDesc {
    /// The matrices a stage multiplies by (its layers', the embedding where it is the first, the head and post projection where the last).
    fn wanted(&self, st: &Stage) -> Vec<bool> {
        let mut w = vec![false; self.mats.len().max(self.lm + 1).max(self.embed + 1)];
        for op in self.layers[st.layers.0..st.layers.1].iter().flatten().chain(st.first.then_some(&self.first)) {
            if let Some(k) = op.mat() {
                w[k] = true;
            }
        }
        if st.first {
            w[self.embed] = true;
        }
        if st.last {
            w[self.lm] = true;
            if let Some(p) = self.post {
                w[p] = true;
            }
        }
        w
    }
}

impl Default for Opts {
    fn default() -> Opts {
        Opts { ctx: 8192, chunk: 1024, graph: true, device: 0, layout: None }
    }
}

/// The head's rows as `Dyn::head_out` reads them: each row's greedy token and its logit, and the rows' logits as bf16 words.
pub type HeadOut = (Vec<(u32, f32)>, Vec<u16>);

/// A draft model to give `Dyn::new`: its directory (bf16, packed here), its description, and its tokens a round.
pub struct DraftSpec<'a> {
    pub dir: &'a Path,
    pub desc: ModelDesc,
    pub k: usize,
}

/// A description's weights on the GPU and its activations (`rows` rows each), its logits and picks (`out` rows).
struct Part {
    d: ModelDesc,
    mats: Vec<Mat>,
    /// The FP8 matrices (`Op::F8Lin`'s), as the family's loader made them.
    f8: Vec<crate::f8::F8Mat>,
    /// GLM's latent attention blocks (`Op::Mla`'s).
    mla: Vec<Box<dyn crate::model_glm::MlaBlock>>,
    /// The mixture-of-experts layers' blocks (`Op::Moe`'s), as the family's loader made them.
    moe: Vec<Box<dyn MoeBlock>>,
    small: Vec<Buf>,
    /// The layers whose matrices are in the computer's memory (src/dense_ram.rs).
    ram: Option<crate::dense_ram::DenseRam>,
    slots: Vec<Buf>,
    logits: Buf,
    tok: Buf,
    val: Buf,
}

impl Part {
    fn slot(&self, s: Slot) -> u64 {
        self.slots[s as usize].ptr
    }

    /// What `v` reads for `n` tokens: Mix's row in place for one token where it is a part of a fused product, else its slot.
    fn view(&self, v: View, n: usize) -> u64 {
        match v.at {
            Some(at) if n == 1 => self.slot(Slot::Mix) + (at * 2) as u64,
            _ => self.slot(v.slot),
        }
    }

    fn bytes(&self) -> usize {
        self.mats.iter().map(Mat::bytes).sum::<usize>()
            + self.f8.iter().map(|m| m.bytes).sum::<usize>()
            + self.mla.iter().map(|b| b.bytes()).sum::<usize>()
            + self.small.iter().map(|b| b.bytes).sum::<usize>()
            + self.moe.iter().map(|b| b.bytes()).sum::<usize>()
            + self.ram.as_ref().map_or(0, |r| r.bytes().0)
    }

    /// The part's weights as loaded by the family's own loader, with its activations and outputs.
    fn from_loaded(d: ModelDesc, l: Loaded, rows: usize, out: usize) -> Result<Part> {
        let slots: Vec<Buf> = d.widths.iter().map(|&w| Buf::new(w.max(1) * rows * 2)).collect::<Result<_>>()?;
        let logits = Buf::new(d.vocab * out * 2)?;
        let (tok, val) = (Buf::zeroed(4 * out)?, Buf::zeroed(4 * out)?);
        Ok(Part { d, mats: l.mats, f8: l.f8, mla: l.mla, moe: l.moe, small: l.small, ram: l.ram, slots, logits, tok, val })
    }
}

/// `d`'s weights from `dir`: a glyd-v7 directory's packs as saved (each under its first member's name), else packed here at `layout`.
fn load_part(dir: &Path, d: ModelDesc, layout: Layout, threads: usize, rows: usize, out: usize, want: Option<&[bool]>) -> Result<Part> {
    let ck = Checkpoint::open(dir).map_err(|e| Error(e.to_string()))?;
    let v7 = crate::weights::v7_manifest(dir)?;
    let one = |k: usize| -> Result<crate::weights::Packed> {
        let it = &d.mats[k];
        match &v7 {
            Some((Layout::Tiered, m)) => crate::weights::read_lossless(&ck, m, it.members[0].trim_end_matches(".weight"), it.rows, it.cols),
            Some((l, m)) => crate::weights::read_v7(&ck, m, *l, &[(it.members[0].trim_end_matches(".weight"), it.rows)], it.cols),
            None => pack_item_map(&ck, it, layout, &|_| {}),
        }
    };
    let mut packed: Vec<Option<PackBuf>> = (0..d.mats.len()).map(|_| None).collect();
    // (a stage of a split model loads only the matrices its layers multiply by)
    let ks: Vec<usize> = (0..d.mats.len()).filter(|&k| want.is_none_or(|w| w[k])).collect();
    ordered(ks.len(), threads, 2 * threads.max(1), &|i| one(ks[i]), |i, pk| {
        crate::status::STATUS.beat();
        packed[ks[i]] = Some(upload_packed(pk)?);
        Ok(())
    })?;
    // ponytail: a matrix another stage holds is an empty dense one here (never multiplied by); a Mat variant of its own if one is ever read by mistake
    let mats: Vec<Mat> = packed
        .into_iter()
        .map(|x| {
            Ok(match x {
                Some(p) => Mat::Packed(p),
                None => Mat::Dense(Buf::new(2)?, 0, 0),
            })
        })
        .collect::<Result<_>>()?;
    let small: Vec<Buf> = d
        .small
        .iter()
        .map(|name| {
            let (_, t) = ck.get(name).ok_or_else(|| Error(format!("no tensor {name}")))?;
            if t.dtype != "BF16" {
                return err(format!("{name}: {} where bf16 is wanted", t.dtype));
            }
            upload_bytes(&ck.read(name).map_err(|e| Error(e.to_string()))?)
        })
        .collect::<Result<_>>()?;
    Part::from_loaded(d, Loaded { mats, small, ..Loaded::default() }, rows, out)
}

/// What a family's loader puts on the GPU for its description: the matrices, the FP8 ones, the small tensors, the blocks of experts and of latent attention.
#[derive(Default)]
pub struct Loaded {
    pub mats: Vec<Mat>,
    /// The layers whose matrices are in the computer's memory (src/dense_ram.rs), copied into a slot before each runs.
    pub ram: Option<crate::dense_ram::DenseRam>,
    pub f8: Vec<crate::f8::F8Mat>,
    pub small: Vec<Buf>,
    pub moe: Vec<Box<dyn MoeBlock>>,
    pub mla: Vec<Box<dyn crate::model_glm::MlaBlock>>,
}

/// A draft model on the executor: its part, its tokens a round, and the positions [the queries', the last key's] its attention reads at.
struct Draft {
    p: Part,
    k: usize,
    pos: Buf,
}

/// The executor: a description's weights on the GPU (and a draft model's), the activations, the rings and the graph, and the passes over them.
pub struct Dyn {
    pub dev: Dev,
    lib: Lib,
    kern: Kern,
    kg: KernG4,
    k35: Kern35,
    knh: KernNh,
    /// The attention layer kind's kernels (`Op::Attn`).
    ka: attn::KernAttn,
    skinny: crate::skinny::SkinnyKern,
    skinny_on: bool,
    big: Part,
    draft: Option<Draft>,
    pub layout: Layout,
    ropes: Vec<(Buf, Buf)>,
    /// Each ring's bytes into kc and vc, and the positions it holds.
    ring_off: Vec<u64>,
    ring_cap: Vec<usize>,
    pub kc: Buf,
    pub vc: Buf,
    /// The per-sequence states (`ModelDesc::states`), zeroed at a new sequence.
    states: Vec<Buf>,
    scratch: Buf,
    embed_groups: Buf,
    embed_pairs: Buf,
    pos: Buf,
    st: Buf,
    lse: Buf,
    pub top: TopGpu,
    /// A decode row's attention parts (attn::SPLITS a row and query head).
    part: Buf,
    /// Nemotron-H's chunked scan's buffers (its loader's), and the mixture-of-experts layers' scratch.
    pub(crate) ssd: Option<SsdWs>,
    moe_sc: std::cell::OnceCell<crate::moe::MoeScratch>,
    /// The tokens before a prompt chunk's rows (a prompt attention's `start`).
    start: std::cell::Cell<usize>,
    /// In a pass over several sequences: the attention's preparation made for every row at once (its keys and values to 12-bit pages), its attention by sequence.
    prepped: std::cell::Cell<bool>,
    ws: Buf,
    done: Buf,
    /// The decoded-then-dense way for long chunks (ModelDesc::decoded): the library's or our own dense product, the decoded matrix's room, the GPU's code.
    blas: Option<crate::ffi::Blas>,
    own: bool,
    scratch_w: Buf,
    gpu_code: i64,
    /// A prompt's delta rule in chunks of 64 (else one token after another).
    chunked: bool,
    graph: Option<Graph>,
    host: Pinned,
    pub n: usize,
    chunk: usize,
    ctx: usize,
    /// The part of the GPU's free memory `serve_rows` takes for the sequences (`Gen::share_memory`).
    share: f64,
    /// Passes of up to this many rows take the decode step's split attention (1, or a drafted round's k + 1: the same kernel, so drafted = plain).
    split_rows: usize,
    pf_kib: [usize; 3],
    pf_shift: i32,
    pub bytes: usize,
    /// `--prof`: an event after each stage of a pass (`prof_take`); with the step graph (`prof_in_graph`), the graph's own, recorded by each replay.
    pub prof: Option<std::cell::RefCell<Vec<(&'static str, Ev)>>>,
    /// The layers this GPU runs (the whole model, or a stage of a split one).
    pub stage: Stage,
    /// DeepSeek-V4's buffers, where the model is one.
    v4: Option<deepseek_v4::V4Run>,
    /// GLM's kernels, the experts' activation, a block's mixing weights (fp32 [chunk, 24]) and the latent attention's scratch.
    kglm: crate::model_glm::KernGlm,
    kmoe: crate::moe::MoeKern,
    hcp: Buf,
    /// Muse Spark's kernels and the experts' pairs' rows (int32 [chunk, 32]).
    kmuse: muse::KernMuse,
    kkimi: kimi::KernKimi,
    mu_inv: Buf,
    mla_scratch: std::cell::OnceCell<crate::mla_glm::MlaScratch>,
    /// A verify pass's room (made at the first one), whether a pass is one, and the last chunk's (start, rows) where it was one (`rewind` goes back in it).
    verify: Option<Verify>,
    verifying: std::cell::Cell<bool>,
    pending: Option<(usize, usize)>,
    /// A verify pass's decode attention parts (VERIFY_ROWS rows; made at the first one): scratch, shared by the sequences.
    vpart: Option<Buf>,
    /// The sequences this engine holds (`add_seqs`): None at `seq`, the one in the fields above (kc, vc, states, n, pending, verify, graph); `select` swaps another in.
    seqs: Vec<Option<Seq>>,
    seq: usize,
    /// A served model with recurrent states: each sequence's states as they were at a mark (`ModelStep::mark`: its prompt's end), [slot][state], and the tokens it held there (0: no
    /// mark). A kept sequence that cannot be cut back anywhere goes on from its mark (src/dynamic/rows.rs `resume`). Empty where the model has no such state or keeps nothing.
    snaps: Vec<Vec<Buf>>,
    snap_n: Vec<usize>,
    /// A served, paged model with recurrent states: the states kept at block boundaries by the names of the blocks ending there, for any request that begins the same
    /// way (src/dynamic/snaps.rs). `every` 0: none.
    named: snaps::Snaps,
    /// A pass over the rows of several sequences (src/dynamic/rows.rs): each sequence's rows (first row, rows, its caches), the operation running on one of them now (its index), and the rows
    /// the head reads, moved to the pass's last rows first. Empty outside such a pass.
    parts: Vec<rows::Part>,
    at: std::cell::Cell<Option<usize>>,
    head_rows: Vec<usize>,
    /// Paged keys and values (R4b: src/dynamic/pages.rs): kc and vc are then the pools every sequence's blocks are in, and each sequence has a block table a layer group. None: each sequence a
    /// room of its own.
    paged: Option<rows::Paged>,
    /// The 12-bit KV tables the paged layout uses for the rings they fit (`set_kv12`), and the pages it made of them.
    kv12: Option<kv12::Tables>,
    lossless: Option<kvpages::Lossless>,
}

/// A sequence's own buffers and place, while another is selected (`Dyn::select`): its keys and values, its states, its position, the verify pass it can go back
/// in, its room for one, and the graph captured on its buffers. The activations, the workspaces and the head's outputs are the engine's, shared.
struct Seq {
    kc: Buf,
    vc: Buf,
    states: Vec<Buf>,
    n: usize,
    pending: Option<(usize, usize)>,
    verify: Option<Verify>,
    graph: Option<Graph>,
}

/// The most rows a verify pass takes (`Dyn::verify_rows`): a product of up to 8 rows runs the kernel a one-row step's does (the library's one m-tile product; the
/// tiered layout's step kernel to 8 rows), so each row's sums are a step's. Checked bit for bit on swift only; the range worker refuses verify on other levels.
pub const VERIFY_ROWS: usize = 8;

/// What `Dyn::rewind` needs of a verify pass: the per-sequence states as they were before it (`states`' sizes), each linear-attention layer's b, a rows then its
/// convolution's input rows (Mix's whole rows) of the pass (VERIFY_ROWS rows of each; None for another kind of layer or another stage's).
struct Verify {
    snap: Vec<Buf>,
    keep: Vec<Option<Buf>>,
}

// ponytail: the big model's buffers; a draft model's own where one has a block of experts
impl crate::kern_nh::NhHost for Dyn {
    fn dev(&self) -> &Dev {
        &self.dev
    }
    fn mark(&self, what: &'static str) {
        let _ = Dyn::mark(self, what);
    }
    fn lib(&self) -> &crate::ffi::Lib {
        &self.lib
    }
    fn mix(&self) -> &Buf {
        &self.big.slots[Slot::Mix as usize]
    }
    fn xc(&self) -> &Buf {
        &self.big.slots[Slot::Xc as usize]
    }
    fn chunk(&self) -> usize {
        self.chunk
    }
    fn moe_scratch(&self) -> &std::cell::OnceCell<crate::moe::MoeScratch> {
        &self.moe_sc
    }
    unsafe fn lin(&self, w: &Mat, x: u64, m: usize, y: u64) -> Result<()> {
        Dyn::lin(self, w, x, m, y)
    }
}

impl crate::model_glm::GlmHost for Dyn {
    fn kg(&self) -> &crate::model_glm::KernGlm {
        &self.kglm
    }
    fn stream(&self) -> crate::ffi::Stream {
        self.dev.stream
    }
    fn sync(&self) -> Result<()> {
        self.dev.sync()
    }
    unsafe fn lin(&self, w: &Mat, x: u64, m: usize, y: u64) -> Result<()> {
        Dyn::lin(self, w, x, m, y)
    }
    fn ctx(&self) -> usize {
        self.ctx
    }
    fn chunk(&self) -> usize {
        self.chunk
    }
    fn hidden(&self) -> usize {
        self.big.d.hidden
    }
    fn mla_scratch(&self) -> &std::cell::OnceCell<crate::mla_glm::MlaScratch> {
        &self.mla_scratch
    }
    fn mark(&self, what: &'static str) {
        let _ = Dyn::mark(self, what);
    }
    fn seq_rows(&self, n: usize) -> Vec<(usize, usize, usize)> {
        if self.parts.is_empty() {
            vec![(0, n, self.seq)]
        } else {
            self.parts.iter().map(|p| (p.row0, p.rows, p.slot)).collect()
        }
    }
}

impl Dyn {
    /// A description's model from `dir` by the generic loader (`ModelDesc::mats`, `small`, `ropes`), and a draft model's.
    pub fn new(dir: &Path, d: ModelDesc, draft: Option<DraftSpec>, o: &Opts) -> Result<Dyn> {
        let st = Stage::whole(d.layers.len());
        Dyn::new_stage(dir, d, draft, st, o)
    }

    /// Stage `st` of the model of `dir` (its layers, and the embedding or the head where it holds them), on `o.device`: a split model's part.
    pub fn new_stage(dir: &Path, d: ModelDesc, draft: Option<DraftSpec>, st: Stage, o: &Opts) -> Result<Dyn> {
        if draft.is_some() && !(st.first && st.last) {
            return err("drafting with a model split across GPUs: not yet");
        }
        let dev = Dev::new(o.device)?;
        let threads = std::thread::available_parallelism().map_or(8, |n| n.get().min(16));
        let matrix_bytes: u64 = d.mats.iter().map(|it| (it.rows * it.cols * 2) as u64).sum();
        let layout = match crate::weights::v7_manifest(dir)? {
            Some((l, _)) => l,
            None => choose_layout(o.layout, dev.cc_pair(), dev.total_mem as u64, matrix_bytes, 0).0,
        };
        let ch = o.chunk.max(1);
        let rows_out = draft.as_ref().map_or(1, |x| x.k + 1);
        if rows_out > ch {
            return err(format!("{} draft tokens a round: the chunk ({ch}) must hold them and the last token", rows_out - 1));
        }
        let mp = (o.ctx + o.chunk + 16).min(d.max_pos);
        let ropes: Vec<(Buf, Buf)> = d
            .ropes
            .iter()
            .map(|r| {
                let (c, s) = rotary(r.theta, r.dim, r.pairs, mp);
                Ok((upload_bytes(crate::weights::as_bytes(&c))?, upload_bytes(crate::weights::as_bytes(&s))?))
            })
            .collect::<Result<_>>()?;
        let want = d.wanted(&st);
        let big = load_part(dir, d, layout, threads, ch, rows_out, Some(&want))?;
        // (a draft model is packed lossy here: the big model checks every draft token, so its precision moves only how many are kept)
        let draft = match draft {
            None => None,
            Some(x) => Some(Draft { p: load_part(x.dir, x.desc, Layout::Int6, threads, 1, 1, None)?, k: x.k, pos: Buf::zeroed(8)? }),
        };
        Dyn::build(dev, big, draft, ropes, layout, st, o)
    }

    /// A model a family's own loader put on the GPU (its matrices, small tensors and rotary tables, in its description's order).
    #[allow(clippy::too_many_arguments)]
    pub fn from_loaded(
        dev: Dev,
        d: ModelDesc,
        mats: Vec<Mat>,
        small: Vec<Buf>,
        moe: Vec<Box<dyn MoeBlock>>,
        ropes: Vec<(Buf, Buf)>,
        layout: Layout,
        st: Stage,
        o: &Opts,
    ) -> Result<Dyn> {
        Dyn::from_loaded_f8(dev, d, mats, Vec::new(), small, moe, ropes, layout, st, o)
    }

    /// `from_loaded` with FP8 matrices too (`Op::F8Lin`'s).
    #[allow(clippy::too_many_arguments)]
    pub fn from_loaded_f8(
        dev: Dev,
        d: ModelDesc,
        mats: Vec<Mat>,
        f8: Vec<crate::f8::F8Mat>,
        small: Vec<Buf>,
        moe: Vec<Box<dyn MoeBlock>>,
        ropes: Vec<(Buf, Buf)>,
        layout: Layout,
        st: Stage,
        o: &Opts,
    ) -> Result<Dyn> {
        Dyn::from_parts(dev, d, Loaded { mats, f8, small, moe, mla: Vec::new(), ram: None }, ropes, layout, st, o)
    }

    /// A model its family's loader put on the GPU, whatever it holds (`Loaded`).
    #[allow(clippy::too_many_arguments)]
    pub fn from_parts(dev: Dev, d: ModelDesc, l: Loaded, ropes: Vec<(Buf, Buf)>, layout: Layout, st: Stage, o: &Opts) -> Result<Dyn> {
        let big = Part::from_loaded(d, l, o.chunk.max(1), 1)?;
        Dyn::build(dev, big, None, ropes, layout, st, o)
    }

    #[allow(clippy::too_many_arguments)]
    fn build(dev: Dev, big: Part, draft: Option<Draft>, ropes: Vec<(Buf, Buf)>, layout: Layout, stage: Stage, o: &Opts) -> Result<Dyn> {
        let lib = Lib::find()?;
        let kern = Kern::load(&dev)?;
        let kg = KernG4::load(&dev)?;
        let k35 = Kern35::load(&dev)?;
        let knh = KernNh::load(&dev)?;
        let ka = attn::KernAttn::load(&dev)?;
        let skinny = crate::skinny::SkinnyKern::load(&dev)?;
        let ch = o.chunk.max(1);
        let rows_out = draft.as_ref().map_or(1, |x| x.k + 1);
        let ctx = o.ctx.div_ceil(16) * 16;
        let d = &big.d;
        let ops = || d.layers[stage.layers.0..stage.layers.1].iter().flatten();
        let rings_used: std::collections::HashSet<usize> = ops().filter_map(Op::ring).collect();
        let whole = stage.first && stage.last;
        let (mut ring_off, mut ring_cap, mut at) = (Vec::new(), Vec::new(), 0u64);
        for (i, r) in d.rings.iter().enumerate() {
            let cap = if !whole && !rings_used.contains(&i) {
                0
            } else if r.window > 0 {
                (r.window + ch).min(ctx)
            } else {
                ctx
            };
            ring_off.push(at);
            ring_cap.push(cap);
            at += (cap * r.nk * r.d * 2) as u64;
        }
        let (kc, vc) = (Buf::zeroed(at as usize)?, Buf::zeroed(at as usize)?);
        let states_used: std::collections::HashSet<usize> = ops().flat_map(Op::states).collect();
        let states: Vec<Buf> =
            d.states.iter().enumerate().map(|(i, &b)| Buf::zeroed(if whole || states_used.contains(&i) { b.max(4) } else { 4 })).collect::<Result<_>>()?;
        let (heads, dmax) = d.heads();
        let hid = d.hidden;
        let scratch = Buf::new(64 * hid * 2)?;
        let embed_groups = Buf::new(EMBED_GROUPS * 64 * hid * 2)?;
        let embed_pairs = Buf::new(ch * 8)?;
        let (pos, st, lse) = (Buf::zeroed(4 * ch)?, Buf::zeroed(16)?, Buf::zeroed(4)?);
        let top = TopGpu::new(d.vocab)?;
        let part = Buf::zeroed(rows_out * heads.max(1) * attn::SPLITS * (dmax + 2) * 4)?;
        // (the mixture-of-experts layers are alike: one scratch for all, made for the first's shapes and a chunk's rows)
        // (a block that runs its own launches makes it the first time it runs: `NhHost::moe_scratch`)
        let moe_sc = std::cell::OnceCell::new();
        if let Some((layer, _)) = big.moe.first().and_then(|b| b.resident()) {
            let _ = moe_sc.set(crate::moe::MoeScratch::new(&dev, &lib, &[layer], ch)?);
        }
        // the decoded-then-dense way, as Qwen3.5's engine chooses it: the library's dense product off Ada, our own on Ampere and Ada
        let want = crate::tune!("GLYD_ENGINE_GEMM").unwrap_or_default();
        let blas = if d.decoded && ch > 64 && (want == "blas" || (want.is_empty() && dev.cc != (8, 9))) { crate::ffi::Blas::find() } else { None };
        let own = d.decoded && (want == "own" || (want.is_empty() && blas.is_none() && matches!(dev.cc.0, 8 | 9)));
        let gpu_code = lib.gpu()? as i64;
        let packed: Vec<&PackBuf> = (0..big.mats.len()).filter(|&k| k != d.lm && k != d.embed).filter_map(|k| big.mats[k].packed_opt()).collect();
        // (the blocks' own matrices too: the experts', the latent attentions')
        let block_shapes: Vec<(i64, i64, bool)> = big.moe.iter().flat_map(|b| b.pack_shapes()).chain(big.mla.iter().flat_map(|b| b.pack_shapes())).collect();
        let biggest = packed.iter().map(|p| (p.pack.rows * p.pack.cols) as usize).chain(block_shapes.iter().map(|s| (s.0 * s.1) as usize)).max().unwrap_or(8);
        let scratch_w = if blas.is_some() || own { Buf::new(biggest * 2)? } else { Buf::new(16)? };
        // the library's workspace: every row count a layer's product takes, the heads' at the rows they read, the draft model's at one row
        let mut shapes: Vec<(i64, i64, bool)> = packed.iter().map(|p| (p.pack.rows, p.pack.cols, p.pack.is_twelve())).collect();
        shapes.extend(block_shapes.iter().copied());
        shapes.sort_unstable();
        shapes.dedup();
        let mut need = 0usize;
        for m_ in 1..=ch as i64 {
            for &(oo, kk, twelve) in &shapes {
                need = need.max(lib.linear_workspace(twelve, oo, kk, m_)?);
            }
        }
        if let Some(lm) = big.mats[d.lm].packed_opt() {
            need = need.max(lib.linear_workspace(lm.pack.is_twelve(), lm.pack.rows, lm.pack.cols, rows_out as i64)?);
        }
        // (the FP8 matrices' products: every row count a call takes, up to 64)
        let mut f8_shapes: Vec<(usize, usize)> = big.f8.iter().map(|m| (m.n, m.k)).collect();
        f8_shapes.sort_unstable();
        f8_shapes.dedup();
        for &(n, k) in &f8_shapes {
            for m_ in 1..=ch.min(64) as i64 {
                need = need.max(lib.f8_linear_workspace(n as i64, k as i64, m_)?);
            }
        }
        // (DeepSeek-V4's grouped low rank as the experts' product: each group a stack's expert)
        if let Some(g) = d.v4.as_ref().map(|v| v.o_groups).filter(|&g| g > 0) {
            for m in big.f8.iter().filter(|m| m.n % g == 0 && (m.n / g) % 64 == 0) {
                need = need.max(m.moe_workspace(&lib, g, ch.min(64), g)?);
            }
        }
        if let Some(dr) = &draft {
            for m in dr.p.mats.iter().filter_map(Mat::packed_opt) {
                need = need.max(lib.linear_workspace(m.pack.is_twelve(), m.pack.rows, m.pack.cols, 1)?);
            }
        }
        let ws = Buf::zeroed(need)?;
        let done = Buf::zeroed(4 << 18)?;
        let host = Pinned::new(4096)?;
        let (kglm, kmoe, hcp) = (crate::model_glm::KernGlm::load(&dev)?, crate::moe::MoeKern::load(&dev)?, Buf::zeroed(ch * 24 * 4)?);
        let (kmuse, mu_inv) = (muse::KernMuse::load(&dev)?, Buf::zeroed(ch * 32 * 4)?);
        let kkimi = kimi::KernKimi::load(&dev)?;
        let v4 = match &d.v4 {
            Some(v) => Some(deepseek_v4::V4Run::new(&dev, d, v, ch, ctx)?),
            None => None,
        };
        let bytes = big.bytes() + draft.as_ref().map_or(0, |x| x.p.bytes());
        let mut e = Dyn {
            dev,
            lib,
            kern,
            kg,
            k35,
            knh,
            ka,
            skinny,
            skinny_on: crate::tune!("GLYD_G4_SKINNY").map_or(big.d.skinny, |v| v != "0"),
            big,
            draft,
            layout,
            ropes,
            ring_off,
            ring_cap,
            kc,
            vc,
            states,
            scratch,
            embed_groups,
            embed_pairs,
            pos,
            st,
            lse,
            top,
            part,
            ssd: None,
            moe_sc,
            start: std::cell::Cell::new(0),
            prepped: std::cell::Cell::new(false),
            ws,
            done,
            blas,
            own,
            scratch_w,
            gpu_code,
            chunked: crate::tune!("GLYD_ENGINE_GDN").as_deref() != Some("recurrent"),
            graph: None,
            host,
            n: 0,
            chunk: ch,
            share: 1.0,
            ctx,
            split_rows: rows_out,
            pf_kib: pf_kib(crate::tune!("GLYD_G4_PF").as_deref())?,
            pf_shift: crate::tune!("GLYD_G4_PF_SHIFT").and_then(|v| v.parse().ok()).filter(|v| (5..=7).contains(v)).unwrap_or(7),
            bytes,
            prof: None,
            stage,
            v4,
            kglm,
            kmoe,
            hcp,
            kmuse,
            kkimi,
            mu_inv,
            mla_scratch: std::cell::OnceCell::new(),
            verify: None,
            verifying: std::cell::Cell::new(false),
            pending: None,
            vpart: None,
            seqs: vec![None],
            seq: 0,
            snaps: Vec::new(),
            named: snaps::Snaps::none(),
            snap_n: Vec::new(),
            parts: Vec::new(),
            at: std::cell::Cell::new(None),
            head_rows: Vec::new(),
            paged: None,
            kv12: None,
            lossless: None,
        };
        // (a group of GPUs: every one's engine made before any one's first pass, src/dynamic/tp.rs)
        if let Some(c) = tp::comm() {
            c.barrier()?;
        }
        // (a block that reads the router's choices on the host, an expert cache's, is not captured)
        if o.graph && e.big.moe.iter().all(|b| !b.reads_host() && (b.resident().is_some() || b.graphable())) {
            e.capture_graph()?;
        }
        Ok(e)
    }

    pub fn desc(&self) -> &ModelDesc {
        &self.big.d
    }

    /// The big model's weights on the GPU (a draft model's not counted).
    pub fn weights_bytes(&self) -> usize {
        self.big.bytes()
    }

    pub fn describe(&self) -> String {
        let gib = |b: usize| b as f64 / (1u64 << 30) as f64;
        let (free, total) = self.dev.mem_info().unwrap_or((0, 0));
        let dr = self.draft.as_ref().map_or(String::new(), |x| format!("; draft model {:.3} GiB, {} draft tokens a round", gib(x.p.bytes()), x.k));
        format!(
            "{} (sm_{}{}): {} on the dynamic engine, {} layers, weights {:.3} GiB (layout {:?}), KV {:.3} GiB ({} positions), {:.2} of {:.2} GiB in use{dr}",
            self.dev.name,
            self.dev.cc.0,
            self.dev.cc.1,
            self.big.d.family,
            if self.stage.first && self.stage.last {
                self.big.d.layers.len().to_string()
            } else {
                format!("{}..{} of {}", self.stage.layers.0, self.stage.layers.1, self.big.d.layers.len())
            },
            gib(self.big.bytes()),
            self.layout,
            gib(self.kc.bytes + self.vc.bytes),
            self.ctx,
            gib(total - free),
            gib(total)
        )
    }

    pub fn has_graph(&self) -> bool {
        self.graph.is_some()
    }

    pub fn reset(&mut self) -> Result<()> {
        self.n = 0;
        self.pending = None;
        self.top.ask = None;
        for b in &self.states {
            b.fill_async(0, self.dev.stream)?;
        }
        self.dev.sync()
    }

    pub fn capacity(&self) -> usize {
        self.ctx
    }

    pub fn drafting(&self) -> bool {
        self.draft.is_some()
    }

    /// The last pass's logits (soft-capped), widened.
    pub fn logits(&self) -> Result<Vec<f32>> {
        Ok(self.logit_words()?.into_iter().map(|w| f32::from_bits((w as u32) << 16)).collect())
    }

    /// The last pass's logits as the bf16 words the head wrote (the last row's).
    pub fn logit_words(&self) -> Result<Vec<u16>> {
        self.dev.sync()?;
        let mut words = vec![0u16; self.big.d.vocab];
        self.big.logits.read(0, &mut words)?;
        Ok(words)
    }

    fn mark(&self, what: &'static str) -> Result<()> {
        if let Some(p) = &self.prof {
            let e = Ev::new()?;
            // (a node of the graph where one is being captured; outside a capture the driver refuses that: an event of a pass without the graph, where
            // there is none, else (a prompt's pass beside the graph's) none)
            if e.record_in_graph(self.dev.stream).is_err() {
                if self.graph.is_some() {
                    return Ok(());
                }
                e.record(self.dev.stream)?;
            }
            p.borrow_mut().push((what, e));
        }
        Ok(())
    }

    /// `--prof` in the step graph (a development build's GLYD_PROF_GRAPH=1): the graph captured again with an event after each stage, which every replay
    /// records (the stages' times as the graph runs them: without it a step of small kernels waits on the host's launches).
    pub fn prof_in_graph(&mut self) -> Result<()> {
        self.prof = Some(std::cell::RefCell::new(Vec::new()));
        if self.graph.take().is_some() {
            self.capture_graph()?;
        }
        Ok(())
    }

    /// The stages' times since the last call (ms, summed by stage, in the order first seen), the events cleared; with the step graph, its last replay's,
    /// the graph's events kept.
    pub fn prof_take(&self) -> Result<Vec<(&'static str, f64)>> {
        let Some(p) = &self.prof else { return Ok(Vec::new()) };
        self.dev.sync()?;
        let evs = if self.graph.is_some() { Vec::new() } else { std::mem::take(&mut *p.borrow_mut()) };
        let kept = p.borrow();
        let evs = if self.graph.is_some() { &kept[..] } else { &evs[..] };
        let mut out: Vec<(&'static str, f64)> = Vec::new();
        for w in evs.windows(2) {
            let ms = w[1].1.since(&w[0].1)? as f64;
            match out.iter_mut().find(|(k, _)| *k == w[1].0) {
                Some(x) => x.1 += ms,
                None => out.push((w[1].0, ms)),
            }
        }
        Ok(out)
    }

    fn skinny_takes(&self, p: &PackBuf, m: usize) -> bool {
        self.skinny_on && (1..=16).contains(&m) && p.pack.rows >= 2048 && crate::skinny::SkinnyKern::takes(&p.pack)
    }

    /// The L2 prefetch of `kib` KiB of each first-wave block of `p` (where `lin` takes it skinny) and the spare blocks of `threads` that issue it (GLY-312).
    fn pf(&self, w: &Mat, m: usize, kib: usize, threads: u32) -> (G4Pf, u32) {
        let Mat::Packed(p) = w else { return (G4Pf::default(), 0) };
        let Some((data, scales, ds, ss)) = crate::skinny::SkinnyKern::regions(&p.pack).filter(|_| kib > 0 && self.skinny_takes(p, m)) else {
            return (G4Pf::default(), 0);
        };
        let dhead = ((kib as u64 * 1024).min(ds) / 128 * 128) as i32;
        let shead = ((dhead as u64 * ss / ds).div_ceil(128) * 128).min(ss) as i32;
        let regions = ((p.pack.rows / 64) as i32).min(2 * self.dev.sms);
        let lines = regions as u32 * ((dhead + shead) >> self.pf_shift) as u32;
        (G4Pf { data, scales, dstride: ds as i64, sstride: ss as i64, dhead, shead, regions, shift: self.pf_shift }, lines.div_ceil(4 * threads))
    }

    unsafe fn lin(&self, w: &Mat, x: u64, m: usize, y: u64) -> Result<()> {
        let s = self.dev.stream;
        let p = match w {
            Mat::Packed(p) => p,
            Mat::Dense(b, rows, cols) => return self.k35.small_linear(s, x, *cols as i32, b.ptr, y, *rows as i32, m as u32, *rows as i32, *cols as i32),
        };
        // (this GPU's slice of the rows: its product, then every GPU's slices gathered into y: src/dynamic/tp.rs)
        if p.tp_rows > 0 {
            let c = tp::comm().ok_or_else(|| Error("a matrix sliced across GPUs, on a thread of no group of them".into()))?;
            let sl = p.pack.rows as usize;
            let t = c.temp(s, m * sl * 2)?;
            self.lin_whole(p, x, m, t)?;
            return c.gather(s, t, m, sl, y, p.tp_rows);
        }
        self.lin_whole(p, x, m, y)
    }

    /// `lin` of a packed matrix as it is (a slice of one: its rows alone).
    unsafe fn lin_whole(&self, p: &PackBuf, x: u64, m: usize, y: u64) -> Result<()> {
        let s = self.dev.stream;
        if self.skinny_takes(p, m) {
            return self.skinny.gemv(s, &p.pack, x, y, m);
        }
        if self.decoded_way(p, m)? {
            self.lib.unpack(&p.pack, 0, p.pack.rows, self.scratch_w.ptr, s)?;
            if let Some(b) = &self.blas {
                return b.gemm_nt(s, self.scratch_w.ptr, x, y, p.pack.rows as i32, p.pack.cols as i32, m as i32);
            }
            let tile = self.kern.pick_tile(m, p.pack.rows as usize, p.pack.cols as usize);
            return self.kern.gemm(s, tile, x, self.scratch_w.ptr, y, m as i32, p.pack.rows as i32, p.pack.cols as i32);
        }
        self.lib.linear(&p.pack, x, m as i64, y, self.ws.ptr, self.ws.bytes, self.done.ptr, s)
    }

    /// Whether a product of `m` rows decodes the matrix and multiplies dense (ModelDesc::decoded, the library's route DECODE or AHEAD), as Qwen3.5's engine.
    unsafe fn decoded_way(&self, p: &PackBuf, m: usize) -> Result<bool> {
        if (self.blas.is_none() && !self.own) || m <= 64 || p.pack.cols % 64 != 0 || p.pack.rows % 128 != 0 {
            return Ok(false);
        }
        let route = self.lib.route(self.gpu_code, p.pack.is_twelve(), p.pack.rows, p.pack.cols, m as i64)?;
        Ok(route == 0 || route == 5)
    }

    /// The matrix the next product after op `i` of layer `l` of `p` reads, and its rows (the LM head's after the last layer: `head`).
    fn next_linear(p: &Part, l: usize, i: usize, n: usize, head: usize) -> (&Mat, usize) {
        let later = p.d.layers[l][i + 1..].iter().chain(p.d.layers[l + 1..].iter().flatten());
        for op in later {
            if let Op::Linear { w, .. } = op {
                return (&p.mats[*w], n);
            }
        }
        (&p.mats[p.d.post.unwrap_or(p.d.lm)], head.max(1))
    }

    /// The tokens' raw embedding rows into the big model's X.
    unsafe fn embed(&self, tokens: &[u32]) -> Result<()> {
        let hid = self.big.d.hidden;
        let s = self.dev.stream;
        let Some(e) = self.big.mats[self.big.d.embed].packed_opt().map(|p| &p.pack) else { return err("the embedding is not packed") };
        let x = self.big.slot(Slot::X);
        if let Some(&t) = tokens.iter().find(|&&t| t as usize >= self.big.d.vocab) {
            return err(format!("token {t}: the vocabulary has {}", self.big.d.vocab));
        }
        if let Some(v) = &self.v4 {
            v.tok.write(0, &tokens.iter().map(|&t| t as i32).collect::<Vec<i32>>())?;
        }
        if tokens.len() == 1 {
            let t = tokens[0] as i64;
            let row0 = (t / 64) * 64;
            self.lib.unpack(e, row0, 64, self.scratch.ptr, s)?;
            return self.kern.copy_row(s, self.scratch.ptr, (t - row0) as i32, hid as i32, x);
        }
        let (groups, by_batch) = embed_plan(tokens, EMBED_GROUPS);
        let counts: Vec<usize> = by_batch.iter().map(Vec::len).collect();
        let pairs: Vec<i32> = by_batch.iter().flatten().flat_map(|&(d, s_)| [d, s_]).collect();
        self.embed_pairs.write(0, &pairs)?;
        let mut done = 0usize;
        for (b, batch) in groups.chunks(EMBED_GROUPS).enumerate() {
            for (j, &g) in batch.iter().enumerate() {
                self.lib.unpack(e, g * 64, 64, self.embed_groups.ptr + (j * 64 * hid * 2) as u64, s)?;
            }
            self.kern.copy_rows(s, self.embed_groups.ptr, self.embed_pairs.ptr + (done * 8) as u64, counts[b] as i32, hid as i32, x)?;
            done += counts[b];
        }
        Ok(())
    }

    /// One operation of part `p` on n rows (op `i` of layer `l`), the queries at positions `pos.0` and the keys read up to `pos.1` (the same for the big model;
    /// a draft model's query and last key differ); `head`: the rows the LM head reads after the last layer.
    #[allow(clippy::too_many_arguments)]
    /// Hummingbird's corpus pass (src/hb_usage.rs): a MoE operation's router choices of its `n` rows read back and counted for layer `l`; a
    /// layer whose experts are in a cache that routes on its own is said to be missed.
    fn record_routes(&self, p: &Part, l: usize, op: &Op, n: usize) -> Result<()> {
        let block = match *op {
            Op::Moe { block } | Op::MuseMoe { block, .. } | Op::V4Moe { block, .. } => block,
            _ => return Ok(()),
        };
        let Some((layer, _)) = p.moe[block].resident() else {
            crate::hb_usage::missed(l);
            return Ok(());
        };
        let sc = self.moe_sc.get().ok_or_else(|| Error("no mixture-of-experts scratch".into()))?;
        self.dev.sync()?;
        let mut ids = vec![0i64; n * layer.cfg.router.top_k];
        sc.ids.read(0, &mut ids)?;
        crate::hb_usage::record(l, layer.cfg.experts, &ids)
    }

    unsafe fn op(&self, p: &Part, l: usize, i: usize, op: &Op, n: usize, pos: (u64, u64), head: usize) -> Result<()> {
        let s = self.dev.stream;
        let (nu, eps, hid) = (n as u32, p.d.eps, p.d.hidden as i32);
        let nt = norm_threads(hid)?;
        match *op {
            Op::EmbedNorm { scale, norm } => {
                launch!(self.kg.embed_norm, [nu, 1, 1], [nt, 1, 1], s; p.slot(Slot::X), scale, p.small[norm].ptr, p.slot(Slot::H), hid, eps)
            }
            Op::Linear { w, x, y, name } => {
                self.lin(&p.mats[w], p.slot(x), n, p.slot(y))?;
                self.mark(name)
            }
            Op::F8Lin { w, x, y, name } => {
                p.f8[w].linear(&self.lib, p.slot(x), n, p.slot(y), self.ws.ptr, self.ws.bytes, self.done.ptr, s)?;
                self.mark(name)
            }
            Op::Attn(ref a) => self.attn(p, l, i, a, n, pos, head),
            Op::AddNorm { y, post, scalar, next } => {
                let (nx, rows) = Self::next_linear(p, l, i, n, head);
                let (pf, pfb) = self.pf(nx, rows, self.pf_kib[1], nt);
                let sc = scalar.map_or(0u64, |k| p.small[k].ptr);
                launch!(self.kg.add_norm, [nu, 1 + pfb.div_ceil(nu), 1], [nt, 1, 1], s; p.slot(Slot::X), p.slot(y), p.small[post].ptr, sc, p.small[next].ptr, p.slot(Slot::H), hid, eps, pf)?;
                self.mark("norms")
            }
            Op::GeluMul { inter } => {
                let gx = (inter as u32).div_ceil(256).min(64);
                let (nx, rows) = Self::next_linear(p, l, i, n, head);
                let (pf, pfb) = self.pf(nx, rows, self.pf_kib[2], 256);
                launch!(self.kg.gelu, [gx, nu, 1 + pfb.div_ceil(gx * nu)], [256, 1, 1], s; p.slot(Slot::Gu), p.slot(Slot::M), inter as i32, pf)?;
                self.mark("gelu")
            }
            Op::Norm { x, w, y } => launch!(self.kg.norm, [nu, 1, 1], [nt, 1, 1], s; p.slot(x), p.small[w].ptr, p.slot(y), hid, eps),
            Op::DraftIn { scale } => {
                let bh = self.big.d.hidden as i32;
                launch!(self.kg.draft_in, [((2 * bh) as u32).div_ceil(256).min(64), 1, 1], [256, 1, 1], s; self.big.slot(Slot::X), scale, p.slot(Slot::Hid), p.slot(Slot::In), bh)
            }
            Op::RmsNorm { x, w, y, plus1 } => {
                // (the slot's own width: the hidden size, or a narrower activation's)
                let k = p.d.width(x) as i32;
                if plus1 {
                    self.k35.rmsnorm(s, p.slot(x), p.small[w].ptr, p.slot(y), nu, k, eps)?;
                } else {
                    self.kern.rmsnorm(s, p.slot(x), p.small[w].ptr, p.slot(y), nu, k, eps)?;
                }
                self.mark("norms")
            }
            Op::NormAdd { y, w, plus1 } => {
                if plus1 {
                    self.k35.rmsnorm_add(s, p.slot(Slot::X), p.slot(y), p.small[w].ptr, p.slot(Slot::H), nu, hid, eps)?;
                } else {
                    self.kern.rmsnorm_add(s, p.slot(Slot::X), p.slot(y), p.small[w].ptr, p.slot(Slot::H), nu, hid, eps)?;
                }
                self.mark("norms")
            }
            Op::GdnConv { w, state, cd, row } => {
                if self.verifying.get() {
                    // (a verify pass: this layer's b, a rows and convolution input kept for `rewind`)
                    let k = self.verify.as_ref().and_then(|v| v.keep[l].as_ref()).ok_or_else(|| Error(format!("layer {l}: no room kept for a verify pass")))?;
                    let ba = p.d.width(Slot::Ba);
                    copy_d2d_async(k.ptr, p.slot(Slot::Ba), n * ba * 2, s)?;
                    copy_d2d_async(k.ptr + (VERIFY_ROWS * ba * 2) as u64, p.slot(Slot::Mix), n * row * 2, s)?;
                }
                let (x, y) = (p.slot(Slot::Mix) + self.rows_before(row), p.slot(Slot::Xc) + self.rows_before(cd));
                self.k35.conv(s, x, row as i32, p.small[w].ptr, self.state(state), y, n as i32, cd as i32)?;
                self.mark("linear attention: convolution")
            }
            Op::GdnDelta { a_log, dt_bias, state, cd, nk, nv } => {
                // (a verify pass's rows one after another, as steps)
                let chunked = self.chunked && n > 1 && !self.verifying.get();
                self.k35.delta(
                    s,
                    chunked,
                    p.slot(Slot::Xc) + self.rows_before(cd),
                    cd as i32,
                    p.slot(Slot::Ba) + self.rows_before(2 * nv),
                    2 * nv as i32,
                    p.small[a_log].ptr,
                    p.small[dt_bias].ptr,
                    self.state(state),
                    p.slot(Slot::Core) + self.rows_before(nv * 128),
                    n as i32,
                    nk as i32,
                    nv as i32,
                )?;
                self.mark("linear attention: delta rule")
            }
            Op::GatedNorm { w, row, nv, z_off } => {
                self.k35.gated_norm(s, p.slot(Slot::Core), p.slot(Slot::Mix) + (z_off * 2) as u64, row as i32, p.small[w].ptr, nu, nv as i32, eps)?;
                self.mark("linear attention: gated norm")
            }
            Op::GateUpSilu { w, inter } => {
                // (our own dense product fuses the SiLU where the matrix is decoded first, as Qwen3.5's engine)
                if let Mat::Packed(pk) = &p.mats[w] {
                    if pk.tp_rows == 0
                        && self.blas.is_none()
                        && self.own
                        && self.decoded_way(pk, n)?
                        && pk.pack.rows as usize == 2 * inter
                        && inter.is_multiple_of(128)
                    {
                        self.lib.unpack(&pk.pack, 0, pk.pack.rows, self.scratch_w.ptr, s)?;
                        self.kern.gemm_silu(s, p.slot(Slot::H), self.scratch_w.ptr, p.slot(Slot::M), nu as i32, pk.pack.rows as i32, pk.pack.cols as i32)?;
                        return self.mark("MLP: gate and up product, SiLU");
                    }
                }
                self.lin(&p.mats[w], p.slot(Slot::H), n, p.slot(Slot::Gu))?;
                self.kern.silu_mul(s, p.slot(Slot::Gu), p.slot(Slot::M), nu, inter as i32)?;
                self.mark("MLP: gate and up product, SiLU")
            }
            Op::MambaConv { w, bias, state, cd, row, x_off } => {
                // (in a pass over several sequences: this sequence's rows and its state)
                self.knh.conv(
                    s,
                    p.slot(Slot::Mix) + self.rows_before(row) + (x_off * 2) as u64,
                    row as i32,
                    p.small[w].ptr,
                    p.small[bias].ptr,
                    self.state(state),
                    p.slot(Slot::Xc) + self.rows_before(cd),
                    n as i32,
                    cd as i32,
                )?;
                self.mark("Mamba-2: convolution")
            }
            Op::MambaScan { dt_bias, a_log, d, state, cd, row, dt_off, nh, ng } => {
                let (dtr, st) = (p.slot(Slot::Mix) + self.rows_before(row) + (dt_off * 2) as u64, self.state(state));
                let (dtb, al, dd) = (p.small[dt_bias].ptr, p.small[a_log].ptr, p.small[d].ptr);
                let (core, xc) = (p.slot(Slot::Core) + self.rows_before(nh * crate::kern_nh::NH_P), p.slot(Slot::Xc) + self.rows_before(cd));
                match &self.ssd {
                    Some(ws) if self.chunked && n > 1 => {
                        self.knh.ssd(s, ws, xc, cd as i32, dtr, row as i32, dtb, al, dd, st, core, n as i32, nh as i32, ng as i32, 0.0)?
                    }
                    _ => self.knh.ssm(s, xc, cd as i32, dtr, row as i32, dtb, al, dd, st, core, n as i32, nh as i32, ng as i32, 0.0)?,
                }
                self.mark("Mamba-2: scan")
            }
            Op::MambaNorm { w, row, di, ng } => {
                self.knh.gated_norm(s, p.slot(Slot::Core), p.slot(Slot::Mix), row as i32, p.small[w].ptr, nu, di as i32, ng as i32, eps)?;
                self.mark("Mamba-2: gated norm")
            }
            Op::Moe { block } => {
                let Some((layer, fc)) = p.moe[block].resident() else {
                    // (experts in a cache, or at the hummingbird level: the block's own launches, over this engine's buffers)
                    p.moe[block].forward(self, p.slot(Slot::H), n, p.slot(Slot::O))?;
                    return self.mark("mixture of experts");
                };
                let sc = self.moe_sc.get().ok_or_else(|| Error("no mixture-of-experts scratch".into()))?;
                let (h, o) = (p.slot(Slot::H), p.slot(Slot::O));
                match fc {
                    None => layer.forward(&self.lib, sc, s, h, n, o)?,
                    Some((fc1, fc2)) => {
                        layer.route(sc, s, h, layer.cfg.router_hidden, n)?;
                        self.lin(fc1, h, n, p.slot(Slot::Mix))?;
                        layer.experts(&self.lib, sc, s, p.slot(Slot::Mix), n, p.slot(Slot::Xc))?;
                        self.lin(fc2, p.slot(Slot::Xc), n, o)?;
                        layer.add_shared(&self.lib, sc, s, h, n, o)?;
                    }
                }
                self.mark("mixture of experts")
            }
            Op::GlmStreams => self.kglm.streams(s, p.slot(Slot::X), p.slot(Slot::S), nu, hid),
            Op::GlmPre { f, base, scale, norm } => {
                self.kglm.hc_mix_raw(
                    s,
                    p.slot(Slot::S),
                    p.small[f].ptr,
                    p.small[base].ptr,
                    p.small[scale].ptr,
                    p.small[norm].ptr,
                    p.slot(Slot::H),
                    self.hcp.ptr,
                    nu,
                    hid,
                    eps,
                    p.d.hc.0,
                    p.d.hc.1,
                )?;
                self.mark("hyper-connections")
            }
            Op::GlmPost => {
                self.kglm.hc_post(s, p.slot(Slot::S), p.slot(Slot::O), self.hcp.ptr, nu, hid)?;
                self.mark("hyper-connections")
            }
            Op::GlmHead { norm } => self.kglm.hc_head(s, p.slot(Slot::S), p.small[norm].ptr, p.slot(Slot::H), nu, hid, eps),
            Op::KdaConv { w, state, c, row, w32 } => {
                let (x, y) = (p.slot(Slot::Mix) + self.rows_before(row), p.slot(Slot::Xc) + self.rows_before(c));
                self.kglm.conv(s, x, row as i32, p.small[w].ptr, w32, self.state(state), y, n as i32, c as i32)?;
                self.mark("linear attention: convolution")
            }
            Op::KdaDelta { a_log, dt_bias, state, heads, bound, ba } => {
                // (in a step of several sequences the gates' rows are the copied slot's (Op::KimiParts ran on all the rows), the sequence's at its offset)
                let bb = if self.at.get().is_some() { p.slot(ba.slot) + self.rows_before(heads) } else { p.view(ba, n) };
                self.kglm.delta(
                    s,
                    self.chunked && n > 1,
                    p.slot(Slot::Xc) + self.rows_before(3 * heads * 128),
                    p.slot(Slot::Qo) + self.rows_before(heads * 128),
                    p.small[dt_bias].ptr,
                    p.small[a_log].ptr,
                    bb,
                    self.state(state),
                    p.slot(Slot::Core) + self.rows_before(heads * 128),
                    n as i32,
                    heads as i32,
                    bound,
                )?;
                self.mark("linear attention: delta rule")
            }
            Op::KdaOut { w, heads, w32, iq } => {
                self.kglm.delta_out(s, p.slot(Slot::Core), p.view(iq, n), p.small[w].ptr, w32, p.slot(Slot::Att), nu, heads as i32, eps)?;
                self.mark("linear attention: gated norm")
            }
            Op::LinearAt { w, x, y, name } => {
                self.lin(&p.mats[w], p.view(x, n), n, p.slot(y))?;
                self.mark(name)
            }
            Op::KimiParts { row, parts } => {
                if n > 1 {
                    for (at, w, slot) in parts {
                        self.kkimi.rows(s, p.slot(Slot::Mix) + (at * 2) as u64, row, p.slot(slot), w, w, n)?;
                    }
                }
                Ok(())
            }
            Op::Mla { block } => {
                p.mla[block].forward(self, p.slot(Slot::H), n, pos.0, self.start.get(), p.slot(Slot::O))?;
                self.mark("latent attention")
            }
            Op::KimiMixNorm { add, nb, w_n, w_p, push, w, eps } => {
                let (blk, hd) = (p.slot(Slot::Blk), hid as usize);
                let o = if add { p.slot(Slot::O) } else { 0 };
                let at = push.map_or(0, |b| blk + (b * hd * 2) as u64);
                self.kkimi.mix_norm(
                    s,
                    blk,
                    p.d.width(Slot::Blk),
                    nb,
                    p.slot(Slot::X),
                    o,
                    p.small[w_n].ptr,
                    p.small[w_p].ptr,
                    at,
                    p.small[w].ptr,
                    p.slot(Slot::H),
                    p.d.width(Slot::H),
                    hd,
                    eps,
                    nu,
                )?;
                self.mark("depth mix and norm")
            }
            Op::KimiAdd { x } => self.kkimi.add(s, p.slot(Slot::X), p.slot(x), n * hid as usize),
            Op::KimiEarly { block, nb, w_n, w_p, w, eps } => {
                if n != 1 || !p.moe[block].early_wanted() {
                    return Ok(());
                }
                let (h, hd) = (p.slot(Slot::Att), hid as usize);
                self.kkimi.mix_norm(
                    s,
                    p.slot(Slot::Blk),
                    p.d.width(Slot::Blk),
                    nb,
                    p.slot(Slot::X),
                    0,
                    p.small[w_n].ptr,
                    p.small[w_p].ptr,
                    0,
                    p.small[w].ptr,
                    h,
                    hd,
                    hd,
                    eps,
                    1,
                )?;
                p.moe[block].early(self, h)?;
                self.mark("experts: the early guess")
            }
            Op::Situ { inter, beta, linear } => {
                self.kmoe.act(s, p.slot(Slot::Gu), p.slot(Slot::M), n, inter, &crate::moe::Act::Situ { beta, linear })?;
                self.mark("MLP: activation")
            }
            Op::Nop => Ok(()),
            Op::SwiGluClamp { inter, limit } => {
                self.kmoe.act(s, p.slot(Slot::Gu), p.slot(Slot::M), n, inter, &crate::moe::Act::SwiGlu { limit: Some(limit) })?;
                self.mark("MLP: activation")
            }
            Op::MuseRms { x, w, y, eps } => {
                self.kern.rmsnorm(s, p.slot(x), p.small[w].ptr, p.slot(y), nu, p.d.width(x) as i32, eps)?;
                self.mark("norms")
            }
            Op::MuseRmsIn { x, w, y, eps } => {
                let k = &self.kmuse;
                launch!(k.rms_in, [nu, 1, 1], [256, 1, 1], s; p.slot(x), p.small[w].ptr, p.slot(y), p.d.width(x) as i32, eps)?;
                self.mark("norms")
            }
            Op::MuseGate { alpha, beta } => {
                let k = &self.kmuse;
                launch!(k.rgate, [nu, (hid as u32).div_ceil(256), 1], [256, 1, 1], s; p.slot(Slot::X), p.slot(Slot::O), p.small[alpha].ptr, p.small[beta].ptr, hid)?;
                self.mark("residual gates")
            }
            Op::MuseMoe { block, post_w, eps: pe } => {
                let (layer, _) = p.moe[block].resident().ok_or_else(|| Error("Muse Spark's experts are not on the GPU".into()))?;
                let sc = self.moe_sc.get().ok_or_else(|| Error("no mixture-of-experts scratch".into()))?;
                layer.forward_muse(
                    &self.lib,
                    sc,
                    s,
                    p.slot(Slot::H),
                    p.slot(Slot::Xc),
                    n,
                    p.slot(Slot::M),
                    layer.w.router.ptr,
                    p.d.logit_mult,
                    p.small[post_w].ptr,
                    pe,
                    self.mu_inv.ptr,
                )?;
                self.mark("mixture of experts")
            }
            Op::V4Streams
            | Op::V4Pre { .. }
            | Op::V4Post
            | Op::V4Head { .. }
            | Op::Fp8 { .. }
            | Op::V4Q { .. }
            | Op::V4Kv { .. }
            | Op::V4Comp { .. }
            | Op::V4IdxQ { .. }
            | Op::V4IdxPick { .. }
            | Op::V4Attn { .. }
            | Op::V4OLora { .. }
            | Op::V4OLoraStack { .. }
            | Op::V4Moe { .. } => self.op_v4(p, op, n, pos.0),
        }
    }

    /// One of DeepSeek-V4's operations (kernels/dsv4.cu) on n rows at positions `pos`.
    unsafe fn op_v4(&self, p: &Part, op: &Op, n: usize, pos: u64) -> Result<()> {
        let v = self.v4.as_ref().ok_or_else(|| Error("a DeepSeek-V4 operation in a model loaded without its buffers".into()))?;
        let (s, nu, eps, hid) = (self.dev.stream, n as u32, p.d.eps, p.d.hidden as i32);
        // (in a pass over several sequences: the first row of the sequence an operation runs on now, and the pass's rows, which the attention's output is laid out by)
        let (row0, rows) = match self.at.get() {
            Some(k) => (self.parts[k].row0, self.parts.iter().map(|q| q.row0 + q.rows).max().unwrap_or(n)),
            None => (0, n),
        };
        let k = &v.k;
        let ks = (4 * p.d.hidden / 256) as u32; // (kernels/dsv4.cu V4_HC_CPB)
        let rope = |r: usize| (self.ropes[r].0.ptr, self.ropes[r].1.ptr);
        match *op {
            Op::V4Streams => launch!(k.streams, [nu, (hid as u32).div_ceil(256), 1], [256, 1, 1], s; p.slot(Slot::X), p.slot(Slot::S), hid),
            Op::V4Pre { f, base, scale, norm } => {
                launch!(k.hc_proj, [nu, ks, 1], [128, 1, 1], s; p.slot(Slot::S), p.small[f].ptr, v.part.ptr, hid, 24i32)?;
                launch!(k.hc_fin, [nu, 1, 1], [512, 1, 1], s; p.slot(Slot::S), v.part.ptr, ks as i32, p.small[base].ptr, p.small[scale].ptr, p.small[norm].ptr, p.slot(Slot::H), v.hcp.ptr, hid, eps, v.dims.hc_eps, v.dims.iters)?;
                self.mark("hyper-connections")
            }
            Op::V4Post => {
                launch!(k.hc_post, [nu, (hid as u32).div_ceil(256), 1], [256, 1, 1], s; p.slot(Slot::S), p.slot(Slot::O), v.hcp.ptr, hid)?;
                self.mark("hyper-connections")
            }
            Op::V4Head { f, base, scale, norm } => {
                launch!(k.hc_proj, [nu, ks, 1], [128, 1, 1], s; p.slot(Slot::S), p.small[f].ptr, v.part.ptr, hid, 4i32)?;
                launch!(k.hc_head, [nu, 1, 1], [512, 1, 1], s; p.slot(Slot::S), v.part.ptr, ks as i32, p.small[base].ptr, p.small[scale].ptr, p.small[norm].ptr, p.slot(Slot::H), hid, eps, v.dims.hc_eps)?;
                self.mark("hyper-connections")
            }
            Op::Fp8 { x, y, .. } if !v.dims.sim => {
                crate::dev::copy_d2d_async(p.slot(y), p.slot(x), n * p.d.width(x) * 2, s)?;
                self.mark("activations to FP8")
            }
            Op::Fp8 { x, y, block } => {
                let w = p.d.width(x);
                launch!(k.fp8, [nu, ((w / block) as u32).div_ceil(4), 1], [128, 1, 1], s; p.slot(x), p.slot(y), w as i32, block as i32)?;
                self.mark("activations to FP8")
            }
            Op::V4Q { rope: r, heads } => {
                let (c, sn) = rope(r);
                launch!(k.q_prep, [nu, heads as u32, 1], [128, 1, 1], s; p.slot(Slot::Qo), c, sn, pos, eps)?;
                self.mark("attention: q norm, rotary")
            }
            Op::V4Kv { norm, rope: r, ring } => {
                let (c, sn) = rope(r);
                let kv = p.slot(Slot::Kv) + self.rows_before(p.d.width(Slot::Kv));
                launch!(k.kv_prep, [nu, 1, 1], [128, 1, 1], s; kv, p.small[norm].ptr, c, sn, pos, self.cache(ring).0, self.ring_cap[ring] as i32, eps, v.dims.sim as i32)?;
                self.mark("attention: kv norm, rotary, cache")
            }
            Op::V4Comp { w, ape, norm, d, ratio, rope: r, cache, state, indexer } => {
                let wd = if ratio == 4 { 2 * d } else { d };
                let (c, sn) = rope(r);
                let (h, cache, st0, st1) = (p.slot(Slot::H) + self.rows_before(p.d.hidden), self.state(cache), self.state(state), self.state(state + 1));
                if n == 1 && hid % 256 == 0 && hid <= 8192 {
                    launch!(k.lin32_1, [((2 * wd) as u32).div_ceil(8), 1, 1], [256, 1, 1], s; h, p.small[w].ptr, p.slot(Slot::Cp), hid, (2 * wd) as i32)?;
                } else {
                    launch!(k.lin32, [((2 * wd) as u32).div_ceil(8), nu.div_ceil(8), 1], [256, 1, 1], s; h, p.small[w].ptr, p.slot(Slot::Cp), n as i32, hid, (2 * wd) as i32)?;
                }
                launch!(k.comp_pool, [(d as u32).div_ceil(128), 1, 1], [128, 1, 1], s; p.slot(Slot::Cp), p.small[ape].ptr, st0, st1, v.pooled.ptr, pos, n as i32, ratio as i32, d as i32, (ratio == 4) as i32)?;
                launch!(k.comp_out, [(n / ratio + 1) as u32, 1, 1], [128, 1, 1], s; v.pooled.ptr, p.small[norm].ptr, c, sn, pos, n as i32, cache, ratio as i32, d as i32, indexer as i32, eps, v.dims.sim as i32)?;
                self.mark(if indexer { "indexer: compressor" } else { "attention: compressor" })
            }
            Op::V4IdxQ { rope: r, heads } => {
                let (c, sn) = rope(r);
                launch!(k.idx_q, [nu, heads as u32, 1], [128, 1, 1], s; p.slot(Slot::Iq), c, sn, pos, v.dims.sim as i32)?;
                self.mark("indexer: q")
            }
            Op::V4IdxPick { cache, ratio, heads, topk, wscale } => {
                let smem = (heads * 129 * 4 + 8 * 64 * 4) as u32; // (the queries and weights, then v4_idx_score's warps' terms: at most 64 heads)
                let (iq, iw, entries) = (p.slot(Slot::Iq) + self.rows_before(heads * 128), p.slot(Slot::Iw) + self.rows_before(heads), self.state(cache));
                for r0 in (0..n).step_by(v.ra) {
                    let rn = v.ra.min(n - r0) as u32;
                    launch!(k.idx_score, [rn, (v.emax as u32).div_ceil(8), 1], [256, 1, 1], smem smem, s; iq + (r0 * heads * 128 * 2) as u64, iw + (r0 * heads * 2) as u64, wscale, entries, pos + (r0 * 4) as u64, v.isc.ptr, heads as i32, ratio as i32, v.emax as i32)?;
                    launch!(k.topk, [rn, 1, 1], [1024, 1, 1], s; v.isc.ptr, v.emax as i32, pos + (r0 * 4) as u64, ratio as i32, topk as i32, v.sel.ptr + ((row0 + r0) * topk * 4) as u64, v.cnt.ptr + ((row0 + r0) * 4) as u64)?;
                }
                self.mark("indexer: picks")
            }
            Op::V4Attn { ring, cache, ratio, sink, rope: r, heads, hpg } => {
                let (c, sn) = rope(r);
                let (kc, cap, window) = (self.cache(ring).0, self.ring_cap[ring] as i32, p.d.rings[ring].window as i32);
                let cache = cache.map_or(0, |x| self.state(x));
                let (topk, picked) = (v.dims.topk, ratio == 4);
                let scale = (512f32).powf(-0.5);
                for r0 in (0..n).step_by(v.ra) {
                    let rn = v.ra.min(n - r0) as u32;
                    let (pr, sel, cnt) = (
                        pos + (r0 * 4) as u64,
                        if picked { v.sel.ptr + ((row0 + r0) * topk * 4) as u64 } else { 0 },
                        if picked { v.cnt.ptr + ((row0 + r0) * 4) as u64 } else { 0 },
                    );
                    let q = p.slot(Slot::Qo) + self.rows_before(heads * 512) + (r0 * heads * 512 * 2) as u64;
                    launch!(k.attn_scores, [rn, (v.lmax as u32).div_ceil(32), (heads as u32).div_ceil(16)], [256, 1, 1], s; q, kc, cap, cache, sel, cnt, topk as i32, pr, v.sc.ptr, heads as i32, v.lmax as i32, window, ratio as i32, scale)?;
                    launch!(k.attn_soft, [rn, (heads as u32).div_ceil(8), 1], [256, 1, 1], s; v.sc.ptr, p.small[sink].ptr, v.den.ptr, pr, cnt, heads as i32, v.lmax as i32, window, ratio as i32)?;
                    // (kernels/dsv4.cu V4_AQ: 8 warps a head, a head a block)
                    launch!(k.attn_out, [rn, heads as u32, 1], [256, 1, 1], s; v.sc.ptr, v.den.ptr, kc, cap, cache, sel, cnt, topk as i32, pr, c, sn, p.slot(Slot::Att), rows as i32, heads as i32, hpg as i32, v.lmax as i32, window, ratio as i32, (row0 + r0) as i32)?;
                }
                self.mark("attention: attention")
            }
            Op::V4OLora { w, in_f8, groups, hpg, ol, y, fp8 } => {
                for g in 0..groups {
                    let (xg, yg) = (p.slot(Slot::Att) + (g * n * hpg * 512 * 2) as u64, p.slot(Slot::Mid) + (g * n * ol * 2) as u64);
                    if in_f8 {
                        p.f8[w + g].linear(&self.lib, xg, n, yg, self.ws.ptr, self.ws.bytes, self.done.ptr, s)?;
                    } else {
                        self.lin(&p.mats[w + g], xg, n, yg)?;
                    }
                }
                launch!(k.ogather, [nu, ((groups * ol / 128) as u32).div_ceil(4), 1], [128, 1, 1], s; p.slot(Slot::Mid), p.slot(y), n as i32, groups as i32, ol as i32, (fp8 && v.dims.sim) as i32)?;
                self.mark("attention: o low rank")
            }
            Op::V4OLoraStack { w, groups, ol, y, fp8 } => {
                // (the pairs (row t, group g) the groups' plan sorts: group g's rows of Att, then of Mid, one after another, as the per-group products read and write them; a prompt's chunk of
                // more than the product's 64 rows in parts, each part's rows of each group copied to ox, its products into oy, and back into each group's rows of Mid)
                let m = &p.f8[w];
                let (att, mid) = (p.slot(Slot::Att), p.slot(Slot::Mid));
                if n <= crate::f8::MOE_TOKENS {
                    self.lib.moe_way(v.gids.ptr, (n * groups) as i64, groups as i64, v.gplan.ptr, s)?;
                    m.moe(&self.lib, groups, att, v.gplan.ptr, n, groups, false, mid, self.ws.ptr, self.ws.bytes, self.done.ptr, s)?;
                } else {
                    for t0 in (0..n).step_by(crate::f8::MOE_TOKENS) {
                        let tn = (n - t0).min(crate::f8::MOE_TOKENS);
                        for g in 0..groups {
                            crate::dev::copy_d2d_async(v.ox.ptr + (g * tn * m.k * 2) as u64, att + ((g * n + t0) * m.k * 2) as u64, tn * m.k * 2, s)?;
                        }
                        self.lib.moe_way(v.gids.ptr, (tn * groups) as i64, groups as i64, v.gplan.ptr, s)?;
                        m.moe(&self.lib, groups, v.ox.ptr, v.gplan.ptr, tn, groups, false, v.oy.ptr, self.ws.ptr, self.ws.bytes, self.done.ptr, s)?;
                        for g in 0..groups {
                            crate::dev::copy_d2d_async(mid + ((g * n + t0) * ol * 2) as u64, v.oy.ptr + (g * tn * ol * 2) as u64, tn * ol * 2, s)?;
                        }
                    }
                }
                launch!(k.ogather, [nu, ((groups * ol / 128) as u32).div_ceil(4), 1], [128, 1, 1], s; p.slot(Slot::Mid), p.slot(y), n as i32, groups as i32, ol as i32, (fp8 && v.dims.sim) as i32)?;
                self.mark("attention: o low rank")
            }
            Op::V4Moe { block, hash } => {
                let (layer, _) = p.moe[block].resident().ok_or_else(|| Error("DeepSeek-V4's experts are not on the GPU".into()))?;
                let sc = self.moe_sc.get().ok_or_else(|| Error("no mixture-of-experts scratch".into()))?;
                layer.forward_release(
                    &self.lib,
                    sc,
                    s,
                    p.slot(Slot::H),
                    p.slot(Slot::Hq),
                    n,
                    p.slot(Slot::O),
                    hash.map(|h| (p.small[h].ptr, v.tok.ptr)),
                    v.ones.ptr,
                    v.dims.sim,
                )?;
                self.mark("mixture of experts")
            }
            _ => unreachable!(),
        }
    }

    /// Part `p`'s layers on the n rows (positions `pos`), then on the last `head` rows its post projection (a draft model's, into Hid), its LM head, the
    /// soft-cap and the argmax (its logits, tok, val a row each).
    unsafe fn forward(&self, p: &Part, n: usize, pos: (u64, u64), head: usize) -> Result<()> {
        let s = self.dev.stream;
        self.mark("start")?;
        let st = if std::ptr::eq(p, &self.big) { self.stage } else { Stage::whole(p.d.layers.len()) };
        // (the 12-bit pages' slots, lengths and first tokens of the pass's rows, once for every layer)
        if self.lossless.is_some() && std::ptr::eq(p, &self.big) {
            self.kv12_prepare(n, pos)?;
        }
        let head = if st.last { head } else { 0 };
        if st.first {
            self.op(p, 0, 0, &p.d.first, n, pos, head)?;
        }
        // (a development build's GLYD_OP_PROBE: after each op, the slots' first rows read back; the first op that leaves a slot not finite named)
        let probe = crate::tune!("GLYD_OP_PROBE").is_some();
        let mut seen = false;
        for (l, ops) in p.d.layers.iter().enumerate().take(st.layers.1).skip(st.layers.0) {
            // (a layer whose matrices are in the computer's memory: copied into the slot on this stream first, a node of the step's graph)
            if let Some(r) = &p.ram {
                r.fetch(l, s)?;
            }
            for (i, op) in ops.iter().enumerate() {
                // (a preparation whose keys and values go to 12-bit pages writes staging rows and no sequence's ring: the pass's rows at once, and one
                // write of them; the slots of each sequence's rows the pass made)
                if !self.parts.is_empty() && !rows::row_wise(op) {
                    if let Op::Attn(a) = op {
                        if a.writes && self.pages12(p, a.ring).is_some() {
                            self.attn_prep(p, l, i, a, n, pos, head)?;
                            self.prepped.set(true);
                        }
                    }
                    // (an operation on a sequence's cache or state: each sequence's rows by themselves, as a pass of that sequence alone runs them)
                    let mut r = Ok(());
                    for (k, part) in self.parts.iter().enumerate() {
                        self.at.set(Some(k));
                        self.start.set(part.start);
                        let off = 4 * part.row0 as u64;
                        r = self.op(p, l, i, op, part.rows, (pos.0 + off, pos.1 + off), head);
                        self.at.set(None);
                        if r.is_err() {
                            break;
                        }
                    }
                    self.prepped.set(false);
                    r?;
                    continue;
                }
                self.op(p, l, i, op, n, pos, head)?;
                if crate::hb_usage::recording() && std::ptr::eq(p, &self.big) {
                    self.record_routes(p, l, op, n)?;
                }
                if probe && !seen {
                    for (k, b) in p.slots.iter().enumerate() {
                        let w = p.d.widths[k].min(b.bytes / 2);
                        let mut v = vec![0u16; w];
                        copy_d2h_async(v.as_mut_ptr() as *mut _, b.ptr, w * 2, s)?;
                        self.dev.sync()?;
                        let bad = v.iter().filter(|&&h| !f32::from_bits((h as u32) << 16).is_finite()).count();
                        if bad > 0 {
                            eprintln!("layer {l} op {i} ({op:?}): slot {k} has {bad} of {w} not finite");
                            seen = true;
                        }
                    }
                }
            }
        }
        if head > 0 {
            // (a pass over several sequences: the rows the head reads, the last of each sequence that asks for a token, moved to the pass's last rows in order)
            let w = (p.d.hidden * 2) as u64;
            let moved: Vec<Slot> = [Slot::X, Slot::H].into_iter().chain(p.d.final_norm.map(|f| f.1)).collect();
            for (j, &src) in self.head_rows.iter().enumerate().rev() {
                let dst = n - self.head_rows.len() + j;
                if dst != src {
                    for &slot in &moved {
                        copy_d2d_async(p.slot(slot) + dst as u64 * w, p.slot(slot) + src as u64 * w, w as usize, s)?;
                    }
                }
            }
            let first = ((n - head) * p.d.hidden * 2) as u64;
            if let Some(post) = p.d.post {
                self.lin(&p.mats[post], p.slot(Slot::H) + first, head, p.slot(Slot::Hid))?;
            }
            let hl = match p.d.final_norm {
                Some((w, y, plus1)) => {
                    let (x, y, w, out, rows, hid) =
                        (p.slot(Slot::X) + first, p.slot(y) + first, p.small[w].ptr, p.slot(Slot::Hlast), head as u32, p.d.hidden as i32);
                    if plus1 {
                        self.k35.rmsnorm_add(s, x, y, w, out, rows, hid, p.d.eps)?;
                    } else {
                        self.kern.rmsnorm_add(s, x, y, w, out, rows, hid, p.d.eps)?;
                    }
                    p.slot(Slot::Hlast)
                }
                None => p.slot(Slot::H) + first,
            };
            self.lin(&p.mats[p.d.lm], hl, head, p.logits.ptr)?;
            if p.d.softcap > 0.0 && p.d.logit_mult > 0.0 {
                launch!(self.kmuse.softcap, [64, 1, 1], [256, 1, 1], s; p.logits.ptr, (head * p.d.vocab) as i32, p.d.logit_mult, p.d.softcap)?;
            } else if p.d.softcap > 0.0 {
                launch!(self.kg.softcap, [64, 1, 1], [256, 1, 1], s; p.logits.ptr, (head * p.d.vocab) as i32, p.d.softcap)?;
            }
            self.kern.argmax_rows(s, p.logits.ptr, p.d.vocab as i32, head as u32, p.tok.ptr, p.val.ptr)?;
            self.mark("head")?;
        }
        Ok(())
    }

    fn capture_graph(&mut self) -> Result<()> {
        self.st.write(0, &[0i32])?;
        let st = self.st.ptr;
        // SAFETY: live buffers of the engine; the stream is synchronized after.
        unsafe {
            if self.stage.first {
                self.embed(&[0u32])?;
            }
            self.forward(&self.big, 1, (st, st), 1)?;
        }
        self.dev.sync()?;
        // (--prof: the graph's events alone kept)
        if let Some(p) = &self.prof {
            p.borrow_mut().clear();
        }
        let g = self.dev.capture(|| {
            // SAFETY: as above.
            unsafe { self.forward(&self.big, 1, (st, st), 1) }
        })?;
        self.graph = Some(g);
        self.reset()
    }

    fn read_sample(&mut self) -> Result<Sample> {
        // SAFETY: the pinned buffer holds 4096 bytes; the copies complete at the synchronize.
        unsafe {
            copy_d2h_async(self.host.ptr, self.big.tok.ptr, 4, self.dev.stream)?;
            copy_d2h_async((self.host.ptr as *mut u8).add(8) as *mut _, self.big.val.ptr, 4, self.dev.stream)?;
            self.top.queue(&self.kern, &self.dev, self.big.logits.ptr, self.big.val.ptr)?;
        }
        self.dev.sync()?;
        // SAFETY: the pinned bytes were just written.
        let (token, logit) = unsafe { (*(self.host.ptr as *const i32), *((self.host.ptr as *const u8).add(8) as *const f32)) };
        Ok(Sample { token: token as u32, logit, logprob: None })
    }

    /// The slots a stage hands the next (the residual and what the next layer's first operation reads: H, D or O), in this order where the model has them.
    const IO_SLOTS: [Slot; 5] = [Slot::X, Slot::H, Slot::D, Slot::O, Slot::S];

    /// What a cut between two layers carries (docs/range-worker.md): the slots the next layer reads before it writes them, in this order, each `[rows, width]` bf16, one after the other. Qwen3.5's
    /// layers end with the MLP's down product into D and begin by adding it to the residual X (NormAdd; the final norm adds it the same way): the pair X, D, and nothing else is live. The other
    /// families' stages hand over IO_SLOTS, as the split across GPUs has (its proofs are on it).
    pub fn boundary(&self) -> Vec<(Slot, usize)> {
        let d = &self.big.d;
        // (Kimi-K3's layers begin with the depth mix of the completed blocks and the partial sum: those two, nothing else, are live)
        let slots: &[Slot] = match d.family {
            "Qwen3.5" => &[Slot::X, Slot::D],
            "Kimi-K3" => &[Slot::X, Slot::Blk],
            _ => &Self::IO_SLOTS,
        };
        slots.iter().map(|&s| (s, d.widths[s as usize])).filter(|&(_, w)| w > 0).collect()
    }

    fn io_words(&self, n: usize) -> usize {
        self.boundary().iter().map(|&(_, w)| n * w).sum()
    }

    /// What each layer costs on this GPU: its matrices' bytes, the keys and values a token adds to its rings (bf16), and its per-sequence states (zeroed at a new sequence); zeros for another
    /// stage's layer. Then the embedding's and the LM head's bytes (0 where another stage holds them) and the small tensors' total (the norms, the linear attention's parameters, all layers').
    pub fn layer_costs(&self) -> (Vec<(usize, usize, usize)>, usize, usize, usize) {
        let p = &self.big;
        let (a, b) = self.stage.layers;
        let layers =
            p.d.layers
                .iter()
                .enumerate()
                .map(|(l, ops)| {
                    if !(a..b).contains(&l) {
                        return (0, 0, 0);
                    }
                    let w = ops.iter().filter_map(Op::mat).map(|k| p.mats[k].bytes()).sum();
                    let mut rings: Vec<usize> = ops.iter().filter_map(Op::ring).collect();
                    rings.dedup();
                    let mut states: Vec<usize> = ops.iter().flat_map(Op::states).collect();
                    states.sort_unstable();
                    states.dedup();
                    (w, rings.iter().map(|&r| 2 * p.d.rings[r].nk * p.d.rings[r].d * 2).sum(), states.iter().map(|&s| self.states[s].bytes).sum())
                })
                .collect();
        let embed = if self.stage.first { p.mats[p.d.embed].bytes() } else { 0 };
        let head = if self.stage.last && p.d.lm != p.d.embed { p.mats[p.d.lm].bytes() } else { 0 };
        (layers, embed, head, p.small.iter().map(|b| b.bytes).sum())
    }

    /// The per-sequence states this stage holds, in bytes.
    pub fn state_bytes(&self) -> usize {
        self.states.iter().map(|b| b.bytes).sum()
    }

    /// n rows through this stage's layers, the first at position `at`: the tokens' embedding where it is the first stage, else the previous stage's slots
    /// from `io`; this stage's slots into `io` where it is not the last; the head on the last `head` rows where it is the last (`head_out` reads them).
    pub fn stage_rows(&mut self, tokens: &[u32], at: usize, io: &mut Vec<u16>, head: usize) -> Result<()> {
        let n = tokens.len();
        if n == 0 || n > self.chunk || at + n + 1 > self.ctx {
            return err(format!("{n} rows at {at}: a chunk holds {} and the cache {}", self.chunk, self.ctx));
        }
        if self.stage.last && (head > n || head > self.head_room()) {
            return err(format!("the head on {head} rows of {n}: this stage has room for {} (set_head_rows)", self.head_room()));
        }
        if !self.stage.first && io.len() != self.io_words(n) {
            return err(format!("{} words from the previous stage, where {n} rows take {}", io.len(), self.io_words(n)));
        }
        // (a chunk past a verify pass: nothing to go back in)
        self.pending = None;
        self.dev.sync()?;
        let one = n == 1 && self.graph.is_some() && !self.verifying.get();
        if one {
            self.st.write(0, &[at as i32])?;
        } else {
            self.pos.write(0, &(0..n).map(|i| (at + i) as i32).collect::<Vec<i32>>())?;
        }
        let pp = if one { self.st.ptr } else { self.pos.ptr };
        self.start.set(at);
        // SAFETY: live buffers of the engine, sized for a chunk; io holds the slots' rows (checked above); the stream orders the copies and launches, and is synchronized before io is read.
        unsafe {
            if self.stage.first {
                self.embed(tokens)?;
            } else {
                let mut off = 0usize;
                for (sl, w) in self.boundary() {
                    copy_h2d_async(self.big.slot(sl), io[off..].as_ptr() as *const _, n * w * 2, self.dev.stream)?;
                    off += n * w;
                }
            }
            match &self.graph {
                Some(g) if one => g.launch(self.dev.stream)?,
                _ => self.forward(&self.big, n, (pp, pp), head)?,
            }
            // (a development build's GLYD_STAGE_DUMP: the last stage's boundary slots too, as they are before the head)
            let dump = crate::tune!("GLYD_STAGE_DUMP").is_some();
            if !self.stage.last || dump {
                io.resize(self.io_words(n) + if dump { n * self.big.d.hidden } else { 0 }, 0);
                let mut off = 0usize;
                for (sl, w) in self.boundary() {
                    copy_d2h_async(io[off..].as_mut_ptr() as *mut _, self.big.slot(sl), n * w * 2, self.dev.stream)?;
                    off += n * w;
                }
                // (and H, the head's input)
                if dump {
                    copy_d2h_async(io[off..].as_mut_ptr() as *mut _, self.big.slot(Slot::H), n * self.big.d.hidden * 2, self.dev.stream)?;
                }
            }
        }
        self.dev.sync()?;
        self.n = at + n;
        Ok(())
    }

    /// A verify pass (drafted tokens checked, docs/range-worker.md): `stage_rows` on 1 to VERIFY_ROWS rows, each row as a decode step runs it (the delta rule one row
    /// after another, the decode step's attention, products of at most 8 rows), so each row's outputs are the step's at its position, bit for bit; and what `rewind`
    /// needs kept (the states before the pass, each linear-attention layer's inputs). One row is a plain step (the graph): `rewind` keeps it. Qwen3.5's stages.
    pub fn verify_rows(&mut self, tokens: &[u32], at: usize, io: &mut Vec<u16>, head: usize) -> Result<()> {
        let n = tokens.len();
        if self.big.d.family != "Qwen3.5" {
            return err(format!("a verify pass on {}: Qwen3.5's stages only", self.big.d.family));
        }
        if n == 0 || n > VERIFY_ROWS {
            return err(format!("a verify pass of {n} rows: 1 to {VERIFY_ROWS}"));
        }
        // (the copy below is this pass's: an earlier one is gone)
        self.pending = None;
        if n > 1 {
            if self.verify.is_none() {
                self.verify = Some(self.verify_room()?);
            }
            if self.vpart.is_none() {
                let (heads, d) = self.big.d.heads();
                self.vpart = Some(Buf::new(VERIFY_ROWS * heads.max(1) * attn::SPLITS * (d + 2) * 4)?);
            }
            self.keep_states()?;
            self.verifying.set(true);
        }
        let r = self.stage_rows(tokens, at, io, head);
        self.verifying.set(false);
        r?;
        self.pending = Some((at, n));
        Ok(())
    }

    /// The per-sequence states copied aside (a verify pass's first step; `rewind` puts them back). Queued on the stream.
    pub fn keep_states(&self) -> Result<()> {
        let v = self.verify.as_ref().ok_or_else(|| Error("no room for a verify pass yet (verify_rows makes it)".into()))?;
        for (o, b) in v.snap.iter().zip(&self.states) {
            // SAFETY: two live buffers of b's size; the stream orders the copy before the pass.
            unsafe { copy_d2d_async(o.ptr, b.ptr, b.bytes, self.dev.stream)? };
        }
        Ok(())
    }

    /// Of the last pass, a verify pass of `rows` rows at `start`, the first `keep` stay (1 to rows): each linear-attention layer's states back to before the pass, then
    /// its convolution and delta rule again over the kept rows' inputs (one row after another, as the pass ran them: the states a row's step leaves); the keys and
    /// values past them are written over by the next chunk's rows before any row reads them. The sequence then holds start + keep tokens.
    pub fn rewind(&mut self, keep: usize) -> Result<()> {
        let Some((start, rows)) = self.pending else { return err("nothing to go back in: the last chunk was not a verify pass") };
        if keep == 0 || keep > rows {
            return err(format!("keep {keep} rows of the verify pass's {rows}: 1 to {rows}"));
        }
        self.pending = None;
        if keep < rows {
            let (p, s) = (&self.big, self.dev.stream);
            let v = self.verify.as_ref().ok_or_else(|| Error("no room for a verify pass".into()))?;
            let ba = p.d.width(Slot::Ba);
            // SAFETY: live buffers of the engine (the kept rows: VERIFY_ROWS of each, `keep` of them read); the stream orders the copies and the kernels.
            unsafe {
                for (b, o) in self.states.iter().zip(&v.snap) {
                    copy_d2d_async(b.ptr, o.ptr, b.bytes, s)?;
                }
                for (l, k) in v.keep.iter().enumerate() {
                    let Some(k) = k else { continue };
                    for op in &p.d.layers[l] {
                        match *op {
                            Op::GdnConv { w, state, cd, row } => self.k35.conv(
                                s,
                                k.ptr + (VERIFY_ROWS * ba * 2) as u64,
                                row as i32,
                                p.small[w].ptr,
                                self.states[state].ptr,
                                p.slot(Slot::Xc),
                                keep as i32,
                                cd as i32,
                            )?,
                            Op::GdnDelta { a_log, dt_bias, state, cd, nk, nv } => self.k35.delta(
                                s,
                                false,
                                p.slot(Slot::Xc),
                                cd as i32,
                                k.ptr,
                                ba as i32,
                                p.small[a_log].ptr,
                                p.small[dt_bias].ptr,
                                self.states[state].ptr,
                                p.slot(Slot::Core),
                                keep as i32,
                                nk as i32,
                                nv as i32,
                            )?,
                            _ => {}
                        }
                    }
                }
            }
            self.dev.sync()?;
        }
        self.n = start + keep;
        Ok(())
    }

    /// The last chunk's (start, rows) where it was a verify pass `rewind` can go back in.
    pub fn verified(&self) -> Option<(usize, usize)> {
        self.pending
    }

    fn verify_room(&self) -> Result<Verify> {
        let d = &self.big.d;
        let ba = d.width(Slot::Ba);
        let snap = self.states.iter().map(|b| Buf::new(b.bytes)).collect::<Result<_>>()?;
        let mine = self.stage.layers.0..self.stage.layers.1;
        let keep = d
            .layers
            .iter()
            .enumerate()
            .map(|(l, ops)| match ops.iter().find_map(|op| if let Op::GdnConv { row, .. } = op { Some(*row) } else { None }) {
                Some(row) if mine.contains(&l) => Buf::new(VERIFY_ROWS * (ba + row) * 2).map(Some),
                _ => Ok(None),
            })
            .collect::<Result<_>>()?;
        Ok(Verify { snap, keep })
    }

    /// Room for `k` more sequences beside the one in use (each its own keys and values for the context, its states, its graph captured on them; the
    /// activations and workspaces stay shared): `select` picks the one the next passes run on. Qwen3.5's stages; after `set_head_rows`.
    pub fn add_seqs(&mut self, k: usize) -> Result<()> {
        if k > 0
            && (!matches!(self.big.d.family, "Qwen3.5" | "Gemma 4" | "Muse Spark" | "Kimi-K3" | "GLM-5.3-Flash" | "Nemotron-H" | "DeepSeek-V4")
                || self.draft.is_some())
        {
            return err(format!(
                "more sequences than one on {}: Qwen3.5's, Gemma 4's, Muse Spark's, Kimi-K3's, GLM-5.3-Flash's, Nemotron-H's and DeepSeek-V4's only, with no draft model",
                self.big.d.family
            ));
        }
        for _ in 0..k {
            let states = self.states.iter().map(|b| Buf::zeroed(b.bytes)).collect::<Result<_>>()?;
            // (paged: every sequence's keys and values are in the pools, kc and vc; its own are its block tables)
            let room = if self.paged.is_some() { 4 } else { self.kc.bytes };
            self.seqs.push(Some(Seq { kc: Buf::zeroed(room)?, vc: Buf::zeroed(room)?, states, n: 0, pending: None, verify: None, graph: None }));
            let (back, graph) = (self.seq, self.graph.is_some());
            if graph {
                // (its own graph: the one-row step captured on its buffers)
                self.select(self.seqs.len() - 1)?;
                self.capture_graph()?;
                self.select(back)?;
            }
        }
        Ok(())
    }

    /// The sequences this engine holds (1, and what `add_seqs` added) and the one the passes run on.
    pub fn seqs(&self) -> (usize, usize) {
        (self.seqs.len(), self.seq)
    }

    /// Sequence `i`'s buffers, place and graph swapped in (nothing copied): the passes, `reset`, `rewind` and `verified` are its until another is selected.
    pub fn select(&mut self, i: usize) -> Result<()> {
        if i == self.seq {
            return Ok(());
        }
        let mut s = self.seqs.get_mut(i).and_then(Option::take).ok_or_else(|| Error(format!("sequence {i}: this engine holds {}", self.seqs.len())))?;
        self.dev.sync()?;
        if self.paged.is_none() {
            std::mem::swap(&mut self.kc, &mut s.kc);
            std::mem::swap(&mut self.vc, &mut s.vc);
        }
        std::mem::swap(&mut self.states, &mut s.states);
        std::mem::swap(&mut self.n, &mut s.n);
        std::mem::swap(&mut self.pending, &mut s.pending);
        std::mem::swap(&mut self.verify, &mut s.verify);
        std::mem::swap(&mut self.graph, &mut s.graph);
        self.seqs[self.seq] = Some(s);
        self.seq = i;
        Ok(())
    }

    /// The token the last pass chose (the last stage's).
    pub fn sample(&mut self) -> Result<Sample> {
        self.read_sample()
    }

    /// The rows the head has room for (its logits, tokens and values): 1, or what `set_head_rows` made.
    pub fn head_room(&self) -> usize {
        self.big.tok.bytes / 4
    }

    /// Room for the head on the last `rows` rows of a chunk (a last stage that checks drafted tokens: a greedy token a row): the logits, tokens and values
    /// that many rows, the LM head's workspace for them, and the graph captured again on the new buffers. At most 64 rows: past that a product decodes
    /// its matrix first, and `build` makes no room to decode the head's. Before a session: the states are zeroed.
    pub fn set_head_rows(&mut self, rows: usize) -> Result<()> {
        if rows <= self.head_room() {
            return Ok(());
        }
        if self.seqs.len() > 1 {
            return err("set_head_rows before add_seqs (each sequence's graph is captured on the head's buffers)");
        }
        if rows > self.chunk.min(64) {
            return err(format!("the head on {rows} rows: at most {} (the chunk, and 64)", self.chunk.min(64)));
        }
        self.dev.sync()?;
        self.big.logits = Buf::new(self.big.d.vocab * rows * 2)?;
        (self.big.tok, self.big.val) = (Buf::zeroed(4 * rows)?, Buf::zeroed(4 * rows)?);
        if let Some(lm) = self.big.mats[self.big.d.lm].packed_opt() {
            let mut need = 0;
            for m in 1..=rows as i64 {
                need = need.max(self.lib.linear_workspace(lm.pack.is_twelve(), lm.pack.rows, lm.pack.cols, m)?);
            }
            if need > self.ws.bytes {
                self.ws = Buf::zeroed(need)?;
            }
        }
        if self.graph.take().is_some() {
            self.capture_graph()?;
        }
        self.reset()
    }

    /// The last pass's head on its last `rows` rows (`stage_rows`' head): each row's greedy token and its logit, and where `words` the rows' logits as
    /// bf16 words, row after row.
    pub fn head_out(&self, rows: usize, words: bool) -> Result<HeadOut> {
        self.dev.sync()?;
        let (mut tok, mut val) = (vec![0i32; rows], vec![0f32; rows]);
        self.big.tok.read(0, &mut tok)?;
        self.big.val.read(0, &mut val)?;
        let mut w = Vec::new();
        if words {
            w = vec![0u16; rows * self.big.d.vocab];
            self.big.logits.read(0, &mut w)?;
        }
        Ok((tok.iter().zip(&val).map(|(&t, &v)| (t as u32, v)).collect(), w))
    }

    /// The log-probability of the last pass's `logit` (its largest, as `Sample::logit`).
    pub fn logprob(&self, logit: f32) -> Result<f32> {
        // SAFETY: live buffers; the lse is read after the synchronize.
        unsafe { self.kern.logsumexp(self.dev.stream, self.big.logits.ptr, self.big.d.vocab as i32, logit, self.lse.ptr)? };
        self.dev.sync()?;
        let mut l = [0f32; 1];
        self.lse.read(0, &mut l)?;
        Ok(logit - l[0])
    }

    /// The prompt after the tokens already in the cache, in chunks; the token that follows it.
    pub fn prefill(&mut self, tokens: &[u32]) -> Result<Sample> {
        if tokens.is_empty() {
            return err("an empty prompt");
        }
        if self.n + tokens.len() + 1 > self.ctx {
            return err(format!("{} tokens and {} more, where the cache holds {}", self.n, tokens.len(), self.ctx));
        }
        let mut at = 0;
        while at < tokens.len() {
            let n = self.chunk.min(tokens.len() - at);
            self.hold_blocks(self.n + n)?;
            let pos: Vec<i32> = (0..n).map(|i| (self.n + i) as i32).collect();
            self.dev.sync()?;
            self.pos.write(0, &pos)?;
            let last = at + n == tokens.len();
            let pp = self.pos.ptr;
            self.start.set(self.n);
            // SAFETY: live buffers of the engine, sized for a chunk.
            unsafe {
                self.embed(&tokens[at..at + n])?;
                self.forward(&self.big, n, (pp, pp), last as usize)?;
                if let (true, Some(dr)) = (last, &self.draft) {
                    // (the draft model's first input: the hidden state of the prompt's last position)
                    self.kern.copy_row(self.dev.stream, self.big.slot(Slot::H), n as i32 - 1, self.big.d.hidden as i32, dr.p.slot(Slot::Hid))?;
                }
            }
            self.n += n;
            at += n;
        }
        // (a pass outside the scheduler, which would take a lost step back: an error)
        self.kv12_check()?;
        self.read_sample()
    }

    /// The selected sequence's blocks grown to hold its tokens before `upto`, where its keys and values are paged: a pass outside the scheduler (`reply`,
    /// `prefill`, `step`) wrote every token to block 0, which a table names until it grows, so a paged executor's replies were not the model's.
    fn hold_blocks(&mut self, upto: usize) -> Result<()> {
        if self.paged.is_some() && !crate::sched::ModelStep::grow(self, self.seq as i32, upto)? {
            return err(format!("the paged cache has no block left for token {upto} of the sequence"));
        }
        Ok(())
    }

    /// One token through the model (the graph where there is one); the token after it.
    pub fn step(&mut self, token: u32) -> Result<Sample> {
        if self.n + 1 > self.ctx {
            return err("the cache is full");
        }
        self.hold_blocks(self.n + 1)?;
        self.dev.sync()?;
        self.st.write(0, &[self.n as i32])?;
        let st = self.st.ptr;
        // SAFETY: live buffers; the graph was captured on them.
        unsafe {
            self.embed(&[token])?;
            match &self.graph {
                Some(g) => g.launch(self.dev.stream)?,
                None => self.forward(&self.big, 1, (st, st), 1)?,
            }
        }
        self.n += 1;
        self.read_sample()
    }

    /// One round of drafting: from `last` (the token after the cache's, not in it yet) the draft model proposes k tokens, then the big model reads `last` and
    /// them in one pass and keeps the drafts up to its first disagreement and its own token there (greedy), or checks each by `sampler`. The tokens returned
    /// (1 to k + 1) are what steps would give one by one; the cache then holds `last` and the kept drafts.
    pub fn spec_round(&mut self, last: u32, sampler: Option<&mut crate::sampler::Sampler>) -> Result<Vec<u32>> {
        let Some(dr) = self.draft.take() else { return err("no draft model") };
        let r = self.spec_round_with(&dr, last, sampler);
        self.draft = Some(dr);
        r
    }

    fn spec_round_with(&mut self, dr: &Draft, last: u32, sampler: Option<&mut crate::sampler::Sampler>) -> Result<Vec<u32>> {
        let n0 = self.n;
        let k = dr.k.min(self.ctx.saturating_sub(n0 + 1));
        if n0 == 0 {
            return err("spec_round before a prompt");
        }
        let mut seq = vec![last];
        self.dev.sync()?;
        dr.pos.write(0, &[n0 as i32, n0 as i32 - 1])?;
        for _ in 0..k {
            let t = *seq.last().expect("a token");
            // SAFETY: live buffers of the engine and the draft model; the token is read back after the synchronize.
            unsafe {
                self.embed(&[t])?;
                self.forward(&dr.p, 1, (dr.pos.ptr, dr.pos.ptr + 4), 1)?;
                copy_d2h_async(self.host.ptr, dr.p.tok.ptr, 4, self.dev.stream)?;
            }
            self.dev.sync()?;
            // SAFETY: the pinned bytes were just written.
            seq.push(unsafe { *(self.host.ptr as *const i32) } as u32);
        }
        let n = seq.len();
        let pos: Vec<i32> = (0..n).map(|i| (n0 + i) as i32).collect();
        self.pos.write(0, &pos)?;
        let pp = self.pos.ptr;
        // SAFETY: live buffers sized for k + 1 rows; the tokens are read after the synchronize.
        unsafe {
            self.embed(&seq)?;
            self.forward(&self.big, n, (pp, pp), n)?;
            copy_d2h_async(self.host.ptr, self.big.tok.ptr, 4 * n, self.dev.stream)?;
        }
        self.dev.sync()?;
        // SAFETY: the pinned buffer holds n ints, just written.
        let big: Vec<u32> = unsafe { std::slice::from_raw_parts(self.host.ptr as *const i32, n) }.iter().map(|&t| t as u32).collect();
        let (out, kept) = match sampler {
            None => {
                let kept = (0..k).take_while(|&i| seq[i + 1] == big[i]).count();
                (big[..=kept].to_vec(), kept)
            }
            Some(sm) => {
                let v = self.big.d.vocab;
                let mut words = vec![0u16; n * v];
                self.big.logits.read(0, &mut words)?;
                let row = |i: usize| -> Vec<f32> { words[i * v..(i + 1) * v].iter().map(|&w| f32::from_bits((w as u32) << 16)).collect() };
                let mut out = Vec::with_capacity(n);
                for i in 0..=k {
                    if i == k {
                        out.push(sm.sample(&row(i)));
                        break;
                    }
                    let (t, same) = sm.check_draft(&row(i), seq[i + 1]);
                    out.push(t);
                    if !same {
                        break;
                    }
                }
                let kept = out.len() - 1;
                (out, kept)
            }
        };
        // SAFETY: h holds the n rows' final hidden states.
        unsafe { self.kern.copy_row(self.dev.stream, self.big.slot(Slot::H), kept as i32, self.big.d.hidden as i32, dr.p.slot(Slot::Hid))? };
        self.n = n0 + kept + 1;
        Ok(out)
    }
}

impl crate::conv::Gen for Dyn {
    fn capacity(&self) -> usize {
        self.ctx
    }

    fn as_step(&mut self, users: usize) -> Result<Option<(usize, &mut dyn crate::sched::ModelStep)>> {
        Ok(self.serve_rows(users)?.map(|k| (k, self as &mut dyn crate::sched::ModelStep)))
    }

    fn share_memory(&mut self, part: f64) {
        self.share = part.clamp(0.0, 1.0);
    }

    /// The weights (a draft model's too), the rings of keys and values for the whole context, and the states; all made in `build` and kept as they are while it runs.
    fn held(&self) -> Option<crate::dev::Held> {
        let state_type = match (self.big.d.family, self.states.is_empty()) {
            (_, true) => "none",
            // (qwen35.rs: the convolution's state two bytes a value, the delta rule's four)
            ("Qwen3.5", _) => "fp32 (the delta rule's), bf16 (the convolution's)",
            _ => "not reported",
        };
        // (every sequence's: they are alike, `add_seqs`)
        let k = self.seqs.len() as u64;
        Some(crate::dev::Held {
            weights: self.bytes as u64,
            kv: if self.paged.is_some() { (self.kc.bytes + self.vc.bytes) as u64 } else { k * (self.kc.bytes + self.vc.bytes) as u64 },
            state: k * self.states.iter().map(|b| b.bytes as u64).sum::<u64>(),
            kv_type: "bf16",
            state_type,
        })
    }

    fn reply(&mut self, prompt: &[u32], o: &crate::model::Reply, each: &mut dyn FnMut(u32) -> bool) -> Result<(Vec<u32>, crate::model::Finish)> {
        use crate::model::Finish;
        if prompt.is_empty() || prompt.len() + 1 > self.ctx {
            return err(format!("{} prompt tokens, where the context holds {}", prompt.len(), self.ctx));
        }
        self.reset()?;
        let greedy = o.params.greedy();
        let mut sampler = crate::sampler::Sampler::new(o.params.clone(), prompt);
        self.top.ask = sampler.top_need();
        self.top.mask = sampler.mask_handle();
        let mut s = self.prefill(prompt)?;
        let mut out = Vec::new();
        let eos = self.big.d.eos.clone();
        if self.drafting() {
            // drafted: the big model's tokens a round at a time (the last of a round is not in the cache yet: the next round starts from it)
            let first = if greedy { s.token } else { sampler.sample_with(self.top.read().as_ref(), || self.logits())? };
            let mut pending = std::collections::VecDeque::from([first]);
            while let Some(tok) = pending.pop_front() {
                if !o.ignore_eos && (eos.contains(&(tok as i64)) || o.stop_ids.contains(&tok)) {
                    return Ok((out, Finish::Stop));
                }
                out.push(tok);
                if !each(tok) {
                    return Ok((out, Finish::Stopped));
                }
                if out.len() >= o.max_new || self.n + 1 > self.ctx {
                    return Ok((out, Finish::Length));
                }
                if pending.is_empty() {
                    pending.extend(self.spec_round(tok, if greedy { None } else { Some(&mut sampler) })?);
                }
            }
        }
        loop {
            let tok = if greedy { s.token } else { sampler.sample_with(self.top.read().as_ref(), || self.logits())? };
            if !o.ignore_eos && (eos.contains(&(tok as i64)) || o.stop_ids.contains(&tok)) {
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
            self.top.mask = sampler.mask_handle();
            s = self.step(tok)?;
        }
    }
}

/// `prodcheck [--rows 163840] [--cols 7168] [--m 1]` (development): a random bf16 matrix of that shape packed as a model's are (the layout's), its product with `m`
/// random rows on the GPU against the CPU's at the first rows, the middle and the last (Kimi-K3's head: 163840 x 7168, past 2^30 values).
#[cfg(feature = "dev")]
pub fn prodcheck(a: &crate::args::Args) -> Result<()> {
    let (rows, cols, m) = (a.num("rows", 163840usize)?, a.num("cols", 7168usize)?, a.num("m", 1usize)?);
    let layout = a.layout()?.unwrap_or(Layout::Tiered);
    let dev = Dev::new(0)?;
    let lib = Lib::find()?;
    let mut seed = 0x2545_f491_4f6c_dd1du64;
    let mut rnd = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    let bf = |v: f32| (v.to_bits() >> 16) as u16;
    // (values about 0.02, as a head's)
    // (--from DIR --name T: a checkpoint's matrix T instead)
    let (w, rows, cols) = match (a.flag("from"), a.flag("name")) {
        (Some(d), Some(t)) => {
            let ck = Checkpoint::open(Path::new(d)).map_err(|e| Error(e.to_string()))?;
            let (w, sh) = kimi::words(&ck, t)?;
            (w, sh[0] as usize, sh[1] as usize)
        }
        _ => ((0..rows * cols).map(|_| bf(((rnd() >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.04)).collect(), rows, cols),
    };
    let x: Vec<u16> = (0..m * cols).map(|_| bf(((rnd() >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 4.0)).collect();
    let pk = upload_packed(crate::weights::pack_words(&w, rows, cols, layout, "prodcheck")?)?;
    let (xb, yb) = (Buf::new(m * cols * 2)?, Buf::zeroed(m * rows * 2)?);
    xb.write(0, &x)?;
    let need = lib.linear_workspace(pk.pack.is_twelve(), pk.pack.rows, pk.pack.cols, m as i64)?;
    let (ws, done) = (Buf::zeroed(need.max(16))?, Buf::zeroed(4 << 18)?);
    // SAFETY: live buffers of the sizes the product reads and writes.
    unsafe { lib.linear(&pk.pack, xb.ptr, m as i64, yb.ptr, ws.ptr, ws.bytes, done.ptr, dev.stream)? };
    dev.sync()?;
    let mut y = vec![0u16; m * rows];
    yb.read(0, &mut y)?;
    let f = |h: u16| f32::from_bits((h as u32) << 16);
    let (mut worst, mut bad) = (0f32, 0usize);
    let picks: Vec<usize> = (0..4).chain(rows / 2..rows / 2 + 4).chain(rows - 4..rows).chain((1..=8).map(|k| rows * k / 9)).collect();
    for i in 0..m {
        for &r in &picks {
            let want: f64 = (0..cols).map(|c| f(w[r * cols + c]) as f64 * f(x[i * cols + c]) as f64).sum();
            let got = f(y[i * rows + r]);
            bad += !got.is_finite() as usize;
            worst = worst.max((got as f64 - want).abs() as f32 / (want.abs() as f32).max(0.05));
        }
    }
    let nonfinite = y.iter().filter(|&&h| !f(h).is_finite()).count();
    println!("{rows} x {cols} ({:.2} GB, layout {layout:?}) at {m} rows: {nonfinite} of {} outputs not finite; at {} rows checked {bad} not finite, the largest relative difference {worst:.4}", (rows * cols * 2) as f64 / 1e9, y.len(), picks.len());
    Ok(())
}

/// `prodbench MODEL [--layers 12] [--reps 20]` (development, GLY-327): a packed model's products at one row (as a decode step runs them), timed on its own
/// matrices: the first `layers` layers' (a glyd-v7 directory's packs as saved, penguin included), each kind of matrix in turn over those layers back to back,
/// so the L2 holds none of them; GB/s against the card's copy rate. Gemma 4's description names the matrices.
#[cfg(feature = "dev")]
pub fn prodbench(a: &crate::args::Args) -> Result<()> {
    use crate::dev::Timer;
    let dir = Path::new(a.need(1, "MODEL")?);
    let c = crate::gemma4::ConfigG4::load(dir)?;
    let d = gemma4::lowered(dir, &c)?;
    let (layers, reps) = (a.num("layers", 12usize)?.min(d.layers.len()), a.num("reps", 20usize)?);
    let dev = Dev::new(0)?;
    let lib = Lib::find()?;
    let skinny = crate::skinny::SkinnyKern::load(&dev)?;
    let ck = Checkpoint::open(dir).map_err(|e| Error(e.to_string()))?;
    let v7 = crate::weights::v7_manifest(dir)?.ok_or_else(|| Error(format!("{}: not a glyd-v7 directory", dir.display())))?;
    let read = |k: usize| -> Result<PackBuf> {
        let it = &d.mats[k];
        let key = it.members[0].trim_end_matches(".weight");
        let pk = match &v7 {
            (Layout::Tiered, m) => crate::weights::read_lossless(&ck, m, key, it.rows, it.cols)?,
            (l, m) => crate::weights::read_v7(&ck, m, *l, &[(key, it.rows)], it.cols)?,
        };
        upload_packed(pk)
    };
    let gbs = crate::report::bandwidth(&dev, 2, 20)?;
    println!("{}: copy {gbs:.1} GB/s; {} layout {:?}, the first {layers} layers' matrices, one row", dev.name, c.layers, v7.0);
    let t = Timer::new()?;
    for (kind, name) in [(0usize, "q/k/v"), (1, "o"), (2, "gate_up"), (3, "down")] {
        let mats: Vec<PackBuf> = (0..layers).map(|l| read(4 * l + kind)).collect::<Result<_>>()?;
        let bytes: usize = mats.iter().map(|m| m.bytes).sum();
        let (rows, cols) = (mats.iter().map(|m| m.pack.rows).max().unwrap_or(0) as usize, mats.iter().map(|m| m.pack.cols).max().unwrap_or(0) as usize);
        let x = Buf::zeroed(cols * 2)?;
        let y = Buf::zeroed(rows * 2)?;
        let need = mats
            .iter()
            .map(|m| lib.linear_workspace(m.pack.is_twelve(), m.pack.rows, m.pack.cols, 1))
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .max()
            .unwrap_or(16);
        let (ws, done) = (Buf::zeroed(need.max(16))?, Buf::zeroed(4 << 18)?);
        let run = |m: &PackBuf| -> Result<()> {
            // SAFETY: live buffers of the sizes the product reads and writes.
            unsafe {
                if crate::skinny::SkinnyKern::takes(&m.pack) && m.pack.rows >= 2048 {
                    skinny.gemv(dev.stream, &m.pack, x.ptr, y.ptr, 1)
                } else {
                    lib.linear(&m.pack, x.ptr, 1, y.ptr, ws.ptr, ws.bytes, done.ptr, dev.stream)
                }
            }
        };
        for m in &mats {
            run(m)?;
        }
        dev.sync()?;
        t.start(dev.stream)?;
        for _ in 0..reps {
            for m in &mats {
                run(m)?;
            }
        }
        let us = t.stop(dev.stream)? as f64 * 1e3 / (reps * layers) as f64;
        let rate = bytes as f64 / layers as f64 / us / 1e3;
        println!(
            "{name:>8} {rows:>6} x {cols:<6} {:.1} MB  {us:7.1} us  {rate:6.1} GB/s  {:5.1}% of the copy rate",
            bytes as f64 / layers as f64 / 1e6,
            100.0 * rate / gbs
        );
    }
    Ok(())
}
