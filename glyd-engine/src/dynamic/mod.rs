//! The dynamic engine (GLY-326, research/dynamic-engine/DESIGN.md): a model is a description (`ModelDesc`: its matrices, its small tensors, its
//! rotary tables, its key/value rings and its layers as lists of operations), and one executor (`Dyn`) runs any description on the shared kernels.
//! A family adds a description (`gemma4.rs`), not an engine; an operation is added once, when a family needs one no other has.

pub mod attn;
mod comp;
pub mod deepseek_v4;
pub mod dense;
pub mod experts;
pub mod gemma4;
pub mod glm;
pub mod kimi;
#[cfg(feature = "dev")]
pub mod kindcheck;
pub mod kv12;
mod kvpages;
#[cfg(feature = "dev")]
pub use kvpages::check as kvpages_check;
#[cfg(feature = "dev")]
pub(crate) use kvpages::room_of as kv12_room;
pub mod lower;
pub mod lower_comp;
pub mod mla;
pub mod muse;
pub mod nemotron;
pub mod ngram;
pub mod pages;
mod picked;
mod pieces;
pub mod prof;
pub mod qwen35;
pub mod qwen4x;
pub mod range;
pub mod residual;
mod rows;
pub(crate) mod snaps;
pub mod split;
mod swap;
pub mod tp;
pub mod tp_cache;

use crate::dev::{copy_d2d_async, copy_d2h_async, copy_h2d_async, Buf, Dev, Ev, Graph, Pinned};
use crate::ffi::{err, Error, Lib, Result};
use crate::gemma4::{rotary, EMBED_GROUPS};
use crate::gpu_ops::{Gemm, GpuOps};
use crate::kern::{embed_plan, TopGpu};
use crate::kern::{norm_threads, Kern};
use crate::kern::{pf_kib, KernG4, NextPf};
use crate::kern35::{Kern35, Mat};
use crate::kern_nh::{KernNh, MoeBlock, SsdWs};
use crate::launch;
use crate::runtime::kinds::Rounding;
use crate::sched::Sample;
use crate::weights::{choose_layout, ordered, pack_item_map, upload_bytes, upload_packed, Item, Layout, PackBuf};
pub use attn::AttnOp;
use glyd_gpu::safetensors::Checkpoint;
pub use mla::{Cand, IdxKeys, IdxPick, IdxScore, KeyNorm, MlaIn, MlaIndexer, MlaOp};
#[cfg(feature = "dev")]
pub(crate) use rows::EXACT_ROWS;
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
/// in place (a prompt's rows copied into the slot first: `Op::Parts`).
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
    /// The RMS norm kind (`Norm` with `NormKind::Rms`): y = bf16(rmsnorm(x) w) with its `eps`, the weight `w` where `weight` says (`NormW`; a weightless norm's is ones).
    RmsNorm {
        x: Slot,
        w: usize,
        y: Slot,
        weight: NormW,
        eps: f32,
    },
    /// The pre-norm residual: x += y, h = bf16(rmsnorm(x) w), times (1 + w) where `plus1` (Qwen3.5's).
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
    /// `sigmoid`: the gate sigmoid(z) (`Linear::gate_act`, Qwen3.8-Flash-Next's; the weight bf16 as saved), else silu(z) (the weight fp32, as Qwen3.5's engine holds it).
    GatedNorm {
        w: usize,
        row: usize,
        nv: usize,
        z_off: usize,
        sigmoid: bool,
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
    /// The mixture-of-experts kind (`Ffn::Moe`), the part's block's: see `MoeOp`.
    Moe(MoeOp),
    /// The residual streams kind (`Residual::Streams`, mHC; `Residual::GatedStreams`, the gated residuals; src/dynamic/residual.rs, kernels/residual.cu), its arithmetic
    /// `round` (`StreamsP::rounding`): the streams S from the embedding row X, four times; before a block, the block's input H by `mix` (`StreamsMix`), the weights of its
    /// output over the streams kept for StreamsPost; after it, S from its output O and those weights (`gated`: each stream plus its weight times O, the gated residuals'; else
    /// mHC's mixing); at the end, the streams collapsed by `mix` (`HeadMix`) into H (the LM head reads its last rows).
    Streams,
    StreamsPre {
        mix: StreamsMix,
        norm: usize,
        round: Rounding,
    },
    StreamsPost {
        round: Rounding,
        gated: bool,
    },
    StreamsHead {
        mix: HeadMix,
        norm: usize,
        round: Rounding,
    },
    /// y = x as the release's products quantize their input: E4M3 in blocks of `block`.
    Fp8 {
        x: Slot,
        y: Slot,
        block: usize,
    },
    /// The compressed attention's (`runtime::kinds::Comp`; src/dynamic/comp.rs, kernels/dsv4.cu): its queries Qo (`heads` of 512), each normed in bf16 arithmetic where `norm`
    /// (`Comp::q_head_norm`), the rotary of table `rope`.
    CompQ {
        rope: usize,
        heads: usize,
        norm: bool,
    },
    /// Its key-value row Kv (keys = values): normed by `norm`, the rotary of `rope`, its first 448 channels to E4M3; into ring `ring` (the window).
    CompKv {
        norm: usize,
        rope: usize,
        ring: usize,
    },
    /// Compressed keys and values: a compressor of H, its two products by `w` (small: [2 W, hidden] bf16, kv's rows then the gate's) in fp32 into Cp, the pooling of every `ratio` rows with `ape` (small,
    /// fp32) and its state (states `state` and `state` + 1: the kv rows and the scores), each completed window's entry normed by `norm`, the rotary of `rope`, quantized (`indexer`: rotated, E2M1; else
    /// E4M3) into ring `cache` (`Ring::entries`: an entry a row); where `raw`, each entry as normed before its rotation too (exact in bf16) into ring `raw` (`Ring::entries`
    /// of the same ratio: the cache's group, one block table): the indexer's keys of a shared compressed attention are made of these (`Op::IdxKeys`). `overlap`: windows of
    /// 2 `ratio` rows, the previous window's first half of the channels (`Comp::overlap`).
    Compress {
        w: usize,
        ape: usize,
        norm: usize,
        d: usize,
        ratio: usize,
        overlap: bool,
        rope: usize,
        cache: usize,
        state: usize,
        indexer: bool,
        raw: Option<usize>,
    },
    /// The indexer's keys of a shared compressed attention's entries (`CompShare`: a kv source with the indexer's key product): each entry the call completes, of state
    /// ring `raw` (`Op::Compress`'s, normed, before its rotation), by the key product `w` (small, bf16 [128, 512]), normed by `norm` (small [128]), rotated by table `rope`
    /// at its window's first position, rounded as the indexer's entries (E2M1 where the release's quantizations are simulated) into ring `keys` (`Ring::entries` of 128;
    /// both rings of the entries' group).
    IdxKeys {
        w: usize,
        norm: usize,
        rope: usize,
        ratio: usize,
        raw: usize,
        keys: usize,
    },
    /// The indexer's queries Iq (`heads` of 128): the rotary of `rope`, the rotation, E2M1.
    IdxQ {
        rope: usize,
        heads: usize,
    },
    /// The indexer's pick (src/dynamic/mla.rs, every indexer's): Iq and Iw (its weights' product) scored against its entries, each row's best.
    IdxPick(IdxPick),
    /// The mean-pooled indexer's queries and keys (`IndexForm::Mean`; src/dynamic/picked.rs, kernels/qwen4x.cu): of the index product in Qr (`heads` + 1 heads of `dim` a
    /// row), each query head normed (`q_norm`, (1 + w)) and rotated by table `rope` into Iq; the raw key into ring `keys` at the row's position; then each pool
    /// of `block` the call completes: the mean of its raw keys, normed (`k_norm`) and rotated at its first position, into ring `pools` (`Ring::entries`, a pool a
    /// row: `IdxKeys::Pools`). The raw keys are ring `keys` [positions, dim]: both paged as any keys.
    IdxMean {
        heads: usize,
        dim: usize,
        block: usize,
        q_norm: usize,
        k_norm: usize,
        rope: usize,
        keys: usize,
        pools: usize,
    },
    /// The hashed n-gram embedding kind (`Body::Ngram`; src/dynamic/ngram.rs): its rows of the table into In, then (after In's product into Gu) its gated add into the
    /// streams S.
    NgramRows(ngram::NgramRows),
    NgramAdd(ngram::NgramAdd),
    /// The attention of Qo over the window (ring `ring`) and the compressed entries (ring `cache`: ratio 4, the indexer's picks; another, every complete entry) with the sink (small, fp32), the
    /// rotary of `rope` taken off its output; into Att in the grouped low rank's order (`hpg` heads a group).
    /// `picked`: the entries the latest indexer's pick chose (its picks' buffer: the layer's own, or the index source's it shares), else every complete entry.
    CompAttn {
        ring: usize,
        cache: Option<usize>,
        ratio: usize,
        picked: bool,
        sink: usize,
        rope: usize,
        heads: usize,
        hpg: usize,
    },
    /// The output's grouped low rank: group g's rows of Att (hpg 512 wide) by matrix `w` + g (an FP8 matrix where `in_f8`) into Mid (`ol` wide), then back in rows into `y` (E4M3 where `fp8`).
    OGroupLora {
        w: usize,
        in_f8: bool,
        groups: usize,
        hpg: usize,
        ol: usize,
        y: Slot,
        fp8: bool,
    },
    /// The same with the groups' matrices one FP8 stack (`f8[w]`, [groups ol, hpg 512]): every group's product in one launch, the groups as the experts' product's experts.
    OGroupLoraStack {
        w: usize,
        groups: usize,
        ol: usize,
        y: Slot,
        fp8: bool,
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
    Parts {
        row: usize,
        parts: [(usize, usize, Slot); 3],
    },
    /// The latent attention layer kind, every family's (src/dynamic/mla.rs): H into O.
    Mla(MlaOp),
    /// The depth mix kind (`Residual::DepthMix`, `Collapse::DepthMix`): see `DepthMixOp`.
    DepthMix(DepthMixOp),
    /// The plain residual (`Residual::Add`): X = bf16(X + x).
    Add {
        x: Slot,
    },
    /// A cached experts block's guess of its experts in a decode step: its input as `mix` makes it (its add and push left out) over X before the layer's mixer adds
    /// to it, into Att (written before it is read in the layer), and the block's router on that (`MoeBlock::early`), so that the copies of the experts it guesses run
    /// beside the mixer.
    MoeEarly {
        block: usize,
        mix: DepthMixOp,
    },
    /// A gated MLP's activation (`Mlp::act`, its `clamp`): M = act(gate, up) of Gu (the gate's `inter` columns, then the up's), rounded once.
    GatedAct {
        inter: usize,
        act: crate::moe::Act,
    },
    /// Nothing (a model whose first operation is none)
    Nop,
    /// The gated residual kind (`Residual::Gated`): X = bf16(bf16(alpha X) + bf16(beta O)), the factors small `alpha`, `beta`.
    GatedResidual {
        alpha: usize,
        beta: usize,
    },
}

/// How a streams kind's block takes its input from the streams (`Op::StreamsPre`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum StreamsMix {
    /// mHC (`Residual::Streams`): the mixing weights of S by the projection `f` (small, [24, 4 hidden]: fp32 where the rounding is `Once`, else bf16), `base` and
    /// `scale`, made doubly stochastic (the model's `hc`), and the collapse normed by the op's `norm` into H.
    Hc { f: usize, base: usize, scale: usize },
    /// Lagged mHC (`StreamsP::lagged`): `Hc`'s weights made, the collapse by the previous lagged block's input weights (one-hot on stream 0 where `first`), this block's own kept for the
    /// next (kernels/residual.cu hc_lag).
    HcLagged { f: usize, base: usize, scale: usize, first: bool },
    /// The gated residuals (`Residual::GatedStreams`): each stream of S normed by the op's `norm` ((1 + w), a weight a stream) into Hq; the bottleneck's products
    /// `down` (Hq into Mid), silu of a quarter, `up` (Mid into Cp); H the mean over the streams of sigmoid(Cp) Hq; the streams' weights 2 sigmoid(`inject` Hq / 4)
    /// (`inject`'s product into Kv) kept for StreamsPost. Four streams (kernels/qwen4x.cu's).
    Gated { down: usize, up: usize, inject: usize },
}

/// How a streams kind's streams become the head's input (`Op::StreamsHead`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum HeadMix {
    /// Their mean, then the op's final norm (`Collapse::Mean`).
    Mean,
    /// The hyper-head's weights of them (its projection, base and scale), then the final norm (`Collapse::Weighted`).
    Weighted([usize; 3]),
    /// The gated residuals' mix of them, normed a stream at a time by the op's `norm` (`Collapse::GatedMix`): `StreamsMix::Gated` with no inject.
    Gated { down: usize, up: usize },
    /// Lagged mHC's: weighed by the last lagged block's input weights, then the final norm (`Collapse::LastPre`).
    LastPre,
}

/// Where an RMS norm's weight goes (`kinds::NormWeight` as the loader holds the small tensor: a (1 + w) it may hold already added).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NormW {
    /// y = bf16(rmsnorm(x) w)
    Plain,
    /// y = bf16(rmsnorm(x) (1 + w)), the one added in the kernel (Qwen3.5's)
    PlusOne,
    /// y = bf16(rmsnorm(bf16(x w))): the weight scales the input, the norm has none (`NormWeight::InputPlusOne`)
    Input,
}

/// A mixture of experts (the part's block `block`): the router reads H, the experts and the shared expert `x` (H; the release's FP8-rounded H, Hq; a latent
/// MoE's normed latent, Xc), their sum into `y` (O, or the latent's M). The kind's variants: hash routing by the token (`Router::hash`: the table, small
/// int64 [vocab, k]); the router's logits times `logit_scale` (`Router::logit_scale`; its weights fp32 then); each expert's output normed before it is
/// weighed (`Latent::expert_norm`: the small input-scaling weight and its eps); each pair's weight applied to its activation before the second product,
/// which then sums with weights of one, the activation rounded as that product's input (`weigh_act`: the release's arithmetic; `fp8`: rounded to E4M3 in
/// blocks of 128, as the release's products quantize their inputs).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MoeOp {
    pub block: usize,
    pub x: Slot,
    pub y: Slot,
    pub hash: Option<usize>,
    pub logit_scale: f32,
    pub expert_norm: Option<(usize, f32)>,
    pub weigh_act: bool,
    pub fp8: bool,
}

impl MoeOp {
    /// The block's experts of H into O, routed by their scores alone.
    pub const fn of(block: usize) -> MoeOp {
        MoeOp { block, x: Slot::H, y: Slot::O, hash: None, logit_scale: 1.0, expert_norm: None, weigh_act: false, fp8: false }
    }
}

/// A mixture-of-experts block whose experts are all on the GPU: the core's layer, run by `Op::Moe` (its `MoeOp` says how).
pub struct ResidentMoe(pub crate::moe::MoeLayer);

impl MoeBlock for ResidentMoe {
    unsafe fn forward(&self, _: &dyn crate::kern_nh::NhHost, _: u64, _: usize, _: u64) -> Result<()> {
        err("these experts run on the dynamic engine")
    }
    fn graphable(&self) -> bool {
        true
    }
    fn bytes(&self) -> usize {
        self.0.weights_bytes()
    }
    fn resident(&self) -> Option<(&crate::moe::MoeLayer, Option<(&Mat, &Mat)>)> {
        Some((&self.0, None))
    }
}

/// A block's start in the depth mix (kernels/residual.cu `depth_mix`, one kernel): X = bf16(X + O) where `add`; the mix of the first `nb` completed sources (Blk)
/// and X, by its norm weight `w_n` and projection `w_p` (small); X into source `push` of Blk, then X = 0, where given (the next group's partial sum starts empty);
/// the norm of the mix by `w` (`eps`, the normed value rounded, times the weight, rounded again) into H.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DepthMixOp {
    pub add: bool,
    pub nb: usize,
    pub w_n: usize,
    pub w_p: usize,
    pub push: Option<usize>,
    pub w: usize,
    pub eps: f32,
}

/// What DeepSeek-V4's operations need beyond the slots (its scratch is sized from these and the context).
#[derive(Clone, Debug)]
pub struct V4Dims {
    pub heads: usize,
    pub idx_heads: usize,
    pub topk: usize,
    pub window: usize,
    /// the smallest compression whose entries a query reads all of (0: none)
    pub all_ratio: usize,
    /// the release's quantizations (the products' inputs to FP8, the caches' and the indexer's simulated FP8 and FP4); off: a development check's arithmetic without them
    pub sim: bool,
    /// the attention output's low-rank groups
    pub o_groups: usize,
    /// the smallest ratio of a layer with an indexer (its entries a chunk makes and a query scores at most; 0: none)
    pub idx_ratio: usize,
}

/// cos and sin [positions, pairs] of pairs i < `pairs` of a head of `dim`: inv_freq_i = theta^(-2i / dim).
#[derive(Clone, Debug)]
pub struct Rope {
    pub theta: f64,
    pub dim: usize,
    pub pairs: usize,
    /// Each angle (fp32) through f64 trig, rounded to fp32 then bf16 (transformers' cos and sin in the activations' type, Muse Spark's); else fp32 trig.
    pub trig64: bool,
}

/// A layer's keys and values, `nk` heads of `d`: a ring of the window and a chunk where `window` > 0, else the whole context. `values` false: keys alone, in kc
/// (a latent cache, an indexer's keys, DeepSeek-V4's window whose keys are its values), no room in vc. `ratio` > 1: a row an entry of `ratio` tokens, not a token
/// (DeepSeek-V4's compressed entries, the attention's and the indexer's: entry e at row e), whose blocks then cover BLOCK `ratio` tokens.
#[derive(Clone, Debug)]
pub struct Ring {
    pub d: usize,
    pub nk: usize,
    pub window: usize,
    pub values: bool,
    pub ratio: usize,
}

impl Ring {
    /// A ring of keys and values.
    pub const fn kv(d: usize, nk: usize, window: usize) -> Ring {
        Ring { d, nk, window, values: true, ratio: 1 }
    }

    /// A ring of keys alone.
    pub const fn keys(d: usize, nk: usize, window: usize) -> Ring {
        Ring { d, nk, window, values: false, ratio: 1 }
    }

    /// The compressed entries of every `ratio` tokens, `d` wide (keys alone, the whole context).
    pub const fn entries(d: usize, ratio: usize) -> Ring {
        Ring { d, nk: 1, window: 0, values: false, ratio }
    }

    /// Bytes a token takes in it (bf16; an entry's over its tokens, rounded up).
    pub fn token_bytes(&self) -> usize {
        ((1 + self.values as usize) * self.nk * self.d * 2).div_ceil(self.ratio.max(1))
    }

    /// Rows it holds for a context of `ctx` tokens: a window's ring the window and a chunk of `chunk`, compressed entries one for every `ratio` tokens (and two more:
    /// a pass's last entry is written past its last complete one), else a row a token.
    pub fn rows(&self, ctx: usize, chunk: usize) -> usize {
        if self.ratio > 1 {
            ctx / self.ratio + 2
        } else if self.window > 0 {
            (self.window + chunk).min(ctx)
        } else {
            ctx
        }
    }
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
    /// The small tensors made from their tensor's values rather than read as saved (`SmallForm`), by index into `small`.
    pub forms: Vec<(usize, SmallForm)>,
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
    /// The residual streams' eps and Sinkhorn rounds (`StreamsP::eps`, `sinkhorn_iters`; 0s where none).
    pub hc: (f32, i32),
    /// The logits times this before the soft-cap (Muse Spark's: cap tanh(x mult / cap), each step rounded); 0: the soft-cap alone.
    pub logit_mult: f32,
}

impl ModelDesc {
    /// The residual streams a row holds (S's width over the hidden width; one where it has none).
    pub fn streams(&self) -> usize {
        (self.width(Slot::S) / self.hidden.max(1)).max(1)
    }

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

/// How a small tensor is made from the values its checkpoint holds (bf16), as the operation reading it takes it: each result rounded to bf16.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SmallForm {
    /// `n` ones, no tensor read (a weightless norm's weight).
    Ones(usize),
    /// 1 + w (a centered norm's factors, as transformers adds them: `muse::plus_one`).
    PlusOne,
    /// A residual gate's two factors of its parameter at temperature T (`muse::residual_gate`): alpha, and beta.
    GateAlpha(f32),
    GateBeta(f32),
    /// Several tensors one after another, each as saved and all of one type: the small tensor's name lists them, `|` between them (a delta attention layer's
    /// three convolutions as the one its kernel reads).
    Joined,
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
    fn mats(&self) -> Vec<usize> {
        match *self {
            Op::Linear { w, .. } | Op::LinearAt { w, .. } | Op::GateUpSilu { w, .. } | Op::OGroupLora { w, in_f8: false, .. } => vec![w],
            Op::StreamsPre { mix: StreamsMix::Gated { down, up, inject }, .. } => vec![down, up, inject],
            Op::StreamsHead { mix: HeadMix::Gated { down, up }, .. } => vec![down, up],
            Op::NgramRows(x) => vec![x.table],
            Op::Mla(ref a) => match a.input {
                MlaIn::Fused { w } => vec![w, a.q_b, a.o],
                MlaIn::Apart { q_a, kv_a } => vec![q_a, kv_a, a.q_b, a.o],
            },
            _ => Vec::new(),
        }
    }

    fn rings(&self) -> Vec<usize> {
        match *self {
            Op::Attn(ref a) => vec![a.ring],
            Op::IdxMean { keys, pools, .. } => vec![keys, pools],
            Op::Mla(ref a) => std::iter::once(a.ring).chain(a.indexer.as_ref().map(|_| a.ring + 1)).collect(),
            Op::CompKv { ring, .. } => vec![ring],
            Op::CompAttn { ring, cache, .. } => std::iter::once(ring).chain(cache).collect(),
            Op::Compress { cache, raw, .. } => std::iter::once(cache).chain(raw).collect(),
            Op::IdxKeys { raw, keys, .. } => vec![raw, keys],
            Op::IdxPick(IdxPick { keys: IdxKeys::Entries(r) | IdxKeys::Pools(r), .. }) => vec![r],
            _ => Vec::new(),
        }
    }

    fn states(&self) -> Vec<usize> {
        match *self {
            Op::GdnConv { state, .. } | Op::GdnDelta { state, .. } | Op::MambaConv { state, .. } | Op::MambaScan { state, .. } => vec![state],
            Op::Compress { state, .. } => vec![state, state + 1],
            Op::KdaConv { state, .. } | Op::KdaDelta { state, .. } => vec![state],
            Op::NgramRows(x) => vec![x.hist],
            Op::NgramAdd(x) => x.conv.map(|c| c.state).into_iter().collect(),
            _ => Vec::new(),
        }
    }
}

impl ModelDesc {
    /// The matrices a stage multiplies by (its layers', the embedding where it is the first, the head and post projection where the last).
    fn wanted(&self, st: &Stage) -> Vec<bool> {
        let mut w = vec![false; self.mats.len().max(self.lm + 1).max(self.embed + 1)];
        for op in self.layers[st.layers.0..st.layers.1].iter().flatten().chain(st.first.then_some(&self.first)) {
            for k in op.mats() {
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

/// One mixture-of-experts call as `Dyn::trace_routes` records it: its layer, the experts a row takes (k), its rows' experts ([rows, k]) and its
/// router's input rows (bf16, [rows, the input's width]).
pub type Routes = (usize, usize, Vec<i64>, Vec<u16>);

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
            + self.small.iter().map(|b| b.bytes).sum::<usize>()
            + self.moe.iter().map(|b| b.bytes()).sum::<usize>()
            + self.ram.as_ref().map_or(0, |r| r.bytes().0)
    }

    /// The part's weights as loaded by the family's own loader, with its activations and outputs.
    fn from_loaded(d: ModelDesc, l: Loaded, rows: usize, out: usize) -> Result<Part> {
        let slots: Vec<Buf> = d.widths.iter().map(|&w| Buf::new(w.max(1) * rows * 2)).collect::<Result<_>>()?;
        let logits = Buf::new(d.vocab * out * 2)?;
        let (tok, val) = (Buf::zeroed(4 * out)?, Buf::zeroed(4 * out)?);
        Ok(Part { d, mats: l.mats, f8: l.f8, moe: l.moe, small: l.small, ram: l.ram, slots, logits, tok, val })
    }
}

/// Where a stage's packed matrices are held beyond its GPU's own memory. `across`: every layer across the thread's group of GPUs where it has one
/// (src/dynamic/tp.rs), this GPU holding its slice of the rows of each matrix big enough to slice (a Hub-format directory's, packed here). `ram`: the layers
/// whose packed matrices are in the computer's memory, copied into one slot on the GPU before each runs (src/dense_ram.rs).
#[derive(Clone, Copy, Debug, Default)]
pub struct Placing {
    pub across: bool,
    pub ram: crate::dense_ram::Placer,
}

/// What reads a description's blocks of experts (`Op::Moe`'s, in its order) from its checkpoint, the matrices' layout given.
pub type Experts<'a> = dyn Fn(&Checkpoint, Layout) -> Result<Vec<Box<dyn MoeBlock>>> + 'a;

/// The first matrix the description's operations, embedding or head use that it does not name (None: the executor's loader can read every one).
fn unnamed(d: &ModelDesc) -> Option<usize> {
    std::iter::once(&d.first).chain(d.layers.iter().flatten()).flat_map(Op::mats).chain([d.embed, d.lm]).find(|&k| k >= d.mats.len())
}

/// The checkpoint's tensors of the matrices layer `l`'s operations multiply by (`ModelDesc::mats`' members; a fused product's zero rows are none): what the
/// executor's loader keeps in the computer's memory for a layer there (`Placing::ram`), each one it packs (the planner's test holds its figures to them).
#[cfg(test)]
pub(crate) fn layer_tensors(d: &ModelDesc, l: usize) -> Vec<String> {
    d.layers[l].iter().flat_map(Op::mats).filter_map(|k| d.mats.get(k)).flat_map(|it| it.members.iter().filter(|m| !m.is_empty()).cloned()).collect()
}

/// Whether matrix `k` of `d` is held as saved (bf16), not packed: a table an operation reads by row (`Op::NgramRows`), or a shape the library does not pack
/// (columns not a multiple of 16, or a member's rows, `rows_of` it where the checkpoint holds it, not a multiple of 64: a save keeps such a merge as its parts;
/// an item padded with zero rows is one matrix of its own rows, whatever its members').
fn held_as_saved(d: &ModelDesc, k: usize, rows_of: &dyn Fn(&str) -> Option<usize>) -> bool {
    let it = &d.mats[k];
    let padded = it.members.last().is_some_and(|m| m == crate::weights::ZERO_ROWS);
    !it.cols.is_multiple_of(16)
        || if padded { !it.rows.is_multiple_of(64) } else { it.members.iter().any(|m| rows_of(m).is_some_and(|r| r == 0 || !r.is_multiple_of(64))) }
        || d.layers.iter().flatten().any(|op| matches!(op, Op::NgramRows(x) if x.table == k))
}

/// The small tensors an operation reads in fp32 (the gated delta rule's A_log; the Kimi delta rule's A_log and dt_bias; the mHC streams' mix's base and
/// scale): a bf16 one is widened.
fn read_f32(d: &ModelDesc) -> Vec<usize> {
    d.layers
        .iter()
        .flatten()
        .flat_map(|op| match *op {
            Op::GdnDelta { a_log, .. } => vec![a_log],
            Op::KdaDelta { a_log, dt_bias, .. } => vec![a_log, dt_bias],
            Op::StreamsPre { mix: StreamsMix::Hc { base, scale, .. } | StreamsMix::HcLagged { base, scale, .. }, .. } => vec![base, scale],
            _ => Vec::new(),
        })
        .collect()
}

/// The small tensors an operation reads as saved in the type its own flag says (fp32 where true, else bf16: the Kimi delta attention's convolutions and gated
/// norm, which a release holds in either): a tensor of the other type is refused.
fn read_typed(d: &ModelDesc) -> Vec<(usize, bool)> {
    d.layers
        .iter()
        .flatten()
        .filter_map(|op| match *op {
            Op::KdaConv { w, w32, .. } | Op::KdaOut { w, w32, .. } => Some((w, w32)),
            _ => None,
        })
        .collect()
}

/// The values a small tensor's operation reads of it at least, where the operation says (a shorter tensor would be read past its end on the GPU; a longer one
/// is read in its first values, as a release's A_log of more heads than the model has): the Kimi delta rule's A_log a head and dt_bias a channel, its gated
/// norm a head's channels (a head's channels: the recurrence's state holds heads x channels x channels fp32), its convolutions' taps a channel (the row's own
/// and the state's, bf16).
fn read_at_least(d: &ModelDesc) -> Vec<(usize, usize)> {
    let mut v = Vec::new();
    for ops in &d.layers {
        let kd = ops
            .iter()
            .find_map(|op| match *op {
                Op::KdaDelta { state, heads, .. } => Some((d.states.get(state).copied().unwrap_or(0) / (4 * heads.max(1))).isqrt()),
                _ => None,
            })
            .unwrap_or(0);
        for op in ops {
            match *op {
                Op::KdaDelta { a_log, dt_bias, heads, .. } => v.extend([(a_log, heads), (dt_bias, heads * kd)]),
                Op::KdaConv { w, state, c, .. } => v.push((w, c + d.states.get(state).map_or(0, |b| b / 2))),
                Op::KdaOut { w, .. } => v.push((w, kd)),
                _ => {}
            }
        }
    }
    v
}

/// The small tensors an operation reads in fp32 with the values the model's dtype holds (the silu-gated norm's weight, `Op::GatedNorm` not `sigmoid`, as vLLM
/// holds it): an fp32 one rounded to bf16, kept four bytes wide; a bf16 one widened.
fn read_f32_as_bf16(d: &ModelDesc) -> Vec<usize> {
    d.layers.iter().flatten().filter_map(|op| if let Op::GatedNorm { w, sigmoid: false, .. } = op { Some(*w) } else { None }).collect()
}

/// cos and sin [positions, pairs] as `rotary` makes them, each angle (fp32) through f64 trig and rounded to fp32 then bf16 (`Rope::trig64`).
fn rotary64(theta: f64, dim: usize, pairs: usize, max_pos: usize) -> (Vec<u16>, Vec<u16>) {
    let inv: Vec<f32> = (0..pairs).map(|i| 1.0f32 / (theta as f32).powf((2 * i) as f32 / dim as f32)).collect();
    let (mut c, mut s) = (Vec::with_capacity(max_pos * pairs), Vec::with_capacity(max_pos * pairs));
    for p in 0..max_pos {
        for &f in &inv {
            let (s_, c_) = ((p as f32 * f) as f64).sin_cos();
            c.push(crate::weights::bf16_bits(c_ as f32));
            s.push(crate::weights::bf16_bits(s_ as f32));
        }
    }
    (c, s)
}

/// Small tensor `name` made in form `f` from its checkpoint's bf16 values: bf16 words.
fn small_of(ck: &Checkpoint, name: &str, f: SmallForm) -> Result<Vec<u16>> {
    let vals = || -> Result<Vec<f32>> {
        let (_, t) = ck.get(name).ok_or_else(|| Error(format!("no tensor {name}")))?;
        if t.dtype != "BF16" {
            return err(format!("{name}: {}, where bf16 is read", t.dtype));
        }
        let b = ck.read(name).map_err(|e| Error(format!("{name}: {e}")))?;
        Ok(b.as_chunks::<2>().0.iter().map(|x| f32::from_bits((u16::from_le_bytes(*x) as u32) << 16)).collect())
    };
    let v = match f {
        SmallForm::Ones(n) => vec![1.0; n],
        SmallForm::PlusOne => crate::muse::plus_one(&vals()?),
        SmallForm::GateAlpha(t) => crate::muse::residual_gate(&vals()?, t).0,
        SmallForm::GateBeta(t) => crate::muse::residual_gate(&vals()?, t).1,
        SmallForm::Joined => return err(format!("{name}: several tensors as one are read as saved, not made from their values")),
    };
    Ok(v.into_iter().map(crate::weights::bf16_bits).collect())
}

/// Small tensor `name` as its checkpoint holds it, its type and bytes; `joined`: the tensors the name lists (`SmallForm::Joined`), one after another, all of one type.
fn small_saved(ck: &Checkpoint, name: &str, joined: bool) -> Result<(String, Vec<u8>)> {
    let one = |n: &str| -> Result<(String, Vec<u8>)> {
        let (_, t) = ck.get(n).ok_or_else(|| Error(format!("no tensor {n}")))?;
        Ok((t.dtype.clone(), ck.read(n).map_err(|e| Error(format!("{n}: {e}")))?))
    };
    if !joined {
        return one(name);
    }
    let (mut dtype, mut bytes, mut each) = (String::new(), Vec::new(), 0);
    for n in name.split('|') {
        let (dt, b) = one(n)?;
        if dtype.is_empty() {
            (dtype, each) = (dt, b.len());
        } else if dt != dtype || b.len() != each {
            return err(format!("{n}: {dt} of {} bytes, where the tensors read as one with it are {dtype} of {each}", b.len()));
        }
        bytes.extend(b);
    }
    Ok((dtype, bytes))
}

/// Bytes a value of small tensor `i` takes on the GPU, stored as `dtype`.
fn small_width(d: &ModelDesc, i: usize, dtype: &str) -> u64 {
    if read_f32(d).contains(&i) || read_f32_as_bf16(d).contains(&i) || read_typed(d).contains(&(i, true)) {
        4
    } else if dtype == "I64" {
        8
    } else {
        2
    }
}

/// A saved directory's layout and manifest: a glyd-v7 one's, or a lossless glyd-v4 save's (the tiered layout); None: a Hub-format directory.
fn stored(dir: &Path) -> Result<Option<(Layout, glyd_gpu::json::Value)>> {
    Ok(match crate::weights::v7_manifest(dir)? {
        Some(x) => Some(x),
        None => crate::weights::lossless_manifest(dir)?.map(|m| (Layout::Tiered, m)),
    })
}

/// `d`'s weights from `dir`: a saved directory's packs as saved (under the first member's name where the save merged the members, else each member's), a matrix
/// it holds as bf16 packed here, a Hub-format directory's packed here at `layout`; the matrices `held_as_saved` names as saved; the small tensors in the type
/// their operations read (`read_f32`, `read_f32_as_bf16`; int64 as saved, a hash's numbers); the experts by `experts`. `placing`: where the packed matrices
/// are held beyond the GPU's own memory (this GPU's slice of a matrix's rows, a layer's matrices in the computer's memory).
#[allow(clippy::too_many_arguments)]
fn load_part(
    dir: &Path,
    d: ModelDesc,
    layout: Layout,
    threads: usize,
    rows: usize,
    out: usize,
    want: Option<&[bool]>,
    experts: &Experts,
    placing: &Placing,
) -> Result<Part> {
    let ck = Checkpoint::open(dir).map_err(|e| Error(e.to_string()))?;
    let v7 = stored(dir)?;
    // (every layer across a group of GPUs: this GPU's slice of the rows of a matrix a product multiplies by, packed here; the embedding, read by row, whole)
    let sl = if placing.across && v7.is_none() { tp::slicer() } else { None };
    if sl.is_some() && placing.ram.count() > 0 {
        return err(format!("{}: dense layers in the computer's memory: a stage on one GPU, not sliced across GPUs", d.family));
    }
    let slice = |k: usize| sl.filter(|t| k != d.embed && t.slices(d.mats[k].rows, d.mats[k].cols));
    // (the layer whose operations multiply by each matrix: a layer in the computer's memory keeps its packed matrices there, src/dense_ram.rs)
    let mut layer_of: Vec<Option<usize>> = vec![None; d.mats.len()];
    for (l, ops) in d.layers.iter().enumerate() {
        for k in ops.iter().flat_map(Op::mats) {
            if let Some(x) = layer_of.get_mut(k) {
                *x = Some(l);
            }
        }
    }
    let in_ram = |k: usize| layer_of[k].filter(|&l| placing.ram.in_ram(l));
    let has = |k: &str| ck.get(&format!("{k}.glyd_data")).is_some();
    let pack_rows = |k: &str| -> usize {
        v7.as_ref().and_then(|(_, m)| m.get("packs")?.get(k)?.get("shape")?.as_u64s()).and_then(|s| s.first().copied()).unwrap_or(0) as usize
    };
    let one = |k: usize| -> Result<crate::weights::Packed> {
        let it = &d.mats[k];
        let each: Vec<&str> = it.members.iter().map(|m| m.trim_end_matches(".weight")).collect();
        let parts: Vec<(&str, usize)> = if v7.is_none() {
            Vec::new()
        } else if has(each[0]) && (each.len() == 1 || !has(each[1])) {
            vec![(each[0], it.rows)]
        } else if each.iter().all(|k| has(k)) {
            each.iter().map(|k| (*k, pack_rows(k))).collect()
        } else {
            Vec::new()
        };
        match (&v7, slice(k)) {
            (Some((Layout::Tiered, m)), _) if !parts.is_empty() => crate::weights::read_lossless_parts(&ck, m, &parts, it.cols),
            (Some((l, m)), _) if !parts.is_empty() => crate::weights::read_v7(&ck, m, *l, &parts, it.cols),
            (_, Some(t)) => {
                let (r0, rn) = t.rows(it.rows);
                let w = crate::weights::item_words(&ck, it)?;
                crate::weights::pack_words(&w[r0 * it.cols..(r0 + rn) * it.cols], rn, it.cols, layout, &it.members.join(" + "))
            }
            _ => pack_item_map(&ck, it, layout, &|_| {}),
        }
    };
    let rows_of = |n: &str| ck.get(n).map(|(_, t)| t.shape[0] as usize);
    let saved: Vec<bool> = (0..d.mats.len()).map(|k| held_as_saved(&d, k, &rows_of)).collect();
    let mut packed: Vec<Option<PackBuf>> = (0..d.mats.len()).map(|_| None).collect();
    let mut hosted: Vec<Option<crate::weights::Packed>> = (0..d.mats.len()).map(|_| None).collect();
    // (a stage of a split model loads only the matrices its layers multiply by)
    let ks: Vec<usize> = (0..d.mats.len()).filter(|&k| want.is_none_or(|w| w[k]) && !saved[k]).collect();
    ordered(ks.len(), threads, 2 * threads.max(1), &|i| one(ks[i]), |i, pk| {
        crate::status::STATUS.beat();
        let k = ks[i];
        if in_ram(k).is_some() {
            hosted[k] = Some(pk);
            return Ok(());
        }
        let mut b = upload_packed(pk)?;
        if slice(k).is_some() {
            b.tp_rows = d.mats[k].rows;
        }
        packed[k] = Some(b);
        Ok(())
    })?;
    // ponytail: a matrix another stage holds is an empty dense one here (never multiplied by); a Mat variant of its own if one is ever read by mistake
    let mut ram = crate::dense_ram::RamBuilder::new(placing.ram);
    let mut mats: Vec<Mat> = Vec::with_capacity(d.mats.len());
    for (k, (x, h)) in packed.into_iter().zip(hosted).enumerate() {
        match (x, h, in_ram(k)) {
            (Some(p), _, _) => mats.push(Mat::Packed(p)),
            (None, Some(p), Some(l)) => {
                ram.push(l, &mut mats, p)?;
            }
            (None, _, _) if want.is_none_or(|w| w[k]) && saved[k] => {
                let it = &d.mats[k];
                mats.push(Mat::Dense(upload_bytes(crate::weights::as_bytes(&crate::weights::item_words(&ck, it)?))?, it.rows, it.cols));
            }
            _ => mats.push(Mat::Dense(Buf::new(2)?, 0, 0)),
        }
    }
    let ram = ram.finish(&mut mats)?;
    let (f32s, as_bf16, typed, least) = (read_f32(&d), read_f32_as_bf16(&d), read_typed(&d), read_at_least(&d));
    let small: Vec<Buf> = d
        .small
        .iter()
        .enumerate()
        .map(|(i, name)| {
            let form = d.forms.iter().find(|x| x.0 == i).map(|x| x.1);
            if let Some(f) = form.filter(|&f| f != SmallForm::Joined) {
                return upload_bytes(crate::weights::as_bytes(&small_of(&ck, name, f)?));
            }
            // (the tensor as saved; several as one, each as saved and all of one type)
            let (dtype, b) = small_saved(&ck, name, form == Some(SmallForm::Joined))?;
            let wide = f32s.contains(&i) || as_bf16.contains(&i);
            let values = b.len()
                / if dtype == "F32" {
                    4
                } else if dtype == "I64" {
                    8
                } else {
                    2
                };
            if let Some(&(_, n)) = least.iter().find(|x| x.0 == i && values < x.1) {
                return err(format!("{name}: {values} values, where its operation reads {n}"));
            }
            if let Some(&(_, w32)) = typed.iter().find(|x| x.0 == i) {
                return match (dtype.as_str(), w32) {
                    ("F32", true) | ("BF16", false) => upload_bytes(&b),
                    (dt, _) => err(format!("{name}: {dt}, where its operation reads {}", if w32 { "fp32" } else { "bf16" })),
                };
            }
            if dtype == "F32" && as_bf16.contains(&i) {
                let v: Vec<f32> = b.as_chunks::<4>().0.iter().map(|x| crate::kern35::bf16_round(f32::from_le_bytes(*x))).collect();
                return upload_bytes(crate::weights::as_bytes(&v));
            }
            match (dtype.as_str(), wide) {
                ("BF16", true) => {
                    let v: Vec<f32> = b.chunks_exact(2).map(|x| f32::from_bits((u16::from_le_bytes([x[0], x[1]]) as u32) << 16)).collect();
                    upload_bytes(crate::weights::as_bytes(&v))
                }
                ("F32", true) | ("BF16", false) | ("I64", false) => upload_bytes(&b),
                (dt, f) => err(format!("{name}: {dt}, where {} is read", if f { "bf16 or fp32" } else { "bf16, or int64 for a hash's numbers" })),
            }
        })
        .collect::<Result<_>>()?;
    let moe = experts(&ck, layout)?;
    Part::from_loaded(d, Loaded { mats, small, moe, ram, ..Loaded::default() }, rows, out)
}

/// What a family's loader puts on the GPU for its description: the matrices, the FP8 ones, the small tensors, the blocks of experts.
#[derive(Default)]
pub struct Loaded {
    pub mats: Vec<Mat>,
    /// The layers whose matrices are in the computer's memory (src/dense_ram.rs), copied into a slot before each runs.
    pub ram: Option<crate::dense_ram::DenseRam>,
    pub f8: Vec<crate::f8::F8Mat>,
    pub small: Vec<Buf>,
    pub moe: Vec<Box<dyn MoeBlock>>,
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
    /// The attention layer kind's kernels (`Op::Attn`) and the library's decode where its shapes take it; the latent attention kind's and the
    /// indexer's (`Op::Mla`, `Op::IdxPick`).
    ka: attn::KernAttn,
    ldec: Option<attn::LibDecode>,
    km: mla::KernMla,
    skinny: crate::skinny::SkinnyKern,
    skinny_on: bool,
    big: Part,
    draft: Option<Draft>,
    pub layout: Layout,
    ropes: Vec<(Buf, Buf)>,
    /// Each ring's bytes into kc and into vc (a ring of keys alone has no room in vc), and the positions it holds.
    ring_off: Vec<(u64, u64)>,
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
    /// The decoded-then-dense way for long chunks (ModelDesc::decoded): the library's or our own dense product, the decoded matrix's room, the GPU's code, and
    /// a development build's GLYD_ENGINE_DECODE_FROM (`decodes`).
    blas: Option<crate::ffi::Blas>,
    own: bool,
    scratch_w: Buf,
    gpu_code: i64,
    decode_from: Option<usize>,
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
    /// `--prof` (and GLYD_PROF=1, src/dynamic/prof.rs): an event after each stage of an eager pass (`prof_take`); the step graph's own (`gmarks`), captured with it
    /// where it was captured with these on, recorded by each replay. The events of eager passes taken are kept for the next (`spare`).
    pub prof: Option<std::cell::RefCell<Vec<(&'static str, Ev)>>>,
    gmarks: Vec<(&'static str, Ev)>,
    spare: std::cell::RefCell<Vec<Ev>>,
    /// A development build's GLYD_OP_DUMP=PATH: after each operation of a pass, every slot's rows of the pass hashed into PATH.N (N this engine's number in the
    /// process), a line each: the pass, the layer, the operation, its kind, the slot, the hash; two engines' files compared show the first operation they differ at.
    dump: Option<std::cell::RefCell<(std::fs::File, usize)>>,
    /// Whether the last pass was the step graph's replay (its events are then `gmarks`), whether the step graph is being captured, and the host's time to queue
    /// the last eager pass (ms).
    replayed: std::cell::Cell<bool>,
    capturing: std::cell::Cell<bool>,
    queued: std::cell::Cell<f64>,
    /// A scheduler's step's sampler candidates (`ModelStep::top`, src/dynamic/rows.rs): made on the GPU for the last step's head rows at the first ask.
    cands: rows::Cands,
    /// The layers this GPU runs (the whole model, or a stage of a split one).
    pub stage: Stage,
    /// DeepSeek-V4's buffers, where the model is one.
    v4: Option<deepseek_v4::V4Run>,
    /// GLM's kernels and the experts' activation.
    kglm: crate::kern::KernGlm,
    kmoe: crate::moe::MoeKern,
    /// The residual rules' kernels, a block's mixing weights of the streams (fp32 [chunk, 24]) and the split mixing projection's partial sums (`residual::part_words`).
    kres: residual::KernRes,
    hcp: Buf,
    /// Lagged mHC's input weights of each row, from block to block of a pass ([rows, 4] fp32).
    hc_lag: Buf,
    hc_part: Buf,
    /// Muse Spark's kernels.
    kmuse: muse::KernMuse,
    /// The mixture-of-experts kind's: the pairs' rows of experts whose outputs are normed (int32 [chunk, 32]); where a block is hash-routed, the pass's tokens
    /// (int32 [chunk]); where one weighs its activations, the weights of one its second product sums with (fp32 [chunk, k]).
    moe_inv: Buf,
    tok: Option<Buf>,
    ones: Option<Buf>,
    /// kernels/qwen4x.cu's, and the unweighted picks' room, where the description has the gated residuals, the n-gram embedding or the mean-pooled indexer.
    kqx: Option<crate::model_qwen4x::KernQx>,
    /// Where Some: each eager pass adds the residual (mHC's or the gated residuals' streams S, else X) of its rows after each layer, a check's view of each layer
    /// (`trace_layers`).
    trace: std::cell::RefCell<Option<Vec<Vec<u16>>>>,
    /// Where Some: the layer whose operations each eager pass records (`trace_ops`), and after each of them its name and the bf16 values of the slots `TRACED`.
    op_trace: std::cell::RefCell<Option<(usize, Vec<(String, Vec<Vec<u16>>)>)>>,
    /// Where Some: each eager pass adds, after each mixture-of-experts operation whose block leaves its choices in the experts' scratch
    /// (`MoeBlock::top_k`: a resident block, the expert cache's codebook block), its layer, its router's choices and its router's input rows (`trace_routes`).
    route_trace: std::cell::RefCell<Option<Vec<Routes>>>,
    picks: Option<picked::PickRun>,
    /// The latent attentions' scratch, where the model has one.
    mla: Option<mla::MlaRun>,
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
    /// In such a pass: its first rows that decode, a token of a sequence each, whose attention runs as one (`attn::Dyn::attn_lead`, through `rows::RowTabs`); 0: none.
    lead: std::cell::Cell<usize>,
    /// The CUDA graphs of steps of rows that decode, one for each padded row count met so far (`rows::Dyn::rows_replay`), whether such graphs may be captured (graphs on, and
    /// no block of experts that reads the host), the last one replayed (for `prof_take`), the replays and the captures made in a request's step rather than before the
    /// first (`usercheck` counts them).
    rgraphs: Vec<RowsGraph>,
    rgraph_on: bool,
    rplayed: std::cell::Cell<Option<usize>>,
    pub(crate) rows_replays: std::cell::Cell<u64>,
    pub(crate) rows_late: std::cell::Cell<u64>,
    /// Piecewise graphs of the passes no whole graph takes (src/dynamic/pieces.rs): the sets captured, how `forward` runs the pass being made, the pieces a capture makes,
    /// whether a piece's capture is open (to end it on an error), the set the last pass replayed and where its pieces' events go among the eager ones (`prof_take`), the
    /// pieces the last pass replayed, and the passes replayed so (`usercheck` counts them).
    pieces: pieces::Pieces,
    pmode: std::cell::Cell<pieces::Mode>,
    pcap: std::cell::RefCell<Vec<pieces::Piece>>,
    popen: std::cell::Cell<bool>,
    pplayed: std::cell::Cell<Option<usize>>,
    psplice: std::cell::RefCell<Vec<(usize, usize)>>,
    pcount: std::cell::Cell<usize>,
    pub(crate) piece_replays: std::cell::Cell<u64>,
    /// usercheck's switch: no graph replayed or captured (the one-row step's, the decode steps', the pieces'), every pass eager, to hold the graphs' replies to.
    graphs_off: std::cell::Cell<bool>,
    /// The attention's decode calls of the pass being made or last made (a launch of the parts kernel or a call of the library's decode: GLYD_PROF's count), and the one-row
    /// step graph's as it was captured.
    dcalls: std::cell::Cell<usize>,
    gcalls: usize,
    /// Paged keys and values (R4b: src/dynamic/pages.rs): kc and vc are then the pools every sequence's blocks are in, and each sequence has a block table a layer group. None: each sequence a
    /// room of its own.
    paged: Option<rows::Paged>,
    /// The 12-bit KV tables the paged layout uses for the rings they fit (`set_kv12`), and the pages it made of them.
    kv12: Option<kv12::Tables>,
    lossless: Option<kvpages::Lossless>,
    /// The host's swap area for paged sequences sent back for room, and its copies (R4d: src/dynamic/swap.rs); None: they are run again from their tokens.
    swap: Option<swap::Swap>,
    /// A served model's room for steps launched before the step before them is read (src/dynamic/rows.rs `Lanes`, `ModelStep::overlaps`); None: a step runs to its end.
    lanes: Option<rows::Lanes>,
    /// GLYD_ENGINE_STEP_PROF=1 or GLYD_PROF=1 on a served model: the GPU's time of each step and its idle time before it (src/dynamic/rows.rs `Gaps`).
    gaps: Option<rows::Gaps>,
    /// What the sampled rows of the next launched step ask of it (`ModelStep::draws`): their candidates and sums are queued with the step.
    draws: Vec<Option<f32>>,
    /// The 12-bit pages' status as the lane of the step landed last copied it (src/dynamic/kvpages.rs `kv12_lost` reads it once); None: the GPU's own.
    status_from: Option<*const u32>,
    /// The grammar masks of the next step's sampled rows (`ModelStep::masks`), put on the rows before their candidates' top-k (`masker`), and the room for a row's sum of
    /// exponentials (`ModelStep::mass`).
    row_masks: Vec<Option<crate::constrain::MaskHandle>>,
    masker: Option<crate::kern::MaskGpu>,
    sums: Option<Buf>,
    /// Launched steps landed with a later one queued, whose expert cache's end of a step is owed (src/dynamic/rows.rs `quiet`).
    owed: usize,
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
    gmarks: Vec<(&'static str, Ev)>,
}

/// The first operation a verify pass (`Dyn::verify_rows`) cannot run as a decode step runs each row, or `Dyn::rewind` cannot go back in: a mixture of experts (its
/// product groups the pass's rows), a cache or state other than the attention kind's rings (the rows past the kept ones are written over) and the gated delta
/// rule's (its convolution and recurrence run again over the kept rows). None: the description verifies.
fn unverifiable_op(d: &ModelDesc) -> Option<&Op> {
    std::iter::once(&d.first).chain(d.layers.iter().flatten()).find(|op| match op {
        Op::GdnConv { .. } | Op::GdnDelta { .. } => false,
        Op::Attn(a) => a.picked,
        Op::Moe(_) | Op::MoeEarly { .. } => true,
        _ => !rows::row_wise(op) || !op.rings().is_empty() || !op.states().is_empty(),
    })
}

/// The per-sequence states' number types as `held` reports them (an engine that holds some): the gated delta rule's where its operations hold every state
/// (src/dynamic/qwen35.rs: the convolution's state two bytes a value, the delta rule's four), else not reported.
fn state_type(d: &ModelDesc) -> &'static str {
    let mut held = d.layers.iter().flatten().filter(|op| !op.states().is_empty()).peekable();
    if held.peek().is_some() && held.all(|op| matches!(op, Op::GdnConv { .. } | Op::GdnDelta { .. })) {
        "fp32 (the delta rule's), bf16 (the convolution's)"
    } else {
        "not reported"
    }
}

/// The slots a cut between two layers hands the next layer: the ones its first operation reads before any operation writes them, with their widths. A layer
/// that begins with the depth mix (`Op::DepthMix`) reads the residual X and the completed blocks Blk; one whose residual add waits for the next layer (the
/// description's `final_norm` adds that slot on the last rows; each layer's first operation is `Op::NormAdd` of it) reads X and that slot; another hands X, H,
/// D, O and S where the model has them (what a layer may read first: its own first operation reads one of them).
fn handed_over(d: &ModelDesc) -> Vec<(Slot, usize)> {
    let slots: Vec<Slot> = if d.layers.iter().flatten().any(|op| matches!(op, Op::DepthMix(_))) {
        vec![Slot::X, Slot::Blk]
    } else if let Some((_, y, _)) = d.final_norm {
        vec![Slot::X, y]
    } else {
        vec![Slot::X, Slot::H, Slot::D, Slot::O, Slot::S]
    };
    slots.into_iter().map(|s| (s, d.widths[s as usize])).filter(|&(_, w)| w > 0).collect()
}

/// The most rows a verify pass takes (`Dyn::verify_rows`): a product of up to 8 rows runs the kernel a one-row step's does (the library's one m-tile product; the
/// tiered layout's step kernel to 8 rows), so each row's sums are a step's. Checked bit for bit on swift only; the range worker refuses verify on other levels.
pub const VERIFY_ROWS: usize = 8;

/// A step of rows that decode as a graph: its rows (padded), the graph, the events in it (`--prof`) and its attention's decode calls.
struct RowsGraph {
    rows: usize,
    graph: Graph,
    marks: Vec<(&'static str, Ev)>,
    calls: usize,
}

/// The library's routes (glyd_gpu.h GLYD_GPU_ROUTE_*) `Dyn::decodes` reads: W decoded for the caller's dense product, the prompt kernel, W decoded ahead.
const ROUTE_DECODE: i32 = 0;
const ROUTE_BIG: i32 = 4;
const ROUTE_AHEAD: i32 = 5;

/// Rows from which a prompt's product of a layout the route table has no crossover for on the GPU (an A100's tiered layout: the fused prompt kernel at any length)
/// is decoded and multiplied dense: the lowest crossover measured for the tiered layout elsewhere (an A10's 512, GeForce Ada's 513: gpu/glyd_gpu.cu `ahead_min`).
/// Not measured on an A100, an Ampere GeForce or a Blackwell card with a model of the last six months: the crossover measured on the other GPUs stands for them.
const DECODE_FROM: usize = 512;

/// The profile's stage for a matrix unpacked to bf16 before its dense product (GLYD_PROF=1: the product's own stage then is the dense product alone).
const DECODED: &str = "products: a matrix decoded to bf16 for the dense product";

/// `Dyn::decodes` for a product of `m` rows (more than 64) of a matrix in the 12-bit layout or not (`twelve`), `route` the library's route for a row count.
fn decodes_by(m: usize, twelve: bool, from: Option<usize>, route: &dyn Fn(i64) -> Result<i32>) -> Result<bool> {
    if let Some(f) = from {
        return Ok(m >= f);
    }
    match route(m as i64)? {
        ROUTE_DECODE | ROUTE_AHEAD => Ok(true),
        // (the prompt kernel past any length here: the table measured no crossover for this layout on this GPU)
        ROUTE_BIG => Ok(!twelve && m >= DECODE_FROM && route(1 << 40)? == ROUTE_BIG),
        _ => Ok(false),
    }
}

/// What `Dyn::rewind` needs of a verify pass: the per-sequence states as they were before it (`states`' sizes), each linear-attention layer's b, a rows then its
/// convolution's input rows (Mix's whole rows) of the pass (VERIFY_ROWS rows of each; None for another kind of layer or another stage's).
struct Verify {
    snap: Vec<Buf>,
    keep: Vec<Option<Buf>>,
}

/// The slots `Dyn::trace_ops` records after each operation: the blocks' input and output, the residual's streams and the mixers' and gated residuals' own.
pub const TRACED: [Slot; 10] = [Slot::H, Slot::O, Slot::S, Slot::Hq, Slot::Mid, Slot::Cp, Slot::Mix, Slot::Ba, Slot::Xc, Slot::Core];

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
    unsafe fn gemm(&self, x: u64, w: u64, y: u64, m: i32, n: i32, k: i32) -> Result<()> {
        self.kern.gemm_fill(self.dev.stream, x, w, y, m, n, k)
    }
}

impl Dyn {
    /// Each layer's residual (bf16, the pass's rows) after the passes run between `trace_layers(true)` and `trace_layers(false)`, which returns them: a
    /// check's view of each layer (the passes eager: no graph or piece holds a layer's end).
    pub fn trace_layers(&self, on: bool) -> Vec<Vec<u16>> {
        std::mem::replace(&mut *self.trace.borrow_mut(), on.then(Vec::new)).unwrap_or_default()
    }

    /// Whether the layers are traced (`trace_layers`, `trace_ops`, `trace_routes`): every pass eager, a decode step too.
    pub(crate) fn tracing(&self) -> bool {
        self.trace.borrow().is_some() || self.op_trace.borrow().is_some() || self.route_trace.borrow().is_some()
    }

    /// The routers' choices in the passes run between `trace_routes(true)` and `trace_routes(false)`, which returns them: for each mixture-of-experts
    /// operation in the order run, its layer, the experts a row takes (k), its rows' experts ([rows, k], each row's in the router's order) and the
    /// router's input as the operation read it (bf16, [rows, its width]). A layer whose block leaves no choices in the experts' scratch adds nothing.
    pub fn trace_routes(&self, on: bool) -> Vec<Routes> {
        std::mem::replace(&mut *self.route_trace.borrow_mut(), on.then(Vec::new)).unwrap_or_default()
    }

    /// The routed experts of each mixture-of-experts block as the GPU holds them, decoded on the host (`MoeLayer::held_experts`), in the blocks' order:
    /// a check's view of what a lossless level loaded. None where the model has no such block, or one with no view of its experts (a cache's, another
    /// format's); an error where a block's experts are a lossy level's block codes.
    pub fn held_experts(&self) -> Result<Option<Vec<(Vec<u16>, Vec<u16>)>>> {
        let mut v = Vec::with_capacity(self.big.moe.len());
        for (i, b) in self.big.moe.iter().enumerate() {
            let Some((l, _)) = b.resident() else { return Ok(None) };
            match l.held_experts().map_err(|e| e.context(&format!("mixture-of-experts block {i}")))? {
                Some(x) => v.push(x),
                None => return Ok(None),
            }
        }
        Ok((!v.is_empty()).then_some(v))
    }

    /// The router's input of `op` on n rows, read before it runs, where `trace_routes` records its call (`MoeBlock::top_k`): its input slot's
    /// rows, bf16.
    fn route_input(&self, p: &Part, op: &Op, n: usize) -> Result<Option<Vec<u16>>> {
        let Op::Moe(MoeOp { block, x, .. }) = *op else { return Ok(None) };
        if !std::ptr::eq(p, &self.big) || self.route_trace.borrow().is_none() || p.moe[block].top_k().is_none() {
            return Ok(None);
        }
        self.dev.sync()?;
        let mut v = vec![0u16; n * p.d.width(x)];
        p.slots[x as usize].read(0, &mut v)?;
        Ok(Some(v))
    }

    /// `trace_routes`' record of `op` of layer `l` on n rows, with its router's input (`route_input`, read before it ran).
    fn trace_route(&self, p: &Part, l: usize, op: &Op, n: usize, input: Option<Vec<u16>>) -> Result<()> {
        let Op::Moe(MoeOp { block, .. }) = *op else { return Ok(()) };
        if !std::ptr::eq(p, &self.big) || self.route_trace.borrow().is_none() {
            return Ok(());
        }
        let Some(k) = p.moe[block].top_k() else { return Ok(()) };
        let sc = self.moe_sc.get().ok_or_else(|| Error("no mixture-of-experts scratch".into()))?;
        self.dev.sync()?;
        let mut ids = vec![0i64; n * k];
        sc.ids.read(0, &mut ids)?;
        let input = input.ok_or_else(|| Error(format!("layer {l}: the router's input was not read before its experts ran")))?;
        if let Some(t) = self.route_trace.borrow_mut().as_mut() {
            t.push((l, k, ids, input));
        }
        Ok(())
    }

    /// Each operation of layer `layer` in the passes run between `trace_ops(Some(layer))` and `trace_ops(None)`, which returns them: its name and, after it ran,
    /// the values of the slots `TRACED` (the pass's rows, each slot's whole width): a check's view inside a layer.
    pub fn trace_ops(&self, layer: Option<usize>) -> Vec<(String, Vec<Vec<u16>>)> {
        std::mem::replace(&mut *self.op_trace.borrow_mut(), layer.map(|l| (l, Vec::new()))).map(|t| t.1).unwrap_or_default()
    }

    /// `trace_ops`' record of `op` of layer `l` on n rows, where that layer is traced.
    fn trace_op(&self, p: &Part, l: usize, op: &Op, n: usize) -> Result<()> {
        if !std::ptr::eq(p, &self.big) || self.op_trace.borrow().as_ref().is_none_or(|t| t.0 != l) {
            return Ok(());
        }
        let mut slots = Vec::with_capacity(TRACED.len());
        for &sl in &TRACED {
            let mut v = vec![0u16; n * p.d.width(sl)];
            if !v.is_empty() {
                // SAFETY: the slot holds a chunk's rows of its width; the stream is synchronized before the copy is read.
                unsafe { copy_d2h_async(v.as_mut_ptr() as *mut _, p.slot(sl), v.len() * 2, self.dev.stream)? };
            }
            slots.push(v);
        }
        self.dev.sync()?;
        let name = match op {
            Op::Linear { name, .. } => name.to_string(),
            o => format!("{o:?}").split([' ', '(', '{']).next().unwrap_or("").to_string(),
        };
        if let Some(t) = self.op_trace.borrow_mut().as_mut() {
            t.1.push((name, slots));
        }
        Ok(())
    }

    /// A description's model from `dir` by the generic loader (`ModelDesc::mats`, `small`, `ropes`), and a draft model's.
    pub fn new(dir: &Path, d: ModelDesc, draft: Option<DraftSpec>, o: &Opts) -> Result<Dyn> {
        let st = Stage::whole(d.layers.len());
        Dyn::new_stage(dir, d, draft, st, o)
    }

    /// Stage `st` of the model of `dir` (its layers, and the embedding or the head where it holds them), on `o.device`: a split model's part.
    pub fn new_stage(dir: &Path, d: ModelDesc, draft: Option<DraftSpec>, st: Stage, o: &Opts) -> Result<Dyn> {
        Dyn::new_with(dir, d, draft, st, o, &|_, _| Ok(Vec::new()))
    }

    /// `new_stage` of a description with experts: `experts` reads its blocks of experts (`Op::Moe`'s, in its order) from the checkpoint.
    pub fn new_with(dir: &Path, d: ModelDesc, draft: Option<DraftSpec>, st: Stage, o: &Opts, experts: &Experts) -> Result<Dyn> {
        Dyn::new_placed(dir, d, draft, st, o, experts, &Placing::default())
    }

    /// `new_with`, the stage's packed matrices held as `placing` says.
    #[allow(clippy::too_many_arguments)]
    pub fn new_placed(dir: &Path, d: ModelDesc, draft: Option<DraftSpec>, st: Stage, o: &Opts, experts: &Experts, placing: &Placing) -> Result<Dyn> {
        if draft.is_some() && !(st.first && st.last) {
            return err("drafting with a model split across GPUs: not yet");
        }
        if let Some(k) = unnamed(&d) {
            return err(format!(
                "{}: matrix {k} is used and the description names {} (a description its family's loader fills by index)",
                d.family,
                d.mats.len()
            ));
        }
        let dev = Dev::new(o.device)?;
        let threads = std::thread::available_parallelism().map_or(8, |n| n.get().min(16));
        // (the layout by the packed matrices' bytes and what is held beside them: the matrices held as saved, the small tensors as their operations read them)
        let ck = Checkpoint::open(dir).map_err(|e| Error(e.to_string()))?;
        let rows_of = |n: &str| ck.get(n).map(|(_, t)| t.shape[0] as usize);
        let bytes = |k: usize| (d.mats[k].rows * d.mats[k].cols * 2) as u64;
        let (saved, packed): (Vec<usize>, Vec<usize>) = (0..d.mats.len()).partition(|&k| held_as_saved(&d, k, &rows_of));
        let small: u64 = d
            .small
            .iter()
            .enumerate()
            .flat_map(|(i, n)| n.split('|').map(move |m| (i, m)))
            .map(|(i, n)| ck.get(n).map_or(0, |(_, t)| t.shape.iter().product::<u64>() * small_width(&d, i, &t.dtype)))
            .sum();
        let (matrix_bytes, other) = (packed.into_iter().map(bytes).sum(), saved.into_iter().map(bytes).sum::<u64>() + small);
        let layout = match stored(dir)? {
            Some((l, _)) => l,
            None => choose_layout(o.layout, dev.cc_pair(), dev.total_mem as u64, matrix_bytes, other).0,
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
                let (c, s) = if r.trig64 { rotary64(r.theta, r.dim, r.pairs, mp) } else { rotary(r.theta, r.dim, r.pairs, mp) };
                Ok((upload_bytes(crate::weights::as_bytes(&c))?, upload_bytes(crate::weights::as_bytes(&s))?))
            })
            .collect::<Result<_>>()?;
        let want = d.wanted(&st);
        let big = load_part(dir, d, layout, threads, ch, rows_out, Some(&want), experts, placing)?;
        // (a draft model is packed lossy here: the big model checks every draft token, so its precision moves only how many are kept)
        let draft = match draft {
            None => None,
            Some(x) => Some(Draft {
                p: load_part(x.dir, x.desc, Layout::Int6, threads, 1, 1, None, &|_, _| Ok(Vec::new()), &Placing::default())?,
                k: x.k,
                pos: Buf::zeroed(8)?,
            }),
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
        Dyn::from_parts(dev, d, Loaded { mats, f8, small, moe, ram: None }, ropes, layout, st, o)
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
        let rings_used: std::collections::HashSet<usize> = ops().flat_map(Op::rings).collect();
        let whole = stage.first && stage.last;
        let (mut ring_off, mut ring_cap, mut at, mut at_v) = (Vec::new(), Vec::new(), 0u64, 0u64);
        for (i, r) in d.rings.iter().enumerate() {
            let cap = if !whole && !rings_used.contains(&i) { 0 } else { r.rows(ctx, ch) };
            ring_off.push((at, at_v));
            ring_cap.push(cap);
            at += (cap * r.nk * r.d * 2) as u64;
            at_v += if r.values { (cap * r.nk * r.d * 2) as u64 } else { 0 };
        }
        let (kc, vc) = (Buf::zeroed(at as usize)?, Buf::zeroed(at_v.max(4) as usize)?);
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
        let attns = || ops().filter_map(|op| if let Op::Attn(a) = op { Some(a) } else { None });
        let ldec = attn::LibDecode::new(&lib, attns(), ch, rows_out.max(VERIFY_ROWS), ctx)?;
        // (the mixture-of-experts layers are alike: one scratch for all, made for the first's shapes and a chunk's rows)
        // (a block that runs its own launches makes it the first time it runs: `NhHost::moe_scratch`)
        let moe_sc = std::cell::OnceCell::new();
        if let Some((layer, _)) = big.moe.first().and_then(|b| b.resident()) {
            let _ = moe_sc.set(crate::moe::MoeScratch::new(&dev, &lib, &[layer], ch)?);
        }
        // the decoded-then-dense way, as Qwen3.5's engine chooses it: the library's dense product off Ada, our own on Ampere and Ada; made where a matrix takes it at a
        // chunk's rows (`decodes`: the most rows a product has; past its first row count a matrix keeps that way)
        let want = crate::tune!("GLYD_ENGINE_GEMM").unwrap_or_default();
        let gpu_code = lib.gpu()? as i64;
        let decode_from = crate::tune!("GLYD_ENGINE_DECODE_FROM").and_then(|v| v.parse().ok());
        let packed: Vec<&PackBuf> = (0..big.mats.len()).filter(|&k| k != d.lm && k != d.embed).filter_map(|k| big.mats[k].packed_opt()).collect();
        let mut decoding = false;
        if d.decoded {
            for p in &packed {
                decoding |= Dyn::decodes(&lib, gpu_code, decode_from, p, ch)?;
            }
        }
        // (the dense product's choice, by V8a's table: gpu_ops::dense_choice)
        let (try_blas, may_own) = crate::gpu_ops::dense_choice(dev.cc, &want);
        let blas = if decoding && try_blas { crate::ffi::Blas::find() } else { None };
        let own = decoding && may_own && (want == "own" || blas.is_none());
        // (the blocks' own matrices too: the experts')
        let block_shapes: Vec<(i64, i64, bool)> = big.moe.iter().flat_map(|b| b.pack_shapes()).collect();
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
        let (kglm, kmoe, hcp) = (crate::kern::KernGlm::load(&dev)?, crate::moe::MoeKern::load(&dev)?, Buf::zeroed(ch * 24 * 4)?);
        let hc_lag = Buf::zeroed(ch * 4 * 4)?;
        // (the split mixing projection's sums: as the model's streams round, one buffer for every block's)
        let round = d.layers.iter().flatten().find_map(|op| if let Op::StreamsPre { round, .. } = *op { Some(round) } else { None });
        let hc_part = Buf::zeroed(round.map_or(Ok(1), |r| residual::part_words(r, ch, d.hidden))? * 4)?;
        let kres = residual::KernRes::load(&dev)?;
        let (kmuse, moe_inv) = (muse::KernMuse::load(&dev)?, Buf::zeroed(ch * 32 * 4)?);
        let moes = || d.layers.iter().flatten().filter_map(|op| if let Op::Moe(m) = op { Some(m) } else { None });
        let ngram = d.layers.iter().flatten().any(|op| matches!(op, Op::NgramRows(_)));
        let tok = if moes().any(|m| m.hash.is_some()) || ngram { Some(Buf::zeroed(ch * 4)?) } else { None };
        let qx = d.layers.iter().flatten().any(|op| {
            matches!(
                op,
                Op::StreamsPre { mix: StreamsMix::Gated { .. }, .. }
                    | Op::GatedNorm { sigmoid: true, .. }
                    | Op::IdxMean { .. }
                    | Op::NgramRows(_)
                    | Op::NgramAdd(_)
            ) || matches!(op, Op::Attn(a) if a.picked)
        });
        let kqx = if qx { Some(crate::model_qwen4x::KernQx::load(&dev)?) } else { None };
        let picks = picked::PickRun::new(d, ch)?;
        let ones = if moes().any(|m| m.weigh_act) {
            let k = big.moe.iter().filter_map(|b| b.resident()).map(|(l, _)| l.cfg.router.top_k).max().unwrap_or(crate::moe::MAX_TOPK);
            let b = Buf::new(ch * k * 4)?;
            b.write(0, &vec![1f32; ch * k])?;
            Some(b)
        } else {
            None
        };
        let (km, mla) = (mla::KernMla::load(&dev)?, mla::MlaRun::new(d, &big.mats, ch, ctx)?);
        let v4 = match &d.v4 {
            Some(v) => Some(deepseek_v4::V4Run::new(&dev, d, v, ch, ctx)?),
            None => None,
        };
        let bytes = big.bytes() + draft.as_ref().map_or(0, |x| x.p.bytes());
        // (the Mamba-2 scan's chunked workspace, for its most heads and groups)
        let ssd = ops()
            .filter_map(|op| if let Op::MambaScan { nh, ng, .. } = *op { Some((nh, ng)) } else { None })
            .reduce(|a, b| (a.0.max(b.0), a.1.max(b.1)))
            .map(|(nh, ng)| SsdWs::sized(nh, ng, ch))
            .transpose()?;
        let mut e = Dyn {
            dev,
            lib,
            kern,
            kg,
            k35,
            knh,
            ka,
            ldec,
            km,
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
            ssd,
            moe_sc,
            start: std::cell::Cell::new(0),
            prepped: std::cell::Cell::new(false),
            ws,
            done,
            blas,
            own,
            scratch_w,
            gpu_code,
            decode_from,
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
            // (GLYD_PROF=1: every pass's operations timed, the step graph's inside it)
            prof: prof::on().then(Default::default),
            gmarks: Vec::new(),
            spare: Default::default(),
            dump: match crate::tune!("GLYD_OP_DUMP") {
                Some(path) => {
                    static ENGINES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
                    let n = ENGINES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let f = std::fs::File::create(format!("{path}.{n}")).map_err(|e| Error(format!("{path}.{n}: {e}")))?;
                    Some(std::cell::RefCell::new((f, 0)))
                }
                None => None,
            },
            replayed: std::cell::Cell::new(false),
            capturing: std::cell::Cell::new(false),
            queued: std::cell::Cell::new(0.0),
            cands: Default::default(),
            stage,
            v4,
            kglm,
            kmoe,
            kres,
            hcp,
            hc_lag,
            hc_part,
            kmuse,
            moe_inv,
            tok,
            ones,
            kqx,
            trace: std::cell::RefCell::new(None),
            op_trace: std::cell::RefCell::new(None),
            route_trace: std::cell::RefCell::new(None),
            picks,
            mla,
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
            lead: std::cell::Cell::new(0),
            rgraphs: Vec::new(),
            rgraph_on: false,
            rplayed: std::cell::Cell::new(None),
            rows_replays: std::cell::Cell::new(0),
            pieces: Default::default(),
            pmode: std::cell::Cell::new(pieces::Mode::Eager),
            pcap: Default::default(),
            popen: std::cell::Cell::new(false),
            pplayed: std::cell::Cell::new(None),
            psplice: Default::default(),
            pcount: std::cell::Cell::new(0),
            piece_replays: std::cell::Cell::new(0),
            graphs_off: std::cell::Cell::new(false),
            rows_late: std::cell::Cell::new(0),
            dcalls: std::cell::Cell::new(0),
            gcalls: 0,
            paged: None,
            kv12: None,
            lossless: None,
            swap: None,
            lanes: None,
            gaps: None,
            draws: Vec::new(),
            status_from: None,
            row_masks: Vec::new(),
            masker: None,
            sums: None,
            owed: 0,
        };
        // (a group of GPUs: every one's engine made before any one's first pass, src/dynamic/tp.rs)
        if let Some(c) = tp::comm() {
            c.barrier()?;
        }
        // (a block that reads the router's choices on the host, an expert cache's, is not captured)
        if o.graph && e.big.moe.iter().all(|b| !b.reads_host() && (b.resident().is_some() || b.graphable())) {
            e.capture_graph()?;
            e.rgraph_on = true;
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
            // (an event of the passes taken before where there is one: a pass of a thousand operations makes none)
            let e = match self.spare.borrow_mut().pop() {
                Some(e) => e,
                None => Ev::new()?,
            };
            // (a node of the graph where one is being captured, which `capture_graph` keeps with it; else recorded now)
            if self.capturing.get() {
                e.record_in_graph(self.dev.stream)?;
            } else {
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
        self.rgraphs.clear();
        self.pieces.clear();
        if self.graph.take().is_some() {
            self.capture_graph()?;
        }
        Ok(())
    }

    /// The stages' times since the last call (ms, summed by stage, in the order first seen), the eager passes' events cleared; where the last pass was the step
    /// graph's replay, its last replay's (the graph's events kept).
    pub fn prof_take(&self) -> Result<Vec<(&'static str, f64)>> {
        let (replayed, rows) = (self.replayed.replace(false), self.rplayed.replace(None));
        let (set, splice) = (self.pplayed.replace(None), std::mem::take(&mut *self.psplice.borrow_mut()));
        let Some(p) = &self.prof else { return Ok(Vec::new()) };
        self.dev.sync()?;
        let evs = std::mem::take(&mut *p.borrow_mut());
        fn refs<'a>(v: &'a [(&'static str, Ev)]) -> Vec<(&'static str, &'a Ev)> {
            v.iter().map(|(l, e)| (*l, e)).collect()
        }
        let out = match (replayed, rows, set.and_then(|i| self.pieces.sets.get(i))) {
            (true, Some(i), _) => prof::between(&refs(&self.rgraphs[i].marks))?,
            (true, None, _) => prof::between(&refs(&self.gmarks))?,
            // (a pass of pieces: each piece's events where its replay was queued among the eager ones)
            (_, _, Some(set)) => prof::between(&pieces::splice(&evs, &splice, |k| &set.pieces[k].marks))?,
            _ => prof::between(&refs(&evs))?,
        };
        self.spare.borrow_mut().extend(evs.into_iter().map(|x| x.1));
        Ok(out)
    }

    /// GLYD_PROF=1 (src/dynamic/prof.rs): the pass just run, of `rows` decode rows and `prompt` prompt tokens, added to the profile (its operations' times, the
    /// stream synchronized for them).
    fn prof_pass(&self, rows: usize, prompt: usize) -> Result<()> {
        if !prof::on() || self.prof.is_none() {
            return Ok(());
        }
        let (graph, pieces) = (self.replayed.get(), self.pcount.get());
        let ops = self.prof_take()?;
        prof::pass(rows, prompt, &ops, graph, (!graph).then(|| self.queued.get()), self.dcalls.get(), pieces);
        Ok(())
    }

    fn skinny_takes(&self, p: &PackBuf, m: usize) -> bool {
        self.skinny_on && (1..=16).contains(&m) && p.pack.rows >= 2048 && crate::skinny::SkinnyKern::takes(&p.pack)
    }

    /// The L2 prefetch of `kib` KiB of each first-wave block of `p` (where `lin` takes it skinny) and the spare blocks of `threads` that issue it (GLY-312).
    fn pf(&self, w: &Mat, m: usize, kib: usize, threads: u32) -> (NextPf, u32) {
        let Mat::Packed(p) = w else { return (NextPf::default(), 0) };
        let Some((data, scales, ds, ss)) = crate::skinny::SkinnyKern::regions(&p.pack).filter(|_| kib > 0 && self.skinny_takes(p, m)) else {
            return (NextPf::default(), 0);
        };
        let dhead = ((kib as u64 * 1024).min(ds) / 128 * 128) as i32;
        let shead = ((dhead as u64 * ss / ds).div_ceil(128) * 128).min(ss) as i32;
        let regions = ((p.pack.rows / 64) as i32).min(2 * self.dev.sms);
        let lines = regions as u32 * ((dhead + shead) >> self.pf_shift) as u32;
        (NextPf { data, scales, dstride: ds as i64, sstride: ss as i64, dhead, shead, regions, shift: self.pf_shift }, lines.div_ceil(4 * threads))
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
            self.mark(DECODED)?;
            if let Some(b) = &self.blas {
                return b.gemm_nt(s, self.scratch_w.ptr, x, y, p.pack.rows as i32, p.pack.cols as i32, m as i32);
            }
            let tile = self.kern.pick_tile(m, p.pack.rows as usize, p.pack.cols as usize);
            return self.kern.gemm(s, tile, x, self.scratch_w.ptr, y, m as i32, p.pack.rows as i32, p.pack.cols as i32);
        }
        self.lib.linear(&p.pack, x, m as i64, y, self.ws.ptr, self.ws.bytes, self.done.ptr, s)
    }

    /// Whether a product of `m` rows decodes the matrix and multiplies dense (ModelDesc::decoded, where the model has that way: `decodes`), as Qwen3.5's engine.
    fn decoded_way(&self, p: &PackBuf, m: usize) -> Result<bool> {
        if self.blas.is_none() && !self.own {
            return Ok(false);
        }
        Dyn::decodes(&self.lib, self.gpu_code, self.decode_from, p, m)
    }

    /// Whether a product of `m` rows by `p` on the GPU of code `gpu` is the decoded matrix's (W unpacked to bf16 once, then a dense product): the library's route
    /// DECODE or AHEAD where its table has a crossover for the pack's layout on that GPU (as measured there: an A100's 12-bit layout from 769 rows, an L4's,
    /// GeForce Ada's, an A10's, an L40S's, every Hopper's); where it has none (the prompt kernel at any length: an A100's tiered and lossy layouts, Ampere GeForce,
    /// Blackwell), from DECODE_FROM rows for a layout that is not the 12-bit one. A development build's GLYD_ENGINE_DECODE_FROM=N replaces both: from N rows
    /// decoded, below them the library's kernels (a large N: never). K a multiple of 64 and the rows of 128 (the dense products' tiles), more than 64 rows.
    fn decodes(lib: &crate::ffi::Lib, gpu: i64, from: Option<usize>, p: &PackBuf, m: usize) -> Result<bool> {
        let (o, k, twelve) = (p.pack.rows, p.pack.cols, p.pack.is_twelve());
        if m <= 64 || k % 64 != 0 || o % 128 != 0 {
            return Ok(false);
        }
        decodes_by(m, twelve, from, &|rows| lib.route(gpu, twelve, o, k, rows))
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
        // (a profile's first event: the embedding is timed by the pass's own first, `forward`'s; and a served step's first work, `Gaps`)
        self.mark("start")?;
        self.gap_begin()?;
        let hid = self.big.d.hidden;
        let s = self.dev.stream;
        let Some(e) = self.big.mats[self.big.d.embed].packed_opt().map(|p| &p.pack) else { return err("the embedding is not packed") };
        let x = self.big.slot(Slot::X);
        if let Some(&t) = tokens.iter().find(|&&t| t as usize >= self.big.d.vocab) {
            return err(format!("token {t}: the vocabulary has {}", self.big.d.vocab));
        }
        if let Some(v) = &self.tok {
            v.write(0, &tokens.iter().map(|&t| t as i32).collect::<Vec<i32>>())?;
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
            Op::Moe(MoeOp { block, .. }) => block,
            _ => return Ok(()),
        };
        let Some(layer) = p.moe[block].core() else {
            crate::hb_usage::missed(l);
            return Ok(());
        };
        let sc = self.moe_sc.get().ok_or_else(|| Error("no mixture-of-experts scratch".into()))?;
        self.dev.sync()?;
        let mut ids = vec![0i64; n * layer.cfg.router.top_k];
        sc.ids.read(0, &mut ids)?;
        crate::hb_usage::record(l, layer.cfg.experts, &ids)
    }

    /// The gated residuals', the n-gram embedding's and the mean-pooled indexer's kernels (kernels/qwen4x.cu), where the description has those kinds.
    fn kqx(&self) -> Result<&crate::model_qwen4x::KernQx> {
        self.kqx.as_ref().ok_or_else(|| Error("a gated residual, n-gram embedding or mean-pooled indexer in a model loaded without its kernels".into()))
    }

    /// The gated residuals' mix of the streams S into H on n rows (`StreamsMix::Gated`, `HeadMix::Gated`): each stream normed by `norm` into Hq, `down`'s product into
    /// Mid and its activation, `up`'s into Cp, the gated mean into H; with `inject`, its product into Kv and the streams' weights into hcp.
    unsafe fn gated_mix(&self, p: &Part, n: usize, norm: usize, down: usize, up: usize, inject: Option<usize>) -> Result<()> {
        let (k, s, d, hc) = (self.kqx()?, self.dev.stream, p.d.hidden, p.d.streams());
        let (sn, mid, gw) = (p.slot(Slot::Hq), p.slot(Slot::Mid), p.slot(Slot::Cp));
        k.hc_norm(s, p.slot(Slot::S), p.small[norm].ptr, sn, n, d, p.d.eps, hc)?;
        self.lin(&p.mats[down], sn, n, mid)?;
        k.silu4(s, mid, n * p.d.width(Slot::Mid), hc)?;
        self.lin(&p.mats[up], mid, n, gw)?;
        let (il, inj) = match inject {
            Some(w) => {
                self.lin(&p.mats[w], sn, n, p.slot(Slot::Kv))?;
                (p.slot(Slot::Kv), self.hcp.ptr)
            }
            None => (0, 0),
        };
        k.hc_mix(s, gw, sn, il, p.slot(Slot::H), inj, n, d, hc)
    }

    unsafe fn op(&self, p: &Part, l: usize, i: usize, op: &Op, n: usize, pos: (u64, u64), head: usize) -> Result<()> {
        let s = self.dev.stream;
        let (nu, eps, hid) = (n as u32, p.d.eps, p.d.hidden as i32);
        let nt = norm_threads(hid)?;
        match *op {
            Op::EmbedNorm { scale, norm } => {
                launch!(self.kg.embed_norm, [nu, 1, 1], [nt, 1, 1], s; p.slot(Slot::X), scale, p.small[norm].ptr, p.slot(Slot::H), hid, eps)?;
                self.mark("norms")
            }
            Op::Linear { w, x, y, name } => {
                self.lin(&p.mats[w], p.slot(x), n, p.slot(y))?;
                self.mark(name)
            }
            Op::F8Lin { w, x, y, name } => {
                p.f8[w].linear(&self.lib, p.slot(x), n, p.slot(y), self.ws.ptr, self.ws.bytes, self.done.ptr, s)?;
                self.mark(name)
            }
            Op::Attn(ref a) => match a.picked {
                true => self.attn_picked(p, a, n, pos.0),
                false => self.attn(p, l, i, a, n, pos, head),
            },
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
            Op::Norm { x, w, y } => {
                launch!(self.kg.norm, [nu, 1, 1], [nt, 1, 1], s; p.slot(x), p.small[w].ptr, p.slot(y), hid, eps)?;
                self.mark("norms")
            }
            Op::DraftIn { scale } => {
                let bh = self.big.d.hidden as i32;
                launch!(self.kg.draft_in, [((2 * bh) as u32).div_ceil(256).min(64), 1, 1], [256, 1, 1], s; self.big.slot(Slot::X), scale, p.slot(Slot::Hid), p.slot(Slot::In), bh)
            }
            Op::RmsNorm { x, w, y, weight, eps } => {
                // (the slot's own width: the hidden size, or a narrower activation's)
                let (k, (x, w, y)) = (p.d.width(x) as i32, (p.slot(x), p.small[w].ptr, p.slot(y)));
                match weight {
                    NormW::Plain => self.kern.rmsnorm(s, x, w, y, nu, k, eps)?,
                    NormW::PlusOne => self.k35.rmsnorm(s, x, w, y, nu, k, eps)?,
                    NormW::Input => self.kres.rms_in(s, x, w, y, nu, k, eps)?,
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
            Op::GatedNorm { w, row, nv, z_off, sigmoid } => {
                let (core, z) = (p.slot(Slot::Core), p.slot(Slot::Mix) + (z_off * 2) as u64);
                if sigmoid {
                    self.kqx()?.gnorm(s, core, z, row, p.small[w].ptr, n, nv, eps)?;
                } else {
                    self.k35.gated_norm(s, core, z, row as i32, p.small[w].ptr, nu, nv as i32, eps)?;
                }
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
                        self.mark(DECODED)?;
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
                let (core, xc) = (p.slot(Slot::Core) + self.rows_before(nh * crate::kern::NH_P), p.slot(Slot::Xc) + self.rows_before(cd));
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
            Op::Moe(ref m) => self.moe(p, m, n),
            Op::Streams => self.kres.streams(s, p.slot(Slot::X), p.slot(Slot::S), nu, hid, p.d.streams()),
            Op::StreamsPre { mix: StreamsMix::Hc { f, base, scale }, norm, round } => {
                let (fp, bp, sp) = (p.small[f].ptr, p.small[base].ptr, p.small[scale].ptr);
                let (h, nw, (hc_eps, iters)) = (p.slot(Slot::H), p.small[norm].ptr, p.d.hc);
                self.kres.pre(s, p.slot(Slot::S), (fp, bp, sp), nw, h, self.hcp.ptr, self.hc_part.ptr, nu, hid, (eps, hc_eps, iters), round)?;
                self.mark("hyper-connections")
            }
            Op::StreamsPre { mix: StreamsMix::HcLagged { f, base, scale, first }, norm, round } => {
                let (fp, bp, sp) = (p.small[f].ptr, p.small[base].ptr, p.small[scale].ptr);
                let (h, nw, (hc_eps, iters)) = (p.slot(Slot::H), p.small[norm].ptr, p.d.hc);
                self.kres.pre(s, p.slot(Slot::S), (fp, bp, sp), nw, h, self.hcp.ptr, self.hc_part.ptr, nu, hid, (eps, hc_eps, iters), round)?;
                // (H again, from the previous lagged block's weights; this block's kept in hc_lag for the next)
                self.kres.lag(s, p.slot(Slot::S), self.hc_lag.ptr, self.hcp.ptr, nw, h, nu, hid, eps, (first, true), round)?;
                self.mark("hyper-connections")
            }
            Op::StreamsHead { mix: HeadMix::LastPre, norm, round } => {
                self.kres.lag(s, p.slot(Slot::S), self.hc_lag.ptr, self.hcp.ptr, p.small[norm].ptr, p.slot(Slot::H), nu, hid, eps, (false, false), round)?;
                self.mark("hyper-connections")
            }
            Op::StreamsPre { mix: StreamsMix::Gated { down, up, inject }, norm, .. } => {
                self.gated_mix(p, n, norm, down, up, Some(inject))?;
                self.mark("gated residuals")
            }
            Op::StreamsPost { gated: true, .. } => {
                self.kqx()?.hc_post(s, p.slot(Slot::S), p.slot(Slot::O), self.hcp.ptr, n, p.d.hidden, p.d.streams())?;
                self.mark("gated residuals")
            }
            Op::StreamsPost { round, .. } => {
                self.kres.post(s, p.slot(Slot::S), p.slot(Slot::O), self.hcp.ptr, nu, hid, round)?;
                self.mark("hyper-connections")
            }
            Op::StreamsHead { mix: HeadMix::Gated { down, up }, norm, .. } => {
                self.gated_mix(p, n, norm, down, up, None)?;
                self.mark("gated residuals")
            }
            Op::StreamsHead { mix, norm, round } => {
                let w = if let HeadMix::Weighted([f, b, sc]) = mix { Some((p.small[f].ptr, p.small[b].ptr, p.small[sc].ptr)) } else { None };
                self.kres.head(s, p.slot(Slot::S), w, self.hc_part.ptr, p.small[norm].ptr, p.slot(Slot::H), nu, hid, (eps, p.d.hc.0), round)?;
                self.mark("hyper-connections")
            }
            Op::IdxMean { heads, dim, block, q_norm, k_norm, rope, keys, pools } => {
                self.idx_mean(p, (heads, dim, block), q_norm, k_norm, rope, keys, pools, n, pos.0)
            }
            Op::IdxPick(ref x) if matches!(x.keys, IdxKeys::Pools(_)) => self.idx_pick_pools(p, x, n, pos.0),
            Op::NgramRows(ref x) => self.ngram_rows(p, x, n),
            Op::NgramAdd(ref x) => self.ngram_add(p, x, n),
            Op::KdaConv { w, state, c, row, w32 } => {
                let (x, y) = (p.slot(Slot::Mix) + self.rows_before(row), p.slot(Slot::Xc) + self.rows_before(c));
                self.kglm.conv(s, x, row as i32, p.small[w].ptr, w32, self.state(state), y, n as i32, c as i32)?;
                self.mark("linear attention: convolution")
            }
            Op::KdaDelta { a_log, dt_bias, state, heads, bound, ba } => {
                // (in a step of several sequences the gates' rows are the copied slot's (Op::Parts ran on all the rows), the sequence's at its offset)
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
            Op::Parts { row, parts } => {
                if n > 1 {
                    for (at, w, slot) in parts {
                        self.kern.parts_rows(s, p.slot(Slot::Mix) + (at * 2) as u64, row, p.slot(slot), w, w, n)?;
                    }
                }
                Ok(())
            }
            Op::Mla(ref a) => self.mla(p, a, n, pos.0),
            Op::DepthMix(m) => {
                let (blk, hd) = (p.slot(Slot::Blk), hid as usize);
                let o = if m.add { p.slot(Slot::O) } else { 0 };
                let at = m.push.map_or(0, |b| blk + (b * hd * 2) as u64);
                let (wn, wp, w) = (p.small[m.w_n].ptr, p.small[m.w_p].ptr, p.small[m.w].ptr);
                self.kres.depth_mix(
                    s,
                    blk,
                    p.d.width(Slot::Blk),
                    m.nb,
                    p.slot(Slot::X),
                    o,
                    (wn, wp),
                    at,
                    w,
                    p.slot(Slot::H),
                    p.d.width(Slot::H),
                    hd,
                    m.eps,
                    nu,
                )?;
                self.mark("depth mix and norm")
            }
            Op::Add { x } => {
                self.kres.add(s, p.slot(Slot::X), p.slot(x), n * hid as usize)?;
                self.mark("residual add")
            }
            Op::MoeEarly { block, mix: m } => {
                if n != 1 || !p.moe[block].early_wanted() {
                    return Ok(());
                }
                let (h, hd) = (p.slot(Slot::Att), hid as usize);
                let (wn, wp, w) = (p.small[m.w_n].ptr, p.small[m.w_p].ptr, p.small[m.w].ptr);
                self.kres.depth_mix(s, p.slot(Slot::Blk), p.d.width(Slot::Blk), m.nb, p.slot(Slot::X), 0, (wn, wp), 0, w, h, hd, hd, m.eps, 1)?;
                p.moe[block].early(self, h)?;
                self.mark("experts: the early guess")
            }
            Op::GatedAct { inter, act } => {
                self.kmoe.act(s, p.slot(Slot::Gu), p.slot(Slot::M), n, inter, &act)?;
                self.mark("MLP: activation")
            }
            Op::Nop => Ok(()),
            Op::GatedResidual { alpha, beta } => {
                self.kres.gate(s, p.slot(Slot::X), p.slot(Slot::O), p.small[alpha].ptr, p.small[beta].ptr, nu, hid)?;
                self.mark("residual gates")
            }
            Op::Fp8 { .. } => self.op_v4(p, op, n),
            Op::CompQ { .. }
            | Op::CompKv { .. }
            | Op::Compress { .. }
            | Op::IdxQ { .. }
            | Op::IdxKeys { .. }
            | Op::IdxPick(_)
            | Op::CompAttn { .. }
            | Op::OGroupLora { .. }
            | Op::OGroupLoraStack { .. } => self.op_comp(p, op, n, pos.0),
        }
    }

    /// A mixture of experts (`MoeOp`) on n rows.
    unsafe fn moe(&self, p: &Part, m: &MoeOp, n: usize) -> Result<()> {
        let s = self.dev.stream;
        let b = &p.moe[m.block];
        let (h, x, y) = (p.slot(Slot::H), p.slot(m.x), p.slot(m.y));
        let plain = m.expert_norm.is_none() && !m.weigh_act;
        if (plain && (m.x != Slot::H || m.logit_scale != 1.0)) || (m.hash.is_some() && !m.weigh_act) {
            return err(format!(
                "{m:?}: an experts' input apart from the router's, a router's logit scale and hash routing are the normed or the weighed experts'"
            ));
        }
        let Some((layer, fc)) = b.resident() else {
            if !plain {
                return err(format!("{m:?}: normed or weighed experts are all on the GPU"));
            }
            // (experts in a cache, or at the hummingbird level: the block's own launches, over this engine's buffers)
            b.forward(self, h, n, y)?;
            return self.mark("mixture of experts");
        };
        let sc = self.moe_sc.get().ok_or_else(|| Error("no mixture-of-experts scratch".into()))?;
        if let Some((post_w, pe)) = m.expert_norm {
            layer.forward_normed(&self.lib, sc, s, h, x, n, y, layer.w.router.ptr, m.logit_scale, p.small[post_w].ptr, pe, self.moe_inv.ptr)?;
        } else if m.weigh_act {
            let hash = match m.hash {
                Some(t) => Some((p.small[t].ptr, self.tok.as_ref().ok_or_else(|| Error("hash routing without the tokens' buffer".into()))?.ptr)),
                None => None,
            };
            let ones = self.ones.as_ref().ok_or_else(|| Error("weighed experts without the weights of one".into()))?;
            layer.forward_weighed(&self.lib, sc, s, h, x, n, y, hash, ones.ptr, m.fp8)?;
        } else {
            match fc {
                None => layer.forward(&self.lib, sc, s, h, n, y)?,
                Some((fc1, fc2)) => {
                    layer.route(sc, s, h, layer.cfg.router_hidden, n)?;
                    self.lin(fc1, h, n, p.slot(Slot::Mix))?;
                    layer.experts(&self.lib, sc, s, p.slot(Slot::Mix), n, p.slot(Slot::Xc))?;
                    self.lin(fc2, p.slot(Slot::Xc), n, y)?;
                    layer.add_shared(&self.lib, sc, s, h, n, y)?;
                }
            }
        }
        self.mark("mixture of experts")
    }

    /// DeepSeek-V4's activations to FP8 as the release's products quantize their input (kernels/dsv4.cu) on n rows.
    unsafe fn op_v4(&self, p: &Part, op: &Op, n: usize) -> Result<()> {
        let v = self.v4.as_ref().ok_or_else(|| Error("a DeepSeek-V4 operation in a model loaded without its buffers".into()))?;
        let (s, nu) = (self.dev.stream, n as u32);
        let k = &v.k;
        match *op {
            Op::Fp8 { x, y, .. } if !v.dims.sim => {
                crate::dev::copy_d2d_async(p.slot(y), p.slot(x), n * p.d.width(x) * 2, s)?;
                self.mark("activations to FP8")
            }
            Op::Fp8 { x, y, block } => {
                let w = p.d.width(x);
                launch!(k.fp8, [nu, ((w / block) as u32).div_ceil(4), 1], [128, 1, 1], s; p.slot(x), p.slot(y), w as i32, block as i32)?;
                self.mark("activations to FP8")
            }
            _ => unreachable!(),
        }
    }

    /// Part `p`'s layers on the n rows (positions `pos`), then on the last `head` rows its post projection (a draft model's, into Hid), its LM head, the
    /// soft-cap and the argmax (its logits, tok, val a row each).
    /// GLYD_OP_DUMP's line of each slot after operation `i` of layer `l` (`usize::MAX`: the first operation) on the pass's `n` rows: its bytes hashed (FNV-1a).
    fn dump_op(&self, p: &Part, l: usize, i: usize, op: &Op, n: usize) -> Result<()> {
        let Some(d) = &self.dump else { return Ok(()) };
        use std::io::Write;
        self.dev.sync()?;
        let mut d = d.borrow_mut();
        let pass = d.1;
        let name = rows::op_name(op);
        for (k, b) in p.slots.iter().enumerate() {
            let w = (n * p.d.widths[k]).min(b.bytes / 2);
            if w == 0 {
                continue;
            }
            let mut v = vec![0u16; w];
            b.read(0, &mut v)?;
            let h = v.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &x| (h ^ x as u64).wrapping_mul(0x0100_0000_01b3));
            writeln!(d.0, "{pass} {} {i} {name} {k} {h:016x}", if l == usize::MAX { -1 } else { l as i64 }).map_err(|e| Error(e.to_string()))?;
        }
        Ok(())
    }

    unsafe fn forward(&self, p: &Part, n: usize, pos: (u64, u64), head: usize) -> Result<()> {
        let (s, queue) = (self.dev.stream, std::time::Instant::now());
        // (a pass of pieces (src/dynamic/pieces.rs): a capture queues nothing but its pieces' operations, captured; a replay launches each piece's graph in its place)
        let mode = if std::ptr::eq(p, &self.big) { self.pmode.get() } else { pieces::Mode::Eager };
        // (a capture or a warm-up of the pieces queues nothing but their operations: no preparation, no operation between them, no head)
        let capture = matches!(mode, pieces::Mode::Capture | pieces::Mode::Warm);
        let mut pc = pieces::Run { mode, k: 0, open: false };
        self.pcount.set(0);
        if !capture {
            self.dcalls.set(0);
            self.mark("embedding")?;
        }
        let st = if std::ptr::eq(p, &self.big) { self.stage } else { Stage::whole(p.d.layers.len()) };
        if !capture {
            self.pass_prepare(p, n, pos)?;
        }
        let head = if st.last { head } else { 0 };
        // (GLYD_OP_DUMP: a pass's number; a capture's operations are not run, so not hashed: the pieces' and the step graphs')
        let dump = self.dump.is_some() && !capture && !self.capturing.get();
        if let Some(d) = self.dump.as_ref().filter(|_| dump) {
            d.borrow_mut().1 += 1;
        }
        if st.first && !self.in_piece(&mut pc, p, 0, 0, &p.d.first, n, pos, head)? {
            self.op(p, 0, 0, &p.d.first, n, pos, head)?;
            if dump {
                self.dump_op(p, usize::MAX, 0, &p.d.first, n)?;
            }
        }
        // (a development build's GLYD_OP_PROBE: after each op, the slots' first rows read back; the first op that leaves a slot not finite named)
        let probe = crate::tune!("GLYD_OP_PROBE").is_some();
        let mut seen = false;
        for (l, ops) in p.d.layers.iter().enumerate().take(st.layers.1).skip(st.layers.0) {
            // (a layer whose matrices are in the computer's memory: copied into the slot on this stream first, a node of the step's graph; no pass of pieces has such layers)
            if let Some(r) = &p.ram {
                r.fetch(l, s)?;
            }
            for (i, op) in ops.iter().enumerate() {
                if self.in_piece(&mut pc, p, l, i, op, n, pos, head)? {
                    continue;
                }
                // (a preparation whose keys and values go to 12-bit pages writes staging rows and no sequence's ring: the pass's rows at once, and one
                // write of them; the slots of each sequence's rows the pass made)
                if !self.parts.is_empty() && !rows::row_wise(op) {
                    // (the pass's first rows that decode, a token of a sequence each: an attention layer's, or a recurrent layer's operation that takes them (`rows::rows_state`),
                    // for all of them at once, each through its own table or state, which is what each row's own pass computes, bit for bit (the decode's plan is a sequence's
                    // own; the recurrent kernels the one-sequence ones' on a token))
                    let lead = match op {
                        Op::Attn(a) if !a.picked => self.lead.get(),
                        op if rows::rows_state(op) => self.lead.get(),
                        _ => 0,
                    };
                    let mut r = Ok(());
                    if let Op::Attn(a) = op {
                        if a.writes && self.pages12(p, a.ring).is_some() {
                            self.attn_prep(p, l, i, a, n, pos, head)?;
                            self.prepped.set(true);
                        }
                        if lead > 0 {
                            r = self.attn_lead(p, l, i, a, (lead, n), pos, head);
                        }
                    } else if lead > 0 {
                        r = self.state_lead(p, op, lead);
                    }
                    // (an operation on a sequence's cache or state: each sequence's rows by themselves, as a pass of that sequence alone runs them)
                    for (k, part) in self.parts.iter().enumerate().skip(lead) {
                        if r.is_err() {
                            break;
                        }
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
                    self.trace_op(p, l, op, n)?;
                    continue;
                }
                let input = self.route_input(p, op, n)?;
                self.op(p, l, i, op, n, pos, head)?;
                self.trace_op(p, l, op, n)?;
                self.trace_route(p, l, op, n, input)?;
                if dump {
                    self.dump_op(p, l, i, op, n)?;
                }
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
            if std::ptr::eq(p, &self.big) && self.tracing() {
                let r = if p.d.width(Slot::S) > 0 { Slot::S } else { Slot::X };
                let mut v = vec![0u16; n * p.d.width(r)];
                copy_d2h_async(v.as_mut_ptr() as *mut _, p.slot(r), v.len() * 2, s)?;
                self.dev.sync()?;
                if let Some(t) = self.trace.borrow_mut().as_mut() {
                    t.push(v);
                }
            }
        }
        self.piece_close(&mut pc)?;
        if capture {
            // (the pieces end at the last layer: the head is eager, its rows the pass's own)
            return Ok(());
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
        if std::ptr::eq(p, &self.big) {
            self.gap_end()?;
        }
        self.queued.set(queue.elapsed().as_secs_f64() * 1e3);
        Ok(())
    }

    /// What a pass of part `p` on n rows at `pos` makes once for every layer before its operations run: the tables of its rows that decode, each its sequence's
    /// (`rows::RowTabs`), the 12-bit pages' slots, and the rows' lengths and first tokens the attention's decode reads. Every caller of `op` runs it first (`forward`, and
    /// kindcheck's operation-by-operation passes).
    unsafe fn pass_prepare(&self, p: &Part, n: usize, pos: (u64, u64)) -> Result<()> {
        if !std::ptr::eq(p, &self.big) {
            return Ok(());
        }
        if !self.parts.is_empty() && self.lead.get() > 0 {
            self.row_tables(pos.0)?;
        }
        if self.lossless.is_some() {
            self.kv12_prepare(n, pos)?;
        }
        self.lib_decode_prepare(n, pos)?;
        self.mark("attention: the rows' lengths")
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
        // (--prof: the warm-up's events given back; the capture's, nodes of the graph, kept with it, the old graph's after it is gone)
        if let Some(p) = &self.prof {
            let evs = std::mem::take(&mut *p.borrow_mut());
            self.spare.borrow_mut().extend(evs.into_iter().map(|x| x.1));
        }
        self.capturing.set(true);
        let g = self.dev.capture(|| {
            // SAFETY: as above.
            unsafe { self.forward(&self.big, 1, (st, st), 1) }
        });
        self.capturing.set(false);
        self.graph = Some(g?);
        self.gcalls = self.dcalls.get();
        if let Some(p) = &self.prof {
            self.gmarks = std::mem::take(&mut *p.borrow_mut());
        }
        self.replayed.set(false);
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

    /// What a cut between two layers carries (docs/range-worker.md): `handed_over` of the description, each `[rows, width]` bf16, one after the other.
    pub fn boundary(&self) -> Vec<(Slot, usize)> {
        handed_over(&self.big.d)
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
                    let w = ops.iter().flat_map(Op::mats).map(|k| p.mats[k].bytes()).sum();
                    let mut rings: Vec<usize> = ops.iter().flat_map(Op::rings).collect();
                    rings.dedup();
                    let mut states: Vec<usize> = ops.iter().flat_map(Op::states).collect();
                    states.sort_unstable();
                    states.dedup();
                    (w, rings.iter().map(|&r| p.d.rings[r].token_bytes()).sum(), states.iter().map(|&s| self.states[s].bytes).sum())
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
                Some(g) if one && !self.graphs_off.get() && !self.tracing() => {
                    g.launch(self.dev.stream)?;
                    self.replayed.set(true);
                    self.rplayed.set(None);
                    self.dcalls.set(self.gcalls);
                }
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
        // (GLYD_PROF=1: a stage's pass, a decode row or a chunk's rows; a verify pass's rows as a chunk's)
        self.prof_pass((n == 1) as usize, if n == 1 { 0 } else { n })
    }

    /// A verify pass (drafted tokens checked, docs/range-worker.md): `stage_rows` on 1 to VERIFY_ROWS rows, each row as a decode step runs it (the delta rule one row
    /// after another, the decode step's attention, products of at most 8 rows), so each row's outputs are the step's at its position, bit for bit; and what `rewind`
    /// needs kept (the states before the pass, each linear-attention layer's inputs). One row is a plain step (the graph): `rewind` keeps it. A description
    /// with no operation `unverifiable_op` names.
    pub fn verify_rows(&mut self, tokens: &[u32], at: usize, io: &mut Vec<u16>, head: usize) -> Result<()> {
        let n = tokens.len();
        if let Some(op) = unverifiable_op(&self.big.d) {
            return err(format!(
                "a verify pass on {}: its operation {} is not run row after row as steps, or not gone back in by rewind",
                self.big.d.family,
                rows::op_name(op)
            ));
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
    /// activations and workspaces stay shared): `select` picks the one the next passes run on. A description every operation of which keeps a sequence's
    /// data in its rings and states (`rows::one_sequence_op`), with no draft model; after `set_head_rows`.
    pub fn add_seqs(&mut self, k: usize) -> Result<()> {
        if k > 0 {
            if self.draft.is_some() {
                return err(format!("more sequences than one on {}: not with a draft model", self.big.d.family));
            }
            if let Some(op) = rows::one_sequence_op(&self.big.d) {
                return err(format!(
                    "more sequences than one on {}: its operation {} keeps a sequence's data outside the rings and states each sequence holds",
                    self.big.d.family,
                    rows::op_name(op)
                ));
            }
        }
        for _ in 0..k {
            let states = self.states.iter().map(|b| Buf::zeroed(b.bytes)).collect::<Result<_>>()?;
            // (paged: every sequence's keys and values are in the pools, kc and vc; its own are its block tables)
            let (room, room_v) = if self.paged.is_some() { (4, 4) } else { (self.kc.bytes, self.vc.bytes) };
            self.seqs.push(Some(Seq {
                kc: Buf::zeroed(room)?,
                vc: Buf::zeroed(room_v)?,
                states,
                n: 0,
                pending: None,
                verify: None,
                graph: None,
                gmarks: Vec::new(),
            }));
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
        std::mem::swap(&mut self.gmarks, &mut s.gmarks);
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
        // (a step's rows that decode take their attention at once (`attn_lead`): the parts kernel's room and the library decode's workspace for that many)
        let (heads, dmax) = self.big.d.heads();
        self.part = Buf::zeroed(rows.max(self.split_rows) * heads.max(1) * attn::SPLITS * (dmax + 2) * 4)?;
        if let Some(ld) = &mut self.ldec {
            ld.grow(&self.lib, rows)?;
        }
        self.rgraphs.clear();
        self.pieces.clear();
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
        let (mut at, timed) = (0, prof::begin());
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
            self.prof_pass(0, n)?;
        }
        // (a pass outside the scheduler, which would take a lost step back: an error)
        self.kv12_check()?;
        let s = self.read_sample()?;
        self.end_expert_step()?;
        prof::end(timed, 0, tokens.len());
        Ok(s)
    }

    /// The selected sequence's blocks grown to hold its tokens before `upto`, where its keys and values are paged: a pass outside the scheduler (`reply`,
    /// `prefill`, `step`) wrote every token to block 0, which a table names until it grows, so a paged executor's replies were not the model's.
    fn hold_blocks(&mut self, upto: usize) -> Result<()> {
        // (the pass before this one is queued: what its growth promised of the 12-bit pools is in their counts when `kv12_room` waits for the stream and reads them)
        self.kv12_stepped();
        if self.paged.is_some() && !crate::sched::ModelStep::grow(self, self.seq as i32, upto)? {
            return err(format!("the paged cache has no block left for token {upto} of the sequence"));
        }
        Ok(())
    }

    /// The expert caches' stores the engine's layers have experts in (src/offload.rs `ExpertStore`), each once.
    pub fn stores(&self) -> Vec<std::rc::Rc<crate::offload::ExpertStore>> {
        let mut out: Vec<std::rc::Rc<crate::offload::ExpertStore>> = Vec::new();
        for s in self.big.moe.iter().filter_map(|b| b.store()) {
            if !out.iter().any(|o| std::rc::Rc::ptr_eq(o, &s)) {
                out.push(s);
            }
        }
        out
    }

    /// A step's end for the stores' statistics (`ExpertStore::end_step`; the step's results read): nothing where no layer has experts in a cache.
    pub(crate) fn end_expert_step(&self) -> Result<()> {
        self.stores().iter().try_for_each(|s| s.end_step().map(|_| ()))
    }

    /// The last step's statistics of the layers whose experts are in a cache, a store's layers after another's (`ExpertStore::last_step`): what a step returns beside its samples.
    pub fn expert_stats(&self) -> Vec<crate::offload::LayerStats> {
        self.stores().iter().flat_map(|s| s.last_step()).collect()
    }

    /// One token through the model (the graph where there is one); the token after it.
    pub fn step(&mut self, token: u32) -> Result<Sample> {
        if self.n + 1 > self.ctx {
            return err("the cache is full");
        }
        let timed = prof::begin();
        self.hold_blocks(self.n + 1)?;
        self.dev.sync()?;
        self.st.write(0, &[self.n as i32])?;
        let st = self.st.ptr;
        // SAFETY: live buffers; the graph was captured on them.
        unsafe {
            self.embed(&[token])?;
            match &self.graph {
                Some(g) if !self.graphs_off.get() && !self.tracing() => {
                    g.launch(self.dev.stream)?;
                    self.replayed.set(true);
                    self.rplayed.set(None);
                    self.dcalls.set(self.gcalls);
                    self.gap_end()?;
                }
                _ => self.forward(&self.big, 1, (st, st), 1)?,
            }
        }
        self.n += 1;
        let s = self.read_sample()?;
        self.end_expert_step()?;
        self.prof_pass(1, 0)?;
        prof::end(timed, 1, 0);
        Ok(s)
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
        // (GLYD_PROF=1: a drafting round as a pass of its rows, the draft model's passes in it)
        self.prof_pass(n, 0)?;
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
        let state_type = if self.states.is_empty() { "none" } else { state_type(&self.big.d) };
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

    fn reply(&mut self, prompt: &[u32], o: &crate::sched::Reply, each: &mut dyn FnMut(u32) -> bool) -> Result<(Vec<u32>, crate::sched::Finish)> {
        use crate::sched::Finish;
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
            let sh = ck.get(t).map(|(_, x)| x.shape.clone()).ok_or_else(|| Error(format!("no tensor {t}")))?;
            let (rows, cols) = (sh[0] as usize, sh.get(1).copied().unwrap_or(1) as usize);
            (crate::weights::item_words(&ck, &crate::weights::Item { members: vec![t.to_string()], rows, cols })?, rows, cols)
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The Kimi delta attention's small tensors as the executor's loader holds them, on the random Kimi-K3's description: A_log and dt_bias read in fp32,
    /// the convolutions and the gated norm in the type their operations' flags say, each at least as long as its operation reads (a head, a channel, a
    /// convolution's taps a channel, a head's channels); the forget gate's second product, multiplied by where the first leaves its input, is one of the
    /// stage's matrices; and the three convolutions are read as one, each as saved, refused where they are not alike.
    #[cfg(feature = "dev")]
    #[test]
    fn the_kimi_delta_attentions_small_tensors_are_read_as_its_operations_say() {
        use crate::runtime::describe::Source;
        let dir = std::env::temp_dir().join(format!("glyd-kda-small-{}", std::process::id()));
        crate::kimi_ref::random(&dir, 1, true, 8).unwrap();
        let c = crate::kimi::ConfigKimi::load(&dir).unwrap();
        let ck = Checkpoint::open(&dir).unwrap();
        let w32 = kimi::dtypes(&ck, &c).unwrap();
        let st = Stage::whole(c.layers);
        let d = kimi::describe(&c, &st, &w32, false).unwrap();
        let (nh, kd, taps) = (c.kda.heads, c.kda.dim, c.kda.conv_kernel);
        let (f32s, typed, least) = (read_f32(&d), read_typed(&d), read_at_least(&d));
        let mut seen = 0;
        for ops in &d.layers {
            for op in ops {
                match *op {
                    Op::KdaDelta { a_log, dt_bias, .. } => {
                        assert!(f32s.contains(&a_log) && f32s.contains(&dt_bias));
                        assert!(least.contains(&(a_log, nh)) && least.contains(&(dt_bias, nh * kd)), "{least:?}");
                        seen += 1;
                    }
                    Op::KdaConv { w, w32, .. } => assert!(typed.contains(&(w, w32)) && least.contains(&(w, 3 * nh * kd * taps)), "{least:?}"),
                    Op::KdaOut { w, w32, .. } => assert!(typed.contains(&(w, w32)) && least.contains(&(w, kd)), "{least:?}"),
                    Op::LinearAt { w, .. } => assert!(op.mats() == [w] && d.wanted(&st)[w]),
                    _ => {}
                }
            }
        }
        assert!(seen > 0 && typed.iter().all(|x| x.1), "the random model's delta layers, their convolutions and gated norms fp32");
        assert_eq!((small_width(&d, typed[0].0, "F32"), small_width(&d, f32s[0], "BF16")), (4, 4));
        // (a delta layer's three convolutions, by the description's role: one small tensor, each as saved)
        let (kinds, _) = crate::runtime::describe::describe_dir(&dir).unwrap();
        let role = |r: &str| kinds.blocks.iter().find_map(|b| b.iter().find(|x| x.0 == r).map(|x| x.1.clone()));
        let (Some(Source::Rows(conv)), Some(Source::Tensor(a_log))) = (role("conv"), role("a_log")) else { panic!("no delta layer's convolutions") };
        let (dtype, bytes) = small_saved(&ck, &conv.join("|"), true).unwrap();
        assert_eq!((dtype.as_str(), bytes.len()), ("F32", 3 * nh * kd * taps * 4));
        let own: Vec<u8> = conv.iter().flat_map(|n| ck.read(n).unwrap()).collect();
        assert_eq!(bytes, own);
        assert!(small_saved(&ck, &format!("{}|{a_log}", conv[0]), true).unwrap_err().msg.contains("read as one"));
        assert_eq!(small_saved(&ck, &a_log, false).unwrap().1.len(), ck.read(&a_log).unwrap().len());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A route table as glyd_gpu.cu's `route_for` gives it past 64 rows: the prompt kernel, and from `dec` rows the decoded way (None: never).
    fn table(dec: Option<(i64, i32)>) -> impl Fn(i64) -> Result<i32> {
        move |m| Ok(dec.filter(|&(from, _)| m >= from).map_or(ROUTE_BIG, |(_, r)| r))
    }

    /// A prompt's product of the tiered layout where the table has no crossover for it (an A100's: the prompt kernel at any length) is decoded and multiplied
    /// dense from DECODE_FROM rows; the 12-bit layout there, and either layout where the table has one, keep the table's; GLYD_ENGINE_DECODE_FROM replaces both.
    #[test]
    fn a_prompt_of_a_layout_the_table_never_decodes_is_decoded_from_512_rows() {
        let a100_tiered = table(None);
        assert!(!decodes_by(511, false, None, &a100_tiered).unwrap());
        assert!(decodes_by(512, false, None, &a100_tiered).unwrap());
        assert!(decodes_by(4096, false, None, &a100_tiered).unwrap());
        // (the 12-bit layout where the table never decodes it: the table's way, the prompt kernel)
        assert!(!decodes_by(4096, true, None, &a100_tiered).unwrap());
        // (an A100's 12-bit layout: from 769, as measured; GeForce Ada's tiered layout decoded ahead from 513, the prompt kernel at 512)
        let a100_twelve = table(Some((769, ROUTE_DECODE)));
        assert!(!decodes_by(768, true, None, &a100_twelve).unwrap() && decodes_by(769, true, None, &a100_twelve).unwrap());
        let ada_tiered = table(Some((513, ROUTE_AHEAD)));
        assert!(!decodes_by(512, false, None, &ada_tiered).unwrap() && decodes_by(513, false, None, &ada_tiered).unwrap());
        // (GLYD_ENGINE_DECODE_FROM: from its rows whatever the table, and never below them)
        assert!(decodes_by(300, true, Some(256), &a100_tiered).unwrap());
        assert!(!decodes_by(4096, false, Some(1 << 30), &a100_twelve).unwrap());
    }
}
