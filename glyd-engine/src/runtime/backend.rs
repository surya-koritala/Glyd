//! The vendor seam (`docs/design/runtime.md` §3-4): a backend says per kind and weight format whether it supports it, and builds a
//! `LayerImpl` for each block of a description, and for the pieces around the blocks' bodies (`Around`: the residual's norms and rules,
//! the embedding, the head). The executor is generic over a backend's associated types; layers are `Box<dyn LayerImpl<B>>`, one virtual
//! call per block per step.
//!
//! The types around the traits (`Placement`, `Arenas`, `StepCx`) carry what the CPU reference needs; the CUDA backend (`runtime::cuda`)
//! keeps a piece's state where its loaded executor holds it (KV pages, recurrent states, the expert cache, a pass's row tables), so its
//! pieces take nothing more from them.

use super::kinds::{Block, Body, Embed, Ffn, Head, Mixer, Norm};
use crate::ffi::{err, Result};

/// A kind without its parameters, for `supports`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum KindRef {
    Attention,
    SlidingAttention,
    SparseAttention,
    Mla,
    Compressed,
    GatedDeltaNet,
    KimiDelta,
    Mamba2,
    Mlp,
    Moe,
    /// A hashed n-gram embedding into the residual streams (Qwen3.8-Flash-Next's, DeepSeek-V4.1's Engram).
    Ngram,
    /// A per-layer input's gated branch (Gemma 4's E2B, E4B).
    PerLayer,
}

impl KindRef {
    pub fn of(block: &Block) -> KindRef {
        use super::kinds::LinearKind;
        match &block.body {
            Body::Mixer(Mixer::Attention(a)) if a.indexer.is_some() => KindRef::SparseAttention,
            Body::Mixer(Mixer::Attention(a)) if a.window.is_some() => KindRef::SlidingAttention,
            Body::Mixer(Mixer::Attention(_)) => KindRef::Attention,
            Body::Mixer(Mixer::Mla(_)) => KindRef::Mla,
            Body::Mixer(Mixer::Compressed(_)) => KindRef::Compressed,
            Body::Mixer(Mixer::Linear(l)) if l.kind == LinearKind::GatedDeltaNet => KindRef::GatedDeltaNet,
            Body::Mixer(Mixer::Linear(_)) => KindRef::KimiDelta,
            Body::Mixer(Mixer::Mamba2(_)) => KindRef::Mamba2,
            Body::Ffn(Ffn::Mlp(_)) => KindRef::Mlp,
            Body::Ffn(Ffn::Moe(_)) => KindRef::Moe,
            Body::Ngram(_) => KindRef::Ngram,
            Body::PerLayer(_) => KindRef::PerLayer,
        }
    }
}

/// A layer's weight format as stored.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Format {
    Bf16,
    F32,
    /// Glyd's lossless formats (10.75-12 bits).
    Lossless,
    /// Int4-6 and codebook lossy levels, by name (kestrel, swift, wren, ...).
    Lossy(String),
    /// FP8 with 128x128 block scales, as DeepSeek and GLM release them.
    Fp8Block,
    Mxfp4,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Support {
    Yes { rate: f32 },
    Slower(String),
    No(String),
}

#[derive(Clone, Debug, PartialEq)]
pub struct DeviceInfo {
    pub vendor: String,
    pub arch: String,
    pub memory: u64,
    pub unified: bool,
}

pub type DeviceId = usize;

/// One block of a description, at its index.
pub struct LayerDesc<'a> {
    pub index: usize,
    pub block: &'a Block,
    pub hidden: usize,
}

/// A host tensor, row-major.
#[derive(Clone, Debug, PartialEq)]
pub struct HostTensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

/// Where a layer's weights come from, by role (`kinds`). A source decodes its format; the reference backend asks for f32.
pub trait WeightSource {
    fn format(&self, role: &str) -> Option<Format>;
    fn f32(&self, role: &str) -> Option<HostTensor>;
}

/// Which device a layer runs on and where its experts live (the planner's, P2; R12 adds the tiers).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Placement {
    pub device: DeviceId,
}

/// The arenas a plan sizes at load: weights, KV pages, state slots, the step workspace, host staging (decision 8).
pub struct Arenas<B: Backend + ?Sized> {
    pub workspace_bytes: u64,
    pub _b: std::marker::PhantomData<fn() -> B>,
}

impl<B: Backend + ?Sized> Default for Arenas<B> {
    fn default() -> Self {
        Arenas { workspace_bytes: 0, _b: std::marker::PhantomData }
    }
}

/// A step's shape, for workspace sizing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StepShape {
    pub tokens: usize,
    pub rows: usize,
}

/// What a layer sees of a step: its queue and the step's metadata. `positions` and `seqs` have one entry per token of the flat batch
/// (`seqs`: the sequence's state slot, which keys its KV and recurrent state).
pub struct StepCx<'a, B: Backend + ?Sized> {
    pub queue: &'a mut B::Queue,
    pub positions: &'a [u32],
    pub seqs: &'a [u32],
    /// Token ids, for kinds that read them (hash routing).
    pub tokens: &'a [u32],
}

/// A piece of a description around its blocks' bodies, each with an entry point of its own on a backend that runs whole models (`build_around`):
/// the residual's operations between the bodies (the rule of the block before, its post-norm, the next block's input norm, a streams kind's mixes),
/// the embedding, and the head. A backend decides which side of a body its residual's operations fall on (the CUDA backend: where its executor's
/// layers put them), so a piece before a body or after one may hold none.
pub enum Around<'a> {
    /// The residual's operations before block `index`'s body (`block`).
    Before { index: usize, block: &'a Block },
    /// The residual's operations after block `index`'s body (the final norm's, after the last).
    After { index: usize, block: &'a Block },
    /// The tokens' rows of the embedding into the residual (`StepCx::tokens`).
    Embed(&'a Embed),
    /// The last rows' final norm (its collapse of the streams), the LM head and its soft-cap.
    Head { norm: &'a Norm, head: &'a Head },
}

/// A captured step (a CUDA or HIP graph); backends that cannot capture return None from `capture`.
pub struct Captured;

/// All-reduce, all-gather, all-to-all across a backend's devices (M4).
pub trait Collective {}

pub trait Backend: Send + Sync {
    type Queue;
    type Fence;
    type Tensor;
    fn devices(&self) -> &[DeviceInfo];
    fn supports(&self, kind: &KindRef, fmt: &Format) -> Support;
    fn build(&self, layer: &LayerDesc, w: &dyn WeightSource, place: &Placement, mem: &mut Arenas<Self>) -> Result<Box<dyn LayerImpl<Self>>>;
    /// An entry point for a piece around the blocks' bodies; a backend that runs blocks alone (the reference's model runs these itself, the MLX spike
    /// runs one kind) refuses it.
    fn build_around(&self, a: &Around, w: &dyn WeightSource, place: &Placement, mem: &mut Arenas<Self>) -> Result<Box<dyn LayerImpl<Self>>> {
        let _ = (w, place, mem);
        let what = match a {
            Around::Before { index, .. } => format!("the residual before block {index}"),
            Around::After { index, .. } => format!("the residual after block {index}"),
            Around::Embed(_) => "the embedding".into(),
            Around::Head { .. } => "the head".into(),
        };
        err(format!("{what}: this backend runs blocks' bodies alone"))
    }
    fn queue(&self, dev: DeviceId) -> Result<Self::Queue>;
    fn capture(&self, q: &Self::Queue, body: &mut dyn FnMut() -> Result<()>) -> Result<Option<Captured>>;
    fn collective(&self) -> Option<&dyn Collective>;
}

pub trait LayerImpl<B: Backend + ?Sized>: Send {
    /// Bytes of the step arena it needs at this shape.
    fn workspace(&self, shape: &StepShape) -> u64;
    fn capturable(&self) -> bool;
    /// Enqueue the block on `x` (the residual stream of the step's tokens); never waits on the device.
    fn run(&mut self, cx: &mut StepCx<B>, x: &mut B::Tensor) -> Result<()>;
}
