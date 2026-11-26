//! The CUDA backend (R2, #135): the layer-kind seam (`Backend`) on an NVIDIA GPU for every kind the executor runs, over a loaded executor (`Dyn`). A
//! kind's entry point is a piece of the executor's own layers (`dynamic::entry::pieces`): a block's body (an attention kind, a latent or compressed
//! attention, a linear attention, Mamba-2, an MLP, the Moe kind's block, an n-gram embedding; its kind checked against its operations when it is built),
//! the residual's operations around the bodies (`Around::Before`, `Around::After`: the norms and the residual rules, mHC's and the gated residuals'
//! mixes, the depth mix), the embedding and the head. A piece runs its operations through `Dyn::op`, the function the executor's `forward` runs every
//! operation through, and the embedding and the head through `forward`'s own (`Dyn::embed`, `Dyn::logits_pass`): the seam and a served step launch the
//! same kernels with the same arguments, and a piece adds one virtual call and nothing on the device. The pieces' state is where the executor holds it
//! (KV pages, recurrent states, the expert cache, a pass's row tables). `kindcheck` (src/dynamic/kindcheck.rs) holds each piece to the reference.
//!
//! The backend is bound to the thread that holds its executor (whose context, stream and cells a piece uses): it is `Send` and `Sync` as the seam asks,
//! and what reaches the executor through it (`queue`, `build`) is refused on another thread. Its pieces hold where they are, no more, and are `Send`.
use super::backend::{
    Arenas, Around, Backend, Captured, Collective, DeviceId, DeviceInfo, Format, KindRef, LayerDesc, LayerImpl, Placement, StepCx, StepShape, Support,
    WeightSource,
};
use crate::dynamic::entry::{pieces, Pieces, Run};
use crate::dynamic::{Dyn, Op};
use crate::ffi::{err, Result};
use std::marker::PhantomData;
use std::ops::Range;

/// The loaded executor, reached on the thread that made the backend alone.
struct Bound<'e> {
    e: *const Dyn,
    thread: std::thread::ThreadId,
    _e: PhantomData<&'e Dyn>,
}

// SAFETY: the pointer is dereferenced on the thread the backend was made on alone (`get` checks), where the borrow 'e keeps the executor alive; nothing of
// the executor is reached from another thread, which may hold the backend and only be refused.
unsafe impl Send for Bound<'_> {}
// SAFETY: as above.
unsafe impl Sync for Bound<'_> {}

impl<'e> Bound<'e> {
    fn get(&self) -> Result<&'e Dyn> {
        if std::thread::current().id() != self.thread {
            return err("the CUDA backend's executor is reached on the thread that holds it alone");
        }
        // SAFETY: on the thread that borrowed the executor for 'e (checked above).
        Ok(unsafe { &*self.e })
    }
}

/// The weights a CUDA backend's pieces run: its executor's, loaded by its plan (no format is said here: the matrices are as the executor packed them).
pub struct Held;

impl WeightSource for Held {
    fn format(&self, _: &str) -> Option<Format> {
        None
    }
    fn f32(&self, _: &str) -> Option<super::backend::HostTensor> {
        None
    }
}

/// The CUDA backend over a loaded executor (`over`).
pub struct Cuda<'e> {
    devices: Vec<DeviceInfo>,
    e: Bound<'e>,
    pieces: Pieces,
}

/// A step's queue: the executor (its stream, its kernels, its loaded part: weights, KV pages, states, the expert cache).
pub struct Exec<'e> {
    e: &'e Dyn,
}

impl Exec<'_> {
    /// A pass's metadata onto the GPU before its pieces run (`Dyn::entry_begin`: the rows' positions, their tokens, what the executor makes once for every
    /// layer), as the executor's own step writes it before its first operation.
    pub fn begin(&self, positions: &[u32], tokens: &[u32]) -> Result<()> {
        self.e.entry_begin(positions, tokens)
    }
}

/// A step's activations on the GPU: n rows of the executor's slots (the residual X or the streams S, a body's input H and its output), of which the head
/// reads the last `head`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rows {
    pub n: usize,
    pub head: usize,
}

/// What a piece runs: operations of a layer, the embedding, or the head.
#[derive(Clone, Debug, PartialEq)]
enum Work {
    Ops { l: usize, ops: Range<usize> },
    Embed,
    Head,
}

/// A kind's entry point: its piece of the executor.
struct Piece {
    work: Work,
}

impl<'e> LayerImpl<Cuda<'e>> for Piece {
    /// The executor's arenas, sized at its load, hold every piece's.
    fn workspace(&self, _: &StepShape) -> u64 {
        0
    }

    fn run(&mut self, cx: &mut StepCx<Cuda<'e>>, x: &mut Rows) -> Result<()> {
        let e = cx.queue.e;
        if cx.positions.len() != x.n || cx.tokens.len() != x.n {
            return err(format!("{} rows with {} positions and {} tokens", x.n, cx.positions.len(), cx.tokens.len()));
        }
        match &self.work {
            Work::Ops { l, ops } => e.entry_ops(*l, ops.clone(), x.n),
            Work::Embed => e.entry_embed(cx.tokens),
            Work::Head => e.entry_head(x.n, x.head),
        }
    }
}

/// The operation that makes a body the kind it is, by the kind's name; none for an MLP's (its products and activation).
fn defining(op: &Op) -> Option<&'static str> {
    Some(match op {
        Op::Attn(_) => "attention",
        Op::Mla(_) => "latent attention",
        Op::CompAttn { .. } => "compressed attention",
        Op::GdnDelta { .. } => "Gated DeltaNet",
        Op::KdaDelta { .. } => "Kimi delta attention",
        Op::MambaScan { .. } => "Mamba-2",
        Op::Moe(_) => "mixture of experts",
        Op::NgramRows(_) => "n-gram embedding",
        _ => return None,
    })
}

/// The defining operation a block of `kind` has (`defining`), none for an MLP.
fn wanted(kind: KindRef) -> Option<&'static str> {
    match kind {
        KindRef::Attention | KindRef::SlidingAttention | KindRef::SparseAttention => Some("attention"),
        KindRef::Mla => Some("latent attention"),
        KindRef::Compressed => Some("compressed attention"),
        KindRef::GatedDeltaNet => Some("Gated DeltaNet"),
        KindRef::KimiDelta => Some("Kimi delta attention"),
        KindRef::Mamba2 => Some("Mamba-2"),
        KindRef::Moe => Some("mixture of experts"),
        KindRef::Ngram => Some("n-gram embedding"),
        KindRef::Mlp | KindRef::PerLayer => None,
    }
}

/// Whether a body's operations (`ops`, a run of what `run` says) are a block of `kind`: the one operation that makes its kind (`defining`) and no other kind's,
/// an n-gram embedding's run for that kind alone; else what they are.
fn kind_ops(kind: KindRef, ops: &[Op], run: Run) -> std::result::Result<(), String> {
    let found: Vec<&str> = ops.iter().filter_map(defining).collect();
    let want = wanted(kind);
    if found != want.into_iter().collect::<Vec<_>>() || (run == Run::Ngram) != (kind == KindRef::Ngram) {
        return Err(format!("are of {found:?} ({run:?}), where the kind's are {want:?}"));
    }
    Ok(())
}

impl<'e> Cuda<'e> {
    /// The backend over executor `e`, on the thread that holds it: its GPU, and its layers' pieces.
    pub fn over(e: &'e Dyn) -> Result<Cuda<'e>> {
        let d = &e.dev;
        let devices = vec![DeviceInfo { vendor: "nvidia".into(), arch: format!("sm_{}{}", d.cc.0, d.cc.1), memory: d.total_mem as u64, unified: false }];
        let thread = std::thread::current().id();
        Ok(Cuda { devices, pieces: pieces(&e.desc().layers)?, e: Bound { e, thread, _e: PhantomData } })
    }

    /// The blocks the executor's layers hold (a body's piece each).
    pub fn blocks(&self) -> usize {
        self.pieces.body.len()
    }

    fn ops(&self, at: &(usize, Range<usize>)) -> Box<dyn LayerImpl<Self>> {
        Box::new(Piece { work: Work::Ops { l: at.0, ops: at.1.clone() } })
    }
}

impl<'e> Backend for Cuda<'e> {
    type Queue = Exec<'e>;
    type Fence = ();
    type Tensor = Rows;

    fn devices(&self) -> &[DeviceInfo] {
        &self.devices
    }

    /// Every kind the executor runs, of the matrices it reads (bf16 packed at a level or held as saved, FP8 as released: weights.rs). Refused: per-layer
    /// inputs (the executor's lowering refuses them, src/dynamic/lower.rs) and matrices of other formats.
    fn supports(&self, kind: &KindRef, fmt: &Format) -> Support {
        match (kind, fmt) {
            (KindRef::PerLayer, _) => Support::No("per-layer inputs (E2B, E4B): not an operation of the executor yet".into()),
            (_, Format::F32 | Format::Mxfp4) => Support::No(format!("{fmt:?} matrices: the executor reads bf16 and FP8 ones")),
            (_, Format::Bf16 | Format::Lossless | Format::Lossy(_) | Format::Fp8Block) => Support::Yes { rate: 1.0 },
        }
    }

    /// Block `layer.index`'s body: the executor's operations of that block, which must hold the one operation of the block's kind (`defining`) and no other
    /// kind's. The weights are the executor's, loaded by its plan; the source says their format alone.
    fn build(&self, layer: &LayerDesc, w: &dyn WeightSource, _: &Placement, _: &mut Arenas<Self>) -> Result<Box<dyn LayerImpl<Self>>> {
        let e = self.e.get()?;
        let (b, kind) = (layer.index, KindRef::of(layer.block));
        if let Support::No(why) = self.supports(&kind, &w.format("").unwrap_or(Format::Bf16)) {
            return err(format!("block {b}: {why}"));
        }
        if layer.hidden != e.desc().hidden {
            return err(format!("block {b}: a hidden width of {}, where the executor's is {}", layer.hidden, e.desc().hidden));
        }
        let Some((l, r, run)) = self.pieces.body.get(b) else {
            return err(format!("block {b}: the executor's layers hold {} blocks", self.pieces.body.len()));
        };
        if let Err(why) = kind_ops(kind, &e.desc().layers[*l][r.clone()], *run) {
            return err(format!("block {b} ({kind:?}): the executor's operations {r:?} of layer {l} {why}"));
        }
        Ok(self.ops(&(*l, r.clone())))
    }

    /// The residual's operations before or after a block's body (a layer's, as the executor cuts them: `Pieces`), the embedding (its rows, then the
    /// description's first operation) or the head.
    fn build_around(&self, a: &Around, _: &dyn WeightSource, _: &Placement, _: &mut Arenas<Self>) -> Result<Box<dyn LayerImpl<Self>>> {
        let e = self.e.get()?;
        let at = |v: &Vec<(usize, Range<usize>)>, b: usize| {
            v.get(b).cloned().ok_or_else(|| crate::ffi::Error(format!("block {b}: the executor's layers hold {}", v.len())))
        };
        Ok(match a {
            Around::Before { index, .. } => self.ops(&at(&self.pieces.before, *index)?),
            Around::After { index, .. } => self.ops(&at(&self.pieces.after, *index)?),
            Around::Embed(_) => {
                if !crate::dynamic::entry::residual_op(&e.desc().first) {
                    return err(format!("the embedding: the description's first operation is {:?}, not the residual's", e.desc().first));
                }
                Box::new(Piece { work: Work::Embed })
            }
            Around::Head { .. } => Box::new(Piece { work: Work::Head }),
        })
    }

    /// The executor's own queue (its one GPU).
    fn queue(&self, dev: DeviceId) -> Result<Exec<'e>> {
        if dev != 0 {
            return err(format!("GPU {dev}: the backend's executor is on one"));
        }
        Ok(Exec { e: self.e.get()? })
    }

    fn capture(&self, _: &Exec<'e>, body: &mut dyn FnMut() -> Result<()>) -> Result<Option<Captured>> {
        body()?;
        Ok(None)
    }

    fn collective(&self) -> Option<&dyn Collective> {
        None
    }
}

#[cfg(all(test, feature = "dev"))]
mod tests {
    use super::*;

    /// Every block of each family's random case is a body piece of its kind in the executor's own lowering of it (what `build` checks on the GPU), and every
    /// operation of every layer is in one piece: the description's blocks, in order, are the executor's runs.
    #[test]
    fn every_block_is_a_piece_of_its_kind() {
        const CASES: [&str; 15] = [
            "gemma4",
            "gemma4-kv",
            "dense",
            "qwen35",
            "qwen35moe",
            "glm",
            "glm-dsa",
            "kimi",
            "kimi-dsa",
            "deepseek-v4",
            "deepseek-v41",
            "muse",
            "qwen4x",
            "qwen4x-s2",
            "nemotron",
        ];
        let dir = std::env::temp_dir().join(format!("glyd-cuda-pieces-{}", std::process::id()));
        for case in CASES {
            let _ = std::fs::remove_dir_all(&dir);
            crate::logits::random_model(case, &dir).unwrap_or_else(|e| panic!("{case}: {}", e.msg));
            let (kinds, _) = crate::runtime::describe::describe_dir(&dir).unwrap();
            let low = match case {
                "gemma4" | "gemma4-kv" => crate::dynamic::gemma4::lowered(&dir, &crate::gemma4::ConfigG4::load(&dir).unwrap()),
                "dense" => crate::dynamic::dense::lowered(&dir),
                _ => crate::dynamic::split::lowered_of(&crate::runtime::config::ModelConfig::load(&dir).unwrap(), 4096),
            }
            .unwrap_or_else(|e| panic!("{case}: {}", e.msg));
            let p = pieces(&low.layers).unwrap();
            assert_eq!(p.body.len(), kinds.desc.blocks.len(), "{case}: the executor's blocks");
            for (b, (block, (l, r, run))) in kinds.desc.blocks.iter().zip(&p.body).enumerate() {
                let kind = KindRef::of(block);
                kind_ops(kind, &low.layers[*l][r.clone()], *run).unwrap_or_else(|e| panic!("{case} block {b} ({kind:?}): {e}"));
            }
            for (l, ops) in low.layers.iter().enumerate() {
                let mut seen = vec![0; ops.len()];
                let all = p.body.iter().map(|x| (x.0, x.1.clone())).chain(p.before.iter().cloned()).chain(p.after.iter().cloned());
                for (_, r) in all.filter(|x| x.0 == l) {
                    r.for_each(|i| seen[i] += 1);
                }
                assert!(seen.iter().all(|&c| c == 1), "{case} layer {l}: {seen:?}");
            }
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A body is refused for another kind: an attention's operations are not an MLP's or the Moe kind's, an MLP's are not an attention's.
    #[test]
    fn a_body_of_another_kind_is_refused() {
        use crate::dynamic::Slot;
        let lin = Op::Linear { w: 0, x: Slot::H, y: Slot::O, name: "" };
        let act = Op::GatedAct { inter: 8, act: crate::moe::Act::SwiGlu { limit: None } };
        assert!(kind_ops(KindRef::Mlp, &[lin.clone(), act.clone(), lin.clone()], Run::Body).is_ok());
        assert!(kind_ops(KindRef::Attention, &[lin.clone(), act, lin.clone()], Run::Body).is_err());
        assert!(kind_ops(KindRef::Ngram, &[lin], Run::Body).is_err());
    }
}
