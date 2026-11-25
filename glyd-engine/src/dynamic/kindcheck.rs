//! `kindcheck [CASE...] [--prompt N] [--steps N] [--scratch DIR] [--chain | --sequence]` (R2, #135): every layer kind the families' descriptions hold, run by the executor's CUDA
//! kernels against its CPU f32 reference (`runtime::reference`), kind by kind, on each logits case's random model (Gemma 4, the Qwen3.5 layout,
//! the plain dense block, the Mamba-2 layout, Muse Spark, Kimi-K3, GLM-5.3-Flash, DeepSeek-V4, DeepSeek-V4.1, Qwen3.8-Flash-Next) at the lossless level.
//! With lagged mHC (DeepSeek-V4.1's) each layer's pass is given the weights its first block takes as well (the first layer's one-hot, another's at
//! random), on both sides; an indexer that picks for itself (a latent attention's, a compressed attention's index source) is traced and its picks taken
//! by the reference, so that a layer reading another's picks reads the same ones.
//!
//! Each layer of the executor's description runs operation by operation on a prompt of `--prompt` rows, then `--steps` decode steps, each pass
//! from a residual made at random (the layer's caches and states carried from pass to pass, as a sequence's). A layer's operations are runs of
//! the residual's (the norms, the residual rule, mHC's mixing, the depth mix) and of the blocks' bodies (a mixer or a feed-forward); each is
//! compared where it ends:
//!
//! - a body's output slot against the reference's body (`RefLayer::body`) on the GPU's own input rows (H), so a body's error is its own;
//! - after the residual's operations, the residual (X, mHC's streams S, the depth mix's partial sum and completed groups) and the next body's
//!   input H against the reference's residual rule, advanced by the GPU's own body outputs, and its norms (the final norm after the last);
//! - an n-gram embedding, whose operations add its output into the streams in place (`Op::NgramAdd`), by the residual it leaves: the streams
//!   against the reference's block on the same rows (and its own output, the streams' change, printed beside: not judged, as the streams round
//!   it to bf16 with them).
//!
//! A comparison is the largest difference over the reference's largest magnitude on a pass's rows; each line has its kind's bound (`bound`,
//! the table in `runtime::reference`), and names the rows left out as near a tie (`NEAR`), which may be few (`near_within`: a body that
//! leaves out more of its rows fails, as one past its bound does). Exit status 1 where one is not kept.
//!
//! `--sequence`: the passes are one sequence's instead (the logits cases' fixed tokens, `logits::token_list`: the prompt, then each step's token).
//! The first layer's residual is the reference's embedding of them and every later layer's is what the executor's layer before it left, so each
//! body is compared as above on the rows a run of the model gives it; and after each layer the executor's residual is compared with the
//! reference's own run of the same tokens (a second copy of the model, alone): how far the two are apart there, with everything before it. That
//! figure has no bound (it is where a difference of the logits first shows, layer by layer); the bodies' and the residual rule's keep theirs.
//! Not for a depth mix or lagged mHC, whose residual is more than the rows a block is handed.
//! GLYD_KINDCHECK_ROWS prints each body's rows one by one. DeepSeek-V4's simulated quantizations of the release (GLYD_V4_SIM) are turned
//! off: no kind names them (GLYD_KINDCHECK_SIM=1 leaves them on, to see where they first take a body away from the reference).

use super::{Dyn, Op, Opts, Slot};
use crate::ffi::{err, Error, Result};
use crate::runtime::kinds::{AttnRounding, Block, Body, Ffn, Indexer, LinearKind, Mixer, NormKind, NormWeight, Residual, StreamsP};
use crate::runtime::reference::comp::CompState;
use crate::runtime::reference::mla::{LatentCache, Picked};
use crate::runtime::reference::model::{checkpoint_tensor, Depth, RefModel};
use crate::runtime::reference::{bf16, collapse, hc_weights, Hc, RefLayer, SeqState};
use crate::weights::Layout;
use std::ops::Range;
use std::path::{Path, PathBuf};

/// The rows of a comparison's kind in the bounds' table.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Class {
    Attention,
    Latent,
    Compressed,
    DeltaNet,
    KimiDelta,
    Mamba2,
    Mlp,
    Moe,
    Residual,
}

/// Each class's bound: the largest difference over the reference's largest magnitude a kernel may have (why each: `runtime::reference`).
pub fn bound(c: Class) -> f32 {
    match c {
        Class::Attention | Class::Latent | Class::Compressed => 2e-2,
        Class::DeltaNet | Class::KimiDelta | Class::Mamba2 => 3e-2,
        Class::Mlp => 1e-2,
        Class::Moe => 2e-2,
        Class::Residual => 2e-2,
    }
}

/// One comparison: where (case, layer, block), its kind, its worst on the prompt's pass and on the steps', the rows left out as near a
/// tie (`NEAR`), and the indexer's choices apart that no near tie explains (`apart`).
struct Line {
    case: &'static str,
    at: String,
    kind: String,
    class: Class,
    prompt: f32,
    steps: f32,
    near: Vec<String>,
    /// The rows this comparison saw (the prompt's and each step's), of which `near` were left out.
    rows: usize,
    /// Rows whose indexer picks are not the reference's own and not a near tie from them on the engine's own input (`indexer_apart`).
    apart: Vec<String>,
}

impl Line {
    /// Within its kind's bound on the prompt and on the steps, with no more rows left out than `near_within` allows and no choice apart.
    fn ok(&self) -> bool {
        self.prompt <= bound(self.class) && self.steps <= bound(self.class) && near_within(self.near.len(), self.rows) && self.apart.is_empty()
    }
}

/// A row past its bound whose choice (a router's experts, an indexer's picks) was nearer than this to another (`reference::ffn::margin`: the
/// gap over the largest score) is left out of its body's comparison and named: the GPU's scores of bf16 inputs can order such a pair the
/// other way, which changes the output by far more than its arithmetic does.
pub(crate) const NEAR: f32 = 1e-2;

/// Whether a comparison of `rows` rows may leave `left` of them out as near a tie: one row in sixteen at most (and one where it has fewer than
/// sixteen). The runs `runtime::reference`'s table records left out 1 row of a body's 52 to 66, and 2 and 3 of five bodies' 260 and 330;
/// a body whose rows were mostly left out was compared with nothing, so it fails.
fn near_within(left: usize, rows: usize) -> bool {
    left <= (rows / 16).max(1)
}

/// The largest difference over `want`'s largest magnitude (infinite where either has a value that is not finite: a reference that is not a
/// number bounds nothing).
fn rel(got: &[f32], want: &[f32]) -> f32 {
    if got.len() != want.len() || got.iter().chain(want).any(|v| !v.is_finite()) {
        return f32::INFINITY;
    }
    let scale = want.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-30);
    got.iter().zip(want).fold(0f32, |m, (a, b)| m.max((a - b).abs())) / scale
}

/// A block's kind as a line names it, and its class.
fn label(b: &Block) -> (Class, String) {
    match &b.body {
        Body::Mixer(Mixer::Attention(a)) => {
            let mut s = format!("attention: {} heads over {} of {}", a.heads, a.kv_heads, a.head_dim);
            if let Some(w) = a.window {
                s += &format!(", window {w}");
            }
            if a.rope.is_none() {
                s += ", no rope";
            } else if a.rope.as_ref().is_some_and(|r| r.freqs * 2 < r.dim || r.dim < a.head_dim) {
                s += ", partial rope";
            }
            if a.k_eq_v {
                s += ", K = V";
            }
            if let Some(g) = a.gate {
                s += &format!(", {:?} gate{}", g.from, if g.norm.is_some() { " after a norm" } else { "" });
            }
            if a.softcap.is_some() {
                s += ", soft-capped";
            }
            if a.sinks {
                s += ", sinks";
            }
            s += &format!(
                ", scale {:.4}, {}",
                a.scale,
                match a.rounding {
                    AttnRounding::F32 => "f32",
                    AttnRounding::Flash => "bf16 probabilities",
                    AttnRounding::Eager => "bf16 scores",
                    AttnRounding::Sdpa => "bf16 rope, f32 softmax",
                }
            );
            (Class::Attention, s)
        }
        Body::Mixer(Mixer::Mla(m)) => {
            let ix =
                m.indexer.as_ref().map_or(String::new(), |i| format!(", indexer of {} of {} keeping {} (pools of {})", i.heads, i.head_dim, i.top_k, i.block));
            (
                Class::Latent,
                format!("latent attention: {} heads, rank {}, rope {}{}{ix}", m.heads, m.kv_lora, m.rope_dim, if m.gate.is_some() { ", gated" } else { "" }),
            )
        }
        Body::Mixer(Mixer::Compressed(c)) => {
            let ix = c.indexer.as_ref().map_or(String::new(), |i| format!(", indexer keeping {}", i.top_k));
            (Class::Compressed, format!("compressed attention: window {}, ratio {}{}{ix}", c.window, c.ratio, if c.overlap { " overlapping" } else { "" }))
        }
        Body::Mixer(Mixer::Linear(l)) => match l.kind {
            LinearKind::GatedDeltaNet => (Class::DeltaNet, format!("Gated DeltaNet: {} key heads, {} value heads", l.k_heads, l.v_heads)),
            LinearKind::KimiDelta => (Class::KimiDelta, format!("Kimi delta attention: {} heads of {}", l.k_heads, l.k_dim)),
        },
        Body::Mixer(Mixer::Mamba2(m)) => (Class::Mamba2, format!("Mamba-2: {} heads of {}, state {}, {} groups", m.heads, m.head_dim, m.state, m.groups)),
        Body::Ffn(Ffn::Mlp(m)) => {
            (Class::Mlp, format!("MLP: {:?}{}{}", m.act, if m.gated { ", gated" } else { "" }, m.clamp.map_or(String::new(), |c| format!(", limit {c}"))))
        }
        Body::Ffn(Ffn::Moe(e)) => {
            let r = &e.router;
            let mut s = format!("MoE: {:?} router, top {} of {}", r.score, r.top_k, e.experts);
            if r.groups.is_some() {
                s += ", groups";
            }
            if r.hash {
                s += ", hash";
            }
            if e.shared.is_some() {
                s += if e.shared_gate { ", gated shared" } else { ", shared" };
            }
            if e.latent.is_some() {
                s += ", latent";
            }
            s += &format!(", {:?} experts", e.expert.act);
            (Class::Moe, s)
        }
        Body::Ngram(n) => (Class::Residual, format!("n-gram embedding: orders up to {}, {} tables each, {} streams", n.order, n.heads, n.streams)),
        Body::PerLayer(pl) => (Class::Mlp, format!("per-layer input {}: {:?} gate", pl.layer, pl.act)),
    }
}

/// The residual's kind before a block (or the final norm): its norm and its rule.
fn residual_label(b: Option<&Block>) -> String {
    let Some(b) = b else { return "the final norm".into() };
    let n = &b.norm_in;
    let norm = match (n.kind, n.weight) {
        (NormKind::Rms, NormWeight::PlusOne) => "RMS (1 + w)",
        (NormKind::Rms, NormWeight::InputPlusOne) => "RMS of x (1 + w)",
        (NormKind::Rms, _) => "RMS",
        (NormKind::Layer, _) => "LayerNorm",
        (NormKind::L2, _) => "l2",
        (NormKind::Identity, _) => "none",
    };
    let rule = match &b.residual {
        Residual::Add => "add".to_string(),
        Residual::AddScaled(s) => format!("add times {s}"),
        Residual::AddThenScale => "add, then the layer's scalar".into(),
        Residual::Gated(_) => "gated".into(),
        Residual::Streams(s) => format!("mHC over {} streams", s.streams),
        Residual::GatedStreams(g) => format!("gated residuals over {} streams", g.streams),
        Residual::DepthMix(_) => "depth mix".into(),
    };
    format!("residual ({rule}) and the {norm} norm{}", if b.norm_out.is_some() { ", with the block's post-norm" } else { "" })
}

/// Whether an operation is the residual's (it reads or writes X, mHC's streams or the depth mix's blocks, or it is a block's post-norm of its
/// output in place) and not a body's.
fn residual_op(op: &Op) -> bool {
    match *op {
        Op::AddNorm { .. } | Op::NormAdd { .. } | Op::EmbedNorm { .. } | Op::GatedResidual { .. } | Op::Nop => true,
        Op::RmsNorm { x, y, .. } => x == Slot::X || x == y,
        Op::Streams | Op::StreamsPre { .. } | Op::StreamsPost { .. } | Op::StreamsHead { .. } => true,
        Op::DepthMix(_) | Op::Add { .. } | Op::MoeEarly { .. } => true,
        _ => false,
    }
}

/// Where a body's last operation writes its output.
fn out_slot(op: &Op) -> Result<Slot> {
    match *op {
        Op::Linear { y, .. } | Op::F8Lin { y, .. } | Op::LinearAt { y, .. } => Ok(y),
        Op::Moe(m) => Ok(m.y),
        Op::Mla { .. } => Ok(Slot::O),
        ref o => err(format!("kindcheck: where {o:?} writes a body's output is not known")),
    }
}

/// What a run of a layer's operations is: the residual's, a body's (its output in a slot), or an n-gram embedding's (`Op::NgramRows` to
/// `Op::NgramAdd`: its output added into the streams in place).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Run {
    Residual,
    Body,
    Ngram,
}

/// A layer's operations as runs: (what each is, its range). An n-gram embedding's operations are a run of their own, ending at its add.
fn runs(ops: &[Op]) -> Vec<(Run, Range<usize>)> {
    let mut v: Vec<(Run, Range<usize>)> = Vec::new();
    let mut open = false;
    for (i, op) in ops.iter().enumerate() {
        let run = if open || matches!(op, Op::NgramRows(_)) {
            Run::Ngram
        } else if residual_op(op) {
            Run::Residual
        } else {
            Run::Body
        };
        let starts = matches!(op, Op::NgramRows(_));
        open = run == Run::Ngram && !matches!(op, Op::NgramAdd(_));
        match v.last_mut() {
            Some((k, r)) if *k == run && !starts => r.end = i + 1,
            _ => v.push((run, i..i + 1)),
        }
    }
    v
}

/// Each layer's first block, and the blocks' count after the last: a block for each run of a body or of an n-gram embedding (a layer may have
/// more blocks than another: Qwen3.8-Flash-Next's n-gram embedding before some layers' mixers).
fn block_starts(layers: &[Vec<Op>]) -> Vec<usize> {
    let mut v = vec![0];
    for ops in layers {
        let n = runs(ops).iter().filter(|(k, _)| *k != Run::Residual).count();
        v.push(v[v.len() - 1] + n);
    }
    v
}

struct Rng(u64);

impl Rng {
    /// n bf16 values uniform in [-1, 1).
    fn v(&mut self, n: usize) -> Vec<f32> {
        (0..n)
            .map(|_| {
                self.0 ^= self.0 << 13;
                self.0 ^= self.0 >> 7;
                self.0 ^= self.0 << 17;
                bf16(2.0 * ((self.0 >> 40) as f32 / (1u64 << 24) as f32) - 1.0)
            })
            .collect()
    }
}

/// A token's residual on the reference: its stream(s), or the depth mix's state; the mHC weights its block's `pre` gave its `post`; and with lagged mHC
/// (`StreamsP::lagged`) the weights the next block's input is weighed by (the block before's own).
struct Row {
    x: Vec<f32>,
    depth: Option<Depth>,
    hc: Option<Hc>,
    lag: Option<Vec<f32>>,
}

impl Dyn {
    fn kc_width(&self, s: Slot) -> usize {
        self.big.d.widths[s as usize]
    }

    /// Where a token's residual is: mHC's streams where the model has them, else X.
    fn kc_state(&self) -> Slot {
        if self.kc_width(Slot::S) > 0 {
            Slot::S
        } else {
            Slot::X
        }
    }

    fn kc_put(&self, s: Slot, v: &[f32]) -> Result<()> {
        let w: Vec<u16> = v.iter().map(|&x| (bf16(x).to_bits() >> 16) as u16).collect();
        self.big.slots[s as usize].write(0, &w)
    }

    fn kc_get(&self, s: Slot, n: usize) -> Result<Vec<f32>> {
        self.dev.sync()?;
        let mut w = vec![0u16; n * self.kc_width(s)];
        self.big.slots[s as usize].read(0, &mut w)?;
        Ok(w.iter().map(|&b| f32::from_bits((b as u32) << 16)).collect())
    }

    /// Operations `ops` of layer `l` on n rows at positions `at`.., the tokens `tok` (DeepSeek-V4's hash routing reads them).
    fn kc_ops(&self, l: usize, ops: Range<usize>, n: usize, at: usize, tok: &[u32]) -> Result<()> {
        self.dev.sync()?;
        self.pos.write(0, &(0..n).map(|i| (at + i) as i32).collect::<Vec<i32>>())?;
        if let Some(v) = &self.tok {
            v.write(0, &tok.iter().map(|&t| t as i32).collect::<Vec<i32>>())?;
        }
        self.start.set(at);
        let pp = self.pos.ptr;
        // (what `forward` makes before a pass's operations: the decode's rows' lengths and first tokens, which the attention's decode steps read)
        // SAFETY: as below.
        unsafe { self.pass_prepare(&self.big, n, (pp, pp))? };
        for i in ops {
            // SAFETY: the engine's own buffers, sized for a chunk (n rows at most a chunk); its slots hold the rows the operation reads (written above or by the
            // operations before it), and the stream orders the launches; it is synchronized before a slot is read.
            unsafe { self.op(&self.big, l, i, &self.big.d.layers[l][i], n, (pp, pp), 0)? };
            self.trace_pick(&self.big, l, &self.big.d.layers[l][i], n, pp)?;
            self.trace_comp_pick(&self.big, l, &self.big.d.layers[l][i], n, pp)?;
        }
        self.dev.sync()
    }

    /// Lagged mHC's weights for the rows of the next pass (`hc_lag` [rows, 4] fp32, which the first lagged block of a layer reads).
    fn kc_put_lag(&self, lag: &[f32]) -> Result<()> {
        self.hc_lag.write(0, lag)
    }
}

/// Sequence `seq`'s latent cache in a block of the reference (made where the block has not run it yet).
fn latent_cache(b: &mut RefLayer, seq: u32) -> Result<&mut LatentCache> {
    match b.seqs.entry(seq).or_insert_with(|| SeqState::Mla(LatentCache::default())) {
        SeqState::Mla(c) => Ok(c),
        _ => err("a latent attention's sequence state is not a latent cache"),
    }
}

/// A latent attention's indexer of its own in a block (none for a shared one, which takes the picks of the layer before), or a compressed attention's that
/// picks for itself (an index source: none for one that reads another's picks), as the picks' judge takes it (an entry a pool of its own).
fn own_indexer(b: &RefLayer) -> Option<Indexer> {
    match &b.block.body {
        Body::Mixer(Mixer::Mla(x)) => x.indexer.clone().filter(|ix| !ix.shared),
        Body::Mixer(Mixer::Compressed(c)) if c.ratio > 0 && c.share.is_none_or(|s| s.picks.is_none()) => c.indexer.clone().map(|ix| Indexer { block: 1, ..ix }),
        _ => None,
    }
}

/// Sequence `seq`'s picks to take and record of choices in a block of the reference whose indexer picks for itself: a latent attention's cache's, or a
/// compressed attention's state's (each made where the block has not run the sequence yet).
fn picks_of(b: &mut RefLayer, seq: u32) -> Result<(&mut std::collections::VecDeque<Vec<usize>>, &mut Option<Vec<Picked>>)> {
    let comp = matches!(b.block.body, Body::Mixer(Mixer::Compressed(_)));
    match b.seqs.entry(seq).or_insert_with(|| if comp { SeqState::Comp(CompState::default()) } else { SeqState::Mla(LatentCache::default()) }) {
        SeqState::Mla(c) => Ok((&mut c.take, &mut c.picked)),
        SeqState::Comp(c) => Ok((&mut c.take, &mut c.picked)),
        _ => err("an indexer's sequence state is neither a latent cache nor a compressed attention's"),
    }
}

/// The rows of a body's pass where the engine's indexer picks (`rows`, traced with their inputs) are not the reference's own (`picked`, the reference's
/// records of the same rows, taking the engine's picks): each judged by the reference's indexer on the engine's own input (`logits::picks_apart`, the rule
/// the logits checks hold an engine to), and named; those whose choice there differs by `NEAR` or more (or that no tie explains: NaN) are returned.
fn indexer_apart(ix: &Indexer, picked: Vec<Picked>, rows: Vec<crate::logits::PickRow>, at: usize, l: usize, b: usize) -> Vec<String> {
    let mut bad = Vec::new();
    for a in crate::logits::picks_apart(&[picked], &vec![rows], std::slice::from_ref(ix)) {
        let line = format!(
            "position {} (pools only the reference's own {:?}, only the engine's {:?}: {:.1e} apart on the engine's own input, {:.1e} on the reference's)",
            at + a.token,
            a.reference,
            a.engine,
            a.margin,
            a.own_margin
        );
        println!("     L{l} B{b} the engine's picks are not the reference's own at {line}");
        if !(a.margin < NEAR) {
            bad.push(line);
        }
    }
    bad
}

/// What one pass of a layer compares: the comparisons' (where, kind, class, value, the rows near a tie, the indexer's choices apart).
type Got = Vec<(String, String, Class, f32, Vec<String>, Vec<String>)>;

/// A pass's rows: their tokens, the first's position, each row's residual before the layer, and with lagged mHC the weights its first block takes ([rows, 4]).
struct Rows<'a> {
    at: usize,
    tok: &'a [u32],
    x0: Vec<f32>,
    lag: Option<Vec<f32>>,
}

/// Layer `l` of the executor (blocks `starts[l]..starts[l + 1]`, `block_starts`) on `rows` (the depth mix's completed groups made by `rng`). With what it
/// compared, the residual the layer left (as the slot holds it).
/// `lend`: the picks the last latent attention with an indexer of its own took for the pass's rows (each row's entries), which a layer whose indexer is
/// shared attends to, as the engine's does (`MlaOp::lent`); set by such a body here.
fn pass(dy: &Dyn, m: &mut RefModel, l: usize, starts: &[usize], rows: Rows, rng: &mut Rng, lend: &mut Option<Vec<Vec<usize>>>) -> Result<(Got, Vec<f32>)> {
    let Rows { at, tok, x0, lag } = rows;
    let (d, layers, n) = (m.desc.hidden, dy.big.d.layers.len(), tok.len());
    let (first, per) = (starts[l], starts[l + 1] - starts[l]);
    let runs_l = runs(&dy.big.d.layers[l]);
    // (the residual: mHC's streams where the model has them, else X; the depth mix's completed groups in Blk)
    let state = dy.kc_state();
    let sw = dy.kc_width(state);
    if x0.len() != n * sw {
        return err(format!("layer {l}: a residual of {} values for {n} rows of {sw}", x0.len()));
    }
    dy.kc_put(state, &x0)?;
    for s in [Slot::O, Slot::D] {
        if dy.kc_width(s) > 0 {
            dy.kc_put(s, &vec![0.0; n * dy.kc_width(s)])?;
        }
    }
    let depth = matches!(m.layers[first].block.residual, Residual::DepthMix(_));
    let (bw, nb) = (dy.kc_width(Slot::Blk), m.layers[..first].iter().filter(|b| matches!(b.block.residual, Residual::DepthMix(dm) if dm.starts)).count());
    let mut rows: Vec<Row> = Vec::with_capacity(n);
    let blk = if bw > 0 {
        let mut b = vec![0f32; n * bw];
        for t in 0..n {
            b[t * bw..][..nb * d].copy_from_slice(&rng.v(nb * d));
        }
        dy.kc_put(Slot::Blk, &b)?;
        b
    } else {
        Vec::new()
    };
    // (lagged mHC: the weights the layer's first block takes, each row's, on the GPU too)
    if let Some(w) = &lag {
        dy.kc_put_lag(w)?;
    }
    for t in 0..n {
        let x = x0[t * sw..][..sw].to_vec();
        let dp = depth.then(|| Depth { sources: (0..nb).map(|j| blk[t * bw + j * d..][..d].to_vec()).collect(), partial: Some(x.clone()) });
        let lw = lag.as_ref().map(|w| w[t * 4..][..4].to_vec());
        rows.push(Row { x, depth: dp, hc: None, lag: lw });
    }
    let mut got: Got = Vec::new();
    let mut put = |at: String, kind: String, class: Class, v: f32, near: Vec<String>, apart: Vec<String>| got.push((at, kind, class, v, near, apart));
    // the reference's input to block b (or the final norm's, b = None), from each row's residual: H as the reference makes it
    let pre = |m: &RefModel, rows: &mut Vec<Row>, b: Option<usize>| -> Result<Vec<f32>> {
        let mut h = Vec::with_capacity(n * d);
        for r in rows.iter_mut() {
            match b {
                None => h.extend(m.final_h_with(&r.x, r.depth.as_ref(), r.lag.as_deref().unwrap_or(&[]))?),
                Some(b) => {
                    let blk = &m.layers[b];
                    let input = match (&blk.block.residual, r.depth.as_mut()) {
                        (Residual::DepthMix(dm), Some(dp)) => dp.input(dm.starts, blk.p.vec("mix_norm")?, blk.p.vec("mix_proj")?),
                        // (lagged mHC: weighed by the block before's weights, this block's own kept for the next)
                        (Residual::Streams(s @ StreamsP { lagged: true, .. }), _) => {
                            let (own, post, comb) = hc_weights(s, &blk.p, &r.x)?;
                            let w = r.lag.replace(own).ok_or_else(|| Error(format!("block {b}: lagged mHC with no weights before it")))?;
                            r.hc = Some((post, comb));
                            collapse(s, &w, &r.x, d)
                        }
                        _ => {
                            let (input, hc) = blk.pre(&r.x)?;
                            r.hc = hc;
                            input
                        }
                    };
                    h.extend(blk.norm_in(&input));
                }
            }
        }
        Ok(h)
    };
    // the residual rule of block b with the GPU's output o of its body
    let post = |m: &RefModel, rows: &mut Vec<Row>, b: usize, o: &[f32]| -> Result<()> {
        let blk = &m.layers[b];
        for (t, r) in rows.iter_mut().enumerate() {
            let ot = blk.norm_out(o[t * d..][..d].to_vec());
            match r.depth.as_mut() {
                Some(dp) => dp.add(&ot),
                None => blk.post(&mut r.x, &ot, r.hc.take())?,
            }
        }
        Ok(())
    };
    // the GPU's residual against the reference's
    let state_rel = |rows: &Vec<Row>| -> Result<f32> {
        let x = dy.kc_get(state, n)?;
        let want: Vec<f32> = rows.iter().flat_map(|r| r.depth.as_ref().map_or(r.x.clone(), |dp| dp.partial.clone().unwrap_or(vec![0.0; d]))).collect();
        let mut v = rel(&x, &want);
        if depth {
            let b = dy.kc_get(Slot::Blk, n)?;
            let (got, want): (Vec<f32>, Vec<f32>) = rows
                .iter()
                .enumerate()
                .flat_map(|(t, r)| {
                    let dp = r.depth.as_ref().expect("a depth mix");
                    dp.sources.iter().enumerate().flat_map(move |(j, s)| (0..d).map(move |i| (t, j, i, s[i])))
                })
                .map(|(t, j, i, w)| (b[t * bw + j * d + i], w))
                .unzip();
            v = v.max(rel(&got, &want));
        }
        Ok(v)
    };
    // (an n-gram embedding's input is the residual itself: before it the residual alone is compared)
    let ngram = |m: &RefModel, b: usize| matches!(m.layers[b].block.body, Body::Ngram(_));
    let mut pending: Option<(usize, Vec<f32>)> = None;
    let mut body = 0;
    let mut last_ran = false;
    for (k, (run, r)) in runs_l.iter().enumerate() {
        let b = first + body;
        if *run == Run::Residual {
            dy.kc_ops(l, r.clone(), n, at, &tok)?;
            last_ran = true;
            if body == per {
                // (the trailing run: compared with the next layer's first run, below)
                continue;
            }
            if let Some((pb, o)) = pending.take() {
                post(m, &mut rows, pb, &o)?;
            }
            // (the reference's input first: a depth mix's group closes there)
            let v = if ngram(m, b) {
                state_rel(&rows)?
            } else {
                let h = pre(m, &mut rows, Some(b))?;
                state_rel(&rows)?.max(rel(&dy.kc_get(Slot::H, n)?, &h))
            };
            put(format!("L{l} before B{b}"), residual_label(Some(&m.layers[b].block)), Class::Residual, v, Vec::new(), Vec::new());
            continue;
        }
        if body >= per {
            return err(format!("layer {l}: more bodies than its {per} blocks"));
        }
        if *run == Run::Ngram {
            if !ngram(m, b) {
                return err(format!("layer {l}: an n-gram embedding's operations where block {b} is {:?}", m.layers[b].block.body));
            }
            if let Some((pb, o)) = pending.take() {
                post(m, &mut rows, pb, &o)?;
            }
            let before: Vec<f32> = rows.iter().flat_map(|r| r.x.clone()).collect();
            dy.kc_ops(l, r.clone(), n, at, &tok)?;
            last_ran = false;
            // (the reference's block on each row's residual: its norms per stream, its body with the row's token, its residual rule)
            for (t, row) in rows.iter_mut().enumerate() {
                let (input, hc) = m.layers[b].pre(&row.x)?;
                let h = m.layers[b].norm_in(&input);
                let o = m.body(b, 0, (at + t) as u32, tok[t], &h)?;
                let o = m.layers[b].norm_out(o);
                m.layers[b].post(&mut row.x, &o, hc)?;
            }
            let (class, kind) = label(&m.layers[b].block);
            let got = dy.kc_get(state, n)?;
            let want: Vec<f32> = rows.iter().flat_map(|r| r.x.clone()).collect();
            let own = |v: &[f32]| v.iter().zip(&before).map(|(a, b)| a - b).collect::<Vec<f32>>();
            println!("     L{l} B{b} {kind}: its own output (the streams' change) {:.2e} of its largest magnitude, not judged", rel(&own(&got), &own(&want)));
            put(format!("L{l} B{b}"), kind, class, rel(&got, &want), Vec::new(), Vec::new());
            body += 1;
            continue;
        }
        if k == 0 {
            // (a layer that begins with a body reads the H the layer before made: the reference's, rounded as a slot holds it)
            let h = pre(m, &mut rows, Some(b))?;
            dy.kc_put(Slot::H, &h)?;
        }
        let h = dy.kc_get(Slot::H, n)?;
        // (a latent attention with an indexer of its own: the engine's picks traced with the inputs it made them from, and the reference taking them, so
        // that each row is compared on the engine's choice and each choice apart is judged on the engine's own input: `indexer_apart`)
        let ix = own_indexer(&m.layers[b]);
        dy.trace_picks(ix.is_some());
        dy.kc_ops(l, r.clone(), n, at, &tok)?;
        let traced = dy.trace_picks(false);
        last_ran = false;
        let out = out_slot(&dy.big.d.layers[l][r.end - 1])?;
        if dy.kc_width(out) != d {
            return err(format!("layer {l}: a body's output slot {out:?} of {} values, not {d}", dy.kc_width(out)));
        }
        let o = dy.kc_get(out, n)?;
        let rows_picked: Vec<crate::logits::PickRow> =
            traced
                .into_iter()
                .flat_map(|t| {
                    let keys = std::rc::Rc::new(t.keys);
                    t.pos
                        .into_iter()
                        .zip(t.picks)
                        .zip(t.q.into_iter().zip(t.w))
                        .zip(t.inside)
                        .map(move |(((pos, picks), (q, w)), inside)| crate::logits::PickRow { pos, picks, q, w, keys: keys.clone(), inside })
                })
                .collect();
        if ix.is_some() {
            if rows_picked.len() != n {
                return err(format!("L{l} B{b}: the picks of {} rows traced, where the pass has {n}", rows_picked.len()));
            }
            let (take, picked) = picks_of(&mut m.layers[b], 0)?;
            (*take, *picked) = (rows_picked.iter().map(|r| r.picks.clone()).collect(), Some(Vec::new()));
            *lend = Some(rows_picked.iter().map(|r| r.picks.clone()).collect());
        }
        // (a layer whose indexer is shared: each row attends to the picks the latent attention with an indexer before it took for that row)
        let shared = matches!(&m.layers[b].block.body, Body::Mixer(Mixer::Mla(x)) if x.indexer.as_ref().is_some_and(|i| i.shared));
        if shared && lend.as_ref().is_none_or(|p| p.len() != n) {
            return err(format!("L{l} B{b}: a shared indexer's layer, with no picks of this pass's {n} rows lent to it"));
        }
        let (mut want, mut margin) = (Vec::with_capacity(n * d), Vec::with_capacity(n));
        for t in 0..n {
            if let Some(p) = lend.as_ref().filter(|_| shared) {
                latent_cache(&mut m.layers[b], 0)?.lent = std::iter::once(p[t].clone()).collect();
            }
            want.extend(m.body(b, 0, (at + t) as u32, tok[t], &h[t * d..][..d])?);
            // (a row on the engine's own picks is compared whatever the reference's own margin: none is left out as near a tie)
            margin.push(if ix.is_some() { f32::INFINITY } else { m.layers[b].margin });
        }
        let apart = match &ix {
            Some(ix) => {
                let (take, picked) = picks_of(&mut m.layers[b], 0)?;
                let picked = picked.take().unwrap_or_default();
                take.clear();
                indexer_apart(ix, picked, rows_picked, at, l, b)
            }
            None => Vec::new(),
        };
        // (each row's largest difference over the pass's scale; a row past the bound whose choice was near a tie is left out, and named)
        let (class, kind) = label(&m.layers[b].block);
        let scale = want.iter().fold(0f32, |a, v| a.max(v.abs())).max(1e-30);
        let (mut worst, mut near) = (0f32, Vec::new());
        for t in 0..n {
            let r = rel(&o[t * d..][..d], &want[t * d..][..d]) * want[t * d..][..d].iter().fold(0f32, |a, v| a.max(v.abs())).max(1e-30) / scale;
            // (GLYD_KINDCHECK_ROWS: each row's difference, a body at a time)
            if crate::tune!("GLYD_KINDCHECK_ROWS").is_some() {
                println!("     L{l} B{b} position {}: {r:.2e} (margin {:.1e})", at + t, margin[t]);
            }
            if margin[t] < NEAR && r > bound(class) {
                near.push(format!("position {} (margin {:.1e}, off by {r:.1e})", at + t, margin[t]));
            } else {
                worst = worst.max(r);
            }
        }
        put(format!("L{l} B{b}"), kind, class, worst, near, apart);
        pending = Some((b, o));
        body += 1;
    }
    if body != per {
        return err(format!("layer {l}: {body} bodies, where it has {per} blocks"));
    }
    // the residual after the last body: the layer's last run and the next layer's first (its norm), or the final norm
    let next = (l + 1 < layers).then(|| runs(&dy.big.d.layers[l + 1]).into_iter().next().filter(|(k, _)| *k == Run::Residual)).flatten();
    if let Some((_, r)) = &next {
        dy.kc_ops(l + 1, r.clone(), n, at, &tok)?;
    }
    if last_ran || next.is_some() {
        if let Some((pb, o)) = pending.take() {
            post(m, &mut rows, pb, &o)?;
        }
        let b = (l + 1 < layers).then_some(starts[l + 1]);
        let v = match b {
            Some(b) if ngram(m, b) => state_rel(&rows)?,
            _ => {
                let h = pre(m, &mut rows, b)?;
                state_rel(&rows)?.max(rel(&dy.kc_get(Slot::H, n)?, &h))
            }
        };
        let what = b.map_or("the final norm".to_string(), |b| format!("B{b}"));
        put(format!("L{l} before {what}"), residual_label(b.map(|b| &m.layers[b].block)), Class::Residual, v, Vec::new(), Vec::new());
    }
    Ok((got, dy.kc_get(state, n)?))
}

/// The tokens of a pass of n rows at `at` (DeepSeek-V4's hash routing and the per-layer inputs read them; `--chain` embeds them).
fn tokens(n: usize, at: usize, vocab: usize) -> Vec<u32> {
    (0..n).map(|t| (((at + t) * 37 + 11) % vocab) as u32).collect()
}

/// How a case's passes are made: each layer's residual at random, or one run of the model (`--chain`, `--sequence`).
#[derive(Clone, Copy, PartialEq)]
enum Passes {
    Random,
    /// a layer's residual the one the GPU's layer before made (the first's the tokens' scaled embedding rows): the whole model's sequence of caches and
    /// positions, each block still on the GPU's own inputs
    Chain,
    /// the module's note
    Sequence,
}

/// A case's model made (its logits case's random one), loaded lossless, described and loaded on the reference; every layer through `pass`, a pass
/// (the prompt's rows, then each step's) at a time, in a model's order.
fn case(id: &'static str, dir: &Path, prompt: usize, steps: usize, how: Passes) -> Result<Vec<Line>> {
    let chain = how == Passes::Chain;
    let c = crate::logits::find_case(id)?;
    if !dir.join("config.json").exists() {
        match c.weights {
            crate::logits::Weights::Random(make) => make(dir)?,
            crate::logits::Weights::Hub(h) => return err(format!("{id}: a Hub model ({h}), not a random one")),
        }
    }
    // (--chain: the logits cases' options, a context of 1,024 and the GPU's layout at the lossless level)
    let o = Opts { ctx: if chain { 1024 } else { 256 }, chunk: 64, graph: false, device: 0, layout: if chain { None } else { Some(Layout::Tiered) } };
    let mut dy = match id {
        // (its logits case runs the experts lossy; here the lossless level's)
        "glm" => super::glm::load(dir, &o, Layout::Tiered)?,
        "nemotron-h" => crate::logits::load_dyn("nemotron", dir, &o, "penguin")?,
        _ => crate::logits::load_dyn(id, dir, &o, "penguin")?,
    };
    dy.reset()?;
    let cfg: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("config.json")).map_err(|e| Error(e.to_string()))?)
        .map_err(|e| Error(format!("config.json: {e}")))?;
    let d = crate::runtime::describe::describe(&cfg)?;
    let ck = glyd_gpu::safetensors::Checkpoint::open(dir).map_err(|e| Error(e.to_string()))?;
    let mut m = RefModel::load(&d, &|n: &str| checkpoint_tensor(&ck, n))?;
    let layers = dy.big.d.layers.len();
    let starts = block_starts(&dy.big.d.layers);
    if layers == 0 || starts[layers] != m.layers.len() {
        return err(format!("{id}: {} blocks, where the executor's {layers} layers run {}", m.layers.len(), starts[layers]));
    }
    if prompt + steps + 1 > o.ctx || prompt > o.chunk {
        return err(format!("a prompt of {prompt} and {steps} steps: a chunk of {} and a cache of {}", o.chunk, o.ctx));
    }
    if chain && (dy.kc_width(Slot::S) > 0 || dy.kc_width(Slot::Blk) > 0) {
        return err(format!("{id}: --chain takes a model of one residual stream"));
    }
    // (--sequence: the reference alone, a second copy, and the sequence's tokens)
    let mut alone = match how {
        Passes::Random | Passes::Chain => None,
        Passes::Sequence
            if m.layers
                .iter()
                .any(|b| matches!(b.block.residual, Residual::DepthMix(_) | Residual::Streams(crate::runtime::kinds::StreamsP { lagged: true, .. }))) =>
        {
            return err(format!("{id}: --sequence is not for a depth mix or lagged mHC"));
        }
        Passes::Sequence => Some((RefModel::load(&d, &|n: &str| checkpoint_tensor(&ck, n))?, crate::logits::token_list(prompt + steps, m.desc.vocab, 0))),
    };
    let sw = dy.kc_width(dy.kc_state());
    let lagged = m.layers.iter().any(|b| matches!(b.block.residual, Residual::Streams(StreamsP { lagged: true, .. })));
    let mut rngs: Vec<Rng> = (0..layers)
        .map(|l| Rng(0x2026_1123_0000_0001 ^ ((l as u64 + 1) << 32) ^ id.bytes().fold(0u64, |h, b| h.wrapping_mul(31).wrapping_add(b as u64))))
        .collect();
    let mut lines: Vec<Vec<Line>> = (0..layers).map(|_| Vec::new()).collect();
    // (each layer's residual from the reference's own after it: the prompt's, the steps' worst)
    let mut apart = vec![(0f32, 0f32); layers];
    let (hd, scale) = (m.desc.hidden, m.desc.embed.scale);
    for (at, n) in std::iter::once((0, prompt)).chain((0..steps).map(|s| (prompt + s, 1))) {
        let tok: Vec<u32> = match &alone {
            Some((_, t)) => t[at..at + n].to_vec(),
            None => tokens(n, at, m.desc.vocab),
        };
        let mut own: Vec<Vec<f32>> = Vec::new();
        if let Some((m2, _)) = alone.as_mut() {
            m2.forward_seen(&tok, at as u32, &mut |x| own.push(x.to_vec()))?;
        }
        // (the first layer's residual: --chain the tokens' scaled embedding rows, --sequence the reference's own; at random each layer's otherwise)
        // (the picks a layer whose indexer is shared takes: the last latent attention's with one of its own, on this pass's rows)
        let mut lend = None;
        let mut x = match how {
            Passes::Chain => Some(tok.iter().flat_map(|&t| m.embed[t as usize * hd..][..hd].iter().map(|&v| bf16(v * bf16(scale)))).collect::<Vec<f32>>()),
            _ => own.first().cloned(),
        };
        for l in 0..layers {
            let x0 = match x.take() {
                Some(x) => x,
                None => rngs[l].v(n * sw),
            };
            // (lagged mHC: the first layer's one-hot on stream 0, as the model's first block takes; another layer's at random in (0.1, 1), as sigmoid's plus
            // hc_eps are)
            let lag = lagged.then(|| match l {
                0 => (0..n * 4).map(|i| (i % 4 == 0) as u32 as f32).collect(),
                _ => rngs[l].v(n * 4).into_iter().map(|v| 0.55 + 0.45 * v).collect(),
            });
            let (got, after) = pass(&dy, &mut m, l, &starts, Rows { at, tok: &tok, x0, lag }, &mut rngs[l], &mut lend)?;
            if alone.is_some() {
                let want = &own[starts[l + 1]];
                let v = rel(&after, want);
                if crate::tune!("GLYD_KINDCHECK_ROWS").is_some() {
                    for t in 0..n.min(want.len() / sw.max(1)) {
                        println!(
                            "     L{l} after the layer, position {}: {:.2e} from the reference's own",
                            at + t,
                            rel(&after[t * sw..][..sw], &want[t * sw..][..sw])
                        );
                    }
                }
                apart[l] = if at == 0 { (v, 0.0) } else { (apart[l].0, apart[l].1.max(v)) };
            }
            if how != Passes::Random {
                x = Some(after);
            }
            if at == 0 {
                lines[l] = got
                    .into_iter()
                    .map(|(w, kind, class, v, near, apart)| Line { case: id, at: w, kind, class, prompt: v, steps: 0.0, near, rows: prompt + steps, apart })
                    .collect();
                continue;
            }
            if got.len() != lines[l].len() {
                return err(format!("{id} layer {l}: a step compared {} things, the prompt {}", got.len(), lines[l].len()));
            }
            for (line, (_, _, _, v, near, apart)) in lines[l].iter_mut().zip(got) {
                line.steps = line.steps.max(v);
                line.near.extend(near);
                line.apart.extend(apart);
            }
        }
    }
    for (l, of) in lines.iter().enumerate() {
        for x in of {
            let ok = x.ok();
            let near = if x.near.is_empty() {
                String::new()
            } else if near_within(x.near.len(), x.rows) {
                format!("; past the bound near a tie, not compared: {}", x.near.join(", "))
            } else {
                format!("; {} of its {} rows past the bound near a tie and not compared, more than one in sixteen: {}", x.near.len(), x.rows, x.near.join(", "))
            };
            let apart = if x.apart.is_empty() {
                String::new()
            } else {
                format!("; the indexer's picks apart from the reference's on the engine's own input, by NEAR or more: {}", x.apart.join(", "))
            };
            println!(
                "{} {:11} {:22} {}: prompt {:.2e}, steps {:.2e} (bound {:.0e}){near}{apart}",
                if ok { "ok  " } else { "FAIL" },
                x.case,
                x.at,
                x.kind,
                x.prompt,
                x.steps,
                bound(x.class)
            );
        }
        if alone.is_some() {
            println!(
                "     {id:11} {:22} the executor's residual from the reference's own run of the sequence: prompt {:.2e}, steps {:.2e} (no bound: everything before it included)",
                format!("L{l} after the layer"),
                apart[l].0,
                apart[l].1
            );
        }
    }
    Ok(lines.into_iter().flatten().collect())
}

/// `kindcheck [CASE...] [--prompt N] [--steps N] [--scratch DIR] [--chain | --sequence]`: the cases (all unless named), then each class's worst against its bound.
pub fn run(a: &crate::args::Args) -> Result<()> {
    const ALL: [&str; 17] = [
        "gemma4",
        "gemma4-kv",
        "qwen35",
        "qwen35-wide",
        "qwen35moe",
        "dense",
        "nemotron-h",
        "muse",
        "kimi",
        "kimi-dsa",
        "glm",
        "glm-dsa",
        "glm-dsa-f8",
        "deepseek-v4",
        "deepseek-v41",
        "qwen4x",
        "qwen4x-s2",
    ];
    let want: Vec<&str> = a.pos.iter().skip(1).map(String::as_str).collect();
    if let Some(w) = want.iter().find(|w| !ALL.contains(w)) {
        return err(format!("no case {w}: {}", ALL.join(", ")));
    }
    let ids: Vec<&'static str> = ALL.iter().copied().filter(|c| want.is_empty() || want.contains(c)).collect();
    let (prompt, steps) = (a.num("prompt", 48usize)?, a.num("steps", 4usize)?);
    let how = match (a.has("chain"), a.has("sequence")) {
        (true, true) => return err("--chain or --sequence, not both".to_string()),
        (true, false) => Passes::Chain,
        (false, true) => Passes::Sequence,
        (false, false) => Passes::Random,
    };
    // (GLYD_KINDCHECK_SIM=1 keeps the release's simulated quantizations on: each body then measures how far they take the executor from the kinds' f32
    // reference, which has none)
    if std::env::var_os("GLYD_KINDCHECK_SIM").is_none() {
        std::env::set_var("GLYD_V4_SIM", "0");
    }
    let (scratch, own) = match a.flag("scratch") {
        Some(s) => (PathBuf::from(s), false),
        None => (std::env::temp_dir().join(format!("glyd-kindcheck-{}", std::process::id())), true),
    };
    let mut all: Vec<Line> = Vec::new();
    let mut failed: Vec<String> = Vec::new();
    for id in ids {
        match case(id, &scratch.join(id), prompt, steps, how) {
            Ok(l) => all.extend(l),
            Err(e) => {
                println!("FAIL {id}: {}", e.msg);
                failed.push(format!("{id}: {}", e.msg));
            }
        }
    }
    if own {
        let _ = std::fs::remove_dir_all(&scratch);
    }
    println!("\nthe worst of each kind (the largest difference over the reference's largest magnitude):");
    let mut classes: Vec<Class> = all.iter().map(|x| x.class).collect();
    classes.sort();
    classes.dedup();
    for c in classes {
        let of: Vec<&Line> = all.iter().filter(|x| x.class == c).collect();
        let worst = of.iter().fold(0f32, |m, x| m.max(x.prompt).max(x.steps));
        let near: usize = of.iter().map(|x| x.near.len()).sum();
        let mut cases: Vec<&str> = of.iter().map(|x| x.case).collect();
        cases.dedup();
        let near = if near > 0 { format!(" ({near} rows past the bound near a tie, not compared)") } else { String::new() };
        println!("  {:12} {worst:.2e} (bound {:.0e}) over {} comparisons{near}: {}", format!("{c:?}"), bound(c), of.len(), cases.join(", "));
    }
    failed.extend(all.iter().filter(|x| !x.ok()).map(|x| format!("{} {} {}", x.case, x.at, x.kind)));
    if !failed.is_empty() {
        return err(format!("kindcheck: {} not within bounds: {}", failed.len(), failed.join("; ")));
    }
    println!("kindcheck: PASS");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A value of the reference that is not a number fails the comparison, as one of the GPU's does: `f32::max` drops a NaN difference, which read as 0.
    #[test]
    fn a_reference_that_is_not_a_number_is_not_within() {
        assert_eq!(rel(&[1.0, 2.0], &[1.0, 2.0]), 0.0);
        assert_eq!(rel(&[1.0, 2.5], &[1.0, 2.0]), 0.25);
        assert_eq!(rel(&[1.0, f32::NAN], &[1.0, 2.0]), f32::INFINITY);
        assert_eq!(rel(&[1.0, 2.0], &[1.0, f32::NAN]), f32::INFINITY);
        assert_eq!(rel(&[1.0, 2.0], &[f32::INFINITY, 2.0]), f32::INFINITY);
        assert_eq!(rel(&[1.0], &[1.0, 2.0]), f32::INFINITY);
    }

    /// A body may leave few rows out as near a tie: the runs on record left out 1 of 52, and one whose rows were all left out fails though its worst reads 0.
    #[test]
    fn a_body_that_leaves_most_rows_out_fails() {
        let line = |near: usize, rows: usize| Line {
            case: "t",
            at: "L0 B0".into(),
            kind: "latent attention".into(),
            class: Class::Latent,
            prompt: 0.0,
            steps: 0.0,
            near: vec!["position 0".to_string(); near],
            rows,
            apart: Vec::new(),
        };
        assert!(line(0, 52).ok() && line(1, 52).ok() && line(3, 52).ok());
        assert!(!line(4, 52).ok() && !line(52, 52).ok());
        assert!(line(1, 5).ok() && !line(2, 5).ok());
        // (and a bound still fails with nothing left out, as an indexer's choice apart does within it)
        let mut past = line(0, 52);
        past.steps = 2.5e-2;
        assert!(!past.ok());
        let mut apart = line(0, 52);
        apart.apart.push("position 3".to_string());
        assert!(!apart.ok());
    }

    /// Qwen3.8-Flash-Next's tiny model: the n-gram embedding's three operations before the second layer's mixer are a run of their own (its add ends it), so
    /// that layer has three blocks and every other two, and the layers' blocks are the description's, in its order.
    #[test]
    fn an_n_gram_embedding_is_a_block_of_its_own() {
        let m: serde_json::Value = serde_json::from_str(include_str!("../../tests/fixtures/qwen4x_tiny_manifest.json")).unwrap();
        let c: serde_json::Value = serde_json::from_str(m["config"].as_str().unwrap()).unwrap();
        let d = crate::runtime::describe::describe(&c).unwrap();
        let x = super::super::lower::qwen4x(&d, vec![2000], 1100).unwrap();
        let kinds = |l: usize| runs(&x.layers[l]).into_iter().map(|(k, r)| (k, r.len())).collect::<Vec<_>>();
        let gated = [(Run::Residual, 1), (Run::Body, 6), (Run::Residual, 2), (Run::Body, 1), (Run::Residual, 1)];
        assert_eq!(kinds(0), gated);
        assert_eq!(kinds(1)[0], (Run::Ngram, 3));
        assert_eq!(kinds(1)[1..], gated);
        let starts = block_starts(&x.layers);
        assert_eq!(starts.len(), x.layers.len() + 1);
        assert_eq!(starts[x.layers.len()], d.desc.blocks.len());
        assert_eq!(&starts[..3], &[0, 2, 5]);
        for (l, b) in starts.windows(2).enumerate() {
            let at = &d.desc.blocks[b[0]..b[1]];
            assert_eq!(matches!(at[0].body, crate::runtime::kinds::Body::Ngram(_)), l == 1, "layer {l}");
            assert!(matches!(at.last().unwrap().body, crate::runtime::kinds::Body::Ffn(_)), "layer {l}");
        }
    }

    /// Two n-gram embeddings one after the other are two runs, and a run of the residual's operations between bodies stays one.
    #[test]
    fn runs_end_where_their_kind_changes_or_an_n_gram_embedding_adds() {
        use super::super::ngram::{NgramAdd, NgramRows};
        let rows = Op::NgramRows(NgramRows { table: 0, hash: [0; 3], hist: 0, per: 1, row: 1, eos: 0, order: 3, map: None, segments: true });
        let add = Op::NgramAdd(NgramAdd { q: 0, k: 0, conv: None, plus_one: true, eps: 1e-6 });
        let lin = Op::Linear { w: 0, x: Slot::In, y: Slot::Gu, name: "" };
        let ops = [rows.clone(), lin.clone(), add.clone(), rows, lin.clone(), add, Op::Nop, Op::Nop, lin];
        let got: Vec<(Run, usize)> = runs(&ops).into_iter().map(|(k, r)| (k, r.len())).collect();
        assert_eq!(got, [(Run::Ngram, 3), (Run::Ngram, 3), (Run::Residual, 2), (Run::Body, 1)]);
        assert_eq!(block_starts(&[ops.to_vec()]), [0, 3]);
    }
}
