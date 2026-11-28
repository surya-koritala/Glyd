//! The executor's entry points by kind (R2, #135), which the CUDA backend (`runtime::cuda`) builds its layer kinds from: a layer's operations cut into
//! runs (the residual's, a body's, an n-gram embedding's: `runs`), each body the block of its kind in the description (`block_starts`), the residual's
//! runs the pieces between them (`pieces`); and the calls a piece makes on a pass: its operations through `Dyn::op`, the one function `forward` runs
//! them through too (a piece adds no copy and no wait to what the executor launches), the embedding (`Dyn::embed` and the description's first
//! operation) and the head (`Dyn::logits_pass`, `forward`'s own).

use super::{Dyn, Op, Slot};
use crate::ffi::{err, Result};
use std::ops::Range;

/// Whether an operation is the residual's (it reads or writes X, mHC's streams or the depth mix's blocks, or it is a block's post-norm of its
/// output in place) and not a body's.
pub(crate) fn residual_op(op: &Op) -> bool {
    match *op {
        Op::AddNorm { .. } | Op::NormAdd { .. } | Op::EmbedNorm { .. } | Op::GatedResidual { .. } | Op::Nop => true,
        Op::RmsNorm { x, y, .. } => x == Slot::X || x == y,
        Op::Streams | Op::StreamsPre { .. } | Op::StreamsPost { .. } | Op::StreamsHead { .. } => true,
        Op::DepthMix(_) | Op::Add { .. } | Op::MoeEarly { .. } => true,
        _ => false,
    }
}

/// What a run of a layer's operations is: the residual's, a body's (its output in a slot), or an n-gram embedding's (`Op::NgramRows` to
/// `Op::NgramAdd`: its output added into the streams in place).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Run {
    Residual,
    Body,
    Ngram,
}

/// A layer's operations as runs: (what each is, its range). An n-gram embedding's operations are a run of their own, ending at its add.
pub(crate) fn runs(ops: &[Op]) -> Vec<(Run, Range<usize>)> {
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
#[cfg(feature = "dev")]
pub(crate) fn block_starts(layers: &[Vec<Op>]) -> Vec<usize> {
    let mut v = vec![0];
    for ops in layers {
        let n = runs(ops).iter().filter(|(k, _)| *k != Run::Residual).count();
        v.push(v[v.len() - 1] + n);
    }
    v
}

/// Where each block's pieces are in the executor's layers, by the block's index in the description: its body (its layer, its operations, what its run
/// is), the residual's operations before it in its layer (an empty range where none), and those after it where it is its layer's last body (the
/// layer's trailing run; empty otherwise: the run between two bodies is the next one's `before`). Every operation of every layer is in exactly one piece.
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct Pieces {
    pub body: Vec<(usize, Range<usize>, Run)>,
    pub before: Vec<(usize, Range<usize>)>,
    pub after: Vec<(usize, Range<usize>)>,
}

/// The pieces of a description's layers (`Pieces`); a layer with no body (its residual's operations belong to no block) is refused.
pub(crate) fn pieces(layers: &[Vec<Op>]) -> Result<Pieces> {
    let mut p = Pieces::default();
    for (l, ops) in layers.iter().enumerate() {
        let mut open: Option<Range<usize>> = None;
        let mut bodies = 0;
        for (run, r) in runs(ops) {
            if run == Run::Residual {
                open = Some(r);
                continue;
            }
            let at = r.start;
            p.before.push((l, open.take().unwrap_or(at..at)));
            p.after.push((l, r.end..r.end));
            p.body.push((l, r, run));
            bodies += 1;
        }
        if bodies == 0 {
            return err(format!("layer {l}: residual operations and no body, which no block's piece holds"));
        }
        if let Some(r) = open {
            let last = p.after.len() - 1;
            p.after[last] = (l, r);
        }
    }
    Ok(p)
}

impl Dyn {
    /// A pass's metadata onto the GPU before its pieces run, as a pass of one sequence takes it: the rows' positions (the first's the sequence's start), their
    /// tokens (the n-gram blocks' rows staged from them), and what `forward` makes once for every layer (`pass_prepare`). Waits for the work before it, as the
    /// executor's own passes do before they write their metadata.
    pub(crate) fn entry_begin(&self, pos: &[u32], tok: &[u32]) -> Result<()> {
        let n = pos.len();
        if n == 0 || n > self.chunk || tok.len() != n || !self.parts.is_empty() {
            return err(format!(
                "a pass of {n} rows ({} tokens): a chunk holds {}, and a pass of several sequences runs through the executor's own",
                tok.len(),
                self.chunk
            ));
        }
        self.dev.sync()?;
        self.pos.write(0, &pos.iter().map(|&p| p as i32).collect::<Vec<i32>>())?;
        if let Some(v) = &self.tok {
            v.write(0, &tok.iter().map(|&t| t as i32).collect::<Vec<i32>>())?;
        }
        self.stage_ngram(tok)?;
        self.start.set(pos[0] as usize);
        let pp = self.pos.ptr;
        // SAFETY: the engine's own buffers, sized for a chunk (n rows at most one, checked above); the stream orders the launches.
        unsafe { self.pass_prepare(&self.big, n, (pp, pp)) }
    }

    /// Operations `ops` of layer `l` on the pass's n rows (`entry_begin`'s), each through `op` as `forward` runs a pass of one sequence, its picks traced where
    /// `trace_picks` asks. Enqueued; nothing waited for (but a traced pick's read).
    pub(crate) fn entry_ops(&self, l: usize, ops: Range<usize>, n: usize) -> Result<()> {
        let layer = self.big.d.layers.get(l).filter(|o| ops.end <= o.len()).ok_or_else(|| crate::ffi::Error(format!("no operations {ops:?} in layer {l}")))?;
        if n == 0 || n > self.chunk {
            return err(format!("{n} rows: a chunk holds {}", self.chunk));
        }
        let pp = self.pos.ptr;
        for i in ops {
            // SAFETY: the engine's own buffers, sized for a chunk (n rows at most one, checked above); its slots hold the rows the operation reads (written by the
            // pieces before it), and the stream orders the launches.
            unsafe { self.op(&self.big, l, i, &layer[i], n, (pp, pp), 0)? };
            self.trace_pick(&self.big, l, &layer[i], n, pp)?;
            self.trace_comp_pick(&self.big, l, &layer[i], n, pp)?;
        }
        Ok(())
    }

    /// The embedding of the pass's tokens: their rows into X (`embed`, from the GPU's table or the host's), then the description's first operation (its scale, norm
    /// or streams), as a pass's first work.
    pub(crate) fn entry_embed(&self, tok: &[u32]) -> Result<()> {
        let n = tok.len();
        if n == 0 || n > self.chunk {
            return err(format!("{n} tokens: a chunk holds {}", self.chunk));
        }
        let pp = self.pos.ptr;
        // SAFETY: as `entry_ops`: X and the slots the first operation writes hold a chunk's rows.
        unsafe {
            self.embed(tok)?;
            self.op(&self.big, 0, 0, &self.big.d.first, n, (pp, pp), 0)
        }
    }

    /// The head on the last `head` of the pass's n rows (`logits_pass`, `forward`'s): their logits, then each row's best token.
    pub(crate) fn entry_head(&self, n: usize, head: usize) -> Result<()> {
        let room = self.big.logits.bytes / (2 * self.big.d.vocab).max(1);
        if head == 0 || head > n || n > self.chunk || head > room {
            return err(format!("the head on {head} rows of {n}: a chunk holds {}, the logits {room}", self.chunk));
        }
        // SAFETY: as `entry_ops`; the logits hold `room` rows (checked above).
        unsafe { self.logits_pass(&self.big, n, head) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A layer's runs make its pieces: the residual's run before a body is that body's `before`, the run after the last body the last's `after`; a body with
    /// none before it has an empty one; every operation is in one piece.
    #[test]
    fn every_operation_is_in_one_piece() {
        let lin = |w| Op::Linear { w, x: Slot::H, y: Slot::O, name: "p" };
        let norm = Op::RmsNorm { x: Slot::X, w: 0, y: Slot::H, weight: super::super::NormW::Plain, eps: 1e-6 };
        let add = Op::Add { x: Slot::O };
        let layers = vec![vec![norm.clone(), lin(0), add.clone(), norm.clone(), lin(1), add.clone()], vec![lin(2), add.clone(), norm.clone(), lin(3)]];
        let p = pieces(&layers).unwrap();
        assert_eq!(p.body.iter().map(|(l, r, _)| (*l, r.clone())).collect::<Vec<_>>(), vec![(0, 1..2), (0, 4..5), (1, 0..1), (1, 3..4)]);
        assert_eq!(p.before, vec![(0, 0..1), (0, 2..4), (1, 0..0), (1, 1..3)]);
        assert_eq!(p.after, vec![(0, 2..2), (0, 5..6), (1, 1..1), (1, 4..4)]);
        for (l, ops) in layers.iter().enumerate() {
            let mut seen = vec![0; ops.len()];
            for r in p.body.iter().map(|x| (x.0, x.1.clone())).chain(p.before.iter().cloned()).chain(p.after.iter().cloned()).filter(|x| x.0 == l).map(|x| x.1)
            {
                r.for_each(|i| seen[i] += 1);
            }
            assert!(seen.iter().all(|&c| c == 1), "layer {l}: {seen:?}");
        }
        assert!(pieces(&[vec![norm, add]]).is_err());
    }
}
