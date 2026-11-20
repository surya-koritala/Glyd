//! The description executor as a scheduler's step (`sched::ModelStep`, docs/design/runtime.md section 2): several sequences, each with its own room of the context (`add_seqs`), run by the one
//! scheduler that runs the dense Qwen3 engine. A step of one sequence is that sequence's pass as a reply alone runs it (its prompt chunk, or its graph's one-token step). A step of several runs their rows
//! in one pass: the operations that act row by row (the products, norms, activations) on all the rows at once, so each weight is read once for the step, and the operations on a sequence's cache or
//! states (the attention's cache write and read, the recurrent layers) on each sequence's rows by themselves with the kernels and row counts its own pass has. Every per-row result is then the one
//! the sequence's own pass gives, except where a product's route depends on its row count (`decoded_way`, past 64 rows; `VERIFY_ROWS`: up to 8 rows the one-row kernel's sums): there the last bits
//! may differ, as they do for the Qwen3 engine's prompts run together. GLYD_ENGINE_BATCH_PREFILL=0 runs prompts one a pass.
//!
//! What a step of several sequences supports is decided per description (`Dyn::batchable`): every operation must be one that acts row by row, or one of the cache operations below that knows where a
//! sequence's rows begin. A description with another runs its sequences one pass each, concurrently but not batched.

use super::{Dyn, Op};
use crate::dev::{copy_d2d_async, memset_async};
use crate::ffi::{err, Result};
use crate::model::FUSED_ROWS;
use crate::sched::{ModelStep, PrefillJob, Row, Sample};

/// One sequence's rows in a pass over several: the first row, how many, where its tokens begin in the sequence, and its caches and recurrent states.
pub(super) struct Part {
    pub row0: usize,
    pub rows: usize,
    pub start: usize,
    pub kc: u64,
    pub vc: u64,
    pub states: Vec<u64>,
}

/// Whether an operation acts on each row alone (no cache, no state, no other row): it runs on the rows of every sequence of a pass at once.
pub(super) fn row_wise(op: &Op) -> bool {
    matches!(
        op,
        Op::EmbedNorm { .. }
            | Op::Linear { .. }
            | Op::F8Lin { .. }
            | Op::AddNorm { .. }
            | Op::GeluMul { .. }
            | Op::Norm { .. }
            | Op::RmsNorm { .. }
            | Op::NormAdd { .. }
            | Op::GatedNorm { .. }
            | Op::GateUpSilu { .. }
            | Op::Nop
    )
}

/// Whether an operation on a sequence's cache or states takes its rows at an offset in a pass over several sequences (`Dyn::rows_before`, `Dyn::cache`, `Dyn::state`): Gemma 4's attention,
/// Qwen3.5's (Qwen3.8's) gated attention and gated delta rule.
fn by_sequence(op: &Op) -> bool {
    matches!(op, Op::QkvPrep { .. } | Op::Attention { .. } | Op::Prep35 { .. } | Op::Attn35 { .. } | Op::GdnConv { .. } | Op::GdnDelta { .. })
}

impl Dyn {
    /// Bytes before the rows of the sequence an operation runs on now, in a slot of `width` values a row (0 outside a pass over several sequences).
    pub(super) fn rows_before(&self, width: usize) -> u64 {
        self.at.get().map_or(0, |k| (self.parts[k].row0 * width * 2) as u64)
    }

    /// Ring `ring`'s keys and values: the sequence's an operation runs on now in a pass over several, else the selected sequence's.
    pub(super) fn cache(&self, ring: usize) -> (u64, u64) {
        let off = self.ring_off[ring];
        match self.at.get() {
            Some(k) => (self.parts[k].kc + off, self.parts[k].vc + off),
            None => (self.kc.ptr + off, self.vc.ptr + off),
        }
    }

    /// Recurrent state `i`: the sequence's an operation runs on now in a pass over several, else the selected sequence's.
    pub(super) fn state(&self, i: usize) -> u64 {
        match self.at.get() {
            Some(k) => self.parts[k].states[i],
            None => self.states[i].ptr,
        }
    }

    /// Whether the rows of several sequences run in one pass (else each sequence's pass by itself): a whole model with no draft model, every operation row-wise or one that takes a sequence's
    /// rows at an offset.
    pub fn batchable(&self) -> bool {
        let d = &self.big.d;
        self.draft.is_none()
            && self.stage.first
            && self.stage.last
            && self.v4.is_none()
            && std::iter::once(&d.first).chain(d.layers.iter().flatten()).all(|op| row_wise(op) || by_sequence(op))
    }

    /// Sequence `slot`'s tokens in its cache.
    fn seq_n(&self, slot: usize) -> usize {
        if slot == self.seq {
            self.n
        } else {
            self.seqs[slot].as_ref().map_or(0, |s| s.n)
        }
    }

    fn set_seq_n(&mut self, slot: usize, n: usize) {
        if slot == self.seq {
            self.n = n;
        } else if let Some(s) = self.seqs[slot].as_mut() {
            s.n = n;
        }
    }

    /// Sequence `slot`'s caches and states (none of them copied).
    fn part_of(&self, slot: usize, row0: usize, rows: usize, start: usize) -> Result<Part> {
        let (kc, vc, states) = if slot == self.seq {
            (self.kc.ptr, self.vc.ptr, self.states.iter().map(|b| b.ptr).collect())
        } else {
            let s = self
                .seqs
                .get(slot)
                .and_then(Option::as_ref)
                .ok_or_else(|| crate::ffi::Error(format!("sequence {slot}: this engine holds {}", self.seqs.len())))?;
            (s.kc.ptr, s.vc.ptr, s.states.iter().map(|b| b.ptr).collect())
        };
        Ok(Part { row0, rows, start, kc, vc, states })
    }

    /// Sequence `slot` empty again: no tokens, its states zeroed (its cache is overwritten as it runs).
    fn empty(&mut self, slot: usize) -> Result<()> {
        if slot == self.seq {
            self.n = 0;
            self.pending = None;
            for b in &self.states {
                memset_async(b.ptr, 0, b.bytes, self.dev.stream)?;
            }
        } else if let Some(s) = self.seqs.get_mut(slot).and_then(Option::as_mut) {
            s.n = 0;
            s.pending = None;
            for b in &s.states {
                memset_async(b.ptr, 0, b.bytes, self.dev.stream)?;
            }
        }
        Ok(())
    }

    /// Sequence `slot`'s prompt chunk `tokens` (at its place in the sequence), as `prefill` runs one chunk: the head on its last row where `last`.
    fn chunk_alone(&mut self, slot: usize, tokens: &[u32], last: bool) -> Result<Option<Sample>> {
        self.select(slot)?;
        let n = tokens.len();
        if n == 0 || n > self.chunk || self.n + n + 1 > self.ctx {
            return err(format!("a chunk of {n} tokens after {}, where a chunk holds {} and the cache {}", self.n, self.chunk, self.ctx));
        }
        let pos: Vec<i32> = (0..n).map(|i| (self.n + i) as i32).collect();
        self.dev.sync()?;
        self.pos.write(0, &pos)?;
        let pp = self.pos.ptr;
        self.start.set(self.n);
        // SAFETY: live buffers of the engine, sized for a chunk.
        unsafe {
            self.embed(tokens)?;
            self.forward(&self.big, n, (pp, pp), last as usize)?;
        }
        self.n += n;
        if last {
            self.read_sample().map(Some)
        } else {
            Ok(None)
        }
    }

    /// One pass over the rows of several sequences: each `(slot, tokens, last)` at the sequence's place; a sample after the last row of each that is `last`, in order.
    fn pass(&mut self, jobs: &[(usize, &[u32], bool)]) -> Result<Vec<Option<Sample>>> {
        let n: usize = jobs.iter().map(|j| j.1.len()).sum();
        let asked = jobs.iter().filter(|j| j.2).count();
        if n > self.chunk || asked > self.head_room() {
            return err(format!("a pass of {n} rows and {asked} samples, where a pass holds {} and the head {}", self.chunk, self.head_room()));
        }
        let (mut parts, mut pos, mut tokens, mut head_rows) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for &(slot, t, last) in jobs {
            let start = self.seq_n(slot);
            if t.is_empty() || start + t.len() + 1 > self.ctx {
                return err(format!("{} tokens after {start}, where the cache holds {}", t.len(), self.ctx));
            }
            parts.push(self.part_of(slot, tokens.len(), t.len(), start)?);
            pos.extend((0..t.len()).map(|i| (start + i) as i32));
            tokens.extend_from_slice(t);
            if last {
                head_rows.push(tokens.len() - 1);
            }
        }
        self.dev.sync()?;
        self.pos.write(0, &pos)?;
        let pp = self.pos.ptr;
        (self.parts, self.head_rows) = (parts, head_rows);
        // SAFETY: live buffers of the engine and of each sequence, the pass's rows within a chunk's.
        let r = unsafe { self.embed(&tokens).and_then(|_| self.forward(&self.big, n, (pp, pp), asked)) };
        self.parts.clear();
        self.head_rows.clear();
        r?;
        for &(slot, t, _) in jobs {
            let to = self.seq_n(slot) + t.len();
            self.set_seq_n(slot, to);
        }
        let (heads, _) = self.head_out(asked, false)?;
        let mut heads = heads.into_iter();
        Ok(jobs.iter().map(|j| j.2.then(|| heads.next().map(|(token, logit)| Sample { token, logit, logprob: None })).flatten()).collect())
    }

    /// Ready to be a scheduler's step for up to `users` sequences: the head's room for a row each, and as many more sequences as the GPU's free memory holds (the first is this one's). None where this
    /// engine runs one sequence only (a draft model, a stage of a split model, a family `add_seqs` refuses).
    pub fn serve_rows(&mut self, users: usize) -> Result<Option<usize>> {
        // (Gemma 4, and Qwen3.5's layout, which Qwen3.8 dense runs on)
        if self.draft.is_some() || !self.stage.first || !self.stage.last || !matches!(self.big.d.family, "Gemma 4" | "Qwen3.5") {
            return Ok(None);
        }
        if self.seqs.len() > 1 {
            // (made ready already)
            return Ok(Some(self.seqs.len()));
        }
        let users = users.clamp(1, self.chunk.min(64));
        if users > 1 {
            self.set_head_rows(users)?;
        }
        let per = (self.kc.bytes + self.vc.bytes + self.states.iter().map(|b| b.bytes).sum::<usize>()).max(1);
        let (free, _) = self.dev.mem_info()?;
        // (room kept for the products' workspaces and the driver)
        const KEEP: usize = 512 << 20;
        let more = (free.saturating_sub(KEEP) / per).min(users - 1);
        self.add_seqs(more)?;
        Ok(Some(more + 1))
    }

    fn batch_prefill() -> bool {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| std::env::var_os("GLYD_ENGINE_BATCH_PREFILL").is_none_or(|v| v != "0"))
    }
}

impl ModelStep for Dyn {
    fn window(&self) -> usize {
        self.ctx
    }
    fn chunk(&self) -> usize {
        self.chunk
    }
    fn pages(&self) -> usize {
        0
    }
    fn slots(&self) -> Option<usize> {
        Some(self.seqs.len())
    }
    fn eos(&self) -> Vec<i64> {
        self.big.d.eos.clone()
    }
    fn prefill(&mut self, pages: &[i32], start: usize, tokens: &[u32], last: bool) -> Result<Option<Sample>> {
        let slot = pages[0] as usize;
        if self.seq_n(slot) != start {
            return err(format!("a chunk at {start} of sequence {slot}, which holds {}", self.seq_n(slot)));
        }
        self.chunk_alone(slot, tokens, last)
    }
    fn prefill_rows(&self) -> usize {
        if self.batchable() && Self::batch_prefill() {
            self.seqs.len().min(self.head_room())
        } else {
            1
        }
    }
    fn prefill_together(&self) -> usize {
        FUSED_ROWS
    }
    fn prefill_many(&mut self, jobs: &[PrefillJob]) -> Result<Vec<Option<Sample>>> {
        if jobs.len() == 1 {
            let j = &jobs[0];
            return Ok(vec![ModelStep::prefill(self, j.pages, j.start, j.tokens, j.last)?]);
        }
        for j in jobs {
            if self.seq_n(j.pages[0] as usize) != j.start {
                return err(format!("a chunk at {} of sequence {}, which holds {}", j.start, j.pages[0], self.seq_n(j.pages[0] as usize)));
            }
        }
        let jobs: Vec<(usize, &[u32], bool)> = jobs.iter().map(|j| (j.pages[0] as usize, j.tokens, j.last)).collect();
        self.pass(&jobs)
    }
    fn decode(&mut self, rows: &[Row]) -> Result<Vec<Sample>> {
        for r in rows {
            let slot = r.pages[0] as usize;
            if self.seq_n(slot) != r.pos {
                return err(format!("a token at {} of sequence {slot}, which holds {}", r.pos, self.seq_n(slot)));
            }
        }
        if rows.len() > self.head_room() {
            return err(format!("a step of {} rows, where the head holds {}", rows.len(), self.head_room()));
        }
        if rows.len() == 1 || !self.batchable() {
            // (one sequence's step as a reply alone takes it, its graph where it has one; several one after another, the last first, each one's logits row moved to its place in the head's rows
            // before the next step writes its own over the first)
            let mut out = vec![None; rows.len()];
            let w = self.big.d.vocab * 2;
            for (j, r) in rows.iter().enumerate().rev() {
                self.select(r.pages[0] as usize)?;
                out[j] = Some(self.step(r.token)?);
                if j > 0 {
                    // SAFETY: the head's logits hold head_room rows; the step's are row 0.
                    unsafe { copy_d2d_async(self.big.logits.ptr + (j * w) as u64, self.big.logits.ptr, w, self.dev.stream)? };
                }
            }
            return Ok(out.into_iter().map(|s| s.expect("every row stepped")).collect());
        }
        let jobs: Vec<(usize, [u32; 1])> = rows.iter().map(|r| (r.pages[0] as usize, [r.token])).collect();
        let jobs: Vec<(usize, &[u32], bool)> = jobs.iter().map(|(s, t)| (*s, &t[..], true)).collect();
        Ok(self.pass(&jobs)?.into_iter().map(|s| s.expect("every row samples")).collect())
    }
    fn logits(&mut self, row: usize) -> Result<Vec<f32>> {
        self.dev.sync()?;
        let v = self.big.d.vocab;
        let mut words = vec![0u16; v];
        self.big.logits.read(row * v * 2, &mut words)?;
        Ok(words.into_iter().map(|w| f32::from_bits((w as u32) << 16)).collect())
    }
    fn release(&mut self, pages: &[i32]) -> Result<()> {
        for &p in pages {
            self.empty(p as usize)?;
        }
        Ok(())
    }
}
